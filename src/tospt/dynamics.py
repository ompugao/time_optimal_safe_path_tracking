import mujoco
import numpy as np
import toppra as ta

from tospt.constraints import build_stage_constraints


class InverseDynamics:
    """tau = M(q) qdd + C(q, qd) qd + g(q), from MuJoCo on a private MjData (armature included)."""

    def __init__(self, model, dof=6):
        self.model = model
        self.data = mujoco.MjData(model)
        self.dof = dof

    def __call__(self, q, qd, qdd):
        self.data.qpos[: self.dof] = q
        self.data.qvel[: self.dof] = qd
        self.data.qacc[: self.dof] = qdd
        mujoco.mj_inverse(self.model, self.data)
        return self.data.qfrc_inverse[: self.dof].copy()


def actuator_torque_limits(model, dof=6):
    return model.actuator_forcerange[:dof, 1].copy()


def torque_limit_constraint(model, tau_max, dof=6):
    tau_max = np.asarray(tau_max, dtype=float)
    return ta.constraint.JointTorqueConstraint(InverseDynamics(model, dof), np.stack([-tau_max, tau_max], axis=1), np.zeros(dof))


def dynamic_limit_constraints(model, path, gridpoints, vlim, alim, tau_max):
    vlim = np.asarray(vlim, dtype=float)
    alim = np.asarray(alim, dtype=float)
    constraints = [
        ta.constraint.JointVelocityConstraint(np.stack([-vlim, vlim], axis=1)),
        ta.constraint.JointAccelerationConstraint(np.stack([-alim, alim], axis=1)),
        torque_limit_constraint(model, tau_max),
    ]
    return build_stage_constraints(path, gridpoints, constraints), constraints


def use_torque_actuators(model):
    """Turn Menagerie's position servos into direct torque motors (ctrl = joint torque)."""
    model.actuator_gainprm[:] = 0.0
    model.actuator_gainprm[:, 0] = 1.0
    model.actuator_biasprm[:] = 0.0
    model.actuator_biastype[:] = mujoco.mjtBias.mjBIAS_NONE
    model.actuator_ctrlrange[:] = model.actuator_forcerange
    model.actuator_ctrllimited[:] = 1


class ComputedTorque:
    """Joint torque that produces a desired joint acceleration, clipped to the actuator limits."""

    def __init__(self, model, dof=6):
        self.inverse = InverseDynamics(model, dof)
        self.limit = actuator_torque_limits(model, dof)

    def __call__(self, q, qd, qdd):
        tau = self.inverse(q, qd, qdd)
        return np.clip(tau, -self.limit, self.limit), tau


def deadbeat_acceleration(q, qd, q_target, h):
    """Acceleration that puts q exactly on q_target after one semi-implicit Euler step of length h.

    A PD law with finite gains was rejected: after the reference stops, it keeps the robot
    moving while it removes the remaining error, so the robot is not at rest at contact.
    """
    return ((q_target - q) / h - qd) / h
