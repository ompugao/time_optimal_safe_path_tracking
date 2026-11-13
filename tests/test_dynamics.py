import mujoco
import numpy as np
import pytest
import toppra as ta

from tospt.constraints import build_stage_constraints
from tospt.dynamics import (
    ComputedTorque,
    InverseDynamics,
    deadbeat_acceleration,
    torque_limit_constraint,
    use_torque_actuators,
)
from tospt.mujoco_scene import UR5E_SCENE, load_scene
from tospt.paths import joint_space_path

pytestmark = pytest.mark.skipif(not UR5E_SCENE.exists(), reason="run scripts/fetch_assets.sh first")
TAU_MAX = np.array([150.0, 150.0, 150.0, 28.0, 28.0, 28.0])


@pytest.fixture(scope="module")
def model():
    model, _ = load_scene(0, 0.05, payload_mass=3.0)
    use_torque_actuators(model)
    model.opt.timestep = 0.001
    return model


def test_torque_rows_equal_inverse_dynamics_along_the_path(model):
    path = joint_space_path(np.array([[-0.6, -1.2, 1.6, -1.9, -1.57, 0.0], [0.15, -1.6, 1.6, -1.9, -1.57, 0.0], [0.9, -1.2, 1.6, -1.9, -1.57, 0.0]]))
    gridpoints = np.linspace(0, path.duration, 21)
    sc = build_stage_constraints(path, gridpoints, [torque_limit_constraint(model, TAU_MAX)])
    inverse = InverseDynamics(model)
    rng = np.random.default_rng(0)
    for i in range(len(gridpoints)):
        u, x = rng.uniform(-5, 5), rng.uniform(0, 4)
        s = gridpoints[i]
        tau = inverse(path(s), path(s, 1) * np.sqrt(x), path(s, 1) * u + path(s, 2) * x)
        np.testing.assert_allclose(sc.a[i] * u + sc.b[i] * x - sc.c[i], np.concatenate([tau - TAU_MAX, -tau - TAU_MAX]), atol=1e-8)


def test_torque_actuators_apply_ctrl_as_joint_torque(model):
    data = mujoco.MjData(model)
    data.qpos[:6] = [0.3, -1.0, 1.2, -1.5, -1.57, 0.2]
    data.ctrl[:6] = [10.0, -20.0, 30.0, -5.0, 4.0, -3.0]
    mujoco.mj_forward(model, data)
    np.testing.assert_allclose(data.qfrc_actuator[:6], data.ctrl[:6])


@pytest.mark.parametrize("seed", range(3))
def test_deadbeat_torque_puts_the_robot_on_the_target_after_one_step(model, seed):
    rng = np.random.default_rng(seed)
    data = mujoco.MjData(model)
    data.qpos[:6] = np.array([-0.6, -1.2, 1.6, -1.9, -1.57, 0.0]) + rng.uniform(-0.5, 0.5, 6)
    data.qvel[:6] = rng.uniform(-1.0, 1.0, 6)
    mujoco.mj_forward(model, data)
    assert data.ncon == 0
    q, qd = data.qpos[:6].copy(), data.qvel[:6].copy()
    target = q + model.opt.timestep * (qd + rng.uniform(-0.01, 0.01, 6))
    data.ctrl[:6], wanted = ComputedTorque(model)(q, qd, deadbeat_acceleration(q, qd, target, model.opt.timestep))
    np.testing.assert_array_equal(data.ctrl[:6], wanted)
    mujoco.mj_step(model, data)
    np.testing.assert_allclose(data.qpos[:6], target, atol=1e-12)


def test_reference_at_rest_keeps_the_robot_at_rest(model):
    data = mujoco.MjData(model)
    data.qpos[:6] = [0.3, -1.0, 1.2, -1.5, -1.57, 0.2]
    mujoco.mj_forward(model, data)
    tracker = ComputedTorque(model)
    target = data.qpos[:6].copy()
    for _ in range(100):
        q, qd = data.qpos[:6].copy(), data.qvel[:6].copy()
        data.ctrl[:6], _ = tracker(q, qd, deadbeat_acceleration(q, qd, target, model.opt.timestep))
        mujoco.mj_step(model, data)
    assert np.abs(data.qvel[:6]).max() < 1e-12
