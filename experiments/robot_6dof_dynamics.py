"""UR5e demo with robot dynamics: torque limits in the path constraints and a physically simulated robot.

The safe path controller plans with joint velocity and torque limits (inverse dynamics from MuJoCo).
The robot is not moved kinematically: MuJoCo integrates its dynamics, and a computed-torque
controller (inverse dynamics with a discrete-time deadbeat law) tracks the reference produced by the path controller.
"""

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import mujoco.viewer
import numpy as np
import toppra as ta
import toppra.algorithm as algo

from robot_6dof import DT, OBSTACLE_RADIUS, OBSTACLE_SPEED, Q_A, Q_B, VLIM, Obstacles, waypoints
from tospt.controller import CycleDecider, PathFollower, SafePathController, time_to_arrive
from tospt.dynamics import ComputedTorque, actuator_torque_limits, deadbeat_acceleration, dynamic_limit_constraints, use_torque_actuators
from tospt.mujoco_scene import CapsuleRobot, load_scene, point_to_capsules_distance
from tospt.paths import joint_space_path
from tospt.stoppable_sets import compute_stoppable_sets
from tospt.time_to_reach import time_to_reach_torch

PHYSICS_DT = 0.001


@dataclass
class Plan:
    path: object
    controller: SafePathController
    stage_segments: np.ndarray
    toppra_duration: float
    toppra_duration_without_torque: float


def toppra_duration(path, gridpoints, constraints):
    trajectory = algo.TOPPRA(constraints, path, gridpoints=gridpoints).compute_trajectory(0, 0)
    return float(trajectory.duration) if trajectory is not None else float("nan")


def precompute(model, robot, wps, args, device):
    path = joint_space_path(wps)
    gridpoints = ta.interpolator.propose_gridpoints(path, min_nb_points=args.stages)
    tau_plan = args.torque_fraction * actuator_torque_limits(model)
    alim = np.full(6, args.alim)
    sc, constraints = dynamic_limit_constraints(model, path, gridpoints, VLIM, alim, tau_plan)
    K = compute_stoppable_sets(sc)
    ttr = time_to_reach_torch(sc, K, args.velocity_steps, device=device)
    return Plan(
        path,
        SafePathController(sc, K, ttr),
        robot.segments(path(gridpoints)),
        toppra_duration(path, gridpoints, constraints),
        toppra_duration(path, gridpoints, constraints[:2]),
    )


def reference(plan, follower):
    s, sd, sdd = follower.position_velocity_acceleration()
    dq = plan.path(s, 1)
    return plan.path(s), dq * sd, dq * sdd + plan.path(s, 2) * sd * sd


def run(args):
    rng = np.random.default_rng(args.seed)
    model, data = load_scene(args.obstacles, OBSTACLE_RADIUS, payload_mass=args.payload if args.actual_payload is None else args.actual_payload)
    nominal_model, _ = load_scene(args.obstacles, OBSTACLE_RADIUS, payload_mass=args.payload)
    for m in (model, nominal_model):
        use_torque_actuators(m)
        m.opt.timestep = PHYSICS_DT
    substeps = round(DT / PHYSICS_DT)
    robot = CapsuleRobot(model, mujoco.MjData(model))
    plans = [precompute(nominal_model, robot, waypoints(a, b), args, args.device) for a, b in ((Q_A, Q_B), (Q_B, Q_A))]
    for name, plan in zip(("A->B", "B->A"), plans):
        print(
            f"plan {name}: exact TOPP-RA {plan.toppra_duration:.3f}s with torque limits "
            f"({plan.toppra_duration_without_torque:.3f}s with velocity/acceleration limits only)"
        )

    tracker = ComputedTorque(nominal_model)
    data.qpos[:6] = Q_A
    mujoco.mj_forward(model, data)
    obstacles = Obstacles(rng, args.obstacles)
    leg = 0
    follower = PathFollower(plans[leg].controller.sc.gridpoints)
    decider = CycleDecider(plans[leg].controller)
    viewer = None
    if args.viewer:
        viewer = mujoco.viewer.launch_passive(model, data)
    recorder = None
    if args.record:
        from tospt.video import VideoRecorder

        recorder = VideoRecorder(model, args.record, fps=args.fps)
        frame_interval = round(1.0 / (args.fps * DT))
    rows = []
    legs_done = 0
    for step in range(int(args.duration / DT)):
        wall_start = time.perf_counter()
        plan = plans[leg]
        stage_distances = point_to_capsules_distance(plan.stage_segments, robot.radii, obstacles.pos, OBSTACLE_RADIUS)
        psi = time_to_arrive(stage_distances, OBSTACLE_SPEED, args.protective_distance + args.tracking_margin, latency=args.settle_time)
        decider.begin_cycle(psi)
        emergency = kept = saturated = 0
        for k in range(substeps):
            offset = k * PHYSICS_DT
            decisions = follower.step(PHYSICS_DT, lambda i, x, elapsed: decider(i, x, offset + elapsed))
            emergency += sum(d.emergency for d in decisions)
            kept += sum(d.kept_previous for d in decisions)
            q, qd = data.qpos[:6].copy(), data.qvel[:6].copy()
            qdd = deadbeat_acceleration(q, qd, reference(plan, follower)[0], PHYSICS_DT)
            data.ctrl[:6], tau_wanted = tracker(q, qd, qdd)
            saturated += np.any(np.abs(tau_wanted) > tracker.limit)
            data.mocap_pos[:] = obstacles.pos
            mujoco.mj_step(model, data)
            obstacles.step(PHYSICS_DT)

        q, qd = data.qpos[:6].copy(), data.qvel[:6].copy()
        obstacle_distances = point_to_capsules_distance(robot.segments(q), robot.radii, obstacles.pos, OBSTACLE_RADIUS)[0]
        distance = obstacle_distances.min(initial=np.inf)
        if recorder is not None and step % frame_interval == 0:
            recorder.capture(q, obstacles.pos, obstacle_distances, args.protective_distance)
        q_ref, qd_ref, _ = reference(plan, follower)
        reference_distance = point_to_capsules_distance(robot.segments(q_ref), robot.radii, obstacles.pos, OBSTACLE_RADIUS).min(initial=np.inf)
        rows.append(
            np.concatenate([[(step + 1) * DT, distance, reference_distance, emergency, kept, saturated, data.ncon], q, qd, q_ref, qd_ref, data.ctrl[:6]])
        )

        if follower.at_end:
            legs_done += 1
            leg = 1 - leg
            follower = PathFollower(plans[leg].controller.sc.gridpoints)
            decider = CycleDecider(plans[leg].controller)
        if viewer is not None:
            if not viewer.is_running():
                break
            viewer.sync()
            time.sleep(max(0.0, DT - (time.perf_counter() - wall_start)))
    if recorder is not None:
        recorder.close()
    return np.array(rows), legs_done, actuator_torque_limits(model)


COLUMNS = {"t": 0, "distance": 1, "reference_distance": 2, "emergency": 3, "kept": 4, "saturated": 5, "contacts": 6}
Q, QD, Q_REF, QD_REF, TAU = (slice(7 + 6 * n, 13 + 6 * n) for n in range(5))


def report(log, legs_done, tau_limit, protective_distance):
    distance = log[:, COLUMNS["distance"]]
    speed = np.abs(log[:, QD]).max(axis=1)
    contact = distance <= protective_distance
    tracking = np.abs(log[:, Q] - log[:, Q_REF]).max(axis=1)
    print(f"legs completed: {legs_done}")
    print(f"cycles in contact (distance <= d_protective): {contact.sum()} / {len(log)}")
    print(f"max joint speed while in contact: {speed[contact].max() if contact.any() else 0.0:.2e} rad/s")
    print(f"emergency decisions: {int(log[:, COLUMNS['emergency']].sum())}, kept previous stop: {int(log[:, COLUMNS['kept']].sum())}")
    print(f"max joint tracking error: {tracking.max():.2e} rad (mean {tracking.mean():.2e})")
    print(f"max |tau| / limit per joint: {np.round(np.abs(log[:, TAU]).max(axis=0) / tau_limit, 3)}")
    print(f"physics steps with saturated torque command: {int(log[:, COLUMNS['saturated']].sum())}")
    print(f"max |qd| / vlim = {np.max(np.abs(log[:, QD]) / VLIM):.3f}")
    print(f"cycles with MuJoCo contacts: {int((log[:, COLUMNS['contacts']] > 0).sum())}")


def plot(log, tau_limit, path, protective_distance):
    t = log[:, 0]
    fig, axes = plt.subplots(4, 1, figsize=(12, 11), sharex=True)
    axes[0].plot(t, log[:, COLUMNS["distance"]], label="simulated robot")
    axes[0].plot(t, log[:, COLUMNS["reference_distance"]], "--", lw=0.8, label="reference")
    axes[0].axhline(protective_distance, color="r", lw=0.8)
    axes[0].set_ylabel("min distance [m]")
    axes[0].legend(loc="upper right")
    axes[1].plot(t, log[:, QD])
    axes[1].set_ylabel("qd [rad/s]")
    axes[2].plot(t, log[:, TAU] / tau_limit)
    axes[2].set_ylabel("tau / limit")
    axes[3].semilogy(t, np.maximum(np.abs(log[:, Q] - log[:, Q_REF]).max(axis=1), 1e-12))
    axes[3].set_ylabel("tracking error [rad]")
    axes[-1].set_xlabel("time [s]")
    for ax in axes:
        ax.grid(True)
    fig.tight_layout()
    fig.savefig(path, dpi=110)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--obstacles", type=int, default=1)
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--stages", type=int, default=500)
    parser.add_argument("--velocity-steps", type=int, default=30)
    parser.add_argument("--alim", type=float, default=100.0, help="loose joint acceleration limit; torque is the active limit")
    parser.add_argument("--torque-fraction", type=float, default=0.8, help="share of the actuator limit used for planning; the rest is left to feedback")
    parser.add_argument("--payload", type=float, default=3.0, help="mass at the flange [kg]")
    parser.add_argument("--actual-payload", type=float, default=None, help="payload of the simulated robot, if it differs from the model used by the controller [kg]")
    parser.add_argument("--protective-distance", type=float, default=0.0)
    parser.add_argument("--tracking-margin", type=float, default=0.0, help="extra distance for the Cartesian tracking error [m]")
    parser.add_argument("--settle-time", type=float, default=0.0, help="time for the robot to settle after the reference stops [s]")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--record", type=Path, default=None, help="write an mp4 rendered offscreen")
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    log, legs_done, tau_limit = run(args)
    report(log, legs_done, tau_limit, args.protective_distance)
    out = args.out or Path(f"results/robot_6dof_dynamics_{args.obstacles}obs.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    plot(log, tau_limit, out, args.protective_distance)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
