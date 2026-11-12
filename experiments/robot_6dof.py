"""Experiment 3 (paper Sec. IV-C, Fig. 5-6): UR5e moving back and forth among moving obstacles.

The robot is driven kinematically along the path; obstacles are mocap spheres
moving at the conservative human speed of ISO/TS 15066 (1.6 m/s).
"""

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import toppra as ta
import toppra.algorithm as algo

from tospt.constraints import joint_limit_constraints
from tospt.controller import CycleDecider, PathFollower, SafePathController, time_to_arrive
from tospt.mujoco_scene import CapsuleRobot, load_scene, point_to_capsules_distance
from tospt.paths import joint_space_path
from tospt.stoppable_sets import compute_stoppable_sets
from tospt.time_to_reach import time_to_reach_serial, time_to_reach_torch

VLIM = np.full(6, np.pi)
ALIM = np.full(6, 20.0)
Q_A = np.array([-0.6, -1.2, 1.6, -1.9, -1.57, 0.0])
Q_B = np.array([0.9, -1.2, 1.6, -1.9, -1.57, 0.0])
LIFT = np.array([0.0, -0.4, 0.0, 0.0, 0.0, 0.0])
OBSTACLE_SPEED = 1.6
OBSTACLE_RADIUS = 0.05
BOX_LO = np.array([-0.9, -0.8, 0.1])
BOX_HI = np.array([-0.1, 0.6, 0.9])
DT = 0.004


def waypoints(q_start, q_goal):
    return np.array([q_start, 0.5 * (q_start + q_goal) + LIFT, q_goal])


@dataclass
class Plan:
    path: object
    controller: SafePathController
    stage_segments: np.ndarray
    time_stoppable_sets: float
    time_dp: float
    dp_estimate_from_rest: float
    toppra_duration: float


def precompute(robot, wps, min_stages, velocity_steps, backend):
    path = joint_space_path(wps)
    gridpoints = ta.interpolator.propose_gridpoints(path, min_nb_points=min_stages)
    sc = joint_limit_constraints(path, gridpoints, VLIM, ALIM)
    start = time.perf_counter()
    K = compute_stoppable_sets(sc)
    time_sets = time.perf_counter() - start
    start = time.perf_counter()
    if backend == "serial":
        ttr = time_to_reach_serial(sc, K, velocity_steps)
    else:
        ttr = time_to_reach_torch(sc, K, velocity_steps, device=backend)
    time_dp = time.perf_counter() - start
    instance = algo.TOPPRA(
        [ta.constraint.JointVelocityConstraint(np.stack([-VLIM, VLIM], 1)), ta.constraint.JointAccelerationConstraint(np.stack([-ALIM, ALIM], 1))],
        path,
        gridpoints=gridpoints,
    )
    trajectory = instance.compute_trajectory(0, 0)
    return Plan(
        path,
        SafePathController(sc, K, ttr),
        robot.segments(path(gridpoints)),
        time_sets,
        time_dp,
        float(ttr.tau[-1, 0, 0]),
        float(trajectory.duration) if trajectory is not None else float("nan"),
    )


class Obstacles:
    """Constant-speed spheres that bounce off the box walls and turn randomly (zig-zag)."""

    def __init__(self, rng, count, turn_interval=0.5):
        self.rng = rng
        self.turn_interval = turn_interval
        self.pos = rng.uniform(BOX_LO, BOX_HI, size=(count, 3))
        self.vel = np.array([self.random_velocity() for _ in range(count)]).reshape(count, 3)

    def random_velocity(self):
        direction = self.rng.normal(size=3)
        return OBSTACLE_SPEED * direction / np.linalg.norm(direction)

    def step(self, dt):
        for q in range(len(self.pos)):
            if self.rng.random() < dt / self.turn_interval:
                self.vel[q] = self.random_velocity()
        self.pos += self.vel * dt
        below, above = self.pos < BOX_LO, self.pos > BOX_HI
        self.vel[below | above] *= -1.0
        self.pos = np.clip(self.pos, BOX_LO, BOX_HI)


def run(args):
    rng = np.random.default_rng(args.seed)
    model, data = load_scene(args.obstacles, OBSTACLE_RADIUS)
    robot = CapsuleRobot(model, data)
    precompute(robot, waypoints(Q_A, Q_B), 100, 10, args.backend)
    plans = [precompute(robot, waypoints(a, b), args.stages, args.velocity_steps, args.backend) for a, b in ((Q_A, Q_B), (Q_B, Q_A))]
    for name, plan in zip(("A->B", "B->A"), plans):
        n = plan.controller.n
        print(
            f"plan {name}: stages={n + 1} stoppable sets={plan.time_stoppable_sets:.3f}s DP={plan.time_dp:.3f}s "
            f"time-to-reach from rest={plan.dp_estimate_from_rest:.3f}s (exact TOPP-RA {plan.toppra_duration:.3f}s)"
        )

    obstacles = Obstacles(rng, args.obstacles)
    leg = 0
    follower = PathFollower(plans[leg].controller.sc.gridpoints)
    decider = CycleDecider(plans[leg].controller)
    viewer = None
    if args.viewer:
        import mujoco.viewer

        viewer = mujoco.viewer.launch_passive(model, data)
    rows = []
    cycle_times = []
    legs_done = 0
    for step in range(int(args.duration / DT)):
        wall_start = time.perf_counter()
        plan = plans[leg]
        start = time.perf_counter()
        stage_distances = point_to_capsules_distance(plan.stage_segments, robot.radii, obstacles.pos, OBSTACLE_RADIUS)
        psi = time_to_arrive(stage_distances, OBSTACLE_SPEED, args.protective_distance)
        decider.begin_cycle(psi)
        decisions = follower.step(DT, decider)
        cycle_times.append(time.perf_counter() - start)

        s, sd, sdd = follower.position_velocity_acceleration()
        dq = plan.path(s, 1)
        q = plan.path(s)
        qd = dq * sd
        qdd = dq * sdd + plan.path(s, 2) * sd * sd
        distance = point_to_capsules_distance(robot.segments(q), robot.radii, obstacles.pos, OBSTACLE_RADIUS).min(initial=np.inf)
        emergency = sum(d.emergency for d in decisions)
        kept = sum(d.kept_previous for d in decisions)
        rows.append(np.concatenate([[(step + 1) * DT, distance, emergency, kept], q, qd, qdd]))
        obstacles.step(DT)

        if follower.at_end:
            legs_done += 1
            leg = 1 - leg
            follower = PathFollower(plans[leg].controller.sc.gridpoints)
            decider = CycleDecider(plans[leg].controller)
        if viewer is not None:
            if not viewer.is_running():
                break
            data.qpos[:6] = q
            data.mocap_pos[:] = obstacles.pos
            viewer.sync()
            time.sleep(max(0.0, DT - (time.perf_counter() - wall_start)))
    return np.array(rows), np.array(cycle_times), legs_done


def report(log, cycle_times, legs_done, protective_distance):
    t, distance, emergency, kept = log[:, 0], log[:, 1], log[:, 2], log[:, 3]
    qd, qdd = log[:, 10:16], log[:, 16:22]
    speed = np.abs(qd).max(axis=1)
    contact = distance <= protective_distance
    print(f"legs completed: {legs_done}")
    print(f"cycles in contact (distance <= d_protective): {contact.sum()} / {len(t)}")
    print(f"max joint speed while in contact: {speed[contact].max() if contact.any() else 0.0:.2e} rad/s")
    print(f"emergency decisions: {int(emergency.sum())}")
    print(f"kept the previously checked stop: {int(kept.sum())}")
    print(f"max |qd| / vlim = {np.max(np.abs(qd) / VLIM):.3f}, max |qdd| / alim = {np.max(np.abs(qdd) / ALIM):.3f}")
    print(f"execution phase per cycle: mean {cycle_times.mean() * 1e3:.2f} ms, p99 {np.percentile(cycle_times, 99) * 1e3:.2f} ms, max {cycle_times.max() * 1e3:.2f} ms")


def plot(log, path, protective_distance):
    t = log[:, 0]
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    axes[0].plot(t, log[:, 1])
    axes[0].axhline(protective_distance, color="r", lw=0.8)
    axes[0].set_ylabel("min distance [m]")
    for ax, block, label in zip(axes[1:], (4, 10, 16), ("q [rad]", "qd [rad/s]", "qdd [rad/s²]")):
        ax.plot(t, log[:, block : block + 6])
        ax.set_ylabel(label)
    axes[-1].set_xlabel("time [s]")
    for ax in axes:
        ax.grid(True)
    fig.tight_layout()
    fig.savefig(path, dpi=110)


def precompute_trials(args):
    """Paper: mean total pre-computation over 20 trials, serial vs. parallel."""
    rng = np.random.default_rng(args.seed)
    model, data = load_scene(0, OBSTACLE_RADIUS)
    robot = CapsuleRobot(model, data)
    precompute(robot, waypoints(Q_A, Q_B), 100, 10, "cuda")
    totals = {"serial": [], "cuda": []}
    for _ in range(args.precompute_trials):
        wps = waypoints(Q_A + rng.uniform(-0.3, 0.3, 6), Q_B + rng.uniform(-0.3, 0.3, 6))
        for backend in totals:
            plan = precompute(robot, wps, args.stages, args.velocity_steps, backend)
            totals[backend].append((plan.time_stoppable_sets, plan.time_dp))
    for backend, values in totals.items():
        values = np.array(values)
        total = values.sum(axis=1)
        print(f"{backend:6s}: total {total.mean():.3f} ± {total.std():.3f} s (stoppable sets {values[:, 0].mean():.3f} s, DP {values[:, 1].mean():.3f} s)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--obstacles", type=int, default=1)
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--stages", type=int, default=500)
    parser.add_argument("--velocity-steps", type=int, default=30)
    parser.add_argument("--protective-distance", type=float, default=0.0)
    parser.add_argument("--backend", default="cuda", choices=["cuda", "cpu", "serial"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--precompute-trials", type=int, default=0)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    if args.precompute_trials:
        precompute_trials(args)
        return
    log, cycle_times, legs_done = run(args)
    report(log, cycle_times, legs_done, args.protective_distance)
    out = args.out or Path(f"results/robot_6dof_{args.obstacles}obs.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    plot(log, out, args.protective_distance)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
