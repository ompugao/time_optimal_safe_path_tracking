"""Experiment 1 (paper Sec. IV-A, Fig. 2-3): 1-D car vs. Zanchettin's method."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tospt.constraints import joint_limit_constraints
from tospt.controller import CycleDecider, PathFollower, SafePathController, time_to_arrive
from tospt.paths import straight_line_path, uniform_gridpoints
from tospt.stoppable_sets import compute_stoppable_sets
from tospt.time_to_reach import time_to_reach_torch
from zanchettin import ZanchettinCar

GOAL = 25.0
VMAX = 20.0
AMAX = 100.0
WALL_SPEED = 20.0
WALL_START = 40.0
WALL_WAIT = 1.0
DT = 1e-3


class Wall:
    """Approaches at constant speed, waits after touching the leading car, then moves back."""

    def __init__(self):
        self.position = WALL_START
        self.mode = "approach"
        self.wait_until = None

    def step(self, t, leading_car):
        if self.mode == "approach":
            self.position -= WALL_SPEED * DT
            if self.position <= leading_car:
                self.position = leading_car
                self.mode = "wait"
                self.wait_until = t + WALL_WAIT
        elif self.mode == "wait":
            if t >= self.wait_until:
                self.mode = "retreat"
        else:
            self.position += WALL_SPEED * DT


def run(num_stages, num_velocity_steps, stopping_time, duration, device):
    path = straight_line_path(GOAL)
    sc = joint_limit_constraints(path, uniform_gridpoints(path, num_stages), [VMAX], [AMAX])
    K = compute_stoppable_sets(sc)
    ttr = time_to_reach_torch(sc, K, num_velocity_steps, device=device)
    decider = CycleDecider(SafePathController(sc, K, ttr))
    ours = PathFollower(sc.gridpoints)
    theirs = ZanchettinCar(VMAX, AMAX, GOAL, stopping_time, human_speed=WALL_SPEED)
    wall = Wall()
    rows = []
    for step in range(int(duration / DT)):
        t = step * DT
        decider.begin_cycle(time_to_arrive((wall.position - sc.gridpoints)[:, None], [WALL_SPEED]))
        ours.step(DT, decider)
        theirs.step(wall.position - theirs.position, DT)
        s, v, a = ours.position_velocity_acceleration()
        wall.step(t + DT, max(s, theirs.position))
        rows.append((t + DT, s, v, a, theirs.position, theirs.velocity, theirs.acceleration, wall.position))
    return np.array(rows), theirs.infeasible_steps


def arrival_time(t, position):
    reached = position >= GOAL - 1e-6
    return t[np.argmax(reached)] if reached.any() else np.inf


def plot(log, path):
    t, s, v, a, zs, zv, za, wall = log.T
    fig, axes = plt.subplots(4, 1, figsize=(8, 9), sharex=True)
    axes[0].plot(t, zs, label="Zanchettin")
    axes[0].plot(t, s, label="Ours")
    axes[0].plot(t, np.minimum(wall, 45), "g--", label="Wall")
    axes[0].set_ylabel("position [m]")
    axes[1].plot(t, wall - zs, label="Zanchettin")
    axes[1].plot(t, wall - s, label="Ours")
    axes[1].set_ylabel("distance to wall [m]")
    axes[1].set_ylim(-1, 20)
    axes[2].plot(t, zv)
    axes[2].plot(t, v)
    axes[2].set_ylabel("velocity [m/s]")
    axes[3].plot(t, za)
    axes[3].plot(t, a)
    axes[3].set_ylabel("acceleration [m/s²]")
    axes[3].set_xlabel("time [s]")
    axes[0].legend()
    for ax in axes:
        ax.grid(True)
    fig.tight_layout()
    fig.savefig(path, dpi=120)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stages", type=int, default=250)
    parser.add_argument("--velocity-steps", type=int, default=100)
    parser.add_argument("--stopping-time", type=float, default=0.55)
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", type=Path, default=Path("results/car_1d.png"))
    parser.add_argument("--sweep-stopping-time", action="store_true")
    args = parser.parse_args()

    if args.sweep_stopping_time:
        for ts in (0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.8):
            log, infeasible = run(args.stages, args.velocity_steps, ts, args.duration, args.device)
            print(f"T_s={ts:.2f}  Zanchettin arrival={arrival_time(log[:, 0], log[:, 4]):.3f} s  infeasible steps={infeasible}")
        return

    log, infeasible = run(args.stages, args.velocity_steps, args.stopping_time, args.duration, args.device)
    t, s, v, a, zs, zv, za, wall = log.T
    touching = wall - s <= 1e-9
    print(f"arrival  ours={arrival_time(t, s):.3f} s  Zanchettin={arrival_time(t, zs):.3f} s")
    print(f"Zanchettin infeasible LP steps: {infeasible}")
    print(f"ours: max |velocity| while touching the wall = {np.abs(v[touching]).max() if touching.any() else 0.0:.2e} m/s")
    print(f"ours: min distance while moving = {(wall - s)[v > 1e-9].min():.3f} m")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    plot(log, args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
