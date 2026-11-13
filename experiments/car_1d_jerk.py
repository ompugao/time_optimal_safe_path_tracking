"""Jerk-limited 1-D car: smoothed second-order plan vs. third-order lattice (same wall as Experiment 1)."""

import argparse
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from car_1d import AMAX, DT, GOAL, VMAX, WALL_SPEED, Wall, arrival_time, time_derivatives
from tospt.constraints import joint_limit_constraints
from tospt.controller import CycleDecider, SafePathController, time_to_arrive
from tospt.jerk_filter import SmoothedFollower, jerk_limited_constraints, window_samples
from tospt.paths import straight_line_path, uniform_gridpoints
from tospt.stoppable_sets import compute_stoppable_sets
from tospt.third_order import CarLattice, LatticeFollower, ThirdOrderController, time_to_stop
from tospt.time_to_reach import time_to_reach_torch


class SmoothedCar:
    name = "smoothed plan"

    def __init__(self, num_stages, velocity_steps, jerk_limit):
        start = time.perf_counter()
        path = straight_line_path(GOAL)
        limited = jerk_limited_constraints(joint_limit_constraints(path, uniform_gridpoints(path, num_stages), [VMAX], [AMAX]), path, [jerk_limit])
        self.gridpoints = limited.sc.gridpoints
        K = compute_stoppable_sets(limited.sc)
        ttr = time_to_reach_torch(limited.sc, K, velocity_steps, device="cpu")
        self.precompute_seconds = time.perf_counter() - start
        self.table_bytes = K.nbytes + ttr.tau.nbytes + ttr.rho.nbytes
        self.controller = SafePathController(limited.sc, K, ttr)
        self.num_samples = window_samples(limited.window, DT)
        mid = num_stages // 2
        top = int(np.flatnonzero(np.isfinite(ttr.tau[-1, mid])).max())
        self.stop_time_from_cruise = float(np.min(ttr.tau[mid + 1 :, mid, top])) + (self.num_samples - 1) * DT
        self.reset()

    def reset(self):
        self.decider = CycleDecider(self.controller)
        self.follower = SmoothedFollower(self.gridpoints, self.num_samples)

    @property
    def position(self):
        return self.follower.position

    def step(self, wall_position):
        psi = time_to_arrive((wall_position - self.gridpoints)[:, None], [WALL_SPEED], latency=self.follower.latency(DT))
        self.decider.begin_cycle(psi)
        return self.follower.step(DT, self.decider)


class LatticeCar:
    name = "third-order lattice"

    def __init__(self, num_stages, jerk_limit, refinement):
        start = time.perf_counter()
        self.lattice = CarLattice(GOAL, num_stages, VMAX, AMAX, jerk_limit, refinement)
        tts = time_to_stop(self.lattice)
        self.precompute_seconds = time.perf_counter() - start
        self.table_bytes = tts.tau.nbytes + tts.rho.nbytes
        self.controller = ThirdOrderController(self.lattice, tts)
        mid = num_stages // 2
        cruise = int(np.flatnonzero(np.isfinite(tts.tau[-1, mid, :, self.lattice.zero])).max())
        self.stop_time_from_cruise = float(np.min(tts.tau[mid + 1 :, mid, cruise, self.lattice.zero]))
        self.reset()

    def reset(self):
        self.follower = LatticeFollower(self.lattice)

    @property
    def position(self):
        return self.follower.position

    def step(self, wall_position):
        psi = time_to_arrive((wall_position - self.lattice.gridpoints)[:, None], [WALL_SPEED])
        return self.follower.step(DT, self.controller, psi)


def simulate(car, duration, with_wall):
    car.reset()
    wall = Wall()
    rows = []
    emergencies = 0
    for step in range(int(duration / DT)):
        decisions = car.step(wall.position if with_wall else np.inf)
        emergencies += sum(d.emergency for d in decisions)
        if with_wall:
            wall.step((step + 1) * DT, car.position)
        rows.append(((step + 1) * DT, car.position, wall.position))
    t, s, wall_position = np.array(rows).T
    v, a, jerk = time_derivatives(s)
    return np.column_stack([t, s, v, a, jerk, wall_position]), emergencies


def summarize(car, duration, jerk_limit):
    free, _ = simulate(car, duration, with_wall=False)
    log, emergencies = simulate(car, duration, with_wall=True)
    t, s, v, a, jerk, wall = log.T
    touching = wall - s <= 1e-9
    print(f"{car.name}")
    print(f"  pre-computation {car.precompute_seconds:.2f} s, tables {car.table_bytes / 1e6:.0f} MB")
    print(f"  arrival without wall {arrival_time(free[:, 0], free[:, 1]):.3f} s, with wall {arrival_time(t, s):.3f} s")
    print(f"  fastest stop from cruise {car.stop_time_from_cruise * 1e3:.0f} ms")
    print(f"  position where the wall touches {s[touching].max() if touching.any() else float('nan'):.2f} m")
    print(f"  max speed while touching {np.abs(v[touching]).max() if touching.any() else 0.0:.2e} m/s")
    print(f"  max |acceleration| / limit {np.abs(a).max() / AMAX:.3f}, max |jerk| / limit {max(np.abs(jerk).max(), np.abs(free[:, 4]).max()) / jerk_limit:.3f}")
    print(f"  emergency decisions {emergencies}")
    return log


def plot(logs, names, jerk_limit, path):
    fig, axes = plt.subplots(4, 1, figsize=(8, 9), sharex=True)
    for log, name in zip(logs, names):
        t, s, v, a, jerk, wall = log.T
        axes[0].plot(t, s, label=name)
        axes[1].plot(t, v)
        axes[2].plot(t, a)
        axes[3].plot(t, jerk)
    axes[0].plot(logs[0][:, 0], np.minimum(logs[0][:, 5], 45), "g--", label=f"Wall ({names[0]})")
    for bound in (-jerk_limit, jerk_limit):
        axes[3].axhline(bound, color="r", lw=0.8)
    for ax, label in zip(axes, ("position [m]", "velocity [m/s]", "acceleration [m/s²]", "jerk [m/s³]")):
        ax.set_ylabel(label)
        ax.grid(True)
    axes[0].legend()
    axes[-1].set_xlabel("time [s]")
    fig.tight_layout()
    fig.savefig(path, dpi=120)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jerk-limit", type=float, default=5000.0)
    parser.add_argument("--stages", type=int, default=250)
    parser.add_argument("--velocity-steps", type=int, default=100)
    parser.add_argument("--refinement", type=int, default=1, help="multiplier of the number of acceleration levels")
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--out", type=Path, default=Path("results/car_1d_jerk_comparison.png"))
    args = parser.parse_args()

    ideal = VMAX / AMAX + AMAX / args.jerk_limit
    print(f"jerk limit {args.jerk_limit:g} m/s³, {args.stages} stages; jerk-optimal stop from cruise {ideal * 1e3:.0f} ms")
    cars = [SmoothedCar(args.stages, args.velocity_steps, args.jerk_limit), LatticeCar(args.stages, args.jerk_limit, args.refinement)]
    logs = [summarize(car, args.duration, args.jerk_limit) for car in cars]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    plot(logs, [car.name for car in cars], args.jerk_limit, args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
