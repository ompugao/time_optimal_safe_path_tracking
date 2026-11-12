"""Experiment 2 (paper Sec. IV-B, Fig. 4): Time-to-Reach computation time, serial vs. parallel.

Timed part: everything after the stoppable sets (paper Alg. 1 line 12 to the end).
"""

import argparse
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from tospt.constraints import joint_limit_constraints
from tospt.paths import random_joint_path, uniform_gridpoints
from tospt.stoppable_sets import compute_stoppable_sets
from tospt.time_to_reach import time_to_reach_serial, time_to_reach_torch

BACKENDS = {
    "serial (python, memoized)": lambda sc, K, M: time_to_reach_serial(sc, K, M),
    "torch cpu float64": lambda sc, K, M: time_to_reach_torch(sc, K, M, device="cpu"),
    "torch cuda float64": lambda sc, K, M: time_to_reach_torch(sc, K, M, device="cuda"),
    "torch cuda float32": lambda sc, K, M: time_to_reach_torch(sc, K, M, device="cuda", dtype=torch.float32),
}


def random_problem(rng, num_stages):
    path = random_joint_path(rng, dof=6, n_waypoints=3)
    sc = joint_limit_constraints(path, uniform_gridpoints(path, num_stages), rng.uniform(0.5, 3.0, 6), rng.uniform(1.0, 10.0, 6))
    return sc, compute_stoppable_sets(sc)


def timed(fn):
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter() - start


def sweep(settings, trials, seed):
    """settings: list of (N, M). Returns {backend: array (len(settings), trials)}."""
    rng = np.random.default_rng(seed)
    warm_sc, warm_K = random_problem(rng, 50)
    for fn in BACKENDS.values():
        fn(warm_sc, warm_K, 10)
    times = {name: np.zeros((len(settings), trials)) for name in BACKENDS}
    for s, (N, M) in enumerate(settings):
        for trial in range(trials):
            sc, K = random_problem(rng, N)
            for name, fn in BACKENDS.items():
                times[name][s, trial] = timed(lambda: fn(sc, K, M))
        summary = "  ".join(f"{name}={times[name][s].mean():.3f}s" for name in BACKENDS)
        print(f"N={N:4d} M={M:3d}  {summary}", flush=True)
    return times


def plot(ax, xs, times, xlabel):
    for name, t in times.items():
        ax.errorbar(xs, t.mean(axis=1), yerr=t.std(axis=1), marker="o", capsize=3, label=name)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("computation time [s]")
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.4)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("results/bench_precompute.png"))
    args = parser.parse_args()

    Ms = [10, 20, 30, 40, 50]
    Ns = [100, 200, 300, 400, 500]
    print("Effect of M (N = 300)")
    times_m = sweep([(300, M) for M in Ms], args.trials, args.seed)
    print("Effect of N (M = 50)")
    times_n = sweep([(N, 50) for N in Ns], args.trials, args.seed + 1)

    fig, axes = plt.subplots(2, 1, figsize=(8, 8))
    plot(axes[0], Ms, times_m, "number of velocity discretization M (N = 300)")
    plot(axes[1], Ns, times_n, "number of stages N (M = 50)")
    axes[0].legend()
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=120)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
