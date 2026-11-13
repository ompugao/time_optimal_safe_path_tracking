"""Figure that explains the third-order lattice model (src/tospt/third_order.py) on the 1-D car."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from car_1d import AMAX, GOAL, VMAX
from tospt.third_order import CarLattice, time_to_stop


def fastest_stop_route(lattice, tts, stage, n, level):
    """Samples (distance, time, speed, acceleration) along the stored route of the fastest stop."""
    stop = stage + 1 + int(np.argmin(tts.tau[stage + 1 :, stage, n, level]))
    samples, nodes = [], []
    distance = time = 0.0
    while stage < stop:
        next_level = int(tts.rho[stop, stage, n, level])
        segment = lattice.segment(stage, n, level, next_level, stop)
        nodes.append((distance, time, segment.kind))
        for t in np.linspace(0.0, segment.duration, 30):
            sigma = segment.distance(t)
            if segment.kind == "cubic":
                acceleration = segment.u0 + segment.rate * t
            else:
                acceleration = segment.u0 + segment.rate * sigma
            samples.append((distance + sigma, time + t, segment.speed(t), acceleration))
        distance += lattice.delta
        time += segment.duration
        n, level = lattice.next_state(stage, n, level, next_level, stop)
        stage += 1
    nodes.append((distance, time, "rest"))
    return np.array(samples), nodes


def draw_model(ax, lattice):
    s = np.arange(6)
    levels = np.array([0, 1, 3, 3, 1, -1]) * lattice.du
    ax.plot(s, levels, "o-", color="C0", label="here: u linear on each segment, m·δu at grid points")
    ax.step(s, np.array([0, 3, 3, 3, -1, -1]) * lattice.du, where="post", color="0.6", ls="--", label="paper: u constant per segment (jumps)")
    for level in np.arange(-2, 5) * lattice.du:
        ax.axhline(level, color="0.9", lw=0.6, zorder=0)
    ax.set_xticks(s)
    ax.set_xticklabels([f"s{i}" for i in s])
    ax.set_ylabel("u = s̈ [m/s²]")
    ax.set_title("1. Model: continuous, piecewise-linear acceleration")
    ax.legend(fontsize=7, loc="lower left")


def draw_integer_step(ax):
    for n in range(9):
        ax.plot(np.arange(-3, 4), np.full(7, n), ".", color="0.75")
    n, m = 2, 1
    for m_next, color in ((-1, "C3"), (0, "C1"), (1, "C2"), (2, "C0")):
        ax.annotate("", xy=(m_next, n + m + m_next), xytext=(m, n), arrowprops=dict(arrowstyle="->", color=color, lw=1.8))
        ax.text(m_next + 0.08, n + m + m_next + 0.15, f"m'={m_next}", color=color, fontsize=8)
    ax.plot(m, n, "ko")
    ax.text(m + 0.1, n - 0.45, "(n, m) at stage i", fontsize=8)
    ax.set_xlabel("m   (u = m·δu)")
    ax.set_ylabel("n   (x = ṡ² = n·Δ·δu)")
    ax.set_title("2. Exact integer step:  n' = n + m + m'\n(x_{i+1} = x_i + Δ(u_i + u_{i+1}), no rounding)")


def draw_last_segment(ax):
    t = np.linspace(0.0, 1.0, 100)
    ax.plot(1 - (1 - t) ** 3, (1 - t) ** 2, color="C0", label="velocity / v0")
    ax.plot(1 - (1 - t) ** 3, -(1 - t), color="C3", label="acceleration / |u0|")
    ax.axhline(0.0, color="k", lw=0.5)
    ax.set_xlabel("position on the last segment  (s − s_{j−1}) / Δ")
    ax.legend(fontsize=8)
    ax.set_title("3. Last segment: constant jerk in time (cubic)\nends with ṡ = 0 and s̈ = 0; needs u = −2x/(3Δ), takes 3Δ/v")


def draw_stoppable_states(ax, lattice, tts, stop, distances):
    levels = np.arange(lattice.num_levels)
    for distance, color, size in zip(distances, ("C0", "C1", "k"), (14, 6, 45)):
        n, level = np.nonzero(np.isfinite(tts.tau[stop, stop - distance]))
        label = f"stage j−{distance}" + (" (entry of the last segment)" if distance == 1 else "")
        ax.scatter(lattice.u(levels[level]), np.sqrt(lattice.x(n)), s=size, color=color, label=label, zorder=3 if distance == 1 else 2)
    ax.set_xlabel("u [m/s²]")
    ax.set_ylabel("ṡ [m/s]")
    ax.set_ylim(0.0, VMAX * 1.05)
    ax.legend(fontsize=8, loc="upper right")
    ax.set_title("4. Lattice states that can stop exactly at stage j\n(finite tau[j, i, n, m]); not intervals")


def draw_route(ax, samples, nodes, jerk_limit):
    distance, time, speed, acceleration = samples.T
    ax.plot(time * 1e3, speed, color="C0", label="ṡ [m/s]")
    ax.set_ylabel("ṡ [m/s]")
    twin = ax.twinx()
    twin.plot(time * 1e3, acceleration, color="C3", label="s̈ [m/s²]")
    twin.set_ylabel("s̈ [m/s²]")
    for _, node_time, _ in nodes:
        ax.axvline(node_time * 1e3, color="0.85", lw=0.5, zorder=0)
    ax.axvspan(nodes[-2][1] * 1e3, nodes[-1][1] * 1e3, color="0.9", zorder=0)
    ideal = VMAX / AMAX + AMAX / jerk_limit
    ax.axvline(ideal * 1e3, color="C2", ls="--", label=f"jerk-optimal stop {ideal * 1e3:.0f} ms")
    ax.set_xlabel("time [ms]  (thin lines: stage boundaries, grey: last segment)")
    ax.set_title(f"5. Stored route of the fastest stop from cruise: {nodes[-1][1] * 1e3:.0f} ms\n(the last segment takes {(nodes[-1][1] - nodes[-2][1]) * 1e3:.0f} ms)")
    ax.legend(loc="upper right", fontsize=8)
    twin.legend(loc="center right", fontsize=8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jerk-limit", type=float, default=5000.0)
    parser.add_argument("--stages", type=int, default=250)
    parser.add_argument("--refinement", type=int, default=1)
    parser.add_argument("--out", type=Path, default=Path("docs/third_order_lattice.png"))
    args = parser.parse_args()

    lattice = CarLattice(GOAL, args.stages, VMAX, AMAX, args.jerk_limit, args.refinement)
    tts = time_to_stop(lattice)
    stop = args.stages // 2 + 40
    start = args.stages // 2
    cruise = int(np.flatnonzero(np.isfinite(tts.tau[-1, start, :, lattice.zero])).max())
    samples, nodes = fastest_stop_route(lattice, tts, start, cruise, lattice.zero)

    fig, axes = plt.subplots(2, 3, figsize=(17, 9.5))
    draw_model(axes[0, 0], lattice)
    draw_integer_step(axes[0, 1])
    draw_last_segment(axes[0, 2])
    draw_stoppable_states(axes[1, 0], lattice, tts, stop, (6, 3, 1))
    draw_stoppable_states(axes[1, 1], lattice, tts, stop, (30, 15, 1))
    draw_route(axes[1, 2], samples, nodes, args.jerk_limit)
    fig.suptitle(
        f"Third-order lattice on the 1-D car: Δ = {lattice.delta:g} m, δu = {lattice.du:g} m/s², J = {args.jerk_limit:g} m/s³, "
        f"{(lattice.top + 1) * lattice.num_levels} states per stage",
        fontsize=13,
    )
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=110)

    entry = np.argwhere(np.isfinite(tts.tau[stop, stop - 1]))
    moving = np.isfinite(tts.tau[-1, start])
    moving[0] = False
    slowest = np.sqrt(lattice.x(np.flatnonzero(moving.any(axis=1)).min()))
    print(f"states per stage: {(lattice.top + 1) * lattice.num_levels}")
    print("entry states of the last segment (ṡ [m/s], u [m/s²]): " + ", ".join(f"({np.sqrt(lattice.x(n)):.2f}, {lattice.u(level):g})" for n, level in entry))
    print(f"lowest non-zero lattice speed: {np.sqrt(lattice.unit):.2f} m/s; lowest speed of a moving state that can reach the goal: {slowest:.2f} m/s")
    print(f"acceleration levels along the fastest stop: {[round(float(a)) for a in samples[::30, 3]] + [0]}")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
