"""Joint jerk limits on top of the second-order planner.

q''' = q'(s) s_dddot + 3 q''(s) s_dot s_ddot + q'''(s) s_dot^3

The last two terms (no s_dddot) are bounded by extra stage rows. The first term
is bounded by executing the moving average of the planned s(t): the robot stays
on the path, is never ahead of the plan, and lags it by at most the window.
"""

import math
from collections import deque
from dataclasses import dataclass

import numpy as np

from tospt.controller import PathFollower
from tospt.lp1d import u_bounds_numpy
from tospt.lp2d import admissible_rows, polygon_vertices


@dataclass(frozen=True)
class JerkLimitedConstraints:
    sc: object
    window: float
    geometric_budget: np.ndarray


def path_third_derivative(path, gridpoints):
    # toppra's SplineInterpolator rejects order 3, so the scipy spline is used directly.
    return np.atleast_2d(path.cspl.derivative(3)(gridpoints))


def _stage_vertices(sc):
    return [polygon_vertices(*admissible_rows(sc, i)) for i in range(len(sc.gridpoints))]


def add_geometric_jerk_rows(sc, path, jlim, share=0.25):
    """Rows sqrt(xmax_i) |3 q'' u + q''' x| <= budget, with budget = min(largest value on the path, share * jlim).

    sqrt(x) is replaced by its stage maximum to keep the rows linear. That is
    only loose at low speed, where these terms are small anyway.
    """
    jlim = np.asarray(jlim, float)
    if not np.all(np.isfinite(sc.xmax)):
        raise ValueError("jerk limits need a velocity limit at every stage")
    speed = np.sqrt(sc.xmax)[:, None]
    coeff_u = 3.0 * speed * np.atleast_2d(path(sc.gridpoints, 2))
    coeff_x = speed * path_third_derivative(path, sc.gridpoints)
    largest = np.zeros(len(jlim))
    for i, (u, x) in enumerate(_stage_vertices(sc)):
        if len(u):
            values = np.abs(np.outer(u, coeff_u[i]) + np.outer(x, coeff_x[i]))
            largest = np.maximum(largest, values.max(axis=0))
    budget = np.minimum(largest, share * jlim)
    limit = np.broadcast_to(budget, coeff_u.shape)
    constrained = sc.with_rows(
        np.concatenate([coeff_u, -coeff_u], axis=1),
        np.concatenate([coeff_x, -coeff_x], axis=1),
        np.concatenate([limit, limit], axis=1),
    )
    return constrained, budget


def acceleration_swing(sc):
    """Per stage, the widest interval of admissible u at one velocity."""
    swing = np.zeros(len(sc.gridpoints))
    for i in range(len(sc.gridpoints)):
        a, b, c = admissible_rows(sc, i)
        _, x = polygon_vertices(a, b, c)
        if len(x) == 0:
            continue
        # The width is concave and piecewise linear in x, so its maximum is at a vertex.
        alpha, beta, _ = u_bounds_numpy(np.broadcast_to(a, (len(x), len(a))), c - np.outer(x, b))
        swing[i] = max(np.max(beta - alpha), 0.0)
    return swing


def smoothing_window(sc, path, path_jerk_limit):
    """Shortest averaging window T with |q'| * swing / T <= path_jerk_limit at every stage.

    The swing is taken at one stage and one velocity, although the window covers
    a short piece of the path. The experiments measure the resulting jerk.
    """
    slope = np.abs(np.atleast_2d(path(sc.gridpoints, 1)))
    return float(np.max(slope * acceleration_swing(sc)[:, None] / path_jerk_limit))


def jerk_limited_constraints(sc, path, jlim, geometric_share=0.25):
    jlim = np.asarray(jlim, float)
    constrained, budget = add_geometric_jerk_rows(sc, path, jlim, geometric_share)
    return JerkLimitedConstraints(constrained, smoothing_window(constrained, path, jlim - budget), budget)


def window_samples(window, dt):
    return max(1, math.ceil(window / dt - 1e-9))


class MovingAverage:
    def __init__(self, num_samples, initial):
        self.samples = deque([float(initial)] * num_samples, maxlen=num_samples)

    @property
    def settled(self):
        return self.samples[0] == self.samples[-1]

    def step(self, value):
        self.samples.append(float(value))
        # A settled filter returns the sample itself, so the output reaches the final value exactly.
        return self.samples[-1] if self.settled else math.fsum(self.samples) / len(self.samples)


class SmoothedFollower:
    """Executes the moving average of the reference follower's position, sampled once per control cycle."""

    def __init__(self, gridpoints, num_samples):
        self.reference = PathFollower(gridpoints)
        self.filter = MovingAverage(num_samples, self.reference.s_grid[0])
        self.position = self.reference.s_grid[0]
        self.num_samples = num_samples

    def latency(self, dt):
        """Longest delay of the executed motion behind the sampled reference."""
        return (self.num_samples - 1) * dt

    @property
    def at_end(self):
        return self.reference.at_end and self.filter.settled

    def step(self, dt, decide):
        decisions = self.reference.step(dt, decide)
        self.position = self.filter.step(self.reference.position_velocity_acceleration()[0])
        return decisions
