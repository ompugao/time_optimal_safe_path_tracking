"""Execution phase (paper Sec. III-B-2)."""

import math
from dataclasses import dataclass

import numpy as np

from tospt.lp1d import max_u_numpy
from tospt.time_to_reach import floor_index

PER_STAGE = "per_stage"
LITERAL = "literal"
SET_TOL = 1e-9


def time_to_arrive(distances, obstacle_max_speeds, protective_distance=0.0, latency=0.0):
    """psi_l = min_q (d_lq - d_protective) / v_q,max  (Eq. 6), minus the sensing latency.

    distances: (num_stages + 1, num_obstacles).
    """
    distances = np.asarray(distances, float).reshape(len(distances), -1)
    speeds = np.broadcast_to(np.asarray(obstacle_max_speeds, float), distances.shape[1:])
    if distances.shape[1] == 0:
        return np.full(len(distances), np.inf)
    return np.min((distances - protective_distance) / speeds, axis=1) - latency


@dataclass
class Decision:
    u: float
    stop_stage: int
    emergency: bool
    kept_previous: bool = False


class SafePathController:
    def __init__(self, sc, K, ttr, rule=PER_STAGE):
        if rule not in (PER_STAGE, LITERAL):
            raise ValueError(rule)
        self.sc = sc
        self.K = K
        self.ttr = ttr
        self.rule = rule
        self.n = sc.num_stages

    def in_stoppable_set(self, j, i, x):
        lo, hi = self.K[j, i, 0], self.K[j, i, 1]
        return (lo - SET_TOL * (1 + lo) <= x) & (x <= hi + SET_TOL * (1 + hi))

    def stoppable_stages(self, i, x):
        js = np.arange(i, self.n + 1)
        return js[self.in_stoppable_set(js, i, x)]

    def select_stop_stage(self, i, x, psi):
        """Farthest j whose fastest stopping route passes every stage before obstacles can reach it (Eq. 7).

        Returns None when no route is safe. Stage n - 1 is never chosen: with
        u constant per segment, a rest-to-rest move over the single last segment
        needs infinite time, so a stop there could never finish the path.
        """
        candidates = self.stoppable_stages(i, x)
        candidates = candidates[(candidates > i) & (candidates != self.n - 1)]
        stationary = x <= SET_TOL
        fallback = i if stationary else None
        if len(candidates) == 0 or psi[i] <= 0.0:
            return fallback
        top_index = floor_index(np.sqrt(self.K[candidates, i, 1]), self.ttr.dv).astype(int)
        k = np.minimum(min(int(floor_index(math.sqrt(x), self.ttr.dv)), self.ttr.num_velocities - 1), top_index)
        start_time = self.ttr.tau[candidates, i, k]
        safe = np.isfinite(start_time)
        if self.rule == LITERAL:
            safe &= start_time < psi[candidates]
        elif safe.any():
            latest_arrival = start_time[safe].max()
            can_fail = np.flatnonzero(psi[i + 1 : candidates.max() + 1] <= latest_arrival) + i + 1
            last = can_fail.max() if len(can_fail) else i
            can_fail_mask = np.zeros(self.n + 1, dtype=bool)
            can_fail_mask[can_fail] = True
            for l in range(i + 1, last + 1):
                active = safe & (candidates >= l)
                if not active.any():
                    break
                k = np.where(active, self.ttr.rho[candidates, l - 1, k], k)
                if not can_fail_mask[l]:
                    continue
                with np.errstate(invalid="ignore"):
                    arrival = start_time - self.ttr.tau[candidates, l, k]
                safe &= ~active | (arrival < psi[l])
        if not safe.any():
            return fallback
        return int(candidates[safe].max())

    def greedy_control(self, i, x, stop_stage):
        """Largest admissible u keeping the next state in K[stop_stage, i + 1]."""
        if stop_stage == i:
            return 0.0
        two_delta = 2.0 * self.sc.deltas[i]
        lo, hi = self.K[stop_stage, i + 1]
        a = np.concatenate([self.sc.a[i], [two_delta, -two_delta]])
        b = np.concatenate([self.sc.c[i] - self.sc.b[i] * x, [hi - x, x - lo]])
        u, feasible = max_u_numpy(a, b)
        if not feasible:
            u = (lo - x) / two_delta
        return float(min(max(u, (lo - x) / two_delta), (hi - x) / two_delta))

    def decide(self, i, x, psi, checked_stop=None):
        """checked_stop: stop stage that passed the safety check at an earlier decision.

        If the grid check now finds no safe stop, that stage is kept while the state
        is still inside its stoppable set. It stays safe because the robot has
        followed the greedy pass inside that set, which is never slower than the
        grid route that was checked (it starts at least as fast and takes the
        largest admissible step), and both the robot and Time-to-Arrive have
        advanced by the same elapsed time.
        """
        if i >= self.n:
            return Decision(0.0, self.n, emergency=False)
        stop = self.select_stop_stage(i, x, psi)
        if stop is not None:
            return Decision(self.greedy_control(i, x, stop), stop, emergency=False)
        if checked_stop is not None and checked_stop > i and self.in_stoppable_set(checked_stop, i, x):
            return Decision(self.greedy_control(i, x, checked_stop), checked_stop, emergency=False, kept_previous=True)
        candidates = self.stoppable_stages(i, x)
        stop = int(candidates.min()) if len(candidates) else self.n
        return Decision(self.greedy_control(i, x, stop), stop, emergency=True)


class CycleDecider:
    """Selects the stop stage once per control cycle, then follows the greedy
    TOPP-RA forward pass inside that stoppable set for the rest of the cycle.

    Time-to-Arrive is computed from obstacle data at the start of the cycle, so a
    decision taken `elapsed` seconds into the cycle uses psi - elapsed.
    """

    def __init__(self, controller):
        self.controller = controller
        self.psi = None
        self.stop = None
        self.checked_stop = None

    def begin_cycle(self, psi):
        self.psi = psi
        self.stop = None

    def __call__(self, i, x, elapsed=0.0):
        controller = self.controller
        if self.stop is not None and self.stop > i and controller.in_stoppable_set(self.stop, i, x):
            return Decision(controller.greedy_control(i, x, self.stop), self.stop, emergency=False)
        decision = controller.decide(i, x, self.psi - elapsed, checked_stop=self.checked_stop)
        self.stop = decision.stop_stage
        self.checked_stop = None if decision.emergency else decision.stop_stage
        return decision


class PathFollower:
    """Robot state on the discretized path; u is held constant over each segment [s_i, s_i+1]."""

    def __init__(self, gridpoints, start_stage=0):
        self.s_grid = np.asarray(gridpoints, float)
        self.stage = start_stage
        self.x = 0.0
        self.u = 0.0
        self.elapsed = 0.0
        self.committed = False

    @property
    def at_end(self):
        return self.stage >= len(self.s_grid) - 1

    def segment_duration(self):
        delta = self.s_grid[self.stage + 1] - self.s_grid[self.stage]
        v0 = math.sqrt(self.x)
        v1 = math.sqrt(max(self.x + 2.0 * delta * self.u, 0.0))
        return math.inf if v0 + v1 <= 0.0 else 2.0 * delta / (v0 + v1)

    def position_velocity_acceleration(self):
        v0 = math.sqrt(self.x)
        if self.at_end:
            return self.s_grid[-1], 0.0, 0.0
        t = self.elapsed
        return self.s_grid[self.stage] + v0 * t + 0.5 * self.u * t * t, v0 + self.u * t, self.u

    def step(self, dt, decide):
        """Advance by dt; decide(stage, x, elapsed_in_step) is called at every stage boundary and while stationary."""
        remaining = dt
        decisions = []
        while remaining > 0.0 and not self.at_end:
            if not self.committed:
                decision = decide(self.stage, self.x, dt - remaining)
                decisions.append(decision)
                self.u = decision.u
                self.elapsed = 0.0
                self.committed = True
            duration = self.segment_duration()
            if math.isinf(duration):
                self.committed = False
                break
            if self.elapsed + remaining < duration:
                self.elapsed += remaining
                break
            remaining -= duration - self.elapsed
            delta = self.s_grid[self.stage + 1] - self.s_grid[self.stage]
            self.x = max(self.x + 2.0 * delta * self.u, 0.0)
            self.stage += 1
            self.elapsed = 0.0
            self.u = 0.0
            self.committed = False
        return decisions
