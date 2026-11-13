"""Third-order (jerk-limited) version for a straight path with uniform stages: q(s) = s.

The acceleration u = s_ddot is a state. It is continuous and linear in s on every
segment, and at grid points it takes values m * du. Then
x_{i+1} = x_i + delta * (u_i + u_{i+1}), so x = s_dot^2 stays on the lattice
n * delta * du and the state is the integer pair (n, m): nothing is rounded.

With u linear in s, rest (x = 0, u = 0) is an equilibrium that is reached or left
only in infinite time. So the segment into a stop and the segment out of rest use
constant jerk in time instead, which requires u = -+ 2 x / (3 delta) at the moving end.
"""

import math
from dataclasses import dataclass

import numpy as np

TOL = 1e-9


@dataclass(frozen=True)
class Segment:
    """Motion over one stage: sigma(t) is the distance from the start of the segment."""

    kind: str
    v0: float
    u0: float
    rate: float
    duration: float

    def distance(self, t):
        if self.kind == "cubic":
            return self.v0 * t + 0.5 * self.u0 * t * t + self.rate * t**3 / 6.0
        w = self.rate
        if w > 0.0:
            k = math.sqrt(w)
            return self.v0 / k * math.sinh(k * t) + self.u0 / w * (math.cosh(k * t) - 1.0)
        if w < 0.0:
            k = math.sqrt(-w)
            return self.v0 / k * math.sin(k * t) + self.u0 / w * (math.cos(k * t) - 1.0)
        return self.v0 * t + 0.5 * self.u0 * t * t

    def speed(self, t):
        if self.kind == "cubic":
            return self.v0 + self.u0 * t + 0.5 * self.rate * t * t
        w = self.rate
        if w > 0.0:
            k = math.sqrt(w)
            return self.v0 * math.cosh(k * t) + self.u0 / k * math.sinh(k * t)
        if w < 0.0:
            k = math.sqrt(-w)
            return self.v0 * math.cos(k * t) + self.u0 / k * math.sin(k * t)
        return self.v0 + self.u0 * t


class CarLattice:
    def __init__(self, length, num_stages, vmax, amax, jmax, refinement=1):
        self.num_stages = num_stages
        self.delta = length / num_stages
        self.gridpoints = np.linspace(0.0, length, num_stages + 1)
        self.jmax = jmax
        # One acceleration level per stage must be allowed at top speed: du * vmax / delta <= jmax.
        # The segments into and out of rest exist only for even levels, so the count is even.
        self.half_levels = 2 * refinement * math.ceil(0.5 * amax * vmax / (jmax * self.delta) - TOL)
        self.du = amax / self.half_levels
        self.unit = self.delta * self.du
        self.top = math.floor(vmax * vmax / self.unit + TOL)
        self.zero = self.half_levels
        self._build_tables()

    @property
    def num_levels(self):
        return 2 * self.half_levels + 1

    def x(self, n):
        return n * self.unit

    def u(self, level):
        return (level - self.zero) * self.du

    def _build_tables(self):
        n = np.arange(self.top + 1)[:, None, None]
        m = (np.arange(self.num_levels) - self.zero)[None, :, None]
        m_next = (np.arange(self.num_levels) - self.zero)[None, None, :]
        n_next = n + m + m_next
        x0, x1 = self.x(n), self.x(n_next)
        u0 = m * self.du
        w = (m_next - m) * self.du / self.delta
        valid = (n >= 1) & (n_next >= 1) & (n_next <= self.top)

        # x(sigma) = x0 + 2 u0 sigma + w sigma^2 has an extremum inside the segment iff u changes sign.
        with np.errstate(divide="ignore", invalid="ignore"):
            x_inside = np.where(m * m_next < 0, x0 - u0 * u0 / w, x0)
        valid &= (x_inside > TOL * self.unit) & (x_inside <= self.x(self.top) * (1 + TOL))
        x_peak = np.maximum(np.maximum(x0, x1), x_inside)
        valid &= np.abs(w) * np.sqrt(np.maximum(x_peak, 0.0)) <= self.jmax * (1 + TOL)

        self.next_n = np.where(valid, n_next, 0)
        self.duration = np.where(valid, _linear_segment_duration(x0, x1, u0, w, self.delta, valid), np.inf)

        # Constant-jerk segment between rest and (x, u = -+ 2x / (3 delta)): n = 1.5 |m|, duration 3 delta / v.
        levels = np.arange(self.num_levels) - self.zero
        rest_n = 3 * np.abs(levels) // 2
        x = self.x(rest_n)
        with np.errstate(divide="ignore"):
            rest_duration = 3.0 * self.delta / np.sqrt(x)
        jerk = 2.0 * x**1.5 / (9.0 * self.delta**2)
        ok = (levels % 2 == 0) & (levels != 0) & (rest_n <= self.top) & (jerk <= self.jmax * (1 + TOL))
        self.rest_n = rest_n
        self.start_duration = np.where(ok & (levels > 0), rest_duration, np.inf)
        self.stop_duration = np.where(ok & (levels < 0), rest_duration, np.inf)

    def next_state(self, stage, n, level, next_level, stop_stage):
        if stop_stage == stage + 1:
            return 0, self.zero
        if n == 0:
            return int(self.rest_n[next_level]), next_level
        return n + (level - self.zero) + (next_level - self.zero), next_level

    def segment(self, stage, n, level, next_level, stop_stage):
        v0 = math.sqrt(self.x(n))
        u0 = self.u(level)
        if stop_stage == stage + 1:
            duration = float(self.stop_duration[level])
            return Segment("cubic", v0, u0, -u0 / duration, duration)
        if n == 0:
            duration = float(self.start_duration[next_level])
            return Segment("cubic", 0.0, 0.0, self.u(next_level) / duration, duration)
        w = (next_level - level) * self.du / self.delta
        return Segment("linear", v0, u0, w, float(self.duration[n, level, next_level]))


def _linear_segment_duration(x0, x1, u0, w, delta, valid):
    """Integral of d sigma / sqrt(x0 + 2 u0 sigma + w sigma^2) over the segment, in closed form."""
    shape = np.broadcast_shapes(x0.shape, x1.shape, u0.shape, w.shape)
    x0, x1, u0, w = (np.broadcast_to(arr, shape).astype(float) for arr in (x0, x1, u0, w))
    v0 = np.sqrt(np.where(valid, x0, 1.0))
    v1 = np.sqrt(np.where(valid, x1, 1.0))
    duration = np.full(shape, np.inf)
    flat = valid & (w == 0)
    duration[flat] = 2.0 * delta / (v0[flat] + v1[flat])
    up = valid & (w > 0)
    k = np.sqrt(w[up])
    u1 = u0[up] + w[up] * delta
    numerator, denominator = k * v1[up] + u1, k * v0[up] + u0[up]
    # When w x = u^2, x is a perfect square in sigma and both terms vanish; the integral is then log(u0 / u1) / k.
    square = np.abs(denominator) <= 1e-9 * (k * v0[up] + np.abs(u0[up]))
    ratio = np.where(square, u0[up], numerator) / np.where(square, u1, denominator)
    duration[up] = np.log(np.abs(ratio)) / k
    down = valid & (w < 0)
    k = np.sqrt(-w[down])
    radius = np.sqrt(u0[down] ** 2 - w[down] * x0[down])
    angle1 = np.arcsin(np.clip((-w[down] * delta - u0[down]) / radius, -1.0, 1.0))
    angle0 = np.arcsin(np.clip(-u0[down] / radius, -1.0, 1.0))
    duration[down] = (angle1 - angle0) / k
    return duration


@dataclass
class TimeToStop:
    """tau[j, i, n, level]: fastest time from stage i in state (n, level) to rest at stage j.

    rho: acceleration level at stage i + 1 on that route (-1 if there is none).
    """

    tau: np.ndarray
    rho: np.ndarray


def time_to_stop(lattice, dtype=np.float32):
    stages = lattice.num_stages
    shape = (stages + 1, stages + 1, lattice.top + 1, lattice.num_levels)
    tau = np.full(shape, np.inf, dtype=dtype)
    rho = np.full(shape, -1, dtype=np.int8)
    every_stage = np.arange(stages + 1)
    tau[every_stage, every_stage, 0, lattice.zero] = 0.0
    duration = lattice.duration.astype(dtype)
    next_level = np.broadcast_to(np.arange(lattice.num_levels), duration.shape)
    start_levels = np.flatnonzero(np.isfinite(lattice.start_duration))
    stop_levels = np.flatnonzero(np.isfinite(lattice.stop_duration))
    for i in range(stages - 1, -1, -1):
        later = slice(i + 1, stages + 1)
        tau_next = tau[later, i + 1]
        candidates = duration[None] + tau_next[:, lattice.next_n, next_level]
        best = candidates.argmin(axis=-1)
        best_time = np.take_along_axis(candidates, best[..., None], axis=-1)[..., 0]
        tau[later, i] = best_time
        rho[later, i] = np.where(np.isfinite(best_time), best, -1)

        from_rest = lattice.start_duration[start_levels].astype(dtype) + tau_next[:, lattice.rest_n[start_levels], start_levels]
        first = from_rest.argmin(axis=-1)
        rest_time = np.take_along_axis(from_rest, first[:, None], axis=-1)[:, 0]
        tau[later, i, 0, lattice.zero] = rest_time
        rho[later, i, 0, lattice.zero] = np.where(np.isfinite(rest_time), start_levels[first], -1)

        tau[i + 1, i, lattice.rest_n[stop_levels], stop_levels] = lattice.stop_duration[stop_levels]
        rho[i + 1, i, lattice.rest_n[stop_levels], stop_levels] = lattice.zero
    return TimeToStop(tau, rho)


@dataclass
class LatticeDecision:
    next_level: int
    stop_stage: int
    emergency: bool
    kept_previous: bool = False


class ThirdOrderController:
    def __init__(self, lattice, tts):
        self.lattice = lattice
        self.tau = tts.tau
        self.rho = tts.rho
        stages = lattice.num_stages
        # A stop is only useful where the car can leave again; one stage before the end it cannot.
        self.can_restart = np.isfinite(self.tau[stages, :, 0, lattice.zero])
        self.can_restart[stages] = True

    def _advance(self, stops, stage, n, level):
        """States one stage later on the stored routes to each stop stage."""
        lattice = self.lattice
        next_level = self.rho[stops, stage, n, level].astype(np.int64)
        regular = n + (level - lattice.zero) + (next_level - lattice.zero)
        next_n = np.where(n == 0, lattice.rest_n[next_level], regular)
        arriving = stops == stage + 1
        return np.where(arriving, 0, next_n), np.where(arriving, lattice.zero, next_level)

    def select_stop_stage(self, stage, n, level, psi):
        """Farthest stop whose stored route passes every stage before an obstacle can reach it (Eq. 7)."""
        at_rest = n == 0
        fallback = stage if at_rest else None
        start_time = self.tau[stage + 1 :, stage, n, level]
        stops = np.flatnonzero(np.isfinite(start_time) & self.can_restart[stage + 1 :]) + stage + 1
        if len(stops) == 0 or psi[stage] <= 0.0:
            return fallback
        start_time = start_time[stops - stage - 1].astype(float)
        can_fail = np.flatnonzero(psi[stage + 1 : stops.max() + 1] <= start_time.max()) + stage + 1
        safe = np.ones(len(stops), dtype=bool)
        ns = np.full(len(stops), n)
        levels = np.full(len(stops), level)
        for l in range(stage + 1, (can_fail.max() if len(can_fail) else stage) + 1):
            active = safe & (stops >= l)
            if not active.any():
                break
            next_n, next_level = self._advance(stops, l - 1, ns, levels)
            ns = np.where(active, next_n, ns)
            levels = np.where(active, next_level, levels)
            arrival = start_time - self.tau[stops, l, ns, levels]
            safe &= ~active | (arrival < psi[l])
        if not safe.any():
            return fallback
        return int(stops[safe].max())

    def decide(self, stage, n, level, psi, checked_stop=None):
        """checked_stop: stop stage whose route passed the check at an earlier decision.

        The car follows that route exactly, and Time-to-Arrive cannot drop faster
        than time passes, so the route stays safe and is kept when no other is found.
        """
        lattice = self.lattice
        if stage >= lattice.num_stages:
            return LatticeDecision(lattice.zero, lattice.num_stages, emergency=False)
        stop = self.select_stop_stage(stage, n, level, psi)
        kept = False
        emergency = False
        if stop is None:
            if checked_stop is not None and checked_stop > stage:
                stop, kept = checked_stop, True
            else:
                stop, emergency = stage + 1 + int(np.argmin(self.tau[stage + 1 :, stage, n, level])), True
        if stop == stage:
            return LatticeDecision(lattice.zero, stage, emergency=False)
        return LatticeDecision(int(self.rho[stop, stage, n, level]), stop, emergency, kept)


class LatticeFollower:
    def __init__(self, lattice):
        self.lattice = lattice
        self.stage = 0
        self.n = 0
        self.level = lattice.zero
        self.segment = None
        self.target = None
        self.elapsed = 0.0
        self.checked_stop = None

    @property
    def at_end(self):
        return self.stage >= self.lattice.num_stages

    @property
    def position(self):
        s = self.lattice.gridpoints[self.stage]
        return s if self.segment is None else s + self.segment.distance(self.elapsed)

    @property
    def velocity(self):
        if self.segment is not None:
            return self.segment.speed(self.elapsed)
        return math.sqrt(self.lattice.x(self.n))

    def step(self, dt, controller, psi):
        """Advance by dt. A decision is taken at every stage boundary with psi minus the time already used."""
        remaining = dt
        decisions = []
        while remaining > 0.0 and not self.at_end:
            if self.segment is None:
                decision = controller.decide(self.stage, self.n, self.level, psi - (dt - remaining), self.checked_stop)
                decisions.append(decision)
                self.checked_stop = None if decision.emergency else decision.stop_stage
                if decision.stop_stage == self.stage:
                    break
                state = (self.stage, self.n, self.level, decision.next_level, decision.stop_stage)
                self.segment = self.lattice.segment(*state)
                self.target = self.lattice.next_state(*state)
                self.elapsed = 0.0
            if self.elapsed + remaining < self.segment.duration:
                self.elapsed += remaining
                break
            remaining -= self.segment.duration - self.elapsed
            self.stage += 1
            self.n, self.level = self.target
            self.segment = None
            self.elapsed = 0.0
        return decisions
