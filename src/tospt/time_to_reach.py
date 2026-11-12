"""Time-to-Reach by dynamic programming (paper Alg. 1, after the stoppable sets).

tau[j, i, k]: fastest time from stage i at velocity k * dv to a stop at stage j.
rho[j, i, k]: velocity index reached at stage i + 1 on that fastest route (-1 if none).
"""

import math
from dataclasses import dataclass

import numpy as np

from tospt.lp1d import max_u_scalar, u_bounds_torch, is_feasible


@dataclass
class TimeToReach:
    tau: np.ndarray
    rho: np.ndarray
    dv: float

    @property
    def num_velocities(self) -> int:
        return self.tau.shape[-1]


def velocity_resolution(K, num_velocity_steps):
    """Uniform grid on s_dot (not on x = s_dot^2) up to the largest stoppable velocity."""
    return math.sqrt(np.nanmax(K[-1, :, 1])) / num_velocity_steps


INDEX_TOL = 1e-9


def floor_index(v, dv):
    """floor(v / dv), tolerant to rounding so a velocity exactly on the grid keeps its index.

    Without the tolerance, a set whose bound is the top grid velocity can lose
    that grid point at some stages, and tau = inf then spreads backward.
    """
    return np.floor(np.asarray(v) / dv + INDEX_TOL)


def ceil_index(v, dv):
    return np.ceil(np.asarray(v) / dv - INDEX_TOL)


def velocity_index_range(lo, hi, dv, num_velocity_steps):
    """Grid indices inside [lo, hi]; ceil/floor keep them inside the set."""
    low = int(ceil_index(math.sqrt(lo), dv))
    high = min(int(floor_index(math.sqrt(hi), dv)), num_velocity_steps)
    return low, high


def time_to_reach_serial(sc, K, num_velocity_steps):
    """Reference implementation: one scalar 1-D LP per (j, i, k).

    LP results are memoized across j when K[j, i] and K[j, i + 1] repeat, as in
    the paper's serial baseline.
    """
    n = sc.num_stages
    M = num_velocity_steps
    dv = velocity_resolution(K, M)
    tau = np.full((n + 1, n + 1, M + 1), np.inf)
    rho = np.full((n + 1, n + 1, M + 1), -1, dtype=np.int64)
    tau[np.arange(n + 1), np.arange(n + 1), 0] = 0.0
    stage_a = sc.a.tolist()
    stage_b = sc.b.tolist()
    stage_c = sc.c.tolist()
    deltas = sc.deltas.tolist()
    memo = {}
    for j in range(n, 0, -1):
        for i in range(j - 1, -1, -1):
            lo_cur, hi_cur = K[j, i]
            lo_next, hi_next = K[j, i + 1]
            if math.isnan(lo_cur) or math.isnan(lo_next):
                continue
            key = (i, lo_cur, hi_cur, lo_next, hi_next)
            steps = memo.get(key)
            if steps is None:
                steps = _greedy_steps(stage_a[i], stage_b[i], stage_c[i], deltas[i], lo_cur, hi_cur, lo_next, hi_next, dv, M)
                memo[key] = steps
            for k, t, r in steps:
                tau[j, i, k] = tau[j, i + 1, r] + t
                rho[j, i, k] = r
    return TimeToReach(tau, rho, dv)


def _greedy_steps(a, b, c, delta, lo_cur, hi_cur, lo_next, hi_next, dv, M):
    two_delta = 2.0 * delta
    steps = []
    low, high = velocity_index_range(lo_cur, hi_cur, dv, M)
    for k in range(low, high + 1):
        v = k * dv
        x = v * v
        rows_a = a + [two_delta, -two_delta]
        rows_b = [c_r - b_r * x for b_r, c_r in zip(b, c)] + [hi_next - x, x - lo_next]
        u, feasible = max_u_scalar(rows_a, rows_b)
        if not feasible:
            continue
        v_next = math.sqrt(min(max(x + two_delta * u, lo_next), hi_next))
        if v_next + v <= 0.0:
            continue
        r = min(int(floor_index(v_next, dv)), M)
        steps.append((k, two_delta / (v_next + v), r))
    return steps


def time_to_reach_torch(sc, K, num_velocity_steps, device="cuda", dtype=None, stage_chunk=64):
    """Parallel version: all 1-D LPs of a chunk of stages are solved in one batch.

    The DP recursion over i stays sequential but only gathers precomputed results.
    """
    import torch

    dtype = dtype or torch.float64
    n = sc.num_stages
    M = num_velocity_steps
    dv = velocity_resolution(K, M)
    as_t = lambda arr: torch.as_tensor(np.ascontiguousarray(arr), dtype=dtype, device=device)  # noqa: E731
    a, b, c = as_t(sc.a), as_t(sc.b), as_t(sc.c)
    two_delta = 2.0 * as_t(sc.deltas)
    Kt = as_t(K)
    ks = torch.arange(M + 1, device=device, dtype=dtype)
    vk = ks * dv
    xk = vk * vk
    stop_stages = torch.arange(n + 1, device=device)

    travel_time = torch.full((n + 1, n, M + 1), math.inf, dtype=dtype, device=device)
    next_index = torch.zeros((n + 1, n, M + 1), dtype=torch.int64, device=device)
    valid = torch.zeros((n + 1, n, M + 1), dtype=torch.bool, device=device)
    for start in range(0, n, stage_chunk):
        I = torch.arange(start, min(start + stage_chunk, n), device=device)
        alpha_s, beta_s, zero_ok = u_bounds_torch(
            a[I][:, None, :].expand(-1, M + 1, -1),
            c[I][:, None, :] - b[I][:, None, :] * xk[None, :, None],
        )
        td = two_delta[I][None, :, None]
        lo_next = Kt[:, I + 1, 0][..., None]
        hi_next = Kt[:, I + 1, 1][..., None]
        beta = torch.minimum(beta_s[None], (hi_next - xk) / td)
        alpha = torch.maximum(alpha_s[None], (lo_next - xk) / td)
        low = torch.ceil(torch.sqrt(Kt[:, I, 0]) / dv - INDEX_TOL)[..., None]
        high = torch.floor(torch.sqrt(Kt[:, I, 1]) / dv + INDEX_TOL)[..., None]
        in_grid = (ks >= low) & (ks <= high) & (stop_stages[:, None, None] > I[None, :, None])
        v_next = torch.sqrt(torch.minimum(torch.maximum(xk + td * beta, lo_next), hi_next))
        speed_sum = v_next + vk
        ok = in_grid & zero_ok[None] & is_feasible(alpha, beta, True) & (speed_sum > 0)
        travel_time[:, I] = torch.where(ok, td / speed_sum, math.inf)
        next_index[:, I] = torch.where(ok, torch.clamp(torch.floor(v_next / dv + INDEX_TOL), 0, M).long(), 0)
        valid[:, I] = ok

    tau = torch.full((n + 1, n + 1, M + 1), math.inf, dtype=dtype, device=device)
    tau[stop_stages, stop_stages, 0] = 0.0
    for i in range(n - 1, -1, -1):
        tau_next = torch.gather(tau[:, i + 1], 1, next_index[:, i])
        tau[:, i] = torch.where(valid[:, i], tau_next + travel_time[:, i], tau[:, i])
    rho = torch.where(valid, next_index, -1)
    rho = torch.cat([rho, torch.full((n + 1, 1, M + 1), -1, dtype=torch.int64, device=device)], dim=1)
    return TimeToReach(tau.cpu().numpy().astype(np.float64), rho.cpu().numpy(), dv)
