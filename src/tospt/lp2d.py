import numpy as np

from tospt.lp1d import is_feasible, u_bounds_numpy

U_BOUND = 1e6


def polygon_vertices(a, b, c, tol=1e-9):
    """Vertices (u, x) of the bounded polygon {a u + b x <= c} (1-D row arrays)."""
    p, q = np.triu_indices(len(a), k=1)
    det = a[p] * b[q] - a[q] * b[p]
    regular = np.abs(det) > 1e-12
    p, q, det = p[regular], q[regular], det[regular]
    u = (c[p] * b[q] - c[q] * b[p]) / det
    x = (a[p] * c[q] - a[q] * c[p]) / det
    lhs = np.outer(u, a) + np.outer(x, b)
    scale = 1.0 + np.abs(c) + np.abs(np.outer(u, a)) + np.abs(np.outer(x, b))
    inside = np.all(lhs <= c + tol * scale, axis=1)
    return u[inside], x[inside]


def admissible_rows(sc, i):
    """Rows of Omega_i in (u, x), closed with 0 <= x <= xmax and a large |u| bound."""
    a = np.concatenate([sc.a[i], [0.0, 0.0, 1.0, -1.0]])
    b = np.concatenate([sc.b[i], [1.0, -1.0, 0.0, 0.0]])
    c = np.concatenate([sc.c[i], [sc.xmax[i], 0.0, U_BOUND, U_BOUND]])
    return a, b, c


def one_step_set(sc, i, target_lo, target_hi, tol=1e-9):
    """Q_i([lo, hi]): states x_i that some admissible u_i steers into [lo, hi] at stage i+1.

    Batched over target intervals. The x-range of Omega_i cut by the strip
    lo <= x + 2 delta u <= hi is attained either at a vertex of Omega_i inside the
    strip or on one of the two strip lines, where it is a 1-D LP in x.
    """
    target_lo = np.asarray(target_lo, float)
    target_hi = np.asarray(target_hi, float)
    two_delta = 2.0 * (sc.gridpoints[i + 1] - sc.gridpoints[i])
    a, b, c = admissible_rows(sc, i)
    vertex_u, vertex_x = polygon_vertices(a, b, c)

    candidates_lo = []
    candidates_hi = []
    line_coeff = b - a / two_delta
    for level in (target_lo, target_hi):
        rhs = c - np.outer(level, a) / two_delta
        alpha, beta, zero_ok = u_bounds_numpy(np.broadcast_to(line_coeff, rhs.shape), rhs, tol)
        on_line = is_feasible(alpha, beta, zero_ok, tol)
        candidates_lo.append(np.where(on_line, alpha, np.inf))
        candidates_hi.append(np.where(on_line, beta, -np.inf))

    strip = vertex_x + two_delta * vertex_u
    inside = (strip >= target_lo[:, None] - tol * (1 + np.abs(target_lo[:, None]))) & (
        strip <= target_hi[:, None] + tol * (1 + np.abs(target_hi[:, None]))
    )
    candidates_lo.append(np.min(np.where(inside, vertex_x, np.inf), axis=1, initial=np.inf))
    candidates_hi.append(np.max(np.where(inside, vertex_x, -np.inf), axis=1, initial=-np.inf))

    x_lo = np.min(candidates_lo, axis=0)
    x_hi = np.max(candidates_hi, axis=0)
    empty = ~np.isfinite(x_hi) | np.isnan(target_lo) | np.isnan(target_hi)
    return np.where(empty, np.nan, np.maximum(x_lo, 0.0)), np.where(empty, np.nan, x_hi)
