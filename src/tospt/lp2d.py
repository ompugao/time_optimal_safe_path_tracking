import numpy as np


def x_range_of_polygon(a, b, c, tol=1e-9):
    """Min and max of x over {(u, x) | a u + b x <= c}, batched over the leading axis.

    The polygon must be bounded. Solved by enumerating vertices, which keeps the
    computation branch-free and vectorized. Returns NaN for empty sets.
    """
    a, b, c = np.broadcast_arrays(np.asarray(a, float), np.asarray(b, float), np.asarray(c, float))
    n_rows = a.shape[-1]
    p, q = np.triu_indices(n_rows, k=1)
    det = a[..., p] * b[..., q] - a[..., q] * b[..., p]
    regular = np.abs(det) > 1e-12
    safe_det = np.where(regular, det, 1.0)
    u = (c[..., p] * b[..., q] - c[..., q] * b[..., p]) / safe_det
    x = (a[..., p] * c[..., q] - a[..., q] * c[..., p]) / safe_det
    lhs = a[..., None, :] * u[..., :, None] + b[..., None, :] * x[..., :, None]
    scale = 1.0 + np.abs(c[..., None, :]) + np.abs(a[..., None, :] * u[..., :, None]) + np.abs(b[..., None, :] * x[..., :, None])
    inside = np.all(lhs <= c[..., None, :] + tol * scale, axis=-1) & regular
    x_lo = np.min(np.where(inside, x, np.inf), axis=-1)
    x_hi = np.max(np.where(inside, x, -np.inf), axis=-1)
    empty = ~np.any(inside, axis=-1)
    return np.where(empty, np.nan, x_lo), np.where(empty, np.nan, x_hi)


def one_step_set(sc, i, target_lo, target_hi):
    """Q_i([lo, hi]): states x_i that some admissible u_i steers into [lo, hi] at stage i+1.

    Batched over target intervals (1-D arrays of the same length).
    """
    target_lo = np.asarray(target_lo, float)
    target_hi = np.asarray(target_hi, float)
    batch = target_lo.shape
    two_delta = 2.0 * (sc.gridpoints[i + 1] - sc.gridpoints[i])
    stage_a = np.broadcast_to(sc.a[i], batch + sc.a[i].shape)
    stage_b = np.broadcast_to(sc.b[i], batch + sc.b[i].shape)
    stage_c = np.broadcast_to(sc.c[i], batch + sc.c[i].shape)
    ones = np.ones(batch)
    zeros = np.zeros(batch)
    a = np.concatenate([stage_a, np.stack([zeros, zeros, two_delta * ones, -two_delta * ones], -1)], -1)
    b = np.concatenate([stage_b, np.stack([ones, -ones, ones, -ones], -1)], -1)
    c = np.concatenate([stage_c, np.stack([sc.xmax[i] * ones, zeros, target_hi, -target_lo], -1)], -1)
    x_lo, x_hi = x_range_of_polygon(a, b, c)
    invalid = np.isnan(target_lo) | np.isnan(target_hi)
    x_lo = np.where(invalid, np.nan, np.maximum(x_lo, 0.0))
    x_hi = np.where(invalid, np.nan, x_hi)
    return x_lo, x_hi
