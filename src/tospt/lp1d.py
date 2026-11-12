"""1-D LP  max u  s.t.  a u <= b  (paper Sec. III-C).

Rows with a > 0 bound u from above (beta), rows with a < 0 from below (alpha);
the LP is feasible iff alpha <= beta, and then u* = beta. Only min/max and
arithmetic are used, so batched versions have no data-dependent branches.
"""

import math

import numpy as np

EPS = 1e-12
TOL = 1e-9


def u_bounds_numpy(a, b, tol=TOL):
    positive = a > EPS
    negative = a < -EPS
    ratio = b / np.where(positive | negative, a, 1.0)
    beta = np.min(np.where(positive, ratio, np.inf), axis=-1)
    alpha = np.max(np.where(negative, ratio, -np.inf), axis=-1)
    zero_rows_ok = np.all(positive | negative | (b >= -tol * (1.0 + np.abs(b))), axis=-1)
    return alpha, beta, zero_rows_ok


def u_bounds_torch(a, b, tol=TOL):
    import torch

    positive = a > EPS
    negative = a < -EPS
    ratio = b / torch.where(positive | negative, a, torch.ones_like(a))
    inf = torch.full_like(ratio, math.inf)
    beta = torch.where(positive, ratio, inf).amin(dim=-1)
    alpha = torch.where(negative, ratio, -inf).amax(dim=-1)
    zero_rows_ok = (positive | negative | (b >= -tol * (1.0 + b.abs()))).all(dim=-1)
    return alpha, beta, zero_rows_ok


def is_feasible(alpha, beta, zero_rows_ok, tol=TOL):
    return (alpha <= beta + tol * (1.0 + abs(beta))) & zero_rows_ok


def max_u_numpy(a, b, tol=TOL):
    alpha, beta, zero_rows_ok = u_bounds_numpy(a, b, tol)
    return beta, is_feasible(alpha, beta, zero_rows_ok, tol)


def max_u_scalar(a, b, tol=TOL):
    beta = math.inf
    alpha = -math.inf
    for a_r, b_r in zip(a, b):
        if a_r > EPS:
            beta = min(beta, b_r / a_r)
        elif a_r < -EPS:
            alpha = max(alpha, b_r / a_r)
        elif b_r < -tol * (1.0 + abs(b_r)):
            return beta, False
    return beta, alpha <= beta + tol * (1.0 + abs(beta))
