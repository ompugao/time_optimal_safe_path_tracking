import numpy as np

from tospt.lp2d import one_step_set


def compute_stoppable_sets(sc):
    """K[j, i] = [x_lo, x_hi]: states at stage i from which the robot can stop exactly at stage j.

    Entries with i > j are NaN. The paper runs one backward pass per j in parallel
    processes; here all j share the same sweep over i, and each distinct target
    interval is solved once, since far from j the sets repeat.
    """
    n = sc.num_stages
    K = np.full((n + 1, n + 1, 2), np.nan)
    stop_stages = np.arange(n + 1)
    K[stop_stages, stop_stages] = 0.0
    for i in range(n - 1, -1, -1):
        js = stop_stages[i + 1 :]
        targets, inverse = np.unique(K[js, i + 1], axis=0, return_inverse=True)
        lo, hi = one_step_set(sc, i, targets[:, 0], targets[:, 1])
        K[js, i, 0] = lo[inverse.ravel()]
        K[js, i, 1] = hi[inverse.ravel()]
    return K
