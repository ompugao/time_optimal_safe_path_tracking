import numpy as np
import pytest
from scipy.optimize import linprog

from tospt.constraints import joint_limit_constraints
from tospt.lp2d import one_step_set
from tospt.paths import random_joint_path, straight_line_path, uniform_gridpoints
from tospt.stoppable_sets import compute_stoppable_sets


def car_constraints(num_stages=50, length=25.0, vmax=20.0, amax=100.0):
    path = straight_line_path(length)
    return joint_limit_constraints(path, uniform_gridpoints(path, num_stages), [vmax], [amax])


def robot_constraints(seed=0, num_stages=40):
    rng = np.random.default_rng(seed)
    path = random_joint_path(rng)
    return joint_limit_constraints(path, uniform_gridpoints(path, num_stages), rng.uniform(1, 3, 6), rng.uniform(2, 10, 6))


@pytest.mark.parametrize("seed", range(5))
def test_one_step_set_matches_linprog(seed):
    sc = robot_constraints(seed)
    rng = np.random.default_rng(seed)
    for i in rng.integers(0, sc.num_stages, size=5):
        lo, hi = np.sort(rng.uniform(0, sc.xmax[i + 1], 2))
        x_lo, x_hi = one_step_set(sc, i, np.array([lo]), np.array([hi]))
        two_delta = 2 * sc.deltas[i]
        A = np.vstack([np.stack([sc.a[i], sc.b[i]], 1), [two_delta, 1], [-two_delta, -1]])
        ub = np.concatenate([sc.c[i], [hi, -lo]])
        bounds = [(None, None), (0, sc.xmax[i])]
        for sign, expected in ((1, x_lo[0]), (-1, x_hi[0])):
            res = linprog([0, sign], A_ub=A, b_ub=ub, bounds=bounds)
            if res.status == 2:
                assert np.isnan(expected)
            else:
                assert expected == pytest.approx(res.x[1], rel=1e-6, abs=1e-9)


def test_car_can_stop_at_its_current_stage_only_when_stationary():
    K = compute_stoppable_sets(car_constraints())
    n = K.shape[0] - 1
    for j in range(n + 1):
        assert K[j, j].tolist() == [0.0, 0.0]


def test_car_stoppable_set_matches_braking_distance():
    sc = car_constraints()
    K = compute_stoppable_sets(sc)
    j = sc.num_stages
    braking_distance = sc.gridpoints[j] - sc.gridpoints
    expected = np.minimum(2 * 100.0 * braking_distance, 20.0**2)
    np.testing.assert_allclose(K[j, :, 1], expected, rtol=1e-9)


@pytest.mark.parametrize("make", [car_constraints, robot_constraints])
def test_stoppable_sets_are_nested_toward_later_stops(make):
    """Theorem 1 of the paper: K[j, i] is a subset of K[k, i] for j < k."""
    K = compute_stoppable_sets(make())
    n = K.shape[0] - 1
    tol = 1e-9
    for i in range(n + 1):
        for j in range(i, n):
            assert K[j + 1, i, 0] <= K[j, i, 0] + tol
            assert K[j, i, 1] <= K[j + 1, i, 1] + tol
