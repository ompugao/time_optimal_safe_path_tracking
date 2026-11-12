import numpy as np
import pytest
from scipy.optimize import linprog

from tospt.constraints import joint_limit_constraints
from tospt.lp1d import max_u_numpy, max_u_scalar
from tospt.paths import random_joint_path, straight_line_path, uniform_gridpoints
from tospt.stoppable_sets import compute_stoppable_sets
from tospt.time_to_reach import time_to_reach_serial, time_to_reach_torch

torch = pytest.importorskip("torch")
requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def car_problem(num_stages=200, length=25.0):
    path = straight_line_path(length)
    sc = joint_limit_constraints(path, uniform_gridpoints(path, num_stages), [20.0], [100.0])
    return sc, compute_stoppable_sets(sc)


def robot_problem(seed=0, num_stages=40):
    rng = np.random.default_rng(seed)
    path = random_joint_path(rng)
    sc = joint_limit_constraints(path, uniform_gridpoints(path, num_stages), rng.uniform(1, 3, 6), rng.uniform(2, 10, 6))
    return sc, compute_stoppable_sets(sc)


@pytest.mark.parametrize("seed", range(20))
def test_1d_lp_matches_linprog(seed):
    rng = np.random.default_rng(seed)
    a = rng.normal(size=6)
    b = rng.normal(size=6)
    res = linprog([-1.0], A_ub=a[:, None], b_ub=b, bounds=[(None, None)])
    u_vec, feasible_vec = max_u_numpy(a, b)
    u_scalar, feasible_scalar = max_u_scalar(a.tolist(), b.tolist())
    assert feasible_vec == feasible_scalar == (res.status != 2)
    if res.status == 0:
        assert u_vec == pytest.approx(-res.fun) and u_scalar == pytest.approx(-res.fun)


def test_car_time_to_reach_from_rest_is_close_to_bang_bang_time():
    sc, K = car_problem(num_stages=100)
    ttr = time_to_reach_serial(sc, K, num_velocity_steps=50)
    bang_bang = 2 * 0.2 + (25.0 - 2 * 2.0) / 20.0
    assert bang_bang <= ttr.tau[-1, 0, 0] <= bang_bang * 1.05


def test_time_to_reach_never_underestimates_the_bang_bang_time():
    sc, K = car_problem(num_stages=100)
    ttr = time_to_reach_serial(sc, K, num_velocity_steps=20)
    for j in (30, 70, 100):
        distance = sc.gridpoints[j] - sc.gridpoints[0]
        assert ttr.tau[j, 0, 0] >= 2 * np.sqrt(distance / 100.0) - 1e-9


def test_coarse_velocity_grid_overestimates_time_when_stages_are_fine():
    """Flooring rho can keep a route at the same velocity index when one stage
    adds less than dv; the estimate becomes slower, never faster."""
    sc, K = car_problem(num_stages=500)
    coarse = time_to_reach_serial(sc, K, num_velocity_steps=30).tau[-1, 0, 0]
    fine = time_to_reach_serial(sc, K, num_velocity_steps=200).tau[-1, 0, 0]
    assert coarse > 1.5 * fine


def test_routes_stay_inside_the_stoppable_set_and_end_stationary():
    sc, K = robot_problem()
    ttr = time_to_reach_serial(sc, K, num_velocity_steps=10)
    j = sc.num_stages
    k = 0
    for i in range(j):
        assert np.isfinite(ttr.tau[j, i, k])
        x = (k * ttr.dv) ** 2
        assert K[j, i, 0] - 1e-9 <= x <= K[j, i, 1] + 1e-9
        k = ttr.rho[j, i, k]
    assert k == 0


def test_unreachable_grid_states_have_infinite_time():
    sc, K = car_problem(num_stages=50)
    ttr = time_to_reach_serial(sc, K, num_velocity_steps=10)
    j = 10
    top_speed = ttr.dv * 10
    assert np.isinf(ttr.tau[j, j - 1, 10]) and top_speed**2 > K[j, j - 1, 1]


def plateau_problem():
    """Path of the 6-DoF experiment, where the velocity limit is active over many stages
    and sqrt(K)/dv lands just below the top index at some of them."""
    import toppra as ta

    from tospt.paths import joint_space_path

    q_a = np.array([-0.6, -1.2, 1.6, -1.9, -1.57, 0.0])
    q_b = np.array([0.9, -1.2, 1.6, -1.9, -1.57, 0.0])
    lift = np.array([0.0, -0.4, 0.0, 0.0, 0.0, 0.0])
    path = joint_space_path([q_a, 0.5 * (q_a + q_b) + lift, q_b])
    gridpoints = ta.interpolator.propose_gridpoints(path, min_nb_points=500)
    sc = joint_limit_constraints(path, gridpoints, np.full(6, np.pi), np.full(6, 20.0))
    return sc, compute_stoppable_sets(sc)


@pytest.mark.parametrize("backend", ["serial", "cpu", pytest.param("cuda", marks=requires_cuda)])
def test_top_grid_velocity_is_usable_wherever_the_set_reaches_it(backend):
    sc, K = plateau_problem()
    M = 30
    ttr = time_to_reach_serial(sc, K, M) if backend == "serial" else time_to_reach_torch(sc, K, M, device=backend)
    j = sc.num_stages
    top = np.isclose(np.sqrt(K[j, :j, 1]), M * ttr.dv, rtol=1e-12)
    assert top.sum() > 100
    assert np.all(np.isfinite(ttr.tau[j, :j, M][top]))


@pytest.mark.parametrize("problem", [car_problem, robot_problem, plateau_problem])
def test_torch_backend_matches_serial(problem):
    sc, K = problem()
    expected = time_to_reach_serial(sc, K, num_velocity_steps=12)
    actual = time_to_reach_torch(sc, K, num_velocity_steps=12, device="cpu", stage_chunk=7)
    assert actual.dv == expected.dv
    np.testing.assert_array_equal(actual.rho, expected.rho)
    np.testing.assert_allclose(actual.tau, expected.tau, rtol=1e-12)
