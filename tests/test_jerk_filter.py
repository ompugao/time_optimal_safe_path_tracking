import numpy as np
import pytest

from tospt.constraints import joint_limit_constraints
from tospt.controller import CycleDecider, SafePathController, time_to_arrive
from tospt.jerk_filter import MovingAverage, SmoothedFollower, jerk_limited_constraints, path_third_derivative, window_samples
from tospt.paths import joint_space_path, straight_line_path, uniform_gridpoints
from tospt.stoppable_sets import compute_stoppable_sets
from tospt.time_to_reach import time_to_reach_torch

VMAX, AMAX, JMAX, LENGTH, DT = 20.0, 100.0, 5000.0, 25.0, 1e-3


@pytest.fixture(scope="module")
def car():
    path = straight_line_path(LENGTH)
    limited = jerk_limited_constraints(joint_limit_constraints(path, uniform_gridpoints(path, 250), [VMAX], [AMAX]), path, [JMAX])
    K = compute_stoppable_sets(limited.sc)
    return limited, K, time_to_reach_torch(limited.sc, K, 100, device="cpu")


def run_car_against_wall(car, wall_position, wall_velocity, duration=3.0):
    """Returns the log rows (t, executed s, reference s, wall)."""
    limited, K, ttr = car
    sc = limited.sc
    decider = CycleDecider(SafePathController(sc, K, ttr))
    follower = SmoothedFollower(sc.gridpoints, window_samples(limited.window, DT))
    log = []
    wall = wall_position
    for step in range(int(duration / DT)):
        psi = time_to_arrive((wall - sc.gridpoints)[:, None], [VMAX], latency=follower.latency(DT))
        decider.begin_cycle(psi)
        follower.step(DT, decider)
        s = follower.position
        wall = max(wall + wall_velocity * DT, s) if wall > s else wall
        log.append((step * DT, s, follower.reference.position_velocity_acceleration()[0], wall))
    return np.array(log)


def derivative(samples, order):
    return np.diff(samples, n=order, axis=0) / DT**order


def test_moving_average_is_never_ahead_of_a_rising_input_and_lags_at_most_its_window():
    rng = np.random.default_rng(0)
    reference = np.cumsum(rng.uniform(0.0, 1.0, 200))
    reference[120:] = reference[120]
    average = MovingAverage(10, reference[0])
    output = np.array([average.step(value) for value in reference])
    assert np.all(output <= reference)
    assert np.all(output[9:] >= reference[:-9])
    assert output[129] == reference[120]


def test_window_of_a_straight_path_is_the_time_to_swing_the_acceleration_at_the_jerk_limit(car):
    limited, _, _ = car
    assert limited.geometric_budget == pytest.approx(0.0)
    assert limited.window == pytest.approx(2 * AMAX / JMAX)


def test_car_jerk_stays_within_the_limit(car):
    log = run_car_against_wall(car, 40.0, -VMAX)
    assert np.abs(derivative(log[:, 1], 3)).max() <= JMAX * (1 + 1e-6)


def test_car_still_respects_velocity_and_acceleration_limits(car):
    log = run_car_against_wall(car, 40.0, -VMAX)
    assert derivative(log[:, 1], 1).min() >= -1e-9
    assert derivative(log[:, 1], 1).max() <= VMAX + 1e-6
    assert np.abs(derivative(log[:, 1], 2)).max() <= AMAX * (1 + 1e-6)


def test_car_is_stationary_whenever_an_approaching_wall_touches_it(car):
    log = run_car_against_wall(car, 40.0, -VMAX)
    s, wall = log[:, 1], log[:, 3]
    velocity = np.concatenate([[0.0], derivative(s, 1)])
    touching = wall - s <= 1e-9
    assert touching.any()
    assert np.all(np.abs(velocity[touching]) <= 1e-9)


def test_car_stops_before_a_static_wall_at_the_stage_where_the_reference_stops(car):
    log = run_car_against_wall(car, 15.0, 0.0)
    assert log[:, 1].max() <= 15.0
    assert log[-1, 1] == log[-1, 2]


def test_car_without_obstacles_reaches_the_goal(car):
    log = run_car_against_wall(car, 1e9, 0.0)
    assert log[-1, 1] == pytest.approx(LENGTH)


@pytest.fixture(scope="module")
def robot():
    rng = np.random.default_rng(3)
    path = joint_space_path(rng.uniform(-1.0, 1.0, size=(5, 6)))
    vlim, alim, jlim = np.full(6, np.pi), np.full(6, 20.0), np.full(6, 1000.0)
    sc = joint_limit_constraints(path, uniform_gridpoints(path, 120), vlim, alim)
    limited = jerk_limited_constraints(sc, path, jlim)
    K = compute_stoppable_sets(limited.sc)
    return path, limited, K, time_to_reach_torch(limited.sc, K, 60, device="cpu"), jlim


def run_robot(robot, dt=0.004, duration=4.0):
    """Returns the executed positions and the decisions (stage, x, u) of the reference."""
    path, limited, K, ttr, _ = robot
    sc = limited.sc
    decider = CycleDecider(SafePathController(sc, K, ttr))
    follower = SmoothedFollower(sc.gridpoints, window_samples(limited.window, dt))
    executed, decisions = [], []

    def recording_decider(stage, x, elapsed):
        decision = decider(stage, x, elapsed)
        decisions.append((stage, x, decision.u))
        return decision

    for _ in range(int(duration / dt)):
        decider.begin_cycle(np.full(sc.num_stages + 1, np.inf))
        follower.step(dt, recording_decider)
        executed.append(follower.position)
    return np.array(executed), np.array(decisions)


def test_geometric_jerk_at_every_decision_stays_within_its_budget(robot):
    path, limited, _, _, _ = robot
    _, decisions = run_robot(robot)
    stage, x, u = decisions.T
    s = limited.sc.gridpoints[stage.astype(int)]
    geometric = np.sqrt(x)[:, None] * (3 * path(s, 2) * u[:, None] + path_third_derivative(path, s) * x[:, None])
    assert np.all(limited.geometric_budget > 0)
    assert np.all(np.abs(geometric).max(axis=0) <= limited.geometric_budget * (1 + 1e-6))


def test_robot_joint_jerk_stays_within_the_limit(robot):
    path, _, _, _, jlim = robot
    executed, _ = run_robot(robot)
    assert executed[-1] == pytest.approx(1.0)
    jerk = np.diff(path(executed), n=3, axis=0) / 0.004**3
    assert np.all(np.abs(jerk).max(axis=0) <= jlim)
