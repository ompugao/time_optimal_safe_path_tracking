import numpy as np
import pytest

from tospt.constraints import joint_limit_constraints
from tospt.controller import LITERAL, CycleDecider, PathFollower, SafePathController, time_to_arrive
from tospt.paths import straight_line_path, uniform_gridpoints
from tospt.stoppable_sets import compute_stoppable_sets
from tospt.time_to_reach import time_to_reach_torch

VMAX, AMAX, LENGTH, DT = 20.0, 100.0, 25.0, 1e-3


@pytest.fixture(scope="module")
def car():
    path = straight_line_path(LENGTH)
    sc = joint_limit_constraints(path, uniform_gridpoints(path, 250), [VMAX], [AMAX])
    K = compute_stoppable_sets(sc)
    return sc, K, time_to_reach_torch(sc, K, 100, device="cpu")


def run_car_against_wall(car, wall_position, wall_velocity, wall_speed_bound, duration=3.0, rule="per_stage"):
    """Returns the log rows (t, s, s_dot, s_ddot, wall, emergency)."""
    sc, K, ttr = car
    decider = CycleDecider(SafePathController(sc, K, ttr, rule=rule))
    follower = PathFollower(sc.gridpoints)
    log = []
    wall = wall_position
    for step in range(int(duration / DT)):
        decider.begin_cycle(time_to_arrive((wall - sc.gridpoints)[:, None], [wall_speed_bound]))
        decisions = follower.step(DT, decider)
        s, v, a = follower.position_velocity_acceleration()
        wall = max(wall + wall_velocity(step * DT) * DT, s) if wall > s else wall
        log.append((step * DT, s, v, a, wall, any(d.emergency for d in decisions)))
    return np.array(log)


def test_car_stops_before_a_static_wall(car):
    log = run_car_against_wall(car, 15.0, lambda t: 0.0, wall_speed_bound=VMAX)
    t, s, v, a, wall, emergency = log.T
    assert s.max() <= 15.0
    assert v[-1] == pytest.approx(0.0, abs=1e-9)
    assert not emergency.any()


def test_car_is_stationary_whenever_an_approaching_wall_touches_it(car):
    log = run_car_against_wall(car, 40.0, lambda t: -VMAX, wall_speed_bound=VMAX)
    t, s, v, a, wall, emergency = log.T
    touching = wall - s <= 1e-9
    assert touching.any()
    assert np.all(np.abs(v[touching]) <= 1e-9)


def test_car_respects_velocity_and_acceleration_limits(car):
    log = run_car_against_wall(car, 40.0, lambda t: -VMAX, wall_speed_bound=VMAX)
    t, s, v, a, wall, emergency = log.T
    assert np.all(v <= VMAX + 1e-6) and np.all(v >= -1e-9)
    assert np.all(np.abs(a) <= AMAX + 1e-6)


def test_car_without_obstacles_reaches_the_goal_at_nearly_bang_bang_time(car):
    log = run_car_against_wall(car, np.inf, lambda t: 0.0, wall_speed_bound=VMAX)
    t, s, v, a, wall, emergency = log.T
    arrival = t[np.argmax(s >= LENGTH - 1e-9)]
    assert s[-1] == pytest.approx(LENGTH)
    assert arrival == pytest.approx(1.45, abs=0.02)


def test_never_stops_one_stage_before_the_end_where_it_could_not_restart(car):
    sc, K, ttr = car
    controller = SafePathController(sc, K, ttr)
    n = sc.num_stages
    psi = np.full(n + 1, 10.0)
    psi[n] = 0.0
    stop = controller.select_stop_stage(n - 5, 0.0, psi)
    assert stop == n - 2
    free = np.full(n + 1, 10.0)
    assert controller.decide(n - 2, 0.0, free).u > 0.0


def test_decisions_later_in_the_cycle_see_less_time_to_arrive(car):
    sc, K, ttr = car
    decider = CycleDecider(SafePathController(sc, K, ttr))
    decider.begin_cycle(np.full(sc.num_stages + 1, 0.5))
    assert decider(0, 0.0, elapsed=0.0).u > 0.0
    decider.begin_cycle(np.full(sc.num_stages + 1, 0.5))
    assert decider(0, 0.0, elapsed=0.5).u == 0.0


def test_keeps_the_previously_checked_stop_when_the_grid_check_finds_none(car):
    sc, K, ttr = car
    controller = SafePathController(sc, K, ttr)
    i = 100
    x = 0.5 * K[150, i, 1]
    no_route = np.full(sc.num_stages + 1, 1e-6)
    kept = controller.decide(i, x, no_route, checked_stop=150)
    assert (kept.stop_stage, kept.kept_previous, kept.emergency) == (150, True, False)
    assert controller.decide(i, K[150, i, 1] * 1.5, no_route, checked_stop=150).emergency


def test_literal_rule_ignores_obstacles_near_intermediate_stages(car):
    sc, K, ttr = car
    psi = np.full(sc.num_stages + 1, 10.0)
    psi[100] = 1e-3
    stage, x = 0, 0.0
    per_stage = SafePathController(sc, K, ttr).select_stop_stage(stage, x, psi)
    literal = SafePathController(sc, K, ttr, rule=LITERAL).select_stop_stage(stage, x, psi)
    assert per_stage < 100 <= literal
