import math

import numpy as np
import pytest

from tospt.third_order import CarLattice, LatticeFollower, ThirdOrderController, time_to_stop

VMAX, AMAX, JMAX, LENGTH, STAGES, DT = 20.0, 100.0, 5000.0, 25.0, 100, 1e-3


@pytest.fixture(scope="module")
def car():
    lattice = CarLattice(LENGTH, STAGES, VMAX, AMAX, JMAX, refinement=2)
    return lattice, time_to_stop(lattice, dtype=np.float64)


def reachable_states(car, count=200, seed=0):
    lattice, tts = car
    states = np.argwhere(np.isfinite(tts.tau) & (tts.rho >= 0))
    return states[np.random.default_rng(seed).choice(len(states), count, replace=False)]


def run_car_against_wall(car, wall_position, wall_velocity, duration=4.0):
    """Returns the log rows (t, s, s_dot, wall, emergency)."""
    lattice, tts = car
    controller = ThirdOrderController(lattice, tts)
    follower = LatticeFollower(lattice)
    wall = wall_position
    log = []
    for step in range(int(duration / DT)):
        decisions = follower.step(DT, controller, (wall - lattice.gridpoints) / VMAX)
        s = follower.position
        wall = max(wall + wall_velocity * DT, s) if wall > s else wall
        log.append((step * DT, s, follower.velocity, wall, any(d.emergency for d in decisions)))
    return np.array(log)


def derivative(samples, order):
    return np.diff(samples, n=order) / DT**order


def test_segment_duration_matches_numerical_integration(car):
    lattice, _ = car
    nodes, weights = np.polynomial.legendre.leggauss(200)
    sigma = 0.5 * lattice.delta * (nodes + 1.0)
    for n, level, next_level in np.argwhere(np.isfinite(lattice.duration))[::7]:
        w = (next_level - level) * lattice.du / lattice.delta
        x = lattice.x(n) + 2.0 * lattice.u(level) * sigma + w * sigma**2
        expected = 0.5 * lattice.delta * np.sum(weights / np.sqrt(x))
        assert lattice.duration[n, level, next_level] == pytest.approx(expected, rel=1e-9)


def test_every_segment_ends_exactly_on_the_next_lattice_state(car):
    lattice, tts = car
    for stop, stage, n, level in reachable_states(car):
        next_level = int(tts.rho[stop, stage, n, level])
        segment = lattice.segment(stage, n, level, next_level, stop)
        next_n, _ = lattice.next_state(stage, n, level, next_level, stop)
        assert segment.distance(segment.duration) == pytest.approx(lattice.delta, rel=1e-9)
        assert segment.speed(segment.duration) == pytest.approx(math.sqrt(lattice.x(next_n)), abs=1e-7)


def test_stored_routes_end_at_rest_at_the_stop_stage_after_the_stored_time(car):
    lattice, tts = car
    for stop, stage, n, level in reachable_states(car, count=50):
        total = 0.0
        expected = tts.tau[stop, stage, n, level]
        while stage < stop:
            next_level = int(tts.rho[stop, stage, n, level])
            total += lattice.segment(stage, n, level, next_level, stop).duration
            n, level = lattice.next_state(stage, n, level, next_level, stop)
            stage += 1
        assert (n, level) == (0, lattice.zero)
        assert total == pytest.approx(expected, rel=1e-9)


def test_car_without_obstacles_reaches_the_goal(car):
    log = run_car_against_wall(car, 1e9, 0.0)
    assert log[-1, 1] == pytest.approx(LENGTH)
    assert not log[:, 4].any()


def test_car_respects_velocity_acceleration_and_jerk_limits(car):
    log = run_car_against_wall(car, 40.0, -VMAX)
    s = log[:, 1]
    assert log[:, 2].min() >= -1e-9 and log[:, 2].max() <= VMAX + 1e-6
    assert np.abs(derivative(s, 2)).max() <= AMAX * (1 + 1e-6)
    assert np.abs(derivative(s, 3)).max() <= JMAX * (1 + 1e-6)


def test_car_is_stationary_whenever_an_approaching_wall_touches_it(car):
    log = run_car_against_wall(car, 40.0, -VMAX)
    t, s, v, wall, emergency = log.T
    touching = wall - s <= 1e-9
    assert touching.any()
    assert np.all(np.abs(v[touching]) <= 1e-9)
    assert not emergency.any()


def test_car_stops_before_a_static_wall(car):
    log = run_car_against_wall(car, 15.0, 0.0)
    assert log[:, 1].max() <= 15.0
    assert log[-1, 2] == pytest.approx(0.0, abs=1e-9)
