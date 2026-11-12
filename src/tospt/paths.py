import numpy as np
import toppra as ta


def straight_line_path(length):
    return ta.SplineInterpolator([0.0, length], [[0.0], [length]], bc_type="natural")


def joint_space_path(waypoints):
    waypoints = np.asarray(waypoints, dtype=float)
    return ta.SplineInterpolator(np.linspace(0.0, 1.0, len(waypoints)), waypoints)


def random_joint_path(rng, dof=6, n_waypoints=3):
    return joint_space_path(rng.uniform(-np.pi / 2, np.pi / 2, size=(n_waypoints, dof)))


def uniform_gridpoints(path, num_stages):
    s0, s1 = path.path_interval
    return np.linspace(s0, s1, num_stages + 1)
