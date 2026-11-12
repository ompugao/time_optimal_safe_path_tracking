import mujoco
import numpy as np
import pytest

from tospt.mujoco_scene import UR5E_SCENE, CapsuleRobot, load_scene, point_to_capsules_distance

pytestmark = pytest.mark.skipif(not UR5E_SCENE.exists(), reason="run scripts/fetch_assets.sh first")
OBSTACLE_RADIUS = 0.05


@pytest.fixture(scope="module")
def scene():
    model, data = load_scene(1, OBSTACLE_RADIUS)
    return model, data, CapsuleRobot(model, data)


@pytest.mark.parametrize("seed", range(5))
def test_capsule_distance_matches_mujoco_and_never_overestimates_cylinders(scene, seed):
    model, data, robot = scene
    rng = np.random.default_rng(seed)
    q = rng.uniform(-np.pi, np.pi, 6)
    point = rng.uniform([-0.8, -0.8, 0.0], [0.8, 0.8, 1.0])
    segments = robot.segments(q)[0]
    data.mocap_pos[0] = point
    mujoco.mj_kinematics(model, data)
    obstacle_geom = model.geom("obstacle0").id
    for g, geom in enumerate(robot.geoms):
        ours = point_to_capsules_distance(segments[g : g + 1], robot.radii[g : g + 1], point[None], OBSTACLE_RADIUS)[0]
        reference = mujoco.mj_geomDistance(model, data, int(geom), obstacle_geom, 10.0, None)
        if int(model.geom_type[geom]) == int(mujoco.mjtGeom.mjGEOM_CAPSULE):
            assert ours == pytest.approx(reference, abs=1e-6)
        else:
            assert ours <= reference + 1e-6


def test_distances_are_batched_over_configurations_and_points(scene):
    _, _, robot = scene
    qs = np.zeros((4, 6))
    points = np.array([[0.5, 0.5, 0.5], [-0.5, 0.2, 0.3], [0.0, 0.0, 2.0]])
    distances = point_to_capsules_distance(robot.segments(qs), robot.radii, points)
    assert distances.shape == (4, 3)
    assert np.allclose(distances, distances[0])
