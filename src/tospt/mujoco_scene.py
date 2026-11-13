from pathlib import Path

import mujoco
import numpy as np

UR5E_SCENE = Path(__file__).resolve().parents[2] / "assets/mujoco_menagerie/universal_robots_ur5e/scene.xml"
COLLISION_GROUP = 3


def load_scene(num_obstacles, obstacle_radius, scene_path=UR5E_SCENE, payload_mass=0.0):
    spec = mujoco.MjSpec.from_file(str(scene_path))
    if payload_mass > 0.0:
        flange = spec.body("wrist_3_link")
        payload = flange.add_body(name="payload", pos=[0.0, 0.15, 0.0])
        payload.add_geom(
            type=mujoco.mjtGeom.mjGEOM_CAPSULE,
            size=[0.04, 0.02, 0],
            mass=payload_mass,
            group=COLLISION_GROUP,
            rgba=[0.8, 0.4, 0.1, 1.0],
            contype=0,
            conaffinity=0,
        )
    for q in range(num_obstacles):
        body = spec.worldbody.add_body(name=f"obstacle{q}", mocap=True, pos=[0.0, 0.0, -1.0])
        body.add_geom(
            name=f"obstacle{q}",
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[obstacle_radius, 0, 0],
            rgba=[0.2, 0.8, 0.2, 0.8],
            contype=0,
            conaffinity=0,
        )
    model = spec.compile()
    return model, mujoco.MjData(model)


class CapsuleRobot:
    """Distances from points to the robot's own collision geoms.

    Capsules are exact. Cylinders are treated as capsules of the same radius and
    half-length; this only adds the end caps, so distances are never overestimated.
    """

    def __init__(self, model, data, dof=6):
        self.model = model
        self.data = data
        self.dof = dof
        shapes = (int(mujoco.mjtGeom.mjGEOM_CAPSULE), int(mujoco.mjtGeom.mjGEOM_CYLINDER))
        self.geoms = np.array(
            [g for g in range(model.ngeom) if model.geom_group[g] == COLLISION_GROUP and int(model.geom_type[g]) in shapes],
            dtype=int,
        )
        if len(self.geoms) == 0:
            raise ValueError("no capsule or cylinder collision geoms found")
        self.radii = model.geom_size[self.geoms, 0].copy()
        self.half_lengths = model.geom_size[self.geoms, 1].copy()

    def segments(self, qs):
        """Capsule axis endpoints for each configuration: (len(qs), num_geoms, 2, 3)."""
        qs = np.atleast_2d(qs)
        out = np.empty((len(qs), len(self.geoms), 2, 3))
        for n, q in enumerate(qs):
            self.data.qpos[: self.dof] = q
            mujoco.mj_kinematics(self.model, self.data)
            centers = self.data.geom_xpos[self.geoms]
            axes = self.data.geom_xmat[self.geoms].reshape(-1, 3, 3)[:, :, 2]
            offset = axes * self.half_lengths[:, None]
            out[n, :, 0] = centers - offset
            out[n, :, 1] = centers + offset
        return out


def point_to_capsules_distance(segments, radii, points, point_radius=0.0):
    """Minimum distance over capsules: segments (..., G, 2, 3), points (Q, 3) -> (..., Q)."""
    start = segments[..., 0, :][..., None, :]
    axis = (segments[..., 1, :] - segments[..., 0, :])[..., None, :]
    rel = points - start
    length_sq = np.maximum(np.sum(axis * axis, axis=-1), 1e-18)
    t = np.clip(np.sum(rel * axis, axis=-1) / length_sq, 0.0, 1.0)
    gap = np.linalg.norm(rel - t[..., None] * axis, axis=-1) - radii[:, None] - point_radius
    return gap.min(axis=-2)
