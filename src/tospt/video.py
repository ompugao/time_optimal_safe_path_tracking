import os
import subprocess

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np

TOUCH_RGBA = np.array([0.9, 0.1, 0.1, 0.9])


class VideoRecorder:
    """Renders the scene offscreen and pipes the frames to ffmpeg (H.264 mp4).

    Obstacles closer than `touch_distance` to the robot are drawn red.
    """

    def __init__(self, model, path, fps=25, width=960, height=540):
        self.model = model
        self.data = mujoco.MjData(model)
        self.fps = fps
        model.vis.global_.offwidth = max(model.vis.global_.offwidth, width)
        model.vis.global_.offheight = max(model.vis.global_.offheight, height)
        self.renderer = mujoco.Renderer(model, height, width)
        self.camera = mujoco.MjvCamera()
        self.camera.lookat[:] = [-0.3, -0.1, 0.35]
        self.camera.distance = 1.5
        self.camera.elevation = -20.0
        self.camera.azimuth = 0.0
        self.obstacle_geoms = np.array([model.geom(f"obstacle{q}").id for q in range(model.nmocap)], dtype=int)
        self.free_rgba = model.geom_rgba[self.obstacle_geoms].copy()
        self.process = subprocess.Popen(
            ["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", str(fps),
             "-i", "-", "-pix_fmt", "yuv420p", "-vcodec", "libx264", str(path)],
            stdin=subprocess.PIPE,
        )

    def capture(self, qpos, obstacle_pos, obstacle_distances, touch_distance=0.0):
        self.data.qpos[: len(qpos)] = qpos
        self.data.mocap_pos[:] = obstacle_pos
        touching = np.asarray(obstacle_distances) <= touch_distance
        self.model.geom_rgba[self.obstacle_geoms] = np.where(touching[:, None], TOUCH_RGBA, self.free_rgba)
        mujoco.mj_forward(self.model, self.data)
        self.renderer.update_scene(self.data, self.camera)
        self.process.stdin.write(self.renderer.render().tobytes())

    def close(self):
        self.model.geom_rgba[self.obstacle_geoms] = self.free_rgba
        self.process.stdin.close()
        self.process.wait()
        self.renderer.close()
