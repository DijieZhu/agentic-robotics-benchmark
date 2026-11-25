"""Offscreen rendering with MuJoCo's classic renderer (needs an OSMesa or EGL library; MUJOCO_GL selects it).

Set MESA_SHADER_CACHE_DISABLE=true before any renderer exists (done at import of this module); a run on
2026-11-17 without it updated ~/.cache/mesa_shader_cache_db.

Inspection media only: the struts are the authors' mesh (end-cap covers coloured red / green / blue by material,
as in the authors' photos and perception config), the six lines are the actuated tendons drawn as thin lines, the
stretch sensors have no 3D shape from the authors and are not drawn.
"""
import math
import os

# Mesa (OSMesa / llvmpipe) would otherwise write its shader cache under ~/.cache (not allowed on this host);
# must be set before the first GL context is created.
os.environ.setdefault("MESA_SHADER_CACHE_DISABLE", "true")

import mujoco
import numpy as np


class Camera:
    """Free camera that follows the robot's centre with exponential smoothing."""

    def __init__(self, azimuth=135.0, elevation=-24.0, distance=0.95, smooth=0.04, lookat_z=0.07):
        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.cam.azimuth, self.cam.elevation, self.cam.distance = azimuth, elevation, distance
        self.smooth = smooth
        self.lookat = None
        self.lookat_z = lookat_z

    def update(self, sim, dt_frame):
        c = sim.com().copy()
        c[2] = self.lookat_z
        if self.lookat is None:
            self.lookat = c
        else:
            a = 1.0 - math.exp(-dt_frame / max(self.smooth, 1e-6))
            self.lookat = self.lookat + a * (c - self.lookat)
        self.cam.lookat[:] = self.lookat


class Recorder:
    def __init__(self, sim, width=1280, height=720, fps=30, camera=None, show_collision=False):
        self.sim = sim
        self.r = mujoco.Renderer(sim.m, height, width)
        self.fps = fps
        self.cam = camera or Camera()
        self.opt = mujoco.MjvOption()
        self.opt.geomgroup[:] = [1, 1, 1, 1 if show_collision else 0, 0, 0]
        self.opt.sitegroup[:] = [0] * 6
        self.opt.flags[mujoco.mjtVisFlag.mjVIS_TENDON] = 1
        self.frames = []
        self.every = max(1, int(round(1.0 / (fps * sim.dt))))

    def grab(self):
        self.cam.update(self.sim, 1.0 / self.fps)
        self.r.update_scene(self.sim.d, self.cam.cam, self.opt)
        img = self.r.render()
        self.frames.append(img.copy())
        return img

    def step_and_record(self, n_steps):
        for _ in range(n_steps):
            if self.sim.nstep % self.every == 0:
                self.grab()
            self.sim.step(1)
