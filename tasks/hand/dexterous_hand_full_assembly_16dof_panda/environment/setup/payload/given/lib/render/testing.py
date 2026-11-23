"""Test helpers for the render layer (used by tests/test_render_*.py; handy for task authors' local checks).

    write_tabletop_mjcf(dirpath)   -> path of a small MJCF: floor, a 0.6 x 0.4 m table top at z = 0.40, a blue box, a
                                      free-floating red sphere (joint "ball_free"), a hinge arm "arm_hinge", one light
    FakeBackend(name, raw_fn)      a second backend whose render_raw is `raw_fn(request, raw_cameras) -> RawRender`;
                                   `dots_raw_fn(points)` draws ideal Gaussian dots at the pinhole projections of 3D points
"""
from __future__ import annotations

import os
from typing import Callable, List

import numpy as np

from .backends.base import BackendCapabilities, RawRender, RawView, RenderBackend
from .spec import CameraSpec, RenderRequest

TABLE_TOP_Z = 0.40

_TABLETOP = """<mujoco model="render_test_tabletop">
  <compiler angle="radian"/>
  <visual><headlight ambient="0.35 0.35 0.35" diffuse="0.45 0.45 0.45" specular="0.05 0.05 0.05"/></visual>
  <asset>
    <texture name="floor_tex" type="2d" builtin="checker" rgb1="0.46 0.44 0.42" rgb2="0.40 0.38 0.36" width="64" height="64"/>
    <material name="floor_mat" texture="floor_tex" texrepeat="6 6"/>
    <material name="table_mat" rgba="0.55 0.42 0.30 1"/>
    <material name="box_mat" rgba="0.12 0.31 0.86 1"/>
  </asset>
  <worldbody>
    <light name="key" pos="0.4 -0.6 1.8" dir="-0.2 0.3 -1" castshadow="true" diffuse="0.6 0.6 0.6"/>
    <geom name="floor" type="plane" size="2 2 0.01" material="floor_mat"/>
    <body name="table" pos="0 0 0">
      <geom name="table_top" type="box" size="0.3 0.2 0.02" pos="0 0 0.38" material="table_mat"/>
      <geom name="table_leg" type="box" size="0.03 0.03 0.18" pos="0 0 0.18" material="table_mat"/>
    </body>
    <body name="box" pos="0.18 0.08 0.43"><geom name="box_geom" type="box" size="0.03 0.03 0.03" material="box_mat"/></body>
    <body name="ball" pos="-0.15 0.1 0.45"><joint name="ball_free" type="free"/><geom name="ball_geom" type="sphere" size="0.025" rgba="0.85 0.15 0.12 1"/></body>
    <body name="arm" pos="-0.2 -0.1 0.40">
      <joint name="arm_hinge" type="hinge" axis="0 0 1"/>
      <geom name="arm_geom" type="capsule" fromto="0 0 0.01 0.12 0 0.01" size="0.01" rgba="0.2 0.7 0.3 1"/>
    </body>
  </worldbody>
</mujoco>
"""


def write_tabletop_mjcf(dirpath: str) -> str:
    os.makedirs(dirpath, exist_ok=True)
    p = os.path.join(dirpath, "tabletop.xml")
    with open(p, "w") as f:
        f.write(_TABLETOP)
    return p


class FakeBackend(RenderBackend):
    """A second backend for tests: same pipeline, pixels from `raw_fn`."""

    def __init__(self, name: str = "fake", raw_fn: Callable = None, outputs=("rgb", "depth", "segmentation")):
        self.name = name
        self.raw_fn = raw_fn
        self._outputs = list(outputs)

    def version(self) -> str:
        return "fake-backend/1.0.0"

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(name=self.name, implemented=True, outputs=self._outputs, notes="test double")

    def render_raw(self, request: RenderRequest, raw_cameras: List[CameraSpec]) -> RawRender:
        return self.raw_fn(request, raw_cameras)


def dots_raw_fn(points_world, sigma_px: float = 1.2, background: int = 30, peak: int = 230):
    """raw_fn drawing a Gaussian dot at the ideal pinhole projection of every point (no distortion: the raw camera)."""
    P = np.asarray(points_world, float)

    def fn(request, raw_cameras):
        views = []
        for k, cam in enumerate(raw_cameras):
            uv, z = cam.project(P, distort=False)
            vv, uu = np.mgrid[0:cam.height, 0:cam.width].astype(np.float64)
            img = np.zeros((cam.height, cam.width))
            s = sigma_px * cam.fx / request.cameras[k].fx    # dot size scales with supersampling
            for (u, v), zz in zip(uv, z):
                if zz > 0:
                    img += np.exp(-((uu - u) ** 2 + (vv - v) ** 2) / (2 * s * s))
            rgb = np.clip(background + (peak - background) * np.clip(img, 0, 1), 0, 255)
            views.append(RawView(rgb=np.repeat(np.rint(rgb).astype(np.uint8)[..., None], 3, axis=2),
                                 depth=np.zeros((cam.height, cam.width), np.float32), seg=np.full((cam.height, cam.width), -1, np.int32)))
        return RawRender(views=views, labels={}, resolved_scene_digest="fake", environment={"fake": "1"})
    return fn
