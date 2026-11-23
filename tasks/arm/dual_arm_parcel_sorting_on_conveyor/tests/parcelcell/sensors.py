"""Public side: what a controller program may be given.

Joint position / velocity / motor torque, gripper state and camera images.
Nothing here reads parcel or belt state (that is ``parcelcell.truth``).

Cameras (name: mount, default resolution, default vertical field of view):
  head_cam         column head, looks down at the work area   640x400, 66 deg
  Left_wrist_cam   left wrist roll link, along the tool axis  424x240, 58 deg
  Right_wrist_cam  right wrist roll link                      424x240, 58 deg
  inspect_cam      overhead, above the upstream belt          640x480, 62 deg
  overview_cam     fixed overview of the station              640x480, 55 deg
The head and wrist defaults follow the source's alignment with its real depth
cameras (head ~91 x 66 deg, wrist ~87 x 58 deg); the robot MJCF itself
carries 75 / 70 deg. Pixels are square; the horizontal field of view follows
from the aspect ratio. ``intrinsics`` gives the OpenCV-style K matrix.

Rendering here uses MuJoCo's classic OpenGL renderer (previews; set
``MUJOCO_GL=egl`` for headless use).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import mujoco
import numpy as np

__all__ = ["CameraSpec", "DEFAULT_CAMERAS", "JointState", "GripperState",
           "CameraRig", "TorqueSensor"]


@dataclass(frozen=True)
class CameraSpec:
    name: str
    width: int
    height: int
    fovy: float                # vertical field of view (deg)
    depth_range: tuple = (0.01, 50.0)   # metres; outside -> 0 in depth images


DEFAULT_CAMERAS = (
    CameraSpec("head_cam", 640, 400, 66.0, (0.25, 10.0)),
    CameraSpec("Left_wrist_cam", 424, 240, 58.0, (0.07, 0.6)),
    CameraSpec("Right_wrist_cam", 424, 240, 58.0, (0.07, 0.6)),
    CameraSpec("inspect_cam", 640, 480, 62.0),
    CameraSpec("overview_cam", 640, 480, 55.0),
)


@dataclass
class JointState:
    names: tuple
    q: np.ndarray            # rad, JOINT_NAMES order
    qd: np.ndarray           # rad/s
    tau: np.ndarray          # motor torque (actuator force after limits), N m
    q_target: np.ndarray     # target applied in the last tick (after slew)


@dataclass
class GripperState:
    command: np.ndarray      # (2,) last command, 0 open .. 1 closed (L, R)
    closure: np.ndarray      # (2,) measured: driver angle / closed angle
    force: np.ndarray        # (2,) actuator force of the finger drive


class CameraRig:
    """Renders the configured cameras with MuJoCo's classic renderer."""

    def __init__(self, model: mujoco.MjModel, cameras=DEFAULT_CAMERAS):
        self.model = model
        self.specs = {c.name: c for c in cameras}
        self.cam_id = {}
        for c in cameras:
            cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, c.name)
            if cid < 0:
                raise ValueError(f"no camera named {c.name!r}")
            model.cam_fovy[cid] = c.fovy
            self.cam_id[c.name] = cid
        w = max(c.width for c in cameras)
        h = max(c.height for c in cameras)
        model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), w)
        model.vis.global_.offheight = max(int(model.vis.global_.offheight), h)
        self._renderers: dict[tuple, mujoco.Renderer] = {}

    def _renderer(self, h: int, w: int) -> mujoco.Renderer:
        key = (h, w)
        r = self._renderers.get(key)
        if r is None:
            r = mujoco.Renderer(self.model, h, w)
            self._renderers[key] = r
        return r

    def intrinsics(self, name: str) -> np.ndarray:
        c = self.specs[name]
        f = 0.5 * c.height / np.tan(np.deg2rad(c.fovy) / 2)
        return np.array([[f, 0, c.width / 2], [0, f, c.height / 2], [0, 0, 1]])

    def extrinsics(self, data: mujoco.MjData, name: str) -> np.ndarray:
        """camera -> world, OpenCV axes (x right, y down, z forward)."""
        cid = self.cam_id[name]
        T = np.eye(4)
        T[:3, :3] = data.cam_xmat[cid].reshape(3, 3) @ np.diag([1.0, -1.0, -1.0])
        T[:3, 3] = data.cam_xpos[cid]
        return T

    def rgb(self, data: mujoco.MjData, name: str) -> np.ndarray:
        c = self.specs[name]
        r = self._renderer(c.height, c.width)
        r.update_scene(data, camera=name)
        return r.render().copy()

    def depth(self, data: mujoco.MjData, name: str) -> np.ndarray:
        """Metric depth (float32, m); outside the camera's depth range -> 0."""
        c = self.specs[name]
        r = self._renderer(c.height, c.width)
        r.enable_depth_rendering()
        try:
            r.update_scene(data, camera=name)
            d = r.render().astype(np.float32)
        finally:
            r.disable_depth_rendering()
        lo, hi = c.depth_range
        d[~np.isfinite(d) | (d < lo) | (d > hi)] = 0.0
        return d

    def close(self) -> None:
        for r in self._renderers.values():
            r.close()
        self._renderers.clear()


class TorqueSensor:
    """The motor torque readings as the drives report them (the ``tau`` of the
    public state): the true motor torque plus a zero-mean white noise (one
    sigma = ``noise_rel`` x the joint's torque limit, a new draw every tick)
    plus a slowly wandering offset (at most ``offset_rel`` x the limit: per
    joint a sum of three slow sinusoids with periods in ``periods_s`` and
    weights summing to one, drawn once per shift). Every draw comes from the
    shift's digest (and the tick for the white noise): the same shift gives the
    same readings in any process, whatever was asked before. Positions and
    speeds are exact; the privileged state and the safety watch read the true
    torques."""

    K = 3

    def __init__(self, limits, digest: int, noise_rel: float, offset_rel: float, periods_s=(40.0, 240.0)):
        lim = np.abs(np.asarray(limits, float))
        n = len(lim)
        d = int(digest) & ((1 << 64) - 1)
        self.seed = (d & 0xFFFFFFFF, d >> 32)
        self.sigma = float(noise_rel) * lim
        rng = np.random.default_rng([self.seed[0], self.seed[1], 0x7A0FF5E7])
        w = rng.dirichlet(np.ones(self.K), size=n)
        self.amp = float(offset_rel) * lim[:, None] * w
        lo, hi = (float(v) for v in periods_s)
        self.omega = 2.0 * math.pi / rng.uniform(lo, hi, size=(n, self.K))
        self.phase = rng.uniform(0.0, 2.0 * math.pi, size=(n, self.K))

    def offset(self, t: float) -> np.ndarray:
        return (self.amp * np.sin(self.omega * float(t) + self.phase)).sum(axis=1)

    def read(self, tau, tick: int, t: float) -> np.ndarray:
        tau = np.asarray(tau, float)
        g = np.random.default_rng([self.seed[0], self.seed[1], 0x7A0B015E, int(tick) + (1 << 24)])
        return tau + g.normal(0.0, 1.0, tau.shape) * self.sigma + self.offset(t)
