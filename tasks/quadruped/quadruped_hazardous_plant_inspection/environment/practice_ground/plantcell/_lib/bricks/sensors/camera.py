"""Colour + metric depth cameras on MuJoCo's classic renderer (MUJOCO_GL=osmesa on CPU), the sensor head and a
pan-tilt rig.

Camera models (pinhole, square pixels, principal point at the image centre ((w-1)/2, (h-1)/2) in OpenCV pixel
coordinates):
  WIDE   'nav'      640 x 480, 90 deg horizontal FOV, colour + depth (navigation / overview)
  NARROW 'inspect' 1280 x 960, 18 deg horizontal FOV, colour only (reading instruments)
MuJoCo stores the vertical FOV (fovy); `fovy_from_hfov` converts.

Colour noise model (documented, applied by `apply_image_noise`): the render x in [0, 1] is taken as the irradiance
at a base exposure; auto-exposure applies a gain g = clip(min(target_mean / mean(luma), 0.95 / p99(luma)), 1,
max_gain) (mean metering with highlight protection); the output is
g * (x + n) quantised to 8 bits, n ~ N(0, shot^2 * x + read^2) (photon + read noise, so a dim scene that needs gain
gets visibly noisier). Optional Gaussian lens blur before the noise. Limitation: the renderer already quantised x
to 8 bits, so a very dark render (x < 0.05) posterises under high gain.
Depth: z-depth (along the optical axis, metres) from the depth buffer, noise sd = a + b z^2, random dropouts, NaN
outside the valid range.

Frames: `T_world_cam` is the pose of the OPTICAL frame (x right, y down, z forward) in the world.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import mujoco
import numpy as np

from ..common import hinge_range, look_at_R, quat_from_mat

__all__ = ["CameraModel", "WIDE", "NARROW", "fovy_from_hfov", "intrinsics", "RenderPool", "Camera", "Frame",
           "apply_image_noise", "segmentation", "visible_pixels", "add_sensor_head", "add_pan_tilt_rig",
           "aim_rig", "CAM_QUAT_BODY", "camera_pose_optical", "project_points", "default_scene_option"]

# MuJoCo cameras look along their -z with +y up; on a body with x forward, y left, z up:
CAM_QUAT_BODY = quat_from_mat(np.array([[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]))
MJ_TO_OPTICAL = np.diag([1.0, -1.0, -1.0])


def fovy_from_hfov(hfov_deg: float, width: int, height: int) -> float:
    return math.degrees(2.0 * math.atan(math.tan(math.radians(hfov_deg) / 2.0) * height / float(width)))


def intrinsics(fovy_deg: float, width: int, height: int) -> np.ndarray:
    f = (height / 2.0) / math.tan(math.radians(fovy_deg) / 2.0)
    return np.array([[f, 0.0, (width - 1) / 2.0], [0.0, f, (height - 1) / 2.0], [0.0, 0.0, 1.0]])


@dataclass(frozen=True)
class CameraModel:
    name: str
    width: int
    height: int
    hfov_deg: float
    depth: bool = False
    depth_range: tuple = (0.15, 8.0)
    depth_noise: tuple = (0.002, 0.002)      # sd = a + b z^2 (m)
    depth_dropout: float = 0.01
    shot: float = 0.012                      # photon-noise sd at x = 1 (relative units)
    read: float = 0.003                      # read-noise sd
    auto_exposure: bool = True
    target_mean: float = 0.40
    max_gain: float = 12.0
    blur_px: float = 0.0

    @property
    def fovy_deg(self) -> float:
        return fovy_from_hfov(self.hfov_deg, self.width, self.height)

    @property
    def K(self) -> np.ndarray:
        return intrinsics(self.fovy_deg, self.width, self.height)

    @property
    def f_px(self) -> float:
        return float(self.K[0, 0])

    def resized(self, width: int, height: int) -> "CameraModel":
        """Same horizontal FOV at another image size."""
        return replace(self, width=int(width), height=int(height))


WIDE = CameraModel("nav", 640, 480, 90.0, depth=True)
NARROW = CameraModel("inspect", 1280, 960, 18.0, depth=False, blur_px=0.6)


def default_scene_option() -> mujoco.MjvOption:
    """What the devices see: geom groups 0-2, no sites, joints or frames."""
    opt = mujoco.MjvOption()
    opt.sitegroup[:] = 0
    for g in range(3, 6):
        opt.geomgroup[g] = 0
    return opt


class RenderPool:
    """One mujoco.Renderer per image size for one model (create once per process, reuse)."""

    def __init__(self, model):
        self.model = model
        self._r = {}

    def get(self, width: int, height: int) -> mujoco.Renderer:
        key = (int(width), int(height))
        r = self._r.get(key)
        if r is None:
            r = mujoco.Renderer(self.model, height=key[1], width=key[0])
            self._r[key] = r
        return r

    def close(self):
        for r in self._r.values():
            try:
                r.close()
            except Exception:  # noqa: BLE001
                pass
        self._r = {}


@dataclass
class Frame:
    rgb: np.ndarray                      # (H, W, 3) uint8
    depth: np.ndarray | None             # (H, W) float32 metres, NaN invalid
    K: np.ndarray                        # 3x3
    T_world_cam: np.ndarray              # 4x4 optical frame in world
    gain: float = 1.0


def camera_pose_optical(model, data, cam_id: int) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = data.cam_xmat[cam_id].reshape(3, 3) @ MJ_TO_OPTICAL
    T[:3, 3] = data.cam_xpos[cam_id]
    return T


def project_points(K, T_world_cam, pts_world) -> np.ndarray:
    """(N, 3) world points -> (N, 3) of (u, v, z) in pixels / metres."""
    T = np.linalg.inv(T_world_cam)
    P = (T[:3, :3] @ np.asarray(pts_world, float).T + T[:3, 3:4]).T
    uv = (K @ P.T).T
    return np.column_stack([uv[:, 0] / uv[:, 2], uv[:, 1] / uv[:, 2], P[:, 2]])


def apply_image_noise(rgb: np.ndarray, cm: CameraModel, rng: np.random.Generator):
    """(noisy uint8 image, gain) following the module's noise model."""
    x = rgb.astype(np.float32) / 255.0
    if cm.blur_px > 0:
        import cv2
        x = cv2.GaussianBlur(x, (0, 0), cm.blur_px)
    g = 1.0
    if cm.auto_exposure:
        Y = 0.299 * x[..., 0] + 0.587 * x[..., 1] + 0.114 * x[..., 2]
        luma = float(Y.mean())
        hi = float(np.percentile(Y[::4, ::4], 99.0))
        # mean metering with highlight protection (a lamp-lit target in a dark room must not saturate)
        g = float(np.clip(min(cm.target_mean / max(luma, 1e-4), 0.95 / max(hi, 1e-4)), 1.0, cm.max_gain))
    sd = np.sqrt(cm.shot * cm.shot * np.clip(x, 0, 1) + cm.read * cm.read)
    y = g * (x + rng.standard_normal(x.shape).astype(np.float32) * sd)
    return np.clip(np.round(y * 255.0), 0, 255).astype(np.uint8), g


class Camera:
    """A named MuJoCo camera with a CameraModel; `capture(data)` returns a Frame."""

    def __init__(self, model, cam_name: str, cm: CameraModel, pool: RenderPool | None = None,
                 rng: np.random.Generator | None = None, noise: bool = True, scene_option=None):
        self.model = model
        self.cm = cm
        self.cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
        if self.cam_id < 0:
            raise KeyError("no camera named %r" % cam_name)
        if abs(float(model.cam_fovy[self.cam_id]) - cm.fovy_deg) > 1e-3:
            model.cam_fovy[self.cam_id] = cm.fovy_deg
        self.pool = pool or RenderPool(model)
        self.rng = rng or np.random.default_rng(0)
        self.noise = noise
        self.opt = scene_option or default_scene_option()

    @property
    def K(self):
        return self.cm.K

    def render_raw(self, data, depth: bool = False):
        r = self.pool.get(self.cm.width, self.cm.height)
        r.update_scene(data, camera=self.cam_id, scene_option=self.opt)
        rgb = r.render().copy()
        z = None
        if depth:
            r.enable_depth_rendering()
            try:
                z = r.render().astype(np.float32).copy()
            finally:
                r.disable_depth_rendering()
        return rgb, z

    def capture(self, data, depth: bool | None = None) -> Frame:
        want_depth = self.cm.depth if depth is None else depth
        rgb, z = self.render_raw(data, want_depth)
        g = 1.0
        if self.noise:
            rgb, g = apply_image_noise(rgb, self.cm, self.rng)
        if z is not None and self.noise:
            a, b = self.cm.depth_noise
            z = z + self.rng.standard_normal(z.shape).astype(np.float32) * (a + b * z * z)
            lo, hi = self.cm.depth_range
            bad = ~np.isfinite(z) | (z < lo) | (z > hi) | (self.rng.random(z.shape) < self.cm.depth_dropout)
            z[bad] = np.nan
        return Frame(rgb, z, self.cm.K, camera_pose_optical(self.model, data, self.cam_id), g)


def segmentation(pool: RenderPool, data, cam_id: int, width: int, height: int, scene_option=None):
    """(object id, object type) per pixel; background reads (-1, -1)."""
    r = pool.get(width, height)
    r.enable_segmentation_rendering()
    try:
        r.update_scene(data, camera=cam_id, scene_option=scene_option or default_scene_option())
        for f in (mujoco.mjtRndFlag.mjRND_FOG, mujoco.mjtRndFlag.mjRND_HAZE, mujoco.mjtRndFlag.mjRND_SKYBOX):
            r.scene.flags[f] = 0
        out = r.render().copy()
    finally:
        r.disable_segmentation_rendering()
    return out[..., 0], out[..., 1]


def visible_pixels(seg_ids: np.ndarray, seg_types: np.ndarray, geom_ids) -> int:
    """How many pixels of a segmentation image show one of `geom_ids` (the grader's visibility test)."""
    m = (seg_types == int(mujoco.mjtObj.mjOBJ_GEOM)) & np.isin(seg_ids, np.asarray(geom_ids))
    return int(m.sum())


# ----------------------------------------------------------------------------------------------- head and rig
def add_sensor_head(spec, body, prefix: str, cameras=(WIDE, NARROW), offsets=None, lamp: bool = True,
                    lamp_on: bool = False, lidar_offset=None) -> dict:
    """Add cameras (x forward on `body`), a head lamp (spotlight along +x) and optionally a lidar site to `body`.
    Camera names: '<prefix>/<model.name>'. Returns {'cameras': {model.name: camera name}, 'lamp': name or None,
    'lidar_site': name or None}."""
    out = {"cameras": {}, "lamp": None, "lidar_site": None}
    offsets = offsets or {}
    for k, cm in enumerate(cameras):
        off = offsets.get(cm.name, (0.0, 0.03 * (k - (len(cameras) - 1) / 2.0), 0.0))
        nm = "%s/%s" % (prefix, cm.name)
        body.add_camera(name=nm, pos=list(off), quat=list(CAM_QUAT_BODY), fovy=cm.fovy_deg)
        out["cameras"][cm.name] = nm
    if lamp:
        nm = prefix + "/lamp"
        body.add_light(name=nm, type=mujoco.mjtLightType.mjLIGHT_SPOT, pos=[0.0, 0.0, 0.035], dir=[1.0, 0.0, 0.0],
                       cutoff=30.0, exponent=4.0, attenuation=[0.25, 0.0, 0.75], diffuse=[0.95, 0.95, 0.9],
                       ambient=[0.0, 0.0, 0.0], specular=[0.15, 0.15, 0.15], castshadow=False, active=bool(lamp_on))
        out["lamp"] = nm
    if lidar_offset is not None:
        nm = prefix + "/lidar"
        body.add_site(name=nm, pos=list(lidar_offset), size=[0.01, 0.01, 0.01], group=5)
        out["lidar_site"] = nm
    return out


def add_pan_tilt_rig(spec, parent, name: str = "rig", pos=(0.0, 0.0, 0.33), yaw_deg: float = 0.0, cameras=(WIDE, NARROW),
                     mocap: bool = True, lamp: bool = True, lidar_offset=(0.0, 0.0, -0.05), post: bool = True) -> dict:
    """A free-standing pan-tilt sensor head at `pos` (the pan / tilt axes cross at `pos`): base body (mocap by default,
    so it can be moved anywhere), a pan hinge about +z, a tilt hinge about +y (positive tilt pitches the nose DOWN),
    the sensor head on the tilt body. With `post`, a thin post down to the floor is drawn (group 2)."""
    from ..common import quat_axis_angle
    base = parent.add_body(name=name, pos=list(pos), quat=list(quat_axis_angle([0, 0, 1], math.radians(yaw_deg))),
                           mocap=mocap)
    if post:
        base.add_geom(name=name + "/post", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.012, max(0.01, pos[2] / 2 - 0.03), 0],
                      pos=[0, 0, -pos[2] / 2 - 0.03], rgba=[0.2, 0.2, 0.22, 1], contype=0, conaffinity=0, group=2)
    pan = base.add_body(name=name + "/pan", gravcomp=1.0)
    tilt = pan.add_body(name=name + "/tilt", gravcomp=1.0)
    for b in (pan, tilt):
        b.explicitinertial = True
        b.mass = 0.1
        b.inertia = [1e-4, 1e-4, 1e-4]
        b.ipos = [0, 0, 0]
        b.iquat = [1, 0, 0, 0]
    pan.add_joint(name=name + "/pan", type=mujoco.mjtJoint.mjJNT_HINGE, axis=[0, 0, 1], damping=1.0, armature=0.01)
    tilt.add_joint(name=name + "/tilt", type=mujoco.mjtJoint.mjJNT_HINGE, axis=[0, 1, 0], damping=1.0, armature=0.01,
                   range=hinge_range(spec, -80.0, 80.0), limited=mujoco.mjtLimited.mjLIMITED_TRUE)
    tilt.add_geom(name=name + "/head", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.03, 0.06, 0.025], pos=[-0.035, 0, 0],
                  rgba=[0.15, 0.15, 0.17, 1], contype=0, conaffinity=0, group=2)
    sensors = add_sensor_head(spec, tilt, name, cameras=cameras, lamp=lamp, lidar_offset=None)
    lid = None
    if lidar_offset is not None:
        lid = name + "/lidar"
        base.add_site(name=lid, pos=list(lidar_offset), size=[0.01, 0.01, 0.01], group=5)
    return {"base": name, "pan_joint": name + "/pan", "tilt_joint": name + "/tilt", "tilt_body": name + "/tilt",
            "cameras": sensors["cameras"], "lamp": sensors["lamp"], "lidar_site": lid, "mocap": mocap}


def aim_rig(model, data, rig: dict, target, eye=None, yaw_deg: float | None = None, forward: bool = True):
    """Point the rig's head at `target` (world point). With `eye` (world point of the pan/tilt centre) the mocap base
    is moved there first (heading `yaw_deg`, default: facing the target). Returns (pan, tilt) in radians."""
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, rig["base"])
    if eye is not None:
        mid = int(model.body_mocapid[bid])
        if mid < 0:
            raise ValueError("rig base is not a mocap body")
        eye = np.asarray(eye, float)
        d = np.asarray(target, float) - eye
        yaw = math.atan2(d[1], d[0]) if yaw_deg is None else math.radians(yaw_deg)
        data.mocap_pos[mid] = eye
        data.mocap_quat[mid] = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
        mujoco.mj_kinematics(model, data)
    p = data.xpos[bid]
    R = data.xmat[bid].reshape(3, 3)
    d = R.T @ (np.asarray(target, float) - p)
    pan = math.atan2(d[1], d[0])
    tilt = -math.atan2(d[2], math.hypot(d[0], d[1]))
    for jn, v in ((rig["pan_joint"], pan), (rig["tilt_joint"], tilt)):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn)
        data.qpos[model.jnt_qposadr[jid]] = v
        data.qvel[model.jnt_dofadr[jid]] = 0.0
    if forward:
        mujoco.mj_kinematics(model, data)
        mujoco.mj_camlight(model, data)
    return pan, tilt


def look_at_quat(eye, target):
    """Quaternion of a body frame at `eye` with +x towards `target` and z up."""
    return quat_from_mat(look_at_R(np.asarray(target, float) - np.asarray(eye, float)))
