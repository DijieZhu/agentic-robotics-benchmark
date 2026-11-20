"""Robot camera layer (optional; the spacekit core never imports it): colour images of the robot's cameras with
MuJoCo's classic renderer (OSMesa on hosts without a GPU) at the configured intrinsics, and depth images by ray
casting for the depth cameras.

Colour: pinhole cameras with the configured focal lengths and principal point (MuJoCo intrinsic camera:
``cam_sensorsize`` = image size in pixels, ``cam_intrinsic`` = (fx, fy, W/2 - cx, H/2 - cy) in pixels; validated
by projecting known points).  The two wide cameras are fisheye lenses
(FOV model, w ~ 1 rad); the pinhole approximation keeps the centre focal length, so it sees less than the lens
(horizontal 93 deg instead of ~116 deg at 1280 px).  ``fisheye_from_pinhole`` builds the FOV-model image from a
wider pinhole render by remapping (one larger render + a remap).
Depth: ``spacekit.sensors.DepthCamera`` rays from the camera pose (z-depth, NaN outside 0.1-4 m).
Image sizes: published pixel counts (wide 1280 x 960, depth 224 x 171); the imaging camera's intrinsics are for a
1366 x 1024 image [G].
"""
from __future__ import annotations

import math

import mujoco
import numpy as np

from .sensors import DepthCamera, camera_model, distort_fov

__all__ = ["CAMERA_ROLES", "IMAGE_SIZE", "set_intrinsics", "ViewportRenderer", "RobotCameras", "fisheye_maps",
           "fisheye_from_pinhole", "fov_model_angles", "LAMPS", "LAMP_DEFAULTS", "add_robot_lamps", "set_near_clip",
           "exposure_gain"]

# role -> camera name in the reference build (neutral names) and the depth flag
CAMERA_ROLES = {"forward_wide": ("cam_fwd_wide", False), "aft_wide": ("cam_aft_wide", False),
                "forward_imaging": ("cam_fwd_img", False), "forward_depth": ("cam_fwd_depth", True),
                "aft_depth": ("cam_aft_depth", True)}
IMAGE_SIZE = {"forward_wide": (1280, 960), "aft_wide": (1280, 960), "forward_imaging": (1366, 1024),
              "forward_depth": (224, 171), "aft_depth": (224, 171)}

# v0.4: the robot's two lamps (positions / axes in the robot body frame: the reference build's lamp windows) and the
# spot-light recipe for the classic renderer.  216 lm in a 45 deg beam = 452 cd [R dev.lamp]; the classic renderer
# has no photometric units, so diffuse / attenuation are a calibration [G]:
# inverse-square fall-off, equal to a typical cabin light's contribution (0.5) at about 1.4 m on axis.
LAMPS = dict(front=dict(pos=(0.1345, 0.0, 0.0369), dir=(1.0, 0.0, 0.0)),
             aft=dict(pos=(-0.1345, -0.054, 0.0303), dir=(-1.0, 0.0, 0.0)))
LAMP_DEFAULTS = dict(intensity_cd=452.0, cutoff_deg=30.0, exponent=4.5, diffuse=1.0, attenuation=(0.0, 0.0, 1.0),
                     castshadow=False)


def add_robot_lamps(spec, body: str = "core", prefix: str = "", lamps=None, active: bool = False, **params) -> list:
    """Add the robot's lamps as spot lights to an MjSpec before compiling (``{prefix}lamp_front``, ``..._aft``; off by
    default -- switch with ``RobotCameras.set_lamp`` or ``model.light_active``).  Spot recipe ``LAMP_DEFAULTS``
    (cutoff 30 deg, cos^4.5 taper: full in the 22.5 deg half beam); the photometric intensity (452 cd) is stored in
    the light's ``intensity`` for renderers that use it.  Returns the light names."""
    P = dict(LAMP_DEFAULTS, **params)
    b = spec.body(prefix + body)
    if b is None:
        raise ValueError(f"body {prefix + body} not in the spec")
    names = []
    for k, L in (lamps or LAMPS).items():
        name = f"{prefix}lamp_{k}"
        lt = b.add_light(name=name, pos=list(L["pos"]), dir=list(L["dir"]))
        lt.type = mujoco.mjtLightType.mjLIGHT_SPOT
        lt.cutoff = float(P["cutoff_deg"])
        lt.exponent = float(P["exponent"])
        lt.diffuse = [float(P["diffuse"])] * 3
        lt.ambient = [0.0, 0.0, 0.0]
        lt.specular = [0.1, 0.1, 0.1]
        lt.attenuation = list(P["attenuation"])
        lt.castshadow = bool(P["castshadow"])
        lt.intensity = float(P["intensity_cd"])
        lt.active = bool(active)
        names.append(name)
    return names


def set_near_clip(model, near_m: float) -> float:
    """Near clipping distance of the renderer [m]: MuJoCo uses vis.map.znear * stat.extent for every camera of the
    model (a model-wide setting).  Returns the previous distance.  Robot cameras need about 1-2 cm (an item in the
    gripper is 0.1-0.2 m from the lens; a station scene's default can be 0.2 m and more)."""
    old = float(model.vis.map.znear) * float(model.stat.extent)
    model.vis.map.znear = float(near_m) / float(model.stat.extent)
    return old


def exposure_gain(img: np.ndarray, gain: float, target: float = 0.18, gain_range=(1.0 / 16.0, 8.0), sigma: float = 0.25) -> float:
    """Next gain of a centre-weighted auto exposure: the image's linear luminance averaged with a Gaussian weight
    (sigma as a fraction of the image size) brought to ``target`` (mid grey), clipped to ``gain_range``."""
    x = img.astype(float) / 255.0
    lin = np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)
    Y = lin @ np.array([0.2126, 0.7152, 0.0722])
    H, W = Y.shape
    v, u = np.mgrid[0:H, 0:W]
    w = np.exp(-0.5 * (((u - (W - 1) / 2) / (sigma * W)) ** 2 + ((v - (H - 1) / 2) / (sigma * H)) ** 2))
    mean = float(np.sum(w * Y) / np.sum(w))
    g = gain * target / max(mean, 1e-6)
    return float(min(max(g, gain_range[0]), gain_range[1]))


def set_intrinsics(model, cam_id: int, K, size) -> None:
    """Make a model camera a pinhole with focal lengths / principal point of K at image size (W, H) [px]."""
    W, H = size
    model.cam_sensorsize[cam_id] = (float(W), float(H))
    model.cam_resolution[cam_id] = (int(W), int(H))
    # MuJoCo's principal-point offset is measured from the principal point to the image centre (sign verified by
    # projecting a marker)
    model.cam_intrinsic[cam_id] = (float(K[0][0]), float(K[1][1]), float(W / 2.0 - K[0][2]), float(H / 2.0 - K[1][2]))


def fov_model_angles(K, size, w: float):
    """(horizontal, vertical) full field of view [deg] of an FOV-model fisheye at its image edges."""
    W, H = size
    out = []
    for half_px, f, c in ((W / 2.0, K[0][0], K[0][2]), (H / 2.0, K[1][1], K[1][2])):
        rd = half_px / f
        ru = math.tan(rd * w) / (2.0 * math.tan(w / 2.0))
        out.append(2.0 * math.degrees(math.atan(ru)))
    return tuple(out)


class ViewportRenderer:
    """One offscreen OpenGL context for images of any size up to ``max_size`` (CPU OSMesa hosts: one renderer per
    process).  ``render(data, camera_id, (W, H), scene_option)`` draws a model camera into the lower-left W x H
    viewport of the offscreen buffer and reads it back (the same calls as ``mujoco.Renderer``: mjv_updateScene,
    mjr_render, mjr_readPixels).  The model's offscreen buffer (vis.global_.offwidth / offheight) is raised to
    max_size before the context is made.  Intrinsic cameras must be drawn at their sensor size (the frustum comes
    from cam_intrinsic / cam_sensorsize, not from the viewport)."""

    def __init__(self, model, max_size=(1366, 1024), max_geom: int = 10000):
        self.m = model
        model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), int(max_size[0]))
        model.vis.global_.offheight = max(int(model.vis.global_.offheight), int(max_size[1]))
        self.max_size = (int(model.vis.global_.offwidth), int(model.vis.global_.offheight))
        self._gl = mujoco.GLContext(*self.max_size) if getattr(mujoco, "GLContext", None) is not None else None
        if self._gl is not None:
            self._gl.make_current()
        self._con = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150.value)
        mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN.value, self._con)
        self._scn = mujoco.MjvScene(model, maxgeom=max_geom)
        self._cam = mujoco.MjvCamera()
        self._pert = mujoco.MjvPerturb()
        self._opt = mujoco.MjvOption()

    def render(self, data, camera_id, size, scene_option=None) -> np.ndarray:
        """(H, W, 3) uint8 image of model camera ``camera_id`` (or of an ``mujoco.MjvCamera``) at size (W, H)."""
        W, H = int(size[0]), int(size[1])
        if W > self.max_size[0] or H > self.max_size[1]:
            raise ValueError(f"image {W} x {H} larger than the offscreen buffer {self.max_size}")
        if self._gl is not None:
            self._gl.make_current()
        if isinstance(camera_id, mujoco.MjvCamera):
            cam = camera_id
        else:
            cam = self._cam
            cam.type = mujoco.mjtCamera.mjCAMERA_FIXED.value
            cam.fixedcamid = int(camera_id)
        mujoco.mjv_updateScene(self.m, data, scene_option if scene_option is not None else self._opt, self._pert,
                               cam, mujoco.mjtCatBit.mjCAT_ALL.value, self._scn)
        vp = mujoco.MjrRect(0, 0, W, H)
        mujoco.mjr_render(vp, self._scn, self._con)
        out = np.empty((H, W, 3), np.uint8)
        mujoco.mjr_readPixels(out, None, vp, self._con)
        return np.flipud(out).copy()

    def close(self) -> None:
        if self._con is not None:
            self._con.free()
            self._con = None
        if self._gl is not None:
            self._gl.free()
            self._gl = None


class RobotCameras:
    """Colour and depth images of the robot's cameras.

    model: compiled scene (the reference build's camera names); roles: subset of CAMERA_ROLES; scale: image size
    factor (e.g. 0.25 for quick strips); exclude_body: the robot core: the depth rays ignore it and every body
    rigidly attached to it (same weld body: modules, bumpers, the forward sensor module that holds the forward
    depth camera on the reference build v1), but still see the moving parts (arm, gripper); on a model that has the
    depth-camera window (reference build v1.2, ``depth_window_in_model``) only exclude_body itself;
    depth_groups: geom groups the depth rays see (default the collision / part / handrail groups 1-3, not the
    high-polygon visual look in group 0: ray tests against 10^5 visual triangles cost seconds per frame);
    renderer: a ViewportRenderer to share (e.g. with a third-person view); default: one is made at the largest
    colour image size (one OpenGL context in total).  ``rgb(data, role)``, ``depth_image(model, data, role)``;
    ``close()`` (closes only a renderer it made).
    v0.4: near_m (default 0.015 m; None keeps the model's) sets the near clipping distance (``set_near_clip``, model
    wide); exposure: None (v0.3), a fixed gain, or 'auto' -- a centre-weighted auto exposure (``exposure_gain``,
    1/16-8x, the gain of a frame comes from the previous frame of that camera, as a camera's AE loop); the gain
    scales every light, the headlight and the material emission during the render (as a longer exposure would) and
    is in ``last_exposure``; ``set_lamp(name, on)`` switches a lamp added with ``add_robot_lamps``.
    """

    def __init__(self, model, roles=("forward_wide", "aft_wide", "forward_depth", "aft_depth"), scale: float = 1.0,
                 exclude_body: int = -1, scene_option=None, depth_stride: int = 1, depth_groups=(1, 2, 3),
                 renderer: "ViewportRenderer | None" = None, near_m: float | None = 0.015, exposure=None,
                 exposure_range=(1.0 / 16.0, 8.0), exposure_target: float = 0.18):
        self.m = model
        if near_m is not None:
            set_near_clip(model, near_m)
        if exposure is not None and exposure != "auto" and not isinstance(exposure, (int, float)):
            raise ValueError("exposure must be None, a gain or 'auto'")
        self.exposure, self.exposure_range, self.exposure_target = exposure, tuple(exposure_range), float(exposure_target)
        self._gain = {}
        self.last_exposure = {}
        self.roles = list(roles)
        self.scale = float(scale)
        self.opt = scene_option
        self.cams = {}
        self.depth = {}
        self._vr = renderer
        self._own = renderer is None
        rigid = None
        if exclude_body >= 0:
            rigid = [b for b in range(model.nbody) if model.body_weldid[b] == model.body_weldid[exclude_body]]
        self.rigid_bodies = rigid
        # reference build v1.2: the forward depth camera looks through a window of its module's proxy (the
        # model's 'refbuild' text): the rays exclude only ``exclude_body``, as before v0.3; older models keep the
        # exclusion of the whole rigid structure
        from .refbuild import has_correction
        self.depth_window_in_model = has_correction(model, "fwd_depth_window")
        ray_excl = None if self.depth_window_in_model else rigid
        for r in self.roles:
            name, is_depth = CAMERA_ROLES[r]
            cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, name)
            if cid < 0:
                raise ValueError(f"camera {name} not in the model")
            cm = camera_model(r)
            W, H = IMAGE_SIZE[r]
            Ws, Hs = max(8, int(round(W * self.scale))), max(8, int(round(H * self.scale)))
            K = np.diag([Ws / W, Hs / H, 1.0]) @ np.asarray(cm["K"], float)    # configured K in the published size
            self.cams[r] = dict(id=cid, name=name, K=K, size=(Ws, Hs), model=cm)
            if is_depth:
                gg = None
                if depth_groups is not None:
                    gg = np.zeros(6, np.uint8)
                    gg[list(depth_groups)] = 1
                self.depth[r] = DepthCamera(model, name, Ws, Hs, K[0, 0], K[1, 1], K[0, 2], K[1, 2], kind="camera",
                                            stride=depth_stride, exclude_body=exclude_body, geomgroup=gg,
                                            exclude_bodies=ray_excl)
            else:
                set_intrinsics(model, cid, K, (Ws, Hs))

    def _renderer(self):
        if self._vr is None:
            sizes = [c["size"] for r, c in self.cams.items() if r not in self.depth] or [(8, 8)]
            self._vr = ViewportRenderer(self.m, (max(w for w, _ in sizes), max(h for _, h in sizes)))
        return self._vr

    def rgb(self, data, role: str) -> np.ndarray:
        """(H, W, 3) uint8 pinhole image of a colour camera (with the exposure setting)."""
        c = self.cams[role]
        if self.exposure is None:
            return self._renderer().render(data, c["id"], c["size"], self.opt)
        g = float(self.exposure) if self.exposure != "auto" else self._gain.get(role, 1.0)
        img = self._render_with_gain(data, c, g)
        self.last_exposure[role] = g
        if self.exposure == "auto":
            self._gain[role] = exposure_gain(img, g, self.exposure_target, self.exposure_range)
        return img

    def _render_with_gain(self, data, c, g: float) -> np.ndarray:
        m = self.m
        saved = (m.light_diffuse.copy(), m.light_ambient.copy(), m.light_specular.copy(), m.mat_emission.copy(),
                 np.array(m.vis.headlight.ambient), np.array(m.vis.headlight.diffuse), np.array(m.vis.headlight.specular))
        try:
            m.light_diffuse[:] = saved[0] * g
            m.light_ambient[:] = saved[1] * g
            m.light_specular[:] = saved[2] * g
            m.mat_emission[:] = saved[3] * g
            m.vis.headlight.ambient = saved[4] * g
            m.vis.headlight.diffuse = saved[5] * g
            m.vis.headlight.specular = saved[6] * g
            return self._renderer().render(data, c["id"], c["size"], self.opt)
        finally:
            m.light_diffuse[:], m.light_ambient[:], m.light_specular[:], m.mat_emission[:] = saved[:4]
            m.vis.headlight.ambient, m.vis.headlight.diffuse, m.vis.headlight.specular = saved[4], saved[5], saved[6]

    def set_lamp(self, name: str, on: bool) -> None:
        """Switch a lamp light (``add_robot_lamps``) of the model on or off."""
        lid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_LIGHT, name)
        if lid < 0:
            raise ValueError(f"light {name} not in the model")
        self.m.light_active[lid] = bool(on)

    def depth_image(self, model, data, role: str) -> np.ndarray:
        """(H, W) z-depth [m] of a depth camera (ray cast)."""
        return self.depth[role].read(model, data)

    def close(self) -> None:
        if self._own and self._vr is not None:
            self._vr.close()
        self._vr = None


def fisheye_maps(K_wide, K_fish, size_fish, w: float):
    """(map_x, map_y) float32 sampling maps from an FOV-model fisheye image (size_fish, K_fish, w) into a wider
    pinhole image with intrinsics K_wide (compute once per camera)."""
    W, H = size_fish
    u, v = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
    xd = np.stack([(u - K_fish[0][2]) / K_fish[0][0], (v - K_fish[1][2]) / K_fish[1][1]], -1).reshape(-1, 2)
    rd = np.linalg.norm(xd, axis=1)
    ru = np.tan(np.minimum(rd * w, 1.55)) / (2.0 * math.tan(w / 2.0))
    s = np.where(rd > 1e-12, ru / np.maximum(rd, 1e-12), 1.0)
    xu = xd * s[:, None]
    px = K_wide[0][0] * xu[:, 0] + K_wide[0][2] - 0.5
    py = K_wide[1][1] * xu[:, 1] + K_wide[1][2] - 0.5
    return px.reshape(H, W).astype(np.float32), py.reshape(H, W).astype(np.float32)


def fisheye_from_pinhole(img_wide: np.ndarray, K_wide, K_fish, size_fish, w: float, maps=None):
    """FOV-model fisheye image (size_fish) from a wider pinhole image with intrinsics K_wide: for every output pixel,
    undistort with the FOV model, project with the wide pinhole, sample bilinearly (OpenCV remap when available,
    else numpy).  Pass ``maps`` from fisheye_maps to skip their computation."""
    px, py = maps if maps is not None else fisheye_maps(K_wide, K_fish, size_fish, w)
    try:
        import cv2
        return cv2.remap(img_wide, px, py, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    except ImportError:
        pass
    W, H = size_fish
    px, py = px.ravel(), py.ravel()
    h, wdt = img_wide.shape[:2]
    x0 = np.clip(np.floor(px).astype(int), 0, wdt - 2)
    y0 = np.clip(np.floor(py).astype(int), 0, h - 2)
    fx, fy = np.clip(px - x0, 0, 1), np.clip(py - y0, 0, 1)
    img = img_wide.astype(np.float32)
    out = ((1 - fx) * (1 - fy))[:, None] * img[y0, x0] + (fx * (1 - fy))[:, None] * img[y0, x0 + 1] \
        + ((1 - fx) * fy)[:, None] * img[y0 + 1, x0] + (fx * fy)[:, None] * img[y0 + 1, x0 + 1]
    inside = (px >= 0) & (px <= wdt - 1) & (py >= 0) & (py <= h - 1)
    out[~inside] = 0
    return out.reshape(H, W, -1).astype(np.uint8)
