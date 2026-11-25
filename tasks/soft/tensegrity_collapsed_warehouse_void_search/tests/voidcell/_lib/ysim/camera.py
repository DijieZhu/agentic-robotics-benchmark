"""Fixed overhead RGB-D camera as in the authors' experiments, behind a small renderer-agnostic interface (stage 2).

In the authors' experiments the robot carries no camera: a fixed overhead RGB-D camera (RealSense L515 in the lab,
640 x 480, 7 Hz) looks down on the arena and an off-board PC estimates the pose (RA-L sections IV and VII). The
authors' calibration of that camera is public (perception repository, ros_branch_mesh/data_cfg.json, read as data):
intrinsics fx 607.686, fy 607.987, cx 326.550, cy 250.662; camera-to-world extrinsic with the camera 2.342 m above
their floor, 2.6 deg off nadir; depth_scale 4000 counts per metre (0.25 mm).

    spec = CameraSpec.authors_lab()                      # the authors' calibrated camera, their extrinsic
    spec = CameraSpec.authors_lab(target_xy=(x, y))      # same camera, moved so its optical axis meets (x, y, 0)
    spec = CameraSpec.overhead((x, y), height, yaw_deg)  # nadir camera, image +u = east, image up = north (yaw 0)
    spec = CameraSpec.tilted(pos, tilt_deg, yaw_deg)     # the site family's search camera (tilted toward north)
    cam = OverheadCamera(sim, spec, backend="classic")   # exposures at rate_hz via sim.tick_hooks (qpos snapshots)
    f = cam.latest()                                     # Frame(t, rgb uint8 HxWx3, depth float32 HxW)
    f.depth_u16(spec)                                    # uint16 counts at the authors' depth scale
    spec.project(points) / spec.unproject(depth)         # the pinhole model (reference implementation)

Conventions (those of the benchmark's arm render layer, adopted here): world z up, metres;
camera frame = OpenCV (+x right, +y down, +z forward); world_T_cam maps camera-frame points to world; pixel centres at
integer coordinates; depth = z along the optical axis in metres, 0.0 where nothing was hit; no distortion, noise or
blur (the sensor model is a later stage's choice: there is no public noise data of the authors' unit).

Renderer-agnostic interface: a backend is any object with `name`, `version()` and `open(model, width, height,
groups, tendons) -> session`; a session has `render(model, data, spec) -> (rgb, depth)` and `close()`.
`register_backend(name, factory)` adds one (e.g. a Filament + baked-scene backend later); `get_backend("classic")`
is the MuJoCo classic OpenGL renderer (needs an OSMesa or EGL library; MUJOCO_GL selects it).
The GL-camera frustum and the reversed-Z depth arithmetic are copied from the arm render layer
(its MuJoCo backend: set_scene_camera, MujocoSession._metric_depth;
not imported).
"""
import math
import os
import time
from dataclasses import dataclass, field

os.environ.setdefault("MESA_SHADER_CACHE_DISABLE", "true")   # no Mesa shader cache under ~/.cache (ysim/render.py)

import numpy as np

AUTHORS_K = {"fx": 607.6858520507812, "fy": 607.9865112304688, "cx": 326.55035400390625, "cy": 250.6623992919922,
             "width": 640, "height": 480}
AUTHORS_EXTR = [[0.9997460684445527, -0.0033635460398849792, -0.02228194757414005, 1.213693411645255],
                [-0.004228487429239647, -0.9992347313320431, -0.03888536477699146, 0.8268696689608708],
                [-0.02213410318309538, 0.03896970949104574, -0.9989952168146069, 2.3420875915712847],
                [0.0, 0.0, 0.0, 1.0]]
AUTHORS_SRC = ("github.com/shiyoung77/tensegrity_perception, branch ros, data_cfg.json: cam_intr, cam_extr "
               "(camera-to-world, camera 2.342 m above the floor), im_w 640, im_h 480, depth_scale 4000, depth_trunc 3")
RATE_SRC = "RA-L (Johnson et al. 2026, arXiv:2511.05798) section IV: pose estimates at 7 Hz from the overhead RGB-D camera"


@dataclass
class CameraSpec:
    name: str = "overhead"
    width: int = 640
    height: int = 480
    fx: float = AUTHORS_K["fx"]
    fy: float = AUTHORS_K["fy"]
    cx: float = AUTHORS_K["cx"]
    cy: float = AUTHORS_K["cy"]
    world_T_cam: list = field(default_factory=lambda: [list(r) for r in AUTHORS_EXTR])
    rate_hz: float = 7.0
    depth_scale: float = 4000.0            # counts per metre of the uint16 depth (authors' depth_scale)
    depth_trunc_m: float = None            # authors' pipeline truncates at 3 m (data_cfg depth_trunc); None = no clip
    provenance: dict = field(default_factory=lambda: {"intrinsics": AUTHORS_SRC, "rate": RATE_SRC})

    # ---- geometry
    @property
    def T(self):
        return np.asarray(self.world_T_cam, float)

    @property
    def R(self):
        return self.T[:3, :3]

    @property
    def position(self):
        return self.T[:3, 3]

    @property
    def K(self):
        return np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1.0]])

    def project(self, pts):
        """World points (n x 3) -> (u, v, z_cam) arrays (pixels, metres along the optical axis)."""
        pc = (np.atleast_2d(np.asarray(pts, float)) - self.position) @ self.R
        z = pc[:, 2]
        return self.fx * pc[:, 0] / z + self.cx, self.fy * pc[:, 1] / z + self.cy, z

    def unproject(self, depth):
        """Depth image (H x W, metres along the axis, 0 = invalid) -> world points (H x W x 3, NaN where invalid)."""
        H, W = depth.shape
        u, v = np.meshgrid(np.arange(W, dtype=float), np.arange(H, dtype=float))
        z = depth.astype(float)
        pc = np.stack([(u - self.cx) / self.fx * z, (v - self.cy) / self.fy * z, z], axis=-1)
        pw = pc @ self.R.T + self.position
        pw[z <= 0] = np.nan
        return pw

    def footprint(self, z=0.0):
        """World (x, y) of the four image corners on the plane z (for plots and coverage checks)."""
        out = []
        for u, v in ((-0.5, -0.5), (self.width - 0.5, -0.5), (self.width - 0.5, self.height - 0.5), (-0.5, self.height - 0.5)):
            d = self.R @ np.array([(u - self.cx) / self.fx, (v - self.cy) / self.fy, 1.0])
            t = (z - self.position[2]) / d[2]
            out.append((self.position + t * d)[:2])
        return np.array(out)

    def to_dict(self):
        d = dict(self.__dict__)
        d["world_T_cam"] = [list(map(float, r)) for r in self.T]
        return d

    # ---- constructors
    @staticmethod
    def _from_R(R, pos, **kw):
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = R, pos
        return CameraSpec(world_T_cam=T.tolist(), **kw)

    @classmethod
    def authors_lab(cls, target_xy=None, **kw):
        """The authors' calibrated camera. target_xy: translate it so that its optical axis meets (x, y, 0); the
        rotation and the 2.342 m height stay the authors'."""
        T = np.asarray(AUTHORS_EXTR, float)
        if target_xy is not None:
            z_ax = T[:3, 2]
            t = -T[2, 3] / z_ax[2]
            hit = T[:3, 3] + t * z_ax
            T = T.copy()
            T[0, 3] += target_xy[0] - hit[0]
            T[1, 3] += target_xy[1] - hit[1]
        return cls(name=kw.pop("name", "overhead"), world_T_cam=T.tolist(), **kw)

    @classmethod
    def overhead(cls, target_xy, height=2.342, yaw_deg=0.0, **kw):
        """Nadir camera `height` above (x, y): image +u = east and image up = north at yaw 0 (the authors' camera
        is 2.6 deg off this; ASSUMPTION when used instead of authors_lab)."""
        c, s = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))
        R = np.stack([[c, s, 0.0], [s, -c, 0.0], [0.0, 0.0, -1.0]], axis=1)
        return cls._from_R(R, [target_xy[0], target_xy[1], height], name=kw.pop("name", "overhead"), **kw)

    @classmethod
    def tilted(cls, pos, tilt_deg, yaw_deg=0.0, **kw):
        """The site family's search camera: optical axis
        tilted tilt_deg from nadir toward north, turned yaw_deg (east of north); image +u = east at yaw 0."""
        t, y = math.radians(tilt_deg), math.radians(yaw_deg)
        fwd = np.array([math.sin(t) * math.sin(y), math.sin(t) * math.cos(y), -math.cos(t)])
        right = np.array([math.cos(y), -math.sin(y), 0.0])
        down = np.cross(fwd, right)
        return cls._from_R(np.stack([right, down, fwd], axis=1), pos, name=kw.pop("name", "search"), **kw)

    @classmethod
    def from_concept07(cls, site_meta):
        cm = site_meta["camera_model"]
        cam = site_meta["camera"]
        return cls.tilted(cam["pos"], cam["tilt_deg"], cam["yaw_deg"], width=cm["width"], height=cm["height"], fx=cm["fx"],
                          fy=cm["fy"], cx=cm["cx"], cy=cm["cy"], rate_hz=cm["rate_hz"], depth_scale=1.0 / cm["depth_unit_m"],
                          provenance={"pose": "the site record's camera pose", "intrinsics": AUTHORS_SRC, "rate": RATE_SRC})


@dataclass
class Frame:
    t: float                 # exposure time (sim seconds; the control tick at or after the nominal exposure)
    index: int               # exposure number k (nominal time k / rate_hz)
    rgb: np.ndarray
    depth: np.ndarray        # float32 metres along the optical axis, 0 = nothing hit
    camera: str = "overhead"

    def depth_u16(self, spec):
        """uint16 depth counts at spec.depth_scale (authors: 4000 / m), 0 = invalid; clipped at depth_trunc_m."""
        d = self.depth.astype(np.float64)
        if spec.depth_trunc_m:
            d = np.where(d > spec.depth_trunc_m, 0.0, d)
        return np.clip(np.round(d * spec.depth_scale), 0, 65535).astype(np.uint16)


# ----------------------------------------------------------------------------- backends
_BACKENDS = {}


def register_backend(name, factory):
    _BACKENDS[name] = factory


def get_backend(name="classic", **kw):
    if name not in _BACKENDS:
        raise KeyError("render backend %r not registered (have %s)" % (name, sorted(_BACKENDS)))
    return _BACKENDS[name](**kw)


class ClassicBackend:
    """MuJoCo's classic OpenGL renderer (EGL / OSMesa via MUJOCO_GL). Default: RGB from the model's multisampled
    (antialiased) context, depth from a single-sample context, i.e. two passes. Measured on this host: the depth
    read back from the 8x multisampled buffer is one sample's, not the pixel centre's, which puts a sloped surface off
    by up to 1.07 mm (tilted views); the single-sample depth matches the ray-plane depth to 2 um. msaa_rgb=False:
    one single-sample pass for both (aliased RGB, about half the cost)."""
    name = "classic"

    def __init__(self, msaa_rgb=True):
        self.msaa_rgb = msaa_rgb

    def version(self):
        import mujoco
        return "mujoco-classic/%s msaa_rgb=%s" % (mujoco.__version__, self.msaa_rgb)

    def open(self, model, width, height, groups=(0, 1, 2), tendons=False):
        return ClassicSession(model, width, height, groups, tendons, self.msaa_rgb)


class ClassicSession:
    def __init__(self, model, width, height, groups, tendons, msaa_rgb=True):
        import mujoco
        self.mj = mujoco
        model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), int(width))
        model.vis.global_.offheight = max(int(model.vis.global_.offheight), int(height))
        self.r = mujoco.Renderer(model, height=int(height), width=int(width)) if msaa_rgb else None
        ns = int(model.vis.quality.offsamples)
        model.vis.quality.offsamples = 0                 # read at context creation: a single-sample framebuffer
        try:
            self.rd = mujoco.Renderer(model, height=int(height), width=int(width))
        finally:
            model.vis.quality.offsamples = ns
        self.opt = mujoco.MjvOption()
        self.opt.geomgroup[:] = 0
        for g in groups:
            self.opt.geomgroup[g] = 1
        self.opt.sitegroup[:] = 0
        self.opt.flags[mujoco.mjtVisFlag.mjVIS_TENDON] = 1 if tendons else 0
        self.free = mujoco.MjvCamera()
        self.w, self.h = int(width), int(height)
        self.near, self.far = 0.05, 50.0          # clip planes [m] of the GL frustum and of the depth conversion

    def _set_camera(self, scn, spec):
        """Point both GL cameras of the scene through spec (OpenCV pinhole, off-axis frustum on the pixel grid) and
        move the headlight with it (copied from the arm render layer set_scene_camera). Own clip planes (not
        znear * extent, which is ~1.5 mm for the robot-only model); the same near / far enter the depth conversion."""
        near, far = self.near, self.far
        R = spec.R
        eye, fwd, up = spec.position, R[:, 2], -R[:, 1]
        W, H = spec.width, spec.height
        left, right = (-0.5 - spec.cx) / spec.fx * near, (W - 0.5 - spec.cx) / spec.fx * near
        top, bottom = (spec.cy + 0.5) / spec.fy * near, -(H - 0.5 - spec.cy) / spec.fy * near
        for c in (scn.camera[0], scn.camera[1]):
            c.pos[:] = eye
            c.forward[:] = fwd
            c.up[:] = up
            c.frustum_near, c.frustum_far = near, far
            c.frustum_center = 0.5 * (left + right)
            c.frustum_width = 0.5 * (right - left)
            c.frustum_bottom, c.frustum_top = bottom, top
            c.orthographic = 0
        for i in range(scn.nlight):
            if scn.lights[i].headlight:
                scn.lights[i].pos[:] = eye
                scn.lights[i].dir[:] = fwd
        return near, far

    def _pass(self, r, data, spec, want_rgb, want_depth):
        mj = self.mj
        r.update_scene(data, camera=self.free, scene_option=self.opt)
        near, far = self._set_camera(r.scene, spec)
        if r._gl_context:
            r._gl_context.make_current()
        rgb = np.empty((self.h, self.w, 3), np.uint8) if want_rgb else None
        z = np.empty((self.h, self.w), np.float32) if want_depth else None
        mj.mjr_render(r._rect, r.scene, r._mjr_context)
        mj.mjr_readPixels(rgb, z, r._rect, r._mjr_context)
        depth = None
        if want_depth:
            z = np.flipud(z)
            # reversed-Z buffer -> metres along the optical axis (the arithmetic of mujoco.Renderer's depth mode)
            zfar, znear = np.float32(far), np.float32(near)
            c = -(zfar + znear) / (zfar - znear)
            d = -(np.float32(2) * zfar * znear) / (zfar - znear)
            c = np.float32(-0.5) * c - np.float32(0.5)
            d = np.float32(-0.5) * d
            depth = (d / (z.astype(np.float64) + c)).astype(np.float32)
            depth[depth >= np.float32(0.999) * zfar] = 0.0
        return (np.flipud(rgb).copy() if want_rgb else None), depth

    def render(self, model, data, spec, depth=True):
        assert (spec.width, spec.height) == (self.w, self.h)
        if self.r is None:
            return self._pass(self.rd, data, spec, True, depth)
        rgb, _ = self._pass(self.r, data, spec, True, False)
        dep = self._pass(self.rd, data, spec, False, True)[1] if depth else None
        return rgb, dep

    def close(self):
        for r in (self.r, self.rd):
            if r is not None:
                r.close()


register_backend("classic", lambda **kw: ClassicBackend(**kw))


# ----------------------------------------------------------------------------- the camera on a simulator
class OverheadCamera:
    """A fixed RGB-D camera on a ysim Simulator. Exposures happen at k / rate_hz (sim time): at the first control
    tick at or after each, the joint positions are copied (no rendering, physics untouched). latest() renders the
    newest exposure on demand (cached), now() renders the current state. Rendering uses its own MjData."""

    def __init__(self, sim, spec, backend="classic", groups=(0, 1, 2), tendons=False):
        import mujoco
        self.mj = mujoco
        self.sim, self.spec = sim, spec
        self.backend = get_backend(backend) if isinstance(backend, str) else backend
        self.session = self.backend.open(sim.m, spec.width, spec.height, groups, tendons)
        self.dr = mujoco.MjData(sim.m)
        self.tendons = tendons
        k0 = int(math.floor(sim.time * spec.rate_hz + 1e-9))
        self.snap = (k0, sim.time, sim.d.qpos.copy())    # (k, t, qpos): the state at attach time stands for exposure k0
        self.k_next = k0 + 1
        self.cache = None
        self.n_rendered = 0
        self.render_s = 0.0
        sim.tick_hooks.append(self._tick)

    def _tick(self, sim):
        if sim.time + 1e-9 >= self.k_next / self.spec.rate_hz:
            self.snap = (self.k_next, sim.time, sim.d.qpos.copy())
            self.k_next = int(math.floor(sim.time * self.spec.rate_hz + 1e-9)) + 1

    def _render_qpos(self, qpos):
        mj = self.mj
        self.dr.qpos[:] = qpos
        if self.tendons:
            mj.mj_fwdPosition(self.sim.m, self.dr)
        else:
            mj.mj_kinematics(self.sim.m, self.dr)
        t0 = time.perf_counter()
        out = self.session.render(self.sim.m, self.dr, self.spec)
        self.render_s += time.perf_counter() - t0
        self.n_rendered += 1
        return out

    def latest(self):
        if self.snap is None:
            return None
        k, t, q = self.snap
        if self.cache is None or self.cache.index != k:
            rgb, depth = self._render_qpos(q)
            self.cache = Frame(t, k, rgb, depth, self.spec.name)
        return self.cache

    def now(self):
        rgb, depth = self._render_qpos(self.sim.d.qpos.copy())
        return Frame(self.sim.time, -1, rgb, depth, self.spec.name)

    def detach(self):
        if self._tick in self.sim.tick_hooks:
            self.sim.tick_hooks.remove(self._tick)

    def close(self):
        self.detach()
        self.session.close()
