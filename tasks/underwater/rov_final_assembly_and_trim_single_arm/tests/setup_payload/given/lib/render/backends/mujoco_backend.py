"""The MuJoCo backend: offscreen OpenGL (EGL first, OSMesa fallback), deterministic, ideal pinhole cameras.

    MujocoBackend(gl=None).render(request)          one-shot: compile the SceneSpec, render, post-process, provenance
    MujocoBackend().open(scene) -> MujocoSession    persistent model + GL contexts for policy-rate rendering (a
                                                    base.RenderSession; contract in base.py):
        session.set_state(StateSpec | qpos=... | joints={...}); session.sync_from(model, data)  (copy qpos by joint name)
        session.render(cameras, photo=PhotoSpec(), seed=0, outputs=("rgb",)) -> RenderResult  (same pipeline)
        session.close()
    camera_from_mjcf(model, data, name, width, height)   CameraSpec at a physics-model camera's current pose (wrist cams)

RGB + depth in ONE pass (RA2): when depth (or sensor_depth) is requested, each camera is drawn once (mjr_render) and
colour and depth are read from the same framebuffer (mjr_readPixels(rgb, depth)); the colour bytes are those of the
rgb-only path and the depth is bitwise the separate (1.0.0) depth pass's (tests/test_render_session.py).  The pass is
used only where that equality is guaranteed (MujocoSession.one_pass_ok): (a) the normal pass writes no depth for
transparent geoms (alpha < 1) while the segment-mode depth pass does, so a render model with a visible transparent geom
keeps the separate pass; (b) the skybox writes depth at >= 0.69 x far_m along the ray, so without segmentation a pixel
whose ray distance is >= 0.5 x far_m would read as background: a camera from which some visible geom (bounding sphere;
infinite planes, skins, flexes count as unbounded) reaches beyond far_m / 2 (25 m at the default far_m = 50 m) keeps the
separate pass.  Segmentation keeps its own pass (grader-only).  LOD (RA3): SceneSpec.lod = "policy" renders decimated visible meshes
(mujoco_scene.apply_lod), recorded in Provenance.lod.

GL: MUJOCO_GL must be fixed before `mujoco` is first imported in the process.  If mujoco is not yet imported,
`select_gl()` probes the preference ($MUJOCO_GL, default EGL) in a subprocess and falls back from EGL to OSMesa when EGL
cannot create a context (recorded in the provenance environment); if mujoco is already imported the process' choice
is used as is.  Inside the task images MUJOCO_GL=osmesa (CPU, llvmpipe); on the authoring host EGL
(NVIDIA).  The two produce slightly different pixels, which is why agent-visible photos are FIXED FILES rendered once
at authoring time and their provenance names the GL vendor / renderer / version (render/README.md).

Cameras: the pinhole projection is set directly on the MjvScene's GL cameras (pos, forward, up and an off-axis
frustum from fx fy cx cy on the OpenCV pixel grid), so fx != fy, off-centre principal points and roll are exact;
the headlight is moved with the camera.  Only geom groups SceneSpec.visible_groups are drawn, no sites / frames /
labels.  Depth comes from MuJoCo's reversed-Z buffer (metres along the optical axis), segmentation from the
segment-id pass (geom ids).
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from typing import Dict, List, Optional, Sequence

import numpy as np

from ..spec import CameraSpec, PhotoSpec, RenderRequest, RenderResult, SceneSpec, StateSpec
from .base import BackendCapabilities, RawRender, RawView, RenderBackend, RenderSession, raw_outputs

BACKEND_VERSION = "mujoco-backend/1.0.0"
_PROBED: Dict[str, bool] = {}


def _probe(gl: str, timeout: float = 60.0) -> bool:
    if gl in _PROBED:
        return _PROBED[gl]
    code = ("import os; os.environ['MUJOCO_GL']=%r; import mujoco; "
            "from mujoco.rendering.classic import gl_context as g; c=g.GLContext(32,32); c.make_current(); c.free(); print('ok')" % gl)
    env = dict(os.environ, MUJOCO_GL=gl, PYTHONDONTWRITEBYTECODE="1")
    env.pop("PYOPENGL_PLATFORM", None)
    try:
        r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=timeout, env=env)
        ok = r.returncode == 0 and "ok" in r.stdout
    except Exception:  # noqa: BLE001
        ok = False
    _PROBED[gl] = ok
    return ok


GL_FALLBACK: Dict[str, str] = {}


def select_gl(preference: Optional[str] = None) -> str:
    """Fix MUJOCO_GL for this process (before mujoco is imported) and return it.

    Order: the preference (argument, else $MUJOCO_GL, else "egl"), then OSMesa if the preference was EGL and EGL cannot
    create a context (no GPU / no EGL device, e.g. inside Docker).  "osmesa" is never silently replaced by EGL (its
    pixels differ).  Once mujoco is imported the process' choice is frozen and returned as is."""
    if "mujoco" in sys.modules:
        cur = os.environ.get("MUJOCO_GL", "")
        if preference and cur and preference != cur:
            raise RuntimeError("mujoco already imported with MUJOCO_GL=%s; cannot switch to %s in this process" % (cur, preference))
        return cur or "glfw"
    pref = preference or os.environ.get("MUJOCO_GL") or "egl"
    cands = [pref] + (["osmesa"] if pref == "egl" else [])
    for gl in cands:
        if _probe(gl):
            os.environ["MUJOCO_GL"] = gl
            if gl == "osmesa":
                os.environ["PYOPENGL_PLATFORM"] = "osmesa"
            if gl != pref:
                GL_FALLBACK["requested"] = pref
                GL_FALLBACK["reason"] = "%s could not create an offscreen context" % pref
            return gl
    raise RuntimeError("no offscreen OpenGL: tried %s (install libosmesa6 or an EGL driver)" % cands)


def _gl_strings() -> Dict[str, str]:
    try:
        from OpenGL import GL
        f = lambda e: (GL.glGetString(e) or b"").decode("ascii", "replace")  # noqa: E731
        return {"gl_vendor": f(GL.GL_VENDOR), "gl_renderer": f(GL.GL_RENDERER), "gl_version": f(GL.GL_VERSION)}
    except Exception as e:  # noqa: BLE001
        return {"gl_error": repr(e)[:200]}


def set_scene_camera(scn, cam: CameraSpec, model) -> None:
    """Point both GL cameras of an MjvScene through `cam` (OpenCV pinhole) and move the headlight with it."""
    ext = float(model.stat.extent)
    near, far = float(model.vis.map.znear) * ext, float(model.vis.map.zfar) * ext
    R = cam.R
    eye, fwd, up = cam.position, R[:, 2], -R[:, 1]
    W, H = cam.width, cam.height
    left, right = (-0.5 - cam.cx) / cam.fx * near, (W - 0.5 - cam.cx) / cam.fx * near
    top, bottom = (cam.cy + 0.5) / cam.fy * near, -(H - 0.5 - cam.cy) / cam.fy * near
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


def camera_from_mjcf(model, data, name: str, width: int, height: int, fx=None, fy=None, cx=None, cy=None, **kw) -> CameraSpec:
    """A CameraSpec at the CURRENT pose of a camera of a MuJoCo model (e.g. a wrist camera riding on the arm in the
    grader's physics model; call after mj_forward - mj_kinematics alone leaves cam_xpos / cam_xmat stale, mj_camlight
    updates them - or on a MujocoSession's own model / data after sync_from / set_state, which run both).  MuJoCo cameras look along -z with +y up, so the
    OpenCV frame is [x, -y, -z] of cam_xmat.  Intrinsics default to the model camera's vertical field of view with square
    pixels and the ideal principal point; pass fx / fy / cx / cy (and k1 ..., exposure_ev, white_balance via **kw) to
    override (e.g. the hidden per-episode draws of a task)."""
    import math

    import mujoco
    cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, name)
    if cid < 0:
        raise ValueError("no camera %r" % name)
    R = np.asarray(data.cam_xmat[cid], float).reshape(3, 3) @ np.diag([1.0, -1.0, -1.0])
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, np.asarray(data.cam_xpos[cid], float)
    ss = np.asarray(getattr(model, "cam_sensorsize", np.zeros((model.ncam, 2)))[cid], float)
    if (fx is None or fy is None) and ss.size == 2 and ss.min() > 0:
        raise ValueError("camera %r is defined by sensorsize / focal (not fovy): pass fx, fy, cx, cy explicitly (MuJoCo "
                         "applies principal_pixel with the opposite sign of OpenCV's cx, cy)" % name)
    if fy is None:
        fy = 0.5 * height / math.tan(math.radians(float(model.cam_fovy[cid])) / 2.0)
    fx = fy if fx is None else fx
    cx = (width - 1) / 2.0 if cx is None else cx
    cy = (height - 1) / 2.0 if cy is None else cy
    return CameraSpec(name=name, width=int(width), height=int(height), fx=float(fx), fy=float(fy), cx=float(cx), cy=float(cy),
                      world_T_cam=T.tolist(), **kw)


class MujocoSession(RenderSession):
    """A compiled render model with cached GL renderers (one per image size)."""

    def __init__(self, backend: "MujocoBackend", scene: SceneSpec):
        import mujoco
        from . import mujoco_scene
        self.backend, self.scene = backend, scene
        self._mj = mujoco
        self.compiled = mujoco_scene.compile_scene(scene)
        self.model, self.data = self.compiled.model, self.compiled.data
        self.labels = self.compiled.labels
        self._renderers: Dict[tuple, object] = {}
        self._opt = mujoco.MjvOption()
        self._opt.geomgroup[:] = 0
        for g in scene.visible_groups:
            self._opt.geomgroup[g] = 1
        self._opt.sitegroup[:] = 0
        self._opt.frame = mujoco.mjtFrame.mjFRAME_NONE
        self._opt.label = mujoco.mjtLabel.mjLABEL_NONE
        self._free = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(self.model, self._free)
        self.gl_facts: Dict[str, str] = {}
        self._digest = None
        self._ray_scale: Dict[tuple, np.ndarray] = {}
        self.one_pass = True                                   # RA2; False = the legacy separate depth pass
        self._depth_geoms = self._drawn_geom_bounds()          # (ids, radius) of drawn geoms; None = unbounded (one-pass guard)
        self.transparent_geoms = self._transparent_drawn_geoms()
        self.last_one_pass: List[bool] = []                    # per camera of the last depth render: one pass used?
        from ..provenance import scene_content_digest
        self.scene_content = scene_content_digest(scene)       # the scene's files are fixed while the session is open

    # ---- state
    def set_state(self, state: Optional[StateSpec] = None, qpos=None, joints=None):
        from . import mujoco_scene
        st = state or StateSpec(keyframe=self.scene.state.keyframe, qpos=qpos, joints=dict(joints or {}))
        mujoco_scene.apply_state(self.model, self.data, st)

    def sync_from(self, model, data):
        """Copy the configuration of another (physics) model by joint name, then mj_kinematics + mj_comPos + mj_camlight
        (bodies, then cameras and lights riding on or tracking them): the grader steps its own model and renders through
        this session (scale refs / overrides / LOD live only in the render model)."""
        mj = self._mj
        for j in range(model.njnt):
            nm = mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, j)
            k = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_JOINT, nm) if nm else -1
            if k < 0:
                continue
            n = {0: 7, 1: 4, 2: 1, 3: 1}[int(model.jnt_type[j])]
            a, b = int(model.jnt_qposadr[j]), int(self.model.jnt_qposadr[k])
            self.data.qpos[b:b + n] = data.qpos[a:a + n]
        mj.mj_kinematics(self.model, self.data)
        mj.mj_comPos(self.model, self.data)                    # subtree_com: cameras / lights in track(com) / targetbody modes
        mj.mj_camlight(self.model, self.data)                  # cam_xpos / light_xpos: moving cameras and lights follow

    def resolved_digest(self) -> str:
        from . import mujoco_scene
        if self._digest is None:
            self._digest = mujoco_scene.resolved_digest(self.model)
        return self._digest

    def lod_record(self):
        return self.compiled.info.get("lod")

    def facts(self):
        f = {"session": True, "rgbd_one_pass": self._one_pass_used()}
        if self.backend.gl == "osmesa":
            f["llvmpipe_threads"] = os.environ.get("LP_NUM_THREADS", "default")
        return f

    # ---- one-pass guard (RA2): where the one-pass depth is provably the separate pass's
    def _drawn(self):
        m = self.model
        groups = np.asarray(m.geom_group, int)
        vis = np.zeros(m.ngeom, bool)
        for g in self.scene.visible_groups:
            vis |= groups == int(g)
        return np.nonzero(vis)[0]

    def _transparent_drawn_geoms(self) -> List[str]:
        """Drawn geoms whose colour or material alpha is < 1: the normal pass writes no depth for them."""
        m, mj = self.model, self._mj
        out = []
        for i in self._drawn():
            a = float(m.geom_rgba[i][3])
            mid = int(m.geom_matid[i])
            if a < 1.0 or (mid >= 0 and float(m.mat_rgba[mid][3]) < 1.0):
                out.append(mj.mj_id2name(m, mj.mjtObj.mjOBJ_GEOM, int(i)) or "geom%d" % i)
        return out

    def _drawn_geom_bounds(self):
        m, mj = self.model, self._mj
        if int(getattr(m, "nskin", 0)) or int(getattr(m, "nflex", 0)):
            return None
        ids = self._drawn()
        r = np.asarray(m.geom_rbound, float)[ids].copy()
        types = np.asarray(m.geom_type, int)[ids]
        size = np.asarray(m.geom_size, float)[ids]
        plane = types == int(mj.mjtGeom.mjGEOM_PLANE)
        if np.any(plane & ((size[:, 0] <= 0) | (size[:, 1] <= 0))):
            return None                                        # an infinite plane reaches the far clip
        r[plane] = np.hypot(size[plane, 0], size[plane, 1])
        return ids, r

    def _one_pass_used(self) -> bool:
        return bool(self.last_one_pass) and all(self.last_one_pass)

    def one_pass_ok(self, cam: CameraSpec, with_segmentation: bool) -> bool:
        """True if the one-pass depth of `cam` is bitwise the separate depth pass's (module docstring)."""
        if not self.one_pass or self.transparent_geoms:
            return False
        if with_segmentation:
            return True                                        # background from the segmentation, as in the separate pass
        if self._depth_geoms is None:
            return False
        ids, r = self._depth_geoms
        far = float(self.model.vis.map.zfar) * float(self.model.stat.extent)
        reach = np.linalg.norm(np.asarray(self.data.geom_xpos, float)[ids] - cam.position[None, :], axis=1) + r
        if int(self.model.ntendon):                            # tendons are drawn between sites / wrap points
            pts = np.concatenate([np.asarray(self.data.site_xpos, float).reshape(-1, 3), np.asarray(self.data.wrap_xpos, float).reshape(-1, 3)])
            if len(pts):
                reach = np.concatenate([reach, np.linalg.norm(pts - cam.position[None, :], axis=1) + 0.05])
        return bool(len(reach) == 0 or float(reach.max()) < 0.5 * far * 0.999)

    # ---- rendering
    def _renderer(self, w, h):
        key = (w, h)
        if key not in self._renderers:
            m = self.model
            m.vis.global_.offwidth, m.vis.global_.offheight = int(w), int(h)
            r = self._mj.Renderer(m, height=int(h), width=int(w))
            self._renderers[key] = r
            if not all(hasattr(r, a) for a in ("_mjr_context", "_rect", "_gl_context")):
                self.one_pass = False                          # a mujoco build without these internals: two passes
            if not self.gl_facts:
                self.gl_facts = _gl_strings()
        return self._renderers[key]

    def _metric_depth(self, zbuf):
        """MuJoCo's reversed-Z buffer -> metres along the optical axis, with exactly the arithmetic of
        mujoco.Renderer's depth mode (float32 near / far, float64 division), so both paths agree bitwise."""
        ext = self.model.stat.extent
        zfar = np.float32(self.model.vis.map.zfar * ext)
        znear = np.float32(self.model.vis.map.znear * ext)
        c = -(zfar + znear) / (zfar - znear)
        d = -(np.float32(2) * zfar * znear) / (zfar - znear)
        c = np.float32(-0.5) * c - np.float32(0.5)
        d = np.float32(-0.5) * d
        return (d / (zbuf.astype(np.float64) + c)).astype(np.float32)

    def _rgbd_one_pass(self, r):
        mj = self._mj
        if r._gl_context:
            r._gl_context.make_current()
        H, W = r.height, r.width
        rgb = np.empty((H, W, 3), np.uint8)
        z = np.empty((H, W), np.float32)
        mj.mjr_render(r._rect, r.scene, r._mjr_context)
        mj.mjr_readPixels(rgb, z, r._rect, r._mjr_context)
        if r._gl_context:                                      # EGL / OSMesa read bottom-up (as mujoco.Renderer)
            rgb, z = np.flipud(rgb), np.flipud(z)
        return np.ascontiguousarray(rgb), self._metric_depth(np.ascontiguousarray(z))

    def _ray_range_scale(self, cam: CameraSpec) -> np.ndarray:
        key = (cam.width, cam.height, cam.fx, cam.fy, cam.cx, cam.cy)
        s = self._ray_scale.get(key)
        if s is None:
            vv, uu = np.mgrid[0:cam.height, 0:cam.width].astype(np.float64)
            s = np.sqrt(1.0 + ((uu - cam.cx) / cam.fx) ** 2 + ((vv - cam.cy) / cam.fy) ** 2)
            if len(self._ray_scale) >= 8:
                self._ray_scale.pop(next(iter(self._ray_scale)))
            self._ray_scale[key] = s
        return s

    def render_raw_views(self, cameras: Sequence[CameraSpec], outputs=("rgb",)) -> List[RawView]:
        mj = self._mj
        outputs = raw_outputs(outputs)
        want_d, want_s = "depth" in outputs, "segmentation" in outputs
        far = float(self.model.vis.map.zfar) * float(self.model.stat.extent)
        out = []
        if want_d:
            self.last_one_pass = []
        for cam in cameras:
            r = self._renderer(cam.width, cam.height)
            r.update_scene(self.data, camera=self._free, scene_option=self._opt)
            set_scene_camera(r.scene, cam, self.model)
            r.scene.flags[mj.mjtRndFlag.mjRND_SHADOW] = int(bool(self.scene.shadows))
            dz = None
            if want_d and self.one_pass_ok(cam, want_s):
                rgb, dz = self._rgbd_one_pass(r)
                self.last_one_pass.append(True)
            else:
                if want_d:
                    self.last_one_pass.append(False)
                rgb = r.render().copy()
            depth = seg = None
            if want_s:
                r.enable_segmentation_rendering()
                s = r.render()
                r.disable_segmentation_rendering()
                seg = np.where(s[..., 1] == int(mj.mjtObj.mjOBJ_GEOM), s[..., 0], -1).astype(np.int32)
            if want_d:
                if dz is None:                                 # legacy second pass (segment flags, no skybox)
                    r.enable_depth_rendering()
                    d = r.render().copy()
                    r.disable_depth_rendering()
                    bg = (d >= far * 0.999) if seg is None else ((seg < 0) | (d >= far * 0.999))
                else:
                    d = dz
                    if seg is not None:
                        bg = (seg < 0) | (d >= far * 0.999)
                    else:                                      # the skybox's depth: ray distance >= 0.69 far; every drawn
                        bg = (d >= far * 0.999) | (d * self._ray_range_scale(cam) >= 0.5 * far)   # geom < far / 2 (guard)
                d[bg] = 0.0
                depth = d.astype(np.float32)
            out.append(RawView(rgb=rgb, depth=depth, seg=seg))
        return out

    def close(self):
        for r in self._renderers.values():
            try:
                r.close()
            except Exception:  # noqa: BLE001
                pass
        self._renderers.clear()

    def __del__(self):
        try:
            self.close()
        except Exception:  # noqa: BLE001
            pass


class MujocoBackend(RenderBackend):
    name = "mujoco"

    def __init__(self, gl: Optional[str] = None):
        self.gl = select_gl(gl)

    def version(self) -> str:
        import mujoco
        return "%s mujoco-%s gl-%s" % (BACKEND_VERSION, mujoco.__version__, self.gl)

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(name=self.name, implemented=True, outputs=["rgb", "depth", "segmentation"],
                                   native_lens_distortion=False, sessions=True, max_image_px=16384 * 16384,
                                   deterministic="bitwise on the same mujoco version, GL driver and device with SceneSpec.offsamples = 0 "
                                                 "(EGL vs OSMesa differ; MSAA + shadows is not deterministic on NVIDIA EGL)",
                                   notes="MuJoCo classic OpenGL renderer; anti-aliasing via PhotoSpec.supersample; shadows from lights with castshadow",
                                   one_pass_rgbd=True, lod_levels=["full", "policy"])

    def environment(self, session: Optional[MujocoSession] = None) -> Dict[str, str]:
        import mujoco
        env = {"mujoco": mujoco.__version__, "mujoco_gl": self.gl}
        if GL_FALLBACK:
            env["mujoco_gl_fallback_from"] = GL_FALLBACK.get("requested", "")
        if os.environ.get("MUJOCO_EGL_DEVICE_ID"):
            env["mujoco_egl_device_id"] = os.environ["MUJOCO_EGL_DEVICE_ID"]
        if session is not None:
            env.update(session.gl_facts)
        return env

    def open(self, scene: SceneSpec) -> MujocoSession:
        return MujocoSession(self, scene)

    def render_raw(self, request: RenderRequest, raw_cameras: List[CameraSpec]) -> RawRender:
        t0 = time.perf_counter()
        with self.open(request.scene) as s:
            t1 = time.perf_counter()
            views = s.render_raw_views(raw_cameras, request.outputs)
            t2 = time.perf_counter()
            facts = {"t_compile_s": round(t1 - t0, 4), "t_render_s": round(t2 - t1, 4)}
            if "depth" in request.outputs or "sensor_depth" in request.outputs:
                facts["rgbd_one_pass"] = s._one_pass_used()
            if self.gl == "osmesa":
                facts["llvmpipe_threads"] = os.environ.get("LP_NUM_THREADS", "default")
            return RawRender(views=views, labels=s.labels, resolved_scene_digest=s.resolved_digest(),
                             environment=self.environment(s), facts=facts, lod=s.lod_record())
