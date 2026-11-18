"""Live frames of your own MuJoCo scene through MuJoCo's Filament renderer in a baked room (task-independent).

    from render.filament.view import open_view
    view = open_view(spec_or_mjcf_path, room="~/render/room", look="path/to/look.json", surface="~/render/oak",
                     bench_xy=(0.45, 0.0))
    rgb = view.render(data, "top", 1280, 960)                     # a camera of the model, by name
    rgb, depth = view.render({"j1": 0.3}, camera, depth=True)     # a CameraSpec / dict of intrinsics and pose
    view.close()

The scene is your MjSpec (or MJCF file) in its own frame, the room bundle (`room/`: an arb-room/1 bundle with baked
lightmaps, its lamps and its HDR image light `ibl/ibl.ktx`) placed in that frame as it was baked, and a look file (JSON:
the fields of `render.filament.backend.Look`: light units, exposure, the filmic display transform, PBR terms per
material class). It is compiled once (`open_view`, about 5-30 s on a CPU with the room); each frame copies the state,
updates the Filament scene and renders RGB as an exact pinhole (OpenCV conventions, below), then the display transform
through an 8-bit lookup table (built once per process, about 10 s). Depth comes from MuJoCo's own renderer on the same
geometry (your scene plus the room's meshes): metres along the optical axis, registered pixel for pixel with the RGB,
0.0 where nothing is hit. No noise, blur or lens distortion is added; `view.session.render(...)` runs the render layer's
photo pipeline (render.spec.PhotoSpec) on the same frames if you want it.

What the view changes in its copy of your scene (never your MjSpec): geoms in groups above `max_group` (MuJoCo's
collision group 3 by convention) and geoms whose names start with one of `drop_prefixes` are removed; the room is
added; only the `max_live_lamps` room lamps nearest `bench_xy` light the live (unbaked) objects, the rest of the room is
lit by its lightmaps; the scene's own lights are removed unless `keep_lights` (the room lights the scene); the look's
material rules give every visible geom physically based terms (first rule whose geom / material globs match wins).

Cameras (`render`'s `camera`):
  - a camera name of the model (with `width`, `height`): its pose at the current state, its vertical field of view,
    square pixels, the principal point at the image centre;
  - a `render.spec.CameraSpec`, or a dict with `width`, `height`, `fx`, `fy`, `cx`, `cy` (pixels) and `world_T_cam`
    (4 x 4, the camera's pose in the scene frame, OpenCV axes: x right, y down, z forward along the optical axis);
    `K` (3 x 3) may stand for fx / fy / cx / cy.
State (`render`'s `state`): an `MjData` (of this scene's model or of any model: joints are copied by name, a free joint
moves its body), a `(model, data)` pair, a dict {joint name: value} (a hinge / slide joint a number, a ball joint a
quaternion wxyz, a free joint [x, y, z, qw, qx, qy, qz]) applied on top of the current state, a full qpos array of the
view's own model (`view.model`), or None (the current state).

Runs offscreen: the display variables are dropped before Filament starts (a display server that does not answer would
block it). One render target and one pixel readback per image size are reused for the life of the view. One view (or
cell session) per process at a time: close it before opening the next (a second open raises RuntimeError). Close
views before the interpreter exits (an exit handler also does it): Filament's context must outlive every scene object.
"""
import atexit
import copy
import gc
import json
import math
import os
import time
import warnings

import numpy as np

from render.backends.base import BackendCapabilities, RawView, RenderSession, raw_outputs
from render.backends.mujoco_backend import MujocoBackend

__all__ = ["open_view", "FilamentView", "load_look", "camera_spec", "available", "check_one_session"]

DEFAULT_DROP_PREFIXES = ()
_LUTS = {}
_SPECS = {}
_OPEN = set()          # every open Filament session of the process (this module's and the task cell's own)
_KEY = [0]


def _shutdown():
    """At exit: close the open sessions, then drop the Filament context (destroying it with live objects panics and
    crashes the interpreter's teardown)."""
    for s in list(_OPEN):
        try:
            s.close()
        except Exception:  # noqa: BLE001
            pass
    gc.collect()
    try:
        from render.filament import backend as FB
        FB._CTX.clear()
    except Exception:  # noqa: BLE001
        pass
    gc.collect()


atexit.register(_shutdown)


def _headless():
    """Filament (its GLFW-backed resource providers) connects to $DISPLAY when one is set; a display server that accepts
    the connection and never answers blocks it for ever. The renderer is offscreen: drop the display variables."""
    for k in ("DISPLAY", "WAYLAND_DISPLAY"):
        os.environ.pop(k, None)


def check_one_session():
    """One Filament session per process at a time: a second scene of a full cell with the room exceeds what the CPU
    renderer can hold (its frames then fail). RuntimeError while another session of this process is open."""
    live = [s for s in _OPEN if getattr(s, "scn", None) is not None]
    if live:
        raise RuntimeError("a Filament render session is already open in this process: close it first (session.close() / "
                           "view.close()); one session per process at a time")


def available(room=None):
    """(ok, reason): the Filament module imports (and the room bundle, if given, is present)."""
    _headless()
    try:
        import mujoco._render_filament  # noqa: F401
    except Exception as e:  # noqa: BLE001
        return False, "no Filament module: %s" % type(e).__name__
    if room is not None and not os.path.isfile(os.path.join(os.path.expanduser(str(room)), "room.json")):
        return False, "no room bundle at %s" % room
    return True, ""


def _rules(table):
    out = []
    for r in table:
        gp, mp, p = r
        out.append((gp, mp, {k: float(v) for k, v in dict(p).items()}))
    return out


def load_look(look=None, room=None, surface=None, exposure_factor=1.0):
    """A `render.filament.backend.Look` from a look file (path), a dict, a Look (copied) or None (the Look defaults).

    The file's keys are Look fields (keys starting with "_" are notes, other unknown keys are skipped with a warning);
    "materials" as a list [[geom glob or null,
    material glob or null, {"roughness": r, "metallic": m}], ...] REPLACES the material rules (first match wins), as a
    dict {glob: {...}} it edits the existing rules (Look.override). The room's HDR image light (`<room>/ibl/ibl.ktx`) and
    the bench top's scanned maps (`surface`: a folder of *_diff_ / *_nor_gl_ / *_arm_ maps) are filled in when the look
    does not name them. `exposure_factor` scales every light, the image light and the baked emission (a camera
    exposure: 1.2 = 20 % brighter) and leaves the display transform's gain as it is."""
    from render.filament.backend import Look
    if isinstance(look, Look):
        lk = copy.deepcopy(look)
    else:
        spec = {}
        if isinstance(look, dict):
            spec = dict(look)
        elif look is not None:
            with open(os.path.expanduser(str(look))) as f:
                spec = json.load(f)
        spec = {k: v for k, v in spec.items() if not str(k).startswith("_")}
        lk = Look(name=str(spec.pop("name", "view")), version=str(spec.pop("version", "1")))
        mats = spec.pop("materials", None)
        if isinstance(mats, list):
            lk.materials = _rules(mats)
        elif isinstance(mats, dict):
            spec["materials"] = mats
        unused = sorted(k for k in spec if not hasattr(lk, k))
        if unused:                         # a cell's look file may carry blocks for its own code (e.g. kit_materials)
            warnings.warn("look keys the renderer does not use: %s" % ", ".join(unused), RuntimeWarning, stacklevel=2)
        lk.override({k: v for k, v in spec.items() if hasattr(lk, k)})
    if room is not None and not lk.room_ibl_ktx:
        ibl = os.path.join(os.path.expanduser(str(room)), "ibl", "ibl.ktx")
        lk.room_ibl_ktx = ibl if os.path.isfile(ibl) else None
    if surface is not None and not lk.surface_dir:
        sd = os.path.expanduser(str(surface))
        lk.surface_dir = sd if os.path.isdir(sd) else None
    f = float(exposure_factor)
    if f != 1.0:
        lk.exposure_scale = float(lk.exposure_scale) * f
        lk.post_tonemap_gain = float(lk.post_tonemap_gain) * f
    return lk


def _lut(look):
    """The display transform (render.filament.agx.regrade) of every 8-bit RGB triple (256^3 x 3 bytes, built once per
    process and gain)."""
    from render.filament.agx import regrade
    g = round(float(look.post_tonemap_gain) / float(look.exposure_scale), 9)
    mode = "punchy" if look.post_tonemap == "agx_punchy" else "none"
    key = (g, mode)
    if key not in _LUTS:
        v = np.arange(256, dtype=np.uint8)
        out = np.empty((256 ** 3, 3), np.uint8)
        for r in range(256):
            tile = np.stack(np.broadcast_arrays(np.uint8(r), v[:, None], v[None, :]), -1).reshape(-1, 3)
            out[r * 65536:(r + 1) * 65536] = regrade(tile, g, mode).reshape(-1, 3)
        _LUTS.clear()
        _LUTS[key] = out
    return _LUTS[key]


def _room_geometry(spec, bundle):
    """The room's meshes only (no textures, no lamps): the depth twin of the room."""
    import mujoco
    with open(os.path.join(bundle, "room.json")) as f:
        room = json.load(f)
    mat = spec.add_material(name="room_depth", rgba=[0.5, 0.5, 0.5, 1.0])
    for me in room["meshes"]:
        z = np.load(os.path.join(bundle, me["file"]))
        mesh = spec.add_mesh(name="room_mesh_" + me["name"])
        mesh.uservert = np.asarray(z["v"], np.float64).ravel()
        mesh.userface = np.asarray(z["f"], np.int32).ravel()
        mesh.inertia = mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL
        spec.worldbody.add_geom(name="room_" + me["name"], type=mujoco.mjtGeom.mjGEOM_MESH, meshname=mesh.name,
                                material=mat.name, contype=0, conaffinity=0, group=2)


def _depth_scene(key):
    """SceneSpec builder of a view's depth twin: a fresh copy of the registered MjSpec."""
    return _SPECS[key].copy()


def _as_spec(model):
    import mujoco
    if isinstance(model, mujoco.MjSpec):
        return model.copy()
    p = os.path.expanduser(str(model))
    if os.path.isfile(p):
        return mujoco.MjSpec.from_file(p)
    raise TypeError("open_view needs a mujoco.MjSpec or the path of an MJCF file, got %r" % (model,))


def _strip(spec, max_group, drop_prefixes):
    for g in list(spec.geoms):
        nm = g.name or ""
        if int(g.group) > int(max_group) or (drop_prefixes and nm.startswith(tuple(drop_prefixes))):
            spec.delete(g)
    return spec


def camera_spec(camera, model=None, data=None, width=None, height=None):
    """A render.spec.CameraSpec from a camera name of (model, data), a CameraSpec or a dict (module docstring)."""
    from render.spec import CameraSpec
    if isinstance(camera, CameraSpec):
        return camera
    if isinstance(camera, str):
        from render.backends.mujoco_backend import camera_from_mjcf
        if width is None or height is None:
            raise ValueError("camera %r by name needs width and height" % camera)
        return camera_from_mjcf(model, data, camera, int(width), int(height))
    if isinstance(camera, dict):
        c = dict(camera)
        if "K" in c:
            K = np.asarray(c.pop("K"), float)
            c.setdefault("fx", K[0, 0])
            c.setdefault("fy", K[1, 1])
            c.setdefault("cx", K[0, 2])
            c.setdefault("cy", K[1, 2])
        if width is not None:
            c["width"] = int(width)
        if height is not None:
            c["height"] = int(height)
        T = np.asarray(c.pop("world_T_cam"), float)
        return CameraSpec(name=str(c.pop("name", "view")), width=int(c.pop("width")), height=int(c.pop("height")),
                          fx=float(c.pop("fx")), fy=float(c.pop("fy")), cx=float(c.pop("cx")), cy=float(c.pop("cy")),
                          world_T_cam=T.tolist(), **c)
    raise TypeError("camera must be a camera name, a CameraSpec or a dict of intrinsics and pose")


class FilamentViewSession(RenderSession):
    """A render-layer session (render.backends.base.RenderSession): RGB from Filament (the room), depth / segmentation
    from a MuJoCo session on the same geometry. `FilamentView` wraps it; use it directly for the photo pipeline."""

    def __init__(self, backend, spec, room=None, look=None, surface=None, bench_xy=(0.0, 0.0), max_live_lamps=6,
                 lamp_range_m=40.0, keep_lights=False, exposure_factor=1.0, lod="full", near_m=0.01, far_m=80.0,
                 content_files=(), api=None):
        _headless()
        check_one_session()
        import mujoco
        import mujoco._render_filament as mjrf
        from render.spec import SceneSpec
        from render.filament.backend import FilamentBackend, _context, apply_look
        from render.room import add_room
        t0 = time.perf_counter()
        self.backend, self._mj, self._mjrf = backend, mujoco, mjrf
        self.room_dir = os.path.abspath(os.path.expanduser(str(room))) if room is not None else None
        _KEY[0] += 1
        key = "view-%d-%d" % (os.getpid(), _KEY[0])
        dspec = spec.copy()
        if self.room_dir:
            _room_geometry(dspec, self.room_dir)
        _SPECS[key] = dspec
        files = [str(p) for p in content_files] + ([os.path.join(self.room_dir, "room.json")] if self.room_dir else [])
        sc = SceneSpec(builder="render.filament.view:_depth_scene", builder_args={"key": key}, content_files=files,
                       shadows=False, lod=str(lod), near_m=float(near_m))
        self._key = key
        try:
            self.depth = MujocoBackend.open(backend, sc)   # depth, labels, scene digest of the twin
        except Exception:
            _SPECS.pop(key, None)
            raise
        self.scene, self.labels, self.scene_content = self.depth.scene, self.depth.labels, self.depth.scene_content
        self.look = load_look(look, room=self.room_dir, surface=surface, exposure_factor=exposure_factor)
        fspec = spec.copy()
        rec = add_room(fspec, self.room_dir, bench_xy=(float(bench_xy[0]), float(bench_xy[1]))) if self.room_dir else None
        lamps = sorted([l for l in fspec.lights if (l.name or "").startswith("room_light_")],
                       key=lambda l: math.hypot(l.pos[0] - float(bench_xy[0]), l.pos[1] - float(bench_xy[1])))
        for l in lamps[int(max_live_lamps):]:
            fspec.delete(l)
        for l in lamps[:int(max_live_lamps)]:
            l.range = float(lamp_range_m)
        if self.room_dir and not keep_lights:
            for l in list(fspec.lights):
                if not (l.name or "").startswith("room_light_"):
                    fspec.delete(l)
        self.look_record = apply_look(fspec, self.look)
        self.model = fspec.compile()
        self.data = mujoco.MjData(self.model)
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_camlight(self.model, self.data)
        self.near, self.far = float(near_m), float(far_m)
        self._jmap = None
        self.ctx = _context(api or os.environ.get("ARB_FILAMENT_API", "vulkan"))
        self.objs = mjrf.ModelObjects(self.ctx, self.model)
        self.scn = self.ctx.create_scene(mjrf.SceneParams())
        self.scn.configure_from_model(self.model)
        self.lights = mjrf.ModelLights(self.scn, self.objs)
        self.rends = mjrf.ModelRenderables(self.scn, self.objs)
        self.ibl = None
        if self.room_dir and self.look.room_ibl_ktx:
            self.ibl = FilamentBackend(look=self.look)._ktx_ibl(self.ctx, self.scn, self.look.room_ibl_ktx,
                                                                 self.look.room_ibl_ktx_intensity * self.look.exposure_scale)
        elif not self.room_dir and not self.look.env_file:
            self.ibl = FilamentBackend(look=self.look)._ktx_ibl(self.ctx, self.scn)
        self.lights.update(self.data)
        self.rends.update(self.data)
        self._targets = {}
        self.lut = _lut(self.look) if self.look.post_tonemap else None
        self.room_record = rec or {}
        self.t_open_s = time.perf_counter() - t0
        self._digest = None
        _OPEN.add(self)

    # ---- state
    def _update(self):
        mj = self._mj
        mj.mj_kinematics(self.model, self.data)
        mj.mj_camlight(self.model, self.data)             # light_xpos / light_xdir (not set by mj_kinematics)
        self.lights.update(self.data)
        self.rends.update(self.data)
        self.depth.sync_from(self.model, self.data)

    def sync_from(self, model, data):
        mj = self._mj
        if self._jmap is None or self._jmap[0] is not model:
            pairs = []
            for j in range(model.njnt):
                nm = mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, j)
                k = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_JOINT, nm) if nm else -1
                if k >= 0:
                    n = {0: 7, 1: 4, 2: 1, 3: 1}[int(model.jnt_type[j])]
                    pairs.append((int(model.jnt_qposadr[j]), int(self.model.jnt_qposadr[k]), n))
            self._jmap = (model, pairs)
        for a, b, n in self._jmap[1]:
            self.data.qpos[b:b + n] = data.qpos[a:a + n]
        self._update()

    def set_state(self, state=None, qpos=None, joints=None):
        """render.spec.StateSpec (keyframe / qpos / joints, applied from the model's reset state), or qpos (the full
        vector of this model) and / or joints {name: value} applied on top of the current state."""
        mj = self._mj
        if state is not None:
            from render.backends import mujoco_scene
            mujoco_scene.apply_state(self.model, self.data, state)
        if qpos is not None:
            q = np.asarray(qpos, float).ravel()
            if q.size != self.model.nq:
                raise ValueError("qpos has %d values, the view's model nq = %d" % (q.size, self.model.nq))
            self.data.qpos[:] = q
        for name, val in dict(joints or {}).items():
            j = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_JOINT, str(name))
            if j < 0:
                raise ValueError("no joint %r in the view's model" % name)
            n = {0: 7, 1: 4, 2: 1, 3: 1}[int(self.model.jnt_type[j])]
            a = int(self.model.jnt_qposadr[j])
            self.data.qpos[a:a + n] = np.asarray(val, float).ravel()[:n] if n > 1 else float(np.asarray(val, float).ravel()[0])
        self._update()

    # ---- rendering
    def _target(self, w, h):
        mjrf = self._mjrf
        if (w, h) not in self._targets:                   # one target and ONE readback request per size (a new
            t = self.ctx.create_render_target(mjrf.RenderTargetConfig(color_format=mjrf.PixelFormat.PIXEL_FORMAT_RGB8))
            t.resize(w, h)                                # request per frame leaks until the process dies)
            rd = mjrf.ReadPixelsRequest()
            rd.target = t
            rd.alloc(w * h * 3)
            self._targets[(w, h)] = (t, rd)
        return self._targets[(w, h)]

    def _rgb(self, cam):
        from render.filament.backend import gl_camera
        mjrf = self._mjrf
        target, rd = self._target(cam.width, cam.height)
        rq = mjrf.RenderRequest(camera=gl_camera(cam, self.near, self.far), draw_mode=mjrf.DrawMode.DRAW_MODE_DEFAULT)
        rq.scene, rq.target = self.scn, target
        rq.viewport.left, rq.viewport.bottom, rq.viewport.width, rq.viewport.height = 0, 0, cam.width, cam.height
        self.ctx.wait_for_frame(self.ctx.render([rq], [rd]))
        rgb = np.frombuffer(rd.buffer(), np.uint8).reshape(cam.height, cam.width, 3)
        if self.lut is None:
            return rgb.copy()
        idx = (rgb[..., 0].astype(np.int32) << 16) | (rgb[..., 1].astype(np.int32) << 8) | rgb[..., 2]
        return self.lut[idx]

    def render_raw_views(self, cameras, outputs=("rgb",)):
        if self.scn is None:
            raise RuntimeError("the view is closed")
        outputs = raw_outputs(outputs)
        extra = [o for o in outputs if o in ("depth", "segmentation")]
        mv = self.depth.render_raw_views(cameras, extra) if extra else [None] * len(cameras)
        out = []
        for cam, m in zip(cameras, mv):
            out.append(RawView(rgb=self._rgb(cam), depth=None if m is None else m.depth, seg=None if m is None else m.seg))
        return out

    def resolved_digest(self):
        if self._digest is None:
            from render.backends import mujoco_scene
            self._digest = mujoco_scene.resolved_digest(self.model)
        return self._digest

    def facts(self):
        return {"session": True, "renderer": "filament", "t_open_s": round(self.t_open_s, 2),
                "room": self.room_record.get("room")}

    def close(self):
        """Free the Filament objects in dependency order (the context outlives every session: _shutdown)."""
        if self.scn is None:
            return
        _OPEN.discard(self)
        if self.ibl is not None:
            try:
                self.scn.remove_light(self.ibl[0])
            except Exception:  # noqa: BLE001
                pass
        self.ibl = None
        self._targets.clear()
        self.rends = None
        self.lights = None
        self.scn = None
        self.objs = None
        gc.collect()
        self.depth.close()
        _SPECS.pop(self._key, None)


class FilamentViewBackend(MujocoBackend):
    """The MuJoCo backend's planning / finishing (render layer) with Filament RGB (sessions only)."""
    name = "filament"

    def version(self):
        import mujoco
        return "filament-view/1 mujoco-%s" % mujoco.__version__

    def capabilities(self):
        c = MujocoBackend.capabilities(self)
        return BackendCapabilities(name=self.name, implemented=True, outputs=c.outputs, native_lens_distortion=False, sessions=True,
                                   max_image_px=c.max_image_px, deterministic="same mujoco build, Vulkan driver and device",
                                   notes="Filament RGB in a baked room; depth / segmentation from the MuJoCo backend",
                                   one_pass_rgbd=False, lod_levels=c.lod_levels)

    def environment(self, session=None):
        env = MujocoBackend.environment(self, getattr(session, "depth", None))
        env["renderer"] = "filament"
        env["filament_api"] = os.environ.get("ARB_FILAMENT_API", "vulkan")
        return env


class FilamentView:
    """A compiled scene for live frames: `render(state, camera, width, height, depth=False)` (module docstring)."""

    def __init__(self, session):
        self.session = session
        self.model, self.data = session.model, session.data
        self.look = session.look
        self.t_open_s = session.t_open_s

    def set_state(self, state=None):
        """Copy a state into the view (render's `state` argument); returns the view."""
        import mujoco
        s = self.session
        if state is None:
            return self
        if isinstance(state, tuple) and len(state) == 2:
            s.sync_from(state[0], state[1])
        elif isinstance(state, mujoco.MjData):
            s.sync_from(state.model, state)
        elif isinstance(state, dict):
            s.set_state(joints=state)
        else:
            s.set_state(qpos=state)
        return self

    def camera(self, camera, width=None, height=None):
        """The CameraSpec that `render` uses for `camera` at the view's current state."""
        return camera_spec(camera, self.model, self.data, width, height)

    def render(self, state=None, camera=None, width=None, height=None, depth=False):
        """RGB uint8 (H, W, 3) of `camera` at `state`; with depth=True (rgb, depth float32 (H, W) metres, 0.0 = no hit)."""
        if camera is None:
            raise ValueError("render needs a camera")
        self.set_state(state)
        cam = self.camera(camera, width, height)
        v = self.session.render_raw_views([cam], ("rgb", "depth") if depth else ("rgb",))[0]
        return (v.rgb, v.depth) if depth else v.rgb

    def close(self):
        self.session.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def open_view(model, room=None, look=None, *, surface=None, bench_xy=(0.0, 0.0), max_live_lamps=6, lamp_range_m=40.0,
              keep_lights=False, exposure_factor=1.0, max_group=2, drop_prefixes=DEFAULT_DROP_PREFIXES, lod="full",
              near_m=0.01, far_m=80.0, content_files=(), api=None):
    """Compile `model` (a mujoco.MjSpec, copied, or the path of an MJCF file) with the room bundle `room` (a directory,
    or None: no room, MuJoCo's studio image light) and the look `look` (a look file's path, a dict or a Look) into a
    FilamentView. `surface`: the bench top's scanned maps (a directory), applied to the look's surface geom / material
    (Look.surface_geom, Look.surface_material) when the scene has them. `bench_xy`: where the scene's work area is in
    the scene frame (the room lamps nearest to it light the live objects; those within 1.6 m cast shadows)."""
    ok, why = available(room)
    if not ok:
        raise RuntimeError("Filament view unavailable: %s" % why)
    spec = _strip(_as_spec(model), max_group, drop_prefixes)
    sess = FilamentViewSession(FilamentViewBackend(), spec, room=room, look=look, surface=surface, bench_xy=bench_xy,
                               max_live_lamps=max_live_lamps, lamp_range_m=lamp_range_m, keep_lights=keep_lights,
                               exposure_factor=exposure_factor, lod=lod, near_m=near_m, far_m=far_m,
                               content_files=content_files, api=api)
    return FilamentView(sess)
