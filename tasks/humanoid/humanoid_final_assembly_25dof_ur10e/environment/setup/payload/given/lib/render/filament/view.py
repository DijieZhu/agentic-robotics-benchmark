"""Live frames of your own MuJoCo scene through MuJoCo's Filament renderer in a baked room (task-independent).

    from render.filament.view import open_view
    view = open_view(spec_or_mjcf_path, room="~/render/room", look="path/to/look.json", surface="~/render/oak",
                     bench_xy=(0.45, 0.0))
    rgb = view.render(data, "top", 1280, 960)                     # a camera of the model, by name
    rgb, depth = view.render({"j1": 0.3}, camera, depth=True)     # a CameraSpec / dict of intrinsics and pose
    view.close()

The scene is your MjSpec (or MJCF file) in its own frame, the room bundle (`room/`: an arb-room/1 bundle with baked
lightmaps, its lamps and its HDR image light `ibl/ibl.ktx`) placed in that frame as it was baked, and a look file
(JSON: the fields of `render.filament.backend.Look`: light units, exposure, the filmic display transform, PBR terms per
material class). It is compiled once (`open_view`: about 5-20 s on 8 CPUs for a full robot cell in the room, the
display transform's 8-bit lookup table, about 2 s, included and then kept for the process); each frame copies the
state, updates the Filament scene and renders RGB as an exact pinhole (OpenCV conventions, below), then the display
transform through that table: there about 1.2-1.5 s at 1280 x 960 and 0.6-0.8 s at 640 x 480 with depth, except the
first frame after Filament's context is made (about 10-15 s) and the first frame of each further image size (a few
seconds more). Depth comes from MuJoCo's own renderer on the same geometry (your scene plus the room's meshes): metres
along the optical axis, registered pixel for pixel with the RGB, 0.0 where nothing is hit. No noise, blur or lens
distortion is added; `view.session.render(...)` runs the render layer's photo pipeline (render.spec.PhotoSpec) on the
same frames if you want it.

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
view's own model (`view.model`), or None (the current state; until a state is given, the model's own pose: qpos0,
with its bodies, cameras and lights placed).

Runs offscreen: the display variables are dropped before Filament starts (a display server that does not answer would
block it). One render target and one pixel readback per image size are reused for the life of the view. One view (or
cell session) per process at a time: close it before opening the next (a second open raises RuntimeError). Closing the
last open view or session of the process releases Filament's context and hands its memory back (about 5.2 GB for a full
cell while open; a closed view keeps its MuJoCo models until you drop it), so the next one's first frame pays for a new
context again; ARB_FILAMENT_KEEP_CONTEXT=1 keeps the context and its memory. Close views before the interpreter exits
(an exit handler also does it): Filament's context must outlive every scene object.
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
KEEP_LIGHT_PREFIXES = ("ringlight_", "worklight_")   # the scene's own lamps that stay beside the room's
TRACE = []                                          # (stage, VmRSS GB, s) of the last session's opening


def _rss_gb():
    try:
        with open("/proc/self/status") as f:
            for ln in f:
                if ln.startswith("VmRSS:"):
                    return int(ln.split()[1]) / 1e6
    except OSError:
        pass
    return float("nan")


def _trace(stage, t0):
    TRACE.append((stage, round(_rss_gb(), 3), round(time.perf_counter() - t0, 2)))


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


def _release_context():
    """Once no Filament session of the process is open (render.filament.view's and the task cell's own share `_OPEN`):
    drop Filament's context too, then hand the freed heap back (glibc's malloc_trim). The context keeps every earlier
    scene's objects while it lives, so a closed view or session would keep most of its memory; the next open makes a
    fresh context (its first frame is slow again). ARB_FILAMENT_KEEP_CONTEXT=1 keeps the context. True if released."""
    if any(getattr(s, "scn", None) is not None for s in _OPEN):
        return False
    if os.environ.get("ARB_FILAMENT_KEEP_CONTEXT", "") == "1":
        return False
    gc.collect()
    try:
        from render.filament import backend as FB
        FB._CTX.clear()
    except Exception:  # noqa: BLE001
        pass
    gc.collect()
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:  # noqa: BLE001
        pass
    return True


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


_LUTG = {}
_LUTT = {}
# the tone paths of record per camera exposure (view._tone_path): the overhead camera (exposure 1) ONE render with
# the factor 4 inside the renderer and 1/4 in float before the display transform (finer dark
# codes); the wrist camera (exposure 4) two renders merged in float with the highlight shoulder (no
# one-render path keeps its dark tones, so the two-render path is adopted and its frame cost goes into the budget)
TONE_OF_RECORD = {"1": {"mode": "one", "in": 4.0}, "4": {"mode": "two_renders"}}


def _lut_tone(look, G, knee=None, white=None):
    """the display table of a one-render tone path: every 8-bit render triple c -> x = srgb_decode(c) x G (the
    exposure not applied inside the renderer), y = x scaled by rolloff_curve(m, knee, white) / m on the brightest
    channel m (when knee is given), display = srgb_encode(agx(y x post_tonemap_gain / exposure_scale)); 256^3 x 3 bytes,
    built once per (G, knee, white, gain, look) and process"""
    from render.filament.agx import srgb_decode, srgb_encode, agx
    g = round(float(look.post_tonemap_gain) / float(look.exposure_scale), 9)
    mode = "punchy" if look.post_tonemap == "agx_punchy" else "none"
    key = (round(float(G), 9), knee, white, g, mode)
    if key not in _LUTT:
        v = np.arange(256, dtype=np.uint8)
        out = np.empty((256 ** 3, 3), np.uint8)
        for r in range(256):
            tile = np.stack(np.broadcast_arrays(np.uint8(r), v[:, None], v[None, :]), -1).reshape(-1, 3)
            x = srgb_decode(tile) * float(G)
            if knee is not None:
                m = x.max(-1)
                m2 = FilamentViewSession.rolloff_curve(m, float(knee), float(white))
                x = x * (m2 / np.maximum(m, 1e-12))[..., None]
            out[r * 65536:(r + 1) * 65536] = srgb_encode(agx(x * g, mode)).reshape(-1, 3)
        _LUTT[key] = out
    return _LUTT[key]


def _lut_gain(look, f):
    """the display table at the look's gain x f (one extra table per factor, kept for the process)"""
    from render.filament.agx import regrade
    g = round(float(look.post_tonemap_gain) / float(look.exposure_scale) * f, 9)
    mode = "punchy" if look.post_tonemap == "agx_punchy" else "none"
    key = (g, mode)
    if key not in _LUTG:
        v = np.arange(256, dtype=np.uint8)
        out = np.empty((256 ** 3, 3), np.uint8)
        for r in range(256):
            tile = np.stack(np.broadcast_arrays(np.uint8(r), v[:, None], v[None, :]), -1).reshape(-1, 3)
            out[r * 65536:(r + 1) * 65536] = regrade(tile, g, mode).reshape(-1, 3)
        _LUTG[key] = out
    return _LUTG[key]


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


class _SharedModelDepth:
    """An optional lever (not the given layer's behaviour): the depth / segmentation session on the Filament model itself
    (the scene plus the room's own textured geoms: the same geometry), so the scene is compiled once; segmentation ids
    are then the view model's geom ids"""

    def __new__(cls, backend, scene, model):
        import mujoco
        from render.backends.mujoco_backend import MujocoSession
        from render.backends import mujoco_scene as MS
        from render.provenance import scene_content_digest
        s = MujocoSession.__new__(MujocoSession)
        s.backend, s.scene, s._mj = backend, scene, mujoco
        ext = float(model.stat.extent)
        model.vis.map.znear = scene.near_m / ext
        model.vis.map.zfar = scene.far_m / ext
        data = mujoco.MjData(model)
        mujoco.mj_kinematics(model, data)
        s.compiled = MS.CompiledScene(model=model, data=data, spec=None, labels=MS._labels(model, {"scale_refs": {}}),
                                      info={"scale_refs": {}, "shared_with_filament": True})
        s.model, s.data, s.labels = model, data, s.compiled.labels
        s._renderers = {}
        s._opt = mujoco.MjvOption()
        s._opt.geomgroup[:] = 0
        for g in scene.visible_groups:
            s._opt.geomgroup[g] = 1
        s._opt.sitegroup[:] = 0
        s._opt.frame = mujoco.mjtFrame.mjFRAME_NONE
        s._opt.label = mujoco.mjtLabel.mjLABEL_NONE
        s._free = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(model, s._free)
        s.gl_facts = {}
        s._digest = None
        s._ray_scale = {}
        s.one_pass = True
        s._depth_geoms = s._drawn_geom_bounds()
        s.transparent_geoms = s._transparent_drawn_geoms()
        s.last_one_pass = []
        s.scene_content = scene_content_digest(scene)
        return s


def _install_depth_cull(ds):
    """An optional lever: before each depth / segmentation render, the drawn geoms whose bounding sphere (geom_xpos,
    geom_rbound; planes by their half sizes) lies wholly outside the camera's frustum go to group 5 (not drawn) and come
    back after it: a geom outside the frustum covers no pixel, so depth and segmentation do not change"""
    import types
    orig = ds.render_raw_views
    m = ds.model
    ids_r = ds._drawn_geom_bounds()
    stats = {"frames": 0, "culled": []}
    ds.cull_stats = stats

    def rrv(self, cameras, outputs=("rgb",)):
        out = []
        for cam in cameras:
            if ids_r is None:
                out += orig([cam], outputs)
                continue
            ids, r = ids_r
            R = np.asarray(cam.R, float)
            P = np.asarray(self.data.geom_xpos, float)[ids] - np.asarray(cam.position, float)[None, :]
            q = P @ R                                      # camera frame (x right, y down, z forward)
            W, Hh = cam.width, cam.height
            lx, rx = (-0.5 - cam.cx) / cam.fx, (W - 0.5 - cam.cx) / cam.fx
            ty, by = (-0.5 - cam.cy) / cam.fy, (Hh - 0.5 - cam.cy) / cam.fy
            out_s = np.zeros(len(ids), bool)
            for nx, nz in ((1.0, -rx), (-1.0, lx)):          # planes x - rx z <= 0 and -x + lx z <= 0 (unit normals)
                nn = math.hypot(nx, nz)
                out_s |= (nx * q[:, 0] + nz * q[:, 2]) / nn > r
            for ny, nz in ((1.0, -by), (-1.0, ty)):
                nn = math.hypot(ny, nz)
                out_s |= (ny * q[:, 1] + nz * q[:, 2]) / nn > r
            out_s |= q[:, 2] < -r                            # wholly behind the camera
            hide = ids[out_s]
            g0 = np.asarray(m.geom_group)[hide].copy()
            m.geom_group[hide] = 5
            try:
                out += orig([cam], outputs)
            finally:
                m.geom_group[hide] = g0
            stats["frames"] += 1
            stats["culled"].append(int(len(hide)))
        return out
    ds.render_raw_views = types.MethodType(rrv, ds)


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
        del TRACE[:]
        _trace("session_start", t0)
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
        self._shared_depth = os.environ.get("ARB_FILAMENT_SHARED_DEPTH", "") == "1"   # an optional lever: no second compile
        self._cull_depth = os.environ.get("ARB_FILAMENT_CULL_DEPTH", "1") == "1"      # INSTALLED (default on; =0 off)
        self._sc = sc
        if not self._shared_depth:
            try:
                self.depth = MujocoBackend.open(backend, sc)   # depth, labels, scene digest of the twin
            except Exception:
                _SPECS.pop(key, None)
                raise
            _trace("depth_twin_open", t0)
            self.scene, self.labels, self.scene_content = self.depth.scene, self.depth.labels, self.depth.scene_content
            if os.environ.get("ARB_FILAMENT_DROP_DEPTH_SPECS", "1") == "1":
                # an optional lever (exact): once the depth twin is compiled, its two spec copies (the registered one and
                # the compiled scene's) are not used again (nothing reads CompiledScene.spec; _depth_scene runs at open
                # only): drop them and hand the heap back; the depth model itself is unchanged
                _SPECS.pop(key, None)
                del dspec
                try:
                    self.depth.compiled.spec = None
                except Exception:  # noqa: BLE001
                    pass
                gc.collect()
                try:
                    import ctypes
                    ctypes.CDLL("libc.so.6").malloc_trim(0)
                except Exception:  # noqa: BLE001
                    pass
                _trace("depth_specs_dropped", t0)
        else:
            _SPECS.pop(key, None)
            del dspec
            _trace("depth_twin_skipped", t0)
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
                if not (l.name or "").startswith(("room_light_",) + KEEP_LIGHT_PREFIXES):
                    fspec.delete(l)
        self.look_record = apply_look(fspec, self.look)
        _trace("filament_spec_built", t0)
        self.model = fspec.compile()
        del fspec
        _trace("filament_model_compiled", t0)
        if os.environ.get("ARB_FILAMENT_TRIM", "1") == "1":       # INSTALLED: hand the freed heap back (malloc_trim)
            gc.collect()
            try:
                import ctypes
                ctypes.CDLL("libc.so.6").malloc_trim(0)
            except Exception:  # noqa: BLE001
                pass
            _trace("heap_trimmed", t0)
        if self._shared_depth:
            self.depth = _SharedModelDepth(backend, sc, self.model)
            self.scene, self.labels, self.scene_content = self.depth.scene, self.depth.labels, self.depth.scene_content
            _trace("depth_on_filament_model", t0)
        if self._cull_depth:
            _install_depth_cull(self.depth)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_camlight(self.model, self.data)
        self.near, self.far = float(near_m), float(far_m)
        self._jmap = None
        self.ctx = _context(api or os.environ.get("ARB_FILAMENT_API", "vulkan"))
        _trace("filament_context", t0)
        self.objs = mjrf.ModelObjects(self.ctx, self.model)
        _trace("filament_model_objects", t0)
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
        _trace("filament_scene_renderables_ibl", t0)
        self.lights.update(self.data)
        self.rends.update(self.data)
        self._targets = {}
        self._lut_on = bool(self.look.post_tonemap)       # the display table is built on its first read
        _trace("display_lut", t0)
        self.room_record = rec or {}
        self.t_open_s = time.perf_counter() - t0
        self._digest = None
        _OPEN.add(self)
        self._update()      # the model's own pose (qpos0) until a state is given: bodies, cameras, lights, depth twin

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

    def _scale_exposure(self, f):
        """a camera exposure applied INSIDE the renderer's linear float pipeline, before the 8-bit target: every
        lamp's intensity, every material's emission (the baked room's lightmaps, the slab's complement) and the image
        light are multiplied by f (the compiled model's arrays, then the Filament lights and renderables updated)"""
        m = self.model
        if not hasattr(self, "_expo0"):
            self._expo0 = (m.light_intensity.copy(), m.mat_emission.copy())
        li0, em0 = self._expo0
        m.light_intensity[:] = li0 * float(f)
        m.mat_emission[:] = em0 * float(f)
        if self.ibl is not None:
            if not hasattr(self, "_ibl0"):
                self._ibl0 = float(self.look.room_ibl_ktx_intensity * self.look.exposure_scale)
            self.ibl[0].set_intensity(self._ibl0 * float(f))
        self.lights.update(self.data)
        self.rends.update(self.data)

    def _rolloff_params(self):
        """the wrist camera's tone path of record (any camera rendered with exposure_factor > 1): the highlight
        shoulder's knee and white point ARB_FILAMENT_ROLLOFF (default "0.6,4.0"; "off" = the earlier path: the 8-bit render at the
        exposure through the display table) and the dark-tone bracket ARB_FILAMENT_BRACKET (default 4)"""
        if not hasattr(self, "_rolloff"):
            s = os.environ.get("ARB_FILAMENT_ROLLOFF", "0.6,4.0").strip()
            self._rolloff = None if s.lower() in ("", "off", "0", "none") else tuple(float(x) for x in s.split(","))
            self.rolloff_stats = {"frames": 0, "second_renders": 0}
        return self._rolloff

    @staticmethod
    def rolloff_curve(m, k, xw):
        """the shoulder on the brightest channel m of the scene-linear frame at the camera's exposure:
        m' = m for m <= k; m' = k + (1 - k) t (1 + t / tw^2) / (1 + t) for m > k, t = (m - k) / (1 - k),
        tw = (xw - k) / (1 - k): slope 1 at the knee k, m' = 1 at the white point xw (above it 1)"""
        t = np.maximum(m - k, 0.0) / (1.0 - k)
        tw = (xw - k) / (1.0 - k)
        y = k + (1.0 - k) * t * (1.0 + t / (tw * tw)) / (1.0 + t)
        return np.where(m <= k, m, np.minimum(y, 1.0))

    def _tone_path(self, ef):
        """the tone path of a frame of exposure ef (the camera's exposure in the renderer's linear pipeline: 1 = the
        overhead camera, 4 = the wrist camera): self.tone_override (a dict; measurement only), else ARB_FILAMENT_TONE (JSON
        {"<ef>": {...}}; "legacy" = no one-render path: the two-render or the earlier path by
        ARB_FILAMENT_ROLLOFF), else TONE_OF_RECORD. A path:
          {"mode": "one", "in": F, "knee": k, "white": w}: ONE render with the factor F inside the renderer (the rest of
              the exposure, ef / F x gain_factor, applied in float), the hue-keeping shoulder rolloff_curve(m, k, w) on
              the brightest channel when k is given, the display transform (AgX) in float, one 8-bit quantisation; done
              as a table over the 2^24 8-bit render triples (built once per path and process: the same values as the
              float arithmetic, at the cost of an index lookup per frame);
          {"mode": "table"}: ONE render at ef, the display table (the earlier path; "one" with F = ef and no shoulder);
          {"mode": "two_renders"}: the wrist camera's two renders (ARB_FILAMENT_ROLLOFF / ARB_FILAMENT_BRACKET);
          {"mode": "raw"}: ONE render at ef, the 8-bit render itself (a timing reference; the frame of a renderer-side tone
              mapper)."""
        ov = getattr(self, "tone_override", None)
        if ov is not None:
            return ov
        if not hasattr(self, "_tone_cfg"):
            s = os.environ.get("ARB_FILAMENT_TONE", "").strip()
            self._tone_cfg = None if s == "legacy" else (json.loads(s) if s else dict(TONE_OF_RECORD))
        if self._tone_cfg is None:
            return None
        return self._tone_cfg.get("%g" % ef)

    @property
    def lut(self):
        """the look's display table (_lut, 2^24 entries), built on its first
        READ and kept for the process (_LUTS); the tone paths of record never read it (the overhead camera's one-render
        path reads _lut_tone at G = 1/4, the wrist camera's path works in float), so their sessions never build it"""
        return _lut(self.look) if self._lut_on else None

    def _rgb_one_render(self, cam, rq, rd, ef, gain_factor, raw_out, tp):
        """the one-render tone paths (_tone_path)"""
        mode = tp.get("mode")
        F = ef if mode in ("table", "raw") else float(tp.get("in", ef))
        raw = self._render_once(cam, rq, rd, F)
        if raw_out is not None:
            raw_out.append(raw.copy())
        if mode == "raw" or not self._lut_on:
            return raw
        G = ef / F * gain_factor
        k = tp.get("knee") if mode == "one" else None
        w = tp.get("white") if mode == "one" else None
        if k is None and G == 1.0:
            lut = self.lut
        else:
            lut = _lut_tone(self.look, G, None if k is None else float(k), None if w is None else float(w))
        idx = (raw[..., 0].astype(np.int32) << 16) | (raw[..., 1].astype(np.int32) << 8) | raw[..., 2]
        return lut[idx]

    def _render_once(self, cam, rq, rd, ef):
        if ef != 1.0:
            self._scale_exposure(ef)
        try:
            self.ctx.wait_for_frame(self.ctx.render([rq], [rd]))
        finally:
            if ef != 1.0:
                self._scale_exposure(1.0)
        return np.frombuffer(rd.buffer(), np.uint8).reshape(cam.height, cam.width, 3).copy()

    def _rgb(self, cam, gain_factor=1.0, raw_out=None, exposure_factor=1.0):
        """gain_factor = a per-camera exposure on the display transform's scene-linear input (2 ** EV; the render
        itself is unchanged); raw_out (a list) receives the 8-bit render before the display transform.
        exposure_factor = the camera's exposure in the renderer's LINEAR pipeline (before the 8-bit quantisation):
        the wrist camera's +2 EV of record is exposure_factor 4 with gain_factor 1"""
        from render.filament.backend import gl_camera
        mjrf = self._mjrf
        target, rd = self._target(cam.width, cam.height)
        rq = mjrf.RenderRequest(camera=gl_camera(cam, self.near, self.far), draw_mode=mjrf.DrawMode.DRAW_MODE_DEFAULT)
        rq.scene, rq.target = self.scn, target
        rq.viewport.left, rq.viewport.bottom, rq.viewport.width, rq.viewport.height = 0, 0, cam.width, cam.height
        ef = float(exposure_factor)
        tp = self._tone_path(ef)                               # the tone path of record for this exposure
        if tp is not None and tp.get("mode") != "two_renders":
            return self._rgb_one_render(cam, rq, rd, ef, float(gain_factor), raw_out, tp)
        ro = self._rolloff_params() if (ef > 1.0 or (tp or {}).get("mode") == "two_renders") else None
        rgb = self._render_once(cam, rq, rd, ef) if ro is None else None
        if ro is not None:
            # the wrist camera's tone path in linear FLOAT, quantised once:
            #   x  = the scene-linear frame at the camera's exposure ef, from two renders of the same state: one at
            #        ef x B (ARB_FILAMENT_BRACKET, default 4: the dark tones with B times the 8-bit codes of a render at ef) and,
            #        where that one reaches the top of its 8-bit range, one at exposure 1 (x ef: the highlights up to ef);
            #        blended over the top 15 % of the first one's range;
            #   y  = x scaled by m' / m, m the brightest channel of x and m' = rolloff_curve(m, knee, white) (identity up
            #        to the knee, a shoulder to 1 at the white point: the hue is kept);
            #   display = the look's display transform (AgX, gain post_tonemap_gain / exposure_scale) of y, in float,
            #        then 8 bits; raw_out receives y in 8 bits (sRGB-encoded), the frame the display transform sees
            k, xw = ro
            B = float(os.environ.get("ARB_FILAMENT_BRACKET", "4") or "4")
            self.rolloff_stats["frames"] += 1
            from render.filament.agx import srgb_decode, srgb_encode, agx
            rh = self._render_once(cam, rq, rd, ef * B)
            hi = srgb_decode(rh)
            x = hi / B
            mh = hi.max(-1)
            if (rh.max(-1) >= 250).any():
                lo = srgb_decode(self._render_once(cam, rq, rd, 1.0)) * ef
                self.rolloff_stats["second_renders"] += 1
                w = np.clip((mh - 0.85) / 0.10, 0.0, 1.0)[..., None]
                x = (1.0 - w) * x + w * np.maximum(lo, x)
            m = x.max(-1)
            m2 = self.rolloff_curve(m, k, xw)
            y = x * (m2 / np.maximum(m, 1e-12))[..., None]
            y8 = srgb_encode(y)
            if raw_out is not None:
                raw_out.append(y8.copy())
            if not self._lut_on:
                return y8
            g = float(self.look.post_tonemap_gain) / float(self.look.exposure_scale) * float(gain_factor)
            mode = "punchy" if self.look.post_tonemap == "agx_punchy" else "none"
            return srgb_encode(agx(y * g, mode))
        if raw_out is not None:
            raw_out.append(rgb.copy())
        if not self._lut_on:
            return rgb.copy()
        lut = self.lut if float(gain_factor) == 1.0 else _lut_gain(self.look, float(gain_factor))
        idx = (rgb[..., 0].astype(np.int32) << 16) | (rgb[..., 1].astype(np.int32) << 8) | rgb[..., 2]
        return lut[idx]

    def render_raw_views(self, cameras, outputs=("rgb",), gain_factor=1.0, raw_out=None, exposure_factor=1.0):
        if self.scn is None:
            raise RuntimeError("the view is closed")
        outputs = raw_outputs(outputs)
        extra = [o for o in outputs if o in ("depth", "segmentation")]
        mv = self.depth.render_raw_views(cameras, extra) if extra else [None] * len(cameras)
        out = []
        for cam, m in zip(cameras, mv):
            out.append(RawView(rgb=self._rgb(cam, gain_factor, raw_out, exposure_factor), depth=None if m is None else m.depth,
                               seg=None if m is None else m.seg))
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
        """Free the Filament objects in dependency order, then (no other session open) Filament's context itself
        (_release_context); the context outlives every scene object (and at exit: _shutdown)."""
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
        self.ctx = None
        _release_context()


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
