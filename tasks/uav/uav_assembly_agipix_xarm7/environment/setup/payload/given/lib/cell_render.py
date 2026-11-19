"""The renderer of the cell's cameras, for your own twin of the cell: MuJoCo's Filament renderer (physically based,
Vulkan; on the CPU through Mesa's lavapipe) in the baked workshop hall of `render/`, with the cell's look. This is the
code the cell renders its camera frames with; what differs between workcells (where the parts and the fixture are, the
cameras' mounts and focal lengths) comes from your twin.

    import cell_render
    session = cell_render.open_session(spec)      # your mujoco.MjSpec (or MJCF path) of the cell, in the world frame
    session.sync_from(model, data)                # your physics state, copied by joint name
    frame = session.capture("top")                # {"rgb": uint8 960 x 1280 x 3, "depth_m": float32 960 x 1280}
    raw = session.render_raw_views([cam], ("rgb", "depth"))[0]    # any render.spec.CameraSpec: raw.rgb, raw.depth
    session.close()

The render model is a copy of your spec (never your spec itself) without its geoms in groups above 2 (MuJoCo's
collision group 3 and up), its plane geoms (the hall has its own floor) and the meshes no remaining geom uses; the
box geom named `table` (the bench top, argument `table`) gets the scanned oak maps of `render/oak`; the bench's steel
frame (four legs and two rails under the 1.6 x 1.5 m top), the hall (`render.room.add_room`: Cycles-baked lightmaps,
the room's HDR image light `render/room/ibl/ibl.ktx`) and the look are added. The six hall lamps nearest the bench
origin light the live objects (the hall itself is lit by its lightmaps); every other light of your scene is removed.
The look (`make_look`): light units, exposure, the AgX display transform, physically based terms per material class:
the kit's roughness / metallic per visual region (`given/kit/kit.json`, `visual_parts`) for geoms named
`visual_<label>_<i>` and `visual_<label>_rotor_<i>`, then `visual_*`, the fixture box `fixture`, the staging stands
`stand_*`, the xArm's materials `white` and `gray`, and a default. `parts` maps each label to its kit part kind;
by default a label that is a kind, or a kind followed by `_<number>`, is that kind.

Cameras: `capture(name)` renders "top" (1280 x 960) or "wrist" (640 x 480) the cell's way: the camera of that name in
the model you last synced (else the session's own copy), its mount (`cam_pos`, `cam_quat`) on its body as posed in
the render copy and its vertical field of view (`cam_fovy`), read at every capture; square pixels, the principal point
at the image centre ((W - 1) / 2, (H - 1) / 2). So a commissioning estimate written into your physics model's
`cam_pos` / `cam_quat` / `cam_fovy` is used by the next capture, as the cell reads its own camera draws. Only
body-fixed cameras. `camera(name)` returns that `render.spec.CameraSpec`; `render_raw_views` takes any CameraSpec
(pixels fx fy cx cy, `world_T_cam` in OpenCV axes: x right, y down, z forward). RGB is the Filament frame through the
display transform (an 8-bit lookup table); the cell sends these bytes as they are. Depth is MuJoCo's renderer (OSMesa)
on exactly the same compiled model and pinhole: metres along the optical axis, 0.0 where nothing is hit;
`sensor_depth` applies what the cell's sensor adds (0.1-4 m, about 3 % missing pixels, millimetres).

One session per scene and one open Filament session per process (close it before opening the next; a second one is
refused): it is compiled once (about 12 s on 8 CPUs for a twin of the whole cell, the display transform's 8-bit lookup
table, about 2 s, included) and holds about 5 GB while open (the process about 6.5 GB); until you give it a state it
shows your twin's own pose (qpos0). Every frame updates the Filament scene to the session's state and renders RGB as an
exact off-axis pinhole: the first frame after Filament's context is made takes about 10-11 s, later ones about 0.7-0.9 s
(1280 x 960) and 0.4-0.6 s (640 x 480) with depth. Closing the last open session (or view) of the process releases
Filament's context and most of its memory, so the next one's first frame takes that long again. What stays adds up: with
LP_NUM_THREADS unset (as in the graded run) about 0.5 GB after the first close and about 0.2-1.2 GB more after each
further open / close cycle, varying from run to run, while an open session renders frame after frame without growing. So
keep one session open for the whole program and sync it to each new state (with LP_NUM_THREADS=8 set before the first
session the growth slows). In the graded run the program shares the sandbox's 24 GB with the cell's own camera session
(given/lib/README.md).

Assets: the scenery folder (`render_dir()`: $ARB_AGENT_RENDER_DIR, else `render/` beside the `given/` folder this
library is in, else ~/render) holds `room/` (an arb-room/1 bundle: room.json, meshes, baked textures, ibl/ibl.ktx) and
`oak/`; the kit's metadata is `given/kit/kit.json` beside this library (`kit_path()`). Build paths into the scenery
from `render_dir()` (open_view's `room` and `surface`), not from ~: in the graded run HOME is not your home.
"""
import gc
import glob
import json
import math
import os
import re
import time
import warnings

import numpy as np

from render.filament.view import _OPEN, _release_context, check_one_session   # the process's Filament sessions

__all__ = ["open_session", "make_look", "available", "render_dir", "kit_path", "sensor_depth", "CellRenderSession",
           "SIZES", "TABLE"]

SIZES = {"top": (1280, 960), "wrist": (640, 480)}      # the cell's cameras (width, height)
TABLE = "table"                                         # the bench top geom: a box, its top at z = 0
LIVE_LAMPS, LAMP_RANGE_M = 6, 40.0                      # hall lamps nearest the bench origin lighting the live objects
NEAR_M, FAR_M = 0.01, 80.0                              # the clip planes of both image streams
_HERE = os.path.dirname(os.path.abspath(__file__))
_LUT = {}


def _root():
    """The folder holding this library's `given/` folder (your home in the session, the cell's copy of the givens in
    the graded run)."""
    return os.path.dirname(os.path.dirname(_HERE))


def render_dir():
    """The cameras' scenery: $ARB_AGENT_RENDER_DIR, else `render/` beside the `given/` folder this library is in, else
    ~/render."""
    if os.environ.get("ARB_AGENT_RENDER_DIR"):
        return os.environ["ARB_AGENT_RENDER_DIR"]
    beside = os.path.join(_root(), "render")
    if os.path.isfile(os.path.join(beside, "room", "room.json")):
        return beside
    return os.path.expanduser("~/render")


def kit_path():
    """`given/kit/kit.json` beside this library, else ~/given/kit/kit.json."""
    beside = os.path.join(_root(), "given", "kit", "kit.json")
    return beside if os.path.isfile(beside) else os.path.expanduser("~/given/kit/kit.json")


def _headless_cpu():
    """Offscreen on the CPU, as the cell renders: no display (a display server that never answers would block
    Filament) and Mesa's lavapipe as the Vulkan driver."""
    for key in ("DISPLAY", "WAYLAND_DISPLAY"):
        os.environ.pop(key, None)
    candidates = sorted(glob.glob("/usr/share/vulkan/icd.d/lvp_icd*.json"))
    if not candidates:
        raise RuntimeError("Filament CPU rendering requires Mesa lavapipe (mesa-vulkan-drivers)")
    os.environ["VK_ICD_FILENAMES"] = candidates[0]
    os.environ["VK_DRIVER_FILES"] = candidates[0]


def available(scenery=None):
    """(ok, reason): the Filament module imports, lavapipe is installed and the room bundle is present."""
    try:
        _headless_cpu()
        import mujoco._render_filament  # noqa: F401
    except Exception as e:  # noqa: BLE001
        return False, "no Filament CPU renderer: %s" % e
    room = os.path.join(scenery or render_dir(), "room")
    for path in (os.path.join(room, "room.json"), os.path.join(room, "ibl", "ibl.ktx")):
        if not os.path.isfile(path):
            return False, "missing %s" % path
    return True, ""


def _kit(kit=None):
    if isinstance(kit, dict):
        return kit
    path = os.path.expanduser(str(kit)) if kit is not None else kit_path()
    if kit is None and not os.path.isfile(path):     # no kit metadata here: the generic material rules only
        warnings.warn("no kit metadata at %s: the parts get the generic material rules" % path, RuntimeWarning,
                      stacklevel=3)
        return {"parts": {}}
    with open(path) as f:
        return json.load(f)


def _labels(spec, kit):
    """{label: kind} of the geoms named visual_<label>_<i> / visual_<label>_rotor_<i>, in the spec's geom order."""
    kinds = kit.get("parts", {})
    out = {}
    for g in spec.geoms:
        nm = g.name or ""
        m = re.match(r"^visual_(.+)_rotor_\d+$", nm) or re.match(r"^visual_(.+)_\d+$", nm)
        if not m or m.group(1) in out:
            continue
        label = m.group(1)
        kind = label if label in kinds else re.sub(r"_\d+$", "", label)
        if kind in kinds:
            out[label] = kind
    return out


def make_look(parts=None, kit=None, spec=None, scenery=None, table=TABLE):
    """The cell's look (a render.filament.backend.Look). `parts` {label: kit part kind} (default: inferred from the
    geom names of `spec`, see the module docstring); `kit` the kit's metadata (a path or a dict, default kit_path());
    `table` the bench top geom that gets the oak maps (with the material render_table_oak)."""
    from render.filament.backend import Look
    kit = _kit(kit)
    if parts is None:
        parts = _labels(spec, kit) if spec is not None else {}
    labels = {"base": "base"}
    labels.update(parts)
    rd = scenery or render_dir()
    look = Look(name="agipix-factory", version="1")
    # the kit's material regions (per visual part); mesh and contact geometry stay independent
    look.materials = []
    for name, kind in labels.items():
        part = kit.get("parts", {}).get(kind)
        if part is None:
            continue
        for i, visual in enumerate(part.get("visual_parts", [])):
            look.materials.append((f"visual_{name}_{i}", None,
                                   dict(roughness=visual.get("roughness", .52),
                                        metallic=visual.get("metallic", 0.))))
        for i, visual in enumerate(part.get("rotor", {}).get("visual_parts", [])):
            look.materials.append((f"visual_{name}_rotor_{i}", None,
                                   dict(roughness=visual.get("roughness", .38),
                                        metallic=visual.get("metallic", .6))))
    look.materials += [
        ("visual_prop_*", None, dict(roughness=.38, metallic=0.)),
        ("visual_*", None, dict(roughness=.52, metallic=0.)),
        ("fixture", None, dict(roughness=.45, metallic=.15)),
        ("stand_*", None, dict(roughness=.55, metallic=0.)),
        (None, "render_table_oak", dict(roughness=1., metallic=0.)),
        (None, "render_bench_steel", dict(roughness=.4, metallic=.7)),
        (None, "white", dict(roughness=.38, metallic=0.)),
        (None, "gray", dict(roughness=.3, metallic=.7)),
        (None, None, dict(roughness=.6, metallic=0.)),
    ]
    # the cell's calibrated baked-room look
    look.exposure_scale = .35
    look.post_tonemap, look.post_tonemap_gain = "agx_punchy", 1.9
    look.room_baked_emission, look.room_baked_light_scale = .51, 6500.
    look.room_ibl_ktx_intensity, look.room_bulb_m = 12000., .6
    look.surface_dir = os.path.join(rd, "oak")
    look.surface_material, look.surface_geom = "render_table_oak", table
    look.surface_albedo_gain = .9
    look.room_ibl_ktx = os.path.join(rd, "room", "ibl", "ibl.ktx")
    return look


def _lut(look):
    """The AgX display transform of every 8-bit RGB triple (256^3 x 3 bytes, built once per process and gain)."""
    from render.filament.agx import regrade
    gain = look.post_tonemap_gain / look.exposure_scale
    if gain not in _LUT:
        v = np.arange(256, dtype=np.uint8)
        out = np.empty((256 ** 3, 3), np.uint8)
        for red in range(256):
            rgb = np.stack(np.broadcast_arrays(np.uint8(red), v[:, None], v[None, :]), -1).reshape(-1, 3)
            out[red * 65536:(red + 1) * 65536] = regrade(rgb, gain, "punchy").reshape(-1, 3)
        _LUT.clear()
        _LUT[gain] = out
    return _LUT[gain]


def _bench_frame(spec):
    """Render-only steel structure below the 1.6 x 1.5 m bench top."""
    import mujoco
    mat = spec.add_material(name="render_bench_steel", rgba=[.26, .29, .32, 1])

    def box(name, pos, size):
        spec.worldbody.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_BOX, pos=pos, size=size, material=mat.name,
                                contype=0, conaffinity=0, group=2)
    for i, x in enumerate((-.73, .73)):
        for j, y in enumerate((-.68, .68)):
            box(f"render_bench_leg_{i}{j}", [x, y, -.52], [.025, .025, .474])
    for i, y in enumerate((-.68, .68)):
        box(f"render_bench_rail_{i}", [0, y, -.82], [.73, .025, .025])


def _as_spec(spec):
    import mujoco
    if isinstance(spec, mujoco.MjSpec):
        return spec.copy()
    path = os.path.expanduser(str(spec))
    if os.path.isfile(path):
        return mujoco.MjSpec.from_file(path)
    raise TypeError("open_session needs a mujoco.MjSpec or the path of an MJCF file, got %r" % (spec,))


def _make_model(spec, look, rd, table):
    import mujoco
    from render.room import add_room
    from render.filament.backend import apply_look
    # collision geoms are excluded identically from both image streams
    for geom in list(spec.geoms):
        if int(geom.group) >= 3 or int(geom.type) == int(mujoco.mjtGeom.mjGEOM_PLANE):
            spec.delete(geom)
    used_meshes = {g.meshname for g in spec.geoms if g.meshname}
    for mesh in list(spec.meshes):
        if mesh.name not in used_meshes:
            spec.delete(mesh)
    table_mat = spec.add_material(name="render_table_oak", rgba=[1, 1, 1, 1])
    for geom in spec.geoms:
        if geom.name == table:
            geom.material = table_mat.name
            geom.rgba = [1, 1, 1, 1]
    _bench_frame(spec)
    rec = add_room(spec, os.path.join(rd, "room"), bench_xy=(0., 0.))
    lamps = sorted([l for l in spec.lights if (l.name or "").startswith("room_light_")],
                   key=lambda l: math.hypot(l.pos[0], l.pos[1]))
    keep = {l.name for l in lamps[:LIVE_LAMPS]}
    for light in list(spec.lights):
        if light.name not in keep:
            spec.delete(light)
        else:
            light.range = LAMP_RANGE_M
    look_record = apply_look(spec, look)
    model = spec.compile()
    # exact common clip planes, independent of the hall's extent
    model.vis.map.znear = NEAR_M / float(model.stat.extent)
    model.vis.map.zfar = FAR_M / float(model.stat.extent)
    model.vis.global_.offwidth = max(w for w, h in SIZES.values())
    model.vis.global_.offheight = max(h for w, h in SIZES.values())
    return model, rec, look_record


class CellRenderSession:
    """A render session of your twin: Filament RGB in the hall, depth from MuJoCo on the same model (module
    docstring)."""

    def __init__(self, spec, look, rd, table=TABLE):
        _headless_cpu()
        check_one_session()                  # one Filament session per process at a time: close the open one first
        import mujoco
        import mujoco._render_filament as filament
        from render.filament.backend import _context, FilamentBackend
        t0 = time.perf_counter()
        self._mj, self._filament = mujoco, filament
        self.scn = self.ibl = self.objs = self.lights = self.rends = None
        self._targets, self._depth = {}, {}
        self._src, self._jmap = None, None
        self.look, self.render_dir = look, rd
        for path in (os.path.join(rd, "room", "room.json"), os.path.join(rd, "room", "ibl", "ibl.ktx")):
            if not os.path.isfile(path):
                raise FileNotFoundError("Required camera rendering asset missing: %s" % path)
        self.model, self.room_record, self.look_record = _make_model(spec, look, rd, table)
        self.data = mujoco.MjData(self.model)
        self.ctx = _context("vulkan")
        try:
            self.objs = filament.ModelResources(self.ctx, self.model)
            self.scn = self.ctx.create_scene(filament.SceneParams())
            self.scn.configure_from_model(self.model)
            self.lights = filament.ModelLights(self.scn, self.objs)
            self.rends = filament.ModelRenderables(self.scn, self.objs)
            self.ibl = FilamentBackend(look=self.look)._ktx_ibl(
                self.ctx, self.scn, self.look.room_ibl_ktx, self.look.room_ibl_ktx_intensity * self.look.exposure_scale)
            self.lut = _lut(self.look)
        except BaseException:
            _OPEN.add(self)
            self.close()
            raise
        _OPEN.add(self)
        self._update()                       # the twin's own pose (qpos0) until a state is given
        self.t_open_s = time.perf_counter() - t0
        self.last_timing = {}

    # ---- state
    def _update(self):
        mj = self._mj
        mj.mj_kinematics(self.model, self.data)
        mj.mj_comPos(self.model, self.data)
        mj.mj_camlight(self.model, self.data)              # camera and light poses (not set by mj_kinematics)
        self.lights.update(self.data)
        self.rends.update(self.data)

    def sync_from(self, model, data):
        """Copy the state of your physics (model, data) by joint name; the cameras are then read from `model`."""
        if self.scn is None:
            raise RuntimeError("Camera session is closed")
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
        self._src = model
        self._update()

    def set_state(self, qpos=None, joints=None):
        """qpos (the full vector of the session's model, `session.model`) and / or joints {name: value} on top."""
        if self.scn is None:
            raise RuntimeError("Camera session is closed")
        mj = self._mj
        if qpos is not None:
            q = np.asarray(qpos, float).ravel()
            if q.size != self.model.nq:
                raise ValueError("qpos has %d values, the session's model nq = %d" % (q.size, self.model.nq))
            self.data.qpos[:] = q
        for name, val in dict(joints or {}).items():
            j = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_JOINT, str(name))
            if j < 0:
                raise ValueError("no joint %r in the session's model" % name)
            n = {0: 7, 1: 4, 2: 1, 3: 1}[int(self.model.jnt_type[j])]
            a = int(self.model.jnt_qposadr[j])
            self.data.qpos[a:a + n] = np.asarray(val, float).ravel()[:n]
        self._update()

    # ---- cameras
    def camera(self, name, width=None, height=None):
        """The CameraSpec of the camera `name` at the session's state, the cell's way (module docstring)."""
        from render.spec import CameraSpec
        src = self._src if self._src is not None else self.model
        mj = self._mj
        source = mj.mj_name2id(src, mj.mjtObj.mjOBJ_CAMERA, str(name))
        if source < 0:
            raise ValueError("Unknown camera %r" % (name,))
        if int(src.cam_mode[source]) != 0:
            raise ValueError("Only body-fixed cameras are supported")
        body = src.body(int(src.cam_bodyid[source])).name
        target = self.model.body(body).id if body else 0
        body_rotation = self.data.xmat[target].reshape(3, 3)
        local_rotation = np.empty(9)
        mj.mju_quat2Mat(local_rotation, src.cam_quat[source])
        rotation = body_rotation @ local_rotation.reshape(3, 3)
        position = self.data.xpos[target] + body_rotation @ src.cam_pos[source]
        pose = np.eye(4)
        pose[:3, :3] = rotation @ np.diag([1., -1., -1.])
        pose[:3, 3] = position
        if width is None or height is None:
            if name not in SIZES:
                raise ValueError("camera %r needs width and height" % (name,))
            width, height = SIZES[name]
        focal = .5 * height / math.tan(math.radians(float(src.cam_fovy[source])) / 2)
        return CameraSpec(name=str(name), width=int(width), height=int(height), fx=focal, fy=focal,
                          cx=(width - 1) / 2, cy=(height - 1) / 2, world_T_cam=pose.tolist())

    # ---- rendering
    def _rgb(self, camera):
        from render.filament.backend import gl_camera
        f = self._filament
        key = camera.width, camera.height
        if key not in self._targets:                     # one target and one readback per image size
            target = self.ctx.create_render_target(f.RenderTargetConfig(color_format=f.PixelFormat.PIXEL_FORMAT_RGB8))
            target.resize(*key)
            pixels = np.empty((camera.height, camera.width, 3), dtype=np.uint8)
            read = f.ReadRequest(target=target)
            read.set_buffer(pixels)
            self._targets[key] = target, read, pixels
        target, read, raw = self._targets[key]
        request = f.RenderRequest(camera=gl_camera(camera, NEAR_M, FAR_M), draw_mode=f.DrawMode.DRAW_MODE_DEFAULT)
        request.scene, request.target = self.scn, target
        request.viewport.left, request.viewport.bottom = 0, 0
        request.viewport.width, request.viewport.height = key
        self.ctx.wait_for_frame(self.ctx.render([request], [read]))
        index = (raw[..., 0].astype(np.int32) << 16) | (raw[..., 1].astype(np.int32) << 8) | raw[..., 2]
        return self.lut[index]

    def _depth_m(self, camera):
        from render.backends.mujoco_backend import set_scene_camera
        mj = self._mj
        key = camera.width, camera.height
        if key not in self._depth:
            self.model.vis.global_.offwidth = max(int(self.model.vis.global_.offwidth), camera.width)
            self.model.vis.global_.offheight = max(int(self.model.vis.global_.offheight), camera.height)
            renderer = mj.Renderer(self.model, height=camera.height, width=camera.width,
                                   max_geom=max(10000, self.model.ngeom + 100))
            renderer.enable_depth_rendering()
            self._depth[key] = renderer
        renderer = self._depth[key]
        named = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_CAMERA, camera.name) >= 0
        renderer.update_scene(self.data, camera=camera.name if named else -1)
        set_scene_camera(renderer.scene, camera, self.model)
        result = renderer.render().copy()
        # no-hit sentinel 0; the sensor's limits, dropouts and quantization: sensor_depth
        result[(result >= 79.99) | ~np.isfinite(result)] = 0.
        return result

    def capture(self, name, model=None, data=None):
        """{"rgb", "depth_m"} of the cell camera `name` ("top" / "wrist"), raw as the cell's renderer gives it; with
        (model, data), sync_from them first."""
        if self.scn is None:
            raise RuntimeError("Camera session is closed")
        started = time.perf_counter()
        if data is not None:
            self.sync_from(model if model is not None else data.model, data)
        camera = self.camera(name)
        rgb = self._rgb(camera)
        middle = time.perf_counter()
        depth = self._depth_m(camera)
        ended = time.perf_counter()
        self.last_timing = dict(camera=name, rgb_s=middle - started, depth_s=ended - middle, total_s=ended - started)
        return {"rgb": rgb, "depth_m": depth}

    def render_raw_views(self, cameras, outputs=("rgb",)):
        """[RawView] (rgb, depth) of render.spec.CameraSpecs at the session's state; outputs "rgb" and / or "depth"."""
        from render.backends.base import RawView
        if self.scn is None:
            raise RuntimeError("Camera session is closed")
        outputs = tuple(outputs)
        bad = [o for o in outputs if o not in ("rgb", "depth")]
        if bad:
            raise ValueError("outputs %s: this session renders rgb and depth" % bad)
        out = []
        for cam in cameras:
            rgb = self._rgb(cam) if "rgb" in outputs else None
            depth = self._depth_m(cam) if "depth" in outputs else None
            out.append(RawView(rgb=rgb, depth=depth))
        return out

    def close(self):
        """Free the Filament objects in dependency order, then (no other session open) Filament's context itself
        (render.filament.view._release_context); the context outlives every scene object (and at exit)."""
        if self.scn is None and self.objs is None:
            return
        _OPEN.discard(self)
        for renderer in self._depth.values():
            renderer.close()
        self._depth.clear()
        if self.ibl is not None and self.scn is not None:
            try:
                self.scn.remove_light(self.ibl[0])
            except Exception:  # noqa: BLE001
                pass
        self.ibl = None
        self._targets.clear()
        self.rends = self.lights = self.scn = self.objs = None
        gc.collect()
        self.ctx = None
        _release_context()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def open_session(spec, parts=None, kit=None, scenery=None, table=TABLE):
    """A render session of your twin `spec` (a mujoco.MjSpec of the cell in its world frame, copied, or an MJCF path):
    Filament in the workshop hall with the cell's look, depth from MuJoCo on the same model.

    parts        {label: kit part kind} for the geoms visual_<label>_<i> / visual_<label>_rotor_<i> (default: inferred)
    kit          the kit's metadata, a path or a dict (default kit_path())
    scenery      the scenery folder with room/ and oak/ (default render_dir())
    table        the name of the bench top box geom (default "table")
    """
    spec = _as_spec(spec)
    rd = os.path.abspath(os.path.expanduser(scenery or render_dir()))
    look = make_look(parts=parts, kit=kit, spec=spec, scenery=rd, table=table)
    return CellRenderSession(spec, look, rd, table=table)


def sensor_depth(depth_m, rng=None, dropout=.03):
    """What the cell's depth sensor adds to a raw depth image: only 0.1-4 m is kept, about 3 % of pixels are missing
    (random), the rest is quantized to millimetres; 0.0 = invalid (float32 metres, as cell_client returns it)."""
    rng = np.random.default_rng() if rng is None else rng
    depth = np.asarray(depth_m)
    good = np.isfinite(depth) & (depth >= .1) & (depth <= 4)
    good &= rng.random(depth.shape) > dropout
    return np.where(good, np.rint(depth * 1000), 0).astype(np.uint16).astype(np.float32) / 1000
