"""The renderer of the cell's cameras, for your own twin of the cell: MuJoCo's Filament renderer (physically based,
Vulkan; on the CPU through Mesa's lavapipe) in the cell's baked workshop with the cell's look. This is the code the cell
renders its camera frames with; what the cell draws per run (the workshop's brightness) is an argument here.

    from arm.assembly_rov import agent_render
    session = agent_render.open_session(spec)          # your mujoco.MjSpec of the cell, in the cell frame (copied)
    session.sync_from(model, data)                     # your physics state (joints by name, free bodies by body name)
    raw = session.render_raw_views([cam], ("rgb", "depth"))[0]    # raw.rgb uint8 H x W x 3, raw.depth float32 metres
    res = session.render([cam], photo=photo, seed=k, outputs=("rgb", "sensor_depth"))   # through render's photo pipeline
    session.close()

`cam` is a render.spec.CameraSpec (pixels fx fy cx cy, world_T_cam in OpenCV axes: x right, y down, z forward; lens
distortion only through the photo pipeline). One session per scene and one open session per process (close it before
opening the next): it is compiled once (the filmic display transform's 8-bit lookup table included) and holds several
GB while open; `cell_api.md`, "Rendering your own frames", has the times and the memory on this machine. Until you
give it a state it shows your twin's own pose (qpos0). Every frame copies the state (`twin_session.sync_map`: by joint
name, a free joint not found by name through its body's name), updates the Filament scene and renders RGB as an exact
off-axis pinhole (the MuJoCo backend's frustum), then the display transform through that table. Closing the last open
session (or view) of the process releases Filament's context and its memory, so the next one's first frame is slow
again.

The render model is your MjSpec without the MuJoCo-only scenery (geoms named `floor`, `backdrop_*`, `bench_leg_*`,
`bench_rail_*`, `bench_shelf`, and every geom in a group above 2; sites and tendons drawn invisible; your lights
removed), plus the robot table's steel frame under the 2.0 x 1.4 m top, the arm's control box with its teach pendant
and the hazard tape on the floor around the cell, the workshop's room bundle (`render.room.add_room`: Cycles-baked
lightmaps, its lamps lighting the live objects, its HDR image light) and the look (`agent_look.json` and `make_look`:
light units, exposure, the filmic display transform, physically based terms per material class). A material named
`mat_vis_<class>` (`twin_session.material_name`) for a class of the look's `kit_materials` palette takes that class's
terms; the classes that the look's `transparent_view` names are drawn in its colour; the bench top geom `table_top` of
material `mat_bench` gets the scanned wood maps. Depth comes from a MuJoCo session (`twin_session.TwinSession`) on the
same geometry plus the room's meshes, every material opaque there, with the cell's no-return rule: depth 0 where the
first surface on a pixel's ray is a geom of a class with alpha below 1 in the palette. Lens distortion, noise, blur,
vignetting, colour jitter and the depth-sensor model are render/postprocess.py (PhotoSpec), the same code as the MuJoCo
path. `exposure_factor` scales the exposure (the cell's workshop is up to about +-20 % brighter or darker
from run to run).

Assets: the scenery folder (`_render_dir()`: $ARB_AGENT_RENDER_DIR, else `render` next to $ARB_GIVEN_DIR, else
~/render) holds `room/` (an arb-room/1 bundle: room.json, meshes, baked textures, ibl/ibl.ktx) and `oak/` (the bench
top's scanned maps). Vulkan: Mesa's lavapipe on the CPU unless a GPU driver is present.

The native renderer's log: every frame Filament writes one line per renderable whose vertex buffer lacks an attribute
its material declares ("[instance=N, primitive @ 0] missing required attributes (0xb), declared=0x3") straight to the
process's file descriptors. `_QuietNative` takes the process's stdout and stderr (file descriptors 1 and 2) aside while
Filament renders a frame and writes everything back afterwards except those lines; the C library's stdio buffers are
flushed on both sides of the block. Nothing of the scene, the materials or the pixels changes; ARB_FILAMENT_LOG=1 keeps
every line.
"""

import gc
import json
import math
import os
import sys
import tempfile
import time

import numpy as np

from render.backends.base import BackendCapabilities, RawView, raw_outputs
from render.backends.mujoco_backend import MujocoBackend
from render.filament.view import _OPEN, _release_context, check_one_session   # the process's Filament sessions
from render.backends.base import RenderSession

from . import twin_session as RT

__all__ = ["open_session", "available", "FilamentAgentBackend", "make_look", "look_config", "kit_material_rules", "depth_spec",
           "DEFAULT_BENCH"]

LOOK_FILE = RT.LOOK_FILE
# the cell's robot table (cell_manual.md, "Frame, bench, robot"): the top's size and centre in the cell frame, its
# thickness and the workshop floor's height
DEFAULT_BENCH = {"size_xy": [2.0, 1.4], "centre_xy": [0.45, 0.0], "thickness": 0.03, "floor_z": -0.994}
MAX_LIVE_LAMPS = 6          # room lamps nearest the bench that light the live objects (the room itself is baked)
# the robot table's frame (60 mm tube, rails at two heights), the control box and pendant, the floor tape
LEG, RAIL_Z = 0.06, (-0.10, -0.80)
CONTROL_BOX = {"centre": (-0.95, -0.55), "size": (0.268, 0.475, 0.423)}
PENDANT = (0.30, 0.231, 0.05)
TAPE = {"x": (-1.35, 2.25), "y": (-1.45, 1.45), "w": 0.075, "t": 0.002}
_DROP = ("backdrop_", "bench_leg_", "bench_rail_", "bench_shelf")
_LUTS = {}
_SPECS = {}
_KEY = [0]


NATIVE_NOISE = (b"missing required attributes",)      # the native per-frame log lines dropped by _QuietNative
_LIBC = []


def _c_flush():
    """fflush(NULL): the C library's buffered streams (the native code's stdout / stderr) out to their descriptors"""
    if not _LIBC:
        try:
            import ctypes
            _LIBC.append(ctypes.CDLL(None))
        except Exception:  # noqa: BLE001
            _LIBC.append(None)
    if _LIBC[0] is not None:
        try:
            _LIBC[0].fflush(None)
        except Exception:  # noqa: BLE001
            pass


class _QuietNative(object):
    """Within the block, file descriptors 1 and 2 point at two temporary files; on leaving, their contents go back to
    the real descriptors without the lines that contain one of NATIVE_NOISE (`dropped` counts them). Python's own
    buffered streams and the C library's are flushed before and after; nothing is lost and the order within each
    stream is kept."""

    def __init__(self):
        self.dropped = 0
        self.on = os.environ.get("ARB_FILAMENT_LOG", "") != "1"

    def __enter__(self):
        if not self.on:
            return self
        for s in (sys.stdout, sys.stderr):
            try:
                s.flush()
            except Exception:  # noqa: BLE001
                pass
        _c_flush()
        self._tmp = [tempfile.TemporaryFile(), tempfile.TemporaryFile()]
        self._saved = [os.dup(1), os.dup(2)]
        os.dup2(self._tmp[0].fileno(), 1)
        os.dup2(self._tmp[1].fileno(), 2)
        return self

    def __exit__(self, *exc):
        if not self.on:
            return False
        for s in (sys.stdout, sys.stderr):
            try:
                s.flush()
            except Exception:  # noqa: BLE001
                pass
        _c_flush()
        for fd, saved, tmp in ((1, self._saved[0], self._tmp[0]), (2, self._saved[1], self._tmp[1])):
            os.dup2(saved, fd)
            os.close(saved)
            tmp.seek(0)
            data = tmp.read()
            tmp.close()
            if not data:
                continue
            keep = []
            for line in data.splitlines(True):
                if any(n in line for n in NATIVE_NOISE):
                    self.dropped += 1
                else:
                    keep.append(line)
            out = b"".join(keep)
            while out:
                k = os.write(fd, out)
                out = out[k:]
        return False


def _render_dir():
    """The cameras' scenery: $ARB_AGENT_RENDER_DIR, else the folder `render` next to $ARB_GIVEN_DIR (where the cell runs
    your program), else ~/render."""
    if os.environ.get("ARB_AGENT_RENDER_DIR"):
        return os.environ["ARB_AGENT_RENDER_DIR"]
    g = os.environ.get("ARB_GIVEN_DIR")
    if g and os.path.isdir(os.path.join(os.path.dirname(os.path.abspath(g)), "render")):
        return os.path.join(os.path.dirname(os.path.abspath(g)), "render")
    return os.path.expanduser("~/render")


def _room_dir(render_dir=None):
    return os.path.join(render_dir or _render_dir(), "room")


def _headless():
    """Filament (its GLFW-backed resource providers) connects to $DISPLAY when one is set, and a display server that
    accepts the connection and never answers blocks it for ever. The renderer is offscreen: drop the display variables."""
    for k in ("DISPLAY", "WAYLAND_DISPLAY"):
        os.environ.pop(k, None)


def available():
    """(ok, reason): the Filament module imports and the room bundle is present."""
    _headless()
    try:
        import mujoco._render_filament  # noqa: F401
    except Exception as e:  # noqa: BLE001
        return False, "no Filament module: %s" % type(e).__name__
    if not os.path.isfile(os.path.join(_room_dir(), "room.json")):
        return False, "no room bundle under %s" % _render_dir()
    return True, ""


def look_config():
    """agent_look.json without its notes and its kit_materials block: (the Look fields, kit_materials)"""
    raw = RT.load_look(LOOK_FILE)
    cfg = {k: v for k, v in raw.items() if not k.startswith("_") and k not in ("kit_materials",)}
    return cfg, raw["kit_materials"]


def kit_material_rules(km, keys=None):
    """[(None, "mat_vis_<class>", {roughness, metallic})] for the palette classes: the look's PBR terms, the roughness of
    the opaque classes clamped to km["roughness"] (transparent classes keep theirs)"""
    lo, hi = (float(v) for v in km["roughness"])
    trans = set(RT.transparent_keys(km))
    out = []
    for k in sorted(keys or km["pbr"]):
        p = km["pbr"][k]
        r = float(p["roughness"]) if k in trans else float(np.clip(float(p["roughness"]), lo, hi))
        out.append((None, RT.material_name(k), dict(roughness=round(r, 4), metallic=round(float(np.clip(p["metallic"], 0.0, 1.0)), 4))))
    return out


def make_look(exposure_factor=1.0, render_dir=None):
    from render.filament.backend import Look
    cfg, km = look_config()
    look = Look(name="factory-agent-rov", version="1")
    look.materials = kit_material_rules(km) + [
        ("ballast_box/*", None, dict(roughness=0.55, metallic=0.0)),     # powder-coated box and pocket walls
        ("thruster_tray/*", None, dict(roughness=0.60, metallic=0.0)),   # PP tray, pads and board tab
        ("enclosure_rack/*", None, dict(roughness=0.40, metallic=0.9)),  # anodised V blocks and base
        ("guard_rack_*/*", None, dict(roughness=0.40, metallic=0.9)),    # aluminium shelves and stops
        ("pos_base/*", None, dict(roughness=0.50, metallic=0.6)),        # the positioner: painted welded sections
        ("pos_carrier/*", None, dict(roughness=0.45, metallic=0.6)),
        ("load_cell_*", None, dict(roughness=0.35, metallic=1.0)),       # steel load cells
        (None, "ur/black", dict(roughness=0.45, metallic=0.0)),          # the arm: black rings / boots
        (None, "ur/jointgray", dict(roughness=0.4, metallic=0.0)),       # joint covers
        (None, "ur/linkgray", dict(roughness=0.42, metallic=1.0)),       # satin aluminium tubes
        (None, "ur/urblue", dict(roughness=0.38, metallic=0.0)),         # painted housings
        (None, "ur/g/metal", dict(roughness=0.36, metallic=1.0)),        # gripper: steel / aluminium
        (None, "ur/g/gray", dict(roughness=0.38, metallic=1.0)),
        (None, "ur/g/black", dict(roughness=0.45, metallic=0.0)),
        (None, "ur/g/silicone", dict(roughness=0.85, metallic=0.0)),
        ("ur/*", None, dict(roughness=0.5, metallic=0.0)),               # the wrist camera housing, pads
        (None, "mat_bench", dict(roughness=1.0, metallic=0.0)),          # x the bench maps' roughness map
        (None, "bench_steel", dict(roughness=0.5, metallic=1.0)),        # the table frame
        (None, "bench_shelf", dict(roughness=0.65, metallic=0.8)),
        (None, "bench_foot", dict(roughness=0.85, metallic=0.0)),
        (None, "prop_*", dict(roughness=0.55, metallic=0.0)),            # control box, pendant, tape
        (None, "*_board__*", dict(roughness=0.8, metallic=0.0)),         # printed datum boards
        (None, "tool_plate__*", dict(roughness=0.8, metallic=0.0)),
        (None, None, dict(roughness=0.6, metallic=0.0)),
    ]
    look.surface_material, look.surface_geom, look.surface_map_m = "mat_bench", "table_top", 2.0
    look.surface_albedo_gain = 1.0
    look.override(cfg)
    oak = os.path.join(render_dir or _render_dir(), "oak")
    look.surface_dir = oak if os.path.isdir(oak) else None
    ibl = os.path.join(_room_dir(render_dir), "ibl", "ibl.ktx")
    look.room_ibl_ktx = ibl if os.path.isfile(ibl) else None
    look.exposure_scale = float(look.exposure_scale) * float(exposure_factor)
    return look


def _workbench(spec, bench=None):
    """the robot table's steel frame under the top (six legs, two rail levels, a lower shelf, feet), the arm's control
    box with the teach pendant on the floor behind the arm, the hazard tape around the cell"""
    import mujoco
    bench = bench or DEFAULT_BENCH
    cx, cy = (float(v) for v in bench["centre_xy"])
    hx, hy, ht = float(bench["size_xy"][0]) / 2, float(bench["size_xy"][1]) / 2, float(bench["thickness"])
    zf = float(bench["floor_z"])
    mats = {}
    for name, rgba in (("bench_steel", [0.20, 0.22, 0.24, 1.0]), ("bench_shelf", [0.30, 0.31, 0.32, 1.0]), ("bench_foot", [0.05, 0.05, 0.05, 1.0]),
                       ("prop_cbox", [0.80, 0.81, 0.80, 1.0]), ("prop_cbox_face", [0.66, 0.67, 0.67, 1.0]), ("prop_cbox_handle", [0.08, 0.08, 0.09, 1.0]),
                       ("prop_pendant", [0.78, 0.79, 0.79, 1.0]), ("prop_pendant_screen", [0.02, 0.025, 0.03, 1.0]), ("prop_estop", [0.80, 0.05, 0.04, 1.0])):
        mats[name] = spec.add_material(name=name, rgba=rgba).name
    n = 256
    i, j = np.mgrid[0:n, 0:n]
    tile = np.where((((i + j) % (n // 2)) < (n // 4))[..., None], np.array([247, 190, 18], np.uint8), np.array([22, 22, 22], np.uint8))
    tex = spec.add_texture(name="prop_hazard_tex", type=mujoco.mjtTexture.mjTEXTURE_2D, width=n, height=n, nchannel=3)
    tex.data = np.ascontiguousarray(tile, np.uint8).tobytes()
    wb = spec.worldbody

    def box(name, c, h, mat, yaw=0.0):
        g = wb.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_BOX, size=list(h), pos=list(c), material=mats[mat], contype=0,
                        conaffinity=0, group=RT.VIS_GROUP)
        if yaw:
            g.quat = [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]
    x0, y0, zt = hx - 0.06, hy - 0.06, -ht
    k = 0
    for sx in (-1, 0, 1):
        for sy in (-1, 1):
            p = (cx + sx * x0, cy + sy * y0)
            box("wb_leg_%d" % k, (p[0], p[1], 0.5 * (zt + zf + 0.03)), (LEG / 2, LEG / 2, 0.5 * (zt - zf - 0.03)), "bench_steel")
            box("wb_foot_%d" % k, (p[0], p[1], zf + 0.015), (0.045, 0.045, 0.015), "bench_foot")
            k += 1
    for i, z in enumerate(RAIL_Z):
        for sy in (-1, 1):
            box("wb_rail_x%d_%d" % (i, sy), (cx, cy + sy * y0, z), (x0, LEG / 2, LEG / 2), "bench_steel")
        for sx in (-1, 0, 1):
            box("wb_rail_y%d_%d" % (i, sx), (cx + sx * x0, cy, z), (LEG / 2, y0, LEG / 2), "bench_steel")
    box("wb_shelf", (cx, cy, RAIL_Z[1] + LEG / 2 + 0.006), (x0, y0, 0.006), "bench_shelf")
    bx, by = CONTROL_BOX["centre"]
    sx_, sy_, sz_ = CONTROL_BOX["size"]
    box("prop_cbox", (bx, by, zf + sz_ / 2 + 0.02), (sx_ / 2, sy_ / 2, sz_ / 2), "prop_cbox")
    box("prop_cbox_plinth", (bx, by, zf + 0.01), (sx_ / 2 - 0.02, sy_ / 2 - 0.02, 0.01), "bench_foot")
    box("prop_cbox_door", (bx + sx_ / 2 + 0.002, by, zf + sz_ / 2 + 0.02), (0.002, sy_ / 2 - 0.03, sz_ / 2 - 0.03), "prop_cbox_face")
    box("prop_cbox_handle", (bx + sx_ / 2 + 0.010, by + sy_ / 2 - 0.07, zf + sz_ / 2 + 0.02), (0.008, 0.012, 0.06), "prop_cbox_handle")
    pz = zf + sz_ + 0.02 + PENDANT[2] / 2
    box("prop_pendant", (bx, by, pz), (PENDANT[1] / 2, PENDANT[0] / 2, PENDANT[2] / 2), "prop_pendant")
    box("prop_pendant_screen", (bx + 0.01, by - 0.02, pz + PENDANT[2] / 2 + 0.001), (0.085, 0.115, 0.001), "prop_pendant_screen")
    (tx0, tx1), (ty0, ty1), w, t = TAPE["x"], TAPE["y"], TAPE["w"], TAPE["t"]
    for name, c, length, yaw in (("prop_tape_s", ((tx0 + tx1) / 2, ty0), tx1 - tx0 + w, 0.0), ("prop_tape_n", ((tx0 + tx1) / 2, ty1), tx1 - tx0 + w, 0.0),
                                 ("prop_tape_w", (tx0, (ty0 + ty1) / 2), ty1 - ty0 - w, math.pi / 2), ("prop_tape_e", (tx1, (ty0 + ty1) / 2), ty1 - ty0 - w, math.pi / 2)):
        key = "prop_tape_%d" % round(length / w)
        if key not in mats:
            m = spec.add_material(name=key, rgba=[1.0, 1.0, 1.0, 1.0])
            m.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "prop_hazard_tex"
            m.texrepeat = [float(length / w), 1.0]
            mats[key] = m.name
        box(name, (c[0], c[1], zf + t / 2 + 0.0005), (length / 2, w / 2, t / 2), key, yaw)


def _base_spec(spec, transparent_view, bench=None):
    """A copy of your twin without the MuJoCo-only scenery (backdrop, bench legs, floor plane, geoms in groups above 2,
    the meshes no geom uses any more) and without its lights, the transparent classes in the look's transparent_view
    colour when transparent_view, with the robot table and the props (no room yet)."""
    spec = spec.copy()
    for l in list(spec.lights):
        spec.delete(l)
    if transparent_view:
        tv = RT.kit_materials().get("transparent_view") or {}
        for m in spec.materials:
            for k, v in tv.items():
                if m.name == RT.material_name(k):
                    m.rgba = list(v["rgba"])
    for g in list(spec.geoms):
        nm = g.name or ""
        if nm == "floor" or nm.startswith(_DROP) or int(g.group) > 2:
            spec.delete(g)
    used = set(g.meshname for g in spec.geoms if g.meshname)
    for me in [me for me in spec.meshes if me.name not in used]:
        spec.delete(me)
    # Filament draws sites and tendons whatever their group (the tcp marker showed in the wrist frame): invisible here
    for s in spec.sites:
        s.rgba = [0.0, 0.0, 0.0, 0.0]
    for t in spec.tendons:
        t.rgba = [0.0, 0.0, 0.0, 0.0]
    _workbench(spec, bench)
    return spec


def _room_geometry(spec, bundle):
    """The room's meshes only (no textures, no lamps): the depth twin of the room."""
    import mujoco
    room = json.load(open(os.path.join(bundle, "room.json")))
    mat = spec.add_material(name="room_depth", rgba=[0.5, 0.5, 0.5, 1.0])
    for me in room["meshes"]:
        z = np.load(os.path.join(bundle, me["file"]))
        mesh = spec.add_mesh(name="room_mesh_" + me["name"])
        mesh.uservert = np.asarray(z["v"], np.float64).ravel()
        mesh.userface = np.asarray(z["f"], np.int32).ravel()
        mesh.inertia = mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL
        spec.worldbody.add_geom(name="room_" + me["name"], type=mujoco.mjtGeom.mjGEOM_MESH, meshname=mesh.name,
                                material=mat.name, contype=0, conaffinity=0, group=RT.VIS_GROUP)


def depth_spec(spec, bench=None, render_dir=None):
    """The depth twin: your twin (its own colours) + the room's meshes, every material and geom colour opaque (the
    MuJoCo session then renders colour and depth in one pass; the no-return rule finds the transparent classes by
    their material names)."""
    spec = _base_spec(spec, transparent_view=False, bench=bench)
    _room_geometry(spec, _room_dir(render_dir))
    for m in spec.materials:
        c = list(m.rgba)
        if c[3] < 1.0:
            m.rgba = c[:3] + [1.0]
    for g in spec.geoms:
        c = list(g.rgba)
        if int(g.group) <= 2 and c[3] < 1.0:
            g.rgba = c[:3] + [1.0]
    return spec


def depth_scene(key):
    """SceneSpec builder of the depth twin: a fresh copy of the registered MjSpec."""
    return _SPECS[key].copy()


def _lut(look):
    """The filmic display transform (regrade) of every 8-bit RGB triple (256^3 x 3 bytes, built once per process and gain)."""
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


class FilamentAgentSession(RenderSession):
    """RGB from Filament (the cell's workshop), depth / labels from the twin's MuJoCo session on the same geometry."""

    def __init__(self, backend, twin_spec, depth_scene_spec, exposure_factor, bench=None, render_dir=None):
        _headless()
        import mujoco
        import mujoco._render_filament as mjrf
        from render.filament.backend import FilamentBackend, _context, apply_look
        from render.room import add_room
        t0 = time.perf_counter()
        self.backend, self._mj, self._mjrf = backend, mujoco, mjrf
        self.depth = RT.TwinSession(backend, depth_scene_spec)          # depth, labels, no-return rule
        self.scene, self.labels, self.scene_content = self.depth.scene, self.depth.labels, self.depth.scene_content
        self.look = make_look(exposure_factor, render_dir)
        self.look.post_tonemap_gain = float(self.look.post_tonemap_gain) * float(exposure_factor)
        spec = _base_spec(twin_spec, transparent_view=True, bench=bench)
        bc = (bench or DEFAULT_BENCH)["centre_xy"]
        rec = add_room(spec, _room_dir(render_dir), bench_xy=(float(bc[0]), float(bc[1])))
        lamps = sorted([l for l in spec.lights if (l.name or "").startswith("room_light_")],
                       key=lambda l: math.hypot(l.pos[0] - float(bc[0]), l.pos[1] - float(bc[1])))
        for l in lamps[MAX_LIVE_LAMPS:]:
            spec.delete(l)
        for l in lamps[:MAX_LIVE_LAMPS]:
            l.range = 40.0           # MuJoCo's default range is 10 m
        for l in list(spec.lights):
            if not (l.name or "").startswith("room_light_"):
                spec.delete(l)
        self.look_record = apply_look(spec, self.look)
        self.model = spec.compile()
        self.data = mujoco.MjData(self.model)
        self.near, self.far = 0.01, 80.0
        self._pairs = None
        self.ctx = _context(os.environ.get("ARB_FILAMENT_API", "vulkan"))
        self.objs = mjrf.ModelObjects(self.ctx, self.model)
        self.scn = self.ctx.create_scene(mjrf.SceneParams())
        self.scn.configure_from_model(self.model)
        self.lights = mjrf.ModelLights(self.scn, self.objs)
        self.rends = mjrf.ModelRenderables(self.scn, self.objs)
        self.ibl = None
        if self.look.room_ibl_ktx:
            self.ibl = FilamentBackend(look=self.look)._ktx_ibl(self.ctx, self.scn, self.look.room_ibl_ktx,
                                                                 self.look.room_ibl_ktx_intensity * self.look.exposure_scale)
        self._targets = {}
        self.native_lines_dropped = 0
        self.lut = _lut(self.look) if self.look.post_tonemap else None
        self.room_record = rec
        self.t_open_s = time.perf_counter() - t0
        self._digest = None
        _OPEN.add(self)
        self.sync_from(self.depth.model, self.depth.data)   # the twin's own pose until a state is given

    @property
    def no_return(self):
        return self.depth.no_return

    @no_return.setter
    def no_return(self, v):
        self.depth.no_return = bool(v)

    @property
    def last_no_return_px(self):
        return self.depth.last_no_return_px

    def set_state(self, state=None, qpos=None, joints=None):
        self.depth.set_state(state, qpos, joints)
        self.sync_from(self.depth.model, self.depth.data)

    def sync_from(self, model, data):
        mj = self._mj
        if self._pairs is None or self._pairs[0] is not model:
            self._pairs = (model, RT.sync_map(model, self.model))
        for a, b, n in self._pairs[1]:
            self.data.qpos[b:b + n] = data.qpos[a:a + n]
        mj.mj_kinematics(self.model, self.data)
        mj.mj_camlight(self.model, self.data)             # light_xpos / light_xdir (not set by mj_kinematics)
        self.lights.update(self.data)
        self.rends.update(self.data)
        self.depth.sync_from(model, data)

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
        q = _QuietNative()
        with q:
            self.ctx.wait_for_frame(self.ctx.render([rq], [rd]))
        self.native_lines_dropped += q.dropped
        rgb = np.frombuffer(rd.buffer(), np.uint8).reshape(cam.height, cam.width, 3)
        if self.lut is None:
            return rgb.copy()
        idx = (rgb[..., 0].astype(np.int32) << 16) | (rgb[..., 1].astype(np.int32) << 8) | rgb[..., 2]
        return self.lut[idx]

    def render_raw_views(self, cameras, outputs=("rgb",)):
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
        return {"session": True, "renderer": "filament", "t_open_s": round(self.t_open_s, 2), "room": self.room_record.get("room"),
                "native_log_lines_dropped": int(self.native_lines_dropped)}

    def close(self):
        """Free the Filament objects in dependency order, then (no other session open) Filament's context itself
        (render.filament.view._release_context); the context outlives every scene object (and at exit: _shutdown)."""
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
        _SPECS.clear()
        self.ctx = None
        _release_context()


class FilamentAgentBackend(MujocoBackend):
    """The MuJoCo backend's planning / finishing with Filament RGB (sessions only)."""
    name = "filament"

    def version(self):
        import mujoco
        return "filament-agent-rov/1 mujoco-%s room-%s" % (mujoco.__version__, os.path.basename(os.path.realpath(_room_dir())))

    def capabilities(self):
        c = MujocoBackend.capabilities(self)
        return BackendCapabilities(name=self.name, implemented=True, outputs=c.outputs, native_lens_distortion=False, sessions=True,
                                   max_image_px=c.max_image_px, deterministic="same mujoco build, Vulkan driver and device",
                                   notes="Filament RGB in the cell's baked workshop; depth / segmentation from the MuJoCo twin",
                                   one_pass_rgbd=False, lod_levels=c.lod_levels)

    def environment(self, session=None):
        env = MujocoBackend.environment(self, getattr(session, "depth", None))
        env["renderer"] = "filament"
        env["filament_api"] = os.environ.get("ARB_FILAMENT_API", "vulkan")
        return env

    def open_agent(self, spec, depth_scene_spec, exposure_factor, bench=None, render_dir=None):
        return FilamentAgentSession(self, spec, depth_scene_spec, exposure_factor, bench=bench, render_dir=render_dir)


def open_session(spec, exposure_factor=1.0, bench=None, render_dir=None, content_files=(), lod="full"):
    """A render session of your twin `spec` (a mujoco.MjSpec of the cell in the cell frame; copied, never changed):
    Filament in the cell's workshop, depth from the MuJoCo twin on the same geometry.

    exposure_factor  the workshop's brightness (1.0 = nominal; the cell's own varies by up to about +-20 % between runs)
    bench            the robot table {"size_xy", "centre_xy", "thickness", "floor_z"} (default DEFAULT_BENCH, the cell's)
    render_dir       the scenery folder with room/ and oak/ (default: _render_dir())
    content_files    files or folders whose bytes enter the frames' provenance digest (e.g. your meshes)
    lod              "full" (the meshes as they are) or "policy" (the render layer's decimated meshes, render/lod.py)
    """
    from render.spec import SceneSpec
    check_one_session()                  # one Filament session per process at a time: close the open one first
    _KEY[0] += 1
    key = "twin-%d-%d" % (os.getpid(), _KEY[0])
    _SPECS.clear()
    _SPECS[key] = depth_spec(spec, bench=bench, render_dir=render_dir)
    sc = SceneSpec(builder="arm.assembly_rov.agent_render:depth_scene", builder_args={"key": key},
                   content_files=[str(p) for p in content_files] + [os.path.join(_room_dir(render_dir), "room.json")],
                   shadows=False, lod=str(lod), near_m=0.01)
    be = FilamentAgentBackend()
    return be.open_agent(spec, sc, float(exposure_factor), bench=bench, render_dir=render_dir)
