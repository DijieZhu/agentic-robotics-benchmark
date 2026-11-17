"""The agent's cameras of the final-assembly-and-trim cell: MuJoCo's Filament renderer (physically based, Vulkan; on the
CPU through Mesa's lavapipe when no GPU driver is present, the task runs with gpus = 0) in the baked factory hall of the
earlier assembly cells, unchanged.

Sources (the form ported, rewritten for this cell's scene):
  docs/plan/driving/tasks/racecar_full_assembly_tenth_scale_ur10e/tests/arm/assembly_car/agent_render.py
    sha256 043de96a77f1b5ae346d29ae4819b948cb684df417d9d2636cbe3bbabd889188 (the session, the workbench, the LUT)
  tasks/quadruped/quadruped_full_assembly_12dof_ur10e/tests/arm/assembly_quad/agent_render.py (arb-main-20261113)
    sha256 26e8d92a31f6f7cfe844599715f9aff2857908f745216e127a430a8dd4d1e204

One session per episode. The render model is the episode's render twin (`render_twin.render_spec`: the cell from the
kit's visual meshes, the same joint names as the physics cell, the datum boards) without the MuJoCo-only backdrop,
with the transparent classes' agent-view tint (agent_look.json kit_materials.transparent_view), plus the robot table's
steel frame under the 2.0 x 1.4 m top, the arm's control box with its teach pendant and the hazard tape on the floor,
the factory room bundle (`room.add_room`: baked lightmaps on the hall, its lamps lighting the live objects, the room's
HDR image light) and the look (`agent_look.json`: the hall's calibrated light units, exposure and filmic display
transform; PBR terms per material class: the vehicle's from the digest's palette classes, the stations', the arm's,
the boards'). It is compiled once; every frame copies the physics state (`render_twin.sync_map`: by joint name, free
bodies by body name), updates the Filament scene and renders RGB as an exact off-axis pinhole, then the filmic display
transform through an 8-bit lookup table.
Depth comes from a MuJoCo session (`render_twin.TwinSession`) on the same geometry plus the room's meshes, every
material opaque there (so the colour and depth passes are one), with the no-return rule of the transparent classes
(the clear acrylic tubes and the dome: depth 0 where they are the first surface; render_twin docstring). Lens
distortion, noise, blur, vignetting, colour jitter and the depth-sensor model stay in render/postprocess.py, the same
code as the MuJoCo path. The episode's drawn key-light factor scales the exposure (+-20 % between episodes).

Assets: $ARB_AGENT_RENDER_DIR (the grader: the root-owned copy of the payload, /opt/arb/render) holds `room/` (an
arb-room/1 bundle: room.json, meshes, baked textures, ibl/ibl.ktx) and `oak/` (the bench top's scanned maps).

The native renderer's log: every frame Filament writes one line per renderable whose vertex buffer lacks an attribute
its material declares ("[instance=N, primitive @ 0] missing required attributes (0xb), declared=0x3", 77 lines a frame
in this scene) straight to the process's file descriptors, which drowned the verifier's log (record P2_P15 5-5).
`_QuietNative` takes the process's stdout and stderr (file descriptors 1 and 2) aside while Filament renders a frame
and writes everything back afterwards except those lines (ruling of group Q, P18a step 6). The C library's stdio
buffers are flushed on both sides of the block (`fflush(NULL)`): when stdout is not a terminal and Python runs without
-u (the verifier) the native lines sit in the C buffer and would otherwise reach the real descriptor at exit. Nothing of
the scene, the materials or the pixels changes; ARB_FILAMENT_LOG=1 keeps every line.
"""

import atexit
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
from render.backends.base import RenderSession

from . import render_twin as RT

__all__ = ["open_session", "available", "FilamentAgentBackend", "make_look", "look_config", "kit_material_rules", "depth_spec"]

LOOK_FILE = RT.LOOK_FILE
NOMINAL_KEY = 0.62          # the info layout's key-light diffuse: the episode's draw over it scales the exposure
MAX_LIVE_LAMPS = 6          # room lamps nearest the bench that light the live objects (the hall itself is baked)
# the robot table's frame (60 mm tube, rails at two heights), the control box and pendant, the floor tape
LEG, RAIL_Z = 0.06, (-0.10, -0.80)
CONTROL_BOX = {"centre": (-0.95, -0.55), "size": (0.268, 0.475, 0.423)}
PENDANT = (0.30, 0.231, 0.05)
TAPE = {"x": (-1.35, 2.25), "y": (-1.45, 1.45), "w": 0.075, "t": 0.002}
_DROP = ("backdrop_", "bench_leg_", "bench_rail_", "bench_shelf")
_LUTS = {}
_SPECS = {}
_OPEN = set()


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


def _shutdown():
    """At exit: close the open sessions, then drop the Filament context (destroying it with live objects panics)."""
    for s in list(_OPEN):
        s.close()
    gc.collect()
    try:
        from render_filament import backend as FB
        FB._CTX.clear()
    except Exception:  # noqa: BLE001
        pass
    gc.collect()


atexit.register(_shutdown)


def _render_dir():
    return os.environ.get("ARB_AGENT_RENDER_DIR", "/opt/arb/render")


def _room_dir():
    return os.path.join(_render_dir(), "room")


def _headless():
    """Filament's GLFW-backed resource providers connect to $DISPLAY when one is set (a validation run of an earlier
    cell hung for hours on a stopped X server): the renderer is offscreen, drop the display variables."""
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


def make_look(exposure_factor=1.0):
    from render_filament.backend import Look
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
        (None, "mat_bench", dict(roughness=1.0, metallic=0.0)),          # x the oak's roughness map
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
    oak = os.path.join(_render_dir(), "oak")
    look.surface_dir = oak if os.path.isdir(oak) else None
    ibl = os.path.join(_room_dir(), "ibl", "ibl.ktx")
    look.room_ibl_ktx = ibl if os.path.isfile(ibl) else None
    look.exposure_scale = float(look.exposure_scale) * float(exposure_factor)
    return look


def _workbench(spec):
    """the robot table's steel frame under the top (six legs, two rail levels, a lower shelf, feet), the arm's control
    box with the teach pendant on the floor behind the arm, the hazard tape around the cell"""
    import mujoco
    bench = RT.BENCH
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


def _base_spec(host, transparent_view):
    """The render twin without the MuJoCo-only scenery (backdrop, bench legs, floor plane), with the robot table and
    the props (no room yet)."""
    spec = RT.render_spec(host, scenery=False, transparent_view=transparent_view)
    for g in list(spec.geoms):
        nm = g.name or ""
        if nm == "floor" or nm.startswith(_DROP) or int(g.group) > 2:
            spec.delete(g)
    # Filament draws sites and tendons whatever their group (the tcp marker showed in the wrist frame): invisible here
    for s in spec.sites:
        s.rgba = [0.0, 0.0, 0.0, 0.0]
    for t in spec.tendons:
        t.rgba = [0.0, 0.0, 0.0, 0.0]
    _workbench(spec)
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


def depth_spec(host):
    """The depth twin: the render twin (palette colours) + the room's meshes, every material and geom colour opaque (the
    MuJoCo session then renders colour and depth in one pass; the no-return rule finds the transparent classes by
    their material names)."""
    spec = _base_spec(host, transparent_view=False)
    _room_geometry(spec, _room_dir())
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
    from render_filament.agx import regrade
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
    """RGB from Filament (factory hall), depth / labels from the twin's MuJoCo session on the same geometry."""

    def __init__(self, backend, host, depth_scene_spec, exposure_factor):
        _headless()
        import mujoco
        import mujoco._render_filament as mjrf
        from render_filament.backend import FilamentBackend, _context, apply_look
        from room import add_room
        t0 = time.perf_counter()
        self.backend, self._mj, self._mjrf = backend, mujoco, mjrf
        self.depth = RT.TwinSession(backend, depth_scene_spec)          # depth, labels, no-return rule
        self.scene, self.labels, self.scene_content = self.depth.scene, self.depth.labels, self.depth.scene_content
        self.look = make_look(exposure_factor)
        self.look.post_tonemap_gain = float(self.look.post_tonemap_gain) * float(exposure_factor)
        spec = _base_spec(host, transparent_view=True)
        bc = RT.BENCH["centre_xy"]
        rec = add_room(spec, _room_dir(), bench_xy=(float(bc[0]), float(bc[1])))
        lamps = sorted([l for l in spec.lights if (l.name or "").startswith("room_light_")],
                       key=lambda l: math.hypot(l.pos[0] - float(bc[0]), l.pos[1] - float(bc[1])))
        for l in lamps[MAX_LIVE_LAMPS:]:
            spec.delete(l)
        for l in lamps[:MAX_LIVE_LAMPS]:
            l.range = 40.0           # the lamps hang ~5.8 m above the bench top (MuJoCo's default range is 10 m)
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
        from render_filament.backend import gl_camera
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
                                   notes="Filament RGB in the baked factory hall; depth / segmentation from the MuJoCo twin",
                                   one_pass_rgbd=False, lod_levels=c.lod_levels)

    def environment(self, session=None):
        env = MujocoBackend.environment(self, getattr(session, "depth", None))
        env["renderer"] = "filament"
        env["filament_api"] = os.environ.get("ARB_FILAMENT_API", "vulkan")
        return env

    def open_agent(self, host, depth_scene_spec, exposure_factor):
        return FilamentAgentSession(self, host, depth_scene_spec, exposure_factor)


def open_session(host):
    """The episode's agent render session: Filament in the factory hall (depth from the MuJoCo twin on the same geometry)."""
    from render.spec import SceneSpec
    key = "rov-episode-%d-%d" % (id(host), int(host.ep_seed) & 0xFFFFFFFF)
    _SPECS.clear()
    _SPECS[key] = depth_spec(host)
    ph = host.cfg["photo"]
    sc = SceneSpec(builder="arm.assembly_rov.agent_render:depth_scene", builder_args={"key": key},
                   content_files=RT._content_files(host) + [os.path.join(_room_dir(), "room.json")],
                   shadows=False, lod=str(ph.get("lod", "full")), near_m=0.01)
    be = FilamentAgentBackend()
    f = float(host.episode["lights"]["key"]) / NOMINAL_KEY
    return be.open_agent(host, sc, f)
