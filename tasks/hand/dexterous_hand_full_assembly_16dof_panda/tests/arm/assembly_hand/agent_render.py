"""The agent's cameras of the hand assembly cell: MuJoCo's Filament renderer (physically based, Vulkan) in a baked room
(the precedent cells' agent_render.py with this cell's render twin): a hardware lab bay, a steel workbench under the
maple bench top (the room bundle and its look).

One session per episode. The render model is the episode's render twin (`render_twin.render_spec`: the physics MjSpec
with the same bodies and joint names, the units' render meshes (the hidden kit's visual meshes: the authors' solids and
the servos' vendor solids, tessellated from their B-reps), the three datum boards) without the MuJoCo-only scenery
(backdrop walls, bench frame, floor plane, collision geoms), plus a steel workbench frame under the bench top, the room
bundle (`room.add_room`: Cycles-baked lightmaps on the room, its lamps lighting the live objects, the room's HDR image
light) and the look (`agent_look.json` and `make_look`: PBR terms per material class, the bench top's maps, light
units, AgX). It is compiled once; every frame copies the physics state by joint name, updates the Filament scene and
renders RGB as an exact off-axis pinhole (the MuJoCo backend's frustum), then the AgX display transform through an
8-bit lookup table. Depth comes from the MuJoCo backend on the same geometry (room meshes without textures): metric and
registered with the RGB. Lens distortion, noise, blur, vignetting, colour jitter and the depth-sensor model stay in
render/postprocess.py, the same code as the MuJoCo path. The episode's drawn key-light factor scales the exposure (the
room's lighting varies by about +-20 % between episodes, as the MuJoCo lights did).

The grader config's `agent_render` keys: `omit_hidden_pieces` leaves the fully hidden pieces of `render_omit.json`
(beside this file: a piece no ray escapes from when its unit stands alone, as face ranges of the kit's visual mesh
files) out of both the RGB and the depth twin -- whole pieces, none changed; `omit_list_sha256` pins that list;
`live_lamps` room lamps nearest the bench light the live objects and the `shadow_lamps` nearest of them cast shadows.
Filament's PBR material declares a UV0 attribute: every mesh of the RGB twin without texture coordinates (the units'
render meshes, the Panda's OBJ meshes) gets a per-vertex UV0 of zeros (`_uv0`), which is what Filament bound in its
place before, minus the 'missing required attributes (0xb), declared=0x3' line per mesh and draw; the Panda's file
meshes become user meshes with the compiled vertices, normals and faces mapped back to the asset frame, so the compiled
meshes are the same.

Assets: $ARB_AGENT_RENDER_DIR (verify.py: the root-owned copy of the payload, /opt/arb/render) holds `room/` (an
arb-room/1 bundle: room.json, meshes, baked textures, ibl/ibl.ktx) and `bench/` (the bench top's maps). Vulkan: Mesa's
lavapipe on the CPU unless a GPU driver is present.
"""

import atexit
import gc
import hashlib
import json
import math
import os
import time

import numpy as np

from render.backends.base import BackendCapabilities, RawView, RenderSession, raw_outputs
from render.backends.mujoco_backend import MujocoBackend

__all__ = ["open_session", "available", "FilamentAgentBackend", "render_settings", "render_omit"]

LOOK_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "agent_look.json")
OMIT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "render_omit.json")
NOMINAL_KEY = 0.62          # the cell config's key-light diffuse: the episode's draw over it scales the exposure
MAX_LIVE_LAMPS = 6          # room lamps nearest the bench that light the live objects (the room itself is baked)
SHADOW_LAMPS = 2            # of those, the nearest that cast shadows
# the grader config's `agent_render` keys and their defaults (grader_config.json states them)
RENDER_DEFAULTS = {"omit_hidden_pieces": True, "omit_list_sha256": None, "live_lamps": MAX_LIVE_LAMPS, "shadow_lamps": SHADOW_LAMPS}
# the steel workbench under the bench top (1.6 x 1.0 m, top at z = 0, 30 mm thick, floor at z = -0.994)
TOP_HALF, TOP_T, FLOOR_Z, LEG, RAIL_Z = (0.8, 0.5), 0.03, -0.994, 0.04, (-0.10, -0.80)
_DROP = ("backdrop_", "bench_leg_", "bench_apron_", "bench_shelf")
_LUTS = {}
_SPECS = {}
_OPEN = set()


def _shutdown():
    """At exit: close the open sessions, then drop the Filament context (destroying it with live objects panics and
    crashes the interpreter's teardown)."""
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


def _release_context():
    """Between episodes, once no session is open: drop the Filament context too, then hand the freed heap back (glibc's
    malloc_trim). The engine keeps every earlier session's objects while its context lives (the HandleAllocator arena
    warning of the second session): on lavapipe each episode's session left ~2.6 GB resident, so verify's three
    episodes in one process ran towards the task's 16 GB. The next session makes a fresh context."""
    gc.collect()
    try:
        from render_filament import backend as FB
        FB._CTX.clear()
    except Exception:  # noqa: BLE001
        pass
    gc.collect()
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:  # noqa: BLE001
        pass


def _render_dir():
    return os.environ.get("ARB_AGENT_RENDER_DIR", "/opt/arb/render")


def _room_dir():
    return os.path.join(_render_dir(), "room")


def _headless():
    """Filament (its GLFW-backed resource providers) connects to $DISPLAY when one is set: the task image sets DISPLAY=:1
    and during verify the agent account's X server is SIGSTOPped (verify_freeze), so the connection blocked forever
    (an oracle validation hung for 4 h in episode 1). The renderer is offscreen: drop the display variables."""
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


def make_look(exposure_factor=1.0):
    from render_filament.backend import Look
    look = Look(name="hand-agent", version="1")
    look.materials = [
        # the units' materials (render_twin: `vis_<key>` of kit_info.json; the terms are that table's)
        (None, "vis_printed_black_016", dict(roughness=0.6, metallic=0.0)),          # printed parts, 0.16 mm layers
        (None, "vis_printed_black_020", dict(roughness=0.6, metallic=0.0)),          # printed parts, 0.20 mm layers
        (None, "vis_screw_black", dict(roughness=0.35, metallic=0.9)),
        (None, "vis_screw_bright", dict(roughness=0.28, metallic=1.0)),
        (None, "vis_servo_case_metal", dict(roughness=0.4, metallic=0.8)),
        (None, "vis_output_horn_metal", dict(roughness=0.25, metallic=1.0)),
        (None, "vis_bearing_steel", dict(roughness=0.2, metallic=1.0)),
        (None, "vis_pin_steel", dict(roughness=0.15, metallic=1.0)),
        (None, "vis_elastomer_black", dict(roughness=0.7, metallic=0.0)),
        (None, "vis_connector_white", dict(roughness=0.5, metallic=0.0)),
        (None, "vis_board_black", dict(roughness=0.45, metallic=0.0)),
        (None, "vis_board_green", dict(roughness=0.35, metallic=0.0)),
        (None, "vis_board_blue", dict(roughness=0.35, metallic=0.0)),
        (None, "vis_component_black", dict(roughness=0.45, metallic=0.0)),
        (None, "vis_contact_gold", dict(roughness=0.25, metallic=1.0)),
        (None, "vis_metal_nickel", dict(roughness=0.3, metallic=1.0)),
        (None, "vis_plastic_beige", dict(roughness=0.45, metallic=0.0)),
        (None, "vis_terminal_green", dict(roughness=0.45, metallic=0.0)),
        (None, "vis_*", dict(roughness=0.5, metallic=0.0)),
        # the cell's equipment: machined aluminium fixture, trays and nests
        (None, "mat_fixture", dict(roughness=0.4, metallic=1.0)),
        (None, "mat_tray", dict(roughness=0.45, metallic=1.0)),
        (None, "mat_nest", dict(roughness=0.5, metallic=1.0)),
        (None, "panda/*", dict(roughness=0.38, metallic=0.0)),                        # Panda's painted shells
        (None, "mat_bench", dict(roughness=1.0, metallic=0.0)),                       # x the bench maps' roughness map
        (None, "bench_steel", dict(roughness=0.4, metallic=1.0)),                     # powder-coated steel tube
        (None, "bench_shelf", dict(roughness=0.55, metallic=1.0)),                    # galvanised sheet shelf
        (None, "*_board__*", dict(roughness=0.8, metallic=0.0)),                      # printed datum boards
        (None, "hand_plate__*", dict(roughness=0.8, metallic=0.0)),
        (None, None, dict(roughness=0.6, metallic=0.0)),
    ]
    look.surface_material, look.surface_geom, look.surface_map_m = "mat_bench", "table_top", 1.6
    look.surface_albedo_gain = 1.0
    look.override({k: v for k, v in json.load(open(LOOK_FILE)).items() if not k.startswith("_")})
    # the bench top's maps: bench/ (the maple bench top) or oak/
    surf = [p for p in (os.path.join(_render_dir(), k) for k in ("bench", "oak")) if os.path.isdir(p)]
    look.surface_dir = surf[0] if surf else None
    ibl = os.path.join(_room_dir(), "ibl", "ibl.ktx")
    look.room_ibl_ktx = ibl if os.path.isfile(ibl) else None
    look.exposure_scale = float(look.exposure_scale) * float(exposure_factor)
    return look


def _workbench(spec):
    import mujoco
    steel = spec.add_material(name="bench_steel", rgba=[0.32, 0.34, 0.36, 1.0])
    shelf = spec.add_material(name="bench_shelf", rgba=[0.18, 0.19, 0.2, 1.0])
    wb = spec.worldbody
    x0, y0, zt, zf = TOP_HALF[0] - 0.06, TOP_HALF[1] - 0.06, -TOP_T, FLOOR_Z

    def box(name, c, h, mat):
        wb.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_BOX, size=list(h), pos=list(c), material=mat.name, contype=0,
                    conaffinity=0, group=2)
    k = 0
    for sx in (-1, 1):
        for sy in (-1, 1):
            box("wb_leg_%d" % k, (sx * x0, sy * y0, 0.5 * (zt + zf)), (LEG / 2, LEG / 2, 0.5 * (zt - zf)), steel)
            box("wb_foot_%d" % k, (sx * x0, sy * y0, zf + 0.012), (0.04, 0.04, 0.012), shelf)
            k += 1
    for i, z in enumerate(RAIL_Z):
        for sy in (-1, 1):
            box("wb_rail_x%d_%d" % (i, sy), (0.0, sy * y0, z), (x0, LEG / 2, LEG / 2), steel)
        for sx in (-1, 1):
            box("wb_rail_y%d_%d" % (i, sx), (sx * x0, 0.0, z), (LEG / 2, y0, LEG / 2), steel)
    box("wb_shelf", (0.0, 0.0, RAIL_Z[1] + LEG / 2 + 0.009), (x0, y0, 0.009), shelf)
    box("wb_apron", (TOP_HALF[0] - 0.001, 0.0, -TOP_T - 0.03), (0.002, TOP_HALF[1] - 0.02, 0.03), steel)


def render_settings(server):
    """The grader config's `agent_render` keys over RENDER_DEFAULTS."""
    out = dict(RENDER_DEFAULTS)
    out.update((getattr(server, "cfg", None) or {}).get("agent_render") or {})
    return out


def render_omit(server):
    """render_twin's `omit` ({"files": {visual mesh file: [(first face, count), ...]}}) from render_omit.json, or None
    when the config switches it off; the list must match the config's pinned sha256 when one is given"""
    rs = render_settings(server)
    if not rs.get("omit_hidden_pieces"):
        return None
    raw = open(OMIT_FILE, "rb").read()
    want = rs.get("omit_list_sha256")
    if want and hashlib.sha256(raw).hexdigest() != want:
        raise RuntimeError("render_omit.json has sha256 %s; the grader config pins %s" % (hashlib.sha256(raw).hexdigest(), want))
    lst = json.loads(raw.decode("utf-8"))
    return {"files": {f: [(int(a), int(n)) for a, n, _p in v] for f, v in lst["omitted_face_ranges"].items()}}


def _base_spec(server):
    """The render twin without the MuJoCo-only scenery, with the steel workbench (no room yet); the fully hidden pieces
    left out when the config says so (render_twin reads `server.render_omit`; set only while the twin is built, so the
    classic renderer's twin and any review render keep every triangle)."""
    from . import render_twin
    omit = render_omit(server)
    had, prev = hasattr(server, "render_omit"), getattr(server, "render_omit", None)
    if omit is not None:
        server.render_omit = omit
    try:
        spec = render_twin.render_spec(server)
    finally:
        if omit is not None:
            if had:
                server.render_omit = prev
            else:
                del server.render_omit
    for g in list(spec.geoms):
        nm = g.name or ""
        if nm == "floor" or nm.startswith(_DROP) or int(g.group) > 2:
            spec.delete(g)
    _workbench(spec)
    _uv0(spec)
    return spec


def _quat_R(q):
    w, x, y, z = (float(v) for v in q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def _uv0(spec):
    """A per-vertex UV0 of zeros on every mesh without texture coordinates (module docstring); no vertex, normal or face
    changes. A file mesh is replaced by a user mesh of its compiled data in the asset frame (MuJoCo applies mesh_quat /
    mesh_pos to the asset's vertices: asset = R(mesh_quat) v + mesh_pos), so it compiles to the same mesh."""
    m = spec.compile()
    index = {m.mesh(i).name: i for i in range(m.nmesh)}
    rec = {"user": 0, "file": 0}
    for ms in list(spec.meshes):
        i = index.get(ms.name)
        if i is None or int(m.mesh_texcoordnum[i]) > 0:
            continue
        if not ms.file:
            ms.usertexcoord = [0.0] * (2 * (len(ms.uservert) // 3))
            rec["user"] += 1
            continue
        nv, nf = int(m.mesh_vertnum[i]), int(m.mesh_facenum[i])
        va, fa, na = int(m.mesh_vertadr[i]), int(m.mesh_faceadr[i]), int(m.mesh_normaladr[i])
        R, p = _quat_R(m.mesh_quat[i]), np.asarray(m.mesh_pos[i], np.float64)
        V = np.asarray(m.mesh_vert[va:va + nv], np.float64) @ R.T + p
        N = np.asarray(m.mesh_normal[na:na + int(m.mesh_normalnum[i])], np.float64) @ R.T
        F = np.asarray(m.mesh_face[fa:fa + nf], np.int64)
        FN = np.asarray(m.mesh_facenormal[fa:fa + nf], np.int64)
        ms.file = ""                                  # the compiled vertices carry the asset's scale already
        ms.scale = [1.0, 1.0, 1.0]
        ms.uservert = V.ravel().tolist()
        ms.usernormal = N.ravel().tolist()
        ms.userface = F.ravel().tolist()
        ms.userfacenormal = FN.ravel().tolist()
        ms.usertexcoord = [0.0] * (2 * nv)
        ms.userfacetexcoord = F.ravel().tolist()
        rec["file"] += 1
    return rec


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
                                material=mat.name, contype=0, conaffinity=0, group=2)


def depth_scene(key):
    """SceneSpec builder of the depth twin: a fresh copy of the registered MjSpec."""
    return _SPECS[key].copy()


def _lut(look):
    """The AgX display transform of every 8-bit RGB triple (256^3 x 3 bytes, built once per process and gain)."""
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
    """RGB from Filament (the baked lab), depth / labels from a MuJoCo session on the same geometry."""

    def __init__(self, backend, server, depth_scene_spec, exposure_factor):
        _headless()
        import mujoco
        import mujoco._render_filament as mjrf
        from render_filament.backend import FilamentBackend, _context, apply_look
        from room import add_room
        t0 = time.perf_counter()
        self.backend, self._mj, self._mjrf = backend, mujoco, mjrf
        self.depth = MujocoBackend.open(backend, depth_scene_spec)   # depth, labels, scene digest of the twin
        self.scene, self.labels, self.scene_content = self.depth.scene, self.depth.labels, self.depth.scene_content
        self.look = make_look(exposure_factor)
        base_gain = self.look.post_tonemap_gain
        # the exposure factor brightens / darkens the linear render; the display transform keeps its nominal gain
        self.look.post_tonemap_gain = float(base_gain) * float(exposure_factor)
        spec = _base_spec(server)
        rec = add_room(spec, _room_dir(), bench_xy=(0.0, 0.0))
        rs = render_settings(server)
        n_live, n_shadow = int(rs["live_lamps"]), int(rs["shadow_lamps"])
        lamps = sorted([l for l in spec.lights if (l.name or "").startswith("room_light_")], key=lambda l: math.hypot(l.pos[0], l.pos[1]))
        for l in lamps[n_live:]:
            spec.delete(l)
        for i, l in enumerate(lamps[:n_live]):
            l.range = 40.0           # MuJoCo's default range is 10 m; the lamps hang 2-6 m from the bench top
            l.castshadow = bool(l.castshadow) and i < n_shadow    # the nearest n_shadow cast shadows
        self.lamps_record = {"live": len(lamps[:n_live]), "shadow": sum(1 for l in lamps[:n_live] if l.castshadow),
                             "omit_hidden_pieces": bool(rs.get("omit_hidden_pieces"))}
        for l in list(spec.lights):  # the episode's MuJoCo bench lights: the room lights the bench
            if not (l.name or "").startswith("room_light_"):
                spec.delete(l)
        self.look_record = apply_look(spec, self.look)
        self.model = spec.compile()
        self.data = mujoco.MjData(self.model)
        ext = float(self.model.stat.extent)
        self.near, self.far = 0.01, 80.0
        del ext
        # joint map (physics qpos address -> render qpos address, width) is built at the first sync
        self._jmap = None
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
        self.lut = _lut(self.look) if self.look.post_tonemap else None
        self.room_record = rec
        self.t_open_s = time.perf_counter() - t0
        self._digest = None
        _OPEN.add(self)

    def set_state(self, state=None, qpos=None, joints=None):
        self.depth.set_state(state, qpos, joints)
        self.sync_from(self.depth.model, self.depth.data)

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
        self.ctx.wait_for_frame(self.ctx.render([rq], [rd]))
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
                "lamps": dict(self.lamps_record)}

    def close(self):
        """Free the Filament objects in dependency order, then (no other session open) the context itself (_release_context);
        the context must outlive every scene object, so it goes last (and at exit: _shutdown)."""
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
        if not _OPEN and os.environ.get("ARB_FILAMENT_KEEP_CONTEXT", "") != "1":
            _release_context()


class FilamentAgentBackend(MujocoBackend):
    """The MuJoCo backend's planning / finishing with Filament RGB (sessions only)."""
    name = "filament"

    def version(self):
        import mujoco
        return "filament-agent/1 mujoco-%s room-%s" % (mujoco.__version__, os.path.basename(os.path.realpath(_room_dir())))

    def capabilities(self):
        c = MujocoBackend.capabilities(self)
        return BackendCapabilities(name=self.name, implemented=True, outputs=c.outputs, native_lens_distortion=False, sessions=True,
                                   max_image_px=c.max_image_px, deterministic="same mujoco build, Vulkan driver and device",
                                   notes="Filament RGB in the baked lab room; depth / segmentation from the MuJoCo backend",
                                   one_pass_rgbd=False, lod_levels=c.lod_levels)

    def environment(self, session=None):
        env = MujocoBackend.environment(self, getattr(session, "depth", None))
        env["renderer"] = "filament"
        env["filament_api"] = os.environ.get("ARB_FILAMENT_API", "vulkan")
        return env

    def open_agent(self, server, depth_scene_spec, exposure_factor):
        return FilamentAgentSession(self, server, depth_scene_spec, exposure_factor)


def open_session(server):
    """The episode's agent render session: Filament in the baked lab room (depth from MuJoCo on the same geometry)."""
    from render.spec import SceneSpec
    key = "hand-episode-%d-%d" % (id(server), int(server.ep_seed) & 0xFFFFFFFF)
    spec = _base_spec(server)
    _room_geometry(spec, _room_dir())
    _SPECS.clear()
    _SPECS[key] = spec
    ph = server.cfg["photo"]
    sc = SceneSpec(builder="arm.assembly_hand.agent_render:depth_scene", builder_args={"key": key},
                   content_files=[os.path.join(server.kit_dir, "meshes"), os.path.join(_room_dir(), "room.json")],
                   shadows=False, lod=str(ph.get("lod", "full")), near_m=0.01)
    be = FilamentAgentBackend()
    f = float(server.episode["lights"]["key"]) / NOMINAL_KEY
    return be.open_agent(server, sc, f)
