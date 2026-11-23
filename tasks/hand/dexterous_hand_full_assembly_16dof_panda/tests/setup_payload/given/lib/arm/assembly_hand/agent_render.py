"""The renderer of the cell's cameras, for your own twin of the cell: MuJoCo's Filament renderer (physically based,
Vulkan; on the CPU through Mesa's lavapipe) in the baked lab bay with the cell's look. This is the code the cell
renders its camera frames with; what the cell draws per run is an argument here (the room's brightness:
`exposure_factor`; the camera units and the frames' noise: `photo_spec` and the seed of `session.render`).

    from arm.assembly_hand import agent_render
    session = agent_render.open_session(spec)          # your mujoco.MjSpec of the cell, in the cell frame (copied)
    session.sync_from(model, data)                     # your physics state, copied by joint name
    raw = session.render_raw_views([cam], ("rgb", "depth"))[0]    # raw.rgb uint8 H x W x 3, raw.depth float32 metres
    res = session.render([cam], photo=agent_render.photo_spec("top"), seed=k, outputs=("rgb", "sensor_depth"))
    session.close()

`cam` is a render.spec.CameraSpec (pixels fx fy cx cy, world_T_cam in OpenCV axes: x right, y down, z forward; lens
distortion only through the photo pipeline); render.filament.view.camera_spec(name, session.model, session.data,
width, height) gives one for a camera of your model. One open Filament session per process (render.filament.view:
close it before opening the next). It is compiled once; every frame copies the state by joint name, updates the
Filament scene and renders RGB as an exact off-axis pinhole (the MuJoCo backend's frustum), then the AgX display
transform through an 8-bit lookup table (built once per process). The render model is your MjSpec (bodies and joint
names as in your physics) without the MuJoCo-only scenery (geoms named `floor`, `backdrop_*`, `bench_leg_*`,
`bench_apron_*`, `bench_shelf`, and every geom in a group above 2), plus the steel workbench frame under the
1.6 x 1.0 m bench top, the lab bay's room bundle (`render.room.add_room`: Cycles-baked lightmaps on the room, its
lamps lighting the live objects, the room's HDR image light) and the look (`agent_look.json` and `make_look`: PBR terms
per material class, the maple maps on the bench top geom `table_top` of material `mat_bench`, light units,
AgX; a material named `vis_<key>` with a key of the kit's material table, e.g. `vis_printed_black_016`, takes that
material's terms, as the cell's units do). Your scene's own lights are replaced by the room's: the `live_lamps` room lamps nearest the bench light the live
objects and the `shadow_lamps` nearest of them cast shadows. Depth comes from the MuJoCo backend on the same geometry
(room meshes without textures): metric and registered with the RGB. Lens distortion, noise, blur, vignetting, colour
jitter and the depth-sensor model are render/postprocess.py (PhotoSpec; the cell cameras' own: `photo_spec`), the same
code as the MuJoCo path.
Filament's PBR material declares a UV0 attribute: every mesh of the RGB twin without texture coordinates gets a
per-vertex UV0 of zeros (`_uv0`), which is what Filament bound in its place before, minus a 'missing required
attributes (0xb), declared=0x3' line per mesh and draw; file meshes become user meshes with the compiled vertices,
normals and faces mapped back to the asset frame, so the compiled meshes are the same.

Assets: the scenery folder (`_render_dir()`: $ARB_AGENT_RENDER_DIR, else `render` next to $ARB_GIVEN_DIR, else
~/render) holds `room/` (an arb-room/1 bundle: room.json, meshes, baked textures, ibl/ibl.ktx) and `bench/` (the bench
top's maps; `oak/` is still read when `bench/` is absent). Vulkan: Mesa's lavapipe on the CPU unless a GPU driver is
present.
"""

import gc
import json
import math
import os
import time

import numpy as np

from render.backends.base import BackendCapabilities, RawView, RenderSession, raw_outputs
from render.backends.mujoco_backend import MujocoBackend
from render.filament.view import _OPEN, check_one_session    # the process's open Filament sessions (closed at exit)

__all__ = ["open_session", "available", "make_look", "photo_spec", "FilamentAgentBackend", "PHOTO"]

LOOK_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "agent_look.json")
MAX_LIVE_LAMPS = 6          # room lamps nearest the bench that light the live objects (the room itself is baked)
SHADOW_LAMPS = 2            # of those, the nearest that cast shadows
# the steel workbench under the bench top (1.6 x 1.0 m, top at z = 0, 30 mm thick, floor at z = -0.994)
TOP_HALF, TOP_T, FLOOR_Z, LEG, RAIL_Z = (0.8, 0.5), 0.03, -0.994, 0.04, (-0.10, -0.80)
_DROP = ("backdrop_", "bench_leg_", "bench_apron_", "bench_shelf")
_LUTS = {}
_SPECS = {}
_KEY = [0]


def _release_context():
    """Once no session is open: drop the Filament context too, then hand the freed heap back (glibc's malloc_trim). The
    engine keeps every earlier session's objects while its context lives (the HandleAllocator arena warning of the
    second session): on lavapipe each session left ~2.6 GB resident. The next session makes a fresh context."""
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
    """Filament (its GLFW-backed resource providers) connects to $DISPLAY when one is set (the desktop of this machine
    sets DISPLAY=:1), and a display server that accepts the connection and never answers blocks it for ever. The renderer
    is offscreen: drop the display variables."""
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


def make_look(exposure_factor=1.0, render_dir=None):
    from render.filament.backend import Look
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
    surf = [p for p in (os.path.join(render_dir or _render_dir(), k) for k in ("bench", "oak")) if os.path.isdir(p)]
    look.surface_dir = surf[0] if surf else None
    ibl = os.path.join(_room_dir(render_dir), "ibl", "ibl.ktx")
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


def _base_spec(spec):
    """A copy of your twin without the MuJoCo-only scenery, with the steel workbench (no room yet)."""
    spec = spec.copy()
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
    """RGB from Filament (the baked lab), depth / labels from a MuJoCo session on the same geometry."""

    def __init__(self, backend, twin_spec, depth_scene_spec, exposure_factor, render_dir=None, live_lamps=MAX_LIVE_LAMPS,
                 shadow_lamps=SHADOW_LAMPS):
        _headless()
        import mujoco
        import mujoco._render_filament as mjrf
        from render.filament.backend import FilamentBackend, _context, apply_look
        from render.room import add_room
        t0 = time.perf_counter()
        self.backend, self._mj, self._mjrf = backend, mujoco, mjrf
        self.depth = MujocoBackend.open(backend, depth_scene_spec)   # depth, labels, scene digest of the twin
        self.scene, self.labels, self.scene_content = self.depth.scene, self.depth.labels, self.depth.scene_content
        self.look = make_look(exposure_factor, render_dir)
        base_gain = self.look.post_tonemap_gain
        # the exposure factor brightens / darkens the linear render; the display transform keeps its nominal gain
        self.look.post_tonemap_gain = float(base_gain) * float(exposure_factor)
        spec = _base_spec(twin_spec)
        rec = add_room(spec, _room_dir(render_dir), bench_xy=(0.0, 0.0))
        n_live, n_shadow = int(live_lamps), int(shadow_lamps)
        lamps = sorted([l for l in spec.lights if (l.name or "").startswith("room_light_")], key=lambda l: math.hypot(l.pos[0], l.pos[1]))
        for l in lamps[n_live:]:
            spec.delete(l)
        for i, l in enumerate(lamps[:n_live]):
            l.range = 40.0           # MuJoCo's default range is 10 m; the lamps hang 2-6 m from the bench top
            l.castshadow = bool(l.castshadow) and i < n_shadow    # the nearest n_shadow cast shadows
        self.lamps_record = {"live": len(lamps[:n_live]), "shadow": sum(1 for l in lamps[:n_live] if l.castshadow)}
        for l in list(spec.lights):  # your scene's own lights: the room lights the bench
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
        self.sync_from(self.depth.model, self.depth.data)   # the twin's own pose until a state is given

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
        _SPECS.clear()
        self.ctx = None
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

    def open_agent(self, spec, depth_scene_spec, exposure_factor, render_dir=None, live_lamps=MAX_LIVE_LAMPS, shadow_lamps=SHADOW_LAMPS):
        return FilamentAgentSession(self, spec, depth_scene_spec, exposure_factor, render_dir=render_dir, live_lamps=live_lamps,
                                    shadow_lamps=shadow_lamps)


def open_session(spec, exposure_factor=1.0, render_dir=None, content_files=(), lod="full", live_lamps=MAX_LIVE_LAMPS,
                 shadow_lamps=SHADOW_LAMPS):
    """A render session of your twin `spec` (a mujoco.MjSpec of the cell in the cell frame; copied, never changed):
    Filament in the baked lab bay with the cell cameras' look, depth from MuJoCo on the same geometry.

    exposure_factor  the room's brightness (1.0 = nominal; the cell's own varies by up to about +-20 % between runs)
    render_dir       the scenery folder with room/ and bench/ (default: _render_dir())
    content_files    files or folders whose bytes enter the frames' provenance digest (e.g. your meshes)
    lod              "full" (the meshes as they are) or "policy" (the render layer's decimated meshes, render/lod.py)
    live_lamps       the room lamps nearest the bench that light the live objects (the cell's: 6)
    shadow_lamps     of those, the nearest that cast shadows (the cell's: 2)
    """
    from render.spec import SceneSpec
    check_one_session()                  # one Filament session per process at a time: close the open one first
    _KEY[0] += 1
    key = "twin-%d-%d" % (os.getpid(), _KEY[0])
    dspec = _base_spec(spec)
    _room_geometry(dspec, _room_dir(render_dir))
    _SPECS.clear()
    _SPECS[key] = dspec
    sc = SceneSpec(builder=__name__ + ":depth_scene", builder_args={"key": key},
                   content_files=[str(p) for p in content_files] + [os.path.join(_room_dir(render_dir), "room.json")],
                   shadows=False, lod=str(lod), near_m=0.01)
    be = FilamentAgentBackend()
    return be.open_agent(spec, sc, float(exposure_factor), render_dir=render_dir, live_lamps=live_lamps, shadow_lamps=shadow_lamps)


# the cell cameras' photo model (render.spec PhotoSpec / DepthSensorSpec terms, as the cell takes its frames): image
# noise, defocus and vignetting, a camera unit's colour response (gain and tint ranges), the depth units (top:
# depth_fixed, wrist: depth_near)
PHOTO = {
    "noise": {"shot": 0.002, "read": 0.004}, "defocus_px": 0.4, "vignetting": 0.1, "jitter": {"brightness": 0.05, "tint": 0.03},
    "depth_fixed": {"sigma_a_m": 0.0005, "sigma_b_per_m": 0.004, "wiggle_amp_m": 0.001, "fixed_pattern_a_m": 0.0005, "fixed_pattern_b_per_m": 0.0015,
                    "pixel_offset_sigma_m": 0.0004},
    "depth_near": {"sigma_a_m": 0.0002, "sigma_b_per_m": 0.003, "wiggle_amp_m": 0.0002, "fixed_pattern_a_m": 0.0001, "fixed_pattern_b_per_m": 0.0015,
                   "pixel_offset_sigma_m": 0.0001},
    "depth_common": {"min_range_m": 0.1, "max_range_m": 4.0, "quantisation_m": 0.001, "edge_threshold_rel": 0.03, "edge_invalid_px": 1,
                     "edge_ragged_px": 1, "dropout_rate": 0.03, "dropout_blob_px": 0},
    "png_compress_level": 1,
}
_UNIT_FIXED = ("wiggle_amp_m", "fixed_pattern_a_m", "fixed_pattern_b_per_m", "pixel_offset_sigma_m")


def photo_spec(camera="top", unit_seed=None):
    """The cell cameras' photo model (PHOTO) as a render.spec.PhotoSpec for `session.render(..., photo=..., seed=...)`:
    image noise, defocus, vignetting and the depth sensor of the `top` or the `wrist` camera. `unit_seed` None = a camera
    unit of nominal response (colour gain 1, no tint, no fixed depth errors); an integer draws a unit's colour response
    and its depth unit's fixed errors within the cell's spreads. The frame's noise comes from the seed of
    session.render (a nonzero integer; render.spec.new_seed() draws a fresh one)."""
    from render.spec import BlurSpec, ColourJitterSpec, DepthSensorSpec, NoiseSpec, PhotoSpec, derive_seed
    ph = PHOTO
    dk = dict(ph["depth_common"])
    dk.update(ph["depth_near"] if camera in ("wrist", "inspect") else ph["depth_fixed"])
    if unit_seed is None:
        dk.update({k: 0.0 for k in _UNIT_FIXED})
        sensor = DepthSensorSpec(**dk)
        jitter = ColourJitterSpec()
    else:
        sensor = DepthSensorSpec(unit_seed=derive_seed(int(unit_seed), "depth_unit"), **dk)
        jitter = ColourJitterSpec(seed=derive_seed(int(unit_seed), "colour_unit", str(camera)), **ph["jitter"])
    return PhotoSpec(noise=NoiseSpec(**ph["noise"]), blur=BlurSpec(defocus_px=float(ph["defocus_px"])), vignetting=float(ph["vignetting"]),
                     colour_jitter=jitter, image_format="png", depth_sensor=sensor, png_compress_level=int(ph.get("png_compress_level", 1)))
