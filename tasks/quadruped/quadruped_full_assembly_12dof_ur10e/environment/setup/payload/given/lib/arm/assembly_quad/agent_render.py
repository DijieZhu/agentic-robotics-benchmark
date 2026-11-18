"""The renderer of the cell's cameras, for your own twin of the cell: MuJoCo's Filament renderer (physically based,
Vulkan; on the CPU through Mesa's lavapipe) in the baked factory hall with the cell's look. This is the code the cell
renders its camera frames with; what the cell draws per run (the hall's brightness) is an argument here.

    from arm.assembly_quad import agent_render
    session = agent_render.open_session(spec)          # your mujoco.MjSpec of the cell, in the cell frame (copied)
    session.sync_from(model, data)                     # your physics state, copied by joint name
    raw = session.render_raw_views([cam], ("rgb", "depth"))[0]    # raw.rgb uint8 H x W x 3, raw.depth float32 metres
    res = session.render([cam], photo=photo, seed=k, outputs=("rgb", "sensor_depth"))   # through render's photo pipeline
    session.close()

`cam` is a render.spec.CameraSpec (pixels fx fy cx cy, world_T_cam in OpenCV axes: x right, y down, z forward; lens
distortion only through the photo pipeline). One session per scene and one open session per process (close it before
opening the next): it is compiled once (on 8 CPUs for a twin of the whole cell about 13-15 s, up to about 30 s on a
busy machine, the AgX display transform's 8-bit lookup table, about 2 s, included) and holds about 6-7 GB while open;
until you give it a state it shows your twin's own pose (qpos0). Every frame copies the state by joint name, updates
the Filament scene and renders RGB as an exact off-axis pinhole (the MuJoCo backend's frustum), then the display
transform through that table: the first frame after Filament's context is made takes about 10-15 s, later ones about
1.2-1.5 s (1280 x 960) and 0.6-0.8 s (640 x 480) with depth. Closing the last open session (or view) of the process
releases Filament's context and its memory, so the next one's first frame takes that long again. The render model is
your MjSpec (bodies and joint names as in your physics) without the MuJoCo-only scenery (geoms named `floor`,
`backdrop_*`, `bench_leg_*`, `bench_rail_*`, `bench_shelf`, and every geom in a group above 2), plus the robot table's
steel frame under the 2.0 x 1.4 m top, the arm's control box with its teach pendant and the hazard tape on the floor
around the cell, the factory room bundle (`render.room.add_room`: Cycles-baked lightmaps on the hall, its lamps
lighting the live objects, the room's HDR image light) and the look (`agent_look.json` and `make_look`: PBR terms per
material class, the scanned oak maps on the bench top geom `table_top` of material `mat_bench`, light units, AgX). Your
scene's own lights are replaced by the hall's. Depth comes from the MuJoCo backend on the same geometry (room meshes
without textures): metric and registered with the RGB. Lens distortion, noise, blur, vignetting, colour jitter and the
depth-sensor model are render/postprocess.py (PhotoSpec), the same code as the MuJoCo path. `exposure_factor` scales
the exposure (the cell's hall is up to about +-20 % brighter or darker from run to run).

Assets: the scenery folder (`_render_dir()`: $ARB_AGENT_RENDER_DIR, else `render` next to $ARB_GIVEN_DIR, else
~/render) holds `room/` (an arb-room/1 bundle: room.json, meshes, baked textures, ibl/ibl.ktx) and `oak/` (the bench
top's scanned maps). Vulkan: Mesa's lavapipe on the CPU unless a GPU driver is present.
"""

import gc
import json
import math
import os
import time

import numpy as np

from render.backends.base import BackendCapabilities, RawView, RenderSession, raw_outputs
from render.backends.mujoco_backend import MujocoBackend
from render.filament.view import _OPEN, _release_context, check_one_session   # the process's Filament sessions

__all__ = ["open_session", "available", "make_look", "FilamentAgentBackend", "DEFAULT_BENCH"]

LOOK_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "agent_look.json")
# the cell's robot table (cell_manual.md, "Frame, bench, robot"): the top's size and centre in the cell frame, its
# thickness and the hall floor's height
DEFAULT_BENCH = {"size_xy": [2.0, 1.4], "centre_xy": [0.45, 0.0], "thickness": 0.03, "floor_z": -0.994}
MAX_LIVE_LAMPS = 6          # room lamps nearest the bench that light the live objects (the hall itself is baked)
# the robot table's frame (60 mm tube, rails at two heights), the control box and pendant, the floor tape (the demo videos')
LEG, RAIL_Z = 0.06, (-0.10, -0.80)
CONTROL_BOX = {"centre": (-0.95, -0.55), "size": (0.268, 0.475, 0.423)}
PENDANT = (0.30, 0.231, 0.05)
TAPE = {"x": (-1.35, 2.25), "y": (-1.45, 1.45), "w": 0.075, "t": 0.002}
_DROP = ("backdrop_", "bench_leg_", "bench_rail_", "bench_shelf")
_LUTS = {}
_SPECS = {}
_KEY = [0]


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
    look = Look(name="factory-agent-quad", version="1")
    look.materials = [
        (None, "mat_printed", dict(roughness=0.62, metallic=0.0)),      # PA12 prints (matte)
        (None, "mat_core", dict(roughness=0.45, metallic=0.0)),         # actuator housings (black)
        (None, "mat_pulley", dict(roughness=0.3, metallic=1.0)),        # aluminium output pulleys
        (None, "mat_rubber", dict(roughness=0.9, metallic=0.0)),        # feet
        (None, "mat_stand", dict(roughness=0.35, metallic=1.0)),        # anodised profiles
        (None, "mat_fork", dict(roughness=0.6, metallic=0.0)),          # printed forks
        (None, "ur/black", dict(roughness=0.45, metallic=0.0)),         # the arm: black rings / boots
        (None, "ur/jointgray", dict(roughness=0.4, metallic=0.0)),      # joint covers
        (None, "ur/linkgray", dict(roughness=0.42, metallic=1.0)),      # satin aluminium tubes
        (None, "ur/urblue", dict(roughness=0.38, metallic=0.0)),        # painted housings
        (None, "ur/g/metal", dict(roughness=0.36, metallic=1.0)),       # gripper: steel / aluminium
        (None, "ur/g/gray", dict(roughness=0.38, metallic=1.0)),
        (None, "ur/g/black", dict(roughness=0.45, metallic=0.0)),
        (None, "ur/g/silicone", dict(roughness=0.85, metallic=0.0)),
        (None, "mat_bench", dict(roughness=1.0, metallic=0.0)),         # x the oak's roughness map
        (None, "bench_steel", dict(roughness=0.5, metallic=1.0)),       # the table frame
        (None, "bench_shelf", dict(roughness=0.65, metallic=0.8)),
        (None, "bench_foot", dict(roughness=0.85, metallic=0.0)),
        (None, "prop_*", dict(roughness=0.55, metallic=0.0)),           # control box, pendant, tape
        (None, "*_board__*", dict(roughness=0.8, metallic=0.0)),        # printed datum boards
        (None, "tool_plate__*", dict(roughness=0.8, metallic=0.0)),
        (None, None, dict(roughness=0.6, metallic=0.0)),
    ]
    look.surface_material, look.surface_geom, look.surface_map_m = "mat_bench", "table_top", 2.0
    look.surface_albedo_gain = 1.0
    look.override({k: v for k, v in json.load(open(LOOK_FILE)).items() if not k.startswith("_")})
    oak = os.path.join(render_dir or _render_dir(), "oak")
    look.surface_dir = oak if os.path.isdir(oak) else None
    ibl = os.path.join(_room_dir(render_dir), "ibl", "ibl.ktx")
    look.room_ibl_ktx = ibl if os.path.isfile(ibl) else None
    look.exposure_scale = float(look.exposure_scale) * float(exposure_factor)
    return look


def _workbench(spec, bench):
    """the robot table's steel frame under the top (six legs, two rail levels, a lower shelf, feet), the arm's control
    box with the teach pendant on the floor behind the arm, the hazard tape around the cell (the demo videos' scene)"""
    import mujoco
    cx, cy = (float(v) for v in bench["centre_xy"])
    hx, hy, ht = float(bench["size_xy"][0]) / 2, float(bench["size_xy"][1]) / 2, float(bench["thickness"])
    zf = float(bench["floor_z"])
    mats = {}
    for name, rgba in (("bench_steel", [0.20, 0.22, 0.24, 1.0]), ("bench_shelf", [0.30, 0.31, 0.32, 1.0]), ("bench_foot", [0.05, 0.05, 0.05, 1.0]),
                       ("prop_cbox", [0.80, 0.81, 0.80, 1.0]), ("prop_cbox_face", [0.66, 0.67, 0.67, 1.0]), ("prop_cbox_handle", [0.08, 0.08, 0.09, 1.0]),
                       ("prop_pendant", [0.78, 0.79, 0.79, 1.0]), ("prop_pendant_screen", [0.02, 0.025, 0.03, 1.0]), ("prop_estop", [0.80, 0.05, 0.04, 1.0]),
                       ):
        mats[name] = spec.add_material(name=name, rgba=rgba).name
    # the tape's 45 deg yellow / black stripes: a seamless tile repeated once per tape width (the demo videos' tape)
    n = 256
    i, j = np.mgrid[0:n, 0:n]
    tile = np.where((((i + j) % (n // 2)) < (n // 4))[..., None], np.array([247, 190, 18], np.uint8), np.array([22, 22, 22], np.uint8))
    tex = spec.add_texture(name="prop_hazard_tex", type=mujoco.mjtTexture.mjTEXTURE_2D, width=n, height=n, nchannel=3)
    tex.data = np.ascontiguousarray(tile, np.uint8).tobytes()
    wb = spec.worldbody

    def box(name, c, h, mat, yaw=0.0):
        g = wb.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_BOX, size=list(h), pos=list(c), material=mats[mat], contype=0,
                        conaffinity=0, group=2)
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


def _base_spec(spec, bench=None):
    """A copy of your twin without the MuJoCo-only scenery, with the robot table and the props (no room yet)."""
    spec = spec.copy()
    for g in list(spec.geoms):
        nm = g.name or ""
        if nm == "floor" or nm.startswith(_DROP) or int(g.group) > 2:
            spec.delete(g)
    _workbench(spec, bench or DEFAULT_BENCH)
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
    """RGB from Filament (factory hall), depth / labels from a MuJoCo session on the same geometry."""

    def __init__(self, backend, twin_spec, depth_scene_spec, exposure_factor, bench=None, render_dir=None):
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
        spec = _base_spec(twin_spec, bench)
        bc = (bench or DEFAULT_BENCH)["centre_xy"]
        rec = add_room(spec, _room_dir(render_dir), bench_xy=(float(bc[0]), float(bc[1])))
        lamps = sorted([l for l in spec.lights if (l.name or "").startswith("room_light_")],
                       key=lambda l: math.hypot(l.pos[0] - float(bc[0]), l.pos[1] - float(bc[1])))
        for l in lamps[MAX_LIVE_LAMPS:]:
            spec.delete(l)
        for l in lamps[:MAX_LIVE_LAMPS]:
            l.range = 40.0           # the lamps hang ~5.8 m above the bench top (MuJoCo's default range is 10 m)
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
        return {"session": True, "renderer": "filament", "t_open_s": round(self.t_open_s, 2), "room": self.room_record.get("room")}

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
        return "filament-agent/1 mujoco-%s room-%s" % (mujoco.__version__, os.path.basename(os.path.realpath(_room_dir())))

    def capabilities(self):
        c = MujocoBackend.capabilities(self)
        return BackendCapabilities(name=self.name, implemented=True, outputs=c.outputs, native_lens_distortion=False, sessions=True,
                                   max_image_px=c.max_image_px, deterministic="same mujoco build, Vulkan driver and device",
                                   notes="Filament RGB in the baked factory hall; depth / segmentation from the MuJoCo backend",
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
    Filament in the factory hall, depth from MuJoCo on the same geometry.

    exposure_factor  the hall's brightness (1.0 = nominal; the cell's own varies by up to about +-20 % between runs)
    bench            the robot table {"size_xy", "centre_xy", "thickness", "floor_z"} (default DEFAULT_BENCH, the cell's)
    render_dir       the scenery folder with room/ and oak/ (default: _render_dir())
    content_files    files or folders whose bytes enter the frames' provenance digest (e.g. your meshes)
    lod              "full" (the meshes as they are) or "policy" (the render layer's decimated meshes, render/lod.py)
    """
    from render.spec import SceneSpec
    check_one_session()                  # one Filament session per process at a time: close the open one first
    _KEY[0] += 1
    key = "twin-%d-%d" % (os.getpid(), _KEY[0])
    dspec = _base_spec(spec, bench)
    _room_geometry(dspec, _room_dir(render_dir))
    _SPECS.clear()
    _SPECS[key] = dspec
    sc = SceneSpec(builder="arm.assembly_quad.agent_render:depth_scene", builder_args={"key": key},
                   content_files=[str(p) for p in content_files] + [os.path.join(_room_dir(render_dir), "room.json")],
                   shadows=False, lod=str(lod), near_m=0.01)
    be = FilamentAgentBackend()
    return be.open_agent(spec, sc, float(exposure_factor), bench=bench, render_dir=render_dir)
