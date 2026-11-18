"""Render twin of the race-car assembly cell for MuJoCo's classic renderer (the agent cameras' fallback renderer and the
base of the Filament scene, agent_render.py; a port of arm/assembly_quad/render_twin.py).

The twin is the episode's physics MjSpec rebuilt by `cell.build_spec` from the kit without its collision pieces
(`twin_base_spec`: same bodies and joint names as `cell.CarCell.spec`, so `session.sync_from(model, data)` copies the
state by joint name; the agent sees the kit's visual meshes, never its convex pieces; the robot's collision geoms,
group 3, deleted), plus visual-only additions:
the kit's three PNG textures on their materials (the boards' printed faces, OpenGL texture coordinates of the OBJ
files), the robot table's frame under the 2.0 x 1.4 m top, the floor and backdrop walls, the five datum boards
(procedural checkerboards of `render.scale_refs`: the bench board on the world, the stand, shelf and tray boards on
their fixtures at `fixtures.boards()`, the tool plate on the wrist camera housing's outer face) and the episode's
lights. One render session per episode (`open_session`).

Host. Everything here reads an episode host with the attributes of the cell server (server.py provides them):
`cell` (a CarCell built with the arm), `model` / `data` (= cell.m / cell.d), `episode` (the information tier's draws,
`assembly_info.layout_info.draw_episode(seed, cfg["info_layout"], strict=False)`: cameras, lights, the tool plate's
mount jitter), `cfg` (with "photo", the sensor models of `assembly_info.cell_info.DEFAULT_INFO_CONFIG`), `kit_dir`
and `ep_seed`. `ViewHost` is a minimal such host (an episode for tools and tests: layout, cell, draws, the
renderer choice); `camera_spec(host, name)`, `photo_spec(host, name)` and `render_frame(host, name, ...)` are the
camera pipeline of the server's `get_camera` (the earlier cell's `QuadInfoServer.camera_spec` / `photo_spec`).

`INFO_LAYOUT` holds the agent cameras' nominal mounts, their per-episode spreads and the bench lights: the earlier
assembly cell's grader configuration (same bench, arm, gripper and hall). It is hidden (tests/ only);
the grader configuration's "info_layout" block takes these values.
"""

import copy
import math
import os
import time

import numpy as np

from ..assembly_info.cell_render import attach_ref, board_ref

__all__ = ["INFO_LAYOUT", "BOARD_GEOMS", "FIXTURE_BOARDS", "scene", "open_session", "render_spec", "twin_base_spec", "visual_kit",
           "board_poses", "strip_hidden", "attach_kit_textures", "visual_spec_hook", "camera_spec", "photo_spec", "render_frame",
           "ViewHost"]

_SPECS = {}
BOARD_GEOMS = ("bench_board", "stand_board", "shelf_board", "tray_board", "tool_plate")
# the fixtures' boards (fixtures.BOARDS keys) and their scale-ref names in the twin
FIXTURE_BOARDS = {"stand": "stand_board", "shelf": "shelf_board", "tray": "tray_board"}
# hidden geom groups: the kit's collision pieces and the robot's collision geoms (MuJoCo group 3)
HIDDEN_GROUP_MIN = 3

# the agent cameras (top 1280 x 960 hfov 70 at 1.6 m looking straight down at the bench's middle; wrist 640 x 480 hfov
# 80 on the gripper's base, 58 mm off the tool axis on the tool's +x side, tilted 15.87 deg towards the fingertips), their
# per-episode spreads (f +-3 %, c +-2 %, distortion, top mount +-30 mm / 3 deg, wrist +-2 mm / 1 deg) and the bench
# lights (+-20 %): the earlier assembly cell's grader configuration, unchanged
INFO_LAYOUT = {
    "camera_draw": {"c_rel": 0.02, "f_rel": 0.03, "k1": [-0.08, 0.02], "k2": [-0.02, 0.02], "mount_ang_deg": 3.0,
                    "mount_pos_m": 0.03, "p": 0.001, "wrist_ang_deg": 1.0, "wrist_pos_m": 0.002},
    "cameras": {
        "top": {"depth": True, "eye": [0.45, 0.0, 1.6], "hfov_deg": 70.0, "size": [1280, 960], "target": [0.45, 0.0, 0.0],
                "up": [0.0, 1.0, 0.0]},
        "wrist": {"depth": True, "hfov_deg": 80.0, "offset_m": 0.058, "size": [640, 480], "tilt_deg": 15.87, "z_m": 0.0252},
    },
    "lights": {"key": {"pos": [1.25, -0.9, 2.4], "target": [0.45, 0.0, 0.0], "diffuse": 0.62},
               "fill": {"pos": [-0.4, 1.4, 1.8], "target": [0.45, 0.0, 0.0], "diffuse": 0.3},
               "dome": 0.22, "jitter_rel": 0.2},
}
AGENT_CAMERAS = ("top", "wrist")


def scene(key):
    """SceneSpec builder: a fresh copy of the registered render MjSpec."""
    return _SPECS[key].copy()


def _frame(z, x):
    z = np.asarray(z, float) / np.linalg.norm(z)
    x = np.asarray(x, float) - z * float(np.dot(x, z))
    x /= np.linalg.norm(x)
    T = np.eye(4)
    T[:3, 0], T[:3, 1], T[:3, 2] = x, np.cross(z, x), z
    return T


def _body_T(m, d, name):
    b = m.body(name).id
    T = np.eye(4)
    T[:3, :3] = d.xmat[b].reshape(3, 3)
    T[:3, 3] = d.xpos[b]
    return T


def board_poses(host):
    """{name: (parent body, T_in_parent, ref)} of the five boards for this episode: the bench board on the world, the
    three fixture boards on their fixtures (fixtures.boards(): the centre of the face the board rests on, x along its
    columns), the tool plate on the wrist camera housing's outer face (the tool's +x side; the episode's mount jitter)"""
    import mujoco
    from . import fixtures as FX
    m, d = host.model, host.data
    out = {}
    b = FX.BENCH_BOARD
    T = _frame([0, 0, 1], [math.cos(math.radians(b["yaw_deg"])), math.sin(math.radians(b["yaw_deg"])), 0])
    T[:3, 3] = b["pos"]
    out["bench_board"] = ("world", T, board_ref("bench_board", b["cols"], b["rows"], b["square_mm"], b["margin_mm"], b["thickness_mm"]))
    for key, bd in FX.boards().items():
        name = FIXTURE_BOARDS[key]
        out[name] = (bd["fixture"], np.asarray(bd["T_rest"], float),
                     board_ref(name, bd["cols"], bd["rows"], bd["square_mm"], bd["margin_mm"], bd["thickness_mm"]))
    t = FX.TOOL_PLATE
    rb = host.cell.robot
    hg = m.geom(rb["wrist_camera_housing"]).id
    mount = m.body(int(m.geom_bodyid[hg])).name
    Tm = _body_T(m, d, mount)
    site = m.site(rb["tcp_site"]).id
    R_tcp = d.site_xmat[site].reshape(3, 3)
    n_m = Tm[:3, :3].T @ R_tcp[:, 0]                       # the tool's +x in the mount body's frame
    z_m = Tm[:3, :3].T @ R_tcp[:, 2]                       # the tool axis
    gp = np.asarray(m.geom_pos[hg], float)
    Rg = np.zeros(9)
    mujoco.mju_quat2Mat(Rg, np.asarray(m.geom_quat[hg], float))
    Rg = Rg.reshape(3, 3)
    half = np.asarray(m.geom_size[hg][:3], float)
    k = int(np.argmax(np.abs(Rg.T @ n_m)))
    sign = float(np.sign((Rg.T @ n_m)[k]))
    centre = gp + Rg[:, k] * sign * half[k]
    T = _frame(n_m, z_m)
    T[:3, 3] = centre + n_m * 0.0005
    J = np.asarray(host.episode["hand_plate_jitter"], float)
    out["tool_plate"] = (mount, T @ J, board_ref("tool_plate", t["cols"], t["rows"], t["square_mm"], t["margin_mm"], t["thickness_mm"]))
    return out


def strip_hidden(spec):
    """Delete the geoms of the hidden groups (>= 3: the kit's convex pieces, the robot's collision geoms) and the meshes
    no remaining geom uses. Bodies and joints stay (the state copies by joint name). Returns (geoms, meshes) deleted."""
    gone = [g for g in spec.geoms if int(g.group) >= HIDDEN_GROUP_MIN]
    for g in gone:
        spec.delete(g)
    used = set(g.meshname for g in spec.geoms if g.meshname)
    orphans = [me for me in spec.meshes if me.name not in used]
    for me in orphans:
        spec.delete(me)
    return len(gone), len(orphans)


def attach_kit_textures(spec, kit):
    """The kit's PNG textures (kit["materials"][key]["texture"], kit-relative) on the cell's visual materials
    mat_vis_<key> as sRGB RGB maps; the OBJ meshes carry the texture coordinates. Returns {material: texture file}."""
    import mujoco
    mats = {m.name: m for m in spec.materials}
    out = {}
    for key, mk in sorted((kit.get("materials") or {}).items()):
        if not isinstance(mk, dict) or not mk.get("texture"):
            continue
        mat = mats.get("mat_vis_" + key)
        if mat is None:
            continue
        path = os.path.join(kit["dir"], mk["texture"])
        tex = spec.add_texture(name="tex_vis_" + key, type=mujoco.mjtTexture.mjTEXTURE_2D, file=path)
        tex.colorspace = mujoco.mjtColorSpace.mjCOLORSPACE_SRGB
        mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = tex.name
        out[mat.name] = mk["texture"]
    return out


def visual_spec_hook(spec):
    """CarCell spec_hook for kits with visual meshes: the model's art meshes are open surfaces, MuJoCo cannot take a
    volume inertia of them ("mesh volume is too small"); they are massless (mass 0 geoms), so the shell rule only lets
    the cell compile (an open item for the cell)"""
    import mujoco
    n = 0
    for mesh in spec.meshes:
        if (mesh.name or "").startswith("vis_"):
            mesh.inertia = mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL
            n += 1
    return n


def visual_kit(kit):
    """The kit as the render twin builds it: every body without its collision pieces when the kit has visual meshes
    (a gt kit: the agent sees the model, not the convex pieces); a kit without visual meshes (the prototype) keeps its
    pieces, shown (cell config show_collision). Returns (kit, show_collision)."""
    if not any(bb.get("visual") for bb in kit["bodies"].values()):
        return kit, True
    out = dict(kit)
    out["bodies"] = {bn: dict(bb, collision=[]) for bn, bb in kit["bodies"].items()}
    return out, False


def twin_base_spec(host):
    """The physics cell's MjSpec rebuilt from the visual kit (`cell.build_spec` with the episode's layout and the cell's
    config, the robot attached the same way): the same bodies, joints, welds and names as the physics model without the
    kit's 14k convex pieces (MjSpec.delete is linear in the element count: deleting them from a copy takes minutes),
    the robot's collision geoms deleted (strip_hidden). Returns (spec, record)."""
    from . import cell as C
    kit, show = visual_kit(host.cell.kit)
    cfg = dict(host.cell.cfg, show_collision=bool(show or host.cell.cfg.get("show_collision")))
    spec, _robot = C.build_spec(kit, host.cell.layout, cfg)
    n_vis = visual_spec_hook(spec)
    n_geoms, n_meshes = strip_hidden(spec)
    return spec, {"visual_meshes": n_vis, "hidden_geoms_deleted": n_geoms, "meshes_deleted": n_meshes, "kit_pieces_shown": bool(show)}


def render_spec(host):
    """The episode's render MjSpec (the physics spec rebuilt without the hidden groups + visual additions)."""
    import mujoco
    spec, _rec = twin_base_spec(host)
    attach_kit_textures(spec, host.cell.kit)
    w = spec.worldbody
    bc = host.cell.cfg["bench"]
    cx, cy = (float(v) for v in bc["centre_xy"])
    sx, sy = float(bc["size_xy"][0]) / 2, float(bc["size_xy"][1]) / 2
    th, fz = float(bc["thickness"]), float(bc["floor_z"])
    spec.add_material(name="mat_frame", rgba=[0.22, 0.23, 0.25, 1.0], specular=0.3, shininess=0.4)
    spec.add_material(name="mat_shelf_f", rgba=[0.30, 0.31, 0.32, 1.0])
    spec.add_material(name="mat_wall", rgba=[0.62, 0.62, 0.60, 1.0])
    spec.add_material(name="mat_floor_f", rgba=[0.36, 0.36, 0.37, 1.0])
    vis = dict(contype=0, conaffinity=0, group=2)
    leg = 0.03
    k = 0
    for ix in (-1, 0, 1):
        for iy in (-1, 1):
            x, y = cx + ix * (sx - 0.06), cy + iy * (sy - 0.06)
            h = (-th - fz) / 2
            w.add_geom(name="bench_leg_%d" % k, type=mujoco.mjtGeom.mjGEOM_BOX, size=[leg, leg, h], pos=[x, y, fz + h], material="mat_frame", **vis)
            k += 1
    for i, (x, y, hx, hy) in enumerate([(cx, cy - (sy - 0.06), sx - 0.06, 0.03), (cx, cy + sy - 0.06, sx - 0.06, 0.03),
                                        (cx - (sx - 0.06), cy, 0.03, sy - 0.06), (cx + sx - 0.06, cy, 0.03, sy - 0.06)]):
        w.add_geom(name="bench_rail_%d" % i, type=mujoco.mjtGeom.mjGEOM_BOX, size=[hx, hy, 0.03], pos=[x, y, -th - 0.07], material="mat_frame", **vis)
    w.add_geom(name="bench_shelf", type=mujoco.mjtGeom.mjGEOM_BOX, size=[sx - 0.09, sy - 0.09, 0.006], pos=[cx, cy, fz + 0.2], material="mat_shelf_f", **vis)
    for i, (x, y, hx, hy) in enumerate([(cx + 3.4, cy, 0.1, 4.0), (cx - 3.4, cy, 0.1, 4.0), (cx, cy + 3.4, 4.0, 0.1), (cx, cy - 3.4, 4.0, 0.1)]):
        w.add_geom(name="backdrop_%d" % i, type=mujoco.mjtGeom.mjGEOM_BOX, size=[hx, hy, 2.2], pos=[x, y, fz + 2.2], material="mat_wall", **vis)
    for g in spec.geoms:
        if g.name == "floor":
            g.material = "mat_floor_f"
    for name, (parent, T, ref) in board_poses(host).items():
        if parent == "world":
            body = spec.worldbody
        else:
            # the exact name over the copied spec's bodies (MjSpec.body() on a copied spec with an attached robot can
            # resolve a prefixed name to the wrong body: assembly_info/cell_render.py)
            body = [b for b in spec.bodies if b.name == parent][0]
        attach_ref(spec, body, ref, T)
    for l in list(spec.lights):                          # the episode's lights replace the cell's own
        spec.delete(l)
    return spec


def _content_files(host):
    """the kit files the render scene reads (its visual meshes and textures; the collision pieces are not drawn)"""
    out = [os.path.join(host.kit_dir, "meshes", sub) for sub in ("visual", "textures")]
    return [p for p in out if os.path.isdir(p)]


def open_session(host):
    """Open the episode's render session (MuJoCo backend; sensor depth; one pass per frame)."""
    from render.backends import get_backend, require_capabilities
    from render.spec import BackgroundSpec, HeadlightSpec, LightSpec, MaterialOverride, SceneSpec
    be = get_backend("mujoco")
    require_capabilities(be, sessions=True, outputs=["rgb", "sensor_depth"])
    key = "car-episode-%d-%d" % (id(host), int(host.ep_seed) & 0xFFFFFFFF)
    _SPECS.clear()
    _SPECS[key] = render_spec(host)
    L = host.episode["lights"]
    lc = host.episode["config"]["lights"]
    lights = []
    for k in ("key", "fill"):
        pos = np.asarray(lc[k]["pos"], float)
        d = np.asarray(lc[k]["target"], float) - pos
        d /= np.linalg.norm(d)
        lights.append(LightSpec(name="light_" + k, pos=pos.tolist(), dir=d.tolist(), diffuse=[L[k]] * 3, specular=[0.2 * L[k]] * 3,
                                castshadow=(k == "key"), cutoff_deg=70.0, exponent=2.0))
    ph = host.cfg["photo"]
    sc = SceneSpec(builder="arm.assembly_car.render_twin:scene", builder_args={"key": key}, content_files=_content_files(host),
                   materials=[MaterialOverride(match="mat_bench", rgba=[0.62, 0.52, 0.38, 1.0], specular=0.2)],
                   lights=lights, replace_lights=True, headlight=HeadlightSpec(ambient=[L["dome"]] * 3, diffuse=[0.08] * 3, specular=[0.02] * 3),
                   background=BackgroundSpec(rgb1=[0.78, 0.78, 0.79], rgb2=[0.42, 0.42, 0.43]), shadows=True,
                   shadowsize=int(ph.get("shadowsize", 2048)), lod=str(ph.get("lod", "full")), near_m=0.01)
    return be.open(sc)


# ------------------------------------------------------------------------------------------------ the camera pipeline
def camera_spec(host, name):
    """The episode's drawn camera (intrinsics, distortion, mount): the top camera's world pose; the wrist camera's on the
    gripper's base body (`<gripper prefix>base`, the earlier cell's wrist parent) as it is now."""
    from render.spec import CameraSpec
    c = host.episode["cameras"][name]
    intr = c["intrinsics"]
    if name == "wrist":
        parent = host.cell.robot["gripper_prefix"] + "base"
        W = _body_T(host.model, host.data, parent) @ np.asarray(c["hand_T_cam"], float)
    else:
        W = np.asarray(c["world_T_cam"], float)
    return CameraSpec(name=name, width=intr["width"], height=intr["height"], fx=intr["fx"], fy=intr["fy"], cx=intr["cx"], cy=intr["cy"],
                      world_T_cam=W.tolist(), k1=intr["k1"], k2=intr["k2"], p1=intr["p1"], p2=intr["p2"])


def photo_spec(host, name):
    """The information tier's photo pipeline (noise, defocus, vignetting, colour jitter, the depth sensor: near model for
    the wrist, fixed for the top), the colour response drawn once per camera unit and episode (a live camera's gain /
    tint does not change from frame to frame), the sensor noise per frame (the earlier cell's photo_spec)."""
    from render.spec import BlurSpec, ColourJitterSpec, DepthSensorSpec, NoiseSpec, PhotoSpec, derive_seed
    ph = host.cfg["photo"]
    dk = dict(ph["depth_common"])
    dk.update(ph["depth_near"] if name in ("wrist", "inspect") else ph["depth_fixed"])
    sensor = DepthSensorSpec(unit_seed=derive_seed(host.ep_seed, "depth_unit"), **dk)
    cj = ColourJitterSpec(**ph["jitter"]).copy(seed=derive_seed(host.ep_seed, "colour_unit", name))
    return PhotoSpec(noise=NoiseSpec(**ph["noise"]), blur=BlurSpec(defocus_px=float(ph["defocus_px"])), vignetting=float(ph["vignetting"]),
                     colour_jitter=cj, image_format="png", depth_sensor=sensor, png_compress_level=int(ph.get("png_compress_level", 1)))


def render_frame(host, name, outputs=("rgb", "sensor_depth"), k=None, encode_png=True):
    """One frame of camera `name` at the host's current physics state through its session (opened on first use):
    the RenderResult (rgb / encoded PNG; sensor_depth; with "depth" / "segmentation" the exact depth and the geom
    labels, grader-only). `k` = the episode's frame index (the noise seed; default: the host's counter)."""
    from render.spec import derive_seed
    s = host.session_open()
    s.sync_from(host.model, host.data)
    if k is None:
        k = host.frames_used
    host.frames_used = int(k) + 1
    return s.render([camera_spec(host, name)], photo=photo_spec(host, name), seed=derive_seed(host.ep_seed, "frame", int(k)),
                    outputs=tuple(outputs), encode_png=encode_png)


def _deep_update(a, b):
    for k, v in (b or {}).items():
        if isinstance(v, dict) and isinstance(a.get(k), dict):
            _deep_update(a[k], v)
        else:
            a[k] = copy.deepcopy(v)
    return a


class ViewHost(object):
    """A minimal episode host for the agent view (for tools and tests; the cell server provides the same attributes):
    the episode layout (`layout.episode_layout(seed, plan, kit, stratum=...)`), the cell with the arm (`CarCell`, the
    kit's visual meshes compiled with `visual_spec_hook`), the information tier's draws (cameras, lights, the tool
    plate's jitter) and one render session of `cfg["agent_renderer"]` ("filament": agent_render.py; "mujoco": this
    module). Timings of the build in `self.timing` (s)."""

    def __init__(self, kit_dir, robot_dir, plan, seed, stratum=None, nominal=False, config=None, kit=None, uv=None, log=None,
                 settle_s=1.0, renderer="filament"):
        from ..assembly_info import layout_info as LI
        from ..assembly_info.cell_info import DEFAULT_INFO_CONFIG
        from . import cell as C
        from . import layout as LY
        self.log = log or (lambda *a: None)
        self.cfg = _deep_update({"photo": copy.deepcopy(DEFAULT_INFO_CONFIG["photo"]), "info_layout": copy.deepcopy(INFO_LAYOUT),
                                 "agent_renderer": renderer, "layout": None}, config or {})
        self.timing = {}
        t0 = time.perf_counter()
        self.kit = kit if kit is not None else C.load_kit(kit_dir)
        self.kit_dir = self.kit["dir"]
        self.timing["load_kit_s"] = time.perf_counter() - t0
        self.ep_seed = int(seed)
        self.episode = LI.draw_episode(self.ep_seed, self.cfg["info_layout"], strict=False)
        t1 = time.perf_counter()
        self.uv = uv if uv is not None else LY.unit_geometry(self.kit)
        t2 = time.perf_counter()
        self.layout = LY.episode_layout(self.ep_seed, plan, self.kit, uv=self.uv, cfg=self.cfg["layout"], stratum=stratum, nominal=nominal)
        t3 = time.perf_counter()
        robot = {"model": "ur10e", "models_dir": robot_dir}
        self.cell = C.CarCell(kit_dir, self.layout, kit=self.kit, config={"robot": robot}, log=self.log, spec_hook=visual_spec_hook,
                              settle_s=settle_s)
        t4 = time.perf_counter()
        self.timing.update({"unit_geometry_s": t2 - t1, "layout_s": t3 - t2, "cell_s": t4 - t3, "cell_internal_s": float(self.cell.build_s)})
        self.model, self.data = self.cell.m, self.cell.d
        self.session = None
        self.renderer = None
        self.frames_used = 0
        self.t_session_open_s = None

    def session_open(self):
        if self.session is None:
            t = time.perf_counter()
            want = str(self.cfg.get("agent_renderer", "filament"))
            if want == "filament":
                from . import agent_render
                ok, why = agent_render.available()
                if not ok:
                    raise RuntimeError("agent renderer filament unavailable: %s" % why)
                self.session = agent_render.open_session(self)
            else:
                self.session = open_session(self)
            self.renderer = want
            self.t_session_open_s = time.perf_counter() - t
        return self.session

    def camera_spec(self, name):
        return camera_spec(self, name)

    def render(self, name, outputs=("rgb", "sensor_depth"), k=None, encode_png=True):
        return render_frame(self, name, outputs=outputs, k=k, encode_png=encode_png)

    def close(self):
        if self.session is not None:
            try:
                self.session.close()
            except Exception:  # noqa: BLE001
                pass
        self.session = None
