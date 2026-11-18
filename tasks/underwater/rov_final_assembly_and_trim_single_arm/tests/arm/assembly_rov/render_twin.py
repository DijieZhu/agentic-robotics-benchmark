"""Render twin of the final-assembly-and-trim cell: the scene of the agent's two RGB-D cameras for MuJoCo's classic
renderer (the fallback renderer, and the depth / segmentation twin under the Filament scene of agent_render.py).

Sources (the form ported, rewritten for this cell):
  docs/plan/driving/tasks/racecar_full_assembly_tenth_scale_ur10e/tests/arm/assembly_car/render_twin.py
    sha256 3c07bca14ba977cfa73a13914f7e1656cf1458a7d15cf1baf0dc5d73d40c17ff (camera pipeline, boards, ViewHost)
  tasks/quadruped/quadruped_full_assembly_12dof_ur10e/tests/arm/assembly_quad/render_twin.py (arb-main-20261113)
    sha256 2352241df77e61949c594a82033847273026809d5330b7e2d92786feeff5dbb5
  -dev/proto/observability/obs_common.py sha256 e48d229e7db6223a4a20abd96dbf47e6233e69fe17547d3bc1abca6f0d72a351
    (the transparent parts' "no depth return" model of gate G5, kept: DESIGN v1.2 section 1.3)

The twin. `build_view_spec` builds an MjSpec of the cell from the kit's VISUAL meshes (never its convex pieces): the
bench top, the five stations (`fixtures.add_station` at the episode's poses), the positioner with the vehicle on its
carrier (`positioner.build_positioner`, the episode's mounting offset), the 19 loose units (free bodies named as the kit
names them, free joints "f_<body>"), the arm with the wide gripper (`gripper_wide.add_ur_wide`; its collision geoms,
group 3, are deleted from the render model) and the datum boards (`render.scale_refs` checkerboards: the bench board
on the world, the tray board on the thruster tray, the L-mag board on the ballast box, the tool plate on the wrist
camera's housing). Every visual geom is in group 2, contype 0. The vertical thrusters' propellers are their own geoms
turned about the thruster axis by the episode's propeller angle (layout `nests[*].prop_deg`). The units' station
fasteners (kit `visual_fasteners`) are not drawn: a screw appearing at lock time would be a visual fastened signal,
which the cell does not give (DESIGN 1.2).
The joint names are the physics cell's (positioner.DEFAULT_CFG names, the arm's "ur/" names); `sync_map` copies a
physics state by joint name and, for a free body whose joint is named otherwise, by body name.

Materials: the digest's render palette as agent_look.json carries it (kit_materials.palette; the kit's visual keys
through kit_materials.alias). A palette class with alpha < 1 (the clear acrylic tubes, the dome) is TRANSPARENT:
in the depth stream a pixel whose first surface on its ray is a transparent geom has no return (depth 0), set before
the sensor model so that the no-return region's border is invalidated like any other (G5 "inv"; DESIGN 1.3). The
rule is implemented with the segmentation pass of the same camera (`TwinSession`): the geom id of the first surface.

Host. The camera pipeline reads an episode host with these attributes (the cell server provides them; `ViewHost`
is a minimal one for the dev tools and the tests):
  model, data          the physics model and data (joint names as above)
  kit, kit_dir         the kit (kit.json dict + "dir")
  layout, fx           layout.episode_layout(...) and its fixtures.Fixtures
  robot                the dict gripper_wide.add_ur_wide returned
  episode              assembly_info.layout_info.draw_episode(ep_seed, cfg["info_layout"], strict=False)
  cfg                  {"photo": sensor models (assembly_info.cell_info.DEFAULT_INFO_CONFIG["photo"]), "info_layout":
                        info_layout(...), "agent_renderer": "filament" | "mujoco"}
  ep_seed, frames_used the episode's seed and frame counter; session_open() -> the render session
`INFO_LAYOUT` holds the cameras' nominal mounts, their per-episode spreads and the bench lights: the earlier assembly
cell's grader configuration (QASM tests/grader_config.json "info_layout"), except the wrist camera's mount, which is
this cell's gripper's (`wrist_mount`: the wide gripper's camera mount, 58 mm off the tool axis, aimed at the tool axis
315 mm out). It is hidden (tests/ only); the grader configuration's "info_layout" takes these values.
"""

import copy
import json
import math
import os
import time

import numpy as np

from . import fixtures as FX

__all__ = ["INFO_LAYOUT", "AGENT_CAMERAS", "LOOK_FILE", "VIS_GROUP", "HOME_Q", "BOARD_NAMES", "scene", "load_look",
           "kit_materials", "palette_key", "transparent_keys", "load_kit", "build_view_spec", "render_spec",
           "strip_hidden", "board_poses", "wrist_mount", "info_layout", "sync_map", "apply_sync", "TwinSession",
           "TwinBackend", "open_session", "camera_spec", "photo_spec", "render_frame",
           "unit_visual_geoms", "visual_face_counts", "ViewHost"]

HERE = os.path.dirname(os.path.abspath(__file__))
LOOK_FILE = os.path.join(HERE, "agent_look.json")
VIS_GROUP = 2
HIDDEN_GROUP_MIN = 3            # the robot's collision geoms (group 3) and the pose-check hulls (group 4)
POSE_GROUP = 4
AGENT_CAMERAS = ("top", "wrist")
BOARD_NAMES = {"bench": "bench_board", "tray": "tray_board", "box": "box_board"}
# the arm's ready posture (the race-car cell's: turned away from the vehicle, q1 = 0 points the arm along the cell's -x,
# the upper arm up, the forearm level, the tool down; out of the top camera's view of the stations)
HOME_Q = [0.0, -1.5708, 1.5708, -1.5708, -1.5708, 0.0]
BENCH = {"centre_xy": [0.5 * (FX.TABLE["x_m"][0] + FX.TABLE["x_m"][1]), 0.5 * (FX.TABLE["y_m"][0] + FX.TABLE["y_m"][1])],
         "size_xy": [FX.TABLE["x_m"][1] - FX.TABLE["x_m"][0], FX.TABLE["y_m"][1] - FX.TABLE["y_m"][0]],
         "thickness": FX.TABLE["thickness_m"], "floor_z": -0.994}

# the cameras (top 1280 x 960 hfov 70 at 1.6 m looking straight down at the bench's middle; wrist 640 x 480 hfov 80 on
# the gripper's base, its mount the robot's), their per-episode spreads (f +-3 %, c +-2 %, distortion, top mount
# +-30 mm / 3 deg, wrist +-2 mm / 1 deg) and the bench lights (+-20 %): the earlier assembly cell's grader configuration
INFO_LAYOUT = {
    "camera_draw": {"c_rel": 0.02, "f_rel": 0.03, "k1": [-0.08, 0.02], "k2": [-0.02, 0.02], "mount_ang_deg": 3.0,
                    "mount_pos_m": 0.03, "p": 0.001, "wrist_ang_deg": 1.0, "wrist_pos_m": 0.002},
    "cameras": {
        "top": {"depth": True, "eye": [0.45, 0.0, 1.6], "hfov_deg": 70.0, "size": [1280, 960], "target": [0.45, 0.0, 0.0],
                "up": [0.0, 1.0, 0.0]},
        "wrist": {"depth": True, "hfov_deg": 80.0, "size": [640, 480], "mount": "robot"},
    },
    "lights": {"key": {"pos": [1.25, -0.9, 2.4], "target": [0.45, 0.0, 0.0], "diffuse": 0.62},
               "fill": {"pos": [-0.4, 1.4, 1.8], "target": [0.45, 0.0, 0.0], "diffuse": 0.3},
               "dome": 0.22, "jitter_rel": 0.2},
}

_SPECS = {}
_LOOK = {}


def scene(key):
    """SceneSpec builder: a fresh copy of the registered render MjSpec."""
    return _SPECS[key].copy()


# ------------------------------------------------------------------------------------------------ look / materials
def load_look(path=None):
    p = path or LOOK_FILE
    if p not in _LOOK:
        with open(p) as f:
            _LOOK[p] = json.load(f)
    return _LOOK[p]


def kit_materials(look=None):
    return (look or load_look())["kit_materials"]


def palette_key(vis_key, km=None):
    """the palette class of a kit visual key (`alias` first, else the key itself)"""
    km = km or kit_materials()
    k = km["alias"].get(vis_key, vis_key)
    if k not in km["palette"]:
        raise KeyError("kit visual key %r has no palette class (agent_look.json kit_materials)" % vis_key)
    return k


def transparent_keys(km=None):
    """palette classes with alpha < 1: no depth return"""
    km = km or kit_materials()
    return sorted(k for k, v in km["palette"].items() if float(v["rgba"][3]) < 1.0)


def material_name(pkey):
    return "mat_vis_" + pkey


def load_kit(kit_dir):
    with open(os.path.join(kit_dir, "kit.json"), encoding="utf-8") as f:
        kit = json.load(f)
    if kit.get("kit_version") != "rov-asm-kit/1":
        raise ValueError("%s: not rov-asm-kit/1" % kit_dir)
    kit["dir"] = os.path.abspath(kit_dir)
    return kit


# ------------------------------------------------------------------------------------------------ small geometry
def _quat(R):
    return [float(v) for v in FX._quat_wxyz(np.asarray(R, float))]


def _quat_R(q):
    w, x, y, z = (float(v) for v in q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


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


def _inertia_matrix(b):
    R = _quat_R(b["inertia_quat"])
    return R @ np.diag(np.asarray(b["inertia_diag"], float)) @ R.T


# ------------------------------------------------------------------------------------------------ the twin
def _ensure_materials(spec, km, keys, transparent_view=False):
    have = set(m.name for m in spec.materials)
    for k in keys:
        nm = material_name(k)
        if nm in have:
            continue
        p = km["palette"][k]
        rgba = list(p["rgba"])
        if transparent_view and k in (km.get("transparent_view") or {}):
            rgba = list(km["transparent_view"][k]["rgba"])
        spec.add_material(name=nm, rgba=rgba, specular=float(p["specular"]), shininess=float(p["shininess"]))
        have.add(nm)


def _add_visual(spec, body, kit, bname, km, meshes, prop_deg=None):
    """the kit body's visual meshes on `body` (group 2, no contacts, no mass); the propeller turned by prop_deg about
    the body's z (the thruster axis of frame T). Returns the geom names."""
    import mujoco
    b = kit["bodies"][bname]
    out = []
    for key, rel in sorted(b["visual"].items()):
        pk = palette_key(key, km)
        mname = "vis_" + os.path.basename(rel)[:-4]
        if mname not in meshes:
            me = spec.add_mesh(name=mname, file=os.path.join(kit["dir"], rel))
            me.inertia = mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL
            meshes.add(mname)
        kw = dict(name="vis_%s_%s" % (bname, key), type=mujoco.mjtGeom.mjGEOM_MESH, meshname=mname, material=material_name(pk),
                  contype=0, conaffinity=0, group=VIS_GROUP, mass=0.0)
        if key == "propeller" and prop_deg is not None:
            kw["quat"] = _quat(FX.rotz(float(prop_deg)))
        out.append(body.add_geom(**kw).name)
    return out


def _prop_angles(lay):
    return {n["unit"]: float(n.get("prop_deg", 0.0)) for n in lay.get("nests", [])}


def _hull_kind(kit, bname):
    t = kit["bodies"][bname]["type"]
    return {"electronics_enclosure": "enclosure", "thruster_guard": "guard", "ballast_block": "ballast"}.get(t, t)


PARTS = ("bench", "stations", "positioner", "units", "robot")


def build_view_spec(kit, lay, fx, robot_dir=None, robot_cfg=None, pose_geoms=False, transparent_view=False, include=PARTS):
    """(spec, record) of the cell's twin for one episode layout (module docstring). pose_geoms=True also adds the
    units' coarse hulls and the vehicle's box in group 4 (hidden, collidable) and keeps the arm's collision geoms:
    the stand-in model a dev tool poses the arm in (ur_ik posture checks). `include`: a subset of PARTS (a dev tool's
    units-only scene: the loose units' unoccluded footprints in a camera)."""
    import mujoco
    from . import gripper_wide as GW
    from . import layout as LY
    from . import positioner as PZ
    km = kit_materials()
    spec = mujoco.MjSpec()
    spec.compiler.degree = False
    spec.visual.global_.offwidth, spec.visual.global_.offheight = 1280, 960
    spec.memory = 256 << 20
    keys = sorted(set(palette_key(k, km) for b in kit["bodies"].values() for k in b["visual"]))
    _ensure_materials(spec, km, keys, transparent_view=transparent_view)
    w = spec.worldbody
    cx, cy = BENCH["centre_xy"]
    sx, sy = BENCH["size_xy"][0] / 2, BENCH["size_xy"][1] / 2
    th = BENCH["thickness"]
    if "bench" in include:
        spec.add_material(name="mat_bench", rgba=[0.62, 0.52, 0.38, 1.0], specular=0.2)
        w.add_geom(name="table_top", type=mujoco.mjtGeom.mjGEOM_BOX, size=[sx, sy, th / 2], pos=[cx, cy, -th / 2],
                   material="mat_bench", group=VIS_GROUP)
        w.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[5.0, 5.0, 0.1], pos=[cx, cy, BENCH["floor_z"]],
                   rgba=[0.36, 0.36, 0.37, 1.0], contype=0, conaffinity=0, group=VIS_GROUP)
    # the five stations at the episode's poses (static bodies named as the stations)
    if "stations" in include:
        for name in FX.STATIONS:
            fx.add_station(spec, name, LY.T_world_station(lay["stations"][name]), group=VIS_GROUP)
    meshes = set()
    props = _prop_angles(lay)
    vb = kit["bodies"][kit["base_body"]]
    vehicle = {"mass_kg": float(vb["mass_kg"]), "cg_B_m": [float(v) for v in vb["com_m"]], "inertia_about_com_B_kgm2": _inertia_matrix(vb)}
    vis = {}

    def add_vehicle(sp, body):
        vis[kit["base_body"]] = _add_visual(sp, body, kit, kit["base_body"], km, meshes)
        if pose_geoms:
            lo, hi = fx.vehicle_box_B(0.0)
            body.add_geom(name="pose_vehicle_box", type=mujoco.mjtGeom.mjGEOM_BOX, size=((hi - lo) / 2).tolist(),
                          pos=((hi + lo) / 2).tolist(), group=POSE_GROUP, mass=0.0, rgba=[0.5, 0.5, 0.5, 0.0])
    H = {}
    if "positioner" in include:
        H = PZ.build_positioner(spec, fx, vehicle=vehicle, mount_T=LY.mount_T(lay), add_vehicle_geoms=add_vehicle)
        for g in spec.geoms:                               # the positioner's boxes and cells: visual group
            if (g.name or "").startswith((H.names["base"] + "/", H.names["carrier"] + "/")) or (g.name or "").endswith("_vis"):
                g.group = VIS_GROUP
    for u in (kit["loose"] if "units" in include else []):
        b = kit["bodies"][u]
        pos, quat = lay["parts"][u]
        body = w.add_body(name=u, pos=[float(v) for v in pos], quat=[float(v) for v in quat])
        body.add_freejoint(name="f_" + u)
        body.mass = float(b["mass_kg"])
        body.ipos = [float(v) for v in b["com_m"]]
        body.iquat = [float(v) for v in b["inertia_quat"]]
        body.inertia = [float(v) for v in b["inertia_diag"]]
        body.explicitinertial = True
        vis[u] = _add_visual(spec, body, kit, u, km, meshes, prop_deg=props.get(u))
        if pose_geoms:
            P = fx.hull(_hull_kind(kit, u))
            mh = spec.add_mesh(name="pose_hull_" + u, uservert=np.asarray(P, float).ravel().tolist())
            mh.inertia = mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL
            body.add_geom(name="pose_hull_" + u, type=mujoco.mjtGeom.mjGEOM_MESH, meshname=mh.name, group=POSE_GROUP, mass=0.0,
                          rgba=[0.5, 0.5, 0.5, 0.0])
    cfg = dict(robot_cfg or {})
    if robot_dir:
        cfg.setdefault("models_dir", robot_dir)
    robot = GW.add_ur_wide(spec, cfg) if "robot" in include else None
    for me in spec.meshes:                                 # art meshes are open shells (no volume inertia)
        me.inertia = mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL
    rec = {"robot": robot, "positioner": dict(H), "visual_geoms": vis, "palette_keys": keys,
           "transparent_keys": [k for k in transparent_keys(km) if k in keys], "props_deg": props}
    return spec, rec


def strip_hidden(spec):
    """Delete the geoms of the hidden groups (>= 3: the arm's collision geoms, the pose hulls) and the meshes no
    remaining geom uses. Bodies and joints stay. Returns (geoms, meshes) deleted."""
    gone = [g for g in spec.geoms if int(g.group) >= HIDDEN_GROUP_MIN]
    for g in gone:
        spec.delete(g)
    used = set(g.meshname for g in spec.geoms if g.meshname)
    orphans = [me for me in spec.meshes if me.name not in used]
    for me in orphans:
        spec.delete(me)
    return len(gone), len(orphans)


def board_poses(host):
    """{name: (parent body, T_in_parent, ref)} of the datum boards for this episode: the bench board on the world, the
    tray board on the thruster tray, the L-mag board on the ballast box (L-mag only; fixtures.boards()), the tool plate
    on the wrist camera housing's outer face (the tool's +x side; the episode's mount jitter)"""
    import mujoco
    from ..assembly_info.cell_render import board_ref
    m, d = host.model, host.data
    out = {}
    for key, bd in host.fx.boards().items():
        name = BOARD_NAMES[key]
        parent = "world" if bd["fixture"] == "bench" else bd["fixture"]
        out[name] = (parent, np.asarray(bd["T_rest"], float),
                     board_ref(name, bd["cols"], bd["rows"], bd["square_mm"], bd["margin_mm"], bd["thickness_mm"]))
    t = FX.TOOL_PLATE
    rb = host.robot
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


def _attach_boards(spec, host):
    from ..assembly_info.cell_render import attach_ref
    for name, (parent, T, ref) in board_poses(host).items():
        body = spec.worldbody if parent == "world" else [b for b in spec.bodies if b.name == parent][0]
        attach_ref(spec, body, ref, T)


def _mujoco_scenery(spec):
    """the MuJoCo renderer's own backdrop: the bench's legs and rails, a lower shelf, four walls (the Filament scene
    drops these names and puts the factory hall around the bench)"""
    import mujoco
    w = spec.worldbody
    cx, cy = BENCH["centre_xy"]
    sx, sy = BENCH["size_xy"][0] / 2, BENCH["size_xy"][1] / 2
    th, fz = BENCH["thickness"], BENCH["floor_z"]
    spec.add_material(name="mat_frame", rgba=[0.22, 0.23, 0.25, 1.0], specular=0.3, shininess=0.4)
    spec.add_material(name="mat_shelf_f", rgba=[0.30, 0.31, 0.32, 1.0])
    spec.add_material(name="mat_wall", rgba=[0.62, 0.62, 0.60, 1.0])
    vis = dict(contype=0, conaffinity=0, group=VIS_GROUP)
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


def render_spec(host, scenery=True, transparent_view=False):
    """The episode's render MjSpec: the twin without the hidden groups, with the boards (and the MuJoCo backdrop when
    scenery=True); no lights (the session adds its own)."""
    spec, _rec = build_view_spec(host.kit, host.layout, host.fx, robot_cfg=_robot_cfg(host), transparent_view=transparent_view)
    strip_hidden(spec)
    _attach_boards(spec, host)
    if scenery:
        _mujoco_scenery(spec)
    for l in list(spec.lights):
        spec.delete(l)
    return spec


def _robot_cfg(host):
    rc = dict(getattr(host, "robot_cfg", None) or {})
    if getattr(host, "robot_dir", None):
        rc.setdefault("models_dir", host.robot_dir)
    return rc


def unit_visual_geoms(model, body_name):
    """the visual mesh geoms of a body (group 2, no contacts)"""
    b = model.body(body_name).id
    return [g for g in range(model.ngeom) if int(model.geom_bodyid[g]) == b and int(model.geom_type[g]) == 7
            and int(model.geom_group[g]) == VIS_GROUP and not (model.geom_contype[g] or model.geom_conaffinity[g])]


def visual_face_counts(kit):
    """{body type: faces of its visual meshes} and the totals: the kit's visual meshes as drawn (fasteners excluded)
    and as stored (with every unit's fasteners), counted from the STL files"""
    import struct
    cache = {}

    def faces(rel):
        if rel not in cache:
            with open(os.path.join(kit["dir"], rel), "rb") as f:
                f.seek(80)
                cache[rel] = struct.unpack("<I", f.read(4))[0]
        return cache[rel]
    by_type, drawn, per_body = {}, 0, {}
    for bn, b in kit["bodies"].items():
        n = sum(faces(r) for r in b["visual"].values())
        per_body[bn] = n
        drawn += n
        by_type[b["type"]] = n
    files = set(r for b in kit["bodies"].values() for r in list(b["visual"].values()) + list(b.get("visual_fasteners") or []))
    return {"by_type": by_type, "drawn_per_episode": drawn, "files_total": sum(faces(r) for r in files), "per_body": per_body}


# ------------------------------------------------------------------------------------------------ cameras
def wrist_mount(model, data, robot):
    """the wrist camera's nominal mount on the hand (the gripper's base body), as layout_info's wrist parameters:
    {"offset_m", "z_m", "tilt_deg"} from the robot's camera mount body (a MuJoCo camera frame: looks along -z, image x
    along the closing axis); raises if the mount is not of that form (lens off the tool axis on the hand's +x, image x
    on the hand's +y)"""
    import mujoco
    mujoco.mj_kinematics(model, data)
    Th = _body_T(model, data, robot["gripper_prefix"] + "base")
    Tm = _body_T(model, data, robot["wrist_camera_body"])
    T = np.linalg.inv(Th) @ Tm
    R_cv = T[:3, :3] @ np.diag([1.0, -1.0, -1.0])          # MuJoCo camera -> OpenCV (+z forward, +y down)
    p = T[:3, 3]
    xdir, zdir = R_cv[:, 0], R_cv[:, 2]
    if abs(p[1]) > 1e-9 or np.abs(xdir - [0.0, 1.0, 0.0]).max() > 1e-9 or abs(zdir[1]) > 1e-9 or p[0] <= 0:
        raise ValueError("the robot's wrist camera mount is not lens-on-+x / image-x-on-+y: p %s x %s z %s" % (p, xdir, zdir))
    return {"offset_m": float(p[0]), "z_m": float(p[2]), "tilt_deg": float(math.degrees(math.atan2(-zdir[0], zdir[2])))}


def info_layout(model, data, robot, base=None):
    """INFO_LAYOUT (or `base`) with the wrist camera's mount resolved from the robot ("mount": "robot")"""
    L = copy.deepcopy(base or INFO_LAYOUT)
    w = L["cameras"]["wrist"]
    if w.get("mount") == "robot":
        w.update(wrist_mount(model, data, robot))
    return L


def camera_spec(host, name):
    """The episode's drawn camera (intrinsics, distortion, mount): the top camera's world pose; the wrist camera's on the
    gripper's base body as it is now."""
    from render.spec import CameraSpec
    c = host.episode["cameras"][name]
    intr = c["intrinsics"]
    if name == "wrist":
        W = _body_T(host.model, host.data, host.robot["gripper_prefix"] + "base") @ np.asarray(c["hand_T_cam"], float)
    else:
        W = np.asarray(c["world_T_cam"], float)
    return CameraSpec(name=name, width=intr["width"], height=intr["height"], fx=intr["fx"], fy=intr["fy"], cx=intr["cx"], cy=intr["cy"],
                      world_T_cam=W.tolist(), k1=intr["k1"], k2=intr["k2"], p1=intr["p1"], p2=intr["p2"])


def photo_spec(host, name):
    """The photo pipeline (noise, defocus, vignetting, colour jitter, the depth sensor: near model for the wrist, fixed
    for the top), the colour response drawn once per camera unit and episode, the sensor noise per frame."""
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


# ------------------------------------------------------------------------------------------------ state copy
def sync_map(src, dst):
    """[(src qpos address, dst qpos address, width)]: every joint of `dst` found by name in `src`; a free joint of
    `dst` not found by name is matched through its body's name (the source body's own free joint)"""
    import mujoco as mj
    pairs, done = [], set()
    width = {0: 7, 1: 4, 2: 1, 3: 1}
    for j in range(dst.njnt):
        nm = mj.mj_id2name(dst, mj.mjtObj.mjOBJ_JOINT, j)
        k = mj.mj_name2id(src, mj.mjtObj.mjOBJ_JOINT, nm) if nm else -1
        if k >= 0 and int(src.jnt_type[k]) == int(dst.jnt_type[j]):
            pairs.append((int(src.jnt_qposadr[k]), int(dst.jnt_qposadr[j]), width[int(dst.jnt_type[j])]))
            done.add(j)
    src_free = {}
    for k in range(src.njnt):
        if int(src.jnt_type[k]) == 0:
            src_free[mj.mj_id2name(src, mj.mjtObj.mjOBJ_BODY, int(src.jnt_bodyid[k]))] = int(src.jnt_qposadr[k])
    for j in range(dst.njnt):
        if j in done or int(dst.jnt_type[j]) != 0:
            continue
        bn = mj.mj_id2name(dst, mj.mjtObj.mjOBJ_BODY, int(dst.jnt_bodyid[j]))
        if bn in src_free:
            pairs.append((src_free[bn], int(dst.jnt_qposadr[j]), 7))
    return pairs


def apply_sync(pairs, src_data, dst_model, dst_data):
    import mujoco as mj
    for a, b, n in pairs:
        dst_data.qpos[b:b + n] = src_data.qpos[a:a + n]
    mj.mj_kinematics(dst_model, dst_data)
    mj.mj_comPos(dst_model, dst_data)
    mj.mj_camlight(dst_model, dst_data)


def _no_return_ids(model, keys):
    import mujoco as mj
    mats = set(mj.mj_name2id(model, mj.mjtObj.mjOBJ_MATERIAL, material_name(k)) for k in keys)
    mats.discard(-1)
    return np.array(sorted(g for g in range(model.ngeom) if int(model.geom_matid[g]) in mats), np.int32)


# ------------------------------------------------------------------------------------------------ the MuJoCo session
_CLASSES = []


def _session_classes():
    if _CLASSES:
        return _CLASSES[0]
    from render.backends.base import RawView, raw_outputs
    from render.backends.mujoco_backend import MujocoBackend, MujocoSession

    class _TwinSession(MujocoSession):
        """A MuJoCo session of the twin with the no-return rule on its depth (module docstring) and the state copy of
        `sync_map`. no_return=False renders the exact depth of every first surface (a dev tool's switch)."""

        def __init__(self, backend, scene, no_return_keys=None):
            MujocoSession.__init__(self, backend, scene)
            self.no_return_keys = list(transparent_keys() if no_return_keys is None else no_return_keys)
            self.no_return_ids = _no_return_ids(self.model, self.no_return_keys)
            self.no_return = True
            self._pairs = None
            self.last_no_return_px = []

        def sync_from(self, model, data):
            if self._pairs is None or self._pairs[0] is not model:
                self._pairs = (model, sync_map(model, self.model))
            apply_sync(self._pairs[1], data, self.model, self.data)

        def render_raw_views(self, cameras, outputs=("rgb",)):
            outs = raw_outputs(outputs)
            want_d, want_s = "depth" in outs, "segmentation" in outs
            mask = want_d and self.no_return and len(self.no_return_ids) > 0
            need = list(outs) + (["segmentation"] if mask and not want_s else [])
            views = MujocoSession.render_raw_views(self, cameras, need)
            out = []
            self.last_no_return_px = []
            for v in views:
                d = v.depth
                if mask:
                    hit = np.isin(v.seg, self.no_return_ids)
                    d = np.where(hit, np.float32(0.0), d).astype(np.float32)
                    self.last_no_return_px.append(int(hit.sum()))
                out.append(RawView(rgb=v.rgb, depth=d, seg=v.seg if want_s else None))
            return out

    class _TwinBackend(MujocoBackend):
        def open(self, scene):
            return _TwinSession(self, scene)

    _CLASSES.append((_TwinSession, _TwinBackend))
    return _CLASSES[0]


def TwinSession(backend, scene, no_return_keys=None):
    return _session_classes()[0](backend, scene, no_return_keys)


def TwinBackend(gl=None):
    return _session_classes()[1](gl)


def _content_files(host):
    out = [os.path.join(host.kit_dir, "meshes", "visual"), LOOK_FILE]
    return [p for p in out if os.path.exists(p)]


def open_session(host):
    """The episode's MuJoCo render session (the fallback renderer; sensor depth with the no-return rule)."""
    from render.backends import require_capabilities
    from render.spec import BackgroundSpec, HeadlightSpec, LightSpec, SceneSpec
    be = TwinBackend()
    require_capabilities(be, sessions=True, outputs=["rgb", "sensor_depth"])
    key = "rov-episode-%d-%d" % (id(host), int(host.ep_seed) & 0xFFFFFFFF)
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
    sc = SceneSpec(builder="arm.assembly_rov.render_twin:scene", builder_args={"key": key}, content_files=_content_files(host),
                   lights=lights, replace_lights=True, headlight=HeadlightSpec(ambient=[L["dome"]] * 3, diffuse=[0.08] * 3, specular=[0.02] * 3),
                   background=BackgroundSpec(rgb1=[0.78, 0.78, 0.79], rgb2=[0.42, 0.42, 0.43]), shadows=True,
                   shadowsize=int(ph.get("shadowsize", 2048)), lod=str(ph.get("lod", "full")), near_m=0.01)
    return be.open(sc)


# ------------------------------------------------------------------------------------------------ a minimal host
def _deep_update(a, b):
    for k, v in (b or {}).items():
        if isinstance(v, dict) and isinstance(a.get(k), dict):
            _deep_update(a[k], v)
        else:
            a[k] = copy.deepcopy(v)
    return a


class ViewHost(object):
    """A minimal episode host for the agent view (dev tools, tests; the cell server provides the same attributes):
    the episode layout (`layout.episode_layout`), a kinematic stand-in of the physics model (the twin with the arm's
    collision geoms and the units' coarse hulls kept in hidden groups: same joint names as the cell), the
    information-tier draws (cameras, lights, the tool plate's jitter) and one render session of
    `cfg["agent_renderer"]` ("filament": agent_render.py; "mujoco": this module). No physics is stepped: the state is
    set (`set_index`, `set_arm`, `set_gripper`). Timings of the build in `self.timing` (s)."""

    def __init__(self, kit_dir, robot_dir, seed, stratum=None, nominal=False, config=None, kit=None, fx=None, renderer="filament",
                 index="keel_up", log=None):
        import mujoco
        from ..assembly_info import layout_info as LI
        from ..assembly_info.cell_info import DEFAULT_INFO_CONFIG
        from . import layout as LY
        self.log = log or (lambda *a: None)
        self.cfg = _deep_update({"photo": copy.deepcopy(DEFAULT_INFO_CONFIG["photo"]), "info_layout": copy.deepcopy(INFO_LAYOUT),
                                 "agent_renderer": renderer, "layout": None, "robot_home_q": list(HOME_Q)}, config or {})
        self.timing = {}
        t0 = time.perf_counter()
        self.kit = kit if kit is not None else load_kit(kit_dir)
        self.kit_dir = self.kit["dir"]
        self.robot_dir = robot_dir
        self.robot_cfg = {}
        lm = bool((self.cfg.get("layout") or {}).get("lmag", False))
        self.fx = fx if fx is not None else FX.Fixtures(FX.load_cell_layout(), FX.load_facts(), lmag=lm)
        self.ep_seed = int(seed)
        t1 = time.perf_counter()
        self.layout = LY.episode_layout(self.ep_seed, fx=self.fx, cfg=self.cfg["layout"], stratum=stratum, nominal=nominal)
        t2 = time.perf_counter()
        spec, rec = build_view_spec(self.kit, self.layout, self.fx, robot_cfg=_robot_cfg(self), pose_geoms=True)
        self.model = spec.compile()
        self.data = mujoco.MjData(self.model)
        t3 = time.perf_counter()
        self.robot = rec["robot"]
        self.view_record = rec
        self.cfg["info_layout"] = info_layout(self.model, self.data, self.robot, self.cfg["info_layout"])
        self.episode = LI.draw_episode(self.ep_seed, self.cfg["info_layout"], strict=False)
        self.set_index(index, forward=False)
        self.set_arm(self.cfg["robot_home_q"], forward=False)
        self.set_gripper(None)
        self.timing.update({"load_kit_s": t1 - t0, "layout_s": t2 - t1, "standin_build_s": t3 - t2})
        self.session = None
        self.renderer = None
        self.frames_used = 0
        self.t_session_open_s = None

    # ---- state
    def set_index(self, index, forward=True):
        import mujoco
        from . import positioner as PZ
        j = self.model.joint(PZ.DEFAULT_CFG["names"]["hinge"]).id
        self.data.qpos[self.model.jnt_qposadr[j]] = PZ.INDEX_ANGLE[index]
        self.index = index
        if forward:
            mujoco.mj_forward(self.model, self.data)

    def arm_qadr(self):
        return np.array([self.model.jnt_qposadr[self.model.joint(n).id] for n in self.robot["joints"]])

    def set_arm(self, q, forward=True):
        import mujoco
        self.data.qpos[self.arm_qadr()] = np.asarray(q, float)
        if forward:
            mujoco.mj_forward(self.model, self.data)

    def set_gripper(self, width=None, forward=True):
        """the gripper's linkage at an opening (m; None = fully open)"""
        import mujoco
        from . import gripper_wide as GW
        GW.set_opening(self.model, self.data, GW.MAX_WIDTH_M if width is None else float(width), prefix=self.robot["prefix"])
        if forward:
            mujoco.mj_forward(self.model, self.data)

    def unit_T(self, u):
        return _body_T(self.model, self.data, u)

    # ---- rendering
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
