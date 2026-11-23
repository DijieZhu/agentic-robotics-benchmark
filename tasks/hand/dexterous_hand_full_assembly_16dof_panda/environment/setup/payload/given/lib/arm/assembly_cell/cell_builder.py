"""Build the assembly cell (MjSpec -> MjModel) from the A2 givens, the model collection Panda and a hidden layout.

Physics as CHAIN_PLAN 4.3: MuJoCo 1 kHz, implicitfast, elliptic cones, CoACD hulls for the printed
parts, servo collision solids as shrunk boxes (side faces 0.4 mm, ears 0.3 mm, cable block 0.5 mm; the
rear / bottom face 3.0 mm and the horn face 2.5 mm, see DESIGN.md: the rear face never bears (the ears seat
on the bosses first), the horn plates are welded at the design pose, and the 5.5 mm axial budget is what lets
the second shoulder servo pass the 44.2 mm slot in front of its pocket (1.0 mm at the ring, 5.2 mm horn to
horn) and leaves ~28 mm behind its horn face for the fingertips that push it home),
horn and rear bosses visual-only, unpowered back-drivable horn hinges, `<weld active="false">` per
candidate pair, servo position actuators in actuator group 1 (disabled until the functional check).

World frame: table top centre, z up. Panda base at (-0.45, 0, 0), fixture plate 0.14 m at
(0.10, 0, 0) +- hidden jitter with its top at z = 0.02 (the base part is fixed on it, base frame =
world axes), trays at y = +-0.32 m with shaped sockets, four cameras (overview, top, oblique, wrist).
"""

import json
import math
import os
import shutil
import xml.etree.ElementTree as ET

import numpy as np

from . import CELL_VERSION
from .layout import DEFAULT_LAYOUT_CONFIG, hidden_layout, quat_from_rotmat

__all__ = ["DEFAULT_CELL_CONFIG", "CellBuild", "build_cell", "load_givens", "pose_mm", "collision_model_gate",
           "write_given_cell_xml", "MM"]

MM = 1e-3
PANDA_JOINTS = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"]
PANDA_HOME = [0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853]
PANDA_LINKS = ["link0", "link1", "link2", "link3", "link4", "link5", "link6", "link7", "hand", "left_finger", "right_finger"]

DEFAULT_CELL_CONFIG = {
    "timestep": 0.001,
    "tick_steps": 10,
    "impratio": 10.0,
    "multiccd": False,      # MuJoCo 3.14's multi-contact CCD reports bogus 10+ mm depths for a box grazing a thin convex piece (measured); single-contact CCD is exact
    "noslip_iterations": 5,
    "shrink_mm": {"body_side": 0.4, "body_bottom": 3.0, "horn_face": 2.5, "ears": 0.3, "cable_exit_block": 0.5},
    "pocket_carve_margin_mm": 0.3,
    "pocket_carve_ear_margin_mm": 4.0,     # the ear tabs' keep-out grows more (away from the body): a servo released in its pocket play must slide over the bosses and the bridge
    "pocket_carve_min_volume_mm3": 0.5,
    "pocket_carve_min_thickness_mm": 1.2,
    "horn_hinge": {"frictionloss": 0.1, "damping": 0.02, "armature": 1e-5},
    "servo_actuator": {"kp": 15.0, "kv": 0.5},
    "gripper": {"kp": 2500.0, "kv": 50.0, "max_force_N": 60.0, "max_width_m": 0.08},
    "panda_pos": [-0.45, 0.0, 0.0],
    "panda_gravcomp": 1.0,
    "table": {"size_xy": [1.2, 0.8], "thickness": 0.05, "floor_z": -0.75},
    "fixture": {"size_m": 0.14, "thickness_m": 0.02, "pin_d_mm": 4.0, "pin_h_mm": 12.0, "pin_xy_mm": 38.0},
    "geom_defaults": {"condim": 4, "friction": [0.8, 0.005, 0.0001], "solref": [0.004, 1.0], "solimp": [0.95, 0.99, 0.001, 0.5, 2.0]},
    "pad_friction": [1.0, 0.005, 0.0001],
    "weld_solref": [0.002, 1.0],
    "cameras": {
        "overview": {"pos": [0.9, 0.9, 0.9], "lookat": [0.10, 0.0, 0.10], "fovy": 45},
        "top": {"pos": [0.10, 0.0, 1.2], "lookat": [0.10, 0.0, 0.0], "fovy": 58, "up": [1.0, 0.0, 0.0]},
        "oblique": {"pos": [0.7, -0.6, 0.6], "lookat": [0.10, 0.0, 0.10], "fovy": 50},
    },
    "wrist_camera": {"pos": [0.0, -0.06, 0.03], "fovy": 70},
    "tcp_offset_m": 0.1034,
}


def pose_mm(frame):
    """{xyz (mm), rpy} -> 4x4 pose in metres (fixed-axis XYZ rpy, URDF)."""
    r, p, y = [float(v) for v in frame["rpy"]]
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    R = np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                  [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                  [-sp, cp * sr, cp * cr]])
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(frame["xyz"], float) * MM
    return T


def inv(T):
    Ti = np.eye(4)
    Ti[:3, :3] = T[:3, :3].T
    Ti[:3, 3] = -T[:3, :3].T @ T[:3, 3]
    return Ti


def load_givens(a2_dir):
    with open(os.path.join(a2_dir, "parts_manifest.json")) as f:
        manifest = json.load(f)
    with open(os.path.join(a2_dir, "mating_graph.json")) as f:
        graph = json.load(f)
    return manifest, graph


def quat_to_mat_np(q):
    w_, x, y, z = [float(v) for v in q]
    n = math.sqrt(w_ * w_ + x * x + y * y + z * z)
    w_, x, y, z = w_ / n, x / n, y / n, z / n
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w_), 2 * (x * z + y * w_)],
                     [2 * (x * y + z * w_), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w_)],
                     [2 * (x * z - y * w_), 2 * (y * z + x * w_), 1 - 2 * (x * x + y * y)]])


def _lookat_quat(pos, lookat, up=(0.0, 0.0, 1.0)):
    """wxyz quaternion of a camera at `pos` looking at `lookat` (camera looks along its -z, y up)."""
    pos = np.asarray(pos, float)
    f = np.asarray(lookat, float) - pos
    f /= np.linalg.norm(f)
    up = np.asarray(up, float)
    x = np.cross(f, up)
    if np.linalg.norm(x) < 1e-6:
        x = np.cross(f, [0.0, 1.0, 0.0])
    x /= np.linalg.norm(x)
    y = np.cross(x, f)
    z = -f
    return quat_from_rotmat(np.column_stack([x, y, z]))


def scope_parts(graph, scope_ids, manifest):
    """Loose parts needed by the mates in scope (the fixed root is excluded) plus the servo candidates."""
    mates = [m for m in graph["mates"] if m["id"] in set(scope_ids)]
    root = None
    names = []
    for m in mates:
        for key in ("part_a", "part_b"):
            n = m[key]
            if n == "fixture":
                continue
            if n not in names:
                names.append(n)
        for s in m.get("servos", []):
            if s not in names:
                names.append(s)
    parts = {p["name"]: p for p in manifest["parts"]}
    # the root link of the design (the base) is fixed on the fixture in rungs 1-3
    for n in names:
        if n in parts and parts[n]["kind"] == "printed" and parts[n].get("servo_occurrences") and n == "base":
            root = n
    if root is None:
        root = "base"
    loose = [n for n in names if n != root]
    return root, loose, mates


class CellBuild(object):
    """Result of `build_cell`: the compiled model, the spec and the bookkeeping the server needs."""

    def __init__(self, spec, model, meta):
        self.spec = spec
        self.model = model
        self.meta = meta


def _servo_boxes(part, shrink):
    """(name, half-size m, centre m) of the shrunk collision boxes of a servo (servo frame)."""
    out = []
    for pr in part["collision_primitives"]:
        mn = np.asarray(pr["min"], float).copy()
        mx = np.asarray(pr["max"], float).copy()
        if pr["name"] == "body":
            mn += [shrink["body_side"], shrink["body_side"], shrink["body_bottom"]]
            mx -= [shrink["body_side"], shrink["body_side"], shrink["horn_face"]]
        elif pr["name"].startswith("ear"):
            mn += shrink["ears"]
            mx -= shrink["ears"]
        else:
            mn += shrink["cable_exit_block"]
            mx -= shrink["cable_exit_block"]
        out.append((pr["name"], (mx - mn) / 2 * MM, (mx + mn) / 2 * MM))
    return out


def _keepout_boxes(part, parts, margin_mm, ear_margin_mm=None, graph=None, sweep_back_mm=2.0):
    """(T_box_in_carrier mm 4x4, half mm, centre mm) of every servo primitive designed into `part`, grown by margin
    (`ear_margin_mm` for the ear tabs, default = margin). With the mating graph the boxes are also SWEPT along
    the servo's insertion path (from `insertion_depth_mm - sweep_back_mm` short of the seat to the seat): the
    carrier must let the servo slide in from its slot pose, e.g. the ear tabs of a shoulder servo released in
    the centre slot pass 1.5 mm above the bridge in the design and must not catch it in the collision model."""
    ear_margin_mm = margin_mm if ear_margin_mm is None else ear_margin_mm
    seats = {}
    if graph is not None:
        for m in graph.get("mates", []):
            if m.get("kind") in ("sleeve_seat", "part_seat") and m.get("part_a") == part["name"]:
                seats[m["part_b"]] = m
    out = []
    for s in parts.values():
        if s.get("kind") != "servo" or s.get("carrier_link") != part["name"]:
            continue
        T = pose_mm(s["designed_pose_in_carrier"])
        T[:3, 3] /= MM
        sweep = 0.0
        m = seats.get(s["name"])
        if m is not None:
            # the servo enters rear-first: its motion is along -z (servo frame); check the graph agrees
            ax_c = np.asarray(m["insertion_axis_in_a"], float)
            if abs(float(ax_c @ (-T[:3, 2]))) > 0.99:
                sweep = max(0.0, float(m.get("insertion_depth_mm", 0.0)) - sweep_back_mm)
        body = [pr for pr in s["collision_primitives"] if pr["name"] == "body"]
        bmin = np.asarray(body[0]["min"], float) if body else None
        bmax = np.asarray(body[0]["max"], float) if body else None
        for pr in s["collision_primitives"]:
            is_ear = pr["name"].startswith("ear")
            mg = ear_margin_mm if is_ear else margin_mm
            mn = np.asarray(pr["min"], float) - mg
            mx = np.asarray(pr["max"], float) + mg
            if is_ear and bmin is not None:
                # the ear keep-out grows away from the body only (down / up / sideways / along the path), never into
                # the pocket cross-section: the pocket walls keep the body's own margin
                for k in range(3):
                    if pr["max"][k] <= bmin[k] + 1e-6:          # the ear lies on the negative side of the body
                        mx[k] = min(mx[k], bmin[k] + margin_mm)
                    if pr["min"][k] >= bmax[k] - 1e-6:          # the ear lies on the positive side of the body
                        mn[k] = max(mn[k], bmax[k] - margin_mm)
            mx[2] += sweep                      # the horn-side extension = the path back to the slot pose
            out.append((T, (mx - mn) / 2, (mx + mn) / 2))
    return out


def _clip_convex(mesh, normal, origin, eps=1e-9):
    """The part of a convex polytope on the positive side of the plane (exact: kept vertices plus the
    intersection points of the crossing edges, then their convex hull). None when nothing is left."""
    import trimesh
    V = np.asarray(mesh.vertices, float)
    n = np.asarray(normal, float)
    dist = (V - np.asarray(origin, float)) @ n
    keep = dist >= -eps
    if not np.any(keep):
        return None
    if np.all(keep):
        return mesh
    pts = [V[keep]]
    edges = mesh.edges_unique
    a, b = dist[edges[:, 0]], dist[edges[:, 1]]
    cross = (a < -eps) & (b > eps) | (a > eps) & (b < -eps)
    if np.any(cross):
        t = (a[cross] / (a[cross] - b[cross]))[:, None]
        pts.append(V[edges[cross, 0]] + t * (V[edges[cross, 1]] - V[edges[cross, 0]]))
    P = np.vstack(pts)
    if len(P) < 4:
        return None
    try:
        hull = trimesh.convex.convex_hull(P)
    except Exception:  # noqa: BLE001 - degenerate (flat) remainder
        return None
    if not hull.is_volume or hull.volume <= 0:
        return None
    return hull


def _min_thickness(mesh):
    """Smallest extent of the oriented bounding box (mm): thin slabs have ambiguous contact normals."""
    import trimesh
    try:
        _T, ext = trimesh.bounds.oriented_bounds(mesh)
        return float(np.min(ext))
    except Exception:  # noqa: BLE001
        return float(np.min(mesh.extents))


def carve_hulls(hulls, boxes, min_volume_mm3=0.5, min_thickness_mm=1.2):
    """Remove the keep-out boxes from a list of convex hulls (trimesh, mm) by half-space clipping.

    Every hull that overlaps a box is replaced by its (convex) parts outside each of the box's six
    faces, clipped so the parts do not overlap each other; the union of the parts is hull minus box,
    so the pocket walls become exact planes where the approximate convex decomposition bulged into
    the clearance zone (DESIGN.md section 3). No shapely needed (the base image has none)."""
    pieces = list(hulls)
    for T, half, centre in boxes:
        Ti = inv(T)
        nxt = []
        for h in pieces:
            hb = h.copy()
            hb.apply_transform(Ti)
            lo, hi = hb.bounds
            if np.any(lo >= centre + half) or np.any(hi <= centre - half):
                nxt.append(h)
                continue
            rest = hb
            for axis in range(3):
                for sgn in (-1.0, 1.0):
                    if rest is None:
                        break
                    n = np.zeros(3)
                    n[axis] = sgn
                    origin = centre + n * half[axis]
                    outside = _clip_convex(rest, n, origin)           # beyond this face: keep
                    if outside is not None and outside.volume >= min_volume_mm3:
                        outside.apply_transform(T)
                        nxt.append(outside)
                    rest = _clip_convex(rest, -n, origin)             # the remainder inside this face's half-space
            # `rest` is now the part inside the box: discarded
        pieces = nxt
    # drop the paper-thin skins the clip leaves where a hull bulged into the pocket by less than the
    # carve margin: a 0.1 mm slab has an ambiguous contact normal and lets a part through; the real
    # walls (>= 2 mm) stay
    return [pc for pc in pieces if _min_thickness(pc) >= min_thickness_mm]


def _clip_to_bbox(pieces, bbox_mm, slack_mm=0.5, min_volume_mm3=0.5):
    """Clip pieces to the part's bounding box: a hull of a subset of the part's vertices can never leave
    it, so anything outside is a numerical artefact (a stray sliver was measured 200 mm off the upper arm)."""
    lo = np.asarray(bbox_mm[0], float) - slack_mm
    hi = np.asarray(bbox_mm[1], float) + slack_mm
    out = []
    for pc in pieces:
        b0, b1 = pc.bounds
        if np.all(b0 >= lo) and np.all(b1 <= hi):
            out.append(pc)
            continue
        rest = pc
        for axis in range(3):
            for sgn, bound in ((1.0, lo[axis]), (-1.0, hi[axis])):
                if rest is None:
                    break
                n = np.zeros(3)
                n[axis] = sgn
                origin = np.zeros(3)
                origin[axis] = bound
                rest = _clip_convex(rest, n, origin)
        if rest is not None and rest.volume >= min_volume_mm3:
            out.append(rest)
    return out


def carved_collision(a2_dir, part, parts, margin_mm=0.3, min_volume_mm3=0.5, min_thickness_mm=1.2, ear_margin_mm=None):
    """The carrier's collision pieces (trimesh, mm): its CoACD hulls minus the designed servo keep-outs swept
    along the insertion paths (mating_graph.json), every piece clipped to the part's bounding box."""
    import trimesh
    hulls = [trimesh.load(os.path.join(a2_dir, rel), force="mesh") for rel in part["mesh_collision"]]
    gpath = os.path.join(a2_dir, "mating_graph.json")
    graph = json.load(open(gpath)) if os.path.isfile(gpath) else None
    boxes = _keepout_boxes(part, parts, margin_mm, ear_margin_mm, graph)
    pieces = carve_hulls(hulls, boxes, min_volume_mm3, min_thickness_mm) if boxes else hulls
    return _clip_to_bbox(pieces, part["bbox_mm"], min_volume_mm3=min_volume_mm3)


def build_cell(a2_dir, panda_dir, layout, config=None, scope_ids=None, manifest=None, graph=None, with_visual=True, collision_cache=None):
    """Compile the cell for one episode.

    a2_dir: A2 output (parts_manifest.json, mating_graph.json, meshes/); panda_dir: model collection
    franka_emika_panda folder; layout: `layout.hidden_layout(...)`; scope_ids: mate ids in scope
    (default graph scopes.rung1_prefix). Returns CellBuild.
    """
    import mujoco
    cfg = json.loads(json.dumps(DEFAULT_CELL_CONFIG))
    for k, v in (config or {}).items():
        if isinstance(v, dict) and isinstance(cfg.get(k), dict):
            cfg[k].update(v)
        else:
            cfg[k] = v
    if manifest is None or graph is None:
        manifest, graph = load_givens(a2_dir)
    parts = {p["name"]: p for p in manifest["parts"]}
    scope_ids = list(scope_ids or graph["scopes"]["rung1_prefix"])
    root, loose, mates = scope_parts(graph, scope_ids, manifest)
    shrink = cfg["shrink_mm"]

    spec = mujoco.MjSpec()
    spec.modelname = "arb_arm_a1_assembly_cell"
    spec.compiler.degree = False
    spec.compiler.autolimits = True
    spec.option.timestep = float(cfg["timestep"])
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    spec.option.noslip_iterations = int(cfg["noslip_iterations"])
    spec.option.impratio = float(cfg["impratio"])
    spec.option.gravity = [0.0, 0.0, -9.81]
    spec.option.disableactuator = 1 << 1          # servo actuators (group 1) off until the functional check
    if not cfg.get("multiccd", False):
        spec.option.disableflags |= mujoco.mjtDisableBit.mjDSBL_MULTICCD
    gd = spec.default.geom
    gd.condim = int(cfg["geom_defaults"]["condim"])
    fr = list(cfg["geom_defaults"]["friction"])
    fr[0] = float(layout.get("friction", fr[0]))
    gd.friction = fr
    gd.solref = cfg["geom_defaults"]["solref"]
    gd.solimp = cfg["geom_defaults"]["solimp"]
    spec.visual.headlight.diffuse = [0.6, 0.6, 0.6]
    spec.visual.headlight.ambient = [0.3, 0.3, 0.3]
    spec.visual.global_.offwidth = 1280
    spec.visual.global_.offheight = 960
    spec.stat.extent = 1.0
    spec.stat.center = [0.1, 0.0, 0.2]

    w = spec.worldbody
    w.add_light(name="sun", pos=[0.3, -0.3, 2.0], dir=[-0.15, 0.15, -1.0], type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL, diffuse=[0.7, 0.7, 0.7])
    w.add_light(name="fill", pos=[-0.5, 0.8, 1.2], dir=[0.4, -0.6, -0.8], diffuse=[0.35, 0.35, 0.35])
    # table, floor
    tb = cfg["table"]
    spec.add_material(name="mat_table", rgba=[0.82, 0.80, 0.76, 1.0])
    spec.add_material(name="mat_fixture", rgba=[0.35, 0.38, 0.42, 1.0])
    spec.add_material(name="mat_tray", rgba=[0.55, 0.58, 0.62, 1.0])
    spec.add_material(name="mat_socket", rgba=[0.42, 0.45, 0.50, 1.0])
    spec.add_material(name="mat_pla", rgba=[0.86, 0.87, 0.90, 1.0])
    spec.add_material(name="mat_pla_shoulder", rgba=[0.78, 0.84, 0.95, 1.0])
    spec.add_material(name="mat_pla_arm", rgba=[0.80, 0.93, 0.82, 1.0])
    spec.add_material(name="mat_servo", rgba=[0.22, 0.22, 0.25, 1.0])
    spec.add_material(name="mat_horn", rgba=[0.85, 0.72, 0.30, 1.0])
    spec.add_material(name="mat_hull", rgba=[0.9, 0.3, 0.3, 0.3])
    w.add_geom(name="table_top", type=mujoco.mjtGeom.mjGEOM_BOX, size=[tb["size_xy"][0] / 2, tb["size_xy"][1] / 2, tb["thickness"] / 2],
               pos=[0.0, 0.0, -tb["thickness"] / 2], material="mat_table")
    w.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[3.0, 3.0, 0.1], pos=[0.0, 0.0, tb["floor_z"]], rgba=[0.3, 0.3, 0.3, 1.0])
    # fixture with the fixed base part
    fx = cfg["fixture"]
    fpos = np.asarray(layout["fixture"]["pos"], float)
    fixture = w.add_body(name="fixture", pos=[fpos[0], fpos[1], 0.0])
    fixture.add_geom(name="fixture_plate", type=mujoco.mjtGeom.mjGEOM_BOX, size=[fx["size_m"] / 2, fx["size_m"] / 2, fx["thickness_m"] / 2],
                     pos=[0.0, 0.0, fx["thickness_m"] / 2], material="mat_fixture")
    for i, (sx, sy) in enumerate([(-1, -1), (-1, 1), (1, -1), (1, 1)]):
        fixture.add_geom(name="fixture_pin_%d" % i, type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[fx["pin_d_mm"] / 2 * MM, fx["pin_h_mm"] / 2 * MM, 0.0],
                         pos=[sx * fx["pin_xy_mm"] * MM, sy * fx["pin_xy_mm"] * MM, fx["thickness_m"] + fx["pin_h_mm"] / 2 * MM],
                         contype=0, conaffinity=0, group=1, material="mat_fixture")
    fixture.add_site(name="fixture_frame", pos=[0.0, 0.0, fx["thickness_m"]], size=[0.003, 0.003, 0.003], group=4)

    mesh_files = {}

    def add_mesh(name, rel_or_abs, scale):
        path = rel_or_abs if os.path.isabs(rel_or_abs) else os.path.join(a2_dir, rel_or_abs)
        if name not in mesh_files:
            spec.add_mesh(name=name, file=path, scale=[scale] * 3)
            mesh_files[name] = path
        return name

    def add_printed(body, name, material):
        p = parts[name]
        pieces = None
        if collision_cache is not None and name in collision_cache:
            pieces = collision_cache[name]
        else:
            pieces = carved_collision(a2_dir, p, parts, cfg["pocket_carve_margin_mm"], cfg["pocket_carve_min_volume_mm3"], cfg["pocket_carve_min_thickness_mm"], cfg.get("pocket_carve_ear_margin_mm"))
            if collision_cache is not None:
                collision_cache[name] = pieces
        for i, piece in enumerate(pieces):
            mn = "%s_c%02d" % (name, i)
            spec.add_mesh(name=mn, uservert=(np.asarray(piece.vertices, float) * MM).ravel().tolist())
            body.add_geom(name=mn, type=mujoco.mjtGeom.mjGEOM_MESH, meshname=mn, group=3, material="mat_hull")
        if with_visual:
            mv = add_mesh(name + "_visual", p["mesh_visual"], MM)
            body.add_geom(name=name + "_visual", type=mujoco.mjtGeom.mjGEOM_MESH, meshname=mv, contype=0, conaffinity=0, group=2, material=material)
        body.mass = float(p["mass_kg"])
        body.ipos = (np.asarray(p["com_mm"], float) * MM).tolist()
        I = np.asarray(p["inertia_kg_mm2"], float) * 1e-6
        body.explicitinertial = True
        body.fullinertia = [I[0, 0], I[1, 1], I[2, 2], I[0, 1], I[0, 2], I[1, 2]]

    base_body = w.add_body(name=root, pos=[fpos[0], fpos[1], fx["thickness_m"]])
    add_printed(base_body, root, "mat_pla")
    base_body.mass = 0.0   # static: the compiler ignores a static body's inertia
    base_body.explicitinertial = False

    # trays + sockets
    ts = layout.get("tray_size_m", DEFAULT_LAYOUT_CONFIG["tray_size_m"])
    lcfg = dict(DEFAULT_LAYOUT_CONFIG)
    for tname, centre in layout["trays"].items():
        w.add_geom(name="tray_%s" % tname, type=mujoco.mjtGeom.mjGEOM_BOX, size=[ts[0] / 2, ts[1] / 2, ts[2] / 2],
                   pos=[centre[0], centre[1], ts[2] / 2], material="mat_tray")
    oversize = float(layout.get("config", {}).get("socket_oversize_mm", lcfg["socket_oversize_mm"])) * MM
    wall = lcfg["socket_wall_mm"] * MM
    sock_h = lcfg["socket_height_mm"] * MM
    if layout.get("sockets", True):
        for name, lp in layout["parts"].items():
            cr = lp.get("cradle")
            if cr:
                # cradle under a lying servo: its footprint in the tray plane from the rest rotation
                Rp = np.asarray(quat_to_mat_np(lp["quat"]), float)
                c = np.asarray(lp["pos"], float) + Rp @ np.asarray(cr["centre_part"], float)
                yaw = lp["socket"]["yaw"]
                qy = quat_from_rotmat(np.array([[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]]))
                # half extents in the yawed tray frame: the servo axis (part z, 30 mm long) lies along the yawed x
                w.add_geom(name="cradle_%s" % name, type=mujoco.mjtGeom.mjGEOM_BOX, size=[cr["half_xy_part"][0], cr["half_xy_part"][1], cr["height_m"] / 2],
                           pos=[c[0], c[1], ts[2] + cr["height_m"] / 2], quat=qy, material="mat_socket")
            s = lp["socket"]
            hx, hy = s["half_xy"][0] + oversize, s["half_xy"][1] + oversize
            cx, cy = s["centre_xy"]
            yaw = s["yaw"]
            R = np.array([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]])
            q = quat_from_rotmat(np.array([[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]]))
            for k, (dx, dy, sx, sy) in enumerate([(hx + wall / 2, 0, wall / 2, hy + wall), (-(hx + wall / 2), 0, wall / 2, hy + wall),
                                                  (0, hy + wall / 2, hx + wall, wall / 2), (0, -(hy + wall / 2), hx + wall, wall / 2)]):
                c = np.array([cx, cy]) + R @ np.array([dx, dy])
                w.add_geom(name="socket_%s_%d" % (name, k), type=mujoco.mjtGeom.mjGEOM_BOX, size=[sx, sy, sock_h / 2],
                           pos=[c[0], c[1], ts[2] + sock_h / 2], quat=q, material="mat_socket")

    # cameras
    for cname, cam in cfg["cameras"].items():
        w.add_camera(name=cname, pos=cam["pos"], quat=_lookat_quat(cam["pos"], cam["lookat"], cam.get("up", (0, 0, 1))), fovy=cam["fovy"])

    # Panda
    panda = mujoco.MjSpec.from_file(os.path.join(panda_dir, "panda.xml"))
    for k in list(panda.keys):
        panda.delete(k)
    frame = w.add_frame(pos=cfg["panda_pos"])
    frame.attach_body(panda.worldbody.first_body(), "panda/", "")
    for b in spec.bodies:
        if b.name.startswith("panda/") and b.name != "panda/link0":
            b.gravcomp = float(cfg["panda_gravcomp"])
    hand = spec.body("panda/hand")
    hand.add_site(name="tcp", pos=[0.0, 0.0, cfg["tcp_offset_m"]], size=[0.004, 0.004, 0.004], group=4, rgba=[0.2, 0.9, 0.2, 0.6])
    hand.add_site(name="hand_ft", pos=[0.0, 0.0, 0.0], size=[0.003, 0.003, 0.003], group=4)
    wc = cfg["wrist_camera"]
    hand.add_camera(name="wrist", pos=wc["pos"], quat=quat_from_rotmat(np.column_stack([[1.0, 0.0, 0.0], [0.0, 0.55, 0.835], np.cross([1.0, 0.0, 0.0], [0.0, 0.55, 0.835])])), fovy=wc["fovy"])
    spec.add_sensor(name="hand_force", type=mujoco.mjtSensor.mjSENS_FORCE, objtype=mujoco.mjtObj.mjOBJ_SITE, objname="hand_ft")
    spec.add_sensor(name="hand_torque", type=mujoco.mjtSensor.mjSENS_TORQUE, objtype=mujoco.mjtObj.mjOBJ_SITE, objname="hand_ft")
    # pads: friction of rubber; name the pad geoms so the server can tell pads from links
    pad_geoms = []
    for g in spec.geoms:
        b = g.parent
        if b is not None and b.name in ("panda/left_finger", "panda/right_finger") and g.type == mujoco.mjtGeom.mjGEOM_BOX:
            side = "left" if "left" in b.name else "right"
            g.name = "panda/pad_%s_%d" % (side, len([p for p in pad_geoms if side in p]))
            g.friction = cfg["pad_friction"]
            pad_geoms.append(g.name)
    # gripper actuator: our own position actuator on the finger tendon (half-width target)
    gr = cfg["gripper"]
    for a in spec.actuators:
        if a.name == "panda/actuator8":
            a.name = "panda/gripper"
            a.gainprm = [gr["kp"], 0, 0, 0, 0, 0, 0, 0, 0, 0]
            a.biasprm = [0, -gr["kp"], -gr["kv"], 0, 0, 0, 0, 0, 0, 0]
            a.ctrlrange = [0.0, gr["max_width_m"] / 2]
            a.forcerange = [-gr["max_force_N"], gr["max_force_N"]]

    # loose parts
    servo_vis = None
    material_of = {"shoulder": "mat_pla_shoulder", "upper_arm": "mat_pla_arm", "forearm": "mat_pla_arm"}
    servo_names = []
    for name in loose:
        p = parts[name]
        lp = layout["parts"][name]
        body = w.add_body(name=name, pos=lp["pos"], quat=lp["quat"])
        body.add_freejoint(name="free_" + name)
        if p["kind"] == "servo":
            servo_names.append(name)
            for bname, half, centre in _servo_boxes(p, shrink):
                body.add_geom(name="%s_%s" % (name, bname), type=mujoco.mjtGeom.mjGEOM_BOX, size=half.tolist(), pos=centre.tolist(),
                              group=3, material="mat_hull")
            if with_visual:
                if servo_vis is None:
                    servo_vis = add_mesh("servo_standin_visual", p["mesh_visual"], MM)
                body.add_geom(name=name + "_visual", type=mujoco.mjtGeom.mjGEOM_MESH, meshname=servo_vis, contype=0, conaffinity=0, group=2, material="mat_servo")
            body.mass = float(p["mass_kg"])
            body.ipos = (np.asarray(p["com_mm"], float) * MM).tolist()
            I = np.asarray(p["inertia_kg_mm2"], float) * 1e-6
            body.explicitinertial = True
            body.fullinertia = [I[0, 0], I[1, 1], I[2, 2], I[0, 1], I[0, 2], I[1, 2]]
            body.add_site(name="servo_frame_" + name, pos=[0, 0, 0], size=[0.002] * 3, group=4)
            # horn: unpowered hinge about the horn axis, visual disc, actuator in group 1
            hp = [0.0, 0.007, 0.046]
            horn = body.add_body(name="horn_" + name, pos=hp)
            hh = cfg["horn_hinge"]
            horn.add_joint(name="hj_" + name, type=mujoco.mjtJoint.mjJNT_HINGE, axis=[0, 0, 1], frictionloss=hh["frictionloss"],
                           damping=hh["damping"], armature=hh["armature"])
            horn.add_geom(name="horn_disc_" + name, type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.005, 0.0015, 0.0], pos=[0, 0, 0.0015],
                          contype=0, conaffinity=0, group=2, material="mat_horn", mass=0.002)
            horn.add_geom(name="horn_mark_" + name, type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.0008, 0.0035, 0.0003], pos=[0, 0.0035, 0.0031],
                          contype=0, conaffinity=0, group=2, rgba=[0.1, 0.1, 0.1, 1], mass=0.0)
            horn.add_site(name="horn_" + name, pos=[0, 0, 0], size=[0.002] * 3, group=4)
            sa = cfg["servo_actuator"]
            act = spec.add_actuator(name="act_horn_" + name)
            act.target = "hj_" + name
            act.trntype = mujoco.mjtTrn.mjTRN_JOINT
            act.gaintype = mujoco.mjtGain.mjGAIN_FIXED
            act.biastype = mujoco.mjtBias.mjBIAS_AFFINE
            act.gainprm = [sa["kp"], 0, 0, 0, 0, 0, 0, 0, 0, 0]
            act.biasprm = [0, -sa["kp"], -sa["kv"], 0, 0, 0, 0, 0, 0, 0]
            stall = float(p.get("stall_torque_nm", 3.0))
            act.forcerange = [-stall, stall]
            act.group = 1
        else:
            add_printed(body, name, material_of.get(name, "mat_pla"))
            body.add_site(name="part_frame_" + name, pos=[0, 0, 0], size=[0.002] * 3, group=4)

    # welds: (carrier, servo) for every seat pocket x every servo of the same SKU; (horn, link) per horn mate servo
    welds = {}
    wsr = cfg["weld_solref"]
    catalogue = {n: parts[n].get("catalogue_id") for n in servo_names}
    servo_of_seat = {}
    for m in mates:
        if m["kind"] in ("sleeve_seat", "part_seat"):
            carrier, target = m["part_a"], m["part_b"]
            cands = [target] if parts[target]["kind"] != "servo" else [s for s in servo_names if catalogue[s] == parts[target]["catalogue_id"]]
            servo_of_seat[m["id"]] = cands
            for s in cands:
                key = (carrier, s)
                if key not in welds:
                    eq = spec.add_equality(name="weld_%s__%s" % key, type=mujoco.mjtEq.mjEQ_WELD, objtype=mujoco.mjtObj.mjOBJ_BODY, name1=carrier, name2=s, active=False, solref=wsr)
                    welds[key] = eq.name
        elif m["kind"] == "horn_mate":
            link = m["part_b"]
            for s in m.get("servos", [m["part_a"]]):
                cands = [c for c in servo_names if catalogue[c] == parts[s]["catalogue_id"]]
                for c in cands:
                    key = ("horn_" + c, link)
                    if key not in welds:
                        eq = spec.add_equality(name="weld_%s__%s" % key, type=mujoco.mjtEq.mjEQ_WELD, objtype=mujoco.mjtObj.mjOBJ_BODY, name1="horn_" + c, name2=link, active=False, solref=wsr)
                        welds[key] = eq.name

    model = spec.compile()
    # keyframe home: Panda home (+ layout noise), gripper open, parts at their layout poses
    qpos = np.array(model.qpos0, float).copy()
    ctrl = np.zeros(model.nu)
    noise = np.asarray(layout.get("panda_joint_noise_rad", [0.0] * 7), float)
    for i, jn in enumerate(PANDA_JOINTS):
        j = model.joint("panda/" + jn)
        qpos[model.jnt_qposadr[j.id]] = PANDA_HOME[i] + noise[i]
        ctrl[model.actuator("panda/actuator%d" % (i + 1)).id] = PANDA_HOME[i] + noise[i]
    for fn in ("finger_joint1", "finger_joint2"):
        qpos[model.jnt_qposadr[model.joint("panda/" + fn).id]] = gr["max_width_m"] / 2
    ctrl[model.actuator("panda/gripper").id] = gr["max_width_m"] / 2
    for name in loose:
        adr = model.jnt_qposadr[model.joint("free_" + name).id]
        qpos[adr:adr + 3] = layout["parts"][name]["pos"]
        qpos[adr + 3:adr + 7] = layout["parts"][name]["quat"]
    spec.add_key(name="home", qpos=qpos.tolist(), ctrl=ctrl.tolist())
    model = spec.compile()

    # bookkeeping for the server
    meta = {
        "cell_version": CELL_VERSION, "config": cfg, "scope_ids": scope_ids, "root": root, "loose": loose, "servos": servo_names,
        "layout": layout, "mesh_files": mesh_files, "welds": welds, "servo_of_seat": servo_of_seat,
        "pad_geoms": pad_geoms, "mates": _mates_metric(mates, parts),
        "parts": parts, "graph": graph, "manifest": manifest,
        "panda_home": list(PANDA_HOME),
    }
    return CellBuild(spec, model, meta)


def _mates_metric(mates, parts):
    """Mate list with frames as 4x4 metres and tolerances in metres / radians."""
    out = []
    for m in mates:
        e = {"id": m["id"], "kind": m["kind"], "part_a": m["part_a"], "part_b": m["part_b"], "weight": float(m["weight"]),
             "requires": list(m.get("requires", [])), "tol_pos": float(m["tol_pos_mm"]) * MM, "tol_ang": math.radians(float(m["tol_ang_deg"])),
             "F_a": pose_mm(m["frame_a_in_a"]), "F_b": pose_mm(m["frame_b_in_b"]),
             "insertion_axis_in_a": np.asarray(m["insertion_axis_in_a"], float), "insertion_depth": float(m["insertion_depth_mm"]) * MM,
             "joint": m.get("joint"), "servos": list(m.get("servos", [m["part_a"]] if m["kind"] == "horn_mate" else []))}
        if m["kind"] == "horn_mate":
            e["axial_tol"] = float(m.get("axial_tol_mm", 1.0)) * MM
            e["yaw_modulo"] = math.radians(float(m.get("yaw_modulo_deg", 90)))
            fb = m.get("frames_b_in_b") or {m["part_a"]: m["frame_b_in_b"]}
            e["F_b_per_servo"] = {s: pose_mm(f) for s, f in fb.items()}
        out.append(e)
    return out


# ------------------------------------------------------------------------------- collision-model gate
def collision_model_gate(a2_dir, config=None, scope_ids=None, max_penetration_mm=0.1, sweep_mm=None, collision_cache=None):
    """Build-time gate (CHAIN_PLAN 4.3): every servo primitive at its designed pose must not penetrate
    the carved carrier, nor at lateral offsets of +-sweep_mm (default: side shrink + carve margin - tolerance,
    the pocket clearance the collision model promises),
    and every horn-side link at its designed pose must not penetrate the servo(s).
    Returns (ok, details) with the worst penetration per mate (mm)."""
    import mujoco
    cfg = json.loads(json.dumps(DEFAULT_CELL_CONFIG))
    for k, v in (config or {}).items():
        if isinstance(v, dict) and isinstance(cfg.get(k), dict):
            cfg[k].update(v)
        else:
            cfg[k] = v
    if sweep_mm is None:
        # the lateral clearance the collision model promises: side shrink + carve margin, minus the tolerance
        sweep_mm = float(cfg["shrink_mm"]["body_side"]) + float(cfg["pocket_carve_margin_mm"]) - max_penetration_mm
    manifest, graph = load_givens(a2_dir)
    parts = {p["name"]: p for p in manifest["parts"]}
    scope_ids = list(scope_ids or graph["scopes"]["rung1_prefix"])
    mates = [m for m in graph["mates"] if m["id"] in set(scope_ids)]
    cache = collision_cache if collision_cache is not None else {}
    results = {}
    ok = True

    def pieces_of(name):
        if name not in cache:
            cache[name] = carved_collision(a2_dir, parts[name], parts, cfg["pocket_carve_margin_mm"], cfg["pocket_carve_min_volume_mm3"], cfg["pocket_carve_min_thickness_mm"], cfg.get("pocket_carve_ear_margin_mm"))
        return cache[name]

    def probe(m, offsets):
        worst = 0.0
        for off in offsets:
            spec = mujoco.MjSpec()
            spec.compiler.degree = False
            if not cfg.get("multiccd", False):
                spec.option.disableflags |= mujoco.mjtDisableBit.mjDSBL_MULTICCD
            w = spec.worldbody

            def add_body(name, T, free):
                b = w.add_body(name=name, pos=T[:3, 3].tolist(), quat=quat_from_rotmat(T[:3, :3]))
                if free:
                    b.add_freejoint(name="f_" + name)
                    b.mass = 0.1
                    b.explicitinertial = True
                    b.fullinertia = [1e-5, 1e-5, 1e-5, 0, 0, 0]
                p = parts[name]
                if p["kind"] == "servo":
                    for bname, half, centre in _servo_boxes(p, cfg["shrink_mm"]):
                        b.add_geom(name="%s_%s" % (name, bname), type=mujoco.mjtGeom.mjGEOM_BOX, size=half.tolist(), pos=centre.tolist())
                else:
                    for i, piece in enumerate(pieces_of(name)):
                        mn = "%s_c%02d" % (name, i)
                        spec.add_mesh(name=mn, uservert=(np.asarray(piece.vertices, float) * MM).ravel().tolist())
                        b.add_geom(name=mn, type=mujoco.mjtGeom.mjGEOM_MESH, meshname=mn)
                return b
            F_a, F_b = pose_mm(m["frame_a_in_a"]), pose_mm(m["frame_b_in_b"])
            if m["kind"] in ("sleeve_seat", "part_seat"):
                add_body(m["part_a"], np.eye(4), False)
                T = F_a @ inv(F_b)
                T = T.copy()
                T[:3, 3] += F_a[:3, :3] @ np.asarray(off, float)
                add_body(m["part_b"], T, True)
            elif m["kind"] == "horn_mate":
                add_body(m["part_b"], np.eye(4), False)
                fb = m.get("frames_b_in_b") or {m["part_a"]: m["frame_b_in_b"]}
                for s_, f in fb.items():
                    add_body(s_, pose_mm(f) @ inv(F_a), True)
            else:
                return None
            model = spec.compile()
            data = mujoco.MjData(model)
            mujoco.mj_forward(model, data)
            for c in range(data.ncon):
                worst = max(worst, -float(data.contact[c].dist))
        return worst

    for m in mates:
        if m["part_a"] == "fixture":
            continue
        d = sweep_mm * MM
        if m["kind"] in ("sleeve_seat", "part_seat") and parts[m["part_b"]]["kind"] == "servo":
            offsets = [(0, 0, 0), (d, 0, 0), (-d, 0, 0), (0, d, 0), (0, -d, 0), (0, 0, d)]
        else:
            offsets = [(0, 0, 0)]
        worst = probe(m, offsets)
        if worst is None:
            continue
        results[m["id"]] = {"max_penetration_mm": worst / MM, "offsets_mm": len(offsets)}
        if worst / MM > max_penetration_mm:
            ok = False
    return ok, results


# ------------------------------------------------------------------------------- the agent-facing XML
def write_given_cell_xml(build, out_dir, panda_dir, name="assembly_cell.xml"):
    """Write the nominal cell as one XML + assets/ folder the agent can load (meshes copied, paths relative)."""
    xml = build.spec.to_xml()
    root = ET.fromstring(xml)
    assets = os.path.join(out_dir, "assets")
    os.makedirs(assets, exist_ok=True)
    comp = root.find("compiler")
    if comp is None:
        comp = ET.SubElement(root, "compiler")
    comp.set("meshdir", "assets")
    comp.set("angle", "radian")
    for mesh in root.iter("mesh"):
        f = mesh.get("file")
        if not f:
            continue     # inline vertices (carved collision pieces) need no file
        base = os.path.basename(f)
        src = f if os.path.isabs(f) else None
        if src is None:
            for cand in (os.path.join(panda_dir, "assets", base), os.path.join(panda_dir, base)):
                if os.path.isfile(cand):
                    src = cand
                    break
        if src is None:
            src = build.meta["mesh_files"].get(mesh.get("name"))
        if src and os.path.isfile(src):
            dst = os.path.join(assets, base)
            if not os.path.isfile(dst):
                shutil.copyfile(src, dst)
        mesh.set("file", base)
    ET.indent(root, space="  ")
    text = ET.tostring(root, encoding="unicode")
    path = os.path.join(out_dir, name)
    with open(path, "w") as fh:
        fh.write(text + "\n")
    return path
