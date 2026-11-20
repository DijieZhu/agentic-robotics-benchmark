"""Build the six-servo arm's assembly cell (MjSpec -> MjModel) from the A2 givens, the model collection Panda, a hidden layout
and a pre-assembly specification.

Same physics as the chain's cell (CHAIN_PLAN 4.3 as built: MuJoCo 1 kHz implicitfast, elliptic cones, impratio 10,
noslip 5, multiccd off, CoACD hulls carved to the servo keep-outs swept along the measured insertion paths, servos as
shrunk boxes, horn and rear bosses visual-only, unpowered back-drivable horn hinges, `<weld active="false">` per
candidate pair, servo actuators in group 1). Differences for THIS arm:

  * one collision box per servo (the case incl. the bottom cover; no ear tabs), shrink classes per face:
    `long_sides` (+-y), `horn_face` (+z), `rear_face` (-z), `horn_end` (+x), `connector_end` (-x)
  * the horn hinge sits at the servo's own horn axis point (12.5 mm off the case centre, from the manifest)
  * PRE-ASSEMBLED GROUPS (`preassembly.groups`): the parts of a group are placed at the design FK for the group's joint
    configuration and become joint-less child bodies of the group's root (a fixed root on the fixture = the proximal
    half of the arm; a free root = the distal compound), so every part keeps its own body (poses, contacts) while the
    group moves as one rigid body; pre-assembled servos carry no hinge (their horns are welded by design)
  * printed parts are rendered in one saturated blue material (owner direction), servos dark grey, horns brass

World frame: table top centre, z up. Panda base at (-0.45, 0, 0); fixture plate 0.14 m at (0.10, 0, 0) +- hidden jitter,
top at z = 0.02, the base part fixed on it yawed by `layout.base_yaw_rad`; trays at y = +-0.32 m with sockets / a stand;
cameras overview / top / oblique / wrist.
"""

import json
import math
import os
import shutil
import xml.etree.ElementTree as ET

import numpy as np

from ..assembly_cell.cell_builder import (MM, PANDA_HOME, PANDA_JOINTS, _clip_convex, _clip_to_bbox, _lookat_quat, carve_hulls, inv,
                                          pose_mm, quat_to_mat_np)
from ..chain_model import Chain
from ..frames import pose_from_dict
from . import CELL_VERSION
from .layout import DEFAULT_LAYOUT_CONFIG, quat_from_rotmat, rest_pose_for, rotz

__all__ = ["DEFAULT_CELL_CONFIG", "DEFAULT_PREASSEMBLY", "CellBuild", "build_cell", "load_givens", "scope_parts", "group_poses",
           "rest_poses", "mate_world_frames", "collision_model_gate", "write_given_cell_xml", "servo_boxes", "carved_collision"]

DEFAULT_CELL_CONFIG = {
    "arena_memory_bytes": 134217728,      # MuJoCo constraint / stack arena (default ~15 MB overflowed under a contact storm)
    "timestep": 0.001,
    "tick_steps": 10,
    "impratio": 10.0,
    "multiccd": False,
    "noslip_iterations": 5,
    "shrink_mm": {"long_sides": 0.4, "horn_face": 0.8, "rear_face": 0.8, "horn_end": 0.4, "connector_end": 0.4},
    "carve_margin_mm": 0.3,
    "carve_sweep_extra_mm": 6.0,
    "carve_min_volume_mm3": 0.5,
    "carve_min_thickness_mm": 1.2,
    # the horn hinge carries the servo's reflected gearbox inertia (armature): with a near-massless horn between a stiff
    # position actuator and a stiff weld to a 0.29 kg compound the hinge oscillated at 10 rad/s and saturated the actuator
    # (measured); 5e-3 kg m^2 makes the driven elbow track within 0.5 deg
    "horn_hinge": {"frictionloss": 0.1, "damping": 0.05, "armature": 5e-3},
    "gripper": {"kp": 2500.0, "kv": 50.0, "max_force_N": 60.0, "max_width_m": 0.08},
    "panda_pos": [-0.45, 0.0, 0.0],
    "panda_gravcomp": 1.0,
    "table": {"size_xy": [1.2, 0.8], "thickness": 0.05, "floor_z": -0.75},
    "fixture": {"size_m": 0.14, "thickness_m": 0.02, "pin_d_mm": 4.0, "pin_h_mm": 12.0, "pin_xy_mm": 38.0},
    "geom_defaults": {"condim": 4, "friction": [0.8, 0.005, 0.0001], "solref": [0.004, 1.0], "solimp": [0.95, 0.99, 0.001, 0.5, 2.0]},
    "pad_friction": [1.0, 0.005, 0.0001],
    "weld_solref": [0.002, 1.0],
    "weld_solimp": [0.99, 0.999, 0.001, 0.5, 2.0],    # the welds carry a 0.29 kg compound: the default impedance (0.9-0.95) let it sag 2 deg per weld
    "horn_body": {"mass_kg": 0.005, "radius_m": 0.010, "height_m": 0.004},  # horn disc + screws (5 g)
    "servo_actuator": {"kp": 60.0, "kv": 2.0},
    "materials": {"printed": [0.10, 0.32, 0.88, 1.0], "servo": [0.20, 0.20, 0.23, 1.0], "horn": [0.85, 0.72, 0.30, 1.0]},
    "cameras": {
        "overview": {"pos": [0.9, 0.9, 0.9], "lookat": [0.15, 0.0, 0.15], "fovy": 45},
        "top": {"pos": [0.15, 0.0, 1.2], "lookat": [0.15, 0.0, 0.0], "fovy": 58, "up": [1.0, 0.0, 0.0]},
        "oblique": {"pos": [0.75, -0.6, 0.6], "lookat": [0.15, 0.0, 0.15], "fovy": 50},
    },
    "wrist_camera": {"pos": [0.0, -0.06, 0.03], "fovy": 70},
    "tcp_offset_m": 0.1034,
}

# the rung-1 pre-assembly (the reference benchmark's stage): proximal half fixed on the fixture with the upper arm horizontal
# (shoulder_pitch -pi/2 turns the elbow pocket's mouth up), distal half as one compound at its straight configuration
DEFAULT_PREASSEMBLY = {
    "groups": [
        {"name": "proximal", "root": "base", "fixed": True, "parts": ["base_bracket", "S1", "shoulder", "shoulder_bracket", "S2", "upper_arm"],
         "q": {"base_yaw": 0.0, "shoulder_pitch": -1.5707963}},
        {"name": "distal", "root": "forearm", "fixed": False, "parts": ["forearm_bracket", "S4", "wrist", "S5", "gripper_carrier", "S6", "jaw_moving"],
         "q": {"wrist_pitch": 0.0, "wrist_roll": 0.0, "jaw": 0.0}},
    ],
}
SERVO_AXES = {"long_sides": [1, 3], "connector_end": [0], "horn_end": [0], "rear_face": [2], "horn_face": [2]}


def load_givens(a2_dir):
    with open(os.path.join(a2_dir, "parts_manifest.json")) as f:
        manifest = json.load(f)
    with open(os.path.join(a2_dir, "mating_graph.json")) as f:
        graph = json.load(f)
    return manifest, graph


def _cfg(config):
    cfg = json.loads(json.dumps(DEFAULT_CELL_CONFIG))
    for k, v in (config or {}).items():
        if isinstance(v, dict) and isinstance(cfg.get(k), dict):
            cfg[k].update(v)
        else:
            cfg[k] = v
    return cfg


class CellBuild(object):
    def __init__(self, spec, model, meta):
        self.spec = spec
        self.model = model
        self.meta = meta


def servo_boxes(part, shrink):
    """(name, half-size m, centre m) of the shrunk collision box(es) of a servo (servo frame)."""
    out = []
    for pr in part["collision_primitives"]:
        mn = np.asarray(pr["min"], float).copy()
        mx = np.asarray(pr["max"], float).copy()
        mn += [shrink["connector_end"], shrink["long_sides"], shrink["rear_face"]]
        mx -= [shrink["horn_end"], shrink["long_sides"], shrink["horn_face"]]
        out.append((pr["name"], (mx - mn) / 2 * MM, (mx + mn) / 2 * MM))
    return out


def group_poses(design, preassembly):
    """Pose of every part of every group in its group root's frame (mm), from the design FK at the group's q.
    Returns {group name: {"root", "fixed", "q_full": {joint: value}, "parts": {name: T_in_root}}}."""
    chain = Chain(design)
    out = {}
    for g in preassembly.get("groups", []):
        q = np.zeros(len(chain.joints))
        for i, j in enumerate(chain.joints):
            q[i] = float(g.get("q", {}).get(j.name, 0.0))
        T = chain.fk(q)
        T_root = T[g["root"]]
        poses = {}
        link_of_part = {}
        for l in design["links"]:
            for ap in l.get("attached_parts", []):
                link_of_part[ap["name"]] = (l["name"], pose_from_dict(ap))
            for occ in l.get("servo_occurrences", []):
                link_of_part[occ["servo_id"]] = (l["name"], pose_from_dict(occ))
        for name in g["parts"]:
            if name in T:
                Tw = T[name]
            else:
                link, T_att = link_of_part[name]
                Tw = T[link] @ T_att
            poses[name] = inv(T_root) @ Tw
        out[g["name"]] = {"root": g["root"], "fixed": bool(g.get("fixed", False)), "q_full": {j.name: float(q[i]) for i, j in enumerate(chain.joints)},
                          "parts": poses}
    return out


def scope_parts(graph, scope_ids, manifest, preassembly):
    """(root, loose bodies, mates in scope, pre-assembled part -> group). Loose bodies = the scope's parts that are not in a
    fixed group, with a group's members collapsed onto the group's root."""
    mates = [m for m in graph["mates"] if m["id"] in set(scope_ids)]
    parts = {p["name"]: p for p in manifest["parts"]}
    member_of = {}
    for g in preassembly.get("groups", []):
        for n in g["parts"]:
            member_of[n] = g
        member_of[g["root"]] = g
    root = "base"
    loose = []
    for m in mates:
        for key in ("part_a", "part_b"):
            n = m[key]
            if n == "fixture":
                continue
            g = member_of.get(n)
            if g is not None:
                if g.get("fixed", False):
                    continue
                n = g["root"]
            if n != root and n not in loose:
                loose.append(n)
        for s in m.get("servos", []):
            g = member_of.get(s)
            if g is None and s not in loose:
                loose.append(s)
    # interchangeable servos: every loose servo of the SKU is a candidate for a scoped seat (all six share one SKU); loose = in scope only
    return root, loose, mates, member_of


def mate_world_frames(design, graph, preassembly, base_yaw_rad, gp=None):
    """World pose (nominal fixture at the origin, base yawed) of every loose part AT ITS MATE: a servo at its seat, a
    compound root at its horn mate. Used by the layout to pre-orient the trays (rung 1) and by the oracle-free checks."""
    gp = gp or group_poses(design, preassembly)
    chain = Chain(design)
    R_base = np.eye(4)
    R_base[:3, :3] = rotz(float(base_yaw_rad))
    # world pose of every fixed-group member (relative to the base)
    fixed_w = {"base": R_base}
    for g in gp.values():
        if g["fixed"]:
            for n, T in g["parts"].items():
                fixed_w[n] = R_base @ T
    out = {}
    mates = {m["id"]: m for m in graph["mates"]}
    for m in graph["mates"]:
        if m["kind"] == "sleeve_seat" and m["part_a"] in fixed_w:
            F_a, F_b = pose_mm(m["frame_a_in_a"]), pose_mm(m["frame_b_in_b"])
            out[m["part_b"]] = fixed_w[m["part_a"]] @ F_a @ inv(F_b)
    for m in graph["mates"]:
        if m["kind"] == "horn_mate" and m["part_a"] in out:
            F_a, F_b = pose_mm(m["frame_a_in_a"]), pose_mm(m["frame_b_in_b"])
            out[m["part_b"]] = out[m["part_a"]] @ F_a @ inv(F_b)
    return out


def rest_poses(design, manifest, a2_dir, preassembly, loose, gp=None, graph=None, base_yaw_rad=0.0):
    """Rest description per loose body (`layout.rest_pose_for`): servos on their connector end, compounds fork-down, each
    yawed so that its horn / elbow axis points the way it does at the mate (a top-down or side grasp then needs no hand
    yaw beyond the tray jitter); the compound's stand footprint is the ROOT part's, its height the whole compound's."""
    parts = {p["name"]: p for p in manifest["parts"]}
    gp = gp or group_poses(design, preassembly)
    mates_w = mate_world_frames(design, graph, preassembly, base_yaw_rad, gp) if graph is not None else {}
    out = {}
    for n in loose:
        p = parts[n]
        Tm = mates_w.get(n)
        if p["kind"] == "servo":
            lo, hi = np.asarray(p["bbox_mm"][0], float) * MM, np.asarray(p["bbox_mm"][1], float) * MM
            corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2] + 0.004)])
            side_world = Tm[:3, 2] if Tm is not None else (0.0, 1.0, 0.0)          # the horn axis as at the seat
            out[n] = rest_pose_for("servo_on_connector_end", corners, down_part=[-1.0, 0.0, 0.0], side_part=[0.0, 0.0, 1.0], side_world=side_world)
            continue
        group = [g for g in gp.values() if g["root"] == n]
        pts = []
        members = {n: np.eye(4)}
        if group:
            members.update(group[0]["parts"])
        for name, T in members.items():
            q = parts[name]
            lo, hi = np.asarray(q["bbox_mm"][0], float), np.asarray(q["bbox_mm"][1], float)
            c = np.array([[x, y, z, 1.0] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
            pts.append((T @ c.T).T[:, :3] * MM)
        P = np.vstack(pts)
        side_world = Tm[:3, 2] if Tm is not None else (0.0, 1.0, 0.0)              # the elbow axis as at the mate
        rest = rest_pose_for("compound_fork_down", P, down_part=[1.0, 0.0, 0.0], side_part=[0.0, 0.0, 1.0], side_world=side_world)
        root_only = rest_pose_for("compound_fork_down", pts[0], down_part=[1.0, 0.0, 0.0], side_part=[0.0, 0.0, 1.0], side_world=side_world)
        rest["footprint_half"] = root_only["footprint_half"]
        rest["centre_part"] = root_only["centre_part"]
        out[n] = rest
    return out


def _keepout_boxes(part_name, parts, mates_all, graph_parts_frame, margin_mm, sweep_extra_mm):
    """Keep-out boxes (T_box_in_part mm, half mm, centre mm) to carve out of `part_name`'s hulls: every servo that this
    printed part receives at a seat (body at the design pose swept along the extraction direction over the pocket depth)
    and, for a horn-side link, the seated servo swept along the link's slide-in direction."""
    out = []
    for m in mates_all:
        if m["kind"] == "sleeve_seat" and m["part_a"] == part_name:
            s = parts[m["part_b"]]
            T = pose_mm(m["frame_a_in_a"])
            T[:3, 3] /= MM
            ext = -np.asarray(m.get("insertion_axis_in_b", [-1.0, 0.0, 0.0]), float)   # extraction, servo frame
            depth = float(m.get("insertion_depth_mm", 20.0)) + sweep_extra_mm
            for pr in s["collision_primitives"]:
                mn = np.asarray(pr["min"], float) - margin_mm
                mx = np.asarray(pr["max"], float) + margin_mm
                # sweep: grow the box along the extraction direction
                for k in range(3):
                    if ext[k] > 0.5:
                        mx[k] += depth
                    elif ext[k] < -0.5:
                        mn[k] -= depth
                out.append((T, (mx - mn) / 2, (mx + mn) / 2))
        elif m["kind"] == "horn_mate" and m["part_b"] == part_name:
            s = parts[m["part_a"]]
            T_horn = pose_mm(m["frame_a_in_a"])           # horn in servo (m)
            T_b = pose_mm(m["frame_b_in_b"])              # horn in link (m)
            T_servo_in_link = T_b @ inv(T_horn)
            T_servo_in_link = T_servo_in_link.copy()
            T_servo_in_link[:3, 3] /= MM
            # the link slides onto the servo along insertion_axis_in_a (horn frame == servo axes); relative to the link the
            # servo moves the opposite way: sweep the servo box along -approach (servo frame)
            appr = np.asarray(m.get("insertion_axis_in_a", [0.0, 0.0, -1.0]), float)
            travel = float(m.get("insertion_depth_mm", 4.0)) + sweep_extra_mm
            for pr in s["collision_primitives"]:
                mn = np.asarray(pr["min"], float) - margin_mm
                mx = np.asarray(pr["max"], float) + margin_mm
                for k in range(3):
                    if -appr[k] > 0.5:
                        mx[k] += travel
                    elif -appr[k] < -0.5:
                        mn[k] -= travel
                out.append((T_servo_in_link, (mx - mn) / 2, (mx + mn) / 2))
    return out


def carved_collision(a2_dir, part, parts, graph, cfg):
    """The printed part's collision pieces (trimesh, mm): its CoACD hulls minus the keep-outs of the servos it receives."""
    import trimesh
    hulls = [trimesh.load(os.path.join(a2_dir, rel), force="mesh") for rel in part["mesh_collision"]]
    boxes = _keepout_boxes(part["name"], parts, graph["mates"], None, cfg["carve_margin_mm"], cfg["carve_sweep_extra_mm"])
    pieces = carve_hulls(hulls, boxes, cfg["carve_min_volume_mm3"], cfg["carve_min_thickness_mm"]) if boxes else hulls
    return _clip_to_bbox(pieces, part["bbox_mm"], min_volume_mm3=cfg["carve_min_volume_mm3"])


def build_cell(a2_dir, panda_dir, layout, config=None, scope_ids=None, manifest=None, graph=None, design=None, preassembly=None,
               with_visual=True, collision_cache=None):
    """Compile the cell for one episode. `design` = the assembly.json dict (mm); `preassembly` = the groups spec."""
    import mujoco
    cfg = _cfg(config)
    if manifest is None or graph is None:
        manifest, graph = load_givens(a2_dir)
    parts = {p["name"]: p for p in manifest["parts"]}
    scope_ids = list(scope_ids or graph["scopes"]["elbow"])
    preassembly = preassembly or DEFAULT_PREASSEMBLY
    root, loose, mates, member_of = scope_parts(graph, scope_ids, manifest, preassembly)
    gp = group_poses(design, preassembly)
    shrink = cfg["shrink_mm"]
    mat = cfg["materials"]

    spec = mujoco.MjSpec()
    spec.modelname = "six_servo_arm_assembly_cell"
    # the constraint arena: a contact storm (the 0.46 m compound pressed into a tray: 326 contacts, 1306 constraint rows)
    # overflowed MuJoCo's default arena inside the grader ("mj_stackAlloc: out of memory" at mj_makeARNumeric); 128 MB
    spec.memory = int(cfg.get("arena_memory_bytes", 128 << 20))
    spec.compiler.degree = False
    spec.compiler.autolimits = True
    spec.option.timestep = float(cfg["timestep"])
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    spec.option.noslip_iterations = int(cfg["noslip_iterations"])
    spec.option.impratio = float(cfg["impratio"])
    spec.option.gravity = [0.0, 0.0, -9.81]
    spec.option.disableactuator = 1 << 1
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
    spec.stat.center = [0.15, 0.0, 0.2]

    w = spec.worldbody
    w.add_light(name="sun", pos=[0.3, -0.3, 2.0], dir=[-0.15, 0.15, -1.0], type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL, diffuse=[0.7, 0.7, 0.7])
    w.add_light(name="fill", pos=[-0.5, 0.8, 1.2], dir=[0.4, -0.6, -0.8], diffuse=[0.35, 0.35, 0.35])
    tb = cfg["table"]
    spec.add_material(name="mat_table", rgba=[0.82, 0.80, 0.76, 1.0])
    spec.add_material(name="mat_fixture", rgba=[0.35, 0.38, 0.42, 1.0])
    spec.add_material(name="mat_tray", rgba=[0.55, 0.58, 0.62, 1.0])
    spec.add_material(name="mat_socket", rgba=[0.42, 0.45, 0.50, 1.0])
    spec.add_material(name="mat_printed", rgba=list(mat["printed"]))
    spec.add_material(name="mat_servo", rgba=list(mat["servo"]))
    spec.add_material(name="mat_horn", rgba=list(mat["horn"]))
    spec.add_material(name="mat_hull", rgba=[0.9, 0.3, 0.3, 0.3])
    w.add_geom(name="table_top", type=mujoco.mjtGeom.mjGEOM_BOX, size=[tb["size_xy"][0] / 2, tb["size_xy"][1] / 2, tb["thickness"] / 2],
               pos=[0.0, 0.0, -tb["thickness"] / 2], material="mat_table")
    w.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[3.0, 3.0, 0.1], pos=[0.0, 0.0, tb["floor_z"]], rgba=[0.3, 0.3, 0.3, 1.0])
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

    def set_inertia(body, p):
        body.mass = float(p["mass_kg"])
        body.ipos = (np.asarray(p["com_mm"], float) * MM).tolist()
        I = np.asarray(p["inertia_kg_mm2"], float) * 1e-6
        body.explicitinertial = True
        body.fullinertia = [I[0, 0], I[1, 1], I[2, 2], I[0, 1], I[0, 2], I[1, 2]]

    def add_printed(body, name):
        p = parts[name]
        if collision_cache is not None and name in collision_cache:
            pieces = collision_cache[name]
        else:
            pieces = carved_collision(a2_dir, p, parts, graph, cfg)
            if collision_cache is not None:
                collision_cache[name] = pieces
        for i, piece in enumerate(pieces):
            mn = "%s_c%02d" % (name, i)
            spec.add_mesh(name=mn, uservert=(np.asarray(piece.vertices, float) * MM).ravel().tolist())
            body.add_geom(name=mn, type=mujoco.mjtGeom.mjGEOM_MESH, meshname=mn, group=3, material="mat_hull")
        if with_visual:
            mv = add_mesh(name + "_visual", p["mesh_visual"], MM)
            body.add_geom(name=name + "_visual", type=mujoco.mjtGeom.mjGEOM_MESH, meshname=mv, contype=0, conaffinity=0, group=2, material="mat_printed")
        set_inertia(body, p)
        body.add_site(name="part_frame_" + name, pos=[0, 0, 0], size=[0.002] * 3, group=4)

    servo_names = []            # loose servos (with hinges)

    def add_servo(body, name, loose_servo):
        p = parts[name]
        for bname, half, centre in servo_boxes(p, shrink):
            body.add_geom(name="%s_%s" % (name, bname), type=mujoco.mjtGeom.mjGEOM_BOX, size=half.tolist(), pos=centre.tolist(), group=3, material="mat_hull")
        if with_visual:
            vis = add_mesh("servo_visual_" + os.path.basename(p["mesh_visual"]).replace(".stl", ""), p["mesh_visual"], MM)
            body.add_geom(name=name + "_visual", type=mujoco.mjtGeom.mjGEOM_MESH, meshname=vis, contype=0, conaffinity=0, group=2, material="mat_servo")
        set_inertia(body, p)
        body.add_site(name="servo_frame_" + name, pos=[0, 0, 0], size=[0.002] * 3, group=4)
        hp = (np.asarray(p["horn_axis_point_mm"], float) * MM).tolist()
        if loose_servo:
            servo_names.append(name)
            horn = body.add_body(name="horn_" + name, pos=hp)
            hh = cfg["horn_hinge"]
            horn.add_joint(name="hj_" + name, type=mujoco.mjtJoint.mjJNT_HINGE, axis=[0, 0, 1], frictionloss=hh["frictionloss"], damping=hh["damping"], armature=hh["armature"])
            hb = cfg["horn_body"]
            horn.add_geom(name="horn_disc_" + name, type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[hb["radius_m"], hb["height_m"] / 2, 0.0], pos=[0, 0, hb["height_m"] / 2],
                          contype=0, conaffinity=0, group=2, material="mat_horn", mass=float(hb["mass_kg"]))
            horn.add_geom(name="horn_mark_" + name, type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.0008, 0.0035, 0.0003], pos=[0, 0.0055, hb["height_m"] + 0.0002],
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

    def add_group_members(root_body, gname):
        g = gp[gname]
        for name, T in g["parts"].items():
            child = root_body.add_body(name=name, pos=(T[:3, 3] * MM).tolist(), quat=quat_from_rotmat(T[:3, :3]))
            if parts[name]["kind"] == "servo":
                add_servo(child, name, loose_servo=False)
            else:
                add_printed(child, name)

    # the base (root) on the fixture, yawed; its fixed group (the proximal half) as joint-less children
    R_base = rotz(float(layout.get("base_yaw_rad", 0.0)))
    base_body = w.add_body(name=root, pos=[fpos[0], fpos[1], fx["thickness_m"]], quat=quat_from_rotmat(R_base))
    add_printed(base_body, root)
    base_body.mass = 0.0
    base_body.explicitinertial = False
    fixed_groups = [n for n, g in gp.items() if g["fixed"] and g["root"] == root]
    for gname in fixed_groups:
        add_group_members(base_body, gname)
    preassembled = {}
    for gname, g in gp.items():
        for n in g["parts"]:
            preassembled[n] = gname

    # trays, sockets / stands
    ts = layout.get("tray_size_m", DEFAULT_LAYOUT_CONFIG["tray_size_m"])
    for tname, centre in layout["trays"].items():
        w.add_geom(name="tray_%s" % tname, type=mujoco.mjtGeom.mjGEOM_BOX, size=[ts[0] / 2, ts[1] / 2, ts[2] / 2], pos=[centre[0], centre[1], ts[2] / 2], material="mat_tray")
    lc = layout.get("config", {})
    oversize = float(lc.get("socket_oversize_mm", DEFAULT_LAYOUT_CONFIG["socket_oversize_mm"])) * MM
    wall = float(lc.get("socket_wall_mm", DEFAULT_LAYOUT_CONFIG["socket_wall_mm"])) * MM
    if layout.get("sockets", True):
        for name, lp in layout["parts"].items():
            s = lp["socket"]
            hx, hy = s["half_xy"][0] + oversize, s["half_xy"][1] + oversize
            cx, cy = s["centre_xy"]
            yaw = s["yaw"]
            sock_h = float(s.get("height_m", 0.008))
            R = np.array([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]])
            q = quat_from_rotmat(rotz(yaw))
            for k, (dx, dy, sx, sy) in enumerate([(hx + wall / 2, 0, wall / 2, hy + wall), (-(hx + wall / 2), 0, wall / 2, hy + wall),
                                                  (0, hy + wall / 2, hx + wall, wall / 2), (0, -(hy + wall / 2), hx + wall, wall / 2)]):
                c = np.array([cx, cy]) + R @ np.array([dx, dy])
                w.add_geom(name="socket_%s_%d" % (name, k), type=mujoco.mjtGeom.mjGEOM_BOX, size=[sx, sy, sock_h / 2], pos=[c[0], c[1], ts[2] + sock_h / 2], quat=q, material="mat_socket")

    for cname, cam in cfg["cameras"].items():
        w.add_camera(name=cname, pos=cam["pos"], quat=_lookat_quat(cam["pos"], cam["lookat"], cam.get("up", (0, 0, 1))), fovy=cam["fovy"])

    # Panda (model collection), tcp site, F/T sensor, wrist camera, pads, own gripper actuator
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
    pad_geoms = []
    for g in spec.geoms:
        b = g.parent
        if b is not None and b.name in ("panda/left_finger", "panda/right_finger") and g.type == mujoco.mjtGeom.mjGEOM_BOX:
            side = "left" if "left" in b.name else "right"
            g.name = "panda/pad_%s_%d" % (side, len([p for p in pad_geoms if side in p]))
            g.friction = cfg["pad_friction"]
            pad_geoms.append(g.name)
    gr = cfg["gripper"]
    for a in spec.actuators:
        if a.name == "panda/actuator8":
            a.name = "panda/gripper"
            a.gainprm = [gr["kp"], 0, 0, 0, 0, 0, 0, 0, 0, 0]
            a.biasprm = [0, -gr["kp"], -gr["kv"], 0, 0, 0, 0, 0, 0, 0]
            a.ctrlrange = [0.0, gr["max_width_m"] / 2]
            a.forcerange = [-gr["max_force_N"], gr["max_force_N"]]

    # loose bodies: servos and compound roots (with their group members as joint-less children)
    for name in loose:
        lp = layout["parts"][name]
        body = w.add_body(name=name, pos=lp["pos"], quat=lp["quat"])
        body.add_freejoint(name="free_" + name)
        if parts[name]["kind"] == "servo":
            add_servo(body, name, loose_servo=True)
        else:
            add_printed(body, name)
            for gname, g in gp.items():
                if g["root"] == name and not g["fixed"]:
                    add_group_members(body, gname)

    # welds: (carrier, servo) for scoped seats x loose servos of the SKU; (horn_<servo>, link) for scoped horn mates
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
                    eq = spec.add_equality(name="weld_%s__%s" % key, type=mujoco.mjtEq.mjEQ_WELD, objtype=mujoco.mjtObj.mjOBJ_BODY, name1=carrier, name2=s, active=False)
                    eq.solref = list(wsr)                    # (a keyword argument to add_equality is not applied: set the attributes)
                    eq.solimp = list(cfg["weld_solimp"])
                    welds[key] = eq.name
        elif m["kind"] == "horn_mate":
            link = m["part_b"]
            for s in m.get("servos", [m["part_a"]]):
                for c in [c for c in servo_names if catalogue[c] == parts[s]["catalogue_id"]]:
                    key = ("horn_" + c, link)
                    if key not in welds:
                        eq = spec.add_equality(name="weld_%s__%s" % key, type=mujoco.mjtEq.mjEQ_WELD, objtype=mujoco.mjtObj.mjOBJ_BODY, name1="horn_" + c, name2=link, active=False)
                        eq.solref = list(wsr)
                        eq.solimp = list(cfg["weld_solimp"])
                        welds[key] = eq.name

    model = spec.compile()
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

    meta = {
        "cell_version": CELL_VERSION, "config": cfg, "scope_ids": scope_ids, "root": root, "loose": loose, "servos": servo_names,
        "layout": layout, "mesh_files": mesh_files, "welds": welds, "servo_of_seat": servo_of_seat, "pad_geoms": pad_geoms,
        "mates": _mates_metric(mates, parts), "parts": parts, "graph": graph, "manifest": manifest, "panda_home": list(PANDA_HOME),
        "preassembly": preassembly, "group_poses": {g: {"root": v["root"], "fixed": v["fixed"], "q_full": v["q_full"], "parts": list(v["parts"])} for g, v in gp.items()},
        "preassembled": preassembled, "design": design,
    }
    return CellBuild(spec, model, meta)


def _mates_metric(mates, parts):
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


def collision_model_gate(a2_dir, design, config=None, scope_ids=None, preassembly=None, max_penetration_mm=0.1, sweep_mm=None, collision_cache=None):
    """Build-time gate: every scoped servo at its designed seat pose (and at lateral offsets of +-sweep_mm = side shrink + carve
    margin - tolerance) must not penetrate its carved carrier; every horn-side link at its designed pose must not penetrate the
    seated servo. Returns (ok, {mate: {max_penetration_mm, offsets_mm}})."""
    import mujoco
    cfg = _cfg(config)
    if sweep_mm is None:
        sweep_mm = float(cfg["shrink_mm"]["long_sides"]) + float(cfg["carve_margin_mm"]) - max_penetration_mm
    manifest, graph = load_givens(a2_dir)
    parts = {p["name"]: p for p in manifest["parts"]}
    scope_ids = list(scope_ids or graph["scopes"]["elbow"])
    mates = [m for m in graph["mates"] if m["id"] in set(scope_ids)]
    cache = collision_cache if collision_cache is not None else {}
    results, ok = {}, True

    def pieces_of(name):
        if name not in cache:
            cache[name] = carved_collision(a2_dir, parts[name], parts, graph, cfg)
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
                    for bname, half, centre in servo_boxes(p, cfg["shrink_mm"]):
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
                T = (F_a @ inv(F_b)).copy()
                T[:3, 3] += F_a[:3, :3] @ np.asarray(off, float)
                add_body(m["part_b"], T, True)
            elif m["kind"] == "horn_mate":
                add_body(m["part_b"], np.eye(4), False)
                T = (F_b @ inv(F_a)).copy()
                T[:3, 3] += np.asarray(off, float)
                add_body(m["part_a"], T, True)
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
        offsets = [(0, 0, 0), (d, 0, 0), (-d, 0, 0), (0, d, 0), (0, -d, 0), (0, 0, d), (0, 0, -d)] if m["kind"] != "part_seat" else [(0, 0, 0)]
        worst = probe(m, offsets)
        if worst is None:
            continue
        results[m["id"]] = {"max_penetration_mm": worst / MM, "offsets_mm": len(offsets)}
        if worst / MM > max_penetration_mm:
            ok = False
    return ok, results


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
            continue
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
