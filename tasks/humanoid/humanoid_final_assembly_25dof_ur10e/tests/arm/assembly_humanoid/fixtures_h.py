"""The cell's fixtures at run time: the fixture design of record (gt/fixtures/fixtures.json,
the hidden kit's) + one layout draw -> the data contract's FIXTURE records, the base's placement on
its stand and every module's rest placement in its fixture.

Two more fixture records, the cell furniture that collides: "camera_portal" (posts, braces, beam:
fixed) and "top_camera" (the overhead camera's housing; it follows the camera's PHYSICAL mount draw: pass the drawn and
the nominal world_T_cam of the overhead camera as layout["top_camera_T"] = (T_drawn, T_nominal) and the lens, body and
fins move with it while the hanger is rebuilt from the drawn body top up to the beam, top_camera_T / camera_hanger_vertices);
both are role "other", static, colliding with the robot, the modules, the props and the base; contact with them counts
under the safety rules (the episode builder wires the monitor). Fixture collision geoms of a named contact carry "rest": true (the fixture
side of a resting pair gets the rest impedance on both geoms); the module side comes from the collision model's
per-piece 'rest' flags (kit_adapter). The leg rack's draw is 12.5 mm (the frozen numbers' leg-rack amendment, the configuration's
"fixtures" block).

New file of this task. The design itself (parts, materials, contact faces placed from the true surfaces) is in
fixtures.json; this file only places it. Layout draws (the frozen numbers' layout_jitter_est and
leg_rack.layout_jitter; the amplitudes in cell_config.json "fixtures"): independent uniform draws per axis, in this
order: stand (x, y mm; yaw deg about the waist axis; the base moves with it), the base's clamp repeat on the stand
(along the cradle = CAD y, mm; yaw deg about the vertical through the rear pad's contact), the leg rack (both stations
and their base plate together: x, y, yaw about the stations' mid point), arm_rack_R, arm_rack_L (x, y, yaw about the
rack's footprint centre), head_nest (x, y, yaw about its footprint centre). No play of a module in its fixture is drawn:
each module rests in its fixture's design pose at its delivered joint state (the frozen numbers), the
legs with the foot on the plate where the station was designed around it (the ankle angles of the delivered state
tilt the leg above the foot). The rack table and the floor board are not drawn.

API: load_design(path=None, cfg=None); nominal_layout(); draw_layout(rng, cfg); fixture_T(design, fid, layout) (4 x 4,
metres; the stand's in the base's CAD frame, the others in the world); fixture_records(design, layout, mesh_dir=None,
visual=True, only=None); base_placement(design, layout); module_rest_placements(design, kit, frozen, layout,
units=None); REST_BODIES.
"""
import copy
import hashlib
import json
import math
import os

import numpy as np

__all__ = ["DESIGN_PATH", "MESH_DIR", "load_design", "nominal_layout", "draw_layout", "fixture_T", "fixture_records",
           "base_placement", "module_rest_placements", "REST_BODIES", "FixtureError", "FIXTURE_ORDER", "FURNITURE",
           "top_camera_T", "camera_hanger_vertices"]

HERE = os.path.dirname(os.path.abspath(__file__))
DESIGN_PATH = os.path.normpath(os.path.join(HERE, "..", "..", "..", "fixtures", "fixtures.json"))
MESH_DIR = os.environ.get("HUMANOID_FIXTURE_MESHES", "<bound at verify start by tests/cell_roots.py>")
FIXTURE_ORDER = ("stand", "leg_station_R", "leg_station_L", "leg_bench", "arm_rack_R", "arm_rack_L", "head_nest",
                 "rack_table", "floor_board", "camera_portal", "top_camera")
FURNITURE = ("camera_portal", "top_camera")      # static collision bodies of the cell furniture
# the module bodies that rest on a fixture face (the whole-body reading of the rest rule; with a collision model
# kit_adapter flags the collision model's 'rest' pieces instead and accepts this table only for a source without a collision model)
REST_BODIES = {"leg_R": ("right_ankle_roll_link", "right_ankle_pitch_link", "right_knee_link"),
               "leg_L": ("left_ankle_roll_link", "left_ankle_pitch_link", "left_knee_link"),
               "arm_R": ("right_arm_root", "right_shoulder_yaw_link", "right_elbow_link"),
               "arm_L": ("left_arm_root", "left_shoulder_yaw_link", "left_elbow_link"),
               "head": ("head_root",)}
SOLE_CENTRE_KEY = "sole_centre_cad_mm"


class FixtureError(ValueError):
    pass


def _rz(deg):
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    T = np.eye(4)
    T[:2, :2] = [[c, -s], [s, c]]
    return T


def _tr(v):
    T = np.eye(4)
    T[:3, 3] = v
    return T


def _quat(R):
    R = np.asarray(R, float)
    t = np.trace(R)
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        q = [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        q = [(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s]
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        q = [(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s]
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        q = [(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s]
    q = np.array(q) / np.linalg.norm(q)
    return (-q if q[0] < 0 else q).tolist()


def _placement(T, frame):
    return {"frame": frame, "pos": [float(v) for v in T[:3, 3]], "quat": [float(v) for v in _quat(T[:3, :3])]}


def load_design(path=None, cfg=None):
    p = path or DESIGN_PATH
    with open(p, "rb") as f:
        raw = f.read()
    d = json.loads(raw.decode("ascii"))
    if d.get("schema") != "humanoid-asm03-fixtures/1":
        raise FixtureError("%s: schema %r" % (p, d.get("schema")))
    if cfg is not None and "fixtures" in cfg and cfg["fixtures"].get("design_sha256"):
        h = hashlib.sha256(raw).hexdigest()
        if h != cfg["fixtures"]["design_sha256"]:
            raise FixtureError("%s: sha256 %s, the cell configuration pins %s" % (p, h, cfg["fixtures"]["design_sha256"]))
    d["_by_id"] = {f["id"]: f for f in d["fixtures"]}
    return d


def nominal_layout():
    z = (0.0, 0.0, 0.0)
    return {"stand": z, "clamp": (0.0, 0.0), "leg_racks": z, "arm_rack_R": z, "arm_rack_L": z, "head_nest": z}


def draw_layout(rng, cfg):
    """one layout draw (mm, mm, deg) from a numpy Generator (OS entropy; the fixture proofs: default_rng(seed)); the
    amplitudes from cell_config.json "fixtures.layout_jitter"; the order of the draws is fixed"""
    J = cfg["fixtures"]["layout_jitter"]
    u = lambda a: float(rng.uniform(-float(a), float(a)))  # noqa: E731
    out = {}
    out["stand"] = (u(J["stand_mm"]), u(J["stand_mm"]), u(J["stand_yaw_deg"]))
    out["clamp"] = (u(J["clamp_repeat_mm"]), u(J["clamp_repeat_deg"]))
    out["leg_racks"] = (u(J["leg_rack_mm"]), u(J["leg_rack_mm"]), u(J["leg_rack_yaw_deg"]))
    for k in ("arm_rack_R", "arm_rack_L"):
        out[k] = (u(J["arm_racks_mm"]), u(J["arm_racks_mm"]), u(J["arm_racks_yaw_deg"]))
    out["head_nest"] = (u(J["head_nest_mm"]), u(J["head_nest_mm"]), u(J["head_nest_yaw_deg"]))
    return out


def _about(piv_m, dx_m, dy_m, yaw_deg):
    return _tr([dx_m, dy_m, 0.0]) @ _tr([piv_m[0], piv_m[1], 0.0]) @ _rz(yaw_deg) @ _tr([-piv_m[0], -piv_m[1], 0.0])


def stand_T_cad(design, layout):
    """the stand's pose in the station's base CAD frame (metres): the draw about the waist axis"""
    fx = design["_by_id"]["stand"]
    piv = np.array(fx["draw_pivot_body_mm"], float) * 1e-3
    dx, dy, dyaw = layout["stand"]
    return _about(piv, dx * 1e-3, dy * 1e-3, dyaw)


def clamp_T(design, layout):
    """the base's pose on its stand (CAD frame, metres): along the cradle (y) and yaw about the vertical through the
    rear pad's contact on the rear wall"""
    cy, cyaw = layout["clamp"]
    rx = float(design["notes"]["pelvis_rear_wall_min_x_mm"]) * 1e-3
    return _tr([0.0, cy * 1e-3, 0.0]) @ _tr([rx, 0.0, 0.0]) @ _rz(cyaw) @ _tr([-rx, 0.0, 0.0])


def fixture_T(design, fid, layout):
    fx = design["_by_id"][fid]
    T0 = np.array(fx["T_nominal_world_m"], float)
    if fid == "stand":
        return stand_T_cad(design, layout)                 # placement frame 'cad'
    if fid in ("leg_station_R", "leg_station_L", "leg_bench"):
        mid = np.array(design["_by_id"]["leg_bench"]["T_nominal_world_m"], float)[:3, 3]
        dx, dy, dyaw = layout["leg_racks"]
        return _about(mid, dx * 1e-3, dy * 1e-3, dyaw) @ T0
    if fid in ("arm_rack_R", "arm_rack_L", "head_nest"):
        piv = np.array(fx["draw_pivot_body_mm"], float) * 1e-3
        dx, dy, dyaw = layout[fid]
        return _tr([dx * 1e-3, dy * 1e-3, 0.0]) @ T0 @ _about(piv, 0.0, 0.0, dyaw)
    if fid == "top_camera":
        return top_camera_T(design, layout)
    return T0


def top_camera_T(design, layout):
    """the overhead camera housing's pose (world, metres): the nominal lens-front frame moved by the drawn mount
    (layout["top_camera_T"] = (world_T_cam drawn, world_T_cam nominal), any camera convention, the same for both);
    without it the nominal (the mount of record)"""
    T0 = np.array(design["_by_id"]["top_camera"]["T_nominal_world_m"], float)
    tc = layout.get("top_camera_T") if isinstance(layout, dict) else None
    if tc is None:
        return T0
    Td, Tn = np.array(tc[0], float), np.array(tc[1], float)
    return Td @ np.linalg.inv(Tn) @ T0


def camera_hanger_vertices(design, T_cam):
    """the hanger's convex piece (8 vertices, metres, in the housing's body frame): the drawn body's top face (70 x 120
    mm about the lens axis, 122 mm above the lens front) up to the same rectangle, unturned, on the beam's underside"""
    info = design["_by_id"]["top_camera"]["info"]
    hx, hy = 0.5e-3 * info["hanger_mm"][0], 0.5e-3 * info["hanger_mm"][1]
    zt = 1e-3 * info["body_top_above_lens_mm"]
    zb = float(info["beam_bottom_cell_z_m"])
    T0 = np.array(design["_by_id"]["top_camera"]["T_nominal_world_m"], float)
    bot = np.array([[sx * hx, sy * hy, zt] for sx in (-1, 1) for sy in (-1, 1)])
    c = T_cam[:3, 3]
    top_w = np.array([[c[0] + sx * hx, c[1] + sy * hy, zb] for sx in (-1, 1) for sy in (-1, 1)])
    Ti = np.linalg.inv(T_cam)
    top = top_w @ Ti[:3, :3].T + Ti[:3, 3]
    if np.min((bot @ T_cam[:3, :3].T + T_cam[:3, 3])[:, 2]) > zb - 1e-4:
        raise FixtureError("the drawn overhead camera reaches the portal's beam (hanger thinner than 0.1 mm)")
    return np.vstack([bot, top])


def clamp_tip_shift_m(design, layout):
    """{side: dx (metres, CAD x)}: the two toggle clamps' spring-loaded spindles keep their PU tips on the pelvis
    shell's front wall when the clamp repeat turns / slides the base on its stand: the tip moves along the spindle
    (CAD x) by the front wall's displacement at the tip"""
    Tc = clamp_T(design, layout)
    out = {}
    for side, p in sorted(design["_by_id"]["stand"]["info"]["clamp_tip_contact_mm"].items()):
        q = np.array([p[0] * 1e-3, p[1] * 1e-3, p[2] * 1e-3, 1.0])
        out[side] = float((Tc @ q)[0] - q[0])
    return out


def fixture_records(design, layout, mesh_dir=None, visual=True, only=None):
    """the data contract's FIXTURE records of the drawn layout (the stand in frame 'cad', the others 'world')"""
    md = mesh_dir or MESH_DIR
    out = []
    for fid in FIXTURE_ORDER:
        if only is not None and fid not in only:
            continue
        fx = design["_by_id"][fid]
        T = fixture_T(design, fid, layout)
        geoms = []
        tip = clamp_tip_shift_m(design, layout) if fid == "stand" else {}
        for g in fx["geoms"]:
            if g["role"] == "visual":
                if not visual:
                    continue
                g = dict(g)
                g["mesh"] = {"file": os.path.join(md, g["mesh"]["file"])}
            g = copy.deepcopy(g)
            if fid == "top_camera" and g.get("part") == "hanger":
                if g["role"] == "visual":
                    if isinstance(layout, dict) and layout.get("top_camera_T") is not None:
                        continue                   # a drawn mount: the nominal hanger mesh would not fit (rebuilt below)
                    geoms.append(g)
                    continue
                g = {"name": g["name"], "role": "collision", "type": "mesh", "part": "hanger",
                     "mesh": {"vertices": camera_hanger_vertices(design, T).tolist()}}
            for side, dx in tip.items():
                if g.get("part") in ("clamp_tip_" + side, "clamp_plunger_" + side):
                    p = list(g.get("pos", [0.0, 0.0, 0.0]))
                    g["pos"] = [p[0] + dx, p[1], p[2]]
            geoms.append(g)
        rec = {"id": fid, "role": fx["role"], "placement": _placement(T, "cad" if fid == "stand" else "world"),
               "geoms": geoms, "rest_surface": bool(fx["rest_surface"]),
               "collide_with_base": bool(fx["collide_with_base"])}
        out.append(rec)
    return out


def base_placement(design, layout):
    """the base's placement (frame 'cad'): the stand's draw, then the clamp repeat on the stand"""
    return _placement(stand_T_cad(design, layout) @ clamp_T(design, layout), "cad")


def _fk_body_m(kit, unit, body, q):
    """4 x 4 (metres) of a kit body in its unit's CAD frame at joint state q (kit convention: (q - ref) about the
    hinge through anchor)"""
    chain = []
    b = body
    while b is not None:
        chain.append(b)
        b = kit["bodies"][b]["parent"]
    T = np.eye(4)
    for b in reversed(chain):
        B = kit["bodies"][b]
        if B["joint"]:
            J = kit["joints"][B["joint"]]
            a = np.asarray(J["axis"], float)
            a = a / np.linalg.norm(a)
            ang = float(q.get(B["joint"], J["ref_rad_cad_pose"])) - float(J["ref_rad_cad_pose"])
            x, y, z = a
            c, s = math.cos(ang), math.sin(ang)
            C = 1.0 - c
            R = np.array([[c + x * x * C, x * y * C - z * s, x * z * C + y * s],
                          [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
                          [z * x * C - y * s, z * y * C + x * s, c + z * z * C]])
            p = np.asarray(J["anchor_mm"], float) * 1e-3
            H = np.eye(4)
            H[:3, :3] = R
            H[:3, 3] = p - R @ p
            T = T @ H
    return T


def module_rest_placements(design, kit, frozen, layout, units=None):
    """{unit: placement (frame 'world')} of the modules resting in their fixtures at their delivered joint states"""
    from . import kit_adapter as KA
    dq = KA.delivered_q(kit, frozen)
    out = {}
    for u in (units or ("leg_R", "leg_L", "arm_R", "arm_L", "head")):
        if u.startswith("leg_"):
            side = u[-1]
            fid = "leg_station_" + side
            Ts = fixture_T(design, fid, layout)
            sc = np.array(design["_by_id"][fid]["info"][SOLE_CENTRE_KEY], float) * 1e-3
            foot = ("right" if side == "R" else "left") + "_ankle_roll_link"
            ref = {j: kit["joints"][j]["ref_rad_cad_pose"] for j in kit["units"][u]["hinges"]}
            Tf_design = _fk_body_m(kit, u, foot, ref)
            Tf_q = _fk_body_m(kit, u, foot, dq)
            T = Ts @ _tr(-sc) @ Tf_design @ np.linalg.inv(Tf_q)
        elif u.startswith("arm_"):
            T = fixture_T(design, "arm_rack_" + u[-1], layout)
        elif u == "head":
            T = fixture_T(design, "head_nest", layout)
        else:
            raise FixtureError("unit %r" % u)
        out[u] = _placement(T, "world")
    return out
