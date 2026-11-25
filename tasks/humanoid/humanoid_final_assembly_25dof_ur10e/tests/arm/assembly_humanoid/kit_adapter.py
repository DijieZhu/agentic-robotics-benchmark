"""The kit-to-cell adapter: the hidden kit
(schema humanoid-asm-kit/1, kit.json + the drawn STL meshes; default the kit of record) and the collision model
(schema humanoid-asm03-collision/1; default the collision model of record) into the cell's data contract (humanoid-cell-data/1,
scene.py's module text) for scene.build_cell.

THE ONE PLACE TO SWITCH THE COLLISION MODEL is COLLISION_SOURCE_OF_RECORD below (the collision model of
record; an earlier one before it); the environment variable HUMANOID_COLLISION_PIECES or collision_source= override it.
From the collision model the adapter reads: per piece 'exact', 'rest' and 'rest_contact' (a rest piece gets the
rest-contact impedance; the fixture side of the pair is flagged by fixtures_h, so both geoms of a resting pair carry it);
model-wide 'exclude_pairs' (into each assembly's exclude_pairs), 'joint_limits_deg' (the cell ranges of the four
hip joints: they replace the kit's provisional cell_range_rad and must equal the values of record HIP_LIMITS_OF_RECORD_DEG,
asserted) and 'delivered_state_deg' (the state the model was carved at: asserted equal to the adapter's own
delivered state within DELIVERED_TOL_DEG, 0.01 deg per joint: the model of record was carved at an earlier delivered state with
its amendment; the settled leg angles of record differ from it by at most 0.0035 deg). The joint model (friction,
damping, armature) still comes from the configuration only.
The delivered states come from the frozen numbers (the delivered states of their earlier versions
unchanged: the knees at 0).

New file of this task. What it does, and the three rules it keeps:
  * JOINT FRICTION, DAMPING, ARMATURE AND THE FRICTION IMPEDANCE COME FROM THE CELL CONFIGURATION ONLY. Every kit
    joint names its actuator; the adapter maps it onto the configuration's joint model (EC-A4310 / 4315 / 6416 / 3814
    / 5013, the parallel ankle's 'ankle_pitch' / 'ankle_roll') and ASSERTS that the kit's own copies (frictionloss_Nm,
    damping_Nms_per_rad, armature_kgm2, solref_friction, solimp_friction) equal the configuration's (1e-9); it never
    copies a kit number into the cell (check_joint_model).
  * THE MODULES' DELIVERED JOINT STATES COME FROM the frozen numbers (arm_delivered_deg, leg_delivered_deg,
    head_delivered_deg), not from the kit's delivered_state; the base's waist at 0 and its three output rotors at the
    kit's start clock (start_q_rad) when the kit carries it, else 0.
  * COLLISION GEOMETRY COMES THROUGH ONE FUNCTION, collision_geoms(unit, body). Without a collision model
    it returns the FALLBACK pieces (precomputed per kit body into a pieces folder:
    CoACD pieces, else convex hull pieces, mirrored for 200_06C /
    600_26B; the hip-pitch actuators 200_12X / 11X as exact stator prisms, the hip-pitch output rotors as an
    exact rotor boss, each leg root with an exact bracket and without the CoACD pieces crossing the rotor's exact
    volume; fasteners, inserts and the six duplicate screws carry no piece; the neck-yaw rotor none). A collision model
    replaces the source (PIECES_SOURCE / collision_source=) and nothing else.

Frames and names: a kit body frame is the CAD frame carried by the body (identity poses at the CAD pose), so every
BODY record is given at the module's DELIVERED state (its pose = the kit's forward kinematics there, ref = the
delivered angle: the CAD stored pose has the arms' elbow and wrist outside their MJCF ranges), its hinge at the kit's
anchor (metres) along the kit's axis (both constant in the body's own frame); body and joint names are the kit's (the mating graph and the coupling use them). Hip yaw ranges: the
kit's cell_range_rad (the CAD free range minus 1 deg, provisional without a collision model); every other range the authors'
MJCF range_rad. Inertials: the kit's mass, com_m and fullinertia_kgm2 (body frame); the three output rotors carry no
mass of the mass table (their actuator is booked on the stator) and get mass 1e-6 kg / inertia 1e-10 kg m2, the
bounds the kit's own loader uses (their hinge armature dominates).

API: load_kit(kit_dir) -> kit; check_kit(kit); check_joint_model(kit, cfg) -> rows; actuator_key(kit, joint);
delivered_q(kit, frozen) -> {joint: rad}; base_datum(kit); collision_geoms(unit, body, source=None) -> (geoms,
arrays); assembly(kit, unit, cfg, frozen, placement, ...) -> (ASSEMBLY, arrays); cell_data(kit_dir, cfg, frozen,
placements, fixtures=(), ...) -> (data, arrays) for scene.build_cell(data, cfg, arrays=arrays).
"""
import json
import math
import os

import numpy as np

__all__ = ["KIT_SCHEMA", "KIT_DIR", "FROZEN_PATH", "FROZEN_AMEND_PATH", "PIECES_SOURCE", "COLLISION_SOURCE_OF_RECORD",
           "HIP_LIMITS_OF_RECORD_DEG", "AdapterError", "load_kit", "check_kit", "check_joint_model", "actuator_key",
           "delivered_q", "base_datum", "collision_geoms", "collision_model", "check_collision_model", "joint_range",
           "assembly", "cell_data", "UNITS"]

KIT_SCHEMA = "humanoid-asm-kit/1"
KIT_DIR = os.environ.get("HUMANOID_KIT_DIR", "<bound at verify start by tests/cell_roots.py>")   # the kit of record
HERE = os.path.dirname(os.path.abspath(__file__))
FROZEN_PATH = os.path.normpath(os.path.join(HERE, "..", "..", "..", "frozen_numbers.json"))     # the frozen numbers, version 1.5
FROZEN_AMEND_PATH = os.path.normpath(os.path.join(HERE, "..", "..", "..", "frozen_numbers_amendment.json"))
# >>> THE ONE PLACE TO SWITCH THE COLLISION MODEL (the collision model of record; an earlier one before it) <<<
COLLISION_SOURCE_OF_RECORD = "<bound at verify start by tests/cell_roots.py>"
PIECES_SOURCE = os.environ.get("HUMANOID_COLLISION_PIECES", COLLISION_SOURCE_OF_RECORD)
COLLISION_SCHEMA = "humanoid-asm03-collision/1"
# the cell's hip limits of record: the collision model's
# joint_limits_deg must equal them
HIP_LIMITS_OF_RECORD_DEG = {"right_hip_roll_joint": [-20.5, 7.5], "left_hip_roll_joint": [-7.5, 20.5],
                            "right_hip_yaw_joint": [-38.0, 43.5], "left_hip_yaw_joint": [-43.5, 38.0]}
DELIVERED_TOL_DEG = 0.01        # the model of record's carve state vs the settled leg angles (<= 0.0035 deg)
UNITS = ("base", "leg_R", "leg_L", "arm_R", "arm_L", "head")
ROTOR_MASS_KG = 1e-6
ROTOR_INERTIA_KGM2 = 1e-10
TOL = 1e-9


class AdapterError(ValueError):
    pass


# ---------------------------------------------------------------------------------------------------------- kit
def load_kit(kit_dir=None):
    kd = kit_dir or KIT_DIR
    with open(os.path.join(kd, "kit.json")) as f:
        kit = json.load(f)
    kit["_dir"] = kd
    check_kit(kit)
    return kit


def check_kit(kit):
    if kit.get("schema") != KIT_SCHEMA:
        raise AdapterError("kit schema %r, the adapter reads %s" % (kit.get("schema"), KIT_SCHEMA))
    if sorted(kit["units"]) != sorted(UNITS):
        raise AdapterError("kit units %r" % sorted(kit["units"]))
    n_h = 0
    for u in UNITS:
        U = kit["units"][u]
        for b in U["bodies"]:
            B = kit["bodies"][b]
            if B["unit"] != u:
                raise AdapterError("body %s: unit %r, listed under %s" % (b, B["unit"], u))
            if B["joint"]:
                J = kit["joints"][B["joint"]]
                if J["type"] != "hinge" or J["child_body"] != b or J["parent_body"] != B["parent"]:
                    raise AdapterError("joint %s does not join %s to its parent" % (B["joint"], b))
                n_h += 1
        if (U["role"] == "carrier") != (u == "base"):
            raise AdapterError("unit %s: role %r" % (u, U["role"]))
    if n_h != 25:
        raise AdapterError("the kit has %d hinges; the robot has 25" % n_h)
    return True


def actuator_key(kit, jname):
    """the configuration's joint-model key of a kit joint: 'EC-A4310' .. 'EC-A5013' from the vendor model name, or
    'ankle_pitch' / 'ankle_roll' for the parallel ankle pair ('ankle_pair:EC-A4310-P2-36' in the kit)"""
    J = kit["joints"][jname]
    a = J["actuator_model"]
    if a.startswith("ankle_pair:"):
        for k in ("ankle_pitch", "ankle_roll"):
            if jname.endswith(k + "_joint"):
                return k
        raise AdapterError("joint %s: an ankle pair actuator on a joint that is not an ankle pitch / roll" % jname)
    return "-".join(a.split("-")[:2])


def _cfg_joint(cfg, key):
    from . import config as CF
    fl, dmp, arm = CF.joint_params(cfg, key)
    jf = cfg["physics"]["joint_friction_impedance"]
    return fl, dmp, arm, [float(v) for v in jf["solref_friction"]], [float(v) for v in jf["solimp_friction"]]


def check_joint_model(kit, cfg, tol=TOL):
    """assert, joint by joint, that the kit's copies of the joint model equal the configuration's (the configuration
    is the one source); returns the table; raises AdapterError on the first difference"""
    rows = []
    for jn in sorted(kit["joints"]):
        J = kit["joints"][jn]
        key = actuator_key(kit, jn)
        fl, dmp, arm, sr, si = _cfg_joint(cfg, key)
        kit_v = [float(J["frictionloss_Nm"]), float(J["damping_Nms_per_rad"]), float(J["armature_kgm2"])]
        bad = []
        for nm, a, b in (("frictionloss", kit_v[0], fl), ("damping", kit_v[1], dmp), ("armature", kit_v[2], arm)):
            if abs(a - b) > tol:
                bad.append((nm, a, b))
        for nm, a, b in (("solref_friction", J["solref_friction"], sr), ("solimp_friction", J["solimp_friction"], si)):
            if len(a) != len(b) or any(abs(float(x) - float(y)) > tol for x, y in zip(a, b)):
                bad.append((nm, a, b))
        rows.append({"joint": jn, "actuator": key, "config": [fl, dmp, arm], "kit": kit_v, "equal": not bad})
        if bad:
            raise AdapterError("joint %s (%s): the kit's %s differ from the cell configuration's" % (jn, key, bad))
    return rows


def load_frozen(path=None, amend_path=None, amend=True):
    """the frozen numbers (schema 1.5 asserted): it carries the earlier amendment merged (the two knees at
    their limit 0), so no amendment step is applied; amend_path and amend are kept for the callers and not
    read (FROZEN_AMEND_PATH stays: tests read the amendment file itself)"""
    with open(path or FROZEN_PATH) as f:
        fr = json.load(f)
    if fr.get("schema") != "humanoid-asm03-frozen-numbers/1.5":
        raise AdapterError("frozen numbers schema %r" % fr.get("schema"))
    return fr


LIMIT_OVERSHOOT_RAD = math.radians(0.1)
CLAMPED = {}


def delivered_q(kit, frozen):
    """{joint: rad} of every kit hinge: arms / legs / head from the frozen numbers; the base's waist and output
    rotors 0 (encoder zero). A frozen angle that lies beyond its joint's range by at most 0.1 deg is a joint resting ON
    its limit measured with MuJoCo's soft limit compressed by the preload (both knees 'on the straight-leg
    limit (q = 0)', recorded as +0.0295 / -0.0317 deg): it is delivered AT the limit (the cell's builder refuses a
    delivered state outside the range) and listed in CLAMPED; anything farther out is an error."""
    out = {}
    for u in UNITS:
        U = kit["units"][u]
        side = {"R": "right", "L": "left"}.get(u[-1])
        for jn in U["hinges"]:
            if u == "base":
                # the waist at MJCF 0; the three output rotors at their start clock when the kit carries one (the
                # kit's joints[j].start_q_rad), else at the encoder zero
                out[jn] = float(kit["joints"][jn].get("start_q_rad", 0.0))
            elif u == "head":
                out[jn] = math.radians(float(frozen["head_delivered_deg"][jn[:-len("_joint")]]))
            else:
                key = jn[len(side) + 1:-len("_joint")]
                blk = frozen["arm_delivered_deg" if u.startswith("arm_") else "leg_delivered_deg"][side]
                if key not in blk:
                    raise AdapterError("the frozen numbers have no delivered angle for %s" % jn)
                out[jn] = math.radians(float(blk[key]))
    for jn in list(out):
        lo, hi = joint_range(kit, jn, None)
        q = out[jn]
        if q < lo or q > hi:
            if min(abs(q - lo), abs(q - hi)) > LIMIT_OVERSHOOT_RAD:
                raise AdapterError("delivered %s = %.6f rad lies %.4f deg outside its range" % (
                    jn, q, math.degrees(min(abs(q - lo), abs(q - hi)))))
            out[jn] = min(max(q, lo), hi)
            CLAMPED[jn] = {"frozen_rad": q, "delivered_rad": out[jn]}
    return out


def joint_range(kit, jname, model=None):
    """the joint's cell range (rad): the collision model's joint_limits_deg where it names the joint (the four hip
    joints); else the kit's cell_range_rad (provisional, the fallback pieces only) or the MJCF range_rad"""
    if model is None and kit.get("_collision_model") is not None:
        model = kit["_collision_model"]
    if model is not None and jname in model.get("joint_limits_deg", {}):
        lo, hi = model["joint_limits_deg"][jname]
        return [math.radians(float(lo)), math.radians(float(hi))]
    J = kit["joints"][jname]
    r = J.get("cell_range_rad") or J["range_rad"]
    return [float(r[0]), float(r[1])]


def base_datum(kit):
    """the base's datum for scene.station_transform: the waist-yaw axis point and the sole plane (CAD, metres)"""
    a = kit["joints"]["waist_yaw_joint"]["anchor_mm"]
    return {"waist_axis_point_m": [float(a[0]) * 1e-3, float(a[1]) * 1e-3, float(a[2]) * 1e-3],
            "sole_plane_z_m": -1078.2693576907593e-3}


def _rot(axis, ang):
    a = np.asarray(axis, float)
    a = a / np.linalg.norm(a)
    x, y, z = a
    c, s = math.cos(ang), math.sin(ang)
    C = 1.0 - c
    return np.array([[c + x * x * C, x * y * C - z * s, x * z * C + y * s],
                     [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
                     [z * x * C - y * s, z * y * C + x * s, c + z * z * C]])


def unit_fk_m(kit, unit, q):
    """{body: 4 x 4 (metres)} of the unit's bodies in its CAD frame at joint state q (the root at identity; a child
    turns by (q - ref_rad_cad_pose) about its hinge through the anchor)"""
    out = {}
    for b in kit["units"][unit]["bodies"]:
        B = kit["bodies"][b]
        if B["parent"] is None:
            out[b] = np.eye(4)
            continue
        J = kit["joints"][B["joint"]]
        R = _rot(J["axis"], float(q.get(B["joint"], J["ref_rad_cad_pose"])) - float(J["ref_rad_cad_pose"]))
        p = np.asarray(J["anchor_mm"], float) * 1e-3
        H = np.eye(4)
        H[:3, :3] = R
        H[:3, 3] = p - R @ p
        out[b] = out[B["parent"]] @ H
    return out


def _pose_of(T):
    R = T[:3, :3]
    t = np.trace(R)
    if t > -0.99:
        w = math.sqrt(max(0.0, 1.0 + t)) / 2.0
        q = [w, (R[2, 1] - R[1, 2]) / (4 * w), (R[0, 2] - R[2, 0]) / (4 * w), (R[1, 0] - R[0, 1]) / (4 * w)]
    else:
        k = int(np.argmax(np.diag(R)))
        i, j = (k + 1) % 3, (k + 2) % 3
        r = math.sqrt(max(0.0, 1.0 + R[k, k] - R[i, i] - R[j, j]))
        v = [0.0, 0.0, 0.0]
        v[k] = r / 2.0
        v[i] = (R[i, k] + R[k, i]) / (2 * r)
        v[j] = (R[j, k] + R[k, j]) / (2 * r)
        q = [(R[j, i] - R[i, j]) / (2 * r)] + v
    q = np.array(q) / np.linalg.norm(q)
    if q[0] < 0:
        q = -q
    return {"pos": [float(v) for v in T[:3, 3]], "quat": [float(v) for v in q]}


# ---------------------------------------------------------------------------------------------------------- collision
_PIECES = {}
_MODELS = {}


def collision_model(source=None):
    """the collision model's collision_model.json of the source folder (None for a source without one: the fallback
    pieces), checked (schema; the hip limits of record)"""
    src = source or PIECES_SOURCE
    if src not in _MODELS:
        p = os.path.join(src, "collision_model.json")
        if not os.path.exists(p):
            _MODELS[src] = None
        else:
            with open(p) as f:
                cm = json.load(f)
            check_collision_model(cm)
            cm["_by_key"] = {r["key"]: r for r in cm["pieces"]}
            _MODELS[src] = cm
    return _MODELS[src]


def check_collision_model(cm):
    """schema, per-piece flags, exclusions and the hip limits of record; raises AdapterError"""
    if cm.get("schema") != COLLISION_SCHEMA:
        raise AdapterError("collision model schema %r, the adapter reads %s" % (cm.get("schema"), COLLISION_SCHEMA))
    for k in ("exclude_pairs", "joint_limits_deg", "delivered_state_deg", "pieces"):
        if k not in cm:
            raise AdapterError("collision model without %r" % k)
    lim = cm["joint_limits_deg"]
    if sorted(lim) != sorted(HIP_LIMITS_OF_RECORD_DEG):
        raise AdapterError("collision model limits %r, the record names %r" % (sorted(lim), sorted(HIP_LIMITS_OF_RECORD_DEG)))
    for j, v in HIP_LIMITS_OF_RECORD_DEG.items():
        if any(abs(float(a) - float(b)) > 1e-9 for a, b in zip(lim[j], v)) or len(lim[j]) != 2:
            raise AdapterError("collision model limit %s = %r, the value of record is %r" % (j, lim[j], v))
    for r in cm["pieces"]:
        if r.get("rest") and not r.get("rest_contact"):
            raise AdapterError("rest piece %s names no fixture contact" % r["key"])
        if r.get("rest") and not r.get("exact"):
            raise AdapterError("rest piece %s is not exact" % r["key"])
    for a, b in cm["exclude_pairs"]:
        if not (isinstance(a, str) and isinstance(b, str)) or a == b:
            raise AdapterError("exclude pair %r" % ([a, b],))
    return True


def check_delivered_state(kit, frozen, cm):
    """the collision model's carve state equals the adapter's delivered state (deg, DELIVERED_TOL_DEG); returns the
    rows; raises AdapterError on a difference"""
    dq = delivered_q(kit, frozen)
    rows, bad = [], []
    for j in sorted(dq):
        a = math.degrees(dq[j])
        b = cm["delivered_state_deg"].get(j)
        ok = b is not None and abs(a - float(b)) <= DELIVERED_TOL_DEG
        rows.append({"joint": j, "adapter_deg": a, "model_deg": b, "equal": ok})
        if not ok:
            bad.append((j, a, b))
    extra = sorted(set(cm["delivered_state_deg"]) - set(dq))
    if bad or extra:
        raise AdapterError("collision model carved at another state: %r, extra %r" % (bad[:5], extra))
    return rows


def _pieces(source):
    src = source or PIECES_SOURCE
    if src not in _PIECES:
        with open(os.path.join(src, "pieces_index.json")) as f:
            idx = json.load(f)
        blob = np.fromfile(os.path.join(src, "pieces.bin"), dtype="<f8")
        import hashlib
        if hashlib.sha256(blob.tobytes()).hexdigest() != idx["bin_sha256"]:
            raise AdapterError("%s: pieces.bin does not match its index" % src)
        _PIECES[src] = (idx, blob)
    return _PIECES[src]


def collision_geoms(unit, body, source=None, kit=None):
    """THE ONE SOURCE OF COLLISION GEOMETRY: (geoms, arrays) of a kit body, in its body frame (metres), from the
    collision source (default COLLISION_SOURCE_OF_RECORD). Every piece is one convex mesh geom (role collision; the
    cell's geom defaults); with a collision model, each geom carries the piece's 'exact' and 'rest' flags and, for a
    rest piece, 'rest_contact' (the fixture contact it serves, 'fixture:part'); a rest geom gets the cell's rest-contact
    impedance in the scene builder. `arrays` maps "npz:<key>" keys to vertex arrays."""
    idx, blob = _pieces(source)
    cm = collision_model(source)
    rows = idx["bodies"].get(body)
    if rows is None:
        raise AdapterError("collision source has no body %r" % body)
    geoms, arrays = [], {}
    for r in rows:
        key = "col/" + r["key"]
        arrays[key] = blob[r["offset"]:r["offset"] + 3 * r["n"]].reshape(-1, 3)
        g = {"name": "c/" + r["key"], "role": "collision", "type": "mesh",
             "mesh": {"vertices": "npz:" + key}, "source": r["source"]}
        if cm is not None:
            pc = cm["_by_key"].get(r["key"])
            if pc is None or pc["body"] != body:
                raise AdapterError("piece %s of %s is not in the collision model" % (r["key"], body))
            if bool(pc["exact"]) != bool(r.get("exact")) or bool(pc["rest"]) != bool(r.get("rest")):
                raise AdapterError("piece %s: the index and the model disagree on its flags" % r["key"])
            g["exact"] = bool(pc["exact"])
            g["rest"] = bool(pc["rest"])
            if pc["rest"]:
                g["rest_contact"] = pc["rest_contact"]
        geoms.append(g)
    return geoms, arrays


# ---------------------------------------------------------------------------------------------------------- assembly
def assembly(kit, unit, cfg, frozen, placement=None, visual=False, collision=True, collision_source=None,
             rest_bodies=(), self_collide=False, materials=None):
    """one ASSEMBLY record of the data contract (+ its arrays). placement: the data contract's placement (default
    the base / modules at the CAD placement: frame 'cad', identity). With a collision model: the pieces' own
    'rest' flags mark what rests on a fixture, the model's exclude_pairs inside this unit go into the record's
    exclude_pairs, the hip joints get the model's limits, and `rest_bodies` is not used (it is only for a source
    without a model, the fallback pieces: every collision geom of those bodies is flagged 'rest'). self_collide False
    = no contacts inside the module (the fallback pieces' reading); the collision model is built for self_collide True."""
    check_joint_model(kit, cfg)
    cm = collision_model(collision_source) if collision else None
    if cm is not None:
        check_delivered_state(kit, frozen, cm)
    U = kit["units"][unit]
    dq = delivered_q(kit, frozen)
    # the bodies are given at the DELIVERED state (ref = the delivered angle): the CAD stored pose has the arms' elbow
    # and wrist beyond their MJCF ranges, and the contract's ref must lie inside the range. A kit body
    # frame is carried rigidly by its body, so the hinge's anchor and axis in the body's own frame stay the kit's.
    fk = unit_fk_m(kit, unit, dq)
    bodies, arrays = [], {}
    for b in U["bodies"]:
        B = kit["bodies"][b]
        rec = {"name": b, "parent": B["parent"], "pose": _pose_of(fk[b]), "joint": None, "geoms": []}
        if B["joint"]:
            J = kit["joints"][B["joint"]]
            rec["joint"] = {"name": B["joint"], "type": "hinge", "axis": [float(v) for v in J["axis"]],
                            "pos": [float(v) * 1e-3 for v in J["anchor_mm"]], "range": joint_range(kit, B["joint"], cm),
                            "ref": float(dq[B["joint"]]), "actuator": actuator_key(kit, B["joint"])}
        if B["parent"] is None and unit == "base":
            rec["inertial"] = {"mass": float(B["mass_kg"]), "pos": [float(v) for v in B["com_m"]],
                               "fullinertia": [float(v) for v in B["fullinertia_kgm2"]]}
        elif float(B["mass_kg"]) > 0:
            rec["inertial"] = {"mass": float(B["mass_kg"]), "pos": [float(v) for v in B["com_m"]],
                               "fullinertia": [float(v) for v in B["fullinertia_kgm2"]]}
        else:
            rec["inertial"] = {"mass": ROTOR_MASS_KG, "pos": [0.0, 0.0, 0.0],
                               "diaginertia": [ROTOR_INERTIA_KGM2] * 3}
        if collision:
            g, a = collision_geoms(unit, b, collision_source, kit)
            if cm is None and b in rest_bodies:
                for gg in g:
                    gg["rest"] = True
            rec["geoms"] += g
            arrays.update(a)
        if visual:
            for k, vf in enumerate(B["visual"]):
                rec["geoms"].append({"name": "v/%s__%02d" % (b, k), "role": "visual", "type": "mesh",
                                     "mesh": {"file": os.path.join(kit["_dir"], vf["file"])},
                                     "rgba": [0.2, 0.2, 0.22, 1.0]})
        bodies.append(rec)
    a = {"id": unit, "kind": "base" if unit == "base" else "module", "mount": "fixed" if unit == "base" else "free",
         "placement": placement or {"frame": "cad", "pos": [0.0, 0.0, 0.0], "quat": [1.0, 0.0, 0.0, 0.0]},
         "bodies": bodies, "delivered_q": {j: dq[j] for j in U["hinges"]}, "self_collide": bool(self_collide)}
    if cm is not None:
        mine = set(U["bodies"])
        a["exclude_pairs"] = [[p, q] for p, q in cm["exclude_pairs"] if p in mine and q in mine]
    if unit == "base":
        a["datum"] = base_datum(kit)
    if materials:
        a["materials"] = materials
    return a, arrays


def cell_data(kit_dir=None, cfg=None, frozen=None, placements=None, fixtures=(), modules=None, visual=False,
              collision=True, collision_source=None, rest_bodies=None, welds=(), name="humanoid_assembly_cell",
              cameras=(), self_collide=None):
    """(data, arrays) for scene.build_cell: the base at the station, the named modules (default all five) at their
    placements ({unit: placement}; default the CAD / design placement), the fixtures' FIXTURE records (fixtures_h),
    welds"""
    from . import config as CF
    cfg = cfg or CF.load_config()
    frozen = frozen or load_frozen()
    kit = load_kit(kit_dir)
    placements = placements or {}
    rest_bodies = rest_bodies or {}
    arrays = {}
    sc = (collision_model(collision_source) is not None) if (self_collide is None and collision) else bool(self_collide)
    base, a = assembly(kit, "base", cfg, frozen, placements.get("base"), visual, collision, collision_source,
                       self_collide=sc)
    arrays.update(a)
    mods = []
    for u in (modules if modules is not None else ("leg_R", "leg_L", "arm_R", "arm_L", "head")):
        m, a = assembly(kit, u, cfg, frozen, placements.get(u), visual, collision, collision_source,
                        rest_bodies=rest_bodies.get(u, ()), self_collide=sc)
        mods.append(m)
        arrays.update(a)
    data = {"schema": "humanoid-cell-data/1", "name": name, "base": base, "modules": mods, "props": [],
            "fixtures": list(fixtures), "cameras": list(cameras), "welds": list(welds),
            "provenance": {"kit": os.path.join(kit["_dir"], "kit.json"), "adapter": "kit_adapter.py",
                           "collision": collision_source or PIECES_SOURCE}}
    return data, arrays
