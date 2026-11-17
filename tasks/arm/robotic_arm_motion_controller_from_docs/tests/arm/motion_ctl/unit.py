"""Unit layer of the motion-controller task (`motion-unit/1`): one physical copy of the documented arm.

The documented arm is the design `H_asm2` of `arm_family_v1` (the arm the published assembly task builds); its GT MJCF
(`gt_arm.mjcf.write_gt_mjcf`) and its calibration layer are frozen in the task's `design/` folder (`design.json`). A UNIT
is a pure function of an integer seed and has these layers, every range inside what the agent's documents state:

  C   installation and calibration: each servo's true zero (the design angle at which it reads bus 0) is the design's
      bus zero + U[-8, 8] deg (the horn sits on a different spline tooth on every unit; the jaw too); the unit's
      calibration RECORD (taken with the assembly jig, given to the controller at runtime) misses the true zero by
      U[-0.5, 0.5] deg (jaw U[-1, 1]) and is quantised to whole ticks; the EEPROM angle limits are the design limits
      written through the RECORD (so the true limits move with the record's error).
  U   geometric tolerances (the drawings' tolerance block): joint-frame errors +-0.3 mm per axis, pitch-axis tilts
      <= 0.15 deg (two angles), upper-arm / forearm axis distances x U[0.997, 1.003], wrist-to-tip +-0.5 mm along the
      roll axis; position-loop gain x U[0.85, 1.15], joint friction x U[0.7, 1.4] (the datasheet).
  M   masses: every printed link's mass and inertia x U[0.95, 1.05] (the mass table's tolerance).
  D   servo dynamics (the datasheet's ranges; the bands of the information tier's control stage): output compliance
      behind the encoder (shoulder / elbow 60-180, wrist pitch 120-400, yaw / roll 200-600 N m/rad), hysteresis 0.05-0.25
      deg, command latency 15-40 ms, max speed x 0.85-1.0, load-reading scale x 0.9-1.1.

`unit_mjcf(design_xml, spec, cell=None)` writes the grader's model of the unit: the design's robot.xml with the U and
M edits, the series compliance hinges `<joint>_slack` (stiff springs whose rest position the plant moves: the play
operator of `plant.py`), the clamped tool bodies (`part`, `needle`; enabled per job at run time), the table and the
cell's true geometry. numpy / xml / mujoco only; names no source.
"""

import copy
import hashlib
import json
import math
import os
import xml.etree.ElementTree as ET
import zlib

import numpy as np

UNIT_FORMAT = "motion-unit/1"
GENERATOR = "motion-unit/1.0.0"
DESIGN_ID = "H_asm2"

JOINTS = ["base_yaw", "shoulder_pitch", "elbow_pitch", "wrist_pitch", "wrist_roll", "jaw"]
ARM_JOINTS = JOINTS[:5]
CHILD = {"base_yaw": "shoulder", "shoulder_pitch": "upper_arm", "elbow_pitch": "forearm", "wrist_pitch": "wrist",
         "wrist_roll": "gripper", "jaw": "jaw"}
LINKS = ["base", "shoulder", "upper_arm", "forearm", "wrist", "gripper", "jaw"]
SERVO_IDS = {j: i + 1 for i, j in enumerate(JOINTS)}

TICKS = 4096
MID = 2048
TICK_RAD = 2.0 * math.pi / TICKS

INSTALL_DEG = 8.0
RECORD_ERR_DEG = {"arm": 0.5, "jaw": 1.0}
FRAME_ERR_MM = 0.3
TILT_DEG = 0.15
BEAM_SCALE = (0.997, 1.003)
WRIST_TIP_MM = 0.5
GAIN = (0.85, 1.15)
FRICTION = (0.7, 1.4)
MASS_SCALE = (0.95, 1.05)
COMPLIANCE_BANDS = {"base_yaw": (200.0, 600.0), "shoulder_pitch": (60.0, 180.0), "elbow_pitch": (60.0, 180.0),
                    "wrist_pitch": (120.0, 400.0), "wrist_roll": (200.0, 600.0)}
HYSTERESIS_DEG = (0.05, 0.25)
LATENCY_MS = (15.0, 40.0)
SPEED_FACTOR = (0.85, 1.0)
LOAD_SCALE = (0.9, 1.1)

SLACK_RANGE_RAD = 0.05
SLACK_DYN_FACTOR = 25.0          # the slack spring is this much stiffer than the compliance; the plant's play operator
                                 # moves its rest position (gt_arm.unit's reason: MuJoCo's soft frictionloss creeps)
SIM_DT = 0.002
MECH_MARGIN_RAD = math.radians(1.0)
CLEAR_DETECT_M = 0.015           # the clearance probe reports contacts up to this distance from a cell geom


# ------------------------------------------------------------------------------------------------------------ design
def load_design(design_dir):
    """design.json: {"id", "bus_zero_deg" {j}, "sign" {j}, "limits_rad" {j: [lo, hi]} (design angles), ...}."""
    with open(os.path.join(design_dir, "design.json")) as f:
        return json.load(f)


# ------------------------------------------------------------------------------------------------------------- draws
def rng(seed, label):
    s = int(seed) & ((1 << 64) - 1)
    return np.random.default_rng(np.random.SeedSequence([s & 0xFFFFFFFF, s >> 32, zlib.crc32(label.encode()) & 0xFFFFFFFF]))


def _r(v, nd=4):
    return round(float(v), nd)


def os_seed():
    return int.from_bytes(os.urandom(8), "little") & ((1 << 63) - 1)


def draw_unit(seed, design, serial=None):
    """The unit spec of `seed` against the design document (`load_design`)."""
    r = rng(seed, "unit/C")
    install = {j: _r(r.uniform(-INSTALL_DEG, INSTALL_DEG)) for j in JOINTS}
    rec_err = {j: _r(r.uniform(-1.0, 1.0) * RECORD_ERR_DEG["jaw" if j == "jaw" else "arm"]) for j in JOINTS}
    r = rng(seed, "unit/U")
    U = {"frame_err_mm": {j: [_r(r.uniform(-FRAME_ERR_MM, FRAME_ERR_MM)) for _ in range(3)] for j in JOINTS},
         "pitch_tilt_deg": {j: [_r(r.uniform(-TILT_DEG, TILT_DEG)), _r(r.uniform(-TILT_DEG, TILT_DEG))]
                            for j in ("shoulder_pitch", "elbow_pitch", "wrist_pitch")},
         "beam_scale": {"upper_arm": _r(r.uniform(*BEAM_SCALE), 6), "forearm": _r(r.uniform(*BEAM_SCALE), 6)},
         "wrist_to_tip_mm": _r(r.uniform(-WRIST_TIP_MM, WRIST_TIP_MM)),
         "gain": {j: _r(r.uniform(*GAIN)) for j in JOINTS}, "friction": {j: _r(r.uniform(*FRICTION)) for j in JOINTS}}
    r = rng(seed, "unit/M")
    M = {"mass_scale": {l: _r(r.uniform(*MASS_SCALE)) for l in LINKS}}
    r = rng(seed, "unit/D")
    D = {"compliance_nm_per_rad": {j: _r(r.uniform(*COMPLIANCE_BANDS[j]), 3) for j in ARM_JOINTS},
         "hysteresis_deg": {j: _r(r.uniform(*HYSTERESIS_DEG)) for j in ARM_JOINTS},
         "latency_ms": _r(r.uniform(*LATENCY_MS), 3),
         "speed_factor": {j: _r(r.uniform(*SPEED_FACTOR)) for j in JOINTS},
         "load_scale": {j: _r(r.uniform(*LOAD_SCALE)) for j in JOINTS},
         "load_noise_seed": int(r.integers(0, 2 ** 62))}
    if serial is None:
        serial = "MU-{:04X}{}".format(int(rng(seed, "serial").integers(0, 1 << 16)), "ABCDEFGH"[int(seed) % 8])
    spec = {"format": UNIT_FORMAT, "generator": GENERATOR, "design": design["id"], "seed": int(seed), "serial": serial,
            "C": {"install_deg": install, "record_err_deg": rec_err}, "U": U, "M": M, "D": D}
    spec["C"]["true"] = true_calibration(spec, design)
    spec["record"] = calibration_record(spec, design)
    return spec


def nominal_spec(design, serial="NOMINAL"):
    """The documented arm with no unit deviations (the record exact, mid-band dynamics): what a controller that knows
    only the documents can model at best; the job generator's second feasibility check."""
    D = {"compliance_nm_per_rad": {j: round(math.sqrt(COMPLIANCE_BANDS[j][0] * COMPLIANCE_BANDS[j][1]), 3) for j in ARM_JOINTS},
         "hysteresis_deg": {j: 0.15 for j in ARM_JOINTS}, "latency_ms": 27.5,
         "speed_factor": {j: 0.925 for j in JOINTS}, "load_scale": {j: 1.0 for j in JOINTS}, "load_noise_seed": 0}
    U_ = {"frame_err_mm": {j: [0.0, 0.0, 0.0] for j in JOINTS},
          "pitch_tilt_deg": {j: [0.0, 0.0] for j in ("shoulder_pitch", "elbow_pitch", "wrist_pitch")},
          "beam_scale": {"upper_arm": 1.0, "forearm": 1.0}, "wrist_to_tip_mm": 0.0,
          "gain": {j: 1.0 for j in JOINTS}, "friction": {j: 1.0 for j in JOINTS}}
    spec = {"format": UNIT_FORMAT, "generator": GENERATOR, "design": design["id"], "seed": 0, "serial": serial,
            "C": {"install_deg": {j: 0.0 for j in JOINTS}, "record_err_deg": {j: 0.0 for j in JOINTS}}, "U": U_,
            "M": {"mass_scale": {l: 1.0 for l in LINKS}}, "D": D}
    spec["C"]["true"] = true_calibration(spec, design)
    spec["record"] = calibration_record(spec, design)
    return spec


def believed_spec(design, record):
    """What a documents-only controller can believe about a unit: the nominal design and servo values, and the unit's
    calibration record taken as exact (the documents-ideal reference of the band calibration)."""
    spec = nominal_spec(design, serial=record.get("serial", "BELIEVED"))
    spec["record"] = json.loads(json.dumps(record))
    for j in JOINTS:
        r = record["servos"][j]
        spec["C"]["true"][j] = {"sign": int(r["direction"]), "zero_rad": -(int(r["zero_ticks"]) - MID) * TICK_RAD * int(r["direction"])}
    return spec


def true_calibration(spec, design):
    """{joint: {"sign", "zero_rad"}}: the unit's TRUE bus map b = sign x (q - zero_rad)."""
    C = spec["C"]
    return {j: {"sign": int(design["sign"][j]),
                "zero_rad": math.radians(float(design["bus_zero_deg"][j]) + float(C["install_deg"][j]))} for j in JOINTS}


def _q_to_ticks(q, zero, sign):
    return MID + sign * (q - zero) / TICK_RAD


def calibration_record(spec, design):
    """The unit's calibration record (what the controller receives at runtime): per servo the id, the direction, the
    tick reading at design angle 0 (whole ticks, from the jig: +-0.5 deg), the EEPROM angle-limit ticks."""
    tc = spec["C"]["true"]
    servos = {}
    for j in JOINTS:
        s = tc[j]["sign"]
        z_rec = tc[j]["zero_rad"] + math.radians(float(spec["C"]["record_err_deg"][j]))
        zero_ticks = int(round(_q_to_ticks(0.0, z_rec, s)))
        lo, hi = [float(v) for v in design["limits_rad"][j]]
        # the assembler writes the design limits through the record (rounded inwards to whole ticks)
        z_eff = -(zero_ticks - MID) * TICK_RAD * s           # the record's zero, de-quantised
        t = sorted([_q_to_ticks(lo, z_eff, s), _q_to_ticks(hi, z_eff, s)])
        servos[j] = {"id": SERVO_IDS[j], "direction": s, "zero_ticks": zero_ticks,
                     "min_ticks": int(math.ceil(t[0] - 1e-9)), "max_ticks": int(math.floor(t[1] + 1e-9))}
    return {"format": "calibration-record/1", "serial": spec["serial"],
            "convention": "q = direction x (ticks - zero_ticks) x 2 pi / 4096 (q = design angle of the kinematics sheet)",
            "uncertainty_deg": {"arm joints": RECORD_ERR_DEG["arm"], "jaw": RECORD_ERR_DEG["jaw"]}, "servos": servos}


def true_limits_rad(spec):
    """Design-angle interval of each joint's EEPROM limits under the TRUE zero."""
    out = {}
    for j in JOINTS:
        s = spec["C"]["true"][j]["sign"]
        z = spec["C"]["true"][j]["zero_rad"]
        rec = spec["record"]["servos"][j]
        qa = s * (rec["min_ticks"] - MID) * TICK_RAD + z
        qb = s * (rec["max_ticks"] - MID) * TICK_RAD + z
        out[j] = sorted([qa, qb])
    return out


def bus_cal(spec):
    """gt_arm.servo.ServoBus calibration dict of the unit (the TRUE map + the EEPROM registers)."""
    out = {}
    for j in JOINTS:
        rec = spec["record"]["servos"][j]
        out[j] = {"sign": spec["C"]["true"][j]["sign"], "zero_offset_rad": spec["C"]["true"][j]["zero_rad"],
                  "min_ticks": rec["min_ticks"], "max_ticks": rec["max_ticks"]}
    return out


def dynamics(spec):
    D = spec["D"]
    return {"latency_s": float(D["latency_ms"]) * 1e-3, "speed_factor": dict(D["speed_factor"]), "load_scale": dict(D["load_scale"])}


def spec_digest(spec):
    return hashlib.sha256(json.dumps(spec, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


# ------------------------------------------------------------------------------------------------------------- MJCF
def _floats(s):
    return np.array([float(v) for v in str(s).split()])


def _fl(v, nd=9):
    return " ".join(("{:." + str(nd) + "g}").format(float(x)) for x in np.atleast_1d(v))


def _rot(axis, ang):
    a = np.asarray(axis, float)
    a = a / np.linalg.norm(a)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + math.sin(ang) * K + (1 - math.cos(ang)) * K @ K


def _quat_R(q):
    w, x, y, z = np.asarray(q, float) / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _slack_params(k, inertia):
    kk = k * SLACK_DYN_FACTOR
    arm = 1.5 * kk * SIM_DT ** 2
    c = 2.0 * math.sqrt(kk * (inertia + arm))
    return {"stiffness": kk, "armature": arm, "damping": c}


_INERTIA_CACHE = {}


def _link_inertias(design_xml):
    """Inertia seen by each arm joint about its axis at a mid-reach pose (sets the slack damping)."""
    key = os.path.abspath(design_xml)
    if key in _INERTIA_CACHE:
        return _INERTIA_CACHE[key]
    import mujoco
    m = mujoco.MjModel.from_xml_path(design_xml)
    d = mujoco.MjData(m)
    d.qpos[:] = [0.0, -0.8, -0.9, -0.6, 0.0, 0.0][:m.nq]
    mujoco.mj_forward(m, d)
    Mf = np.zeros((m.nv, m.nv))
    try:
        mujoco.mj_fullM(m, d, Mf)                      # MuJoCo >= 3.3 signature
    except TypeError:
        mujoco.mj_fullM(m, Mf, d.qM)
    out = {}
    for j in ARM_JOINTS:
        dof = m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)]
        out[j] = max(1e-6, float(Mf[dof, dof] - m.dof_armature[dof]))
    _INERTIA_CACHE[key] = out
    return out


def tool_seat(design, tool):
    """The clamped tool's placement in the TCP frame: part centre = TCP + seat_x x + seat_z z with its -x face on the
    fixed finger's pad face; the needle holder fills the pad gap at q6 = 0, the needle along the TCP's +z. Returns
    {"seat_x", "seat_z", "jaw_q"} (jaw_q = the jaw's design angle while clamped, from the design's pad-gap table)."""
    fixed_x = -(float(design["fixed_pad_face_gripper_y_m"]) - float(design["tcp_gripper_m"][1]))   # TCP x = gripper -y
    table = design["jaw_gap_mm"]
    qs = [t[0] for t in table]
    gs = [t[1] for t in table]
    if tool["kind"] == "part":
        hx = float(tool["params"]["half"][0])
        jaw_q = float(np.interp(2.0 * hx * 1e3, gs, qs))
        return {"seat_x": fixed_x + hx, "seat_z": float(tool["params"]["seat_z"]), "jaw_q": jaw_q}
    if tool["kind"] == "needle":
        jaw_q = float(np.interp(2.0 * NEEDLE_HOLDER_HALF_M[0] * 1e3, gs, qs))
        return {"seat_x": fixed_x + NEEDLE_HOLDER_HALF_M[0], "seat_z": 0.0, "jaw_q": jaw_q}
    return {"seat_x": 0.0, "seat_z": 0.0, "jaw_q": None}


NEEDLE_HOLDER_HALF_M = (0.00784, 0.006, 0.009)      # the dispensing needle's holder block, clamped in the pad gap
NEEDLE_HOLDER_Z_M = -0.003                          # its centre along the TCP's z (the needle leaves it at +0.006 m)


def unit_mjcf(design_xml, spec, cell=None, meshdir=None, model_name="motion_unit", probe=False, tool=None, design=None):
    """MJCF text of the unit (grader form). `cell` = a cells.Cell (true geometry) or None (table only). `tool` = the
    cell's clamped tool ({"kind": "part" | "needle", "params"}) or None, compiled in with its collision geometry
    (`design` = the design document, for the seat). `probe`: the clearance probe's copy (kinematics and collision only,
    never stepped): the cell and table geoms report contacts up to CLEAR_DETECT_M away (MuJoCo 3.14 makes margin
    contacts active, so the physics model has no margin)."""
    tree = ET.parse(design_xml)
    root = tree.getroot()
    root.set("model", model_name)
    comp = root.find("compiler")
    comp.set("meshdir", meshdir if meshdir is not None else os.path.join(os.path.dirname(os.path.abspath(design_xml)), "assets"))
    opt = root.find("option")
    opt.set("integrator", "implicitfast")
    U, M, D = spec["U"], spec["M"], spec["D"]
    lim = true_limits_rad(spec)
    I_link = _link_inertias(design_xml)
    bodies = {b.get("name"): b for b in root.iter("body")}
    for j in JOINTS:
        child = bodies[CHILD[j]]
        pos = _floats(child.get("pos", "0 0 0"))
        if j == "elbow_pitch":
            pos = pos * float(U["beam_scale"]["upper_arm"])
        if j == "wrist_pitch":
            pos = pos * float(U["beam_scale"]["forearm"])
        if j == "wrist_roll":
            pos = pos + np.array([0.0, 0.0, float(U["wrist_to_tip_mm"]) * 1e-3])
        pos = pos + np.asarray(U["frame_err_mm"][j], float) * 1e-3
        child.set("pos", _fl(pos))
        jel = child.find("joint")
        assert jel is not None and jel.get("name") == j, j
        ax = _floats(jel.get("axis"))
        if j in U["pitch_tilt_deg"]:
            a, b = [math.radians(v) for v in U["pitch_tilt_deg"][j]]
            ax = _rot([0, 0, 1], b) @ _rot([0, 1, 0], a) @ ax
            ax = ax / np.linalg.norm(ax)
        jel.set("axis", _fl(ax, 12))
        lo, hi = lim[j]
        jel.set("range", _fl([lo - MECH_MARGIN_RAD, hi + MECH_MARGIN_RAD]))
        jel.set("frictionloss", _fl(float(jel.get("frictionloss", "0.052")) * float(U["friction"][j])))
        if j in ARM_JOINTS:
            sp = _slack_params(float(D["compliance_nm_per_rad"][j]), I_link[j])
            sl = ET.Element("joint", {"name": j + "_slack", "type": "hinge", "axis": _fl(ax, 12),
                                      "range": _fl([-SLACK_RANGE_RAD, SLACK_RANGE_RAD]), "limited": "true",
                                      "stiffness": _fl(sp["stiffness"]), "frictionloss": "0", "springref": "0",
                                      "damping": _fl(sp["damping"]), "armature": _fl(sp["armature"])})
            child.insert(list(child).index(jel) + 1, sl)
    for l, s in M["mass_scale"].items():
        inert = bodies[l].find("inertial")
        inert.set("mass", _fl(float(inert.get("mass")) * float(s)))
        if inert.get("fullinertia"):
            inert.set("fullinertia", _fl(_floats(inert.get("fullinertia")) * float(s), 12))
        if inert.get("diaginertia"):
            inert.set("diaginertia", _fl(_floats(inert.get("diaginertia")) * float(s), 12))
    for a in root.find("actuator"):
        j = a.get("joint")
        a.set("kp", _fl(float(a.get("kp", "998.22")) * float(U["gain"][j])))
        lo, hi = lim[j]
        a.set("ctrlrange", _fl([lo - MECH_MARGIN_RAD, hi + MECH_MARGIN_RAD]))
    # the clamped tool (one per cell), compiled in: its geoms collide with the cell (never with the gripper, jaw, wrist)
    g = bodies["gripper"]
    tcp_el = next(st for st in g.iter("site") if st.get("name") == "tcp")
    tcp_p = _floats(tcp_el.get("pos"))
    tcp_q = _floats(tcp_el.get("quat"))
    Rt = _quat_R(tcp_q)
    tool_bodies = []
    if tool is not None and tool.get("kind") in ("part", "needle"):
        seat = tool_seat(design, tool)
        P_ = tool["params"]
        if tool["kind"] == "part":
            h = np.asarray(P_["half"], float)
            mass = float(P_["mass_kg"])
            pos = tcp_p + Rt @ np.array([seat["seat_x"], 0.0, seat["seat_z"]])
            b = ET.SubElement(g, "body", {"name": "part", "pos": _fl(pos), "quat": _fl(tcp_q)})
            I = mass / 3.0 * np.array([h[1] ** 2 + h[2] ** 2, h[0] ** 2 + h[2] ** 2, h[0] ** 2 + h[1] ** 2])
            ET.SubElement(b, "inertial", {"pos": _fl(P_["com_part"]), "mass": _fl(mass), "diaginertia": _fl(I, 12)})
            ET.SubElement(b, "geom", {"name": "part_geom", "type": "box", "size": _fl(h), "contype": "1", "conaffinity": "1",
                                      "mass": "0", "group": "1", "rgba": "0.85 0.6 0.2 1"})
            tool_bodies.append("part")
        else:
            L, r = float(P_["length_m"]), float(P_["radius_m"])
            pos = tcp_p + Rt @ np.array([seat["seat_x"], 0.0, 0.0])
            b = ET.SubElement(g, "body", {"name": "needle", "pos": _fl(pos), "quat": _fl(tcp_q)})
            ET.SubElement(b, "inertial", {"pos": "0 0 0.008", "mass": _fl(float(P_.get("mass_kg", 0.02))), "diaginertia": "4e-6 4e-6 2e-6"})
            ET.SubElement(b, "geom", {"name": "holder_geom", "type": "box", "size": _fl(NEEDLE_HOLDER_HALF_M), "pos": _fl([0, 0, NEEDLE_HOLDER_Z_M]),
                                      "contype": "1", "conaffinity": "1", "mass": "0", "group": "1", "rgba": "0.3 0.5 0.75 1"})
            ET.SubElement(b, "geom", {"name": "needle_geom", "type": "capsule", "fromto": _fl([0, 0, 0.006, 0, 0, L - r]), "size": _fl(r),
                                      "contype": "1", "conaffinity": "1", "mass": "0", "group": "1", "rgba": "0.75 0.75 0.78 1"})
            tool_bodies.append("needle")
    wb = root.find("worldbody")
    ET.SubElement(wb, "geom", {"name": "table", "type": "plane", "size": "1.2 1.2 0.01", "pos": "0 0 0",
                               "rgba": "0.55 0.55 0.52 1", "contype": "2", "conaffinity": "1",
                               "margin": _fl(CLEAR_DETECT_M if probe else 0.0)})
    if cell is not None:
        for o in cell.geoms():
            attrs = {"name": o["name"], "type": o["type"], "size": _fl(o["size"]), "pos": _fl(o["pos"]),
                     "quat": _fl(o["quat"]), "rgba": o.get("rgba", "0.7 0.7 0.72 1"), "contype": "2", "conaffinity": "1",
                     "margin": _fl(CLEAR_DETECT_M if probe else 0.0), "group": "1"}
            ET.SubElement(wb, "geom", attrs)
    con = root.find("contact")
    if con is None:
        con = ET.SubElement(root, "contact")
    for name in tool_bodies:
        for other in ("gripper", "jaw", "wrist"):
            ET.SubElement(con, "exclude", {"body1": name, "body2": other})
    for key in root.iter("key"):
        key.set("qpos", " ".join(["0"] * (len(JOINTS) + len(ARM_JOINTS))))
        key.set("ctrl", " ".join(["0"] * len(JOINTS)))
    ET.indent(tree, "  ")
    return ET.tostring(root, encoding="unicode") + "\n"


def public_summary(spec):
    """Grader stats of a unit (never shown to the agent during the episode)."""
    return {"serial": spec["serial"], "record_err_deg": dict(spec["C"]["record_err_deg"]),
            "latency_ms": spec["D"]["latency_ms"], "compliance_nm_per_rad": dict(spec["D"]["compliance_nm_per_rad"])}
