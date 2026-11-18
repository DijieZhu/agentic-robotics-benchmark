"""The wide parallel gripper of the final-assembly cell: a 140 mm-stroke two-finger adaptive gripper model with custom
fingertips, its measured controller tables, its force register, and its attachment to the arm's real tool flange.

The module keeps the gripper API the cell's controller and planner call (`PadGeometry`, `MAX_WIDTH_M`,
`ctrl_of_width`, `set_opening`; `ur_controller.py` imports this module as its gripper) and adds:
* `wide_gripper_spec()`: the gripper model (`<robot models>/wide_gripper/wide_gripper.xml`, a BSD-2 MJCF, read only)
  with the patches below applied in code;
* `add_ur_wide(spec, cfg)`: the arm (`ur_robot`'s model, flange, motors, sensors and helpers) with this gripper, the
  same returned names as `ur_robot.add_ur`;
* the force register (`force_cap`, `ForceRegister`, `WideController` = the arm controller + `command_force(F)`);
* `pad_offset(width)`: the grasp centre (midway between the two face middles) in the tcp frame. It is NOT the tcp:
  aim a part's centre with it, not with the tcp.
Every number of the opening and force tables comes from `gripper_tables.json` next to this file, measured on this
model under the cell's physics (the table's `meta.physics`: 0.5 ms implicitfast, 10 ms control tick, pad contact
time constant 1 ms, elliptic cones, impratio 100, noslip 5, g 9.80665 m/s^2).

Frames and names (gripper prefix `ur/g/` once attached):
* tool frame = the `ur/tcp` site on the gripper base: z = approach, y = closing axis (the right finger on +y), x across
  the pads. It sits at the face middle of a 114.3 mm opening (`TCP_IN_BASE`; `tcp_from_flange_m` in add_ur_wide's dict).
* the gripping face: one flat 27 x 65 mm face per finger on the follower link, made of two coplanar pads,
  `<side>_pad1` (the main pad, 53 mm long) and `<side>_pad2` (the fingertip blade, the last 12 mm, 3.0 mm thick).
  `s` = distance along the face from its root (0) to the fingertip end (`PAD_ALONG_M`); `PAD_MID_Z` = its middle.
* `width` = the gap between the two faces at their middle, along the closing axis.

The patches (the source file is never edited; every change is made here, `PATCHES`):
  P1 the two coupler joints deleted (the fingertip link rigid on the driver at the coupler's stop): the parallel
     (pinch) mode only; a free coupler folds the finger far below the grip force the cell needs.
  P2 the spring links' stiffness 0 (its preload only held the free coupler on its stop).
  P3 gravcomp 0 on every gripper body: the tool's weight is real (the arm's computed torque carries it; the flange
     force / torque sensor reads it).
  P4 custom fingertips: the stock pad box replaced by pad1 (27 x 53 x 5.0 mm, its back where the stock back is) and the
     fingertip blade pad2 (27 x 12 x 3.0 mm), the gripping face moved 2.5 mm outward per side. P4b the face wedged
     2.0 deg about its middle, tips inward: the 4-bar is not a parallelogram, the stock faces open towards the tips.
  P5 pad contact: condim 4, friction 0.7 / 0.005 / 0.0001 (torsional 0.005 m: a line contact on a cylinder),
     solref = PAD_SOLREF (the cell's pad time constant, 1 ms at a 0.5 ms step), priority 2 (above every part).
  P6 the drive: the source's position servo on the split tendon (kp 1000 per unit of tendon length, ctrl 0-255 ->
     0-0.7 rad) is kept; its force range is the FORCE REGISTER (rewritten every tick from the measured width, so
     the pads press with the commanded force at any opening). P6b its velocity gain kv 300 -> 50: under
     implicitfast a force-limited servo's velocity term is integrated explicitly and kv 300 (or 150) chatters
     period-2 at the force limit; the finger speed is set by the controller's 0.10 m/s width ramp instead.
  P7 visuals: the stock pad meshes removed, visual boxes for the custom fingertips (group 2, no contact).
  P8 the loop closures (two connects, the driver-to-driver equality) at solref 0.002 / solimp 0.99 0.999 (the
     source's 0.005 / 0.95 0.99 stretched 0.5-0.74 mm under a 100-125 N grip).

Provenance: the merge of the two prototype modules build_gripper.py (sha256 fc4331fb527e...c9b1d) and gripper_wide.py
(sha256 6d856b466141...eefb). Changes from them: one module; the source file is the given robot models folder's
wide_gripper/wide_gripper.xml; the pad solref is the cell's 1 ms (it was 2 ms at a 1 ms step); the tables carry the
grasp-centre columns (pad_offset) and the physics they were measured under; add_ur_wide carries ur_robot.add_ur's
wrist-camera mount; the force register is a class (ForceRegister) and the controller with it is WideController.
"""
import json
import math
import os

import numpy as np

from . import ur_robot as _ur

__all__ = ["GRIPPER_DIRNAME", "GRIPPER_XML", "DESIGN", "PATCHES", "PAD_ACROSS_M", "PAD_ALONG_M", "PAD_FACE_Y",
           "PAD_MID_Z", "TIP_SEG_M", "PAD_SOLREF", "OPENING_TABLE", "OPENING_COLUMNS", "FORCE_MODEL", "PadGeometry",
           "ctrl_of_width", "width_of_ctrl", "driver_of_width", "pad_centre_z", "pad_tip_z", "fingertip_z",
           "finger_outer_y", "linkage_qpos", "set_opening", "pad_offset", "pad_force", "force_ratio", "force_cap",
           "squeeze_for_force", "MAX_WIDTH_M", "FORCE_RANGE_N", "DEFAULT_FORCE_N", "TCP_IN_BASE", "LINKAGE",
           "PAD_GEOMS", "GRIPPER_JOINTS", "FINGER_PARTS", "TABLE_META", "TABLE_PHYSICS", "wide_gripper_spec",
           "add_ur_wide", "ForceRegister", "WideController", "set_pad_solref", "physics_mismatch", "export_xml",
           "face_local", "DEFAULT_CFG"]

HERE = os.path.dirname(os.path.abspath(__file__))
TABLES = os.path.join(HERE, "gripper_tables.json")
GRIPPER_DIRNAME = "wide_gripper"          # the gripper's folder in the robot models folder (ur_robot.ROBOT_MODELS)
GRIPPER_XML = "wide_gripper.xml"

# ------------------------------------------------------------------------------------------------------- the design
# the stock pad box of the source model (its default class pad_box1), follower frame, m: x across the finger (= the
# hinge axis), y along the finger (+ towards the tip), z the pad normal (+ outward). Checked against the loaded file.
STOCK_PAD = {"pos": [0.0, 0.0457554, -0.0272203], "half": [0.0135, 0.0325, 0.00375]}
STOCK_INNER_Z = STOCK_PAD["pos"][2] - STOCK_PAD["half"][2]       # the stock gripping face
STOCK_BACK_Z = STOCK_PAD["pos"][2] + STOCK_PAD["half"][2]
STOCK_ROOT_Y = STOCK_PAD["pos"][1] - STOCK_PAD["half"][1]
STOCK_TIP_Y = STOCK_PAD["pos"][1] + STOCK_PAD["half"][1]

# the pad contact's time constant: 2 physics steps of the cell's 0.5 ms (MuJoCo's minimum with refsafe on)
PAD_SOLREF = [0.001, 1.0]
DESIGN = {
    "face_shift_out_m": 0.0025,      # the gripping face moved outward per side (opening +5.0 mm)
    "main_thick_m": 0.0050,          # pad1 thickness (its back face stays at the stock back face)
    "tip_len_m": 0.012,              # pad2: the last 12 mm of the finger (the fingertip segment)
    "tip_thick_m": 0.0030,           # pad2 thickness
    "across_m": 2 * STOCK_PAD["half"][0],      # 27 mm, unchanged
    "wedge_deg": 2.0,                # P4b: the face turned about its middle, tips inward
    "friction": [0.7, 0.005, 0.0001],
    "solref": list(PAD_SOLREF),
    "solimp": [0.95, 0.99, 0.001, 0.5, 2.0],
    "priority": 2,
    "condim": 4,
}
FACE_Z = STOCK_INNER_Z + DESIGN["face_shift_out_m"]              # the gripping face at its middle, follower z
FACE_LEN = STOCK_TIP_Y - STOCK_ROOT_Y                             # 65 mm along the (wedged) face
FACE_MID_Y = 0.5 * (STOCK_ROOT_Y + STOCK_TIP_Y)
WEDGE_RAD = math.radians(DESIGN["wedge_deg"])
CW, SW = math.cos(WEDGE_RAD), math.sin(WEDGE_RAD)
N_OUT = (0.0, SW, CW)                                             # the face's outward normal, follower frame
assert abs((STOCK_BACK_Z - FACE_Z) - DESIGN["main_thick_m"]) < 1e-9, "pad1 must keep the stock back face"


def face_local(s):
    """follower-frame point of the gripping face at s (m) from its root towards the fingertip end"""
    u = s - 0.5 * FACE_LEN
    return (0.0, FACE_MID_Y + u * CW, FACE_Z - u * SW)


def _box_on_face(s0, s1, thick):
    """a box lying on the face between s0 and s1, `thick` outward: {pos, half, quat}"""
    c = face_local(0.5 * (s0 + s1))
    pos = [c[0], c[1] + 0.5 * thick * N_OUT[1], c[2] + 0.5 * thick * N_OUT[2]]
    half = [0.5 * DESIGN["across_m"], 0.5 * (s1 - s0), 0.5 * thick]
    th = -WEDGE_RAD                                                 # about x: local y -> (0, cos, -sin)
    return {"pos": pos, "half": half, "quat": [math.cos(0.5 * th), math.sin(0.5 * th), 0.0, 0.0]}


S_STEP = FACE_LEN - DESIGN["tip_len_m"]                           # where pad1 ends and the blade begins
PAD1 = _box_on_face(0.0, S_STEP, DESIGN["main_thick_m"])
PAD2 = _box_on_face(S_STEP, FACE_LEN, DESIGN["tip_thick_m"])

SERVO = {"kp": 1000.0, "kv": 50.0, "target_rad": 0.7}       # kv: the source's 300 (P6b)
FORCE_RANGE_N = (10.0, 125.0)        # the force register's range, per pad
DEFAULT_FORCE_N = 100.0
BUILD_FORCE_RATIO = 8.0              # pad N per tendon unit for the build-time force range (the register rewrites it)
LOOP_SOLREF = [0.002, 1.0]
LOOP_SOLIMP = [0.99, 0.999, 0.001, 0.5, 2.0]
SIDES = ("right", "left")
PATCHES = ["P1 coupler joints deleted (pinch mode)", "P2 spring-link stiffness 0", "P3 gravcomp 0",
           "P4 custom fingertips: face +2.5 mm out, pad1 5.0 mm, blade 12 mm x 3.0 mm; P4b face wedge 2.0 deg",
           "P5 pad contact condim 4, friction 0.7/0.005/0.0001, solref PAD_SOLREF, priority 2",
           "P6 force register on the tendon servo's force range; P6b servo kv 300 -> 50", "P7 visual fingertip boxes",
           "P8 loop closures solref 0.002 / solimp 0.99 0.999"]

PAD_ACROSS_M = DESIGN["across_m"]          # 0.027
PAD_ALONG_M = FACE_LEN                      # 0.065
PAD_FACE_Y = FACE_Z                         # the face middle's follower-frame depth (the controller's name kept)
PAD_MID_Z = 0.5 * FACE_LEN                  # s of the face middle
TIP_SEG_M = DESIGN["tip_len_m"]
LINKAGE = ["right_driver_joint", "right_spring_link_joint", "right_follower_joint", "left_driver_joint",
           "left_spring_link_joint", "left_follower_joint"]
GRIPPER_JOINTS = list(LINKAGE)
PAD_GEOMS = ["right_pad1", "right_pad2", "left_pad1", "left_pad2"]
FINGER_PARTS = ("driver", "coupler", "spring_link", "follower", "pad")
OPENING_COLUMNS = ["ctrl", "driver_rad", "gap_mid_m", "gap_root_m", "gap_tip_m", "pad_centre_z_m", "pad_tip_z_m",
                   "fingertip_z_m", "finger_outer_y_m", "tilt_deg"] + ["q_" + j for j in LINKAGE] + \
                  ["pad_off_x_m", "pad_off_y_m", "pad_off_z_m"]

# ------------------------------------------------------------------------------------------------ measured tables
OPENING_TABLE, FORCE_MODEL, TCP_IN_BASE, MAX_WIDTH_M, TABLE_META, TABLE_PHYSICS = None, None, [0.0, 0.0, 0.1854], 0.130, {}, {}
_BYW = {}                            # column name -> (widths ascending, values), built once
if os.path.exists(TABLES):
    with open(TABLES) as _f:
        _T = json.load(_f)
    if _T.get("columns") != OPENING_COLUMNS:
        raise RuntimeError("gripper_tables.json: the opening table's columns differ from OPENING_COLUMNS")
    OPENING_TABLE = _T["opening_table"]
    FORCE_MODEL = _T["force_model"]
    TCP_IN_BASE = _T["tcp_in_base_m"]
    MAX_WIDTH_M = _T["max_width_m"]
    TABLE_META = _T.get("meta", {})
    TABLE_PHYSICS = TABLE_META.get("physics", {})
    _rows = sorted(OPENING_TABLE, key=lambda r: r[2])
    _w = np.array([r[2] for r in _rows])
    for _i, _n in enumerate(OPENING_COLUMNS):
        _BYW[_n] = (_w, np.array([r[_i] for r in _rows]))
    _c = np.array([r[0] for r in OPENING_TABLE])
    _o = np.argsort(_c)
    _BYC = (_c[_o], np.array([r[2] for r in OPENING_TABLE])[_o])


def _need(x, what):
    if x is None:
        raise RuntimeError("gripper_tables.json missing (%s): the gripper's tables are not measured" % what)
    return x


# ------------------------------------------------------------------------------------------------ the model
def default_xml(models_dir=None):
    return os.path.join(models_dir or _ur.ROBOT_MODELS, GRIPPER_DIRNAME, GRIPPER_XML)


def wide_gripper_spec(xml=None, design=None):
    """the patched gripper as an MjSpec (meshes absolute). Body / joint / geom / actuator names are the source's
    (attach with a prefix); the pad geoms are <side>_pad1 (main) and <side>_pad2 (fingertip blade)."""
    import mujoco
    des = dict(DESIGN, **(design or {}))
    xml = xml or default_xml()
    spec = _ur._absolute_meshes(mujoco.MjSpec.from_file(xml), xml)
    for k in list(spec.keys):            # the source's keyframe has its 8 joints
        spec.delete(k)
    for s in SIDES:                      # the source must be the model these patches were measured on
        g = spec.geom(s + "_pad1")
        if (np.max(np.abs(np.asarray(g.pos, float) - STOCK_PAD["pos"])) > 1e-9 or
                np.max(np.abs(np.asarray(g.size, float)[:3] - STOCK_PAD["half"])) > 1e-9):
            raise ValueError("wide_gripper_spec: %s is not the expected stock pad" % (s + "_pad1"))
    # P1
    for s in SIDES:
        spec.delete(spec.joint(s + "_coupler_joint"))
    # P2
    for s in SIDES:
        spec.joint(s + "_spring_link_joint").stiffness = [0.0, 0.0, 0.0]
    # P3
    for b in spec.bodies:
        b.gravcomp = 0.0
    # P4 / P5 / P7
    for s in SIDES:
        fol = spec.body(s + "_follower")
        g1 = spec.geom(s + "_pad1")
        g1.pos = list(PAD1["pos"])
        g1.size = list(PAD1["half"])
        g1.quat = list(PAD1["quat"])
        g2 = fol.add_geom(name=s + "_pad2", type=mujoco.mjtGeom.mjGEOM_BOX, pos=list(PAD2["pos"]),
                          size=list(PAD2["half"]), quat=list(PAD2["quat"]), mass=0.0, group=3)
        for g in (g1, g2):
            g.condim = int(des["condim"])
            g.friction = list(des["friction"])
            g.solref = list(des["solref"])
            g.solimp = list(des["solimp"])
            g.priority = int(des["priority"])
            g.rgba = [0.55, 0.55, 0.55, 1.0]
        pb = spec.body(s + "_pad")
        for g in list(pb.geoms):
            spec.delete(g)
        for nm, P in (("_pad1_vis", PAD1), ("_pad2_vis", PAD2)):
            fol.add_geom(name=s + nm, type=mujoco.mjtGeom.mjGEOM_BOX, pos=list(P["pos"]), size=list(P["half"]),
                         quat=list(P["quat"]), contype=0, conaffinity=0, group=2, mass=0.0,
                         rgba=[0.10, 0.10, 0.11, 1.0])
    # P8
    for e in spec.equalities:
        e.solref = list(LOOP_SOLREF)
        e.solimp = list(LOOP_SOLIMP)
    # P6
    act = spec.actuator("fingers_actuator")
    cap = DEFAULT_FORCE_N / BUILD_FORCE_RATIO
    act.forcerange = [-cap, cap]
    act.gainprm[0] = SERVO["kp"] * SERVO["target_rad"] / 255.0
    act.biasprm[1] = -SERVO["kp"]
    act.biasprm[2] = -float(des.get("servo_kv", SERVO["kv"]))
    return spec


def export_xml(path, xml=None, design=None):
    """write the patched gripper alone as an MJCF (mesh paths relative to `path`'s folder); returns the path"""
    spec = wide_gripper_spec(xml, design)
    spec.modelname = "wide_gripper_patched"
    out_dir = os.path.dirname(os.path.abspath(path))
    for mesh in spec.meshes:
        if mesh.file and os.path.isabs(mesh.file):
            mesh.file = os.path.relpath(mesh.file, out_dir)
    text = spec.to_xml()
    with open(path, "w") as f:
        f.write(text)
    return path


DEFAULT_CFG = {
    "model": "ur10e",
    "pos": [0.0, 0.0, 0.0],
    "yaw_deg": 0.0,
    "prefix": "ur/",
    "gripper_prefix": "g/",
    "models_dir": None,                 # None: ur_robot.ROBOT_MODELS (the given robot models folder)
    "gripper_xml": None,                # None: <models_dir>/wide_gripper/wide_gripper.xml
    "design": None,                     # overrides of DESIGN (e.g. {"solref": [0.002, 1.0]} for a 1 ms cell)
    "impratio": 100.0,
    "tcp_name": "tcp",
    "ft_site": "flange_ft",
    "force_sensor": "flange_force",
    "torque_sensor": "flange_torque",
    # the wrist camera mount as ur_robot.add_ur's (lens 58 mm off the tool axis on the flange's -y side = the tcp's
    # +x side, across the pads, so the view runs between the fingers), aimed past this gripper's longer fingers
    "wrist_camera": {"mount": "wrist_cam_mount", "site": "wrist_cam", "camera": "wrist", "fovy_deg": 70.0,
                     "lens_in_flange_m": [0.0, -0.058, 0.036], "aim_in_flange_m": [0.0, 0.0, 0.315],
                     "housing_centre_in_flange_m": [0.0, -0.056, 0.0185], "housing_half_m": [0.018, 0.012, 0.0175],
                     "housing_mass_kg": 0.0},
}


def add_ur_wide(spec, cfg=None):
    """Attach the UR arm with the wide gripper into `spec` (an MjSpec): the arm, its flange, motors, sensors, numeric
    and text exactly as `ur_robot.add_ur` (its constants and helpers are used, not copied); the gripper is
    `wide_gripper_spec()`, the tcp site at TCP_IN_BASE on the gripper base. Returns the dict `add_ur` returns (same
    keys; the pads are on the followers)."""
    import mujoco
    cfg = _ur._merge(DEFAULT_CFG, cfg)
    rob = _ur.UR_MODELS[cfg["model"]]
    P, G = cfg["prefix"], cfg["gripper_prefix"]
    mdir = cfg["models_dir"] or _ur.ROBOT_MODELS
    ax = os.path.join(mdir, rob["dir"], rob["xml"])
    arm = _ur._absolute_meshes(mujoco.MjSpec.from_file(ax), ax)
    grip = wide_gripper_spec(cfg["gripper_xml"] or default_xml(mdir), cfg.get("design"))
    for k in list(arm.keys):
        arm.delete(k)
    for a in list(arm.actuators):
        arm.delete(a)
    lim = rob["torque_limits_Nm"]
    for an, jn, tl in zip(_ur.ACTUATOR_NAMES, _ur.JOINT_NAMES, lim):
        a = arm.add_actuator(name=an, target=jn, trntype=mujoco.mjtTrn.mjTRN_JOINT)
        a.set_to_motor()
        a.ctrlrange = [-tl, tl]
        a.ctrllimited = mujoco.mjtLimited.mjLIMITED_TRUE
        a.forcerange = [-tl, tl]
        a.forcelimited = mujoco.mjtLimited.mjLIMITED_TRUE
    site = arm.site("attachment_site")
    w3 = arm.body("wrist_3_link")
    flange = w3.add_frame(pos=[0.0, float(rob["d6_m"]), 0.0], quat=list(site.quat))
    _ur._align_options(grip, arm)
    flange.attach_body(grip.body("base_mount"), G, "")
    mount = arm.body(G + "base_mount")
    base = arm.body(G + "base")
    mount_z = float(mount.pos[2])
    if abs(mount_z - _ur.GRIP_MOUNT_Z) > 1e-12:
        raise ValueError("add_ur_wide: the gripper's base_mount is not %.4f m out from the flange" % _ur.GRIP_MOUNT_Z)
    base_z = float(base.pos[2])
    base.add_site(name=cfg["tcp_name"], pos=list(TCP_IN_BASE), size=[0.004, 0.004, 0.004], group=4,
                  rgba=[0.2, 0.9, 0.2, 0.6])
    mount.add_site(name=cfg["ft_site"], pos=[0.0, 0.0, -mount_z], size=[0.003, 0.003, 0.003], group=4)
    wc = cfg["wrist_camera"]
    off = np.array([0.0, 0.0, -mount_z])                 # flange frame -> base_mount frame
    hc = np.asarray(wc["housing_centre_in_flange_m"], float) + off
    mount.add_geom(name="wrist_cam_housing", type=mujoco.mjtGeom.mjGEOM_BOX, size=list(wc["housing_half_m"]),
                   pos=hc.tolist(), rgba=[0.12, 0.12, 0.13, 1.0], mass=float(wc["housing_mass_kg"]), group=2)
    Rc = _ur._camera_frame(wc["lens_in_flange_m"], wc["aim_in_flange_m"])
    cam_body = mount.add_body(name=wc["mount"], pos=(np.asarray(wc["lens_in_flange_m"], float) + off).tolist(),
                              quat=_ur.quat_wxyz(Rc).tolist())
    cam_body.add_site(name=wc["site"], size=[0.003, 0.003, 0.003], group=4, rgba=[0.9, 0.6, 0.1, 0.6])
    if wc.get("camera"):
        cam_body.add_camera(name=wc["camera"], fovy=float(wc["fovy_deg"]))
    frame = spec.worldbody.add_frame(pos=[float(v) for v in cfg["pos"]],
                                     quat=_ur.quat_wxyz(_ur.rotz(math.radians(float(cfg["yaw_deg"])))).tolist())
    _ur._align_options(arm, spec)
    frame.attach_body(arm.body("base"), P, "")
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    spec.option.impratio = float(cfg["impratio"])
    spec.add_sensor(name=P + cfg["force_sensor"], type=mujoco.mjtSensor.mjSENS_FORCE,
                    objtype=mujoco.mjtObj.mjOBJ_SITE, objname=P + cfg["ft_site"])
    spec.add_sensor(name=P + cfg["torque_sensor"], type=mujoco.mjtSensor.mjSENS_TORQUE,
                    objtype=mujoco.mjtObj.mjOBJ_SITE, objname=P + cfg["ft_site"])
    spd = [math.radians(v) for v in rob["speed_limits_deg_s"]]
    spec.add_numeric(name=P + "joint_speed_limit_rad_s", data=spd)
    spec.add_text(name=P + "robot", data=cfg["model"])
    fingers = {s: [P + G + s + "_" + n for n in FINGER_PARTS] for s in SIDES}
    gb = [P + G + "base_mount", P + G + "base"] + fingers["right"] + fingers["left"]
    return {
        "model": cfg["model"], "prefix": P, "gripper_prefix": P + G,
        "joints": [P + j for j in _ur.JOINT_NAMES],
        "actuators": [P + a for a in _ur.ACTUATOR_NAMES],
        "torque_limits_Nm": list(lim),
        "speed_limits_rad_s": spd,
        "base_body": P + "base",
        "arm_bodies": [P + b for b in ("base", "shoulder_link", "upper_arm_link", "forearm_link", "wrist_1_link",
                                       "wrist_2_link", "wrist_3_link")],
        "flange_body": P + "wrist_3_link",
        "tool_body": P + G + "base_mount",
        "gripper_bodies": gb + [P + wc["mount"]],
        "finger_bodies": fingers,
        "pad_bodies": {"right": P + G + "right_follower", "left": P + G + "left_follower"},
        "pad_geoms": [P + G + n for n in PAD_GEOMS],
        "gripper_joints": [P + G + n for n in GRIPPER_JOINTS],
        "driver_joints": [P + G + "right_driver_joint", P + G + "left_driver_joint"],
        "gripper_actuator": P + G + "fingers_actuator",
        "tcp_site": P + cfg["tcp_name"],
        "ft_site": P + cfg["ft_site"],
        "force_sensor": P + cfg["force_sensor"],
        "torque_sensor": P + cfg["torque_sensor"],
        "wrist_camera_body": P + wc["mount"],
        "wrist_camera_site": P + wc["site"],
        "wrist_camera": (P + wc["camera"]) if wc.get("camera") else None,
        "wrist_camera_housing": P + "wrist_cam_housing",
        "d6_m": float(rob["d6_m"]),
        "tcp_from_flange_m": mount_z + base_z + float(TCP_IN_BASE[2]),
        "gripper": "wide",
    }


def set_pad_solref(model, solref, prefix="ur/"):
    """rewrite the four pad geoms' solref on a compiled model (a variable-step scheme keeps it at 2 steps)"""
    for n in PAD_GEOMS:
        g = model.geom(prefix + "g/" + n).id
        model.geom_solref[g, :] = [float(solref[0]), float(solref[1])]


def physics_mismatch(model, prefix="ur/", tick_s=0.01):
    """the differences between a compiled cell's physics and the physics the tables were measured under
    ({key: (model, tables)}; empty when they agree)"""
    import mujoco
    ph = TABLE_PHYSICS or {}
    out = {}
    pad = model.geom(prefix + "g/" + PAD_GEOMS[0]).id
    have = {"timestep_s": float(model.opt.timestep), "tick_steps": int(round(tick_s / model.opt.timestep)),
            "pad_solref": [float(v) for v in model.geom_solref[pad]],
            "integrator": "implicitfast" if model.opt.integrator == mujoco.mjtIntegrator.mjINT_IMPLICITFAST else
            int(model.opt.integrator),
            "cone": "elliptic" if model.opt.cone == mujoco.mjtCone.mjCONE_ELLIPTIC else int(model.opt.cone),
            "impratio": float(model.opt.impratio), "noslip_iterations": int(model.opt.noslip_iterations),
            "gravity_m_s2": float(-model.opt.gravity[2])}
    for k, v in have.items():
        if k not in ph:
            continue
        w = ph[k]
        if isinstance(v, (int, float)) and isinstance(w, (int, float)):
            if abs(float(v) - float(w)) > 1e-12 * max(1.0, abs(float(w))):
                out[k] = (v, w)
        elif isinstance(v, list):
            if len(v) != len(w) or any(abs(a - b) > 1e-12 for a, b in zip(v, w)):
                out[k] = (v, w)
        elif v != w:
            out[k] = (v, w)
    return out


# ------------------------------------------------------------------------------------------------ pad geometry
class PadGeometry(object):
    """The faces of one attached wide gripper: gap, face points, pad-midpoint frame, pad forces on a geom set."""

    def __init__(self, model, prefix="ur/", gripper_prefix=None):
        G = gripper_prefix or (prefix + "g/")
        self.m = model
        self.rf = model.body(G + "right_follower").id
        self.lf = model.body(G + "left_follower").id
        self.rp, self.lp = self.rf, self.lf          # the controller's attribute names: the bodies carrying the faces
        try:
            self.tcp = model.site(prefix + "tcp").id
        except KeyError:
            self.tcp = None
        self.base = model.body(G + "base").id
        self.pad_geoms = {"right": [model.geom(G + n).id for n in ("right_pad1", "right_pad2")],
                          "left": [model.geom(G + n).id for n in ("left_pad1", "left_pad2")]}
        self.finger_bodies = {s: [model.body(G + s + "_" + n).id for n in FINGER_PARTS] for s in SIDES}
        self.linkage_qadr = [int(model.jnt_qposadr[model.joint(G + j).id]) for j in LINKAGE]
        self.driver_qadr = [self.linkage_qadr[0], self.linkage_qadr[3]]

    def _closing_axis(self, data):
        if self.tcp is not None:
            return data.site_xmat[self.tcp].reshape(3, 3)[:, 1]
        return data.xmat[self.base].reshape(3, 3)[:, 1]

    def face_point(self, data, side, s):
        """the point of a finger's gripping face at distance s from the face root (towards the fingertip)"""
        b = self.rf if side == "right" else self.lf
        return data.xpos[b] + data.xmat[b].reshape(3, 3) @ np.array(face_local(s))

    def face_normal(self, data, side):
        """the inward normal of a finger's face (world)"""
        b = self.rf if side == "right" else self.lf
        return -(data.xmat[b].reshape(3, 3) @ np.array(N_OUT))

    def gap(self, data, s=PAD_MID_Z):
        """face-to-face gap along the closing axis at face position s (default: the middle)"""
        y = self._closing_axis(data)
        return float((self.face_point(data, "right", s) - self.face_point(data, "left", s)) @ y)

    def pad_frame(self, data):
        """(origin, R): origin midway between the two face middles (the grasp centre), y from the left to the right
        face, z along the faces' mean length axis (towards the tips)"""
        cr, cl = self.face_point(data, "right", PAD_MID_Z), self.face_point(data, "left", PAD_MID_Z)
        o = 0.5 * (cr + cl)
        y = cr - cl
        y /= np.linalg.norm(y)
        t = np.array([0.0, CW, -SW])                 # the face's direction towards the tip, follower frame
        z = data.xmat[self.rf].reshape(3, 3) @ t + data.xmat[self.lf].reshape(3, 3) @ t
        z -= (z @ y) * y
        z /= np.linalg.norm(z)
        return o, np.column_stack([np.cross(y, z), y, z])

    def grasp_centre_in_tcp(self, data):
        """the grasp centre (pad_frame's origin) in the tcp frame (m), from the present state"""
        o, _ = self.pad_frame(data)
        if self.tcp is None:
            p, R = data.xpos[self.base], data.xmat[self.base].reshape(3, 3)
            return R.T @ (o - p) - np.asarray(TCP_IN_BASE, float)
        R = data.site_xmat[self.tcp].reshape(3, 3)
        return R.T @ (o - data.site_xpos[self.tcp])

    def tilt_deg(self, data):
        """mean signed tilt of the two faces from parallel, deg (+: converging towards the tips)"""
        root = self.gap(data, 0.0)
        tip = self.gap(data, PAD_ALONG_M)
        return math.degrees(math.atan2(0.5 * (root - tip), PAD_ALONG_M))

    def forces(self, data, other_geoms):
        """(right, left) normal force (N) of the pad geoms on any geom in `other_geoms`, and the contact count"""
        import mujoco
        other = set(int(g) for g in other_geoms)
        f6 = np.zeros(6)
        out = {"right": 0.0, "left": 0.0}
        n = 0
        rs, ls = set(self.pad_geoms["right"]), set(self.pad_geoms["left"])
        for i in range(data.ncon):
            c = data.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            for a, b in ((g1, g2), (g2, g1)):
                if b in other and (a in rs or a in ls):
                    mujoco.mj_contactForce(self.m, data, i, f6)
                    out["right" if a in rs else "left"] += float(f6[0])
                    n += 1
        return out["right"], out["left"], n


# ------------------------------------------------------------------------------------------------ table lookups
def _interp_by_width(width, name):
    w, v = _BYW[name] if _BYW else (None, None)
    _need(w, "opening table")
    return float(np.interp(min(max(float(width), w[0]), w[-1]), w, v))


def ctrl_of_width(width):
    """servo ctrl (0-255) at which the free fingers settle at `width` (m, face middle); below the smallest gap: 255"""
    return _interp_by_width(width, "ctrl")


def width_of_ctrl(ctrl):
    _need(OPENING_TABLE, "opening table")
    c, w = _BYC
    return float(np.interp(float(ctrl), c, w))


def driver_of_width(width):
    return _interp_by_width(width, "driver_rad")


def pad_centre_z(width):
    """the face middle along the approach, from the tcp (m; negative = towards the flange)"""
    return _interp_by_width(width, "pad_centre_z_m")


def pad_tip_z(width):
    return _interp_by_width(width, "pad_tip_z_m")


def fingertip_z(width):
    """the fingers' farthest collision point along the approach, from the tcp (m)"""
    return _interp_by_width(width, "fingertip_z_m")


def finger_outer_y(width):
    """half the fingers' outer envelope along the closing axis (m)"""
    return _interp_by_width(width, "finger_outer_y_m")


def pad_offset(width):
    """the grasp centre (midway between the two face middles) in the tcp frame at opening `width` (m, 3-vector: x
    across the pads, y closing, z approach), from the gripper model's kinematics at the table's linkage angles"""
    return np.array([_interp_by_width(width, "pad_off_%s_m" % a) for a in "xyz"])


def linkage_qpos(width):
    return {j: _interp_by_width(width, "q_" + j) for j in LINKAGE}


def set_opening(model, data, width, prefix="ur/"):
    """pose the gripper of a model copy at an opening (kinematic: qpos only; call mj_kinematics after)"""
    G = prefix + "g/"
    for j, v in linkage_qpos(width).items():
        data.qpos[model.jnt_qposadr[model.joint(G + j).id]] = v


# ------------------------------------------------------------------------------------------------ force register
def force_ratio(width_m):
    """pad normal force per unit of tendon force at an opening (the measured calibration, face middle)"""
    fm = _need(FORCE_MODEL, "force model")
    return float(np.interp(float(width_m), fm["width_m"], fm["ratio"]))


def clamp_force(force_N):
    F = float(force_N)
    if not math.isfinite(F):
        raise ValueError("non-finite force")
    return min(max(F, FORCE_RANGE_N[0]), FORCE_RANGE_N[1])


def force_cap(force_N, width_m):
    """the servo's force range (tendon units) for a commanded pad force at the measured opening"""
    return clamp_force(force_N) / force_ratio(width_m)


def pad_force(squeeze_m, width_m=0.0579, force_N=DEFAULT_FORCE_N):
    """steady normal force per pad (N) with the force register at force_N when the commanded width is squeeze_m below
    the part's width: the force is the register's value once the squeeze exceeds the servo's saturation squeeze
    (FORCE_MODEL["sat_squeeze_m"] at 100 N, scaled with the force), proportional below it"""
    fm = _need(FORCE_MODEL, "force model")
    F = clamp_force(force_N)
    sat = float(np.interp(width_m, fm["width_m"], fm["sat_squeeze_m"])) * F / DEFAULT_FORCE_N
    if squeeze_m <= 0.0:
        return 0.0
    return float(F * min(1.0, squeeze_m / max(sat, 1e-9)))


def squeeze_for_force(force_N, width_m=0.0579):
    """the squeeze (part width - commanded width, m) at which the register's force is reached"""
    fm = _need(FORCE_MODEL, "force model")
    return float(np.interp(width_m, fm["width_m"], fm["sat_squeeze_m"])) * clamp_force(force_N) / DEFAULT_FORCE_N


class ForceRegister(object):
    """the gripper's force setting on a compiled model: `command(F)` (10-125 N per pad), `apply(data)` rewrites the
    tendon servo's force range for F at the present measured width (call once per control tick)"""

    def __init__(self, model, prefix="ur/", force_N=DEFAULT_FORCE_N, pads=None):
        self.model = model
        self.act = model.actuator(prefix + "g/fingers_actuator").id
        self.pads = pads if pads is not None else PadGeometry(model, prefix)
        self.force_N = clamp_force(force_N)

    def command(self, force_N):
        self.force_N = clamp_force(force_N)
        return self.force_N

    def apply(self, data, width=None):
        w = self.pads.gap(data) if width is None else float(width)
        cap = force_cap(self.force_N, max(0.0, w))
        self.model.actuator_forcerange[self.act] = [-cap, cap]
        return cap


_WC = None


def _wide_controller_class():
    """the arm controller with the force register (built on first use: ur_controller imports this module)"""
    global _WC
    if _WC is not None:
        return _WC
    from .ur_controller import URController

    class WideController(URController):
        """the arm controller (`ur_controller.URController`, this gripper as its gripper) + the force register:
        `command_force(F)` (10-125 N per pad; the pads press with F whenever the commanded width is below the part),
        the servo's force range rewritten every tick from the measured width."""

        def __init__(self, model, data, force_N=DEFAULT_FORCE_N, **kw):
            super(WideController, self).__init__(model, data, **kw)
            self.force_register = ForceRegister(model, self.prefix, force_N, pads=self.pads)
            self.force_register.apply(data)

        @property
        def force_cmd(self):
            return self.force_register.force_N

        def command_force(self, force_N):
            self.force_register.command(force_N)
            self.force_register.apply(self.data)

        def apply_force(self):
            self.force_register.apply(self.data)

        def tick(self):
            super(WideController, self).tick()
            self.force_register.apply(self.data)

    _WC = WideController
    return _WC


def __getattr__(name):
    if name == "WideController":
        return _wide_controller_class()
    raise AttributeError("module %r has no attribute %r" % (__name__, name))
