"""The Unitree G1 humanoid of the cell (MuJoCo Menagerie `unitree_g1/g1_with_hands.xml`, BSD-3-Clause), modified for
the house-cleaning cell, and its kinematics helpers (stand-height leg posture, 6-DoF damped-least-squares arm IK).

The model, its modifications and the kinematics helpers are those of furniture_assembly_from_drawings's cell (the same
G1, the same locomotion abstraction, grasp sites and head camera); only the collision bits differ (one robot here).

Modifications (the cell manual states them):
  * locomotion abstraction: the floating base is replaced by four base joints on the pelvis (slide x, slide y,
    hinge yaw, slide z), driven by force-limited position servos: the robot's own "locomotion controller" carries the
    pelvis; the legs are posed from the stand height and never touch the floor;
  * every robot body is gravity-compensated (the balance and gravity feed-forward of the real controller); payloads
    are not: a held part loads the arm servos and the base;
  * primitive collision geometry: Menagerie's mjlab capsules for the body and oriented boxes around the Dex3-1 hand
    links replace the collision meshes; no self-collision inside a robot;
  * stiffer upper-body servos (the arms hold payloads); torque limits unchanged;
  * a site `<side>_grasp` between the thumb and the two fingers of each hand (the grasp frame: x = approach,
    y = pinch axis from the thumb towards the fingers) and the head RGB-D camera `head_cam` at the D435 mount of the
    Unitree URDF (torso_link).
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field

import mujoco
import numpy as np

G1_DIR = os.environ.get("ARB_G1_DIR", "/opt/arb-assets/unitree_g1")

ARM_JOINTS = ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw")
HAND_JOINTS = ("thumb_0", "thumb_1", "thumb_2", "middle_0", "middle_1", "index_0", "index_1")
LEG_JOINTS = ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")
WAIST_JOINTS = ("waist_yaw", "waist_roll", "waist_pitch")
BASE_JOINTS = ("base_x", "base_y", "base_yaw", "base_z")
SIDES = ("left", "right")

# arm posture of Menagerie's "stand" keyframe, elbows bent (left; the right arm mirrors roll and yaw)
ARM_HOME = {"left": (0.30, 0.25, 0.0, 1.10, 0.0, 0.0, 0.0), "right": (0.30, -0.25, 0.0, 1.10, 0.0, 0.0, 0.0)}
# Dex3-1 hand synergies: open and closed (pinch) finger targets, left hand; the right hand mirrors the signs
HAND_OPEN = {"left": (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)}
HAND_CLOSED = {"left": (0.0, 0.55, 1.10, -0.85, -0.95, -0.85, -0.95)}
HAND_OPEN["right"] = tuple(-v for v in HAND_OPEN["left"])
HAND_CLOSED["right"] = tuple(-v if i != 0 else v for i, v in enumerate(HAND_CLOSED["left"]))

PELVIS_HEIGHT = 0.793              # Menagerie's standing pelvis height
STAND_RANGE = (0.48, 0.80)         # stand height (pelvis above the floor) the locomotion controller accepts
# the Unitree URDF's d435_joint in torso_link (g1_29dof_with_hand_rev_1_0.urdf): position and downward pitch
D435_POS = (0.0576235, 0.01753, 0.42987)
D435_PITCH = 0.8307767239493009
HEAD_CAM = {"fovy": 58.0, "size": (640, 480)}
GRASP_SITE_POS = {"left": (0.150, -0.046, 0.0), "right": (0.150, 0.046, 0.0)}   # wrist_yaw_link frame

BASE_KP = {"base_x": 6000.0, "base_y": 6000.0, "base_yaw": 1500.0, "base_z": 20000.0}
BASE_FORCE = {"base_x": 300.0, "base_y": 300.0, "base_yaw": 150.0, "base_z": 900.0}
ARM_KP = {"shoulder_pitch": 1200.0, "shoulder_roll": 1200.0, "shoulder_yaw": 900.0, "elbow": 1000.0,
          "wrist_roll": 300.0, "wrist_pitch": 200.0, "wrist_yaw": 200.0}
WAIST_KP = 2500.0
LEG_KP = 800.0
HAND_KP = 40.0

# collision bits: 1 = the static room and its fixtures, 2 = the robot, 4 = the loose objects and tools, 8 = the floor,
# 16 = an object while a hand holds it (it touches the room, the other objects and the floor, never the robot)
WORLD, ROBOT, OBJ, FLOOR, HELD = 1, 2, 4, 8, 16
LEG_WORDS = ("hip", "knee", "ankle", "pelvis")


def _ref_geometry(g1_dir: str):
    """Primitive collision geometry from Menagerie's mjlab G1 (capsules) plus boxes around the Dex3-1 hand meshes."""
    mh = mujoco.MjModel.from_xml_path(os.path.join(g1_dir, "g1_with_hands.xml"))
    ml = mujoco.MjModel.from_xml_path(os.path.join(g1_dir, "g1_mjlab.xml"))
    caps = []
    for g in range(ml.ngeom):
        if ml.geom_contype[g] == 0 and ml.geom_conaffinity[g] == 0:
            continue
        gname = mujoco.mj_id2name(ml, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
        if "hand_collision" in gname:
            continue
        caps.append((mujoco.mj_id2name(ml, mujoco.mjtObj.mjOBJ_BODY, ml.geom_bodyid[g]), int(ml.geom_type[g]),
                     ml.geom_size[g].copy(), ml.geom_pos[g].copy(), ml.geom_quat[g].copy()))
    boxes = []
    for g in range(mh.ngeom):
        if mh.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH or (mh.geom_contype[g] == 0 and mh.geom_conaffinity[g] == 0):
            continue
        body = mujoco.mj_id2name(mh, mujoco.mjtObj.mjOBJ_BODY, mh.geom_bodyid[g])
        mesh = mujoco.mj_id2name(mh, mujoco.mjtObj.mjOBJ_MESH, mh.geom_dataid[g])
        if "hand" not in mesh:
            continue
        c, h = mh.geom_aabb[g, :3], mh.geom_aabb[g, 3:]
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, mh.geom_quat[g])
        pos = mh.geom_pos[g] + R.reshape(3, 3) @ c
        boxes.append((body, mesh, pos, mh.geom_quat[g].copy(), h.copy()))
    return caps, boxes


def camera_quat(pitch: float) -> np.ndarray:
    """MuJoCo camera frame (looks along -z, +y up) for a camera whose optical axis is the body's +x pitched down."""
    s, c = math.sin(pitch), math.cos(pitch)
    x_c = np.array([0.0, -1.0, 0.0])
    y_c = np.array([s, 0.0, c])
    z_c = np.cross(x_c, y_c)
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.stack([x_c, y_c, z_c], axis=1).reshape(-1))
    return q


def g1_spec(name: str = "a", g1_dir: str = G1_DIR) -> mujoco.MjSpec:
    """The cell's G1 (prefix `name`), pelvis on the base joints, ready to attach to the room."""
    caps, boxes = _ref_geometry(g1_dir)
    s = mujoco.MjSpec.from_file(os.path.join(g1_dir, "g1_with_hands.xml"))
    for k in list(s.keys):
        s.delete(k)
    for lt in list(s.lights):
        s.delete(lt)
    pelvis = s.body("pelvis")
    pelvis.pos = [0.0, 0.0, PELVIS_HEIGHT]
    for j in list(pelvis.joints):
        s.delete(j)
    types = {"base_x": mujoco.mjtJoint.mjJNT_SLIDE, "base_y": mujoco.mjtJoint.mjJNT_SLIDE,
             "base_yaw": mujoco.mjtJoint.mjJNT_HINGE, "base_z": mujoco.mjtJoint.mjJNT_SLIDE}
    axes = {"base_x": [1, 0, 0], "base_y": [0, 1, 0], "base_yaw": [0, 0, 1], "base_z": [0, 0, 1]}
    for n in BASE_JOINTS:
        j = pelvis.add_joint(name=n, type=types[n], axis=axes[n])
        j.armature = 0.0
        j.frictionloss = 0.0
        j.damping = [0.0, 0.0, 0.0] if hasattr(j, "damping") else 0.0
        if n == "base_z":
            j.range = [STAND_RANGE[0] - PELVIS_HEIGHT - 0.02, STAND_RANGE[1] - PELVIS_HEIGHT + 0.02]
            j.limited = mujoco.mjtLimited.mjLIMITED_TRUE
    # actuators: replace the stock ones (kp 500 everywhere) with the cell's servos
    stock = {a.name: a for a in s.actuators}
    for a in list(stock.values()):
        s.delete(a)
    for n in BASE_JOINTS:
        a = s.add_actuator(name=n, target=n, trntype=mujoco.mjtTrn.mjTRN_JOINT)
        a.set_to_position(kp=BASE_KP[n], dampratio=1.0)
        a.forcelimited = mujoco.mjtLimited.mjLIMITED_TRUE
        a.forcerange = [-BASE_FORCE[n], BASE_FORCE[n]]

    def add_servo(joint: str, kp: float):
        jt = s.joint(joint)
        a = s.add_actuator(name=joint, target=joint, trntype=mujoco.mjtTrn.mjTRN_JOINT)
        a.set_to_position(kp=kp, dampratio=1.0)
        lim = jt.actfrcrange
        if lim is not None and lim[1] > lim[0]:
            a.forcelimited = mujoco.mjtLimited.mjLIMITED_TRUE
            a.forcerange = [float(lim[0]), float(lim[1])]
        a.ctrllimited = mujoco.mjtLimited.mjLIMITED_TRUE
        a.ctrlrange = [float(jt.range[0]), float(jt.range[1])]

    for side in SIDES:
        for w in LEG_JOINTS:
            add_servo(f"{side}_{w}_joint", LEG_KP)
    for w in WAIST_JOINTS:
        add_servo(f"{w}_joint", WAIST_KP)
    for side in SIDES:
        for w in ARM_JOINTS:
            add_servo(f"{side}_{w}_joint", ARM_KP[w])
        for w in HAND_JOINTS:
            add_servo(f"{side}_hand_{w}_joint", HAND_KP)
    # gravity compensation of every robot body
    for b in s.bodies:
        if b.name and b.name != "world":
            b.gravcomp = 1.0
    # primitive collision geometry
    for g in list(s.geoms):
        if g.type == mujoco.mjtGeom.mjGEOM_MESH and (g.contype or g.conaffinity):
            s.delete(g)
    for body, t, size, pos, quat in caps:
        s.body(body).add_geom(type=t, size=size, pos=pos, quat=quat, group=3, rgba=[0.2, 0.6, 0.2, 0.3])
    for body, mesh, pos, quat, half in boxes:
        s.body(body).add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=half, pos=pos, quat=quat, group=3,
                              rgba=[0.2, 0.6, 0.2, 0.3], name=f"{mesh}_box")
    hand_bodies = {f"{sd}_wrist_yaw_link" for sd in SIDES} | {f"{sd}_hand_{w}_link" for sd in SIDES for w in HAND_JOINTS}
    for g in s.geoms:
        if g.contype == 0 and g.conaffinity == 0:
            continue
        if g.parent.name in hand_bodies:
            # the hands carry no contact: grasping is the cell's rule (physics.Grips), not finger contact
            g.contype = 0
            g.conaffinity = 0
            continue
        g.contype = ROBOT
        g.conaffinity = WORLD | OBJ           # never the floor (FLOOR), never itself
        g.friction = [0.9, 0.01, 0.0001]
        g.condim = 3
    # grasp sites and the head camera
    for side in SIDES:
        b = s.body(f"{side}_wrist_yaw_link")
        b.add_site(name=f"{side}_grasp", pos=list(GRASP_SITE_POS[side]), size=[0.008, 0, 0], group=4)
    s.body("torso_link").add_camera(name="head_cam", pos=list(D435_POS), quat=list(camera_quat(D435_PITCH)),
                                    fovy=HEAD_CAM["fovy"])
    s.body("torso_link").add_site(name="head_cam_site", pos=list(D435_POS), quat=list(camera_quat(D435_PITCH)),
                                  size=[0.005, 0, 0], group=4)
    return s


@dataclass
class RobotIndex:
    """Addresses of one attached robot (prefix "a/" or "b/") in the compiled cell model."""

    name: str
    base_q: np.ndarray = field(default=None)
    base_v: np.ndarray = field(default=None)
    base_act: np.ndarray = field(default=None)
    leg_q: dict = field(default_factory=dict)
    leg_act: dict = field(default_factory=dict)
    waist_q: np.ndarray = field(default=None)
    waist_v: np.ndarray = field(default=None)
    waist_act: np.ndarray = field(default=None)
    arm_q: dict = field(default_factory=dict)
    arm_v: dict = field(default_factory=dict)
    arm_act: dict = field(default_factory=dict)
    arm_lo: dict = field(default_factory=dict)
    arm_hi: dict = field(default_factory=dict)
    hand_q: dict = field(default_factory=dict)
    hand_act: dict = field(default_factory=dict)
    grasp_site: dict = field(default_factory=dict)
    hand_body: dict = field(default_factory=dict)
    hand_bodies: dict = field(default_factory=dict)   # side -> body ids of the wrist_yaw link and the hand links
    pelvis: int = -1
    torso: int = -1
    camera: int = -1
    body_ids: np.ndarray = field(default=None)
    geom_ids: np.ndarray = field(default=None)


def _id(m, kind, n):
    i = mujoco.mj_name2id(m, kind, n)
    if i < 0:
        raise KeyError(n)
    return i


def robot_index(m: mujoco.MjModel, name: str) -> RobotIndex:
    p = name + "/"
    J, A, B = mujoco.mjtObj.mjOBJ_JOINT, mujoco.mjtObj.mjOBJ_ACTUATOR, mujoco.mjtObj.mjOBJ_BODY
    r = RobotIndex(name=name)
    bj = [_id(m, J, p + n) for n in BASE_JOINTS]
    r.base_q = np.array([m.jnt_qposadr[j] for j in bj])
    r.base_v = np.array([m.jnt_dofadr[j] for j in bj])
    r.base_act = np.array([_id(m, A, p + n) for n in BASE_JOINTS])
    for side in SIDES:
        js = [_id(m, J, f"{p}{side}_{w}_joint") for w in LEG_JOINTS]
        r.leg_q[side] = np.array([m.jnt_qposadr[j] for j in js])
        r.leg_act[side] = np.array([_id(m, A, f"{p}{side}_{w}_joint") for w in LEG_JOINTS])
        js = [_id(m, J, f"{p}{side}_{w}_joint") for w in ARM_JOINTS]
        r.arm_q[side] = np.array([m.jnt_qposadr[j] for j in js])
        r.arm_v[side] = np.array([m.jnt_dofadr[j] for j in js])
        r.arm_lo[side] = m.jnt_range[js, 0].copy()
        r.arm_hi[side] = m.jnt_range[js, 1].copy()
        r.arm_act[side] = np.array([_id(m, A, f"{p}{side}_{w}_joint") for w in ARM_JOINTS])
        js = [_id(m, J, f"{p}{side}_hand_{w}_joint") for w in HAND_JOINTS]
        r.hand_q[side] = np.array([m.jnt_qposadr[j] for j in js])
        r.hand_act[side] = np.array([_id(m, A, f"{p}{side}_hand_{w}_joint") for w in HAND_JOINTS])
        r.grasp_site[side] = _id(m, mujoco.mjtObj.mjOBJ_SITE, f"{p}{side}_grasp")
        r.hand_body[side] = _id(m, B, f"{p}{side}_wrist_yaw_link")
        hb = [r.hand_body[side]] + [_id(m, B, f"{p}{side}_hand_{w}_link") for w in
                                     ("thumb_0", "thumb_1", "thumb_2", "middle_0", "middle_1", "index_0", "index_1")]
        r.hand_bodies[side] = np.array(hb)
    js = [_id(m, J, f"{p}{w}_joint") for w in WAIST_JOINTS]
    r.waist_q = np.array([m.jnt_qposadr[j] for j in js])
    r.waist_v = np.array([m.jnt_dofadr[j] for j in js])
    r.waist_act = np.array([_id(m, A, f"{p}{w}_joint") for w in WAIST_JOINTS])
    r.pelvis = _id(m, B, p + "pelvis")
    r.torso = _id(m, B, p + "torso_link")
    r.camera = _id(m, mujoco.mjtObj.mjOBJ_CAMERA, p + "head_cam")
    bodies = [b for b in range(m.nbody) if (mujoco.mj_id2name(m, B, b) or "").startswith(p)]
    r.body_ids = np.array(bodies)
    r.geom_ids = np.array([g for g in range(m.ngeom) if m.geom_bodyid[g] in set(bodies)])
    return r


# ------------------------------------------------------------------------------------------------ leg posture

class LegPosture:
    """Leg joint angles for a stand height: a symmetric squat (hip pitch -a, knee 2a, ankle pitch -a keeps the foot
    flat and under the hip); the pelvis height of each squat angle is tabulated once by forward kinematics."""

    def __init__(self, m: mujoco.MjModel, r: RobotIndex):
        d = mujoco.MjData(m)
        sole = _id(m, mujoco.mjtObj.mjOBJ_BODY, r.name + "/left_ankle_roll_link")
        self.alphas = np.linspace(0.0, 1.25, 64)
        heights = []
        for a in self.alphas:
            d.qpos[:] = m.qpos0
            d.qpos[r.leg_q["left"][[0, 3, 4]]] = (-a, 2 * a, -a)
            mujoco.mj_kinematics(m, d)
            heights.append(d.xpos[r.pelvis][2] - d.xpos[sole][2])
        heights = np.array(heights)
        self.sole_drop = float(heights[0] - PELVIS_HEIGHT)          # sole below the floor at the stock height
        self.heights = heights - self.sole_drop                     # pelvis above the floor, feet on the floor

    def alpha(self, pelvis_height: float) -> float:
        h = float(np.clip(pelvis_height, self.heights[-1], self.heights[0]))
        return float(np.interp(-h, -self.heights, self.alphas))

    def targets(self, pelvis_height: float) -> np.ndarray:
        """Six joint targets (hip pitch, roll, yaw, knee, ankle pitch, roll) of either leg."""
        a = self.alpha(pelvis_height)
        return np.array([-a, 0.0, 0.0, 2 * a, -a, 0.0])


# ------------------------------------------------------------------------------------------------ arm IK

def _quat_err(q_target: np.ndarray, q_cur: np.ndarray) -> np.ndarray:
    """Rotation vector (world) taking q_cur to q_target."""
    qc = np.array([q_cur[0], -q_cur[1], -q_cur[2], -q_cur[3]])
    qe = np.zeros(4)
    mujoco.mju_mulQuat(qe, q_target, qc)
    if qe[0] < 0:
        qe = -qe
    v = np.zeros(3)
    mujoco.mju_quat2Vel(v, qe, 1.0)
    return v


class ArmIK:
    """6-DoF damped least squares for one arm's grasp site, base / waist / legs taken from the given state."""

    def __init__(self, m: mujoco.MjModel, r: RobotIndex):
        self.m, self.r = m, r
        self.dk = mujoco.MjData(m)
        self.jp = np.zeros((3, m.nv))
        self.jr = np.zeros((3, m.nv))

    def solve(self, qpos: np.ndarray, side: str, pos, quat, q_init, iters: int = 30, damping: float = 0.05,
              w_rot: float = 0.35, w_home: float = 0.02):
        """Arm joint angles (7) putting `side`'s grasp site at pos (world) with orientation quat (world, wxyz).
        Returns (q, pos_err_m, rot_err_rad)."""
        m, r, dk = self.m, self.r, self.dk
        dk.qpos[:] = qpos
        q = np.array(q_init, dtype=float)
        lo, hi = r.arm_lo[side] + 1e-3, r.arm_hi[side] - 1e-3
        home = np.array(ARM_HOME[side])
        sid = r.grasp_site[side]
        pos = np.asarray(pos, dtype=float)
        quat = None if quat is None else np.asarray(quat, dtype=float) / np.linalg.norm(quat)
        ep, er = np.zeros(3), np.zeros(3)
        for it in range(iters + 1):
            dk.qpos[r.arm_q[side]] = q
            mujoco.mj_kinematics(m, dk)
            ep = pos - dk.site_xpos[sid]
            if quat is not None:
                cur = np.zeros(4)
                mujoco.mju_mat2Quat(cur, dk.site_xmat[sid])
                er = _quat_err(quat, cur)
            if it == iters or (np.linalg.norm(ep) < 2e-4 and np.linalg.norm(er) < 2e-3):
                break
            mujoco.mj_comPos(m, dk)
            mujoco.mj_jacSite(m, dk, self.jp, self.jr, sid)
            Jp = self.jp[:, r.arm_v[side]]
            if quat is not None:
                J = np.vstack([Jp, w_rot * self.jr[:, r.arm_v[side]]])
                e = np.concatenate([ep, w_rot * er])
            else:
                J, e = Jp, ep
            JJt = J @ J.T + (damping ** 2) * np.eye(J.shape[0])
            Jpinv = J.T @ np.linalg.solve(JJt, np.eye(J.shape[0]))
            dq = Jpinv @ e + (np.eye(7) - Jpinv @ J) @ (w_home * (home - q))
            step = np.max(np.abs(dq))
            if step > 0.35:
                dq *= 0.35 / step
            q = np.clip(q + dq, lo, hi)
        return q, float(np.linalg.norm(ep)), float(np.linalg.norm(er))

    SEEDS = {"left": ((0.30, 0.25, 0.0, 1.10, 0.0, 0.0, 0.0), (-0.50, 0.30, 0.50, 0.80, 1.00, 0.0, 0.0),
                      (-1.00, 0.20, -0.50, 0.50, -1.00, 0.50, 0.0), (-1.40, 0.10, 0.0, 0.20, 0.0, 0.0, 0.0),
                      (-0.80, 0.60, 0.80, 1.20, 1.50, -0.50, 0.0), (0.20, 0.10, -0.80, 1.40, -1.20, 0.80, 0.30))}
    SEEDS["right"] = tuple(tuple(v * (-1 if i in (1, 2, 4, 6) else 1) for i, v in enumerate(sd)) for sd in SEEDS["left"])

    def solve_global(self, qpos, side, pos, quat, q_current, iters=60):
        """Best of several seeded solves (the current joint target first): (q, pos_err, rot_err)."""
        best = None
        for seed in (tuple(q_current),) + self.SEEDS[side]:
            q, ep, er = self.solve(qpos, side, pos, quat, np.array(seed, float), iters=iters)
            score = ep + 0.25 * er
            if best is None or score < best[0] - 1e-6:
                best = (score, q, ep, er)
            if ep < 5e-4 and er < 5e-3:
                break
        return best[1], best[2], best[3]
