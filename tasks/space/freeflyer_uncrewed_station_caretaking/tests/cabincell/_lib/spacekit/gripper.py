"""Perching arm of the free-flyer: spring-closed / tendon-opened gripper, rate-limited arm servo, deploy and stow
(in the reference build's names).

Gripper principle (published for the real arm): torsional springs close the two two-joint fingers; a tendon wound by
a small gearmotor opens them, and power-off keeps the grip.  Model: each finger joint has a spring whose reference
angle lies ``preload_span`` rad beyond the closed limit (near-constant torque; at full opening it equals the
published 105.9 / 44.3 N mm for the proximal / distal joints).  The tendon is a MOVING closing-side joint limit: it
can hold a finger open but never push it closed, so a finger stopped by an object keeps only its spring force.

Contact recipe of the fingers (v0.3, reference build v1; in the model itself from reference build v1.2, which says so in
its 'refbuild' text: ``spacekit.refbuild``): the part-level fingers carry separate silicone pad geoms
(names ending ``_pad``); only the pads get the pad friction, the printed phalanges / binding posts / springs get
``hull_friction``, and the two fingers collide with each other, so an empty or caging grip stops where the fingertips
touch (the description's closing limits would let them cross by about 11 mm).  Calibration on the published 6.43 N
handrail pull-off (bar centre 76-88 mm in front of the palm origin = an enveloping cage): pad friction 1.2 gives
6.47-6.60 N, monotonic in the friction (measured in the simulation).  Models
without pad geoms (the test placeholder's box fingers, the superseded v0 build) keep the v0.2 recipe: every finger
geom gets the pad friction and the fingers do not collide (v0 needed 1.8: pass ``gripper=dict(pad_friction=1.8)``).
``grasp_profile(model)`` returns the grasp depths that go with each recipe.

Arm: two joints (tilt = proximal, pan = distal).  The firmware's tilt angle is 0 when deployed aft and 180 deg when
stowed; joint q = tilt - 90 deg.  ``ArmServo`` moves position setpoints at most 0.12 rad/s (description limit);
the joint actuators are position servos with a 1 N m torque limit.

``rig_spec`` edits an MjSpec of the reference robot before compiling: removes the coupled position-actuated
gripper of the geometry model (one actuator + three joint equalities), installs the springs, pad friction and limit
softness, sets the arm servo gains, excludes contacts inside the arm / robot, and adds the inactive grasp weld used
by ``spacekit.grasp_rule``.  Everything is deterministic and local to the model (no global state).
"""
from __future__ import annotations

import math

import mujoco
import numpy as np

__all__ = ["GRIPPER_DEFAULTS", "ARM_DEFAULTS", "ARM_NAMES", "GRASP_PROFILES", "tilt_to_q", "q_to_tilt", "rig_spec",
           "TendonGripper", "ArmServo", "finger_ranges", "set_fingers", "set_arm", "has_pad_geoms", "fingers_collide",
           "touch_opening", "grasp_profile"]

GRIPPER_DEFAULTS = dict(
    spring_torque_open=(0.1059, 0.0443),   # N m at full opening, proximal / distal  [R arm paper]
    preload_span=2.0,                      # rad beyond the closed limit  [G]
    pad_friction=1.2,                      # silicone pads, calibrated on the 6.43 N pull-off: v1 cage 6.47-6.60 N  [V]
                                           # (v0 CoACD fingers needed 1.8 -> 6.37 N)
    hull_friction=0.4,                     # printed phalanges, posts, springs (models with pad geoms only)  [G]
    pad_suffix="_pad",                     # geom-name suffix of the pads of the part-level fingers
    finger_contact=None,                   # None: the fingers collide with each other iff the model has pad geoms
    stroke_s=1.0,                          # full open <-> closed  [G; 50 rpm tendon motor]
    damping=(0.002, 0.001), armature=(1e-4, 5e-5),
    solreflimit=(0.004, 1.0),
)
# Grasp depths (bar centre along the palm x axis; fingers point along -x) and the capture-box entry of each recipe.
# v1 fingers: the cage closes behind a handrail-section bar 76-88 mm in front of the palm origin (envelopment, 6.5 N);
# at 68-72 mm the bar is only pinched by the proximal pads (1-5 N); closer than 66 mm the open fingertips reach the
# wall behind a handrail (bar inner face 43.9 mm off the wall) [V].
GRASP_PROFILES = {
    "pads": dict(grasp_depth=-0.082, close_x=-0.090, perch_depth=-0.082, perch_close_x=-0.088),
    "generic": dict(grasp_depth=-0.070, close_x=-0.085, perch_depth=-0.065, perch_close_x=-0.080),   # v0.2 values
}
ARM_DEFAULTS = dict(kp=20.0, kv=0.6, effort=1.0, vel=0.12, armature=0.005, damping=0.05)   # [G] / URDF limits

ARM_NAMES = dict(
    joints=("arm_tilt", "arm_pan"), actuators=("act_arm_tilt", "act_arm_pan"),
    fingers=("grip_left_proximal", "grip_left_distal", "grip_right_proximal", "grip_right_distal"),
    finger_bodies=("gripper_left_proximal", "gripper_left_distal", "gripper_right_proximal", "gripper_right_distal"),
    palm_body="arm_distal", arm_bodies=("arm_base", "arm_proximal", "arm_distal"), root_body="core",
    gripper_actuator="act_gripper", grasp_weld="grasp_weld", gripper_site="gripper_center",
)
# opening direction of each finger joint: +1 = the open side is the upper limit (left fingers), -1 = lower
OPEN_SIGN = (+1, +1, -1, -1)
TILT_STOWED_DEG, TILT_DEPLOYED_DEG = 180.0, 0.0


def tilt_to_q(tilt_deg: float) -> float:
    """Firmware tilt [deg] (0 deployed aft, 180 stowed) -> proximal joint angle [rad]."""
    return math.radians(tilt_deg - 90.0)


def q_to_tilt(q: float) -> float:
    """Proximal joint angle [rad] -> firmware tilt [deg]."""
    return math.degrees(q) + 90.0


def rig_spec(spec, names: dict | None = None, gripper: dict | None = None, arm: dict | None = None,
             prefix: str = "", add_weld: bool = True, self_collision: bool = False) -> dict:
    """Install the spring / tendon gripper, arm servo gains, intra-robot contact excludes and the grasp weld in an
    MjSpec that contains the reference robot (names from ``ARM_NAMES``, with ``prefix``).  Returns a summary."""
    n = dict(ARM_NAMES, **(names or {}))
    g = dict(GRIPPER_DEFAULTS, **(gripper or {}))
    a = dict(ARM_DEFAULTS, **(arm or {}))
    P = prefix
    out = dict(removed=[], springs={}, pads=[], excludes=0)
    act = spec.actuator(P + n["gripper_actuator"]) if n.get("gripper_actuator") else None
    if act is not None:
        spec.delete(act)
        out["removed"].append(P + n["gripper_actuator"])
    fingers = [P + f for f in n["fingers"]]
    for e in list(spec.equalities):
        if e.type == mujoco.mjtEq.mjEQ_JOINT and (e.name1 in fingers or e.name2 in fingers):
            out["removed"].append(e.name)
            spec.delete(e)
    for k, (jn, sgn) in enumerate(zip(fingers, OPEN_SIGN)):
        j = spec.joint(jn)
        lo, hi = float(j.range[0]), float(j.range[1])
        prox = k % 2 == 0
        tau = g["spring_torque_open"][0 if prox else 1]
        span = (hi - lo) + g["preload_span"]
        kstiff = tau / span
        ref = (lo - g["preload_span"]) if sgn > 0 else (hi + g["preload_span"])
        j.stiffness = [kstiff, 0.0, 0.0]
        j.springref = ref
        j.damping = [g["damping"][0 if prox else 1], 0.0, 0.0]
        j.armature = g["armature"][0 if prox else 1]
        j.solref_limit = list(g["solreflimit"])
        j.limited = mujoco.mjtLimited.mjLIMITED_TRUE
        out["springs"][jn] = dict(stiffness=kstiff, springref=ref, range=(lo, hi))
    fb = set(P + b for b in n["finger_bodies"])
    fgeoms = [geom for geom in spec.geoms if geom.parent is not None and geom.parent.name in fb
              and (geom.contype or geom.conaffinity)]
    pads_present = any(geom.name.endswith(g["pad_suffix"]) for geom in fgeoms)
    # reference build v1.2: the model may carry this recipe itself (spacekit.refbuild); setting the same values
    # again changes nothing, so the recipe is applied in any case (it also carries a caller's friction override)
    from .refbuild import has_correction
    out["build_recipe"] = bool(has_correction(spec, "gripper_pads"))
    out["hulls"] = []
    for geom in fgeoms:
        hull = pads_present and not geom.name.endswith(g["pad_suffix"])
        geom.friction = [g["hull_friction"] if hull else g["pad_friction"], 0.02, 0.002]
        geom.priority = 2
        geom.condim = 4
        (out["hulls"] if hull else out["pads"]).append(geom.name)
    finger_contact = pads_present if g["finger_contact"] is None else bool(g["finger_contact"])
    out["finger_contact"] = finger_contact
    left = {P + b for b in n["finger_bodies"] if "left" in b}
    right = {P + b for b in n["finger_bodies"] if "right" in b}

    def left_right(b1, b2):
        return (b1 in left and b2 in right) or (b1 in right and b2 in left)
    if finger_contact:          # the two fingers stop each other (the base model excludes these pairs)
        for ex in list(spec.excludes):
            if left_right(ex.bodyname1, ex.bodyname2):
                spec.delete(ex)
                out.setdefault("finger_excludes_removed", 0)
                out["finger_excludes_removed"] += 1
    for jn, an in zip(n["joints"], n["actuators"]):
        j = spec.joint(P + jn)
        j.armature = a["armature"]
        j.damping = [a["damping"], 0.0, 0.0]
        ac = spec.actuator(P + an)
        ac.gainprm = [a["kp"]] + [0.0] * 9
        ac.biasprm = [0.0, -a["kp"], -a["kv"]] + [0.0] * 7
        ac.forcerange = [-a["effort"], a["effort"]]
        ac.forcelimited = mujoco.mjtLimited.mjLIMITED_TRUE
    if not self_collision:
        # every moving arm / finger body against every body of the robot (welded children included)
        root = spec.body(P + n["root_body"])
        robot = [root.name] + [b.name for b in root.find_all(mujoco.mjtObj.mjOBJ_BODY)]
        moving = [P + b for b in n["arm_bodies"][1:]] + [P + b for b in n["finger_bodies"]]
        pairs = set()
        for b1 in moving:
            for b2 in robot:
                if b1 != b2 and (b2, b1) not in pairs and not (finger_contact and left_right(b1, b2)):
                    pairs.add((b1, b2))
        for b1, b2 in sorted(pairs):
            ex = spec.add_exclude()
            ex.bodyname1, ex.bodyname2 = b1, b2
            out["excludes"] += 1
    if add_weld:
        eq = spec.add_equality()
        eq.name = P + n["grasp_weld"]
        eq.type = mujoco.mjtEq.mjEQ_WELD
        eq.objtype = mujoco.mjtObj.mjOBJ_BODY
        eq.name1 = P + n["palm_body"]
        eq.name2 = "world"
        eq.active = False
        eq.solref = [0.004, 1.0]
        out["weld"] = eq.name
    return out


def finger_ranges(model, joints) -> list:
    """Model joint ranges (lo, hi) of the four finger joints (left prox, left dist, right prox, right dist)."""
    return [tuple(float(v) for v in model.jnt_range[j]) for j in joints]


class TendonGripper:
    """Tendon-opened, spring-closed gripper.  cmd in [0, 1]: 1 = fully open, 0 = tendon released (the springs close
    on whatever is between the fingers).  The tendon is the closing-side joint limit, moving at the stroke rate.

    joints: finger joint ids (left prox, left dist, right prox, right dist).  ``opening(data)`` = mean opening of the
    proximal joints (encoder-like); ``tendon_load(model, data)`` = limit-constraint torque held by the tendon
    [N m] (proxy of the tendon motor current).  ``latch(data)`` freezes the closing limits at the current angles
    (used while the grasp rule holds an item), ``unlatch()`` returns to tendon control.
    """

    def __init__(self, model, joints, stroke_s: float = GRIPPER_DEFAULTS["stroke_s"], open0: float = 0.0):
        self.m = model
        self.j = [int(x) for x in joints]
        self.R = finger_ranges(model, self.j)
        self.qadr = [int(model.jnt_qposadr[j]) for j in self.j]
        self.dadr = [int(model.jnt_dofadr[j]) for j in self.j]
        self.rate = 1.0 / float(stroke_s)
        self.c = float(open0)
        self.cmd = float(open0)
        self.latched = None
        self._apply()

    def _apply(self) -> None:
        if self.latched is not None:
            return
        c = self.c
        for jid, (lo, hi), sgn in zip(self.j, self.R, OPEN_SIGN):
            if sgn > 0:      # closing toward the lower limit: the tendon raises the lower bound
                self.m.jnt_range[jid] = (lo + c * (hi - lo) - 1e-4 * (c >= 1.0), hi)
            else:            # closing toward the upper limit: the tendon lowers the upper bound
                self.m.jnt_range[jid] = (lo, hi - c * (hi - lo) + 1e-4 * (c >= 1.0))

    def set(self, cmd: float) -> None:
        """Tendon command (1 open, 0 released)."""
        self.cmd = float(np.clip(cmd, 0.0, 1.0))

    def step(self, dt: float) -> None:
        """Move the tendon toward the command at the stroke rate."""
        d = float(np.clip(self.cmd - self.c, -self.rate * dt, self.rate * dt))
        if d != 0.0:
            self.c += d
            self._apply()

    @property
    def moving(self) -> bool:
        """True while the tendon is moving."""
        return abs(self.cmd - self.c) > 1e-9

    def opening(self, data) -> float:
        """Opening fraction of the proximal joints (1 = fully open, 0 = closed limits)."""
        out = []
        for k in (0, 2):
            lo, hi = self.R[k]
            q = float(data.qpos[self.qadr[k]])
            out.append((q - lo) / (hi - lo) if OPEN_SIGN[k] > 0 else (hi - q) / (hi - lo))
        return float(np.mean(out))

    def tendon_load(self, model, data) -> float:
        """Sum of |joint-limit constraint torques| on the finger joints [N m] (the tendon holding the springs)."""
        tot = 0.0
        js = set(self.j)
        for i in range(data.nefc):
            if data.efc_type[i] == mujoco.mjtConstraint.mjCNSTR_LIMIT_JOINT and int(data.efc_id[i]) in js:
                tot += abs(float(data.efc_force[i]))
        return tot

    def latch(self, data) -> None:
        """Freeze each finger's closing limit at its current angle (the fingers stay where they touch)."""
        self.latched = []
        for jid, adr, (lo, hi), sgn in zip(self.j, self.qadr, self.R, OPEN_SIGN):
            q = float(data.qpos[adr])
            self.latched.append(q)
            if sgn > 0:
                self.m.jnt_range[jid] = (min(max(q, lo), hi), hi)
            else:
                self.m.jnt_range[jid] = (lo, max(min(q, hi), lo))

    def unlatch(self) -> None:
        """Back to tendon control."""
        self.latched = None
        self._apply()


class ArmServo:
    """Rate-limited position setpoints for the arm joints (tilt, pan).  ``set(tilt=q, pan=q)`` in joint angles
    [rad]; ``step(data, dt)`` advances the setpoints at most ``vel`` rad/s and writes the actuator controls;
    ``done(data, tol)`` when the setpoints arrived and the joints are within tol."""

    def __init__(self, model, data, actuators, joints, vel: float = ARM_DEFAULTS["vel"]):
        self.a = [int(x) for x in actuators]
        self.jq = [int(model.jnt_qposadr[j]) for j in joints]
        self.vel = float(vel)
        self.sp = np.array([float(data.qpos[i]) for i in self.jq])
        self.goal = self.sp.copy()
        data.ctrl[self.a] = self.sp

    def set(self, tilt: float | None = None, pan: float | None = None) -> None:
        """Joint-angle goals [rad]."""
        if tilt is not None:
            self.goal[0] = float(tilt)
        if pan is not None:
            self.goal[1] = float(pan)

    def step(self, data, dt: float) -> None:
        """Advance the setpoints and write ctrl."""
        self.sp += np.clip(self.goal - self.sp, -self.vel * dt, self.vel * dt)
        data.ctrl[self.a] = self.sp

    def done(self, data, tol: float = 0.02) -> bool:
        """Setpoints at the goals and joints within tol [rad]."""
        return bool(np.all(np.abs(self.goal - self.sp) < 1e-9)
                    and np.all(np.abs(np.array([data.qpos[i] for i in self.jq]) - self.goal) < tol))

    def joint_angles(self, data) -> np.ndarray:
        """(tilt q, pan q) [rad]."""
        return np.array([float(data.qpos[i]) for i in self.jq])


def _finger_geoms(model, joints):
    """(left geoms, right geoms) with contact of the finger bodies of ``joints`` (left prox, left dist, right ...)."""
    bodies = [int(model.jnt_bodyid[j]) for j in joints]
    out = ([], [])
    for k, b in enumerate(bodies):
        out[0 if k < 2 else 1].extend(g for g in range(model.ngeom) if model.geom_bodyid[g] == b
                                      and (model.geom_contype[g] or model.geom_conaffinity[g]))
    return out


def has_pad_geoms(model, names: dict | None = None, suffix: str = GRIPPER_DEFAULTS["pad_suffix"]) -> bool:
    """True if the finger bodies carry separate pad geoms (the part-level fingers of the reference build v1)."""
    n = dict(ARM_NAMES, **(names or {}))
    fb = {mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, b) for b in n["finger_bodies"]} - {-1}
    return any(model.geom_bodyid[g] in fb and (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or "").endswith(suffix)
               for g in range(model.ngeom))


def fingers_collide(model, joints) -> bool:
    """True if a left and a right finger body may collide (no exclude between them, compatible contype bits)."""
    bodies = [int(model.jnt_bodyid[j]) for j in joints]
    excl = set(int(s) for s in model.exclude_signature)
    L, R = _finger_geoms(model, joints)
    for b1 in bodies[:2]:
        for b2 in bodies[2:]:
            lo, hi = min(b1, b2), max(b1, b2)
            if (lo << 16) + hi in excl:
                continue
            for g1 in L:
                for g2 in R:
                    if model.geom_bodyid[g1] == b1 and model.geom_bodyid[g2] == b2 and (
                            (model.geom_contype[g1] & model.geom_conaffinity[g2]) or (model.geom_contype[g2] & model.geom_conaffinity[g1])):
                        return True
    return False


def touch_opening(model, data, joints, margin: float = 2e-4, iters: int = 24) -> float:
    """Smallest coupled opening s in [0, 1] (q = closed limit + s (open limit - closed limit) on every joint) at which
    no left / right finger collision geoms overlap (by bisection, kinematics only; ``data`` qpos is restored)."""
    L, R = _finger_geoms(model, joints)
    rng = [tuple(float(v) for v in model.jnt_range[j]) for j in joints]
    adr = [int(model.jnt_qposadr[j]) for j in joints]
    q0 = data.qpos.copy()
    ft = np.zeros(6)

    def gap(s):
        for a, (lo, hi), sgn in zip(adr, rng, OPEN_SIGN):
            closed, opened = (lo, hi) if sgn > 0 else (hi, lo)
            data.qpos[a] = closed + s * (opened - closed)
        mujoco.mj_kinematics(model, data)
        return min(mujoco.mj_geomDistance(model, data, g1, g2, 0.05, ft) for g1 in L for g2 in R)
    lo_s, hi_s = 0.0, 1.0
    if gap(0.0) >= margin:
        lo_s = hi_s = 0.0
    else:
        for _ in range(iters):
            mid = 0.5 * (lo_s + hi_s)
            if gap(mid) >= margin:
                hi_s = mid
            else:
                lo_s = mid
    data.qpos[:] = q0
    mujoco.mj_kinematics(model, data)
    return hi_s


def grasp_profile(model, names: dict | None = None) -> dict:
    """Grasp depths of the model's finger recipe (``GRASP_PROFILES['pads']`` for part-level fingers, else 'generic')."""
    kind = "pads" if has_pad_geoms(model, names) else "generic"
    return dict(GRASP_PROFILES[kind], kind=kind)


def set_fingers(model, data, joints, open_: bool, opening: float | None = None) -> None:
    """Put the finger joints at their open or closed position in qpos (scene initialisation only).  Closed = the
    closed limits, or, when the two fingers collide with each other, the coupled opening at which their tips touch
    (``touch_opening``; the springs then settle them); ``opening`` overrides (0 closed limits, 1 open)."""
    if not open_ and opening is None and fingers_collide(model, joints):
        opening = touch_opening(model, data, joints)
    for jid, sgn in zip(joints, OPEN_SIGN):
        lo, hi = (float(v) for v in model.jnt_range[jid])
        adr = int(model.jnt_qposadr[jid])
        if open_:
            data.qpos[adr] = hi if sgn > 0 else lo
        else:
            s = 0.0 if opening is None else float(opening)
            closed, opened = (lo, hi) if sgn > 0 else (hi, lo)
            data.qpos[adr] = closed + s * (opened - closed)


def set_arm(model, data, joints, tilt_deg: float, pan_rad: float = 0.0) -> None:
    """Put the arm joints at a firmware tilt / pan in qpos (scene initialisation only)."""
    data.qpos[int(model.jnt_qposadr[joints[0]])] = tilt_to_q(tilt_deg)
    data.qpos[int(model.jnt_qposadr[joints[1]])] = pan_rad
