"""Privileged xArm 7 motion planning. Live motion only writes actuator controls.

The planner can inspect ground truth. It cannot move a live free body or fasten a
mate. All live integration is through ``cell.tick()``, including gripper waits.
The fixed grasp frame is at z=158 mm in xarm_gripper_base_link; its +z points
towards the fingers and its y axis is the jaw closing direction.

Finger links. The stock gripper collides only its four pad boxes and the
knuckles; the two finger link meshes are visual geoms (contype 0). Each finger's
plate is 1 mm wider than its pads on both sides, ends 1.5 mm beyond their tip
and has a back and a sloped link towards the knuckle, none of which the physics
or a copy of the live model can see. The planning copy (``Motion.pm``, never the
live model) therefore collides each finger link mesh as its convex hull, on its
own contype bit: the hulls never touch the robot itself or the measured held
part (a rigid, already-accepted grasp), and against a part whose pad contact is
permitted only a contact on the pad face counts as a grip (see collision()).
"""
from __future__ import annotations

import math
import time
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation


HOME = np.array([0., -.247, 0., .909, 0., 1.15644, 0.])
DOWN = np.diag([1., -1., -1.])
# RRT-Connect budget per nominal second of `timeout` (12 s -> 300 extend attempts,
# about 2-3x the search a loaded 64-core host managed in 12 wall seconds).
PLAN_EXTENDS_PER_S = 25
PLAN_HANG_GUARD_S = 900.
# Finger link bodies of the stock gripper and the planning copy's collision bit
# for their mesh hulls (every non-robot collidable geom gets this bit too).
FINGER_LINKS = ("left_finger", "right_finger")
FINGER_BIT = 2


class PlanningError(RuntimeError):
    pass


def _quat(R):
    q = Rotation.from_matrix(R).as_quat()
    return np.r_[q[3], q[:3]]


def body_subtree(model, body):
    """The selected physical assembly only, including its passive children."""
    root=model.body(body).id if isinstance(body,str) else int(body)
    result={root}
    for child in range(root+1,model.nbody):
        if int(model.body_parentid[child]) in result:
            result.add(child)
    return frozenset(result)


def axis_aligned_rotation(current, target):
    """Align local Z while retaining the measured, unconstrained axial spin."""
    a,b=np.asarray(current)[:,2],np.asarray(target)[:,2]
    cross=np.cross(a,b);sine=float(np.linalg.norm(cross));cosine=float(np.clip(a@b,-1,1))
    angle=math.atan2(sine,cosine)
    if sine>1e-10:
        axis=cross/sine
    elif cosine<0:
        axis=np.asarray(current)[:,0]
    else:
        return np.asarray(current).copy(),0.
    return Rotation.from_rotvec(axis*angle).as_matrix() @ current,angle


def _finger_link_model(cell):
    """Planning copy of the live model whose finger link meshes are colliders.

    MuJoCo builds a mesh's convex hull and the body's collision BVH only for
    geoms that are collidable at compile time, so enabling contype on the
    visual finger geom of a model copy has no effect. Recompile the cell's own
    XML with exactly those two mesh geoms collidable: every body, joint and
    geom keeps its index and kinematic parameters (checked below). The hulls
    get their own contype bit (FINGER_BIT, conaffinity 0) and every non-robot
    collidable geom gains it, so a hull meets parts and the bench but never the
    robot links, the other hull or its own pads. Returns (model, hull geoms).
    """
    m = cell.m
    root = ET.fromstring(cell.xml)
    for body in root.iter("body"):
        if body.get("name") in FINGER_LINKS:
            for geom in body.findall("geom"):
                if geom.get("mesh") is not None:
                    geom.set("contype", "1")
                    geom.set("conaffinity", "1")
    pm = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    same = all(getattr(pm, n) == getattr(m, n) for n in ("nq", "nv", "nu", "nbody", "njnt", "ngeom", "neq"))
    for n in ("geom_bodyid", "geom_type", "geom_size", "geom_pos", "geom_quat", "body_parentid", "body_pos",
              "body_quat", "jnt_type", "jnt_qposadr", "jnt_pos", "jnt_axis", "jnt_range", "qpos0"):
        same = same and np.array_equal(getattr(pm, n), getattr(m, n))
    if not same:
        raise PlanningError("finger-link planning model differs from the live model")
    # Post-compile edits of the live model (couplings, development teleports).
    for n in ("eq_obj1id", "eq_obj2id", "eq_data", "eq_solref", "eq_solimp"):
        getattr(pm, n)[:] = getattr(m, n)
    fingers = {m.body(n).id for n in FINGER_LINKS}
    hulls = frozenset(g for g in range(m.ngeom)
                      if int(m.geom_bodyid[g]) in fingers and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH
                      and not (m.geom_contype[g] or m.geom_conaffinity[g]))
    robot = body_subtree(m, m.body("link_base").id)
    for g in range(pm.ngeom):
        if g in hulls:
            pm.geom_contype[g], pm.geom_conaffinity[g] = FINGER_BIT, 0
        elif int(pm.geom_bodyid[g]) not in robot and (pm.geom_contype[g] or pm.geom_conaffinity[g]):
            pm.geom_contype[g] |= FINGER_BIT
            pm.geom_conaffinity[g] |= FINGER_BIT
    for b in range(pm.nbody):
        members = np.flatnonzero(pm.geom_bodyid == b)
        pm.body_contype[b] = int(np.bitwise_or.reduce(pm.geom_contype[members])) if len(members) else 0
        pm.body_conaffinity[b] = int(np.bitwise_or.reduce(pm.geom_conaffinity[members])) if len(members) else 0
    return pm, hulls


class Motion:
    def __init__(self, cell, *, grasp_offset=(0., 0., .158), seed=512,
                 arm_clearance=.004, finger_clearance=.001, held_clearance=.002,
                 finger_body_clearance=None, finger_body_tolerance=.0001, pad_face_angle_deg=35.):
        self.cell, self.m, self.d = cell, cell.m, cell.d
        self.pm, self.finger_hulls = _finger_link_model(cell)
        self.pd = mujoco.MjData(self.pm)
        self.rng = np.random.default_rng(seed)
        self.jids = np.array([self.m.joint(f"joint{i}").id for i in range(1, 8)])
        self.qadr = self.m.jnt_qposadr[self.jids]
        self.vadr = self.m.jnt_dofadr[self.jids]
        self.aids = np.array([self.m.actuator(f"act{i}").id for i in range(1, 8)])
        self.ga = self.m.actuator("gripper").id
        self.hand = self.m.body("xarm_gripper_base_link").id
        self.base = self.m.body("link_base").id
        self.offset = np.asarray(grasp_offset, float)
        self.limits = self.m.jnt_range[self.jids].copy()
        self.robot_bodies = {self.base}
        for b in range(self.base+1, self.m.nbody):
            if self.m.body_parentid[b] in self.robot_bodies:
                self.robot_bodies.add(b)
        self.pad_geoms = {self.m.geom(n).id for n in (
            "left_finger_pad_1", "left_finger_pad_2", "right_finger_pad_1", "right_finger_pad_2")}
        self.finger_bodies = {self.m.geom_bodyid[g] for g in self.pad_geoms}
        # Pad face of each finger in its body frame: the jaw-side face plane of
        # the pad boxes (the finger mesh's pad layer lies on it), its outward
        # normal sign along body y and the x/z rectangle the pads cover.
        self._pad_faces = {}
        for f in self.finger_bodies:
            pads = [g for g in self.pad_geoms if self.m.geom_bodyid[g] == f]
            pos, size = self.m.geom_pos[pads], self.m.geom_size[pads]
            sign = float(np.sign(pos[0, 1]))
            self._pad_faces[int(f)] = (sign, float(np.mean(pos[:, 1]+sign*size[:, 1])),
                                       float(np.min(pos[:, 0]-size[:, 0])), float(np.max(pos[:, 0]+size[:, 0])),
                                       float(np.min(pos[:, 2]-size[:, 2])), float(np.max(pos[:, 2]+size[:, 2])))
        self._pad_face_cos = math.cos(math.radians(pad_face_angle_deg))
        self.finger_body_tolerance = float(finger_body_tolerance)
        self.robot_geoms = {g for g in range(self.m.ngeom)
                            if self.m.geom_bodyid[g] in self.robot_bodies
                            and (self.pm.geom_contype[g] or self.pm.geom_conaffinity[g])}
        self._robot_geom_mask = np.zeros(self.m.ngeom, bool)
        self._robot_geom_mask[list(self.robot_geoms)] = True
        self.margin = np.zeros(self.m.ngeom)
        for g in self.robot_geoms:
            self.margin[g] = finger_clearance if g in self.pad_geoms else arm_clearance
        for g in self.finger_hulls:
            self.margin[g] = finger_clearance if finger_body_clearance is None else finger_body_clearance
        self._margin_max = max(arm_clearance, finger_clearance, held_clearance)
        # MuJoCo combines geom contact margins with max/priority, whereas our
        # clearance test uses the sum. Generate a conservatively large contact
        # envelope and filter using true signed distance below.
        self._query_margin = self.m.geom_margin.copy()
        for g in self.robot_geoms:
            self._query_margin[g] = max(self._query_margin[g], self.margin[g]+self._margin_max)
        for g in self.finger_hulls:
            # A hull meets neither robot geoms nor the held part (FINGER_BIT),
            # so its largest test distance is its own clearance.
            self._query_margin[g] = max(self.m.geom_margin[g], self.margin[g]+1e-4)
        self.pm.geom_margin[:] = self._query_margin
        self._finger_bits = (self.pm.geom_contype.copy(), self.pm.geom_conaffinity.copy())
        self.held_clearance = held_clearance
        self.jaw_names = ("left_driver_joint", "left_finger_joint", "left_inner_knuckle_joint",
                          "right_driver_joint", "right_finger_joint", "right_inner_knuckle_joint")
        self.jaw_q = np.array([self.m.jnt_qposadr[self.m.joint(n).id] for n in self.jaw_names])
        self.jacp = np.zeros((3, self.m.nv))
        self.jacr = np.zeros((3, self.m.nv))
        self.held = None
        self._body_groups = {}
        self.mate_contact = None
        self.allow_pad_body = None
        self.last_contacts = []
        self.last_plan_stats = {}
        self._ref_last = None
        self._last_drive_time = None
        self._servo_integral = np.zeros(7)
        self.sync()

    @property
    def q(self):
        return self.d.qpos[self.qadr].copy()

    def sync(self):
        self.pd.qpos[:] = self.d.qpos
        self.pd.qvel[:] = 0
        self.pd.eq_active[:] = self.d.eq_active
        self.pm.eq_data[:] = self.m.eq_data
        self.pm.eq_solref[:] = self.m.eq_solref
        self.pd.ctrl[:] = self.d.ctrl
        mujoco.mj_forward(self.pm, self.pd)

    def tool_pose(self, data=None):
        d = self.d if data is None else data
        R = d.xmat[self.hand].reshape(3, 3).copy()
        return d.xpos[self.hand] + R @ self.offset, R

    def body_pose(self, body, data=None):
        d = self.d if data is None else data
        b = self.m.body(body).id if isinstance(body, str) else int(body)
        return d.xpos[b].copy(), d.xmat[b].reshape(3, 3).copy()

    def width(self, data=None):
        d = self.d if data is None else data
        a = self.m.geom("left_finger_pad_2").id
        b = self.m.geom("right_finger_pad_2").id
        return float(np.linalg.norm(d.geom_xpos[a] - d.geom_xpos[b]) - .004)

    @staticmethod
    def angle_for_width(width):
        """Parallel linkage geometry, not actuator command calibration."""
        # Pad centre y=.035+.035465*cos(t)-.042039*sin(t)-.024003.
        # Inner face is 2 mm inside the centre.
        r = math.hypot(.035465, .042039)
        v = (.5 * float(width) - (.035-.024003-.002)) / r
        return float(np.clip(math.acos(np.clip(v, -1, 1)) - math.atan2(.042039, .035465), 0, .85))

    def _jaw(self, width):
        if width is not None:
            self.pd.qpos[self.jaw_q] = self.angle_for_width(width)

    def hold_body(self, body=None):
        """Track the MEASURED grasp transform only in the planning copy.

        The held part travels rigidly with the hand and both finger links, so
        their hull contacts with it are the same at every configuration of a
        plan: they are judged once, when the grasp is chosen (the reference's
        closed-jaw check), and the held part's geoms drop FINGER_BIT here.
        """
        self.pm.geom_margin[:] = self._query_margin
        self.pm.geom_contype[:], self.pm.geom_conaffinity[:] = self._finger_bits
        if body is None:
            self.held = None
            self.mate_contact = None
            return
        b = self.m.body(body).id if isinstance(body, str) else int(body)
        j = int(self.m.body_jntadr[b])
        if j < 0 or self.m.jnt_type[j] != mujoco.mjtJoint.mjJNT_FREE:
            raise PlanningError("held body must have a free joint")
        R = self.d.xmat[self.hand].reshape(3, 3)
        p = R.T @ (self.d.xpos[b]-self.d.xpos[self.hand])
        rel = R.T @ self.d.xmat[b].reshape(3, 3)
        self.held = (b, int(self.m.jnt_qposadr[j]), p.copy(), rel.copy())
        members=self.body_group(b)
        for g in range(self.m.ngeom):
            if self.m.geom_bodyid[g] in members:
                self.pm.geom_margin[g] = max(self.m.geom_margin[g], self.held_clearance+self._margin_max)
                self.pm.geom_contype[g] &= ~FINGER_BIT
                self.pm.geom_conaffinity[g] &= ~FINGER_BIT

    def body_group(self, body):
        if body is None:
            return frozenset()
        if isinstance(body,(tuple,list,set,frozenset)):
            return frozenset().union(*(self.body_group(b) for b in body))
        bid=self.m.body(body).id if isinstance(body,str) else int(body)
        if bid not in self._body_groups:
            self._body_groups[bid]=body_subtree(self.m,bid)
        return self._body_groups[bid]

    def body_goal(self, pos, R, body=None):
        """Tool pose for a desired part pose, using its current measured grasp.

        Call again after transport: contact grasps can slip slightly. This only
        measures the transform and updates the planning copy's held-body model.
        """
        if body is not None:
            self.hold_body(body)
        elif self.held is not None:
            self.hold_body(self.held[0])
        if self.held is None:
            raise PlanningError("no measured grasp for body_goal")
        _b, _a, p, rel = self.held
        hand_R = np.asarray(R, float) @ rel.T
        tool_p = np.asarray(pos, float) + hand_R @ (self.offset-p)
        return tool_p, hand_R

    def set_mate_contact(self, carrier=None, pos=None, R=None, *, distance=.003, angle=.14, axis_only=False):
        """Permit an intended held-part/carrier contact only near its seat.

        Other part pairs and every robot contact remain checked. This affects
        collision planning, never live contact masks or the safety monitor.
        """
        if carrier is None:
            self.mate_contact = None
            return
        carriers=carrier if isinstance(carrier,(list,tuple,set,frozenset)) else (carrier,)
        bodies=frozenset(self.m.body(b).id if isinstance(b,str) else int(b) for b in carriers)
        self.mate_contact = (bodies, np.asarray(pos, float), np.asarray(R, float), float(distance), float(angle),bool(axis_only))

    def align_body(self, pos, R, *, body=None, tolerance=.0003, angle=.003,
                   max_passes=5, speed=.01, clearance=False, axis_only=False):
        """Iteratively correct transport slip with measured live part poses."""
        if body is not None:
            self.hold_body(body)
        if self.held is None:
            raise PlanningError("no held body for alignment")
        b = self.held[0]
        for _ in range(max_passes):
            p_now, r_now = self.body_pose(b)
            ep = float(np.linalg.norm(np.asarray(pos)-p_now))
            desired,er=axis_aligned_rotation(r_now,R) if axis_only else (np.asarray(R),float(np.linalg.norm(Rotation.from_matrix(np.asarray(R) @ r_now.T).as_rotvec())))
            if ep <= tolerance and er <= angle:
                return {"position_error_m": ep, "angle_error_rad": er}
            p, r = self.body_goal(pos, desired, b)
            self.line(p, r, speed=speed, clearance=clearance, allow_pad_body=b)
        p_now, r_now = self.body_pose(b)
        ep = float(np.linalg.norm(np.asarray(pos)-p_now))
        er = axis_aligned_rotation(r_now,R)[1] if axis_only else float(np.linalg.norm(Rotation.from_matrix(np.asarray(R) @ r_now.T).as_rotvec()))
        if ep > tolerance or er > angle:
            raise PlanningError(f"part alignment did not converge: {ep*1000:.3f} mm, {math.degrees(er):.3f} deg")
        return {"position_error_m": ep, "angle_error_rad": er}

    def _fk(self, q, width=None, held=True):
        self.pd.qpos[self.qadr] = q
        self._jaw(width)
        mujoco.mj_kinematics(self.pm, self.pd)
        mujoco.mj_comPos(self.pm, self.pd)
        if held and self.held is not None:
            _b, a, p, rel = self.held
            R = self.pd.xmat[self.hand].reshape(3, 3)
            self.pd.qpos[a:a+3] = self.pd.xpos[self.hand] + R @ p
            self.pd.qpos[a+3:a+7] = _quat(R @ rel)
            mujoco.mj_kinematics(self.pm, self.pd)
            mujoco.mj_comPos(self.pm, self.pd)
        return self.tool_pose(self.pd)

    def pad_face_contact(self, hull, pos, frame, hull_first):
        """True if a finger-hull contact belongs to that finger's pad (a grip).

        The hull's jaw-side face is the pads' contact plane, so a grip (also a
        pad squeezed into the part) shows as hull contacts (midpoints) on the
        pad rectangle (+-0.5 mm), at most 1.5 mm behind the face plane, whose
        normal is within pad_face_angle_deg of the pad normal. The finger mesh
        steps back 2 mm right at the pad edges (its pad layer ends 0.5 mm
        inside the pad boxes, its plate lies 2 mm behind the face); the hull
        bridges that step with a chamfer, so a contact on the rectangle at most
        1 mm behind the face counts as the pad's whatever its normal: no finger
        material is there outside the pad layer (a side or tip intrusion into
        the plate lies ~2 mm behind the face). A side, tip, back or link of the
        finger inside a part, or a part much wider than the planned jaw gap,
        does not count.
        """
        f = int(self.pm.geom_bodyid[hull])
        sign, face, x0, x1, z0, z1 = self._pad_faces[f]
        R = self.pd.xmat[f].reshape(3, 3)
        p = R.T @ (np.asarray(pos)-self.pd.xpos[f])
        if not (x0-.0005 <= p[0] <= x1+.0005 and z0-.0005 <= p[2] <= z1+.0005):
            return False
        behind = sign*(face-p[1])
        if -.0005 <= behind <= .001:
            return True
        normal = np.asarray(frame[:3]) * (1. if hull_first else -1.)
        return bool(sign*float(R[:, 1] @ normal) >= self._pad_face_cos and abs(behind) <= .0015)

    def collision(self, q, width=None, allow_pad_body=None, clearance=True):
        self._fk(q, width)
        mujoco.mj_collision(self.pm, self.pd)
        allowed = self.allow_pad_body if allow_pad_body is None else allow_pad_body
        allowed_bodies=self.body_group(allowed)
        held_body = None if self.held is None else self.held[0]
        held_bodies=self.body_group(held_body)
        bad = []
        n = int(self.pd.ncon)
        contacts = self.pd.contact
        pairs = np.asarray(contacts.geom).reshape(-1, 2)[:n]
        dists = np.asarray(contacts.dist)[:n]
        # Only contacts with a robot geom or the held part matter below; the
        # parts resting on the bench alone produce over a thousand. Same
        # contacts in the same order as testing each one.
        relevant = self._robot_geom_mask[pairs[:, 0]] | self._robot_geom_mask[pairs[:, 1]]
        if held_bodies:
            held_geoms = np.isin(self.pm.geom_bodyid[pairs], list(held_bodies))
            relevant |= held_geoms[:, 0] | held_geoms[:, 1]
        geometry = None
        for i in np.flatnonzero(relevant):
            g, h = int(pairs[i, 0]), int(pairs[i, 1])
            dist = float(dists[i])
            a, b = int(self.pm.geom_bodyid[g]), int(self.pm.geom_bodyid[h])
            if a in self.robot_bodies and b in self.robot_bodies:
                if self.pm.body_parentid[a] == b or self.pm.body_parentid[b] == a:
                    continue  # mechanically adjacent housings share a joint interface
            if a == self.base or b == self.base:
                # The bolted robot base itself is allowed to contact its support.
                if (a == self.base and b not in self.robot_bodies) or (b == self.base and a not in self.robot_bodies):
                    continue
            if ((g in self.pad_geoms and b in allowed_bodies|held_bodies) or
                    (h in self.pad_geoms and a in allowed_bodies|held_bodies)):
                continue
            hull = g if g in self.finger_hulls else h if h in self.finger_hulls else -1
            if hull >= 0 and (b if hull == g else a) in allowed_bodies:
                # A part the pads may grip (not yet held, or just released):
                # pad-face contact is the grip itself; any other finger-link
                # contact may only graze it (finger_body_tolerance), whatever
                # the clearance setting.
                if dist < -self.finger_body_tolerance:
                    if geometry is None:
                        geometry = (np.asarray(contacts.pos), np.asarray(contacts.frame))
                    if not self.pad_face_contact(hull, geometry[0][i], geometry[1][i], hull == g):
                        bad.append((self.m.body(a).name, self.m.body(b).name, dist))
                continue
            if a in held_bodies and b in held_bodies and (
                    a==b or self.pm.body_parentid[a]==b or self.pm.body_parentid[b]==a):
                continue  # internal joint interface of this selected assembly
            if self.mate_contact is not None and held_body is not None:
                carriers, target_p, target_R, distance, angle, axis_only = self.mate_contact
                # The stator mounts to the leg; its passive rotor is not an
                # additional mating surface even though it travels with it.
                if (a == held_body and b in carriers) or (b == held_body and a in carriers):
                    p, R = self.body_pose(held_body, self.pd)
                    error=axis_aligned_rotation(R,target_R)[1] if axis_only else np.linalg.norm(Rotation.from_matrix(target_R @ R.T).as_rotvec())
                    if (np.linalg.norm(p-target_p) <= distance and
                            error <= angle):
                        continue
            limit = (self.margin[g]+self.margin[h]) if clearance else 0.
            if held_body is not None and (a in held_bodies or b in held_bodies):
                limit += self.held_clearance if clearance else 0.
            if dist < limit - 2e-5:
                bad.append((self.m.body(a).name, self.m.body(b).name, dist))
        self.last_contacts = bad
        return bool(bad)

    def _ik_one(self, pos, R, seed, width=None, max_iter=160):
        q = np.clip(np.asarray(seed, float), self.limits[:, 0]+1e-5, self.limits[:, 1]-1e-5)
        for _ in range(max_iter):
            p, r = self._fk(q, width, held=False)
            ep = pos-p
            er = Rotation.from_matrix(R @ r.T).as_rotvec()
            if np.linalg.norm(ep) < 1e-4 and np.linalg.norm(er) < 1e-3:
                return q
            mujoco.mj_jac(self.pm, self.pd, self.jacp, self.jacr, p, self.hand)
            J = np.vstack((self.jacp[:, self.vadr], .25*self.jacr[:, self.vadr]))
            e = np.r_[ep, .25*er]
            lam = 1e-4 + min(.01, .03*np.linalg.norm(ep))
            dq = J.T @ np.linalg.solve(J @ J.T + lam*lam*np.eye(6), e)
            dq *= min(1., .18/(np.max(np.abs(dq))+1e-12))
            q = np.clip(q+dq, self.limits[:, 0]+1e-5, self.limits[:, 1]-1e-5)
        return None

    def ik(self, pos, R=DOWN, *, width=None, attempts=24, max_solutions=8,
           collision=True, allow_pad_body=None, seed=None, clearance=True):
        self.sync()
        pos, R = np.asarray(pos, float), np.asarray(R, float)
        q0 = self.q if seed is None else np.asarray(seed, float)
        seeds = [q0, HOME]
        for j3 in (-1.3, 1.3):
            s = HOME.copy(); s[2] = j3; seeds.append(s)
        while len(seeds) < attempts:
            if len(seeds) % 3:
                seeds.append(np.clip(q0+self.rng.normal(0, .9, 7), self.limits[:, 0], self.limits[:, 1]))
            else:
                lo, hi = np.maximum(self.limits[:, 0], -math.pi), np.minimum(self.limits[:, 1], math.pi)
                seeds.append(self.rng.uniform(lo, hi))
        solutions = []
        for s in seeds[:attempts]:
            q = self._ik_one(pos, R, s, width=width)
            if q is None or any(np.linalg.norm(q-v) < .08 for v in solutions):
                continue
            if collision and self.collision(q, width, allow_pad_body, clearance):
                continue
            solutions.append(q)
            if len(solutions) >= max_solutions:
                break
        return sorted(solutions, key=lambda q: float(np.linalg.norm(q-q0)))

    def segment_free(self, a, b, *, width=None, allow_pad_body=None,
                     clearance=True, resolution=.007):
        n = max(1, int(math.ceil(np.max(np.abs(np.asarray(b)-a))/resolution)))
        return all(not self.collision((1-t)*np.asarray(a)+t*np.asarray(b), width,
                                      allow_pad_body, clearance)
                   for t in np.linspace(0, 1, n+1))

    def _clearance_escape(self, start, goal, width, allow_pad_body):
        """Leave a soft clearance shell without accepting any actual collision.

        Jaw opening can consume a few tenths of a millimetre of a previously
        checked clearance. Only the existing body pairs may remain inside that
        shell, and their minimum separation cannot decrease along the exit.
        """
        self.collision(start,width,allow_pad_body,True)
        baseline={}
        for a,b,d in self.last_contacts:
            key=tuple(sorted((a,b)));baseline[key]=min(d,baseline.get(key,math.inf))
        if self.collision(start,width,allow_pad_body,False):
            return None
        delta=np.asarray(goal)-start
        directions=[delta/(np.linalg.norm(delta)+1e-12)]
        directions.extend(self.rng.normal(size=7) for _ in range(60))
        for direction in directions:
            direction=direction/(np.linalg.norm(direction)+1e-12)
            for length in (.05,.12,.25):
                candidate=np.clip(start+direction*length,self.limits[:,0],self.limits[:,1])
                if self.collision(candidate,width,allow_pad_body,True):
                    continue
                previous=baseline.copy();cleared=set();okay=True
                n=max(2,int(math.ceil(np.max(np.abs(candidate-start))/.007)))
                for t in np.linspace(0,1,n+1):
                    q=start+t*(candidate-start)
                    self.collision(q,width,allow_pad_body,True)
                    current={}
                    for a,b,d in self.last_contacts:
                        key=tuple(sorted((a,b)));current[key]=min(d,current.get(key,math.inf))
                    if any(key not in baseline or key in cleared or d < previous.get(key,baseline[key])-2e-5 for key,d in current.items()):
                        okay=False;break
                    cleared.update(set(previous)-set(current))
                    previous.update(current)
                    if self.collision(q,width,allow_pad_body,False):
                        okay=False;break
                if okay:
                    return candidate
        return None

    def plan(self, goal, *, width=None, allow_pad_body=None, timeout=12., clearance=True):
        self.sync()
        start, goal = self.q, np.asarray(goal, float)
        prefix=[]
        if self.collision(start, width, allow_pad_body, clearance):
            contacts=self.last_contacts[:3]
            escape=self._clearance_escape(start,goal,width,allow_pad_body) if clearance else None
            if escape is None:
                raise PlanningError(f"start in collision: {contacts}")
            prefix=[start];start=escape
        if self.collision(goal, width, allow_pad_body, clearance):
            raise PlanningError(f"goal in collision: {self.last_contacts[:3]}")
        if self.segment_free(start, goal, width=width, allow_pad_body=allow_pad_body, clearance=clearance):
            self.last_plan_stats = {"direct": True, "nodes": 2}
            return prefix+[start, goal]
        ts = time.monotonic()
        # Deterministic search budget: extend attempts, not wall seconds, so
        # the same start state plans the same path on any host or load. The
        # `timeout` argument keeps its meaning as a nominal budget (seconds of
        # search at PLAN_EXTENDS_PER_S); the wall clock is only a hang guard.
        budget = max(1, int(round(timeout*PLAN_EXTENDS_PER_S)))
        spent = [0]
        def searching():
            if time.monotonic()-ts > PLAN_HANG_GUARD_S:
                raise PlanningError("RRT-Connect hang guard (wall clock)")
            return spent[0] < budget
        trees = [([start], [-1]), ([goal], [-1])]
        swapped = False
        def extend(tree, target):
            spent[0] += 1
            nodes, parents = tree
            near = int(np.argmin([np.linalg.norm(n-target) for n in nodes]))
            delta = target-nodes[near]
            distance = np.linalg.norm(delta)
            q = nodes[near]+delta*min(1., .18/(distance+1e-12))
            if not self.segment_free(nodes[near], q, width=width, allow_pad_body=allow_pad_body, clearance=clearance):
                return None, False
            nodes.append(q); parents.append(near)
            return len(nodes)-1, distance <= .18
        def trace(tree, index):
            out = []
            while index >= 0:
                out.append(tree[0][index]); index = tree[1][index]
            return out[::-1]
        while searching():
            a, b = trees
            if self.rng.random() < .2:
                sample = b[0][-1]
            else:
                lo = np.maximum(self.limits[:, 0], np.minimum(start, goal)-1.5)
                hi = np.minimum(self.limits[:, 1], np.maximum(start, goal)+1.5)
                sample = self.rng.uniform(lo, hi)
            ia, _ = extend(a, sample)
            if ia is not None:
                while searching():
                    ib, reached = extend(b, a[0][ia])
                    if ib is None:
                        break
                    if reached:
                        pa, pb = trace(a, ia), trace(b, ib)
                        path = pa+pb[-2::-1]
                        if swapped:
                            path.reverse()
                        # Deterministic greedy shortcuts, each checked at <=7 mrad.
                        out, i = [path[0]], 0
                        while i < len(path)-1:
                            j = len(path)-1
                            while j > i+1 and not self.segment_free(path[i], path[j], width=width, allow_pad_body=allow_pad_body, clearance=clearance):
                                j -= 1
                            out.append(path[j]); i = j
                        self.last_plan_stats = {"direct": False, "nodes": len(a[0])+len(b[0]), "waypoints": len(out), "extends": spent[0], "budget": budget, "wall_s": time.monotonic()-ts}
                        return prefix+out
            trees.reverse(); swapped = not swapped
        raise PlanningError(f"RRT-Connect search budget exhausted ({budget} extends)")

    def wait(self, seconds):
        for _ in range(max(0, int(math.ceil(seconds/.01)))):
            self.cell.tick()

    def _drive(self, target):
        # Compensate the robot's own gravity with the existing position actuators.
        # This shifts the setpoint, not qpos or the actuator's torque limits.
        target = np.asarray(target, float)
        gains = self.m.actuator_gainprm[self.aids, 0]
        velocity = np.zeros(7)
        dt = None if self._last_drive_time is None else self.d.time-self._last_drive_time
        if dt is not None and .005 < dt < .02:
            velocity = np.clip((target-self._ref_last)/dt, -1., 1.)
        error = target-self.q
        if np.max(np.abs(velocity)) < .02:
            self._servo_integral = np.clip(self._servo_integral+.02*error, -.01, .01)
        damping = -self.m.actuator_biasprm[self.aids, 2]+self.m.dof_damping[self.vadr]
        friction = self.m.dof_frictionloss[self.vadr]*np.tanh(error/.0003)
        feedforward = (self.d.qfrc_bias[self.vadr]-self.d.qfrc_applied[self.vadr]+damping*velocity+friction) / gains
        self.d.ctrl[self.aids] = np.clip(target+feedforward+self._servo_integral,
                                        self.m.actuator_ctrlrange[self.aids, 0],
                                        self.m.actuator_ctrlrange[self.aids, 1])
        self._ref_last, self._last_drive_time = target.copy(), float(self.d.time)

    def execute(self, path, *, speed=.35, settle=.3):
        for a, b in zip(path[:-1], path[1:]):
            duration = max(.2, float(np.max(np.abs(b-a)))*1.875/speed)
            n = int(math.ceil(duration/.01))
            for i in range(1, n+1):
                u = i/n
                s = u*u*u*(10+u*(-15+6*u))
                self._drive((1-s)*a+s*b)
                self.cell.tick()
        if path:
            for _ in range(int(math.ceil(settle/.01))):
                self._drive(path[-1]); self.cell.tick()
        return self.q

    def move_joints(self, target, *, width=None, allow_pad_body=None, speed=.35, clearance=True):
        path = self.plan(target, width=width, allow_pad_body=allow_pad_body, clearance=clearance)
        return self.execute(path, speed=speed)

    def move_pose(self, pos, R=DOWN, *, width=None, allow_pad_body=None,
                  attempts=24, speed=.35, clearance=True):
        solutions = self.ik(pos, R, width=width, attempts=attempts,
                            allow_pad_body=allow_pad_body, clearance=clearance)
        errors = []
        for q in solutions:
            try:
                return self.move_joints(q, width=width, allow_pad_body=allow_pad_body,
                                        speed=speed, clearance=clearance)
            except PlanningError as e:
                errors.append(str(e))
        raise PlanningError(f"no feasible motion to {np.round(pos,4)} ({len(solutions)} IK solutions): {errors[:2]}")

    def line(self, pos, R=None, *, width=None, allow_pad_body=None, speed=.025,
             clearance=True, escape_start=True, angular_speed=.3, hold=30, smooth=False,
             start_q=None, execute=True):
        """Precheck the whole Cartesian segment before physically executing it.

        With ``start_q`` the segment starts from that planning configuration
        instead of the live one; ``execute=False`` returns the checked joint
        chain without moving (both only affect the planning copy).
        """
        self.sync()
        if start_q is None:
            p0, r0 = self.tool_pose()
            start_q = self.q
        elif execute:
            raise ValueError("start_q is only for checking a segment (execute=False)")
        else:
            start_q = np.asarray(start_q, float)
            p0, r0 = self._fk(start_q, width)
        pos = np.asarray(pos, float)
        R = r0 if R is None else np.asarray(R, float)
        rv = Rotation.from_matrix(R @ r0.T).as_rotvec()
        n = max(1, int(math.ceil(np.linalg.norm(pos-p0)/.003)), int(math.ceil(np.linalg.norm(rv)/.025)))
        qs, prev = [start_q], start_q
        # A resting grasp starts at a compliant table contact. Permit only
        # monotonically escaping existing contacts; no new/deeper contact.
        self.collision(prev, width, allow_pad_body, clearance)
        initial = {tuple(sorted((a,b))): min(d for x,y,d in self.last_contacts
                   if tuple(sorted((x,y))) == tuple(sorted((a,b))))
                   for a,b,_ in self.last_contacts} if escape_start else {}
        escaping = dict(initial)
        for u in np.linspace(0, 1, n+1)[1:]:
            p = (1-u)*p0+u*pos
            rr = Rotation.from_rotvec(u*rv).as_matrix() @ r0
            q = self._ik_one(p, rr, prev, width=width)
            valid = q is not None
            if valid:
                ns = max(1, int(math.ceil(np.max(np.abs(q-prev))/.007)))
                for v in np.linspace(0, 1, ns+1)[1:]:
                    self.collision((1-v)*prev+v*q, width, allow_pad_body, clearance)
                    current = {}
                    for a,b,d in self.last_contacts:
                        key = tuple(sorted((a,b)))
                        current[key] = min(current.get(key, math.inf), d)
                    if any(key not in escaping or dist < escaping[key]-5e-5
                           for key,dist in current.items()):
                        valid = False; break
                    escaping = current
            if not valid:
                raise PlanningError(f"Cartesian path blocked at {u:.2f}: {self.last_contacts[:3]}")
            qs.append(q); prev = q
        if escaping:
            raise PlanningError(f"Cartesian path does not escape initial contact: {escaping}")
        if not execute:
            return qs
        total_t = max(.2, np.linalg.norm(pos-p0)/speed, np.linalg.norm(rv)/angular_speed)
        ticks = max(n, int(math.ceil(total_t/.01)))
        for k in range(1, ticks+1):
            u = k/ticks
            if smooth:
                u = u*u*u*(10+u*(-15+6*u))
            t = u*n
            i = min(int(t), n-1)
            f = t-i
            self._drive((1-f)*qs[i]+f*qs[i+1]); self.cell.tick()
        for _ in range(hold):
            self._drive(qs[-1]); self.cell.tick()
        return self.tool_pose()

    def set_width(self, width, *, seconds=.6, squeeze=.0):
        width = float(np.clip(width-squeeze, 0., .085))
        angle = self.angle_for_width(width)
        command = float(np.clip(100*angle/.333, 0, 255))
        start = float(self.d.ctrl[self.ga])
        q = self.q
        n = max(1, int(math.ceil(seconds/.01)))
        for i in range(1, n+1):
            self.d.ctrl[self.ga] = start+(command-start)*i/n
            self._drive(q); self.cell.tick()
        return self.width()

    def open(self, seconds=.6):
        self.hold_body(None)
        return self.set_width(.085, seconds=seconds)

    def close(self, width=0., seconds=.8, squeeze=.004):
        return self.set_width(width, seconds=seconds, squeeze=squeeze)
