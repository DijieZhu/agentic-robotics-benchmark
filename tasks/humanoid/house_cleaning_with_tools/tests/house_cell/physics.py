"""The cell's physical rules beyond rigid-body contact: hand grips, the tool clips, the fixtures' mechanisms (the bin
lids' hold-open springs, the faucet's water, the soap pump's strokes), fragile dishes, safety.

Grip (assisted grasp, the cell manual states it): a hand that closes while its grasp site lies at one of an object's
grasp features holds that object rigidly at the relative pose of the closing instant (a soft weld whose force acts at
the grasp site) until the hand opens, or slips when the hold needs more than the hand's limits (one hand: 30 N or
5 N m; k hands: 30 N per hand of the object's net load; for 50 ms). A fixture's moving part (a bin lid's tab, the
faucet's lever, the soap pump's head) is held at that point only (it turns about its hinge in the hand) and slips
at 60 N. The features (objects.py): bars and cylinders
(the pinch closes across the axis, the grasp site within 15 mm of the surface along the length), slabs (the pinch
across the thickness, the site within 15 mm of the slab and at most 45 mm inside an edge), discs (the pinch along
the disc's axis, the site within 15 mm of its mid-plane and at most 45 mm inside the rim), lumps (the site within
12 mm of the surface).

Clips: a tool whose handle is let go within 12 mm of its clip's jaw centre, the handle within 12 deg of the vertical,
clicks in and hangs there (a soft weld to the wall at that pose); a hand that grips the tool takes it out of the clip.
"""

from __future__ import annotations

import math

import mujoco
import numpy as np

from . import objects as O
from . import room as RM
from .robot import FLOOR, HAND_CLOSED, HAND_JOINTS, HAND_OPEN, HELD, OBJ, ROBOT, WORLD

TICK = 0.01

GRIP_MAX_FORCE = 30.0
GRIP_MAX_TORQUE = 5.0
GRIP_MAX_TORQUE_BAR = 12.0         # a bar or cylinder held in the closed hand (a power grasp) resists more twisting
POINT_GRIP_MAX_FORCE = 60.0        # a fixture's moving part held at a point (a lid's tab, the lever, the pump's head)
GRIP_SLIP_TICKS = 5
GRIP_SOLREF = (0.02, 1.0)
GRIP_REACH = 0.045                 # m: an edge / rim at most this far inside the hand
GRIP_SURFACE = 0.015               # m: the grasp site within this of a bar's surface / a slab / a disc's mid-plane
GRIP_LUMP = 0.012                  # m: ... of a lump's surface
GRIP_AXIS_COS = math.cos(math.radians(35.0))
GRIP_ACROSS_SIN = math.sin(math.radians(35.0))

# the hand's shape on what it holds: each finger (the thumb, the middle and the index finger) closes along the pinch
# synergy until it meets the held object's surface (into a soft one it sinks by SOFT_INDENT: the sponge's foam, a
# crumpled napkin) or anything else solid within SHAPE_REACH of the grasp site (the surface the object lies on, a
# tray's rim), so the fingers rest on surfaces instead of passing through them; a hand that closes on nothing closes
# fully. The hands carry no contact (the grip rule holds the object): this is their drawn and measured shape.
FINGERS = {"thumb": ("thumb_1", "thumb_2"), "middle": ("middle_0", "middle_1"), "index": ("index_0", "index_1")}
SOFT_INDENT = {"sponge": 0.004, "napkin": 0.006}
RIGID_INDENT = 0.0005
CLOSE_STEPS = 32
SHAPE_REACH = 0.25

CLIP_TILT_COS = math.cos(math.radians(12.0))
CLIP_SOLREF = (0.02, 1.0)

LID_SPRING = 0.35                  # N m: the hold-open spring beyond RM.LID_HOLD
SAFETY = {"robot_contact_N": 150.0, "robot_contact_ticks": 20, "throw_speed": 1.5, "strike_speed": 4.0,
          "strike_grace_s": 2.0}
# Ordinary tableware (the cell manual's section 8 states it at a data sheet's level): a porcelain dinner plate or a
# soda-lime tumbler survives a fall onto a hard surface of up to about 30 cm (an impact at up to about 2.5 m/s; a
# harder one chips or breaks it) and a steady load of several hundred newtons (a stack of 40 plates weighs about 180 N, a person
# leaning on a plate lying flat about 300 N); beyond either it breaks. An impact is measured by the kinetic energy
# (moving and turning) the dish loses while it touches something within IMPACT_TICKS (the length of an impact on the
# cell's contacts), as the speed of a fall that carries that energy: sqrt(2 x energy lost / mass) (a dish striking
# flat loses its whole fall; one striking with its rim first loses part, and the rest when it slaps down); the load
# must last LOAD_TICKS (a crush, not the force spike of an impact, which the energy rule judges).
FRAGILE = {"impact_speed": 2.5, "force_N": {"plate": 600.0, "glass": 400.0}}
IMPACT_TICKS = 4
LOAD_TICKS = 5


def _set_weld(m, d, eq, relpos, relR, solref, anchor=None):
    """Activate weld `eq` holding body2 at (relpos, relR) in body1's frame; the force acts at `anchor` (body2 frame)."""
    a = np.zeros(3) if anchor is None else np.asarray(anchor, float)
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(relR, float).reshape(-1))
    m.eq_data[eq, 0:3] = a
    m.eq_data[eq, 3:6] = np.asarray(relpos, float) + np.asarray(relR, float) @ a
    m.eq_data[eq, 6:10] = q
    m.eq_data[eq, 10] = 1.0
    m.eq_solref[eq] = solref
    d.eq_active[eq] = 1


def rel_pose(d, b1, b2):
    R1 = d.xmat[b1].reshape(3, 3)
    R2 = d.xmat[b2].reshape(3, 3)
    return R1.T @ (d.xpos[b2] - d.xpos[b1]), R1.T @ R2


def weld_wrench(d, eq):
    """(force vector, torque norm) an active grip constraint transmits (a point grip transmits no torque)."""
    rows = np.where((d.efc_type[:d.nefc] == mujoco.mjtConstraint.mjCNSTR_EQUALITY) & (d.efc_id[:d.nefc] == eq))[0]
    if len(rows) < 3:
        return np.zeros(3), 0.0
    f = d.efc_force[rows]
    return f[:3].copy(), (float(np.linalg.norm(f[3:6])) if len(rows) >= 6 else 0.0)


def feature_score(feat, pl, yl):
    """Distance score of a valid grasp of `feat` by a grasp site at pl with pinch axis yl (both in the object's frame),
    or None."""
    kind = feat[0]
    if kind == "cyl":
        a, b, r = np.asarray(feat[1], float), np.asarray(feat[2], float), float(feat[3])
        u = b - a
        L = float(np.linalg.norm(u))
        u = u / L
        s = float((pl - a) @ u)
        if s < -0.005 or s > L + 0.005:
            return None
        rho = float(np.linalg.norm(pl - a - s * u))
        if rho > r + GRIP_SURFACE:
            return None
        if abs(float(yl @ u)) > GRIP_ACROSS_SIN:
            return None
        return rho
    if kind == "slab":
        c, half, k = np.asarray(feat[1], float), np.asarray(feat[2], float), int(feat[3])
        dl = pl - c
        if abs(float(yl[k])) < GRIP_AXIS_COS:
            return None
        if abs(dl[k]) > half[k] + GRIP_SURFACE:
            return None
        u, v = [i for i in range(3) if i != k]
        best = None
        for p_, q_ in ((u, v), (v, u)):
            inside = half[p_] - abs(dl[p_])
            if -0.005 <= inside <= GRIP_REACH and abs(dl[q_]) <= half[q_] - 0.005:
                best = inside if best is None else min(best, inside)
        return best
    if kind == "disc":
        c, R, t = np.asarray(feat[1], float), float(feat[2]), float(feat[3])
        dl = pl - c
        if abs(float(yl[2])) < GRIP_AXIS_COS:
            return None
        if abs(dl[2]) > t + GRIP_SURFACE:
            return None
        rho = float(math.hypot(dl[0], dl[1]))
        if not (R - GRIP_REACH <= rho <= R + 0.005):
            return None
        return R - rho
    if kind == "ball":
        c, r = np.asarray(feat[1], float), float(feat[2])
        dist = float(np.linalg.norm(pl - c))
        return dist if dist <= r + GRIP_LUMP else None
    return None


class Grips:
    """Assisted grasps of the robot's two hands."""

    def __init__(self, m, d, robot, meta):
        self.m, self.d, self.r = m, d, robot
        self.features = meta["features"]
        self.names = list(meta["graspable"])
        self.body = {n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n) for n in self.names}
        self.eq = {(s, n): mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_EQUALITY, f"grip_{s}_{n}")
                   for s in ("left", "right") for n in self.names}
        self.state = {s: {"obj": None, "eq": -1, "over": 0, "closed": False, "load": 0.0} for s in ("left", "right")}
        self.point_parts = {n for (sd, n), eq in self.eq.items() if m.eq_type[eq] == mujoco.mjtEq.mjEQ_CONNECT}
        self.events = []
        self.on_engage = []            # callbacks (side, name): the clips, the safety's release record
        self.on_release = []
        # a held loose object does not touch the robot (it is part of the hand while held): its colliders switch to
        # the HELD bit and back when no hand holds it any more
        self.loose_geoms = {n: [g for g in range(m.ngeom) if m.geom_bodyid[g] in self._subtree(m, self.body[n])
                                and (m.geom_contype[g] or m.geom_conaffinity[g])] for n in self.names if n in O.LOOSE}
        self.saved_bits = {n: (m.geom_contype[gs].copy(), m.geom_conaffinity[gs].copy())
                           for n, gs in self.loose_geoms.items()}
        # the hand's shape: the fingers' drawn meshes, the colliders of everything a hand can take hold of
        B = mujoco.mjtObj.mjOBJ_BODY
        self.finger_geoms, self.finger_idx = {}, {}
        for sd in ("left", "right"):
            for f, links in FINGERS.items():
                bodies = {mujoco.mj_name2id(m, B, f"{robot.name}/{sd}_hand_{w}_link") for w in links}
                self.finger_geoms[(sd, f)] = [g for g in range(m.ngeom) if int(m.geom_bodyid[g]) in bodies
                                              and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH]
                self.finger_idx[f] = [HAND_JOINTS.index(w) for w in links]
        self.part_geoms = {n: [g for g in range(m.ngeom) if m.geom_bodyid[g] in self._subtree(m, self.body[n])
                               and (m.geom_contype[g] or m.geom_conaffinity[g])] for n in self.names}
        self.kind = {n: (O.LOOSE[n][0] if n in O.LOOSE else "fixture") for n in self.names}
        self.solid = [g for g in range(m.ngeom) if (m.geom_contype[g] or m.geom_conaffinity[g])
                      and not (m.geom_contype[g] & ROBOT)]
        self.dk = mujoco.MjData(m)
        self._fromto = np.zeros(6)

    @staticmethod
    def _subtree(m, root):
        out = {root}
        for b in range(m.nbody):
            p = b
            while p > 0:
                if p == root:
                    out.add(b)
                    break
                p = int(m.body_parentid[p])
        return out

    def _set_held(self, name, held):
        gs = self.loose_geoms.get(name)
        if not gs:
            return
        if held:
            self.m.geom_contype[gs] = HELD
            self.m.geom_conaffinity[gs] = WORLD | OBJ | FLOOR
        else:
            ct, ca = self.saved_bits[name]
            self.m.geom_contype[gs] = ct
            self.m.geom_conaffinity[gs] = ca

    def candidates(self, side):
        sid = self.r.grasp_site[side]
        p = self.d.site_xpos[sid].copy()
        y = self.d.site_xmat[sid].reshape(3, 3)[:, 1].copy()
        out = []
        for n in self.names:
            b = self.body[n]
            R = self.d.xmat[b].reshape(3, 3)
            pl = R.T @ (p - self.d.xpos[b])
            yl = R.T @ y
            best, kind = None, None
            for f in self.features[n]:
                sc = feature_score(f, pl, yl)
                if sc is not None and (best is None or sc < best):
                    best, kind = sc, f[0]
            if best is not None:
                out.append((best, n, kind))
        return sorted(out)

    def command(self, side, close, t):
        st = self.state[side]
        if close and not st["closed"]:
            st["closed"] = True
            other = self.state["right" if side == "left" else "left"]["obj"]
            for _, n, kind in self.candidates(side):
                self._engage(side, n, t)
                st["kind"] = kind
                break
            _ = other
        elif not close and st["closed"]:
            st["closed"] = False
            if st["obj"] is not None:
                self._release(side, t, "opened")

    def _engage(self, side, name, t):
        st = self.state[side]
        eq = self.eq[(side, name)]
        hb = self.r.hand_body[side]
        bo = self.body[name]
        site = self.d.site_xpos[self.r.grasp_site[side]]
        if self.m.eq_type[eq] == mujoco.mjtEq.mjEQ_CONNECT:
            # a point grip: the grasp site's point of the hand held on the same point of the part
            self.m.eq_data[eq, 0:3] = self.d.xmat[hb].reshape(3, 3).T @ (site - self.d.xpos[hb])
            self.m.eq_data[eq, 3:6] = self.d.xmat[bo].reshape(3, 3).T @ (site - self.d.xpos[bo])
            self.m.eq_solref[eq] = GRIP_SOLREF
            self.d.eq_active[eq] = 1
        else:
            p, R = rel_pose(self.d, hb, bo)
            anchor = self.d.xmat[bo].reshape(3, 3).T @ (site - self.d.xpos[bo])
            _set_weld(self.m, self.d, eq, p, R, GRIP_SOLREF, anchor=anchor)
        st.update(obj=name, eq=eq, over=0, shape=self.hand_shape(side, name))
        self._set_held(name, True)
        self.events.append((round(t, 3), side, "grip", name))
        for cb in self.on_engage:
            cb(side, name, t)

    def _release(self, side, t, why):
        st = self.state[side]
        name = st["obj"]
        if st["eq"] >= 0:
            self.d.eq_active[st["eq"]] = 0
        self.events.append((round(t, 3), side, why, name))
        st.update(obj=None, eq=-1, over=0, load=0.0, shape=None)
        if name is not None and name not in self.held():
            self._set_held(name, False)
        for cb in self.on_release:
            cb(side, name, t, why)

    def hand_shape(self, side, name, qpos=None):
        """The hand's finger targets on `name` (see FINGERS): per finger, the closing fraction at which it first sinks
        into the object's colliders by more than the object's indentation, or into anything else solid nearby, found
        on a scratch copy of the present state (or of `qpos`, a planner's); the fingers start from the open hand. A
        finger that already intersects something open takes the fraction of the least intersection."""
        m, dk = self.m, self.dk
        dk.qpos[:] = self.d.qpos if qpos is None else qpos
        adr = self.r.hand_q[side]
        qo, qc = np.array(HAND_OPEN[side], float), np.array(HAND_CLOSED[side], float)
        dk.qpos[adr] = qo
        mujoco.mj_kinematics(m, dk)
        site = dk.site_xpos[self.r.grasp_site[side]].copy()
        out = qc.copy()
        part = self.part_geoms[name]
        ps = set(part)
        near = [g for g in self.solid if g not in ps and (m.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE or
                float(np.linalg.norm(dk.geom_xpos[g] - site)) < float(m.geom_rbound[g]) + SHAPE_REACH)]
        indent = SOFT_INDENT.get(self.kind[name], RIGID_INDENT)
        ft = self._fromto
        for f, idx in self.finger_idx.items():
            geoms = self.finger_geoms[(side, f)]

            def excess(u):
                q = qo.copy()
                q[idx] = qo[idx] + u * (qc[idx] - qo[idx])
                dk.qpos[adr] = q
                mujoco.mj_kinematics(m, dk)
                e = max(-mujoco.mj_geomDistance(m, dk, g, h, 0.05, ft) for g in geoms for h in part) - indent
                for h in near:
                    for g in geoms:
                        e = max(e, -mujoco.mj_geomDistance(m, dk, g, h, 0.05, ft) - RIGID_INDENT)
                return e

            us = np.linspace(0.0, 1.0, CLOSE_STEPS + 1)
            e0 = excess(0.0)
            if e0 > 0.0:
                es = [e0] + [excess(u) for u in us[1:]]
                u_stop = float(us[int(np.argmin(es))])
            else:
                u_stop = 1.0
                for k in range(1, len(us)):
                    if excess(us[k]) > 0.0:
                        lo, hi = float(us[k - 1]), float(us[k])
                        for _ in range(6):
                            mid = 0.5 * (lo + hi)
                            if excess(mid) > 0.0:
                                hi = mid
                            else:
                                lo = mid
                        u_stop = lo
                        break
            out[idx] = qo[idx] + u_stop * (qc[idx] - qo[idx])
        return out

    def shape(self, side):
        """The finger targets of a hand that holds something (None: the plain open / closed synergy)."""
        st = self.state[side]
        return st.get("shape") if st["obj"] is not None else None

    def holding(self, side):
        return self.state[side]["obj"]

    def held(self):
        return {st["obj"] for st in self.state.values() if st["obj"]}

    def tick(self, t):
        by_obj = {}
        for side, st in self.state.items():
            if st["obj"] is None:
                continue
            fvec, torque = weld_wrench(self.d, st["eq"])
            by_obj.setdefault(st["obj"], []).append((side, fvec, torque))
        for name, hands in by_obj.items():
            k = len(hands)
            if name in self.point_parts:
                share = float(np.linalg.norm(sum(h[1] for h in hands))) / k
                over = share > POINT_GRIP_MAX_FORCE
            elif k == 1:
                (side, fvec, torque), = hands
                share = float(np.linalg.norm(fvec))
                tmax = GRIP_MAX_TORQUE_BAR if self.state[side].get("kind") == "cyl" else GRIP_MAX_TORQUE
                over = share > GRIP_MAX_FORCE or torque > tmax
            else:
                share = float(np.linalg.norm(sum(h[1] for h in hands))) / k
                over = share > GRIP_MAX_FORCE
            for side, _, _ in hands:
                st = self.state[side]
                st["over"] = st["over"] + 1 if over else 0
                st["load"] = share
            if over and all(self.state[h[0]]["over"] >= GRIP_SLIP_TICKS for h in hands):
                for side, _, _ in hands:
                    self._release(side, t, "slipped")

    def tactile(self, side):
        st = self.state[side]
        return float(st.get("load", 0.0)) if st["obj"] is not None else 0.0


class Clips:
    """The two spring clips on the tool rail: the broom's and the dustpan's."""

    def __init__(self, m, d, meta, grips: Grips):
        self.m, self.d, self.g = m, d, grips
        self.body = {t: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, t) for t in RM.CLIPS}
        self.eq = {t: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_EQUALITY, f"clip_{t}") for t in RM.CLIPS}
        self.axis = {t: next(f for f in meta["features"][t] if f[0] == "cyl") for t in RM.CLIPS}
        self.engaged = {t: False for t in RM.CLIPS}
        self.events = []
        grips.on_engage.append(self._taken)

    def _taken(self, side, name, t):
        if name in self.engaged and self.engaged[name]:
            self.d.eq_active[self.eq[name]] = 0
            self.engaged[name] = False
            self.events.append((round(t, 3), "out", name))

    def in_jaws(self, tool):
        """The handle's axis passes within the capture distance of the jaw centre, near vertical."""
        b = self.body[tool]
        R, p = self.d.xmat[b].reshape(3, 3), self.d.xpos[b]
        _, a, c, _r = self.axis[tool]
        A = p + R @ np.asarray(a, float)
        C = p + R @ np.asarray(c, float)
        u = C - A
        L = float(np.linalg.norm(u))
        u /= L
        if abs(float(u[2])) < CLIP_TILT_COS:
            return False
        jaw = np.asarray(RM.CLIPS[tool], float)
        s = float((jaw - A) @ u)
        if s < 0.05 or s > L - 0.02:
            return False
        return float(np.linalg.norm(jaw - A - s * u)) <= RM.CLIP_CAPTURE

    def engage(self, tool, t):
        b = self.body[tool]
        _set_weld(self.m, self.d, self.eq[tool], self.d.xpos[b].copy(), self.d.xmat[b].reshape(3, 3).copy(),
                  CLIP_SOLREF)
        self.engaged[tool] = True
        self.events.append((round(t, 3), "in", tool))

    def tick(self, t):
        held = self.g.held()
        for tool in RM.CLIPS:
            if not self.engaged[tool] and tool not in held and self.in_jaws(tool):
                self.engage(tool, t)


class Mechanisms:
    """The bin lids' hold-open springs, the faucet's water and the soap pump's strokes."""

    def __init__(self, m, d, meta):
        self.m, self.d = m, d
        J = mujoco.mjtObj.mjOBJ_JOINT
        self.lid = {}
        for slot in meta["bins"]:
            j = mujoco.mj_name2id(m, J, f"bin{slot}_lid_hinge")
            self.lid[slot] = (int(m.jnt_qposadr[j]), int(m.jnt_dofadr[j]))
        j = mujoco.mj_name2id(m, J, "faucet_lever_hinge")
        self.lever = (int(m.jnt_qposadr[j]), int(m.jnt_dofadr[j]))
        j = mujoco.mj_name2id(m, J, "soap_pump_slide")
        self.pump = (int(m.jnt_qposadr[j]), int(m.jnt_dofadr[j]))
        self.pump_armed = True
        self.strokes = 0
        self.water_geom = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "water")
        self.water_s = 0.0          # seconds the water ran

    def lid_angle(self, slot):
        return float(self.d.qpos[self.lid[slot][0]])

    def lever_angle(self):
        return float(self.d.qpos[self.lever[0]])

    def flow(self):
        F = RM.FAUCET
        return float(np.clip((self.lever_angle() - F["on_above"]) / (F["full_at"] - F["on_above"]), 0.0, 1.0)) \
            if self.lever_angle() > F["on_above"] else 0.0

    def water_on(self):
        return self.lever_angle() > RM.FAUCET["on_above"]

    def pre_step(self):
        """Per tick, before the physics substeps: the lids' springs."""
        for slot, (qa, da) in self.lid.items():
            self.d.qfrc_applied[da] = LID_SPRING if self.d.qpos[qa] > RM.LID_HOLD else 0.0

    def post_step(self):
        """Per tick, after the substeps: the water and the pump. Returns True on a pump stroke (one dose)."""
        f = self.flow()
        self.m.geom_rgba[self.water_geom, 3] = 0.34 if f > 0 else 0.0
        self.m.geom_size[self.water_geom, 0] = 0.0045 + 0.004 * f
        if f > 0:
            self.water_s += TICK
        q = float(self.d.qpos[self.pump[0]])
        stroke = False
        if self.pump_armed and q < -RM.SOAP["stroke_at"]:
            self.pump_armed = False
            self.strokes += 1
            stroke = True
        elif not self.pump_armed and q > -0.004:
            self.pump_armed = True
        return stroke


class Fragile:
    """Plates and glasses break when they strike something hard (an impact taking more than the kinetic energy of a
    fall at 2.5 m/s: about 30 cm) or are crushed (a contact load above their strength for LOAD_TICKS); a held dish is
    judged once it is let go of."""

    def __init__(self, m, d):
        self.m, self.d = m, d
        self.body = {n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n) for n in O.FRAGILE}
        self.of_body = {v: k for k, v in self.body.items()}
        self.strength = {n: FRAGILE["force_N"][O.LOOSE[n][0]] for n in O.FRAGILE}
        self.mass = {n: float(m.body_mass[b]) for n, b in self.body.items()}
        self.energy = {n: [] for n in O.FRAGILE}         # the kinetic energy (J) of the last IMPACT_TICKS ticks
        self.loaded = {n: 0 for n in O.FRAGILE}          # consecutive ticks above the dish's strength
        self.broken = {}
        self.peak_force = {n: 0.0 for n in O.FRAGILE}
        self.peak_impact = {n: 0.0 for n in O.FRAGILE}   # m/s: the hardest impact so far, as a fall speed

    def _kinetic(self, b):
        m, d = self.m, self.d
        v6 = np.zeros(6)
        mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, b, v6, 0)
        w, v0 = v6[:3], v6[3:]
        vc = v0 + np.cross(w, d.xipos[b] - d.xpos[b])            # the centre of mass's velocity
        Ri = d.ximat[b].reshape(3, 3)
        wl = Ri.T @ w
        return 0.5 * float(m.body_mass[b]) * float(vc @ vc) + 0.5 * float(wl @ (m.body_inertia[b] * wl))

    def tick(self, t, held):
        m, d = self.m, self.d
        touching, force = set(), {}
        f6 = np.zeros(6)
        for i in range(d.ncon):
            c = d.contact[i]
            for g in (c.geom1, c.geom2):
                n = self.of_body.get(int(m.geom_bodyid[g]))
                if n is not None:
                    touching.add(n)
                    mujoco.mj_contactForce(m, d, i, f6)
                    force[n] = force.get(n, 0.0) + abs(float(f6[0]))
        for n, b in self.body.items():
            f = force.get(n, 0.0)
            self.peak_force[n] = max(self.peak_force[n], f)
            hist = self.energy[n]
            if n in held:
                hist.clear()
                self.loaded[n] = 0
                continue
            e = self._kinetic(b)
            self.loaded[n] = self.loaded[n] + 1 if f > self.strength[n] else 0
            lost = max(0.0, max(hist) - e) if (hist and n in touching) else 0.0
            v_eq = math.sqrt(2.0 * lost / self.mass[n])
            self.peak_impact[n] = max(self.peak_impact[n], v_eq)
            if n not in self.broken:
                if v_eq > FRAGILE["impact_speed"]:
                    self.broken[n] = {"t": round(t, 3), "why": "impact", "impact_m_s": round(v_eq, 3),
                                      "force_N": round(f, 1)}
                elif self.loaded[n] >= LOAD_TICKS:
                    self.broken[n] = {"t": round(t, 3), "why": "load", "impact_m_s": round(v_eq, 3),
                                      "force_N": round(f, 1)}
            hist.append(e)
            if len(hist) > IMPACT_TICKS:
                hist.pop(0)


class Safety:
    """Per-tick safety watch: a violation zeroes the episode's score (the episode goes on)."""

    def __init__(self, m, d, robot, grips: Grips):
        self.m, self.d, self.r = m, d, robot
        self.robot_bodies = set(int(b) for b in robot.body_ids)
        self.hand_bodies = set()
        for side in ("left", "right"):
            self.hand_bodies.update(int(x) for x in robot.hand_bodies[side])
        self.loose = {n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n) for n in O.LOOSE}
        self.released_at = {n: -1e9 for n in O.LOOSE}
        self.count = 0
        self.violations = []
        self.peaks = {"robot_contact_N": 0.0, "object_speed": 0.0, "release_speed": 0.0}
        self.peak_who = {}
        self.g = grips
        grips.on_release.append(self._released)
        self._t = 0.0

    def _released(self, side, name, t, why):
        if name not in self.loose:
            return
        self.released_at[name] = t
        v = np.zeros(6)
        mujoco.mj_objectVelocity(self.m, self.d, mujoco.mjtObj.mjOBJ_BODY, self.loose[name], v, 0)
        sp = float(np.linalg.norm(v[3:]))
        self.peaks["release_speed"] = max(self.peaks["release_speed"], sp)
        if sp > SAFETY["throw_speed"]:
            self._violate(t, f"thrown:{name}", sp)

    def tick(self, t):
        m, d = self.m, self.d
        self._t = t
        per_body = {}
        f6 = np.zeros(6)
        who = {}
        for i in range(d.ncon):
            c = d.contact[i]
            b1, b2 = int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])
            r1, r2 = b1 in self.robot_bodies, b2 in self.robot_bodies
            if not (r1 or r2):
                continue
            rb = b1 if r1 else b2
            if rb in self.hand_bodies:
                continue
            mujoco.mj_contactForce(m, d, i, f6)
            per_body[rb] = per_body.get(rb, 0.0) + abs(float(f6[0]))
            who[rb] = (b1, b2)
        worst = max(per_body.values(), default=0.0)
        if worst > self.peaks["robot_contact_N"]:
            rb = max(per_body, key=per_body.get)
            self.peaks["robot_contact_N"] = worst
            self.peak_who["robot_contact_N"] = (round(t, 2), [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b)
                                                              or "world" for b in who[rb]])
        self.count = self.count + 1 if worst > SAFETY["robot_contact_N"] else 0
        if self.count == SAFETY["robot_contact_ticks"]:
            rb = max(per_body, key=per_body.get)
            names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or "world" for b in who[rb]]
            self._violate(t, "robot_contact:%s/%s" % tuple(names), worst)
        held = self.g.held()
        v = np.zeros(6)
        for n, b in self.loose.items():
            if n in held or t - self.released_at[n] < SAFETY["strike_grace_s"]:
                continue
            mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, b, v, 0)
            sp = float(np.linalg.norm(v[3:]))
            self.peaks["object_speed"] = max(self.peaks["object_speed"], sp)
            if sp > SAFETY["strike_speed"]:
                self._violate(t, f"struck:{n}", sp)
                self.released_at[n] = t           # one violation per strike

    def _violate(self, t, what, val):
        if len(self.violations) < 50:
            self.violations.append({"t": round(t, 3), "what": what, "value": round(float(val), 2)})

    @property
    def ok(self):
        return not self.violations
