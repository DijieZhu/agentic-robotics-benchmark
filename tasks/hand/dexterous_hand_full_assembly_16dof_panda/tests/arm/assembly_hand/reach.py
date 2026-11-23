"""Reach and clearance checks on the built cell's own model (hidden side; the server's reach check): a grasp of the cell's grasp table (grasp_table.json) is FEASIBLE where a unit rests when
the Panda reaches it with an IK solution (posture-aware DLS from several seeds, panda_ik.py) at the grasp pose and
along its approach (10, 30 and 60 mm back along the approach, the hand open to span + `open_extra_mm`), and at every
one of those poses every Panda link and the hand stays `arm_mm` (4 mm) from every other body and each finger
`finger_mm` (2 mm) from every other body but the gripped unit, and `grip_mm` (0.5 mm) from the gripped unit (the
planner's margins). Contacts are read from MuJoCo's own collision on a
scratch copy of the state with the robot geoms' margins raised (restored afterwards).

    nest_reach(cell, max_grasps=24, need=3)     {"units": {unit: {"feasible": n, "tried": k, "first": i}}}
    grasp_world(cell, unit, g)                  the tcp pose (4x4, world) of a table row on the unit as it is
    Checker(cell).pose_ok(q, width, ...)        (ok, min clearance m, culprit)
"""
import json
import math
import os

import numpy as np

from .panda_ik import PandaKin

HERE = os.path.dirname(os.path.abspath(__file__))
_TABLE = None


def table():
    global _TABLE
    if _TABLE is None:
        _TABLE = json.load(open(os.path.join(HERE, "grasp_table.json")))
    return _TABLE


def open_extra_m(kind):
    """how far the hand is open beyond the grasp's span on the approach: 6 mm (3 mm a side); the upper shell's end pinch
    2 mm (its 4 mm section stands between the plate's end and an opening 14 mm wide, narrower than a finger)"""
    return float(table()["kinds"][kind].get("open_extra_mm", 6.0)) * 1e-3


def tcp_frames(c_w, u_w, a_w):
    """the two tcp frames (closing axis = the hand's y, either sign) of a grasp at centre c_w, closing u_w, approach a_w"""
    out = []
    z = np.asarray(a_w, float) / np.linalg.norm(a_w)
    for sg in (1.0, -1.0):
        y = sg * np.asarray(u_w, float)
        y = y - z * float(y @ z)
        y /= np.linalg.norm(y)
        x = np.cross(y, z)
        T = np.eye(4)
        T[:3, :3] = np.column_stack([x, y, z])
        T[:3, 3] = c_w
        out.append(T)
    return out


def grasp_world(cell, unit, g, link=None):
    kind = cell.kit.kind_of(unit)
    ln = link or table()["kinds"][kind]["link"]
    T_l = cell.unit_T(unit) if ln == "body" else cell.link_T(unit, ln)
    c = T_l[:3, :3] @ (np.array(g["centre_mm"], float) * 1e-3) + T_l[:3, 3]
    u = T_l[:3, :3] @ np.array(g["u"], float)
    a = T_l[:3, :3] @ np.array(g["a"], float)
    return c, u, a


class Checker(object):
    def __init__(self, cell, arm_mm=4.0, finger_mm=2.0, grip_mm=0.5):
        import mujoco
        self.mj = mujoco
        self.cell = cell
        self.m = cell.m
        self.d = mujoco.MjData(cell.m)
        self.kin = PandaKin(model=cell.m)
        m = self.m
        self.robot = set(b for b in range(m.nbody) if m.body(b).name.startswith("panda/"))
        self.fingers = set(m.body(n).id for n in ("panda/left_finger", "panda/right_finger"))
        self.arm_m, self.finger_m, self.grip_m = arm_mm * 1e-3, finger_mm * 1e-3, grip_mm * 1e-3
        self.ik_margin = 0.02          # rad from the joint limits along the approach (the solver's default 0.08 at the grasp)
        self.robot_geoms = [g for g in range(m.ngeom) if int(m.geom_bodyid[g]) in self.robot and (m.geom_contype[g] or m.geom_conaffinity[g])]
        self.fq = [m.jnt_qposadr[m.joint("panda/finger_joint%d" % i).id] for i in (1, 2)]

    def sync(self):
        self.d.qpos[:] = self.cell.d.qpos
        self.d.qvel[:] = 0.0

    def pose_ok(self, q, width, skip_bodies=(), grip_bodies=()):
        m, d, mj = self.m, self.d, self.mj
        d.qpos[self.kin.qadr] = q
        for a in self.fq:
            d.qpos[a] = max(0.0, min(0.04, 0.5 * width))
        saved = m.geom_margin[self.robot_geoms].copy()
        try:
            for g in self.robot_geoms:
                m.geom_margin[g] = self.finger_m if int(m.geom_bodyid[g]) in self.fingers else self.arm_m
            mj.mj_forward(m, d)
            worst, culprit = 1.0, None
            for i in range(d.ncon):
                c = d.contact[i]
                b1, b2 = int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])
                r1, r2 = b1 in self.robot, b2 in self.robot
                if r1 == r2:
                    continue
                rb, ob = (b1, b2) if r1 else (b2, b1)
                og = c.geom2 if r1 else c.geom1
                if ob in skip_bodies:
                    continue
                lim = self.finger_m if rb in self.fingers else self.arm_m
                if rb in self.fingers and ob in grip_bodies:
                    lim = self.grip_m
                if float(c.dist) < lim:
                    if float(c.dist) - lim < worst:
                        worst, culprit = float(c.dist) - lim, (m.body(rb).name, m.geom(og).name or m.body(ob).name, round(float(c.dist) * 1e3, 2))
            return worst >= 0.0, worst, culprit
        finally:
            m.geom_margin[self.robot_geoms] = saved

    def grasp_ok(self, T_tcp, span_m, back_mm=(10.0, 30.0, 60.0), q_seed=None, skip_bodies=(), grip_bodies=(), open_extra_m=0.006):
        """(ok, q at the grasp, why): IK at the grasp and along the approach (straight back along the tcp's -z), the
        hand open to span + 6 mm, every pose clear"""
        width = min(0.08, span_m + open_extra_m)
        q_now = np.array([0.0, -0.4, 0.0, -2.3, 0.0, 1.9, 0.785])
        # the IK seeds: the solver's own postures plus four aimed at the target's azimuth (a branch search)
        p0 = T_tcp[:3, 3] - np.array(self.cell.cfg["panda_pos"], float)
        az = math.atan2(p0[1], p0[0])
        extra = [np.array([az, 0.2, 0.0, -2.0, 0.0, 2.2, s]) for s in (-0.785, 0.785)] + \
                [np.array([az, -0.2, 0.0, -2.4, 0.0, 2.0, s]) for s in (-0.785, 0.785)]
        cands = self.kin.candidates(T_tcp, q_seed if q_seed is not None else q_now, extra_seeds=extra)
        if len(cands) < 2:
            # the posture seeds miss branches of a level hand (the thumb's grasp at two corners of the fixture's
            # jitter): 24 more seeds from a fixed stream, inside the middle 70 % of every joint's range
            rng = np.random.default_rng(20261120)
            lo, hi = self.kin.lower, self.kin.upper
            for _k in range(24):
                s0 = lo + (hi - lo) * rng.uniform(0.15, 0.85, 7)
                q, ok = self.kin.solve_one(T_tcp[:3, 3], T_tcp[:3, :3], s0, s0)
                if ok and not any(np.max(np.abs(q - o[1])) < 0.05 for o in cands):
                    cands.append((self.kin.margin(q), q, self.kin.margin(q)))
            cands.sort(key=lambda o: -o[0])
        if not cands:
            return False, None, "no IK"
        why = None
        for _s, q, _mrg in cands[:6]:
            ok, w, cul = self.pose_ok(q, width, skip_bodies, grip_bodies)
            if not ok:
                why = ("grasp", cul)
                continue
            qq, good = q, True
            for b in back_mm:
                T = T_tcp.copy()
                T[:3, 3] = T_tcp[:3, 3] - T_tcp[:3, 2] * b * 1e-3
                q2, ok2 = self.kin.solve_one(T[:3, 3], T[:3, :3], qq, qq, margin=self.ik_margin)
                if not ok2:
                    good, why = False, ("approach IK at %.0f mm" % b, None)
                    break
                ok3, _w, cul = self.pose_ok(q2, width, skip_bodies, grip_bodies)
                if not ok3:
                    good, why = False, ("approach at %.0f mm" % b, cul)
                    break
                qq = q2
            if good:
                return True, q, None
        return False, None, why


def nest_reach(cell, max_grasps=24, need=3, units=None, checker=None):
    ck = checker or Checker(cell)
    ck.sync()
    out = {}
    for u in (units or cell.loose):
        kind = cell.kit.kind_of(u)
        rows = table()["kinds"][kind]["grasps"][:max_grasps]
        n, first, tried, whys = 0, None, 0, []
        for i, g in enumerate(rows):
            c, uu, a = grasp_world(cell, u, g)
            tried += 1
            hit = False
            for T in tcp_frames(c, uu, a):
                ok, _q, why = ck.grasp_ok(T, float(g["span_mm"]) * 1e-3, grip_bodies=set(cell.bodies_of[u]),
                                          open_extra_m=open_extra_m(kind))
                if ok:
                    hit = True
                    break
                whys.append(why)
            if hit:
                n += 1
                if first is None:
                    first = i
                if n >= need:
                    break
        out[u] = {"feasible": n, "tried": tried, "first": first,
                  "why_not": [str(w) for w in whys[:3]] if n < need else []}
    return {"units": out}
