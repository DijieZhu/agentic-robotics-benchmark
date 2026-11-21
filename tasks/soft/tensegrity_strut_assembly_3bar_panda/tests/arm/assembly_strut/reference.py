"""The Panda assembles a strut on the cradle: the privileged reference program (no observation).
After the quadruped cell's reference (slot x candidate coupling, the bench plan's grasp first) and the SO-101 one
(the Panda, its IK and planner).

Privileged state only: the modules' exact poses, the slots' targets (`StrutCell.sleeve_pose` / `design_world`: the
strut core as it is now), the Panda's own kinematics (`panda_ik.PandaKin`) and a planner on a copy of the cell
(`motion.Planner`). Nothing but the Panda, its stock flat pads and the physics moves a module: no jig, no position
lock, no collision switched off. The order is the guide's (`requires`): both motor modules, the lid, then the end
caps. The modules start where the bench plan put them (`bench_plan.json`: which unit in which nest,
planned in reverse order, with the grasp that works there and at the slot).

Per step:
  grasps   the bench plan's grasp first, then the module's own candidates (`grasps.py`: antipodal pinches
           of the pads on its collision pieces, 150 kept); each checked at the pick (EVERY IK solution:
           pre-grasp 80 mm back along the approach, the approach line, the fingers open to the width + 16 mm, then
           closing at three widths; the planner's clearances) and at the slot (`place_checks`: the pre-insertion pose
           and inwards in 2 mm steps on one IK branch, from the outside in, the fingers closed to the
           grasp's width keeping 1 mm, the links 3 mm, from the strut and the modules on it)
  pick     plan to the pre-grasp, approach line, close to the width - 12 mm (15 N per finger), lift 50 mm; the module
           must stay in the hand (3 mm / 4 deg)
  motor    (a) a motor module lies in its nest 6-16 deg off its mounting turn: picked with the hand tilted by that
           offset, turned level about its own axis after the lift (`level`); carried (planned) to the pre-insertion
           pose: its bore coaxial with the rod, the leading mouth 10 mm out of the rod end; pushed along the rod at
           10 mm/s at stiffness 0.5, closed loop on its live pose in the hand, past the capture to the design axial
           position and turn; held still there 0.6 s and until the cell fastens it; let go; straight up
  lid      (b) lowered from 30 mm above the seat along the seat direction, closed loop, to home; held still until
           fastened; let go; straight up
  cap      (c) its turn is free: the target turn is the one that keeps the hand straight down; coaxial with the rod
           end, pushed along the rod until the rod end meets the bore's bottom (CAP_PRESS_M past it); held still until
           fastened; let go; straight up
  stall    the module more than 4 mm behind the command along the insertion for 0.3 s: back off 15 mm, re-align,
           retry (twice) at a lower stiffness
A module the arm cannot place stays where it is (`placed_by: nobody`); no fallback. A contact monitor logs every
contact above 0.5 N of the Panda with anything but the module in its fingers (and the module being grasped):
`unplanned_contacts`; and of the module in work with anything but its strut core, the hand and the tray (while it is
picked): `unit_contacts`, except that once the module is fastened its contacts with the other modules fastened on the
same strut core are `cluster_contacts` (a fastening's preload: an end cap's mouth face and its motor module's outer
end face are coplanar at the design, module z 131.40 mm, and rest against each other). With the server (`server=`), every tick's readings of the four rules R1-R4 are logged
(`force_log`): the reference's force record that fixes the rules' numbers.
"""
import math
import time

import numpy as np

from .coupling import _inv, quat_wxyz, rotation_angle

GRIP_SQUEEZE_M = 0.012          # 2500 N/m x 12 mm / 2 = 15 N per finger (the precedent; the grasp trials)
OPEN_EXTRA_M = 0.016            # the fingers open to the grasp's width + this for the approach (nests 70 mm apart)
RELEASE_EXTRA_M = 0.006         # ... and to the width + this to let a fastened module go
PREGRASP_M = 0.08
LIFT_M = 0.05
PRE_ROD_M = 0.010               # the leading bore mouth 10 mm out of the rod end
PRE_SEAT_M = 0.030              # the lid 30 mm above its seat
INSERT_SPEED_M_S = 0.010
INSERT_STIFFNESS = 0.5
CARRY_SPEED_M_S = 0.5           # the tcp's speed cap on planned moves (the cell's rule R4: 3 m/s)
APPROACH_SPEED_M_S = 0.03
HOLD_S = 0.6                    # a motor module held still at its design place before the hand lets go
LOCK_WAIT_S = 3.0
CAP_PRESS_M = 0.0002            # an end cap commanded this far past the rod end meeting its bore's bottom
STALL_M = 0.004
STALL_TICKS = 30
RETRIES = 2
HELD_TOL_M, HELD_TOL_RAD = 0.003, math.radians(4.0)
FROM_ABOVE = -0.85              # the approach's z in the world at the pick: within ~32 deg of straight down
DOWN = np.array([0.0, 0.0, -1.0])
DEFAULT_STEPS = (("s2_motor_p", "M1_1"), ("s2_motor_n", "M1_2"), ("s2_lid", "M2_1"), ("s2_cap_p", "M3_1"), ("s2_cap_n", "M3_2"))
# the three-strut cell without a plan: strut by strut (s1, s2, s3), the guide's order, strut k's units (layout's nominal)
DEFAULT_STEPS_FULL = tuple(step for k in (1, 2, 3) for step in (
    ("s%d_motor_p" % k, "M1_%d" % (2 * k - 1)), ("s%d_motor_n" % k, "M1_%d" % (2 * k)), ("s%d_lid" % k, "M2_%d" % k),
    ("s%d_cap_p" % k, "M3_%d" % (2 * k - 1)), ("s%d_cap_n" % k, "M3_%d" % (2 * k))))
_GRASP_CACHE = {}               # (kit dir, build, extra approach angles, seat direction) -> candidates, for the process
ENGAGED = ("ENGAGED", "HOLDING", "LOCKED", "LOCKED_OFF")
FASTENED = ("LOCKED", "LOCKED_OFF")


def _T(p, R):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = p
    return T


def _rot_about(axis, ang):
    from scipy.spatial.transform import Rotation
    a = np.asarray(axis, float)
    return Rotation.from_rotvec(a / np.linalg.norm(a) * float(ang)).as_matrix()


class Assembler(object):
    def __init__(self, cell, plan=None, log=print, server=None, monitor=True, wall_cap_s=None, snap=None):
        from .motion import Planner
        from .panda_ik import PandaKin
        self.cell, self.m, self.d, self.ct, self.sm = cell, cell.m, cell.d, cell.ctrl, cell.sm
        self.kit, self.log, self.srv = cell.kit, log, server
        self.pl = Planner(cell)
        self.kin = PandaKin(model=self.pl.m)
        self.max_width = float(self.ct.max_width)
        self.q_ready = np.array(cell.cfg["robot_home_q"], float)
        self.plan = dict(plan or {})
        self.plan_grasp = {st["unit"]: st.get("grasp") for st in self.plan.get("steps", []) if st.get("grasp")}
        self.work = {"unit": None, "slot": None, "phase": "start", "held": None}
        self.contacts, self.unit_contacts, self.cluster_contacts = [], [], []
        self.panda = set(b for b in range(self.m.nbody) if self.m.body(b).name.startswith("panda/"))
        self.t0_wall = time.time()
        self.wall_cap = None if wall_cap_s is None else float(wall_cap_s)
        self._grasps = {}
        self.snap = snap
        self.pull = {}
        self.flog = {"t": [], "slot": [], "phase": [], "R1_N": [], "R1_on": [], "R2_N": [], "R2_pair": [], "R3_N": [], "R3_where": [],
                     "R4_m_s": [], "pad_N": []}
        if monitor:
            cell.hooks.append(self._monitor)
        if server is not None:
            server.reading_hooks.append(self._force_hook)

    # ------------------------------------------------------------------ state
    def tcp_T(self):
        p, R = self.ct.tcp_actual()
        return _T(p, R)

    def unit_T(self, u):
        return self.cell.body_T(u)

    def G_act(self, u):
        """the unit's pose in the tcp frame, now"""
        return _inv(self.tcp_T()) @ self.unit_T(u)

    def out_of_time(self):
        return self.wall_cap is not None and time.time() - self.t0_wall > self.wall_cap

    def _monitor(self):
        """every tick: contacts above 0.5 N of the Panda with anything but the unit it holds / grasps, and of the unit
        in work with anything but its strut core (the coupling's), the hand and the tray (while it is picked)"""
        m, d = self.m, self.d
        w = self.work
        self.cell.label["text"] = "%s  %s  %s" % (w.get("slot") or "", w.get("unit") or "", w.get("phase") or "")
        u = w.get("unit")
        ub = m.body(u).id if u else -1
        sid = w.get("slot")
        if sid and sid in self.sm.pull:
            self.pull[sid] = max(self.pull.get(sid, 0.0), float(self.sm.pull[sid]["dist"]))
        carrier = m.body(self.cell.carriers[self.cell.slot(sid)["strut"]]).id if sid else -1
        f6 = np.zeros(6)
        for i in range(d.ncon):
            c = d.contact[i]
            b1, b2 = int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])
            r1, r2 = b1 in self.panda, b2 in self.panda
            if r1 and r2:
                continue
            kind = None
            if r1 or r2:
                other = b2 if r1 else b1
                if other == ub:
                    continue
                kind = "robot"
            elif ub in (b1, b2):
                other = b2 if b1 == ub else b1
                nm = m.body(other).name
                if other == carrier or (w["phase"] in ("plan", "to-pick", "pick", "close", "lift") and nm.startswith("tray")):
                    continue
                kind = "unit"
                if sid and self.sm.rec[sid]["fastened"] and nm in self._fastened_on(self.cell.slot(sid)["strut"]):
                    kind = "cluster"          # two modules fastened on one strut core resting against each other
            else:
                continue
            self.cell.mj.mj_contactForce(m, d, i, f6)
            if abs(f6[0]) < 0.5:
                continue
            n1, n2 = m.body(b1).name, m.body(b2).name
            lst = {"robot": self.contacts, "unit": self.unit_contacts, "cluster": self.cluster_contacts}[kind]
            key = (w["phase"], sid, n1, n2)
            if lst and lst[-1]["key"] == key:
                lst[-1]["force_N"] = max(lst[-1]["force_N"], round(float(abs(f6[0])), 2))
                lst[-1]["t1"] = round(float(d.time), 2)
                continue
            lst.append({"key": key, "t": round(float(d.time), 2), "t1": round(float(d.time), 2), "phase": w["phase"], "slot": sid,
                        "bodies": [n1, n2], "geoms": [m.geom(c.geom1).name, m.geom(c.geom2).name], "force_N": round(float(abs(f6[0])), 2)})

    def _fastened_on(self, strut):
        return set(r["unit"] for sid, r in self.sm.rec.items() if r["fastened"] and r["unit"] and self.cell.slot(sid)["strut"] == strut)

    def _force_hook(self, srv):
        r = srv.readings
        f = self.flog
        f["t"].append(round(float(srv.data.time), 3))
        f["slot"].append(self.work.get("slot") or "-")
        f["phase"].append(self.work.get("phase") or "-")
        f["R1_N"].append(float(r.get("R1_N", 0.0)))
        p = r.get("R1_pair")
        f["R1_on"].append(p[1] if p else None)
        f["R2_N"].append(float(r.get("R2_N", 0.0)))
        p = r.get("R2_pair")
        f["R2_pair"].append(list(p) if p else None)
        f["R3_N"].append(float(r.get("R3_N", 0.0)))
        p = r.get("R3_where")
        f["R3_where"].append(list(p) if p else None)
        f["R4_m_s"].append(float(r.get("R4_m_s", 0.0)))
        f["pad_N"].append(float(r.get("pad_N", 0.0)))

    # ------------------------------------------------------------------ motion primitives
    def tick(self, n=1):
        for _ in range(n):
            self.cell.tick()

    def wait_arrived(self, tol=0.002, max_ticks=400):
        ct = self.ct
        for _ in range(max_ticks):
            self.cell.tick()
            if (np.max(np.abs(ct.q_goal - ct.q_cmd)) < 1e-9 and np.max(np.abs(ct.q() - ct.q_cmd)) < tol and np.max(np.abs(ct.dq())) < 0.02):
                return True
        return False

    def move_q(self, q, tol=0.002):
        self.ct.set_mode("joint")
        self.ct.command_joint(np.asarray(q, float))
        return self.wait_arrived(tol=tol)

    def follow(self, path, v_max=CARRY_SPEED_M_S):
        """a planned joint path, every segment in per-tick steps on the checked straight segment, the tcp no faster than
        v_max (the quadruped reference's `follow`)"""
        ct = self.ct
        ct.set_mode("joint")
        dq_max = 1.5 * ct.tick_s
        dp_max = float(v_max) * ct.tick_s
        qa = ct.q_cmd.copy()
        pa = self.pl.tcp_T(qa)[:3, 3]
        for qb in path[1:]:
            qb = np.asarray(qb, float)
            n = max(1, int(math.ceil(float(np.max(np.abs(qb - qa) / dq_max)) - 1e-9)))
            q_prev, p_prev = qa, pa
            for k in range(1, n + 1):
                q = qa + (qb - qa) * (k / n)
                p = self.pl.tcp_T(q)[:3, 3]
                m_ = max(1, int(math.ceil(float(np.linalg.norm(p - p_prev)) / dp_max - 1e-9)))
                for j in range(1, m_ + 1):
                    ct.command_joint(q_prev + (q - q_prev) * (j / m_))
                    self.cell.tick()
                q_prev, p_prev = q, p
            qa, pa = qb, p_prev
        return self.wait_arrived()

    def go(self, q, held=None, allow=(), width=None):
        """planned free-space move to q (the held unit carried as held now); via the ready posture if no direct path"""
        self.pl.sync(held=held, allow=allow, width=width)
        q = np.asarray(q, float)
        path = self.pl.plan(self.ct.q(), q)
        if path is None and np.max(np.abs(q - self.q_ready)) > 1e-3:
            p1 = self.pl.plan(self.ct.q(), self.q_ready)
            if p1 is not None:
                p2 = self.pl.plan(self.q_ready, q)
                if p2 is not None:
                    path = p1 + p2[1:]
        if path is None:
            return False
        return self.follow(path)

    def line_qs(self, T1, step_m=0.002, T0=None, q0=None, step_rad=math.radians(1.0)):
        """IK postures every step_m (and step_rad) along the straight tcp line T0 -> T1 (orientation slerped), one branch"""
        from scipy.spatial.transform import Rotation, Slerp
        T0 = self.tcp_T() if T0 is None else T0
        q = self.ct.q() if q0 is None else np.asarray(q0, float)
        n = max(1, int(math.ceil(max(np.linalg.norm(T1[:3, 3] - T0[:3, 3]) / step_m, rotation_angle(T0[:3, :3].T @ T1[:3, :3]) / step_rad))))
        sl = Slerp([0, 1], Rotation.from_matrix([T0[:3, :3], T1[:3, :3]]))
        qs = [q]
        for k in range(1, n + 1):
            s = k / n
            p = (1 - s) * T0[:3, 3] + s * T1[:3, 3]
            R = sl([s]).as_matrix()[0]
            q2, ok = self.kin.solve_one(p, R, q, q, margin=0.0)
            if not ok or np.max(np.abs(q2 - q)) > 0.3:
                return None
            qs.append(q2)
            q = q2
        return qs

    def move_line(self, T1, speed=APPROACH_SPEED_M_S, held=None, allow=(), check=True, width=None, ang_speed=math.radians(20.0)):
        """a Cartesian line to T1 (checked on the planner first); the controller's Cartesian mode streams the line"""
        T0 = self.tcp_T()
        if check:
            qs = self.line_qs(T1, T0=T0)
            if qs is None:
                return False, "line IK"
            self.pl.sync(held=held, allow=allow, width=width)
            if not self.pl.line_free(qs[::3] + [qs[-1]]):
                return False, "line blocked (%s)" % self.pl.why
        from scipy.spatial.transform import Rotation, Slerp
        ct = self.ct
        ct.set_mode("cartesian")
        dist = float(np.linalg.norm(T1[:3, 3] - T0[:3, 3]))
        ang = rotation_angle(T0[:3, :3].T @ T1[:3, :3])
        n = max(1, int(math.ceil(max(dist / speed, ang / ang_speed) / self.cell.tick_s)))
        sl = Slerp([0, 1], Rotation.from_matrix([T0[:3, :3], T1[:3, :3]]))
        for k in range(1, n + 1):
            s = k / n
            ct.command_cartesian((1 - s) * T0[:3, 3] + s * T1[:3, 3], quat_wxyz(sl([s]).as_matrix()[0]))
            self.cell.tick()
        self.tick(30)
        ct.set_mode("joint")
        return True, ""

    def grip(self, width, settle_ticks=60):
        self.ct.command_width(float(min(self.max_width, max(0.0, width))))
        last = None
        for i in range(400):
            self.cell.tick()
            w = self.ct.width()
            if i > 20 and abs(self.ct.width_goal - self.ct.width_cmd) < 1e-6 and last is not None and abs(w - last) < 2e-5:
                break
            last = w
        self.tick(settle_ticks)
        return self.ct.width()

    # ------------------------------------------------------------------ targets
    def free_roll(self, sid, G, s=None):
        """an end cap's roll about the rod (free) that keeps the hand of grasp G straight down at depth s"""
        x = self.cell.slot(sid)
        s = x["home"] if s is None else s
        A = self.sm.rail_base_world(x)
        v = (self.sm.canonical(x, False) @ G)[:3, 2]            # the approach in the rail's roll frame at roll 0
        dn = A[:3, :3].T @ DOWN                                 # straight down in the rail frame
        return math.atan2(dn[1], dn[0]) - math.atan2(v[1], v[0])

    def home_pose(self, sid, G):
        """(the unit's pose at home, the rail roll) for grasp G: a motor module at its design turn (roll 0), an end cap
        at the free roll that keeps the hand down, the lid at its design pose"""
        x = self.cell.slot(sid)
        if x["kind"] == "rod_sleeve":
            roll = 0.0 if x["tol_roll"] is not None else self.free_roll(sid, G)
            return self.cell.sleeve_pose(sid, x["home"], roll), roll
        return self.cell.design_world(sid), None

    def seat_ins(self, sid):
        """the lid's insertion direction in the world (into the seat)"""
        x = self.cell.slot(sid)
        return self.cell.body_T(self.cell.carriers[x["strut"]])[:3, :3] @ x["ins"]

    def path_pose(self, sid, roll, back):
        """the unit `back` (m) short of home along its insertion (rod: the leading mouth's depth home - back)"""
        x = self.cell.slot(sid)
        if x["kind"] == "rod_sleeve":
            return self.cell.sleeve_pose(sid, x["home"] - back, roll)
        T = self.cell.design_world(sid).copy()
        T[:3, 3] -= self.seat_ins(sid) * back
        return T

    def pre_back(self, sid):
        x = self.cell.slot(sid)
        return (x["home"] + PRE_ROD_M) if x["kind"] == "rod_sleeve" else PRE_SEAT_M

    # ------------------------------------------------------------------ grasp candidates
    def candidates(self, u, sid):
        """[(G, w, desc)] the bench plan's grasp first, then the unit's own candidates (computed once, lazily)"""
        out = []
        hint = self.plan_grasp.get(u)
        if hint is not None:
            out.append((np.array(hint["G"], float), float(hint["w"]), hint.get("desc", "") + " (bench plan)"))
        return out, (lambda: self._own_candidates(u, sid))

    def _own_candidates(self, u, sid, extra_from_pose=True):
        """the unit's own grasp candidates for slot sid (grasps.unit_grasps); they depend only on the unit's build and
        the extra approach angle (a motor module: straight down at the slot's design; an end cap: straight down where
        it rests now, unless `extra_from_pose` is False: the 10 deg grid only) and the lid's seat direction, so they
        are cached per (kit, build, angle, seat) for the process (the three struts' modules share them)"""
        if u not in self._grasps or not extra_from_pose:
            from .grasps import unit_grasps
            x = self.cell.slot(sid)
            t = self.kit.units[u]["type"]
            ang = ()
            if t == "M1":
                R = self.cell.sleeve_pose(sid, x["home"], 0.0)[:3, :3]
                dn = R.T @ DOWN
                ang = (round(math.degrees(math.atan2(dn[1], dn[0])), 6),)
            elif t == "M3" and extra_from_pose:
                dn = self.unit_T(u)[:3, :3].T @ DOWN
                ang = (round(math.degrees(math.atan2(dn[1], dn[0])), 6),)
            seat_u = None
            if t == "M2":
                seat_u = -np.asarray(x["ins"], float)
            b = self.kit.units[u]["build"]
            key = (self.kit.dir, b, ang, None if seat_u is None else tuple(np.round(seat_u, 9)))
            if key not in _GRASP_CACHE:
                t0 = time.time()
                _GRASP_CACHE[key] = unit_grasps(self.kit, [u], seat_u=seat_u, extra_angles_deg={u: list(ang)})[u]
                self.log("  grasp candidates of %s (%s): %d in %.0f s" % (u, b, len(_GRASP_CACHE[key]), time.time() - t0))
            if not extra_from_pose:
                return _GRASP_CACHE[key]
            self._grasps[u] = _GRASP_CACHE[key]
        return self._grasps[u]

    # ------------------------------------------------------------------ checks
    def pick_checks(self, u, T_u, G, w, poses=None):
        """(q_pick, q_pg, margin) for grasp G of the unit resting at T_u, every IK solution tried; else (None, None, why)"""
        P = T_u @ G
        if P[2, 2] > FROM_ABOVE:
            return None, None, "approach not from above"
        P_pg = P.copy()
        P_pg[:3, 3] -= P[:3, 2] * PREGRASP_M
        sols = self.kin.candidates(P, self.q_ready)
        if not sols:
            return None, None, "pick IK"
        width = min(self.max_width, w + OPEN_EXTRA_M)
        P_ = dict(poses or {})
        P_[u] = T_u
        why = "pre-grasp IK"
        for _sc, q_pick, mrg in sols:
            q_pg, ok = self.kin.solve_one(P_pg[:3, 3], P_pg[:3, :3], q_pick, q_pick, margin=0.0)
            if not ok or np.max(np.abs(q_pg - q_pick)) > 1.0:
                continue
            self.pl.sync(allow=(u,), poses=P_, width=width)
            hs = []
            qk = q_pg
            for k in range(6):
                pk = (1 - k / 5) * P_pg[:3, 3] + k / 5 * P[:3, 3]
                qk, ok = self.kin.solve_one(pk, P[:3, :3], qk, qk, margin=0.0)
                hs = self.pl.hits(qk) if ok else [("ik", "ik", 0.0)]
                if hs:
                    break
            if hs:
                why = "hits %s on the approach" % sorted(set((h[0], h[1]) for h in hs))[:3]
                continue
            for wk in (width, 0.5 * (width + w), w):
                self.pl.sync(allow=(u,), poses=P_, width=wk)
                hs = self.pl.hits(q_pick)
                if hs:
                    break
            if hs:
                why = "hits %s while closing" % sorted(set((h[0], h[1]) for h in hs))[:3]
                continue
            return q_pick, q_pg, float(mrg)
        return None, None, why

    def place_checks(self, sid, u, G, w, poses=None, q_seed=None, step=0.002):
        """(q_pre, hold, margin) for grasp G at the slot: the pre-insertion pose and inwards in 2 mm steps on one IK
        branch, every branch tried; hold = how far short of home the fingers can follow (a rod module: must be 0, the
        hand carries it home; the lid: within the seat's capture); else (None, why, 0)"""
        x = self.cell.slot(sid)
        _T_home, roll = self.home_pose(sid, G)
        pre = self.pre_back(sid)
        need = 0.0 if x["kind"] == "rod_sleeve" else self.sm.seat_depth - 0.0005
        T_pre = self.path_pose(sid, roll, pre)
        sols = self.kin.candidates(T_pre @ G, self.q_ready if q_seed is None else q_seed)
        if not sols:
            return None, "pre-insertion IK", 0.0
        P_ = dict(poses or {})
        best, why = None, ""
        xs = list(np.arange(pre, -1e-9, -step))
        if xs[-1] > 1e-9:
            xs.append(0.0)
        for _sc, q0, mrg in sols:
            q, hold = q0, None
            for i, back in enumerate(xs):
                T_x = self.path_pose(sid, roll, back)
                P_[u] = T_x
                if i == 0:
                    qx, ok = q0, True
                else:
                    qx, ok = self.kin.solve_one((T_x @ G)[:3, 3], (T_x @ G)[:3, :3], q, q, margin=0.0)
                    ok = ok and np.max(np.abs(qx - q)) < 0.3
                hit = []
                if ok:
                    self.pl.sync(allow=(u,), poses=P_, width=w)
                    hit = self.pl.hits(qx, finger_clear=0.001 if i else 0.005, link_clear=0.003 if i else None)
                if not ok or hit:
                    why = ("IK at %.0f mm" % (back * 1e3)) if not ok else ("hits %s at %.0f mm" % (sorted(set((h[0], h[1]) for h in hit))[:3], back * 1e3))
                    hold = float(xs[i - 1]) if i > 0 else None
                    break
                q = qx
            else:
                hold = 0.0
            if hold is not None and hold <= need + 1e-9 and (best is None or (hold, -mrg) < (best[1], -best[2])):
                best = (q0, hold, float(mrg))
            if best is not None and best[1] == 0.0:
                break
        if best is None:
            return None, why or "insertion", 0.0
        return best

    def feasible_grasp(self, sid, u, poses_pick=None, poses_place=None):
        """the first grasp (the bench plan's first) that passes the pick checks where the unit rests now and the place
        checks at its slot: (G, w, desc, q_pick, q_pg, q_pre, hold) or (None, rejections)"""
        T_pick = self.unit_T(u)
        first, rest = self.candidates(u, sid)
        rejected = {}
        for batch in (first, None):
            cands = first if batch is not None else rest()
            for G, w, desc in cands:
                q_pick, q_pg, r1 = self.pick_checks(u, T_pick, G, w, poses=poses_pick)
                if q_pick is None:
                    rejected[desc] = "pick: %s" % r1
                    continue
                q_pre, hold, r2 = self.place_checks(sid, u, G, w, poses=poses_place, q_seed=q_pick)
                if q_pre is None:
                    rejected[desc] = "place: %s" % hold
                    continue
                return (G, w, desc, q_pick, q_pg, q_pre, hold), rejected
        return None, rejected

    # ------------------------------------------------------------------ one step
    def place_unit(self, sid, u):
        cell, sm = self.cell, self.sm
        rec = {"slot": sid, "unit": u, "t0_s": round(float(self.d.time), 2), "placed_by": "nobody", "attempts": []}
        self.work.update({"unit": u, "slot": sid, "phase": "plan", "held": None})
        t_wall = time.time()
        for round_ in range(3):
            if self.out_of_time():
                rec["why"] = "wall-clock cap"
                break
            feas, rejected = self.feasible_grasp(sid, u)
            if feas is None:
                rec["why"] = "no feasible grasp (%s)" % (sorted(set(str(v) for v in rejected.values()))[:4],)
                self.log("  %s %s: %s" % (sid, u, rec["why"]))
                break
            G, w, desc, q_pick, q_pg, q_pre, hold = feas
            att = {"round": round_, "grasp": desc, "width_mm": round(w * 1e3, 2), "hold_mm": round(hold * 1e3, 1),
                   "rejected_before": len(rejected)}
            rec["attempts"].append(att)
            ok, why = self.pick(u, G, w, q_pick, q_pg)
            if not ok:
                att["result"] = why
                self.log("  %s %s: pick failed (%s)" % (sid, u, why))
                self.let_go(w)
                continue
            if self.snap:
                self.snap("%s_picked" % sid, u)
            ok, why = self.insert(sid, u, w, att, q_branch=q_pre)
            att["result"] = why
            if ok:
                rec["placed_by"] = "panda"
                break
            self.log("  %s %s: insertion failed (%s)" % (sid, u, why))
            if self.work["held"] is not None:
                self.let_go(w)
            break
        st = sm.rec[sid]
        rec.update({"state": sm.state[sid], "fastened": st["fastened"], "achieved": st["achieved"], "axial_err_mm": st["axial_err_mm"],
                    "roll_err_deg": st["roll_err_deg"], "pos_err_mm": st["pos_err_mm"], "ang_err_deg": st["ang_err_deg"],
                    "why_off": st["why_off"], "capture": st.get("capture"), "pull_mm": round(self.pull.get(sid, 0.0) * 1e3, 3),
                    "t1_s": round(float(self.d.time), 2), "wall_s": round(time.time() - t_wall, 1)})
        self.log("  %-11s %-5s -> %s (%s) sim %.1f s wall %.0f s" % (sid, u, sm.state[sid], rec["attempts"][-1].get("result") if rec["attempts"] else "-",
                                                                rec["t1_s"] - rec["t0_s"], rec["wall_s"]))
        return rec

    def let_go(self, w):
        """a unit the arm could not place stays where it is: open, back out along the approach, ready posture"""
        self.work.update({"phase": "let go", "held": None})
        self.grip(min(self.max_width, w + OPEN_EXTRA_M), settle_ticks=10)
        T_out = self.tcp_T()
        T_out[:3, 3] -= T_out[:3, 2] * 0.06
        self.move_line(T_out, check=False)
        if not self.go(self.q_ready):
            self.move_q(self.q_ready)

    def pick(self, u, G, w, q_pick, q_pg):
        wo = min(self.max_width, w + OPEN_EXTRA_M)
        self.work.update({"phase": "to-pick", "held": None})
        self.grip(wo, settle_ticks=5)
        if not self.go(q_pg, width=wo):
            return False, "no path to the pre-grasp (%s)" % self.pl.why
        self.work["phase"] = "pick"
        P = self.unit_T(u) @ G
        ok, why = self.move_line(P, allow=(u,), width=wo)
        if not ok:
            return False, "approach: " + why
        self.work["phase"] = "close"
        self.grip(max(0.0, w - GRIP_SQUEEZE_M))
        G0 = self.G_act(u)
        self.work.update({"phase": "lift", "held": u})
        T_up = self.tcp_T()
        T_up[2, 3] += LIFT_M
        ok, why = self.move_line(T_up, held=(u, G0), check=True)
        if not ok:
            self.move_line(T_up, held=(u, G0), check=False)
        G1 = self.G_act(u)
        dp = float(np.linalg.norm(G1[:3, 3] - G0[:3, 3]))
        da = rotation_angle(G1[:3, :3].T @ G0[:3, :3])
        if dp > HELD_TOL_M or da > HELD_TOL_RAD or self.ct.width() < 0.002:
            self.work["held"] = None
            return False, "the unit moved in the hand (%.1f mm, %.1f deg)" % (dp * 1e3, math.degrees(da))
        return True, ""

    def level(self, sid, u):
        """(a) a motor module picked off its mounting turn: turned about its own axis back to its design turn"""
        x = self.cell.slot(sid)
        T_u = self.unit_T(u)
        a = T_u[:3, 2]
        R_h = self.cell.sleeve_pose(sid, x["home"], 0.0)[:3, :3]
        if float(R_h[:, 2] @ a) < 0:                            # the n end: the module turns half a turn about the vertical
            R_h = _rot_about([0, 0, 1.0], math.pi) @ R_h
        x_now, x_h = T_u[:3, 0], R_h[:, 0]
        ang = math.atan2(float(np.cross(x_now, x_h) @ a), float(x_now @ x_h))
        if abs(ang) < math.radians(0.3):
            return True, 0.0
        Rot = _rot_about(a, ang)
        T_tcp = self.tcp_T()
        piv = T_u[:3, 3] + a * float((T_tcp[:3, 3] - T_u[:3, 3]) @ a)          # the bore axis point by the tcp
        T_new = np.eye(4)
        T_new[:3, :3] = Rot @ T_tcp[:3, :3]
        T_new[:3, 3] = piv + Rot @ (T_tcp[:3, 3] - piv)
        self.work["phase"] = "level"
        ok, why = self.move_line(T_new, held=(u, self.G_act(u)), check=True, speed=0.02, ang_speed=math.radians(15.0))
        return ok, math.degrees(ang)

    def align(self, sid, u, roll, back, iters=4, tol_m=0.0002, tol_rad=math.radians(0.2)):
        """closed loop at a pose on the insertion path: the unit `back` short of home, its pose error read from the
        privileged state and corrected through the hand; returns the remaining (mm, deg)"""
        T_t = self.path_pose(sid, roll, back)
        e_p = e_a = None
        for _ in range(iters):
            E = _inv(T_t) @ self.unit_T(u)
            e_p, e_a = float(np.linalg.norm(E[:3, 3])), rotation_angle(E[:3, :3])
            if e_p < tol_m and e_a < tol_rad:
                break
            T_cmd = T_t @ _inv(self.G_act(u))
            self.move_line(T_cmd, check=False, speed=0.01, ang_speed=math.radians(5.0))
        E = _inv(T_t) @ self.unit_T(u)
        return round(float(np.linalg.norm(E[:3, 3])) * 1e3, 3), round(math.degrees(rotation_angle(E[:3, :3])), 3)

    def insert(self, sid, u, w, att, q_branch=None):
        """carry to the pre-insertion, align, in along the insertion closed loop on the live pose, hold until
        fastened, let go, back out"""
        cell, ct, sm = self.cell, self.ct, self.sm
        x = cell.slot(sid)
        if x["kind"] == "rod_sleeve" and x["tol_roll"] is not None:
            ok, ang = self.level(sid, u)
            att["level_deg"] = round(ang, 2)
            if not ok:
                return False, "level: blocked"
        G = self.G_act(u)                                   # the unit in the tcp frame, as held now
        _T_home, roll = self.home_pose(sid, _inv(G))        # home_pose takes the grasp: the tcp in the unit frame
        att["roll_target_deg"] = None if roll is None else round(math.degrees(roll), 2)
        pre = self.pre_back(sid)
        T_t = self.path_pose(sid, roll, pre) @ _inv(G)
        sols = self.kin.candidates(T_t, ct.q())
        if q_branch is not None:
            q_v, ok_v = self.kin.solve_one(T_t[:3, 3], T_t[:3, :3], q_branch, q_branch, margin=0.0)
            if ok_v:
                sols = [(0.0, q_v, 0.0)] + [s for s in sols if np.max(np.abs(s[1] - q_v)) > 1e-3]
        self.work["phase"] = "carry"
        moved = False
        for _sc, q_pre, _m in sols[:6]:
            if self.go(q_pre, held=(u, G), width=w):
                moved = True
                break
        if not moved:
            return False, "no path to the pre-insertion (%s)" % self.pl.why
        self.work["phase"] = "align"
        att["align_at_pre"] = self.align(sid, u, roll, pre)
        if self.snap:
            self.snap("%s_pre" % sid, u)
        self.work["phase"] = "insert"
        stiff = INSERT_STIFFNESS
        back = pre
        end = -CAP_PRESS_M if (x["kind"] == "rod_sleeve" and x["tol_roll"] is None) else 0.0
        for attempt in range(RETRIES + 1):
            ct.set_stiffness(stiff)
            ct.set_mode("cartesian")
            stall = 0
            for _ in range(int((back - end + 0.02) / (INSERT_SPEED_M_S * cell.tick_s)) + 100):
                back = max(end, back - INSERT_SPEED_M_S * cell.tick_s)
                T_u = self.path_pose(sid, roll, back)
                T_cmd = T_u @ _inv(self.G_act(u))
                ct.command_cartesian(T_cmd[:3, 3], quat_wxyz(T_cmd[:3, :3]))
                cell.tick()
                lag = self._lag(sid, u, roll, back)
                stall = stall + 1 if lag > STALL_M else 0
                if stall > STALL_TICKS or back <= end:
                    break
            if stall <= STALL_TICKS:
                break
            att.setdefault("stalls", []).append(round(back * 1e3, 1))
            back = min(pre, back + 0.015)
            self.work["phase"] = "back off"
            T_b = self.path_pose(sid, roll, back) @ _inv(self.G_act(u))
            self.move_line(T_b, check=False, speed=0.01)
            self.work["phase"] = "align"
            self.align(sid, u, roll, back)
            self.work["phase"] = "insert"
            stiff = max(0.2, stiff - 0.15)
        if sm.state[sid] not in ENGAGED:
            ct.set_stiffness(1.0)
            ct.set_mode("joint")
            return False, "not captured (stalls at %s mm)" % att.get("stalls")
        att["captured"] = sm.rec[sid].get("capture")
        # hold still at home, closed loop (the coupling and the hand pull to the same pose), until fastened
        self.work["phase"] = "hold"
        t_hold = float(self.d.time)
        for _ in range(int((HOLD_S + LOCK_WAIT_S) / cell.tick_s)):
            T_u = self.path_pose(sid, roll, end)
            T_cmd = T_u @ _inv(self.G_act(u))
            ct.command_cartesian(T_cmd[:3, 3], quat_wxyz(T_cmd[:3, :3]))
            cell.tick()
            if sm.state[sid] in FASTENED and float(self.d.time) - t_hold >= HOLD_S:
                break
        att["state_at_release"] = sm.state[sid]
        if self.snap:
            self.snap("%s_home" % sid, u)
        ct.set_mode("joint")
        ct.set_stiffness(1.0)
        # let go (measured: a 0.1-0.15 s load across the rod of up to 7.4 N when some end cap grasps let go: the ends of one
        # finger's pad boxes bear on the cap with their normals along the rod and their friction drags the fastened cap
        # sideways as the fingers open; easing off, a compliant arm or relaxing the grip first did not remove it; the rule
        # R3 counts 0.2 s)
        self.work.update({"phase": "release", "held": None})
        self.grip(min(self.max_width, w + RELEASE_EXTRA_M), settle_ticks=10)
        # straight back out along the approach
        self.work["phase"] = "retreat"
        T_out = self.tcp_T()
        T_out[:3, 3] -= T_out[:3, 2] * 0.06
        ok, _ = self.move_line(T_out, allow=(u,), check=True)
        if not ok:
            self.move_line(T_out, check=False)
        self.work.update({"phase": "wait", "unit": None})
        for _ in range(300):
            if sm.state[sid] in FASTENED:
                break
            cell.tick()
        return sm.state[sid] == "LOCKED", sm.state[sid]

    def _lag(self, sid, u, roll, back):
        """how far the unit is behind the command along the insertion (m)"""
        x = self.cell.slot(sid)
        T_u = self.path_pose(sid, roll, back)
        if x["kind"] == "rod_sleeve":
            ins = self.sm.rail_base_world(x)[:3, 2]
        else:
            ins = self.seat_ins(sid)
        return float((T_u[:3, 3] - self.unit_T(u)[:3, 3]) @ ins)


def steps_of(plan, scope_slots):
    """[(slot, unit)] in the assembly order: the bench plan's, else the default nominal one of the scope"""
    if plan and plan.get("steps"):
        return [(st["slot"], st["unit"]) for st in plan["steps"]]
    full = len(scope_slots) > len(DEFAULT_STEPS)
    return [(s, u) for s, u in (DEFAULT_STEPS_FULL if full else DEFAULT_STEPS) if s in scope_slots]


def program(cell, plan=None, log=print, server=None, only=None, wall_cap_s=None, snap=None, end_ticks=300, keep=None):
    """run the privileged assembler on `cell` (a StrutCell with the Panda; its tick routed through the server's watch
    when the server built it); returns the report (`keep`, a dict, receives the Assembler: its force log)"""
    t0 = time.time()
    asm = Assembler(cell, plan=plan, log=log, server=server, wall_cap_s=wall_cap_s, snap=snap)
    steps = []
    asm.move_q(asm.q_ready)
    for sid, u in steps_of(plan, set(x["id"] for x in cell.sm.slots)):
        if only and sid not in only:
            continue
        if asm.out_of_time():
            steps.append({"slot": sid, "unit": u, "placed_by": "nobody", "why": "wall-clock cap"})
            continue
        log("step %s <- %s" % (sid, u))
        steps.append(asm.place_unit(sid, u))
    asm.work.update({"phase": "end", "unit": None, "slot": None, "held": None})
    if not asm.go(asm.q_ready):
        asm.move_q(asm.q_ready)
    asm.tick(end_ticks)
    rep = {"score": round(cell.sm.score(), 4), "steps": steps, "events": [(round(t, 3), e) for t, e in cell.sm.events],
           "unplanned_contacts": [{k: v for k, v in c.items() if k != "key"} for c in asm.contacts],
           "unit_contacts": [{k: v for k, v in c.items() if k != "key"} for c in asm.unit_contacts],
           "cluster_contacts": [{k: v for k, v in c.items() if k != "key"} for c in asm.cluster_contacts],
           "pull_mm": {k: round(v * 1e3, 3) for k, v in asm.pull.items()},
           "sim_s": round(float(cell.d.time), 2), "wall_s": round(time.time() - t0, 1),
           "planner_checks": int(asm.pl.n_checks)}
    log("score %.3f, %d / %d slots, sim %.0f s, wall %.0f s, unplanned robot contacts %d, unit contacts %d" % (
        rep["score"], sum(1 for x in cell.sm.rec.values() if x["achieved"]), len(cell.sm.rec), rep["sim_s"], rep["wall_s"],
        len(rep["unplanned_contacts"]), len(rep["unit_contacts"])))
    if keep is not None:
        keep["assembler"] = asm
    return rep
