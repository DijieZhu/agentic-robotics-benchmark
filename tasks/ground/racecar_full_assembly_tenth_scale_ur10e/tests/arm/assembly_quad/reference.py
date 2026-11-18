"""The UR10e assembles the quadruped from its body on the stand: the privileged reference program (no observation).

Privileged state only: the units' exact poses, the slots' live targets (`QuadCell.live_target`: the carrier as it is
now), the UR's own kinematics (`ur_ik.URKin`) and a planner on a copy of the cell (`motion.Planner`). Nothing but the
UR, its 2F-85 and the physics moves a unit (no jig, no feeder). The robot being built is physical: its output pulleys
turn on unpowered hinges, the fastened shoulders swing about HAA against the belt drives' friction, so every target is
taken from the robot as it is when the unit's turn comes and followed while the unit goes in.

Order (`shop`, the owner's decision, DESIGN.md 9: the knee made on the bench): per leg FL, FR, HL, HR: the shoulder
onto the HAA output, its support onto the body, then the leg: the upper leg picked, lowered with its knee socket down
onto its lower leg lying key up on the bench (the lower leg's most likely rest: its flat outer face; the upper leg has
no stable knee-up rest), held until the knee locks, lifted with the lower leg and, with the same grasp, put onto the
shoulder's HFE output. The units start where the bench plan put them (`bench_plan.rest_plan`, computed in REVERSE
order, so that when a unit's spot is chosen every unit still on the bench at its turn is already there).

Per step:
  grasps   the unit's own candidates (`grasps.grasp_candidates` on its root body's pieces, in its own frame: the units
           of a type are turned 180 deg against each other, a grasp does not carry over); each checked at the pick (every
           IK branch: pre-grasp 80 mm back along the approach, the approach line, the fingers open to the width +
           15 mm, the planner's clearances) and at the slot (`place_checks`: the pre-insertion pose and inwards in 2 mm
           steps on one IK branch, the fingers closed to the width keep 1 mm and the links 3 mm from the robot being
           built; `hold_x` = how far short of home the fingers can follow, at most the capture depth less 0.5 mm)
  pick     plan to the pre-grasp, approach line, close to the width - GRIP_SQUEEZE_M (about 30 N per pad), lift 60 mm;
           the unit must stay in the hand (its pose in the tool frame within 3 mm / 4 deg of the grasp's)
  carry    plan (the held unit, and what is fastened to it, carried rigidly; the robot grown 8 mm, the held unit 4 mm)
           to the pre-insertion posture of the grasp AS HELD
  insert   Cartesian, closed loop on the live target at 10 mm/s from the pre-insertion; the unit is let go once the cell
           has it ENGAGED (captured: the key in its socket / the support within the seat's capture), the fingers
           open and back out along the approach; a stall (the unit 4 mm behind the command) backs off 15 mm and retries
           (twice) with the target re-read
  retreat  open, back out 60 mm along the approach (checked), plan to the next pick.
A unit the arm cannot place stays where it is (`placed_by: nobody`); no fallback. A contact monitor logs every contact
of the UR with anything but the unit in its fingers (and the unit being grasped), and of the moving unit with anything
but its carrier's unit, the hand and the bench (`unplanned_contacts`).
"""
import math
import time

import numpy as np

from .coupling import _inv, quat_wxyz, rotation_angle

GRIP_SQUEEZE_M = 0.005          # the closing command this far below the grasp's width: ~6 N/mm -> about 30 N per pad
OPEN_EXTRA_M = 0.015            # the fingers open to the grasp's width + this for the approach (2F-85: 85 mm at most)
RELEASE_EXTRA_M = 0.006         # ... and to the width + this to let a placed unit go (a wider opening at the HAA slot put
                                # a finger link on the body at 8 N)
PREGRASP_M = 0.08
LIFT_M = 0.06
PRE_INSERT_M = {"keyed": 0.060, "seat": 0.060, "kfe": 0.040}
INSERT_SPEED_M_S = 0.010
APPROACH_SPEED_M_S = 0.030
CARRY_SPEED_M_S = 1.0           # the tcp's speed cap on planned moves (a held unit rides at the tcp; the cell's rule: 3 m/s)
STALL_M = 0.004
INSERT_STIFFNESS = 0.5          # the arm's stiffness scale going in: a key jammed 4 mm on its face pushes ~50 N, not ~100
RETRIES = 2
HELD_TOL_M, HELD_TOL_RAD = 0.003, math.radians(4.0)
CAPTURE_MARGIN_M = 0.0005


def _T(p, R):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = p
    return T


def unit_grasps(kit):
    """{unit: [(G, width, desc)]} the 2F-85 grasp candidates of every loose unit (its frame; ~27 s for the 16 units)"""
    from .grasps import grasp_candidates, unit_convex_set
    out = {}
    for u in kit["loose"]:
        cs = unit_convex_set(kit["dir"], kit, u)
        out[u] = grasp_candidates(cs, np.array(kit["bodies"][u]["com_m"]))
    return out


class Assembler(object):
    def __init__(self, cell, log=print, plan=None, grasps=None, monitor=True):
        from .gripper_2f85 import MAX_WIDTH_M
        from .kitpose import designed_candidates
        from .motion import Planner
        from .ur_ik import URKin
        self.cell, self.m, self.d, self.ct, self.sm = cell, cell.m, cell.d, cell.ctrl, cell.sm
        self.kit, self.rb, self.log = cell.kit, cell.robot, log
        self.max_width = MAX_WIDTH_M
        self.pl = Planner(cell)
        self.kin = URKin(model=self.pl.m, prefix=self.rb["prefix"], tcp_site=self.rb["tcp_site"])
        self.designed = designed_candidates(self.kit)
        self.plan_hint = plan or {}
        # the grasp candidates are in the units' frames (layout-independent): `grasps` = unit_grasps(kit) computed once
        self.grasps = grasps if grasps is not None else unit_grasps(self.kit)
        self.q_ready = np.array(cell.cfg["robot_home_q"], float)
        self.work = {"unit": None, "slot": None, "phase": "start", "held": None}
        self.contacts = []
        self.robot_bodies = set(b for b in range(self.m.nbody) if self.m.body(b).name.startswith(self.rb["prefix"]))
        self.steps = []
        if monitor:
            cell.hooks.append(self._monitor)

    # ------------------------------------------------------------------ state
    def tcp_T(self):
        p, R = self.ct.tcp_actual()
        return _T(p, R)

    def unit_T(self, u):
        return self.cell.body_T(u)

    @staticmethod
    def natural(q):
        """the bench-mounted UR's working family: elbow up (q3 > 0 with the upper arm above it) and the upper arm not
        more than ~11 deg below horizontal (q2 < 0.2); the other branches (elbow down, the upper arm under the shoulder)
        reach the bench only through reconfigurations no planner finds past the stand"""
        return q[2] > 0.05 and -math.pi - 0.2 < q[1] < 0.2

    def ik(self, T, q_ref):
        """[(score, q, margin)]: the natural branches for the tool pose T (each at its alias nearest q_ref), nearest
        q_ref first"""
        sols = [x for x in self.kin.candidates(T, q_ref) if self.natural(x[1])]
        sols.sort(key=lambda x: float(np.linalg.norm(x[1] - np.asarray(q_ref, float))) - 0.5 * x[2])
        return sols

    def tcp_T_inv_unit(self, u):
        """the tool's pose in the unit's frame as the unit is held now"""
        return _inv(self.unit_T(u)) @ self.tcp_T()

    def group_of(self, u):
        """the unit and every unit fastened to it, directly or through others, the body excluded (a sub-assembly)"""
        grp, changed = {u}, True
        kit, sm = self.kit, self.sm
        while changed:
            changed = False
            for sid, st in sm.rec.items():
                if not st["fastened"] or st["unit"] is None:
                    continue
                cu = kit["bodies"][self.cell.slots[sid]["carrier"]]["unit"]
                if cu == "base":
                    continue
                for a_, b_ in ((cu, st["unit"]), (st["unit"], cu)):
                    if a_ in grp and b_ not in grp:
                        grp.add(b_)
                        changed = True
        return grp

    def _monitor(self):
        """every tick: contacts above 0.5 N of the UR with anything but the unit it holds / grasps, and of the moving
        unit (its group) with anything but its carrier's unit, the hand and the bench"""
        m, d = self.m, self.d
        w = self.work
        self.cell.label["text"] = "%s  %s  %s" % (w.get("slot") or "", w.get("unit") or "", w.get("phase") or "")
        held = w.get("held")
        grp = set(self.group_of(held)) if held else set()
        carrier_unit = None
        if w.get("slot"):
            carrier_unit = self.kit["bodies"][self.cell.slots[w["slot"]]["carrier"]]["unit"]
        f6 = np.zeros(6)
        ub = self.cell.unit_of_body
        for i in range(d.ncon):
            c = d.contact[i]
            b1, b2 = int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])
            r1, r2 = b1 in self.robot_bodies, b2 in self.robot_bodies
            u1, u2 = ub.get(b1), ub.get(b2)
            if r1 and r2:
                continue
            if r1 or r2:
                other_u = u2 if r1 else u1
                if other_u is not None and (other_u in grp or other_u == w.get("unit")):
                    continue
            elif (u1 in grp) != (u2 in grp):
                other_u = u2 if u1 in grp else u1
                if other_u in (carrier_unit, w.get("part")) or (w["phase"] in ("pick", "lift") and other_u is None):
                    continue
                if other_u is None and m.body(b2 if u1 in grp else b1).name == "world":
                    continue
            else:
                continue
            mujoco_ = self.cell.mj
            mujoco_.mj_contactForce(m, d, i, f6)
            if abs(f6[0]) < 0.5:
                continue
            n1, n2 = m.body(b1).name, m.body(b2).name
            key = (w["phase"], w.get("slot"), n1, n2)
            if not self.contacts or self.contacts[-1]["key"] != key:
                self.contacts.append({"key": key, "t": round(float(d.time), 2), "phase": w["phase"], "slot": w.get("slot"),
                                      "bodies": [n1, n2], "force_N": round(float(abs(f6[0])), 1)})

    # ------------------------------------------------------------------ motion primitives
    def tick(self, n=1):
        for _ in range(n):
            self.cell.tick()

    def wait_arrived(self, tol=0.002, max_ticks=400):
        ct = self.ct
        for _ in range(max_ticks):
            self.cell.tick()
            if (np.max(np.abs(ct.q_goal - ct.q_cmd)) < 1e-9 and np.max(np.abs(ct.q() - ct.q_cmd)) < tol
                    and np.max(np.abs(ct.dq())) < 0.02):
                return True
        return False

    def move_q(self, q, tol=0.002):
        ct = self.ct
        ct.set_mode("joint")
        ct.command_joint(np.asarray(q, float))
        return self.wait_arrived(tol=tol)

    def follow(self, path, v_max=CARRY_SPEED_M_S):
        """run a planned joint path segment by segment in per-tick steps (every joint on the straight segment the
        planner checked: one far goal per segment would let the controller's per-joint rate clamp run the joints with
        small moves out first, off the checked segment), the tcp no faster than v_max: where the arm's joints at their
        rate limit would swing the tcp faster (up to 3.2 m/s with the arm stretched: a held support at 3.3 m/s broke
        the cell's 3 m/s part-speed rule), the step is cut finer"""
        ct = self.ct
        ct.set_mode("joint")
        dq_max = ct.rate_limit * ct.tick_s
        dp_max = float(v_max) * ct.tick_s
        qa = ct.q_cmd.copy()
        pa = self.kin.fk_T(qa)[:3, 3]
        for qb in path[1:]:
            qb = np.asarray(qb, float)
            n = max(1, int(math.ceil(float(np.max(np.abs(qb - qa) / dq_max)) - 1e-9)))
            q_prev, p_prev = qa, pa
            for k in range(1, n + 1):
                q = qa + (qb - qa) * (k / n)
                p = self.kin.fk_T(q)[:3, 3]
                m_ = max(1, int(math.ceil(float(np.linalg.norm(p - p_prev)) / dp_max - 1e-9)))
                for j in range(1, m_ + 1):
                    ct.command_joint(q_prev + (q - q_prev) * (j / m_))
                    self.cell.tick()
                q_prev, p_prev = q, p
            qa, pa = qb, p_prev
        return self.wait_arrived()

    def sync(self, held=None, allow=(), poses=None, width=None):
        """the planner's scene: the cell now (or `poses`), the hand holding `held` (unit, its pose in the tcp frame)
        with whatever is fastened to it"""
        group = tuple(sorted(self.group_of(held[0]) - {held[0]})) if held else ()
        self.pl.sync(held=held, group=group, allow=allow, poses=poses, width=width)

    def go(self, q, held=None, allow=(), width=None):
        """planned free-space move to q (the held unit carried as held now); if no path is found directly, via the
        ready posture"""
        self.sync(held=held, allow=allow, width=width)
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

    def line_qs(self, T1, step_m=0.002, T0=None, q0=None):
        """IK postures every step_m along the straight tcp line T0 -> T1 (orientation slerped), on one branch"""
        from scipy.spatial.transform import Rotation, Slerp
        T0 = self.tcp_T() if T0 is None else T0
        q = self.ct.q() if q0 is None else np.asarray(q0, float)
        n = max(1, int(math.ceil(np.linalg.norm(T1[:3, 3] - T0[:3, 3]) / step_m)))
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

    def move_line(self, T1, speed=APPROACH_SPEED_M_S, held=None, allow=(), check=True, width=None):
        """a Cartesian line to T1 (checked on the planner first); the controller's Cartesian mode streams the line"""
        T0 = self.tcp_T()
        if check:
            qs = self.line_qs(T1, T0=T0)
            if qs is None:
                return False, "line IK"
            self.sync(held=held, allow=allow, width=width)
            if not self.pl.line_free(qs[::3] + [qs[-1]]):
                return False, "line blocked %s" % (sorted(set(h[1] for q in qs[::6] for h in self.pl.hits(q)))[:4],)
        from scipy.spatial.transform import Rotation, Slerp
        ct = self.ct
        ct.set_mode("cartesian")
        dist = float(np.linalg.norm(T1[:3, 3] - T0[:3, 3]))
        ang = rotation_angle(T0[:3, :3].T @ T1[:3, :3])
        n = max(1, int(math.ceil(max(dist / speed, ang / math.radians(20.0)) / self.cell.tick_s)))
        sl = Slerp([0, 1], Rotation.from_matrix([T0[:3, :3], T1[:3, :3]]))
        for k in range(1, n + 1):
            s = k / n
            p = (1 - s) * T0[:3, 3] + s * T1[:3, 3]
            R = sl([s]).as_matrix()[0]
            ct.command_cartesian(p, quat_wxyz(R))
            self.cell.tick()
        for _ in range(30):
            self.cell.tick()
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
    def slot_kind(self, sid):
        s = self.cell.slots[sid]
        if s["kind"] == "seat":
            return "seat"
        return "kfe" if sid.startswith("kfe_") else "keyed"

    def capture_depth(self, sid):
        """how far short of home the cell captures the unit (m)"""
        s = self.sm.by_id[sid]
        if s["kind"] == "seat":
            return self.sm.seat_depth
        return s["key_len"] * (1.0 - self.sm.key_frac)

    def target(self, sid, u, part=None):
        """(T_home, insertion) of the moving unit at the slot from the other side as it is now (u the slot's
        candidate, or the carrier's unit lowered onto `part`)"""
        tg = self.cell.live_target_moving(sid, u, part)
        return tg["T"], tg["insertion"]

    # ------------------------------------------------------------------ checks
    def pick_checks(self, u, T_u, G, w, poses=None, n_branches=4):
        """(q_pick, q_pg, margin) for the grasp G (tcp pose in the unit frame) of the unit resting at T_u; else
        (None, None, reason)"""
        P = T_u @ G
        if P[2, 2] > -0.5:
            # from above only: a side approach puts the 2F-85's linkage (+-62-74 mm along the closing axis) or its
            # fingertips into the bench next to a part a few cm tall
            return None, None, "approach not from above"
        P_pg = P.copy()
        P_pg[:3, 3] -= P[:3, 2] * PREGRASP_M
        sols = self.ik(P, self.q_ready)
        if not sols:
            return None, None, "pick IK"
        width = min(self.max_width, w + OPEN_EXTRA_M)
        P_ = dict(poses or {})
        P_[u] = T_u
        why = "pre-grasp IK"
        for _sc, q_pick, mrg in sols[:n_branches]:
            q_pg, ok = self.kin.solve_one(P_pg[:3, 3], P_pg[:3, :3], q_pick, q_pick, margin=0.0)
            if not ok:
                continue
            self.pl.sync(allow=(u,), poses=P_, width=width)
            hs = []
            for k in range(6):
                pk = (1 - k / 5) * P_pg[:3, 3] + k / 5 * P[:3, 3]
                qk, ok = self.kin.solve_one(pk, P[:3, :3], q_pg, q_pg, margin=0.0)
                hs = self.pl.hits(qk) if ok else [("ik", "ik", 0.0)]
                if hs:
                    break
            if hs:
                why = "hits %s at the pick" % sorted(set(h[1] for h in hs))[:3]
                continue
            # the close: the 2F-85's pads swing ~13 mm towards the fingertips between open and closed, so beside a
            # part a few cm tall they can reach the bench only while closing (a support's pick pressed both pads into
            # the bench at 43 N): the pick posture at three widths down to the grasp's, the part allowed
            for wk in (width, 0.5 * (width + w), max(0.0, w - GRIP_SQUEEZE_M)):
                self.pl.sync(allow=(u,), poses=P_, width=wk)
                hs = self.pl.hits(q_pick)
                if hs:
                    break
            if hs:
                why = "hits %s while closing" % sorted(set(h[1] for h in hs))[:3]
                continue
            return q_pick, q_pg, float(mrg)
        return None, None, why

    def place_checks(self, sid, u, G, w, T_home, ins, poses=None, group_rel=None, q_seed=None):
        """(q_pre, hold_x, margin) for the grasp G at the slot: the pre-insertion pose and inwards in 2 mm steps on one
        IK branch; hold_x = how far short of home the fingers can follow (must be within the capture); else
        (None, reason, 0)"""
        kind = self.slot_kind(sid)
        pre = PRE_INSERT_M[kind]
        cap = self.capture_depth(sid) - CAPTURE_MARGIN_M
        P_ = dict(poses or {})
        best = None
        T_pre = T_home.copy()
        T_pre[:3, 3] -= ins * pre
        sols = self.ik(T_pre @ G, self.q_ready if q_seed is None else q_seed)
        if not sols:
            return None, "pre-insertion IK", 0.0
        why = ""
        for _sc, q0, mrg in sols:                           # every branch (<= 8): the result must not depend on the seed
            xs = list(np.arange(pre, -1e-9, -0.002))
            q, hold = q0, None
            for i, x in enumerate(xs):
                T_x = T_home.copy()
                T_x[:3, 3] -= ins * x
                P_[u] = T_x
                for gu, Trel in (group_rel or {}).items():
                    P_[gu] = T_x @ Trel
                if i == 0:
                    qx, ok = q0, True
                else:
                    qx, ok = self.kin.solve_one((T_x @ G)[:3, 3], (T_x @ G)[:3, :3], q, q, margin=0.0)
                hit = []
                if ok:
                    self.pl.sync(allow=(u,), poses=P_, width=w)
                    # the pre-insertion pose is reached by a planned move at speed: 5 mm for the fingers there (a carry's
                    # tracking error put a finger link on the body at 31 N with 2 mm); going in 1 mm, the links 3 mm
                    hit = self.pl.hits(qx, finger_clear=0.001 if i else 0.005, link_clear=0.003 if i else None)
                if not ok or hit:
                    why = ("IK at %.0f mm" % (x * 1e3)) if not ok else ("hits %s at %.0f mm" % (sorted(set(h[1] for h in hit))[:3], x * 1e3))
                    hold = float(xs[i - 1]) if i > 0 else None
                    break
                q = qx
            else:
                hold = 0.0
            if hold is not None and hold <= cap and (best is None or (hold, -mrg) < (best[1], -best[2])):
                best = (q0, hold, float(mrg))
            if best is not None and best[1] == 0.0:
                break
        if best is None:
            return None, why or "insertion", 0.0
        return best

    def feasible_grasps(self, u, T_pick, checks, poses_pick=None, poses_place=None, limit=None):
        """[(score, G, w, desc, q_pick, q_pg, [hold_x per check], [validated pre-insertion posture per check])] best
        first, and the rejections; `checks` =
        [(slot, T_home, insertion, group_rel)] the grasp must pass in turn (a leg: the knee, then the HFE)"""
        out, rejected = [], {}
        com = np.array(self.kit["bodies"][u]["com_m"])
        cands = self.grasps[u]
        hint = (self.plan_hint.get(u) or {}).get("grasp")
        if hint is not None:
            cands = [(np.array(hint["G"], float), float(hint["w"]), hint["desc"] + " (bench plan)")] + list(cands)
        for G, w, desc in cands:
            q_pick, q_pg, r1 = self.pick_checks(u, T_pick, G, w, poses=poses_pick)
            if q_pick is None:
                rejected[desc] = r1
                continue
            holds, q_pres, mrg, q_seed, why = [], [], r1, q_pick, None
            for sid, T_home, ins, group_rel in checks:
                q_pre, hold, r2 = self.place_checks(sid, u, G, w, T_home, ins, poses=poses_place, group_rel=group_rel, q_seed=q_seed)
                if q_pre is None:
                    why = "%s: %s" % (sid, hold)
                    break
                holds.append(hold)
                q_pres.append(q_pre)
                mrg = min(mrg, r2)
                q_seed = q_pre
            if why:
                rejected[desc] = why
                continue
            score = mrg - 2.0 * float(np.linalg.norm(G[:3, 3] - com)) - 20.0 * sum(holds) - (0.3 if w < 0.012 else 0.0)
            out.append((score, G, w, desc, q_pick, q_pg, holds, q_pres))
            if limit and len(out) >= limit:
                break
        out.sort(key=lambda x: -x[0])
        return out, rejected

    # ------------------------------------------------------------------ one step
    def place_unit(self, sid, u, knee=None, part=None):
        """pick the unit where it rests and insert it at the slot; with `knee` (a leg step) first lower it onto `part`
        lying on the bench, hold it until the knee locks and go on with the same grasp. Returns the record."""
        cell, ct, sm = self.cell, self.ct, self.sm
        rec = {"slot": sid, "unit": u, "knee": knee, "t0_s": round(float(self.d.time), 2), "placed_by": "nobody", "attempts": []}
        self.work.update({"unit": u, "slot": knee or sid, "phase": "plan", "held": None, "part": part})
        t_wall = time.time()
        for round_ in range(3):
            # the grasps from where the unit rests now (a failed pick may have moved it)
            T_pick = self.unit_T(u)
            T_home, ins = self.target(sid, u)
            checks = []
            group_rel = {g: _inv(T_pick) @ self.unit_T(g) for g in self.group_of(u) if g != u}
            if knee:
                T_k, ins_k = self.target(knee, u, part)
                checks.append((knee, T_k, ins_k, {}))
                group_rel = {part: _inv(T_k) @ self.unit_T(part)}    # the lower leg rides on it after the knee
            checks.append((sid, T_home, ins, group_rel))
            feas, rejected = self.feasible_grasps(u, T_pick, checks)
            rec.setdefault("grasps", []).append({"candidates": len(self.grasps[u]), "feasible": len(feas)})
            if not feas:
                rec["why"] = "no feasible grasp (%s)" % (sorted(set(str(v) for v in rejected.values()))[:4],)
                self.log("  %s %s: %s" % (sid, u, rec["why"]))
                break
            score, G, w, desc, q_pick, q_pg, holds, q_pres = feas[0]
            att = {"round": round_, "grasp": desc, "width_mm": round(w * 1e3, 1), "hold_mm": [round(h * 1e3, 1) for h in holds]}
            rec["attempts"].append(att)
            ok, why = self.pick(u, G, w, q_pick, q_pg)
            if not ok:
                att["result"] = why
                self.log("  %s %s: pick failed (%s)" % (sid, u, why))
                if self.ct.width() < w + 0.5 * OPEN_EXTRA_M:          # the hand closed on it: let go, look again
                    self.let_go(w)
                continue
            G_act = _inv(self.tcp_T()) @ self.unit_T(u)          # the unit in the tool frame, as held
            if knee:
                self.work["slot"] = knee
                ok, why = self.insert(knee, u, G_act, w, att, keep=True, part=part, q_branch=q_pres[0])
                att["knee"] = why
                if not ok:
                    att["result"] = "knee: " + why
                    self.log("  %s %s: knee failed (%s)" % (knee, u, why))
                    self.let_go(w)
                    break
                T_up = self.tcp_T()
                T_up[2, 3] += LIFT_M
                self.work["phase"] = "lift"
                self.move_line(T_up, held=(u, G_act), check=False)
                self.work.update({"slot": sid, "part": None})
            ok, why = self.insert(sid, u, G_act, w, att, q_branch=q_pres[-1])
            att["result"] = why
            if ok:
                rec["placed_by"] = "ur"
                break
            self.log("  %s %s: insertion failed (%s)" % (sid, u, why))
            if self.work["held"] is not None:
                self.let_go(w)
            break
        for x in ([knee] if knee else []) + [sid]:
            st = sm.rec[x]
            rec.setdefault("slots", {})[x] = {"state": sm.state[x], "fastened": st["fastened"], "achieved": st["achieved"],
                                               "pos_err_mm": st["pos_err_mm"], "ang_err_deg": st["ang_err_deg"]}
        rec.update({"state": sm.state[sid], "t1_s": round(float(self.d.time), 2), "wall_s": round(time.time() - t_wall, 1)})
        self.log("  %-8s %-3s -> %s (%s) sim %.1f s wall %.0f s" % (sid, u, {x: v["state"] for x, v in rec["slots"].items()},
                                                             rec["attempts"][-1].get("result") if rec["attempts"] else "-",
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
        self.work.update({"phase": "to-pick", "held": None})
        self.grip(min(self.max_width, w + OPEN_EXTRA_M), settle_ticks=5)
        if not self.go(q_pg, width=min(self.max_width, w + OPEN_EXTRA_M)):
            return False, "no path to the pre-grasp (%s)" % getattr(self.pl, "why", "")
        self.work["phase"] = "pick"
        T_u = self.unit_T(u)
        P = T_u @ G
        ok, why = self.move_line(P, allow=(u,), width=min(self.max_width, w + OPEN_EXTRA_M))
        if not ok:
            return False, "approach: " + why
        self.grip(max(0.0, w - GRIP_SQUEEZE_M))
        G_act = _inv(self.tcp_T()) @ self.unit_T(u)
        self.work.update({"phase": "lift", "held": u})
        T_up = self.tcp_T()
        T_up[2, 3] += LIFT_M
        ok, why = self.move_line(T_up, held=(u, G_act), check=True)
        if not ok:
            ok, why = self.move_line(T_up, held=(u, G_act), check=False)
        G_now = _inv(self.tcp_T()) @ self.unit_T(u)
        dp = float(np.linalg.norm(G_now[:3, 3] - G_act[:3, 3]))
        da = rotation_angle(G_now[:3, :3].T @ G_act[:3, :3])
        if dp > HELD_TOL_M or da > HELD_TOL_RAD or self.ct.width() < 0.002:
            self.work["held"] = None
            return False, "the unit moved in the hand (%.1f mm, %.1f deg)" % (dp * 1e3, math.degrees(da))
        return True, ""

    def insert(self, sid, u, G_act, w, att, keep=False, part=None, q_branch=None):
        """carry to the pre-insertion, then in along the live target until the cell captures the unit; keep = hold
        it there until the slot locks (the knee: the upper leg stays in the hand); q_branch = the IK branch the place
        checks validated (tried first)"""
        cell, ct, sm = self.cell, self.ct, self.sm
        kind = self.slot_kind(sid)
        pre = PRE_INSERT_M[kind]
        G_act = _inv(self.tcp_T()) @ self.unit_T(u)            # the unit in the tool frame as held NOW (the knee, a lift)
        Gi = _inv(G_act)                                        # the tool in the unit frame, as held
        T_home, ins = self.target(sid, u, part)
        T_pre = T_home.copy()
        T_pre[:3, 3] -= ins * pre
        T_t = T_pre @ Gi
        sols = self.ik(T_t, ct.q())
        if q_branch is not None:
            q_v, ok_v = self.kin.solve_one(T_t[:3, 3], T_t[:3, :3], q_branch, q_branch, margin=0.0)
            if ok_v:
                sols = [(0.0, q_v, 0.0)] + [x for x in sols if np.max(np.abs(x[1] - q_v)) > 1e-3]
        self.work["phase"] = "carry"
        moved = False
        for _sc, q_pre, _m in sols[:6]:
            if self.go(q_pre, held=(u, G_act), width=w):
                moved = True
                break
        if not moved:
            return False, "no path to the pre-insertion"
        self.work["phase"] = "insert"
        engaged = ("ENGAGED", "HOLDING", "LOCKED", "LOCKED_OFF")
        ct.set_stiffness(INSERT_STIFFNESS)
        for attempt in range(RETRIES + 1):
            # the hold as it is now (a stall or the carry may have moved the unit in the pads)
            Gi = self.tcp_T_inv_unit(u)
            ct.set_mode("cartesian")
            x = pre
            stall_ticks = 0
            done = False
            for _ in range(int((pre + 0.02) / (INSERT_SPEED_M_S * cell.tick_s)) + 200):
                T_home, ins = self.target(sid, u, part)
                x = max(-0.0005, x - INSERT_SPEED_M_S * cell.tick_s)
                T_u = T_home.copy()
                T_u[:3, 3] -= ins * x
                T_tcp = T_u @ Gi
                ct.command_cartesian(T_tcp[:3, 3], quat_wxyz(T_tcp[:3, :3]))
                cell.tick()
                if sm.state[sid] in engaged:
                    done = True
                    break
                # how far the unit is behind the command along the insertion
                lag = float((T_u[:3, 3] - self.unit_T(u)[:3, 3]) @ ins)
                stall_ticks = stall_ticks + 1 if lag > STALL_M else 0
                if stall_ticks > 30:
                    break
            if done:
                break
            att.setdefault("stalls", []).append(round(x * 1e3, 1))
            # back off 15 mm along the insertion and try again with the target re-read
            T_back = self.tcp_T()
            T_back[:3, 3] -= ins * 0.015
            self.move_line(T_back, held=(u, G_act), check=False, speed=0.02)
            pre = 0.015 + max(0.0, x)
        ct.set_stiffness(1.0)
        if sm.state[sid] not in engaged:
            ct.set_mode("joint")
            return False, "not captured (stalled at %s mm)" % att.get("stalls")
        # captured: go on home with the unit in the hand and hold it there until the cell locks it (the hand and the
        # coupling then pull towards the same pose); let go only after that: a light part let go at its capture is
        # dragged round by the opening pads (a support turned 5.5 deg about its journal and jammed, never locking).
        # With `keep` the hand stays where the capture found it (the knee: the cell pulls the lower leg up to it).
        ct.set_mode("cartesian")
        T_hold = self.tcp_T()
        for _ in range(500):
            if not keep:
                T_home, ins = self.target(sid, u, part)
                x = max(-0.0005, x - INSERT_SPEED_M_S * cell.tick_s)
                T_u = T_home.copy()
                T_u[:3, 3] -= ins * x
                T_hold = T_u @ Gi
            ct.command_cartesian(T_hold[:3, 3], quat_wxyz(T_hold[:3, :3]))
            cell.tick()
            if sm.state[sid] in ("LOCKED", "LOCKED_OFF"):
                break
        ct.set_mode("joint")
        if keep:
            return sm.state[sid] in ("LOCKED", "LOCKED_OFF"), sm.state[sid]
        # let go, back out along the approach
        self.work.update({"phase": "release", "held": None})
        self.grip(min(self.max_width, w + RELEASE_EXTRA_M), settle_ticks=10)
        T_out = self.tcp_T()
        T_out[:3, 3] -= T_out[:3, 2] * 0.06
        ok, _ = self.move_line(T_out, allow=(u,), check=True)
        if not ok:
            self.move_line(T_out, check=False)
        self.work.update({"phase": "wait", "unit": None})
        for _ in range(300):
            cell.tick()
            if sm.state[sid] in ("LOCKED", "LOCKED_OFF"):
                break
        return sm.state[sid] in ("LOCKED", "LOCKED_OFF"), sm.state[sid]


def shop_steps(kit, legs=("FL", "FR", "HL", "HR")):
    """[{"kind": "haa" | "sup" | "leg", "slot", "unit", ...}] in the shop order with the designed units; a leg step
    moves the upper leg: "knee" (its kfe slot, the lower leg "part" lying on the bench) then "slot" (the HFE)"""
    from .kitpose import designed_candidates
    des = designed_candidates(kit)
    out = []
    for leg in legs:
        S = des["haa_" + leg]["unit"]
        A = des["sup_" + leg]["unit"]
        U = des["hfe_" + S]["unit"]
        L = des["kfe_" + U]["unit"]
        out += [{"kind": "haa", "slot": "haa_" + leg, "unit": S}, {"kind": "sup", "slot": "sup_" + leg, "unit": A},
                {"kind": "leg", "slot": "hfe_" + S, "unit": U, "knee": "kfe_" + U, "part": L}]
    return out


def program(cell, plan=None, legs=("FL", "FR", "HL", "HR"), log=print, final_checks=True):
    """run the privileged assembler on `cell` (a QuadCell with the UR); returns the report"""
    t0 = time.time()
    asm = Assembler(cell, log=log, plan=plan)
    steps = []
    asm.move_q(asm.q_ready)
    for st in shop_steps(cell.kit, legs):
        log("step %s <- %s%s" % (st["slot"], st["unit"], (" (knee %s on %s)" % (st["knee"], st["part"])) if st["kind"] == "leg" else ""))
        steps.append(asm.place_unit(st["slot"], st["unit"], knee=st.get("knee"), part=st.get("part")))
    # back to the ready posture, 3 s standing
    asm.work.update({"phase": "end", "unit": None, "slot": None, "held": None})
    asm.go(asm.q_ready)
    asm.tick(300)
    rep = {"score": round(cell.sm.score(), 4), "steps": steps, "events": [(round(t, 3), e) for t, e in cell.sm.events],
           "unplanned_contacts": [{k: v for k, v in c.items() if k != "key"} for c in asm.contacts],
           "sim_s": round(float(cell.d.time), 1), "wall_s": round(time.time() - t0, 1)}
    log("score %.3f, %d / %d slots, sim %.0f s, wall %.0f s, unplanned contacts %d" % (
        rep["score"], sum(1 for x in cell.sm.rec.values() if x["achieved"]), len(cell.sm.rec), rep["sim_s"], rep["wall_s"],
        len(rep["unplanned_contacts"])))
    return rep
