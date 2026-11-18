"""The UR10e builds the race car on its work stand: the privileged reference program, part I (no observation): the
assembler core and the four top-down units (the stick pack, the deck assembly, the compute module, the 2D LiDAR).
Part II (`reference_shop.py`) adds the wheels, the episode's flow and the server's entry point; where it needs the core
to behave differently it subclasses `Assembler` (this file stays as it is).

Privileged state only: the units' exact poses, the slots' live targets (`CarCell.live_target`: the carrier as it is
now), the UR's own kinematics (`ur_ik.URKin`) and a planner on a copy of the cell (`motion.Planner`). Nothing but the
UR, its 2F-85 and the physics moves a unit (no jig). Ported from the quadruped cell's assembler
(`assembly_quad/reference.py`): the same checks, motions, insertion and contact monitor (its planner with the per-body
margins fixed: `make_planner`); the grasps are this car's (`car_grasps.reference_grasps`: the frozen table of the kit
recheck, primary entries only, with the stick pack's grasp added in front), the four units' procedures are new:

  battery  the stick pack picked across its wrap (row 358, `car_grasps.ADDED`: station +30 mm, the closing axis tilted
           +10 deg about the pack's long axis, the fingertips 2 mm below the contact line, the right pad outboard; its
           pad flip 356; the frozen row 178 last: it cannot reach the release on the kit), carried to 60 mm above its
           release pose and lowered to it: the pack's underside BATTERY_RELEASE_M (24 mm) above its seat (the inner
           finger cannot go lower: the brace's lower flange); held 0.3 s, opened by 2 mm per pad, backed out upwards; the
           pack falls into the bay and the cell captures it on the way (no insertion). The pack lies on the bench with
           its lead end either way (the stratum's 0 or 180 deg): the grasp is in the pack's frame, the carry turns it.
  deck     the side pinch on the shelf (frozen rows 130, 129, 131, 149, 165, 167, 118, 136: the station car x 120-142.5,
           the pads on the plate's faces 6-9 mm in from its right long edge, the approach yaw -10 or 0 deg): 30 mm back
           along the horizontal approach, the line in, closed (the plate is 5.19 mm: the command saturates at the
           2F-85's force limit), lifted 40 mm, carried no faster than DECK_CARRY_SPEED_M_S (0.25 m/s: the kit recheck's
           sag line holds there on every carry path), lowered at 10 mm/s onto the four post tops (the hand compensating
           the deck's sag in the pads: the hold is re-measured before going down), held until the cell locks it,
           opened, backed out horizontally. Loaded (the compute module and the LiDAR fastened on it on the shelf), the
           same grasp carries the sub-assembly (535 g).
  compute  from the tray's nest, top-down across the heat sink's two 34.56 mm faces (frozen rows 100, 94, 92, 98, 96,
  lidar    101, 93; the LiDAR across its base or its head, rows 59, 70, 0), carried (the hand turns the unit from its
           storage turn into its mounting orientation: the live target's design key orientation) to 60 mm above its
           place on the deck (wherever the deck is: on the car or on its shelf), lowered at 10 mm/s until captured,
           held until locked, opened, backed out upwards.

Per step (`place_unit`, the precedent's):
  grasps   the unit's frozen candidates, each checked at the pick (`pick_checks`: every natural IK branch, the
           pre-grasp, the approach line, the fingers at the approach opening, half-way and the squeeze; the planner's
           clearances against the cell as it is now) and at the slot (`place_checks`: the pre-insertion and inwards in
           2 mm steps on one IK branch, the fingers 1 mm and the links 3 mm off everything, hold_x within the capture;
           the stick pack: `drop_checks`, the pre-release, the lowering, the release opening and the retreat)
  pick     plan to the pre-grasp, the approach line, close to the width - GRIP_SQUEEZE_M, lift; the unit must stay in
           the hand (its pose in the tool frame within 3 mm / 4 deg of the grasp's)
  carry    plan (the held unit and whatever is fastened to it carried rigidly) to the pre-insertion posture of the
           grasp AS HELD, the tcp no faster than the unit's carry speed
  insert   Cartesian, closed loop on the live target at 10 mm/s from the pre-insertion; a stall backs off 15 mm and
           retries (twice); once captured, on home and held until the cell locks it; then let go
  retreat  open, back out along the approach (checked), the next step plans from there.
A unit the arm cannot place stays where it is (`placed_by: nobody`).

The contact monitor (`_monitor`, every tick) logs every contact above 0.5 N of the UR with anything but the unit it
holds or grasps (and what is fastened onto it), and of the moving unit (held, or let go and not yet locked; with what is
fastened onto it, `carried_by`, never its carrier) with anything but its carrier's unit, the hand, the bench and, while
it is picked and lifted, its fixture (`unplanned_contacts`).
It also keeps per step the largest normal force of a robot link (the four pads apart) on one body, the largest pad
force, the largest per (unit, body) sum of a loose unit's contact forces (the pads apart) and its 20-tick minimum
(the cell's sustained-force rule), the phases' times, and the moving unit's pose against its slot at the tick the cell
captures it (`capture`: lateral / tilt / clock against the slot's window).

`program(cell, order="shop" | "sub", units=None)` runs the four steps (shop: battery, deck onto the car, compute,
LiDAR; sub: battery, compute and LiDAR onto the deck on its shelf, then the loaded deck onto the car) and returns the
report.
"""
import math
import time

import numpy as np

from . import car_grasps as CG
from .coupling import _inv, quat_wxyz, rotation_angle

GRIP_SQUEEZE_M = 0.005          # the closing command this far below the grasp's width (~6 N/mm: ~30 N per pad; the deck's
                                # 5.19 mm plate saturates at the 2F-85's force limit, 46-48 N per pad)
OPEN_EXTRA_M = 0.015            # the fingers open to the grasp's width + this for a top-down approach
RELEASE_EXTRA_M = 0.004         # ... and to the width + this to let a placed unit go (2 mm per pad: the gates' value)
PREGRASP_M = 0.08
LIFT_M = 0.06
LIFT_SPEED_M_S = 0.05
PRE_INSERT_M = 0.060
INSERT_SPEED_M_S = 0.010
APPROACH_SPEED_M_S = 0.030
CARRY_SPEED_M_S = 0.5           # the tcp's speed cap on planned moves (the dynamic gates' value; the cell's rule: 3 m/s)
DECK_CARRY_SPEED_M_S = 0.25     # the deck's carries (kit recheck: the 1 deg sag line holds at 0.25 m/s on every carry path)
STALL_M = 0.004
INSERT_STIFFNESS = 0.5          # the arm's stiffness scale going in
RETRIES = 2
HELD_TOL_M, HELD_TOL_RAD = 0.003, math.radians(4.0)
CAPTURE_MARGIN_M = 0.0005
RETREAT_M = 0.06
LOCK_HOLD_TICKS = 500           # at most this long holding a captured unit until the cell locks it
LOCK_WAIT_TICKS = 300           # ... and after letting go
CONTACT_N = 0.5                 # the monitor's threshold
BATTERY_RELEASE_M = 0.024       # the pack's underside above its seat at the release (the fingers reach no lower past the brace)
BATTERY_LOWER_SPEED_M_S = 0.030
BATTERY_HOLD_TICKS = 30
BATTERY_FINGER_CLEAR_M = 0.001  # the fingers off the brace at the release (the battery grasp's static clearance line: 1.0 mm)
PART_FORCE_TICKS = 20

# the four top-down units' procedures (per unit type of the kit)
SPEC = {
    "BAT": {"slot": "battery", "kind": "drop", "pre_grasp_m": PREGRASP_M, "open_extra_m": OPEN_EXTRA_M, "lift_m": LIFT_M,
            "from_above": True, "carry_v": CARRY_SPEED_M_S, "release_extra_m": RELEASE_EXTRA_M},
    "DCK": {"slot": "deck", "kind": "insert", "pre_grasp_m": 0.03, "open_extra_m": 0.010, "lift_m": 0.04,
            "from_above": False, "carry_v": DECK_CARRY_SPEED_M_S, "release_extra_m": RELEASE_EXTRA_M},
    "CMP": {"slot": "compute", "kind": "insert", "pre_grasp_m": PREGRASP_M, "open_extra_m": OPEN_EXTRA_M, "lift_m": LIFT_M,
            "from_above": True, "carry_v": CARRY_SPEED_M_S, "release_extra_m": RELEASE_EXTRA_M},
    "LDR": {"slot": "lidar", "kind": "insert", "pre_grasp_m": PREGRASP_M, "open_extra_m": OPEN_EXTRA_M, "lift_m": LIFT_M,
            "from_above": True, "carry_v": CARRY_SPEED_M_S, "release_extra_m": RELEASE_EXTRA_M},
}
ORDERS = {"shop": ("BAT", "DCK", "CMP", "LDR"), "sub": ("BAT", "CMP", "LDR", "DCK")}
ENGAGED = ("ENGAGED", "HOLDING", "LOCKED", "LOCKED_OFF")


def _T(p, R):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = p
    return T


def make_planner(cell, seed=0):
    """`motion.Planner` on a copy of the cell, with MuJoCo's per-body margins kept equal to the largest geom margin of
    each body. The planner raises the robot's (and the held unit's) geom margins on its copy after the compile, but
    MuJoCo's midphase culls geom pairs by the COMPILED `body_margin` (0 here): its clearances (robot 8 mm, fingers
    2 mm, held unit 4 mm) were then seen only once two bounding boxes overlapped (measured on the 14k-piece chassis:
    a pad 0.71 mm from a piece gave no contact, one 0.27 mm into it did)."""
    from ..assembly_quad.motion import Planner

    class CarPlanner(Planner):
        def _body_margins(self):
            m = self.m
            bm = np.zeros(m.nbody)
            np.maximum.at(bm, np.asarray(m.geom_bodyid), np.asarray(m.geom_margin))
            m.body_margin[:] = bm

        def sync(self, *a, **kw):
            super().sync(*a, **kw)
            self._body_margins()

    pl = CarPlanner(cell, seed=seed)
    pl._body_margins()
    return pl


def unit_grasps(kit):
    """{unit: [(G (m), width (m), desc, entry)]}: the reference grasps of the four top-down units (`car_grasps.
    reference_grasps`: the added ones (`ADDED`), then the frozen primaries; the unit of each slot from the kit; the
    grasps are in the unit's frame, layout-independent)"""
    out = {}
    for s in kit["slots"]:
        typ = CG.SLOT_UNIT.get(s["id"])
        if typ in SPEC and SPEC[typ]["slot"] == s["id"]:
            for c in s["candidates"]:
                out[c["unit"]] = CG.reference_grasps(s["id"])
    return out


def unit_fallback_grasps(kit):
    """{unit: [(G, width, desc, entry)]}: the second tier (`car_grasps.fallback_grasps`), tried only when no grasp of
    the first tier is feasible"""
    out = {}
    for s in kit["slots"]:
        typ = CG.SLOT_UNIT.get(s["id"])
        if typ in SPEC and SPEC[typ]["slot"] == s["id"]:
            for c in s["candidates"]:
                out[c["unit"]] = CG.fallback_grasps(s["id"])
    return out


def capture_ratios(r, x):
    """the pose error at a capture against the slot's window (`coupling.evaluate` result r, slot entry x of the state
    machine): {lateral_mm, tilt_deg, clock_deg, axial_mm, ratio: {lateral, tilt, clock}, ratio_max}"""
    lat = float(r["lateral"] if x["kind"] == "seat" else r["pos_err"])
    out = {"lateral_mm": lat * 1e3, "tilt_deg": math.degrees(float(r["ang_err"])), "axial_mm": float(r["axial"]) * 1e3,
           "ratio": {"lateral": lat / x["cap_lat"], "tilt": float(r["ang_err"]) / x["cap_tilt"]}}
    if x["kind"] == "keyed":
        out["clock_deg"] = math.degrees(abs(float(r["clock_res"])))
        out["ratio"]["clock"] = abs(float(r["clock_res"])) / x["cap_clock"]
    out["ratio_max"] = max(out["ratio"].values())
    return out


class Assembler(object):
    def __init__(self, cell, log=print, grasps=None, monitor=True, planner_seed=0, family_fallback=False):
        from ..assembly_quad.gripper_2f85 import MAX_WIDTH_M
        from ..assembly_quad.ur_ik import URKin
        self.cell, self.m, self.d, self.ct, self.sm = cell, cell.m, cell.d, cell.ctrl, cell.sm
        self.kit, self.rb, self.log = cell.kit, cell.robot, log
        self.mj = cell.mj
        self.max_width = MAX_WIDTH_M
        self.pl = make_planner(cell, seed=planner_seed)
        self.kin = URKin(model=self.pl.m, prefix=self.rb["prefix"], tcp_site=self.rb["tcp_site"])
        self.grasps = grasps if grasps is not None else unit_grasps(self.kit)
        self.fallback = unit_fallback_grasps(self.kit) if grasps is None else {}
        self.family_fallback = bool(family_fallback)
        self._family = {}
        self.q_ready = np.array(cell.cfg["robot_home_q"], float)
        self.loose = set(self.kit["loose"])
        self.type_of = {u: self.kit["bodies"][u].get("type", u) for u in self.kit["loose"]}
        self.slot_of = {}
        for s in self.kit["slots"]:
            for c in s["candidates"]:
                t = self.type_of.get(c["unit"])
                if t in SPEC and SPEC[t]["slot"] == s["id"]:
                    self.slot_of[c["unit"]] = s["id"]
        self.work = {"unit": None, "slot": None, "phase": "start", "held": None, "moving": None}
        self.contacts = []
        self.robot_bodies = set(b for b in range(self.m.nbody) if self.m.body(b).name.startswith(self.rb["prefix"]))
        self.pad_geoms = set(self.m.geom(n).id for n in self.rb.get("pad_geoms", []))
        self.world_body = 0
        self.steps = []
        self.step_stats = None
        self._f6 = np.zeros(6)
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
        more than ~11 deg below horizontal (q2 < 0.2)"""
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
        """the unit and every unit fastened to it, directly or through others, the base excluded (a sub-assembly)"""
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

    def carried_by(self, u):
        """the unit and every unit fastened onto it, directly or onto those (what moves with it); unlike `group_of`
        its carrier chain is not in it: a compute module fastened on the deck on its shelf does not make the deck's
        rest on the shelf a moving contact"""
        out, changed = {u}, True
        kit, sm = self.kit, self.sm
        while changed:
            changed = False
            for sid, st in sm.rec.items():
                if not st["fastened"] or st["unit"] is None or st["unit"] in out:
                    continue
                if kit["bodies"][self.cell.slots[sid]["carrier"]]["unit"] in out:
                    out.add(st["unit"])
                    changed = True
        return out

    def _phase(self, name, **kw):
        self.work["phase"] = name
        self.work.update(kw)
        if self.step_stats is not None:
            self.step_stats["timeline"].append([round(float(self.d.time), 3), name])

    def _monitor(self):
        """every tick: the unplanned contacts, the per-step forces and the capture pose (module docstring)"""
        m, d = self.m, self.d
        w = self.work
        self.cell.label["text"] = "%s  %s  %s" % (w.get("slot") or "", w.get("unit") or "", w.get("phase") or "")
        mov = w.get("held") or w.get("moving")
        grp = self.carried_by(mov) if mov else set()
        carrier_unit = None
        if w.get("slot"):
            carrier_unit = self.kit["bodies"][self.cell.slots[w["slot"]]["carrier"]]["unit"]
        ub = self.cell.unit_of_body
        st = self.step_stats
        rob_sum, part_sum = {}, {}
        f6 = self._f6
        for i in range(d.ncon):
            c = d.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            b1, b2 = int(m.geom_bodyid[g1]), int(m.geom_bodyid[g2])
            r1, r2 = b1 in self.robot_bodies, b2 in self.robot_bodies
            if r1 and r2:
                continue
            u1, u2 = ub.get(b1), ub.get(b2)
            l1, l2 = u1 in self.loose, u2 in self.loose
            if not (r1 or r2 or l1 or l2):
                continue
            self.mj.mj_contactForce(m, d, i, f6)
            fn = abs(float(f6[0]))
            if st is not None:
                if r1 or r2:
                    rg, og = (g1, b2) if r1 else (g2, b1)
                    if rg in self.pad_geoms:
                        st["pad_max_N"] = max(st["pad_max_N"], fn)
                    else:
                        k = (int(m.geom_bodyid[rg]), og)
                        rob_sum[k] = rob_sum.get(k, 0.0) + fn
                else:
                    for uu, ll, ob in ((u1, l1, b2), (u2, l2, b1)):
                        if ll:
                            k = (uu, ob)
                            part_sum[k] = part_sum.get(k, 0.0) + fn
            if fn < CONTACT_N:
                continue
            if r1 or r2:
                other_u = u2 if r1 else u1
                if other_u is not None and (other_u in grp or other_u == w.get("unit")):
                    continue
            elif (u1 in grp) != (u2 in grp):
                ob = b2 if u1 in grp else b1
                other_u = u2 if u1 in grp else u1
                if other_u == carrier_unit and carrier_unit is not None:
                    continue
                if other_u is None and (ob == self.world_body or w["phase"] in ("pick", "lift")):
                    continue
            else:
                continue
            n1, n2 = m.body(b1).name, m.body(b2).name
            key = (w["phase"], w.get("slot"), n1, n2)
            if not self.contacts or self.contacts[-1]["key"] != key:
                self.contacts.append({"key": key, "t": round(float(d.time), 2), "phase": w["phase"], "slot": w.get("slot"),
                                      "bodies": [n1, n2], "geoms": [m.geom(g1).name, m.geom(g2).name],
                                      "force_N": round(fn, 1)})
        if st is None:
            return
        if rob_sum:
            (lb, ob), fmax = max(rob_sum.items(), key=lambda kv: kv[1])
            if fmax > st["robot_link_max_N"]:
                st["robot_link_max_N"] = fmax
                st["robot_link_max_pair"] = [m.body(lb).name, m.body(ob).name]
        pm, pk = 0.0, None
        for k_, v_ in part_sum.items():
            if v_ > pm:
                pm, pk = v_, k_
        if pm > st["part_instant_max_N"]:
            st["part_instant_max_N"] = pm
            st["part_instant_max_pair"] = [pk[0], m.body(pk[1]).name]
        hist = st["_hist"]
        hist.append(pm)
        if len(hist) > PART_FORCE_TICKS:
            del hist[0]
        if len(hist) == PART_FORCE_TICKS:
            s_ = min(hist)
            if s_ > st["part_sustained_max_N"]:
                st["part_sustained_max_N"] = s_
                st["part_sustained_max_pair"] = None if pk is None else [pk[0], m.body(pk[1]).name]
        sid, u = w.get("slot"), w.get("unit")
        if sid and u and st.get("capture") is None and self.sm.state.get(sid) in ENGAGED:
            x = self.sm.by_id[sid]
            cand = [cc for cc in x["cands"] if cc["unit"] == u]
            if cand:
                r = self.sm.evaluate(x, cand[0])
                st["capture"] = dict(capture_ratios(r, x), t=round(float(d.time), 3), state=self.sm.state[sid])

    def _new_stats(self, sid, u):
        self.step_stats = {"slot": sid, "unit": u, "timeline": [], "robot_link_max_N": 0.0, "robot_link_max_pair": None,
                           "pad_max_N": 0.0, "part_instant_max_N": 0.0, "part_instant_max_pair": None,
                           "part_sustained_max_N": 0.0, "part_sustained_max_pair": None, "capture": None, "_hist": []}
        return self.step_stats

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
        planner checked), the tcp no faster than v_max"""
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
        group = tuple(sorted(self.group_of(held[0]) - {held[0]})) if held else ()
        self.pl.sync(held=held, group=group, allow=allow, poses=poses, width=width)

    def go(self, q, held=None, allow=(), width=None, v_max=CARRY_SPEED_M_S):
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
        return self.follow(path, v_max=v_max)

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

    def move_line(self, T1, speed=APPROACH_SPEED_M_S, held=None, allow=(), check=True, width=None, finger_clear=None):
        """a Cartesian line to T1 (checked on the planner first, the fingers at `finger_clear` if given); the
        controller's Cartesian mode streams the line"""
        T0 = self.tcp_T()
        if check:
            qs = self.line_qs(T1, T0=T0)
            if qs is None:
                return False, "line IK"
            self.sync(held=held, allow=allow, width=width)
            fc0 = self.pl.finger_clear
            if finger_clear is not None:
                self.pl.finger_clear = float(finger_clear)
            try:
                free = self.pl.line_free(qs[::3] + [qs[-1]])
                hs = [] if free else sorted(set(h[1] for q in qs[::6] for h in self.pl.hits(q)))[:4]
            finally:
                self.pl.finger_clear = fc0
            if not free:
                return False, "line blocked %s" % (hs,)
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

    def hold_cart(self, T, ticks):
        ct = self.ct
        ct.set_mode("cartesian")
        ct.command_cartesian(T[:3, 3], quat_wxyz(T[:3, :3]))
        self.tick(ticks)
        ct.set_mode("joint")

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
    def capture_depth(self, sid):
        """how far short of home the cell captures the unit (m)"""
        return float(self.sm.by_id[sid]["cap_depth"])

    def target(self, sid, u):
        """(T_home, insertion): the unit's home on the slot from its carrier as it is now, its direction of travel"""
        tg = self.cell.live_target(sid, u)
        return tg["T"], np.asarray(tg["insertion"], float)

    # ------------------------------------------------------------------ checks
    def pick_checks(self, u, T_u, G, w, spec, poses=None, n_branches=4):
        """(q_pick, q_pg, margin) for the grasp G (tcp pose in the unit frame) of the unit resting at T_u; else
        (None, None, reason)"""
        P = T_u @ G
        if spec["from_above"] and P[2, 2] > -0.5:
            return None, None, "approach not from above"
        P_pg = P.copy()
        P_pg[:3, 3] -= P[:3, 2] * spec["pre_grasp_m"]
        sols = self.ik(P, self.q_ready)
        if not sols:
            return None, None, "pick IK"
        width = min(self.max_width, w + spec["open_extra_m"])
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
        pre = PRE_INSERT_M
        cap = self.capture_depth(sid) - CAPTURE_MARGIN_M
        P_ = dict(poses or {})
        best = None
        T_pre = T_home.copy()
        T_pre[:3, 3] -= ins * pre
        sols = self.ik(T_pre @ G, self.q_ready if q_seed is None else q_seed)
        if not sols:
            return None, "pre-insertion IK", 0.0
        why = ""
        for _sc, q0, mrg in sols:
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

    def drop_checks(self, sid, u, G, w, T_home, ins, spec, poses=None, q_seed=None, height=BATTERY_RELEASE_M):
        """(q_pre, height, margin) for the grasp G letting the unit fall onto its seat: 60 mm above the release pose
        (`height` above home against the insertion) and down to it in 2 mm steps on one IK branch (the fingers
        BATTERY_FINGER_CLEAR_M off everything, 5 mm at the pre-release; the links 3 mm), the fingers opened there by
        the release opening, and the retreat 60 mm upwards; else (None, reason, 0)"""
        P_ = dict(poses or {})
        T_rel = T_home.copy()
        T_rel[:3, 3] -= ins * float(height)
        T_pre = T_rel.copy()
        T_pre[:3, 3] -= ins * PRE_INSERT_M
        sols = self.ik(T_pre @ G, self.q_ready if q_seed is None else q_seed)
        if not sols:
            return None, "pre-release IK", 0.0
        why = ""
        w_open = min(self.max_width, w + spec["release_extra_m"])
        fc = BATTERY_FINGER_CLEAR_M
        for _sc, q0, mrg in sols:
            xs = list(np.arange(PRE_INSERT_M, -1e-9, -0.002))
            q, ok_all = q0, True
            for i, x in enumerate(xs):
                T_x = T_rel.copy()
                T_x[:3, 3] -= ins * x
                P_[u] = T_x
                if i == 0:
                    qx, ok = q0, True
                else:
                    qx, ok = self.kin.solve_one((T_x @ G)[:3, 3], (T_x @ G)[:3, :3], q, q, margin=0.0)
                hit = []
                if ok:
                    self.pl.sync(allow=(u,), poses=P_, width=w)
                    hit = self.pl.hits(qx, finger_clear=fc if i else 0.005, link_clear=0.003 if i else None)
                if not ok or hit:
                    why = ("IK at %.0f mm above the release" % (x * 1e3)) if not ok else (
                        "hits %s at %.0f mm above the release" % (sorted(set((h[0], h[1]) for h in hit))[:3], x * 1e3))
                    ok_all = False
                    break
                q = qx
            if not ok_all:
                continue
            # the release opening at the release pose, the unit where it was let go
            P_[u] = T_rel
            self.pl.sync(allow=(u,), poses=P_, width=w_open)
            hit = self.pl.hits(q, finger_clear=fc, link_clear=0.003)
            if hit:
                why = "hits %s opening at the release" % (sorted(set((h[0], h[1]) for h in hit))[:3],)
                continue
            # the retreat upwards (moving away from a near contact allowed)
            T_out = T_rel @ G
            T_out[2, 3] += RETREAT_M
            qs = self.line_qs(T_out, T0=T_rel @ G, q0=q)
            if qs is None:
                why = "retreat IK"
                continue
            fc0 = self.pl.finger_clear
            self.pl.finger_clear = fc
            try:
                free = self.pl.line_free(qs[::3] + [qs[-1]])
            finally:
                self.pl.finger_clear = fc0
            if not free:
                why = "retreat blocked"
                continue
            return q0, float(height), float(mrg)
        return None, why or "release", 0.0

    def candidate_tiers(self, u):
        """the unit's grasp candidates by tier: the reference grasps, the second tier (`car_grasps.FALLBACK`), and,
        with family_fallback, the unit's family through the pad analysis on the kit (not dynamically validated)"""
        tiers = [list(self.grasps.get(u, []))]
        if self.fallback.get(u):
            tiers.append(list(self.fallback[u]))
        if self.family_fallback:
            if u not in self._family:
                self._family[u] = [(G, w, desc, dict(a, role="family")) for G, w, desc, a in CG.family_candidates(self.kit, u)]
            tiers.append(self._family[u])
        return tiers

    def feasible_grasps(self, u, T_pick, checks, spec, poses_pick=None, poses_place=None, limit=None):
        """[(score, G, w, desc, q_pick, q_pg, [hold per check], [validated pre-insertion posture per check])] best first,
        and the rejections; `checks` = [{"kind": "insert" | "drop", "slot", "T_home", "ins", "group_rel"}]. The
        candidate tiers in turn (`candidate_tiers`): a later tier only when no grasp of the earlier ones is feasible."""
        out, rejected = [], {}
        for tier in self.candidate_tiers(u):
            o, r = self._feasible(u, T_pick, checks, spec, tier, poses_pick, poses_place, limit)
            out += o
            rejected.update(r)
            if out:
                break
        return out, rejected

    def _feasible(self, u, T_pick, checks, spec, cands, poses_pick=None, poses_place=None, limit=None):
        out, rejected = [], {}
        com = np.array(self.kit["bodies"][u]["com_m"])
        for G, w, desc, _e in cands:
            q_pick, q_pg, r1 = self.pick_checks(u, T_pick, G, w, spec, poses=poses_pick)
            if q_pick is None:
                rejected[desc] = r1
                continue
            holds, q_pres, mrg, q_seed, why = [], [], r1, q_pick, None
            for ck in checks:
                if ck["kind"] == "drop":
                    q_pre, hold, r2 = self.drop_checks(ck["slot"], u, G, w, ck["T_home"], ck["ins"], spec, poses=poses_place,
                                                       q_seed=q_seed)
                    hold_score = 0.0
                else:
                    q_pre, hold, r2 = self.place_checks(ck["slot"], u, G, w, ck["T_home"], ck["ins"], poses=poses_place,
                                                        group_rel=ck.get("group_rel"), q_seed=q_seed)
                    hold_score = hold if q_pre is not None else 0.0
                if q_pre is None:
                    why = "%s: %s" % (ck["slot"], hold)
                    break
                holds.append(hold_score)
                q_pres.append(q_pre)
                mrg = min(mrg, r2)
                q_seed = q_pre
            if why:
                rejected[desc] = why
                continue
            score = mrg - 2.0 * float(np.linalg.norm(G[:3, 3] - com)) - 20.0 * sum(holds)
            out.append((score, G, w, desc, q_pick, q_pg, holds, q_pres))
            if limit and len(out) >= limit:
                break
        out.sort(key=lambda x: -x[0])
        return out, rejected

    # ------------------------------------------------------------------ one step
    def spec_of(self, u):
        return SPEC[self.type_of[u]]

    def place_unit(self, u):
        """pick the unit where it rests and put it on its slot (an insertion, or the stick pack's drop). Returns the
        step's record."""
        cell, sm = self.cell, self.sm
        spec = self.spec_of(u)
        sid = self.slot_of[u]
        rec = {"slot": sid, "unit": u, "kind": spec["kind"], "t0_s": round(float(self.d.time), 3), "placed_by": "nobody",
               "attempts": [], "carrier_unit": self.kit["bodies"][cell.slots[sid]["carrier"]]["unit"]}
        stats = self._new_stats(sid, u)
        n_contacts0 = len(self.contacts)
        self.work.update({"unit": u, "slot": sid, "held": None, "moving": None})
        self._phase("plan")
        t_wall = time.time()
        for round_ in range(3):
            T_pick = self.unit_T(u)
            T_home, ins = self.target(sid, u)
            group_rel = {g: _inv(T_pick) @ self.unit_T(g) for g in self.group_of(u) if g != u}
            checks = [{"kind": spec["kind"], "slot": sid, "T_home": T_home, "ins": ins, "group_rel": group_rel}]
            t_chk = time.time()
            feas, rejected = self.feasible_grasps(u, T_pick, checks, spec)
            rec.setdefault("grasps", []).append({"candidates": len(self.grasps.get(u, [])), "feasible": len(feas),
                                                 "tier": None if not feas else ("fallback" if feas[0][3].endswith("(P2-10 fallback)") else
                                                                                ("family" if " family " in feas[0][3] else "first")),
                                                 "check_wall_s": round(time.time() - t_chk, 1),
                                                 "rejected": {k: str(v) for k, v in list(rejected.items())[:8]}})
            if not feas:
                rec["why"] = "no feasible grasp (%s)" % (sorted(set(str(v) for v in rejected.values()))[:4],)
                self.log("  %s %s: %s" % (sid, u, rec["why"]))
                break
            score, G, w, desc, q_pick, q_pg, holds, q_pres = feas[0]
            att = {"round": round_, "grasp": desc, "width_mm": round(w * 1e3, 2), "hold_mm": [round(h * 1e3, 1) for h in holds],
                   "feasible": [f[3] for f in feas]}
            rec["attempts"].append(att)
            ok, why = self.pick(u, G, w, q_pick, q_pg, spec)
            if not ok:
                att["result"] = why
                self.log("  %s %s: pick failed (%s)" % (sid, u, why))
                if self.ct.width() < w + 0.5 * spec["open_extra_m"]:
                    self.let_go(w, spec)
                continue
            att["held"] = self.held_report(u, G)
            if spec["kind"] == "drop":
                ok, why = self.drop(sid, u, w, att, spec, q_branch=q_pres[-1])
            else:
                ok, why = self.insert(sid, u, w, att, spec, q_branch=q_pres[-1])
            att["result"] = why
            if ok:
                rec["placed_by"] = "ur"
                break
            self.log("  %s %s: placing failed (%s)" % (sid, u, why))
            if self.work["held"] is not None:
                self.let_go(w, spec)
            break
        stt = sm.rec[sid]
        rec["slot_state"] = {"state": sm.state[sid], "fastened": stt["fastened"], "achieved": stt["achieved"],
                             "pos_err_mm": stt["pos_err_mm"], "ang_err_deg": stt["ang_err_deg"], "t_s": stt["t_s"]}
        rec["coupling_events"] = [(round(float(t), 3), e) for t, e in sm.events if t is not None and t >= rec["t0_s"] - 1e-9
                                  and (sid in e or u in e)]
        stats.pop("_hist", None)
        rec.update({"state": sm.state[sid], "t1_s": round(float(self.d.time), 3), "wall_s": round(time.time() - t_wall, 1),
                    "timeline": stats["timeline"], "capture": stats["capture"],
                    "forces": {k: (round(v, 3) if isinstance(v, float) else v) for k, v in stats.items()
                               if k.endswith("_N") or k.endswith("_pair")},
                    "unplanned_contacts": [{k: v for k, v in c.items() if k != "key"} for c in self.contacts[n_contacts0:]]})
        rec["sim_s"] = round(rec["t1_s"] - rec["t0_s"], 3)
        self.work.update({"unit": None, "slot": None, "held": None, "moving": None, "phase": "between"})
        self.step_stats = None
        self.log("  %-8s %-3s -> %s (%s) sim %.1f s wall %.0f s, capture %s, unplanned %d" % (
            sid, u, rec["state"], rec["attempts"][-1].get("result") if rec["attempts"] else rec.get("why"), rec["sim_s"],
            rec["wall_s"], None if rec["capture"] is None else "%.2f of the window" % rec["capture"]["ratio_max"],
            len(rec["unplanned_contacts"])))
        return rec

    def held_report(self, u, G):
        """the unit's pose in the hand against the grasp's (mm, deg)"""
        G_act = _inv(self.tcp_T()) @ self.unit_T(u)
        E = G @ G_act                    # the tool in the unit (G) times the unit in the tool: identity when held as planned
        return {"offset_mm": round(float(np.linalg.norm(E[:3, 3])) * 1e3, 3), "angle_deg": round(math.degrees(rotation_angle(E[:3, :3])), 3),
                "width_mm": round(self.ct.width() * 1e3, 2)}

    def let_go(self, w, spec):
        """a unit the arm could not place stays where it is: open, back out along the approach, ready posture"""
        self._phase("let go", held=None)
        self.grip(min(self.max_width, w + spec["open_extra_m"]), settle_ticks=10)
        T_out = self.tcp_T()
        T_out[:3, 3] -= T_out[:3, 2] * RETREAT_M
        self.move_line(T_out, check=False)
        if not self.go(self.q_ready):
            self.move_q(self.q_ready)

    def pick(self, u, G, w, q_pick, q_pg, spec):
        w_open = min(self.max_width, w + spec["open_extra_m"])
        self._phase("to-pick", held=None)
        self.grip(w_open, settle_ticks=5)
        if not self.go(q_pg, width=w_open):
            return False, "no path to the pre-grasp (%s)" % getattr(self.pl, "why", "")
        self._phase("pick")
        P = self.unit_T(u) @ G
        ok, why = self.move_line(P, allow=(u,), width=w_open)
        if not ok:
            return False, "approach: " + why
        self.grip(max(0.0, w - GRIP_SQUEEZE_M))
        G_act = _inv(self.tcp_T()) @ self.unit_T(u)
        self._phase("lift", held=u)
        T_up = self.tcp_T()
        T_up[2, 3] += spec["lift_m"]
        ok, why = self.move_line(T_up, held=(u, G_act), check=True, speed=LIFT_SPEED_M_S)
        if not ok:
            ok, why = self.move_line(T_up, held=(u, G_act), check=False, speed=LIFT_SPEED_M_S)
        G_now = _inv(self.tcp_T()) @ self.unit_T(u)
        dp = float(np.linalg.norm(G_now[:3, 3] - G_act[:3, 3]))
        da = rotation_angle(G_now[:3, :3].T @ G_act[:3, :3])
        if dp > HELD_TOL_M or da > HELD_TOL_RAD or self.ct.width() < 0.002:
            self.work["held"] = None
            return False, "the unit moved in the hand (%.1f mm, %.1f deg)" % (dp * 1e3, math.degrees(da))
        return True, ""

    def carry_to(self, u, T_unit, w, spec, q_branch=None):
        """planned move with the unit held as it is now to the tool pose that puts it at T_unit (the validated branch
        first, then the natural branches nearest the current posture); returns the posture or None"""
        G_act = _inv(self.tcp_T()) @ self.unit_T(u)
        T_t = T_unit @ _inv(G_act)
        sols = self.ik(T_t, self.ct.q())
        if q_branch is not None:
            q_v, ok_v = self.kin.solve_one(T_t[:3, 3], T_t[:3, :3], q_branch, q_branch, margin=0.0)
            if ok_v:
                sols = [(0.0, q_v, 0.0)] + [x for x in sols if np.max(np.abs(x[1] - q_v)) > 1e-3]
        for _sc, q_pre, _m in sols[:6]:
            if self.go(q_pre, held=(u, G_act), width=w, v_max=spec["carry_v"]):
                return q_pre
        return None

    def insert(self, sid, u, w, att, spec, q_branch=None):
        """carry to the pre-insertion, then in along the live target until the cell captures the unit, on home and held
        there until the cell locks it, then let go and back out"""
        cell, ct, sm = self.cell, self.ct, self.sm
        pre = PRE_INSERT_M
        T_home, ins = self.target(sid, u)
        T_pre = T_home.copy()
        T_pre[:3, 3] -= ins * pre
        self._phase("carry")
        if self.carry_to(u, T_pre, w, spec, q_branch=q_branch) is None:
            return False, "no path to the pre-insertion"
        self._phase("insert")
        ct.set_stiffness(INSERT_STIFFNESS)
        x = pre
        for attempt in range(RETRIES + 1):
            # the hold as it is now (the carry, a sag or a stall may have moved the unit in the pads)
            Gi = self.tcp_T_inv_unit(u)
            ct.set_mode("cartesian")
            x = pre
            stall_ticks = 0
            done = False
            for _ in range(int((pre + 0.02) / (INSERT_SPEED_M_S * cell.tick_s)) + 200):
                T_home, ins = self.target(sid, u)
                x = max(-0.0005, x - INSERT_SPEED_M_S * cell.tick_s)
                T_u = T_home.copy()
                T_u[:3, 3] -= ins * x
                T_tcp = T_u @ Gi
                ct.command_cartesian(T_tcp[:3, 3], quat_wxyz(T_tcp[:3, :3]))
                cell.tick()
                if sm.state[sid] in ENGAGED:
                    done = True
                    break
                lag = float((T_u[:3, 3] - self.unit_T(u)[:3, 3]) @ ins)
                stall_ticks = stall_ticks + 1 if lag > STALL_M else 0
                if stall_ticks > 30:
                    break
            if done:
                break
            att.setdefault("stalls", []).append(round(x * 1e3, 1))
            T_back = self.tcp_T()
            T_back[:3, 3] -= ins * 0.015
            self.move_line(T_back, check=False, speed=0.02)
            pre = 0.015 + max(0.0, x)
        ct.set_stiffness(1.0)
        if sm.state[sid] not in ENGAGED:
            ct.set_mode("joint")
            return False, "not captured (stalled at %s mm)" % att.get("stalls")
        # captured: on home with the unit in the hand, held there until the cell locks it (the hand and the coupling
        # pull towards the same pose); let go only after that
        self._phase("captured")
        ct.set_mode("cartesian")
        ct.set_stiffness(INSERT_STIFFNESS)
        T_hold = self.tcp_T()
        for _ in range(LOCK_HOLD_TICKS):
            T_home, ins = self.target(sid, u)
            x = max(-0.0005, x - INSERT_SPEED_M_S * cell.tick_s)
            T_u = T_home.copy()
            T_u[:3, 3] -= ins * x
            T_hold = T_u @ Gi
            ct.command_cartesian(T_hold[:3, 3], quat_wxyz(T_hold[:3, :3]))
            cell.tick()
            if sm.state[sid] in ("LOCKED", "LOCKED_OFF"):
                break
        ct.set_stiffness(1.0)
        ct.set_mode("joint")
        att["state_at_release"] = sm.state[sid]
        return self.release(sid, u, w, spec)

    def release(self, sid, u, w, spec, retreat_dir=None):
        """open by the release opening, back out RETREAT_M along -approach (or `retreat_dir`, world), wait for the lock"""
        cell, sm = self.cell, self.sm
        self._phase("release", held=None, moving=u)
        self.grip(min(self.max_width, w + spec["release_extra_m"]), settle_ticks=10)
        T_out = self.tcp_T()
        d_out = -T_out[:3, 2] if retreat_dir is None else np.asarray(retreat_dir, float)
        T_out[:3, 3] += d_out * RETREAT_M
        self._phase("retreat")
        ok, _ = self.move_line(T_out, allow=(u,), check=True)
        if not ok:
            self.move_line(T_out, check=False)
        self._phase("wait")
        for _ in range(LOCK_WAIT_TICKS):
            if sm.state[sid] in ("LOCKED", "LOCKED_OFF"):
                break
            cell.tick()
        self.work["moving"] = None
        return sm.state[sid] in ("LOCKED", "LOCKED_OFF"), sm.state[sid]

    def drop(self, sid, u, w, att, spec, q_branch=None, height=BATTERY_RELEASE_M):
        """the stick pack: carried to 60 mm above its release pose (its underside `height` above its seat), lowered
        there at 30 mm/s, held 0.3 s, let go, backed out upwards; the cell captures it as it falls"""
        T_home, ins = self.target(sid, u)
        T_rel = T_home.copy()
        T_rel[:3, 3] -= ins * float(height)
        T_pre = T_rel.copy()
        T_pre[:3, 3] -= ins * PRE_INSERT_M
        self._phase("carry")
        if self.carry_to(u, T_pre, w, spec, q_branch=q_branch) is None:
            return False, "no path to the pre-release"
        self._phase("lower")
        G_act = _inv(self.tcp_T()) @ self.unit_T(u)
        T_home, ins = self.target(sid, u)
        T_rel = T_home.copy()
        T_rel[:3, 3] -= ins * float(height)
        T_tcp = T_rel @ _inv(G_act)
        ok, why = self.move_line(T_tcp, speed=BATTERY_LOWER_SPEED_M_S, held=(u, G_act), check=True, width=w,
                                 finger_clear=BATTERY_FINGER_CLEAR_M)
        if not ok:
            att["lower"] = why
            ok, why = self.move_line(T_tcp, speed=BATTERY_LOWER_SPEED_M_S, check=False)
        self.hold_cart(T_tcp, BATTERY_HOLD_TICKS)
        Tb = self.unit_T(u)
        E = _inv(T_home) @ Tb
        att["at_release"] = {"underside_above_seat_mm": round(float((Tb[:3, 3] - T_home[:3, 3]) @ -ins) * 1e3, 3),
                             "across_mm": round(float(np.linalg.norm((Tb[:3, 3] - T_home[:3, 3]) - ((Tb[:3, 3] - T_home[:3, 3]) @ ins) * ins)) * 1e3, 3),
                             "angle_deg": round(math.degrees(rotation_angle(E[:3, :3])), 3)}
        return self.release(sid, u, w, spec, retreat_dir=-ins)


def top_down_units(kit):
    """{unit type: unit} of the four top-down slots' candidates"""
    out = {}
    for s in kit["slots"]:
        typ = CG.SLOT_UNIT.get(s["id"])
        if typ in SPEC and SPEC[typ]["slot"] == s["id"]:
            out[typ] = s["candidates"][0]["unit"]
    return out


def program(cell, order="shop", units=None, log=print, planner_seed=0, end_ticks=300, assembler=None):
    """run the privileged assembler's four top-down steps on `cell` (a CarCell with the UR) in `order` ("shop" or
    "sub", or a sequence of unit types); returns the report"""
    t0 = time.time()
    s0 = float(cell.d.time)
    asm = assembler or Assembler(cell, log=log, planner_seed=planner_seed)
    by_type = top_down_units(cell.kit)
    seq = ORDERS[order] if isinstance(order, str) else tuple(order)
    if units is not None:
        seq = [t for t in seq if t in units or by_type.get(t) in units]
    asm.move_q(asm.q_ready)
    steps = []
    for t in seq:
        u = by_type[t]
        log("step %s <- %s" % (asm.slot_of[u], u))
        steps.append(asm.place_unit(u))
    asm.work.update({"phase": "end", "unit": None, "slot": None, "held": None, "moving": None})
    asm.go(asm.q_ready)
    asm.tick(end_ticks)
    sm = cell.sm
    slots = [asm.slot_of[by_type[t]] for t in seq]
    rep = {"order": order if isinstance(order, str) else list(order), "score": round(sm.score(), 4),
           "states": {s: sm.state[s] for s in slots}, "all_locked": all(sm.state[s] == "LOCKED" for s in slots),
           "steps": steps, "events": [(round(float(t), 3), e) for t, e in sm.events],
           "unplanned_contacts": [{k: v for k, v in c.items() if k != "key"} for c in asm.contacts],
           "sim_s": round(float(cell.d.time) - s0, 2), "wall_s": round(time.time() - t0, 1)}
    log("score %.3f, %s, sim %.0f s, wall %.0f s, unplanned contacts %d" % (rep["score"], rep["states"], rep["sim_s"], rep["wall_s"],
                                                                            len(rep["unplanned_contacts"])))
    return rep
