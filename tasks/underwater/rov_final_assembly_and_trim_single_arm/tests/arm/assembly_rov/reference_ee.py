"""The arm closes the vehicle: the privileged reference program, part III (the R4 segment and the whole episode). With
the vehicle deck up, the guards and the vertical thrusters on, it takes the electronics enclosure off its V-block rack
(gripped across the tube's diameter in the centre-plate gap, the hand tilted by the tube's roll on the rack), turns it
upright and dome forward, lowers it between the centre plates onto its two cradles, holds it still, lets it go and
backs out. `EEAssembler.full` chains the three segments into the episode (part I `reference.py`: weigh, trim, ballast,
turn; part II `reference_steps.py`: the guards and the thrusters; this file: the enclosure) and `reference_assemble`
is the entry point the cell's privileged call runs (server.RovInfoServer.reference_assemble). Parts I and II are not
changed: this file subclasses part II's `StepAssembler` (its IK family and modes, path checks, the planned carries with
their clearances, the strict line checks, the joint-mode lines from the commanded tcp, the monitor and the records).

Privileged state is used for the units' POSES only (DESIGN v1.2 section 3.2): where the enclosure lies on its rack
(`cell.body_T`: its roll about the tube axis, its dome end, the rack's pose), the enclosure in the hand as it is held,
the slot's live target (`cell.live_target`). The program never reads the coupling (no capture / lock feedback: the
enclosure is held still EE_HOLD_S whatever happens) and never reads the dome rule's or the watch's state. It reads the
kit's collision geometry of the base vehicle and of the enclosure (the given STL files carry the same shapes): the
centre-plate gap the fingers go down in (`ee_gap`) and the tube's width across the pads (`ee_grasps`), and the slot's
published capture depth (the cell manual's window table) to start its last line just outside the window.

Sources (copied, then rewritten for the product cell; none is imported):
  -dev/proto/bigparts/bp_trials.py sha256 82abfdc29d71f555a1c9f8f3cf78c44d823cf9e61d7547d4a975f6bb97608837
      (gate G6 trial (a): the tube gripped across its diameter at the centre of the centre-plate gap, the tool's approach
      tilted by the tube's roll on the rack (the part frame's -z), the alternative grasp turned 180 deg about the tool
      axis, the hand fully open on the 0.10 m approach, 100 N, the lift 0.10 m straight up, the target re-computed
      from the in-hand pose after the carry, lowered onto the cradles, held, released fully open, 80 mm back up)
  -dev/proto/bigparts/bp_cell.py sha256 72923f67b67fba18c211b8d303abb9d0244c4b798db40cd96220ba5d98a8c74a
      (EE_GRIP_B: the grip window centre on the tube axis, x midway in the centre-plate gap -17.9..43.4 mm)
  -dev/study/v3/mates_v3.py (P06; ee_grasps: x = the middle of the centre-plate gap from the vehicle's geometry, the
      closing axis across the tube, the approach -z tilted about x)
  docs/plan/driving/tasks/racecar_full_assembly_tenth_scale_ur10e-dev/run_reference.py (the episode's report shape:
      one list of step records, the unplanned contacts, the checkpoints; read only)
Gate G6's lessons and the rulings, kept as rules of this program:
  * L19: the yaw window of the enclosure is 1.5 deg (2.5 deg lands the tube's end caps on the centre plates' inner
    edges and the cradles, 154 N); the program's budget is EE_YAW_MAX_DEG 1.0 deg and EE_ROLL_MAX_DEG 2.0 deg (logged
    as `placement_predicted` from the re-read in-hand pose, and measured at the arrival: `at_arrive`);
    `past_mm` 2.0: the tube rests 1.49 mm below home on the saddles once let go, still inside the window;
  * the centre plates' inner edges are 3.35 mm off the tube: the enclosure goes down from EE_PRE_PLACE_M over home
    upright and straight (the target from the in-hand pose re-read after the carry, the arm settled, against the
    COMMANDED tcp: part I's rule) and only its last line (the capture depth + CAPTURE_MARGIN_M) is inside the window
    (part II's rule: captured on it, the weld's pull keeps pace with the hand);
  * the fingers go down in the centre-plate gap (`ee_gap`: the largest gap along x of the base vehicle's collision
    geometry in the fingers' slab beside the tube): the grasp centre at the gap's middle (EE_X_OFFSETS_MM around it);
  * the hand is set fully open BEFORE the move to the pre-grasp (part I's rule: the path checked at the opening the
    fingers have), the carry planned on a kinematic copy with everything CARRY_CLEAR_M + CARRY_PLAN_MARGIN_M clear
    (part II's `plan_carry`: straight, via waypoints, RRT-Connect), re-planned from where the arm is after the lift;
  * the dome is never touched: the fingers stay in the gap between the centre plates (the tube's middle), the
    released hand goes straight up.
Carrying: part I's limits (ruling M8: tcp <= 0.25 m/s, joints <= 8 rad/s^2; planned at 0.20 m/s and, for the 2.4 kg
enclosure, EE_CARRY_JOINT_ACC_PLAN 5.0 rad/s^2: at part I's 6.0 the arm overshot to 7.4 rad/s^2 in the pilot runs).

The step (`EEAssembler.enclosure`, `EEAssembler.r4`):
  plan      grasp candidates (`ee_grasps`: the hand either way round about the tool axis, x offsets in the gap); for each
            and each IK branch at the grasp: the pre-grasp line (EE_PRE_M back along the tilted approach, the hand fully
            open), the lift (EE_LIFT_M straight up, the rack allowed), the place lines (from EE_PRE_PLACE_M over home
            down to the window, then the last line to HOVER_M short of home; the vehicle allowed on the last line
            only), the retreat (EE_RETREAT_M straight up, open) checked, the carry planned; the first with every piece
            clear is run
  run       fully open, transit (empty hand, part I's `plan_path`), the approach line, close (EE_GRIP_N,
            EE_GRIP_SETTLE_S), lift, carry (re-planned), settle, the in-hand pose re-read, down to the window, the last
            line, hold EE_HOLD_S, open fully, retreat
  then      the ready posture.
The episode (`EEAssembler.full`, `program_full`, `reference_assemble`): part I's R1 (the work order, the weighing
through the cell's public readings, the public trim, the ballast, the turn deck up), part II's R2 / R3 (guards first by
default), the enclosure, the ready posture. The report: `steps` (every block, guard, thruster and the enclosure, in
the order run), the three segments' reports, the whole episode's carry log, unplanned contacts and checkpoints.
"""
import math
import time

import numpy as np

from .coupling import _inv, rotation_angle
from . import reference as R1
from . import reference_steps as R23
from .reference import CARRY_TCP_PLAN_M_S, EMPTY_JOINT_ACC, EMPTY_TCP_M_S, _T
from .reference_steps import StepAssembler, _shift

EE_SLOT = "ee"
EE_RACK = "enclosure_rack"
# ---- the budget of the placement (ruling L19: the window is 1.5 deg of yaw; this program keeps within these)
EE_YAW_MAX_DEG = 1.0
EE_ROLL_MAX_DEG = 2.0
# ---- the grasp
EE_GRIP_N = 100.0                # G6 trial (a)
EE_GRIP_SETTLE_S = 1.0
EE_X_OFFSETS_MM = (0.0, -4.0, 4.0)   # the grasp centre from the middle of the centre-plate gap, along the tube axis
EE_SLAB_Y_MM = (-7.0, 18.0)      # the fingers' slab beside the tube: from the tube's radius (|y| in the vehicle frame)
EE_SLAB_Z_MM = (-60.0, 60.0)     # and from the tube's axis (z); the gap is the largest one along x in it
EE_GAP_X_MM = (-120.0, 150.0)    # where the gap is looked for (between the two cradles, vehicle frame x)
EE_PAD_HALF_M = 0.0325           # half the pad face's length (65 mm): the tube's width is taken over it
# ---- the motions
EE_PRE_M = 0.10                  # the pre-grasp: this far back along the (tilted) approach, the hand fully open
EE_LIFT_M = 0.10                 # off the V blocks, straight up
EE_PRE_PLACE_M = (0.09, 0.12, 0.15)   # the pre-place height over home (the carry's end), first that clears
EE_RETREAT_M = 0.12              # straight up after the release (the fingers out of the gap)
EE_HOLD_S = 1.0                  # self-standing: held still while it is captured and locks; then let go
EE_CARRY_JOINT_ACC_PLAN = 5.0    # the enclosure's carry is planned below part I's 6.0 rad/s^2: with the 2.4 kg tube the arm's
                                 # computed-torque loop overshot 6.0 to 7.2-7.4 rad/s^2 (pilot runs, the limit is 8)
V_DOWN_HELD = R23.V_DOWN_HELD
V_IN = R23.V_IN
V_EMPTY_LINE = R23.V_EMPTY_LINE
HOVER_M = R23.HOVER_M


def _rpy_small(R):
    """(roll about x, pitch about y, yaw about z) in deg of a small rotation (its rotation vector's components)"""
    from scipy.spatial.transform import Rotation
    rv = Rotation.from_matrix(np.asarray(R, float)).as_rotvec()
    return tuple(math.degrees(float(v)) for v in rv)


# ====================================================================================================== the assembler
class EEAssembler(StepAssembler):
    """part II's StepAssembler + the enclosure step and the whole episode (module doc)"""

    def __init__(self, cell, log=print, read_fixture=None, monitor=True, kin=None):
        super(EEAssembler, self).__init__(cell, log=log, read_fixture=read_fixture, monitor=monitor, kin=kin)
        self.ee_unit = [u for u in cell.loose if self.kit["bodies"][u]["type"] == "electronics_enclosure"][0]

    # ------------------------------------------------------------------ geometry (the kit's collision shapes)
    def _tube_radius(self, u):
        """the enclosure tube's outer radius (m): the largest |(y, z)| of the tube piece's collision vertices (the
        enclosure's frame E: x along the tube axis)"""
        from .bench_plan import _quat_R, _read_stl
        import os
        best = 0.0
        for g in self.kit["bodies"][u]["collision"]:
            if g["type"] != "mesh" or "tube" not in g.get("name", ""):
                continue
            V = _read_stl(os.path.join(self.kit["dir"], g["file"])) @ _quat_R(g.get("quat", [1, 0, 0, 0])).T + \
                np.asarray(g.get("pos", [0, 0, 0]), float)
            best = max(best, float(np.max(np.hypot(V[:, 1], V[:, 2]))))
        if best <= 0.0:
            raise RuntimeError("no tube piece in the enclosure's collision")
        return best

    def ee_gap(self):
        """(x0, x1) m in the vehicle frame: the centre-plate gap the fingers go down in = the largest gap along x of
        the base vehicle's collision vertices in the fingers' slab beside the tube (|y| = the tube's radius +
        EE_SLAB_Y_MM, z = the tube axis at home + EE_SLAB_Z_MM), between EE_GAP_X_MM"""
        if getattr(self, "_ee_gap", None) is None:
            from .bench_plan import collision_points
            u = self.ee_unit
            P, _S = collision_points(self.kit, self.kit["base_body"])
            c = self.cell.candidate(EE_SLOT, u)
            za = float(np.asarray(c["T_part_in_carrier"], float)[2, 3])        # the tube axis (E's x axis) at home
            r = self._tube_radius(u)
            ay = np.abs(P[:, 1])
            sel = P[(ay >= r + EE_SLAB_Y_MM[0] / 1e3) & (ay <= r + EE_SLAB_Y_MM[1] / 1e3) &
                    (P[:, 2] >= za + EE_SLAB_Z_MM[0] / 1e3) & (P[:, 2] <= za + EE_SLAB_Z_MM[1] / 1e3) &
                    (P[:, 0] >= EE_GAP_X_MM[0] / 1e3) & (P[:, 0] <= EE_GAP_X_MM[1] / 1e3)]
            xs = np.unique(np.round(sel[:, 0], 5))
            if len(xs) < 2:
                raise RuntimeError("no centre-plate gap found in the vehicle's collision geometry")
            k = int(np.argmax(np.diff(xs)))
            self._ee_gap = (float(xs[k]), float(xs[k + 1]))
        return self._ee_gap

    def ee_grasps(self, u):
        """[(G = the tcp in the enclosure's frame E, opening m, meta)]: the closing axis across the tube (E's y), the
        approach down E's -z (tilted with the tube's roll on the rack, vertical at home), the grasp centre on the tube
        axis at x = the gap's middle + EE_X_OFFSETS_MM (E's x = the vehicle's x at home), the hand either way round
        about the tool axis (`turn`); the opening = the tube's width across the pads (its collision vertices within
        the pad's length of the axis)"""
        from . import gripper_wide as GW
        from .bench_plan import _quat_R, _read_stl
        import os
        g0, g1 = self.ee_gap()
        c = self.cell.candidate(EE_SLOT, u)
        Tpc = np.asarray(c["T_part_in_carrier"], float)                    # E at home in the vehicle frame
        xm = 0.5 * (g0 + g1) - float(Tpc[0, 3])
        ys = []
        for g in self.kit["bodies"][u]["collision"]:
            if g["type"] != "mesh":
                continue
            V = _read_stl(os.path.join(self.kit["dir"], g["file"])) @ _quat_R(g.get("quat", [1, 0, 0, 0])).T + \
                np.asarray(g.get("pos", [0, 0, 0]), float)
            lo, hi = float(V[:, 0].min()), float(V[:, 0].max())
            if hi < xm - 0.02 or lo > xm + 0.02:                            # the pieces under the pads (the tube)
                continue
            ys.append(V[np.abs(V[:, 2]) <= EE_PAD_HALF_M, 1])
        ys = np.concatenate(ys) if ys else np.zeros(0)
        if not len(ys):
            raise RuntimeError("no enclosure geometry under the pads")
        w = float(ys.max() - ys.min())
        yc = 0.5 * float(ys.max() + ys.min())
        po = GW.pad_offset(w)
        out = []
        for turn in (False, True):
            s = -1.0 if turn else 1.0
            Rt = np.column_stack([[s, 0.0, 0.0], [0.0, -s, 0.0], [0.0, 0.0, -1.0]])
            for dx_mm in EE_X_OFFSETS_MM:
                cen = np.array([xm + dx_mm / 1000.0, yc, 0.0])
                p = cen - Rt @ po
                out.append((_T(p, Rt), w, {"turn": bool(turn), "x_E_mm": round(1e3 * float(cen[0]), 2), "dx_mm": dx_mm,
                                           "width_mm": round(1e3 * w, 3), "gap_B_mm": [round(1e3 * g0, 2), round(1e3 * g1, 2)]}))
        return out

    def ee_error(self, u):
        """the enclosure against its home (geometry only, from the live target): the position error (mm, world),
        along the insertion (mm, < 0 short of home), lateral (mm), the tilt (deg), and the small rotation's roll
        (about the tube axis, E x), pitch (E y) and yaw (about the vertical, E z), deg"""
        out = self.slot_error(EE_SLOT, u)
        tg = self.cell.live_target(EE_SLOT, u)
        rel = tg["T"][:3, :3].T @ self.unit_T(u)[:3, :3]
        r, p, y = _rpy_small(rel)
        out.update({"roll_deg": round(r, 4), "pitch_deg": round(p, 4), "yaw_deg": round(y, 4)})
        return out

    # ------------------------------------------------------------------ the enclosure
    def plan_ee(self, u, rec=None):
        """the enclosure's plan (module doc): the first (grasp, branch at the grasp, branch at the place, pre-place
        height) with every line clear and a carry; the plan dict (RuntimeError with the failure counts when none)"""
        rec = {} if rec is None else rec
        c = self.cell
        t_plan = time.process_time()
        T0 = self.unit_T(u)
        tg = c.live_target(EE_SLOT, u)
        Th = tg["T"]
        ins = tg["insertion"]                                   # down
        holders = self.unit_holders(EE_SLOT)
        open_w = float(self.ct.max_width)
        gap = self.approach_gap(EE_SLOT)
        q_now = self.ct.q_cmd.copy()
        tried = []
        chosen = None
        for G, w, meta in self.ee_grasps(u):
            Tg = T0 @ G
            zt = Tg[:3, 2]
            T_pre = _shift(Tg, -EE_PRE_M * zt)
            T_lift = _shift(Tg, [0.0, 0.0, EE_LIFT_M])
            Hg = (u, _inv(G))
            Tp_hov = _shift(Th @ G, -HOVER_M * ins)
            Tp_win = _shift(Tp_hov, -gap * ins)
            Tr_up = _shift(Tp_hov, -EE_RETREAT_M * ins)
            for qg in self.branches(Tg, q_now):
                key = (meta["turn"], meta["dx_mm"])
                if not self.in_family(qg):
                    tried.append((key, "grasp branch off the family"))
                    continue
                pick = self.track([Tg, T_pre], qg)
                lift = self.track([Tg, T_lift], qg)
                if pick is None or lift is None:
                    tried.append((key, "pick IK"))
                    continue
                q_pre = pick[1]
                self.ikmode = "free"
                try:
                    down, _s = self.line_plan(Tg, R1.V_APPROACH_EMPTY, T0=T_pre, q0=q_pre)
                finally:
                    self.ikmode = "down"
                if down is None or not self.check_line(self.pl, [q_pre] + down, allow=(u,), width=open_w):
                    tried.append((key, "pick line"))
                    continue
                qs_l = self._line_postures([Tg, T_lift], qg)
                if qs_l is None or not self.check_line(self.pl, qs_l, held=Hg, width=w, held_allow=(EE_RACK,)):
                    tried.append((key, "lift line"))
                    continue
                ok2, hits = self.posture_ok(self.pl_carry, lift[1], held=Hg, width=w)
                if not ok2:
                    tried.append((key, "lift pose within the carry clearance: %s" % sorted(set(h[1] for h in hits))[:3]))
                    continue
                placed = None
                for h_pre in EE_PRE_PLACE_M:
                    Tp_pre = _shift(Tp_hov, -h_pre * ins)
                    for qh in self.branches(Tp_hov, lift[1]):
                        if not self.in_family(qh):
                            continue
                        up = self.track([Tp_hov, Tp_win, Tp_pre], qh)
                        back = self.track([Tp_hov, Tr_up], qh)
                        if up is None or back is None:
                            continue
                        ok3, _h = self.posture_ok(self.pl_carry, up[2], held=Hg, width=w)
                        if not ok3:
                            continue
                        qs_down = self._line_postures([Tp_pre, Tp_win], up[2])
                        qs_last = self._line_postures([Tp_win, Tp_hov], up[1])
                        if qs_down is None or qs_last is None:
                            continue
                        if not self.check_line(self.pl_fine, qs_down, held=Hg, width=w):
                            continue
                        if not self.check_line(self.pl_fine, qs_last, held=Hg, width=w, held_allow=(c.base,) + holders):
                            continue
                        qs_back = self._line_postures([Tp_hov, Tr_up], qh)
                        if qs_back is None or not self.check_line(self.pl, qs_back, width=open_w, allow=(u,)):
                            continue
                        placed = (qh, up, back, h_pre, Tp_pre)
                        break
                    if placed is not None:
                        break
                if placed is None:
                    tried.append((key, "place lines"))
                    continue
                qh, up, back, h_pre, Tp_pre = placed
                vias = self._carry_vias(lift[1], up[2], T_lift, Tp_pre)
                path, how = self.plan_carry(lift[1], up[2], Hg, w, vias)
                if path is None:
                    tried.append((key, "carry: %s" % how))
                    continue
                chosen = dict(G=G, w=w, meta=meta, qg=qg, q_pre=q_pre, down=down, Tg=Tg, T_pre=T_pre, T_lift=T_lift,
                              Tp_pre=Tp_pre, Tp_win=Tp_win, Tp_hov=Tp_hov, Tr_up=Tr_up, up=up, carry=path, carry_how=how,
                              h_pre=h_pre, open_w=open_w)
                break
            if chosen is not None:
                break
        rec["candidates_tried"] = len(tried)
        rec["plan_cpu_s"] = round(time.process_time() - t_plan, 2)
        if chosen is None:
            import collections
            rec["tried_why"] = dict(collections.Counter(t[1].split(":")[0] for t in tried))
            raise RuntimeError("no enclosure plan (%d tried: %s)" % (len(tried), rec["tried_why"]))
        X = chosen
        rec.update({"grasp": X["meta"], "width_mm": round(1e3 * X["w"], 3), "pre_place_m": X["h_pre"],
                    "turn_deg": round(math.degrees(rotation_angle(T0[:3, :3].T @ Th[:3, :3])), 2)})
        cl, pair, _ns = self.carry_clearance(X["carry"], (u, _inv(X["G"])), X["w"])
        rec["carry_clearance_plan_mm"] = round(1e3 * cl, 2) if pair is not None else None
        return X

    def enclosure(self, u=None):
        u = u or self.ee_unit
        rec = {"slot": EE_SLOT, "unit": u, "kind": "enclosure", "placed_by": "nobody", "t0_s": round(self.t(), 3)}
        self.records[EE_SLOT] = rec
        self._blocks[EE_SLOT] = rec
        self._uf_hist = []
        self.work.update({"site": EE_SLOT, "unit": u})
        try:
            self._enclosure(u, rec)
            rec["placed_by"] = "ur"
        except RuntimeError as e:
            rec["why"] = str(e)
            self.log("  %s <- %s: %s" % (EE_SLOT, u, e))
            if self.work.get("held"):
                self.let_go()
        self.set_phase("idle", held=False, unit=None, site=None)
        rec["t1_s"] = round(self.t(), 3)
        rec["unit_s"] = round(rec["t1_s"] - rec["t0_s"], 3)
        self.log("  %-8s <- %s %s  %.1f s sim" % (EE_SLOT, u, rec["placed_by"], rec["unit_s"]))
        return rec

    def _enclosure(self, u, rec):
        c = self.cell
        X = self.plan_ee(u, rec)
        w, open_w = X["w"], X["open_w"]
        # ---- transit to the pre-grasp, the hand fully open first
        self.set_phase("to-pick", held=False, unit=u)
        width_chk = self.set_opening(open_w)
        q_now = self.ct.q_cmd.copy()
        vias = [("over", [self.high_of_free(X["T_pre"], X["q_pre"])]),
                ("up", [self.high_of_free(self.tcp_cmd_T(), q_now)]),
                ("up-over", [self.high_of_free(self.tcp_cmd_T(), q_now), self.high_of_free(X["T_pre"], X["q_pre"])])]
        path, how = self.plan_path(q_now, X["q_pre"], width=width_chk, vias=vias)
        if path is None:
            raise RuntimeError("no path to the pre-grasp of %s: %s" % (u, how))
        rec["path_to_pick"] = how
        self.follow_smooth(path, EMPTY_JOINT_ACC, EMPTY_TCP_M_S)
        self.wait(0.1)
        self.set_phase("approach-pick")
        self.run_line(X["down"])
        self.set_phase("grip")
        self.grip(0.0, force_N=EE_GRIP_N, settle_s=EE_GRIP_SETTLE_S)
        G0 = _inv(self.tcp_cmd_T()) @ self.unit_T(u)
        # ---- lift off the V blocks
        self.set_phase("lift", held=True)
        self._run_to(X["T_lift"], R1.V_LIFT)
        G1 = _inv(self.tcp_cmd_T()) @ self.unit_T(u)
        rec["in_hand_shift_lift_mm"] = round(1e3 * float(np.linalg.norm(G1[:3, 3] - G0[:3, 3])), 4)
        rec["in_hand_turn_lift_deg"] = round(math.degrees(rotation_angle(G0[:3, :3].T @ G1[:3, :3])), 4)
        rec["pad_width_mm"] = round(1e3 * float(self.ct.width()), 3)
        if self.ct.width() < 0.5 * w:
            raise RuntimeError("%s not in the hand after the lift" % u)
        # ---- the carry (re-planned from where the arm is, the enclosure as it is held now)
        self.set_phase("carry")
        Hg = (u, G1)
        q_now = self.ct.q_cmd.copy()
        path, how = self.plan_carry(q_now, X["up"][2], Hg, w, self._carry_vias(q_now, X["up"][2], self.tcp_cmd_T(), X["Tp_pre"]))
        if path is None:
            raise RuntimeError("no carry for %s: %s" % (u, how))
        rec["carry_how"] = how
        rec["carry_waypoints"] = len(path)
        q_log = []
        self._qlog = q_log
        self.follow_smooth(path, EE_CARRY_JOINT_ACC_PLAN, CARRY_TCP_PLAN_M_S)
        self._qlog = None
        cl, pair, ns = self.carry_clearance(path, Hg, w)
        rec["carry_clearance_planned_mm"] = round(1e3 * cl, 2) if pair is not None else None
        rec["carry_clearance_planned_pair"] = pair
        if q_log:
            cle, paire, _n = self.carry_clearance(q_log, Hg, w, step=1.0)
            rec["carry_clearance_executed_mm"] = round(1e3 * cle, 2) if paire is not None else None
            rec["carry_clearance_executed_pair"] = paire
        # ---- the place: settle, the in-hand pose re-read, the target, down to the window, the last line, hold
        self.wait_arrived(tol=R23.SETTLE_TOL_RAD, timeout_s=1.0)
        self.wait(R23.SETTLE_S)
        G2 = _inv(self.tcp_cmd_T()) @ self.unit_T(u)
        rec["carry_creep_mm"] = round(1e3 * float(np.linalg.norm(G2[:3, 3] - G1[:3, 3])), 4)
        tg = c.live_target(EE_SLOT, u)
        ins = tg["insertion"]
        T_hov = _shift(tg["T"] @ _inv(G2), -HOVER_M * ins)
        T_win = _shift(T_hov, -self.approach_gap(EE_SLOT) * ins)
        pred = T_hov @ G2
        r_, p_, y_ = _rpy_small(tg["T"][:3, :3].T @ pred[:3, :3])
        rec["placement_predicted"] = {"roll_deg": round(r_, 5), "pitch_deg": round(p_, 5), "yaw_deg": round(y_, 5),
                                      "yaw_limit_deg": EE_YAW_MAX_DEG, "roll_limit_deg": EE_ROLL_MAX_DEG}
        d_pre = float(np.linalg.norm(_shift(T_hov, -X["h_pre"] * ins)[:3, 3] - self.tcp_cmd_T()[:3, 3]))
        rec["carry_end_off_mm"] = round(1e3 * d_pre, 3)
        if d_pre > 0.02:
            raise RuntimeError("the carry ended %.1f mm from the pre-place pose" % (1e3 * d_pre))
        self.set_phase("approach", held=True, sub="down")
        self._run_to(T_win, V_DOWN_HELD)
        self.set_phase("approach", held=True, sub="last")
        self._run_to(T_hov, V_IN)
        rec["t_arrive_s"] = round(self.t(), 3)
        rec["at_arrive"] = self.ee_error(u)
        self.set_phase("hold")
        self.wait(EE_HOLD_S)
        rec["t_release_s"] = round(self.t(), 3)
        rec["at_release"] = self.ee_error(u)
        rec["placement"] = {"yaw_deg": rec["at_arrive"]["yaw_deg"], "roll_deg": rec["at_arrive"]["roll_deg"],
                            "yaw_limit_deg": EE_YAW_MAX_DEG, "roll_limit_deg": EE_ROLL_MAX_DEG}
        # ---- release, retreat
        self.set_phase("release", held=False)
        self.ct.command_width(open_w)
        self.wait(R1.RELEASE_WAIT_S + 0.1)
        rec["at_release_open"] = self.ee_error(u)
        self.set_phase("retreat")
        self._run_to(_shift(self.tcp_cmd_T(), -EE_RETREAT_M * ins), V_EMPTY_LINE, check=True, width=float(self.ct.width()))
        rec["after_retreat"] = self.ee_error(u)

    def r4(self):
        """the R4 segment: the enclosure, then the ready posture; the report"""
        rep = {"t0_s": round(self.t(), 3)}
        t_wall = time.time()
        rep["unit"] = self.enclosure()
        self.checkpoint("unit %s" % EE_SLOT)
        rep["path_to_ready"] = self.go_ready()
        rep["t1_s"] = round(self.t(), 3)
        rep["wall_s"] = round(time.time() - t_wall, 1)
        return rep

    # ------------------------------------------------------------------ the episode
    def full(self, work_order, order="guards_first", rho=None, rho_read_err=0.0):
        """the whole episode (module doc): R1 (weigh, trim, ballast, turn), R2 / R3 (`order`), R4, the ready posture"""
        c = self.cell
        rep = {"t0_s": round(self.t(), 3), "order": order}
        t_wall = time.time()
        self.go_ready()
        rep["r1"] = self.r1(work_order, rho=rho, rho_read_err=rho_read_err)
        if not c.done:
            rep["r23"] = self.r23(order=order)
        if not c.done:
            rep["r4"] = self.r4()
        steps = list(rep["r1"].get("blocks") or [])
        if "r23" in rep:
            steps += list(rep["r23"].get("units") or [])
        if "r4" in rep:
            steps.append(rep["r4"]["unit"])
        rep["steps"] = steps
        rep["t1_s"] = round(self.t(), 3)
        rep["sim_s"] = round(rep["t1_s"] - rep["t0_s"], 3)
        rep["wall_s"] = round(time.time() - t_wall, 1)
        rep["cell_done"] = c.done
        rep["carry"] = self.carry_log()
        rep["unplanned_contacts"] = [{k: v for k, v in x.items() if k != "key"} for x in self.contacts]
        rep["checkpoints"] = list(self.checkpoints)
        rep["placed"] = sum(1 for s in steps if s.get("placed_by") == "ur")
        rep["not_placed"] = [(s.get("site") or s.get("slot"), s.get("why")) for s in steps if s.get("placed_by") != "ur"]
        return rep


def program_r4(cell, log=print, asm=None, read_fixture=None):
    """run the R4 segment on `cell` (deck up, the guards and thrusters on); asm: an EEAssembler to go on from or None"""
    if asm is None or not isinstance(asm, EEAssembler):
        asm = EEAssembler(cell, log=log, read_fixture=read_fixture)
    return asm.r4(), asm


def program_full(cell, work_order, read_fixture=None, log=print, order="guards_first", rho=None, rho_read_err=0.0):
    """run the whole episode of the privileged reference program on `cell` (a RovCell with the arm, keel up at the
    start); the report and the assembler"""
    asm = EEAssembler(cell, log=log, read_fixture=read_fixture)
    rep = asm.full(work_order, order=order, rho=rho, rho_read_err=rho_read_err)
    return rep, asm


def reference_assemble(cell, server=None, log=None, order="guards_first"):
    """the cell's privileged call (server.RovInfoServer.reference_assemble): the whole episode on the server's cell,
    the work order from the episode (= describe()["work_order"]), the weighing through the server's public readings
    (get_state()["fixture"]: the load cells' noise of this episode, one sample per tick). Returns the report with
    `ok`, `steps`, `unplanned_contacts`"""
    log = log or (lambda *a: None)
    wo = server.entry["work_order"] if server is not None else None
    read = (lambda: server.get_state()["fixture"]) if server is not None else None
    rep, _asm = program_full(cell, wo, read_fixture=read, log=log, order=order)
    rep["ok"] = True
    return rep
