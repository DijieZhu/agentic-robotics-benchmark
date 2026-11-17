"""The UR10e builds the propulsion side of the vehicle: the privileged reference program, part II (the R2 / R3
segments). With the vehicle deck up it sets the two thruster guards on their lower brackets and the four vertical
thrusters on the side plates, each held still on its place before the fingers open (the two non-self-standing kinds:
DESIGN v1.2 section 1.2, ruling L19: hold >= 1.5 s), guards first by default or thrusters first (both orders score,
DESIGN 1.1 / B1). Part I (`reference.py`: weigh, trim, ballast, turn) is not changed: this file subclasses its
`Assembler` (its IK family, path checks, smooth carries, joint-mode lines from the commanded tcp, monitor and records).
Part III (the enclosure, the whole episode) goes on from here.

Privileged state is used for the units' POSES only (DESIGN 3.2): where each guard lies on its rack and each thruster
stands in its nest (`cell.body_T`; which rack a guard is on: the layout), the unit in the hand as it is held, the
slot's live target (`cell.live_target`), and the units' part types (which thruster carries which propeller hand, for
the slots' design part). The program never reads the coupling: no capture or lock feedback (it holds every unit
HOLD_S whatever happens); the units it has already set are its own records. It reads the slots' published capture
depth (the cell manual's window table) to start its last line just outside the window.

Sources (copied, then rewritten for the product cell; none is imported):
  -dev/proto/bigparts/bp_trials.py sha256 82abfdc29d71f555a1c9f8f3cf78c44d823cf9e61d7547d4a975f6bb97608837
      (gate G6's large-unit motions: the guard clamped on its outer edge from outboard, lowered STANDOFF_GUARD_M off the
      side plate, moved onto the plate, the last millimetres down; the thruster held on the duct's cylindrical band,
      down through the guard's cut-out off the plate, along the insertion onto the plate; the hold before the release;
      the retreat: the guard outboard, the thruster up; the guard's grasp family x / bite / yaw)
  -dev/proto/bigparts/run_trials.py sha256 d5e5cd463d7445e737e58fc63d3236981b1b24e3d327f6c114efd535fb3b69d9 and
  -dev/proto/bigparts/bp_cell.py sha256 72923f67b67fba18c211b8d303abb9d0244c4b798db40cd96220ba5d98a8c74a (the trial flow)
  <task>/tests/arm/assembly_rov/motion.py sha256 1a84f9e67726f99f9c6837be589503ca330a4c4f55cb8d3ad3cbc40622a311e5 (the
      vendored RRT-Connect planner, through reference.SweepPlanner; imported, not copied)
  -dev/records/P2_P06.md section 5 (the guard's seven solid cross-sections, the thruster's grasp band at B z -3.5 and
      the <= 30 deg closing axis, the 80 mm approach)
G6's lessons and the rulings, kept as rules of this program:
  * L19: every guard / thruster held still on its place HOLD_S (1.5 s) before the fingers open; the placement tilt
    (`placement` of each record: the unit's insertion axis against the slot's when the hand arrives) is logged against
    GUARD_TILT_MAX_DEG (0.5 deg) and THR_TILT_MAX_DEG (1.0 deg; G6 g4win: 1.45 deg missed the narrowed 1.4 deg
    window). The target is computed from the unit's in-hand pose re-read after the carry, the arm settled
    (SETTLE_TOL_RAD, SETTLE_S), against the COMMANDED tcp (part I's rule), so what is left is the hand's drift on the
    last lines;
  * the last line alone inside the capture window: it starts approach_gap() = the slot's capture depth +
    CAPTURE_MARGIN_M short of home along the insertion, so the unit is captured on it (the 10-tick rest gate passes at
    V_IN) and the soft weld's pull keeps pace with the hand; the lock follows ~0.5 s into the hold. A first version
    moved the guard in at 3 mm over its brackets, inside the 3 mm depth window: captured on the way in, pulled home
    ahead of the hand and locked before the hand arrived (the pads dragged over the plate, 45 N, a joint at 8.6 rad/s^2);
  * L13: the narrowed angular windows (thruster 1.4 / 1.8 deg, guard 1.1 / 0.7 deg) for both orders;
  * G6 7: the guard goes down STANDOFF_GUARD_M (8 mm) off the side plate (its ends swing with the yaw), then in, then
    down; the thruster is held on the duct's cylindrical band, the fingertips THR_TIP_OVER_GUARD_M over the guard's top
    face (on the taper below it the pads squeeze it down), the closing axis within 30 deg of the vehicle's x (more and
    the gripper's linkage meets the side plate);
  * L19 / G6 6: each guard has its own side's rack; every carry (both guards and the thrusters) is planned by
    `plan_carry` (straight; else via waypoints; else RRT-Connect, motion.Planner.plan) on a kinematic copy of the cell
    and is accepted only when the arm (links, gripper, fingers) and the held unit stay CARRY_CLEAR_M (10 mm) +
    CARRY_PLAN_MARGIN_M (1.5 mm, for the arm's tracking) off everything else, sampled every 0.004 rad (part I's
    sweep step); the accepted path is re-sampled at 0.001 rad and the executed carry (the postures of every tick)
    measured again on a third copy with 30 mm margins (`carry_clearance`). Every carry ends CARRY_OUT_M out from the
    side plate (beyond the 10 mm), the first place line moves in to the standoff;
  * the strict line checks (`check_line`): no contact a line's first posture has is tolerated (part I's
    `line_check` tolerates them; that let a self-colliding pre-grasp posture through in a first version).
Carrying: part I's limits (ruling M8: tcp <= 0.25 m/s, joints <= 8 rad/s^2; planned at 0.20 / 6.0).

The steps (`StepAssembler.guard`, `StepAssembler.thruster`, `StepAssembler.r23`, `program_r23`):
  guard     grasp candidates on the guard's outer edge (GUARD_X_MM along the guard, GUARD_BITE_MM in from the edge, the
            approach turned GUARD_YAW_DEG about the vertical (towards the arm first: the racks are at the edge of the
            reach), the hand either way up); for each candidate and each IK branch at the grasp: the pre-grasp line
            (PICK_OUT_M out along the approach, the hand open GUARD_PRE_OPEN_M over the plate), the lift (LIFT_GUARD_M,
            then RISE_M straight up; the single-shelf rack has nothing over the guard, so no draw-off: DRAW_M 0), the
            place lines at the vehicle (`place_chain`) and the retreat (RETREAT_M back along the approach, then
            RETREAT_UP_M up) checked, the carry planned; the first candidate with every piece clear is run: transit
            (empty hand, part I's `plan_path`), the pre-grasp line in, close (GRIP_N, GRIP_SETTLE_S), lift, rise, the
            carry re-planned from where the arm is, settle, the in-hand pose re-read, the place lines, hold HOLD_S,
            open, retreat
  thruster  each vertical-thruster slot takes a unit of its design part type (of a type's two pairings the one that
            turns the units less); grasp candidates about the duct axis (the closing axis at THR_PHI_DEG from the
            vehicle's x at the slot, either finger first), the tool down (part I's IK family); pre-grasp THR_PRE_M over,
            down, close, lift LIFT_THR_M, carry to RISE_THR_M over the standoff point (CARRY_OUT_M out), in to the
            standoff (>= approach_gap), down, in along the insertion to HOVER_M short of home, hold HOLD_S, open
            (THR_OPEN_M over the duct), THR_RETREAT_M straight up
  orders    "guards_first" (default): guard_P, guard_S, vt_FL, vt_RL, vt_FR, vt_RR; "thrusters_first": vt_FL, vt_RL,
            vt_FR, vt_RR, guard_P, guard_S (the guard then goes down over the two ducts STANDOFF_GUARD_TF_M off the
            plate: the cut-outs leave 5.4 mm radially)
  then      the ready posture (part I's `go_ready`).
Every record (`StepAssembler.records[slot]`): the grasp chosen, the candidates tried and why they failed, the paths
(how each was found), the carry's clearances (planned / executed; None = nothing within the 30 mm of the check), the
in-hand shift on the lift, the times (t0, arrive, release, t1), the unit against home at the arrival, before and
after the release and after the retreat (`slot_error`: geometry only).
"""
import math
import time

import numpy as np

from .coupling import _inv, rotation_angle
from . import reference as R1
from .reference import (_T, Assembler, CARRY_JOINT_ACC_PLAN, CARRY_TCP_PLAN_M_S, EMPTY_JOINT_ACC, EMPTY_TCP_M_S,
                        LINE_ACC, SWEEP_STEP_RAD)

# ---- the rulings
HOLD_S = 1.5                     # L19: held still on the place this long before the fingers open
GUARD_TILT_MAX_DEG = 0.5         # L19: the guard's placement tilt
THR_TILT_MAX_DEG = 1.0           # the thruster's placement tilt (G6 g4win: 1.45 deg missed the 1.4 deg window)
CARRY_CLEAR_M = 0.010            # the carry: arm, gripper, fingers and the held unit this far off everything else
CARRY_PLAN_MARGIN_M = 0.0015     # planned this much wider: the arm's tracking on the executed carry (a pilot run planned
                                 # at exactly 10 mm measured 10.06 mm on the re-sampled plan and 9.86 mm executed)
# ---- the grips
GRIP_N = 100.0                   # G6 (both units)
GRIP_SETTLE_S = 1.0
GUARD_PRE_OPEN_M = 0.022         # the pre-grasp opening over the plate's thickness (G6)
GUARD_OPEN_M = 0.022             # the release opening over the plate's thickness
THR_PRE_OPEN_M = 0.016           # over the duct's width (G6)
THR_OPEN_M = 0.016
# ---- the guard's grasp family (vehicle frame at home, mm / deg)
GUARD_X_MM = (12.7, -47.3, 72.7, -107.3, 132.7, -156.8, 157.7)   # P06: the guard's solid cross-sections
GUARD_BITE_MM = (30.0, 40.0, 20.0)
GUARD_YAW_DEG = (30.0, 15.0, 0.0, -15.0, -30.0)   # x the side's sign: + = the approach turned to come from the arm's side
# ---- the guard's motions
PICK_OUT_M = 0.08                # the pre-grasp: this far out along the approach (P06; 0.12 m is out of reach)
LIFT_GUARD_M = 0.003             # off the shelf
DRAW_M = 0.0                     # outboard after the lift (0: none; the single-shelf rack has nothing over the guard)
RISE_M = 0.10                    # straight up
RISE_PLACE_M = 0.06              # the pre-place pose over the standoff point
STANDOFF_GUARD_M = 0.008         # guards first: lowered this far off the side plate (G6 7)
STANDOFF_GUARD_TF_M = 0.0025     # thrusters first: over the two ducts (5.4 mm radial in the cut-outs, G6 bt)
CAPTURE_MARGIN_M = 0.0006        # the last line starts this far outside the slot's capture depth (the cell manual's
                                 # window table: kit slot capture.depth_mm): only the last line is inside the window,
                                 # so the unit is captured on it and locks while the hand holds it still (a capture on an
                                 # earlier line let the weld pull it home ahead of the hand: locked before the hold)
RETREAT_M = 0.10                 # back along the approach after the release
RETREAT_UP_M = 0.08
# ---- the thruster's grasp family and motions
THR_PHI_DEG = (0.0, 15.0, -15.0, 30.0, -30.0)   # the closing axis from the vehicle's x at the slot (<= 30 deg: P06)
THR_TIP_OVER_GUARD_M = 0.0072    # the fingertips' lowest point over the guard's top face (P06: B z -3.5)
THR_PRE_M = 0.06                 # the pre-grasp over the grasp
LIFT_THR_M = 0.10                # out of the nest (the nose cone through the nest's hole)
RISE_THR_M = 0.10                # the pre-place pose over the standoff point (the nose cone over the guard)
STANDOFF_THR_M = None            # lowered this far off the side plate: None = the capture depth + CAPTURE_MARGIN_M
                                 # (3.6 mm; G6 b used 2.5 mm, inside the 3 mm depth window; the duct keeps 1.8 mm in the
                                 # guard's cut-out)
CARRY_OUT_M = 0.016              # every carry ends this far out from the side plate (beyond CARRY_CLEAR_M), the
                                 # first place line moves in to the standoff at the pre-place height
THR_RETREAT_M = 0.08
# ---- speeds and the hover
V_DOWN_HELD = 0.03               # the held unit's straight lines near the vehicle / station
V_IN = 0.01                      # the last millimetres onto the place
V_EMPTY_LINE = 0.05
HOVER_M = 0.0002                 # the hold this far short of home along the insertion
SETTLE_TOL_RAD = 0.0002          # the carry's end: the arm within this of the command before the in-hand re-read
SETTLE_S = 0.2
HOVER_OUT_M = 0.0002             # the guard's hold this far outboard of home (its inner edge off the side plate)
ORDERS = {"guards_first": ("guard_P", "guard_S", "vt_FL", "vt_RL", "vt_FR", "vt_RR"),
          "thrusters_first": ("vt_FL", "vt_RL", "vt_FR", "vt_RR", "guard_P", "guard_S")}
GUARD_RACK_OF = {"guard_P": "guard_rack_port", "guard_S": "guard_rack_stbd"}   # each guard lies on its side's rack


def _shift(T, dp):
    T2 = np.array(T, float).copy()
    T2[:3, 3] = T2[:3, 3] + np.asarray(dp, float)
    return T2


def _rotz(deg):
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _tilt_deg(A, B):
    """the angle (deg) between the z axes (the insertion) of two frames"""
    return math.degrees(math.acos(max(-1.0, min(1.0, float(A[:3, 2] @ B[:3, 2])))))


# ====================================================================================================== the assembler
class StepAssembler(Assembler):
    """part I's Assembler + the guard and thruster steps (module doc)"""

    def __init__(self, cell, log=print, read_fixture=None, monitor=True, kin=None):
        super(StepAssembler, self).__init__(cell, log=log, read_fixture=read_fixture, monitor=monitor, kin=kin)
        P = R1._planner_class()
        # the carry: everything CARRY_CLEAR_M + CARRY_PLAN_MARGIN_M (the held unit's threshold is its clearance - 0.5 mm
        # in `hits`)
        cc = CARRY_CLEAR_M + CARRY_PLAN_MARGIN_M
        self.pl_carry = P(cell, clearance_m=cc, held_clearance_m=cc + 0.0005, finger_clear_m=cc)
        # the placement lines: the held unit >= 1 mm off everything but what it is allowed to touch
        self.pl_fine = P(cell, clearance_m=R1.LINK_CLEAR_M, held_clearance_m=0.0015, finger_clear_m=R1.FINGER_CLEAR_M)
        self.pl_verify = None
        self.ikmode = "down"
        self.records = {}

    # ------------------------------------------------------------------ IK modes
    def ik(self, T, q_ref, margin=0.02):
        """mode 'down': part I's family (tool down); mode 'free': the analytic solution nearest q_ref (its aliases
        inside the joint ranges), away from the singularities (URKin.margin >= 0.02)"""
        if self.ikmode == "down":
            return super(StepAssembler, self).ik(T, q_ref, margin)
        q_ref = np.asarray(q_ref, float)
        best = None
        for q in self.kin.ik(T, q6_ref=float(q_ref[5])):
            qn = self.kin.nearest(q, q_ref, margin)
            if qn is None or self.kin.margin(qn) < 0.02:
                continue
            dist = float(np.max(np.abs(qn - q_ref)))
            if best is None or dist < best[0]:
                best = (dist, qn)
        return None if best is None else best[1]

    def branches(self, T, q_ref, margin=0.02):
        """every analytic solution at T (aliases nearest q_ref, inside the ranges, away from singularities)"""
        out = []
        q_ref = np.asarray(q_ref, float)
        for q in self.kin.ik(T, q6_ref=float(q_ref[5])):
            qn = self.kin.nearest(q, q_ref, margin)
            if qn is None or self.kin.margin(qn) < 0.02:
                continue
            out.append(qn)
        out.sort(key=lambda q: float(np.max(np.abs(q - q_ref))))
        return out

    def track(self, Ts, q0):
        """postures along a chain of tcp poses, each by straight lines of 2 mm / 1 deg steps tracked from q0 (mode
        'free'); [q at each pose] or None"""
        mode, self.ikmode = self.ikmode, "free"
        try:
            out = []
            q = np.asarray(q0, float)
            T0 = None
            for T in Ts:
                if T0 is None:                                  # q0 is the posture at the first pose
                    out.append(q)
                    T0 = T
                    continue
                dist = float(np.linalg.norm(T[:3, 3] - T0[:3, 3]))
                ang = rotation_angle(T0[:3, :3].T @ T[:3, :3])
                n = max(1, int(math.ceil(max(dist / 0.002, math.degrees(ang)))))
                qs = self.line_qs(T0, T, q, list(np.linspace(0.0, 1.0, n + 1)[1:]))
                if qs is None:
                    return None
                out.append(qs[-1])
                q, T0 = qs[-1], T
            return out
        finally:
            self.ikmode = mode

    # ------------------------------------------------------------------ checks on a planner
    def check_line(self, pl, qs, held=None, allow=(), width=None, held_allow=()):
        """every posture of the line (each 2nd and the last; the postures are <= 3 mm apart) free on planner pl,
        strictly: no contact the first posture already has is tolerated (part I's `line_check` tolerates them, which
        let a self-colliding pre-grasp posture through)"""
        pl.sync(held=held, allow=allow, width=width, held_allow=held_allow)
        return all(not pl.hits(q) for q in list(qs[::2]) + [qs[-1]])

    def posture_ok(self, pl, q, held=None, allow=(), width=None, held_allow=()):
        pl.sync(held=held, allow=allow, width=width, held_allow=held_allow)
        return pl.ok(q), pl.hits(q)

    def plan_carry(self, q0, q1, held, width, vias=()):
        """a carry [q0, ..., q1] with the arm, gripper, fingers and the held unit CARRY_CLEAR_M off everything else:
        straight; else through each via in turn; else RRT-Connect (`motion.Planner.plan`, shortcut). (path, how) or
        (None, why)"""
        pl = self.pl_carry
        pl.sync(held=held, width=width)
        q0, q1 = np.asarray(q0, float), np.asarray(q1, float)
        if not pl.ok(q0):
            return None, "start within %.1f mm: %s" % ((CARRY_CLEAR_M + CARRY_PLAN_MARGIN_M) * 1e3, sorted(set((h[0], h[1]) for h in pl.hits(q0)))[:4])
        if not pl.ok(q1):
            return None, "goal within %.1f mm: %s" % ((CARRY_CLEAR_M + CARRY_PLAN_MARGIN_M) * 1e3, sorted(set((h[0], h[1]) for h in pl.hits(q1)))[:4])
        if pl.free(q0, q1, step=SWEEP_STEP_RAD):
            return [q0, q1], "straight"
        for name, vs in vias:
            if not vs or any(v is None for v in vs):
                continue
            path = [q0] + [np.asarray(v, float) for v in vs] + [q1]
            if all(pl.free(a, b, step=SWEEP_STEP_RAD) for a, b in zip(path[:-1], path[1:])):
                return path, name
        path = pl.plan(q0, q1)
        if path is not None:
            return path, "rrt"
        return None, "no carry path (%s)" % getattr(pl, "why", "")

    def carry_clearance(self, qs, held, width, step=0.001):
        """(the smallest distance m, the pair) of the arm / gripper / held unit to everything else over the postures
        (each segment re-sampled at `step` rad), measured with 30 mm margins on a planner copy of its own"""
        if self.pl_verify is None:
            self.pl_verify = R1._planner_class()(self.cell, clearance_m=0.03, held_clearance_m=0.03, finger_clear_m=0.03)
        pv = self.pl_verify
        pv.sync(held=held, width=width)
        samples = []
        for a, b in zip(qs[:-1], qs[1:]):
            a, b = np.asarray(a, float), np.asarray(b, float)
            n = max(1, int(math.ceil(float(np.max(np.abs(b - a))) / step)))
            samples += [a + (b - a) * (k / float(n)) for k in range(n)]
        samples.append(np.asarray(qs[-1], float))
        best, pair = pv.min_dist(samples)
        return best, pair, len(samples)

    def slot_error(self, sid, u):
        """the unit's pose against its home on the slot (geometry only, from the live target; the coupling is not
        read): position error (mm, world), tilt (deg: the slot's insertion axis against the unit's same axis), full
        rotation (deg), along the insertion (mm, < 0 short of home), lateral (mm), turn about the insertion (deg)"""
        tg = self.cell.live_target(sid, u)
        Th, Tb = tg["T"], self.unit_T(u)
        ins = tg["insertion"]
        e = Th[:3, :3].T @ ins                              # the insertion axis in the unit's frame
        a_now = Tb[:3, :3] @ e
        tilt = math.degrees(math.acos(max(-1.0, min(1.0, float(a_now @ ins)))))
        rel = Th[:3, :3].T @ Tb[:3, :3]
        dv = Tb[:3, 3] - Th[:3, 3]
        ax = float(dv @ ins)
        # the turn about the insertion: the rotation's component about e
        from scipy.spatial.transform import Rotation
        rv = Rotation.from_matrix(rel).as_rotvec()
        return {"pos_err_mm": [round(1e3 * float(v), 4) for v in dv], "tilt_deg": round(tilt, 4),
                "rot_deg": round(math.degrees(rotation_angle(rel)), 4), "axial_mm": round(1e3 * ax, 4),
                "lateral_mm": round(1e3 * float(np.linalg.norm(dv - ax * ins)), 4),
                "clock_deg": round(math.degrees(float(rv @ e)), 4)}

    # ------------------------------------------------------------------ geometry of the units (kit collision points)
    def _points_world(self, u, T):
        from .bench_plan import collision_points
        P, _S = collision_points(self.kit, u)
        return P @ T[:3, :3].T + T[:3, 3]

    def guard_grasps(self, sid, u):
        """[(G = the tcp in the guard's body frame, opening m, meta)] of the guard family (module doc), at home on
        slot sid; the vehicle frame (deck up) gives the directions: sg = +1 port / -1 starboard"""
        from . import gripper_wide as GW
        tg = self.cell.live_target(sid, u)
        Th = tg["T"]
        Tv = self.cell.vehicle_T()
        Pb = (self._points_world(u, Th) - Tv[:3, 3]) @ Tv[:3, :3]          # the guard at home in the vehicle frame (m)
        sg = 1.0 if float(np.mean(Pb[:, 1])) > 0 else -1.0
        out = []
        for x_mm in GUARD_X_MM:
            x = x_mm / 1000.0
            band = Pb[np.abs(Pb[:, 0] - x) <= 0.0135 + 0.002]
            if not len(band):
                continue
            edge = float(np.max(sg * band[:, 1]))
            for bite_mm in GUARD_BITE_MM:
                b = bite_mm / 1000.0
                sel = band[(sg * band[:, 1] >= edge - b - 0.002)]
                z0, z1 = float(sel[:, 2].min()), float(sel[:, 2].max())
                w = z1 - z0
                zm = 0.5 * (z0 + z1)
                for yaw in GUARD_YAW_DEG:
                    yaw = sg * yaw
                    for flip in (False, True):
                        Rz = _rotz(yaw)
                        zt = Rz @ np.array([0.0, -sg, 0.0])                 # the approach: inboard
                        yt = np.array([0.0, 0.0, -1.0 if flip else 1.0])    # the closing axis: vertical
                        xt = np.cross(yt, zt)
                        Rt = np.column_stack([xt, yt, zt])
                        # the fingertips' end `bite` in from the edge, the pads' middle across at x
                        pc = np.array([x, sg * (edge - 0.5 * b), zm])           # the bite's middle
                        tip = pc + 0.5 * b * zt
                        po = GW.pad_offset(w)
                        tcp = tip - GW.fingertip_z(w) * zt - Rt @ np.array([po[0], po[1], 0.0])
                        Tt_B = _T(tcp, Rt)
                        Tt_w = Tv @ Tt_B
                        G = _inv(Th) @ Tt_w
                        out.append((G, w, {"x_mm": x_mm, "bite_mm": bite_mm, "yaw_deg": yaw, "flip": flip,
                                           "edge_abs_y_mm": round(1e3 * edge, 2), "width_mm": round(1e3 * w, 3)}))
        # preference: the approach from the arm's side first (reach), then the middle cross-sections, the 30 mm bite
        pref = {y: i for i, y in enumerate(GUARD_YAW_DEG)}
        out.sort(key=lambda o: (pref[sg * o[2]["yaw_deg"]], GUARD_X_MM.index(o[2]["x_mm"]), GUARD_BITE_MM.index(o[2]["bite_mm"])))
        return out, sg

    def thruster_grasps(self, sid, u):
        """[(G = the tcp in the thruster's body frame, opening m, meta)]: the tool down along the duct axis (the body's
        -z), the closing axis at THR_PHI_DEG from the vehicle's x at the slot, the fingertips' lowest point
        THR_TIP_OVER_GUARD_M over the guard's top face, the grasp centre on the duct axis"""
        from . import gripper_wide as GW
        tg = self.cell.live_target(sid, u)
        Th = tg["T"]
        Tv = self.cell.vehicle_T()
        P = self._points_world(u, Th)
        Pb = (P - Tv[:3, 3]) @ Tv[:3, :3]
        axis_b = (_inv(Tv) @ Th)[:3, 3]                                     # the body origin: on the duct axis
        gtop = self.guard_top_B()
        z_tip = gtop + THR_TIP_OVER_GUARD_M
        out = []
        for phi in THR_PHI_DEG:
            for flip in (False, True):
                yt = _rotz(phi) @ np.array([1.0, 0.0, 0.0])
                if flip:
                    yt = -yt
                zt = np.array([0.0, 0.0, -1.0])
                xt = np.cross(yt, zt)
                Rt = np.column_stack([xt, yt, zt])
                rel = Pb - axis_b
                band = rel[(Pb[:, 2] >= z_tip) & (np.abs(rel @ xt) <= 0.0135 + 0.002)]
                if not len(band):
                    continue
                s = band @ yt
                w = float(s.max() - s.min())
                c_off = 0.5 * float(s.max() + s.min())
                po = GW.pad_offset(w)
                tip = axis_b + c_off * yt
                tip[2] = z_tip
                tcp = tip - GW.fingertip_z(w) * zt - Rt @ np.array([po[0], po[1], 0.0])
                G = _inv(Th) @ (Tv @ _T(tcp, Rt))
                out.append((G, w, {"phi_deg": phi, "flip": flip, "width_mm": round(1e3 * w, 3),
                                   "tip_z_B_mm": round(1e3 * z_tip, 3)}))
        return out

    def guard_top_B(self):
        """the guards' top face height in the vehicle frame (m): from the kit's guard collision at home on guard_P"""
        if not hasattr(self, "_gtop"):
            Th = self.cell.live_target("guard_P", "GRD1")["T"]
            Tv = self.cell.vehicle_T()
            Pb = (self._points_world("GRD1", Th) - Tv[:3, 3]) @ Tv[:3, :3]
            self._gtop = float(Pb[:, 2].max())
        return self._gtop

    # ------------------------------------------------------------------ the guard
    def guard_unit(self, sid):
        """the guard lying on slot sid's side's rack (where the units lie: the layout)"""
        us = [u for u, g in sorted(self.cell.layout["guards"].items()) if g["rack"] == GUARD_RACK_OF[sid]]
        if len(us) != 1:
            raise RuntimeError("no single guard on %s: %s" % (GUARD_RACK_OF[sid], us))
        return us[0]

    def guard(self, sid, u=None, order="guards_first"):
        u = u or self.guard_unit(sid)
        rec = {"slot": sid, "unit": u, "kind": "guard", "placed_by": "nobody", "t0_s": round(self.t(), 3)}
        self.records[sid] = rec
        self._blocks[sid] = rec
        self._uf_hist = []
        self.work.update({"site": sid, "unit": u})
        try:
            self._guard(sid, u, rec, order)
            rec["placed_by"] = "ur"
        except RuntimeError as e:
            rec["why"] = str(e)
            self.log("  %s <- %s: %s" % (sid, u, e))
            if self.work.get("held"):
                self.let_go()
        self.set_phase("idle", held=False, unit=None, site=None)
        rec["t1_s"] = round(self.t(), 3)
        rec["unit_s"] = round(rec["t1_s"] - rec["t0_s"], 3)
        self.log("  %-8s <- %s %s  %.1f s sim" % (sid, u, rec["placed_by"], rec["unit_s"]))
        return rec

    def plan_guard(self, sid, u, order="guards_first", rec=None):
        """the guard's plan (module doc): the first candidate (grasp, IK branch at the grasp, branch at the place) with
        every line clear and a carry; the plan dict (raises RuntimeError with the failure counts when there is none).
        Plans only: nothing moves (the harness's planner-only sweeps call it on hypothetical racks)"""
        rec = {} if rec is None else rec
        c = self.cell
        t_plan = time.process_time()
        T0 = self.unit_T(u)
        tg = c.live_target(sid, u)
        Th = tg["T"]
        ins = tg["insertion"]                                   # down
        cands, sg = self.guard_grasps(sid, u)
        Tv = c.vehicle_T()
        out_B = Tv[:3, :3] @ np.array([0.0, sg, 0.0])          # outboard at the vehicle
        rack = c.layout["guards"][u]["rack"]
        Trk = c.body_T(rack)
        out_rack = Trk[:3, :3] @ np.array([0.0, sg, 0.0])      # outboard at the rack
        standoff = STANDOFF_GUARD_M if order == "guards_first" else STANDOFF_GUARD_TF_M
        q_now = self.ct.q_cmd.copy()
        tried = []
        chosen = None
        for G, w, meta in cands:
            Tg = T0 @ G
            zt = Tg[:3, 2]
            T_pre = _shift(Tg, -PICK_OUT_M * zt)
            T_lift = _shift(Tg, [0.0, 0.0, LIFT_GUARD_M])
            T_draw = _shift(T_lift, DRAW_M * out_rack)
            T_rise = _shift(T_draw, [0.0, 0.0, RISE_M])
            Gi = _inv(G)                                        # the guard in the tcp frame
            # at the vehicle: home with the hover, standoff out and up
            Tp_home = Th @ G
            Tp_hov = self.hover_of(sid, Tp_home, ins, out_B)
            Tp_pre, chain = self.place_chain(True, Tp_hov, ins, out_B, standoff, self.approach_gap(sid))
            Tp_top, Tp_off, Tp_in = chain[0][1], chain[1][1], chain[2][1]
            zt_v = Tp_hov[:3, 2]
            Tr_out = _shift(Tp_hov, -RETREAT_M * zt_v)
            Tr_up = _shift(Tr_out, [0.0, 0.0, RETREAT_UP_M])
            for qg in self.branches(Tg, q_now):
                key = (meta["x_mm"], meta["bite_mm"], meta["yaw_deg"], meta["flip"])
                pick = self.track([Tg, T_pre], qg)
                if pick is None:
                    tried.append((key, "pick IK"))
                    continue
                q_pre = pick[1]
                lift = self.track([Tg, T_lift, T_draw, T_rise], qg)
                if lift is None:
                    tried.append((key, "lift IK"))
                    continue
                # the pre-grasp line (open hand, the guard allowed)
                self.ikmode = "free"
                try:
                    down, _s = self.line_plan(Tg, V_EMPTY_LINE, T0=T_pre, q0=q_pre)
                finally:
                    self.ikmode = "down"
                if down is None or not self.check_line(self.pl, [q_pre] + down, allow=(u,), width=w + GUARD_PRE_OPEN_M):
                    tried.append((key, "pick line"))
                    continue
                # the lift / draw-off / rise with the guard (the rack allowed while it is lifted off)
                Hg = (u, Gi)
                ok1 = self.check_line(self.pl, self._line_postures([Tg, T_lift, T_draw, T_rise], qg), held=Hg, width=w,
                                      held_allow=(rack,))
                if not ok1:
                    tried.append((key, "draw-off line"))
                    continue
                ok2, hits = self.posture_ok(self.pl_carry, lift[3], held=Hg, width=w)
                if not ok2:
                    tried.append((key, "rise pose within the carry clearance: %s" % sorted(set(h[1] for h in hits))[:3]))
                    continue
                # the place: a branch at the hover, the lines up to the pre-place and out to the retreat
                placed = None
                for qh in self.branches(Tp_hov, lift[3]):
                    up = self.track([Tp_hov, Tp_in, Tp_off, Tp_top, Tp_pre], qh)
                    if up is None:
                        continue
                    back = self.track([Tp_hov, Tr_out, Tr_up], qh)
                    if back is None:
                        continue
                    ok3, hits3 = self.posture_ok(self.pl_carry, up[4], held=Hg, width=w)
                    if not ok3:
                        continue
                    qs_down = self._line_postures([Tp_pre, Tp_top, Tp_off], up[4])
                    qs_last = self._line_postures([Tp_off, Tp_in, Tp_hov], up[2])
                    if qs_down is None or qs_last is None:
                        continue
                    holders = self.unit_holders(sid)
                    if not self.check_line(self.pl_fine, qs_down, held=Hg, width=w):
                        continue
                    if not self.check_line(self.pl_fine, qs_last, held=Hg, width=w, held_allow=(c.base,) + holders):
                        continue
                    qs_back = self._line_postures([Tp_hov, Tr_out, Tr_up], qh)
                    if qs_back is None or not self.check_line(self.pl, qs_back, width=w + GUARD_OPEN_M):
                        continue
                    placed = (qh, up, back)
                    break
                if placed is None:
                    tried.append((key, "place lines"))
                    continue
                qh, up, back = placed
                # the carry: rise pose -> pre-place, everything CARRY_CLEAR_M
                vias = self._carry_vias(lift[3], up[4], T_rise, Tp_pre)
                path, how = self.plan_carry(lift[3], up[4], Hg, w, vias)
                if path is None:
                    tried.append((key, "carry: %s" % how))
                    continue
                chosen = dict(G=G, w=w, meta=meta, qg=qg, q_pre=q_pre, down=down, lift=lift, Tg=Tg, T_pre=T_pre,
                              T_lift=T_lift, T_draw=T_draw, T_rise=T_rise, Tp_pre=Tp_pre, Tp_off=Tp_off, Tp_in=Tp_in, Tp_top=Tp_top,
                              Tp_hov=Tp_hov, Tr_out=Tr_out, Tr_up=Tr_up, qh=qh, up=up, back=back, carry=path,
                              carry_how=how)
                break
            if chosen is not None:
                break
        rec["candidates_tried"] = len(tried)
        rec["plan_cpu_s"] = round(time.process_time() - t_plan, 2)
        if chosen is None:
            import collections
            rec["tried_why"] = dict(collections.Counter(t[1].split(":")[0] for t in tried))
            raise RuntimeError("no guard plan for %s (%d tried: %s)" % (sid, len(tried), rec["tried_why"]))
        X = chosen
        X.update({"standoff": standoff, "out_B": out_B})
        rec.update({"grasp": X["meta"], "width_mm": round(1e3 * X["w"], 3), "carry_how": X["carry_how"],
                    "carry_waypoints": len(X["carry"]), "standoff_mm": round(1e3 * standoff, 2)})
        cl, pair, ns = self.carry_clearance(X["carry"], (u, _inv(X["G"])), X["w"])
        rec["carry_clearance_plan_mm"] = round(1e3 * cl, 2) if pair is not None else None
        return X

    def _guard(self, sid, u, rec, order):
        X = self.plan_guard(sid, u, order, rec)
        w, standoff, out_B = X["w"], X["standoff"], X["out_B"]
        # ---- transit to the pre-grasp (the fingers set first)
        self.set_phase("to-pick", held=False, unit=u)
        width_chk = self.set_opening(w + GUARD_PRE_OPEN_M)
        q_now = self.ct.q_cmd.copy()
        vias = [("up", [self.high_of_free(self.tcp_cmd_T(), q_now)]),
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
        self.grip(0.0, force_N=GRIP_N, settle_s=GRIP_SETTLE_S)
        G0 = _inv(self.tcp_cmd_T()) @ self.unit_T(u)
        # ---- lift, draw off, rise
        self.set_phase("lift", held=True)
        self._run_to(X["T_lift"], 0.01)
        self.set_phase("lift", sub="draw")
        self._run_to(X["T_draw"], 0.05)
        self.set_phase("lift", sub="rise")
        self._run_to(X["T_rise"], 0.06)
        G1 = _inv(self.tcp_cmd_T()) @ self.unit_T(u)
        rec["in_hand_shift_lift_mm"] = round(1e3 * float(np.linalg.norm(G1[:3, 3] - G0[:3, 3])), 4)
        rec["in_hand_turn_lift_deg"] = round(math.degrees(rotation_angle(G0[:3, :3].T @ G1[:3, :3])), 4)
        rec["pad_width_mm"] = round(1e3 * float(self.ct.width()), 3)
        if self.ct.width() < 0.5 * w:
            raise RuntimeError("%s not in the hand after the lift" % u)
        # ---- the carry (re-planned from where the arm is, the guard as it is held now)
        self.set_phase("carry")
        Hg = (u, _inv(self.tcp_cmd_T()) @ self.unit_T(u))
        q_now = self.ct.q_cmd.copy()
        path, how = self.plan_carry(q_now, X["up"][4], Hg, w, self._carry_vias(q_now, X["up"][4], self.tcp_cmd_T(), X["Tp_pre"]))
        if path is None:
            raise RuntimeError("no carry for %s: %s" % (u, how))
        rec["carry_how"] = how
        rec["carry_waypoints"] = len(path)
        q_log = []
        self._qlog = q_log
        self.follow_smooth(path, CARRY_JOINT_ACC_PLAN, CARRY_TCP_PLAN_M_S)
        self._qlog = None
        cl, pair, ns = self.carry_clearance(path, Hg, w)
        rec["carry_clearance_planned_mm"] = round(1e3 * cl, 2) if pair is not None else None
        rec["carry_clearance_planned_pair"] = pair
        rec["carry_clearance_samples"] = ns
        if q_log:
            cle, paire, nse = self.carry_clearance(q_log, Hg, w, step=1.0)
            rec["carry_clearance_executed_mm"] = round(1e3 * cle, 2) if paire is not None else None
            rec["carry_clearance_executed_pair"] = paire
            rec["carry_clearance_executed_ticks"] = nse
        # ---- the place: the in-hand pose re-read, the lines
        self._place_lines(sid, u, rec, X["Tp_pre"], standoff, out_B, w, guard=True)
        # ---- release, retreat
        self.set_phase("release", held=False)
        self.ct.command_width(w + GUARD_OPEN_M)
        self.wait(R1.RELEASE_WAIT_S + 0.1)
        rec["at_release_open"] = self.slot_error(sid, u)
        self.set_phase("retreat")
        T = self.tcp_cmd_T()
        self._run_to(_shift(T, -RETREAT_M * T[:3, 2]), V_EMPTY_LINE, check=True, width=float(self.ct.width()))
        self._run_to(_shift(self.tcp_cmd_T(), [0.0, 0.0, RETREAT_UP_M]), V_EMPTY_LINE, check=True, width=float(self.ct.width()))
        rec["after_retreat"] = self.slot_error(sid, u)

    def approach_gap(self, sid):
        """the last line's length: the slot's capture depth (the published window) + CAPTURE_MARGIN_M (m)"""
        return float(self.cell.slots[sid]["capture"]["depth_mm"]) / 1000.0 + CAPTURE_MARGIN_M

    def hover_of(self, sid, T_home, ins, out_dir=None):
        """the hold pose of the tcp from its pose with the unit at home: HOVER_M short along the insertion (and the
        guard HOVER_OUT_M outboard)"""
        T = _shift(T_home, -HOVER_M * ins)
        if out_dir is not None:
            T = _shift(T, HOVER_OUT_M * np.asarray(out_dir, float))
        return T

    @staticmethod
    def place_chain(guard, T_hov, ins, out_dir, standoff, gap):
        """(the carry's end pose, [(sub-phase, tcp pose, speed)]) of the place lines from the hold pose T_hov:
        guard (ins down, out_dir outboard): CARRY_OUT_M out at RISE_PLACE_M over the standoff point -> in to the
        standoff -> down to `gap` over home -> in onto the plate -> the last `gap` down (the only line inside the
        capture window);
        thruster (ins towards the plate): CARRY_OUT_M out at RISE_THR_M over the standoff point -> in to the standoff
        -> down -> in along the insertion onto the plate (standoff >= gap: the last line alone in the window)"""
        if guard:
            T_in = _shift(T_hov, -gap * ins)
            T_off = _shift(T_in, standoff * out_dir)
            T_top = _shift(T_off, [0.0, 0.0, RISE_PLACE_M])
            T_pre = _shift(T_top, (CARRY_OUT_M - standoff) * out_dir)
            return T_pre, [("in-top", T_top, V_DOWN_HELD), ("down", T_off, V_DOWN_HELD), ("in", T_in, V_IN),
                           ("last", T_hov, V_IN)]
        standoff = gap if standoff is None else max(standoff, gap)
        T_off = _shift(T_hov, -standoff * ins)
        T_top = _shift(T_off, [0.0, 0.0, RISE_THR_M])
        T_pre = _shift(T_top, -(CARRY_OUT_M - standoff) * ins)
        return T_pre, [("in-top", T_top, V_DOWN_HELD), ("down", T_off, V_DOWN_HELD), ("last", T_hov, V_IN)]

    def _place_lines(self, sid, u, rec, Tp_pre_plan, standoff, out_dir, w, guard):
        """from the carry's end: the target re-computed from the in-hand pose as it is now (the commanded tcp), the
        lines down / in / last, the hold; fills rec['placement'] and the timing"""
        c = self.cell
        # the arm settled on the carry's end (follow_smooth stops within 2 mrad of the command: a transient tracking
        # offset of the order of a millimetre at the hand), then the unit's pose in the commanded tool frame re-read
        self.wait_arrived(tol=SETTLE_TOL_RAD, timeout_s=1.0)
        self.wait(SETTLE_S)
        G2 = _inv(self.tcp_cmd_T()) @ self.unit_T(u)
        tg = c.live_target(sid, u)
        ins = tg["insertion"]
        T_hov = self.hover_of(sid, tg["T"] @ _inv(G2), ins, out_dir if guard else None)
        # the predicted placement: the unit at T_hov @ G2 against home (tilt = the insertion axes)
        pred = T_hov @ G2
        lim = GUARD_TILT_MAX_DEG if guard else THR_TILT_MAX_DEG
        rec["placement_predicted"] = {"tilt_deg": round(_tilt_deg(pred, tg["T"]), 5),
                                      "rot_deg": round(math.degrees(rotation_angle(tg["T"][:3, :3].T @ pred[:3, :3])), 5)}
        T_pre, chain = self.place_chain(guard, T_hov, ins, out_dir, standoff, self.approach_gap(sid))
        d_pre = float(np.linalg.norm(T_pre[:3, 3] - self.tcp_cmd_T()[:3, 3]))
        rec["carry_end_off_mm"] = round(1e3 * d_pre, 3)
        if d_pre > 0.02:
            raise RuntimeError("the carry ended %.1f mm from the pre-place pose" % (1e3 * d_pre))
        for k, (sub, T, v) in enumerate(chain):
            self.set_phase("approach", held=True, sub=sub)
            self._run_to(T, v)
        rec["t_arrive_s"] = round(self.t(), 3)
        rec["at_arrive"] = self.slot_error(sid, u)
        self.set_phase("hold")
        self.wait(HOLD_S)
        rec["t_release_s"] = round(self.t(), 3)
        rec["at_release"] = self.slot_error(sid, u)
        rec["placement"] = {"tilt_deg": rec["at_arrive"]["tilt_deg"], "limit_deg": lim}

    def _run_to(self, T1, v, check=False, width=None):
        """a straight tcp line from the commanded tcp to T1 (joint mode, one posture per tick)"""
        mode, self.ikmode = self.ikmode, "free"
        try:
            qs, _s = self.line_plan(T1, v)
        finally:
            self.ikmode = mode
        if qs is None:
            raise RuntimeError("no line (IK) in phase %s" % self.work.get("phase"))
        if check and not self.check_line(self.pl, [self.ct.q_cmd.copy()] + qs, width=width):
            self.log("    line in phase %s not clear by the check" % self.work.get("phase"))
        self.run_line(qs)
        return qs

    def _line_postures(self, Ts, q0):
        """postures every 3 mm / 1 deg along the chain of straight lines (mode 'free') or None"""
        mode, self.ikmode = self.ikmode, "free"
        try:
            out = [np.asarray(q0, float)]
            q = out[0]
            for T0, T1 in zip(Ts[:-1], Ts[1:]):
                dist = float(np.linalg.norm(T1[:3, 3] - T0[:3, 3]))
                ang = rotation_angle(T0[:3, :3].T @ T1[:3, :3])
                n = max(1, int(math.ceil(max(dist / 0.003, math.degrees(ang)))))
                qs = self.line_qs(T0, T1, q, list(np.linspace(0.0, 1.0, n + 1)[1:]))
                if qs is None:
                    return None
                out += qs
                q = qs[-1]
            return out
        finally:
            self.ikmode = mode

    def high_of_free(self, T, q_ref):
        mode, self.ikmode = self.ikmode, "free"
        try:
            return self.high_of(T, q_ref)
        finally:
            self.ikmode = mode

    def _carry_vias(self, q0, q1, T0, T1):
        """via postures for a carry: the tcp straight over the start / the goal at their own height + 0.15 m"""
        mode, self.ikmode = self.ikmode, "free"
        try:
            A = _shift(T0, [0.0, 0.0, 0.15])
            B = _shift(T1, [0.0, 0.0, 0.15])
            qa = self.ik(A, q0)
            qb = self.ik(B, q1)
            return [("up-over", [qa, qb]), ("up", [qa]), ("over", [qb])]
        finally:
            self.ikmode = mode

    def unit_holders(self, sid):
        """the units this program has already set on the vehicle (its own records: the coupling is not read): a unit
        placed on slot sid may touch them on its last millimetres (the side's guard for a thruster, the side's
        thrusters for a guard)"""
        return tuple(r["unit"] for s2, r in sorted(self.records.items()) if s2 != sid and r.get("placed_by") == "ur")

    # ------------------------------------------------------------------ the thruster
    def thruster(self, sid, u):
        rec = {"slot": sid, "unit": u, "kind": "thruster", "placed_by": "nobody", "t0_s": round(self.t(), 3),
               "unit_type": self.kit["bodies"][u]["type"], "design_part": self.cell.slots[sid].get("design_part")}
        self.records[sid] = rec
        self._blocks[sid] = rec
        self._uf_hist = []
        self.work.update({"site": sid, "unit": u})
        try:
            self._thruster(sid, u, rec)
            rec["placed_by"] = "ur"
        except RuntimeError as e:
            rec["why"] = str(e)
            self.log("  %s <- %s: %s" % (sid, u, e))
            if self.work.get("held"):
                self.let_go()
        self.set_phase("idle", held=False, unit=None, site=None)
        rec["t1_s"] = round(self.t(), 3)
        rec["unit_s"] = round(rec["t1_s"] - rec["t0_s"], 3)
        self.log("  %-8s <- %s %s  %.1f s sim" % (sid, u, rec["placed_by"], rec["unit_s"]))
        return rec

    def _thruster(self, sid, u, rec):
        c = self.cell
        t_plan = time.process_time()
        T0 = self.unit_T(u)
        tg = c.live_target(sid, u)
        Th = tg["T"]
        ins = tg["insertion"]
        holders = self.unit_holders(sid)
        q_now = self.ct.q_cmd.copy()
        tried = []
        chosen = None
        for G, w, meta in self.thruster_grasps(sid, u):
            key = (meta["phi_deg"], meta["flip"])
            Tg = T0 @ G
            T_pre = _shift(Tg, [0.0, 0.0, THR_PRE_M])
            T_lift = _shift(Tg, [0.0, 0.0, LIFT_THR_M])
            Tp_hov = self.hover_of(sid, Th @ G, ins)
            Tp_pre, chain = self.place_chain(False, Tp_hov, ins, None, STANDOFF_THR_M, self.approach_gap(sid))
            Tp_top, Tp_off = chain[0][1], chain[1][1]
            Tr_up = _shift(Tp_hov, [0.0, 0.0, THR_RETREAT_M])
            for qg in self.branches(Tg, q_now):
                if not self.in_family(qg):
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
                if down is None or not self.check_line(self.pl, [q_pre] + down, allow=(u,), width=w + THR_PRE_OPEN_M):
                    tried.append((key, "pick line"))
                    continue
                Hg = (u, _inv(G))
                qs_l = self._line_postures([Tg, T_lift], qg)
                if qs_l is None or not self.check_line(self.pl, qs_l, held=Hg, width=w, held_allow=("thruster_tray",)):
                    tried.append((key, "lift line"))
                    continue
                ok2, hits = self.posture_ok(self.pl_carry, lift[1], held=Hg, width=w)
                if not ok2:
                    tried.append((key, "lift pose within the carry clearance: %s" % sorted(set(h[1] for h in hits))[:3]))
                    continue
                placed = None
                for qh in self.branches(Tp_hov, lift[1]):
                    if not self.in_family(qh):
                        continue
                    up = self.track([Tp_hov, Tp_off, Tp_top, Tp_pre], qh)
                    back = self.track([Tp_hov, Tr_up], qh)
                    if up is None or back is None:
                        continue
                    ok3, _h = self.posture_ok(self.pl_carry, up[3], held=Hg, width=w)
                    if not ok3:
                        continue
                    qs_down = self._line_postures([Tp_pre, Tp_top, Tp_off], up[3])
                    qs_in = self._line_postures([Tp_off, Tp_hov], up[1])
                    if qs_down is None or qs_in is None:
                        continue
                    if not self.check_line(self.pl_fine, qs_down, held=Hg, width=w):
                        continue
                    if not self.check_line(self.pl_fine, qs_in, held=Hg, width=w, held_allow=(c.base,) + holders):
                        continue
                    qs_back = self._line_postures([Tp_hov, Tr_up], qh)
                    if qs_back is None or not self.check_line(self.pl, qs_back, width=w + THR_OPEN_M, allow=(u,)):
                        continue
                    placed = (qh, up, back)
                    break
                if placed is None:
                    tried.append((key, "place lines"))
                    continue
                qh, up, back = placed
                vias = self._carry_vias(lift[1], up[3], T_lift, Tp_pre)
                path, how = self.plan_carry(lift[1], up[3], Hg, w, vias)
                if path is None:
                    tried.append((key, "carry: %s" % how))
                    continue
                chosen = dict(G=G, w=w, meta=meta, qg=qg, q_pre=q_pre, down=down, Tg=Tg, T_pre=T_pre, T_lift=T_lift,
                              Tp_pre=Tp_pre, Tp_off=Tp_off, Tp_hov=Tp_hov, Tr_up=Tr_up, up=up, carry=path, carry_how=how)
                break
            if chosen is not None:
                break
        rec["candidates_tried"] = len(tried)
        rec["plan_cpu_s"] = round(time.process_time() - t_plan, 2)
        if chosen is None:
            import collections
            rec["tried_why"] = dict(collections.Counter(t[1].split(":")[0] for t in tried))
            raise RuntimeError("no thruster plan for %s (%d tried: %s)" % (sid, len(tried), rec["tried_why"]))
        X = chosen
        w = X["w"]
        rec.update({"grasp": X["meta"], "width_mm": round(1e3 * w, 3), "turn_deg": round(math.degrees(
            rotation_angle(T0[:3, :3].T @ Th[:3, :3])), 2)})
        self.set_phase("to-pick", held=False, unit=u)
        width_chk = self.set_opening(w + THR_PRE_OPEN_M)
        q_now = self.ct.q_cmd.copy()
        vias = [("over", [self.high_of(X["T_pre"], X["q_pre"])]),
                ("up", [self.high_of_free(self.tcp_cmd_T(), q_now)]),
                ("up-over", [self.high_of_free(self.tcp_cmd_T(), q_now), self.high_of(X["T_pre"], X["q_pre"])])]
        path, how = self.plan_path(q_now, X["q_pre"], width=width_chk, vias=vias)
        if path is None:
            raise RuntimeError("no path to the pick of %s: %s" % (u, how))
        rec["path_to_pick"] = how
        self.follow_smooth(path, EMPTY_JOINT_ACC, EMPTY_TCP_M_S)
        self.wait(0.1)
        self.set_phase("approach-pick")
        self.run_line(X["down"])
        self.set_phase("grip")
        self.grip(0.0, force_N=GRIP_N, settle_s=GRIP_SETTLE_S)
        G0 = _inv(self.tcp_cmd_T()) @ self.unit_T(u)
        self.set_phase("lift", held=True)
        self._run_to(X["T_lift"], R1.V_LIFT)
        G1 = _inv(self.tcp_cmd_T()) @ self.unit_T(u)
        rec["in_hand_shift_lift_mm"] = round(1e3 * float(np.linalg.norm(G1[:3, 3] - G0[:3, 3])), 4)
        rec["in_hand_turn_lift_deg"] = round(math.degrees(rotation_angle(G0[:3, :3].T @ G1[:3, :3])), 4)
        rec["pad_width_mm"] = round(1e3 * float(self.ct.width()), 3)
        if self.ct.width() < 0.5 * w:
            raise RuntimeError("%s not in the hand after the lift" % u)
        self.set_phase("carry")
        Hg = (u, G1)
        q_now = self.ct.q_cmd.copy()
        path, how = self.plan_carry(q_now, X["up"][3], Hg, w, self._carry_vias(q_now, X["up"][3], self.tcp_cmd_T(), X["Tp_pre"]))
        if path is None:
            raise RuntimeError("no carry for %s: %s" % (u, how))
        rec["carry_how"] = how
        rec["carry_waypoints"] = len(path)
        q_log = []
        self._qlog = q_log
        self.follow_smooth(path, CARRY_JOINT_ACC_PLAN, CARRY_TCP_PLAN_M_S)
        self._qlog = None
        cl, pair, ns = self.carry_clearance(path, Hg, w)
        rec["carry_clearance_planned_mm"] = round(1e3 * cl, 2) if pair is not None else None
        rec["carry_clearance_planned_pair"] = pair
        if q_log:
            cle, paire, nse = self.carry_clearance(q_log, Hg, w, step=1.0)
            rec["carry_clearance_executed_mm"] = round(1e3 * cle, 2) if paire is not None else None
            rec["carry_clearance_executed_pair"] = paire
        self._place_lines(sid, u, rec, X["Tp_pre"], STANDOFF_THR_M, None, w, guard=False)
        self.set_phase("release", held=False)
        self.ct.command_width(w + THR_OPEN_M)
        self.wait(R1.RELEASE_WAIT_S + 0.1)
        rec["at_release_open"] = self.slot_error(sid, u)
        self.set_phase("retreat")
        self._run_to(_shift(self.tcp_cmd_T(), [0.0, 0.0, THR_RETREAT_M]), V_EMPTY_LINE, check=True,
                     width=float(self.ct.width()))
        rec["after_retreat"] = self.slot_error(sid, u)

    # ------------------------------------------------------------------ follow_smooth with a posture log
    def follow_smooth(self, path, a_joint, v_tcp, v_joint=R1.JOINT_SPEED, tol=0.002):
        ql = getattr(self, "_qlog", None)
        if ql is None:
            return super(StepAssembler, self).follow_smooth(path, a_joint, v_tcp, v_joint, tol)
        hook = lambda: ql.append(self.ct.q().copy())       # noqa: E731
        self.cell.hooks.append(hook)
        try:
            return super(StepAssembler, self).follow_smooth(path, a_joint, v_tcp, v_joint, tol)
        finally:
            self.cell.hooks.remove(hook)

    # ------------------------------------------------------------------ R2 / R3
    def assign_thrusters(self):
        """{slot: unit}: each vertical-thruster slot takes a unit of its design part type; of the two ways to pair a
        type's two units with its two slots, the one turning the units less (the hand turns with them)"""
        out = {}
        slots = [s for s in ORDERS["guards_first"] if s.startswith("vt_")]
        types = {}
        for s in slots:
            types.setdefault(self.cell.slots[s]["design_part"], []).append(s)
        for tp, ss in sorted(types.items()):
            placed = set(r["unit"] for r in self.records.values() if r.get("placed_by") == "ur")
            us = [u for u in self.cell.loose if self.kit["bodies"][u]["type"] == tp and u not in placed]
            best = None
            import itertools
            for perm in itertools.permutations(us, len(ss)):
                cost = 0.0
                for s, u in zip(ss, perm):
                    Th = self.cell.live_target(s, u)["T"]
                    cost += rotation_angle(self.unit_T(u)[:3, :3].T @ Th[:3, :3])
                if best is None or cost < best[0]:
                    best = (cost, perm)
            for s, u in zip(ss, best[1]):
                out[s] = u
        return out

    def r23(self, order="guards_first", slots=None):
        """the R2 / R3 segment: the guards and the vertical thrusters in `order` (module doc); returns the report"""
        rep = {"order": order, "t0_s": round(self.t(), 3)}
        t_wall = time.time()
        assign = self.assign_thrusters()
        rep["assignment"] = dict(assign)
        seq = [s for s in ORDERS[order] if slots is None or s in slots]
        rep["sequence"] = seq
        recs = []
        for sid in seq:
            if self.cell.done:
                break
            if sid.startswith("guard_"):
                recs.append(self.guard(sid, order=order))
            else:
                recs.append(self.thruster(sid, assign[sid]))
            self.checkpoint("unit %s" % sid)
        how = self.go_ready()
        rep["path_to_ready"] = how
        rep["units"] = recs
        rep["t1_s"] = round(self.t(), 3)
        rep["wall_s"] = round(time.time() - t_wall, 1)
        rep["carry"] = self.carry_log()
        rep["unplanned_contacts"] = [{k: v for k, v in x.items() if k != "key"} for x in self.contacts]
        rep["checkpoints"] = list(self.checkpoints)
        return rep


def program_r23(cell, order="guards_first", log=print, asm=None, read_fixture=None, slots=None):
    """run the R2 / R3 segment of the privileged reference program on `cell` (deck up; a RovCell with the arm); asm:
    part I's Assembler to go on from (its monitor and checkpoints carry over) or None. The report and the assembler."""
    if asm is None or not isinstance(asm, StepAssembler):
        asm = StepAssembler(cell, log=log, read_fixture=read_fixture)
    rep = asm.r23(order=order, slots=slots)
    return rep, asm
