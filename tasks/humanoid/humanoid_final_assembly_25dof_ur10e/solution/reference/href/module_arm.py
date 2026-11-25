"""The arms' assembly (mates "arm_R" / "arm_L": the arm's shoulder ring 04B into the torso shell's shoulder socket, slid
in along the shoulder axis and fastened while it is held) as
named phases, each with its checks. ONE FILE FOR BOTH SIDES: the side is the mate id. The poses are the TRUE poses of
the privileged calls (the module's root body = the mate's part body, the torso = its carrier); the motion is the API's
own (motion.py). Constants and where each comes from:

  GRASP_FORCE_N  125 N per pad commanded (the top of the force register's 10-125 N; the grasp table's torsion margin
                 at 0.5 m/s^2 by the house rule 2 x 0.005 m x 125 N: 1.97 right row 173, 1.78 left row 170). MEASURED
                 in the cell (runs of both arms and, for the left arm's row 170, six runs of its own:
                 the medians of the pad forces in the carry): once the
                 arm's weight is in the hand the pads press
                   113.2-116.4 N each on the right arm, 128.0-132.9 N each on the left arm (row 170; an earlier,
                   superseded grasp of the left arm gave 44.2-46.0 N: the arm turned about 4 deg in the grip as it left the cradle),
                 so by the house rule the left grasp's torsion capacity (1.305-1.305 N m) is above the computed 0.700 N m
                 of the arm's off-centre weight and the carry; the measured slip in the carry of the right arm is at most
                   0.038 mm / 0.024 deg (in the same runs)
  GRASP ROWS     data/grasps.json (the grasp table's rows on the shoulder pitch link from above, closing front to back:
                 usable at the pick and at the mate in all 301 layouts of the fixtures of record; the hand 7.0 mm (right)
                 / 6.1 mm (left, an earlier, superseded grasp) clear of the torso when the arm is seated (a static check of the seated poses);
                 the left row 170 of the same family, 15 mm higher on the link, keeps the pads' force)
  OPEN_CLEAR_M   the open hand = the grasp width + 25 mm (at most the largest opening)
  Z_TRANSIT_M    0.60 m: the tcp height of the carry (the held arm hangs about 0.43 m under the tcp; the joint carry
                 swings it behind the cell's UR base: a static check of its poses found nothing within 15 mm); the carry
                 ends above the staging point at that height or, where the UR cannot reach that pose with the grasp as
                 it is, at the highest height under it that it reaches (10 mm steps, at least 0.45 m)
  PRE_PICK_M     0.12 m: the open hand's straight approach from above
  LIFT           the straight lift out of the rack's half cradle, past its axial shoulders (the fixtures' design:
                 contact-free from 0.25 mm for a module moved rigidly): the first 6 mm at 2 mm/s while the arm's weight
                 comes into the hand (the ring centre kept on its vertical line, the hand not turned) and up to 30 mm at
                 10 mm/s with the ROOT servoed on its true pose (motion.py's servo rule): the ring centre kept on the vertical
                 line through its rack pose and the ring square to the cradle's groove, its turn about the shoulder axis
                 left free (measured: the arm turns that way in the grip as it leaves the cradle,
                   3.8-4.01 deg (right) / 0.84-0.85 deg (left, row 170; 4.07-4.09 deg with the earlier, superseded grasp), and 5.6-6.41 deg
                   off it (right);
                 without the servo the ring swings 6 mm sideways into the cradle's wall and its flange wedges between
                 the axial shoulders, 0.2 mm each side, in the measured runs); then on to 60 mm
  STAGE_M        0.120 m: the carry ends ON the shoulder axis this far outboard of the seat (the stator's
                 inboard end 11.7 mm beyond the collar's mouth; a descent onto a point nearer drives the stator through
                 the collar wall)
  ENTRY_M        0.060 m: the second alignment on the axis before the ring meets the collar's mouth (about 56 mm out)
  SLIDE_V        0.015 m/s along the axis (0.03 m/s outside the collar), servoed on the true pose (motion.py's servo rule:
                 every 2 ticks 0.5 x the lateral and angular error, at most 0.5 mm / 0.1 deg), the slide through the
                 socket guarded on the wrist force (GUARD_INSERT_DF_N 15 N change)
  SLOW_FROM_M    0.0015 m: the last 1.5 mm at SLOW_V 1 mm/s (the frozen numbers' reference behaviour), under a guard on
                 the wrist force (GUARD_DF_N 10 N change: the seating push stays under the collar force of record,
                 22.6 N, PUSH_LIMIT_N)
  HOLD           the arm is HELD at its seat until the cell reports the lock (without the hold: a
                 released arm cocks 0.88-0.91 deg, backs out 0.86-1.06 mm and is never fastened); LOCK_WAIT_S 4 s
  RETREAT        the open hand straight up 0.10 m (the first 0.03 m at 0.02 m/s), after it is set back on the module's
                 true pose when it has moved in the hand by more than RECENTRE_M / RECENTRE_RAD
The reference always places the arm at its design bolt position (k_offset 0); the k_offset argument exists only for
seating it that many bolt positions off (solve(cell, k_offset=...)), never for the reference."""
import math

import numpy as np

from . import grasp as G
from . import tf
from .errors import ReferenceFailure

GRASP_FORCE_N = 125.0
OPEN_CLEAR_M = 0.025
Z_TRANSIT_M = 0.60
PRE_PICK_M = 0.12
APPROACH_V = 0.05
LIFT_SLOW_M = 0.06
LIFT_SLOW_V = 0.01
LIFT_TRANSFER_M = 0.006           # the first 6 mm at 2 mm/s: the arm's weight comes into the hand (the tcp sags about 4.4 mm
LIFT_TRANSFER_V = 0.002           # and the tool turns about 0.4 deg under it: the first measured run) while the servo
                                  # keeps the ring centre on its vertical line, the hand not turned (cap_rad 0)
LIFT_SERVO_M = 0.030              # the servo up to 30 mm: the ring flange leaves the cradle's groove between the axial
                                  # shoulders (0.2 mm each side: a turn of the ring by 0.22 deg jams it there [computed]; the first
                                  # measured run: the tool's turn under the load wedged the left arm's flange, 548 N)
STAGE_M = 0.120
ENTRY_M = 0.060
SLIDE_V = 0.015
SERVO_STEP_M = 0.005
SERVO_CAP_M = 0.0005
SERVO_CAP_RAD = math.radians(0.1)
SLOW_FROM_M = 0.0015
SLOW_V = 0.001
GUARD_DF_N = 10.0
GUARD_INSERT_DF_N = 15.0          # the slide through the socket stops when the wrist force changes by 15 N (a ring that
                                  # meets the collar's mouth or wall): under the collar force of record, 22.6 N
PUSH_LIMIT_N = 22.6
RACK_STILL_M = 0.0002             # the open hand stays clear of the arm until it closes: the root moves less than 0.2 mm
RACK_STILL_RAD = math.radians(0.1)   # and 0.1 deg in its rack during the approach
PUSH_PAST_M = 0.0003
LOCK_WAIT_S = 4.0
RETREAT_M = 0.10
RETREAT_SLOW_M = 0.03
RECENTRE_M = 0.001
RECENTRE_RAD = math.radians(0.5)
ABOVE_STAGE_DZ = 0.01             # the carry ends above the staging point at the transit height or, when the UR cannot reach
ABOVE_STAGE_MIN_Z = 0.45          # that pose with the grasp as it is (the arm turned in the grip), at the highest height under
                                  # it that it reaches, in steps of 10 mm, not under 0.45 m (a measured run: the
                                  # right stage 1.03 m out from the UR's base, the arm turned 6.9 deg in the grip: no IK at
                                  # 0.60 m, IK at 0.59 m); the descent from there is
                                  # the same vertical line
SLIP_GUARD_M = 0.012              # the CARRY's guard against a failing grip (the measured slip is reported; the seat uses the
SLIP_GUARD_RAD = math.radians(10.0)  # grasp as it is)
LIFT_RISE_TOL_M = 0.015           # the LIFT's guard against a failing grip: the ring centre must rise with the hand to within
                                  # 15 mm and the arm's turn in the grip stay under LIFT_TURN_RAD: in the measured runs the
LIFT_TURN_RAD = math.radians(15.0)  # arm turns in the grip as it leaves the cradle (the report's slip_after_lift):
                                  #   about the shoulder axis 3.8-4.01 deg (right) / 4.07-4.09 deg (left), off it 5.6-6.41 / 1.39-1.58 deg,
                                  #   6.78-7.56 / 4.3-4.37 deg in all; the ring centre moved 9.74-10.82 / 6.12-6.44 mm in the hand;
                                  # the turn is reported and taken out by the alignment 120 mm out
ALIGN_LIMITS = {"lateral_m": 0.00010, "tilt_rad": math.radians(0.05), "residual_rad": math.radians(0.3)}
ALIGN_STOP = {"m": 3e-5, "rad": math.radians(0.01)}
HELD = {"tol_p": 0.010, "tol_r": 0.035}   # a move with the arm in the hand ends when the COMMANDED tcp is at the target and the
                                          # arm is at rest; the actual tcp then sags under the 4.4 kg module by about 4-5 mm
                                          # and 0.4 deg (the first measured run: 4.39 mm, 0.41 deg), reported; more than
                                          # 10 mm / 2 deg would mean something holds the arm back


def _up(T, dz):
    out = np.array(T, float)
    out[2, 3] += dz
    return out


def _at_height(T, z):
    out = np.array(T, float)
    out[2, 3] = z
    return out


def _shift(T, d, s):
    out = np.array(T, float)
    out[:3, 3] += np.asarray(d, float) * float(s)
    return out


def _joints_of(st, side):
    pre = "right_" if side == "R" else "left_"
    return {k: float(v) for k, v in st["joints_rad"].items() if k.startswith(pre) and ("shoulder" in k or "elbow" in k or "wrist" in k)}


class Arm(object):
    """one arm's plan and its phases (both sides: mate 'arm_R' / 'arm_L')"""

    def __init__(self, ctx, mate, k_offset=0):
        self.ctx, self.mate, self.k_offset = ctx, mate, int(k_offset)
        self.side = mate[-1]
        self.s, self.log, self.kin, self.mo, self.truth, self.info = (ctx["session"], ctx["log"], ctx["kin"], ctx["motion"],
                                                                      ctx["truth"], ctx["info"])
        self.home = np.array(self.info["robot"]["home_q"], float)
        self.x = self.truth.ifaces[mate]
        self.mid = self.x["module"]
        self.gr = G.Grasp(G.load_grasps()[mate][0])
        self.w_open = G.open_width(self.info, self.gr.width, OPEN_CLEAR_M)
        self.w_close = self.gr.closing_width(GRASP_FORCE_N)
        self.rep = {"mate": mate, "k_offset": self.k_offset, "grasp_row": self.gr.row.get("row"), "w_open_m": self.w_open,
                    "w_close_m": self.w_close}

    # ------------------------------------------------------------------ poses
    def seat_axis(self, st=None):
        """the seat's axis in the world (unit, pointing OUT of the socket) and the seat frame (a fresh read if no state)"""
        st = st or self.truth.read()
        Ts = self.truth.body_T(self.mate, "carrier", st) @ self.x["S"]
        return Ts[:3, 2].copy(), Ts

    def part_want(self, s_out, st=None):
        """the part body's world pose s_out metres out along the seat axis from its design pose on the carrier as it
        stands now (k_offset bolt positions turned about the seat axis)"""
        Tc = self.truth.body_T(self.mate, "carrier", st)
        Tz = np.eye(4)
        Tz[2, 3] = float(s_out)
        return Tc @ self.x["S"] @ Tz @ tf.rotz(self.k_offset * self.x["pitch"]) @ tf.inv(self.x["F"])

    def iface_point(self, T_part):
        """the interface frame's origin (the ring's seating face centre on the axis) in the world, for a part pose"""
        return (T_part @ self.x["F"])[:3, 3]

    def grasp_in_tcp(self, st=None):
        """the part body in the actual tcp frame now (the grasp as it is)"""
        st = st or self.truth.read()
        return tf.inv(self.mo.tcp_actual()) @ self.truth.module_T(self.mid, st)

    def slip(self, T_tcp_part_a, T_tcp_part_b):
        """the part's move in the hand between two grasp readings: at the interface point (m) and its angle (rad)"""
        pa = (T_tcp_part_a @ self.x["F"])[:3, 3]
        pb = (T_tcp_part_b @ self.x["F"])[:3, 3]
        return float(np.linalg.norm(pb - pa)), tf.angle(T_tcp_part_a[:3, :3].T @ T_tcp_part_b[:3, :3])

    def errors(self, st=None):
        st = st or self.truth.read()            # a FRESH read (truth.last may be from before a hold)
        e = self.truth.seat_errors(self.mate, st)
        return {"axial_mm": e["axial_m"] * 1e3, "lateral_mm": e["lateral_m"] * 1e3, "tilt_deg": math.degrees(e["tilt_rad"]),
                "clock_deg": math.degrees(e["clock_rad"]), "residual_deg": math.degrees(e["residual_rad"]), "k_near": e["k_near"]}

    # ------------------------------------------------------------------ the true-pose correction (lateral + angular)
    def correct_once(self, s_out, axial=False, cap_m=None, cap_rad=None, st=None):
        """one correction of the commanded tcp from the held part's true pose against its wanted pose s_out out on
        the axis (motion.Motion.part_correction: rotation about the interface point, translation of the interface
        point; axial only when asked); returns (the new commanded tcp target, the correction's size)"""
        st = st or self.truth.read()
        Tp = self.truth.module_T(self.mid, st)
        Tw = self.part_want(s_out, st)
        axis, _ = self.seat_axis(st)
        T_cmd = self.kin.fk_T(self.mo.cmd_q())
        T_new, size = self.mo.part_correction(T_cmd, Tp, Tw, self.x["F"], axis, axial=axial, cap_m=cap_m, cap_rad=cap_rad)
        return T_new, size

    def align(self, s_out, what, n_max=4):
        corr = []
        for it in range(n_max):
            self.mo.hold(0.3)
            st = self.truth.read()
            T_new, size = self.correct_once(s_out, axial=True, st=st)
            e = self.errors(st)
            corr.append(dict(e, move_mm=size["m"] * 1e3, turn_deg=math.degrees(size["rad"])))
            if size["m"] < ALIGN_STOP["m"] and size["rad"] < ALIGN_STOP["rad"]:
                break
            self.mo.move_linear(T_new, v=0.01, a=self.mo.a_carry, what=what, **HELD)
        self.log.write("align", where=what, steps=corr)
        return corr

    def slide(self, s_from, s_to, v, what, guard_df=None, stop_on_lock=False, T0=None):
        """the held arm along the seat axis from s_from to s_to (m out of the seat) at v by motion.servo_line: the part's
        true pose against its wanted pose on the axis every 2 ticks (lateral and angular, 0.5 x the error, capped);
        optional wrist-force guard (F0 read at rest before) and a stop on the lock. Returns the servo's report with the
        part's largest lateral error seen and the reached s"""
        axis, _ = self.seat_axis()
        F0 = self.mo.wrench_mean(10) if guard_df is not None else None
        truth, mid = self.truth, self.mid
        worst = {"lateral_mm": 0.0, "tilt_deg": 0.0}

        def part_now():
            st = truth.read()
            e = self.errors(st)
            worst["lateral_mm"] = max(worst["lateral_mm"], e["lateral_mm"])
            worst["tilt_deg"] = max(worst["tilt_deg"], e["tilt_deg"])
            return truth.module_T(mid, st)

        stop_when = None
        if stop_on_lock:
            stop_when = lambda: truth.state_of(self.mate, truth.read()) in ("LOCKED", "LOCKED_OFF", "VOID")
        out = self.mo.servo_line(-axis, s_from - s_to, v, part_now, lambda s: self.part_want(s_from - s), self.x["F"],
                                 T0=T0, cap_m=SERVO_CAP_M, cap_rad=SERVO_CAP_RAD, F0=F0, guard_df=guard_df,
                                 stop_when=stop_when, what=what)
        out["lateral_mm_max"] = worst["lateral_mm"]
        out["tilt_deg_max"] = worst["tilt_deg"]
        out["s_end_mm"] = (s_from - out["travel_m"]) * 1e3
        self.T_stream = out.pop("T_end")
        self.log.write("slide", what=what, **out)
        return out

    # ------------------------------------------------------------------ the phases
    def plan(self):
        log, kin, truth, home = self.log, self.kin, self.truth, self.home
        with log.phase(self.mate + ".plan"):
            st = truth.read()
            T_part = truth.module_T(self.mid, st)
            self.T_pick = T_part @ self.gr.T_tcp_in_part(self.w_close)
            self.T_pre = _up(self.T_pick, PRE_PICK_M)
            self.T_above_rack = _at_height(self.T_pick, Z_TRANSIT_M)
            T_stage_part = self.part_want(STAGE_M, st)
            self.T_stage = T_stage_part @ self.gr.T_tcp_in_part(self.w_close)
            self.T_above_stage = _at_height(self.T_stage, Z_TRANSIT_M)
            T_seat = self.part_want(0.0, st) @ self.gr.T_tcp_in_part(self.w_close)
            qs = {"above_rack": kin.ik_branch(self.T_above_rack, home), "pre_pick": kin.ik_branch(self.T_pre, home),
                  "pick": kin.ik_branch(self.T_pick, home), "above_stage": kin.ik_branch(self.T_above_stage, home),
                  "stage": kin.ik_branch(self.T_stage, home), "seat": kin.ik_branch(T_seat, home)}
            log.check("ik for every waypoint", all(q is not None for q in qs.values()), **{k: q for k, q in qs.items()})
            log.check("pick on one branch", float(np.max(np.abs(qs["pick"] - qs["pre_pick"]))) < 0.6
                      and float(np.max(np.abs(qs["pre_pick"] - qs["above_rack"]))) < 1.0,
                      dq_pick_pre=float(np.max(np.abs(qs["pick"] - qs["pre_pick"]))),
                      dq_pre_above=float(np.max(np.abs(qs["pre_pick"] - qs["above_rack"]))))
            log.check("insertion on one branch", float(np.max(np.abs(qs["seat"] - qs["stage"]))) < 0.5
                      and float(np.max(np.abs(qs["stage"] - qs["above_stage"]))) < 1.0,
                      dq_seat_stage=float(np.max(np.abs(qs["seat"] - qs["stage"]))),
                      dq_stage_above=float(np.max(np.abs(qs["stage"] - qs["above_stage"]))))
            jq = _joints_of(st, self.side)
            log.check("the arm lies still in its rack at its delivered state (elbow flexed about 45 deg)",
                      abs(abs(jq[("right_" if self.side == "R" else "left_") + "elbow_joint"]) - math.radians(45.0)) < math.radians(1.0),
                      joints_rad=jq)
            self.qs = qs
            self.rep["joints_in_rack_rad"] = jq
            log.write("plan", part_pos=T_part[:3, 3], pick_tcp=self.T_pick[:3, 3], stage_tcp=self.T_stage[:3, 3],
                      seat_tcp=T_seat[:3, 3], w_open_m=self.w_open, w_close_m=self.w_close)

    def approach(self):
        mo = self.mo
        with self.log.phase(self.mate + ".approach"):
            mo.gripper(self.w_open, GRASP_FORCE_N, what="open for the approach", max_s=1.5)
            mo.move_joint(self.qs["above_rack"], what="home -> above the rack")
            mo.move_linear(self.T_pre, what="down to the pre-pick")
            T0 = self.truth.module_T(self.mid, self.truth.read())
            mo.move_linear(self.T_pick, v=APPROACH_V, what="the open hand onto the shoulder pitch link")
            T1 = self.truth.module_T(self.mid, self.truth.read())
            dp, da = float(np.linalg.norm(T1[:3, 3] - T0[:3, 3])), tf.angle(T0[:3, :3].T @ T1[:3, :3])
            self.log.check("the open hand came onto the arm without moving it in its rack", dp <= RACK_STILL_M and da <= RACK_STILL_RAD,
                           moved_m=dp, turned_rad=da)

    def grip(self):
        mo, log = self.mo, self.log
        with log.phase(self.mate + ".grip"):
            g = mo.gripper(self.w_close, GRASP_FORCE_N, what="close on the shoulder pitch link", max_s=2.0)
            log.check("the pads hold the arm (measured width from 5 mm under to 4 mm over the true width: the collision "
                      "pieces stand a little proud of the true surface)",
                      self.gr.width - 0.005 <= g["width_m"] <= self.gr.width + 0.004, width_m=g["width_m"], arm_width_m=self.gr.width)
            mo.hold(0.2)
            st = self.truth.read()
            self.G0 = self.grasp_in_tcp(st)
            self.T_grip_actual = mo.tcp_actual()
            Gn = tf.inv(self.gr.T_tcp_in_part(self.w_close))
            self.rep["grip_offset"] = dict(zip(("m", "rad"), self.slip(Gn, self.G0)))
            self.rep["joints_gripped_rad"] = _joints_of(st, self.side)

    def extract(self):
        mo, log = self.mo, self.log
        with log.phase(self.mate + ".extract"):
            # the ROOT is kept on the vertical line through its rack pose (the ring centre's x and y) and square to the
            # cradle's groove (its turns about the world x and z), while its turn about the shoulder axis (world y: the
            # ring turning in its own plane, harmless in the groove) is left free: the arm turns 3-4 deg about that axis
            # in the grip as it leaves the cradle (measured: the pads on the pitch link's convex pieces),
            # which without the servo swings the ring 6 mm sideways into the cradle's wall while it is still in its groove
            truth, mid = self.truth, self.mid
            T_root0 = truth.module_T(mid, truth.read())
            up = np.array([0.0, 0.0, 1.0])
            y_axis = T_root0[:3, :3] @ (self.x["F"][:3, 2])          # the shoulder axis of the arm in its rack
            part_now = lambda: truth.module_T(mid, truth.read())
            want = lambda s0: (lambda s: _shift(T_root0, up, s0 + s))
            r1 = mo.servo_line(up, LIFT_TRANSFER_M, LIFT_TRANSFER_V, part_now, want(0.0), self.x["F"], cap_m=SERVO_CAP_M,
                               cap_rad=0.0, rot_free=y_axis,
                               what="the arm's weight into the hand, the ring kept on its rack line (no turn of the hand while the "
                                    "arm still rests in its cradle)")
            r2 = mo.servo_line(up, LIFT_SERVO_M - LIFT_TRANSFER_M, LIFT_SLOW_V, part_now, want(LIFT_TRANSFER_M), self.x["F"],
                               T0=r1["T_end"], cap_m=SERVO_CAP_M, cap_rad=SERVO_CAP_RAD, rot_free=y_axis,
                               what="straight up out of the cradle past its axial shoulders, the ring kept square")
            self.rep["extract_servo"] = [{k: v for k, v in r.items() if k != "T_end"} for r in (r1, r2)]
            c_rack = (T_root0 @ self.x["F"])[:3, 3]
            mo.move_linear(_up(r2["T_end"], LIFT_SLOW_M - LIFT_SERVO_M), v=LIFT_SLOW_V, a=mo.a_carry, T0=r2["T_end"],
                           what="on up to 60 mm", **HELD)
            T_cmd = self.kin.fk_T(mo.cmd_q())
            mo.move_linear(_at_height(T_cmd, Z_TRANSIT_M), v=mo.v_carry, a=mo.a_carry, what="up to the transit height", **HELD)
            mo.hold(0.5)
            st = self.truth.read()
            self.G1 = self.grasp_in_tcp(st)
            dp, da = self.slip(self.G0, self.G1)
            rv = tf.rotvec(self.G1[:3, :3] @ self.G0[:3, :3].T)            # the arm's turn in the grip (tcp frame)
            a_tcp = self.G0[:3, :3] @ self.x["F"][:3, 2]                    # the shoulder axis in the tcp frame
            about = float(rv @ a_tcp)
            off = float(np.linalg.norm(rv - about * a_tcp))
            c_now = (self.truth.module_T(self.mid, st) @ self.x["F"])[:3, 3]
            rise_part = float(c_now[2] - c_rack[2])
            rise_hand = float(mo.tcp_actual()[2, 3] - self.T_grip_actual[2, 3])
            self.rep["slip_after_lift"] = {"m": dp, "rad": da, "turn_about_the_shoulder_axis_rad": about, "turn_off_the_axis_rad": off,
                                           "ring_centre_rise_m": rise_part, "hand_rise_m": rise_hand}
            self.rep["joints_lifted_rad"] = _joints_of(st, self.side)
            log.check("the grip holds after the lift (a guard against a failing grip: the ring centre rose with the hand and the "
                      "arm's turn in the grip is under 15 deg; the slip and its turn about and off the shoulder axis are reported)",
                      abs(rise_part - rise_hand) <= LIFT_RISE_TOL_M and da <= LIFT_TURN_RAD, **self.rep["slip_after_lift"])

    def carry(self):
        mo, log, kin, home = self.mo, self.log, self.kin, self.home
        with log.phase(self.mate + ".carry"):
            st = self.truth.read()
            T_stage = self.part_want(STAGE_M, st) @ tf.inv(self.G1)           # the grasp as it is
            q_above, z_above = None, None
            n_steps = int(round((Z_TRANSIT_M - ABOVE_STAGE_MIN_Z) / ABOVE_STAGE_DZ))
            for i in range(n_steps + 1):
                z = Z_TRANSIT_M - i * ABOVE_STAGE_DZ
                q_above = kin.ik_branch(_at_height(T_stage, z), home)
                if q_above is not None:
                    z_above = z
                    break
            self.rep["above_stage_z_m"] = z_above
            log.check("ik above the staging point with the grasp as it is (at the transit height or the highest height "
                      "under it that the UR reaches, at least 0.45 m)", q_above is not None, z_m=z_above)
            far = [tuple(float(v) for v in (self.G1 @ np.append(p, 1.0))[:3]) for p in self.module_far_points()]
            mo.move_joint(q_above, v=mo.v_carry, a=mo.a_carry, offsets=[(0.0, 0.0, 0.0)] + far,
                          what="carry behind the cell's UR to above the staging point")
            mo.move_linear(T_stage, v=mo.v_carry, a=mo.a_carry, what="down onto the shoulder axis, 120 mm outboard", **HELD)
            st = self.truth.read()
            self.G2 = self.grasp_in_tcp(st)
            dp, da = self.slip(self.G1, self.G2)
            self.rep["slip_after_carry"] = {"m": dp, "rad": da}
            j0, j1 = self.rep["joints_lifted_rad"], _joints_of(st, self.side)
            self.rep["joints_after_carry_rad"] = j1
            self.rep["joint_change_in_carry_deg"] = {k: math.degrees(j1[k] - j0[k]) for k in j1}
            log.check("the grip holds after the carry", dp <= SLIP_GUARD_M and da <= SLIP_GUARD_RAD, slip_m=dp, slip_rad=da)

    def module_far_points(self):
        """points of the held module in its part body frame that the carry's speed limit must cover: the interface
        point and the far end of the forearm (the module's own description: its delivered pose in the part frame)"""
        F = self.x["F"]
        pts = [F[:3, 3]]
        sgn = -1.0 if self.side == "R" else 1.0
        pts.append(np.array([-0.02, sgn * 0.17, -0.44]))        # about the wrist plate at the delivered state (CAD frame)
        return pts

    def pre_insert(self):
        log = self.log
        with log.phase(self.mate + ".pre_insert"):
            self.rep["align_stage"] = self.align(STAGE_M, "set the ring on the axis and its 12 x 30 deg pattern on its "
                                                          "design position (120 mm out)")
            e = self.errors()
            self.rep["at_stage"] = e
            x = self.x
            log.check("on the axis, square and on its bolt position before the slide",
                      e["lateral_mm"] * 1e-3 <= ALIGN_LIMITS["lateral_m"] and math.radians(e["tilt_deg"]) <= ALIGN_LIMITS["tilt_rad"]
                      and abs(math.radians(e["residual_deg"])) <= ALIGN_LIMITS["residual_rad"]
                      and e["k_near"] == (x["design_k"] + self.k_offset) % x["positions"], **e)

    def insert(self):
        log, mo = self.log, self.mo
        with log.phase(self.mate + ".insert"):
            self.rep["slide_outer"] = self.slide(STAGE_M, ENTRY_M, 2.0 * SLIDE_V, "along the axis to 60 mm out")
            mo.hold(0.3)
            self.rep["align_entry"] = self.align(ENTRY_M, "set the ring on the axis before the collar (60 mm out)")
            e = self.errors()
            log.check("on the axis before the collar", e["lateral_mm"] * 1e-3 <= ALIGN_LIMITS["lateral_m"]
                      and math.radians(e["tilt_deg"]) <= ALIGN_LIMITS["tilt_rad"], **e)
            self.rep["slide_inner"] = self.slide(ENTRY_M, SLOW_FROM_M, SLIDE_V, "the ring into the collar to 1.5 mm out (guarded)",
                                                 guard_df=GUARD_INSERT_DF_N)
            log.check("the slide through the socket was not stopped by the force guard", self.rep["slide_inner"]["stop"] == "distance",
                      stop=self.rep["slide_inner"]["stop"], dF_N_max=self.rep["slide_inner"].get("dF_N_max"))
            mo.hold(0.3)
            self.rep["at_slow"] = self.errors()
            log.check("the ring is at the slow point (the slide was not held back)",
                      abs(self.rep["at_slow"]["axial_mm"] - SLOW_FROM_M * 1e3) < 1.0, **self.rep["at_slow"])

    def seat(self):
        log, mo = self.log, self.mo
        with log.phase(self.mate + ".seat"):
            sl = self.slide(SLOW_FROM_M, -PUSH_PAST_M, SLOW_V, "the last 1.5 mm at 1 mm/s onto the seat face (guarded)",
                            guard_df=GUARD_DF_N, stop_on_lock=True, T0=self.T_stream)
            self.rep["slide_seat"] = sl
            log.check("the seating push stays under the collar force of record (the wrist force change)",
                      (sl.get("dF_N_max") or 0.0) < PUSH_LIMIT_N, dF_N_max=sl.get("dF_N_max"), limit_N=PUSH_LIMIT_N)
            self.rep["at_seat"] = self.errors()

    def hold_until_lock(self):
        log, mo, truth = self.log, self.mo, self.truth
        with log.phase(self.mate + ".hold"):
            n = int(round(LOCK_WAIT_S / 0.05))
            state = None
            for i in range(n):
                state = truth.state_of(self.mate, truth.read())
                if state in ("LOCKED", "LOCKED_OFF", "VOID"):
                    break
                mo.hold(0.05)
            rec = truth.record_of(self.mate)
            self.rep["state"] = state
            self.rep["record"] = {k: rec.get(k) for k in ("state", "k", "design_ok", "t_lock_s", "t_holding_start_s", "lock_errors")} if rec else None
            self.rep["at_lock"] = self.errors()
            log.write("lock", state=state, record=self.rep["record"], seated=self.rep["at_lock"])
            if self.k_offset == 0:
                log.check("the arm is LOCKED at its design bolt position", state == "LOCKED", state=state)
            else:
                log.check("the neighbouring bolt position ends LOCKED_OFF (or not fastened)", state in ("LOCKED_OFF", "FREE", "ENGAGED", "HOLDING"),
                          state=state)

    def release(self):
        mo, log = self.mo, self.log
        with log.phase(self.mate + ".release"):
            g = mo.gripper(self.w_open, GRASP_FORCE_N, what="let go of the arm", max_s=1.0)
            mo.hold(0.3)
            st = self.truth.read()
            self.rep["after_release"] = self.errors(st)
            self.rep["joints_released_rad"] = _joints_of(st, self.side)
            log.check("the hand is open off the arm and the arm stays fastened", g["width_m"] >= self.gr.width + 0.010
                      and self.truth.state_of(self.mate, st) == ("LOCKED" if self.k_offset == 0 else self.rep.get("state")),
                      width_m=g["width_m"], state=self.truth.state_of(self.mate, st))

    def retreat(self):
        mo, log, kin, home, s = self.mo, self.log, self.kin, self.home, self.s
        with log.phase(self.mate + ".retreat"):
            st = self.truth.read()
            Tp = self.truth.module_T(self.mid, st)
            T_cmd = kin.fk_T(mo.cmd_q())
            T_hand_now = mo.tcp_actual()
            T_nominal_open = Tp @ tf.inv(self.G1)                    # the hand where it held the arm (the grasp as it was)
            dp = float(np.linalg.norm(T_nominal_open[:3, 3] - T_hand_now[:3, 3]))
            da = tf.angle(T_nominal_open[:3, :3].T @ T_hand_now[:3, :3])
            self.rep["hand_offset_at_release"] = {"m": dp, "rad": da}
            if dp > RECENTRE_M or da > RECENTRE_RAD:
                T_fix = T_cmd.copy()
                T_fix[:3, 3] += T_nominal_open[:3, 3] - T_hand_now[:3, 3]
                mo.move_linear(T_fix, v=0.005, what="set the open hand back on the arm's true pose")
                T_cmd = kin.fk_T(mo.cmd_q())
            mo.move_linear(_up(T_cmd, RETREAT_SLOW_M), v=0.02, what="the open hand up off the arm")
            mo.move_linear(_up(T_cmd, RETREAT_M), v=0.05, what="up clear of the shoulder")
            mo.move_linear(_at_height(T_cmd, Z_TRANSIT_M), what="up to the transit height")
            mo.move_joint(home, what="home")
            st = s.state()
            log.check("the arm is at home", float(np.max(np.abs(np.array(st["robot"]["q"]) - home))) < 5e-3, q=st["robot"]["q"])


def assemble(ctx, mate, k_offset=0):
    a = Arm(ctx, mate, k_offset)
    a.plan()
    a.approach()
    a.grip()
    a.extract()
    a.carry()
    a.pre_insert()
    a.insert()
    a.seat()
    a.hold_until_lock()
    a.release()
    a.retreat()
    return a.rep


class _Side(object):
    """runner.MODULES entry for one side: assemble(ctx, k_offset=0)"""

    def __init__(self, mate):
        self.mate = mate

    def assemble(self, ctx, k_offset=0):
        return assemble(ctx, self.mate, k_offset=k_offset)


arm_R = _Side("arm_R")
arm_L = _Side("arm_L")
