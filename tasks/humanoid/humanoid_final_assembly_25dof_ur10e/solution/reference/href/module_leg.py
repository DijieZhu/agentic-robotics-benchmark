"""THE LEGS' assembly (mates "leg_R" and "leg_L": the leg, carried with its hip-roll unit, onto the hip-pitch output
rotor's flange, SIDEWAYS: the leg moves inboard along the seat axis into the rotor's shallow pilot; the design bolt
position; the leg route of record, the leg
stations of the fixtures, the coupling's windows) as named phases, each with its checks. ONE FILE FOR BOTH
SIDES: LegModule(side) with side "R" or "L" (RIGHT and LEFT below); the runner calls RIGHT.assemble(ctx) /
LEFT.assemble(ctx). The poses are the TRUE poses of the privileged calls (the part body = the hip-roll unit's bracket
that carries the flange; the thigh, which the hand holds, is placed from it by the two hip joints' readings and the
hip chain of data/legs.json); the motion is the API's own (motion.py). Constants and where each comes from:

  GRASP_FORCE_N    125 N per pad (the force register's top; the leg is 6.97 kg): the pads' torsion capacity 2 x 0.005 m
                   x 125 N = 1.25 N m; the grasp is a row of data/grasps.json (its rule there)
  OPEN_CLEAR_M     25 mm: the open hand at the approach = the grasp width + 25 mm (the grasp table's open hand)
  RELEASE_CLEAR_M  8 mm: the hand opened at the release to the grasp width + 8 mm, 4 mm per side (the inner finger
                   stays clear of the stand's front stem and of the other leg)
  PRE_PICK_M       0.12 m: the open hand's straight approach along the grasp's approach axis (the grasp table's)
  VIA_UP_M         0.35 m: the via point above the pre-pick; home -> via by the TUCKED turn: the forearm
                   folded in first (TUCK_Q), then the pan, then the via (a static check of the poses: the short pan turn
                   at home's posture passes the hand 29 mm into a mounted head; the tucked turn keeps 10 mm off the
                   fully assembled robot); back home the same way in reverse
  APPROACH_V       0.04 m/s: the last 0.12 m of the approach
  EXTRACT_*        the leg station's straight extraction line (data/legs.json, in the station's frame = the leg's part
                   body frame as it stands): 15 mm and 60 mm at 0.02 m/s (off the stops, out of the sole nest; the
                   fixtures' design: contact-free from <= 0.25 mm of travel), a 1 s hold (the slip check), then to
                   0.45 m at the carry limits; EXTRACT_BACK_M[side]: from 15 to 60 mm the hand also moves
                   this far BACK from the station's front bar, along the normal of the bar's back face
                   (FRONT_BAR_AWAY, the station's frame: the fixtures' front bar is inclined 30 deg), and keeps
                   the offset up the rest of the line: 0 for the right leg, 35 mm for the left leg (the reason:
                   on the straight line the left knee bears on the bar while the leg turns in the hand, and the
                   knee's drawn-only M3 socket-head screw (U0337) and its washer passed 2.3 / 0.9 mm into the bar
                   in the true meshes; this change checked the same way, the left leg alone on the nominal
                   layout: 20 mm back leaves the screw 0.37 mm off the bar, 35 mm leaves it 3.6 mm and the washer
                   5.7 mm, the knee's collision pieces 7.0 mm off the bar; the grasp is unchanged)
  FRONT_M          0.17 m: the front point in front of the pre-insertion pose, at its height (170 mm ahead), and
                   FRONT_OUT_M outboard of it along the seat axis: 0 for the right leg, 0.18 m for
                   the left leg (a static check of the poses: with the RIGHT leg still standing in its station, the
                   arm's wrist passes through the standing leg's hip end at the left leg's front point 170 mm ahead
                   (42 mm overlap at the nominal layout, a 319 N touch in a measured run); 120 mm outboard keeps
                   8.7 mm on the drawn layout that brings them closest (seed 25), 180 mm keeps more); the carry: the
                   end of the extraction -> above the front point (its height kept) -> the front point (Cartesian lines
                   at the carry limits) -> a straight line to the pre-insertion pose at PRE_LINE_V 0.05 m/s
  PRE_OUT_M        18 mm: the pre-insertion pose, the lateral gap open: 18 mm outboard of the seat along its axis (the leg
                   route's lateral gap); there the leg is set right on its seat by the true pose (ALIGN_*)
  ALIGN2_M         4 mm outboard: a second setting-right before the pilot engages (it engages 2.5 mm out: the graph's
                   pilot overlap)
  INSERT_V         0.008 m/s from 18 to 4 mm and 0.005 m/s from 4 to 1.5 mm (8 mm/s first; slower into the pilot)
  SLOW_LAST_M      1.5 mm at SEAT_V 1 mm/s (the frozen numbers' stand.last_insertion_mm_at_1_mm_s; reference behaviour
                   'slows the last 1.5 mm of an insertion to 1 mm/s'), aimed OVERTRAVEL_M 1.0 mm past the face
                   (the frozen numbers' stand.flange_overtravel_mm); the line stops when the coupling reports the lock
                   (the cell locks a module slower than 5 mm/s inside the lock window after 0.50 s: at 1 mm/s that is at
                   the face, while the leg is still held), or on the wrist force (GUARD_DF_N 40 N of change
                   from the reading at rest before the line: a jam), or at its end
  HOLD_AFTER_LOCK_S 0.6 s in the hand after the lock (held at least 0.6 s)
  RECENTRE_V       0.005 m/s: the opened hand set back on the thigh's TRUE pose where the grasp put it before it backs
                   out (a straight retreat dragged a mounted thigh about its hip yaw and the foot struck the stand)
  RETREAT_*        30 mm back along the approach drifting 3 mm outboard at 0.02 m/s; then the hand opened to the
                   grasp width + RETREAT_OPEN_M 20 mm (10 mm per side: a static check of the retreat
                   against the RELEASED leg found the inner pad passing the thigh 0.12-0.49 mm off it with 4 mm per
                   side, 5.96 mm or more with 10 mm; the inner finger's coupler stays 5.6 mm or more from the stand's
                   front stem; opened at the seat instead it comes within 2.7 mm of the stem); then the side's
                   segments RETREAT_2[side] (back, outboard) at 0.05 m/s: the right leg 120 mm straight back;
                   the left leg 60 mm straight back, then 60 mm back with 180 mm outboard (straight back all the way
                   the wrist passed 4.5 mm from the hip end of a right leg still standing in its station on the
                   nominal layout and overlapped it 1.3 mm on seed 82; drifting outboard from the first 30 mm on, the
                   inner pad dragged the released thigh: the left leg alone on seed 82 in a measured run;
                   with 90 mm straight first the pads keep 5.96 mm or more from the released thigh and the arm 26.5 mm
                   or more from a standing right leg on the six layouts), up to the via height, then home by the
                   tucked turn
The reference always places the leg at its design bolt position (k_offset 0); the k_offset argument exists only for
seating it that many bolt positions off (solve(cell, k_offset=...)), never for the reference."""
import json
import math
import os

import numpy as np

from . import grasp as G
from . import tf
from .errors import ReferenceFailure

HERE = os.path.dirname(os.path.abspath(__file__))
GRASP_FORCE_N = 125.0
OPEN_CLEAR_M = 0.025
RELEASE_CLEAR_M = 0.008
PRE_PICK_M = 0.12
VIA_UP_M = 0.35
APPROACH_V = 0.04
EXTRACT_SLOW_M, EXTRACT_FREE_M, EXTRACT_END_M = 0.015, 0.060, 0.45
EXTRACT_SLOW_V = 0.02
EXTRACT_BACK_M = {"R": 0.0, "L": 0.035}
FRONT_BAR_AWAY = (-math.cos(math.radians(30.0)), 0.0, math.sin(math.radians(30.0)))   # the station's frame
SLIP_HOLD_S = 1.0
FRONT_M = 0.17
FRONT_OUT_M = {"R": 0.0, "L": 0.18}
PRE_LINE_V = 0.05
PRE_OUT_M = 0.018
ALIGN2_M = 0.004
INSERT_V = (0.008, 0.005)
SLOW_LAST_M = 0.0015
SEAT_V = 0.001
OVERTRAVEL_M = 0.001
GUARD_DF_N = 40.0
LOCK_WAIT_S = 3.0
HOLD_AFTER_LOCK_S = 0.6
RECENTRE_V = 0.005
RETREAT_1_M, RETREAT_1_OUT_M, RETREAT_1_V = 0.03, 0.003, 0.02
RETREAT_OPEN_M = 0.020
RETREAT_2_V = 0.05
RETREAT_2 = {"R": ((0.12, 0.0),), "L": ((0.06, 0.0), (0.06, 0.18))}   # (back, outboard) per segment, m
TUCK_Q = (1, -1.5708, 2, 2.6)          # joints 1 and 2 (shoulder lift, elbow) of the tucked posture (rad)
HELD_TOL_P, HELD_TOL_R = 0.010, 0.05    # a held leg's static sag: the actual tcp may lie this far from the commanded
SLIP_MAX_M, SLIP_MAX_RAD = 0.008, math.radians(15.0)  # a guard against a FAILING grip (the leg sliding out of the pads),
                                        # read at the grasp centre after the extraction: the leg turns in the hand about the
                                        # closing axis as it leaves its tilted station (the gravity torque there is above the pads'
                                        # 1.25 N m of torsion, and the shank drags on the front bar while the knee is still on it):
                                        # three measured runs gave 3.98 deg / 0.88 mm (right) and 7.03 deg / 3.52 mm (left);
                                        # the carry and the seat use the grasp as it is and the hand is set back on the thigh's
                                        # true pose before it backs out. 8 mm keeps the grasp centre inside the pads' 13.5 mm half
                                        # width
ALIGN_LIMITS = {"lateral_m": 0.0003, "tilt_rad": math.radians(0.2), "residual_rad": math.radians(1.0)}
STATION_JOINT_DEG = 0.5                 # a leg in its station: every joint within 0.5 deg of its delivered angle (the
                                        # fixtures' design: on a stop or within 0.3 deg; the knees at their limit)
WORD = {"R": "right", "L": "left"}


def load_legs(path=None):
    with open(path or os.path.join(HERE, "data", "legs.json")) as f:
        return json.load(f)


def _about(axis, point, th):
    """the rigid motion that turns by th about the axis through point"""
    R = tf.exp_rot(np.asarray(axis, float) / np.linalg.norm(axis) * th)
    p = np.asarray(point, float)
    H = np.eye(4)
    H[:3, :3] = R
    H[:3, 3] = p - R @ p
    return H


def _up(T, dz):
    out = np.array(T, float)
    out[2, 3] += dz
    return out


def _at_height(T, z):
    out = np.array(T, float)
    out[2, 3] = z
    return out


def _moved(T, d):
    out = np.array(T, float)
    out[:3, 3] += np.asarray(d, float)
    return out


class LegModule(object):
    def __init__(self, side):
        if side not in ("R", "L"):
            raise ValueError("side must be 'R' or 'L'")
        self.side = side
        self.mate = "leg_" + side
        self.word = WORD[side]
        self._legs = None

    def legs(self):
        if self._legs is None:
            self._legs = load_legs()["sides"][self.side]
        return self._legs

    # ---------------------------------------------------------------- the leg's bodies from the privileged state
    def thigh_T(self, truth, st):
        """the thigh body (the hip yaw link) from the part body's true pose and the two hip joints' readings"""
        T = truth.body_T(self.mate, "part", st)
        for j in self.legs()["hip_chain"]:
            T = T @ _about(j["axis"], j["anchor_m"], float(st["joints_rad"][j["joint"]]) - float(j["ref_rad"]))
        return T

    def joints_deg(self, st):
        return {k: math.degrees(v) for k, v in st["joints_rad"].items() if k.startswith(self.word + "_") and
                any(s in k for s in ("hip_roll", "hip_yaw", "knee", "ankle"))}

    def _part_want(self, truth, k_offset, h, st):
        """the part body's pose h outboard of its design pose along the seat axis (the carrier as it stands now)"""
        x = truth.ifaces[self.mate]
        Tc = truth.body_T(self.mate, "carrier", st)
        Tz = np.eye(4)
        Tz[2, 3] = h
        return Tc @ x["S"] @ Tz @ tf.rotz(k_offset * x["pitch"]) @ tf.inv(x["F"])

    def _seat_axis(self, truth, st):
        """the seat's outward axis in the world (from the carrier into the module)"""
        x = truth.ifaces[self.mate]
        return (truth.body_T(self.mate, "carrier", st) @ x["S"])[:3, 2]

    def _set_right(self, ctx, k_offset, h, what, steps=3):
        """set the held leg right on its seat by its TRUE pose, h outboard (the head's align: the world correction of
        the held part applied to the commanded tcp)"""
        mo, truth, kin, log = ctx["motion"], ctx["truth"], ctx["kin"], ctx["log"]
        corr = []
        for it in range(steps):
            mo.hold(0.3)
            st = truth.read()
            Tp_now = truth.body_T(self.mate, "part", st)
            dT = self._part_want(truth, k_offset, h, st) @ tf.inv(Tp_now)
            e = truth.seat_errors(self.mate, st)
            corr.append({"lateral_mm": e["lateral_m"] * 1e3, "axial_mm": e["axial_m"] * 1e3, "tilt_deg": math.degrees(e["tilt_rad"]),
                         "residual_deg": math.degrees(e["residual_rad"]), "move_mm": float(np.linalg.norm(dT[:3, 3])) * 1e3,
                         "turn_deg": math.degrees(tf.angle(dT[:3, :3]))})
            if np.linalg.norm(dT[:3, 3]) < 5e-5 and tf.angle(dT[:3, :3]) < math.radians(0.02):
                break
            mo.move_linear(dT @ kin.fk_T(mo.cmd_q()), v=0.01, a=mo.a_carry, tol_p=HELD_TOL_P, tol_r=HELD_TOL_R, what=what)
        log.write("align", at_mm=h * 1e3, steps=corr)
        return corr

    def _tucked_home(self, ctx, q_target, home, what):
        """home -> the tucked posture -> its pan turned to q_target's -> q_target (joint moves)"""
        mo = ctx["motion"]
        qt = np.array(home, float)
        qt[TUCK_Q[0]], qt[TUCK_Q[2]] = TUCK_Q[1], TUCK_Q[3]
        qt2 = qt.copy()
        qt2[0] = q_target[0]
        mo.move_joint(qt, what="home -> tucked")
        mo.move_joint(qt2, what="tucked: the pan turned")
        mo.move_joint(q_target, what=what)

    def _home_tucked(self, ctx, home, what):
        mo = ctx["motion"]
        q = mo.cmd_q()
        qt = np.array(home, float)
        qt[TUCK_Q[0]], qt[TUCK_Q[2]] = TUCK_Q[1], TUCK_Q[3]
        qt[0] = q[0]
        mo.move_joint(qt, what="tucked over the cell")
        qt2 = qt.copy()
        qt2[0] = home[0]
        mo.move_joint(qt2, what="tucked: the pan turned home")
        mo.move_joint(home, what=what)

    # ---------------------------------------------------------------- the plan (from the true poses)
    def plan(self, ctx, k_offset=0):
        """the leg's waypoints from the TRUE poses as they stand (no motion): the grasp, the pick and its approach, the
        extraction line, the design pose, the seat, the pre-insertion and front points, and the closed-form IK of each
        on the home branch"""
        kin, truth, info = ctx["kin"], ctx["truth"], ctx["info"]
        M = self.mate
        home = np.array(info["robot"]["home_q"], float)
        legs = self.legs()
        gr = G.Grasp(G.load_grasps()[M][0])
        w_open = G.open_width(info, gr.width, OPEN_CLEAR_M)
        w_rel = G.open_width(info, gr.width, RELEASE_CLEAR_M)
        w_wide = G.open_width(info, gr.width, RETREAT_OPEN_M)
        w_close = gr.closing_width(GRASP_FORCE_N)
        Tg = gr.T_tcp_in_part(w_close)                       # the tcp in the thigh's frame at the grasp
        st = truth.read()
        T_part = truth.body_T(M, "part", st)
        T_thigh = self.thigh_T(truth, st)
        jd = self.joints_deg(st)
        T_pick = T_thigh @ Tg
        T_pre = _moved(T_pick, -PRE_PICK_M * T_pick[:3, 2])
        T_via = _up(T_pre, VIA_UP_M)
        u = T_part[:3, :3] @ np.asarray(legs["extraction_line_in_station"], float)
        back = EXTRACT_BACK_M[self.side] * (T_part[:3, :3] @ np.asarray(FRONT_BAR_AWAY, float))
        T_l1 = _moved(T_pick, EXTRACT_SLOW_M * u)
        T_l2, T_lift = (_moved(T_pick, d * u + back) for d in (EXTRACT_FREE_M, EXTRACT_END_M))
        G_part = tf.inv(T_part) @ T_thigh                     # the thigh in the part body's frame (the hips as they are)
        T_des = truth.design_part_T(M, k_offset, st)
        T_seat = T_des @ G_part @ Tg
        z_out = self._seat_axis(truth, st)
        fwd = T_des[:3, 0].copy()
        fwd[2] = 0.0
        fwd /= np.linalg.norm(fwd)
        T_preins = _moved(T_seat, PRE_OUT_M * z_out)
        T_front = _moved(T_preins, FRONT_M * fwd + FRONT_OUT_M[self.side] * z_out)
        T_high = _at_height(T_front, T_lift[2, 3])
        q = {k: kin.ik_branch(T, home) for k, T in (("via", T_via), ("pre", T_pre), ("pick", T_pick), ("lift", T_lift),
                                                   ("high", T_high), ("front", T_front), ("preins", T_preins), ("seat", T_seat))}
        ok = all(v is not None for v in q.values())
        jumps = ({"via_pre": float(np.max(np.abs(q["pre"] - q["via"]))), "pre_pick": float(np.max(np.abs(q["pick"] - q["pre"]))),
                  "preins_seat": float(np.max(np.abs(q["seat"] - q["preins"])))} if ok else {})
        return {"gr": gr, "Tg": Tg, "w_open": w_open, "w_rel": w_rel, "w_wide": w_wide, "w_close": w_close, "home": home, "st": st,
                "T_part": T_part, "T_thigh": T_thigh, "joints_deg": jd,
                "station_still": all(abs(v) <= STATION_JOINT_DEG for v in jd.values()),
                "T_pick": T_pick, "T_pre": T_pre, "T_via": T_via, "u": u, "back": back, "T_l1": T_l1, "T_l2": T_l2, "T_lift": T_lift,
                "G_part": G_part, "T_des": T_des, "T_seat": T_seat, "z_out": z_out, "fwd": fwd, "T_preins": T_preins,
                "T_front": T_front, "T_high": T_high, "q": q, "ik_ok": ok, "jumps": jumps}

    # ---------------------------------------------------------------- the assembly
    def assemble(self, ctx, k_offset=0):
        s, log, kin, mo, truth, info = ctx["session"], ctx["log"], ctx["kin"], ctx["motion"], ctx["truth"], ctx["info"]
        M = self.mate
        P = M + "."
        x = truth.ifaces[M]
        legs = self.legs()
        rep = {"mate": M, "k_offset": int(k_offset)}

        with log.phase(P + "plan"):
            p = self.plan(ctx, k_offset)
            gr, Tg, home = p["gr"], p["Tg"], p["home"]
            w_open, w_rel, w_close = p["w_open"], p["w_rel"], p["w_close"]
            T_pick, T_pre, T_via, T_l1, T_l2, T_lift = p["T_pick"], p["T_pre"], p["T_via"], p["T_l1"], p["T_l2"], p["T_lift"]
            fwd, q = p["fwd"], p["q"]
            rep.update({"grasp_row": gr.row.get("row"), "w_open_m": w_open, "w_close_m": w_close, "w_release_m": w_rel,
                        "w_retreat_m": p["w_wide"]})
            log.check("the leg stands still in its station (every joint within %.1f deg of 0)" % STATION_JOINT_DEG,
                      p["station_still"], joints_deg=p["joints_deg"])
            log.check("ik for every waypoint", p["ik_ok"], **{k: v for k, v in q.items()})
            jumps = p["jumps"]
            log.check("the pick and the seat each on one branch", jumps["via_pre"] < 1.0 and jumps["pre_pick"] < 0.6 and jumps["preins_seat"] < 0.2,
                      **jumps)
            log.write("plan", part_pos=p["T_part"][:3, 3], pick_tcp=T_pick[:3, 3], lift_tcp=T_lift[:3, 3], front_tcp=p["T_front"][:3, 3],
                      preins_tcp=p["T_preins"][:3, 3], seat_tcp=p["T_seat"][:3, 3], design_part_pos=p["T_des"][:3, 3],
                      extraction_line=p["u"], seat_axis_out=p["z_out"], w_open_m=w_open, w_close_m=w_close)

        with log.phase(P + "approach"):
            mo.gripper(w_open, GRASP_FORCE_N, what="open for the approach", max_s=1.5)
            self._tucked_home(ctx, q["via"], home, "tucked -> above the station")
            mo.move_linear(T_pre, what="down to the pre-pick")
            mo.move_linear(T_pick, v=APPROACH_V, what="the open hand onto the thigh")

        with log.phase(P + "grip"):
            g = mo.gripper(w_close, GRASP_FORCE_N, what="close on the thigh", max_s=2.0)
            log.check("the pads hold the thigh (measured width from 5 mm under to 4 mm over the thigh's true width: the "
                      "collision pieces may stand a little proud of the true surface)",
                      gr.width - 0.005 <= g["width_m"] <= gr.width + 0.004, width_m=g["width_m"], thigh_width_m=gr.width)
            mo.hold(0.2)
            st = truth.read()
            T_tcp = mo.tcp_actual()
            grip_thigh = tf.inv(T_tcp) @ self.thigh_T(truth, st)       # the thigh in the tcp frame (the grip as made)
            grip_part0 = tf.inv(T_tcp) @ truth.body_T(M, "part", st)
            rep["grip_offset_m"] = float(np.linalg.norm(grip_thigh[:3, 3] - tf.inv(Tg)[:3, 3]))
            rep["grip_turn_deg"] = math.degrees(tf.angle(grip_thigh[:3, :3].T @ Tg[:3, :3].T))

        with log.phase(P + "extract"):
            mo.move_linear(T_l1, v=EXTRACT_SLOW_V, a=mo.a_carry, tol_p=HELD_TOL_P, tol_r=HELD_TOL_R, what="off the stops")
            mo.move_linear(T_l2, v=EXTRACT_SLOW_V, a=mo.a_carry, tol_p=HELD_TOL_P, tol_r=HELD_TOL_R, what="out of the sole nest")
            mo.hold(SLIP_HOLD_S)
            st = truth.read()
            sl = _slip(grip_thigh, tf.inv(mo.tcp_actual()) @ self.thigh_T(truth, st), gr.centre)
            rep["slip_after_extraction"] = sl
            log.check("the leg holds in the hand off the station", sl["m"] <= SLIP_MAX_M and sl["rad"] <= SLIP_MAX_RAD,
                      slip_m=sl["m"], slip_rad=sl["rad"])
            mo.move_linear(T_lift, v=mo.v_carry, a=mo.a_carry, tol_p=HELD_TOL_P, tol_r=HELD_TOL_R, what="up the extraction line")

        with log.phase(P + "carry"):
            st = truth.read()
            T_tcp = mo.tcp_actual()
            grip_part1 = tf.inv(T_tcp) @ truth.body_T(M, "part", st)  # the grasp as it is
            T_des = truth.design_part_T(M, k_offset, st)
            T_seat = T_des @ tf.inv(grip_part1)
            z_out = self._seat_axis(truth, st)
            T_preins = _moved(T_seat, PRE_OUT_M * z_out)
            T_front = _moved(T_preins, FRONT_M * fwd + FRONT_OUT_M[self.side] * z_out)
            T_high = _at_height(T_front, kin.fk_T(mo.cmd_q())[2, 3])
            q_high = kin.ik_branch(T_high, home)
            q_front = kin.ik_branch(T_front, home)
            log.check("ik above the front point and at it with the grasp as it is", q_high is not None and q_front is not None)
            mo.move_linear(T_high, v=mo.v_carry, a=mo.a_carry, tol_p=HELD_TOL_P, tol_r=HELD_TOL_R, what="carry above the front point")
            mo.move_linear(T_front, v=mo.v_carry, a=mo.a_carry, tol_p=HELD_TOL_P, tol_r=HELD_TOL_R, what="down to the front point")
            mo.move_linear(T_preins, v=PRE_LINE_V, a=mo.a_carry, tol_p=HELD_TOL_P, tol_r=HELD_TOL_R,
                           what="back to the pre-insertion pose, the gap open")
            rep["slip_after_carry"] = _slip(grip_thigh, tf.inv(mo.tcp_actual()) @ self.thigh_T(truth, truth.read()), gr.centre)

        with log.phase(P + "align"):
            rep["align"] = self._set_right(ctx, k_offset, PRE_OUT_M, "set the leg right over its seat, the gap open")
            st = truth.read()
            e = truth.seat_errors(M, st)
            log.check("at the pre-insertion pose: lateral, tilt and phase set right",
                      e["lateral_m"] <= ALIGN_LIMITS["lateral_m"] and e["tilt_rad"] <= ALIGN_LIMITS["tilt_rad"]
                      and abs(e["residual_rad"]) <= ALIGN_LIMITS["residual_rad"]
                      and e["k_near"] == (x["design_k"] + k_offset) % x["positions"] and abs(e["axial_m"] - PRE_OUT_M) < 0.001,
                      lateral_mm=e["lateral_m"] * 1e3, tilt_deg=math.degrees(e["tilt_rad"]), residual_deg=math.degrees(e["residual_rad"]),
                      axial_mm=e["axial_m"] * 1e3, k_near=e["k_near"])

        with log.phase(P + "insert"):
            d_in = -self._seat_axis(truth, truth.read())
            e = truth.seat_errors(M)
            g1 = mo.guarded_line(d_in, max(0.0, e["axial_m"] - ALIGN2_M), INSERT_V[0], GUARD_DF_N, what="sideways towards the pilot")
            log.check("no jam before the pilot", g1["stop"] == "distance", **g1)
            rep["align2"] = self._set_right(ctx, k_offset, ALIGN2_M, "set the leg right before the pilot", steps=2)
            e = truth.seat_errors(M, truth.read())
            g2 = mo.guarded_line(d_in, max(0.0, e["axial_m"] - SLOW_LAST_M), INSERT_V[1], GUARD_DF_N, what="into the pilot")
            log.check("no jam in the pilot", g2["stop"] == "distance", **g2)
            st = truth.read()
            e = truth.seat_errors(M, st)
            rep["at_slow"] = {"axial_mm": e["axial_m"] * 1e3, "lateral_mm": e["lateral_m"] * 1e3, "tilt_deg": math.degrees(e["tilt_rad"]),
                              "residual_deg": math.degrees(e["residual_rad"]), "state": truth.state_of(M, st)}
            log.check("1.5 mm out, square, on its phase, not yet locked",
                      e["axial_m"] <= SLOW_LAST_M + 0.0005 and e["lateral_m"] <= 0.0005 and e["tilt_rad"] <= math.radians(0.3)
                      and truth.state_of(M, st) in ("FREE", "ENGAGED"), **rep["at_slow"])

        with log.phase(P + "seat"):
            e = truth.seat_errors(M)

            def locked():
                return truth.state_of(M, truth.read()) in ("LOCKED", "LOCKED_OFF", "VOID")

            gs = mo.guarded_line(d_in, max(0.0, e["axial_m"]) + OVERTRAVEL_M, SEAT_V, GUARD_DF_N, stop_when=locked,
                                 what="the last 1.5 mm at 1 mm/s, held")
            rep["seat_line"] = gs
            state = truth.state_of(M, truth.read())
            n = int(round(LOCK_WAIT_S / mo.tick))
            for i in range(0, n, 5):
                if state in ("LOCKED", "LOCKED_OFF", "VOID"):
                    break
                mo.hold(0.05)
                state = truth.state_of(M, truth.read())
            rec = truth.record_of(M)
            rep["state_in_hand"] = state
            rep["record"] = {k: rec.get(k) for k in ("state", "k", "design_ok", "t_lock_s", "t_holding_start_s", "lock_errors")} if rec else None
            log.write("lock", state=state, record=rep["record"], held=True)
            if k_offset == 0:
                log.check("the leg is LOCKED at its design bolt position while held", state == "LOCKED", state=state)
            mo.hold(HOLD_AFTER_LOCK_S)

        with log.phase(P + "release"):
            mo.gripper(w_rel, GRASP_FORCE_N, what="let go of the thigh", max_s=1.0)
            st = truth.read()
            T_want = self.thigh_T(truth, st) @ tf.inv(grip_thigh)       # the hand where the grasp put it on the thigh now
            T_now = mo.tcp_actual()
            rep["recentre"] = {"move_mm": float(np.linalg.norm(T_want[:3, 3] - T_now[:3, 3])) * 1e3,
                               "turn_deg": math.degrees(tf.angle(T_now[:3, :3].T @ T_want[:3, :3]))}
            mo.move_linear(T_want, v=RECENTRE_V, a=mo.a_carry, what="the open hand set back on the thigh's true pose")
            e = truth.seat_errors(M, truth.read())
            rep["seated"] = {"axial_mm": e["axial_m"] * 1e3, "lateral_mm": e["lateral_m"] * 1e3, "tilt_deg": math.degrees(e["tilt_rad"]),
                             "residual_deg": math.degrees(e["residual_rad"]), "k_near": e["k_near"]}

        with log.phase(P + "retreat"):
            st = truth.read()
            z_out = self._seat_axis(truth, st)
            T_r = retreat_waypoints(kin.fk_T(mo.cmd_q()), z_out, self.side)
            mo.move_linear(T_r[0], v=RETREAT_1_V, what="the open hand off the thigh")
            mo.gripper(p["w_wide"], GRASP_FORCE_N, what="the hand opened wider, clear of the thigh", max_s=1.0)
            for i, T in enumerate(T_r[1:]):
                mo.move_linear(T, v=RETREAT_2_V, what="back along the approach" if RETREAT_2[self.side][i][1] == 0.0
                               else "back and outboard, clear of the thigh")
            mo.move_linear(_at_height(T_r[-1], T_via[2, 3]), what="up to the via height")

        with log.phase(P + "home"):
            self._home_tucked(ctx, home, "home")
            st = s.state()
            log.check("the arm is at home", float(np.max(np.abs(np.array(st["robot"]["q"]) - home))) < 5e-3, q=st["robot"]["q"])
        return rep


def retreat_waypoints(T_cmd, z_out, side):
    """the retreat's tcp waypoints from the hand's pose after the release (T_cmd): the first 30 mm back along the
    approach (RETREAT_1_OUT_M outboard), then the side's segments RETREAT_2[side] (back, outboard), one after the other"""
    a = T_cmd[:3, 2]
    z = np.asarray(z_out, float)
    T = _moved(T_cmd, -RETREAT_1_M * a + RETREAT_1_OUT_M * z)
    out = [T]
    for back, side_out in RETREAT_2[side]:
        T = _moved(T, -back * a + side_out * z)
        out.append(T)
    return out


def _slip(G0, G1, centre):
    """the thigh's motion in the hand between two readings of its pose in the tcp frame: the grasp centre's shift (m) and
    the turn (rad)"""
    c = np.asarray(centre, float)
    p0 = G0[:3, :3] @ c + G0[:3, 3]
    p1 = G1[:3, :3] @ c + G1[:3, 3]
    return {"m": float(np.linalg.norm(p1 - p0)), "rad": tf.angle(G0[:3, :3].T @ G1[:3, :3])}


RIGHT = LegModule("R")
LEFT = LegModule("L")
