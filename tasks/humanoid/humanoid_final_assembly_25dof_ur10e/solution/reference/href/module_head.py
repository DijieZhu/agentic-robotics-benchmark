"""The head's assembly (mate "head": the head's bracket onto the neck-yaw output rotor's boss, lowered from above, the
design bolt position) as named phases, each with its checks. The poses are the TRUE poses of the
privileged calls; the motion is the API's own (motion.py). Constants and where each comes from:

  GRASP_FORCE_N  80 N per pad (the force register's range 10-125 N): torsion capacity 2 x 0.005 m x 80 N = 0.8 N m
                 against about 0.07 N m about the closing axis at the carry limit's 0.5 m/s^2 (the head's centre of mass
                 about 107 mm under the grasp axis, 5 mm off it); slip capacity 2 x 0.7 x 80 N = 112 N against 11.9 N
  OPEN_CLEAR_M   the open hand = the grasp width + 25 mm, at most the largest opening (the grasp table's open hand)
  Z_TRANSIT_M    0.62 m: the tcp height of every carry and free move above the cell (the held head hangs about 0.19 m
                 under the tcp; the tallest things it passes over are the arm modules on their racks, top at about
                 0.20 m, and the torso at 0.34 m)
  PRE_PICK_M     0.12 m: the open hand's straight approach along the grasp's approach axis (the grasp table's 0.12 m)
  LIFT_SLOW_M    0.06 m at 0.04 m/s: the straight lift out of the nest's locating skirt (0.3 mm off the head, 45 mm tall)
  PRE_SEAT_M     6 mm above the seat: where the head is set right on its seat by the true pose (two corrections)
  GUARD_FROM_M   2 mm above the seat: from there a guarded descent at 1 mm/s (the last 1.5 mm of an insertion at 1 mm/s,
                 the frozen numbers' reference behaviour) down to RELEASE_AT_M by the true pose; it stops earlier if the
                 wrist force changes by GUARD_DF_N (4 N; the head weighs 11.9 N: a seat met sooner than the true pose says)
  RELEASE_AT_M   0.8 mm above the seat: the head is let go there, OUTSIDE the lock window (axial 0.5 mm), with its
                 bracket's bore already around the boss (the pilot overlaps 1.3 mm when seated): it settles on the boss
                 under its own weight and the cell fastens it by itself (the coupling's release table: the head locks by
                 itself; the head is the mate seated by gravity from above). Held inside the lock window the
                 head would be fastened in the hand: at 1 mm/s the lock condition (relative speed under 5 mm/s) runs the
                 whole last 0.5 mm, and a measured run locked 7 um above the face before any contact
  LOCK_WAIT_S    3 s for the coupling's lock after the release (it locks 0.50 s after the lock condition starts)
The reference always places the head at the design bolt position (k_offset 0); the k_offset argument exists only for
seating it that many bolt positions off (solve(cell, k_offset=...)), never for the reference."""
import math

import numpy as np

from . import grasp as G
from . import tf
from .errors import ReferenceFailure

MATE = "head"
GRASP_FORCE_N = 80.0
OPEN_CLEAR_M = 0.025
Z_TRANSIT_M = 0.62
PRE_PICK_M = 0.12
LIFT_SLOW_M = 0.06
LIFT_SLOW_V = 0.04
APPROACH_V = 0.05
PRE_SEAT_M = 0.006
GUARD_FROM_M = 0.002
GUARD_V = 0.001
GUARD_DF_N = 4.0
RELEASE_AT_M = 0.0008
RELEASE_WINDOW_M = (0.0006, 0.0012)
LOCK_WAIT_S = 3.0
SLIP_MAX_M = 0.001
SLIP_MAX_RAD = math.radians(1.0)
SEAT_LIMITS = {"axial_m": 0.0005, "lateral_m": 0.0005, "tilt_rad": math.radians(0.5), "residual_rad": math.radians(3.0)}


def _up(T, dz):
    out = np.array(T, float)
    out[2, 3] += dz
    return out


def _at_height(T, z):
    out = np.array(T, float)
    out[2, 3] = z
    return out


def assemble(ctx, k_offset=0):
    s, log, kin, mo, truth, info = ctx["session"], ctx["log"], ctx["kin"], ctx["motion"], ctx["truth"], ctx["info"]
    home = np.array(info["robot"]["home_q"], float)
    x = truth.ifaces[MATE]
    mid = x["module"]
    gr = G.Grasp(G.load_grasps()[MATE][0])
    w_open = G.open_width(info, gr.width, OPEN_CLEAR_M)
    w_close = gr.closing_width(GRASP_FORCE_N)
    carry_pts = [(0.0, 0.0, 0.0), (0.0, 0.0, 0.19)]          # the tcp and the held head's far end (tcp frame)
    rep = {"mate": MATE, "k_offset": int(k_offset), "grasp_row": gr.row.get("row"), "w_open_m": w_open, "w_close_m": w_close}

    with log.phase("head.plan"):
        st = truth.read()
        T_part = truth.module_T(mid, st)
        q_neck_pitch = [v for k, v in st["joints_rad"].items() if "neck_pitch" in k]
        T_pick = T_part @ gr.T_tcp_in_part(w_close)
        T_pre = T_pick.copy()
        T_pre[:3, 3] -= (T_part[:3, :3] @ gr.a) * PRE_PICK_M        # back along the approach (into the part)
        T_above_nest = _at_height(T_pre, Z_TRANSIT_M)
        T_des = truth.design_part_T(MATE, k_offset, st)
        T_place = T_des @ gr.T_tcp_in_part(w_close)
        q_above_nest = kin.ik_branch(T_above_nest, home)
        q_pre = kin.ik_branch(T_pre, home)
        q_pick = kin.ik_branch(T_pick, home)
        q_above_neck = kin.ik_branch(_at_height(T_place, Z_TRANSIT_M), home)
        log.check("ik for every waypoint", all(q is not None for q in (q_above_nest, q_pre, q_pick, q_above_neck)),
                  above_nest=q_above_nest, pre=q_pre, pick=q_pick, above_neck=q_above_neck)
        log.check("pick on one branch", float(np.max(np.abs(q_pick - q_pre))) < 0.5 and float(np.max(np.abs(q_pre - q_above_nest))) < 1.0,
                  dq_pick_pre=float(np.max(np.abs(q_pick - q_pre))), dq_pre_above=float(np.max(np.abs(q_pre - q_above_nest))))
        log.check("the head lies still in its nest (neck pitch)", all(abs(v) < math.radians(1.0) for v in q_neck_pitch), neck_pitch_rad=q_neck_pitch)
        log.write("plan", part_pos=T_part[:3, 3], pick_tcp=T_pick[:3, 3], place_tcp=T_place[:3, 3],
                  design_part_pos=T_des[:3, 3], w_open_m=w_open, w_close_m=w_close)

    with log.phase("head.approach"):
        mo.gripper(w_open, GRASP_FORCE_N, what="open for the approach", max_s=1.5)
        mo.move_joint(q_above_nest, what="home -> above the nest")
        mo.move_linear(T_pre, what="down to the pre-pick")
        mo.move_linear(T_pick, v=APPROACH_V, what="the open hand onto the head")

    with log.phase("head.grip"):
        g = mo.gripper(w_close, GRASP_FORCE_N, what="close on the head", max_s=2.0)
        log.check("the pads hold the head (measured width from 5 mm under to 4 mm over the head's true width: the "
                  "collision pieces stand a little proud of the true surface)",
                  gr.width - 0.005 <= g["width_m"] <= gr.width + 0.004, width_m=g["width_m"], head_width_m=gr.width)
        mo.hold(0.2)
        st = truth.read()
        T_tcp = mo.tcp_actual()
        T_tcp_part0 = tf.inv(T_tcp) @ truth.module_T(mid, st)
        rep["grip_offset_m"] = float(np.linalg.norm(T_tcp_part0[:3, 3] - tf.inv(gr.T_tcp_in_part(w_close))[:3, 3]))

    with log.phase("head.lift"):
        mo.move_linear(_up(T_pick, LIFT_SLOW_M), v=LIFT_SLOW_V, a=mo.a_carry, what="straight out of the nest")
        mo.move_linear(_at_height(T_pick, Z_TRANSIT_M), v=mo.v_carry, a=mo.a_carry, what="up to the transit height")
        mo.hold(0.5)
        st = truth.read()
        T_tcp_part1 = tf.inv(mo.tcp_actual()) @ truth.module_T(mid, st)
        dp = float(np.linalg.norm(T_tcp_part1[:3, 3] - T_tcp_part0[:3, 3]))
        da = tf.angle(T_tcp_part0[:3, :3].T @ T_tcp_part1[:3, :3])
        log.check("no slip in the hand after the lift", dp <= SLIP_MAX_M and da <= SLIP_MAX_RAD, slip_m=dp, slip_rad=da)
        rep["slip_after_lift"] = {"m": dp, "rad": da}

    with log.phase("head.carry"):
        st = truth.read()
        T_des = truth.design_part_T(MATE, k_offset, st)
        T_place = T_des @ tf.inv(T_tcp_part1)                     # the grasp as it is
        q_above_neck = kin.ik_branch(_at_height(T_place, Z_TRANSIT_M), home)
        log.check("ik above the neck with the grasp as it is", q_above_neck is not None)
        mo.move_joint(q_above_neck, v=mo.v_carry, a=mo.a_carry, offsets=carry_pts, what="carry over the cell to above the neck")
        mo.move_linear(_up(T_place, PRE_SEAT_M), v=mo.v_carry, a=mo.a_carry, what="down to 6 mm above the seat")

    with log.phase("head.align"):
        corr = []
        for it in range(3):
            mo.hold(0.3)
            st = truth.read()
            Tp_now = truth.module_T(mid, st)
            Tp_want = _up_seat(truth, k_offset, PRE_SEAT_M, st)
            dT = Tp_want @ tf.inv(Tp_now)                        # world correction of the held part
            e = truth.seat_errors(MATE, st)
            corr.append({"lateral_mm": e["lateral_m"] * 1e3, "axial_mm": e["axial_m"] * 1e3,
                         "tilt_deg": math.degrees(e["tilt_rad"]), "residual_deg": math.degrees(e["residual_rad"]),
                         "move_mm": float(np.linalg.norm(dT[:3, 3])) * 1e3, "turn_deg": math.degrees(tf.angle(dT[:3, :3]))})
            if np.linalg.norm(dT[:3, 3]) < 5e-5 and tf.angle(dT[:3, :3]) < math.radians(0.02):
                break
            T_cmd = kin.fk_T(mo.cmd_q())
            mo.move_linear(dT @ T_cmd, v=0.01, a=mo.a_carry, what="set the head right over its seat")
        rep["align"] = corr
        log.write("align", steps=corr)
        e = truth.seat_errors(MATE, truth.read())
        log.check("over the seat: lateral, tilt and phase inside the lock window",
                  e["lateral_m"] <= SEAT_LIMITS["lateral_m"] and e["tilt_rad"] <= SEAT_LIMITS["tilt_rad"]
                  and abs(e["residual_rad"]) <= SEAT_LIMITS["residual_rad"] and e["k_near"] == (x["design_k"] + k_offset) % x["positions"],
                  lateral_mm=e["lateral_m"] * 1e3, tilt_deg=math.degrees(e["tilt_rad"]), residual_deg=math.degrees(e["residual_rad"]),
                  axial_mm=e["axial_m"] * 1e3, k_near=e["k_near"])

    with log.phase("head.set_down"):
        d_seat = -(truth.body_T(MATE, "carrier") @ x["S"])[:3, 2]
        T_cmd = kin.fk_T(mo.cmd_q())
        T2 = T_cmd.copy()
        T2[:3, 3] += d_seat * (PRE_SEAT_M - GUARD_FROM_M)
        mo.move_linear(T2, v=0.005, a=mo.a_carry, what="down to 2 mm above the seat")
        mo.hold(0.2)
        e = truth.seat_errors(MATE, truth.read())
        travel = max(0.0, e["axial_m"] - RELEASE_AT_M)
        gd = mo.guarded_line(d_seat, travel, GUARD_V, GUARD_DF_N, what="guarded descent to just above the boss")
        rep["guarded"] = gd
        mo.hold(0.3)
        e = truth.seat_errors(MATE, truth.read())
        rep["at_release"] = {"axial_mm": e["axial_m"] * 1e3, "lateral_mm": e["lateral_m"] * 1e3, "tilt_deg": math.degrees(e["tilt_rad"]),
                             "residual_deg": math.degrees(e["residual_rad"]), "state": truth.state_of(MATE), "guard_stop": gd["stop"]}
        log.check("just above the boss, still, outside the lock window, square and on its phase",
                  RELEASE_WINDOW_M[0] <= e["axial_m"] <= RELEASE_WINDOW_M[1] and e["lateral_m"] <= 0.0003
                  and e["tilt_rad"] <= math.radians(0.3) and abs(e["residual_rad"]) <= math.radians(1.0)
                  and truth.state_of(MATE) in ("FREE", "ENGAGED") and gd["stop"] == "distance", **rep["at_release"])

    with log.phase("head.release"):
        mo.gripper(w_open, GRASP_FORCE_N, what="let go of the head", max_s=1.0)
        n = int(round(LOCK_WAIT_S / mo.tick))
        state = None
        for i in range(0, n, 5):
            state = truth.state_of(MATE, truth.read())
            if state in ("LOCKED", "LOCKED_OFF", "VOID"):
                break
            mo.hold(0.05)
        rec = truth.record_of(MATE)
        e = truth.seat_errors(MATE)
        rep["state"] = state
        rep["seated"] = {"axial_mm": e["axial_m"] * 1e3, "lateral_mm": e["lateral_m"] * 1e3, "tilt_deg": math.degrees(e["tilt_rad"]),
                         "residual_deg": math.degrees(e["residual_rad"]), "k_near": e["k_near"]}
        rep["record"] = {k: rec.get(k) for k in ("state", "k", "design_ok", "t_lock_s", "t_holding_start_s", "lock_errors")} if rec else None
        log.write("lock", state=state, record=rep["record"], seated=rep["seated"])
        if k_offset == 0:
            log.check("the head is LOCKED at its design bolt position", state == "LOCKED", state=state)

    with log.phase("head.retreat"):
        T_cmd = kin.fk_T(mo.cmd_q())
        mo.move_linear(_up(T_cmd, 0.03), v=0.02, what="the open hand up off the head")
        mo.move_linear(_at_height(T_cmd, Z_TRANSIT_M), what="up to the transit height")
        mo.move_joint(home, what="home")
        st = s.state()
        log.check("the arm is at home", float(np.max(np.abs(np.array(st["robot"]["q"]) - home))) < 5e-3,
                  q=st["robot"]["q"])
    return rep


def _up_seat(truth, k_offset, h, st):
    """the part body's pose `h` above its design pose along the seat axis (the carrier as it stands now)"""
    x = truth.ifaces[MATE]
    Tc = truth.body_T(MATE, "carrier", st)
    Tz = np.eye(4)
    Tz[2, 3] = h
    return Tc @ x["S"] @ Tz @ tf.rotz(k_offset * x["pitch"]) @ tf.inv(x["F"])
