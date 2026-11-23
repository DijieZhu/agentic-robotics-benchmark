"""Reverse end-over-end roll ("roll D"): a quarter turn about the parcel's
downstream bottom edge, top towards downstream (world -x rotation):

    -y -> up -> +y -> down -> -y

so a label on the upstream face comes up in one roll.

  stop    the left (downstream) palm catches the parcel low: pads side by
          side across the belt at the downstream face's bottom edge, tool
          tilted down; the belt keeps the parcel pressed against it and the
          stop takes the belt drag and the push (so the parcel does not slide)
  push    the right (upstream) palm sets its pads on the upstream face near
          the top and drives the contact along the arc about the downstream
          bottom edge (closed loop on the measured tilt: the target leads by
          <= rd_lead_deg at <= rd_rate_deg_s; J^T F feedforward: a normal
          force into the face and a tangential force from the gravity moment
          about the pivot); the right palm's workspace pointing downstream
          reaches y = -0.15 .. +0.30
  leave   at min(balance - 8 deg, rd_stop_leave_deg) the stop leaves
          downstream and up, out of the downstream face's sweep
  brake   past the balance the pad holds its place along the face and stays
          lightly pressed (friction brake); let go at rd_end_deg; the parcel
          lands on its former downstream face and rides on

The fall goes with the belt (the landing face meets a belt moving the same
way), so the landing kick is smaller than in roll U.

Entry: the parcel caught by ``EndOver.catch(slot, low_stop=True)``.
Exit: released on the belt (or, when the roll did not commit, lowered back
and held by the stop: ``RollResult.held``).
"""

from __future__ import annotations

import math

import numpy as np

from .endover import DT, G, PAD_TIP, PAD_X, RollResult, SL, rot_x
from .paths import PlanFail, palm_R


def _tilt_d(R, R0):
    """Rotation of the parcel about world -x since the roll's start (rad,
    positive = top towards downstream)."""
    Rr = R @ R0.T
    return -math.atan2(Rr[2, 1], Rr[1, 1])


def pusher_pose_r(eo, c, th_tool, psi, yaw, q0, cap):
    """Right palm pose with its pad tips at c, the tool pointing downstream
    and tilted down by th_tool (capped)."""
    p = eo.p
    tt = float(np.clip(th_tool, 0.0, cap))
    for dpsi in (0.0, 0.035, -0.035, 0.07, -0.07):
        R = palm_R("Right", psi + dpsi, yaw, tt)
        tcp = c + R[:, 2] * PAD_TIP
        q, ok, ep = eo.ik_R("Right", tcp, R, q0, iters=60)
        if ok and p.belt_clear > 0 and eo.arm_belt_gap("Right", q) < p.belt_clear:
            ok = False
        if ok and np.abs(q - q0).max() < 0.25:
            return q, R, tcp
    return None, None, None


def roll_down(eo, slot):
    p = eo.p
    sq0 = p.sq_soft
    p.sq_soft = p.rd_sq_soft          # the stop carries the belt drag and the push
    try:
        return _roll_down(eo, slot)
    finally:
        p.sq_soft = sq0


def _roll_down(eo, slot):
    p = eo.p
    cell = eo.cell
    res = RollResult(t_start=cell.time)
    pos0, R0, (W, L, H), yaw_err, mass = eo.dims(slot)
    eo._R0 = R0
    com_b = np.asarray(cell.slots[slot].parcel.com_offset, float)
    mg = mass * G
    z_bot = pos0[2] - H / 2
    x_c = float(pos0[0])
    y_u, y_d = pos0[1] - L / 2, pos0[1] + L / 2
    piv_w0 = np.array([x_c, y_d, z_bot])
    e_b = R0.T @ (piv_w0 - pos0)
    com_w0 = pos0 + R0 @ com_b
    d = com_w0 - piv_w0
    th0 = math.atan2(-d[1], d[2])                     # balance tilt
    th_commit = th0 + math.radians(p.rd_commit_deg)
    th_leave = min(th0 - math.radians(p.rd_leave_before_deg), math.radians(p.rd_stop_leave_deg))
    th_end = math.radians(p.rd_end_deg)
    cap = math.radians(p.rd_tool_cap_deg)
    res.log.append(dict(W=W, L=L, H=H, th0=math.degrees(th0)))
    Fs = p.rd_stop_force + p.rd_stop_force_mg * mg
    f_stop = {"Left": np.array([0.0, -Fs, 0.0])}

    # ---- 1. the pusher: right palm behind the parcel, pads on the upstream
    #         face near the top
    # pad roll: pads stacked as steeply as the face allows (the right palm's
    # workspace pointing downstream is widest at psi 90 / 45 deg), the
    # upper pad just below the top edge
    psi = 0.0
    dz = 0.0
    for cand in p.rd_psi_list:
        dz_ = PAD_X * abs(math.sin(math.radians(cand)))
        if 2 * dz_ + 0.02 <= H:
            psi, dz = math.radians(cand), dz_
            break
    below = max(p.rd_push_below, p.rd_push_below_frac * H)
    z_c = z_bot + max(dz + 0.012, H - below - dz)
    x_p = float(np.clip(x_c, 0.215, 0.34))
    c_w0 = np.array([x_p, y_u, z_c])
    c_b = R0.T @ (c_w0 - pos0)
    Rr0 = palm_R("Right", psi, yaw_err, 0.0)
    res.log.append(dict(psi=math.degrees(psi), z_c=z_c - z_bot))
    q_pre = None
    for dy, dz in ((0.06, 0.03), (0.05, 0.05), (0.07, 0.02), (0.05, 0.08)):
        pre = c_w0 + Rr0[:, 2] * PAD_TIP + np.array([0.0, -dy, dz])
        q_pre = eo.glob_R("Right", pre, Rr0, eo.ex.q[SL["Right"]])
        if q_pre is not None:
            break
    if q_pre is None:
        raise PlanFail("roll D: push pre-pose unreachable")
    q_goal = eo.ex.q.copy()
    q_goal[SL["Right"]] = q_pre
    eo.joint_goal(q_goal, vmax=p.stop_vmax, tmin=p.stop_tmin, forces=f_stop)
    tcp_in = c_w0 + Rr0[:, 2] * (PAD_TIP + 0.002)
    eo.move_cart("Right", c_w0 + Rr0[:, 2] * PAD_TIP + np.array([0.0, -0.02, 0.0]), R1=Rr0,
                 T=0.3, forces=f_stop)
    Fn0 = p.rd_push_fn + p.rd_push_fn_mg * mg

    def f_in(i, n=12):
        a = min(1.0, i / n)
        return {"Left": f_stop["Left"], "Right": Rr0[:, 2] * Fn0 * a}
    eo.move_cart("Right", tcp_in, R1=Rr0, T=0.3, forces=f_in)
    eo.trace_row(slot, "rd_push_set")

    # ---- 2. the arc (closed loop on the tilt), stop leaving, brake, let go
    qL = eo.ex.q[SL["Left"]].copy()
    qR = eo.ex.q[SL["Right"]].copy()
    # the stop's pad tips (the left palm's), fixed in the parcel frame: the
    # ducking stop keeps them at the bottom of the turning downstream face
    pl0, Rl0 = cell.tcp_pose("Left")
    s_tip_b = R0.T @ ((pl0 - Rl0[:, 2] * PAD_TIP) - pos0)
    tilt_s0 = math.radians(p.rd_stop_tilt_deg)
    lead = math.radians(p.rd_lead_deg)
    rate = math.radians(p.rd_rate_deg_s) * DT
    th_t = _tilt_d(eo.body(slot)[1], R0)
    t0 = cell.time
    stop_on = True
    leave_from = None
    leave_i = 0
    brake_P = None
    n_fail = 0
    best = 0.0
    while cell.time - t0 < p.rd_timeout_s and not cell.done:
        if not cell.slots[slot].active:
            res.reason = "parcel gone"
            break
        pos, R, *_ = eo.body(slot)
        th_m = _tilt_d(R, R0)
        res.tilt_max_deg = max(res.tilt_max_deg, math.degrees(th_m))
        if th_m >= th_commit:
            res.committed = True
        if th_m >= th_end - math.radians(1.0):
            break
        past = th_m > th0 + math.radians(p.rd_free_deg)
        th_t = max(th_t, min(th_t + rate, th_m + lead, th_end))
        dth = 0.0 if past else float(np.clip(th_t - th_m, 0.0, lead))
        piv = pos + R @ e_b
        c_now = pos + R @ c_b
        n_out = R @ R0.T @ np.array([0.0, -1.0, 0.0])
        Rd = rot_x(-dth)
        c_tgt = piv + Rd @ (c_now - piv)
        n_in = -(Rd @ n_out)
        if past:
            if brake_P is None:
                pr_, Rr_ = cell.tcp_pose("Right")
                brake_P = pr_ - Rr_[:, 2] * PAD_TIP
            d_n = float((brake_P - c_now) @ n_out)
            c_tgt = brake_P - d_n * n_out
            if float(np.linalg.norm(c_tgt - c_now)) > 0.45 * H:
                c_tgt = c_now.copy()
        n_in[0] = 0.0
        n_in /= max(np.linalg.norm(n_in), 1e-9)
        if not past:
            c_tgt = c_tgt + n_in * p.rd_press
        c_tgt[0] = c_w0[0]
        th_tool = math.atan2(-n_in[2], n_in[1])
        qn, Rp, tcp_t = pusher_pose_r(eo, c_tgt, th_tool, psi, yaw_err, qR, cap)
        if qn is None:
            n_fail += 1
            qn = qR
            if th_m > best + math.radians(0.5):
                best = th_m
                n_fail = 0
            if n_fail > p.hold_ticks:
                res.reason = "push IK"
                break
        else:
            best = max(best, th_m)
        step = np.abs(qn - qR).max()
        if step > p.step_max:
            qn = qR + (qn - qR) * (p.step_max / step)
        qR = qn
        # feedforward: normal into the face + tangential along the arc from
        # the gravity moment about the pivot
        r = c_now - piv
        rr = max(math.hypot(r[1], r[2]), 1e-3)
        tan = np.array([0.0, r[2], -r[1]]) / rr
        dg = (pos + R @ com_b) - piv
        Mg = mg * max(0.0, -float(dg[1]))
        h_n = max(abs(float((c_now - piv) @ (R @ R0.T @ np.array([0.0, 0.0, 1.0])))), 0.02)
        M_need = Mg * (1.0 + p.moment_margin)
        Fn = float(np.clip(M_need * p.fn_share / h_n + p.fn_min, p.fn_min, Fn0))
        Ft = max(0.0, M_need - Fn * h_n) / rr
        Ft = min(Ft, p.mu_eff * Fn)
        if past:
            Fn, Ft = p.rd_brake_fn + 0.3 * mg, 0.0
        FR = n_in * Fn + tan * Ft
        # the stop: holds, then leaves downstream and up
        if stop_on and th_m >= th_leave:
            stop_on = False
            leave_from = cell.tcp_pose("Left")[0]
        if stop_on and p.rd_follow:
            # ducking stop: the pads stay at the bottom of the downstream face,
            # the tool turns down with the face
            tip = piv_w0 + rot_x(-max(0.0, th_m)) @ (pos0 + R0 @ s_tip_b - piv_w0)
            tt = min(tilt_s0 + p.rd_duck * max(0.0, th_m), math.radians(88.0))
            Rs = palm_R("Left", 0.0, 0.0, tt)
            qln, okl, epl = eo.ik_R("Left", tip + Rs[:, 2] * PAD_TIP, Rs, qL, iters=60,
                                    rot_tol=0.01, pos_tol=1.5e-3)
            if okl:
                stl = np.abs(qln - qL).max()
                if stl > p.step_max:
                    qln = qL + (qln - qL) * (p.step_max / stl)
                qL = qln
            FL = rot_x(-max(0.0, th_m)) @ f_stop["Left"]
        elif stop_on:
            FL = f_stop["Left"]
        else:
            leave_i += 1
            # out of the descending downstream face's sweep: downstream along
            # the belt first (the face comes down from above), then up
            u1 = min(1.0, leave_i / 10.0)
            u2 = min(1.0, max(0.0, (leave_i - 10) / 12.0))
            s1 = u1 * u1 * (3 - 2 * u1)
            s2 = u2 * u2 * (3 - 2 * u2)
            tgt = leave_from + np.array([0.0, p.rd_leave_dy, 0.0]) * s1 + np.array([0.0, 0.02, 0.08]) * s2
            qln, okl, epl = eo.ik_R("Left", tgt, cell.tcp_pose("Left")[1], qL, iters=40)
            stl = np.abs(qln - qL).max()
            if stl > p.step_max:
                qln = qL + (qln - qL) * (p.step_max / stl)
            qL = qln
            FL = None
        row = eo.ex.q.copy()
        row[SL["Left"]] = qL
        row[SL["Right"]] = qR
        eo.tick(row, {"Left": FL, "Right": FR})
        eo.trace_row(slot, "rd_arc" if not past else "rd_brake", th_t)
    res.log.append(dict(t=cell.time, phase="rd_arc_end", tilt=res.tilt_max_deg,
                        committed=res.committed))
    if not res.committed and cell.slots[slot].active and res.reason in ("", "push IK"):
        # lower back: the pusher eases off, the parcel falls back onto its
        # base against the stop (still in place below th_leave)
        res.reason = "no commit"
        pr_, Rr_ = cell.tcp_pose("Right")
        eo.move_cart("Right", pr_ - Rr_[:, 2] * 0.05 + np.array([0.0, 0.0, 0.03]), R1=Rr_, T=0.4,
                     forces=f_stop if stop_on else None)
        eo.hold(0.4, forces=f_stop if stop_on else None)
        res.held = bool(stop_on)
        res.t_end = cell.time
        res.label_after, res.label_err_deg = eo.label_now(slot)
        return res
    # ---- 3. let go: both palms away; the parcel lands and rides on
    pr_, Rr_ = cell.tcp_pose("Right")
    eo.move_cart("Right", pr_ - Rr_[:, 2] * 0.03 + np.array([0.0, 0.0, 0.05]), R1=Rr_, T=0.3)
    if stop_on:
        pl_, Rl_ = cell.tcp_pose("Left")
        eo.move_cart("Left", pl_ + np.array([0.0, 0.08, 0.07]), R1=Rl_, T=0.35)

    def settled():
        if not cell.slots[slot].active:
            return True
        ps = cell.truth.parcel_state(slot)
        return float(np.linalg.norm(ps.angvel)) < 0.15 and ps.on_belt and cell.time - res.t_start > 1.0
    eo.hold(2.0, check=settled)
    res.t_end = cell.time
    if cell.slots[slot].active:
        res.label_after, res.label_err_deg = eo.label_now(slot)
    res.peaks = dict(eo.peak)
    return res
