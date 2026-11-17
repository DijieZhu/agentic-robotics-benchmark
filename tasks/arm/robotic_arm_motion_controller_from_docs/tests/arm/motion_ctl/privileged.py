"""The privileged tick trajectory of a job (the oracle's "truth" payload; grader side, never documented to the agent).

For a feasible job the grader turns its privileged plan (`jobs.generate`: joint waypoints of the TRUE unit, rigid, with
the clearance checked; for a path the warm-started IK samples along it) into the exact bus commands a controller that
knew the unit would send:

  timing        reach: every waypoint segment as a cosine blend at SPEED_FRAC of the rated speed (segments >= 0.15 s);
                path: the approach plan, a DWELL_S rest at the start, then the path samples at their times
  compensation  the link angle q is reached by commanding the output angle q - g(q) / k (the series elasticity's static
                deflection under the gravity load g of the true unit with its tool) and the position loop's static
                error g / kp is added (kp = 998.22 x the unit's gain)
  latency       each cycle commands the setpoint of time t + latency + one cycle, so the delayed bus follows the plan
  flags         `path_start` at the cycle whose end is the path's start time, `done` DONE_S after the last setpoint

Returns {"ticks": [[6] per cycle], "path_start_cycle": k or None, "done_cycle": k, "jaw_q": ...}.
"""

import math

import numpy as np

from . import plant as P
from . import unit as U

SPEED_FRAC = 0.55
SEG_MIN_S = 0.15
DWELL_S = 0.3
DONE_S = 0.35
KP = 998.22


def _blend(qa, qb, T, dt=P.CYCLE_S):
    n = max(1, int(math.ceil(T / dt)))
    out = []
    for k in range(1, n + 1):
        s = 0.5 - 0.5 * math.cos(math.pi * k / n)
        out.append(qa + (qb - qa) * s)
    return out


def joint_timeline(waypoints, v_rated=6.49 * 0.85):
    """Setpoints every cycle along joint waypoints (cosine blends; the slowest unit's speed x SPEED_FRAC)."""
    W = [np.asarray(w, float) for w in waypoints]
    seq = [W[0].copy()]
    v = SPEED_FRAC * v_rated
    for a, b in zip(W[:-1], W[1:]):
        T = max(SEG_MIN_S, float(np.max(np.abs(b[:5] - a[:5]))) * math.pi / (2.0 * v))
        seq += _blend(a, b, T)
    return seq


def compensated_ticks(kin, spec, q_link_seq):
    """Bus goal ticks for link angles: output = q - g/k, plus the loop's static error g/kp; the TRUE map."""
    tc = spec["C"]["true"]
    sign = np.array([tc[j]["sign"] for j in U.JOINTS], float)
    zero = np.array([tc[j]["zero_rad"] for j in U.JOINTS], float)
    k = np.array([float(spec["D"]["compliance_nm_per_rad"][j]) for j in U.ARM_JOINTS])
    kp = KP * np.array([float(spec["U"]["gain"][j]) for j in U.JOINTS])
    rec = spec["record"]["servos"]
    tmin = np.array([rec[j]["min_ticks"] for j in U.JOINTS])
    tmax = np.array([rec[j]["max_ticks"] for j in U.JOINTS])
    out = []
    mj = kin.mj
    for q in q_link_seq:
        q = np.asarray(q, float)
        with kin.rigid():
            kin._set(q)
        mj.mj_comPos(kin.m, kin.d)
        mj.mj_rne(kin.m, kin.d, 0, kin.d.qfrc_bias)
        g = -kin.d.qfrc_bias[kin.slack_dadr]                   # gravity load on each arm joint (N m, along +q)
        q_out = q.copy()
        q_out[:5] = q[:5] - g / k
        ctrl = q_out.copy()
        ctrl[:5] = q_out[:5] - g / kp[:5]                      # at rest kp (ctrl - q_out) = bias = -g: lead the setpoint by it
        b = sign * (ctrl - zero)
        t = np.clip(np.rint(b / U.TICK_RAD + U.MID).astype(int), tmin, tmax)
        out.append([int(x) for x in t])
    return out


ACC_FRAC = 0.9                    # of the servo's default 40 rad/s^2
INSERT_FRAC = 0.2                 # the last segment of a reach (pre-entry -> target) at this fraction of the rated speed


def _frac(i, slow):
    """Speed fraction of segment i (1-based): INSERT_FRAC for the plan's straight exit / entry segments, else SPEED_FRAC."""
    if slow is not None and 0 <= i - 1 < len(slow) and slow[i - 1]:
        return INSERT_FRAC
    return SPEED_FRAC


def ptp_duration(waypoints, slow=None, v_rated=6.49 * 0.85):
    """Seconds the synchronised PTP moves of `_ptp_segments` take along the waypoints (no compensation needed)."""
    W = [np.asarray(w, float) for w in waypoints]
    A = ACC_FRAC * 40.0
    T = 0.0
    for i in range(1, len(W)):
        dmax = max(1e-6, float(np.max(np.abs(W[i][:5] - W[i - 1][:5]))))
        V = _frac(i, slow) * v_rated
        T += 2.0 * math.sqrt(dmax / A) if dmax < V * V / A else dmax / V + V / A
        T += 2 * P.CYCLE_S
    return T


def _ptp_segments(waypoints, spec, kin, jaw_q, v_rated=6.49 * 0.85, slow=None):
    """Synchronised point-to-point moves along joint waypoints: per segment the compensated end ticks, per-joint speed
    and acceleration registers proportional to each joint's travel (so the servos' trapezoids stay in step and the arm
    follows the straight joint-space segment), and the segment's duration. Returns [(ticks, speed, acc, n_cycles)]."""
    W = [np.asarray(w, float).copy() for w in waypoints]
    for w in W:
        if jaw_q is not None:
            w[5] = jaw_q
    ticks = compensated_ticks(kin, spec, W)
    vmin = v_rated
    A = ACC_FRAC * 40.0
    out = []
    for i in range(1, len(W)):
        d = np.abs(W[i][:6] - W[i - 1][:6])
        dmax = max(1e-6, float(np.max(d[:5])))
        V = _frac(i, slow) * vmin
        if dmax < V * V / A:                                 # a triangle profile
            T = 2.0 * math.sqrt(dmax / A)
            V = math.sqrt(dmax * A)
        else:
            T = dmax / V + V / A
        sc = d / dmax
        spd = [max(1, int(round(float(sc[j]) * V / U.TICK_RAD))) for j in range(6)]
        acc = [max(1, int(round(float(sc[j]) * A / U.TICK_RAD / 100.0))) for j in range(6)]
        out.append((ticks[i], spd, acc, int(math.ceil(T / P.CYCLE_S)) + 2))
    return out


def trajectory(kin, spec, job, jaw_q=None):
    """The privileged command sequence of a feasible job: synchronised PTP segments along the privileged plan; for a
    path then a dwell and the path streamed at 10 ms with the latency lead. Per cycle: (ticks, speed or None, acc or
    None)."""
    lat = float(spec["D"]["latency_ms"]) * 1e-3
    lead = int(round(lat / P.CYCLE_S)) + 1
    cmds = []
    segs = _ptp_segments(job["plan_true"], spec, kin, jaw_q, slow=job.get("plan_slow"))
    for t, spd, acc, n in segs:
        cmds.append((t, spd, acc))
        cmds += [None] * (n - 1)
    cmds += [None] * (int(round(lat / P.CYCLE_S)) + 3)          # the latency and the last profile's tail
    path_start = None
    if job["type"] == "path":
        cmds += [None] * int(round(DWELL_S / P.CYCLE_S))
        qs = [np.asarray(q, float) for q in job["q_true"]]
        ts = np.asarray(job["path"]["t"], float)
        idx = np.linspace(0, len(ts) - 1, len(qs))
        t_s = np.interp(idx, np.arange(len(ts)), ts)
        tail = []
        for tt in np.arange(0.0, ts[-1] + 1e-9, P.CYCLE_S):
            q = np.array([np.interp(tt, t_s, [qq[j] for qq in qs]) for j in range(6)])
            if jaw_q is not None:
                q[5] = jaw_q
            tail.append(q)
        tk = compensated_ticks(kin, spec, tail)
        # the reference starts at the end of the path_start cycle; the setpoint for path time t must be on the bus
        # `lead` cycles earlier
        k0 = len(cmds) - lead
        for i, tki in enumerate(tk):
            c = k0 + i
            item = (tki, None, None)
            if c < len(cmds):
                cmds[c] = item
            else:
                cmds.append(item)
        path_start = k0 + lead - 1
        cmds += [None] * (lead + 2)
    n_done = len(cmds) + int(round(DONE_S / P.CYCLE_S))
    cmds += [None] * (n_done - len(cmds))
    ticks = [c[0] if c else None for c in cmds]
    speed = [c[1] if c else None for c in cmds]
    acc = [c[2] if c else None for c in cmds]
    return {"ticks": ticks, "speed": speed, "acc": acc, "done_cycle": n_done - 1, "path_start_cycle": path_start}
