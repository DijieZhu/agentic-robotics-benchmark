# Generated file (the task's solution): do not edit; regenerate it from the maintained source.
import math
import numpy as np
from . import rot

def trap_profile(x, A, V, T=None):
    if x <= 1e-12:
        return (0.0 if T is None else T, A, 0.0, 0.0)
    if T is None:
        if x <= V * V / A:
            t1 = math.sqrt(x / A)
            return (2 * t1, A, A * t1, t1)
        return (x / V + V / A, A, V, V / A)
    disc = A * A * T * T - 4 * A * x
    disc = max(disc, 0.0)
    vc = 0.5 * (A * T - math.sqrt(disc))
    t1 = vc / A
    return (T, A, vc, t1)

def trap_eval(prof, x, t):
    T, A, vc, t1 = prof
    if x <= 1e-12 or T <= 0:
        return (0.0, 0.0, 0.0)
    t = min(max(t, 0.0), T)
    t2 = T - t1
    eps = 1e-07
    if t < t1 - eps:
        return (0.5 * A * t * t, A * t, A)
    if t < t2 - eps:
        if t < t1:
            t = t1
        return (0.5 * A * t1 * t1 + vc * (t - t1), vc, 0.0)
    d = max(0.0, T - t)
    return (x - 0.5 * A * d * d, A * d, -A if t < T - eps else 0.0)

def _pose_move_once(p0, q0, p1, q1, lim, t0, accel_factor=1.0, pivot_body=None, dt_sp=0.25, T_min=0.0):
    v_l, a_l, w_l, al_l = lim
    a_l *= accel_factor
    al_l *= accel_factor
    p0 = np.asarray(p0, float)
    p1 = np.asarray(p1, float)
    q0 = rot.qnormalize(q0)
    q1 = rot.qnormalize(q1)
    if np.dot(q0, q1) < 0:
        q1 = -q1
    dq = rot.qmul(q1, rot.qconj(q0))
    rv = rot.qlog(dq)
    th = float(np.linalg.norm(rv))
    ax_w = rv / th if th > 1e-12 else np.array([1.0, 0.0, 0.0])
    if pivot_body is not None:
        m = np.asarray(pivot_body, float)
        c0 = p0 + rot.qmat(q0) @ m
        c1 = p1 + rot.qmat(q1) @ m
    else:
        m = np.zeros(3)
        c0, c1 = (p0, p1)
    dc = c1 - c0
    d = float(np.linalg.norm(dc))
    u = dc / d if d > 1e-12 else np.zeros(3)
    scale = 1.0
    for _ in range(60):
        Tt = trap_profile(d, a_l * scale, v_l)[0]
        Tr = trap_profile(th, al_l * scale, w_l)[0]
        T = max(Tt, Tr, T_min)
        pt = trap_profile(d, a_l * scale, v_l, T)
        pr = trap_profile(th, al_l * scale, w_l, T)
        if pivot_body is None or not np.any(m):
            break
        ra = rv_ = 0.0
        for tt in np.linspace(0, T, 401):
            s, sd, sdd = trap_eval(pt, d, tt)
            g, gd, gdd = trap_eval(pr, th, tt)
            R = rot.qmat(rot.qmul(rot.qexp(ax_w * g), q0))
            rm = R @ m
            wv = ax_w * gd
            al = ax_w * gdd
            vel = u * sd - np.cross(wv, rm)
            acc = u * sdd - np.cross(al, rm) - np.cross(wv, np.cross(wv, rm))
            ra = max(ra, np.linalg.norm(acc) / a_l)
            rv_ = max(rv_, np.linalg.norm(vel) / v_l)
        if ra <= 0.998 and rv_ <= 0.998:
            break
        scale *= min(0.999, 0.996 / max(ra, 1e-09), (0.996 / max(rv_, 1e-09)) ** 2)
    T = max(T, 1e-09)
    n = max(2, int(math.ceil(T / dt_sp)) + 1)
    ts = list(np.linspace(0.0, T, n))
    for b in (pt[3], T - pt[3], pr[3], T - pr[3]):
        if 1e-06 < b < T - 1e-06:
            ts.append(b)
    ts = sorted(set([round(x, 9) for x in ts]))
    sps = []
    for tt in ts:
        s, sd, sdd = trap_eval(pt, d, tt)
        g, gd, gdd = trap_eval(pr, th, tt)
        qq = rot.qnormalize(rot.qmul(rot.qexp(ax_w * g), q0))
        R = rot.qmat(qq)
        wv = ax_w * gd
        al = ax_w * gdd
        rm = R @ m
        c = c0 + u * s
        pos = c - rm
        vel = u * sd - np.cross(wv, rm)
        acc = u * sdd - np.cross(al, rm) - np.cross(wv, np.cross(wv, rm))
        sps.append((t0 + tt, pos, vel, acc, qq if qq[3] >= 0 else qq, R.T @ wv, R.T @ al))
    last = sps[-1]
    sps[-1] = (last[0], last[1], np.zeros(3), np.zeros(3), last[4], np.zeros(3), np.zeros(3))
    return sps

def ref_at(sps, t):
    if t <= sps[0][0]:
        s = sps[0]
        return (s[1], s[2], s[4], s[5])
    lo, hi = (0, len(sps) - 1)
    if t >= sps[-1][0]:
        s = sps[-1]
        return (s[1], np.zeros(3), s[4], np.zeros(3))
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if sps[mid][0] <= t:
            lo = mid
        else:
            hi = mid
    s = sps[lo]
    dt = t - s[0]
    p = s[1] + s[2] * dt + 0.5 * s[3] * dt * dt
    v = s[2] + s[3] * dt
    q = rot.qnormalize(rot.qmul(s[4], rot.qexp(s[5] * dt + 0.5 * s[6] * dt * dt)))
    w = s[5] + s[6] * dt
    return (p, v, q, w)

def check_plan(sps, lim, start_pose, goal_pose, kind, rules):
    v_l, a_l, w_l, al_l = lim
    rt = 1e-06
    if not sps or len(sps) > rules['max_setpoints']:
        return 'max_setpoints'
    for s in sps:
        if not all((np.all(np.isfinite(np.asarray(x, float))) for x in s[1:])) or not math.isfinite(s[0]):
            return 'non_finite'
        if np.linalg.norm(s[2]) > v_l * (1 + rt) or np.linalg.norm(s[3]) > a_l * (1 + rt) or np.linalg.norm(s[5]) > w_l * (1 + rt) or (np.linalg.norm(s[6]) > al_l * (1 + rt)):
            return 'limit_rule'
    tol_p, tol_a = rules['consistency']
    for a, b in zip(sps[:-1], sps[1:]):
        dt = b[0] - a[0]
        if dt <= 0 or dt > rules['max_gap_s'] + 1e-12:
            return 'consistency_times'
        p = a[1] + a[2] * dt + 0.5 * a[3] * dt * dt
        v = a[2] + a[3] * dt
        q = rot.qmul(a[4], rot.qexp(a[5] * dt + 0.5 * a[6] * dt * dt))
        w = a[5] + a[6] * dt
        if np.linalg.norm(p - b[1]) > tol_p[0] or rot.qangle(q, b[4]) > tol_a[0] or np.linalg.norm(v - b[2]) > tol_p[1] or (np.linalg.norm(w - b[5]) > tol_a[1]):
            return 'consistency'
    sp_p, sp_a = rules['start_tol']
    if np.linalg.norm(sps[0][1] - start_pose[0]) > sp_p or rot.qangle(sps[0][4], start_pose[1]) > sp_a:
        return 'start_tolerance'
    e = sps[-1]
    if np.any(np.abs(e[2]) > 1e-12) or np.any(np.abs(e[3]) > 1e-12) or np.any(np.abs(e[5]) > 1e-12) or np.any(np.abs(e[6]) > 1e-12):
        return 'end_tolerance_rates'
    if kind in ('leg', 'slew'):
        ep, ea = rules['end_tol']
        if np.linalg.norm(e[1] - goal_pose[0]) > ep or rot.qangle(e[4], goal_pose[1]) > ea:
            return 'end_tolerance'
    return None
PLAN_RULE_KEYS = {'max_setpoints': 'plan.max_setpoints', 'limit_rule': 'plan.limit_rule', 'consistency_times': 'plan.consistency', 'consistency': 'plan.consistency', 'start_tolerance': 'plan.start_tolerance', 'end_tolerance_rates': 'plan.end_tolerance', 'end_tolerance': 'plan.end_tolerance'}
ZERO_NORM = 1e-12

def normalise_plan(sps):
    out = []
    for s in sps:
        try:
            if len(s) != 7 or not math.isfinite(float(s[0])):
                return (None, 'malformed setpoint')
            parts = [np.asarray(x, float) for x in s[1:]]
        except (TypeError, ValueError):
            return (None, 'malformed setpoint')
        if [p.shape for p in parts] != [(3,), (3,), (3,), (4,), (3,), (3,)]:
            return (None, 'malformed setpoint')
        if not all((np.all(np.isfinite(p)) for p in parts)):
            return (None, 'a number that is not finite')
        n = float(np.linalg.norm(parts[3]))
        if n <= ZERO_NORM:
            return (None, 'an attitude of norm zero')
        out.append((float(s[0]), parts[0], parts[1], parts[2], parts[3] / n, parts[4], parts[5]))
    return (out, None)

def check_plan_keyed(sps, lim, start_pose, goal_pose, kind, rules):
    v_l, a_l, w_l, al_l = lim
    rt = 1e-06
    if not sps or len(sps) > rules['max_setpoints']:
        return ('plan.max_setpoints', None, 'max_setpoints')
    for i, s in enumerate(sps):
        if np.linalg.norm(s[2]) > v_l * (1 + rt) or np.linalg.norm(s[3]) > a_l * (1 + rt) or np.linalg.norm(s[5]) > w_l * (1 + rt) or (np.linalg.norm(s[6]) > al_l * (1 + rt)):
            return ('plan.limit_rule', i, 'limit_rule')
    tol_p, tol_a = rules['consistency']
    for i, (a, b) in enumerate(zip(sps[:-1], sps[1:])):
        dt = b[0] - a[0]
        if dt <= 0 or dt > rules['max_gap_s'] + 1e-12:
            return ('plan.consistency', i + 1, 'consistency_times')
        p = a[1] + a[2] * dt + 0.5 * a[3] * dt * dt
        v = a[2] + a[3] * dt
        q = rot.qmul(a[4], rot.qexp(a[5] * dt + 0.5 * a[6] * dt * dt))
        w = a[5] + a[6] * dt
        if np.linalg.norm(p - b[1]) > tol_p[0] or rot.qangle(q, b[4]) > tol_a[0] or np.linalg.norm(v - b[2]) > tol_p[1] or (np.linalg.norm(w - b[5]) > tol_a[1]):
            return ('plan.consistency', i + 1, 'consistency')
    sp_p, sp_a = rules['start_tol']
    if np.linalg.norm(sps[0][1] - start_pose[0]) > sp_p or rot.qangle(sps[0][4], start_pose[1]) > sp_a:
        return ('plan.start_tolerance', 0, 'start_tolerance')
    e = sps[-1]
    if np.any(np.abs(e[2]) > ZERO_NORM) or np.any(np.abs(e[3]) > ZERO_NORM) or np.any(np.abs(e[5]) > ZERO_NORM) or np.any(np.abs(e[6]) > ZERO_NORM):
        return ('plan.end_tolerance', len(sps) - 1, 'end_tolerance_rates')
    if kind in ('leg', 'slew'):
        ep, ea = rules['end_tol']
        if np.linalg.norm(e[1] - goal_pose[0]) > ep or rot.qangle(e[4], goal_pose[1]) > ea:
            return ('plan.end_tolerance', len(sps) - 1, 'end_tolerance')
    return None

def pose_move(p0, q0, p1, q1, lim, t0, accel_factor=1.0, pivot_body=None, dt_sp=0.25, T_min=0.0):
    v_l, a_l, w_l, al_l = lim
    f = accel_factor
    for _ in range(40):
        sps = _pose_move_once(p0, q0, p1, q1, lim, t0, f, pivot_body, dt_sp, T_min)
        r = 0.0
        for s_ in sps:
            r = max(r, np.linalg.norm(s_[2]) / v_l, np.linalg.norm(s_[3]) / (a_l * accel_factor), np.linalg.norm(s_[5]) / w_l, np.linalg.norm(s_[6]) / (al_l * accel_factor))
        if r <= 1.0 - 1e-07:
            return sps
        f *= min(0.995, 0.999 / r)
    return sps
