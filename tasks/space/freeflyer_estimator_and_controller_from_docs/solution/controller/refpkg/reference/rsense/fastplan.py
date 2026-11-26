# Generated file (the task's solution): do not edit; regenerate it from the maintained source.
import math
import numpy as np
from world import rot
from world.traj import trap_profile, trap_eval

def _trap_eval_v(prof, x, t):
    T, A, vc, t1 = prof
    t = np.asarray(t, float)
    if x <= 1e-12 or T <= 0:
        z = np.zeros_like(t)
        return (z, z.copy(), z.copy())
    t = np.minimum(np.maximum(t, 0.0), T)
    t2 = T - t1
    eps = 1e-07
    s = np.empty_like(t)
    sd = np.empty_like(t)
    sdd = np.empty_like(t)
    m1 = t < t1 - eps
    m2 = ~m1 & (t < t2 - eps)
    m3 = ~(m1 | m2)
    s[m1] = 0.5 * A * t[m1] ** 2
    sd[m1] = A * t[m1]
    sdd[m1] = A
    tt = np.where(t[m2] < t1, t1, t[m2])
    s[m2] = 0.5 * A * t1 * t1 + vc * (tt - t1)
    sd[m2] = vc
    sdd[m2] = 0.0
    d = np.maximum(0.0, T - t[m3])
    s[m3] = x - 0.5 * A * d * d
    sd[m3] = A * d
    sdd[m3] = np.where(t[m3] < T - eps, -A, 0.0)
    return (s, sd, sdd)

def _grid_ratios(pt, pr, d, th, T, ax_w, q0, u, m, a_l, v_l):
    tt = np.linspace(0, T, 401)
    s, sd, sdd = _trap_eval_v(pt, d, tt)
    g, gd, gdd = _trap_eval_v(pr, th, tt)
    R0 = rot.qmat(q0)
    m0 = R0 @ m
    K = rot.skew(ax_w)
    Km0 = K @ m0
    KKm0 = K @ Km0
    rm = m0[None, :] + np.sin(g)[:, None] * Km0[None, :] + (1.0 - np.cos(g))[:, None] * KKm0[None, :]
    wv = gd[:, None] * ax_w[None, :]
    al = gdd[:, None] * ax_w[None, :]
    vel = u[None, :] * sd[:, None] - np.cross(wv, rm)
    acc = u[None, :] * sdd[:, None] - np.cross(al, rm) - np.cross(wv, np.cross(wv, rm))
    ra = float(np.max(np.linalg.norm(acc, axis=1)) / a_l)
    rv_ = float(np.max(np.linalg.norm(vel, axis=1)) / v_l)
    return (ra, rv_)

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
        ra, rv_ = _grid_ratios(pt, pr, d, th, T, ax_w, q0, u, m, a_l, v_l)
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

def duration(p0, q0, p1, q1, lim, accel_factor=1.0, pivot_body=None):
    v_l, a_l, w_l, al_l = lim
    a_l *= accel_factor
    al_l *= accel_factor
    p0 = np.asarray(p0, float)
    p1 = np.asarray(p1, float)
    q0 = rot.qnormalize(q0)
    q1 = rot.qnormalize(q1)
    if np.dot(q0, q1) < 0:
        q1 = -q1
    rv = rot.qlog(rot.qmul(q1, rot.qconj(q0)))
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
    T = 0.0
    for _ in range(60):
        T = max(trap_profile(d, a_l * scale, v_l)[0], trap_profile(th, al_l * scale, w_l)[0])
        if pivot_body is None or not np.any(m):
            break
        pt = trap_profile(d, a_l * scale, v_l, T)
        pr = trap_profile(th, al_l * scale, w_l, T)
        ra, rv_ = _grid_ratios(pt, pr, d, th, T, ax_w, q0, u, m, a_l, v_l)
        if ra <= 0.998 and rv_ <= 0.998:
            break
        scale *= min(0.999, 0.996 / max(ra, 1e-09), (0.996 / max(rv_, 1e-09)) ** 2)
    return T

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
