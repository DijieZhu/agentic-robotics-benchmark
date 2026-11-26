"""Block budgets (jobs.block_budget, jobs.spin_up_term): the functions that compute a block's budget from the session
paper's job list and published quantities. A runner may pass its own budget function to the World instead.

"""
import math

import numpy as np

from . import rot
from .config import V


def view_body_pose(c, a, u, mount):
    a = np.asarray(a, float); u = np.asarray(u, float)
    ez = -u
    ey = np.cross(ez, a)
    R = np.stack([a, ey, ez], 1)
    q = rot.qfrom_mat(R)
    return np.asarray(c, float) - R @ np.asarray(mount, float), q


def body_goal(job, mount):
    """the body pose that meets a job's goal (for a view: the pose the camera mounting gives)."""
    if job["kind"] == "view" or job.get("request_shape") == "view":
        return view_body_pose(job["goal"]["centre"], job["goal"]["axis"], job["goal"]["up"], mount)
    return np.asarray(job["goal"]["position"], float), np.asarray(job["goal"]["attitude"], float)


def trap(x, A, Vv):
    if x <= Vv * Vv / A:
        return 2.0 * math.sqrt(x / A)
    return x / Vv + Vv / A


def limits(mode):
    sp = V("plan.speed_limit." + mode)
    ac = V("plan.acceleration_limit")[mode]
    return sp["linear_m_s"], ac["linear_m_s2"], sp["angular_rad_s"], ac["angular_rad_s2"]


def spin_up(mode_from, mode_to):
    """jobs.spin_up_term: S = |n_new - n_old| / 200 + 2.0 s (the constants are read from the entry's text)."""
    txt = V("jobs.spin_up_term")
    assert "n = 2000, 2500, 2800 rpm for fan levels 1, 2, 3" in txt and "/ 200 + 2.0 s" in txt
    rpm = {1: 2000.0, 2: 2500.0, 3: 2800.0}
    lv = V("mode.fan_level")
    return abs(rpm[lv[mode_to]] - rpm[lv[mode_from]]) / 200.0 + 2.0


def fixed_term(job):
    txt = V("jobs.block_budget")
    assert "F_j = 2.5 s for a leg or slew" in txt and "1.7 s for a view" in txt
    if job["kind"] in ("leg", "slew") or (job["kind"] == "request" and job["request_shape"] in ("leg", "slew")):
        return 2.5
    if job["kind"] == "hold":
        return float(job["goal"]["duration_s"])
    if job["kind"] == "view" or job.get("request_shape") == "view":
        return float(V("jobs.capture_s"))
    return 2.5


def block_budget(block, k, mount):
    """B = S + sum over every announced job of (k t_j + F_j); returns B and the per-job (t_j, F_j)."""
    mode = block["mode"]
    prev_p, prev_q = np.asarray(block["start_pose"]["position"]), np.asarray(block["start_pose"]["attitude"])
    S = spin_up(block["mode_before"], mode) if block["mode_before"] != mode else 0.0
    terms = []
    total = S
    for job in block["jobs"]:
        jm = job["mode"] if job["mode"] in ("nominal", "quiet", "precision") else mode
        v, a, w, al = limits(jm)
        if job["kind"] == "hold":
            t = 0.0
        else:
            p, q = body_goal(job, mount)
            d = float(np.linalg.norm(p - prev_p))
            th = rot.qangle(prev_q, q)
            t = max(trap(d, a, v), trap(th, al, w))
        F = fixed_term(job)
        terms.append([t, F])
        total += k * t + F
        if job["kind"] != "hold":
            prev_p, prev_q = body_goal(job, mount)
    return total, terms, S
