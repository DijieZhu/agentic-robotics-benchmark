"""Motion primitives on the facade: a pure-pursuit path follower, turning in place, stopping, and the square-approach
crossing of a traversal element. The pose source is a function: Truth.pose for the reference, the odometry for the
unprivileged baselines. Time passes only through robot.step().

Blocked reports (the controller's stumble stop, or the world's stand-in for the classical controller) follow the
robot manual (capability.json, notes for programs): a 'blocked' can be false at rated sizes, so the follower
retries up to twice in the same mode (it stands still RETRY_PAUSE_S while the controller backs off, then commands the
same path again); at the third report it stops and returns with last_status 'blocked', and the caller changes the
mode or re-plans.

Version 2: for 5 s after a 'blocked' the robot takes no walking command (walk() answers ok false, held true).
The follower keeps sending its command and goes on when it is taken; its own stall window (stall=...) counts only
the time in which the robot may walk, so the hold is never taken for a stall.
"""

from __future__ import annotations

import math

import numpy as np

SPEED = {"walk": 0.35, "high_step": 0.15, "crouch_walk": 0.13}
YAW_RATE = {"walk": 1.0, "high_step": 0.5, "crouch_walk": 0.5}
MAX_BLOCKED = 3          # the third 'blocked' report of one attempt ends it (two retries in the same mode)
RETRY_PAUSE_S = 1.0      # stand still this long after a 'blocked' before the retry


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class Halt(Exception):
    """The mission is over (done) while a primitive was running."""


def refusable(fn, *args, **kw):
    """fn(*args, **kw), or None when the robot refuses the call (a CellError / CallError carrying a `code`: a used-up
    budget, the end of the mission); any other exception propagates."""
    try:
        return fn(*args, **kw)
    except Exception as e:  # noqa: BLE001
        if getattr(e, "code", None) is not None:
            return None
        raise


class Driver:
    def __init__(self, robot, pose_fn, log=None, tick=5, every=None):
        """every: optional (period_s, callback) called while time passes (e.g. a lidar sweep each second)."""
        self.r = robot
        self.pose = pose_fn
        self.tick = int(tick)
        self.events = []
        self.blocked = 0
        self.log = log or (lambda *a, **k: None)
        self.mode = "stand"
        self.every = every
        self._next_cb = 0.0
        self.last_status = "ok"          # result of the last follow(): ok | blocked | timeout
        self.retries = 0                 # retries after a 'blocked' report, over the whole mission

    # ------------------------------------------------------------------ basics
    def step(self, n=None):
        out = self.r.step(self.tick if n is None else n)
        for e in out["events"]:
            self.events.append(e)
            if e["type"] == "blocked":
                self.blocked += 1
        if out["done"]:
            raise Halt()
        if self.every is not None and out["sim_time"] >= self._next_cb:
            self._next_cb = out["sim_time"] + float(self.every[0])
            self.every[1]()
        return out

    def wait(self, seconds):
        n = int(round(seconds / 0.01))
        while n > 0:
            k = min(n, 100)
            self.step(k)
            n -= k

    def set_mode(self, m, settle=True):
        if m != self.mode:
            self.r.walk(0.0, 0.0, 0.0)
            if settle:
                self.wait(0.4)
            self.r.set_mode(m)
            self.mode = m
            if settle:
                self.wait(0.6 if m in ("crouch_walk", "high_step", "walk") else 0.8)

    def stop(self, settle=0.6):
        self.r.walk(0.0, 0.0, 0.0)
        self.wait(settle)
        # wait until the base is still (2 cm / 0.3 s), at most 2 s more
        p0 = self.pose()
        for _ in range(10):
            self.wait(0.2)
            p1 = self.pose()
            if math.hypot(p1["x"] - p0["x"], p1["y"] - p0["y"]) < 0.01 and abs(wrap(p1["yaw"] - p0["yaw"])) < 0.01:
                break
            p0 = p1

    # ------------------------------------------------------------------ motion
    def turn_to(self, yaw, tol=0.06, mode=None, max_s=20.0):
        mode = mode or (self.mode if self.mode in SPEED else "walk")
        if self.mode != mode:
            self.set_mode(mode)
        t = 0.0
        wmax = YAW_RATE[mode]
        while t < max_s:
            p = self.pose()
            e = wrap(yaw - p["yaw"])
            if abs(e) < tol:
                break
            wz = float(np.clip(1.6 * e, -wmax, wmax))
            if abs(wz) < 0.15:
                wz = math.copysign(0.15, wz)
            self.r.walk(0.0, 0.0, wz)
            self.step()
            t += self.tick * 0.01
        self.r.walk(0.0, 0.0, 0.0)

    def follow(self, pts, mode="walk", speed=None, tol=0.10, lookahead=0.35, final_slow=0.5, max_s=None,
               lateral_gain=0.8, heading_hold=None, stop_at_end=True, stop_on_blocked=False, max_blocked=MAX_BLOCKED,
               stall=None):
        """Follow the polyline pts (N, 2) in `mode`; returns True when the end is reached within tol. False on
        timeout (last_status 'timeout'), or when the attempt ends blocked (last_status 'blocked'): at the first
        'blocked' report with stop_on_blocked, else at the max_blocked-th (each earlier one is retried in the same
        mode after RETRY_PAUSE_S). stall=(window_s, min_m): give up (last_status 'stalled') when the progress along
        the path over the last window_s is below min_m (with a true pose source this sees slipping feet, which the
        controller's stumble stop does not; with odometry it cannot)."""
        nb0 = self.blocked
        handled = 0
        self.last_status = "timeout"
        prog_hist = []
        pts = np.asarray(pts, float).reshape(-1, 2)
        if len(pts) == 1:
            p = self.pose()
            pts = np.vstack([[p["x"], p["y"]], pts])
        if self.mode != mode:
            self.set_mode(mode)
        v_nom = SPEED[mode] if speed is None else float(speed)
        wmax = YAW_RATE[mode]
        seg = np.diff(pts, axis=0)
        L = np.linalg.norm(seg, axis=1)
        cum = np.concatenate([[0.0], np.cumsum(L)])
        total = float(cum[-1])
        max_s = max_s if max_s is not None else 30.0 + 4.0 * total / max(v_nom, 0.05)
        s_prog = 0.0
        t = 0.0
        ok = False
        while t < max_s:
            nb = self.blocked - nb0
            if nb > handled:
                handled = nb
                if stop_on_blocked or nb >= max_blocked:
                    self.last_status = "blocked"
                    self.log("follow_blocked", mode=mode, reports=nb)
                    break
                # retry in the same mode (the manual): stand still while the controller backs off, then go on
                self.retries += 1
                self.log("blocked_retry", mode=mode, report=nb)
                self.r.walk(0.0, 0.0, 0.0)
                self.wait(RETRY_PAUSE_S)
                t += RETRY_PAUSE_S
                continue
            p = self.pose()
            xy = np.array([p["x"], p["y"]])
            # projection on the polyline, not going backwards by more than 0.3 m
            best = (1e9, s_prog)
            for i in range(len(seg)):
                if L[i] < 1e-9:
                    continue
                u = float(np.clip(np.dot(xy - pts[i], seg[i]) / (L[i] ** 2), 0.0, 1.0))
                q = pts[i] + u * seg[i]
                s = cum[i] + u * L[i]
                if s < s_prog - 0.3:
                    continue
                d = float(np.linalg.norm(xy - q))
                if d < best[0]:
                    best = (d, s)
            s_prog = max(s_prog, best[1])
            if stall is not None:
                prog_hist.append((t, s_prog))
                while len(prog_hist) > 1 and t - prog_hist[0][0] > float(stall[0]):
                    prog_hist.pop(0)
                if t >= float(stall[0]) and t - prog_hist[0][0] >= float(stall[0]) - 0.06 and \
                        s_prog - prog_hist[0][1] < float(stall[1]):
                    self.last_status = "stalled"
                    self.log("follow_stalled", mode=mode, progress=round(s_prog, 3))
                    break
            remain = total - s_prog
            dend = float(np.linalg.norm(xy - pts[-1]))
            if dend < tol or (remain < tol and dend < 2 * tol):
                ok = True
                self.last_status = "ok"
                break
            if heading_hold is not None:
                # with the heading held the robot only walks forward: once the end is not ahead of it along the held
                # heading and it is near the end's line, the end is reached; else it walks on, away from the end,
                # until the timeout (practice site t1: 1.06 m beyond the far side of a floor pipe)
                hv = np.array([math.cos(heading_hold), math.sin(heading_hold)])
                rel = pts[-1] - xy
                lat_end = float(hv[0] * rel[1] - hv[1] * rel[0])
                if float(np.dot(rel, hv)) <= 0.0 and abs(lat_end) < 0.25:
                    ok = True
                    self.last_status = "ok"
                    self.log("follow_past_end", mode=mode, lateral=round(lat_end, 3))
                    break
            s_t = min(total, s_prog + lookahead)
            i = int(np.clip(np.searchsorted(cum, s_t) - 1, 0, len(seg) - 1))
            u = (s_t - cum[i]) / max(L[i], 1e-9)
            tgt = pts[i] + np.clip(u, 0, 1) * seg[i]
            if remain < lookahead:
                tgt = pts[-1]
            dx, dy = tgt - xy
            want = math.atan2(dy, dx) if heading_hold is None else heading_hold
            e = wrap(want - p["yaw"])
            c, s_ = math.cos(p["yaw"]), math.sin(p["yaw"])
            # lateral error toward the target in the body frame
            lat = -s_ * dx + c * dy
            v = v_nom * min(1.0, max(0.25, remain / max(final_slow, 1e-3)))
            if abs(e) > 0.7:
                vx, vy = 0.0, 0.0
            else:
                vx = v * max(0.0, math.cos(e)) ** 2
                vy = float(np.clip(lateral_gain * lat, -0.4 * v_nom, 0.4 * v_nom)) if heading_hold is not None or \
                    abs(e) < 0.3 else 0.0
            wz = float(np.clip(2.0 * e, -wmax, wmax))
            rep = self.r.walk(vx, vy, wz)
            if isinstance(rep, dict) and rep.get("held"):
                prog_hist = []                  # the hold after a 'blocked': not a stall
            self.step()
            t += self.tick * 0.01
        if stop_at_end:
            self.r.walk(0.0, 0.0, 0.0)
        return ok

    def goto(self, x, y, mode="walk", tol=0.08, speed=None, yaw=None):
        ok = self.follow([[x, y]], mode=mode, tol=tol, speed=speed)
        if yaw is not None:
            self.stop(0.3)
            self.turn_to(yaw)
        return ok


# ---------------------------------------------------------------------------------------------- elements
def near_polygon(poly, q, r):
    """True when point q is inside the polygon grown by r / 2 (matplotlib's Path.contains_point(q, radius=+-r):
    the path stroked with width r, mitred corners): for a convex polygon every edge's outward signed distance is at
    most r / 2; otherwise inside (even-odd) or within r / 2 of the boundary."""
    P = np.asarray(poly, float).reshape(-1, 2)
    n = len(P)
    x, y = float(q[0]), float(q[1])
    m = 0.5 * abs(float(r))
    area = 0.5 * sum(P[i, 0] * P[(i + 1) % n, 1] - P[(i + 1) % n, 0] * P[i, 1] for i in range(n))
    cross = [(P[(i + 1) % n, 0] - P[i, 0]) * (P[(i + 2) % n, 1] - P[(i + 1) % n, 1]) -
             (P[(i + 1) % n, 1] - P[i, 1]) * (P[(i + 2) % n, 0] - P[(i + 1) % n, 0]) for i in range(n)]
    convex = abs(area) > 1e-12 and (all(c >= -1e-12 for c in cross) or all(c <= 1e-12 for c in cross))
    if convex:
        s = 1.0 if area > 0 else -1.0              # counter-clockwise: the outward normal is (dy, -dx)
        for i in range(n):
            x1, y1 = P[i]
            x2, y2 = P[(i + 1) % n]
            L = math.hypot(x2 - x1, y2 - y1)
            if L <= 1e-12:
                continue
            out = s * ((x - x1) * (y2 - y1) - (y - y1) * (x2 - x1)) / L
            if out > m:
                return False
        return True
    inside = False
    dmin = float("inf")
    for i in range(n):
        x1, y1 = P[i]
        x2, y2 = P[(i + 1) % n]
        if (y1 > y) != (y2 > y):
            if x < x1 + (y - y1) * (x2 - x1) / (y2 - y1):
                inside = not inside
        dx, dy = x2 - x1, y2 - y1
        L2 = dx * dx + dy * dy
        t = 0.0 if L2 <= 0 else max(0.0, min(1.0, ((x - x1) * dx + (y - y1) * dy) / L2))
        dmin = min(dmin, math.hypot(x - (x1 + t * dx), y - (y1 + t * dy)))
    return inside or dmin <= m


def element_axes(footprint):
    """(centre, long unit axis u, normal n, half length, half depth) of an element footprint polygon."""
    P = np.asarray(footprint, float).reshape(-1, 2)
    c = P.mean(0)
    C = np.cov((P - c).T)
    ev, evec = np.linalg.eigh(C)
    u = evec[:, 1]
    n = np.array([-u[1], u[0]])
    hu = float(np.max(np.abs((P - c) @ u)))
    hn = float(np.max(np.abs((P - c) @ n)))
    return c, u, n, hu, hn


def crossing_points(footprint, p_from, p_to, before=0.50, after=0.55, end_margin=0.30):
    """Approach point A and exit point B of a square crossing of the element between p_from and p_to."""
    c, u, n, hu, hn = element_axes(footprint)
    d = np.asarray(p_to, float) - np.asarray(p_from, float)
    if np.dot(d, n) < 0:
        n = -n
    # where the straight line p_from -> p_to crosses the centreline (else the middle)
    den = float(np.dot(d, n))
    if abs(den) > 1e-9:
        tt = float(np.dot(c - np.asarray(p_from, float), n)) / den
        q = np.asarray(p_from, float) + tt * d
    else:
        q = c
    s = float(np.dot(q - c, u))
    lim = max(0.0, hu - end_margin)
    s = float(np.clip(s, -lim, lim))
    q = c + s * u
    A = q - (hn + before) * n
    B = q + (hn + after) * n
    return A, B, n
