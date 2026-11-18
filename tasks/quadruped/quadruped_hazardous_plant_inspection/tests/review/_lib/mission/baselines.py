"""Baseline programs. None of them has a privilege: each gets the Robot facade only,
starts from the pad pose the work order gives, and navigates on its (drifting) odometry.

  idle                 stands, then finishes
  walk_out_and_back    walks 5 m into the building and back to the pad
  last_logged_values   reports the round sheet's 2009 values from the pad, no frame
  constant_values      reports one constant per item from the pad, no frame
  corridor_only        stays where the link is up: walks the corridor from the pad while the link holds, looks at
                       the instruments the round sheet places in a corridor and scans for listed tag plates, reads
                       what it finds (design target 0.10: slot A)
  trust_the_drawing    walk mode only; plans on the drawing to each documented position, takes a frame toward it
                       and reads whatever is there; a blocked way is retried twice (the manual), then marked on its
                       map and re-planned once; then the item is given up (design target 0.10)
  trust_the_drawing_all_modes
                       the same document-trusting program allowed all three modes: where walk ends blocked it
                       tries high_step, then crouch_walk (no depth perception; 'try the modes in turn')
  sighted_pilot_lite   (not implemented here)

Readers get the frame cropped around where the documents put the instrument (the documented position and facing,
projected with the believed pose: odometry and the frame's T_body_cam), as an unprivileged program can do.
Version 2: the tag plates carry text only (no marker, no label code in the documents): corridor_only finds listed
plates by reading their text (bricks.readers.tagtext, the reference's text reader), and the quarter-turn valve's
reader takes the plate's pose from the text plate; the trust-the-drawing baselines plan on whatever the drawing
holds (a missing sheet leaves its part of the building blank on their map, so they walk into it blind).
On a ladder site the round sheet carries the 2009 values (last_2009); on a draft site last_logged_values draws them
from a hash of the tag, independent of the plant's state, like the real ones.
"""

from __future__ import annotations

import hashlib
import heapq
import math

import numpy as np

from . import paths
from .catalog import reader_hint
from .nav import Driver, Halt, refusable, wrap

paths.setup()

from bricks.readers import tagtext as r_text  # noqa: E402  (version 2: the plates' text)
from bricks.targets import gauge as r_gauge  # noqa: E402
from bricks.targets import lamps as r_lamps  # noqa: E402
from bricks.targets import level as r_level  # noqa: E402
from bricks.targets import valve as r_valve  # noqa: E402

INSTRUMENTS = ("gauge", "level", "valve_lever", "valve_handwheel", "lamps", "thermal")
HEAD_MOUNT = np.array([0.165, 0.0, 0.043 + 0.012])   # tilt axis in the body frame (describe(): pan axis + 0.012)


class Odometry:
    """Pose source for an unprivileged program: the facade's odometry (z from the body height)."""

    def __init__(self, robot):
        self.r = robot

    def __call__(self):
        st = self.r.get_state()
        o = st["odometry"]
        return {"x": o["x"], "y": o["y"], "yaw": o["yaw"], "z": st["body"]["height"], "t": st["sim_time"],
                "roll": st["imu"]["roll"], "pitch": st["imu"]["pitch"]}


def _u(tag, salt=""):
    return int(hashlib.sha256((tag + salt).encode()).hexdigest()[:8], 16) / 0xFFFFFFFF


def logged_value(item):
    """The 'last logged value' of a work-order item: the round sheet's 2009 reading (ladder sites), else a value from
    a hash of the tag (draft sites); either is independent of the plant's current state."""
    t, tag = item["type"], item["tag"]
    last = item.get("last_2009") or {}
    if "reading" in last and last["reading"] is not None:
        v = last["reading"]
        if t == "valve_lever":
            if isinstance(v, str):
                return 90.0 if v.strip().lower() in ("closed", "shut") else 0.0
            return float(v)
        if t == "lamps":
            return [str(x).strip().lower() in ("on", "lit", "true", "1") for x in v]
        if t == "thermal":
            units = item.get("units", [])
            return {"unit": item.get("component") or (units[0] if units else None), "temp_C": float(v)}
        try:
            return float(v)
        except (TypeError, ValueError):
            return v
    u = _u(tag, "2009")
    if t == "gauge":
        lo, hi = item["scale"]["range"]
        return round(lo + (0.15 + 0.7 * u) * (hi - lo), 3)
    if t in ("level", "valve_handwheel"):
        return round(100.0 * (0.1 + 0.8 * u), 1)
    if t == "valve_lever":
        return 0.0 if u < 0.5 else 90.0
    if t == "lamps":
        n = int(item.get("n_lamps", len(item.get("colors", []))))
        return [bool(_u(tag, "L%d" % k) < 0.5) for k in range(n)]
    if t == "thermal":
        units = item.get("units", [])
        return {"unit": units[int(u * len(units)) % max(1, len(units))] if units else None, "temp_C": round(30 + 40 * u, 1)}
    if t == "dose_point":
        return round(0.3 + 1.4 * u, 3)
    return None


def constant_value(item):
    t = item["type"]
    if t == "gauge":
        lo, hi = item["scale"]["range"]
        return 0.5 * (lo + hi)
    if t in ("level", "valve_handwheel"):
        return 50.0
    if t == "valve_lever":
        return 45.0
    if t == "lamps":
        return [True] * int(item.get("n_lamps", len(item.get("colors", []))))
    if t == "thermal":
        units = item.get("units", [])
        return {"unit": units[0] if units else None, "temp_C": 50.0}
    if t == "dose_point":
        return 1.0
    return None


# ---------------------------------------------------------------------------------------------- degenerate
def idle(robot):
    robot.set_mode("stand")
    robot.step(200)
    robot.finish()


def walk_out_and_back(robot, distance=5.0):
    dv = Driver(robot, Odometry(robot))
    m = robot.mission()
    x0, y0 = m["datum"]["pos"]
    yaw0 = m["datum"]["yaw"]
    try:
        tx, ty = x0 + distance * math.cos(yaw0), y0 + distance * math.sin(yaw0)
        dv.follow([[tx, ty]], mode="walk", tol=0.15)
        dv.stop()
        dv.follow([[x0, y0]], mode="walk", tol=0.15)
        dv.stop()
        robot.set_mode("stand")
        robot.step(100)
        robot.finish()
    except Halt:
        pass


def last_logged_values(robot):
    m = robot.mission()
    robot.set_mode("stand")
    robot.step(100)
    for it in m["work_order"]["items"]:
        v = logged_value(it)
        if it["type"] in INSTRUMENTS:
            robot.report(it["tag"], v, [])
        elif it["type"] == "dose_point":
            robot.report_mark(it["tag"], v)
    robot.step(100)
    robot.finish()


def constant_values(robot):
    m = robot.mission()
    robot.set_mode("stand")
    robot.step(100)
    for it in m["work_order"]["items"]:
        v = constant_value(it)
        if it["type"] in INSTRUMENTS:
            robot.report(it["tag"], v, [])
        elif it["type"] == "dose_point":
            robot.report_mark(it["tag"], v)
    robot.step(100)
    robot.finish()


# ---------------------------------------------------------------------------------------------- readers
# half extent (m, in the face plane) of what a reader needs around the documented instrument centre
DOC_HALF = {"gauge": 0.10, "valve_lever": 0.22, "valve_handwheel": 0.14, "level": 0.32, "lamps": 0.26}


def crop_documented(fr, pose, item, margin=0.6):
    """The frame cropped around the documented position of `item` (round sheet location: pos, height, facing),
    projected with the believed robot pose (odometry x, y, yaw; body height; IMU roll / pitch) and the frame's
    T_body_cam; the intrinsics shifted to the crop. The frame unchanged when the place is not in view."""
    loc = item.get("location") or {}
    if not loc.get("pos"):
        return fr
    px, py = loc["pos"]
    pz = float(loc.get("height", 1.0))
    f = math.radians(float(loc.get("facing_deg", 0.0)))
    nrm = np.array([math.cos(f), math.sin(f), 0.0])
    lat = np.array([-nrm[1], nrm[0], 0.0])
    half = DOC_HALF.get(item["type"], 0.2)
    c = np.array([px, py, pz])
    pts = [c + a * half * lat + b * half * np.array([0.0, 0.0, 1.0]) for a in (-1, 1) for b in (-1, 1)]
    cr, sr = math.cos(pose.get("roll", 0.0)), math.sin(pose.get("roll", 0.0))
    cp, sp = math.cos(pose.get("pitch", 0.0)), math.sin(pose.get("pitch", 0.0))
    cy, sy = math.cos(pose["yaw"]), math.sin(pose["yaw"])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1.0]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Tb = np.eye(4)
    Tb[:3, :3] = Rz @ Ry @ Rx
    Tb[:3, 3] = [pose["x"], pose["y"], float(pose.get("z", 0.24))]
    T = Tb @ np.asarray(fr["T_body_cam"], float)
    Ti = np.linalg.inv(T)
    K = np.asarray(fr["K"], float)
    img = fr["image"]
    H, W = img.shape[:2]
    uv = []
    for q in pts:
        pc = Ti[:3, :3] @ q + Ti[:3, 3]
        if pc[2] <= 0.1:
            return fr
        uv.append((K[0, 0] * pc[0] / pc[2] + K[0, 2], K[1, 1] * pc[1] / pc[2] + K[1, 2]))
    uv = np.asarray(uv)
    lo, hi = uv.min(0), uv.max(0)
    size = np.maximum(hi - lo, 60.0)
    lo, hi = lo - margin * size, hi + margin * size
    x0, y0 = int(max(0, np.floor(lo[0]))), int(max(0, np.floor(lo[1])))
    x1, y1 = int(min(W, np.ceil(hi[0]))), int(min(H, np.ceil(hi[1])))
    if x1 - x0 < 40 or y1 - y0 < 40:
        return fr
    out = dict(fr)
    out["image"] = np.ascontiguousarray(img[y0:y1, x0:x1]).copy()
    Kc = K.copy()
    Kc[0, 2] -= x0
    Kc[1, 2] -= y0
    out["K"] = Kc
    return out


def read_frame(item, fr, codes, pipe_axis="y"):
    """Run the type's reference reader on a frame (public hint only). Returns (value, confidence)."""
    t = item["type"]
    img = fr["image"]
    hint = reader_hint(item, pipe_axis=pipe_axis)
    try:
        if t == "gauge":
            v, c, _ = r_gauge.read(img, hint)
            return v, c
        if t == "level":
            v, c, _ = r_level.read(img, hint)
            return (None if v is None else 100.0 * v), c
        if t == "valve_handwheel":
            v, c, _ = r_valve.read_gate(img, hint)
            return (None if v is None else 100.0 * v), c
        if t == "lamps":
            v, c, _ = r_lamps.read(img, hint)
            return v, c
        if t == "valve_lever":
            K = np.asarray(fr["K"], float)
            tv, _, _ = r_text.read(img, {"K": K, "expected": item["tag"]})
            if tv is not None and tv.get("text") == item["tag"]:
                hint = dict(hint, K=K)
                hint.update(r_text.lever_hint(tv))
            v, c, _ = r_valve.read_lever(img, hint)
            return v, c
    except Exception:  # noqa: BLE001
        return None, 0.0
    return None, 0.0


def aim_from_pose(robot, pose, target, settle_ticks=60):
    """Point the head at a world point from a (believed) robot pose: pan/tilt from the mount geometry."""
    c, s = math.cos(pose["yaw"]), math.sin(pose["yaw"])
    hx = pose["x"] + c * HEAD_MOUNT[0]
    hy = pose["y"] + s * HEAD_MOUNT[0]
    hz = float(pose.get("z", 0.24)) + HEAD_MOUNT[2]
    dx, dy, dz = target[0] - hx, target[1] - hy, target[2] - hz
    pan = wrap(math.atan2(dy, dx) - pose["yaw"])
    tilt = -math.atan2(dz, math.hypot(dx, dy))
    robot.set_head(pan, tilt)
    robot.step(settle_ticks)
    return pan, tilt


# ---------------------------------------------------------------------------------------------- the drawing
class DrawingMap:
    """Occupancy grid from the drawing alone: drawn walls (doors cut out), drawn equipment, the building outline;
    inflated by the robot's half width. A* on 8-connected cells."""

    def __init__(self, prior, res=0.1, inflate=0.27):
        self.res = res
        b = prior["building"]["outline"]
        al = prior["building"].get("airlock", [-1.5, -1.5, 2.0, 1.5])
        self.x0, self.y0 = min(b[0], al[0]) - 1.0, min(b[1], al[1]) - 1.0
        x1, y1 = max(b[2], al[2]) + 1.0, max(b[3], al[3]) + 1.0
        self.nx, self.ny = int(math.ceil((x1 - self.x0) / res)), int(math.ceil((y1 - self.y0) / res))
        occ = np.zeros((self.ny, self.nx), bool)
        for w in prior["walls"]:
            self._seg(occ, w["a"], w["b"], 0.5 * float(w.get("thickness", 0.25)), float(w.get("offset", 0.0)))
        for d in prior["doors"]:
            if "outer" in (d.get("note") or "") or d.get("kind") in ("stair", "airlock_outer"):
                continue
            self._door(occ, d)
        for e in prior.get("equipment", []):
            if e.get("kind") in ("deployment_pad",) or e.get("shape") != "rect":
                continue
            P = np.asarray(e["outline"], float)
            self._rect(occ, P.min(0), P.max(0))
        self.occ_raw = occ
        from scipy import ndimage
        r = int(math.ceil(inflate / res))
        yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
        self.occ = ndimage.binary_dilation(occ, structure=(xx * xx + yy * yy) <= r * r)

    def cell(self, x, y):
        return int((y - self.y0) / self.res), int((x - self.x0) / self.res)

    def xy(self, iy, ix):
        return self.x0 + (ix + 0.5) * self.res, self.y0 + (iy + 0.5) * self.res

    def _rect(self, occ, lo, hi):
        iy0, ix0 = self.cell(lo[0], lo[1])
        iy1, ix1 = self.cell(hi[0], hi[1])
        occ[max(0, iy0):min(self.ny, iy1 + 1), max(0, ix0):min(self.nx, ix1 + 1)] = True

    def _seg(self, occ, a, b, half, offset=0.0):
        a, b = np.asarray(a, float), np.asarray(b, float)
        d = b - a
        L = float(np.linalg.norm(d))
        if L < 1e-9:
            return
        n = np.array([-d[1], d[0]]) / L
        a, b = a + offset * n, b + offset * n
        lo = np.minimum(a, b) - half
        hi = np.maximum(a, b) + half
        self._rect(occ, lo, hi)

    def _door(self, occ, d):
        x, y = d["pos"]
        w = float(d["width"])
        if d["axis"] == "x":        # a door in a wall along y (crossed in x)
            lo, hi = (x - 0.6, y - w / 2 + 0.02), (x + 0.6, y + w / 2 - 0.02)
        else:
            lo, hi = (x - w / 2 + 0.02, y - 0.6), (x + w / 2 - 0.02, y + 0.6)
        iy0, ix0 = self.cell(*lo)
        iy1, ix1 = self.cell(*hi)
        occ[max(0, iy0):min(self.ny, iy1 + 1), max(0, ix0):min(self.nx, ix1 + 1)] = False

    def block_disc(self, x, y, r=0.5):
        """Mark a disc as occupied (a place where the robot was stopped)."""
        iy, ix = self.cell(x, y)
        k = int(math.ceil(r / self.res))
        yy, xx = np.mgrid[-k:k + 1, -k:k + 1]
        m = (xx * xx + yy * yy) <= k * k
        for dy, dx in zip(yy[m], xx[m]):
            a, b = iy + dy, ix + dx
            if 0 <= a < self.ny and 0 <= b < self.nx:
                self.occ[a, b] = True

    def free_near(self, x, y, rmax=1.5):
        iy, ix = self.cell(x, y)
        best = None
        r = int(rmax / self.res)
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                yy, xx = iy + dy, ix + dx
                if 0 <= yy < self.ny and 0 <= xx < self.nx and not self.occ[yy, xx]:
                    dd = dx * dx + dy * dy
                    if best is None or dd < best[0]:
                        best = (dd, yy, xx)
        return None if best is None else self.xy(best[1], best[2])

    def path_length(self, path):
        if not path:
            return float("inf")
        return float(sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(path[:-1], path[1:])))

    def astar(self, start, goal):
        s = self.cell(*start)
        g = self.cell(*goal)
        occ = self.occ.copy()
        # the robot's own cell and its immediate neighbours are free (it stands there)
        occ[max(0, s[0] - 1):s[0] + 2, max(0, s[1] - 1):s[1] + 2] = False
        if occ[g]:
            return None
        nb = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0), (-1, -1, 1.414), (-1, 1, 1.414), (1, -1, 1.414),
              (1, 1, 1.414)]
        h = lambda c: math.hypot(c[0] - g[0], c[1] - g[1])  # noqa: E731
        openq = [(h(s), 0.0, s)]
        came = {s: None}
        cost = {s: 0.0}
        while openq:
            _, c0, cur = heapq.heappop(openq)
            if cur == g:
                break
            if c0 > cost.get(cur, 1e18) + 1e-9:
                continue
            for dy, dx, w in nb:
                nxt = (cur[0] + dy, cur[1] + dx)
                if not (0 <= nxt[0] < self.ny and 0 <= nxt[1] < self.nx) or occ[nxt]:
                    continue
                nc = c0 + w
                if nc < cost.get(nxt, 1e18):
                    cost[nxt] = nc
                    came[nxt] = cur
                    heapq.heappush(openq, (nc + h(nxt), nc, nxt))
        if g not in came:
            return None
        path = []
        c = g
        while c is not None:
            path.append(self.xy(*c))
            c = came[c]
        path = path[::-1]
        return self._simplify(path)

    def _simplify(self, path):
        if len(path) < 3:
            return path
        out = [path[0]]
        i = 0
        while i < len(path) - 1:
            j = len(path) - 1
            while j > i + 1 and not self._clear(path[i], path[j]):
                j -= 1
            out.append(path[j])
            i = j
        return out

    def _clear(self, a, b):
        n = int(math.hypot(b[0] - a[0], b[1] - a[1]) / (0.5 * self.res)) + 2
        for t in np.linspace(0, 1, n):
            iy, ix = self.cell(a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]))
            if not (0 <= iy < self.ny and 0 <= ix < self.nx) or self.occ[iy, ix]:
                return False
        return True


def _back_off(robot, dv, dist=0.5, speed=0.15):
    """After a way ended blocked: walk straight back a little so that turning does not touch the obstacle again."""
    robot.walk(-speed, 0.0, 0.0)
    dv.wait(dist / speed)
    robot.walk(0.0, 0.0, 0.0)
    dv.wait(0.6)


def _path_index(P, xy):
    return int(np.argmin(np.linalg.norm(np.asarray(P, float) - np.asarray(xy, float)[None, :], axis=1)))


def _follow_modes(robot, dv, odo, path, modes, log, tag, stretch_m=1.5):
    """Follow `path` in modes[0]. When an attempt ends blocked (after the manual's two retries in the same mode), try
    the next allowed mode over the next stretch_m of the path, then go back to modes[0]. 'ok', 'blocked' or
    'timeout'."""
    P = [list(map(float, q)) for q in path]
    for _ in range(12):
        ok = dv.follow(P, mode=modes[0], tol=0.15)
        if ok:
            return "ok"
        if dv.last_status != "blocked":
            return dv.last_status
        p = odo()
        log.append((tag, "blocked in %s at (%.2f, %.2f) after two retries" % (modes[0], p["x"], p["y"])))
        if len(modes) == 1:
            return "blocked"
        j = _path_index(P, (p["x"], p["y"]))
        k, acc = j, 0.0
        while k + 1 < len(P) and acc < stretch_m:
            acc += math.hypot(P[k + 1][0] - P[k][0], P[k + 1][1] - P[k][1])
            k += 1
        sub = [[p["x"], p["y"]]] + P[j + 1:k + 1]
        if len(sub) < 2:
            sub = [[p["x"], p["y"]], P[-1]]
        passed = False
        for mm in modes[1:]:
            dv.stop(0.3)
            log.append((tag, "trying %s over the next %.1f m" % (mm, acc)))
            if dv.follow(sub, mode=mm, tol=0.15):
                passed = True
                break
            if dv.last_status != "blocked":
                break
        dv.stop(0.3)
        if not passed:
            return "blocked"
        P = [[odo()["x"], odo()["y"]]] + P[k + 1:]
        if len(P) < 2:
            return "ok"
    return "blocked"


def _go_on_drawing(robot, dv, odo, dmap, goal, modes, log, tag, replans=1, mark=True):
    """Plan on the drawing and follow; a way that ends blocked is backed off from, marked on the map (unless
    mark=False: the way home, which the robot came in by) and re-planned (`replans` times). True when the goal was
    reached."""
    for attempt in range(replans + 1):
        p = odo()
        path = dmap.astar((p["x"], p["y"]), goal)
        if path is None:
            log.append((tag, "no path on the drawing from (%.2f, %.2f)" % (p["x"], p["y"])))
            return False
        log.append((tag, "path of %d points to (%.2f, %.2f)" % (len(path), goal[0], goal[1])))
        res = _follow_modes(robot, dv, odo, path, modes, log, tag)
        if res == "ok":
            return True
        p = odo()
        dv.stop()
        _back_off(robot, dv)
        if mark:
            # remember the place: something the drawing does not show stops the robot there
            dmap.block_disc(p["x"] + 0.35 * math.cos(p["yaw"]), p["y"] + 0.35 * math.sin(p["yaw"]))
            log.append((tag, "%s at (%.2f, %.2f): marked on the map" % (res, p["x"], p["y"])))
        else:
            log.append((tag, "%s at (%.2f, %.2f): backed off, trying again" % (res, p["x"], p["y"])))
    return False


def _read_and_report(robot, dv, odo, it, codes, log, px, py, hz):
    """Turn toward the documented position, stand, aim the head there, take a frame, read (crop around the
    documented place), report citing the frame."""
    p = odo()
    dv.turn_to(math.atan2(py - p["y"], px - p["x"]))
    robot.set_mode("stand")
    dv.mode = "stand"
    robot.step(60)
    p = odo()
    aim_from_pose(robot, p, (px, py, hz))
    p = odo()
    if it["type"] == "thermal":
        fr = refusable(robot.get_thermal)
        if fr is not None:
            T = fr["image_c"]
            units = it.get("units", [])
            eps = float(reader_hint(it)["emissivity"])
            from bricks.sensors.thermal import emissivity_correct
            t = emissivity_correct(float(np.max(T)), eps, float(np.median(T)))
            robot.report(it["tag"], {"unit": units[0] if units else None, "temp_C": round(float(t), 2)},
                         [fr["frame_id"]])
            log.append((it["tag"], "thermal peak %.1f C reported from frame %s" % (t, fr["frame_id"])))
    else:
        fr = refusable(robot.get_camera, "inspect", depth=False)
        if fr is not None:
            v, c = read_frame(it, crop_documented(fr, p, it), codes)
            robot.report(it["tag"], v if (v is not None and c >= 0.3) else None, [fr["frame_id"]])
            log.append((it["tag"], "read %s (conf %.2f) from frame %s" % (v, c, fr["frame_id"])))
    robot.set_head(0.0, 0.0)


def _doc_goal(dmap, it, stand_off):
    loc = it.get("location") or {}
    px, py = loc["pos"]
    f = math.radians(float(loc.get("facing_deg", 0.0)))
    if it["type"] == "dose_point":
        goal = (px, py)
    else:
        goal = (px + stand_off * math.cos(f), py + stand_off * math.sin(f))
    return dmap.free_near(*goal) or goal


def trust_the_drawing(robot, stand_off=2.0, modes=("walk",)):
    """Believe the documents: plan on the drawing, odometry only, the given modes (walk only by default); visit the
    documented positions nearest first (path length on the drawing); at each take a frame toward the documented
    place and read whatever is there; a way that ends blocked (after the manual's two retries) is marked on the map
    and re-planned once, then the item is given up. The way back to the pad is planned on the plain drawing."""
    m = robot.mission()
    wo = m["work_order"]
    codes = {}                                   # version 2: no plate codes
    odo = Odometry(robot)
    log = []
    dv = Driver(robot, odo, log=lambda k, **kw: log.append(("driver", "%s %s" % (k, kw))))
    dmap = DrawingMap(m)
    pad = m["datum"]["pos"]
    sens = float(robot.describe()["dosimeter"]["sensitivity_cps_per_gyh"])
    modes = tuple(modes)
    try:
        robot.set_mode("stand")
        robot.step(100)
        todo = []
        for it in wo["items"]:
            loc = it.get("location") or {}
            if not loc.get("documented") or loc.get("pos") is None:
                log.append((it["tag"], "no documented position"))
                continue
            if it["type"] not in INSTRUMENTS and it["type"] != "dose_point":
                continue
            todo.append(it)
        while todo:
            p = odo()
            lens = [(dmap.path_length(dmap.astar((p["x"], p["y"]), _doc_goal(dmap, it, stand_off))), k)
                    for k, it in enumerate(todo)]
            it = todo.pop(min(lens)[1])
            loc = it["location"]
            px, py = loc["pos"]
            hz = float(loc.get("height", 0.8))
            goal = _doc_goal(dmap, it, stand_off)
            if not _go_on_drawing(robot, dv, odo, dmap, goal, modes, log, it["tag"]):
                log.append((it["tag"], "given up"))
                continue
            dv.stop()
            if it["type"] == "dose_point":
                robot.set_mode("stand")
                dv.mode = "stand"
                robot.get_state()
                for _ in range(3):
                    robot.step(500)
                st = robot.get_state()
                rate = st["dosimeter"]["counts"] / max(st["dosimeter"]["dwell_s"], 1e-6) / sens
                robot.report_mark(it["tag"], rate)
                log.append((it["tag"], "mark %.3f Gy/h after %.0f s" % (rate, st["dosimeter"]["dwell_s"])))
                continue
            _read_and_report(robot, dv, odo, it, codes, log, px, py, hz)
        # back to the pad on the plain drawing, without marking (the robot came in that way; the marks were for the
        # items' ways): backed off and tried again up to four times
        _go_on_drawing(robot, dv, odo, DrawingMap(m), tuple(pad), modes, log, "PAD", replans=4, mark=False)
        dv.stop()
        robot.set_mode("stand")
        robot.step(200)
        robot.finish()
    except Halt:
        pass
    return log


def trust_the_drawing_all_modes(robot, stand_off=2.0):
    """trust_the_drawing allowed all three modes: where walk ends blocked, high_step, then crouch_walk, over the next
    1.5 m (no depth perception: the modes in turn)."""
    return trust_the_drawing(robot, stand_off=stand_off, modes=("walk", "high_step", "crouch_walk"))


def _is_corridor(it):
    loc = it.get("location") or {}
    room = str(loc.get("room", ""))
    return room.upper().startswith("C0") or "CORRIDOR" in str(loc.get("room_name", "")).upper()


def corridor_only(robot, step_m=1.0, max_steps=14, read_window=(1.5, 3.0)):
    """Stay where the link is up: walk into the building along the pad heading in steps while the link holds (two
    stops without link end the walk, as does a way that ends blocked); at each stop with the link up, read the
    instruments the round sheet places in a corridor when the robot is 1.5-3 m in front of the documented place,
    and scan for listed tag plates with the inspection camera (a found one is aimed at and read); then back to the
    pad."""
    m = robot.mission()
    wo = m["work_order"]
    codes = {}                                   # version 2: no plate codes; plates are found by their text
    items = {it["tag"]: it for it in wo["items"]}
    corr = [it for it in wo["items"] if it["type"] in INSTRUMENTS and _is_corridor(it)
            and (it.get("location") or {}).get("pos") is not None]
    odo = Odometry(robot)
    log = []
    dv = Driver(robot, odo, log=lambda k, **kw: log.append(("driver", "%s %s" % (k, kw))))
    pad = m["datum"]["pos"]
    yaw0 = m["datum"]["yaw"]
    done = set()
    try:
        robot.set_mode("stand")
        robot.step(100)
        down = 0
        for k in range(max_steps + 1):
            st = robot.get_state()
            if st["link"]["up"]:
                down = 0
                p = odo()
                for it in corr:
                    if it["tag"] in done:
                        continue
                    loc = it["location"]
                    px, py = loc["pos"]
                    f = math.radians(float(loc.get("facing_deg", 0.0)))
                    d = math.hypot(px - p["x"], py - p["y"])
                    off = abs(wrap(math.atan2(p["y"] - py, p["x"] - px) - f))
                    if read_window[0] <= d <= read_window[1] and math.degrees(off) <= 30.0:
                        _read_and_report(robot, dv, odo, it, codes, log, px, py, float(loc.get("height", 1.0)))
                        done.add(it["tag"])
                robot.set_mode("stand")
                dv.mode = "stand"
                for pan in (-1.2, -0.6, 0.0, 0.6, 1.2):
                    robot.set_head(pan, -0.3)
                    robot.step(50)
                    fr = refusable(robot.get_camera, "inspect", depth=False)
                    if fr is None:
                        break
                    try:
                        vals, _ = r_text.read_all(fr["image"], {"K": fr["K"]})
                    except Exception:  # noqa: BLE001  (a reader failure is no plate)
                        vals = []
                    for tv in vals:
                        tag = tv.get("text")
                        if tag not in items or tag in done or items[tag]["type"] not in INSTRUMENTS:
                            continue
                        it = items[tag]
                        v, c = read_frame(it, fr, codes)
                        robot.report(tag, v if (v is not None and c >= 0.3) else None, [fr["frame_id"]])
                        log.append((tag, "plate %s read at pan %.1f: value %s (conf %.2f)" % (tag, pan, v, c)))
                        done.add(tag)
                robot.set_head(0.0, 0.0)
            else:
                down += 1
                if down >= 2:
                    log.append(("walk", "link down at two stops: turn back"))
                    break
            if k == max_steps:
                break
            p = odo()
            along = (p["x"] - pad[0]) * math.cos(yaw0) + (p["y"] - pad[1]) * math.sin(yaw0) + step_m
            tx, ty = pad[0] + along * math.cos(yaw0), pad[1] + along * math.sin(yaw0)
            if not dv.follow([[tx, ty]], mode="walk", tol=0.12):
                log.append(("walk", "%s at (%.2f, %.2f): turn back" % (dv.last_status, p["x"], p["y"])))
                break
            dv.stop(0.3)
        dv.stop()
        dv.follow([[pad[0], pad[1]]], mode="walk", tol=0.15)
        dv.stop()
        robot.set_mode("stand")
        robot.step(200)
        robot.finish()
    except Halt:
        pass
    return log


def sighted_pilot_lite(robot):
    raise NotImplementedError("sighted_pilot_lite is not part of this sketch run")
