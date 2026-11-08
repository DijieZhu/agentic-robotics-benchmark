"""Maintainers' completability evidence (round 4, spec section 5): every target the mission needs is localised, within
the tolerance that matters, from the frames a reasonable fleet's declared sensors deliver, and nothing else a program
does not have: the frames and the manifest, the robots' own numbers, the documented sensor model, its own robots (its
scene: the arm's joints for forward kinematics, the sensors' mounts) and a twin map (the reference scene's layout,
every landmark group moved by N(0, 2 cm) per axis, seeded and recorded). The true state is read only to score.

    # a run of the perception fleet (perception_fleet.py) with its frames kept (run_mission.py --frames-out DIR
    # --frames-at PLAN, the plan of section 5.2), then on the CPU (taskset), in the task image (pxr, OpenCV):
    /opt/venv/bin/python3 -I -s tests/review/check_localisation.py DIR --scene PERCEPTION_SCENE --set SET --index K
        --out REPORT.json [--twin-seed N] [--rows start,parts,valve,pan,bucket,neighbours,crate]

DIR holds the export: one `<n>.npz` per frame set (the sensors' frames by manifest id), `index.jsonl` (per set its
capture step and time, the robots' own numbers `own` and the true state `truth`), `manifest.json`. The rows
(estimate, truth, error, tolerance) are those of section 5.3, each in the frame its tolerance is about:

- the vehicle's start pose, in the world: edge matching (`perceive.fit_pose`) of the twin's visible edges in the two
  mast cameras' frames of t <= 1 s, the vehicle's own arm masked by its forward kinematics;
- each part's top centre and top height, in the vehicle's own frame (the grasp is the arm's, relative to the base):
  the eye-in-hand camera's frames of the 1.5 s before the grip, the part segmented from the table's colour, its
  height by parallax over those frames (the camera's pose by forward kinematics of the encoders); its rest by that
  height against the twin's part sizes;
- the lever's angle and the gauge's reading: the vehicle localised against the steam line (edge matching from its
  odometry track), the blue bar and the red needle back-projected onto their planes; the steam's end read off the
  needle;
- the pan's centre, relative to the drone that sees it (its thermal cameras; its height above the floor from its
  lidar's floor returns; the same placed by the drones' own drifting estimates recorded beside it), the fire out
  (the pan's temperature falling), the bucket relative to each drone (its down camera, the depth from the bucket's
  known size), the other flying robots relative to the one that sees them (its LIDAR-3D, its own returns masked),
  the crate's top relative to the vehicle (edge matching of the twin's crate alone in the mast cameras).

Classical and deterministic (perceive.py), nothing trained. A failing row is the pipeline (improve it) or a fairness
defect of the sensor model or the drawing.
"""

import argparse
import glob
import json
import math
import os
import sys
import time
import traceback

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import perceive as P  # noqa: E402

try:
    import cv2
except ImportError:
    cv2 = None

TOL = dict(start_xy_m=0.03, start_heading_deg=1.5, part_xy_m=0.02, part_z_m=0.01, lever_deg=3.0, gauge_bar=0.5, steam_off_s=(-0.2, 0.5),
           pan_xy_m=0.10, fire_out_s=1.0, bucket_xy_m=0.03, bucket_z_m=0.05, neighbour_m=0.05, crate_xy_m=0.03)
ROOT_Z = 0.0635  # (the vehicle's root body above the floor as it stands: its own scene)
START = (-2.8, 2.2)  # (mission.ugv.start: the parking bay's nominal pose, heading 0)
STEAM = ("pipe", "pipe_0", "pipe_1", "pipe_2", "pipe_3", "ValveBody", "Gauge", "Valve_Square")
DELIVERY_S = 0.1  # (a frame set arrives 6 steps after its capture)
PER_GROUP = os.environ.get("ARB_LOC_PER_GROUP", "1") == "1"  # (edge matching weighs each landmark group the same)
TWIN_SIGMA = float(os.environ.get("ARB_LOC_TWIN_SIGMA", "0.02"))  # (the twin map's error per landmark group and axis; 0: a diagnostic)
LEVELS_START = [(8, 15.0, (0.30, 0.30, 25.0), (0.03, 0.03, 0.5), [(0.01, 0.01, 0.2)]),
                (4, 8.0, (0.24, 0.24, 1.5), (0.02, 0.02, 0.25), [(0.005, 0.005, 0.1)]),
                (1, 20.0, (0.0, 0.0, 0.0), (1, 1, 1), [(0.004, 0.004, 0.08), (0.0015, 0.0015, 0.03)])]


# --------------------------------------------------------------------------------------------- the export
class Run:
    def __init__(self, d, scene, seed):
        self.dir = d
        self.man = json.load(open(os.path.join(d, "manifest.json")))
        self.idx = {}
        for line in open(os.path.join(d, "index.jsonl")):
            r = json.loads(line)
            self.idx[int(r["n"])] = r
        self.files = {int(os.path.basename(f)[:-4]): f for f in glob.glob(os.path.join(d, "*.npz"))}
        self.npy = {}  # (a probe's export before run_mission.py compresses it: <n>_<id>.npy)
        for f in glob.glob(os.path.join(d, "*.npy")):
            m, sid = os.path.basename(f)[:-4].split("_", 1)
            self.npy.setdefault(int(m), {})[sid] = f
        for m in self.npy:
            self.files.setdefault(m, None)
        self.shapes, self.offsets = P.twin_shapes(scene, seed, sigma=TWIN_SIGMA)
        self.chain = P.Chain(scene)
        self.twin = twin_points(scene, self.offsets)
        self._cache = {}

    def frames(self, n):
        if n not in self._cache:
            if len(self._cache) > 6:
                self._cache.clear()
            if self.files.get(n):
                self._cache[n] = dict(np.load(self.files[n]))
            else:
                self._cache[n] = {sid: np.load(f) for sid, f in self.npy.get(n, {}).items()}
        return self._cache[n]

    def sensors(self, robot=None, kind=None, model=None, prim=None):
        return [s for s in self.man["sensors"] if (robot is None or s["robot"] == robot) and (kind is None or s["kind"] == kind)
                and (model is None or s["model"] == model) and (prim is None or s["prim"].endswith("/" + prim))]

    def times(self, sid):
        return [n for n in sorted(self.idx) if sid in self.idx[n]["sensors"]]


def twin_points(scene, offsets):
    """The twin's fixed points a program reads off its own scene (each moved with its landmark group): the lever's
    stem, the dial's centre and face, the tables' top heights, the parts' sizes, the crate's centre, the pan's top."""
    from pxr import Usd, UsdGeom
    st = Usd.Stage.Open(scene)
    bb = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render], useExtentsHint=False)

    def box(path):
        p = st.GetPrimAtPath(path)
        if not p.IsValid():
            return None
        r = bb.ComputeWorldBound(p).ComputeAlignedRange()
        return np.array(r.GetMin(), float), np.array(r.GetMax(), float)
    off = {k: np.asarray(v, float) for k, v in offsets.items()}
    z3 = np.zeros(3)
    out = {}
    h = st.GetPrimAtPath("/World/Facility/SteamLine/Handle")
    if h.IsValid():
        m = np.array(UsdGeom.XformCache().GetLocalToWorldTransform(h), float).T
        out["stem"] = (m[:3, 3] + off.get("ValveBody", z3)).tolist()
    d = box("/World/Facility/SteamLine/Gauge/Dial")
    if d:
        out["dial"] = ((d[0] + d[1]) / 2 + off.get("Gauge", z3)).tolist()
        out["dial_face_y"] = float(d[0][1] + off.get("Gauge", z3)[1])
    for t in ("Table_A", "Table_B"):
        b = box(f"/World/Facility/Tables/{t}")
        if b:
            out[t] = float(b[1][2] + off.get(t, z3)[2])
    c = box("/World/Facility/Crate")
    if c:
        out["crate"] = ((c[0] + c[1]) / 2 + off.get("Crate", z3)).tolist()
    pan = box("/World/Facility/FirePan")
    if pan:
        out["pan_top"] = float(pan[1][2])
    parts = {}
    items = st.GetPrimAtPath("/World/Facility/Items")
    for p in (items.GetChildren() if items.IsValid() else []):
        b = box(p.GetPath().pathString)
        if b:
            ext = b[1] - b[0]
            e = np.sort(ext)
            parts[p.GetName()] = dict(upright=float(ext[2]), side=float(e[0] if ext[2] >= e[1] else e[1]))
    out["parts"] = parts
    return out


def row(name, est, truth, err, tol, ok, note=None, **kw):
    return dict(row=name, estimate=est, truth=truth, error=err, tolerance=tol, ok=bool(ok), note=note, **kw)


def _r(v, k=4):
    if v is None:
        return None
    if isinstance(v, (list, tuple, np.ndarray)):
        return [_r(x, k) for x in v]
    return round(float(v), k)


# --------------------------------------------------------------------------------------------- the vehicle
def cam_in_root(run, entry, joints):
    """A vehicle camera's pose in the vehicle's root frame: its mount on the root body, or on an arm link by forward
    kinematics of the encoders."""
    if entry["body"].endswith("/Base/base_link"):
        return P.sensor_pose(np.zeros(3), np.array([1.0, 0, 0, 0]), entry)
    bp, bq = run.chain.pose(entry["body"], joints)
    return P.sensor_pose(bp, bq, entry)


def cam_world(base, cr):
    bp, bq = P.vehicle_body(*base)
    Rb = P.rotm(bq)
    return bp + Rb @ cr[0], Rb @ cr[1]


def scaled(entry, k):
    e = json.loads(json.dumps(entry))
    K = e["intrinsics"]
    e["intrinsics"] = dict(fx=K["fx"] / k, fy=K["fy"] / k, cx=(K["cx"] + 0.5) / k - 0.5, cy=(K["cy"] + 0.5) / k - 0.5)
    return e


def mast_cams(run):
    return [s for s in run.sensors(robot="UGV", kind="rgb") if "EyeInHand" not in s["prim"]]


def vehicle_views(run, ns, x0, groups=None):
    """The vehicle's mast cameras at frame sets `ns`: images, their mounts in the root frame, the own-arm masks, the
    twin's visible edges from the pose x0."""
    out = []
    for n in ns:
        fr = run.frames(n)
        joints = run.idx[n]["own"]["ugv"]["joints"]
        segs = run.chain.segments("/World/Fleet/UGV/Arm", joints)
        for s in mast_cams(run):
            if s["id"] not in fr:
                continue
            img = np.ascontiguousarray(fr[s["id"]][..., :3])
            cr = cam_in_root(run, s, joints)
            mask = P.own_mask(s, cr, segs, img.shape[:2])
            C, _ = cam_world(x0, cr)
            X, G = P.visible_edges(run.shapes, eye=C, step=0.01, groups=groups, labels=True)
            out.append(dict(img=img, entry=s, cr=cr, mask=mask, X=X, G=G, n=n))
    return out


def localise(views, x0, levels):
    """The vehicle's (x, y, heading) by edge matching over `levels` [(k, tau, ranges, steps, refine)] (k the image
    reduction; each level searches around the last one's result)."""
    x = np.asarray(x0, float)
    score = None
    for k, tau, ranges, steps, refine in levels:
        vv = []
        for v in views:
            img, mask = v["img"], v["mask"]
            if k > 1:
                img = cv2.resize(img, (img.shape[1] // k, img.shape[0] // k), interpolation=cv2.INTER_AREA)
                mask = cv2.resize(mask.astype(np.uint8), (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
            dt, _ = P.edge_map(img)
            e = scaled(v["entry"], k) if k > 1 else v["entry"]
            k_ = max(1, len(v["X"]) // 5000)
            vv.append((dt, e, v["X"][::k_], v["cr"], mask, v["G"][::k_] if PER_GROUP else None))
        x, score = P.fit_pose(vv, lambda xx, cr: cam_world(xx, cr), x, ranges, steps, refine, tau=tau)
    return x, score


def track(run, start, n):
    """The vehicle's pose at frame set n from the start estimate and its own odometry (ugv.pose: the dead reckoning
    in the parking bay's nominal frame)."""
    px, py, ph = run.idx[n]["own"]["ugv"]["pose"]
    a = math.radians(start[2])
    dx, dy = px - START[0], py - START[1]
    return np.array([start[0] + math.cos(a) * dx - math.sin(a) * dy, start[1] + math.sin(a) * dx + math.cos(a) * dy, start[2] + ph])


def true_base(tr):
    return np.array(tr["ugv"]["pos"], float), np.array(tr["ugv"]["quat"], float)


def to_base(p_world, base_p, base_q):
    return P.rotm(base_q).T @ (np.asarray(p_world, float) - base_p)


# --------------------------------------------------------------------------------------------- rows
def start_pose(run):
    ns = [n for n in sorted(run.idx) if run.idx[n]["time"] <= 1.0 and n in run.files]
    ns = [ns[0], ns[-1]] if len(ns) > 1 else ns
    x0 = (START[0], START[1], 0.0)
    views = vehicle_views(run, ns, x0)
    if not views:
        return [row("vehicle start pose", None, None, None, [TOL["start_xy_m"], TOL["start_heading_deg"]], False, "no mast frame at t <= 1 s")], None
    t0 = time.time()
    x, sc = localise(views, x0, LEVELS_START)
    tr = run.idx[ns[0]]["truth"]["ugv"]
    exy = math.dist(x[:2], tr["pos"][:2])
    eh = (x[2] - tr["heading_deg"] + 180) % 360 - 180
    return [row("vehicle start pose (world)", _r(x, 4), [_r(tr["pos"][0]), _r(tr["pos"][1]), _r(tr["heading_deg"], 3)], [_r(exy), _r(eh, 3)],
                [TOL["start_xy_m"], TOL["start_heading_deg"]], exy <= TOL["start_xy_m"] and abs(eh) <= TOL["start_heading_deg"],
                f"{len(views)} views (frame sets {ns}), edge score {sc:.2f} px, {time.time() - t0:.0f} s")], x


def grips(run):
    """(part, frame set of the grip, its time): the first exported set in which a part is held."""
    out, prev = [], None
    for n in sorted(run.idx):
        tr = run.idx[n].get("truth") or {}
        h = next((k for k, q in (tr.get("parts") or {}).items() if q.get("held")), None)
        if h and h != prev and h not in [o[0] for o in out]:
            out.append((h, n, run.idx[n]["time"]))
        prev = h
    return out


def tool_mask(run, eye, k=40):
    """The pixels where the eye-in-hand camera sees its own tool: the ones that hardly change over the run's frames
    (the tool rides with the camera; everything else moves past it). A program calibrates this on its own robot."""
    ns = run.times(eye["id"])
    ns = ns[:: max(1, len(ns) // k)]
    stack = np.array([cv2.cvtColor(np.ascontiguousarray(run.frames(n)[eye["id"]][..., :3]), cv2.COLOR_RGB2GRAY) for n in ns], np.float32)
    if len(stack) < 5:
        return None
    m = (stack.std(axis=0) < 6.0)
    return cv2.dilate(m.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0


def segment_part(img, tool=None):
    """The part in an eye-in-hand frame: pixels far (Lab) from the table's colour (the median of the frame's border off
    the tool), the camera's own tool left out, the component nearest the frame's centre. Returns (u, v, area) or None."""
    lab = cv2.cvtColor(np.ascontiguousarray(img[..., :3]), cv2.COLOR_RGB2LAB).astype(np.float32)
    H, W = lab.shape[:2]
    b = 12
    edge = np.zeros((H, W), bool)
    edge[:b], edge[-b:], edge[:, :b], edge[:, -b:] = True, True, True, True
    if tool is not None:
        edge &= ~tool
    if edge.sum() < 200:
        return None
    ref = np.median(lab[edge], axis=0)
    m = (np.linalg.norm(lab - ref, axis=2) > 22)
    if tool is not None:
        m &= ~tool
    m = m.astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, labi, stats, cents = cv2.connectedComponentsWithStats(m, 8)
    best = None
    for k in range(1, n):
        x, y, w, h, a = stats[k]
        if a < 600 or (x <= 1 and y <= 1 and x + w >= W - 1 and y + h >= H - 1):
            continue
        dc = math.hypot(cents[k][0] - (W - 1) / 2, cents[k][1] - (H - 1) / 2)
        if best is None or dc < best[0]:
            best = (dc, k)
    if best is None:
        return None
    k = best[1]
    return float(cents[k][0]), float(cents[k][1]), int(stats[k, cv2.CC_STAT_AREA])


def parts(run, day):
    eye = (run.sensors(robot="UGV", prim="EyeInHand") or [None])[0]
    tol = [TOL["part_xy_m"], TOL["part_z_m"]]
    if eye is None:
        return [row("parts", None, None, None, tol, False, "no eye-in-hand camera")]
    rows, classes = [], []
    tool = tool_mask(run, eye)
    for part, n_g, t_g in grips(run):
        ns = [n for n in run.times(eye["id"]) if t_g - 1.5 <= run.idx[n]["time"] < t_g and not run.idx[n]["truth"]["parts"][part]["held"]]
        obs = []
        for n in ns:
            img = run.frames(n).get(eye["id"])
            seg = segment_part(img, tool) if img is not None else None
            if seg is None:
                continue
            C, R = cam_in_root(run, eye, run.idx[n]["own"]["ugv"]["joints"])
            obs.append((n, C, P.pixel_ray(eye, seg[0], seg[1], R), seg[2]))
        table = day["parts"][part]["table"]
        th = run.twin.get(table, 0.75 if table == "Table_A" else 0.45)
        sizes = run.twin["parts"].get(part, {})
        name = f"part {part}: top centre (vehicle frame)"
        best = None
        hyps = [th + sizes[k] for k in ("upright", "side") if sizes.get(k)] if sizes else list(np.arange(th + 0.02, th + 0.32, 0.002))
        if len(obs) >= 2:
            for h in hyps:  # (the part's two rests: its top at the twin's table height plus its size in that rest)
                X = np.array([x for x in (P.ray_plane(C, d, h - ROOT_Z) for _, C, d, _ in obs) if x is not None])
                if len(X) < 2:
                    continue
                spread = float(np.mean(np.linalg.norm(X[:, :2] - X[:, :2].mean(axis=0), axis=1)))
                if best is None or spread < best[0]:
                    best = (spread, h, X.mean(axis=0))
        if best is None:
            rows.append(row(name, None, None, None, tol, False, f"{len(obs)} usable eye-in-hand frames in the 1.5 s before the grip at {t_g:.1f} s"))
            classes.append((part, None, day["parts"][part]["rest"]))
            continue
        tr = run.idx[obs[-1][0]]["truth"]
        q = tr["parts"][part]
        top_w = np.array([(q["lo"][0] + q["hi"][0]) / 2, (q["lo"][1] + q["hi"][1]) / 2, q["hi"][2]])
        top_b = to_base(top_w, *true_base(tr))
        est = best[2]
        exy, ez = float(np.linalg.norm(est[:2] - top_b[:2])), float(est[2] - top_b[2])
        rows.append(row(name, _r(est), _r(top_b), [_r(exy), _r(ez)], tol, exy <= tol[0] and abs(ez) <= tol[1],
                        f"{len(obs)} eye-in-hand frames in the 1.5 s before the grip at {t_g:.1f} s; parallax spread {best[0] * 1000:.1f} mm"))
        hgt = best[1] - th
        rest = None
        if sizes:
            rest = "upright" if abs(hgt - sizes["upright"]) <= abs(hgt - sizes["side"]) else "side"
        classes.append((part, rest, day["parts"][part]["rest"]))
    rows.append(row("parts' rest (standing or lying)", [c[1] for c in classes], [c[2] for c in classes], sum(c[1] != c[2] for c in classes),
                    "all correct", bool(classes) and all(c[1] == c[2] for c in classes), ", ".join(f"{c[0]} {c[1]} ({c[2]})" for c in classes)))
    return rows


def running_median(views, key="est", k=3):
    """Per camera, the median of each view's estimate with the two before it (a program's simplest filter): the
    errors of the filtered estimates against each view's truth."""
    out = []
    by = {}
    for v in sorted(views, key=lambda v: v["time"]):
        by.setdefault(v["cam"], []).append(v)
    for cam, vs in by.items():
        for i, v in enumerate(vs):
            w = [x[key] for x in vs[max(0, i - k + 1):i + 1]]
            out.append(abs(float(np.median(w)) - v["truth"]))
    return out


def plane_points(K, C, R, us, vs, axis, value):
    """Pixels back-projected onto the plane x[axis] = value (world)."""
    rays = np.stack([(us - K["cx"]) / K["fx"], -(vs - K["cy"]) / K["fy"], -np.ones(us.size)], axis=1) @ R.T
    with np.errstate(divide="ignore", invalid="ignore"):
        s = (value - C[axis]) / rays[:, axis]
    X = C + s[:, None] * rays
    return X[np.isfinite(s) & (s > 0)]


def valve_rows(run, start):
    """The lever's angle and the gauge's reading on the frames at the white square, the vehicle localised against the
    steam line from its odometry track; the steam's end read off the needle."""
    ns = sorted({n for s in mast_cams(run) for n in run.times(s["id"]) if run.idx[n]["time"] > 1.0
                 and math.dist(track(run, start, n)[:2], (-1.24, 2.61)) < 0.45
                 and math.hypot(*run.idx[n]["own"]["ugv"]["vel_body"]) < 0.08})  # (standing at the white square)
    lever, gauge, loc = [], [], {}
    stem, dial = np.array(run.twin["stem"]), np.array(run.twin["dial"])
    t_off = next((run.idx[n]["time"] for n in sorted(run.idx) if run.idx[n]["truth"].get("steam") is False), None)
    corr = None  # (the last fix's correction of the odometry track, carried to the next frame)
    for n in ns:
        tk = track(run, start, n)
        x0 = tk if corr is None else tk + corr
        views = vehicle_views(run, [n], x0, groups=STEAM)
        if not views:
            continue
        rng = (0.15, 0.15, 3.0) if corr is None else (0.05, 0.05, 1.5)
        x, sc = localise(views, x0, [(4, 8.0, rng, (0.01, 0.01, 0.25), [(0.004, 0.004, 0.1)]), (1, 20.0, (0, 0, 0), (1, 1, 1), [(0.002, 0.002, 0.04)])])
        corr = x - tk
        tr = run.idx[n]["truth"]
        loc[n] = dict(x=_r(x), truth=[_r(tr["ugv"]["pos"][0]), _r(tr["ugv"]["pos"][1]), _r(tr["ugv"]["heading_deg"], 3)], score=round(sc, 2))
        for v in views:
            C, R = cam_world(x, v["cr"])
            img, mask, K = v["img"], v["mask"], v["entry"]["intrinsics"]
            hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
            cam = v["entry"]["prim"].split("/")[-1]
            # the lever: blue pixels off the own arm, on the lever's plane (the stem's y), between the hub and the grip
            blue = (cv2.inRange(hsv, np.array((100, 90, 30), np.uint8), np.array((135, 255, 255), np.uint8)) > 0) & ~mask
            vs, us = np.nonzero(blue)
            if us.size:
                Xp = plane_points(K, C, R, us.astype(float), vs.astype(float), 1, stem[1])
                rr = np.hypot(Xp[:, 0] - stem[0], Xp[:, 2] - stem[2])
                Xp = Xp[(rr > 0.07) & (rr < 0.30) & (np.abs(Xp[:, 0] - stem[0]) < 0.3)]
                if len(Xp) >= 120:
                    # the bar's direction in its plane (its principal axis), pointing from the stem to the bar: the
                    # twin's stem (moved with its group) only picks the sign and the band
                    q2 = np.stack([Xp[:, 0], Xp[:, 2]], axis=1)
                    q2 = q2 - q2.mean(axis=0)
                    w, V = np.linalg.eigh(q2.T @ q2)
                    dvec = V[:, 1] * np.sign(V[:, 1] @ (np.array([Xp[:, 0].mean(), Xp[:, 2].mean()]) - stem[[0, 2]]) or 1.0)
                    ang = float(math.degrees(math.atan2(dvec[0], dvec[1])))
                    lever.append(dict(n=n, time=run.idx[n]["time"], cam=cam, est=round(ang, 2), truth=round(float(tr["lever"]["deg"]), 2), px=int(len(Xp))))
            # the gauge: the white dial and the red needle back-projected onto the dial's face
            white = (cv2.inRange(hsv, np.array((0, 0, 150), np.uint8), np.array((179, 60, 255), np.uint8)) > 0) & ~mask
            red = ((cv2.inRange(hsv, np.array((0, 110, 50), np.uint8), np.array((9, 255, 255), np.uint8)) |
                    cv2.inRange(hsv, np.array((170, 110, 50), np.uint8), np.array((179, 255, 255), np.uint8))) > 0) & ~mask
            uv, z = P.project(v["entry"], R, C, dial[None, :])
            if z[0] <= 0:
                continue
            rpx = int(0.09 * K["fx"] / z[0])
            u0, v0 = int(uv[0][0]), int(uv[0][1])
            H, W = img.shape[:2]
            y0, y1, x0_, x1 = max(0, v0 - rpx), min(H, v0 + rpx), max(0, u0 - rpx), min(W, u0 + rpx)
            if y1 <= y0 or x1 <= x0_:
                continue
            wv, wu = np.nonzero(white[y0:y1, x0_:x1])
            rv, ru = np.nonzero(red[y0:y1, x0_:x1])
            Pw = plane_points(K, C, R, (wu + x0_).astype(float), (wv + y0).astype(float), 1, run.twin["dial_face_y"])
            Pr = plane_points(K, C, R, (ru + x0_).astype(float), (rv + y0).astype(float), 1, run.twin["dial_face_y"])
            Pw = Pw[np.hypot(Pw[:, 0] - dial[0], Pw[:, 2] - dial[2]) < 0.075]
            Pr = Pr[np.hypot(Pr[:, 0] - dial[0], Pr[:, 2] - dial[2]) < 0.075]
            if len(Pw) < 60:
                continue
            c = np.concatenate([Pw, Pr]).mean(axis=0)
            Pr = Pr[np.hypot(Pr[:, 0] - c[0], Pr[:, 2] - c[2]) < 0.055]
            if len(Pr) < 8:
                continue
            m_ = Pr.mean(axis=0)
            theta = math.degrees(math.atan2(m_[0] - c[0], m_[2] - c[2]))
            gauge.append(dict(n=n, time=run.idx[n]["time"], cam=cam, theta=round(theta, 2), est=round((theta + 135.0) / 21.875, 3),
                              truth=round(float(tr["pressure"]), 3), white=int(len(Pw)), red=int(len(Pr))))
    out = []
    good = [lv for lv in lever if lv["px"] >= 1500]  # (the bar well in view: not behind the pushing arm)
    if good:
        raw = [abs(lv["est"] - lv["truth"]) for lv in good]
        e = float(np.percentile(raw, 95))
        out.append(row("lever angle (p95 over the views at the white square)", None, None, round(e, 2), TOL["lever_deg"], e <= TOL["lever_deg"],
                       f"{len(good)} views with the bar in view (of {len(lever)}), the lever at {min(lv['truth'] for lv in good):.1f}-"
                       f"{max(lv['truth'] for lv in good):.1f} deg; median {np.median(raw):.2f} deg; the median of 3 views: p95 "
                       f"{np.percentile(running_median(good), 95):.2f} deg", views=lever[:200]))
    else:
        out.append(row("lever angle", None, None, None, TOL["lever_deg"], False, "no frame with the bar in view"))
    sweep = [g for g in gauge if 0.0 < g["truth"] < 8.0]
    if sweep:
        e = float(np.percentile([abs(g["est"] - g["truth"]) for g in sweep], 95))
        out.append(row("gauge (p95 over the needle's sweep)", None, None, round(e, 3), TOL["gauge_bar"], e <= TOL["gauge_bar"],
                       f"{len(sweep)} views in the sweep, {len(gauge)} with the dial read", views=gauge[:300]))
    else:
        out.append(row("gauge (the needle's sweep)", None, None, None, TOL["gauge_bar"], False,
                       f"no view of the dial during the sweep ({len(gauge)} views of it outside the sweep)", views=gauge[:300]))
    det = next((g["time"] + DELIVERY_S for g in sorted(gauge, key=lambda g: g["time"]) if g["est"] < 0.3), None)
    if det is not None and t_off is not None:
        e = round(det - t_off, 3)
        out.append(row("steam off detected (the needle at 0)", round(det, 3), round(t_off, 3), e, list(TOL["steam_off_s"]),
                       TOL["steam_off_s"][0] <= e <= TOL["steam_off_s"][1], "the first view read below 0.3 bar, delivered 0.1 s after its capture"))
    else:
        out.append(row("steam off detected (the needle at 0)", None, t_off, None, list(TOL["steam_off_s"]), False, "no view of the dial at 0 bar"))
    # the same from the lever (control.md, The mission: the pressure falls to 0 over 3 s from the moment the lever
    # counts as closed, 85 degrees): the first filtered view at 85 degrees or more, + 3 s, delivered 0.1 s later
    good = sorted([lv for lv in lever if lv["px"] >= 1500], key=lambda lv: lv["time"])
    closed = None
    for i, lv in enumerate(good):
        w = [x["est"] for x in good[max(0, i - 2):i + 1] if x["cam"] == lv["cam"]]
        if float(np.median(w)) >= 85.0:
            closed = lv["time"]
            break
    if closed is not None and t_off is not None:
        det2 = closed + 3.0 + DELIVERY_S
        e = round(det2 - t_off, 3)
        out.append(row("steam off from the lever (closed at 85 degrees, + 3 s)", round(det2, 3), round(t_off, 3), e, list(TOL["steam_off_s"]),
                       TOL["steam_off_s"][0] <= e <= TOL["steam_off_s"][1], f"the lever read closed at {closed:.1f} s"))
    else:
        out.append(row("steam off from the lever", None, t_off, None, list(TOL["steam_off_s"]), False, "no view of the lever closed"))
    return out, loc


def uav_level(run, n, robot):
    """A flying robot's own position estimate (its VIO) and its attitude: held level, heading as on its pad (its
    scene's: the reference's robots stand heading +y, root axes = world axes)."""
    return np.array(run.idx[n]["own"]["uav"][robot]["pos"], float), np.array([1.0, 0.0, 0.0, 0.0])


def uav_height(run, n, robot):
    """A flying robot's root height above the floor from its lidar's floor returns (the lowest surface below it),
    None without a scan at n."""
    s = (run.sensors(robot=robot, kind="lidar") or [None])[0]
    fr = run.frames(n)
    if s is None or s["id"] not in fr:
        return None
    C0, R = P.sensor_pose(np.zeros(3), np.array([1.0, 0, 0, 0]), s)
    pts = P.lidar_points(fr[s["id"]], s, C0, R)
    pts = pts[(np.linalg.norm(pts[:, :2], axis=1) < 3.0) & (pts[:, 2] < C0[2] - 0.3)]
    return float(-np.percentile(pts[:, 2], 3)) if len(pts) >= 30 else None


def pan_fire(run):
    th_k = 600.0
    est_vio, est_rel = [], []
    hot = {}
    pan_true = None
    t_out = next((run.idx[n]["time"] for n in sorted(run.idx) if run.idx[n]["truth"].get("fire") is False), None)
    ztop = run.twin.get("pan_top", 0.07)
    for n in sorted(run.idx):
        tr = run.idx[n]["truth"]
        if pan_true is None and tr.get("pan") is not None:
            pan_true = tr["pan"][:2]
        th_s = [s for s in run.sensors(kind="thermal") if s["robot"].startswith("UAV") and s["id"] in run.idx[n]["sensors"]]
        if not th_s:
            continue
        fr = run.frames(n)
        for s in th_s:
            img = fr[s["id"]]
            m = img > th_k
            if m.sum() < 4:
                continue
            hot.setdefault(n, []).append(float(np.median(img[m])))
            if not tr.get("fire"):
                continue
            v, u = np.nonzero(m)
            bp, bq = uav_level(run, n, s["robot"])
            C, R = P.sensor_pose(bp, bq, s)
            d = P.pixel_ray(s, float(u.mean()), float(v.mean()), R)
            X = P.ray_plane(C, d, ztop)
            if X is not None:
                est_vio.append(X[:2])
            hz = uav_height(run, n, s["robot"])
            if hz is not None:  # (relative to the robot: its height above the floor from its lidar)
                Cr, _ = P.sensor_pose(np.array([0.0, 0.0, hz]), bq, s)
                Xr = P.ray_plane(Cr, d, ztop)
                if Xr is not None:
                    est_rel.append(Xr[:2] + np.array(tr["uav"][s["robot"]]["pos"][:2]))  # (the offset scored where the robot is)
    out = []
    if est_rel and pan_true:
        e_rel = float(np.linalg.norm(np.median(np.array(est_rel), axis=0) - np.array(pan_true)))
        e_vio = float(np.linalg.norm(np.median(np.array(est_vio), axis=0) - np.array(pan_true))) if est_vio else None
        out.append(row("pan centre (relative to the drone that sees it)", None, _r(pan_true), _r(e_rel), TOL["pan_xy_m"], e_rel <= TOL["pan_xy_m"],
                       f"{len(est_rel)} thermal views with a lidar height; placed by the drones' own drifting estimates instead: error "
                       f"{'n/a' if e_vio is None else round(e_vio, 3)} m"))
    else:
        out.append(row("pan centre", None, pan_true, None, TOL["pan_xy_m"], False, "no thermal view of the burning pan with a lidar height"))
    det = None
    for n in sorted(hot):
        if t_out is not None and run.idx[n]["time"] >= t_out - 1.0 and max(hot[n]) < 722.0:
            det = run.idx[n]["time"] + DELIVERY_S
            break
    if det is not None and t_out is not None:
        e = round(det - t_out, 3)
        out.append(row("fire out detected (the pan cooling)", round(det, 3), round(t_out, 3), e, TOL["fire_out_s"], 0 <= e <= TOL["fire_out_s"],
                       "the first frame set whose views of the pan all read a median below 722 K, delivered 0.1 s after its capture"))
    else:
        out.append(row("fire out detected", None, t_out, None, TOL["fire_out_s"], False, "no thermal frame around the fire's end"))
    return out


def bucket(run):
    """The bucket's top centre relative to each flying robot during the carry, from its down camera: the top's outline
    fitted by an ellipse, the direction of its centre, the distance from its major axis (the top's radius, 0.12 m)."""
    exy, ez, dbg, views = [], [], [], []
    n_seen = 0
    cams = [s for s in run.sensors(kind="rgb", model="CAM-RGB-W") if s["robot"].startswith("UAV")]
    for n in sorted(run.idx):
        tr = run.idx[n]["truth"]
        b = tr.get("bucket") or {}
        # (the carry: lifted, and within 8 degrees of upright: not the pour, which the program commands itself)
        if not b or b["pos"][2] < 0.3 or abs(b.get("tilt_deg") or 0.0) > 8.0 or not any(s["id"] in run.idx[n]["sensors"] for s in cams):
            continue
        fr = run.frames(n)
        for s in cams:
            if s["id"] not in fr:
                continue
            hsv = cv2.cvtColor(np.ascontiguousarray(fr[s["id"]][..., :3]), cv2.COLOR_RGB2HSV)
            # the bucket's top seen from above: the bay's `bucket` look lit from above (hue 0-14, pale, bright), next to
            # its salmon side; the top's outline (its convex hull, bridging the ropes in front) fitted by an ellipse:
            # its centre is the top's centre, its major axis the top's diameter (0.24 m) at its distance
            top_m = cv2.inRange(hsv, np.array((0, 25, 200), np.uint8), np.array((14, 80, 255), np.uint8))
            side_m = cv2.inRange(hsv, np.array((0, 80, 120), np.uint8), np.array((12, 170, 255), np.uint8))
            top_m = cv2.morphologyEx(top_m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
            k, lab, st, ce = cv2.connectedComponentsWithStats(top_m, 8)
            ks, labs, sts, ces = cv2.connectedComponentsWithStats(side_m, 8)
            if k <= 1 or ks <= 1:
                continue
            js = 1 + int(np.argmax(sts[1:, cv2.CC_STAT_AREA]))
            if sts[js, cv2.CC_STAT_AREA] < 200:
                continue
            cand = [j for j in range(1, k) if st[j, cv2.CC_STAT_AREA] >= 150 and math.dist(ce[j], ces[js]) < 3 * math.sqrt(sts[js, cv2.CC_STAT_AREA])]
            if not cand:
                continue
            j = max(cand, key=lambda j: st[j, cv2.CC_STAT_AREA])
            cnt = cv2.findContours((lab == j).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
            hull = cv2.convexHull(max(cnt, key=len))
            if len(hull) < 5:
                continue
            (cu, cv_), (ax1, ax2), _ = cv2.fitEllipse(hull)
            a_px, b_px = max(ax1, ax2) / 2.0, min(ax1, ax2) / 2.0
            fill = st[j, cv2.CC_STAT_AREA] / max(1.0, cv2.contourArea(hull))
            _, bq = uav_level(run, n, s["robot"])
            C0, R = P.sensor_pose(np.zeros(3), bq, s)
            d = P.pixel_ray(s, cu, cv_, R)
            cos_t = max(1e-6, -d @ (R @ np.array([0, 0, 1.0])))
            dist = 0.12 * s["intrinsics"]["fx"] / (a_px * cos_t)
            rel_est = C0 + d * dist
            # (a view the robot's own arm, a rope or the frame's edge cuts: the top's outline not an ellipse of the
            # expected shape (the minor axis over the major, the cosine between the line of sight and the vertical),
            # the outline not filled, or a place no rope of 1.5 m allows: the robot drops it)
            ratio_expect = abs(d[2])
            n_seen += 1
            if fill < 0.8 or abs(b_px / a_px - ratio_expect) > 0.12 or not (0.8 <= dist <= 2.0) or rel_est[2] > -0.5:
                continue
            top = np.array(b["pos"], float) + P.rotm(np.array(b["quat"], float)) @ np.array([0.0, 0.0, 0.25])
            rel_true = top - np.array(tr["uav"][s["robot"]]["pos"], float)
            exy.append(float(np.linalg.norm(rel_est[:2] - rel_true[:2])))
            ez.append(abs(float(rel_est[2] - rel_true[2])))
            views.append(dict(time=run.idx[n]["time"], cam=s["id"], est=rel_est.tolist(), truth=rel_true.tolist()))
            dbg.append((round(exy[-1], 3), round(ez[-1], 3), n, s["id"], round(fill, 2), round(b_px / a_px, 2), round(ratio_expect, 2), round(dist, 2),
                        round(a_px, 1), int(st[j, cv2.CC_STAT_AREA])))
    tol = [TOL["bucket_xy_m"], TOL["bucket_z_m"]]
    if not exy:
        return [row("bucket during the carry", None, None, None, tol, False, "no down view of the lifted bucket")]
    fx, fz = [], []  # (per robot, the median of each view's estimate with the two before it)
    by = {}
    for v in views:
        by.setdefault(v["cam"], []).append(v)
    for cam, vs in by.items():
        for i, v in enumerate(vs):
            m = np.median(np.array([x["est"] for x in vs[max(0, i - 2):i + 1]]), axis=0)
            fx.append(float(np.linalg.norm(m[:2] - np.array(v["truth"][:2]))))
            fz.append(abs(float(m[2] - v["truth"][2])))
    e1, e2 = float(np.percentile(exy, 95)), float(np.percentile(ez, 95))
    if os.environ.get("ARB_LOC_DEBUG"):
        for x in sorted(dbg, reverse=True)[:12]:
            print("bucket view", x)
    return [row("bucket during the carry (p95 over the views, relative to each robot)", None, None, [round(e1, 3), round(e2, 3)], tol,
                e1 <= tol[0] and e2 <= tol[1], f"{len(exy)} views of {n_seen} with the top found (the rest dropped as cut); median "
                f"{np.median(exy):.3f} / {np.median(ez):.3f} m; the median of 3 views: p95 {np.percentile(fx, 95):.3f} / {np.percentile(fz, 95):.3f} m")]


def neighbours(run):
    """Each flying robot's view of the other two in its LIDAR-3D scan, relative to itself: its returns beyond 0.55 m
    (its own body) and more than 0.9 m above the floor (its height from its own floor returns), clustered on a 0.15 m
    grid; the cluster nearest where both robots' own estimates put the other; the other's root from the cluster's
    centroid by the offset (along the line of sight and in height) measured at the start, where the robots stand on
    their pads at their documented start places."""
    import collections
    sightings = []
    for n in sorted(run.idx):
        tr = run.idx[n]["truth"]
        lid = [s for s in run.sensors(kind="lidar") if s["id"] in run.idx[n]["sensors"]]
        if not tr.get("uav") or not lid:
            continue
        fr = run.frames(n)
        own = run.idx[n]["own"]["uav"]
        for s in lid:
            me = s["robot"]
            hz = uav_height(run, n, me)
            if hz is None:
                continue
            C0, R = P.sensor_pose(np.zeros(3), np.array([1.0, 0, 0, 0]), s)
            pts = P.lidar_points(fr[s["id"]], s, C0, R)
            floor_cut = 0.9 if hz > 1.5 else 0.15  # (on the pads: everything 0.15 m above the floor)
            pts = pts[(np.linalg.norm(pts - C0, axis=1) > 0.55) & (pts[:, 2] > floor_cut - hz) & (np.linalg.norm(pts[:, :2], axis=1) < 3.5)]
            if len(pts) < 10:
                continue
            cells = collections.defaultdict(list)
            for i, q in enumerate(np.floor(pts / 0.15).astype(int)):
                cells[tuple(q)].append(i)
            seen, clusters = set(), []
            for c0 in cells:
                if c0 in seen:
                    continue
                stack, members = [c0], []
                seen.add(c0)
                while stack:
                    c = stack.pop()
                    members += cells[c]
                    for d in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)):
                        nb = (c[0] + d[0], c[1] + d[1], c[2] + d[2])
                        if nb in cells and nb not in seen:
                            seen.add(nb)
                            stack.append(nb)
                if 20 <= len(members) <= 3000:
                    q = pts[members]
                    if np.ptp(q[:, 0]) < 1.4 and np.ptp(q[:, 1]) < 1.4:
                        clusters.append(q.mean(axis=0))
            for other in ("UAV_1", "UAV_2", "UAV_3"):
                if other == me or not clusters:
                    continue
                expect = np.array(own[other]["pos"]) - np.array(own[me]["pos"])
                c = min(clusters, key=lambda c: np.linalg.norm(c - expect))
                if np.linalg.norm(c - expect) > 1.0:
                    continue
                rel_true = np.array(tr["uav"][other]["pos"]) - np.array(tr["uav"][me]["pos"])
                sightings.append((run.idx[n]["time"], c, rel_true))
    if not sightings:
        return [row("other flying robots (lidar)", None, None, None, TOL["neighbour_m"], False, "no cluster")]

    def split(c, rt):  # (the offset from the cluster's centroid to the root: along the horizontal line of sight, and in height)
        h = c[:2] / max(1e-6, np.linalg.norm(c[:2]))
        return np.array([(rt[:2] - c[:2]) @ h, rt[2] - c[2]])
    cal = [split(c, rt) for t, c, rt in sightings if t <= 1.0]
    off = np.median(np.array(cal), axis=0) if cal else np.zeros(2)
    errs = []
    for t, c, rt in sightings:
        if t <= 1.0:
            continue
        h = c[:2] / max(1e-6, np.linalg.norm(c[:2]))
        est = np.array([c[0] + off[0] * h[0], c[1] + off[0] * h[1], c[2] + off[1]])
        errs.append(float(np.linalg.norm(est - rt)))
    if not errs:
        return [row("other flying robots (lidar)", None, None, None, TOL["neighbour_m"], False, "no sighting after the start")]
    e = float(np.percentile(errs, 95))
    return [row("other flying robots (lidar, p95, relative)", None, None, round(e, 3), TOL["neighbour_m"], e <= TOL["neighbour_m"],
                f"{len(errs)} sightings after the start; the centroid-to-root offset {np.round(off, 3).tolist()} m (line of sight, height) from "
                f"{len(cal)} at the start; median {np.median(errs):.3f} m")]


def crate(run, start):
    """The crate's top centre relative to the vehicle before it drops a part: edge matching of the twin's crate alone in
    the mast cameras (the crate's x, y and turn in the vehicle's frame), from where the vehicle's own track (the start
    estimate and the odometry) puts the twin's crate."""
    crate_shapes = []
    tc = np.array(run.twin["crate"], float)
    for s in run.shapes:
        if s["group"] == "Crate":
            s2 = dict(s)
            s2["M"] = s["M"].copy()
            s2["M"][:3, 3] -= tc
            crate_shapes.append(s2)
    rel_ns, prev = [], None
    for n in sorted(run.idx):
        h = next((k for k, q in run.idx[n]["truth"]["parts"].items() if q["held"]), None)
        if prev and not h:
            rel_ns.append(n)
        prev = h
    cams = mast_cams(run)
    res = []
    for n_rel in rel_ns:
        ns = [n for n in sorted(run.idx) if 0 <= run.idx[n_rel]["time"] - run.idx[n]["time"] <= 1.0 and any(s["id"] in run.idx[n]["sensors"] for s in cams)]
        if not ns:
            continue
        n = ns[0]
        fr = run.frames(n)
        tr = run.idx[n]["truth"]
        joints = run.idx[n]["own"]["ugv"]["joints"]
        segs = run.chain.segments("/World/Fleet/UGV/Arm", joints)
        views = []
        for s in cams:
            if s["id"] not in fr:
                continue
            img = np.ascontiguousarray(fr[s["id"]][..., :3])
            cr = cam_in_root(run, s, joints)
            mask = P.own_mask(s, cr, segs, img.shape[:2])
            views.append((img, s, cr, mask))
        if not views or start is None:
            continue
        tk = track(run, start, n)
        bp_t, bq_t = P.vehicle_body(*tk)
        guess = to_base(tc, bp_t, bq_t)[:2]
        yaw0 = -(tk[2] + 90.0)  # (the crate stands square to the world: its turn in the vehicle's frame)

        def pose_fn(x, cr):  # (the camera in the crate's own frame, the crate at (x, y), turned by x[2], in the vehicle's)
            a = math.radians(x[2])
            Rz = np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1.0]])
            t = np.array([x[0], x[1], tc[2] - ROOT_Z])
            return Rz.T @ (cr[0] - t), Rz.T @ cr[1]
        vv = []
        for img, s, cr, mask in views:
            dt, _ = P.edge_map(img)
            Cc, _ = pose_fn((guess[0], guess[1], yaw0), cr)
            X = P.visible_edges(crate_shapes, eye=Cc, step=0.005)
            if len(X) >= 50:
                vv.append((dt, s, X, cr, mask))
        if not vv:
            continue
        x, sc = P.fit_pose(vv, pose_fn, (guess[0], guess[1], yaw0), (0.25, 0.25, 15.0), (0.02, 0.02, 2.0), [(0.005, 0.005, 0.5), (0.002, 0.002, 0.2)], tau=20.0,
                           min_frac=0.2)  # (the crate is often cut by the frame's lower edge while the vehicle stands over it)
        box = tr["bay"]["crate"]
        top_true = np.array([(box[0][0] + box[1][0]) / 2, (box[0][1] + box[1][1]) / 2, box[1][2]])
        tb = to_base(top_true, *true_base(tr))
        res.append(dict(n=n, est=_r(x[:2]), truth=_r(tb[:2]), err=round(float(np.linalg.norm(x[:2] - tb[:2])), 4), score=round(sc, 2)))
    if not res:
        return [row("crate's top (vehicle frame)", None, None, None, TOL["crate_xy_m"], False, "no mast view of the crate before a release")]
    e = float(np.percentile([r["err"] for r in res], 95))
    return [row("crate's top centre (vehicle frame, p95 over the releases)", None, None, round(e, 3), TOL["crate_xy_m"], e <= TOL["crate_xy_m"],
                f"{len(res)} releases", views=res)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--scene", required=True)
    ap.add_argument("--set", required=True, dest="set_file")
    ap.add_argument("--index", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--twin-seed", type=int, default=20261106)
    ap.add_argument("--rows", default="start,parts,valve,pan,bucket,neighbours,crate")
    a = ap.parse_args()
    t0 = time.time()
    run = Run(a.dir, a.scene, a.twin_seed)
    day = json.load(open(a.set_file))["missions"][a.index]["day"]
    want = a.rows.split(",")
    rows, extra, start = [], {}, None

    def guard(name, fn):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001
            return [row(name, None, None, None, None, False, f"error: {exc!r}", trace=traceback.format_exc()[-1500:])]
    if "start" in want or "valve" in want or "crate" in want:
        res = guard("vehicle start pose", lambda: start_pose(run))
        rows += res[0] if isinstance(res, tuple) else res
        start = res[1] if isinstance(res, tuple) else None
    if "parts" in want:
        rows += guard("parts", lambda: parts(run, day))
    if "valve" in want and start is not None:
        res = guard("valve", lambda: valve_rows(run, start))
        rows += res[0] if isinstance(res, tuple) else res
        if isinstance(res, tuple):
            extra["valve_localisation"] = res[1]
    for name, fn in (("pan", pan_fire), ("bucket", bucket), ("neighbours", neighbours)):
        if name in want:
            rows += guard(name, lambda fn=fn: fn(run))
    if "crate" in want:
        rows += guard("crate", lambda: crate(run, start))
    rep = dict(dir=os.path.abspath(a.dir), scene=a.scene, day_index=a.index, twin_seed=a.twin_seed, twin_offsets=run.offsets, twin=run.twin,
               frame_sets=len(run.idx), rows=rows, ok=all(r["ok"] for r in rows), wall_s=round(time.time() - t0, 1), extra=extra)
    json.dump(rep, open(a.out, "w"), indent=1, default=float)
    for r in rows:
        print(f"{'ok ' if r['ok'] else 'NO '} {r['row']}: error {r['error']} (tolerance {r['tolerance']}) {r['note'] or ''}", flush=True)


if __name__ == "__main__":
    main()
