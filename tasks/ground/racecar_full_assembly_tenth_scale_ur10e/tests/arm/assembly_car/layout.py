"""Per-episode physical draws of the race-car cell (grader side): where the work stand stands and how the chassis sits
on it, where the deck shelf and the parts tray stand and how their units lie in them, where the four wheels and the
stick pack lie on the bench, and the four hubs' start angles. The episode seed, the stratum and the hidden bench
plan in, a layout out; a pure function of its inputs (`layout_digest` fingerprints the result).

`episode_layout(seed, plan, kit, uv=None, cfg=None, stratum=None, nominal=False)` ->
  {"format", "seed", "stratum", "nominal", "stand": {xy, yaw_deg}, "clamp": {dx_m, dy_m, dyaw_deg},
   "fixtures": {"stand" | "shelf" | "tray": {xy, yaw_deg}}, "plays": {DCK | CMP | LDR: {dx_mm, dy_mm, dyaw_deg}},
   "parts": {unit: (pos, quat wxyz)}, "hub_deg": {hub body: deg}, "lead_end_deg", "jitter": {...}}

The draws (amplitudes in DEFAULT_LAYOUT, the grader config's "layout" block):
* the stand at its nominal place (the car's reference point at nominal_xy, the car facing the arm: yaw 180 deg)
  moved by +-jitter_m in x and in y and turned by a magnitude in (0, jitter_yaw_deg] whose sign is the stratum's;
  the chassis on the stand off its drawn place by the clamping repeatability (+-clamp_repeat_m, +-clamp_repeat_deg);
* the shelf +-jitter_m / +-jitter_yaw_deg (nominal yaw = the car's, so that the deck on it is parallel to its place
  on the car), the deck inside its play against the fence and the end stops (fixtures.SHELF_PLAY_MM, turned by at most
  play_yaw_deg, kept off the fence and the stops on the kit's collision points);
* the tray +-jitter_m / +-jitter_yaw_deg; the compute module and the LiDAR in their nests turned by their storage
  turns (fixtures.NEST_POSE) and inside the nest play (+-1.5 mm, +-2 deg, kept off the ribs on the kit's collision
  points below the ribs' top);
* the four wheels (inner face down) and the stick pack (on its underside) around the hidden bench plan: turned about
  the vertical through the footprint's centre by +-yaw_deg and moved by +-xy_m; the stick pack in addition turned by
  the stratum's lead end (0 or 180 deg from the plan, which lays it as it is mounted); redrawn until its footprint
  (the convex hull of its collision points) lies on the bench, its centre in the reach ring, gap_m clear of every unit
  placed before it and fixture_gap_m clear of every fixture (stand, car, shelf with the deck, tray, bench board), and
  out of the wheel lanes; in a seeded random order; a unit with no valid draw in `tries` restarts the layout, the
  amplitudes halve every `layouts` failed layouts, the plan itself last (the precedent's rule);
* each hub's start angle uniform in [hub_start_deg[0], hub_start_deg[1]) from its angle in the model.

Strata (the fixed hidden set, tests/hidden/episodes.json): the stand's yaw sign x the stick pack's lead end: "yaw+lead0",
"yaw+lead180", "yaw-lead0", "yaw-lead180". stratum=None draws both from the seed.

nominal=True (the privileged oracle's fallback): the plan itself, every fixture at its nominal place, no clamping
offset, the units at the middle of their plays, the hubs at the plan's "oracle" angles (or 0); the stratum's lead end
still applies.

`check_layout(lay, kit, uv, cfg)` -> (ok, problems, measures): the validity rules re-checked on a drawn layout (also run
over many seeds as a check of the draws).
"""
import copy
import hashlib
import json
import math
import os
import struct

import numpy as np

from . import fixtures as FX

__all__ = ["DEFAULT_LAYOUT", "STRATA", "parse_stratum", "unit_pieces", "unit_vertices", "unit_hulls", "unit_geometry", "episode_layout", "T_world_car",
           "T_world_fixture", "layout_digest", "layout_digest_of", "check_layout", "obstacles", "lanes_world"]

STRATA = ("yaw+lead0", "yaw+lead180", "yaw-lead0", "yaw-lead180")

DEFAULT_LAYOUT = {
    "stand": {"nominal_xy": [0.62, 0.0], "yaw_deg": 180.0, "jitter_m": 0.015, "jitter_yaw_deg": 2.0,
              "axle_height_m": FX.AXLE_HEIGHT_MM * 1e-3, "clamp_repeat_m": 0.001, "clamp_repeat_deg": 0.3},
    "shelf": {"nominal_xy": [0.30, -0.45], "yaw_deg": 180.0, "jitter_m": 0.020, "jitter_yaw_deg": 5.0, "play_yaw_deg": 0.5},
    "tray": {"nominal_xy": [0.30, 0.45], "yaw_deg": 0.0, "jitter_m": 0.040, "jitter_yaw_deg": 10.0},
    "bench_jitter": {"xy_m": 0.03, "yaw_deg": 30.0, "gap_m": 0.06, "tries": 200, "layouts": 50},
    "hub_start_deg": [0.0, 60.0],
    # the bench top (2.0 x 1.4 m centred at (0.45, 0)) less 2 mm
    "bench_xy": [[-0.548, 1.448], [-0.698, 0.698]],
    # the loose units' footprint centres from the arm's base axis, their footprints' nearest point from it
    "reach_m": None,                 # None: bench_plan.RING_M
    "base_clear_m": 0.20,
    "fixture_gap_m": 0.06,
    "lanes": None,                   # None: bench_plan.LANES
    "settle_lift_m": 0.0003,
}
# the car's plan outline (car frame, mm; the digest's whole-car bounds: x -53.3..334.5, |y| <= 92.4) grown by 1 mm
CAR_PLAN_MM = (-54.3, 335.5, -93.4, 93.4)
# the hub bodies and the loose units the layout places
HUBS = ("hub_FL", "hub_FR", "hub_RL", "hub_RR")
BENCH_UNITS = ("WHL1", "WHL2", "WHL3", "WHL4", "BAT")
FIXTURE_UNITS = ("DCK", "CMP", "LDR")


def parse_stratum(s):
    """(yaw sign +1 / -1, lead end deg 0 / 180) of a stratum label"""
    if s not in STRATA:
        raise ValueError("unknown stratum %r (have %s)" % (s, ", ".join(STRATA)))
    return (1 if s.startswith("yaw+") else -1), (180 if s.endswith("lead180") else 0)


def _merge(cfg):
    from . import bench_plan as BP
    out = copy.deepcopy(DEFAULT_LAYOUT)
    for k, v in (cfg or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k].update(v)
        else:
            out[k] = v
    if out.get("reach_m") is None:
        out["reach_m"] = list(BP.RING_M)
    if out.get("lanes") is None:
        out["lanes"] = dict(BP.LANES)
    return out


def _rng(seed, tag):
    return np.random.default_rng([int(seed) & 0xFFFFFFFF, (int(seed) >> 32) & 0xFFFFFFFF, int(tag)])


def _rotz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _T(R=None, p=None):
    T = np.eye(4)
    if R is not None:
        T[:3, :3] = R
    if p is not None:
        T[:3, 3] = p
    return T


def _quat_wxyz(R):
    """wxyz of a rotation matrix (Shepperd; w >= 0)"""
    R = np.asarray(R, float)
    t = np.trace(R)
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        q = [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        q = [(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s]
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        q = [(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s]
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        q = [(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s]
    q = np.array(q)
    q /= np.linalg.norm(q)
    return q if q[0] >= 0 else -q


def _quat_R(q):
    w, x, y, z = (float(v) for v in q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


# ------------------------------------------------------------------ the kit's units as point sets
def _stl_vertices(path):
    with open(path, "rb") as f:
        data = f.read()
    if len(data) >= 84:
        n = struct.unpack("<I", data[80:84])[0]
        if 84 + 50 * n == len(data):
            rec = np.frombuffer(data, dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]), count=n, offset=84)
            return np.unique(rec["v"].reshape(-1, 3).astype(np.float64), axis=0)
    pts = [[float(x) for x in ln.split()[1:4]] for ln in data.decode("ascii", "replace").splitlines() if ln.strip().startswith("vertex")]
    return np.unique(np.asarray(pts, np.float64).reshape(-1, 3), axis=0)


def geom_points(g, kit_dir, n_circle=64):
    """the points of one kit collision geom (an entry of a kit body's "collision" list) in its body's frame (m): a box's 8 corners, a
    cylinder's two rims (n_circle points each, circumscribed), a mesh's STL vertices"""
    t = g["type"]
    if t == "box":
        hx, hy, hz = (float(v) for v in g["size"])
        P = np.array([[sx * hx, sy * hy, sz * hz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    elif t == "cylinder":
        r, h = float(g["size"][0]), float(g["size"][1])
        r = r / math.cos(math.pi / n_circle)
        a = np.arange(n_circle) * 2 * math.pi / n_circle
        ring = np.c_[r * np.cos(a), r * np.sin(a)]
        P = np.r_[np.c_[ring, np.full(n_circle, -h)], np.c_[ring, np.full(n_circle, h)]]
    elif t == "mesh":
        P = _stl_vertices(os.path.join(kit_dir, g["file"]))
    else:
        raise ValueError("kit geom %s: unknown type %r" % (g.get("name"), t))
    R = _quat_R(g.get("quat", [1, 0, 0, 0]))
    return P @ R.T + np.asarray(g.get("pos", [0, 0, 0]), float)


def unit_pieces(kit, units=None, roles=("col",)):
    """{unit: [(n_i, 3)]} the points of each collision geom of every body of each loose unit, in the unit's root frame
    (the bodies placed by their T_design chain); the wheels' stops left out (roles)"""
    bodies = kit["bodies"]
    kd = kit.get("dir") or "."
    out = {}
    for u in (units or kit["loose"]):
        Tui = np.linalg.inv(np.asarray(bodies[u]["T_design"], float))
        pts = []
        for bn, bb in bodies.items():
            if bb.get("unit") != u:
                continue
            Tb = Tui @ np.asarray(bb["T_design"], float)
            for g in bb.get("collision", []):
                if g.get("role", "col") not in roles:
                    continue
                v = geom_points(g, kd)
                pts.append(v @ Tb[:3, :3].T + Tb[:3, 3])
        out[u] = pts
    return out


def unit_vertices(kit, units=None, max_per_unit=None, roles=("col",)):
    """{unit: (N, 3)} every collision point of each loose unit (unit_pieces concatenated); max_per_unit subsamples
    (deterministically)"""
    out = {}
    for u, pts in unit_pieces(kit, units=units, roles=roles).items():
        V = np.concatenate(pts)
        if max_per_unit and len(V) > max_per_unit:
            V = V[np.random.default_rng(0).choice(len(V), max_per_unit, replace=False)]
        out[u] = V
    return out


def unit_hulls(kit, units=None):
    """{unit: the vertices of the convex hull of its collision points (unit frame, m)}: the footprints and the rest
    heights of the layout"""
    return unit_geometry(kit, units=units)["hulls"]


def unit_geometry(kit, units=None):
    """{"hulls": {unit: hull vertices (m)}, "pieces": {unit: [the hull vertices of each collision geom (m)]}} in the
    units' frames: what episode_layout and check_layout need (compute it once per kit and pass it as `uv`)"""
    from scipy.spatial import ConvexHull
    pcs = unit_pieces(kit, units=units)
    hulls, pieces = {}, {}
    for u, pts in pcs.items():
        P = np.concatenate(pts)
        hulls[u] = P[ConvexHull(P).vertices]
        pieces[u] = [q for q in pts]
    return {"hulls": hulls, "pieces": pieces}


def _low_polys(pieces, R, t, z_top):
    """the plan polygons (2D hulls) of the pieces that reach below z_top once turned by R and shifted by t (mm in, mm
    out; R 3x3)"""
    out = []
    for q in pieces:
        P = q @ R.T + t
        if P[:, 2].min() <= z_top + 1e-6:
            out.append(_hull2(P))
    return out


# ------------------------------------------------------------------ 2D convex geometry
def _hull2(P):
    from scipy.spatial import ConvexHull
    P = np.asarray(P, float)[:, :2]
    if len(P) <= 4:
        c = P.mean(0)
        a = np.arctan2(P[:, 1] - c[1], P[:, 0] - c[0])
        return P[np.argsort(a)]
    return P[ConvexHull(P).vertices]


def _overlap(A, B):
    """convex polygons (ccw) intersect (separating axis)"""
    for P, Q in ((A, B), (B, A)):
        E = np.roll(P, -1, axis=0) - P
        N = np.c_[E[:, 1], -E[:, 0]]
        pa = P @ N.T
        pb = Q @ N.T
        if np.any(pa.min(0) > pb.max(0)) or np.any(pb.min(0) > pa.max(0)):
            return False
    return True


def _pt_seg(P, A, B):
    """min distance of points P (n, 2) to the segments A[i]B[i] (m, 2): (n, m)"""
    AB = B - A
    L = np.maximum((AB * AB).sum(1), 1e-18)
    t = np.clip(((P[:, None, :] - A[None]) * AB[None]).sum(2) / L[None], 0.0, 1.0)
    C = A[None] + t[..., None] * AB[None]
    return np.linalg.norm(P[:, None, :] - C, axis=2)


def poly_dist(A, B):
    """distance between two convex polygons (0 when they overlap)"""
    if _overlap(A, B):
        return 0.0
    da = _pt_seg(A, B, np.roll(B, -1, axis=0)).min()
    db = _pt_seg(B, A, np.roll(A, -1, axis=0)).min()
    return float(min(da, db))


def _rect(x0, x1, y0, y1):
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], float)


# ------------------------------------------------------------------ poses
def T_world_fixture(pose):
    """4x4 of a fixture frame from its {xy, yaw_deg}"""
    return _T(_rotz(math.radians(float(pose["yaw_deg"]))), [float(pose["xy"][0]), float(pose["xy"][1]), 0.0])


def T_world_car(lay):
    """4x4 of the car frame in the cell: the stand's frame, the clamping offset about the stand's origin (under the
    reference point), the car on the stand (fixtures.T_stand_car)"""
    Ts = T_world_fixture(lay["fixtures"]["stand"] if "fixtures" in lay else lay["stand"])
    c = lay.get("clamp") or {}
    J = _T(_rotz(math.radians(float(c.get("dyaw_deg", 0.0)))), [float(c.get("dx_m", 0.0)), float(c.get("dy_m", 0.0)), 0.0])
    return Ts @ J @ FX.T_stand_car()


def lanes_world(lay, L=None):
    """[(4, 2)] the wheel lanes beside the car: the nominal rectangles in the cell frame and the same rectangles moved
    with the drawn stand (the lanes are where wheels travel onto the hubs)"""
    L = _merge(L)["lanes"]
    rects = []
    for sy in (1, -1):
        y0, y1 = sorted((sy * L["y_min_m"], sy * L["y_max_m"]))
        rects.append(_rect(L["x_min_m"], L["x_max_m"], y0, y1))
    nom = _merge(None)["stand"]
    T0 = T_world_fixture({"xy": nom["nominal_xy"], "yaw_deg": nom["yaw_deg"]})
    T1 = T_world_fixture(lay["fixtures"]["stand"])
    X = T1 @ np.linalg.inv(T0)
    moved = [R @ X[:2, :2].T + X[:2, 3] for R in rects]
    return rects + moved


def obstacles(lay, kit, hulls):
    """{name: [(polygon (n, 2) world m, height m)]} of everything that is not a loose unit on the bench: the stand's
    plates with the car's plan outline, the shelf with the deck, the tray with its units, the bench board"""
    out = {}
    Ts = T_world_fixture(lay["fixtures"]["stand"])
    stand = [(P, 0.015) for P in FX.footprint_xy("stand", Ts)[:2]]
    Tc = T_world_car(lay)
    x0, x1, y0, y1 = (v * 1e-3 for v in CAR_PLAN_MM)
    stand.append((_rect(x0, x1, y0, y1) @ Tc[:2, :2].T + Tc[:2, 3], 0.32))
    out["stand"] = stand
    for fx in ("shelf", "tray"):
        Tf = T_world_fixture(lay["fixtures"][fx])
        hmax = max(hi[2] for _n, _lo, hi, _p in FX.FIXTURES[fx]["boxes"]) * 1e-3
        out[fx] = [(P, hmax) for P in FX.footprint_xy(fx, Tf)]
    for u in FIXTURE_UNITS:
        if u not in lay["parts"]:
            continue
        pos, quat = lay["parts"][u]
        R = _quat_R(quat)
        P = hulls[u] @ R.T + np.asarray(pos, float)
        fx = "shelf" if u == "DCK" else "tray"
        out[fx].append((_hull2(P), float(P[:, 2].max())))
    b = FX.BENCH_BOARD
    bx = (b["cols"] * b["square_mm"] + 2 * b["margin_mm"]) * 1e-3 / 2
    by = (b["rows"] * b["square_mm"] + 2 * b["margin_mm"]) * 1e-3 / 2
    Tb = _T(_rotz(math.radians(b["yaw_deg"])), [b["pos"][0], b["pos"][1], 0.0])
    out["bench_board"] = [(_rect(-bx, bx, -by, by) @ Tb[:2, :2].T + Tb[:2, 3], b["thickness_mm"] * 1e-3)]
    return out


# ------------------------------------------------------------------ the draws
def _stand_pose(seed, L, sign, nominal):
    s = L["stand"]
    x, y = (float(v) for v in s["nominal_xy"])
    yaw = float(s["yaw_deg"])
    if nominal:
        return {"xy": [x, y], "yaw_deg": yaw}, {"dx_m": 0.0, "dy_m": 0.0, "dyaw_deg": 0.0}
    r = _rng(seed, 0x57A0)
    j = float(s["jitter_m"])
    x, y = x + float(r.uniform(-j, j)), y + float(r.uniform(-j, j))
    mag = float(s["jitter_yaw_deg"]) * (1.0 - float(r.uniform(0.0, 1.0)))       # (0, jitter_yaw_deg]
    rc = _rng(seed, 0xC1A3)
    cj, cd = float(s["clamp_repeat_m"]), float(s["clamp_repeat_deg"])
    clamp = {"dx_m": float(rc.uniform(-cj, cj)), "dy_m": float(rc.uniform(-cj, cj)), "dyaw_deg": float(rc.uniform(-cd, cd))}
    return {"xy": [x, y], "yaw_deg": yaw + sign * mag}, clamp


def _fixture_pose(seed, L, name, tag, nominal):
    s = L[name]
    x, y = (float(v) for v in s["nominal_xy"])
    yaw = float(s["yaw_deg"])
    if not nominal:
        r = _rng(seed, tag)
        j, jy = float(s["jitter_m"]), float(s["jitter_yaw_deg"])
        x, y, yaw = x + float(r.uniform(-j, j)), y + float(r.uniform(-j, j)), yaw + float(r.uniform(-jy, jy))
    return {"xy": [x, y], "yaw_deg": yaw}


def _deck_play(seed, L, piecesD, nominal, tries=200):
    """(dx, dy mm, dyaw deg) of the deck on the shelf: inside +-SHELF_PLAY_MM, turned by at most play_yaw_deg, the
    plan hull of each of its collision pieces that reaches into the fence / stops' height range clear of them"""
    if nominal:
        return 0.0, 0.0, 0.0
    r = _rng(seed, 0xDEC0)
    g = FX.SHELF_PLAY_MM
    a = float(L["shelf"]["play_yaw_deg"])
    pcs = [q * 1e3 + np.array([0.0, 0.0, FX.DECK_TOP_MM]) for q in piecesD]
    lim = []
    for bn, lo, hi, _p in FX.SHELF["boxes"]:
        if bn in ("fence", "stop_front", "stop_rear"):
            sel = [q[:, :2] for q in pcs if q[:, 2].min() < hi[2] and q[:, 2].max() > lo[2]]
            lim.append((sel, _rect(lo[0], hi[0], lo[1], hi[1])))
    for _ in range(tries):
        dx, dy, yaw = float(r.uniform(-g, g)), float(r.uniform(-g, g)), float(r.uniform(-a, a))
        R = _rotz(math.radians(yaw))[:2, :2]
        if all(not _overlap(_hull2(Q @ R.T + np.array([dx, dy])), box) for sel, box in lim for Q in sel):
            return dx, dy, yaw
    return 0.0, 0.0, 0.0


def _nest_play(seed, unit, polys, nominal, tries=200):
    """(dx, dy mm, dyaw deg) of a unit in its nest: +-NEST_PLAY_MM, +-NEST_PLAY_DEG (about the nest's centre), the plan
    hulls of its pieces below the ribs' top (tray frame, at the nest's nominal pose, mm) clear of the ribs"""
    if nominal:
        return 0.0, 0.0, 0.0
    r = _rng(seed, 0x4E57 + (1 if unit == "LDR" else 0))
    g, a = FX.NEST_PLAY_MM, FX.NEST_PLAY_DEG
    x0, y0, _yaw = FX.NEST_POSE[unit]
    pre = "cmp_" if unit == "CMP" else "ldr_"
    ribs = [_rect(lo[0], hi[0], lo[1], hi[1]) for bn, lo, hi, _p in FX.TRAY["boxes"] if bn.startswith(pre)]
    C = np.array([x0, y0])
    for _ in range(tries):
        dx, dy, yaw = float(r.uniform(-g, g)), float(r.uniform(-g, g)), float(r.uniform(-a, a))
        R = _rotz(math.radians(yaw))[:2, :2]
        if all(not _overlap((H - C) @ R.T + C + np.array([dx, dy]), rb) for H in polys for rb in ribs):
            return dx, dy, yaw
    return 0.0, 0.0, 0.0


def _fixture_units(seed, lay, L, geo, nominal):
    """the deck on the shelf, the compute module and the LiDAR in the tray (world poses), and their plays"""
    parts, plays = {}, {}
    nom = FX.nominal_unit_poses()
    lift = float(L["settle_lift_m"])
    for u in FIXTURE_UNITS:
        fxn, Tn = nom[u]
        if u == "DCK":
            dx, dy, yaw = _deck_play(seed, L, geo["pieces"]["DCK"], nominal)
            J = _T(_rotz(math.radians(yaw)), [dx * 1e-3, dy * 1e-3, 0.0])
            Tu = T_world_fixture(lay["fixtures"]["shelf"]) @ J @ Tn
        else:
            x0, y0, yaw0 = FX.NEST_POSE[u]
            polys = _low_polys([q * 1e3 for q in geo["pieces"][u]], _rotz(math.radians(yaw0)), np.array([x0, y0, 0.0]), FX.RIB_MM)
            dx, dy, yaw = _nest_play(seed, u, polys, nominal)
            # the turn about the nest's centre (the unit's origin), then the shift
            J = _T(None, [x0 * 1e-3, y0 * 1e-3, 0.0]) @ _T(_rotz(math.radians(yaw)), [dx * 1e-3, dy * 1e-3, 0.0]) @ \
                _T(None, [-x0 * 1e-3, -y0 * 1e-3, 0.0])
            Tu = T_world_fixture(lay["fixtures"]["tray"]) @ J @ Tn
        Tu[2, 3] += lift
        parts[u] = ([float(v) for v in Tu[:3, 3]], [float(v) for v in _quat_wxyz(Tu[:3, :3])])
        plays[u] = {"dx_mm": dx, "dy_mm": dy, "dyaw_deg": yaw}
    return parts, plays


def _unit_ok(u, Ph, lay_obs, placed, L, lanes):
    """the footprint (2D hull, world) of a bench unit: on the bench, its centre in the ring, clear of the arm's base,
    the fixtures, the lanes and the units placed before it. (ok, why)"""
    (bx0, bx1), (by0, by1) = L["bench_xy"]
    if Ph[:, 0].min() < bx0 or Ph[:, 0].max() > bx1 or Ph[:, 1].min() < by0 or Ph[:, 1].max() > by1:
        return False, "bench"
    c = 0.5 * (Ph.min(0) + Ph.max(0))
    rc = math.hypot(float(c[0]), float(c[1]))
    r0, r1 = (float(v) for v in L["reach_m"])
    if rc < r0 or rc > r1:
        return False, "ring"
    if float(np.min(np.hypot(Ph[:, 0], Ph[:, 1]))) < float(L["base_clear_m"]):
        return False, "base"
    gf = float(L["fixture_gap_m"])
    for name, polys in lay_obs.items():
        for P, _h in polys:
            if poly_dist(Ph, _hull2(P)) < gf:
                return False, "fixture:" + name
    for R in lanes:
        if _overlap(Ph, _hull2(R)):
            return False, "lane"
    g = float(L["bench_jitter"]["gap_m"])
    for v, Q in placed:
        if poly_dist(Ph, Q) < g:
            return False, "gap:" + v
    return True, ""


def episode_layout(seed, plan, kit, uv=None, cfg=None, stratum=None, nominal=False):
    """the episode's layout (module docstring). `plan` = the hidden bench plan (its "rest" block, or the whole file),
    `uv` = unit_geometry(kit) (computed when None), `stratum` one of STRATA (None: drawn from the seed)"""
    L = _merge(cfg)
    if "rest" in plan:
        oracle = plan.get("oracle") or {}
        rest = plan["rest"]
    else:
        oracle, rest = {}, plan
    geo = uv if uv is not None else unit_geometry(kit)
    hulls = geo["hulls"]
    if stratum is None:
        rs = _rng(seed, 0x5157)
        stratum = STRATA[int(rs.integers(0, len(STRATA)))]
    sign, lead = parse_stratum(stratum)
    stand, clamp = _stand_pose(seed, L, sign, nominal)
    lay = {"format": "racecar-asm-layout/1", "seed": int(seed), "stratum": stratum, "nominal": bool(nominal),
           "stand": stand, "clamp": clamp, "lead_end_deg": lead,
           "fixtures": {"stand": stand, "shelf": _fixture_pose(seed, L, "shelf", 0x5E1F, nominal),
                        "tray": _fixture_pose(seed, L, "tray", 0x7A40, nominal)}}
    if nominal:
        hd = oracle.get("hub_deg") or {}
        lay["hub_deg"] = {h: float(hd.get(h, 0.0)) for h in HUBS}
    else:
        rh = _rng(seed, 0x40B5)
        a0, a1 = (float(v) for v in L["hub_start_deg"])
        lay["hub_deg"] = {h: float(rh.uniform(a0, a1)) for h in HUBS}
    parts, plays = _fixture_units(seed, lay, L, geo, nominal)
    lay["plays"] = plays
    lay["parts"] = dict(parts)
    obs = obstacles(lay, kit, hulls)
    lanes = lanes_world(lay, L)
    names = [u for u in BENCH_UNITS if u in rest]
    lift = float(L["settle_lift_m"])
    base = {}
    for u in names:
        p0 = np.asarray(rest[u]["pos"], float)
        R0 = _quat_R(rest[u]["quat"])
        if u == "BAT" and lead:
            # the lead end turns the pack about the vertical through its footprint's centre
            fp = (hulls[u] @ R0.T)[:, :2]
            c0 = 0.5 * (fp.min(0) + fp.max(0))
            R0 = _rotz(math.radians(lead)) @ R0
            P = (hulls[u] @ R0.T)[:, :2]
            p0 = p0.copy()
            p0[:2] += c0 - 0.5 * (P.min(0) + P.max(0))
        base[u] = (p0, R0)
    if nominal:
        for u in names:
            p0, R0 = base[u]
            lay["parts"][u] = ([float(v) for v in p0], [float(v) for v in _quat_wxyz(R0)])
        lay["jitter"] = {"amp": 0.0, "layouts_tried": 0}
        return lay
    bj = L["bench_jitter"]
    rng = _rng(seed, 0xB3C4)
    amp = 1.0
    nl = int(bj["layouts"])
    for attempt in range(4 * nl):
        if attempt and attempt % nl == 0:
            amp *= 0.5
        order = [names[i] for i in rng.permutation(len(names))]
        placed, out, failed = [], {}, False
        for u in order:
            p0, R0 = base[u]
            P0 = hulls[u] @ R0.T + p0
            c = np.array([0.5 * (P0[:, 0].min() + P0[:, 0].max()), 0.5 * (P0[:, 1].min() + P0[:, 1].max()), 0.0])
            off = p0 - c
            choice = None
            for _ in range(int(bj["tries"])):
                psi = math.radians(float(bj["yaw_deg"])) * float(rng.uniform(-1, 1)) * amp
                d = np.array([float(rng.uniform(-1, 1)), float(rng.uniform(-1, 1)), 0.0]) * float(bj["xy_m"]) * amp
                Rz = _rotz(psi)
                p = c + Rz @ off + d
                p[2] = p0[2]
                R = Rz @ R0
                Ph = _hull2(hulls[u] @ R.T + p)
                ok, _why = _unit_ok(u, Ph, obs, placed, L, lanes)
                if ok:
                    choice = (R, p, Ph)
                    break
            if choice is None:
                failed = True
                break
            R, p, Ph = choice
            placed.append((u, Ph))
            out[u] = ([float(v) for v in p], [float(v) for v in _quat_wxyz(R)])
        if not failed:
            for u in names:
                lay["parts"][u] = out[u]
            lay["jitter"] = {"amp": amp, "layouts_tried": attempt + 1}
            return lay
    for u in names:
        p0, R0 = base[u]
        lay["parts"][u] = ([float(v) for v in p0], [float(v) for v in _quat_wxyz(R0)])
    lay["jitter"] = {"amp": 0.0, "layouts_tried": -1}
    return lay


# ------------------------------------------------------------------ the fingerprint
def _canon(x, nd):
    if isinstance(x, dict):
        return {k: _canon(v, nd) for k, v in sorted(x.items())}
    if isinstance(x, (list, tuple)):
        return [_canon(v, nd) for v in x]
    if isinstance(x, (float, np.floating)):
        v = round(float(x), nd)
        return 0.0 if v == 0 else v
    if isinstance(x, (np.integer,)):
        return int(x)
    return x


def layout_digest(lay):
    """sha256 (hex) of the layout's physical content (stand, clamp, fixtures, plays, parts, hubs, stratum, lead end),
    floats rounded to 1e-7 (m, quaternion components, deg): the same seed, plan, kit and config give the same digest in
    any process"""
    keys = ("format", "stratum", "nominal", "stand", "clamp", "fixtures", "plays", "parts", "hub_deg", "lead_end_deg")
    body = {k: lay.get(k) for k in keys}
    s = json.dumps(_canon(body, 7), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def layout_digest_of(seed, plan, kit, uv=None, cfg=None, stratum=None, nominal=False):
    return layout_digest(episode_layout(seed, plan, kit, uv=uv, cfg=cfg, stratum=stratum, nominal=nominal))


# ------------------------------------------------------------------ the layout check
def check_layout(lay, kit, uv=None, cfg=None):
    """(ok, problems, measures) of a drawn layout: every bench unit on the bench, in the ring, clear of the arm's base,
    >= fixture_gap_m from every fixture, >= gap_m from every other bench unit, out of the lanes; the fixtures on the
    bench and apart; nothing taller than lanes.height_m in a lane (the stand and the car excepted: the lanes are
    beside it); the deck and the nested units inside their plays; the stratum's stand-yaw sign and lead end"""
    L = _merge(cfg)
    geo = uv if uv is not None else unit_geometry(kit)
    hulls = geo["hulls"]
    probs = []
    meas = {"unit_gap_min_m": None, "fixture_gap_min_m": None, "ring_m": {}, "lane_clear_min_m": None}
    obs = obstacles(lay, kit, hulls)
    lanes = lanes_world(lay, L)
    (bx0, bx1), (by0, by1) = L["bench_xy"]
    fps = {}
    for u in BENCH_UNITS:
        if u not in lay["parts"]:
            continue
        pos, quat = lay["parts"][u]
        P = hulls[u] @ _quat_R(quat).T + np.asarray(pos, float)
        fps[u] = (_hull2(P), P)
    for u, (Ph, P) in fps.items():
        if Ph[:, 0].min() < bx0 or Ph[:, 0].max() > bx1 or Ph[:, 1].min() < by0 or Ph[:, 1].max() > by1:
            probs.append("%s off the bench" % u)
        c = 0.5 * (Ph.min(0) + Ph.max(0))
        rc = math.hypot(float(c[0]), float(c[1]))
        meas["ring_m"][u] = round(rc, 4)
        if not (L["reach_m"][0] <= rc <= L["reach_m"][1]):
            probs.append("%s outside the reach ring (%.3f m)" % (u, rc))
        if float(np.min(np.hypot(Ph[:, 0], Ph[:, 1]))) < float(L["base_clear_m"]):
            probs.append("%s within %.2f m of the arm's base axis" % (u, L["base_clear_m"]))
        for name, polys in obs.items():
            for Q, _h in polys:
                dd = poly_dist(Ph, _hull2(Q))
                meas["fixture_gap_min_m"] = dd if meas["fixture_gap_min_m"] is None else min(meas["fixture_gap_min_m"], dd)
                if dd < float(L["fixture_gap_m"]) - 1e-9:
                    probs.append("%s %.0f mm from the %s" % (u, dd * 1e3, name))
        for R in lanes:
            Rh = _hull2(R)
            dl = poly_dist(Ph, Rh)
            meas["lane_clear_min_m"] = dl if meas["lane_clear_min_m"] is None else min(meas["lane_clear_min_m"], dl)
            if _overlap(Ph, Rh):
                probs.append("%s in a wheel lane" % u)
        # resting: the lowest collision point on the bench top
        zmin = float(P[:, 2].min())
        if not (-1e-6 <= zmin <= 0.002):
            probs.append("%s lowest point %.2f mm above the bench" % (u, zmin * 1e3))
    us = list(fps)
    for i in range(len(us)):
        for j in range(i + 1, len(us)):
            dd = poly_dist(fps[us[i]][0], fps[us[j]][0])
            meas["unit_gap_min_m"] = dd if meas["unit_gap_min_m"] is None else min(meas["unit_gap_min_m"], dd)
            if dd < float(L["bench_jitter"]["gap_m"]) - 1e-9:
                probs.append("%s and %s %.0f mm apart" % (us[i], us[j], dd * 1e3))
    # the fixtures: on the bench, apart from each other; tall ones out of the lanes
    groups = {k: v for k, v in obs.items()}
    for name, polys in groups.items():
        for Q, h in polys:
            if Q[:, 0].min() < bx0 or Q[:, 0].max() > bx1 or Q[:, 1].min() < by0 or Q[:, 1].max() > by1:
                probs.append("%s off the bench" % name)
            if name != "stand" and h > float(L["lanes"]["height_m"]):
                for R in lanes:
                    if _overlap(_hull2(Q), _hull2(R)):
                        probs.append("%s (%.2f m tall) in a wheel lane" % (name, h))
    names = list(groups)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            dmin = min(poly_dist(_hull2(A), _hull2(B)) for A, _ in groups[names[i]] for B, _ in groups[names[j]])
            if dmin <= 0.0:
                probs.append("%s overlaps %s" % (names[i], names[j]))
    # plays and strata
    for u, pl in (lay.get("plays") or {}).items():
        lim = FX.SHELF_PLAY_MM if u == "DCK" else FX.NEST_PLAY_MM
        if abs(pl["dx_mm"]) > lim + 1e-9 or abs(pl["dy_mm"]) > lim + 1e-9:
            probs.append("%s outside its play" % u)
        if u != "DCK" and abs(pl["dyaw_deg"]) > FX.NEST_PLAY_DEG + 1e-9:
            probs.append("%s turned beyond its play" % u)
    sign, lead = parse_stratum(lay["stratum"])
    dyaw = float(lay["fixtures"]["stand"]["yaw_deg"]) - float(L["stand"]["yaw_deg"])
    if not lay.get("nominal") and (dyaw * sign <= 0 or abs(dyaw) > float(L["stand"]["jitter_yaw_deg"]) + 1e-9):
        probs.append("stand yaw %.3f deg against the stratum %s" % (dyaw, lay["stratum"]))
    if int(lay.get("lead_end_deg", -1)) != lead:
        probs.append("lead end %s against the stratum %s" % (lay.get("lead_end_deg"), lay["stratum"]))
    return not probs, probs, meas
