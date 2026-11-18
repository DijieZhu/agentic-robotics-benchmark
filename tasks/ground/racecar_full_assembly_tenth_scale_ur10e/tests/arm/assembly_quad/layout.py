"""Per-episode physical draws of the quadruped cell: where the stand stands, where each loose unit rests on the bench,
and the output pulleys' start angles (grader side; the episode seed and the hidden bench plan in, a layout out).

`episode_layout(seed, plan, kit, uv, cfg, nominal=False)` -> {"stand": {"xy", "yaw_deg"}, "parts": {unit: (pos, quat
wxyz)}, "rotor_rad": {rotor body: angle}, "jitter": {...}}, a pure function of its inputs:

* the stand at its nominal place (cfg["stand"]["nominal_xy"], yaw cfg["stand"]["yaw_deg"]) moved by +-jitter_m per
  axis, never turned;
* every unit of the bench plan (`plan` = the hidden plan's "rest": {unit: {pos, quat, ...}}) in its planned resting
  orientation, turned about the vertical through its footprint's centre and moved: shoulders, supports and upper legs
  by +-xy_m / +-yaw_deg, the lower legs lying at the bench's side edges (the knee made over the edge) along the edge by
  +-edge_along_m, across it by +-edge_across_m and turned by +-edge_yaw_deg; redrawn until its footprint (the collision
  vertices of every body of the unit, turned) lies on the bench (a lower leg may reach `edge_overhang_m` past a side
  edge, its centre of mass on the bench), its footprint's centre in the UR's reach annulus and no corner inside
  `base_clear_m` of the UR's axis, the footprint clear of the stand's own footprint by `stand_clear_m` and `gap_m` clear
  of every unit placed before it; in a seeded random order; a unit with no valid draw in `tries` restarts the layout,
  the amplitudes halve every `layouts` failed layouts, the plan itself last. Whether the robot can still pick every
  unit and make every knee there is the grader's reach check on the built cell (server.py), not a geometric rule;
* the 12 output pulleys (the kit bodies with a hinge) at their index +-rotor_jitter_deg.

nominal=True (the privileged oracle): the plan itself, the stand at its nominal place, the pulleys as `plan_oracle`
gives them (the reference run's own draw) or at the index.
"""
import math
import os
import struct

import numpy as np

from .coupling import quat_wxyz

__all__ = ["unit_vertices", "footprint", "episode_layout", "DEFAULT_LAYOUT"]

DEFAULT_LAYOUT = {
    "stand": {"nominal_xy": [0.62, 0.0], "yaw_deg": 90.0, "jitter_m": 0.01},
    "bench_jitter": {"xy_m": 0.03, "yaw_deg": 30.0, "edge_along_m": 0.03, "edge_across_m": 0.01, "edge_yaw_deg": 10.0,
                     "gap_m": 0.02, "tries": 200, "layouts": 50},
    "rotor_jitter_deg": 2.0,
    # the bench top (2.0 x 1.4 m centred at (0.45, 0)) less 2 mm; the UR's reach annulus about its base axis (the bench
    # plan's spots, bench_plan.REACH_M, with 30 mm of slack) for a footprint's centre, the UR's base column for its
    # corners; the stand's feet (in its own frame) and the clearance kept from them; the side edges' overhang
    "bench_xy": [[-0.548, 1.448], [-0.698, 0.698]],
    "reach_m": [0.35, 0.95],
    "base_clear_m": 0.22,
    "stand_half_xy": [0.12, 0.10],
    "stand_clear_m": 0.05,
    "edge_overhang_m": 0.03,
    "edge_units": ["L1", "L2", "L3", "L4"],
}


def _stl_vertices(path):
    """the vertices of a binary or ASCII STL (unique rows)"""
    with open(path, "rb") as f:
        data = f.read()
    if len(data) >= 84:
        n = struct.unpack("<I", data[80:84])[0]
        if 84 + 50 * n == len(data):
            rec = np.frombuffer(data, dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]), count=n, offset=84)
            return np.unique(rec["v"].reshape(-1, 3).astype(np.float64), axis=0)
    pts = [[float(x) for x in ln.split()[1:4]] for ln in data.decode("ascii", "replace").splitlines() if ln.strip().startswith("vertex")]
    return np.unique(np.asarray(pts, np.float64).reshape(-1, 3), axis=0)


def unit_vertices(kit, max_per_unit=4000, seed=0):
    """{unit: (N, 3)} the collision vertices of every body of each loose unit in the unit's root frame (the bodies placed
    by the kit's T_design chain, as the cell builds them), subsampled to max_per_unit"""
    bodies = kit["bodies"]
    rng = np.random.default_rng(seed)
    out = {}
    for u in kit["loose"]:
        Tu = np.asarray(bodies[u]["T_design"], float)
        Tui = np.linalg.inv(Tu)
        pts = []
        for bn, bb in bodies.items():
            if bb.get("unit") != u:
                continue
            Tb = Tui @ np.asarray(bb["T_design"], float)
            for f in bb.get("collision", []):
                v = _stl_vertices(os.path.join(kit["dir"], f))
                pts.append(v @ Tb[:3, :3].T + Tb[:3, 3])
        V = np.concatenate(pts)
        if len(V) > max_per_unit:
            V = V[rng.choice(len(V), max_per_unit, replace=False)]
        out[u] = V
    return out


def _quat_R(q):
    w, x, y, z = (float(v) for v in q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def _rotz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def footprint(V, R, p):
    """(xmin, xmax, ymin, ymax) of the vertices V (unit frame) placed at rotation R, position p"""
    P = (V @ np.asarray(R, float).T)[:, :2] + np.asarray(p, float)[:2]
    return (float(P[:, 0].min()), float(P[:, 0].max()), float(P[:, 1].min()), float(P[:, 1].max()))


def _com(kit, u):
    """the unit's centre of mass in its root frame (its bodies' masses and com_m placed by T_design)"""
    bodies = kit["bodies"]
    Tui = np.linalg.inv(np.asarray(bodies[u]["T_design"], float))
    mass, mom = 0.0, np.zeros(3)
    for bn, bb in bodies.items():
        if bb.get("unit") != u or float(bb.get("mass_kg", 0.0)) <= 0.0:
            continue
        Tb = Tui @ np.asarray(bb["T_design"], float)
        c = Tb[:3, :3] @ np.asarray(bb["com_m"], float) + Tb[:3, 3]
        mass += float(bb["mass_kg"])
        mom += float(bb["mass_kg"]) * c
    return mom / mass if mass > 0 else np.zeros(3)


def _rng(seed, tag):
    return np.random.default_rng([int(seed) & 0xFFFFFFFF, (int(seed) >> 32) & 0xFFFFFFFF, tag])


def _merge(cfg):
    out = {k: (dict(v) if isinstance(v, dict) else v) for k, v in DEFAULT_LAYOUT.items()}
    for k, v in (cfg or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k].update(v)
        else:
            out[k] = v
    return out


def stand_pose(seed, cfg=None, nominal=False):
    c = _merge(cfg)["stand"]
    x, y = (float(v) for v in c["nominal_xy"])
    if not nominal and float(c["jitter_m"]) > 0:
        r = _rng(seed, 0xF1C7)
        j = float(c["jitter_m"])
        x, y = x + float(r.uniform(-j, j)), y + float(r.uniform(-j, j))
    return {"xy": [x, y], "yaw_deg": float(c["yaw_deg"])}


def _stand_box(stand, L):
    """the world AABB of the stand's feet grown by stand_clear_m"""
    a = math.radians(stand["yaw_deg"])
    hx, hy = (float(v) for v in L["stand_half_xy"])
    c, s_ = abs(math.cos(a)), abs(math.sin(a))
    ex, ey = c * hx + s_ * hy, s_ * hx + c * hy
    g = float(L["stand_clear_m"])
    x, y = (float(v) for v in stand["xy"])
    return (x - ex - g, x + ex + g, y - ey - g, y + ey + g)


def _ok(u, fp, com_xy, stand, placed, L, gap):
    (bx0, bx1), (by0, by1) = L["bench_xy"]
    edge = u in L["edge_units"]
    oh = float(L["edge_overhang_m"]) if edge else 0.0
    if fp[0] < bx0 or fp[1] > bx1 or fp[2] < by0 - oh or fp[3] > by1 + oh:
        return False
    if edge and not (bx0 <= com_xy[0] <= bx1 and by0 <= com_xy[1] <= by1):
        return False
    r0, r1 = (float(v) for v in L["reach_m"])
    rc = math.hypot(0.5 * (fp[0] + fp[1]), 0.5 * (max(fp[2], by0) + min(fp[3], by1)))
    if rc < r0 or rc > r1:
        return False
    if min(math.hypot(x, y) for x in (fp[0], fp[1]) for y in (fp[2], fp[3])) < float(L["base_clear_m"]):
        return False
    sb = _stand_box(stand, L)
    if fp[0] < sb[1] and fp[1] > sb[0] and fp[2] < sb[3] and fp[3] > sb[2]:
        return False
    return not any(fp[0] < q[1] + gap and fp[1] > q[0] - gap and fp[2] < q[3] + gap and fp[3] > q[2] - gap for q in placed)


def episode_layout(seed, plan, kit, uv, cfg=None, nominal=False, oracle=None):
    """the episode's {"stand", "parts", "rotor_rad", "jitter"} (module docstring); `plan` = the hidden bench plan's
    "rest", `uv` = unit_vertices(kit), `oracle` = the plan's "oracle" block (the privileged run's stand / pulleys)"""
    L = _merge(cfg)
    stand = stand_pose(seed, L, nominal=nominal)
    if nominal and oracle and oracle.get("stand"):
        stand = {"xy": [float(v) for v in oracle["stand"]["xy"]], "yaw_deg": float(oracle["stand"]["yaw_deg"])}
    rotors = [bn for bn, bb in kit["bodies"].items() if bb.get("joint")]
    if nominal:
        rr = (oracle or {}).get("rotor_rad") or {}
        rotor = {bn: float(rr.get(bn, 0.0)) for bn in rotors}
    else:
        r = _rng(seed, 0x2070)
        j = math.radians(float(L["rotor_jitter_deg"]))
        rotor = {bn: float(r.uniform(-j, j)) for bn in rotors}
    names = [u for u in kit["loose"] if u in plan]
    base = {u: (np.asarray(plan[u]["pos"], float), _quat_R(plan[u]["quat"])) for u in names}
    bj = L["bench_jitter"]
    if nominal:
        parts = {u: ([float(v) for v in base[u][0]], [float(v) for v in plan[u]["quat"]]) for u in names}
        return {"stand": stand, "parts": parts, "rotor_rad": rotor, "jitter": {"nominal": True, "amp": 0.0, "layouts_tried": 0}}
    com = {u: _com(kit, u) for u in names}
    rng = _rng(seed, 0xB3C4)
    gap = float(bj["gap_m"])
    amp = 1.0
    for attempt in range(4 * int(bj["layouts"])):
        if attempt and attempt % int(bj["layouts"]) == 0:
            amp *= 0.5
        order = [names[i] for i in rng.permutation(len(names))]
        placed, out, failed = [], {}, False
        for u in order:
            p0, R0 = base[u]
            fp0 = footprint(uv[u], R0, p0)
            c = np.array([0.5 * (fp0[0] + fp0[1]), 0.5 * (fp0[2] + fp0[3]), 0.0])
            off = p0 - c
            edge = u in L["edge_units"]
            choice = None
            for _ in range(int(bj["tries"])):
                if edge:
                    psi = math.radians(float(bj["edge_yaw_deg"])) * float(rng.uniform(-1, 1)) * amp
                    d = np.array([float(rng.uniform(-1, 1)) * float(bj["edge_along_m"]), float(rng.uniform(-1, 1)) * float(bj["edge_across_m"]), 0.0]) * amp
                else:
                    psi = math.radians(float(bj["yaw_deg"])) * float(rng.uniform(-1, 1)) * amp
                    d = np.array([float(rng.uniform(-1, 1)), float(rng.uniform(-1, 1)), 0.0]) * float(bj["xy_m"]) * amp
                Rz = _rotz(psi)
                p = c + Rz @ off + d
                p[2] = p0[2]
                R = Rz @ R0
                fp = footprint(uv[u], R, p)
                cm = R @ com[u] + p
                if _ok(u, fp, cm[:2], stand, placed, L, gap):
                    choice = (R, p, fp)
                    break
            if choice is None:
                failed = True
                break
            R, p, fp = choice
            placed.append(fp)
            out[u] = ([float(v) for v in p], [float(v) for v in quat_wxyz(R)])
        if not failed:
            return {"stand": stand, "parts": {u: out[u] for u in names}, "rotor_rad": rotor,
                    "jitter": {"nominal": False, "amp": amp, "layouts_tried": attempt + 1}}
    parts = {u: ([float(v) for v in base[u][0]], [float(v) for v in plan[u]["quat"]]) for u in names}
    return {"stand": stand, "parts": parts, "rotor_rad": rotor, "jitter": {"nominal": False, "amp": 0.0, "layouts_tried": -1}}


def check_layout(lay, kit, uv, cfg=None):
    """the constraints of a drawn layout: (ok, problems) (the dev check over many seeds)"""
    L = _merge(cfg)
    placed, probs = [], []
    for u, (pos, quat) in lay["parts"].items():
        R = _quat_R(quat)
        fp = footprint(uv[u], R, pos)
        cm = R @ _com(kit, u) + np.asarray(pos, float)
        if not _ok(u, fp, cm[:2], lay["stand"], [], L, 0.0):
            probs.append("%s: bench / reach / stand zone" % u)
        for v, q in placed:
            g = float(L["bench_jitter"]["gap_m"])
            if fp[0] < q[1] + g and fp[1] > q[0] - g and fp[2] < q[3] + g and fp[3] > q[2] - g:
                probs.append("%s within %.0f mm of %s" % (u, g * 1e3, v))
        placed.append((u, fp))
    return not probs, probs
