"""predicates.py -- what counts, by geometry (freeflyer_uncrewed_station_caretaking; the geometric
predicates of the scoring rules).

Two layers, one definition each:

1. Geometry (numpy, imported inside the functions): what the cell measures into the record every
   ``cell.record_period_s`` -- the sealed area of every item on every suction face (a return inlet, a holding screen
   or a pouch pad), the covered fraction of an inlet, and 'in the gripper'.  The grader never calls these; the cell
   does, so the record and the rules share one measure.
     sealed area   = the projected area, inside the face's opening outline, of the item surface that faces the
                     suction face and lies within the gap band of the face plane, counting only the layer that
                     faces the face directly (an item lying on another item is shadowed).  Measured by
                     casting a ray from every cell of a raster of the outline along the face normal: the first item
                     face it meets owns the cell when the gap is inside the band.  Projected areas, so the covered
                     fraction of an inlet is the union of the items on it (never more than the opening).
     in the gripper = a grasp bar of the item inside the closed finger cage (palm frame box) at one of 5 points
                     along the bar, and the tendon commanded closed (finger-pad contacts are not counted).
2. Record level (standard library only): the sample, window, run and gate helpers that grade.py applies to the
   record.  Missing or non-finite inputs never raise the score: every reader has a conservative default.

Thresholds are never literals here: callers pass the tests/grader_config.json values (``measure_params``).

Measures: the longest clear run (longest_run), the fan-stop intervals, the evidence check (evidence_ok), loss of
control at the record rate, the bar in the cage (5 points along the bar), the covered area (strip_metrics: the back
face within the gap band and inside the opening, as the projected raster area, exact to the raster cell, of the
first layer only; the gap band is symmetric: distance <= max gap on either side of the face plane), caged /
in_gripper (the cage box with the plane condition |z - z_plane|), conservative readers for non-finite values.
"""
from __future__ import annotations

import math
import re

INF = float("inf")
EPS = 1e-9

_RX_INLET = re.compile(r"^ri_(l2|h2|l1|l3)_[1-3]$")
_RX_SCREEN = re.compile(r"^SB-(L2|H2|L1|L3)-S$")
_RX_PAD = re.compile(r"^SB-(L2|H2|L1|L3)-P[12]$")
_RX_REPORT_INLET = re.compile(r"^RI-(L2|H2|L1|L3)-([1-3])$")
FACE_KINDS = ("return_inlet", "holding_screen", "pouch_pad")


# ------------------------------------------------------------------------------------------------ parameters
def measure_params(values: dict) -> dict:
    """The measurement definitions of the grader_config values (the cell measures the record with them)."""
    g = float(values["grader.seal.max_gap_mm"]) / 1000.0
    cv = values["grader.covered"]
    gc = float(cv["max_gap_m"])
    lc = values["grader.loss_of_control"]
    return dict(seal_band=(-g, g), cover_band=(-gc, gc), opening_area_m2=float(cv["opening_area_m2"]),
                cage_mm=dict(values["grader.in_gripper_cage_mm"]), loss_disp_m=float(lc["disp_m"]), loss_min_s=float(lc["min_s"]))


# ------------------------------------------------------------------------------------------------ 1. geometry
def _get(obj, key):
    return obj[key] if isinstance(obj, dict) else getattr(obj, key)


def face_frame(face):
    """(centre, normal, t1, t2, (half_1, half_2)) of a suction face given as a dict {c, n, t1, [t2], half} or an object
    with those attributes (spacekit.suction.Inlet works as is): n points into the cabin, t1 is the outline's first
    axis, t2 = n x t1 (the sign of a given t2 is kept)."""
    import numpy as np
    c = np.asarray(_get(face, "c"), float)
    n = np.asarray(_get(face, "n"), float)
    n = n / np.linalg.norm(n)
    t1 = np.asarray(_get(face, "t1"), float)
    t1 = t1 - (t1 @ n) * n
    t1 = t1 / np.linalg.norm(t1)
    t2 = np.cross(n, t1)
    given = face.get("t2") if isinstance(face, dict) else getattr(face, "t2", None)
    if given is not None and float(np.asarray(given, float) @ t2) < 0.0:
        t2 = -t2
    h = _get(face, "half")
    return c, n, t1, t2, (float(h[0]), float(h[1]))


def seal_areas(face, boxes, band, cell: float = 0.0025) -> dict:
    """First-layer sealed area [m2] per item on one suction face.

    face: see ``face_frame``.  boxes: iterable of (item_id, pos (3,), R (3, 3) body-to-world with the box axes as
    columns (MuJoCo's xmat reshaped), half sizes (3,)); an item may own several boxes (the strips of a sheet).  Only
    the two large faces (local +-z) are surfaces of the item (the suction model's convention); the one whose outward
    normal opposes the face normal faces the suction face.  band: (lo, hi) of the gap [m] (negative = behind the face
    plane).  cell: raster cell [m] (a measurement resolution, not a threshold).  Items without area are omitted."""
    import numpy as np
    c, n, t1, t2, (ha, hb) = face_frame(face)
    lo, hi = float(band[0]), float(band[1])
    nu = max(1, int(math.ceil(2.0 * ha / cell - EPS)))
    nv = max(1, int(math.ceil(2.0 * hb / cell - EPS)))
    du, dv = 2.0 * ha / nu, 2.0 * hb / nv
    us = -ha + (np.arange(nu) + 0.5) * du
    vs = -hb + (np.arange(nv) + 0.5) * dv
    X0 = (c[None, None, :] + us[:, None, None] * t1[None, None, :] + vs[None, :, None] * t2[None, None, :]).reshape(-1, 3)
    best = np.full(len(X0), INF)
    owner = np.full(len(X0), -1, int)
    ids = []
    for k, (iid, pos, R, half) in enumerate(boxes):
        ids.append(iid)
        pos = np.asarray(pos, float)
        R = np.asarray(R, float).reshape(3, 3)
        hx, hy, hz = (float(v) for v in half)
        r = math.sqrt(hx * hx + hy * hy + hz * hz)
        d = pos - c
        dn = float(d @ n)
        if dn - r > hi or dn + r < lo or abs(float(d @ t1)) > ha + r or abs(float(d @ t2)) > hb + r:
            continue
        z = R[:, 2]
        dzn = float(z @ n)
        if abs(dzn) < EPS:
            continue                                   # edge-on: no projected area
        s = -1.0 if dzn > 0.0 else 1.0                 # the large face whose outward normal opposes n
        m = s * z
        P0 = pos + s * hz * z
        t = ((P0[None, :] - X0) @ m) / float(n @ m)
        rel = X0 + t[:, None] * n[None, :] - pos[None, :]
        hit = (np.abs(rel @ R[:, 0]) <= hx) & (np.abs(rel @ R[:, 1]) <= hy) & (t >= lo)
        better = hit & (t < best)
        best[better] = t[better]
        owner[better] = k
    sealed = (owner >= 0) & (best <= hi)
    out = {}
    if sealed.any():
        cnt = np.bincount(owner[sealed], minlength=len(ids))
        for k, iid in enumerate(ids):
            if cnt[k]:
                out[iid] = out.get(iid, 0.0) + float(cnt[k]) * du * dv
    return out


def covered_fraction(face, boxes, band, opening_area_m2: float, cell: float = 0.0025) -> float:
    """Covered fraction of a return inlet: the union of the items' projected area within the band / the opening."""
    a = sum(seal_areas(face, boxes, band, cell).values())
    return min(1.0, a / float(opening_area_m2))


def bar_points(centre, axis, length: float, n: int = 5) -> list:
    """n points along a grasp bar (centre, unit axis, length) from -length / 2 to +length / 2."""
    c = [float(v) for v in centre]
    a = [float(v) for v in axis]
    na = math.sqrt(sum(v * v for v in a)) or 1.0
    a = [v / na for v in a]
    if n <= 1:
        return [tuple(c)]
    return [tuple(c[i] + a[i] * length * (k / (n - 1) - 0.5) for i in range(3)) for k in range(n)]


def _mat3(R) -> list:
    """3 x 3 rows of a nested 3 x 3 sequence (or numpy array) or of a flat 9-sequence (row-major, MuJoCo's xmat)."""
    try:
        flat = [float(v) for row in R for v in row]
    except TypeError:
        flat = [float(v) for v in R]
    return [flat[0:3], flat[3:6], flat[6:9]]


def to_palm(points, palm_pos, palm_R) -> list:
    """World points in the palm frame: R^T (p - palm_pos) (palm_R body-to-world, axes as columns)."""
    R = _mat3(palm_R)
    o = [float(v) for v in palm_pos]
    out = []
    for p in points:
        d = [float(p[i]) - o[i] for i in range(3)]
        out.append(tuple(sum(R[i][j] * d[i] for i in range(3)) for j in range(3)))
    return out


def bar_in_cage(points_palm, cage_mm: dict, z_plane_m: float) -> bool:
    """Some point of the bar inside the closed finger cage: palm x in cage x, |y| <= y_abs_max, |z - z_plane| <=
    plane_max (millimetres in the config, metres in the points)."""
    x0, x1 = (float(v) / 1000.0 for v in cage_mm["x"])
    ym = float(cage_mm["y_abs_max"]) / 1000.0
    zm = float(cage_mm["plane_max"]) / 1000.0
    for p in points_palm:
        x, y, z = (float(v) for v in p)
        if all(math.isfinite(v) for v in (x, y, z)) and x0 <= x <= x1 and abs(y) <= ym and abs(z - float(z_plane_m)) <= zm:
            return True
    return False


def in_gripper(bars_points_palm, tendon_closed: bool, cage_mm: dict, z_plane_m: float) -> bool:
    """'In the gripper': the tendon commanded closed and one of the item's bars in the cage."""
    return bool(tendon_closed) and any(bar_in_cage(b, cage_mm, z_plane_m) for b in bars_points_palm)


# ------------------------------------------------------------------------------------------------ 2. record level
def fnum(x, default=None):
    """x as a finite float, else ``default`` (bools are not numbers here)."""
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        return default
    x = float(x)
    return x if math.isfinite(x) else default


def face_kind(face_id) -> str | None:
    """Kind of a suction face from its contract id (the record's patterns)."""
    if not isinstance(face_id, str):
        return None
    if _RX_INLET.match(face_id):
        return "return_inlet"
    if _RX_SCREEN.match(face_id):
        return "holding_screen"
    if _RX_PAD.match(face_id):
        return "pouch_pad"
    return None


def inlet_from_report(s) -> str | None:
    """Internal inlet id of a reported id: 'RI-L1-2' (the agent's form) or 'ri_l1_2' -> 'ri_l1_2'; else None."""
    if not isinstance(s, str):
        return None
    t = s.strip()
    if _RX_INLET.match(t):
        return t
    m = _RX_REPORT_INLET.match(t.upper())
    return f"ri_{m.group(1).lower()}_{m.group(2)}" if m else None


def module_of_inlet(inlet: str) -> str | None:
    m = _RX_INLET.match(inlet or "")
    return m.group(1) if m else None


def samples(record) -> list:
    """The record's world-state samples with a finite time, sorted by time, one per time (the first kept)."""
    st = record.get("state") if isinstance(record, dict) else None
    out = [s for s in st if isinstance(s, dict) and fnum(s.get("t")) is not None] if isinstance(st, list) else []
    ts = [float(s["t"]) for s in out]
    if all(a < b for a, b in zip(ts, ts[1:])):
        return out
    uniq, last = [], None
    for i in sorted(range(len(out)), key=ts.__getitem__):
        if last is None or ts[i] > last:
            uniq.append(out[i])
            last = ts[i]
    return uniq


def item_state(s: dict, name: str) -> dict:
    it = s.get("items") if isinstance(s, dict) else None
    v = it.get(name) if isinstance(it, dict) else None
    return v if isinstance(v, dict) else {}


def cover(s: dict, inlet: str) -> float:
    """Covered fraction of an inlet in a sample; missing, non-finite or out-of-range values read as covered (1)."""
    inl = s.get("inlets") if isinstance(s, dict) else None
    v = fnum(inl.get(inlet)) if isinstance(inl, dict) else None
    return v if (v is not None and 0.0 <= v <= 1.0) else 1.0


def in_grip(st: dict) -> bool:
    return st.get("in_gripper") is True


def seals(st: dict) -> dict:
    """{face: sealed area >= 0} of an item state (non-finite or negative entries read as 0)."""
    sm = st.get("seal_m2")
    out = {}
    if isinstance(sm, dict):
        for f, a in sm.items():
            v = fnum(a, 0.0)
            out[f] = v if v > 0.0 else 0.0
    return out


def max_seal(st: dict, kind: str) -> tuple:
    """(largest sealed area on a face of ``kind``, that face id) of an item state."""
    best, face = 0.0, None
    for f, a in seals(st).items():
        if face_kind(f) == kind and a > best:
            best, face = a, f
    return best, face


def attached(st: dict) -> bool:
    """Lying on some suction face: any sealed area, or the cell's on_face set."""
    return any(a > 0.0 for a in seals(st).values()) or isinstance(st.get("on_face"), str)


def disp(st: dict) -> float:
    """Centroid displacement from t = 0; missing or non-finite reads as displaced (inf)."""
    v = fnum(st.get("disp_m"))
    return v if (v is not None and v >= 0.0) else INF


def window(smp: list, t_end: float, w: float) -> list:
    return [s for s in smp if t_end - w - 1e-6 <= float(s["t"]) <= t_end + 1e-6]


GAP_PERIODS = 1.5                       # a gap between samples longer than this many record periods is missing data


def window_complete(win: list, t_end: float, w: float, period: float) -> bool:
    """The samples of a window cover it: the first within one period of its start, the last within one period of
    its end, no gap longer than GAP_PERIODS periods (one missing sample is enough to void the window)."""
    if not win:
        return False
    ts = [float(s["t"]) for s in win]
    if ts[0] > t_end - w + period + 1e-6 or ts[-1] < t_end - period - 1e-6:
        return False
    return all(b - a <= GAP_PERIODS * period + 1e-6 for a, b in zip(ts, ts[1:]))


def holds_throughout(win: list, pred) -> bool:
    return bool(win) and all(pred(s) for s in win)


def longest_run(smp: list, pred, period: float) -> tuple:
    """(duration, start, end) of the longest run of consecutive samples where ``pred`` holds; a gap longer than
    GAP_PERIODS periods ends a run (missing data never extends a credit).  duration = last time - first time of the run."""
    best_d, best_a, best_b = -1.0, None, None
    start = prev = None
    for s in smp:
        t = float(s["t"])
        if pred(s):
            if start is None or t - prev > GAP_PERIODS * period + 1e-6:
                start = t
            prev = t
            if t - start > best_d:
                best_d, best_a, best_b = t - start, start, t
        else:
            start = prev = None
    return max(best_d, 0.0), best_a, best_b


def loss_of_control(smp: list, name: str, disp_m: float, min_s: float) -> float | None:
    """Start time of the item's first loss of control: centroid more than disp_m from its t = 0
    position for at least min_s, neither in the gripper nor on any suction face.  A gap in the samples does not end
    a run (missing data never removes a penalty).  None when it never happened."""
    start = None
    for s in smp:
        st = item_state(s, name)
        t = float(s["t"])
        if st:
            away = disp(st) > disp_m and not in_grip(st) and not attached(st)
        else:
            away = start is not None            # an item missing from a sample keeps a run going, starts none
        if away:
            if start is None:
                start = t
            if t - start >= min_s - 1e-9:
                return start
        else:
            start = None
    return None


def fan_stop_intervals(fan_stops, t_end: float) -> list:
    """[(unit, t_stop, t_restart)] clipped to [0, t_end]; an open stop runs to the end; a stop without a finite start
    reads as starting at mission second 0 (conservative)."""
    out = []
    if not isinstance(fan_stops, list):
        return out
    for e in fan_stops:
        if not isinstance(e, dict):
            continue
        u = e.get("unit") if isinstance(e.get("unit"), str) else "?"
        a = fnum(e.get("t_stop"))
        b = fnum(e.get("t_restart"))
        a = 0.0 if a is None else min(max(a, 0.0), t_end)
        b = t_end if b is None else min(max(b, a), t_end)
        out.append((u, a, b))
    return out


def fan_stop_total(fan_stops, t_end: float) -> float:
    return sum(b - a for _, a, b in fan_stop_intervals(fan_stops, t_end))


def hard_collisions(contacts, max_speed: float, max_force: float, force_classes) -> int:
    """New robot-station contacts over a threshold: normal speed over max_speed (any class), or force over max_force
    on a delicate class; non-finite speed or force counts (conservative)."""
    n = 0
    if not isinstance(contacts, list):
        return 0
    for c in contacts:
        if not isinstance(c, dict):
            continue
        v = fnum(c.get("speed_m_s"), INF)
        f = fnum(c.get("force_N"), INF)
        if v > max_speed or (c.get("other_class") in force_classes and f > max_force):
            n += 1
    return n


def evidence_ok(entry, E: dict, need_cover: bool) -> tuple:
    """(ok, why) of a frame's evidence entry for an inlet against grader.D.evidence."""
    if not isinstance(entry, dict):
        return False, "inlet not in the frame's evidence"
    if entry.get("in_frame") is not True:
        return False, "inlet centre not in the image"
    d = fnum(entry.get("dist_m"), INF)
    inc = fnum(entry.get("incidence_deg"), INF)
    px = fnum(entry.get("long_px"), -INF)
    occ = fnum(entry.get("occluded"), INF)
    if d > float(E["max_dist_m"]):
        return False, "too far"
    if inc > float(E["max_incidence_deg"]):
        return False, "incidence"
    if px < float(E["min_long_px"]):
        return False, "long side too short"
    if occ > float(E["max_occluded"]):
        return False, "occluded"
    if need_cover and not (fnum(entry.get("blocker_cover"), -INF) > float(E["min_blocker_cover"])):
        return False, "the blocker no longer covered the inlet at the shot"
    return True, "ok"


def linear_credit(x, full: float, zero: float) -> float:
    """1 at x <= full, 0 at x >= zero, linear between (lower is better); non-finite x gives 0."""
    v = fnum(x)
    if v is None:
        return 0.0
    if v <= full:
        return 1.0
    if v >= zero:
        return 0.0
    return (zero - v) / (zero - full)


def finite_metrics(m: dict) -> dict:
    """Only finite float values (the verdict's 'metrics' may hold nothing else)."""
    out = {}
    for k, v in m.items():
        f = fnum(v) if not isinstance(v, bool) else (1.0 if v else 0.0)
        if f is not None:
            out[str(k)] = f
    return out
