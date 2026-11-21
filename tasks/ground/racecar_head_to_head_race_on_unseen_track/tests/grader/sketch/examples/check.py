"""Reference checker of the sketch contract `sketch-contract/1.2` (as amended).

What it checks:
  world   a world file against `sketch/world.schema.json` and the semantic rules of the contract
  wire    one request / reply, or a whole JSON-lines transcript, against `wire.schema.json`
  record  a grader configuration, a venue result, a venue reference or a score against `records.schema.json`
  model   a compiled MuJoCo model against the naming, geom group and contact bit rules

It also holds the reference implementation of the track geometry every package must agree with (`Track`: station `s`,
lateral offset, inside test, boundary points) and of the floor lookup (`FloorMap.grip_at`, `FloorMap.zone_at`).

Use as a library (`from sketch.examples import check`) or from the folder that holds `sketch/`:
    python -m sketch.examples.check world sketch/examples/min_world.json
    python -m sketch.examples.check wire  sketch/examples/wire_transcript.jsonl
    python -m sketch.examples.check record config sketch/examples/grader_config.json
Every check returns a list of error strings; an empty list means the object conforms.
"""

import hashlib
import json
import math
import os
import re
import sys

import numpy as np

from . import constants as K

HERE = os.path.dirname(os.path.abspath(__file__))
SKETCH = os.path.dirname(HERE)
WORLD_SCHEMA_PATH = os.path.join(SKETCH, "world.schema.json")
WIRE_SCHEMA_PATH = os.path.join(HERE, "wire.schema.json")
RECORDS_SCHEMA_PATH = os.path.join(HERE, "records.schema.json")

# tolerances of the semantic rules (metres unless noted)
TOL_JOIN = 1e-3            # consecutive pieces of a duct share their end points; junction ends sit on duct ends
TOL_WALL_FACE = 0.10       # a duct axis sits at half width + duct radius from the centreline
TOL_LINE = 0.05            # timing line / checkpoint end points sit on the corridor boundary
TOL_TAPE = 0.01            # tape end points against the rule of the contract
TOL_STATION = 0.10         # a stored station `s_m` against the projection of its point
MAX_SAMPLE_SPACING = 0.25  # centreline samples are at most this far apart
CORRIDOR_OWN_STATION_M = 3.0   # a corridor-boundary point projects back to within this of its own station
BEHIND_OWN_STATION_M = 1.0     # behind a tape, a corridor point within this of the opening's station is its own
BEHIND_SAMPLES_ACROSS = 7      # points across the opening's width (the venue generator's walls._track_behind)
BEHIND_STEP_M = 0.1            # step along `outward` from OPEN_WALL_BEHIND_NEAR_M to OPEN_WALL_BEHIND_CLEAR_M
CLASS_LIMITS = {           # the design (starting values)
    "classic": {"length_m": (60.0, 80.0), "footprint_m": (20.0, 20.0)},
    "featured": {"length_m": (90.0, 110.0), "footprint_m": (30.0, 20.0)},
}
GRIP_TIERS = {"low": (0.45, 0.55), "mid": (0.60, 0.75), "high": (0.80, 0.95)}
TIER_FLOOR_KINDS = {"low": ("painted_concrete", "sealed_concrete"), "mid": ("polished_concrete",),
                    "high": ("exhibition_carpet",)}


# ---------------------------------------------------------------------------------------------- small helpers
def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _schema(path):
    return load_json(path)


def canonical_json(obj):
    """The canonical text of a JSON value: sorted keys, no spaces, ASCII."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def world_digest(world):
    """`layout_digest` of a world: sha256 over the canonical text of {schema, venue_sheet, truth}."""
    body = {"schema": world["schema"], "venue_sheet": world["venue_sheet"], "truth": world["truth"]}
    return hashlib.sha256(canonical_json(body).encode("ascii")).hexdigest()


def public_view(world):
    """What `describe()` may show of a world file: the venue sheet, nothing else."""
    return json.loads(json.dumps(world["venue_sheet"]))


def round_floats(obj, decimals=K.WORLD_FLOAT_DECIMALS):
    """A copy of a JSON value with every float rounded (the writing rule of world files); -0.0 becomes 0.0."""
    if isinstance(obj, float):
        r = round(obj, decimals)
        return 0.0 if r == 0 else r
    if isinstance(obj, dict):
        return {k: round_floats(v, decimals) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [round_floats(v, decimals) for v in obj]
    return obj


def wrap_angle(a):
    """An angle in (-pi, pi]."""
    a = (a + math.pi) % (2.0 * math.pi) - math.pi
    return math.pi if a == -math.pi else a


def piece_ends(piece, length):
    """(tail, head) end points of a duct piece [x, y, yaw] of the given length."""
    x, y, yaw = piece
    dx, dy = 0.5 * length * math.cos(yaw), 0.5 * length * math.sin(yaw)
    return (x - dx, y - dy), (x + dx, y + dy)


def duct_ends(duct):
    """(tail, head) end points of a whole duct: the tail of its first piece and the head of its last piece."""
    c = duct["piece_length_m"]
    return piece_ends(duct["pieces"][0], c)[0], piece_ends(duct["pieces"][-1], c)[1]


def point_in_polygon(x, y, poly):
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1) + x1:
            inside = not inside
    return inside


def polygon_area(poly):
    a = 0.0
    for i in range(len(poly)):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % len(poly)]
        a += x1 * y2 - x2 * y1
    return 0.5 * a


def box_corners(pose, lx, ly):
    x, y, yaw = pose
    c, s = math.cos(yaw), math.sin(yaw)
    return [(x + c * dx - s * dy, y + s * dx + c * dy)
            for dx, dy in ((0.5 * lx, 0.5 * ly), (-0.5 * lx, 0.5 * ly), (-0.5 * lx, -0.5 * ly), (0.5 * lx, -0.5 * ly))]


def _seg_point_dist(p, a, b):
    """Distances from point p to segments a[i] -> b[i] (arrays n x 2)."""
    ab = b - a
    t = np.clip(np.einsum("ij,ij->i", p - a, ab) / np.maximum(np.einsum("ij,ij->i", ab, ab), 1e-18), 0.0, 1.0)
    return np.linalg.norm(a + ab * t[:, None] - p, axis=1)


# ---------------------------------------------------------------------------------------------- track geometry
class Track(object):
    """The centreline of a world and the corridor it defines: the reference semantics of the station
    `s`, of the lateral offset `d` (left of the driving direction is positive) and of "inside the track"."""

    def __init__(self, world):
        cl = world["truth"]["centreline"]
        arr = np.asarray(cl["samples"], dtype=float)
        self.p = arr[:, :2]
        self.left = arr[:, 2]
        self.right = arr[:, 3]
        self.n = len(arr)
        self.seg = np.roll(self.p, -1, axis=0) - self.p
        self.seg_len = np.linalg.norm(self.seg, axis=1)
        self.s = np.concatenate([[0.0], np.cumsum(self.seg_len)[:-1]])
        self.length = float(self.seg_len.sum())
        t = np.roll(self.p, -1, axis=0) - np.roll(self.p, 1, axis=0)
        self.tan = t / np.linalg.norm(t, axis=1, keepdims=True)
        self.nl = np.stack([-self.tan[:, 1], self.tan[:, 0]], axis=1)      # unit normals, left of the direction

    def signed_area(self):
        x, y = self.p[:, 0], self.p[:, 1]
        return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))

    def _locate(self, s):
        s = s % self.length
        i = int(np.searchsorted(self.s, s, side="right") - 1)
        u = (s - self.s[i]) / self.seg_len[i]
        return i, (i + 1) % self.n, u

    def at(self, s):
        """(x, y, heading, half width left, half width right) at station s."""
        i, j, u = self._locate(s)
        p = self.p[i] + u * self.seg[i]
        t = (1.0 - u) * self.tan[i] + u * self.tan[j]
        return (float(p[0]), float(p[1]), math.atan2(t[1], t[0]),
                float((1.0 - u) * self.left[i] + u * self.left[j]), float((1.0 - u) * self.right[i] + u * self.right[j]))

    def project(self, x, y):
        """(s, d): the station of the nearest centreline point and the signed lateral offset of (x, y) from it."""
        q = np.array([x, y], dtype=float)
        u = np.clip(np.einsum("ij,ij->i", q - self.p, self.seg) / (self.seg_len ** 2), 0.0, 1.0)
        foot = self.p + self.seg * u[:, None]
        dist = np.linalg.norm(foot - q, axis=1)
        i = int(np.argmin(dist))
        cross = self.seg[i, 0] * (q[1] - foot[i, 1]) - self.seg[i, 1] * (q[0] - foot[i, 0])
        d = float(dist[i]) if cross >= 0.0 else -float(dist[i])
        return float((self.s[i] + u[i] * self.seg_len[i]) % self.length), d

    def boundary(self, s, side):
        """The corridor boundary point at station s: side 'left' or 'right'."""
        i, j, u = self._locate(s)
        p = self.p[i] + u * self.seg[i]
        n = (1.0 - u) * self.nl[i] + u * self.nl[j]
        n = n / np.linalg.norm(n)
        if side == "left":
            w = (1.0 - u) * self.left[i] + u * self.left[j]
            return float(p[0] + w * n[0]), float(p[1] + w * n[1])
        w = (1.0 - u) * self.right[i] + u * self.right[j]
        return float(p[0] - w * n[0]), float(p[1] - w * n[1])

    def inside(self, x, y, margin=0.0):
        """True when (x, y) is in the corridor: -right(s) + margin <= d <= left(s) - margin at its own station."""
        s, d = self.project(x, y)
        _, _, _, wl, wr = self.at(s)
        return -wr + margin <= d <= wl - margin

    def min_radius(self):
        """Smallest circumradius of three consecutive samples (a discrete estimate of the minimum bend radius)."""
        a, b, c = np.roll(self.p, 1, axis=0), self.p, np.roll(self.p, -1, axis=0)
        la, lb, lc = np.linalg.norm(b - c, axis=1), np.linalg.norm(a - c, axis=1), np.linalg.norm(a - b, axis=1)
        area2 = np.abs((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0]))
        with np.errstate(divide="ignore"):
            r = np.where(area2 > 1e-12, la * lb * lc / (2.0 * np.maximum(area2, 1e-300)), np.inf)
        return float(r.min())

    def self_intersects(self):
        """True when two non-adjacent centreline segments cross."""
        p, q = self.p, np.roll(self.p, -1, axis=0)
        n = self.n
        for i in range(n):
            a, b = p[i], q[i]
            idx = np.arange(i + 2, n if i > 0 else n - 1)
            if not len(idx):
                continue
            c, d = p[idx], q[idx]
            d1 = (b[0] - a[0]) * (c[:, 1] - a[1]) - (b[1] - a[1]) * (c[:, 0] - a[0])
            d2 = (b[0] - a[0]) * (d[:, 1] - a[1]) - (b[1] - a[1]) * (d[:, 0] - a[0])
            d3 = (d[:, 0] - c[:, 0]) * (a[1] - c[:, 1]) - (d[:, 1] - c[:, 1]) * (a[0] - c[:, 0])
            d4 = (d[:, 0] - c[:, 0]) * (b[1] - c[:, 1]) - (d[:, 1] - c[:, 1]) * (b[0] - c[:, 0])
            if np.any((d1 * d2 < 0) & (d3 * d4 < 0)):
                return True
        return False


class FloorMap(object):
    """The floor of a world as data: the grip and the zone under a point."""

    def __init__(self, world):
        floor = world["truth"]["floor"]
        self.base_grip = float(floor["base_grip"])
        self.patches = floor["patches"]

    def zone_at(self, x, y):
        """Id of the patch under (x, y): the last listed patch that contains the point; None on the base floor."""
        for patch in reversed(self.patches):
            if point_in_polygon(x, y, patch["polygon"]):
                return patch["id"]
        return None

    def grip_at(self, x, y):
        """Friction coefficient of this car's tyres on the floor under (x, y)."""
        zid = self.zone_at(x, y)
        if zid is None:
            return self.base_grip
        return float(next(p["grip"] for p in self.patches if p["id"] == zid))


def expected_features(world):
    """The feature kinds the venue sheet must list for this truth."""
    t = world["truth"]
    cl = np.asarray(t["centreline"]["samples"], dtype=float)
    out = set()
    if any(g["width_m"] > 1e-6 for g in t["gaps"]):
        out.add("duct_gaps")
    if t["open_walls"]:
        out.add("open_wall")
    for p in t["floor"]["patches"]:
        out.add("floor_patch_" + p["kind"])
    if any(d["wall_class"] == "black_plain" for d in t["ducts"]):
        out.add("black_ducts")
    if float((cl[:, 2] + cl[:, 3]).min()) < K.NARROW_SECTION_M:
        out.add("narrow_section")
    if any(w["side"] == "shared" for w in t["walls"]):
        out.add("shared_wall")
    if t["clutter"]:
        out.add("outside_clutter")
    return out


# ---------------------------------------------------------------------------------------------- world checks
def schema_errors(obj, schema, limit=20):
    import jsonschema
    v = jsonschema.Draft202012Validator(schema)
    out = []
    for e in sorted(v.iter_errors(obj), key=lambda e: [str(x) for x in e.absolute_path]):
        out.append("schema: /%s: %s" % ("/".join(str(x) for x in e.absolute_path), e.message[:200]))
        if len(out) >= limit:
            out.append("schema: more errors not shown")
            break
    return out


def _unique(ids, what, err):
    seen = set()
    for i in ids:
        if i in seen:
            err.append("%s id %r is used twice" % (what, i))
        seen.add(i)


def check_world(world):
    """All errors of a world file (schema first; the semantic rules only run on a schema-valid file)."""
    err = schema_errors(world, _schema(WORLD_SCHEMA_PATH))
    if err:
        return err
    t = world["truth"]
    sheet = world["venue_sheet"]
    cls = t["cell"]["layout_class"]
    graded = cls in CLASS_LIMITS
    R = K.DUCT_RADIUS_M

    if world["layout_digest"] != world_digest(world):
        err.append("layout_digest does not match the content (expected %s)" % world_digest(world))
    if round_floats(world) != world:
        err.append("floats are not rounded to %d decimals" % K.WORLD_FLOAT_DECIMALS)

    # ---- centreline
    tr = Track(world)
    cl = t["centreline"]
    if float(tr.seg_len.max()) > MAX_SAMPLE_SPACING:
        err.append("centreline samples are up to %.3f m apart (max %.2f)" % (tr.seg_len.max(), MAX_SAMPLE_SPACING))
    if float(tr.seg_len.min()) < 1e-3:
        err.append("centreline has coincident samples")
    if abs(cl["length_m"] - tr.length) > 1e-3:
        err.append("centreline.length_m %.4f differs from the polyline length %.4f" % (cl["length_m"], tr.length))
    area = tr.signed_area()
    if (area > 0) != (cl["direction"] == "ccw"):
        err.append("centreline.direction %r contradicts the order of the samples" % cl["direction"])
    if tr.self_intersects():
        err.append("centreline crosses itself")
    width = tr.left + tr.right
    wmin = float(width.min())
    if wmin < K.TRACK_WIDTH_RANGE_M[0] - 1e-6 or float(width.max()) > K.TRACK_WIDTH_RANGE_M[1] + 1e-6:
        err.append("track width %.3f..%.3f m is outside %.1f..%.1f" % ((wmin, float(width.max())) + K.TRACK_WIDTH_RANGE_M))
    rmin = tr.min_radius()
    if rmin < K.MIN_RADIUS_M - 0.02:
        err.append("minimum centreline radius %.3f m is below %.1f" % (rmin, K.MIN_RADIUS_M))

    # ---- stats
    st = t["stats"]
    frac = float(np.sum(tr.seg_len[width >= 1.5 - 1e-9]) / tr.length)
    n_pieces = sum(len(d["pieces"]) for d in t["ducts"])
    for key, val, tol in (("length_m", tr.length, 1e-3), ("min_width_m", wmin, 1e-6), ("width_ge_1p5_frac", frac, 0.01),
                          ("min_radius_m", rmin, 0.05 * rmin), ("n_ducts", len(t["ducts"]), 0), ("n_pieces", n_pieces, 0)):
        if abs(st[key] - val) > tol:
            err.append("stats.%s %r differs from the computed %r" % (key, st[key], round(val, 6)))
    b = t["bounds"]
    fp = (b["xmax"] - b["xmin"], b["ymax"] - b["ymin"])
    if abs(st["footprint_m"][0] - fp[0]) > 1e-6 or abs(st["footprint_m"][1] - fp[1]) > 1e-6:
        err.append("stats.footprint_m differs from the size of bounds")
    if abs(b["xmax"] + b["xmin"]) > 0.02 or abs(b["ymax"] + b["ymin"]) > 0.02:
        err.append("bounds are not centred on the world origin")
    if fp[0] < fp[1] - 1e-9:
        err.append("the longer side of bounds must lie along x")
    if graded:
        lim = CLASS_LIMITS[cls]
        if not lim["length_m"][0] <= tr.length <= lim["length_m"][1]:
            err.append("class %s wants a centreline of %.0f..%.0f m, got %.2f" % ((cls,) + lim["length_m"] + (tr.length,)))
        if fp[0] > lim["footprint_m"][0] + 1e-6 or fp[1] > lim["footprint_m"][1] + 1e-6:
            err.append("class %s wants a footprint within %.0f x %.0f m, got %.2f x %.2f" % ((cls,) + lim["footprint_m"] + fp))
        if frac < 0.70:
            err.append("only %.0f %% of the length is at least 1.5 m wide (70 %% wanted)" % (100 * frac))

    # ---- ducts and walls
    ducts = {d["id"]: d for d in t["ducts"]}
    _unique([d["id"] for d in t["ducts"]], "duct", err)
    _unique([w["id"] for w in t["walls"]], "wall", err)
    walls = {w["id"]: w for w in t["walls"]}
    in_wall = {}
    for w in t["walls"]:
        for did in w["ducts"]:
            if did not in ducts:
                err.append("wall %s lists the unknown duct %s" % (w["id"], did))
            elif did in in_wall:
                err.append("duct %s is in two walls" % did)
            else:
                in_wall[did] = w["id"]
    seg_a, seg_b = [], []
    for d in t["ducts"]:
        did, c, n = d["id"], d["piece_length_m"], len(d["pieces"])
        if in_wall.get(did) != d["wall"]:
            err.append("duct %s names wall %s but is listed in %s" % (did, d["wall"], in_wall.get(did)))
        if abs(d["length_m"] - c * n) > 1e-4:
            err.append("duct %s: length_m %.4f is not pieces x piece_length_m %.4f" % (did, d["length_m"], c * n))
        if d["length_m"] > K.DUCT_LENGTH_M + 1e-6:
            err.append("duct %s is longer than one duct (%.2f m)" % (did, K.DUCT_LENGTH_M))
        if not K.PIECE_LENGTH_RANGE_M[0] <= c <= K.PIECE_LENGTH_RANGE_M[1]:
            err.append("duct %s: piece_length_m %.3f is outside %.2f..%.2f" % ((did, c) + K.PIECE_LENGTH_RANGE_M))
        side = walls[d["wall"]]["side"] if d["wall"] in walls else None
        for k, piece in enumerate(d["pieces"]):
            tail, head = piece_ends(piece, c)
            seg_a.append(tail)
            seg_b.append(head)
            if k:
                prev_head = piece_ends(d["pieces"][k - 1], c)[1]
                if math.dist(prev_head, tail) > TOL_JOIN:
                    err.append("duct %s: pieces %d and %d do not share an end point (%.4f m apart)"
                               % (did, k - 1, k, math.dist(prev_head, tail)))
            for x, y in (tail, head):
                if not (b["xmin"] - 1e-6 <= x - R and x + R <= b["xmax"] + 1e-6
                        and b["ymin"] - 1e-6 <= y - R and y + R <= b["ymax"] + 1e-6):
                    err.append("duct %s piece %d lies outside bounds" % (did, k))
                    break
            s, off = tr.project(piece[0], piece[1])
            _, _, _, wl, wr = tr.at(s)
            want = (wl if off > 0 else wr) + R
            if abs(abs(off) - want) > TOL_WALL_FACE:
                err.append("duct %s piece %d: axis is %.3f m from the centreline, the widths say %.3f"
                           % (did, k, abs(off), want))
            if side in ("left", "right") and (off > 0) != (side == "left"):
                err.append("duct %s piece %d is on the %s of the centreline but its wall is the %s wall"
                           % (did, k, "left" if off > 0 else "right", side))
    seg_a, seg_b = np.asarray(seg_a, dtype=float), np.asarray(seg_b, dtype=float)

    # ---- junctions: gaps and open walls
    _unique([g["id"] for g in t["gaps"]] + [o["id"] for o in t["open_walls"]], "junction", err)
    junctions = {}
    for kind, items in (("gap", t["gaps"]), ("open_wall", t["open_walls"])):
        for j in items:
            key = tuple(j["ducts"])
            if key in junctions:
                err.append("the junction %s -> %s is listed twice" % key)
            junctions[key] = (kind, j)
            if j["ducts"][0] not in ducts or j["ducts"][1] not in ducts:
                err.append("%s %s names an unknown duct" % (kind, j["id"]))
                continue
            head = duct_ends(ducts[j["ducts"][0]])[1]
            tail = duct_ends(ducts[j["ducts"][1]])[0]
            if math.dist(head, j["ends"][0]) > TOL_JOIN or math.dist(tail, j["ends"][1]) > TOL_JOIN:
                err.append("%s %s: ends are not the head of %s and the tail of %s" % ((kind, j["id"]) + key))
            if abs(j["width_m"] - math.dist(j["ends"][0], j["ends"][1])) > TOL_JOIN:
                err.append("%s %s: width_m %.4f differs from the distance of its ends %.4f"
                           % (kind, j["id"], j["width_m"], math.dist(j["ends"][0], j["ends"][1])))
            if j["wall"] != in_wall.get(j["ducts"][0]) or j["wall"] != in_wall.get(j["ducts"][1]):
                err.append("%s %s: its ducts are not both in wall %s" % (kind, j["id"], j["wall"]))
    wanted = set()
    for w in t["walls"]:
        seq = [d for d in w["ducts"] if d in ducts]
        pairs = list(zip(seq, seq[1:])) + ([(seq[-1], seq[0])] if w["closed"] and seq else [])
        for pair in pairs:
            wanted.add(pair)
            if pair not in junctions:
                err.append("wall %s: no gap or open wall is listed between %s and %s" % ((w["id"],) + pair))
    for key in junctions:
        if key not in wanted:
            err.append("junction %s -> %s does not join consecutive ducts of a wall" % key)

    openings = [(0.5 * (j["ends"][0][0] + j["ends"][1][0]), 0.5 * (j["ends"][0][1] + j["ends"][1][1]), j["width_m"])
                for j in list(t["gaps"]) + list(t["open_walls"])]
    for o in t["open_walls"]:
        if o["wall"] in walls and walls[o["wall"]]["side"] == "shared":
            err.append("open wall %s is in a shared wall" % o["id"])
        mid = (0.5 * (o["tape"][0][0] + o["tape"][1][0]), 0.5 * (o["tape"][0][1] + o["tape"][1][1]))
        s, off = tr.project(*mid)
        if min(abs(s - o["s_m"]), tr.length - abs(s - o["s_m"])) > TOL_STATION:
            err.append("open wall %s: s_m %.3f differs from the station of its tape %.3f" % (o["id"], o["s_m"], s))
        _, _, _, wl, wr = tr.at(s)
        if abs(o["track_width_m"] - (wl + wr)) > 0.05:
            err.append("open wall %s: track_width_m %.3f differs from the widths at its station %.3f"
                       % (o["id"], o["track_width_m"], wl + wr))
        if not o["width_m"] < o["track_width_m"]:
            err.append("open wall %s is not narrower than the track at its station" % o["id"])
        out = np.asarray(o["outward"], dtype=float)
        if abs(np.linalg.norm(out) - 1.0) > 1e-3:
            err.append("open wall %s: outward is not a unit vector" % o["id"])
        to_centre = np.array([tr.at(s)[0] - mid[0], tr.at(s)[1] - mid[1]])
        if float(out @ to_centre) >= 0.0:
            err.append("open wall %s: outward points into the track" % o["id"])
        if o["ducts"][0] in ducts and o["ducts"][1] in ducts:
            for end, tape, did, last in ((o["ends"][0], o["tape"][0], o["ducts"][0], True),
                                         (o["ends"][1], o["tape"][1], o["ducts"][1], False)):
                yaw = ducts[did]["pieces"][-1 if last else 0][2]
                nrm = np.array([-math.sin(yaw), math.cos(yaw)])
                if float(nrm @ out) > 0.0:
                    nrm = -nrm                                   # the normal of the end piece that points to the track
                want = np.asarray(end, dtype=float) + R * nrm
                if float(np.linalg.norm(want - np.asarray(tape, dtype=float))) > TOL_TAPE:
                    err.append("open wall %s: tape end (%.3f, %.3f) is not the track-side edge of duct %s (%.3f, %.3f)"
                               % (o["id"], tape[0], tape[1], did, want[0], want[1]))

    # ---- every stretch of the corridor boundary has a wall, a gap or an open wall
    if len(seg_a):
        for i in range(tr.n):
            for side in ("left", "right"):
                bp = np.array(tr.boundary(tr.s[i], side))
                if float(_seg_point_dist(bp, seg_a, seg_b).min()) - R <= TOL_WALL_FACE:
                    continue
                if any(math.dist(bp, (ox, oy)) <= 0.5 * ow + 0.3 for ox, oy, ow in openings):
                    continue
                err.append("no wall on the %s at station %.2f m" % (side, tr.s[i]))
                break
            else:
                continue
            break

    # ---- every stretch's corridor in its own nearest-centreline region: a boundary point
    # projects back to (nearly) its own station, so a shared wall or a near pass never mixes two stretches
    worst = None
    for i in range(tr.n):
        for side in ("left", "right"):
            bx_, by_ = tr.boundary(tr.s[i], side)
            s2, _ = tr.project(bx_, by_)
            gap = abs(s2 - tr.s[i]) % tr.length
            gap = min(gap, tr.length - gap)
            if gap > CORRIDOR_OWN_STATION_M and (worst is None or gap > worst[0]):
                worst = (gap, side, float(tr.s[i]), s2)
    if worst is not None:
        err.append("corridor: the %s boundary point at station %.2f m projects to station %.2f m (%.2f m away along the "
                   "track; at most %.1f): two stretches of the corridor overlap" % (worst[1], worst[2], worst[3], worst[0],
                                                                                   CORRIDOR_OWN_STATION_M))

    # ---- behind every open wall lies open floor, never another stretch of the track
    for o in t["open_walls"]:
        (ax_, ay_), (bx_, by_) = o["tape"]
        ox_, oy_ = float(o["outward"][0]), float(o["outward"][1])
        hit = None
        for f in np.linspace(0.0, 1.0, BEHIND_SAMPLES_ACROSS):
            px0, py0 = ax_ + f * (bx_ - ax_), ay_ + f * (by_ - ay_)
            for dd in np.arange(K.OPEN_WALL_BEHIND_NEAR_M, K.OPEN_WALL_BEHIND_CLEAR_M + 1e-9, BEHIND_STEP_M):
                px, py = px0 + dd * ox_, py0 + dd * oy_
                if tr.inside(px, py):
                    s2, _ = tr.project(px, py)
                    gap = abs(s2 - o["s_m"]) % tr.length
                    gap = min(gap, tr.length - gap)
                    if gap > BEHIND_OWN_STATION_M and (hit is None or dd < hit[0]):
                        hit = (float(dd), s2)
        if hit is not None:
            err.append("open wall %s: another stretch of the track (station %.2f m) lies %.2f m behind its tape "
                       "(open floor wanted up to %.1f m)" % (o["id"], hit[1], hit[0], K.OPEN_WALL_BEHIND_CLEAR_M))

    # ---- floor
    floor = t["floor"]
    if floor["kind"] != sheet["floor"]["kind"]:
        err.append("truth.floor.kind and venue_sheet.floor.kind differ")
    tier = t["cell"]["grip_tier"]
    if tier in GRIP_TIERS:
        lo, hi = GRIP_TIERS[tier]
        if not lo - 1e-9 <= floor["base_grip"] <= hi + 1e-9:
            err.append("base_grip %.3f is outside the %s tier %.2f..%.2f" % (floor["base_grip"], tier, lo, hi))
        if floor["kind"] not in TIER_FLOOR_KINDS[tier]:
            err.append("floor kind %s does not belong to the %s tier" % (floor["kind"], tier))
    _unique([p["id"] for p in floor["patches"]], "patch", err)
    if graded and not 1 <= len(floor["patches"]) <= 3:
        err.append("a graded venue has 1 to 3 floor patches, got %d" % len(floor["patches"]))
    if graded and not any(p["on_racing_line"] for p in floor["patches"]):
        err.append("no floor patch is on the racing line")
    for p in floor["patches"]:
        poly = p["polygon"]
        if polygon_area(poly) <= 1e-4:
            err.append("patch %s: polygon is not counter-clockwise with a positive area" % p["id"])
        if p["edge_height_m"] > 0.0:
            if p["kind"] != "carpet_tile":
                err.append("patch %s: only a carpet tile has an edge height" % p["id"])
            v = np.asarray(poly, dtype=float)
            e = np.roll(v, -1, axis=0) - v
            if len(poly) != 4 or any(abs(float(e[i] @ e[(i + 1) % 4])) > 1e-3 for i in range(4)):
                err.append("patch %s: a patch with an edge height must be a rectangle" % p["id"])
        cx, cy = np.mean(np.asarray(poly, dtype=float), axis=0)
        s, _ = tr.project(cx, cy)
        if min(abs(s - p["s_m"]), tr.length - abs(s - p["s_m"])) > 0.5:
            err.append("patch %s: s_m %.2f differs from the station of its centre %.2f" % (p["id"], p["s_m"], s))
        if p["on_racing_line"] and not any(tr.inside(x, y) for x, y in poly + [[cx, cy]]):
            err.append("patch %s is marked on the racing line but lies outside the track" % p["id"])

    # ---- clutter
    _unique([c["id"] for c in t["clutter"]], "clutter", err)
    for c in t["clutter"]:
        lx, ly, _ = c["size_m"]
        if c["shape"] == "cylinder" and abs(lx - ly) > 1e-9:
            err.append("clutter %s: a cylinder has size_m[0] == size_m[1] (its diameter)" % c["id"])
        pts = box_corners(c["pose"], lx, ly) + [(c["pose"][0], c["pose"][1])]
        if any(tr.inside(x, y) for x, y in pts):
            err.append("clutter %s is inside the track" % c["id"])
        if len(seg_a):
            clear = min(float(_seg_point_dist(np.array(q), seg_a, seg_b).min()) for q in pts) - R
            if clear < K.CLUTTER_MIN_CLEARANCE_M - 1e-6:
                err.append("clutter %s is %.3f m from a wall (%.1f m wanted)" % (c["id"], clear, K.CLUTTER_MIN_CLEARANCE_M))

    # ---- timing line, checkpoints, start pose
    def line_ok(name, ln):
        la, lb = tr.boundary(ln["s_m"], "left"), tr.boundary(ln["s_m"], "right")
        if math.dist(la, ln["a"]) > TOL_LINE or math.dist(lb, ln["b"]) > TOL_LINE:
            err.append("%s: a / b are not the left / right boundary points at s_m %.3f" % (name, ln["s_m"]))
        h = tr.at(ln["s_m"])[2]
        if ln["forward"][0] * math.cos(h) + ln["forward"][1] * math.sin(h) < 0.999:
            err.append("%s: forward is not the driving direction at its station" % name)
        if not 0.0 <= ln["s_m"] < tr.length:
            err.append("%s: s_m is outside [0, length)" % name)

    line_ok("timing_line", t["timing_line"])
    _unique([c["id"] for c in t["checkpoints"]], "checkpoint", err)
    s0 = t["timing_line"]["s_m"]
    last = 0.0
    for c in t["checkpoints"]:
        line_ok("checkpoint %s" % c["id"], c)
        rel = (c["s_m"] - s0) % tr.length
        if rel <= last + 0.5:
            err.append("checkpoint %s is not at least 0.5 m after the previous line in driving order" % c["id"])
        last = rel
    if tr.length - last < 0.5:
        err.append("the last checkpoint is less than 0.5 m before the timing line")
    sp = t["start_pose"]
    s, off = tr.project(sp["pose"][0], sp["pose"][1])
    if min(abs(s - sp["s_m"]), tr.length - abs(s - sp["s_m"])) > TOL_STATION:
        err.append("start_pose.s_m %.3f differs from the station of the pose %.3f" % (sp["s_m"], s))
    if abs(wrap_angle(sp["pose"][2] - tr.at(s)[2])) > 0.2:
        err.append("start pose does not face the driving direction")
    cs, sn = math.cos(sp["pose"][2]), math.sin(sp["pose"][2])
    for wx, wy, _ in K.WHEEL_GROUND_C.values():
        if not tr.inside(sp["pose"][0] + cs * wx - sn * wy, sp["pose"][1] + sn * wx + cs * wy, margin=0.05):
            err.append("a wheel of the car at the start pose is outside the track")
            break
    ahead = (s0 - (s + K.X_FRONT_M)) % tr.length
    if not 0.2 <= ahead <= 3.0:
        err.append("the timing line is %.2f m ahead of the car's front at the start pose (0.2..3.0 wanted)" % ahead)

    # ---- avoidance set
    av = t["avoidance"]
    _unique([x["id"] for x in av["boxes"]], "box", err)
    _unique([x["id"] for x in av["slow_cars"]], "slow car", err)
    if graded and len(av["boxes"]) != 3:
        err.append("a graded venue has 3 boxes, got %d" % len(av["boxes"]))
    if graded and not 1 <= len(av["slow_cars"]) <= 2:
        err.append("a graded venue has 1 or 2 slow cars, got %d" % len(av["slow_cars"]))
    for bx in av["boxes"]:
        corners = box_corners(bx["pose"], bx["size_m"][0], bx["size_m"][1])
        if not all(tr.inside(x, y) for x, y in corners):
            err.append("box %s is not wholly inside the track" % bx["id"])
            continue
        s, _ = tr.project(bx["pose"][0], bx["pose"][1])
        offs = [tr.project(x, y)[1] for x, y in corners]
        _, _, _, wl, wr = tr.at(s)
        passage = max(wl - max(offs), min(offs) + wr)
        if abs(passage - bx["free_passage_m"]) > 0.05:
            err.append("box %s: free_passage_m %.3f differs from the computed %.3f" % (bx["id"], bx["free_passage_m"], passage))
        if passage < K.BOX_FREE_PASSAGE_M - 1e-6:
            err.append("box %s leaves a passage of %.3f m (%.1f wanted)" % (bx["id"], passage, K.BOX_FREE_PASSAGE_M))
        if min(abs(s - bx["s_m"]), tr.length - abs(s - bx["s_m"])) > TOL_STATION:
            err.append("box %s: s_m differs from the station of its pose" % bx["id"])
    for sc in av["slow_cars"]:
        path = np.asarray(sc["path"]["samples"], dtype=float)
        if not all(tr.inside(x, y, margin=0.5 * K.ENVELOPE_M[1]) for x, y in path[:, :2]):
            err.append("slow car %s: its path leaves the track (half a car width from the walls wanted)" % sc["id"])
        step = np.linalg.norm(np.roll(path[:, :2], -1, axis=0) - path[:, :2], axis=1)
        if float(step.max()) > MAX_SAMPLE_SPACING:
            err.append("slow car %s: path samples are up to %.3f m apart" % (sc["id"], step.max()))
        if not 0.0 <= sc["entry_s_m"] < tr.length:
            err.append("slow car %s: entry_s_m is outside [0, length)" % sc["id"])
        x, y = sc["staging_pose"][0], sc["staging_pose"][1]
        if b["xmin"] - 1.0 < x < b["xmax"] + 1.0 and b["ymin"] - 1.0 < y < b["ymax"] + 1.0:
            err.append("slow car %s: staging pose is less than 1 m outside bounds" % sc["id"])

    # ---- venue sheet against the truth
    want = expected_features(world)
    have = set(sheet["features"])
    if want != have:
        err.append("venue_sheet.features: missing %s, not in the truth %s" % (sorted(want - have), sorted(have - want)))
    used = {d["wall_class"] for d in t["ducts"]}
    if not used <= set(sheet["walls"]["kinds"]):
        err.append("venue_sheet.walls.kinds does not list %s" % sorted(used - set(sheet["walls"]["kinds"])))
    if sheet["track"]["min_width_m"] > wmin + 1e-6:
        err.append("venue_sheet.track.min_width_m %.2f is above the real minimum %.3f" % (sheet["track"]["min_width_m"], wmin))
    if sheet["track"]["max_footprint_m"][0] < fp[0] - 1e-6 or sheet["track"]["max_footprint_m"][1] < fp[1] - 1e-6:
        err.append("venue_sheet.track.max_footprint_m is smaller than the real footprint")
    return err


# ---------------------------------------------------------------------------------------------- wire checks
def _wire_defs():
    return _schema(WIRE_SCHEMA_PATH)


def _sub(schema, name):
    return {"$schema": schema["$schema"], "$ref": "#/$defs/" + name, "$defs": schema["$defs"]}


PUBLIC_METHODS = ("describe", "open_session", "step", "close_session", "request_pickup")
PRIVILEGED_METHODS = ("priv_unlock", "priv_world", "priv_state")
METHODS = PUBLIC_METHODS + PRIVILEGED_METHODS
ERROR_CODES = ("forbidden", "bad_params", "no_session", "bad_order", "venue_over")
EVENT_TYPES = ("lap", "stopped", "released", "warning", "phase", "session_end")


def check_request(obj):
    """Errors of a request: the envelope and, for a method of the table, its parameters."""
    return schema_errors(obj, _sub(_wire_defs(), "request"))


def check_reply(method, obj):
    """Errors of the reply to a request of the given method (an error reply is valid for every method)."""
    s = _wire_defs()
    if isinstance(obj, dict) and "error" in obj:
        return schema_errors(obj, _sub(s, "error_reply"))
    name = "reply_" + method
    if name not in s["$defs"]:
        return ["method %r is not in the table: the only valid reply is the error `forbidden`" % method]
    return schema_errors(obj, _sub(s, name))


def check_transcript(lines):
    """Errors of a transcript: an alternation of request and reply objects, ids counting up from 1."""
    err, pending, last_id = [], None, 0
    for n, line in enumerate(lines, 1):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except ValueError as e:
            err.append("line %d: not JSON (%s)" % (n, e))
            continue
        if pending is None:
            e = schema_errors(obj, _sub(_wire_defs(), "envelope"))
            err += ["line %d: %s" % (n, x) for x in e]
            if not e:
                if obj["id"] != last_id + 1:
                    err.append("line %d: request id %r, expected %d" % (n, obj["id"], last_id + 1))
                last_id = obj["id"]
                pending = obj
        else:
            if check_request(pending):
                # a request that is not well formed for its method is answered with `bad_params`, nothing else
                if not (isinstance(obj, dict) and obj.get("error") == "bad_params"):
                    err.append("line %d: the request before is ill-formed, the reply must be the error `bad_params`" % n)
                err += ["line %d: %s" % (n, x) for x in schema_errors(obj, _sub(_wire_defs(), "error_reply"))]
            elif pending["method"] not in METHODS:
                if not (isinstance(obj, dict) and obj.get("error") == "forbidden"):
                    err.append("line %d: method %r is not in the table, the reply must be the error `forbidden`" % (n, pending["method"]))
            else:
                err += ["line %d: %s" % (n, x) for x in check_reply(pending["method"], obj)]
            if isinstance(obj, dict) and obj.get("id") != pending["id"]:
                err.append("line %d: reply id %r does not answer request %r" % (n, obj.get("id"), pending["id"]))
            pending = None
    if pending is not None:
        err.append("the last request has no reply")
    return err


def check_record(kind, obj):
    """kind: 'config', 'result', 'reference', 'score' or 'lab_log_header' / 'lab_log_row'."""
    s = _schema(RECORDS_SCHEMA_PATH)
    if kind not in s["$defs"]:
        return ["unknown record kind %r" % kind]
    err = schema_errors(obj, _sub(s, kind))
    if kind == "config" and not err:
        sim = obj["sim"]
        if abs(sim["physics_dt_s"] * sim["physics_steps_per_tick"] - sim["tick_s"]) > 1e-12:
            err.append("sim: physics_dt_s x physics_steps_per_tick is not tick_s")
        if [x["name"] for x in obj["sessions"]] != list(K.SESSIONS):
            err.append("sessions are not %s in this order" % (K.SESSIONS,))
        for ses in obj["sessions"]:
            t0 = [p["t0_s"] for p in ses["phases"]]
            if t0[0] != 0.0 or t0 != sorted(t0) or t0[-1] >= ses["duration_s"]:
                err.append("session %s: phases must start at 0, ascend and begin before the session ends" % ses["name"])
            if abs(round(ses["duration_s"] / sim["tick_s"]) * sim["tick_s"] - ses["duration_s"]) > 1e-9:
                err.append("session %s: duration is not a whole number of ticks" % ses["name"])
        w = obj["scoring"]["weights"]
        if abs(w["fast"] + w["cons"] + w["avoid"] - 1.0) > 1e-9:
            err.append("scoring.weights do not sum to 1")
    return err


JOIN_EPS_S = 1e-5          # a lap joins the one before when its start (t_end_s - lap_s) is within this of that end


def laps_joined(prev_end_s, lap):
    """True when `lap` continues a run that ended at `prev_end_s` (the t_end_s of the lap before it in the list):
    no `chain_break` flag and its start within JOIN_EPS_S of that end (both times carry 6 decimals). Records written
    before contract 1.2 have no flag; their times decide alone."""
    if lap.get("chain_break"):
        return False
    if prev_end_s is None:
        return False
    return abs((lap["t_end_s"] - lap["lap_s"]) - prev_end_s) <= JOIN_EPS_S


def summarise_laps(result, config):
    """The scored quantities of a venue result, from its lap lists: {"time_trial": ..., "avoid": ...}.

    T_best_s  fastest lap of tt1 and tt2 that was not interrupted (touches do not matter)
    N_best    longest run of consecutive laps without an interrupted one, inside one heat; consecutive = joined end
              to start: a lap with `chain_break` (the lap clock threw a lap away
              before it), or one that does not start where the lap before it ended, begins a new run
    T_clean_s per phase of the avoidance test: fastest lap that lies wholly inside the phase, is clean (not
              interrupted, no touch), has no standstill and, in the dynamic phase, passed at least
              scoring.avoid.min_overtakes_dynamic slow cars"""
    by_name = {s["name"]: s for s in result["sessions"]}
    t_best, n_best = None, 0
    for name in ("tt1", "tt2"):
        run = 0
        prev_end = None
        for lap in by_name.get(name, {"laps": []})["laps"]:
            if lap["interrupted"]:
                run = 0
                prev_end = None
                continue
            if run > 0 and not laps_joined(prev_end, lap):
                run = 0
            run += 1
            prev_end = lap["t_end_s"]
            n_best = max(n_best, run)
            t_best = lap["lap_s"] if t_best is None else min(t_best, lap["lap_s"])
    ses = next(x for x in config["sessions"] if x["name"] == "avoid")
    edges = [p["t0_s"] for p in ses["phases"]] + [ses["duration_s"]]
    avoid = {}
    for i, phase in enumerate(ses["phases"]):
        need = config["scoring"]["avoid"]["min_overtakes_dynamic"] if phase["name"] == "dynamic" else 0
        laps = [lap["lap_s"] for lap in by_name.get("avoid", {"laps": []})["laps"]
                if lap["clean"] and not lap["standstill"] and lap["overtakes"] >= need
                and lap["t_end_s"] - lap["lap_s"] >= edges[i] - 1e-9 and lap["t_end_s"] <= edges[i + 1] + 1e-9]
        avoid[phase["name"]] = {"T_clean_s": min(laps) if laps else None, "clean_laps": len(laps)}
    return {"time_trial": {"T_best_s": t_best, "N_best": n_best}, "avoid": avoid}


def check_result(result, config):
    """Errors of a venue result: the schema, then the summary fields against the lap lists."""
    err = check_record("result", result)
    if err:
        return err
    names = [s["name"] for s in result["sessions"]]
    if names != list(K.SESSIONS[:len(names)]):
        err.append("sessions are not a prefix of %s" % (K.SESSIONS,))
    for s in result["sessions"]:
        for i, lap in enumerate(s["laps"]):
            if lap["n"] != i + 1:
                err.append("session %s: lap numbers do not count up from 1" % s["name"])
                break
            if lap["clean"] != ((not lap["interrupted"]) and lap["touches"] == 0):
                err.append("session %s lap %d: clean is not (not interrupted and no touch)" % (s["name"], lap["n"]))
            if lap["t_end_s"] > s["ticks"] * config["sim"]["tick_s"] + 1e-9 or lap["t_end_s"] - lap["lap_s"] < -1e-9:
                err.append("session %s lap %d lies outside the session" % (s["name"], lap["n"]))
            if i == 0 and lap["chain_break"]:
                err.append("session %s lap 1 has chain_break" % s["name"])
            elif i > 0 and not lap["chain_break"] and not laps_joined(s["laps"][i - 1]["t_end_s"], lap):
                err.append("session %s lap %d does not start where lap %d ended but has no chain_break"
                           % (s["name"], lap["n"], lap["n"] - 1))
    want = summarise_laps(result, config)
    for key in ("time_trial", "avoid"):
        if result[key] != want[key]:
            err.append("%s %s differs from the lap lists, which give %s" % (key, json.dumps(result[key]), json.dumps(want[key])))
    if result["stats"]["laps_by_session"] != {s["name"]: len(s["laps"]) for s in result["sessions"]}:
        err.append("stats.laps_by_session differs from the lap lists")
    return err


def reference_score(results, references, config, gates=None, anchor_value=1.0):
    """The arithmetic of the design on the contract's records: a list of venue results and the
    matching venue references -> a score record. The scoring module is the one the task uses; this function is
    the cross-check both must agree with."""
    sc = config["scoring"]

    def clip(v):
        return max(0.0, min(1.0, v))

    venues = []
    for res, ref in zip(results, references):
        if res["layout_digest"] != ref["layout_digest"]:
            raise ValueError("result and reference are of different venues")
        t_ref, n_ref = ref["T_ref_s"], ref["N_ref"]
        t_best, n_best = res["time_trial"]["T_best_s"], res["time_trial"]["N_best"]
        fast = 0.0 if t_best is None else clip((sc["fast"]["zero_ratio"] * t_ref - t_best) / (sc["fast"]["span_ratio"] * t_ref))
        zero = sc["cons"]["zero_frac"] * n_ref
        cons = clip((n_best - zero) / ((n_ref - sc["cons"]["full_margin_laps"]) - zero))
        a = {}
        for phase in ("static", "dynamic"):
            t_clean, t_pref = res["avoid"][phase]["T_clean_s"], ref["T_phase_ref_s"][phase]
            a[phase] = 0.0 if t_clean is None else clip((sc["avoid"]["zero_ratio"] * t_pref - t_clean) / (sc["avoid"]["span_ratio"] * t_pref))
        avoid = sc["avoid"]["phase_weights"]["static"] * a["static"] + sc["avoid"]["phase_weights"]["dynamic"] * a["dynamic"]
        w = sc["weights"]
        rank = None
        if t_best is not None:
            ratio = t_best / t_ref
            rank = 1 + sum(1 for r in [1.0] + list(sc["virtual_rank_ratios"]) if r < ratio)
        venues.append({"world_id": res["world_id"], "fast": round(fast, 6), "cons": round(cons, 6),
                       "avoid_static": round(a["static"], 6), "avoid_dynamic": round(a["dynamic"], 6), "avoid": round(avoid, 6),
                       "venue": round(w["fast"] * fast + w["cons"] * cons + w["avoid"] * avoid, 6), "virtual_rank": rank})
    gates = dict(gates or {"submission_ok": True, "step_called": all(r["step_called"] for r in results), "clean_run": True,
                           "no_network": True, "no_late_write": True})
    gate = 1 if all(gates.values()) else 0
    score = round(gate * sum(v["venue"] for v in venues) / len(venues), 6)
    return {"schema": K.SCORE_SCHEMA, "venues": venues, "gates": gates, "gate": gate, "time_trial_score": score,
            "anchor": {"value": anchor_value, "full_at": sc["full_at"]},
            "reward": round(clip(score / anchor_value / sc["full_at"]), 6)}


# ---------------------------------------------------------------------------------------------- model checks
def model_text(model, name):
    """The custom text `name` of a compiled model, or None (the duct model writes `ducts/mode` and `ducts/params_sha256`)."""
    import mujoco
    i = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_TEXT, name)
    if i < 0:
        return None
    a, n = int(model.text_adr[i]), int(model.text_size[i])
    return bytes(model.text_data[a:a + n - 1]).decode("ascii")


def check_model(model, world=None, variant="track", pushable=None):
    """Errors of a compiled MuJoCo model against the contract.

    Every geom must carry a contract name and the group and contact bits of its rule; every LiDAR target (groups 2
    and 4) has an alpha above 0. With `world`, the world's ducts, clutter, slabs (and, for variant 'avoid',
    boxes) must be present with the geometry of the file. `pushable` True / False additionally requires / forbids
    pushable ducts: True wants one body `duct/<D>/<k>` per piece, or, when the model's custom text `ducts/mode` is
    `active_window` (the duct model's default), the frozen layer on the world body and at least one slot chain
    `duct/<slot>/<k>` with planar joints; False forbids piece bodies, slot chains and a pushable `ducts/mode`. Check a
    compiled model before a DuctTracker is bound to it: the window switches contact bits at run time."""
    import mujoco
    err = []
    names = {}
    for g in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g)
        if not name:
            err.append("geom %d has no name" % g)
            continue
        names[name] = g
        rule = K.geom_rule(name)
        if rule is None:
            err.append("geom %r is not a contract name" % name)
            continue
        group, cat = rule
        if int(model.geom_group[g]) != group:
            err.append("geom %r is in group %d, the contract says %d" % (name, int(model.geom_group[g]), group))
        bits = (int(model.geom_contype[g]), int(model.geom_conaffinity[g]))
        allowed = [K.CONTACT[cat]] + ([(K.BIT_DUCT, K.DUCT_DUCT_CONAFFINITY)] if cat == "duct" else [])
        if bits not in allowed:
            err.append("geom %r has contype / conaffinity %s, the contract says %s" % (name, bits, allowed[0]))
        if group in (K.GROUP_LIDAR, K.GROUP_CLUTTER):
            mat = int(model.geom_matid[g])
            alpha = float(model.mat_rgba[mat][3]) if mat >= 0 else float(model.geom_rgba[g][3])
            if not alpha > 0.0:
                err.append("geom %r is a LiDAR target with alpha 0 (mj_multiRay skips it; CONTRACT 5.5)" % name)

    def has(objtype, name):
        return mujoco.mj_name2id(model, objtype, name) >= 0

    prefixes = sorted({m.group(1) for m in (re.match(r"^(car/|slow[0-9]/)", n) for n in names) if m})
    for b in range(model.nbody):
        m = re.match(r"^(car/|slow[0-9]/)chassis$", mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or "")
        if m and m.group(1) not in prefixes:
            prefixes.append(m.group(1))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    for pre in prefixes:
        for body in K.CAR_BODIES:
            if not has(mujoco.mjtObj.mjOBJ_BODY, pre + body):
                err.append("car %r: body %r is missing" % (pre, pre + body))
        for joint in K.CAR_JOINTS:
            if not has(mujoco.mjtObj.mjOBJ_JOINT, pre + joint):
                err.append("car %r: joint %r is missing" % (pre, pre + joint))
        for geom in K.CAR_TYRE_GEOMS + K.CAR_LIDAR_GEOMS + (("tail_box/col", "tail_box/lidar") if pre != K.EGO_PREFIX else ()):
            if pre + geom not in names:
                err.append("car %r: geom %r is missing" % (pre, pre + geom))
        if pre == K.EGO_PREFIX and (pre + "tail_box/lidar" in names or pre + "tail_box/col" in names):
            err.append("the ego car carries a tail box")
        cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, pre + "chassis")
        if cid < 0:
            continue
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, pre + "root")
        if jid >= 0 and (int(model.jnt_type[jid]) != int(mujoco.mjtJoint.mjJNT_FREE) or int(model.jnt_bodyid[jid]) != cid):
            err.append("car %r: joint %r is not the free joint of the chassis" % (pre, pre + "root"))
        for site, want in (("lidar", K.LIDAR_POS_C), ("imu", K.IMU_POS_C), ("ref", K.REF_POINT_C)):
            sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, pre + site)
            if sid < 0:
                err.append("car %r: site %r is missing" % (pre, pre + site))
            elif int(model.site_bodyid[sid]) != cid or float(np.abs(model.site_pos[sid] - np.asarray(want)).max()) > 1e-6:
                err.append("car %r: site %r is not at %s of the chassis frame" % (pre, pre + site, want))
            elif float(np.abs(model.site_quat[sid] - np.array([1.0, 0, 0, 0])).max()) > 1e-9:
                err.append("car %r: site %r is rotated against the chassis frame" % (pre, pre + site))
        for name, g in names.items():
            if name.startswith(pre) and int(model.geom_group[g]) == K.GROUP_LIDAR and int(model.geom_bodyid[g]) != cid:
                err.append("car %r: LiDAR target %r is not attached to the chassis body" % (pre, name))
    if world is None:
        return err

    t = world["truth"]
    if "floor" not in names:
        err.append("the floor geom is missing")
    duct_mode = model_text(model, "ducts/mode")
    window = duct_mode == "active_window"
    world_ducts = {d["id"] for d in t["ducts"]}
    if pushable is False and duct_mode in ("sleep", "awake", "active_window"):
        err.append("the model's ducts are pushable (custom text ducts/mode = %r) but a static model was asked for" % duct_mode)
    if window and pushable is not False:
        # the slot chains of the duct model's active window: duct geoms of ids that are not ducts of the world, on bodies
        slots = {}
        for name, g in names.items():
            m = re.match(r"^duct/(%s)/([0-9]+)/(col|lidar|vis)$" % K.ID_RE, name)
            if m and m.group(1) not in world_ducts:
                slots.setdefault(m.group(1), []).append((int(m.group(2)), m.group(3), g))
        if pushable is True and not slots:
            err.append("ducts/mode is active_window but the model has no slot chain")
        for sid, items in sorted(slots.items()):
            for k, role, g in items:
                if role == "lidar":
                    err.append("slot chain %s carries a LiDAR target (only the frozen layer is seen)" % sid)
                    continue
                bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, K.duct_body(sid, k))
                if bid < 0 or int(model.geom_bodyid[g]) != bid:
                    err.append("slot chain %s: geom %r is not on body %r" % (sid, K.duct_geom(sid, k, role), K.duct_body(sid, k)))
                elif int(model.body_jntnum[bid]) < 1:
                    err.append("slot chain %s: body %r has no joint" % (sid, K.duct_body(sid, k)))
    for d in t["ducts"]:
        c = d["piece_length_m"]
        for k, piece in enumerate(d["pieces"]):
            for role in ("col", "lidar"):
                name = K.duct_geom(d["id"], k, role)
                if name not in names:
                    err.append("geom %r is missing" % name)
                    continue
                g = names[name]
                along = np.array([math.cos(piece[2]), math.sin(piece[2]), 0.0])
                rel = data.geom_xpos[g] - np.array([piece[0], piece[1], K.DUCT_RADIUS_M])
                axis = data.geom_xmat[g].reshape(3, 3)[:, 2]
                if abs(abs(float(axis @ along)) - 1.0) > 1e-6:
                    err.append("geom %r does not lie along its piece axis" % name)
                if role == "lidar":
                    # the LiDAR target is exactly the piece: a flat-ended cylinder at the piece pose
                    if float(np.abs(rel).max()) > 1e-4:
                        err.append("geom %r is not at its piece pose" % name)
                    if int(model.geom_type[g]) != int(mujoco.mjtGeom.mjGEOM_CYLINDER):
                        err.append("geom %r must be a cylinder (flat ends)" % name)
                    elif (abs(float(model.geom_size[g][0]) - K.DUCT_RADIUS_M) > 1e-6
                          or abs(float(model.geom_size[g][1]) - 0.5 * c) > 1e-6):
                        err.append("geom %r has not the radius and half length of its piece" % name)
                else:
                    # the collision geom sits on the piece axis; at a free end of the duct (the tail of piece 0, the
                    # head of the last piece) it does not reach beyond the end plane, so nothing can be touched where
                    # the LiDAR sees no duct; inside the duct it may overlap its neighbours
                    if float(np.linalg.norm(rel - (rel @ along) * along)) > 1e-4:
                        err.append("geom %r is not on its piece axis" % name)
                    ext = float(model.geom_size[g][1])
                    if int(model.geom_type[g]) == int(mujoco.mjtGeom.mjGEOM_CAPSULE):
                        ext += float(model.geom_size[g][0])
                    mid = float(rel @ along)
                    if (k == 0 and mid - ext < -0.5 * c - 1e-4) or (k == len(d["pieces"]) - 1 and mid + ext > 0.5 * c + 1e-4):
                        err.append("geom %r reaches beyond the end plane of its duct" % name)
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, K.duct_body(d["id"], k))
            if window:
                # the frozen layer: on the world body, at the piece pose (checked above); no per-piece bodies
                if bid >= 0:
                    err.append("body %r exists in an active_window model (pieces are a frozen layer)" % K.duct_body(d["id"], k))
                for role in ("col", "lidar"):
                    gname = K.duct_geom(d["id"], k, role)
                    if gname in names and int(model.geom_bodyid[names[gname]]) != 0:
                        err.append("geom %r of the frozen layer is not on the world body" % gname)
                continue
            if pushable is True and bid < 0:
                err.append("body %r is missing in a pushable model" % K.duct_body(d["id"], k))
            if pushable is False and bid >= 0:
                err.append("body %r exists in a static model" % K.duct_body(d["id"], k))
    for c in t["clutter"]:
        if K.clutter_geom(c["id"]) not in names:
            err.append("geom %r is missing" % K.clutter_geom(c["id"]))
    for p in t["floor"]["patches"]:
        slab = K.zone_geom(p["id"], "slab")
        if p["edge_height_m"] > 0.0 and slab not in names:
            err.append("geom %r is missing (the patch has an edge height)" % slab)
        if p["edge_height_m"] == 0.0 and slab in names:
            err.append("geom %r exists but the patch is flush with the floor" % slab)
    for bx in t["avoidance"]["boxes"]:
        present = has(mujoco.mjtObj.mjOBJ_BODY, K.box_body(bx["id"]))
        if variant == "avoid" and not present:
            err.append("body %r is missing in the avoid variant" % K.box_body(bx["id"]))
        if variant == "track" and present:
            err.append("body %r exists in the track variant" % K.box_body(bx["id"]))
        if present:
            for role in ("col", "lidar"):
                if K.box_geom(bx["id"], role) not in names:
                    err.append("geom %r is missing" % K.box_geom(bx["id"], role))
    return err


# ---------------------------------------------------------------------------------------------- documentation mirror
def constants_rows():
    """The rows of the contract: every public constant of `constants.py` in source order, as JSON."""
    def plain(v):
        if isinstance(v, (tuple, list)):
            return [plain(x) for x in v]
        if isinstance(v, dict):
            return {k: plain(x) for k, x in v.items()}
        return v

    rows = []
    for name, value in vars(K).items():
        if not name.isupper() or name.startswith("_") or name.endswith("_RE") and name != "ID_RE":
            continue
        rows.append("| `%s` | `%s` |" % (name, json.dumps(plain(value)).replace("|", "\\|")))
    return rows


# ---------------------------------------------------------------------------------------------- command line
def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["constants"]:
        print("\n".join(constants_rows()))
        return 0
    if len(argv) >= 2 and argv[0] == "world":
        err = check_world(load_json(argv[1]))
    elif len(argv) >= 2 and argv[0] == "wire":
        with open(argv[1], "r", encoding="utf-8") as f:
            err = check_transcript(f.readlines())
    elif len(argv) >= 3 and argv[0] == "record":
        err = check_record(argv[1], load_json(argv[2]))
    else:
        print(__doc__)
        return 2
    for e in err:
        print("ERROR " + e)
    print("%s: %d error(s)" % (argv[-1], len(err)))
    return 1 if err else 0


if __name__ == "__main__":
    sys.exit(main())
