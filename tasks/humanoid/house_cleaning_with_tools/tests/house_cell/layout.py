"""Per-episode layout draw: the mess after the customer's dinner (where every loose object lies and how it faces, the
grease stain's shape and place, the crumbs), which bin stands where, and the robot's start pose. A pure function of
the episode seed; `nominal` gives the privileged reference's fixed layout.

The dinner happened at the dining table: two plates, two glasses, a can, two crumpled napkins and two books on its top,
a greasy spill between the place settings, crumbs on the top around the plates and on the floor under and around the
table. The coffee table holds the rest of the trash: a can, a water bottle, a napkin and an apple core. The tools are where they belong: the sponge in its tray by the sink, the broom and the dustpan in
their clips on the south wall.
"""

from __future__ import annotations

import math

import numpy as np

from . import objects as O
from . import room as RM

GREASE_CELL = 0.020          # m: the grease film's cell size on the dining table
GREASE_MARGIN = 0.06         # m: clean cells round the stain (a dirty sponge can smear grease onto them)
N_TABLE_CRUMBS = 14
N_FLOOR_CRUMBS = 46


def _quat_z(yaw):
    return np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])


def _qmul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def _quat_R(R):
    R = np.asarray(R, float)
    t = np.trace(R)
    if t > 0:
        s = 0.5 / math.sqrt(t + 1.0)
        return np.array([0.25 / s, (R[2, 1] - R[1, 2]) * s, (R[0, 2] - R[2, 0]) * s, (R[1, 0] - R[0, 1]) * s])
    i = int(np.argmax(np.diag(R)))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = 2.0 * math.sqrt(1.0 + R[i, i] - R[j, j] - R[k, k])
    q = np.zeros(4)
    q[0] = (R[k, j] - R[j, k]) / s
    q[1 + i] = 0.25 * s
    q[1 + j] = (R[j, i] + R[i, j]) / s
    q[1 + k] = (R[k, i] + R[i, k]) / s
    return q


def _place_discs(rng, specs, region, placed, tries=400):
    """Rejection-sample disc footprints: specs [(name, radius)], region (x0, y0, x1, y1) for the centres; `placed`
    [(x, y, r)] already there (extended in place). Returns {name: (x, y)}."""
    out = {}
    x0, y0, x1, y1 = region
    for name, r in specs:
        for _ in range(tries):
            x, y = rng.uniform(x0 + r, x1 - r), rng.uniform(y0 + r, y1 - r)
            if all(math.hypot(x - px, y - py) >= r + pr + 0.015 for px, py, pr in placed):
                break
        else:
            raise RuntimeError("layout: no room for %s" % name)
        placed.append((x, y, r))
        out[name] = (x, y)
    return out


def _grease(rng, center, nominal):
    """The stain: the union of 2-4 soft ellipses; a grid of GREASE_CELL cells over its bounding box plus a margin.
    Returns {"cell", "cells": [(i, j, x, y)], "g0": [grease per cell, 0..1], "bbox"}."""
    cx, cy = center
    n = 3 if nominal else int(rng.integers(2, 5))
    blobs = []
    for k in range(n):
        if nominal:
            dx, dy, sx, sy, a = ((0.0, 0.0, 0.07, 0.05, 0.3), (0.07, 0.03, 0.05, 0.035, -0.4),
                                 (-0.06, -0.025, 0.045, 0.03, 1.0))[k]
        else:
            dx, dy = rng.uniform(-0.08, 0.08), rng.uniform(-0.05, 0.05)
            sx, sy = rng.uniform(0.035, 0.075), rng.uniform(0.025, 0.05)
            a = rng.uniform(0, math.pi)
        blobs.append((cx + dx, cy + dy, sx, sy, a))
    half = (0.18, 0.13)
    x0, x1 = cx - half[0] - GREASE_MARGIN, cx + half[0] + GREASE_MARGIN
    y0, y1 = cy - half[1] - GREASE_MARGIN, cy + half[1] + GREASE_MARGIN
    nx, ny = int(round((x1 - x0) / GREASE_CELL)), int(round((y1 - y0) / GREASE_CELL))
    cells, g0 = [], []
    for i in range(nx):
        for j in range(ny):
            x = x0 + (i + 0.5) * GREASE_CELL
            y = y0 + (j + 0.5) * GREASE_CELL
            v = 0.0
            for bx, by, sx, sy, a in blobs:
                ca, sa = math.cos(a), math.sin(a)
                u = ((x - bx) * ca + (y - by) * sa) / sx
                w = (-(x - bx) * sa + (y - by) * ca) / sy
                v += math.exp(-0.5 * (u * u + w * w))
            g = min(1.0, max(0.0, 1.35 * v - 0.22))
            cells.append((i, j, x, y))
            g0.append(g if g > 0.03 else 0.0)
    return {"cell": GREASE_CELL, "cells": cells, "g0": g0, "bbox": (x0, y0, x1, y1), "shape": (nx, ny),
            "blobs": blobs}


def clip_pose(tool):
    """(pos, quat) of a tool hanging in its clip (its handle vertical through the jaw centre)."""
    cx, cy, cz = RM.CLIPS[tool]
    if tool == "broom":
        z1 = O.BROOM["handle"][1]
        # the handle's top 0.30 m above the clip: the head's bristles (0.08 m below the swivel axis) 2 cm off the floor
        return np.array([cx, cy, cz + 0.30 - z1]), np.array([1.0, 0.0, 0.0, 0.0])
    # the dustpan: rotate its frame so the handle is vertical (Rx(tilt)), then turn it to face the room (Rz(pi)):
    # the lip towards the room, the back towards the wall
    tilt = O.DUSTPAN["handle_tilt"]
    c, s = math.cos(tilt), math.sin(tilt)
    Rx = np.array([[1, 0, 0], [0, c, -s], [0, s, c]], float)
    Rz = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]], float)
    R = Rz @ Rx
    base, axis = O.dustpan_handle_axis()
    # the handle point 0.62 m along the handle sits in the clip
    p_local = base + axis * 0.62
    pos = np.array([cx, cy, cz]) - R @ p_local
    return pos, _quat_R(R)


def draw(seed: int, nominal: bool = False) -> dict:
    """The episode's layout; a draw that leaves no room for something is drawn again (from the same seed's stream)."""
    rng = np.random.default_rng(int(seed) & ((1 << 64) - 1))
    for _ in range(50):
        try:
            return _draw(rng, nominal)
        except RuntimeError:
            if nominal:
                raise
    raise RuntimeError("layout: no arrangement found")


def _draw(rng, nominal: bool) -> dict:
    objects, facts = {}, {}
    # ---- bins: which stands where (slot 0 = south, slot 1 = north)
    bins = ("recycling", "general") if (nominal or rng.integers(2) == 0) else ("general", "recycling")
    # ---- the dining table: two place settings, the spill between them
    D = RM.DINING
    (dx, dy), (lx, ly) = D["center"], D["size"]
    top = D["top_z"]
    region = (dx - lx / 2 + 0.04, dy - ly / 2 + 0.04, dx + lx / 2 - 0.04, dy + ly / 2 - 0.04)
    placed = []
    if nominal:
        plates = {"plate_1": (dx - 0.30, dy - 0.15), "plate_2": (dx + 0.28, dy + 0.15)}
        for k, (x, y) in plates.items():
            placed.append((x, y, O.PLATE["r"]))
    else:
        sides = rng.permutation([-1, 1])
        plates = {}
        for k, sgn in zip(("plate_1", "plate_2"), sides):
            x = dx + float(rng.uniform(-0.36, 0.36))
            y = dy + sgn * float(rng.uniform(0.12, 0.17))
            if placed and abs(x - placed[0][0]) < 0.30:
                x = dx - math.copysign(0.30, placed[0][0] - dx) + float(rng.uniform(-0.05, 0.05))
            plates[k] = (x, y)
            placed.append((x, y, O.PLATE["r"]))
    # the stain: between the two plates (it may run under one), on the table
    px = (plates["plate_1"][0] + plates["plate_2"][0]) / 2
    py = (plates["plate_1"][1] + plates["plate_2"][1]) / 2
    gc = (px, py) if nominal else (px + float(rng.uniform(-0.12, 0.12)), py + float(rng.uniform(-0.05, 0.05)))
    gc = (float(np.clip(gc[0], dx - lx / 2 + 0.25, dx + lx / 2 - 0.25)),
          float(np.clip(gc[1], dy - ly / 2 + 0.20, dy + ly / 2 - 0.20)))
    grease = _grease(rng, gc, nominal)
    # the books (the long side of each along the table's depth, a little turned): placed next, before the small things
    if nominal:
        books_spots = {"book_red": (dx - 0.46, dy + 0.10), "book_blue": (dx + 0.47, dy - 0.10)}
    else:
        books_spots = _place_discs(rng, [("book_red", 0.118), ("book_blue", 0.110)], region, placed)
    # glasses near their plates, then the can and two napkins anywhere free
    for g, p in (("glass_1", "plate_1"), ("glass_2", "plate_2")):
        x0, y0 = plates[p]
        found = False
        for _ in range(200):
            a = (rng.uniform(0, 2 * math.pi) if not nominal else (0.6 if g == "glass_1" else 3.7))
            x, y = x0 + 0.19 * math.cos(a), y0 + 0.19 * math.sin(a)
            if nominal or (region[0] + 0.04 < x < region[2] - 0.04 and region[1] + 0.04 < y < region[3] - 0.04 and
                           all(math.hypot(x - qx, y - qy) >= 0.035 + qr + 0.015 for qx, qy, qr in placed)):
                found = True
                break
        if found:
            placed.append((x, y, O.GLASS["r"]))
        else:                                   # anywhere free on the table
            x, y = _place_discs(rng, [(g, O.GLASS["r"])], region, placed)[g]
        objects[g] = (np.array([x, y, top + 0.001]), np.array([1.0, 0.0, 0.0, 0.0]))
    if nominal:
        spots = {"can_red": (dx - 0.05, dy + 0.27), "napkin_1": (dx + 0.12, dy - 0.27), "napkin_2": (dx - 0.48, dy + 0.24)}
    else:
        spots = _place_discs(rng, [("can_red", O.CAN["r"]), ("napkin_1", O.NAPKIN["r"]), ("napkin_2", O.NAPKIN["r"])],
                             region, placed)
    for k, (x, y) in plates.items():
        objects[k] = (np.array([x, y, top + 0.001]), _quat_z(0.0 if nominal else rng.uniform(0, 2 * math.pi)))
    objects["can_red"] = (np.array([spots["can_red"][0], spots["can_red"][1], top + 0.001]),
                          _quat_z(0.0 if nominal else rng.uniform(0, 2 * math.pi)))
    for k in ("napkin_1", "napkin_2"):
        objects[k] = (np.array([spots[k][0], spots[k][1], top + O.NAPKIN["r"] * 0.78 + 0.001]),
                      _quat_z(0.0 if nominal else rng.uniform(0, 2 * math.pi)))
    # ---- the two books, left on the dining table (lying on their back covers, never on one another)
    lie = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], float)    # body y (thickness) up, body z (height) along -y
    bspots = books_spots
    for bk in ("book_red", "book_blue"):
        w, t, h = O.LOOSE[bk][1]["size"]
        yaw_b = (math.pi / 2 if nominal else float(rng.choice([-1.0, 1.0])) * math.pi / 2 + float(rng.uniform(-0.35, 0.35)))
        Rz = np.array([[math.cos(yaw_b), -math.sin(yaw_b), 0], [math.sin(yaw_b), math.cos(yaw_b), 0], [0, 0, 1]])
        x, y = bspots[bk]
        objects[bk] = (np.array([x, y, top + 0.001 + t / 2]), _quat_R(Rz @ lie))
        facts[bk] = {"yaw": yaw_b}
    # ---- the coffee table: the rest of the trash, in its west half (within the robot's reach from the west)
    C = RM.COFFEE
    (cx, cy), (cw, cl) = C["center"], C["size"]
    ctop = C["top_z"]
    trash_region = (cx - cw / 2 + 0.03, cy - cl / 2 + 0.06, cx - cw / 2 + 0.27, cy + cl / 2 - 0.06)
    cplaced = []
    if nominal:
        cspots = {"can_green": (cx - 0.17, cy - 0.25), "bottle": (cx - 0.12, cy - 0.08),
                  "napkin_3": (cx - 0.18, cy + 0.07), "apple_core": (cx - 0.11, cy + 0.22)}
    else:
        cspots = _place_discs(rng, [("bottle", O.BOTTLE["r"]), ("can_green", O.CAN["r"]),
                                    ("napkin_3", O.NAPKIN["r"]), ("apple_core", 0.045)], trash_region, cplaced)
    objects["can_green"] = (np.array([cspots["can_green"][0], cspots["can_green"][1], ctop + 0.001]),
                            _quat_z(0.0 if nominal else rng.uniform(0, 2 * math.pi)))
    objects["bottle"] = (np.array([cspots["bottle"][0], cspots["bottle"][1], ctop + 0.001]),
                         _quat_z(0.0 if nominal else rng.uniform(0, 2 * math.pi)))
    objects["napkin_3"] = (np.array([cspots["napkin_3"][0], cspots["napkin_3"][1],
                                     ctop + O.NAPKIN["r"] * 0.78 + 0.001]),
                           _quat_z(0.0 if nominal else rng.uniform(0, 2 * math.pi)))
    objects["apple_core"] = (np.array([cspots["apple_core"][0], cspots["apple_core"][1],
                                       ctop + O.APPLE["r"] * 0.9 + 0.001]),
                             _quat_z(0.0 if nominal else rng.uniform(0, 2 * math.pi)))
    # ---- the tools at home
    T = RM.SPONGE_TRAY
    sl = O.SPONGE["size"]
    objects["sponge"] = (np.array([T["center"][0], T["center"][1], RM.KITCHEN["top_z"] + 0.002 + sl[2] / 2 + 0.0005]),
                         _quat_z(math.pi / 2 + (0.0 if nominal else rng.uniform(-0.15, 0.15))))
    for tool in ("broom", "dustpan"):
        objects[tool] = clip_pose(tool)
    # ---- crumbs: on the dining table round the plates, on the floor under and round the table
    crumbs, radii = [], []
    for k in range(N_TABLE_CRUMBS):
        p = plates["plate_1" if k % 2 == 0 else "plate_2"]
        for _ in range(100):
            a, r = rng.uniform(0, 2 * math.pi), rng.uniform(0.14, 0.24)
            x, y = p[0] + r * math.cos(a), p[1] + r * math.sin(a)
            if region[0] < x < region[2] and region[1] < y < region[3] and \
                    all(math.hypot(x - qx, y - qy) >= qr + 0.012 for qx, qy, qr in placed):
                break
        rr = float(rng.uniform(0.0035, 0.0065))
        crumbs.append((x, y, top + rr * 0.55))
        radii.append(rr)
    obstacles = floor_obstacles()
    clusters = []
    n_cl = 3 if nominal else int(rng.integers(2, 4))
    for k in range(n_cl):
        side = (-1, 1)[k % 2] if nominal else int(rng.choice([-1, 1]))
        clusters.append((dx + float(rng.uniform(-0.50, 0.50)), dy + side * float(rng.uniform(0.52, 0.72)),
                         float(rng.uniform(0.08, 0.16))))
    # (round the table, not under it or under a chair: the floor round the dining set where the crumbs fell)
    keep_out = [(dx - lx / 2 - 0.06, dy - ly / 2 - 0.06, dx + lx / 2 + 0.06, dy + ly / 2 + 0.06)]
    for ch in RM.CHAIRS.values():
        w_, l_ = RM.CHAIR["seat"]
        keep_out.append((ch["center"][0] - w_ / 2 - 0.08, ch["center"][1] - l_ / 2 - 0.08,
                         ch["center"][0] + w_ / 2 + 0.08, ch["center"][1] + l_ / 2 + 0.08))
    for k in range(N_FLOOR_CRUMBS):
        for _ in range(400):
            if k < 6:          # a few strays further out
                x = dx + float(rng.uniform(-1.0, 1.0))
                y = dy + float(rng.uniform(-0.95, 0.95))
            else:
                cxk, cyk, sk = clusters[k % len(clusters)]
                x, y = cxk + float(rng.normal(0, sk)), cyk + float(rng.normal(0, sk * 0.8))
            if 1.60 < x < 4.05 and 1.25 < y < 3.65 and not blocked(x, y, 0.01, obstacles) and \
                    not any(a0 < x < a1 and b0 < y < b1 for a0, b0, a1, b1 in keep_out):
                break
        rr = float(rng.uniform(0.0035, 0.007))
        crumbs.append((x, y, rr * 0.55))
        radii.append(rr)
    # ---- the robot
    x, y, yaw = RM.ROBOT_HOME
    if not nominal:
        x += float(rng.uniform(-0.10, 0.10))
        y += float(rng.uniform(-0.08, 0.08))
        yaw += float(rng.uniform(-0.25, 0.25))
    return {"bins": bins, "objects": objects, "crumbs": crumbs, "crumb_r": radii, "crumb_look_seed": 7,
            "grease": grease, "robot": (x, y, yaw), "facts": facts, "plates": plates}


# ------------------------------------------------------------------------------------------------ floor obstacles
def floor_obstacles():
    """Footprints (x0, y0, x1, y1) of whatever stands on the floor and stops crumbs (the cell's sweeping rule): walls
    are the room's bounds; the kitchen's toe kick, the bins, the table's and chairs' legs, the coffee table's legs,
    the couch's legs, the plant's pot, the bookshelf."""
    obs = []
    K = RM.KITCHEN
    obs.append((K["x0"], K["y0"], K["plinth_x1"], K["y1"]))
    for yc in RM.BIN_SLOTS_Y:
        obs.append((RM.BIN["x0"], yc - RM.BIN["half_y"], RM.BIN["x1"], yc + RM.BIN["half_y"]))
    D = RM.DINING
    (dx, dy), (lx, ly) = D["center"], D["size"]
    lg, ins = D["leg"], D["leg_inset"]
    for sx in (-1, 1):
        for sy in (-1, 1):
            x, y = dx + sx * (lx / 2 - ins), dy + sy * (ly / 2 - ins)
            obs.append((x - lg / 2, y - lg / 2, x + lg / 2, y + lg / 2))
    C = RM.CHAIR
    for ch in RM.CHAIRS.values():
        x, y = ch["center"]
        cy, sy_ = math.cos(ch["yaw"]), math.sin(ch["yaw"])
        w, l = C["seat"]
        for lx_, ly_ in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
            px, py = lx_ * (w / 2 - 0.03), ly_ * (l / 2 - 0.03)
            qx, qy = x + cy * px - sy_ * py, y + sy_ * px + cy * py
            obs.append((qx - C["leg"] / 2, qy - C["leg"] / 2, qx + C["leg"] / 2, qy + C["leg"] / 2))
    CO = RM.COFFEE
    (cx, cy_), (cw, cl) = CO["center"], CO["size"]
    for sx in (-1, 1):
        for sy in (-1, 1):
            x, y = cx + sx * (cw / 2 - CO["leg_inset"]), cy_ + sy * (cl / 2 - CO["leg_inset"])
            obs.append((x - CO["leg"] / 2, y - CO["leg"] / 2, x + CO["leg"] / 2, y + CO["leg"] / 2))
    for (lx_, ly_) in RM.COUCH["legs"]["centers"]:
        s = RM.COUCH["legs"]["side"] / 2
        obs.append((lx_ - s, ly_ - s, lx_ + s, ly_ + s))
    pc, pr = RM.PLANT["pot"]["center"], RM.PLANT["pot"]["radius"]
    obs.append((pc[0] - pr, pc[1] - pr, pc[0] + pr, pc[1] + pr))
    S = RM.SHELF
    obs.append((S["x0"], S["y0"], S["x1"], S["y1"]))
    return obs


def blocked(x, y, r, obstacles):
    if x < r + 0.005 or y < r + 0.005 or x > RM.ROOM_X - r - 0.005 or y > RM.ROOM_Y - r - 0.005:
        return True
    return any(x0 - r < x < x1 + r and y0 - r < y < y1 + r for x0, y0, x1, y1 in obstacles)
