"""The plan of a collapsed building (cell internals): which bays still stand, which pancaked into crawl voids, which
are filled with rubble, where the openings between bays are and what blocks them, the one route from the street to
the survivor's void and the obstacles placed on it. Geometry is made from this plan by `site.py`.

The building is a reinforced-concrete frame with masonry infill: `nx` x `ny` bays (x east, y north), the facade on
y = 0 facing the street (y < 0). Openings form a spanning tree of the passable bays (plus the street), so the route
from the deployment point to the survivor is unique and its obstacles cannot be bypassed; every other branch is a
dead end. All draws come from one numpy Generator seeded with the site seed.
"""

import math

import numpy as np

__all__ = ["draw_plan", "CHALLENGES", "ROBOT"]

# the crawler's envelope the obstacles are sized against (robot.CRAWLER; repeated here as plain numbers)
ROBOT = {"width_narrow": 0.51, "width_wide": 0.71, "height_stowed": 0.555, "height_mast_up": 1.005,
         "step_tracks": 0.25, "step_flippers": 0.36, "gap_flippers": 0.55, "gap_folded": 0.25}

# obstacle kinds placed on openings
CHALLENGES = ("debris", "narrow", "low", "step", "gap")


def _shuffle(rng, xs):
    xs = list(xs)
    rng.shuffle(xs)
    return xs


def draw_plan(seed, kind="hidden"):
    """The plan of site `seed`. kind: "hidden" / "training" (the same generator) or "lanes" (the test lanes)."""
    if kind == "lanes":
        return _lanes_plan(seed)
    rng = np.random.default_rng(int(seed) & ((1 << 63) - 1))
    nx = int(rng.integers(5, 7))
    ny = 3
    wx = [float(rng.uniform(4.2, 5.2)) for _ in range(nx)]
    wy = [float(rng.uniform(5.0, 5.8))] + [float(rng.uniform(4.2, 5.0)) for _ in range(ny - 1)]
    xs = [0.0]
    for w in wx:
        xs.append(xs[-1] + w)
    ys = [0.0]
    for w in wy:
        ys.append(ys[-1] + w)
    plan = {"seed": int(seed), "kind": kind, "nx": nx, "ny": ny, "xs": xs, "ys": ys, "storeys": int(rng.integers(3, 5))}
    # ---- damage: a collapse centre in the back half; damage falls off with distance plus noise
    cxc = float(rng.uniform(0.25, 0.75)) * xs[-1]
    cyc = float(rng.uniform(0.55, 1.0)) * ys[-1]
    stair = (int(rng.integers(1, nx - 1)), 1)
    bays = {}
    for i in range(nx):
        for j in range(ny):
            bx = (xs[i] + xs[i + 1]) / 2
            by = (ys[j] + ys[j + 1]) / 2
            dist = math.hypot((bx - cxc) / xs[-1], (by - cyc) / ys[-1])
            dmg = 1.0 - dist * 1.6 + float(rng.normal(0, 0.25))
            if (i, j) == stair:
                state = "stair"
            elif dmg > 0.55:
                state = "filled" if rng.random() < 0.25 else "void"
            elif dmg > 0.15:
                state = "void" if rng.random() < 0.55 else "damaged"
            else:
                state = "standing" if rng.random() < 0.6 else "damaged"
            room = "shop" if j == 0 else ("flat" if j == ny - 1 else rng.choice(["flat", "office", "corridor"]))
            if (i, j) == stair:
                room = "stair"
            bays[(i, j)] = {"i": i, "j": j, "state": state, "room": str(room), "dmg": dmg,
                            "x0": xs[i], "x1": xs[i + 1], "y0": ys[j], "y1": ys[j + 1]}
    # at least two void bays in the back rows (the survivor's candidates)
    voids = [k for k, b in bays.items() if b["state"] == "void" and k[1] >= 1]
    while len(voids) < 2:
        cand = [k for k, b in bays.items() if k[1] >= 1 and b["state"] in ("damaged", "standing", "filled")]
        k = cand[int(rng.integers(len(cand)))]
        bays[k]["state"] = "void"
        voids = [k for k, b in bays.items() if b["state"] == "void" and k[1] >= 1]
    # the front row is never filled (the robot has to get in somewhere) and keeps one entry
    for i in range(nx):
        if bays[(i, 0)]["state"] == "filled":
            bays[(i, 0)]["state"] = "damaged"
    # void parameters: the dropped slab's height over the floor (crawl clearance) and its form
    for k, b in bays.items():
        if b["state"] == "void":
            b["void_form"] = str(rng.choice(["pancake", "leanto", "pancake", "vshape"]))
            b["slab_h"] = float(rng.uniform(0.76, 0.95))
            b["leanto_side"] = str(rng.choice(["W", "E", "N", "S"]))
        if b["state"] == "damaged":
            b["piles"] = int(rng.integers(1, 3))
    plan["bays"] = bays
    # ---- the survivor's bay: a void in the back rows, far from the street
    def far_key(k):
        return k[1] * 10 + abs(k[0] - nx / 2.0) * 0.5 + float(rng.random())
    target = max(voids, key=far_key)
    plan["target"] = target
    # ---- the spanning tree of openings: passable bays + the street node "S"
    passable = [k for k, b in bays.items() if b["state"] not in ("filled", "stair")]
    edges = []
    for (i, j) in passable:
        for di, dj in ((1, 0), (0, 1)):
            k2 = (i + di, j + dj)
            if k2 in bays and k2 in passable:
                edges.append(((i, j), k2))
    # entries from the street: front-row bays
    street_edges = [("S", (i, 0)) for i in range(nx) if (i, 0) in passable]
    # random spanning tree (randomised Kruskal) over the bays, then connect the street to ONE front bay
    parent = {k: k for k in passable}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    tree = []
    for e in _shuffle(rng, edges):
        a, b = find(e[0]), find(e[1])
        if a != b:
            parent[a] = b
            tree.append(e)
    # components: keep the one with the target; others are cut off (isolated rooms behind rubble)
    comp = find(target)
    tree = [e for e in tree if find(e[0]) == comp]
    in_comp = set(k for k in passable if find(k) == comp)
    front = [k for k in in_comp if k[1] == 0]
    if not front:
        # make sure the component reaches the front: force the column of the target open down to the facade
        i = target[0]
        for j in range(target[1] - 1, -1, -1):
            if bays[(i, j)]["state"] in ("filled", "stair"):
                bays[(i, j)]["state"] = "damaged"
                bays[(i, j)]["piles"] = 1
            tree.append(((i, j), (i, j + 1)))
            in_comp.add((i, j))
        front = [(i, 0)]
    entry = front[int(rng.integers(len(front)))]
    plan["entry"] = entry
    # the entry bay keeps its ceiling up: the street's rubble spill rises into it, which a dropped slab would close
    if bays[entry]["state"] == "void":
        bays[entry]["state"] = "damaged"
        bays[entry]["piles"] = 1
    # graph adjacency
    adj = {k: [] for k in in_comp}
    for a, b in tree:
        adj[a].append(b)
        adj[b].append(a)
    # route = path entry -> target in the tree
    prev = {entry: None}
    stack = [entry]
    while stack:
        u = stack.pop()
        for v in adj[u]:
            if v not in prev:
                prev[v] = u
                stack.append(v)
    path = [target]
    while path[-1] != entry:
        path.append(prev[path[-1]])
    path.reverse()
    # a route of at least 3 bays: if the tree path is too short, pick another target among the voids
    if len(path) < 3:
        best = None
        for v in voids:
            if v not in prev:
                continue
            p = [v]
            while p[-1] != entry:
                p.append(prev[p[-1]])
            if best is None or len(p) > len(best):
                best = p[::-1]
        if best is not None and len(best) > len(path):
            path = best
            target = best[-1]
            plan["target"] = target
    plan["route_bays"] = path
    # ---- openings
    openings = []
    route_pairs = set()
    for a, b in zip(path[:-1], path[1:]):
        route_pairs.add((a, b))
        route_pairs.add((b, a))
    for a, b in tree:
        openings.append(_opening(plan, a, b, rng, on_route=(a, b) in route_pairs))
    # the street entry
    openings.append(_street_opening(plan, entry, rng))
    # dead-end facade openings that look like ways in (blocked by heavy rubble or a prop)
    for i in range(nx):
        if (i, 0) != entry and rng.random() < 0.5:
            o = _street_opening(plan, (i, 0), rng)
            o["challenge"] = str(rng.choice(["blocked", "blocked", "prop"]))
            o["on_route"] = False
            openings.append(o)
    # extra false openings between non-tree neighbours (blocked by heavy debris or props: never passable)
    for e in edges:
        if e not in tree and (e[1], e[0]) not in tree and rng.random() < 0.3:
            o = _opening(plan, e[0], e[1], rng, on_route=False)
            o["challenge"] = str(rng.choice(["blocked", "prop", "blocked"]))
            openings.append(o)
    plan["openings"] = openings
    # ---- challenges on the route: every kind at least once where possible, in a random order
    route_ops = [o for o in openings if o["on_route"]]
    route_ops.sort(key=lambda o: o["order"])
    kinds = ["debris", "narrow", "step", "gap"]
    rng.shuffle(kinds)
    # the street entry gets a rubble spill (a slope) and maybe one more obstacle
    n = len(route_ops)
    assign = {}
    pool = list(kinds)
    # spread: the first opening after the street is the facade (spill); assign kinds to the interior openings first
    interior = [o for o in route_ops if o["a"] != "S"]
    for o in interior:
        if pool:
            assign[id(o)] = pool.pop()
    # leftovers go to the facade opening (with the spill) or are doubled up (a debris and a step on one opening)
    for o in route_ops:
        if o["a"] == "S":
            assign[id(o)] = pool.pop() if pool else "spill"
    # kinds left over on a short route are dropped (two obstacles in one doorway make a trap, not a task)
    pool = []
    for o in route_ops:
        o["challenge"] = assign.get(id(o), "none")
    # a step leads up onto a fallen slab: the bay it leads into keeps its ceiling (step + robot > a crawl void)
    for o in route_ops:
        ks = [o.get("challenge")] + list(o.get("extra", []))
        if "step" in ks and o["b"] != "S":
            far = o["b"] if (o["a"] == "S" or path.index(o["b"]) > path.index(o["a"])) else o["a"]
            if bays[far]["state"] == "void" and far != plan["target"]:
                bays[far]["state"] = "damaged"
                bays[far]["piles"] = 1
            elif far == plan["target"]:
                # never into the survivor's void: trade the step for a gap
                if o.get("challenge") == "step":
                    o["challenge"] = "gap"
                else:
                    o["extra"] = ["gap" if k == "step" else k for k in o.get("extra", [])]
    # off-route tree openings: some obstacles too (a dead end costs as much as a way in)
    for o in openings:
        if not o["on_route"] and o.get("challenge") is None:
            o["challenge"] = str(rng.choice(["none", "none", "debris", "narrow", "step", "blocked"]))
    # the target void is always a crawl ("low") at its far end; the void's slab defines the clearance
    plan["survivor"] = _survivor(plan, rng)
    plan["decoys"] = _decoys(plan, rng)
    plan["start"] = _start(plan, rng)
    plan["env"] = {"ambient_c": float(rng.uniform(8.0, 24.0)), "sun_elev": float(rng.uniform(25.0, 65.0)),
                   "sun_az": float(rng.uniform(-60.0, 60.0)), "sun": float(rng.uniform(0.55, 1.0)),
                   "haze": float(rng.uniform(0.3, 1.0))}
    return plan


def _opening(plan, a, b, rng, on_route):
    """An opening in the wall between bays a and b (a, b adjacent)."""
    bays = plan["bays"]
    A, B = bays[a], bays[b]
    if a[0] != b[0]:                       # wall along y at x = boundary
        i = max(a[0], b[0])
        x = plan["xs"][i]
        j = a[1]
        y0, y1 = plan["ys"][j], plan["ys"][j + 1]
        pos = float(rng.uniform(y0 + 1.2, y1 - 1.2))
        o = {"a": a, "b": b, "axis": "x", "x": x, "y": pos, "wall_lo": y0, "wall_hi": y1}
    else:                                   # wall along x at y = boundary
        j = max(a[1], b[1])
        y = plan["ys"][j]
        i = a[0]
        x0, x1 = plan["xs"][i], plan["xs"][i + 1]
        pos = float(rng.uniform(x0 + 1.2, x1 - 1.2))
        o = {"a": a, "b": b, "axis": "y", "x": pos, "y": y, "wall_lo": x0, "wall_hi": x1}
    o["on_route"] = bool(on_route)
    o["width"] = float(rng.uniform(0.95, 1.30))   # a doorway or a breach in the wall
    o["form"] = "door" if (A["state"] == "standing" and B["state"] == "standing" and rng.random() < 0.7) else "breach"
    o["challenge"] = None
    order = 0
    if on_route:
        rb = plan.get("route_bays") or []
        try:
            order = min(rb.index(a), rb.index(b))
        except ValueError:
            order = 0
    o["order"] = order + 1
    o["params"] = {"rng": float(rng.random()), "rng2": float(rng.random()), "rng3": float(rng.random())}
    return o


def _street_opening(plan, bay, rng):
    i = bay[0]
    x0, x1 = plan["xs"][i], plan["xs"][i + 1]
    pos = float(rng.uniform(x0 + 1.3, x1 - 1.3))
    return {"a": "S", "b": bay, "axis": "y", "x": pos, "y": 0.0, "wall_lo": x0, "wall_hi": x1, "on_route": True,
            "width": float(rng.uniform(1.4, 2.2)), "form": "shopfront", "challenge": None, "order": 0,
            "params": {"rng": float(rng.random()), "rng2": float(rng.random()), "rng3": float(rng.random())}}


def _survivor(plan, rng):
    """Where the survivor lies in the target void: against the bay's far side from the route's last opening, in the
    pocket a sturdy piece of furniture kept open; legs under debris."""
    t = plan["target"]
    b = plan["bays"][t]
    rb = plan["route_bays"]
    prev = rb[-2] if len(rb) >= 2 else None
    # the side of the bay the robot comes from
    if prev is None:
        come = "S"
    elif prev[0] < t[0]:
        come = "W"
    elif prev[0] > t[0]:
        come = "E"
    elif prev[1] < t[1]:
        come = "S"
    else:
        come = "N"
    far = {"W": "E", "E": "W", "S": "N", "N": "S"}[come]
    x0, x1, y0, y1 = b["x0"], b["x1"], b["y0"], b["y1"]
    m = 0.75
    if far in ("E", "W"):
        x = x1 - m if far == "E" else x0 + m
        y = float(rng.uniform(y0 + 1.4, y1 - 1.4))
        yaw = math.pi / 2 if rng.random() < 0.5 else -math.pi / 2          # lying along the wall
    else:
        y = y1 - m if far == "N" else y0 + m
        x = float(rng.uniform(x0 + 1.4, x1 - 1.4))
        yaw = 0.0 if rng.random() < 0.5 else math.pi
    return {"x": x, "y": y, "yaw": yaw, "far": far, "come": come, "pose": str(rng.choice(["side", "back", "prone"])),
            "skin": int(rng.integers(0, 4)), "shirt": int(rng.integers(0, 5)), "trousers": int(rng.integers(0, 4)),
            "legs_covered": bool(rng.random() < 0.8), "skin_c": float(rng.uniform(32.5, 34.5)),
            "cloth_c": float(rng.uniform(27.0, 31.0))}


def _decoys(plan, rng):
    """Things that look like a person to a camera (a mannequin in a shop, a heap of clothes) or warm to a thermal
    camera (a boiler that is still warm, a sun-heated car body on the street)."""
    out = []
    shops = [k for k, b in plan["bays"].items() if b["room"] == "shop" and b["state"] != "filled"]
    if shops and rng.random() < 0.8:
        k = shops[int(rng.integers(len(shops)))]
        b = plan["bays"][k]
        out.append({"type": "mannequin", "bay": k, "x": float(rng.uniform(b["x0"] + 0.8, b["x1"] - 0.8)),
                    "y": float(rng.uniform(b["y0"] + 0.8, b["y1"] - 0.8)), "yaw": float(rng.uniform(-math.pi, math.pi))})
    flats = [k for k, b in plan["bays"].items() if b["room"] in ("flat", "office") and b["state"] in ("standing", "damaged")
             and k != plan["target"]]
    if flats:
        k = flats[int(rng.integers(len(flats)))]
        b = plan["bays"][k]
        out.append({"type": "boiler", "bay": k, "temp_c": float(rng.uniform(38.0, 55.0)),
                    "x": b["x0"] + 0.35, "y": float(rng.uniform(b["y0"] + 0.8, b["y1"] - 0.8))})
    if flats and rng.random() < 0.6:
        k = flats[int(rng.integers(len(flats)))]
        b = plan["bays"][k]
        out.append({"type": "clothes", "bay": k, "x": float(rng.uniform(b["x0"] + 0.8, b["x1"] - 0.8)),
                    "y": float(rng.uniform(b["y0"] + 0.8, b["y1"] - 0.8)), "yaw": float(rng.uniform(-math.pi, math.pi))})
    return out


def _start(plan, rng):
    """The deployment point on the street, facing the building, a few metres from the entry."""
    entry_op = [o for o in plan["openings"] if o["a"] == "S" and o["on_route"]][0]
    x = entry_op["x"] + float(rng.uniform(-1.2, 1.2))
    x = min(max(x, 1.0), plan["xs"][-1] - 1.0)
    y = float(rng.uniform(-5.4, -4.6))
    yaw = math.pi / 2 + float(rng.uniform(-0.2, 0.2))
    return {"x": x, "y": y, "yaw": yaw}


def _lanes_plan(seed):
    """The test lanes of the training ground: one corridor of bays, one obstacle of each kind in turn, a dummy at the
    end (the standard test methods for response robots, laid out the way the sites use them)."""
    rng = np.random.default_rng(int(seed))
    nx, ny = 6, 1
    wx = [4.0] * nx
    xs = [0.0]
    for w in wx:
        xs.append(xs[-1] + w)
    ys = [0.0, 4.5]
    bays = {}
    for i in range(nx):
        bays[(i, 0)] = {"i": i, "j": 0, "state": "standing" if i < nx - 1 else "void", "room": "lane", "dmg": 0.0,
                        "x0": xs[i], "x1": xs[i + 1], "y0": ys[0], "y1": ys[1]}
    bays[(nx - 1, 0)].update({"void_form": "pancake", "slab_h": 0.80, "leanto_side": "N"})
    plan = {"seed": int(seed), "kind": "lanes", "nx": nx, "ny": ny, "xs": xs, "ys": ys, "storeys": 1, "bays": bays,
            "target": (nx - 1, 0), "entry": (0, 0), "route_bays": [(i, 0) for i in range(nx)]}
    ops = [_street_opening(plan, (0, 0), rng)]
    ops[0]["challenge"] = "spill"
    kinds = ["debris", "step", "narrow", "gap", "low"]
    for i in range(nx - 1):
        o = _opening(plan, (i, 0), (i + 1, 0), rng, on_route=True)
        o["y"] = 2.25
        o["order"] = i + 1
        o["form"] = "breach"
        o["challenge"] = kinds[i]
        ops.append(o)
    plan["openings"] = ops
    plan["survivor"] = {"x": xs[-1] - 0.75, "y": 2.25, "yaw": math.pi / 2, "far": "E", "come": "W", "pose": "back",
                        "skin": 1, "shirt": 2, "trousers": 1, "legs_covered": True, "skin_c": 33.5, "cloth_c": 29.0,
                        "dummy": True}
    plan["decoys"] = []
    plan["start"] = {"x": 2.0, "y": -4.5, "yaw": math.pi / 2}
    plan["env"] = {"ambient_c": 18.0, "sun_elev": 50.0, "sun_az": 20.0, "sun": 0.9, "haze": 0.3}
    return plan
