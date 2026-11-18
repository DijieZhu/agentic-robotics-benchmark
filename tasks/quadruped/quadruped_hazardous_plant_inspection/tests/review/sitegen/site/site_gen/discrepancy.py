"""What the drawing gets wrong. The drawing is the original design; reality has changed:
door_blocked, wall_added, opening_added change the truth world; room_missing (the annex), ip_moved,
ip_undocumented, zone_outdated, sheet_missing change only what the prior shows."""

from __future__ import annotations

import math

import numpy as np

from .layout import add_door
from .structure import rebuild_leaves
from .util import uniform
from .world import sample_size


# ------------------------------------------------------------------------------------------------ room graph
def room_graph(W, closed=(), extra_edges=(), split=None):
    """Adjacency between rooms through passable doors. `split`: (leg id, axis, coord) cuts a corridor leg into
    two nodes '<leg>:a' (coord below) and '<leg>:b'."""
    lay = W.lay
    adj = {}

    def node(rid, x, y):
        if split and rid == split[0]:
            c = x if split[1] == "x" else y
            return rid + (":a" if c < split[2] else ":b")
        return rid

    def link(a, b):
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)
    for r in lay.rooms:
        if split and r == split[0]:
            adj.setdefault(r + ":a", set())
            adj.setdefault(r + ":b", set())
        else:
            adj.setdefault(r, set())
    # corridor parts are connected to each other where they touch
    pts = {"C1": None}
    for a, b in (("C1", "C2"), ("C2", "C3")):
        ra, rb = lay.rooms[a].rect, lay.rooms[b].rect
        x = 0.5 * (max(ra[0], rb[0]) + min(ra[2], rb[2]))
        y = 0.5 * (max(ra[1], rb[1]) + min(ra[3], rb[3]))
        link(node(a, x, y), node(b, x, y))
    del pts
    for d in lay.doors:
        if d.state == "closed" or d.id in closed or d.kind in ("airlock_outer", "stair"):
            continue
        a, b = d.rooms
        if a == "OUT" or b == "OUT":
            continue
        x, y = d.xy()
        link(node(a, x, y), node(b, x, y))
    for a, b in extra_edges:
        link(a, b)
    return adj


def connected_all(adj, start="A1", need=None):
    seen = {start}
    st = [start]
    while st:
        n = st.pop()
        for m in adj.get(n, ()):
            if m not in seen:
                seen.add(m)
                st.append(m)
    need = need if need is not None else [k for k in adj]
    return all(any(s == n or s.startswith(n + ":") for s in seen) for n in need)


# ------------------------------------------------------------------------------------------------ choose + apply
def choose_and_apply(W, cfg, rng, ips, sources):
    g = rng("discrepancy")
    C = cfg["discrepancies"]
    lay = W.lay
    out = []

    def add(kind, refs, where, text, prior_effect):
        out.append({"id": len(out) + 1, "kind": kind, "refs": refs, "where": [float(v) for v in where[:2]],
                    "text": text, "prior_effect": prior_effect})

    an = lay.rooms[lay.params["annex"]]
    add("room_missing", [an.id] + [i["tag"] for i in ips if i["room"] == an.id],
        [0.5 * (an.rect[0] + an.rect[2]), 0.5 * (an.rect[1] + an.rect[3])],
        "the annex %s (%s) is not on the drawing" % (an.number, an.name), "room, walls and door not drawn")
    und = [s for s in sources if not s["documented"]]
    for s in und:
        add("zone_outdated", [s["id"]], s["pos"], "unlisted source (%s): the zoning is out of date there" % s["text"],
            "zoning from documented sources only")
    n_total = int(g.integers(C["count"][0], C["count"][1] + 1))
    lo_total = int(C["count"][0])
    fns = {"door_blocked": _door_blocked, "wall_added": _wall_added, "opening_added": _opening_added,
           "ip_moved": _ip_moved, "ip_undocumented": _ip_undocumented, "sheet_missing": _sheet_missing}
    ip_rooms = sorted({i["room"] for i in ips})
    w = dict(C["weights"])
    need = list(C["require_one_of"])
    first = need[int(g.integers(len(need)))]
    order = [first] + [k for k in need if k != first]
    done = False
    for k in order:                                   # at least one of ip_moved / ip_undocumented
        if fns[k](W, cfg, g, ips, ip_rooms, add):
            done = True
            w.pop(k, None)
            break
    del done
    # then draw kinds by weight until the count is reached (a kind with no candidate is dropped and redrawn)
    while len(out) < max(n_total, lo_total) and w:
        keys = sorted(w.keys())
        p = np.array([w[k] for k in keys], float)
        k = keys[int(g.choice(len(keys), p=p / p.sum()))]
        w.pop(k)
        fns[k](W, cfg, g, ips, ip_rooms, add)
    return out


def _door_blocked(W, cfg, g, ips, ip_rooms, add):
    lay = W.lay
    cands = []
    for d in lay.doors:
        if d.kind not in ("door", "double") or not d.on_drawing or d.state == "closed":
            continue
        if any(lay.rooms.get(r) is not None and lay.rooms[r].type in ("airlock", "annex") for r in d.rooms):
            continue
        adj = room_graph(W, closed=(d.id,))
        if connected_all(adj, need=[r for r in lay.rooms if lay.rooms[r].type != "corridor"]):
            cands.append(d)
    if not cands:
        return False
    # prefer doors that sit on a route to an IP room
    cands.sort(key=lambda d: (0 if any(r in ip_rooms for r in d.rooms) else 1, d.id))
    top = [d for d in cands if any(r in ip_rooms for r in d.rooms)] or cands
    d = top[int(g.integers(len(top)))]
    if g.random() < 0.5:
        d.state = "closed"
        rebuild_leaves(W, d)
        how = "closed and locked"
    else:
        # stored material stacked in the doorway on the room side
        d.blocked_by = "stored material"
        x, y = d.xy()
        w = lay.wall(d.wall)
        s = 1 if d.swing_into == w.rooms[1] else -1
        face = (w.t_pos if s > 0 else w.t_neg) + 0.05
        body = "block/%s" % d.id
        if d.state == "ajar":
            d.state = "open"
        rebuild_leaves(W, d)
        for k in range(2):
            off = face + 0.55 + k * 1.05
            if w.axis == "y":
                c = (x + uniform(g, -0.1, 0.1), w.coord + s * off)
                W.P.box("%s/pal%d" % (body, k), (c[0], c[1], 0.07), (0.6, 0.5, 0.07), mat="pallet_wood", body=body,
                        owner=d.id, radio=False)
                W.P.box("%s/load%d" % (body, k), (c[0], c[1], 0.14 + 0.55), (0.55, 0.45, 0.55), mat="carton",
                        body=body, owner=d.id)
            else:
                c = (w.coord + s * off, y + uniform(g, -0.1, 0.1))
                W.P.box("%s/pal%d" % (body, k), (c[0], c[1], 0.07), (0.5, 0.6, 0.07), mat="pallet_wood", body=body,
                        owner=d.id)
                W.P.box("%s/load%d" % (body, k), (c[0], c[1], 0.14 + 0.55), (0.45, 0.55, 0.55), mat="carton",
                        body=body, owner=d.id)
        # the leaf must not swing through the material: keep it closed
        d.state = "closed"
        rebuild_leaves(W, d)
        how = "blocked by stored material (pallets stacked behind it)"
    x, y = d.xy()
    add("door_blocked", [d.id], (x, y), "door %s between %s and %s is %s" % (d.id, d.rooms[0], d.rooms[1], how),
        "drawn as a passage")
    return True


def _wall_added(W, cfg, g, ips, ip_rooms, add, no_chase=False, x_min=None):
    """A temporary shielding partition across a corridor leg, where every room stays reachable (no_chase: without
    passing through the pipe chase, i.e. without crouching; the ladder needs that for its B slots)."""
    lay = W.lay
    chase_mouths = tuple(d.id for d in lay.doors if d.kind == "chase_mouth") if no_chase else ()
    base_rooms = None
    if no_chase:
        # the rooms reachable without the chase before the partition must stay so
        adj0 = room_graph(W, closed=chase_mouths)
        seen = {"A1"}
        st = ["A1"]
        while st:
            n = st.pop()
            for m in adj0.get(n, ()):
                if m not in seen:
                    seen.add(m)
                    st.append(m)
        base_rooms = {s.split(":")[0] for s in seen}
    for _ in range(40):
        leg = "C3" if g.random() < 0.6 else "C1"
        r = lay.rooms[leg]
        x0, y0, x1, y1 = r.inner
        lo_x = x0 + 4.0 if (x_min is None or leg != "C1") else max(x0 + 4.0, x_min)
        if lo_x >= x1 - 3.0:
            continue
        x = round(uniform(g, lo_x, x1 - 3.0) / 0.05) * 0.05
        # not in front of a door
        if any(abs(x - d.xy()[0]) < d.width / 2 + 0.5 for d in lay.doors if leg in d.rooms):
            continue
        if any(abs(x - f["coord"]) < 1.2 for f in getattr(W, "fire_doors", []) if leg in f["rooms"]):
            continue
        if any(e.room == leg and min(p[0] for p in e.footprint) - 0.6 < x < max(p[0] for p in e.footprint) + 0.6
               for e in W.elements.values()):
            continue
        adj = room_graph(W, split=(leg, "x", x), closed=chase_mouths)
        need = [k for k in lay.rooms if lay.rooms[k].type not in ("corridor",) and
                (base_rooms is None or k in base_rooms)]
        # the chase counts (a crouching robot passes it); require every room reachable
        if not connected_all(adj, need=need):
            continue
        body = "partition/%s" % leg
        th = 0.4
        W.P.box_aabb("%s/blocks" % body, (x - th / 2, r.rect[1], 0.0), (x + th / 2, r.rect[3], 1.8),
                     mat="block_wall", body=body, owner=leg, radio=True)
        W.P.box_aabb("%s/cap" % body, (x - th / 2 - 0.02, r.rect[1], 1.8), (x + th / 2 + 0.02, r.rect[3], 1.86),
                     mat="steel_dark", body=body, owner=leg, group=1)
        mat = W.sign("partition_%s" % leg, "TEMPORARY SHIELDING - NO ENTRY", "room")
        W.P.box("%s/sign" % body, (x - th / 2 - 0.008, 0.5 * (y0 + y1), 1.4), (0.006, 0.3, 0.075), yaw=math.pi,
                mat=mat, body=body, owner=leg, group=1)
        W.partitions = getattr(W, "partitions", []) + [{"leg": leg, "x": x, "thickness": th, "height": 1.8}]
        add("wall_added", [leg], (x, 0.5 * (y0 + y1)), "a temporary shielding wall of blocks across corridor %s" %
            r.number, "not drawn")
        return True
    return False


def _opening_added(W, cfg, g, ips, ip_rooms, add):
    """A removed wall panel between two neighbouring rooms (or a room and the corridor) without a door."""
    lay = W.lay
    cands = []
    for w in lay.walls:
        a, b = w.rooms
        ra, rb = lay.rooms.get(a), lay.rooms.get(b)
        if ra is None or rb is None:
            continue
        if {ra.type, rb.type} & {"airlock", "annex", "waste_store", "pipe_chase"}:
            continue
        if w.kind != "ordinary" or w.hi - w.lo < 2.6:
            continue
        if any(lay.door(o).kind != "chase_mouth" for o in w.openings):
            continue
        cands.append(w)
    cands.sort(key=lambda w: (0 if any(r in ip_rooms for r in w.rooms) else 1, w.id))
    if not cands:
        return False
    for _ in range(20):
        w = cands[int(g.integers(min(len(cands), 6)))]
        width = round(uniform(g, 1.0, 1.3) / 0.05) * 0.05
        lo_, hi_ = w.lo + 0.9 + width / 2, w.hi - 0.9 - width / 2
        if hi_ < lo_:
            continue
        c = round(uniform(g, lo_, hi_) / 0.05) * 0.05
        # keep the furniture out of the way: reject if a collision prim sits in the opening's approach
        x, y = (w.coord, c) if w.axis == "x" else (c, w.coord)
        clash = False
        for p in W.P.prims:
            if p.group not in (0, 3) or p.body.startswith("wall/"):
                continue
            if abs(p.pos[0] - x) < width / 2 + 0.9 and abs(p.pos[1] - y) < width / 2 + 0.9 and p.zrange()[0] < 1.0:
                clash = True
                break
        if clash:
            continue
        d = add_door(lay, w, c, width, kind="opening", g=g)
        d.on_drawing = False
        d.state = "open"
        _cut_opening(W, w, d)
        # rubble at the foot of the opening: a debris field (needs_high_step) half of the time
        if g.random() < 0.5:
            h = sample_size(cfg, g, "debris", "needs_high_step")
            body = "elem/debris_%s" % d.id
            prims = []
            fp = []
            for k in range(7):
                along = c + uniform(g, -width / 2 + 0.08, width / 2 - 0.08)
                across = uniform(g, -0.35, 0.35)
                px, py = (w.coord + across, along) if w.axis == "x" else (along, w.coord + across)
                prims.append(W.P.box("%s/piece%d" % (body, k), (px, py, h / 2), (uniform(g, 0.04, 0.09),
                                     uniform(g, 0.04, 0.09), h / 2), yaw=uniform(g, 0, 3.1), mat="block_wall",
                                     body=body, owner=d.id))
            if w.axis == "x":
                fp = [[w.coord - 0.45, c - width / 2], [w.coord + 0.45, c - width / 2], [w.coord + 0.45, c + width / 2],
                      [w.coord - 0.45, c + width / 2]]
            else:
                fp = [[c - width / 2, w.coord - 0.45], [c + width / 2, w.coord - 0.45], [c + width / 2, w.coord + 0.45],
                      [c - width / 2, w.coord + 0.45]]
            e = W.element("debris", "needs_high_step", {"height": h}, w.rooms[1], fp, zone=True,
                          note="rubble from the removed panel %s" % d.id)
            W.tag(e, prims)
            d.sill = e.id
        add("opening_added", [d.id, w.rooms[0], w.rooms[1]], (x, y),
            "a wall panel between %s and %s has been removed (opening %.1f m wide)" %
            (lay.rooms[w.rooms[0]].number, lay.rooms[w.rooms[1]].number, width), "not drawn")
        return True
    return False


def _cut_opening(W, w, d):
    """Remove the wall prims of w overlapping the opening and rebuild them around it."""
    from .structure import WALL_MAT, _span_world
    P = W.P
    body = "wall/%s" % w.id
    a, b = d.center - d.width / 2, d.center + d.width / 2
    H = float(W.cfg["building"]["wall_height"])
    new = []
    for p in [p for p in P.prims if p.body == body]:
        R = p.R()
        if p.type != "box":
            continue
        # extent along the wall
        if w.axis == "x":
            lo, hi = p.pos[1] - p.size[1], p.pos[1] + p.size[1]
        else:
            lo, hi = p.pos[0] - p.size[0], p.pos[0] + p.size[0]
        if hi <= a or lo >= b or "skirt" in p.name:
            if "skirt" in p.name and not (hi <= a or lo >= b):
                P.prims.remove(p)
            continue
        z0, z1 = p.zrange()
        P.prims.remove(p)
        for k, (u0, u1) in enumerate(((lo, a), (b, hi))):
            if u1 - u0 > 1e-3:
                q0, q1 = _span_world(w, u0, u1, z0, z1)
                new.append(P.box_aabb("%s/cut%s_%d" % (body, d.id, k), q0, q1, mat=p.mat, body=body, owner=w.id,
                                      radio=True))
        del R
    q0, q1 = _span_world(w, a, b, 2.0, H)
    P.box_aabb("%s/cut%s_lintel" % (body, d.id), q0, q1, mat=WALL_MAT.get(w.kind, "wall_paint"), body=body,
               owner=w.id, radio=True)
    return new


def _ip_moved(W, cfg, g, ips, ip_rooms, add):
    cands = [i for i in ips if i["type"] in ("gauge", "valve_lever", "valve_handwheel", "lamps", "level") and
             i["reason"] not in ("room_missing",) and "documented" not in i]
    if not cands:
        return False
    cands.sort(key=lambda i: i["tag"])
    ip = cands[int(g.integers(len(cands)))]
    used = {i["slot"]["id"] for i in ips}
    alts = [s for s in W.slots if s["type"] == ip["type"] and s["id"] not in used and
            (s["room"] != ip["room"] or abs(s["yaw"] - ip["slot"]["yaw"]) > 1.0 or
             math.hypot(s["pos"][0] - ip["slot"]["pos"][0], s["pos"][1] - ip["slot"]["pos"][1]) > 3.0)
            and W.lay.rooms[s["room"]].on_drawing]
    if not alts:
        return False
    alts.sort(key=lambda s: s["id"])
    s = alts[int(g.integers(len(alts)))]
    ip["documented"] = {"slot": s["id"], "room": s["room"], "pos": s["pos"], "yaw": s["yaw"], "mount": s["mount"]}
    ip["discrepancy"] = "ip_moved"
    add("ip_moved", [ip["tag"]], ip["slot"]["pos"], "%s is not where the work order says (%s, room %s); it is in "
        "room %s" % (ip["tag"], s["mount"], W.lay.rooms[s["room"]].number, W.lay.rooms[ip["room"]].number),
        "documented location wrong")
    return True


def _ip_undocumented(W, cfg, g, ips, ip_rooms, add):
    cands = [i for i in ips if i["type"] not in ("hotspot",) and "documented" not in i and
             i["reason"] not in ("room_missing",)]
    if not cands:
        return False
    cands.sort(key=lambda i: i["tag"])
    ip = cands[int(g.integers(len(cands)))]
    ip["documented"] = None
    ip["discrepancy"] = "ip_undocumented"
    add("ip_undocumented", [ip["tag"]], ip["slot"]["pos"], "the work order gives no location for %s" % ip["tag"],
        "location not documented")
    return True


def _sheet_missing(W, cfg, g, ips, ip_rooms, add):
    lay = W.lay
    bx0, by0, bx1, by1 = lay.building_rect
    nx, ny = cfg["building"]["sheets"]
    xm, ym = 0.5 * (bx0 + bx1), 0.5 * (by0 + by1)
    sheets = []
    n = 0
    for j in range(ny):
        for i in range(nx):
            n += 1
            r = [bx0 + i * (bx1 - bx0) / nx, by0 + j * (by1 - by0) / ny, bx0 + (i + 1) * (bx1 - bx0) / nx,
                 by0 + (j + 1) * (by1 - by0) / ny]
            sheets.append((n, r))
    cands = [(n, r) for n, r in sheets if not (r[0] - 4.0 <= 0.0 <= r[2] and r[1] <= 0.0 <= r[3])]
    if not cands:
        return False
    n, r = cands[int(g.integers(len(cands)))]
    W.missing_sheet = {"sheet": n, "rect": r, "of": len(sheets)}
    del xm, ym
    add("sheet_missing", ["sheet %d" % n], (0.5 * (r[0] + r[2]), 0.5 * (r[1] + r[3])),
        "sheet %d of %d of the drawing is missing" % (n, len(sheets)), "blank area on the drawing")
    return True
