"""Structure: walls split at their openings, lintels, door frames and leaves, sills (traversal elements), chase
hatches, floor finishes, ceilings / roof slabs, room signs, and the door-approach keep-out zones."""

from __future__ import annotations

import math

from .util import choice, uniform, weighted_key
from .world import World, ajar_angle_for_gap, restrict_class, sample_size

FLOOR_MAT = {"corridor": "floor_corridor", "airlock": "floor_airlock", "pump_room": "floor_epoxy_green",
             "valve_gallery": "floor_epoxy", "tank_room": "floor_epoxy", "electrical_room": "floor_epoxy",
             "waste_store": "floor_concrete", "pipe_chase": "floor_concrete", "annex": "floor_concrete",
             "change_room": "floor_lino", "sample_lab": "floor_lino", "office": "floor_lino",
             "store": "floor_concrete", "workshop": "floor_concrete", "hvac_room": "floor_epoxy",
             "decon_room": "floor_lino"}
WALL_MAT = {"ordinary": "wall_paint", "outer": "wall_paint", "shield": "shield_concrete",
            "partition": "block_wall"}


def _span_world(w, a, b, z0, z1):
    """World AABB of the part [a, b] (along the wall) x [z0, z1] of wall w."""
    if w.axis == "x":
        return (w.coord - w.t_neg, a, z0), (w.coord + w.t_pos, b, z1)
    return (a, w.coord - w.t_neg, z0), (b, w.coord + w.t_pos, z1)


def build_structure(W: World):
    cfg, lay, P = W.cfg, W.lay, W.P
    B = cfg["building"]
    H = float(B["wall_height"])
    hd = float(B["door_height"])
    g = W.rng("structure")
    doors_by_wall = {}
    for d in lay.doors:
        doors_by_wall.setdefault(d.wall, []).append(d)
    # the chase hatches: one clearance per chase (in the needs_crouch band), an element per mouth
    chase_c = None
    for d in lay.doors:
        if d.kind != "chase_mouth":
            continue
        if chase_c is None:
            chase_c = sample_size(cfg, g, "low_overhead", "needs_crouch")
        d.gap = chase_c
        w = lay.wall(d.wall)
        a, b = d.center - d.width / 2, d.center + d.width / 2
        if w.axis == "x":
            fp = [[w.coord - w.t_neg, a], [w.coord + w.t_pos, a], [w.coord + w.t_pos, b], [w.coord - w.t_neg, b]]
        else:
            fp = [[a, w.coord - w.t_neg], [b, w.coord - w.t_neg], [b, w.coord + w.t_pos], [a, w.coord + w.t_pos]]
        chase_room = d.rooms[0] if lay.rooms.get(d.rooms[0]) and lay.rooms[d.rooms[0]].type == "pipe_chase" \
            else d.rooms[1]
        e = W.element("low_overhead", "needs_crouch", {"clearance": chase_c, "door": d.id}, chase_room, fp,
                      note="pipe chase hatch")
        d.sill = e.id
    lay.params["chase_clearance"] = chase_c

    # ---------------------------------------------------------------- walls
    for w in lay.walls:
        ext = 0.175 if w.kind == "outer" else 0.125
        lo, hi = w.lo - ext, w.hi + ext
        ops = sorted(doors_by_wall.get(w.id, []), key=lambda d: d.center)
        mat = WALL_MAT[w.kind]
        body = "wall/%s" % w.id
        cur = lo
        k = 0
        for d in ops:
            a, b = d.center - d.width / 2, d.center + d.width / 2
            if a > cur + 1e-4:
                p0, p1 = _span_world(w, cur, a, 0.0, H)
                P.box_aabb("%s/s%d" % (body, k), p0, p1, mat=mat, body=body, owner=w.id, radio=True)
                k += 1
            top = hd
            if d.kind == "chase_mouth":
                top = d.gap  # hatch clearance, set by the chase rule below
            p0, p1 = _span_world(w, a, b, top, H)
            if d.kind == "chase_mouth":
                # hazard-striped hatch head (the low overhead element), then the wall above it
                p0b, p1b = _span_world(w, a - 0.02, b + 0.02, top, top + 0.16)
                hp = P.box_aabb("%s/hatch%d" % (body, k), p0b, p1b, mat="hazard_stripe", body="door/%s" % d.id,
                                owner=d.id, radio=True)
                e = W.elements[d.sill]
                W.tag(e, [hp])
                p0, p1 = _span_world(w, a - 0.02, b + 0.02, top + 0.16, H)
            P.box_aabb("%s/lintel%d" % (body, k), p0, p1, mat=mat, body=body, owner=w.id, radio=True)
            k += 1
            cur = b
        if hi > cur + 1e-4:
            p0, p1 = _span_world(w, cur, hi, 0.0, H)
            P.box_aabb("%s/s%d" % (body, k), p0, p1, mat=mat, body=body, owner=w.id, radio=True)
        # skirting band (decor): a darker strip on both faces of ordinary walls
        if w.kind in ("ordinary", "outer"):
            for side in (-1, 1):
                if (side < 0 and w.rooms[0] == "OUT") or (side > 0 and w.rooms[1] == "OUT"):
                    continue
                cur = lo
                pieces = []
                for d in ops:
                    a = d.center - d.width / 2
                    if a > cur:
                        pieces.append((cur, a))
                    cur = d.center + d.width / 2
                pieces.append((cur, hi))
                for j, (a, b) in enumerate(pieces):
                    if b - a < 0.05:
                        continue
                    t = (w.t_pos if side > 0 else w.t_neg) + 0.004
                    if w.axis == "x":
                        c = (w.coord + side * t, 0.5 * (a + b), 0.06)
                        P.box("%s/skirt%d_%d" % (body, side, j), c, (0.004, 0.5 * (b - a), 0.06), mat="skirting",
                              body=body, owner=w.id, group=1)
                    else:
                        c = (0.5 * (a + b), w.coord + side * t, 0.06)
                        P.box("%s/skirt%d_%d" % (body, side, j), c, (0.5 * (b - a), 0.004, 0.06), mat="skirting",
                              body=body, owner=w.id, group=1)

    # ---------------------------------------------------------------- doors, sills, frames, leaves
    D = cfg["doors"]
    rooms = lay.rooms
    gated = _choose_gated_room(W, g)
    W.lay.params["gated_room"] = gated
    for d in lay.doors:
        w = lay.wall(d.wall)
        _door_geometry(W, d, w, D, g, gated)

    # ---------------------------------------------------------------- floors, roofs, signs
    roof_t = float(B["roof_thickness"])
    for r in rooms.values():
        x0, y0, x1, y1 = r.rect
        key = r.kind if r.type == "decor" else r.type
        P.box_aabb("floor/%s" % r.id, (x0 - 0.01, y0 - 0.01, -0.01), (x1 + 0.01, y1 + 0.01, 0.0015),
                   mat=FLOOR_MAT.get(key, "floor_concrete"), body="room/%s" % r.id, owner=r.id, group=1)
        P.box_aabb("roof/%s" % r.id, (x0 - 0.2, y0 - 0.2, H), (x1 + 0.2, y1 + 0.2, H + roof_t), mat="roof",
                   body="room/%s" % r.id, owner=r.id, group=2)
        if r.type not in ("corridor",):
            _room_sign(W, r)
    _keepouts(W)


def _choose_gated_room(W: World, g):
    """One room whose every door has a needs_high_step sill (the generator's high-step guarantee; in the ladder the
    room of slot B-step, chosen before the structure is built)."""
    if W.ladder and W.lay.params.get("ladder_gated"):
        g.random()
        return W.lay.params["ladder_gated"]
    if W.ladder and "ladder_gated" in W.lay.params and W.lay.params["ladder_gated"] is None:
        # version 2, B-step in the room: no room gets high-step sills on every door
        g.random()
        return None
    cands = [r.id for r in W.lay.rooms.values() if r.type in ("pump_room", "electrical_room", "tank_room",
                                                              "waste_store", "valve_gallery")]
    cands.sort()
    if not cands:
        return None
    wts = [{"pump_room": 1.0, "electrical_room": 1.0, "tank_room": 0.6, "waste_store": 0.8,
            "valve_gallery": 0.5}[W.lay.rooms[c].type] for c in cands]
    return choice(g, cands, wts)


def _sill_class(W, d, D, g, gated):
    lay = W.lay
    if d.kind in ("stair", "airlock_outer", "chase_mouth", "opening"):
        return None
    if lay.rooms.get(d.rooms[0]) is not None and lay.rooms[d.rooms[0]].type == "airlock" or \
            lay.rooms.get(d.rooms[1]) is not None and lay.rooms[d.rooms[1]].type == "airlock":
        return "free"
    if gated and gated in d.rooms:
        return restrict_class(W.cfg, "sill", "needs_high_step")
    cls = restrict_class(W.cfg, "sill", weighted_key(g, D["room_sill_weights"]))
    if set(d.rooms) & set(lay.params.get("free_sill_rooms") or ()):
        cls = "free"          # version 2: the door of a room with an in-room strip or rack has a free sill
    return cls


def _door_geometry(W: World, d, w, D, g, gated):
    P = W.P
    cfg = W.cfg
    hd = float(cfg["building"]["door_height"])
    body = "door/%s" % d.id
    a, b = d.center - d.width / 2, d.center + d.width / 2
    t = w.t_neg + w.t_pos
    room_of = W.lay.rooms.get

    # chase mouths: a low hatch; the element is the hatch head (needs_crouch)
    if d.kind == "chase_mouth":
        return  # built by the chase rule (it needs the clearance first)

    # sill
    cls = _sill_class(W, d, D, g, gated)
    if cls is not None:
        h = sample_size(cfg, g, "sill", cls)
        sd = float(D["sill_depth"]) if cls != "free" else min(t, 0.22)
        if w.axis == "x":
            p0, p1 = (w.coord - sd / 2, a, 0.0), (w.coord + sd / 2, b, h)
        else:
            p0, p1 = (a, w.coord - sd / 2, 0.0), (b, w.coord + sd / 2, h)
        mat = "hazard_stripe" if cls != "free" else "steel_plate"
        sp = P.box_aabb("%s/sill" % body, p0, p1, mat=mat, body=body, owner=d.id, friction=0.8)
        fp = [[p0[0], p0[1]], [p1[0], p0[1]], [p1[0], p1[1]], [p0[0], p1[1]]]
        e = W.element("sill", cls, {"height": h, "depth": sd}, d.rooms[1] if d.swing_into == d.rooms[1]
                      else d.rooms[0], fp, note="door %s" % d.id)
        W.tag(e, [sp])
        d.sill = e.id

    # frame (decor): jamb strips and a head strip on both faces
    fw = float(D["frame_width"])
    for side in (-1, 1):
        off = (w.t_pos if side > 0 else w.t_neg) + 0.006
        for j, (c0, c1) in enumerate(((a - fw, a), (b, b + fw))):
            if w.axis == "x":
                P.box_aabb("%s/jamb%d%d" % (body, side, j), (w.coord + side * off - 0.006, c0, 0.0),
                           (w.coord + side * off + 0.006, c1, hd + fw), mat="door_frame", body=body, owner=d.id,
                           group=1)
            else:
                P.box_aabb("%s/jamb%d%d" % (body, side, j), (c0, w.coord + side * off - 0.006, 0.0),
                           (c1, w.coord + side * off + 0.006, hd + fw), mat="door_frame", body=body, owner=d.id,
                           group=1)
        if d.kind not in ("opening",):
            if w.axis == "x":
                P.box_aabb("%s/head%d" % (body, side), (w.coord + side * off - 0.006, a - fw, hd),
                           (w.coord + side * off + 0.006, b + fw, hd + fw), mat="door_frame", body=body,
                           owner=d.id, group=1)
            else:
                P.box_aabb("%s/head%d" % (body, side), (a - fw, w.coord + side * off - 0.006, hd),
                           (b + fw, w.coord + side * off + 0.006, hd + fw), mat="door_frame", body=body,
                           owner=d.id, group=1)

    if d.kind == "opening":
        return
    # state: ajar for some ordinary doors; never over a sill that needs a mode (an ajar leaf over a high-step sill
    # would make a tight high-step crossing that no rated mode passes)
    if d.state == "open" and d.kind == "door" and d.on_drawing and cls in (None, "free") and \
            g.random() < float(D["ajar_probability"]):
        r_into = room_of(d.swing_into)
        if r_into is not None and r_into.type not in ("airlock",) and len(r_into.doors) >= 1:
            d.state = "ajar"
    _leaves(W, d, w, D, g)


def _leaves(W: World, d, w, D, g):
    """Door leaves for the current state (called again when a discrepancy closes a door)."""
    P = W.P
    cfg = W.cfg
    hd = float(cfg["building"]["door_height"])
    body = "door/%s" % d.id
    for p in [p for p in P.prims if p.body == body and "/leaf" in p.name]:
        P.names.discard(p.name)
    P.prims = [p for p in P.prims if not (p.body == body and "/leaf" in p.name)]
    # a rebuild (the waste store's maze re-hinges its door, door_blocked closes a door) must not leave the
    # narrow-gap record of an earlier call behind: it would be a duplicate, or a 'tight' record on a closed door
    for k in [k for k, e in W.elements.items() if e.kind == "narrow_gap" and e.param.get("door") == d.id]:
        del W.elements[k]
    th = float(D["leaf_thickness"])
    a, b = d.center - d.width / 2, d.center + d.width / 2
    # side of the swing: +1 if the room the leaf swings into is on the +side of the wall line
    s = 1 if d.swing_into == w.rooms[1] else -1
    face = (w.t_pos if s > 0 else w.t_neg)
    leaves = []
    if d.kind == "double":
        leaves = [(a, +1, d.width / 2), (b, -1, d.width / 2)]   # (hinge position, closing direction, leaf length)
    else:
        if d.hinge > 0:
            leaves = [(b, -1, d.width)]
        else:
            leaves = [(a, +1, d.width)]
    if d.state == "open":
        ang = 92.0
    elif d.state == "ajar":
        gap = d.gap if d.gap else sample_size(cfg, g, "narrow_gap", "tight")
        d.gap = gap
        ang = ajar_angle_for_gap(d.width, th, 0.0, gap)
        if d.sill is None or True:
            pass
    else:
        ang = 0.0
    d.leaf_angle_deg = ang
    mat = "door_leaf_fire" if d.kind in ("fire", "stair") else ("door_leaf_heavy" if d.kind == "airlock_outer"
                                                              else "door_leaf")
    for j, (hpos, cdir, L) in enumerate(leaves):
        thr = math.radians(ang)
        # in wall-aligned coords: along = position along the wall, across = offset from the wall line
        dir_along = cdir * math.cos(thr)
        dir_across = s * math.sin(thr)
        base_across = s * (face + th / 2 + 0.002)
        c_along = hpos + dir_along * L / 2
        c_across = base_across + dir_across * L / 2
        yaw_wall = 0.0 if w.axis == "y" else math.pi / 2       # direction of 'along' in the world
        leaf_yaw = math.atan2(dir_across, dir_along)            # leaf direction in (along, across) coords
        if w.axis == "y":
            cx, cy = c_along, w.coord + c_across
            yaw = leaf_yaw
        else:
            # along = +y, across = +x  ->  world (x, y) = (across, along)
            cx, cy = w.coord + c_across, c_along
            yaw = math.atan2(math.cos(leaf_yaw), math.sin(leaf_yaw))
        del yaw_wall
        P.box("%s/leaf%d" % (body, j), (cx, cy, 0.012 + (hd - 0.03) / 2), (L / 2 - 0.006, th / 2, (hd - 0.03) / 2),
              yaw=yaw, mat=mat, body=body, owner=d.id, radio=(d.state == "closed"))
        # handle (decor)
        hx = cx + math.cos(yaw) * (L / 2 - 0.08)
        hy = cy + math.sin(yaw) * (L / 2 - 0.08)
        P.box("%s/leafhandle%d" % (body, j), (hx, hy, 1.05), (0.06, th / 2 + 0.025, 0.012), yaw=yaw,
              mat="steel", body=body, owner=d.id, group=1)
    if d.state == "ajar":
        # the narrow gap as a zone element (informational: the footprint geometry makes it tight)
        x, y = d.xy()
        fp = [[x - 0.6, y - 0.6], [x + 0.6, y - 0.6], [x + 0.6, y + 0.6], [x - 0.6, y + 0.6]]
        e = W.element("narrow_gap", "tight", {"width": d.gap, "door": d.id}, d.swing_into, fp,
                      note="door %s left ajar" % d.id)
        e.geoms = [p.name for p in P.prims if p.body == body and "/leaf" in p.name]
        d.blocked_by = None


def rebuild_leaves(W: World, d):
    _leaves(W, d, W.lay.wall(d.wall), W.cfg["doors"], W.rng("structure_leaves"))


def _room_sign(W: World, r):
    """Room number sign beside the main door on the corridor side (1.6 m), plus a low hazard sign where due."""
    lay = W.lay
    P = W.P
    if not r.doors:
        return
    if r.type == "annex" and (W.cfg.get("ladder") or {}).get("v2"):
        return                # version 2: a room never numbered or drawn has no plate
    d = lay.door(r.doors[0])
    if d.kind in ("airlock_outer", "stair"):
        return
    w = lay.wall(d.wall)
    other = d.rooms[0] if d.rooms[1] == r.id else d.rooms[1]
    if other == "OUT" and r.type != "annex":
        return
    s = 1 if other == w.rooms[1] else -1        # the sign goes on the face toward `other`
    face = (w.t_pos if s > 0 else w.t_neg) + 0.008
    along = d.center + (d.width / 2 + 0.28) * (1 if d.hinge < 0 else -1)
    text = "%s  %s" % (r.number, r.name) if r.on_drawing else "%s" % r.name
    if (W.cfg.get("ladder") or {}).get("v2"):
        # version 2: the room plate in the plant lettering, number and name
        mat = W.sign("room_%s" % r.id, text, "room_plate", (800, 200))
        W.signs[mat].update(number=r.number, name=r.name)
    else:
        mat = W.sign("room_%s" % r.id, text, "room")
    if w.axis == "y":
        c = (along, w.coord + s * face, 1.6)
        yaw = math.pi / 2 if s > 0 else -math.pi / 2
    else:
        c = (w.coord + s * face, along, 1.6)
        yaw = 0.0 if s > 0 else math.pi
    P.box("sign/%s" % r.id, c, (0.006, 0.2, 0.05), yaw=yaw, mat=mat, body="room/%s" % r.id, owner=r.id, group=1)
    if r.type in ("waste_store", "tank_room", "pump_room"):
        mat2 = W.sign("haz_%s" % r.id, "CONTROLLED AREA" if r.type == "waste_store" else "HEARING PROTECTION",
                      "trefoil" if r.type == "waste_store" else "notice", (256, 256))
        if w.axis == "y":
            c2 = (along, w.coord + s * face, 1.15)
        else:
            c2 = (w.coord + s * face, along, 1.15)
        P.box("sign/%s/haz" % r.id, c2, (0.006, 0.14, 0.14), yaw=yaw, mat=mat2, body="room/%s" % r.id,
              owner=r.id, group=1)


def _keepouts(W: World):
    """Door approach zones (world rects) on both sides of every passable door."""
    lay = W.lay
    for d in lay.doors:
        if d.kind in ("airlock_outer",):
            continue
        w = lay.wall(d.wall)
        for s, rid in ((-1, w.rooms[0]), (1, w.rooms[1])):
            if rid == "OUT" or rid not in lay.rooms:
                continue
            depth = max(1.4, d.width + 0.3) if d.swing_into == rid else 1.2
            face = w.t_pos if s > 0 else w.t_neg
            a, b = d.center - d.width / 2 - 0.35, d.center + d.width / 2 + 0.35
            if w.axis == "y":
                y0 = w.coord + s * face
                y1 = y0 + s * depth
                rect = [a, min(y0, y1), b, max(y0, y1)]
            else:
                x0 = w.coord + s * face
                x1 = x0 + s * depth
                rect = [min(x0, x1), a, max(x0, x1), b]
            W.keepouts.setdefault(rid, []).append(rect)
