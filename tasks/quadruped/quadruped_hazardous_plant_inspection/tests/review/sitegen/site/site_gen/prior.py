"""prior.json: what the robot is given. The old drawing as vectors (rooms, walls, doors, grid lines, grid markers'
documented positions, the zoning of the last survey, drawn equipment, the link infrastructure as drawn), the work
order, the budgets and the datum. Derived from the original design (before the changes reality went through),
degraded by the discrepancies; nothing that only the ground truth knows."""

from __future__ import annotations

import math

from .ips import REPORT
from .radiation import zone_class
from .util import rect_center

DRAWN_OPEN = "open as drawn"
DRAWN_SHUT = "exterior, kept shut"
GT_ONLY_KEYS = ("value", "truth", "readable", "readable_region", "observation_pose", "class", "state", "blocked_by",
                "strength", "hot_index", "temp_C", "level_pct", "opening_pct", "angle_deg", "states", "present",
                "on_drawing", "discrepancy", "documented_slot", "elements", "route", "leaf_angle_deg", "gap",
                "sources", "geoms", "face_geoms")


def _in_rect(x, y, r):
    return r is not None and r[0] <= x <= r[2] and r[1] <= y <= r[3]


def _door_drawn(v2m, lay, d):
    """Version 2: a door is drawn when its wall is drawn where the door is (one of its two sides is a drawn room or
    a drawn corridor part)."""
    w = lay.wall(d.wall)
    for k, rid in enumerate(w.rooms):
        off = -(w.t_neg + 0.05) if k == 0 else (w.t_pos + 0.05)
        x, y = (w.coord + off, d.center) if w.axis == "x" else (d.center, w.coord + off)
        if rid in lay.rooms and not lay.rooms[rid].on_drawing:
            continue
        if v2m.side_drawn(rid, x, y):
            return True
    return False


def merge_hidden_joints(walls, rooms, tol=1e-6):
    """Version 2: a drawn wall shared with missing rooms would be drawn in pieces, one per missing
    room, and the pieces' ends would mark where the missing sheet's partitions meet it (with the outline, the missing
    rooms' widths). Collinear drawn pieces of one kind and thickness that meet at a point which is no corner of a
    drawn room (or drawn corridor part) are drawn as one wall."""
    corners = set()
    for r in rooms:
        x0, y0, x1, y1 = [float(v) for v in r["rect"]]
        for p in ((x0, y0), (x0, y1), (x1, y0), (x1, y1)):
            corners.add((round(p[0], 4), round(p[1], 4)))

    def key(w):
        (ax, ay), (bx, by) = w["a"], w["b"]
        vert = abs(ax - bx) < tol
        return (vert, round(ax if vert else ay, 6), round(float(w["thickness"]), 6), w["kind"],
                round(float(w["offset"]), 6))

    groups = {}
    for i, w in enumerate(walls):
        groups.setdefault(key(w), []).append(i)
    out = []
    for kk, idx in groups.items():
        vert = kk[0]
        segs = sorted(((min(walls[i]["a"][1], walls[i]["b"][1]), max(walls[i]["a"][1], walls[i]["b"][1])) if vert
                       else (min(walls[i]["a"][0], walls[i]["b"][0]), max(walls[i]["a"][0], walls[i]["b"][0]))
                       for i in idx))
        merged = [list(segs[0])]
        for lo, hi in segs[1:]:
            j = merged[-1][1]
            pt = (round(kk[1], 4), round(j, 4)) if vert else (round(j, 4), round(kk[1], 4))
            if abs(lo - j) < tol and pt not in corners:
                merged[-1][1] = max(j, hi)
            else:
                merged.append([lo, hi])
        w0 = walls[idx[0]]
        for lo, hi in merged:
            a, b = ([kk[1], lo], [kk[1], hi]) if vert else ([lo, kk[1]], [hi, kk[1]])
            out.append(dict(w0, a=a, b=b))
    # the drawing's order: as the walls came (first piece of each merged run)
    order = {}
    for i, w in enumerate(walls):
        order.setdefault(key(w), i)
    out.sort(key=lambda w: (order[key(w)], w["a"][1] if abs(w["a"][0] - w["b"][0]) < tol else w["a"][0]))
    return out


def _clip_seg(a, b, r):
    """Parts of the axis-aligned segment a-b outside rect r."""
    if r is None:
        return [(a, b)]
    (x0, y0), (x1, y1) = a, b
    if abs(y0 - y1) < 1e-9:
        if not (r[1] <= y0 <= r[3]):
            return [(a, b)]
        lo, hi = min(x0, x1), max(x0, x1)
        out = []
        if lo < r[0]:
            out.append(((lo, y0), (min(hi, r[0]), y0)))
        if hi > r[2]:
            out.append(((max(lo, r[2]), y0), (hi, y0)))
        return out
    if not (r[0] <= x0 <= r[2]):
        return [(a, b)]
    lo, hi = min(y0, y1), max(y0, y1)
    out = []
    if lo < r[1]:
        out.append(((x0, lo), (x0, min(hi, r[1]))))
    if hi > r[3]:
        out.append(((x0, max(lo, r[3])), (x0, hi)))
    return out


def build_prior(W, cfg, rng, ips, sources, discs, budgets, markers, zoning):
    g = rng("prior")
    lay = W.lay
    D = cfg["drawing"]
    miss = getattr(W, "missing_sheet", None)
    mrect = miss["rect"] if miss else None
    v2m = getattr(W, "missing", None)           # version 2's missing sheet (site_gen.sheet.Missing): whole rooms
    if v2m is not None:
        miss = {"sheet": v2m.sheet, "rect": v2m.rect, "of": v2m.of}
        mrect = None                             # nothing is clipped by the rectangle
    k = int(g.integers(len(D["revisions"])))
    num = "%s%02d" % (D["number_prefix"], int(g.integers(1, 99)))
    nsx, nsy = cfg["building"]["sheets"]
    bx0, by0, bx1, by1 = lay.building_rect
    sheets = []
    n = 0
    for j in range(nsy):
        for i in range(nsx):
            n += 1
            r = [bx0 + i * (bx1 - bx0) / nsx, by0 + j * (by1 - by0) / nsy, bx0 + (i + 1) * (bx1 - bx0) / nsx,
                 by0 + (j + 1) * (by1 - by0) / nsy]
            sheets.append({"sheet": n, "rect": r, "missing": bool(miss and miss["sheet"] == n)})
    drawing = {"number": num, "title": D["title"], "revision": D["revisions"][k], "revision_date":
               D["revision_dates"][k], "scale": D["scale_text"], "sheets": sheets,
               "notes": ["DIMENSIONS IN METRES FROM THE AIRLOCK DATUM",
                         "MODIFICATIONS AFTER THE REVISION DATE ARE NOT SHOWN" if str(cfg.get("layout", "")).startswith(
                             "ladder") else "AS-BUILT, NOT RE-SURVEYED"]}
    if miss:
        drawing["notes"].append("SHEET %d OF %d MISSING FROM THE ARCHIVE COPY" % (miss["sheet"], miss["of"]))
    # rooms
    rooms = []
    for r in sorted(lay.rooms.values(), key=lambda r: r.number):
        if not r.on_drawing:
            continue
        cx, cy = rect_center(r.rect)
        if _in_rect(cx, cy, mrect):
            continue
        ent = {"number": r.number, "name": r.name, "type": r.type if r.type != "decor" else r.kind, "rect": r.rect}
        if v2m is not None:
            # whole rooms: a room of the missing sheet is not listed; a corridor leg it cuts is listed with its drawn
            # part and `continues`
            rr = v2m.drawn_rect(r.id, r.rect)
            if rr is None:
                continue
            if v2m.cut_of(r.id) is not None:
                ent["rect"] = [round(float(v), 4) for v in rr]
                ent["continues"] = True
        rooms.append(ent)
    drawn_rooms = {r["number"] for r in rooms}
    # walls of the original drawing: those of the drawn rooms alone (no partition, nothing of the annex: the facade
    # it stands against is one outer wall like every other), clipped by the missing sheet
    # (version 2: a wall is drawn where it bounds a drawn room or a drawn corridor part, in full)
    from .layout import drawing_walls
    walls = []
    for w in drawing_walls(lay, cfg):
        a = (w.coord, w.lo) if w.axis == "x" else (w.lo, w.coord)
        b = (w.coord, w.hi) if w.axis == "x" else (w.hi, w.coord)
        kind = "shield" if w.kind == "shield" else ("outer" if w.kind == "outer" or "OUT" in w.rooms else "wall")
        if v2m is not None:
            from .sheet import wall_pieces
            segs = []
            for lo, hi in wall_pieces(v2m, w.axis, w.coord, w.lo, w.hi, w.rooms, w.t_neg, w.t_pos):
                segs.append(((w.coord, lo), (w.coord, hi)) if w.axis == "x" else ((lo, w.coord), (hi, w.coord)))
        else:
            segs = _clip_seg(a, b, mrect)
        for p, q in segs:
            walls.append({"a": list(p), "b": list(q), "thickness": w.thickness(), "kind": kind,
                          "offset": 0.5 * (w.t_pos - w.t_neg)})
    if v2m is not None:
        walls = merge_hidden_joints(walls, rooms)
    doors = []
    for d in lay.doors:
        if not d.on_drawing or d.kind == "opening":
            continue
        x, y = d.xy()
        if _in_rect(x, y, mrect):
            continue
        if v2m is not None and not _door_drawn(v2m, lay, d):
            continue
        w = lay.wall(d.wall)
        rn = [lay.rooms[r].number if r in lay.rooms else "OUT" for r in d.rooms]
        kind = {"chase_mouth": "hatch", "airlock_outer": "door", "stair": "door", "double": "double_door"}.get(
            d.kind, "door")
        # the door's state as the drawing gives it (the two exterior doors were listed like passages): the
        # airlock's outer door and the stair door are kept shut; every other drawn door is a passage on the drawing
        # (the drawing predates every later change, so a door sealed since is drawn like the others)
        ent = {"pos": [x, y], "axis": d.axis, "width": d.width, "rooms": rn, "kind": kind,
               "hinge": d.hinge, "swing_into": lay.rooms[d.swing_into].number if d.swing_into in lay.rooms else "",
               "drawn_state": DRAWN_SHUT if d.kind in ("airlock_outer", "stair") else DRAWN_OPEN}
        if d.kind == "chase_mouth":
            ent["note"] = "low access hatch, clear height %.2f m" % float(lay.params["chase_clearance"] or 0.35)
        if d.kind == "stair":
            ent["note"] = "STAIR 2 (no access)"
        if d.kind == "airlock_outer":
            ent["note"] = "airlock outer door"
        doors.append(ent)
    for f in getattr(W, "fire_doors", []):
        if _in_rect(f["pos"][0], f["pos"][1], mrect):
            continue
        if v2m is not None and v2m.in_cut(f["pos"][0], f["pos"][1]):
            continue                  # a cut-off corridor part takes its fire doors with it
        doors.append({"pos": f["pos"], "axis": f["axis"], "width": f["width"], "rooms": [lay.rooms[f["rooms"][0]].number] * 2,
                      "kind": "fire_door", "hinge": 0, "swing_into": "", "drawn_state": DRAWN_OPEN})
    # equipment as drawn
    equipment = []
    for e in W.equipment:
        if not e["on_drawing"]:
            continue
        r = lay.rooms.get(e["room"])
        if r is not None and not r.on_drawing:
            continue
        if e["shape"] == "circle":
            cx, cy = e["center"]
        else:
            xs = [p[0] for p in e["footprint"]]
            ys = [p[1] for p in e["footprint"]]
            cx, cy = 0.5 * (min(xs) + max(xs)), 0.5 * (min(ys) + max(ys))
        if _in_rect(cx, cy, mrect):
            continue
        if v2m is not None and (v2m.room_missing(e["room"]) or v2m.in_part(cx, cy)):
            continue                  # a missing room takes its equipment with it
        ent = {"label": e["label"], "kind": e["kind"], "room": r.number if r else ""}
        if e["shape"] == "circle":
            ent.update({"shape": "circle", "center": e["center"], "radius": e["radius"]})
        else:
            ent.update({"shape": "rect", "outline": e["footprint"]})
        equipment.append(ent)
    # grid markers (documented positions)
    gm = []
    for m in markers:
        if not m["documented"]:
            continue
        gm.append({"id": m["id"], "marker_id": m["marker_id"], "pos": m["pos"][:2], "height": m["pos"][2],
                   "facing_deg": round(math.degrees(m["yaw"]), 1), "grid": m["grid"], "on": m["on"]})
    # zoning (documented sources only), not inside the missing sheet
    zones = [z for z in zoning if not _in_rect(z["center"][0], z["center"][1], mrect)]
    if v2m is not None:
        zones = [z for z in zones if z.get("drawn", True)]      # no zone of a missing room or cut-off part
    zones = [{"zone_class": z["class"], "polygon": z["polygon"], "label": z["label"]} for z in zones]
    # link infrastructure as drawn: the base station and both repeaters "in service"
    link = {"base_station": {"pos": W.base_station["antenna"][:2], "range_m": float(cfg["link"]["base_range"])},
            "repeaters": [{"id": r["id"], "pos": r["pos"][:2], "status": "in service",
                           "range_m": float(cfg["link"]["repeater_range"])} for r in W.repeaters],
            "rule": "reports are delivered when the robot has line of sight to the base station within its range, "
                    "or to a working repeater; elsewhere they are queued"}
    # documented sources: the waste store inventory (drum count per room, not positions or activities)
    inv = {}
    for s in sources:
        if s["documented"]:
            inv.setdefault(lay.rooms[s["room"]].number, 0)
            inv[lay.rooms[s["room"]].number] += 1
    # the work order (version 2: whole recorded locations on the missing sheet too, facing to 5 degrees)
    items = []
    hs_rooms = None
    fstep = float(((cfg.get("ladder") or {}).get("v2_params") or {}).get("facing_step_deg", 45.0)) \
        if (cfg.get("ladder") or {}).get("v2") else 45.0
    slots_by_id = {sl["id"]: sl for sl in W.slots}
    for ip in sorted(ips, key=lambda i: i["tag"]):
        e = {"tag": ip["tag"], "type": ip["type"], "report": REPORT[ip["type"]], "weight": ip["weight"]}
        s = ip["slot"]
        prm = ip["params"]
        # the service text of a moved IP is the documented slot's (the true slot's service, e.g. "seal water level"
        # beside a documented "T-163A level gauge", told the reader where the instrument really is). Not for a ladder
        # item (2026-11-14): the round sheet describes the listed instrument as it is,
        # and its documented slot is another instance (C-moved: the decoy now at the drawn place; C-relocated: a
        # dry-built pose whose slot id a later slot reuses, the C-annex instrument's or the decoy's)
        doc0 = ip.get("documented", "same")
        own = bool(ip.get("ladder_slot"))
        service = s["service"]
        if isinstance(doc0, dict) and doc0.get("slot") in slots_by_id and not own:
            service = slots_by_id[doc0["slot"]]["service"]
        if ip["type"] == "gauge":
            e["scale"] = {"range": [prm["lo"], prm["hi"]], "unit": prm["unit"], "case_mm": prm["case_mm"]}
            e["service"] = service
        elif ip["type"] == "lamps":
            e["service"] = service
            e["n_lamps"] = len(prm["colors"])
            e["colors"] = prm["colors"]
        elif ip["type"] == "thermal":
            e["service"] = service
            e["n_units"] = prm["n_units"]
            labels = [u["label"] for u in ip["truth"]["units"]]
            if isinstance(doc0, dict) and not own and not any(ip["tag"] in lb for lb in labels):
                # a moved IP lists the documented slot's units (the true group's labels carry the true equipment
                # tag: a review's last point); generic labels when the documented slot has none. (The
                # ladder's stations are labelled by the instrument's own tag, which is numbered from the drawn room.)
                # Not for a ladder item: its units are those of the tagged component that carries its
                # tag plate, wherever it stands (practice t2's TS-1414 on SWG-141 cubicle 1 was listed with the
                # decoy's "TS-1411 heater terminal box")
                ds = slots_by_id.get(doc0.get("slot")) or {}
                dl = [u.get("label") for u in ds.get("units", [])]
                labels = dl[:len(labels)] if len(dl) >= len(labels) and all(dl) else \
                    ["unit %d" % (k + 1) for k in range(len(labels))]
            e["units"] = labels
        else:
            e["service"] = service
        # location
        room = lay.rooms[ip["room"]]
        doc = ip.get("documented", "same")
        if ip["type"] == "hotspot":
            # the work order only says that one unlisted source exists somewhere in the building (the
            # earlier four 'survey rooms' always contained the true room). survey_rooms is kept for readers of the
            # old format and lists every drawn room
            area = sorted(r.number for r in lay.rooms.values() if r.on_drawing and r.type not in ("airlock",))
            e["location"] = {"documented": False, "survey_rooms": area,
                             "text": "an unlisted source exists somewhere in the building; report its position and "
                                     "peak dose rate"}
            hs_rooms = area
        elif doc is None or not room.on_drawing:
            e["location"] = {"documented": False, "text": "location not documented"}
        elif doc == "same":
            e["location"] = _loc(lay, s, room, mrect, fstep)
        else:
            e["location"] = _loc(lay, {"pos": doc["pos"], "yaw": doc["yaw"], "mount": doc["mount"]},
                                 lay.rooms[doc["room"]], mrect, fstep)
        if ip["type"] == "dose_point" and e["location"].get("documented"):
            e["dwell_s"] = float(cfg["ips"]["dwell_s"]["dose_point"])
        items.append(e)
    wo = {"number": "WO-%05d" % int(g.integers(10000, 99999)), "title": "Pre-decommissioning characterisation, "
          "auxiliary building ground floor", "items": items,
          "radiation_survey": {"report": "dose-rate readings along the route; any unlisted source",
                               "survey_rooms": hs_rooms or []},
          "return": "return to the deployment pad in the airlock and stop upright",
          "documented_sources": [{"room": k, "count": v, "text": "active waste drums (inventory)"}
                                 for k, v in sorted(inv.items())]}
    prior = {"schema": "site_gen/prior/0.1", "frame": {"origin": "deployment pad centre (airlock datum)",
             "x": "into the building", "y": "left", "z": "up", "units": "m, rad"},
             "datum": {"pos": [0.0, 0.0], "yaw": 0.0, "uncertainty": {"pos_m": 0.1, "yaw_deg": 3.0},
                       "pad_size_m": float(cfg["airlock"]["pad_size"])},
             "drawing": drawing, "building": {"outline": lay.building_rect,
                                              "airlock": lay.rooms["A1"].rect, "wall_height": float(
                                                  cfg["building"]["wall_height"])},
             "grid": lay.grid, "rooms": rooms, "walls": walls, "doors": doors, "equipment": equipment,
             "grid_markers": gm, "zoning": {"survey_date": cfg["radiation"]["survey_date"], "zones": zones,
                                            "classes": [[c, b] for c, b in cfg["radiation"]["zone_classes"]]},
             "link": link, "work_order": wo, "budgets": budgets}
    del drawn_rooms
    return prior


def _loc(lay, s, room, mrect, facing_step=45.0):
    """A documented location: the room, the instrument's position and height (the instrument's position as
    mission/evidence.py's face_frame names it: a rising-stem valve's is its pipe's centre line, the indicator plate
    0.42 m above it; the catalogue gives every type's offset to the face centre) and its facing, to the
    nearest `facing_step` degrees (version 1: 45; version 2: 5). Version 2 never clips by the missing sheet:
    the round sheet was copied from the instrument index while the drawing was whole."""
    x, y = s["pos"][0], s["pos"][1]
    ent = {"documented": True, "room": room.number, "room_name": room.name, "mount": s["mount"]}
    if _in_rect(x, y, mrect):
        ent["pos"] = None
        ent["text"] = "on the missing sheet: room %s only" % room.number
    else:
        ent["pos"] = [round(x, 1), round(y, 1)]
        ent["height"] = round(s["pos"][2], 1) if len(s["pos"]) > 2 else None
        st = float(facing_step)
        f = round(math.degrees(s["yaw"]) / st) * st
        if st != 45.0:
            f = ((f + 180.0) % 360.0) - 180.0
            f = 180.0 if abs(f + 180.0) < 1e-9 else f
            f = float(round(f, 1))
        ent["facing_deg"] = f
    return ent


def _survey_area(W, ip):
    lay = W.lay
    x, y = ip["slot"]["pos"][:2]
    r = lay.rooms[ip["room"]]
    near = []
    for o in lay.rooms.values():
        if not o.on_drawing or o.type in ("airlock",):
            continue
        cx, cy = rect_center(o.rect)
        d = math.hypot(cx - x, cy - y)
        near.append((d, o.number))
    near.sort()
    out = sorted({n for _, n in near[:4]} | ({r.number} if r.on_drawing else set()))
    return out
