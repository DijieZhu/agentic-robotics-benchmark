"""Instrument stations for the ladder: the mounting hardware that lets any of the eight instrument types stand at a
mount spot (a free wall span, a tank shell, a free-standing post), so that the type assigned to a slot does not have
to match whatever slot the room rule happened to make.

  dial gauge       impulse line from the ceiling, wall bracket, the gauge on its end
  level gauge      a small vertical pot against the wall with the tubular gauge on its nozzles
  lever / gate     a process line along the wall on brackets (elbows into the wall) with the valve on it
  lamp panel       the local panel on the wall with a conduit up to the cable tray
  temperature pt.  a hot component (heater terminal box) on the wall, its front plate is the measured face

`spots` are registered by the room rules (W.spot / W.spots) and by scan_wall_spots(); build() turns a chosen spot
into geometry and returns an instrument slot (the same dict as the room rules' W.slot records).
"""

from __future__ import annotations

import math

import numpy as np

from .furnish import SERVICE_TEXT, Ctx
from .util import uniform

WALL_N = {"front": (0, 1), "back": (0, -1), "left": (1, 0), "right": (-1, 0)}
WALL_T = {"front": (1, 0), "back": (1, 0), "left": (0, 1), "right": (0, 1)}


def wall_point(c: Ctx, wall, s, d):
    """Local (u, v) of a point at position s along `wall` and distance d from its face (into the room)."""
    if wall == "front":
        return s, d
    if wall == "back":
        return s, c.Lv - d
    if wall == "left":
        return d, s
    return c.Lu - d, s


def wall_len(c: Ctx, wall):
    return c.Lu if wall in ("front", "back") else c.Lv


# ============================================================================================ spot scanning
def scan_wall_spots(W, c: Ctx, zone="room", walls=("front", "back", "left", "right"), depth=0.6, min_len=0.9,
                    margin=0.45):
    """Register the free stretches of a room's walls (free of the room rule's objects and door approach zones)."""
    out = []
    for wall in walls:
        L = wall_len(c, wall)
        ss = np.arange(margin, L - margin + 1e-9, 0.1)
        free = []
        for s in ss:
            if wall == "front":
                r = [s - 0.3, 0.0, s + 0.3, depth]
            elif wall == "back":
                r = [s - 0.3, c.Lv - depth, s + 0.3, c.Lv]
            elif wall == "left":
                r = [0.0, s - 0.3, depth, s + 0.3]
            else:
                r = [c.Lu - depth, s - 0.3, c.Lu, s + 0.3]
            free.append(c.free(r, pad=0.05, keep=True, keep_pad=0.05))
        run = []
        for s, f in zip(ss, free):
            if f:
                run.append(s)
            elif run:
                if run[-1] - run[0] >= min_len - 0.6:
                    out.append(W.spot(c, wall, run[0], run[-1], zone, orients=("out", "along+", "along-")))
                run = []
        if run and run[-1] - run[0] >= min_len - 0.6:
            out.append(W.spot(c, wall, run[0], run[-1], zone, orients=("out", "along+", "along-")))
    return [o for o in out if o]


def corridor_spots(W):
    """Wall stretches of the corridor legs (slot A in the airlock-side stretch of the south leg, C-relocated on the
    corridor wall outside a room), clear of doors, fire doors, crossing elements and repeaters."""
    lay = W.lay
    xs = lay.params.get("corridor_sill_x")
    for leg in ("C1", "C2", "C3"):
        c = W.ctx.get(leg)
        if c is None:
            continue
        Lmax = max(c.Lu, c.Lv)
        for wall in ("front", "back", "left", "right"):
            L = wall_len(c, wall)
            if L < Lmax - 1e-6:
                continue                       # the long walls of the leg only
            ss = np.arange(0.35, L - 0.35 + 1e-9, 0.1)
            ok = []
            wx = []
            for s in ss:
                x, y = c.f.w(*wall_point(c, wall, s, 0.1))[:2]
                wx.append(x)
                good = True
                for d in lay.doors:
                    if leg not in d.rooms:
                        continue
                    dx, dy = d.xy()
                    if math.hypot(dx - x, dy - y) < d.width / 2 + 0.75:
                        good = False
                        break
                for f in getattr(W, "fire_doors", []):
                    if f["rooms"][0] == leg and math.hypot(f["pos"][0] - x, f["pos"][1] - y) < 1.6:
                        good = False
                for e in W.elements.values():
                    if e.room != leg:
                        continue
                    ex = [p[0] for p in e.footprint]
                    ey = [p[1] for p in e.footprint]
                    if min(ex) - 0.7 <= x <= max(ex) + 0.7 and min(ey) - 0.7 <= y <= max(ey) + 0.7:
                        good = False
                for r in W.repeaters:
                    if math.hypot(r["pos"][0] - x, r["pos"][1] - y) < 0.6:
                        good = False
                for p in getattr(W, "partitions", []):
                    if p["leg"] == leg and abs(p["x"] - x) < 1.0:
                        good = False
                ok.append(good)
            # runs of free positions; on the south leg the part on the airlock side of the sill is zone a_stretch
            zone_of = []
            for x in wx:
                if leg == "C1" and xs is not None:
                    zone_of.append("a_stretch" if x <= xs - 0.75 else ("corridor" if x >= xs + 1.0 else None))
                else:
                    zone_of.append("corridor")
            run = []
            rz = None
            for s, f, zz in list(zip(ss, ok, zone_of)) + [(None, False, None)]:
                if f and zz is not None and (not run or zz == rz):
                    run.append(s)
                    rz = zz
                    continue
                if run and run[-1] - run[0] >= 0.3:
                    W.spot(c, wall, run[0], run[-1], rz, orients=("out", "along-", "along+") if rz == "a_stretch"
                           else ("out",))
                run = [s] if (f and zz is not None) else []
                rz = zz if run else None


# ============================================================================================ geometry
def _wv(c, wall, du, dv):
    """World direction of a local (du, dv)."""
    return c.wyaw(du, dv)


def build(W, spot, typ, s, z, orient, name, service, g):
    """Build the station for instrument type `typ` (gauge_63 / ... / thermal) at spot `spot`, position s along its
    wall (or on the tank shell), height z, orientation 'out' / 'along+' / 'along-'. Returns (slot, params)."""
    if spot.get("kind") == "tank":
        return _tank_station(W, spot, typ, z, name, service, g)
    if spot.get("kind") == "rule":
        return _rule_slot(W, spot, typ)
    c = W.ctx[spot["room"]]
    wall = spot["wall"]
    nu, nv = WALL_N[wall]
    tu, tv = WALL_T[wall]
    body = "station/%s" % name
    if orient in ("out", "vpipe"):
        fu, fv = nu, nv                                # face normal (local)
    else:
        sg = 1 if orient == "along+" else -1
        fu, fv = sg * tu, sg * tv
    yaw = c.wyaw(fu, fv)

    def L(sv, d):                                      # local point at s + sv along the wall, d from the face
        return wall_point(c, wall, s + sv, d)

    def box(nm, sa, sb, da, db, z0, z1, mat, group=0):
        (u0, v0), (u1, v1) = L(sa, da), L(sb, db)
        return c.box("%s" % nm, min(u0, u1), min(v0, v1), z0, max(u0, u1), max(v0, v1), z1, mat, group=group,
                     body=body)

    v2 = bool((W.cfg.get("ladder") or {}).get("v2"))     # version 1 keeps its materials

    def pipe(nm, pts, r, sv, group=0):
        # the station's tubing on radwaste service: not signal yellow (the dressing leaves station geoms
        # as they are): furnish.PIPE_MAT["rw_station"]
        sv = "rw_station" if sv == "rw" and v2 else sv
        return c.pipe(nm, [(L(a, d)[0], L(a, d)[1], zz) for a, d, zz in pts], r, sv, body=body, group=group)
    key = typ.split("_")[0] if typ.startswith("gauge") else typ
    sv = service.get("code", "cw")
    out = {"mount": service.get("mount", ""), "service": service.get("text", SERVICE_TEXT.get(sv, sv))}
    if key == "gauge":
        case = int(typ.split("_")[1])
        D = case / 1000.0
        if orient == "out":
            d_face = 0.10 + 0.34 * D
            # bracket plate on the wall, an impulse line from the ceiling down to the gauge
            box("plate", -0.07, 0.07, 0.0, 0.012, z - 0.5 * D - 0.12, z + 0.5 * D + 0.04, "steel_dark")
            box("arm", -0.012, 0.012, 0.012, d_face - 0.34 * D, z - 0.012, z + 0.012, "steel")
            pipe("impulse", [(0.12, 0.05, 2.45), (0.12, 0.05, z + 0.25), (0.12, d_face - 0.17 * D, z + 0.25),
                             (0.0, d_face - 0.17 * D, z + 0.25), (0.0, d_face - 0.17 * D, z + 0.5 * D + 0.01)],
                 0.009, sv, group=1)
            fu_l, fv_l = L(0.0, d_face)
        else:
            # a post 0.3 m from the wall, the gauge on it facing along the wall
            box("post", -0.03, 0.03, 0.27, 0.33, 0.0, z - 0.5 * D - 0.02, "galv")
            box("base", -0.12, 0.12, 0.18, 0.42, 0.0, 0.012, "steel_dark")
            box("arm", -0.035, 0.035, 0.0, 0.33, z - 0.5 * D - 0.06, z - 0.5 * D - 0.03, "galv")
            fu_l, fv_l = L(sg * (0.34 * D + 0.02), 0.30)
        slot = c.slot("gauge", fu_l, fv_l, z, fu, fv, out["mount"] or "station", out["service"], body,
                      case_hint=case)
        slot["case_fixed"] = case
        return slot, {"case_mm": case}
    if key == "level":
        Lw = float(W.cfg["ladder"]["level_window_m"])
        zc = z
        zf = Lw / 2 + 0.06
        rp = 0.16
        if orient == "out":
            dp = 0.03 + rp
            c.vcyl("pot", *L(0.0, dp), 0.15, zc + zf + 0.35, rp, "tank_grey", body=body)
            c.vcyl("pot_head", *L(0.0, dp), zc + zf + 0.35, zc + zf + 0.42, rp * 0.85, "tank_grey", body=body)
            c.vcyl("pot_leg", *L(0.0, dp), 0.0, 0.15, 0.05, "steel_dark", body=body)
            pipe("pot_in", [(0.0, dp, zc + zf + 0.42), (0.0, dp, 2.45)], 0.025, sv, group=1)
            tube_d = dp + rp + 0.13
            for zz in (zc - zf, zc + zf):
                pipe("noz%.2f" % zz, [(0.0, dp + rp - 0.01, zz), (0.0, tube_d - 0.01, zz)], 0.02, "steel", group=1)
            fu_l, fv_l = L(0.0, tube_d)
        else:
            dp = 0.35
            c.vcyl("pot", *L(0.0, dp), 0.15, zc + zf + 0.35, rp, "tank_grey", body=body)
            c.vcyl("pot_leg", *L(0.0, dp), 0.0, 0.15, 0.05, "steel_dark", body=body)
            box("pot_bracket", -0.04, 0.04, 0.0, dp - rp, zc - 0.05, zc + 0.05, "galv")
            pipe("pot_in", [(0.0, dp, zc + zf + 0.35), (0.0, dp, 2.45)], 0.025, sv, group=1)
            for zz in (zc - zf, zc + zf):
                pipe("noz%.2f" % zz, [(sg * (rp - 0.01), dp, zz), (sg * (rp + 0.12), dp, zz)], 0.02, "steel",
                     group=1)
            fu_l, fv_l = L(sg * (rp + 0.13), dp)
        slot = c.slot("level", fu_l, fv_l, zc, fu, fv, out["mount"] or "level pot", out["service"], body,
                      span=[zc - Lw / 2 - 0.06, zc + Lw / 2 + 0.06], tank="POT")
        return slot, {"window_m": Lw}
    if key in ("valve_lever", "valve_handwheel"):
        rr = 0.045 if key == "valve_lever" else 0.055
        if orient == "vpipe" and key == "valve_lever":
            # a vertical line on wall brackets, floor to ceiling, the quarter-turn valve on it facing out of the wall
            # (the training hall's vertical-pipe valve: the lever's angle is read against
            # the pipe, and the buildings have such valves on their rule slots)
            dl = 0.06 + rr + 0.04
            pipe("line", [(0.0, dl, 0.06), (0.0, dl, 2.45)], rr, sv)
            for k, zz in enumerate((0.35, 1.95)):
                box("brk%d" % k, -0.03, 0.03, 0.0, dl - rr, zz - 0.04, zz + 0.04, "galv")
            fu_l, fv_l = L(0.0, dl)
            slot = c.slot(key, fu_l, fv_l, z, fu, fv, out["mount"] or "vertical process line", out["service"], body,
                          pipe={"axis": [0.0, 0.0, 1.0], "r": rr}, stem="out")
            return slot, {"pipe_axis": [0.0, 0.0, 1.0], "pipe_r": rr}
        if orient == "out":
            dl = 0.06 + rr + 0.04
            run = 0.9
            pts = [(-run, -0.1, z), (-run, dl, z), (run, dl, z), (run, -0.1, z)]
            pipe("line", pts, rr, sv)
            for k, a in enumerate((-0.6, 0.6)):
                box("brk%d" % k, a - 0.03, a + 0.03, 0.0, dl - rr, z - 0.04, z + 0.04, "galv")
            fu_l, fv_l = L(0.0, dl)
            axis = c.wvec(tu, tv)
        else:
            dl = 0.45
            pts = [(0.0, -0.1, z), (0.0, dl + 0.45, z), (0.0, dl + 0.45, 2.45)]
            pipe("line", pts, rr, sv)
            box("brk", -0.03, 0.03, 0.0, 0.15, z - 0.04, z + 0.04, "galv")
            fu_l, fv_l = L(0.0, dl)
            axis = c.wvec(nu, nv)
        slot = c.slot(key, fu_l, fv_l, z, fu, fv, out["mount"] or "process line", out["service"], body,
                      pipe={"axis": axis, "r": rr}, stem="up")
        return slot, {}
    if key == "lamps":
        if orient == "out":
            fu_l, fv_l = L(0.0, 0.125)
            pipe("conduit", [(0.1, 0.03, z + 0.1), (0.1, 0.03, 2.45)], 0.014, "ia", group=1)
        else:
            box("post", -0.03, 0.03, 0.27, 0.33, 0.0, z - 0.12, "galv")
            box("base", -0.12, 0.12, 0.18, 0.42, 0.0, 0.012, "steel_dark")
            box("bracket", -0.06, 0.06, 0.22, 0.38, z - 0.14, z - 0.11, "galv")
            fu_l, fv_l = L(sg * 0.13, 0.30)
        slot = c.slot("lamps", fu_l, fv_l, z, fu, fv, out["mount"] or "local panel", out["service"], body)
        return slot, {}
    if key == "thermal":
        fw, fh, dep = 0.36, 0.30, 0.18
        if orient == "out":
            box("hotbox", -fw / 2, fw / 2, 0.0, dep, z - fh / 2, z + fh / 2, "cabinet")
            hp = box("hotface", -fw / 2 + 0.01, fw / 2 - 0.01, dep, dep + 0.008, z - fh / 2 + 0.01, z + fh / 2 - 0.01,
                     "panel_grey", group=1)
            pipe("conduit", [(0.0, dep / 2, z + fh / 2), (0.0, dep / 2, 2.45)], 0.016, "ia", group=1)
            fu_l, fv_l = L(0.0, dep + 0.008)
        else:
            box("post", -0.03, 0.03, 0.3, 0.36, 0.0, z - fh / 2, "galv")
            box("base", -0.12, 0.12, 0.21, 0.45, 0.0, 0.012, "steel_dark")
            box("hotbox", -dep / 2 + sg * dep / 2, dep / 2 + sg * dep / 2, 0.33 - fw / 2, 0.33 + fw / 2,
                z - fh / 2, z + fh / 2, "cabinet")
            hp = box("hotface", sg * dep - 0.004, sg * dep + 0.004, 0.33 - fw / 2 + 0.01, 0.33 + fw / 2 - 0.01,
                     z - fh / 2 + 0.01, z + fh / 2 - 0.01, "panel_grey", group=1)
            fu_l, fv_l = L(sg * (dep + 0.004), 0.33)
        slot = c.slot("thermal", fu_l, fv_l, z, fu, fv, out["mount"] or "heater terminal box", out["service"], body,
                      units=[{"geom": hp.name, "pos": c.f.w(fu_l, fv_l, z), "label": "%s heater terminal box" % name,
                              "offset": 0.01}])
        return slot, {"face_w": fw - 0.02, "face_h": fh - 0.02, "tag_offset": [fw / 2 + 0.02 + 0.0954, 0.0],
                      "hot_geom": hp.name}
    raise KeyError(typ)


def _rule_slot(W, spot, typ):
    """A slot the room rule made (a pump's discharge gauge stub, a gallery valve, a panel on an MCC, a bearing
    housing): its mounting is already built; the ladder only decides the instrument (type and size)."""
    s = dict(spot["slot"])
    prm = {}
    if typ.startswith("gauge"):
        s["case_fixed"] = int(typ.split("_")[1])
        prm["case_mm"] = s["case_fixed"]
    elif typ == "level":
        Lw = float(W.cfg["ladder"]["level_window_m"])
        z = s["pos"][2]
        s["span"] = [z - Lw / 2 - 0.06, z + Lw / 2 + 0.06]
        prm["window_m"] = Lw
    elif typ == "thermal":
        u = spot["unit"]
        n = np.array([math.cos(s["yaw"]), math.sin(s["yaw"]), 0.0])
        off = float(u.get("offset", 0.13))
        pos = list(np.asarray(u["pos"], float) + off * n)
        s = dict(s, pos=pos, units=[u])
        fw, fh = spot.get("face", (0.22, 0.22))
        prm = {"face_w": fw, "face_h": fh, "tag_offset": [fw / 2 + 0.03 + 0.0954, 0.0], "hot_geom": u["geom"]}
    return s, prm


def rule_spots(W, room, typ):
    """Mount spots made of the room rule's own instrument slots of the base type of `typ` in `room`."""
    base = "gauge" if typ.startswith("gauge") else typ
    out = []
    for s in W.slots:
        if s["room"] != room or s["type"] != base or str(s.get("owner", "")).startswith("station/"):
            continue
        if base == "thermal":
            for k, u in enumerate(s.get("units", [])):
                face = (0.5, 1.4) if "door" in u["geom"] else (2 * max(0.1, u.get("offset", 0.13) - 0.03),) * 2
                out.append({"id": "%s:%d" % (s["id"], k), "kind": "rule", "room": room, "zone": "room", "slot": s,
                            "unit": u, "face": face, "wall": None})
            continue
        if base == "gauge" and typ == "gauge_63" and s["pos"][2] > 1.3:
            continue
        out.append({"id": s["id"], "kind": "rule", "room": room, "zone": "room", "slot": s, "wall": None})
    return out


def _tank_station(W, spot, typ, z, name, service, g):
    """An instrument on a tank shell at azimuth spot['phi'] (local degrees from +u), facing radially out."""
    c = W.ctx[spot["room"]]
    uc, vc = spot["center"]
    r = float(spot["r"])
    phi = math.radians(float(spot["phi"]))
    du, dv = math.cos(phi), math.sin(phi)
    tu, tv = -dv, du
    body = "station/%s" % name
    yaw = c.wyaw(du, dv)
    key = typ.split("_")[0] if typ.startswith("gauge") else typ
    sv = service.get("code", "rw")
    txt = service.get("text", "tank %s" % spot.get("label", ""))

    def at(rad, along=0.0):
        return uc + rad * du + along * tu, vc + rad * dv + along * tv

    if key == "gauge":
        case = int(typ.split("_")[1])
        D = case / 1000.0
        a, b = at(r - 0.02), at(r + 0.10)
        c.cyl("noz", (a[0], a[1], z - 0.5 * D - 0.04), (b[0], b[1], z - 0.5 * D - 0.04), 0.016, "steel", body=body)
        c.cyl("riser", (b[0], b[1], z - 0.5 * D - 0.05), (b[0], b[1], z - 0.5 * D + 0.01), 0.012, "steel", body=body)
        f = at(r + 0.12 + 0.34 * D)
        slot = c.slot("gauge", f[0], f[1], z, du, dv, "%s nozzle" % spot.get("label", "tank"), txt, body,
                      case_hint=case)
        slot["case_fixed"] = case
        return slot, {"case_mm": case}
    if key == "level":
        Lw = float(W.cfg["ladder"]["level_window_m"])
        zf = Lw / 2 + 0.06
        for zz in (z - zf, z + zf):
            a, b = at(r - 0.02), at(r + 0.12)
            c.cyl("noz%.2f" % zz, (a[0], a[1], zz), (b[0], b[1], zz), 0.02, "steel", body=body)
        f = at(r + 0.13)
        slot = c.slot("level", f[0], f[1], z, du, dv, "%s level gauge" % spot.get("label", "tank"), txt, body,
                      span=[z - Lw / 2 - 0.06, z + Lw / 2 + 0.06], tank=spot.get("label", "TANK"))
        return slot, {"window_m": Lw}
    if key in ("valve_lever", "valve_handwheel"):
        rr = 0.045 if key == "valve_lever" else 0.05
        dl = r + 0.22
        a, b = at(r - 0.02), at(dl)
        # radwaste tank piping of the station: not signal yellow in version 2 (station geoms are not
        # re-dressed); version 1 keeps its material
        rwm = "pipe_rw_station" if (W.cfg.get("ladder") or {}).get("v2") else "pipe_rw"
        c.cyl("noz", (a[0], a[1], z), (b[0], b[1], z), rr, rwm, body=body)
        e1, e2 = at(dl, -0.45), at(dl, 0.45)
        c.cyl("line", (e1[0], e1[1], z), (e2[0], e2[1], z), rr, rwm, body=body)
        c.sph("elb", b[0], b[1], z, rr * 1.12, rwm, body=body)
        for k, e in enumerate((e1, e2)):
            c.vcyl("cap%d" % k, e[0], e[1], z - rr * 1.4, z + rr * 1.4, 0.012, "steel_dark", group=1, body=body)
        f = at(dl)
        slot = c.slot(key, f[0], f[1], z, du, dv, "%s drain line" % spot.get("label", "tank"), txt, body,
                      pipe={"axis": c.wvec(tu, tv), "r": rr}, stem="up")
        return slot, {}
    if key == "lamps":
        a, b = at(r - 0.02), at(r + 0.06)
        c.cyl("brk", (a[0], a[1], z - 0.05), (b[0], b[1], z - 0.05), 0.02, "galv", body=body)
        f = at(r + 0.06 + 0.125)
        slot = c.slot("lamps", f[0], f[1], z, du, dv, "%s level switch panel" % spot.get("label", "tank"), txt, body)
        return slot, {}
    if key == "thermal":
        fw, fh, dep = 0.30, 0.26, 0.16
        pm = at(r + 0.5 * (dep - 0.02))
        c.P.box("%s/hotbox" % body, c.f.w(pm[0], pm[1], z), (0.5 * (dep + 0.02), fw / 2, fh / 2), yaw=yaw,
                mat="cabinet", body=body, owner=c.rid)
        pf = at(r + dep + 0.004)
        hp = c.P.box("%s/hotface" % body, c.f.w(pf[0], pf[1], z), (0.004, fw / 2 - 0.01, fh / 2 - 0.01), yaw=yaw,
                     mat="panel_grey", body=body, owner=c.rid, group=1)
        f = at(r + dep + 0.008)
        slot = c.slot("thermal", f[0], f[1], z, du, dv, "%s heater terminal box" % spot.get("label", "tank"), txt,
                      body, units=[{"geom": hp.name, "pos": c.f.w(f[0], f[1], z), "label": "%s heater" % name,
                                    "offset": 0.01}])
        return slot, {"face_w": fw - 0.02, "face_h": fh - 0.02, "tag_offset": [fw / 2 + 0.02 + 0.0954, 0.0],
                      "hot_geom": hp.name}
    raise KeyError(typ)


def blank_mount(W, spot, s, z, orient, name):
    """The empty mounting left where an instrument was removed (slot C-relocated's drawn place): a bare bracket
    plate with a capped stub."""
    if spot.get("kind") == "tank":
        c = W.ctx[spot["room"]]
        uc, vc = spot["center"]
        r = float(spot["r"])
        phi = math.radians(float(spot["phi"]))
        a = (uc + (r - 0.02) * math.cos(phi), vc + (r - 0.02) * math.sin(phi))
        b = (uc + (r + 0.1) * math.cos(phi), vc + (r + 0.1) * math.sin(phi))
        c.cyl("blank/%s" % name, (a[0], a[1], z), (b[0], b[1], z), 0.016, "steel", body="station/%s" % name)
        c.cyl("blank/%s/cap" % name, (b[0], b[1], z), (b[0] + 0.01 * math.cos(phi), b[1] + 0.01 * math.sin(phi), z),
              0.024, "steel_dark", body="station/%s" % name)
        return
    c = W.ctx[spot["room"]]
    wall = spot["wall"]
    (u0, v0), (u1, v1) = wall_point(c, wall, s - 0.07, 0.0), wall_point(c, wall, s + 0.07, 0.012)
    c.box("blank/%s/plate" % name, min(u0, u1), min(v0, v1), z - 0.12, max(u0, u1), max(v0, v1), z + 0.08,
          "steel_dark", body="station/%s" % name)
    (a0, b0), (a1, b1) = wall_point(c, wall, s - 0.012, 0.012), wall_point(c, wall, s + 0.012, 0.1)
    c.box("blank/%s/stub" % name, min(a0, a1), min(b0, b1), z - 0.012, max(a0, a1), max(b0, b1), z + 0.012, "steel",
          body="station/%s" % name)


def spot_rect(c, spot, s, orient, depth=0.55, half=0.5):
    """The local floor rect a station at s occupies (for keeping stations apart)."""
    wall = spot["wall"]
    (u0, v0) = wall_point(c, wall, s - half, 0.0)
    (u1, v1) = wall_point(c, wall, s + half, depth)
    return [min(u0, u1), min(v0, v1), max(u0, u1), max(v0, v1)]


def random_z(cfg, typ, g):
    lo, hi = cfg["ladder"]["mount"]["z"][typ]
    return round(uniform(g, float(lo), float(hi)) / 0.05) * 0.05
