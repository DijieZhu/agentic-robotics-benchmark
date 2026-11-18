"""Furnishing rules, part 1: the room context (a local frame with u along the main-door wall and v into the room),
common items, and the rules for the airlock, the corridor, the pipe chase, the annex and the decor rooms.
Part 2 (furnish_plant.py) has the process rooms."""

from __future__ import annotations

import math

import numpy as np

from .prims import Frame2D
from .util import choice, quat_axis_angle, rect_overlap, uniform, weighted_key
from .world import World, sample_size

OPP = {"S": "N", "N": "S", "W": "E", "E": "W"}
PIPE_MAT = {"cw": "pipe_cw", "dw": "pipe_dw", "st": "pipe_st", "rw": "pipe_rw", "ia": "pipe_ia", "fw": "pipe_fw",
            "dr": "pipe_dr", "oil": "pipe_oil",
            # an instrument station's own tubing on radwaste service (stations.py): the dressed look's radwaste colour
            # (the dressing recolours `pipe_rw` everywhere but on the protected `station/` geoms, which kept signal
            # yellow there: found by a colour test, 2026-11-16)
            "rw_station": "pipe_rw_station"}
SERVICE_TEXT = {"cw": "cooling water", "dw": "demineralised water", "st": "steam", "rw": "radwaste liquor",
                "ia": "instrument air", "fw": "fire water", "dr": "drain", "oil": "lube oil"}


def frame_for(room) -> Frame2D:
    x0, y0, x1, y1 = room.inner
    f = room.front or "S"
    if f == "S":
        return Frame2D((x0, y0), (1.0, 0.0), x1 - x0, y1 - y0)
    if f == "N":
        return Frame2D((x1, y1), (-1.0, 0.0), x1 - x0, y1 - y0)
    if f == "W":
        return Frame2D((x0, y1), (0.0, -1.0), y1 - y0, x1 - x0)
    return Frame2D((x1, y0), (0.0, 1.0), y1 - y0, x1 - x0)


def _local_sides(front):
    left = {"S": "W", "N": "E", "W": "N", "E": "S"}[front]
    return {front: "front", OPP[front]: "back", left: "left", OPP[left]: "right"}


class Ctx:
    def __init__(self, W: World, room):
        self.W = W
        self.P = W.P
        self.cfg = W.cfg
        self.room = room
        self.rid = room.id
        self.g = W.rng("furnish/" + room.id)
        self.f = frame_for(room)
        self.Lu, self.Lv = self.f.Lu, self.f.Lv
        self.body = "room/%s" % room.id
        self.placed = []
        self.keep = [self.rect_l(*r) for r in W.keepouts.get(room.id, [])]
        sides = _local_sides(room.front or "S")
        self.doors = []
        lay = W.lay
        for did in room.doors:
            d = lay.door(did)
            w = lay.wall(d.wall)
            if w.axis == "y":
                side = "S" if abs(w.coord - room.rect[1]) < 1e-6 else "N"
            else:
                side = "W" if abs(w.coord - room.rect[0]) < 1e-6 else "E"
            x, y = d.xy()
            u, v = self.f.to_local(x, y)
            lw = sides[side]
            c = u if lw in ("front", "back") else v
            self.doors.append({"d": d, "wall": lw, "c": c, "w": d.width, "into": d.swing_into == room.id})

    # ------------------------------------------------------------------ geometry helpers
    def rect_l(self, x0, y0, x1, y1):
        pts = [self.f.to_local(x, y) for x, y in ((x0, y0), (x1, y0), (x1, y1), (x0, y1))]
        us = [p[0] for p in pts]
        vs = [p[1] for p in pts]
        return [min(us), min(vs), max(us), max(vs)]

    def rect_w(self, u0, v0, u1, v1):
        return self.f.lrect_to_world(u0, v0, u1, v1)

    def wyaw(self, du, dv) -> float:
        x = du * self.f.u[0] + dv * self.f.v[0]
        y = du * self.f.u[1] + dv * self.f.v[1]
        return math.atan2(y, x)

    def wvec(self, du, dv, dz=0.0):
        return [du * self.f.u[0] + dv * self.f.v[0], du * self.f.u[1] + dv * self.f.v[1], dz]

    def box(self, name, u0, v0, z0, u1, v1, z1, mat, group=0, body=None, **kw):
        if u1 - u0 < 1e-4 or v1 - v0 < 1e-4 or z1 - z0 < 1e-4:
            return None
        c = self.f.w(0.5 * (u0 + u1), 0.5 * (v0 + v1), 0.5 * (z0 + z1))
        return self.P.box("%s/%s" % (body or self.body, name), c,
                          (0.5 * (u1 - u0), 0.5 * (v1 - v0), 0.5 * (z1 - z0)), yaw=self.f.yaw, mat=mat,
                          group=group, body=body or self.body, owner=kw.pop("owner", self.rid), **kw)

    def boxc(self, name, u, v, z, hu, hv, hz, mat, yaw_l=0.0, group=0, body=None, **kw):
        c = self.f.w(u, v, z)
        return self.P.box("%s/%s" % (body or self.body, name), c, (hu, hv, hz), yaw=self.f.yaw + yaw_l, mat=mat,
                          group=group, body=body or self.body, owner=kw.pop("owner", self.rid), **kw)

    def cyl(self, name, a, b, r, mat, group=0, body=None, kind="cylinder", **kw):
        return self.P.cyl("%s/%s" % (body or self.body, name), self.f.w(*a), self.f.w(*b), r, kind=kind, mat=mat,
                          group=group, body=body or self.body, owner=kw.pop("owner", self.rid), **kw)

    def vcyl(self, name, u, v, z0, z1, r, mat, group=0, body=None, **kw):
        return self.cyl(name, (u, v, z0), (u, v, z1), r, mat, group=group, body=body, **kw)

    def sph(self, name, u, v, z, r, mat, group=0, body=None, **kw):
        return self.P.sphere("%s/%s" % (body or self.body, name), self.f.w(u, v, z), r, mat=mat, group=group,
                             body=body or self.body, owner=kw.pop("owner", self.rid), **kw)

    def pipe(self, name, pts, r, service, body=None, group=0, elbows=True, flange_every=None):
        mat = PIPE_MAT.get(service, "pipe_cw")
        return self.P.pipe("%s/%s" % (body or self.body, name), [self.f.w(*p) for p in pts], r, mat,
                           body or self.body, owner=self.rid, elbows=elbows, group=group, flange_every=flange_every)

    # ------------------------------------------------------------------ occupancy
    def free(self, rect, pad=0.0, keep=True, keep_pad=None) -> bool:
        if keep:
            kp = pad if keep_pad is None else keep_pad
            for k in self.keep:
                if rect_overlap(rect, k, kp):
                    return False
        for k in self.placed:
            if rect_overlap(rect, k, pad):
                return False
        return rect[0] >= -1e-6 and rect[1] >= -1e-6 and rect[2] <= self.Lu + 1e-6 and rect[3] <= self.Lv + 1e-6

    def take(self, rect):
        self.placed.append(list(rect))

    def door_spans(self, wall):
        return [(dd["c"] - dd["w"] / 2, dd["c"] + dd["w"] / 2) for dd in self.doors if dd["wall"] == wall]

    def wall_free_spans(self, wall, margin=0.45):
        L = self.Lu if wall in ("front", "back") else self.Lv
        spans = sorted(self.door_spans(wall))
        out = []
        cur = 0.0
        for a, b in spans:
            if a - margin > cur + 0.3:
                out.append((cur, a - margin))
            cur = max(cur, b + margin)
        if L - cur > 0.3:
            out.append((cur, L))
        return out

    # ------------------------------------------------------------------ records
    def slot(self, typ, u, v, z, du, dv, mount, service, owner, pitch=0.0, **extra):
        return self.W.slot(typ, self.f.w(u, v, z), self.wyaw(du, dv), self.rid, mount, service, owner,
                           pitch=pitch, **extra)

    def element(self, kind, cls, param, lrect=None, zone=False, note="", poly=None):
        if poly is None:
            r = self.rect_w(*lrect)
            poly = [[r[0], r[1]], [r[2], r[1]], [r[2], r[3]], [r[0], r[3]]]
        return self.W.element(kind, cls, param, self.rid, poly, zone=zone, note=note)

    def equip(self, kind, label, lrect, height, on_drawing=True, eid=None, shape="rect", center=None,
              radius=None):
        r = self.rect_w(*lrect)
        c = self.f.w(*center)[:2] if center is not None else None
        return self.W.equip(kind, label, self.rid, r, height, on_drawing=on_drawing, eid=eid, shape=shape,
                            center=c, radius=radius)

    def view(self, key, what, u, v, tu, tv, tz, h=None, fovy=70.0):
        h = float(self.cfg["render"]["eye_height"]) if h is None else h
        self.W.views.append({"name": "cam/eye_%s" % key, "pos": self.f.w(u, v, h), "target": self.f.w(tu, tv, tz),
                             "what": what, "fovy": fovy})

    def light(self, u, v, level=None):
        self.W.light(self.f.w(u, v, 2.45), level or self.room.light, self.rid)

    def lights_grid(self, n_u=None, n_v=None, level=None):
        n_u = n_u or max(1, int(round(self.Lu / 4.0)))
        n_v = n_v or max(1, int(round(self.Lv / 4.5)))
        for i in range(n_u):
            for j in range(n_v):
                self.light((i + 0.5) * self.Lu / n_u, (j + 0.5) * self.Lv / n_v, level)
                self.box("lamp%d_%d" % (i, j), (i + 0.5) * self.Lu / n_u - 0.6, (j + 0.5) * self.Lv / n_v - 0.08,
                         2.38, (i + 0.5) * self.Lu / n_u + 0.6, (j + 0.5) * self.Lv / n_v + 0.08, 2.45,
                         "lamp_on" if (level or self.room.light) != "dark" else "lamp_dim", group=2)


# ============================================================================================ common items
def cabinet(c: Ctx, name, u0, v0, u1, v1, h, mat="cabinet", face=(0, -1), body=None, plinth=True, handle=True,
            seam=True, handle_z=(1.0, 1.25)):
    """A floor cabinet. seam=False: a single-leaf door (no centre seam): the cubicle that carries an indicator panel
    (since 2026-11-16: a tag plate or an instrument face never lies over a joint or seam of the surface it hangs
    on; the panel and its plate are mounted on one leaf, here the cubicle's only one); that cubicle's handle sits
    below the panel and its plate (handle_z)."""
    b = body or "%s/%s" % (c.body, name)
    z0 = 0.0
    if plinth:
        c.box("%s/plinth" % name, u0 + 0.02, v0 + 0.02, 0.0, u1 - 0.02, v1 - 0.02, 0.1, "steel_dark", body=b)
        z0 = 0.1
    p = c.box("%s/body" % name, u0, v0, z0, u1, v1, h, mat, body=b)
    if handle:
        du, dv = face
        cu, cv = 0.5 * (u0 + u1), 0.5 * (v0 + v1)
        hu, hv = 0.5 * (u1 - u0), 0.5 * (v1 - v0)
        fu, fv = cu + du * (hu + 0.012), cv + dv * (hv + 0.012)
        if du == 0:
            c.box("%s/handle" % name, fu + hu * 0.6 - 0.02, fv - 0.012, handle_z[0], fu + hu * 0.6, fv + 0.012,
                  handle_z[1], "steel", group=1, body=b)
            if seam:
                c.box("%s/seam" % name, cu - 0.004, fv - 0.006, z0 + 0.05, cu + 0.004, fv + 0.006, h - 0.05,
                      "seam", group=1, body=b)
        else:
            c.box("%s/handle" % name, fu - 0.012, fv + hv * 0.6 - 0.02, 1.0, fu + 0.012, fv + hv * 0.6, 1.25,
                  "steel", group=1, body=b)
    c.take([u0, v0, u1, v1])
    return p


def shelf_rack(c: Ctx, name, u0, v0, u1, v1, h=2.0, levels=4, fill=0.7, mat="rack_blue"):
    b = "%s/%s" % (c.body, name)
    t = 0.04
    for i, (uu, vv) in enumerate(((u0, v0), (u1 - t, v0), (u0, v1 - t), (u1 - t, v1 - t))):
        c.box("%s/post%d" % (name, i), uu, vv, 0.0, uu + t, vv + t, h, mat, body=b)
    zs = np.linspace(0.08, h - 0.15, levels)
    g = c.g
    for k, z in enumerate(zs):
        c.box("%s/shelf%d" % (name, k), u0, v0, z, u1, v1, z + 0.03, "galv", body=b)
        # boxes on the shelf
        u = u0 + 0.05
        j = 0
        while u < u1 - 0.25:
            w = uniform(g, 0.2, 0.5)
            if g.random() < fill and u + w < u1 - 0.03:
                hh = uniform(g, 0.12, min(0.38, (zs[1] - zs[0]) - 0.08 if levels > 1 else 0.4))
                col = choice(g, ["carton", "carton", "box_grey", "box_blue", "drum_small"])
                c.box("%s/item%d_%d" % (name, k, j), u, v0 + 0.05, z + 0.03, u + w, v1 - 0.05, z + 0.03 + hh,
                      col, group=1, body=b)
            u += w + 0.04
            j += 1
    c.take([u0, v0, u1, v1])


def pallet(c: Ctx, name, u, v, n_drums=4, body=None, along_u=True, drum_mat="drum_yellow", drums=True):
    """Wooden pallet (1.2 x 1.0) with up to 4 drums (0.6 dia, 0.88 high); returns drum centres (world)."""
    b = body or "%s/%s" % (c.body, name)
    hu, hv = (0.6, 0.5) if along_u else (0.5, 0.6)
    c.box("%s/pallet" % name, u - hu, v - hv, 0.0, u + hu, v + hv, 0.14, "pallet_wood", body=b)
    out = []
    if drums:
        offs = [(-0.3, -0.25), (0.3, -0.25), (-0.3, 0.25), (0.3, 0.25)] if along_u else \
            [(-0.25, -0.3), (0.25, -0.3), (-0.25, 0.3), (0.25, 0.3)]
        for k, (du, dv) in enumerate(offs[:n_drums]):
            uu, vv = u + du, v + dv
            c.vcyl("%s/drum%d" % (name, k), uu, vv, 0.14, 1.02, 0.285, drum_mat, body=b)
            c.vcyl("%s/lid%d" % (name, k), uu, vv, 1.02, 1.035, 0.29, "drum_lid", group=1, body=b)
            c.vcyl("%s/band%d" % (name, k), uu, vv, 0.55, 0.59, 0.29, "drum_band", group=1, body=b)
            out.append(c.f.w(uu, vv, 0.58))
    c.take([u - hu, v - hv, u + hu, v + hv])
    return out


def wall_panel(c: Ctx, name, wall, s, z0, z1, width, depth=0.22, mat="panel_grey", body=None):
    """A box mounted on a local wall at position s along it; returns the face centre (u, v), the face normal
    (du, dv) and the box rect."""
    if wall == "back":
        u0, u1, v0, v1 = s - width / 2, s + width / 2, c.Lv - depth, c.Lv
        face = (0, -1)
        fc = (s, c.Lv - depth)
    elif wall == "front":
        u0, u1, v0, v1 = s - width / 2, s + width / 2, 0.0, depth
        face = (0, 1)
        fc = (s, depth)
    elif wall == "left":
        u0, u1, v0, v1 = 0.0, depth, s - width / 2, s + width / 2
        face = (1, 0)
        fc = (depth, s)
    else:
        u0, u1, v0, v1 = c.Lu - depth, c.Lu, s - width / 2, s + width / 2
        face = (-1, 0)
        fc = (c.Lu - depth, s)
    c.box(name, u0, v0, z0, u1, v1, z1, mat, body=body)
    return fc, face, [u0, v0, u1, v1]


def puddle(c: Ctx, name, u0, v0, u1, v1, mu):
    """A slippery patch inside the rect: one wet disc (3 mm) and two smaller lobes at 2 mm and 1 mm, so no two top
    faces are coplanar (the earlier four stacked transparent discs 0.2 mm apart z-fought and read as a lump:
    a review's finding); opaque dark glossy material, low friction, high contact priority."""
    g = c.g
    out = []
    cu, cv = 0.5 * (u0 + u1), 0.5 * (v0 + v1)
    ru, rv = 0.5 * (u1 - u0), 0.5 * (v1 - v0)
    for k, top in enumerate((0.003, 0.002, 0.001)):
        r = min(ru, rv) * (uniform(g, 0.8, 0.95) if k == 0 else uniform(g, 0.45, 0.65))
        du = uniform(g, -1, 1) * max(0.0, ru - r)
        dv = uniform(g, -1, 1) * max(0.0, rv - r)
        out.append(c.vcyl("%s%d" % (name, k), cu + du, cv + dv, 0.0, top, r, "puddle", friction=mu,
                          priority=1, role="walkable"))
    return out


def extinguisher(c: Ctx, name, u, v):
    c.vcyl(name, u, v, 0.0, 0.55, 0.085, "red_fire")
    c.vcyl(name + "/head", u, v, 0.55, 0.62, 0.03, "steel_dark", group=1)


def bench(c: Ctx, name, u0, v0, u1, v1, h=0.45):
    c.box(name, u0, v0, 0.0, u1, v1, h, "bench_wood")
    c.take([u0, v0, u1, v1])


def ceiling_pipes(c: Ctx, along="u", n=2, services=("cw", "ia", "st")):
    """Decor pipes and a cable tray under the roof (group 2 by height)."""
    g = c.g
    L = c.Lu if along == "u" else c.Lv
    for k in range(n):
        off = uniform(g, 0.3, (c.Lv if along == "u" else c.Lu) - 0.3)
        z = uniform(g, 2.3, 2.5)
        r = uniform(g, 0.03, 0.07)
        sv = services[k % len(services)]
        if along == "u":
            c.pipe("ovh%d" % k, [(-0.15, off, z), (L + 0.15, off, z)], r, sv, group=2)
        else:
            c.pipe("ovh%d" % k, [(off, -0.15, z), (off, L + 0.15, z)], r, sv, group=2)


def cable_tray(c: Ctx, name, a, b, z=2.3, w=0.3):
    (u0, v0), (u1, v1) = a, b
    if abs(v1 - v0) < 1e-6:
        c.box(name, min(u0, u1), v0 - w / 2, z, max(u0, u1), v0 + w / 2, z + 0.06, "tray", group=2)
    else:
        c.box(name, u0 - w / 2, min(v0, v1), z, u0 + w / 2, max(v0, v1), z + 0.06, "tray", group=2)


# ============================================================================================ airlock
BASE_STATION_S = 0.35            # the base station on the airlock's left wall at this share of the room depth
BASE_STATION_DEPTH = 0.22        # its wall box
AERIAL_OUT = 0.08                # the aerial stands on an arm this far in front of the box's face (2026-11-14: behind the
                                 # face, 0.12 m inside it, the box itself cut the cell's link ray from the pad circle's
                                 # edge below it: 31 of 1128 reachable cells within 1 m of the pad on every site of an earlier generator version)


def base_station_plan(room):
    """Where f_airlock puts the base station's aerial (world x, y) and which world side of the airlock its wall is
    (the local 'left' wall), from the room's frame alone (inner rect and front side), so that the layout can hinge
    the airlock's inner door away from it before anything is built."""
    f = frame_for(room)
    s = f.Lv * BASE_STATION_S
    ax, ay = f.w(BASE_STATION_DEPTH + AERIAL_OUT, s + 0.12, 0.0)[:2]
    left = {"S": "W", "N": "E", "W": "N", "E": "S"}[room.front or "S"]
    return {"antenna_xy": (float(ax), float(ay)), "wall_side": left}


def hinge_inner_door_away(lay, room_id="A1"):
    """Hinge the airlock's inner door on the jamb farther from the base station's aerial, so that its open leaf never
    stands between the aerial and the doorway (a documents review: on a quarter of
    the candidates the leaf shaded the corridor and part of the pad circle). Call after the layout, before the
    structure is built. Returns (door id, hinge)."""
    room = lay.rooms[room_id]
    ant = base_station_plan(room)["antenna_xy"]
    for did in room.doors:
        d = lay.door(did)
        if d.kind not in ("door", "double") or "OUT" in d.rooms:
            continue
        along = ant[1] if d.axis == "x" else ant[0]
        a, b = d.center - d.width / 2, d.center + d.width / 2
        d.hinge = 1 if abs(b - along) >= abs(a - along) else -1       # _leaves: hinge > 0 puts the leaf at b
        return d.id, d.hinge
    return None, None


def f_airlock(c: Ctx):
    W = c.W
    P = c.P
    ps = float(c.cfg["airlock"]["pad_size"])
    # the deployment pad (datum at the S origin)
    P.box("pad/paint", (0.0, 0.0, 0.002), (ps / 2, ps / 2, 0.0008), mat="pad_paint", body="room/%s" % c.rid,
          owner="pad", group=1)
    W.equip("deployment_pad", "DEPLOYMENT PAD (DATUM)", c.rid, [-ps / 2, -ps / 2, ps / 2, ps / 2], 0.0)
    # base station on the back (outer door) wall side near a corner: on the left wall (base_station_plan says where)
    s = c.Lv * BASE_STATION_S
    fc, face, rect = wall_panel(c, "base_station", "left", s, 1.0, 1.55, 0.45, BASE_STATION_DEPTH, mat="base_station")
    ua = fc[0] + AERIAL_OUT                                       # the aerial clear of its own box (see AERIAL_OUT)
    c.box("base_station/arm", fc[0] - 0.02, s + 0.105, 1.52, ua + 0.015, s + 0.135, 1.55, "steel_dark", group=1)
    c.cyl("base_station/antenna", (ua, s + 0.12, 1.55), (ua, s + 0.12, 1.95), 0.012, "steel_dark", group=1)
    c.box("base_station/led", fc[0], s - 0.12, 1.45, fc[0] + 0.012, s - 0.08, 1.48, "led_green", group=1)
    W.base_station = {"pos": c.f.w(fc[0], s, 1.3), "antenna": c.f.w(ua, s + 0.12, 1.9),
                      "label": "LINK BASE STATION", "room": c.rid}
    c.take(rect)
    # bench along the right wall, frisker by the inner door, sticky mat
    bench(c, "bench", c.Lu - 0.42, c.Lv * 0.45, c.Lu, c.Lv * 0.95)
    c.box("frisker", c.Lu - 0.45, 0.25, 0.0, c.Lu - 0.05, 0.6, 1.25, "cabinet_white")
    c.take([c.Lu - 0.45, 0.25, c.Lu - 0.05, 0.6])
    c.box("frisker/screen", c.Lu - 0.47, 0.32, 1.0, c.Lu - 0.45, 0.53, 1.18, "screen", group=1)
    for dd in c.doors:
        if dd["d"].kind != "airlock_outer" and dd["wall"] == "front":
            c.box("stickymat", dd["c"] - 0.6, 0.05, 0.0, dd["c"] + 0.6, 0.85, 0.003, "rubber_blue", group=1)
    c.box("boundary_line", 0.0, 0.95, 0.0, c.Lu, 1.05, 0.003, "zone_tape", group=1)
    c.lights_grid(1, 1, "bright")


# ============================================================================================ corridor
def corridor_all(W: World):
    """Corridor features over the three legs: walkway lines, fire doors, hoses / cable protectors, repeaters,
    extinguishers, overhead services."""
    lay = W.lay
    cfg = W.cfg
    g = W.rng("corridor")
    Cc = cfg["corridor"]
    P = W.P
    legs = [lay.rooms[k] for k in ("C1", "C2", "C3")]
    W.fire_doors = []
    # door positions on the corridor walls, by leg
    occupied = {k: [] for k in ("C1", "C2", "C3")}
    for d in lay.doors:
        for rid in d.rooms:
            if rid in occupied:
                occupied[rid].append((d.center, d.width, d.axis))
    for leg in legs:
        c = Ctx(W, leg)
        W.ctx[leg.id] = c
        x0, y0, x1, y1 = leg.rect
        along_x = (x1 - x0) > (y1 - y0)
        # ------------------------------------------------ walkway lines (painted) along the leg's walls only
        lw = float(Cc["walkway_line_width"])
        ins = float(Cc["walkway_line_inset"])
        ix0, iy0, ix1, iy1 = leg.inner
        k = 0
        for w in lay.walls:
            if leg.id not in w.rooms or (w.axis == "y") != along_x:
                continue
            s_ = 1 if w.rooms[1] == leg.id else -1
            face = (w.t_pos if s_ > 0 else w.t_neg)
            off = w.coord + s_ * (face + ins)
            lo, hi = (max(w.lo, ix0), min(w.hi, ix1)) if along_x else (max(w.lo, iy0), min(w.hi, iy1))
            if hi - lo < 0.6:
                continue
            if along_x:
                P.box("%s/walkline%d" % (c.body, k), (0.5 * (lo + hi), off, 0.0025), (0.5 * (hi - lo), lw / 2, 0.001),
                      mat="walkway_yellow", body=c.body, owner=leg.id, group=1)
            else:
                P.box("%s/walkline%d" % (c.body, k), (off, 0.5 * (lo + hi), 0.0025), (lw / 2, 0.5 * (hi - lo), 0.001),
                      mat="walkway_yellow", body=c.body, owner=leg.id, group=1)
            k += 1
        # ------------------------------------------------ lights and overhead services
        n = max(1, int(round((max(x1 - x0, y1 - y0)) / 4.5)))
        for i in range(n):
            t = (i + 0.5) / n
            if along_x:
                px, py = ix0 + t * (ix1 - ix0), 0.5 * (iy0 + iy1)
                P.box("%s/lamp%d" % (c.body, i), (px, py, 2.42), (0.6, 0.08, 0.035), mat="lamp_on",
                      body=c.body, owner=leg.id, group=2)
            else:
                px, py = 0.5 * (ix0 + ix1), iy0 + t * (iy1 - iy0)
                P.box("%s/lamp%d" % (c.body, i), (px, py, 2.42), (0.08, 0.6, 0.035), mat="lamp_on",
                      body=c.body, owner=leg.id, group=2)
            W.light((px, py, 2.45), "bright", leg.id)
        if along_x:
            P.box("%s/tray" % c.body, (0.5 * (ix0 + ix1), iy1 - 0.35, 2.32), (0.5 * (ix1 - ix0), 0.15, 0.03),
                  mat="tray", body=c.body, owner=leg.id, group=2)
            for k, (yy, sv, r) in enumerate(((iy0 + 0.25, "cw", 0.06), (iy0 + 0.45, "fw", 0.05),
                                             (iy1 - 0.7, "ia", 0.025))):
                P.pipe("%s/ovh%d" % (c.body, k), [(ix0 - 0.1, yy, 2.45), (ix1 + 0.1, yy, 2.45)], r,
                       PIPE_MAT[sv], c.body, owner=leg.id, group=2)
        else:
            P.box("%s/tray" % c.body, (ix1 - 0.35, 0.5 * (iy0 + iy1), 2.32), (0.15, 0.5 * (iy1 - iy0) + 1.0, 0.03),
                  mat="tray", body=c.body, owner=leg.id, group=2)
        # ------------------------------------------------ fire doors across the long legs
        if leg.id in ("C1", "C3"):
            if W.ladder and leg.id == "C1":
                _corridor_sill(W, c, leg, g, occupied[leg.id])
            _fire_doors(W, c, leg, g, occupied[leg.id])
        # ------------------------------------------------ hoses / cable protectors
        nh = int(g.integers(Cc["hoses"][0], Cc["hoses"][1] + 1)) if leg.id != "C2" else int(g.random() < 0.5)
        for k in range(nh):
            _hose(W, c, leg, g, k, occupied[leg.id])
        # ------------------------------------------------ extinguishers along the walls
        if along_x:
            for k in range(int((ix1 - ix0) // 7.0)):
                xx = ix0 + 3.0 + 7.0 * k + uniform(g, -0.5, 0.5)
                yy = iy0 + 0.1 if k % 2 == 0 else iy1 - 0.1
                if any(abs(xx - dc) < dw / 2 + 0.6 for dc, dw, ax in occupied[leg.id] if ax == "y"):
                    continue
                P.vcyl("%s/ext%d" % (c.body, k), xx, yy, 0.0, 0.55, 0.085, mat="red_fire", body=c.body,
                       owner=leg.id)
                P.box("%s/extsign%d" % (c.body, k), (xx, yy + (0.012 if k % 2 else -0.012) * 0 + 0.0, 1.45),
                      (0.1, 0.006, 0.1), mat="red_fire", body=c.body, owner=leg.id, group=1)
    # ------------------------------------------------ link repeaters: one working (south leg), one dead
    _repeaters(W, g)


def _corridor_sill(W, c, leg, g, occupied):
    """Ladder, site-wide rule: a needs_high_step sill or hose across the south leg separates the base
    station's coverage (the airlock and the slot-A stretch) from the rest of the building."""
    cfg = W.cfg
    xs = float(W.lay.params["corridor_sill_x"])
    kind = weighted_key(g, cfg["ladder"]["corridor"]["sill_kind_weights"])
    allowed = [str(k) for k in cfg["ladder"].get("high_step_kinds", [])]
    if kind == "hose" and "hose" not in allowed:
        kind = "fire_door"
    if kind == "fire_door":
        e = _fire_door(W, leg, xs, "needs_high_step", "F0%s" % leg.id[-1], g, note="corridor sill (slot-A boundary)")
    else:
        x0, y0, x1, y1 = leg.inner
        h = sample_size(cfg, g, "hose", "needs_high_step")
        ang = uniform(g, -0.12, 0.12)
        a = (xs - math.tan(ang) * (y1 - y0) / 2, y0 + 0.02, h / 2)
        b = (xs + math.tan(ang) * (y1 - y0) / 2, y1 - 0.02, h / 2)
        body = "elem/%s/sillhose" % leg.id
        p = W.P.cyl("%s/hose" % body, a, b, h / 2, kind="capsule", mat="hose_red", body=body, owner=leg.id,
                    friction=0.9)
        wdt = h + 0.04
        poly = [[a[0] - wdt / 2, a[1]], [a[0] + wdt / 2, a[1]], [b[0] + wdt / 2, b[1]], [b[0] - wdt / 2, b[1]]]
        e = W.element("hose", "needs_high_step", {"height": h}, leg.id, poly, note="corridor sill (slot-A boundary)")
        W.tag(e, [p])
    occupied.append((xs, 1.2, "fire"))
    W.lay.params["corridor_sill"] = e.id


def _fire_door(W, leg, xx, cls, fid, g, note=None):
    """A fire door across a corridor leg at x = xx (leaves held open flat against the walls) with its sill."""
    cfg = W.cfg
    x0, y0, x1, y1 = leg.inner
    P = W.P
    body = "door/%s" % fid
    h = sample_size(cfg, g, "sill", cls)
    clear = (y1 - y0) - 0.24
    for j, yy in enumerate((y0, y1 - 0.12)):
        P.box_aabb("%s/jamb%d" % (body, j), (xx - 0.06, yy, 0.0), (xx + 0.06, yy + 0.12, 2.1),
                   mat="door_frame_fire", body=body, owner=fid)
    P.box_aabb("%s/head" % body, (xx - 0.06, y0, 2.05), (xx + 0.06, y1, 2.25), mat="door_frame_fire",
               body=body, owner=fid)
    lw = clear / 2
    for j, (yy, s) in enumerate(((y0 + 0.12 + 0.03, 1), (y1 - 0.12 - 0.03, -1))):
        P.box_aabb("%s/leaf%d" % (body, j), (xx - 0.06 - lw, yy - 0.025, 0.012), (xx - 0.06, yy + 0.025, 2.02),
                   mat="door_leaf_fire", body=body, owner=fid)
        P.box_aabb("%s/glass%d" % (body, j), (xx - 0.06 - lw * 0.7, yy - 0.027, 1.2),
                   (xx - 0.06 - lw * 0.35, yy + 0.027, 1.7), mat="glass", body=body, owner=fid, group=1)
    sp = P.box_aabb("%s/sill" % body, (xx - 0.06, y0 + 0.12, 0.0), (xx + 0.06, y1 - 0.12, h),
                    mat="hazard_stripe" if cls != "free" else "steel_plate", body=body, owner=fid, friction=0.8)
    e = W.element("sill", cls, {"height": h, "depth": 0.12}, leg.id,
                  [[xx - 0.06, y0 + 0.12], [xx + 0.06, y0 + 0.12], [xx + 0.06, y1 - 0.12], [xx - 0.06, y1 - 0.12]],
                  note=note or "fire door %s" % fid)
    W.tag(e, [sp])
    W.fire_doors.append({"id": fid, "pos": [xx, 0.5 * (y0 + y1)], "axis": "x", "coord": xx, "center": 0.5 * (y0 + y1),
                         "width": clear, "rooms": [leg.id, leg.id], "kind": "fire", "state": "open", "sill": e.id,
                         "on_drawing": True, "label": "FIRE DOOR"})
    return e


def _fire_doors(W, c, leg, g, occupied):
    cfg = W.cfg
    Cc = cfg["corridor"]
    x0, y0, x1, y1 = leg.inner
    nf = int(g.integers(Cc["fire_doors"][0], Cc["fire_doors"][1] + 1))
    if leg.id == "C3":
        nf = max(0, nf - 1)
    xlo = x0 + 3.0
    if W.ladder and leg.id == "C1":
        xlo = max(xlo, float(W.lay.params["corridor_sill_x"]) + 4.0)
    cands = []
    for k in range(200):
        xx = round(uniform(g, xlo, x1 - 3.5) / 0.05) * 0.05
        if any(abs(xx - dc) < dw / 2 + 1.25 for dc, dw, ax in occupied if ax == "y"):
            continue
        if any(abs(xx - fx) < 4.0 for fx in cands):
            continue
        cands.append(xx)
        if len(cands) >= nf:
            break
    for k, xx in enumerate(cands):
        fid = "F%d%s" % (k + 1, leg.id[-1])
        cls = W.elem_class(g, "sill", Cc["fire_door_sill_weights"])
        if leg.id == "C1" and xx < 10.0 and not W.ladder:
            cls = "free"          # the first stretch from the airlock stays walkable
        _fire_door(W, leg, xx, cls, fid, g)
        occupied.append((xx, 1.2, "fire"))


def _hose(W, c, leg, g, k, occupied):
    cfg = W.cfg
    P = W.P
    x0, y0, x1, y1 = leg.inner
    along_x = (leg.rect[2] - leg.rect[0]) > (leg.rect[3] - leg.rect[1])
    kind = "hose" if g.random() < 0.5 else "cable_protector"
    cls = W.elem_class(g, kind, cfg["corridor"]["hose_class_weights"])
    h = sample_size(cfg, g, kind, cls)
    xmin = x0 + 2.0
    if W.ladder and leg.id == "C1":
        xmin = max(xmin, float(W.lay.params.get("corridor_sill_x", x0)) + 1.6)   # never in the slot-A stretch
    for _ in range(50):
        if along_x:
            t = round(uniform(g, xmin, x1 - 2.0) / 0.05) * 0.05
            if any(abs(t - dc) < dw / 2 + 0.6 for dc, dw, ax in occupied):
                continue
            break
        else:
            t = round(uniform(g, y0 + 1.0, y1 - 1.0) / 0.05) * 0.05
            break
    if leg.id == "C1" and along_x and t < 10.0 and cls != "free" and not W.ladder:
        cls = "free"                      # the first stretch from the airlock stays walkable
        h = sample_size(cfg, g, kind, cls)
    body = "elem/%s/%s%d" % (leg.id, kind, k)
    ang = uniform(g, -0.25, 0.25) if kind == "hose" else 0.0
    if along_x:
        a = (t - math.tan(ang) * (y1 - y0) / 2, y0 + 0.02, h / 2)
        b = (t + math.tan(ang) * (y1 - y0) / 2, y1 - 0.02, h / 2)
    else:
        a = (x0 + 0.02, t - math.tan(ang) * (x1 - x0) / 2, h / 2)
        b = (x1 - 0.02, t + math.tan(ang) * (x1 - x0) / 2, h / 2)
    if kind == "hose":
        p = P.cyl("%s/hose" % body, a, b, h / 2, kind="capsule", mat="hose_red" if g.random() < 0.5 else "hose_black",
                  body=body, owner=leg.id, friction=0.9)
        prims = [p]
    else:
        if along_x:
            prims = ramped_protector(P, body, owner=leg.id, axis="y", c=t, lo=y0 + 0.02, hi=y1 - 0.02, h=h)
        else:
            prims = ramped_protector(P, body, owner=leg.id, axis="x", c=t, lo=x0 + 0.02, hi=x1 - 0.02, h=h)
    pts = np.array([a[:2], b[:2]])
    wdt = protector_width(h) if kind == "cable_protector" else h + 0.04
    if along_x:
        poly = [[pts[0][0] - wdt / 2, pts[0][1]], [pts[0][0] + wdt / 2, pts[0][1]],
                [pts[1][0] + wdt / 2, pts[1][1]], [pts[1][0] - wdt / 2, pts[1][1]]]
    else:
        poly = [[pts[0][0], pts[0][1] - wdt / 2], [pts[0][0], pts[0][1] + wdt / 2],
                [pts[1][0], pts[1][1] + wdt / 2], [pts[1][0], pts[1][1] - wdt / 2]]
    e = W.element(kind, cls, {"height": h}, leg.id, poly, note="across the corridor")
    W.tag(e, prims)
    occupied.append((t, 0.6, "hose"))


PROT_TOP = 0.10                 # flat top of a ramped cable protector (m)
PROT_SLOPE_DEG = 25.0           # its side ramps (the gait's 'bump' terrain: 25 deg sides)


def protector_width(h):
    return PROT_TOP + 2.0 * h / math.tan(math.radians(PROT_SLOPE_DEG))


def ramped_protector(P, body, owner, axis, c, lo, hi, h, mat="cable_protector"):
    """A cable protector with ramped sides across a passage: a flat top PROT_TOP wide and two 25 deg ramps.
    axis: the direction it runs along ('y': it lies along y at x = c, from lo to hi)."""
    th = math.radians(PROT_SLOPE_DEG)
    run = h / math.tan(th)
    out = []
    if axis == "y":
        out.append(P.box_aabb("%s/prot" % body, (c - PROT_TOP / 2, lo, 0.0), (c + PROT_TOP / 2, hi, h), mat=mat,
                              body=body, owner=owner, friction=0.9))
    else:
        out.append(P.box_aabb("%s/prot" % body, (lo, c - PROT_TOP / 2, 0.0), (hi, c + PROT_TOP / 2, h), mat=mat,
                              body=body, owner=owner, friction=0.9))
    L = math.hypot(run, h)
    T = 0.5 * h
    for sgn in (-1, 1):
        # top face of the ramp from (edge, z = h) down to (edge + sgn * run, z = 0); the box hangs below that face
        e = c + sgn * PROT_TOP / 2
        nv, nz = sgn * math.sin(th), math.cos(th)          # outward-up normal of the ramp face in (across, z)
        mc, mz = e + sgn * run / 2, h / 2
        cc, cz = mc - nv * T / 2, mz - nz * T / 2
        if axis == "y":
            q = quat_axis_angle((0.0, 1.0, 0.0), sgn * th)
            out.append(P.box("%s/ramp%+d" % (body, sgn), (cc, 0.5 * (lo + hi), cz), (L / 2, 0.5 * (hi - lo), T / 2),
                             quat=q, mat=mat, body=body, owner=owner, friction=0.9))
        else:
            q = quat_axis_angle((1.0, 0.0, 0.0), -sgn * th)
            out.append(P.box("%s/ramp%+d" % (body, sgn), (0.5 * (lo + hi), cc, cz), (0.5 * (hi - lo), L / 2, T / 2),
                             quat=q, mat=mat, body=body, owner=owner, friction=0.9))
    return [p for p in out if p is not None]


def _repeaters(W, g):
    lay = W.lay
    P = W.P
    out = []
    for k, (leg, working) in enumerate((("C1", True), ("C3", False))):
        r = lay.rooms[leg]
        x0, y0, x1, y1 = r.inner
        xx = round(uniform(g, x0 + 0.45 * (x1 - x0), x0 + 0.75 * (x1 - x0)) / 0.05) * 0.05
        side = 1 if g.random() < 0.5 else -1
        yy = y1 - 0.08 if side > 0 else y0 + 0.08
        body = "link/repeater%d" % (k + 1)
        P.box("%s/box" % body, (xx, yy, 2.0), (0.14, 0.07, 0.12), mat="base_station", body=body, owner="link")
        P.cyl("%s/ant" % body, (xx + 0.08, yy, 2.12), (xx + 0.08, yy, 2.32), 0.01, mat="steel_dark", body=body,
              owner="link", group=1)
        P.box("%s/led" % body, (xx - 0.05, yy - side * 0.072, 2.05), (0.012, 0.004, 0.012),
              mat="led_green" if working else "led_off", body=body, owner="link", group=1)
        out.append({"id": "RP%d" % (k + 1), "pos": [xx, yy, 2.0], "room": leg, "working": working,
                    "label": "LINK REPEATER %d" % (k + 1)})
    W.repeaters = out


# ============================================================================================ pipe chase
def f_pipe_chase(c: Ctx):
    g = c.g
    clr = float(c.W.lay.params["chase_clearance"])
    Lu, Lv = c.Lu, c.Lv
    ladder = bool(c.W.ladder)
    # the valve station: a stretch in the middle where the pipes rise and a robot can stand
    lo = 0.3 * Lv
    st_len = 1.2 if not ladder else round(min(3.4, max(2.4, Lv - 2.5)) / 0.05) * 0.05
    if ladder:
        vs0 = round(0.5 * (Lv - st_len) / 0.05) * 0.05
    else:
        vs0 = round(uniform(g, lo, max(lo + 0.01, 0.7 * Lv - 1.2)) / 0.05) * 0.05
    vs1 = vs0 + st_len
    elem_prims = []
    # support frames: cross beams every ~1.5 m outside the station. Their underside IS the lowest point of the nest
    # (the recorded clearance, sampled inside the needs_crouch band); the lowest pipes rest on them (a review:
    # the beams used to hang 0.05 m below the recorded clearance)
    bt = 0.05
    v = 0.4
    k = 0
    while v < Lv - 0.3:
        if not (vs0 - 0.4 < v < vs1 + 0.4):
            elem_prims.append(c.box("beam%d" % k, 0.0, v - 0.03, clr, Lu, v + 0.03, clr + bt, "galv"))
            c.box("hanger%d" % k, 0.0, v - 0.02, clr, 0.04, v + 0.02, 2.4, "galv")
            c.box("hangerb%d" % k, Lu - 0.04, v - 0.02, clr, Lu, v + 0.02, 2.4, "galv")
            k += 1
        v += 1.5
    # lowest layer: pipes side by side (bottoms on the beams), jogging up over the station
    if not ladder:
        lay1 = [(0.2, 0.1, "cw"), (0.62, 0.085, "rw"), (1.04, 0.1, "dw")]
    else:
        n1 = max(3, int(round((Lu - 0.1) / 0.42)))
        svs = ["cw", "rw", "dw", "st", "cw"]
        lay1 = [(0.2 + k * (Lu - 0.4) / (n1 - 1), 0.1 if k % 2 == 0 else 0.085, svs[k % 5]) for k in range(n1)]
    for k, (u, r, sv) in enumerate(lay1):
        u = min(u, Lu - r - 0.02)
        z = clr + bt + r
        zt = 1.05 + 0.25 * k if not ladder else 1.5 + 0.12 * k
        pts = [(u, -0.1, z), (u, vs0 - 0.25, z), (u, vs0 - 0.05, zt), (u, vs1 + 0.05, zt), (u, vs1 + 0.25, z),
               (u, Lv + 0.1, z)]
        ps = c.pipe("nest%d" % k, pts, r, sv)
        elem_prims += [p for p in ps if p.zrange()[0] < 0.5]
    if not ladder:
        # upper layers
        for k, (u, z, r, sv) in enumerate(((0.3, 0.75, 0.06, "st"), (0.85, 0.8, 0.07, "cw"), (0.55, 1.45, 0.05, "ia"),
                                           (0.25, 1.75, 0.08, "rw"), (0.95, 1.9, 0.06, "fw"))):
            c.pipe("upper%d" % k, [(u, -0.1, z), (u, Lv + 0.1, z)], r, sv, flange_every=2.0)
    else:
        # a packed chase: three more layers over the full width (they screen the station from the hatches), all
        # rising over the station so that a robot standing there has 1.35 m of headroom
        layers = [(0.71, 0.065), (0.94, 0.07), (1.17, 0.06)]
        kk = 0
        for j, (z, r) in enumerate(layers):
            n2 = max(3, int(round((Lu - 0.1) / (2.6 * r + 0.16))))
            for i in range(n2):
                u = 0.12 + r + (i + 0.5 * (j % 2)) * (Lu - 0.24 - 2 * r) / max(1, n2 - 0.5)
                u = min(u, Lu - r - 0.04)
                sv = ("st", "ia", "cw", "fw", "dr", "rw")[(i + 2 * j) % 6]
                zt = 1.62 + 0.11 * ((i + j) % 4) + 0.03 * j
                pts = [(u, -0.1, z), (u, vs0 - 0.3 - 0.04 * j, z), (u, vs0 - 0.1, zt), (u, vs1 + 0.1, zt),
                       (u, vs1 + 0.3 + 0.04 * j, z), (u, Lv + 0.1, z)]
                c.pipe("mid%d_%d" % (j, i), pts, r, sv, flange_every=2.0)
                kk += 1
        for k, (u, z, r, sv) in enumerate(((0.55, 1.95, 0.05, "ia"), (0.25, 2.05, 0.08, "rw"),
                                           (Lu - 0.3, 2.0, 0.06, "fw"))):
            c.pipe("upper%d" % k, [(u, -0.1, z), (u, Lv + 0.1, z)], r, sv, flange_every=2.0)
    c.box("tray", 0.05, -0.1, 2.15, 0.45, Lv + 0.1, 2.2, "tray")
    # floor conduits across the chase (free class)
    nfc = int(g.integers(1, 3))
    for k in range(nfc):
        vv = uniform(g, 0.6, Lv - 0.6)
        if vs0 - 0.2 < vv < vs1 + 0.2:
            vv = vs1 + 0.5 if vs1 + 0.5 < Lv - 0.4 else vs0 - 0.5
        h = sample_size(c.cfg, g, "floor_pipe", "free")
        p = c.cyl("conduit%d" % k, (0.0, vv, h / 2), (Lu, vv, h / 2), h / 2, "pipe_dr")
        e = c.element("floor_pipe", "free", {"diameter": h}, [0.0, vv - h / 2, Lu, vv + h / 2],
                      note="conduit in the pipe chase")
        c.W.tag(e, [p])
    e = c.element("low_overhead", "needs_crouch", {"clearance": clr}, [0.0, 0.0, Lu, vs0 - 0.05],
                  note="pipe nest (chase, part 1)")
    e2 = c.element("low_overhead", "needs_crouch", {"clearance": clr}, [0.0, vs1 + 0.05, Lu, Lv],
                   note="pipe nest (chase, part 2)")
    c.W.tag(e, [p for p in elem_prims if c.f.to_local(*p.pos[:2])[1] < vs0])
    c.W.tag(e2, [p for p in elem_prims if c.f.to_local(*p.pos[:2])[1] >= vs0])
    # the station: a gauge on the side wall, a lever valve on a short branch, a survey point on the floor
    side_u = Lu - 0.04
    zg = uniform(g, 0.45, 0.7)
    c.box("station/plate", side_u - 0.02, vs0 + 0.2, zg - 0.25, side_u, vs0 + 0.6, zg + 0.2, "panel_grey")
    c.cyl("station/stub", (Lu - 0.06, vs0 + 0.4, zg), (Lu - 0.06, vs0 + 0.4, zg - 0.3), 0.012, "steel", group=1)
    c.slot("gauge", side_u - 0.07, vs0 + 0.4, zg, -1, 0, "pipe chase valve station, side wall", "chase header",
           "station", case_hint=100)
    zl = uniform(g, 0.55, 0.8)
    c.pipe("station/branch", [(Lu - 0.05, vs0 + 0.95, zl), (Lu - 0.45, vs0 + 0.95, zl), (Lu - 0.45, vs0 + 0.95, 1.05)],
           0.03, "cw")
    c.slot("valve_lever", Lu - 0.3, vs0 + 0.95, zl, 0, -1, "pipe chase valve station, drain branch",
           "chase drain", "station", pipe={"axis": c.wvec(1, 0), "r": 0.03})
    c.slot("dose_point", 0.55, vs0 + 0.6, 0.0, 0, 1, "pipe chase valve station, floor", "survey point", "station")
    c.light(0.6, 0.5 * (vs0 + vs1), "dark")
    c.box("station/lamp", 0.3, vs0 + 0.4, 2.1, 0.9, vs0 + 0.55, 2.15, "lamp_dim", group=2)
    c.W.lay.params["chase_station"] = [c.f.w(0, vs0)[:2], c.f.w(Lu, vs1)[:2]]
    c.view("chase_inside", "inside the pipe chase (crouched head height)", 0.62, 0.35, 0.62, vs0 + 0.6, 0.35,
           h=float(c.cfg["modes"]["crouch_walk"]["head_height"]), fovy=80.0)
    c.W.lay.params["chase_station_local"] = [vs0, vs1]
    if ladder:
        # mount spots for the ladder (slot B-low): both long walls of the station, facing across the chase; a pot
        # with a level gauge can also stand there facing along the chase
        for wall in ("left", "right"):
            c.W.spot(c, wall, vs0 + 0.45, vs1 - 0.45, "chase_station", orients=("out", "along+", "along-"))


# ============================================================================================ annex
def f_annex(c: Ctx):
    g = c.g
    Lu, Lv = c.Lu, c.Lv
    # a temporary filtration skid on a plinth along the back wall
    u0, u1 = 0.5, min(Lu - 0.5, 3.2)
    v0, v1 = Lv - 1.3, Lv - 0.25
    c.box("skid/plinth", u0, v0, 0.0, u1, v1, 0.12, "concrete_plinth")
    c.box("skid/frame", u0 + 0.05, v0 + 0.05, 0.12, u1 - 0.05, v1 - 0.05, 0.2, "steel_dark")
    c.vcyl("skid/tank", u0 + 0.5, 0.5 * (v0 + v1), 0.2, 1.4, 0.4, "tank_grey")
    c.vcyl("skid/filter1", u0 + 1.3, 0.5 * (v0 + v1), 0.2, 1.3, 0.16, "steel")
    c.vcyl("skid/filter2", u0 + 1.75, 0.5 * (v0 + v1), 0.2, 1.3, 0.16, "steel")
    c.box("skid/pump", u1 - 0.75, v0 + 0.15, 0.2, u1 - 0.15, v1 - 0.15, 0.6, "pump_blue")
    c.pipe("skid/hdr", [(u0 + 0.5, 0.5 * (v0 + v1), 1.0), (u0 + 0.9, 0.5 * (v0 + v1), 1.0),
                        (u0 + 0.9, 0.5 * (v0 + v1), 1.35), (u1 - 0.45, 0.5 * (v0 + v1), 1.35),
                        (u1 - 0.45, 0.5 * (v0 + v1), 0.6)], 0.035, "rw")
    c.pipe("skid/out", [(u1 - 0.3, v1 - 0.3, 0.45), (u1 - 0.3, Lv + 0.1, 0.45)], 0.035, "rw")
    c.take([u0, v0, u1, v1])
    c.equip("skid", "FILTRATION SKID (TEMPORARY)", [u0, v0, u1, v1], 1.4, on_drawing=False)
    # a control box with lamps on the skid end, a gauge on the filter header, a valve on the outlet
    c.box("skid/ctrl", u1 - 0.6, v0 - 0.02, 0.9, u1 - 0.1, v0 + 0.15, 1.5, "panel_grey")
    c.slot("lamps", u1 - 0.35, v0 - 0.03, 1.25, 0, -1, "filtration skid control box", "skid", "skid")
    c.cyl("skid/gstub", (u0 + 1.52, 0.5 * (v0 + v1), 1.3), (u0 + 1.52, v0 - 0.05, 1.3), 0.012, "steel", group=1)
    c.slot("gauge", u0 + 1.52, v0 - 0.1, 1.3, 0, -1, "filtration skid, filter header (dP)", "skid filters",
           "skid", case_hint=100)
    # drums and boxes along a side wall
    pallet(c, "pal0", 0.75, 1.6 if Lv > 3.5 else 1.2, 2, along_u=False)
    c.box("crate", Lu - 1.0, 0.9, 0.0, Lu - 0.2, 1.6, 0.7, "crate_wood")
    c.take([Lu - 1.0, 0.9, Lu - 0.2, 1.6])
    c.lights_grid(1, 1, "dim")


# ============================================================================================ decor rooms
def f_decor(c: Ctx):
    k = c.room.kind
    {"change_room": _change_room, "sample_lab": _sample_lab, "store": _store, "workshop": _workshop,
     "hvac_room": _hvac, "decon_room": _decon, "office": _office}.get(k, _store)(c)
    c.lights_grid(level=c.room.light)


def _along_walls(c: Ctx, depth, fn, walls=("back", "left", "right"), margin=0.45, step=None):
    for wall in walls:
        for a, b in c.wall_free_spans(wall, margin):
            if b - a < 0.6:
                continue
            fn(wall, a, b, depth)


def _wall_rect(c, wall, a, b, depth):
    if wall == "back":
        return [a, c.Lv - depth, b, c.Lv]
    if wall == "front":
        return [a, 0.0, b, depth]
    if wall == "left":
        return [0.0, a, depth, b]
    return [c.Lu - depth, a, c.Lu, b]


def _face_of(wall):
    return {"back": (0, -1), "front": (0, 1), "left": (1, 0), "right": (-1, 0)}[wall]


def _islands(c: Ctx, depth, length, fn, margin=1.05, max_n=4, pad=0.8):
    """Place free-standing items (rects depth x length, long side along u) on a 0.25 m search grid across the
    middle of a room, keeping `pad` (>= the free width of the footprint) to everything placed and `margin` to the
    walls; rows are separated by aisles."""
    out = []
    k = 0
    v = margin
    while v + depth <= c.Lv - margin + 1e-6 and k < max_n:
        u = margin
        placed_row = False
        while u + length <= c.Lu - margin + 1e-6 and k < max_n:
            r = [u, v, u + length, v + depth]
            if c.free(r, pad=pad, keep_pad=0.1):
                fn(k, r)
                c.take(r)
                out.append(r)
                k += 1
                placed_row = True
                u += length + margin
            else:
                u += 0.25
        v += (depth + margin) if placed_row else 0.25
    return out


def _change_room(c: Ctx):
    def lockers(wall, a, b, depth):
        n = int((b - a - 0.1) // 0.42)
        for i in range(n):
            s0 = a + 0.05 + i * 0.42
            r = _wall_rect(c, wall, s0, s0 + 0.4, depth)
            if c.free(r):
                cabinet(c, "locker_%s%d" % (wall, i), *r, 1.85, mat="locker_blue", face=_face_of(wall),
                        plinth=False)
    _along_walls(c, 0.5, lockers, ("back", "left", "right"))
    _islands(c, 0.36, min(2.2, c.Lu - 2.2), lambda k, r: bench(c, "bench%d" % k, *r), margin=1.0)
    r = [0.1, 0.1, 0.7, 0.45]
    if c.free(r):
        c.box("bootrack", *r[:2], 0.0, *r[2:], 0.35, "rack_blue")
        c.take(r)


def _sample_lab(c: Ctx):
    g = c.g

    def benches(wall, a, b, depth):
        r = _wall_rect(c, wall, a + 0.05, b - 0.05, depth)
        if c.free(r):
            c.box("bench_%s" % wall, *r[:2], 0.0, *r[2:], 0.9, "lab_bench")
            c.take(r)
            n = int(uniform(g, 2, 5))
            for i in range(n):
                s_ = a + (i + 0.5) * (b - a) / n
                rr = _wall_rect(c, wall, s_ - 0.1, s_ + 0.1, depth * 0.6)
                c.box("bench_%s/item%d" % (wall, i), rr[0], rr[1], 0.9, rr[2], rr[3], 0.9 + uniform(g, 0.08, 0.3),
                      choice(g, ["glass", "box_grey", "screen", "box_blue"]), group=1)
    r = [c.Lu - 1.3, c.Lv - 0.85, c.Lu - 0.05, c.Lv]
    if c.free(r):
        c.box("hood", *r[:2], 0.0, *r[2:], 2.2, "cabinet_white")
        c.box("hood/sash", r[0] + 0.1, r[1] - 0.01, 0.95, r[2] - 0.1, r[1], 1.5, "glass", group=1)
        c.take(r)
        c.W.sources_candidates.append({"kind": "process", "sub": "sample_store", "room": c.rid, "label": "FUME HOOD",
                                       "pos": c.f.w(0.5 * (r[0] + r[2]), 0.5 * (r[1] + r[3]), 1.0), "inside_r": 0.4,
                                       "text": "sample store in the fume hood (inventory)"})
    _along_walls(c, 0.7, benches, ("back", "left", "right"))

    def island(k, r):
        c.box("island%d" % k, *r[:2], 0.0, *r[2:], 0.9, "lab_bench")
        c.box("island%d/shelf" % k, r[0] + 0.1, 0.5 * (r[1] + r[3]) - 0.1, 0.9, r[2] - 0.1, 0.5 * (r[1] + r[3]) + 0.1,
              1.5, "galv", group=1)
        c.take(r)
    _islands(c, 1.2, min(2.4, c.Lu - 2.4), island, margin=1.05, max_n=2)


def _store(c: Ctx):
    g = c.g

    def racks(wall, a, b, depth):
        s_ = a + 0.05
        i = 0
        while s_ + 1.0 <= b:
            L = min(uniform(g, 1.2, 2.0), b - s_)
            r = _wall_rect(c, wall, s_, s_ + L, depth)
            if c.free(r):
                shelf_rack(c, "rack_%s%d" % (wall, i), *r, h=uniform(g, 1.8, 2.1), levels=4,
                           mat=choice(g, ["rack_blue", "rack_orange"]))
            s_ += L + 0.1
            i += 1
    _along_walls(c, 0.5, racks, ("back", "left", "right"))

    def island(k, r):
        if k % 2 == 0:
            shelf_rack(c, "isl%d" % k, *r, h=1.8, levels=4, mat="rack_orange")
        else:
            pallet(c, "pal%d" % k, 0.5 * (r[0] + r[2]), 0.5 * (r[1] + r[3]), 0, drums=False)
            c.box("pal%d/load" % k, r[0] + 0.1, r[1] + 0.05, 0.14, r[2] - 0.1, r[3] - 0.05, 0.14 + uniform(g, 0.5, 1.1),
                  "carton")
    _islands(c, 1.0, 1.4, island, margin=1.0, max_n=4)
    r = [0.05, 0.05, 0.75, 0.55]
    if c.free(r):
        c.box("cage", *r[:2], 0.0, *r[2:], 1.4, "galv", group=0)
        for k in range(3):
            c.vcyl("cyl%d" % k, 0.2 + 0.2 * k, 0.3, 0.0, 1.35, 0.08, choice(g, ["gas_green", "pipe_ia", "red_fire"]),
                   group=1)
        c.take(r)


def _workshop(c: Ctx):
    g = c.g
    r = [0.5, c.Lv - 0.8, min(c.Lu - 0.5, 3.2), c.Lv]
    if c.free(r):
        c.box("workbench", *r[:2], 0.0, *r[2:], 0.9, "bench_wood")
        c.box("workbench/vice", r[0] + 0.3, r[1] + 0.1, 0.9, r[0] + 0.5, r[1] + 0.3, 1.05, "steel_dark", group=1)
        c.box("workbench/board", r[0], r[3] - 0.04, 1.1, r[2], r[3], 1.9, "pegboard" if False else "panel_grey",
              group=1)
        c.take(r)
    r = [c.Lu - 1.0, c.Lv - 0.6, c.Lu - 0.05, c.Lv]
    if c.free(r):
        cabinet(c, "toolcab", *r, 1.2, mat="red_fire", face=(0, -1))

    def stock(wall, a, b, depth):
        if b - a < 1.5:
            return
        r = _wall_rect(c, wall, a + 0.1, min(b, a + 2.6), depth)
        if c.free(r):
            shelf_rack(c, "stock_%s" % wall, *r, h=1.4, levels=3, fill=0.5, mat="rack_orange")
    _along_walls(c, 0.5, stock, ("left", "right"))

    def machine(k, r):
        if k == 0:
            mb = "%s/lathe" % c.body
            c.box("lathe/bed", r[0], r[1] + 0.1, 0.0, r[2], r[3] - 0.1, 0.85, "machine_green", body=mb)
            c.box("lathe/head", r[0], r[1], 0.85, r[0] + 0.5, r[3], 1.25, "machine_green", body=mb)
            c.cyl("lathe/chuck", (r[0] + 0.5, 0.5 * (r[1] + r[3]), 1.05), (r[0] + 0.6, 0.5 * (r[1] + r[3]), 1.05),
                  0.12, "steel", body=mb)
            c.box("lathe/tail", r[2] - 0.3, r[1] + 0.15, 0.85, r[2] - 0.05, r[3] - 0.15, 1.15, "machine_green", body=mb)
        else:
            c.box("cart%d" % k, r[0], r[1], 0.0, r[2], r[3], 0.85, "cabinet")
            c.vcyl("cart%d/bottle" % k, r[2] - 0.15, 0.5 * (r[1] + r[3]), 0.85, 1.5, 0.09, "gas_green")
        c.take(r)
    _islands(c, 0.7, 1.8, machine, margin=1.05, max_n=2)
    r = [0.05, 1.6, 0.6, 2.2]
    if c.free(r):
        c.box("drillpress/base", *r[:2], 0.0, *r[2:], 0.1, "steel_dark")
        c.vcyl("drillpress/col", 0.3, 1.9, 0.1, 1.6, 0.05, "steel")
        c.box("drillpress/head", 0.15, 1.75, 1.3, 0.55, 2.1, 1.6, "machine_green")
        c.take(r)


def _hvac(c: Ctx):
    g = c.g
    du = min(c.Lu - 1.0, uniform(g, 2.8, 3.8))
    u0 = 0.5 if c.free([0.5, c.Lv - 1.4, 0.5 + du, c.Lv - 0.2]) else max(0.5, c.Lu - du - 0.5)
    u1 = u0 + du
    v0, v1 = c.Lv - 1.4, c.Lv - 0.25
    if not c.free([u0, v0, u1, v1]):
        return _store(c)
    c.box("ahu/plinth", u0, v0, 0.0, u1, v1, 0.15, "concrete_plinth")
    c.box("ahu/body", u0 + 0.05, v0 + 0.05, 0.15, u1 - 0.05, v1 - 0.05, 1.75, "ahu_grey")
    for i in range(int(du // 0.9)):
        uu = u0 + 0.2 + i * 0.9
        c.box("ahu/door%d" % i, uu, v0 + 0.03, 0.3, uu + 0.7, v0 + 0.05, 1.6, "ahu_panel", group=1)
    c.box("ahu/duct", u0 + 0.3, v0 + 0.3, 1.75, u0 + 1.2, v1 - 0.2, 2.6, "galv")
    c.box("ahu/duct2", u1 - 1.1, v0 + 0.3, 1.75, u1 - 0.3, v1 - 0.2, 2.6, "galv")
    c.take([u0, v0, u1, v1])
    c.equip("ahu", "AIR HANDLING UNIT", [u0, v0, u1, v1], 1.75)
    c.W.sources_candidates.append({"kind": "process", "sub": "hepa_filters", "room": c.rid, "label": "AHU",
                                   "pos": c.f.w(u0 + 0.5 * du, 0.5 * (v0 + v1), 1.0), "inside_r": 0.5,
                                   "text": "HEPA filter bank of the air handling unit (inventory)"})
    c.slot("gauge", u0 + 0.5 * du, v0 + 0.02, 1.35, 0, -1, "AHU filter dP", "ventilation", "ahu", case_hint=100,
           decoy_only=True)
    c.box("duct_x", -0.1, 0.4 * c.Lv, 2.25, c.Lu + 0.1, 0.4 * c.Lv + 0.5, 2.55, "galv", group=2)

    def filters(wall, a, b, depth):
        n = min(4, int((b - a) // 0.7))
        for i in range(n):
            r = _wall_rect(c, wall, a + 0.1 + 0.7 * i, a + 0.75 + 0.7 * i, depth)
            if c.free(r):
                cabinet(c, "filter_%s%d" % (wall, i), *r, 1.8, mat="ahu_panel", face=_face_of(wall), plinth=True)
    _along_walls(c, 0.65, filters, ("left", "right"))

    def fan(k, r):
        cu, cv = 0.5 * (r[0] + r[2]), 0.5 * (r[1] + r[3])
        c.box("fan%d/frame" % k, *r[:2], 0.0, *r[2:], 0.25, "steel_dark")
        c.cyl("fan%d/scroll" % k, (r[0] + 0.1, cv, 0.75), (r[2] - 0.1, cv, 0.75), 0.48, "ahu_grey")
        c.box("fan%d/motor" % k, cu - 0.2, r[1], 0.25, cu + 0.2, r[1] + 0.35, 0.55, "motor_blue")
        c.box("fan%d/duct" % k, cu - 0.3, cv - 0.3, 1.2, cu + 0.3, cv + 0.3, 2.6, "galv")
        c.take(r)
    _islands(c, 1.1, 1.4, fan, margin=1.1, max_n=1)


def _decon(c: Ctx):
    g = c.g
    r = [c.Lu - 1.2, c.Lv - 1.2, c.Lu, c.Lv]
    if c.free(r):
        c.box("shower/wall1", r[0], r[1], 0.0, r[0] + 0.06, r[3], 2.1, "cabinet_white")
        c.box("shower/tray", r[0] + 0.06, r[1], 0.0, r[2], r[3], 0.12, "cabinet_white")
        c.box("shower/wall2", r[0], r[1], 0.0, r[2] - 0.6, r[1] + 0.06, 2.1, "cabinet_white")
        c.take(r)
    for k, r in enumerate(([0.05, c.Lv - 0.6, 1.0, c.Lv], [1.1, c.Lv - 0.6, 2.05, c.Lv])):
        if c.free(r):
            c.box("sink%d" % k, *r[:2], 0.0, *r[2:], 0.85, "steel")
            c.box("sink%d/splash" % k, r[0], r[3] - 0.03, 0.85, r[2], r[3], 1.3, "cabinet_white", group=1)
            c.take(r)
            if k == 0:
                c.W.sources_candidates.append({"kind": "process", "sub": "decon_sink", "room": c.rid, "label": "SINK",
                                               "pos": c.f.w(0.5 * (r[0] + r[2]), 0.5 * (r[1] + r[3]), 0.7),
                                               "inside_r": 0.3, "text": "decontamination sink drain (inventory)"})

    def ppe(wall, a, b, depth):
        if b - a < 1.2:
            return
        r = _wall_rect(c, wall, a + 0.1, min(b, a + 1.8), depth)
        if c.free(r):
            shelf_rack(c, "ppe_%s" % wall, *r, h=1.8, levels=4, fill=0.8, mat="rack_blue")
    _along_walls(c, 0.45, ppe, ("left",))
    _islands(c, 1.0, 1.2, lambda k, r: pallet(c, "pal%d" % k, 0.5 * (r[0] + r[2]), 0.5 * (r[1] + r[3]), 2),
             margin=1.05, max_n=1)


def _office(c: Ctx):
    g = c.g
    r = [0.3, c.Lv - 0.8, min(c.Lu - 0.3, 1.9), c.Lv - 0.05]
    if c.free(r):
        c.box("desk", *r[:2], 0.0, *r[2:], 0.75, "bench_wood")
        c.box("desk/screen", r[0] + 0.3, r[3] - 0.25, 0.75, r[0] + 0.8, r[3] - 0.2, 1.1, "screen", group=1)
        c.take(r)
        rc = [r[0] + 0.4, r[1] - 0.55, r[0] + 0.85, r[1] - 0.1]
        if not c.W.lay.params.get("free_sill_rooms") or c.free(rc, keep_pad=0.0):
            # (version 2: the chair too keeps out of an in-room strip's or rack's band; version 1 unchanged)
            c.box("chair", *rc[:2], 0.0, *rc[2:], 0.5, "rubber_black")
            c.take(rc)

    def cabs(wall, a, b, depth):
        s_ = a + 0.05
        i = 0
        while s_ + 0.5 <= b and i < 4:
            rr = _wall_rect(c, wall, s_, s_ + 0.5, depth)
            if c.free(rr):
                cabinet(c, "file_%s%d" % (wall, i), *rr, 1.3, mat="cabinet", face=_face_of(wall), plinth=False)
            s_ += 0.55
            i += 1
    _along_walls(c, 0.45, cabs, ("left", "right"))

    def table(k, r):
        c.box("table%d" % k, *r[:2], 0.0, *r[2:], 0.74, "bench_wood")
        c.take(r)
    _islands(c, 0.9, 1.6, table, margin=1.1, max_n=1)
