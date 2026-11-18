"""Building and topology: a U-shaped corridor spine (two turns) with room bands on both sides, a pipe chase across
the middle band (the low-robot shortcut between the two legs), an airlock vestibule on the west facade, an annex
outside one facade (not on the drawing), and the doors.

Plan in the building frame B (interior x in [0, W], y in [0, H]) first, then shifted into the site frame S
(origin at the deployment pad centre in the airlock, x into the building).

        y=H +---------------------------------------------+-----------+
            |  north band rooms (doors on the north leg)  |    E3     |
            +--D-------D---------D------------D-----------+           |
            |  north leg (C3)                                 bight   |
            +-----+--+---------------+---------+------+--+  (C2)  +---+
            | W   |ch| valve gallery |  ...    |      |  |        |E2 |
            |     |as|  (doors on    |         |      |  |        |   |
            +-----+e +---------------+---------+------+--+        +---+
     airlock|  south leg (C1)                                     |E1 |
            +--D-------D-----------D---------------D--------------+   |
            |  south band rooms                                   |   |
        y=0 +-----------------------------------------------------+---+
           x=0                                                        x=W
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .util import choice, lattice_uniform, rect_h, rect_w, shared_edge, uniform

NAMES = {
    "airlock": "AIRLOCK", "corridor": "CORRIDOR", "pump_room": "PUMP ROOM", "valve_gallery": "VALVE GALLERY",
    "tank_room": "TANK ROOM", "electrical_room": "ELECTRICAL ROOM", "waste_store": "WASTE STORE",
    "pipe_chase": "PIPE CHASE", "annex": "ANNEX", "change_room": "CHANGE ROOM", "sample_lab": "SAMPLE LAB",
    "store": "STORE", "workshop": "WORKSHOP", "hvac_room": "HVAC PLANT", "decon_room": "DECON ROOM",
    "office": "HP OFFICE",
}
DECOR = ("change_room", "sample_lab", "store", "workshop", "hvac_room", "decon_room", "office")
OUT = "OUT"


@dataclass
class Room:
    id: str
    type: str
    rect: list
    band: str
    kind: str = ""            # decor sub-type (change_room, store, ...)
    number: str = ""
    name: str = ""
    front: str = ""           # side of the main door: S, N, W, E (in the final frame)
    light: str = "mid"
    on_drawing: bool = True
    doors: list = field(default_factory=list)
    inner: list = field(default_factory=list)

    def as_dict(self):
        return {"id": self.id, "number": self.number, "type": self.type, "kind": self.kind or self.type,
                "name": self.name, "rect": self.rect, "inner": self.inner, "band": self.band, "front": self.front,
                "light": self.light, "on_drawing": self.on_drawing, "doors": list(self.doors)}


@dataclass
class Wall:
    id: str
    axis: str                 # 'x': the wall lies on the line x = coord (spans y); 'y': on y = coord
    coord: float
    lo: float
    hi: float
    t_neg: float              # extent toward -axis from the line
    t_pos: float
    kind: str                 # ordinary | outer | shield | partition
    rooms: list               # [room on the -side, room on the +side]
    on_drawing: bool = True
    openings: list = field(default_factory=list)   # door ids

    def thickness(self):
        return self.t_neg + self.t_pos

    def as_dict(self):
        return {"id": self.id, "axis": self.axis, "coord": self.coord, "lo": self.lo, "hi": self.hi,
                "t_neg": self.t_neg, "t_pos": self.t_pos, "thickness": self.thickness(), "kind": self.kind,
                "rooms": self.rooms, "on_drawing": self.on_drawing, "openings": list(self.openings)}


@dataclass
class Door:
    id: str
    wall: str
    axis: str
    coord: float
    center: float             # along the wall
    width: float
    rooms: list               # [room on the -side, room on the +side]
    kind: str = "door"        # door | double | fire | opening | chase_mouth | stair | airlock_outer
    state: str = "open"       # open | ajar | closed
    swing_into: str = ""      # room id the leaf swings into
    hinge: int = 1            # +1 hinge at the high end of the opening, -1 at the low end
    on_drawing: bool = True
    sill: str | None = None
    leaf_angle_deg: float = 90.0
    gap: float | None = None
    blocked_by: str | None = None
    label: str = ""

    def xy(self):
        return (self.coord, self.center) if self.axis == "x" else (self.center, self.coord)

    def as_dict(self):
        x, y = self.xy()
        return {"id": self.id, "wall": self.wall, "axis": self.axis, "coord": self.coord, "center": self.center,
                "pos": [x, y], "width": self.width, "rooms": self.rooms, "kind": self.kind, "state": self.state,
                "swing_into": self.swing_into, "hinge": self.hinge, "on_drawing": self.on_drawing,
                "sill": self.sill, "leaf_angle_deg": self.leaf_angle_deg, "gap": self.gap,
                "blocked_by": self.blocked_by, "label": self.label}


@dataclass
class Layout:
    rooms: dict
    walls: list
    doors: list
    W: float
    H: float
    offset: tuple             # S = B + offset (after mirroring)
    mirrored: bool
    params: dict
    building_rect: list       # main rectangle in S
    grid: dict

    def room(self, rid) -> Room:
        return self.rooms[rid]

    def door(self, did) -> Door:
        for d in self.doors:
            if d.id == did:
                return d
        raise KeyError(did)

    def wall(self, wid) -> Wall:
        for w in self.walls:
            if w.id == wid:
                return w
        raise KeyError(wid)

    def walls_between(self, a, b):
        return [w for w in self.walls if set(w.rooms) == {a, b}]

    def bbox(self):
        xs = [r.rect[0] for r in self.rooms.values()] + [r.rect[2] for r in self.rooms.values()]
        ys = [r.rect[1] for r in self.rooms.values()] + [r.rect[3] for r in self.rooms.values()]
        return [min(xs) - 0.5, min(ys) - 0.5, max(xs) + 0.5, max(ys) + 0.5]


# ============================================================================================ band partition
def _partition(g, L, majors, wdec, min_dec=3.0, lat=0.5):
    """Split a band of length L into the `majors` [(type, wmin, wmax)] plus decor rooms of width in wdec.
    Returns [(type, width)] in a random order, or None."""
    widths = [lattice_uniform(g, a, b, lat) for _, a, b in majors]
    R = L - sum(widths)
    # shrink majors toward their minimum if needed
    i = 0
    while R < -1e-9 and i < 200:
        cand = [k for k, (t, a, b) in enumerate(majors) if widths[k] - lat >= a - 1e-9]
        if not cand:
            return None
        k = cand[int(g.integers(len(cand)))]
        widths[k] -= lat
        R += lat
        i += 1
    items = [(t, w) for (t, _, _), w in zip(majors, widths)]
    dmin, dmax = wdec
    if R > 1e-9:
        if R < min_dec - 1e-9:
            # give the rest to the majors that can grow
            while R > 1e-9:
                cand = [k for k, (t, a, b) in enumerate(majors) if items[k][1] + lat <= b + 1e-9]
                if not cand:
                    return None
                k = cand[int(g.integers(len(cand)))]
                items[k] = (items[k][0], items[k][1] + lat)
                R -= lat
        else:
            nmin = max(1, int(math.ceil(R / dmax - 1e-9)))
            nmax = max(nmin, int(math.floor(R / dmin + 1e-9)))
            n = int(g.integers(nmin, nmax + 1))
            # random split of R into n lattice pieces in [dmin, dmax]
            ws = [dmin] * n
            rest = R - dmin * n
            guard = 0
            while rest > 1e-9 and guard < 1000:
                k = int(g.integers(n))
                if ws[k] + lat <= dmax + 1e-9:
                    ws[k] += lat
                    rest -= lat
                guard += 1
            if rest > 1e-9:
                return None
            for w in ws:
                items.append(("decor", w))
    order = list(g.permutation(len(items)))
    return [items[i] for i in order]


# ============================================================================================ main
def build_layout(cfg, rng) -> Layout:
    g = rng("layout")
    B = cfg["building"]
    W, H, cw = float(B["width"]), float(B["depth"]), float(B["corridor_width"])
    lat = float(B["lattice"])
    for _ in range(50):
        Ds = float(choice(g, B["south_band_depth"]))
        Dm = float(choice(g, B["middle_band_depth"]))
        Dn = H - 2 * cw - Ds - Dm
        if B["north_band_depth"][0] - 1e-9 <= Dn <= B["north_band_depth"][1] + 1e-9:
            break
    We = float(choice(g, B["east_block_width"]))
    xb = W - We - cw
    yS0, yS1 = Ds, Ds + cw                 # south leg
    yN0, yN1 = Ds + cw + Dm, Ds + 2 * cw + Dm
    rooms: dict[str, Room] = {}
    R = cfg["rooms"]

    def add(rid, typ, rect, band, kind=""):
        rooms[rid] = Room(id=rid, type=typ, rect=[float(v) for v in rect], band=band, kind=kind)
        return rooms[rid]

    add("C1", "corridor", [0, yS0, xb + cw, yS1], "spine")
    add("C2", "corridor", [xb, yS1, xb + cw, yN0], "spine")
    add("C3", "corridor", [0, yN0, xb + cw, yN1], "spine")
    A = cfg["airlock"]
    yc = 0.5 * (yS0 + yS1)
    add("A1", "airlock", [-A["depth_x"], yc - A["width_y"] / 2, 0, yc + A["width_y"] / 2], "airlock")

    # ---------------------------------------------------------------- east block: three rooms stacked in y
    for _ in range(100):
        y1 = lattice_uniform(g, yS0 + 1.5, min(9.0, yS0 + 3.0), lat)
        y2 = lattice_uniform(g, max(y1 + B["east_room_min"], yN1 - 3.5), yN1 - 1.5, lat)
        if y1 >= 6.0 and y2 - y1 >= B["east_room_min"] and H - y2 >= 5.0 and y1 <= 9.0:
            break
    east = {"E1": [xb + cw, 0, W, y1], "E2": [xb + cw, y1, W, y2], "E3": [xb + cw, y2, W, H]}

    # ---------------------------------------------------------------- assign the major rooms
    tank_slot = "E1" if (rect_h(east["E1"]) >= 7.0 and (g.random() < 0.5 or rect_h(east["E3"]) < 7.0)) else "E3"
    if rect_h(east[tank_slot]) < 7.0:
        tank_slot = "E1" if rect_h(east["E1"]) >= rect_h(east["E3"]) else "E3"
    other_corner = "E3" if tank_slot == "E1" else "E1"
    slots = {"south": [], "north": [], "middle": [], other_corner: [], "E2": []}
    # pump room and waste store: south band or the east slots
    for typ in ("pump_room", "waste_store"):
        cands = ["south", other_corner, "E2"]
        if Dn >= R[typ]["min_depth"]:
            cands.append("north")
        w = []
        for c in cands:
            if c in ("E2", other_corner):
                ok = (not slots[c]) and min(rect_h(east[c]), We) >= R[typ]["min_depth"] - 1e-9 \
                    and max(rect_h(east[c]), We) >= R[typ]["width"][0] - 1.0
                w.append(1.0 if ok else 0.0)
            elif c == "south":
                w.append(1.4 if len(slots["south"]) < 2 else 0.0)
            else:
                w.append(0.6)
        if sum(w) <= 0:
            w = [1.0] + [0.0] * (len(cands) - 1)
        c = choice(g, cands, w)
        slots[c].append(typ)
    # electrical room: north band (preferred), middle band, south band
    ladder_l = str(cfg.get("layout", "draft")).startswith("ladder")
    c = choice(g, ["north", "middle", "south"], [0.6, 0.25 if not ladder_l else 0.0,
                                                 0.15 if len(slots["south"]) < 2 else 0.0])
    slots[c].append("electrical_room")

    # ---------------------------------------------------------------- the bands
    def band_rooms(key, band_rect, majors_types, along="x"):
        L = rect_w(band_rect) if along == "x" else rect_h(band_rect)
        majors = [(t, R[t]["width"][0], R[t]["width"][1]) for t in majors_types]
        for _ in range(60):
            parts = _partition(g, L, majors, R["decor"]["width"], lat=lat)
            if parts:
                return parts
        raise RuntimeError("band partition failed: %s" % key)

    counter = {"n": 0}
    deck = []

    def next_decor(allowed=DECOR):
        for _ in range(3):
            if not deck:
                deck.extend(DECOR[i] for i in g.permutation(len(DECOR)))
            for i, k in enumerate(deck):
                if k in allowed:
                    return deck.pop(i)
            deck.clear()
        return allowed[0]

    def place_band(prefix, band_rect, parts, band):
        x = band_rect[0]
        out = []
        for t, w in parts:
            counter["n"] += 1
            rid = "%s%02d" % (prefix, counter["n"])
            rect = [x, band_rect[1], x + w, band_rect[3]]
            kind = ""
            if t == "decor":
                kind = next_decor()
            out.append(add(rid, t if t != "decor" else "decor", rect, band, kind))
            x += w
        return out

    sparts = band_rooms("south", [0, 0, xb + cw, Ds], slots["south"])
    if sparts and sparts[0][0] == "waste_store" and len(sparts) > 1:
        sparts.append(sparts.pop(0))          # keep the drums away from the airlock
    if str(cfg.get("layout", "draft")).startswith("ladder") and len(sparts) > 2 and sparts[0][1] < 6.5 and \
            sparts[1][0] == "waste_store":
        # ladder: a narrow first room cannot open onto the slot-A stretch and must open into its neighbour
        sparts.append(sparts.pop(1))
    south = place_band("R", [0, 0, xb + cw, Ds], sparts, "south")
    north = place_band("R", [0, yN1, xb + cw, H], band_rooms("north", [0, yN1, xb + cw, H], slots["north"]),
                       "north")
    # middle band: [west room] + chase + the rest (valve gallery always here)
    xm = 0.0
    if g.random() < R["west_room_probability"]:
        ww = lattice_uniform(g, R["west_room_width"][0], R["west_room_width"][1], lat)
        counter["n"] += 1
        add("R%02d" % counter["n"], "decor", [0, yS1, ww, yN0], "middle",
            next_decor(("store", "decon_room", "office")))
        xm = ww
    counter["n"] += 1
    chase_id = "R%02d" % counter["n"]
    cwid = float(R["pipe_chase"]["width"])
    if abs(cwid / lat - round(cwid / lat)) > 1e-6:
        raise ValueError("rooms.pipe_chase.width must be a multiple of building.lattice")
    add(chase_id, "pipe_chase", [xm, yS1, xm + cwid, yN0], "middle")
    xm += cwid
    mid_parts = band_rooms("middle", [xm, yS1, xb, yN0], ["valve_gallery"] + slots["middle"])
    middle = place_band("R", [xm, yS1, xb, yN0], mid_parts, "middle")
    # east block
    for slot, rect in sorted(east.items()):
        counter["n"] += 1
        rid = "R%02d" % counter["n"]
        if slot == tank_slot:
            add(rid, "tank_room", rect, "east")
        elif slots.get(slot):
            add(rid, slots[slot][0], rect, "east")
        else:
            add(rid, "decor", rect, "east", next_decor(("workshop", "hvac_room", "store", "change_room")))

    # ---------------------------------------------------------------- annex outside one facade
    an = R["annex"]["size"]
    cands, wts = [], []
    for r in list(south) + list(north) + [rooms[k] for k in rooms if rooms[k].band == "east"]:
        if r.type in ("waste_store", "pipe_chase", "airlock"):
            continue
        if str(cfg.get("layout", "draft")).startswith("ladder") and (r.type == "tank_room" or min(
                rect_w(r.rect), rect_h(r.rect)) < 4.0):
            continue                          # ladder: the bund's walls carry no door (slot B-kerb); a 4 m facade
        cands.append(r)
        wts.append(0.4 + r.rect[0] / W)
    host = choice(g, cands, wts)
    aw = lattice_uniform(g, an[0][0], an[0][1], lat)
    ad = lattice_uniform(g, an[1][0], an[1][1], lat)
    hr = host.rect
    if host.band == "south":
        side = "S"
    elif host.band == "north":
        side = "N"
    else:
        side = "E"
    ladder_a = str(cfg.get("layout", "draft")).startswith("ladder")
    if side in ("S", "N"):
        lo = max(0.0, hr[0] - 1.0)
        hi = min(W, hr[2] + 1.0) - aw
        if ladder_a:
            # the annex lies against its host's facade only (attached outside one outer-wall room)
            aw = min(aw, hr[2] - hr[0])
            lo, hi = hr[0], hr[2] - aw
            a0 = lattice_uniform(g, lo, hi, lat) if hi > lo else hr[0]
        else:
            a0 = lattice_uniform(g, max(lo, hr[0] + 1.5 - aw), min(hi, hr[2] - 1.5), lat) if hi > lo else hr[0]
        a0 = min(max(a0, 0.0), W - aw)
        rect = [a0, -ad, a0 + aw, 0.0] if side == "S" else [a0, H, a0 + aw, H + ad]
    else:
        lo, hi = max(0.0, hr[1] - 1.0), min(H, hr[3] + 1.0) - aw
        if ladder_a:
            aw = min(aw, hr[3] - hr[1])
            lo, hi = hr[1], hr[3] - aw
            a0 = lattice_uniform(g, lo, hi, lat) if hi > lo else hr[1]
        else:
            a0 = lattice_uniform(g, max(lo, hr[1] + 1.5 - aw), min(hi, hr[3] - 1.5), lat) if hi > lo else hr[1]
        a0 = min(max(a0, 0.0), H - aw)
        rect = [W, a0, W + ad, a0 + aw]
    counter["n"] += 1
    annex = add("R%02d" % counter["n"], "annex", rect, "annex")
    annex.on_drawing = False

    # ---------------------------------------------------------------- mirror, shift to S, numbers, names
    mirrored = bool(g.random() < B["mirror_probability"])
    if mirrored:
        for r in rooms.values():
            y0, y1_ = r.rect[1], r.rect[3]
            r.rect[1], r.rect[3] = H - y1_, H - y0
        yc = H - yc
    ox, oy = float(A["pad_from_inner_wall"]), -yc
    for r in rooms.values():
        r.rect = [r.rect[0] + ox, r.rect[1] + oy, r.rect[2] + ox, r.rect[3] + oy]
    building_rect = [ox, oy, W + ox, H + oy]
    if mirrored:
        building_rect = [ox, -yc + 0.0, W + ox, H - yc]
    # numbering: by band and x
    order = {"airlock": 0, "south": 1, "middle": 2, "north": 3, "east": 4, "annex": 5}
    num = {"south": 101, "middle": 121, "north": 141, "east": 161, "annex": 181}
    for r in sorted(rooms.values(), key=lambda r: (order.get(r.band, 9), r.rect[0], r.rect[1])):
        if r.type == "corridor":
            r.number = {"C1": "C01", "C2": "C02", "C3": "C03"}[r.id]
        elif r.type == "airlock":
            r.number = "100"
        else:
            r.number = str(num[r.band])
            num[r.band] += 1
        base = NAMES[r.kind] if r.type == "decor" else NAMES[r.type]
        r.name = base
        r.light = R[r.type]["light"] if r.type in R and isinstance(R[r.type], dict) and "light" in R[r.type] \
            else R["decor"]["light"]
        if r.kind in ("store", "hvac_room"):
            r.light = "dim"
    params = {"Ds": Ds, "Dm": Dm, "Dn": Dn, "We": We, "xb": xb, "W": W, "H": H, "cw": cw, "mirrored": mirrored,
              "tank_slot": tank_slot, "chase": chase_id, "annex": annex.id, "annex_host": host.id,
              "annex_side": side}
    ladder = str(cfg.get("layout", "draft")).startswith("ladder")
    params["ladder"] = ladder
    if ladder:
        # slot A: the stretch of the south leg between the airlock door wall and the corridor sill (the
        # site-wide rule); no door opens onto it, and the chase mouth lies beyond the sill
        a0, a1 = cfg["ladder"]["corridor"]["a_stretch"]
        g2 = rng("ladder_layout")
        params["corridor_sill_x"] = round((ox + uniform(g2, float(a0), float(a1))) / 0.05) * 0.05
        if str(cfg["ladder"].get("lighting", "")) == "bright":
            for r in rooms.values():
                r.light = "bright"

    lay = Layout(rooms=rooms, walls=[], doors=[], W=W, H=H, offset=(ox, oy), mirrored=mirrored, params=params,
                 building_rect=building_rect, grid={})
    _build_walls(lay, cfg)
    _build_doors(lay, cfg, rng("doors"))
    _grid(lay, cfg)
    _inner_rects(lay)
    return lay


# ============================================================================================ walls
def _wall_kind(ra, rb):
    if ra is None or rb is None:
        return "outer"
    ts = {ra.type, rb.type}
    if "waste_store" in ts:
        return "shield"
    return "ordinary"


def _build_walls(lay: Layout, cfg):
    B = cfg["building"]
    t = float(B["wall_thickness"])
    to = float(B["outer_wall_thickness"])
    g_sh = float(B["shield_wall_thickness"][0]), float(B["shield_wall_thickness"][1])
    rooms = list(lay.rooms.values())
    segs = {}   # (axis, coord, lo, hi) -> (room_neg, room_pos)
    for r in rooms:
        x0, y0, x1, y1 = r.rect
        edges = [("y", y0, x0, x1, "neg"), ("y", y1, x0, x1, "pos"), ("x", x0, y0, y1, "neg"),
                 ("x", x1, y0, y1, "pos")]
        for axis, c, lo, hi, side in edges:
            # pieces of this edge shared with other rooms
            cuts = []
            for o in rooms:
                if o is r:
                    continue
                se = shared_edge(r.rect, o.rect)
                if se and se[0] == axis and abs(se[1] - c) < 1e-6:
                    cuts.append((se[2], se[3], o.id))
            pts = sorted({lo, hi} | {a for a, _, _ in cuts} | {b for _, b, _ in cuts})
            for a, b in zip(pts[:-1], pts[1:]):
                if b - a < 1e-6:
                    continue
                m = 0.5 * (a + b)
                other = OUT
                for ca, cb, oid in cuts:
                    if ca - 1e-9 <= m <= cb + 1e-9:
                        other = oid
                # the room on the -side of the line and the one on the +side
                neg, pos = (other, r.id) if side == "neg" else (r.id, other)
                key = (axis, round(c, 4), round(a, 4), round(b, 4))
                segs[key] = (neg, pos)
    walls = []
    k = 0
    for key in sorted(segs.keys()):
        axis, c, a, b = key
        neg, pos = segs[key]
        rn = lay.rooms.get(neg)
        rp = lay.rooms.get(pos)
        if rn is not None and rp is not None and rn.type == "corridor" and rp.type == "corridor":
            continue
        kind = _wall_kind(rn, rp)
        # the wall between an undrawn annex and its host is the host's original facade (the annex was built against
        # it later): an outer wall like every other, its extra thickness on the annex side
        party = rn is not None and rp is not None and (rn.type == "annex") != (rp.type == "annex")
        if party:
            kind = "outer"
        if kind == "outer":
            # the extra thickness goes outside
            if rn is None or (party and rn.type == "annex"):
                tn, tp = to - t / 2, t / 2
            else:
                tn, tp = t / 2, to - t / 2
            if not party and ((rn is not None and rn.type == "annex") or (rp is not None and rp.type == "annex")):
                tn, tp = t / 2, t / 2          # the annex's own (later, lighter) outer walls
        elif kind == "shield":
            ts = round(0.5 * (g_sh[0] + g_sh[1]) / 0.05) * 0.05
            if rn is not None and rn.type == "waste_store":
                tn, tp = ts - t / 2, t / 2
            else:
                tn, tp = t / 2, ts - t / 2
        else:
            tn = tp = t / 2
        k += 1
        walls.append(Wall(id="W%03d" % k, axis=axis, coord=c, lo=a, hi=b, t_neg=tn, t_pos=tp, kind=kind,
                          rooms=[neg, pos]))
    # merge collinear neighbours with the same rooms and kind
    walls.sort(key=lambda w: (w.axis, w.coord, w.lo))
    merged = []
    for w in walls:
        if merged:
            p = merged[-1]
            if p.axis == w.axis and abs(p.coord - w.coord) < 1e-6 and abs(p.hi - w.lo) < 1e-6 and \
                    p.rooms == w.rooms and p.kind == w.kind and abs(p.t_neg - w.t_neg) < 1e-6:
                p.hi = w.hi
                continue
        merged.append(w)
    for i, w in enumerate(merged):
        w.id = "W%03d" % (i + 1)
    lay.walls = merged


def drawing_walls(lay: Layout, cfg) -> list:
    """The walls of the original drawing: the walls the building has with its drawn rooms alone (the same rules as
    _build_walls). An undrawn room (the annex) leaves no trace in them: the facade it stands against is drawn as
    one outer wall of the host's full length with the outer-wall thickness and offset of every other one (the
    first version drew the party wall as its own stretch, 0.25 m thick with offset 0, which located the annex)."""
    drawn = {k: r for k, r in lay.rooms.items() if r.on_drawing}
    L2 = Layout(rooms=drawn, walls=[], doors=[], W=lay.W, H=lay.H, offset=lay.offset, mirrored=lay.mirrored,
                params=lay.params, building_rect=lay.building_rect, grid=lay.grid)
    _build_walls(L2, cfg)
    return L2.walls


def wall_between(lay, a, b):
    ws = [w for w in lay.walls if set(w.rooms) == {a, b}]
    ws.sort(key=lambda w: -(w.hi - w.lo))
    return ws


# ============================================================================================ doors
def _door_on(lay, g, wall: Wall, width, prefer="any", margin=0.45, avoid=()):
    lo = wall.lo + margin + width / 2
    hi = wall.hi - margin - width / 2
    if hi < lo - 1e-9:
        return None
    hi = max(hi, lo)
    if prefer == "end":
        span = min(1.2, hi - lo)
        if g.random() < 0.5:
            c = uniform(g, lo, lo + span)
        else:
            c = uniform(g, hi - span, hi)
    elif prefer == "center":
        m = 0.5 * (lo + hi)
        c = uniform(g, m - 0.25 * (hi - lo), m + 0.25 * (hi - lo))
    else:
        c = uniform(g, lo, hi)
    c = round(c / 0.05) * 0.05
    for a, b in avoid:
        if a - width / 2 - 0.3 < c < b + width / 2 + 0.3:
            return None
    return c


def add_door(lay, wall: Wall, center, width, kind="door", into=None, g=None, label="") -> Door:
    did = "D%02d" % (len(lay.doors) + 1)
    neg, pos = wall.rooms
    if into is None:
        into = pos if lay.rooms.get(pos) is not None and lay.rooms[pos].type != "corridor" else neg
    d = Door(id=did, wall=wall.id, axis=wall.axis, coord=wall.coord, center=float(center), width=float(width),
             rooms=[neg, pos], kind=kind, swing_into=into, hinge=(1 if (g is None or g.random() < 0.5) else -1),
             label=label)
    lay.doors.append(d)
    wall.openings.append(did)
    for rid in (neg, pos):
        if rid in lay.rooms:
            lay.rooms[rid].doors.append(did)
    return d


def _side_of(room: Room, wall: Wall):
    if wall.axis == "y":
        return "S" if abs(wall.coord - room.rect[1]) < 1e-6 else "N"
    return "W" if abs(wall.coord - room.rect[0]) < 1e-6 else "E"


def _build_doors(lay: Layout, cfg, g):
    R = cfg["rooms"]
    D = cfg["doors"]
    rooms = lay.rooms
    corr = ("C1", "C2", "C3")

    def width_for(r):
        wr = R[r.type]["door_width"] if r.type in R and "door_width" in R[r.type] else D["width"]
        return round(uniform(g, wr[0], wr[1]) / 0.05) * 0.05

    prefer = {"pump_room": "end", "tank_room": "end", "waste_store": "end", "valve_gallery": "end",
              "electrical_room": "end"}
    # airlock: inner door (to the corridor) and the outer door (closed)
    a = rooms["A1"]
    w_in = wall_between(lay, "A1", "C1")[0]
    d = add_door(lay, w_in, 0.5 * (w_in.lo + w_in.hi), cfg["airlock"]["inner_door_width"], into="A1", g=g,
                 label="AIRLOCK INNER DOOR")
    a.front = _side_of(a, w_in)
    w_out = [w for w in lay.walls if "A1" in w.rooms and OUT in w.rooms and w.axis == "x"
             and abs(w.coord - a.rect[0]) < 1e-6][0]
    d = add_door(lay, w_out, 0.5 * (w_out.lo + w_out.hi), cfg["airlock"]["outer_door_width"], kind="airlock_outer",
                 into="A1", g=g, label="AIRLOCK OUTER DOOR")
    d.state = "closed"
    # stair door at the far west end of the north leg
    if cfg["building"]["stair_door"]:
        ws = [w for w in lay.walls if "C3" in w.rooms and OUT in w.rooms and w.axis == "x"]
        if ws:
            w = min(ws, key=lambda w: w.coord)
            d = add_door(lay, w, 0.5 * (w.lo + w.hi), 1.0, kind="stair", into="C3", g=g, label="STAIR 2")
            d.state = "closed"
    # every room to the corridor
    for r in sorted(rooms.values(), key=lambda r: r.id):
        if r.type in ("corridor", "airlock", "annex"):
            continue
        cands = []
        for c in corr:
            for w in wall_between(lay, r.id, c):
                cands.append((c, w))
        if not cands:
            # landlocked (should not happen): connect to a neighbour
            continue
        if r.type == "pipe_chase":
            for c, w in cands:
                if c in ("C1", "C3") and w.hi - w.lo >= 1.0:
                    add_door(lay, w, 0.5 * (w.lo + w.hi), R["pipe_chase"]["mouth_width"], kind="chase_mouth",
                             into=r.id, g=g, label="CHASE")
            r.front = _side_of(r, cands[0][1])
            continue
        legs = sorted({c for c, _ in cands})
        want = []
        xs_l = lay.params.get("corridor_sill_x") if lay.params.get("ladder") else None
        west_of_sill = xs_l is not None and r.band == "middle" and r.rect[0] < xs_l + 1.0
        if r.type == "valve_gallery" and "C1" in legs and "C3" in legs:
            want = ["C1", "C3"]
        elif west_of_sill and "C3" in legs:
            want = ["C3"]                                   # ladder: the west room opens onto the north leg
            g.random()
            g.random()
        elif r.band == "middle" and "C1" in legs and "C3" in legs:
            first = choice(g, ["C1", "C3"])
            want = [first]
            if g.random() < R["second_door_probability"] and r.type not in ("waste_store",):
                want.append("C3" if first == "C1" else "C1")
        else:
            # east block rooms: prefer the bight; others: the corridor they touch
            ws_len = {c: sum(w.hi - w.lo for cc, w in cands if cc == c) for c in legs}
            okc = [c for c in legs if ws_len[c] >= 1.9]
            if not okc:
                okc = legs
            if r.band == "east" and "C2" in okc and g.random() < 0.75:
                want = ["C2"]
            else:
                want = [choice(g, okc)]
        flip = bool(g.random() < 0.5)
        for k, c in enumerate(want):
            ws = [w for cc, w in cands if cc == c]
            ws.sort(key=lambda w: -(w.hi - w.lo))
            w = ws[0]
            width = width_for(r)
            if r.type == "valve_gallery" and len(want) == 2:
                # doors at opposite ends of the gallery (the through route runs along both aisles)
                low_end = (k == 0) != flip
                lo = w.lo + 0.5 + width / 2
                hi = w.hi - 0.5 - width / 2
                span = min(0.8, hi - lo)
                c0 = uniform(g, lo, lo + span) if low_end else uniform(g, hi - span, hi)
                c0 = round(c0 / 0.05) * 0.05
            else:
                c0 = None
                avoid = ()
                if xs_l is not None and c == "C1":
                    avoid = ((-1e3, xs_l + 0.4),)            # ladder: no door in the slot-A stretch
                for _ in range(20):
                    c0 = _door_on(lay, g, w, width, prefer.get(r.type, "any"), avoid=avoid)
                    if c0 is not None:
                        break
                if c0 is None and avoid:
                    width = 0.9
                    for _ in range(20):
                        c0 = _door_on(lay, g, w, width, "any", margin=0.3, avoid=avoid)
                        if c0 is not None:
                            break
                    if c0 is None:
                        lay.params.setdefault("needs_interconnect", []).append(r.id)
                        continue
                if c0 is None:
                    width = 0.9
                    c0 = 0.5 * (w.lo + w.hi)
            kind = "double" if width >= 1.4 else "door"
            add_door(lay, w, c0, width, kind=kind, into=r.id, g=g)
            if not r.front:
                r.front = _side_of(r, w)
    # interconnecting doors inside a band
    no_ic = ("corridor", "airlock", "annex", "pipe_chase", "waste_store") + \
        (("tank_room",) if lay.params.get("ladder") else ())
    for w in lay.walls:
        ra, rb = rooms.get(w.rooms[0]), rooms.get(w.rooms[1])
        if ra is None or rb is None:
            continue
        if ra.type in no_ic or rb.type in no_ic:
            continue
        if ra.band != rb.band and not {ra.band, rb.band} <= {"south", "east", "north"}:
            continue
        if w.hi - w.lo < 2.5:
            continue
        if g.random() < R["interconnect_probability"]:
            c0 = _door_on(lay, g, w, 0.9, "any", margin=0.8)
            if c0 is not None:
                add_door(lay, w, c0, 0.9, kind="door", into=rb.id, g=g)
    # ladder: a room whose only corridor wall lies in the slot-A stretch opens into its neighbour of the band
    for rid in lay.params.get("needs_interconnect", []):
        r = rooms[rid]
        if any(lay.door(d).kind not in ("airlock_outer", "stair") for d in r.doors):
            continue        # the interconnect pass above already gave it a door (a second one would overlap it)
        best = None
        for w in lay.walls:
            if rid not in w.rooms or w.hi - w.lo < 1.6:
                continue
            o = w.rooms[0] if w.rooms[1] == rid else w.rooms[1]
            ro = rooms.get(o)
            if ro is None or ro.type in ("corridor", "airlock", "annex", "pipe_chase", "waste_store", "tank_room"):
                continue
            if best is None or (w.hi - w.lo) > (best.hi - best.lo):
                best = w
        if best is None:
            raise RuntimeError("ladder: room %s has no door" % rid)
        c0 = _door_on(lay, g, best, 0.9, "any", margin=0.45) or 0.5 * (best.lo + best.hi)
        o = best.rooms[0] if best.rooms[1] == rid else best.rooms[1]
        add_door(lay, best, c0, 0.9, kind="door", into=rid, g=g)
        if not r.front:
            r.front = _side_of(r, best)
    # the annex door (through the host's facade; not on the drawing)
    an = rooms[lay.params["annex"]]
    ws = wall_between(lay, an.id, lay.params["annex_host"])
    if not ws:
        raise RuntimeError("annex not attached")
    w = ws[0]
    width = round(uniform(g, *R["annex"]["door_width"]) / 0.05) * 0.05
    c0 = _door_on(lay, g, w, width, "center", margin=0.4) or 0.5 * (w.lo + w.hi)
    d = add_door(lay, w, c0, width, kind="door", into=an.id, g=g)
    d.on_drawing = False
    an.front = _side_of(an, w)
    # sills: classes per door kind
    for r in rooms.values():
        if not r.front and r.doors:
            d0 = lay.door(r.doors[0])
            r.front = _side_of(r, lay.wall(d0.wall))
    _check_openings(lay)


def _check_openings(lay, gap=0.2):
    """Two openings in one wall must not overlap (nor nearly touch): a grammar error, the sub-seed is redrawn."""
    for w in lay.walls:
        ds = sorted((lay.door(i) for i in w.openings), key=lambda d: d.center)
        for a, b in zip(ds[:-1], ds[1:]):
            if b.center - b.width / 2 < a.center + a.width / 2 + gap:
                raise RuntimeError("doors %s and %s overlap in wall %s" % (a.id, b.id, w.id))


def _inner_rects(lay: Layout):
    """Clear interior of every room: the rectangle minus the wall thickness on each side (max along the side)."""
    for r in lay.rooms.values():
        x0, y0, x1, y1 = r.rect
        e = {"S": 0.0, "N": 0.0, "W": 0.0, "E": 0.0}
        for w in lay.walls:
            if r.id not in w.rooms:
                continue
            on_pos = w.rooms[1] == r.id      # room on the +side of the wall line
            ext = w.t_pos if on_pos else w.t_neg
            side = _side_of(r, w)
            e[side] = max(e[side], ext)
        r.inner = [x0 + e["W"], y0 + e["S"], x1 - e["E"], y1 - e["N"]]


def _grid(lay: Layout, cfg):
    p = float(cfg["building"]["grid_pitch"])
    bx0, by0, bx1, by1 = lay.building_rect
    ox, oy = lay.offset
    # grid lines in the building frame at multiples of the pitch, from the SW corner (after mirroring)
    xs = []
    k = 0
    while bx0 + k * p <= bx1 + 1e-6:
        xs.append({"label": str(k + 1), "x": bx0 + k * p})
        k += 1
    ys = []
    k = 0
    while by0 + k * p <= by1 + 1e-6:
        ys.append({"label": "ABCDEFGH"[k], "y": by0 + k * p})
        k += 1
    lay.grid = {"pitch": p, "x_lines": xs, "y_lines": ys}
