"""The missing drawing sheet (version 2: one sheet of the drawing is missing in every
building).

The drawing has 2 x 2 sheets over `building.outline`, cut at its mid-lines. One of the three sheets other than the
airlock's is missing from the archive copy; its place, seen from the airlock's sheet, is fixed by the candidate number:
along (the airlock's band, far half), across (the other band, the airlock's half), diagonal (the other band,
far half).

What the missing sheet takes ("whole rooms, corridors cut"):
  * every room whose label point (the centre of its rectangle) lies on it, whole: the room, its equipment, its zone,
    and any wall or door that bounds only missing rooms or cut-off corridor parts;
  * the part of a corridor leg on it: the leg is cut across, the cut moved along the leg by up to 1.0 m (a seeded
    offset, then the clear position nearest to it) so that it stays 0.3 m clear of every door opening, hatch and fire
    door; the drawn part keeps its `rooms[]` entry with the drawn rectangle and `continues: true`.
A wall is drawn when it bounds a drawn room or a drawn corridor part (in full and with its doors); so the walls
between the known and the unknown part are on the drawing and a door in them names the missing room's number.

Membership uses half-open sheet intervals (x < mid-line: the west column; y < mid-line: the first row), so a label
point exactly on a mid-line belongs to the east / upper sheet. The annex is on no sheet (it is never drawn); for the
count of listed instruments on the missing sheet it goes with its host room.
"""

from __future__ import annotations

import math

PLACES = ("along", "across", "diagonal")
CORRIDOR_LEGS = ("C1", "C2", "C3")


# ============================================================================================ sheets
def mid_lines(building_rect):
    bx0, by0, bx1, by1 = [float(v) for v in building_rect]
    return 0.5 * (bx0 + bx1), 0.5 * (by0 + by1)


def sheet_rects(building_rect, nx=2, ny=2):
    """[(n, rect)] numbered as the drawing numbers them (prior.json drawing.sheets): row by row from the building's
    lower edge, west to east."""
    bx0, by0, bx1, by1 = [float(v) for v in building_rect]
    out = []
    n = 0
    for j in range(ny):
        for i in range(nx):
            n += 1
            out.append((n, [bx0 + i * (bx1 - bx0) / nx, by0 + j * (by1 - by0) / ny, bx0 + (i + 1) * (bx1 - bx0) / nx,
                            by0 + (j + 1) * (by1 - by0) / ny]))
    return out


def sheet_ij(building_rect, x, y):
    """Column and row (i, j) of the sheet that holds point (x, y) (half-open: a point on a mid-line goes east / up;
    points outside the outline go to the nearest column / row)."""
    xm, ym = mid_lines(building_rect)
    return (0 if x < xm else 1), (0 if y < ym else 1)


def sheet_of(building_rect, x, y):
    i, j = sheet_ij(building_rect, x, y)
    return j * 2 + i + 1


def airlock_ij(building_rect):
    """The airlock's sheet: the one of the deployment pad (the site origin)."""
    return sheet_ij(building_rect, 0.0, 0.0)


def place_ij(building_rect, place):
    ia, ja = airlock_ij(building_rect)
    if place == "along":
        return 1 - ia, ja
    if place == "across":
        return ia, 1 - ja
    if place == "diagonal":
        return 1 - ia, 1 - ja
    raise ValueError("missing sheet place %r (along, across, diagonal)" % (place,))


def place_of_sheet(building_rect, n):
    """along / across / diagonal / airlock for sheet number n."""
    ia, ja = airlock_ij(building_rect)
    i, j = (n - 1) % 2, (n - 1) // 2
    if (i, j) == (ia, ja):
        return "airlock"
    if j == ja:
        return "along"
    if i == ia:
        return "across"
    return "diagonal"


def _ij_rect(building_rect, i, j):
    for n, r in sheet_rects(building_rect):
        if (n - 1) % 2 == i and (n - 1) // 2 == j:
            return n, r
    raise KeyError((i, j))


# ============================================================================================ the missing part
class Missing:
    """The missing sheet of one site: its number and place, the rooms it takes whole, the corridor cuts."""

    def __init__(self, sheet, of, rect, place, rooms, cuts, building_rect):
        self.sheet = int(sheet)
        self.of = int(of)
        self.rect = [float(v) for v in rect]
        self.place = str(place)
        self.rooms = sorted(rooms)
        self.cuts = list(cuts)
        self.building_rect = [float(v) for v in building_rect]
        self._rooms = set(self.rooms)
        self._rects = {}                   # room id -> rect (set by compute / from_json)
        self.items = []                    # tags of listed instruments on it (set after placement)

    # ---------------------------------------------------------------- records
    def to_json(self):
        return {"sheet": self.sheet, "of": self.of, "rect": self.rect, "place": self.place, "rooms": list(self.rooms),
                "room_rects": {k: list(v) for k, v in sorted(self._rects.items())}, "corridor_cuts": self.cuts,
                "items": list(self.items), "count": len(self.items)}

    @classmethod
    def from_json(cls, d, building_rect):
        m = cls(d["sheet"], d.get("of", 4), d["rect"], d.get("place", ""), d.get("rooms", []),
                d.get("corridor_cuts", []), building_rect)
        m._rects = {k: list(v) for k, v in (d.get("room_rects") or {}).items()}
        m.items = list(d.get("items", []))
        return m

    # ---------------------------------------------------------------- predicates
    def room_missing(self, rid):
        return rid in self._rooms

    def cut_of(self, leg):
        for c in self.cuts:
            if c["leg"] == leg:
                return c
        return None

    def in_cut(self, x, y, margin=0.0):
        """Is (x, y) in a cut-off corridor part (shrunk by `margin`)?"""
        for c in self.cuts:
            r = c["cut_rect"]
            if r[0] + margin <= x <= r[2] - margin and r[1] + margin <= y <= r[3] - margin:
                return True
        return False

    def cut_side(self, leg, x, y, margin=0.0):
        """For a point of corridor leg `leg`: +1 on the cut-off side at least `margin` from the cut, -1 on the drawn
        side at least `margin` from it (also when the leg is not cut), 0 within `margin` of the cut."""
        c = self.cut_of(leg)
        if c is None:
            return -1
        s = float(x) if c["axis"] == "x" else float(y)
        d = (s - c["at"]) if c["missing_side"] == "hi" else (c["at"] - s)
        return 1 if d >= margin else (-1 if d <= -margin else 0)

    def in_missing_room(self, x, y, margin=0.0):
        for rid in self.rooms:
            r = self._rects.get(rid)
            if r and r[0] + margin <= x <= r[2] - margin and r[1] + margin <= y <= r[3] - margin:
                return True
        return False

    def in_part(self, x, y, margin=0.0):
        """Is (x, y) on the missing part (a missing room or a cut-off corridor part, each shrunk by `margin`)?"""
        return self.in_missing_room(x, y, margin) or self.in_cut(x, y, margin)

    def areas(self):
        """Rects of the missing part: the missing rooms' rects and the cut-off corridor parts."""
        return [list(self._rects[r]) for r in self.rooms if r in self._rects] + [list(c["cut_rect"]) for c in self.cuts]

    def side_drawn(self, rid, x, y):
        """Is the side of a wall that belongs to room `rid`, at point (x, y), drawn? (OUT and missing rooms are not; a
        cut corridor leg is drawn on its drawn part.)"""
        if rid in (None, "OUT") or rid in self._rooms:
            return False
        c = self.cut_of(rid)
        if c is not None:
            r = c["cut_rect"]
            if r[0] - 1e-6 <= x <= r[2] + 1e-6 and r[1] - 1e-6 <= y <= r[3] + 1e-6:
                return False
        return True

    def drawn_rect(self, rid, rect):
        """The drawn rectangle of room `rid` (its drawn part for a cut corridor leg), or None when it is missing."""
        if rid in self._rooms:
            return None
        c = self.cut_of(rid)
        return list(c["drawn_rect"]) if c is not None else list(rect)


def _leg_axis(rect):
    """'x' for a leg that runs along x (C1, C3), 'y' for one that runs along y (C2)."""
    return "x" if (rect[2] - rect[0]) > (rect[3] - rect[1]) else "y"


def _openings_along(lay, W, leg, axis):
    """Intervals (lo, hi) along a corridor leg of every door opening, hatch and fire door on it."""
    out = []
    for d in lay.doors:
        if leg not in d.rooms:
            continue
        w = lay.wall(d.wall)
        # an opening in a wall parallel to the leg spans [center - width/2, center + width/2] along it; one in a wall
        # across the leg (an end wall) sits at the wall's coordinate
        if (axis == "x" and w.axis == "y") or (axis == "y" and w.axis == "x"):
            out.append((d.center - d.width / 2, d.center + d.width / 2))
        else:
            out.append((w.coord - 0.15, w.coord + 0.15))
    for f in getattr(W, "fire_doors", []) if W is not None else []:
        if f["rooms"][0] != leg:
            continue
        out.append((float(f["coord"]) - 0.06, float(f["coord"]) + 0.06))
    return out


def missing_rooms(lay, place):
    """(sheet number, sheet rect, [room ids], {room id: rect}) of the rooms the missing sheet at `place` takes whole
    (known right after the layout: the roles are steered with it before anything is built)."""
    br = lay.building_rect
    i, j = place_ij(br, place)
    n, rect = _ij_rect(br, i, j)
    rooms = []
    rects = {}
    for r in sorted(lay.rooms.values(), key=lambda r: r.id):
        if not r.on_drawing or r.type in ("corridor", "airlock"):
            continue
        cx, cy = 0.5 * (r.rect[0] + r.rect[2]), 0.5 * (r.rect[1] + r.rect[3])
        if sheet_ij(br, cx, cy) == (i, j):
            rooms.append(r.id)
            rects[r.id] = list(r.rect)
    return n, rect, rooms, rects


def compute(lay, place, g, W=None, shift_m=1.0, clear_m=0.3, step=0.05):
    """The missing sheet at `place` for layout `lay` (fire doors from world W: call it once the corridors are
    furnished). Returns a Missing, or None when a corridor leg cannot be cut 0.3 m clear of its openings within
    `shift_m` of the mid-line."""
    br = lay.building_rect
    i, j = place_ij(br, place)
    n, rect, rooms, rects = missing_rooms(lay, place)
    xm, ym = mid_lines(br)
    cuts = []
    for leg in CORRIDOR_LEGS:
        if leg not in lay.rooms:
            continue
        R = list(lay.rooms[leg].rect)
        axis = _leg_axis(R)
        if axis == "x":
            # a leg along x lies in one row; it is cut when that row is the missing sheet's
            if sheet_ij(br, xm, 0.5 * (R[1] + R[3]))[1] != j:
                continue
            c0, lo, hi = xm, R[0], R[2]
            missing_side = "hi" if i == 1 else "lo"
        else:
            if sheet_ij(br, 0.5 * (R[0] + R[2]), ym)[0] != i:
                continue
            c0, lo, hi = ym, R[1], R[3]
            missing_side = "hi" if j == 1 else "lo"
        if not (lo < c0 < hi):
            continue
        ops = _openings_along(lay, W, leg, axis)
        # the doors of drawn rooms (rooms the sheet leaves) along the leg: a cut that leaves one of them on the cut-off
        # side makes that room reachable on the drawing only through the missing part; such cuts come last
        own = []
        for d in lay.doors:
            if leg not in d.rooms or not d.on_drawing:
                continue
            other = d.rooms[0] if d.rooms[1] == leg else d.rooms[1]
            if other in lay.rooms and lay.rooms[other].on_drawing and other not in rects and \
                    lay.rooms[other].type != "corridor":
                w = lay.wall(d.wall)
                if (axis == "x" and w.axis == "y") or (axis == "y" and w.axis == "x"):
                    own.append(d.center)
        target = c0 + float(g.uniform(-shift_m, shift_m))
        best = None
        k = int(round(shift_m / step))
        for t in range(-k, k + 1):
            c = round((c0 + t * step) / step) * step
            if not (lo + 0.5 < c < hi - 0.5):
                continue
            if any(a - clear_m < c < b + clear_m for a, b in ops):
                continue
            cut_off = sum(1 for x in own if (x > c if (axis == "x" and i == 1) or (axis == "y" and j == 1) else x < c))
            key = (cut_off, abs(c - target), c)
            if best is None or key < best[0]:
                best = (key, c)
        if best is None:
            return None
        c = float(round(best[1], 4))
        if axis == "x":
            drawn = [R[0], R[1], c, R[3]] if missing_side == "hi" else [c, R[1], R[2], R[3]]
            cut = [c, R[1], R[2], R[3]] if missing_side == "hi" else [R[0], R[1], c, R[3]]
        else:
            drawn = [R[0], R[1], R[2], c] if missing_side == "hi" else [R[0], c, R[2], R[3]]
            cut = [R[0], c, R[2], R[3]] if missing_side == "hi" else [R[0], R[1], R[2], c]
        cuts.append({"leg": leg, "axis": axis, "at": c, "mid_line": round(c0, 4), "missing_side": missing_side,
                     "drawn_rect": [float(v) for v in drawn], "cut_rect": [float(v) for v in cut]})
    m = Missing(n, 4, rect, place, rooms, cuts, br)
    m._rects = rects
    return m


# ============================================================================================ drawing pieces
def wall_pieces(miss, w_axis, coord, lo, hi, rooms, t_neg, t_pos):
    """The drawn sub-intervals of a drawing wall: the wall split at every corridor cut it runs along, each piece kept
    when one of its two sides is drawn (missing rooms and cut-off corridor parts are not). Returns [(lo, hi)]."""
    pts = {float(lo), float(hi)}
    for c in miss.cuts:
        if c["leg"] in rooms and ((c["axis"] == "x" and w_axis == "y") or (c["axis"] == "y" and w_axis == "x")):
            if lo + 1e-6 < c["at"] < hi - 1e-6:
                pts.add(float(c["at"]))
    pts = sorted(pts)
    out = []
    for a, b in zip(pts[:-1], pts[1:]):
        if b - a < 1e-6:
            continue
        mid = 0.5 * (a + b)
        drawn = False
        for k, rid in enumerate(rooms):
            off = -(t_neg + 0.05) if k == 0 else (t_pos + 0.05)
            x, y = (coord + off, mid) if w_axis == "x" else (mid, coord + off)
            if miss.side_drawn(rid, x, y):
                drawn = True
        if drawn:
            if out and abs(out[-1][1] - a) < 1e-6:
                out[-1] = (out[-1][0], b)
            else:
                out.append((a, b))
    return out


def point_on_pieces(pieces, s, tol=1e-6):
    return any(a - tol <= s <= b + tol for a, b in pieces)


def dist_point_rect(x, y, r):
    dx = max(r[0] - x, 0.0, x - r[2])
    dy = max(r[1] - y, 0.0, y - r[3])
    return math.hypot(dx, dy)
