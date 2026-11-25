"""The calibration site `cal` of task 07, version 2.0 (calgen 0.4.0): one hand-laid site package built from the
families' own parts (w2gen: boxes, opening classes, the camera rule, lighting, MJCF, the free-block spacing rule;
w3gen: the site depth, the camera rectangle, the plan radius and clear heights of the slab, point 3's clearance, the
blind-route window; w4gen: the heap library, the placement transform, the heaps' collision bodies and their
measurements on the compiled model), every dimension public (the calibration page states them; it never says what
passes).

    python -I -B calgen.py --out <dir> [--no-render] [--envelope <capability table json>]
                           [--passage-length L]   (not the calibration site unless L = PASSAGE_LEN)

writes <dir>/cal/{site.json, site.xml, qc.json}, <dir>/cal.tar.xz (the package the practice ground serves; members
cal/site.json, cal/site.xml; deterministic: sorted, mtime 0, uid / gid 0), <dir>/plan.json (the public plan: every
dimension, for the calibration page; no friction, no 'passes' / 'does not pass') and <dir>/heap_pieces.json (the
maintainers' table: which library piece made each heap; never part of the package). The envelope is the family's
(w2gen.envelope.load(): the simulation base's capability sheet, the table every scored site of W2, W3 and W4 is sized
from); --envelope replaces it with a local table (then the generator string says so: not the calibration site). The
heap library is the one w4gen reads (w4gen.library.Library(): its registry, by default the library of version 3, and
its piece files, every file checked against the registry's sha256).

calgen 0.4.0 against 0.3.0: only the heap library and the two pieces change (heap library version 3: rubble heaps of
real bricks dropped against a wall, mounds of fill with a natural toe and no rim); the heaps' places are 0.3.0's; the
class geometry is w4gen 0.3.0's (heapchecks: rubble heaps by the edge rule and the core, mounds of fill by their depth,
their gap to the line, the registered scale and the piece's trial verdict; no slope band for either class); the fit
line of G24 is the registry's per class. Everything else (every other collision body, the camera, the marks) is
0.3.0's byte for byte.

Version 2.0 against version 1.1: everything of version 1.1 stays where it was (the flat floor, the marks, the ridge
and its openings, the roofed room, passage, block and chamber, the crates, the strips, the chips, camera and light,
byte for byte the same collision bodies), except
  * two heaps of the scored sites' two kinds on the open floor, hand-laid, each leaning on a line, both seen by the
    camera (HEAPS): a rubble heap against the site's south wall in the south-east corner, flush with the blue crate's
    west face (the only stretch of a perimeter wall in the picture that is neither the flat floor nor next to a ridge
    opening: the ridge's two faces lie within 0.40 m of an opening's floor point everywhere), and a mound of fill
    against the free-standing wall's south face, centred on it. Both at scale 1.0: the library pieces exactly as their
    trials ran them. Their collision bodies are inline height fields (w4gen's transform and writing: the piece's
    5 mm grid, sunk 10 mm, a ring of zero cells, max == size[2]); their occupancy for every rule is the placed
    rectangle, and every rule is judged again on the compiled model (qc: heaps_G20 ... heaps_G24).
  * the free-standing wall stands 0.30 m further north (y 3.50-3.65; version 1.1: 3.20-3.35), so that the mound in
    front of it stands on the concrete, clear of the gravel strip and of search point 2's keep-out.

Layout (4.4 x 5.8 m; the camera rule of the families on the camera rectangle 4.4 x 4.0 m, CAMERA_SEED 9001 as in
version 1.0, so the camera is version 1.0's; floor friction 0.80; x east, y north, origin south-west corner):
  south    (unchanged) a 3.0 x 1.5 m flat floor with visual 0.5 m grid lines and four crosses; the pad; search point
           1; a dust strip
  ridge    (unchanged) y 1.60-2.15, west to east: gap_XN (A), gap_T (B), gap_P (C, open), slot_XL (D, covered,
           clear 0.15), a covered mouth closed at depth 0.30 (E, class B), tunnel_P (F, covered, flat roof, 0.55)
  north    (unchanged up to y 3.0) the roofed room x 2.80-4.40, y 2.15-3.00 behind tunnel_P; a free-standing wall
           (y 3.50-3.65) with the mound of fill against its south face; a gravel strip; search point 2; six colour
           chips (the red crate and the rubble heap stand in the south band)
  roofed   (new) from the room's north-west corner a roofed passage x 2.80-3.65 northward (0.85 wide, 1.40 long:
           see PASSAGE_LEN), in it one rubble block against its west wall (rubble__passage_block:
           x 2.80 to 2.80 + (0.85 - P), along it 0.25 m centred on the passage's length, floor to roof; the clear
           width beside it = the P class's width = the width of openings C-F); then a roofed chamber x 2.80-4.40 up
           to the site's back wall (its east wall = the site's east wall); one side wall x 2.62-2.80 from the ridge
           to the back wall; east of the passage solid (x 3.65-4.40); ONE roof over the room, passage and chamber at
           the slab clear height of the families (H_ROLL + 0.14). Search point 3 = (3.75, 5.15): 0.65 m from the
           east and the back wall.
  blue crate: version 1.0's place is solid now; it stands flush in the south-east corner (FREE_BLOCK_RULE, CRATES).
Search points 1 and 2 are version 1.0's; weights as every site (read from the acceptance file).
"""
import argparse
import hashlib
import io
import json
import math
import os
import sys
import tarfile

HERE = os.path.dirname(os.path.abspath(__file__))
DEV = os.path.normpath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(DEV, "tools", "sitegen"))

import numpy as np  # noqa: E402

from w2gen import GENERATOR_VERSION as W2_VERSION  # noqa: E402
from w2gen import envelope as envmod  # noqa: E402
from w2gen import family as fam  # noqa: E402
from w2gen import grid as gridmod  # noqa: E402
from w2gen import mjcf, predicates  # noqa: E402
from w3gen import GENERATOR_VERSION as W3_VERSION  # noqa: E402
from w3gen import family as w3fam  # noqa: E402
from w3gen import predicates as w3p  # noqa: E402
from w4gen import GENERATOR_VERSION as W4_VERSION  # noqa: E402
from w4gen import heapchecks as hc  # noqa: E402
from w4gen import library as libmod  # noqa: E402
from w4gen import measure as ms  # noqa: E402
from w4gen import mjcf as w4mjcf  # noqa: E402

if "heap" not in gridmod.SOLID_KINDS:
    gridmod.SOLID_KINDS = tuple(gridmod.SOLID_KINDS) + ("heap",)      # run time: a heap is a solid for every route

CAL_VERSION = "calgen 0.4.0"
LX, LY = 4.4, w3fam.LY                 # 4.4 x 5.8: the W3 site rectangle
LY_CAM = w3fam.LY_CAM                  # 4.0: the camera rule's rectangle (as every W3 site)
CAMERA_SEED = 9001                     # the camera rule's draw for this one site (version 1.0's: the camera is unchanged)
FLOOR_MU = 0.80
PAD = (2.20, 0.45)
SP = {1: (0.95, 1.05), 2: (1.75, 2.70), 3: (3.75, 5.15)}
WHERE = {1: "flat floor", 2: "north of the ridge, open floor", 3: "roofed chamber"}
RIDGE_Y = (1.60, 2.15)
END_W = 0.15                           # west end solid of the ridge
SOLID_W = 0.14                         # solid between two open gaps / an open gap and the covered block
BLIND_DEPTH = 0.30
ROOM = dict(x0=2.80, x1=4.40, y0=2.15, y1=3.00)          # interior; east side = the site's east wall
SIDE_T = 0.18                          # the side wall west of the room, the passage and the chamber (x 2.62-2.80)
PASSAGE_X = (2.80, 3.65)               # 0.85 wide, from the room's north-west corner northward
PASSAGE_LEN = 1.40                     # y 3.00-4.40 (a first layout with 1.20 m was laid before it; the blind route
                                       # measured 3.755 m with either length)
BLOCK_ALONG = 0.25                     # the passage block's length along the passage, centred on its length
FREE_WALL = dict(x0=0.55, x1=2.05, y0=3.50, y1=3.65, h=0.60)          # version 1.1: y 3.20-3.35 (moved 0.30 m north)
FREE_WALL_MOVED = "0.30 m north: y 3.20-3.35 in version 1.1"
GRAVEL = dict(x0=0.15, x1=1.25, y0=2.40, y1=3.00)        # 1.10 x 0.60
DUST = dict(x0=3.75, x1=4.35, y0=0.30, y1=1.20)          # 0.60 x 0.90 (0.90 x 0.60 turned)
FLAT = dict(x0=0.70, x1=3.70, y0=0.05, y1=1.55)          # 3.0 x 1.5
CROSSES = [(1.20, 0.55), (3.20, 0.55), (1.20, 1.30), (3.20, 1.30)]
# the crates (free blocks, hand-laid). RED is version 1.0's (unchanged with everything at y <= 3.0; it does not meet
# FREE_BLOCK_RULE: 0.15 / 0.20 m from the west / south walls, and is not judged by it). The blue crate of version 1.0
# (x 3.90-4.18, y 3.45-3.75) is inside the solid east of the passage now; BLUE is its version 1.1 place, same size and
# turn: flush in the south-east corner. An exhaustive search (0.01 m grid, both turns, the routes pad -> 1 -> 2 -> 3
# -> pad kept) found exactly two places that pass the rule: this one and its 0.30 x 0.28 turn.
RED = dict(name="free_red", x0=0.15, x1=0.45, y0=0.20, y1=0.48, h=0.30, rgb=fam.CRATE_RGB[0])
BLUE = dict(name="free_blue", x0=4.12, x1=4.40, y0=0.00, y1=0.30, h=0.36, rgb=fam.CRATE_RGB[1])
CRATES = [c for c in (RED, BLUE) if c]
RULED = ("free_blue",)                  # the crates FREE_BLOCK_RULE gates (the ones placed in version 1.1)
FREE_BLOCK_RULE = ("the families' free-block rule (w2gen.family.place_free_blocks / spacing_ok): to every other solid "
                   "(perimeter, ridge, walls, other crates) a gap >= free_block_clearance (W_SWEEP + 0.18) or flush "
                   "(gap 0, no overlap); the centre > r + half the diagonal from the pad (0.45), points 1 and 2 (0.40) "
                   "and 0.25 m before and behind every opening (0.40); on no ground strip; plus, here: not on the flat "
                   "floor (the page says nothing stands on it), inside the camera rectangle, its top seen by the camera")
CHIPS = [("red", "disc", 0.040, (1.55, 0.95)), ("green", "square", 0.050, (2.85, 1.15)),
         ("blue", "disc", 0.055, (0.95, 0.40)), ("red", "square", 0.035, (1.25, 3.70)),
         ("green", "disc", 0.045, (0.40, 2.25)), ("blue", "square", 0.060, (2.60, 0.85))]
LINE_W, LINE_T = 0.012, 0.0008
MARK_GROUP = 2                          # visual only, rendered by the search camera (as the family's chips)
TOL = 1e-6 + 1e-9                       # package coordinates are rounded to 1e-6 m
# the two heaps (hand-laid). piece = the library piece (for the maintainers only: the package never names it), scale 1.0
# (the piece as its trials ran it), line = the side of the heap the line it leans on is on; S0's place: its east end
# flush with the blue crate's west face (x1 = BLUE x0), its south side on the site's south wall (y = 0); N0's place:
# its north side on the free-standing wall's south face, centred on the wall (both places are calgen 0.3.0's). The
# pieces (heap library version 3): per class the usable pieces ranked by how central they are (the sum over the
# registered rectangle's width and depth and the registered top of |value - the class's median| / the class's spread
# of that value); the most central one whose placed rectangle at scale 1.0 passes every rule at its place. Rubble
# heaps: the most central piece (0.353 x 0.320 m) is 0.020 m too deep for the south-east corner (it reaches the dust
# strip, and north of the crate it leaves a gap of open floor in [0.15, 0.60) m to the east wall), and at the other
# lines of the picture (the free-standing wall's ends, its north face) it breaks the gap promise or is unseen; the
# second (0.330 x 0.290 m) passes. Mounds of fill: the most central piece passes. Mirror: the one whose collision body
# fits the piece's mesh better (the fit bound's share within 3 mm).
HEAPS = (
    dict(name="heap__S0", band="south", cls="R", piece="R83", scale=1.0, mirror=True, line="south",
         against="site wall", stands="against the site's south wall", x1=BLUE["x0"]),
    dict(name="heap__N0", band="north", cls="M", piece="M56", scale=1.0, mirror=False, line="north",
         against="free-standing wall", stands="against the free-standing wall's south face",
         cx=0.5 * (FREE_WALL["x0"] + FREE_WALL["x1"])),
)
HEAP_KIND = {"R": "rubble heap", "M": "mound of fill"}
FIT_SHARE = dict(hc.FIT_SHARE_BY_CLASS)  # fallback fit lines (share of points within 3 mm) per class; the registry's
                                        # own (doc.classes.<cls>.fit, heapchecks.fit_line) are the ones judged
FIT_MAX_MM = 8.0
VERTEX_MM = 0.5
PUBLIC_STEP, PUBLIC_Z = 0.001, 0.001    # the public outline: a 1 mm lattice of the compiled surface, > 1 mm above floor
HEAP_KEEP = (("pad", 0.45), ("point 1", 0.40), ("point 2", 0.40))     # the families' keep-outs (W4: G21)
OPEN_KEEP, OPEN_OFF = 0.40, 0.25        # around the floor points 0.25 m before / behind every ridge opening
VIS_MIN = 0.70


def r6(x):
    return round(float(x), 6)


def layout(passage_len=PASSAGE_LEN):
    """Rectangles (x0, x1, y0, y1) of the roofed part (room, passage, chamber, side wall, fill, roof, block)."""
    py0 = ROOM["y1"]
    py1 = py0 + passage_len
    passage = dict(x0=PASSAGE_X[0], x1=PASSAGE_X[1], y0=r6(py0), y1=r6(py1))
    chamber = dict(x0=ROOM["x0"], x1=LX, y0=r6(py1), y1=LY)
    side = dict(x0=r6(ROOM["x0"] - SIDE_T), x1=ROOM["x0"], y0=ROOM["y0"], y1=LY)
    fill = dict(x0=PASSAGE_X[1], x1=LX, y0=r6(py0), y1=r6(py1))
    roof = dict(x0=side["x0"], x1=LX, y0=ROOM["y0"], y1=LY)
    ym = 0.5 * (py0 + py1)
    block_y = (r6(ym - BLOCK_ALONG / 2), r6(ym + BLOCK_ALONG / 2))
    return dict(room=dict(ROOM), passage=passage, chamber=chamber, side=side, fill=fill, roof=roof, block_y=block_y)


def ridge(cls):
    """Openings and boxes of the ridge (west to east)."""
    a, y1b = RIDGE_Y
    depth = y1b - a
    P_w, P_h, T_w, XN_w = cls["P"]["w"], cls["P"]["clear_h"], cls["T"]["w"], cls["XN"]["w"]
    SUP = fam.SUP
    boxes, ops = [], []
    x = 0.0

    def solid(name, x0, x1, h=0.70):
        boxes.append(fam._box(name, "rubble", x0, x1, a, y1b, 0.0, h))

    def open_gap(gid, typ, c, w):
        nonlocal x
        ops.append(dict(id=gid, row="R1", type=typ, cls=c, x0=r6(x), x1=r6(x + w), y0=r6(a), y1=r6(y1b),
                        foot=[r6(x), r6(x + w)], covered=False, depth=r6(depth), role="opening"))
        x += w

    solid("ridge_w", 0.0, END_W)
    x = END_W
    open_gap("R1_0", "gap_XN", "XN", XN_w)
    solid("ridge_1", x, x + SOLID_W)
    x += SOLID_W
    open_gap("R1_1", "gap_T", "T", T_w)
    solid("ridge_2", x, x + SOLID_W)
    x += SOLID_W
    open_gap("R1_2", "gap_P", "P", P_w)
    solid("ridge_3", x, x + SOLID_W)
    x += SOLID_W
    # covered block: sup | slot_XL | sup | blind | sup | tunnel_P | sup
    sup_h = P_h
    covered = [("R1_3", "slot_XL", "XL", fam.XL_CLEAR), ("R1_4", "mouth_blind", "B", P_h), ("R1_5", "tunnel_P", "P", P_h)]
    xs = x
    boxes.append(fam._box("sup_0", "support", xs, xs + SUP, a, y1b, 0.0, sup_h))
    x = xs + SUP
    for k, (gid, typ, c, h) in enumerate(covered):
        x0, x1 = x, x + P_w
        f0, f1 = x0 - SUP, x1 + SUP
        op = dict(id=gid, row="R1", type=typ, cls=c, x0=r6(x0), x1=r6(x1), y0=r6(a), y1=r6(y1b), foot=[r6(f0), r6(f1)],
                  covered=True, clear_h=r6(h), clear_h_min=r6(h), roof="flat", roof_t=fam.R1_ROOF_T, depth=r6(depth),
                  role="opening")
        if typ == "slot_XL":
            # the slot's own low supports inside the shared ones would leave nothing to see: its roof spans the
            # opening between the two (taller) shared supports at the low clear height
            boxes.append(fam._box("roof_" + gid, "roof", x0, x1, a - fam.ROOF_OVERHANG, y1b, h, h + fam.R1_ROOF_T))
        else:
            # the covered mouth and the passage share the support between them: each roof covers half of it (no
            # two coplanar roof boxes overlap: they would flicker in the camera's frames)
            r0 = f0 if typ == "mouth_blind" else x0 - SUP / 2
            r1 = x1 + SUP / 2 if typ == "mouth_blind" else f1
            boxes.append(fam._box("roof_" + gid, "roof", r0, r1, a - fam.ROOF_OVERHANG, y1b, h, h + fam.R1_ROOF_T))
        if typ == "mouth_blind":
            boxes.append(fam._box("plug_" + gid, "plug", x0, x1, a + BLIND_DEPTH, y1b, 0.0, h))
            op.update(depth=r6(BLIND_DEPTH), plug_depth_m=r6(BLIND_DEPTH), role="blind mouth")
        ops.append(op)
        x = x1
        boxes.append(fam._box("sup_%d" % (k + 1), "support", x, x + SUP, a, y1b, 0.0, sup_h))
        x += SUP
    solid("ridge_e", x, LX)
    return ops, boxes, dict(end_w=r6(END_W), solid_w=r6(SOLID_W), east_end_w=r6(LX - x))


def roofed(cls, L):
    """The roofed part behind tunnel_P: side wall, fill east of the passage, the passage block, ONE roof (the
    families' slab: underside at the clear height H_ROLL + 0.14, 0.12 thick)."""
    hz = cls["gallery"]["clear_h"]
    P_w = cls["P"]["w"]
    s, f, r, p = L["side"], L["fill"], L["roof"], L["passage"]
    bx1 = r6(p["x0"] + (p["x1"] - p["x0"]) - P_w)          # x 2.80 to 2.80 + (0.85 - P)
    by0, by1 = L["block_y"]
    boxes = [fam._box("side_w", "wall", s["x0"], s["x1"], s["y0"], s["y1"], 0.0, hz),
             fam._box("passage_fill_e", "wall", f["x0"], f["x1"], f["y0"], f["y1"], 0.0, hz),
             fam._box("passage_block", "rubble", p["x0"], bx1, by0, by1, 0.0, hz),
             fam._box("void_slab", "roof", r["x0"], r["x1"], r["y0"], r["y1"], hz, hz + fam.SLAB_T)]
    slab = dict(x0=r6(r["x0"]), x1=r6(r["x1"]), y0=r6(r["y0"]), y1=r6(r["y1"]), z0=r6(hz), z1=r6(hz + fam.SLAB_T))
    block = dict(name="rubble__passage_block", kind="rubble", x0=r6(p["x0"]), x1=bx1, y0=by0, y1=by1, z0=0.0, z1=r6(hz),
                 against="west wall of the passage", across_m=r6(bx1 - p["x0"]), along_m=r6(by1 - by0),
                 gap_east_m=r6(p["x1"] - bx1), gap_rule="clear width beside it = the P class's width (openings C-F)")
    return boxes, slab, block, hz


def visual_marks():
    """0.5 m grid lines on the flat floor and four crosses (visual only, MJCF text, and their public description)."""
    F = FLAT
    L = []
    lines = []
    gx = np.arange(F["x0"], F["x1"] + 1e-9, 0.5)
    gy = np.arange(F["y0"] + 0.45, F["y1"] + 1e-9, 0.5)          # y lines at 0.5, 1.0, 1.5
    z = LINE_T / 2
    for k, xv in enumerate(gx):
        L.append('<geom name="mark__gx%d" type="box" pos="%s %s %s" size="%s %s %s" rgba="0.92 0.92 0.88 1" '
                 'contype="0" conaffinity="0" group="%d"/>' % (k, mjcf._f(xv), mjcf._f((F["y0"] + F["y1"]) / 2), mjcf._f(z),
                                                            mjcf._f(LINE_W / 2), mjcf._f((F["y1"] - F["y0"]) / 2),
                                                            mjcf._f(LINE_T / 2), MARK_GROUP))
        lines.append(dict(axis="x", at=r6(xv), from_=r6(F["y0"]), to=r6(F["y1"])))
    for k, yv in enumerate(gy):
        L.append('<geom name="mark__gy%d" type="box" pos="%s %s %s" size="%s %s %s" rgba="0.92 0.92 0.88 1" '
                 'contype="0" conaffinity="0" group="%d"/>' % (k, mjcf._f((F["x0"] + F["x1"]) / 2), mjcf._f(yv), mjcf._f(z),
                                                            mjcf._f((F["x1"] - F["x0"]) / 2), mjcf._f(LINE_W / 2),
                                                            mjcf._f(LINE_T / 2), MARK_GROUP))
        lines.append(dict(axis="y", at=r6(yv), from_=r6(F["x0"]), to=r6(F["x1"])))
    arm, w = 0.10, 0.02
    for k, (cx, cy) in enumerate(CROSSES):
        for j, (sx, sy) in enumerate(((arm, w / 2), (w / 2, arm))):
            L.append('<geom name="mark__cross%d_%d" type="box" pos="%s %s %s" size="%s %s %s" rgba="0.95 0.85 0.10 1" '
                     'contype="0" conaffinity="0" group="%d"/>' % (k, j, mjcf._f(cx), mjcf._f(cy), mjcf._f(1.5 * LINE_T),
                                                                mjcf._f(sx), mjcf._f(sy), mjcf._f(LINE_T), MARK_GROUP))
    return "\n    ".join(L), dict(grid_lines=lines, line_width_m=LINE_W, crosses=[dict(xy=[r6(a), r6(b)], arm_m=arm,
                                                                                         bar_width_m=w) for a, b in CROSSES])


def chips():
    out = []
    for k, (colour, shape, s, (x, y)) in enumerate(CHIPS):
        out.append(dict(name="chip__%d" % k, shape=shape, size_m=r6(s), thickness_m=0.002, xy=[r6(x), r6(y)], z_m=0.0,
                        yaw_deg=r6(17.0 * k), colour=colour, lightness=1.0, rgb=[r6(c) for c in fam.CAP_RGB[colour]],
                        dist_to_route_m=None))
    return out


def build(env, passage_len=PASSAGE_LEN, crates=None):
    crates = CRATES if crates is None else crates
    cls = w3fam.classes(env)
    L = layout(passage_len)
    ops, rboxes, ridge_info = ridge(cls)
    vboxes, slab, block, hz = roofed(cls, L)
    boxes = []
    walls = []
    for nm, (x0, x1, y0, y1) in dict(s=(-fam.WALL_T, LX + fam.WALL_T, -fam.WALL_T, 0.0),
                                     n=(-fam.WALL_T, LX + fam.WALL_T, LY, LY + fam.WALL_T),
                                     w=(-fam.WALL_T, 0.0, 0.0, LY), e=(LX, LX + fam.WALL_T, 0.0, LY)).items():
        boxes.append(fam._box(nm, "wall", x0, x1, y0, y1, 0.0, fam.WALL_H))
        walls.append("wall__" + nm)
    boxes += rboxes + vboxes
    W = FREE_WALL
    boxes.append(fam._box("free_wall", "wall", W["x0"], W["x1"], W["y0"], W["y1"], 0.0, W["h"]))
    for c in crates:
        boxes.append(fam._box(c["name"], "crate", c["x0"], c["x1"], c["y0"], c["y1"], 0.0, c["h"], rgb=c["rgb"],
                              free_block=True))
    patches = [dict(name="dust", x0=DUST["x0"], x1=DUST["x1"], y0=DUST["y0"], y1=DUST["y1"], mu=fam.PATCH_MU["dust"],
                    thickness=0.002, rgb=list(fam.PATCH_RGB["dust"])),
               dict(name="gravel", x0=GRAVEL["x0"], x1=GRAVEL["x1"], y0=GRAVEL["y0"], y1=GRAVEL["y1"],
                    mu=fam.PATCH_MU["gravel"], thickness=0.002, rgb=list(fam.PATCH_RGB["gravel"]))]
    rng = np.random.default_rng(CAMERA_SEED)
    camera = fam.draw_camera(rng, LX, LY_CAM)
    lighting = dict(key_dir=[r6(q) for q in np.array([0.15, 0.25, -1.0]) / np.linalg.norm([0.15, 0.25, -1.0])],
                    key_off_nominal_deg=0.0, key_diffuse=0.75, ambient=0.35,
                    nominal=dict(key_dir=None, key_diffuse=0.75, ambient=0.35),
                    rule="the family's nominal key light and ambient (the calibration site's lighting does not vary)")
    lighting["nominal"]["key_dir"] = lighting["key_dir"]
    R, P, C = L["room"], L["passage"], L["chamber"]
    runs = dict(room=dict(x0=r6(R["x0"]), x1=r6(R["x1"]), y0=r6(R["y0"]), y1=r6(R["y1"]), width=r6(R["y1"] - R["y0"]),
                          axis="x"),
                passage=dict(x0=r6(P["x0"]), x1=r6(P["x1"]), y0=r6(P["y0"]), y1=r6(P["y1"]), width=r6(P["x1"] - P["x0"]),
                             axis="y"))
    plug_op = [o for o in ops if o["cls"] == "B"][0]
    weights = w3fam.WEIGHTS
    gen = "%s (parts of %s, %s, %s)" % (CAL_VERSION, W2_VERSION, W3_VERSION, W4_VERSION)
    if abs(passage_len - PASSAGE_LEN) > 1e-9:
        gen += " [passage length %.2f m: not the calibration site]" % passage_len
    site = dict(
        family="CAL", generator=gen, seed=0, kind="cal",
        frame="x east, y north, z up, origin = south-west floor corner; metres",
        size_m=[LX, LY], camera_depth_m=LY_CAM, body_length_m=fam.BL, envelope=env, classes=cls,
        params=dict(ridge_y=[RIDGE_Y[0], RIDGE_Y[1]], slack_R1_m=0.0, slack_R2_m=0.0, y2=RIDGE_Y[0], **ridge_info),
        bands=dict(south=[0.0, RIDGE_Y[0]], ridge=list(RIDGE_Y), north=[RIDGE_Y[1], LY]),
        r1=dict(order=[o["type"] for o in ops], third="all", tunnel_roof="flat", openings=ops),
        r2=dict(order=[], decoy_type=None, openings=[]),
        void=dict(sp3_mode="chamber", true_mouth="R1_5", blind_mouth="R1_4", runs=runs,
                  chamber=dict(x0=r6(C["x0"]), x1=r6(C["x1"]), y0=r6(C["y0"]), y1=r6(C["y1"])), pockets=[],
                  gallery=dict(x0=r6(R["x0"]), x1=r6(R["x1"]), y0=r6(R["y0"]), y1=r6(R["y1"]), clear_h=r6(hz),
                               note="= the roofed room (legacy key)"),
                  slab=slab, passage_block=block,
                  plug=dict(x0=plug_op["x0"], x1=plug_op["x1"], y0=r6(RIDGE_Y[0] + BLIND_DEPTH), y1=r6(RIDGE_Y[1]),
                            z1=r6(cls["P"]["clear_h"])),
                  note="the calibration site's roofed part behind tunnel_P (room -> passage -> chamber, one roof) "
                       "stands in the void's place (truth fields named as in W3: runs room and passage, chamber C, "
                       "slab = the one roof; gallery = the room, legacy key; params.y2 = the ridge's south face)"),
        pad=dict(xy=[PAD[0], PAD[1]], retrieval_radius_m=0.60, setdown_error_m=0.10),
        search_points=[dict(id=i, xy=[SP[i][0], SP[i][1]], weight=weights[i], where=WHERE[i]) for i in (1, 2, 3)],
        patches=patches, boxes=boxes, walls=walls, camera=camera, lighting=lighting,
        physics=dict(floor_mu=FLOOR_MU, box_mu=fam.BOX_MU, patch_mu=dict(fam.PATCH_MU),
                     floor_mu_rule="the calibration site's floor: 0.80"),
        gravel_relief=dict(enabled=False, grid_m=0.01, wavelength_m=[0.03, 0.08], rms_m=0.008, peak_max_m=0.025,
                           edge_taper_m=0.05, heights="off", rng_stream=None),
        free_blocks=dict(placed=len(crates), dropped_for_G9=0, kept=["crate__" + c["name"] for c in crates],
                         rule="hand-laid; " + FREE_BLOCK_RULE + " (judged for: %s)" % ", ".join(RULED),
                         moved=(["crate__free_blue: from x 3.90-4.18, y 3.45-3.75 (version 1.0; solid now) to the "
                                 "south-east corner"] if BLUE else []),
                         removed=([] if BLUE else ["the blue crate of version 1.0 (its place is solid now; no place "
                                                   "passes the rule)"])),
        chips=chips(), chip_rule="six chips, hand-laid, visual only",
        room=dict(interior=dict(R), wall_t=SIDE_T, clear_h=r6(hz), slab_t=fam.SLAB_T),
        passage=dict(interior=dict(P), width_m=r6(P["x1"] - P["x0"]), length_m=r6(P["y1"] - P["y0"]), clear_h=r6(hz),
                     block=block),
        chamber=dict(interior=dict(C), size_m=[r6(C["x1"] - C["x0"]), r6(C["y1"] - C["y0"])], clear_h=r6(hz)),
        side_wall=dict(L["side"], t=SIDE_T, h=r6(hz)),
        passage_fill=dict(L["fill"], h=r6(hz)),
        free_wall=dict(FREE_WALL, moved=FREE_WALL_MOVED),
    )
    marks_xml, marks = visual_marks()
    site["marks"] = marks
    return site, marks_xml


# ------------------------------------------------------------------------------------------------ heaps
def heap_box(h):
    x0, x1, y0, y1 = h["rect"]
    return fam._box(h["name"].split("__", 1)[1], "heap", x0, x1, y0, y1, 0.0, h["top_m"], rgb=w4mjcf.HEAP_RGB[h["cls"]],
                    heap=dict(cls=h["cls"], band=h["band"], line=h["line"], against=h["against"]))


def place_heaps(site, lib, heaps=HEAPS):
    """Add the hand-laid heaps to the site (records in site['heaps'], box records of kind heap in site['boxes'] for
    the routes and the cell's truth); (grids by heap name, the maintainers' piece table by heap name)."""
    grids, pieces, recs = {}, {}, []
    for k, spec in enumerate(heaps):
        pid = spec["piece"]
        if pid not in lib.pieces:
            raise SystemExit("heap library: piece %s is not usable (%s)" % (pid, lib.excluded.get(pid, "not registered")))
        piece = lib.pieces[pid]
        sig = float(spec["scale"])
        lo, hi = piece["scale"]
        if not lo - 1e-9 <= sig <= hi + 1e-9:
            raise SystemExit("heap %s: scale %.4f outside the piece's interval [%.4f, %.4f]" % (spec["name"], sig, lo, hi))
        W = sig * piece["W"]
        if spec["line"] == "south":
            yaw, y_line = 180, 0.0
        else:
            yaw, y_line = 0, FREE_WALL["y0"]
        cx = spec["cx"] if "cx" in spec else spec["x1"] - W / 2
        place = dict(scale=sig, mirror=bool(spec["mirror"]), yaw_deg=yaw, cx=cx, y_line=y_line)
        rect = libmod.placed_rect(piece, sig, yaw, cx, y_line)
        place["rect"] = rect
        g = libmod.collision_grid(lib, pid, place)
        rec = dict(name=spec["name"], cls=spec["cls"], kind=HEAP_KIND[spec["cls"]], band=spec["band"], line=spec["line"],
                   against=spec["against"], stands=spec["stands"], rect=[r6(q) for q in rect], top_m=r6(g["top_m"]),
                   yaw_deg=yaw, mirror=bool(spec["mirror"]), scale=sig, x_mode="hand-laid", y_line=r6(y_line), cx=r6(cx),
                   grid=dict(origin=[r6(q) for q in g["origin"]], shape=list(g["shape"]), cell_m=r6(g["grid"]),
                             sink_m=r6(-g["base_z"]), size_z=r6(g["size_z"])))
        recs.append(rec)
        grids[rec["name"]] = g
        pieces[rec["name"]] = dict(piece=pid, scale=sig, mirror=bool(spec["mirror"]), yaw_deg=yaw,
                                   scale_interval_used=[r6(q) for q in piece["scale"]],
                                   scale_interval_registered=piece["scale_registered"],
                                   trials_verdict_ok=bool((piece.get("trials_verdict") or {}).get("ok")))
    site["boxes"] = [b for b in site["boxes"] if b["kind"] != "heap"] + [heap_box(h) for h in recs]
    site["heaps"] = dict(
        n=len(recs), by_band={b: sum(1 for h in recs if h["band"] == b) for b in ("south", "north")},
        by_class={c: sum(1 for h in recs if h["cls"] == c) for c in "RM"}, heaps=recs,
        library=dict(sha256=lib.sha256, version=lib.version),
        classes={c: dict(w=list(lib.classes[c]["w"]), d=list(lib.classes[c]["d"]), top=list(lib.classes[c]["top"]))
                 for c in ("R", "M")},
        collision=dict(kind="inline height field per heap: the library piece's 5 mm grid turned, mirrored and scaled "
                            "vertex for vertex (cell 0.005 m x scale)", piece_cell_m=libmod.GRID, sink_m=libmod.SINK,
                       zero_ring=True, max_equals_size_z=True, elevation_rows="from +y to -y", geom="heap__<band><k>",
                       field="field_<band><k>"),
        rule="hand-laid: one rubble heap (class R) and one mound of fill (class M), scale 1.0, each leaning on a line "
             "over its whole width (the site's south wall; the free-standing wall's south face); occupancy for every "
             "rule = the placed rectangle; the families' keep-outs (pad 0.45 m, search points 1-2 0.40 m, 0.40 m around "
             "the floor points 0.25 m before / behind every ridge opening), not on the dust strip nor on the flat floor, "
             "rectangle spacing 0 or >= W_SWEEP + 0.18 to every other solid (the line not counted; a gap that a third "
             "solid fills from side to side is no gap of open floor), the gap promise < 0.15 m or >= 0.60 m and the "
             "camera's view (>= 0.70 of the surface and the top) judged on the compiled model")
    return grids, pieces


def site_xml(site, marks_xml, grids=None):
    """The family's MJCF (mjcf.world_fragment) plus the calibration marks (visual only) and the heaps (w4gen's
    writing: one inline height field per heap in the asset block, its hfield geom after everything else, so every other
    geom keeps version 1.1's index)."""
    lt = site["lighting"]
    a = lt["ambient"]
    head = ['<mujoco model="cal">', '  <visual>', '    <global offwidth="640" offheight="480"/>',
            '    <quality shadowsize="4096"/>',
            '    <headlight ambient="%s" diffuse="0 0 0" specular="0 0 0"/>' % mjcf._v([a, a, a]),
            '    <map znear="0.01" zfar="50"/>', '  </visual>']
    s = dict(site)
    s["boxes"] = [b for b in site["boxes"] if b["kind"] != "heap"]
    extra = marks_xml
    if site.get("heaps", {}).get("heaps"):
        assets, geoms = w4mjcf.heap_fragments(site, grids)
        head += ['  <asset>'] + ['    ' + q for q in assets] + ['  </asset>']
        extra = marks_xml + "".join("\n    " + g for g in geoms)
    body = ['  <worldbody>', mjcf.world_fragment(s, extra=extra), '  </worldbody>', '</mujoco>']
    return "\n".join(head + body) + "\n"


# ------------------------------------------------------------------------------------------------ predicates
def _rect_pt(r, p):
    return math.hypot(max(r[0] - p[0], 0.0, p[0] - r[1]), max(r[2] - p[1], 0.0, p[1] - r[3]))


def _solids(site, skip=()):
    return [(b["name"], tuple(b["aabb"][:4])) for b in site["boxes"]
            if b["kind"] in gridmod.SOLID_KINDS and b["name"] not in skip]


def crate_rule(site, c, m=None, d=None, cpos=None):
    """FREE_BLOCK_RULE for one crate (box dict): the families' spacing rule and keep-out discs, ground strips, the flat
    floor, the camera rectangle and the camera's view of its top."""
    cls = site["classes"]
    clr = cls["free_block_clearance"]
    a = c["aabb"]
    rect = (a[0], a[1], a[2], a[3])
    w, dd = a[1] - a[0], a[3] - a[2]
    cx, cy = 0.5 * (a[0] + a[1]), 0.5 * (a[2] + a[3])
    pad = site["pad"]["xy"]
    sp = {s["id"]: s["xy"] for s in site["search_points"]}
    keep = [("pad", pad, 0.45), ("point 1", sp[1], 0.40), ("point 2", sp[2], 0.40)]
    for row in ("r1", "r2"):
        for g in site[row]["openings"]:
            gx = 0.5 * (g["x0"] + g["x1"])
            keep += [(g["id"] + " south", (gx, g["y0"] - 0.25), 0.40), (g["id"] + " north", (gx, g["y1"] + 0.25), 0.40)]
    lx, ly = site["size_m"]
    T = fam.WALL_T
    perim = [("perimeter w", (-T, 0.0, -T, ly + T)), ("perimeter e", (lx, lx + T, -T, ly + T)),
             ("perimeter s", (-T, lx + T, -T, 0.0)), ("perimeter n", (-T, lx + T, ly, ly + T))]
    others = perim + [(nm, r) for nm, r in _solids(site, skip=(c["name"], "wall__w", "wall__e", "wall__s", "wall__n"))]
    bad_gap = []
    for nm, o in others:
        if fam._rect_overlap(rect, o):
            bad_gap.append((nm, "overlap"))
            continue
        gp = fam._rect_gap(rect, o)
        if 0.0 < gp < clr - 1e-9:
            bad_gap.append((nm, round(gp, 4)))
    bad_keep = [nm for nm, p, r in keep if not math.hypot(cx - p[0], cy - p[1]) > r + 0.5 * math.hypot(w, dd)]
    on_patch = [p["name"] for p in site["patches"] if fam._rect_overlap(rect, (p["x0"], p["x1"], p["y0"], p["y1"]))]
    on_flat = bool(fam._rect_overlap(rect, (FLAT["x0"], FLAT["x1"], FLAT["y0"], FLAT["y1"])))
    in_cam_rect = bool(a[0] >= 0 and a[1] <= lx and a[2] >= 0 and a[3] <= LY_CAM)
    seen = None
    if m is not None:
        xs = np.linspace(a[0] + 0.02, a[1] - 0.02, 5)
        ys = np.linspace(a[2] + 0.02, a[3] - 0.02, 5)
        P = np.array([(x, y, a[5] + 0.001) for x in xs for y in ys])
        seen = round(float(predicates.visible(m, d, cpos, P).mean()), 4)
    ok = (not bad_gap and not bad_keep and not on_patch and not on_flat and in_cam_rect
          and (seen is None or seen >= 0.90))
    return dict(pass_=bool(ok), name=c["name"], rect=[r6(q) for q in rect], gap_violations=bad_gap,
                keep_out_violations=bad_keep, on_strip=on_patch, on_flat_floor=on_flat, in_camera_rectangle=in_cam_rect,
                top_seen_fraction=seen, clearance_m=clr)


def _screened(a, b, screens):
    """True when one rectangle of `screens` fills the open floor between rectangles a and b from side to side (so
    their gap is no gap of open floor)."""
    xo = min(a[1], b[1]) - max(a[0], b[0])
    yo = min(a[3], b[3]) - max(a[2], b[2])
    if yo > 1e-9:
        reg = (min(a[1], b[1]), max(a[0], b[0]), max(a[2], b[2]), min(a[3], b[3]))
    elif xo > 1e-9:
        reg = (max(a[0], b[0]), min(a[1], b[1]), min(a[3], b[3]), max(a[2], b[2]))
    else:
        return False
    return any(s[0] <= reg[0] + 1e-9 and s[1] >= reg[1] - 1e-9 and s[2] <= reg[2] + 1e-9 and s[3] >= reg[3] - 1e-9
               for s in screens)


def heap_keepouts(site):
    pad = site["pad"]["xy"]
    sp = {s["id"]: s["xy"] for s in site["search_points"]}
    pts = dict(zip(("pad", "point 1", "point 2"), (pad, sp[1], sp[2])))
    keep = [(nm, tuple(pts[nm]), r) for nm, r in HEAP_KEEP]
    for g in site["r1"]["openings"]:
        gx = 0.5 * (g["x0"] + g["x1"])
        keep += [(g["id"] + " south", (gx, g["y0"] - OPEN_OFF), OPEN_KEEP),
                 (g["id"] + " north", (gx, g["y1"] + OPEN_OFF), OPEN_KEEP)]
    return keep


def heap_public(m, d, gid, h):
    """The public numbers of one heap, measured on the compiled model: the bounding rectangle of its outline (a 1 mm
    lattice of vertical-ish rays at its geom alone; outline = more than 1 mm above the floor and not inside a solid:
    a lattice point strictly inside the rectangle of a floor box whose top is at least the heap's height there is
    buried in that box, where neither a ray nor the robot reaches: the height field's roll-off over its last cell
    beyond the line it leans on) and its highest point (the compiled height field's highest vertex: geom z + size[2]
    x max(data))."""
    x0, x1, y0, y1 = h["rect"]
    xs = np.round(np.arange(round(x0 - 0.01, 3), x1 + 0.01 + 1e-9, PUBLIC_STEP), 6)
    ys = np.round(np.arange(round(y0 - 0.01, 3), y1 + 0.01 + 1e-9, PUBLIC_STEP), 6)
    X, Y = np.meshgrid(xs, ys)
    hit = ms.lattice_hits(m, d, gid, np.c_[X.ravel(), Y.ravel()])
    z = hit[:, 2]
    sel = np.isfinite(z) & (z > PUBLIC_Z)
    boxes, tops = ms.floor_boxes(m, d), ms.floor_box_tops(m, d)
    px_all, py_all = X.ravel(), Y.ravel()
    inside = np.zeros(len(z), bool)
    for nm, r in boxes:
        inside |= ((px_all > r[0] + 1e-9) & (px_all < r[1] - 1e-9) & (py_all > r[2] + 1e-9) & (py_all < r[3] - 1e-9)
                   & (tops.get(nm, -1.0) >= np.where(np.isfinite(z), z, 0.0)))
    bur = sel & inside
    keep = sel & ~inside
    px, py = px_all[keep], py_all[keep]
    cg = ms.compiled_grid(m, gid)
    top = float(d.geom_xpos[gid][2]) + cg["max_height_m"]
    beyond = np.maximum.reduce([px_all - x1, x0 - px_all, py_all - y1, y0 - py_all, np.zeros(len(z))])
    bd = float(beyond[bur].max()) if bur.any() else 0.0
    buried_in = sorted({nm for nm, r in boxes if bool((bur & (px_all > r[0] + 1e-9) & (px_all < r[1] - 1e-9)
                                                       & (py_all > r[2] + 1e-9) & (py_all < r[3] - 1e-9)).any())})
    return dict(x0=round(float(px.min()), 3), x1=round(float(px.max()), 3), y0=round(float(py.min()), 3),
                y1=round(float(py.max()), 3), top_height_m=round(top, 3), top_exact_m=r6(top),
                lattice_m=PUBLIC_STEP, outline_above_floor_m=PUBLIC_Z, n_rays=int(len(z)),
                lattice_max_m=round(float(np.nanmax(z)), 6), lattice_max_open_floor_m=round(float(np.max(z[keep])), 6),
                n_buried=int(bur.sum()), buried_beyond_rect_max_m=round(bd, 6), buried_in=buried_in)


def heap_qc(site, xml, lib, grids, pieces):
    """The heaps' predicates on the compiled model (W4's G20-G24 as they apply to a hand-laid site):
      heaps_G20  class geometry (w4gen.heapchecks, by class: rectangle in the class's w / d, compiled footprint inside
                 it to within one placed cell, top in the class's range; R: edge rule >= 95 % on every exposed side and
                 the core; M, natural toe: compiled depth >= 0.28 m, gap to the line <= 0.05 m, scale in the registered
                 interval, the piece's trial verdict; no slope band)
      heaps_G21  keep-outs on the compiled occupancy (5 mm lattice, > 3 mm above the floor), nothing on the dust strip
                 or the flat floor, no chip in a heap's rectangle, rectangle spacing (0 or >= the clearance, the line
                 not counted, a gap filled by a third solid from side to side not counted), the gap promise
                 (< 0.15 m or >= 0.60 m; measure.gap_promise with the heap's line run as one entity)
      heaps_G22  the camera sees >= 0.70 of each heap's surface and its top; no heap under the roof
      heaps_G23  each heap leans on its line (the run of boxes behind it covers its width; its occupancy comes within
                 0.03 m of the line, median per column <= 0.05 m)
      heaps_G24  zero ring, max == size[2], 200 vertices against the piece's grid <= 0.5 mm, the fit bound against the
                 piece's mesh (share within 3 mm >= the class's line, max <= 8 mm), scale in the piece's interval
    and the public numbers (heap_public)."""
    m, d = ms.compile_xml(xml)
    ids = ms.geom_ids(m, "heap__")
    cpos, Rcam = predicates.cam_pose(site)
    cam = (Rcam, site["camera"]["model"])
    heaps = site["heaps"]["heaps"]
    boxes = ms.floor_boxes(m, d)
    clr = site["classes"]["free_block_clearance"]
    keep = heap_keepouts(site)
    solids = [(nm, r) for nm, r in _solids(site) if not nm.startswith("heap__")]
    dust = [p for p in site["patches"] if p["name"] == "dust"]
    gravel = [p for p in site["patches"] if p["name"] == "gravel"]
    roof = site["void"]["slab"]
    q20, q21, q22, q23, q24, pub = {}, {}, {}, {}, {}, {}
    occ, lines = {}, {}

    def n_in(P, r):
        return int(((P[:, 0] > r["x0"]) & (P[:, 0] < r["x1"]) & (P[:, 1] > r["y0"]) & (P[:, 1] < r["y1"])).sum())

    for k, h in enumerate(heaps):
        gid = ids[h["name"]]
        pid = pieces[h["name"]]["piece"]
        piece = lib.pieces[pid]
        H, meta = lib.grid(pid)
        a20, a24 = hc.heap_shape(m, d, gid, h, lib.classes, piece, H, meta, 0, k, lib.mesh(pid))
        line_fit = hc.fit_line(lib.classes, h["cls"])
        a24["fit_line_share_le_3mm"] = line_fit
        a24["pass_"] = bool(a24["zero_ring"] and a24["max_is_size_z"] and a24["vertices"]["nan"] == 0
                            and a24["vertices"]["maxdiff_mm"] <= VERTEX_MM and a24["fit"]["share_le_3mm"] >= line_fit
                            and a24["fit"]["max_mm"] <= FIT_MAX_MM and a24["scale_ok"])
        q20[h["name"]], q24[h["name"]] = a20, a24
        r = tuple(h["rect"])
        P = ms.heap_occupancy(m, d, gid, r)
        occ[h["name"]] = P
        dk = {nm: round(float(np.min(np.hypot(P[:, 0] - p[0], P[:, 1] - p[1]))), 4) for nm, p, _r in keep}
        keep_bad = [nm for nm, _p, rr in keep if dk[nm] < rr - 1e-9]
        if h["line"] == "north":
            near = [(nm, q) for nm, q in solids if q[2] < h["y_line"] - 1e-9]
        else:
            near = [(nm, q) for nm, q in solids if q[3] > h["y_line"] + 1e-9]
        others = near + [(o["name"], tuple(o["rect"])) for o in heaps if o["name"] != h["name"]]
        screens = [q for _nm, q in solids] + [tuple(o["rect"]) for o in heaps if o["name"] != h["name"]]
        bad_sp, screened = [], []
        for nm, o in others:
            if fam._rect_overlap(r, o):
                bad_sp.append((nm, "overlap"))
                continue
            gp = fam._rect_gap(r, o)
            if 0.0 < gp < clr - 1e-9:
                if _screened(r, o, [s for s in screens if s != o]):
                    screened.append((nm, round(gp, 4)))
                else:
                    bad_sp.append((nm, round(gp, 4)))
        chips_in = [c["name"] for c in site.get("chips", []) if r[0] < c["xy"][0] < r[1] and r[2] < c["xy"][1] < r[3]]
        F = dict(x0=FLAT["x0"], x1=FLAT["x1"], y0=FLAT["y0"], y1=FLAT["y1"])
        q21[h["name"]] = dict(keepout_dist_m=dk, keepout_violations=keep_bad, dust_points=sum(n_in(P, p) for p in dust),
                              flat_floor_points=n_in(P, F), gravel_points=sum(n_in(P, p) for p in gravel),
                              chips_in_rect=chips_in, spacing_violations=bad_sp, gaps_filled_by_a_third_solid=screened,
                              n_occupied=int(len(P)))
        run, cover = ms.line_run(boxes, r, h["line"], h["y_line"])
        lines[h["name"]] = run
        per = []
        for x in np.unique(np.round(P[:, 0], 6)):
            yy = P[np.abs(P[:, 0] - x) < 1e-7, 1]
            per.append(h["y_line"] - yy.max() if h["line"] == "north" else yy.min() - h["y_line"])
        per = np.asarray(per)
        lg = dict(min_m=round(float(per.min()), 4), median_m=round(float(np.median(per)), 4))
        q23[h["name"]] = dict(line_run=run, line_cover=round(cover, 6), line_gap=lg,
                              pass_=bool(cover >= 1.0 - 1e-6 and lg["min_m"] <= 0.03 and lg["median_m"] <= 0.05))
        vis = ms.visibility(m, d, gid, r, cpos, cam)
        q22[h["name"]] = dict(visible_share=round(vis["share"], 4), top_seen=vis["top_seen"],
                              in_frame_share=round(vis["in_frame_share"], 4), points_under_roof=n_in(P, roof))
        pub[h["name"]] = heap_public(m, d, gid, h)
    gp = ms.gap_promise(occ, boxes, lines)
    for n, row in q21.items():
        row["gaps"] = gp["pairs"].get(n, {})
        row["gap_ok"] = bool(all(v["ok"] for v in row["gaps"].values()))
        row["pass_"] = bool(not row["keepout_violations"] and row["dust_points"] == 0 and row["flat_floor_points"] == 0
                            and not row["chips_in_rect"] and not row["spacing_violations"] and row["gap_ok"])
    for n, row in q22.items():
        row["pass_"] = bool(row["visible_share"] >= VIS_MIN and row["top_seen"] and row["points_under_roof"] == 0)
    out = {}
    for name, per_h, extra in (("heaps_G20", q20, {}), ("heaps_G21", q21, dict(rule=gp["rule"])), ("heaps_G22", q22, {}),
                               ("heaps_G23", q23, {}),
                               ("heaps_G24", q24, dict(fit_lines={c: hc.fit_line(lib.classes, c) for c in "RM"}))):
        out[name] = dict(pass_=bool(heaps and all(v["pass_"] for v in per_h.values())), heaps=per_h, **extra)
    out["heaps_public"] = dict(heaps=pub, what="public numbers of each heap, measured on the compiled model "
                                               "(outline: 1 mm lattice, > 1 mm above the floor, not inside a solid; top: "
                                               "highest vertex)")
    return out


def blind_route(site, with_block=True):
    """The families' blind route (w3gen: grid shortest path, disc of the plan radius, exact clearance) from the true
    mouth (tunnel_P = opening F) at (its centre, y2 + 0.10) to search point 3, over y >= y2 - 0.30."""
    F = [o for o in site["r1"]["openings"] if o["id"] == site["void"]["true_mouth"]][0]
    y2 = site["params"]["y2"]
    skip = () if with_block else ("rubble__passage_block",)
    rects = [r for _nm, r in _solids(site, skip=skip)]
    # covered openings too low for the robot are solid (gridmod.solid_footprints' rule)
    need = site["envelope"]["values"]["H_ROLL"] + 0.02
    rects += [(g["x0"], g["x1"], g["y0"], g["y1"]) for g in site["r1"]["openings"]
              if g["covered"] and g["clear_h_min"] < need]
    sp3 = [s["xy"] for s in site["search_points"] if s["id"] == 3][0]
    start = (0.5 * (F["x0"] + F["x1"]), y2 + 0.10)
    L, path, _d = w3fam._route(rects, site["size_m"][0], site["size_m"][1], start, tuple(sp3),
                              site["classes"]["plan_radius"], y2 - 0.30)
    return (None if L is None else round(L, 3)), path, [r6(start[0]), r6(start[1])]


def qc(site, xml, render=True):
    """G1 (compiles), G2 (corner margin on the camera rectangle), G3 (projection; rendered), routes pad -> 1 -> 2 ->
    3 -> pad at the families' plan radius (as G9a), the roofed part unseen (room, passage, chamber, point 3's disc and
    the passage block: no sampled point visible, no camera ray hits the block or ends inside them), pad and points
    1, 2 seen, the blind route from opening F to point 3 (window of W3's G8b), the passage block's dimensions (gap =
    the P class = openings C-F, floor to roof, centred, away from the passage's corners), point 3's clearance (W3's
    SP3_CLEAR) and the crates (FREE_BLOCK_RULE)."""
    q = {}
    m, d = predicates.compile_world(xml)
    q["G1"] = dict(pass_=True, compiles=True, ngeom=int(m.ngeom))
    cpos, R = predicates.cam_pose(site)
    margin, _u, _v = fam.corner_margin(cpos, R, LX, LY_CAM, site["camera"]["model"])
    q["G2"] = dict(pass_=bool(margin >= 12.0), corner_margin_px=round(margin, 2), threshold_px=12.0,
                   rectangle="[0, LX] x [0, LY_CAM]")
    if render:
        q["G3"] = w3p.g3_projection(site)
    else:
        q["G3"] = dict(pass_=None, note="not rendered")
    # routes
    G = gridmod.Grid(site, 0.02)
    ok = G.clearance(gridmod.solid_footprints(site)) >= site["classes"]["plan_radius"]
    pts = predicates._points(site)
    legs = G.legs(ok, pts, ("pad", 1, 2, 3, "pad"))
    q["routes"] = dict(pass_=all(l["length_m"] is not None for l in legs), radius_m=site["classes"]["plan_radius"],
                       legs=[dict(leg="%s->%s" % (l["a"], l["b"]), length_m=l["length_m"]) for l in legs])
    # the roofed part unseen
    v = site["void"]
    hz = v["slab"]["z0"]
    rects = [("room", v["runs"]["room"]), ("passage", v["runs"]["passage"]), ("chamber", v["chamber"])]
    blk = v["passage_block"]
    vis_rows = {}
    n_vis = 0
    for name, r in rects:
        xs = np.arange(r["x0"] + 0.025, r["x1"] - 0.02, 0.05)
        ys = np.arange(r["y0"] + 0.025, r["y1"] - 0.02, 0.05)
        P = np.array([(x, y, z) for x in xs for y in ys for z in (0.05, 0.20, 0.30)
                      if not (blk["x0"] - 1e-9 <= x <= blk["x1"] + 1e-9 and blk["y0"] - 1e-9 <= y <= blk["y1"] + 1e-9)])
        vv = predicates.visible(m, d, cpos, P)
        vis_rows[name] = dict(points=int(len(P)), visible=int(vv.sum()))
        n_vis += int(vv.sum())
    sp3 = pts[3]
    P3 = []
    for rr in (0.0, 0.10, 0.20, 0.30):
        for aa in np.linspace(0, 2 * math.pi, 12, endpoint=False):
            for z in (0.05, 0.20, 0.30):
                P3.append((sp3[0] + rr * math.cos(aa), sp3[1] + rr * math.sin(aa), z))
    v3 = predicates.visible(m, d, cpos, np.array(P3))
    vis_rows["point3_disc_0p30"] = dict(points=len(P3), visible=int(v3.sum()))
    n_vis += int(v3.sum())
    # the block's free faces (east, south, north), 1 mm outside
    zs = np.linspace(0.02, hz - 0.02, 6)
    FB = [(blk["x1"] + 0.001, y, z) for y in np.linspace(blk["y0"] + 0.01, blk["y1"] - 0.01, 6) for z in zs]
    FB += [(x, yy, z) for x in np.linspace(blk["x0"] + 0.01, blk["x1"] - 0.01, 6) for yy in (blk["y0"] - 0.001, blk["y1"] + 0.001)
           for z in zs]
    vb = predicates.visible(m, d, cpos, np.array(FB))
    vis_rows["block_faces"] = dict(points=len(FB), visible=int(vb.sum()))
    n_vis += int(vb.sum())
    gid, _dist, hit, _shape = predicates.cast_image(m, d, site, ss=2)
    block_ids = predicates._geom_ids(m, "rubble__passage_block")
    n_block_rays = int(np.isin(gid, block_ids).sum())
    good = gid >= 0
    inside = np.zeros(len(gid), bool)
    for _name, r in rects:
        inside |= good & (hit[:, 0] > r["x0"] + 1e-6) & (hit[:, 0] < r["x1"] - 1e-6) & (hit[:, 1] > r["y0"] + 1e-6) \
            & (hit[:, 1] < r["y1"] - 1e-6) & (hit[:, 2] < hz - 1e-6)
    n_inside = int(inside.sum())
    q["roofed_unseen"] = dict(pass_=bool(n_vis == 0 and n_block_rays == 0 and n_inside == 0), sampled=vis_rows,
                              camera_rays=int(len(gid)), rays_on_block=n_block_rays, rays_ending_inside=n_inside,
                              what="2 x 2 supersampled camera rays; points every 0.05 m at 0.05 / 0.20 / 0.30 m")
    # pad, points 1 and 2 seen
    P1 = []
    for key in ("pad", 1, 2):
        cx, cy = pts[key]
        for rr in (0.0, 0.15, 0.30):
            for aa in np.linspace(0, 2 * math.pi, 8, endpoint=False):
                for z in (0.05, 0.20):
                    P1.append((cx + rr * math.cos(aa), cy + rr * math.sin(aa), z))
    v1 = predicates.visible(m, d, cpos, np.array(P1))
    q["open_points_seen"] = dict(pass_=bool(v1.mean() >= 0.90), visible_fraction=round(float(v1.mean()), 4),
                                 threshold=0.90, what="pad and search points 1, 2: discs of 0.30 m at 0.05 and 0.20 m")
    # the blind route from F's south mouth to point 3
    Lb, path_b, start = blind_route(site, True)
    Ln, _p, _s = blind_route(site, False)
    win = w3fam.ROUTE_WIN
    q["blind_route"] = dict(pass_=bool(Lb is not None and win[0] <= Lb <= win[1]), window_m=list(win), start=start,
                            with_block_m=Lb, without_block_m=Ln,
                            rule="w3gen._route: 0.02 m grid, disc of the plan radius, from (opening F's centre, y2 + "
                                 "0.10), y2 = the ridge's south face, to point 3")
    if path_b is not None:
        pa = np.asarray(path_b)
        sel = (pa[:, 1] >= blk["y0"]) & (pa[:, 1] <= blk["y1"])
        q["blind_route"]["x_beside_block"] = [round(float(pa[sel, 0].min()), 3), round(float(pa[sel, 0].max()), 3)] \
            if sel.any() else None
    # the passage block
    P_w = site["classes"]["P"]["w"]
    pas = v["runs"]["passage"]
    cd = [o for o in site["r1"]["openings"] if o["id"] in ("R1_2", "R1_3", "R1_4", "R1_5")]
    widths = sorted({r6(o["x1"] - o["x0"]) for o in cd})
    gap = r6(pas["x1"] - blk["x1"])
    corners = [(pas["x1"], pas["y0"]), (pas["x1"], pas["y1"])]
    dcorn = [round(_rect_pt((blk["x0"], blk["x1"], blk["y0"], blk["y1"]), c), 4) for c in corners]
    others = [(nm, r) for nm, r in _solids(site, skip=("rubble__passage_block",))
              if fam._rect_overlap(r, (blk["x1"], pas["x1"], blk["y0"], blk["y1"]))]
    centred = abs(0.5 * (blk["y0"] + blk["y1"]) - 0.5 * (pas["y0"] + pas["y1"])) <= TOL
    against_w = abs(blk["x0"] - pas["x0"]) <= TOL
    floor_roof = abs(blk["z0"]) <= TOL and abs(blk["z1"] - hz) <= TOL
    q["passage_block"] = dict(
        pass_=bool(abs(gap - P_w) <= TOL and widths == [r6(P_w)] and centred and against_w and floor_roof
                   and min(dcorn) >= 0.5 - TOL and not others and abs(blk["y1"] - blk["y0"] - BLOCK_ALONG) <= TOL
                   and n_block_rays == 0),
        gap_east_m=gap, P_w=r6(P_w), widths_C_to_F=widths, centred_on_passage=bool(centred), against_west_wall=bool(against_w),
        floor_to_roof=bool(floor_roof), d_corners_east_wall_m=dcorn, corners=[list(c) for c in corners],
        solids_in_gap=[nm for nm, _r in others], camera_rays=n_block_rays)
    # point 3's clearance (W3's rule)
    sol = [r for _nm, r in _solids(site)]
    clr = w3fam._clearance_at(sol, sp3, LX, LY)
    q["sp3_clearance"] = dict(pass_=bool(clr >= w3fam.SP3_CLEAR - 1e-9), clearance_m=round(clr, 4),
                              threshold_m=w3fam.SP3_CLEAR,
                              d_walls_m=dict(east=r6(LX - sp3[0]), back=r6(LY - sp3[1]),
                                             chamber_west=r6(sp3[0] - v["chamber"]["x0"]),
                                             chamber_south=r6(sp3[1] - v["chamber"]["y0"])))
    # crates
    rows = []
    for b in site["boxes"]:
        if b["kind"] == "crate":
            r = crate_rule(site, b, m, d, cpos)
            r["gated"] = any(b["name"].endswith(k) for k in RULED)
            rows.append(r)
    q["crates"] = dict(pass_=all(r["pass_"] for r in rows if r["gated"]), rows=rows, rule=FREE_BLOCK_RULE,
                       gated=list(RULED), note="the red crate is version 1.0's, unchanged; it is not judged by the rule")
    q["pass"] = bool(all(v_.get("pass_") is not False for k, v_ in q.items() if isinstance(v_, dict)))
    q["failed"] = [k for k, v_ in q.items() if isinstance(v_, dict) and v_.get("pass_") is False]
    return q


# ------------------------------------------------------------------------------------------------ public plan
def public_plan(site, heaps_pub=None):
    """Every dimension of the calibration site (the calibration page and its figure), nothing else. heaps_pub = the
    heaps' public numbers measured on the compiled model (qc heaps_public)."""
    def rect(b):
        return dict(x0=b["x0"], x1=b["x1"], y0=b["y0"], y1=b["y1"])
    ops = []
    label = "ABCDEF"
    desc = {"gap_XN": "open gap", "gap_T": "open gap", "gap_P": "open gap", "slot_XL": "covered slot",
            "mouth_blind": "covered mouth with a back wall", "tunnel_P": "covered passage"}
    for k, o in enumerate(site["r1"]["openings"]):
        e = dict(label="opening %s" % label[k], kind=desc[o["type"]], x0=o["x0"], x1=o["x1"], width_m=r6(o["x1"] - o["x0"]),
                 y0=o["y0"], y1=o["y1"])
        if o["covered"]:
            e.update(clear_height_m=o["clear_h"], roof_thickness_m=o["roof_t"])
        if o["type"] == "mouth_blind":
            e.update(depth_to_back_wall_m=o["plug_depth_m"])
        else:
            e.update(length_m=o["depth"])
        if o["type"] == "tunnel_P":
            e.update(leads_to="the roofed room")
        ops.append(e)
    solids = [dict(name=b["name"], kind=b["kind"], x0=b["aabb"][0], x1=b["aabb"][1], y0=b["aabb"][2], y1=b["aabb"][3],
                   height_m=r6(b["aabb"][5])) for b in site["boxes"]
              if (b["kind"] in ("rubble", "support") and b["aabb"][2] == RIDGE_Y[0] and b["aabb"][3] == RIDGE_Y[1])]
    R, P, C = site["room"], site["passage"], site["chamber"]
    blk = P["block"]
    sw, pf, sl = site["side_wall"], site["passage_fill"], site["void"]["slab"]
    plan = dict(
        what="the calibration site of the practice ground: every dimension (metres, site frame: x east, y north, "
             "origin = south-west floor corner)",
        size_m=site["size_m"], perimeter_wall=dict(thickness_m=fam.WALL_T, height_m=fam.WALL_H),
        flat_floor=dict(rect(FLAT), grid_lines=site["marks"]["grid_lines"], line_width_m=site["marks"]["line_width_m"],
                        crosses=site["marks"]["crosses"]),
        pad=dict(xy=site["pad"]["xy"], retrieval_radius_m=site["pad"]["retrieval_radius_m"]),
        search_points=[dict(id=p["id"], xy=p["xy"], where=p["where"]) for p in site["search_points"]],
        ridge=dict(y0=RIDGE_Y[0], y1=RIDGE_Y[1], openings=ops, solids=solids),
        roofed_room=dict(interior=rect(R["interior"]),
                         size_m=[r6(R["interior"]["x1"] - R["interior"]["x0"]), r6(R["interior"]["y1"] - R["interior"]["y0"])],
                         clear_height_m=R["clear_h"], roof_thickness_m=R["slab_t"], wall_thickness_m=R["wall_t"],
                         entrance="opening F (covered passage) on its south side",
                         exit="the roofed passage, from its north-west corner",
                         sides=dict(west="the side wall", east="the site's east wall",
                                    north="the roofed passage (x %s to %s) and the solid east of it (x %s to %s)"
                                          % (P["interior"]["x0"], P["interior"]["x1"], pf["x0"], pf["x1"]),
                                    south="the ridge")),
        roofed_passage=dict(interior=rect(P["interior"]), width_m=P["width_m"], length_m=P["length_m"],
                            clear_height_m=P["clear_h"], leads_from="the roofed room's north-west corner",
                            leads_to="the roofed chamber",
                            sides=dict(west="the side wall", east="the solid east of the passage"),
                            block=dict(kind="rubble block", x0=blk["x0"], x1=blk["x1"], y0=blk["y0"], y1=blk["y1"],
                                       width_m=blk["across_m"], length_m=blk["along_m"], height_m=blk["z1"],
                                       stands="on the floor against the passage's west wall, up to the roof's underside",
                                       clear_width_beside_m=blk["gap_east_m"], beside="east of it")),
        roofed_chamber=dict(interior=rect(C["interior"]), size_m=C["size_m"], clear_height_m=C["clear_h"],
                            sides=dict(west="the side wall", east="the site's east wall", north="the site's back wall",
                                       south="the roofed passage (x %s to %s) and the solid east of it (x %s to %s)"
                                             % (P["interior"]["x0"], P["interior"]["x1"], pf["x0"], pf["x1"]))),
        side_wall=dict(x0=sw["x0"], x1=sw["x1"], y0=sw["y0"], y1=sw["y1"], thickness_m=sw["t"], height_m=sw["h"],
                       note="west of the room, the passage and the chamber; up to the roof's underside"),
        solid_east_of_passage=dict(x0=pf["x0"], x1=pf["x1"], y0=pf["y0"], y1=pf["y1"], height_m=pf["h"],
                                   note="solid up to the roof's underside, from the passage to the site's east wall"),
        roof=dict(x0=sl["x0"], x1=sl["x1"], y0=sl["y0"], y1=sl["y1"], underside_height_m=sl["z0"],
                  thickness_m=r6(sl["z1"] - sl["z0"]), covers="the roofed room, the roofed passage and the roofed chamber"),
        free_standing_wall=dict(x0=FREE_WALL["x0"], x1=FREE_WALL["x1"], y0=FREE_WALL["y0"], y1=FREE_WALL["y1"],
                                height_m=FREE_WALL["h"], length_m=r6(FREE_WALL["x1"] - FREE_WALL["x0"])),
        strips=[dict(name="gravel strip", **rect(GRAVEL)), dict(name="dust strip", **rect(DUST))],
        crates=[dict(colour=b["name"].split("_")[-1], x0=b["aabb"][0], x1=b["aabb"][1], y0=b["aabb"][2], y1=b["aabb"][3],
                     height_m=b["aabb"][5]) for b in site["boxes"] if b["kind"] == "crate"],
        chips=[dict(colour=c["colour"], shape=c["shape"], size_m=c["size_m"], xy=c["xy"]) for c in site["chips"]],
        heaps=[dict(label="heap %d" % (k + 1), kind=h["kind"], x0=heaps_pub[h["name"]]["x0"],
                    x1=heaps_pub[h["name"]]["x1"], y0=heaps_pub[h["name"]]["y0"], y1=heaps_pub[h["name"]]["y1"],
                    top_height_m=heaps_pub[h["name"]]["top_height_m"], stands=h["stands"])
               for k, h in enumerate(site.get("heaps", {}).get("heaps", []))] if heaps_pub else [],
        heaps_what="the bounding rectangle of each heap's outline (where it stands more than 1 mm above the floor) and "
                   "its highest point, measured to 1 mm on the site's collision bodies",
        time_limit_s=None,
        note="dimensions only; the page built from this file states no 'passes' or 'does not pass'",
    )
    return plan


def plan_vs_site(plan, site):
    """Every number of the plan against the site package (boxes by name, openings, points, patches, crates);
    (number of comparisons, list of mismatches)."""
    B = {b["name"]: b["aabb"] for b in site["boxes"]}
    out, n = [], [0]

    def eq(what, a, b):
        n[0] += 1
        if abs(float(a) - float(b)) > TOL:
            out.append("%s: plan %r, site %r" % (what, a, b))

    def box(what, pr, name, h_key=None):
        a = B[name]
        for k, i in (("x0", 0), ("x1", 1), ("y0", 2), ("y1", 3)):
            eq("%s.%s" % (what, k), pr[k], a[i])
        if h_key:
            eq("%s.%s" % (what, h_key), pr[h_key], a[5])
    eq("size_m.x", plan["size_m"][0], site["size_m"][0])
    eq("size_m.y", plan["size_m"][1], site["size_m"][1])
    eq("perimeter.t", plan["perimeter_wall"]["thickness_m"], B["wall__w"][1] - B["wall__w"][0])
    eq("perimeter.h", plan["perimeter_wall"]["height_m"], B["wall__w"][5])
    eq("pad.x", plan["pad"]["xy"][0], site["pad"]["xy"][0])
    eq("pad.y", plan["pad"]["xy"][1], site["pad"]["xy"][1])
    eq("pad.r", plan["pad"]["retrieval_radius_m"], site["pad"]["retrieval_radius_m"])
    for p, s in zip(plan["search_points"], site["search_points"]):
        eq("sp%d.x" % s["id"], p["xy"][0], s["xy"][0])
        eq("sp%d.y" % s["id"], p["xy"][1], s["xy"][1])
    for o, g in zip(plan["ridge"]["openings"], site["r1"]["openings"]):
        for k in ("x0", "x1", "y0", "y1"):
            eq("%s.%s" % (o["label"], k), o[k], g[k])
        eq("%s.width" % o["label"], o["width_m"], g["x1"] - g["x0"])
        if "clear_height_m" in o:
            eq("%s.clear_h" % o["label"], o["clear_height_m"], g["clear_h"])
            roof = B["roof__roof_" + g["id"]]
            eq("%s.roof_t" % o["label"], o["roof_thickness_m"], roof[5] - roof[4])
            eq("%s.roof_z0" % o["label"], o["clear_height_m"], roof[4])
        if "depth_to_back_wall_m" in o:
            eq("%s.depth" % o["label"], o["y0"] + o["depth_to_back_wall_m"], B["plug__plug_" + g["id"]][2])
    for s in plan["ridge"]["solids"]:
        box(s["name"], s, s["name"], "height_m")
    hz = B["roof__void_slab"][4]
    rm, pa, ch = plan["roofed_room"], plan["roofed_passage"], plan["roofed_chamber"]
    for k in ("x0", "x1", "y0", "y1"):
        eq("room." + k, rm["interior"][k], site["void"]["runs"]["room"][k])
        eq("passage." + k, pa["interior"][k], site["void"]["runs"]["passage"][k])
        eq("chamber." + k, ch["interior"][k], site["void"]["chamber"][k])
    eq("room.size_x", rm["size_m"][0], rm["interior"]["x1"] - rm["interior"]["x0"])
    eq("room.size_y", rm["size_m"][1], rm["interior"]["y1"] - rm["interior"]["y0"])
    eq("chamber.size_x", ch["size_m"][0], ch["interior"]["x1"] - ch["interior"]["x0"])
    eq("chamber.size_y", ch["size_m"][1], ch["interior"]["y1"] - ch["interior"]["y0"])
    eq("passage.width", pa["width_m"], pa["interior"]["x1"] - pa["interior"]["x0"])
    eq("passage.length", pa["length_m"], pa["interior"]["y1"] - pa["interior"]["y0"])
    for nm, x in (("room", rm), ("passage", pa), ("chamber", ch)):
        eq(nm + ".clear_h", x["clear_height_m"], hz)
    eq("room.roof_t", rm["roof_thickness_m"], B["roof__void_slab"][5] - B["roof__void_slab"][4])
    eq("room.wall_t", rm["wall_thickness_m"], B["wall__side_w"][1] - B["wall__side_w"][0])
    # the side walls of the roofed part are really where the plan says: west faces, east faces
    eq("room.west_face", rm["interior"]["x0"], B["wall__side_w"][1])
    eq("passage.west_face", pa["interior"]["x0"], B["wall__side_w"][1])
    eq("chamber.west_face", ch["interior"]["x0"], B["wall__side_w"][1])
    eq("passage.east_face", pa["interior"]["x1"], B["wall__passage_fill_e"][0])
    eq("chamber.south_face_east", ch["interior"]["y0"], B["wall__passage_fill_e"][3])
    eq("room.north_face_east", rm["interior"]["y1"], B["wall__passage_fill_e"][2])
    eq("chamber.east_face", ch["interior"]["x1"], B["wall__e"][0])
    eq("chamber.north_face", ch["interior"]["y1"], B["wall__n"][2])
    eq("room.east_face", rm["interior"]["x1"], B["wall__e"][0])
    blk = pa["block"]
    box("passage block", blk, "rubble__passage_block", "height_m")
    eq("block.width", blk["width_m"], blk["x1"] - blk["x0"])
    eq("block.length", blk["length_m"], blk["y1"] - blk["y0"])
    eq("block.gap", blk["clear_width_beside_m"], B["wall__passage_fill_e"][0] - B["rubble__passage_block"][1])
    eq("block.gap = P", blk["clear_width_beside_m"], site["classes"]["P"]["w"])
    eq("block.height = roof underside", blk["height_m"], hz)
    box("side wall", plan["side_wall"], "wall__side_w", "height_m")
    eq("side wall.t", plan["side_wall"]["thickness_m"], B["wall__side_w"][1] - B["wall__side_w"][0])
    box("solid east of passage", plan["solid_east_of_passage"], "wall__passage_fill_e", "height_m")
    box("roof", plan["roof"], "roof__void_slab")
    eq("roof.underside", plan["roof"]["underside_height_m"], hz)
    eq("roof.t", plan["roof"]["thickness_m"], B["roof__void_slab"][5] - B["roof__void_slab"][4])
    fw = plan["free_standing_wall"]
    box("free wall", fw, "wall__free_wall", "height_m")
    eq("free wall.length", fw["length_m"], fw["x1"] - fw["x0"])
    pt = {p["name"]: p for p in site["patches"]}
    for s in plan["strips"]:
        p = pt[s["name"].split()[0]]
        for k in ("x0", "x1", "y0", "y1"):
            eq("%s.%s" % (s["name"], k), s[k], p[k])
    for c in plan["crates"]:
        box("crate " + c["colour"], c, "crate__free_" + c["colour"], "height_m")
    if len(plan["crates"]) != sum(1 for b in site["boxes"] if b["kind"] == "crate"):
        out.append("crates: plan %d, site %d" % (len(plan["crates"]), sum(1 for b in site["boxes"] if b["kind"] == "crate")))
    for c, s in zip(plan["chips"], site["chips"]):
        eq("chip.x", c["xy"][0], s["xy"][0])
        eq("chip.y", c["xy"][1], s["xy"][1])
        eq("chip.size", c["size_m"], s["size_m"])
    fl = plan["flat_floor"]
    for k in ("x0", "x1", "y0", "y1"):
        eq("flat." + k, fl[k], FLAT[k])
    # the heaps: one plan row per heap of the package, its kind, its measured outline inside the placed rectangle (to
    # within one placed cell: the height field's roll-off beyond its last raised vertex, as G20), its highest point =
    # the package's top (to the plan's 1 mm)
    sh = site.get("heaps", {}).get("heaps", [])
    if len(plan.get("heaps", [])) != len(sh):
        out.append("heaps: plan %d, site %d" % (len(plan.get("heaps", [])), len(sh)))
    for p, h in zip(plan.get("heaps", []), sh):
        n[0] += 1
        if p["kind"] != HEAP_KIND[h["cls"]]:
            out.append("%s kind: plan %r, site %r" % (p["label"], p["kind"], h["cls"]))
        tcell = libmod.GRID * float(h["scale"]) + 1e-6     # G20's footprint tolerance: one placed cell (roll-off)
        for k, i, sgn in (("x0", 0, 1), ("x1", 1, -1), ("y0", 2, 1), ("y1", 3, -1)):
            n[0] += 1
            if sgn * (p[k] - h["rect"][i]) < -tcell:
                out.append("%s.%s: plan %r outside the placed rectangle %r by more than one cell"
                           % (p["label"], k, p[k], h["rect"][i]))
        n[0] += 1
        if abs(p["top_height_m"] - h["top_m"]) > 0.0005 + 1e-9:
            out.append("%s.top: plan %r, site %r" % (p["label"], p["top_height_m"], h["top_m"]))
    return n[0], out


# ------------------------------------------------------------------------------------------------ files
def _clean(o):
    if isinstance(o, dict):
        return {(k[:-1] if k.endswith("_") and k != "pass_" else k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    return o


def dumps(o):
    return json.dumps(_clean(o), indent=1, ensure_ascii=True) + "\n"


def write_archive(path, members):
    """A deterministic .tar.xz of {name: bytes} (sorted, mtime 0, uid / gid 0, mode 0644)."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w", format=tarfile.PAX_FORMAT) as tf:
        for name in sorted(members):
            data = members[name]
            ti = tarfile.TarInfo("cal/" + name)
            ti.size, ti.mtime, ti.mode, ti.uid, ti.gid, ti.uname, ti.gname = len(data), 0, 0o644, 0, 0, "", ""
            tf.addfile(ti, io.BytesIO(data))
    import lzma
    raw = lzma.compress(buf.getvalue(), format=lzma.FORMAT_XZ, check=lzma.CHECK_CRC64, preset=6)
    with open(path, "wb") as fh:
        fh.write(raw)
    return hashlib.sha256(raw).hexdigest()


def load_envelope(path=None):
    """The families' envelope (capability sheet), or a local table (--envelope)."""
    if path:
        env = envmod.load(cap_path=None, local_path=path)
        bad = [k for k, s in env["source"].items() if s != "capability_local"]
    else:
        env = envmod.load()
        bad = [k for k, s in env["source"].items() if s != "capability_sheet"]
    if bad:
        raise SystemExit("envelope keys not from the expected table: %s" % bad)
    # no key uses the provisional values: their provenance note does not go into the package
    env = dict(env)
    env.pop("provisional_source", None)
    return env


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--envelope", default=None, help="a local capability table instead of the families' sheet")
    ap.add_argument("--passage-length", type=float, default=PASSAGE_LEN)
    ap.add_argument("--no-render", action="store_true")
    a = ap.parse_args(argv)
    env = load_envelope(a.envelope)
    site, marks_xml = build(env, passage_len=a.passage_length)
    if a.envelope:
        site["generator"] += " [local envelope table: not the calibration site]"
    lib = libmod.Library()
    grids, pieces = place_heaps(site, lib)
    xml = site_xml(site, marks_xml, grids)
    q = qc(site, xml, render=not a.no_render)
    q.update(heap_qc(site, xml, lib, grids, pieces))
    plan = public_plan(site, q["heaps_public"]["heaps"])
    n_cmp, bad = plan_vs_site(plan, site)
    q["plan_vs_site"] = dict(pass_=not bad, compared=n_cmp, mismatches=bad)
    q["pass"] = bool(all(v_.get("pass_") is not False for v_ in q.values() if isinstance(v_, dict)))
    q["failed"] = [k for k, v in q.items() if isinstance(v, dict) and v.get("pass_") is False]
    d = os.path.join(a.out, "cal")
    os.makedirs(d, exist_ok=True)
    sj, sx = dumps(site).encode(), xml.encode()
    with open(os.path.join(d, "site.json"), "wb") as fh:
        fh.write(sj)
    with open(os.path.join(d, "site.xml"), "wb") as fh:
        fh.write(sx)
    with open(os.path.join(d, "qc.json"), "w") as fh:
        fh.write(dumps(dict(generator=site["generator"], envelope_source=site["envelope"]["source"],
                            envelope_files=site["envelope"]["files_read"], **q)))
    sha = write_archive(os.path.join(a.out, "cal.tar.xz"), {"site.json": sj, "site.xml": sx})
    with open(os.path.join(a.out, "plan.json"), "w") as fh:
        fh.write(dumps(plan))
    with open(os.path.join(a.out, "heap_pieces.json"), "w") as fh:
        fh.write(dumps(dict(site="cal", generator=site["generator"], library_sha256=lib.sha256, library_version=lib.version,
                            heaps=[dict(name=nm, **v) for nm, v in pieces.items()])))
    hp = q["heaps_public"]["heaps"]
    print(json.dumps(_clean(dict(pass_=q["pass"], failed=q["failed"], G2=q["G2"]["corner_margin_px"],
                                 G3=q["G3"].get("as_built", {}).get("max_px") if isinstance(q["G3"], dict) else None,
                                 routes=q["routes"], roofed_unseen={k: q["roofed_unseen"][k] for k in
                                                                    ("rays_on_block", "rays_ending_inside")},
                                 blind=q["blind_route"], block=q["passage_block"]["gap_east_m"],
                                 sp3=q["sp3_clearance"]["clearance_m"], crates=[r["pass_"] for r in q["crates"]["rows"]],
                                 heaps={n: dict(G20=q["heaps_G20"]["heaps"][n]["pass_"], G21=q["heaps_G21"]["heaps"][n]["pass_"],
                                                G22=q["heaps_G22"]["heaps"][n]["visible_share"],
                                                G23=q["heaps_G23"]["heaps"][n]["pass_"],
                                                G24=q["heaps_G24"]["heaps"][n]["fit"]["share_le_3mm"],
                                                public=[hp[n][k] for k in ("x0", "x1", "y0", "y1", "top_height_m")])
                                        for n in hp},
                                 plan_vs_site=[n_cmp, len(bad)], archive_sha256=sha))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
