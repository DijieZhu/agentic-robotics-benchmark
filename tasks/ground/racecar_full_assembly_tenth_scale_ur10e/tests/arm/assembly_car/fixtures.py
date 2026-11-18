"""The race-car cell's fixtures: the work stand that holds the rolling chassis, the deck shelf, the parts tray with its
two nests, and the datum boards that lie on them (grader side; the drawings and the photo manual are made from
`export()`).

Every fixture is a static body made of boxes, in its own frame: origin on the bench top (z = 0 is the bench top),
x and y as named below, z up; lengths in mm here, metres in the cell. Its pose in the cell is a turn about the
vertical and a shift on the bench (`layout.py` draws it per episode).

* **work stand** ("stand"; frame: origin under the car's reference point, the midpoint between the axles at axle
  height, x = the car's forward axis): a base plate on the bench, a 60 x 60 post, a cradle plate on which the tub's
  flat underside is clamped (its top = the tub's underside, the wheel axles AXLE_HEIGHT_MM above the bench), and a
  tab of the base plate reaching forward beyond the car's nose with the stand board on it (in front of the nose so
  that the overhead camera sees it). The stand stays inside |y| <= 64 mm above the base plate: nothing of it is in
  the lower finger's path beside a wheel.
* **deck shelf** ("shelf"; frame: origin under the deck frame's origin as the deck lies on the shelf, x = the deck's
  forward axis): a foot plate, a pedestal, two rails under the deck plate's underside (the right long edge overhangs
  the nearer rail by ~47 mm: the gripper's linkage hangs there), a back fence along the left long edge and two end
  stops, each 1.5 mm off the plate (the play), and the shelf board on the foot beyond the deck's rear end.
* **parts tray** ("tray"; frame: origin under the plate's centre): a 12 mm plate with the tray board and two nests:
  the compute module stands on its eight standoffs, turned 90 deg from its mounting direction, ribs 1.5 mm off its
  standoffs; the LiDAR stands on its base, turned 180 deg, ribs 1.5 mm off its base (none where its connector block
  hangs out). The ribs are SHORTER than the compute module's lowest drawn part (its thin bottom plate, 5.78 mm above
  the feet's plane), so nothing of the drawn module passes through them. Clearance holes for the screw ends under the
  standoffs are cut only in the exported mesh (the screw ends have no collision; the cell's plate is a solid box).

The rib and stop positions are fixed numbers of the fixture (they are drawn and photographed); they were set from
the collision shapes of the kit's units (the compute module's eight foot cylinders, r 1.392 mm; the LiDAR's square
base, 40.802 mm) and `nest_report(kit)` re-measures the play against any kit (the tests require 1.5 mm).

Datum boards (`BOARDS`): a checkerboard printed on a rigid plate (render.scale_refs.checkerboard: the pattern's
origin in the middle, x along the columns, the printed face 0.05 mm above the plate's top), lying flat on a face of
its fixture. In the physics they are not modelled (nothing rests on them); the render twin attaches them at
`board_pose(name)` (the centre of the face the plate rests on, +z out of it, x along the columns), and `export()`
writes them for the drawings and the photos.
"""
import json
import math
import os
import struct

import numpy as np

__all__ = ["BENCH_BOARD", "TOOL_PLATE", "AXLE_HEIGHT_MM", "REF_POINT_CAR_MM", "TUB_BOTTOM_Z_CAR_MM", "STAND", "SHELF", "TRAY", "BOARDS", "FIXTURES",
           "T_stand_car", "fixture_boxes", "nominal_unit_poses", "board_pose", "board_face_height_mm", "boards",
           "footprint_xy", "nest_report", "shelf_report", "export"]

# ------------------------------------------------------------------ the car on the stand
# the car frame's reference point `wheelbase_midpoint_at_axle_height` (unit map v2, car_frame.reference_points_mm) and
# the tub's flat underside (z of the car frame; = the cradle plate's top)
REF_POINT_CAR_MM = (128.922, 0.0, 32.901)
TUB_BOTTOM_Z_CAR_MM = 14.690
AXLE_HEIGHT_MM = 200.0            # the reference point above the bench (the stand's height H = 0.20 m)


def T_stand_car():
    """4x4 (m) of the car frame in the stand's frame: the reference point AXLE_HEIGHT_MM above the stand's origin"""
    T = np.eye(4)
    T[:3, 3] = np.array([-REF_POINT_CAR_MM[0], -REF_POINT_CAR_MM[1], AXLE_HEIGHT_MM - REF_POINT_CAR_MM[2]]) * 1e-3
    return T


_Z_CAR = AXLE_HEIGHT_MM - REF_POINT_CAR_MM[2]          # 167.099: the car frame's z = 0 in the stand frame
_X_CAR = -REF_POINT_CAR_MM[0]                         # -128.922: the car frame's x = 0 in the stand frame
CRADLE_TOP_MM = _Z_CAR + TUB_BOTTOM_Z_CAR_MM            # 181.789

# boxes: (name, lo, hi, part) in the fixture frame, mm
STAND = {
    "name": "stand",
    "label": "work stand",
    "frame": "origin on the bench top under the car's reference point (midway between the axles); x = the car's "
             "forward axis, z up",
    "boxes": [
        # the base plate under the car (car x 0..260, y +-110) and its tab forward of the nose (car x 260..455, y +-65)
        ("base", [_X_CAR + 0.0, -110.0, 0.0], [_X_CAR + 260.0, 110.0, 15.0], "base plate"),
        ("tab", [_X_CAR + 260.0, -65.0, 0.0], [_X_CAR + 455.0, 65.0, 15.0], "base plate (tab with the stand board)"),
        # the 60 x 60 post (car x 99..159) and the cradle plate (car x 40..218, y +-45) under the tub's flat underside
        ("post", [_X_CAR + 99.0, -30.0, 15.0], [_X_CAR + 159.0, 30.0, CRADLE_TOP_MM - 10.0], "post"),
        ("cradle", [_X_CAR + 40.0, -45.0, CRADLE_TOP_MM - 10.0], [_X_CAR + 218.0, 45.0, CRADLE_TOP_MM], "cradle plate"),
    ],
}

# the deck plate on the shelf (unit map v2: DCK top z 70.1621, bottom 64.9747; outline x -126.456..126.624, y +-64.78;
# the nose and the tail are narrow: |y| <= 17.0 at the front edge, <= 17.6 at the rear edge)
DECK_TOP_MM = 140.0
DECK_THICK_MM = 70.1621 - 64.9747
DECK_UNDER_MM = DECK_TOP_MM - DECK_THICK_MM            # 134.8126: the rails' top
DECK_EDGE_MM = {"front": 126.624, "rear": -126.456, "left": 64.78, "right": -64.78}
SHELF_PLAY_MM = 1.5
SHELF = {
    "name": "shelf",
    "label": "deck shelf",
    "frame": "origin on the bench top under the deck frame's origin as the deck lies on the shelf at the middle of its "
             "play; x = the deck's forward axis, z up",
    "boxes": [
        ("foot", [-240.0, -30.0, 0.0], [145.0, 80.0, 10.0], "foot plate"),
        ("pedestal", [-90.0, -15.0, 10.0], [90.0, 55.0, DECK_UNDER_MM - 20.0], "pedestal"),
        # rails under the plate: clear of the power board's screw ends (deck y +-29.4..32.2) and >= 45 mm from the
        # right long edge (the 2F-85's linkage hangs below the plate there)
        ("rail_a", [-110.0, -17.5, DECK_UNDER_MM - 20.0], [110.0, -2.5, DECK_UNDER_MM], "rail"),
        ("rail_b", [-110.0, 40.0, DECK_UNDER_MM - 20.0], [110.0, 55.0, DECK_UNDER_MM], "rail"),
        # the back fence along the left long edge and the two end stops: 1.5 mm off the plate, 1.19 mm below its top
        ("fence", [-100.0, DECK_EDGE_MM["left"] + SHELF_PLAY_MM, 10.0], [100.0, DECK_EDGE_MM["left"] + SHELF_PLAY_MM + 8.0,
                                                                         DECK_UNDER_MM + 4.0], "back fence"),
        ("stop_front", [DECK_EDGE_MM["front"] + SHELF_PLAY_MM, -15.0, 10.0], [DECK_EDGE_MM["front"] + SHELF_PLAY_MM + 8.0, 15.0,
                                                                              DECK_UNDER_MM + 4.0], "end stop"),
        ("stop_rear", [DECK_EDGE_MM["rear"] - SHELF_PLAY_MM - 8.0, -15.0, 10.0], [DECK_EDGE_MM["rear"] - SHELF_PLAY_MM, 15.0,
                                                                                  DECK_UNDER_MM + 4.0], "end stop"),
    ],
}

# the parts tray: plate, ribs (RIB_MM high, 3 mm thick, 6 mm long; the LiDAR's two rear ribs 3.4 mm long)
TRAY_PLATE_MM = (240.0, 210.0, 12.0)
RIB_MM = 5.5
NEST_PLAY_MM = 1.5
NEST_PLAY_DEG = 2.0
# the units in their nests at the middle of the play: (x, y, yaw deg) in the tray frame, on the plate's top. The
# compute module turned 90 deg and the LiDAR 180 deg from their mounting direction (at the nominal placements the car
# faces the cell's -x and the tray the cell's +x: the compute module's x along the cell's +y, the LiDAR's along +x)
NEST_POSE = {"CMP": (0.0, 55.0, 90.0), "LDR": (75.0, -50.0, 0.0)}
STORAGE_TURN_DEG = {"CMP": 90.0, "LDR": 180.0}


def _rib(name, along, ax, half_len, face, out, thick=3.0, part="locating rib"):
    """a rib whose locating face lies on the line {ax} = face, the rib on the side `out` (+1 / -1) of it (away from the
    unit), centred at `along` on the other axis"""
    a0, a1 = sorted((face, face + out * thick))
    b0, b1 = along - half_len, along + half_len
    x0, x1, y0, y1 = (a0, a1, b0, b1) if ax == "x" else (b0, b1, a0, a1)
    return (name, [x0, y0, TRAY_PLATE_MM[2]], [x1, y1, TRAY_PLATE_MM[2] + RIB_MM], part)


def _tray_boxes():
    t = TRAY_PLATE_MM
    out = [("plate", [-t[0] / 2, -t[1] / 2, 0.0], [t[0] / 2, t[1] / 2, t[2]], "tray plate")]
    # compute module nest: the standoffs' extremes (tray frame, relative to the unit's origin, yaw 90): x -41.386
    # (two feet at y +-17.14), x +32.092 (feet at y +35.17, -35.07), y +36.566 (feet at x -21.27, +30.52),
    # y -36.644 (feet at x -21.18, +30.70)
    cx, cy, _ = NEST_POSE["CMP"]
    g = NEST_PLAY_MM
    out += [_rib("cmp_xm1", cy + 17.14, "x", 3.0, cx - 41.386 - g, -1),
            _rib("cmp_xm2", cy - 17.155, "x", 3.0, cx - 41.386 - g, -1),
            _rib("cmp_xp1", cy + 35.174, "x", 3.0, cx + 32.092 + g, +1),
            _rib("cmp_xp2", cy - 35.074, "x", 3.0, cx + 32.092 + g, +1),
            _rib("cmp_yp1", cx - 21.273, "y", 3.0, cy + 36.566 + g, +1),
            _rib("cmp_yp2", cx + 30.522, "y", 3.0, cy + 36.566 + g, +1),
            _rib("cmp_ym1", cx - 21.184, "y", 3.0, cy - 36.644 - g, -1),
            _rib("cmp_ym2", cx + 30.700, "y", 3.0, cy - 36.644 - g, -1)]
    # LiDAR nest: the square base +-20.401; two ribs per side at +-12 mm, the rear (-x) ones at |y| 17.0..20.4 (clear
    # of the connector block, |y| <= 13.9 as drawn)
    lx, ly, _ = NEST_POSE["LDR"]
    s = 20.401
    out += [_rib("ldr_xp1", ly + 12.0, "x", 3.0, lx + s + g, +1),
            _rib("ldr_xp2", ly - 12.0, "x", 3.0, lx + s + g, +1),
            _rib("ldr_yp1", lx + 12.0, "y", 3.0, ly + s + g, +1),
            _rib("ldr_yp2", lx - 12.0, "y", 3.0, ly + s + g, +1),
            _rib("ldr_ym1", lx + 12.0, "y", 3.0, ly - s - g, -1),
            _rib("ldr_ym2", lx - 12.0, "y", 3.0, ly - s - g, -1),
            _rib("ldr_xm1", ly + 18.7, "x", 1.7, lx - s - g, -1),
            _rib("ldr_xm2", ly - 18.7, "x", 1.7, lx - s - g, -1)]
    return out


TRAY = {
    "name": "tray",
    "label": "parts tray",
    "frame": "origin on the bench top under the plate's centre; x along the plate's 240 mm side, z up",
    "boxes": _tray_boxes(),
    # the compute module's standoffs (CMP frame, the kit's foot cylinders): the clearance holes of the exported plate
    "feet_cmp_mm": [(35.174, 21.273), (35.174, -30.522), (17.12, 39.994), (17.167, -30.611), (-17.155, 39.994),
                    (-17.155, -30.611), (-35.252, 21.184), (-35.074, -30.7)],
    "foot_radius_mm": 1.392,
    "hole_diameter_mm": 7.0, "hole_depth_mm": 8.0,
}

FIXTURES = {"stand": STAND, "shelf": SHELF, "tray": TRAY}
MATERIAL = {"stand": [0.62, 0.64, 0.67, 1.0], "shelf": [0.55, 0.58, 0.62, 1.0], "tray": [0.20, 0.22, 0.25, 1.0]}

# ------------------------------------------------------------------ datum boards
# the two surveyed boards, as in the earlier assembly cell (arm/assembly_quad/server.py BENCH_BOARD, TOOL_PLATE): the bench
# board flat on the bench (its pattern's origin at pos, x along the cell's x), the tool plate on the wrist camera's housing
BENCH_BOARD = {"cols": 9, "rows": 6, "square_mm": 20.0, "margin_mm": 20.0, "thickness_mm": 3.0, "pos": [1.20, -0.50, 0.0],
               "yaw_deg": 0.0}
TOOL_PLATE = {"cols": 5, "rows": 4, "square_mm": 6.0, "margin_mm": 3.0, "thickness_mm": 1.5}
# rest: (x, y) of the board's centre on the face it lies on (fixture frame), the face's z, the turn of the board's x
# (its columns) from the fixture's x, anticlockwise seen from above
BOARDS = {
    "stand": {"fixture": "stand", "cols": 6, "rows": 5, "square_mm": 15.0, "margin_mm": 15.0, "thickness_mm": 3.0,
              "rest_xy_mm": [_X_CAR + 398.922, 0.0], "rest_z_mm": 15.0, "yaw_deg": 90.0,
              "where": "on the stand's base plate tab, ahead of the car's nose"},
    "shelf": {"fixture": "shelf", "cols": 5, "rows": 4, "square_mm": 12.0, "margin_mm": 12.0, "thickness_mm": 3.0,
              "rest_xy_mm": [-195.0, 20.0], "rest_z_mm": 10.0, "yaw_deg": 90.0,
              "where": "on the shelf's foot plate, beyond the deck's rear end"},
    "tray": {"fixture": "tray", "cols": 6, "rows": 5, "square_mm": 15.0, "margin_mm": 15.0, "thickness_mm": 3.0,
             "rest_xy_mm": [-55.0, -50.0], "rest_z_mm": TRAY_PLATE_MM[2], "yaw_deg": 0.0,
             "where": "on the tray plate, beside the nests"},
}
PRINT_ABOVE_PLATE_MM = 0.05        # render.scale_refs: the printed face above the board plate's top


def _rotz(deg):
    a = math.radians(float(deg))
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def board_pose(name):
    """4x4 (m) in the fixture frame: the centre of the face the board's plate rests on, +z out of it, x along the
    board's columns (render twin: assembly_info.cell_render.attach_ref)"""
    b = BOARDS[name]
    T = np.eye(4)
    T[:3, :3] = _rotz(b["yaw_deg"])
    T[:3, 3] = [b["rest_xy_mm"][0] * 1e-3, b["rest_xy_mm"][1] * 1e-3, b["rest_z_mm"] * 1e-3]
    return T


def board_size_mm(name):
    b = BOARDS[name]
    return (b["cols"] * b["square_mm"] + 2 * b["margin_mm"], b["rows"] * b["square_mm"] + 2 * b["margin_mm"], b["thickness_mm"])


def board_face_height_mm(name):
    """the printed face above the bench top (mm)"""
    b = BOARDS[name]
    return b["rest_z_mm"] + b["thickness_mm"] + PRINT_ABOVE_PLATE_MM


def boards():
    """{name: {fixture, T_rest (m, fixture frame), cols, rows, square_mm, margin_mm, thickness_mm, plate_mm,
    face_height_mm}} for the render twin and the documents"""
    out = {}
    for n, b in BOARDS.items():
        out[n] = {"fixture": b["fixture"], "T_rest": board_pose(n), "cols": b["cols"], "rows": b["rows"],
                  "square_mm": b["square_mm"], "margin_mm": b["margin_mm"], "thickness_mm": b["thickness_mm"],
                  "plate_mm": list(board_size_mm(n)), "face_height_mm": board_face_height_mm(n), "where": b["where"]}
    return out


# ------------------------------------------------------------------ geometry helpers
def fixture_boxes(name):
    """[(name, centre m, half sizes m, part)] of a fixture in its frame (MuJoCo boxes)"""
    out = []
    for bn, lo, hi, part in FIXTURES[name]["boxes"]:
        lo, hi = np.asarray(lo, float) * 1e-3, np.asarray(hi, float) * 1e-3
        out.append((bn, (0.5 * (lo + hi)).tolist(), (0.5 * (hi - lo)).tolist(), part))
    return out


def nominal_unit_poses():
    """{unit: (fixture, 4x4 m in the fixture frame)} of the units that start on a fixture, at the middle of its play:
    the deck on the shelf (its frame's axes = the shelf's; its plate's underside on the rails), the compute module and
    the LiDAR on the tray plate in their nests (the feet / base on the plate)"""
    out = {}
    T = np.eye(4)
    T[:3, 3] = [0.0, 0.0, DECK_TOP_MM * 1e-3]
    out["DCK"] = ("shelf", T)
    for u in ("CMP", "LDR"):
        x, y, yaw = NEST_POSE[u]
        T = np.eye(4)
        T[:3, :3] = _rotz(yaw)
        T[:3, 3] = [x * 1e-3, y * 1e-3, TRAY_PLATE_MM[2] * 1e-3]
        out[u] = ("tray", T)
    return out


def footprint_xy(name, T_world=None, extra_mm=0.0):
    """[(4, 2) array] the plan rectangles (m) of the fixture's boxes in the world (or fixture frame), grown by extra_mm"""
    T_world = np.eye(4) if T_world is None else np.asarray(T_world, float)
    out = []
    e = float(extra_mm)
    for _bn, lo, hi, _p in FIXTURES[name]["boxes"]:
        P = np.array([[lo[0] - e, lo[1] - e], [hi[0] + e, lo[1] - e], [hi[0] + e, hi[1] + e], [lo[0] - e, hi[1] + e]]) * 1e-3
        out.append(P @ T_world[:2, :2].T + T_world[:2, 3])
    return out


def _unit_pieces(kit, unit):
    """the unit's collision pieces' points in its frame (m): mesh vertices, box corners, cylinder rims"""
    from .layout import unit_pieces
    return unit_pieces(kit, units=[unit])[unit]


def nest_report(kit):
    """the play of the compute module and the LiDAR in their nests, measured on the kit's collision shapes at the
    nests' nominal poses: {unit: {"min_gap_mm", "per_rib_mm": {rib: plan distance of the rib to the nearest plan hull
    of the unit's collision pieces that reach below the ribs' top}, "pieces_below_rib_top", "lowest_mm"}}"""
    from .layout import _hull2, _rect, poly_dist
    out = {}
    ribs = [(bn, lo, hi) for bn, lo, hi, _p in TRAY["boxes"] if bn != "plate"]
    for u, pre in (("CMP", "cmp_"), ("LDR", "ldr_")):
        x, y, yaw = NEST_POSE[u]
        R, t = _rotz(yaw), np.array([x, y, 0.0])
        pcs = [q * 1e3 @ R.T + t for q in _unit_pieces(kit, u)]
        low = [_hull2(q) for q in pcs if q[:, 2].min() <= RIB_MM + 1e-6]
        per = {bn: min(poly_dist(H, _rect(lo[0], hi[0], lo[1], hi[1])) for H in low) for bn, lo, hi in ribs if bn.startswith(pre)}
        out[u] = {"min_gap_mm": round(min(per.values()), 4), "per_rib_mm": {k: round(v, 4) for k, v in per.items()},
                  "pieces_below_rib_top": int(len(low)), "lowest_mm": round(float(min(q[:, 2].min() for q in pcs)), 4)}
    return out


def shelf_report(kit):
    """the deck's plate against the shelf at the middle of the play: the plan distances of the convex hull of the
    kit's DCK collision points at the fence / stops' height to the fence and the end stops, the rails' top vs the
    lowest DCK point, the overhang of the right long edge beyond the nearer rail (per collision piece)"""
    from .layout import _hull2, _rect, poly_dist
    pcs = [q * 1e3 + np.array([0.0, 0.0, DECK_TOP_MM]) for q in _unit_pieces(kit, "DCK")]
    V = np.concatenate(pcs)
    out = {}
    boxes = {bn: (lo, hi) for bn, lo, hi, _p in SHELF["boxes"]}
    for bn in ("fence", "stop_front", "stop_rear"):
        lo, hi = boxes[bn]
        sel = [q for q in pcs if q[:, 2].min() < hi[2] and q[:, 2].max() > lo[2]]
        out[bn + "_gap_mm"] = round(min(poly_dist(_hull2(q), _rect(lo[0], hi[0], lo[1], hi[1])) for q in sel), 4)
    out["lowest_mm"] = round(float(V[:, 2].min()), 4)
    out["rail_top_mm"] = round(DECK_UNDER_MM, 4)
    out["overhang_right_mm"] = round(boxes["rail_a"][0][1] - float(V[:, 1].min()), 3)
    return out


# ------------------------------------------------------------------ export (drawings, photos)
STL_HEADER = b"assembly cell fixture, millimetres, fixture frame"


def _box_mesh(lo, hi):
    x0, y0, z0 = lo
    x1, y1, z1 = hi
    V = np.array([[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0], [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]], float)
    F = np.array([[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7], [0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5], [2, 3, 7], [2, 7, 6],
                  [3, 0, 4], [3, 4, 7]], int)
    return V, F


def _write_stl(path, V, F):
    V = np.asarray(V, np.float32)
    F = np.asarray(F, int)
    tri = V[F]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    ln = np.linalg.norm(n, axis=1, keepdims=True)
    n = (n / np.where(ln > 0, ln, 1.0)).astype(np.float32)
    rec = np.zeros(len(F), dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]))
    rec["n"], rec["v"] = n, tri
    with open(path, "wb") as f:
        f.write(STL_HEADER.ljust(80, b" ")[:80])
        f.write(struct.pack("<I", len(F)))
        f.write(rec.tobytes())


def _tray_plate_mesh():
    """the tray plate with the compute module's eight clearance holes (manifold3d), (V, F, holes_cut)"""
    t = TRAY_PLATE_MM
    lo, hi = [-t[0] / 2, -t[1] / 2, 0.0], [t[0] / 2, t[1] / 2, t[2]]
    try:
        import manifold3d as M3
    except Exception:
        V, F = _box_mesh(lo, hi)
        return V, F, False
    plate = M3.Manifold.cube([hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]]).translate(lo)
    cx, cy, yaw = NEST_POSE["CMP"]
    R = _rotz(yaw)
    d, depth = TRAY["hole_diameter_mm"], TRAY["hole_depth_mm"]
    for fx, fy in TRAY["feet_cmp_mm"]:
        p = R @ np.array([fx, fy, 0.0]) + np.array([cx, cy, 0.0])
        hole = M3.Manifold.cylinder(depth + 1.0, d / 2, d / 2, 32).translate([float(p[0]), float(p[1]), t[2] - depth])
        plate = plate - hole
    mesh = plate.to_mesh()
    V = np.asarray(mesh.vert_properties, float)[:, :3]
    F = np.asarray(mesh.tri_verts, int)
    return V, F, True


def export(out_dir, kit=None):
    """write the fixtures for the drawings and the photo manual: <out_dir>/fixtures.json (every box, board and nominal
    unit pose in the fixture frames, mm; the plays, the rib height, the holes; the stand's car frame) and per fixture
    <name>.stl (binary, mm, fixture frame: every box; the tray plate with its clearance holes) plus
    board_<name>.stl (the board plates, mm, fixture frame). With a kit: the nests' and the shelf's measured plays.
    Returns the json dict."""
    os.makedirs(out_dir, exist_ok=True)
    doc = {"format": "racecar-cell-fixtures/1", "units": "mm, deg", "z": "0 = the bench top",
           "car_frame_in_stand_mm": {"x": _X_CAR, "y": 0.0, "z": _Z_CAR,
                                     "note": "the car frame (origin on the ground plane under the rear axle's middle, x forward) "
                                             "in the stand's frame: parallel axes, offset by this"},
           "axle_height_mm": AXLE_HEIGHT_MM, "cradle_top_mm": CRADLE_TOP_MM, "deck_top_mm": DECK_TOP_MM,
           "shelf_play_mm": SHELF_PLAY_MM, "rib_mm": RIB_MM, "nest_play_mm": NEST_PLAY_MM, "nest_play_deg": NEST_PLAY_DEG,
           "storage_turn_deg": dict(STORAGE_TURN_DEG), "fixtures": {}, "boards": {}, "units": {}}
    files = {}
    for name, fx in FIXTURES.items():
        doc["fixtures"][name] = {"label": fx["label"], "frame": fx["frame"],
                                 "boxes": [{"name": bn, "lo": [round(v, 4) for v in lo], "hi": [round(v, 4) for v in hi], "part": p}
                                           for bn, lo, hi, p in fx["boxes"]]}
        Vs, Fs, off = [], [], 0
        for bn, lo, hi, _p in fx["boxes"]:
            if name == "tray" and bn == "plate":
                V, F, cut = _tray_plate_mesh()
                doc["fixtures"][name]["plate_holes_cut"] = bool(cut)
            else:
                V, F = _box_mesh(lo, hi)
            Vs.append(V)
            Fs.append(F + off)
            off += len(V)
        p = os.path.join(out_dir, name + ".stl")
        _write_stl(p, np.concatenate(Vs), np.concatenate(Fs))
        files[name + ".stl"] = p
    doc["fixtures"]["tray"]["clearance_holes"] = {"diameter_mm": TRAY["hole_diameter_mm"], "depth_mm": TRAY["hole_depth_mm"],
                                                  "centres_mm": [[round(float(v), 4) for v in (_rotz(NEST_POSE["CMP"][2]) @ np.array([fx_, fy_, 0.0]))[:2]
                                                                  + np.array(NEST_POSE["CMP"][:2])] for fx_, fy_ in TRAY["feet_cmp_mm"]]}
    for n, b in boards().items():
        T = b["T_rest"]
        bx, by, th = b["plate_mm"]
        V, F = _box_mesh([-bx / 2, -by / 2, 0.0], [bx / 2, by / 2, th])
        V = V @ T[:3, :3].T + T[:3, 3] * 1e3
        p = os.path.join(out_dir, "board_%s.stl" % n)
        _write_stl(p, V, F)
        files["board_%s.stl" % n] = p
        doc["boards"][n] = {"fixture": b["fixture"], "cols": b["cols"], "rows": b["rows"], "square_mm": b["square_mm"],
                            "margin_mm": b["margin_mm"], "thickness_mm": b["thickness_mm"], "plate_mm": b["plate_mm"],
                            "rest_centre_mm": [round(float(v) * 1e3, 4) for v in T[:3, 3]],
                            "x_axis_turn_deg": BOARDS[n]["yaw_deg"], "face_height_mm": round(b["face_height_mm"], 4),
                            "where": b["where"],
                            "pattern": "render.scale_refs.checkerboard: origin at the pattern's middle, x along the columns, the "
                                       "square at the board's (-x, +y) corner black"}
    for u, (fxn, T) in nominal_unit_poses().items():
        yaw = math.degrees(math.atan2(T[1, 0], T[0, 0]))
        doc["units"][u] = {"fixture": fxn, "origin_mm": [round(float(v) * 1e3, 4) for v in T[:3, 3]], "yaw_deg": round(yaw, 6),
                           "note": "the unit's body frame (kit) at the middle of its play"}
    if kit is not None:
        doc["measured"] = {"nests": nest_report(kit), "shelf": shelf_report(kit)}
    with open(os.path.join(out_dir, "fixtures.json"), "w") as f:
        json.dump(doc, f, indent=1, sort_keys=True)
        f.write("\n")
    doc["files"] = sorted(files)
    return doc
