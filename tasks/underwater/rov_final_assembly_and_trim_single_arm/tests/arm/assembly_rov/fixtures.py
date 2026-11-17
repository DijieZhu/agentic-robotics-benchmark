"""The final-assembly cell's fixtures (grader side): the turnover positioner's open-frame base and fork, the ballast
box, the thruster tray, the enclosure rack, the two guard racks, and the datum boards that lie on the bench and on the
tray. The drawings and the photo manual are made from `export()`; the cell builds its static bodies from
`station_boxes()` and `positioner_boxes()`; `layout.py` draws their per-episode poses.

Inputs (read at run time, never typed in here):
* the frozen cell layout (`cell_layout.json`, schema rov-cell-layout/1): the positioner's placement, its boxes
  (`derived.turning_boxes_B_mm`, `derived.static_boxes_cell_m`), the rotation column, the load cells' stations, every
  station's nominal pose, jitter and inner dimensions, the duct / block / guard dimensions it was frozen on;
* the fixture facts (`fixture_facts.json`, schema rov-fixture-facts/1, generated in the development tree from the
  vehicle's unit map): the units' body frames in the vehicle frame, the tube's radius, the guard's edges, the
  vehicle's box with every unit at home, the swept radii, coarse convex hulls of the parts (footprints).
Sources (copied, then changed): the station builders of the frozen layout's study (P06)
  -dev/study/v3/cell_v3.py sha256 2ce4a5cd654a9572885463e245c2b422fa6bb7ca883d3725ba91410f1e0f997c (the same boxes,
  read from the layout and the facts instead of the digest; + the tray's board tab, the L-mag box tab, the rest poses in
  the units' body frames); the export / board conventions of the race-car cell
  docs/plan/driving/tasks/racecar_full_assembly_tenth_scale_ur10e/tests/arm/assembly_car/fixtures.py
  sha256 01a1ed1b180c662f54d2e9cd13e1b6ebb326a6295f7f5d5fc5ef1521c19a6e07.
The numbers written below are the fixtures' own design values (plays, boards, the tray's board tab, the structure's
masses, the load cells' spring and damper): they are drawn and printed as they are.

Frames: the cell frame (m, origin at the arm's base, z up); the vehicle frame B (FLU, origin on the 4-inch tube axis;
mm in the facts, m here); every station has its own frame with its origin ON THE BENCH TOP (z = 0) under the point
named in `STATION_ORIGINS`, x / y as named there, z up; a station's pose in the cell is (x, y, yaw) on the bench.
The units' body frames are the kit's (vehicle unit map): T (thrusters), E (enclosure), G (guard), K (ballast block).

Where each unit lies in its station (all of this is in the station frame, at the middle of the play):
* ballast block: one per pocket, the face that seats against the bottom plate DOWN, its long side along the pocket's
  long side (the box's y), its own sides along the pocket's walls, the screw-hole axis on the pocket's centre;
* thruster: nose cone down through the nest's hole, the duct's lower edge on the three support pads (20 mm above the
  tray top), the duct axis on the nest's centre, the mounting face (the four bosses) towards the tray's +y (mount 0 deg) or
  -y (mount 180 deg); the propeller at any angle about the axis;
* enclosure: tube axis horizontal along the rack's x on the two V blocks (axis `tube_axis_z_mm` above the bench),
  the dome towards +x (dome 0 deg) or -x (180 deg), turned about its axis by the episode's roll;
* guard: flat on the shelf as it lies on its side of the vehicle (front end towards +x, outer edge outboard of the
  rack: +y on the port rack, -y on the starboard rack), its inner edge `stop_gap_mm` outboard of the low stop.
"""
import copy
import hashlib
import json
import math
import os
import struct

import numpy as np

__all__ = ["CELL_LAYOUT_SCHEMA", "FACTS_SCHEMA", "STATIONS", "STATION_ORIGINS", "NESTS", "NEST_PADS", "TABLE", "STRUCTURE",
           "LOAD_CELL", "PLAYS", "BENCH_BOARD", "TOOL_PLATE", "BOARDS", "TRAY_TAB", "BOX_TAB", "LMAG",
           "load_cell_layout", "load_facts", "Fixtures", "rotx", "rotz", "T_of", "box_mesh", "write_stl"]

CELL_LAYOUT_SCHEMA = "rov-cell-layout/1"
FACTS_SCHEMA = "rov-fixture-facts/1"
HERE = os.path.dirname(os.path.abspath(__file__))
TESTS_DIR = os.path.abspath(os.path.join(HERE, "..", ".."))

STATIONS = ("ballast_box", "thruster_tray", "enclosure_rack", "guard_rack_port", "guard_rack_stbd")
STATION_ORIGINS = {
    "ballast_box": "under the middle of the pocket field; x along the 4-pocket side",
    "thruster_tray": "under the middle of the four nests; x, y along the nest rows",
    "enclosure_rack": "under the point midway between the two V blocks; x along the tube axis",
    "guard_rack_port": "under the centre of the shelf's top face; x along the guard's long side, +y outboard (port)",
    "guard_rack_stbd": "under the centre of the shelf's top face; x along the guard's long side, -y outboard (starboard)",
}
# the tray's nests, numbered as on the drawing (tray frame, in units of half the nest pitch)
NESTS = (("N1", (-1, -1)), ("N2", (-1, 1)), ("N3", (1, -1)), ("N4", (1, 1)))

# ------------------------------------------------------------------ the fixtures' own design values
TABLE = {"x_m": (-0.55, 1.45), "y_m": (-0.70, 0.70), "thickness_m": 0.03}       # the bench (the cell's, 2.0 x 1.4 m)
# the positioner's structure: welded hollow sections, so their MASSES are set (the boxes are their envelopes)
STRUCTURE = {"base_mass_kg": 22.0, "carrier_mass_kg": 4.7,
             "rgba_base": [0.55, 0.57, 0.60, 1.0], "rgba_carrier": [0.35, 0.45, 0.60, 1.0]}
# three vertical load cells: linear spring + damper each (heave mode damping ratio about 0.27 with the base vehicle
# on: the reading settles within 0.1 s of a removed force; the manual calls a reading valid 0.5 s after)
LOAD_CELL = {"stiffness_N_m": 5.0e6, "damping_N_s_m": 4.0e3, "height_m": 0.015, "radius_m": 0.02}
# per-episode placement of the units inside their stations (uniform, each component; the draws are layout.py's)
PLAYS = {
    "ballast_block": {"xy_mm": 0.4, "yaw_deg": 0.3},         # inside the 0.5 mm per side of its pocket
    "guard": {"x_mm": 1.0, "y_mm": 1.0, "yaw_deg": 0.3},     # laid by hand on its marks (the stop 5.8 mm inboard)
    "enclosure_axial_mm": 1.0,                               # along the V blocks
    "enclosure_roll_deg": None,                              # None: the layout's stations.enclosure_rack.roll_range_deg
    "thruster": {"xy_mm": 0.0, "tilt_deg": 0.0},             # centred and upright on its pads
    "settle_lift_m": 0.0003,                                 # every unit starts this far above its support
}
# the support under each nest: three pads at the support ring's height (stations/thruster_tray/rim_above_top_mm), from
# the ring's inner radius (ring_r_mm[0]) to 50 mm, away from the mounting face's directions (+-y); three pads carry the
# duct's lower edge statically determinate. Why not the frozen ring itself (dev record P2_P10 section 4): the duct's
# lower edge lies at r 45.4-47.9 mm on the fine mesh, the frozen ring's 41-46 mm overlaps it by 0.6 mm only; and a
# full 24-segment ring gave each resting thruster 24 contacts (96 of the start's 156, more than half the tick's cost)
NEST_PADS = {"at_deg": (0.0, 120.0, 240.0), "width_deg": 20.0, "r_mm": (None, 50.0)}    # None: ring_r_mm[0]
# the thruster's tilt bound on its pads and the guard's play on its rack (the four values the frozen layout leaves to
# the fixtures: stations/thruster_tray/{board_origin_mm, tilt_max_deg}, stations/guard_rack/{play_mm, play_deg})
TILT_MAX_DEG = 0.5
# the tray's board on a tab of the tray block on the tray's -x side (towards the arm); tray frame, mm
TRAY_TAB = {"x_mm": (-240.0, None), "half_y_mm": 50.0}     # None: the tray block's -x face (-outline_half_mm)
# L-mag (ruling J8 / M10: a configuration switch, OFF by default): the ballast box jitters and carries a board on a
# tab on its +y side (towards the vehicle's axis); box frame, mm
BOX_TAB = {"half_x_mm": 60.0, "y_mm": (None, 235.0)}       # None: the box's +y face
LMAG = {"jitter_xy_m": 0.020, "jitter_yaw_deg": 5.0}
PRINT_ABOVE_PLATE_MM = 0.05                                  # render.scale_refs: the printed face above the plate
BENCH_BOARD = {"cols": 9, "rows": 6, "square_mm": 20.0, "margin_mm": 20.0, "thickness_mm": 3.0, "pos": [1.20, -0.50, 0.0],
               "yaw_deg": 0.0}
# the tool plate on the wrist camera housing's outer face: 5 x 4 squares of 8 mm in a 6 mm white margin (plate 52 x 44 mm;
# group Q ruling P18a-2: 6 mm squares in a 3 mm margin, one standard call of OpenCV's checkerboard detector found the
# plate at 0.5 m under the top camera in 16 of 18 frames and returned misplaced corners in the other 2; the 6 mm margin
# alone made it 5 of 18; with the 8 mm squares 18 of 18, dev record P2_P18d)
TOOL_PLATE = {"cols": 5, "rows": 4, "square_mm": 8.0, "margin_mm": 6.0, "thickness_mm": 1.5}
BOARDS = {
    "tray": {"fixture": "thruster_tray", "cols": 6, "rows": 5, "square_mm": 12.0, "margin_mm": 10.0, "thickness_mm": 3.0,
             "rest_xy_mm": [-190.0, 0.0], "yaw_deg": 90.0, "where": "on the tray's tab, on the tray's -x side"},
    "box": {"fixture": "ballast_box", "cols": 6, "rows": 4, "square_mm": 15.0, "margin_mm": 10.0, "thickness_mm": 3.0,
            "rest_xy_mm": [0.0, None], "yaw_deg": 0.0, "where": "on the box's tab, on the box's +y side (L-mag only)",
            "lmag_only": True},
}
MATERIAL = {"ballast_box": [0.30, 0.32, 0.35, 1.0], "thruster_tray": [0.20, 0.22, 0.25, 1.0],
            "enclosure_rack": [0.55, 0.58, 0.62, 1.0], "guard_rack_port": [0.62, 0.64, 0.67, 1.0],
            "guard_rack_stbd": [0.62, 0.64, 0.67, 1.0]}


# ------------------------------------------------------------------ small geometry
def rotz(deg):
    a = math.radians(float(deg))
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def rotx(deg):
    a = math.radians(float(deg))
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def T_of(R=None, p=None):
    T = np.eye(4)
    if R is not None:
        T[:3, :3] = R
    if p is not None:
        T[:3, 3] = p
    return T


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _default_path(fname, env):
    p = os.environ.get(env)
    return p if p else os.path.join(TESTS_DIR, fname)


def load_cell_layout(path=None):
    """the frozen cell layout (dict). path None: $ROV_CELL_LAYOUT, else <tests>/cell_layout.json"""
    p = path or _default_path("cell_layout.json", "ROV_CELL_LAYOUT")
    with open(p) as f:
        d = json.load(f)
    if d.get("schema") != CELL_LAYOUT_SCHEMA or d.get("status") != "frozen":
        raise ValueError("%s: not a frozen %s" % (p, CELL_LAYOUT_SCHEMA))
    d["_path"], d["_sha256"] = p, _sha256(p)
    return d


def load_facts(path=None):
    """the fixture facts (dict). path None: $ROV_FIXTURE_FACTS, else <tests>/fixture_facts.json"""
    p = path or _default_path("fixture_facts.json", "ROV_FIXTURE_FACTS")
    with open(p) as f:
        d = json.load(f)
    if d.get("schema") != FACTS_SCHEMA:
        raise ValueError("%s: not %s" % (p, FACTS_SCHEMA))
    d["_path"], d["_sha256"] = p, _sha256(p)
    return d


def _plate_with_holes(x0, x1, y0, y1, holes):
    """rectangle minus axis-aligned rectangular holes [(hx0, hx1, hy0, hy1)] -> list of rectangles"""
    xs = sorted({x0, x1} | {h[0] for h in holes} | {h[1] for h in holes})
    ys = sorted({y0, y1} | {h[2] for h in holes} | {h[3] for h in holes})
    out = []
    for i in range(len(xs) - 1):
        for j in range(len(ys) - 1):
            cx, cy = (xs[i] + xs[i + 1]) / 2, (ys[j] + ys[j + 1]) / 2
            if any(h[0] < cx < h[1] and h[2] < cy < h[3] for h in holes):
                continue
            out.append((xs[i], xs[i + 1], ys[j], ys[j + 1]))
    return out


def box_corners(c, R, h):
    """the 8 corners (8, 3) of an oriented box"""
    s = np.array([[a, b, d] for a in (-1, 1) for b in (-1, 1) for d in (-1, 1)], float)
    return np.asarray(c, float) + (s * np.asarray(h, float)) @ np.asarray(R, float).T


# ------------------------------------------------------------------ the fixtures
class Fixtures(object):
    """the cell's fixtures from a frozen cell layout and the fixture facts. lmag=True: the L-mag lever (the ballast
    box jitters and carries its board); every other fixture is the same either way."""

    def __init__(self, cell_layout=None, facts=None, lmag=False):
        self.L = cell_layout if cell_layout is not None else load_cell_layout()
        self.F = facts if facts is not None else load_facts()
        self.lmag = bool(lmag)
        self.st = self.L["stations"]
        self.pos = self.L["positioner"]
        self.der = self.L["derived"]
        u = self.F["units"]
        self.T_frame_in_B = {k: np.asarray(v["T_frame_in_B_mm"], float) for k, v in u.items()}

    # ------------------------------------------------------------ the positioner
    def vehicle_origin(self):
        return np.asarray(self.pos["vehicle_origin_cell_m"], float)

    def T_cell_B(self, index, mount_T=None):
        """4x4 (m) of the vehicle frame B in the cell at an index position ('deck_up' / 'keel_up'): the carrier's
        frame at the hinge (the vehicle's origin), turned 180 deg about the vehicle's x at keel_up (psi 0 only); then
        the vehicle's mounting offset on the carrier (mount_T, B frame, m)"""
        if abs(float(self.pos["psi_deg"])) > 1e-9:
            raise ValueError("only psi 0 (the stern towards the arm) is supported")
        q = {"deck_up": 0.0, "keel_up": 180.0}[index]
        T = T_of(rotx(q), self.vehicle_origin())
        return T if mount_T is None else T @ np.asarray(mount_T, float)

    def load_cells_cell_xy(self):
        """[(name, x, y)] of the three load cells in the cell frame (m): the stations' deck_up projections in B"""
        o = self.vehicle_origin()
        lc = self.pos["load_cells_vehicle_mm_deck_up"]
        return [(k, float(o[0] + lc[k][0] / 1000.0), float(o[1] + lc[k][1] / 1000.0)) for k in ("A", "B", "C")]

    def positioner_boxes(self):
        """{"static": [(name, centre m (cell), half m)], "turning": [(name, centre m (B), half m)]}: the open-frame
        base (drive seat, two outrigger beams, cross member) on the load cells and the carrier's fork / shaft, as the
        frozen layout gives them"""
        stat = [(n, np.asarray(c, float), np.asarray(h, float)) for n, c, h in self.der["static_boxes_cell_m"]]
        turn = [(n, np.asarray(c, float) / 1000.0, np.asarray(h, float) / 1000.0) for n, c, h in self.der["turning_boxes_B_mm"]]
        return {"static": stat, "turning": turn}

    def column(self):
        """the rotation column: axis = the hinge (the vehicle's x through its origin), radius, cell-x range (m)"""
        x0, x1 = (float(v) for v in self.der["column_x_range_cell_m"])
        o = self.vehicle_origin()
        return {"axis_point": [float(o[0]), float(o[1]), float(o[2])], "axis_dir": [1.0, 0.0, 0.0],
                "radius_m": float(self.pos["column_radius_m"]), "x_range_m": [x0, x1]}

    def vehicle_box_B(self, margin_m=0.0):
        """(lo, hi) m: the vehicle with every unit of this station at home (and a block on every ballast site), B axes"""
        lo, hi = (np.asarray(v, float) / 1000.0 for v in self.F["vehicle"]["box_all_home_B_mm"])
        return lo - margin_m, hi + margin_m

    # ------------------------------------------------------------ stations
    def station_spec(self, name):
        """(spec, (x, y, yaw) nominal, (jitter m, jitter deg)) of a station; the L-mag lever sets the box's jitter"""
        if name.startswith("guard_rack_"):
            g = self.st["guard_rack"]
            s = g[name.split("_")[-1]]
            return g, (s["nominal_xy_m"][0], s["nominal_xy_m"][1], s["nominal_yaw_deg"]), (g["jitter_xy_m"], g["jitter_yaw_deg"])
        s = self.st[name]
        if name == "ballast_box":
            j = (LMAG["jitter_xy_m"], LMAG["jitter_yaw_deg"]) if self.lmag else (s["jitter_xy_m"], s["jitter_yaw_deg"])
            return s, (s["origin_xy_m"][0], s["origin_xy_m"][1], s["yaw_deg"]), j
        return s, (s["nominal_xy_m"][0], s["nominal_xy_m"][1], s["nominal_yaw_deg"]), (s["jitter_xy_m"], s["jitter_yaw_deg"])

    # ---- the ballast box
    def _box_dims(self):
        s = self.st["ballast_box"]
        nx, ny = int(s["pockets_x"]), int(s["pockets_y"])
        px, py = s["pitch_x_mm"] / 1000.0, s["pitch_y_mm"] / 1000.0
        W, D = nx * px + 2 * s["rim_mm"] / 1000.0, ny * py + 2 * s["rim_mm"] / 1000.0
        own = self.der["ballast_block_own_mm"]
        pw = (own[0] + 2 * s["play_per_side_mm"]) / 1000.0
        pd = (own[1] + 2 * s["play_per_side_mm"]) / 1000.0
        return nx, ny, px, py, W, D, pw, pd, s["floor_z_mm"] / 1000.0, s["wall_height_mm"] / 1000.0

    def pockets(self):
        """[(name, (x, y) m)] of the 12 pockets in the box frame, numbered B01..B12 (x outer, y inner loop)"""
        nx, ny, px, py = self._box_dims()[:4]
        out = []
        for i in range(nx):
            for j in range(ny):
                out.append(("B%02d" % (len(out) + 1), (-nx * px / 2 + (i + 0.5) * px, -ny * py / 2 + (j + 0.5) * py)))
        return out

    def _ballast_box_boxes(self):
        nx, ny, px, py, W, D, pw, pd, fl, wh = self._box_dims()
        out = [("box_body", [0.0, 0.0, fl / 2], np.eye(3), [W / 2, D / 2, fl / 2], "box body")]
        holes = [(c[0] - pw / 2, c[0] + pw / 2, c[1] - pd / 2, c[1] + pd / 2) for _n, c in self.pockets()]
        for k, (a0, a1, b0, b1) in enumerate(_plate_with_holes(-W / 2, W / 2, -D / 2, D / 2, holes)):
            out.append(("wall_%d" % k, [(a0 + a1) / 2, (b0 + b1) / 2, fl + wh / 2], np.eye(3), [(a1 - a0) / 2, (b1 - b0) / 2, wh / 2],
                        "pocket wall"))
        if self.lmag:
            y0 = D / 2 if BOX_TAB["y_mm"][0] is None else BOX_TAB["y_mm"][0] / 1000.0
            y1 = BOX_TAB["y_mm"][1] / 1000.0
            hx = BOX_TAB["half_x_mm"] / 1000.0
            out.append(("box_tab", [0.0, (y0 + y1) / 2, (fl + wh) / 2], np.eye(3), [hx, (y1 - y0) / 2, (fl + wh) / 2], "board tab"))
        return out

    def pocket_pose(self, k, dx_mm=0.0, dy_mm=0.0, dyaw_deg=0.0, lift_m=0.0):
        """4x4 (m) of a ballast block's frame K in the box frame, in pocket k (0..11): the plate face down on the pocket
        floor, the hole axis on the pocket centre (+ the play), the block's own sides along the walls"""
        _n, c = self.pockets()[k]
        fl = self._box_dims()[8]
        yaw = float(self.der["ballast_block_yaw_in_vendor_cad_deg"])
        return (T_of(rotz(dyaw_deg), [c[0] + dx_mm / 1000.0, c[1] + dy_mm / 1000.0, fl + lift_m])
                @ T_of(rotx(180.0)) @ T_of(rotz(-yaw)))

    # ---- the thruster tray
    def _tray_dims(self):
        s = self.st["thruster_tray"]
        return (s["nest_pitch_mm"] / 1000.0, s["top_z_mm"] / 1000.0, s["hole_diameter_mm"] / 2000.0,
                s["outline_half_mm"] / 1000.0, s["ring_r_mm"][0] / 1000.0, s["ring_r_mm"][1] / 1000.0,
                s["rim_above_top_mm"] / 1000.0)

    def nests(self):
        """[(name, (x, y) m)] of the four nests in the tray frame"""
        P = self._tray_dims()[0]
        return [(n, (i * P / 2, j * P / 2)) for n, (i, j) in NESTS]

    def _tray_boxes(self):
        P, top, hh, oh, r0, r1, rim = self._tray_dims()
        holes = [(c[0] - hh, c[0] + hh, c[1] - hh, c[1] + hh) for _n, c in self.nests()]
        out = []
        for k, (a0, a1, b0, b1) in enumerate(_plate_with_holes(-oh, oh, -oh, oh, holes)):
            out.append(("tray_%d" % k, [(a0 + a1) / 2, (b0 + b1) / 2, top / 2], np.eye(3), [(a1 - a0) / 2, (b1 - b0) / 2, top / 2],
                        "tray block"))
        p0 = r0 if NEST_PADS["r_mm"][0] is None else NEST_PADS["r_mm"][0] / 1000.0
        p1 = r1 if NEST_PADS["r_mm"][1] is None else NEST_PADS["r_mm"][1] / 1000.0
        half_t = (p1 - p0) / 2
        rc = p0 + half_t
        hc = p0 * math.tan(math.radians(NEST_PADS["width_deg"] / 2.0))
        for k, (_n, c) in enumerate(self.nests()):
            for q, a in enumerate(NEST_PADS["at_deg"]):
                ca, sa = math.cos(math.radians(a)), math.sin(math.radians(a))
                out.append(("pad_%d_%d" % (k, q), [c[0] + rc * ca, c[1] + rc * sa, top + rim / 2], rotz(a), [half_t, hc, rim / 2],
                            "support pad"))
        x0 = -oh if TRAY_TAB["x_mm"][1] is None else TRAY_TAB["x_mm"][1] / 1000.0
        xt = TRAY_TAB["x_mm"][0] / 1000.0
        hy = TRAY_TAB["half_y_mm"] / 1000.0
        out.append(("tray_tab", [(xt + x0) / 2, 0.0, top / 2], np.eye(3), [(x0 - xt) / 2, hy, top / 2], "board tab"))
        return out

    def nest_pose(self, k, mount_deg=0.0, lift_m=0.0):
        """4x4 (m) of a thruster's frame T in the tray frame, in nest k (0..3): nose down, the duct's lower edge on the
        support pads, the duct axis on the nest centre, the mounting face towards +y (mount 0) / -y (mount 180)"""
        _n, c = self.nests()[k]
        P, top, hh, oh, r0, r1, rim = self._tray_dims()
        zb = float(self.F["thruster_T_mm"]["duct_bottom_z"]) / 1000.0
        return T_of(rotz(float(mount_deg) + 180.0), [c[0], c[1], top + rim - zb + lift_m])

    # ---- the enclosure rack
    def _rack_boxes(self):
        s = self.st["enclosure_rack"]
        az = s["tube_axis_z_mm"] / 1000.0
        sp = s["vblock_spacing_mm"] / 1000.0
        bw = s["vblock_width_mm"] / 2000.0
        y0, y1 = s["vblock_y_mm"][0] / 1000.0, s["vblock_y_mm"][1] / 1000.0
        t = s["vblock_t_mm"] / 1000.0
        r_tube = float(self.F["enclosure_E_mm"]["tube_r"]) / 1000.0
        half_ang = math.radians(s["vblock_angle_deg"] / 2.0)
        zc_contact = az - r_tube * math.sin(half_ang)
        yc_contact = r_tube * math.cos(half_ang)
        slope = math.pi / 2 - half_ang
        out = []
        for k, xb in enumerate((-sp / 2, sp / 2)):
            for sd in (1, -1):
                Rl = rotx(sd * math.degrees(slope))
                ym = sd * (y0 + y1) / 2
                zm = zc_contact + (abs(ym) - yc_contact) * math.tan(slope)
                n = Rl @ np.array([0.0, 0.0, 1.0])
                hl = (y1 - y0) / 2 / math.cos(slope)
                c = np.array([xb, ym, zm]) - n * t / 2
                out.append(("vface_%d_%s" % (k, "L" if sd > 0 else "R"), c.tolist(), Rl, [bw, hl, t / 2], "V block face"))
            zb = zc_contact - (yc_contact - y0) * math.tan(slope) - t
            out.append(("vbase_%d" % k, [xb, 0.0, zb / 2], np.eye(3), [bw, y1, zb / 2], "V block"))
        return out

    def enclosure_pose(self, roll_deg=0.0, dome_deg=0.0, axial_mm=0.0, lift_m=0.0):
        """4x4 (m) of the enclosure's frame E in the rack frame: the tube axis on the rack's x at the axis height, the
        dome towards +x (dome 0) / -x (dome 180), turned about the axis by roll; the rack origin under the point of the
        axis midway between the two cradles' places (the enclosure slot's origin)"""
        s = self.st["enclosure_rack"]
        az = s["tube_axis_z_mm"] / 1000.0
        xm = (float(self.der["enclosure_rack_E_x_at_origin_mm"]) - float(self.F["enclosure_E_mm"]["E_origin_x_B"])) / 1000.0
        return (T_of(rotz(float(dome_deg)), [0.0, 0.0, az + lift_m]) @ T_of(rotx(float(roll_deg)))
                @ T_of(None, [-xm + axial_mm / 1000.0, 0.0, 0.0]))

    # ---- the guard racks
    def _guard_geom(self, side):
        g = self.st["guard_rack"]
        sg = 1.0 if side == "port" else -1.0
        gf = self.F["guard_B_mm"]
        width = float(gf["width"]) / 1000.0
        on_shelf = width - g["overhang_mm"] / 1000.0
        shelf_w = on_shelf + g["stop_inset_mm"] / 1000.0
        yi = -shelf_w / 2 + g["stop_inset_mm"] / 1000.0          # the guard's inner edge (outboard = +y for the port rack)
        return g, sg, width, shelf_w, yi

    def _guard_rack_boxes(self, side):
        g, sg, width, shelf_w, yi = self._guard_geom(side)
        st = g["shelf_top_z_mm"] / 1000.0
        t = g["shelf_t_mm"] / 1000.0
        ln = g["shelf_len_mm"] / 2000.0
        out = [("shelf", [0.0, 0.0, st - t / 2], np.eye(3), [ln, shelf_w / 2, t / 2], "shelf")]
        ped_w = shelf_w / 2 - g["pedestal_inset_mm"] / 1000.0
        out.append(("pedestal", [0.0, sg * (-shelf_w / 2 + ped_w), (st - t) / 2], np.eye(3), [ln * 0.6, ped_w, (st - t) / 2], "pedestal"))
        ys0 = yi - g["stop_gap_mm"] / 1000.0
        out.append(("stop", [0.0, sg * (ys0 - 0.0025), st + g["stop_h_mm"] / 2000.0], np.eye(3), [ln * 0.8, 0.0025, g["stop_h_mm"] / 2000.0],
                    "low stop"))
        return out

    def guard_pose(self, side, dx_mm=0.0, dy_mm=0.0, dyaw_deg=0.0, lift_m=0.0):
        """4x4 (m) of a guard's frame G in its rack's frame (side 'port' / 'stbd'): lying as it lies on that side of
        the vehicle, its middle (along x) over the shelf's centre, its inner edge on the mark, its lower face on the
        shelf; then the play (about the shelf centre)"""
        g, sg, width, shelf_w, yi = self._guard_geom(side)
        gf = self.F["guard_B_mm"]
        st = g["shelf_top_z_mm"] / 1000.0
        unit = "guard_P" if side == "port" else "guard_S"
        # B -> rack: the guard's B coordinates shifted (no turn: it lies in its vehicle orientation)
        p = np.array([-float(gf["x_centre"]) / 1000.0, sg * yi - sg * float(gf["inner_abs_y"]) / 1000.0,
                      st - float(gf["lower_z"]) / 1000.0 + lift_m])
        T_rack_B = T_of(rotz(dyaw_deg), [dx_mm / 1000.0, dy_mm / 1000.0, 0.0]) @ T_of(None, p)
        Tf = self.T_frame_in_B[unit].copy()
        Tf[:3, 3] /= 1000.0
        return T_rack_B @ Tf

    # ---- every station
    def station_boxes(self, name):
        """[(box name, centre m, R 3x3, half m, part)] of a station in its frame"""
        if name == "ballast_box":
            return self._ballast_box_boxes()
        if name == "thruster_tray":
            return self._tray_boxes()
        if name == "enclosure_rack":
            return self._rack_boxes()
        if name in ("guard_rack_port", "guard_rack_stbd"):
            return self._guard_rack_boxes(name.split("_")[-1])
        raise ValueError("unknown station %r" % name)

    def station_top_z(self, name):
        return max(float(box_corners(c, R, h)[:, 2].max()) for _n, c, R, h, _p in self.station_boxes(name))

    # ------------------------------------------------------------ datum boards
    def boards(self):
        """{name: {fixture, T_rest (4x4 m in the fixture frame: the centre of the face the plate rests on, +z out of it,
        x along the columns), cols, rows, square_mm, margin_mm, thickness_mm, plate_mm, face_height_mm, where}}; the
        bench board's fixture is the bench (its T_rest is in the cell frame)"""
        out = {}
        b = BENCH_BOARD
        Tb = T_of(rotz(b["yaw_deg"]), b["pos"])
        out["bench"] = self._board("bench", "bench", b, Tb, "on the bench, beyond the starboard guard rack")
        for n, b in BOARDS.items():
            if b.get("lmag_only") and not self.lmag:
                continue
            if n == "tray":
                top = self._tray_dims()[1]
                Tr = T_of(rotz(b["yaw_deg"]), [b["rest_xy_mm"][0] / 1000.0, b["rest_xy_mm"][1] / 1000.0, top])
            else:
                nx, ny, px, py, W, D, pw, pd, fl, wh = self._box_dims()
                y0 = D / 2 if BOX_TAB["y_mm"][0] is None else BOX_TAB["y_mm"][0] / 1000.0
                y1 = BOX_TAB["y_mm"][1] / 1000.0
                Tr = T_of(rotz(b["yaw_deg"]), [b["rest_xy_mm"][0] / 1000.0, (y0 + y1) / 2, fl + wh])
            out[n] = self._board(n, b["fixture"], b, Tr, b["where"])
        return out

    @staticmethod
    def _board(name, fixture, b, T, where):
        size = (b["cols"] * b["square_mm"] + 2 * b["margin_mm"], b["rows"] * b["square_mm"] + 2 * b["margin_mm"], b["thickness_mm"])
        return {"fixture": fixture, "T_rest": T, "cols": b["cols"], "rows": b["rows"], "square_mm": b["square_mm"],
                "margin_mm": b["margin_mm"], "thickness_mm": b["thickness_mm"], "plate_mm": list(size),
                "face_height_mm": float(T[2, 3] * 1000.0 + b["thickness_mm"] + PRINT_ABOVE_PLATE_MM), "where": where}

    def board_corners(self, name, T_fixture_world=None):
        """the board's inner pattern corners (n, 3) in the world (or fixture frame), m"""
        b = self.boards()[name]
        T = (np.eye(4) if T_fixture_world is None else np.asarray(T_fixture_world, float)) @ b["T_rest"]
        z = (b["thickness_mm"] + PRINT_ABOVE_PLATE_MM) / 1000.0
        pts = [[(-b["cols"] / 2 + i) * b["square_mm"] / 1000.0, (b["rows"] / 2 - j) * b["square_mm"] / 1000.0, z]
               for j in range(1, b["rows"]) for i in range(1, b["cols"])]
        return np.asarray(pts) @ T[:3, :3].T + T[:3, 3]

    def board_plate_corners(self, name, T_fixture_world=None):
        b = self.boards()[name]
        T = (np.eye(4) if T_fixture_world is None else np.asarray(T_fixture_world, float)) @ b["T_rest"]
        bx, by, th = (v / 1000.0 for v in b["plate_mm"])
        P = box_corners([0.0, 0.0, th / 2], np.eye(3), [bx / 2, by / 2, th / 2])
        return P @ T[:3, :3].T + T[:3, 3]

    # ------------------------------------------------------------ the parts' geometry (coarse hulls, unit frames)
    def hull(self, kind):
        """(n, 3) m: a coarse convex hull of a unit kind's geometry in its body frame: 'thruster_a', 'thruster_b',
        'enclosure', 'guard', 'ballast'"""
        return np.asarray(self.F["hulls_mm"][kind], float) / 1000.0

    # ------------------------------------------------------------ the four values the frozen layout leaves open
    def fixture_values(self):
        """the values `cell_layout.json` lists under fixture_values_for_P10 (keys as listed there)"""
        b = self.boards()["tray"]
        return {"stations/thruster_tray/board_origin_mm": [round(float(b["T_rest"][0, 3]) * 1000.0, 3),
                                                           round(float(b["T_rest"][1, 3]) * 1000.0, 3)],
                "stations/thruster_tray/tilt_max_deg": TILT_MAX_DEG,
                "stations/guard_rack/play_mm": PLAYS["guard"]["x_mm"],
                "stations/guard_rack/play_deg": PLAYS["guard"]["yaw_deg"]}

    # ------------------------------------------------------------ the cell (MuJoCo) side
    def add_station(self, spec, name, T_world, prefix=None, rgba=None, group=0):
        """a static body `<prefix or name>` at T_world with the station's boxes (world-fixed, no joint); returns
        its geom names"""
        import mujoco
        nm = prefix or name
        b = spec.worldbody.add_body(name=nm, pos=[float(v) for v in T_world[:3, 3]], quat=_quat_wxyz(T_world[:3, :3]).tolist())
        out = []
        for bn, c, R, h, _p in self.station_boxes(name):
            g = b.add_geom(name="%s/%s" % (nm, bn), type=mujoco.mjtGeom.mjGEOM_BOX, size=[float(v) for v in h],
                           pos=[float(v) for v in c], quat=_quat_wxyz(R).tolist(), rgba=list(rgba or MATERIAL.get(name, [0.5] * 3 + [1.0])),
                           group=group)
            out.append(g.name)
        return out

    # ------------------------------------------------------------ export (drawings, photos)
    def export(self, out_dir):
        """write <out_dir>/fixtures.json (every box, board, pocket, nest and the units' rest poses in the station frames,
        mm / deg; the positioner's boxes; the plays; the four fixture values) and <out_dir>/<station>.stl,
        positioner_static.stl (cell frame), positioner_turning.stl (B frame), board_<name>.stl (binary, mm). Returns
        the json dict (its "files" = the written file names)."""
        os.makedirs(out_dir, exist_ok=True)
        doc = {"format": "rov-cell-fixtures/1", "units": "mm, deg", "z": "0 = the bench top", "lmag": self.lmag,
               "cell_layout_sha256": self.L.get("_sha256"), "facts_sha256": self.F.get("_sha256"),
               "station_origins": dict(STATION_ORIGINS), "plays": copy.deepcopy(PLAYS), "fixture_values": self.fixture_values(),
               "load_cell": dict(LOAD_CELL), "structure": {k: v for k, v in STRUCTURE.items() if not k.startswith("rgba")},
               "stations": {}, "boards": {}, "positioner": {}}
        files = []
        for name in STATIONS:
            boxes = self.station_boxes(name)
            doc["stations"][name] = {"boxes": [{"name": bn, "centre": [round(float(v) * 1000.0, 4) for v in c],
                                                "yaw_pitch_roll_deg": [round(v, 6) for v in _ypr(R)],
                                                "half": [round(float(v) * 1000.0, 4) for v in h], "part": p}
                                               for bn, c, R, h, p in boxes]}
            V, F = _boxes_mesh([(c, R, h) for _bn, c, R, h, _p in boxes])
            write_stl(os.path.join(out_dir, name + ".stl"), V * 1000.0, F, b"assembly cell fixture, millimetres, fixture frame")
            files.append(name + ".stl")
        doc["stations"]["ballast_box"]["pockets"] = [{"name": n, "centre": [round(c[0] * 1000.0, 4), round(c[1] * 1000.0, 4)]}
                                                     for n, c in self.pockets()]
        doc["stations"]["ballast_box"]["block_rest"] = [_pose_doc(self.pocket_pose(k)) for k in range(len(self.pockets()))]
        doc["stations"]["thruster_tray"]["nests"] = [{"name": n, "centre": [round(c[0] * 1000.0, 4), round(c[1] * 1000.0, 4)]}
                                                     for n, c in self.nests()]
        doc["stations"]["thruster_tray"]["thruster_rest"] = {"mount_0": [_pose_doc(self.nest_pose(k, 0.0)) for k in range(4)],
                                                             "mount_180": [_pose_doc(self.nest_pose(k, 180.0)) for k in range(4)]}
        doc["stations"]["enclosure_rack"]["enclosure_rest"] = {"dome_0": _pose_doc(self.enclosure_pose(0.0, 0.0)),
                                                               "dome_180": _pose_doc(self.enclosure_pose(0.0, 180.0))}
        for side in ("port", "stbd"):
            doc["stations"]["guard_rack_" + side]["guard_rest"] = _pose_doc(self.guard_pose(side))
        pb = self.positioner_boxes()
        doc["positioner"] = {"static_cell": [{"name": n, "centre": [round(float(v) * 1000.0, 4) for v in c],
                                              "half": [round(float(v) * 1000.0, 4) for v in h]} for n, c, h in pb["static"]],
                             "turning_B": [{"name": n, "centre": [round(float(v) * 1000.0, 4) for v in c],
                                            "half": [round(float(v) * 1000.0, 4) for v in h]} for n, c, h in pb["turning"]],
                             "load_cells_cell_mm": {k: [round(x * 1000.0, 4), round(y * 1000.0, 4)] for k, x, y in self.load_cells_cell_xy()},
                             "column": self.column()}
        for key, boxes in (("positioner_static", pb["static"]), ("positioner_turning", pb["turning"])):
            V, F = _boxes_mesh([(c, np.eye(3), h) for _n, c, h in boxes])
            write_stl(os.path.join(out_dir, key + ".stl"), V * 1000.0, F, b"assembly cell fixture, millimetres, " +
                      (b"cell frame" if key.endswith("static") else b"vehicle frame"))
            files.append(key + ".stl")
        for n, b in self.boards().items():
            T = b["T_rest"]
            bx, by, th = b["plate_mm"]
            V, F = box_mesh([-bx / 2, -by / 2, 0.0], [bx / 2, by / 2, th])
            V = V @ T[:3, :3].T + T[:3, 3] * 1000.0
            write_stl(os.path.join(out_dir, "board_%s.stl" % n), V, F, b"assembly cell datum board, millimetres, fixture frame")
            files.append("board_%s.stl" % n)
            doc["boards"][n] = {"fixture": b["fixture"], "cols": b["cols"], "rows": b["rows"], "square_mm": b["square_mm"],
                                "margin_mm": b["margin_mm"], "thickness_mm": b["thickness_mm"], "plate_mm": b["plate_mm"],
                                "rest_centre_mm": [round(float(v) * 1000.0, 4) for v in T[:3, 3]],
                                "x_axis_turn_deg": round(math.degrees(math.atan2(T[1, 0], T[0, 0])), 6),
                                "face_height_mm": round(b["face_height_mm"], 4), "where": b["where"],
                                "pattern": "render.scale_refs.checkerboard: origin at the pattern's middle, x along the "
                                           "columns, the square at the board's (-x, +y) corner black"}
        doc["tool_plate"] = dict(TOOL_PLATE)
        with open(os.path.join(out_dir, "fixtures.json"), "w") as f:
            json.dump(doc, f, indent=1, sort_keys=True)
            f.write("\n")
        doc["files"] = sorted(files + ["fixtures.json"])
        return doc


# ------------------------------------------------------------------ helpers
def _quat_wxyz(R):
    R = np.asarray(R, float)
    t = np.trace(R)
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        q = [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        q = [(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s]
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        q = [(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s]
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        q = [(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s]
    q = np.array(q)
    q /= np.linalg.norm(q)
    return q if q[0] >= 0 else -q


def _ypr(R):
    R = np.asarray(R, float)
    yaw = math.degrees(math.atan2(R[1, 0], R[0, 0]))
    pitch = math.degrees(math.asin(max(-1.0, min(1.0, -R[2, 0]))))
    roll = math.degrees(math.atan2(R[2, 1], R[2, 2]))
    return yaw, pitch, roll


def _pose_doc(T):
    return {"origin_mm": [round(float(v) * 1000.0, 4) for v in T[:3, 3]], "yaw_pitch_roll_deg": [round(v, 6) for v in _ypr(T[:3, :3])]}


def box_mesh(lo, hi):
    x0, y0, z0 = lo
    x1, y1, z1 = hi
    V = np.array([[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0], [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]], float)
    F = np.array([[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7], [0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5], [2, 3, 7], [2, 7, 6],
                  [3, 0, 4], [3, 4, 7]], int)
    return V, F


def _boxes_mesh(boxes):
    Vs, Fs, off = [], [], 0
    for c, R, h in boxes:
        h = np.asarray(h, float)
        V, F = box_mesh(-h, h)
        Vs.append(V @ np.asarray(R, float).T + np.asarray(c, float))
        Fs.append(F + off)
        off += len(V)
    return np.concatenate(Vs), np.concatenate(Fs)


def write_stl(path, V, F, header=b"assembly cell fixture"):
    V = np.asarray(V, np.float32)
    F = np.asarray(F, int)
    tri = V[F]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    ln = np.linalg.norm(n, axis=1, keepdims=True)
    n = (n / np.where(ln > 0, ln, 1.0)).astype(np.float32)
    rec = np.zeros(len(F), dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]))
    rec["n"], rec["v"] = n, tri
    with open(path, "wb") as f:
        f.write(header.ljust(80, b" ")[:80])
        f.write(struct.pack("<I", len(F)))
        f.write(rec.tobytes())
