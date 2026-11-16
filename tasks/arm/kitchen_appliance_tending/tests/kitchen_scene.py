"""The kitchen around the cell (an apartment from the ArchVizPRO Interior Vol.8 art pack): pure functions.

kitchen_render.py (the cameras' renderer) calls this module for the dressing of each unit and episode (the station
around the unit's support, the cabinets beside it, the robot's pedestal, the shelf), the pose of the room bundle and
the lamps. It imports neither MuJoCo nor Filament (math + numpy only) and never touches collision geometry:
everything it returns is drawn only.

Frames
  cell    the environment's world (appliance_env.py): robot base at (0, 0, 0.65) on a pedestal, facing +x,
          z up, floor z = 0.  Metres.
  bundle  the room bundle of the apartment (room.npz in KITCHEN_RENDER_DIR): x, y horizontal, z up, floor z = 0.

Placement
  The cell stands at the pack's kitchen run on the south wall, in the bay where the pack's hob section was.
  Bundle -y (toward the south wall) is cell +x; bundle x = -2.50 (the bay centre) is cell y = 0; the apartment's
  floor is the cell's floor (no lift).  The bay's visible wall face (the herringbone tiles, bundle y = -3.623)
  coincides with the front face of the environment's back wall (cell x = front + max(depth, 0.6) + 0.13).  So the
  room's pose depends on the unit's depth and on jx only; the station follows jy inside the bay.

Unit arguments: `kind` ('stovetop' | 'microwave' | 'oven'), `unit_lo`, `unit_hi` = the unit's box in its model
frame (appliance_env.unit_bbox: x = width, y = depth, z = height), `jitter` = (jx, jy, jyaw) of the episode.

Parts: dicts {"name", "shape": "box" | "cylinder", "center": [x, y, z], "size": [sx, sy, sz] (box: full sizes;
cylinder: [radius, half length, 0]), "axis": "x" | "y" | "z" (cylinders), "material", "role", "grain": "h" | "v"
(textured wood: grain direction on the part's largest face)}.  All boxes are axis-aligned in the cell frame.
role: "body" (counts for the station footprint), "closure" (fills the slot between the support and the wall),
"hardware" (handles: they stand 23 mm proud of the fronts), "filler", "pedestal".
"""
import math

import numpy as np

# ---------------------------------------------------------------- the environment's mounting (appliance_env.py)
FRONT_X = 0.55
ROBOT_Z = 0.65
BOTTOM_Z = {"stovetop": 0.92, "microwave": 0.92, "oven": 0.55, "stove": 0.0}
SUPPORT_MIN = 0.6                  # max(depth, 0.6), max(width, 0.6)
SUPPORT_PAD_X = 0.05               # half-size pads of the support box
SUPPORT_PAD_Y = 0.25
SUPPORT_SETBACK = 0.03             # the support's front face is 3 cm in front of the unit's front face
BACK_WALL_GAP = 0.13               # back wall front face = front + max(depth, 0.6) + 0.13
BACK_WALL = (0.04, 3.0, 2.8)       # thickness, width, height
PEDESTAL = (0.28, 0.28, 0.65)      # centred at x = -0.05
PEDESTAL_X = -0.05
CAMERAS = {"left": ((-0.45, 0.62, 0.45), (0.12, 0.0, 0.0), 55.0, 320, 240),
           "right": ((-0.45, -0.62, 0.45), (0.12, 0.0, 0.0), 55.0, 320, 240)}

# ---------------------------------------------------------------- the apartment (bundle frame)
PACK_WALL_Y = -3.733               # the south wall's main plane (sink run, fridge)
BAY_PLASTER_Y = -3.633             # the box-out behind the pack's hob section (x -3.495 .. -1.605, floor to ceiling)
BAY_TILE_Y = -3.623                # its herringbone tiles (z 0.82 .. 1.42), 10 mm proud of the plaster
BAY_X_WEST = -3.495                # the west wall's face (the counter run starts here)
BAY_X_SECTION = -1.605             # the end of the pack's hob section (carcass, uppers, box-out and tiles end here)
END_PANEL_T = 0.018
BAY_X_EAST = BAY_X_SECTION - END_PANEL_T   # -1.623: outer face of the new end panels of the remaining run
BAY_CX = -2.50                     # -> cell y = 0
PACK_TOP_Z = 0.82
PACK_SLAB_T = 0.04
PACK_FRONT_Y = -3.033              # the pack's counter front edge = its door faces
PACK_DEPTH = 0.70                  # from the main wall plane
PACK_PLINTH_H = 0.08
PACK_PLINTH_SETBACK = 0.018
PACK_DOOR_Z = (0.081, 0.779)
PACK_HANDLE_Z = (0.608, 0.758)     # vertical 150 mm bars, 22 mm from the opening edge
UPPER_BOTTOM_Z = 1.42
UPPER_FRONT_Y = -3.261
TILE_Z = (0.82, 1.42)
CEILING_Z = 2.65
SOCKETS_LIFT = 0.10                # the bay's socket / LAN / switch plates are raised 0.10 m, above a 0.92 top

# ---------------------------------------------------------------- dressing dimensions
FRONT_T = 0.019
SIDE_T = 0.019
GAP = 0.003                        # shadow gap between fronts
SLAB_T = 0.03                      # the station's stone top
PLINTH_H = 0.10
PLINTH_SETBACK = 0.05
HANDLE_R = 0.003                   # the pack's handle: 6 mm bar on two 20 mm standoffs, 150 mm long
HANDLE_STANDOFF = 0.020
HANDLE_INSET = 0.0085              # standoff centre from the bar's end
HANDLE_L = 0.15
HANDLE_L_WIDE = 0.30               # on drawer fronts wider than 0.5 m
FILLER_MIN = 0.05                  # a gap under 5 cm stays open
FILLER_STRIP_MAX = 0.15            # narrower than this: a plain filler strip (no door, no handle)
DOOR_MAX = 0.60
LOW_BENCH_KINDS = ("oven",)        # the oven bay is one low appliance bench at the station's height with the
                                   # station's stone top; cooktop / microwave keep the pack's 0.82 run

# the floating oak shelf on the bay's plaster: it belongs to the room (bundle frame, does not follow jy)
SHELF_L, SHELF_D, SHELF_T, SHELF_TOP_Z = 1.60, 0.20, 0.035, 1.85
SHELF_CX = (BAY_X_WEST + BAY_X_EAST) / 2           # -2.559: centred on the bay (x -3.495 .. -1.623)
SHELF_BUNDLE = {"lo": [SHELF_CX - SHELF_L / 2, BAY_PLASTER_Y + SHELF_D, SHELF_TOP_Z - SHELF_T],   # lo[1]: front edge
                "hi": [SHELF_CX + SHELF_L / 2, BAY_PLASTER_Y, SHELF_TOP_Z]}      # y: from the plaster 0.20 into the room
SHELF_LED = {"len": 1.50, "width": 0.012, "setback": 0.030, "recess": 0.004,      # under the shelf, 30 mm behind the
             "profile_w": 0.020}                                                 # front edge, in a 4 mm groove
# the shelf's props (part of the room bundle): footprints in the bundle frame, all standing on z 1.85
SHELF_PROPS = {"plant_bowl": ([-3.1005, -3.6075], [-2.9395, -3.4585]),
               "milk_jug": ([-2.238, -3.589], [-2.082, -3.477]),
               "spice_rack": ([-2.006, -3.607], [-1.854, -3.459])}


def _v(x):
    return [float(a) for a in x]


def unit_dims(unit_lo, unit_hi):
    lo, hi = np.asarray(unit_lo, float), np.asarray(unit_hi, float)
    s = hi - lo
    return float(s[0]), float(s[1]), float(s[2])        # width (cell y), depth (cell x), height


def cell_geometry(kind, unit_lo, unit_hi, jitter=(0.0, 0.0, 0.0)):
    """The environment's own boxes for this unit and episode (build_model's numbers), cell frame."""
    width, depth, height = unit_dims(unit_lo, unit_hi)
    jx, jy, jyaw = (float(a) for a in jitter)
    bz = BOTTOM_Z[kind]
    front = FRONT_X + jx
    D = max(depth, SUPPORT_MIN)
    half = (D / 2 + SUPPORT_PAD_X, max(width, SUPPORT_MIN) / 2 + SUPPORT_PAD_Y, bz / 2)
    support = {"center": [front + half[0] - SUPPORT_SETBACK, jy, bz / 2], "half": list(half)} if bz > 0.01 else None
    wall_x = front + D + BACK_WALL_GAP
    back_wall = {"center": [wall_x + BACK_WALL[0] / 2, 0.0, BACK_WALL[2] / 2],
                 "half": [BACK_WALL[0] / 2, BACK_WALL[1] / 2, BACK_WALL[2] / 2]}
    front_centre = np.array([FRONT_X, 0.0, bz + height / 2])   # nominal, never jittered
    cams = {}
    for n, (p, t, fovy, w, h) in CAMERAS.items():
        cams[n] = {"pos": _v(front_centre + p), "target": _v(front_centre + t), "fovy": fovy, "res": [w, h]}
    return {"kind": kind, "width": width, "depth": depth, "height": height, "bottom_z": bz, "front": front,
            "D": D, "jitter": [jx, jy, jyaw], "support": support, "back_wall": back_wall, "wall_x": wall_x,
            "pedestal": {"center": [PEDESTAL_X, 0.0, PEDESTAL[2] / 2], "half": [p / 2 for p in PEDESTAL]},
            "cameras": cams}


def room_pose(kind, unit_lo, unit_hi, jitter=(0.0, 0.0, 0.0)):
    """4x4 bundle -> cell: p_cell = T @ p_bundle.  A turn of +90 deg about z (bundle -y -> cell +x) and a shift;
    it depends on the unit's depth and on jx only."""
    g = cell_geometry(kind, unit_lo, unit_hi, jitter)
    T = np.eye(4)
    T[:3, :3] = [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    T[0, 3] = g["wall_x"] + BAY_TILE_Y
    T[1, 3] = -BAY_CX
    return T


def to_cell(T, p):
    p = np.asarray(p, float)
    return (T[:3, :3] @ p.T).T + T[:3, 3] if p.ndim == 2 else T[:3, :3] @ p + T[:3, 3]


def bundle_box_to_cell(T, lo, hi):
    """Axis-aligned bundle box -> axis-aligned cell box (the pose is a quarter turn)."""
    c = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    q = to_cell(T, c)
    return q.min(0), q.max(0)


def _box(name, lo, hi, material, role, grain=None):
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    assert np.all(hi > lo), (name, lo, hi)
    p = {"name": name, "shape": "box", "center": _v((lo + hi) / 2), "size": _v(hi - lo), "material": material,
         "role": role}
    if grain:
        p["grain"] = grain
    return p


def _cyl(name, centre, radius, half_len, axis, material, role):
    return {"name": name, "shape": "cylinder", "center": _v(centre), "size": [float(radius), float(half_len), 0.0],
            "axis": axis, "material": material, "role": role}


def part_bounds(p):
    c = np.asarray(p["center"], float)
    if p["shape"] == "box":
        h = np.asarray(p["size"], float) / 2
    else:
        r, hl = p["size"][0], p["size"][1]
        h = np.array([r, r, r])
        h["xyz".index(p["axis"])] = hl
    return c - h, c + h


def bar_handle(name, centre, axis, length, out_dir=-1.0, material="handle_steel", role="hardware"):
    """The pack's bar handle (6 mm bar on two 20 mm standoffs) on a front whose face is at centre[0], standing out
    along cell x by out_dir (-1: toward the robot).  axis: 'y' horizontal, 'z' vertical."""
    c = np.asarray(centre, float)
    bar_x = c[0] + out_dir * (HANDLE_STANDOFF + HANDLE_R)
    parts = [_cyl(name + "_bar", [bar_x, c[1], c[2]], HANDLE_R, length / 2, axis, material, role)]
    k = "xyz".index(axis)
    for s in (-1, 1):
        p = c.copy()
        p[0] = c[0] + out_dir * HANDLE_STANDOFF / 2
        p[k] += s * (length / 2 - HANDLE_INSET)
        parts.append(_cyl(name + "_post%+d" % s, p, HANDLE_R, HANDLE_STANDOFF / 2, "x", material, role))
    return parts


def _split(a, b, n, gap):
    """n equal spans over [a, b] with `gap` between them."""
    w = (b - a - (n - 1) * gap) / n
    return [(a + i * (w + gap), a + i * (w + gap) + w) for i in range(n)]


def station_parts(kind, unit_lo, unit_hi, jitter=(0.0, 0.0, 0.0)):
    """The environment's support box, dressed: light-oak fronts and side panels, a dark-grey plinth set back 0.05,
    a 0.03 m stone top whose upper face is at bottom_z, brushed-steel bar handles.  Its body parts fill exactly
    the support's box; the closure parts continue the top and the side panels back to the wall face."""
    g = cell_geometry(kind, unit_lo, unit_hi, jitter)
    s = g["support"]
    if s is None:
        return []
    c, h = np.asarray(s["center"]), np.asarray(s["half"])
    X0, X1 = c[0] - h[0], c[0] + h[0]
    Y0, Y1 = c[1] - h[1], c[1] + h[1]
    bz = 2 * h[2]
    zt = bz - SLAB_T
    P = [_box("st_top", [X0, Y0, zt], [X1, Y1, bz], "station_stone", "body"),
         _box("st_side_w", [X0, Y0, 0.0], [X1, Y0 + SIDE_T, zt], "station_oak", "body", "v"),
         _box("st_side_e", [X0, Y1 - SIDE_T, 0.0], [X1, Y1, zt], "station_oak", "body", "v"),
         _box("st_plinth", [X0 + PLINTH_SETBACK, Y0 + SIDE_T, 0.0], [X1, Y1 - SIDE_T, PLINTH_H], "plinth_dark", "body"),
         _box("st_carcass", [X0 + FRONT_T, Y0 + SIDE_T, PLINTH_H], [X1, Y1 - SIDE_T, zt], "station_carcass", "body")]
    # fronts: columns of ~0.6-0.8 m; a cooktop / microwave base gets a top drawer and a deep pan drawer per column,
    # the oven's plinth cabinet one drawer per column
    ya, yb = Y0 + SIDE_T + GAP, Y1 - SIDE_T - GAP
    za, zb = PLINTH_H, zt - GAP
    ncol = max(1, int(round((yb - ya) / 0.62)))
    rows = [(za, zb)] if (zb - za) < 0.5 else [(za, zb - 0.18 - GAP), (zb - 0.18, zb)]
    for i, (y0, y1) in enumerate(_split(ya, yb, ncol, GAP)):
        for j, (z0, z1) in enumerate(rows):
            nm = "st_front_c%d_r%d" % (i, j)
            P.append(_box(nm, [X0, y0, z0], [X0 + FRONT_T, y1, z1], "station_oak", "body", "h"))
            L = HANDLE_L_WIDE if (y1 - y0) > 0.5 else HANDLE_L
            P += bar_handle(nm + "_h", [X0, (y0 + y1) / 2, z1 - 0.045], "y", L)
    # closure: the slot between the support's back face and the wall face (the env's back wall front face)
    wx = g["wall_x"]
    if wx > X1 + 1e-6:
        P.append(_box("st_top_back", [X1, Y0, zt], [wx, Y1, bz], "station_stone", "closure"))
        P.append(_box("st_side_w_back", [X1, Y0, 0.0], [wx, Y0 + SIDE_T, zt], "station_oak", "closure", "v"))
        P.append(_box("st_side_e_back", [X1, Y1 - SIDE_T, 0.0], [wx, Y1, zt], "station_oak", "closure", "v"))
    return P


def station_footprint_parts(parts):
    return [p for p in parts if p["role"] == "body"]


def filler_spans(kind, unit_lo, unit_hi, jitter=(0.0, 0.0, 0.0)):
    """[(name, y0, y1)] of the base cabinets beside the station (cell y), gaps under FILLER_MIN left open."""
    g = cell_geometry(kind, unit_lo, unit_hi, jitter)
    yw, ye = BAY_X_WEST - BAY_CX, BAY_X_EAST - BAY_CX
    if g["support"] is None:
        return [("fill_bay", yw, ye)]
    c, h = g["support"]["center"], g["support"]["half"]
    out = []
    if (c[1] - h[1]) - yw >= FILLER_MIN:
        out.append(("fill_w", yw, c[1] - h[1]))
    if ye - (c[1] + h[1]) >= FILLER_MIN:
        out.append(("fill_e", c[1] + h[1], ye))
    return out


def filler_parts(kind, unit_lo, unit_hi, jitter=(0.0, 0.0, 0.0)):
    """Base cabinets like the pack's own beside the station: white fronts, white plinth 0.08 set back 18 mm, the
    pack's vertical bar handles, a 0.04 m slab at 0.82, the pack's front line (0.70 from its main wall plane),
    back to the bay's wall face.  One door up to 0.60 m, two doors above, a plain strip under 0.15 m."""
    g = cell_geometry(kind, unit_lo, unit_hi, jitter)
    T = room_pose(kind, unit_lo, unit_hi, jitter)
    xf = float(to_cell(T, [0.0, PACK_FRONT_Y, 0.0])[0])     # the pack's front line in the cell
    wx = g["wall_x"]
    sy = g["support"]["center"][1] if g["support"] else 0.0
    low = kind in LOW_BENCH_KINDS
    if low:
        # one low appliance bench: the fillers at the station's height, the station's 0.03 stone top; below the
        # tiles (z < 0.82) the bay's wall is its plaster 10 mm behind the tile face, so the bench runs back to it
        top_z, slab_t, slab_mat = BOTTOM_Z[kind], SLAB_T, "station_stone"
        wb = wx + (BAY_TILE_Y - BAY_PLASTER_Y)
    else:
        top_z, slab_t, slab_mat = PACK_TOP_Z, PACK_SLAB_T, "counter_stone"
        wb = wx
    door_z = (PACK_DOOR_Z[0], top_z - slab_t - (PACK_TOP_Z - PACK_SLAB_T - PACK_DOOR_Z[1]))
    handle_top = door_z[1] - (PACK_DOOR_Z[1] - PACK_HANDLE_Z[1])
    handle_z = handle_top - HANDLE_L / 2
    P = []
    for name, y0, y1 in filler_spans(kind, unit_lo, unit_hi, jitter):
        P.append(_box(name + "_slab", [xf, y0, top_z - slab_t], [wb, y1, top_z], slab_mat, "filler"))
        P.append(_box(name + "_plinth", [xf + PACK_PLINTH_SETBACK, y0, 0.0], [wb, y1, PACK_PLINTH_H], "pack_plinth",
                      "filler"))
        P.append(_box(name + "_carcass", [xf + FRONT_T, y0, PACK_PLINTH_H], [wb, y1, top_z - slab_t],
                      "pack_white", "filler"))
        w = y1 - y0
        # half gaps at the ends that meet another front (station side panel, the pack's end panel), none at the wall
        ga = 0.0 if abs(y0 - (BAY_X_WEST - BAY_CX)) < 1e-9 else GAP / 2
        gb = GAP / 2
        za, zb = door_z
        if w < FILLER_STRIP_MAX:
            P.append(_box(name + "_strip", [xf, y0 + ga, za], [xf + FRONT_T, y1 - gb, zb], "pack_white", "filler"))
            continue
        n = 1 if w <= DOOR_MAX else 2
        doors = _split(y0 + ga, y1 - gb, n, GAP)
        for i, (d0, d1) in enumerate(doors):
            nm = "%s_door%d" % (name, i)
            P.append(_box(nm, [xf, d0, za], [xf + FRONT_T, d1, zb], "pack_white", "filler"))
            # the handle on the opening edge: toward the station for one door, toward the meeting edge for two
            if n == 1:
                edge = d1 if abs(d1 - sy) < abs(d0 - sy) else d0
                yh = edge - 0.022 if edge == d1 else edge + 0.022
            else:
                yh = d1 - 0.022 if i == 0 else d0 + 0.022
            P += bar_handle(nm + "_h", [xf, yh, handle_z], "z", HANDLE_L, role="filler")
    return P


def shelf_parts_bundle():
    """The floating oak shelf above the bay (bundle frame): the board (the pack's light oak, grain along the run) with
    a 20 x 6 mm groove in its underside, 30 mm behind the front edge, for the LED: an aluminium profile lines the
    groove's top and a 12 mm opal diffuser sits 1.5 mm up inside it, so the strip is hidden from low, grazing views.
    Hidden fixings; the board's back face is on the plaster.  Boxes {name, lo, hi, material, grain}."""
    lo, hi = np.array(SHELF_BUNDLE["lo"], float), np.array(SHELF_BUNDLE["hi"], float)
    x0, x1 = lo[0], hi[0]
    yw = BAY_PLASTER_Y                                 # the wall side
    yf = lo[1]                                         # the front edge (toward the room, bundle +y)
    z0, z1 = lo[2], hi[2]
    L = SHELF_LED
    gd = 0.006                                         # groove depth
    g0, g1 = yf - L["setback"] - L["profile_w"], yf - L["setback"]      # groove's y band (wall side .. front side)
    gx0, gx1 = SHELF_CX - L["len"] / 2 - 0.01, SHELF_CX + L["len"] / 2 + 0.01
    zg = z0 + gd
    B = lambda n, a, b, m="shelf_oak", gr="h": {"name": n, "lo": list(map(float, a)), "hi": list(map(float, b)),  # noqa: E731
                                                "material": m, "grain": gr}
    return [
        B("shelf_board", [x0, yw, zg], [x1, yf, z1]),
        B("shelf_board_lip_front", [x0, g1, z0], [x1, yf, zg]),
        B("shelf_board_lip_back", [x0, yw, z0], [x1, g0, zg]),
        B("shelf_board_lip_west", [x0, g0, z0], [gx0, g1, zg]),
        B("shelf_board_lip_east", [gx1, g0, z0], [x1, g1, zg]),
        B("shelf_led_profile", [gx0, g0, zg - 0.003], [gx1, g1, zg], "led_profile_alu", None),
        B("shelf_led_diffuser", [SHELF_CX - L["len"] / 2, (g0 + g1) / 2 - L["width"] / 2, z0 + 0.0015],
          [SHELF_CX + L["len"] / 2, (g0 + g1) / 2 + L["width"] / 2, zg - 0.003], "shelf_led_diffuser", None),
    ]


def shelf_parts(kind, unit_lo, unit_hi, jitter=(0.0, 0.0, 0.0)):
    """shelf_parts_bundle() in the cell frame (it moves with the room: the unit's depth and jx, not jy)."""
    T = room_pose(kind, unit_lo, unit_hi, jitter)
    P = []
    for q in shelf_parts_bundle():
        a, b = bundle_box_to_cell(T, q["lo"], q["hi"])
        P.append(_box(q["name"], a, b, q["material"], "shelf", q.get("grain")))
    return P


def pedestal_parts():
    """A powder-coated steel column on a base plate; the base plate has the environment's pedestal footprint."""
    sx, sy, sz = PEDESTAL
    cx = PEDESTAL_X
    t = 0.015
    P = [_box("ped_base", [cx - sx / 2, -sy / 2, 0.0], [cx + sx / 2, sy / 2, t], "pedestal_powder", "pedestal"),
         _box("ped_column", [cx - 0.10, -0.10, t], [cx + 0.10, 0.10, sz - t], "pedestal_powder", "pedestal"),
         _box("ped_flange", [cx - 0.12, -0.12, sz - t], [cx + 0.12, 0.12, sz], "pedestal_powder", "pedestal")]
    for i, (dx, dy) in enumerate(((1, 1), (1, -1), (-1, 1), (-1, -1))):
        P.append(_cyl("ped_bolt%d" % i, [cx + dx * 0.115, dy * 0.115, t + 0.006], 0.009, 0.006, "z", "pedestal_bolt",
                      "pedestal"))
    # four gussets (6 mm plates, 60 x 80 mm) between the column and the base plate, one on each face
    for i, (ax, s) in enumerate((("x", 1), ("x", -1), ("y", 1), ("y", -1))):
        if ax == "x":
            lo, hi = [cx + s * 0.10 + min(0, s * 0.06), -0.003, t], [cx + s * 0.10 + max(0, s * 0.06), 0.003, t + 0.08]
        else:
            lo, hi = [cx - 0.003, s * 0.10 + min(0, s * 0.06), t], [cx + 0.003, s * 0.10 + max(0, s * 0.06), t + 0.08]
        P.append(_box("ped_gusset%d" % i, lo, hi, "pedestal_powder", "pedestal"))
    return P


def all_parts(kind, unit_lo, unit_hi, jitter=(0.0, 0.0, 0.0)):
    return (station_parts(kind, unit_lo, unit_hi, jitter) + filler_parts(kind, unit_lo, unit_hi, jitter) +
            shelf_parts(kind, unit_lo, unit_hi, jitter) + pedestal_parts())


# ---------------------------------------------------------------- materials
def _lin(srgb):
    c = np.asarray(srgb, float)
    return _v(np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4).round(4))


def materials(room_dir="room_hi"):
    """PBR parameters of the dressing (linear base colour, roughness, metallic) and texture files (relative to the
    room package; tile_m = metres per texture repeat, box-projected in the cell frame).  The room's own materials
    are in the bundle's room.json; the renderer takes the oak's texture from there too."""
    tex = lambda f: room_dir + "/textures/" + f  # noqa: E731
    return {
        # the pack's light oak ("3D MATERIALS/Wood"): base, normal and ORM maps, ~1.1 m per repeat in the pack
        "station_oak": {"base": [1.0, 1.0, 1.0], "roughness": 0.62, "metallic": 0.0,
                        "textures": {"rgb": tex("Wood_f9f4c5_rgb.png"), "normal": tex("Wood_f9f4c5_nrm.png"),
                                     "orm": tex("Wood_f9f4c5_orm.png")}, "tile_m": 0.9, "normal_strength": 0.6,
                        "note": "pack Wood material; mean base sRGB (0.80, 0.71, 0.58); ORM roughness 0.69"},
        "station_carcass": {"base": _lin([0.78, 0.70, 0.57]), "roughness": 0.7, "metallic": 0.0,
                            "note": "inside of the carcass, seen only through the 3 mm gaps"},
        # matched to the pack's counter: Kitchen atlas region u 0.16-0.81, v 0.56-0.69 (the slab's top face):
        # mean sRGB (0.428, 0.432, 0.412) x the material tint 0.9, roughness (ORM G) 0.18, AO 1.0, metallic 0
        "station_stone": {"base": [round(0.9 * c, 4) for c in _lin([0.428, 0.432, 0.412])], "roughness": 0.20,
                          "metallic": 0.0, "noise": 0.05, "note": "grey stone matched to the pack's counter top"},
        "counter_stone": {"base": [round(0.9 * c, 4) for c in _lin([0.428, 0.432, 0.412])], "roughness": 0.20,
                          "metallic": 0.0, "noise": 0.05, "note": "the filler slabs: the same stone as the pack's"},
        "plinth_dark": {"base": _lin([0.20, 0.20, 0.21]), "roughness": 0.55, "metallic": 0.0},
        "handle_steel": {"base": [0.62, 0.62, 0.62], "roughness": 0.32, "metallic": 1.0, "anisotropic": 0.6,
                         "note": "brushed stainless; the pack's 6 mm bar profile"},
        # the pack's white lacquer fronts (atlas sRGB 0.96 x tint 0.9) and its plinth (0.98 x 0.9)
        "pack_white": {"base": [round(0.9 * c, 4) for c in _lin([0.962, 0.965, 0.965])], "roughness": 0.30,
                       "metallic": 0.0},
        "pack_plinth": {"base": [round(0.9 * c, 4) for c in _lin([0.983, 0.983, 0.983])], "roughness": 0.43,
                        "metallic": 0.0},
        "pedestal_powder": {"base": _lin([0.22, 0.24, 0.26]), "roughness": 0.45, "metallic": 0.0, "coat": 0.15,
                            "note": "anthracite powder coat (RAL 7016-like)"},
        "pedestal_bolt": {"base": [0.75, 0.75, 0.76], "roughness": 0.3, "metallic": 1.0},
        "led_strip": {"base": [0.9, 0.9, 0.9], "roughness": 0.4, "metallic": 0.0, "emission_K": 4000,
                      "emission_strength": 3.0,
                      "note": "the under-cabinet LED diffuser (4000 K), dim enough not to read as a blown-out bar in "
                              "the right fixed camera"},
        "shelf_oak": {"base": [1.0, 1.0, 1.0], "roughness": 0.62, "metallic": 0.0,
                      "textures": {"rgb": tex("Wood_f9f4c5_rgb.png"), "normal": tex("Wood_f9f4c5_nrm.png"),
                                   "orm": tex("Wood_f9f4c5_orm.png")}, "tile_m": 0.9, "normal_strength": 0.6,
                      "note": "the floating shelf: the station's light oak (the pack's Wood), grain along the board"},
        "led_profile_alu": {"base": [0.80, 0.80, 0.80], "roughness": 0.35, "metallic": 1.0,
                            "note": "anodised aluminium LED profile set into the shelf's underside"},
        "shelf_led_diffuser": {"base": [0.9, 0.9, 0.9], "roughness": 0.4, "metallic": 0.0, "emission_K": 4000,
                               "emission_strength": 6.0,
                               "note": "the opal diffuser of the shelf LED; dim so it does not read as a bright bar"},
        "bulb_on": {"base": [1.0, 1.0, 1.0], "roughness": 0.3, "metallic": 0.0, "emission_K": 2700},
    }


# ---------------------------------------------------------------- lights (bundle frame: they belong to the room)
def kelvin_rgb(K):
    """Approximate linear-sRGB colour of a black body (Tanner Helland's fit, normalised to max 1)."""
    t = K / 100.0
    r = 255.0 if t <= 66 else 329.698727446 * (t - 60) ** -0.1332047592
    g = 99.4708025861 * math.log(t) - 161.1195681661 if t <= 66 else 288.1221695283 * (t - 60) ** -0.0755148492
    b = 255.0 if t >= 66 else (0.0 if t <= 19 else 138.5177312231 * math.log(t - 10) - 305.0447927307)
    c = np.clip(np.array([r, g, b]) / 255.0, 0, 1)
    c = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    return _v((c / c.max()).round(4))


LM_PER_W = 683.0                   # Blender's light "W" read as radiant watts at 555 nm; sun strength W/m2 = lux / 683


def spot_watts(lumens, cone_deg):
    """Blender spot power for a real spot of `lumens` in a cone of `cone_deg`: Blender spreads a spot's power as
    if it were a point light (4 pi sr) and only masks the cone, so the power is scaled by 4 pi / cone solid angle."""
    omega = 2 * math.pi * (1 - math.cos(math.radians(cone_deg) / 2))
    return round(lumens / LM_PER_W * 4 * math.pi / omega, 3)


def sun_direction(elev_deg, azim_deg):
    """Unit vector from the scene toward the sun; azimuth from bundle +x (east), counter-clockwise."""
    e, a = math.radians(elev_deg), math.radians(azim_deg)
    return [math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e)]


SUN = {"elev_deg": 24.0, "azim_deg": 197.0}   # low, from the west-south-west: patches fall to the ENE of the openings
SKY_LUX = 20000.0                  # horizontal illuminance of the overcast-ish sky dome outside


def lights():
    """The kitchen's lights (bundle frame).  Each: name, type, pos, dir (where it shines), size, cct_K, lumens (or lux
    for the sun and the sky), and the path tracer's values of the same lights (photometric -> Blender W at 683 lm/W,
    spots corrected for their cone, see spot_watts), from which the image light was rendered.  The sky and the sun
    are in the image light; kitchen_render.py makes a Filament lamp of every other one."""
    sd = sun_direction(SUN["elev_deg"], SUN["azim_deg"])
    L = [
        {"name": "sky_dome", "type": "sky", "model": "uniform dome, clear-sky blue (9000 K)", "cct_K": 9000,
         "lux_horizontal": SKY_LUX, "blender": {"strength": round(SKY_LUX / LM_PER_W / math.pi, 3)},
         "note": "daylight through the kitchen terrace door (x -3.627) and the relax-room window (x -4.945); the "
                 "pack's white shell outside the windows stays as a camera-only backdrop"},
        {"name": "sun_low_west", "type": "sun", "dir": [-c for c in sd], "elev_deg": SUN["elev_deg"],
         "azim_deg": SUN["azim_deg"], "angle_deg": 1.0, "cct_K": 4800, "lux": 35000,
         "blender": {"strength_W_m2": round(35000 / LM_PER_W, 2)},
         "note": "elevation 24 deg from 17 deg south of west: the patches through the terrace door and the relax window "
                 "fall on the floor to the east-north-east of the openings, north of the station"},
    ]
    # the two Ranarp dining pendants (bulb centres at z 1.52; Unity spots at z 1.578 pointing down, 100 deg)
    for i, x in enumerate((-1.0061, -0.1844)):
        L.append({"name": "pendant_%d_bulb" % i, "type": "point", "pos": [x, -1.438, 1.52], "radius": 0.025,
                  "cct_K": 2700, "lumens": 650, "blender": {"power_W": round(650 / LM_PER_W, 3)},
                  "emitter_mesh": {"sphere_r": 0.028, "material": "bulb_on"},
                  "note": "a 60 W-equivalent filament bulb; the pack's bulb meshes of these pendants were lost in the "
                          "export (hashed material overrides), a small emitting sphere stands in"})
        L.append({"name": "pendant_%d_down" % i, "type": "spot", "pos": [x, -1.438, 1.47], "dir": [0, 0, -1],
                  "spot_deg": 100, "blend": 0.6, "radius": 0.04, "cct_K": 2700, "lumens": 250,
                  "blender": {"power_W": spot_watts(250, 100)}, "note": "the shade's downward cone (Unity spot 100 deg)"})
    # the kitchen ceiling spots (Ceiling_Props_Spotlight over the kitchen aisle), switched to the ON material; 4000 K
    for i, x in enumerate((-1.211, 0.9975)):
        L.append({"name": "ceiling_spot_%d" % i, "type": "spot", "pos": [x, -2.533, 2.63], "dir": [0, 0, -1],
                  "spot_deg": 60, "blend": 0.35, "radius": 0.025, "cct_K": 4000, "lumens": 450,
                  "blender": {"power_W": spot_watts(450, 60)}, "fixture": "Ceiling_Props_Spotlight -> ON material"})
    # the station's work light: a third fixture of the same kind on the aisle line, over the station
    L.append({"name": "work_light", "type": "spot", "pos": [-2.50, -2.533, 2.63], "dir": [0, 0, -1],
              "spot_deg": 70, "blend": 0.35, "radius": 0.025, "cct_K": 4000, "lumens": 700,
              "blender": {"power_W": spot_watts(700, 70)},
              "fixture": "Ceiling_Props_Spotlight copy at (-2.50, -2.533) with the ON material (room_hi)"})
    # linear light under the wall cabinets over the sink run (x -1.605 .. -0.372); 4000 K
    L.append({"name": "undercabinet_led", "type": "area_line", "pos": [-0.99, -3.31, 1.412], "dir": [0, 0, -1],
              "size": [1.18, 0.012], "cct_K": 4000, "lumens": 1100, "blender": {"power_W": round(1100 / LM_PER_W, 3)},
              "emitter_mesh": {"box": [1.18, 0.012, 0.004], "material": "led_strip"},
              "note": "10 W/m LED tape in an aluminium profile, 50 mm behind the cabinet fronts"})
    # the linear LED under the floating shelf: lights the station top and the tiles
    sp = [p for p in shelf_parts_bundle() if p["name"] == "shelf_led_diffuser"][0]
    L.append({"name": "shelf_led", "type": "area_line",
              "pos": [SHELF_CX, (sp["lo"][1] + sp["hi"][1]) / 2, SHELF_BUNDLE["lo"][2] - 0.0003], "dir": [0, 0, -1],
              "size": [SHELF_LED["len"], SHELF_LED["width"]], "cct_K": 4000, "lumens": 900,
              "blender": {"power_W": round(900 / LM_PER_W, 3)},
              "note": "LED tape in the shelf's recessed aluminium profile (shelf_parts: shelf_led_profile, "
                      "shelf_led_diffuser); the diffuser is drawn by the shelf, so no emitter mesh here"})
    for l in L:
        l["rgb"] = kelvin_rgb(l["cct_K"])
        l.setdefault("demo_only", False)
    return L


def task_lights():
    """The lights the task's cameras see (every light: none is demo_only)."""
    return [l for l in lights() if not l["demo_only"]]


def lights_cell(kind, unit_lo, unit_hi, jitter=(0.0, 0.0, 0.0)):
    """lights() turned into the cell frame."""
    T = room_pose(kind, unit_lo, unit_hi, jitter)
    out = []
    for l in lights():
        l = dict(l)
        if "pos" in l:
            l["pos"] = _v(to_cell(T, l["pos"]))
        if "dir" in l:
            l["dir"] = _v(T[:3, :3] @ np.asarray(l["dir"], float))
        if l["type"] == "sun":
            l["azim_deg"] = l["azim_deg"] + 90.0
        out.append(l)
    return out
