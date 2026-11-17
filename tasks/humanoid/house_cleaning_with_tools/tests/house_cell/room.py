"""The customer's studio apartment: fixed geometry in metres (the survey the cell manual publishes exactly).

Room frame: origin at the south-west floor corner, x east, y north, z up (the room of furniture_assembly_from_drawings:
the same walls, door, window, couch, plant and rug). The cleaning job's fixtures: a kitchenette along the west wall
(base cabinets with a quartz top, an undermount sink, a single-lever faucet, a dish-soap pump bottle, a sponge tray,
an induction hob, wall cabinets, a tiled splashback), two flip-lid waste bins south of it, a wall rail with two spring
clips for the broom and the lobby dustpan on the south wall, a dining table with two chairs in the middle of the room,
a coffee table on the rug in front of the couch, and a bookshelf against the north wall. Boxes are
(x0, y0, z0, x1, y1, z1).
"""

from __future__ import annotations

import math

ROOM_X, ROOM_Y, WALL_H = 6.0, 5.0, 2.6
WALL_T = 0.12
DOOR_X = (4.80, 5.70)            # opening in the south wall, 2.05 m high
DOOR_H = 2.05
WINDOW_Y = (1.60, 3.00)          # window in the east wall (sill 0.90 m, head 2.10 m)

# ---------------------------------------------------------------------------------- the customer's own (furniture room)
COUCH = {
    "footprint": (5.09, 3.36, 5.95, 4.94),
    "boxes": {
        "seat": (5.09, 3.49, 0.16, 5.75, 4.81, 0.48),
        "back": (5.75, 3.49, 0.16, 5.95, 4.81, 0.94),
        "arm_s": (5.09, 3.36, 0.16, 5.95, 3.49, 0.72),
        "arm_n": (5.09, 4.81, 0.16, 5.95, 4.94, 0.72),
    },
    "legs": {"centers": ((5.12, 3.39), (5.12, 4.91), (5.93, 3.39), (5.93, 4.91)), "side": 0.035, "height": 0.16},
}
PLANT = {"pot": {"center": (0.51, 4.51), "radius": 0.13, "z": (0.0, 0.25)},
         "foliage": {"center": (0.45, 4.55), "radius": 0.40, "z": (0.25, 1.62)}}
RUG = {"center": (4.85, 4.15), "size": (1.27, 1.52), "thickness": 0.01}

# ---------------------------------------------------------------------------------- kitchenette (west wall)
# a compact kitchenette (0.52 m deep: the G1's reach), its cabinets' fronts at x = 0.50
KITCHEN = {"x0": 0.0, "x1": 0.50, "y0": 1.15, "y1": 3.35,
           "plinth_h": 0.10, "plinth_x1": 0.44,          # recessed toe kick
           "top_z": 0.90, "top_t": 0.04, "top_x1": 0.52,  # quartz top, 2 cm overhang
           "modules": (1.15, 1.70, 2.25, 2.80, 3.35)}     # cabinet module boundaries along y (doors + drawers)
# an apron-front (farmhouse) sink: white fireclay, its front wall is the cabinets' front (no counter strip in front)
SINK = {"x0": 0.12, "x1": 0.50, "y0": 2.28, "y1": 2.78, "bottom_z": 0.70, "wall_t": 0.012, "apron_x1": 0.515}
SINK_CENTER = ((SINK["x0"] + SINK["x1"]) / 2, (SINK["y0"] + SINK["y1"]) / 2)
# single-lever faucet behind the sink: a column, a spout over the basin, a lever on top (lift = water on)
# the faucet stands behind the basin's north end: a low spout arching over to the north part of the basin, a single
# lever on top of the column pointing towards the room (lift it: water)
FAUCET = {"base": (0.065, 2.66), "column_r": 0.022, "column_top": 1.06,
          "outlet": (0.30, 2.66, 1.00),                  # the spout's mouth (water falls straight down from it)
          "pivot": (0.065, 2.66, 1.085), "lever_len": 0.17, "lever_r": 0.0075,
          "range": (0.0, 0.62), "on_above": 0.15, "full_at": 0.40}
STREAM_R = 0.045                 # m: the rinse zone around the falling water's axis
# dish-soap pump with a sponge rest on top: pressing the sponge down on the pump's wide head (12 mm of its 20 mm travel,
# a spring returns it) puts one dose of soap into the sponge
SOAP = {"base": (0.30, 2.99), "bottle_r": 0.040, "bottle_h": 0.12, "neck_top": 1.035,
        "head_r": 0.040, "head_h": 0.014, "travel": 0.020, "stroke_at": 0.012}
# the sponge's home: a shallow steel tray on the counter north of the sink
SPONGE_TRAY = {"center": (0.37, 3.14), "size": (0.13, 0.16), "rim_h": 0.015}
# where the dirty dishes go: a grey rubber dish tray on the counter south of the sink
DISH_TRAY = {"center": (0.27, 2.09), "size": (0.40, 0.32), "rim_h": 0.018, "base_t": 0.004}
HOB = (0.04, 1.32, 0.48, 1.90)   # induction hob in the counter (x0, y0, x1, y1): flush, drawn only
UPPER = {"x0": 0.0, "x1": 0.32, "y0": 1.15, "y1": 3.35, "z0": 1.52, "z1": 2.22}
SPLASHBACK = {"y0": 1.15, "y1": 3.35, "z0": 0.90, "z1": 1.52}

# ---------------------------------------------------------------------------------- waste bins (west wall, south end)
# two pedal-less flip-lid bins, their backs to the west wall; the lid hinges at its back edge, opens up towards the wall,
# a hold-open spring keeps it open beyond LID_HOLD and lets it fall shut below; which bin stands where is drawn per
# episode (the blue one is the recycling bin, the grey one the general-waste bin)
BIN = {"x0": 0.05, "x1": 0.33, "half_y": 0.16, "rim_z": 0.60, "wall_t": 0.010,
       "lid_t": 0.012, "hinge_x": 0.06, "tab_r": 0.008, "tab_len": 0.12, "tab_out": 0.022}
BIN_SLOTS_Y = (0.36, 0.80)       # the two bins' centres along y
LID_RANGE = (0.0, math.radians(82.0))
LID_HOLD = math.radians(55.0)    # past this the hold-open spring takes the lid to its stop

# ---------------------------------------------------------------------------------- tool rail (south wall)
# a wall rail with two spring clips at 1.00 m: the broom (swivel head) and the lobby dustpan hang in them
TOOL_RAIL = {"x0": 0.82, "x1": 1.62, "z": 1.00, "depth": 0.025}
CLIPS = {"broom": (1.00, 0.048, 1.00), "dustpan": (1.42, 0.048, 1.00)}   # jaw centres (x, y, z)
CLIP_CAPTURE = 0.012             # m: a handle axis this close to the jaw centre clicks in (released by a hand)

# ---------------------------------------------------------------------------------- dining table and chairs
DINING = {"center": (2.85, 2.45), "size": (1.20, 0.64), "top_z": 0.75, "top_t": 0.03,
          "leg": 0.045, "leg_inset": 0.06}
CHAIRS = {"west": {"center": (2.10, 2.45), "yaw": 0.0}, "east": {"center": (3.60, 2.45), "yaw": math.pi}}
CHAIR = {"seat": (0.44, 0.44), "seat_z": 0.46, "seat_t": 0.035, "back_h": 0.88, "back_t": 0.03, "leg": 0.032}

# ---------------------------------------------------------------------------------- coffee table (on the rug)
COFFEE = {"center": (4.58, 4.15), "size": (0.50, 0.80), "top_z": 0.45, "top_t": 0.03, "leg": 0.035,
          "leg_inset": 0.04}

# ---------------------------------------------------------------------------------- bookshelf (north wall)
SHELF = {"x0": 1.05, "x1": 2.35, "y0": 4.66, "y1": 5.00, "height": 0.92, "board": 0.018, "plinth": 0.06,
         "shelves_z": (0.06, 0.36, 0.64), "divider_x": 1.70}
# where the two loose books belong: the top compartment of the west bay, east of the books already there
BOOK_TARGET = {"shelf_z": SHELF["shelves_z"][2] + SHELF["board"], "x0": 1.25, "x1": 1.69,
               "y0": SHELF["y0"] + 0.01, "y1": SHELF["y1"] - 0.01,
               "clear_h": SHELF["height"] - SHELF["board"] - SHELF["shelves_z"][2] - SHELF["board"]}

# ---------------------------------------------------------------------------------- the robot and the cameras
ROBOT_HOME = (5.25, 0.75, math.pi / 2)     # just inside the door, facing north (before the per-episode draw)

# fixed ceiling cameras (exact calibration is published): position, look-at point, vertical fov (deg), size
CEILING_CAMERAS = {
    "ceiling_kitchen": {"pos": (2.05, 0.55, 2.48), "target": (0.35, 2.05, 0.55), "fovy": 72.0, "size": (960, 720)},
    "ceiling_dining": {"pos": (4.10, 1.15, 2.48), "target": (2.15, 3.35, 0.45), "fovy": 72.0, "size": (960, 720)},
    "ceiling_living": {"pos": (2.90, 2.55, 2.48), "target": (4.55, 4.15, 0.35), "fovy": 72.0, "size": (960, 720)},
}

# the supports the cell's crumbs come to rest on (horizontal surfaces, highest wins): name -> (box of the surface's
# footprint and height). Computed from the survey above in surfaces().


def surfaces() -> dict:
    k, d, c, s = KITCHEN, DINING, COFFEE, SHELF
    (dx, dy), (sx, sy) = d["center"], d["size"]
    (cx, cy), (cw, cl) = c["center"], c["size"]
    out = {
        "dining_top": ((dx - sx / 2, dy - sy / 2, dx + sx / 2, dy + sy / 2), d["top_z"]),
        "coffee_top": ((cx - cw / 2, cy - cl / 2, cx + cw / 2, cy + cl / 2), c["top_z"]),
        "counter": ((k["x0"], k["y0"], k["top_x1"], k["y1"]), k["top_z"]),
        "sink": ((SINK["x0"], SINK["y0"], SINK["x1"], SINK["y1"]), SINK["bottom_z"]),
        "dish_tray": ((DISH_TRAY["center"][0] - DISH_TRAY["size"][0] / 2, DISH_TRAY["center"][1] - DISH_TRAY["size"][1] / 2,
                       DISH_TRAY["center"][0] + DISH_TRAY["size"][0] / 2, DISH_TRAY["center"][1] + DISH_TRAY["size"][1] / 2),
                      k["top_z"] + DISH_TRAY["base_t"]),
        "shelf_top": ((s["x0"], s["y0"], s["x1"], s["y1"]), s["height"]),
    }
    for name, ch in CHAIRS.items():
        x, y = ch["center"]
        w, l = CHAIR["seat"]
        out[f"chair_{name}"] = ((x - w / 2, y - l / 2, x + w / 2, y + l / 2), CHAIR["seat_z"])
    return out
