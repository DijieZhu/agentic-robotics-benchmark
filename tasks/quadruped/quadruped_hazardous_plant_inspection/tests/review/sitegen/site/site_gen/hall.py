"""The training hall: a hand-laid site in the site package format.

    python -m site_gen.hall --out <dir>               # the hall folder (site.json, prior.json, scene.xml, ...)
                                                      # and its documents under <dir>/hall_docs/

One room of 30 x 21 m behind the same airlock as the plant sites (the pad is the datum, the base station in the
airlock), lit bright, drawn correct and complete, with no deviation of any kind:

  obstacle lanes   42 dead-end lanes 1.2 m wide off a central aisle (row S south of it, row N north of it), one
                   obstacle each at 1.6 m from the lane mouth: square sills, hoses, round floor pipes (square-on and
                   at 20 deg), kerbs (a step up onto a platform), ramped cable protectors, low racks, ramps up to a
                   platform, narrow gaps, wet patches, and one empty lane; 4-6 sizes per class, including sizes just
                   outside the rated ones (classed 'unrated': no rating applies). The dividers are 0.9 m walls.
  instrument wall  the south wall, eight bays of 3.2 m, one per instrument type, each type at 0.5, 1.0 and 1.5 m
                   (the station's mounting height; the rising-stem valve's indicator plate sits 0.42 m above its
                   pipe); the sheet gives each instrument's position (as the round sheet) and its face centre (the
                   evidence rules' reference point, mission/evidence.py face_frame), every value;
                   bay 9: a quarter-turn valve on a vertical pipe and a lamp
                   panel mounted at 1.9 m under a canopy, read from below
  check source     a 25 Gy/h-at-1-m source in a shield-block cave on the east wall, and a survey mark 4 m in front
  grid markers     15 surveyed tag plates (south wall, west and east walls, the island), all present
  odometry         a 28 m straight along y = -6.0 (floor line, 1 m ticks) and a closed loop around a pallet island
  link             the airlock's base station and one working repeater on the east wall

The round sheet lists the 24 wall instruments, the survey mark and the check source (as 'the source'); the
practice ground caps a hall mission at 900 s. The calibration sheet, the lane table and the plan go to hall_docs/.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import time

import mujoco
import numpy as np

from . import furnish as F
from . import radiation as RAD
from . import stations as ST
from .config import load_config
from .ip_models import IPWorks
from .layout import Layout, Room, _build_walls, _grid, _inner_rects, _side_of, add_door, wall_between
from .link import coverage, opaque_raster
from .mjcf import build_scene, write_xml
from .structure import build_structure
from .trav import MODES, Grid, TravMap
from .util import Rng, dump_json, poly_rect, quat_axis_angle
from .world import World

AIRLOCK_RECT = [-1.5, -1.5, 2.0, 1.5]
HALL_RECT = [2.0, -10.5, 32.0, 10.5]
LANE_X0, LANE_W, LANES_PER_ROW = 5.0, 1.2, 22      # version 2: 22 a row (a trench cover and a pipe bridge added)
ROW_S = (1.8, 5.4)                 # y range; entered from the north end (the aisle)
AISLE = (5.4, 6.8)
ROW_N_Y0 = 6.8                     # to the north wall; entered from the south end
OBST_AT = 1.6                      # obstacle centre from the lane mouth
DIVIDER_T, DIVIDER_H = 0.06, 0.9
PLATFORM_H = 0.20                  # the ramps' platform
STRAIGHT = ((3.0, -6.0), (31.0, -6.0))
ISLAND = [10.0, -3.5, 18.0, -1.0]
CAVE = {"x0": 29.4, "y_lo": (-2.1, -1.8), "y_hi": (0.6, 0.9), "h": 1.6}
SOURCE_XYZ = (31.2, -0.6, 0.30)
SOURCE_GYH_AT_1M = 25.0
MARK_XY = (27.2, -0.6)
REPEATER_XYZ = (31.70, -5.0, 2.0)
BAY_X0, BAY_W = 3.0, 3.2
WALL_TYPES = ("gauge_63", "gauge_100", "gauge_160", "level", "valve_lever", "valve_handwheel", "lamps", "thermal")
HEIGHTS = (0.5, 1.0, 1.5)
PREFIX = {"gauge": "PI", "level": "LG", "valve_lever": "HV", "valve_handwheel": "V", "lamps": "LCP", "thermal": "TS"}
# the calibration values (fixed: this is a calibration stand)
VALUES = {
    "gauge_63": [(0, 6, "bar", 1.5), (0, 10, "bar", 5.0), (0, 16, "bar", 12.0)],
    "gauge_100": [(0, 25, "bar", 6.25), (-1, 3, "bar", 1.0), (0, 120, "°C", 84.0)],
    "gauge_160": [(0, 6, "bar", 4.2), (0, 16, "bar", 3.2), (0, 25, "bar", 18.75)],
    "level": [22.0, 55.0, 86.0],
    "valve_lever": [0.0, 45.0, 90.0],
    "valve_handwheel": [12.0, 50.0, 88.0],
    "lamps": [(["green", "red", "amber"], [True, False, True]),
              (["green", "red", "amber", "white"], [False, True, True, False]),
              (["green", "red", "amber", "white", "green", "red"], [True, True, False, True, False, False])],
    "thermal": [("painted steel", 45.0), ("powder-coated steel", 75.0), ("oxidised cast iron", 102.0)],
}
# the lanes: (kind, size); sizes in m (slope in deg, friction for wet patches)
# version 2: two lanes more, which only confirm the frozen firmware's tables for the in-room obstacles: a
# cable trench cover of 5.2 cm (the kerb's section, 0.20 m deep; row S) and a pipe bridge 0.8 m long at 0.29 m (the
# rack's construction; row N)
LANES = ([("sill", h) for h in (0.030, 0.045, 0.052, 0.057, 0.070)] +
         [("hose", h) for h in (0.030, 0.045, 0.052, 0.062)] +
         [("trench_cover", 0.052)] +
         [("floor_pipe", d) for d in (0.030, 0.045, 0.052, 0.060, 0.070)] +
         [("floor_pipe_20deg", d) for d in (0.052, 0.060)] +
         [("kerb", h) for h in (0.040, 0.052, 0.070, 0.110)] +
         [("flat", 0.0)] +
         [("cable_protector", h) for h in (0.030, 0.050, 0.070, 0.090)] +
         [("low_overhead", c) for c in (0.420, 0.340, 0.310, 0.290, 0.260)] +
         [("pipe_bridge", 0.290)] +
         [("ramp", a) for a in (8.0, 12.0, 15.0, 18.0)] +
         [("narrow_gap", w) for w in (0.65, 0.55, 0.48, 0.42)] +
         [("puddle", mu) for mu in (0.30, 0.25, 0.20, 0.15)])
PIPE_BRIDGE_LENGTH = 0.8
TRENCH_COVER_DEPTH = 0.20
ELEM_KIND = {"sill": "sill", "hose": "hose", "floor_pipe": "floor_pipe", "floor_pipe_20deg": "floor_pipe",
             "kerb": "bund", "cable_protector": "cable_protector", "low_overhead": "low_overhead", "ramp": "ramp",
             "narrow_gap": "narrow_gap", "puddle": "puddle", "trench_cover": "trench_cover",
             "pipe_bridge": "low_overhead"}


# ============================================================================================ classes
def rated_class(cfg, kind, size) -> str:
    """The class a size has against the rated bands of capability.json (the gait's measurements): free,
    the mode it needs (inside the band), barrier, or 'unrated' when it lies between a band and the next one."""
    c = cfg["capability"]
    hs_kinds = [str(k) for k in cfg["ladder"].get("high_step_kinds", [])] + \
        [str(k) for k in (cfg["ladder"].get("v2_params") or {}).get("extra_high_step_kinds", [])]
    k = ELEM_KIND.get(kind, kind)

    def inside(v, band):
        return float(band[0]) - 1e-9 <= v <= float(band[1]) + 1e-9
    if k in ("sill", "hose", "bund", "cable_protector", "trench_cover"):
        b = c["step"]
        if size <= float(b["free_max"]) + 1e-9:
            return "free"
        if k in hs_kinds and inside(size, b["needs_high_step"]):
            return "needs_high_step"
        return "barrier" if size >= float(b["barrier_min"]) - 1e-9 else "unrated"
    if k == "floor_pipe":
        b = c["floor_pipe"]
        if size <= float(b["free_max"]) + 1e-9:
            return "free"
        if k in hs_kinds and inside(size, b["needs_high_step"]):
            return "needs_high_step"
        return "barrier" if size >= float(b["barrier_min"]) - 1e-9 else "unrated"
    if k == "low_overhead":
        b = c["overhead"]
        if size >= float(b["free_min"]) - 1e-9:
            return "free"
        if inside(size, b["needs_crouch"]):
            return "needs_crouch"
        return "barrier" if size <= float(b["barrier_max"]) + 1e-9 else "unrated"
    if k == "narrow_gap":
        w = c["width"]
        if size >= float(w["free_min"]) - 1e-9:
            return "free"
        if inside(size, w["tight"]):
            return "tight"
        return "barrier" if size <= float(w["barrier_max"]) + 1e-9 else "unrated"
    if k == "ramp":
        sl = c["slope_deg"]
        if size <= float(sl["free_max"]) + 1e-9:
            return "free"
        if size <= float(sl["slow"][1]) + 1e-9:
            return "slow"
        return "barrier" if size >= float(sl["barrier_min"]) - 1e-9 else "unrated"
    if k == "puddle":
        return "slow"
    return "free"


# ============================================================================================ layout
def hall_layout(cfg, rng) -> Layout:
    rooms = {"A1": Room(id="A1", type="airlock", rect=list(AIRLOCK_RECT), band="airlock", number="100",
                        name="AIRLOCK", light="bright"),
             "H1": Room(id="H1", type="hall", rect=list(HALL_RECT), band="hall", number="200",
                        name="TRAINING HALL", light="bright", front="S")}
    W = HALL_RECT[2] - HALL_RECT[0]
    H = HALL_RECT[3] - HALL_RECT[1]
    lay = Layout(rooms=rooms, walls=[], doors=[], W=W, H=H, offset=(HALL_RECT[0], HALL_RECT[1]), mirrored=False,
                 params={"hall": True, "chase_clearance": None, "ladder": False}, building_rect=list(HALL_RECT),
                 grid={})
    _build_walls(lay, cfg)
    g = rng("doors")
    w_in = wall_between(lay, "A1", "H1")[0]
    add_door(lay, w_in, 0.0, float(cfg["airlock"]["inner_door_width"]), into="A1", g=g, label="AIRLOCK INNER DOOR")
    rooms["A1"].front = _side_of(rooms["A1"], w_in)
    w_out = [w for w in lay.walls if "A1" in w.rooms and "OUT" in w.rooms and w.axis == "x"
             and abs(w.coord - AIRLOCK_RECT[0]) < 1e-6][0]
    d = add_door(lay, w_out, 0.5 * (w_out.lo + w_out.hi), float(cfg["airlock"]["outer_door_width"]),
                 kind="airlock_outer", into="A1", g=g, label="AIRLOCK OUTER DOOR")
    d.state = "closed"
    _grid(lay, cfg)
    _inner_rects(lay)
    # the inner leaf opens away from the base station's aerial
    F.hinge_inner_door_away(lay)
    return lay


# ============================================================================================ furnishing
def _lane_geom(i_row, row):
    """(x lo, x hi) of the lane between its dividers, the lane's y range, the obstacle centre y, the direction
    into the lane (-1: toward -y)."""
    x0 = LANE_X0 + LANE_W * i_row
    xa, xb = x0 + DIVIDER_T / 2, x0 + LANE_W - DIVIDER_T / 2
    if row == "S":
        y_mouth, y_end = ROW_S[1], ROW_S[0] + 0.05
        return xa, xb, (y_end, y_mouth), y_mouth - OBST_AT, -1
    y_mouth, y_end = ROW_N_Y0, HALL_RECT[3] - 0.125
    return xa, xb, (y_mouth, y_end), y_mouth + OBST_AT, +1


def _build_lane(W, c, k, kind, size, cfg):
    """One lane: its obstacle (prims, element) and its record for the lane table."""
    P = W.P
    row = "S" if k < LANES_PER_ROW else "N"
    i = k % LANES_PER_ROW
    xa, xb, (ylo, yhi), yc, sgn = _lane_geom(i, row)
    lid = "L%02d" % (k + 1)
    body = "elem/%s/%s" % ("H1", lid)
    xc = 0.5 * (xa + xb)
    cls = rated_class(cfg, kind, size)
    rec = {"lane": lid, "row": row, "kind": kind, "size": size, "unit": {"ramp": "deg", "puddle": "friction"}.get(
        kind, "m"), "class": cls, "x": [round(xa, 3), round(xb, 3)], "y": [round(ylo, 3), round(yhi, 3)],
        "obstacle_y": round(yc, 3), "mouth_y": round(yhi if row == "S" else ylo, 3),
        "into": "-y" if sgn < 0 else "+y"}
    ek = ELEM_KIND.get(kind)
    prims, fp, param, zone = [], None, {}, False
    if kind == "sill":
        p = P.box_aabb("%s/sill" % body, (xa, yc - 0.06, 0.0), (xb, yc + 0.06, size), mat="hazard_stripe", body=body,
                       owner="H1", friction=0.8)
        prims, fp, param = [p], poly_rect([xa, yc - 0.06, xb, yc + 0.06]), {"height": size, "depth": 0.12}
    elif kind == "hose":
        p = P.cyl("%s/hose" % body, (xa, yc, size / 2), (xb, yc, size / 2), size / 2, kind="capsule", mat="hose_red",
                  body=body, owner="H1", friction=0.9)
        wd = size + 0.04
        prims, fp, param = [p], poly_rect([xa, yc - wd / 2, xb, yc + wd / 2]), {"height": size}
    elif kind in ("floor_pipe", "floor_pipe_20deg"):
        ang = math.radians(20.0) if kind == "floor_pipe_20deg" else 0.0
        dy = 0.5 * (xb - xa) * math.tan(ang)
        p = P.cyl("%s/pipe" % body, (xa, yc - dy, size / 2), (xb, yc + dy, size / 2), size / 2, mat="steel",
                  body=body, owner="H1", friction=0.8)
        wd = size + 0.04
        prims = [p]
        fp = [[xa, yc - dy - wd / 2], [xb, yc + dy - wd / 2], [xb, yc + dy + wd / 2], [xa, yc - dy + wd / 2]]
        param = {"diameter": size, "angle_deg": round(math.degrees(ang), 1)}
        rec["angle_deg"] = round(math.degrees(ang), 1)
    elif kind == "kerb":
        ya, yb = (ylo, yc) if sgn < 0 else (yc, yhi)
        P.box_aabb("%s/platform" % body, (xa, ya, 0.0), (xb, yb, size), mat="concrete_light", body=body, owner="H1",
                   role="walkable", friction=0.9)
        ye = yc
        strip = poly_rect([xa, ye - 0.08, xb, ye + 0.08])
        P.box_aabb("%s/edge" % body, (xa, ye - 0.04, size - 0.002), (xb, ye + 0.04, size + 0.001), mat="hazard_stripe",
                   body=body, owner="H1", group=1)
        fp, param, zone = strip, {"height": size, "platform": [round(xa, 3), round(ya, 3), round(xb, 3), round(yb, 3)]}, True
        W.equip("platform", "%s PLATFORM %.3f m" % (lid, size), "H1", [xa, ya, xb, yb], size)
    elif kind == "cable_protector":
        prims = F.ramped_protector(P, body, owner="H1", axis="x", c=yc, lo=xa, hi=xb, h=size)
        wd = F.protector_width(size)
        fp, param = poly_rect([xa, yc - wd / 2, xb, yc + wd / 2]), {"height": size}
    elif kind == "low_overhead":
        p = P.box_aabb("%s/rack" % body, (xa, yc - 0.5, size), (xb, yc + 0.5, size + 0.05), mat="steel_dark", body=body,
                       owner="H1")
        for j, yy in enumerate((yc - 0.5, yc + 0.5)):
            P.box_aabb("%s/beam%d" % (body, j), (xa, yy - 0.04, size), (xb, yy + 0.04, size + 0.12), mat="hazard_stripe",
                       body=body, owner="H1", group=1)
        prims, fp, param = [p], poly_rect([xa, yc - 0.5, xb, yc + 0.5]), {"clearance": size, "length": 1.0}
        W.equip("rack", "%s LOW RACK %.3f m CLEAR" % (lid, size), "H1", [xa, yc - 0.5, xb, yc + 0.5], size + 0.05)
    elif kind == "trench_cover":
        dd = TRENCH_COVER_DEPTH
        p = P.box_aabb("%s/cover" % body, (xa, yc - dd / 2, 0.0), (xb, yc + dd / 2, size), mat="steel_plate", body=body,
                       owner="H1", friction=0.8)
        prims, fp, param = [p], poly_rect([xa, yc - dd / 2, xb, yc + dd / 2]), {"height": size, "depth": dd}
    elif kind == "pipe_bridge":
        prims, fp, param = _pipe_bridge(P, body, xa, xb, yc, size, PIPE_BRIDGE_LENGTH)
        rec["length_m"] = PIPE_BRIDGE_LENGTH
        W.equip("rack", "%s PIPE BRIDGE %.3f m CLEAR, %.1f m LONG" % (lid, size, PIPE_BRIDGE_LENGTH), "H1",
                [xa, yc - PIPE_BRIDGE_LENGTH / 2, xb, yc + PIPE_BRIDGE_LENGTH / 2], 1.7)
    elif kind == "ramp":
        th = math.radians(size)
        Lr = PLATFORM_H / math.tan(th)
        foot = yc + sgn * (-Lr / 2)                           # the ramp centred on the obstacle position
        top = foot + sgn * Lr
        T = PLATFORM_H + 0.06
        L = Lr / math.cos(th) + 0.02
        nv, nz = -sgn * math.sin(th), math.cos(th)            # outward-up normal of the ramp face in (y, z)
        cy, cz = 0.5 * (foot + top) - nv * T / 2, PLATFORM_H / 2 - nz * T / 2
        q = quat_axis_angle((1.0, 0.0, 0.0), sgn * th)
        p = P.box("%s/ramp" % body, (xc, cy, cz), ((xb - xa) / 2, L / 2, T / 2), quat=q, mat="ramp_steel", body=body,
                  owner="H1", role="walkable", friction=0.9)
        ya, yb = (ylo, top) if sgn < 0 else (top, yhi)
        P.box_aabb("%s/platform" % body, (xa, ya, 0.0), (xb, yb, PLATFORM_H), mat="concrete_light", body=body,
                   owner="H1", role="walkable", friction=0.9)
        prims = [p]
        fp = poly_rect([xa, min(foot, top), xb, max(foot, top)])
        param = {"slope_deg": size, "height": PLATFORM_H, "length": round(Lr, 3)}
        rec["ramp_y"] = [round(min(foot, top), 3), round(max(foot, top), 3)]
        W.equip("platform", "%s RAMP %.0f DEG + PLATFORM %.2f m" % (lid, size, PLATFORM_H), "H1",
                [xa, min(ylo, ya), xb, max(yhi, yb)], PLATFORM_H)
    elif kind == "narrow_gap":
        for j, (a, b) in enumerate(((xa, xc - size / 2), (xc + size / 2, xb))):
            P.box_aabb("%s/panel%d" % (body, j), (a, yc - 0.02, 0.0), (b, yc + 0.02, DIVIDER_H), mat="steel_plate",
                       body=body, owner="H1")
            W.equip("panel", "%s GAP PANEL" % lid, "H1", [a, yc - 0.02, b, yc + 0.02], DIVIDER_H)
        fp, param, zone = poly_rect([xc - size / 2, yc - 0.15, xc + size / 2, yc + 0.15]), {"width": size}, True
    elif kind == "puddle":
        u0, v0 = c.f.to_local(xa + 0.05, yc - 0.4)
        u1, v1 = c.f.to_local(xb - 0.05, yc + 0.4)
        prims = F.puddle(c, "%s_puddle" % lid, min(u0, u1), min(v0, v1), max(u0, u1), max(v0, v1), size)
        fp, param = poly_rect([xa + 0.05, yc - 0.4, xb - 0.05, yc + 0.4]), {"friction": size}
    # the lane's number on the floor at its mouth (a painted plate, readable from the aisle)
    ym = yhi - 0.25 if row == "S" else ylo + 0.25
    mat = W.sign("lane_%s" % lid, lid, "room", (256, 96))
    P.box("%s/plate" % body, (xc, ym, 0.0015), (0.20, 0.075, 0.0012), yaw=(math.pi if row == "S" else 0.0),
          mat=mat, body=body, owner="H1", group=1)
    if ek is not None and fp is not None:
        e = W.element(ek, cls, param, "H1", fp, zone=zone, note="training lane %s (%s %s)" % (lid, kind, size))
        if prims:
            W.tag(e, prims)
        rec["element"] = e.id
    return rec


def _pipe_bridge(P, body, xa, xb, yc, clr, L, top=1.7):
    """A pipe bridge across a lane (x from xa to xb), centred on y = yc, L long along the lane: cross beams at both
    ends (hazard striped) and two supports, their underside at the clearance `clr`; pipes across the lane in staggered
    layers up to `top` under a cable tray: the construction of the plant sites' racks (site_gen.variants.build_rack).
    Returns (element prims, footprint, param)."""
    from .furnish import PIPE_MAT
    bt = 0.05
    y0, y1 = yc - L / 2, yc + L / 2
    prims = [P.box_aabb("%s/beam0" % body, (xa, y0, clr), (xb, y0 + 0.06, clr + bt), mat="hazard_stripe", body=body,
                        owner="H1"),
             P.box_aabb("%s/beam1" % body, (xa, y1 - 0.06, clr), (xb, y1, clr + bt), mat="hazard_stripe", body=body,
                        owner="H1")]
    for k, xx in enumerate((xa + 0.10, xb - 0.10)):
        prims.append(P.box_aabb("%s/support%d" % (body, k), (xx - 0.025, y0 + 0.06, clr), (xx + 0.025, y1 - 0.06,
                                clr + bt), mat="galv", body=body, owner="H1"))
    r = 0.085
    pitch = 2 * r + 0.04
    z_bot = clr + bt
    layer = 0
    while z_bot + 2 * r <= top - 0.06 + 1e-9:
        zc = z_bot + r
        n = max(2, int((L - 0.02) // pitch))
        v_first = y0 + 0.5 * (L - (n - 1) * pitch)
        pos = [v_first + i * pitch for i in range(n)] if layer % 2 == 0 else \
            [v_first + (i + 0.5) * pitch for i in range(-1, n)]
        for i, yy in enumerate([v for v in pos if v - r >= y0 + 0.005 and v + r <= y1 - 0.005]):
            sv = ("cw", "rw", "st", "dw", "ia")[(i + layer) % 5]
            ps = P.pipe("%s/l%d_%d" % (body, layer, i), [(xa - 0.02, yy, zc), (xb + 0.02, yy, zc)], r, PIPE_MAT[sv],
                        body, owner="H1", elbows=False)
            prims += [p for p in ps if p.zrange()[0] < 0.5]
        z_bot = zc + 0.6 * r
        layer += 1
    P.box_aabb("%s/tray" % body, (xa, y0, top - 0.06), (xb, y1, top), mat="tray", body=body, owner="H1")
    return prims, poly_rect([xa, y0, xb, y1]), {"clearance": clr, "length": L, "rack": True, "layers": layer,
                                                "top": top}


# version 2: the plate board at the west end of the 28 m odometry straight, on the west wall facing the
# straight (walking it gives every distance from 1 to 12 m head-on): 16 text plates alone in four rows (0.4, 0.8,
# 1.2, 1.6 m), every character the plant's tags hold at least twice (the letters of the tag prefixes PI TI LG HV V
# LCP TS, the ten digits, the hyphen), three pairs one digit apart, four plates on a dark steel backing; beside it two
# posts with a plate turned 45 and 60 degrees
BOARD_TAGS = (("PI-1047", "PI-1048", "TI-2385", "LG-3906"),
              ("LG-3956", "HV-4172", "V-5629", "V-5620"),
              ("LCP-6813", "TS-7254", "HV-8031", "TI-9640"),
              ("LCP-2597", "TS-0368", "PI-7412", "LG-1859"))
BOARD_ROWS_Z = (1.6, 1.2, 0.8, 0.4)
BOARD_Y = (-6.54, -6.18, -5.82, -5.46)          # four columns across the straight's line (y = -6.0), 0.36 m apart:
                                                # the plate finder asks for >= 0.31 m between three plates in a row
                                                # (its wide finder takes a closer row of yellow regions for a hazard
                                                # stripe); 0.30 m before 03:25
BOARD_DARK = {(0, 3), (1, 1), (2, 0), (3, 2)}   # (row, column) on the dark steel backing
POSTS = (("TS-5530", 45.0, (2.85, -7.35, 1.0)), ("HV-9061", 60.0, (2.85, -4.65, 1.0)))


def _plate_board(W, ipw):
    """Build the board's backings and posts and register its plates with the instrument works (version 2). Returns
    the board's records (the calibration sheet: text, centre, height, facing, backing)."""
    P = W.P
    x_face = HALL_RECT[0] + 0.125                  # the west wall's inner face
    out = []
    body = "eq/H1/plateboard"
    P.box_aabb("%s/frame" % body, (x_face, BOARD_Y[0] - 0.17, 0.25), (x_face + 0.01, BOARD_Y[-1] + 0.17, 1.75),
               mat="panel_grey", body=body, owner="H1", group=1)
    for i, row in enumerate(BOARD_TAGS):
        for j, text in enumerate(row):
            z = BOARD_ROWS_Z[i]
            y = BOARD_Y[j]
            dark = (i, j) in BOARD_DARK
            x = x_face + 0.01
            if dark:
                P.box_aabb("%s/backing%d%d" % (body, i, j), (x_face + 0.01, y - 0.125, z - 0.065),
                           (x_face + 0.016, y + 0.125, z + 0.065), mat="steel_dark", body=body, owner="H1", group=1)
                x = x_face + 0.016
            name = "plate/board/%s" % text
            ipw.add_plate(name, text, (x + 0.002, y, z), 0.0)
            out.append({"text": text, "pos": [round(x + 0.002, 4), y, z], "height_m": z, "facing_deg": 0.0,
                        "backing": "dark steel" if dark else "board (grey)", "where": "plate board, row %d column %d" %
                        (i + 1, j + 1), "body": name})
    for k, (text, ang, (x, y, z)) in enumerate(POSTS):
        pb = "eq/H1/platepost%d" % k
        P.box_aabb("%s/post" % pb, (x - 0.03, y - 0.03, 0.0), (x + 0.03, y + 0.03, z - 0.05), mat="galv", body=pb,
                   owner="H1")
        P.box_aabb("%s/base" % pb, (x - 0.15, y - 0.15, 0.0), (x + 0.15, y + 0.15, 0.012), mat="steel_dark", body=pb,
                   owner="H1")
        yaw = math.radians(ang if y < -6.0 else -ang)      # turned toward the straight from either side
        name = "plate/post/%s" % text
        ipw.add_plate(name, text, (x + 0.035 * math.cos(yaw), y + 0.035 * math.sin(yaw), z), yaw)
        out.append({"text": text, "pos": [round(x + 0.035 * math.cos(yaw), 4), round(y + 0.035 * math.sin(yaw), 4), z],
                    "height_m": z, "facing_deg": round(math.degrees(yaw), 1), "backing": "post",
                    "where": "post %d, turned %d deg from the straight" % (k + 1, int(ang)), "body": name})
        W.equip("post", "PLATE POST %d" % (k + 1), "H1", [x - 0.15, y - 0.15, x + 0.15, y + 0.15], z)
    W.equip("plate_board", "PLATE BOARD", "H1", [x_face, BOARD_Y[0] - 0.17, x_face + 0.02, BOARD_Y[-1] + 0.17], 1.75)
    return out


def _room_plates(W):
    """The room plates the structure hung (structure._room_sign: `sign/<room id>`, 0.40 x 0.10 m, beside the room's
    main door on the side toward the neighbouring room, 1.6 m up), as calibration-sheet records."""
    out = []
    names = {p.name: p for p in W.P.prims}
    for rid in sorted(W.lay.rooms):
        r = W.lay.rooms[rid]
        p = names.get("sign/%s" % rid)
        if p is None:
            continue
        R = p.R()
        n = R[:, 0]
        c = np.asarray(p.pos, float) + float(p.size[0]) * n          # the centre of the plate's front face
        d = W.lay.door(r.doors[0]) if r.doors else None
        other = (d.rooms[0] if d.rooms[1] == rid else d.rooms[1]) if d is not None else None
        on = W.lay.rooms[other].number if other in W.lay.rooms else other
        out.append({"number": r.number, "name": r.name, "pos": [round(float(v), 3) for v in c[:2]],
                    "height_m": round(float(c[2]), 3), "facing_deg": round(math.degrees(math.atan2(n[1], n[0])), 1),
                    "size_m": [0.40, 0.10],
                    "where": "beside the door of room %s, on the side of %s" % (r.number, on)})
    return out


def _dividers_and_walls(W):
    P = W.P
    for row in ("S", "N"):
        ylo, yhi = (ROW_S[0], ROW_S[1]) if row == "S" else (ROW_N_Y0, HALL_RECT[3] - 0.125)
        for i in range(LANES_PER_ROW + 1):
            x = LANE_X0 + LANE_W * i
            P.box_aabb("eq/H1/div%s%02d" % (row, i), (x - DIVIDER_T / 2, ylo, 0.0), (x + DIVIDER_T / 2, yhi, DIVIDER_H),
                       mat="cabinet_white", body="eq/H1/lanes%s" % row, owner="H1")
            W.equip("divider", "LANE DIVIDER", "H1", [x - DIVIDER_T / 2, ylo, x + DIVIDER_T / 2, yhi], DIVIDER_H)
    x1 = LANE_X0 + LANE_W * LANES_PER_ROW
    P.box_aabb("eq/H1/backwallS", (LANE_X0 - DIVIDER_T / 2, ROW_S[0] - 0.05, 0.0), (x1 + DIVIDER_T / 2, ROW_S[0] + 0.05,
               1.2), mat="blockwork", body="eq/H1/lanesS", owner="H1")
    W.equip("partition", "LANE BACK WALL (1.2 m)", "H1", [LANE_X0, ROW_S[0] - 0.05, x1, ROW_S[0] + 0.05], 1.2)


def _island(W):
    P = W.P
    x0, y0, x1, y1 = ISLAND
    body = "eq/H1/island"
    n = 4
    for j in range(n):
        a = x0 + j * (x1 - x0) / n
        P.box_aabb("%s/crate%d" % (body, j), (a + 0.02, y0, 0.0), (a + (x1 - x0) / n - 0.02, y1, 1.0), mat="crate_wood",
                   body=body, owner="H1")
    W.equip("pallets", "PALLET ISLAND (LOOP)", "H1", ISLAND, 1.0)


def _cave(W, cfg):
    P = W.P
    body = "eq/H1/cave"
    x0, x1 = CAVE["x0"], HALL_RECT[2] - 0.125
    for nm, (ya, yb) in (("s", CAVE["y_lo"]), ("n", CAVE["y_hi"])):
        p = P.box_aabb("%s/wall_%s" % (body, nm), (x0, ya, 0.0), (x1, yb, CAVE["h"]), mat="shield_concrete", body=body,
                       owner="H1")
        W.shields.append({"id": "SH-cave-%s" % nm, "prim": p.name, "room": "H1", "kind": "shield blocks"})
        W.equip("shield", "SHIELD BLOCKS", "H1", [x0, ya, x1, yb], CAVE["h"])
    sx, sy, sz = SOURCE_XYZ
    P.cyl("src/check/pot", (sx, sy, 0.0), (sx, sy, 0.38), 0.12, mat="drum_yellow", body="src/check", owner="H1")
    P.box("src/check/plate", (sx - 0.125, sy, 0.25), (0.004, 0.08, 0.05), mat=W.sign("check_src", "CHECK SOURCE", "room"),
          body="src/check", owner="H1", group=1)
    W.equip("check_source", "CHECK SOURCE (25 Gy/h AT 1 m)", "H1", [sx - 0.15, sy - 0.15, sx + 0.15, sy + 0.15], 0.38)


def _floor_lines(W):
    P = W.P
    (xa, ya), (xb, yb) = STRAIGHT
    P.box_aabb("eq/H1/straight/line", (xa, ya - 0.03, 0.0), (xb, ya + 0.03, 0.0025), mat="zone_tape", body="eq/H1/straight",
               owner="H1", group=1)
    for k in range(int(round(xb - xa)) + 1):
        x = xa + k
        P.box_aabb("eq/H1/straight/tick%02d" % k, (x - 0.015, ya - 0.15, 0.0), (x + 0.015, ya + 0.15, 0.0026),
                   mat="zone_tape", body="eq/H1/straight", owner="H1", group=1)
    x0, y0, x1, y1 = ISLAND
    m = 1.0
    for j, (a, b) in enumerate((((x0 - m, y0 - m), (x1 + m, y0 - m)), ((x1 + m, y0 - m), (x1 + m, y1 + m)),
                                ((x1 + m, y1 + m), (x0 - m, y1 + m)), ((x0 - m, y1 + m), (x0 - m, y0 - m)))):
        lo = (min(a[0], b[0]) - 0.03, min(a[1], b[1]) - 0.03, 0.0)
        hi = (max(a[0], b[0]) + 0.03, max(a[1], b[1]) + 0.03, 0.0025)
        P.box_aabb("eq/H1/loop/line%d" % j, lo, hi, mat="zone_tape", body="eq/H1/loop", owner="H1", group=1)


def _markers():
    out = []
    y_s = HALL_RECT[1] + 0.125 + 0.004
    for k in range(1, 8):
        out.append(([BAY_X0 + BAY_W * k, y_s, 0.25], math.pi / 2, "S%d" % k, "wall"))
    x_w = HALL_RECT[0] + 0.125 + 0.004
    for y in (-8.0, -4.0, 3.0):
        out.append(([x_w, y, 0.6], 0.0, "W%+d" % int(y), "wall"))
    x_e = HALL_RECT[2] - 0.125 - 0.004
    for y in (-8.0, -3.5):
        out.append(([x_e, y, 0.6], math.pi, "E%+d" % int(y), "wall"))
    x0, y0, x1, y1 = ISLAND
    out.append(([14.0, y1 + 0.004, 0.6], math.pi / 2, "IN", "island"))
    out.append(([14.0, y0 - 0.004, 0.6], -math.pi / 2, "IS", "island"))
    out.append(([x0 - 0.004, 0.5 * (y0 + y1), 0.6], math.pi, "IW", "island"))
    recs = []
    for k, (pos, yaw, grid, on) in enumerate(out):
        recs.append({"pos": [round(v, 4) for v in pos], "yaw": yaw, "grid": grid, "on": on, "room": "H1",
                     "id": "GM-%02d" % (k + 1), "marker_id": 300 + k, "documented": True, "present": True})
    return recs


def _repeater(W):
    P = W.P
    x, y, z = REPEATER_XYZ
    body = "link/repeater1"
    P.box("%s/box" % body, (x, y, z), (0.06, 0.12, 0.09), mat="cabinet", body=body, owner="link", group=1)
    P.cyl("%s/ant" % body, (x - 0.02, y + 0.08, z + 0.12), (x - 0.02, y + 0.08, z + 0.32), 0.01, mat="steel_dark",
          body=body, owner="link", group=1)
    P.box("%s/led" % body, (x - 0.065, y - 0.05, z + 0.05), (0.004, 0.012, 0.012), mat="led_green", body=body,
          owner="link", group=1)
    W.repeaters = [{"id": "RP1", "pos": [x, y, z], "room": "H1", "working": True, "label": "LINK REPEATER 1"}]


# ============================================================================================ instruments
def _truth(typ, k, cfg):
    v = VALUES[typ][k]
    if typ.startswith("gauge"):
        lo, hi, unit, val = v
        return {"lo": lo, "hi": hi, "unit": unit, "case_mm": int(typ.split("_")[1])}, {"value": float(val), "unit": unit}
    if typ == "level":
        return {"window_m": float(cfg["ladder"]["level_window_m"])}, {"level_pct": float(v)}
    if typ == "valve_lever":
        return {}, {"angle_deg": float(v), "state": "open" if v == 0 else ("closed" if v == 90 else "throttled")}
    if typ == "valve_handwheel":
        return {}, {"opening_pct": float(v)}
    if typ == "lamps":
        cols, st = v
        return {"colors": list(cols)}, {"states": list(st)}
    if typ == "thermal":
        sfc, t = v
        T = cfg["ladder"]["thermal"]
        return {"surface": sfc, "range": list(T["range"]), "unit": "°C"}, \
            {"temp_C": float(t), "surface": sfc, "emissivity": float(T["surfaces"][sfc])}
    raise KeyError(typ)


def _instruments(W, c, cfg, g):
    """The instrument wall: eight bays on the south wall (local 'front' wall of the hall), three heights each."""
    from .ips import tag_prefix
    from .ladder import BASE_TYPE
    items = []
    x_in0 = W.lay.rooms["H1"].inner[0]
    spot = W.spot(c, "front", 0.4, c.Lu - 0.4, "room", orients=("out",))
    mid = 0
    for b, typ in enumerate(WALL_TYPES):
        u_c = BAY_X0 + BAY_W * (b + 0.5) - x_in0
        for k, z in enumerate(HEIGHTS):
            s = u_c + (k - 1) * 1.0
            base = BASE_TYPE[typ]
            unit = VALUES[typ][k][2] if typ.startswith("gauge") else None
            tag = "%s-2%d%d" % (tag_prefix(base, unit) if base == "gauge" else PREFIX[base], b + 1, k + 1)
            svc = {"code": "cw", "text": "calibration stand", "mount": "instrument wall bay %d" % (b + 1)}
            slot, prm2 = ST.build(W, spot, typ, s, z, "out", tag, svc, g)
            prm, tr = _truth(typ, k, cfg)
            prm.update(prm2)
            if typ == "level":
                prm["span"] = list(slot["span"])
            mid += 1
            it = {"tag": tag, "type": base, "ladder_type": typ, "slot": slot, "role": "ip", "params": prm,
                  "truth": tr, "room": "H1", "reason": "training hall instrument wall", "body": "ip/%s" % tag,
                  "ladder_slot": "HALL", "last_2009": {},
                  "hall": {"bay": b + 1, "height_m": z, "position": k + 1}}
            if not (cfg.get("ladder") or {}).get("v2"):
                prm["marker_id"] = mid
                it["label_code"] = "36h11-%03d" % mid
            if typ == "thermal":
                prm["n_units"] = 1
                u = slot["units"][0]
                it["truth"]["units"] = [dict(u, temp_C=tr["temp_C"])]
                it["truth"]["hot_label"] = u["label"]
                it["truth"]["geom"] = prm2.get("hot_geom")
                W.thermal_groups.append({"room": "H1", "kind": "hall_hot_component", "units": [dict(u, temp_C=tr["temp_C"])],
                                         "tag": tag})
            items.append(it)
    # bay 9: the reading conditions the eight bays lack and the buildings have
    b = len(WALL_TYPES)
    u_c = BAY_X0 + BAY_W * (b + 0.5) - x_in0
    for k, (typ, z, orient, note) in enumerate(EXTRA_BAY):
        s = u_c + (k - 0.5) * 1.6
        base = BASE_TYPE[typ]
        tag = "%s-2%d%d" % (PREFIX[base], b + 1, k + 1)
        svc = {"code": "cw", "text": "calibration stand", "mount": "instrument wall bay %d" % (b + 1)}
        slot, prm2 = ST.build(W, spot, typ, s, z, orient, tag, svc, g)
        prm, tr = _truth_extra(typ)
        prm.update(prm2)
        if typ == "lamps":
            # a canopy over the panel (a cable tray 0.25 m above it, 0.75 m deep): the panel hangs high in its shade
            c.box("canopy_%s" % tag, s - 0.6, 0.0, z + 0.22, s + 0.6, 0.75, z + 0.26, "galv",
                  body="station/%s" % tag)
        mid += 1
        items.append({"tag": tag, "type": base, "ladder_type": typ, "slot": slot, "role": "ip", "params": prm,
                      "truth": tr, "room": "H1", "reason": "training hall instrument wall", "body": "ip/%s" % tag,
                      "ladder_slot": "HALL", "last_2009": {},
                      "hall": {"bay": b + 1, "height_m": z, "position": k + 1, "note": note}})
        if not (cfg.get("ladder") or {}).get("v2"):
            prm["marker_id"] = mid
            items[-1]["label_code"] = "36h11-%03d" % mid
    return items


# bay 9: (type, mounting height, station orientation, what it shows)
EXTRA_BAY = (("valve_lever", 1.0, "vpipe", "quarter-turn valve on a vertical pipe (0 deg: the lever along the pipe)"),
             ("lamps", 1.9, "out", "lamp panel mounted high under a canopy, read from below in its shade"))
VALUE_EXTRA = {"valve_lever": 30.0, "lamps": (["green", "amber", "red", "white"], [True, False, True, True])}


def _truth_extra(typ):
    v = VALUE_EXTRA[typ]
    if typ == "valve_lever":
        return {}, {"angle_deg": float(v), "state": "open" if v == 0 else ("closed" if v == 90 else "throttled")}
    cols, st = v
    return {"colors": list(cols)}, {"states": list(st)}


# ============================================================================================ the hall
def build_hall(out_dir, cfg=None, seed=0, maps=False):
    """Build the training hall into out_dir; returns (site, prior, docs)."""
    # version 2: text plates, room plates and the survey mark in the plant lettering, the plate board
    cfg = cfg or load_config(overrides={"layout": "ladder_hall", "ladder": {"v2": {"k": 0, "label": "hall",
                                                                                   "hall": True}}})
    v2 = bool((cfg.get("ladder") or {}).get("v2"))
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    rng = Rng(seed, 0)
    lay = hall_layout(cfg, rng)
    W = World(cfg, lay, rng)
    build_structure(W)
    ca = F.Ctx(W, lay.rooms["A1"])
    W.ctx["A1"] = ca
    F.f_airlock(ca)
    c = F.Ctx(W, lay.rooms["H1"])
    W.ctx["H1"] = c
    c.lights_grid(8, 5, "bright")
    for Lg in W.lights:
        Lg["level"] = "bright"
    _dividers_and_walls(W)
    lanes = [_build_lane(W, c, k, kind, size, cfg) for k, (kind, size) in enumerate(LANES)]
    _island(W)
    _cave(W, cfg)
    _floor_lines(W)
    _repeater(W)
    g = rng("hall_instruments")
    items = _instruments(W, c, cfg, g)
    # the survey mark and the check source
    D = cfg["ladder"]["dose"]
    f = float(D["unit_factor"])
    src = {"id": "SRCK", "kind": "check_source", "pos": list(SOURCE_XYZ), "room": "H1",
           "strength": round(SOURCE_GYH_AT_1M / f, 3), "documented": True, "text": "check source"}
    sources = [src]
    msl = W.slot("dose_point", (MARK_XY[0], MARK_XY[1], 0.0), 0.0, "H1", "survey mark on the floor", "survey point",
                 "mark")
    items.append({"tag": "SM-01", "type": "dose_point", "ladder_type": "survey_mark", "slot": msl, "role": "ip",
                  "params": {"label": "SM-01"}, "truth": {}, "room": "H1", "reason": "training hall survey mark",
                  "body": "ip/SM-01", "ladder_slot": "M", "last_2009": {}})
    xsl = W.slot("hotspot", [SOURCE_XYZ[0], SOURCE_XYZ[1], 0.0], 0.0, "H1", "check source", "radiation survey", "SRCK")
    items.append({"tag": "X", "type": "hotspot", "ladder_type": "check_source", "slot": xsl, "role": "ip", "params": {},
                  "truth": {"source": "SRCK", "pos": list(SOURCE_XYZ[:2])}, "room": "H1",
                  "reason": "training hall check source", "body": "src/X", "ladder_slot": "X", "last_2009": {}})
    markers = _markers()
    ipw = IPWorks(cfg)
    for it in items:
        if it["type"] == "hotspot":
            continue
        ipw.add(it["slot"], it["role"], it["tag"], it["truth"], it["params"])
    for m in markers:
        ipw.add_marker(m)
    board = _plate_board(W, ipw) if v2 else []
    if os.path.isdir(os.path.join(out_dir, "assets")):
        shutil.rmtree(os.path.join(out_dir, "assets"))
    cams = [{"name": "cam/eye_pad", "pos": [0.0, 0.0, 0.33], "target": [4.0, -0.25, 0.43], "what": "the pad"},
            {"name": "cam/eye_lanes", "pos": [4.0, 6.1, 0.33], "target": [12.0, 6.1, 0.4], "what": "the lane aisle"},
            {"name": "cam/eye_wall", "pos": [8.0, -6.5, 0.33], "target": [8.0, -10.3, 1.0], "what": "the instrument wall"},
            {"name": "cam/eye_cave", "pos": [25.5, -0.6, 0.33], "target": [31.0, -0.6, 0.3], "what": "the check source"}]
    spec, info = build_scene(W, ipw, out_dir, cams)
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    for it in items:
        mi = info["ip_models"].get(it["body"])
        if mi:
            it["backend"] = mi["backend"]
            it["model_names"] = {k: v for k, v in mi.items() if k in ("body", "geoms", "face_geoms", "joints", "backend",
                                                                       "bricks", "marker_id")}
    for m in markers:
        mi = info["ip_models"].get("marker/%s" % m["id"])
        if mi:
            m["model_names"] = {k: v for k, v in mi.items() if k in ("body", "geoms", "face_geoms", "backend",
                                                                       "dictionary")}
    from .loader.states import apply_truth_states
    apply_truth_states(model, data, [{"model_names": it.get("model_names", {})} for it in items])
    # the traversability map, link, dose
    bb = lay.bbox()
    grid = Grid(bb, float(cfg["grid_res"]))
    trav = TravMap(grid, cfg)
    trav.rasterize(W.P.prims, W.elements, [r.rect for r in lay.rooms.values()])
    write_xml(spec, out_dir, info["assets"])
    trav.save(os.path.join(out_dir, "trav.npz"))
    lg = Grid(bb, 0.1)
    opaque = opaque_raster(W, lg)
    inside_l = np.zeros(lg.shape, np.uint8)
    for r in lay.rooms.values():
        lg.fill_poly(inside_l, poly_rect(r.rect), 1)
    ants = [(W.base_station["antenna"][0], W.base_station["antenna"][1], float(cfg["link"]["base_range"]))]
    ants += [(r["pos"][0], r["pos"][1], float(cfg["link"]["repeater_range"])) for r in W.repeaters if r["working"]]
    cov = coverage(lg, opaque, ants, inside_l.astype(bool))
    cov_base = coverage(lg, opaque, ants[:1], inside_l.astype(bool))
    boxes = RAD.attenuators(W, cfg)
    bg = 0.10
    xs_, ys_, Dt = RAD.field_grid(bb, 0.1, sources, boxes, cfg, bg)
    np.savez_compressed(os.path.join(out_dir, "fields.npz"), dose_xs=xs_, dose_ys=ys_, dose_truth=Dt.astype(np.float32),
                        dose_documented=Dt.astype(np.float32), link_origin=np.array([lg.x0, lg.y0]), link_res=lg.res,
                        link=cov, link_base=cov_base, inside=inside_l.astype(bool))
    # zoning from the (documented) check source, by room and in 6 m stretches of the hall
    zoning = _zoning(lay, cfg, trav, sources, boxes, bg)
    # documents and the site record
    budgets = {"time_limit_s": 900.0, "dose_rating_gy": float(D["rating_gy"]),
               "frames": {"nav": 300, "inspect": 240, "depth": 1500, "thermal": 150, "lidar": 4000},
               "reports": {"instrument": 40, "mark": 6, "source": 3}}
    # the weights the cell gives these items (mission.scoring.item_weights: the 24 wall instruments share 0.8), so the
    # work order and the round sheet carry the same numbers (0.1 per instrument before)
    wts = _cell_weights(cfg, items)
    for it in items:
        it["weight"] = round(float(wts[it["tag"]]), 4)
    from .prior import build_prior
    prior = build_prior(W, cfg, rng, [dict(i) for i in items], sources, [], budgets, markers, zoning)
    prior = _hall_documents(prior, items, lay, cfg, markers)
    site = _site_json(W, cfg, seed, items, sources, boxes, bg, markers, grid, info, cov, lg, budgets, zoning, lanes)
    if board:
        site["hall"]["plates"] = board
    if v2:
        # the hall's room plates: number, name, centre of the front
        # face, height, facing, where, so that the calibration sheet lists them beside the plate board
        site["hall"]["room_plates"] = _room_plates(W)
    if v2:
        site["ladder"]["v2"] = dict(cfg["ladder"]["v2"])        # as on the generated sites (a development test checks it)
    # regions and observation poses (the debrief and a privileged run use them)
    from . import proofs as PR
    ctx = PR.Ctx(site, prior, trav, model, data, cfg, out_dir, fields={"link_base": cov_base, "link_origin":
                 [lg.x0, lg.y0], "link_res": lg.res})
    PR.compute_regions(ctx)
    from scipy.sparse.csgraph import dijkstra
    G = trav.build_graph(MODES)
    dist_pad = dijkstra(G, directed=True, indices=[ctx.pad])[0]
    for rec in site["ips"]:
        node, pose = PR._obs_pose(ctx, rec, dist_pad)
        rec["observation_pose"] = pose
    PR.store_regions(ctx)
    # the face centre of every wall instrument as the grader's evidence rules take it (mission/evidence.py face_frame
    # on the compiled hall): the sheet's face position (a rising-stem valve's is its indicator plate)
    face_c = {tag: [round(float(v), 3) for v in eg.c] for tag, eg in ctx.evgeoms().items()}
    # no build time in site.json: two builds of the hall give the same files (two earlier builds differed by 'timing_s' only)
    dump_json(site, os.path.join(out_dir, "site.json"))
    dump_json(prior, os.path.join(out_dir, "prior.json"))
    docs = hall_documents(site, prior, cfg, os.path.join(out_dir, "hall_docs"), face_centres=face_c)
    if maps:
        from .maps import make_maps
        make_maps(out_dir)
    return site, prior, docs


def _zoning(lay, cfg, trav, sources, boxes, bg):
    classes = cfg["radiation"]["zone_classes"]
    X, Y = trav.g.centers()
    acc = trav.enter["walk"] | trav.enter["high_step"] | trav.enter["crouch_walk"]
    out = []
    for r in lay.rooms.values():
        x0, y0, x1, y1 = r.inner
        n = max(1, int(round((x1 - x0) / 6.0)))
        for k in range(n):
            rect = [x0 + k * (x1 - x0) / n, y0, x0 + (k + 1) * (x1 - x0) / n, y1]
            m = acc & (X >= rect[0]) & (X <= rect[2]) & (Y >= rect[1]) & (Y <= rect[3])
            if not m.any():
                continue
            pts = np.stack([X[m], Y[m]], 1)                  # every accessible cell
            v = float(RAD.dose_rate(pts, sources, boxes, cfg, bg).max())
            out.append({"class": RAD.zone_class(v, classes), "polygon": poly_rect(rect), "label": r.number,
                        "center": [0.5 * (rect[0] + rect[2]), 0.5 * (rect[1] + rect[3])], "max_documented": v,
                        "room": r.id})
    return out


def _site_json(W, cfg, seed, items, sources, boxes, bg, markers, grid, info, cov, lg, budgets, zoning, lanes):
    from .config import acceptance_path, file_label
    from .generator import _site_json as base
    route0 = {"legs": [], "order": []}
    site = base(W, cfg, seed, 0, items, [], [], sources, boxes, bg, [], markers, route0, {"ok": True}, [], budgets, grid,
                info, file_label(acceptance_path(cfg)), cov, lg, [i["type"] for i in items], None)
    site["schema"] = "site_gen/site/0.3-hall"
    site["layout"] = "hall_v1"
    from .config import evidence_label
    site["evidence_source"] = evidence_label()
    for rec, it in zip(site["ips"], items):
        rec["slot"] = it["ladder_slot"]
        rec["ladder_type"] = it.get("ladder_type")
        if it.get("label_code"):
            rec["label_code"] = it.get("label_code")
        else:
            rec["plate"] = {"text": it["tag"], "style": "text"}
            (rec.get("model_names") or {}).pop("marker_id", None)
        rec["last_2009"] = None
        if it.get("hall"):
            rec["hall"] = it["hall"]
    site["zoning_truth"] = zoning
    site["hall"] = {"lanes": lanes, "straight": {"from": list(STRAIGHT[0]), "to": list(STRAIGHT[1]),
                                                 "length_m": STRAIGHT[1][0] - STRAIGHT[0][0]},
                    "loop": {"around": ISLAND, "offset_m": 1.0,
                             "length_m": 2 * ((ISLAND[2] - ISLAND[0] + 2.0) + (ISLAND[3] - ISLAND[1] + 2.0))},
                    "source": {"id": "SRCK", "pos": list(SOURCE_XYZ), "gyh_at_1m": SOURCE_GYH_AT_1M},
                    "mark": {"tag": "SM-01", "pos": list(MARK_XY)}, "time_cap_s": 900.0}
    site["ladder"] = {"slots": {"M": "SM-01", "X": "X"}, "dose": {"unit_factor": float(cfg["ladder"]["dose"]["unit_factor"]),
                                                                "rating_gy": float(cfg["ladder"]["dose"]["rating_gy"])},
                      "config": {k: cfg["ladder"][k] for k in ("margin", "head", "reading_place", "findable", "dose",
                                                               "high_step_draw", "crouch_draw", "thermal")
                                 if k in cfg["ladder"]},
                      "acceptance_file": cfg["ladder"]["acceptance_file"]}
    site["deviations"] = []
    site["discrepancies"] = []
    return site


HALL_FACTS = [
    "The training hall's drawing is complete and correct: every wall, lane, platform, panel, island and shield block "
    "is drawn where it is.",
    "Every instrument on the instrument wall holds the value of the calibration sheet.",
    "The check source's strength and position, the survey mark and every grid marker are on the calibration sheet.",
    "Every lane holds one obstacle; the lane table gives its kind, size and the rating that applies to it.",
    "Both link nodes (the base station and the repeater) work.",
]


def _cell_weights(cfg, items):
    """The cell's item weights (mission.scoring.item_weights with the acceptance file's weights)."""
    from . import accept  # noqa: F401  (puts the library root on sys.path for mission)
    from mission.config import load_cfg
    from mission.scoring import item_weights
    from .config import acceptance_path
    wo = {"work_order": {"items": [{"tag": it["tag"], "type": it["type"]} for it in items]}}
    return item_weights(wo, load_cfg(acceptance=acceptance_path(cfg))["weights"])


def _hall_documents(prior, items, lay, cfg, markers):
    """prior.json of the hall: the drawing as built_prior made it (complete: no deviation), the round sheet (every
    wall instrument, the survey mark, the check source), no modification log, the hall's facts, the public format."""
    from .ladder_docs import REPORT, TYPE_TEXT, _scale, public_format
    wo = {i["tag"]: i for i in prior["work_order"]["items"]}
    rows = []
    for it in sorted([i for i in items if i["ladder_slot"] != "X"], key=lambda i: i["tag"]):
        e = wo[it["tag"]]
        loc = dict(e["location"])
        rec = {"tag": it["tag"], "type": it["type"], "report": REPORT[it["type"]], "weight": it["weight"]}
        if it["ladder_slot"] == "M":
            rec.update({"kind": "survey_mark", "type_text": "survey mark (dose rate on the floor)"})
        else:
            rec.update({"kind": "instrument", "type_text": TYPE_TEXT[it["ladder_type"]], "scale": _scale(it)})
            if it.get("label_code"):
                rec["label_code"] = it["label_code"]
            t = it["ladder_type"]
            if t.startswith("gauge"):
                rec["case_mm"] = int(it["params"]["case_mm"])
            if t == "level":
                rec["window_m"] = float(it["params"]["window_m"])
            if t == "thermal":
                rec["surface"] = it["params"]["surface"]
                rec["component"] = (e.get("units") or [""])[0]
            if t == "lamps":
                rec["n_lamps"] = len(it["params"]["colors"])
                rec["colors"] = list(it["params"]["colors"])
            rec["service"] = e.get("service", "")
        loc = {"recorded": True, "room": loc.get("room"), "room_name": loc.get("room_name", ""), "pos": loc.get("pos"),
               "height": loc.get("height"), "facing_deg": loc.get("facing_deg"), "mount": loc.get("mount", "")}
        rec["location"] = loc
        rows.append(rec)
    rows.append({"tag": "UNLISTED", "type": "hotspot", "kind": "unlisted_source", "weight": 0.14,
                 "text": "One source is in the hall: the check source in the shield-block cave on the east wall. "
                         "Report its position.", "report": "x, y of the source in the site frame"})
    prior["round_sheet"] = {"title": "PRACTICE ROUND - TRAINING HALL", "number": prior["work_order"]["number"],
                            "drawing": prior["drawing"]["number"], "last_round": None,
                            "note": "Locations and values: see the calibration sheet. Report each reading with the "
                                    "frame that shows it.", "items": rows}
    prior["modification_log"] = {"title": "MODIFICATION RECORD", "note": "No modification since the drawing.",
                                 "entries": []}
    prior["facility_facts"] = {"statements": HALL_FACTS, "location_bounds": {"listed": "at the drawn place"}}
    prior["drawing"]["title"] = "TRAINING HALL - GENERAL ARRANGEMENT"
    prior["drawing"]["notes"] = ["DIMENSIONS IN METRES FROM THE AIRLOCK DATUM", "DRAWING CHECKED AGAINST THE HALL"]
    prior["drawing"]["revision_date"] = "2026-11-01"
    prior["zoning"]["survey_date"] = "2026-11-01"
    prior["work_order"]["documented_sources"] = [{"room": "200", "count": 1, "text": "check source"}]
    prior = public_format(prior, cfg)
    prior["work_order"]["title"] = "Practice round, training hall"
    return prior


# ============================================================================================ hall documents
def hall_documents(site, prior, cfg, out, face_centres=None):
    """The calibration sheet (JSON + markdown), the lane table and the plan (PNG) of the hall. face_centres: {tag: the
    face centre of mission/evidence.py's face_frame}; without it the sheet's face position is the recorded position."""
    os.makedirs(out, exist_ok=True)
    face_centres = face_centres or {}
    f = float(cfg["ladder"]["dose"]["unit_factor"])
    from .loader.dose import dose_rate
    insts = []
    for r in site["ips"]:
        if r["slot"] in ("M", "X"):
            continue
        tr, prm = r["truth"], r["params"]
        val = {"gauge": lambda: {"value": tr["value"], "unit": tr["unit"], "scale": [prm["lo"], prm["hi"]]},
               "level": lambda: {"level_pct": tr["level_pct"]},
               "valve_lever": lambda: {"angle_deg": tr["angle_deg"], "state": tr["state"]},
               "valve_handwheel": lambda: {"opening_pct": tr["opening_pct"]},
               "lamps": lambda: {"colors": prm["colors"], "states": ["on" if s else "off" for s in tr["states"]]},
               "thermal": lambda: {"temp_C": tr["temp_C"], "surface": tr["surface"], "emissivity": tr["emissivity"]}
               }[r["type"]]()
        rec = {"tag": r["tag"], "type": r["ladder_type"],
               "bay": r["hall"]["bay"], "mounting_height_m": r["hall"]["height_m"],
               # position: the instrument's position as the round sheet gives it (a rising-stem valve: its pipe's centre
               # line); face_pos: the face centre, the reference point of the evidence rules (mission/evidence.py
               # face_frame; a rising-stem valve: its indicator plate, 0.42 m above)
               "position": [round(v, 3) for v in r["pose"]["pos"]],
               "face_pos": face_centres.get(r["tag"], [round(v, 3) for v in r["pose"]["pos"]]),
               "facing_deg": round(math.degrees(r["pose"]["yaw"]), 1), "value": val}
        if r["hall"].get("note"):
            rec["note"] = r["hall"]["note"]
        if r.get("label_code"):
            rec["label_code"] = r["label_code"]          # version 1 only (version 2: text plates)
        insts.append(rec)
    mark = [r for r in site["ips"] if r["slot"] == "M"][0]
    mpos = mark["pose"]["pos"]
    m_rate = float(dose_rate(site, [[mpos[0], mpos[1]]])[0]) * f
    sheet = {"title": "TRAINING HALL - CALIBRATION SHEET", "units": "m, deg, Gy/h; site frame: origin at the pad "
             "centre, x into the hall, y left, z up",
             "instruments": insts,
             "check_source": {"pos": list(SOURCE_XYZ), "strength_gyh_at_1m": SOURCE_GYH_AT_1M,
                              "note": "inverse square in the open; the shield blocks attenuate behind them"},
             "survey_mark": {"tag": "SM-01", "pos": [round(mpos[0], 3), round(mpos[1], 3)],
                             "dose_rate_gyh": round(m_rate, 3), "at_height_m": float(site["radiation"]["height"])},
             "background_gyh": round(float(site["radiation"]["background_uSv_h"]) * f, 4),
             "grid_markers": [{"id": m["id"], "marker_id": m["marker_id"], "pos": m["pos"][:2], "height_m": m["pos"][2],
                               "facing_deg": round(math.degrees(m["yaw"]), 1), "on": m["on"]} for m in site["grid_markers"]],
             "lanes": site["hall"]["lanes"],
             "plates": [{k: v for k, v in p.items() if k != "body"} for p in site["hall"].get("plates", [])],
             "room_plates": list(site["hall"].get("room_plates", [])),
             "odometry": {"straight": site["hall"]["straight"], "loop": site["hall"]["loop"]},
             "link": {"base_station": [round(v, 3) for v in site["link"]["base_station"]["antenna"]],
                      "repeaters": [{"id": r["id"], "pos": r["pos"], "working": True} for r in site["link"]["repeaters"]],
                      "base_range_m": float(cfg["link"]["base_range"]), "repeater_range_m": float(cfg["link"]["repeater_range"])},
             "time_cap_s": 900.0}
    json.dump(sheet, open(os.path.join(out, "calibration_sheet.json"), "w"), indent=1, ensure_ascii=False)
    md = ["# Training hall: calibration sheet", "",
          "Site frame: origin at the deployment pad centre (the datum), x into the hall, y to the left, z up; metres, "
          "degrees, Gy/h. The drawing of the hall is complete and correct.", "",
          "## Instrument wall (south wall)", "",
          "Position: the instrument's position as the round sheet gives it. Face centre: the centre of the face the "
          "reading and the evidence rules are about (a rising-stem valve: its indicator plate, 0.42 m above the pipe; "
          "a quarter-turn valve: the lever's pivot in front of the pipe; a level gauge: the middle of the tube and "
          "its scale plate; a lamp panel: the front of the panel).", "",
          ]
    codes = any(i.get("label_code") for i in insts)          # version 1 only: the plates' label codes
    md += (["| tag | code | type | bay | mounting height (m) | position (m) | face centre (m) | value | note |",
            "|---|---|---|---|---|---|---|---|---|"] if codes else
           ["| tag | type | bay | mounting height (m) | position (m) | face centre (m) | value | note |",
            "|---|---|---|---|---|---|---|---|"])
    for i in insts:
        cells = [i["tag"]] + ([i.get("label_code", "-")] if codes else []) + [
            i["type"], "%d" % i["bay"], "%.1f" % i["mounting_height_m"], str(i["position"]), str(i["face_pos"]),
            ", ".join("%s %s" % (k, v) for k, v in i["value"].items()), i.get("note", "")]
        md.append("| " + " | ".join(cells) + " |")
    if sheet["plates"]:
        md += ["", "## Plate board and posts (west wall, at the end of the odometry straight)", "",
               "Text plates of the plant standard (yellow, black lettering and border line). Centre: the centre of "
               "the plate's front face. Facing: the direction its front faces.", "",
               "| text | centre (m) | height (m) | facing (deg) | backing | where |", "|---|---|---|---|---|---|"]
        for q in sheet["plates"]:
            md.append("| %s | %s | %.2f | %.1f | %s | %s |" % (q["text"], q["pos"], q["height_m"], q["facing_deg"],
                                                                q["backing"], q["where"]))
    if sheet["room_plates"]:
        md += ["", "## Room plates", "",
               "Room plates of the plant standard (white, 0.40 x 0.10 m, the room's number and name in the plant "
               "lettering). Centre: the centre of the plate's front face.", "",
               "| number | name | centre (m) | height (m) | facing (deg) | where |", "|---|---|---|---|---|---|"]
        for q in sheet["room_plates"]:
            md.append("| %s | %s | %s | %.2f | %.1f | %s |" % (q["number"], q["name"], q["pos"], q["height_m"],
                                                                q["facing_deg"], q["where"]))
    md += ["", "## Check source and survey mark", "",
           "- check source at (%.2f, %.2f), %.2f m above the floor: %.0f Gy/h at 1 m" % (
               SOURCE_XYZ[0], SOURCE_XYZ[1], SOURCE_XYZ[2], SOURCE_GYH_AT_1M),
           "- survey mark SM-01 at (%.2f, %.2f): %.3f Gy/h at the dosimeter height %.2f m" % (
               mpos[0], mpos[1], m_rate, float(site["radiation"]["height"])),
           "- background %.4f Gy/h" % (float(site["radiation"]["background_uSv_h"]) * f), "",
           "## Lanes", "", "| lane | row | obstacle | size | rating | x (m) | obstacle at y (m) | mouth at y (m) |",
           "|---|---|---|---|---|---|---|---|"]
    for L in site["hall"]["lanes"]:
        md.append("| %s | %s | %s | %s %s | %s | %s | %.2f | %.2f |" % (L["lane"], L["row"], L["kind"], L["size"],
                                                                      L["unit"], L["class"], L["x"], L["obstacle_y"],
                                                                      L["mouth_y"]))
    md += ["", "## Grid markers", "", "| id | marker id | x, y (m) | height (m) | facing (deg) |", "|---|---|---|---|---|"]
    for m in sheet["grid_markers"]:
        md.append("| %s | %d | %s | %.2f | %.1f |" % (m["id"], m["marker_id"], m["pos"], m["height_m"], m["facing_deg"]))
    st = site["hall"]["straight"]
    lp = site["hall"]["loop"]
    md += ["", "## Odometry", "", "- straight: from %s to %s (%.0f m), a floor line with 1 m ticks" % (
        st["from"], st["to"], st["length_m"]),
        "- loop: a floor line 1.0 m around the pallet island %s (%.0f m)" % (lp["around"], lp["length_m"]), ""]
    open(os.path.join(out, "calibration_sheet.md"), "w").write("\n".join(md) + "\n")
    try:
        from .drawing import prior_drawing
        prior_drawing(prior, os.path.join(out, "hall_plan.png"), site["building"]["bbox"])
    except Exception as e:  # noqa: BLE001  (the plan image is a convenience; the sheet is the record)
        open(os.path.join(out, "hall_plan_error.txt"), "w").write("%s: %s\n" % (type(e).__name__, e))
    return sheet


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--maps", action="store_true")
    a = ap.parse_args(argv)
    t0 = time.time()
    site, prior, docs = build_hall(a.out, maps=a.maps)
    print(json.dumps({"out": a.out, "ips": len(site["ips"]), "lanes": len(site["hall"]["lanes"]),
                      "elements": len(site["elements"]), "markers": len(site["grid_markers"]),
                      "timing_s": round(time.time() - t0, 1)}))


if __name__ == "__main__":
    main()
