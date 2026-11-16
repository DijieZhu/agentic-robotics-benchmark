"""Build the house-cleaning cell's MuJoCo model: the room and its decor, the kitchenette, the bins, the tool rail, the
dining set, the coffee table, the bookshelf, the loose objects (free bodies), the crumbs (drawn-only mocap bodies),
the grease film on the dining table (drawn-only cells), the water stream, one G1 humanoid, the inactive weld pool
(hand grips, tool clips), the cameras and the lights.

Collision bits (robot.py): 1 = the static room and its fixtures, 2 = the robot, 4 = loose objects and tools, 8 = the
floor. What the cameras see of the room shell (floor, walls, ceiling, window, door, pictures, clock, ceiling panels,
couch, plants, rug) is furniture_assembly_from_drawings's room decor (`add_decor`, the same room); the room's own
floor, walls, window pane, couch, plant and rug geoms are colliders hidden from the cameras (group 3). Everything
else the cameras see is drawn from the procedural meshes and textures of props.py / objects.py.
"""

from __future__ import annotations

import json
import math
import os

import mujoco
import numpy as np

from . import objects as O
from . import props as P
from . import room as RM
from .render import filament_view_keys
from .robot import FLOOR, OBJ, ROBOT, WORLD, g1_spec

G = mujoco.mjtGeom
DECOR_DIR = os.environ.get("ARB_ROOM_DECOR_DIR", "/opt/arb-assets/room_decor")
DECOR_REPLACES = ("floor", "wall_", "window", "couch_", "plant_", "rug")   # geom names (prefixes) the decor draws
HIDDEN_GROUP = 3
ROBOT_NAME = "a"
STATIC = dict(contype=WORLD, conaffinity=ROBOT | OBJ, condim=3, group=3, friction=[0.6, 0.005, 0.0001])
# the model's extent and centre without the decor (the classic renderer's clip planes and the sun's shadow map are fitted
# to them; the decor's street view reaches 100 m out and would move both); measured on this cell's model
ROOM_EXTENT, ROOM_CENTER = 11.970811909470342, (3.0, 2.76998941488475, 1.3)
SUN_DIR = (-0.8715359, -0.1852507, -0.4539905)
PANELS = ((1.8, 2.0), (4.0, 2.2))
# upward shadow-free spots under the sunlit patches (the light they throw back up), as furniture_assembly's cell
# places them: per patch cell its centre and the depth below it of its spot (scene.bounce_cells measures the patches
# of this room by ray casting; the numbers below are its output for the cell's fixed furniture)
BOUNCE = (("floor0", (2.45, 1.30, 0.00), 0.26),
          ("floor1", (3.05, 1.55, 0.00), 0.28),
          ("floor2", (3.55, 1.95, 0.00), 0.30),
          ("floor3", (4.05, 2.15, 0.00), 0.27),
          ("table0", (2.95, 2.40, 0.75), 0.24))
CRUMB_COLOURS = ((0.80, 0.62, 0.40), (0.70, 0.50, 0.30), (0.90, 0.78, 0.50), (0.45, 0.30, 0.18), (0.93, 0.90, 0.82),
                 (0.62, 0.42, 0.22))
GREASE_RGB = (0.34, 0.22, 0.07)
FIXTURE_PARTS = ("bin0_lid", "bin1_lid", "faucet_lever", "soap_pump")


def _quat(R):
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(R, float).reshape(-1))
    return list(q)


def _quat_z(yaw):
    return [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]


def _look_at_quat(pos, target):
    pos, target = np.asarray(pos, float), np.asarray(target, float)
    zc = pos - target
    zc /= np.linalg.norm(zc)
    xc = np.cross([0.0, 0.0, 1.0], zc)
    xc /= np.linalg.norm(xc)
    yc = np.cross(zc, xc)
    return _quat(np.stack([xc, yc, zc], axis=1))


def box_geom(body, name, box, *, collide=True, visual_mat=None, b=None, radius=0.0, uv_scale=(1.0, 1.0),
             contact=STATIC, visual_name=None, **kw):
    """A box given as (x0, y0, z0, x1, y1, z1) in the body's frame: a hidden collider (if collide) and, with
    visual_mat, a drawn mesh (rounded edges, box-projected texture coordinates in metres x uv_scale)."""
    x0, y0, z0, x1, y1, z1 = box
    c = ((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2)
    h = ((x1 - x0) / 2, (y1 - y0) / 2, (z1 - z0) / 2)
    if collide:
        args = dict(contact)
        args.update(kw)
        body.add_geom(name=name, type=G.mjGEOM_BOX, size=list(h), pos=list(c), **args)
    if visual_mat is not None:
        mesh = P.scale_uv(P.rounded_box(h, radius, 2 if radius == 0 else 3), *uv_scale)
        b.visual(body, visual_name or (name + "_vis"), mesh, visual_mat, pos=c)


def add_decor(s: mujoco.MjSpec, decor_dir: str = DECOR_DIR) -> None:
    """The room's decor (furniture_assembly_from_drawings's decor.json: meshes in the room frame, their materials and
    base maps) as drawn-only geoms of one static body `room_decor`; the geoms it draws instead are moved to the
    hidden group. Raises when the decor is missing."""
    path = os.path.join(decor_dir, "decor.json")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"the room's decor is missing: {path} (ARB_ROOM_DECOR_DIR)")
    with open(path) as f:
        decor = json.load(f)
    for g in s.worldbody.geoms:
        if (g.name or "").startswith(DECOR_REPLACES):
            g.group = HIDDEN_GROUP
    s.stat.extent, s.stat.center = ROOM_EXTENT, list(ROOM_CENTER)
    for name, rec in decor["materials"].items():
        roles = []
        if rec.get("texture"):
            tex = s.add_texture(name=f"decor_{name}", type=mujoco.mjtTexture.mjTEXTURE_2D,
                                file=os.path.join(decor_dir, rec["texture"]),
                                colorspace=mujoco.mjtColorSpace.mjCOLORSPACE_SRGB)
            roles = ["", tex.name]
        s.add_material(name=f"decor_{name}", rgba=list(rec["rgba"]), textures=roles,
                       emission=float(rec.get("emission", 0.0)))
    body = s.worldbody.add_body(name="room_decor")
    for rec in decor["meshes"]:
        z = np.load(os.path.join(decor_dir, rec["file"]))
        uv = np.asarray(z["uv"], np.float64).copy()
        uv[:, 1] = 1.0 - uv[:, 1]
        mesh = s.add_mesh(name=f"decor_{rec['name']}", inertia=mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL)
        mesh.uservert = np.asarray(z["v"], np.float64).ravel()
        mesh.usernormal = np.asarray(z["n"], np.float64).ravel()
        mesh.usertexcoord = uv.ravel()
        mesh.userface = np.asarray(z["f"], np.int32).ravel()
        body.add_geom(name=f"decor_{rec['name']}", type=G.mjGEOM_MESH, meshname=mesh.name,
                      material=f"decor_{rec['material']}", contype=0, conaffinity=0, group=1, mass=0.0)


# ================================================================================================ room shell


def add_shell(s, b):
    wb = s.worldbody
    s.add_material(name="floor_mat", rgba=[0.55, 0.40, 0.27, 1.0])
    s.add_material(name="wall_mat", rgba=[0.86, 0.85, 0.82, 1.0])
    s.add_material(name="window_mat", rgba=[0.70, 0.82, 0.92, 1.0])
    wb.add_geom(name="floor", type=G.mjGEOM_PLANE, size=[RM.ROOM_X, RM.ROOM_Y, 0.05],
                pos=[RM.ROOM_X / 2, RM.ROOM_Y / 2, 0.0], material="floor_mat", contype=FLOOR, conaffinity=OBJ,
                friction=[0.6, 0.005, 0.0001])
    t = RM.WALL_T
    walls = {
        "wall_w": ([-t / 2, RM.ROOM_Y / 2, RM.WALL_H / 2], [t / 2, RM.ROOM_Y / 2 + t, RM.WALL_H / 2]),
        "wall_e": ([RM.ROOM_X + t / 2, RM.ROOM_Y / 2, RM.WALL_H / 2], [t / 2, RM.ROOM_Y / 2 + t, RM.WALL_H / 2]),
        "wall_n": ([RM.ROOM_X / 2, RM.ROOM_Y + t / 2, RM.WALL_H / 2], [RM.ROOM_X / 2, t / 2, RM.WALL_H / 2]),
    }
    d0, d1 = RM.DOOR_X
    walls["wall_s1"] = ([d0 / 2, -t / 2, RM.WALL_H / 2], [d0 / 2, t / 2, RM.WALL_H / 2])
    walls["wall_s2"] = ([(d1 + RM.ROOM_X) / 2, -t / 2, RM.WALL_H / 2], [(RM.ROOM_X - d1) / 2, t / 2, RM.WALL_H / 2])
    walls["wall_s3"] = ([(d0 + d1) / 2, -t / 2, (RM.WALL_H + RM.DOOR_H) / 2],
                        [(d1 - d0) / 2, t / 2, (RM.WALL_H - RM.DOOR_H) / 2])
    for name, (c, h) in walls.items():
        wb.add_geom(name=name, type=G.mjGEOM_BOX, size=h, pos=c, material="wall_mat", contype=WORLD,
                    conaffinity=ROBOT | OBJ, friction=[0.4, 0.005, 0.0001])
    w0, w1 = RM.WINDOW_Y
    wb.add_geom(name="window", type=G.mjGEOM_BOX, size=[0.005, (w1 - w0) / 2, 0.6],
                pos=[RM.ROOM_X - 0.004, (w0 + w1) / 2, 1.5], material="window_mat", contype=0, conaffinity=0)
    # the customer's couch and plant (obstacles, drawn by the decor) and rug (drawn only)
    still = dict(contype=WORLD, conaffinity=ROBOT | OBJ, friction=[0.6, 0.005, 0.0001])
    s.add_material(name="couch_mat", rgba=[0.15, 0.14, 0.13, 1.0])
    s.add_material(name="plant_pot_mat", rgba=[0.17, 0.15, 0.15, 1.0])
    s.add_material(name="plant_foliage_mat", rgba=[0.34, 0.42, 0.15, 1.0])
    s.add_material(name="rug_mat", rgba=[0.33, 0.34, 0.35, 1.0])
    for name, (x0, y0, z0, x1, y1, z1) in RM.COUCH["boxes"].items():
        wb.add_geom(name=f"couch_{name}", type=G.mjGEOM_BOX, size=[(x1 - x0) / 2, (y1 - y0) / 2, (z1 - z0) / 2],
                    pos=[(x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2], material="couch_mat", **still)
    legs = RM.COUCH["legs"]
    for i, (cx, cy) in enumerate(legs["centers"]):
        wb.add_geom(name=f"couch_leg{i}", type=G.mjGEOM_BOX, size=[legs["side"] / 2, legs["side"] / 2,
                                                                  legs["height"] / 2],
                    pos=[cx, cy, legs["height"] / 2], material="couch_mat", **still)
    for name, cyl in RM.PLANT.items():
        z0, z1 = cyl["z"]
        wb.add_geom(name=f"plant_{name}", type=G.mjGEOM_CYLINDER, size=[cyl["radius"], (z1 - z0) / 2, 0.0],
                    pos=[cyl["center"][0], cyl["center"][1], (z0 + z1) / 2], material=f"plant_{name}_mat", **still)
    rug = RM.RUG
    wb.add_geom(name="rug", type=G.mjGEOM_BOX, size=[rug["size"][0] / 2, rug["size"][1] / 2, rug["thickness"] / 2],
                pos=[rug["center"][0], rug["center"][1], rug["thickness"] / 2], material="rug_mat",
                contype=0, conaffinity=0, group=1)


# ================================================================================================ kitchenette


def add_kitchen(s, b):
    wb = s.worldbody
    K = RM.KITCHEN
    sage = b.material("cabinet_sage", tex="sage")
    quartz = b.material("quartz", tex="quartz")
    plinth = b.material("plinth_dark", (0.16, 0.16, 0.16, 1))
    brass = b.material("handle_brass", (0.72, 0.58, 0.32, 1))
    body = wb.add_body(name="kitchen")
    x0, x1, y0, y1 = K["x0"], K["x1"], K["y0"], K["y1"]
    top, tt = K["top_z"], K["top_t"]
    box_geom(body, "kitchen_plinth", (x0, y0, 0.0, K["plinth_x1"], y1, K["plinth_h"]), visual_mat=plinth, b=b)
    # the carcass collider leaves room for the sink's basin (under it only up to the basin's bottom)
    S = RM.SINK
    sy0, sy1 = S["y0"] - S["wall_t"], S["y1"] + S["wall_t"]
    box_geom(body, "kitchen_carcass", (x0, y0, K["plinth_h"], x1 - 0.019, sy0, top - tt), visual_mat=None, b=b)
    box_geom(body, "kitchen_carcass_n", (x0, sy1, K["plinth_h"], x1 - 0.019, y1, top - tt), visual_mat=None, b=b)
    box_geom(body, "kitchen_carcass_sink", (x0, sy0, K["plinth_h"], x1 - 0.019, sy1, S["bottom_z"] - S["wall_t"]),
             visual_mat=None, b=b)
    box_geom(body, "kitchen_carcass_back", (x0, sy0, S["bottom_z"] - S["wall_t"], S["x0"] - S["wall_t"], sy1,
                                            top - tt), visual_mat=None, b=b)
    for k, (ya, yb) in (("s", (y0, y0 + 0.018)), ("n", (y1 - 0.018, y1))):
        box_geom(body, f"kitchen_end_{k}", (x0, ya, 0.0, x1, yb, top - tt), collide=False, visual_mat=sage, b=b,
                 radius=0.002, uv_scale=(2.0, 2.0))
    # door and drawer fronts per module, 3 mm gaps, brass bar handles
    mods = K["modules"]
    S = RM.SINK
    for i in range(len(mods) - 1):
        ya, yb = mods[i] + 0.0015, mods[i + 1] - 0.0015
        under_sink = ya < (S["y0"] + S["y1"]) / 2 < yb
        box_geom(body, f"kitchen_door{i}", (x1 - 0.019, ya, K["plinth_h"] + 0.003, x1, yb, 0.672 if not under_sink
                                            else 0.615), collide=False, visual_mat=sage, b=b, radius=0.002,
                 uv_scale=(2.0, 2.0))
        if not under_sink:
            box_geom(body, f"kitchen_drawer{i}", (x1 - 0.019, ya, 0.678, x1, yb, top - tt - 0.003), collide=False,
                     visual_mat=sage, b=b, radius=0.002, uv_scale=(2.0, 2.0))
        yc = (ya + yb) / 2
        q = _quat([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
        if under_sink:
            continue
        b.prim(body, f"kitchen_handle{i}", G.mjGEOM_CAPSULE, [0.006, 0.07, 0], (x1 + 0.016, yc, 0.785),
               _quat([[0, 0, 1], [1, 0, 0], [0, 1, 0]]), visual=True, material=brass)
        for dy in (-0.07, 0.07):
            b.prim(body, f"kitchen_handle{i}_post{'ab'[dy > 0]}", G.mjGEOM_CYLINDER, [0.004, 0.008, 0],
                   (x1 + 0.008, yc + dy, 0.785), q, visual=True, material=brass)
    # countertop round the apron-front sink (three pieces: no strip in front of the basin)
    S = RM.SINK
    pieces = {"s": (x0, y0, K["top_x1"], S["y0"]), "n": (x0, S["y1"], K["top_x1"], y1),
              "w": (x0, S["y0"], S["x0"], S["y1"])}
    for k, (a0, b0, a1, b1) in pieces.items():
        box_geom(body, f"counter_{k}", (a0, b0, top - tt, a1, b1, top), visual_mat=quartz, b=b, radius=0.003,
                 uv_scale=(1.6, 1.6))
    # the sink: a white fireclay basin, its front wall (the apron) flush with the cabinets' doors
    clay = b.material("fireclay", (0.95, 0.95, 0.94, 1))
    wt, bz = S["wall_t"], S["bottom_z"]
    box_geom(body, "sink_bottom", (S["x0"], S["y0"], bz - wt, S["x1"], S["y1"], bz), visual_mat=clay, b=b)
    box_geom(body, "sink_wall_w", (S["x0"] - wt, S["y0"] - wt, bz - wt, S["x0"], S["y1"] + wt, top - tt),
             visual_mat=clay, b=b)
    box_geom(body, "sink_apron", (S["x1"], S["y0"] - wt, 0.62, S["apron_x1"], S["y1"] + wt, top), visual_mat=clay,
             b=b, radius=0.004)
    box_geom(body, "sink_wall_s", (S["x0"], S["y0"] - wt, bz - wt, S["x1"], S["y0"], top - tt), visual_mat=clay, b=b)
    box_geom(body, "sink_wall_n", (S["x0"], S["y1"], bz - wt, S["x1"], S["y1"] + wt, top - tt), visual_mat=clay, b=b)
    sx, sy = RM.SINK_CENTER
    b.prim(body, "sink_drain", G.mjGEOM_CYLINDER, [0.024, 0.0008, 0], (sx, sy, bz + 0.0008), visual=True,
           material=b.material("drain_steel", (0.62, 0.63, 0.64, 1)))
    steel = b.material("steel", (0.72, 0.73, 0.74, 1))
    # the sponge tray: a shallow steel tray on the counter
    T = RM.SPONGE_TRAY
    (tx, ty), (tw, tl), rh = T["center"], T["size"], T["rim_h"]
    box_geom(body, "tray_bottom", (tx - tw / 2, ty - tl / 2, top, tx + tw / 2, ty + tl / 2, top + 0.002),
             visual_mat=steel, b=b)
    for k, bx in (("w", (tx - tw / 2, ty - tl / 2, top, tx - tw / 2 + 0.003, ty + tl / 2, top + rh)),
                  ("e", (tx + tw / 2 - 0.003, ty - tl / 2, top, tx + tw / 2, ty + tl / 2, top + rh)),
                  ("s", (tx - tw / 2, ty - tl / 2, top, tx + tw / 2, ty - tl / 2 + 0.003, top + rh)),
                  ("n", (tx - tw / 2, ty + tl / 2 - 0.003, top, tx + tw / 2, ty + tl / 2, top + rh))):
        box_geom(body, f"tray_rim_{k}", bx, visual_mat=steel, b=b)
    # the dish tray: a grey rubber tray with a low rim (where the dirty dishes go)
    D = RM.DISH_TRAY
    (dx_, dy_), (dw, dl), drh, dbt = D["center"], D["size"], D["rim_h"], D["base_t"]
    rubber = b.material("tray_rubber", (0.30, 0.31, 0.32, 1))
    box_geom(body, "dish_tray", (dx_ - dw / 2, dy_ - dl / 2, top, dx_ + dw / 2, dy_ + dl / 2, top + dbt),
             visual_mat=rubber, b=b, radius=0.002)
    for k, bx in (("w", (dx_ - dw / 2, dy_ - dl / 2, top, dx_ - dw / 2 + 0.008, dy_ + dl / 2, top + drh)),
                  ("e", (dx_ + dw / 2 - 0.008, dy_ - dl / 2, top, dx_ + dw / 2, dy_ + dl / 2, top + drh)),
                  ("s", (dx_ - dw / 2, dy_ - dl / 2, top, dx_ + dw / 2, dy_ - dl / 2 + 0.008, top + drh)),
                  ("n", (dx_ - dw / 2, dy_ + dl / 2 - 0.008, top, dx_ + dw / 2, dy_ + dl / 2, top + drh))):
        box_geom(body, f"dish_tray_rim_{k}", bx, visual_mat=rubber, b=b, radius=0.003)
    # the induction hob (flush, drawn only)
    hx0, hy0, hx1, hy1 = RM.HOB
    hobmat = b.material("hob_glass", tex="hob")
    hob = P.scale_uv(P.rounded_box([(hx1 - hx0) / 2, (hy1 - hy0) / 2, 0.002], 0.004, 3), 1 / (hx1 - hx0),
                     1 / (hy1 - hy0))
    b.visual(body, "hob", hob, hobmat, pos=((hx0 + hx1) / 2, (hy0 + hy1) / 2, top + 0.0005))
    # wall cabinets and the tiled splashback
    U = RM.UPPER
    box_geom(body, "upper_carcass", (U["x0"], U["y0"], U["z0"], U["x1"] - 0.019, U["y1"], U["z1"]), b=b)
    offwhite = b.material("cabinet_offwhite", tex="offwhite")
    for i in range(len(mods) - 1):
        ya, yb = mods[i] + 0.0015, mods[i + 1] - 0.0015
        box_geom(body, f"upper_door{i}", (U["x1"] - 0.019, ya, U["z0"] + 0.003, U["x1"], yb, U["z1"] - 0.003),
                 collide=False, visual_mat=offwhite, b=b, radius=0.002, uv_scale=(2.0, 2.0))
        b.prim(body, f"upper_knob{i}", G.mjGEOM_SPHERE, [0.011, 0, 0],
               (U["x1"] + 0.010, yb - 0.05 if i % 2 == 0 else ya + 0.05, U["z0"] + 0.06), visual=True, material=brass)
    box_geom(body, "upper_underside", (U["x0"], U["y0"], U["z0"] - 0.004, U["x1"], U["y1"], U["z0"]), collide=False,
             visual_mat=offwhite, b=b)
    for k, (ya, yb) in (("s", (U["y0"], U["y0"] + 0.018)), ("n", (U["y1"] - 0.018, U["y1"]))):
        box_geom(body, f"upper_end_{k}", (U["x0"], ya, U["z0"], U["x1"], yb, U["z1"]), collide=False,
                 visual_mat=offwhite, b=b, radius=0.002, uv_scale=(2.0, 2.0))
    SB = RM.SPLASHBACK
    tilemat = b.material("tiles", tex="tiles")
    tile_mesh = P.scale_uv(P.rounded_box([0.003, (SB["y1"] - SB["y0"]) / 2, (SB["z1"] - SB["z0"]) / 2], 0.0, 1),
                           1 / 0.45, 1 / 0.31)
    b.visual(body, "splashback", tile_mesh, tilemat, pos=(0.003, (SB["y0"] + SB["y1"]) / 2, (SB["z0"] + SB["z1"]) / 2))
    # kitchen props (static, drawn; colliders where something might be put down on them)
    oak = b.material("oak", tex="oak")
    board = P.scale_uv(P.rounded_box([0.009, 0.16, 0.21], 0.008, 3), 2, 2)
    b.visual(body, "cutting_board", board, oak, pos=(0.035, 1.60, top + 0.215),
             quat=_quat(np.array([[math.cos(0.18), 0, math.sin(0.18)], [0, 1, 0], [-math.sin(0.18), 0,
                                                                                    math.cos(0.18)]])))
    crock = b.material("crock", (0.85, 0.83, 0.78, 1))
    b.visual(body, "crock", P.lathe([(0.0, 0.0), (0.052, 0.0), (0.056, 0.02), (0.055, 0.14), (0.058, 0.15),
                                     (0.050, 0.15), (0.048, 0.02), (0.0, 0.02)], segments=28), crock,
             pos=(0.10, 3.27, top))
    box_geom(body, "crock_col", (0.045, 3.215, top, 0.155, 3.325, top + 0.15), b=b)
    for i, (dx, dy, tilt) in enumerate(((0.012, 0.0, 0.10), (-0.012, 0.01, -0.12), (0.0, -0.014, 0.05))):
        q = _quat(np.array([[1, 0, 0], [0, math.cos(tilt), -math.sin(tilt)], [0, math.sin(tilt), math.cos(tilt)]]))
        b.prim(body, f"utensil{i}", G.mjGEOM_CAPSULE, [0.008, 0.11, 0], (0.10 + dx, 3.27 + dy, top + 0.22), q,
               visual=True, material=b.material("utensil_wood", (0.55, 0.38, 0.22, 1)) if i != 1 else
               b.material("utensil_black", (0.08, 0.08, 0.09, 1)))
    pot = b.material("herb_pot", (0.70, 0.36, 0.24, 1))
    leaf = b.material("herb_leaf", (0.20, 0.42, 0.18, 1))
    b.visual(body, "herb_pot", P.lathe([(0.0, 0.0), (0.045, 0.0), (0.058, 0.11), (0.0, 0.11)], segments=24), pot,
             pos=(0.11, 1.25, top))
    box_geom(body, "herb_col", (0.05, 1.19, top, 0.17, 1.31, top + 0.11), b=b)
    rng = np.random.default_rng(5)
    for i in range(14):
        a = rng.uniform(0, 2 * math.pi)
        rr = rng.uniform(0.0, 0.04)
        b.prim(body, f"herb_leaf{i}", G.mjGEOM_ELLIPSOID, [0.022, 0.012, 0.004],
               (0.11 + rr * math.cos(a), 1.25 + rr * math.sin(a), top + 0.13 + rng.uniform(0, 0.08)),
               _quat_z(a)[:1] + _quat_z(a)[1:], visual=True, material=leaf)
    return body


# ================================================================================================ the rest of the room


def add_tool_rail(s, b):
    wb = s.worldbody
    T = RM.TOOL_RAIL
    steel = b.material("steel", (0.72, 0.73, 0.74, 1))
    black = b.material("clip_black", (0.07, 0.07, 0.08, 1))
    body = wb.add_body(name="tool_rail")
    box_geom(body, "tool_rail", (T["x0"], 0.0, T["z"] - 0.03, T["x1"], T["depth"], T["z"] + 0.03), visual_mat=steel,
             b=b, radius=0.004)
    for name, (cx, cy, cz) in RM.CLIPS.items():
        # a spring clip: two rubber jaws either side of the handle's place (drawn only: the clip rule holds the tool)
        for sgn in (-1, 1):
            b.prim(body, f"clip_{name}_jaw{'ab'[sgn > 0]}", G.mjGEOM_BOX, [0.006, (cy - T["depth"]) / 2 + 0.012, 0.022],
                   (cx + sgn * 0.019, (cy + T["depth"]) / 2 + 0.006, cz), visual=True, material=black)
        b.prim(body, f"clip_{name}_base", G.mjGEOM_BOX, [0.03, 0.006, 0.026], (cx, T["depth"] + 0.006, cz),
               visual=True, material=black)


def add_dining(s, b):
    wb = s.worldbody
    D = RM.DINING
    oak = b.material("oak", tex="oak")
    blacksteel = b.material("black_steel", (0.07, 0.07, 0.075, 1))
    body = wb.add_body(name="dining")
    (cx, cy), (lx, ly) = D["center"], D["size"]
    top, tt = D["top_z"], D["top_t"]
    box_geom(body, "dining_top", (cx - lx / 2, cy - ly / 2, top - tt, cx + lx / 2, cy + ly / 2, top),
             visual_mat=oak, b=b, radius=0.006, uv_scale=(1.0, 2.0))
    lg, ins = D["leg"], D["leg_inset"]
    for i, (sx, sy) in enumerate(((-1, -1), (-1, 1), (1, -1), (1, 1))):
        x = cx + sx * (lx / 2 - ins)
        y = cy + sy * (ly / 2 - ins)
        box_geom(body, f"dining_leg{i}", (x - lg / 2, y - lg / 2, 0.0, x + lg / 2, y + lg / 2, top - tt),
                 visual_mat=blacksteel, b=b, radius=0.004)
    box_geom(body, "dining_apron", (cx - lx / 2 + ins, cy - ly / 2 + ins, top - tt - 0.06, cx + lx / 2 - ins,
                                    cy + ly / 2 - ins, top - tt), visual_mat=blacksteel, b=b)
    C = RM.CHAIR
    fab = b.material("seat_fabric", tex="seat_fabric")
    for name, ch in RM.CHAIRS.items():
        x, y = ch["center"]
        yaw = ch["yaw"]
        cb = wb.add_body(name=f"chair_{name}", pos=[x, y, 0.0], quat=_quat_z(yaw))
        w, l = C["seat"]
        sz, st = C["seat_z"], C["seat_t"]
        box_geom(cb, f"chair_{name}_seat", (-w / 2, -l / 2, sz - st, w / 2, l / 2, sz), visual_mat=fab, b=b,
                 radius=0.01, uv_scale=(3.0, 3.0))
        # the back on the chair's far side (local -x: the chair faces +x, towards the table)
        box_geom(cb, f"chair_{name}_back", (-w / 2, -l / 2 + 0.01, sz + 0.12, -w / 2 + C["back_t"], l / 2 - 0.01,
                                            C["back_h"]), visual_mat=oak, b=b, radius=0.008, uv_scale=(2.0, 2.0))
        for i, (sx, sy) in enumerate(((-1, -1), (-1, 1), (1, -1), (1, 1))):
            px, py = sx * (w / 2 - 0.03), sy * (l / 2 - 0.03)
            z1 = C["back_h"] - 0.02 if sx < 0 else sz - st
            box_geom(cb, f"chair_{name}_leg{i}", (px - C["leg"] / 2, py - C["leg"] / 2, 0.0, px + C["leg"] / 2,
                                                  py + C["leg"] / 2, z1), visual_mat=oak, b=b, radius=0.006,
                     uv_scale=(2.0, 2.0))


def add_pendant(s, b):
    """A pendant lamp over the dining table (drawn only): a cord from the ceiling and a cone shade glowing inside."""
    wb = s.worldbody
    (cx, cy) = RM.DINING["center"]
    body = wb.add_body(name="pendant")
    cord = b.material("pendant_cord", (0.05, 0.05, 0.05, 1))
    shade = b.material("pendant_shade", (0.12, 0.30, 0.26, 1))
    glow = b.material("pendant_glow", (1.0, 0.92, 0.78, 1), emission=0.8)
    b.prim(body, "pendant_cord", G.mjGEOM_CYLINDER, [0.003, 0.36, 0], (cx, cy, RM.WALL_H - 0.36), visual=True,
           material=cord)
    zt = RM.WALL_H - 0.72
    # (the profile bottom to top, so the shade's outside faces out: Filament culls back faces)
    b.visual(body, "pendant_shade", P.lathe([(0.175, zt - 0.175), (0.17, zt - 0.17), (0.05, zt - 0.02), (0.02, zt)],
                                            segments=40), shade)
    b.visual(body, "pendant_inner", P.lathe([(0.168, zt - 0.168), (0.048, zt - 0.025), (0.0, zt - 0.02)],
                                            segments=40), glow)
    b.prim(body, "pendant_bulb", G.mjGEOM_SPHERE, [0.035, 0, 0], (cx, cy, zt - 0.07), visual=True, material=glow)
    # the cord and the shade are drawn about the lamp's axis: shift the meshes to the table's middle
    for g in body.geoms:
        if g.type == G.mjGEOM_MESH:
            g.pos = [cx, cy, 0.0]


def add_coffee(s, b):
    wb = s.worldbody
    C = RM.COFFEE
    walnut = b.material("walnut", tex="walnut")
    blacksteel = b.material("black_steel", (0.07, 0.07, 0.075, 1))
    body = wb.add_body(name="coffee")
    (cx, cy), (lx, ly) = C["center"], C["size"]
    top, tt = C["top_z"], C["top_t"]
    box_geom(body, "coffee_top", (cx - lx / 2, cy - ly / 2, top - tt, cx + lx / 2, cy + ly / 2, top),
             visual_mat=walnut, b=b, radius=0.008, uv_scale=(2.0, 2.0))
    lg, ins = C["leg"], C["leg_inset"]
    for i, (sx, sy) in enumerate(((-1, -1), (-1, 1), (1, -1), (1, 1))):
        x = cx + sx * (lx / 2 - ins)
        y = cy + sy * (ly / 2 - ins)
        box_geom(body, f"coffee_leg{i}", (x - lg / 2, y - lg / 2, 0.0, x + lg / 2, y + lg / 2, top - tt),
                 visual_mat=blacksteel, b=b, radius=0.004)
    # nothing under the top between the legs: room for the robot's knees when it squats at the table


def add_bookshelf(s, b):
    wb = s.worldbody
    S = RM.SHELF
    birch = b.material("birch", tex="birch")
    body = wb.add_body(name="bookshelf")
    x0, x1, y0, y1, H, t = S["x0"], S["x1"], S["y0"], S["y1"], S["height"], S["board"]
    box_geom(body, "shelf_side_w", (x0, y0, 0.0, x0 + t, y1, H), visual_mat=birch, b=b, uv_scale=(2, 2))
    box_geom(body, "shelf_side_e", (x1 - t, y0, 0.0, x1, y1, H), visual_mat=birch, b=b, uv_scale=(2, 2))
    box_geom(body, "shelf_top", (x0, y0, H - t, x1, y1, H), visual_mat=birch, b=b, uv_scale=(2, 2))
    box_geom(body, "shelf_plinth", (x0 + t, y0 + 0.01, 0.0, x1 - t, y0 + 0.03, S["plinth"]), visual_mat=birch, b=b)
    box_geom(body, "shelf_back", (x0, y1 - 0.006, 0.0, x1, y1, H), visual_mat=birch, b=b, uv_scale=(2, 2))
    for i, z in enumerate(S["shelves_z"]):
        box_geom(body, f"shelf_board{i}", (x0 + t, y0, z, x1 - t, y1 - 0.006, z + t), visual_mat=birch, b=b,
                 uv_scale=(2, 2))
    dx = S["divider_x"]
    box_geom(body, "shelf_divider", (dx - t / 2, y0, S["shelves_z"][0] + t, dx + t / 2, y1 - 0.006, H - t),
             visual_mat=birch, b=b, uv_scale=(2, 2))
    # the books already on the shelves (rows of spines, static) and a few things on top
    rows = (("a", (x0 + t, S["shelves_z"][0] + t, dx - t / 2 - 0.05), 0.255),     # bottom, west bay
            ("b", (dx + t / 2, S["shelves_z"][0] + t, x1 - t), 0.27),             # bottom, east bay
            ("a", (dx + t / 2 + 0.12, S["shelves_z"][1] + t, x1 - t), 0.25),      # middle, east bay
            ("a", (x0 + t, S["shelves_z"][1] + t, x0 + 0.42), 0.24),              # middle, west bay
            ("b", (x0 + t, S["shelves_z"][2] + t, RM.BOOK_TARGET["x0"] - 0.016), 0.23),  # top, west bay (the target's)
            ("b", (dx + t / 2 + 0.25, S["shelves_z"][2] + t, x1 - t), 0.23))      # top, east bay
    for i, (pal, (xa, z, xb), h) in enumerate(rows):
        mat = b.material(f"spines_{pal}_mat", tex=f"spines_{pal}")
        depth = 0.20
        mesh = P.scale_uv(P.rounded_box([(xb - xa) / 2, depth / 2, h / 2], 0.002, 2), 1 / (xb - xa), 1 / h)
        b.visual(body, f"book_row{i}", mesh, mat, pos=((xa + xb) / 2, y0 + 0.03 + depth / 2, z + h / 2))
        box_geom(body, f"book_row{i}_col", (xa, y0 + 0.03, z, xb, y0 + 0.03 + depth, z + h), b=b)
    # a bookend at the end of the target bay's books
    bk = b.material("bookend", (0.12, 0.12, 0.13, 1))
    xe = RM.BOOK_TARGET["x0"] - 0.016
    box_geom(body, "bookend", (xe, y0 + 0.06, S["shelves_z"][2] + t, xe + 0.004, y0 + 0.20, S["shelves_z"][2] + t + 0.16),
             visual_mat=bk, b=b)
    vase = b.material("vase", (0.80, 0.48, 0.34, 1))
    b.visual(body, "vase", P.lathe([(0.0, 0.0), (0.045, 0.0), (0.065, 0.08), (0.05, 0.18), (0.032, 0.22),
                                    (0.036, 0.24), (0.0, 0.24)], segments=32), vase, pos=(x0 + 0.20, y0 + 0.17, H))
    frame = b.material("photo_frame", (0.10, 0.09, 0.08, 1))
    b.visual(body, "photo", P.rounded_box([0.08, 0.008, 0.11], 0.003, 2), frame, pos=(x1 - 0.25, y0 + 0.22, H + 0.11),
             quat=_quat(np.array([[1, 0, 0], [0, math.cos(-0.25), -math.sin(-0.25)], [0, math.sin(-0.25),
                                                                                     math.cos(-0.25)]])))
    box_geom(body, "shelf_top_items", (x0 + 0.13, y0 + 0.10, H, x0 + 0.27, y0 + 0.24, H + 0.24), b=b)


# ================================================================================================ the build


def add_lights(s):
    wb = s.worldbody
    wb.add_light(name="sun", type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL, pos=[9.0, 3.0, 4.0], dir=list(SUN_DIR),
                 castshadow=True, diffuse=[0.6, 0.57, 0.52], specular=[0.1, 0.1, 0.1])
    for i, (lx, ly) in enumerate(PANELS):
        wb.add_light(name=f"panel{i}", type=mujoco.mjtLightType.mjLIGHT_SPOT, pos=[lx, ly, RM.WALL_H - 0.03],
                     dir=[0.0, 0.0, -1.0], cutoff=75.0, exponent=1.0, castshadow=True, diffuse=[0.25, 0.24, 0.22],
                     specular=[0.03, 0.03, 0.03], attenuation=[1.0, 0.0, 0.0])
    for name, (bx, by, bz), depth in BOUNCE:
        wb.add_light(name=f"bounce_{name}", type=mujoco.mjtLightType.mjLIGHT_SPOT, pos=[bx, by, bz - depth],
                     dir=[0.0, 0.0, 1.0], cutoff=50.0, exponent=1.0, castshadow=False, diffuse=[0.0, 0.0, 0.0],
                     specular=[0.0, 0.0, 0.0], attenuation=[1.0, 0.0, 0.0])
    # the strip light under the wall cabinets (warm, soft)
    wb.add_light(name="undercabinet", type=mujoco.mjtLightType.mjLIGHT_SPOT, pos=[0.20, 2.25, RM.UPPER["z0"] - 0.02],
                 dir=[0.15, 0.0, -1.0], cutoff=80.0, exponent=1.0, castshadow=False, diffuse=[0.3, 0.28, 0.24],
                 specular=[0.0, 0.0, 0.0], attenuation=[1.0, 0.0, 0.0])
    # the pendant's bulb over the dining table (warm, a downward cone through the shade's opening)
    (cx, cy) = RM.DINING["center"]
    wb.add_light(name="pendant", type=mujoco.mjtLightType.mjLIGHT_SPOT, pos=[cx, cy, RM.WALL_H - 0.80],
                 dir=[0.0, 0.0, -1.0], cutoff=70.0, exponent=1.0, castshadow=True, diffuse=[0.2, 0.18, 0.15],
                 specular=[0.0, 0.0, 0.0], attenuation=[1.0, 0.0, 0.0])


def build(layout: dict, *, offscreen=(960, 720), decor: bool = True):
    """The cell spec for one episode layout (layout.draw): returns (spec, meta). meta: {"features": {body: [grasp
    features]}, "graspable": [body names], "bins": {slot: kind}, "grease": [(i, j, x, y)], "crumbs": n, ...}."""
    s = mujoco.MjSpec()
    s.modelname = "house_cell"
    s.compiler.degree = False          # every angle of the cell's own joints is in radians (the G1 keeps its own)
    o = s.option
    o.timestep = 0.002
    o.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    o.cone = mujoco.mjtCone.mjCONE_PYRAMIDAL
    o.iterations = 60
    o.ls_iterations = 30
    # multi-point convex contacts on (the furniture cell turns them off; here the plates, glasses, cans and the
    # bottle are cylinders that must rest flat on the tables)
    s.memory = 256 * 1024 * 1024
    s.visual.global_.offwidth, s.visual.global_.offheight = offscreen
    s.visual.quality.shadowsize = 2048
    s.visual.headlight.ambient = [0.30, 0.30, 0.30]
    s.visual.headlight.diffuse = [0.35, 0.35, 0.35]
    s.visual.headlight.specular = [0.0, 0.0, 0.0]
    s.visual.map.znear = 0.01
    numeric, text = filament_view_keys()
    for key, value in sorted(numeric.items()):
        s.add_numeric(name=key, data=[value])
    for key, value in sorted(text.items()):
        s.add_text(name=key, data=value)
    b = O.Builder(s)
    add_lights(s)
    add_shell(s, b)
    add_kitchen(s, b)
    wb = s.worldbody
    meta = {"features": {}, "graspable": [], "bins": {}, "fixture_bodies": {}}
    bin_recs = []
    for slot, kind in enumerate(layout["bins"]):
        rec = O.add_bin(b, wb, slot, RM.BIN_SLOTS_Y[slot], kind)
        bin_recs.append(rec)
        meta["bins"][slot] = kind
        meta["features"][f"bin{slot}_lid"] = rec["features"]
        meta["graspable"].append(f"bin{slot}_lid")
    fau = O.add_faucet(b, wb)
    meta["features"]["faucet_lever"] = fau["features"]
    meta["graspable"].append("faucet_lever")
    soap = O.add_soap(b, wb)
    meta["features"]["soap_pump"] = soap["features"]
    meta["graspable"].append("soap_pump")
    add_tool_rail(s, b)
    add_dining(s, b)
    add_pendant(s, b)
    add_coffee(s, b)
    add_bookshelf(s, b)
    # the loose objects (free bodies at the layout's poses)
    for name, (kind, params) in O.LOOSE.items():
        pos, quat = layout["objects"][name]
        body = wb.add_body(name=name, pos=list(pos), quat=list(quat))
        body.add_freejoint(name=f"{name}_free")
        feats = O.BUILDERS[kind](b, body, name, params)
        meta["features"][name] = feats
        meta["graspable"].append(name)
    # crumbs: drawn-only mocap bodies (the cell moves them; cleaning.Crumbs)
    rng = np.random.default_rng(int(layout.get("crumb_look_seed", 0)))
    for i, p in enumerate(layout["crumbs"]):
        cb = wb.add_body(name=f"crumb{i}", pos=list(p), mocap=True)
        r = float(layout["crumb_r"][i])
        col = CRUMB_COLOURS[int(rng.integers(len(CRUMB_COLOURS)))]
        sh = rng.uniform(0.6, 1.0, size=3)
        cb.add_geom(name=f"crumb{i}_vis", type=G.mjGEOM_ELLIPSOID, size=[r * sh[0], r * sh[1], r * 0.55 * sh[2]],
                    rgba=list(col) + [1.0], contype=0, conaffinity=0, group=1, mass=0.0,
                    quat=_quat_z(rng.uniform(0, math.pi)))
    meta["crumbs"] = len(layout["crumbs"])
    # the grease film: drawn-only cells on the dining table's top (cleaning.Grease colours them)
    gr = layout["grease"]
    gb = wb.add_body(name="grease")
    cell = gr["cell"]
    # square cells edge to edge (one film, no overlaps)
    for k, (i, j, x, y) in enumerate(gr["cells"]):
        gb.add_geom(name=f"grease{k}", type=G.mjGEOM_BOX, size=[cell / 2, cell / 2, 0.0002],
                    pos=[x, y, RM.DINING["top_z"] + 0.0004], rgba=list(GREASE_RGB) + [0.0], contype=0,
                    conaffinity=0, group=1, mass=0.0)
    meta["grease"] = gr
    # the water stream under the faucet's outlet (drawn only; shown while the water runs)
    ox, oy, oz = RM.FAUCET["outlet"]
    zb = RM.SINK["bottom_z"]
    wb.add_geom(name="water", type=G.mjGEOM_CYLINDER, size=[0.0065, (oz - zb) / 2, 0], pos=[ox, oy, (oz + zb) / 2],
                rgba=[0.80, 0.88, 0.95, 0.0], contype=0, conaffinity=0, group=1, mass=0.0)
    # the robot
    frame = wb.add_frame(name="robot_frame", pos=[0.0, 0.0, 0.0])
    s.attach(g1_spec(ROBOT_NAME), prefix=f"{ROBOT_NAME}/", frame=frame)
    # ceiling cameras
    for cname, c in RM.CEILING_CAMERAS.items():
        wb.add_camera(name=cname, pos=list(c["pos"]), quat=_look_at_quat(c["pos"], c["target"]), fovy=c["fovy"])
    # inactive weld pool: every hand x every graspable body (grips), each tool x the world (its clip)
    weld = dict(type=mujoco.mjtEq.mjEQ_WELD, objtype=mujoco.mjtObj.mjOBJ_BODY, active=False,
                solref=[0.01, 1.0], solimp=[0.95, 0.99, 0.001, 0.5, 2.0],
                data=[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1.0])
    # a fixture's moving part (a lid's tab, the faucet's lever, the soap pump's head) is held by a point grip (a
    # connect constraint: the part turns about its hinge in the hand), a loose object rigidly (a weld)
    connect = dict(weld, type=mujoco.mjtEq.mjEQ_CONNECT, data=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    for side in ("left", "right"):
        for name in meta["graspable"]:
            kw = connect if name in FIXTURE_PARTS else weld
            s.add_equality(name=f"grip_{side}_{name}", name1=f"{ROBOT_NAME}/{side}_wrist_yaw_link", name2=name, **kw)
    for tool in RM.CLIPS:
        s.add_equality(name=f"clip_{tool}", name1="world", name2=tool, **weld)
    # the bins' insides (drawn only), after every other body of the cell (objects.add_bin_inside)
    for slot, rec in enumerate(bin_recs):
        O.add_bin_inside(b, wb, slot, rec)
    if decor:
        add_decor(s)
    return s, meta
