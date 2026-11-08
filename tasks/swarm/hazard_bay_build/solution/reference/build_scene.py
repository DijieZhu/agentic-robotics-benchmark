"""The reference build of the hazard bay: the oracle's artifact.

    python3 build_scene.py --out /home/user/submission/scene.usda [--truth truth.json]

Writes one USD stage that follows the agent's `~/given/interface.md`: the bay (floor, calibration mat, painted pads
and parking bays, two tables with six objects, a parts crate), the steam line (a pipe along the
back wall, a quarter-turn valve on a branch, a pressure gauge, a leaking flange with its plume), the fire pan, the water
tank with the rigged bucket, and the fleet the reference designed for it: three hexacopters (each with an inverted
SO-101 arm, a thermal and an RGB camera, a cargo hook with a physical rope to the bucket) and a Jackal with a Kinova Gen3 and a
mast camera. Every mechanism and every hazard rule is physics plus OmniGraph action graphs (no script nodes). With
--truth it also writes the authored layout (author use only).

The grader's real bay comes from the same code (author use only, once; the outputs live in tests/hidden/):

    python3 build_scene.py --bay-only --out tests/hidden/bay_true.usda --bay-facts tests/hidden/bay_facts.json

--bay-only leaves out the backdrop and the lights, the fleet and its ropes, the hazard logic and the two Flow
volumes: what is left is the bay itself (its physics scene, floor, markings, tables and parts, crate, steam line and
valve, gauge, plume box, pan, tank and the bucket with its rings), the layer the bay mission's stage is built on; its
floor collider is a 0.5 m slab instead of the twin's 2 cm box (the floor's top, z = 0, is the same: see below).
--bay-facts writes the numbers that stage is built from (pads, parking bay, table tops, pan, valve, rings, rope).
"""

import argparse
import json
import math
import os

from isaacsim import SimulationApp

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--truth")
ap.add_argument("--bay-only", action="store_true", help="the real bay without fleet, logic, backdrop, lights or Flow volumes")
ap.add_argument("--bay-facts", help="write the real bay's facts (JSON) here")
args = ap.parse_args()
BAY_ONLY = bool(args.bay_only)

app = SimulationApp({"headless": True})
import omni.kit.app  # noqa: E402

omni.kit.app.get_app().get_extension_manager().set_extension_enabled_immediate("isaacsim.core.nodes", True)
import omni.graph.core as og  # noqa: E402
import omni.usd  # noqa: E402
from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade  # noqa: E402

ASSETS = "/opt/assets/isaac/Isaac"
ARB = "/opt/assets/arb"
FAC, FLEET = "/World/Facility", "/World/Fleet"

# ---------------------------------------------------------------------------------------------------------------------
# layout (world frame: z up, metres; the calibration mat on the floor fixes it)
# ---------------------------------------------------------------------------------------------------------------------
MAT = dict(center=(0.0, -0.60), squares=(7, 5), square=0.10)
TABLES = {  # top centre xy, top size (x, y), top height, yaw [deg], top thickness
    "Table_A": dict(center=(-1.30, 0.70), size=(1.20, 0.60), height=0.75, yaw=0.0, thick=0.04),
    "Table_B": dict(center=(1.45, 0.55), size=(0.80, 0.60), height=0.45, yaw=20.0, thick=0.03),
}
CRATE = dict(center=(0.10, 1.50), size=(0.60, 0.40, 0.32), wall=0.02, yaw=0.0)  # the parts crate beside the station: an open plastic crate (600 x 400 x 320 mm)
ITEMS = {  # name: (asset below Props/YCB, table, xy, yaw [deg], mass [kg]); stood upright (their meshes' up is -y)
    "cracker_box": ("Axis_Aligned_Physics/003_cracker_box.usd", "Table_A", (-1.62, 0.62), 10.0, 0.411),
    "mustard_bottle": ("Axis_Aligned_Physics/006_mustard_bottle.usd", "Table_A", (-1.18, 0.80), -30.0, 0.603),
    "sugar_box": ("Axis_Aligned_Physics/004_sugar_box.usd", "Table_A", (-0.90, 0.58), 75.0, 0.514),
    "soup_can": ("Axis_Aligned_Physics/005_tomato_soup_can.usd", "Table_B", (1.30, 0.50), 0.0, 0.349),
    "power_drill": ("Axis_Aligned/035_power_drill.usd", "Table_B", (1.62, 0.66), 40.0, 0.895),
    "mug": ("Axis_Aligned/025_mug.usd", "Table_B", (1.42, 0.34), -60.0, 0.118),
}
PADS = {"Pad_1": (0.0, -1.50), "Pad_2": (-0.95, -3.15), "Pad_3": (0.95, -3.15)}  # launch pads (0.7 m squares)
PAD_SIZE = 0.70
UGV_BAY = dict(center=(-2.80, 2.20), size=(1.20, 0.90))  # the ground vehicle's parking bay (heading +y)
VALVE_SQUARE = dict(center=(-1.20, 2.45), size=(0.80, 0.80))  # where the vehicle stands to work the valve
BUCKET = dict(xy=(0.0, -2.60), r=0.12, h=0.25, mass=1.20, water=10.0, freeboard=0.03, bails=(90.0, 210.0, 330.0))  # a 10 L bucket, full to 3 cm below the rim
TETHER_L = 1.50
ROPE = dict(length=1.50, links=12, radius=0.005, kg_per_m=0.30)  # 10 mm wire rope: hook-to-ring length [m], link count, radius [m], mass per metre
TANK = dict(center=(-1.90, -3.10), size=(0.80, 0.60, 0.70), wall=0.02)
PIPE = dict(y=3.35, z=1.20, r=0.04, x0=-3.6, x1=3.6)
VALVE = dict(x=-1.20, stem=(-1.20, 3.20, 0.95), lever=0.26, close_deg=90.0, steam_stop_s=3.0)  # a quarter-turn valve
GAUGE = dict(center=(-0.60, 3.22, 1.20), r=0.05)
LEAK = dict(x=2.40)
PLUME = dict(lo=(1.80, 2.20, 1.10), hi=(3.00, 3.50, 3.20))  # the hazard volume (the photographs show only its core; facts.md gives this box)
FIRE = dict(center=(2.60, 1.90), r=0.30, rim=0.08, hazard_r=2.0, pour_s=1.0, pour_tilt_deg=15.0, pour_r=0.6, pour_zmax=2.5)
APPROACH = dict(pos=(2.60, -0.40, 2.50), yaw=90.0)  # the water drop's approach hover (heading +y)
FLOOR_FRICTION = 0.5  # the floor's static and dynamic friction (facts.md; the PhysX default the truth was measured with)
RING_OF_PAD = {"Pad_1": 0, "Pad_2": 1, "Pad_3": 2}  # which of the bucket's rings (BUCKET["bails"]) faces each pad


# the fleet the reference designed (the catalog the agent gets lists the same models)
SENSORS = {"CAM-RGB-M": dict(focal=8.0, hap=11.26, vap=7.04, mass=0.065), "THERM-C": dict(focal=9.0, hap=7.68, vap=6.14, mass=0.095),
           "THERM-A": dict(focal=1.7, hap=1.92, vap=1.44, mass=0.015), "CAM-RGB-W": dict(focal=4.0, hap=11.26, vap=7.04, mass=0.06),
           "LIDAR-3D": dict(mass=0.35)}
UAV_SENSORS = ("THERM-C", "CAM-RGB-M", "THERM-A", "CAM-RGB-W", "LIDAR-3D")
AIRFRAME = dict(model="arb_hex_800", file=f"{ARB}/airframes/arb_hex_800.usda", empty=4.10, hull=(0.26, 0.26, 0.10), gear=0.42,
                hook=(0.0, -0.104, -0.08))
UGV_BASE = dict(model="jackal", file=f"{ASSETS}/Robots/Clearpath/Jackal/jackal.usd")
UGV_ARM = dict(model="kinova_gen3", file=f"{ASSETS}/Robots/Kinova/Gen3/gen3n7_instanceable.usd", base="base_link", root_joint="root_joint")

ctx = omni.usd.get_context()
ctx.new_stage()
st = ctx.get_stage()
UsdGeom.SetStageUpAxis(st, UsdGeom.Tokens.z)
UsdGeom.SetStageMetersPerUnit(st, 1.0)
UsdPhysics.SetStageKilogramsPerUnit(st, 1.0)
world = UsdGeom.Xform.Define(st, "/World")
st.SetDefaultPrim(world.GetPrim())
scene = UsdPhysics.Scene.Define(st, "/World/PhysicsScene")
scene.CreateGravityDirectionAttr(Gf.Vec3f(0, 0, -1))
scene.CreateGravityMagnitudeAttr(9.81)
PhysxSchema.PhysxSceneAPI.Apply(scene.GetPrim()).CreateTimeStepsPerSecondAttr(120)
truth = {"tables": {}, "bin": {}, "items": {}, "fleet": {}}
# the group prims as proper Xforms (a prim created only as an ancestor has no type: it is neither transformable nor
# imageable, so moving or hiding it does nothing)
for _g in (FAC, FLEET, f"{FAC}/Tables", f"{FAC}/Items", f"{FAC}/Pads", f"{FAC}/SteamLine", f"{FAC}/SteamLine/Pipe",
           f"{FAC}/SteamLine/Gauge", f"{FAC}/SteamLine/Leak", f"{FAC}/FirePan", f"{FAC}/Tank", f"{FAC}/CalibMat", "/World/Lights",
           "/World/Looks"):
    if BAY_ONLY and _g in (FLEET, "/World/Lights"):
        continue
    UsdGeom.Xform.Define(st, _g) if not _g.endswith("Looks") else UsdGeom.Scope.Define(st, _g)


def rotz(deg):
    return Gf.Rotation(Gf.Vec3d(0, 0, 1), deg)


def xform(path, pos=(0, 0, 0), yaw=0.0, quat=None):
    x = UsdGeom.Xform.Define(st, path)
    x.AddTranslateOp().Set(Gf.Vec3d(*pos))
    q = quat if quat is not None else Gf.Quatf(rotz(yaw).GetQuat())
    x.AddOrientOp().Set(Gf.Quatf(q))
    return x


def material(name, rgb, rough=0.6, metal=0.0, opacity=1.0, emissive=None):
    path = f"/World/Looks/{name}"
    m = UsdShade.Material.Define(st, path)
    s = UsdShade.Shader.Define(st, path + "/S")
    s.CreateIdAttr("UsdPreviewSurface")
    s.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*rgb))
    s.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(rough)
    s.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metal)
    if opacity < 1.0:
        s.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(opacity)
    if emissive is not None:
        s.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*emissive))
    m.CreateSurfaceOutput().ConnectToSource(s.ConnectableAPI(), "surface")
    return m


M = {k: material(k, *v) for k, v in {
    "floor": ((0.42, 0.42, 0.44), 0.85), "wood": ((0.55, 0.40, 0.26), 0.55), "steel": ((0.60, 0.62, 0.65), 0.35, 0.7),
    "bin": ((0.20, 0.42, 0.30),), "lid": ((0.16, 0.34, 0.24),), "dark": ((0.12, 0.12, 0.13),),
    "metal": ((0.75, 0.75, 0.78), 0.3, 0.8), "red": ((0.85, 0.12, 0.10), 0.4), "white": ((0.92, 0.92, 0.92), 0.7),
    "black": ((0.03, 0.03, 0.03), 0.7), "frame": ((0.18, 0.19, 0.21), 0.5, 0.3), "orange": ((0.95, 0.45, 0.08), 0.5),
    "yellow": ((0.95, 0.80, 0.10), 0.6), "pipe": ((0.55, 0.57, 0.60), 0.4, 0.6), "valve": ((0.70, 0.10, 0.08), 0.5, 0.2),
    "wheel": ((0.15, 0.15, 0.60), 0.5, 0.3), "water": ((0.15, 0.35, 0.60), 0.2, 0.0), "tank": ((0.30, 0.32, 0.35), 0.5, 0.5),
    "bucket": ((0.80, 0.20, 0.15), 0.5, 0.1), "pan": ((0.20, 0.20, 0.20), 0.6, 0.4),
    "crate": ((0.16, 0.30, 0.62), 0.55), "crate_dark": ((0.10, 0.18, 0.40), 0.6),
    "paint_y": ((0.95, 0.85, 0.15), 0.8), "paint_g": ((0.20, 0.70, 0.30), 0.8), "paint_w": ((0.95, 0.95, 0.95), 0.8)}.items()}
M["steam"] = material("steam", (0.93, 0.94, 0.96), 0.95, 0.0, opacity=0.22)


def box(path, center, size, mat, col=True, yaw=0.0):
    g = UsdGeom.Cube.Define(st, path)
    g.CreateSizeAttr(1.0)
    g.AddTranslateOp().Set(Gf.Vec3d(*center))
    if yaw:
        g.AddOrientOp().Set(Gf.Quatf(rotz(yaw).GetQuat()))
    g.AddScaleOp().Set(Gf.Vec3f(*size))
    if col:
        UsdPhysics.CollisionAPI.Apply(g.GetPrim())
    UsdShade.MaterialBindingAPI.Apply(g.GetPrim()).Bind(mat)
    return g


def box_lohi(path, lo, hi, mat, col=True):
    return box(path, [(a + b) / 2 for a, b in zip(lo, hi)], [b - a for a, b in zip(lo, hi)], mat, col)


def cyl(path, center, r, h, axis, mat, col=True):
    g = UsdGeom.Cylinder.Define(st, path)
    g.CreateRadiusAttr(r)
    g.CreateHeightAttr(h)
    g.CreateAxisAttr(axis)
    g.AddTranslateOp().Set(Gf.Vec3d(*center))
    if col:
        UsdPhysics.CollisionAPI.Apply(g.GetPrim())
    UsdShade.MaterialBindingAPI.Apply(g.GetPrim()).Bind(mat)
    return g


def sphere(path, center, r, mat):
    g = UsdGeom.Sphere.Define(st, path)
    g.CreateRadiusAttr(r)
    g.AddTranslateOp().Set(Gf.Vec3d(*center))
    UsdShade.MaterialBindingAPI.Apply(g.GetPrim()).Bind(mat)
    return g


def cone(path, center, r, h, mat):
    g = UsdGeom.Cone.Define(st, path)
    g.CreateRadiusAttr(r)
    g.CreateHeightAttr(h)
    g.CreateAxisAttr("Z")
    g.AddTranslateOp().Set(Gf.Vec3d(*center))
    UsdShade.MaterialBindingAPI.Apply(g.GetPrim()).Bind(mat)
    return g


def paint(path, center, size, mat, yaw=0.0):
    """A painted floor marking: a 1 mm slab, no collision."""
    return box(path, (center[0], center[1], 0.0006), (size[0], size[1], 0.001), mat, col=False, yaw=yaw)


def frame_marking(root, center, size, mat, line=0.05, yaw=0.0, arrow=True):
    """A painted rectangle outline (four strokes) with an arrow along its local +y."""
    w, d = size
    paint(f"{root}/n", (center[0], center[1] + d / 2, 0), (w, line), mat)
    paint(f"{root}/s", (center[0], center[1] - d / 2, 0), (w, line), mat)
    paint(f"{root}/e", (center[0] + w / 2, center[1], 0), (line, d), mat)
    paint(f"{root}/w", (center[0] - w / 2, center[1], 0), (line, d), mat)
    if arrow:
        paint(f"{root}/arrow", (center[0], center[1], 0), (line, d * 0.5), mat)
        paint(f"{root}/head_l", (center[0] - 0.06, center[1] + d * 0.25 - 0.06, 0), (line, 0.17), mat, yaw=45.0)
        paint(f"{root}/head_r", (center[0] + 0.06, center[1] + d * 0.25 - 0.06, 0), (line, 0.17), mat, yaw=-45.0)


def state_attr(prim, name, value):
    prim.CreateAttribute(name, Sdf.ValueTypeNames.Bool).Set(value)


# ---------------------------------------------------------------------------------------------------------------------
# backdrop, lights, floor, markings
# ---------------------------------------------------------------------------------------------------------------------
if not BAY_ONLY:  # (the plant's hall: visual only, it authors no physics and has no geometry inside the bay's volume)
    bd = UsdGeom.Xform.Define(st, "/World/Backdrop")
    bd.GetPrim().GetReferences().AddReference(f"{ARB}/hall/bay_hall.usda")
    dome = UsdLux.DomeLight.Define(st, "/World/Lights/Dome")
    dome.CreateIntensityAttr(300.0)
    sun = UsdLux.DistantLight.Define(st, "/World/Lights/Sun")  # (an area light in the hall shows up in the photographs)
    sun.CreateIntensityAttr(1800.0)
    sun.CreateAngleAttr(4.0)
    UsdGeom.Xformable(sun).AddOrientOp().Set(Gf.Quatf((Gf.Rotation(Gf.Vec3d(0, 0, 1), 35.0) * Gf.Rotation(Gf.Vec3d(1, 0, 0), -50.0)).GetQuat()))

# the floor: a collider under the hall's own floor, its top at z = 0. In the real bay a 0.5 m slab: on the 2 cm
# box of the twin (the truth was measured with it) a loaded wheel of a heavy vehicle can sink 8 mm into it, near the
# box's middle, where the contact cannot tell which face it is under; the vehicle then shakes for the rest of the run.
FLOOR_T = 0.5 if BAY_ONLY else 0.02
floor = box(f"{FAC}/Floor", (0, 0, -FLOOR_T / 2), (14.0, 14.0, FLOOR_T), M["floor"])
UsdGeom.Imageable(floor).MakeInvisible()
# the floor's surface for the simulation (facts.md): sealed concrete, friction 0.5 static and dynamic, no restitution
floor_mat = UsdShade.Material.Define(st, "/World/Looks/FloorSurface")
fm = UsdPhysics.MaterialAPI.Apply(floor_mat.GetPrim())
fm.CreateStaticFrictionAttr(FLOOR_FRICTION)
fm.CreateDynamicFrictionAttr(FLOOR_FRICTION)
fm.CreateRestitutionAttr(0.0)
UsdShade.MaterialBindingAPI.Apply(floor.GetPrim()).Bind(floor_mat, UsdShade.Tokens.weakerThanDescendants, "physics")
cx, cy = MAT["center"]
nx, ny = MAT["squares"]
s = MAT["square"]
for i in range(nx):
    for j in range(ny):
        paint(f"{FAC}/CalibMat/sq_{i}_{j}", (cx + (i - (nx - 1) / 2) * s, cy + (j - (ny - 1) / 2) * s), (s, s),
              M["white" if (i + j) % 2 == 0 else "black"])
truth["calib_mat"] = dict(center=list(MAT["center"]), squares=list(MAT["squares"]), square=s)
for name, (px, py) in PADS.items():
    frame_marking(f"{FAC}/Pads/{name}", (px, py), (PAD_SIZE, PAD_SIZE), M["paint_y"])
    paint(f"{FAC}/Pads/{name}/h", (px, py), (0.20, 0.04), M["paint_y"])
    paint(f"{FAC}/Pads/{name}/h2", (px, py), (0.04, 0.20), M["paint_y"])
frame_marking(f"{FAC}/Pads/UGV_Bay", UGV_BAY["center"], UGV_BAY["size"], M["paint_g"])
frame_marking(f"{FAC}/Pads/Valve_Square", VALVE_SQUARE["center"], VALVE_SQUARE["size"], M["paint_w"])
truth["pads"] = dict(PADS, UGV_Bay=UGV_BAY, Valve_Square=VALVE_SQUARE, approach=APPROACH)

# ---------------------------------------------------------------------------------------------------------------------
# tables
# ---------------------------------------------------------------------------------------------------------------------
for name, t in TABLES.items():
    x0, y0 = t["center"]
    xform(f"{FAC}/Tables/{name}", (x0, y0, 0.0), t["yaw"])
    w, d, h, th = t["size"][0], t["size"][1], t["height"], t["thick"]
    box(f"{FAC}/Tables/{name}/top", (0, 0, h - th / 2), (w, d, th), M["wood"])
    for k, (sx, sy) in enumerate(((1, 1), (1, -1), (-1, 1), (-1, -1))):
        box(f"{FAC}/Tables/{name}/leg_{k}", (sx * (w / 2 - 0.05), sy * (d / 2 - 0.05), (h - th) / 2), (0.05, 0.05, h - th),
            M["steel"])
    truth["tables"][name] = dict(center=[x0, y0], size=[w, d], height=h, yaw=t["yaw"])

# ---------------------------------------------------------------------------------------------------------------------
# the parts crate: an open plastic crate (a floor and four walls, static colliders) standing beside the station
# ---------------------------------------------------------------------------------------------------------------------
C = f"{FAC}/Crate"
cx, cy = CRATE["center"]
cw, cd, ch = CRATE["size"]
ct = CRATE["wall"]
xform(C, (cx, cy, 0.0), CRATE["yaw"])
box(f"{C}/floor", (0, 0, ct / 2), (cw, cd, ct), M["crate"])
box(f"{C}/wall_n", (0, cd / 2 - ct / 2, ch / 2), (cw, ct, ch), M["crate"])
box(f"{C}/wall_s", (0, -cd / 2 + ct / 2, ch / 2), (cw, ct, ch), M["crate"])
box(f"{C}/wall_e", (cw / 2 - ct / 2, 0, ch / 2), (ct, cd, ch), M["crate"])
box(f"{C}/wall_w", (-cw / 2 + ct / 2, 0, ch / 2), (ct, cd, ch), M["crate"])
for k in range(3):  # the handle slots and ribs that a plant's crate has
    box(f"{C}/rib_n_{k}", ((k - 1) * 0.18, cd / 2 + 0.006, ch * 0.55), (0.03, 0.012, ch * 0.7), M["crate_dark"])
    box(f"{C}/rib_s_{k}", ((k - 1) * 0.18, -cd / 2 - 0.006, ch * 0.55), (0.03, 0.012, ch * 0.7), M["crate_dark"])
truth["crate"] = dict(center=list(CRATE["center"]), size=list(CRATE["size"]), wall=CRATE["wall"], yaw=CRATE["yaw"])

# --- OmniGraph helpers (the bay's hazard logic) ---
K = og.Controller.Keys


class Graph:
    def __init__(self, path, variables):
        self.path, self.variables = path, variables
        self.nodes, self.vals, self.conns = [], [], []

    def node(self, nid, typ, **inputs):
        self.nodes.append((nid, typ))
        for k, v in inputs.items():
            self.vals.append((f"{nid}.inputs:{k}", v))
        return nid

    def const(self, nid, v, typ="Double"):
        return self.node(nid, f"omni.graph.nodes.Constant{typ}", value=v) + ".inputs:value"

    def link(self, src, dst):
        self.conns.append((src, dst))

    def op(self, nid, typ, a, b, **kw):
        self.node(nid, typ, **kw)
        self.link(a, f"{nid}.inputs:a")
        self.link(b, f"{nid}.inputs:b")
        return nid

    def cmp(self, nid, a, b, operation):
        return self.op(nid, "omni.graph.nodes.Compare", a, b, operation=operation) + ".outputs:result"

    def sel(self, nid, cond, yes, no):
        self.node(nid, "omni.graph.nodes.SelectIf")
        self.link(cond, f"{nid}.inputs:condition")
        self.link(yes, f"{nid}.inputs:ifTrue")
        self.link(no, f"{nid}.inputs:ifFalse")
        return f"{nid}.outputs:result"

    def band(self, nid, a, b):
        return self.op(nid, "omni.graph.nodes.BooleanAnd", a, b) + ".outputs:result"

    def bor(self, nid, a, b):
        return self.op(nid, "omni.graph.nodes.BooleanOr", a, b) + ".outputs:result"

    def bvar(self, name):
        """A bool variable as a typed bool (a variable's output is untyped until the graph runs; Compare takes any)."""
        return self.cmp(f"{name}_isb", self.var(name), self.const(f"{name}_ct", True, "Bool"), "==")

    def add(self, nid, a, b):
        return self.op(nid, "omni.graph.nodes.Add", a, b) + ".outputs:sum"

    def sub(self, nid, a, b):
        return self.op(nid, "omni.graph.nodes.Subtract", a, b) + ".outputs:difference"

    def mul(self, nid, a, b):
        return self.op(nid, "omni.graph.nodes.Multiply", a, b) + ".outputs:product"

    def var(self, name):
        if not any(nid == f"{name}_r" for nid, _ in self.nodes):
            self.node(f"{name}_r", "omni.graph.core.ReadVariable", variableName=name)
        return f"{name}_r.outputs:value"

    def world_pos(self, nid, prim_path):
        """World translation (x, y, z as doubles) of a prim, read every step."""
        self.node(f"{nid}_m", "omni.graph.nodes.GetPrimLocalToWorldTransform", primPath=prim_path, usePath=True)
        self.node(f"{nid}_t", "omni.graph.nodes.GetMatrix4Translation")
        self.link(f"{nid}_m.outputs:localToWorldTransform", f"{nid}_t.inputs:matrix")
        self.node(f"{nid}_b", "omni.graph.nodes.BreakVector3")
        self.link(f"{nid}_t.outputs:translation", f"{nid}_b.inputs:tuple")
        return f"{nid}_b.outputs:x", f"{nid}_b.outputs:y", f"{nid}_b.outputs:z"

    def world_up_z(self, nid, prim_path):
        """z component of a prim's own +z axis in the world: 1 upright, 0 on its side."""
        self.node(f"{nid}_m", "omni.graph.nodes.GetPrimLocalToWorldTransform", primPath=prim_path, usePath=True)
        self.node(f"{nid}_q", "omni.graph.nodes.GetMatrix4Quaternion")
        self.link(f"{nid}_m.outputs:localToWorldTransform", f"{nid}_q.inputs:matrix")
        self.node(f"{nid}_r", "omni.graph.nodes.RotateVector")  # (its vector input is untyped: a constant node feeds it)
        self.link(f"{nid}_q.outputs:quaternion", f"{nid}_r.inputs:rotation")
        self.link(self.const(f"{nid}_up", (0.0, 0.0, 1.0), "Double3"), f"{nid}_r.inputs:vector")
        self.node(f"{nid}_b", "omni.graph.nodes.BreakVector3")
        self.link(f"{nid}_r.outputs:result", f"{nid}_b.inputs:tuple")
        return f"{nid}_b.outputs:z"

    def in_box(self, nid, xyz, lo, hi):
        conds = []
        for k, (v, a, b) in enumerate(zip(xyz, lo, hi)):
            conds.append(self.cmp(f"{nid}_lo{k}", v, self.const(f"{nid}_clo{k}", a), ">="))
            conds.append(self.cmp(f"{nid}_hi{k}", v, self.const(f"{nid}_chi{k}", b), "<="))
        expr = conds[0]
        for k, c in enumerate(conds[1:]):
            expr = self.band(f"{nid}_and{k}", expr, c)
        return expr

    def dist2_xy(self, nid, xy, center):
        dx = self.sub(f"{nid}_dx", xy[0], self.const(f"{nid}_cx", center[0]))
        dy = self.sub(f"{nid}_dy", xy[1], self.const(f"{nid}_cy", center[1]))
        return self.add(f"{nid}_d2", self.mul(f"{nid}_dx2", dx, dx), self.mul(f"{nid}_dy2", dy, dy))

    def write_attr(self, nid, prev, prim_path, name, expr):
        self.node(nid, "omni.graph.nodes.WritePrimAttribute", primPath=prim_path, usePath=True, name=name)
        self.link(prev, f"{nid}.inputs:execIn")
        self.link(expr, f"{nid}.inputs:value")
        return f"{nid}.outputs:execOut"

    def write_var(self, prev, name, expr):
        self.node(f"{name}_w", "omni.graph.core.WriteVariable", variableName=name)
        self.link(prev, f"{name}_w.inputs:execIn")
        self.link(expr, f"{name}_w.inputs:value")
        return f"{name}_w.outputs:execOut"

    def build(self):
        og.Controller.edit({"graph_path": self.path, "evaluator_name": "execution",
                            "pipeline_stage": og.GraphPipelineStage.GRAPH_PIPELINE_STAGE_ONDEMAND},
                           {K.CREATE_NODES: self.nodes, K.CREATE_VARIABLES: self.variables, K.SET_VALUES: self.vals,
                            K.CONNECT: self.conns})



# ---------------------------------------------------------------------------------------------------------------------
# items: YCB meshes (y-up) stood upright on the tables
# ---------------------------------------------------------------------------------------------------------------------
cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"])
for name, (asset, table, (ix, iy), yaw, mass) in ITEMS.items():
    path = f"{FAC}/Items/{name}"
    up = Gf.Rotation(Gf.Vec3d(1, 0, 0), -90.0)  # the YCB meshes' up is their -y: -> world +z
    q = (up * rotz(yaw)).GetQuat()
    it = xform(path, (ix, iy, 0.0), quat=Gf.Quatf(q))
    it.GetPrim().GetReferences().AddReference(f"{ASSETS}/Props/YCB/{asset}")
    prim = it.GetPrim()
    if not prim.HasAPI(UsdPhysics.RigidBodyAPI):  # the plain YCB meshes: give them a body and a convex collider
        UsdPhysics.RigidBodyAPI.Apply(prim)
        for p in Usd.PrimRange(prim):
            if p.IsA(UsdGeom.Mesh):
                UsdPhysics.CollisionAPI.Apply(p)
                UsdPhysics.MeshCollisionAPI.Apply(p).CreateApproximationAttr("convexDecomposition")
    UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(mass)
    lo = cache.ComputeWorldBound(prim).ComputeAlignedRange().GetMin()[2]
    top = TABLES[table]["height"]
    it.GetOrderedXformOps()[0].Set(Gf.Vec3d(ix, iy, top - lo + 0.0005))
    truth["items"][name] = dict(asset=asset.split("/")[-1], table=table, xy=[ix, iy], yaw=yaw, mass=mass)
    cache.Clear()

# ---------------------------------------------------------------------------------------------------------------------
# the steam line: pipe, supports, the valve on a branch (a fixed articulation with the lever as its one joint),
# the gauge, the leaking flange and its plume
# ---------------------------------------------------------------------------------------------------------------------
S = f"{FAC}/SteamLine"
sl = UsdGeom.Xform.Define(st, S)
state_attr(sl.GetPrim(), "arb:steam_active", True)
py, pz, pr = PIPE["y"], PIPE["z"], PIPE["r"]
pipe_len = PIPE["x1"] - PIPE["x0"]
cyl(f"{S}/Pipe/main", ((PIPE["x0"] + PIPE["x1"]) / 2, py, pz), pr, pipe_len, "X", M["pipe"])
for k, sx in enumerate((-3.0, -1.0, 1.0, 3.0)):
    box(f"{S}/Pipe/support_{k}", (sx, py, pz / 2 - 0.02), (0.06, 0.06, pz - 0.04 - pr), M["steel"])
    box(f"{S}/Pipe/foot_{k}", (sx, py, 0.01), (0.16, 0.16, 0.02), M["steel"])
# a tee drops a branch to the valve body
vx = VALVE["x"]
wx, wy, wz = VALVE["stem"]
cyl(f"{S}/Pipe/tee", (vx, py, pz), pr + 0.015, 0.14, "X", M["pipe"])
cyl(f"{S}/Pipe/branch", (vx, py, (pz + wz + 0.075) / 2), pr * 0.8, pz - wz - 0.075, "Z", M["pipe"])  # (ends at the valve block)
cyl(f"{S}/Pipe/branch_end", (vx, py, wz - 0.075 - 0.15), pr * 0.8, 0.30, "Z", M["pipe"])
vb = UsdGeom.Xform.Define(st, f"{S}/ValveBody")
UsdPhysics.RigidBodyAPI.Apply(vb.GetPrim())
UsdPhysics.MassAPI.Apply(vb.GetPrim()).CreateMassAttr(6.0)
UsdPhysics.ArticulationRootAPI.Apply(vb.GetPrim())
PhysxSchema.PhysxArticulationAPI.Apply(vb.GetPrim()).CreateEnabledSelfCollisionsAttr(False)
box(f"{S}/ValveBody/block", (vx, py, wz), (0.16, 0.16, 0.14), M["valve"])
cyl(f"{S}/ValveBody/bonnet", (vx, py - 0.08 - 0.03, wz), 0.035, 0.06, "Y", M["valve"])
cyl(f"{S}/ValveBody/stem", (vx, wy + 0.025, wz), 0.010, 0.05, "Y", M["metal"], col=False)
# the stop plate: the lever rests against it open (pointing up) and closed (pointing along +x)
box(f"{S}/ValveBody/stop_open", (vx - 0.03, wy + 0.006, wz + 0.09), (0.02, 0.012, 0.06), M["dark"])
box(f"{S}/ValveBody/stop_closed", (vx + 0.09, wy + 0.006, wz - 0.03), (0.06, 0.012, 0.02), M["dark"])
vj = UsdPhysics.FixedJoint.Define(st, f"{S}/ValveBody/RootJoint")
vj.CreateBody1Rel().SetTargets([f"{S}/ValveBody"])
hw = UsdGeom.Xform.Define(st, f"{S}/Handle")  # a quarter-turn lever on the stem: up = open, along +x = closed
UsdPhysics.RigidBodyAPI.Apply(hw.GetPrim())
UsdPhysics.MassAPI.Apply(hw.GetPrim()).CreateMassAttr(0.5)
hw.AddTranslateOp().Set(Gf.Vec3d(wx, wy, wz))
lv = VALVE["lever"]
cyl(f"{S}/Handle/hub", (0, 0, 0), 0.03, 0.03, "Y", M["wheel"])
box(f"{S}/Handle/bar", (0, 0, lv / 2), (0.022, 0.018, lv), M["wheel"])
box(f"{S}/Handle/grip", (0, 0, lv - 0.04), (0.03, 0.03, 0.08), M["red"])
wj = UsdPhysics.RevoluteJoint.Define(st, f"{S}/HandleJoint")
wj.CreateBody0Rel().SetTargets([f"{S}/ValveBody"])
wj.CreateBody1Rel().SetTargets([f"{S}/Handle"])
wj.CreateAxisAttr("Y")
wj.CreateLowerLimitAttr(-2.0)  # (the lever swings about -y: positive = from up toward +x)
wj.CreateUpperLimitAttr(VALVE["close_deg"] + 2.0)
wj.CreateLocalPos0Attr(Gf.Vec3f(wx, wy, wz))
wj.CreateLocalPos1Attr(Gf.Vec3f(0, 0, 0))
wd = UsdPhysics.DriveAPI.Apply(wj.GetPrim(), "angular")  # viscous resistance only: the stem's packing
wd.CreateTypeAttr("force")
wd.CreateStiffnessAttr(0.0)
wd.CreateDampingAttr(0.02)  # (angular drive damping is per degree/s: 1.15 N m s/rad)
wd.CreateTargetPositionAttr(0.0)
wd.CreateMaxForceAttr(50.0)
# the pressure gauge: a dial on a stub, its face toward the bay (-y)
gx, gy, gz = GAUGE["center"]
cyl(f"{S}/Gauge/stub", (gx, py - 0.05, gz - 0.04), 0.012, 0.10, "Y", M["metal"])
cyl(f"{S}/Gauge/stub_up", (gx, gy + 0.02, gz - 0.02), 0.012, 0.05, "Z", M["metal"])
cyl(f"{S}/Gauge/case", (gx, gy + 0.012, gz), GAUGE["r"] + 0.006, 0.025, "Y", M["dark"])
cyl(f"{S}/Gauge/Dial", (gx, gy, gz), GAUGE["r"], 0.002, "Y", M["white"], col=False)
needle = box(f"{S}/Gauge/needle", (gx + 0.012, gy - 0.002, gz + 0.014), (0.004, 0.001, 0.070), M["red"], col=False)
needle.AddOrientOp().Set(Gf.Quatf(Gf.Rotation(Gf.Vec3d(0, 1, 0), 40.0).GetQuat()))
ops = needle.GetOrderedXformOps()
needle.SetXformOpOrder([ops[0], ops[2], ops[1]])
for k in range(11):  # the scale's ticks
    a = math.radians(-135.0 + 27.0 * k)
    tk = box(f"{S}/Gauge/tick_{k}", (gx + 0.042 * math.sin(a), gy - 0.002, gz + 0.042 * math.cos(a)), (0.002, 0.001, 0.008), M["black"], col=False)
    tk.AddOrientOp().Set(Gf.Quatf(Gf.Rotation(Gf.Vec3d(0, 1, 0), math.degrees(a)).GetQuat()))
    ops = tk.GetOrderedXformOps()
    tk.SetXformOpOrder([ops[0], ops[2], ops[1]])
# the leaking flange and its plume (a cloud of translucent spheres rising and drifting into the bay)
lx = LEAK["x"]
cyl(f"{S}/Leak/flange_a", (lx - 0.035, py, pz), pr + 0.03, 0.025, "X", M["steel"])
cyl(f"{S}/Leak/flange_b", (lx + 0.035, py, pz), pr + 0.03, 0.025, "X", M["steel"])
plume = UsdGeom.Xform.Define(st, f"{S}/Plume")  # the hazard volume: an invisible box (the grader reads its bounds)
plo, phi = PLUME["lo"], PLUME["hi"]
pbox = box_lohi(f"{S}/Plume/volume", plo, phi, M["steam"], col=False)
UsdGeom.Imageable(pbox).MakeInvisible()
# the steam itself: an NVIDIA Flow volume at the flange, under a switchable Xform
if not BAY_ONLY:
    steam_fx = UsdGeom.Xform.Define(st, f"{S}/Steam")
    flow_steam = UsdGeom.Xform.Define(st, f"{S}/Steam/Flow")
    flow_steam.AddTranslateOp().Set(Gf.Vec3d(lx, py - 0.09, pz + 0.03))
    flow_steam.GetPrim().GetReferences().AddReference(f"{ARB}/flow/FlowSteam.usda")
truth["steam"] = dict(pipe=PIPE, valve=VALVE, gauge=GAUGE, leak_x=lx, plume=PLUME)

# ---------------------------------------------------------------------------------------------------------------------
# the fire pan, the water tank, the bucket
# ---------------------------------------------------------------------------------------------------------------------
F = f"{FAC}/FirePan"
fp = UsdGeom.Xform.Define(st, F)
state_attr(fp.GetPrim(), "arb:fire_active", True)
fx, fy = FIRE["center"]
cyl(f"{F}/pan", (fx, fy, FIRE["rim"] / 2), FIRE["r"], FIRE["rim"], "Z", M["pan"])
cyl(f"{F}/oil", (fx, fy, FIRE["rim"] - 0.012), FIRE["r"] - 0.015, 0.004, "Z", M["black"], col=False)
# the fire itself: an NVIDIA Flow volume (the task's own asset), under a switchable Xform; the grader never renders
if not BAY_ONLY:
    flames = UsdGeom.Xform.Define(st, f"{F}/Flames")
    flow_fire = UsdGeom.Xform.Define(st, f"{F}/Flames/Flow")
    flow_fire.AddTranslateOp().Set(Gf.Vec3d(fx, fy, FIRE["rim"] - 0.02))
    flow_fire.GetPrim().GetReferences().AddReference(f"{ARB}/flow/FlowFire.usda")
truth["fire"] = FIRE

T = f"{FAC}/Tank"
tx, ty = TANK["center"]
tw, td, th = TANK["size"]
tt = TANK["wall"]
box_lohi(f"{T}/bottom", (tx - tw / 2, ty - td / 2, 0), (tx + tw / 2, ty + td / 2, tt), M["tank"])
box_lohi(f"{T}/wall_n", (tx - tw / 2, ty + td / 2 - tt, 0), (tx + tw / 2, ty + td / 2, th), M["tank"])
box_lohi(f"{T}/wall_s", (tx - tw / 2, ty - td / 2, 0), (tx + tw / 2, ty - td / 2 + tt, th), M["tank"])
box_lohi(f"{T}/wall_e", (tx + tw / 2 - tt, ty - td / 2, 0), (tx + tw / 2, ty + td / 2, th), M["tank"])
box_lohi(f"{T}/wall_w", (tx - tw / 2, ty - td / 2, 0), (tx - tw / 2 + tt, ty + td / 2, th), M["tank"])
water = box_lohi(f"{T}/water", (tx - tw / 2 + tt, ty - td / 2 + tt, tt), (tx + tw / 2 - tt, ty + td / 2 - tt, th - 0.08), M["water"], col=False)
truth["tank"] = TANK

Bk = f"{FAC}/Bucket"
bkx, bky = BUCKET["xy"]
br, bh = BUCKET["r"], BUCKET["h"]
xform(Bk, (bkx, bky, 0.0), 0.0)
bb = UsdGeom.Xform.Define(st, f"{Bk}/Body")
UsdPhysics.RigidBodyAPI.Apply(bb.GetPrim())
bm = UsdPhysics.MassAPI.Apply(bb.GetPrim())
bm.CreateMassAttr(BUCKET["mass"] + BUCKET["water"])  # full of water
bm.CreateCenterOfMassAttr(Gf.Vec3f(0, 0, 0.11))
pbk = PhysxSchema.PhysxRigidBodyAPI.Apply(bb.GetPrim())
pbk.CreateSolverPositionIterationCountAttr(32)
pbk.CreateSolverVelocityIterationCountAttr(4)
cyl(f"{Bk}/Body/shell", (0, 0, bh / 2), br, bh, "Z", M["bucket"])
cyl(f"{Bk}/Body/water_top", (0, 0, bh - 0.02), br - 0.006, 0.004, "Z", M["water"], col=False)
bail_local = []
for k, a in enumerate(BUCKET["bails"]):
    bp = ((br + 0.02) * math.cos(math.radians(a)), (br + 0.02) * math.sin(math.radians(a)), bh)  # the ring's centre, outside the shell
    bail_local.append(bp)
    bl = UsdGeom.Xform.Define(st, f"{Bk}/Body/Bail_{k}")
    bl.AddTranslateOp().Set(Gf.Vec3d(*bp))
    cyl(f"{Bk}/Body/Bail_{k}/ring", (0, 0, 0), 0.015, 0.006, "Z", M["metal"], col=False)
truth["bucket"] = dict(BUCKET, bail_local=bail_local, tether_length=TETHER_L, rope=ROPE)

# ---------------------------------------------------------------------------------------------------------------------
# the fleet: three hexacopters (airframe + inverted SO-101 + cameras + hook) on their pads, tethered to the bucket;
# one Jackal with a Kinova Gen3 and a mast camera in its bay
# ---------------------------------------------------------------------------------------------------------------------
ARM_DROP = 0.02  # [m] the arm's base plate below the airframe's bottom


def mount_arm(root, frame_body, mount_xy, frame_bottom_z):
    """Reference the SO-101 under `root`/Arm, drop its world anchor, fix its base upside down under the frame."""
    arm = UsdGeom.Xform.Define(st, f"{root}/Arm")
    arm.GetPrim().GetReferences().AddReference(f"{ASSETS}/Robots/RobotStudio/so101_new_calib/so101_new_calib.usd")
    arm.GetPrim().CreateAttribute("arb:arm_model", Sdf.ValueTypeNames.String).Set("so101")
    rj = st.GetPrimAtPath(f"{root}/Arm/root_joint")
    rj.SetActive(False)  # the asset bolts its base to the world; here the drone carries it
    base = st.GetPrimAtPath(f"{root}/Arm/base")
    flip = Gf.Quatf(Gf.Rotation(Gf.Vec3d(1, 0, 0), 180.0).GetQuat())
    xa = UsdGeom.Xformable(arm)
    xa.ClearXformOpOrder()  # (the asset's own root ops would compose underneath ours)
    xa.AddTranslateOp(opSuffix="mount").Set(Gf.Vec3d(mount_xy[0], mount_xy[1], frame_bottom_z - ARM_DROP))
    xa.AddOrientOp(opSuffix="mount").Set(flip)
    j = UsdPhysics.FixedJoint.Define(st, f"{root}/ArmMount")
    j.CreateBody0Rel().SetTargets([frame_body])
    j.CreateBody1Rel().SetTargets([base.GetPath()])
    j.CreateLocalPos0Attr(Gf.Vec3f(mount_xy[0], mount_xy[1], frame_bottom_z - ARM_DROP))
    j.CreateLocalRot0Attr(flip)
    j.CreateLocalPos1Attr(Gf.Vec3f(0, 0, 0))
    j.CreateLocalRot1Attr(Gf.Quatf(1, 0, 0, 0))
    return arm


def camera(path, model, pos, pitch_down, forward="y", yaw=0.0):
    """A catalog camera looking along the carrier's forward axis (+y, or +x), turned by `yaw` about the carrier's z
    (positive: left), pitched down (negative: up)."""
    sp = SENSORS[model]
    c = UsdGeom.Camera.Define(st, path)
    c.CreateFocalLengthAttr(sp["focal"])
    c.CreateHorizontalApertureAttr(sp["hap"])
    c.CreateVerticalApertureAttr(sp["vap"])
    c.CreateClippingRangeAttr(Gf.Vec2f(0.02, 100.0))
    c.GetPrim().CreateAttribute("arb:sensor_model", Sdf.ValueTypeNames.String).Set(model)
    # USD cameras look down their -z with +y up: turn -z to forward, then pitch it down
    q = Gf.Rotation(Gf.Vec3d(1, 0, 0), 90.0 - pitch_down)
    if forward == "x":
        q = q * Gf.Rotation(Gf.Vec3d(0, 0, 1), -90.0)
    if yaw:
        q = q * Gf.Rotation(Gf.Vec3d(0, 0, 1), yaw)
    x = UsdGeom.Xformable(c)
    x.AddTranslateOp().Set(Gf.Vec3d(*pos))
    x.AddOrientOp().Set(Gf.Quatf(q.GetQuat()))
    return dict(model=model, pos=list(pos), pitch_down=pitch_down, hfov_deg=math.degrees(2 * math.atan(sp["hap"] / 2 / sp["focal"])))



def quat_x_to(d):
    """A rotation taking +x onto the unit direction d."""
    return Gf.Quatf(Gf.Rotation(Gf.Vec3d(1, 0, 0), Gf.Vec3d(*d)).GetQuat())


def rope_chain(root, root_pos, body_path, hook_local, hook_world, bucket_body, bail_local, bail_world):
    """The rope from the robot's hook to one ring of the bucket: a chain of capsule links, plain rigid bodies on
    spherical joints (the hook joint keeps the chain out of the robot's articulation: PhysX cannot hang a heavy body
    off an articulation's leaf link), the last one joined to the bucket at its ring. Authored the way a slack rope
    lies: down from the hook, along the floor, up to the ring."""
    n, L, r = ROPE["links"], ROPE["length"], ROPE["radius"]
    seg = L / n
    h, b = tuple(hook_world), tuple(bail_world)
    foot = (h[0], h[1], r + 0.001)
    dx, dy = b[0] - h[0], b[1] - h[1]
    dh = math.hypot(dx, dy)

    side = (-dy / dh, dx / dh)  # (perpendicular to the hook-ring line, on the floor)

    def march(w):
        """Joint points along the path hook -> foot -> P(w) on the floor (half-way along, w off to the side) -> ring,
        each exactly one link (a chord) from the previous one; returns the points and how far the last one misses one
        link from the ring."""
        P = (h[0] + dx / 2 + side[0] * w, h[1] + dy / 2 + side[1] * w, r + 0.001)
        pts = [h, foot, P, b]
        segs = [(pts[i], pts[i + 1]) for i in range(3)]
        out = [h]
        si, f0 = 0, 0.0  # the current segment and the fraction along it
        for _ in range(n - 1):
            c = out[-1]
            found = None
            while si < 3 and found is None:
                a, e = segs[si]
                d = [e[k] - a[k] for k in range(3)]
                m = [a[k] - c[k] for k in range(3)]
                A = sum(v * v for v in d)
                Bq = 2 * sum(d[k] * m[k] for k in range(3))
                C = sum(v * v for v in m) - seg * seg
                disc = Bq * Bq - 4 * A * C
                if A > 0 and disc >= 0:
                    t = (-Bq + math.sqrt(disc)) / (2 * A)
                    if f0 <= t <= 1.0:
                        found = [a[k] + t * d[k] for k in range(3)]
                        f0 = t
                if found is None:
                    si, f0 = si + 1, 0.0
            if found is None:  # (the path ran out: stretch the rest along the last direction)
                found = [c[k] + (b[k] - c[k]) / max(1e-9, math.dist(c, b)) * seg for k in range(3)]
            out.append(found)
        return out, math.dist(out[-1], b) - seg

    lo, hi = 0.0, 2.0
    for _ in range(60):  # the sideways offset so that the last link ends exactly on the ring (more offset: a longer path)
        mid = (lo + hi) / 2
        if march(mid)[1] > 0:
            hi = mid
        else:
            lo = mid
    pts, miss = march((lo + hi) / 2)
    pts.append(b)
    if abs(miss) > 0.01:
        print(f"[reference] rope {root}: last link misses the ring by {miss:.3f} m", flush=True)

    def at(i):
        """Link i: its centre and direction, from its two joint points."""
        p0, p1 = pts[i], pts[i + 1]
        d = [p1[k] - p0[k] for k in range(3)]
        ln = max(1e-9, math.sqrt(sum(v * v for v in d)))
        return [(p0[k] + p1[k]) / 2 for k in range(3)], [v / ln for v in d]

    UsdGeom.Xform.Define(st, f"{root}/Rope")
    links = []
    for i in range(n):
        c, d = at(i)
        lp = f"{root}/Rope/Link_{i}"
        lx = UsdGeom.Xform.Define(st, lp)
        lx.AddTranslateOp().Set(Gf.Vec3d(c[0] - root_pos[0], c[1] - root_pos[1], c[2] - root_pos[2]))  # (under the robot's root: local)
        lx.AddOrientOp().Set(quat_x_to(d))
        UsdPhysics.RigidBodyAPI.Apply(lx.GetPrim())
        UsdPhysics.MassAPI.Apply(lx.GetPrim()).CreateMassAttr(ROPE["kg_per_m"] * seg)
        prb = PhysxSchema.PhysxRigidBodyAPI.Apply(lx.GetPrim())
        prb.CreateAngularDampingAttr(0.2)
        prb.CreateLinearDampingAttr(0.02)
        prb.CreateSolverPositionIterationCountAttr(32)
        prb.CreateSolverVelocityIterationCountAttr(4)
        cap = UsdGeom.Capsule.Define(st, f"{lp}/geom")
        cap.CreateAxisAttr("X")
        cap.CreateRadiusAttr(r)
        cap.CreateHeightAttr(seg - 2 * r)
        UsdPhysics.CollisionAPI.Apply(cap.GetPrim())
        pc = PhysxSchema.PhysxCollisionAPI.Apply(cap.GetPrim())
        pc.CreateContactOffsetAttr(0.006)
        pc.CreateRestOffsetAttr(0.0)
        UsdShade.MaterialBindingAPI.Apply(cap.GetPrim()).Bind(M["black"])
        links.append(lp)
    prev_body, prev_lp = body_path, tuple(hook_local)
    for i, lp in enumerate(links):
        j = UsdPhysics.SphericalJoint.Define(st, f"{root}/Rope/Joint_{i}")
        j.CreateBody0Rel().SetTargets([prev_body])
        j.CreateBody1Rel().SetTargets([lp])
        j.CreateLocalPos0Attr(Gf.Vec3f(*prev_lp))
        j.CreateLocalPos1Attr(Gf.Vec3f(-seg / 2, 0, 0))
        if i == 0:
            j.CreateExcludeFromArticulationAttr(True)  # the chain is not part of the robot's articulation
        prev_body, prev_lp = lp, (seg / 2, 0, 0)
    j = UsdPhysics.SphericalJoint.Define(st, f"{root}/Rope/Joint_{n}")  # the ring: a joint outside the articulation
    j.CreateBody0Rel().SetTargets([links[-1]])
    j.CreateBody1Rel().SetTargets([bucket_body])
    j.CreateLocalPos0Attr(Gf.Vec3f(seg / 2, 0, 0))
    j.CreateLocalPos1Attr(Gf.Vec3f(*bail_local))
    j.CreateExcludeFromArticulationAttr(True)
    # the links never collide with the bucket (the last ones run up its side to the ring)
    fp = UsdPhysics.FilteredPairsAPI.Apply(st.GetPrimAtPath(bucket_body))
    rel = fp.GetFilteredPairsRel()
    for lp in links:
        rel.AddTarget(lp)
    return links


if not BAY_ONLY:  # the fleet the reference designed (the real bay has none: the mission brings the agent's)
    hx_, hy_, hz_ = AIRFRAME["hull"]
    rest_z = hz_ / 2 + AIRFRAME["gear"] + 0.0075  # the skids on the floor
    for k, (pad, (px, py_)) in enumerate(PADS.items(), start=1):
        root = f"{FLEET}/UAV_{k}"
        rx = xform(root, (px, py_, rest_z), 0.0)  # heading +y (the pad's arrow)
        state_attr(rx.GetPrim(), "arb:disabled", False)
        air = UsdGeom.Xform.Define(st, f"{root}/Airframe")
        air.GetPrim().GetReferences().AddReference(AIRFRAME["file"])
        body_p = st.GetPrimAtPath(f"{root}/Airframe/Body")
        UsdPhysics.ArticulationRootAPI.Apply(body_p)
        pa_ = PhysxSchema.PhysxArticulationAPI.Apply(body_p)
        pa_.CreateEnabledSelfCollisionsAttr(False)
        pa_.CreateSolverPositionIterationCountAttr(32)
        pa_.CreateSolverVelocityIterationCountAttr(4)
        # the sensors' masses ride on the hull
        UsdPhysics.MassAPI(body_p).GetMassAttr().Set(AIRFRAME["empty"] + sum(SENSORS[m]["mass"] for m in UAV_SENSORS))
        mount_arm(root, f"{root}/Airframe/Body", (0.0, 0.07), -hz_ / 2)
        # forward thermal (the pan from the approach hover), forward RGB, a nadir thermal (the pan and the fire below while
        # pouring), a nadir wide RGB (the bucket on its ropes), a 3D lidar (the other robots)
        cams = [camera(f"{root}/Airframe/Body/Thermal", "THERM-C", (0.0, hy_ / 2 + 0.01, -0.025), 50.0),
                camera(f"{root}/Airframe/Body/Camera", "CAM-RGB-M", (0.0, hy_ / 2 + 0.01, 0.025), 25.0),
                camera(f"{root}/Airframe/Body/ThermalDown", "THERM-A", (0.06, 0.0, -hz_ / 2 - 0.01), 90.0),
                camera(f"{root}/Airframe/Body/CameraDown", "CAM-RGB-W", (-0.06, 0.0, -hz_ / 2 - 0.01), 90.0)]
        lid = UsdGeom.Xform.Define(st, f"{root}/Airframe/Body/Lidar")
        lid.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, hz_ / 2 + 0.03))
        lid.GetPrim().CreateAttribute("arb:sensor_model", Sdf.ValueTypeNames.String).Set("LIDAR-3D")
        cyl(f"{root}/Airframe/Body/Lidar/puck", (0.0, 0.0, 0.0), 0.035, 0.05, "Z", M["dark"], col=False)
        # the tether: a physical rope (a chain) from the hook to one ring of the bucket
        hook_w = (px + AIRFRAME["hook"][0], py_ + AIRFRAME["hook"][1], rest_z + AIRFRAME["hook"][2])
        bail_w = (bkx + bail_local[k - 1][0], bky + bail_local[k - 1][1], bail_local[k - 1][2])
        rope_chain(root, (px, py_, rest_z), f"{root}/Airframe/Body", AIRFRAME["hook"], hook_w, f"{Bk}/Body", bail_local[k - 1], bail_w)
        truth["fleet"][f"UAV_{k}"] = dict(airframe=AIRFRAME["model"], arm="so101", pad=pad, xy=[px, py_], rest_z=rest_z, cameras=cams,
                                          hook=AIRFRAME["hook"])

    # --- the ground vehicle: Jackal base + Kinova Gen3 on its deck + a mast camera, parked in its bay heading +y ---
    root = f"{FLEET}/UGV"
    ux, uy = UGV_BAY["center"]
    rx = xform(root, (ux, uy, 0.0), 90.0)  # the Jackal's forward is its +x: turned to +y
    state_attr(rx.GetPrim(), "arb:disabled", False)
    base = UsdGeom.Xform.Define(st, f"{root}/Base")
    base.GetPrim().GetReferences().AddReference(UGV_BASE["file"])
    base.GetPrim().CreateAttribute("arb:base_model", Sdf.ValueTypeNames.String).Set(UGV_BASE["model"])
    cache.Clear()
    r = cache.ComputeWorldBound(base.GetPrim()).ComputeAlignedRange()
    base_lo, base_hi = r.GetMin(), r.GetMax()
    rx.GetOrderedXformOps()[0].Set(Gf.Vec3d(ux, uy, -base_lo[2] + 0.002))  # wheels on the floor
    deck_z = base_hi[2] - base_lo[2] + 0.002  # the deck's height above the floor, in the vehicle's frame (z offset)
    deck_local_z = base_hi[2]
    for p in Usd.PrimRange(base.GetPrim()):  # the asset's own sensors (cameras, a lidar, an IMU) are not part of this design
        if p.IsA(UsdGeom.Camera) or p.GetTypeName() in ("Lidar", "IsaacImuSensor", "OmniLidar", "IsaacRtxLidarSensor"):
            p.SetActive(False)
    # the asset puts the articulation root on its top Xform; PhysX then takes the first body it meets as the root link, which
    # can be the arm's once the arm is joined on. The root goes on the chassis itself.
    base.GetPrim().RemoveAPI(UsdPhysics.ArticulationRootAPI)
    chassis = st.GetPrimAtPath(f"{root}/Base/base_link")
    UsdPhysics.ArticulationRootAPI.Apply(chassis)
    PhysxSchema.PhysxArticulationAPI.Apply(chassis).CreateEnabledSelfCollisionsAttr(False)
    arm = UsdGeom.Xform.Define(st, f"{root}/Arm")
    arm.GetPrim().GetReferences().AddReference(UGV_ARM["file"])
    arm.GetPrim().CreateAttribute("arb:arm_model", Sdf.ValueTypeNames.String).Set(UGV_ARM["model"])
    arm.GetPrim().RemoveAPI(UsdPhysics.ArticulationRootAPI)  # one articulation: the vehicle's
    st.GetPrimAtPath(f"{root}/Arm/{UGV_ARM['root_joint']}").SetActive(False)
    STOW = {"joint_2": -35.0, "joint_4": -100.0, "joint_6": -50.0}  # folded back over the deck (the asset's own targets fold it forward, off the front axle)
    for p in Usd.PrimRange(arm.GetPrim()):  # start where the drives hold it, not swinging there
        if p.IsA(UsdPhysics.RevoluteJoint):
            tgt = STOW.get(p.GetName(), 0.0)
            UsdPhysics.DriveAPI(p, "angular").GetTargetPositionAttr().Set(tgt)
            PhysxSchema.JointStateAPI.Apply(p, "angular").CreatePositionAttr(float(tgt))
    arm_mount = (0.0, 0.0, deck_local_z + 0.01)  # on the deck's centre (vehicle frame: +x forward)
    xa = UsdGeom.Xformable(arm)
    xa.ClearXformOpOrder()
    xa.AddTranslateOp(opSuffix="mount").Set(Gf.Vec3d(*arm_mount))
    xa.AddOrientOp(opSuffix="mount").Set(Gf.Quatf(1, 0, 0, 0))
    box(f"{root}/Base/base_link/deck_plate", (0.0, 0.0, deck_local_z + 0.005), (0.30, 0.30, 0.01), M["steel"])
    j = UsdPhysics.FixedJoint.Define(st, f"{root}/ArmMount")
    j.CreateBody0Rel().SetTargets([f"{root}/Base/base_link"])
    j.CreateBody1Rel().SetTargets([f"{root}/Arm/{UGV_ARM['base']}"])
    j.CreateLocalPos0Attr(Gf.Vec3f(*arm_mount))
    j.CreateLocalRot0Attr(Gf.Quatf(1, 0, 0, 0))
    j.CreateLocalPos1Attr(Gf.Vec3f(0, 0, 0))
    j.CreateLocalRot1Attr(Gf.Quatf(1, 0, 0, 0))
    mast_top = deck_local_z + 0.45
    box(f"{root}/Base/base_link/mast", (-0.16, 0.14, (deck_local_z + mast_top) / 2), (0.03, 0.03, mast_top - deck_local_z), M["steel"])
    ugv_cam = camera(f"{root}/Base/base_link/Camera", "CAM-RGB-M", (-0.16, 0.14, mast_top + 0.03), -10.0, forward="x", yaw=-62.0)
    # the sensor's mass rides on the chassis (the asset's mass is on its collision body)
    cb = st.GetPrimAtPath(f"{root}/Base/base_link/collisions/body")
    if cb.IsValid() and cb.HasAPI(UsdPhysics.MassAPI):
        cm = UsdPhysics.MassAPI(cb).GetMassAttr()
        cm.Set(float(cm.Get()) + SENSORS["CAM-RGB-M"]["mass"])
    truth["fleet"]["UGV"] = dict(base=UGV_BASE["model"], arm=UGV_ARM["model"], bay=UGV_BAY["center"], yaw=90.0, camera=ugv_cam,
                                 deck_z=deck_z, arm_mount=list(arm_mount))

# ---------------------------------------------------------------------------------------------------------------------
# the bay's rules, as one OmniGraph on every physics step:
#   steam: a quarter turn of the valve's lever closes it; the leak stops 3 s later (and resumes when it is opened);
#   a flying robot inside the plume while it is active is disabled (and stays so);
#   the ground vehicle within 2 m of the burning pan is disabled (and stays so);
#   the bucket tipped past 15 deg over the burning pan, low enough, for 1 s in all, puts the fire out (for good)
# ---------------------------------------------------------------------------------------------------------------------
if not BAY_ONLY:  # (the hazard logic is the agent's to build; the bay mission keeps the rules itself)
    L = Graph(f"{FAC}/Logic", [("closed_t", og.Type(og.BaseDataType.DOUBLE)), ("pour_t", og.Type(og.BaseDataType.DOUBLE)),
                               ("fire", og.Type(og.BaseDataType.BOOL)), ("init", og.Type(og.BaseDataType.BOOL))]
              + [(f"dis_{n}", og.Type(og.BaseDataType.BOOL)) for n in ("uav1", "uav2", "uav3", "ugv")])
    L.node("tick", "isaacsim.core.nodes.OnPhysicsStep")
    dt = "tick.outputs:deltaSimulationTime"
    L.node("vstate", "isaacsim.core.nodes.IsaacArticulationState", robotPath=f"{S}/ValveBody", jointNames=["HandleJoint"])
    L.link("tick.outputs:step", "vstate.inputs:execIn")
    L.node("q_lever", "omni.graph.nodes.ArrayIndex", index=0)
    L.link("vstate.outputs:jointPositions", "q_lever.inputs:array")
    turned2 = L.mul("turned2", "q_lever.outputs:value", "q_lever.outputs:value")  # (no Abs node: compare the square)
    closed = L.cmp("closed", turned2, L.const("c_closed", math.radians(VALVE["close_deg"] - 5.0) ** 2), ">=")
    # `fire` starts true: the variable's default is false, so the first step (init false) treats the fire as burning
    not_init = L.cmp("not_init", L.var("init"), L.const("c_false", False, "Bool"), "==")
    fire_now = L.bor("fire_now", L.bvar("fire"), not_init)
    closed_t_next = L.sel("closed_t_next", closed, L.add("closed_t_inc", L.var("closed_t"), dt), L.const("zero_t", 0.0))
    steam_on = L.cmp("steam_on", closed_t_next, L.const("c_stop", VALVE["steam_stop_s"]), "<")
    prev = "tick.outputs:step"
    prev = L.write_attr("w_steam", prev, S, "arb:steam_active", steam_on)
    dis_next = {}
    for k in (1, 2, 3):
        xyz = L.world_pos(f"p{k}", f"{FLEET}/UAV_{k}/Airframe/Body")
        inside = L.band(f"in_plume{k}", L.in_box(f"box{k}", xyz, PLUME["lo"], PLUME["hi"]), steam_on)
        dis_next[f"uav{k}"] = L.bor(f"dis_next{k}", L.bvar(f"dis_uav{k}"), inside)
        prev = L.write_attr(f"w_dis{k}", prev, f"{FLEET}/UAV_{k}", "arb:disabled", dis_next[f"uav{k}"])
    gxyz = L.world_pos("pg", f"{FLEET}/UGV/Base/base_link")
    near_fire = L.band("near_fire", L.cmp("near_fire_c", L.dist2_xy("gd", gxyz, FIRE["center"]), L.const("c_haz", FIRE["hazard_r"] ** 2), "<="), fire_now)
    dis_next["ugv"] = L.bor("dis_next_ugv", L.bvar("dis_ugv"), near_fire)
    prev = L.write_attr("w_dis_ugv", prev, f"{FLEET}/UGV", "arb:disabled", dis_next["ugv"])
    bxyz = L.world_pos("pb", f"{Bk}/Body")
    upz = L.world_up_z("ub", f"{Bk}/Body")
    tipped = L.cmp("tipped", upz, L.const("c_tip", math.cos(math.radians(FIRE["pour_tilt_deg"]))), "<")
    over = L.band("over_pan", L.cmp("over_r", L.dist2_xy("bd", bxyz, FIRE["center"]), L.const("c_pour_r", FIRE["pour_r"] ** 2), "<="),
                  L.cmp("over_z", bxyz[2], L.const("c_pour_z", FIRE["pour_zmax"]), "<="))
    pouring = L.band("pouring", L.band("tip_over", tipped, over), fire_now)
    pour_t_next = L.sel("pour_t_next", pouring, L.add("pour_inc", L.var("pour_t"), dt), L.var("pour_t"))  # (what is poured stays poured)
    fire_next = L.band("fire_next", fire_now, L.cmp("not_doused", pour_t_next, L.const("c_pour_s", FIRE["pour_s"]), "<"))
    prev = L.write_attr("w_fire", prev, F, "arb:fire_active", fire_next)
    prev = L.write_attr("w_pour", prev, f"{Bk}/Body", "arb:pouring", pouring)
    bb.GetPrim().CreateAttribute("arb:pouring", Sdf.ValueTypeNames.Bool).Set(False)
    # variables last, each after everything that reads it: the flags, the fire (reads pour_t), pour_t, closed_t, init
    for n in ("uav1", "uav2", "uav3", "ugv"):
        prev = L.write_var(prev, f"dis_{n}", dis_next[n])
    prev = L.write_var(prev, "fire", fire_next)
    prev = L.write_var(prev, "pour_t", pour_t_next)
    prev = L.write_var(prev, "closed_t", closed_t_next)
    prev = L.write_var(prev, "init", L.const("c_true", True, "Bool"))
    L.build()

# the robot assets ship their meshes as instanceable prims; rendered, those stayed at the asset origin while the bodies
# moved. Plain prims follow their bodies.
for p in list(st.Traverse()):
    if p.IsInstanceable() and p.GetPath().pathString.startswith((FLEET, f"{FAC}/Items")):
        p.SetInstanceable(False)

if not BAY_ONLY:
    st.GetRootLayer().customLayerData = {"renderSettings": {"rtx:flow:enabled": True, "rtx:flow:rayTracedReflectionsEnabled": True,
                                                            "rtx:flow:rayTracedTranslucencyEnabled": True, "rtx:flow:pathTracingEnabled": True}}
os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
st.GetRootLayer().Export(args.out)
if BAY_ONLY:  # (only the bay: not the render settings and viewport camera a new Kit stage starts with)
    _lyr = Sdf.Layer.FindOrOpen(args.out)
    for _spec in list(_lyr.rootPrims):
        if _spec.name != "World":
            del _lyr.rootPrims[_spec.name]
    _lyr.Save()
if args.truth:
    with open(args.truth, "w") as f:
        json.dump(truth, f, indent=1)
if args.bay_facts:  # the numbers the grader's bay mission stage is built from (tests/hidden/bay_facts.json)
    facts = dict(
        pads={n: dict(center=list(c), size=[PAD_SIZE, PAD_SIZE]) for n, c in PADS.items()},
        ugv_bay=dict(center=list(UGV_BAY["center"]), size=list(UGV_BAY["size"])),
        valve_square=dict(center=list(VALVE_SQUARE["center"]), size=list(VALVE_SQUARE["size"])),
        table_tops=[dict(name=n, center=list(t["center"]), size=list(t["size"]), yaw_deg=t["yaw"], height=t["height"])
                    for n, t in TABLES.items()],
        crate=dict(center=list(CRATE["center"]), size=list(CRATE["size"]), wall=CRATE["wall"], yaw_deg=CRATE["yaw"]),
        pan=dict(center=list(FIRE["center"]), r=FIRE["r"], rim=FIRE["rim"]),
        valve=dict(stem=list(VALVE["stem"]), lever_m=VALVE["lever"], open_deg=0.0, closed_deg=VALVE["close_deg"],
                   axis="-y: positive from up toward +x"),
        floor=dict(static_friction=FLOOR_FRICTION, dynamic_friction=FLOOR_FRICTION, restitution=0.0),
        rig=dict(bucket_body=f"{FAC}/Bucket/Body", bucket_center=list(BUCKET["xy"]), rings=[list(b) for b in bail_local],
                 ring_bearing_deg=list(BUCKET["bails"]), ring_of_pad=RING_OF_PAD,
                 rope=dict(length=ROPE["length"], links=ROPE["links"], radius=ROPE["radius"], kg_per_m=ROPE["kg_per_m"],
                           angular_damping=0.2, linear_damping=0.02, solver_position_iterations=32, solver_velocity_iterations=4,
                           contact_offset=0.006, rest_offset=0.0)))
    os.makedirs(os.path.dirname(os.path.abspath(args.bay_facts)), exist_ok=True)
    with open(args.bay_facts, "w") as f:
        json.dump(facts, f, indent=1)
print("[reference] scene ->", args.out, flush=True)
os._exit(0)
