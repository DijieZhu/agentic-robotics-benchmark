"""The loose objects of the cleaning job and the moving parts of the fixtures: geometry, mass, colliders, what the
cameras see of them (procedural meshes, props.py) and where a hand can take hold of them (grasp features).

Every builder adds its geoms to a body and returns the body's grasp features in the body's frame:
  ("cyl", a, b, r)            a cylinder or bar from point a to point b of radius r (cans, bottles, glasses,
                              handles, lid tabs, the faucet lever): the hand's pinch closes across its axis;
  ("slab", c, half, k)        a box of half sizes `half` at c whose thin axis is k (books, the sponge): the pinch
                              closes across its thickness, near an edge;
  ("disc", c, R, t)           a flat disc of radius R and half thickness t about the body's z axis (plates): the
                              pinch closes across the rim;
  ("ball", c, r)              a compact lump (crumpled napkins, the apple core, the soap pump's head).
physics.Grips turns them into the cell's grip rule (the cell manual states it).

Collision bits (robot.py): 1 = the static room and its fixtures, 2 = the robot, 4 = loose objects and tools, 8 = the
floor. Loose objects touch everything; the fixtures' moving parts are fixtures (bit 1).
"""

from __future__ import annotations

import math

import mujoco
import numpy as np

from . import props as P
from . import room as RM
from .robot import FLOOR, OBJ, ROBOT, WORLD

OBJ_CONTACT = dict(contype=OBJ, conaffinity=WORLD | ROBOT | OBJ | FLOOR, condim=3, group=3)
FIX_CONTACT = dict(contype=WORLD, conaffinity=ROBOT | OBJ, condim=3, group=3)
VISUAL = dict(contype=0, conaffinity=0, group=1, mass=0.0)

# what a crumpled napkin, a can, ... is in the scoring (instruction: recycling = cans, bottles; general = the rest)
CATEGORY = {"can": "recycling", "bottle": "recycling", "napkin": "general", "apple": "general",
            "plate": "dish", "glass": "dish", "book": "book", "sponge": "tool", "broom": "tool", "dustpan": "tool"}

# the loose objects of every episode: name -> (kind, parameters)
LOOSE = {
    "can_red": ("can", {"tex": "can_red"}),
    "can_green": ("can", {"tex": "can_green"}),
    "bottle": ("bottle", {}),
    "napkin_1": ("napkin", {"seed": 1}),
    "napkin_2": ("napkin", {"seed": 2}),
    "napkin_3": ("napkin", {"seed": 3}),
    "apple_core": ("apple", {}),
    "plate_1": ("plate", {}),
    "plate_2": ("plate", {}),
    "glass_1": ("glass", {}),
    "glass_2": ("glass", {}),
    "book_red": ("book", {"size": (0.160, 0.030, 0.235), "tex": "book_red", "mass": 0.45}),
    "book_blue": ("book", {"size": (0.150, 0.024, 0.210), "tex": "book_blue", "mass": 0.34}),
    "sponge": ("sponge", {}),
    "broom": ("broom", {}),
    "dustpan": ("dustpan", {}),
}
TRASH = tuple(k for k, (kind, _) in LOOSE.items() if CATEGORY[kind] in ("recycling", "general"))
DISHES = tuple(k for k, (kind, _) in LOOSE.items() if CATEGORY[kind] == "dish")
BOOKS = tuple(k for k, (kind, _) in LOOSE.items() if CATEGORY[kind] == "book")
FRAGILE = DISHES

# dimensions the layout and the scoring use
CAN = {"r": 0.033, "h": 0.122, "mass": 0.030}
BOTTLE = {"r": 0.034, "h": 0.205, "cap_r": 0.015, "cap_h": 0.02, "mass": 0.032}
NAPKIN = {"r": 0.030, "mass": 0.012}
APPLE = {"r": 0.022, "half": 0.018, "mass": 0.030}
PLATE = {"r": 0.130, "h": 0.022, "mass": 0.45}
GLASS = {"r": 0.035, "h": 0.115, "mass": 0.22}
SPONGE = {"size": (0.110, 0.072, 0.034), "pad": 0.010, "mass": 0.022}
BROOM = {"handle_r": 0.0115, "handle": (0.10, 1.20), "head": (0.30, 0.055, 0.035), "bristle": (0.29, 0.045, 0.045),
         "mass_handle": 0.22, "mass_head": 0.30, "swivel": math.radians(50.0)}
DUSTPAN = {"width": 0.30, "depth": 0.24, "wall_h": 0.09, "t": 0.004, "handle_r": 0.012, "handle_len": 0.80,
           "handle_tilt": math.radians(8.0), "mass_pan": 0.30, "mass_handle": 0.15}


def _quat_from_R(R):
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(R, float).reshape(-1))
    return q


def _rot_x(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], float)


class Builder:
    """Adds meshes, materials and textures to a spec once each (names are unique)."""

    def __init__(self, spec: mujoco.MjSpec):
        self.s = spec
        self.meshes = set()
        self.textures = set()
        self.materials = set()

    def texture(self, name):
        if name not in self.textures:
            img = P.texture(name)
            t = self.s.add_texture(name=f"tex_{name}", type=mujoco.mjtTexture.mjTEXTURE_2D, width=img.shape[1],
                                   height=img.shape[0], nchannel=3, colorspace=mujoco.mjtColorSpace.mjCOLORSPACE_SRGB)
            t.data = np.ascontiguousarray(img).tobytes()
            self.textures.add(name)
        return f"tex_{name}"

    def material(self, name, rgba=(1, 1, 1, 1), tex=None, **kw):
        if name not in self.materials:
            roles = ["", self.texture(tex)] if tex else []
            self.s.add_material(name=name, rgba=list(rgba), textures=roles, **kw)
            self.materials.add(name)
        return name

    def mesh(self, name, mesh):
        if name not in self.meshes:
            V, F, N, UV = mesh
            m = self.s.add_mesh(name=name, inertia=mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL)
            m.uservert = np.asarray(V, np.float64).ravel()
            m.usernormal = np.asarray(N, np.float64).ravel()
            m.usertexcoord = np.asarray(UV, np.float64).ravel()
            m.userface = np.asarray(F, np.int32).ravel()
            self.meshes.add(name)
        return name

    def visual(self, body, name, mesh, material, pos=(0, 0, 0), quat=(1, 0, 0, 0), rgba=None):
        kw = dict(VISUAL)
        if rgba is not None:
            kw["rgba"] = list(rgba)
        return body.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_MESH, meshname=self.mesh(name + "_mesh", mesh)
                             if isinstance(mesh, tuple) else mesh, material=material, pos=list(pos), quat=list(quat),
                             **kw)

    def prim(self, body, name, gtype, size, pos=(0, 0, 0), quat=(1, 0, 0, 0), *, material=None, rgba=None,
             visual=False, contact=OBJ_CONTACT, mass=None, friction=None, **kw):
        """A primitive geom: a collider (group 3, hidden: the visual mesh draws it) or, with visual=True, a drawn-only
        geom (group 1)."""
        args = dict(VISUAL) if visual else dict(contact)
        args.update(kw)
        if mass is not None and not visual:
            args["mass"] = mass
        if friction is not None:
            args["friction"] = list(friction)
        if material is not None:
            args["material"] = material
        if rgba is not None:
            args["rgba"] = list(rgba)
        if visual and kw.get("group") is None:
            args["group"] = 1
        return body.add_geom(name=name, type=gtype, size=list(size), pos=list(pos), quat=list(quat), **args)


# ================================================================================================ loose objects
G = mujoco.mjtGeom


def add_can(b: Builder, body, name, p):
    r, h = CAN["r"], CAN["h"]
    b.prim(body, f"{name}_col", G.mjGEOM_CYLINDER, [r, h / 2, 0], (0, 0, h / 2), mass=CAN["mass"],
           friction=(0.6, 0.005, 0.0001))
    prof = [(0.0, 0.0), (0.024, 0.0), (0.030, 0.004), (r, 0.012), (r, h - 0.014), (0.029, h - 0.004),
            (0.027, h), (0.0, h)]
    body_mesh = P.lathe(prof[1:7], segments=36)
    alu = b.material("alu", (0.78, 0.79, 0.80, 1))
    lab = b.material(f"label_{p['tex']}", tex=p["tex"])
    label = P.lathe([(r + 0.0004, 0.016), (r + 0.0004, h - 0.018)], segments=36)
    label = P.scale_uv(label, 1.0, 1.0 / (h - 0.034))
    b.visual(body, f"{name}_alu", P.merge(body_mesh, P.lathe([(0.027, h), (0.0, h)], segments=36)), alu)
    b.visual(body, f"{name}_label", label, lab)
    return [("cyl", (0, 0, 0.012), (0, 0, h - 0.012), r)]


def add_bottle(b: Builder, body, name, p):
    r, h = BOTTLE["r"], BOTTLE["h"]
    b.prim(body, f"{name}_col", G.mjGEOM_CYLINDER, [r, h * 0.42, 0], (0, 0, h * 0.42), mass=BOTTLE["mass"],
           friction=(0.6, 0.005, 0.0001))
    b.prim(body, f"{name}_neck", G.mjGEOM_CYLINDER, [0.016, 0.06, 0], (0, 0, h * 0.84 + 0.06), mass=0.002)
    prof = [(0.0, 0.0), (0.026, 0.0), (r, 0.010), (r, 0.115), (0.031, 0.125), (r, 0.135), (r, 0.150),
            (0.026, 0.178), (0.016, 0.195), (0.0155, h)]
    pet = b.material("pet", (0.80, 0.90, 0.97, 0.38))
    b.visual(body, f"{name}_pet", P.lathe(prof[1:], segments=36, cap_bottom=True), pet)
    lab = b.material("label_water", tex="label_water")
    label = P.scale_uv(P.lathe([(r + 0.0005, 0.060), (r + 0.0005, 0.108)], segments=36), 1.0, 1 / 0.048)
    b.visual(body, f"{name}_label", label, lab)
    cap = b.material("cap_white", (0.94, 0.95, 0.96, 1))
    b.visual(body, f"{name}_cap", P.lathe([(0.0, h), (0.0155, h), (0.0155, h + 0.022), (0.0, h + 0.022)],
                                          segments=24), cap)
    return [("cyl", (0, 0, 0.012), (0, 0, 0.16), r)]


def add_napkin(b: Builder, body, name, p):
    r = NAPKIN["r"]
    # a crumpled napkin does not roll: a box of its bulk is its collider
    b.prim(body, f"{name}_col", G.mjGEOM_BOX, [r * 0.80, r * 0.76, r * 0.78], (0, 0, 0), mass=NAPKIN["mass"],
           friction=(0.8, 0.01, 0.0002))
    mat = b.material("napkin_paper", tex="paper")
    b.visual(body, f"{name}_vis", P.crumpled_ball(r, seed=100 + int(p.get("seed", 0))), mat)
    return [("ball", (0, 0, 0), r)]


def add_apple(b: Builder, body, name, p):
    r, hl = APPLE["r"], APPLE["half"]
    # lying on its side: the core's axis along the body's x
    q = _quat_from_R(np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], float))
    # a bitten core does not roll: a box of its bulk is its collider
    b.prim(body, f"{name}_col", G.mjGEOM_BOX, [hl + r * 0.8, r * 0.85, r * 0.9], (0, 0, 0), mass=APPLE["mass"],
           friction=(0.8, 0.01, 0.0002))
    prof = [(0.012, -0.040), (0.026, -0.030), (0.017, -0.012), (0.014, 0.0), (0.017, 0.012), (0.026, 0.030),
            (0.012, 0.040)]
    mat = b.material("apple_core", tex="apple")
    mesh = P.scale_uv(P.lathe(prof, segments=24, cap_bottom=True, cap_top=True), 1.0, 1 / 0.09)
    b.visual(body, f"{name}_vis", mesh, mat, quat=q)
    stem = b.material("stem", (0.30, 0.20, 0.10, 1))
    b.prim(body, f"{name}_stem", G.mjGEOM_CAPSULE, [0.0025, 0.008, 0], (0.048, 0, 0.0), q, visual=True,
           material=stem)
    return [("ball", (0, 0, 0), 0.03)]


def add_plate(b: Builder, body, name, p):
    R, h = PLATE["r"], PLATE["h"]
    b.prim(body, f"{name}_col", G.mjGEOM_CYLINDER, [R, h / 2, 0], (0, 0, h / 2), mass=PLATE["mass"],
           friction=(0.5, 0.005, 0.0001))
    prof = [(0.0, 0.006), (0.080, 0.006), (0.084, 0.0), (0.090, 0.0), (0.094, 0.006), (0.118, 0.015),
            (R, 0.021), (R - 0.004, h), (0.112, 0.0185), (0.088, 0.0095), (0.0, 0.0095)]
    mat = b.material("porcelain", (0.96, 0.96, 0.95, 1))
    b.visual(body, f"{name}_vis", P.lathe(prof, segments=48), mat)
    rim = b.material("plate_rim", (0.28, 0.38, 0.52, 1))
    b.visual(body, f"{name}_band", P.lathe([(R - 0.010, 0.0195), (R - 0.0035, 0.0212)], segments=48), rim)
    return [("disc", (0, 0, h / 2), R, h / 2)]


def add_glass(b: Builder, body, name, p):
    r, h = GLASS["r"], GLASS["h"]
    b.prim(body, f"{name}_col", G.mjGEOM_CYLINDER, [0.033, h / 2, 0], (0, 0, h / 2), mass=GLASS["mass"],
           friction=(0.5, 0.005, 0.0001))
    outer = [(0.0, 0.0), (0.030, 0.0), (0.031, 0.012), (r, h)]
    inner = [(r - 0.003, h), (0.028, 0.016), (0.0, 0.016)]
    mat = b.material("glass", (0.88, 0.94, 0.97, 0.22))
    b.visual(body, f"{name}_vis", P.lathe(outer[1:] + inner, segments=40, cap_bottom=True), mat)
    return [("cyl", (0, 0, 0.012), (0, 0, h - 0.008), r)]


def add_book(b: Builder, body, name, p):
    w, t, hgt = p["size"]                    # x = width (spine at -x), y = thickness, z = height (head at +z)
    b.prim(body, f"{name}_col", G.mjGEOM_BOX, [w / 2, t / 2, hgt / 2], mass=p["mass"], friction=(0.6, 0.005, 0.0001))
    cover = b.material(f"cover_{p['tex']}", tex=p["tex"])
    pages = b.material("pages", (0.93, 0.90, 0.80, 1))
    # the cover: a thin shell round the box (front, back and spine), the page block inset
    shell = P.rounded_box([w / 2 + 0.0005, t / 2 + 0.0005, hgt / 2 + 0.0005], radius=0.0015, n=2)
    shell = P.scale_uv(shell, 1 / w, 1 / hgt)
    b.visual(body, f"{name}_cover", shell, cover)
    b.visual(body, f"{name}_pages", P.rounded_box([w / 2 - 0.002, t / 2 - 0.0015, hgt / 2 - 0.0025], 0.0, 1), pages,
             pos=(0.0035, 0, 0))
    return [("slab", (0, 0, 0), (w / 2, t / 2, hgt / 2), 1)]


def add_sponge(b: Builder, body, name, p):
    (lx, ly, lz), pad = SPONGE["size"], SPONGE["pad"]
    b.prim(body, f"{name}_col", G.mjGEOM_BOX, [lx / 2, ly / 2, lz / 2], mass=SPONGE["mass"],
           friction=(0.9, 0.01, 0.0002))
    # foam (top, yellow) and scouring pad (bottom, green): drawn-only boxes without a material, so the cell can colour
    # the foam as it loads up with grease (geom rgba)
    b.prim(body, f"{name}_foam", G.mjGEOM_BOX, [lx / 2 - 0.0008, ly / 2 - 0.0008, (lz - pad) / 2],
           (0, 0, pad / 2), visual=True, rgba=(0.96, 0.80, 0.22, 1))
    b.prim(body, f"{name}_pad", G.mjGEOM_BOX, [lx / 2, ly / 2, pad / 2], (0, 0, -(lz - pad) / 2), visual=True,
           rgba=(0.16, 0.42, 0.22, 1))
    b.prim(body, f"{name}_suds", G.mjGEOM_ELLIPSOID, [0.026, 0.017, 0.004], (0.008, 0.0, lz / 2), visual=True,
           rgba=(0.97, 0.98, 0.99, 0.0))
    # a slab both ways: pinched through its thickness at an edge, or across its width (from above)
    return [("slab", (0, 0, 0), (lx / 2, ly / 2, lz / 2), 2), ("slab", (0, 0, 0), (lx / 2, ly / 2, lz / 2), 1)]


def add_broom(b: Builder, body, name, p):
    """The broom: `body` is the handle (its frame: origin on the head's swivel axis, z along the handle); the head
    swivels about its long axis (x) on a child body, so its bristles stay flat on the floor as the handle tilts."""
    r = BROOM["handle_r"]
    z0, z1 = BROOM["handle"]
    b.prim(body, f"{name}_handle_col", G.mjGEOM_CAPSULE, [r, (z1 - z0) / 2, 0], (0, 0, (z0 + z1) / 2),
           mass=BROOM["mass_handle"], friction=(0.6, 0.005, 0.0001))
    alu = b.material("broom_alu", (0.70, 0.72, 0.75, 1))
    grip = b.material("grip_black", (0.06, 0.06, 0.07, 1))
    b.visual(body, f"{name}_handle", P.lathe([(r, z0), (r, z1 - 0.16)], segments=20), alu)
    b.visual(body, f"{name}_grip", P.lathe([(r + 0.002, z1 - 0.16), (r + 0.002, z1 - 0.005), (r - 0.002, z1),
                                            (0.0, z1)], segments=20), grip)
    b.visual(body, f"{name}_hanger", P.lathe([(0.0, z1), (0.006, z1), (0.006, z1 + 0.012), (0.0, z1 + 0.012)],
                                             segments=12), grip)
    b.visual(body, f"{name}_socket", P.lathe([(0.016, 0.02), (0.014, z0 + 0.03), (r, z0 + 0.05)], segments=20),
             b.material("broom_red", (0.72, 0.10, 0.10, 1)))
    head = body.add_body(name=f"{name}_head", pos=[0, 0, 0])
    head.add_joint(name=f"{name}_swivel", type=mujoco.mjtJoint.mjJNT_HINGE, axis=[1, 0, 0],
                   range=[-BROOM["swivel"], BROOM["swivel"]], limited=mujoco.mjtLimited.mjLIMITED_TRUE,
                   damping=0.5, stiffness=0.05, armature=0.0005)
    hx, hy, hz = BROOM["head"]
    bx, by, bz = BROOM["bristle"]
    red = b.material("broom_red", (0.72, 0.10, 0.10, 1))
    bristle = b.material("bristle", (0.13, 0.12, 0.12, 1))
    b.prim(head, f"{name}_block_col", G.mjGEOM_BOX, [hx / 2, hy / 2, hz / 2], (0, 0, -hz / 2),
           mass=BROOM["mass_head"] * 0.6, friction=(0.6, 0.005, 0.0001))
    # bristles: soft (a compliant contact) and slippery on the floor, as real ones are
    b.prim(head, f"{name}_bristle_col", G.mjGEOM_BOX, [bx / 2, by / 2, bz / 2], (0, 0, -hz - bz / 2),
           mass=BROOM["mass_head"] * 0.4, friction=(0.1, 0.005, 0.0001), solref=[0.04, 1.0],
           solimp=[0.6, 0.9, 0.01, 0.5, 2.0])
    b.visual(head, f"{name}_block", P.rounded_box([hx / 2, hy / 2, hz / 2], 0.008, 4), red, pos=(0, 0, -hz / 2))
    b.visual(head, f"{name}_bristles", P.rounded_box([bx / 2, by / 2, bz / 2], 0.004, 3), bristle,
             pos=(0, 0, -hz - bz / 2))
    # a few tufts of lighter bristles along the edge (looks)
    tuft = b.material("bristle_tip", (0.55, 0.50, 0.42, 1))
    b.prim(head, f"{name}_tips", G.mjGEOM_BOX, [bx / 2 - 0.004, by / 2 + 0.001, 0.006], (0, 0, -hz - bz + 0.006),
           visual=True, material=tuft)
    return [("cyl", (0, 0, z0 + 0.08), (0, 0, z1 - 0.01), r)]


def add_dustpan(b: Builder, body, name, p):
    """The lobby dustpan: `body` frame at the middle of the lip (the pan's front edge, on the floor when the pan rests
    flat), x across the pan, y from the lip to the back wall, z up; the handle rises from the back wall's top,
    leaning back by 8 deg from the vertical (the dustpan stands on its own on a level floor)."""
    W, D, H, t = DUSTPAN["width"], DUSTPAN["depth"], DUSTPAN["wall_h"], DUSTPAN["t"]
    grey = b.material("pan_grey", (0.34, 0.36, 0.38, 1))
    lip = b.material("pan_lip", (0.05, 0.05, 0.05, 1))
    mp = DUSTPAN["mass_pan"]
    b.prim(body, f"{name}_floor_col", G.mjGEOM_BOX, [W / 2, D / 2, t / 2], (0, D / 2, t / 2), mass=mp * 0.45,
           friction=(0.5, 0.005, 0.0001))
    b.prim(body, f"{name}_back_col", G.mjGEOM_BOX, [W / 2, t / 2, H / 2], (0, D - t / 2, H / 2), mass=mp * 0.25)
    for sgn in (-1, 1):
        b.prim(body, f"{name}_side{'pn'[sgn > 0]}_col", G.mjGEOM_BOX, [t / 2, D * 0.36, H * 0.34],
               (sgn * (W / 2 - t / 2), D * 0.62, H * 0.34), mass=mp * 0.15)
    b.visual(body, f"{name}_floor", P.rounded_box([W / 2, D / 2, t / 2], 0.0015, 2), grey, pos=(0, D / 2, t / 2))
    b.visual(body, f"{name}_back", P.rounded_box([W / 2, t / 2, H / 2], 0.0015, 2), grey, pos=(0, D - t / 2, H / 2))
    for sgn in (-1, 1):
        b.visual(body, f"{name}_side{'pn'[sgn > 0]}", P.rounded_box([t / 2, D * 0.36, H * 0.34], 0.0015, 2), grey,
                 pos=(sgn * (W / 2 - t / 2), D * 0.62, H * 0.34))
    b.prim(body, f"{name}_lip", G.mjGEOM_BOX, [W / 2, 0.012, 0.0012], (0, 0.012, 0.0012), visual=True, material=lip)
    tilt = DUSTPAN["handle_tilt"]
    L, r = DUSTPAN["handle_len"], DUSTPAN["handle_r"]
    base = np.array([0.0, D - 0.01, H])
    axis = np.array([0.0, math.sin(tilt), math.cos(tilt)])
    mid = base + axis * L / 2
    qh = _quat_from_R(_rot_x(-tilt))
    b.prim(body, f"{name}_handle_col", G.mjGEOM_CAPSULE, [r, L / 2, 0], tuple(mid), qh, mass=DUSTPAN["mass_handle"])
    alu = b.material("broom_alu", (0.70, 0.72, 0.75, 1))
    grip = b.material("grip_black", (0.06, 0.06, 0.07, 1))
    hmesh = P.transform(P.lathe([(r, 0.0), (r, L - 0.14)], segments=20), _rot_x(-tilt), base)
    gmesh = P.transform(P.lathe([(r + 0.002, L - 0.14), (r + 0.002, L - 0.004), (0.0, L)], segments=20),
                        _rot_x(-tilt), base)
    b.visual(body, f"{name}_handle", hmesh, alu)
    b.visual(body, f"{name}_grip", gmesh, grip)
    return [("cyl", tuple(base + axis * 0.04), tuple(base + axis * (L - 0.01)), r)]


BUILDERS = {"can": add_can, "bottle": add_bottle, "napkin": add_napkin, "apple": add_apple, "plate": add_plate,
            "glass": add_glass, "book": add_book, "sponge": add_sponge, "broom": add_broom, "dustpan": add_dustpan}


def dustpan_handle_axis():
    """(base point, unit axis) of the dustpan's handle in the dustpan's frame."""
    D, H, tilt = DUSTPAN["depth"], DUSTPAN["wall_h"], DUSTPAN["handle_tilt"]
    return np.array([0.0, D - 0.01, H]), np.array([0.0, math.sin(tilt), math.cos(tilt)])


# ================================================================================================ fixtures' moving parts


def _open_shell(hx, hy, h, wt, rc, nc=6, c_in=0.002):
    """A bin's body as drawn: the rounded box the bins were drawn as (P.rounded_box, radius rc, 4 subdivisions: its
    walls, bottom and rounded edges kept vertex for vertex, so that a bin with its lid shut looks as before), without
    its top face: from the rounded edge a rim of the wall's thickness, the inner wall down to the bottom collider's
    top face, and the inside's floor. Centred like the box (x in [-hx, hx], y in [-hy, hy], z in [-h/2, h/2]). (The
    closed box drew a bin as a solid block: an open lid showed a solid top and litter vanished into it, while the
    colliders are open walls.) Returns (the outside: walls, bottom, rim; the inside: inner wall and floor), each
    (V, F, N, UV)."""
    V, F, N, UV = P.rounded_box([hx, hy, h / 2], rc, 4)
    F = F[:-2 * 4 * 4]                                # (the top face: the last face rounded_box builds, 2 n^2 triangles)
    used = np.unique(F)
    remap = -np.ones(len(V), int)
    remap[used] = np.arange(len(used))
    V, N, UV, F = V[used], N[used], UV[used], remap[F].astype(np.int32)
    # the open edge round the top: by position (each face of the box has its own vertices, for its uv), the edges of
    # one triangle only, chained into a loop
    from collections import Counter
    key = lambda i: tuple(np.round(V[i], 7))
    E_ = Counter()
    for t in F:
        for a, b in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
            ka, kb = key(a), key(b)
            E_[(min(ka, kb), max(ka, kb))] += 1
    adj = {}
    for (ka, kb), c in E_.items():
        if c == 1:
            adj.setdefault(ka, []).append(kb)
            adj.setdefault(kb, []).append(ka)
    cur = min(adj, key=lambda q: math.atan2(q[1], q[0]) % (2 * math.pi))
    loop, prev = [cur], None
    while True:
        nx = [q for q in adj[cur] if q != prev]
        if not nx or nx[0] == loop[0]:
            break
        prev, cur = cur, nx[0]
        loop.append(cur)
    O_ = np.array(loop, float)
    # (counter-clockwise from above: the signed area of the loop)
    if float(np.sum(O_[:, 0] * np.roll(O_[:, 1], -1) - np.roll(O_[:, 0], -1) * O_[:, 1])) < 0:
        O_ = O_[::-1]

    def ring(d, z):
        ax, ay, r = hx - d, hy - d, max(rc - d, 0.002)
        pts, nrm = [], []
        for cx, cy, a0 in ((ax - r, ay - r, 0.0), (-(ax - r), ay - r, 90.0), (-(ax - r), -(ay - r), 180.0),
                           (ax - r, -(ay - r), 270.0)):
            for k in range(nc + 1):
                a = math.radians(a0 + 90.0 * k / nc)
                pts.append((cx + r * math.cos(a), cy + r * math.sin(a), z))
                nrm.append((math.cos(a), math.sin(a)))
        P_ = np.array(pts, float)
        seg = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(P_[:, :2], axis=0), axis=1))]
        return P_, np.array(nrm, float), seg
    z1 = h / 2
    zin = -h / 2 + 0.03 + wt                          # the inside's floor: the bottom collider's top face
    Vs, Fs, Ns, UVs = [V], [F], [N], [UV]
    base = len(V)
    # the rim: the open edge zipped to the inner wall's top (by angle round the centre), facing up
    I_, _, _ = ring(wt, z1 - c_in)
    ang = lambda Q: np.arctan2(Q[:, 1], Q[:, 0]) % (2 * math.pi)
    ao, ai = ang(O_), ang(I_)
    so, si = int(np.argmin(ao)), int(np.argmin(ai))
    O2, I2 = np.roll(O_, -so, axis=0), np.roll(I_, -si, axis=0)
    ao, ai = ang(O2), ang(I2)
    no, ni = len(O2), len(I2)
    Vs.append(O2); Vs.append(I2)
    Ns.append(np.tile([0.0, 0.0, 1.0], (no + ni, 1)))
    UVs.append(np.stack([np.r_[O2[:, 0], I2[:, 0]] + hx, np.r_[O2[:, 1], I2[:, 1]] + hy], axis=1))
    i, j, tri = 0, 0, []
    while i < no or j < ni:
        ao_n = ao[(i + 1) % no] + (2 * math.pi if i + 1 >= no else 0.0)
        ai_n = ai[(j + 1) % ni] + (2 * math.pi if j + 1 >= ni else 0.0)
        if j >= ni or (i < no and ao_n <= ai_n):
            tri.append((base + i % no, base + (i + 1) % no, base + no + j % ni))
            i += 1
        else:
            tri.append((base + i % no, base + no + (j + 1) % ni, base + no + j % ni))
            j += 1
    Fs.append(np.array(tri, np.int32))
    base += no + ni
    # the inner wall, facing in, and the inside's floor, facing up (drawn apart: their own material)
    n_outer = sum(len(f_) for f_ in Fs)
    A, nA, sA = ring(wt, z1 - c_in)
    B, _, _ = ring(wt, zin)
    M = len(A)
    Vs.append(np.vstack([A, B]))
    nn = np.concatenate([-nA, np.zeros((M, 1))], axis=1)
    Ns.append(np.vstack([nn, nn]))
    UVs.append(np.stack([np.r_[sA, sA], np.r_[A[:, 2], B[:, 2]] + h / 2], axis=1))
    tri = []
    for k in range(M):
        a, b_, c, d = base + k, base + (k + 1) % M, base + M + (k + 1) % M, base + M + k
        tri += [(a, b_, c), (a, c, d)]
    Fs.append(np.array(tri, np.int32))
    base += 2 * M
    Vs.append(np.vstack([[0.0, 0.0, zin], B]))
    Ns.append(np.tile([0.0, 0.0, 1.0], (M + 1, 1)))
    UVs.append(np.stack([np.r_[0.0, B[:, 0]] + hx, np.r_[0.0, B[:, 1]] + hy], axis=1))
    Fs.append(np.array([(base, base + 1 + k, base + 1 + (k + 1) % M) for k in range(M)], np.int32))
    V2, N2, UV2 = np.concatenate(Vs), np.concatenate(Ns), np.concatenate(UVs)
    F2 = np.concatenate(Fs).astype(np.int32)
    # winding of the added parts: each triangle's normal along its vertices' normals (the box's faces are kept)
    nb = len(F)
    e = np.cross(V2[F2[:, 1]] - V2[F2[:, 0]], V2[F2[:, 2]] - V2[F2[:, 0]])
    flip = (e * (N2[F2[:, 0]] + N2[F2[:, 1]] + N2[F2[:, 2]])).sum(1) < 0
    flip[:nb] = False
    F2[flip] = F2[flip][:, [0, 2, 1]]

    def part(T):
        used = np.unique(T)
        rm = -np.ones(len(V2), int)
        rm[used] = np.arange(len(used))
        return V2[used], rm[T].astype(np.int32), N2[used], UV2[used]
    return part(F2[:n_outer]), part(F2[n_outer:])


def add_bin(b: Builder, wb, slot: int, y_c: float, kind: str):
    """One flip-lid bin (kind "recycling" = blue, "general" = grey) with its back to the west wall, centred at y_c.
    Returns {"body", "lid", "lid_joint", "features"}."""
    B = RM.BIN
    x0, x1, hy, rim, wt = B["x0"], B["x1"], B["half_y"], B["rim_z"], B["wall_t"]
    body = wb.add_body(name=f"bin{slot}", pos=[0, y_c, 0])
    tex = "bin_blue" if kind == "recycling" else "bin_grey"
    mat = b.material(f"{tex}_mat", tex=tex)
    xc = (x0 + x1) / 2
    hx = (x1 - x0) / 2
    walls = {"bottom": ([hx, hy, wt / 2], (xc, 0, 0.03 + wt / 2)),
             "front": ([wt / 2, hy, rim / 2], (x1 - wt / 2, 0, rim / 2)),
             "back": ([wt / 2, hy, rim / 2], (x0 + wt / 2, 0, rim / 2)),
             "south": ([hx, wt / 2, rim / 2], (xc, -hy + wt / 2, rim / 2)),
             "north": ([hx, wt / 2, rim / 2], (xc, hy - wt / 2, rim / 2))}
    for wn, (half, pos) in walls.items():
        b.prim(body, f"bin{slot}_{wn}_col", G.mjGEOM_BOX, half, pos, contact=FIX_CONTACT,
               friction=(0.6, 0.005, 0.0001))
    shell, inner = _open_shell(hx, hy, rim, wt, 0.012)
    b.visual(body, f"bin{slot}_shell", P.scale_uv(shell, 2.0, 2.0), mat, pos=(xc, 0, rim / 2))
    dark = b.material("bin_inside", (0.10, 0.10, 0.11, 1))
    b.prim(body, f"bin{slot}_inside", G.mjGEOM_BOX, [hx - wt, hy - wt, 0.002], (xc, 0, 0.035 + wt), visual=True,
           material=dark)
    if kind == "recycling":
        badge = b.material("badge_mat", tex="badge")
        q = _quat_from_R(np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], float))
        badge_mesh = P.scale_uv(P.rounded_box([0.0006, 0.07, 0.07], 0.0, 1), 1 / 0.14, 1 / 0.14)
        b.visual(body, f"bin{slot}_badge", badge_mesh, badge, pos=(x1 + 0.0015, 0, rim * 0.62), quat=(1, 0, 0, 0))
        _ = q
    # the lid: hinged at the back, its frame at the hinge (x along the lid towards the front)
    lt = B["lid_t"]
    L = x1 - B["hinge_x"] + 0.01
    lid = body.add_body(name=f"bin{slot}_lid", pos=[B["hinge_x"], 0, rim + 0.002])
    lid.add_joint(name=f"bin{slot}_lid_hinge", type=mujoco.mjtJoint.mjJNT_HINGE, axis=[0, -1, 0],
                  range=list(RM.LID_RANGE), limited=mujoco.mjtLimited.mjLIMITED_TRUE, damping=0.15, armature=0.002)
    b.prim(lid, f"bin{slot}_lid_col", G.mjGEOM_BOX, [L / 2, hy + 0.006, lt / 2], (L / 2, 0, lt / 2),
           contact=FIX_CONTACT, mass=0.25, friction=(0.6, 0.005, 0.0001))
    lidmat = b.material(f"{tex}_lid", tex=tex)
    b.visual(lid, f"bin{slot}_lid_vis", P.scale_uv(P.rounded_box([L / 2, hy + 0.006, lt / 2], 0.005, 3), 2, 2), lidmat,
             pos=(L / 2, 0, lt / 2))
    tab_x = L + B["tab_out"] - B["tab_r"]
    q = _quat_from_R(np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], float))     # cylinder axis along y
    b.prim(lid, f"bin{slot}_tab_col", G.mjGEOM_CAPSULE, [B["tab_r"], B["tab_len"] / 2, 0], (tab_x, 0, lt / 2), q,
           contact=FIX_CONTACT, mass=0.01)
    b.prim(lid, f"bin{slot}_tab", G.mjGEOM_CAPSULE, [B["tab_r"], B["tab_len"] / 2, 0], (tab_x, 0, lt / 2), q,
           visual=True, material=b.material("tab_dark", (0.12, 0.12, 0.13, 1)))
    feats = [("cyl", (tab_x, -B["tab_len"] / 2, lt / 2), (tab_x, B["tab_len"] / 2, lt / 2), B["tab_r"])]
    return {"body": body, "lid": lid, "features": feats, "inner": inner, "pos": (xc, y_c, rim / 2), "tex": tex}


def add_bin_inside(b: Builder, wb, slot: int, rec: dict):
    """A bin's inside as drawn (its inner wall and floor, `_open_shell`): a body of its own, added after every
    other body of the cell, so that the geoms of the model before it keep their indices (a recorded episode's
    drawn-state changes, by geom index, replay on it); its own material, so that the cameras' look can darken the
    inside the way the path-traced look does."""
    body = wb.add_body(name=f"bin{slot}_inside_drawn", pos=list(rec["pos"]))
    mat = b.material(f"{rec['tex']}_inner", tex=f"{rec['tex']}_inner")
    b.visual(body, f"bin{slot}_inner", P.scale_uv(rec["inner"], 2.0, 2.0), mat)


def add_faucet(b: Builder, wb):
    F = RM.FAUCET
    bx, by = F["base"]
    top = RM.KITCHEN["top_z"]
    chrome = b.material("chrome", (0.80, 0.81, 0.83, 1))
    body = wb.add_body(name="faucet", pos=[bx, by, top])
    col_h = F["column_top"] - top
    b.prim(body, "faucet_column_col", G.mjGEOM_CYLINDER, [F["column_r"], col_h / 2, 0], (0, 0, col_h / 2),
           contact=FIX_CONTACT)
    b.visual(body, "faucet_column", P.lathe([(0.030, 0.0), (0.030, 0.006), (F["column_r"], 0.012),
                                             (F["column_r"], col_h), (0.0, col_h)], segments=28), chrome)
    # the spout: out of the column's front at 0.07 m above the counter, a low arch over to the outlet above the basin
    # (below the lever)
    ox, oy, oz = F["outlet"]
    path = [(0.0, 0.0, 0.07), (0.03, 0.0, 0.115), (0.08, 0.0, 0.135), (0.14, 0.0, 0.138), (0.19, 0.0, 0.130),
            (ox - bx - 0.01, 0.0, oz - top + 0.022), (ox - bx, 0.0, oz - top + 0.010)]
    b.visual(body, "faucet_spout", P.tube(path, 0.012, segments=18), chrome)
    for i in range(len(path) - 1):
        a, c = np.array(path[i]), np.array(path[i + 1])
        mid, d = (a + c) / 2, c - a
        q = np.zeros(4)
        mujoco.mju_quatZ2Vec(q, d / np.linalg.norm(d))
        b.prim(body, f"faucet_spout_col{i}", G.mjGEOM_CAPSULE, [0.012, np.linalg.norm(d) / 2, 0], tuple(mid),
               tuple(q), contact=FIX_CONTACT)
    px, py, pz = F["pivot"]
    # gravity compensated: the lever stays where it is left (MuJoCo's dry friction alone creeps under its weight)
    lever = body.add_body(name="faucet_lever", pos=[px - bx, py - by, pz - top], gravcomp=1.0)
    lever.add_joint(name="faucet_lever_hinge", type=mujoco.mjtJoint.mjJNT_HINGE, axis=[0, -1, 0],
                    range=list(F["range"]), limited=mujoco.mjtLimited.mjLIMITED_TRUE, damping=0.02,
                    frictionloss=0.08, armature=0.0005)
    Ll, rl = F["lever_len"], F["lever_r"]
    q = _quat_from_R(np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], float))     # capsule axis along x
    b.prim(lever, "faucet_lever_col", G.mjGEOM_CAPSULE, [rl, Ll / 2, 0], (Ll / 2, 0, 0), q, contact=FIX_CONTACT,
           mass=0.04)
    b.prim(lever, "faucet_cap", G.mjGEOM_CYLINDER, [0.024, 0.012, 0], (0, 0, -0.004), visual=True, material=chrome)
    b.visual(lever, "faucet_lever_vis", P.transform(P.lathe([(0.0, 0.0), (0.0085, 0.004), (rl, Ll - 0.012),
                                                             (0.0095, Ll - 0.006), (0.0, Ll)], segments=16), q_to_R(q)),
             chrome)
    return {"body": body, "lever": lever,
            "features": [("cyl", (0.03, 0, 0), (Ll, 0, 0), rl)]}


def q_to_R(q):
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, np.asarray(q, float))
    return R.reshape(3, 3)


def add_soap(b: Builder, wb):
    """The dish-soap pump with a sponge rest: a squat bottle whose wide, flat pump head is where the sponge is pressed
    down for a dose."""
    S = RM.SOAP
    sx, sy = S["base"]
    top = RM.KITCHEN["top_z"]
    body = wb.add_body(name="soap_bottle", pos=[sx, sy, top])
    r, h = S["bottle_r"], S["bottle_h"]
    b.prim(body, "soap_bottle_col", G.mjGEOM_CYLINDER, [r, h / 2, 0], (0, 0, h / 2), contact=FIX_CONTACT)
    green = b.material("soap_green", (0.42, 0.72, 0.38, 0.62))
    liquid = b.material("soap_liquid", (0.30, 0.62, 0.22, 0.85))
    b.visual(body, "soap_bottle_vis", P.lathe([(0.0, 0.0), (r - 0.004, 0.0), (r, 0.006), (r, h - 0.012),
                                               (r - 0.010, h)], segments=32, cap_bottom=True), green)
    b.visual(body, "soap_liquid_vis", P.lathe([(0.0, 0.004), (r - 0.006, 0.004), (r - 0.006, 0.62 * h),
                                               (0.0, 0.62 * h)], segments=28), liquid)
    white = b.material("pump_white", (0.95, 0.95, 0.94, 1))
    neck = S["neck_top"] - top
    b.prim(body, "soap_collar", G.mjGEOM_CYLINDER, [0.018, (neck - h) / 2, 0], (0, 0, (neck + h) / 2), visual=True,
           material=white)
    head = body.add_body(name="soap_pump", pos=[0, 0, neck])
    head.add_joint(name="soap_pump_slide", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[0, 0, 1],
                   range=[-S["travel"], 0.0], limited=mujoco.mjtLimited.mjLIMITED_TRUE, stiffness=90.0,
                   springref=0.004, damping=1.5, armature=0.002)
    hr, hh = S["head_r"], S["head_h"]
    b.prim(head, "soap_pump_col", G.mjGEOM_CYLINDER, [hr, hh / 2, 0], (0, 0, hh / 2 + 0.010), contact=FIX_CONTACT,
           mass=0.02, friction=(0.9, 0.01, 0.0002))
    b.prim(head, "soap_stem", G.mjGEOM_CYLINDER, [0.008, 0.006, 0], (0, 0, 0.005), visual=True, material=white)
    b.visual(head, "soap_pump_vis", P.lathe([(0.0, 0.010), (hr, 0.010), (hr, 0.010 + hh - 0.003),
                                             (hr - 0.004, 0.010 + hh), (0.0, 0.010 + hh)], segments=32), white)
    ridged = b.material("pump_pad", (0.80, 0.86, 0.82, 1))
    b.prim(head, "soap_pad", G.mjGEOM_CYLINDER, [hr - 0.006, 0.0008, 0], (0, 0, 0.010 + hh + 0.0008), visual=True,
           material=ridged)
    return {"body": body, "head": head, "features": [("ball", (0, 0, 0.010 + hh / 2), 0.04)]}
