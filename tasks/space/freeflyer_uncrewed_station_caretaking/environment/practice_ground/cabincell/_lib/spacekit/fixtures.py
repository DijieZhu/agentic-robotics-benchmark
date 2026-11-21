"""MJCF fragments of the cabin mechanics (neutral names): crew handrail of the
measured stadium cross-section, loose items with graspable handrail-profile handles (rigid plate, hinged soft item
with a centre handle and / or a corner loop, small item), a stowage bag with a spring flap (one flap or a split pair), a hook-and-loop board,
and a helper that wraps fragments into a stand-alone model for ``MjSpec.attach``.

Measured / published values: handrail section stadium 35.0 mm (off the wall) x 19.0 mm (along the wall),
bar inner face 43.9 mm off the wall (public handrail mesh); the gripper needs that gap to wrap behind a bar, so the
item handles copy it.  Stowage bag rules: flap preload <= ~2.5 mN m, pocket deeper than the flap is long,
mouth >= 2 cm larger than the item on each side, aim within ~2 cm.  Contact recipe: solref 0.004 1, solimp
0.95 0.99 0.001, condim 4 (dt 2 ms, implicitfast, elliptic cone, impratio 10).
"""
from __future__ import annotations

import numpy as np

__all__ = ["RAIL_DEPTH", "RAIL_WIDTH", "RAIL_STANDOFF", "CONTACT", "MATERIALS", "STANDARD_BAG", "handrail", "handle_geoms",
           "plate_item", "soft_item", "small_item", "stowage_bag", "standard_bag", "fiducial_plate", "hook_loop_board",
           "fragment_model", "frame_axes"]

RAIL_DEPTH = 0.0350      # m, stadium section normal to the wall  [V public handrail mesh]
RAIL_WIDTH = 0.0190      # m, along the wall
RAIL_STANDOFF = 0.0439   # m, bar inner face off the wall
CONTACT = 'solref="0.004 1" solimp="0.95 0.99 0.001" condim="4" priority="1"'
MATERIALS = """
    <material name="mat_rail" rgba="0.16 0.32 0.40 1" specular="0.5"/>
    <material name="mat_item_soft" rgba="0.90 0.55 0.20 1"/>
    <material name="mat_item_bag" rgba="0.15 0.45 0.85 1"/>
    <material name="mat_item_dark" rgba="0.10 0.10 0.12 1" specular="0.6"/>
    <material name="mat_handle" rgba="0.90 0.10 0.10 1"/>
    <material name="mat_pouch" rgba="0.35 0.45 0.30 1"/>
    <material name="mat_flap" rgba="0.30 0.38 0.26 0.9"/>
    <material name="mat_hook_loop" rgba="0.95 0.95 0.95 1"/>
    <material name="mat_fid_black" rgba="0.03 0.03 0.03 1"/>
    <material name="mat_fid_white" rgba="0.96 0.96 0.94 1"/>
"""
# The standard stowage bag of the task station (the short-flap rule; v0.3).  The robot's arm is short:
# with its body 2.5 cm outside the mouth an item's centre reaches only ~0.14-0.19 m past the mouth, and a soft item
# held in the v1 fingers' cage turns about its handle while it is pushed in, so its end strips stop 0.04-0.09 m past
# the mouth: the flaps must be shorter than that (two 0.08 m flaps; with 0.09 m they rest on the end strips and push
# the item out after the release -- the mission variants), and the slit between their tips (0.20 m) must pass the arm
# and the open fingers (0.11 m across the bar, 0.09 m along it); the soft item (0.30 m) is wider than the slit, the
# 0.20 m pouch only as wide (it stays because the pocket is still air, aero.Shelters; a design that needs geometric
# retention of 0.20 m items narrows the mouth along b).  Insert with the handle bar along the slit (the bag's b axis);
# mouth 3 cm larger than the soft item on each side.  A fiducial plate beside the mouth (+a side wall, in the mouth
# plane) lets the robot position itself relative to the mouth to about 1 cm (primitives.FlyToFiducial).  [C]
STANDARD_BAG = dict(mouth_half=(0.18, 0.18), depth=0.24, flap_len=0.08, flap_k=0.0005, flap_ref=-1.2,
                    fiducial_size=(0.10, 0.06), fiducial_gap=0.012)


def _v(a) -> str:
    return " ".join(f"{x:.6g}" for x in np.ravel(a))


def frame_axes(normal, t1):
    """(n, a, b): unit normal, unit in-plane axis a (t1 projected), b = n x a."""
    n = np.asarray(normal, float) / np.linalg.norm(normal)
    a = np.asarray(t1, float) - (np.asarray(t1, float) @ n) * n
    a /= np.linalg.norm(a)
    return n, a, np.cross(n, a)


def handrail(name: str, center_on_wall, axis, normal, length: float = 0.546, legs: bool = True) -> tuple:
    """Static handrail: stadium bar (box core + two cylinders) on two stand-off legs.  center_on_wall: wall point
    below the bar centre; axis: bar direction; normal: out of the wall.  Returns (worldbody xml, bar centre)."""
    a = np.asarray(axis, float) / np.linalg.norm(axis)
    n = np.asarray(normal, float) / np.linalg.norm(normal)
    w = np.cross(n, a)
    c0 = np.asarray(center_on_wall, float)
    rad = RAIL_WIDTH / 2
    flat = (RAIL_DEPTH - RAIL_WIDTH) / 2
    cb = c0 + n * (RAIL_STANDOFF + RAIL_DEPTH / 2)
    L2 = length / 2
    xy = f'xyaxes="{_v(a)} {_v(w)}"'
    fr = 'friction="1.0 0.02 0.002"'
    g = [f'<geom name="{name}" type="box" pos="{_v(cb)}" size="{L2} {rad} {flat}" {xy} material="mat_rail" {fr} {CONTACT}/>',
         f'<geom name="{name}_c1" type="cylinder" fromto="{_v(cb + n * flat - a * L2)} {_v(cb + n * flat + a * L2)}" size="{rad}" material="mat_rail" {fr} {CONTACT}/>',
         f'<geom name="{name}_c2" type="cylinder" fromto="{_v(cb - n * flat - a * L2)} {_v(cb - n * flat + a * L2)}" size="{rad}" material="mat_rail" {fr} {CONTACT}/>']
    if legs:
        for k, sgn in ((0, -1), (1, 1)):
            pl = c0 + a * sgn * (L2 - 0.02) + n * (RAIL_STANDOFF / 2 + 0.002)
            g.append(f'<geom name="{name}_leg{k}" type="box" pos="{_v(pl)}" size="0.02 {rad} {RAIL_STANDOFF / 2 + 0.002}" {xy} material="mat_rail"/>')
    return "\n".join(g), cb


def handle_geoms(name: str, z0: float, sg: int = 1, length: float = 0.07, yoff: float = 0.0, xoff: float = 0.0,
                 material: str = "mat_handle") -> tuple:
    """Handrail-profile handle on an item face at local z = z0 (sg = +-1): stadium bar 19 mm x 35 mm, inner face
    43.9 mm off the face, axis along local x, two posts.  Returns (geom xml list, bar centre (local))."""
    r = RAIL_WIDTH / 2
    fl = (RAIL_DEPTH - RAIL_WIDTH) / 2
    hc = RAIL_STANDOFF + RAIL_DEPTH / 2
    zc = z0 + sg * hc
    L2 = length / 2
    fr = 'friction="1.0 0.02 0.002"'
    g = [f'<geom name="{name}" type="box" pos="{xoff} {yoff} {zc}" size="{L2} {r} {fl}" mass="0.006" material="{material}" {fr} {CONTACT}/>',
         f'<geom name="{name}_c1" type="cylinder" fromto="{xoff - L2} {yoff} {zc + fl} {xoff + L2} {yoff} {zc + fl}" size="{r}" mass="0.002" material="{material}" {fr} {CONTACT}/>',
         f'<geom name="{name}_c2" type="cylinder" fromto="{xoff - L2} {yoff} {zc - fl} {xoff + L2} {yoff} {zc - fl}" size="{r}" mass="0.002" material="{material}" {fr} {CONTACT}/>',
         f'<geom name="{name}_l" type="box" pos="{xoff - L2 - 0.004} {yoff} {z0 + sg * hc / 2}" size="0.004 {r} {hc / 2}" mass="0.002" material="{material}" {CONTACT}/>',
         f'<geom name="{name}_r" type="box" pos="{xoff + L2 + 0.004} {yoff} {z0 + sg * hc / 2}" size="0.004 {r} {hc / 2}" mass="0.002" material="{material}" {CONTACT}/>']
    return g, (xoff, yoff, zc)


def plate_item(name: str, pos, quat=(1, 0, 0, 0), size=(0.30, 0.20, 0.01), mass: float = 0.1, material: str = "mat_item_bag",
               friction: float = 0.4, handle: bool = True, handle_len: float = 0.07) -> tuple:
    """Rigid thin box with a handrail-profile handle in the middle of its +z face.  Returns (xml, half sizes)."""
    hx, hy, hz = (float(v) / 2 for v in size)
    s = [f'<body name="{name}" pos="{_v(pos)}" quat="{_v(quat)}">', f'  <freejoint name="{name}_root"/>',
         f'  <geom name="{name}_g" type="box" size="{hx} {hy} {hz}" mass="{mass}" material="{material}" friction="{friction} 0.01 0.001" {CONTACT}/>']
    if handle:
        gg, c = handle_geoms(f"{name}_handle", hz, 1, length=handle_len)
        s += ["  " + x for x in gg] + [f'  <site name="{name}_grasp" pos="{_v(c)}" size="0.004"/>']
    s.append("</body>")
    return "\n".join(s), (hx, hy, hz)


SOFT_ITEM_LOOPS = ("corner", "edge_mid")


def soft_item(name: str, pos, quat=(1, 0, 0, 0), n: int = 5, size=(0.30, 0.30, 0.004), mass: float = 0.10,
              hinge_k: float = 0.002, friction: float = 0.5, centre_handle: bool = True, corner_loop: bool = True,
              material: str = "mat_item_soft", loop: str | None = "default", loop_yoff: float | None = None) -> tuple:
    """Soft item: chain of n rigid strips along local x joined by hinges (local y axes).  Handles (handrail profile,
    on the +z face, bar along local x): a centre handle on the middle strip and / or a loop on strip 0.  Body names
    ``{name}0 .. {name}{n-1}``; sites ``{name}_grasp_centre`` / ``{name}_grasp_corner`` (the loop keeps this site name
    wherever it is).  loop (v0.4): 'corner' (near the +y corner, 0.04 m in from the edge: the v0.3 handle),
    'edge_mid' (the middle of the end strip, y = 0: the task's loop), None (no loop); the default follows
    ``corner_loop`` (v0.3).  loop_yoff: the loop's local y [m] (overrides the named position).
    Returns (xml, contact-exclude xml, strip half sizes)."""
    lx, ly, lz = size
    sw = lx / n
    m = mass / n
    kc = n // 2
    if loop == "default":
        loop = "corner" if corner_loop else None
    if loop is not None and loop not in SOFT_ITEM_LOOPS:
        raise ValueError(f"loop {loop!r} not in {SOFT_ITEM_LOOPS} or None")
    if loop is not None:
        yl = float(loop_yoff) if loop_yoff is not None else (ly / 2 - 0.04 if loop == "corner" else 0.0)

    def handles(i):
        out = []
        if centre_handle and i == kc:
            gg, c = handle_geoms(f"{name}_hc", lz / 2, 1, length=sw - 0.016)
            out += ["  " + x for x in gg] + [f'  <site name="{name}_grasp_centre" pos="{_v(c)}" size="0.004"/>']
        if loop is not None and i == 0:
            gg, c = handle_geoms(f"{name}_hk", lz / 2, 1, length=sw - 0.016, yoff=yl)
            out += ["  " + x for x in gg] + [f'  <site name="{name}_grasp_corner" pos="{_v(c)}" size="0.004"/>']
        return out

    fr = f'friction="{friction} 0.01 0.001"'
    s = [f'<body name="{name}0" pos="{_v(pos)}" quat="{_v(quat)}">', f'  <freejoint name="{name}_root"/>',
         f'  <geom name="{name}0_g" type="box" size="{sw / 2} {ly / 2} {lz / 2}" mass="{m}" material="{material}" {fr} {CONTACT}/>']
    s += handles(0)
    for i in range(1, n):
        ind = "  " * i
        s += [f'{ind}  <body name="{name}{i}" pos="{sw} 0 0">',
              f'{ind}    <joint name="{name}_h{i}" type="hinge" axis="0 1 0" pos="{-sw / 2} 0 0" stiffness="{hinge_k}" damping="0.002" range="-2.5 2.5"/>',
              f'{ind}    <geom name="{name}{i}_g" type="box" size="{sw / 2} {ly / 2} {lz / 2}" mass="{m}" material="{material}" {fr} {CONTACT}/>']
        s += [ind + x for x in handles(i)]
    s += ["  " * i + "  </body>" for i in range(n - 1, 0, -1)]
    s.append("</body>")
    excl = [f'<exclude body1="{name}{i}" body2="{name}{i + 1}"/>' for i in range(n - 1)]
    return "\n".join(s), "\n".join(excl), (sw / 2, ly / 2, lz / 2)


def small_item(name: str, pos, quat=(1, 0, 0, 0), size=(0.15, 0.10, 0.005), mass: float = 0.02, handle: bool = True,
               material: str = "mat_item_bag", friction: float = 0.6) -> tuple:
    """Small loose item (sample bag, pouch): thin box, optional handle on +z.  Returns (xml, half sizes)."""
    return plate_item(name, pos, quat, size, mass, material, friction, handle)


def stowage_bag(name: str, mouth_center, normal, t1, mouth_half=(0.18, 0.12), depth: float = 0.26,
                flap_k: float = 0.0005, flap_ref: float = -1.2, protrude: bool = True, wall_t: float = 0.004,
                split: bool = False, flap_len: float | None = None) -> tuple:
    """Pocket with a spring flap.  mouth_center: centre of the mouth plane (the opening faces +normal); the pocket
    extends ``depth`` behind the mouth (protrude=True: the caller puts the mouth ``depth`` off a wall).  The flap is
    hinged at the +b edge of the mouth (b = normal x t1), opens INTO the pocket only (0..1.45 rad), closed by a
    torsional spring with preload k * |springref| (default 0.6 mN m).
    split=True: short flaps hinged at the +b and -b edges (an elastic slit): flap_len = L (both edges) or
    (L_plus_b, L_minus_b) with 0 for no flap on that edge (default hy - 1 cm on both); the swing radius is the
    longest flap instead of the whole mouth, the gap between the tips (or tip and bare edge) is 2 * hy - L+ - L-;
    an item retained by it must be larger than the gap and pushed past the swing.
    Returns (static geoms xml, flap body xml, dict(mouth_center, normal, a, b, mouth_half, depth, preload_Nm,
    flap_joints, swing_radius, gap))."""
    n, a, b = frame_axes(normal, t1)
    mc = np.asarray(mouth_center, float)
    hx, hy = mouth_half
    back = mc - n * depth
    t = wall_t
    xy = f'xyaxes="{_v(a)} {_v(b)}"'
    c = f'{CONTACT} friction="0.3 0.01 0.001"'
    g = [f'<geom name="{name}_back" type="box" pos="{_v(back - n * t)}" size="{hx + 2 * t} {hy + 2 * t} {t}" {xy} material="mat_pouch" {c}/>']
    mid = mc - n * depth / 2
    g.append(f'<geom name="{name}_s1" type="box" pos="{_v(mid + a * (hx + t))}" size="{t} {hy} {depth / 2}" {xy} material="mat_pouch" {c}/>')
    g.append(f'<geom name="{name}_s2" type="box" pos="{_v(mid - a * (hx + t))}" size="{t} {hy} {depth / 2}" {xy} material="mat_pouch" {c}/>')
    g.append(f'<geom name="{name}_s3" type="box" pos="{_v(mid + b * (hy + t))}" size="{hx + 2 * t} {t} {depth / 2}" {xy} material="mat_pouch" {c}/>')
    g.append(f'<geom name="{name}_s4" type="box" pos="{_v(mid - b * (hy + t))}" size="{hx + 2 * t} {t} {depth / 2}" {xy} material="mat_pouch" {c}/>')
    jnt = f'type="hinge" range="0 1.45" stiffness="{flap_k}" springref="{flap_ref}" damping="0.002" armature="0.0002"'
    if not split:
        hinge = mc + b * hy
        flap = [f'<body name="{name}_flap" pos="{_v(hinge)}" {xy}>',
                f'  <joint name="{name}_flap_j" axis="1 0 0" {jnt}/>',
                f'  <geom name="{name}_flap_g" type="box" pos="0 {-hy + 0.002} 0.002" size="{hx - 0.004} {hy - 0.004} 0.002" mass="0.03" material="mat_flap" {c}/>',
                "</body>"]
        joints, swing, gap = [f"{name}_flap_j"], 2 * hy, 0.0
    else:
        Ls = (hy - 0.01, hy - 0.01) if flap_len is None else (tuple(float(x) for x in flap_len) if np.ndim(flap_len)
                                                               else (float(flap_len), float(flap_len)))
        if not all(0.0 <= L < 2 * hy for L in Ls) or max(Ls) <= 0.0 or sum(Ls) >= 2 * hy:
            raise ValueError("flap lengths must be >= 0, one > 0, and leave a gap")
        flap, joints = [], []
        for k, sg, L in ((1, 1.0, Ls[0]), (2, -1.0, Ls[1])):   # flap 1 at the +b edge, flap 2 at -b; both open inward
            if L <= 0.0:
                continue
            hinge = mc + sg * b * hy
            flap += [f'<body name="{name}_flap{k}" pos="{_v(hinge)}" {xy}>',
                     f'  <joint name="{name}_flap{k}_j" axis="{sg:g} 0 0" {jnt}/>',
                     f'  <geom name="{name}_flap{k}_g" type="box" pos="0 {-sg * (L / 2 + 0.002):.6g} 0.002" size="{hx - 0.004} {L / 2} 0.002" mass="{0.03 * L / (2 * hy):.6g}" material="mat_flap" {c}/>',
                     "</body>"]
            joints.append(f"{name}_flap{k}_j")
        swing, gap = max(Ls), 2 * hy - Ls[0] - Ls[1]
    info = dict(mouth_center=mc, normal=n, a=a, b=b, mouth_half=(hx, hy), depth=depth, preload_Nm=abs(flap_k * flap_ref),
                flap_joints=joints, swing_radius=swing, gap=gap,
                flap_edges=["+b" if j.endswith("flap1_j") else "-b" if j.endswith("flap2_j") else "+b" for j in joints])
    return "\n".join(g), "\n".join(flap), info


def fiducial_plate(name: str, center, normal, t1, size=(0.10, 0.06), thick: float = 0.002) -> tuple:
    """Planar fiducial marker (display only, no contact): black plate, white field, an asymmetric pattern of black
    cells; site ``{name}`` at the plate centre with the plate axes (x = t1, z = normal).  Returns (xml, info)."""
    n, a, b = frame_axes(normal, t1)
    c = np.asarray(center, float)
    xy = f'xyaxes="{_v(a)} {_v(b)}"'
    hx, hy = size[0] / 2, size[1] / 2
    g = [f'<geom name="{name}_plate" type="box" pos="{_v(c - n * thick / 2)}" size="{hx} {hy} {thick / 2}" {xy} material="mat_fid_black" contype="0" conaffinity="0"/>',
         f'<geom name="{name}_field" type="box" pos="{_v(c + n * 0.0002)}" size="{hx * 0.80} {hy * 0.72} 0.0002" {xy} material="mat_fid_white" contype="0" conaffinity="0"/>']
    cell = 0.18 * min(size)
    for k, (u, v) in enumerate(((-0.55, 0.35), (0.10, 0.35), (0.55, -0.30), (-0.20, -0.30))):
        pc = c + n * 0.0004 + a * (u * hx) + b * (v * hy)
        g.append(f'<geom name="{name}_cell{k}" type="box" pos="{_v(pc)}" size="{cell / 2} {cell / 2} 0.0002" {xy} material="mat_fid_black" contype="0" conaffinity="0"/>')
    g.append(f'<site name="{name}" pos="{_v(c)}" {xy} size="0.004"/>')
    return "\n".join(g), dict(name=name, center=c, normal=n, a=a, b=b, size=tuple(size))


def standard_bag(name: str, mouth_center, normal, t1, params: dict | None = None) -> tuple:
    """The task station's standard stowage bag (``STANDARD_BAG``): split short flaps + a fiducial plate beside the
    mouth.  Returns (static geoms xml incl. the fiducial, flap body xml, info as ``stowage_bag`` + ``fiducial``)."""
    P = dict(STANDARD_BAG, **(params or {}))
    geoms, flap, info = stowage_bag(name, mouth_center, normal, t1, mouth_half=P["mouth_half"], depth=P["depth"],
                                    flap_k=P["flap_k"], flap_ref=P["flap_ref"], split=True, flap_len=P["flap_len"])
    n, a, b = info["normal"], info["a"], info["b"]
    hx = P["mouth_half"][0]
    fc = np.asarray(mouth_center, float) + a * (hx + 0.008 + P["fiducial_gap"] + P["fiducial_size"][0] / 2)
    fx, finfo = fiducial_plate(name + "_fid", fc, n, a, P["fiducial_size"])
    info["fiducial"] = finfo
    info["standard"] = True
    return geoms + "\n" + fx, flap, info


def hook_loop_board(name: str, center, normal, t1, half=(0.20, 0.15), thick: float = 0.006) -> str:
    """Static board flush on a wall (the cabin face passes ``thick`` in front of center)."""
    n, a, b = frame_axes(normal, t1)
    c = np.asarray(center, float) + n * (thick / 2)
    return (f'<geom name="{name}" type="box" pos="{_v(c)}" size="{half[0]} {half[1]} {thick / 2}" '
            f'xyaxes="{_v(a)} {_v(b)}" material="mat_hook_loop" {CONTACT}/>')


def fragment_model(worldbody: str, contact: str = "", name: str = "fragment") -> str:
    """Stand-alone MJCF with the fixture materials, for ``mujoco.MjSpec.from_string`` + ``attach``."""
    return (f'<mujoco model="{name}">\n  <compiler angle="radian" autolimits="true"/>\n  <asset>{MATERIALS}  </asset>\n'
            f"  <worldbody>\n{worldbody}\n  </worldbody>\n  <contact>\n{contact}\n  </contact>\n</mujoco>\n")
