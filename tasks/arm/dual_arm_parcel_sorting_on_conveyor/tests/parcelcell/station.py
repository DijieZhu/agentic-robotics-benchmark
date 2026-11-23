"""The conveyor station in front of the dual-arm platform (one straight belt).

Layout (world frame, column at the origin, robot faces +x): the belt runs
along y in front of the robot, top surface at ``belt_height``, centre line at
x = ``belt_center_x``; parcels travel from -y (upstream) to +y (downstream).

Belt physics: a long collision-only slab on a slide joint, driven by a
velocity actuator (kv 5000 N s/m, force limit 2 kN, slide damping 50); parcels
are carried by friction, so sticking, slipping and pushing back against the
belt emerge from the contact model. The slab's slide position is wrapped
back by whole periods of 5 cm every control tick (the surface is uniform, so
this is physically neutral) and the slab never runs out.

Static hardware: belt frame (shell 2 mm below the true surface, three legs),
a far rail along the full visible length, near rails upstream and downstream
of the work window (open in |y| < work_window_half so the arms can reach in;
their tops ``near_guide_up_height`` / ``near_guide_down_height`` above the
belt), and a low lip on the near side of the work window (upstream half only,
or the whole window with ``lip_full_window``; its top ``lip_height`` above the
belt). The near rails' inner faces and the lip's are in one plane.

Fixed cameras: the overhead infeed camera (``inspect_cam``) and an overview
camera (tooling).

With ``solid_frame`` (the task's station) the shell and the legs are solid:
each gets collision boxes (``belt_shell_col``; per leg ``belt_leg_col_<i>``,
and with ``leg_style`` "posts" also ``belt_leg_col_<i>_far``,
``belt_leg_cap_<i>``, ``belt_leg_brace_<i>``) in their own static bodies,
equal to the drawn boxes; the moving slab is excluded from
contact with them. Without it (the generic simulator's default) they are drawn
only and ``frame_probes`` may add report-only probe boxes.

``hardware``: further station hardware near the arms as collision boxes, each
in its own static body (name, centre, half sizes), added after everything
else; they are not drawn (the look draws the hardware itself).

Compile-time structure rules (found the hard way in the source; keep them):
1. geom_sameframe: a geom compiled at its body origin with identity
   orientation gets sameframe=1 and later geom_pos writes are ignored. Every
   geom whose pos is written at run time (the label pieces) is compiled with a
   non-zero pos; parcel geoms whose pos is never written may sit at the origin
   but must never be inertia-framed (asserted in ``ParcelCell``).
2. The midphase BVH is baked at compile time from geom_pos / geom_size;
   parcel slots change geom sizes at run time, so the midphase is disabled
   (``mjDSBL_MIDPHASE``; the broadphase stays, physics is unchanged).
   Every collision part of the station sits in its own static body.
3. The base link's mesh hull is replaced by primitives (see the robot MJCF).

Geom groups: 0 station / parcel visuals, 1 robot visuals, 2 gripper visuals,
3 every collision geom, 4 sites and probes. Visual geoms never collide and
carry no mass (or sit in bodies with explicit inertials), so a look pass may
replace them without touching physics.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

__all__ = ["StationConfig", "BELT_HARDWARE", "BELT_PROBES", "FRAME_SOLIDS",
           "add_station", "frame_boxes", "BELT_KV", "BELT_DAMPING",
           "SLAB_HALF_LEN", "WRAP_PERIOD", "STRUCTURE"]

SLAB_HALF_LEN = 6.0          # slab half length (12 m)
SLAB_HZ = 0.02               # slab half thickness
WRAP_PERIOD = 0.05           # slide position wrap period (m)
BELT_KV = 5000.0             # velocity actuator gain
BELT_DAMPING = 50.0          # slide joint damping
BELT_FORCE = 2000.0
BELT_MAX_SPEED = 1.0

RAIL_RGBA = (0.55, 0.57, 0.6, 1)
SHELL_RGBA = (0.23, 0.25, 0.28, 1)
LEG_RGBA = (0.35, 0.36, 0.4, 1)
FLOOR_RGBA = (0.42, 0.45, 0.48, 1)

# collision parts of the belt (names of collision geoms). The solid frame
# boxes (shell, legs) and the downstream lip piece exist only with
# StationConfig.solid_frame / lip_full_window; ``ParcelCell`` resolves the
# names present.
LEG_PARTS = ("belt_leg_col_%d", "belt_leg_col_%d_far", "belt_leg_cap_%d", "belt_leg_brace_%d")
BELT_HARDWARE = ("belt_surface", "belt_rail_far", "belt_rail_near_up",
                 "belt_rail_near_down", "belt_lip_near_up",
                 "belt_lip_near_down", "belt_shell_col") + \
    tuple(f % i for i in range(3) for f in LEG_PARTS)
# station structure = every belt part except the moving belt surface
STRUCTURE = tuple(n for n in BELT_HARDWARE if n != "belt_surface")
# non-physical probes of the belt frame (see StationConfig.frame_probes)
BELT_PROBES = ("belt_frame_probe", "belt_leg_probe_0", "belt_leg_probe_1",
               "belt_leg_probe_2")
# collision parts of the solid belt frame (StationConfig.solid_frame)
FRAME_SOLIDS = ("belt_shell_col",) + tuple(f % i for i in range(3) for f in LEG_PARTS)


@dataclass(frozen=True)
class StationConfig:
    """Belt geometry (defaults = the source's straight layout)."""
    belt_height: float = 0.84          # belt top (m)
    belt_center_x: float = 0.42        # belt centre line to the column (m)
    belt_width: float = 0.44           # m
    visible_half_len: float = 1.3      # half length of the belt frame (m)
    work_window_half: float = 0.42     # near rails open for |y| < this (m)
    belt_friction: float = 0.9         # rubber belt
    entry_y: float = -1.15             # parcels land here (upstream)
    exit_y: float = 1.32               # parcels are removed past this line
    fall_drop: float = 0.35            # removed as fallen below top - this
    fall_xy: float = 2.2               # or beyond |x| / |y| > this
    lip_full_window: bool = False
    """Near-side lip through the whole work window (|y| < work_window_half)
    instead of its upstream half only (a continuous low lip)."""
    lip_height: float = 0.025
    """Lip top above the belt top (m)."""
    near_guide_up_height: float = 0.09
    """Top of the near-side guide upstream of the work window above the belt
    top (m)."""
    near_guide_down_height: float = 0.09
    """Top of the near-side guide downstream of the work window above the
    belt top (m)."""
    leg_style: str = "block"
    """The belt's legs: "block" (one box per leg from the floor to 10 cm below
    the belt surface, the generic simulator's) or "posts" (the station's own
    legs as drawn: per leg two square posts, a cap plate under the frame and a
    brace between the posts)."""
    solid_frame: bool = False
    """Collision boxes for the belt shell and the legs, equal to the drawn
    boxes (the conveyor body is solid). They never touch the moving slab
    (explicit body excludes)."""
    hardware: tuple = ()
    """Station hardware near the arms as collision boxes: a tuple of
    (name, (cx, cy, cz), (hx, hy, hz)) in world coordinates, each in its own
    static body, not drawn."""
    frame_probes: bool = True
    """Add non-physical probe boxes covering the belt frame (shell, legs).
    The source's frame has no collision at all, so an arm can pass through
    the belt body unnoticed; the truth side tests the robot's collision
    geoms against these boxes every tick (geometric distance query, no
    contact, no force) and reports overlaps. Physics are identical with or
    without probes."""

    @property
    def half_width(self) -> float:
        return self.belt_width / 2


def _static(w, name, pos, size, friction, rgba, priority=0, solimp=None,
            solref=None):
    """A collision box in its own static body + its visual twin."""
    body = w.add_body(name=f"{name}_body")
    kw = dict(name=name, type=mujoco.mjtGeom.mjGEOM_BOX, pos=pos, size=size,
              rgba=list(rgba), friction=friction, mass=5.0, group=3)
    if priority:
        kw["priority"] = priority
    if solimp is not None:
        kw.update(solimp=solimp, solref=solref)
    body.add_geom(**kw)
    body.add_geom(name=f"{name}_visual", type=mujoco.mjtGeom.mjGEOM_BOX,
                  pos=pos, size=size, rgba=list(rgba), contype=0,
                  conaffinity=0, mass=0, group=0)


def _visual(w, name, pos, size, rgba):
    w.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_BOX, pos=pos, size=size,
               rgba=list(rgba), contype=0, conaffinity=0, mass=0, group=0)


def _probe(w, name, pos, size):
    w.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_BOX, pos=pos, size=size,
               rgba=[1, 0, 0, 0.3], contype=0, conaffinity=0, mass=0,
               group=4)


def add_station(spec: mujoco.MjSpec, st: StationConfig) -> None:
    """Floor, lights, scene cameras and the belt into ``spec``."""
    w = spec.worldbody
    cx, h, hx, L = (st.belt_center_x, st.belt_height, st.half_width,
                    st.visible_half_len)
    W = st.work_window_half

    # floor: a box, not a plane (idle parcel slots park below it, contact-free)
    w.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_BOX,
               pos=[0, 0, -0.05], size=[4, 4, 0.05], rgba=list(FLOOR_RGBA),
               group=3)
    w.add_geom(name="floor_visual", type=mujoco.mjtGeom.mjGEOM_BOX,
               pos=[0, 0, -0.05], size=[4, 4, 0.05], rgba=list(FLOOR_RGBA),
               contype=0, conaffinity=0, mass=0, group=0)
    w.add_light(name="light_top", pos=[0.5, 0, 2.6], dir=[0, 0, -1],
                diffuse=[0.9, 0.9, 0.9])
    w.add_light(name="light_side", pos=[1.5, -1.5, 2.2],
                dir=[-0.5, 0.5, -0.8], diffuse=[0.45, 0.45, 0.45])

    # scene cameras: overhead inspection camera above the upstream belt and
    # an overview camera (head and wrist cameras come with the robot)
    w.add_camera(name="inspect_cam", pos=[cx, -0.75, h + 0.85],
                 xyaxes=[-1, 0, 0, 0, -1, 0], fovy=62)
    w.add_camera(name="overview_cam", pos=[2.3, -1.9, 1.9],
                 xyaxes=[0.64, 0.77, 0, -0.3, 0.25, 0.92], fovy=55)

    # moving slab: own body at the origin, geom at a non-zero pos (rule 1)
    slab = w.add_body(name="belt_slab")
    slab.add_joint(name="belt_slide", type=mujoco.mjtJoint.mjJNT_SLIDE,
                   axis=[0, 1, 0], damping=BELT_DAMPING)
    slab.add_geom(name="belt_surface", type=mujoco.mjtGeom.mjGEOM_BOX,
                  pos=[cx, 0, h - SLAB_HZ], size=[hx, SLAB_HALF_LEN, SLAB_HZ],
                  rgba=[0, 0, 0, 0], group=3,
                  friction=[st.belt_friction, 0.01, 0.0001],
                  mass=200.0, priority=1,
                  solimp=[0.95, 0.99, 0.001, 0.5, 2], solref=[0.004, 1])
    spec.add_actuator(name="belt_drive", target="belt_slide",
                      trntype=mujoco.mjtTrn.mjTRN_JOINT,
                      dyntype=mujoco.mjtDyn.mjDYN_NONE,
                      gaintype=mujoco.mjtGain.mjGAIN_FIXED,
                      biastype=mujoco.mjtBias.mjBIAS_AFFINE,
                      gainprm=[BELT_KV] + [0] * 9,
                      biasprm=[0, 0, -BELT_KV] + [0] * 7,
                      ctrlrange=[-BELT_MAX_SPEED, BELT_MAX_SPEED],
                      forcerange=[-BELT_FORCE, BELT_FORCE])

    # rails and lip (collision + visual twin, each in its own body: rule 2)
    rail_far = [0.15, 0.003, 0.0001]
    rail_near = [0.3, 0.005, 0.0001]
    _static(w, "belt_rail_far", [cx + hx + 0.022, 0, h - 0.005],
            [0.012, L, 0.05], rail_far, RAIL_RGBA)
    seg = (L - W) / 2
    # each guide reaches from 5 cm below the belt surface to its top (the centre and half height rounded to the
    # micrometre, so that a 9 cm guide is the generic simulator's box to the last bit)
    for nm, top_above, ys in (("belt_rail_near_up", float(st.near_guide_up_height), -(W + seg)),
                              ("belt_rail_near_down", float(st.near_guide_down_height), W + seg)):
        _static(w, nm, [cx - hx - 0.022, ys, h + round((top_above - 0.05) / 2, 9)],
                [0.012, seg, round((top_above + 0.05) / 2, 9)], rail_near, RAIL_RGBA)
    # low near-side lip in the work window: keeps a parcel being rolled /
    # pushed from sliding off the near edge (upstream half only, or the whole
    # window: a second piece of the same section downstream of y = 0)
    lip_top = h + float(st.lip_height)
    lip_bot = h - 0.021
    lip_hz, lip_cz = (lip_top - lip_bot) / 2, (lip_top + lip_bot) / 2
    _static(w, "belt_lip_near_up", [cx - hx - 0.018, -W / 2, lip_cz],
            [0.008, W / 2, lip_hz], rail_far, RAIL_RGBA)
    if st.lip_full_window:
        _static(w, "belt_lip_near_down", [cx - hx - 0.018, W / 2, lip_cz],
                [0.008, W / 2, lip_hz], rail_far, RAIL_RGBA)

    # belt frame (drawn; collision boxes with ``solid_frame``)
    fb = frame_boxes(st)
    shell_pos, shell_size = fb["belt_shell_col"]
    legs = [(n, ps) for n, ps in fb.items() if n != "belt_shell_col"]
    _visual(w, "belt_shell", shell_pos, shell_size, SHELL_RGBA)
    for n, (p, s) in legs:
        _visual(w, n.replace("_col", "") if "_col" in n else n + "_visual", p, s, LEG_RGBA)
    if st.solid_frame:
        # solid conveyor body: collision boxes equal to the frame probes,
        # each in its own static body (rule 2); the shell box encloses the
        # moving slab, so the slab / shell pair is excluded explicitly
        for name, (p, s) in [("belt_shell_col", (shell_pos, shell_size))] + legs:
            body = w.add_body(name=f"{name}_body")
            body.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_BOX, pos=p,
                          size=s, rgba=[0, 0, 0, 0], friction=[0.6, 0.005,
                          0.0001], mass=5.0, group=3)
            ex = spec.add_exclude()
            ex.bodyname1, ex.bodyname2 = f"{name}_body", "belt_slab"

    if st.frame_probes and not st.solid_frame:
        _probe(w, "belt_frame_probe", shell_pos, shell_size)
        for i, (n, (p, s)) in enumerate(legs[:3]):
            _probe(w, f"belt_leg_probe_{i}", p, s)

    # further station hardware near the arms (collision only; drawn by the
    # look): each box in its own static body, after everything else
    for name, p, s in st.hardware:
        body = w.add_body(name=f"{name}_body")
        body.add_geom(name=str(name), type=mujoco.mjtGeom.mjGEOM_BOX,
                      pos=[float(v) for v in p], size=[float(v) for v in s],
                      rgba=[0, 0, 0, 0], friction=[0.6, 0.005, 0.0001],
                      mass=5.0, group=3)


def frame_boxes(st: StationConfig) -> dict:
    """The belt frame's boxes {collision name: (centre, half sizes)}: the
    shell (its top 2 mm below the belt surface, 1 cm wider than the belt on
    each side) and three legs from the floor to 10 cm below the belt
    surface."""
    cx, h, hx, L = (st.belt_center_x, st.belt_height, st.half_width,
                    st.visible_half_len)
    out = {"belt_shell_col": ([cx, 0, h - 0.052], [hx + 0.01, L, 0.05])}
    leg_hz = (h - 0.1) / 2
    for i, ys in enumerate((-L + 0.15, 0.0, L - 0.15)):
        if st.leg_style == "posts":
            # the station's legs as drawn: two 40 mm square posts 0.11 m apart across the belt (from the floor
            # plates to the cap), a 6 mm cap plate under the frame, a brace between the posts 0.2 m above the floor
            top, bot = h - 0.102, 0.012
            pz, ph = (top + bot) / 2, (top - bot) / 2
            out[f"belt_leg_col_{i}"] = ([cx - 0.055, ys, pz], [0.02, 0.02, ph])
            out[f"belt_leg_col_{i}_far"] = ([cx + 0.055, ys, pz], [0.02, 0.02, ph])
            out[f"belt_leg_cap_{i}"] = ([cx, ys, h - 0.105], [0.085, 0.03, 0.003])
            out[f"belt_leg_brace_{i}"] = ([cx, ys, 0.2], [0.035, 0.015, 0.01])
        else:
            out[f"belt_leg_col_{i}"] = ([cx, ys, leg_hz], [hx * 0.55, 0.04, leg_hz])
    return out


def belt_ctrl_for_speed(v: float, compensate: bool) -> float:
    """Velocity command giving a steady slab speed ``v``. The slide damping
    makes the uncompensated steady speed kv / (kv + damping) = 0.990 of the
    command; with ``compensate`` the command is scaled up by the inverse."""
    if compensate:
        v = v * (BELT_KV + BELT_DAMPING) / BELT_KV
    return float(np.clip(v, -BELT_MAX_SPEED, BELT_MAX_SPEED))
