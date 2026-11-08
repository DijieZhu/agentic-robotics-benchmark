"""carton.py - one-piece hinged-lid cardboard shoe-box blank for MuJoCo (MjSpec API).

The blank is a tree of rigid cardboard panels connected by hinge joints at the creases:

    floor (free joint)
     |- front wall  (+x edge)   |- left wall (+y edge)
     |- back wall   (-x edge)   |- right wall (-y edge)
          |- lid  (hinged on the back wall's far edge = top edge when erected)
               |- flap (tuck flap hinged on the lid's far edge)

Conventions (shared with the other blocks)
------------------------------------------
* units: m / rad / s / kg; quaternions are wxyz (MuJoCo order).
* The *floor frame* is the floor body frame: x along L, y along W, z up, origin at the
  centre of the floor panel's BOTTOM face.  ``pos`` passed to :func:`add_carton` is that
  origin in the parent frame, so ``pos=(x, y, 0)`` lays the flat blank on a bench whose
  top is z = 0; the floor's top face is then at z = thickness.
* Every hinge angle is 0 when the panel is coplanar with its parent panel and positive
  when it folds toward the inside of the box (walls fold up, the lid folds over the box,
  the flap folds down over the outside of the front wall).  Hinge ranges are [0, 1.62].
* Panel local frames (walls, lid, flap): origin on the crease line (the hinge axis is the
  local -y axis so that positive angles fold "up"), local +x runs from the crease toward
  the panel's free edge, local +z is the panel's INNER-face normal (= up when the blank
  is flat).  The panel occupies local x in [x0, x0 + length], z in [-thickness, 0].
* Flat blank: the inner (lighter) faces of every panel are on top; the brown outer faces
  and the lid's logo are underneath (exactly like a real die-cut blank: the inside of
  the box is one side of the sheet).
* Handle sites (where a gripper pinches a panel edge): their +z axis points outward
  from the panel's free edge in the panel's plane, their +y axis is the panel's
  inner-face normal, +x runs along the edge (right-handed).  Each site sits in the
  panel's mid-plane, ``grip_inset`` (default 0.012) inward from the free edge.  To pinch
  at a site: approach along -site.z (fingertips toward the crease) and close the fingers
  along site.y (across the cardboard thickness).  Walls have three sites along the free
  edge (centre, +0.3*edge, -0.3*edge); the lid and the flap have one at the centre.
* Erected geometry (floor frame): walls stand just outside the floor footprint, so the
  interior is x in [-L/2, L/2], y in [-W/2, W/2], z in [t, t + H + clearance]; the
  outer footprint is (L + 2t) x (W + 2t); the closed lid, (L + 2t + 2*overhang) x
  (W + 2t + 2*overhang), covers it with ``overhang`` (2 mm) to spare on every side and
  its hinge sits ``clearance`` (1 mm) above the wall tops.

Snap locks: one inactive ``mjEQ_JOINT`` equality per hinge holding it at pi/2.
:func:`update_locks` activates them (walls: angle >= 87 deg for 0.3 s; lid + flap: lid in
[87, 93] deg AND all walls locked AND flap >= 80 deg, for 0.3 s).  Locks never release.

Runtime dependencies: mujoco + numpy only.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Sequence, Tuple

import mujoco
import numpy as np

HINGES: Tuple[str, ...] = ("front", "back", "left", "right", "lid", "flap")
WALLS: Tuple[str, ...] = ("front", "back", "left", "right")
LOCK_ANGLE: float = math.pi / 2          # every lock holds its hinge here
WALL_LOCK_DEG: float = 87.0              # wall locks once angle >= this for LOCK_DWELL_S
LID_LOCK_RANGE_DEG: Tuple[float, float] = (87.0, 93.0)
FLAP_LOCK_DEG: float = 80.0
LOCK_DWELL_S: float = 0.3
# site quaternion (wxyz) in the panel body frame giving site x = body y, site y = body z,
# site z = body x (a 120 deg rotation about (1,1,1)); verified numerically in the tests.
HANDLE_SITE_QUAT: Tuple[float, float, float, float] = (0.5, 0.5, 0.5, 0.5)


@dataclass
class CartonState:
    """Mutable lock state owned by one carton descriptor (``desc['state']``)."""

    time: float = 0.0
    dwell: Dict[str, float] = field(default_factory=lambda: {h: 0.0 for h in HINGES})
    locked: Dict[str, bool] = field(default_factory=lambda: {h: False for h in HINGES})
    lock_time: Dict[str, Optional[float]] = field(default_factory=lambda: {h: None for h in HINGES})

    @property
    def erected(self) -> bool:
        return all(self.locked[w] for w in WALLS)

    @property
    def closed(self) -> bool:
        return self.locked["lid"]

    def reset(self) -> None:
        self.time = 0.0
        for h in HINGES:
            self.dwell[h] = 0.0
            self.locked[h] = False
            self.lock_time[h] = None


# --------------------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------------------
def _yaw_quat(yaw: float) -> Tuple[float, float, float, float]:
    return (math.cos(0.5 * yaw), 0.0, 0.0, math.sin(0.5 * yaw))


def _range_for_spec(spec: mujoco.MjSpec, rng_rad: Sequence[float]) -> Tuple[float, float]:
    """Joint ranges are interpreted in the spec's angle unit (degrees by default)."""
    if spec.compiler.degree:
        return (math.degrees(rng_rad[0]), math.degrees(rng_rad[1]))
    return (float(rng_rad[0]), float(rng_rad[1]))


def _add_panel(body, name: str, x_range, y_range, z_range, rgba_outer, rgba_inner,
               mass: float, friction, condim: int, group_collision: int,
               group_visual: int, margin: float, solref, inner_face: bool = True) -> Dict[str, str]:
    """One cardboard panel: a collision box (outer colour) + a visual slab on its +z face."""
    x0, x1 = x_range
    y0, y1 = y_range
    z0, z1 = z_range
    centre = ((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2)
    half = ((x1 - x0) / 2, (y1 - y0) / 2, (z1 - z0) / 2)
    names = {}
    body.add_geom(name=f"{name}_panel", type=mujoco.mjtGeom.mjGEOM_BOX, pos=centre,
                  size=half, rgba=rgba_outer, mass=mass, friction=friction,
                  condim=condim, group=group_collision, margin=margin, solref=list(solref))
    names["panel"] = f"{name}_panel"
    if inner_face:
        eps = 0.0001
        inset = 0.0003
        body.add_geom(name=f"{name}_inner", type=mujoco.mjtGeom.mjGEOM_BOX,
                      pos=(centre[0], centre[1], z1 + eps),
                      size=(half[0] - inset, half[1] - inset, eps), rgba=rgba_inner,
                      contype=0, conaffinity=0, density=0, group=group_visual)
        names["inner"] = f"{name}_inner"
    return names


# --------------------------------------------------------------------------------------
# builder
# --------------------------------------------------------------------------------------
def add_carton(spec: mujoco.MjSpec, parent, name: str, L: float = 0.33, W: float = 0.20,
               H: float = 0.115, thickness: float = 0.003,
               pos: Sequence[float] = (0.0, 0.0, 0.0), yaw: float = 0.0,
               rgba_outer: Sequence[float] = (0.72, 0.55, 0.35, 1.0),
               rgba_inner: Sequence[float] = (0.86, 0.78, 0.62, 1.0),
               logo_rgba: Sequence[float] = (0.12, 0.32, 0.78, 1.0),
               mass: float = 0.09, flap_height: float = 0.04, flap_width: Optional[float] = None,
               overhang: float = 0.002, clearance: float = 0.001, corner_gap: float = 0.0005,
               grip_inset: float = 0.012, crease_stiffness: float = 0.012,
               crease_damping: float = 0.006, lid_damping: float = 0.02,
               hinge_range: Sequence[float] = (0.0, 1.62),
               friction: Sequence[float] = (0.6, 0.005, 0.0001), condim: int = 3,
               margin: float = 0.0, contact_solref: Sequence[float] = (0.004, 1.0),
               lock_solref: Sequence[float] = (0.005, 1.0),
               lock_solimp: Sequence[float] = (0.99, 0.999, 0.001, 0.5, 2.0),
               free: bool = True, group_collision: int = 0, group_visual: int = 1,
               site_group: int = 4, with_lid: bool = True) -> dict:
    """Add a flat shoe-box blank under ``parent`` and return its descriptor.

    Parameters
    ----------
    L, W, H : box length (0.30-0.36), width (0.18-0.22) and height (0.10-0.13).
    thickness : cardboard thickness (3 mm).
    pos, yaw : pose of the floor frame (bottom-centre of the floor panel) in the parent
        frame; the blank is flat (all hinges 0) at creation.
    mass : total blank mass, spread over the panels by area (uniform areal density).
    flap_height / flap_width : tuck flap size (width defaults to W - 0.02).
    overhang : how far the lid extends beyond the erected walls' OUTER faces on every
        side: the lid is (L + 2*thickness + 2*overhang) x (W + 2*thickness + 2*overhang).
        (A flush lid catches on the side walls' top edges while it swings down.)
    clearance : 1 mm air gap between the closed lid and the wall tops, and between the
        folded flap and the front wall (avoids zero-distance contacts in the locked pose).
    corner_gap : the long walls are shortened by this at each end so that erected walls
        touch at the corners without interpenetrating.
    grip_inset : handle-site distance inward from the panel's free edge.
    crease_stiffness / crease_damping : return spring (N m/rad per metre of crease
        length, toward angle 0) and damping (N m s/rad per metre) of every hinge.
    lid_damping : absolute damping of the lid hinge (N m s/rad).  The lid + flap tower
        topples over the box as soon as the back wall passes vertical (its hinge cannot
        go negative); this damping limits the fall to ~2 rad/s so the lid does not
        tunnel through the thin front wall (undamped it hits at ~3 m/s).
    margin : contact margin of the collision panels.  Keep it 0: the erected corners
        are only ``corner_gap`` apart and the lid's inner face sweeps ~1 mm above the
        side walls' top corners next to its hinge, so any margin >= 1 mm creates large
        spurious contact forces there (measured: 97 N at a corner, lid jammed at 52 deg).
    contact_solref : contact stiffness of the panels (pairs average it with the other
        geom's solref); 0.004 is the stiffest value valid for a 0.002 timestep.
    lock_solref / lock_solimp : stiffness of the snap-lock equality constraints.
    free : add a free joint to the floor (the parent must then be the world body).

    Returns
    -------
    dict with keys ``name, params, bodies, joints, equalities, sites, geoms,
    interior_aabb_floor, keypoints, handle_sites_local, state``; see module docstring.
    """
    t = float(thickness)
    L, W, H = float(L), float(W), float(H)
    if min(L, W, H, t) <= 0:
        raise ValueError("L, W, H, thickness must be positive")
    if flap_width is None:
        flap_width = W - 0.02
    rng = _range_for_spec(spec, hinge_range)

    # ---- masses: uniform areal density -------------------------------------------------
    wall_edge = {"front": W, "back": W, "left": L - 2 * corner_gap, "right": L - 2 * corner_gap}
    lid_ext = t + overhang                       # lid extension beyond the floor edge
    lid_len, lid_wid = L + 2 * lid_ext, W + 2 * lid_ext
    areas = {"floor": L * W, "lid": lid_len * lid_wid, "flap": flap_height * flap_width}
    for w in WALLS:
        areas[w] = H * wall_edge[w]
    areal = mass / sum(areas.values())
    pmass = {k: areal * a for k, a in areas.items()}

    bodies, joints, eqs, sites, geoms = {}, {}, {}, {}, {}
    pfx = f"{name}/"

    def hinge(body, hname: str, crease_len: float, damping: Optional[float] = None):
        jn = f"{pfx}{hname}_hinge"
        if damping is None:
            damping = crease_damping * crease_len
        body.add_joint(name=jn, type=mujoco.mjtJoint.mjJNT_HINGE, axis=(0.0, -1.0, 0.0),
                       pos=(0.0, 0.0, 0.0), range=rng, stiffness=crease_stiffness * crease_len,
                       damping=damping, springref=0.0)
        joints[hname] = jn
        en = f"{pfx}{hname}_lock"
        spec.add_equality(name=en, type=mujoco.mjtEq.mjEQ_JOINT, name1=jn, name2="",
                          active=False, data=[LOCK_ANGLE] + [0.0] * 10,
                          solref=list(lock_solref), solimp=list(lock_solimp))
        eqs[hname] = en

    def handle_sites(body, hname: str, x_edge: float, ys: Sequence[float], tags: Sequence[str]):
        lst = []
        for y, tag in zip(ys, tags):
            sn = f"{pfx}{hname}_handle_{tag}"
            body.add_site(name=sn, pos=(x_edge - grip_inset, y, -t / 2), quat=HANDLE_SITE_QUAT,
                          type=mujoco.mjtGeom.mjGEOM_SPHERE, size=(0.003, 0.003, 0.003),
                          group=site_group, rgba=(1.0, 0.2, 0.2, 1.0))
            lst.append(sn)
        sites[hname] = lst

    common = dict(rgba_outer=rgba_outer, rgba_inner=rgba_inner, friction=friction,
                  condim=condim, group_collision=group_collision, group_visual=group_visual,
                  margin=margin, solref=contact_solref)

    # ---- floor ----------------------------------------------------------------------
    floor = parent.add_body(name=f"{pfx}floor", pos=tuple(pos), quat=_yaw_quat(yaw))
    bodies["floor"] = f"{pfx}floor"
    if free:
        floor.add_freejoint(name=f"{pfx}floor_free")
        joints["free"] = f"{pfx}floor_free"
    geoms["floor"] = _add_panel(floor, f"{pfx}floor", (-L / 2, L / 2), (-W / 2, W / 2),
                                (0.0, t), mass=pmass["floor"], **common)
    floor.add_site(name=f"{pfx}floor_centre", pos=(0, 0, 0), size=(0.003, 0.003, 0.003),
                   group=site_group)
    floor.add_site(name=f"{pfx}interior_centre", pos=(0, 0, t + H / 2),
                   size=(0.003, 0.003, 0.003), group=site_group)
    sites["floor_centre"] = f"{pfx}floor_centre"
    sites["interior_centre"] = f"{pfx}interior_centre"

    # ---- walls ----------------------------------------------------------------------
    c45 = math.sqrt(0.5)
    wall_pose = {
        "front": ((L / 2, 0.0, t), (1.0, 0.0, 0.0, 0.0)),
        "back": ((-L / 2, 0.0, t), (0.0, 0.0, 0.0, 1.0)),
        "left": ((0.0, W / 2, t), (c45, 0.0, 0.0, c45)),
        "right": ((0.0, -W / 2, t), (c45, 0.0, 0.0, -c45)),
    }
    wall_bodies = {}
    for w in WALLS:
        bpos, bquat = wall_pose[w]
        wb = floor.add_body(name=f"{pfx}{w}", pos=bpos, quat=bquat)
        wall_bodies[w] = wb
        bodies[w] = f"{pfx}{w}"
        hinge(wb, w, wall_edge[w])
        e = wall_edge[w]
        geoms[w] = _add_panel(wb, f"{pfx}{w}", (0.0, H), (-e / 2, e / 2), (-t, 0.0),
                              mass=pmass[w], **common)
        handle_sites(wb, w, H, (0.0, 0.3 * e, -0.3 * e), ("c", "p", "m"))

    if with_lid:
        # ---- lid (child of the back wall) ------------------------------------------------
        lid = wall_bodies["back"].add_body(name=f"{pfx}lid", pos=(H + clearance, 0.0, 0.0),
                                           quat=(1.0, 0.0, 0.0, 0.0))
        bodies["lid"] = f"{pfx}lid"
        hinge(lid, "lid", lid_wid, damping=lid_damping)
        geoms["lid"] = _add_panel(lid, f"{pfx}lid", (-lid_ext, L + lid_ext),
                                  (-lid_wid / 2, lid_wid / 2), (-t, 0.0), mass=pmass["lid"], **common)
        eps = 0.0001
        lid.add_geom(name=f"{pfx}lid_logo", type=mujoco.mjtGeom.mjGEOM_BOX,
                     pos=(L / 2, 0.0, -t - eps), size=(0.25 * L, 0.25 * W, eps), rgba=logo_rgba,
                     contype=0, conaffinity=0, density=0, group=group_visual)
        geoms["lid"]["logo"] = f"{pfx}lid_logo"
        handle_sites(lid, "lid", L + lid_ext, (0.0,), ("c",))

        # ---- flap (child of the lid) -----------------------------------------------------
        flap = lid.add_body(name=f"{pfx}flap", pos=(L + lid_ext + clearance, 0.0, 0.0),
                            quat=(1.0, 0.0, 0.0, 0.0))
        bodies["flap"] = f"{pfx}flap"
        hinge(flap, "flap", flap_width)
        geoms["flap"] = _add_panel(flap, f"{pfx}flap", (0.0, flap_height),
                                   (-flap_width / 2, flap_width / 2), (-t, 0.0),
                                   mass=pmass["flap"], **common)
        handle_sites(flap, "flap", flap_height, (0.0,), ("c",))

    # ---- descriptor -------------------------------------------------------------------
    handle_local = {}
    for w in WALLS:
        e = wall_edge[w]
        handle_local[w] = [(H - grip_inset, y, -t / 2) for y in (0.0, 0.3 * e, -0.3 * e)]
    if with_lid:
        handle_local["lid"] = [(L + lid_ext - grip_inset, 0.0, -t / 2)]
        handle_local["flap"] = [(flap_height - grip_inset, 0.0, -t / 2)]

    # crease lines in the floor frame when the blank is flat (hinge axes), and the
    # outline of the flat blank (useful for placing it on the bench).
    keypoints = {
        "crease_front": ((L / 2, -W / 2, t), (L / 2, W / 2, t)),
        "crease_back": ((-L / 2, -W / 2, t), (-L / 2, W / 2, t)),
        "crease_left": ((-L / 2, W / 2, t), (L / 2, W / 2, t)),
        "crease_right": ((-L / 2, -W / 2, t), (L / 2, -W / 2, t)),
        "crease_lid": ((-L / 2 - H - clearance, -W / 2, t), (-L / 2 - H - clearance, W / 2, t)),
        "crease_flap": ((-L / 2 - H - L - 2 * clearance - lid_ext, -W / 2, t),
                        (-L / 2 - H - L - 2 * clearance - lid_ext, W / 2, t)),
        "flat_extent_x": (-L / 2 - H - L - 2 * clearance - lid_ext - flap_height, L / 2 + H),
        "lid_size": (lid_len, lid_wid),
        "flat_extent_y": (-W / 2 - H, W / 2 + H),
        "erected_outer_footprint": ((-L / 2 - t, L / 2 + t), (-W / 2 - t, W / 2 + t)),
        "erected_top_z": t + H,
        "closed_top_z": t + H + clearance + t,
    }
    desc = {
        "name": name,
        "params": dict(L=L, W=W, H=H, thickness=t, mass=mass, flap_height=flap_height,
                       flap_width=flap_width, overhang=overhang, clearance=clearance,
                       corner_gap=corner_gap, grip_inset=grip_inset,
                       crease_stiffness=crease_stiffness, crease_damping=crease_damping,
                       lid_damping=lid_damping, margin=margin, contact_solref=tuple(contact_solref),
                       hinge_range=tuple(hinge_range), pos=tuple(pos), yaw=yaw,
                       panel_mass=pmass, wall_edge=wall_edge, lock_angle=LOCK_ANGLE),
        "bodies": bodies, "joints": joints, "equalities": eqs, "sites": sites, "geoms": geoms,
        "hinges": list(HINGES) if with_lid else list(WALLS), "walls": list(WALLS), "with_lid": bool(with_lid),
        "interior_aabb_floor": ((-L / 2, L / 2), (-W / 2, W / 2), (t, t + H + clearance)),
        "handle_sites_local": handle_local,
        "keypoints": keypoints,
        "state": CartonState(),
    }
    return desc


def prefix_desc(desc: dict, prefix: str) -> dict:
    """Copy of ``desc`` with every MuJoCo name prefixed (for ``spec.attach(..., prefix=)``)."""
    def fix(v):
        if isinstance(v, str):
            return prefix + v
        if isinstance(v, list):
            return [fix(x) for x in v]
        if isinstance(v, dict):
            return {k: fix(x) for k, x in v.items()}
        return v
    out = dict(desc)
    for key in ("bodies", "joints", "equalities", "sites", "geoms"):
        out[key] = fix(desc[key])
    out["name"] = prefix + desc["name"]
    out["state"] = CartonState()
    return out


# --------------------------------------------------------------------------------------
# runtime helpers
# --------------------------------------------------------------------------------------
def _ids(model: mujoco.MjModel, desc: dict) -> dict:
    """Cached joint / qpos / equality / body ids for this (model, desc) pair."""
    cache = desc.get("_idcache")
    if cache is not None and cache["model_id"] == id(model):
        jid = cache["jnt"]["front"]
        if mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid) == desc["joints"]["front"]:
            return cache
    jnt = {h: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, desc["joints"][h]) for h in desc["hinges"]}
    for h, j in jnt.items():
        if j < 0:
            raise KeyError(f"joint {desc['joints'][h]!r} not found in model")
    cache = {
        "model_id": id(model),
        "jnt": jnt,
        "qadr": {h: int(model.jnt_qposadr[j]) for h, j in jnt.items()},
        "dadr": {h: int(model.jnt_dofadr[j]) for h, j in jnt.items()},
        "eq": {h: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, desc["equalities"][h]) for h in desc["hinges"]},
        "body": {k: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, v) for k, v in desc["bodies"].items()},
    }
    desc["_idcache"] = cache
    return cache


def hinge_angles(model: mujoco.MjModel, data: mujoco.MjData, desc: dict, degrees: bool = False) -> Dict[str, float]:
    """Current angle of the six hinges (front, back, left, right, lid, flap)."""
    qadr = _ids(model, desc)["qadr"]
    f = math.degrees if degrees else float
    return {h: f(data.qpos[qadr[h]]) for h in desc["hinges"]}


def _engage(model, data, desc, st: CartonState, hname: str) -> None:
    eq = _ids(model, desc)["eq"][hname]
    data.eq_active[eq] = 1
    st.locked[hname] = True
    st.lock_time[hname] = float(data.time)
    st.dwell[hname] = 0.0


def update_locks(model: mujoco.MjModel, data: mujoco.MjData, desc: dict, dt: float) -> CartonState:
    """Advance the snap-lock state machine by ``dt`` seconds (call every step or tick).

    Walls lock after their hinge has been >= 87 deg for 0.3 s continuously.  The lid and
    the flap lock together after the lid has been within [87, 93] deg while all four
    walls are locked and the flap is >= 80 deg, for 0.3 s continuously.  Locks never
    release; locked equalities are re-asserted active on every call.
    """
    st: CartonState = desc["state"]
    ids = _ids(model, desc)
    ang = hinge_angles(model, data, desc, degrees=True)
    st.time += dt
    for w in WALLS:
        if st.locked[w]:
            data.eq_active[ids["eq"][w]] = 1
            continue
        st.dwell[w] = st.dwell[w] + dt if ang[w] >= WALL_LOCK_DEG else 0.0
        if st.dwell[w] >= LOCK_DWELL_S - 1e-9:
            _engage(model, data, desc, st, w)
    if "lid" not in desc["hinges"]:
        return st
    if st.locked["lid"]:
        data.eq_active[ids["eq"]["lid"]] = 1
        data.eq_active[ids["eq"]["flap"]] = 1
    else:
        ok = (st.erected and LID_LOCK_RANGE_DEG[0] <= ang["lid"] <= LID_LOCK_RANGE_DEG[1]
              and ang["flap"] >= FLAP_LOCK_DEG)
        st.dwell["lid"] = st.dwell["lid"] + dt if ok else 0.0
        if st.dwell["lid"] >= LOCK_DWELL_S - 1e-9:
            _engage(model, data, desc, st, "lid")
            _engage(model, data, desc, st, "flap")
    return st


def reset_locks(model: mujoco.MjModel, data: mujoco.MjData, desc: dict) -> CartonState:
    """Deactivate every lock and clear the state (use after mj_resetData)."""
    st: CartonState = desc["state"]
    st.reset()
    for eq in _ids(model, desc)["eq"].values():
        data.eq_active[eq] = 0
    return st


def floor_pose(model: mujoco.MjModel, data: mujoco.MjData, desc: dict) -> Tuple[np.ndarray, np.ndarray]:
    """World position and wxyz quaternion of the floor frame (needs mj_forward/mj_step)."""
    b = _ids(model, desc)["body"]["floor"]
    return data.xpos[b].copy(), data.xquat[b].copy()


def carton_state(model: mujoco.MjModel, data: mujoco.MjData, desc: dict) -> dict:
    """Snapshot of the carton: angles (deg), locks, erected/closed, floor pose, interior.

    ``inside_box(point_world) -> bool`` tests a world point against the interior AABB
    expressed in the floor frame (captured at call time).  ``interior_aabb_floor`` is
    ((xmin, xmax), (ymin, ymax), (zmin, zmax)) in the floor frame.
    """
    st: CartonState = desc["state"]
    ang = hinge_angles(model, data, desc, degrees=True)
    pos, quat = floor_pose(model, data, desc)
    aabb = desc["interior_aabb_floor"]
    lo = np.array([aabb[0][0], aabb[1][0], aabb[2][0]])
    hi = np.array([aabb[0][1], aabb[1][1], aabb[2][1]])
    conj = np.array([quat[0], -quat[1], -quat[2], -quat[3]])

    def to_floor(point_world) -> np.ndarray:
        out = np.zeros(3)
        mujoco.mju_rotVecQuat(out, np.asarray(point_world, dtype=float) - pos, conj)
        return out

    def inside_box(point_world) -> bool:
        p = to_floor(point_world)
        return bool(np.all(p >= lo) and np.all(p <= hi))

    centre_w = np.zeros(3)
    mujoco.mju_rotVecQuat(centre_w, (lo + hi) / 2, quat)
    centre_w += pos
    return {
        "angles_deg": ang,
        "locked": dict(st.locked),
        "lock_time": dict(st.lock_time),
        "erected": st.erected,
        "closed": st.closed,
        "n_walls_locked": sum(st.locked[w] for w in WALLS),
        "floor_pos": pos,
        "floor_quat": quat,
        "interior_aabb_floor": aabb,
        "interior_centre_world": centre_w,
        "to_floor_frame": to_floor,
        "inside_box": inside_box,
    }


def site_pose(model: mujoco.MjModel, data: mujoco.MjData, site_name: str) -> Tuple[np.ndarray, np.ndarray]:
    """World position and wxyz quaternion of a site (needs mj_forward/mj_step)."""
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    if sid < 0:
        raise KeyError(site_name)
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, data.site_xmat[sid])
    return data.site_xpos[sid].copy(), quat
