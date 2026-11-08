"""shoe.py -- procedurally built hollow sneaker shell and a lacing cradle (MuJoCo MjSpec).

Conventions (shared with the other blocks of the shoe_lacing_and_boxing task)
-----------------------------------------------------------------------------
* units: metres / radians / seconds / kg; quaternions wxyz (MuJoCo order).
* world frame: bench top plane is z = 0, x along the bench, y across, z up.
* every builder takes ``(spec, parent_body, name, **params)``, prefixes every
  name it creates with ``'<name>/'`` and returns a plain dict descriptor
  (names + dimensions + local key points) that later code uses by name.

Shoe local frame
----------------
heel at -x, toe at +x, y across (the wearer's LEFT side is +y, side 'L'),
z up, sole bottom at z = 0.  The body origin is the centre of the sole bottom.

The shoe is ONE rigid body made of boxes / cylinders / capsules.  It is hollow:
the cavity (sole top .. underside of the eyelet flaps) is reachable from above
through the throat slot between the two horizontal eyelet flaps.  Each flap is
a 3 mm plate at height ``z_f`` with ``n_rows`` eyelet rings; the plate is built
from box segments between consecutive rings plus two edge strips, so every
ring hole is a real hole (a vertical ray through a hole centre reaches the
insole).

Parameter conflict and how it is resolved
-----------------------------------------
The task sheet asks for an outer width W in [0.095, 0.11], a flap width
wf = 0.022 measured from the wall inner face, AND a throat gap g in
[0.028, 0.034] between the flap inner edges.  These cannot all hold
(Wc - 2*wf is 0.039..0.054).  The eyelet positions (y = +-(g/2 + 0.009)) and
the throat gap matter most for lacing, so g is drawn as specified, the flap
inner edge sits at |y| = g/2 and the flap extends outward to the wall inner
face (sealed to the wall, no crack a lace could fall into).  The actual flap
width wf = (Wc - g)/2 (0.0245..0.035) is reported in the descriptor next to
``wf_nominal = 0.022``.

Row positions: the sheet writes x_k = x_toecap_back + 0.02 + k*s; with the toe
at +x the rows must run from the toe cap towards the heel, so
x_k = x_toecap_back - 0.02 - k*s (row 0 is nearest the toe).

Runtime dependencies: mujoco + numpy only.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import mujoco
import numpy as np

# --------------------------------------------------------------------------
# fixed design constants (metres)
# --------------------------------------------------------------------------
SOLE_H = 0.015          # sole thickness; sole top at z = SOLE_H
WALL_T = 0.006          # side wall thickness
FLAP_T = 0.003          # eyelet flap plate thickness
TOE_CAP_LEN = 0.06      # closed length at the toe
HEEL_LEN = 0.03         # closed length at the heel
TOE_CAP_RISE = 0.015    # toe cap top above the flap level
HEEL_RISE = 0.025       # heel counter top above the flap level
CAVITY_DEPTH = 0.035    # flap underside above the sole top ("dc")
INSOLE_T = 0.002        # thin insole plate lying on the sole top
HOLE_R = 0.0045         # eyelet hole radius (minimum clearance)
WIRE_R = 0.0012         # eyelet ring wire radius
EYELET_INSET = 0.009    # hole centre inward of the flap inner edge
FIRST_ROW_OFFSET = 0.02  # first row behind the toe cap back face
WF_NOMINAL = 0.022
STEEL_DENSITY = 7800.0
ZF = 0.0004             # z-fighting offset: overlapping geoms never share a visible face
Z_LIFT = 0.0005         # sole end pieces start this high: only the main sole box touches the bench
TOE_CORNER_R = TOE_CAP_LEN / 2    # rounding radius of the toe corners (0.03)
HEEL_CORNER_R = HEEL_LEN / 2      # rounding radius of the heel corners (0.015)

UPPER_PALETTE: Dict[str, Tuple[float, float, float, float]] = {
    'red':    (0.85, 0.10, 0.10, 1.0),
    'blue':   (0.10, 0.30, 0.90, 1.0),
    'green':  (0.10, 0.70, 0.20, 1.0),
    'orange': (1.00, 0.50, 0.05, 1.0),
    'purple': (0.60, 0.15, 0.85, 1.0),
    'yellow': (0.95, 0.85, 0.05, 1.0),
}
SOLE_PALETTE: Dict[str, Tuple[float, float, float, float]] = {
    'white': (0.92, 0.92, 0.92, 1.0),
    'black': (0.08, 0.08, 0.08, 1.0),
    'grey':  (0.50, 0.50, 0.50, 1.0),
}
RING_RGBA = (0.85, 0.85, 0.90, 1.0)
INSOLE_RGBA = (0.82, 0.72, 0.52, 1.0)
STITCH_RGBA = (0.12, 0.12, 0.12, 1.0)


# --------------------------------------------------------------------------
# small math helpers (quaternions wxyz)
# --------------------------------------------------------------------------
def quat_rotz(yaw: float) -> Tuple[float, float, float, float]:
    """Quaternion (wxyz) of a rotation about +z by ``yaw``."""
    return (math.cos(0.5 * yaw), 0.0, 0.0, math.sin(0.5 * yaw))


def quat_mul(a, b) -> np.ndarray:
    """Hamilton product a*b (both wxyz)."""
    out = np.zeros(4)
    mujoco.mju_mulQuat(out, np.asarray(a, dtype=float), np.asarray(b, dtype=float))
    return out


def quat_to_mat(q) -> np.ndarray:
    """3x3 rotation matrix of quaternion q (wxyz)."""
    out = np.zeros(9)
    mujoco.mju_quat2Mat(out, np.asarray(q, dtype=float))
    return out.reshape(3, 3)


def quat_rotate(q, v) -> np.ndarray:
    """Rotate vector v by quaternion q."""
    out = np.zeros(3)
    mujoco.mju_rotVecQuat(out, np.asarray(v, dtype=float), np.asarray(q, dtype=float))
    return out


def yaw_of_mat(R: np.ndarray) -> float:
    """Yaw (about world z) of the x axis of rotation matrix R."""
    return math.atan2(R[1, 0], R[0, 0])


def _unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if n < 1e-12:
        raise ValueError('zero-length axis')
    return v / n


def _basis_perp(axis: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Two unit vectors orthogonal to ``axis`` (deterministic choice)."""
    a = _unit(axis)
    helper = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = _unit(np.cross(a, helper))
    v = np.cross(a, u)
    return u, v


def _f(x) -> List[float]:
    """Plain python list of floats (keeps descriptors JSON-able)."""
    return [float(t) for t in np.asarray(x, dtype=float).ravel()]


# --------------------------------------------------------------------------
# parameters
# --------------------------------------------------------------------------
def shoe_params(rng) -> dict:
    """Draw a shoe parameter set from ``rng`` (int seed or numpy Generator).

    Returns a plain dict:  L, W, Wc, n_rows, s, g, wf (actual), wf_nominal,
    dc, upper, sole (palette keys), upper_rgba, sole_rgba, mass.
    """
    rng = np.random.default_rng(rng)
    L = float(rng.uniform(0.24, 0.30))
    W = float(rng.uniform(0.095, 0.11))
    n_rows = int(rng.integers(4, 7))          # 4..6 inclusive
    s = float(rng.uniform(0.018, 0.022))
    g = float(rng.uniform(0.028, 0.034))
    upper = list(UPPER_PALETTE)[int(rng.integers(0, len(UPPER_PALETTE)))]
    sole = list(SOLE_PALETTE)[int(rng.integers(0, len(SOLE_PALETTE)))]
    Wc = W - 2.0 * WALL_T
    return dict(L=L, W=W, Wc=Wc, n_rows=n_rows, s=s, g=g,
                wf=(Wc - g) / 2.0, wf_nominal=WF_NOMINAL, dc=CAVITY_DEPTH,
                upper=upper, sole=sole,
                upper_rgba=list(UPPER_PALETTE[upper]),
                sole_rgba=list(SOLE_PALETTE[sole]),
                mass=0.25)


# --------------------------------------------------------------------------
# geom helpers (every helper returns (geom, volume))
# --------------------------------------------------------------------------
def _box_aabb(body, name, lo, hi, rgba, **kw):
    """Axis-aligned box geom spanning [lo, hi] in the body frame."""
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)
    half = 0.5 * (hi - lo)
    if np.any(half <= 0):
        raise ValueError(f'degenerate box {name}: lo={lo} hi={hi}')
    g = body.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_BOX,
                      pos=_f(0.5 * (lo + hi)), size=_f(half), rgba=list(rgba), **kw)
    return g, float(8.0 * half[0] * half[1] * half[2])


def _cyl_z(body, name, cx, cy, z0, z1, r, rgba, **kw):
    """Vertical cylinder geom (axis = body z) between z0 and z1."""
    h = 0.5 * (z1 - z0)
    g = body.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                      pos=[float(cx), float(cy), float(0.5 * (z0 + z1))],
                      size=[float(r), float(h), 0.0], rgba=list(rgba), **kw)
    return g, float(math.pi * r * r * 2.0 * h)


def _capsule_volume(p1, p2, r) -> float:
    ell = float(np.linalg.norm(np.asarray(p2, float) - np.asarray(p1, float)))
    return math.pi * r * r * ell + 4.0 / 3.0 * math.pi * r ** 3


def add_eyelet_ring(spec, body, name, center, axis, hole_radius, wire_radius,
                    n_capsules=8, rgba=RING_RGBA, **geom_kw) -> dict:
    """Add a torus-like eyelet ring made of ``n_capsules`` overlapping capsules.

    The capsules are the edges of a regular polygon whose *apothem* equals
    ``hole_radius + wire_radius``, so the clear hole has radius >= hole_radius
    everywhere (exactly hole_radius at the edge midpoints, which lie on the
    +-u / +-v directions of the ring frame).  Geoms are named
    ``'<name>/c<j>'``.  Extra keyword arguments go to ``add_geom``
    (e.g. density, condim, friction).

    Returns {'geoms', 'center', 'axis', 'hole_radius', 'wire_radius',
    'vertex_radius', 'outer_radius', 'volume'}.
    """
    c = np.asarray(center, dtype=float)
    a = _unit(axis)
    u, v = _basis_perp(a)
    r_mid = hole_radius + wire_radius
    r_v = r_mid / math.cos(math.pi / n_capsules)
    pts = []
    for j in range(n_capsules):
        th = 2.0 * math.pi * j / n_capsules + math.pi / n_capsules
        pts.append(c + r_v * (math.cos(th) * u + math.sin(th) * v))
    names, vol = [], 0.0
    for j in range(n_capsules):
        p1, p2 = pts[j], pts[(j + 1) % n_capsules]
        gname = f'{name}/c{j}'
        body.add_geom(name=gname, type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                      fromto=_f(np.concatenate([p1, p2])),
                      size=[float(wire_radius), 0.0, 0.0], rgba=list(rgba), **geom_kw)
        names.append(gname)
        vol += _capsule_volume(p1, p2, wire_radius)
    return dict(geoms=names, center=_f(c), axis=_f(a), hole_radius=float(hole_radius),
                wire_radius=float(wire_radius), vertex_radius=float(r_v),
                outer_radius=float(r_v + wire_radius), volume=float(vol))


# --------------------------------------------------------------------------
# the shoe
# --------------------------------------------------------------------------
def add_shoe(spec, parent, name, params=None, seed=0, pos=(0.0, 0.0, 0.0),
             quat=(1.0, 0.0, 0.0, 0.0), free=True, condim=3,
             friction=(0.8, 0.005, 0.0001), solref=None, solimp=None,
             ring_condim=None, ring_friction=None, stitches=True) -> dict:
    """Build one rigid hollow sneaker shell under ``parent`` (normally worldbody).

    params : dict from :func:`shoe_params`, or an int seed, or None (-> seed).
    pos, quat : initial pose of the shoe body (sole-bottom centre) in the
        parent frame; with ``free=True`` a free joint ``'<name>/free'`` is added
        (the parent must then be the worldbody).
    condim, friction : contact parameters of all collision geoms; solref /
        solimp (optional) are applied to all collision geoms as well.
    ring_condim, ring_friction : override for the eyelet ring capsules (e.g.
        condim=1 to let a lace slide freely); default = same as the shell.
    stitches : add a few visual-only dark seam geoms (contype 0, group 1).

    Returns the descriptor (plain dict): body / joint / site / geom names,
    params, dims, eyelets (row, side, center, axis, geoms), throat, cavity,
    aabb, grip info.
    """
    if params is None:
        params = shoe_params(np.random.default_rng(seed))
    elif isinstance(params, (int, np.integer)):
        params = shoe_params(np.random.default_rng(int(params)))
    p = dict(params)
    L, W, g, s, n_rows = p['L'], p['W'], p['g'], p['s'], int(p['n_rows'])
    Wc = p.get('Wc', W - 2 * WALL_T)
    dc = p.get('dc', CAVITY_DEPTH)
    mass = float(p.get('mass', 0.25))
    upper_rgba = tuple(p.get('upper_rgba', UPPER_PALETTE[p.get('upper', 'red')]))
    sole_rgba = tuple(p.get('sole_rgba', SOLE_PALETTE[p.get('sole', 'white')]))
    toe_cap_len = float(p.get('toe_cap_len', TOE_CAP_LEN))
    heel_len = float(p.get('heel_len', HEEL_LEN))
    first_row_offset = float(p.get('first_row_offset', FIRST_ROW_OFFSET))
    toe_corner_r = toe_cap_len / 2.0
    heel_corner_r = heel_len / 2.0
    if W / 2.0 <= toe_corner_r + 0.005:
        raise ValueError('shoe too narrow for the toe corner rounding')

    z_f = SOLE_H + dc + FLAP_T          # flap TOP surface; plate occupies [z_f - FLAP_T, z_f]
    x_tc = L / 2.0 - toe_cap_len        # toe cap back face
    x_hc = -L / 2.0 + heel_len          # heel counter front face
    z_tc_top = z_f + TOE_CAP_RISE
    z_hc_top = z_f + HEEL_RISE
    y_e = g / 2.0 + EYELET_INSET        # |y| of the eyelet hole centres
    z_floor = SOLE_H + INSOLE_T         # cavity floor (insole top)
    x_rows = [x_tc - first_row_offset - k * s for k in range(n_rows)]
    if x_rows[-1] - HOLE_R - WIRE_R * 2 <= x_hc + 0.002:
        raise ValueError('eyelet rows do not fit between toe cap and heel counter')

    ckw = dict(condim=int(condim), friction=list(friction))
    if solref is not None:
        ckw['solref'] = list(solref)
    if solimp is not None:
        ckw['solimp'] = list(solimp)
    rkw = dict(ckw)
    rkw['priority'] = 2                 # the ring's condim / friction win over the lace's (priority 1, condim 1)
    if ring_condim is not None:
        rkw['condim'] = int(ring_condim)
    if ring_friction is not None:
        rkw['friction'] = list(ring_friction)
    body = parent.add_body(name=f'{name}/body', pos=_f(pos), quat=_f(quat))
    joint_name = None
    if free:
        joint_name = f'{name}/free'
        body.add_freejoint(name=joint_name)

    shell: List[Tuple[object, float]] = []   # (geom, volume) of the shell geoms
    groups: Dict[str, List[str]] = {}

    def add(group, geom_vol):
        geom, vol = geom_vol
        shell.append((geom, vol))
        groups.setdefault(group, []).append(geom.name)

    pre = f'{name}/'
    r_t, r_h = toe_corner_r, heel_corner_r
    # sole: rounded-corner slab = main box + two end boxes + four corner cylinders.
    # Only the main box reaches z = 0; the end pieces start Z_LIFT (0.5 mm) higher,
    # so a shoe resting flat has one box/plane contact patch (4 contacts) instead of ~24.
    add('sole', _box_aabb(body, pre + 'sole/main', (-L / 2 + r_h, -W / 2, 0.0), (L / 2 - r_t, W / 2, SOLE_H), sole_rgba, **ckw))
    add('sole', _box_aabb(body, pre + 'sole/heel', (-L / 2, -(W / 2 - r_h), Z_LIFT), (-L / 2 + r_h, W / 2 - r_h, SOLE_H), sole_rgba, **ckw))
    add('sole', _box_aabb(body, pre + 'sole/toe', (L / 2 - r_t, -(W / 2 - r_t), Z_LIFT), (L / 2, W / 2 - r_t, SOLE_H), sole_rgba, **ckw))
    for side, sg in (('L', 1.0), ('R', -1.0)):
        add('sole', _cyl_z(body, pre + f'sole/heel_{side}', -L / 2 + r_h, sg * (W / 2 - r_h), Z_LIFT + ZF, SOLE_H, r_h, sole_rgba, **ckw))
        add('sole', _cyl_z(body, pre + f'sole/toe_{side}', L / 2 - r_t, sg * (W / 2 - r_t), Z_LIFT + ZF, SOLE_H, r_t, sole_rgba, **ckw))
    # heel counter: closes the back HEEL_LEN up to z_hc_top, back corners rounded (r_h)
    add('heel_counter', _box_aabb(body, pre + 'heel/box', (-L / 2 + r_h, -W / 2, SOLE_H), (x_hc, W / 2, z_hc_top), upper_rgba, **ckw))
    add('heel_counter', _box_aabb(body, pre + 'heel/back', (-L / 2, -(W / 2 - r_h), SOLE_H), (-L / 2 + r_h, W / 2 - r_h, z_hc_top - ZF), upper_rgba, **ckw))
    for side, sg in (('L', 1.0), ('R', -1.0)):
        add('heel_counter', _cyl_z(body, pre + f'heel/corner_{side}', -L / 2 + r_h, sg * (W / 2 - r_h), SOLE_H, z_hc_top - 2 * ZF, r_h, upper_rgba, **ckw))
    # toe cap: closes the front TOE_CAP_LEN up to z_tc_top, front corners rounded (r_t)
    add('toe_cap', _box_aabb(body, pre + 'toe/box', (x_tc, -W / 2, SOLE_H), (L / 2 - r_t, W / 2, z_tc_top), upper_rgba, **ckw))
    add('toe_cap', _box_aabb(body, pre + 'toe/front', (L / 2 - r_t, -(W / 2 - r_t), SOLE_H), (L / 2, W / 2 - r_t, z_tc_top - ZF), upper_rgba, **ckw))
    for side, sg in (('L', 1.0), ('R', -1.0)):
        add('toe_cap', _cyl_z(body, pre + f'toe/corner_{side}', L / 2 - r_t, sg * (W / 2 - r_t), SOLE_H, z_tc_top - 2 * ZF, r_t, upper_rgba, **ckw))
    # side walls (sole top .. flap top), overlapping 2 mm into toe cap / heel counter
    for side, sg in (('L', 1.0), ('R', -1.0)):
        ys = sorted([sg * Wc / 2, sg * W / 2])
        add('walls', _box_aabb(body, pre + f'wall/{side}', (x_hc - 0.002, ys[0], SOLE_H), (x_tc + 0.002, ys[1], z_f - ZF), upper_rgba, **ckw))
    # insole plate on the sole top (cavity floor)
    add('insole', _box_aabb(body, pre + 'insole', (x_hc - 0.002, -Wc / 2, SOLE_H), (x_tc + 0.002, Wc / 2, z_floor), INSOLE_RGBA, **ckw))

    # eyelet flaps: edge strips + segments between rings, then the rings
    eyelets = []
    ring_vol = 0.0
    y_out = Wc / 2 + WALL_T / 2          # overlap half-way into the wall (sealed)
    z0, z1 = z_f - FLAP_T, z_f
    xa, xb = x_hc - 0.002, x_tc + 0.002  # overlap into heel counter / toe cap
    for side, sg in (('L', 1.0), ('R', -1.0)):
        grp = f'flap_{side}'

        def ybox(lo, hi):
            return sorted([sg * lo, sg * hi])

        ys = ybox(g / 2, y_e - HOLE_R)
        add(grp, _box_aabb(body, pre + f'flap/{side}/inner', (xa, ys[0], z0 + ZF), (xb, ys[1], z1 - ZF), upper_rgba, **ckw))
        ys = ybox(y_e + HOLE_R, y_out)
        add(grp, _box_aabb(body, pre + f'flap/{side}/outer', (xa, ys[0], z0 + ZF), (xb, ys[1], z1 - ZF), upper_rgba, **ckw))
        ys = ybox(g / 2, y_out)
        # segment boxes: front (toe cap .. row 0), between rows, back (last row .. heel counter)
        bounds = [xb] + x_rows + [xa]
        for k in range(n_rows + 1):
            hi = bounds[k] - (HOLE_R if k > 0 else 0.0)
            lo = bounds[k + 1] + (HOLE_R if k < n_rows else 0.0)
            add(grp, _box_aabb(body, pre + f'flap/{side}/seg{k}', (lo, ys[0], z0), (hi, ys[1], z1), upper_rgba, **ckw))
        for k, xk in enumerate(x_rows):
            rname = pre + f'eyelet/{side}{k}'
            ring = add_eyelet_ring(spec, body, rname, (xk, sg * y_e, z_f), (0.0, 0.0, 1.0),
                                   HOLE_R, WIRE_R, 8, RING_RGBA, density=STEEL_DENSITY, **rkw)
            ring_vol += ring['volume']
            groups.setdefault('rings', []).extend(ring['geoms'])
            eyelets.append(dict(row=k, side=side, center=ring['center'], axis=ring['axis'],
                                hole_radius=HOLE_R, wire_radius=WIRE_R,
                                outer_radius=ring['outer_radius'], geoms=ring['geoms']))

    # visual-only stitches / seams (no collision, no mass)
    if stitches:
        vkw = dict(contype=0, conaffinity=0, group=1, density=0.0)
        body.add_geom(name=pre + 'stitch/toe_seam', type=mujoco.mjtGeom.mjGEOM_BOX,
                      pos=[x_tc + 0.0015, 0.0, z_tc_top], size=[0.0015, W / 2 + 0.0002, 0.0004],
                      rgba=list(STITCH_RGBA), **vkw)
        for side, sg in (('L', 1.0), ('R', -1.0)):
            body.add_geom(name=pre + f'stitch/collar_{side}', type=mujoco.mjtGeom.mjGEOM_BOX,
                          pos=[0.5 * (x_hc + x_tc), sg * (W / 2 - 0.001), z_f],
                          size=[0.5 * (x_tc - x_hc), 0.0012, 0.0004], rgba=list(STITCH_RGBA), **vkw)
        groups['stitches'] = [pre + 'stitch/toe_seam', pre + 'stitch/collar_L', pre + 'stitch/collar_R']

    # mass: metallic rings at steel density, the shell shares the rest of ``mass``
    m_rings = STEEL_DENSITY * ring_vol
    v_shell = sum(v for _, v in shell)
    rho = max(mass - m_rings, 0.05) / v_shell
    for geom, _ in shell:
        geom.density = float(rho)

    # sites: grip handles (site z = +z, site y = pinch direction = local x) and centre
    grip_quat = quat_rotz(-math.pi / 2)      # y_site = x_local, z_site = z_local
    heel_grip_pos = (-L / 2 + heel_len / 2, 0.0, z_hc_top)
    toe_grip_pos = (L / 2 - toe_cap_len / 2, 0.0, z_tc_top)
    centre_pos = (0.0, 0.0, 0.5 * z_hc_top)
    skw = dict(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.003, 0, 0], group=3)
    body.add_site(name=pre + 'heel_grip', pos=_f(heel_grip_pos), quat=_f(grip_quat), rgba=[1, 0, 0, 0.6], **skw)
    body.add_site(name=pre + 'toe_grip', pos=_f(toe_grip_pos), quat=_f(grip_quat), rgba=[0, 1, 0, 0.6], **skw)
    body.add_site(name=pre + 'centre', pos=_f(centre_pos), rgba=[0, 0, 1, 0.6], **skw)

    all_geoms = [gname for lst in groups.values() for gname in lst]
    dims = dict(L=L, W=W, Wc=Wc, g=g, s=s, n_rows=n_rows, wf=(Wc - g) / 2, wf_nominal=WF_NOMINAL,
                dc=dc, sole_h=SOLE_H, wall_t=WALL_T, flap_t=FLAP_T, z_f=z_f, z_flap_bottom=z_f - FLAP_T,
                x_tc=x_tc, x_hc=x_hc, z_tc_top=z_tc_top, z_hc_top=z_hc_top, z_top=z_hc_top,
                z_floor=z_floor, toe_corner_radius=r_t, heel_corner_radius=r_h,
                toe_corner_centres=[[L / 2 - r_t, W / 2 - r_t], [L / 2 - r_t, -(W / 2 - r_t)]],
                heel_corner_centres=[[-L / 2 + r_h, W / 2 - r_h], [-L / 2 + r_h, -(W / 2 - r_h)]],
                z_lift=Z_LIFT,
                hole_radius=HOLE_R, wire_radius=WIRE_R, ring_outer_radius=eyelets[0]['outer_radius'],
                eyelet_y=y_e, x_rows=x_rows)
    desc = dict(
        name=name, body=f'{name}/body', joint=joint_name,
        sites=dict(heel_grip=pre + 'heel_grip', toe_grip=pre + 'toe_grip', centre=pre + 'centre'),
        geoms=groups, geom_names=all_geoms,
        params=p, dims=dims, mass=mass, mass_rings=float(m_rings),
        eyelets=eyelets,
        throat=dict(x=[x_hc, x_tc], y=[-g / 2, g / 2], z=z_f),
        cavity=dict(min=[x_hc, -Wc / 2, z_floor], max=[x_tc, Wc / 2, z_f - FLAP_T]),
        aabb=dict(min=[-L / 2, -W / 2, 0.0], max=[L / 2, W / 2, z_hc_top]),
        grip=dict(heel_grip=dict(pos=_f(heel_grip_pos), quat=_f(grip_quat), pinch_width=heel_len,
                                 pinch_axis_local=[1.0, 0.0, 0.0]),
                  toe_grip=dict(pos=_f(toe_grip_pos), quat=_f(grip_quat), pinch_width=toe_cap_len,
                                pinch_axis_local=[1.0, 0.0, 0.0])),
        centre_local=_f(centre_pos),
    )
    return desc


# --------------------------------------------------------------------------
# descriptor helpers
# --------------------------------------------------------------------------
_NAME_KEYS = ('body', 'joint', 'eq', 'shoe_body')


def prefix_desc(desc: dict, prefix: str) -> dict:
    """Copy of a shoe / cradle descriptor with ``prefix`` prepended to every
    MuJoCo element name (body, joint, sites, geoms, equality).  Use it when the
    shoe was built in a child MjSpec that is attached with
    ``spec.attach(child, prefix=prefix, ...)``."""
    def fix(v):
        return prefix + v if isinstance(v, str) else v
    out = dict(desc)
    for k in _NAME_KEYS:
        if out.get(k) is not None:
            out[k] = fix(out[k])
    if 'sites' in out:
        out['sites'] = {k: fix(v) for k, v in out['sites'].items()}
    if 'geoms' in out and isinstance(out['geoms'], dict):
        out['geoms'] = {k: [fix(n) for n in v] for k, v in out['geoms'].items()}
    elif 'geoms' in out:
        out['geoms'] = [fix(n) for n in out['geoms']]
    if 'geom_names' in out:
        out['geom_names'] = [fix(n) for n in out['geom_names']]
    if 'eyelets' in out:
        out['eyelets'] = [dict(e, geoms=[fix(n) for n in e['geoms']]) for e in out['eyelets']]
    return out


# --------------------------------------------------------------------------
# runtime queries
# --------------------------------------------------------------------------
def _body_id(model, desc) -> int:
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, desc['body'])


def shoe_pose(model, data, desc) -> Tuple[np.ndarray, np.ndarray]:
    """(pos, quat wxyz) of the shoe body origin (sole-bottom centre) in world."""
    bid = _body_id(model, desc)
    return data.xpos[bid].copy(), data.xquat[bid].copy()


def shoe_rotation(model, data, desc) -> np.ndarray:
    """3x3 world-from-shoe rotation matrix."""
    return data.xmat[_body_id(model, desc)].reshape(3, 3).copy()


def set_shoe_pose(model, data, desc, pos, quat=(1.0, 0.0, 0.0, 0.0), forward=True) -> None:
    """Write the free-joint qpos (and zero its qvel); calls mj_forward if ``forward``."""
    if desc['joint'] is None:
        raise ValueError('shoe has no free joint')
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, desc['joint'])
    qa, va = model.jnt_qposadr[jid], model.jnt_dofadr[jid]
    data.qpos[qa:qa + 3] = np.asarray(pos, dtype=float)
    data.qpos[qa + 3:qa + 7] = np.asarray(quat, dtype=float)
    data.qvel[va:va + 6] = 0.0
    if forward:
        mujoco.mj_forward(model, data)


def shoe_velocity(model, data, desc) -> Tuple[np.ndarray, np.ndarray]:
    """(linear, angular) velocity of the shoe free joint (world frame, m/s, rad/s)."""
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, desc['joint'])
    va = model.jnt_dofadr[jid]
    return data.qvel[va:va + 3].copy(), data.qvel[va + 3:va + 6].copy()


def eyelets_world(model, data, desc) -> List[dict]:
    """Eyelets in world coordinates: list of {row, side, center, axis} (numpy arrays)."""
    bid = _body_id(model, desc)
    R = data.xmat[bid].reshape(3, 3)
    p = data.xpos[bid]
    out = []
    for e in desc['eyelets']:
        out.append(dict(row=e['row'], side=e['side'],
                        center=p + R @ np.asarray(e['center']),
                        axis=R @ np.asarray(e['axis'])))
    return out


def world_to_shoe(model, data, desc, point_world) -> np.ndarray:
    """Express a world point in the shoe local frame."""
    bid = _body_id(model, desc)
    R = data.xmat[bid].reshape(3, 3)
    return R.T @ (np.asarray(point_world, dtype=float) - data.xpos[bid])


def inside_cavity(model, data, desc, point_world, margin=0.0) -> bool:
    """True when the world point lies inside the cavity AABB (shoe frame), shrunk by ``margin``."""
    q = world_to_shoe(model, data, desc, point_world)
    lo = np.asarray(desc['cavity']['min']) + margin
    hi = np.asarray(desc['cavity']['max']) - margin
    return bool(np.all(q >= lo) and np.all(q <= hi))


def upright(model, data, desc, max_tilt_deg=10.0) -> bool:
    """True when the sole normal (shoe +z) is within ``max_tilt_deg`` of world +z."""
    R = data.xmat[_body_id(model, desc)].reshape(3, 3)
    return bool(R[2, 2] >= math.cos(math.radians(max_tilt_deg)))


# --------------------------------------------------------------------------
# cradle
# --------------------------------------------------------------------------
def add_cradle(spec, parent, name, pos, yaw, shoe_desc, clearance=0.004, plate_t=0.008,
               bumper_h=0.012, bumper_t=0.006, bumper_len=0.016,
               rgba_plate=(0.30, 0.30, 0.34, 1.0), rgba_bumper=(0.16, 0.16, 0.18, 1.0),
               clamp_solref=(0.005, 1.0), clamp_solimp=(0.95, 0.99, 0.001, 0.5, 2.0),
               condim=3, friction=(0.8, 0.005, 0.0001)) -> dict:
    """Static lacing cradle for ``shoe_desc`` (any shoe of the same dims fits).

    ``pos`` (3) and ``yaw`` are the cradle pose in the parent frame (normally
    the worldbody; ``pos[2]`` is the bench surface, usually 0).  Geoms (all in a
    static body ``'<name>/base'``): base plate (L+0.02 x W+0.02 x plate_t),
    two posts alongside the heel, two posts tangent to the rounded toe corners
    at 45 deg, and a heel stop; every stop face is ``clearance`` away from the sole when the
    shoe sits sole-down heel-back at the cradle pose.  A weld equality
    ``'<name>/clamp'`` (world <-> shoe body, inactive) is the clamp; see
    :func:`set_clamp`.
    """
    d = shoe_desc['dims']
    L, W = d['L'], d['W']
    c, b, hl = clearance, bumper_t, bumper_len
    base = parent.add_body(name=f'{name}/base', pos=_f(pos), quat=_f(quat_rotz(yaw)))
    ckw = dict(condim=int(condim), friction=list(friction))
    pre = f'{name}/'
    geoms = []

    def box(gname, centre, half, rgba, quat=(1, 0, 0, 0)):
        base.add_geom(name=pre + gname, type=mujoco.mjtGeom.mjGEOM_BOX, pos=_f(centre),
                      size=_f(half), quat=_f(quat), rgba=list(rgba), **ckw)
        geoms.append(pre + gname)

    zc = plate_t + bumper_h / 2
    box('plate', (0, 0, plate_t / 2), (L / 2 + 0.01, W / 2 + 0.01, plate_t / 2), rgba_plate)
    # heel-side posts alongside the straight part of the sole just ahead of the rounded
    # heel corners (inner face at |y| = W/2 + c)
    r_h = d['heel_corner_radius']
    for side, sg in (('L', 1.0), ('R', -1.0)):
        box(f'post_heel_{side}', (-L / 2 + r_h + hl / 2, sg * (W / 2 + c + b / 2), zc), (hl / 2, b / 2, bumper_h / 2), rgba_bumper)
    # toe posts tangent (at 45 deg) to the rounded toe corners, offset by c (box x axis = radial)
    r_t = d['toe_corner_radius']
    for (cx, cy), sg in zip(d['toe_corner_centres'], (1.0, -1.0)):
        phi = sg * math.pi / 4
        rad = r_t + c + b / 2
        box(f'post_toe_{"L" if sg > 0 else "R"}', (cx + rad * math.cos(phi), cy + rad * math.sin(phi), zc),
            (b / 2, hl / 2, bumper_h / 2), rgba_bumper, quat=quat_rotz(phi))
    # heel stop (inner face at x = -L/2 - c), joined to the heel posts
    box('heel_stop', (-L / 2 - c - b / 2, 0, zc), (b / 2, W / 2 + c + b, bumper_h / 2), rgba_bumper)

    eq_name = pre + 'clamp'
    eq = spec.add_equality(type=mujoco.mjtEq.mjEQ_WELD, name=eq_name, objtype=mujoco.mjtObj.mjOBJ_BODY,
                           name1='world', name2=shoe_desc['body'], active=False,
                           data=[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1])
    eq.solref = list(clamp_solref)
    eq.solimp = list(clamp_solimp)

    q = quat_rotz(yaw)
    seat_local = (0.0, 0.0, plate_t)
    seat_pos = np.asarray(pos, dtype=float) + quat_rotate(q, seat_local)
    return dict(name=name, body=f'{name}/base', pos=_f(pos), yaw=float(yaw), quat=_f(q),
                seat_local=_f(seat_local), seat_pos=_f(seat_pos), seat_quat=_f(q),
                plate_top=float(pos[2] + plate_t), clearance=float(c), geoms=geoms,
                eq=eq_name, shoe_body=shoe_desc['body'],
                dims=dict(L=L, W=W, plate_t=plate_t, bumper_h=bumper_h, bumper_t=b, bumper_len=hl,
                          plate_size=[L + 0.02, W + 0.02, plate_t]),
                tolerances=dict(z=0.004, xy=0.006, yaw_deg=4.0, tilt_deg=10.0))


def cradle_pose(desc, model=None, data=None) -> Tuple[np.ndarray, np.ndarray]:
    """(pos, quat) of a shoe body seated in the cradle.

    Without model/data the cradle parent is assumed to be the worldbody (pose
    composed at build time); with them the live pose of the base body is used
    (kinematics must be current: call mj_forward / mj_step first).
    """
    if model is None or data is None:
        return np.asarray(desc['seat_pos'], float), np.asarray(desc['seat_quat'], float)
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, desc['body'])
    p, q = data.xpos[bid], data.xquat[bid]
    return p + quat_rotate(q, desc['seat_local']), q.copy()


def seated(model, data, shoe_desc, cradle_desc, verbose=False) -> bool:
    """Shoe sole-down heel-back in the cradle: origin z within 0.004 of the plate top,
    xy within 0.006 of the seat, yaw within 4 deg, upright (sole normal within 10 deg)."""
    tol = cradle_desc['tolerances']
    seat_p, seat_q = cradle_pose(cradle_desc, model, data)
    p, _ = shoe_pose(model, data, shoe_desc)
    R = shoe_rotation(model, data, shoe_desc)
    dz = abs(float(p[2] - seat_p[2]))
    dxy = float(np.linalg.norm(p[:2] - seat_p[:2]))
    dyaw = yaw_of_mat(R) - yaw_of_mat(quat_to_mat(seat_q))
    dyaw = abs(math.degrees(math.atan2(math.sin(dyaw), math.cos(dyaw))))
    up = upright(model, data, shoe_desc, tol['tilt_deg'])
    ok = dz <= tol['z'] and dxy <= tol['xy'] and dyaw <= tol['yaw_deg'] and up
    if verbose:
        print(f'  seated: dz={dz:.4f} dxy={dxy:.4f} dyaw={dyaw:.2f}deg upright={up} -> {ok}')
    return bool(ok)


def set_clamp(model, data, cradle_desc, shoe_desc, on: bool) -> None:
    """Toggle the cradle clamp: weld ``shoe_desc['body']`` to the world at its current pose."""
    eq = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, cradle_desc['eq'])
    if on:
        bid = _body_id(model, shoe_desc)
        model.eq_obj1id[eq] = 0
        model.eq_obj2id[eq] = bid
        model.eq_objtype[eq] = mujoco.mjtObj.mjOBJ_BODY
        model.eq_data[eq, 0:3] = 0.0
        model.eq_data[eq, 3:6] = data.xpos[bid]
        model.eq_data[eq, 6:10] = data.xquat[bid]
        model.eq_data[eq, 10] = 1.0
        data.eq_active[eq] = 1
    else:
        data.eq_active[eq] = 0


def clamp_active(model, data, cradle_desc) -> bool:
    eq = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, cradle_desc['eq'])
    return bool(data.eq_active[eq])
