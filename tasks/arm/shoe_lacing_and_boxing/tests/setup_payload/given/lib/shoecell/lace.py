"""lace.py - a shoelace as a chain of capsules, eyelet rings and lace-topology helpers.

Building block for the "shoe_lacing_and_boxing" MuJoCo benchmark task.

Conventions (shared with the other blocks)
------------------------------------------
* units: metres / radians / seconds / kg; quaternions are wxyz (MuJoCo order);
  world frame: bench top is z = 0, x along the bench, y across it, z up.
* every builder takes ``(spec, parent_body, name, **params)``, names everything
  it adds with the prefix ``'<name>/'`` and returns a plain dict descriptor.
* runtime dependencies: mujoco (>= 3.1, MjSpec API) and numpy only.

Lace model
----------
A serial chain of ``n_segments`` rigid capsules.  Body ``<name>/seg00`` (the
first aglet) carries a free joint; every following body ``<name>/segXX`` is a
child of the previous one, placed at the end of the parent's capsule, and
carries a ball joint ``<name>/jXX`` with a small bending stiffness/damping
(spring reference = straight).  Each capsule runs from the body origin along
the body's local +z axis.  The *joint points* of the chain are therefore the
body origins plus the far end of the last capsule; ``lace_polyline`` returns
them in world coordinates (``n_segments + 1`` points from aglet A's tip to
aglet B's tip).

Site ``<name>/aglet_a`` sits at the tip of the first aglet with its z axis
pointing *out* of the lace (away from the chain), ``<name>/aglet_b`` at the tip
of the last aglet, z pointing out as well.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Sequence, Tuple

import mujoco
import numpy as np

__all__ = [
    "configure_options",
    "add_lace",
    "add_eyelet_ring",
    "add_site_grip",
    "add_plate_with_hole",
    "ring_world",
    "ring_tuple",
    "lace_body_ids",
    "lace_polyline",
    "set_lace_polyline",
    "straight_polyline",
    "polyline_from_directions",
    "lace_segment_lengths",
    "lace_length",
    "segment_slack",
    "ring_crossings",
    "pass_sequence",
    "grip_set",
    "grip_force",
    "rename_desc",
    "quat_between",
    "quat_mul",
    "quat_slerp",
    "quat_to_mat",
    "mat_to_quat",
    "rot_between",
]

_EZ = np.array([0.0, 0.0, 1.0])

# ---------------------------------------------------------------------------
# small rotation helpers (quaternions are wxyz, right-handed, unit)
# ---------------------------------------------------------------------------


def _unit(v: Sequence[float]) -> np.ndarray:
    """Return v / |v| as a float64 array; raises on a zero vector."""
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v)
    if n < 1e-12:
        raise ValueError("zero-length vector cannot be normalised")
    return v / n


def _perp(v: np.ndarray) -> np.ndarray:
    """A deterministic unit vector perpendicular to unit vector v."""
    k = int(np.argmin(np.abs(v)))  # axis least aligned with v
    e = np.zeros(3)
    e[k] = 1.0
    return _unit(np.cross(v, e))


def quat_mul(q1: Sequence[float], q2: Sequence[float]) -> np.ndarray:
    """Hamilton product q1 * q2 (both wxyz)."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )


def quat_to_mat(q: Sequence[float]) -> np.ndarray:
    """3x3 rotation matrix of unit quaternion q (wxyz)."""
    w, x, y, z = np.asarray(q, dtype=np.float64) / np.linalg.norm(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def mat_to_quat(R: np.ndarray) -> np.ndarray:
    """Unit quaternion (wxyz) of a 3x3 rotation matrix (uses mujoco.mju_mat2Quat)."""
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.ascontiguousarray(R, dtype=np.float64).reshape(9))
    return q


def rot_between(a: Sequence[float], b: Sequence[float]) -> np.ndarray:
    """Minimal rotation matrix R with R @ unit(a) == unit(b).

    For antiparallel vectors a 180-degree rotation about a deterministic
    perpendicular axis is returned.
    """
    a = _unit(a)
    b = _unit(b)
    axis = np.cross(a, b)
    s = np.linalg.norm(axis)
    c = float(np.dot(a, b))
    if s < 1e-12:
        if c > 0.0:
            return np.eye(3)
        axis = _perp(a)
        angle = math.pi
    else:
        axis = axis / s
        angle = math.atan2(s, c)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + math.sin(angle) * K + (1.0 - math.cos(angle)) * (K @ K)


def quat_between(a: Sequence[float], b: Sequence[float]) -> np.ndarray:
    """Minimal rotation quaternion (wxyz) taking direction a to direction b."""
    return mat_to_quat(rot_between(a, b))


def quat_slerp(q0: Sequence[float], q1: Sequence[float], u: float) -> np.ndarray:
    """Spherical interpolation between unit quaternions (wxyz), u in [0, 1]."""
    q0 = np.asarray(q0, dtype=np.float64)
    q1 = np.asarray(q1, dtype=np.float64)
    d = float(np.dot(q0, q1))
    if d < 0.0:
        q1 = -q1
        d = -d
    if d > 0.9995:
        q = q0 + u * (q1 - q0)
        return q / np.linalg.norm(q)
    th = math.acos(d)
    return (math.sin((1 - u) * th) * q0 + math.sin(u * th) * q1) / math.sin(th)


# ---------------------------------------------------------------------------
# recommended simulation options
# ---------------------------------------------------------------------------


def configure_options(spec: mujoco.MjSpec, timestep: float = 0.001,
                      cone: str = "elliptic", solver: str = "newton", iterations: int = 20) -> None:
    """Apply the solver settings the lace was tuned with (see REPORT.md).

    implicitfast integrator, dense Jacobian, Newton solver with 20 iterations
    and elliptic friction cones: stable at 1 ms and 2 ms steps.
    ``cone="pyramidal"`` is ~2.7x faster while the lace slides but makes a
    resting lace jitter at a 2 ms step, so use it only with timestep <= 1 ms;
    ``solver="cg"`` is cheaper per step but rarely converges within
    ``iterations``.
    """
    o = spec.option
    o.timestep = float(timestep)
    o.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    o.jacobian = mujoco.mjtJacobian.mjJAC_DENSE
    o.cone = {"pyramidal": mujoco.mjtCone.mjCONE_PYRAMIDAL, "elliptic": mujoco.mjtCone.mjCONE_ELLIPTIC}[cone]
    o.solver = {"newton": mujoco.mjtSolver.mjSOL_NEWTON, "cg": mujoco.mjtSolver.mjSOL_CG,
                "pgs": mujoco.mjtSolver.mjSOL_PGS}[solver]
    o.iterations = int(iterations)
    o.ls_iterations = 50
    o.tolerance = 1e-8


# ---------------------------------------------------------------------------
# lace builder
# ---------------------------------------------------------------------------


def lace_segment_lengths(n_segments: int, seg_len: float, aglet_len: float) -> np.ndarray:
    """Per-segment lengths: aglet_len for the first and last segment, seg_len otherwise."""
    if n_segments < 3:
        raise ValueError("a lace needs at least 3 segments (two aglets + cord)")
    L = np.full(n_segments, float(seg_len))
    L[0] = aglet_len
    L[-1] = aglet_len
    return L


def add_lace(
    spec: mujoco.MjSpec,
    parent,
    name: str,
    n_segments: int = 44,
    seg_len: float = 0.015,
    radius: float = 0.0015,
    aglet_len: float = 0.020,
    aglet_radius: float = 0.0018,
    rgba: Sequence[float] = (0.95, 0.95, 0.95, 1.0),
    aglet_rgba: Sequence[float] = (0.85, 0.85, 0.9, 1.0),
    start: Sequence[float] = (0.0, 0.0, 0.05),
    direction: Sequence[float] = (1.0, 0.0, 0.0),
    bend_stiffness: float = 2e-4,
    bend_damping: float = 4e-6,
    aglet_joint_scale: float = 3.0,
    armature: float = 5e-9,
    mass_total: float = 0.006,
    friction: Sequence[float] = (0.4, 0.005, 0.0001),
    condim: int = 3,
    solref: Sequence[float] = (0.004, 1.0),
    solimp: Sequence[float] = (0.95, 0.99, 0.0005, 0.5, 2.0),
    margin: float = 0.0003,
    contype: int = 1,
    conaffinity: int = 0,
    priority: int = 1,
    group: int = 0,
    joint_type: str = "ball",
) -> Dict:
    """Add a shoelace chain under ``parent`` (usually ``spec.worldbody``).

    The chain is created straight, starting with aglet A's tip at ``start`` and
    running along ``direction``; use :func:`set_lace_polyline` to pose it
    afterwards.

    Parameters
    ----------
    n_segments, seg_len, radius
        Number of capsule segments, cord segment length and cord radius.
    aglet_len, aglet_radius, aglet_rgba
        Geometry and colour of the first and last segment (the metal aglets).
    rgba
        Colour of the cord.
    start, direction
        Initial (qpos0) pose: tip of aglet A and the unit direction of the chain.
    bend_stiffness, bend_damping
        Ball-joint spring (N m/rad) and damper (N m s/rad) of the cord joints;
        the two joints adjacent to the aglets are scaled by ``aglet_joint_scale``.
    armature
        Rotational armature added to every ball-joint DOF (kg m^2).  The twist
        inertia of a 1.5 mm capsule is ~1.5e-10, far too small for an explicit
        spring at 1-2 ms steps; the armature keeps the twist mode stable.
    mass_total
        Total lace mass; distributed proportionally to segment length.
    joint_type
        "ball": one ball joint per cord joint (3 DOF, twist included). "hinge2": two hinges per cord joint, about the
        child segment's x and then its y axis (2 DOF: bending only; a round cord needs no twist), which cuts the
        lace's DOF count by a third and lets the cord be cut finer for the same cost.
    friction, condim, solref, solimp, margin, contype, conaffinity, priority
        Contact parameters of every lace capsule.  With the defaults
        (contype=1, conaffinity=0) the lace collides with ordinary geoms
        (contype/conaffinity 1) but never with itself or another lace.
        ``priority=1`` makes the lace's solref/solimp win over the defaults of
        whatever it touches.

    Returns
    -------
    dict with keys: name, n, seg_len, radius, aglet_len, aglet_radius, lengths
    (np.ndarray, n), total_length, mass_total, root_body, bodies (list, n),
    geoms (list, n), free_joint, ball_joints (list, n-1; empty for "hinge2"), hinge_joints (list of (x, y) name
    pairs, n-1; empty for "ball"), joint_type, joints (free joint first), sites {'aglet_a': ..., 'aglet_b': ...},
    start, direction.
    """
    name = str(name)
    lengths = lace_segment_lengths(n_segments, seg_len, aglet_len)
    total_length = float(lengths.sum())
    masses = mass_total * lengths / total_length
    start = np.asarray(start, dtype=np.float64)
    direction = _unit(direction)

    bodies: List[str] = []
    geoms: List[str] = []
    joints: List[str] = []
    ball_joints: List[str] = []
    hinge_joints: List[Tuple[str, str]] = []
    if joint_type not in ("ball", "hinge2"):
        raise ValueError("joint_type must be 'ball' or 'hinge2'")

    root_quat = quat_between(_EZ, direction)
    parent_body = parent
    for i in range(n_segments):
        bname = f"{name}/seg{i:02d}"
        if i == 0:
            body = parent_body.add_body(name=bname, pos=start, quat=root_quat)
            jname = f"{name}/free"
            body.add_freejoint(name=jname)
        else:
            body = parent_body.add_body(name=bname, pos=[0.0, 0.0, float(lengths[i - 1])])
            jname = f"{name}/j{i:02d}"
            scale = aglet_joint_scale if (i == 1 or i == n_segments - 1) else 1.0
            if joint_type == "ball":
                body.add_joint(
                    name=jname,
                    type=mujoco.mjtJoint.mjJNT_BALL,
                    stiffness=bend_stiffness * scale,
                    damping=bend_damping * scale,
                    armature=armature,
                )
                ball_joints.append(jname)
            else:
                pair = (jname + "x", jname + "y")
                for jn, axis in zip(pair, ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0))):
                    body.add_joint(
                        name=jn,
                        type=mujoco.mjtJoint.mjJNT_HINGE,
                        axis=list(axis),
                        stiffness=bend_stiffness * scale,
                        damping=bend_damping * scale,
                        armature=armature,
                    )
                hinge_joints.append(pair)
                joints.extend(pair)
        if joint_type == "ball" or i == 0:
            joints.append(jname)
        is_aglet = i == 0 or i == n_segments - 1
        gname = f"{name}/g{i:02d}"
        body.add_geom(
            name=gname,
            type=mujoco.mjtGeom.mjGEOM_CAPSULE,
            fromto=[0.0, 0.0, 0.0, 0.0, 0.0, float(lengths[i])],
            size=[aglet_radius if is_aglet else radius, 0.0, 0.0],
            rgba=list(aglet_rgba if is_aglet else rgba),
            mass=float(masses[i]),
            condim=condim,
            friction=list(friction),
            solref=list(solref),
            solimp=list(solimp),
            margin=margin,
            contype=contype,
            conaffinity=conaffinity,
            priority=priority,
            group=group,
        )
        if i == 0:
            # tip of aglet A: body origin, z axis pointing out (= local -z)
            body.add_site(name=f"{name}/aglet_a", pos=[0, 0, 0], quat=[0, 1, 0, 0],
                          size=[0.001, 0.001, 0.001], group=3)
        if i == n_segments - 1:
            body.add_site(name=f"{name}/aglet_b", pos=[0, 0, float(lengths[i])],
                          quat=[1, 0, 0, 0], size=[0.001, 0.001, 0.001], group=3)
        bodies.append(bname)
        geoms.append(gname)
        parent_body = body

    return {
        "name": name,
        "n": int(n_segments),
        "seg_len": float(seg_len),
        "radius": float(radius),
        "aglet_len": float(aglet_len),
        "aglet_radius": float(aglet_radius),
        "lengths": lengths,
        "total_length": total_length,
        "mass_total": float(mass_total),
        "root_body": bodies[0],
        "bodies": bodies,
        "geoms": geoms,
        "free_joint": joints[0],
        "ball_joints": ball_joints,
        "hinge_joints": hinge_joints,
        "joint_type": joint_type,
        "joints": joints,
        "sites": {"aglet_a": f"{name}/aglet_a", "aglet_b": f"{name}/aglet_b"},
        "start": start.copy(),
        "direction": direction.copy(),
    }


def rename_desc(desc: Dict, prefix: str) -> Dict:
    """Return a copy of a descriptor with ``prefix`` prepended to every MuJoCo name.

    Useful after ``spec.attach(child_spec, prefix=...)`` which prefixes all
    names of the attached sub-model.
    """
    def fix(v):
        if isinstance(v, str):
            return prefix + v
        if isinstance(v, list):
            return [fix(x) for x in v]
        if isinstance(v, dict):
            return {k: fix(x) for k, x in v.items()}
        return v

    out = {}
    for k, v in desc.items():
        if k in ("name", "root_body", "bodies", "geoms", "free_joint", "ball_joints", "hinge_joints",
                 "joints", "sites", "body", "mocap_body", "site", "weld",
                 "target_site", "boxes"):
            out[k] = fix(v)
        else:
            out[k] = v
    return out


# ---------------------------------------------------------------------------
# polylines
# ---------------------------------------------------------------------------


def polyline_from_directions(desc: Dict, start: Sequence[float], directions) -> np.ndarray:
    """Build an (n+1, 3) polyline from aglet A's tip ``start`` and one unit
    direction per segment (an (n, 3) array or a list)."""
    dirs = np.asarray(directions, dtype=np.float64)
    if dirs.shape != (desc["n"], 3):
        raise ValueError(f"directions must have shape ({desc['n']}, 3), got {dirs.shape}")
    dirs = dirs / np.linalg.norm(dirs, axis=1, keepdims=True)
    pts = np.zeros((desc["n"] + 1, 3))
    pts[0] = np.asarray(start, dtype=np.float64)
    pts[1:] = pts[0] + np.cumsum(dirs * desc["lengths"][:, None], axis=0)
    return pts


def straight_polyline(desc: Dict, start: Sequence[float], direction: Sequence[float]) -> np.ndarray:
    """(n+1, 3) polyline of a straight lace from ``start`` along ``direction``."""
    d = _unit(direction)
    return polyline_from_directions(desc, start, np.tile(d, (desc["n"], 1)))


def lace_body_ids(model: mujoco.MjModel, desc: Dict) -> np.ndarray:
    """Body ids of the chain segments in order (cache this for speed)."""
    return np.array([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, b) for b in desc["bodies"]])


def lace_polyline(model: mujoco.MjModel, data: mujoco.MjData, desc: Dict, body_ids=None) -> np.ndarray:
    """World joint points of the chain, (n+1, 3), from aglet A's tip to aglet B's tip.

    Reads ``data.xpos``/``data.xmat`` (valid after mj_forward / mj_step).
    """
    if body_ids is None:
        body_ids = lace_body_ids(model, desc)
    pts = np.empty((desc["n"] + 1, 3))
    pts[:-1] = data.xpos[body_ids]
    last = body_ids[-1]
    pts[-1] = data.xpos[last] + data.xmat[last].reshape(3, 3)[:, 2] * desc["lengths"][-1]
    return pts


def set_lace_polyline(model: mujoco.MjModel, data: mujoco.MjData, desc: Dict, points,
                      check: bool = True, atol: float = 1e-5, forward: bool = True) -> None:
    """Pose the chain along ``points`` ((n+1, 3) world polyline) and zero its velocities.

    Segment i runs from points[i] to points[i+1]; consecutive distances must
    equal the segment lengths (``desc['lengths']``) to within ``atol`` when
    ``check`` is true.  The free-joint pose and every ball-joint quaternion are
    written with the minimal-twist (parallel transport) frame, so
    :func:`lace_polyline` reproduces the points to machine precision.  Calls
    ``mj_forward`` when ``forward`` is true so that xpos/site_xpos are current.
    """
    pts = np.asarray(points, dtype=np.float64)
    n = desc["n"]
    if pts.shape != (n + 1, 3):
        raise ValueError(f"points must have shape ({n + 1}, 3), got {pts.shape}")
    seg = np.diff(pts, axis=0)
    dist = np.linalg.norm(seg, axis=1)
    if check:
        bad = np.where(np.abs(dist - desc["lengths"]) > atol)[0]
        if bad.size:
            raise ValueError(
                f"segment lengths differ from the lace geometry at segments {bad.tolist()}: "
                f"{dist[bad].tolist()} vs {desc['lengths'][bad].tolist()}")
    dirs = seg / dist[:, None]

    R_prev = rot_between(_EZ, dirs[0])
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, desc["free_joint"])
    adr = model.jnt_qposadr[jid]
    data.qpos[adr:adr + 3] = pts[0]
    data.qpos[adr + 3:adr + 7] = mat_to_quat(R_prev)
    dadr = model.jnt_dofadr[jid]
    data.qvel[dadr:dadr + 6] = 0.0
    data.qacc_warmstart[dadr:dadr + 6] = 0.0

    if desc.get("joint_type", "ball") == "hinge2":
        # child frame = parent frame . Rx(a) . Ry(b); its z axis must be the segment direction v (parent frame):
        # v = (sin b, -sin a cos b, cos a cos b)
        for i, (jx, jy) in enumerate(desc["hinge_joints"], start=1):
            v = R_prev.T @ dirs[i]
            b = math.asin(float(np.clip(v[0], -1.0, 1.0)))
            a = math.atan2(-float(v[1]), float(v[2]))
            ca, sa, cb, sb = math.cos(a), math.sin(a), math.cos(b), math.sin(b)
            Rx = np.array([[1.0, 0.0, 0.0], [0.0, ca, -sa], [0.0, sa, ca]])
            Ry = np.array([[cb, 0.0, sb], [0.0, 1.0, 0.0], [-sb, 0.0, cb]])
            for jn, ang in ((jx, a), (jy, b)):
                jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn)
                data.qpos[model.jnt_qposadr[jid]] = ang
                dadr = model.jnt_dofadr[jid]
                data.qvel[dadr] = 0.0
                data.qacc_warmstart[dadr] = 0.0
            R_prev = R_prev @ Rx @ Ry
        if forward:
            mujoco.mj_forward(model, data)
        return
    for i, jname in enumerate(desc["ball_joints"], start=1):
        R_i = rot_between(dirs[i - 1], dirs[i]) @ R_prev
        Q_i = R_prev.T @ R_i  # relative rotation expressed in the parent frame
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
        adr = model.jnt_qposadr[jid]
        data.qpos[adr:adr + 4] = mat_to_quat(Q_i)
        dadr = model.jnt_dofadr[jid]
        data.qvel[dadr:dadr + 3] = 0.0
        data.qacc_warmstart[dadr:dadr + 3] = 0.0
        R_prev = R_i
    if forward:
        mujoco.mj_forward(model, data)


def lace_length(poly) -> float:
    """Arc length of a polyline."""
    poly = np.asarray(poly, dtype=np.float64)
    return float(np.linalg.norm(np.diff(poly, axis=0), axis=1).sum())


def segment_slack(poly, i0: int, i1: int) -> float:
    """Slack of the lace between joint points i0 and i1 (i0 < i1):

    arc length along the chain from point i0 to point i1 minus the straight
    distance between the two points.  Zero means that part is taut.
    """
    poly = np.asarray(poly, dtype=np.float64)
    if not 0 <= i0 < i1 < len(poly):
        raise ValueError("need 0 <= i0 < i1 < len(poly)")
    arc = float(np.linalg.norm(np.diff(poly[i0:i1 + 1], axis=0), axis=1).sum())
    return arc - float(np.linalg.norm(poly[i1] - poly[i0]))


# ---------------------------------------------------------------------------
# eyelet rings
# ---------------------------------------------------------------------------


def add_eyelet_ring(
    spec: mujoco.MjSpec,
    body,
    name: str,
    center: Sequence[float],
    axis: Sequence[float],
    hole_radius: float = 0.0045,
    wire_radius: float = 0.0012,
    n_capsules: int = 8,
    rgba: Sequence[float] = (0.85, 0.85, 0.9, 1.0),
    mass: float = 0.0003,
    friction: Sequence[float] = (0.3, 0.005, 0.0001),
    condim: int = 3,
    solref: Sequence[float] = (0.004, 1.0),
    solimp: Sequence[float] = (0.95, 0.99, 0.0005, 0.5, 2.0),
    margin: float = 0.0,
    contype: int = 1,
    conaffinity: int = 1,
    priority: int = 2,
    group: int = 0,
) -> Dict:
    """Add a torus-like eyelet ring made of ``n_capsules`` capsules to ``body``.

    ``center`` and ``axis`` are in the body's local frame.  The capsule axes
    form the polygon *circumscribed* about the circle of radius
    ``hole_radius + wire_radius`` (each capsule is tangent to that circle at
    its midpoint), consecutive capsules share an end point, so the ring is a
    closed tube without gaps: nothing can slip between neighbours.  The clear
    hole radius is exactly ``hole_radius`` at the capsule midpoints and
    ``(hole_radius + wire_radius) / cos(pi / n) - wire_radius`` at the corners.

    ``priority=2`` (> the lace's 1) makes the ring's friction/solref/solimp
    win at lace-ring contacts: a smooth metal eyelet has lower friction than
    the fabric the lace otherwise touches, hence the default 0.3 vs 0.4.

    Returns dict: name, body, center, axis (unit, local), hole_radius,
    wire_radius, ring_radius (centre-line circle), outer_radius (max radial
    extent incl. the wire), n_capsules, geoms.
    """
    name = str(name)
    c = np.asarray(center, dtype=np.float64)
    a = _unit(axis)
    u = _perp(a)
    v = np.cross(a, u)
    R = hole_radius + wire_radius
    half = R * math.tan(math.pi / n_capsules)
    geoms = []
    for k in range(n_capsules):
        phi = 2.0 * math.pi * k / n_capsules
        radial = math.cos(phi) * u + math.sin(phi) * v
        tangent = -math.sin(phi) * u + math.cos(phi) * v
        mid = c + R * radial
        p0 = mid - half * tangent
        p1 = mid + half * tangent
        gname = f"{name}/cap{k:02d}"
        body.add_geom(
            name=gname,
            type=mujoco.mjtGeom.mjGEOM_CAPSULE,
            fromto=[*p0, *p1],
            size=[wire_radius, 0.0, 0.0],
            rgba=list(rgba),
            mass=mass / n_capsules,
            condim=condim,
            friction=list(friction),
            solref=list(solref),
            solimp=list(solimp),
            margin=margin,
            contype=contype,
            conaffinity=conaffinity,
            priority=priority,
            group=group,
        )
        geoms.append(gname)
    body_name = body.name if body.name else "world"
    return {
        "name": name,
        "body": body_name,
        "center": c.copy(),
        "axis": a.copy(),
        "hole_radius": float(hole_radius),
        "wire_radius": float(wire_radius),
        "ring_radius": float(R),
        "outer_radius": float(R / math.cos(math.pi / n_capsules) + wire_radius),
        "n_capsules": int(n_capsules),
        "geoms": geoms,
    }


def ring_world(model: mujoco.MjModel, data: mujoco.MjData, body_name: str, ring_desc: Dict
               ) -> Tuple[np.ndarray, np.ndarray]:
    """(center_xyz, axis_xyz) of a ring in world coordinates (after mj_forward)."""
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    Rm = data.xmat[bid].reshape(3, 3)
    center = data.xpos[bid] + Rm @ ring_desc["center"]
    axis = Rm @ ring_desc["axis"]
    return center, axis / np.linalg.norm(axis)


def ring_tuple(model: mujoco.MjModel, data: mujoco.MjData, ring_desc: Dict, radius=None
               ) -> Tuple[str, np.ndarray, np.ndarray, float]:
    """``(name, center, axis, radius)`` for :func:`pass_sequence`; the default
    radius is the clear ``hole_radius`` of the ring."""
    center, axis = ring_world(model, data, ring_desc["body"], ring_desc)
    return (ring_desc["name"], center, axis, float(ring_desc["hole_radius"] if radius is None else radius))


# ---------------------------------------------------------------------------
# topology
# ---------------------------------------------------------------------------


def ring_crossings(poly, center, axis, radius: float) -> List[Tuple[int, int, float]]:
    """Segments of ``poly`` whose centre line crosses the disc (center, axis, radius).

    Returns a list of ``(segment_index, direction, t)``.  ``direction`` is the
    sign of ``(poly[i] - poly[i+1]) . axis``: +1 when the lace, followed from
    aglet B *towards aglet A*, goes along ``axis`` and -1 when it goes against
    it.  This is the direction in which aglet A travelled when it was the end
    that threaded the ring (an IN pass driven down through a ring with axis +z
    gives -1, an OUT pass driven up gives +1).  ``t`` is the arc-length
    position of the crossing along the lace, normalised to [0, 1] from the
    first point (aglet A's tip).  A vertex lying exactly in the ring plane is
    counted on the positive side so it is never double counted.
    """
    poly = np.asarray(poly, dtype=np.float64)
    c = np.asarray(center, dtype=np.float64)
    a = _unit(axis)
    rel = poly - c
    s = rel @ a
    seg = np.diff(poly, axis=0)
    L = np.linalg.norm(seg, axis=1)
    cum = np.concatenate([[0.0], np.cumsum(L)])
    total = cum[-1] if cum[-1] > 0 else 1.0
    s0, s1 = s[:-1], s[1:]
    cross = (s0 < 0) != (s1 < 0)
    out = []
    for i in np.where(cross)[0]:
        u = s0[i] / (s0[i] - s1[i])
        q = rel[i] + u * seg[i]
        r_in_plane = np.linalg.norm(q - np.dot(q, a) * a)
        if r_in_plane < radius:
            direction = -1 if s1[i] > s0[i] else 1  # B->A tangent along the axis
            out.append((int(i), direction, float((cum[i] + u * L[i]) / total)))
    return out


def pass_sequence(poly, rings: Iterable[Tuple[str, Sequence[float], Sequence[float], float]]
                  ) -> List[Tuple[str, int, float]]:
    """All ring crossings of a lace polyline, as ``(ring_name, direction, t)`` sorted by t.

    ``rings`` is an iterable of ``(ring_name, center, axis, radius)`` tuples
    (see :func:`ring_tuple`); ``direction`` follows :func:`ring_crossings`
    (+1: the lace runs along the ring axis when followed from aglet B to A).
    """
    out = []
    for rname, center, axis, radius in rings:
        for _, direction, t in ring_crossings(poly, center, axis, radius):
            out.append((rname, direction, t))
    out.sort(key=lambda x: x[2])
    return out


# ---------------------------------------------------------------------------
# mocap grips (for scripted tests and demos)
# ---------------------------------------------------------------------------


def add_site_grip(
    spec: mujoco.MjSpec,
    name: str,
    target_site: str,
    pos: Sequence[float] = (0.0, 0.0, 0.0),
    quat: Sequence[float] = (1.0, 0.0, 0.0, 0.0),
    radius: float = 0.004,
    rgba: Sequence[float] = (0.2, 0.9, 0.2, 0.4),
    solref: Sequence[float] = (0.004, 1.0),
    solimp: Sequence[float] = (0.95, 0.99, 0.0005, 0.5, 2.0),
    torquescale: float = 1.0,
    active: bool = False,
) -> Dict:
    """A mocap body with a site, welded (site-to-site, initially ``active``) to ``target_site``.

    Set the mocap pose with :func:`grip_set`; the weld makes the target site's
    frame coincide with the mocap site's frame.  The sphere geom is visual only.

    Returns dict: name, mocap_body, site, weld, target_site.
    """
    name = str(name)
    body = spec.worldbody.add_body(name=f"{name}/mocap", pos=list(pos), quat=list(quat), mocap=True)
    body.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[radius, 0, 0], rgba=list(rgba),
                  contype=0, conaffinity=0, group=1)
    body.add_site(name=f"{name}/site", pos=[0, 0, 0], quat=[1, 0, 0, 0], size=[0.001] * 3, group=3)
    data = [0.0] * 10 + [float(torquescale)]
    spec.add_equality(
        type=mujoco.mjtEq.mjEQ_WELD,
        name=f"{name}/weld",
        name1=f"{name}/site",
        name2=target_site,
        objtype=mujoco.mjtObj.mjOBJ_SITE,
        active=bool(active),
        solref=list(solref),
        solimp=list(solimp),
        data=data,
    )
    return {"name": name, "mocap_body": f"{name}/mocap", "site": f"{name}/site",
            "weld": f"{name}/weld", "target_site": target_site}


def grip_set(model: mujoco.MjModel, data: mujoco.MjData, grip: Dict,
             active=None, pos=None, quat=None) -> None:
    """Set the mocap pose (world) and/or (de)activate the weld of a grip."""
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, grip["mocap_body"])
    mid = model.body_mocapid[bid]
    if pos is not None:
        data.mocap_pos[mid] = np.asarray(pos, dtype=np.float64)
    if quat is not None:
        q = np.asarray(quat, dtype=np.float64)
        data.mocap_quat[mid] = q / np.linalg.norm(q)
    if active is not None:
        eid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, grip["weld"])
        data.eq_active[eid] = 1 if active else 0


def grip_force(model: mujoco.MjModel, data: mujoco.MjData, grip: Dict) -> Tuple[np.ndarray, np.ndarray]:
    """(force[3], torque[3]) currently applied by the grip's weld (constraint-space efc rows).

    Zeros when the weld is inactive or not in the constraint set."""
    eid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, grip["weld"])
    nefc = data.nefc
    rows = np.where((data.efc_type[:nefc] == mujoco.mjtConstraint.mjCNSTR_EQUALITY)
                    & (data.efc_id[:nefc] == eid))[0]
    if rows.size != 6:
        return np.zeros(3), np.zeros(3)
    f = data.efc_force[rows]
    return f[:3].copy(), f[3:].copy()


# ---------------------------------------------------------------------------
# simple fixtures
# ---------------------------------------------------------------------------


def add_plate_with_hole(
    spec: mujoco.MjSpec,
    body,
    name: str,
    center: Sequence[float] = (0.0, 0.0, 0.1),
    size_xy: Sequence[float] = (0.8, 0.4),
    thickness: float = 0.003,
    hole_xy: Sequence[float] = (0.0, 0.0),
    hole_halfwidth: float = 0.0057,
    rgba: Sequence[float] = (0.55, 0.45, 0.35, 1.0),
    friction: Sequence[float] = (0.4, 0.005, 0.0001),
    condim: int = 3,
    contype: int = 1,
    conaffinity: int = 1,
) -> Dict:
    """A horizontal plate (4 boxes) with a square hole of half-width ``hole_halfwidth``
    centred at world ``(hole_xy[0], hole_xy[1], center[2])``; sizes are full extents.

    Returns dict: name, body, center, size_xy, thickness, hole_xy, hole_halfwidth, boxes.
    """
    name = str(name)
    cx, cy, cz = (float(x) for x in center)
    hx, hy = size_xy[0] / 2.0, size_xy[1] / 2.0
    ox, oy = (float(x) for x in hole_xy)
    w = float(hole_halfwidth)
    t2 = thickness / 2.0
    x_lo, x_hi, y_lo, y_hi = cx - hx, cx + hx, cy - hy, cy + hy
    if not (x_lo < ox - w and ox + w < x_hi and y_lo < oy - w and oy + w < y_hi):
        raise ValueError("hole must lie strictly inside the plate")
    pieces = {
        "yp": ((x_lo, x_hi), (oy + w, y_hi)),
        "ym": ((x_lo, x_hi), (y_lo, oy - w)),
        "xp": ((ox + w, x_hi), (oy - w, oy + w)),
        "xm": ((x_lo, ox - w), (oy - w, oy + w)),
    }
    boxes = []
    for key, ((xa, xb), (ya, yb)) in pieces.items():
        gname = f"{name}/{key}"
        body.add_geom(
            name=gname,
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=[(xa + xb) / 2.0, (ya + yb) / 2.0, cz],
            size=[(xb - xa) / 2.0, (yb - ya) / 2.0, t2],
            rgba=list(rgba),
            friction=list(friction),
            condim=condim,
            contype=contype,
            conaffinity=conaffinity,
        )
        boxes.append(gname)
    return {"name": name, "body": body.name if body.name else "world", "center": np.array([cx, cy, cz]),
            "size_xy": (float(size_xy[0]), float(size_xy[1])), "thickness": float(thickness),
            "hole_xy": (ox, oy), "hole_halfwidth": w, "boxes": boxes}
