"""Grasp / hook attachment through retargetable weld equalities, handle search and the aglet catch test.

The cell carries two inactive welds (see cell.build_welds): ``'left/grasp'`` (body1 = the hand body
``left/hand``) and ``'right/hook'`` (body1 = the hook tool body ``right/hook``), both with body2 = world
as a placeholder.  :func:`attach` retargets ``model.eq_obj2id`` at runtime to the object body, writes the
relative pose into ``model.eq_data`` and activates the constraint; :func:`release` deactivates it.

eq_data layout of a MuJoCo weld (3.x): [pivot(3) in body2's frame, relpose(7) = pose of body2 in body1's
frame (pos, wxyz), torquescale(1)].  Verified at runtime: with pivot 0 and relpose (p, q) the solver pulls
body2 to  x2 = x1 + R1 p,  R2 = R1 R(q).

Contact between the holder and the attached object: a welded object that overlaps the tool's own
geoms would fight the weld with contact forces.  The hook tool's collision geoms are built with
contype = 1, conaffinity = 0; :func:`attach` switches the attached body's geoms to the same (1, 0) pair,
which keeps every contact with ordinary (1, 1) geoms (bench, fingers, loose objects: (1 & 1) != 0) but
removes contacts between two (1, 0) geoms, i.e. between the hook and what it carries.  The original
bits are restored by :func:`release`.

Canonical aglet poses (an aglet is a capsule whose axis is its body +z):
* left hand: axis along ``left/tcp`` +z, centre at tcp + 0.008 z  (the 0.02 m segment then spans
  z in [-0.002, +0.018] below the finger-pad midpoint: 0.018 of it protrudes along the approach direction);
* hook: axis along ``right/hook_tip`` +z, centre at hook_tip + 0.008 z (segment spans [-0.002, +0.018]
  along the tip direction measured from the tip's segment end; the tip capsule's cap ends at +0.002, so
  0.016 of segment / 0.0178 of capsule protrude beyond the physical tip; the aglet emerges first when
  the hook rises through an eyelet).

Runtime dependencies: mujoco + numpy only.
"""

import math

import numpy as np
import mujoco

from .control import quat_mul, quat_inv, quat_to_mat, mat_to_quat

__all__ = ["attach", "release", "is_attached", "body_rel_pose", "canonical_aglet_pose", "find_handle", "catch_test",
           "AGLET_OFFSET", "PINCH_BOX", "CATCH_RADIUS", "WELD_SITE", "advance_seats"]

AGLET_OFFSET = 0.008
PINCH_BOX = {"x": 0.012, "z_lo": -0.010, "z_hi": 0.012}
CATCH_RADIUS = 0.008
# the site that defines the canonical object frame of each weld
WELD_SITE = {"left/grasp": "left/tcp", "right/hook": "right/hook_tip"}
ATTACHED_BITS = (1, 0)

_saved = {}   # (model address, eq id) -> (geom ids, contype, conaffinity, body id)
_seats = {}   # (model address, eq id) -> seating glide: (start pos, start quat, goal pos, goal quat, t0, duration)


def _key(model, eq):
    return (model._address if hasattr(model, "_address") else id(model), int(eq))


def body_rel_pose(model, data, body1, body2):
    """(pos, wxyz quat) of body2 expressed in body1's frame from the current data (needs kinematics up to date)."""
    b1 = model.body(body1).id if isinstance(body1, str) else int(body1)
    b2 = model.body(body2).id if isinstance(body2, str) else int(body2)
    R1 = data.xmat[b1].reshape(3, 3)
    p = R1.T @ (data.xpos[b2] - data.xpos[b1])
    q = quat_mul(quat_inv(data.xquat[b1]), data.xquat[b2])
    return p, q


def attach(model, data, eq_name, target_body_name, rel_pos=None, rel_quat=None, bits=None, seat_s=0.0):
    """Activate weld eq_name on target_body_name.

    rel_pos / rel_quat: pose of the target in body1's (the tool body's) frame.  None = the CURRENT relative
    pose (the object stays where it is); given = the canonical pose (the constraint pulls the object into
    alignment within a few ms).  seat_s > 0 with a given pose: the weld starts at the current relative pose and
    its target glides to the given one over seat_s seconds of simulated time (:func:`advance_seats`, once per
    control tick), so a light object (an aglet, with its lace) is drawn into the seat instead of jumping there.
    bits: (contype, conaffinity) written to the held body's collision geoms while it is held (restored by
    :func:`release`); None leaves them.  Returns the equality id.
    """
    eq = model.equality(eq_name).id
    b1 = int(model.eq_obj1id[eq])
    b2 = model.body(target_body_name).id
    if model.eq_type[eq] != mujoco.mjtEq.mjEQ_WELD:
        raise ValueError("%s is not a weld" % eq_name)
    if data.eq_active[eq]:
        release(model, data, eq_name)
    if rel_pos is None or rel_quat is None:
        p, q = body_rel_pose(model, data, b1, b2)
        if rel_pos is not None:
            p = np.asarray(rel_pos, float)
        if rel_quat is not None:
            q = np.asarray(rel_quat, float)
    else:
        p, q = np.asarray(rel_pos, float), np.asarray(rel_quat, float)
    q = q / np.linalg.norm(q)
    _seats.pop(_key(model, eq), None)
    if seat_s > 0.0 and rel_pos is not None and rel_quat is not None:
        p_now, q_now = body_rel_pose(model, data, b1, b2)
        q_now = np.asarray(q_now, float) / np.linalg.norm(q_now)
        _seats[_key(model, eq)] = (np.asarray(p_now, float), q_now, p, q, float(data.time), float(seat_s))
        p, q = np.asarray(p_now, float), q_now
    model.eq_obj2id[eq] = b2
    model.eq_data[eq, 0:3] = 0.0
    model.eq_data[eq, 3:6] = p
    model.eq_data[eq, 6:10] = q
    model.eq_data[eq, 10] = 1.0
    data.eq_active[eq] = 1
    if bits is not None:
        gids = [g for g in range(model.ngeom) if model.geom_bodyid[g] == b2
                and (model.geom_contype[g] != 0 or model.geom_conaffinity[g] != 0)]
        _saved[_key(model, eq)] = (gids, model.geom_contype[gids].copy(), model.geom_conaffinity[gids].copy(), b2)
        for g in gids:
            model.geom_contype[g] = int(bits[0])
            model.geom_conaffinity[g] = int(bits[1])
    return eq


def release(model, data, eq_name):
    """Deactivate weld eq_name and restore the released body's contact bits.  Returns the body id released (or -1)."""
    eq = model.equality(eq_name).id
    was = int(model.eq_obj2id[eq]) if data.eq_active[eq] else -1
    data.eq_active[eq] = 0
    _seats.pop(_key(model, eq), None)
    saved = _saved.pop(_key(model, eq), None)
    if saved is not None:
        gids, ct, ca, _ = saved
        for i, g in enumerate(gids):
            model.geom_contype[g] = ct[i]
            model.geom_conaffinity[g] = ca[i]
    return was


def _slerp(q0, q1, u):
    if float(q0 @ q1) < 0.0:
        q1 = -q1
    d = float(np.clip(q0 @ q1, -1.0, 1.0))
    if d > 0.9995:
        q = q0 + u * (q1 - q0)
    else:
        th = math.acos(d)
        q = (math.sin((1.0 - u) * th) * q0 + math.sin(u * th) * q1) / math.sin(th)
    return q / np.linalg.norm(q)


def advance_seats(model, data):
    """Move every seating weld's target along its glide (linear in position, slerp in orientation) to the state at
    data.time; a finished or released glide is dropped.  Call once per control tick, before stepping."""
    for key, (p0, q0, p1, q1, t0, dur) in list(_seats.items()):
        if key[0] != _key(model, 0)[0]:
            continue
        eq = key[1]
        if not data.eq_active[eq]:
            _seats.pop(key, None)
            continue
        u = min(1.0, max(0.0, (float(data.time) - t0) / dur))
        model.eq_data[eq, 3:6] = (1.0 - u) * p0 + u * p1
        model.eq_data[eq, 6:10] = _slerp(q0, q1, u)
        if u >= 1.0:
            _seats.pop(key, None)


def is_attached(model, data, eq_name):
    """Name of the body currently held by eq_name, or None."""
    eq = model.equality(eq_name).id
    if not data.eq_active[eq]:
        return None
    return model.body(int(model.eq_obj2id[eq])).name


def canonical_aglet_pose(model, eq_name, offset=AGLET_OFFSET):
    """(rel_pos, rel_quat) of an aglet in the weld's body1 frame: axis along the weld site's +z, centre offset along it."""
    eq = model.equality(eq_name).id
    b1 = int(model.eq_obj1id[eq])
    site = model.site(WELD_SITE[eq_name]).id
    if model.site_bodyid[site] != b1:
        raise ValueError("site %s is not on the weld body of %s" % (WELD_SITE[eq_name], eq_name))
    sp = np.array(model.site_pos[site], float)
    sq = np.array(model.site_quat[site], float)
    Rs = quat_to_mat(sq)
    return sp + Rs @ np.array([0.0, 0.0, offset]), sq


def find_handle(tcp_pos, tcp_mat, handles, opening=0.08):
    """Names of the handles inside the pinch box of a parallel gripper, nearest first.

    handles: iterable of (name, world_pos, world_mat, pinch_width).  A handle qualifies when its position
    in the tcp frame satisfies |x| <= 0.012, |y| <= 0.5 * opening, -0.010 <= z <= +0.012 (x across the
    pads, y along the closing axis, z along the approach) and its pinch_width <= opening.
    """
    R = np.asarray(tcp_mat, float).reshape(3, 3)
    p0 = np.asarray(tcp_pos, float)
    out = []
    for name, pos, _mat, width in handles:
        d = R.T @ (np.asarray(pos, float) - p0)
        if abs(d[0]) <= PINCH_BOX["x"] and abs(d[1]) <= 0.5 * opening and PINCH_BOX["z_lo"] <= d[2] <= PINCH_BOX["z_hi"] \
                and float(width) <= opening + 1e-9:
            out.append((float(np.linalg.norm(d)), name))
    out.sort()
    return [n for _, n in out]


def catch_test(tip_pos, tip_mat, aglet_center, aglet_axis, radius=CATCH_RADIUS):
    """True when the aglet centre lies within radius of the hook tip (any orientation)."""
    d = np.linalg.norm(np.asarray(aglet_center, float) - np.asarray(tip_pos, float))
    return bool(d <= radius)
