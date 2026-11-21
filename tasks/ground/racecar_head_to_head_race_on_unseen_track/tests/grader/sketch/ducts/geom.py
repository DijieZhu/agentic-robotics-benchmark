"""Planar geometry of duct pieces: end points, collision extents, poses from joint values, the
put-back transform. Standard library and numpy only.

A piece pose is [x, y, yaw] of its axis midpoint; its tracked points are the two end points of its
axis. A dynamic piece (modes `sleep`, `awake`, and the window's slots) is a body whose frame at
qpos0 is a laid pose P0 = (x0, y0, yaw0) with joints, in this order, slide along its x, slide along its y, hinge
about z through its (moved) origin: with joint values (sx, sy, th) its pose is
    (x0, y0) + R(yaw0) (sx, sy),  yaw0 + th
exactly (`pose_from_q`), and `q_from_pose` is the inverse.
"""

import math

import numpy as np

from sketch.examples import constants as K


def wrap(a):
    """Angles in [-pi, pi) (vectorised)."""
    return (np.asarray(a, dtype=float) + math.pi) % (2.0 * math.pi) - math.pi


def ends(poses, lengths):
    """(tail, head) end points, arrays (n, 2), of pieces `poses` (n, 3) with lengths (n,)."""
    p = np.asarray(poses, dtype=float).reshape(-1, 3)
    h = 0.5 * np.asarray(lengths, dtype=float).reshape(-1)
    u = np.stack([np.cos(p[:, 2]), np.sin(p[:, 2])], axis=1) * h[:, None]
    return p[:, :2] - u, p[:, :2] + u


def displacement(poses, ref_poses, lengths):
    """Per piece: the larger horizontal distance of its two end points from where they are in `ref_poses`."""
    t1, h1 = ends(poses, lengths)
    t0, h0 = ends(ref_poses, lengths)
    return np.maximum(np.linalg.norm(t1 - t0, axis=1), np.linalg.norm(h1 - h0, axis=1))


def collision_extent(k, n, c):
    """(lo, hi) of the collision capsule's segment along the axis of piece k of an n-piece duct with pieces of
    length c: the whole axis, except that at a free end of the duct the segment stops one radius short, so the
    round cap does not reach beyond the end plane (the contract: nothing can be touched where the LiDAR sees no
    duct)."""
    r = K.DUCT_RADIUS_M
    lo = -0.5 * c + (r if k == 0 else 0.0)
    hi = 0.5 * c - (r if k == n - 1 else 0.0)
    if hi < lo:                                  # a one-piece duct shorter than a diameter: a sphere at the centre
        lo = hi = 0.0
    return lo, hi


def yaw_quat(yaw):
    return [math.cos(0.5 * yaw), 0.0, 0.0, math.sin(0.5 * yaw)]


def pose_from_q(p0, q):
    """Poses (n, 3) of dynamic pieces whose frames at qpos0 are `p0` (n, 3) and whose joint values are q (n, 3) =
    (slide x, slide y, hinge)."""
    p0 = np.asarray(p0, dtype=float).reshape(-1, 3)
    q = np.asarray(q, dtype=float).reshape(-1, 3)
    c, s = np.cos(p0[:, 2]), np.sin(p0[:, 2])
    out = np.empty_like(p0)
    out[:, 0] = p0[:, 0] + c * q[:, 0] - s * q[:, 1]
    out[:, 1] = p0[:, 1] + s * q[:, 0] + c * q[:, 1]
    out[:, 2] = p0[:, 2] + q[:, 2]
    return out


def q_from_pose(p0, poses):
    """Joint values (n, 3) that put dynamic pieces with qpos0 frames `p0` at `poses` (the inverse of pose_from_q;
    the hinge value is the yaw difference wrapped to [-pi, pi))."""
    p0 = np.asarray(p0, dtype=float).reshape(-1, 3)
    p = np.asarray(poses, dtype=float).reshape(-1, 3)
    c, s = np.cos(p0[:, 2]), np.sin(p0[:, 2])
    dx, dy = p[:, 0] - p0[:, 0], p[:, 1] - p0[:, 1]
    return np.stack([c * dx + s * dy, -s * dx + c * dy, wrap(p[:, 2] - p0[:, 2])], axis=1)


def rigid(poses, dx, dy, dth, centre):
    """Poses (n, 3) moved by a planar rigid motion: rotation dth about `centre`, then translation (dx, dy)."""
    p = np.asarray(poses, dtype=float).reshape(-1, 3)
    cx, cy = float(centre[0]), float(centre[1])
    c, s = math.cos(dth), math.sin(dth)
    out = np.empty_like(p)
    out[:, 0] = cx + c * (p[:, 0] - cx) - s * (p[:, 1] - cy) + dx
    out[:, 1] = cy + s * (p[:, 0] - cx) + c * (p[:, 1] - cy) + dy
    out[:, 2] = p[:, 2] + dth
    return out


def putback_pose(world_poses, lengths, max_err, rng):
    """The pose of a duct re-laid by the staff: its world-file shape moved rigidly, translation uniform in a disc of
    radius max_err / 2, rotation about the midpoint of its end points uniform within +-(max_err / 2) / (half the
    end-to-end distance). Every end point then lies within max_err of its world-file position. Draws three numbers
    from `rng` (numpy Generator) in a fixed order."""
    p = np.asarray(world_poses, dtype=float).reshape(-1, 3)
    tail, head = ends(p, lengths)
    pts = np.concatenate([tail, head])
    centre = 0.5 * (tail[0] + head[-1])
    reach = float(np.linalg.norm(pts - centre, axis=1).max())
    half = 0.5 * float(max_err)
    r = half * math.sqrt(float(rng.uniform(0.0, 1.0)))
    a = float(rng.uniform(-math.pi, math.pi))
    dth = float(rng.uniform(-1.0, 1.0)) * (half / max(reach, 1e-9))
    return rigid(p, r * math.cos(a), r * math.sin(a), dth, centre)


def point_segment_distance(px, py, ax, ay, bx, by):
    """Distances (vectorised over segments a -> b and/or points p) from points to segments."""
    abx, aby = bx - ax, by - ay
    den = np.maximum(abx * abx + aby * aby, 1e-18)
    t = np.clip(((px - ax) * abx + (py - ay) * aby) / den, 0.0, 1.0)
    return np.hypot(ax + t * abx - px, ay + t * aby - py)
