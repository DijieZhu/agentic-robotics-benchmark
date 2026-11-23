"""Parcel geometry from privileged truth: world-aligned dimensions, yaw
alignment and the label's world direction."""

from __future__ import annotations

import math

import numpy as np
import mujoco

# world directions of the label normal
UP, DOWN, XP, XN, YP, YN = "up", "down", "+x", "-x", "+y", "-y"
DIRS = {UP: (0, 0, 1), DOWN: (0, 0, -1), XP: (1, 0, 0), XN: (-1, 0, 0),
        YP: (0, 1, 0), YN: (0, -1, 0)}


def quat_mat(q):
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, np.asarray(q, float))
    return R.reshape(3, 3)


def world_dims(size, quat):
    """Extent of the parcel along world x, y, z (axis-aligned bounding box
    of the oriented box) and the yaw misalignment (rad, in [-pi/4, pi/4]) of
    the parcel's horizontal axes to the world axes."""
    R = quat_mat(quat)
    h = np.asarray(size, float) / 2
    ext = np.abs(R) @ h * 2
    # yaw misalignment: angle of the parcel axis that is most horizontal-y
    best = None
    for i in range(3):
        a = R[:, i]
        if abs(a[2]) > 0.7:
            continue
        ang = math.atan2(a[1], a[0])          # direction in the plane
        # misalignment to the nearest multiple of 90 deg
        mis = (ang + math.pi / 4) % (math.pi / 2) - math.pi / 4
        if best is None or abs(mis) < abs(best):
            best = mis
    return ext, (0.0 if best is None else best)


def aligned_dims(size, quat):
    """Dimensions along world x, y, z assuming the parcel is (nearly) axis
    aligned: each body axis assigned to the world axis it is closest to."""
    R = quat_mat(quat)
    d = np.zeros(3)
    used = set()
    for i in np.argsort(-np.max(np.abs(R), axis=0)):
        w = int(np.argmax(np.abs(R[:, i]) * [1 if k not in used else 0
                                             for k in range(3)]))
        used.add(w)
        d[w] = size[i]
    return d


def label_dir(label_normal) -> tuple[str, float]:
    n = np.asarray(label_normal, float)
    best, bd = None, -2.0
    for k, v in DIRS.items():
        c = float(n @ np.asarray(v, float))
        if c > bd:
            best, bd = k, c
    return best, math.degrees(math.acos(max(-1.0, min(1.0, bd))))
