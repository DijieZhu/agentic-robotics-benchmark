"""Scan geometry of the 2D LiDAR (the numbers of the datasheet and of `describe()["car"]["lidar"]`).

Beam i points along angle a_i = ANGLE_MIN_RAD + i * ANGLE_INC_RAD in the LiDAR frame (x forward, y left, parallel to
the car frame): beam 0 at -135 degrees (right rear), beam 540 straight ahead, beam 1080 at +135 degrees. A range is
an integer in millimetres; 0 means no return. The revolution is fired as five sub-scans (SUBSCAN_SLICES) while the
car moves. Works on lists; returns numpy arrays when numpy can be imported.
"""

import math

N_BEAMS = 1081
ANGLE_MIN_RAD = -0.75 * math.pi
ANGLE_INC_RAD = math.pi / 720.0
RANGE_MIN_M = 0.06
RANGE_MAX_M = 10.0
NO_RETURN = 0
SUBSCAN_SLICES = ((0, 216), (216, 432), (432, 648), (648, 864), (864, 1081))
LIDAR_POS_M = (0.21315, 0.0, 0.1099)        # scan origin in the car frame (describe()["car"]["lidar"]["pos_m"])

try:
    import numpy as _np
except ImportError:                          # pragma: no cover
    _np = None


def beam_angle(i):
    """Angle of beam i in the LiDAR frame, rad."""
    return ANGLE_MIN_RAD + i * ANGLE_INC_RAD


def beam_index(angle_rad):
    """Nearest beam to an angle in the LiDAR frame (None outside the 270 degree field)."""
    i = int(round((angle_rad - ANGLE_MIN_RAD) / ANGLE_INC_RAD))
    return i if 0 <= i < N_BEAMS else None


def beam_angles():
    """The 1081 beam angles (numpy array, or list without numpy)."""
    if _np is not None:
        return ANGLE_MIN_RAD + ANGLE_INC_RAD * _np.arange(N_BEAMS)
    return [beam_angle(i) for i in range(N_BEAMS)]


def ranges_m(scan):
    """Ranges in metres; NaN (or None without numpy) where there is no return."""
    if _np is not None:
        s = _np.asarray(scan, dtype=float)
        return _np.where(s > 0, s * 0.001, _np.nan)
    return [v * 0.001 if v > 0 else None for v in scan]


def points(scan, frame="lidar", lidar_pos=LIDAR_POS_M):
    """(xy, index) of the beams with a return: xy in metres in the LiDAR frame (`frame="lidar"`) or in the car frame
    (`frame="car"`: shifted by the scan origin's x and y). Plain lists without numpy."""
    ox, oy = (lidar_pos[0], lidar_pos[1]) if frame == "car" else (0.0, 0.0)
    if _np is not None:
        s = _np.asarray(scan, dtype=float)
        idx = _np.nonzero(s > 0)[0]
        a = ANGLE_MIN_RAD + ANGLE_INC_RAD * idx
        r = s[idx] * 0.001
        return _np.stack([ox + r * _np.cos(a), oy + r * _np.sin(a)], axis=1), idx
    idx = [i for i, v in enumerate(scan) if v > 0]
    xy = [(ox + scan[i] * 0.001 * math.cos(beam_angle(i)), oy + scan[i] * 0.001 * math.sin(beam_angle(i))) for i in idx]
    return xy, idx


def subscan_of(i):
    """Index (0..4) of the sub-scan that holds beam i."""
    for k, (lo, hi) in enumerate(SUBSCAN_SLICES):
        if lo <= i < hi:
            return k
    raise ValueError("beam index out of range")
