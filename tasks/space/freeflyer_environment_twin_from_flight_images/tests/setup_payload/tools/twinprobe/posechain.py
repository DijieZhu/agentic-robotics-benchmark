"""Frames, quaternions and the pose chain of the flight twin; the time convention; cameras.json.

Frames (all in metres):
  twin    the frame the submission chooses; the grader neither aligns nor scales it
  wide    the forward wide camera: origin at the camera centre, +z along the optical axis, +x image right,
          +y image down
  depth   the forward depth camera: origin at its centre, the frame of the ray table (one unit ray per pixel)
  body    the robot's body frame, the frame whose pose the on-board estimate records

A rigid transform is a 4 x 4 matrix T_a_b that maps coordinates in frame b to coordinates in frame a:
X_a = T_a_b X_b. Its rotation part turns b vectors into a vectors; its translation is the origin of b in a.

  T_body_wide   the camera mounting of the sensor sheet: the camera centre in the body frame and the rotation that
                turns camera vectors into body vectors (camera +z, the optical axis, is body +x, forward)
  T_depth_wide  the two-camera transform: X_depth = R X_wide + t (the wide camera's centre maps to t)
  T_twin_depth  = T_twin_wide . inverse(T_depth_wide)

Quaternions. cameras.json writes (w, x, y, z) ("quaternion_wxyz"); the sensor files write (x, y, z, w) and name the
order in their keys ("quaternion_xyzw"). A quaternion q and -q are the same rotation. Every quaternion is
normalised before it becomes a matrix.

Time. A frame's t_s is its recorded image stamp. Where the on-board estimate is matched to a frame it is read at
(frame stamp - offset), the offset being the configuration's key_frames.time_offset_wide_minus_depth_stamp:
position linearly, orientation along the shortest arc (spherical linear interpolation) between the two estimate
samples around that instant; outside the recorded span the nearest end sample is held and the reading is marked
outside.

cameras.json ("flight-twin-cameras/1"):
    {"format": "flight-twin-cameras/1",
     "frames": {"f_0123": {"position": [x, y, z], "quaternion_wxyz": [w, x, y, z]}, ...}}
the wide camera's pose in the twin frame per frame: position = the camera centre, the quaternion turns camera
vectors into twin vectors. Only the key frames are read; other entries are ignored.
"""
import json
import math

import numpy as np

from . import need

CAMERAS_FORMAT = "flight-twin-cameras/1"


# ---------------------------------------------------------------------------------------------------------------
# quaternions and rigid transforms

def _unit(q):
    q = np.asarray(q, float).reshape(4)
    n = math.sqrt(float(q @ q))
    if not np.isfinite(n) or n == 0.0:
        raise ValueError("a quaternion must be finite and non-zero")
    return q / n


def quat_wxyz_to_R(q):
    """Rotation matrix of a (w, x, y, z) quaternion (normalised first)."""
    w, x, y, z = _unit(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def quat_xyzw_to_R(q):
    """Rotation matrix of an (x, y, z, w) quaternion (normalised first)."""
    x, y, z, w = np.asarray(q, float).reshape(4)
    return quat_wxyz_to_R([w, x, y, z])


def xyzw_to_wxyz(q):
    x, y, z, w = np.asarray(q, float).reshape(4)
    return np.array([w, x, y, z])


def wxyz_to_xyzw(q):
    w, x, y, z = np.asarray(q, float).reshape(4)
    return np.array([x, y, z, w])


def R_to_quat_wxyz(R):
    """(w, x, y, z) unit quaternion of a rotation matrix, w >= 0 (Shepperd's method)."""
    R = np.asarray(R, float).reshape(3, 3)
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr >= max(R[0, 0], R[1, 1], R[2, 2]):
        s = 2.0 * math.sqrt(1.0 + tr)
        q = np.array([0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s])
    elif R[0, 0] >= R[1, 1] and R[0, 0] >= R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        q = np.array([(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s])
    elif R[1, 1] >= R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        q = np.array([(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s])
    else:
        s = 2.0 * math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        q = np.array([(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s])
    q = q / math.sqrt(float(q @ q))
    return -q if q[0] < 0 else q


def quat_mul_wxyz(a, b):
    """Hamilton product a * b of (w, x, y, z) quaternions (the rotation b first, then a)."""
    aw, ax, ay, az = np.asarray(a, float)
    bw, bx, by, bz = np.asarray(b, float)
    return np.array([aw * bw - ax * bx - ay * by - az * bz,
                     aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw])


def rigid(R, t):
    T = np.eye(4)
    T[:3, :3] = np.asarray(R, float).reshape(3, 3)
    T[:3, 3] = np.asarray(t, float).reshape(3)
    return T


def inv_rigid(T):
    T = np.asarray(T, float)
    R = T[:3, :3]
    return rigid(R.T, -R.T @ T[:3, 3])


def apply(T, X):
    """X (n, 3) or (3,) points of frame b mapped by T_a_b into frame a."""
    X = np.asarray(X, float)
    return X @ T[:3, :3].T + T[:3, 3]


def rotation_angle_deg(Ra, Rb):
    """Angle (degrees) of the rotation that takes Ra to Rb (atan2 form: accurate near 0 and near 180 degrees)."""
    M = np.asarray(Ra, float).T @ np.asarray(Rb, float)
    s = 0.5 * math.sqrt((M[2, 1] - M[1, 2]) ** 2 + (M[0, 2] - M[2, 0]) ** 2 + (M[1, 0] - M[0, 1]) ** 2)
    c = 0.5 * (M[0, 0] + M[1, 1] + M[2, 2] - 1.0)
    return math.degrees(math.atan2(s, c))


# ---------------------------------------------------------------------------------------------------------------
# the sensor sheet's transforms

def T_body_wide(cfg):
    """The camera mounting: X_body = T_body_wide X_wide."""
    return rigid(quat_xyzw_to_R(need(cfg, "wide_camera.body_mount_quaternion_xyzw")),
                 need(cfg, "wide_camera.body_mount_translation"))


def T_depth_wide(cfg):
    """The two-camera transform: X_depth = R X_wide + t."""
    return rigid(quat_xyzw_to_R(need(cfg, "two_camera_transform.quaternion_xyzw")),
                 need(cfg, "two_camera_transform.translation"))


def depth_pose(T_twin_wide, T_dw):
    """T_twin_depth = T_twin_wide . inverse(T_depth_wide)."""
    return np.asarray(T_twin_wide, float) @ inv_rigid(T_dw)


def body_origin_in_depth(cfg):
    """The robot's centre (the body frame's origin) in the depth camera frame, from the mounting and the transform:
    X_depth = T_depth_wide . inverse(T_body_wide) . 0. The hazard check starts its moves here."""
    T = T_depth_wide(cfg) @ inv_rigid(T_body_wide(cfg))
    return T[:3, 3].copy()


# ---------------------------------------------------------------------------------------------------------------
# the time convention and the on-board estimate

def _check_times(times):
    t = np.asarray(times, float).reshape(-1)
    if len(t) < 2 or not np.all(np.isfinite(t)) or not np.all(np.diff(t) > 0):
        raise ValueError("the estimate's times must be finite and strictly increasing (at least two samples)")
    return t


def slerp_wxyz(q0, q1, a):
    """Spherical linear interpolation along the shortest arc from q0 (a = 0) to q1 (a = 1); (w, x, y, z)."""
    q0 = _unit(q0)
    q1 = _unit(q1)
    rel = quat_mul_wxyz(q0 * np.array([1.0, -1.0, -1.0, -1.0]), q1)        # conj(q0) * q1
    if rel[0] < 0:
        rel = -rel
    v = rel[1:]
    nv = math.sqrt(float(v @ v))
    if nv == 0.0:
        return q0.copy()
    ang = 2.0 * math.atan2(nv, rel[0])
    half = 0.5 * a * ang
    step = np.concatenate([[math.cos(half)], math.sin(half) * v / nv])
    out = quat_mul_wxyz(q0, step)
    return out / math.sqrt(float(out @ out))


def interpolate_estimate(times, positions, quats_xyzw, at):
    """The estimate's body pose at time `at`: (T_ref_body 4 x 4, inside). Position linear, orientation by slerp
    between the two samples around `at`; outside the recorded span the nearest end sample is held (inside False)."""
    t = _check_times(times)
    P = np.asarray(positions, float).reshape(-1, 3)
    Q = np.asarray(quats_xyzw, float).reshape(-1, 4)
    if len(P) != len(t) or len(Q) != len(t):
        raise ValueError("times, positions and quaternions must have the same length")
    at = float(at)
    if at <= t[0] or at >= t[-1]:
        k = 0 if at <= t[0] else len(t) - 1
        inside = bool(t[0] <= at <= t[-1])
        return rigid(quat_xyzw_to_R(Q[k]), P[k]), inside
    k = int(np.searchsorted(t, at, side="right"))
    a = (at - t[k - 1]) / (t[k] - t[k - 1])
    p = (1.0 - a) * P[k - 1] + a * P[k]
    q = slerp_wxyz(xyzw_to_wxyz(Q[k - 1]), xyzw_to_wxyz(Q[k]), a)
    return rigid(quat_wxyz_to_R(q), p), True


def estimate_instant(frame_stamp, offset):
    """The instant at which the estimate is read for a frame: its recorded stamp minus the offset."""
    return float(frame_stamp) - float(offset)


def estimate_wide_pose(times, positions, quats_xyzw, frame_stamp, cfg):
    """The wide camera's pose by the on-board estimate for one frame: (T_ref_wide, inside) with
    T_ref_wide = T_ref_body(frame stamp - offset) . T_body_wide."""
    off = need(cfg, "key_frames.time_offset_wide_minus_depth_stamp")
    Tb, inside = interpolate_estimate(times, positions, quats_xyzw, estimate_instant(frame_stamp, off))
    return Tb @ T_body_wide(cfg), inside


# ---------------------------------------------------------------------------------------------------------------
# cameras.json

def _finite_numbers(v, n):
    if not isinstance(v, (list, tuple)) or len(v) != n:
        return None
    out = []
    for x in v:
        if isinstance(x, bool) or not isinstance(x, (int, float)):
            return None
        out.append(float(x))
    a = np.array(out)
    return a if np.all(np.isfinite(a)) else None


def parse_cameras(raw, key_frames, cfg):
    """Check cameras.json bytes. Returns (poses, frame_errors, file_error):
    file_error: why the file as a whole is not a valid cameras.json (the flight's validity gate fails), or None;
    poses: {frame: T_twin_wide} for the key frames with a valid pose;
    frame_errors: {frame: why} for the key frames without one (no pose declared, a number not finite, the
    quaternion's norm outside the configured range)."""
    lo, hi = need(cfg, "gate.quaternion_norm_range")
    try:
        obj = json.loads(raw.decode("utf-8"))
    except MemoryError:
        raise
    except Exception as exc:  # noqa: BLE001 - UnicodeDecodeError, ValueError, RecursionError
        return {}, {}, "cameras.json does not parse as JSON: %s" % str(exc)[:160]
    if not isinstance(obj, dict):
        return {}, {}, 'cameras.json must be an object {"format", "frames"}'
    if obj.get("format") != CAMERAS_FORMAT:
        return {}, {}, 'cameras.json "format" must be %r' % CAMERAS_FORMAT
    frames = obj.get("frames")
    if not isinstance(frames, dict):
        return {}, {}, 'cameras.json "frames" must be an object {frame name: {"position", "quaternion_wxyz"}}'
    poses, errors = {}, {}
    for name in key_frames:
        e = frames.get(name)
        if e is None:
            errors[name] = "no pose declared"
            continue
        if not isinstance(e, dict):
            errors[name] = "the entry is not an object"
            continue
        p = _finite_numbers(e.get("position"), 3)
        q = _finite_numbers(e.get("quaternion_wxyz"), 4)
        if p is None:
            errors[name] = "position is not three finite numbers"
            continue
        if q is None:
            errors[name] = "quaternion_wxyz is not four finite numbers"
            continue
        nq = math.sqrt(float(q @ q))
        if not (lo <= nq <= hi):
            errors[name] = "the quaternion's norm %.6g is outside [%g, %g]" % (nq, lo, hi)
            continue
        poses[name] = rigid(quat_wxyz_to_R(q), p)
    return poses, errors, None


def cameras_entry(T_twin_wide):
    """The cameras.json entry of a pose (full precision)."""
    T = np.asarray(T_twin_wide, float)
    return {"position": [float(x) for x in T[:3, 3]], "quaternion_wxyz": [float(x) for x in R_to_quat_wxyz(T[:3, :3])]}
