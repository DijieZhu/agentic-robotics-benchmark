"""URDF conventions and forward kinematics in plain Python floats (no numpy, no BLAS: the same bits on every CPU).

    origin(xyz, rpy)   T = [R | xyz] with R = Rz(yaw) Ry(pitch) Rx(roll)   (URDF <origin>, fixed axes x, y, z)
    joint(axis, q)     rotation by q about the unit axis (Rodrigues), the URDF joint motion in the joint frame
    child frame        T_parent_child(q) = origin(joint xyz, joint rpy) . joint(axis, q)
    visual             T_body_visual = T_body_link . origin(visual xyz, visual rpy)
    axis line          point = translation of T_body_parent . origin(joint), direction = its rotation . unit axis
"""
import math


def identity():
    return [[1.0 if i == j else 0.0 for j in range(4)] for i in range(4)]


def mul(A, B):
    return [[((A[i][0] * B[0][j] + A[i][1] * B[1][j]) + A[i][2] * B[2][j]) + A[i][3] * B[3][j] for j in range(4)]
            for i in range(4)]


def rpy_rotation(roll, pitch, yaw):
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    Rx = [[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]]
    Ry = [[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]]
    Rz = [[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]]
    return _mul3(_mul3(Rz, Ry), Rx)


def _mul3(A, B):
    return [[(A[i][0] * B[0][j] + A[i][1] * B[1][j]) + A[i][2] * B[2][j] for j in range(3)] for i in range(3)]


def origin(xyz, rpy):
    R = rpy_rotation(*rpy)
    return [[R[0][0], R[0][1], R[0][2], float(xyz[0])],
            [R[1][0], R[1][1], R[1][2], float(xyz[1])],
            [R[2][0], R[2][1], R[2][2], float(xyz[2])],
            [0.0, 0.0, 0.0, 1.0]]


def unit(v):
    n = math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
    return [v[0] / n, v[1] / n, v[2] / n]


def joint(axis, q):
    x, y, z = unit(axis)
    c, s = math.cos(q), math.sin(q)
    t = 1.0 - c
    return [[t * x * x + c, t * x * y - s * z, t * x * z + s * y, 0.0],
            [t * x * y + s * z, t * y * y + c, t * y * z - s * x, 0.0],
            [t * x * z - s * y, t * y * z + s * x, t * z * z + c, 0.0],
            [0.0, 0.0, 0.0, 1.0]]


def rotate(T, v):
    return [(T[i][0] * v[0] + T[i][1] * v[1]) + T[i][2] * v[2] for i in range(3)]


def apply(T, p):
    return [((T[i][0] * p[0] + T[i][1] * p[1]) + T[i][2] * p[2]) + T[i][3] for i in range(3)]


def chain(joints, q, root="body"):
    """Link frames in the root frame. joints: [{name, type, parent, child, xyz, rpy, axis}] in an order where each
    parent comes before its children; q: {joint name: angle} for the revolute joints.
    Returns ({link: T}, {joint name: {'point': [x, y, z], 'dir': [x, y, z]}})."""
    frames = {root: identity()}
    axes = {}
    for j in joints:
        if j["parent"] not in frames:
            raise ValueError("joint %s: parent %s not placed yet" % (j["name"], j["parent"]))
        T = mul(frames[j["parent"]], origin(j["xyz"], j["rpy"]))
        if j["type"] == "revolute":
            a = unit(j["axis"])
            axes[j["name"]] = {"point": [T[0][3], T[1][3], T[2][3]], "dir": rotate(T, a)}
            T = mul(T, joint(a, q[j["name"]]))
        elif j["type"] != "fixed":
            raise ValueError("joint %s: type %s not handled" % (j["name"], j["type"]))
        frames[j["child"]] = T
    return frames, axes
