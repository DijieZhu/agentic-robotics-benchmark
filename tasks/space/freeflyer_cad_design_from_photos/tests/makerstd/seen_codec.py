"""The scoring-sample table ffcad.seen/1: which points of a maker file's triangles are graded.

One table per maker file. No header. Records of 8 bytes, little-endian struct '<IHH':
    uint32 t   index into the file's flat triangle list (makerstd.dae: Mesh.F)
    uint16 U   first barycentric coordinate times 65535
    uint16 V   second barycentric coordinate times 65535, with U + V <= 65535
Point: p = V0 + (U / 65535) (V1 - V0) + (V / 65535) (V2 - V0), V0..V2 the triangle's corners in the file's frame
(mm, node matrices applied); the instance transform of a tag then places it.

Quantisation error of a point is at most 0.5 / 65535 (|V1 - V0| + |V2 - V0|) (bound()). The writer (the visibility
sampling) and the reader (the grader) both use this module.
"""
import struct

import numpy as np

from .errors import MakerStdError

SCHEMA = "ffcad.seen/1"
RECORD = struct.Struct("<IHH")
DTYPE = np.dtype([("t", "<u4"), ("u", "<u2"), ("v", "<u2")])
SCALE = 65535
MAX_RECORDS = 125000


def quantize(u, v):
    """Integer barycentrics (U, V), U + V <= 65535, each within half a step of u * 65535 and v * 65535."""
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    if u.shape != v.shape:
        raise MakerStdError("u and v differ in shape")
    if u.size and (u.min() < 0.0 or v.min() < 0.0 or (u + v).max() > 1.0 + 1e-12):
        raise MakerStdError("barycentric coordinates outside the triangle")
    xu, xv = u * SCALE, v * SCALE
    U = np.rint(xu)
    V = np.rint(xv)
    over = (U + V) > SCALE
    # both were rounded up by exactly half a step: take the half step back from the one rounded up more
    take_u = over & ((U - xu) >= (V - xv))
    U = np.where(take_u, U - 1, U)
    V = np.where(over & ~take_u, V - 1, V)
    U = np.clip(U, 0, SCALE).astype(np.uint16)
    V = np.clip(V, 0, SCALE).astype(np.uint16)
    return U, V


def encode_q(tri, U, V):
    tri = np.asarray(tri, dtype=np.int64)
    U = np.asarray(U, dtype=np.int64)
    V = np.asarray(V, dtype=np.int64)
    if not (tri.shape == U.shape == V.shape) or tri.ndim != 1:
        raise MakerStdError("tri, U and V must be 1-D of one length")
    if tri.size and (tri.min() < 0 or tri.max() > 0xFFFFFFFF):
        raise MakerStdError("triangle index out of the uint32 range")
    if U.size and (U.min() < 0 or V.min() < 0 or (U + V).max() > SCALE):
        raise MakerStdError("U, V out of range")
    rec = np.empty(tri.size, dtype=DTYPE)
    rec["t"] = tri
    rec["u"] = U
    rec["v"] = V
    return rec.tobytes()


def encode(tri, u, v):
    U, V = quantize(u, v)
    return encode_q(tri, U, V)


def decode(data, n_triangles=None, max_records=MAX_RECORDS):
    """(tri uint32, U uint16, V uint16) of a table; checks length, ranges and (if given) the triangle count."""
    data = bytes(data)
    if len(data) % RECORD.size:
        raise MakerStdError("seen table of %d bytes is not a whole number of %d-byte records" % (len(data), RECORD.size))
    rec = np.frombuffer(data, dtype=DTYPE)
    if max_records is not None and rec.size > max_records:
        raise MakerStdError("seen table holds %d records, more than %d" % (rec.size, max_records))
    tri = rec["t"].astype(np.int64)
    U = rec["u"].astype(np.int64)
    V = rec["v"].astype(np.int64)
    if U.size and (U + V).max() > SCALE:
        raise MakerStdError("seen table record with U + V > 65535")
    if n_triangles is not None and tri.size and tri.max() >= n_triangles:
        raise MakerStdError("seen table names triangle %d of a file with %d" % (int(tri.max()), n_triangles))
    return tri, U, V


def barycentric(U, V):
    return np.asarray(U, dtype=np.float64) / SCALE, np.asarray(V, dtype=np.float64) / SCALE


def points(triangles_mm, tri, U, V):
    """The decoded points, in the frame of the given triangles (M, 3, 3)."""
    T = np.asarray(triangles_mm, dtype=np.float64)[np.asarray(tri, dtype=np.int64)]
    u, v = barycentric(U, V)
    return (T[:, 0] + u[:, None] * (T[:, 1] - T[:, 0])) + v[:, None] * (T[:, 2] - T[:, 0])


def bound(triangles_mm, tri=None):
    """Upper bound of the quantisation error (mm) of each named triangle (all when tri is None)."""
    T = np.asarray(triangles_mm, dtype=np.float64)
    if tri is not None:
        T = T[np.asarray(tri, dtype=np.int64)]
    return 0.5 / SCALE * (np.linalg.norm(T[:, 1] - T[:, 0], axis=1) + np.linalg.norm(T[:, 2] - T[:, 0], axis=1))
