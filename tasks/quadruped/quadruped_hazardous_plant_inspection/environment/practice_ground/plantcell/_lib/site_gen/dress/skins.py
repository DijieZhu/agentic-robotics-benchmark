"""Skins: subdivided visual meshes over the faces of the structure boxes, with metric texture coordinates.

A wall face's texture runs in world metres: u along the face (the viewer's right), v from the floor up, so that the
dado's top is at the same height on every wall and the pattern runs on across segments, lintels and reveals. The
classic renderer lights per vertex; a grid of about 0.33 m keeps the room lights smooth on large faces.

MuJoCo's texture coordinate v = 0 is the first row of the image (its top): v = 1 - z / tile_v.
"""

from __future__ import annotations

import math

import numpy as np

from .xmath import rows_dot

Z = np.array([0.0, 0.0, 1.0])


def face_mesh(f: dict, off: float, cell: float, tile, wall: bool):
    """Vertices, triangles (front side = f['n']) and texture coordinates of a grid over face f, pushed `off` along
    its normal. f: o (corner), a (unit, along), b (unit, up / second axis), la, lb, n. a x b = n."""
    la, lb = float(f["la"]), float(f["lb"])
    if la < 1e-4 or lb < 1e-4:
        return np.zeros((0, 3)), np.zeros((0, 3), int), np.zeros((0, 2))
    na = max(1, int(math.ceil(la / cell)))
    nb = max(1, int(math.ceil(lb / cell)))
    s = np.linspace(0.0, la, na + 1)
    t = np.linspace(0.0, lb, nb + 1)
    S, T = np.meshgrid(s, t)                                  # (nb+1, na+1)
    P = (np.asarray(f["o"], float)[None, None, :] + S[..., None] * np.asarray(f["a"], float)
         + T[..., None] * np.asarray(f["b"], float) + off * np.asarray(f["n"], float))
    P = P.reshape(-1, 3)
    tu, tv = float(tile[0]), float(tile[1])
    if wall:
        n = np.asarray(f["n"], float)
        right = np.cross(-n, Z)
        u = rows_dot(P, right) / tu
        v = 1.0 - P[:, 2] / tv
    else:
        u = P[:, 0] / tu
        v = 1.0 - P[:, 1] / tv
    # keep texture coordinates small (precision of float32 in the renderer): shift by whole tiles
    u = u - math.floor(float(u.min()))
    if not wall:
        v = v - math.floor(float(v.min()))
    UV = np.stack([u, v], 1)
    F = []
    w = na + 1
    for j in range(nb):
        r0 = j * w
        r1 = (j + 1) * w
        i = np.arange(na)
        F.append(np.stack([r0 + i, r0 + i + 1, r1 + i + 1], 1))
        F.append(np.stack([r0 + i, r1 + i + 1, r1 + i], 1))
    return P, np.concatenate(F, 0).astype(int), UV


def _inside_any(p, W, skip, kinds_lo, kinds_hi):
    """Is point p inside one of the structure boxes (other than `skip`)?"""
    m = np.all((kinds_lo <= p + 1e-6) & (p - 1e-6 <= kinds_hi), axis=1)
    if skip is not None:
        m[skip] = False
    return bool(m.any())


def wall_faces(lo, hi, W, gi, struct_idx=None):
    """The visible faces of an axis-aligned structure box [lo, hi]: the four vertical faces and, for a box lower
    than the ceiling, its top. A face whose centre is buried in another structure box is skipped. Each face carries a
    probe point 5 cm in front of it at 1 m (for the room lookup)."""
    lo = np.asarray(lo, float)
    hi = np.asarray(hi, float)
    if struct_idx is None:
        struct_idx = W.get("_struct_idx")
    SL = W["lo"][struct_idx]
    SH = W["hi"][struct_idx]
    skip = None
    k = np.where(struct_idx == gi)[0]
    if len(k):
        skip = int(k[0])
    out = []
    z0, z1 = lo[2], hi[2]
    zp = min(max(z0 + 0.05, 1.0), z1 - 0.05)
    for axis in (0, 1):
        for sgn in (-1.0, 1.0):
            n = np.zeros(3)
            n[axis] = sgn
            right = np.cross(-n, Z)
            other = 1 - axis
            coord = hi[axis] if sgn > 0 else lo[axis]
            # the corner where the 'right' coordinate is smallest, at the bottom
            if right[other] > 0:
                start = lo[other]
            else:
                start = hi[other]
            o = np.zeros(3)
            o[axis] = coord
            o[other] = start
            o[2] = z0
            la = hi[other] - lo[other]
            c = o + right * la / 2
            c[2] = zp
            if _inside_any(c + n * 0.01, W, skip, SL, SH):
                continue
            out.append({"o": o, "a": right, "b": Z.copy(), "la": la, "lb": z1 - z0, "n": n,
                        "probe": c + n * 0.05, "geom": int(gi)})
    if z1 < 2.5:
        o = np.array([lo[0], lo[1], z1])
        out.append({"o": o, "a": np.array([1.0, 0, 0]), "b": np.array([0, 1.0, 0]), "la": hi[0] - lo[0],
                    "lb": hi[1] - lo[1], "n": Z.copy(), "probe": np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2,
                                                                          z1 + 0.05]), "geom": int(gi), "top": True})
    return out
