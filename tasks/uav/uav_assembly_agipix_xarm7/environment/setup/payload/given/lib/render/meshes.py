"""Small triangle meshes with texture coordinates for the scale references (backend-agnostic numpy data).

A Mesh is (vertices (N, 3) metres, faces (M, 3) int, uv (N, 2) or None); faces are counter-clockwise seen from
outside (outward normals).  UV convention (textures.py): u = 0 .. 1 left to right of the texture, v = 0 at the
texture's top row.  For a face lying in a plane z = const seen from +z, `quad` / `polygon_face` put u along +x and
v = 0 at the +y edge, so a texture reads upright when the face is viewed from above with +y pointing away.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class Mesh:
    vertices: np.ndarray
    faces: np.ndarray
    uv: Optional[np.ndarray] = None

    def transformed(self, R=None, t=None) -> "Mesh":
        v = self.vertices
        if R is not None:
            v = v @ np.asarray(R, float).T
        if t is not None:
            v = v + np.asarray(t, float)
        return Mesh(v, self.faces.copy(), None if self.uv is None else self.uv.copy())


def quad(sx, sy, z=0.0) -> Mesh:
    """A textured rectangle [-sx/2, sx/2] x [-sy/2, sy/2] at height z, normal +z."""
    v = np.array([[-sx / 2, -sy / 2, z], [sx / 2, -sy / 2, z], [sx / 2, sy / 2, z], [-sx / 2, sy / 2, z]], float)
    uv = np.array([[0, 1], [1, 1], [1, 0], [0, 0]], float)
    f = np.array([[0, 1, 2], [0, 2, 3]], int)
    return Mesh(v, f, uv)


def rounded_rect(sx, sy, r, n_arc=8) -> np.ndarray:
    """Counter-clockwise outline (K, 2) of a rectangle with corner radius r, centred at the origin."""
    pts = []
    corners = [(sx / 2 - r, -sy / 2 + r, -90.0), (sx / 2 - r, sy / 2 - r, 0.0), (-sx / 2 + r, sy / 2 - r, 90.0),
               (-sx / 2 + r, -sy / 2 + r, 180.0)]
    for cx, cy, a0 in corners:
        for k in range(n_arc + 1):
            a = math.radians(a0 + 90.0 * k / n_arc)
            pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    return np.array(pts, float)


def polygon_face(poly_xy, z, bbox=None, up=True) -> Mesh:
    """Fan-triangulated convex polygon at height z; uv from the bounding box (u along +x, v = 0 at max y)."""
    P = np.asarray(poly_xy, float)
    x0, y0, x1, y1 = bbox if bbox is not None else (P[:, 0].min(), P[:, 1].min(), P[:, 0].max(), P[:, 1].max())
    c = P.mean(axis=0)
    V = np.vstack([np.r_[c, z][None, :], np.c_[P, np.full(len(P), z)]])
    uv = np.c_[(V[:, 0] - x0) / (x1 - x0), (y1 - V[:, 1]) / (y1 - y0)]
    n = len(P)
    f = np.array([[0, 1 + k, 1 + (k + 1) % n] for k in range(n)], int)
    if not up:
        f = f[:, ::-1]
    return Mesh(V, f, uv)


def prism(poly_xy, z0, z1, top=True, bottom=True) -> Mesh:
    """Closed extrusion of a convex CCW polygon between z0 and z1 (no uv; plain material)."""
    P = np.asarray(poly_xy, float)
    n = len(P)
    bot = np.c_[P, np.full(n, z0)]
    tp = np.c_[P, np.full(n, z1)]
    V = [bot, tp]
    F = []
    for k in range(n):
        a, b = k, (k + 1) % n
        F += [[a, b, n + b], [a, n + b, n + a]]
    base = 2 * n
    cb = np.r_[P.mean(axis=0), z0]
    ct = np.r_[P.mean(axis=0), z1]
    V.append(cb[None, :])
    V.append(ct[None, :])
    if bottom:
        F += [[base, (k + 1) % n, k] for k in range(n)]
    if top:
        F += [[base + 1, n + k, n + (k + 1) % n] for k in range(n)]
    return Mesh(np.vstack(V), np.array(F, int), None)


def revolve(profile_rz: Sequence[Tuple[float, float]], n_seg=64, z_top=None, z_bottom=None) -> Mesh:
    """Surface of revolution about +z of a profile [(r, z), ...] listed bottom to top; u = angle / 2 pi,
    v = (z_top - z) / (z_top - z_bottom) (v = 0 at the top).  Seam vertices duplicated for a clean u wrap."""
    prof = np.asarray(profile_rz, float)
    zt = prof[:, 1].max() if z_top is None else z_top
    zb = prof[:, 1].min() if z_bottom is None else z_bottom
    m = len(prof)
    V, UV = [], []
    for j in range(n_seg + 1):
        a = 2 * math.pi * j / n_seg
        ca, sa = math.cos(a), math.sin(a)
        for r, z in prof:
            V.append((r * ca, r * sa, z))
            UV.append((j / n_seg, (zt - z) / (zt - zb)))
    F = []
    for j in range(n_seg):
        for i in range(m - 1):
            a0 = j * m + i
            a1 = a0 + 1
            b0 = (j + 1) * m + i
            b1 = b0 + 1
            F += [[a0, b0, b1], [a0, b1, a1]]
    return Mesh(np.array(V, float), np.array(F, int), np.array(UV, float))


def disc(r, z, n_seg=64, up=True) -> Mesh:
    ang = np.linspace(0, 2 * math.pi, n_seg, endpoint=False)
    poly = np.c_[r * np.cos(ang), r * np.sin(ang)]
    return polygon_face(poly, z, up=up)


def merge(meshes: List[Mesh]) -> Mesh:
    V, F, UV, off = [], [], [], 0
    has_uv = all(m.uv is not None for m in meshes)
    for m in meshes:
        V.append(m.vertices)
        F.append(m.faces + off)
        if has_uv:
            UV.append(m.uv)
        off += len(m.vertices)
    return Mesh(np.vstack(V), np.vstack(F), np.vstack(UV) if has_uv else None)
