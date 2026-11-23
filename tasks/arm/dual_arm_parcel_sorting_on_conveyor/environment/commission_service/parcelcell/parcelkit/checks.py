"""Mesh-versus-collision numbers: how far the visual mesh pokes out of the simulator's collision shape, and how far
the collision surface lies outside the visible surface (what a finger meets before the camera says it should)."""
from __future__ import annotations

import numpy as np


def sd_box(P, half):
    q = np.abs(P) - np.asarray(half)
    return np.linalg.norm(np.maximum(q, 0), axis=-1) + np.minimum(np.max(q, axis=-1), 0)


def out_ellipsoid(P, semi):
    """Distance from P to an axis-aligned ellipsoid for points outside (0 inside)."""
    a = np.asarray(semi, float)
    P = np.asarray(P, float)
    inside = np.sum((P / a) ** 2, axis=-1) <= 1
    p = np.abs(P)
    lo = np.zeros(len(P))
    hi = np.full(len(P), np.max(a) * (np.linalg.norm(p, axis=-1) + 1.0) * 4)
    for _ in range(80):
        t = 0.5 * (lo + hi)
        f = np.sum((a * p / (a * a + t[:, None])) ** 2, axis=-1) - 1
        lo = np.where(f > 0, t, lo)
        hi = np.where(f > 0, hi, t)
    t = 0.5 * (lo + hi)
    x = a * a * p / (a * a + t[:, None])
    d = np.linalg.norm(p - x, axis=-1)
    return np.where(inside, 0.0, d)


def sample_triangles(V, F, spacing):
    A, B, C = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    area = 0.5 * np.linalg.norm(np.cross(B - A, C - A), axis=-1)
    n = np.maximum(1, np.ceil(area / (spacing * spacing)).astype(int))
    rng = np.random.default_rng(0)
    idx = np.repeat(np.arange(len(F)), n)
    r1, r2 = rng.uniform(size=len(idx)), rng.uniform(size=len(idx))
    s = np.sqrt(r1)
    pts = (1 - s)[:, None] * A[idx] + (s * (1 - r2))[:, None] * B[idx] + (s * r2)[:, None] * C[idx]
    return np.concatenate([V, pts])


def outside_distance(shape, P):
    c = shape.collision()
    if c["type"] == "box":
        return np.maximum(sd_box(P, c["half"]), 0)
    db = np.maximum(sd_box(P, c["box_half"]), 0)
    de = out_ellipsoid(P, c["ellipsoid_semi"])
    return np.minimum(db, de)


def collision_surface_samples(shape, spacing=0.003):
    c = shape.collision()
    pts = []

    def box_pts(half):
        out = []
        for ax in range(3):
            t1, t2 = (ax + 1) % 3, (ax + 2) % 3
            u = np.arange(-half[t1], half[t1] + 1e-9, spacing)
            v = np.arange(-half[t2], half[t2] + 1e-9, spacing)
            U, W = np.meshgrid(u, v)
            for s in (1, -1):
                P = np.zeros(U.shape + (3,))
                P[..., ax] = s * half[ax]
                P[..., t1], P[..., t2] = U, W
                out.append(P.reshape(-1, 3))
        return np.concatenate(out)
    if c["type"] == "box":
        return box_pts(c["half"])
    bp = box_pts(c["box_half"])
    a = np.asarray(c["ellipsoid_semi"])
    bp = bp[np.sum((bp / a) ** 2, axis=-1) > 1.0 + 1e-6]
    n_th = int(np.pi * a.max() / spacing)
    th = np.linspace(0, np.pi, n_th)
    ph = np.linspace(0, 2 * np.pi, 2 * n_th, endpoint=False)
    T, Pp = np.meshgrid(th, ph)
    E = np.stack([a[0] * np.sin(T) * np.cos(Pp), a[1] * np.sin(T) * np.sin(Pp), a[2] * np.cos(T)], -1).reshape(-1, 3)
    E = E[sd_box(E, c["box_half"]) > 1e-6]
    return np.concatenate([bp, E])


def mesh_vs_collision(shape, V, F, label_vertices=None, spacing=0.0015):
    from scipy.spatial import cKDTree
    P = sample_triangles(np.asarray(V), np.asarray(F), spacing)
    d_out = outside_distance(shape, P)
    res = {"visual_max_outside_mm": round(float(d_out.max()) * 1000, 3),
           "visual_points_outside_over_0p1mm": int(np.sum(d_out > 1e-4)),
           "n_visual_samples": int(len(P))}
    if label_vertices is not None:
        dl = outside_distance(shape, np.asarray(label_vertices).reshape(-1, 3))
        res["label_surface_max_outside_mm"] = round(float(dl.max()) * 1000, 3)
    S = collision_surface_samples(shape)
    tree = cKDTree(P)
    dist, _ = tree.query(S)
    res["collision_surface_to_visual_mm"] = {"max": round(float(dist.max()) * 1000, 2),
                                             "p95": round(float(np.percentile(dist, 95)) * 1000, 2),
                                             "mean": round(float(dist.mean()) * 1000, 2)}
    res["note"] = ("visual_max_outside: worst distance of the visual mesh (waybill sheet included) outside the "
                   "collision shape; collision_surface_to_visual: how far the collision surface lies from the "
                   "visible surface (sampling error about 1 mm)")
    return res
