"""The robot's shadow on the plate and the sun on the robot (steps R6-R7 of plate_filament.py; the method that met
every acceptance threshold against a path-traced reference):

    V_env    sun visibility of the environment at the robot's surface points: trilinear look-up in the bundle's 3D grid
             (sunvis: CPU rays on the twin of the arranged scene, 2 x 2 x 5 cm, z 0-0.30 m);
    V_r      sun visibility past the robot of the plate's surface points: exact rays (mj_ray) against the robot's
             decimated meshes, after a prefilter on the three strut capsules (only points whose ray toward the sun
             passes within candidate_radius of a strut axis are cast);
    O        sky occlusion by the robot: share of n cosine-weighted directions over the upper hemisphere that hit one of
             the three strut capsules (radius 20 mm), for points within sky_near_radius of the robot;
    darkening = phi (1 - V_r) + (1 - phi) sky_gain O, applied as E = P - (P - A) darkening.
"""
import math

import numpy as np

STRUT_BODIES = ("r01", "r23", "r45")


def venv_lookup(G, P):
    """Trilinear look-up of the grid G (V[z, y, x] with axes x, y, z) at points P (n x 3); clamped at the faces."""
    xs, ys, zs, V = G["x"], G["y"], G["z"], G["V"]
    fx = np.clip((P[:, 0] - xs[0]) / (xs[1] - xs[0]), 0, len(xs) - 1.001)
    fy = np.clip((P[:, 1] - ys[0]) / (ys[1] - ys[0]), 0, len(ys) - 1.001)
    fz = np.clip((P[:, 2] - zs[0]) / (zs[1] - zs[0]), 0, len(zs) - 1.001)
    ix, iy, iz = fx.astype(int), fy.astype(int), fz.astype(int)
    ax, ay, az = fx - ix, fy - iy, fz - iz
    out = np.zeros(len(P), np.float64)
    for dz in (0, 1):
        wz = az if dz else 1 - az
        for dy in (0, 1):
            wy = ay if dy else 1 - ay
            for dx in (0, 1):
                wx = ax if dx else 1 - ax
                out += wx * wy * wz * V[iz + dz, iy + dy, ix + dx]
    return out


def hemi_dirs(n=32):
    """n cosine-weighted directions over the upper hemisphere (Fibonacci spiral), z up."""
    i = np.arange(n) + 0.5
    r = np.sqrt(i / n)
    ph = i * math.pi * (3 - math.sqrt(5))
    return np.stack([r * np.cos(ph), r * np.sin(ph), np.sqrt(np.maximum(1 - r * r, 0))], 1)


def capsules(mj, model, data, body_ids, half_len):
    """The three strut axes as segments (a, b) in the world (each body's local z axis, +- half_len)."""
    out = []
    for i in body_ids:
        p, R = data.xpos[i].copy(), data.xmat[i].reshape(3, 3).copy()
        out.append((p - half_len * R[:, 2], p + half_len * R[:, 2]))
    return out


def ray_segment_distance(P, s, a, b):
    """Distance between the rays P + t s (t >= 0, s unit) and the segment a-b, for every row of P (n x 3): the
    line-line optimum on the segment clamped to it, the ray parameter recomputed for that point and clamped to the
    ray, the segment parameter recomputed once more (a prefilter: it may only err towards casting a ray)."""
    ab = b - a
    w0 = P - a
    B_, C_ = float(s @ ab), max(float(ab @ ab), 1e-12)
    D_, E_ = w0 @ s, w0 @ ab
    den = C_ - B_ * B_
    tc = np.clip(E_ / C_, 0, 1) if den < 1e-12 else np.clip((E_ - B_ * D_) / den, 0, 1)
    sc = np.clip(tc * B_ - D_, 0, None)
    tc = np.clip((E_ + sc * B_) / C_, 0, 1)
    diff = w0 + sc[:, None] * s - tc[:, None] * ab
    return np.linalg.norm(diff, axis=1)


def vr_exact(mj, model, data, P, valid, to_sun, caps, cand_radius, geomgroup):
    """(V_r float32 n, number of cast rays): 0 where the ray from P toward the sun hits the robot's meshes."""
    n = len(P)
    V = np.ones(n, np.float32)
    cand = np.zeros(n, bool)
    for a, b in caps:
        cand |= ray_segment_distance(P, to_sun, a, b) < cand_radius
    cand &= valid
    idx = np.nonzero(cand)[0]
    if len(idx):
        gid = np.zeros(1, np.int32)
        orig = P[idx] + 1e-3 * to_sun
        for j, p in zip(idx, orig):
            if mj.mj_ray(model, data, p, to_sun, geomgroup, 1, -1, gid) >= 0:
                V[j] = 0.0
    return V, int(len(idx))


def capsule_hits(P, D, caps, radius):
    """(len(P), len(D)) bool: the ray P + t D (t > 0) passes within radius of a capsule segment."""
    hit = np.zeros((len(P), len(D)), bool)
    for a, b in caps:
        ab = b - a
        C_ = float(ab @ ab)
        w0 = P - a                                      # (M, 3)
        Bd = D @ ab                                     # (N,)
        Dd = w0 @ D.T                                   # (M, N): w0 . d
        Ed = (w0 @ ab)[:, None]                         # (M, 1): w0 . ab
        den = np.maximum(C_ - Bd[None, :] ** 2, 1e-12)
        sc = np.clip((Bd[None, :] * Ed - C_ * Dd) / den, 0, None)
        tc = np.clip((Ed - Bd[None, :] * Dd) / den, 0, 1)
        diff = w0[:, None, :] + sc[..., None] * D[None, :, :] - tc[..., None] * ab
        hit |= np.einsum("mnk,mnk->mn", diff, diff) < radius * radius
    return hit


def sky_occlusion(P, caps, dirs, radius):
    """O (len(P)): share of the directions dirs from P (offset 1 mm up) that hit one of the capsules."""
    if len(P) == 0:
        return np.zeros(0, np.float32)
    return capsule_hits(P + 1e-3 * np.array([0.0, 0.0, 1.0]), dirs, caps, radius).mean(1).astype(np.float32)
