"""Small numpy noise and image helpers (deterministic from a numpy Generator)."""
from __future__ import annotations

import cv2
import numpy as np


def lin(c):
    c = np.asarray(c, np.float32)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4).astype(np.float32)


def srgb(x):
    x = np.clip(np.asarray(x, np.float32), 0, 1)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1 / 2.4) - 0.055).astype(np.float32)


def fbm2(rng, h, w, cell_px, octaves=4, gain=0.5, lac=2.0):
    """Value-noise fBm on an h x w grid, feature size ``cell_px`` pixels for the first octave. Mean 0, about +-1."""
    out = np.zeros((h, w), np.float32)
    amp, tot, c = 1.0, 0.0, float(cell_px)
    for _ in range(octaves):
        gh, gw = max(2, int(np.ceil(h / c)) + 2), max(2, int(np.ceil(w / c)) + 2)
        g = rng.uniform(-1, 1, (gh, gw)).astype(np.float32)
        up = cv2.resize(g, (int(gw * c), int(gh * c)), interpolation=cv2.INTER_CUBIC)
        oy, ox = int(rng.integers(0, max(1, up.shape[0] - h))), int(rng.integers(0, max(1, up.shape[1] - w)))
        out += amp * up[oy:oy + h, ox:ox + w]
        tot += amp
        amp *= gain
        c = max(1.0, c / lac)
    return out / tot


def _hash3(ix, iy, iz, seed):
    h = (ix.astype(np.int64) * 73856093) ^ (iy.astype(np.int64) * 19349663) ^ (iz.astype(np.int64) * 83492791) \
        ^ ((int(seed) & 0xFFFFF) * 2654435761)
    h = (h ^ (h >> 13)) * 1274126177
    h = h ^ (h >> 16)
    return ((h & 0xFFFFFF).astype(np.float32) / float(0xFFFFFF)) * 2.0 - 1.0


def value3(P, scale, seed):
    """Smooth 3-D value noise at points P (..., 3) in metres, feature size ``scale`` m. About +-1."""
    q = np.asarray(P, np.float64) / scale
    i0 = np.floor(q).astype(np.int64)
    f = (q - i0).astype(np.float32)
    f = f * f * (3 - 2 * f)
    out = 0.0
    for dx in (0, 1):
        wx = f[..., 0] if dx else 1 - f[..., 0]
        for dy in (0, 1):
            wy = f[..., 1] if dy else 1 - f[..., 1]
            for dz in (0, 1):
                wz = f[..., 2] if dz else 1 - f[..., 2]
                out = out + wx * wy * wz * _hash3(i0[..., 0] + dx, i0[..., 1] + dy, i0[..., 2] + dz, seed)
    return out


def fbm3(P, scale, seed, octaves=3, gain=0.5):
    out, amp, tot, s = 0.0, 1.0, 0.0, scale
    for k in range(octaves):
        out = out + amp * value3(P, s, seed + 101 * k)
        tot += amp
        amp *= gain
        s *= 0.5
    return out / tot


def smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0, 1)
    return t * t * (3 - 2 * t)


def blur(img, sigma_px):
    if sigma_px <= 0.05:
        return img
    k = int(np.ceil(sigma_px * 3)) * 2 + 1
    return cv2.GaussianBlur(img, (k, k), sigma_px, borderType=cv2.BORDER_REFLECT)


def height_to_normal(h, texel_m, strength=1.0):
    """OpenGL tangent-space normal map from a height field (metres) whose rows run top -> bottom (v decreasing)."""
    gy, gx = np.gradient(h.astype(np.float32))
    dhdu = gx / texel_m * strength
    dhdv = -gy / texel_m * strength            # row index grows downwards = v decreasing
    n = np.stack([-dhdu, -dhdv, np.ones_like(dhdu)], axis=-1)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    return n


def encode_normal(n):
    return np.clip(np.round((n * 0.5 + 0.5) * 255), 0, 255).astype(np.uint8)


def downsample(img, factor):
    """Area average by an integer factor (2-D or 3-D arrays)."""
    if factor == 1:
        return img
    h, w = img.shape[:2]
    if img.ndim == 2:
        return img.reshape(h // factor, factor, w // factor, factor).mean(axis=(1, 3))
    return img.reshape(h // factor, factor, w // factor, factor, img.shape[2]).mean(axis=(1, 3))
