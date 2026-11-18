"""Bright-point (bubble) detection in camera frames and its projection onto a horizontal plane (numpy + scipy only).
Port of proto/g3_plume/detect.py (camera part, sha256 5d88279a...) with the rays built from `pose.Cam`.

detect_spots: white top-hat -> local maxima above k_sigma x robust sigma -> Hessian blob test (a grating bar or an
edge has one eigenvalue near zero) -> no large-scale gradient at the centre (an aliased edge step sits on a strong
edge).  moving_spots: two frames from the same pose dt apart; a spot of the second frame is kept only if the first
frame's top-hat is below half its threshold around it (bubbles rise, the structure and the vehicle's own gripper do
not; this also removes the one-pixel silhouette highlights of lit thin bars, G3 section 1.3).

    u, v, amp, thr = detect_spots(rgb)
    u, v = moving_spots(rgb1, rgb2, exclude=jaw_mask)
    xy, t = project_to_plane(cam, u, v, z_plane)       # L / world coordinates of the plane hits, ray lengths
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi


def detect_spots(rgb, k_sigma=6.0, min_delta=10.0, size=7, sigma=1.0, iso=4.0, edge_sigma=2.5, edge_ratio=0.25,
                 return_map=False):
    g = rgb.astype(np.float32).mean(2)
    th = g - ndi.grey_opening(g, size=(size, size))
    med = np.median(th)
    sig = 1.4826 * np.median(np.abs(th - med)) + 1e-3
    thr = max(k_sigma * sig, min_delta)
    peak = (th == ndi.maximum_filter(th, 5)) & (th > thr)
    v, u = np.nonzero(peak)
    if len(u):
        gs = ndi.gaussian_filter(g, sigma)
        Ixx = ndi.correlate1d(ndi.correlate1d(gs, [1, -2, 1], 1), [1, 2, 1], 0) / 4.0
        Iyy = ndi.correlate1d(ndi.correlate1d(gs, [1, -2, 1], 0), [1, 2, 1], 1) / 4.0
        Ixy = ndi.correlate1d(ndi.correlate1d(gs, [-1, 0, 1], 0), [-1, 0, 1], 1) / 4.0
        a, b, c = Ixx[v, u], Iyy[v, u], Ixy[v, u]
        tr = 0.5 * (a + b)
        dd = np.sqrt(np.maximum(0.25 * (a - b) ** 2 + c ** 2, 0.0))
        l1, l2 = tr - dd, tr + dd
        ok = (l2 < 0) & (l1 >= iso * l2)
        ge = ndi.gaussian_filter(g, edge_sigma)
        gy = ndi.sobel(ge, 0) / 8.0
        gx = ndi.sobel(ge, 1) / 8.0
        gm = np.hypot(gx[v, u], gy[v, u])
        ok &= gm < edge_ratio * th[v, u]
        v, u = v[ok], u[ok]
    if return_map:
        return u.astype(float) + 0.5, v.astype(float) + 0.5, th[v, u], thr, th
    return u.astype(float) + 0.5, v.astype(float) + 0.5, th[v, u], thr


def moving_spots(rgb1, rgb2, exclude=None, det_kw=None):
    """Spots of rgb2 that were not there in rgb1 (same pose, ~0.2 s earlier); `exclude` = a boolean mask of pixels
    to ignore (e.g. blade.jaw_mask of the own gripper)."""
    _, _, _, thr1, th1 = detect_spots(rgb1, return_map=True, **(det_kw or {}))
    u2, v2, a2, thr = detect_spots(rgb2, **(det_kw or {}))
    if len(u2):
        m1 = ndi.maximum_filter(th1, 3)
        keep = m1[v2.astype(int), u2.astype(int)] < 0.5 * thr1
        if exclude is not None:
            keep &= ~exclude[v2.astype(int), u2.astype(int)]
        u2, v2 = u2[keep], v2[keep]
    return u2, v2


def project_to_plane(cam, u, v, z_plane: float):
    """Hits of the pixel rays (cam = pose.Cam, L / world axes) with the horizontal plane z = z_plane: (xy (N, 2),
    ray length t (N,), NaN where the ray does not go down to the plane)."""
    d = cam.rays(u, v)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (z_plane - cam.p[2]) / np.where(np.abs(d[:, 2]) > 1e-6, d[:, 2], np.nan)
    pts = cam.p[None, :] + t[:, None] * d
    return pts[:, :2], t
