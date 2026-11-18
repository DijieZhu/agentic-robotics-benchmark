"""Small image-processing kit for the G4 readers (numpy + scipy only; no OpenCV in the dev venv).

Pixel convention: continuous coordinates (u, v), pixel (col i, row j) has its centre at (i + 0.5, j + 0.5).
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage
from scipy.spatial import ConvexHull, QhullError

LUMA = np.array([0.299, 0.587, 0.114], np.float32)


def to_gray(frame_rgb: np.ndarray) -> np.ndarray:
    """uint8 sRGB (H, W, 3) -> float32 luma in [0, 1] (computed on the encoded values, like most decoders)."""
    return (frame_rgb.astype(np.float32) @ LUMA) * np.float32(1.0 / 255.0)


def bilinear(img: np.ndarray, u, v) -> np.ndarray:
    """Sample img at continuous (u, v); outside -> nan."""
    u = np.asarray(u, np.float64) - 0.5
    v = np.asarray(v, np.float64) - 0.5
    H, W = img.shape[:2]
    ok = (u >= 0) & (v >= 0) & (u <= W - 1) & (v <= H - 1)
    uc = np.clip(u, 0, W - 1.000001)
    vc = np.clip(v, 0, H - 1.000001)
    i0 = np.floor(uc).astype(np.int64)
    j0 = np.floor(vc).astype(np.int64)
    fu = uc - i0
    fv = vc - j0
    a = img[j0, i0] * (1 - fu) * (1 - fv) + img[j0, i0 + 1] * fu * (1 - fv) \
        + img[j0 + 1, i0] * (1 - fu) * fv + img[j0 + 1, i0 + 1] * fu * fv
    return np.where(ok, a, np.nan)


def adaptive_mask(gray: np.ndarray, win: int, c: float, dark: bool = True) -> np.ndarray:
    m = ndimage.uniform_filter(gray, size=int(win), mode="nearest")
    return gray < (m - c) if dark else gray > (m + c)


def component_boxes(mask: np.ndarray, min_area: int, min_side: int, max_frac: float = 0.9):
    """Label 8-connected components; yield (label_img_slice_mask, (r0, c0)) for plausible ones."""
    lab, n = ndimage.label(mask, structure=np.ones((3, 3), bool))
    if n == 0:
        return
    areas = ndimage.sum_labels(np.ones_like(lab), lab, index=np.arange(1, n + 1))
    sl = ndimage.find_objects(lab)
    H, W = mask.shape
    for k, (s, a) in enumerate(zip(sl, areas)):
        if s is None or a < min_area:
            continue
        h = s[0].stop - s[0].start
        w = s[1].stop - s[1].start
        if max(h, w) < min_side or h > max_frac * H or w > max_frac * W:
            continue
        yield (lab[s] == k + 1), (s[0].start, s[1].start), float(a)


def region_hull_points(sub_mask: np.ndarray, origin) -> np.ndarray:
    """Corner points of the boundary pixels of a binary region (u, v), in image coordinates."""
    er = ndimage.binary_erosion(sub_mask, structure=np.ones((3, 3), bool), border_value=0)
    bnd = sub_mask & ~er
    jj, ii = np.nonzero(bnd)
    u = ii + origin[1]
    v = jj + origin[0]
    pts = np.concatenate([np.stack([u, v], 1), np.stack([u + 1, v], 1), np.stack([u, v + 1], 1),
                          np.stack([u + 1, v + 1], 1)]).astype(np.float64)
    return pts


def polygon_area(p: np.ndarray) -> float:
    x, y = p[:, 0], p[:, 1]
    return 0.5 * float(np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def hull_to_quad(pts: np.ndarray):
    """Convex hull reduced to 4 vertices (collinear pre-pass, then Visvalingam).  Returns (quad (4, 2), area ratio)."""
    if len(pts) < 8:
        return None, 0.0
    try:
        hull = ConvexHull(pts)
    except QhullError:
        return None, 0.0
    poly = pts[hull.vertices].copy()
    hull_area = float(hull.volume)
    if hull_area <= 0:
        return None, 0.0
    # drop nearly collinear vertices (turning angle < 8 deg) in one pass, keep at least 6
    if len(poly) > 12:
        a = poly - np.roll(poly, 1, axis=0)
        b = np.roll(poly, -1, axis=0) - poly
        ang = np.abs(np.arctan2(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0], (a * b).sum(1)))
        keep = ang > np.radians(8.0)
        if keep.sum() >= 6:
            poly = poly[keep]
    poly = [tuple(x) for x in poly]
    while len(poly) > 4:
        n = len(poly)
        best, bi = 1e18, -1
        for i in range(n):
            (x0, y0), (x1, y1), (x2, y2) = poly[i - 1], poly[i], poly[(i + 1) % n]
            t = abs((x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0))
            if t < best:
                best, bi = t, i
        del poly[bi]
    poly = np.asarray(poly)
    if len(poly) < 4:
        return None, 0.0
    return poly, polygon_area(poly) / hull_area


def order_quad(q: np.ndarray) -> np.ndarray:
    """Order 4 points as image TL, TR, BR, BL (clockwise on screen), TL = the corner nearest the upper-left
    diagonal direction (assumes the quad is rotated less than 45 deg in the image for the TL choice)."""
    c = q.mean(0)
    ang = np.arctan2(q[:, 1] - c[1], q[:, 0] - c[0])      # v down: increasing angle = clockwise on screen
    q = q[np.argsort(ang)]
    k = int(np.argmin(q[:, 0] + q[:, 1]))
    return np.roll(q, -k, axis=0)


def homography(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """DLT, src/dst (N >= 4, 2)."""
    A = []
    for (x, y), (u, v) in zip(src, dst):
        A.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
        A.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])
    A = np.asarray(A, np.float64)
    _, _, Vt = np.linalg.svd(A)
    Hm = Vt[-1].reshape(3, 3)
    return Hm / Hm[2, 2]


def apply_h(Hm: np.ndarray, pts) -> np.ndarray:
    p = np.asarray(pts, np.float64)
    q = np.concatenate([p, np.ones((len(p), 1))], 1) @ Hm.T
    return q[:, :2] / q[:, 2:3]


def _line_tls(pts: np.ndarray):
    c = pts.mean(0)
    _, _, Vt = np.linalg.svd(pts - c)
    d = Vt[0]
    return c, d


def _intersect(c1, d1, c2, d2):
    A = np.array([d1, -d2]).T
    if abs(np.linalg.det(A)) < 1e-9:
        return None
    t = np.linalg.solve(A, c2 - c1)
    return c1 + t[0] * d1


def refine_quad(gray: np.ndarray, quad: np.ndarray, band_px: float, inside_dark: bool, n_samples: int = 12,
                max_shift_px: float = None) -> np.ndarray:
    """Sub-pixel corners: along each side find the mid-level crossing of the profile across the edge
    (inside within band_px, outside within band_px), fit a line, intersect neighbours.  Returns the input quad
    (same object) if the refinement is not trustworthy."""
    if max_shift_px is None:
        max_shift_px = max(1.5, 0.8 * band_px)
    ctr = quad.mean(0)
    lines = []
    s = np.arange(-band_px, band_px + 1e-9, 0.25)
    ins_sel = s < -0.3 * band_px
    out_sel = s > 0.3 * band_px
    sgn = 1.0 if inside_dark else -1.0
    ts = np.linspace(0.15, 0.85, n_samples)
    for k in range(4):
        p0, p1 = quad[k], quad[(k + 1) % 4]
        t_dir = p1 - p0
        L = float(np.hypot(*t_dir))
        if L < 4:
            return quad
        t_dir = t_dir / L
        n = np.array([t_dir[1], -t_dir[0]])
        if np.dot(n, (p0 + p1) / 2 - ctr) < 0:
            n = -n                                            # outward
        base = p0[None, :] + ts[:, None] * (p1 - p0)[None, :]        # (ns, 2)
        uu = base[:, 0:1] + s[None, :] * n[0]
        vv = base[:, 1:2] + s[None, :] * n[1]
        prof = bilinear(gray, uu, vv)                                 # (ns, len(s))
        good = ~np.any(np.isnan(prof), 1)
        if good.sum() < 3:
            return quad
        prof, base = prof[good], base[good]
        lo = np.median(prof[:, ins_sel], 1)
        hi = np.median(prof[:, out_sel], 1)
        ok = (hi - lo) * sgn > 0.02
        mid = 0.5 * (lo + hi)
        sg = (prof - mid[:, None]) * sgn
        cross = (sg[:, :-1] < 0) & (sg[:, 1:] >= 0)
        edge_pts = []
        for r in np.nonzero(ok)[0]:
            idx = np.nonzero(cross[r])[0]
            if len(idx) == 0:
                continue
            j = idx[np.argmin(np.abs(s[idx]))]
            f = -sg[r, j] / (sg[r, j + 1] - sg[r, j] + 1e-12)
            edge_pts.append(base[r] + (s[j] + f * (s[j + 1] - s[j])) * n)
        if len(edge_pts) < 3:
            return quad
        lines.append(_line_tls(np.asarray(edge_pts)))
    out = []
    for k in range(4):
        c1, d1 = lines[(k - 1) % 4]
        c2, d2 = lines[k]
        p = _intersect(c1, d1, c2, d2)
        if p is None or np.hypot(*(p - quad[k])) > max_shift_px:
            return quad
        out.append(p)
    return np.asarray(out)


def convex_poly_mask(shape, poly: np.ndarray):
    """Boolean mask of pixel centres inside a convex polygon (any vertex order).  Returns (mask_sub, (r0, c0)) over
    the polygon's bounding box clipped to the image, or (None, None) if empty."""
    H, W = shape
    u0 = int(max(0, np.floor(poly[:, 0].min())))
    u1 = int(min(W, np.ceil(poly[:, 0].max()) + 1))
    v0 = int(max(0, np.floor(poly[:, 1].min())))
    v1 = int(min(H, np.ceil(poly[:, 1].max()) + 1))
    if u1 <= u0 or v1 <= v0:
        return None, None
    try:
        hull = ConvexHull(poly)
        P = poly[hull.vertices]
    except QhullError:
        return None, None
    uu, vv = np.meshgrid(np.arange(u0, u1) + 0.5, np.arange(v0, v1) + 0.5)
    inside = np.ones(uu.shape, bool)
    n = len(P)
    for k in range(n):
        a, b = P[k], P[(k + 1) % n]
        cross = (b[0] - a[0]) * (vv - a[1]) - (b[1] - a[1]) * (uu - a[0])
        inside &= cross >= 0
    if not inside.any():
        # vertex order of the hull may be the other orientation in image axes
        inside = np.ones(uu.shape, bool)
        for k in range(n):
            a, b = P[k], P[(k + 1) % n]
            cross = (b[0] - a[0]) * (vv - a[1]) - (b[1] - a[1]) * (uu - a[0])
            inside &= cross <= 0
    return inside, (v0, u0)


def otsu_threshold(x: np.ndarray, nbins: int = 128) -> float:
    x = x[np.isfinite(x)]
    if x.size == 0:
        return 0.5
    hist, edges = np.histogram(x, bins=nbins, range=(float(x.min()), float(x.max()) + 1e-6))
    w = hist.astype(np.float64)
    cent = 0.5 * (edges[1:] + edges[:-1])
    w0 = np.cumsum(w)
    w1 = w0[-1] - w0
    m0 = np.cumsum(w * cent)
    mt = m0[-1]
    with np.errstate(invalid="ignore", divide="ignore"):
        mu0 = m0 / w0
        mu1 = (mt - m0) / w1
        var = w0 * w1 * (mu0 - mu1) ** 2
    var = np.nan_to_num(var)
    return float(cent[int(np.argmax(var))])
