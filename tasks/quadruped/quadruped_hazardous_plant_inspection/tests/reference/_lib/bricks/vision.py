"""Classical image helpers shared by the reference readers (OpenCV + numpy only, no learned models).

Conventions: images are RGB uint8 (H, W, 3); `gray01` gives float32 in [0, 1]; pixel coordinates are OpenCV's (x right,
y down, pixel centres at integers). Ellipses are (cx, cy, a, b, ang) with semi-axes a >= b and `ang` the direction of
the a-axis in radians in pixel coordinates (y down). Angles on a rectified dial are counter-clockwise as the viewer
sees it (the y-up convention), so they match the targets' face frame.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

__all__ = ["gray01", "hsv01", "find_bright_ellipses", "polar_sample", "find_bright_quads", "order_quad", "refine_quad",
           "warp_quad", "circ_smooth", "longest_run", "peaks_1d", "center_weight", "largest_component",
           "subpixel_peak", "highpass_1d", "scale_comb", "comb_index", "flatten_plate_image"]


def gray01(rgb: np.ndarray) -> np.ndarray:
    if rgb.ndim == 2:
        g = rgb.astype(np.float32)
    else:
        g = cv2.cvtColor(np.ascontiguousarray(rgb[..., :3]), cv2.COLOR_RGB2GRAY).astype(np.float32)
    return g / 255.0


def hsv01(rgb: np.ndarray):
    """(h in degrees [0, 360), s in [0, 1], v in [0, 1]) float32 arrays."""
    hsv = cv2.cvtColor(np.ascontiguousarray(rgb[..., :3]), cv2.COLOR_RGB2HSV_FULL).astype(np.float32)
    return hsv[..., 0] * (360.0 / 256.0), hsv[..., 1] / 255.0, hsv[..., 2] / 255.0


def center_weight(x, y, w, h, sigma=0.35) -> float:
    """1 at the image centre, falling off with the distance from it (in units of the half diagonal)."""
    d = math.hypot(x - (w - 1) / 2.0, y - (h - 1) / 2.0) / (0.5 * math.hypot(w, h))
    return math.exp(-(d / sigma) ** 2)


def _otsu(img8):
    t, _ = cv2.threshold(img8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return float(t)


def find_bright_ellipses(gray: np.ndarray, min_semi_axis_px: float = 5.0, min_axis_ratio: float = 0.22,
                         max_candidates: int = 6, centre_sigma: float = 0.35, work_width: int = 800,
                         hull: bool = False, hi_fracs=(), islands: bool = False):
    """Candidate bright elliptical regions (a dial, a lamp lens) with their scores, best first.

    The image is thresholded at a few Otsu levels (whole image, central crop), dark marks inside a bright region are
    closed at several kernel sizes, and each external contour is fitted with an ellipse; a candidate must fill its
    ellipse and follow it closely. Returns [(ellipse, score)], ellipse in full-resolution pixel coordinates.
    hull=True (2026-11-14, the gauge reader's second try): the ellipse is fitted to the contour's convex hull
    and must follow the hull closely, so the notches that the needle and the digits at the rim cut into the face's
    contour do not reject it (a closing large enough to fill them joins the face to a bright wall over a bezel that
    looks thin from below). hi_fracs (same pass): extra thresholds at these fractions of the central crop's 99th
    percentile; a white face inside a dark bezel on a light wall is a hole of the wall's region at every lower
    threshold (only external contours are fitted), and becomes a region of its own between the wall's and the face's
    brightness. islands=True (2026-11-16): the outer contour of every bright region is fitted,
    also of a region lying inside a hole of another (a grey dial face inside its dark bezel on a dressed wall brighter
    than the face: the face is an island of the wall's region at every threshold, which only external contours never
    see)."""
    H, W = gray.shape
    s = min(1.0, float(work_width) / max(W, 1))
    g = gray if s >= 1.0 else cv2.resize(gray, (int(round(W * s)), int(round(H * s))), interpolation=cv2.INTER_AREA)
    h, w = g.shape
    g8 = np.clip(g * 255.0, 0, 255).astype(np.uint8)
    g8 = cv2.GaussianBlur(g8, (0, 0), 0.8)
    thr = {_otsu(g8)}
    cy0, cy1, cx0, cx1 = h // 4, 3 * h // 4, w // 4, 3 * w // 4
    tc = _otsu(np.ascontiguousarray(g8[cy0:cy1, cx0:cx1]))
    thr.add(tc)
    hi = float(np.percentile(g8[cy0:cy1, cx0:cx1], 99))
    thr.add(0.5 * (tc + hi))
    thr.add(max(8.0, 0.6 * tc))
    for f in hi_fracs:
        if f * hi > tc:
            thr.add(float(f) * hi)
    min_ax = max(2.0, min_semi_axis_px * s)
    cands = []
    for t in sorted(thr):
        bw = (g8 > t).astype(np.uint8)
        for k in (0, 3, 5, 9, 15, 25):
            if k:
                m = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
            else:
                m = bw
            if islands:
                cnts, hier = cv2.findContours(m, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
                cnts = [c for c, h_ in zip(cnts, hier[0] if hier is not None else []) if h_[3] < 0]
            else:
                cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            for c in cnts:
                if len(c) < 10:
                    continue
                area = cv2.contourArea(c)
                if area < math.pi * min_ax * min_ax * min_axis_ratio:
                    continue
                if area > 0.6 * h * w:
                    continue
                if hull:
                    c = cv2.convexHull(c)
                    if len(c) < 6:
                        continue
                (ex, ey), (ew, eh), ang = cv2.fitEllipse(c)
                a, b = ew / 2.0, eh / 2.0
                ang_r = math.radians(ang)
                if b > a:
                    a, b = b, a
                    ang_r += math.pi / 2
                if b < min_ax or b / max(a, 1e-6) < min_axis_ratio:
                    continue
                fill = area / (math.pi * a * b)
                if not 0.82 < fill < 1.08:
                    continue
                p = c[:, 0, :].astype(np.float64) - (ex, ey)
                ca, sa = math.cos(ang_r), math.sin(ang_r)
                u = (p[:, 0] * ca + p[:, 1] * sa) / a
                v = (-p[:, 0] * sa + p[:, 1] * ca) / b
                resid = float(np.mean(np.abs(np.sqrt(u * u + v * v) - 1.0)))
                if resid > 0.07:
                    continue
                # touching the image border: the region may be cut
                x, y, bw_, bh_ = cv2.boundingRect(c)
                border = (x <= 1 or y <= 1 or x + bw_ >= w - 1 or y + bh_ >= h - 1)
                score = (1.0 - resid / 0.07) * (1.0 - abs(1.0 - fill) * 3) * center_weight(ex, ey, w, h, centre_sigma)
                score *= math.sqrt(a * b) ** 0.5 * (0.3 if border else 1.0)
                cands.append(((ex / s, ey / s, a / s, b / s, ang_r), score))
    # de-duplicate: keep the best of candidates with nearly the same centre and size
    cands.sort(key=lambda e: -e[1])
    out = []
    for e, sc in cands:
        dup = False
        for f, _ in out:
            if (math.hypot(e[0] - f[0], e[1] - f[1]) < 0.15 * f[2] and abs(e[2] - f[2]) < 0.12 * f[2]
                    and abs(e[3] - f[3]) < 0.15 * f[3]):
                dup = True
                break
        if not dup:
            out.append((e, sc))
        if len(out) >= max_candidates:
            break
    return out


def polar_sample(gray: np.ndarray, ell, n_theta: int = 720, rho_max: float = 1.0, n_rho: int = 100,
                 rho_min: float = 0.0):
    """Sample `gray` on the affinely rectified ellipse: out[i, j] at dial angle theta_i = 2 pi i / n_theta
    (counter-clockwise as the viewer sees it, from the ellipse's a-axis) and relative radius rho_j. Returns
    (samples (n_theta, n_rho) float32, theta (n_theta,), rho (n_rho,))."""
    cx, cy, a, b, ang = ell
    th = np.arange(n_theta) * (2 * math.pi / n_theta)
    rho = np.linspace(rho_min, rho_max, n_rho)
    ca, sa = math.cos(ang), math.sin(ang)
    T, Rr = np.meshgrid(th, rho, indexing="ij")
    ex, ey = a * np.cos(T) * Rr, b * np.sin(T) * Rr
    # a-axis u = (ca, sa); the axis 90 deg counter-clockwise as the viewer sees it is (sa, -ca) in y-down pixels
    mx = (cx + ex * ca + ey * sa).astype(np.float32)
    my = (cy + ex * sa - ey * ca).astype(np.float32)
    out = cv2.remap(gray.astype(np.float32), mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return out, th, rho


def order_quad(pts) -> np.ndarray:
    """Order 4 points as top-left, top-right, bottom-right, bottom-left (pixel coordinates, y down)."""
    p = np.asarray(pts, np.float64).reshape(4, 2)
    c = p.mean(0)
    ang = np.arctan2(p[:, 1] - c[1], p[:, 0] - c[0])
    p = p[np.argsort(ang)]          # clockwise on screen starting from about -180 deg (left)
    # rotate so that the first is the top-left (smallest x + y)
    k = int(np.argmin(p[:, 0] + p[:, 1]))
    p = np.roll(p, -k, axis=0)
    return p.astype(np.float32)


def find_bright_quads(gray: np.ndarray, min_area_px: float = 120.0, aspect=(0.1, 10.0), max_candidates: int = 6,
                      centre_sigma: float = 0.35, work_width: int = 800, closes=(0, 3, 7, 13), hi_fracs=()):
    """Candidate bright quadrilateral regions (a scale plate, a legend plate) best first: [(quad 4x2, score)].
    `aspect` bounds the plate's side ratio (long / short is not assumed; width / height as seen). `hi_fracs` adds
    thresholds at these fractions of the central crop's 99th percentile (separates a white plate from lighter-grey
    parts that touch it, e.g. under a head lamp in the dark)."""
    H, W = gray.shape
    s = min(1.0, float(work_width) / max(W, 1))
    g = gray if s >= 1.0 else cv2.resize(gray, (int(round(W * s)), int(round(H * s))), interpolation=cv2.INTER_AREA)
    h, w = g.shape
    g8 = cv2.GaussianBlur(np.clip(g * 255.0, 0, 255).astype(np.uint8), (0, 0), 0.8)
    cy0, cy1, cx0, cx1 = h // 4, 3 * h // 4, w // 4, 3 * w // 4
    tc = _otsu(np.ascontiguousarray(g8[cy0:cy1, cx0:cx1]))
    hi = float(np.percentile(g8[cy0:cy1, cx0:cx1], 99))
    thr = {_otsu(g8), tc, 0.5 * (tc + hi)}
    for f in hi_fracs:
        if f * hi > tc:
            thr.add(float(f) * hi)
    cands = []
    for t in sorted(thr):
        bw = (g8 > t).astype(np.uint8)
        for k in closes:
            m = bw if not k else cv2.morphologyEx(bw, cv2.MORPH_CLOSE,
                                                  cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
            cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in cnts:
                area = cv2.contourArea(c)
                if area < min_area_px * s * s or area > 0.7 * h * w:
                    continue
                hull = cv2.convexHull(c)
                peri = cv2.arcLength(hull, True)
                q = None
                for eps in (0.02, 0.035, 0.05, 0.08):
                    ap = cv2.approxPolyDP(hull, eps * peri, True)
                    if len(ap) == 4:
                        q = ap[:, 0, :].astype(np.float64)
                        break
                if q is None:
                    rect = cv2.minAreaRect(c)
                    q = cv2.boxPoints(rect).astype(np.float64)
                q = refine_quad(c[:, 0, :].astype(np.float64), order_quad(q))
                qa = cv2.contourArea(q.astype(np.float32))
                if qa <= 0:
                    continue
                fill = area / qa
                if fill < 0.85 or fill > 1.1:
                    continue
                oq = order_quad(q)
                wd = 0.5 * (np.linalg.norm(oq[1] - oq[0]) + np.linalg.norm(oq[2] - oq[3]))
                ht = 0.5 * (np.linalg.norm(oq[3] - oq[0]) + np.linalg.norm(oq[2] - oq[1]))
                if ht < 3 or wd < 3:
                    continue
                r = wd / ht
                if not aspect[0] <= r <= aspect[1]:
                    continue
                x, y, bw_, bh_ = cv2.boundingRect(c)
                border = (x <= 1 or y <= 1 or x + bw_ >= w - 1 or y + bh_ >= h - 1)
                mx, my = oq.mean(0)
                score = (1.0 - abs(1.0 - fill) * 3) * center_weight(mx, my, w, h, centre_sigma) * area ** 0.25
                score *= 0.3 if border else 1.0
                cands.append((oq / s, score))
    cands.sort(key=lambda e: -e[1])
    out = []
    for q, sc in cands:
        if any(np.max(np.linalg.norm(q - p, axis=1)) < 0.08 * max(1.0, np.linalg.norm(p[2] - p[0])) for p, _ in out):
            continue
        out.append((q, sc))
        if len(out) >= max_candidates:
            break
    return out


def flatten_plate_image(gray: np.ndarray, sigma_frac: float = 0.12, close_frac: float = 0.012) -> np.ndarray:
    """A copy of `gray` for finding printed plates under uneven light: thin dark marks (ticks, digits) closed with a
    vertical element (close_frac x the image height), then divided by a broad blur (sigma_frac x the image height) and
    rescaled to [0, 1]. A plate lit with a gradient becomes evenly bright, so one threshold separates all of it."""
    H, W = gray.shape
    g = np.asarray(gray, np.float32)
    k = max(3, int(round(close_frac * H)) | 1)
    g = cv2.morphologyEx(g, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (1, k)))
    sig = max(3.0, sigma_frac * H)
    bg = cv2.GaussianBlur(g, (0, 0), sig)
    f = g / np.maximum(bg, 1e-3)
    lo, hi = np.percentile(f, 1), np.percentile(f, 99.5)
    return np.clip((f - lo) / max(hi - lo, 1e-6), 0.0, 1.0).astype(np.float32)


def refine_quad(pts: np.ndarray, quad: np.ndarray) -> np.ndarray:
    """Refine a quad's corners: fit a line to the contour points along the middle 70 % of each side and intersect
    neighbouring lines (robust to a rounded or partly occluded corner). Falls back to the input quad."""
    q = np.asarray(quad, np.float64)
    lines = []
    for i in range(4):
        a, b = q[i], q[(i + 1) % 4]
        d = b - a
        L = np.linalg.norm(d)
        if L < 4:
            return q.astype(np.float32)
        u = d / L
        nrm = np.array([-u[1], u[0]])
        t = (pts - a) @ u
        e = (pts - a) @ nrm
        sel = (t > 0.15 * L) & (t < 0.85 * L) & (np.abs(e) < max(2.0, 0.06 * L))
        if sel.sum() < 4:
            return q.astype(np.float32)
        P = pts[sel]
        m = P.mean(0)
        w, v = np.linalg.eigh(np.cov(P.T))
        lines.append((m, v[:, 1]))
    out = []
    for i in range(4):
        (p1, d1), (p2, d2) = lines[(i - 1) % 4], lines[i]
        A = np.column_stack([d1, -d2])
        if abs(np.linalg.det(A)) < 1e-6:
            return q.astype(np.float32)
        s_, _ = np.linalg.solve(A, p2 - p1)
        out.append(p1 + s_ * d1)
    out = np.array(out)
    if np.max(np.linalg.norm(out - q, axis=1)) > 0.25 * max(np.linalg.norm(q[2] - q[0]), 1.0):
        return q.astype(np.float32)
    return out.astype(np.float32)


def refine_plate_sides(gray: np.ndarray, quad, search: float = 0.45, min_rows: int = 60, max_rows: int = 400):
    """A plate candidate with its two long sides re-found in its own rectified frame (2026-11-14).

    The bright-region quads of find_bright_quads take a side's line from the region's contour; where something
    touches or crosses the plate's edge (the indicator's pointer arm, a level gauge's light-grey tube backing beside
    the scale plate) the contour bulges and the side comes out tilted or too far out, and the tick band of the
    rectified plate then misses the tick roots. Here, in the quad's rectified frame (about one pixel per image
    pixel), every row's strongest dark-to-bright step within `search` x the quad's width of the left side (and
    bright-to-dark of the right side) is taken, a straight line is fitted to those steps robustly (median start,
    then trimmed least squares; rows where the step is weak or off the line are dropped, e.g. under the pointer
    arm or at the ticks), and the two lines give the new corners at the quad's top and bottom rows. Returns the
    refined quad (TL, TR, BR, BL; float32) or None when a side has no line (fewer than 30 % of the rows agree)."""
    q = np.asarray(quad, np.float64)
    wq = 0.5 * (np.linalg.norm(q[1] - q[0]) + np.linalg.norm(q[2] - q[3]))
    hq = 0.5 * (np.linalg.norm(q[3] - q[0]) + np.linalg.norm(q[2] - q[1]))
    if wq < 6 or hq < 12:
        return None
    n_rows = int(np.clip(round(hq), min_rows, max_rows))
    W = float(np.clip(round(wq * n_rows / max(hq, 1.0)), 12, 400))
    ml = mr = search + 0.15
    out_w = int(round(W * (1 + ml + mr)))
    rect, Hm = warp_quad(np.asarray(gray, np.float32), q, out_w, n_rows, margin=(ml, 0.0, mr, 0.0))
    g = cv2.GaussianBlur(rect, (0, 0), 1.0)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    rows_all = np.arange(n_rows, dtype=np.float64)
    lines = []
    for x0, sgn in ((ml * W, 1.0), (ml * W + W, -1.0)):
        a, b = int(max(1, x0 - search * W)), int(min(out_w - 2, x0 + search * W))
        if b - a < 3:
            return None
        band = sgn * gx[:, a:b]
        xs = np.argmax(band, 1)
        st = band[np.arange(n_rows), xs]
        ok = st > 0.25 * max(float(np.percentile(st, 90)), 1e-6)
        if ok.sum() < 0.3 * n_rows:
            return None
        rows, xx = rows_all[ok], (xs[ok] + a).astype(np.float64)
        step = max(1, len(rows) // 40)
        dr = np.diff(rows[::step])
        sl = float(np.median(np.diff(xx[::step]) / np.maximum(dr, 1e-6))) if len(dr) >= 3 else 0.0
        ic = float(np.median(xx - sl * rows))
        keep = None
        for _ in range(3):
            r = xx - (ic + sl * rows)
            keep = np.abs(r) < max(1.5, 2.5 * float(np.median(np.abs(r))))
            if keep.sum() < 0.3 * n_rows:
                return None
            sl, ic = (float(v) for v in np.polyfit(rows[keep], xx[keep], 1))
        lines.append((ic, sl))
    Hinv = np.linalg.inv(Hm)

    def img(x, y):
        p = Hinv @ np.array([x, y, 1.0])
        return p[:2] / p[2]

    (il, sl_l), (ir, sl_r) = lines
    y1 = float(n_rows)                       # the quad's bottom side in the rectified frame
    if not (ir - il > 0.3 * W and (ir + sl_r * y1) - (il + sl_l * y1) > 0.3 * W):
        return None
    out = np.array([img(il, 0.0), img(ir, 0.0), img(ir + sl_r * y1, y1), img(il + sl_l * y1, y1)])
    return out.astype(np.float32)


def tick_plate_quads(gray: np.ndarray, major_frac: float, end_margin: float, k: int = 7, max_lines: int = 3,
                     min_ticks: int = 12, max_tilt_deg: float = 35.0):
    """Scale-plate candidates found from the printed ticks instead of the plate's brightness (2026-11-14).

    A plate as bright as the wall behind it (a level gauge's plate against a light wall, an indicator plate seen at
    40-45 deg against a light wall) gives no bright quad at any threshold. Its ticks still show: thin dark marks,
    near horizontal (within max_tilt_deg), every root on the plate's left edge (both catalogue plates print the
    ticks from the left edge to the right). Thin dark marks are taken from a black top-hat with a vertical element of
    k px; a line through at least `min_ticks` mark roots (left ends; consecutive pairs of roots, RANSAC-style, then a
    least-squares fit of the inliers) is a scale's root line; the plate's width in the image is the long (major)
    ticks' length over `major_frac` (their catalogue length in plate widths), and the plate extends `end_margin` x
    the root span beyond the end roots. Returns [(quad TL, TR, BR, BL float32, score)], best first; the reader checks
    each candidate against the catalogue exactly as a bright quad."""
    g = np.asarray(gray, np.float32)
    H, W = g.shape
    el = cv2.getStructuringElement(cv2.MORPH_RECT, (1, int(k) | 1))
    th = cv2.morphologyEx(g, cv2.MORPH_CLOSE, el, borderType=cv2.BORDER_REPLICATE) - g
    top = float(np.percentile(th, 99.5))
    if top < 0.04:
        return []
    bw = (th > max(0.04, 0.35 * top)).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
    if n < min_ticks + 1:
        return []
    ys, xs = np.nonzero(lab)
    ll = lab[ys, xs]
    order = np.argsort(ll, kind="stable")
    ys, xs, ll = ys[order].astype(np.float64), xs[order].astype(np.float64), ll[order]
    cnt = np.bincount(ll, minlength=n).astype(np.float64)
    cnt[0] = 1.0
    mx = np.bincount(ll, xs, n) / cnt
    my = np.bincount(ll, ys, n) / cnt
    cxx = np.bincount(ll, xs * xs, n) / cnt - mx * mx
    cyy = np.bincount(ll, ys * ys, n) / cnt - my * my
    cxy = np.bincount(ll, xs * ys, n) / cnt - mx * my
    tr, det = cxx + cyy, cxx * cyy - cxy * cxy
    disc = np.sqrt(np.maximum(0.25 * tr * tr - det, 0.0))
    l1, l0 = 0.5 * tr + disc, np.maximum(0.5 * tr - disc, 1e-3)
    ang = 0.5 * np.arctan2(2 * cxy, cxx - cyy)                     # principal axis (radians, image frame)
    ux, uy = np.cos(ang), np.sin(ang)
    # endpoints along the principal axis
    proj = (xs - mx[ll]) * ux[ll] + (ys - my[ll]) * uy[ll]
    starts = np.searchsorted(ll, np.arange(n))
    pmin = np.full(n, np.nan)
    pmax = np.full(n, np.nan)
    nz = np.nonzero(np.bincount(ll, minlength=n)[1:])[0] + 1
    pmin[nz] = np.minimum.reduceat(proj, starts[nz])
    pmax[nz] = np.maximum.reduceat(proj, starts[nz])
    length = pmax - pmin + 1.0
    area = stats[:, cv2.CC_STAT_AREA].astype(np.float64)
    tilt = np.degrees(np.abs(np.arctan2(uy, ux)))
    tilt = np.minimum(tilt, 180.0 - tilt)
    good = ((area >= 3) & (length >= 3.0) & (np.sqrt(l1 / l0) >= 2.5) & (tilt <= max_tilt_deg)
            & (2.0 * np.sqrt(l0) <= k))
    good[0] = False
    idx = np.nonzero(good)[0]
    if len(idx) < min_ticks:
        return []
    if len(idx) > 160:
        idx = idx[np.argsort(-length[idx])[:160]]
    # oriented to +x: the root is the left end (the principal angle lies in (-90, 90] deg, so ux >= 0 in practice)
    sgn = np.where(ux[idx] >= 0, 1.0, -1.0)
    vx, vy = ux[idx] * sgn, uy[idx] * sgn
    p_root = np.where(sgn > 0, pmin[idx], pmax[idx])
    rx, ry = mx[idx] + p_root * ux[idx], my[idx] + p_root * uy[idx]
    L = length[idx]
    R = np.column_stack([rx, ry])
    alive = np.ones(len(idx), bool)
    out = []
    for _line in range(max_lines):
        cand = np.nonzero(alive)[0]
        if len(cand) < min_ticks:
            break
        srt = cand[np.argsort(R[cand, 1])]
        best = None
        for a_i in range(len(srt)):
            for step in (1, 2, 3):
                if a_i + step >= len(srt):
                    break
                i, j = srt[a_i], srt[a_i + step]
                d = R[j] - R[i]
                dl = float(np.hypot(*d))
                if dl < 2.0:
                    continue
                u = d / dl
                vm = np.array([vx[i] + vx[j], vy[i] + vy[j]])
                vm /= max(float(np.hypot(*vm)), 1e-9)
                if abs(float(u @ vm)) > 0.6:                      # roots run across the ticks
                    continue
                nrm = np.array([-u[1], u[0]])
                dist = np.abs((R[cand] - R[i]) @ nrm)
                par = np.abs(vx[cand] * vm[0] + vy[cand] * vm[1]) > math.cos(math.radians(20.0))
                inl = cand[(dist <= 1.5 + 0.02 * L[cand]) & par]
                if best is None or len(inl) > len(best[0]):
                    best = (inl, u)
        if best is None or len(best[0]) < min_ticks:
            break
        inl, _ = best
        P = R[inl]
        m = P.mean(0)
        ev_, evec = np.linalg.eigh(np.cov(P.T))
        u = evec[:, 1] if evec[1, 1] >= 0 else -evec[:, 1]          # along the scale, downwards in the image
        t = (P - m) @ u
        span = float(t.max() - t.min())
        if span < 8.0:
            alive[inl] = False
            continue
        v = np.array([vx[inl].mean(), vy[inl].mean()])
        v /= max(float(np.hypot(*v)), 1e-9)
        w_img = float(np.percentile(L[inl], 85)) / max(major_frac, 1e-3)
        ts = np.sort(t)
        # the whole root line, and the line without its two outermost roots at each end (a mark beyond the scale's
        # end, e.g. a fitting's edge or the unit sign, would put the plate's end outside the image; the reader's
        # comb search extends beyond the candidate anyway)
        spans = [(ts[0], ts[-1])] + ([(ts[2], ts[-3])] if len(ts) >= min_ticks + 4 else [])
        for t_lo, t_hi in spans:
            sp = float(t_hi - t_lo)
            c0, c1 = m + u * (t_lo - end_margin * sp), m + u * (t_hi + end_margin * sp)
            quad = np.array([c0, c0 + w_img * v, c1 + w_img * v, c1], np.float32)
            if (quad[:, 0].min() >= -0.5 * w_img and quad[:, 0].max() <= W + 0.5 * w_img
                    and quad[:, 1].min() >= -0.2 * sp and quad[:, 1].max() <= H + 0.2 * sp):
                out.append((quad, float(len(inl)) - (0.5 if t_lo != ts[0] else 0.0)))
        alive[inl] = False
    out.sort(key=lambda e: -e[1])
    return out


def rect_points_in_image(Hm, pts, shape, border: float = 1.0) -> bool:
    """Every point (x, y) of a rectified frame (warp_quad's homography Hm) maps inside the image of `shape`
    (2026-11-14: a scale's end ticks must have been seen, not found in the border's replicated pixels)."""
    Hinv = np.linalg.inv(np.asarray(Hm, np.float64))
    H, W = shape[:2]
    for x, y in pts:
        p = Hinv @ np.array([float(x), float(y), 1.0])
        if abs(p[2]) < 1e-12:
            return False
        u, v = p[0] / p[2], p[1] / p[2]
        if not (border <= u <= W - 1 - border and border <= v <= H - 1 - border):
            return False
    return True


def warp_quad(img: np.ndarray, quad, out_w: int, out_h: int, margin=(0.0, 0.0, 0.0, 0.0)):
    """Rectify the quadrilateral `quad` (TL, TR, BR, BL) to an out_w x out_h image. `margin` = (left, top, right,
    bottom) extends the sampled region beyond the quad in units of its width / height. Returns (warped, H)."""
    l, t, r, b = margin
    W = out_w / (1.0 + l + r)
    Hh = out_h / (1.0 + t + b)
    dst = np.float32([[l * W, t * Hh], [l * W + W, t * Hh], [l * W + W, t * Hh + Hh], [l * W, t * Hh + Hh]])
    Hm = cv2.getPerspectiveTransform(np.float32(quad), dst)
    out = cv2.warpPerspective(img, Hm, (int(out_w), int(out_h)), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REPLICATE)
    return out, Hm


def circ_smooth(x: np.ndarray, half: int) -> np.ndarray:
    """Circular box smoothing with a window of 2 * half + 1 samples."""
    if half <= 0:
        return x.copy()
    k = 2 * half + 1
    xp = np.concatenate([x[-half:], x, x[:half]])
    c = np.cumsum(np.concatenate([[0.0], xp]))
    return (c[k:] - c[:-k]) / k


def longest_run(mask: np.ndarray, circular: bool = True):
    """(start, length) of the longest run of True in a 1D mask (circular if asked); (None, 0) when none."""
    n = len(mask)
    if not mask.any():
        return None, 0
    if mask.all():
        return 0, n
    m = np.concatenate([mask, mask]) if circular else mask
    best, best_s, cur, cur_s = 0, None, 0, 0
    for i, v in enumerate(m):
        if v:
            if cur == 0:
                cur_s = i
            cur += 1
            if cur > best:
                best, best_s = cur, cur_s
        else:
            cur = 0
    if circular:
        best = min(best, n)
        best_s = best_s % n
    return best_s, best


def peaks_1d(x: np.ndarray, min_height: float, min_dist: int = 2):
    """Indices of local maxima of x above min_height, at least min_dist apart (greedy by height)."""
    idx = np.where((x[1:-1] >= x[:-2]) & (x[1:-1] > x[2:]) & (x[1:-1] >= min_height))[0] + 1
    idx = idx[np.argsort(-x[idx])]
    keep = []
    for i in idx:
        if all(abs(int(i) - j) >= min_dist for j in keep):
            keep.append(int(i))
    return sorted(keep)


def subpixel_peak(y: np.ndarray, i: int, circular: bool = False) -> float:
    n = len(y)
    if circular:
        a, b, c = y[(i - 1) % n], y[i], y[(i + 1) % n]
    else:
        if i <= 0 or i >= n - 1:
            return float(i)
        a, b, c = y[i - 1], y[i], y[i + 1]
    den = a - 2 * b + c
    if abs(den) < 1e-12:
        return float(i)
    return float(i) + float(np.clip(0.5 * (a - c) / den, -0.5, 0.5))


def highpass_1d(x: np.ndarray, win: int) -> np.ndarray:
    """Black top-hat of a 1D brightness profile: closing(x) - x with a flat element of `win` samples. Keeps dark
    features thinner than `win` (tick marks) as positive peaks and gives about 0 on a dark plateau wider than it (the
    background beyond a plate), unlike a darkness profile, which is high on such a plateau."""
    x = np.ascontiguousarray(np.asarray(x, np.float32).reshape(-1, 1))
    win = max(3, int(win) | 1)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (1, win))
    closed = cv2.morphologyEx(x, cv2.MORPH_CLOSE, k, borderType=cv2.BORDER_REPLICATE)
    return (closed - x).ravel().astype(np.float64)


def _comb_fit(ks, rows):
    """Least-squares 1D projective map row(k) = (a k + b) / (c k + 1) (perspective along a straight scale; c = 0 is
    the affine case). Returns (a, b, c, rms residual)."""
    ks = np.asarray(ks, np.float64)
    rows = np.asarray(rows, np.float64)
    if len(ks) >= 5:
        A = np.column_stack([ks, np.ones_like(ks), -ks * rows])
        sol, *_ = np.linalg.lstsq(A, rows, rcond=None)
        a, b, c = (float(v) for v in sol)
        if abs(c) * float(np.max(np.abs(ks))) > 0.5:          # degenerate: fall back to the affine fit
            c = 0.0
            a, b = (float(v) for v in np.polyfit(ks, rows, 1))
    else:
        a, b = (float(v) for v in np.polyfit(ks, rows, 1))
        c = 0.0
    pred = (a * ks + b) / (c * ks + 1.0)
    return a, b, c, float(np.sqrt(np.mean((pred - rows) ** 2)))


def comb_index(comb: dict, row: float) -> float:
    """Fractional tick index (0 = the comb's first tick in profile order) of a profile position, through the comb's
    projective fit."""
    a, b, c = comb["fit"]
    den = a - c * row
    if abs(den) < 1e-12:
        return float("nan")
    return float((row - b) / den)


def scale_comb(prof: np.ndarray, n_ticks: int, major_every: int, major_prof: np.ndarray | None = None,
               min_pitch: float = 3.0, max_pitch: float | None = None, tol: float = 0.28, max_gap: int = 2,
               peak_frac: float = 0.25, end_slack: int = 0):
    """Find a complete, regular scale of `n_ticks` tick marks in a 1D darkness profile (rows of a rectified scale
    plate, the region around the plate included) and check it against the printed scale's convention.

    prof        darkness of the tick roots along the scale axis (high = dark), e.g. high-passed
    major_prof  darkness of the band that only the major (long) ticks reach; majors are every `major_every`-th tick
                and both end ticks are majors (the catalogue's scale layout)
    Returns None when no comb of the right length exists, else a dict:
      rows (n_ticks fitted positions, profile order), fit (a, b, c of row(k)), pitch (mean spacing), matched (ticks
      found / n_ticks), resid (rms residual / pitch), major_sep (major vs minor darkness separation, 0..1),
      major_ok (end ticks major and the alternation right), quality (0..1).
    A comb is a chain of peaks each within tol x pitch of the position predicted from the ticks already chained;
    up to max_gap consecutive ticks may be missing inside (a pointer or a parallax shadow over the tick roots). A
    chain that spans fewer than n_ticks is no comb (an end of the scale is not seen: the plate is cut, hidden or
    truncated); one that spans more is cut to the n_ticks window whose major pattern fits best. With end_slack > 0
    the window may extend up to that many ticks beyond the chain (end ticks hidden, e.g. under a pointer) when the
    major pattern anchors it (use it only when the majors are sparse enough to fix the index: major_every >= 3); the
    result's 'present' tells which ticks were seen."""
    prof = np.asarray(prof, np.float64)
    n = len(prof)
    if n < 3 * n_ticks:
        return None
    top = float(prof.max())
    if top <= 1e-6:
        return None
    pk = peaks_1d(prof, peak_frac * top, min_dist=2)
    if len(pk) < max(4, int(0.6 * n_ticks)):
        return None
    P = np.array([subpixel_peak(prof, i) for i in pk], np.float64)
    max_pitch = max_pitch or n / max(n_ticks - 1, 1)
    best = None
    tried = set()
    for i in range(len(P) - 1):
        d0 = P[i + 1] - P[i]
        if not (min_pitch <= d0 <= max_pitch):
            continue
        key = (i, int(round(d0)))
        if key in tried:
            continue
        tried.add(key)
        chain = {0: P[i], 1: P[i + 1]}
        for direction in (1, -1):
            k = 1 if direction == 1 else 0
            miss = 0
            while True:
                k += direction
                ks = np.array(sorted(chain))
                a, b, c, _ = _comb_fit(ks, [chain[j] for j in ks])
                pred = (a * k + b) / (c * k + 1.0)
                loc_pitch = abs(a - c * pred)                     # local spacing d row / d k
                if pred < -0.5 * loc_pitch or pred > n - 1 + 0.5 * loc_pitch or loc_pitch < 0.5 * min_pitch:
                    break
                j = int(np.argmin(np.abs(P - pred)))
                if abs(P[j] - pred) <= tol * loc_pitch and P[j] not in chain.values():
                    chain[k] = P[j]
                    miss = 0
                else:
                    miss += 1
                    if miss > max_gap:
                        break
                if len(chain) > 3 * n_ticks:
                    break
        ks = np.array(sorted(chain))
        span = int(ks[-1] - ks[0] + 1)
        # ticks found, less half the ticks missing inside the chain: a chain at half the pitch collects every real
        # tick as well, but misses every other position (a seed pair of a tick and a spurious peak next to it)
        cand = (len(chain) - 0.5 * (span - len(chain)), -span, ks, chain)
        if best is None or cand[:2] > best[:2]:
            best = cand
    if best is None:
        return None
    _, _, ks, chain = best
    span = int(ks[-1] - ks[0] + 1)
    if span < n_ticks - 2 * end_slack:
        return None
    # the n_ticks window: end ticks present (or, with end_slack, hidden and anchored by the majors), best major
    # pattern, most ticks matched
    a, b, c, _ = _comb_fit(ks, [chain[j] for j in ks])
    win_best = None
    for k0 in range(int(ks[0]) - end_slack, int(ks[-1]) - n_ticks + 2 + end_slack):
        kk = np.arange(k0, k0 + n_ticks)
        if end_slack == 0 and (k0 not in chain or k0 + n_ticks - 1 not in chain):
            continue
        if k0 < ks[0] - end_slack or k0 + n_ticks - 1 > ks[-1] + end_slack:
            continue
        r_lo, r_hi = (a * k0 + b) / (c * k0 + 1.0), (a * (k0 + n_ticks - 1) + b) / (c * (k0 + n_ticks - 1) + 1.0)
        if min(r_lo, r_hi) < 0 or max(r_lo, r_hi) > n - 1:
            continue
        m = sum(1 for k in kk if k in chain)
        rows_k = (a * kk + b) / (c * kk + 1.0)
        sep, ok = 1.0, True
        if major_prof is not None:
            vals = []
            pitch = abs(rows_k[-1] - rows_k[0]) / max(n_ticks - 1, 1)
            for r in rows_k:
                lo_, hi_ = int(max(0, math.floor(r - 0.3 * pitch))), int(min(n - 1, math.ceil(r + 0.3 * pitch)))
                vals.append(float(np.max(major_prof[lo_:hi_ + 1])) if hi_ >= lo_ else 0.0)
            vals = np.asarray(vals)
            is_major = ((kk - k0) % major_every) == 0
            vmaj, vmin = vals[is_major], vals[~is_major]
            scale = max(float(vals.max()), 1e-6)
            sep = float(np.clip((np.median(vmaj) - np.median(vmin)) / scale, 0.0, 1.0))
            thr = 0.5 * (np.median(vmaj) + np.median(vmin))
            right = np.mean((vals > thr) == is_major)
            ends_major = (vals[0] > thr or k0 not in chain) and (vals[-1] > thr or k0 + n_ticks - 1 not in chain)
            seen = np.array([k in chain for k in kk])
            n_maj_seen = int(np.sum(is_major & seen))
            ok = bool(sep >= 0.15 and right >= 0.8 and ends_major
                      and (end_slack == 0 or n_maj_seen >= max(3, int(is_major.sum()) - 1)))
        score = (ok, m + 2.0 * sep)
        if win_best is None or score > win_best[0]:
            win_best = (score, k0, m, sep, ok)
    if win_best is None:
        return None
    (_, _), k0, m, sep, ok = win_best
    kk_m = np.array([k for k in range(k0, k0 + n_ticks) if k in chain])
    a, b, c, rms = _comb_fit(kk_m - k0, [chain[k] for k in kk_m])
    kk = np.arange(n_ticks)
    rows = (a * kk + b) / (c * kk + 1.0)
    pitch = float(abs(rows[-1] - rows[0]) / max(n_ticks - 1, 1))
    resid = rms / max(pitch, 1e-6)
    matched = m / float(n_ticks)
    quality = float(np.clip((matched - 0.6) / 0.3, 0, 1) * np.clip(1.0 - (resid - 0.08) / 0.17, 0, 1)
                    * (np.clip(sep / 0.3, 0, 1) if major_prof is not None else 1.0))
    present = [bool((k0 + k) in chain) for k in range(n_ticks)]
    return {"rows": rows.tolist(), "fit": (a, b, c), "pitch": pitch, "matched": matched, "resid": resid,
            "major_sep": sep, "major_ok": ok, "quality": quality, "n_peaks": int(len(P)), "present": present}


def largest_component(mask: np.ndarray, prefer_xy=None, min_area: int = 4):
    """Label image of the connected component of `mask` that is largest (or, with prefer_xy, the one with the best
    area / distance-to-point trade-off). Returns (component mask, stats row) or (None, None)."""
    n, lab, stats, cent = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    best, best_s = None, -1.0
    for k in range(1, n):
        area = stats[k, cv2.CC_STAT_AREA]
        if area < min_area:
            continue
        s = float(area)
        if prefer_xy is not None:
            d = math.hypot(cent[k, 0] - prefer_xy[0], cent[k, 1] - prefer_xy[1])
            s = area / (1.0 + (d / (0.15 * math.hypot(*mask.shape))) ** 2)
        if s > best_s:
            best, best_s = k, s
    if best is None:
        return None, None
    return lab == best, stats[best]
