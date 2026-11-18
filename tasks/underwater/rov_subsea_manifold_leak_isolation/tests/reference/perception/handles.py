"""Handle-state reader (numpy + scipy only): is the paddle blade horizontal (open) or vertical (closed)?

Anchored on a plate seen in the same frame (the panel standard places the stem 0.22 m below the plate centre, +-5 mm):
panel-local points with known image positions (the decoded code's 4 corners and / or the plate's 4 corners from the
tag reader) -> homography -> rough camera pose of the panel plane (intrinsics from the public camera spec) -> a search
window around the predicted blade centre (radius 0.16 m at the blade's range, the plate itself masked out) -> bright
pixels above the window's median (the blade is painted orange / white, the panel is dark) -> largest blob -> principal
axis from second moments -> compared with the image direction of the panel's horizontal at the blade.

The pose is only used to place the window: a small code seen obliquely gives a pose good to ~1-2 cm at the blade,
which is too coarse for a template test on a 12 mm thick blade but ample for a window.

    read_handle(frame_rgb, local_xy (N, 2), image_uv (N, 2), hfov_deg) -> ('open' | 'closed' | None, info)
"""
from __future__ import annotations

import math

import numpy as np
from scipy import ndimage

from . import imgproc as IP  # P14: package-relative import (G4: sys.path insert + top-level import)
from . import panel_std as PS

WINDOW_M = 0.16
PLATE_PAD_M = 0.02


def pose_from_homography(Hm: np.ndarray, K: np.ndarray):
    """Plane Z = 0 of the panel-local frame -> camera frame (x right, y down, z forward): P_c = R P + t."""
    B = np.linalg.solve(K, Hm)
    lam = 2.0 / (np.linalg.norm(B[:, 0]) + np.linalg.norm(B[:, 1]))
    if B[2, 2] * lam < 0:
        lam = -lam
    r1, r2, t = lam * B[:, 0], lam * B[:, 1], lam * B[:, 2]
    R = np.stack([r1, r2, np.cross(r1, r2)], 1)
    U, _, Vt = np.linalg.svd(R)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        R = U @ np.diag([1, 1, -1]) @ Vt
    return R, t


def project(K, R, t, P):
    Pc = (R @ np.atleast_2d(np.asarray(P, float)).T).T + t
    z = np.maximum(Pc[:, 2], 1e-6)
    return np.stack([K[0, 0] * Pc[:, 0] / z + K[0, 2], K[1, 1] * Pc[:, 1] / z + K[1, 2]], 1), Pc[:, 2]


def blade_box(state: str) -> np.ndarray:
    """8 corners of the blade box in panel-local coordinates for 'open' (horizontal) or 'closed' (vertical)."""
    hl, ht, hd = PS.BLADE_L / 2, PS.BLADE_T / 2, PS.BLADE_D / 2
    sx, sy = (hl, ht) if state == "open" else (ht, hl)
    z0 = PS.STEM_OUT
    return np.array([[x, y, z] for x in (-sx, sx) for y in (-sy, sy) for z in (z0 - hd, z0 + hd)])


def read_handle(frame_rgb: np.ndarray, local_xy: np.ndarray, image_uv: np.ndarray, hfov_deg: float,
                min_area_px: int = 8, min_elong: float = 1.6):
    gray = IP.to_gray(frame_rgb)
    H, W = gray.shape
    K = PS.intrinsics(W, H, hfov_deg)
    Hm = IP.homography(np.asarray(local_xy, float), np.asarray(image_uv, float))
    R, t = pose_from_homography(Hm, K)
    c_uv, z = project(K, R, t, [[0.0, 0.0, PS.STEM_OUT]])
    if z[0] <= 0.1:
        return None, {"reason": "behind"}
    c_uv = c_uv[0]
    s_px = K[0, 0] / z[0]
    rad = WINDOW_M * s_px
    u0, u1 = int(max(0, c_uv[0] - rad)), int(min(W, c_uv[0] + rad + 1))
    v0, v1 = int(max(0, c_uv[1] - rad)), int(min(H, c_uv[1] + rad + 1))
    if u1 - u0 < 4 or v1 - v0 < 4:
        return None, {"reason": "window_outside"}
    uu, vv = np.meshgrid(np.arange(u0, u1) + 0.5, np.arange(v0, v1) + 0.5)
    win = (uu - c_uv[0]) ** 2 + (vv - c_uv[1]) ** 2 <= rad * rad
    # mask the plate (bright) out of the window
    pc = PS.plate_corners_local()
    pad = np.array([[-1, 1], [1, 1], [1, -1], [-1, -1]]) * PLATE_PAD_M
    pq = IP.apply_h(Hm, pc + pad)
    pm, org = IP.convex_poly_mask((H, W), pq)
    if pm is not None:
        full = np.zeros((H, W), bool)
        full[org[0]:org[0] + pm.shape[0], org[1]:org[1] + pm.shape[1]] = pm
        win &= ~full[v0:v1, u0:u1]
    if win.sum() < 20:
        return None, {"reason": "window_empty"}
    g = gray[v0:v1, u0:u1]
    vals = g[win]
    bg = float(np.median(vals))
    top = float(np.percentile(vals, 99))
    thr = bg + max(0.03, 0.4 * (top - bg))
    bright = win & (g > thr)
    lab, n = ndimage.label(bright, structure=np.ones((3, 3), bool))
    if n == 0:
        return None, {"reason": "no_blob", "bg": round(bg, 3), "top": round(top, 3)}
    areas = ndimage.sum_labels(np.ones_like(lab), lab, np.arange(1, n + 1))
    k = int(np.argmax(areas)) + 1
    if areas[k - 1] < min_area_px:
        return None, {"reason": "blob_small", "area": int(areas[k - 1])}
    jj, ii = np.nonzero(lab == k)
    X = np.stack([ii + 0.5, jj + 0.5], 1).astype(float)
    C = np.cov(X.T)
    ev, evec = np.linalg.eigh(C)
    elong = math.sqrt(max(ev[1], 1e-9) / max(ev[0], 1e-9))
    major = evec[:, 1]
    hx, _ = project(K, R, t, [[0.05, 0.0, PS.STEM_OUT], [-0.05, 0.0, PS.STEM_OUT]])
    hdir = hx[0] - hx[1]
    hdir = hdir / (np.linalg.norm(hdir) + 1e-12)
    dang = math.degrees(math.acos(min(1.0, abs(float(major @ hdir)))))
    info = {"area": int(areas[k - 1]), "elong": round(elong, 2), "angle_from_panel_h_deg": round(dang, 1),
            "bg": round(bg, 3), "thr": round(thr, 3)}
    if elong < min_elong:
        return None, dict(info, reason="not_elongated")
    return ("open" if dang < 45.0 else "closed"), info


def _norm_cov(mask: np.ndarray):
    jj, ii = np.nonzero(mask)
    if len(ii) < 4:
        return None
    C = np.cov(np.stack([ii + 0.5, jj + 0.5]))
    return C / (np.trace(C) + 1e-12)


def read_handle_v2(frame_rgb: np.ndarray, local_xy: np.ndarray, image_uv: np.ndarray, hfov_deg: float,
                   min_area_px: int = 8, max_ratio: float = 0.7):
    """v2 (after the G4 sweep): same window and segmentation idea, but the blob's normalised second-moment matrix is
    compared with those of the two projected blade boxes (open / closed), so the 60 mm side faces seen obliquely and
    the perspective of the panel's horizontal are accounted for; the segmentation threshold is lowered so that the
    orange half is included.  Returns (state | None, info)."""
    gray = IP.to_gray(frame_rgb)
    H, W = gray.shape
    K = PS.intrinsics(W, H, hfov_deg)
    Hm = IP.homography(np.asarray(local_xy, float), np.asarray(image_uv, float))
    R, t = pose_from_homography(Hm, K)
    c_uv, z = project(K, R, t, [[0.0, 0.0, PS.STEM_OUT]])
    if z[0] <= 0.1:
        return None, {"reason": "behind"}
    c_uv = c_uv[0]
    rad = WINDOW_M * K[0, 0] / z[0]
    u0, u1 = int(max(0, c_uv[0] - rad)), int(min(W, c_uv[0] + rad + 1))
    v0, v1 = int(max(0, c_uv[1] - rad)), int(min(H, c_uv[1] + rad + 1))
    if u1 - u0 < 4 or v1 - v0 < 4:
        return None, {"reason": "window_outside"}
    uu, vv = np.meshgrid(np.arange(u0, u1) + 0.5, np.arange(v0, v1) + 0.5)
    win = (uu - c_uv[0]) ** 2 + (vv - c_uv[1]) ** 2 <= rad * rad
    pad = np.array([[-1, 1], [1, 1], [1, -1], [-1, -1]]) * PLATE_PAD_M
    pm, org = IP.convex_poly_mask((H, W), IP.apply_h(Hm, PS.plate_corners_local() + pad))
    if pm is not None:
        full = np.zeros((H, W), bool)
        full[org[0]:org[0] + pm.shape[0], org[1]:org[1] + pm.shape[1]] = pm
        win &= ~full[v0:v1, u0:u1]
    if win.sum() < 20:
        return None, {"reason": "window_empty"}
    g = gray[v0:v1, u0:u1]
    vals = g[win]
    bg = float(np.median(vals))
    top = float(np.percentile(vals, 99))
    thr = bg + max(0.03, 0.25 * (top - bg))
    lab, n = ndimage.label(win & (g > thr), structure=np.ones((3, 3), bool))
    if n == 0:
        return None, {"reason": "no_blob"}
    areas = ndimage.sum_labels(np.ones_like(lab), lab, np.arange(1, n + 1))
    k = int(np.argmax(areas)) + 1
    if areas[k - 1] < min_area_px:
        return None, {"reason": "blob_small", "area": int(areas[k - 1])}
    Cb = _norm_cov(lab == k)
    sc = {}
    for st in ("open", "closed"):
        uv, zz = project(K, R, t, blade_box(st))
        m, o = IP.convex_poly_mask((H, W), uv)
        Ch = _norm_cov(m) if m is not None else None
        sc[st] = float(np.linalg.norm(Cb - Ch)) if (Ch is not None and Cb is not None) else 9.0
    best = min(sc, key=sc.get)
    other = "closed" if best == "open" else "open"
    info = {"area": int(areas[k - 1]), "score_open": round(sc["open"], 4), "score_closed": round(sc["closed"], 4)}
    if sc[best] > max_ratio * sc[other]:
        return None, dict(info, reason="ambiguous")
    return best, info


def read_handle_v3(frame_rgb: np.ndarray, local_xy: np.ndarray, image_uv: np.ndarray, hfov_deg: float,
                   min_area_px: int = 8, max_ratio: float = 0.7, window_m: float = 0.13, min_cover: float = 0.3):
    """v3 = v2 + (a) every plate position of the 3 x 3 neighbourhood (panel standard pitch) is masked out of the
    window, not only the own plate (in clear water the plate of the row below entered the window when looking down
    20 deg), (b) a tighter window (0.13 m at the blade), (c) the blob must cover >= min_cover of the projected blade
    area of the chosen hypothesis (truncated or partial blobs -> None)."""
    gray = IP.to_gray(frame_rgb)
    H, W = gray.shape
    K = PS.intrinsics(W, H, hfov_deg)
    Hm = IP.homography(np.asarray(local_xy, float), np.asarray(image_uv, float))
    R, t = pose_from_homography(Hm, K)
    c_uv, z = project(K, R, t, [[0.0, 0.0, PS.STEM_OUT]])
    if z[0] <= 0.1:
        return None, {"reason": "behind"}
    c_uv = c_uv[0]
    rad = window_m * K[0, 0] / z[0]
    u0, u1 = int(max(0, c_uv[0] - rad)), int(min(W, c_uv[0] + rad + 1))
    v0, v1 = int(max(0, c_uv[1] - rad)), int(min(H, c_uv[1] + rad + 1))
    if u1 - u0 < 4 or v1 - v0 < 4:
        return None, {"reason": "window_outside"}
    uu, vv = np.meshgrid(np.arange(u0, u1) + 0.5, np.arange(v0, v1) + 0.5)
    win = (uu - c_uv[0]) ** 2 + (vv - c_uv[1]) ** 2 <= rad * rad
    pad = np.array([[-1, 1], [1, 1], [1, -1], [-1, -1]]) * PLATE_PAD_M
    for dx in (-PS.COL_PITCH, 0.0, PS.COL_PITCH):
        for dy in (-PS.ROW_PITCH, 0.0, PS.ROW_PITCH):
            pq = IP.apply_h(Hm, PS.plate_corners_local() + pad + np.array([dx, dy]))
            pm, org = IP.convex_poly_mask((H, W), pq)
            if pm is None:
                continue
            r0, c0 = org
            # intersect the plate mask with the window box
            ys, xs = slice(max(r0, v0), min(r0 + pm.shape[0], v1)), slice(max(c0, u0), min(c0 + pm.shape[1], u1))
            if ys.stop <= ys.start or xs.stop <= xs.start:
                continue
            sub = pm[ys.start - r0:ys.stop - r0, xs.start - c0:xs.stop - c0]
            win[ys.start - v0:ys.stop - v0, xs.start - u0:xs.stop - u0] &= ~sub
    if win.sum() < 20:
        return None, {"reason": "window_empty"}
    g = gray[v0:v1, u0:u1]
    vals = g[win]
    bg = float(np.median(vals))
    top = float(np.percentile(vals, 99))
    thr = bg + max(0.03, 0.25 * (top - bg))
    lab, n = ndimage.label(win & (g > thr), structure=np.ones((3, 3), bool))
    if n == 0:
        return None, {"reason": "no_blob"}
    areas = ndimage.sum_labels(np.ones_like(lab), lab, np.arange(1, n + 1))
    k = int(np.argmax(areas)) + 1
    if areas[k - 1] < min_area_px:
        return None, {"reason": "blob_small", "area": int(areas[k - 1])}
    Cb = _norm_cov(lab == k)
    sc, area_h = {}, {}
    for st in ("open", "closed"):
        uv, zz = project(K, R, t, blade_box(st))
        m, o = IP.convex_poly_mask((H, W), uv)
        Ch = _norm_cov(m) if m is not None else None
        area_h[st] = int(m.sum()) if m is not None else 0
        sc[st] = float(np.linalg.norm(Cb - Ch)) if (Ch is not None and Cb is not None) else 9.0
    best = min(sc, key=sc.get)
    other = "closed" if best == "open" else "open"
    info = {"area": int(areas[k - 1]), "score_open": round(sc["open"], 4), "score_closed": round(sc["closed"], 4),
            "area_pred": area_h[best]}
    if sc[best] > max_ratio * sc[other]:
        return None, dict(info, reason="ambiguous")
    if area_h[best] > 0 and areas[k - 1] < min_cover * area_h[best]:
        return None, dict(info, reason="partial_blob")
    return best, info


def read_handle_v4(frame_rgb: np.ndarray, local_xy: np.ndarray, image_uv: np.ndarray, hfov_deg: float,
                   margin_px: float = 3.0, **kw):
    """v4 = v3, but no answer unless both hypothetical blade boxes project completely inside the frame (margin_px):
    a blade cut by the frame edge cannot be told apart reliably (G4 offset frames)."""
    H, W = frame_rgb.shape[:2]
    K = PS.intrinsics(W, H, hfov_deg)
    Hm = IP.homography(np.asarray(local_xy, float), np.asarray(image_uv, float))
    R, t = pose_from_homography(Hm, K)
    for st in ("open", "closed"):
        uv, z = project(K, R, t, blade_box(st))
        if np.any(z <= 0.1) or uv[:, 0].min() < margin_px or uv[:, 1].min() < margin_px \
                or uv[:, 0].max() > W - margin_px or uv[:, 1].max() > H - margin_px:
            return None, {"reason": "blade_not_in_frame"}
    return read_handle_v3(frame_rgb, local_xy, image_uv, hfov_deg, **kw)


# ----------------------------------------------------------------------------------------------- v5 (package P14)
V5 = dict(window_m=0.16, min_px=6, margin_px=3.0, chroma_margin=0.03, axis_ratio=2.0, sep_m=(0.03, 0.13))


def _blade_plane_h(K, R, t):
    """Homography of the blade's front plane Z = STEM_OUT (panel-local X, Y -> image)."""
    return K @ np.stack([R[:, 0], R[:, 1], R[:, 2] * PS.STEM_OUT + t], 1)


def read_handle_v5(frame_rgb: np.ndarray, local_xy: np.ndarray, image_uv: np.ndarray, hfov_deg: float,
                   exclude=None, **kw):
    """Handle state from the PAINT ARRANGEMENT (P14): the panel standard paints the half of the blade on the viewer's
    right (open, horizontal) / bottom (closed, vertical) orange and the other half white, so the direction from the
    white paint's centroid to the orange paint's centroid, mapped into the blade plane, is +X for open and -Y for
    closed -- whatever the panel face's brightness (v1-v4 take the brightest blob, which on the finished skid's lit
    light-grey face can be the face itself).  Window around the blade predicted from the anchor (as v3, the 3 x 3
    neighbourhood's plates and `exclude` masked); orange = strong red-over-blue chroma; white = neutral, warmer than
    the window's panel face (30th percentile of r - b over the non-orange window) and brighter than it.  Answers only
    when both blade hypotheses project fully inside the frame (as v4), both colours have >= min_px pixels, the
    centroid separation is 30-130 mm and one axis dominates by axis_ratio; otherwise None."""
    P = dict(V5, **kw)
    H, W = frame_rgb.shape[:2]
    K = PS.intrinsics(W, H, hfov_deg)
    Hm = IP.homography(np.asarray(local_xy, float), np.asarray(image_uv, float))
    R, t = pose_from_homography(Hm, K)
    mp = P["margin_px"]
    for st in ("open", "closed"):
        uv, z = project(K, R, t, blade_box(st))
        if np.any(z <= 0.1) or uv[:, 0].min() < mp or uv[:, 1].min() < mp or uv[:, 0].max() > W - mp \
                or uv[:, 1].max() > H - mp:
            return None, {"reason": "blade_not_in_frame"}
    c_uv, z = project(K, R, t, [[0.0, 0.0, PS.STEM_OUT]])
    c_uv = c_uv[0]
    rad = P["window_m"] * K[0, 0] / z[0]
    u0, u1 = int(max(0, c_uv[0] - rad)), int(min(W, c_uv[0] + rad + 1))
    v0, v1 = int(max(0, c_uv[1] - rad)), int(min(H, c_uv[1] + rad + 1))
    if u1 - u0 < 4 or v1 - v0 < 4:
        return None, {"reason": "window_outside"}
    uu, vv = np.meshgrid(np.arange(u0, u1) + 0.5, np.arange(v0, v1) + 0.5)
    win = (uu - c_uv[0]) ** 2 + (vv - c_uv[1]) ** 2 <= rad * rad
    pad = np.array([[-1, 1], [1, 1], [1, -1], [-1, -1]]) * PLATE_PAD_M
    for dx in (-PS.COL_PITCH, 0.0, PS.COL_PITCH):
        for dy in (-PS.ROW_PITCH, 0.0, PS.ROW_PITCH):
            pq = IP.apply_h(Hm, PS.plate_corners_local() + pad + np.array([dx, dy]))
            pm, org = IP.convex_poly_mask((H, W), pq)
            if pm is None:
                continue
            r0, c0 = org
            ys, xs = slice(max(r0, v0), min(r0 + pm.shape[0], v1)), slice(max(c0, u0), min(c0 + pm.shape[1], u1))
            if ys.stop <= ys.start or xs.stop <= xs.start:
                continue
            win[ys.start - v0:ys.stop - v0, xs.start - u0:xs.stop - u0] &= \
                ~pm[ys.start - r0:ys.stop - r0, xs.start - c0:xs.stop - c0]
    if exclude is not None:
        win &= ~exclude[v0:v1, u0:u1]
    if win.sum() < 30:
        return None, {"reason": "window_empty"}
    x = frame_rgb[v0:v1, u0:u1].astype(np.float32) / 255.0
    r, g, b = x[..., 0], x[..., 1], x[..., 2]
    Y = 0.2126 * r + 0.7152 * g + 0.0722 * b
    rb = r - b
    orange = win & (r > 0.12) & (rb > 0.12) & (b < 0.55 * r) & (g < 0.75 * r)   # the yellow rail / frame: g/r ~ 0.8
    rest = win & ~orange
    if rest.sum() < 20:
        return None, {"reason": "window_empty"}
    c_face = float(np.percentile(rb[rest], 30))
    y_face = float(np.median(Y[rest]))
    white = rest & (np.abs(r - g) < 0.10) & (rb > c_face + P["chroma_margin"]) & (Y > y_face + max(0.04, 0.15 * y_face))
    no, nw = int(orange.sum()), int(white.sum())
    info = {"n_orange": no, "n_white": nw, "face_rb": round(c_face, 4), "face_Y": round(y_face, 4)}
    if no < P["min_px"] or nw < P["min_px"]:
        return None, dict(info, reason="paint_not_found")
    jo, io_ = np.nonzero(orange)
    jw, iw = np.nonzero(white)
    po = np.array([io_.mean() + u0 + 0.5, jo.mean() + v0 + 0.5])
    pw = np.array([iw.mean() + u0 + 0.5, jw.mean() + v0 + 0.5])
    Hi = np.linalg.inv(_blade_plane_h(K, R, t))
    lo, lw = IP.apply_h(Hi, np.stack([po, pw]))
    d = lo - lw
    sep = float(np.hypot(*d))
    info.update(d_mm=[round(1e3 * float(d[0]), 1), round(1e3 * float(d[1]), 1)])
    if not (P["sep_m"][0] <= sep <= P["sep_m"][1]):
        return None, dict(info, reason="separation")
    if d[0] > 0 and abs(d[0]) >= P["axis_ratio"] * abs(d[1]):
        return "open", info
    if d[1] < 0 and abs(d[1]) >= P["axis_ratio"] * abs(d[0]):
        return "closed", info
    return None, dict(info, reason="ambiguous_direction")
