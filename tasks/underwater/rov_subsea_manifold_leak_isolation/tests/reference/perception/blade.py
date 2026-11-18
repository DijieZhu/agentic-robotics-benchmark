"""Two-colour blade fit for the last 0.1 m of a grasp approach (numpy + scipy only), with the vehicle's own gripper
masked out by the manual's gripper drawing.  Port of the G2 pilot's vision (proto/g2_grasp_approach/vision.py,
sha256 e3ce8839...): the outline comes from `vehicle.Manual.outline` (or an explicit list) instead of a file next to
the module, the camera from `pose.Cam`, and the closed (vertical) blade is supported.

In the last ~0.1 m the target code leaves the frame (G2 finding 7) and the upper jaw covers the middle of the blade.
The blade's two halves (orange = the viewer's right when open, white = left; panel standard) are segmented in a
region around the predicted blade; their pixel centroids and areas are matched to the model prediction (visible
faces of the blade box sampled, weighted by projected area, samples behind the own gripper dropped) by Gauss-Newton
on the 3 translation components of the panel pose.  Over-masking only costs information: the same mask is applied
to the measurement and to the prediction.

    r = measure_blade(rgb, psi, p0, cam, outline)     # p0 = the current estimate of the target's stem point in L
    r["ok"], r["p0"] (corrected stem point, L), r["dp"], r["info"]
"""
from __future__ import annotations

import math

import numpy as np
from scipy import ndimage

from . import panel_std as PSTD
from .pose import Cam, panel_R

BLADE_L, BLADE_T, BLADE_D, STEM_OUT = PSTD.BLADE_L, PSTD.BLADE_T, PSTD.BLADE_D, PSTD.STEM_OUT
_CW = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])     # 90 deg clockwise seen from the front


# ----------------------------------------------------------------------------------------------------- blade model
def _face_samples(L_blade, n=14):
    """Sample points + outward normals + area elements on the five visible faces of each blade half (open =
    horizontal), in P.  Orange half = X > 0 (the viewer's right), white = X < 0."""
    hl, ht, hd = 0.5 * L_blade, 0.5 * BLADE_T, 0.5 * BLADE_D
    zc = STEM_OUT
    out = {}
    for half, (x0, x1) in (("orange", (0.0, hl)), ("white", (-hl, 0.0))):
        P, N, A = [], [], []
        xs = np.linspace(x0, x1, n + 1)[:-1] + (x1 - x0) / (2 * n)
        zs = np.linspace(zc - hd, zc + hd, 7)[:-1] + hd / 6
        ys = np.array([-ht / 2, ht / 2])
        dx, dz, dy = (x1 - x0) / n, 2 * hd / 6, ht
        for x in xs:                       # top / bottom faces (Y = +-ht)
            for z in zs:
                for sy in (1, -1):
                    P.append([x, sy * ht, z]); N.append([0, sy, 0]); A.append(dx * dz)
            for y in ys:                   # front / back faces (Z = zc +- hd)
                for sz in (1, -1):
                    P.append([x, y, zc + sz * hd]); N.append([0, 0, sz]); A.append(dx * dy)
        xe = x1 if half == "orange" else x0
        for y in ys:                       # outer end face
            for z in zs:
                P.append([xe, y, z]); N.append([np.sign(xe), 0, 0]); A.append(dy * dz)
        out[half] = (np.asarray(P, float), np.asarray(N, float), np.asarray(A, float))
    return out


_SAMPLES = {}


def blade_samples(L_blade=BLADE_L, state: str = "open"):
    k = (round(float(L_blade), 4), state)
    if k not in _SAMPLES:
        s = _face_samples(L_blade)
        if state == "closed":
            s = {h: (P @ _CW.T, N @ _CW.T, A) for h, (P, N, A) in s.items()}
        _SAMPLES[k] = s
    return _SAMPLES[k]


# ----------------------------------------------------------------------------------------------------- own gripper
def jaw_mask(cam: Cam, shape, outline, pad_px: float = 3.0) -> np.ndarray:
    """Image region covered by the vehicle's own gripper: the projected convex hulls of the manual's gripper drawing
    (body-frame point sets), dilated by pad_px."""
    H, W = shape
    m = np.zeros((H, W), bool)
    from scipy.spatial import ConvexHull
    for PB in outline or ():
        P_L = cam.o + (cam.R_nb @ np.asarray(PB, float).T).T
        Pc = (P_L - cam.p) @ cam.R
        keep = Pc[:, 2] > 0.02
        if keep.sum() < 3:
            continue
        uv, _ = cam.project(P_L[keep])
        try:
            hull = ConvexHull(uv)
        except Exception:
            continue
        poly = uv[hull.vertices]
        lo = np.floor(np.maximum(poly.min(0) - pad_px, 0)).astype(int)
        hi = np.ceil(np.minimum(poly.max(0) + pad_px, [W, H])).astype(int)
        if hi[0] <= lo[0] or hi[1] <= lo[1]:
            continue
        jj, ii = np.mgrid[lo[1]:hi[1], lo[0]:hi[0]]
        pts = np.stack([ii.ravel() + 0.5, jj.ravel() + 0.5], 1)
        inside = np.ones(len(pts), bool)                 # convex polygon: inside every edge's half-plane (+ pad)
        cen = poly.mean(0)
        for k in range(len(poly)):
            p1, p2 = poly[k], poly[(k + 1) % len(poly)]
            e = p2 - p1
            nrm = np.array([e[1], -e[0]]) / (np.linalg.norm(e) + 1e-12)
            sg = 1.0 if (cen - p1) @ nrm > 0 else -1.0
            inside &= ((pts - p1) @ nrm) * sg >= -pad_px
        m[lo[1]:hi[1], lo[0]:hi[0]] |= inside.reshape(jj.shape)
    return m


# ----------------------------------------------------------------------------------------------------- prediction
def _predict(cam, psi, p0, smp, excl):
    """Predicted pixel centroid and pixel area of each blade half (visible faces, weighted by projected area), with
    the samples that fall in the excluded (jaw) region or outside the frame dropped."""
    out = {}
    R = panel_R(psi)
    for half, (P, N, A) in smp.items():
        X = p0 + P @ R.T
        Nl = N @ R.T
        v = cam.p - X
        dist = np.linalg.norm(v, axis=1)
        cosv = np.einsum("ij,ij->i", Nl, v) / dist
        uv, z = cam.project(X)
        ok = (cosv > 0) & (z > 0.02) & (uv[:, 0] >= 0) & (uv[:, 0] < cam.W) & (uv[:, 1] >= 0) & (uv[:, 1] < cam.H)
        if excl is not None:
            ui = np.clip(uv[:, 0].astype(int), 0, cam.W - 1)
            vi = np.clip(uv[:, 1].astype(int), 0, cam.H - 1)
            ok &= ~excl[vi, ui]
        # pixel area of a surface element: dA cos(view) f^2 d / z^3 (pinhole: f^2 / cos^3 px per steradian)
        w = np.where(ok, A * cosv * dist / np.maximum(z, 1e-3) ** 3, 0.0) * cam.K[0, 0] ** 2
        sw = float(w.sum())
        if sw <= 0:
            out[half] = (np.array([np.nan, np.nan]), 0.0)
        else:
            out[half] = ((uv * w[:, None]).sum(0) / sw, sw)
    return out


PANEL_CHROMA_MARGIN = 0.03


def panel_chroma(rgb, ring, q: float = 30.0):
    """r - b (0..1 units) of the panel face around the blade: the q-th percentile over `ring` = pixels in a band
    outside the predicted blade region (own gripper excluded), orange-ish pixels dropped.  A low percentile because
    the only contamination is warmer (paint glow, a misplaced region edge).  None if too few pixels."""
    if ring is None or ring.sum() < 30:
        return None
    x = rgb[ring].astype(np.float32) / 255.0
    r, g, b = x[:, 0], x[:, 1], x[:, 2]
    keep = ~((r - b > 0.16) & (b < 0.5 * r))
    return float(np.percentile((r - b)[keep], q)) if keep.sum() >= 30 else None


def segment(rgb, win, region=None, chroma=None):
    """Orange and white paint masks inside a window (v0, v1, u0, u1): paint = clearly brighter than the window's
    background (the panel face), then split by colour (orange: red well above blue, incl. the yellow-ish
    highlights; white: neutral, and -- when the panel face's chroma is given -- warmer than the face).  P14: the
    panel face of the finished skid is a light blue-grey that the lamps make as bright as the white paint's top face
    at close range (G2's darker face was not): white must exceed the face's r - b (panel_chroma, measured on a ring
    around the predicted blade) by PANEL_CHROMA_MARGIN (white paint r - b = +0.03..+0.05, the lit face -0.04..-0.07
    at 0.3 m, c = 1.2)."""
    v0, v1, u0, u1 = win
    x = rgb[v0:v1, u0:u1].astype(np.float32) / 255.0
    r, g, b = x[..., 0], x[..., 1], x[..., 2]
    Y = 0.2126 * r + 0.7152 * g + 0.0722 * b
    sel = np.ones(Y.shape, bool) if region is None else region
    bg = float(np.percentile(Y[sel], 30)) if sel.any() else float(np.percentile(Y, 30))
    bright = Y > max(bg + 0.10, 1.25 * bg)
    rb = r - b
    orange = (r > 0.22) & (rb > 0.16) & (b < 0.5 * r) & (g < 0.70 * r)       # the yellow guard rail has g/r ~0.82
    neutral = bright & (np.abs(rb) < 0.12) & (np.abs(r - g) < 0.10) & ~orange
    if chroma is not None:
        neutral &= rb > chroma + PANEL_CHROMA_MARGIN
    osel = orange & sel
    if osel.sum() > 20:
        # white and orange paint under the same lamps: white luminance ~1.7 x orange (albedos); the grey stem ~0.6 x
        white = neutral & (Y > 0.85 * float(np.median(Y[osel])))
    else:
        nsel = neutral & sel
        top = float(np.percentile(Y[nsel], 98)) if nsel.sum() > 20 else 1.0
        white = neutral & (Y > 0.55 * top)
    st = np.ones((3, 3), bool)
    orange = ndimage.binary_opening(orange, st)
    white = ndimage.binary_opening(white, st)
    return orange, white


def _pred_region(cam, psi, p0, smp, shape, dil_px):
    """Pixels within dil_px of the projected visible blade samples (both halves)."""
    H, W = shape
    m = np.zeros((H, W), bool)
    R = panel_R(psi)
    for half, (P, N, A) in smp.items():
        X = p0 + P @ R.T
        uv, z = cam.project(X)
        ok = (z > 0.02) & (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
        m[uv[ok, 1].astype(int), uv[ok, 0].astype(int)] = True
    m = ndimage.binary_closing(m, np.ones((9, 9), bool))
    if dil_px > 0:
        m = ndimage.binary_dilation(m, np.ones((3, 3), bool), iterations=int(dil_px))
    return m


# ----------------------------------------------------------------------------------------------------- measurement
def measure_blade(rgb, psi, p0, cam: Cam, outline, L_blade=BLADE_L, state: str = "open", iters=6, outer=2,
                  use_area=True):
    """Blade measurement around the predicted pose; on failure (i) a wider search region, (ii) a global
    re-acquisition (the largest orange blob outside the gripper mask gives a new initial position), then the normal
    two-colour fit.  Returns dict(ok, p0, dp, info, why[, retry])."""
    r = _measure_blade(rgb, psi, p0, cam, outline, L_blade, state, iters, outer, use_area, dil_frac=0.10)
    if r.get("ok"):
        return r
    r2 = _measure_blade(rgb, psi, p0, cam, outline, L_blade, state, iters, outer + 1, use_area, dil_frac=0.40)
    if r2.get("ok"):
        r2["retry"] = "wide"
        return r2
    p_init = reacquire(rgb, psi, p0, cam, outline, L_blade, state)
    if p_init is not None:
        r3 = _measure_blade(rgb, psi, p_init, cam, outline, L_blade, state, iters, outer + 1, use_area, dil_frac=0.30)
        am, ap = (r3.get("info") or {}).get("area_meas", {}), (r3.get("info") or {}).get("area_pred", {})
        areas_ok = all(0.6 * ap.get(k, 1e9) <= am.get(k, 0) <= 1.6 * ap.get(k, 0) for k in ("orange", "white"))
        if r3.get("ok") and np.linalg.norm(r3["p0"] - p_init) < 0.05 and areas_ok:
            r3["retry"] = "reacquire"
            r3["dp"] = r3["p0"] - np.asarray(p0, float)
            return r3
    return r


def reacquire(rgb, psi, p0, cam: Cam, outline, L_blade=BLADE_L, state: str = "open"):
    """Initial blade position from the largest orange blob in the frame (outside the gripper mask): solve the panel
    position so that the orange half's predicted centroid and pixel area match the blob (3 equations, 3 unknowns)."""
    H, W = rgb.shape[:2]
    excl = jaw_mask(cam, (H, W), outline)
    o, _ = segment(rgb, (0, H, 0, W))
    o &= ~excl
    lab, n = ndimage.label(o, structure=np.ones((3, 3), bool))
    if n == 0:
        return None
    areas = ndimage.sum_labels(np.ones_like(lab), lab, np.arange(1, n + 1))
    k = int(np.argmax(areas)) + 1
    if areas[k - 1] < 40:
        return None
    jj, ii = np.nonzero(lab == k)
    y = np.array([ii.mean() + 0.5, jj.mean() + 0.5, math.sqrt(float(len(ii)))])
    smp = {"orange": blade_samples(L_blade, state)["orange"]}

    def f(q):
        pr = _predict(cam, psi, q, smp, excl)["orange"]
        return np.array([pr[0][0], pr[0][1], math.sqrt(max(pr[1], 1e-9))])
    p = np.asarray(p0, float).copy()
    for _ in range(10):
        r = f(p) - y
        if not np.all(np.isfinite(r)):
            return None
        J = np.zeros((3, 3))
        for kk in range(3):
            dp = np.zeros(3)
            dp[kk] = 1e-4
            J[:, kk] = (f(p + dp) - f(p)) / 1e-4
        try:
            step = np.clip(np.linalg.lstsq(J, -r, rcond=None)[0], -0.05, 0.05)
        except np.linalg.LinAlgError:
            return None
        p = p + step
        if np.max(np.abs(step)) < 5e-5:
            break
    return p if np.linalg.norm(p - p0) < 0.30 else None


def _measure_blade(rgb, psi, p0, cam, outline, L_blade, state, iters, outer, use_area, dil_frac):
    """Correct the panel position p0 (L) from the blade's two colour halves: measured pixel centroids (and areas) of the
    orange and the white half, inside a region around the predicted blade, matched to the model prediction by
    Gauss-Newton on the 3 translation components."""
    smp = blade_samples(L_blade, state)
    H, W = rgb.shape[:2]
    excl = jaw_mask(cam, (H, W), outline)
    p = np.asarray(p0, float).copy()
    info = {}
    f = None
    wts = None
    y = None
    for it in range(outer):
        pr = _predict(cam, psi, p, smp, excl)
        if pr["orange"][1] < 30 or pr["white"][1] < 30:
            return dict(ok=False, why="pred_small", info=dict(area_pred={k: v[1] for k, v in pr.items()}))
        L_px = float(np.linalg.norm(pr["orange"][0] - pr["white"][0])) * 2.0
        reg = _pred_region(cam, psi, p, smp, (H, W), max(8.0, (dil_frac if it == 0 else 0.08) * L_px)) & ~excl
        vv, uu = np.nonzero(reg)
        if len(uu) < 30:
            return dict(ok=False, why="region_empty", info=info)
        v0, v1, u0, u1 = int(vv.min()), int(vv.max()) + 1, int(uu.min()), int(uu.max()) + 1
        st3 = np.ones((3, 3), bool)
        ring = ndimage.binary_dilation(reg, st3, iterations=12) & ~ndimage.binary_dilation(reg, st3, iterations=4) & ~excl
        ch = panel_chroma(rgb, ring)
        o_w, w_w = segment(rgb, (v0, v1, u0, u1), reg[v0:v1, u0:u1], chroma=ch)
        o_all = np.zeros((H, W), bool)
        w_all = np.zeros((H, W), bool)
        o_all[v0:v1, u0:u1] = o_w
        w_all[v0:v1, u0:u1] = w_w
        meas = {}
        for half, msk in (("orange", o_all & reg), ("white", w_all & reg)):
            jj, ii = np.nonzero(msk)
            if len(ii) < 15:
                return dict(ok=False, why=f"no_{half}", info=info)
            meas[half] = (np.array([ii.mean() + 0.5, jj.mean() + 0.5]), float(len(ii)))
        info = dict(area_meas={k: v[1] for k, v in meas.items()}, area_pred={k: round(v[1], 1) for k, v in pr.items()},
                    L_px=round(L_px, 1))
        for half in ("orange", "white"):
            a_m, a_p = meas[half][1], pr[half][1]
            if a_m < 0.3 * a_p or a_m > 3.0 * a_p:
                return dict(ok=False, why=f"area_{half}", info=info)
        y = np.concatenate([meas["orange"][0], meas["white"][0]])
        if use_area:
            y = np.concatenate([y, [math.sqrt(meas["orange"][1]), math.sqrt(meas["white"][1])]])

        def f(q):
            prq = _predict(cam, psi, q, smp, excl)
            out = np.concatenate([prq["orange"][0], prq["white"][0]])
            if use_area:
                out = np.concatenate([out, [math.sqrt(max(prq["orange"][1], 1e-9)), math.sqrt(max(prq["white"][1], 1e-9))]])
            return out
        wts = np.ones(len(y))
        if use_area:
            wts[4:] = 0.5
        for _ in range(iters):
            r = (f(p) - y) * wts
            if not np.all(np.isfinite(r)):
                return dict(ok=False, why="pred_nan", info=info)
            J = np.zeros((len(y), 3))
            for k in range(3):
                dp = np.zeros(3)
                dp[k] = 1e-4
                J[:, k] = ((f(p + dp) * wts) - (f(p) * wts)) / 1e-4
            try:
                step = np.linalg.lstsq(J, -r, rcond=None)[0]
            except np.linalg.LinAlgError:
                return dict(ok=False, why="lstsq", info=info)
            step = np.clip(step, -0.03, 0.03)
            p = p + step
            if np.max(np.abs(step)) < 2e-5:
                break
    r = (f(p) - y) * wts
    info["res_px"] = float(np.sqrt(np.mean(r[:4] ** 2)))
    ok = info["res_px"] < max(4.0, 0.02 * info.get("L_px", 0.0)) and np.linalg.norm(p - p0) < 0.08
    return dict(ok=bool(ok), dp=p - np.asarray(p0, float), p0=p, info=info, why=None if ok else "residual")


def bright_plates(rgb, min_area=120):
    """Plate candidates when no code decodes (too far / too turbid): bright, roughly 2:1 blobs.  Returns
    [(u, v, width_px, height_px)] sorted by area (largest first)."""
    x = rgb.astype(np.float32) / 255.0
    Y = 0.2126 * x[..., 0] + 0.7152 * x[..., 1] + 0.0722 * x[..., 2]
    thr = max(0.35, float(np.median(Y)) * 1.8)
    m = ndimage.binary_opening(Y > thr, np.ones((3, 3), bool))
    lab, n = ndimage.label(m)
    out = []
    for k, sl in enumerate(ndimage.find_objects(lab), start=1):
        if sl is None:
            continue
        h, w = sl[0].stop - sl[0].start, sl[1].stop - sl[1].start
        area = int((lab[sl] == k).sum())
        if area < min_area or w < 12 or not (1.2 < w / max(h, 1) < 3.5) or area < 0.5 * w * h:
            continue
        out.append((0.5 * (sl[1].start + sl[1].stop), 0.5 * (sl[0].start + sl[0].stop), float(w), float(h), area))
    out.sort(key=lambda t: -t[4])
    return [o[:4] for o in out]
