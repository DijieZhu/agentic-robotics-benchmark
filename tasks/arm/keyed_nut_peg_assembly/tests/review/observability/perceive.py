"""Camera-only perception for keyed_nut_peg_assembly (a maintainer's evidence that the observations suffice; the
grader never runs it). Everything here uses what a controller receives: the observation's RGB images, its
T_world_wrist and the published camera matrices (the data sheet's CAMERAS, WRIST, fixed_camera_matrices); nothing else.

  * coarse_nut: each fixed camera's largest blob of the nut's colour (kit-aware hue segmentation on the hues the
    cameras see under the cell's warm lamps), its centroid back-projected onto the ring's mid-height plane;
  * find_nuts (refine=True): a close-up from the wrist camera right above the nut: the blob's principal axis, the handle
    end told by its narrow spread, then a circle fitted to the ring's outer edge away from the handle (radius 28 mm);
  * find_pegs: the shaft tops (bright, unsaturated discs) in the wrist camera's view from the start pose, back-projected
    onto the shaft-top plane and grouped;
  * count_bands: the intensity on the camera-facing side of each shaft at the three band heights against the gaps
    between them, in the fixed left and right cameras.
"""
import colorsys
import math

import numpy as np

TABLE_Z = 0.80
NUT_TOP_Z = TABLE_Z + 0.045
SHAFT_TOP_Z = TABLE_Z + 0.010 + 0.065
SHAFT_R = 0.0123                    # the data sheet's shaft radius
NUT_RGB = {"blue": [0.16, 0.36, 0.78], "green": [0.20, 0.62, 0.30], "red": [0.78, 0.18, 0.16],
           "yellow": [0.86, 0.74, 0.16], "orange": [0.85, 0.45, 0.12], "purple": [0.55, 0.25, 0.70],
           "teal": [0.15, 0.65, 0.68], "pink": [0.80, 0.35, 0.55]}
NUT_HUE = {k: colorsys.rgb_to_hsv(*v)[0] * 360.0 for k, v in NUT_RGB.items()}
CAMERAS = {"left": ((-0.30, 0.55, 1.45), (0.05, 0.0, 0.80), 50.0, 320, 240),
           "right": ((-0.30, -0.55, 1.45), (0.05, 0.0, 0.80), 50.0, 320, 240)}
WRIST = (75.0, 256, 192)


def intrinsics(fovy, w, h):
    f = (h / 2.0) / math.tan(math.radians(fovy) / 2.0)
    return np.array([[f, 0, w / 2.0], [0, f, h / 2.0], [0, 0, 1.0]])


def look_at(pos, target, up=(0, 0, 1)):
    pos, target = np.asarray(pos, float), np.asarray(target, float)
    f = target - pos
    f = f / np.linalg.norm(f)
    z = -f
    x = np.cross(np.array(up, dtype=float), z)
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)
    return np.stack([x, y, z], axis=1)


def fixed_camera(name):
    pos, tgt, fovy, w, h = CAMERAS[name]
    T = np.eye(4)
    T[:3, :3] = look_at(pos, tgt) @ np.diag([1.0, -1.0, -1.0])
    T[:3, 3] = pos
    return intrinsics(fovy, w, h), T


def hsv_image(img):
    x = img.astype(np.float32) / 255.0
    mx, mn = x.max(axis=2), x.min(axis=2)
    d = mx - mn + 1e-6
    r, g, b = x[..., 0], x[..., 1], x[..., 2]
    h = np.where(mx == r, ((g - b) / d) % 6.0, np.where(mx == g, (b - r) / d + 2.0, (r - g) / d + 4.0)) * 60.0
    s = np.where(mx > 1e-6, d / (mx + 1e-6), 0.0)
    return h, s, mx


def backproject(us, vs, K, T, z_plane):
    """Pixel centres (u + 0.5, v + 0.5) -> world points on the plane z = z_plane."""
    pix = np.stack([us + 0.5, vs + 0.5, np.ones_like(us, dtype=float)], axis=0)
    d = T[:3, :3] @ (np.linalg.inv(K) @ pix)
    c = T[:3, 3]
    t = (z_plane - c[2]) / d[2]
    return (c[:, None] + d * t).T


def project(points, K, T):
    Rt = T[:3, :3].T
    pc = (Rt @ (np.asarray(points, float).T - T[:3, 3][:, None]))
    uv = (K @ pc)[:2] / pc[2]
    return uv.T - 0.5


# The nuts' hues as the cameras see them: the cell's lamps and the room are warm, which moves every hue towards the
# light's (green and purple by about 20 degrees, red by 10). Medians over maintainer kits that are not among the
# checked ones (seeds 880000 ...); a controller would measure them on the commissioning service's development kits.
SEEN_HUE = {"blue": 224.0, "green": 116.0, "red": 11.0, "yellow": 49.0, "orange": 31.0, "purple": 300.0,
            "teal": 177.0, "pink": 342.0}
# The bench top is light oak (hue about 33 degrees, saturation 0.2 to 0.45) and the floor beside it a darker,
# redder wood; where the hand shades the bench from both lamps the oak is lit by the room alone and turns a
# saturated orange-brown (hue about 25 degrees, saturation 0.55 to 0.8, value below 0.8). The warm colours are told
# from the lit oak by their saturation and from the shaded oak by their brightness.
SAT_MIN = {"orange": 0.50, "yellow": 0.42, "red": 0.45}
TABLE_HALF = 0.40


def colour_mask(img, colour, kit_colours=None, hue_tol=30.0):
    """Pixels of `colour`: nearer in hue to it than to the kit's other colours, saturated enough, and not the bench
    top's oak (hue 18-44 deg: below saturation 0.50 in the light, below value 0.82 in the hand's shade)."""
    h, s, v = hsv_image(img)
    cols = list(kit_colours) if kit_colours else [colour]
    if colour not in cols:
        cols.append(colour)

    def dist(c):
        return np.abs(((h - SEEN_HUE[c]) + 180.0) % 360.0 - 180.0)

    d = dist(colour)
    nearest = np.ones_like(d, dtype=bool)
    for c in cols:
        if c != colour:
            nearest &= d <= dist(c)
    wood = (h > 18.0) & (h < 44.0) & ((s < 0.50) | (v < 0.82))
    return nearest & (d < hue_tol) & (s > SAT_MIN.get(colour, 0.24)) & (v > 0.30) & ~wood


def on_table(mask, K, T, z_plane, margin=0.01):
    """`mask` without the pixels whose ray meets the plane z = z_plane outside the bench top (the floor beside the
    bench is a red-brown wood)."""
    vs, us = np.nonzero(mask)
    if len(us) == 0:
        return mask
    pts = backproject(us.astype(float), vs.astype(float), K, T, z_plane)
    out = (np.abs(pts[:, 0]) > TABLE_HALF - margin) | (np.abs(pts[:, 1]) > TABLE_HALF - margin)
    mask = mask.copy()
    mask[vs[out], us[out]] = False
    return mask


def nut_from_points(pts):
    """(ring centre xy, handle yaw) from the back-projected points of one nut (ring + handle, one blob)."""
    xy = pts[:, :2]
    m = xy.mean(axis=0)
    c = np.cov((xy - m).T)
    w, V = np.linalg.eigh(c)
    u = V[:, 1]
    s = (xy - m) @ u
    n = np.array([-u[1], u[0]])
    # the ring end is the wide one: compare the lateral spread of the points near each end
    lat0 = (xy - m) @ n
    lo, hi = np.percentile(s, 2.0), np.percentile(s, 98.0)

    def spread(sel):
        return float(np.percentile(lat0[sel], 95) - np.percentile(lat0[sel], 5)) if sel.sum() > 3 else 0.0

    if spread(s > hi - 0.03) > spread(s < lo + 0.03):
        u, s = -u, -s
    n = np.array([-u[1], u[0]])
    # ring centre: 28 mm in from the ring side's extreme, laterally at the ring points' mean
    s_ring = np.percentile(s, 1.0) + 0.028
    ring = np.abs(s - s_ring) < 0.024
    lat = ((xy[ring] - m) @ n).mean() if ring.any() else 0.0
    centre = m + u * s_ring + n * lat
    # handle direction refined from the handle points' (beyond 45 mm) mean direction from the centre
    hp = xy[(xy - centre) @ u > 0.045]
    if len(hp) > 5:
        dvec = (hp - centre).mean(axis=0)
        u = dvec / np.linalg.norm(dvec)
    return centre, math.atan2(u[1], u[0])


def largest_blob(mask):
    import cv2
    n, lab, stats, _c = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if n <= 1:
        return None
    j = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return lab == j


def fit_circle(xy):
    """Least-squares circle (Kasa): centre, radius."""
    A = np.column_stack([2 * xy[:, 0], 2 * xy[:, 1], np.ones(len(xy))])
    b = (xy ** 2).sum(axis=1)
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    c = sol[:2]
    return c, math.sqrt(max(0.0, sol[2] + c @ c))


def ring_circle(blob, K, T, centre, handle_yaw):
    """Refine the ring centre by a circle through the blob's outer boundary away from the handle (the top face's
    edge seen from above: radius 28 mm)."""
    import cv2
    cnts, _h = cv2.findContours(blob.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return centre
    c = max(cnts, key=len)[:, 0, :].astype(float)
    pts = backproject(c[:, 0], c[:, 1], K, T, NUT_TOP_Z)[:, :2]
    rel = pts - centre
    ang = np.arctan2(rel[:, 1], rel[:, 0]) - handle_yaw
    ang = (ang + math.pi) % (2 * math.pi) - math.pi
    r = np.linalg.norm(rel, axis=1)
    keep = (np.abs(ang) > math.radians(40.0)) & (r > 0.018) & (r < 0.036)
    if keep.sum() < 20:
        return centre
    cc, rr = fit_circle(pts[keep])
    if abs(rr - 0.028) > 0.005 or np.linalg.norm(cc - centre) > 0.006:
        return centre
    return cc


def find_nuts(img, K, T, colours, min_px=40, kit_colours=None, refine=False):
    out = {}
    for col in colours:
        blob = largest_blob(on_table(colour_mask(img, col, kit_colours), K, T, NUT_TOP_Z))
        if blob is None or blob.sum() < min_px:
            out[col] = None
            continue
        vs, us = np.nonzero(blob)
        pts = backproject(us.astype(float), vs.astype(float), K, T, NUT_TOP_Z)
        centre, yaw = nut_from_points(pts)
        if refine:
            centre = ring_circle(blob, K, T, centre, yaw)
        out[col] = (centre, yaw)
    return out


def find_pegs(img, K, T, exclude_rows_from=None):
    """Shaft tops: bright, unsaturated pixels, back-projected onto the shaft-top plane and grouped (3 pegs). The
    brightness threshold follows the image's brightest unsaturated pixels (the lamps' brightness varies)."""
    h, s, v = hsv_image(img)
    grey = s < 0.10
    if exclude_rows_from is not None:
        grey[exclude_rows_from:, :] = False
    if not grey.any():
        return []
    mask = grey & (v > 0.80 * float(np.percentile(v[grey], 99.8)))
    vs, us = np.nonzero(mask)
    if len(us) == 0:
        return []
    pts = backproject(us.astype(float), vs.astype(float), K, T, SHAFT_TOP_Z)[:, :2]
    inside = (np.abs(pts[:, 0]) < 0.40) & (np.abs(pts[:, 1]) < 0.40)
    pts = pts[inside]
    groups = []
    for p in pts:
        for g in groups:
            if np.linalg.norm(g["c"] - p) < 0.03:
                g["pts"].append(p)
                g["c"] = np.mean(g["pts"], axis=0)
                break
        else:
            groups.append({"c": p.copy(), "pts": [p]})
    groups = [g for g in groups if len(g["pts"]) >= 4]
    groups.sort(key=lambda g: -len(g["pts"]))
    return [g["c"] for g in groups[:3]]


def refine_peg_top(img, K, T, xy, search_px=16):
    """A shaft top in a close-up from above, by its shape instead of its brightness (the arm's shadow may lie
    across it): the centre of the disc of the shaft's radius that stands out most against the ring around it, in
    log intensity, within `search_px` pixels of where `xy` projects. Returns the world (x, y) on the shaft-top
    plane, or None."""
    import cv2
    pc = T[:3, :3].T @ (np.array([xy[0], xy[1], SHAFT_TOP_Z]) - T[:3, 3])
    if pc[2] <= 0.02:
        return None
    u0, v0 = (K @ pc)[:2] / pc[2] - 0.5
    r = SHAFT_R * K[0, 0] / pc[2]                       # the shaft's radius in pixels
    n = int(math.ceil(1.8 * r)) + 1
    yy, xx = np.mgrid[-n:n + 1, -n:n + 1]
    rr = np.hypot(xx, yy)
    disc = (rr <= 0.85 * r).astype(np.float32)
    ring = ((rr >= 1.15 * r) & (rr <= 1.7 * r)).astype(np.float32)
    logi = np.log(img.astype(np.float32).mean(axis=2) + 8.0)
    score = cv2.filter2D(logi, -1, disc / disc.sum(), borderType=cv2.BORDER_REFLECT) \
        - cv2.filter2D(logi, -1, ring / ring.sum(), borderType=cv2.BORDER_REFLECT)
    H, W = score.shape
    ui, vi = int(round(u0)), int(round(v0))
    a0, a1, b0, b1 = max(0, vi - search_px), min(H, vi + search_px + 1), max(0, ui - search_px), min(W, ui + search_px + 1)
    if a1 - a0 < 3 or b1 - b0 < 3:
        return None
    win = score[a0:a1, b0:b1]
    j, i = np.unravel_index(int(np.argmax(win)), win.shape)
    vj, ui2 = a0 + j, b0 + i
    c0, c1, d0, d1 = max(0, vj - 2), min(H, vj + 3), max(0, ui2 - 2), min(W, ui2 + 3)
    w = score[c0:c1, d0:d1] - score[c0:c1, d0:d1].min()
    if w.sum() <= 0:
        return None
    gy, gx = np.mgrid[c0:c1, d0:d1]
    us, vs = float((w * gx).sum() / w.sum()), float((w * gy).sum() / w.sum())
    return backproject(np.array([us]), np.array([vs]), K, T, SHAFT_TOP_Z)[0][:2]


def _bilinear(gray, u, v):
    H, W = gray.shape
    if not (0 <= u < W - 1 and 0 <= v < H - 1):
        return float("nan")
    u0, v0 = int(u), int(v)
    a, b = u - u0, v - v0
    return float((1 - a) * (1 - b) * gray[v0, u0] + a * (1 - b) * gray[v0, u0 + 1]
                 + (1 - a) * b * gray[v0 + 1, u0] + a * b * gray[v0 + 1, u0 + 1])


BAND_DARKNESS = 6.3                     # what one band takes out of the profile down the shaft (mm of full darkness)


def count_bands(img, K, T, peg_xy, detail=None):
    """Bands on the shaft side facing the camera, from the intensity down that side, millimetre by millimetre (a
    band is a good pixel tall in the fixed cameras): the count is the number of dark stretches, 13 mm apart, and
    must agree with the darkness the bands take out of the profile in all. The published calibration is only the
    drawing's and the peg's position an estimate, so the shaft is anchored in the image itself: the bright shaft
    top nearest to where `peg_xy` projects is found and the profile is taken at its drawn offsets below it. Returns
    (bands, contrast) or (None, contrast) when the shaft is not seen clearly."""
    import cv2
    cam = T[:3, 3]
    gray = img.astype(np.float32).mean(axis=2)
    d = np.array([cam[0] - peg_xy[0], cam[1] - peg_xy[1]])
    d = d / np.linalg.norm(d)
    top = project([[peg_xy[0], peg_xy[1], SHAFT_TOP_Z]], K, T)[0]
    h, s, v = hsv_image(img)
    # judged within a window around where the top projects: elsewhere in the image the robot's white covers are
    # brighter than a shaft top in the arm's shade
    H, W = gray.shape
    u0, v0, r = int(round(top[0])), int(round(top[1])), 14
    a0, a1, b0, b1 = max(0, v0 - r), min(H, v0 + r + 1), max(0, u0 - r), min(W, u0 + r + 1)
    if a1 - a0 < 5 or b1 - b0 < 5:
        return None, 0.0
    grey = np.zeros(s.shape, dtype=bool)
    grey[a0:a1, b0:b1] = s[a0:a1, b0:b1] < 0.12
    if not grey.any():
        return None, 0.0
    mask = (grey & (v > 0.88 * float(np.percentile(v[grey], 98.0)))).astype(np.uint8)
    n_lab, _lab, stats, cent = cv2.connectedComponentsWithStats(mask, connectivity=8)
    blobs = [cent[k] for k in range(1, n_lab) if 3 <= stats[k, cv2.CC_STAT_AREA] <= 60
             and math.hypot(cent[k][0] - top[0], cent[k][1] - top[1]) < 10.0]
    if not blobs:
        return None, 0.0
    seen = min(blobs, key=lambda c: math.hypot(c[0] - top[0], c[1] - top[1]))
    shift = np.array([seen[0] - top[0], seen[1] - top[1]])          # the shaft as seen against the shaft as drawn

    def sample(z):
        pts = [[peg_xy[0] + (SHAFT_R + 0.0002) * d[0] + o * -d[1], peg_xy[1] + (SHAFT_R + 0.0002) * d[1] + o * d[0], TABLE_Z + z]
               for o in (-0.003, 0.0, 0.003)]
        uv = project(pts, K, T) + shift
        return float(np.median([_bilinear(gray, u, w) for (u, w) in uv]))

    zs = np.arange(0.074, 0.0245, -0.001)          # every millimetre down the shaft, from under its top edge
    prof = np.array([sample(z) for z in zs])
    if np.isnan(prof).any():
        return None, 0.0
    bright = float(np.percentile(prof, 80.0))       # the clear shaft
    rel = prof / max(bright, 1.0)
    darkness = float(np.clip(1.0 - rel, 0.0, 1.0).sum())            # mm of fully dark shaft: about 6.3 per band
    runs, start = [], None                          # stretches darker than 0.6 of the clear shaft
    for i, low in enumerate(list(rel < 0.60) + [False]):
        if low and start is None:
            start = i
        elif not low and start is not None:
            if i - start >= 2 and float(rel[start:i].min()) < 0.50:
                runs.append((start, i))
            start = None
    n = len(runs)
    contrast = bright - float(prof.min())
    if detail is not None:
        detail.update(darkness=darkness, runs=runs, bright=bright,
                      text="D %.1f runs %d bright %.0f | %s" % (darkness, n, bright, "".join(
                          "#" if r < 0.4 else ("+" if r < 0.6 else ("-" if r < 0.8 else ".")) for r in rel)))
    if n < 1 or n > 3 or abs(darkness / BAND_DARKNESS - n) > 0.75:
        return None, contrast
    mids = [(a + b) / 2.0 for a, b in runs]
    if any(abs((mids[i + 1] - mids[i]) - 13.0) > 4.0 for i in range(n - 1)):      # the bands are 13 mm apart
        return None, contrast
    return n, contrast


def coarse_nut(imgs, colour, kit_colours=None):
    """Rough nut position from the fixed cameras: each camera's largest blob of the colour, its pixel centroid
    back-projected onto the ring's mid-height plane; the mean of the cameras that see it (within a few cm)."""
    est = []
    for cam in ("left", "right"):
        if imgs.get(cam) is None:
            continue
        K, T = fixed_camera(cam)
        mask = colour_mask(imgs[cam], colour, kit_colours)
        vs, us = np.nonzero(mask)
        if len(us):
            pts = backproject(us.astype(float), vs.astype(float), K, T, TABLE_Z + 0.025)
            out = (pts[:, 0] < -0.30) | (pts[:, 0] > 0.0) | (np.abs(pts[:, 1]) > 0.30)
            mask[vs[out], us[out]] = False         # only the nuts' region of the table
        blob = largest_blob(mask)
        if blob is None or blob.sum() < 15:
            continue
        vs, us = np.nonzero(blob)
        p = backproject(np.array([us.mean()]), np.array([vs.mean()]), K, T, TABLE_Z + 0.025)[0]
        est.append(p[:2])
    return np.mean(est, axis=0) if est else None
