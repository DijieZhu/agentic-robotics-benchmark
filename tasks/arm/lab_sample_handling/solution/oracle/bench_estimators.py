"""Non-privileged estimators for worklist v1 (probe 8), of the kind an agent could write.

- `locate_tubes`: the tubes in the incoming rack from top-down wrist images: the blue cap tops (the brightest blue:
  the light is overhead, the caps' sides are darker), each blob's ray met with the cap tops' plane (the rack's
  height and the tube's length are known), assigned to the nearest slot of the rack's nominal layout; per tube the
  view nearest straight above it.
Pixel positions are taken in the published intrinsics' convention (a pixel's centre at its index + 0.5, `PX`).
- `locate_cold`: the cold rack's holes from one top-down wrist image: the round ring openings (as probe 3's), each
  ray met with the entrance plane; the rack's shift from the nominal layout fitted to them, every hole the layout
  moved by it.
- `rotor_hole`: a rotor hole's pose from the rotor reading's fitted centre and angle (`rotor_angle.read_angle`)
  and the rotor's known geometry; no image per hole.
"""

import math

import cv2
import numpy as np

import centrifuge_probe as C
import lab_sim as S
import worklist_scene as WS
from localize_probe import cam_pose, intrinsics, render
from rotor_angle import aim_down, blobs, ray_to_z
from tube_insert_probe import B

FIT_TOL = 0.005                 # m, an opening this near a hole of the shifted layout lines up with it
PX = 0.5                        # px: a pixel index's centre in the intrinsics' convention (centre at index + 0.5)


def wrist_image(p, rng, target, dist, heading=0.0):
    reached = aim_down(p, target, dist, heading)
    ww, wh = B.WRIST["width"], B.WRIST["height"]
    K = intrinsics(B.WRIST["fovy"], ww, wh)
    img = render(p.model, p.data, "r_eye_in_hand", ww, wh, rng)
    R, pc = cam_pose(p.data, p.model.camera("r_eye_in_hand").id)
    return img, K, R, pc, bool(reached)


def cap_top_z(a):
    return B.TABLE_Z + 0.015 + 0.0005 + a.tube_len


def locate_tubes(p, rng, a, prior_slots, dist=0.20, views=3, save=None):
    """Estimated (x, y) of the tube in each slot of the incoming rack (None where none was seen). `prior_slots`:
    the rack's nominal slot positions (the layout; the rack may sit up to RACK_SHIFT off)."""
    prior = np.asarray(prior_slots, float)
    z_top = cap_top_z(a)
    n = len(prior)
    found = [None] * n                  # per slot: (distance from the image centre in px, position)
    chunks = np.array_split(np.arange(n), views)
    for v, idx in enumerate(chunks):
        c = prior[idx].mean(0)
        img, K, R, pc, _ok = wrist_image(p, rng, [c[0], c[1], z_top], dist)
        if save:
            cv2.imwrite(f"{save}_{v}.png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        blue = cv2.inRange(hsv, (100, 90, 40), (130, 255, 255))
        vmax = np.percentile(hsv[:, :, 2][blue > 0], 90) if (blue > 0).any() else 255
        top = (blue > 0) & (hsv[:, :, 2] >= 0.8 * vmax)            # the lit top faces
        size_px = 2 * (a.tube_d / 2 + 0.001) * K[0, 0] / dist
        for b in blobs(top.astype(np.uint8) * 255, 6):
            if not (0.5 * size_px <= b["major"] <= 1.6 * size_px):
                continue
            w = ray_to_z(K, R, pc, b["c"] + PX, z_top)[:2]
            d = np.linalg.norm(prior - w, axis=1)
            j = int(np.argmin(d))
            off = float(np.linalg.norm(b["c"] - [img.shape[1] / 2, img.shape[0] / 2]))
            if d[j] < WS.RACK_PITCH / 2 and (found[j] is None or off < found[j][0]):
                found[j] = (off, w)     # the view nearest straight above it (an edge view shows the cap's side)
    return [None if f is None else f[1] for f in found]


def ring_openings(img, size_px):
    """Centres (px, index convention) of the round openings of orange rings in the image: the inner contours of
    the orange mask whose fitted ellipse is an opening's size and round (the gaps between four rings are smaller
    and pointed)."""
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, (5, 100, 60), (35, 255, 255))
    contours, hier = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    out = []
    if hier is None:
        return out
    for c, hh in zip(contours, hier[0]):
        if hh[3] < 0 or len(c) < 5:
            continue
        (cx, cy), axes, _ang = cv2.fitEllipse(c)
        if 0.75 * size_px <= max(axes) <= 1.4 * size_px and min(axes) >= 0.75 * max(axes):
            out.append(np.array([cx, cy]))
    return out


def locate_cold(p, rng, a, prior_holes, dist=0.20, save=None):
    """Estimated entrance centres of the cold rack's holes (dict name -> top), from one top-down wrist image above
    the rack's nominal centre: every round opening in the image met with the entrance plane, the nominal layout's
    shift fitted to them (the one that lines up the most), every hole the layout moved by it
    (the rack is rigid, so an opening hidden by the gripper or a gap taken for one does not move a hole)."""
    tops = {n: np.asarray(h["top"], float) for n, h in prior_holes.items()}
    c = np.mean(list(tops.values()), axis=0)
    img, K, R, pc, _ok = wrist_image(p, rng, c, dist)
    if save:
        cv2.imwrite(save, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    z = float(c[2])
    size_px = 2 * (a.tube_d / 2 + 0.002) * K[0, 0] / dist
    pts = [ray_to_z(K, R, pc, uv + PX, z)[:2] for uv in ring_openings(img, size_px)]
    nominal = np.array([t[:2] for t in tops.values()])
    # the shift that lines up the most openings with the layout (every opening-hole pairing tried; the rack may sit
    # more than half a pitch off diagonally, where the nearest hole is the wrong one), then the median offset of
    # the openings it lines up
    best, shift = (-1, 0.0), np.zeros(2)
    for w in pts:
        for q in nominal:
            cand = w - q
            inl = [np.min(np.linalg.norm(nominal + cand - v, axis=1)) < FIT_TOL for v in pts]
            key = (sum(inl), -float(np.linalg.norm(cand)))
            if key > best:
                best, shift = key, cand
    offs = []
    for w in pts:
        d = np.linalg.norm(nominal + shift - w, axis=1)
        if d.min() < FIT_TOL:
            offs.append(w - nominal[int(np.argmin(d))])
    if offs:
        shift = np.median(offs, axis=0)
    p.cold_openings = len(pts)
    return {n: t + [shift[0], shift[1], 0.0] for n, t in tops.items()}


def rotor_hole(centre_xy, theta_deg, k, a):
    """Rotor hole k's pose from the rotor's estimated centre and angle (hole 0's direction) and its geometry."""
    th = math.radians(theta_deg)
    Rr = C.rot_z(th)
    top, R = C.hole_local(k, S.hole_r(a))
    pr = np.array([centre_xy[0], centre_xy[1], S.rotor_top(a)])
    RR = Rr @ R
    return {"top": pr + Rr @ top, "axis": RR[:, 2], "R": RR, "radius": a.tube_d / 2 + a.clearance,
            "depth": C.HOLE_DEPTH}


def across_mm(est, true):
    """Distance between an estimated and a true hole entrance across the true axis (mm)."""
    e = np.asarray(est["top"] if isinstance(est, dict) else est) - true["top"]
    ax = true["axis"]
    return float(np.linalg.norm(e - np.dot(e, ax) * ax) * 1000)
