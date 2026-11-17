"""Reading the rotor's angle with the wrist camera (probe 6), of the kind an agent could write.

The arm puts the wrist camera `dist` straight above the rotor's axis (known to a few millimetres from the
centrifuge's position), looking down. In the image (with the environment's pixel noise): the dark hole openings in
the light rotor (blobs of the expected size, the knob bar excluded by its shape) and the green position-1 mark.
Each blob's centre ray meets the plane of the hole entrances; the eight entrance points give the rotor's centre and
its angle modulo 45 deg (each point's angle about the centre, minus 45 deg times its index, averaged on the
circle); the mark picks which of the eight it is. Part of the gripper shows at the image's bottom edge and can hide
the mark: a caller that gets no reading looks again with the camera turned (`heading`). The handle is no guide: its play lets it sit up to half the play
off the rotor.
"""

import math

import cv2
import numpy as np

import centrifuge_probe as C
import lab_sim as S
from localize_probe import cam_pose, intrinsics, render
from tube_insert_probe import B

STEP = 2 * math.pi / C.NHOLES
BORE_D = 0.019                                       # m, a hole's opening (the tube plus clearance)
CAP_ALONG = 0.014                                    # m, a seated tube's cap centre above the entrance, along the axis


def aim_down(p, target, dist, heading=0.0, steps=250):
    """Servo the hand so the wrist camera sits `dist` above `target`, its optical axis straight down (the site
    pose follows from the camera's mount on the hand); the camera keeps its current heading, turned by `heading`
    (rad)."""
    d, m = p.data, p.model
    wcam = m.camera("r_eye_in_hand").id
    Rs = d.site_xmat[p.site].reshape(3, 3)
    C_s = Rs.T @ d.cam_xmat[wcam].reshape(3, 3)
    o_s = Rs.T @ (d.cam_xpos[wcam] - d.site_xpos[p.site])
    xc = d.cam_xmat[wcam].reshape(3, 3)[:, 0]
    psi = math.atan2(xc[1], xc[0]) + heading
    x_c = np.array([math.cos(psi), math.sin(psi), 0.0])
    z_c = np.array([0.0, 0.0, -1.0])
    R_cv = np.column_stack([x_c, np.cross(z_c, x_c), z_c])
    R_goal = R_cv @ np.diag([1.0, -1.0, -1.0]) @ C_s.T
    site_goal = np.asarray(target, float) + [0, 0, dist] - R_goal @ o_s
    ok = p.servo(site_goal, R_goal, -1, 0.025, tol=0.004, max_steps=steps)
    for _ in range(5):
        p.step([0, 0, 0], [0, 0, 0], -1)
    return ok


def blobs(mask, min_px):
    n, lab, stats, cents = cv2.connectedComponentsWithStats(mask)
    out = []
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < min_px:
            continue
        ys, xs = np.nonzero(lab == i)
        pts = np.column_stack([xs, ys]).astype(np.float32)
        if len(pts) >= 5:
            (cx, cy), axes, _ang = cv2.fitEllipse(pts)
        else:
            (cx, cy), axes = cents[i], (0.0, 0.0)
        out.append({"c": np.array([cx, cy]), "major": max(axes), "minor": min(axes), "area": int(stats[i, 4])})
    return out


def ray_to_z(K, R, pc, uv, z):
    ray = R @ np.linalg.solve(K, np.array([uv[0], uv[1], 1.0]))
    s = (z - pc[2]) / ray[2]
    return pc + s * ray


def read_angle(p, rng, centre_prior, dist=0.20, save=None, heading=0.0, px_offset=0.0):
    """The rotor's angle (rad, world frame, hole 0's direction from the axis) from one wrist image, or None.
    Also returns diagnostics. `px_offset` (probe 8: 0.5) turns a blob's pixel-index centre into the published
    intrinsics' convention (a pixel's centre at index + 0.5); probe 6 ran without it, which put the fitted centre
    about 0.8 mm off per axis at 0.20 m and left the angle unchanged."""
    m, d = p.model, p.data
    top_z, r_holes = S.rotor_top(p.a), S.hole_r(p.a)
    target = np.array([centre_prior[0], centre_prior[1], top_z])
    reached = aim_down(p, target, dist, heading)
    ww, wh = B.WRIST["width"], B.WRIST["height"]
    K = intrinsics(B.WRIST["fovy"], ww, wh)
    img = render(m, d, "r_eye_in_hand", ww, wh, rng)
    if save:
        cv2.imwrite(save, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    R, pc = cam_pose(d, m.camera("r_eye_in_hand").id)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    v = hsv[:, :, 2]
    dark = (v < 70).astype(np.uint8) * 255
    light = cv2.dilate((v > 120).astype(np.uint8) * 255, np.ones((5, 5), np.uint8))
    bore_px = BORE_D * K[0, 0] / dist                # an opening's major axis in pixels, about
    holes, caps = [], 0
    for b in blobs(dark, 6):
        if not (0.5 * bore_px <= b["major"] <= 1.6 * bore_px and b["minor"] >= 0.4 * b["major"]):
            continue                                     # the knob bar (long, thin), shadows, the chamber's corners
        x, y = int(b["c"][0] + 0.5), int(b["c"][1] + 0.5)
        if not (0 <= x < ww and 0 <= y < wh) or not light[max(0, y - 6):y + 7, max(0, x - 6):x + 7].any():
            continue                                     # an opening sits in the light rotor
        w = ray_to_z(K, R, pc, b["c"] + px_offset, top_z)
        if abs(np.linalg.norm(w[:2] - centre_prior) - r_holes) > 0.02:
            continue                                     # not on the hole circle
        holes.append(w[:2])
    # a loaded hole shows its tube's blue cap instead: the cap's centre sits CAP_ALONG up the hole's axis (up and
    # in), so its ray meets the plane that high and the point moves out by as much
    for b in blobs(cv2.inRange(hsv, (100, 90, 40), (130, 255, 255)), 6):
        if not (0.5 * bore_px <= b["major"] <= 1.8 * bore_px):
            continue
        w = ray_to_z(K, R, pc, b["c"] + px_offset, top_z + CAP_ALONG * math.cos(C.TILT))
        dv = w[:2] - np.asarray(centre_prior, float)
        w2 = w[:2] + dv / np.linalg.norm(dv) * CAP_ALONG * math.sin(C.TILT)
        if abs(np.linalg.norm(w2 - centre_prior) - r_holes) > 0.02:
            continue
        holes.append(w2)
        caps += 1
    green = cv2.inRange(hsv, (45, 80, 60), (85, 255, 255))
    marks = blobs(green, 4)
    diag = {"reached": bool(reached), "holes_seen": len(holes), "caps_seen": caps, "mark_seen": bool(marks)}
    if len(holes) < 3 or not marks:
        return None, diag
    H = np.array(holes)
    # centre: least squares circle of radius HOLE_RADIUS_AT through the points, started at the prior
    c = np.asarray(centre_prior, float).copy()
    for _ in range(10):
        dv = H - c
        r = np.linalg.norm(dv, axis=1)
        J = -dv / r[:, None]
        res = r - r_holes
        c -= np.linalg.lstsq(J, res, rcond=None)[0]
    ang = np.arctan2(H[:, 1] - c[1], H[:, 0] - c[0])
    base = math.atan2(np.mean(np.sin(C.NHOLES * ang)), np.mean(np.cos(C.NHOLES * ang))) / C.NHOLES
    mk = max(marks, key=lambda b: b["area"])
    # the mark sits on the bowl at MARK_R, about 0.025 m above the entrances' plane
    mw = ray_to_z(K, R, pc, mk["c"] + px_offset, top_z + C.MARK_R - C.HOLE_RADIUS_AT)
    mang = math.atan2(mw[1] - c[1], mw[0] - c[0])
    k = round(((mang - base) % (2 * math.pi)) / STEP) % C.NHOLES
    theta = base + k * STEP
    diag.update(centre=c.tolist(), mark_off_deg=round(math.degrees(C.wrap(mang - theta)), 2))
    return theta, diag


def true_angle(p):
    """The rotor's angle in the same convention (hole 0's direction), from the simulation (for errors only)."""
    b = p.model.body("rotor").id
    x = p.data.xmat[b].reshape(3, 3)[:, 0]
    return math.atan2(x[1], x[0])
