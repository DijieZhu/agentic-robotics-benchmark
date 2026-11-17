"""Reading a tube's fill in its rack with the wrist camera (probe 8), of the kind an agent could write.

The arm puts the wrist camera `dist` from the tube's axis, `el` above the column's middle and at azimuth `az`
(the camera's side of the tube), looking at it, the image level. In the image (with the environment's pixel
noise): probe 4's column reading (`mass_probe.column_rows`): the topmost red row (the liquid surface's far rim) and
the lowest blue row above it (the cap's near bottom edge) in a band of columns around the tube's axis; their rays
met with those cylinders give the column's height below the cap, hence the volume. The azimuth is one whose line of
sight passes clear of the other tubes, in front and behind (`clear_azimuths`). The rack is a wire rack, so the column shows
below its rim.
"""

import math

import numpy as np

import centrifuge_probe as C
from localize_probe import cam_pose, intrinsics, render
from mass_probe import L_BODY_OFF, LIQ_BASE, ML, column_rows

RED_S_MIN = 30                  # the column seen through the tube and the wire rack is pale (saturation 40-60)
from tube_insert_probe import B

BAND = 2                        # px, half the band of columns around the axis


def aim_at(p, target, dist, az, el, steps=250):
    """Servo the hand so the wrist camera sits `dist` from `target` at azimuth `az` and elevation `el` (rad),
    its optical axis through `target`, the image level. Returns whether the servo converged."""
    d = p.data
    wcam = p.model.camera("r_eye_in_hand").id
    Rs = d.site_xmat[p.site].reshape(3, 3)
    C_s = Rs.T @ d.cam_xmat[wcam].reshape(3, 3)
    o_s = Rs.T @ (d.cam_xpos[wcam] - d.site_xpos[p.site])
    off = np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
    cam = np.asarray(target, float) + dist * off
    z_c = -off                                           # the optical axis (OpenCV: z forward, y down)
    x_c = np.cross([0.0, 0.0, -1.0], z_c)
    x_c /= np.linalg.norm(x_c)
    R_cv = np.column_stack([x_c, np.cross(z_c, x_c), z_c])
    R_goal = R_cv @ np.diag([1.0, -1.0, -1.0]) @ C_s.T
    goal = cam - R_goal @ o_s
    ok = p.servo(goal, R_goal, -1, 0.025, tol=0.003, max_steps=steps)
    for _ in range(5):
        p.step([0, 0, 0], [0, 0, 0], -1)
    p.aim_err = (float(np.linalg.norm(p.ee() - goal)), float(np.linalg.norm(C.rotvec(R_goal @ p.ee_R().T))))
    return bool(ok)


def read_fill(p, rng, axis_xy, mid_z, az, el, dist=0.15, save=None):
    """The fill (mL) of the tube standing at axis_xy (an estimate), from one wrist image; None if the column or
    the cap is not found. Also returns diagnostics."""
    a = p.a
    reached = aim_at(p, [axis_xy[0], axis_xy[1], mid_z], dist, az, el)
    ww, wh = B.WRIST["width"], B.WRIST["height"]
    K = intrinsics(B.WRIST["fovy"], ww, wh)
    img = render(p.model, p.data, "r_eye_in_hand", ww, wh, rng)
    if save:
        import cv2
        cv2.imwrite(save, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    R, pc = cam_pose(p.data, p.model.camera("r_eye_in_hand").id)
    u = K @ (R.T @ (np.array([axis_xy[0], axis_xy[1], mid_z]) - pc))
    col = u[0] / u[2]
    diag = {"reached": reached, "col_px": round(float(col), 1), "aim_err_mm": round(1000 * p.aim_err[0], 1),
            "aim_err_deg": round(math.degrees(p.aim_err[1]), 1)}
    if not (BAND <= col < ww - BAND) or u[2] <= 0:
        return None, diag
    got = column_rows(img, col, BAND)
    if got is None:
        return None, diag
    v_liq, v_cap = got
    r_in = a.tube_d / 2 - 0.0012
    # the topmost red row is the liquid surface's far rim, the lowest blue row the cap's near bottom edge: meet
    # each ray with that cylinder (its far and near side), not with the axis (a bias growing with the elevation)
    z_liq = ray_cylinder_z(K, R, pc, col, v_liq, axis_xy, r_in, far=True)
    z_cap = ray_cylinder_z(K, R, pc, col, v_cap, axis_xy, a.tube_d / 2 + 0.001, far=False)
    if z_liq is None or z_cap is None:
        return None, diag
    h = z_liq - z_cap + (a.tube_len - L_BODY_OFF) - LIQ_BASE
    return h * math.pi * r_in ** 2 / ML, diag


def ray_cylinder_z(K, R, pc, u, v, axis_xy, radius, far):
    """Height where the pixel's ray meets the vertical cylinder of `radius` about axis_xy (its far or near side);
    None if it misses."""
    ray = R @ np.linalg.solve(K, np.array([u, v, 1.0]))
    w = pc[:2] - np.asarray(axis_xy, float)
    qa, qb, qc = ray[:2] @ ray[:2], 2 * (w @ ray[:2]), w @ w - radius ** 2
    disc = qb * qb - 4 * qa * qc
    if disc < 0:
        return None
    s = (-qb + (1 if far else -1) * math.sqrt(disc)) / (2 * qa)
    return float(pc[2] + s * ray[2])


def clear_azimuths(target_xy, others_xy, candidates_deg, min_gap=0.02, reach=0.35):
    """Of the candidate azimuths (deg), those whose line of sight through the target (both ways, `reach` m) passes
    every other tube's axis by min_gap or more: no tube in front of the target or behind it (the tubes are
    translucent: a column behind shows through)."""
    out = []
    t = np.asarray(target_xy, float)
    for az in candidates_deg:
        u = np.array([math.cos(math.radians(az)), math.sin(math.radians(az))])
        ok = True
        for o in others_xy:
            dv = np.asarray(o, float) - t
            along = float(dv @ u)
            if abs(along) <= reach and abs(float(dv[0] * u[1] - dv[1] * u[0])) < min_gap:
                ok = False
                break
        if ok:
            out.append(az)
    return out


def read_rack(p, rng, tubes_xy, mid_z, cam="rack_cam", width=640, height=480, fovy=50.0, save=None):
    """The fills (mL, None where not found) of the tubes standing at tubes_xy (estimates), from one image of the
    fixed rack camera (probe 8: the wrist camera's side views put the hand among the tubes). Per tube, the column
    and the cap in a band of columns around its axis, met with their cylinders as in `read_fill`."""
    a = p.a
    K = intrinsics(fovy, width, height)
    img = render(p.model, p.data, cam, width, height, rng)
    if save:
        import cv2
        cv2.imwrite(save, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    R, pc = cam_pose(p.data, p.model.camera(cam).id)
    r_in = a.tube_d / 2 - 0.0012
    out = []
    for xy in tubes_xy:
        u = K @ (R.T @ (np.array([xy[0], xy[1], mid_z]) - pc))
        col = u[0] / u[2]
        got = column_rows_pale(img, col, BAND) if BAND <= col < width - BAND else None
        if got is None:
            out.append(None)
            continue
        z_liq = ray_cylinder_z(K, R, pc, col, got[0] + 0.5, xy, r_in, far=True)     # index edges -> the
        z_cap = ray_cylinder_z(K, R, pc, col, got[1] + 0.5, xy, a.tube_d / 2 + 0.001, far=False)  # K's convention
        if z_liq is None or z_cap is None:
            out.append(None)
            continue
        h = z_liq - z_cap + (a.tube_len - L_BODY_OFF) - LIQ_BASE
        out.append(h * math.pi * r_in ** 2 / ML)
    return out


def column_rows_pale(img, col_px, half_w, s_min=RED_S_MIN):
    """`mass_probe.column_rows` with a lower saturation floor for the red: the topmost red row and the lowest
    blue row above it, in a band of columns around the axis."""
    import cv2
    w = img.shape[1]
    x0, x1 = int(max(0, col_px - half_w)), int(min(w, col_px + half_w + 1))
    hsv = cv2.cvtColor(img[:, x0:x1], cv2.COLOR_RGB2HSV)
    red = cv2.inRange(hsv, (0, s_min, 50), (8, 255, 255)) | cv2.inRange(hsv, (165, s_min, 50), (180, 255, 255))
    blue = cv2.inRange(hsv, (100, 90, 40), (130, 255, 255))
    min_px = max(2, (x1 - x0) // 3)
    red_rows = np.nonzero((red > 0).sum(1) >= min_px)[0]
    blue_rows = np.nonzero((blue > 0).sum(1) >= min_px)[0]
    if len(red_rows) == 0 or len(blue_rows) == 0:
        return None
    top_red = int(red_rows.min())
    above = blue_rows[blue_rows < top_red]
    if len(above) == 0:
        return None
    return top_red - 0.5, int(above.max()) + 0.5
