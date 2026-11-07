"""A camera-only relative-pose estimator for the docking camera (a maintainer's tool; the grader never runs it).

It uses only what an agent has: a frame, the star tracker's yaw, and the numbers of the agent's documents
(environment/setup/payload/given/camera.md and target.md, read from their JSON blocks, never from camera.py). It is
the completability evidence's estimator (check_perception.py), deliberately simple and classical:

  * LEDs: pixels at 250 DN or more, connected components, centroids.
  * The docking target (DT): the camera sits at the DT's height (z = -0.115 m) and the motion is planar, so the
    standoff LED S lies on the image's centre row and each column of flush LEDs (0.025 m above and below it) is a pair
    of blobs symmetric about that row in the same image column. Two such pairs and S identify the DT.
  * Pose from the DT, closed form then Gauss-Newton: a column's vertical spread gives its depth along the boresight
    (fy dz / dv, dz the column's height), its column its lateral offset; the two columns' camera-frame points against their port-frame
    places give the camera's planar pose (x, y, boresight yaw) in the port frame; Gauss-Newton on the reprojection
    error of the five LEDs refines it (S breaks the yaw / lateral ambiguity).
  * With the star tracker's yaw: the target's phase phi = yaw - (boresight yaw - pi), the craft's position in the
    inertial frame.
  * The boom-tip target (BT): with the camera's pose from the DT, the three BT LEDs' image places are a function of the
    port's radius alone: a 1-D search, then Gauss-Newton, gives r_port.
  * Far field: the BT triangle (0.144 m wide) is resolved farther out than the DT's 0.040 m square. The camera's
    height below the boom's axis gives BT1's and BT2's depths (fy 0.115 / dv), their columns their lateral offsets:
    the camera's planar pose relative to the boom tip in closed form, Gauss-Newton with BT3. Its place in the port
    frame needs the port's radius: the boom's encoder, its polarity calibrated where the DT and the BT are both seen.
    (The drum's silhouette is no help at this exposure: its unlit side is as dark as the sky, and the boom stands out
    past the limb when the port is near it.)
"""

import json
import math
import os
import re

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
GIVEN = os.path.join(os.path.dirname(os.path.dirname(HERE)), "environment", "setup", "payload", "given")


def _block(name):
    text = open(os.path.join(GIVEN, name)).read()
    return json.loads(re.findall(r"```json\n(.*?)\n```", text, re.S)[-1])


CAM = _block("camera.md")
TGT = _block("target.md")
FX, FY = CAM["intrinsics"]["fx"], CAM["intrinsics"]["fy"]
CX, CY = CAM["intrinsics"]["cx"], CAM["intrinsics"]["cy"]
W, H = CAM["image"]["width"], CAM["image"]["height"]
LX, LY, LZ = CAM["mount_body_m"]
DT = TGT["docking_target"]["leds_m"]
BT = TGT["boom_tip_target"]["leds_above_apex_m"]
APEX_OFF = float(TGT["boom_tip_target"]["drogue_apex_radius_m"].split("-")[1])
DRUM_R = TGT["drum_radius_m"]
SAT = 250
# Gates on a fix, from the frame alone: the reprojection RMS of the LEDs used (a right match of saturated discs fits to
# a few hundredths of a pixel; glints that happen to mimic the pattern do not), and the camera in front of the target.
GATE_DT_PX = 0.2
GATE_BT_PX = 0.3


def wrap(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def blobs(img):
    """[(u, v, area)] of the saturated components."""
    import cv2
    mask = (img >= SAT).astype(np.uint8)
    n, lab, stats, cents = cv2.connectedComponentsWithStats(mask, connectivity=8)
    return [(float(cents[i][0]), float(cents[i][1]), int(stats[i][cv2.CC_STAT_AREA])) for i in range(1, n)]


def _tol(area):
    return 1.0 + 0.15 * math.sqrt(area)


def find_dt(bl):
    """{"DT1": (u, v), ...} of the docking target's five LEDs, or None (vectorised over the blobs)."""
    if len(bl) < 5:
        return None
    B = np.asarray(bl, float)
    u, v, a = B[:, 0], B[:, 1], B[:, 2]
    tol = 1.0 + 0.15 * np.sqrt(a)
    top, bot = np.nonzero(v < CY)[0], np.nonzero(v > CY)[0]
    if not len(top) or not len(bot):
        return None
    T, Bo = np.meshgrid(top, bot, indexing="ij")
    t = np.maximum(tol[T], tol[Bo])
    ratio = a[T] / np.maximum(a[Bo], 1)
    ok = (np.abs(u[T] - u[Bo]) <= t) & (np.abs(v[T] + v[Bo] - 2 * CY) <= 1.5 * t) & (ratio >= 0.5) & (ratio <= 2.0)
    pi, pj = T[ok], Bo[ok]
    if len(pi) < 2 or len(pi) > 400:
        return None
    pu = 0.5 * (u[pi] + u[pj])
    spread = v[pj] - v[pi]
    centre = np.nonzero(np.abs(v - CY) <= tol)[0]
    if not len(centre):
        return None
    best = None
    order = np.argsort(pu)
    for x in range(len(order)):
        A = order[x]
        for y in range(x + 1, len(order)):
            Bq = order[y]
            if len({pi[A], pj[A], pi[Bq], pj[Bq]}) < 4:
                continue
            sep = pu[Bq] - pu[A]
            sp = 0.5 * (spread[A] + spread[Bq])
            if not (0.25 * sp <= sep <= 4.0 * sp):
                continue
            mid = 0.5 * (pu[A] + pu[Bq])
            c = centre[(np.abs(u[centre] - mid) <= 1.5 * sep) & ~np.isin(centre, [pi[A], pj[A], pi[Bq], pj[Bq]])]
            if not len(c):
                continue
            sc = np.abs(u[c] - mid) / sep + abs(spread[A] - spread[Bq]) / sp
            k = int(np.argmin(sc))
            if best is None or sc[k] < best[0]:
                best = (float(sc[k]), A, Bq, int(c[k]))
    if best is None:
        return None
    _, A, Bq, k = best
    return {"DT2": (pu[A], v[pj[A]]), "DT4": (pu[A], v[pi[A]]), "DT1": (pu[Bq], v[pj[Bq]]),
            "DT3": (pu[Bq], v[pi[Bq]]), "DTS": (u[k], v[k])}


def _project(pose, P):
    """(u, v) of port-frame points P (n x 3) for a camera at pose (x, y, b): lens at (x, y, LZ), boresight yaw b."""
    x, y, b = pose
    f = np.array([math.cos(b), math.sin(b)])
    r = np.array([math.sin(b), -math.cos(b)])
    d = P[:, :2] - np.array([x, y])
    zc = d @ f
    xc = d @ r
    yc = -(P[:, 2] - LZ)
    return np.stack([CX + FX * xc / zc, CY + FY * yc / zc], 1), zc


def _gn(pose, P, uv, iters=12):
    pose = np.asarray(pose, float)
    for _ in range(iters):
        pr, zc = _project(pose, P)
        if np.any(zc <= 0.01):
            return None
        res = (pr - uv).reshape(-1)
        J = np.zeros((res.size, 3))
        for k in range(3):
            dp = np.zeros(3)
            dp[k] = 1e-6
            J[:, k] = ((_project(pose + dp, P)[0] - uv).reshape(-1) - res) / 1e-6
        step = np.linalg.lstsq(J, -res, rcond=None)[0]
        pose = pose + step
        if np.max(np.abs(step)) < 1e-9:
            break
    return pose


def pose_from_dt(dt):
    """The camera's planar pose (x, y, boresight yaw) in the port frame from the five DT LEDs, and the RMS residual."""
    cols = {}
    for name_top, name_bot, py in (("DT3", "DT1", DT["DT1"][1]), ("DT4", "DT2", DT["DT2"][1])):
        (ut, vt), (ub, vb) = dt[name_top], dt[name_bot]
        dz = DT[name_bot][2] - DT[name_top][2]          # -0.040: bottom LED lower
        zc = FY * (-dz) / (vb - vt)
        xc = (0.5 * (ut + ub) - CX) * zc / FX
        cols[py] = (xc, zc)
    (xa, za), (xb, zb) = cols[DT["DT1"][1]], cols[DT["DT2"][1]]
    pa = np.array(DT["DT1"][:2])
    pb = np.array(DT["DT2"][:2])
    # P = C + zc f + xc r, f = (cos b, sin b), r = (sin b, -cos b): the camera-frame vector (dx, dz) maps to dP
    dP = pa - pb
    ang_cam = math.atan2(xa - xb, za - zb)                # in the (f, r) basis: dP = dz f + dx r
    # dP = dz (cos b, sin b) + dx (sin b, -cos b) = |d| (cos(b - ang_cam), sin(b - ang_cam))
    b = math.atan2(dP[1], dP[0]) + ang_cam
    f = np.array([math.cos(b), math.sin(b)])
    r = np.array([math.sin(b), -math.cos(b)])
    C = pa - za * f - xa * r
    names = ("DT1", "DT2", "DT3", "DT4", "DTS")
    P = np.array([DT[n] for n in names], float)
    uv = np.array([dt[n] for n in names], float)
    pose = _gn((C[0], C[1], b), P, uv)
    if pose is None:
        return None, None
    res = _project(pose, P)[0] - uv
    return pose, float(np.sqrt(np.mean(np.sum(res ** 2, 1))))


def accept_dt(pose, resid):
    """A DT fix an estimator keeps: residual under the gate, the lens in front of S's face, facing the target."""
    return (pose is not None and resid is not None and resid < GATE_DT_PX and pose[0] > DT["DTS"][0]
            and abs(wrap(pose[2] - math.pi)) < 1.4)


def craft_from_camera(pose, star_yaw):
    """(craft x, y in the inertial frame, target phase) from the camera's pose in the port frame and the star tracker."""
    x, y, b = pose
    psi = wrap(b - math.pi)                               # the craft's yaw relative to the port frame
    phi = wrap(star_yaw - psi)
    c, s = math.cos(psi), math.sin(psi)
    craft_p = np.array([x - (c * LX - s * LY), y - (s * LX + c * LY)])   # lens = craft + R(psi) (LX, LY)
    cp, sp = math.cos(phi), math.sin(phi)
    return np.array([cp * craft_p[0] - sp * craft_p[1], sp * craft_p[0] + cp * craft_p[1]]), phi


def port_radius_from_bt(pose, bl, dt_names=()):
    """(r_port, the LEDs used, their reprojection RMS) from the boom-tip LEDs given the camera's pose in the port frame,
    or (None, 0, None)."""
    cands = np.array([(u, v) for (u, v, a) in bl], float)
    if len(cands) == 0:
        return None, 0, None
    names = ("BT1", "BT2", "BT3")

    def pts(rp):
        return np.array([[rp - APEX_OFF + BT[n][0], BT[n][1], BT[n][2]] for n in names], float)

    best = None
    for rp in np.linspace(0.95, 1.33, 77):
        pr, zc = _project(np.asarray(pose), pts(rp))
        if np.any(zc <= 0.02):
            continue
        cost, used = 0.0, 0
        for k in range(3):
            if not (0 <= pr[k, 0] < W and 0 <= pr[k, 1] < H):
                continue
            d = np.min(np.hypot(cands[:, 0] - pr[k, 0], cands[:, 1] - pr[k, 1]))
            cost += min(d, 30.0)
            used += 1
        if used >= 2 and (best is None or cost / used < best[0]):
            best = (cost / used, rp)
    if best is None or best[0] > 8.0:
        return None, 0, None
    rp = best[1]
    for _ in range(10):                                    # Gauss-Newton on the matched LEDs
        pr, zc = _project(np.asarray(pose), pts(rp))
        res, jac, used = [], [], 0
        pr2 = _project(np.asarray(pose), pts(rp + 1e-6))[0]
        for k in range(3):
            if not (0 <= pr[k, 0] < W and 0 <= pr[k, 1] < H):
                continue
            j = int(np.argmin(np.hypot(cands[:, 0] - pr[k, 0], cands[:, 1] - pr[k, 1])))
            if math.hypot(*(cands[j] - pr[k])) > 6.0:
                continue
            res.extend(pr[k] - cands[j])
            jac.extend((pr2[k] - pr[k]) / 1e-6)
            used += 1
        if used < 2:
            return None, 0, None
        res, jac = np.array(res), np.array(jac)
        if float(jac @ jac) < 1e-12:
            break
        step = -float(jac @ res) / float(jac @ jac)
        rp += step
        if abs(step) < 1e-9:
            break
    return rp, used, float(np.sqrt(np.mean(res.reshape(-1, 2) ** 2) * 2))


def find_bt(bl):
    """{"BT1": (u, v), "BT2": .., "BT3": ..} of the boom-tip LEDs, or None. The camera sits 0.115 m below the boom's
    axis: BT1 and BT2 (on the axis's height, 0.144 m apart) lie above the centre row by fy 0.115 / Z, BT3 (0.072 m
    below the axis) by fy 0.043 / Z, between them. Vectorised over the blob pairs."""
    if len(bl) < 3:
        return None
    B = np.asarray(bl, float)
    up = B[B[:, 1] < CY - 1.0]
    if len(up) < 3 or len(up) > 600:
        return None
    u, v, a = up[:, 0], up[:, 1], up[:, 2]
    h = CY - v
    z = FY * 0.115 / h
    x = (u - CX) * z / FX
    I, J = np.meshgrid(np.arange(len(up)), np.arange(len(up)), indexing="ij")
    sep = np.hypot(x[J] - x[I], z[J] - z[I])
    hr = h[I] / h[J]
    ar = a[I] / np.maximum(a[J], 1)
    ok = (u[J] > u[I]) & (hr >= 0.5) & (hr <= 2.0) & (ar >= 0.4) & (ar <= 2.5) & (np.abs(sep - 0.144) <= 0.036)
    best = None
    for i, j in zip(I[ok], J[ok]):
        want = 0.374 * 0.5 * (h[i] + h[j])
        m = (u >= u[i] - 1) & (u <= u[j] + 1)
        m[[i, j]] = False
        if not m.any():
            continue
        k = np.nonzero(m)[0]
        sc = np.abs(h[k] - want) / max(want, 1.0) + abs(sep[i, j] - 0.144) / 0.144
        q = int(np.argmin(sc))
        if best is None or sc[q] < best[0]:
            best = (float(sc[q]), i, j, int(k[q]))
    if best is None or best[0] > 0.6:
        return None
    _, i, j, k = best
    return {"BT2": (u[i], v[i]), "BT1": (u[j], v[j]), "BT3": (u[k], v[k])}   # image right = port +y facing the port


def pose_from_bt(bt, port_radius):
    """The camera's planar pose in the port frame from the three boom-tip LEDs and a port radius (the encoder's, with
    its polarity known), and the RMS residual."""
    names = ("BT1", "BT2", "BT3")
    P = np.array([[port_radius - APEX_OFF + BT[n][0], BT[n][1], BT[n][2]] for n in names], float)
    pts = {}
    for n, row in zip(names[:2], P[:2]):
        u, v = bt[n]
        zc = FY * (row[2] - LZ) / (CY - v)
        pts[n] = ((u - CX) * zc / FX, zc)
    (xa, za), (xb, zb) = pts["BT1"], pts["BT2"]
    dP = P[0, :2] - P[1, :2]
    b = math.atan2(dP[1], dP[0]) + math.atan2(xa - xb, za - zb)
    f = np.array([math.cos(b), math.sin(b)])
    r = np.array([math.sin(b), -math.cos(b)])
    C = P[0, :2] - za * f - xa * r
    uv = np.array([bt[n] for n in names], float)
    pose = _gn((C[0], C[1], b), P, uv)
    if pose is None:
        return None, None
    res = _project(pose, P)[0] - uv
    return pose, float(np.sqrt(np.mean(np.sum(res ** 2, 1))))
