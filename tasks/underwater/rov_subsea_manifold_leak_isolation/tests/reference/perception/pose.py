"""Camera model and plate-code pose estimation (numpy only): 4-DoF (attitude from the AHRS) and 6-DoF (attitude-free)
PnP on the decoded square codes.  Port of the G2 pilot's vision (proto/g2_grasp_approach/vision.py, sha256 e3ce8839...)
with the panel geometry made explicit (`PanelLayout`, defaults = the finished panel standard) and the camera built
from `vehicle.Manual`.

Frames:
  L = "local level" frame: origin at the vehicle body origin, axes = the navigation frame's axes (AHRS attitude R_nb);
      nothing here depends on the drifting navigation position;
  P = panel-local frame of ONE valve position (the target): origin = stem axis on the panel face, X = the viewer's
      right, Y = up, Z = out of the panel (towards the viewer);
  C = camera, OpenCV axes (x right, y down, z forward).

Panel pose (4-DoF) = (psi, p0): the panel face is vertical (panel standard), Z_P = (cos psi, sin psi, 0) in L, Y_P = up,
X_P = Y_P x Z_P; p0 = the target's stem point in L.  Roll / pitch come from the AHRS, so the 4-DoF solve is robust
with a single small oblique code, but its vertical inherits the AHRS tilt error (1-2 deg during accelerations: 7-17 mm
at 0.5-0.7 m, G2 finding 4).  `codes_pose6` solves the full pose of P in C from the codes alone (attitude-free), and
`tcp_in_panel` turns either into "where is my gripper TCP relative to the target's stem" in panel axes.

    cam = Cam(man, tilt_rad, R_nb, W, H)
    p4 = codes_pose(dets, offsets, cam)            # dets: [(tag, corners)], offsets: {tag: (dcol, drow)} from target
    p6 = codes_pose6(dets, offsets, cam, p4)       # (R_cP, t_c, rms) or None
    e = tcp_in_panel6(man, tilt_rad, p6)          # TCP - stem in P axes (X right, Y up, Z out), attitude-free
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from . import vehicle as VH


@dataclass(frozen=True)
class PanelLayout:
    """Plate code position relative to its stem (panel standard): plate 424 x 200 mm, centre 0.22 m above the stem,
    code = 160 mm outer black square, 20 mm quiet zone at the plate's left edge, printed on a 3 mm plate mounted on
    the panel face (code plane Z = code_z, plate back = code_z - plate_thick); grid pitch 0.60 m for columns and
    rows."""
    plate_w: float = 0.424
    plate_h: float = 0.200
    plate_above_stem: float = 0.22
    quiet: float = 0.020
    marker: float = 0.160
    code_z: float = 0.003
    plate_thick: float = 0.003
    col_pitch: float = 0.60
    row_pitch: float = 0.60

    @property
    def code_centre(self) -> np.ndarray:
        return np.array([-self.plate_w / 2 + self.quiet + self.marker / 2, self.plate_above_stem])

    def marker_corners(self) -> np.ndarray:
        """(4, 2) code TL, TR, BR, BL (X right, Y up) in P of its own stem."""
        c, h = self.code_centre, self.marker / 2
        return np.array([[c[0] - h, c[1] + h], [c[0] + h, c[1] + h], [c[0] + h, c[1] - h], [c[0] - h, c[1] - h]])

    def plate_corners(self) -> np.ndarray:
        cx, cy, w, h = 0.0, self.plate_above_stem, self.plate_w / 2, self.plate_h / 2
        return np.array([[cx - w, cy + h], [cx + w, cy + h], [cx + w, cy - h], [cx - w, cy - h]])


PANEL = PanelLayout()


# ----------------------------------------------------------------------------------------------------- camera
class Cam:
    """Camera pose in L for one frame (the manual's mount + the frame's tilt reading + the AHRS attitude R_nb)."""

    def __init__(self, man: VH.Manual, tilt_rad: float, R_nb, W: int = 640, H: int = 480, p_body=None):
        self.man = man
        self.K = man.intrinsics(W, H)
        self.W, self.H = int(W), int(H)
        self.tilt = float(tilt_rad)
        p_B, R_Bc = man.camera_in_body(tilt_rad)
        self.R_nb = np.asarray(R_nb, float)
        o = np.zeros(3) if p_body is None else np.asarray(p_body, float)
        self.o = o                                  # body origin in L (0 = the L frame's own origin)
        self.p = o + self.R_nb @ p_B                # camera centre in L
        self.R = self.R_nb @ R_Bc                   # C -> L

    def project(self, X_L):
        X = np.atleast_2d(np.asarray(X_L, float))
        Pc = (X - self.p) @ self.R                  # rows: R^T (x - p)
        z = Pc[:, 2]
        zz = np.where(z > 1e-4, z, 1e-4)
        u = self.K[0, 0] * Pc[:, 0] / zz + self.K[0, 2]
        v = self.K[1, 1] * Pc[:, 1] / zz + self.K[1, 2]
        return np.stack([u, v], 1), z

    def rays(self, u, v) -> np.ndarray:
        """Unit viewing rays (L axes) through continuous pixel coordinates (u, v)."""
        u, v = np.atleast_1d(np.asarray(u, float)), np.atleast_1d(np.asarray(v, float))
        d = np.stack([(u - self.K[0, 2]) / self.K[0, 0], (v - self.K[1, 2]) / self.K[1, 1], np.ones_like(u)], 1)
        d = d @ self.R.T
        return d / np.linalg.norm(d, axis=1, keepdims=True)


def panel_R(psi: float) -> np.ndarray:
    Z = np.array([math.cos(psi), math.sin(psi), 0.0])
    Y = np.array([0.0, 0.0, 1.0])
    X = np.cross(Y, Z)
    return np.stack([X, Y, Z], 1)                   # P -> L


def to_L(psi, p0, P_local):
    return np.asarray(p0, float) + np.atleast_2d(P_local) @ panel_R(psi).T


# ----------------------------------------------------------------------------------------------------- codes
def code_obj(off, panel: PanelLayout = PANEL) -> np.ndarray:
    """(4, 3) corners of the code of the plate at grid offset off = (dcol, drow) from the target, in the target's P."""
    c = panel.marker_corners()
    return np.concatenate([c + np.array([off[0] * panel.col_pitch, off[1] * panel.row_pitch]),
                           np.full((4, 1), panel.code_z)], 1)


def homography(src, dst) -> np.ndarray:
    A = []
    for (x, y), (u, v) in zip(src, dst):
        A.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
        A.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])
    _, _, Vt = np.linalg.svd(np.asarray(A, float))
    Hm = Vt[-1].reshape(3, 3)
    return Hm / Hm[2, 2]


def pose_from_h(Hm, K):
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


def _init_from_code(corners, off, cam, panel):
    loc = panel.marker_corners() + np.array([off[0] * panel.col_pitch, off[1] * panel.row_pitch])
    R_cP, t = pose_from_h(homography(loc, corners), cam.K)
    R_LP = cam.R @ R_cP
    psi = math.atan2(R_LP[1, 2], R_LP[0, 2])
    p0 = cam.p + cam.R @ t                     # initial guess (the code plane's origin; _solve4 refines)
    return psi, p0


def _solve4(cam, obj, img, psi, p0, iters=10, huber_px=2.0):
    x = np.array([psi, *p0], float)

    def res(x):
        u, z = cam.project(to_L(x[0], x[1:], obj))
        return (u - img).reshape(-1), z
    for _ in range(iters):
        r, z = res(x)
        a = np.abs(r)
        w = np.where(a <= huber_px, 1.0, huber_px / np.maximum(a, 1e-9))
        J = np.zeros((r.size, 4))
        for k, h in enumerate((1e-5, 1e-6, 1e-6, 1e-6)):
            dx = np.zeros(4)
            dx[k] = h
            J[:, k] = (res(x + dx)[0] - r) / h
        Jw = J * w[:, None]
        step = np.linalg.lstsq(Jw.T @ J + 1e-9 * np.eye(4), -(Jw.T @ r), rcond=None)[0]
        x = x + step
        if np.max(np.abs(step[1:])) < 1e-6 and abs(step[0]) < 1e-7:
            break
    r, z = res(x)
    return x[0], x[1:].copy(), float(np.sqrt(np.mean(r ** 2))), r.reshape(-1, 2), z


def code_size_px(corners) -> float:
    c = np.asarray(corners, float)
    return float(np.mean(np.linalg.norm(c - np.roll(c, 1, 0), axis=1)))


def single_code_pose(corners, cam, panel: PanelLayout = PANEL) -> dict:
    """4-DoF pose of one code alone (offset 0): its own stem point and the panel heading."""
    c = np.asarray(corners, float)
    psi, p0 = _init_from_code(c, (0, 0), cam, panel)
    psi, p0, rms, _, z = _solve4(cam, code_obj((0, 0), panel), c, psi, p0)
    return dict(psi=float(psi), p0=p0, rms=rms, ok=bool(np.all(z > 0.05) and rms < 3.0))


def codes_pose(dets, offsets, cam, min_px: float = 18.0, panel: PanelLayout = PANEL):
    """4-DoF pose over all decoded codes with known grid offsets from the target.  dets: [(tag, corners)];
    offsets: {tag: (dcol, drow)} (pitches, X right / Y up).  Returns dict(psi, p0, rms, n, tags) or None."""
    use = [(t, np.asarray(c, float)) for t, c in dets if t in offsets and code_size_px(c) >= min_px]
    if not use:
        return None
    use.sort(key=lambda x: -code_size_px(x[1]))
    psi, p0 = _init_from_code(use[0][1], offsets[use[0][0]], cam, panel)
    for _ in range(2):
        obj = np.concatenate([code_obj(offsets[t], panel) for t, _ in use])
        img = np.concatenate([c for _, c in use])
        psi, p0, rms, r, z = _solve4(cam, obj, img, psi, p0)
        per = np.sqrt(np.mean(r.reshape(len(use), 4, 2) ** 2, axis=(1, 2)))
        bad = per > max(3.0, 3.0 * np.median(per))
        if not bad.any() or len(use) == 1:
            break
        use = [u for u, b in zip(use, bad) if not b]
    if not np.all(z > 0.05):
        return None
    return dict(psi=float(psi), p0=p0, rms=rms, n=len(use), tags=[t for t, _ in use])


def rodrigues(w) -> np.ndarray:
    th = float(np.linalg.norm(w))
    if th < 1e-12:
        return np.eye(3)
    k = np.asarray(w, float) / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * K @ K


def plate_obj(off, panel: PanelLayout = PANEL) -> np.ndarray:
    """(4, 3) plate outline corners (TL, TR, BR, BL) of the plate at grid offset off, in the target's P."""
    c = panel.plate_corners()
    return np.concatenate([c + np.array([off[0] * panel.col_pitch, off[1] * panel.row_pitch]),
                           np.full((4, 1), panel.code_z)], 1)


def plate_edges_from_code(rgb_or_gray, code_corners, panel: PanelLayout = PANEL, min_cell_px: float = 3.0,
                          n_samples: int = 24, max_line_rms_px: float = 0.6, min_pts: int = 6, exclude=None):
    """Sub-pixel edge points of the outline of the plate that carries a decoded code, side by side (TL-TR, TR-BR,
    BR-BL, BL-TL): the code's homography predicts the outline (same plane); along each side, where it lies inside
    the frame (and outside `exclude`, e.g. the own-gripper mask), the white-plate / darker-panel edge is located on
    the profile across it (mid-level crossing, as imgproc.refine_quad); a side is kept if >= min_pts points fit a line
    within max_line_rms_px after one 1-px outlier cut.  Returns [(side, points (n, 2))].  The outline's wider baseline
    (424 x 200 mm against the 160 mm code) tightens the rotation of a pose from one or two codes; partial sides work
    when the plate is cut by the frame."""
    from . import imgproc as IP
    g = IP.to_gray(rgb_or_gray) if np.ndim(rgb_or_gray) == 3 else np.asarray(rgb_or_gray, np.float32)
    H, W = g.shape
    cc = np.asarray(code_corners, float)
    cell_px = code_size_px(cc) / 8.0
    if cell_px < min_cell_px:
        return []
    Hm = homography(panel.marker_corners(), cc)
    loc = panel.plate_corners()
    q = np.array([(Hm @ np.r_[p, 1.0])[:2] / (Hm @ np.r_[p, 1.0])[2] for p in loc])
    band = max(2.0, 0.5 * cell_px)
    s = np.arange(-band, band + 1e-9, 0.25)
    ins_sel, out_sel = s < -0.3 * band, s > 0.3 * band
    ctr = q.mean(0)
    ts = (np.arange(n_samples) + 0.5) / n_samples
    out = []
    for k in range(4):
        p0, p1 = q[k], q[(k + 1) % 4]
        L = float(np.hypot(*(p1 - p0)))
        if L < 8:
            continue
        t_dir = (p1 - p0) / L
        n = np.array([t_dir[1], -t_dir[0]])
        if np.dot(n, (p0 + p1) / 2 - ctr) < 0:
            n = -n                                            # outward
        base = p0[None, :] + ts[:, None] * (p1 - p0)[None, :]
        m = band + 2.0
        keep = (base[:, 0] > m) & (base[:, 0] < W - m) & (base[:, 1] > m) & (base[:, 1] < H - m)
        # stay clear of the corners (the neighbouring side's edge enters the profile there)
        keep &= (ts * L > 1.5 * band) & ((1 - ts) * L > 1.5 * band)
        if exclude is not None:
            bi = np.clip(base.astype(int), 0, [W - 1, H - 1])
            keep &= ~exclude[bi[:, 1], bi[:, 0]]
        base = base[keep]
        if len(base) < min_pts:
            continue
        uu = base[:, 0:1] + s[None, :] * n[0]
        vv = base[:, 1:2] + s[None, :] * n[1]
        prof = IP.bilinear(g, uu, vv)
        good = ~np.any(np.isnan(prof), 1)
        prof, base = prof[good], base[good]
        if len(base) < min_pts:
            continue
        lo = np.median(prof[:, ins_sel], 1)                   # inside = the bright plate
        hi = np.median(prof[:, out_sel], 1)
        ok = (lo - hi) > 0.02
        mid = 0.5 * (lo + hi)
        sg = (mid[:, None] - prof)                            # < 0 inside (brighter), > 0 outside
        cross = (sg[:, :-1] < 0) & (sg[:, 1:] >= 0)
        pts = []
        for r in np.nonzero(ok)[0]:
            idx = np.nonzero(cross[r])[0]
            if len(idx) == 0:
                continue
            j = idx[np.argmin(np.abs(s[idx]))]
            f = -sg[r, j] / (sg[r, j + 1] - sg[r, j] + 1e-12)
            pts.append(base[r] + (s[j] + f * (s[j + 1] - s[j])) * n)
        if len(pts) < min_pts:
            continue
        P = np.asarray(pts)
        for _ in range(2):
            c = P.mean(0)
            _, _, Vt = np.linalg.svd(P - c)
            nn = Vt[1]
            dist = (P - c) @ nn
            if len(P) >= min_pts and np.max(np.abs(dist)) > 1.0:
                P = P[np.abs(dist) <= 1.0]
                if len(P) < min_pts:
                    break
                continue
            break
        if len(P) < min_pts:
            continue
        c = P.mean(0)
        _, _, Vt = np.linalg.svd(P - c)
        rms = float(np.sqrt(np.mean(((P - c) @ Vt[1]) ** 2)))
        if rms <= max_line_rms_px:
            out.append((k, P))
    return out


def codes_pose6(dets, offsets, cam, pose4, min_px: float = 40.0, iters: int = 8, panel: PanelLayout = PANEL,
                max_rms_px: float = 2.0, max_rot_deg: float = 6.0, edges=None, edge_weight: float = 4.0):
    """Full 6-DoF pose of the target's panel frame P in the CAMERA frame from the decoded codes, initialised by the
    4-DoF solution: P_c = R_cP P + t_c.  Independent of the AHRS attitude.  `edges` = {tag: plate_edges_from_code(...)}
    adds the plate outline sides as point-to-line constraints (each side weighted like edge_weight / 2 corners).
    Returns (R_cP, t_c, rms of the code corners) or None (codes below min_px, rms above max_rms_px, or a rotation away
    from the 4-DoF start above max_rot_deg)."""
    use = [(t, np.asarray(c, float)) for t, c in dets if t in offsets and code_size_px(c) >= min_px]
    if not use or pose4 is None:
        return None
    obj = np.concatenate([code_obj(offsets[t], panel) for t, _ in use])
    img = np.concatenate([c for _, c in use]).reshape(-1)
    n_pt = len(img)
    E = []                         # plate outline sides: front edge A B, back edge, outward normal, points, weight
    for t, sides in (edges or {}).items():
        if t not in offsets or not sides:
            continue
        po = plate_obj(offsets[t], panel)
        pb = po.copy()
        pb[:, 2] = panel.code_z - panel.plate_thick            # the plate's back edges (on the panel face)
        cen = po.mean(0)
        for k, pts in sides:
            pts = np.asarray(pts, float)
            a, b = po[k], po[(k + 1) % 4]
            nk = 0.5 * (a + b) - cen
            nk[2] = 0.0
            nk /= max(np.linalg.norm(nk), 1e-9)               # outward normal of that side (plate plane)
            E.append((a, b, pb[k], pb[(k + 1) % 4], nk, pts, math.sqrt(edge_weight / len(pts))))
    R0 = cam.R.T @ panel_R(pose4["psi"])
    t0 = cam.R.T @ (np.asarray(pose4["p0"], float) - cam.p)
    x = np.zeros(6)
    x[3:] = t0
    K = cam.K

    def proj(R, t_, X):
        Pc = np.atleast_2d(X) @ R.T + t_
        z = np.maximum(Pc[:, 2], 1e-4)
        return np.stack([K[0, 0] * Pc[:, 0] / z + K[0, 2], K[1, 1] * Pc[:, 1] / z + K[1, 2]], 1)

    def res(x):
        R = rodrigues(x[:3]) @ R0
        r = proj(R, x[3:], obj).reshape(-1) - img
        if not E:
            return r
        out = [r]
        cam_P = -R.T @ x[3:]                                  # camera centre in P
        for A, B, Ab, Bb, nk, pts, w in E:
            # silhouette of the plate box: where the camera sees that side face, the outline is the back edge
            if (cam_P - 0.5 * (A + B)) @ nk > 0:
                A, B = Ab, Bb
            ab = proj(R, x[3:], np.stack([A, B]))
            d = ab[1] - ab[0]
            nl = np.array([-d[1], d[0]]) / max(np.linalg.norm(d), 1e-9)
            out.append(w * ((pts - ab[0]) @ nl))
        return np.concatenate(out)
    for _ in range(iters):
        r = res(x)
        J = np.zeros((r.size, 6))
        for k in range(6):
            dx = np.zeros(6)
            dx[k] = 1e-6
            J[:, k] = (res(x + dx) - r) / 1e-6
        try:
            step = np.linalg.lstsq(J, -r, rcond=None)[0]
        except np.linalg.LinAlgError:
            return None
        x = x + step
        if np.max(np.abs(step)) < 1e-8:
            break
    r = res(x)
    rms = float(np.sqrt(np.mean(r[:n_pt] ** 2)))
    R = rodrigues(x[:3]) @ R0
    ang = math.degrees(math.acos(min(1.0, (np.trace(R.T @ R0) - 1) / 2)))
    if rms > max_rms_px or ang > max_rot_deg:
        return None
    return R, x[3:].copy(), rms


def grid_offset(p_tag, pose_target, panel: PanelLayout = PANEL):
    """Integer grid offset (dcol, drow) of a code's stem point from the target stem (pitches) and the raw offset."""
    d = panel_R(pose_target["psi"]).T @ (np.asarray(p_tag, float) - pose_target["p0"])
    return (int(round(d[0] / panel.col_pitch)), int(round(d[1] / panel.row_pitch))), d


# ----------------------------------------------------------------------------------------------------- TCP vs stem
def tcp_in_panel4(man: VH.Manual, cam: Cam, pose4: dict) -> np.ndarray:
    """TCP - target stem in P axes (X right, Y up, Z out) from the 4-DoF pose (inherits the AHRS tilt error)."""
    tcp_L = cam.o + cam.R_nb @ man.tcp_B
    return panel_R(pose4["psi"]).T @ (tcp_L - np.asarray(pose4["p0"], float))


def tcp_in_panel6(man: VH.Manual, tilt_rad: float, pose6) -> np.ndarray:
    """TCP - target stem in P axes from the 6-DoF pose (camera-relative; no attitude used)."""
    R_cP, t_c, _ = pose6
    p_B, R_Bc = man.camera_in_body(tilt_rad)
    tcp_c = R_Bc.T @ (man.tcp_B - p_B)
    return R_cP.T @ (tcp_c - t_c)


def offsets_from_tags(tags, target: str, tag_rc) -> dict:
    """Grid offsets (dcol, drow) of known tags from the target, given a {tag: (col, row)} layout (as-built panel
    drawing or as inferred); rows count upwards, columns to the viewer's right."""
    c0, r0 = tag_rc[target]
    return {t: (tag_rc[t][0] - c0, tag_rc[t][1] - r0) for t in tags if t in tag_rc}


def panel_pose6(rgb, dets, offsets, cam, pose4=None, use_edges: bool = True, exclude=None,
                panel: PanelLayout = PANEL, **kw):
    """The attitude-free pose the programs use: 4-DoF start (if not given), the plate outline edges of every decoded
    code with a known offset (plate_edges_from_code; `exclude` = pixels to ignore, e.g. the own-gripper mask), then
    codes_pose6.  Returns (pose6 or None, pose4, n_edge_sides)."""
    if pose4 is None:
        pose4 = codes_pose(dets, offsets, cam, panel=panel)
    if pose4 is None:
        return None, None, 0
    edges = {}
    if use_edges:
        for t, c in dets:
            if t in offsets:
                edges[t] = plate_edges_from_code(rgb, c, panel=panel, exclude=exclude)
    p6 = codes_pose6(dets, offsets, cam, pose4, panel=panel, edges=edges, **kw)
    return p6, pose4, sum(len(v) for v in edges.values())
