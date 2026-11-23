"""Parcel geometry: faces, UV islands, the carton's rounded box, the mailer's pillow, waybill placement and the
waybill's sheet mesh. Parcel frame = the simulator's (metres, the parcel's own x, y, z; face ids 0:+z 1:-z 2:+x 3:-x
4:+y 5:-y). Visual geometry stays inside the simulator's collision shape: the carton inside its box, the mailer inside
the union of its flattened box and ellipsoid pad; only the waybill sheet sits 0.3 mm above the surface.
"""
from __future__ import annotations

import math

import numpy as np

FACE_NORMALS = np.array([[0, 0, 1], [0, 0, -1], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0]], float)
FACE_NAMES = ("+z", "-z", "+x", "-x", "+y", "-y")
# in-image axes of each face's UV island: u (right) and v (up), eu x ev = outward normal; side faces have v = +z
FACE_UV_AXES = {
    0: ((1, 0, 0), (0, 1, 0)), 1: ((1, 0, 0), (0, -1, 0)),
    2: ((0, 1, 0), (0, 0, 1)), 3: ((0, -1, 0), (0, 0, 1)),
    4: ((-1, 0, 0), (0, 0, 1)), 5: ((1, 0, 0), (0, 0, 1)),
}
LABEL_LIFT = 0.0003            # waybill sheet above the surface (m)
PAPERS = {"100x180": (0.100, 0.180), "76x130": (0.076, 0.130)}
LABEL_MIN_FACE = 0.08          # the simulator's rule


def face_axes(f):
    eu, ev = (np.array(v, float) for v in FACE_UV_AXES[f])
    n = FACE_NORMALS[f]
    assert np.allclose(np.cross(eu, ev), n)
    return eu, ev, n


def sim_tangents(f):
    """The simulator's tangent axes of face f (label_uv runs along these)."""
    ax = int(np.argmax(np.abs(FACE_NORMALS[f])))
    return (ax + 1) % 3, (ax + 2) % 3


def face_half(half, f):
    eu, ev, n = face_axes(f)
    ia, ib, inn = int(np.argmax(np.abs(eu))), int(np.argmax(np.abs(ev))), int(np.argmax(np.abs(n)))
    return half[ia], half[ib], half[inn]


def smooth_pos(u, k):
    """Smooth ramp (continuous slope) <= max(u, 0): 0 for u < 0, u^2 / 2k for 0 <= u < k, u - k/2 beyond."""
    u = np.asarray(u, float)
    return np.where(u <= 0, 0.0, np.where(u < k, u * u / (2 * k), u - k / 2))


# ======================================================================================================== shapes
class Carton:
    """Rounded box inside the collision box; top / bottom flaps may sag inward by ``sag`` (m)."""
    kind = "carton"

    def __init__(self, size, radius=0.004, sag=0.0012, sag_bottom=0.0006):
        self.size = np.asarray(size, float)
        self.h = self.size / 2
        self.r = float(min(radius, 0.45 * self.h.min()))
        self.sag = (float(sag), float(sag_bottom))
        self.faces = (0, 1, 2, 3, 4, 5)

    def label_margin(self):
        return 0.007, 0.002             # nominal, minimum distance from the face edge (m)

    def project(self, f, a, b, lift=0.0):
        """Face-plane coordinates (a along eu, b along ev) -> surface points and normals."""
        eu, ev, n = face_axes(f)
        ha, hb, hn = face_half(self.h, f)
        a, b = np.asarray(a, float), np.asarray(b, float)
        P = n * hn + eu * a[..., None] + ev * b[..., None]
        inner = self.h - self.r
        C = np.clip(P, -inner, inner)
        D = P - C
        L = np.linalg.norm(D, axis=-1, keepdims=True)
        N = D / np.maximum(L, 1e-12)
        S = C + self.r * N
        if f in (0, 1):
            s = self.sag[0 if f == 0 else 1]
            if s > 0:
                w = np.clip(1 - (np.clip(np.abs(a) / (ha - self.r), 0, 1)) ** 2, 0, 1) * \
                    np.clip(1 - (np.clip(np.abs(b) / (hb - self.r), 0, 1)) ** 2, 0, 1)
                S = S - n * (s * w)[..., None]
                # the sag tilts the normal a little
                ga = s * (-2 * a / (ha - self.r) ** 2) * np.clip(1 - (b / (hb - self.r)) ** 2, 0, 1)
                gb = s * (-2 * b / (hb - self.r) ** 2) * np.clip(1 - (a / (ha - self.r)) ** 2, 0, 1)
                inside = (np.abs(a) < ha - self.r) & (np.abs(b) < hb - self.r)
                Nt = n + eu * (ga * inside)[..., None] + ev * (gb * inside)[..., None]
                Nt /= np.linalg.norm(Nt, axis=-1, keepdims=True)
                N = np.where(inside[..., None], Nt, N)
        return S + N * lift, N

    def axis_samples(self, h, step, nb):
        r = self.r
        k = np.arange(nb + 1) / nb
        bev = r * (1 - np.tan(np.pi / 4 * (1 - k)))           # 0 .. r, uniform in bevel angle
        n_in = max(1, int(math.ceil((2 * h - 2 * r) / step)))
        mid = np.linspace(-h + r, h - r, n_in + 1)[1:-1]
        return np.concatenate([-h + bev, mid, (h - bev)[::-1]])

    def face_grid(self, f, quality):
        step, nb = (0.02, 4) if quality == "full" else (0.06, 2)
        ha, hb, _ = face_half(self.h, f)
        return self.axis_samples(ha, step, nb), self.axis_samples(hb, step, nb)

    def island_faces(self):
        return [(f,) + face_half(self.h, f)[:2] for f in self.faces]

    def collision(self):
        return {"type": "box", "half": self.h.tolist()}


class Mailer:
    """Poly mailer: two film sheets welded at a perimeter fin (at z = 0), filled to the collision shape's top.

    Collision shape (the simulator's soft parcel): box half (hx, hy, 0.85 hz) union ellipsoid (0.92 hx, 0.92 hy, hz).
    Top / bottom surface: z = +-t(x, y) with t <= max(box top, ellipsoid top), rising from the fin over a rounded
    flank of width R, and inward-only wrinkle displacement."""
    kind = "mailer"

    # a fuller bag. A narrower seam fin, a much squarer flank (superellipse power 5 over a
    # flank of about half the box height) so the film fills the collision box to within about 8 mm at its rims,
    # gathered creases at the four corners, a seam that droops unevenly, a real step at the sealed flap's free edge.
    def __init__(self, size, rng, fin=0.0028, wrinkle=0.0033, flank_power=7.0, fin_droop=0.0022, flank_scale=0.4):
        self.size = np.asarray(size, float)
        self.h = self.size / 2
        hx, hy, hz = self.h
        self.hb = 0.85 * hz
        self.semi = np.array([0.92 * hx, 0.92 * hy, hz])
        self.fin = float(min(fin, 0.08 * min(hx, hy)))
        self.R = float(min(flank_scale * self.hb + 0.003, 0.25 * min(hx, hy)))
        self.flap = None                 # (face side 0 / 1, end -1 / +1 along x, length m): set by parcel.build
        self.flap_step = 0.00045
        self.flank_power = float(flank_power)
        self.fin_droop = float(min(fin_droop, 0.5 * self.hb))
        self.t_fin = 0.0004
        self.faces = (0, 1)
        self.wr = float(wrinkle)
        # wrinkle field: sum of a few long creases and low-frequency dimples, inward only, per side
        self._wr = []
        for side in range(2):
            creases = []
            for _ in range(int(rng.integers(5, 10))):
                c = rng.uniform(-1, 1, 2) * [hx, hy]
                ang = rng.uniform(0, np.pi)
                ln = rng.uniform(0.25, 0.8) * min(hx, hy) * 2
                wd = rng.uniform(0.006, 0.02)
                dp = rng.uniform(0.3, 1.0)
                creases.append((c, ang, ln, wd, dp))
            dim = [(rng.uniform(-0.8, 0.8, 2) * [hx, hy], rng.uniform(0.02, 0.05), rng.uniform(0.3, 1.0))
                   for _ in range(int(rng.integers(2, 5)))]
            # gathered film at the four corners: two or three short creases running in from each corner
            for sx_ in (-1, 1):
                for sy_ in (-1, 1):
                    diag = math.atan2(-sy_ * hy, -sx_ * hx)
                    for _ in range(int(rng.integers(2, 4))):
                        ln = float(rng.uniform(0.035, 0.075))
                        ang = diag + math.radians(float(rng.uniform(-28, 28)))
                        c = np.array([sx_ * hx, sy_ * hy]) + 0.5 * ln * np.array([math.cos(ang), math.sin(ang)])
                        creases.append((c, ang, ln, float(rng.uniform(0.0025, 0.006)), float(rng.uniform(0.6, 1.1))))
            # puckers where the film is gathered into the seam: two sines along the perimeter, random phases
            lam = rng.uniform(0.016, 0.028, 2)
            ph = rng.uniform(0, 2 * np.pi, 4)
            amp = float(rng.uniform(0.0008, 0.0017))
            env = (rng.uniform(0.05, 0.12, 2), rng.uniform(0, 2 * np.pi, 2))   # puckers come and go along the seam
            self._wr.append((creases, dim))
            self._pk = getattr(self, "_pk", []) + [(lam, ph, amp, env)]

    def label_margin(self):
        return self.fin + 0.004, self.fin + 0.0015

    def _dist(self, x, y):
        hx, hy, _ = self.h
        dx = np.clip(hx - np.abs(x), 1e-6, None)
        dy = np.clip(hy - np.abs(y), 1e-6, None)
        p = 6.0                          # squarer plan corners (was 4)
        return (dx ** -p + dy ** -p) ** (-1 / p)

    def _wrinkle(self, x, y, side):
        creases, dim = self._wr[side]
        w = np.zeros(np.broadcast(x, y).shape)
        for c, ang, ln, wd, dp in creases:
            ux, uy = math.cos(ang), math.sin(ang)
            px, py = x - c[0], y - c[1]
            s = px * ux + py * uy
            q = -px * uy + py * ux
            along = np.clip(1 - (2 * np.abs(s) / ln) ** 2, 0, 1)
            w = w + dp * along * np.exp(-0.5 * (q / wd) ** 2)
        for c, rad, dp in dim:
            w = w + dp * np.exp(-0.5 * (((x - c[0]) ** 2 + (y - c[1]) ** 2) / rad ** 2))
        return np.clip(w, 0, 1.6) / 1.6

    def thickness(self, x, y, side=0):
        x, y = np.asarray(x, float), np.asarray(y, float)
        sx, sy, sz = self.semi
        q = 1 - (x / sx) ** 2 - (y / sy) ** 2
        h_ell = sz * np.sqrt(np.clip(q, 0, None))
        T = self.hb + smooth_pos(h_ell - self.hb, 0.3 * (sz - self.hb) + 1e-6)
        d = self._dist(x, y)
        s = np.clip((d - self.fin) / self.R, 0, 1)
        pw = self.flank_power
        e = np.clip(1 - (1 - s) ** pw, 0, 1) ** (1 / pw)       # superellipse flank: squarer than a circle
        t = np.where(d < self.fin, self.t_fin * np.clip(d / self.fin, 0, 1), self.t_fin + (T - self.t_fin) * e)
        t = t - self.wr * self._wrinkle(x, y, side) * e
        if self.flap is not None and side == self.flap[0]:      # the main face lies a film's thickness under the flap
            xe = self.flap[1] * x
            edge = hx_ = self.h[0] - self.flap[2]
            w = np.clip((xe - (edge - 0.0008)) / 0.0016, 0, 1)
            w = w * w * (3 - 2 * w)
            t = t - self.flap_step * (1 - w) * e
        # seam puckers: strongest a little inside the fin, fading towards the middle
        lam, ph, amp, (elam, eph) = self._pk[side]
        hx, hy, _ = self.h
        along = np.where(hx - np.abs(x) < hy - np.abs(y), y + 0.37 * hy * np.sign(x), x + 0.29 * hx * np.sign(y))
        wave = 0.5 + 0.25 * np.sin(2 * np.pi * along / lam[0] + ph[0]) + 0.25 * np.sin(2 * np.pi * along / lam[1] + ph[1])
        envl = np.clip(0.55 + 0.45 * np.sin(2 * np.pi * along / elam[0] + eph[0])
                       + 0.35 * np.sin(2 * np.pi * along / elam[1] + eph[1]), 0.0, 1.0)
        wave = wave * envl
        wd = np.exp(-0.5 * ((d - self.fin - 0.3 * self.R) / (0.35 * self.R + 1e-6)) ** 2) * (d > self.fin)
        t = t - amp * wave * wd
        return np.maximum(t, 0.0)

    def droop(self, x, y):
        x, y = np.asarray(x, float), np.asarray(y, float)
        d = self._dist(x, y)
        hx, hy, _ = self.h
        along = np.where(hx - np.abs(x) < hy - np.abs(y), y + 0.37 * hy * np.sign(x), x + 0.29 * hx * np.sign(y))
        lam, ph = self._pk[0][0], self._pk[0][1]
        mod = 0.55 + 0.45 * np.sin(2 * np.pi * along / (4.3 * lam[0]) + ph[2])      # the seam droops unevenly
        return -self.fin_droop * mod * np.clip(1 - d / (self.fin + 0.3 * self.R), 0, 1) ** 2

    def _surf(self, x, y, side):
        z = self.thickness(x, y, side)
        return (z if side == 0 else -z) + self.droop(x, y)

    def project(self, f, a, b, lift=0.0):
        eu, ev, n = face_axes(f)
        a, b = np.asarray(a, float), np.asarray(b, float)
        if f in (0, 1):
            side = 0 if f == 0 else 1
            P = eu * a[..., None] + ev * b[..., None]
            x, y = P[..., 0], P[..., 1]
            z = self._surf(x, y, side)
            e = 2e-4
            zx = (self._surf(x + e, y, side) - self._surf(x - e, y, side)) / (2 * e)
            zy = (self._surf(x, y + e, side) - self._surf(x, y - e, side)) / (2 * e)
            sgn = 1.0 if side == 0 else -1.0
            N = np.stack([-zx * sgn, -zy * sgn, np.full_like(zx, sgn)], -1)
            N /= np.linalg.norm(N, axis=-1, keepdims=True)
            S = np.stack([x, y, z], -1)
            return S + N * lift, N
        # side faces: march inward along -n from the face plane until the film is met (labels wrapping a flank)
        ha, hb, hn = face_half(self.h, f)
        P0 = n * hn + eu * a[..., None] + ev * b[..., None]
        lo = np.zeros(a.shape)
        hi = np.full(a.shape, 0.6 * hn)
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            P = P0 - n * mid[..., None]
            zt = np.where(P[..., 2] >= 0, self.thickness(P[..., 0], P[..., 1], 0),
                          self.thickness(P[..., 0], P[..., 1], 1))
            inside = np.abs(P[..., 2]) <= zt
            hi = np.where(inside, mid, hi)
            lo = np.where(inside, lo, mid)
        S = P0 - n * hi[..., None]
        # normal from the implicit surface |z| - t(x, y)
        e = 2e-4
        side = (S[..., 2] < 0).astype(int)

        def g(xx, yy, zz):
            return np.abs(zz) - np.where(zz >= 0, self.thickness(xx, yy, 0), self.thickness(xx, yy, 1))
        gx = (g(S[..., 0] + e, S[..., 1], S[..., 2]) - g(S[..., 0] - e, S[..., 1], S[..., 2])) / (2 * e)
        gy = (g(S[..., 0], S[..., 1] + e, S[..., 2]) - g(S[..., 0], S[..., 1] - e, S[..., 2])) / (2 * e)
        gz = (g(S[..., 0], S[..., 1], S[..., 2] + e) - g(S[..., 0], S[..., 1], S[..., 2] - e)) / (2 * e)
        N = np.stack([gx, gy, gz], -1)
        N /= np.maximum(np.linalg.norm(N, axis=-1, keepdims=True), 1e-12)
        del side
        return S + N * lift, N

    def face_grid(self, f, quality):
        hx, hy, _ = self.h
        step, nflank = (0.005, 12) if quality == "full" else (0.012, 6)

        def ax(h):
            k = np.arange(nflank + 1) / nflank
            flank = self.fin + self.R * (1 - np.cos(k * np.pi / 2))     # dense near the fin
            edge = np.array([0.0, 0.5 * self.fin])
            near = np.concatenate([edge, flank])
            n_in = max(1, int(math.ceil((2 * (h - near[-1])) / step)))
            mid = np.linspace(-h + near[-1], h - near[-1], n_in + 1)[1:-1]
            return np.concatenate([-h + near, mid, (h - near)[::-1]])
        ha, hb, _ = face_half(self.h, f)
        return ax(ha), ax(hb)

    def island_faces(self):
        return [(f,) + face_half(self.h, f)[:2] for f in self.faces]

    def collision(self):
        return {"type": "box+ellipsoid", "box_half": [self.h[0], self.h[1], self.hb], "ellipsoid_semi":
                self.semi.tolist()}


# ======================================================================================================== waybill placement
def _extent(W, H, theta):
    c, s = abs(math.cos(theta)), abs(math.sin(theta))
    return W * c + H * s, W * s + H * c


def place_waybill(shape, face, rng, label_uv=None, paper=None, skew_max_deg=4.0):
    """Choose the paper size, in-plane direction, skew and offset of the waybill on ``face``.

    Directions: 0..3 = the label's text-up vector turned by k x 90 deg about the face normal from the face's
    island v axis (+z on side faces). Either paper (100 x 180 or 76 x 130 mm) with equal chance when both fit inside the
    face with the nominal margin in some direction, else the paper that fits; the minimum margin is the last resort. Offset:
    ``label_uv`` (the simulator's fractions of the free margin along its face tangents) or uniform random."""
    eu, ev, n = face_axes(face)
    ha, hb, hn = face_half(shape.h, face)
    m_nom, m_min = shape.label_margin()
    skew0 = math.radians(float(rng.uniform(-skew_max_deg, skew_max_deg)))
    # both papers equally often where both fit; the small one where only it fits
    order = [paper] if paper else (["100x180", "76x130"] if rng.uniform() < 0.5 else ["76x130", "100x180"])
    choice = None
    # nominal margin first, then less skew, then the minimum margin; the drawn paper first
    for margin, sk in ((m_nom, 1.0), (m_nom, 0.0), (m_min, 1.0), (m_min, 0.0)):
        skew = skew0 * sk
        for pk in order:
            W, H = PAPERS[pk]
            dirs = []
            for k in range(4):
                th = k * math.pi / 2 + skew
                ea, eb = _extent(W, H, th)
                if ea / 2 <= ha - margin + 1e-9 and eb / 2 <= hb - margin + 1e-9:
                    dirs.append(k)
            if dirs:
                choice = (pk, dirs, margin)
                break
        if choice:
            break
    if not choice:
        skew = skew0
    fits = choice is not None
    note = None
    if shape.kind == "mailer" and face not in (0, 1):
        fits = False                               # a mailer has no side wall: the sheet wraps over the flank
        note = "mailer side face: the waybill wraps over the rounded flank onto the top / bottom film"
    if not fits:                                   # overhang: smallest paper, the direction that fits best
        pk = paper if paper else "76x130"
        W, H = PAPERS[pk]
        best = min(range(4), key=lambda k: max(_extent(W, H, k * math.pi / 2 + skew)[0] / 2 - ha,
                                               _extent(W, H, k * math.pi / 2 + skew)[1] / 2 - hb))
        choice = (pk, [best], m_min)
    pk, dirs, margin = choice
    k = int(dirs[int(rng.integers(len(dirs)))])
    W, H = PAPERS[pk]
    th = k * math.pi / 2 + skew
    ea, eb = _extent(W, H, th)
    fa, fb = max(0.0, ha - margin - ea / 2), max(0.0, hb - margin - eb / 2)
    if label_uv is None:
        u, v = rng.uniform(-0.9, 0.9, 2)
        oa, ob = u * fa, v * fb
    else:
        # label_uv runs along the simulator's tangent axes t1, t2 of this face
        t1, t2 = sim_tangents(face)
        free = {int(np.argmax(np.abs(eu))): fa, int(np.argmax(np.abs(ev))): fb}
        off = np.zeros(3)
        off[t1] = float(np.clip(label_uv[0], -1, 1)) * free[t1]
        off[t2] = float(np.clip(label_uv[1], -1, 1)) * free[t2]
        oa, ob = float(off @ eu), float(off @ ev)
    # label frame in the parcel frame: x = reading direction, y = text up, z = outward normal
    c, s = math.cos(th), math.sin(th)
    up = -s * eu + c * ev
    right = c * eu + s * ev
    centre_plane = n * hn + eu * oa + ev * ob
    return {
        "face": int(face), "face_name": FACE_NAMES[face], "paper": pk, "size_m": [W, H], "direction": k,
        "skew_deg": math.degrees(skew), "theta_rad": th, "offset_ab_m": [oa, ob], "margin_m": margin,
        "fits": fits, "note": note, "normal": n.tolist(), "right": right.tolist(), "up": up.tolist(),
        "centre_plane": centre_plane.tolist(),
    }


def waybill_points(pl, xl, yl):
    """Label-frame coordinates (m; x reading, y up, centred) -> face-plane (a, b)."""
    th = pl["theta_rad"]
    c, s = math.cos(th), math.sin(th)
    oa, ob = pl["offset_ab_m"]
    a = oa + c * xl - s * yl
    b = ob + s * xl + c * yl
    return a, b


def label_frame_quat(pl):
    """wxyz quaternion of the label frame (x right, y up, z normal) in the parcel frame."""
    R = np.stack([pl["right"], pl["up"], pl["normal"]], axis=1)
    from scipy.spatial.transform import Rotation
    q = Rotation.from_matrix(R).as_quat()          # xyzw
    return [float(q[3]), float(q[0]), float(q[1]), float(q[2])]
