#!/usr/bin/env python3
"""Small mesh builders for what the cameras draw (spool_parts.py, spool_scene.py). Every builder returns
(V, N, UV, F): vertices, vertex normals, uv in metres (the writer divides by the look's tile), triangles. Faces carry
their own vertices, so a flat face shades flat and a bevel strip shades round (its two edges take the normals of the
faces they join).
"""
import math

import numpy as np


class Mesh(object):
    def __init__(self):
        self.V, self.N, self.UV, self.F = [], [], [], []

    def quad(self, p, n, uv):
        """four corners counter-clockwise seen from outside; n: one normal or four; uv: four pairs"""
        b = len(self.V)
        n = np.asarray(n, float)
        for k in range(4):
            self.V.append(np.asarray(p[k], float))
            nk = n if n.ndim == 1 else n[k]
            self.N.append(nk / (np.linalg.norm(nk) + 1e-12))
            self.UV.append(uv[k])
        self.F += [(b, b + 1, b + 2), (b, b + 2, b + 3)]

    def tri(self, p, n, uv):
        b = len(self.V)
        n = np.asarray(n, float)
        for k in range(3):
            self.V.append(np.asarray(p[k], float))
            nk = n if n.ndim == 1 else n[k]
            self.N.append(nk / (np.linalg.norm(nk) + 1e-12))
            self.UV.append(uv[k])
        self.F.append((b, b + 1, b + 2))

    def add(self, m):
        b = len(self.V)
        V, N, UV, F = m
        self.V += list(V)
        self.N += list(N)
        self.UV += list(UV)
        self.F += [tuple(int(i) + b for i in f) for f in F]

    def arrays(self):
        return (np.asarray(self.V, float).reshape(-1, 3), np.asarray(self.N, float).reshape(-1, 3),
                np.asarray(self.UV, float).reshape(-1, 2), np.asarray(self.F, int).reshape(-1, 3))


def transform(m, R, t):
    V, N, UV, F = m
    R = np.asarray(R, float)
    return V @ R.T + np.asarray(t, float), N @ R.T, UV, F


def split(m, quads):
    """a mesh built of quads only (4 vertices, 2 triangles each, in order), cut in two by a per-quad flag: returns
    (the quads whose flag is False, the quads whose flag is True)"""
    V, N, UV, F = m
    out = []
    for want in (False, True):
        idx = [q for q in range(len(V) // 4) if bool(quads[q]) == want]
        sel = np.concatenate([np.arange(4 * q, 4 * q + 4) for q in idx]) if idx else np.zeros(0, int)
        Fq = np.concatenate([np.array([(0, 1, 2), (0, 2, 3)]) + 4 * k for k in range(len(idx))]) if idx else np.zeros((0, 3), int)
        out.append((V[sel], N[sel], UV[sel], Fq))
    return out


def bevel_box_parts(h, b):
    """bevel_box as (the six faces, the twelve edge strips with the eight corner triangles): the edges can carry a
    look of their own (worn paint)"""
    V, N, UV, F = bevel_box(h, b)
    return (V[:24], N[:24], UV[:24], F[:12]), (V[24:], N[24:], UV[24:], F[12:] - 24)


def bevel_box(h, b):
    """a box of half sizes h centred at the origin with every edge broken by b (a 45-degree strip shaded round) and
    the corners closed; uv: each face's own two axes in metres, continuous over the strips"""
    h = np.asarray(h, float)
    b = float(min(b, 0.45 * h.min()))
    m = Mesh()
    ax = np.eye(3)
    faces = [(0, 1, 2), (1, 2, 0), (2, 0, 1)]                      # (normal axis, u axis, v axis): u x v = normal
    for a, iu, iv in faces:
        for s in (1, -1):
            n = ax[a] * s
            u, v = (ax[iu], ax[iv]) if s > 0 else (ax[iv], ax[iu])
            hu, hv = (h[iu], h[iv]) if s > 0 else (h[iv], h[iu])
            c = n * h[a]
            p = [c + u * su * (hu - b) + v * sv * (hv - b) for su, sv in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
            m.quad(p, n, [(su * (hu - b), sv * (hv - b)) for su, sv in ((-1, -1), (1, -1), (1, 1), (-1, 1))])
    # edge strips: for each pair of adjacent faces (axes a < c), four edges along the third axis e
    for a in range(3):
        for c in range(a + 1, 3):
            e = 3 - a - c
            for sa in (1, -1):
                for sc in (1, -1):
                    na, nc = ax[a] * sa, ax[c] * sc
                    pa = na * h[a] + nc * (h[c] - b)                  # on face a, at its edge towards face c
                    pc = nc * h[c] + na * (h[a] - b)
                    le = h[e] - b
                    quad = [pa - ax[e] * le, pc - ax[e] * le, pc + ax[e] * le, pa + ax[e] * le]
                    nq = [na, nc, nc, na]
                    if np.dot(np.cross(quad[1] - quad[0], quad[2] - quad[0]), na + nc) < 0:
                        quad, nq = [quad[1], quad[0], quad[3], quad[2]], [nc, na, na, nc]
                    w = b * math.sqrt(2.0)
                    m.quad(quad, np.asarray(nq), [(0, -le), (w, -le), (w, le), (0, le)])
    for sx in (1, -1):
        for sy in (1, -1):
            for sz in (1, -1):
                s = np.array([sx, sy, sz], float)
                p = [s * (h - b * (1 - ax[k])) for k in range(3)]     # the three face corners meeting here
                nn = [ax[k] * s[k] for k in range(3)]
                if np.dot(np.cross(p[1] - p[0], p[2] - p[0]), s) < 0:
                    p, nn = [p[0], p[2], p[1]], [nn[0], nn[2], nn[1]]
                m.tri(p, np.asarray(nn), [(0, 0), (b, 0), (0, b)])
    return m.arrays()


def box_frame(V24):
    """centre, axes (columns) and half sizes of a box written by record_dump.prim_box (24 vertices: +x, -x, +y, -y,
    +z, -z faces)"""
    c = V24.mean(0)
    R, h = np.zeros((3, 3)), np.zeros(3)
    for k, i0 in enumerate((0, 8, 16)):
        d = V24[i0:i0 + 4].mean(0) - c
        h[k] = np.linalg.norm(d)
        R[:, k] = d / (h[k] + 1e-15)
    return c, R, h


def ring_shell(nsec, R, H, r_in, b, facet0=0.5, smooth=False):
    """the outer shell of a ring built of nsec flat sectors (apothem R, height H, facets centred at (s + facet0) x
    360 / nsec degrees): the side facets, a broken top and bottom rim (b, shaded round), the top and bottom annulus
    in to the apothem r_in and the inward-facing wall at r_in (what shows through a slot cut in the inner wall).
    The origin is on the axis at the bottom face. smooth: the side facets shaded as one round wall (the corner
    normals radial; the outline stays the prism's). Quads per sector, in order: side, top rim, bottom rim, top
    face, bottom face, inner wall (split() takes the rims out)."""
    m = Mesh()
    da = 2 * math.pi / nsec
    k = 1.0 / math.cos(da / 2)
    w = 2 * R * math.tan(da / 2)

    def corner(r, i):
        a = (i + facet0 - 0.5) * da
        return np.array([r * k * math.cos(a), r * k * math.sin(a), 0.0])

    for s in range(nsec):
        am = (s + facet0) * da
        nr = np.array([math.cos(am), math.sin(am), 0.0])
        up = np.array([0, 0, 1.0])
        P0, P1 = corner(R, s), corner(R, s + 1)
        Q0, Q1 = corner(R - b, s), corner(R - b, s + 1)
        I0, I1 = corner(r_in, s), corner(r_in, s + 1)
        u0, u1 = s * w, (s + 1) * w
        n0, n1 = (P0 / np.linalg.norm(P0), P1 / np.linalg.norm(P1)) if smooth else (nr, nr)
        m.quad([P0 + up * b, P1 + up * b, P1 + up * (H - b), P0 + up * (H - b)], np.asarray([n0, n1, n1, n0]),
               [(u0, b), (u1, b), (u1, H - b), (u0, H - b)])
        m.quad([P0 + up * (H - b), P1 + up * (H - b), Q1 + up * H, Q0 + up * H], np.asarray([n0, n1, up, up]),
               [(u0, H - b), (u1, H - b), (u1, H + 0.41 * b), (u0, H + 0.41 * b)])
        m.quad([Q0, Q1, P1 + up * b, P0 + up * b], np.asarray([-up, -up, n1, n0]), [(u0, -0.41 * b), (u1, -0.41 * b), (u1, b), (u0, b)])
        m.quad([Q0 + up * H, Q1 + up * H, I1 + up * H, I0 + up * H], up, [tuple(q[:2]) for q in (Q0, Q1, I1, I0)])
        m.quad([I0, I1, Q1, Q0], -up, [tuple(q[:2]) for q in (I0, I1, Q1, Q0)])
        wi = 2 * r_in * math.tan(da / 2)
        m.quad([I1, I0, I0 + up * H, I1 + up * H], -nr, [((s + 1) * wi, 0), (s * wi, 0), (s * wi, H), ((s + 1) * wi, H)])
    return m.arrays()


def cylinder(p0, p1, r, nseg=48, caps=True, bevel=0.0):
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    ax = p1 - p0
    L = np.linalg.norm(ax)
    z = ax / L
    x = np.cross(z, [0, 0, 1.0])
    if np.linalg.norm(x) < 1e-6:
        x = np.array([1.0, 0, 0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    m = Mesh()
    b = min(bevel, 0.4 * r, 0.4 * L)
    for j in range(nseg):
        a0, a1 = 2 * math.pi * j / nseg, 2 * math.pi * (j + 1) / nseg
        d0, d1 = math.cos(a0) * x + math.sin(a0) * y, math.cos(a1) * x + math.sin(a1) * y
        m.quad([p0 + z * b + r * d0, p0 + z * b + r * d1, p0 + z * (L - b) + r * d1, p0 + z * (L - b) + r * d0],
               np.asarray([d0, d1, d1, d0]), [(r * a0, b), (r * a1, b), (r * a1, L - b), (r * a0, L - b)])
        if b > 0:
            m.quad([p0 + z * (L - b) + r * d0, p0 + z * (L - b) + r * d1, p0 + z * L + (r - b) * d1, p0 + z * L + (r - b) * d0],
                   np.asarray([d0, d1, z, z]), [(r * a0, L - b), (r * a1, L - b), (r * a1, L), (r * a0, L)])
            m.quad([p0 + (r - b) * d0, p0 + (r - b) * d1, p0 + z * b + r * d1, p0 + z * b + r * d0],
                   np.asarray([-z, -z, d1, d0]), [(r * a0, 0), (r * a1, 0), (r * a1, b), (r * a0, b)])
        if caps:
            rc = r - b
            m.tri([p0 + z * L, p0 + z * L + rc * d0, p0 + z * L + rc * d1], z, [(0, 0), (rc * math.cos(a0), rc * math.sin(a0)), (rc * math.cos(a1), rc * math.sin(a1))])
            m.tri([p0, p0 + rc * d1, p0 + rc * d0], -z, [(0, 0), (rc * math.cos(a1), rc * math.sin(a1)), (rc * math.cos(a0), rc * math.sin(a0))])
    return m.arrays()


def tslot_outline(a=0.0225, slot=0.0041, depth=0.0062, undercut=0.0035, chamfer=0.0018):
    """the outline (counter-clockwise) of a square T-slot extrusion of half size a: on every side a slot of half
    opening `slot` that widens by `undercut` behind a lip and is `depth` deep, the corners chamfered"""
    lip = 0.0022
    side = [(-a + chamfer, -a), (-slot, -a), (-slot, -a + lip), (-slot - undercut, -a + lip), (-slot - undercut, -a + depth),
            (slot + undercut, -a + depth), (slot + undercut, -a + lip), (slot, -a + lip), (slot, -a), (a - chamfer, -a)]
    out = []
    for q in range(4):
        c, s = math.cos(q * math.pi / 2), math.sin(q * math.pi / 2)
        out += [(c * x - s * y, s * x + c * y) for x, y in side]
    return out


def extrude(outline, p0, p1, up):
    """a closed outline (2D, counter-clockwise, in the frame x = cross(axis, up) ... see below) swept from p0 to p1;
    the cross-section's x axis is `up` projected off the axis, its y axis completes the frame; flat side faces, no
    end caps. uv: u = length along the outline, v = along the sweep."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    z = p1 - p0
    L = np.linalg.norm(z)
    z /= L
    x = np.asarray(up, float) - z * float(np.dot(up, z))
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    m = Mesh()
    u = 0.0
    n = len(outline)
    for i in range(n):
        (x0, y0), (x1, y1) = outline[i], outline[(i + 1) % n]
        e = np.array([x1 - x0, y1 - y0])
        le = float(np.linalg.norm(e))
        if le < 1e-9:
            continue
        nrm = (e[1] * x - e[0] * y) / le                             # outward for a counter-clockwise outline
        a0, a1 = p0 + x0 * x + y0 * y, p0 + x1 * x + y1 * y
        m.quad([a0, a1, a1 + z * L, a0 + z * L], nrm, [(u, 0), (u + le, 0), (u + le, L), (u, L)])
        u += le
    return m.arrays()


# ---- turned and swept shapes ----

def lathe(profile, nseg=96, a0=0.0, a1=2 * math.pi, brk=35.0, r_uv=None):
    """Surface of revolution about z of a polyline [(r, z), ...]. The outside is to the right of the direction of
    travel in the (r, z) plane: going up an outer wall, inward over a top face, down an inner wall. Normals are
    smoothed across a profile vertex whose two segments meet at less than `brk` degrees. uv: (arc length at r_uv
    or at each ring's own radius, length along the profile), metres."""
    P = [np.asarray(p, float) for p in profile]
    segs = []
    for i in range(len(P) - 1):
        t = P[i + 1] - P[i]
        le = float(np.linalg.norm(t))
        if le < 1e-12:
            continue
        segs.append((P[i], P[i + 1], np.array([t[1], -t[0]]) / le, le))
    m = Mesh()
    cb = math.cos(math.radians(brk))
    s = 0.0
    closed = abs(a1 - a0 - 2 * math.pi) < 1e-9
    for k, (p0, p1, n, le) in enumerate(segs):
        n0 = n1 = n
        if k > 0 and float(np.dot(segs[k - 1][2], n)) > cb and np.allclose(segs[k - 1][1], p0):
            n0 = segs[k - 1][2] + n
        if k + 1 < len(segs) and float(np.dot(segs[k + 1][2], n)) > cb and np.allclose(segs[k + 1][0], p1):
            n1 = segs[k + 1][2] + n
        n0, n1 = n0 / np.linalg.norm(n0), n1 / np.linalg.norm(n1)
        for j in range(nseg):
            b0 = a0 + (a1 - a0) * j / nseg
            b1 = a0 + (a1 - a0) * (j + 1) / nseg
            c0, s0, c1, s1 = math.cos(b0), math.sin(b0), math.cos(b1), math.sin(b1)

            def pt(p, c, sn):
                return np.array([p[0] * c, p[0] * sn, p[1]])

            def nr(nn, c, sn):
                return np.array([nn[0] * c, nn[0] * sn, nn[1]])

            ru0 = r_uv if r_uv is not None else max(p0[0], p1[0])
            q = [pt(p0, c0, s0), pt(p0, c1, s1), pt(p1, c1, s1), pt(p1, c0, s0)]
            nq = [nr(n0, c0, s0), nr(n0, c1, s1), nr(n1, c1, s1), nr(n1, c0, s0)]
            uv = [(ru0 * b0, s), (ru0 * b1, s), (ru0 * b1, s + le), (ru0 * b0, s + le)]
            if p0[0] < 1e-9:
                tri, tn, tuv = [q[0], q[2], q[3]], [nq[0], nq[2], nq[3]], [uv[0], uv[2], uv[3]]
            elif p1[0] < 1e-9:
                tri, tn, tuv = [q[0], q[1], q[2]], [nq[0], nq[1], nq[2]], [uv[0], uv[1], uv[2]]
            else:
                tri = None
            nm = nr(n, (c0 + c1) / 2, (s0 + s1) / 2)
            if tri is not None:
                if np.dot(np.cross(tri[1] - tri[0], tri[2] - tri[0]), nm) < 0:
                    tri, tn, tuv = [tri[0], tri[2], tri[1]], [tn[0], tn[2], tn[1]], [tuv[0], tuv[2], tuv[1]]
                m.tri(tri, np.asarray(tn), tuv)
                continue
            if np.dot(np.cross(q[1] - q[0], q[2] - q[0]), nm) < 0:
                q, nq, uv = [q[1], q[0], q[3], q[2]], [nq[1], nq[0], nq[3], nq[2]], [uv[1], uv[0], uv[3], uv[2]]
            m.quad(q, np.asarray(nq), uv)
        s += le
    del closed
    return m.arrays()


def rounded_rect(hx, hy, r, n=10):
    """outline (counter-clockwise, starting on the +x side) of a rectangle of half sizes hx, hy with corner radius r"""
    r = min(r, hx - 1e-6, hy - 1e-6)
    out = []
    for cx, cy, a_start in ((hx - r, hy - r, 0.0), (-hx + r, hy - r, 0.5), (-hx + r, -hy + r, 1.0), (hx - r, -hy + r, 1.5)):
        for i in range(n + 1):
            a = (a_start + 0.5 * i / n) * math.pi
            out.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    return out


def circle(r, n=64, c=(0.0, 0.0), phase=0.0):
    return [(c[0] + r * math.cos(phase + 2 * math.pi * i / n), c[1] + r * math.sin(phase + 2 * math.pi * i / n)) for i in range(n)]


def _outline_normals(o):
    """per-vertex outward normals of a closed counter-clockwise outline (mean of the two edges' normals) and per-edge
    normals"""
    o = np.asarray(o, float)
    e = np.roll(o, -1, axis=0) - o
    le = np.linalg.norm(e, axis=1, keepdims=True) + 1e-15
    en = np.stack([e[:, 1], -e[:, 0]], 1) / le
    vn = en + np.roll(en, 1, axis=0)
    vn /= np.linalg.norm(vn, axis=1, keepdims=True) + 1e-15
    return vn, en


def inset(outline, d):
    """the outline moved inward by d along its vertex normals (fine for rounded outlines and small d)"""
    o = np.asarray(outline, float)
    vn, en = _outline_normals(o)
    k = 1.0 / np.maximum(0.5, np.einsum("ij,ij->i", vn, en))[:, None]      # mitre
    return [tuple(p) for p in (o - vn * d * k)]


def wall(outline, z0, z1, outward=True, brk=35.0, n_top=None, n_bot=None):
    """the side wall of a closed counter-clockwise outline between z0 and z1 (z1 > z0); smooth where the outline
    turns by less than brk degrees. outward=False: seen from inside (a pocket's wall). uv: (length round, z)."""
    o = np.asarray(outline, float)
    vn, en = _outline_normals(o)
    n = len(o)
    m = Mesh()
    cb = math.cos(math.radians(brk))
    u = 0.0
    sgn = 1.0 if outward else -1.0
    for i in range(n):
        j = (i + 1) % n
        le = float(np.linalg.norm(o[j] - o[i]))
        if le < 1e-12:
            continue
        ni = vn[i] if float(np.dot(en[i - 1], en[i])) > cb else en[i]
        nj = vn[j] if float(np.dot(en[i], en[j])) > cb else en[i]
        p = [(o[i][0], o[i][1], z0), (o[j][0], o[j][1], z0), (o[j][0], o[j][1], z1), (o[i][0], o[i][1], z1)]
        nn = [(sgn * ni[0], sgn * ni[1], 0), (sgn * nj[0], sgn * nj[1], 0), (sgn * nj[0], sgn * nj[1], 0), (sgn * ni[0], sgn * ni[1], 0)]
        uv = [(u, z0), (u + le, z0), (u + le, z1), (u, z1)]
        if not outward:
            p, nn, uv = [p[1], p[0], p[3], p[2]], [nn[1], nn[0], nn[3], nn[2]], [uv[1], uv[0], uv[3], uv[2]]
        m.quad([np.asarray(q, float) for q in p], np.asarray(nn, float), uv)
        u += le
    return m.arrays()


def band(o_out, o_in, z_out, z_in, up=True, n_out=None, n_in=None):
    """quads between two closed outlines with the same number of points (o_out at height z_out, o_in at z_in): a flat
    ring (z_out == z_in) or a chamfer strip. up: seen from above. n_out / n_in: normals at the two edges (default:
    the strip's own face normal). uv: planar (x, y)."""
    a, b = np.asarray(o_out, float), np.asarray(o_in, float)
    n = len(a)
    m = Mesh()
    for i in range(n):
        j = (i + 1) % n
        p = [np.array([a[i][0], a[i][1], z_out]), np.array([a[j][0], a[j][1], z_out]),
             np.array([b[j][0], b[j][1], z_in]), np.array([b[i][0], b[i][1], z_in])]
        if np.linalg.norm(p[1] - p[0]) < 1e-12 and np.linalg.norm(p[2] - p[3]) < 1e-12:
            continue
        fn = np.cross(p[1] - p[0], p[2] - p[0])
        if np.linalg.norm(fn) < 1e-18:
            fn = np.cross(p[2] - p[0], p[3] - p[0])
        fn = fn / (np.linalg.norm(fn) + 1e-18)
        if (fn[2] < 0) == up:
            p = [p[1], p[0], p[3], p[2]]
            fn = -fn
            order = (1, 0, 0, 1)
        else:
            order = (0, 1, 1, 0)
        del order
        nn = fn
        if n_out is not None or n_in is not None:
            no = np.asarray(n_out if n_out is not None else fn, float)
            ni = np.asarray(n_in if n_in is not None else fn, float)
            zs = [q[2] for q in p]
            nn = np.asarray([no if abs(zq - z_out) < 1e-12 and z_out != z_in else (ni if z_out != z_in else fn) for zq in zs])
        m.quad(p, nn, [(q[0], q[1]) for q in p])
    return m.arrays()


def cap(outline, z, up=True):
    """a flat cap of a convex outline (fan from its centroid)"""
    o = np.asarray(outline, float)
    c = o.mean(0)
    m = Mesh()
    n = np.array([0, 0, 1.0 if up else -1.0])
    for i in range(len(o)):
        j = (i + 1) % len(o)
        p = [np.array([c[0], c[1], z]), np.array([o[i][0], o[i][1], z]), np.array([o[j][0], o[j][1], z])]
        if not up:
            p = [p[0], p[2], p[1]]
        m.tri(p, n, [(q[0], q[1]) for q in p])
    return m.arrays()


def ray_poly(c, ang, poly):
    """where a ray from c (inside) at angle ang leaves a convex counter-clockwise polygon"""
    d = np.array([math.cos(ang), math.sin(ang)])
    c = np.asarray(c, float)
    best = None
    P = np.asarray(poly, float)
    for i in range(len(P)):
        a, b = P[i], P[(i + 1) % len(P)]
        e = b - a
        den = d[0] * e[1] - d[1] * e[0]
        if abs(den) < 1e-15:
            continue
        w = a - c
        t = (w[0] * e[1] - w[1] * e[0]) / den
        s = (w[0] * d[1] - w[1] * d[0]) / den
        if t > 1e-9 and -1e-9 <= s <= 1 + 1e-9 and (best is None or t < best):
            best = t
    return c + d * best


def hole_cell(c, r, poly, z, up=True, n=48):
    """a flat cell (convex polygon) with a round hole of radius r at c: quads from the hole's edge out to the
    polygon; the polygon's own corners are kept"""
    c = np.asarray(c, float)
    angs = set(2 * math.pi * i / n for i in range(n))
    for p in poly:
        angs.add(math.atan2(p[1] - c[1], p[0] - c[0]) % (2 * math.pi))
    angs = sorted(angs)
    inner = [(c[0] + r * math.cos(a), c[1] + r * math.sin(a)) for a in angs]
    outer = [tuple(ray_poly(c, a, poly)) for a in angs]
    return band(outer, inner, z, z, up=up)


def merge(*ms):
    m = Mesh()
    for x in ms:
        if x is not None and len(x[0]):
            m.add(x)
    return m.arrays()


def xform2(m, yaw=0.0, t=(0.0, 0.0, 0.0)):
    c, s = math.cos(yaw), math.sin(yaw)
    return transform(m, np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]]), t)


def slab(outline, z0, z1, b_top=0.0, b_bot=0.0, top=True, bottom=True):
    """a solid of a convex outline between z0 and z1 with its top and bottom edges broken by b_top / b_bot (strips
    shaded round), caps as fans"""
    parts = [wall(outline, z0 + b_bot, z1 - b_top)]
    o_t = inset(outline, b_top) if b_top > 0 else outline
    o_b = inset(outline, b_bot) if b_bot > 0 else outline
    if b_top > 0:
        parts.append(_round_strip(outline, o_t, z1 - b_top, z1, True))
    if b_bot > 0:
        parts.append(_round_strip(outline, o_b, z0 + b_bot, z0, False))
    if top:
        parts.append(cap(o_t, z1, True))
    if bottom:
        parts.append(cap(o_b, z0, False))
    return merge(*parts)


def _round_strip(o_out, o_in, z_out, z_in, up):
    """a chamfer strip between an outline at z_out and its inset at z_in, shaded from the wall's normal to the cap's"""
    a, b = np.asarray(o_out, float), np.asarray(o_in, float)
    vn, _ = _outline_normals(a)
    n = len(a)
    m = Mesh()
    zc = np.array([0, 0, 1.0 if up else -1.0])
    for i in range(n):
        j = (i + 1) % n
        if np.linalg.norm(a[j] - a[i]) < 1e-12:
            continue
        p = [np.array([a[i][0], a[i][1], z_out]), np.array([a[j][0], a[j][1], z_out]),
             np.array([b[j][0], b[j][1], z_in]), np.array([b[i][0], b[i][1], z_in])]
        nn = [np.array([vn[i][0], vn[i][1], 0]), np.array([vn[j][0], vn[j][1], 0]), zc, zc]
        fn = np.cross(p[1] - p[0], p[2] - p[0])
        if np.dot(fn, nn[0] + zc) < 0:
            p, nn = [p[1], p[0], p[3], p[2]], [nn[1], nn[0], nn[3], nn[2]]
        m.quad(p, np.asarray(nn), [(q[0] + q[2], q[1]) for q in p])
    return m.arrays()


def pocket(outline, z_top, z_bot, floor=True, chamfer=0.0):
    """a pocket of a convex outline sunk from z_top to z_bot: its wall seen from inside and its floor"""
    parts = []
    o = outline
    if chamfer > 0:
        o_big = inset(outline, -chamfer)
        parts.append(band(o_big, o, z_top, z_top - chamfer, up=True))
        z_top = z_top - chamfer
    parts.append(wall(o, z_bot, z_top, outward=False))
    if floor:
        parts.append(cap(o, z_bot, True))
    return merge(*parts)


def uv_planar(m, ax=(0, 1)):
    V, N, UV, F = m
    return V, N, V[:, list(ax)].copy(), F


def uv_box(m):
    """uv by each vertex's dominant normal axis, metres"""
    V, N, UV, F = m
    ax = np.abs(N).argmax(1)
    uv = np.zeros((len(V), 2))
    for k, (i, j) in enumerate(((1, 2), (0, 2), (0, 1))):
        sel = ax == k
        uv[sel, 0] = V[sel, i]
        uv[sel, 1] = V[sel, j]
    return V, N, uv, F
