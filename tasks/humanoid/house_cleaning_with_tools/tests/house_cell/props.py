"""Procedural props of the house-cleaning cell: meshes and textures made in code (no asset file of the cell's own).

Meshes are (vertices (n, 3), faces (m, 3) int32, normals (n, 3), texture coordinates (n, 2)) in metres; every
generator is deterministic. Textures are uint8 RGB arrays (rows top-down), made from seeded value noise; they are
built once per process (`texture()` caches them by name). What the cameras see of the cell's own furniture and
objects is drawn from these; their colliders are primitive shapes (scene.py).
"""

from __future__ import annotations

import functools
import math

import numpy as np

# ================================================================================================ meshes


def _normalize(v):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, 1e-12)


def rounded_box(half, radius=0.0, n=6):
    """A box of half sizes `half` with rounded edges and corners (radius, clipped to the smallest half), as a
    subdivided cube projected onto the rounded shape. Box-projected texture coordinates: each face maps its two
    in-plane axes onto [0, 1] (u along the first remaining axis), scaled by the face's size so a texture keeps its
    aspect across faces of different sizes (one texture unit = 1 m)."""
    h = np.asarray(half, float)
    r = float(min(radius, 0.999 * h.min()))
    core = h - r
    V, F, N, UV = [], [], [], []
    for axis in range(3):
        for sgn in (-1.0, 1.0):
            u_ax, v_ax = [a for a in range(3) if a != axis]
            g = np.linspace(-1.0, 1.0, n + 1)
            uu, vv = np.meshgrid(g, g, indexing="ij")
            P = np.zeros((n + 1, n + 1, 3))
            P[..., axis] = sgn * h[axis]
            P[..., u_ax] = uu * h[u_ax]
            P[..., v_ax] = vv * h[v_ax]
            P = P.reshape(-1, 3)
            if r > 0:
                c = np.clip(P, -core, core)
                d = _normalize(P - c)
                P = c + r * d
                nrm = d
            else:
                nrm = np.zeros_like(P)
                nrm[:, axis] = sgn
            base = len(np.concatenate(V)) if V else 0
            V.append(P)
            N.append(nrm)
            UV.append(np.stack([(P[:, u_ax] + h[u_ax]), (P[:, v_ax] + h[v_ax])], axis=1))
            for i in range(n):
                for j in range(n):
                    a = base + i * (n + 1) + j
                    b, c_, d_ = a + (n + 1), a + (n + 1) + 1, a + 1
                    tri = [(a, b, c_), (a, c_, d_)]
                    # outward winding: the face normal along +sgn of `axis`
                    for t in tri:
                        p0, p1, p2 = P[t[0] - base], P[t[1] - base], P[t[2] - base]
                        if np.dot(np.cross(p1 - p0, p2 - p0), np.eye(3)[axis] * sgn) < 0:
                            t = (t[0], t[2], t[1])
                        F.append(t)
    V = np.concatenate(V)
    N = np.concatenate(N)
    UV = np.concatenate(UV)
    return _weld(V, np.array(F, np.int32), N, UV)


def _weld(V, F, N, UV, tol=1e-7):
    """Merge duplicate vertices (same position, normal and texture coordinate)."""
    key = np.round(np.concatenate([V, N, UV], axis=1) / tol).astype(np.int64)
    _, idx, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    return V[idx], inv.reshape(-1)[F].astype(np.int32), N[idx], UV[idx]


def lathe(profile, segments=40, cap_bottom=False, cap_top=False, phase=0.0):
    """Revolve a profile [(r, z), ...] (bottom to top, the outside of the shape; r >= 0) about z. Texture
    coordinates: u = angle / 2 pi, v = arc length along the profile (m). Smooth normals along the profile, hard
    creases where the profile turns by more than 50 degrees (the vertex is duplicated)."""
    prof = np.asarray(profile, float)
    # split the profile at sharp corners so each run gets its own normals
    runs, cur = [], [prof[0]]
    for i in range(1, len(prof)):
        cur.append(prof[i])
        if 0 < i < len(prof) - 1:
            a = prof[i] - prof[i - 1]
            b = prof[i + 1] - prof[i]
            cosang = float(np.dot(a, b) / max(1e-12, np.linalg.norm(a) * np.linalg.norm(b)))
            if cosang < math.cos(math.radians(50.0)):
                runs.append(np.array(cur))
                cur = [prof[i]]
    runs.append(np.array(cur))
    V, F, N, UV = [], [], [], []
    ang = phase + np.linspace(0.0, 2 * math.pi, segments + 1)
    s_off = 0.0
    for run in runs:
        seg = np.diff(run, axis=0)
        L = np.linalg.norm(seg, axis=1)
        s = np.concatenate([[0.0], np.cumsum(L)]) + s_off
        s_off = s[-1]
        # 2D normals of the profile (outward: rotate the tangent by -90 deg in the (r, z) plane)
        t = np.zeros_like(run)
        t[1:-1] = run[2:] - run[:-2]
        t[0] = run[1] - run[0]
        t[-1] = run[-1] - run[-2]
        t = t / np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-12)
        n2 = np.stack([t[:, 1], -t[:, 0]], axis=1)
        base = sum(len(v) for v in V)
        P = np.zeros((len(run), segments + 1, 3))
        Nn = np.zeros_like(P)
        P[..., 0] = run[:, :1] * np.cos(ang)[None, :]
        P[..., 1] = run[:, :1] * np.sin(ang)[None, :]
        P[..., 2] = run[:, 1:2]
        Nn[..., 0] = n2[:, :1] * np.cos(ang)[None, :]
        Nn[..., 1] = n2[:, :1] * np.sin(ang)[None, :]
        Nn[..., 2] = n2[:, 1:2]
        U = np.broadcast_to(ang[None, :] / (2 * math.pi), (len(run), segments + 1))
        Vv = np.broadcast_to(s[:, None], (len(run), segments + 1))
        V.append(P.reshape(-1, 3))
        N.append(Nn.reshape(-1, 3))
        UV.append(np.stack([U, Vv], axis=-1).reshape(-1, 2))
        w = segments + 1
        for i in range(len(run) - 1):
            for j in range(segments):
                a = base + i * w + j
                F.append((a, a + 1, a + w + 1))
                F.append((a, a + w + 1, a + w))
    V, N, UV = np.concatenate(V), np.concatenate(N), np.concatenate(UV)
    F = list(F)
    for cap, idx in ((cap_bottom, 0), (cap_top, -1)):
        if not cap:
            continue
        r, z = prof[idx]
        if r < 1e-6:
            continue
        nz = -1.0 if idx == 0 else 1.0
        base = len(V)
        ring = np.stack([r * np.cos(ang), r * np.sin(ang), np.full_like(ang, z)], axis=1)
        V = np.concatenate([V, [[0.0, 0.0, z]], ring])
        N = np.concatenate([N, np.tile([0.0, 0.0, nz], (len(ring) + 1, 1))])
        UV = np.concatenate([UV, [[0.5, 0.5]], 0.5 + 0.5 * ring[:, :2] / max(r, 1e-9)])
        for j in range(segments):
            tri = (base, base + 1 + j, base + 2 + j)
            F.append(tri if nz > 0 else (tri[0], tri[2], tri[1]))
    return V, np.array(F, np.int32), N, UV


def tube(path, radius, segments=16, cap=True):
    """A tube of `radius` along a polyline `path` (k, 3), with parallel-transported frames; u around, v along (m)."""
    P = np.asarray(path, float)
    T = np.zeros_like(P)
    T[1:-1] = P[2:] - P[:-2]
    T[0] = P[1] - P[0]
    T[-1] = P[-1] - P[-2]
    T = _normalize(T)
    ref = np.array([0.0, 0.0, 1.0]) if abs(T[0][2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    n0 = _normalize(np.cross(T[0], ref))
    frames = [n0]
    for i in range(1, len(P)):
        n = frames[-1] - np.dot(frames[-1], T[i]) * T[i]
        frames.append(_normalize(n))
    ang = np.linspace(0.0, 2 * math.pi, segments + 1)
    V, N, UV, F = [], [], [], []
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])
    for i in range(len(P)):
        nrm = frames[i]
        bin_ = np.cross(T[i], nrm)
        for a in ang:
            d = math.cos(a) * nrm + math.sin(a) * bin_
            V.append(P[i] + radius * d)
            N.append(d)
            UV.append((a / (2 * math.pi), s[i]))
    w = segments + 1
    for i in range(len(P) - 1):
        for j in range(segments):
            a = i * w + j
            F.append((a, a + w, a + w + 1))
            F.append((a, a + w + 1, a + 1))
    V, N, UV = np.array(V), np.array(N), np.array(UV)
    if cap:
        for end, sgn in ((0, -1.0), (len(P) - 1, 1.0)):
            base = len(V)
            V = np.concatenate([V, [P[end]], V[end * w:(end + 1) * w]])
            N = np.concatenate([N, np.tile(T[end] * sgn, (w + 1, 1))])
            UV = np.concatenate([UV, np.full((w + 1, 2), 0.5)])
            for j in range(segments):
                tri = (base, base + 1 + j, base + 2 + j)
                F.append(tri if sgn < 0 else (tri[0], tri[2], tri[1]))
    return V, np.array(F, np.int32), N, UV


def crumpled_ball(radius, seed, subdiv=3, roughness=0.28):
    """A crumpled paper ball: an icosphere with seeded low-frequency radial creases (and facet normals, for the paper's
    folds). Texture coordinates: spherical."""
    t = (1.0 + 5 ** 0.5) / 2.0
    V = [(-1, t, 0), (1, t, 0), (-1, -t, 0), (1, -t, 0), (0, -1, t), (0, 1, t), (0, -1, -t), (0, 1, -t),
         (t, 0, -1), (t, 0, 1), (-t, 0, -1), (-t, 0, 1)]
    F = [(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11), (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6),
         (7, 1, 8), (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9), (4, 9, 5), (2, 4, 11), (6, 2, 10),
         (8, 6, 7), (9, 8, 1)]
    V = [np.array(v, float) / np.linalg.norm(v) for v in V]
    for _ in range(subdiv):
        cache, F2 = {}, []

        def mid(i, j):
            k = (min(i, j), max(i, j))
            if k not in cache:
                cache[k] = len(V)
                V.append(_normalize((V[i] + V[j]) / 2.0))
            return cache[k]

        for a, b, c in F:
            ab, bc, ca = mid(a, b), mid(b, c), mid(c, a)
            F2 += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
        F = F2
    V = np.array(V)
    rng = np.random.default_rng(seed)
    dirs = _normalize(rng.normal(size=(9, 3)))
    amp = rng.uniform(0.4, 1.0, size=9)
    freq = rng.uniform(2.0, 5.0, size=9)
    disp = np.zeros(len(V))
    for d, a, f in zip(dirs, amp, freq):
        disp += a * np.abs(np.sin(f * (V @ d) * math.pi))
    disp = (disp - disp.mean()) / (np.abs(disp).max() + 1e-9)
    P = V * radius * (1.0 + roughness * disp)[:, None]
    P[:, 2] *= 0.82                                   # a little flattened (it lies on a table)
    # facet normals: duplicate vertices per face
    F = np.array(F, np.int32)
    tri = P[F]
    fn = _normalize(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]))
    Vf = tri.reshape(-1, 3)
    Nf = np.repeat(fn, 3, axis=0)
    sph = _normalize(Vf)
    UV = np.stack([0.5 + np.arctan2(sph[:, 1], sph[:, 0]) / (2 * math.pi), 0.5 - np.arcsin(np.clip(sph[:, 2], -1, 1))
                   / math.pi], axis=1)
    return Vf, np.arange(len(Vf), dtype=np.int32).reshape(-1, 3), Nf, UV


def transform(mesh, R=None, t=None):
    V, F, N, UV = mesh
    R = np.eye(3) if R is None else np.asarray(R, float)
    t = np.zeros(3) if t is None else np.asarray(t, float)
    return V @ R.T + t, F, N @ R.T, UV


def merge(*meshes):
    V, F, N, UV, off = [], [], [], [], 0
    for m in meshes:
        V.append(m[0])
        F.append(m[1] + off)
        N.append(m[2])
        UV.append(m[3])
        off += len(m[0])
    return np.concatenate(V), np.concatenate(F).astype(np.int32), np.concatenate(N), np.concatenate(UV)


def scale_uv(mesh, su, sv):
    V, F, N, UV = mesh
    return V, F, N, UV * np.array([su, sv])


# ================================================================================================ textures


def _value_noise(h, w, cells, rng, octaves=4, persistence=0.5):
    """Seamless-enough value noise in [0, 1], (h, w)."""
    out = np.zeros((h, w))
    amp, total = 1.0, 0.0
    for o in range(octaves):
        ch, cw = max(2, int(cells[0] * 2 ** o)), max(2, int(cells[1] * 2 ** o))
        grid = rng.random((ch + 1, cw + 1))
        y = np.linspace(0, ch, h, endpoint=False)
        x = np.linspace(0, cw, w, endpoint=False)
        y0, x0 = y.astype(int), x.astype(int)
        fy, fx = y - y0, x - x0
        fy = fy * fy * (3 - 2 * fy)
        fx = fx * fx * (3 - 2 * fx)
        g00 = grid[y0][:, x0]
        g01 = grid[y0][:, x0 + 1]
        g10 = grid[y0 + 1][:, x0]
        g11 = grid[y0 + 1][:, x0 + 1]
        top = g00 * (1 - fx)[None, :] + g01 * fx[None, :]
        bot = g10 * (1 - fx)[None, :] + g11 * fx[None, :]
        out += amp * (top * (1 - fy)[:, None] + bot * fy[:, None])
        total += amp
        amp *= persistence
    return out / total


def _u8(img):
    return np.clip(np.round(img * 255.0), 0, 255).astype(np.uint8)


def _mix(a, b, t):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return a[None, None, :] * (1 - t[..., None]) + b[None, None, :] * t[..., None]


def wood(h, w, light, dark, seed, grain=38.0, planks=0, warp=6.0):
    """Wood grain along the image's x axis (u): stretched rings with noise; `planks` > 0 adds plank seams along x."""
    rng = np.random.default_rng(seed)
    n1 = _value_noise(h, w, (6, 2), rng, octaves=4)
    n2 = _value_noise(h, w, (40, 3), rng, octaves=3)
    y = np.linspace(0, 1, h)[:, None]
    rings = np.sin((y * grain + warp * n1) * 2 * math.pi) * 0.5 + 0.5
    rings = rings ** 2.2
    t = np.clip(0.55 * rings + 0.35 * n2 + 0.10 * rng.random((h, w)), 0, 1)
    img = _mix(light, dark, t)
    if planks:
        seam = (np.arange(h) % (h // planks)) < 2
        img[seam] *= 0.72
    return _u8(img)


def quartz(h, w, base, seed):
    """White quartz: a soft cloudy base with fine grey and a few dark specks."""
    rng = np.random.default_rng(seed)
    cloud = _value_noise(h, w, (5, 5), rng, octaves=5)
    img = np.asarray(base, float)[None, None, :] * (0.94 + 0.08 * cloud[..., None])
    sp = rng.random((h, w))
    img[sp > 0.985] *= 0.78
    img[sp > 0.997] *= 0.55
    return _u8(img)


def tiles(h, w, tile, grout, colours, seed, rows=6, cols=3, bevel=3):
    """Subway tiles (running bond), `rows` x `cols` per texture, grout lines, a slight per-tile tint and a bevel."""
    rng = np.random.default_rng(seed)
    img = np.ones((h, w, 3)) * np.asarray(grout, float)[None, None, :]
    th, tw = h // rows, w // cols
    g = max(2, th // 12)
    for r in range(rows):
        off = (tw // 2) if r % 2 else 0
        for c in range(-1, cols + 1):
            x0 = c * tw + off
            col = np.asarray(colours[int(rng.integers(len(colours)))], float) * rng.uniform(0.97, 1.03)
            y0 = r * th
            ys, ye = y0 + g, y0 + th - g
            xs, xe = max(0, x0 + g), min(w, x0 + tw - g)
            if xe <= xs:
                continue
            block = np.ones((ye - ys, xe - xs, 3)) * col[None, None, :]
            # bevel: lighter top / left rim, darker bottom / right rim
            block[:bevel] *= 1.06
            block[:, :bevel] *= 1.04
            block[-bevel:] *= 0.90
            block[:, -bevel:] *= 0.93
            img[ys:ye, xs:xe] = block
    img *= (0.97 + 0.05 * _value_noise(h, w, (8, 8), rng, octaves=3))[..., None]
    return _u8(img)


def plain(h, w, colour, seed, amp=0.05, cells=(6, 6)):
    """A painted / plastic surface: the colour with a faint noise (keeps it from looking flat)."""
    rng = np.random.default_rng(seed)
    n = _value_noise(h, w, cells, rng, octaves=4)
    return _u8(np.asarray(colour, float)[None, None, :] * (1 - amp + 2 * amp * n[..., None]))


def fabric(h, w, colour, seed):
    """A woven upholstery fabric: a fine cross weave plus mottling."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w]
    weave = 0.5 + 0.25 * np.sin(xx * 1.6) * np.sin(yy * 1.6) + 0.25 * np.sin((xx + yy) * 0.8)
    n = _value_noise(h, w, (10, 10), rng, octaves=3)
    return _u8(np.asarray(colour, float)[None, None, :] * (0.86 + 0.12 * weave[..., None] + 0.06 * n[..., None]))


def stripes_label(h, w, colours, seed, logo=None):
    """A drinks-can / bottle label (u around, v up): horizontal colour bands, a circular emblem, fine print noise."""
    rng = np.random.default_rng(seed)
    img = np.zeros((h, w, 3))
    edges = np.sort(rng.uniform(0.15, 0.85, size=len(colours) - 1))
    v = np.linspace(0, 1, h)[:, None] * np.ones((1, w))
    band = np.searchsorted(edges, v)
    for i, c in enumerate(colours):
        img[band == i] = c
    if logo is not None:
        yy, xx = np.mgrid[0:h, 0:w]
        for cx in (0.25, 0.75):
            r = np.hypot((xx / w - cx) * w / h * 0.5, yy / h - 0.5)
            img[r < 0.16] = logo
            img[(r > 0.12) & (r < 0.14)] = np.asarray(colours[0]) * 0.6
    img *= (0.95 + 0.08 * rng.random((h, w)))[..., None]
    return _u8(img)


def book_cover(h, w, colour, band, seed):
    """A cloth-bound book cover laid out round the box (scene.py maps it): the colour, a title band, a faint weave."""
    rng = np.random.default_rng(seed)
    img = np.ones((h, w, 3)) * np.asarray(colour, float)[None, None, :]
    y0, y1 = int(0.18 * h), int(0.26 * h)
    img[y0:y1] = band
    img[int(0.70 * h):int(0.72 * h)] = band
    yy, xx = np.mgrid[0:h, 0:w]
    img *= (0.92 + 0.06 * np.sin(xx * 2.1) * np.sin(yy * 2.1) + 0.05 * _value_noise(h, w, (6, 6), rng))[..., None]
    return _u8(img)


def spines(h, w, seed, palette):
    """A row of book spines (u along the row): books of random widths, colours, heights and title bands."""
    rng = np.random.default_rng(seed)
    img = np.ones((h, w, 3)) * 0.12
    x = 0
    while x < w:
        bw = int(rng.integers(w // 40, w // 14))
        col = np.asarray(palette[int(rng.integers(len(palette)))], float) * rng.uniform(0.85, 1.1)
        top = int(rng.uniform(0.0, 0.22) * h)
        img[top:, x:x + bw] = col
        if rng.random() < 0.8:
            b0 = int(rng.uniform(0.25, 0.45) * h)
            img[b0:b0 + max(2, h // 30), x + 2:x + bw - 2] = (0.85, 0.78, 0.55)
        img[top:, x:x + 1] *= 0.6
        x += bw
    img *= (0.94 + 0.08 * rng.random((h, w)))[..., None]
    return _u8(img)


def recycle_badge(h, w, bg, fg):
    """A recycling badge: three chasing arrows round a triangle, light on the bin's colour."""
    img = np.ones((h, w, 3)) * np.asarray(bg, float)[None, None, :]
    yy, xx = (np.mgrid[0:h, 0:w] + 0.5) / np.array([h, w])[:, None, None]
    cx, cy = 0.5, 0.52
    for k in range(3):
        a0 = math.pi / 2 + k * 2 * math.pi / 3
        p0 = np.array([cx + 0.34 * math.cos(a0), cy - 0.34 * math.sin(a0)])
        a1 = a0 + 2 * math.pi / 3
        p1 = np.array([cx + 0.34 * math.cos(a1), cy - 0.34 * math.sin(a1)])
        d = p1 - p0
        L = np.linalg.norm(d)
        u = ((xx - p0[0]) * d[0] + (yy - p0[1]) * d[1]) / L
        v = np.abs((xx - p0[0]) * d[1] - (yy - p0[1]) * d[0]) / L
        bar = (u > 0.10 * L) & (u < 0.72 * L) & (v < 0.045)
        tip = (u >= 0.72 * L) & (u < 0.90 * L) & (v < 0.10 * (0.90 * L - u) / (0.18 * L))
        img[bar | tip] = fg
    return _u8(img)


def hob(h, w, seed):
    """An induction hob's black glass with two cooking-zone rings and a touch-control strip."""
    rng = np.random.default_rng(seed)
    img = np.ones((h, w, 3)) * 0.035
    yy, xx = (np.mgrid[0:h, 0:w] + 0.5) / np.array([h, w])[:, None, None]
    for cx, cy, r in ((0.30, 0.45, 0.23), (0.72, 0.45, 0.17)):
        rr = np.hypot((xx - cx) * w / h, yy - cy)
        img[np.abs(rr - r) < 0.008] = 0.30
        img[np.abs(rr - 0.55 * r) < 0.004] = 0.20
    img[(yy > 0.86) & (yy < 0.90) & (xx > 0.30) & (xx < 0.70)] = 0.22
    img *= (0.9 + 0.2 * rng.random((h, w)))[..., None]
    return _u8(img)


def paper(h, w, seed, colour=(0.95, 0.95, 0.93)):
    rng = np.random.default_rng(seed)
    n = _value_noise(h, w, (12, 12), rng, octaves=4)
    return _u8(np.asarray(colour, float)[None, None, :] * (0.86 + 0.16 * n[..., None]))


def apple_core(h, w, seed):
    """An apple core's skin and flesh (u around, v up): red skin bands at the ends, creamy flesh in the middle."""
    rng = np.random.default_rng(seed)
    v = np.linspace(0, 1, h)[:, None] * np.ones((1, w))
    flesh = np.array([0.93, 0.86, 0.62])
    skin = np.array([0.62, 0.10, 0.08])
    t = np.clip((np.abs(v - 0.5) - 0.28) / 0.08, 0, 1)
    img = flesh[None, None, :] * (1 - t[..., None]) + skin[None, None, :] * t[..., None]
    img *= (0.88 + 0.16 * _value_noise(h, w, (8, 4), rng))[..., None]
    browned = _value_noise(h, w, (5, 3), rng) > 0.62
    img[browned & (t < 0.5)] *= np.array([0.85, 0.72, 0.55])
    return _u8(img)


# The cell's textures by name: (builder, args). Built once per process.
_TEXTURES = {
    "oak": (wood, (256, 512, (0.80, 0.63, 0.44), (0.58, 0.40, 0.24), 11)),
    "oak_end": (wood, (128, 128, (0.78, 0.60, 0.41), (0.55, 0.38, 0.22), 12, 12.0)),
    "walnut": (wood, (256, 512, (0.42, 0.27, 0.17), (0.24, 0.14, 0.08), 13)),
    "birch": (wood, (256, 512, (0.86, 0.76, 0.60), (0.70, 0.57, 0.40), 14, 30.0)),
    "quartz": (quartz, (512, 512, (0.93, 0.92, 0.90), 21)),
    "tiles": (tiles, (384, 384, None, (0.80, 0.79, 0.76), ((0.86, 0.90, 0.88), (0.84, 0.89, 0.86),
                                                           (0.88, 0.91, 0.89)), 22)),
    "sage": (plain, (128, 128, (0.56, 0.64, 0.55), 31, 0.03)),
    "offwhite": (plain, (128, 128, (0.90, 0.89, 0.86), 32, 0.03)),
    "bin_blue": (plain, (128, 128, (0.12, 0.33, 0.62), 33, 0.04)),
    "bin_grey": (plain, (128, 128, (0.30, 0.31, 0.32), 34, 0.04)),
    # (the bins' insides: the same plastic as a texture of their own, so that the cameras' look darkens them alone;
    # render.apply_look scales a textured material's texture data in place)
    "bin_blue_inner": (plain, (128, 128, (0.12, 0.33, 0.62), 33, 0.04)),
    "bin_grey_inner": (plain, (128, 128, (0.30, 0.31, 0.32), 34, 0.04)),
    "badge": (recycle_badge, (128, 128, (0.12, 0.33, 0.62), (0.92, 0.94, 0.95))),
    "seat_fabric": (fabric, (128, 128, (0.55, 0.50, 0.44), 35)),
    "hob": (hob, (256, 256, 36)),
    "paper": (paper, (128, 128, 37)),
    "kraft": (paper, (128, 128, 38, (0.80, 0.70, 0.52))),
    "apple": (apple_core, (128, 128, 39)),
    "can_red": (stripes_label, (128, 256, ((0.78, 0.07, 0.09), (0.93, 0.93, 0.92), (0.78, 0.07, 0.09)), 41,
                                (0.96, 0.96, 0.95))),
    "can_green": (stripes_label, (128, 256, ((0.10, 0.45, 0.25), (0.85, 0.86, 0.80), (0.10, 0.45, 0.25)), 42,
                                  (0.95, 0.80, 0.15))),
    "label_water": (stripes_label, (64, 256, ((0.15, 0.45, 0.75), (0.93, 0.95, 0.97), (0.15, 0.45, 0.75)), 43)),
    "book_red": (book_cover, (256, 256, (0.62, 0.12, 0.10), (0.85, 0.72, 0.35), 51)),
    "book_blue": (book_cover, (256, 256, (0.12, 0.22, 0.45), (0.90, 0.88, 0.80), 52)),
    "spines_a": (spines, (128, 512, 53, ((0.55, 0.12, 0.10), (0.13, 0.25, 0.45), (0.20, 0.38, 0.28),
                                         (0.80, 0.68, 0.40), (0.25, 0.22, 0.20), (0.72, 0.70, 0.66)))),
    "spines_b": (spines, (128, 512, 54, ((0.35, 0.18, 0.40), (0.85, 0.50, 0.20), (0.15, 0.15, 0.17),
                                         (0.60, 0.15, 0.20), (0.30, 0.45, 0.60), (0.88, 0.85, 0.78)))),
}


@functools.lru_cache(maxsize=None)
def texture(name: str) -> np.ndarray:
    fn, args = _TEXTURES[name]
    if fn is tiles:
        h, w, _, grout, cols, seed = args
        return tiles(h, w, None, grout, cols, seed)
    return fn(*args)


TEXTURE_NAMES = tuple(_TEXTURES)
