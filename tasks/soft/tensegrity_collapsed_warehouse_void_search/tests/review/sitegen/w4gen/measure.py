"""Measurements of the heaps on the COMPILED MuJoCo model of a site (no package array or label is read here).

Every ray at a heap is cast with that heap's geom alone in geom group 5 (a run-time change of the compiled model's
geom_group, undone afterwards; the geometry is not touched), so floor, patches and neighbours cannot answer for it.
  vertical rays   mujoco.mj_ray straight down, nudged by (1.1e-6, 0.7e-6) m off the target point: a vertical ray
                  exactly through a height-field vertex can slip through the surface
  lattice rays    mujoco.mj_multiRay from one point 1e4 m above the heap towards each (nudged) lattice point; the
                  3-D hit point is kept (the rays are within 4e-5 rad of vertical; heights read at the nominal (x, y)
                  differ from vertical rays by < 0.01 mm times the local gradient)
Definitions follow the maintainers' heap capability tool (heapcap, version 1.0.0; not published) where it defines them:
footprint = lattice heights > 1 mm; edge rule = the height 0.02 m inside the outline >= 0.03 m along >= 95 % of every
exposed side; slope = median slope between 0.03 m and 0.8 x top (two cells in from the outline); core = footprint
shrunk by half about its area centroid; line gap = per lattice column over the middle 60 % of the rectangle's width,
the distance from the line to the heap's nearest point standing > 3 mm (the registry's definition, on the compiled
surface; largest and median kept); the fit bound = 1 mm lattice of the compiled surface against the piece
mesh's upper envelope (1 mm raster, cliffs densified at 1 mm, floor z = 0 included), nearest-point distance.
"""
import math

import numpy as np

NUDGE = (1.1e-6, 0.7e-6)
GROUP = 5
EDGE_IN, EDGE_MIN, EDGE_FRAC = 0.020, 0.030, 0.95
RAY_STEP = 0.0025
HIGH = 1.0e4
OCC_Z = 0.003            # a point of the floor is occupied when a solid stands > 3 mm above it
OCC_STEP = 0.005
LINE_Z = 0.003           # a heap stands at a point for the line gap when > 3 mm high (the registry's line gap)


def compile_xml(xml):
    import mujoco
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    return m, d


def geom_ids(m, prefix):
    import mujoco
    out = {}
    for i in range(m.ngeom):
        nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i) or ""
        if nm.startswith(prefix):
            out[nm] = i
    return out


class Alone:
    """Context: the geom gid alone in group 5."""

    def __init__(self, m, gid):
        self.m, self.gid = m, gid

    def __enter__(self):
        self.saved = self.m.geom_group.copy()
        self.m.geom_group[self.m.geom_group == GROUP] = 0
        self.m.geom_group[self.gid] = GROUP
        return self

    def __exit__(self, *a):
        self.m.geom_group[:] = self.saved


def _grp():
    g = np.zeros(6, np.uint8)
    g[GROUP] = 1
    return g


def vertical_z(m, d, gid, XY):
    """Surface z of geom gid under each (x, y) by a vertical ray (nan where missed)."""
    import mujoco
    out = np.full(len(XY), np.nan)
    gg = _grp()
    hit = np.zeros(1, np.int32)
    v = np.array([0.0, 0.0, -1.0])
    with Alone(m, gid):
        for i, (x, y) in enumerate(np.asarray(XY, np.float64)):
            dist = mujoco.mj_ray(m, d, np.array([x + NUDGE[0], y + NUDGE[1], 2.0]), v, gg, 1, -1, hit)
            if dist >= 0:
                out[i] = 2.0 - dist
    return out


def lattice_hits(m, d, gid, XY):
    """3-D hit points (n x 3, nan where missed) of rays from 1e4 m above the lattice's centre towards each point."""
    import mujoco
    XY = np.asarray(XY, np.float64)
    tgt = np.c_[XY[:, 0] + NUDGE[0], XY[:, 1] + NUDGE[1], np.zeros(len(XY))]
    o = np.array([0.5 * (XY[:, 0].min() + XY[:, 0].max()), 0.5 * (XY[:, 1].min() + XY[:, 1].max()), HIGH])
    vec = tgt - o
    vec /= np.linalg.norm(vec, axis=1)[:, None]
    n = len(XY)
    gidout = np.zeros(n, np.int32)
    dist = np.zeros(n, np.float64)
    with Alone(m, gid):
        mujoco.mj_multiRay(m, d, o, vec.reshape(-1).copy(), _grp(), 1, -1, gidout, dist, None, n, 2.0 * HIGH)
    hit = o + vec * dist[:, None]
    hit[dist < 0] = np.nan
    return hit


def lattice(rect, line, step, pad_free, pad_line=0.0):
    """Lattice over rect (x0, x1, y0, y1) + pad_free on the three free sides and pad_line beyond the line side.
    Returns (xs, ys) in the heap's LOCAL row order: rows run away from the line's far side towards the line (the line is
    the last row's side), so the arrays read as in a line-north frame for either line."""
    x0, x1, y0, y1 = rect
    xs = np.arange(x0 - pad_free, x1 + pad_free + 1e-9, step)
    if line == "north":
        ys = np.arange(y0 - pad_free, y1 + pad_line + 1e-9, step)
    else:
        ys = (y1 + pad_free) - np.arange(0.0, (y1 + pad_free) - (y0 - pad_line) + 1e-9, step)
    return xs, ys


def heights(m, d, gid, xs, ys):
    X, Y = np.meshgrid(xs, ys)
    hit = lattice_hits(m, d, gid, np.c_[X.ravel(), Y.ravel()])
    Z = hit[:, 2].reshape(X.shape)
    return X, Y, Z, hit


def shape(m, d, gid, heap):
    """Class geometry of one heap on the compiled model (local frame: line = last row)."""
    from scipy import ndimage
    line = heap["line"]
    xs, ys = lattice(heap["rect"], line, RAY_STEP, 0.02)
    X, Y, Zc, _hit = heights(m, d, gid, xs, ys)
    Zc = np.where(np.isnan(Zc), -1.0, Zc)
    F = Zc > 0.001
    res = dict(n_ray=int(Zc.size))
    if not F.any():
        res["empty"] = True
        return res
    rows, cols = np.nonzero(F)
    wx = X[F]
    wy = Y[F]
    res["footprint_rect_world"] = [float(wx.min()), float(wx.max()), float(wy.min()), float(wy.max())]
    res["W_meas"] = float(wx.max() - wx.min())
    res["D_meas"] = float(wy.max() - wy.min())
    res["top_meas"] = float(Zc.max())
    padn = 12
    Fp = np.concatenate([F, np.ones((padn, F.shape[1]), bool)], axis=0)
    Fp = np.pad(Fp, ((8, 0), (8, 8)), constant_values=False)
    s = (ndimage.distance_transform_edt(Fp)[8:8 + F.shape[0], 8:8 + F.shape[1]] - 0.5) * RAY_STEP
    ring = F & (np.abs(s - EDGE_IN) <= RAY_STEP / 2.0 + 1e-12)
    ring &= np.arange(F.shape[0])[:, None] < rows.max() - 0.5 + 0 * F
    ok = Zc >= EDGE_MIN
    # local coordinates: u along the line (local W -> E), w away from the far side (S) towards the line (N)
    U = np.arange(F.shape[1])[None, :] * RAY_STEP + 0 * Zc
    Wl = np.arange(F.shape[0])[:, None] * RAY_STEP + 0 * Zc
    ru0, ru1 = U[0, cols.min()], U[0, cols.max()]
    rw0 = Wl[rows.min(), 0]
    dist_side = np.stack([U - ru0, Wl - rw0, ru1 - U], 0)
    side = np.argmin(dist_side, 0)
    per = {}
    for k, nm in enumerate("WSE"):
        sel = ring & (side == k)
        if sel.sum() >= 4:
            per[nm] = float(ok[sel].mean())
    res["edge_rule_frac_by_side"] = per
    res["edge_rule_min_side"] = min(per.values()) if per else None
    gy, gx = np.gradient(np.where(F, Zc, 0.0), RAY_STEP)
    band = F & (Zc >= 0.03) & (Zc <= 0.8 * res["top_meas"])
    inner = ndimage.binary_erosion(F, iterations=2)
    sl = np.degrees(np.arctan(np.hypot(gx, gy)))[band & inner]
    res["slope_median_deg"] = float(np.median(sl)) if sl.size else None
    res["vertical_share"] = float((sl > 75.0).mean()) if sl.size else None
    cu, cw = float(U[F].mean()), float(Wl[F].mean())
    QU, QW = cu + 2.0 * (U - cu), cw + 2.0 * (Wl - cw)
    qi = np.round(QW / RAY_STEP).astype(int)
    qj = np.round(QU / RAY_STEP).astype(int)
    okq = (qi >= 0) & (qi < F.shape[0]) & (qj >= 0) & (qj < F.shape[1])
    core = np.zeros_like(F)
    core[okq] = F[qi[okq], qj[okq]]
    res["core_ge010_frac"] = float((Zc[core] >= 0.10).mean()) if core.any() else None
    x0, x1 = heap["rect"][0], heap["rect"][1]
    y_line = heap.get("y_line", heap["rect"][3] if line == "north" else heap["rect"][2])
    wr = x1 - x0
    G = Zc > LINE_Z
    gaps = []
    for j in range(G.shape[1]):
        xj = float(X[0, j])
        if not (x0 + 0.2 * wr <= xj <= x1 - 0.2 * wr):
            continue
        yy = Y[G[:, j], j]
        if yy.size:
            gaps.append(float(y_line - yy.max()) if line == "north" else float(yy.min() - y_line))
    res["line_gap_mid60_max_m"] = max(gaps) if gaps else None
    res["line_gap_mid60_median_m"] = float(np.median(gaps)) if gaps else None
    return res


# ------------------------------------------------------------------------------------------------ fit bound
def zbuf(V, F, x0, y0, nx, ny, res):
    """max-z raster of a triangle mesh at cell centres (numpy, chunked); nan where no triangle covers a cell."""
    Z = np.full(ny * nx, -np.inf)
    P = V[F]
    for a in range(0, len(F), 20000):
        T = P[a:a + 20000]
        mn = T.min(1)
        mx = T.max(1)
        i0 = np.floor((mn[:, 0] - x0) / res - 0.5).astype(int).clip(0, nx - 1)
        i1 = np.ceil((mx[:, 0] - x0) / res - 0.5).astype(int).clip(0, nx - 1)
        j0 = np.floor((mn[:, 1] - y0) / res - 0.5).astype(int).clip(0, ny - 1)
        j1 = np.ceil((mx[:, 1] - y0) / res - 0.5).astype(int).clip(0, ny - 1)
        span = (i1 - i0 + 1) * (j1 - j0 + 1)
        for s in np.unique(span):
            sel = np.nonzero(span == s)[0]
            w = (i1 - i0 + 1)[sel]
            for k_ in range(int(s)):
                ii = i0[sel] + k_ % w
                jj = j0[sel] + k_ // w
                ok = (ii <= i1[sel]) & (jj <= j1[sel])
                if not ok.any():
                    continue
                t = T[sel[ok]]
                px = x0 + (ii[ok] + 0.5) * res
                py = y0 + (jj[ok] + 0.5) * res
                ax, ay, az = t[:, 0, 0], t[:, 0, 1], t[:, 0, 2]
                bx, by, bz = t[:, 1, 0], t[:, 1, 1], t[:, 1, 2]
                cx, cy, cz = t[:, 2, 0], t[:, 2, 1], t[:, 2, 2]
                det = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
                det = np.where(np.abs(det) < 1e-18, np.nan, det)
                l1 = ((by - cy) * (px - cx) + (cx - bx) * (py - cy)) / det
                l2 = ((cy - ay) * (px - cx) + (ax - cx) * (py - cy)) / det
                l3 = 1 - l1 - l2
                ins = (l1 >= -1e-9) & (l2 >= -1e-9) & (l3 >= -1e-9)
                z = l1 * az + l2 * bz + l3 * cz
                idx = jj[ok][ins] * nx + ii[ok][ins]
                np.maximum.at(Z, idx, z[ins])
    Z = Z.reshape(ny, nx)
    Z[np.isinf(Z)] = np.nan
    return Z


def envelope_cloud(V, rect, F_, res=0.001, pad=0.03):
    """Points of the mesh's upper envelope (floor z = 0 included) on a 1 mm raster over rect + pad, cliffs between
    neighbouring cells densified at 1 mm in z."""
    x0, x1, y0, y1 = rect
    xs0, ys0 = x0 - pad, y0 - pad
    nx = int(math.ceil((x1 - x0 + 2 * pad) / res))
    ny = int(math.ceil((y1 - y0 + 2 * pad) / res))
    Z = zbuf(V, F_, xs0, ys0, nx, ny, res)
    E = np.where(np.isfinite(Z), np.maximum(Z, 0.0), 0.0)
    xs = xs0 + (np.arange(nx) + 0.5) * res
    ys = ys0 + (np.arange(ny) + 0.5) * res
    X, Y = np.meshgrid(xs, ys)
    pts = [np.stack([X.ravel(), Y.ravel(), E.ravel()], 1)]
    for ax in (0, 1):
        a = E[:-1, :] if ax == 0 else E[:, :-1]
        b = E[1:, :] if ax == 0 else E[:, 1:]
        xa = X[:-1, :] if ax == 0 else X[:, :-1]
        ya = Y[:-1, :] if ax == 0 else Y[:, :-1]
        dz = np.abs(a - b)
        sel = dz > res
        if not sel.any():
            continue
        lo = np.minimum(a, b)[sel]
        hi = np.maximum(a, b)[sel]
        xm = xa[sel] + (0 if ax == 0 else 0.5 * res)
        ym = ya[sel] + (0.5 * res if ax == 0 else 0)
        n = np.ceil((hi - lo) / res).astype(int)
        rep = np.repeat(np.arange(len(n)), n)
        k = np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n)
        pts.append(np.stack([xm[rep], ym[rep], lo[rep] + (k + 0.5) * res], 1))
    return np.concatenate(pts, 0)


def fit_bound(m, d, gid, heap, V_site, F_):
    """Compiled height field (1 mm lattice, points above 1 mm) against the placed mesh's upper envelope."""
    from scipy.spatial import cKDTree
    xs, ys = lattice(heap["rect"], heap["line"], 0.001, 0.01)
    X, Y = np.meshgrid(xs, ys)
    hit = lattice_hits(m, d, gid, np.c_[X.ravel(), Y.ravel()])
    sel = np.isfinite(hit[:, 2]) & (hit[:, 2] > 0.001)
    P = hit[sel]
    EP = envelope_cloud(V_site, heap["rect"], F_)
    dist, _ = cKDTree(EP).query(P, k=1)
    return dict(n=int(len(P)), share_le_3mm=float((dist <= 0.003).mean()), max_mm=float(1000 * dist.max()),
                p99_mm=float(1000 * np.percentile(dist, 99)), mean_mm=float(1000 * dist.mean()))


# ------------------------------------------------------------------------------------------------ the compiled grid
def compiled_grid(m, gid):
    """Facts of the compiled height field of geom gid: size, zero ring, max, world origin of its vertex lattice."""
    hid = int(m.geom_dataid[gid])
    nrow, ncol = int(m.hfield_nrow[hid]), int(m.hfield_ncol[hid])
    adr = int(m.hfield_adr[hid])
    data = np.asarray(m.hfield_data[adr:adr + nrow * ncol], np.float64).reshape(nrow, ncol)
    size = [float(q) for q in m.hfield_size[hid]]
    ring = np.concatenate([data[0, :], data[-1, :], data[:, 0], data[:, -1]])
    return dict(nrow=nrow, ncol=ncol, size=size, data_min=float(data.min()), data_max=float(data.max()),
                ring_zero=bool(np.all(ring == 0.0)), max_is_one=bool(float(data.max()) == 1.0),
                max_height_m=float(size[2] * data.max()), data=data)


# ------------------------------------------------------------------------------------------------ occupancy and gaps
def heap_occupancy(m, d, gid, rect, step=OCC_STEP, pad=0.02):
    """(x, y) of the 5 mm lattice points over rect + pad where the heap stands > 3 mm above the floor."""
    x0, x1, y0, y1 = rect
    gx0 = math.floor(round((x0 - pad) / step, 6)) * step
    gy0 = math.floor(round((y0 - pad) / step, 6)) * step
    xs = gx0 + step * np.arange(int(math.ceil((x1 + pad - gx0) / step)) + 1)
    ys = gy0 + step * np.arange(int(math.ceil((y1 + pad - gy0) / step)) + 1)
    X, Y = np.meshgrid(xs, ys)
    hit = lattice_hits(m, d, gid, np.c_[X.ravel(), Y.ravel()])
    z = hit[:, 2]
    sel = np.isfinite(z) & (z > OCC_Z)
    return np.c_[X.ravel()[sel], Y.ravel()[sel]]


def floor_boxes(m, d):
    """[(name, (x0, x1, y0, y1))] of the compiled model's colliding box geoms that stand on the floor (their z range
    holds 3 mm), axis-aligned ones only (a turned box here would be a roof slab, which does not reach the floor)."""
    import mujoco
    out = []
    for i in range(m.ngeom):
        if int(m.geom_type[i]) != int(mujoco.mjtGeom.mjGEOM_BOX):
            continue
        if not (m.geom_contype[i] or m.geom_conaffinity[i]):
            continue
        nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i) or ""
        if nm.startswith("patch__"):
            continue
        p = d.geom_xpos[i]
        s = m.geom_size[i]
        R = d.geom_xmat[i].reshape(3, 3)
        if not (p[2] - s[2] <= OCC_Z <= p[2] + s[2]):
            continue
        if np.abs(R - np.eye(3)).max() > 1e-9:
            continue
        out.append((nm, (float(p[0] - s[0]), float(p[0] + s[0]), float(p[1] - s[1]), float(p[1] + s[1]))))
    return out


def floor_box_tops(m, d):
    """{name: top z} of the boxes floor_boxes returns."""
    import mujoco
    out = {}
    names = dict(floor_boxes(m, d))
    for i in range(m.ngeom):
        nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i) or ""
        if nm in names:
            out[nm] = float(d.geom_xpos[i][2] + m.geom_size[i][2])
    return out


def buried(P, boxes, tops, z_min):
    """Boolean per point (n x 2): inside the (x, y) rectangle of a floor box whose top is >= z_min."""
    P = np.asarray(P, np.float64).reshape(-1, 2)
    ok = np.zeros(len(P), bool)
    for n, r in boxes:
        if tops.get(n, -1.0) >= z_min:
            ok |= (P[:, 0] >= r[0] - 1e-9) & (P[:, 0] <= r[1] + 1e-9) & (P[:, 1] >= r[2] - 1e-9) & (P[:, 1] <= r[3] + 1e-9)
    return ok


def _rect_pts_dist(r, P):
    dx = np.maximum(np.maximum(r[0] - P[:, 0], 0.0), P[:, 0] - r[1])
    dy = np.maximum(np.maximum(r[2] - P[:, 1], 0.0), P[:, 1] - r[3])
    return np.hypot(dx, dy)


def _closest_on_rect(r, P):
    return np.c_[np.clip(P[:, 0], r[0], r[1]), np.clip(P[:, 1], r[2], r[3])]


def gap_promise(heaps_pts, boxes, lines=None, lo=0.15, hi=0.60, err=OCC_STEP * math.sqrt(2.0), reach=0.65,
                n_samples=130):
    """The gap promise for every heap against every other floor entity.

    heaps_pts: {name: (n x 2) occupied lattice points on the 5 mm lattice}; boxes: [(name, rect)]. For a heap A and an
    entity B the gap is the shortest segment from a boundary point of A to B (nearest point of B per boundary point)
    that crosses no third entity: a 5 mm label raster of every entity (box cells by their centres, heap cells by their
    lattice points) is read at 130 points along the segment; a segment that crosses another entity is not a gap
    across open floor (the passage between A and B is then the gaps to that entity). Lattice points lie inside the
    solids, so a measured gap g satisfies true <= g <= true + err. Kept: g < lo, or g - err >= hi, or no unblocked
    segment shorter than reach.
    lines: {heap: [entity names]}: the run of boxes the heap leans on (boxes with a face on the heap's line whose x
    intervals touch one another, from the one behind the heap; line_run()) is ONE entity, "line", for that heap: the
    gap to it is the gap to the nearest of them. A box on the same line beyond an opening of the line is an entity of
    its own."""
    from scipy.spatial import cKDTree
    names = list(heaps_pts) + [b[0] for b in boxes]
    idx = {n: k for k, n in enumerate(names)}
    rects = dict(boxes)
    allx = [r[0] for r in rects.values()] + [r[1] for r in rects.values()]
    ally = [r[2] for r in rects.values()] + [r[3] for r in rects.values()]
    for p in heaps_pts.values():
        if len(p):
            allx += [float(p[:, 0].min()), float(p[:, 0].max())]
            ally += [float(p[:, 1].min()), float(p[:, 1].max())]
    gx0 = math.floor(min(allx) / OCC_STEP) - 2
    gy0 = math.floor(min(ally) / OCC_STEP) - 2
    nx = math.ceil(max(allx) / OCC_STEP) - gx0 + 3
    ny = math.ceil(max(ally) / OCC_STEP) - gy0 + 3
    lab = np.full((ny, nx), -1, np.int32)
    cx = (gx0 + np.arange(nx)) * OCC_STEP
    cy = (gy0 + np.arange(ny)) * OCC_STEP
    for n, r in boxes:
        j0, j1 = np.searchsorted(cx, r[0] + 1e-9), np.searchsorted(cx, r[1] - 1e-9)
        i0, i1 = np.searchsorted(cy, r[2] + 1e-9), np.searchsorted(cy, r[3] - 1e-9)
        if j1 > j0 and i1 > i0:
            lab[i0:i1, j0:j1] = idx[n]
    cells = {}
    for n, p in heaps_pts.items():
        if not len(p):
            continue
        ij = np.round(p / OCC_STEP).astype(np.int64) - np.array([gx0, gy0])
        lab[ij[:, 1], ij[:, 0]] = idx[n]
        cells[n] = ij
    trees = {n: cKDTree(p) for n, p in heaps_pts.items() if len(p)}
    tt = np.linspace(0.0, 1.0, n_samples + 2)[1:-1]
    out, ok_all = {}, True
    for a, PA in heaps_pts.items():
        if not len(PA):
            continue
        ij = cells[a]
        occ = np.zeros((ny, nx), bool)
        occ[ij[:, 1], ij[:, 0]] = True
        inner = (occ[ij[:, 1] + 1, ij[:, 0]] & occ[ij[:, 1] - 1, ij[:, 0]] & occ[ij[:, 1], ij[:, 0] + 1]
                 & occ[ij[:, 1], ij[:, 0] - 1])
        BA = PA[~inner]
        ra = (float(PA[:, 0].min()), float(PA[:, 0].max()), float(PA[:, 1].min()), float(PA[:, 1].max()))
        rows = {}
        own_line = set((lines or {}).get(a, ()))
        line_rows = []
        for b in names:
            if b == a:
                continue
            if b in rects:
                rb = rects[b]
            elif b in trees:
                pb = heaps_pts[b]
                rb = (float(pb[:, 0].min()), float(pb[:, 0].max()), float(pb[:, 1].min()), float(pb[:, 1].max()))
            else:
                continue
            if math.hypot(max(0.0, rb[0] - ra[1], ra[0] - rb[1]), max(0.0, rb[2] - ra[3], ra[2] - rb[3])) >= reach:
                continue
            if b in rects:
                dist = _rect_pts_dist(rb, BA)
                near = _closest_on_rect(rb, BA)
            else:
                dist, k_ = trees[b].query(BA, k=1)
                near = heaps_pts[b][k_]
            cand = np.nonzero(dist < reach)[0]
            if not len(cand):
                continue
            P, Q = BA[cand], near[cand]
            S = P[:, None, :] + tt[None, :, None] * (Q - P)[:, None, :]
            si = np.round(S / OCC_STEP).astype(np.int64)
            jj = np.clip(si[..., 0] - gx0, 0, nx - 1)
            ii = np.clip(si[..., 1] - gy0, 0, ny - 1)
            L = lab[ii, jj]
            blocked = ((L >= 0) & (L != idx[a]) & (L != idx[b])).any(axis=1)
            dc = dist[cand]
            row = {}
            if (~blocked).any():
                g = float(dc[~blocked].min())
                row["gap_m"] = round(g, 4)
                row["ok"] = bool(g < lo or g - err >= hi)
            else:
                row["gap_m"] = None
                row["ok"] = True
            if blocked.any():
                k = int(np.argmin(np.where(blocked, dc, np.inf)))
                row["blocked_min_m"] = round(float(dc[k]), 4)
                bl = L[k][(L[k] >= 0) & (L[k] != idx[a]) & (L[k] != idx[b])]
                row["blocked_by"] = names[int(bl[0])] if len(bl) else None
            if b in own_line:
                line_rows.append((b, row))
                continue
            ok_all &= row["ok"]
            rows[b] = row
        if own_line:
            gl = [r["gap_m"] for _b, r in line_rows if r.get("gap_m") is not None]
            row = dict(gap_m=min(gl) if gl else None, members=sorted(own_line))
            row["ok"] = bool(row["gap_m"] is not None and row["gap_m"] < lo)
            ok_all &= row["ok"]
            rows["line"] = row
        out[a] = rows
    return dict(ok=bool(ok_all), pairs=out, rule="gap < %.2f m or >= %.2f m (lattice error %.4f m)" % (lo, hi, err))


# ------------------------------------------------------------------------------------------------ what the camera sees
def visibility(m, d, gid, rect, cpos, cam=None, step=0.01, z_min=0.02, lift=0.004, eps_hit=0.015):
    """Share of the heap's surface (1 cm lattice of the compiled surface above 2 cm, weighted by surface area) that the
    search camera sees with every collision geom as an occluder, and whether its top (within 1 cm of its highest point)
    is seen. A sample counts as seen when it projects inside the image (cam = (R_world_camera, model): pixel centres
    0 .. W - 1, 0 .. H - 1) and the camera's ray meets nothing more than 1.5 cm before the point 4 mm above the
    surface along its normal."""
    import mujoco
    x0, x1, y0, y1 = rect
    xs = np.arange(x0 + step / 2, x1, step)
    ys = np.arange(y0 + step / 2, y1, step)
    X, Y = np.meshgrid(xs, ys)
    hit = lattice_hits(m, d, gid, np.c_[X.ravel(), Y.ravel()])
    Z = hit[:, 2].reshape(X.shape)
    Zf = np.where(np.isfinite(Z), Z, 0.0)
    gy, gx = np.gradient(Zf, step)
    nrm = np.stack([-gx, -gy, np.ones_like(Zf)], -1)
    w = np.linalg.norm(nrm, axis=-1)
    nrm = nrm / w[..., None]
    sel = np.isfinite(Z) & (Z >= z_min)
    if not sel.any():
        return dict(share=0.0, top_seen=False, n=0)
    P = hit.reshape(X.shape + (3,))[sel] + lift * nrm[sel]
    vec = P - cpos
    dt = np.linalg.norm(vec, axis=1)
    vn = (vec / dt[:, None]).copy()
    n = len(P)
    gidout = np.zeros(n, np.int32)
    dist = np.zeros(n, np.float64)
    grp = np.zeros(6, np.uint8)
    grp[0] = 1
    mujoco.mj_multiRay(m, d, np.asarray(cpos, np.float64), vn.reshape(-1), grp, 1, -1, gidout, dist, None, n, 50.0)
    seen = ~((dist >= 0) & (dist < dt - eps_hit))
    in_frame = np.ones(n, bool)
    if cam is not None:
        R, cm = cam
        pc = (P - np.asarray(cpos, np.float64)) @ np.asarray(R, np.float64)
        u = cm["fx"] * pc[:, 0] / pc[:, 2] + cm["cx"]
        v = cm["fy"] * pc[:, 1] / pc[:, 2] + cm["cy"]
        in_frame = (pc[:, 2] > 0) & (u >= 0) & (u <= cm["width"] - 1) & (v >= 0) & (v <= cm["height"] - 1)
    seen &= in_frame
    wt = w[sel]
    top = Z[sel] >= np.nanmax(Z[sel]) - 0.01
    return dict(share=float((wt * seen).sum() / wt.sum()), top_seen=bool(seen[top].any()), n=int(n),
                in_frame_share=float(in_frame.mean()), top_m=float(np.nanmax(Z[sel])))


def line_run(boxes, rect, line, y_line, tol=1e-5):
    """Names of the boxes that form the line a heap leans on: boxes with a face on y = y_line on the line's side
    (north: their south face; south: their north face), grouped into runs whose x intervals touch; the runs that
    overlap the heap's x interval. Also returns the share of the heap's x interval those runs cover."""
    on = [(n, r) for n, r in boxes if (abs(r[2] - y_line) <= tol if line == "north" else abs(r[3] - y_line) <= tol)]
    on.sort(key=lambda q: (q[1][0], q[1][1]))
    runs, cur, hi_ = [], [], None
    for n, r in on:
        if cur and r[0] <= hi_ + tol:
            cur.append((n, r))
            hi_ = max(hi_, r[1])
        else:
            if cur:
                runs.append(cur)
            cur, hi_ = [(n, r)], r[1]
    if cur:
        runs.append(cur)
    x0, x1 = rect[0], rect[1]
    names, covered = [], 0.0
    for run in runs:
        a0, a1 = min(r[0] for _n, r in run), max(r[1] for _n, r in run)
        ov = min(a1, x1) - max(a0, x0)
        if ov > tol:
            names += [n for n, _r in run]
            covered += ov
    return names, covered / max(x1 - x0, 1e-12)
