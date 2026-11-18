# The code below is the code that built the reference tables of tests/reference/ (the hole steps and flat faces
# of features.json); the hole and flat-face items run it unchanged on the submitted parts, so that both sides are
# measured alike. Only its comments and docstrings were edited since: the syntax tree without docstrings is the
# one the reference tables were built with.
"""Find flat faces, concave cylinders (holes), spheres and radial profiles on a triangle mesh.

Pure numpy (no FreeCAD, no trimesh, no scipy), so that the reference tables (built from the authors' mesh pieces) and
the grader's submission side run the same code.

Every function works on a triangle mesh given as V (n x 3 float, mm) and F (m x 3 int). Nothing here builds, repairs
or changes geometry: it only groups the given triangles and fits primitives to the given vertices.

The rules (fixed; the same for every design; recorded in features.json under "rules"):

  PLANE_RULE  flat face = a set of edge-connected triangles whose adjacent normals differ by <= coplanar_deg;
              its area is >= area_min_mm2; and
                (a) it is not a strip by the design-time test: the share of its boundary length that is "smooth"
                    (angle to the neighbouring cluster's normal <= smooth_deg) is <= smooth_share_max, or its area
                    is >= big_ratio x the mean area of its smooth neighbours (mean over the smooth boundary edges);
                (c) it is flat: every vertex within flatness_max_mm of its plane (a fan of nearly
                    coplanar triangles at the pole of a faceted sphere chains into one cluster but is not flat);
                (b) it is not part of a fitted curved surface (fitted_surfaces, strip_of): its
                    triangles are not in a smooth component that fits a cylinder, sphere or cone, and it is not a
                    narrow strip (angular width <= smooth_deg, normal radial) along a fitted cylinder's axis (all
                    vertices at one distance from it, axial extent >= 2 x chord), or on a fitted sphere or cone.
  HOLE_RULE   hole step = a set of edge-connected triangles whose adjacent normals differ by < smooth_deg, at least
              min_facets of them, whose normals span a plane (smallest eigenvalue of the normal covariance
              <= ev0_max, middle one >= ev1_min), whose vertices fit a circle about the normal-plane axis with
              max residual <= max(res_abs_mm, res_rel x r), and whose area-weighted normals point towards the
              axis (concave). Steps that are coaxial (axis angle <= coax_deg, axis-line distance <= coax_mm) and
              touch along the axis (gap <= touch_mm) are one hole.
"""
import math

import numpy as np

PLANE_RULE = {
    "coplanar_deg": 0.2,
    "area_min_mm2": 2.0,
    "smooth_deg": 20.0,
    "smooth_share_max": 0.5,
    "big_ratio": 4.0,
    "flatness_max_mm": 0.005,
    "surface_smooth_deg": 14.0,
    "surface_fit_res_mm": 0.005,
    "on_surface_tol_mm": 0.005,
}

HOLE_RULE = {
    "smooth_deg": 14.0,
    "min_facets": 6,
    "ev0_max": 2e-3,
    "ev1_min": 0.02,
    "res_abs_mm": 0.02,
    "res_rel": 0.02,
    "coax_deg": 0.5,
    "coax_mm": 0.02,
    "touch_mm": 0.05,
    "span_bin_deg": 0.1,
}


# ----------------------------------------------------------------------------------------------------------- reading
def read_ply(path):
    """Read a binary little-endian or ASCII PLY with float/double x y z and a uchar-count int face list (the
    reference files). Returns V (float64) and F (int64)."""
    with open(path, "rb") as fh:
        raw = fh.read()
    end = raw.index(b"end_header") + len(b"end_header")
    while raw[end:end + 1] in (b"\r", b"\n"):
        end += 1
        if raw[end - 1:end] == b"\n":
            break
    header = raw[:end].decode("ascii", "replace").splitlines()
    fmt, nv, nf, vprops, flist = None, 0, 0, [], None
    cur = None
    for line in header:
        tok = line.split()
        if not tok:
            continue
        if tok[0] == "format":
            fmt = tok[1]
        elif tok[0] == "element":
            cur = tok[1]
            if cur == "vertex":
                nv = int(tok[2])
            elif cur == "face":
                nf = int(tok[2])
        elif tok[0] == "property" and cur == "vertex":
            vprops.append((tok[-1], tok[1]))
        elif tok[0] == "property" and cur == "face" and tok[1] == "list":
            flist = (tok[2], tok[3])
    tmap = {"float": "<f4", "float32": "<f4", "double": "<f8", "float64": "<f8", "uchar": "u1", "uint8": "u1",
            "int": "<i4", "int32": "<i4", "uint": "<u4", "uint32": "<u4", "short": "<i2", "ushort": "<u2"}
    body = raw[end:]
    if fmt == "ascii":
        vals = body.split()
        k = len(vprops)
        Vall = np.array(vals[:nv * k], float).reshape(nv, k)
        names = [p[0] for p in vprops]
        V = Vall[:, [names.index("x"), names.index("y"), names.index("z")]]
        rest = np.array(vals[nv * k:], np.int64)
        F = rest.reshape(nf, 4)[:, 1:]
        return V.astype(np.float64), F.astype(np.int64)
    if fmt != "binary_little_endian":
        raise ValueError("unsupported PLY format %r" % fmt)
    vdt = np.dtype([(n, tmap[t]) for n, t in vprops])
    varr = np.frombuffer(body, dtype=vdt, count=nv)
    V = np.stack([varr["x"], varr["y"], varr["z"]], axis=1).astype(np.float64)
    off = vdt.itemsize * nv
    cdt, idt = tmap[flist[0]], tmap[flist[1]]
    fdt = np.dtype([("n", cdt), ("v", idt, (3,))])
    farr = np.frombuffer(body, dtype=fdt, count=nf, offset=off)
    if np.any(farr["n"] != 3):
        raise ValueError("non-triangle faces")
    return V, farr["v"].astype(np.int64)


# ---------------------------------------------------------------------------------------------------------- topology
def face_geometry(V, F):
    """unit normals (zero for degenerate faces), areas, centroids"""
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    cr = np.cross(b - a, c - a)
    nrm = np.linalg.norm(cr, axis=1)
    N = np.where(nrm[:, None] > 0, cr / np.maximum(nrm, 1e-300)[:, None], 0.0)
    return N, 0.5 * nrm, (a + b + c) / 3.0


def edge_adjacency(F):
    """Pairs of faces that share an edge: (fa, fb, va, vb) with va < vb the shared edge's vertices. An edge used by
    k > 2 faces links its first face (lowest index) to every other one. Also returns the boundary edges (k == 1)."""
    F = np.asarray(F, np.int64)
    m = len(F)
    e = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    fid = np.concatenate([np.arange(m)] * 3)
    es = np.sort(e, axis=1)
    base = int(es.max()) + 1 if len(es) else 1
    key = es[:, 0] * base + es[:, 1]
    order = np.lexsort((fid, key))
    k = key[order]
    starts = np.r_[0, np.nonzero(np.diff(k))[0] + 1]
    ends = np.r_[starts[1:], len(k)]
    cnt = ends - starts
    fa, fb, va, vb = [], [], [], []
    two = cnt == 2
    s2 = starts[two]
    fa.append(fid[order[s2]])
    fb.append(fid[order[s2 + 1]])
    va.append(es[order[s2], 0])
    vb.append(es[order[s2], 1])
    for s, c in zip(starts[cnt > 2], cnt[cnt > 2]):
        for j in range(1, int(c)):
            fa.append(fid[order[[s]]])
            fb.append(fid[order[[s + j]]])
            va.append(es[order[[s]], 0])
            vb.append(es[order[[s]], 1])
    one = cnt == 1
    boundary = es[order[starts[one]]]
    return (np.concatenate(fa).astype(np.int64), np.concatenate(fb).astype(np.int64),
            np.concatenate(va).astype(np.int64), np.concatenate(vb).astype(np.int64), boundary)


def connected_labels(n, a, b):
    """Connected components of the graph on n nodes with edges (a[i], b[i]); labels 0..k-1 ordered by the smallest
    node index of each component (deterministic)."""
    lab = np.arange(n, dtype=np.int64)
    a = np.asarray(a, np.int64)
    b = np.asarray(b, np.int64)
    if len(a):
        while True:
            la, lb = lab[a], lab[b]
            m = np.minimum(la, lb)
            new = lab.copy()
            np.minimum.at(new, la, m)  # hook the roots
            np.minimum.at(new, lb, m)
            np.minimum.at(new, a, m)
            np.minimum.at(new, b, m)
            while True:  # pointer jumping
                nn = new[new]
                if np.array_equal(nn, new):
                    break
                new = nn
            if np.array_equal(new, lab):
                break
            lab = new
    _u, inv = np.unique(lab, return_inverse=True)
    return inv.astype(np.int64)


def angle_deg(u, v):
    return np.degrees(np.arccos(np.clip(np.einsum("ij,ij->i", u, v), -1.0, 1.0)))


class Mesh:
    """Topology and per-face geometry computed once."""

    def __init__(self, V, F):
        self.V = np.asarray(V, np.float64)
        self.F = np.asarray(F, np.int64)
        self.N, self.A, self.C = face_geometry(self.V, self.F)
        self.fa, self.fb, self.va, self.vb, self.boundary = edge_adjacency(self.F)
        self.elen = np.linalg.norm(self.V[self.va] - self.V[self.vb], axis=1)
        self.dih = angle_deg(self.N[self.fa], self.N[self.fb])

    def components(self, max_deg, strict=False):
        keep = self.dih < max_deg if strict else self.dih <= max_deg
        return connected_labels(len(self.F), self.fa[keep], self.fb[keep])


# ------------------------------------------------------------------------------------------------------------- fits
def fit_circle_2d(x, y):
    A = np.c_[2 * x, 2 * y, np.ones(len(x))]
    sol, *_ = np.linalg.lstsq(A, x * x + y * y, rcond=None)
    cx, cy = sol[0], sol[1]
    r2 = sol[2] + cx * cx + cy * cy
    if r2 <= 0:
        return None
    r = math.sqrt(r2)
    res = np.abs(np.hypot(x - cx, y - cy) - r)
    return cx, cy, r, res


def perp_basis(d):
    d = d / np.linalg.norm(d)
    h = np.array([1.0, 0.0, 0.0]) if abs(d[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = np.cross(d, h)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(d, e1)
    return e1, e2


def canonical_dir(d):
    """sign convention for an axis: largest |component| positive, ties resolved z, y, x"""
    d = np.asarray(d, float) / np.linalg.norm(d)
    i = int(np.argmax(np.abs(d) + np.array([0.0, 1e-12, 2e-12])))
    return -d if d[i] < 0 else d


def fit_cylinder(P, N, w):
    """Axis from the normals (smallest-eigenvalue direction of the normal covariance), centre and radius from a
    circle fit of the points projected on the plane normal to the axis. Returns dict or None."""
    C = (N * w[:, None]).T @ N / w.sum()
    ev, E = np.linalg.eigh(C)
    d = canonical_dir(E[:, 0])
    e1, e2 = perp_basis(d)
    fit = fit_circle_2d(P @ e1, P @ e2)
    if fit is None:
        return None
    cx, cy, r, res = fit
    centre = cx * e1 + cy * e2
    return {"dir": d, "centre": centre, "radius": r, "res": res, "ev": ev}


def fit_sphere(P):
    A = np.c_[2 * P, np.ones(len(P))]
    sol, *_ = np.linalg.lstsq(A, (P * P).sum(1), rcond=None)
    c = sol[:3]
    r2 = sol[3] + c @ c
    if r2 <= 0:
        return None
    r = math.sqrt(r2)
    res = np.abs(np.linalg.norm(P - c, axis=1) - r)
    return {"centre": c, "radius": r, "res": res}


def fit_cone(P, N, w, Fc):
    """Cone through the facet planes: axis = direction a with n . a constant (smallest eigenvalue of the covariance
    of the centred normals), apex = least-squares common point of the facet planes. Fc = facet centroids."""
    nm = (N * w[:, None]).sum(0) / w.sum()
    D = N - nm
    C = (D * w[:, None]).T @ D / w.sum()
    ev, E = np.linalg.eigh(C)
    a = E[:, 0]
    if abs(nm @ a) < 1e-3:  # normals perpendicular to the axis: a cylinder, not a cone
        return None
    sol, *_ = np.linalg.lstsq(N * np.sqrt(w)[:, None], np.einsum("ij,ij->i", N, Fc) * np.sqrt(w), rcond=None)
    apex = sol
    v = P - apex
    t = v @ a
    if np.all(t <= 0):
        a, t = -a, -t
    rho = np.linalg.norm(v - np.outer(t, a), axis=1)
    half = math.atan2(float(np.sum(rho * t)), float(np.sum(t * t)))  # rho = t tan(half), least squares
    res = np.abs(rho * math.cos(half) - t * math.sin(half))  # distance to the cone surface (for t > 0)
    return {"dir": canonical_dir(a), "apex": apex, "half_angle_deg": math.degrees(half), "res": res}


def angular_span(P, centre, d, faces_local, bin_deg):
    """Angle about the axis covered by the facets (union of each facet's vertex-angle interval), in degrees, and the
    median angular width of a facet."""
    e1, e2 = perp_basis(d)
    q = P - centre
    ang = np.arctan2(q @ e2, q @ e1)
    nb = int(round(360.0 / bin_deg))
    cover = np.zeros(nb, bool)
    widths = []
    for tri in faces_local:
        a = ang[tri]
        a = a[0] + ((a - a[0] + np.pi) % (2 * np.pi) - np.pi)  # unwrap around the first vertex
        lo, hi = float(a.min()), float(a.max())
        widths.append(hi - lo)
        i0 = int(math.floor((lo + np.pi) / (2 * np.pi) * nb))
        i1 = int(math.ceil((hi + np.pi) / (2 * np.pi) * nb))
        idx = np.arange(i0, max(i1, i0 + 1)) % nb
        cover[idx] = True
    widths = np.array(widths)
    w_pos = widths[widths > 1e-6]
    step = float(np.degrees(np.median(w_pos))) if len(w_pos) else 0.0
    return float(cover.sum() * bin_deg), step


# ------------------------------------------------------------------------------------------------------------ holes
def find_cylinders(mesh, rule=HOLE_RULE):
    """All cylinder patches (concave and convex) under HOLE_RULE. Each: dict with axis dir, point (the centre of the
    patch's axial extent), radius, residual, concave flag, axial extent, span, facet step, face ids."""
    lab = mesh.components(rule["smooth_deg"], strict=True)
    order = np.argsort(lab, kind="stable")
    ls = lab[order]
    starts = np.r_[0, np.nonzero(np.diff(ls))[0] + 1]
    ends = np.r_[starts[1:], len(ls)]
    out = []
    for s, e in zip(starts, ends):
        fi = order[s:e]
        if len(fi) < rule["min_facets"]:
            continue
        w = mesh.A[fi]
        if w.sum() <= 0:
            continue
        Nn = mesh.N[fi]
        vid = np.unique(mesh.F[fi])
        P = mesh.V[vid]
        cyl = fit_cylinder(P, Nn, w)
        if cyl is None:
            continue
        ev = cyl["ev"]
        if ev[0] > rule["ev0_max"] or ev[1] < rule["ev1_min"]:
            continue
        r, res, d, centre = cyl["radius"], cyl["res"], cyl["dir"], cyl["centre"]
        if res.max() > max(rule["res_abs_mm"], rule["res_rel"] * r):
            continue
        cen_f = mesh.C[fi]
        rel = cen_f - centre
        radial = rel - np.outer(rel @ d, d)
        concave = float(np.sum(w * np.einsum("ij,ij->i", Nn, radial))) < 0
        t = (P - centre) @ d
        t0, t1 = float(t.min()), float(t.max())
        point = centre + d * 0.5 * (t0 + t1)
        loc = {v: i for i, v in enumerate(vid.tolist())}
        faces_local = [[loc[v] for v in tri] for tri in mesh.F[fi].tolist()]
        span, step = angular_span(P, point, d, faces_local, rule["span_bin_deg"])
        out.append({"concave": bool(concave), "radius": float(r), "dir": d, "point": point,
                    "t_range": (t0 - 0.5 * (t0 + t1), t1 - 0.5 * (t0 + t1)), "length": t1 - t0,
                    "max_residual": float(res.max()), "span_deg": span, "facet_step_deg": step,
                    "faces": np.sort(fi), "area": float(w.sum())})
    return out


def axis_line_distance(p, d, q):
    """distance of point q from the line through p along unit d"""
    v = q - p
    return float(np.linalg.norm(v - (v @ d) * d))


def group_holes(cyls, rule=HOLE_RULE):
    """indices of the concave cylinders grouped into holes (coaxial and touching along the axis)"""
    idx = [i for i, c in enumerate(cyls) if c["concave"]]
    a, b = [], []
    for ii, i in enumerate(idx):
        ci = cyls[i]
        for j in idx[ii + 1:]:
            cj = cyls[j]
            ang = math.degrees(math.acos(min(1.0, abs(float(ci["dir"] @ cj["dir"])))))
            if ang > rule["coax_deg"]:
                continue
            if max(axis_line_distance(ci["point"], ci["dir"], cj["point"]),
                   axis_line_distance(cj["point"], cj["dir"], ci["point"])) > rule["coax_mm"]:
                continue
            d = ci["dir"]
            ti = (ci["point"] @ d) + np.array(ci["t_range"]) * (1 if ci["dir"] @ d > 0 else -1)
            tj = (cj["point"] @ d) + np.array(cj["t_range"]) * (1 if cj["dir"] @ d > 0 else -1)
            gap = max(ti.min(), tj.min()) - min(ti.max(), tj.max())
            if gap <= rule["touch_mm"]:
                a.append(ii)
                b.append(idx.index(j))
    lab = connected_labels(len(idx), a, b) if idx else np.zeros(0, np.int64)
    groups = {}
    for k, g in enumerate(lab.tolist()):
        groups.setdefault(g, []).append(idx[k])
    return [groups[g] for g in sorted(groups)]


# ----------------------------------------------------------------------------------------------------------- planes
def coplanar_clusters(mesh, coplanar_deg):
    lab = mesh.components(coplanar_deg)
    nc = int(lab.max()) + 1 if len(lab) else 0
    area = np.bincount(lab, weights=mesh.A, minlength=nc)
    nrm = np.zeros((nc, 3))
    for k in range(3):
        nrm[:, k] = np.bincount(lab, weights=mesh.A * mesh.N[:, k], minlength=nc)
    nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-300)
    off = np.bincount(lab, weights=mesh.A * np.einsum("ij,ij->i", nrm[lab], mesh.C), minlength=nc)
    off /= np.maximum(area, 1e-300)
    return lab, nc, area, nrm, off


def fit_curved(mesh, faces):
    """Best of cylinder / cone / sphere fitted to a set of faces (by max vertex residual)."""
    vid = np.unique(mesh.F[faces])
    P = mesh.V[vid]
    w = mesh.A[faces]
    Nn = mesh.N[faces]
    if len(vid) < 6 or w.sum() <= 0:
        return None
    best = None
    cyl = fit_cylinder(P, Nn, w)
    if cyl is not None:
        best = {"kind": "cylinder", "dir": cyl["dir"], "centre": cyl["centre"], "radius": cyl["radius"],
                "max_res": float(cyl["res"].max())}
    sph = fit_sphere(P)
    if sph is not None and (best is None or sph["res"].max() < best["max_res"]):
        best = {"kind": "sphere", "centre": sph["centre"], "radius": sph["radius"], "max_res": float(sph["res"].max())}
    cone = fit_cone(P, Nn, w, mesh.C[faces])
    if cone is not None and (best is None or cone["res"].max() < best["max_res"]):
        best = {"kind": "cone", "dir": cone["dir"], "apex": cone["apex"], "half_angle_deg": cone["half_angle_deg"],
                "max_res": float(cone["res"].max())}
    return best


def surface_distance(prim, P):
    """unsigned distance of points from a fitted primitive's surface"""
    if prim["kind"] == "sphere":
        return np.abs(np.linalg.norm(P - prim["centre"], axis=1) - prim["radius"])
    if prim["kind"] == "cylinder":
        d = prim["dir"]
        v = P - prim["centre"]
        rho = np.linalg.norm(v - np.outer(v @ d, d), axis=1)
        return np.abs(rho - prim["radius"])
    d = prim["dir"]
    v = P - prim["apex"]
    t = v @ d
    rho = np.linalg.norm(v - np.outer(t, d), axis=1)
    h = math.radians(prim["half_angle_deg"])
    if np.mean(t) < 0:
        t = -t
    return np.abs(rho * math.cos(h) - t * math.sin(h))


def angular_width(prim, P):
    """angular width (deg) of a point set about the primitive's axis (cylinder, cone) or seen from its centre
    (sphere: largest angle between two points' directions)"""
    if prim["kind"] == "sphere":
        u = P - prim["centre"]
        u /= np.maximum(np.linalg.norm(u, axis=1, keepdims=True), 1e-300)
        g = np.clip(u @ u.T, -1, 1)
        return float(np.degrees(np.arccos(g.min())))
    d = prim["dir"]
    o = prim["centre"] if prim["kind"] == "cylinder" else prim["apex"]
    e1, e2 = perp_basis(d)
    q = P - o
    ang = np.sort(np.arctan2(q @ e2, q @ e1))
    gaps = np.diff(np.r_[ang, ang[0] + 2 * np.pi])
    return float(np.degrees(2 * np.pi - gaps.max()))


def _cluster_faces(lab, nc):
    order = np.argsort(lab, kind="stable")
    ls = lab[order]
    if not len(ls):
        return []
    starts = np.r_[0, np.nonzero(np.diff(ls))[0] + 1]
    ends = np.r_[starts[1:], len(ls)]
    return [np.sort(order[s:e]) for s, e in zip(starts, ends)]


def fitted_surfaces(mesh, rule=PLANE_RULE):
    """The design's fitted curved surfaces: every smooth component (faces joined across edges with a dihedral angle
    < rule["surface_smooth_deg"], at least 6 faces) whose normals are curved (middle eigenvalue of the normal
    covariance >= 0.02) and whose vertices fit a cylinder, a sphere or a cone with max residual
    <= rule["surface_fit_res_mm"]. Returns (list of primitives, component label of every face)."""
    comp = mesh.components(rule["surface_smooth_deg"], strict=True)
    out = []
    for k, fi in enumerate(_cluster_faces(comp, int(comp.max()) + 1 if len(comp) else 0)):
        if len(fi) < 6:
            continue
        vid = np.unique(mesh.F[fi])
        P = mesh.V[vid]
        w = mesh.A[fi]
        Nn = mesh.N[fi]
        if len(vid) < 6 or w.sum() <= 0:
            continue
        C = (Nn * w[:, None]).T @ Nn / w.sum()
        ev = np.linalg.eigvalsh(C)
        if ev[1] < 0.02:
            continue
        cands = []
        if ev[0] <= 2e-3:
            cyl = fit_cylinder(P, Nn, w)
            if cyl is not None:
                cands.append({"kind": "cylinder", "dir": cyl["dir"], "centre": cyl["centre"], "radius": cyl["radius"],
                              "max_res": float(cyl["res"].max())})
        else:
            sph = fit_sphere(P)
            if sph is not None:
                cands.append({"kind": "sphere", "centre": sph["centre"], "radius": sph["radius"],
                              "max_res": float(sph["res"].max())})
        cone = fit_cone(P, Nn, w, mesh.C[fi])
        if cone is not None:
            cands.append({"kind": "cone", "dir": cone["dir"], "apex": cone["apex"],
                          "half_angle_deg": cone["half_angle_deg"], "max_res": float(cone["res"].max())})
        cands = [c for c in cands if c["max_res"] <= rule["surface_fit_res_mm"]]
        if not cands:
            continue
        best = min(cands, key=lambda c: (c["max_res"], c["kind"]))
        best["component"] = k
        best["faces"] = fi
        best["area"] = float(w.sum())
        out.append(best)
    return out, comp


def strip_of(mesh, faces, comp, surfaces, rule=PLANE_RULE):
    """Rule (b): is this face set part of a fitted curved surface? Either
    (i)  its triangles belong to a fitted smooth component (fitted_surfaces), or
    (ii) it is a narrow strip (angular width <= rule["smooth_deg"], normal radial within half its width + 1 deg)
         lying on a surface of the same kind as a fitted one: all vertices at one distance (spread
         <= on_surface_tol_mm) from a fitted cylinder's axis, at least twice as long along the axis as the chord it
         subtends; or on a fitted sphere (same centre and radius) or a fitted cone, within on_surface_tol_mm."""
    k = int(comp[faces[0]])
    for s in surfaces:
        if s["component"] == k:
            return {"test": "member", "kind": s["kind"], "component": k}
    P = mesh.V[np.unique(mesh.F[faces])]
    w = mesh.A[faces]
    n = (mesh.N[faces] * w[:, None]).sum(0)
    n /= max(np.linalg.norm(n), 1e-300)
    cen = (mesh.C[faces] * w[:, None]).sum(0) / w.sum()
    tol = rule["on_surface_tol_mm"]
    for s in surfaces:
        if s["kind"] == "cylinder":
            d = s["dir"]
            if abs(float(n @ d)) > math.sin(math.radians(1.0)):
                continue
            v = P - s["centre"]
            rho = np.linalg.norm(v - np.outer(v @ d, d), axis=1)
            if np.ptp(rho) > tol:
                continue
            # elongated along the axis: axial extent >= 2 x the chord the strip subtends
            prim = {"kind": "cylinder", "dir": d, "centre": s["centre"], "radius": float(rho.mean())}
            chord = 2.0 * prim["radius"] * math.sin(0.5 * math.radians(angular_width(prim, P)))
            if float(np.ptp(v @ d)) < 2.0 * chord:
                continue
            vc = cen - prim["centre"]
            rad = vc - (vc @ d) * d
        elif s["kind"] == "sphere":
            if np.abs(np.linalg.norm(P - s["centre"], axis=1) - s["radius"]).max() > tol:
                continue
            prim = s
            rad = cen - s["centre"]
        else:
            if surface_distance(s, P).max() > tol:
                continue
            prim = s
            d = s["dir"]
            vc = cen - s["apex"]
            t = float(vc @ d)
            radial = vc - t * d
            radial /= max(np.linalg.norm(radial), 1e-300)
            h = math.radians(s["half_angle_deg"])
            rad = math.cos(h) * radial - math.sin(h) * d * (1.0 if t >= 0 else -1.0)
        wdt = angular_width(prim, P)
        if wdt > rule["smooth_deg"]:
            continue
        rad = rad / max(np.linalg.norm(rad), 1e-300)
        dev = math.degrees(math.acos(min(1.0, abs(float(n @ rad)))))
        if dev <= 0.5 * wdt + 1.0:
            return {"test": "on_surface", "kind": prim["kind"], "component": int(s["component"]),
                    "radius_mm": float(prim.get("radius", 0.0)), "width_deg": wdt, "normal_dev_deg": dev}
    return None


def find_planes(mesh, rule=PLANE_RULE):
    """Flat faces under PLANE_RULE. Returns (planes, diag): planes = list of dicts (normal, offset, area, faces,
    centroid, flatness, smooth share, ...) in cluster order; diag = cluster labels and, for the QC, every cluster at or
    above the area floor that is not a plane, with the reason."""
    lab, nc, area, nrm, off = coplanar_clusters(mesh, rule["coplanar_deg"])
    a, b = lab[mesh.fa], lab[mesh.fb]
    diff = a != b
    a, b, L = a[diff], b[diff], mesh.elen[diff]
    cang = angle_deg(nrm[a], nrm[b])
    smooth = cang <= rule["smooth_deg"]
    blen = np.bincount(a, weights=L, minlength=nc) + np.bincount(b, weights=L, minlength=nc)
    slen = (np.bincount(a[smooth], weights=L[smooth], minlength=nc)
            + np.bincount(b[smooth], weights=L[smooth], minlength=nc))
    sn_area = (np.bincount(a[smooth], weights=area[b[smooth]], minlength=nc)
               + np.bincount(b[smooth], weights=area[a[smooth]], minlength=nc))
    sn_cnt = np.bincount(a[smooth], minlength=nc) + np.bincount(b[smooth], minlength=nc)
    sn_mean = np.where(sn_cnt > 0, sn_area / np.maximum(sn_cnt, 1), 0.0)
    sfrac = np.where(blen > 0, slen / np.maximum(blen, 1e-300), 0.0)
    rule_a = (sfrac <= rule["smooth_share_max"]) | (area >= rule["big_ratio"] * np.maximum(sn_mean, 1e-12))
    faces_of = _cluster_faces(lab, nc)
    surfaces, comp = fitted_surfaces(mesh, rule)
    planes, excluded = [], []
    for c in range(nc):
        if area[c] < rule["area_min_mm2"]:
            continue
        fi = faces_of[c]
        P = mesh.V[np.unique(mesh.F[fi])]
        flat = float(np.abs(P @ nrm[c] - off[c]).max())
        info = {"cluster": c, "area": float(area[c]), "normal": nrm[c], "offset": float(off[c]), "faces": fi,
                "flatness": flat, "smooth_share": float(sfrac[c]), "smooth_nb_mean_area": float(sn_mean[c]),
                "centroid": (mesh.A[fi][:, None] * mesh.C[fi]).sum(0) / mesh.A[fi].sum()}
        st = strip_of(mesh, fi, comp, surfaces, rule)
        info["strip_test"] = st
        if not rule_a[c]:
            info["why_excluded"] = "a"
        elif flat > rule["flatness_max_mm"]:
            info["why_excluded"] = "c"
        elif st is not None:
            info["why_excluded"] = "b"
        else:
            planes.append(info)
            continue
        excluded.append(info)
    diag = {"clusters": nc, "rule_a_ge_floor": int(np.sum(rule_a & (area >= rule["area_min_mm2"]))),
            "excluded_ge_floor": excluded, "cluster_labels": lab, "surfaces": surfaces}
    return planes, diag


# --------------------------------------------------------------------------------------------- spheres and profiles
def find_spheres(mesh, smooth_deg=20.0, min_facets=20, res_abs_mm=0.02):
    """Smooth-connected face sets (dihedral < smooth_deg) whose normals span all three directions (normal covariance
    eigenvalues >= 2e-3 and >= 0.02) and whose vertices fit a sphere with max residual <= res_abs_mm. Each: centre,
    radius, residual, convex flag, faces."""
    lab = mesh.components(smooth_deg, strict=True)
    out = []
    for fi in _cluster_faces(lab, int(lab.max()) + 1 if len(lab) else 0):
        if len(fi) < min_facets:
            continue
        w = mesh.A[fi]
        Nn = mesh.N[fi]
        ev = np.linalg.eigvalsh((Nn * w[:, None]).T @ Nn / w.sum())
        if ev[0] < 2e-3 or ev[1] < 0.02:
            continue
        P = mesh.V[np.unique(mesh.F[fi])]
        sph = fit_sphere(P)
        if sph is None or sph["res"].max() > res_abs_mm:
            continue
        rel = mesh.C[fi] - sph["centre"]
        convex = float(np.sum(w * np.einsum("ij,ij->i", Nn, rel))) > 0
        out.append({"centre": sph["centre"], "radius": float(sph["radius"]), "max_residual": float(sph["res"].max()),
                    "convex": bool(convex), "faces": np.sort(fi), "area": float(w.sum())})
    return out


def edge_samples(mesh, step_mm):
    """deterministic points along every mesh edge, at most step_mm apart (the outer radius of a faceted surface at a
    given axial position is reached on an edge, so these points give the exact radial profile)"""
    e = np.concatenate([mesh.F[:, [0, 1]], mesh.F[:, [1, 2]], mesh.F[:, [2, 0]]])
    e = np.unique(np.sort(e, axis=1), axis=0)
    a, b = mesh.V[e[:, 0]], mesh.V[e[:, 1]]
    n = np.maximum(1, np.ceil(np.linalg.norm(b - a, axis=1) / step_mm).astype(np.int64))
    idx = np.repeat(np.arange(len(e)), n + 1)
    k = np.concatenate([np.arange(m + 1) for m in n.tolist()])
    t = (k / np.repeat(n, n + 1))[:, None]
    return a[idx] * (1 - t) + b[idx] * t


def radial_profile(mesh, point, d, bin_mm=0.01):
    """outer radius r(t) about an axis: max distance from the axis of edge points in each axial bin of width bin_mm.
    Returns the bin centres t (relative to point, along d) and r."""
    d = np.asarray(d, float) / np.linalg.norm(d)
    Q = edge_samples(mesh, 0.5 * bin_mm) - point
    t = Q @ d
    rho = np.linalg.norm(Q - np.outer(t, d), axis=1)
    lo = math.floor(t.min() / bin_mm) * bin_mm
    nb = int(math.ceil((t.max() - lo) / bin_mm)) + 1
    k = np.clip(((t - lo) / bin_mm).astype(np.int64), 0, nb - 1)
    r = np.full(nb, -1.0)
    np.maximum.at(r, k, rho)
    tc = lo + (np.arange(nb) + 0.5) * bin_mm
    ok = r >= 0
    return tc[ok], r[ok]


def drum_band(t, r, delta_mm=0.01, middle_share=0.4):
    """The drum of a winch-like profile: drum radius = median outer radius over the middle middle_share of the axial
    range; the band = the contiguous run around the middle where r <= drum radius + delta_mm, its ends interpolated
    linearly at r = drum radius + delta_mm. Returns (drum radius, t_lo, t_hi)."""
    tmid = 0.5 * (t.min() + t.max())
    half = 0.5 * middle_share * (t.max() - t.min())
    core = np.abs(t - tmid) <= half
    rd = float(np.median(r[core]))
    lim = rd + delta_mm
    i0 = int(np.argmin(np.abs(t - tmid)))

    def walk(step):
        i = i0
        while 0 <= i + step < len(t) and r[i + step] <= lim:
            i += step
        j = i + step
        if not (0 <= j < len(t)):
            return float(t[i])
        f = (lim - r[i]) / (r[j] - r[i]) if r[j] != r[i] else 0.0
        return float(t[i] + f * (t[j] - t[i]))

    return rd, walk(-1), walk(1)
