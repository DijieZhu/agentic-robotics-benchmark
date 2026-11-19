"""Closed solids made from the maker's triangles, triangle for triangle: the geometry of the reference submission.

The rule (the approved design, section 9: "closed pieces as they are, every other orientable patch thickened by
+-0.15 mm along its vertex normals and closed along its boundary"; no simplification):
  1. weld: vertices are merged on a grid of WELD_MM (1e-3 mm); each merged vertex keeps the coordinates of the first
     vertex of its group in input order, so no vertex is moved; a triangle with two corners in one group collapses
     and makes no face (it has no area);
  2. patches: the remaining triangles are split into orientable patches at every edge that is not shared by exactly
     two of them; a patch grows breadth first from its lowest-numbered triangle, whose winding as given sets the
     patch's orientation, and every other triangle of the patch is re-wound to agree with it;
  3. a patch that is closed (every edge shared by two of its triangles) and consistently oriented is one of the
     maker's own closed bodies: it is kept as it is, turned inside out only when its enclosed volume is negative;
  4. every other patch becomes a thin skin: the patch moved along its area-weighted vertex normals by the two offsets
     (the first copy keeps the patch's winding, the second is reversed), closed along the patch boundary by side walls
     of two triangles per boundary edge; the reference uses REFERENCE_OFFSETS_MM = (+0.15, -0.15), a 0.3 mm skin
     centred on the maker's surface (the one authored item of the reference submission that the design lists);
  5. nothing is simplified, smoothed, re-meshed or added.
Every patch carries the indices of the input triangles it is made of (rows of the F given, before the weld; with the
task's mesh reader these are the triangle numbers inside the maker's file, the numbering of the graded-sample table),
so a selection by triangles keeps exactly the patches that carry them.

build_solid() makes one FreeCAD solid of one closed, consistently oriented triangle mesh: one planar face per triangle
(Part.Shape.makeShapeFromMesh with a FACE_TOLERANCE_MM tolerance), Part.Shell of those faces, Part.Solid, reversed
when its volume is negative; no faces are merged and no vertex is moved.

Module level: the standard library and numpy only (FreeCAD's own interpreter in the task image, Python 3.11 with
numpy 1.26, has nothing else). FreeCAD and Part are imported inside build_solid() only.
"""
from collections import deque

import numpy as np

WELD_MM = 1e-3
REFERENCE_OFFSETS_MM = (0.15, -0.15)
FACE_TOLERANCE_MM = 0.001


# ------------------------------------------------------------------------------------------------ mesh helpers
def weld(V, F, tol=WELD_MM):
    """(Vw, Fw, kept): the vertices merged on a tol grid (the first vertex of each group kept, unmoved); Fw = the
    triangles that keep three distinct corners, in input order, re-indexed into Vw; kept = their rows in F."""
    V = np.asarray(V, np.float64)
    F = np.asarray(F, np.int64).reshape(-1, 3)
    q = np.round(V / float(tol)).astype(np.int64)
    _u, first, inv = np.unique(q, axis=0, return_index=True, return_inverse=True)
    inv = np.asarray(inv).reshape(-1)
    F2 = inv[F]
    keep = (F2[:, 0] != F2[:, 1]) & (F2[:, 1] != F2[:, 2]) & (F2[:, 0] != F2[:, 2])
    return V[first], F2[keep], np.nonzero(keep)[0].astype(np.int64)


def orient_patches(F):
    """(Fo, label, n): the triangles split into orientable patches across the edges shared by exactly two triangles;
    each patch grows breadth first from its lowest-numbered triangle and its other triangles are re-wound to agree."""
    T = [list(t) for t in np.asarray(F, np.int64).reshape(-1, 3).tolist()]
    edges = {}
    for t, (a, b, c) in enumerate(T):
        for u, v in ((a, b), (b, c), (c, a)):
            key = (u, v) if u < v else (v, u)
            lst = edges.get(key)
            if lst is None:
                edges[key] = [t]
            else:
                lst.append(t)
    lab = [-1] * len(T)
    n = 0
    for s in range(len(T)):
        if lab[s] >= 0:
            continue
        lab[s] = n
        dq = deque([s])
        while dq:
            t = dq.popleft()
            a, b, c = T[t]
            for u, v in ((a, b), (b, c), (c, a)):
                ts = edges[(u, v) if u < v else (v, u)]
                if len(ts) != 2:
                    continue
                o = ts[0] if ts[1] == t else ts[1]
                if lab[o] >= 0:
                    continue
                oa, ob, oc = T[o]
                # o must traverse the shared edge as (v, u)
                if (oa, ob) == (u, v) or (ob, oc) == (u, v) or (oc, oa) == (u, v):
                    T[o] = [oa, oc, ob]
                lab[o] = n
                dq.append(o)
        n += 1
    return np.asarray(T, np.int64).reshape(-1, 3), np.asarray(lab, np.int64), n


def closed_and_oriented(Fp):
    """True when every edge of the triangles is shared by exactly two of them and each directed edge occurs once."""
    Fp = np.asarray(Fp, np.int64).reshape(-1, 3)
    if len(Fp) == 0:
        return False
    d = np.vstack([Fp[:, [0, 1]], Fp[:, [1, 2]], Fp[:, [2, 0]]])
    und = np.sort(d, axis=1)
    _u, cnt = np.unique(und, axis=0, return_counts=True)
    key = d[:, 0] * (int(d.max()) + 1) + d[:, 1]
    _k, kc = np.unique(key, return_counts=True)
    return bool((cnt == 2).all() and (kc == 1).all())


def compact(V, Fp):
    """(P, F): the vertices the triangles use (in index order) and the triangles re-indexed into them."""
    Fp = np.asarray(Fp, np.int64).reshape(-1, 3)
    used = np.unique(Fp)
    remap = -np.ones(len(V), np.int64)
    remap[used] = np.arange(len(used))
    return np.asarray(V, np.float64)[used], remap[Fp]


def signed_volume(V, F):
    """Enclosed volume of an oriented triangle mesh in double precision (about the vertex centroid)."""
    V = np.asarray(V, np.float64)
    F = np.asarray(F, np.int64).reshape(-1, 3)
    if len(F) == 0:
        return 0.0
    o = V.mean(axis=0)
    a, b, c = V[F[:, 0]] - o, V[F[:, 1]] - o, V[F[:, 2]] - o
    return float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0)


def triangle_areas(V, F):
    V = np.asarray(V, np.float64)
    F = np.asarray(F, np.int64).reshape(-1, 3)
    return 0.5 * np.linalg.norm(np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]]), axis=1)


def vertex_normals(P, F):
    """(unit area-weighted vertex normals, number of vertices whose normal vanished and was set to +z)."""
    fn = np.cross(P[F[:, 1]] - P[F[:, 0]], P[F[:, 2]] - P[F[:, 0]])
    vn = np.zeros_like(P)
    for k in range(3):
        np.add.at(vn, F[:, k], fn)
    ln = np.linalg.norm(vn, axis=1)
    bad = ln < 1e-12
    vn[bad] = [0.0, 0.0, 1.0]
    ln[bad] = 1.0
    return vn / ln[:, None], int(bad.sum())


def closed_body(V, Fp):
    """(P, F, turned): a closed, consistently oriented patch as it is; turned inside out when its volume is negative."""
    P, F = compact(V, Fp)
    turned = signed_volume(P, F) < 0.0
    if turned:
        F = F[:, ::-1].copy()
    return P, F, turned


def skin_mesh(V, Fp, offsets=REFERENCE_OFFSETS_MM):
    """(Vs, Fs, vanished_normals): the patch as a closed thin skin between the two offsets along its vertex normals
    (offsets[0] > offsets[1]; the copy at offsets[0] keeps the patch's winding, the other is reversed), closed by two
    wall triangles per boundary edge. Vertices: the patch's vertices moved by offsets[0], then by offsets[1]."""
    hi, lo = float(offsets[0]), float(offsets[1])
    if not hi > lo:
        raise ValueError("skin offsets must be decreasing (outer, inner), got %r" % (tuple(offsets),))
    P, F = compact(V, Fp)
    vn, vanished = vertex_normals(P, F)
    n = len(P)
    Vs = np.vstack([P + hi * vn, P + lo * vn])
    top = F
    bot = F[:, ::-1] + n
    d = np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    und = np.sort(d, axis=1)
    _u, inv, cnt = np.unique(und, axis=0, return_inverse=True, return_counts=True)
    b = d[cnt[np.asarray(inv).reshape(-1)] == 1]          # boundary edges u -> v of the first copy, in order
    side = np.empty((2 * len(b), 3), np.int64)
    side[0::2] = np.stack([b[:, 1], b[:, 0], b[:, 0] + n], axis=1)       # wall (v, u, u')
    side[1::2] = np.stack([b[:, 1], b[:, 0] + n, b[:, 1] + n], axis=1)   # wall (v, u', v')
    return Vs, np.vstack([top, bot, side]), vanished


# ------------------------------------------------------------------------------------------------ patches
def _selector(select):
    if select is None or (isinstance(select, str) and select == "all"):
        return lambda rec: True
    if callable(select):
        return select
    if isinstance(select, dict) and "patch_indices" in select:
        keep = set(int(i) for i in select["patch_indices"])
        return lambda rec: rec["patch"] in keep
    if isinstance(select, dict) and "triangle_indices" in select:
        keep = np.unique(np.asarray(list(select["triangle_indices"]), np.int64))
        return lambda rec: bool(np.isin(rec["source_triangles"], keep).any())
    raise ValueError("select must be None, 'all', a callable, {'patch_indices': [...]} or {'triangle_indices': [...]}")


def make_patches(V, F, offsets=REFERENCE_OFFSETS_MM, select=None, weld_mm=WELD_MM):
    """The closed meshes of one maker file's triangles (V in mm, F rows = the file's triangles).

    offsets: the skin's two offsets along the vertex normals (outer, inner), mm; the reference: (+0.15, -0.15).
    select:  which patches to keep: None or "all" (the default: every patch), {"patch_indices": [...]},
             {"triangle_indices": [...]} (a patch is kept when one of its source triangles is listed), or a callable
             taking the patch record {patch, kind, source_triangles} and returning True to keep it.
    Returns (patches, summary). patches: [{patch (index, in order of the patch's lowest welded triangle), kind
    ("closed" | "skin"), source_triangles (sorted rows of F the patch is made of), V (n, 3) mm, F (m, 3), turned (closed
    patch turned inside out), vanished_normals (skin)}], only the selected ones. summary: counts for the whole file."""
    V = np.asarray(V, np.float64)
    F = np.asarray(F, np.int64).reshape(-1, 3)
    Vw, Fw, kept = weld(V, F, weld_mm)
    Fo, lab, n = orient_patches(Fw)
    order = np.argsort(lab, kind="stable")
    bounds = np.searchsorted(lab[order], np.arange(n + 1))
    keep = _selector(select)
    out = []
    n_closed = n_skin = 0
    for p in range(n):
        rows = order[bounds[p]:bounds[p + 1]]
        Fp = Fo[rows]
        kind = "closed" if closed_and_oriented(Fp) else "skin"
        if kind == "closed":
            n_closed += 1
        else:
            n_skin += 1
        rec = {"patch": p, "kind": kind, "source_triangles": kept[rows]}
        if not keep(rec):
            continue
        if kind == "closed":
            Pv, Pf, turned = closed_body(Vw, Fp)
            rec.update(V=Pv, F=Pf, turned=bool(turned), vanished_normals=0)
        else:
            Pv, Pf, vanished = skin_mesh(Vw, Fp, offsets)
            rec.update(V=Pv, F=Pf, turned=False, vanished_normals=int(vanished))
        out.append(rec)
    summary = {"input_triangles": int(len(F)), "input_vertices": int(len(V)), "welded_vertices": int(len(Vw)),
               "collapsed_triangles": int(len(F) - len(kept)), "welded_triangles": int(len(kept)),
               "patches": int(n), "closed_patches": int(n_closed), "skin_patches": int(n_skin),
               "selected_patches": len(out), "selected_triangles_out": int(sum(len(r["F"]) for r in out)),
               "weld_mm": float(weld_mm), "offsets_mm": [float(offsets[0]), float(offsets[1])]}
    return out, summary


# ------------------------------------------------------------------------------------------------ FreeCAD
def build_solid(V, F, tolerance_mm=FACE_TOLERANCE_MM):
    """(solid, record): one FreeCAD solid of one closed, consistently oriented triangle mesh in mm.

    Part.Shape.makeShapeFromMesh((points, triangles), tolerance_mm) gives one planar face per triangle (corners at the
    mesh's vertices, double precision), Part.Shell of those faces, Part.Solid, reversed when its volume is negative.
    record: triangles, faces, valid (FreeCAD isValid, BRepCheck), closed (isClosed), volume_mm3 (OCC), mesh_volume_mm3
    (the mesh's own, double precision), volume_diff_mm3, reversed."""
    import FreeCAD as App
    import Part
    pts = [App.Vector(x, y, z) for x, y, z in np.asarray(V, np.float64).tolist()]
    tris = [(a, b, c) for a, b, c in np.asarray(F, np.int64).reshape(-1, 3).tolist()]
    sh = Part.Shape()
    sh.makeShapeFromMesh((pts, tris), float(tolerance_mm))
    shell = Part.Shell(sh.Faces)
    solid = Part.Solid(shell)
    reversed_ = False
    if solid.Volume < 0:
        solid.reverse()
        reversed_ = True
    mv = signed_volume(V, F)
    rec = {"triangles": len(tris), "faces": len(solid.Faces), "shell_closed": bool(shell.isClosed()),
           "closed": bool(solid.isClosed()), "valid": bool(solid.isValid()), "volume_mm3": float(solid.Volume),
           "mesh_volume_mm3": float(mv), "reversed": reversed_}
    rec["volume_diff_mm3"] = rec["volume_mm3"] - rec["mesh_volume_mm3"]
    return solid, rec
