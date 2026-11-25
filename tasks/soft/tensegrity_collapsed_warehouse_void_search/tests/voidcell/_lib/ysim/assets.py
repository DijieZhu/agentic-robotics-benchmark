"""Build the derived strut assets from the authors' whole-strut mesh (dev tool, run once).

Input: the authors' mesh ``Will_strut_scaled_centered.ply`` (metres, 1 118 331 triangles, sha256 in
``paths.AUTHORS_PLY_SHA256``). Nothing is modelled, redrawn or completed: every output is the authors' triangles
(regrouped by part for materials), a decimated copy of them, or a measurement / fit of them.

Outputs (asset dir, see ``paths.assets_dir``):
  full/<group>.stl   the authors' triangles of each material group, strut frame, metres (full resolution)
  dec/<group>.stl    decimated copy (fast_simplification, settings recorded) for fast rendering
  col/elec_hull.stl  convex hull of the two electronics-enclosure halves (collision)
  col/cap_hull_pz.stl, col/cap_hull_nz.stl  convex hulls of the two end caps (optional collision)
and ``notes/sim_base_data/strut_geometry.json`` with fits, residuals, features and file hashes.

Strut frame: origin = centre of the rod's bounding box,
z = rod axis (PCA of the rod piece; equals file z), x = file x projected. Units here: mm internally, m on output.

usage: python -I -B -c "import sys; sys.path.insert(0, '<yale dir>'); from ysim import assets; assets.build()"
"""
import hashlib
import json
import os
import sys

import numpy as np

from . import paths

GROUPS = ("sock", "endcap", "housing", "rod", "other")
MAX_STL_FACES = 190000


def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def load_pieces(ply):
    import trimesh
    m = trimesh.load(ply, process=False, force="mesh")
    V = np.asarray(m.vertices, float) * 1000.0
    F = np.asarray(m.faces, np.int64)
    # edge-connected components as disjoint face sets (trimesh's split() returns 124 faces twice on this mesh)
    comps = trimesh.graph.connected_components(m.face_adjacency, nodes=np.arange(len(F)), min_len=1)
    parts = []
    for cidx in sorted(comps, key=lambda c: int(np.min(c))):
        Fc = F[np.sort(cidx)]
        uv, inv = np.unique(Fc.reshape(-1), return_inverse=True)
        parts.append(trimesh.Trimesh(V[uv], inv.reshape(-1, 3), process=False))
    assert sum(len(p.faces) for p in parts) == len(F)
    rod = [p for p in parts if abs(p.extents.max() - 304.8) < 0.5 and np.sort(p.extents)[1] < 4.0]
    assert len(rod) == 1, "rod piece not unique"
    rod = rod[0]
    c_rod = rod.bounds.mean(axis=0)
    w, E = np.linalg.eigh(np.cov((rod.vertices - c_rod).T))
    axis = E[:, 2] if E[2, 2] > 0 else -E[:, 2]
    x = np.array([1.0, 0, 0]) - axis * axis[0]
    x /= np.linalg.norm(x)
    R = np.stack([x, np.cross(axis, x), axis])
    order = np.argsort([-float(np.prod(p.extents)) for p in parts], kind="stable")
    out = []
    for i in order:
        p = parts[i]
        Vp = (np.asarray(p.vertices) - c_rod) @ R.T
        out.append((Vp, np.asarray(p.faces, np.int64)))
    return out, {"n_faces": int(len(F)), "n_vertices": int(len(V)), "n_pieces": len(parts),
                 "rod_centre_file_mm": c_rod.tolist(), "rod_axis_file": axis.tolist()}


def classify(pieces):
    """Assign each piece to a part label by its measured signature (asserting the expected counts)."""
    lab = []
    for V, F in pieces:
        ext = V.max(0) - V.min(0)
        ctr = 0.5 * (V.max(0) + V.min(0))
        nf = len(F)
        az = abs(ctr[2])
        e = np.sort(ext)[::-1]
        if abs(e[0] - 304.8) < 0.5 and e[1] < 4:
            lab.append("rod")
        elif abs(e[0] - 35.0) < 0.3 and abs(e[1] - 35.0) < 0.3 and abs(e[2] - 17.5) < 0.3 and abs(az - 154.15) < 0.5:
            lab.append("sock")
        elif nf == 6340 and abs(az - 145.9) < 0.5:
            lab.append("endcap")
        elif nf == 2070 and abs(ext[2] - 55.0) < 0.2:
            lab.append("encA")
        elif nf == 1422 and abs(az - 120.9) < 0.5:
            lab.append("encB")
        elif abs(ext[2] - 143.0) < 0.2 and nf in (1706, 3428):
            lab.append("elec")
        elif nf == 6282 and abs(az - 124.15) < 0.5:
            lab.append("winch")
        elif nf == 6396 and abs(az - 124.79) < 0.5:
            lab.append("guide")
        elif nf == 43688 and abs(ext[2] - 44.0) < 0.2:
            lab.append("arduino")
        elif nf == 788 and abs(ext[2] - 140.0) < 0.2:
            lab.append("motherboard")
        else:
            lab.append("other")
    cnt = {k: lab.count(k) for k in set(lab)}
    for k, n in (("rod", 1), ("sock", 2), ("endcap", 2), ("encA", 2), ("encB", 2), ("elec", 2), ("winch", 2),
                 ("guide", 1), ("arduino", 1), ("motherboard", 1)):
        assert cnt.get(k) == n, "expected %d %s pieces, found %s" % (n, k, cnt.get(k))
    return lab


GROUP_OF = {"sock": "sock", "endcap": "endcap", "encA": "housing", "encB": "housing", "elec": "housing",
            "winch": "housing", "guide": "housing", "rod": "rod", "arduino": "other", "motherboard": "other",
            "other": "other"}


def fit_sphere(P):
    A = np.c_[2 * P, np.ones(len(P))]
    b = (P ** 2).sum(1)
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    c = sol[:3]
    r = float(np.sqrt(sol[3] + c @ c))
    res = np.linalg.norm(P - c, axis=1) - r
    return c, r, res


def hook_features(V, sign):
    """Centroids of the four hook tabs of one end cap (band 136.5 < s*z < 142.0 mm, radius > 7 mm)."""
    z = sign * V[:, 2]
    r = np.hypot(V[:, 0], V[:, 1])
    th = np.arctan2(V[:, 1], V[:, 0])
    k = (z > 136.5) & (z < 142.0) & (r > 7.0)
    a = np.sort(np.degrees(th[k]))
    gaps = np.diff(np.r_[a, a[0] + 360.0])
    cut = np.argsort(gaps)[-4:]
    starts = sorted(a[(c + 1) % len(a)] for c in cut)
    hooks = []
    for s in starts:
        rel = (np.degrees(th) - s) % 360.0
        kk = k & (rel < 60.0)
        P = V[kk]
        ang = np.degrees(np.arctan2(np.sin(th[kk]).mean(), np.cos(th[kk]).mean()))
        hooks.append({"angle_deg": float(ang), "centroid_mm": P.mean(0).tolist(), "n_vertices": int(kk.sum()),
                      "r_range_mm": [float(r[kk].min()), float(r[kk].max())],
                      "z_range_mm": [float(V[kk, 2].min()), float(V[kk, 2].max())]})
    hooks.sort(key=lambda h: h["angle_deg"])
    return hooks


def two_fold_axis(A, B):
    """Rotation (Procrustes, same vertex order) mapping piece A onto piece B; returns R, max residual."""
    ca, cb = A.mean(0), B.mean(0)
    U, S, Vt = np.linalg.svd((A - ca).T @ (B - cb))
    D = np.eye(3)
    D[2, 2] = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ D @ U.T
    t = cb - ca @ R.T
    return R, t, float(np.linalg.norm(A @ R.T + t - B, axis=1).max())


def build(ply=None, out=None, other_keep=0.06):
    import trimesh
    import fast_simplification
    ply = ply or paths.AUTHORS_PLY
    out = paths.assets_dir(out)
    sha_in = _sha(ply)
    assert sha_in == paths.AUTHORS_PLY_SHA256, "unexpected input mesh %s" % sha_in
    pieces, meta = load_pieces(ply)
    lab = classify(pieces)
    for d in ("full", "dec", "col"):
        os.makedirs(os.path.join(out, d), exist_ok=True)
    geo = {"source_mesh": ply, "source_sha256": sha_in, "split": meta, "frame": (
        "strut frame: origin = rod bounding-box centre, z = rod axis (file z), x = file x; metres. Mesh +z end is "
        "mapped to the authors' even node (s0, s2, s4) of each strut (ASSUMPTION A-END, see params)"),
        "piece_labels": {k: lab.count(k) for k in sorted(set(lab))}, "files": {}}
    # ---- material groups (full + decimated) ----
    for g in GROUPS:
        local = [(V, F) for (V, F), l in zip(pieces, lab) if GROUP_OF[l] == g]
        Vs, Fs, off = [], [], 0
        for V, F in local:
            Vs.append(V)
            Fs.append(F + off)
            off += len(V)
        V = np.vstack(Vs) / 1000.0
        F = np.vstack(Fs)
        # full resolution, in chunks of whole pieces (MuJoCo's STL decoder accepts at most 200 000 faces per file)
        chunks, cur, nf = [], [], 0
        for Vp, Fp in local:
            if cur and nf + len(Fp) > MAX_STL_FACES:
                chunks.append(cur)
                cur, nf = [], 0
            cur.append((Vp, Fp))
            nf += len(Fp)
        chunks.append(cur)
        geo["files"].setdefault("full_chunks", {})[g] = []
        for ci, ch in enumerate(chunks):
            Vc, Fc, o2 = [], [], 0
            for Vp, Fp in ch:
                Vc.append(Vp)
                Fc.append(Fp + o2)
                o2 += len(Vp)
            Vc = np.vstack(Vc) / 1000.0
            Fc = np.vstack(Fc)
            fn = os.path.join(out, "full", "%s_%d.stl" % (g, ci))
            trimesh.Trimesh(Vc, Fc, process=False).export(fn)
            geo["files"]["full_chunks"][g].append({"file": "full/%s_%d.stl" % (g, ci), "sha256": _sha(fn), "faces": int(len(Fc))})
        if g == "other":
            Vd, Fd = fast_simplification.simplify(V.astype(np.float32), F.astype(np.int32),
                                                  target_reduction=1.0 - other_keep, agg=7)
            setting = {"tool": "fast_simplification %s" % fast_simplification.__version__ if hasattr(
                fast_simplification, "__version__") else "fast_simplification 0.2.0", "target_reduction": 1.0 - other_keep, "agg": 7}
        else:
            Vd, Fd = V, F
            setting = {"tool": "none (kept at full resolution)"}
        fn = os.path.join(out, "dec", g + ".stl")
        trimesh.Trimesh(np.asarray(Vd, float), np.asarray(Fd), process=False).export(fn)
        geo["files"]["dec/%s.stl" % g] = {"sha256": _sha(fn), "faces": int(len(Fd)), "decimation": setting}
    # ---- collision fits ----
    col = {}
    for (V, F), l in zip(pieces, lab):
        if l == "sock":
            s = 1.0 if V[:, 2].mean() > 0 else -1.0
            c0 = np.array([0.0, 0.0, s * 145.4])
            d0 = np.linalg.norm(V - c0, axis=1)
            outer = V[(d0 > 16.5) & (s * (V[:, 2] - c0[2]) > 0.3)]
            c, r, res = fit_sphere(outer)
            col["sock_%s" % ("pz" if s > 0 else "nz")] = {
                "type": "sphere", "centre_m": (c / 1000).tolist(), "radius_m": r / 1000,
                "fit": "algebraic least-squares sphere on %d outer-dome vertices of the end-cap cover" % len(outer),
                "residual_rms_mm": float(np.sqrt((res ** 2).mean())), "residual_max_abs_mm": float(np.abs(res).max())}
        if l == "encA":
            s = 1.0 if V[:, 2].mean() > 0 else -1.0
            rr = np.hypot(V[:, 0], V[:, 1])
            outer = rr > 19.5
            r = float(rr[outer].mean())
            col["encA_%s" % ("pz" if s > 0 else "nz")] = {
                "type": "cylinder", "axis": "strut z", "radius_m": r / 1000,
                "z_range_m": [float(V[:, 2].min() / 1000), float(V[:, 2].max() / 1000)],
                "fit": "radius = mean distance from the rod axis of %d outer-wall vertices (r > 19.5 mm); axis = rod axis" % int(outer.sum()),
                "residual_rms_mm": float(np.sqrt(((rr[outer] - r) ** 2).mean())),
                "residual_max_abs_mm": float(np.abs(rr[outer] - r).max()),
                "max_vertex_radius_mm": float(rr.max())}
    E = [pieces[i] for i in range(len(pieces)) if lab[i] == "elec"]
    U = np.vstack([e[0] for e in E])
    hull = trimesh.convex.convex_hull(U)
    fn = os.path.join(out, "col", "elec_hull.stl")
    trimesh.Trimesh(hull.vertices / 1000.0, hull.faces, process=False).export(fn)
    vol_parts = float(sum(abs(trimesh.Trimesh(v, f, process=False).volume) for v, f in E))
    samp, _ = trimesh.sample.sample_surface_even(hull, 4000, seed=1)
    parts_mesh = trimesh.util.concatenate([trimesh.Trimesh(v, f, process=False) for v, f in E])
    _, dist, _ = trimesh.proximity.closest_point(parts_mesh, samp)
    col["elec_hull"] = {"type": "mesh (convex hull)", "file": "col/elec_hull.stl", "sha256": _sha(fn),
                        "hull_vertices": int(len(hull.vertices)), "hull_volume_mm3": float(hull.volume),
                        "parts_volume_mm3": vol_parts,
                        "fit": "convex hull of both electronics-enclosure halves",
                        "residual_standoff_p95_mm": float(np.percentile(dist, 95)),
                        "residual_standoff_max_mm": float(dist.max()),
                        "max_vertex_radius_mm": float(np.hypot(U[:, 0], U[:, 1]).max())}
    for (V, F), l in zip(pieces, lab):
        if l == "endcap":
            s = "pz" if V[:, 2].mean() > 0 else "nz"
            h = trimesh.convex.convex_hull(V)
            fn = os.path.join(out, "col", "cap_hull_%s.stl" % s)
            trimesh.Trimesh(h.vertices / 1000.0, h.faces, process=False).export(fn)
            col["cap_hull_%s" % s] = {"type": "mesh (convex hull)", "file": "col/cap_hull_%s.stl" % s, "sha256": _sha(fn),
                                      "hull_vertices": int(len(h.vertices)),
                                      "fit": "convex hull of the end cap (hooks included); optional collision geom"}
    geo["collision"] = col
    # ---- features ----
    feat = {}
    caps = {("pz" if v[:, 2].mean() > 0 else "nz"): v for (v, f), l in zip(pieces, lab) if l == "endcap"}
    feat["hooks_pz"] = hook_features(caps["pz"], 1.0)
    feat["hooks_nz"] = hook_features(caps["nz"], -1.0)
    G = [v for (v, f), l in zip(pieces, lab) if l == "guide"][0]
    gc = 0.5 * (G.min(0) + G.max(0))
    encB = {("pz" if v[:, 2].mean() > 0 else "nz"): v for (v, f), l in zip(pieces, lab) if l == "encB"}
    win = {("pz" if v[:, 2].mean() > 0 else "nz"): v for (v, f), l in zip(pieces, lab) if l == "winch"}
    Rb, tb, resb = two_fold_axis(encB["pz"], encB["nz"])
    Rw, tw, resw = two_fold_axis(win["pz"], win["nz"])
    g2 = Rb @ gc + tb
    feat["guide_pz"] = {"centre_mm": gc.tolist(), "source": "ring cable guide piece (BOM 'Cable guide'), bbox centre"}
    feat["guide_nz"] = {"centre_mm": g2.tolist(), "source": (
        "image of guide_pz under the two-fold symmetry that maps motor enclosure B (+z) onto B (-z) exactly "
        "(Procrustes residual %.2e mm; same map takes the +z winch onto the -z winch, residual %.2e mm). "
        "The mesh carries only one of the two guides listed in the BOM" % (resb, resw))}
    feat["two_fold_R"] = Rb.tolist()
    for s in ("pz", "nz"):
        Wv = win[s]
        wc = 0.5 * (Wv.min(0) + Wv.max(0))
        rr = np.hypot(Wv[:, 0] - wc[0], Wv[:, 1] - wc[1])
        drum = (np.abs(Wv[:, 2] - wc[2]) < 2.5) & (rr < 4.5)       # between the flanges
        feat["winch_%s" % s] = {"centre_mm": wc.tolist(), "drum_radius_mm": float(rr[drum].max()),
                                "flange_radius_mm": float(rr.max()),
                                "method": "radius about the winch's own axis (parallel to the rod), max over the drum band |dz| < 2.5 mm"}
    A = [v for (v, f), l in zip(pieces, lab) if l == "arduino"][0]
    ca = A.mean(0)
    w_, E_ = np.linalg.eigh(np.cov((A - ca).T))
    n = E_[:, 0]
    feat["arduino_board"] = {"centroid_mm": ca.tolist(), "normal": n.tolist(), "long_axis": E_[:, 2].tolist(),
                             "source": "Arduino Nano 33 IoT piece (community model inside the authors' mesh); PCA axes"}
    geo["features"] = feat
    os.makedirs(paths.DATA_DIR, exist_ok=True)
    with open(paths.data_path("strut_geometry.json"), "w") as fh:
        json.dump(geo, fh, indent=1)
    write_hull_json(out)
    return geo


def read_binary_stl(path):
    """Vertices (float32, first-occurrence order of the de-duplicated corners) and faces of a binary STL: the
    same vertex set and face order MuJoCo's STL loader produces (checked: an inline mesh with these
    vertex / face lists compiles to the identical mesh, geom pose and bounding data)."""
    import struct
    raw = open(path, "rb").read()
    n = struct.unpack("<I", raw[80:84])[0]
    assert len(raw) == 84 + 50 * n, "binary STL expected"
    tri = np.frombuffer(raw[84:], dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]), count=n)
    V = tri["v"].reshape(-1, 3)
    uniq, first, inv = np.unique(V, axis=0, return_index=True, return_inverse=True)
    order = np.argsort(first)
    rank = np.empty_like(order)
    rank[order] = np.arange(len(order))
    return uniq[order], rank[inv.reshape(-1)].reshape(-1, 3)


def write_hull_json(asset_dir=None):
    """Copy the electronics-enclosure collision hull (31 vertices) into notes/sim_base_data/collision_hull_elec.json
    so that the physics model needs no asset directory (ysim 0.3.0)."""
    fn = os.path.join(paths.assets_dir(asset_dir), "col", "elec_hull.stl")
    V, F = read_binary_stl(fn)
    doc = {"note": "convex hull of the two electronics-enclosure halves of the authors' strut mesh (strut frame, "
                   "metres); copied from the asset col/elec_hull.stl by ysim.assets.write_hull_json",
           "source_file": "col/elec_hull.stl", "source_sha256": _sha(fn),
           "vertices_m": [[float(x) for x in v] for v in V], "faces": F.astype(int).tolist()}
    with open(paths.data_path("collision_hull_elec.json"), "w") as fh:
        json.dump(doc, fh, indent=1)
        fh.write("\n")
    return doc


if __name__ == "__main__":
    g = build(*(sys.argv[1:3]))
    print(json.dumps({k: v for k, v in g.items() if k in ("piece_labels", "collision")}, indent=1)[:4000])
    print(json.dumps(g["features"], indent=1)[:3000])
