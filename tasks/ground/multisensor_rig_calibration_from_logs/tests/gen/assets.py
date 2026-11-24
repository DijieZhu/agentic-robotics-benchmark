"""The site's assets as triangles: the hall and the props, read from the USD files that both renderers draw.

    load(usd_path) -> dict(V (n,3) float64 in the asset's frame, F (m,3) int32, mat (m,) int32, names [material name])
Cached as <cache>/<name>.npz. Needs usd-core (generation time only; nothing here runs at verify)."""
import hashlib, json, os
import numpy as np


def _tri(counts, idx):
    out = []; k = 0
    for c in counts:
        for j in range(1, c - 1):
            out.append((idx[k], idx[k + j], idx[k + j + 1]))
        k += c
    return np.asarray(out, np.int64).reshape(-1, 3)


def load(usd_path, cache_dir):
    key = hashlib.sha256(open(usd_path, "rb").read()).hexdigest()[:16]
    cp = os.path.join(cache_dir, os.path.splitext(os.path.basename(usd_path))[0] + "_" + key + ".npz")
    if os.path.isfile(cp):
        z = np.load(cp, allow_pickle=False)
        return {"V": z["V"], "F": z["F"], "mat": z["mat"], "names": [str(x) for x in z["names"]]}
    from pxr import Usd, UsdGeom, UsdShade
    st = Usd.Stage.Open(usd_path)
    mpu = UsdGeom.GetStageMetersPerUnit(st); up = UsdGeom.GetStageUpAxis(st)
    assert up == "Z", (usd_path, up)
    Vs, Fs, Ms, names, off = [], [], [], [], 0
    for p in st.Traverse():
        if not p.IsA(UsdGeom.Mesh):
            continue
        if UsdGeom.Imageable(p).ComputeVisibility() == "invisible":
            continue
        m = UsdGeom.Mesh(p)
        pts = m.GetPointsAttr().Get()
        if pts is None or len(pts) == 0:
            continue
        X = np.array(UsdGeom.Xformable(p).ComputeLocalToWorldTransform(Usd.TimeCode.Default()))
        V = (np.asarray(pts, np.float64) @ X[:3, :3] + X[3, :3]) * mpu
        F = _tri(list(m.GetFaceVertexCountsAttr().Get()), list(m.GetFaceVertexIndicesAttr().Get()))
        subsets = UsdGeom.Subset.GetAllGeomSubsets(m)
        bound = UsdShade.MaterialBindingAPI(p).ComputeBoundMaterial()[0]
        base = bound.GetPath().name if bound else p.GetName()
        mat = np.full(len(F), -1, np.int64)
        if subsets:
            # face index -> triangle indices
            counts = list(m.GetFaceVertexCountsAttr().Get()); first = np.concatenate([[0], np.cumsum([c - 2 for c in counts])])
            for s in subsets:
                b = UsdShade.MaterialBindingAPI(s.GetPrim()).ComputeBoundMaterial()[0]
                nm = b.GetPath().name if b else base
                if nm not in names:
                    names.append(nm)
                for f in s.GetIndicesAttr().Get():
                    mat[first[f]:first[f + 1]] = names.index(nm)
        if base not in names:
            names.append(base)
        mat[mat < 0] = names.index(base)
        Vs.append(V); Fs.append(F + off); Ms.append(mat); off += len(V)
    out = {"V": np.concatenate(Vs), "F": np.concatenate(Fs).astype(np.int32), "mat": np.concatenate(Ms).astype(np.int32), "names": names}
    os.makedirs(cache_dir, exist_ok=True)
    np.savez_compressed(cp, V=out["V"], F=out["F"], mat=out["mat"], names=np.array(names))
    return out
