"""Render level of detail (RA3): deterministic, backend-neutral decimation of triangle meshes (numpy only).

    SceneSpec(lod="policy", lod_params={...})      a render model whose visible mesh geoms are decimated (live cameras at
                                                   policy rate); lod="full" (default) renders the meshes as they are
    resolve_recipe(params) -> recipe               DEFAULT_POLICY_RECIPE updated by params (unknown keys are an error)
    vertex_cluster(V, F, cell_m, representative)   -> (V', F') one mesh
    recipe_digest(recipe, meshes)                  sha256 of the recipe + the decimated vertex / face bytes

Method (Rossignac-Borrel vertex clustering on a uniform grid): every vertex falls in a cubic cell of edge `cell_m`
(grid anchored at the mesh frame's origin); a cell's vertices merge into one representative, faces whose corners
merge are dropped, duplicate faces (the same three clusters in any order) are kept once, in their first order.
Representatives: "qem" (default) = the point minimising the area-weighted squared distances to the planes of the
faces around the cell (Lindstrom 2000), solved per cell by Cramer's rule with a small pull towards the cell mean and
clamped back to the mean when it leaves the cell (elementwise arithmetic and order-fixed sums only, so the result is
bit-reproducible); "mean" = the mean of the cell's vertices.  Budget: with `max_faces` set, the
cell grows by `grow` until the decimated meshes of the scene hold at most that many faces.  The MuJoCo application
(which meshes, frames, recompile) is backends/mujoco_scene.apply_lod; a USD / neural backend consumes the resolved
(decimated) scene through mujoco_scene.snapshot, so every backend draws the same LOD geometry.

Measured on the six-servo desktop arm in a drawer cell (render/README.md "RA3"): 3 mm cells, qem: the drawn mesh faces
fall from 401 k to 19 k (unique visible meshes 325 k -> 16 k), the mesh silhouette keeps IoU 0.997-0.998 against the full
meshes at 320 x 240 / 640 x 480, exact depth on the arm moves by a median 0.01 mm (p90 0.12-0.19 mm).
"""
from __future__ import annotations

import hashlib
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np

LOD_VERSION = "arb-lod/1.0.0"

DEFAULT_POLICY_RECIPE: Dict[str, Any] = {
    "method": "vertex_clustering",
    "cell_m": 0.003,             # grid cell edge (metres, in the mesh frame of the compiled model)
    "representative": "qem",     # "qem" (default) | "mean"
    "min_faces": 500,            # meshes with fewer faces are kept as they are (scale references, small parts)
    "max_faces": None,           # optional budget for the decimated meshes of the scene (unique meshes)
    "grow": 1.25,                # cell growth factor per budget iteration
    "max_iterations": 12,
    "skip_textured": True,       # meshes with texture coordinates are kept (their UV layout would not survive)
}
_METHODS = ("vertex_clustering",)
_REPRESENTATIVES = ("mean", "qem")


def check_recipe(params: Dict[str, Any]) -> None:
    unknown = set(params) - set(DEFAULT_POLICY_RECIPE)
    if unknown:
        raise ValueError("unknown lod_params %s (known: %s)" % (sorted(unknown), sorted(DEFAULT_POLICY_RECIPE)))
    r = dict(DEFAULT_POLICY_RECIPE, **params)
    if r["method"] not in _METHODS:
        raise ValueError("lod method must be one of %s" % (_METHODS,))
    if r["representative"] not in _REPRESENTATIVES:
        raise ValueError("lod representative must be one of %s" % (_REPRESENTATIVES,))
    if not float(r["cell_m"]) > 0 or float(r["grow"]) <= 1.0 or int(r["min_faces"]) < 0 or int(r["max_iterations"]) < 1:
        raise ValueError("lod: cell_m > 0, grow > 1, min_faces >= 0, max_iterations >= 1")
    if r["max_faces"] is not None and int(r["max_faces"]) < 4:
        raise ValueError("lod: max_faces >= 4 or None")


def resolve_recipe(params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    params = dict(params or {})
    check_recipe(params)
    r = dict(DEFAULT_POLICY_RECIPE, **params)
    r["cell_m"] = float(r["cell_m"])
    r["grow"] = float(r["grow"])
    r["min_faces"] = int(r["min_faces"])
    r["max_iterations"] = int(r["max_iterations"])
    r["max_faces"] = None if r["max_faces"] is None else int(r["max_faces"])
    r["skip_textured"] = bool(r["skip_textured"])
    return r


def _det3(a00, a01, a02, a10, a11, a12, a20, a21, a22):
    return a00 * (a11 * a22 - a12 * a21) - a01 * (a10 * a22 - a12 * a20) + a02 * (a10 * a21 - a11 * a20)


def _qem_points(V, F, inv, n_clusters, mean, cell_m):
    """Per-cluster minimiser of the area-weighted plane quadrics of the faces touching the cluster (Cramer's rule)."""
    p0, p1, p2 = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    cr = np.cross(p1 - p0, p2 - p0)
    area2 = np.sqrt((cr * cr).sum(axis=1))
    ok = area2 > 0
    n = np.zeros_like(cr)
    n[ok] = cr[ok] / area2[ok, None]
    d = -(n * p0).sum(axis=1)
    w = 0.5 * area2
    # quadric terms per face: A = w n n^T (6 unique), b = w d n, for each of the 3 corners' clusters
    terms = np.stack([w * n[:, 0] * n[:, 0], w * n[:, 0] * n[:, 1], w * n[:, 0] * n[:, 2], w * n[:, 1] * n[:, 1],
                      w * n[:, 1] * n[:, 2], w * n[:, 2] * n[:, 2], w * d * n[:, 0], w * d * n[:, 1], w * d * n[:, 2]], axis=1)
    acc = np.zeros((n_clusters, 9))
    for k in range(3):
        np.add.at(acc, inv[F[:, k]], terms)
    a00, a01, a02, a11, a12, a22, b0, b1, b2 = acc.T
    lam = 1e-3 * (a00 + a11 + a22) / 3.0 + 1e-18
    a00, a11, a22 = a00 + lam, a11 + lam, a22 + lam
    r0, r1, r2 = lam * mean[:, 0] - b0, lam * mean[:, 1] - b1, lam * mean[:, 2] - b2
    det = _det3(a00, a01, a02, a01, a11, a12, a02, a12, a22)
    good = np.abs(det) > 1e-30
    safe = np.where(good, det, 1.0)
    x = _det3(r0, a01, a02, r1, a11, a12, r2, a12, a22) / safe
    y = _det3(a00, r0, a02, a01, r1, a12, a02, r2, a22) / safe
    z = _det3(a00, a01, r0, a01, a11, r1, a02, a12, r2) / safe
    P = np.stack([x, y, z], axis=1)
    inside = good & (np.abs(P - mean).max(axis=1) <= cell_m)
    return np.where(inside[:, None], P, mean)


def vertex_cluster(V, F, cell_m: float, representative: str = "mean") -> Tuple[np.ndarray, np.ndarray]:
    """Decimate one triangle mesh (V (N, 3) float, F (M, 3) int) on a grid of `cell_m`; returns (V', F') with only
    referenced vertices, F' int32 in the first-occurrence order of the surviving faces."""
    V = np.asarray(V, np.float64)
    F = np.asarray(F, np.int64)
    if len(F) == 0:
        return V.copy(), F.astype(np.int32)
    q = np.floor(V / float(cell_m)).astype(np.int64)
    uq, inv = np.unique(q, axis=0, return_inverse=True)
    inv = inv.reshape(-1)
    n = len(uq)
    sums = np.zeros((n, 3))
    np.add.at(sums, inv, V)
    cnt = np.bincount(inv, minlength=n).astype(np.float64)
    mean = sums / cnt[:, None]
    reps = mean if representative == "mean" else _qem_points(V, F, inv, n, mean, float(cell_m))
    G = inv[F]
    keep = (G[:, 0] != G[:, 1]) & (G[:, 1] != G[:, 2]) & (G[:, 0] != G[:, 2])
    G = G[keep]
    if len(G) == 0:
        return np.zeros((0, 3)), np.zeros((0, 3), np.int32)
    _, first = np.unique(np.sort(G, axis=1), axis=0, return_index=True)
    G = G[np.sort(first)]
    used, remap = np.unique(G.reshape(-1), return_inverse=True)
    return reps[used], remap.reshape(-1, 3).astype(np.int32)


def usable(V2, F2) -> bool:
    """A decimated mesh MuJoCo can compile and that still has a surface: >= 4 vertices spanning 3-D, >= 4 faces."""
    if len(F2) < 4 or len(V2) < 4:
        return False
    ext = np.ptp(V2, axis=0)
    return bool(np.all(ext > 0))


def recipe_digest(recipe: Dict[str, Any], meshes: Iterable[Tuple[str, np.ndarray, np.ndarray]]) -> str:
    """sha256 of the canonical recipe + (name, float64 LE vertices, int32 LE faces) of every decimated mesh, in order."""
    from .spec import canonical_json
    h = hashlib.sha256()
    h.update(canonical_json({"lod_version": LOD_VERSION, "recipe": recipe}).encode())
    for name, V2, F2 in meshes:
        h.update(b"\0" + str(name).encode() + b"\0")
        h.update(np.ascontiguousarray(V2, "<f8").tobytes())
        h.update(np.ascontiguousarray(F2, "<i4").tobytes())
    return h.hexdigest()
