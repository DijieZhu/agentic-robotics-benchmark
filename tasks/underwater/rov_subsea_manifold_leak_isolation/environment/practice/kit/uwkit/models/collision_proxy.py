"""Collision proxies of the reference vehicle beyond the digest's convex pieces (used by models/vehicle.py).

merged hulls  (build_vehicle(collision="merged"))
    The digest decomposes every external part into convex pieces (CoACD / hulls; 502 for mission).  The merged proxy
    groups the parts per physical assembly (frame plates, the two enclosures, each thruster, fairing + foam per corner,
    guards, guard brackets, centre fairings, lights, gripper body, sonar, DVL, ballast + bottom plate) and replaces
    each group by ONE convex hull of the union of its pieces' vertices (MuJoCo computes the hull at compile time).
    mission: 34 hulls instead of 500 pieces.  The grouping only merges parts whose hull adds material where nothing
    outside the vehicle can reach (inside the frame, under the bottom plate, between a fairing and its foam); the
    deviation from the piece proxy is measured by the 07 gate G0 (DEV/proto/g0_vehicle).
jaw shells    (build_vehicle(jaw_collision="shell"))
    The jaws keep the gripping pads and get two more convex pieces each, cut from the jaw's own convex hull so that
    they never enter the gap between the gripping faces: a "base" piece behind the pads (throat and pin knuckles,
    gripping-face side clipped at the pad face) and an "outer" piece above / below the pad (the jaw's tapered back).
    Without them a paddle can pass through the jaw behind the pads or over the jaw's back (only the pads collide).

Pure numpy + MuJoCo (the convex hull comes from a scratch MjSpec compile); no scipy.
"""
from __future__ import annotations

import os
from collections import OrderedDict

import numpy as np
import mujoco

MERGED_RULES_DOC = (
    "bottom = frame bottom plate + ballast weights; side_plate_<side>; centre_plates_<side> (front + rear centre plate); "
    "battery_enclosure (+ 3-inch tube clamp halves); electronics_enclosure (+ 4-inch tube cradles); "
    "fairing_<fore|aft>_<side> (top fairing shell + its foam block); thruster_<k>; guard_<side>; "
    "guard_bracket_<fore|aft>_<side>; centre_fairing_<side> (centre fairing cap + round foam); light_<k>; "
    "gripper (body + clamp halves); sonar (+ mount); dvl (+ bracket); any other part: its own hull")


# --------------------------------------------------------------------------------------------- mesh io
def read_stl_vertices(path: str) -> np.ndarray:
    """Unique vertices (n, 3) of a binary or ASCII STL file (mesh units = metres in the digest library)."""
    with open(path, "rb") as f:
        data = f.read()
    n = int.from_bytes(data[80:84], "little") if len(data) >= 84 else -1
    if n >= 0 and len(data) == 84 + 50 * n:
        rec = np.frombuffer(data, dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]),
                            count=n, offset=84)
        v = rec["v"].reshape(-1, 3).astype(float)
    else:
        txt = data.decode("ascii", "replace").split()
        v = np.array([[float(txt[i + 1]), float(txt[i + 2]), float(txt[i + 3])]
                      for i, w in enumerate(txt) if w == "vertex"], float)
    return np.unique(np.round(v, 9), axis=0)


def transform(points: np.ndarray, T) -> np.ndarray:
    T = np.asarray(T, float)
    return points @ T[:3, :3].T + T[:3, 3]


# --------------------------------------------------------------------------------------------- hulls
def convex_hull(points: np.ndarray):
    """(vertices, faces) of the convex hull of a point cloud, computed by MuJoCo's own hull code (the one used for
    collision), so the proxy is exactly what the simulator sees.  faces index into the returned vertices."""
    pts = np.unique(np.round(np.asarray(points, float), 9), axis=0)
    s = mujoco.MjSpec()
    s.add_mesh(name="h", uservert=pts.ravel().tolist())
    b = s.worldbody.add_body(name="b")
    b.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname="h", mass=0.0)
    b.explicitinertial = True
    b.mass = 1.0
    b.fullinertia = [1, 1, 1, 0, 0, 0]
    m = s.compile()
    g = m.mesh_graph[m.mesh_graphadr[0]:]
    nv, nf = int(g[0]), int(g[1])
    vid = np.asarray(g[2 + nv:2 + 2 * nv], int)
    fglob = np.asarray(g[2 + 3 * nv + 3 * nf:2 + 3 * nv + 6 * nf], int).reshape(nf, 3)
    if m.mesh_vertnum[0] != len(pts):
        raise RuntimeError("MuJoCo changed the vertex count of a hull input")
    remap = -np.ones(len(pts), int)
    remap[vid] = np.arange(nv)
    return pts[vid], remap[fglob]


def hull_edges(faces: np.ndarray) -> np.ndarray:
    e = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    return np.unique(np.sort(e, axis=1), axis=0)


def clip_convex(points: np.ndarray, normal, offset) -> np.ndarray:
    """Vertices of hull(points) intersected with the half-space {x : normal . x <= offset} (exact for a convex set:
    kept hull vertices + the crossings of hull edges with the plane).  Empty array if nothing is left."""
    V, F = convex_hull(points)
    nrm = np.asarray(normal, float)
    s = V @ nrm - float(offset)
    keep = V[s <= 1e-12]
    E = hull_edges(F)
    a, b = s[E[:, 0]], s[E[:, 1]]
    cross = (a < -1e-12) & (b > 1e-12) | (a > 1e-12) & (b < -1e-12)
    t = a[cross] / (a[cross] - b[cross])
    P = V[E[cross, 0]] + t[:, None] * (V[E[cross, 1]] - V[E[cross, 0]])
    out = np.vstack([keep, P]) if len(P) else keep
    return out if len(out) >= 4 else np.zeros((0, 3))


def clip_many(points: np.ndarray, planes) -> np.ndarray:
    for n, d in planes:
        if len(points) < 4:
            return np.zeros((0, 3))
        points = clip_convex(points, n, d)
    return points


# --------------------------------------------------------------------------------------------- merged groups
def merged_group_key(item: dict) -> str:
    """Neutral group name of a manifest item for the merged proxy (see MERGED_RULES_DOC)."""
    gn = str(item.get("generic_name", ""))
    cat, grp = item.get("category", ""), item.get("group", "")
    lo, hi = (np.asarray(x, float) for x in item["bbox_B_m"])
    c = 0.5 * (lo + hi)
    side = "port" if c[1] > 0 else "stbd"
    fore = "fwd" if c[0] > 0 else "aft"
    if cat == "ballast" or gn == "frame bottom plate":
        return "bottom"
    if gn == "frame side plate":
        return f"side_plate_{side}"
    if gn in ("front centre plate", "rear centre plate"):
        return f"centre_plates_{side}"
    if grp == "battery_enclosure" or gn == "3-inch tube clamp half":
        return "battery_enclosure"
    if grp == "electronics_enclosure" or gn == "4-inch tube cradle":
        return "electronics_enclosure"
    if gn in ("top fairing shell", "buoyancy foam block"):
        return f"fairing_{fore}_{side}"
    if cat == "thruster":
        return "thruster_" + str(item["id"]).rsplit("_", 1)[-1]
    if gn == "vertical-thruster guard plate":
        return f"guard_{side}"
    if gn == "guard bracket":
        return f"guard_bracket_{fore}_{side}"
    if gn in ("centre fairing cap", "round buoyancy foam block"):
        return f"centre_fairing_{side}"
    if cat == "light":
        for tok in str(item["id"]).split("."):
            if tok.startswith("light_"):
                return tok
    if grp in ("gripper", "sonar", "dvl"):
        return grp
    return "part_" + "".join(ch if ch.isalnum() else "_" for ch in gn)[:40] + f"_{fore}_{side}"


def merged_groups(items, root: str, exclude_ids=()) -> "OrderedDict[str, dict]":
    """group key -> dict(items=[ids], points=(n, 3) vertices in B of all collision pieces of the group)."""
    out: "OrderedDict[str, dict]" = OrderedDict()
    for it in items:
        if it["id"] in exclude_ids or not it.get("mesh") or not it["mesh"].get("collision"):
            continue
        key = merged_group_key(it)
        pts = [transform(read_stl_vertices(os.path.join(root, c)), it["T_B"]) for c in it["mesh"]["collision"]]
        g = out.setdefault(key, dict(items=[], points=[]))
        g["items"].append(it["id"])
        g["points"].extend(pts)
    for g in out.values():
        g["points"] = np.unique(np.round(np.vstack(g["points"]), 9), axis=0)
    return out


# --------------------------------------------------------------------------------------------- jaw shells
def jaw_shell_pieces(points_B: np.ndarray, tcp_B, R_grip, side_sign: float, pad: dict) -> dict:
    """Convex pieces of one jaw (closed CAD state) that stay out of the gripping gap.

    points_B : vertices of the jaw's convex hull in B (closed state);  R_grip : columns = gripper axis, lateral,
    opening direction (B);  side_sign : +1 upper jaw, -1 lower jaw;  pad : the gripper meta 'pad' dict
    (face offset, thickness, x_from = pad back end along the axis from the TCP, ...).
    Returns {"base": pts_B, "outer": pts_B}: base = behind the pad (axis coordinate <= x_from), gripping side clipped
    at the pad face; outer = over the pad's length, clipped at the pad's outer face (the pad itself is a separate box)."""
    tcp = np.asarray(tcp_B, float)
    R = np.asarray(R_grip, float)
    g = (np.asarray(points_B, float) - tcp) @ R               # gripper coordinates (x along the axis, z opening)
    face, th, x_from = float(pad["face"]), float(pad["thickness"]), float(pad["x_from"])
    zdir = np.array([0.0, 0.0, -side_sign])                    # keep side_sign * z >= limit  <=>  -side_sign*z <= -limit
    base = clip_many(g, [(np.array([1.0, 0, 0]), x_from), (zdir, -face)])
    outer = clip_many(g, [(np.array([-1.0, 0, 0]), -x_from), (zdir, -(face + th))])
    back = lambda q: (tcp + q @ R.T) if len(q) else q            # noqa: E731
    return {"base": back(base), "outer": back(outer)}
