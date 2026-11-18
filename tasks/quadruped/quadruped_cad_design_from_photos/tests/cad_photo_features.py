"""cad_photo.features: model-agnostic feature probes of the hardware-design grader (hardware_design.md 6.3
feature_fidelity). Shipped twice, byte-identical (tests/cad_photo_features.py and the practice checker). Pure geometry
(numpy, scipy, PIL, manifold3d, trimesh); the ground-truth records are passed in by the caller. Lengths mm.

    section_mask(mesh, frame, window, px)       material of a planar section, rasterised (even-odd over the loops)
    cutout_item(agent_mesh, gt_mesh, feature)   one GT cut-out on the ALIGNED submitted part (both in the GT link frame)
    pocket_floor_mm(mesh, T_servo)              median ray distance from inside a seated servo to the pocket floor
    base_openings_item(agent_link, gt_link)     the through-openings of the base part seen from below (Hungarian matched)

Cut-out rule. The GT record gives the cut's probe plane (the centre at mid-depth, the cut direction `normal`, in-plane axes
e1 / e2) and its footprint polygon (mm, about the centre). Both parts are sliced through that plane at 0.1 mm. The
reference region R = the GT section with the cut filled (GT material + footprint): the submitted part's EMPTY region is
E = R minus its material, so only the region where the GT part is otherwise solid counts (the space outside the part
never looks like a cut-out). The footprint is cross-correlated with E over +-6 mm; the cut-out is present iff the best
overlap reaches 0.6 of the footprint area; then the connected empty component with the largest overlap gives the centre
offset (from the GT centre) and the area; item = present x mean(band(offset; 0.3, 3 mm), band(|area error| / area;
0.10, 0.50)). A missing cut-out scores 0; an extra one costs shape credit, not here.

Pocket floor rule. 25 rays start inside the seated servo's case (servo frame x = -20 mm, y in {-8, -4, 0, 4, 8}, z in
{5, 10, 14.4, 19, 24}) and run along the servo's -x (away from the pocket mouth); the median distance to the first
surface is the floor position. Both the GT part and the aligned submitted part are probed with the GT servo pose, so a
spacer the submission ignores (or invents) shows as the floor error.
"""

import math

import numpy as np

PX = 0.1
SEARCH_MM = 6.0
PRESENT_FRAC = 0.6
RAY_Y = (-8.0, -4.0, 0.0, 4.0, 8.0)
RAY_Z = (5.0, 10.0, 14.4, 19.0, 24.0)
RAY_X0 = -20.0
RAY_MISS_MM = 60.0


def band(value, full, zero):
    v = abs(float(value))
    if not math.isfinite(v):
        return 0.0
    if v <= full:
        return 1.0
    if v >= zero:
        return 0.0
    return (zero - v) / (zero - full)


def _manifold(mesh):
    import manifold3d as m3d
    mesh = mesh if not hasattr(mesh, "vertices") else mesh
    return m3d.Manifold(m3d.Mesh(vert_properties=np.asarray(mesh.vertices, np.float32), tri_verts=np.asarray(mesh.faces, np.uint32)))


def plane_frame(feature):
    c = np.asarray(feature["centre_mm"], float)
    d = np.asarray(feature["normal"], float)
    e1 = np.asarray(feature["e1"], float)
    e2 = np.asarray(feature["e2"], float)
    d, e1, e2 = d / np.linalg.norm(d), e1 / np.linalg.norm(e1), e2 / np.linalg.norm(e2)
    return c, d, e1, e2


def _loops_in_plane(mesh, frame):
    """Section loops (u, v) of `mesh` (trimesh, link frame) cut by the plane through c normal d; u = e1, v = e2."""
    c, d, e1, e2 = frame
    R = np.vstack([e1, e2, d])                     # rows: plane coordinates
    V = (np.asarray(mesh.vertices, float) - c) @ R.T
    import trimesh
    m = trimesh.Trimesh(V, np.asarray(mesh.faces), process=False)
    try:
        cs = _manifold(m).slice(0.0)
        return [np.asarray(p, float) for p in cs.to_polygons() if len(p) >= 3]
    except Exception:  # noqa: BLE001 - an invalid submitted mesh has no section
        return []


def rasterise(loops, window, px=PX):
    """Even-odd fill of the loops over window (u0, v0, u1, v1) at px mm: bool array [nv, nu] (row = v)."""
    from PIL import Image, ImageDraw
    u0, v0, u1, v1 = window
    nu, nv = int(round((u1 - u0) / px)), int(round((v1 - v0) / px))
    acc = np.zeros((nv, nu), bool)
    for lp in loops:
        img = Image.new("1", (nu, nv), 0)
        pts = [((float(x) - u0) / px - 0.5, (float(y) - v0) / px - 0.5) for x, y in lp]
        if len(pts) < 3:
            continue
        ImageDraw.Draw(img).polygon(pts, fill=1)
        acc ^= np.asarray(img, bool)
    return acc


def section_mask(mesh, frame, window, px=PX):
    return rasterise(_loops_in_plane(mesh, frame), window, px)


def cutout_item(agent_mesh, gt_mesh, feature, px=PX, search_mm=SEARCH_MM):
    """Score one GT cut-out on the aligned submitted part (module docstring). Returns a dict with score and stats."""
    from scipy import ndimage, signal
    frame = plane_frame(feature)
    fp = np.asarray(feature["footprint_mm"], float)
    lo, hi = fp.min(axis=0), fp.max(axis=0)
    pad = search_mm + 1.0
    win = (float(lo[0] - pad), float(lo[1] - pad), float(hi[0] + pad), float(hi[1] + pad))
    F = rasterise([fp], win, px)
    gt = section_mask(gt_mesh, frame, win, px)
    ag = section_mask(agent_mesh, frame, win, px)
    R = gt | F
    E = R & ~ag
    fa = int(F.sum())
    out = {"present": 0.0, "overlap": 0.0, "offset_mm": float("nan"), "area_error": float("nan"), "score": 0.0}
    if fa == 0:
        return out
    # the footprint kernel (its bounding box) correlated over +-search in both directions
    ys, xs = np.nonzero(F)
    ky0, ky1, kx0, kx1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    K = F[ky0:ky1, kx0:kx1].astype(np.float32)
    s = int(round(search_mm / px))
    y0, x0 = ky0 - s, kx0 - s
    y0c, x0c = max(0, y0), max(0, x0)
    sub = E[y0c:ky1 + s, x0c:kx1 + s].astype(np.float32)
    corr = signal.fftconvolve(sub, K[::-1, ::-1], mode="valid")
    k = np.unravel_index(int(np.argmax(corr)), corr.shape)
    best = float(corr[k]) / fa
    out["overlap"] = round(best, 4)
    if best < PRESENT_FRAC:
        return out
    dy, dx = (k[0] + y0c) - ky0, (k[1] + x0c) - kx0
    Fs = np.zeros_like(F)
    Fs[ky0 + dy:ky1 + dy, kx0 + dx:kx1 + dx] = F[ky0:ky1, kx0:kx1]
    lab, n = ndimage.label(E, structure=np.ones((3, 3)))
    if n == 0:
        return out
    ov = ndimage.sum(Fs, lab, index=np.arange(1, n + 1))
    comp = lab == (int(np.argmax(ov)) + 1)
    area = float(comp.sum()) * px * px
    cy, cx = [float(v) for v in ndimage.center_of_mass(comp)]
    u = win[0] + (cx + 0.5) * px
    v = win[1] + (cy + 0.5) * px
    fcy, fcx = [float(q) for q in ndimage.center_of_mass(F)]
    fu, fv = win[0] + (fcx + 0.5) * px, win[1] + (fcy + 0.5) * px
    off = math.hypot(u - fu, v - fv)
    aerr = abs(area - fa * px * px) / (fa * px * px)
    out.update({"present": 1.0, "offset_mm": round(off, 4), "area_error": round(aerr, 4),
                "score": 0.5 * (band(off, 0.3, 3.0) + band(aerr, 0.10, 0.50))})
    return out


def pocket_rays(T_servo):
    """(origins, directions) of the 25 floor rays in the link frame of the servo pose T_servo (4x4, mm)."""
    R, t = T_servo[:3, :3], T_servo[:3, 3]
    o = np.array([[RAY_X0, y, z] for y in RAY_Y for z in RAY_Z], float)
    return o @ R.T + t, np.tile(R @ np.array([-1.0, 0.0, 0.0]), (len(o), 1))


def pocket_floor_mm(mesh, T_servo):
    """Median distance (mm) from the ray origins to the first surface along the rays; a miss counts RAY_MISS_MM."""
    import trimesh
    origins, dirs = pocket_rays(T_servo)
    try:
        inter = trimesh.ray.ray_triangle.RayMeshIntersector(mesh)
        locs, idx_ray, _tri = inter.intersects_location(origins, dirs, multiple_hits=True)
    except Exception:  # noqa: BLE001
        return RAY_MISS_MM
    dist = np.full(len(origins), RAY_MISS_MM)
    for p, i in zip(locs, idx_ray):
        dd = float(np.dot(p - origins[i], dirs[i]))
        if 1e-6 < dd < dist[i]:
            dist[i] = dd
    return float(np.median(dist))


def hungarian(cost):
    from scipy.optimize import linear_sum_assignment
    r, c = linear_sum_assignment(np.asarray(cost, float))
    return list(zip(r.tolist(), c.tolist()))


def openings_error(agent_holes, gt_holes, zero_mm=3.0):
    """Mean matched centre distance of the base's bottom openings, missing / extra ones counted at zero_mm; None when
    the GT has none."""
    ha = np.asarray(agent_holes, float).reshape(-1, 3)
    hr = np.asarray(gt_holes, float).reshape(-1, 3)
    if len(hr) == 0:
        return None
    if len(ha) == 0:
        return zero_mm
    cost = np.linalg.norm(hr[:, None, :2] - ha[None, :, :2], axis=2)
    pairs = hungarian(cost)
    errs = [float(cost[i, j]) for i, j in pairs]
    missing = len(hr) - len(pairs)
    extra = max(0, len(ha) - len(pairs))
    return float(np.mean(errs + [zero_mm] * (missing + extra)))
