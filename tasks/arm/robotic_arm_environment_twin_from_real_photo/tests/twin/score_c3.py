"""C3 environment: 0.75 x depth share + 0.25 x vertical.

Depth share. The scored pixels are the hidden reference's (depth confirmed by the fused sensor depth and the GT
furniture), every `stride`-th row and column. For each pixel the ray through the TRUE intrinsics is cast into the
aligned agent world (R_pp^-1, then the agent's camera pose) against the agent's furniture: every collision geom that
is not part of an item (static bodies and the drawer), at qpos0, a mesh as the convex hull of its vertices (the
shape it collides with). The hit's ray length times a scale s is compared with the reference ray length. A pixel
counts when the normal distance |error| x max(|n . d|, cos_min) is within 15 mm, n being the true surface normal at
that pixel (from the GT furniture), so oblique surfaces are not over-charged; a miss fails. s is the better of s*
(fitted on the items) and 1 (the agent's own metric scale): a poor item fit must not move correct furniture, and
C6 charges the scale error anyway.

Vertical: band(angle between the agent world's +z mapped into the true camera frame and the true up, 2, 8 deg).
"""
import numpy as np

from .config import band_down
from . import pngio


def raycast_model(twin):
    """(model, data, source geom ids): the twin's furniture collision geoms as static world geoms at their qpos0 poses;
    mesh geoms become vertex-only meshes (MuJoCo ray-casts those as their convex hull)."""
    import mujoco
    m, d = twin.model, twin.data
    spec = mujoco.MjSpec()
    gids = twin.furniture_geoms()
    for g in gids:
        t = int(m.geom_type[g])
        geom = spec.worldbody.add_geom()
        geom.name = "g%d" % g
        q = np.zeros(4)
        mujoco.mju_mat2Quat(q, np.asarray(d.geom_xmat[g], float))
        geom.pos = np.asarray(d.geom_xpos[g], float)
        geom.quat = q
        geom.contype = 0
        geom.conaffinity = 0
        geom.rgba = [1.0, 1.0, 1.0, 1.0]
        if t == int(mujoco.mjtGeom.mjGEOM_MESH):
            mid = int(m.geom_dataid[g])
            va, vn = int(m.mesh_vertadr[mid]), int(m.mesh_vertnum[mid])
            mesh = spec.add_mesh()
            mesh.name = "m%d" % g
            mesh.uservert = np.asarray(m.mesh_vert[va:va + vn], float).ravel().tolist()
            mesh.inertia = mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL
            geom.type = mujoco.mjtGeom.mjGEOM_MESH
            geom.meshname = mesh.name
        else:
            geom.type = mujoco.mjtGeom(t)
            geom.size = np.asarray(m.geom_size[g], float)
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data, gids


def cast(model, data, origin, dirs, cutoff):
    import mujoco
    n = len(dirs)
    gid = np.full(n, -1, np.int32)
    dist = np.full(n, -1.0)
    nrm = np.zeros(3 * n)
    if n:
        mujoco.mj_multiRay(model, data, np.asarray(origin, float).copy(), np.ascontiguousarray(dirs, float).ravel(),
                           None, 1, -1, gid, dist, nrm, n, float(cutoff))
    return dist, gid, nrm.reshape(n, 3)


def load_reference(depth_png, class_png):
    return pngio.read_png(depth_png), pngio.read_png(class_png)


def score(truth, twin, R_pp, s_star, ref, cfg, gt_cast=None):
    """C3 of `twin` against the reference (depth mm uint16, class uint8). R_pp None means the identity."""
    ec = cfg["environment"]
    depth, cls = ref
    stride = int(ec["stride"])
    sel = depth > 0
    if stride > 1:
        grid = np.zeros_like(sel)
        grid[::stride, ::stride] = True
        sel &= grid
    v, u = np.nonzero(sel)
    z = depth[v, u].astype(float) / 1000.0
    K = truth.K
    dc = np.c_[(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones(len(u))]
    dhat = dc / np.linalg.norm(dc, axis=1)[:, None]
    t_ref = z / dhat[:, 2]
    cutoff = float(ec["cutoff_m"])
    if gt_cast is None:
        gm, gd, _ = raycast_model(truth)
        dist_t, gid_t, n_t = cast(gm, gd, truth.p, dhat @ truth.R_cv.T, cutoff)
        gt_cast = (dist_t, gid_t, n_t)
    dist_t, gid_t, n_t = gt_cast
    n_cam = n_t @ truth.R_cv                          # true surface normals in the true camera frame
    cosn = np.abs(np.einsum("ij,ij->i", n_cam, dhat))
    cosn = np.where(gid_t >= 0, cosn, 1.0)
    gt_err = np.abs(dist_t - t_ref) * 1000.0
    gt_agree = float(np.mean((gid_t >= 0) & (gt_err <= float(ec["selftest_tol_mm"])))) if len(u) else 0.0
    # the agent's furniture
    da = dhat if R_pp is None else dhat @ R_pp
    am, ad, _ = raycast_model(twin)
    dist_a, gid_a, _ = cast(am, ad, twin.p, da @ twin.R_cv.T, cutoff)
    hit = gid_a >= 0
    wn = np.maximum(cosn, float(ec["cos_min"]))

    def judge(s):
        e = np.where(hit, np.abs(s * dist_a - t_ref) * 1000.0, np.inf)
        return e, hit & (e * wn <= float(ec["tol_mm"]))

    c = cls[v, u]
    classes = np.unique(c)

    def share_of(ok_):
        # class_balance: every furniture class counts the same, so the large top cannot stand in for the drawer,
        # the shelves or the wall; otherwise the plain share of scored pixels
        if not len(u):
            return 0.0
        if ec.get("class_balance", False):
            return float(np.mean([np.mean(ok_[c == k]) for k in classes]))
        return float(np.mean(ok_))

    err_mm, ok = judge(s_star)
    share_sstar = share_of(ok)
    share_own = share_sstar
    s_used = float(s_star)
    if ec.get("scale_choice", "best_of_sstar_and_own") == "best_of_sstar_and_own" and s_star != 1.0:
        e1, ok1 = judge(1.0)
        share_own = share_of(ok1)
        if share_own > share_sstar:
            err_mm, ok, s_used = e1, ok1, 1.0
    share = share_of(ok)
    per = {}
    for k in classes:
        s_ = c == k
        hs = s_ & hit                                  # the median over the hit pixels only (never NaN)
        per[int(k)] = {"px": int(s_.sum()), "share": float(np.mean(ok[s_])),
                       "median_err_mm": float(np.median(err_mm[hs])) if np.any(hs) else None}
    up_a = twin.up_cam if R_pp is None else R_pp @ twin.up_cam
    up_t = truth.up_cam
    cang = float(np.dot(up_a, up_t) / (np.linalg.norm(up_a) * np.linalg.norm(up_t)))
    vdeg = float(np.degrees(np.arccos(max(-1.0, min(1.0, cang)))))
    vb = band_down(vdeg, *ec["vertical_deg"])
    c3 = float(ec["w_depth"]) * share + float(ec["w_vertical"]) * vb
    info = {"depth_share": share, "depth_share_at_sstar": share_sstar, "depth_share_at_own_scale": share_own,
            "depth_scale_used": s_used, "vertical": vb, "vertical_deg": vdeg, "n_px": int(len(u)), "stride": stride,
            "hit_frac": float(np.mean(hit)) if len(u) else 0.0, "per_class": per,
            "gt_agree_frac": gt_agree, "gt_max_err_mm": float(np.max(gt_err[gid_t >= 0])) if np.any(gid_t >= 0) else None}
    return c3, info, gt_cast
