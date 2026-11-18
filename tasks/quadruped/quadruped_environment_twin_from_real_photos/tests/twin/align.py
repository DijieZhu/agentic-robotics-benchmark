"""The alignment of a twin's free frame onto the site frame.

    al = align_twin(twin, truth, cfg, scene)   # Alignment: .map (final SimMap agent -> site), .map2, .map1, .s,
                                               # .up_tilt_deg, .refined, .inliers, .info, .fail (None or the reason)
    m = SimMap(s, R, t)                        # x_site = s R x_agent + t; to_site / to_agent / dir_* / len_*

Camera i of the agent: centre c_i, rotation A_i (camera to agent world, OpenCV axes); true: C_i, T_i.
1. Gate fit (G0): RANSAC over every triple of the agent's valid photo cameras that have a true counterpart, in
   lexicographic photo order, skipping triples whose TRUE triangle is smaller than ransac.min_triangle_m2; a
   similarity by Umeyama per triple; inliers = cameras within ransac.inlier_m; the triple with the most inliers wins
   (ties: the first); the similarity is refitted on the inliers until the set is stable. G0 needs at least
   ransac.min_inliers inliers, an inlier RMS of at most ransac.max_rms_m and a median angle between R A_i and T_i
   of at most ransac.max_median_angle_deg.
2. Rotation from the orientations: R0 = the SO(3) projection of sum_i T_i A_i^T over the inliers; then s0, t0 by
   least squares on the centres with R0 fixed.
3. Structure refinement from (s0, R0, t0): every (subsampled) c3 ray is mapped into the agent's frame and cast
   against the agent's static collision geometry; the hit, mapped back, is on the true ray, and r_k is its signed
   distance along the true normal to the true surface point. A Huber (refine.huber_m) Gauss-Newton fit of a
   7-parameter similarity update taken about c0, the centroid of the inlier TRUE camera centres (scale, rotation,
   translation at c0), re-casting every iteration, residuals above refine.ignore_above_m ignored, at most
   refine.iterations iterations. Trust region against step 2: rotation angle refine.trust_rot_deg, |s / s0 - 1|
   refine.trust_scale, displacement at c0 refine.trust_trans_m. Leaving it at any iteration, or ending with fewer
   than refine.fallback_min_share of the rays within refine.fallback_within_m (a miss counts as outside), keeps the
   map of step 2.
4. The final map is snapped to the identity when it is within identity_snap of it (the true twin's own map is then
   exactly the identity and s* exactly 1). up_tilt = the angle between R e_z and e_z.
"""
import itertools
import math

import numpy as np

from .raycast import cast_grouped


# ---------------------------------------------------------------------------------------------------------------
# rotations and similarities

def rodrigues(w):
    w = np.asarray(w, float)
    th = float(np.linalg.norm(w))
    if th < 1e-15:
        K = np.array([[0.0, -w[2], w[1]], [w[2], 0.0, -w[0]], [-w[1], w[0], 0.0]])
        return np.eye(3) + K
    a = w / th
    K = np.array([[0.0, -a[2], a[1]], [a[2], 0.0, -a[0]], [-a[1], a[0], 0.0]])
    return np.eye(3) + math.sin(th) * K + (1.0 - math.cos(th)) * (K @ K)


def rot_angle_deg(R):
    """The rotation angle of R in degrees, accurate near 0 (atan2 of the sine and cosine parts; acos of the trace
    alone has a floor of about 1e-6 deg, above the identity snap)."""
    R = np.asarray(R, float)
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return math.degrees(math.atan2(0.5 * float(np.linalg.norm(v)), 0.5 * (float(np.trace(R)) - 1.0)))


def proj_so3(M):
    U, _S, Vt = np.linalg.svd(np.asarray(M, float))
    D = np.diag([1.0, 1.0, float(np.sign(np.linalg.det(U @ Vt))) or 1.0])
    return U @ D @ Vt


class SimMap:
    """x_site = s R x_agent + t (directions: R d; lengths: s L)."""

    def __init__(self, s=1.0, R=None, t=None):
        self.s = float(s)
        self.R = np.eye(3) if R is None else np.asarray(R, float).reshape(3, 3)
        self.t = np.zeros(3) if t is None else np.asarray(t, float).reshape(3)

    @classmethod
    def identity(cls):
        return cls(1.0, np.eye(3), np.zeros(3))

    @classmethod
    def from_dict(cls, d):
        return cls(d["s"], d["R"], d["t"])

    def as_dict(self):
        return {"s": self.s, "R": [[float(v) for v in row] for row in self.R], "t": [float(v) for v in self.t]}

    def is_identity(self):
        return self.s == 1.0 and bool(np.all(self.R == np.eye(3))) and bool(np.all(self.t == 0.0))

    def to_site(self, P):
        return self.s * (np.asarray(P, float) @ self.R.T) + self.t

    def to_agent(self, X):
        return ((np.asarray(X, float) - self.t) @ self.R) / self.s

    def dir_to_site(self, D):
        return np.asarray(D, float) @ self.R.T

    def dir_to_agent(self, D):
        return np.asarray(D, float) @ self.R

    def len_to_site(self, L):
        return self.s * np.asarray(L, float)

    def len_to_agent(self, L):
        return np.asarray(L, float) / self.s

    def inverse(self):
        Ri = self.R.T
        return SimMap(1.0 / self.s, Ri, -(Ri @ self.t) / self.s)

    def compose(self, other):
        """self o other: x -> self(other(x))."""
        return SimMap(self.s * other.s, self.R @ other.R, self.s * (self.R @ other.t) + self.t)

    def up_tilt_deg(self):
        z = self.R[:, 2]
        return math.degrees(math.acos(max(-1.0, min(1.0, float(z[2]) / float(np.linalg.norm(z))))))


def umeyama(src, dst, with_scale=True):
    """(s, R, t) minimising sum |dst - (s R src + t)|^2 (Umeyama 1991, proper rotation)."""
    src = np.asarray(src, float)
    dst = np.asarray(dst, float)
    ms, md = src.mean(axis=0), dst.mean(axis=0)
    Xs, Xd = src - ms, dst - md
    Sig = Xd.T @ Xs / len(src)
    U, D, Vt = np.linalg.svd(Sig)
    S = np.diag([1.0, 1.0, -1.0 if np.linalg.det(U) * np.linalg.det(Vt) < 0 else 1.0])
    R = U @ S @ Vt
    var = float((Xs ** 2).sum()) / len(src)
    s = float(np.trace(np.diag(D) @ S)) / var if (with_scale and var > 0) else 1.0
    return s, R, md - s * (R @ ms)


def _umeyama_batch(src, dst):
    """Vectorised Umeyama over triples: src, dst (T, k, 3) -> s (T,), R (T, 3, 3), t (T, 3)."""
    ms, md = src.mean(axis=1), dst.mean(axis=1)
    Xs, Xd = src - ms[:, None], dst - md[:, None]
    Sig = np.einsum("tki,tkj->tij", Xd, Xs) / src.shape[1]
    U, D, Vt = np.linalg.svd(Sig)
    d = np.where(np.linalg.det(U) * np.linalg.det(Vt) < 0, -1.0, 1.0)
    S = np.ones((len(src), 3))
    S[:, 2] = d
    R = np.einsum("tij,tj,tjk->tik", U, S, Vt)
    var = (Xs ** 2).sum(axis=(1, 2)) / src.shape[1]
    s = (D * S).sum(axis=1) / np.where(var > 0, var, 1.0)
    t = md - s[:, None] * np.einsum("tij,tj->ti", R, ms)
    return s, R, t


# ---------------------------------------------------------------------------------------------------------------
# the three steps

def _angles_deg(R, A, T):
    """Angle between R A_i and T_i per camera (deg)."""
    M = np.einsum("nji,jk,nkl->nil", T, R, A)          # T_i^T R A_i
    v = np.stack([M[:, 2, 1] - M[:, 1, 2], M[:, 0, 2] - M[:, 2, 0], M[:, 1, 0] - M[:, 0, 1]], axis=1)
    return np.degrees(np.arctan2(0.5 * np.linalg.norm(v, axis=1), 0.5 * (np.trace(M, axis1=1, axis2=2) - 1.0)))


def gate_fit(c, A, C, T, photos, cfg):
    """Step 1. c, A: agent centres (n, 3) and rotations (n, 3, 3); C, T: the true ones; photos (n,). Returns
    (SimMap or None, inlier mask (n,), info)."""
    rc = cfg["alignment"]["ransac"]
    n = len(c)
    info = {"n_common": int(n)}
    if n < 3:
        info["fail"] = "only %d photo cameras match a true photo (at least 3 for the fit)" % n
        return None, np.zeros(n, bool), info
    tri = np.array(list(itertools.combinations(range(n), 3)), int)
    P = C[tri]
    area = 0.5 * np.linalg.norm(np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0]), axis=1)
    tri = tri[area >= float(rc["min_triangle_m2"])]
    info["n_triples"] = int(len(tri))
    if not len(tri):
        info["fail"] = "no three matched cameras span a true triangle of %g m2" % float(rc["min_triangle_m2"])
        return None, np.zeros(n, bool), info
    thr = float(rc["inlier_m"])
    best = (-1, None)
    for k0 in range(0, len(tri), 4096):
        tb = tri[k0:k0 + 4096]
        s, R, t = _umeyama_batch(c[tb], C[tb])
        pred = s[:, None, None] * np.einsum("tij,nj->tni", R, c) + t[:, None, :]
        res = np.linalg.norm(pred - C[None], axis=2)
        cnt = (res <= thr).sum(axis=1)
        j = int(np.argmax(cnt))
        if int(cnt[j]) > best[0]:
            best = (int(cnt[j]), k0 + j)
    inl = None
    s, R, t = _umeyama_batch(c[tri[best[1]]][None], C[tri[best[1]]][None])
    s, R, t = float(s[0]), R[0], t[0]
    hist = []
    for _ in range(int(rc["refit_iterations"])):
        res = np.linalg.norm(s * (c @ R.T) + t - C, axis=1)
        new = res <= thr
        hist.append(int(new.sum()))
        if inl is not None and np.array_equal(new, inl):
            break
        inl = new
        if inl.sum() < 3:
            break
        s, R, t = umeyama(c[inl], C[inl])
    res = np.linalg.norm(s * (c @ R.T) + t - C, axis=1)
    inl = res <= thr
    ni = int(inl.sum())
    rms = float(np.sqrt(np.mean(res[inl] ** 2))) if ni else math.inf
    ang = _angles_deg(R, A[inl], T[inl]) if ni else np.zeros(0)
    med = float(np.median(ang)) if ni else math.inf
    info.update({"best_triple": [int(photos[k]) for k in tri[best[1]]], "triple_inliers": best[0],
                 "refit_history": hist, "n_inliers": ni, "inlier_rms_m": rms, "median_angle_deg": med,
                 "inliers": [int(p) for p in photos[inl]], "s": s,
                 "residual_m": {int(photos[k]): float(res[k]) for k in range(n)}})
    why = []
    if ni < int(rc["min_inliers"]):
        why.append("%d cameras agree with the camera fit (at least %d within %.2f m are needed)" % (
            ni, int(rc["min_inliers"]), thr))
    if ni and rms > float(rc["max_rms_m"]):
        why.append("the inlier RMS of the camera fit is %.3f m (at most %.3f m)" % (rms, float(rc["max_rms_m"])))
    if ni and med > float(rc["max_median_angle_deg"]):
        why.append("after the camera fit the median camera orientation is %.2f deg off (at most %.1f deg): check "
                   "the camera convention (MuJoCo cameras look along their -z; twin_camera.camera_xml)" % (
                       med, float(rc["max_median_angle_deg"])))
    if why:
        info["fail"] = "; ".join(why)
        return None, inl, info
    return SimMap(s, R, t), inl, info


def rotation_from_orientations(c, A, C, T):
    """Step 2 on the inliers: R0 = proj_SO3(sum T_i A_i^T); s0, t0 by least squares with R0 fixed."""
    R0 = proj_so3(np.einsum("nij,nkj->ik", T, A))
    cb, Cb = c.mean(axis=0), C.mean(axis=0)
    X, Y = (c - cb) @ R0.T, C - Cb
    den = float((X * X).sum())
    s0 = float((X * Y).sum()) / den if den > 0 else 1.0
    t0 = Cb - s0 * (R0 @ cb)
    res = np.linalg.norm(s0 * (c @ R0.T) + t0 - C, axis=1)
    return SimMap(s0, R0, t0), {"s0": s0, "rms_m": float(np.sqrt(np.mean(res ** 2))),
                                "median_angle_deg": float(np.median(_angles_deg(R0, A, T)))}


def _subsample(n, cap):
    if n <= cap:
        return np.arange(n)
    step = int(math.ceil(n / float(cap)))
    return np.arange(0, n, step)


def _residuals(cur, scene, O, U, X, Nn):
    """Cast the true rays through `cur` into the agent scene: (hit (n,), r (n,), p (n, 3) site hit points)."""
    oa = cur.to_agent(O)
    ua = cur.dir_to_agent(U)
    dist, _own, _src, _nrm = cast_grouped(scene, oa, ua)
    hit = dist >= 0
    lam = np.where(hit, cur.s * dist, 0.0)
    p = O + lam[:, None] * U
    r = np.where(hit, np.einsum("ij,ij->i", Nn, p - X), np.inf)
    return hit, r, p


def refine_structure(map2, scene, c3, c0, cfg):
    """Step 3. Returns (SimMap, refined (bool), info)."""
    rc = cfg["alignment"]["refine"]
    n_all = len(c3["origin"])
    info = {"n_samples": int(n_all)}
    if scene is None or n_all == 0:
        info["fallback"] = "no structure to refine on"
        return map2, False, info
    idx = _subsample(n_all, int(rc["max_rays"]))
    O = np.asarray(c3["origin"], float)[idx]
    U = np.asarray(c3["direction"], float)[idx]
    U = U / np.linalg.norm(U, axis=1)[:, None]
    X = np.asarray(c3["point"], float)[idx]
    Nn = np.asarray(c3["normal"], float)[idx]
    Nn = Nn / np.linalg.norm(Nn, axis=1)[:, None]
    info["n_rays"] = int(len(idx))
    delta = float(rc["huber_m"])
    ign = float(rc["ignore_above_m"])
    a0 = map2.to_agent(c0[None])[0]
    cur = map2
    hist = []
    for it in range(int(rc["iterations"])):
        hit, r, p = _residuals(cur, scene, O, U, X, Nn)
        use = hit & (np.abs(r) <= ign)
        row = {"iteration": it, "hits": int(hit.sum()), "used": int(use.sum())}
        if use.sum() < 20:
            hist.append(row)
            info.update(history=hist, fallback="fewer than 20 rays hit the static geometry within %.2f m" % ign)
            return map2, False, info
        ru = r[use]
        w = np.where(np.abs(ru) <= delta, 1.0, delta / np.maximum(np.abs(ru), delta))
        q = p[use] - c0
        n_ = Nn[use]
        J = np.c_[np.einsum("ij,ij->i", n_, q), np.cross(q, n_), n_]
        sw = np.sqrt(w)
        x = np.linalg.lstsq(J * sw[:, None], -ru * sw, rcond=1e-12)[0]
        eps, om, dt = float(x[0]), x[1:4], x[4:7]
        Rw = rodrigues(om)
        new = SimMap(cur.s * (1.0 + eps), Rw @ cur.R, (1.0 + eps) * (Rw @ (cur.t - c0)) + c0 + dt)
        row.update({"rms_used_m": float(np.sqrt(np.mean(ru ** 2))), "step_rad": float(np.linalg.norm(om)),
                    "step_m": float(np.linalg.norm(dt)), "step_scale": eps})
        hist.append(row)
        rot = rot_angle_deg(new.R @ map2.R.T)
        dsc = abs(new.s / map2.s - 1.0)
        dtr = float(np.linalg.norm(new.to_site(a0[None])[0] - c0))
        if rot > float(rc["trust_rot_deg"]) or dsc > float(rc["trust_scale"]) or dtr > float(rc["trust_trans_m"]):
            info.update(history=hist, fallback="the refinement left the trust region at iteration %d (rotation %.3f "
                        "deg, scale %.4f, translation %.3f m from the camera map)" % (it, rot, dsc, dtr))
            return map2, False, info
        cur = new
        if (float(np.linalg.norm(om)) < float(rc["converge_rad"]) and abs(eps) < float(rc["converge_rad"])
                and float(np.linalg.norm(dt)) < float(rc["converge_m"])):
            break
    hit, r, _p = _residuals(cur, scene, O, U, X, Nn)
    share = float(np.mean(hit & (np.abs(r) <= float(rc["fallback_within_m"]))))
    rot = rot_angle_deg(cur.R @ map2.R.T)
    info.update({"history": hist, "iterations": len(hist), "final_share_within": share,
                 "final_hits": int(hit.sum()), "delta_rot_deg": rot, "delta_scale": cur.s / map2.s - 1.0,
                 "delta_trans_m": float(np.linalg.norm(cur.to_site(a0[None])[0] - c0))})
    if share < float(rc["fallback_min_share"]):
        info["fallback"] = "only %.1f %% of the rays end within %.2f m of the true surfaces after the refinement" % (
            100 * share, float(rc["fallback_within_m"]))
        return map2, False, info
    return cur, True, info


def snap(m, cfg):
    sn = cfg["alignment"]["identity_snap"]
    if (abs(m.s - 1.0) <= float(sn["scale"]) and math.radians(rot_angle_deg(m.R)) <= float(sn["rad"])
            and float(np.linalg.norm(m.t)) <= float(sn["m"])):
        return SimMap.identity(), True
    return m, False


class Alignment:
    def __init__(self):
        self.map = self.map1 = self.map2 = None
        self.refined = False
        self.inliers = []
        self.info = {}
        self.fail = None
        self.snapped = False

    @property
    def s(self):
        return self.map.s

    @property
    def up_tilt_deg(self):
        return self.map.up_tilt_deg()

    def summary(self):
        out = {"fail": self.fail, "refined": self.refined, "snapped": self.snapped, "inliers": self.inliers}
        for k in ("map1", "map2", "map"):
            m = getattr(self, k)
            out[k] = m.as_dict() if m is not None else None
        if self.map is not None:
            out["s_star"] = self.map.s
            out["up_tilt_deg"] = self.map.up_tilt_deg()
        out["info"] = self.info
        return out


def align_cameras(agent, truth, cfg, scene):
    """The three steps for agent cameras {photo: (centre (3,), R_cv (3, 3))} against truth.cameras, refining on
    `scene` (the agent's static geometry, agent frame; None skips step 3)."""
    al = Alignment()
    photos = np.array(sorted(p for p in agent if p in truth.cameras), int)
    c = np.array([agent[p][0] for p in photos], float).reshape(-1, 3)
    A = np.array([agent[p][1] for p in photos], float).reshape(-1, 3, 3)
    C = np.array([truth.cameras[p].C for p in photos], float).reshape(-1, 3)
    T = np.array([truth.cameras[p].T for p in photos], float).reshape(-1, 3, 3)
    m1, inl, i1 = gate_fit(c, A, C, T, photos, cfg)
    al.info["step1"] = i1
    if m1 is None:
        al.fail = i1.get("fail", "the camera fit failed")
        return al
    al.map1 = m1
    al.inliers = [int(p) for p in photos[inl]]
    m2, i2 = rotation_from_orientations(c[inl], A[inl], C[inl], T[inl])
    al.map2 = m2
    al.info["step2"] = dict(i2, angle_to_step1_deg=rot_angle_deg(m2.R @ m1.R.T))
    c0 = C[inl].mean(axis=0)
    mf, refined, i3 = refine_structure(m2, scene, truth.c3, c0, cfg)
    al.info["step3"] = i3
    al.refined = refined
    al.map, al.snapped = snap(mf, cfg)
    al.info["c0"] = [float(v) for v in c0]
    return al


def align_twin(twin, truth, cfg, scene):
    agent = {p: (cam["p"], cam["R_cv"]) for p, cam in twin.cameras.items()}
    return align_cameras(agent, truth, cfg, scene)


# ---------------------------------------------------------------------------------------------------------------
# errors between maps, and the real-SfM-frame self-test

def map_error(m_est, m_ref, centre, radii=(0.0, 5.0, 10.0, 15.0), z=0.0, n_bearings=32):
    """Displacement in the site frame between two agent -> site maps, at site points on circles of the given radii
    around `centre` (xy) at height z: {r: {"h_med", "h_max", "v_med", "v_max"}} (m), plus the rotation angle."""
    out = {"rot_deg": rot_angle_deg(m_est.R @ m_ref.R.T), "scale": m_est.s / m_ref.s - 1.0}
    inv = m_ref.inverse()
    for r in radii:
        th = 2 * math.pi * np.arange(n_bearings) / n_bearings
        P = np.c_[centre[0] + r * np.cos(th), centre[1] + r * np.sin(th), np.full(n_bearings, z)]
        D = m_est.to_site(inv.to_site(P)) - P
        h = np.linalg.norm(D[:, :2], axis=1)
        v = np.abs(D[:, 2])
        out[float(r)] = {"h_med": float(np.median(h)), "h_max": float(h.max()), "v_med": float(np.median(v)),
                         "v_max": float(v.max()), "d_max": float(np.linalg.norm(D, axis=1).max())}
    return out


def selftest_sfm(truth, cfg):
    """Self-test of the alignment: the true twin re-expressed in a stored real SfM frame (its static scene seen through
    S_site_from_sfm) is aligned with that reconstruction's cameras; the recovered map must equal S_site_from_sfm
    within selftest_sfm.max_disp_m at selftest_sfm.radius_m from the camera path and selftest_sfm.max_rot_deg.
    Returns the measured numbers; raises truth.TruthError on a failure."""
    from .raycast import MappedScene
    from .truth import TruthError
    st = cfg["alignment"]["selftest_sfm"]
    d = truth.load("sfm_selftest")
    if d.get("format") != "site-sfm-selftest/1":
        raise TruthError("sfm_selftest.json has format %r" % d.get("format"))
    S = SimMap.from_dict(d["S_site_from_sfm"])
    agent = {int(r["photo"]): (np.asarray(r["centre"], float), np.asarray(r["R_sfm_from_cam"], float))
             for r in d["cameras"]}
    al = align_cameras(agent, truth, cfg, MappedScene(truth.static_scene(), S))
    if al.fail:
        raise TruthError("the SfM-frame self-test fails the camera fit: %s" % al.fail)
    c0 = np.asarray(al.info["c0"], float)
    err = map_error(al.map, S, c0, radii=(0.0, float(st["radius_m"])), z=0.0)
    worst = max(err[float(st["radius_m"])]["d_max"], err[0.0]["d_max"])
    out = {"refined": al.refined, "rot_deg": err["rot_deg"], "disp_m_at_radius": worst,
           "step2_error": map_error(al.map2, S, c0, radii=(0.0, float(st["radius_m"])), z=0.0)}
    if not al.refined:
        raise TruthError("the SfM-frame self-test fell back to the camera map: %s" % al.info["step3"].get("fallback"))
    if worst > float(st["max_disp_m"]) or err["rot_deg"] > float(st["max_rot_deg"]):
        raise TruthError("the SfM-frame self-test recovers the stored frame only within %.4f m / %.4f deg at %g m "
                         "(limits %g m / %g deg)" % (worst, err["rot_deg"], float(st["radius_m"]),
                                                    float(st["max_disp_m"]), float(st["max_rot_deg"])))
    return out
