"""Plume source localisation (numpy + scipy only): sonar sections of the drifted bubble column at two heights above the
roof, the 1/7-law drift line, a camera look at the roof exit, and the general-arrangement flanges as source
candidates.  Port of the G3 pilot v3 (proto/g3_plume/pilot_plume.py 39a417e3..., mission.py fa7de69c...,
naive_exit.py find_exit) against an abstract platform instead of G3's kinematic test world.

Algorithm (G3 v3, records/G3.md section 1.5):
  1. coarse: one 360 deg scan at roof + 3.5 m above the structure centre (12 m range) -> column c35 (component with
     the largest integrated echo above the median + 6 dB); one 360 deg scan at roof + 2 m from c35 (8 m) -> c2;
     drift vector s0 = (c35 - c2) / (G(3.5) - G(2)), G = the 1/7-law height integral (current given 2 m above bed);
  2. sections: heights roof + 1.5 / 3.5 m interleaved, `cycles` (5) visits each, `n_sweeps` (3) +-50 deg sector
     sweeps from a cross-flow standoff; each visit aimed at the median of that height's earlier centroids; persistent
     component >= 2 m^2 within 4.5 m of the prediction; excess-weighted centroid;
  3. line c(z) = e + s0 G(z) through the per-height median centroids (e = roof exit, s0 = U(2 m) / w_b);
  4. camera: look around / downstream 1 m above the roof, 45 deg down, two frames 0.2 s apart (moving bright points),
     project onto the roof plane, upstream end of the densest streak = exit; averaged with the sonar exit when the
     two agree within 1.5 m;
  5. source: each of the 12 drawing flanges predicts exits x_kj + a s0, a in [0.2 G_in(z_hi), 1.0 G_in(z_lo)]
     (in-structure reduction 0.2-1.0 and leak height unknown); the flange whose segment is nearest to the exit wins.
     A continuous estimate x = e - 0.6 s0 G_in(1.3 m) is reported too (G3: 0.2-1.5 m error; the flanges make it work).

The platform (the program's motion + sensor layer) provides:
    put_sonar(xy_w, z_w, yaw)          move so that the sonar head is at (xy, z) with heading yaw (time passes)
    scan360(range_m) -> [ping]         one full turn of the head (pings: see sonar_map)
    sector(bearing, half_deg, range_m, n_sweeps) -> [[ping]]   heading = bearing, +-half_deg back and forth
    put_camera(xyz_w, yaw, tilt_rad)   move so that the camera pupil is at xyz with heading yaw and the given tilt
    frame() -> (rgb, cam)              one frame at the current pose, lamps on (cam = pose.Cam of that frame)
    wait(dt);  t                       time
    outline                            the gripper drawing (body point sets) or None: own-gripper mask for spots

    res = locate(platform, ga)          # res["branch"] (1..6), res["slot"], res["flange"], res["exit_S"], ...
"""
from __future__ import annotations

import math

import numpy as np
from scipy import ndimage as ndi

from . import sonar_map as SM
from . import spots as SP
from .blade import jaw_mask

F_HAT, Z_HAT = 0.6, 1.3                # continuous estimate: assumed in-structure reduction and leak height
F_PRIOR = (0.2, 1.0)                   # GA candidates: in-structure reduction range
MIN_SECTION_AREA = 2.0                 # m^2: plume sections in the persistent map are 8-36 m^2 (G3)
H_CAM = 1.0                            # camera height above the roof for the exit look
TILT_CAM = -45.0                       # deg (the lamps point forward: a nadir camera is unlit, G3)


# ----------------------------------------------------------------------------------------------------- 1/7 law
def I_prof(a, b):
    """int_a^b (h / 2 m)^(1/7) dh, heights above the sea bed."""
    a = np.maximum(a, 0.0)
    b = np.maximum(b, 0.0)
    return (7.0 / 8.0) * (b ** (8.0 / 7.0) - a ** (8.0 / 7.0)) / 2.0 ** (1.0 / 7.0)


def G(ga: SM.ManifoldGA, z_s: float) -> float:
    """int_{roof}^{z} (h / 2 m)^(1/7) dh for an S height z (above the mudmat top)."""
    hb = ga.mudmat_thk
    if z_s >= ga.roof_z:
        return float(I_prof(ga.roof_z + hb, z_s + hb))
    return -float(I_prof(z_s + hb, ga.roof_z + hb))


def G_in(ga: SM.ManifoldGA, z_src_s: float) -> float:
    return float(I_prof(z_src_s + ga.mudmat_thk, ga.roof_z + ga.mudmat_thk))


# ----------------------------------------------------------------------------------------------------- sonar steps
def scan_component(pf, ga, z_s, xy_w, range_m, half=None, res=0.2, delta_db=6.0, min_area=0.3, codec=SM.CODEC):
    pf.put_sonar(xy_w, ga.origin_w[2] + z_s, 0.0)
    pings = pf.scan360(range_m)
    half = half or range_m + 1.0
    g = SM.Grid(xy_w[0], xy_w[1], half, res)
    S, N = SM.accumulate(g, pings, codec)
    M = SM.smooth_map(g, S, N, 1.5)
    valid = ~SM.structure_mask(g, ga, pings[0]["o"])
    best, comps, bg = SM.single_sweep_best(g, M, valid, delta_db=delta_db, min_area_m2=min_area)
    return best, comps, bg, dict(grid=g, M=M, valid=valid, pings=pings)


def coarse(pf, ga, codec=SM.CODEC) -> dict:
    c0 = ga.to_w(np.array([0.0, 0.0, 0.0]))[:2]
    b3, comps3, bg3, _ = scan_component(pf, ga, ga.roof_z + 3.5, c0, 12.0, codec=codec)
    out = dict(ok=False, n_comp35=len(comps3), bg35_db=bg3)
    if b3 is None:
        return out
    p3w = np.array([b3["cx"], b3["cy"]])
    b2, comps2, bg2, aux = scan_component(pf, ga, ga.roof_z + 2.0, p3w, 8.0, half=9.0, res=0.1, min_area=0.1,
                                          codec=codec)
    out.update(n_comp2=len(comps2), bg2_db=bg2, scan2_t=(aux["pings"][0]["t"], aux["pings"][-1]["t"]))
    if b2 is None:
        return out
    p3 = ga.to_s(np.r_[p3w, 0.0])[:2]
    p2 = ga.to_s(np.array([b2["cx"], b2["cy"], 0.0]))[:2]
    s0 = (p3 - p2) / (G(ga, ga.roof_z + 3.5) - G(ga, ga.roof_z + 2.0))
    out.update(ok=True, p3_S=p3, p2_S=p2, s0_S=s0, exit_lin_S=p2 - s0 * G(ga, ga.roof_z + 2.0), area2=b2["area"])
    return out


def predict_S(ga, co, z_s):
    return co["p2_S"] + co["s0_S"] * (G(ga, z_s) - G(ga, ga.roof_z + 2.0))


def flow_dir(co):
    s = co["s0_S"]
    n = float(np.linalg.norm(s))
    return (s / n) if n > 1e-6 else np.array([1.0, 0.0]), n


def section(pf, ga, co, z_s, n_sweeps=12, prior_r=4.5, pred_S=None, codec=SM.CODEC) -> dict:
    """Persistent section at S height z_s from a cross-flow standoff (sonar 2 / 3 / 4 / 5 / 7 m from the predicted
    column at roof + 1 / 1.5 / 2 / 3.5 / 6 m: the fan's 20 deg lower edge meets the roof at dh / tan 20), +-50 deg."""
    pS = predict_S(ga, co, z_s) if pred_S is None else np.asarray(pred_S, float)
    dh = z_s - ga.roof_z
    standoff = 2.0 if dh <= 1.1 else (3.0 if dh <= 1.6 else (4.0 if dh <= 2.5 else (5.0 if dh <= 4.0 else 7.0)))
    range_m = standoff + 4.0
    d, _ = flow_dir(co)
    nrm = np.array([-d[1], d[0]])
    side = 1.0 if float((np.zeros(2) - pS) @ nrm) >= 0 else -1.0
    stS = pS + side * standoff * nrm
    pw = ga.to_w(np.r_[pS, 0.0])[:2]
    sw_ = ga.to_w(np.r_[stS, 0.0])[:2]
    bearing = math.atan2(pw[1] - sw_[1], pw[0] - sw_[0])
    pf.put_sonar(sw_, ga.origin_w[2] + z_s, bearing)
    sweeps = pf.sector(bearing, 50.0, range_m, n_sweeps)
    g = SM.Grid(pw[0], pw[1], 7.0, 0.1)
    maps = SM.sweep_maps(g, sweeps, codec)
    valid = ~SM.structure_mask(g, ga, sweeps[0][0]["o"])
    best, comps, info = SM.persistent_section(g, maps, valid, prior_xy=pw, prior_r=prior_r)
    rejected = None
    if best is not None and best["area"] < MIN_SECTION_AREA:
        rejected, best = dict(area=best["area"]), None
    res = dict(z_s=z_s, pred_S=pS, standoff_S=stS, bg_db=info["bg_db"], n_comp=len(comps),
               t0=sweeps[0][0]["t"], t1=sweeps[-1][-1]["t"], rejected_small=rejected)
    if best is not None:
        reg = ndi.binary_dilation(best["mask"], iterations=5) & valid & np.isfinite(info["mean"])
        bg = 10 ** (info["bg_db"] / 10.0)
        ex = np.where(reg, np.maximum(np.nan_to_num(info["mean"]) - 2.0 * bg, 0.0), 0.0)
        X, Y = g.centres()
        wsum = ex.sum()
        cxy = np.array([(X * ex).sum() / wsum, (Y * ex).sum() / wsum]) if wsum > 0 else np.array([best["cx"], best["cy"]])
        res["c_S"] = ga.to_s(np.r_[cxy, 0.0])[:2]
        res["area"] = best["area"]
        res["energy"] = best["energy"]
    return res


def fit_line(ga, secs):
    """Line c(z) = e + s0 G(z) through the per-height MEDIAN centroid (robust to one bad visit)."""
    good = [x for x in secs if "c_S" in x]
    hs = sorted({round(x["z_s"], 3) for x in good})
    if len(hs) < 2:
        return None, None
    Gs = np.array([G(ga, h) for h in hs])
    Cs = np.array([np.median(np.array([x["c_S"] for x in good if round(x["z_s"], 3) == h]), axis=0) for h in hs])
    A = np.column_stack([np.ones_like(Gs), Gs])
    sol, *_ = np.linalg.lstsq(A, Cs, rcond=None)
    return sol[0], sol[1]


# ----------------------------------------------------------------------------------------------------- camera exit
def put_camera_S(pf, ga, xy_s, yaw_s):
    pf.put_camera(ga.to_w(np.r_[xy_s, ga.roof_z + H_CAM]), ga.psi + yaw_s, math.radians(TILT_CAM))


def shoot(pf, ga, cam_S, yaw_s, dt=0.2):
    """Camera H_CAM above the roof at cam_S, heading yaw_s (S), 45 deg down: moving bright points (two frames dt
    apart, own gripper masked) projected onto the roof plane (S), and the nearest roof point in view."""
    put_camera_S(pf, ga, cam_S, yaw_s)
    f1, _ = pf.frame()
    pf.wait(max(dt - 1.0 / 30.0, 0.0))
    f2, cam = pf.frame()
    excl = jaw_mask(cam, f2.shape[:2], pf.outline) if getattr(pf, "outline", None) else None
    u, v = SP.moving_spots(f1, f2, exclude=excl)
    xy, tt = SP.project_to_plane(cam, u, v, ga.roof_w)
    okp = np.isfinite(tt) & (tt > 0) & (tt < 4.0)
    P = ga.to_s(np.column_stack([xy[okp], np.zeros(okp.sum())]))[:, :2] if np.any(okp) else np.zeros((0, 2))
    gxy, _ = SP.project_to_plane(cam, np.array([cam.W / 2.0]), np.array([cam.H - 0.5]), ga.roof_w)
    near_S = ga.to_s(np.r_[gxy[0], 0.0])[:2]
    return P, near_S, dict(n_spots=int(len(u)))


def look_around(pf, ga, centre_S, n=6, min_pts=20, max_moves=4, step=1.5):
    c = np.asarray(centre_S, float).copy()
    allp = []
    for mv in range(max_moves + 1):
        counts, pts = [], []
        for k in range(n):
            P, _, _ = shoot(pf, ga, c, 2 * math.pi * k / n)
            pts.append(P)
            counts.append(P.shape[0])
        allp += pts
        if sum(counts) >= min_pts or mv == max_moves:
            break
        kb = int(np.argmax(counts))
        a = 2 * math.pi * kb / n
        c = c + step * np.array([math.cos(a), math.sin(a)])
    return np.vstack(allp) if allp else np.zeros((0, 2))


def look_downstream(pf, ga, cam_S, d, offsets_deg=(-40.0, 0.0, 40.0)):
    pts, nears = [], []
    yaw0 = math.atan2(d[1], d[0])
    for o in offsets_deg:
        P, near_S, _ = shoot(pf, ga, cam_S, yaw0 + math.radians(o))
        pts.append(P)
        nears.append(near_S)
    return np.vstack(pts), nears


def streak_exit(P, start_S, d, nrm, cell=0.25, min_per_cell=2, min_n=10):
    if P.shape[0] < min_n:
        return None
    s = (P - start_S) @ d
    t = (P - start_S) @ nrm
    i = np.floor(s / cell).astype(int)
    j = np.floor(t / cell).astype(int)
    i0, j0 = i.min(), j.min()
    H = np.zeros((i.max() - i0 + 1, j.max() - j0 + 1), int)
    np.add.at(H, (i - i0, j - j0), 1)
    lab, n = ndi.label(H >= min_per_cell, structure=np.ones((3, 3)))
    if n == 0:
        return None
    sizes = ndi.sum(H, lab, index=np.arange(1, n + 1))
    keep = lab[i - i0, j - j0] == (1 + int(np.argmax(sizes)))
    if keep.sum() < min_n:
        return None
    ss, tt = s[keep], t[keep]
    s10 = float(np.percentile(ss, 10))
    near = ss <= np.percentile(ss, 30)
    t_end = float(np.median(tt[near]))
    return dict(s10=s10, t_end=t_end, n=int(keep.sum()), s_min=float(ss.min()), s_max=float(ss.max()),
                centroid=start_S + float(ss.mean()) * d + float(tt.mean()) * nrm)


def find_exit(pf, ga, start, d, max_iter=3, seed=None):
    """a) look around (6 headings) 1 m above the roof at `seed` (the sonar's column position at the camera height)
    -> bubble streak -> its upstream end; b) 1 m upstream of the estimate, looking downstream (+-40 deg): refine;
    while the upstream end lies within 0.4 m of the nearest roof point in view, move 1 m upstream (<= max_iter)."""
    nrm = np.array([-d[1], d[0]])
    hist = []
    seed = start if seed is None else np.asarray(seed, float)
    P = look_around(pf, ga, seed)
    res = streak_exit(P, seed, d, nrm)
    if res is None or res["n"] < 30:
        P2 = look_around(pf, ga, start)
        r2 = streak_exit(P2, start, d, nrm)
        if r2 is not None and (res is None or r2["n"] > res["n"]):
            P, res, seed = P2, r2, start
    hist.append(dict(stage="look_around", centre_S=seed.copy(), n_pts=int(P.shape[0]),
                     streak=None if res is None else {k: v for k, v in res.items() if k != "centroid"}))
    est = start.copy() if res is None else seed + res["s10"] * d + res["t_end"] * nrm
    found = res is not None
    for it in range(max_iter + 1):
        cam = est - 1.0 * d
        P2, nears = look_downstream(pf, ga, cam, d)
        r2 = streak_exit(P2, cam, d, nrm)
        s_near = float(np.min([(q - cam) @ d for q in nears]))
        hist.append(dict(stage="downstream", cam_S=cam.copy(), n_pts=int(P2.shape[0]), s_near=s_near,
                         streak=None if r2 is None else {k: v for k, v in r2.items() if k != "centroid"}))
        if r2 is None:
            break
        found = True
        est = cam + r2["s10"] * d + r2["t_end"] * nrm
        if r2["s10"] < s_near + 0.4 and it < max_iter:
            continue
        break
    return (est if found else None), hist


# ----------------------------------------------------------------------------------------------------- source
def ga_candidates(ga, e, s0):
    """Rank the 12 drawing flanges by the distance of their exit segment x + a s0 (a in the prior range) to e."""
    a_lo = F_PRIOR[0] * G_in(ga, ga.src_z_range[1])
    a_hi = F_PRIOR[1] * G_in(ga, ga.src_z_range[0])
    a_mid = 0.5 * (a_lo + a_hi)
    ss = float(s0 @ s0)
    cands = []
    for k in range(len(ga.slots)):
        for j in range(ga.flanges.shape[1]):
            x = ga.flanges[k, j]
            a = float(np.clip(((e - x) @ s0) / ss, a_lo, a_hi)) if ss > 1e-12 else a_mid
            r = float(np.linalg.norm(e - (x + a * s0)))
            cands.append(dict(slot=k, branch=k + 1, flange=j, residual=r, a=a, key=(round(r, 3), abs(a - a_mid))))
    cands.sort(key=lambda c: c["key"])
    for c in cands:
        c.pop("key")
    return cands


def locate(pf, ga, heights=(1.5, 3.5), n_sweeps=3, cycles=5, camera_exit=True, cam_consistent_m=1.5,
           codec=SM.CODEC) -> dict:
    """The G3 v3 pilot.  Returns dict(branch, slot, flange, exit_S, s0_S, x_cont_S, x_ga_S, candidates, ...);
    branch None if no plume was found."""
    t_start = pf.t
    co = coarse(pf, ga, codec)
    out = dict(program="plume_fit.locate", coarse={k: v for k, v in co.items()}, heights=list(heights),
               cycles=cycles, n_sweeps=n_sweeps)
    if not co.get("ok"):
        out.update(slot=None, branch=None, reason="no plume candidate in the coarse scans", t_used=pf.t - t_start)
        return out
    secs = []
    cur = dict(co)
    for cyc in range(cycles):
        for dh in heights:
            prev = [x["c_S"] for x in secs if "c_S" in x and abs(x["z_s"] - (ga.roof_z + dh)) < 1e-6]
            pred = np.median(np.array(prev), axis=0) if prev else None
            s = section(pf, ga, cur, ga.roof_z + dh, n_sweeps=n_sweeps, pred_S=pred, codec=codec)
            secs.append(s)
            e_, s0_ = fit_line(ga, secs)
            if e_ is not None:
                cur = dict(cur, p2_S=e_ + s0_ * G(ga, ga.roof_z + 2.0), s0_S=s0_)
    out["sections"] = secs
    e, s0 = fit_line(ga, secs)
    if e is None:
        out.update(slot=None, branch=None, reason="fewer than two section heights", t_used=pf.t - t_start)
        return out
    out["exit_sonar_S"] = e
    if camera_exit:
        n = float(np.linalg.norm(s0))
        d = s0 / n if n > 1e-6 else np.array([1.0, 0.0])
        ec, hist = find_exit(pf, ga, e.copy(), d, seed=e + s0 * G(ga, ga.roof_z + H_CAM))
        out["camera_exit_iterations"] = hist
        if ec is not None:
            out["exit_camera_S"] = ec
            out["camera_minus_sonar_m"] = float(np.linalg.norm(ec - e))
            if out["camera_minus_sonar_m"] <= cam_consistent_m:      # two independent looks that agree: average
                e = 0.5 * (e + ec)
                out["exit_used"] = "mean(sonar, camera)"
            else:
                out["exit_used"] = "sonar (camera inconsistent)"
    out.setdefault("exit_used", "sonar")
    x_cont = e - F_HAT * s0 * G_in(ga, Z_HAT)
    cands = ga_candidates(ga, e, s0)
    best = cands[0]
    other = [c for c in cands if c["slot"] != best["slot"]]
    out.update(exit_S=e, s0_S=s0, x_cont_S=x_cont, slot_cont=ga.nearest_slot(x_cont), slot=int(best["slot"]),
               branch=int(best["branch"]), flange=int(best["flange"]), x_ga_S=ga.flanges[best["slot"], best["flange"]],
               candidates=cands[:4], best_other_branch_residual=other[0]["residual"] if other else None,
               t_used=pf.t - t_start)
    return out
