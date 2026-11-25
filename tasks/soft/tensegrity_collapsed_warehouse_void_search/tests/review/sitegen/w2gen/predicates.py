"""Predicates G1-G11 of the W2 family (no robot dynamics).

Visibility = MuJoCo mj_multiRay from the search camera's TRUE position against geom group 0 (boxes, floor,
patches; colour chips and QC markers are excluded). Routes = grid.py (0.02 m grid, disc radius W_SWEEP/2 + 0.04).
G3 renders a QC-only copy of the site with MuJoCo's classic renderer (OSMesa on this host).
"""
import math

import numpy as np

from . import family as fam
from . import grid as gridmod
from . import mjcf

EPS_HIT = 0.015


def credit(d):
    return float(np.clip(1.0 - (d - fam.BL) / fam.BL, 0.0, 1.0))


def compile_world(xml):
    import mujoco
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    return m, d


def cam_pose(site, which="true"):
    T = np.asarray(site["camera"][which]["T_world_camera"], float)
    return T[:3, 3].copy(), T[:3, :3].copy()


def visible(m, d, cpos, pts, groups=(0,)):
    """True where nothing in the given geom groups is hit before the point."""
    import mujoco
    pts = np.asarray(pts, float).reshape(-1, 3)
    vec = pts - cpos
    dist_t = np.linalg.norm(vec, axis=1)
    vn = (vec / dist_t[:, None]).copy()
    n = len(pts)
    gid = np.zeros(n, np.int32)
    dist = np.zeros(n, np.float64)
    grp = np.zeros(6, np.uint8)
    grp[list(groups)] = 1
    mujoco.mj_multiRay(m, d, np.asarray(cpos, float), vn.reshape(-1), grp, 1, -1, gid, dist, None, n, 50.0)
    hit = (dist >= 0) & (dist < dist_t - EPS_HIT)
    return ~hit


def image_rays(site, ss=1):
    cm = site["camera"]["model"]
    W, H = cm["width"], cm["height"]
    offs = (np.arange(ss) + 0.5) / ss - 0.5
    us = (np.arange(W)[:, None] + offs[None, :]).ravel()
    vs = (np.arange(H)[:, None] + offs[None, :]).ravel()
    uu, vv = np.meshgrid(us, vs)
    pos, R = cam_pose(site)
    dirs = np.stack([(uu - cm["cx"]) / cm["fx"], (vv - cm["cy"]) / cm["fy"], np.ones_like(uu)], -1).reshape(-1, 3) @ R.T
    dirs /= np.linalg.norm(dirs, axis=1)[:, None]
    return pos, dirs, uu.shape


def cast_image(m, d, site, ss=1, groups=(0,)):
    import mujoco
    pos, dirs, shape = image_rays(site, ss)
    n = len(dirs)
    gid = np.zeros(n, np.int32)
    dist = np.zeros(n, np.float64)
    grp = np.zeros(6, np.uint8)
    grp[list(groups)] = 1
    mujoco.mj_multiRay(m, d, pos, dirs.copy().reshape(-1), grp, 1, -1, gid, dist, None, n, 50.0)
    hit = pos + dirs * dist[:, None]
    return gid, dist, hit, shape


def _geom_ids(m, prefix):
    import mujoco
    out = []
    for i in range(m.ngeom):
        nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i) or ""
        if nm.startswith(prefix):
            out.append(i)
    return out


def _points(site):
    pts = {"pad": site["pad"]["xy"]}
    for s in site["search_points"]:
        pts[s["id"]] = s["xy"]
    return pts


def _opening(site, gid):
    for row in ("r1", "r2"):
        for g in site[row]["openings"]:
            if g["id"] == gid:
                return g
    raise KeyError(gid)


# ------------------------------------------------------------------------------------------------ G9 (and the route)
def g9_routes(site, xml=None, md=None, step=0.02):
    m, d = md if md else compile_world(xml)
    G = gridmod.Grid(site, step)
    r = site["classes"]["plan_radius"]
    pts = _points(site)
    order = ("pad", 1, 2, 3, "pad")
    base = gridmod.solid_footprints(site)
    tun = [g for g in site["r1"]["openings"] if g["type"] == "tunnel_P"][0]
    gap = [g for g in site["r1"]["openings"] if g["type"] == "gap_P"][0]
    out = {}
    for key, extra in (("a", []), ("b", [(tun["x0"], tun["x1"], tun["y0"], tun["y1"])]),
                       ("c", [(gap["x0"], gap["x1"], gap["y0"], gap["y1"])])):
        ok = G.clearance(base + extra) >= r
        legs = G.legs(ok, pts, order)
        out[key] = dict(legs=legs, connected=all(l["length_m"] is not None for l in legs),
                        total_m=round(sum(l["length_m"] for l in legs), 3) if all(l["length_m"] for l in legs) else None)
    cpos, _R = cam_pose(site)
    vis_frac = None
    if out["b"]["connected"]:
        seg_len, seg_vis = 0.0, 0.0
        for leg in out["b"]["legs"][:2]:            # pad -> 1 -> 2 with the R1 tunnel blocked
            pa = np.asarray(leg["path"])
            mid = 0.5 * (pa[:-1] + pa[1:])
            sl = np.linalg.norm(np.diff(pa, axis=0), axis=1)
            vv = visible(m, d, cpos, np.c_[mid, np.full(len(mid), 0.20)])
            seg_len += float(sl.sum())
            seg_vis += float(sl[vv].sum())
        vis_frac = seg_vis / seg_len if seg_len > 0 else 1.0
    out["d"] = dict(visible_fraction_pad_1_2_tunnel_blocked=None if vis_frac is None else round(vis_frac, 4), threshold=0.95)
    out["radius_m"] = r
    out["pass"] = bool(out["a"]["connected"] and out["b"]["connected"] and out["c"]["connected"]
                       and vis_frac is not None and vis_frac >= 0.95)
    return out


# ------------------------------------------------------------------------------------------------ G3
def qc_marker_positions(site):
    lx, ly = site["size_m"]
    floor = [(x, y, 0.001) for y in (0.35, ly / 2, ly - 0.35) for x in (0.40, lx / 2, lx - 0.40)]
    high = [(lx / 2 + sx * 1.2, ly / 2 + sy * 0.8, 0.20) for sy in (-1, 1) for sx in (-1, 1)]
    return floor + high


def g3_projection(site, flip_sign=True):
    """Segmentation render of the QC copy (markers only), centroid vs projection with the given K and the true
    extrinsics, pixel centres at integer coordinates. Also the flipped principalpixel sign for the record."""
    import copy
    import mujoco
    marks = qc_marker_positions(site)
    tops = np.array([(x, y, z + 0.001) for x, y, z in marks])
    pos, R = cam_pose(site)
    u, v, _ = fam.project(pos, R, tops, site["camera"]["model"])
    res = {}
    variants = [("as_built", site)]
    if flip_sign:
        s2 = copy.deepcopy(site)
        pp = s2["camera"]["model"]["principalpixel_mujoco"]
        s2["camera"]["model"]["principalpixel_mujoco"] = [-pp[0], -pp[1]]
        variants.append(("flipped_sign", s2))
    for name, s in variants:
        xml = mjcf.site_xml(s, qc_markers=marks)
        m = mujoco.MjModel.from_xml_string(xml)
        d = mujoco.MjData(m)
        mujoco.mj_forward(m, d)
        cm = s["camera"]["model"]
        r = mujoco.Renderer(m, cm["height"], cm["width"])
        opt = mujoco.MjvOption()
        opt.geomgroup[:] = [0, 0, 0, 0, 0, 1]
        r.enable_segmentation_rendering()
        r.update_scene(d, camera="search", scene_option=opt)
        seg = r.render().copy()
        r.close()
        ids = _geom_ids(m, "qcmark__")
        errs, npx = [], []
        for k, gid in enumerate(ids):
            ys, xs = np.nonzero(seg[..., 0] == gid)
            npx.append(int(len(xs)))
            if len(xs) == 0:
                errs.append(None)
                continue
            errs.append([float(xs.mean() - u[k]), float(ys.mean() - v[k])])
        e = np.array([q for q in errs if q is not None])
        res[name] = dict(principalpixel=s["camera"]["model"]["principalpixel_mujoco"],
                         max_px=round(float(np.hypot(e[:, 0], e[:, 1]).max()), 3) if len(e) else None,
                         mean_du_dv_px=[round(float(q), 3) for q in e.mean(0)] if len(e) else None,
                         markers=len(ids), markers_seen=int(sum(q > 0 for q in npx)), min_marker_px=int(min(npx)))
    ok = res["as_built"]["max_px"] is not None and res["as_built"]["markers_seen"] == len(marks) >= 13
    return dict(pass_=bool(ok and res["as_built"]["max_px"] <= 1.0), threshold_px=1.0, n_markers=len(marks),
                pixel_convention=site["camera"]["model"]["pixel_convention"], **res)


# ------------------------------------------------------------------------------------------------ all of G
def evaluate(site, xml, render=True, g9=None):
    import mujoco
    q = {}
    lx, ly = site["size_m"]
    env = site["envelope"]["values"]
    # G1
    try:
        m, d = compile_world(xml)
        q["G1"] = dict(pass_=bool(site["params"]["slack_R1_m"] >= 0 and site["params"]["slack_R2_m"] >= 0),
                       compiles=True, ngeom=int(m.ngeom), slack_R1_m=site["params"]["slack_R1_m"],
                       slack_R2_m=site["params"]["slack_R2_m"], mujoco=mujoco.__version__)
    except Exception as exc:          # noqa: BLE001  (the predicate records the failure)
        q["G1"] = dict(pass_=False, compiles=False, error=str(exc)[:300])
        q["pass"] = False
        return q
    cpos, R = cam_pose(site)
    cm = site["camera"]["model"]
    # G2
    margin, _u, _v = fam.corner_margin(cpos, R, lx, ly, cm)
    q["G2"] = dict(pass_=bool(margin >= 12.0), corner_margin_px=round(margin, 2), threshold_px=12.0)
    # G3
    if render:
        try:
            q["G3"] = g3_projection(site)
        except Exception as exc:      # noqa: BLE001
            q["G3"] = dict(pass_=False, error=str(exc)[:300])
    else:
        q["G3"] = dict(pass_=None, note="not rendered")
    # grids
    G = gridmod.Grid(site, 0.04)
    solids = gridmod.solid_footprints(site)
    roofs = [tuple(b["aabb"][:4]) for b in site["boxes"] if b["kind"] == "roof"]
    in_solid = G.inside_any(solids)
    under_roof = G.inside_any(roofs)
    openf = (~in_solid) & (~under_roof)
    P20 = np.c_[G.X.ravel(), G.Y.ravel(), np.full(G.X.size, 0.20)]
    vis20 = visible(m, d, cpos, P20).reshape(G.X.shape)
    # G4
    frac = float((vis20 & openf).sum() / max(1, openf.sum()))
    q["G4"] = dict(pass_=bool(frac >= 0.75), visible_fraction_open_floor_0p20=round(frac, 4), threshold=0.75)
    # G5
    G2_ = gridmod.Grid(site, 0.02)
    in_solid2 = G2_.inside_any(solids)
    g5 = {}
    ok5 = True
    pts = _points(site)
    for key in ("pad", 1, 2):
        cx, cy = pts[key]
        disc = (np.hypot(G2_.X - cx, G2_.Y - cy) <= 0.30) & ~in_solid2
        row = {}
        for h in (0.05, 0.20):
            P = np.c_[G2_.X[disc], G2_.Y[disc], np.full(int(disc.sum()), h)]
            fr = float(visible(m, d, cpos, P).mean())
            row["h%.2f" % h] = round(fr, 4)
            ok5 &= fr >= 0.90
        g5[str(key)] = row
    gal = site["void"]["gallery"]
    sp3 = pts[3]
    hid = {}
    for name, mask in (("sp3_disc", np.hypot(G2_.X - sp3[0], G2_.Y - sp3[1]) <= 0.30),
                       ("gallery", (G2_.X >= gal["x0"]) & (G2_.X <= gal["x1"]) & (G2_.Y >= gal["y0"]) & (G2_.Y <= gal["y1"]))):
        mask = mask & ~in_solid2
        row = {}
        for h in (0.05, 0.20, 0.30):
            P = np.c_[G2_.X[mask], G2_.Y[mask], np.full(int(mask.sum()), h)]
            fr = float(visible(m, d, cpos, P).mean()) if len(P) else 0.0
            row["h%.2f" % h] = round(fr, 5)
            ok5 &= fr == 0.0
        hid[name] = row
    q["G5"] = dict(pass_=bool(ok5), visible_near_pad_sp1_sp2=g5, visible_sp3_and_gallery=hid,
                   thresholds=">= 0.90 within 0.30 m of pad, SP1, SP2 at 0.05 and 0.20 m; = 0 for SP3 and the gallery")
    # G6
    far = np.array([[lx / 2, ly - 0.3, 0.10], [0.3, ly - 0.3, 0.10], [lx - 0.3, ly - 0.3, 0.10]])
    _uu, _vv, zc = fam.project(cpos, R, far, cm)
    mmpx = [float(z) / cm["fx"] * 1000 for z in zc]
    q["G6"] = dict(pass_=bool(mmpx[0] <= 10.0), mm_per_px_far_centre=round(mmpx[0], 3),
                   mm_per_px_far_corners=[round(mmpx[1], 3), round(mmpx[2], 3)], threshold=10.0,
                   definition="depth along the optical axis / fx at (LX/2, LY - 0.3, 0.10)")
    # G7
    M, Bm = _opening(site, "M"), _opening(site, "Bm")
    slab = site["void"]["slab"]
    sup_h = {}
    for b in site["boxes"]:
        if b["name"] in ("support__sup_w_M", "support__sup_e_M", "support__sup_w_Bm", "support__sup_e_Bm"):
            sup_h[b["name"]] = b["aabb"][5]
    same = dict(width=abs((M["x1"] - M["x0"]) - (Bm["x1"] - Bm["x0"])) < 1e-6,
                clear_h=abs(M["clear_h"] - Bm["clear_h"]) < 1e-6,
                support_h=len(set(sup_h.values())) == 1,
                slab_covers_both=bool(slab["x0"] <= min(M["foot"][0], Bm["foot"][0]) + 1e-9
                                      and slab["x1"] >= max(M["foot"][1], Bm["foot"][1]) - 1e-9),
                slab_top_and_thickness="one slab over both mouths")
    gid, dist, hit, shape = cast_image(m, d, site, ss=2)
    plug_ids = set(_geom_ids(m, "plug__"))
    floor_ids = set(_geom_ids(m, "floor")) | set(_geom_ids(m, "patch__"))
    n_plug = int(np.isin(gid, list(plug_ids)).sum())
    isfloor = np.isin(gid, list(floor_ids))
    depth_in = {}
    for g in (M, Bm):
        inside = isfloor & (hit[:, 0] > g["x0"]) & (hit[:, 0] < g["x1"]) & (hit[:, 1] > g["y0"]) & (hit[:, 1] < g["y1"])
        depth_in[g["id"]] = dict(visible_floor_depth_m=round(float((hit[inside, 1] - g["y0"]).max()), 4) if inside.any() else 0.0,
                                 visible_floor_rays=int(inside.sum()))
    ok7 = all(v_ for k, v_ in same.items() if isinstance(v_, bool)) and n_plug == 0 \
        and all(v_["visible_floor_depth_m"] <= 0.20 for v_ in depth_in.values())
    q["G7"] = dict(pass_=bool(ok7), same_geometry=same, plug_rays_hit=n_plug, rays=int(len(gid)), supersample=2,
                   mouth_floor=depth_in, threshold_visible_depth_m=0.20)
    # G9 (computed before the G8 route parts)
    g9 = g9 or g9_routes(site, md=(m, d))
    q["G9"] = dict(pass_=g9["pass"], radius_m=g9["radius_m"],
                   a=dict(connected=g9["a"]["connected"], total_m=g9["a"]["total_m"],
                          legs=[{k: l[k] for k in ("a", "b", "length_m")} for l in g9["a"]["legs"]]),
                   b_tunnel_blocked=dict(connected=g9["b"]["connected"], total_m=g9["b"]["total_m"]),
                   c_gap_blocked=dict(connected=g9["c"]["connected"], total_m=g9["c"]["total_m"]),
                   d=g9["d"])
    # G8a
    vd = site["void"]
    y2, yg = site["bands"]["R2_front"]
    gym = 0.5 * (gal["y0"] + gal["y1"])
    cM = 0.5 * (M["x0"] + M["x1"])
    cB = 0.5 * (Bm["x0"] + Bm["x1"])
    X = _opening(site, "X")
    cX = 0.5 * (X["x0"] + X["x1"])
    stops = {"true mouth, inner end": [(cM, yg)]}
    for ds in env["D_STOP"]:
        stops.setdefault("true mouth axis, stopped by the north wall", []).append((cM, ly - ds))
        stops.setdefault("blind mouth axis, stopped by the plug (= its inner end)", []).append((cB, vd["plug"]["y0"] - ds))
        stops.setdefault("decoy, stopped at its front", []).append((cX, y2 - ds))
        stops.setdefault("gallery axis, stopped at the west end", []).append((gal["x0"] + ds, gym))
        stops.setdefault("gallery axis, stopped at the east end", []).append((gal["x1"] - ds, gym))
    cr = {k: round(max(credit(math.dist(p, sp3)) for p in v_), 4) for k, v_ in stops.items()}
    free2 = ~in_solid2
    P2 = np.c_[G2_.X[free2], G2_.Y[free2], np.full(int(free2.sum()), 0.20)]
    vis2 = visible(m, d, cpos, P2)
    dd = np.hypot(P2[vis2, 0] - sp3[0], P2[vis2, 1] - sp3[1])
    cr["any visible pose (centre where a point 0.20 m above it is seen)"] = round(credit(float(dd.min())), 4) if len(dd) else 0.0
    q["G8a"] = dict(pass_=bool(max(cr.values()) <= 0.30), credit=cr, max_credit=max(cr.values()), threshold=0.30,
                    D_STOP_m=env["D_STOP"])
    # G8b: from the last visible pose on the G9a leg SP2 -> SP3
    leg = g9["a"]["legs"][2]
    if leg["path"]:
        pa = np.asarray(leg["path"])
        vv = visible(m, d, cpos, np.c_[pa, np.full(len(pa), 0.20)])
        last = int(np.nonzero(vv)[0].max()) if vv.any() else 0
        blind = gridmod.path_length(pa[last:])
        q["G8b"] = dict(pass_=bool(1.4 <= blind <= 2.6), blind_path_m=round(blind, 3), window_m=[1.4, 2.6],
                        last_visible_xy=[round(float(pa[last, 0]), 3), round(float(pa[last, 1]), 3)])
    else:
        q["G8b"] = dict(pass_=False, blind_path_m=None, window_m=[1.4, 2.6])
    # G8c (set level; per site the flag)
    q["G8c"] = dict(pass_=None, nearest_P_mouth_is_true=vd["nearest_P_mouth_is_true"], sp3_mode=vd["sp3_mode"],
                    note="set level: 4-6 of the 10 hidden sites; the mode stratum 5 / 5 gives it")
    # G10
    dpad = {str(k): round(math.dist(pts["pad"], pts[k]), 3) for k in (1, 2, 3)}
    dsp = {"1-2": round(math.dist(pts[1], pts[2]), 3), "1-3": round(math.dist(pts[1], pts[3]), 3),
           "2-3": round(math.dist(pts[2], pts[3]), 3)}
    q["G10"] = dict(pass_=bool(min(dpad.values()) >= 1.2 and min(dsp.values()) >= 1.0), pad_to_sp=dpad, sp_pairs=dsp)
    # G11
    q["G11"] = g11(site, m, g9)
    keys = ["G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8a", "G8b", "G9", "G10", "G11"]
    if not render:
        keys.remove("G3")
    q["failed"] = [k for k in keys if not q[k]["pass_"]]
    q["pass"] = not q["failed"]
    return q


def g11(site, m, g9):
    import mujoco
    lx, ly = site["size_m"]
    clr = site["classes"]["free_block_clearance"]
    free = [b for b in site["boxes"] if b.get("free_block")]
    walls = [(-fam.WALL_T, 0.0, -fam.WALL_T, ly + fam.WALL_T), (lx, lx + fam.WALL_T, -fam.WALL_T, ly + fam.WALL_T),
             (-fam.WALL_T, lx + fam.WALL_T, -fam.WALL_T, 0.0), (-fam.WALL_T, lx + fam.WALL_T, ly, ly + fam.WALL_T)]
    solids = walls + [tuple(b["aabb"][:4]) for b in site["boxes"] if b["kind"] in ("rubble", "support", "plug")
                      and not b.get("free_block")]
    bad = []
    nA = sum(1 for b in free if b["name"].startswith("crate__free_A") or b["name"].startswith("rubble__free_A"))
    nB = len(free) - nA
    for i, b in enumerate(free):
        others = solids + [tuple(o["aabb"][:4]) for j, o in enumerate(free) if j != i]
        if not fam.spacing_ok(tuple(b["aabb"][:4]), others, clr):
            bad.append(b["name"])
    chips = site.get("chips", [])
    pts = [site["pad"]["xy"]] + [s["xy"] for s in site["search_points"]]
    route = []
    for leg in g9["a"]["legs"]:
        if leg["path"]:
            route += leg["path"]
    route = np.asarray(route) if route else np.zeros((0, 2))
    near = 0
    chip_bad = []
    for c in chips:
        half = c["size_m"] / 2 * (1.0 if c["shape"] == "disc" else math.sqrt(2.0))
        if not 0.03 - 1e-9 <= c["size_m"] <= 0.06 + 1e-9:
            chip_bad.append((c["name"], "size"))
        if any(math.dist(c["xy"], p) < 0.15 + half - 1e-9 for p in pts):
            chip_bad.append((c["name"], "near a point"))
        if not 0.9 - 1e-9 <= c["lightness"] <= 1.1 + 1e-9:
            chip_bad.append((c["name"], "lightness"))
        if len(route) and float(np.min(np.hypot(route[:, 0] - c["xy"][0], route[:, 1] - c["xy"][1]))) <= 0.4:
            near += 1
        gidc = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, c["name"])
        if gidc < 0 or m.geom_contype[gidc] != 0 or m.geom_conaffinity[gidc] != 0:
            chip_bad.append((c["name"], "collides"))
    ok = (not bad) and nA <= 3 and nB <= 2 and 8 <= len(chips) <= 14 and not chip_bad and near >= 3
    return dict(pass_=bool(ok), free_blocks=dict(A=nA, B=nB, spacing_violations=bad, clearance_m=clr),
                chips=dict(n=len(chips), within_0p4_of_route=near, violations=chip_bad))
