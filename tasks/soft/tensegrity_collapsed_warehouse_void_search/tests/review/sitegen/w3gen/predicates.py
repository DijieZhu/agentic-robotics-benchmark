"""Predicates of W3 (version 1.1): G1-G11 rewritten for three mouths and the deep void, G12-G16 for the void's
shape, G17-G19 for the rubble (no robot dynamics).

Visibility = MuJoCo mj_multiRay from the search camera's TRUE position against geom group 0 (W2's helpers in
w2gen.predicates). Routes = w2gen.grid (0.02 m grid, disc radius W_SWEEP / 2 + 0.04). G3 renders a QC copy (OSMesa
on this host). Every predicate is judged on the site with its rubble. With the bundle of build_sites():
  G7c  the two other counterfactual sites (each with ITS OWN rubble) give, ray for ray, the same hit geometry and
       hit position for every 2 x 2 supersampled camera ray, the same public geometry (w2art's redaction of the
       boxes those rays hit, the ground patches, the chips, the lighting, the camera) and the same mission() fields;
       optionally (cf_render) the same classic-renderer RGB and depth frame of the world;
  G9, G17-G19 are judged on all three counterfactual sites (mouth tau read as the true mouth on site tau); any
       failure rejects the seed, so admission does not depend on which mouth is the true one.
G17-G19 are recomputed from the package's boxes and void layout; they never read the record void.rubble.
"""
import copy
import hashlib
import json
import math
import os
import sys

import numpy as np

from w2gen import grid as gridmod
from w2gen import predicates as w2p

from . import family as fam
from . import mjcf

FULL, ZERO = fam.CREDIT_RADIUS
TOL = 1e-6 + 1e-9            # package coordinates are rounded to 1e-6 m
DEBRIS = "rubble__void_debris"
HERE = os.path.dirname(os.path.abspath(__file__))
ART_DIR = os.path.normpath(os.path.join(HERE, "..", "..", "art"))


def credit(d):
    return float(np.clip(1.0 - (d - FULL) / (ZERO - FULL), 0.0, 1.0))


def _void_rects(site):
    v = site["void"]
    out = [("C", v["chamber"])]
    for k, r in v["runs"].items():
        out.append((k, r))
    for p in v["pockets"]:
        out.append((p["name"], dict(x0=p["rect"][0], x1=p["rect"][1], y0=p["rect"][2], y1=p["rect"][3])))
    return out


def _in_rect(X, Y, r):
    return (X >= r["x0"]) & (X <= r["x1"]) & (Y >= r["y0"]) & (Y <= r["y1"])


def _overlap(a, b, eps=1e-9):
    """Open overlap of two rectangles (x0, x1, y0, y1)."""
    return min(a[1], b[1]) - max(a[0], b[0]) > eps and min(a[3], b[3]) - max(a[2], b[2]) > eps


def _rect_pt_dist(r, p):
    return math.hypot(max(r[0] - p[0], 0.0, p[0] - r[1]), max(r[2] - p[1], 0.0, p[1] - r[3]))


def g3_projection(site):
    """W2's G3 with the QC markers in the camera rectangle [0, LX] x [0, LY_CAM]."""
    s = copy.deepcopy(site)
    lx = s["size_m"][0]
    orig = w2p.qc_marker_positions
    w2p.qc_marker_positions = lambda _s: orig(dict(size_m=[lx, fam.LY_CAM]))
    try:
        return w2p.g3_projection(s)
    finally:
        w2p.qc_marker_positions = orig


# ------------------------------------------------------------------------------------------------ camera-side views
def ray_view(site, xml, md=None, ss=2):
    """(names of the hit geoms, hit positions rounded to 1e-6, set of hit names) of every ss x ss camera ray."""
    import mujoco
    m, d = md if md else w2p.compile_world(xml)
    gid, _dist, hit, _shape = w2p.cast_image(m, d, site, ss=ss)
    names = np.array([(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, int(g)) or "") for g in range(m.ngeom)] + [""])
    nm = names[np.where(gid >= 0, gid, m.ngeom)]
    return nm, np.round(hit, 6), set(nm.tolist()) - {""}


def mission_view(site):
    """Every field mission() takes from a site package (the cell's mission.build reads exactly these)."""
    keys = dict(size_m=site["size_m"], pad=site["pad"]["xy"], search_points=site["search_points"],
                camera_model={k: site["camera"]["model"][k] for k in ("width", "height", "fx", "fy", "cx", "cy",
                                                                      "depth_unit_m", "depth_range_m", "rate_hz")},
                camera_given=site["camera"]["given"]["T_world_camera"])
    return json.dumps(keys, sort_keys=True)


def public_view(site, hit_names):
    """w2art's redaction (the art tools' only reader of site.json) of the site restricted to the boxes the camera
    rays hit: kinds and boxes, ground patches, chips, lighting, camera, site rectangle; canonical JSON text."""
    if ART_DIR not in sys.path:
        sys.path.insert(0, ART_DIR)
    from w2art import redact
    sub = dict(site)
    sub["boxes"] = [b for b in site["boxes"] if b["name"] in hit_names]
    return json.dumps(redact.public_geometry(sub), sort_keys=True, separators=(",", ":"))


def render_world(site, xml):
    """sha256 of the classic renderer's RGB and depth frame of the world (no robot) from the site's search camera
    with the site's light (the MJCF's own camera `search` and key light), and the frame's shape."""
    import mujoco
    m, d = w2p.compile_world(xml)
    cm = site["camera"]["model"]
    r = mujoco.Renderer(m, cm["height"], cm["width"])
    try:
        r.update_scene(d, camera="search")
        rgb = r.render().copy()
        r.enable_depth_rendering()
        r.update_scene(d, camera="search")
        depth = r.render().copy()
        r.disable_depth_rendering()
    finally:
        r.close()
    return dict(rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),
                depth_sha256=hashlib.sha256(depth.astype("<f4").tobytes()).hexdigest(),
                shape=list(rgb.shape), depth_range=[round(float(depth.min()), 4), round(float(depth.max()), 4)])


# ------------------------------------------------------------------------------------------------ rubble predicates
def void_frame(site):
    """The layout facts G17-G19 need, from the package (runs, chamber, pockets, the true mouth of this site)."""
    v = site["void"]
    runs = v["runs"]
    tm = [o for o in site["r2"]["openings"] if o["id"] == v["true_mouth"]][0]
    m0, m1, y2, yg = tm["x0"], tm["x1"], tm["y0"], tm["y1"]
    g1, n_ = runs["G1"], runs["N"]
    nx0, nx1, yc0 = n_["x0"], n_["x1"], n_["y1"]
    xN = 0.5 * (nx0 + nx1)
    east = xN > 0.5 * (m0 + m1)
    ch = v["chamber"]
    corners = [((nx0 if east else nx1), g1["y1"]), ((nx1 if east else nx0), g1["y0"]), (nx0, yc0), (nx1, yc0)]
    squares = [("mouth", (m0, m1, g1["y0"], g1["y1"])), ("G1/N", (nx0, nx1, g1["y0"], g1["y1"]))]
    if "K" in runs:
        k_ = runs["K"]
        xc = ch["x0"] if 0.5 * (ch["x0"] + ch["x1"]) > xN else ch["x1"]
        corners.append((xc, k_["y1"]))
        squares.append(("N/K", (nx0, nx1, k_["y0"], k_["y1"])))
    pockets = [tuple(p["rect"]) for p in v["pockets"]]
    hz = [b for b in site["boxes"] if b["name"] == "roof__void_slab"][0]["aabb"][4]
    return dict(runs=runs, mouth=(m0, m1, y2, yg), corners=corners, squares=squares, pockets=pockets,
                chamber=(ch["x0"], ch["x1"], ch["y0"], ch["y1"]), hz=hz, true_mouth=tm["id"])


def rubble_predicates(site, env):
    """G17, G18, G19 of one site (its own true mouth), from the boxes and the void layout of the package."""
    vf = void_frame(site)
    lx, ly = site["size_m"]
    P = site["classes"]["P"]["w"]
    W = env["W_SWEEP"]
    sp3 = [s["xy"] for s in site["search_points"] if s["id"] == 3][0]
    deb = [b for b in site["boxes"] if b["name"].startswith(DEBRIS)]
    solids = [(b["name"], tuple(b["aabb"][:4])) for b in site["boxes"] if b["kind"] in gridmod.SOLID_KINDS]
    m0, m1, y2, yg = vf["mouth"]
    rows, ok17, ok18, ok19 = [], True, True, True
    on_route = 0
    for b in deb:
        x0, x1, y0, y1, z0, z1 = b["aabb"]
        r = dict(name=b["name"])
        run = None
        for k, rr in vf["runs"].items():
            if x0 >= rr["x0"] - TOL and x1 <= rr["x1"] + TOL and y0 >= rr["y0"] - TOL and y1 <= rr["y1"] + TOL:
                run = k
        r["run"] = run
        if run is None:
            r["G17"] = dict(pass_=False, why="not inside a run")
            ok17 = ok18 = ok19 = False
            rows.append(r)
            continue
        rr = vf["runs"][run]
        if rr["axis"] == "x":
            lo, hi, a0, a1, wlo, whi = y0, y1, x0, x1, rr["y0"], rr["y1"]
        else:
            lo, hi, a0, a1, wlo, whi = x0, x1, y0, y1, rr["x0"], rr["x1"]
        w = whi - wlo
        at_lo, at_hi = abs(lo - wlo) <= TOL, abs(hi - whi) <= TOL
        gap = (whi - hi) if at_lo else (lo - wlo)
        if rr["axis"] == "x":
            gap_rect = (x0, x1, hi, whi) if at_lo else (x0, x1, wlo, lo)
        else:
            gap_rect = (hi, whi, y0, y1) if at_lo else (wlo, lo, y0, y1)
        in_gap = [nm for nm, s in solids if nm != b["name"] and _overlap(s, gap_rect)]
        g17 = dict(against_wall=bool(at_lo != at_hi), z_floor_to_slab=bool(abs(z0) <= TOL and abs(z1 - vf["hz"]) <= TOL),
                   length_m=round(a1 - a0, 6), gap_m=round(gap, 6), gap_is_P=bool(abs(gap - P) <= TOL),
                   solids_in_gap=in_gap)
        g17["pass_"] = bool(g17["against_wall"] and g17["z_floor_to_slab"] and 0.20 - TOL <= a1 - a0 <= 0.30 + TOL
                            and g17["gap_is_P"] and not in_gap)
        across = hi - lo
        intrusion = across - (w - W) / 2.0
        g18 = dict(intrusion_into_centre_band_m=round(intrusion, 6), pass_=bool(intrusion >= 0.04 - TOL))
        rect = (x0, x1, y0, y1)
        hits = [nm for nm, sq in vf["squares"] if _overlap(rect, sq)]
        if _overlap(rect, vf["chamber"]):
            hits.append("chamber")
        hits += ["pocket" for pk in vf["pockets"] if _overlap(rect, pk)]
        d_mouth = math.hypot(max(m0 - x1, 0.0, x0 - m1), max(y0 - yg, 0.0, yg - y1))
        d_corner = min(_rect_pt_dist(rect, q) for q in vf["corners"])
        d_sp3 = _rect_pt_dist(rect, sp3)
        g19 = dict(in_squares_chamber_pockets=hits, d_mouth_inner_end_m=round(d_mouth, 6), d_corner_m=round(d_corner, 6),
                   d_sp3_m=round(d_sp3, 6))
        g19["pass_"] = bool(not hits and d_mouth >= 0.5 - TOL and d_corner >= 0.5 - TOL and d_sp3 >= fam.SP3_CLEAR - TOL)
        if run in ("G1", "K") and not hits:
            on_route += 1
        r.update(G17=g17, G18=g18, G19=g19, lo_wall=bool(at_lo), across_m=round(across, 6), run_width_m=round(w, 6))
        ok17 &= g17["pass_"]
        ok18 &= g18["pass_"]
        ok19 &= g19["pass_"]
        rows.append(r)
    # G19: two boxes of one run >= 0.8 m apart along it
    pairs = []
    for run in ("G1", "N", "K"):
        bb = [rw for rw in rows if rw.get("run") == run]
        ax = vf["runs"][run]["axis"] if run in vf["runs"] else "x"
        iv = []
        for rw in bb:
            a = [q for q in deb if q["name"] == rw["name"]][0]["aabb"]
            iv.append((a[0], a[1]) if ax == "x" else (a[2], a[3]))
        iv.sort()
        for p, q in zip(iv, iv[1:]):
            pairs.append(round(q[0] - p[1], 6))
    pair_ok = all(g >= 0.8 - TOL for g in pairs)
    ok19 &= pair_ok
    # G17 (f): the grid route through the gaps exists (start = the true mouth's centre at y2 + 0.10)
    solid_rects = [s for _nm, s in solids]
    route_m, _path, _d = fam._route(solid_rects, lx, ly, (0.5 * (m0 + m1), y2 + 0.10), tuple(sp3),
                                    site["classes"]["plan_radius"], y2 - 0.30)
    ok17 &= route_m is not None
    ok18 &= on_route >= 1
    return dict(n_boxes=len(deb), on_route=on_route, route_with_rubble_m=None if route_m is None else round(route_m, 3),
                pair_clear_m=pairs, boxes=rows, true_mouth=vf["true_mouth"],
                G17=bool(ok17), G18=bool(ok18), G19=bool(ok19))


# ------------------------------------------------------------------------------------------------ all of G
KEYS_G1_G16 = ["G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8a", "G8b", "G9", "G10", "G11", "G12", "G13", "G14", "G15",
               "G16"]


def evaluate(site, xml, render=True, g9=None, bundle=None, cf_render=False):
    """qc dict of the site; with the bundle of build_sites(): G7c, and G9 / G17-G19 on all three counterfactual sites.
    `pass` = G1-G16 on the site, G9 and G17-G19 on all three counterfactual sites."""
    import mujoco
    q = {}
    lx, ly = site["size_m"]
    env = site["envelope"]["values"]
    try:
        m, d = w2p.compile_world(xml)
        q["G1"] = dict(pass_=bool(site["params"]["slack_R1_m"] >= 0 and site["params"]["slack_R2_m"] >= 0), compiles=True,
                       ngeom=int(m.ngeom), slack_R1_m=site["params"]["slack_R1_m"], slack_front_m=site["params"]["slack_R2_m"],
                       mujoco=mujoco.__version__)
    except Exception as exc:  # noqa: BLE001
        q["G1"] = dict(pass_=False, compiles=False, error=str(exc)[:300])
        q["failed"], q["pass"] = ["G1"], False
        return q
    cpos, R = w2p.cam_pose(site)
    cm = site["camera"]["model"]
    margin, _u, _v = fam.corner_margin(cpos, R, lx, fam.LY_CAM, cm)
    q["G2"] = dict(pass_=bool(margin >= 12.0), corner_margin_px=round(margin, 2), threshold_px=12.0,
                   rectangle="[0, LX] x [0, LY_CAM]")
    if render:
        try:
            q["G3"] = g3_projection(site)
        except Exception as exc:  # noqa: BLE001
            q["G3"] = dict(pass_=False, error=str(exc)[:300])
    else:
        q["G3"] = dict(pass_=None, note="not rendered")
    G = gridmod.Grid(site, 0.04)
    solids = gridmod.solid_footprints(site)
    roofs = [tuple(b["aabb"][:4]) for b in site["boxes"] if b["kind"] == "roof"]
    in_solid = G.inside_any(solids)
    under_roof = G.inside_any(roofs)
    openf = (~in_solid) & (~under_roof)
    P20 = np.c_[G.X.ravel(), G.Y.ravel(), np.full(G.X.size, 0.20)]
    vis20 = w2p.visible(m, d, cpos, P20).reshape(G.X.shape)
    frac = float((vis20 & openf).sum() / max(1, openf.sum()))
    q["G4"] = dict(pass_=bool(frac >= 0.75), visible_fraction_open_floor_0p20=round(frac, 4), threshold=0.75)
    G2_ = gridmod.Grid(site, 0.02)
    in_solid2 = G2_.inside_any(solids)
    g5, ok5 = {}, True
    pts = w2p._points(site)
    for key in ("pad", 1, 2):
        cx, cy = pts[key]
        disc = (np.hypot(G2_.X - cx, G2_.Y - cy) <= 0.30) & ~in_solid2
        row = {}
        for h in (0.05, 0.20):
            P = np.c_[G2_.X[disc], G2_.Y[disc], np.full(int(disc.sum()), h)]
            fr = float(w2p.visible(m, d, cpos, P).mean())
            row["h%.2f" % h] = round(fr, 4)
            ok5 &= fr >= 0.90
        g5[str(key)] = row
    sp3 = pts[3]
    hid = {}
    void_mask = np.zeros(G2_.X.shape, bool)
    for _n, r in _void_rects(site):
        void_mask |= _in_rect(G2_.X, G2_.Y, r)
    for name, mask in (("sp3_disc", np.hypot(G2_.X - sp3[0], G2_.Y - sp3[1]) <= 0.30), ("void", void_mask)):
        mask = mask & ~in_solid2
        row = {}
        for h in (0.05, 0.20, 0.30):
            P = np.c_[G2_.X[mask], G2_.Y[mask], np.full(int(mask.sum()), h)]
            fr = float(w2p.visible(m, d, cpos, P).mean()) if len(P) else 0.0
            row["h%.2f" % h] = round(fr, 5)
            ok5 &= fr == 0.0
        hid[name] = row
    q["G5"] = dict(pass_=bool(ok5), visible_near_pad_sp1_sp2=g5, visible_sp3_and_void=hid)
    far = np.array([[lx / 2, fam.LY_CAM - 0.3, 0.10], [0.3, fam.LY_CAM - 0.3, 0.10], [lx - 0.3, fam.LY_CAM - 0.3, 0.10]])
    _uu, _vv, zc = fam.project(cpos, R, far, cm)
    mmpx = [float(z) / cm["fx"] * 1000 for z in zc]
    q["G6"] = dict(pass_=bool(mmpx[0] <= 10.0), mm_per_px_far_centre=round(mmpx[0], 3), threshold=10.0)
    # G7: three mouths alike, no ray on a plug or any hidden box, <= 0.20 m of floor seen inside, counterfactuals
    mouths = site["r2"]["openings"]
    sup = {}
    for b in site["boxes"]:
        if b["name"].startswith("support__sup_"):
            sup[b["name"]] = (round(b["aabb"][1] - b["aabb"][0], 6), round(b["aabb"][3] - b["aabb"][2], 6), b["aabb"][5])
    widths = {round(o["x1"] - o["x0"], 6) for o in mouths}
    same = dict(width=len(widths) == 1, clear_h=len({o["clear_h"] for o in mouths}) == 1,
                supports=len(set(v for k, v in sup.items() if "_m" in k)) == 1,
                one_slab=sum(1 for b in site["boxes"] if b["kind"] == "roof" and "void_slab" in b["name"]) == 1)
    gid, _dist, hit, _shape = w2p.cast_image(m, d, site, ss=2)
    plug_ids = set(w2p._geom_ids(m, "plug__"))
    floor_ids = set(w2p._geom_ids(m, "floor")) | set(w2p._geom_ids(m, "patch__"))
    n_plug = int(np.isin(gid, list(plug_ids)).sum())
    hidden_ids = set(w2p._geom_ids(m, "support__void_fill")) | plug_ids | set(w2p._geom_ids(m, DEBRIS))
    n_hidden = int(np.isin(gid, list(hidden_ids)).sum())
    n_debris = int(np.isin(gid, list(w2p._geom_ids(m, DEBRIS))).sum())
    isfloor = np.isin(gid, list(floor_ids))
    depth_in = {}
    for g in mouths:
        inside = isfloor & (hit[:, 0] > g["x0"]) & (hit[:, 0] < g["x1"]) & (hit[:, 1] > g["y0"]) & (hit[:, 1] < g["y1"])
        depth_in[g["id"]] = round(float((hit[inside, 1] - g["y0"]).max()), 4) if inside.any() else 0.0
    cf = None
    if bundle is not None:
        ts = bundle["tau_star"]
        nm0, hp0, names0 = ray_view(site, xml, md=(m, d))
        mv0 = mission_view(site)
        pv0 = public_view(site, names0)
        rw0 = render_world(site, xml) if cf_render else None
        cf = dict(tau=[], rays_identical=[], hit_names_identical=[], mission_identical=[], public_identical=[],
                  n_debris=[], render_identical=[] if cf_render else None)
        for t in range(3):
            if t == ts:
                continue
            s, x = bundle["sites"][t], bundle["xmls"][t]
            nm, hp, names = ray_view(s, x)
            cf["tau"].append(t)
            cf["rays_identical"].append(bool((nm == nm0).all() and np.array_equal(hp, hp0)))
            cf["hit_names_identical"].append(names == names0)
            cf["mission_identical"].append(mission_view(s) == mv0)
            cf["public_identical"].append(public_view(s, names) == pv0)
            cf["n_debris"].append(sum(1 for b in s["boxes"] if b["name"].startswith(DEBRIS)))
            if cf_render:
                rw = render_world(s, x)
                cf["render_identical"].append(rw["rgb_sha256"] == rw0["rgb_sha256"] and rw["depth_sha256"] == rw0["depth_sha256"])
        if cf_render:
            cf["render"] = rw0
        cf["n_debris_true"] = sum(1 for b in site["boxes"] if b["name"].startswith(DEBRIS))
    ok7 = all(same.values()) and n_plug == 0 and n_hidden == 0 and all(v_ <= 0.20 for v_ in depth_in.values())
    if cf is not None:
        ok7 = ok7 and all(cf["rays_identical"]) and all(cf["hit_names_identical"]) and all(cf["mission_identical"]) \
            and all(cf["public_identical"]) and (not cf_render or all(cf["render_identical"]))
    q["G7"] = dict(pass_=bool(ok7), same_geometry=same, plug_rays_hit=n_plug, hidden_box_rays_hit=n_hidden,
                   debris_rays_hit=n_debris, mouth_floor_depth_m=depth_in, G7c=cf)
    g9 = g9 or w2p.g9_routes(site, md=(m, d))
    q["G9"] = dict(pass_=g9["pass"], a=dict(connected=g9["a"]["connected"], total_m=g9["a"]["total_m"]),
                   b_tunnel_blocked=g9["b"]["connected"], c_gap_blocked=g9["c"]["connected"], d=g9["d"])
    # G8a: point 3 to every solid (rubble included) and every site edge >= SP3_CLEAR; no visible pose within the zero
    # radius; the credit of a pose against any solid or wall (centre within the largest stop distance) is 0
    solids_all = [tuple(b["aabb"][:4]) for b in site["boxes"] if b["kind"] in gridmod.SOLID_KINDS]
    clr = fam._clearance_at(solids_all, sp3, lx, ly)
    free2 = ~in_solid2
    P2 = np.c_[G2_.X[free2], G2_.Y[free2], np.full(int(free2.sum()), 0.20)]
    vis2 = w2p.visible(m, d, cpos, P2)
    dd = np.hypot(P2[vis2, 0] - sp3[0], P2[vis2, 1] - sp3[1])
    dvis = float(dd.min()) if len(dd) else 99.0
    touch = credit(clr - max(env["D_STOP"]))
    q["G8a"] = dict(pass_=bool(clr >= fam.SP3_CLEAR - 1e-9 and dvis >= ZERO and touch == 0.0 and credit(dvis) == 0.0),
                    sp3_clearance_m=round(clr, 4), nearest_visible_pose_m=round(dvis, 3), max_credit_touching=round(touch, 4),
                    max_credit_visible=round(credit(dvis), 4), credit_radius_m=[FULL, ZERO])
    # G8b: the generator's blind route (corridor, without rubble; start (true mouth centre, y2 + 0.10)); the route
    # through the rubble gaps is G17 (f); W2's way for the record
    v = site["void"]
    blind = v["blind_route_m"]
    leg = g9["a"]["legs"][2]
    w2way = None
    if leg["path"]:
        pa = np.asarray(leg["path"])
        vv = w2p.visible(m, d, cpos, np.c_[pa, np.full(len(pa), 0.20)])
        last = int(np.nonzero(vv)[0].max()) if vv.any() else 0
        w2way = dict(blind_m=round(gridmod.path_length(pa[last:]), 3), last_visible_xy=[round(float(pa[last, 0]), 3), round(float(pa[last, 1]), 3)])
    q["G8b"] = dict(pass_=bool(fam.ROUTE_WIN[0] <= blind <= fam.ROUTE_WIN[1]), blind_route_m=blind, window_m=list(fam.ROUTE_WIN),
                    w2_style=w2way)
    dpad = {str(k): round(math.dist(pts["pad"], pts[k]), 3) for k in (1, 2, 3)}
    dsp = {"1-2": round(math.dist(pts[1], pts[2]), 3), "1-3": round(math.dist(pts[1], pts[3]), 3), "2-3": round(math.dist(pts[2], pts[3]), 3)}
    q["G10"] = dict(pass_=bool(min(dpad.values()) >= 1.2 and min(dsp.values()) >= 1.0), pad_to_sp=dpad, sp_pairs=dsp)
    q["G11"] = w2p.g11(site, m, g9)
    # G12: >= 2 right-angle bends, every straight >= 0.45 m (the route's runs in order)
    runs = v["runs"]
    tm = [o for o in mouths if o["id"] == v["true_mouth"]][0]
    xM = 0.5 * (tm["x0"] + tm["x1"])
    g1, n_ = runs["G1"], runs["N"]
    yG1 = 0.5 * (g1["y0"] + g1["y1"])
    xN = 0.5 * (n_["x0"] + n_["x1"])
    legs_ = [("mouth->G1", abs(yG1 - (tm["y0"] + 0.10))), ("G1", abs(xN - xM))]
    if "K" in runs:
        k_ = runs["K"]
        yK = 0.5 * (k_["y0"] + k_["y1"])
        legs_.append(("N", abs(yK - yG1)))
        legs_.append(("K", abs(sp3[0] - xN)))
        legs_.append(("C", abs(sp3[1] - yK)))
    else:
        legs_.append(("N->C", abs(sp3[1] - yG1)))
    bends = len(legs_) - 1
    q["G12"] = dict(pass_=bool(bends >= 2 and min(L for _n, L in legs_[:-1]) >= 0.45), bends=bends,
                    straights_m={n: round(L, 3) for n, L in legs_})
    pk = v["pockets"]
    q["G13"] = dict(pass_=bool(len(pk) >= 1 and all(0.45 - 1e-6 <= p["length"] <= 0.90 + 1e-6 for p in pk)),
                    pockets=[dict(name=p["name"], run=p["run"], length=p["length"]) for p in pk])
    vlo, vhi = site["classes"]["V"]["w_range"]
    wds = {k: r["width"] for k, r in runs.items()}
    ch = v["chamber"]
    slab = [b for b in site["boxes"] if b["name"] == "roof__void_slab"][0]["aabb"]
    hz_rule = env["H_ROLL"] + 0.14
    q["G14"] = dict(pass_=bool(all(vlo - 1e-6 <= w <= vhi + 1e-6 for w in wds.values()) and ch["x1"] - ch["x0"] >= 1.26
                               and ch["y1"] - ch["y0"] >= 1.26 and v["partition_m"] >= 0.30 - 1e-9
                               and abs(slab[4] - hz_rule) <= TOL),
                    widths=wds, chamber_m=[round(ch["x1"] - ch["x0"], 3), round(ch["y1"] - ch["y0"], 3)],
                    partition_m=v["partition_m"], clear_h_m=slab[4])
    # G15: no straight shot from any mouth's inner end to point 3 (a W_SWEEP-wide swept segment meets a solid)
    blocked = {}
    hw = env["W_SWEEP"] / 2
    for o in mouths:
        a = np.array([0.5 * (o["x0"] + o["x1"]), o["y1"]])
        b = np.array(sp3)
        tt = np.linspace(0, 1, 200)[:, None]
        seg = a + tt * (b - a)
        hitp = False
        for x0, x1, y0, y1 in solids_all:
            ddx = np.maximum(np.maximum(x0 - seg[:, 0], 0.0), seg[:, 0] - x1)
            ddy = np.maximum(np.maximum(y0 - seg[:, 1], 0.0), seg[:, 1] - y1)
            if (np.hypot(ddx, ddy) < hw).any():
                hitp = True
                break
        blocked[o["id"]] = hitp
    q["G15"] = dict(pass_=bool(all(blocked.values())), blocked=blocked)
    # G16: point 3 0.64-0.72 m from the back wall and the corner's side wall; the two bands to them empty
    corner = site["stratum"]["corner"]
    d_back = ly - sp3[1]
    d_side = sp3[0] if corner == "W" else lx - sp3[0]
    bw = (env["W_SWEEP"] + 0.20) / 2
    band_back = (sp3[0] - bw, sp3[0] + bw, sp3[1], ly)
    band_side = ((0.0, sp3[0]) if corner == "W" else (sp3[0], lx)) + (sp3[1] - bw, sp3[1] + bw)
    empty = []
    for bx in (band_back, band_side):
        empty.append(not any(min(bx[1], s[1]) - max(bx[0], s[0]) > 1e-9 and min(bx[3], s[3]) - max(bx[2], s[2]) > 1e-9
                             for s in solids_all))
    q["G16"] = dict(pass_=bool(0.64 - 1e-6 <= d_back <= 0.72 + 1e-6 and 0.64 - 1e-6 <= d_side <= 0.72 + 1e-6 and all(empty)),
                    d_back_m=round(d_back, 4), d_side_m=round(d_side, 4), bands_empty=empty)
    # G17-G19 (and G9 again) on every counterfactual site
    per = {}
    if bundle is not None:
        for t in range(3):
            s = bundle["sites"][t]
            rp = rubble_predicates(s, env)
            g9t = bundle["g9s"][t]
            per[str(t)] = dict(mouth=s["void"]["true_mouth"], G9=bool(g9t["pass"]) if g9t else None, **rp)
    else:
        rp = rubble_predicates(site, env)
        per["self"] = dict(mouth=site["void"]["true_mouth"], G9=bool(g9["pass"]), **rp)
    q["G17_G19"] = dict(counterfactual=per, rule="judged on every counterfactual site, mouth tau read as the true mouth "
                                                  "on site tau; any failure rejects the seed")
    for k in ("G17", "G18", "G19"):
        q[k] = dict(pass_=bool(all(p[k] for p in per.values())))
    q["G9_all"] = dict(pass_=bool(all(p["G9"] for p in per.values())))
    # registered regularities: the dead-end pockets against a site wall; shape vs rank
    p1 = pk[0]
    p2 = [p for p in pk if p["name"] == "P2"]
    q["regularities"] = dict(shape=v["shape"], near_rank=site["stratum"]["near_rank"], g1_dir=v["g1_dir"],
                             g1_pocket_on_site_wall=bool(p1["rect"][0] <= 1e-6 or p1["rect"][1] >= lx - 1e-6),
                             k_pocket=bool(p2), k_pocket_on_site_wall=bool(p2 and (p2[0]["rect"][0] <= 1e-6 or p2[0]["rect"][1] >= lx - 1e-6)),
                             k_on_site_wall=bool("K" in runs and (runs["K"]["x0"] <= 1e-6 or runs["K"]["x1"] >= lx - 1e-6)))
    keys = list(KEYS_G1_G16)
    if not render:
        keys.remove("G3")
    failed = [k for k in keys if not q[k]["pass_"]]
    q["failed_G1_G16"] = failed
    q["pass_G1_G16"] = not failed
    failed = failed + [k for k in ("G9_all", "G17", "G18", "G19") if not q[k]["pass_"]]
    q["failed"] = failed
    q["pass"] = not failed
    return q
