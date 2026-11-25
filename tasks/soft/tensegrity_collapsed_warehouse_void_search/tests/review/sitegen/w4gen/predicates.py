"""Predicates of W4 (version 2.0): G1-G19 as in W3 with every heap's rectangle a solid, and G20-G24 for the heaps,
each of these five measured on the COMPILED model (module measure: rays at the geoms; no package array or label is
read for a measured quantity; the package's heap records say only which geom is which heap and where it claims to be).

  G1-G16   w3gen.predicates.evaluate on the site (heaps are solids: w2gen.grid.SOLID_KINDS gains "heap" at run time)
  G7c      the two other counterfactual sites give, ray for ray, the same hit geom and hit position for every 2 x 2
           camera ray, the same mission() fields, the same public geometry (W2's redaction of the boxes hit, plus the
           heaps' public records and the exact text of their collision bodies) and optionally (cf_render) the same
           classic-renderer RGB and depth frame
  G9, G17-G19  on all three counterfactual sites (w3gen.predicates.rubble_predicates; G9 = the bundle's routes)
  G20  class geometry per heap (module heapchecks): both classes: placed rectangle in the class's w / d ranges,
       compiled footprint inside the rectangle (to within the field's one-cell roll-off), compiled top in the class's
       range; class R (rubble heap): edge rule
       >= 95 % on every exposed side, core >= 0.10 m on >= 90 %; class M (mound of fill, natural toe, no rim; the edge
       rule and the core are reported, not judged): compiled depth >= 0.28 m, compiled gap to its line <= 0.05 m,
       scale inside the piece's registered interval, the piece's registered trial verdict true (the one item read
       from the registry); no slope band for either class (the slope median and upright faces are recorded)
  G21  keep-outs on the compiled occupancy (5 mm lattice, > 3 mm above the floor): pad 0.45 m, search points 1-2
       0.40 m, the ridge openings' floor points 0.40 m, nothing in a mouth's apron or on the dust patch; rectangle
       spacing 0 or >= W_SWEEP + 0.18 (the line not counted); the gap promise (< 0.15 m or >= 0.60 m) against every
       entity standing on the floor (measure.gap_promise, the heap's line run as one entity); no colour chip inside a
       heap's rectangle
  G22  each heap: the search camera sees >= 0.70 of its surface and its top (inside its image, unoccluded); no camera
       ray ends on anything under the slab (W3's hidden-box count and every ray ending inside the site north of the
       front row below the slab's underside on a non-roof geom); every heap's rectangle south of the slab's front edge
       and its compiled footprint too, except where the footprint lies inside a floor box at least as high as the heap
       (a height field rolls off over one cell beyond its last raised vertex: a heap whose surface reaches its line
       ends a few millimetres inside the box it leans on, where no ray and no robot can meet it; the largest such
       depth is recorded)
  G23  >= 3 heaps; band A >= 1; band B >= 2 with >= 1 against the slab front; each heap leans on its line (the run of
       boxes behind it covers its width, and its compiled occupancy comes within 0.03 m of the line, median per
       column <= 0.05 m)
  G24  per heap: the compiled height field has a ring of zero cells and max == size[2]; 200 vertices (vertical rays)
       against the library piece's grid under the same turn, mirror and scale <= 0.5 mm; the fit bound against the
       piece's mesh placed in the site (1 mm lattice, nearest-point distance <= 3 mm on >= the class's fit line: the
       registry's, else R 0.98 / M 0.99; max <= 8 mm); scale inside the interval the generator draws from (the
       registered one, mounds of fill >= 1.0, cut to keep the scaled rectangle inside the class ranges)
G20-G24 are judged on all three counterfactual models. G20 and G24 are measured in full on the true site's model;
on the two others the compiled heap bodies are compared array for array (height-field data, size, geom pose, type,
group, friction) and must be identical, so the same measurement holds; full=True measures them on all three.
"""
import hashlib
import json
import sys

import numpy as np

from w2gen import predicates as w2p
from w3gen import predicates as w3p

from . import family as fam
from . import heapchecks as hc
from . import measure as ms

ART_DIR = w3p.ART_DIR


def public_view(site, hit_names, xml):
    if ART_DIR not in sys.path:
        sys.path.insert(0, ART_DIR)
    from w2art import redact
    sub = dict(site)
    sub["boxes"] = [b for b in site["boxes"] if b["name"] in hit_names and b["kind"] != "heap"]
    pub = redact.public_geometry(sub)
    pub["heaps"] = [dict(cls=h["cls"], rect=h["rect"], top_m=h["top_m"], line=h["line"], yaw_deg=h["yaw_deg"],
                         mirror=h["mirror"], scale=h["scale"]) for h in site["heaps"]["heaps"]]
    pub["heap_bodies_sha256"] = hashlib.sha256("\n".join(
        ln.strip() for ln in xml.splitlines() if "<hfield " in ln or 'type="hfield"' in ln).encode()).hexdigest()
    return json.dumps(pub, sort_keys=True, separators=(",", ":"))


def _heap_arrays(m, d, gid):
    hid = int(m.geom_dataid[gid])
    adr, n = int(m.hfield_adr[hid]), int(m.hfield_nrow[hid] * m.hfield_ncol[hid])
    h = hashlib.sha256()
    for a in (m.hfield_data[adr:adr + n], m.hfield_size[hid], np.array([m.hfield_nrow[hid], m.hfield_ncol[hid]]),
              d.geom_xpos[gid], d.geom_xmat[gid], m.geom_size[gid], np.array([m.geom_type[gid], m.geom_group[gid]]),
              m.geom_friction[gid]):
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def heap_place_checks(m, d, site, cls, keep, aprons, dust, solids_rect):
    """G21, G22 (per heap part), G23 of one compiled counterfactual model."""
    heaps = site["heaps"]["heaps"]
    ids = ms.geom_ids(m, "heap__")
    cpos, Rcam = w2p.cam_pose(site)
    cam = (Rcam, site["camera"]["model"])
    occ, rows = {}, {}
    boxes = ms.floor_boxes(m, d)
    tops = ms.floor_box_tops(m, d)
    clr = cls["free_block_clearance"]
    y2 = site["bands"]["B"][1]
    lines = {}
    for h in heaps:
        gid = ids[h["name"]]
        P = ms.heap_occupancy(m, d, gid, h["rect"])
        occ[h["name"]] = P
        r = h["rect"]
        dk = [round(float(np.min(np.hypot(P[:, 0] - p[0], P[:, 1] - p[1]))), 4) if len(P) else None for p, _r in keep]
        keep_ok = all(v is not None and v >= rr - 1e-9 for v, (_p, rr) in zip(dk, keep))
        in_apron = int(sum(((P[:, 0] > a[0]) & (P[:, 0] < a[1]) & (P[:, 1] > a[2]) & (P[:, 1] < a[3])).sum()
                           for a in aprons)) if len(P) else 0
        on_dust = int(sum(((P[:, 0] > q[0]) & (P[:, 0] < q[1]) & (P[:, 1] > q[2]) & (P[:, 1] < q[3])).sum()
                          for q in dust)) if len(P) else 0
        if h["line"] == "north":
            near = [q for _n, q in solids_rect if q[2] < h["y_line"] - 1e-9]
        else:
            near = [q for _n, q in solids_rect if q[3] > h["y_line"] + 1e-9]
        others = [tuple(o["rect"]) for o in heaps if o["name"] != h["name"]]
        spacing_ok = bool(fam.w2.spacing_ok(tuple(r), near + others, clr))
        run, cover = ms.line_run(boxes, r, h["line"], h["y_line"])
        lines[h["name"]] = run
        if len(P):
            xs = np.unique(np.round(P[:, 0], 6))
            per = []
            for x in xs:
                yy = P[np.abs(P[:, 0] - x) < 1e-7, 1]
                per.append(h["y_line"] - yy.max() if h["line"] == "north" else yy.min() - h["y_line"])
            per = np.asarray(per)
            lg = dict(min_m=round(float(per.min()), 4), median_m=round(float(np.median(per)), 4))
        else:
            lg = dict(min_m=None, median_m=None)
        lean_ok = bool(cover >= 1.0 - 1e-6 and lg["min_m"] is not None and lg["min_m"] <= 0.03
                       and lg["median_m"] <= 0.05)
        vis = ms.visibility(m, d, gid, r, cpos, cam)
        north_pts = P[P[:, 1] > y2 + 1e-6] if len(P) else P
        top_h = float(vis.get("top_m") or h["top_m"])
        in_box = ms.buried(north_pts, boxes, tops, top_h) if len(north_pts) else np.zeros(0, bool)
        south = bool(len(P) and r[3] <= y2 + 1e-6 and bool(np.all(in_box)))
        beyond = dict(n=int(len(north_pts)), not_buried=int((~in_box).sum()) if len(north_pts) else 0,
                      depth_m=round(float(north_pts[:, 1].max() - y2), 4) if len(north_pts) else 0.0)
        rows[h["name"]] = dict(keepout_dist_m=dk, keep_ok=bool(keep_ok), apron_points=in_apron, dust_points=on_dust,
                               spacing_ok=spacing_ok, line_cover=round(cover, 6), line_gap=lg, lean_ok=lean_ok,
                               line_run=run, visible_share=round(vis["share"], 4), top_seen=vis["top_seen"],
                               in_frame_share=round(vis["in_frame_share"], 4),
                               south_of_slab_front=south, beyond_slab_front=beyond, n_occupied=int(len(P)))
    chips_in = sorted(c["name"] for c in site.get("chips", []) for h in heaps
                      if h["rect"][0] < c["xy"][0] < h["rect"][1] and h["rect"][2] < c["xy"][1] < h["rect"][3])
    gp = ms.gap_promise(occ, boxes, lines)
    for n, row in rows.items():
        row["gaps"] = gp["pairs"].get(n, {})
        row["gap_ok"] = bool(all(v["ok"] for v in row["gaps"].values()))
    g21 = dict(pass_=bool(not chips_in and all(r["keep_ok"] and r["apron_points"] == 0 and r["dust_points"] == 0
                                               and r["spacing_ok"] and r["gap_ok"] for r in rows.values())),
               rule=gp["rule"], chips_in_heap_rects=chips_in)
    g22_heaps = bool(all(r["visible_share"] >= fam.VIS_MIN and r["top_seen"] and r["south_of_slab_front"]
                         for r in rows.values()))
    nA = sum(1 for h in heaps if h["band"] == "A")
    nB = sum(1 for h in heaps if h["band"] == "B")
    nF = sum(1 for h in heaps if h["against"] == "slab front")
    g23 = dict(n=len(heaps), A=nA, B=nB, slab_front=nF, all_lean=bool(all(r["lean_ok"] for r in rows.values())))
    g23["pass_"] = bool(len(heaps) >= 3 and nA >= 1 and nB >= 2 and nF >= 1 and g23["all_lean"])
    return rows, g21, g22_heaps, g23


def under_slab_rays(m, d, site):
    """Camera rays (2 x 2 per pixel) ending inside the site north of the front row below the slab's underside on
    anything but a roof (the walls' tops and the slab's top are above it; the floor plane outside the walls is not
    the site)."""
    import mujoco
    gid, _dist, hit, _shape = w2p.cast_image(m, d, site, ss=2)
    yg = site["bands"]["R2_front"][1]
    hz = [b for b in site["boxes"] if b["name"] == "roof__void_slab"][0]["aabb"][4]
    names = np.array([(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, int(g)) or "") for g in range(m.ngeom)] + [""])
    nm = names[np.where(gid >= 0, gid, m.ngeom)]
    roof = np.char.startswith(nm.astype(str), "roof__")
    lx, ly = site["size_m"]
    inside = (hit[:, 0] > 0.0) & (hit[:, 0] < lx) & (hit[:, 1] < ly)
    return int(((gid >= 0) & ~roof & inside & (hit[:, 1] > yg + 1e-6) & (hit[:, 2] < hz - 1e-6)).sum())


def evaluate(site, xml, bundle, render=True, cf_render=False, full=False):
    """qc dict of the site (bundle = family.build_sites(...)): G1-G24; `pass_G1_G19` and `pass` (G1-G24)."""
    q = w3p.evaluate(site, xml, render=render, g9=bundle["g9"], bundle=None)
    env = site["envelope"]["values"]
    ts = bundle["tau_star"]
    lib = bundle["lib"]
    # ---- G7c with the heaps' public records
    m0, d0 = w2p.compile_world(xml)
    nm0, hp0, names0 = w3p.ray_view(site, xml, md=(m0, d0))
    mv0, pv0 = w3p.mission_view(site), public_view(site, names0, xml)
    rw0 = w3p.render_world(site, xml) if cf_render else None
    cf = dict(tau=[], rays_identical=[], hit_names_identical=[], mission_identical=[], public_identical=[],
              render_identical=[] if cf_render else None,
              heap_rays=int(np.char.startswith(nm0.astype(str), "heap__").sum()))
    for t in range(3):
        if t == ts:
            continue
        s, x = bundle["sites"][t], bundle["xmls"][t]
        nm, hp, names = w3p.ray_view(s, x)
        cf["tau"].append(t)
        cf["rays_identical"].append(bool((nm == nm0).all() and np.array_equal(hp, hp0)))
        cf["hit_names_identical"].append(names == names0)
        cf["mission_identical"].append(w3p.mission_view(s) == mv0)
        cf["public_identical"].append(public_view(s, names, x) == pv0)
        if cf_render:
            rw = w3p.render_world(s, x)
            cf["render_identical"].append(rw["rgb_sha256"] == rw0["rgb_sha256"]
                                          and rw["depth_sha256"] == rw0["depth_sha256"])
    if cf_render:
        cf["render"] = rw0
    ok7c = all(cf["rays_identical"]) and all(cf["hit_names_identical"]) and all(cf["mission_identical"]) \
        and all(cf["public_identical"]) and (not cf_render or all(cf["render_identical"]))
    q["G7"]["G7c"] = cf
    q["G7"]["pass_"] = bool(q["G7"]["pass_"] and ok7c)
    # ---- G9 and G17-G19 on every counterfactual site
    per = {}
    for t in range(3):
        s = bundle["sites"][t]
        rp = w3p.rubble_predicates(s, env)
        g9t = bundle["g9s"][t]
        per[str(t)] = dict(mouth=s["void"]["true_mouth"], G9=bool(g9t["pass"]) if g9t else None, **rp)
    q["G17_G19"] = dict(counterfactual=per, rule="judged on every counterfactual site, mouth tau read as the true "
                                                  "mouth on site tau; any failure rejects the seed")
    for k in ("G17", "G18", "G19"):
        q[k] = dict(pass_=bool(all(p[k] for p in per.values())))
    q["G9_all"] = dict(pass_=bool(all(p["G9"] for p in per.values())))
    # ---- G20-G24 on every counterfactual model
    cls = site["classes"]
    keep, aprons = fam.keepouts(site)
    dust = [(p["x0"], p["x1"], p["y0"], p["y1"]) for p in site["patches"] if p["name"] == "dust"]
    lib_cls = lib.classes
    heaps = site["heaps"]["heaps"]
    pieces = bundle["pieces"]
    g20, g21, g22, g23, g24 = ({} for _ in range(5))
    shape_ref = {}
    order = [ts] + [t for t in range(3) if t != ts]
    for t in order:
        s, x = bundle["sites"][t], bundle["xmls"][t]
        m, d = (m0, d0) if t == ts else ms.compile_xml(x)
        ids = ms.geom_ids(m, "heap__")
        solids_rect = fam.union_solids([s])
        rows, r21, r22h, r23 = heap_place_checks(m, d, s, cls, keep, aprons, dust, solids_rect)
        n_under = under_slab_rays(m, d, s)
        r20, r24 = {}, {}
        for k, h in enumerate(heaps):
            gid = ids[h["name"]]
            sig = _heap_arrays(m, d, gid)
            if t == ts or full:
                pid = pieces[h["name"]]["piece"]
                piece = lib.pieces[pid]
                H, meta = lib.grid(pid)
                a20, a24 = hc.heap_shape(m, d, gid, h, lib_cls, piece, H, meta, site["seed"], k, lib.mesh(pid))
                shape_ref[h["name"]] = (sig, a20, a24)
                r20[h["name"]], r24[h["name"]] = a20, a24
            else:
                ref = shape_ref[h["name"]]
                same = sig == ref[0]
                r20[h["name"]] = dict(pass_=bool(same and ref[1]["pass_"]), identical_to_true_site_model=same)
                r24[h["name"]] = dict(pass_=bool(same and ref[2]["pass_"]), identical_to_true_site_model=same)
        key = str(t)
        g20[key] = dict(pass_=bool(all(v["pass_"] for v in r20.values())), heaps=r20)
        g24[key] = dict(pass_=bool(all(v["pass_"] for v in r24.values())), heaps=r24)
        k21 = ("keepout_dist_m", "keep_ok", "apron_points", "dust_points", "spacing_ok", "gaps", "gap_ok")
        g21[key] = dict(r21, heaps={n: {kk: v for kk, v in r.items() if kk in k21} for n, r in rows.items()})
        g22[key] = dict(pass_=bool(r22h and n_under == 0 and q["G7"]["hidden_box_rays_hit"] == 0),
                        under_slab_rays=n_under, heaps={n: {kk: r[kk] for kk in ("visible_share", "top_seen",
                                                                                 "in_frame_share",
                                                                                 "south_of_slab_front",
                                                                                 "beyond_slab_front")}
                                                         for n, r in rows.items()})
        g23[key] = dict(r23, heaps={n: {kk: r[kk] for kk in ("line_cover", "line_gap", "lean_ok", "line_run")}
                                    for n, r in rows.items()})
    for name, per_cf in (("G20", g20), ("G21", g21), ("G22", g22), ("G23", g23), ("G24", g24)):
        q[name] = dict(pass_=bool(all(v["pass_"] for v in per_cf.values())), counterfactual=per_cf)
    q["heaps_summary"] = dict(n=len(heaps), by_band=site["heaps"]["by_band"], by_class=site["heaps"]["by_class"],
                              at_slab_front=site["heaps"]["at_slab_front"])
    keys = list(w3p.KEYS_G1_G16)
    if not render:
        keys.remove("G3")
    f16 = [k for k in keys if not q[k]["pass_"]]
    q["failed_G1_G16"] = f16
    q["pass_G1_G16"] = not f16
    f19 = f16 + [k for k in ("G9_all", "G17", "G18", "G19") if not q[k]["pass_"]]
    q["failed_G1_G19"] = f19
    q["pass_G1_G19"] = not f19
    failed = f19 + [k for k in ("G20", "G21", "G22", "G23", "G24") if not q[k]["pass_"]]
    q["failed"] = failed
    q["pass"] = not failed
    return q
