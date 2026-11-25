"""Site family W4 of version 2.0: seed -> the three counterfactual sites and the site.

W4 = W3 (module w3gen, imported and never changed: the visible stage, the three mouths, the void, the rubble under
the slab by rule A, camera, light, friction) + heaps on the open floor:
  * of W2's free blocks only the first two, the red and the blue crate, stay; the grey blocks are not placed (the
    free-block draws are made exactly as in W3, so every later number of the visible stream is the same);
  * heaps (classes R, rubble heap, and M, mound of fill, from the heap library) are placed from a NEW visible stream
    default_rng([seed, 14]) after everything of W3's visible stage is fixed and against the union of the solids of the
    three counterfactual sites; band A wants 2-3, band B 2-3 (its first one is tried against the slab's front edge
    first), class R with probability P_R, else M. Every heap leans on the south or the north line of its band (A: the
    site's south wall or the ridge's south face; B: the ridge's north face or the slab's front edge): no heap stands
    free.
    A heap's occupancy for every rule is its placed rectangle.
  * the heaps are dropped from the last one while G9 fails on ANY of the three counterfactual sites (judged before
    the rubble, like W3's free blocks), then the colour chips are drawn (W2's rule, stream [seed, 2], route of the
    tau = 0 counterfactual; a heap's rectangle counts as a solid for them), then the rubble under the slab (W3's rule
    A, stream [seed, 13]) closes the assembly. Under the slab nothing changes against W3: same boxes, same rule.
Nothing a camera ray or mission() can see depends on which mouth is the true one.

Heap draws (one try, always in this order): class, piece index, scale U(lo, hi) of the piece's registered interval
(mounds of fill: lo >= 1.0), mirror, line (south / north), x mode (flush west < 0.12, flush east < 0.24, else free),
x fraction. Rejected (all by the placed rectangle): within 0.45 m of the pad's centre, within 0.40 m of search point 1
or 2, within 0.40 m of the floor point 0.25 m in front of or behind the middle of a ridge opening; meeting a mouth's
apron [m0 - 0.05, m1 + 0.05] x [y2 - 0.70, y2]; meeting the dust patch; a gap of open floor in (0, W_SWEEP + 0.18)
to another solid's or heap's rectangle (the line it leans on not counted); the gap promise (< 0.15 m or >= 0.60 m on
the 5 mm lattice of the heap's own grid, measure.gap_promise) failing against any entity; the search camera seeing
(inside its image, unoccluded) less than 0.70 of the heap's surface or not its top (on the compiled world with the
heaps placed so far); the class geometry or the piece bound failing on that compiled world (module heapchecks).
Heap library version 3 (generator 0.3.0): mounds of fill have the scan's natural toe and no rim; their class check
is the rectangle, the top, the depth, the gap to the line, the scale and the piece's registered trial verdict, not the
edge rule (module heapchecks); no slope band is judged for either class. The spill sheet at a heap's foot is a
picture-only layer: it is not read here and enters neither the package nor the collision.
"""
import copy
import math
import os

import numpy as np

from w2gen import family as w2
from w2gen import grid as gridmod
from w2gen import predicates as w2p
from w3gen import family as w3

GENERATOR_VERSION = "w4gen 0.3.0"     # the one definition (w4gen.GENERATOR_VERSION reads it from here)
# 0.3.0: heap library version 3 (registry_v3.json; mounds with a natural toe, judged without the edge rule; no slope
#        band; the scale drawn inside the class rectangle; class R with probability 0.2); 0.2.0: class R with
#        probability 0.3; 0.1.0: even odds
from . import heapchecks as hc
from . import library as libmod
from . import measure as ms
from . import mjcf

if "heap" not in gridmod.SOLID_KINDS:
    gridmod.SOLID_KINDS = tuple(gridmod.SOLID_KINDS) + ("heap",)      # run time: a heap is a solid for every route

HERE = os.path.dirname(os.path.abspath(__file__))
DEV_DIR = os.path.normpath(os.path.join(HERE, "..", "..", ".."))
_ROOT = os.environ.get("ARB_SRC_ROOT")
TASK_DIR = os.path.join(_ROOT, os.path.basename(DEV_DIR)[:-len("-dev")]) if _ROOT else DEV_DIR[:-len("-dev")]
ACCEPTANCE = os.path.join(TASK_DIR, "tests", "voidscore", "acceptance.yaml")
WEIGHTS, CREDIT_RADIUS = w3.load_acceptance(ACCEPTANCE)
if WEIGHTS != w3.WEIGHTS or tuple(CREDIT_RADIUS) != tuple(w3.CREDIT_RADIUS):
    raise RuntimeError("the task's acceptance file differs from the one W3 reads: scoring numbers would differ")

FAMILY = "W4"
STREAM = 14
N_BAND = {"A": (2, 3), "B": (2, 3)}
TRIES = 1500
APRON = 0.70
APRON_SIDE = 0.05
KEEP_PAD, KEEP_SP, KEEP_OPEN, OPEN_OFF = 0.45, 0.40, 0.40, 0.25
VIS_MIN = 0.70
X_FLUSH = (0.12, 0.24)
P_R = 0.2                      # class draw: R with probability P_R, else M (mounds are wider and rejected more)
# 0.2 with heap library version 3: at 0.3 the mounds were 0.317 of the heaps of admitted development sites (18 seeds),
# below the target 0.35-0.60; one step of 0.1 down
KIND = {"R": "rubble heap", "M": "mound of fill"}
LINE_KIND = {("A", "south"): "site wall", ("A", "north"): "ridge", ("B", "south"): "ridge",
             ("B", "north"): "slab front"}
r6 = w2.r6
_LIB = None


def library():
    global _LIB
    if _LIB is None:
        _LIB = libmod.Library()
    return _LIB


def _rect_pt(r, p):
    return math.hypot(max(r[0] - p[0], 0.0, p[0] - r[1]), max(r[2] - p[1], 0.0, p[1] - r[3]))


def keepouts(site):
    pad = site["pad"]["xy"]
    sp = {s["id"]: s["xy"] for s in site["search_points"]}
    keep = [(tuple(pad), KEEP_PAD), (tuple(sp[1]), KEEP_SP), (tuple(sp[2]), KEEP_SP)]
    for g in site["r1"]["openings"]:
        gx = 0.5 * (g["x0"] + g["x1"])
        keep += [((gx, g["y0"] - OPEN_OFF), KEEP_OPEN), ((gx, g["y1"] + OPEN_OFF), KEEP_OPEN)]
    aprons = [(g["x0"] - APRON_SIDE, g["x1"] + APRON_SIDE, g["y0"] - APRON, g["y0"]) for g in site["r2"]["openings"]]
    return keep, aprons


def union_solids(sites):
    """Walls and the boxes standing on the floor (rubble, support, plug, crate) of the three counterfactual sites."""
    lx, ly = sites[0]["size_m"]
    WT = w2.WALL_T
    out = [("wall__w", (-WT, 0.0, -WT, ly + WT)), ("wall__e", (lx, lx + WT, -WT, ly + WT)),
           ("wall__s", (-WT, lx + WT, -WT, 0.0)), ("wall__n", (-WT, lx + WT, ly, ly + WT))]
    seen = set(n for n, _r in out)
    for s in sites:
        for b in s["boxes"]:
            if b["kind"] in ("rubble", "support", "plug", "crate"):
                key = b["name"] + repr(tuple(b["aabb"][:4]))
                if key not in seen:
                    seen.add(key)
                    out.append((key, tuple(b["aabb"][:4])))
    return out


def heap_box(h):
    x0, x1, y0, y1 = h["rect"]
    return w2._box(h["name"].split("__", 1)[1], "heap", x0, x1, y0, y1, 0.0, h["top_m"], rgb=mjcf.HEAP_RGB[h["cls"]],
                   heap=dict(cls=h["cls"], band=h["band"], line=h["line"], against=h["against"]))


def grid_occupancy(g):
    """5 mm lattice points of a site grid where the heap stands > 3 mm above the floor (generator side)."""
    H = np.asarray(g["H"])
    ny, nx = H.shape
    X = g["origin"][0] + g["grid"] * np.arange(nx)[None, :] + 0 * H
    Y = g["origin"][1] + g["grid"] * np.arange(ny)[:, None] + 0 * H
    sel = (H + g["base_z"]) > ms.OCC_Z
    return np.c_[X[sel], Y[sel]]


def _with_heaps(site, heaps):
    s = dict(site)
    s["boxes"] = [b for b in site["boxes"] if b["kind"] != "heap"] + [heap_box(h) for h in heaps]
    s["heaps"] = dict(heaps=heaps)
    return s


def world_xml(site, heaps, grids):
    return mjcf.site_xml(_with_heaps(site, heaps), grids)


def place_heaps(seed, sites, cls, lib):
    """Heaps of the three counterfactual sites (identical in all three): (heap dicts, grids, piece map, record)."""
    rng = np.random.default_rng([int(seed), STREAM])
    site = sites[0]
    lx, ly = site["size_m"]
    clr = cls["free_block_clearance"]
    keep, aprons = keepouts(site)
    solids = union_solids(sites)
    dust = [(p["x0"], p["x1"], p["y0"], p["y1"]) for p in site["patches"] if p["name"] == "dust"]
    bands = {"A": tuple(site["bands"]["A"]), "B": tuple(site["bands"]["B"])}
    cpos, Rcam = w2p.cam_pose(site)
    cam = (Rcam, site["camera"]["model"])
    heaps, grids, pieces, occupancy = [], {}, {}, {}
    rej = {}
    tried = 0
    n_want = {}

    def no(k):
        rej[k] = rej.get(k, 0) + 1

    for band in ("A", "B"):
        ya, yb = bands[band]
        n_want[band] = int(rng.integers(N_BAND[band][0], N_BAND[band][1] + 1))
        k = 0
        for t in range(TRIES):
            if k >= n_want[band]:
                break
            tried += 1
            c = "R" if rng.random() < P_R else "M"
            ids = lib.by_cls[c]
            pid = ids[int(rng.integers(0, len(ids)))]
            piece = lib.pieces[pid]
            sig = float(rng.uniform(piece["scale"][0], piece["scale"][1]))
            mirror = bool(rng.random() < 0.5)
            north = bool(rng.random() < 0.5)
            mx = float(rng.random())
            xu = float(rng.random())
            if band == "B" and k == 0 and t < TRIES // 2:
                north = True                         # band B's first heap is tried against the slab's front edge first
            line = "north" if north else "south"
            W = sig * piece["W"]
            if mx < X_FLUSH[0]:
                x0 = 0.0
            elif mx < X_FLUSH[1]:
                x0 = lx - W
            else:
                x0 = xu * (lx - W)
            y_line = yb if north else ya
            yaw = 0 if north else 180
            rect = libmod.placed_rect(piece, sig, yaw, x0 + W / 2, y_line)
            if rect[3] - rect[2] > yb - ya:
                no("deeper than the band")
                continue
            if not all(_rect_pt(rect, p) >= r for p, r in keep):
                no("keep-out")
                continue
            if any(w2._rect_overlap(rect, q) for q in aprons):
                no("mouth apron")
                continue
            if any(w2._rect_overlap(rect, q) for q in dust):
                no("dust patch")
                continue
            if north:
                near = [r for _n, r in solids if r[2] < yb - 1e-9]
            else:
                near = [r for _n, r in solids if r[3] > ya + 1e-9]
            if not w2.spacing_ok(rect, near + [h["rect"] for h in heaps], clr):
                no("spacing")
                continue
            name = "heap__%s%d" % (band, k)
            place = dict(rect=rect, scale=sig, mirror=mirror, yaw_deg=yaw, cx=x0 + W / 2, y_line=y_line)
            g = libmod.collision_grid(lib, pid, place)
            occ_g = libmod.lattice_grid(lib, pid, place)
            cand = dict(name=name, cls=c, kind=KIND[c], band=band, line=line, against=LINE_KIND[(band, line)],
                        rect=[float(q) for q in rect], top_m=r6(g["top_m"]), yaw_deg=yaw, mirror=mirror, scale=sig,
                        x_mode="flush west" if mx < X_FLUSH[0] else "flush east" if mx < X_FLUSH[1] else "free",
                        y_line=float(y_line), cx=float(x0 + W / 2),
                        grid=dict(origin=g["origin"], shape=list(g["shape"]), cell_m=g["grid"], sink_m=-g["base_z"],
                                  size_z=g["size_z"]))
            run, cover = ms.line_run(solids, rect, line, y_line)
            if cover < 1.0 - 1e-6:
                no("not against the line over its whole width")
                continue
            occ = {h["name"]: occupancy[h["name"]] for h in heaps}
            occ[name] = grid_occupancy(occ_g)
            lines = {h["name"]: h["_line_run"] for h in heaps}
            lines[name] = run
            gp = ms.gap_promise(occ, solids, lines)
            if not gp["ok"]:
                no("gap promise")
                continue
            grids_c = dict(grids)
            grids_c[name] = g
            m, d = ms.compile_xml(world_xml(site, heaps + [cand], grids_c))
            gid = ms.geom_ids(m, name)[name]
            vis = ms.visibility(m, d, gid, rect, cpos, cam)
            if vis["share"] < VIS_MIN or not vis["top_seen"]:
                no("camera")
                continue
            pid_piece, (H_, meta_) = lib.pieces[pid], lib.grid(pid)
            a20, a24 = hc.heap_shape(m, d, gid, cand, lib.classes, pid_piece, H_, meta_, seed, len(heaps),
                                     lib.mesh(pid))
            if not a20["pass_"]:
                no("class geometry")
                continue
            if not a24["pass_"]:
                no("piece bound")
                continue
            cand["visible_share_at_placement"] = round(vis["share"], 4)
            cand["_line_run"] = run
            heaps.append(cand)
            grids[name] = g
            occupancy[name] = occ[name]
            pieces[name] = dict(piece=pid, scale=sig, mirror=mirror, yaw_deg=yaw,
                                scale_interval_used=[r6(q) for q in piece["scale"]],
                                scale_interval_registered=piece["scale_registered"])
            k += 1
    rec = dict(tried=tried, wanted=n_want, rejections=dict(sorted(rej.items())))
    for h in heaps:
        h.pop("_line_run")
    return heaps, grids, pieces, rec


def build_sites(seed, env=None, lx=w3.LX_DEFAULT, with_world_checks=True, lib=None):
    """dict(site, xml, sites, xmls, g9s, grids, pieces, tau_star, vis, hid, cls, env, lib) of one seed."""
    lib = lib or library()
    env, cls, rng, vis, hid, attempts, reasons = w3.make_layouts(seed, env, lx)
    tau_star = int(np.random.default_rng([int(seed), 12]).integers(0, 3))
    sites = [w3._assemble(seed, vis, hid[t], t, env, cls, lx, attempts, reasons) for t in range(3)]
    camera = w3.draw_camera(rng, lx, w3.LY_CAM)
    lighting = w3.draw_lighting(rng)
    physics = dict(floor_mu=r6(rng.uniform(0.76, 0.92)), box_mu=w3.BOX_MU, patch_mu=dict(w3.PATCH_MU),
                   floor_mu_rule="U(0.76, 0.92) per site (authors' sysID range)")
    for s in sites:
        s["family"] = FAMILY
        s["camera"], s["lighting"], s["physics"] = camera, lighting, physics
        s["camera_rule_note"] = "W2's camera rule with the camera rectangle [0, LX] x [0, LY_CAM = 4.0]"
        s["gravel_relief"] = dict(enabled=False, grid_m=0.01, wavelength_m=[0.03, 0.08], rms_m=0.008, peak_max_m=0.025,
                                  edge_taper_m=0.05, heights="bumps only, >= 0 on top of the floor",
                                  rng_stream="default_rng([seed, 1])")
    union = dict(sites[0])
    seen, ub = set(), []
    for s in sites:
        for b in s["boxes"]:
            key = (b["name"], tuple(b["aabb"]))
            if key not in seen:
                seen.add(key)
                ub.append(b)
    union["boxes"] = ub
    free = w2.place_free_blocks(rng, union, cls)           # W3's draws, so the visible stream stays W3's
    crates = [b for b in free if b["kind"] == "crate"]
    for s in sites:
        s["boxes"] += [dict(b) for b in crates]
    heaps, grids, pieces, prec = place_heaps(seed, sites, cls, lib)
    dropped, dropped_crates = 0, 0
    g9s = None
    while True:
        xmls = [world_xml(s, heaps, grids) for s in sites]
        ss = [_with_heaps(s, heaps) for s in sites]
        g9s = [w2p.g9_routes(s, x) for s, x in zip(ss, xmls)] if with_world_checks else [None] * 3
        if g9s[0] is None or all(g["pass"] for g in g9s):
            break
        if heaps:
            gone = heaps.pop()
            grids.pop(gone["name"])
            pieces.pop(gone["name"])
            dropped += 1
            continue
        fr = [b["name"] for b in sites[0]["boxes"] if b.get("free_block")]
        if not fr:
            break
        for s in sites:
            s["boxes"] = [b for b in s["boxes"] if b["name"] != fr[-1]]
        dropped_crates += 1
    for s in sites:
        s["free_blocks"] = dict(placed=len(free), grey_not_placed=len(free) - len(crates),
                                dropped_for_G9=dropped_crates,
                                kept=[b["name"] for b in s["boxes"] if b.get("free_block")],
                                rule="W2's free-block draws as in W3 (A <= 3, B <= 2, keep-outs, spacing) against the "
                                     "union of the three counterfactual sites' solids; only the first two, the red and "
                                     "the blue crate, are placed; a crate is dropped from the last one while G9 fails "
                                     "on any counterfactual site and no heap is left to drop")
        s["boxes"] += [heap_box(h) for h in heaps]
        s["heaps"] = dict(
            n=len(heaps), by_band={b: sum(1 for h in heaps if h["band"] == b) for b in "AB"},
            by_class={c: sum(1 for h in heaps if h["cls"] == c) for c in "RM"},
            at_slab_front=sum(1 for h in heaps if h["against"] == "slab front"),
            heaps=[dict(h) for h in heaps], dropped_for_G9=dropped, placement=prec,
            library=dict(sha256=lib.sha256, version=lib.version),
            classes={c: dict(w=list(lib.classes[c]["w"]), d=list(lib.classes[c]["d"]), top=list(lib.classes[c]["top"]))
                     for c in ("R", "M")},
            collision=dict(kind="inline height field per heap: the library piece's 5 mm grid turned, mirrored and "
                                "scaled vertex for vertex (cell 0.005 m x scale)", piece_cell_m=libmod.GRID,
                           sink_m=libmod.SINK, zero_ring=True, max_equals_size_z=True, elevation_rows="from +y to -y",
                           geom="heap__<band><k>", field="field_<band><k>"),
            stream="default_rng([seed, %d])" % STREAM,
            rule="2-3 heaps wanted in band A and 2-3 in band B, class R with probability %g, M with %g; each leans on its band's south "
                 "or north line (A: site wall or ridge; B: ridge or slab front; band B's first heap is tried against "
                 "the slab front first); occupancy = the placed rectangle; keep-outs pad 0.45 m, search points 1-2 "
                 "0.40 m, 0.40 m around the floor points 0.25 m before / behind each ridge opening, the mouths' aprons "
                 "(width + 0.05 m each side, 0.70 m south), the dust patch; gap 0 or >= W_SWEEP + 0.18 to every other "
                 "rectangle (the line not counted); gap promise < 0.15 m or >= 0.60 m on the heap's own 5 mm grid; "
                 "camera sees >= 0.70 of its surface and its top; placed against the union of the three "
                 "counterfactual sites, dropped from the last one while G9 fails on any of them" % (P_R, round(1.0 - P_R, 6)))
    route_pts = []
    if g9s[0] is not None and g9s[0]["a"]["legs"]:
        for leg in g9s[0]["a"]["legs"][:2]:
            if leg["path"]:
                route_pts += leg["path"][::5]
    chip_site = dict(sites[0])
    chip_site["boxes"] = [dict(b, kind="rubble") if b["kind"] == "heap" else b for b in sites[0]["boxes"]]
    chips = w2.place_chips(np.random.default_rng([int(seed), 2]), chip_site, route_pts)
    for s in sites:
        s["chips"] = [dict(c) for c in chips]
        s["chip_rule"] = ("W2's chip rule, default_rng([seed, 2]); route = pad -> 1 -> 2 of the tau = 0 counterfactual "
                          "(never enters the void); a heap's rectangle counts as a solid")
    for t, s in enumerate(sites):
        w3.add_rubble(seed, s, vis, hid[t], cls)
        s["generator"] = GENERATOR_VERSION
    xmls = [mjcf.site_xml(s, grids) for s in sites]
    g9s = [w2p.g9_routes(s, x) for s, x in zip(sites, xmls)] if with_world_checks else [None] * 3
    return dict(site=sites[tau_star], xml=xmls[tau_star], g9=g9s[tau_star], sites=sites, xmls=xmls, g9s=g9s,
                grids=grids, pieces=pieces, tau_star=tau_star, vis=vis, hid=hid, cls=cls, env=env, lib=lib)


def sidecar(bundle):
    """The maintainers' table: which library piece made each heap (never in the package)."""
    lib = bundle["lib"]
    return dict(seed=bundle["site"]["seed"], generator=GENERATOR_VERSION, library_sha256=lib.sha256,
                library_version=lib.version, registry=lib.path, lib_root=lib.root,
                excluded_pieces=lib.excluded,
                heaps=[dict(name=n, **v) for n, v in bundle["pieces"].items()])


def copy_site(site):
    return copy.deepcopy(site)
