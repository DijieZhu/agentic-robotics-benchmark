"""The generator pipeline: seed -> site folder (site.json, prior.json, scene.xml + assets/, trav.npz, fields.npz,
maps/). Deterministic in (seed, config): retries use sub-seeds (attempt 0, 1, ...) until the guarantees hold."""

from __future__ import annotations

import math
import os
import shutil
import time

import mujoco
import numpy as np

from . import furnish as F
from . import furnish_plant as FP
from . import radiation as RAD
from .config import config_hash, derived_thresholds, load_config, observability_windows
from .discrepancy import choose_and_apply
from .ip_models import IPWorks
from .ips import retag_draft, select
from .layout import build_layout
from .link import coverage, opaque_raster
from .mjcf import build_scene, write_xml
from .prior import build_prior
from .readable import compute as compute_readable, readable_states
from .route import observation_pose, plan
from .structure import build_structure
from .trav import MODES, Grid, TravMap, _hull
from .util import Rng, dump_json, poly_rect, rect_center, runs_from_mask, uniform
from .world import World

RULES = {"airlock": F.f_airlock, "pipe_chase": F.f_pipe_chase, "annex": F.f_annex, "decor": F.f_decor,
         "pump_room": FP.f_pump_room, "valve_gallery": FP.f_valve_gallery, "tank_room": FP.f_tank_room,
         "electrical_room": FP.f_electrical_room, "waste_store": FP.f_waste_store}


class GuaranteeError(RuntimeError):
    pass


# ============================================================================================ world building
def furnish_all(W: World):
    for r in sorted(W.lay.rooms.values(), key=lambda r: r.id):
        if r.type == "corridor":
            continue
        c = F.Ctx(W, r)
        W.ctx[r.id] = c
        if W.ladder and r.type in ("decor", "annex", "electrical_room"):
            _reserve_instrument_walls(W, c)
        RULES[r.type](c)
    F.corridor_all(W)
    _corridor_slots(W)


def _reserve_instrument_walls(W, c, n=2, length=1.8, depth=0.6):
    """Ladder: keep one or two wall stretches of a room free of furniture (a process line or an instrument rack runs
    there) and register them as mount spots; the room rule then furnishes around them."""
    from .stations import wall_len
    g = W.rng("reserve/" + c.rid)
    v_min = (getattr(W, "v2_back_rooms", None) or {}).get(c.rid)
    if v_min is not None:
        # version 2: a room with an in-room strip or rack keeps its instrument walls behind the band (the back wall,
        # and the side walls beyond the far approach); the slot is placed there
        walls = ["back"] + [w for w in ("left", "right") if wall_len(c, w) - (v_min + 0.05) >= length + 0.45]
        order = [walls[int(i)] for i in g.permutation(len(walls))]
        got = 0
        for wall in order:
            L = wall_len(c, wall)
            lo = 0.45 if wall == "back" else v_min + 0.05
            for _ in range(12):
                if L - length - 0.45 < lo:
                    break
                s0 = float(g.uniform(lo, L - length - 0.45))
                s1 = s0 + length
                r = [s0, c.Lv - depth, s1, c.Lv] if wall == "back" else (
                    [0.0, s0, depth, s1] if wall == "left" else [c.Lu - depth, s0, c.Lu, s1])
                if c.free(r, pad=0.0, keep=True, keep_pad=0.1):
                    c.take(r)
                    W.spot(c, wall, s0 + 0.45, s1 - 0.45, "room", orients=("out", "along+", "along-"))
                    got += 1
                    break
            if got >= n:
                break
        return got
    walls = [w for w in ("back", "left", "right", "front") if wall_len(c, w) >= length + 0.9]
    order = [walls[int(i)] for i in g.permutation(len(walls))]
    got = 0
    for wall in order:
        L = wall_len(c, wall)
        for _ in range(12):
            s0 = float(g.uniform(0.45, L - length - 0.45))
            s1 = s0 + length
            if wall == "front":
                r = [s0, 0.0, s1, depth]
            elif wall == "back":
                r = [s0, c.Lv - depth, s1, c.Lv]
            elif wall == "left":
                r = [0.0, s0, depth, s1]
            else:
                r = [c.Lu - depth, s0, c.Lu, s1]
            if c.free(r, pad=0.0, keep=True, keep_pad=0.1):
                c.take(r)
                W.spot(c, wall, s0 + 0.45, s1 - 0.45, "room", orients=("out", "along+", "along-"))
                got += 1
                break
        if got >= n:
            break
    return got


def _corridor_slots(W):
    """Painted survey points in the corridor (candidates for dose-point IPs) and a fire-alarm panel."""
    lay = W.lay
    g = W.rng("corridor_slots")
    ws = [r for r in lay.rooms.values() if r.type == "waste_store"]
    for r in ws:
        for did in r.doors:
            d = lay.door(did)
            other = d.rooms[0] if d.rooms[1] == r.id else d.rooms[1]
            if other in ("C1", "C2", "C3"):
                x, y = d.xy()
                c = lay.rooms[other]
                cx, cy = rect_center(c.rect)
                if d.axis == "y":
                    p = (x + (1.2 if g.random() < 0.5 else -1.2), cy)
                else:
                    p = (cx, y + 1.2)
                W.slot("dose_point", (p[0], p[1], 0.0), 0.0, other, "corridor survey point outside the waste store",
                       "survey point", "corridor")
    c2 = lay.rooms["C2"]
    cx, cy = rect_center(c2.rect)
    W.slot("dose_point", (cx, cy, 0.0), 0.0, "C2", "corridor survey point at the bend", "survey point", "corridor")


def grid_markers(W: World, cfg, rng):
    """Fiducial plates on corridor walls near grid lines: documented on the drawing; most exist in reality."""
    g = rng("markers")
    lay = W.lay
    cands = []
    for leg in ("C1", "C3", "C2"):
        r = lay.rooms[leg]
        x0, y0, x1, y1 = r.inner
        along_x = (r.rect[2] - r.rect[0]) > (r.rect[3] - r.rect[1])
        lines = lay.grid["x_lines"] if along_x else lay.grid["y_lines"]
        for ln in lines:
            c0 = ln["x"] if along_x else ln["y"]
            lo, hi = (x0, x1) if along_x else (y0, y1)
            for side in (-1, 1):
                # the plate sits near the grid line, shifted along the wall when a door / fire door is in the way
                wall_c = (y0 if side < 0 else y1) if along_x else (x0 if side < 0 else x1)
                c = None
                for off in (0.0, 0.5, -0.5, 0.9, -0.9, 1.3, -1.3):
                    cc = c0 + off
                    if not (lo + 0.4 < cc < hi - 0.4):
                        continue
                    bad = False
                    for d in lay.doors:
                        if leg not in d.rooms:
                            continue
                        dx, dy = d.xy()
                        if abs((dy if along_x else dx) - wall_c) > 0.6:
                            continue
                        if abs((dx if along_x else dy) - cc) < d.width / 2 + 0.3:
                            bad = True
                    for f in getattr(W, "fire_doors", []):
                        if f["rooms"][0] == leg and abs(f["coord"] - cc) < 1.2:
                            bad = True
                    if not bad:
                        c = cc
                        break
                if c is None:
                    continue
                if along_x:
                    pos = [c, (y0 + 0.004) if side < 0 else (y1 - 0.004), 0.6]
                    yaw = math.pi / 2 if side < 0 else -math.pi / 2
                    gl = "%s/%s" % (ln["label"], _nearest(lay.grid["y_lines"], pos[1], "y"))
                else:
                    pos = [(x0 + 0.004) if side < 0 else (x1 - 0.004), c, 0.6]
                    yaw = 0.0 if side < 0 else math.pi
                    gl = "%s/%s" % (_nearest(lay.grid["x_lines"], pos[0], "x"), ln["label"])
                cands.append({"pos": pos, "yaw": yaw, "grid": gl, "on": "wall", "room": leg})
    miss = getattr(W, "missing_sheet", None)
    cands = [m for m in cands if not (miss and miss["rect"][0] <= m["pos"][0] <= miss["rect"][2] and
                                      miss["rect"][1] <= m["pos"][1] <= miss["rect"][3])]
    v2miss = getattr(W, "missing", None)
    if v2miss is not None:
        # version 2: no marker plate is listed or hangs on the missing part (its cut-off corridor parts;
        # a plate on a drawn corridor part's wall toward a missing room hangs in drawn space)
        cands = [m for m in cands if not (v2miss.in_cut(m["pos"][0], m["pos"][1], margin=-0.25) or
                                          v2miss.in_missing_room(m["pos"][0], m["pos"][1]))]
    n = min(len(cands), int(g.integers(cfg["discrepancies"]["grid_markers"][0],
                                        cfg["discrepancies"]["grid_markers"][1] + 1)))
    idx = sorted(g.choice(len(cands), size=n, replace=False).tolist()) if cands else []
    out = []
    for k, i in enumerate(idx):
        m = dict(cands[i])
        m["id"] = "GM-%02d" % (k + 1)
        m["marker_id"] = 300 + k
        m["documented"] = True
        m["present"] = bool(g.random() < float(cfg["discrepancies"]["grid_markers_present"]))
        out.append(m)
    return out


def _nearest(lines, v, key):
    best = min(lines, key=lambda l: abs(l[key] - v))
    return best["label"]


# ============================================================================================ checks
def _graph_reach(trav, modes, pad):
    G = trav.build_graph(modes)
    return trav.reach(G, pad)


def check_guarantees(trav, ips, pad_node, reach_all, reach_nocrouch, reach_nohs, blind, discs):
    N = trav.g.nx * trav.g.ny
    rep = {"all_reachable_readable": True, "unreachable": [], "needs_crouch": [], "needs_high_step": [],
           "behind_discrepancy": [], "blind_fail": [b["tag"] for b in blind if b["fail"]]}
    for ip in ips:
        rs = readable_states(trav, ip)
        nodes = []
        for mi, m in enumerate(MODES):
            iy, ix = np.nonzero(rs[m])
            nodes.append(mi * N + iy * trav.g.nx + ix)
        nodes = np.concatenate(nodes) if nodes else np.zeros(0, int)
        ok_all = bool(reach_all[nodes].any()) if len(nodes) else False
        ip["reach"] = {"all": ok_all, "without_crouch": bool(reach_nocrouch[nodes].any()) if len(nodes) else False,
                       "without_high_step": bool(reach_nohs[nodes].any()) if len(nodes) else False,
                       "readable_cells": int(len(nodes))}
        if not ok_all:
            rep["all_reachable_readable"] = False
            rep["unreachable"].append(ip["tag"])
            continue
        if not ip["reach"]["without_crouch"]:
            rep["needs_crouch"].append(ip["tag"])
        if not ip["reach"]["without_high_step"]:
            rep["needs_high_step"].append(ip["tag"])
        rel = [d["id"] for d in discs if ip["tag"] in d["refs"] or ip["room"] in d["refs"]]
        if ip.get("discrepancy") or rel:
            rep["behind_discrepancy"].append(ip["tag"])
    rep["ok"] = (rep["all_reachable_readable"] and len(rep["needs_crouch"]) >= 1 and
                 len(rep["needs_high_step"]) >= 1 and len(rep["behind_discrepancy"]) >= 1 and
                 len(rep["blind_fail"]) >= 2)
    return rep


def blind_follow(W, cfg, grid, trav, ips, pad_xy):
    """Plan on the drawing (drawn walls and doors, drawn equipment; walk with the footprint) to every documented
    location, replay the path against the real structure, and check the reading from the planned spot."""
    from scipy.sparse.csgraph import dijkstra as dj
    lay = W.lay
    th = derived_thresholds(cfg)
    sh = grid.shape
    drawn = np.zeros(sh, np.uint8)
    real = np.zeros(sh, np.uint8)
    inside = np.zeros(sh, np.uint8)
    for r in lay.rooms.values():
        grid.fill_poly(inside, poly_rect(r.rect), 1)
    # drawn walls with drawn openings
    for w in lay.walls:
        if any(lay.rooms.get(x) is not None and not lay.rooms[x].on_drawing for x in w.rooms) and \
                not any(lay.rooms.get(x) is not None and lay.rooms[x].on_drawing for x in w.rooms):
            continue
        ops = sorted([lay.door(o) for o in w.openings if lay.door(o).on_drawing], key=lambda d: d.center)
        cur = w.lo - 0.125
        segs = []
        for d in ops:
            segs.append((cur, d.center - d.width / 2))
            cur = d.center + d.width / 2
        segs.append((cur, w.hi + 0.125))
        for a, b in segs:
            if b - a < 1e-3:
                continue
            if w.axis == "x":
                grid.fill_poly(drawn, poly_rect([w.coord - w.t_neg, a, w.coord + w.t_pos, b]), 1)
            else:
                grid.fill_poly(drawn, poly_rect([a, w.coord - w.t_neg, b, w.coord + w.t_pos]), 1)
    for e in W.equipment:
        if not e["on_drawing"] or e["kind"] in ("deployment_pad",):
            continue
        if e["shape"] == "circle":
            cx, cy = e["center"]
            rr = e["radius"]
            poly = [[cx + rr * math.cos(a), cy + rr * math.sin(a)] for a in np.linspace(0, 2 * math.pi, 16,
                                                                                          endpoint=False)]
        else:
            poly = e["footprint"]
        grid.fill_poly(drawn, poly, 1)
    # the real structure: walls / partitions / closed leaves / stored material
    for p in W.P.prims:
        if p.group not in (0, 3):
            continue
        if p.body.startswith(("wall/", "partition/", "block/")) or (p.body.startswith("door/") and "/leaf" in p.name):
            z0, z1 = p.zrange()
            if z0 > 0.4:
                continue
            grid.fill_poly(real, _hull(p.corners()[:, :2]), 1)
    out_mask = inside == 0
    solid = drawn.astype(bool) | out_mask
    from scipy import ndimage
    d = (ndimage.distance_transform_edt(~solid) - 0.5) * grid.res
    en = (~solid) & (d >= th["r_in"])
    # graph over the drawing
    ny, nx = sh
    idx = np.arange(ny * nx).reshape(sh)
    rows, cols, vals = [], [], []
    for dy, dx, dist in ((-1, 0, 1), (1, 0, 1), (0, -1, 1), (0, 1, 1)):
        ys0, ys1 = max(0, -dy), ny - max(0, dy)
        xs0, xs1 = max(0, -dx), nx - max(0, dx)
        ok = en[ys0:ys1, xs0:xs1] & en[ys0 + dy:ys1 + dy, xs0 + dx:xs1 + dx]
        rows.append(idx[ys0:ys1, xs0:xs1][ok])
        cols.append(idx[ys0 + dy:ys1 + dy, xs0 + dx:xs1 + dx][ok])
        vals.append(np.full(int(ok.sum()), dist * grid.res))
    from scipy.sparse import csr_matrix
    G = csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(ny * nx, ny * nx))
    piy, pix = grid.cell(*pad_xy)
    D, pred = dj(G, directed=True, indices=[piy * nx + pix], return_predecessors=True)
    D, pred = D[0], pred[0]
    real_b = real.astype(bool)
    windows, _ = observability_windows(cfg)
    res = []
    for ip in ips:
        if ip["type"] == "hotspot":
            continue
        doc = ip.get("documented", "same")
        room = lay.rooms[ip["room"]]
        if doc is None or not room.on_drawing:
            res.append({"tag": ip["tag"], "fail": True, "why": "no documented location"})
            continue
        s = ip["slot"] if doc == "same" else {"pos": doc["pos"], "yaw": doc["yaw"]}
        mr = getattr(W, "missing_sheet", None)
        if mr and mr["rect"][0] <= s["pos"][0] <= mr["rect"][2] and mr["rect"][1] <= s["pos"][1] <= mr["rect"][3]:
            res.append({"tag": ip["tag"], "fail": True, "why": "on the missing sheet"})
            continue
        if ip["type"] == "dose_point":
            gx, gy = s["pos"][0], s["pos"][1]
        else:
            from .readable import window_for
            win = window_for(ip, windows)
            dd = win["d"][0] + 0.45 * (win["d"][1] - win["d"][0])
            gx = s["pos"][0] + math.cos(s["yaw"]) * dd * 0.85
            gy = s["pos"][1] + math.sin(s["yaw"]) * dd * 0.85
        gi = grid.cell(gx, gy)
        # nearest drawn-enterable cell
        best = None
        for r_ in range(0, 16):
            for dy in range(-r_, r_ + 1):
                for dx in range(-r_, r_ + 1):
                    yy, xx = gi[0] + dy, gi[1] + dx
                    if 0 <= yy < ny and 0 <= xx < nx and en[yy, xx] and np.isfinite(D[yy * nx + xx]):
                        if best is None or dy * dy + dx * dx < best[0]:
                            best = (dy * dy + dx * dx, yy, xx)
            if best:
                break
        if best is None:
            res.append({"tag": ip["tag"], "fail": True, "why": "no path on the drawing"})
            continue
        tgt = best[1] * nx + best[2]
        path = []
        n = tgt
        while n >= 0 and n != piy * nx + pix:
            path.append(n)
            n = int(pred[n])
        blocked = any(real_b.flat[c] for c in path)
        rd = ip["readable"]["stand"]
        y0, x0 = best[1], best[2]
        win_ok = rd[max(0, y0 - 10):y0 + 11, max(0, x0 - 10):x0 + 11].any()
        why = "blocked on the way" if blocked else ("not readable from the documented spot" if not win_ok else "")
        res.append({"tag": ip["tag"], "fail": bool(blocked or not win_ok), "why": why})
    return res


# ============================================================================================ main
def generate(seed, out_dir, cfg=None, maps=True, verbose=False):
    cfg = cfg if cfg is not None else load_config()
    if str(cfg.get("layout", "draft")).startswith("ladder"):
        # the ten-slot ladder (config switch `layout: ladder_v1`); same folder format, more fields
        from .ladder import generate as generate_ladder
        return generate_ladder(seed, out_dir, cfg, maps=maps, verbose=verbose)
    t0 = time.time()
    # absolute: write_xml strips the assets prefix from texture paths, and the bricks targets register their
    # textures by absolute path, so a relative out_dir left broken file= attributes in scene.xml
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    last = None
    for attempt in range(int(cfg["max_attempts"])):
        try:
            site, prior, stats = _attempt(seed, attempt, out_dir, cfg, verbose)
            stats["attempts"] = attempt + 1
            timing = dict(stats.pop("timing_s"))
            timing["generate_s"] = time.time() - t0
            site["stats"] = stats                       # deterministic statistics only (byte-identical site.json)
            dump_json(site, os.path.join(out_dir, "site.json"))
            dump_json(prior, os.path.join(out_dir, "prior.json"))
            if maps:
                from .maps import make_maps
                tm = time.time()
                make_maps(out_dir)
                timing["maps_s"] = time.time() - tm
            dump_json(timing, os.path.join(out_dir, "timing.json"))
            site["_timing"] = timing
            return site, prior
        except GuaranteeError as e:
            last = str(e)
            if verbose:
                print("seed %d attempt %d: %s" % (seed, attempt, e))
            continue
        except (ValueError, IndexError, KeyError, RuntimeError, ZeroDivisionError) as e:
            # a grammar dead end in this sub-seed (e.g. no room for an element): the next sub-seed decides
            last = "%s: %s" % (type(e).__name__, e)
            if verbose:
                print("seed %d attempt %d crashed: %s" % (seed, attempt, last))
            continue
    raise GuaranteeError("seed %d: no attempt satisfied the guarantees (%s)" % (seed, last))


def _attempt(seed, attempt, out_dir, cfg, verbose):
    T = {}
    t = time.time()
    rng = Rng(seed, attempt)
    lay = build_layout(cfg, rng)
    W = World(cfg, lay, rng)
    build_structure(W)
    furnish_all(W)
    T["world_s"] = time.time() - t
    sources = RAD.choose_sources(W, cfg, rng)
    und = [s for s in sources if not s["documented"]]
    ips, decoys, plain, types = select(W, cfg, rng, und[0] if und else None)
    discs = choose_and_apply(W, cfg, rng, ips, sources)
    retag_draft(W, rng, ips, decoys, discs)
    markers = grid_markers(W, cfg, rng)
    # dose points get their painted marker; values need the sources
    boxes = RAD.attenuators(W, cfg)
    bg = round(uniform(rng("radiation_bg"), *cfg["radiation"]["background"]), 3)
    for ip in ips:
        if ip["type"] == "dose_point":
            p = ip["slot"]["pos"]
            ip["truth"]["dose_rate_uSv_h"] = float(RAD.dose_rate([[p[0], p[1]]], sources, boxes, cfg, bg)[0])
    # instruments and the MJCF
    ipw = IPWorks(cfg)
    for it in ips + decoys + plain:
        if it["type"] == "hotspot":
            continue
        ipw.add(it["slot"], it["role"], it["tag"], it["truth"], it["params"])
    for m in markers:
        if m["present"]:
            ipw.add_marker(m)
    cams = eye_cameras(W, ips, decoys)
    t = time.time()
    if os.path.isdir(os.path.join(out_dir, "assets")):
        shutil.rmtree(os.path.join(out_dir, "assets"))
    spec, info = build_scene(W, ipw, out_dir, cams)
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    T["compile_s"] = time.time() - t
    for it in ips + decoys + plain:
        mi = info["ip_models"].get(it["body"])
        if mi:
            it["backend"] = mi["backend"]
            it["model_names"] = {k: v for k, v in mi.items() if k in ("body", "geoms", "face_geoms", "joints", "backend",
                                                                       "bricks", "marker_id")}
    for m in markers:
        mi = info["ip_models"].get("marker/%s" % m["id"])
        if mi:
            m["model_names"] = {k: v for k, v in mi.items() if k in ("body", "geoms", "face_geoms", "backend",
                                                                       "dictionary")}
    # traversability
    t = time.time()
    bb = lay.bbox()
    grid = Grid(bb, float(cfg["grid_res"]))
    trav = TravMap(grid, cfg)
    rects = [r.rect for r in lay.rooms.values()]
    trav.rasterize(W.P.prims, W.elements, rects)
    G = trav.build_graph()
    pad = trav.node(0.0, 0.0, "walk")
    if not trav.enter["walk"].flat[pad - 0]:
        raise GuaranteeError("pad not enterable")
    reach_all = trav.reach(G, pad)
    reach_nc = _graph_reach(trav, ("walk", "high_step"), pad)
    reach_nh = _graph_reach(trav, ("walk", "crouch_walk"), pad)
    T["trav_s"] = time.time() - t
    # readable regions
    t = time.time()
    wsrc = compute_readable(model, data, grid, trav, ips, cfg)
    T["readable_s"] = time.time() - t
    # the hotspot's truth: peak dose rate over its readable region
    for ip in ips:
        if ip["type"] == "hotspot":
            m = ip["readable"]["stand"]
            iy, ix = np.nonzero(m)
            pts = np.stack([grid.x0 + (ix + 0.5) * grid.res, grid.y0 + (iy + 0.5) * grid.res], 1)
            if len(pts):
                r = RAD.dose_rate(pts, sources, boxes, cfg, bg)
                k = int(np.argmax(r))
                ip["truth"] = {"source": ip["slot"]["owner"], "pos": ip["slot"]["pos"][:2],
                               "peak_dose_rate_uSv_h": float(r[k]), "peak_at": pts[k].tolist()}
    blind = blind_follow(W, cfg, grid, trav, ips, (0.0, 0.0))
    rep = check_guarantees(trav, ips, pad, reach_all, reach_nc, reach_nh, blind, discs)
    if not rep["ok"]:
        raise GuaranteeError("guarantees: unreachable=%s crouch=%s high_step=%s discrepancy=%s blind=%s" % (
            rep["unreachable"], rep["needs_crouch"], rep["needs_high_step"], rep["behind_discrepancy"],
            rep["blind_fail"]))
    # observation poses and the route
    t = time.time()
    from scipy.sparse.csgraph import dijkstra
    dist_pad = dijkstra(G, directed=True, indices=[pad])[0]
    obs = []
    for ip in ips:
        node, pose = observation_pose(trav, ip, dist_pad, cfg)
        if node is None:
            raise GuaranteeError("no observation pose for %s" % ip["tag"])
        ip["observation_pose"] = pose
        ip["_node"] = node
        obs.append((ip["tag"], node, float(cfg["ips"]["dwell_s"][ip["type"]])))

    def dose_fn(xy):
        return RAD.dose_rate(xy, sources, boxes, cfg, bg)
    route = plan(trav, G, pad, obs, cfg, dose_fn)
    dwell_dose = 0.0
    for ip in ips:
        p = ip["observation_pose"]["pos"]
        dwell_dose += float(dose_fn(np.array([p]))[0]) * float(cfg["ips"]["dwell_s"][ip["type"]]) / 3600.0
    route["dose_dwell_uSv"] = dwell_dose
    route["dose_total_uSv"] = route["dose_travel_uSv"] + dwell_dose
    T["route_s"] = time.time() - t
    # energy: power per mode over the segment times plus the dwell at idle power
    M = cfg["modes"]
    energy = 0.0
    for leg in route["legs"]:
        for s in leg["segments"]:
            energy += float(M[s["mode"]]["power_W"]) * s["time_s"]
    energy += float(M["idle_power_W"]) * route["dwell_s"]
    route["energy_Wh"] = energy / 3600.0
    budgets = {"time_limit_s": float(cfg["mission"]["time_limit_s"]),
               "dose_budget_uSv": _nice(route["dose_total_uSv"] / float(cfg["radiation"]["budget_fraction"])),
               "energy_budget_Wh": _nice(route["energy_Wh"] * float(cfg["mission"]["energy_margin"])),
               "frames": dict(cfg["mission"]["frames"])}
    # link coverage
    t = time.time()
    lg = Grid(bb, 0.1)
    opaque = opaque_raster(W, lg)
    inside = np.zeros(lg.shape, np.uint8)
    for r in rects:
        lg.fill_poly(inside, poly_rect(r), 1)
    ants = [(W.base_station["antenna"][0], W.base_station["antenna"][1], float(cfg["link"]["base_range"]))]
    ants += [(r["pos"][0], r["pos"][1], float(cfg["link"]["repeater_range"])) for r in W.repeaters if r["working"]]
    cov = coverage(lg, opaque, ants, inside.astype(bool))
    T["link_s"] = time.time() - t
    # dose fields (truth and the documented-only "last survey") on 0.1 m
    t = time.time()
    xs, ys, Dt = RAD.field_grid(bb, 0.1, sources, boxes, cfg, bg)
    _, _, Dd = RAD.field_grid(bb, 0.1, sources, boxes, cfg, bg, documented_only=True)
    zoning = _zoning(W, cfg, xs, ys, Dd)
    T["dose_s"] = time.time() - t
    # the prior
    prior = build_prior(W, cfg, rng, ips, sources, discs, budgets, markers, zoning)
    # files
    write_xml(spec, out_dir, info["assets"])
    trav.save(os.path.join(out_dir, "trav.npz"))
    np.savez_compressed(os.path.join(out_dir, "fields.npz"), dose_xs=xs, dose_ys=ys, dose_truth=Dt.astype(np.float32),
                        dose_documented=Dd.astype(np.float32), link_origin=np.array([lg.x0, lg.y0]), link_res=lg.res,
                        link=cov, inside=inside.astype(bool))
    site = _site_json(W, cfg, seed, attempt, ips, decoys, plain, sources, boxes, bg, discs, markers, route, rep,
                      blind, budgets, grid, info, wsrc, cov, lg, types, model)
    stats = {"timing_s": T, "ngeom": int(model.ngeom), "nbody": int(model.nbody), "nlight": int(model.nlight),
             "nprims": len(W.P.prims), "elements": len(W.elements), "slots": len(W.slots),
             "decoys": len(decoys), "plain_instruments": len(plain),
             "geoms_by_group": {str(k): int((model.geom_group == k).sum()) for k in range(4)},
             "route_rooms": site["route"]["rooms_visited"], "classes_used": route["classes_used"]}
    return site, prior, stats


def _nice(v):
    if v <= 0:
        return 1.0
    e = 10 ** math.floor(math.log10(v))
    return float(math.ceil(v / e * 2) / 2 * e)


def _zoning(W, cfg, xs, ys, Dd):
    """Zones of the last survey (documented sources only): per room (and per 6 m corridor stretch), the class of the
    highest documented dose rate inside."""
    lay = W.lay
    classes = cfg["radiation"]["zone_classes"]
    out = []
    X, Y = np.meshgrid(xs, ys)
    areas = []
    for r in lay.rooms.values():
        if not r.on_drawing:
            continue
        if r.type == "corridor":
            x0, y0, x1, y1 = r.inner
            along_x = (x1 - x0) > (y1 - y0)
            L = (x1 - x0) if along_x else (y1 - y0)
            n = max(1, int(round(L / 6.0)))
            for k in range(n):
                if along_x:
                    areas.append(([x0 + k * L / n, y0, x0 + (k + 1) * L / n, y1], r))
                else:
                    areas.append(([x0, y0 + k * L / n, x1, y0 + (k + 1) * L / n], r))
        else:
            areas.append((r.inner, r))
    for rect, r in areas:
        m = (X >= rect[0]) & (X <= rect[2]) & (Y >= rect[1]) & (Y <= rect[3])
        if not m.any():
            continue
        v = float(Dd[m].max())
        cls = RAD.zone_class(v, classes)
        out.append({"class": cls, "polygon": poly_rect(rect), "label": r.number, "center": list(rect_center(rect)),
                    "max_documented": v})
    return out


def eye_cameras(W, ips, decoys=()):
    """Eye-level (0.33 m) viewing cameras: the pad, the chase hatch, a high sill, a floor pipe, a gauge from 1.5 m,
    and the views the room rules registered (pumps, gallery, tight point, bund, ramp, drums, MCC, chase inside)."""
    lay = W.lay
    h = float(W.cfg["render"]["eye_height"])
    cams = [{"name": "cam/eye_pad", "pos": [0.0, 0.0, h], "target": [4.0, -0.25, h + 0.1], "what": "the deployment pad"}]
    for d in lay.doors:
        if d.kind == "chase_mouth":
            x, y = d.xy()
            w = lay.wall(d.wall)
            other = d.rooms[0] if lay.rooms.get(d.rooms[0]) and lay.rooms[d.rooms[0]].type == "corridor" else d.rooms[1]
            s_ = -1 if other == w.rooms[0] else 1
            if w.axis == "y":
                p, tgt = [x + 0.9, y + s_ * 1.6, h], [x, y, 0.25]
            else:
                p, tgt = [x + s_ * 1.6, y + 0.9, h], [x, y, 0.25]
            cams.append({"name": "cam/eye_chase", "pos": p, "target": tgt, "what": "the pipe chase hatch"})
            break
    # a needs_high_step door sill seen from the corridor side
    for d in lay.doors:
        if not d.sill or d.sill not in W.elements or W.elements[d.sill].cls != "needs_high_step":
            continue
        w = lay.wall(d.wall)
        side = None
        for k, rid in enumerate(w.rooms):
            if lay.rooms.get(rid) is not None and lay.rooms[rid].type == "corridor":
                side = -1 if k == 0 else 1
        if side is None:
            continue
        x, y = d.xy()
        if w.axis == "y":
            p = [x + 0.35, y + side * 1.25, h]
        else:
            p = [x + side * 1.25, y + 0.35, h]
        if _inside_any(lay, p):
            cams.append({"name": "cam/eye_sill", "pos": p, "target": [x, y, 0.12], "what": "a needs_high_step door sill"})
            break
    for e in W.elements.values():
        if e.kind != "floor_pipe" or lay.rooms[e.room].type == "pipe_chase":
            continue
        xs = [q[0] for q in e.footprint]
        ys = [q[1] for q in e.footprint]
        cx, cy = 0.5 * (min(xs) + max(xs)), 0.5 * (min(ys) + max(ys))
        along_y = (max(xs) - min(xs)) < (max(ys) - min(ys))
        for sgn in (-1, 1):
            p = [cx + sgn * 1.1, cy + 0.25, h] if along_y else [cx + 0.25, cy + sgn * 1.1, h]
            if _inside_room(lay.rooms[e.room], p, 0.35):
                cams.append({"name": "cam/eye_floorpipe", "pos": p, "target": [cx, cy, 0.03],
                             "what": "a floor pipe (%s)" % e.cls})
                break
        else:
            continue
        break
    gauges = [i for i in ips if i["type"] == "gauge" and lay.rooms[i["room"]].type != "pipe_chase"] + \
        [d for d in decoys if d["type"] == "gauge"]
    for ip in gauges:
        s_ = ip["slot"]
        for dist in (1.5, 1.3, 1.1):
            p = [s_["pos"][0] + dist * math.cos(s_["yaw"]), s_["pos"][1] + dist * math.sin(s_["yaw"]), h]
            if _inside_any(lay, p):
                cams.append({"name": "cam/eye_gauge", "pos": p, "target": s_["pos"], "what": "gauge %s from %.1f m" %
                             (ip["tag"], dist), "fovy": 45.0})
                break
        else:
            continue
        break
    names = {c["name"] for c in cams}
    for v in W.views:
        if v["name"] not in names:
            cams.append(dict(v))
            names.add(v["name"])
    # a camera must stand in free space: slide it toward its target until no collision geom whose height range
    # covers the camera height is within 0.2 m of it
    boxes = []
    for p in W.P.prims:
        if p.group not in (0, 3):
            continue
        c = p.corners()
        boxes.append((c[:, 0].min(), c[:, 1].min(), c[:, 0].max(), c[:, 1].max(), c[:, 2].min(), c[:, 2].max()))
    B = np.array(boxes) if boxes else np.zeros((0, 6))

    def blocked(q, r=0.2):
        h = q[2]
        return bool(((B[:, 0] - r <= q[0]) & (q[0] <= B[:, 2] + r) & (B[:, 1] - r <= q[1]) & (q[1] <= B[:, 3] + r) &
                     (B[:, 4] < h + 0.08) & (B[:, 5] > h - 0.08)).any())
    out = []
    for cam in cams:
        p0 = np.asarray(cam["pos"], float)
        t = np.asarray(cam["target"], float)
        d = t[:2] - p0[:2]
        L = float(np.linalg.norm(d))
        ok = not blocked(p0)
        k = 0
        while not ok and k < 6 and L > 0.9:
            k += 1
            q = p0.copy()
            q[:2] = p0[:2] + d / L * 0.2 * k
            if not blocked(q):
                cam = dict(cam, pos=q.tolist())
                ok = True
        if ok or cam["name"] == "cam/eye_pad":
            out.append(cam)
    return out


def _inside_room(r, p, m=0.2):
    return r.inner[0] + m < p[0] < r.inner[2] - m and r.inner[1] + m < p[1] < r.inner[3] - m


def _inside_any(lay, p):
    return any(_inside_room(r, p) for r in lay.rooms.values())


def _site_json(W, cfg, seed, attempt, ips, decoys, plain, sources, boxes, bg, discs, markers, route, rep, blind,
               budgets, grid, info, wsrc, cov, lg, types, model):
    lay = W.lay
    th = derived_thresholds(cfg)

    def ip_rec(ip):
        s = ip["slot"]
        rec = {"tag": ip["tag"], "type": ip["type"], "room": ip["room"], "room_number": lay.rooms[ip["room"]].number,
               "weight": ip.get("weight"), "reason": ip.get("reason"), "body": ip["body"],
               "pose": {"pos": s["pos"], "yaw": s["yaw"], "pitch": s.get("pitch", 0.0)},
               "mount": s["mount"],
               "service": s["service"], "params": ip["params"], "truth": ip["truth"],
               "documented": ip.get("documented", "as built"), "discrepancy": ip.get("discrepancy"),
               "reach": ip.get("reach"), "observation_pose": ip.get("observation_pose")}
        if "readable" in ip:
            rd = ip["readable"]
            rec["readable_region"] = {"grid": grid.info(), "window": rd["window"],
                                      "runs_stand": runs_from_mask(rd["stand"]),
                                      "runs_crouch": runs_from_mask(rd["crouch"]),
                                      "cells_stand": int(rd["stand"].sum()), "cells_crouch": int(rd["crouch"].sum())}
        rec["model_names"] = ip.get("model_names", {})
        return rec

    rooms_visited = []
    for leg in route["legs"]:
        for s in leg["segments"]:
            for x, y in s["waypoints"]:
                for r in lay.rooms.values():
                    if r.rect[0] <= x <= r.rect[2] and r.rect[1] <= y <= r.rect[3]:
                        if not rooms_visited or rooms_visited[-1] != r.id:
                            rooms_visited.append(r.id)
    rv = sorted(set(rooms_visited))
    route_out = {k: v for k, v in route.items() if not k.startswith("_")}
    route_out["rooms_visited"] = rv
    route_out["n_rooms_visited"] = len(rv)
    doors = [d.as_dict() for d in lay.doors] + [dict(f) for f in getattr(W, "fire_doors", [])]
    site = {
        "schema": "site_gen/site/0.1", "seed": int(seed), "attempt": int(attempt), "config_hash": config_hash(cfg),
        "capability_source": cfg.get("_capability_source"), "observability_source": wsrc,
        "frame": {"origin": "deployment pad centre (airlock datum)", "x": "into the building", "y": "left",
                  "z": "up", "units": "m, rad"},
        "building": {"rect": lay.building_rect, "bbox": lay.bbox(), "wall_height": float(cfg["building"]["wall_height"]),
                     "params": {k: v for k, v in lay.params.items() if not k.startswith("_") and k not in
                                ("waste_drums",)}, "grid": lay.grid},
        "thresholds": th, "capability_bands": cfg["capability"], "robot_footprint": cfg["robot"]["footprint"],
        "rooms": [r.as_dict() for r in sorted(lay.rooms.values(), key=lambda r: r.number)],
        "walls": [w.as_dict() for w in lay.walls],
        "partitions": getattr(W, "partitions", []),
        "doors": doors,
        "elements": [e.as_dict() for e in sorted(W.elements.values(), key=lambda e: e.id)],
        "equipment": W.equipment,
        "ips": [ip_rec(i) for i in ips],
        "decoys": [{"tag": d["tag"], "type": d["type"], "room": d["room"], "pose": {"pos": d["slot"]["pos"],
                    "yaw": d["slot"]["yaw"]}, "mount": d["slot"]["mount"], "params": d["params"], "truth": d["truth"],
                    "body": d["body"], "model_names": d.get("model_names", {})} for d in decoys],
        "plain_instruments": [{"type": d["type"], "room": d["room"], "pos": d["slot"]["pos"], "yaw": d["slot"]["yaw"],
                               "body": d["body"], "params": d["params"], "truth": d["truth"],
                               "model_names": d.get("model_names", {})} for d in plain],
        "thermal_groups": W.thermal_groups,
        "sources": sources,
        "radiation": {"model": RAD.model_name() + ": inverse square + exp(-mu L) through wall / shield boxes",
                      "background_uSv_h": bg, "mu_wall": cfg["radiation"]["mu_wall"],
                      "mu_shield": cfg["radiation"]["mu_shield"], "height": cfg["radiation"]["height"],
                      "r_min": cfg["radiation"]["r_min"], "boxes": [list(b) for b in boxes],
                      "field_file": "fields.npz"},
        "shields": W.shields,
        "link": {"base_station": W.base_station, "repeaters": W.repeaters, "base_range": cfg["link"]["base_range"],
                 "repeater_range": cfg["link"]["repeater_range"], "coverage_file": "fields.npz",
                 "coverage_grid": {"origin": [lg.x0, lg.y0], "res": lg.res, "shape": [lg.ny, lg.nx]},
                 "covered_fraction_inside": float(cov.sum()) / max(1, int(cov.size))},
        "grid_markers": markers,
        "discrepancies": discs,
        "missing_sheet": getattr(W, "missing_sheet", None),
        "traversability": {"file": "trav.npz", "grid": grid.info(), "modes": list(MODES)},
        "route": route_out,
        "guarantees": rep,
        "blind_follow": blind,
        "budgets": budgets,
        "ip_types": types,
        "cameras": info["cameras"],
        "lights": [{"name": L.get("name"), "pos": L["pos"], "level": L["level"], "room": L["room"]} for L in W.lights],
        "ip_backends": {k: v for k, v in sorted({i["type"]: i["backend"] for i in info["ip_models"].values()}.items())},
    }
    return site
