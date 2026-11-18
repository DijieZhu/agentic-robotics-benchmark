"""Observation poses and the reference route: from the pad through every IP's observation pose and back, ordered
by nearest neighbour plus 2-opt on the (cell, mode) search costs; each leg stored as mode segments with waypoints
and the traversal classes it meets."""

from __future__ import annotations

import math

import numpy as np

from .readable import readable_states, target_points, window_for
from .config import observability_windows
from .trav import MODES

MODE_PEN = {"walk": 0.0, "high_step": 0.35, "crouch_walk": 0.6}


def observation_pose(trav, ip, dist_pad, cfg):
    """Choose the observation state (node index) of an IP; returns (node, pose dict) or (None, None)."""
    g = trav.g
    N = g.nx * g.ny
    X, Y = g.centers()
    rs = readable_states(trav, ip)
    windows, _ = observability_windows(cfg)
    t = ip["type"]
    best = None
    pts, n = target_points(ip)
    P = np.mean(pts, 0)
    if t not in ("dose_point", "hotspot"):
        win = window_for(ip, windows)
        dmin, dmax = win["d"]
        f = float(windows.get("ideal_d_fraction", 0.35))
        d_ideal = dmin + f * (dmax - dmin)
        th_max = math.radians(win["off_normal_deg"])
    for mi, m in enumerate(MODES):
        mask = rs[m]
        if not mask.any():
            continue
        iy, ix = np.nonzero(mask)
        nodes = mi * N + iy * g.nx + ix
        tp = dist_pad[nodes]
        fin = np.isfinite(tp)
        if not fin.any():
            continue
        iy, ix, nodes, tp = iy[fin], ix[fin], nodes[fin], tp[fin]
        hh = float(cfg["modes"][m]["head_height"])
        H = np.stack([X[iy, ix], Y[iy, ix], np.full(len(iy), hh)], 1)
        if t in ("dose_point", "hotspot"):
            q = np.hypot(H[:, 0] - P[0], H[:, 1] - P[1]) / 0.3
        else:
            v = H - P[None, :]
            d = np.linalg.norm(v, axis=1)
            dh = np.linalg.norm(v[:, :2], axis=1)
            ang = np.arccos(np.clip((v @ n) / np.maximum(d, 1e-9), -1, 1))
            q = np.abs(dh - d_ideal) / max(dmax - dmin, 1e-6) + 0.8 * ang / th_max
        q = q + MODE_PEN[m] + 0.3 * trav.tight[m][iy, ix] + 0.15 * trav.slow[iy, ix] + tp / 900.0
        k = int(np.argmin(q))
        if best is None or q[k] < best[0]:
            best = (float(q[k]), int(nodes[k]), m, float(H[k, 0]), float(H[k, 1]), hh)
    if best is None:
        return None, None
    _, node, m, x, y, hh = best
    yaw = math.atan2(P[1] - y, P[0] - x)
    horiz = math.hypot(P[0] - x, P[1] - y)
    tilt = math.atan2(P[2] - hh, max(horiz, 1e-6)) if t not in ("dose_point", "hotspot") else -0.6
    return node, {"pos": [x, y], "yaw": yaw, "mode": m, "head_height": hh, "head_tilt": tilt,
                  "distance": math.hypot(horiz, P[2] - hh)}


def _two_opt(order, C):
    best = list(order)

    def cost(o):
        seq = [0] + o + [0]
        return sum(C[a][b] for a, b in zip(seq[:-1], seq[1:]))
    improved = True
    bc = cost(best)
    while improved:
        improved = False
        for i in range(len(best) - 1):
            for j in range(i + 1, len(best)):
                cand = best[:i] + best[i:j + 1][::-1] + best[j + 1:]
                c = cost(cand)
                if c < bc - 1e-9:
                    best, bc = cand, c
                    improved = True
    return best, bc


def _rdp(pts, eps):
    if len(pts) < 3:
        return pts
    a, b = np.asarray(pts[0]), np.asarray(pts[-1])
    ab = b - a
    L = np.linalg.norm(ab)
    P = np.asarray(pts)
    if L < 1e-9:
        d = np.linalg.norm(P - a, axis=1)
    else:
        d = np.abs(ab[0] * (P[:, 1] - a[1]) - ab[1] * (P[:, 0] - a[0])) / L
    k = int(np.argmax(d))
    if d[k] > eps:
        left = _rdp(pts[:k + 1], eps)
        right = _rdp(pts[k:], eps)
        return left[:-1] + right
    return [pts[0], pts[-1]]


def plan(trav, G, pad_node, obs, cfg, dose_fn=None, dose_parts=None):
    """obs: [(tag, node, dwell_s)]. Returns the route dict. dose_parts: optional {name: rate function}; the travel
    dose of each is returned in route['dose_parts_uSv'] (the ladder splits the unlisted source off for calibration);
    the cells of every leg are kept in route['_cells'] (not written to site.json)."""
    srcs = [pad_node] + [o[1] for o in obs]
    D, Pred = trav.dijkstra(G, srcs)
    k = len(srcs)
    C = [[float(D[i, srcs[j]]) for j in range(k)] for i in range(k)]
    # nearest neighbour from the pad, then 2-opt on the closed tour
    left = list(range(1, k))
    order = []
    cur = 0
    while left:
        nxt = min(left, key=lambda j: C[cur][j])
        order.append(nxt)
        left.remove(nxt)
        cur = nxt
    order, total = _two_opt(order, C)
    seq = [0] + order + [0]
    g = trav.g
    legs = []
    tt = 0.0
    tl = 0.0
    dose = 0.0
    classes_used = set()
    rooms_seen = []
    parts = {k: 0.0 for k in (dose_parts or {})}
    leg_cells = []
    for a, b in zip(seq[:-1], seq[1:]):
        path = trav.path(Pred[a], srcs[a], srcs[b])
        cells = [trav.unnode(n) for n in path]
        segs = []
        cur_mode = None
        pts = []
        cls = set()
        seg_len = 0.0
        seg_t = 0.0
        prev = None
        for iy, ix, m in cells:
            x, y = g.xy(iy, ix)
            if m != cur_mode:
                if pts:
                    segs.append({"mode": cur_mode, "waypoints": _rdp(pts, 0.03), "classes": sorted(cls),
                                 "length_m": seg_len, "time_s": seg_t})
                    pts = [pts[-1]]
                else:
                    pts = []
                cur_mode = m
                cls = set()
                seg_len = 0.0
                seg_t = 0.0
            pts.append([x, y])
            c = "free"
            if m == "high_step" and not trav.enter["walk"][iy, ix]:
                c = "needs_high_step"
            elif m == "crouch_walk" and not trav.enter["walk"][iy, ix]:
                c = "needs_crouch"
            cls.add(c)
            if trav.slow[iy, ix]:
                cls.add("slow")
            if trav.tight[m][iy, ix]:
                cls.add("tight")
            if prev is not None and prev[2] == m:
                dl = math.hypot(x - prev[3], y - prev[4])
                v = max(float(trav.speed[m][iy, ix]), 1e-3)
                seg_len += dl
                seg_t += dl / v
            prev = (iy, ix, m, x, y)
        if pts:
            segs.append({"mode": cur_mode, "waypoints": _rdp(pts, 0.03), "classes": sorted(cls), "length_m": seg_len,
                         "time_s": seg_t})
        L = sum(s["length_m"] for s in segs)
        T = float(C[a][b])
        for s in segs:
            classes_used.update(s["classes"])
        if cells:
            xy = np.array([g.xy(iy, ix) for iy, ix, _ in cells])
            leg_cells.append((xy, [m for _, _, m in cells], T))
            if dose_fn is not None:
                rates = dose_fn(xy)
                # time per cell ~ leg time / cells
                dose += float(np.sum(rates) * (T / max(len(cells), 1)) / 3600.0)
            for k, fn in (dose_parts or {}).items():
                parts[k] += float(np.sum(fn(xy)) * (T / max(len(cells), 1)) / 3600.0)
        legs.append({"from": "PAD" if a == 0 else obs[a - 1][0], "to": "PAD" if b == 0 else obs[b - 1][0],
                     "time_s": T, "length_m": L, "segments": segs,
                     "modes": sorted({s["mode"] for s in segs})})
        tt += T
        tl += L
    dwell = sum(o[2] for o in obs)
    out = {"order": [obs[i - 1][0] for i in order], "legs": legs, "travel_time_s": tt, "dwell_s": dwell,
           "total_time_s": tt + dwell, "length_m": tl, "dose_travel_uSv": dose,
           "classes_used": sorted(classes_used), "_cost_matrix": C, "_cells": leg_cells}
    if dose_parts:
        out["dose_parts_uSv"] = parts
    return out
