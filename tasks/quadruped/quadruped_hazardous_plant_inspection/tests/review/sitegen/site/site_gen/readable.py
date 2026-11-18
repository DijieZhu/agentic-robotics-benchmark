"""Readable regions: for every IP, the grid cells from which the robot's sensor head (standing or crouched) sees
the target inside its stand-off window and angle window, with line of sight checked by ray casting in the
compiled site model (all geom groups, the target's own body excluded)."""

from __future__ import annotations

import math

import mujoco
import numpy as np

from .config import observability_windows


def windows_for(ip, windows):
    """The list of alternative windows (union) of an IP type."""
    t = ip["type"]
    if t == "gauge":
        k = "gauge_%d" % int(ip["params"].get("case_mm", 100))
        w = windows.get(k, windows["gauge_100"])
    else:
        w = windows.get(t)
    return list(w) if isinstance(w, (list, tuple)) else [w]


def window_for(ip, windows):
    """The primary (first) window."""
    return windows_for(ip, windows)[0]


def target_points(ip):
    """(points, normal) the camera must see: offset slightly out of the surface toward the viewer."""
    s = ip["slot"]
    p = np.asarray(s["pos"], float)
    yaw, pitch = s["yaw"], s.get("pitch", 0.0)
    n = np.array([math.cos(yaw) * math.cos(pitch), math.sin(yaw) * math.cos(pitch), math.sin(pitch)])
    prm = ip["params"]
    t = ip["type"]
    bricks = str(ip.get("backend", "")).startswith("bricks")
    if bricks and t == "gauge":
        pts = [p + 0.03 * n]
    elif bricks and t == "valve_lever":
        pts = [p + (prm.get("pipe_r", 0.04) + 0.16) * n]
    elif bricks and t == "valve_handwheel":
        pts = [p + np.array([0, 0, 0.42]) + 0.06 * n]
    elif bricks and t == "lamps":
        pts = [p + 0.03 * n]
    elif t == "gauge":
        pts = [p + 0.008 * n]
    elif t == "valve_lever":
        ax = np.asarray(prm.get("pipe_axis", [1, 0, 0]), float)
        if abs(ax[2]) / max(np.linalg.norm(ax), 1e-9) > 0.7:     # vertical pipe: the lever is in front of it
            pts = [p + (prm.get("pipe_r", 0.04) + 0.13) * n]
        else:
            pts = [p + np.array([0, 0, prm.get("pipe_r", 0.04) + 0.11]) + 0.03 * n]
    elif t == "valve_handwheel":
        pts = [p + np.array([0, 0, prm.get("pipe_r", 0.05) + 0.40]) + 0.03 * n]
    elif t == "level":
        z0, z1 = prm["span"]
        h = z1 - z0
        pts = [np.array([p[0], p[1], z]) + 0.035 * n for z in (z0 + 0.15 * h, 0.5 * (z0 + z1), z1 - 0.15 * h)]
    elif t == "lamps":
        pts = [p + 0.02 * n]
    elif t == "thermal":
        pts = [np.asarray(u["pos"], float) + (u.get("offset", 0.13)) * n for u in ip["truth"]["units"]]
    else:
        pts = [p]
    return pts, n


def compute(model, data, grid, trav, ips, cfg):
    """Fill ip['readable'] = {'stand': bool grid, 'crouch': bool grid, 'window': ...} for every IP."""
    windows, wsrc = observability_windows(cfg)
    M = cfg["modes"]
    h_stand = float(M["walk"]["head_height"])
    h_crouch = float(M["crouch_walk"]["head_height"])
    X, Y = grid.centers()
    any_enter = trav.enter["walk"] | trav.enter["high_step"] | trav.enter["crouch_walk"]
    geomgroup = np.ones(6, np.uint8)
    for ip in ips:
        t = ip["type"]
        if t in ("dose_point", "hotspot"):
            c = np.asarray(ip["slot"]["pos"], float)
            r = float(cfg["ips"]["dose_point_radius"] if t == "dose_point" else cfg["ips"]["hotspot_radius"])
            m = ((X - c[0]) ** 2 + (Y - c[1]) ** 2 <= r * r) & any_enter
            ip["readable"] = {"stand": m.copy(), "crouch": m.copy(),
                              "window": {"radius": r, "rule": "stand within the radius of the point"}}
            continue
        wins = windows_for(ip, windows)
        dmax_all = max(float(w["d"][1]) for w in wins)
        pts, n = target_points(ip)
        c = np.mean(pts, 0)
        sel = ((X - c[0]) ** 2 + (Y - c[1]) ** 2 <= (dmax_all + 0.3) ** 2) & any_enter
        iy, ix = np.nonzero(sel)
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, ip["body"]) if ip.get("body") else -1
        res = {}
        for key, hh in (("stand", h_stand), ("crouch", h_crouch)):
            H = np.stack([X[iy, ix], Y[iy, ix], np.full(len(iy), hh)], 1)
            ok = np.zeros(len(iy), bool)
            for w in wins:
                dmin, dmax = float(w["d"][0]), float(w["d"][1])
                th_max = math.radians(float(w["off_normal_deg"]))
                el_max = math.radians(float(w["elev_deg"]))
                okw = np.ones(len(iy), bool)
                for P in pts:
                    v = H - P[None, :]
                    d = np.linalg.norm(v, axis=1)
                    horiz = np.linalg.norm(v[:, :2], axis=1)
                    cosang = (v @ n) / np.maximum(d, 1e-9)
                    elev = np.arctan2(P[2] - hh, np.maximum(horiz, 1e-6))
                    okw &= (horiz >= dmin) & (horiz <= dmax) & (cosang >= math.cos(th_max)) & (elev <= el_max) & \
                        (elev >= -1.2)
                ok |= okw
            cand = np.nonzero(ok)[0]
            vis = np.zeros(len(iy), bool)
            if len(cand):
                vis_c = np.ones(len(cand), bool)
                for P in pts:
                    v = H[cand] - P[None, :]
                    d = np.linalg.norm(v, axis=1)
                    vec = (v / d[:, None]).astype(np.float64)
                    # the exact ray test (mj_ray per ray, as mission/evidence.py first_hits: mj_multiRay's culling
                    # misses geoms of static bodies holding several geoms, MuJoCo 3.14)
                    from .accept import first_hits          # (accept puts the library root on sys.path for mission)
                    gid, dist = first_hits(model, data, P.astype(np.float64), vec, geomgroup, int(bid))
                    clear = (gid < 0) | (dist < 0) | (dist >= d - 0.04)
                    vis_c &= clear
                vis[cand] = vis_c
            m = np.zeros(grid.shape, bool)
            m[iy[vis], ix[vis]] = True
            res[key] = m
        res["window"] = {"alternatives": wins, "distance": "horizontal", "head_heights": [h_stand, h_crouch],
                         "source": wsrc}
        ip["readable"] = res
    return wsrc


def readable_states(trav, ip):
    """Node mask pieces: per mode, the cells from which the IP is readable and which the mode can enter."""
    r = ip["readable"]
    return {"walk": r["stand"] & trav.enter["walk"], "high_step": r["stand"] & trav.enter["high_step"],
            "crouch_walk": r["crouch"] & trav.enter["crouch_walk"]}
