"""What a dressing pass must keep clear (version 2, the dressing's checks call these), from a site
package's site.json alone (run-time code: numpy only, no authoring module).

  instrument_faces(site)     every instrument (listed, decoy, plain): its position, the face point the evidence
                             rules measure to (a rising-stem valve's plate 0.42 m above its pipe), the 0.3 m to keep
  sight_segments(site, ...)  segments from the camera of sampled stored margin-region cells (the accepted reading
                             places, standing and crouching; the robot's head as the site records it) to the listed
                             instrument's face point: nothing new may cut one
  floor_keepout(site)        floor rects where nothing visual-only may stand: every traversal element's footprint and
                             version 2's approach zones (1.0 m of flat floor before and behind an in-room strip / rack)
  segments_hit_box(C, P, lo, hi)   which segments meet an axis-aligned box (for a decor item's bounds)
  keepclear(site)            the three together in the form the dressing asked for: {"geoms": [names],
                             "segments": (A, B) (camera point -> face point, world m), "floor_rects": [[x0, y0, x1, y1]]}
"""

from __future__ import annotations

import math

import numpy as np

FACE_DZ = {"valve_handwheel": 0.42}          # the indicator plate above the pipe the position names
CLEAR_M = 0.30


def instrument_faces(site):
    """[(name, kind, (x, y, z) face point, clearance m)] of every instrument of the site."""
    out = []
    for r in site.get("ips", []):
        if r.get("type") in ("dose_point", "hotspot"):
            continue
        p = list(r["pose"]["pos"])
        p[2] += FACE_DZ.get(r["type"], 0.0)
        out.append((r.get("body") or "ip/%s" % r["tag"], "listed", tuple(float(v) for v in p), CLEAR_M))
    for r in site.get("decoys", []):
        p = list(r["pose"]["pos"])
        p[2] += FACE_DZ.get(r["type"], 0.0)
        out.append((r.get("body") or "ip/%s" % r["tag"], "decoy", tuple(float(v) for v in p), CLEAR_M))
    for r in site.get("plain_instruments", []):
        p = list(r["pos"])
        p[2] += FACE_DZ.get(r["type"], 0.0)
        out.append((r.get("body", ""), "plain", tuple(float(v) for v in p), CLEAR_M))
    return out


def _mask(runs, shape):
    """A stored region (row runs [iy, ix0, ix1] with ix1 inclusive, site_gen.util.runs_from_mask) as a bool grid."""
    m = np.zeros((int(shape[0]), int(shape[1])), bool)
    for iy, a, b in runs:
        m[int(iy), int(a):int(b) + 1] = True
    return m


def sight_segments(site, n_per_item=300, seed=0):
    """(C, P, tags): N cameras and N face points (N x 3 each) of up to `n_per_item` stored margin cells per listed
    instrument (standing and crouching), the camera where the robot's camera is when it stands in the cell facing the
    instrument (site.json readable_region.window.camera), and the tag of each segment."""
    g = np.random.default_rng(seed)
    C, P, T = [], [], []
    for r in site.get("ips", []):
        if r.get("type") in ("dose_point", "hotspot") or not r.get("readable_region"):
            continue
        rr = r["readable_region"]
        gi = rr["grid"]
        cam = (rr.get("window") or {}).get("camera") or {}
        face = np.asarray(r["pose"]["pos"], float).copy()
        face[2] += FACE_DZ.get(r["type"], 0.0)
        x0, y0 = gi["origin"]
        for key, hk in (("runs_stand", "stand_height_m"), ("runs_crouch", "crouch_height_m")):
            m = _mask(rr.get(key, []), gi["shape"])
            iy, ix = np.nonzero(m)
            if not len(iy):
                continue
            k = g.choice(len(iy), size=min(len(iy), int(n_per_item)), replace=False)
            bx = x0 + (ix[k] + 0.5) * gi["res"]
            by = y0 + (iy[k] + 0.5) * gi["res"]
            h = float(cam.get(hk, 0.3))
            ahead = float(cam.get("mast_x_m", 0.165)) + float(cam.get("cam_fwd_m", 0.0155))
            for x, y in zip(bx, by):
                yaw = math.atan2(face[1] - y, face[0] - x)
                C.append((x + ahead * math.cos(yaw), y + ahead * math.sin(yaw), h))
                P.append(tuple(face))
                T.append(r["tag"])
    return np.asarray(C, float).reshape(-1, 3), np.asarray(P, float).reshape(-1, 3), T


def floor_keepout(site):
    """[(what, [x0, y0, x1, y1])]: traversal element footprints and version 2's approach zones."""
    out = []
    for e in site.get("elements", []):
        fp = np.asarray(e["footprint"], float)
        out.append(("element %s %s" % (e["id"], e["kind"]), [float(fp[:, 0].min()), float(fp[:, 1].min()),
                                                            float(fp[:, 0].max()), float(fp[:, 1].max())]))
    for k, pair in enumerate(((site.get("ladder") or {}).get("approach_zones") or [])):
        for nm, r in zip(("front", "behind"), pair):
            out.append(("approach %d %s" % (k, nm), [float(v) for v in r]))
    return out


def keepclear(site, n_per_item=300, seed=0):
    """The dressing's request: the geoms to keep clear (every instrument's, decoy's, plain instrument's and grid
    marker's geoms as the site records them: faces, text plates, brackets), the sight segments of the stored margin
    cells (sight_segments) and the floor rects (floor_keepout), from site.json alone."""
    geoms = []
    for r in list(site.get("ips", [])) + list(site.get("decoys", [])) + list(site.get("plain_instruments", [])):
        geoms += list((r.get("model_names") or {}).get("geoms") or [])
    for m in site.get("grid_markers", []):
        geoms += list((m.get("model_names") or {}).get("geoms") or [])
    A, B, _ = sight_segments(site, n_per_item, seed)
    return {"geoms": sorted(set(geoms)), "segments": (A, B), "floor_rects": [r for _, r in floor_keepout(site)]}


def segments_hit_box(C, P, lo, hi):
    """Bool per segment C[i] -> P[i]: does it meet the box [lo, hi] (slab test)?"""
    C = np.asarray(C, float)
    D = np.asarray(P, float) - C
    lo = np.asarray(lo, float)[None, :]
    hi = np.asarray(hi, float)[None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = np.where(np.abs(D) > 1e-12, 1.0 / D, np.inf)
        t0 = (lo - C) * inv
        t1 = (hi - C) * inv
    tmin = np.nanmax(np.where(np.isnan(np.minimum(t0, t1)), -np.inf, np.minimum(t0, t1)), axis=1)
    tmax = np.nanmin(np.where(np.isnan(np.maximum(t0, t1)), np.inf, np.maximum(t0, t1)), axis=1)
    inside = np.all((C >= lo) & (C <= hi), axis=1)
    return inside | ((tmax >= np.maximum(tmin, 0.0)) & (tmin <= 1.0))
