#!/usr/bin/env python3
"""QC Q7 (task version 2.0): the heaps' depth in a site's plate bundle against the compiled collision model, by the
one-sided measure (gated: art nearer than the collision surface; the other side is a reading).

    q7_plate_depth.py --site PKG --bundle DIR --b3 DIR [--plate-npz PLATE.npz] --out OUT.json [--name NAME]

PKG is a site package (a .tar.xz archive or a folder with site.json and site.xml); DIR the site's bundle folder
(<plate_root>/<sha256>/: meta.json, depth4.npz); the b3 folder of the same bake (pixel_fields.npz and b3_meta.json:
which art layer the 1x ray meets first; not in the bundle); PLATE.npz the plate job's own distance AOV
(DistanceToImagePlaneSD on the same 4x film; not in the bundle). Nothing is written but OUT.json; the console carries
the neutral name and pass / fail.

Rays: the compiled model of the package's site.xml (MuJoCo, no robot), mj_multiRay from the camera's TRUE pose
(site.json camera.true.T_world_camera) through every pixel centre of the 4x film (fx, fy x 4, principal point
4 c + 1.5), in the union of the heaps' windows (the projection of each heap's rectangle x [0, top] +- 16 px); the
depth compared is along the optical axis (as the plates store it).
Retained pixels (the plate bake's definition, unchanged; the 2 px bands are not widened): on the 4x film, not
counted are 2 px on both sides of the depth edges (adjacent depth pixels differing by > 10 mm), 2 px on both sides of
the heaps' outline (the boundary of the compiled heap mask; for (1) also of the art heap mask), and pixels whose first
art hit is not a heap (b3 layer code 'heap', repeated 4 x 4).
  (1) near (gate <= 0.1 %): among the pixels whose first ART hit is a heap and whose compiled ray hits anything, the
      share where the art is NEARER the camera than the compiled hit by > 10 mm (seen but not touchable); on the
      bundle's depth4 and, given --plate-npz, again on the plate job's distance AOV (its own depth edges).
  (2) toe (gate >= 99 %): pixels where both the art and the compiled ray hit a heap and the compiled hit is <= 0.03 m
      high: the share with |art - compiled| <= 10 mm. Its off pixels are counted by side (readings): the art nearer
      the camera than the compiled hit by > 10 mm, or farther (the compiled surface the higher one).
  (3) readings, per heap, not gates: both-hit-a-heap within 10 mm (the two-sided share), the share where the art is
      FARTHER by > 10 mm, and the excluded shares (depth edges, outline, art not a heap).
  (4) under the slab (gate 0): whole 4x film: compiled rays that hit a box of the rubble under the slab
      (rubble__void_debris*), and plate pixels (depth4; the AOV when given) whose back-projected point lies inside such
      a box (+ 5 mm).
Per heap, (1) counts the pixels whose compiled ray hits that heap, plus those whose compiled ray hits no heap and whose
art point (back-projected) lies within 0.05 m of that heap's rectangle.
Binding sanity (not Q9): the bundle's meta.json site_json_sha256 / site_xml_sha256 equal the package's members and its
camera equals the package's true camera (1e-9).
"""
import argparse
import hashlib
import io
import json
import os
import sys
import tarfile

import numpy as np
from scipy import ndimage

TOOL_VERSION = "1.1.0"
EDGE_M = 0.010
TOL_M = 0.010
TOE_Z = 0.03
BAND_PX = 2
WIN_PX = 16
NEAR_LINE = 0.001
TOE_LINE = 0.99
DEBRIS = "rubble__void_debris"


def sha(b):
    return hashlib.sha256(b).hexdigest()


def members(pkg):
    if os.path.isdir(pkg):
        return {n: open(os.path.join(pkg, n), "rb").read() for n in ("site.json", "site.xml")}
    out = {}
    with tarfile.open(pkg, mode="r:*") as tf:
        for ti in tf.getmembers():
            b = os.path.basename(ti.name)
            if ti.isreg() and b in ("site.json", "site.xml"):
                out[b] = tf.extractfile(ti).read()
    return out


def band(mask):
    return ndimage.binary_dilation(mask, iterations=BAND_PX)


def depth_edges(Dp):
    e = np.zeros(Dp.shape, bool)
    a = np.abs(np.diff(Dp, axis=1)) > EDGE_M
    e[:, :-1] |= a
    e[:, 1:] |= a
    b = np.abs(np.diff(Dp, axis=0)) > EDGE_M
    e[:-1, :] |= b
    e[1:, :] |= b
    return band(e)


def outline(M):
    return band(M & ~ndimage.binary_erosion(M, iterations=1))


def cast(m, d, t, R, fx, fy, cx, cy, ys, xs, mujoco):
    dirc = np.stack([(xs - cx) / fx, (ys - cy) / fy, np.ones(len(xs))], 1)
    dirc /= np.linalg.norm(dirc, axis=1, keepdims=True)
    dirw = dirc @ R.T
    n = len(xs)
    gid = np.full(n, -1, np.int32)
    dist = np.full(n, -1.0)
    pnt = np.asarray(t, np.float64)
    for s0 in range(0, n, 250000):
        s1 = min(n, s0 + 250000)
        g = np.zeros(s1 - s0, np.int32)
        di = np.zeros(s1 - s0, np.float64)
        mujoco.mj_multiRay(m, d, pnt, np.ascontiguousarray(dirw[s0:s1].ravel()), None, True, -1, g, di, None, s1 - s0,
                           50.0)
        gid[s0:s1] = g
        dist[s0:s1] = di
    return gid, dist, dirc, dirw


def backproject(Dp, R, t, fx, fy, cx, cy):
    H, W = Dp.shape
    v, u = np.mgrid[0:H, 0:W]
    pc = np.stack([(u - cx) / fx * Dp, (v - cy) / fy * Dp, Dp], -1)
    return pc @ R.T + t


def measure(Dart, name, ctx, own_edge=None):
    """(1), (2), (3) of one art depth (the bundle's depth4 or the plate AOV)."""
    HM, HIT, ZR, ZH, GID, art_heap, outl, outl_art, hg, Pw_of, rects = (ctx[k] for k in (
        "HM", "HIT", "ZR", "ZH", "GID", "art_heap", "outl", "outl_art", "hg", "Pw_of", "rects"))
    edge = depth_edges(Dart) if own_edge is None else own_edge
    c1 = art_heap & (Dart > 0) & HIT & ~edge & ~outl_art
    near = c1 & ((ZR - Dart) > TOL_M)
    Pw = Pw_of(Dart)
    per = {}
    for gid, h in hg.items():
        x0, x1, y0, y1 = rects[gid]
        close = (Pw[..., 0] >= x0 - 0.05) & (Pw[..., 0] <= x1 + 0.05) & (Pw[..., 1] >= y0 - 0.05) & (Pw[..., 1] <= y1 + 0.05)
        own = c1 & ((GID == gid) | (~HM & close))
        cand = HM & (GID == gid) & (Dart > 0)
        keep = cand & art_heap & ~edge & ~outl
        diff = Dart - ZR
        ad = np.abs(diff)
        toe = keep & (ZH <= TOE_Z)
        nc = max(int(cand.sum()), 1)
        per[h["name"]] = dict(
            cls=h["cls"], near_px=int((near & own).sum()), near_of_px=int(own.sum()),
            near_share=round(float((near & own).sum() / max(own.sum(), 1)), 6),
            toe_px=int(toe.sum()), toe_within=int((toe & (ad <= TOL_M)).sum()),
            toe_share=round(float((toe & (ad <= TOL_M)).sum() / max(toe.sum(), 1)), 5),
            toe_off_art_nearer=int((toe & (diff < -TOL_M)).sum()), toe_off_art_farther=int((toe & (diff > TOL_M)).sum()),
            heap_px=int(cand.sum()), kept_px=int(keep.sum()),
            two_sided_share=round(float((keep & (ad <= TOL_M)).sum() / max(keep.sum(), 1)), 5),
            art_farther_share=round(float((keep & (diff > TOL_M)).sum() / max(keep.sum(), 1)), 5),
            art_nearer_share=round(float((keep & (diff < -TOL_M)).sum() / max(keep.sum(), 1)), 5),
            excluded=dict(depth_edges=round(float((cand & edge).sum() / nc), 4),
                          outline=round(float((cand & outl & ~edge).sum() / nc), 4),
                          art_not_heap=round(float((cand & ~art_heap & ~edge & ~outl).sum() / nc), 4),
                          total=round(float((cand & ~keep).sum() / nc), 4)))
    cand = HM & (Dart > 0)
    keep = cand & art_heap & ~edge & ~outl
    ad = np.abs(Dart - ZR)
    toe = keep & (ZH <= TOE_Z)
    n1 = max(int(c1.sum()), 1)
    out = dict(source=name,
               near=dict(px=int(c1.sum()), near_px=int(near.sum()), share=round(float(near.sum() / n1), 6),
                         line=NEAR_LINE, pass_=bool(near.sum() / n1 <= NEAR_LINE),
                         of_which_compiled_not_heap=int((near & ~HM).sum())),
               toe=dict(px=int(toe.sum()), within=int((toe & (ad <= TOL_M)).sum()),
                        share=round(float((toe & (ad <= TOL_M)).sum() / max(toe.sum(), 1)), 5), line=TOE_LINE,
                        off_art_nearer=int((toe & ((Dart - ZR) < -TOL_M)).sum()),
                        off_art_farther=int((toe & ((Dart - ZR) > TOL_M)).sum()),
                        pass_=bool((toe & (ad <= TOL_M)).sum() / max(toe.sum(), 1) >= TOE_LINE)),
               two_sided=dict(kept_px=int(keep.sum()), share=round(float((keep & (ad <= TOL_M)).sum() / max(keep.sum(), 1)), 5),
                              art_farther_share=round(float((keep & ((Dart - ZR) > TOL_M)).sum() / max(keep.sum(), 1)), 5)),
               per_heap=per)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--site", required=True)
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--b3", required=True)
    ap.add_argument("--plate-npz")
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", default="site")
    a = ap.parse_args(argv)
    import mujoco
    mem = members(a.site)
    site = json.loads(mem["site.json"])
    m = mujoco.MjModel.from_xml_string(mem["site.xml"].decode("utf-8"))
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    meta = json.load(open(os.path.join(a.bundle, "meta.json")))
    K = site["camera"]["model"]
    Tw = np.asarray(site["camera"]["true"]["T_world_camera"], float)
    R, t = Tw[:3, :3], Tw[:3, 3]
    S4 = 4
    fx, fy = S4 * K["fx"], S4 * K["fy"]
    cx, cy = S4 * K["cx"] + 1.5, S4 * K["cy"] + 1.5
    W4, H4 = S4 * K["width"], S4 * K["height"]
    mc = meta.get("camera", {})
    binding = dict(site_json=meta.get("site_json_sha256") == sha(mem["site.json"]),
                   site_xml=meta.get("site_xml_sha256") == sha(mem["site.xml"]),
                   camera=bool(all(abs(float(mc.get(k, np.nan)) - float(K[k])) <= 1e-9 for k in ("fx", "fy", "cx", "cy"))
                               and np.allclose(np.asarray(mc.get("T_world_camera"), float), Tw, atol=1e-9, rtol=0)),
                   supersample=meta.get("supersample") == S4)
    z4 = np.load(os.path.join(a.bundle, "depth4.npz"), allow_pickle=False)
    D4 = z4["depth4"].astype(np.float64)
    if D4.shape != (H4, W4):
        raise SystemExit("q7: depth4 shape %s is not the 4x film %s" % (D4.shape, (H4, W4)))
    heaps = (site.get("heaps") or {}).get("heaps") or []
    hg, rects = {}, {}
    for h in heaps:
        gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, h["name"])
        if gid < 0:
            raise SystemExit("q7: no compiled geom for a heap of the package")
        hg[gid] = h
        rects[gid] = tuple(map(float, h["rect"]))
    region = np.zeros((H4, W4), bool)
    for h in heaps:
        x0, x1, y0, y1 = h["rect"]
        Cn = np.array([[x, y, z] for x in (x0, x1) for y in (y0, y1) for z in (0.0, h["top_m"])])
        pc = (Cn - t) @ R
        u = fx * pc[:, 0] / pc[:, 2] + cx
        v = fy * pc[:, 1] / pc[:, 2] + cy
        region[max(0, int(v.min()) - WIN_PX):min(H4, int(v.max()) + WIN_PX + 1),
               max(0, int(u.min()) - WIN_PX):min(W4, int(u.max()) + WIN_PX + 1)] = True
    ys, xs = np.nonzero(region)
    gid, dist, dirc, dirw = cast(m, d, t, R, fx, fy, cx, cy, ys, xs, mujoco)
    hit = dist >= 0
    HM = np.zeros((H4, W4), bool)
    HM[ys[hit & np.isin(gid, list(hg))], xs[hit & np.isin(gid, list(hg))]] = True
    HIT = np.zeros((H4, W4), bool)
    HIT[ys[hit], xs[hit]] = True
    ZR = np.zeros((H4, W4))
    ZR[ys, xs] = np.where(hit, dist * dirc[:, 2], 0.0)
    ZH = np.full((H4, W4), np.nan)
    ZH[ys, xs] = np.where(hit, t[2] + dist * dirw[:, 2], np.nan)
    GID = np.full((H4, W4), -1, np.int32)
    GID[ys, xs] = gid
    pf = np.load(os.path.join(a.b3, "pixel_fields.npz"), allow_pickle=False)
    bm = json.load(open(os.path.join(a.b3, "b3_meta.json")))
    code = bm["pixel_layer_codes"]["heap"]
    art_heap = np.repeat(np.repeat(pf["layer"] == code, S4, 0), S4, 1) & region
    outl = outline(HM)
    ctx = dict(HM=HM, HIT=HIT, ZR=ZR, ZH=ZH, GID=GID, art_heap=art_heap, outl=outl, outl_art=outl | outline(art_heap),
               hg=hg, rects=rects, Pw_of=lambda Dp: backproject(Dp, R, t, fx, fy, cx, cy))
    res = dict(tool="q7_plate_depth %s" % TOOL_VERSION, name=a.name, binding=binding, mujoco=mujoco.__version__,
               rays_window=int(len(xs)), heaps=len(heaps),
               definition="one-sided measure; retained pixels as the plate bake defines them (4x film, 2 px bands)")
    res["bundle_depth4"] = measure(D4, "bundle depth4.npz", ctx)
    DP = None
    if a.plate_npz:
        zp = np.load(a.plate_npz, allow_pickle=False)
        DP = zp["DistanceToImagePlaneSD"].astype(np.float64)
        if DP.ndim == 3:
            DP = DP[..., 0]
        DP = np.where(np.isfinite(DP), DP, 0.0)
        if DP.shape != (H4, W4):
            raise SystemExit("q7: plate AOV shape %s is not the 4x film" % (DP.shape,))
        res["plate_aov"] = measure(DP, "plate job distance AOV", ctx)
        res["plate_aov"]["no_hit_px_whole_frame"] = int((DP <= 0).sum())
    # (4) under the slab, whole 4x film
    deb = {}
    for i in range(m.ngeom):
        nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i) or ""
        if nm.startswith(DEBRIS):
            deb[i] = nm
    boxes = [tuple(map(float, b["aabb"])) for b in site.get("boxes", []) if str(b.get("name", "")).startswith(DEBRIS)]
    yy, xx = np.mgrid[0:H4, 0:W4]
    g_all, d_all, _dc, _dw = cast(m, d, t, R, fx, fy, cx, cy, yy.ravel(), xx.ravel(), mujoco)
    under = dict(boxes=len(boxes), compiled_rays_hit=int(np.isin(g_all[d_all >= 0], list(deb)).sum()))

    def inside_boxes(Dp):
        P = backproject(Dp, R, t, fx, fy, cx, cy)
        ok = Dp > 0
        n = 0
        for b in boxes:
            e = 0.005
            n += int((ok & (P[..., 0] >= b[0] - e) & (P[..., 0] <= b[1] + e) & (P[..., 1] >= b[2] - e) &
                      (P[..., 1] <= b[3] + e) & (P[..., 2] >= b[4] - e) & (P[..., 2] <= b[5] + e)).sum())
        return n
    under["depth4_px_in_debris_boxes"] = inside_boxes(D4)
    if DP is not None:
        under["plate_aov_px_in_debris_boxes"] = inside_boxes(DP)
    under["pass_"] = bool(under["compiled_rays_hit"] == 0 and under["depth4_px_in_debris_boxes"] == 0 and
                          under.get("plate_aov_px_in_debris_boxes", 0) == 0)
    res["under_slab"] = under
    gates = dict(near_depth4=res["bundle_depth4"]["near"]["pass_"], toe_depth4=res["bundle_depth4"]["toe"]["pass_"],
                 under_slab=under["pass_"])
    if DP is not None:
        gates["near_plate_aov"] = res["plate_aov"]["near"]["pass_"]
    res["gates"] = gates
    res["pass_"] = bool(all(gates.values()))
    with open(a.out + ".part", "w") as fh:
        json.dump(res, fh, indent=1, sort_keys=True)
    os.replace(a.out + ".part", a.out)
    print("q7 %s: %s" % (a.name, "pass" if res["pass_"] else "FAIL"))
    return 0 if res["pass_"] else 1


if __name__ == "__main__":
    sys.exit(main())
