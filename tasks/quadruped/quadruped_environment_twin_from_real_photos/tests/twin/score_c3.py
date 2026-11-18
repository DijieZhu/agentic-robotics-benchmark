"""C3: the site's geometry in the robot's terms.

    C3 = C1 x (w_slice slice + w_depth depth + w_up up x min(1, depth / up_depth_full)) / (w_slice + w_depth + w_up)

This module returns the UNGATED value; the orchestrator multiplies it by C1. The slice weight is dropped (and the
others renormalised) when the truth scores fewer than two classes.

slice. The robot-terms map (payload twin_slice.py, the same code the truth was built with) of the agent's twin as
  submitted (units where they were put, gates at their photographed position, the gate's excluded units removed),
  computed at the TRUE cells mapped into the agent's frame by the final alignment map and in the agent's own units:
  the agent's geometry is moved rigidly into the frame "site / s*" (x -> R x + t / s*), where the true cells are the
  site cells divided by s* and the reference level is the true floor height divided by s*; the robot's sizes stay
  absolute. Scored cells = slice_truth's eligible cells (region, seen by >= 2 photos, scanned, robust under the truth
  perturbations) that are decisive (within decisive_m of a class boundary of the truth OR of the agent's map, or with
  a true clearance below clearance_decisive_m) and whose true class is one of the scored classes (those with at least
  min_class_cells scored truth cells, fixed by the truth). BA = mean over the K scored classes of the recall (the
  share of a class's scored cells that the agent's map gives that class); slice = max(0, (BA - 1/K) / (1 - 1/K)):
  a one-class map (empty floor, solid block) scores 0.
depth. Every c3 sample ray (true camera, true pixel), mapped into the agent's frame, is cast against the agent's
  static collision geometry; the hit mapped back to the site frame (at s*) has a distance along the true normal to the
  true surface point; a sample scores band_down(|distance|, band_m) (1 at 2 cm, 0 at 6 cm), no hit 0. depth = the
  mean over the environment classes present of the class's mean.
up. band_down(up_tilt, up_band_deg) (1 at 0.5 deg, 0 at 3 deg), counted in proportion min(1, depth / up_depth_full)
  (no credit for a level frame without structure).
"""
import json
import math

import numpy as np

from . import config, raycast
from .truth import TruthError

FORMAT_TRUTH = "site-slice-truth/1"
# keys of C3.slice the stored truth must have been built with (a mismatch is a task error: rebuild the truth)
TRUTH_KEYS = ("cell_m", "low_from_m", "support_tol_m", "h_crouch_m", "h_stand_m", "r_robot_m", "clear_cap_m",
              "decisive_m", "clearance_decisive_m", "min_class_cells", "perturb_height_m", "perturb_footprint_m",
              "min_photos", "ground_classes")


def slice_geoms(tw):
    """The collision geoms the map counts: the environment, the units the gate kept and the gates (a unit the gate
    excluded is deleted from the physics and from the map). Sorted ids."""
    gone = set()
    g = list(tw.static_geoms())
    for u in tw.units:
        if u["status"] == "ok":
            g += u["geoms"]
        else:
            gone.update(u["geoms"])
    for gt in tw.gates:
        g += gt["geoms"]
    return sorted(set(int(v) for v in g) - gone)


def slice_truth(ctx):
    """The stored truth map, checked against this configuration (TruthError on any mismatch)."""
    st = ctx.truth.load("slice_truth")
    sec = ctx.sec["slice"]
    if str(st.get("format", "")) != FORMAT_TRUTH:
        raise TruthError("slice_truth.npz has format %r, not %r" % (str(st.get("format")), FORMAT_TRUTH))
    try:
        params = json.loads(str(st["params"]))
    except (KeyError, ValueError) as exc:
        raise TruthError("slice_truth.npz has no readable params: %s" % exc)
    built = params.get("slice", {})
    bad = [k for k in TRUTH_KEYS if built.get(k) != sec.get(k)]
    if bad:
        raise TruthError("slice_truth.npz was built with other C3.slice values (%s): rebuild it" % ", ".join(bad))
    nx, ny = int(st["nx"]), int(st["ny"])
    for k in ("floor_z", "true_class", "eligible", "decisive_truth"):
        if st[k].shape != (nx, ny):
            raise TruthError("slice_truth.npz: %s has shape %s, not (%d, %d)" % (k, st[k].shape, nx, ny))
    if abs(float(st["cell_m"]) - float(sec["cell_m"])) > 0:
        raise TruthError("slice_truth.npz: cell size %.4f differs from C3.slice.cell_m" % float(st["cell_m"]))
    return st


def agent_map(ctx, st):
    """The agent's class map on the true cells ((nx, ny) int8) and the twin_slice.SliceMap."""
    TS = ctx.payload["twin_slice"]
    if TS is None or not hasattr(TS, "map_shapes"):
        raise TruthError("the payload has no twin_slice module")
    tw = ctx.twin
    mp = ctx.align.map
    sp = ctx.sec["slice"]
    shapes = TS.twin_shapes(tw.model, tw.data, slice_geoms(tw))
    grid = TS.Grid(np.asarray(st["origin_xy"], float), float(st["cell_m"]), int(st["nx"]), int(st["ny"]))
    zref = np.asarray(st["floor_z"], np.float32).astype(np.float64)
    if not mp.is_identity():
        s = float(mp.s)
        shapes = TS.transform_shapes(shapes, mp.R, mp.t / s)
        grid = grid.scaled(1.0 / s)
        zref = zref / s
    sm = TS.map_shapes(shapes, grid, zref, TS.params_from(sp))
    return sm.classes, sm


def score_slice(ctx, st, agent_cls):
    TS = ctx.payload["twin_slice"]
    sp = ctx.sec["slice"]
    tc = np.asarray(st["true_class"], np.int8)
    elig = np.asarray(st["eligible"], bool)
    dec_t = np.asarray(st["decisive_truth"], bool)
    K = [int(k) for k in np.asarray(st["scored_classes"]).ravel()]
    names = TS.CLASS_NAMES
    if len(K) < 2:
        return None, {"scorable": False, "why": "fewer than 2 classes have %d scored truth cells (%s)" % (
            int(sp["min_class_cells"]), {names[k]: int(v) for k, v in enumerate(st["class_cells"]) if k}),
            "scored_classes": [names[k] for k in K]}
    dec_a = TS.near_boundary(agent_cls, float(st["cell_m"]), float(sp["decisive_m"]))
    scored = elig & (dec_t | dec_a) & np.isin(tc, K)
    rec = {}
    conf = {}
    for k in K:
        sel = scored & (tc == k)
        n = int(sel.sum())
        hit = int(np.sum(sel & (agent_cls == k)))
        rec[k] = hit / n if n else 0.0
        conf[names[k]] = {names[j]: int(np.sum(sel & (agent_cls == j))) for j in range(4)}
    BA = float(sum(rec[k] for k in K)) / len(K)
    chance = 1.0 / len(K)
    val = max(0.0, (BA - chance) / (1.0 - chance))
    detail = {"scorable": True, "value": val, "balanced_accuracy": BA, "scored_classes": [names[k] for k in K],
              "recall": {names[k]: rec[k] for k in K},
              "n_scored": {names[k]: int(np.sum(scored & (tc == k))) for k in K},
              "n_scored_truth_only": {names[k]: int(np.sum(elig & dec_t & (tc == k))) for k in K},
              "confusion_true_by_agent": conf,
              "agent_cells_in_region": {names[j]: int(np.sum(np.asarray(st["in_region"], bool) & (agent_cls == j)))
                                        for j in range(4)}}
    return val, detail


def _band_down(x, full, zero):
    """band_down elementwise (1 at or below full, 0 at or above zero; non-finite 0)."""
    x = np.asarray(x, float)
    out = np.clip((zero - x) / (zero - full), 0.0, 1.0)
    return np.where(np.isfinite(x), out, 0.0)


def score_depth(ctx):
    dsec = ctx.sec["depth"]
    full, zero = (float(v) for v in dsec["band_m"])
    c3 = ctx.truth.c3
    n = ctx.truth.n_c3
    if n == 0:
        return 0.0, {"n": 0, "why": "no c3 samples"}
    mp = ctx.align.map
    O = np.asarray(c3["origin"], float)
    U = np.asarray(c3["direction"], float)
    U = U / np.linalg.norm(U, axis=1)[:, None]
    X = np.asarray(c3["point"], float)
    Nn = np.asarray(c3["normal"], float)
    Nn = Nn / np.linalg.norm(Nn, axis=1)[:, None]
    cls = np.asarray(c3["class"], int)
    names = [str(v) for v in np.asarray(c3["classes"]).ravel()]
    scene = ctx.static_scene()
    if mp.is_identity():
        dist, _o, _s, _n = raycast.cast_grouped(scene, O, U)
        lam = dist
    else:
        ua = mp.dir_to_agent(U)
        ua = ua / np.linalg.norm(ua, axis=1)[:, None]
        dist, _o, _s, _n = raycast.cast_grouped(scene, mp.to_agent(O), ua)
        lam = mp.s * dist
    hit = dist >= 0
    P = O + np.where(hit, lam, 0.0)[:, None] * U
    r = np.where(hit, np.abs(np.einsum("ij,ij->i", Nn, P - X)), np.inf)
    b = _band_down(r, full, zero)
    per = {}
    for k, nm in enumerate(names):
        sel = cls == k
        m = int(sel.sum())
        if not m:
            continue
        rk = r[sel & hit]
        per[nm] = {"n": m, "mean": float(np.mean(b[sel])), "hit_share": float(np.mean(hit[sel])),
                   "median_abs_m": float(np.median(rk)) if len(rk) else None,
                   "within_full_share": float(np.mean(r[sel] <= full))}
    if not per:
        return 0.0, {"n": n, "why": "no class has samples"}
    val = float(sum(v["mean"] for v in per.values())) / len(per)
    return val, {"value": val, "n": n, "per_class": per, "band_m": [full, zero]}


def score(ctx):
    sec = ctx.sec
    st = slice_truth(ctx)
    agent_cls, sm = ctx.cache("C3_slice_map", lambda: agent_map(ctx, st))
    sl, sl_detail = score_slice(ctx, st, agent_cls)
    dp, dp_detail = score_depth(ctx)
    ub = [float(v) for v in sec["up_band_deg"]]
    tilt = float(ctx.align.up_tilt_deg)
    up = config.band_down(tilt, ub[0], ub[1])
    full_d = float(sec.get("up_depth_full", 0.0) or 0.0)
    up_factor = min(1.0, dp / full_d) if full_d > 0 else 1.0
    up_eff = up * up_factor
    ws = float(sec["w_slice"]) if sl is not None else 0.0
    wd, wu = float(sec["w_depth"]), float(sec["w_up"])
    tot = ws + wd + wu
    val = (ws * (sl or 0.0) + wd * dp + wu * up_eff) / tot if tot > 0 else 0.0
    detail = {"slice": sl_detail, "depth": dp_detail,
              "up": {"tilt_deg": tilt, "band_deg": ub, "band": up, "depth_factor": up_factor, "value": up_eff},
              "weights": {"slice": ws, "depth": wd, "up": wu}, "components": {
                  "slice": sl, "depth": dp, "up": up_eff},
              "map": {"grid": sm.grid.as_dict(), "n_shapes": sm.info.get("n_shapes"),
                      "n_shapes_used": sm.info.get("n_shapes_used")}}
    diag = "slice %s, depth %.3f, up %.3f (tilt %.2f deg)" % (
        "%.3f (BA %.3f over %d classes)" % (sl, sl_detail["balanced_accuracy"], len(sl_detail["scored_classes"]))
        if sl is not None else "not scored", dp, up_eff, tilt)
    return {"value": float(min(1.0, max(0.0, val))), "detail": detail, "diagnostic": diag}


def selftest(ctx, result):
    """On the true twin: its map at the stored cells is exactly the stored true map, and each component is 1."""
    st = slice_truth(ctx)
    agent_cls, _sm = ctx.cache("C3_slice_map", lambda: agent_map(ctx, st))
    tc = np.asarray(st["true_class"], np.int8)
    if agent_cls.shape != tc.shape or not np.array_equal(agent_cls, tc):
        diff = int(np.sum(agent_cls != tc)) if agent_cls.shape == tc.shape else -1
        raise TruthError("the true twin's robot-terms map differs from slice_truth.npz in %d cells: rebuild the truth"
                         % diff)
    comp = (result or {}).get("detail", {}).get("components", {})
    bad = {k: v for k, v in comp.items() if v is not None and v != 1.0}
    if bad:
        raise TruthError("C3 components of the true twin are not 1: %s" % bad)
    names = ctx.payload["twin_slice"].CLASS_NAMES
    return {"cells": int(tc.size), "scored_classes": [names[int(v)] for v in np.asarray(st["scored_classes"]).ravel()],
            "components": comp}
