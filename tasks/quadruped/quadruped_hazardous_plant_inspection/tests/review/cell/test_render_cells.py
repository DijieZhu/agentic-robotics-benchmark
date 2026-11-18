"""Renderer robustness of the cell across missions in one process.

    MUJOCO_GL=osmesa python -I -B tests/review/cell/test_render_cells.py cells SITE [SITE ...] [--missions 8]
        [--thermal 10] [--lifecycle verify|stub|late-release] [--json OUT]
    MUJOCO_GL=osmesa python -I -B tests/review/cell/test_render_cells.py inject SITE [--json OUT]
    MUJOCO_GL=osmesa python -I -B tests/review/cell/test_render_cells.py parity SITE [--json OUT]

`cells`: `--missions` consecutive missions in this one process (the sites cycled), each a fresh PlantCellServer with
the grader's run settings, each taking `--thermal` thermal frames between navigation and inspection frames and steps
(the head moving), through `handle` (the wire's replies). The lifecycle of the cells:
  verify        each cell closed at the end of its mission, as tests/verify.py does (_close_cell);
  stub          each cell dropped without close() and the cyclic garbage collector run after every thermal frame
                (an earlier test stub over the source tree's world closed nothing);
  late-release  each cell kept unclosed and released (with a collection) right after the next mission's first
                thermal frame: an old cell's renderers are finalised while the live thermal renderer's context is
                current (the mechanism behind that test stub's 6th and 7th cell; mission/render_guard.py).
Every thermal frame is checked: an answer (no error), a finite image in 0-200 degC, and an independent check of the
segmentation the frame was computed from: about 300 sub-pixels with a uniform 3 x 3 neighbourhood are compared with
mujoco.mj_ray from the camera (same geom groups, hits nearer than the near clip skipped): the share that agrees.
`inject`: in one cell, the live thermal renderer's GL objects are deleted the way an unsafe close does it (a plain
mujoco.Renderer on the same model closed while the thermal context is current); the next thermal frame must still be
right (the guard re-renders) and equal, bit for bit, the same frame of a second cell without the injection; then a
read-back that stays garbage on both attempts must answer `sensor_unavailable` and change nothing (frame count, next
frame id), and the cell's record must count it (render_failures, frames_lost, render_incidents).
`parity`: the guarded segmentation equals mujoco.Renderer's bit for bit on 12 views; two cells on one site with one
program give bit-identical thermal frames; the cost of a thermal frame (handle, encoding included).
"""

import argparse
import base64
import gc
import hashlib
import json
import math
import os
import sys
import time

TESTS = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, TESTS)

import numpy as np  # noqa: E402

from plantcell.client import decode_thermal  # noqa: E402
from plantcell.server import PlantCellServer  # noqa: E402

import mujoco  # noqa: E402
import bricks.sensors.thermal as bth  # noqa: E402

GEOM = int(mujoco.mjtObj.mjOBJ_GEOM)
REC = {}
_orig_seg = bth.segmentation


def _gcfg():
    with open(os.path.join(TESTS, "grader_config.json")) as f:
        return json.load(f)


def ray_agreement(model, data, cam_id, width, height, opt, ids, typ, step=24):
    """(agreeing, compared) sub-pixels of a segmentation image against mj_ray from the camera."""
    fovy = math.radians(float(model.cam_fovy[cam_id]))
    f = (height / 2.0) / math.tan(fovy / 2.0)
    cx, cy = (width - 1) / 2.0, (height - 1) / 2.0
    R = data.cam_xmat[cam_id].reshape(3, 3) @ np.diag([1.0, -1.0, -1.0])
    p = np.array(data.cam_xpos[cam_id], float)
    gg = np.array([int(opt.geomgroup[i]) for i in range(6)], np.uint8)
    znear = float(model.vis.map.znear) * float(model.stat.extent)
    gid = np.zeros(1, np.int32)
    agree = n = 0
    for v in range(step // 2, height - 1, step):
        for u in range(step // 2, width - 1, step):
            bi, bt = ids[v - 1:v + 2, u - 1:u + 2], typ[v - 1:v + 2, u - 1:u + 2]
            if not ((bi == bi[1, 1]).all() and (bt == bt[1, 1]).all()):
                continue
            d = R @ np.array([(u - cx) / f, (v - cy) / f, 1.0])
            zf = 1.0 / np.linalg.norm(d)
            d = d * zf
            dist = mujoco.mj_ray(model, data, p, d, gg, 1, -1, gid)
            if 0 <= dist and dist * zf < znear:
                continue                      # inside the near clip: the render cannot show it
            ray_g = int(gid[0]) if dist >= 0 else -1
            seg_g = int(ids[v, u]) if int(typ[v, u]) == GEOM else -1
            n += 1
            agree += int(ray_g == seg_g)
    return agree, n


def recording_segmentation(pool, data, cam_id, width, height, scene_option=None):
    REC.clear()
    try:
        ids, typ = _orig_seg(pool, data, cam_id, width, height, scene_option)
    except Exception as e:  # noqa: BLE001 - recorded, then the cell handles it as it does
        REC["error"] = "%s: %s" % (type(e).__name__, str(e)[:160])
        raise
    a, n = ray_agreement(pool.model, data, cam_id, width, height, scene_option, ids, typ)
    REC.update({"agree": a, "n": n, "max_id": int(ids.max()), "ngeom": int(pool.model.ngeom)})
    return ids, typ


def check_frame(rep, rows, mission, k, min_agree=0.85, rec=None):
    rec = REC if rec is None else rec
    row = {"mission": mission, "k": k}
    if not isinstance(rep, dict) or "error" in rep:
        row["error"] = (rep or {}).get("error") if isinstance(rep, dict) else repr(rep)
        row["seg_error"] = rec.get("error")
        row["ok"] = False
        rows.append(row)
        return row
    img = decode_thermal(rep)
    row.update({"frame_id": rep.get("frame_id"), "t_min": round(float(np.min(img)), 2),
                "t_max": round(float(np.max(img)), 2), "agree": rec.get("agree"), "n": rec.get("n"),
                "sha": hashlib.sha256(base64.b64decode(rep["thermal_ck_png_b64"])).hexdigest()[:16]})
    share = (rec.get("agree", 0) / rec["n"]) if rec.get("n") else 0.0
    row["share"] = round(share, 3)
    row["ok"] = bool(np.isfinite(img).all() and 0.0 <= row["t_min"] and row["t_max"] <= 200.0
                     and rec.get("n", 0) >= 50 and share >= min_agree)
    rows.append(row)
    return row


def call(srv, method, **params):
    return srv.handle(method, params)


# ============================================================================================== cells
def cmd_cells(a):
    bth.segmentation = recording_segmentation
    gcfg = _gcfg()
    rows, missions, keep = [], [], []
    t0 = time.time()
    for k in range(a.missions):
        site = a.sites[k % len(a.sites)]
        srv = PlantCellServer(config=gcfg, mode="hidden")
        srv.reset(site, seed=1000 + k, kind="hidden")
        call(srv, "set_head", pan=0.0, tilt=0.15)
        call(srv, "step", n_ticks=50)
        for i in range(a.thermal):
            call(srv, "set_head", pan=0.6 * math.sin(0.9 * i), tilt=0.1 + 0.25 * math.cos(1.3 * i))
            call(srv, "step", n_ticks=25)
            if i % 3 == 0:
                call(srv, "get_camera", name="nav", depth=True)
            elif i % 3 == 1:
                call(srv, "get_camera", name="inspect")
            else:
                call(srv, "get_depth")
            r = check_frame(call(srv, "get_thermal"), rows, k + 1, i + 1)
            if a.lifecycle == "late-release" and i == 0 and keep:
                keep.clear()
                gc.collect()
            elif a.lifecycle == "stub":
                gc.collect()
            if not r["ok"] and a.verbose:
                print(json.dumps(r), flush=True)
        call(srv, "finish")
        s = srv.finalize()
        missions.append({"mission": k + 1, "site": os.path.basename(site), "render_failures": s.get("render_failures"),
                         "frames_lost": s.get("frames_lost"), "render_retries": s.get("render_retries"),
                         "internal_errors": s.get("internal_errors"), "thermal_frames": s["frames"].get("thermal"),
                         "incidents": s.get("render_incidents")})
        print(json.dumps({k2: v for k2, v in missions[-1].items() if k2 != "incidents"}), flush=True)
        if a.lifecycle == "verify":
            srv.close()
        elif a.lifecycle == "late-release":
            keep.append(srv)
        srv = None
    keep.clear()
    bad = [r for r in rows if not r["ok"]]
    shares = [r["share"] for r in rows if r.get("share") is not None and r["ok"]]
    res = {"lifecycle": a.lifecycle, "missions": a.missions, "thermal_frames": len(rows), "bad_frames": len(bad),
           "bad": bad[:20], "agreement_min_ok": min(shares) if shares else None,
           "agreement_median": float(np.median(shares)) if shares else None,
           "missions_detail": missions, "wall_s": round(time.time() - t0, 1)}
    ok = not bad and all(not m["render_failures"] and not m["frames_lost"] and not m["internal_errors"]
                         for m in missions)
    return ok, res


# ============================================================================================== inject
def _first_frames(site, inject):
    srv = PlantCellServer(config=_gcfg(), mode="hidden")
    srv.reset(site, seed=77, kind="hidden")
    call(srv, "set_head", pan=0.2, tilt=0.2)
    call(srv, "step", n_ticks=60)
    f1 = call(srv, "get_thermal")
    call(srv, "get_camera", name="nav", depth=False)
    call(srv, "set_head", pan=-0.3, tilt=0.3)
    call(srv, "step", n_ticks=40)
    info = {}
    if inject:
        w = srv.world
        ss = int(w.cam_models["thermal"].supersample)
        key = (w.cam_models["thermal"].width * ss, w.cam_models["thermal"].height * ss)
        rth = w.pool._r[key]
        foreign = mujoco.Renderer(w.model, 64, 64)             # a plain renderer, closed the unsafe way
        rth._gl_context.make_current()                          # ... while the live thermal context is current
        mujoco.Renderer.close(foreign)
        info["injected"] = type(rth).__name__
    REC.clear()
    f2 = call(srv, "get_thermal")
    info["seg"] = dict(REC)
    return srv, f1, f2, info


def cmd_inject(a):
    bth.segmentation = recording_segmentation
    problems = []
    site = a.sites[0]
    srv, f1, f2, info = _first_frames(site, inject=True)
    ref, g1, g2, ginfo = _first_frames(site, inject=False)
    rows = []
    r2 = check_frame(f2, rows, 1, 2, rec=info["seg"])
    if not r2["ok"]:
        problems.append("the frame after the injection is wrong: %s" % r2)
    elif r2.get("sha") != check_frame(g2, [], 2, 2, rec=ginfo["seg"]).get("sha"):
        problems.append("the re-rendered frame differs from the same frame of a clean cell")
    s1 = srv.finalize()
    inc = s1.get("render_incidents") or []
    if not any(x.get("what") == "segmentation_rerendered" for x in inc):
        problems.append("no re-render recorded after the injection (incidents %s)" % inc)
    # a read-back that stays garbage: sensor_unavailable, nothing changes
    persistent = {}
    try:
        from mission.render_guard import SafeRenderer
    except ImportError:
        SafeRenderer = None
        problems.append("no mission.render_guard in this cell")
    if SafeRenderer is not None:
        cell = PlantCellServer(config=_gcfg(), mode="hidden")
        cell.reset(site, seed=78, kind="hidden")
        call(cell, "step", n_ticks=30)
        ok1 = call(cell, "get_thermal")
        used0 = call(cell, "get_state")["frames_used"]["thermal"]
        orig = SafeRenderer._segment_raw
        SafeRenderer._segment_raw = lambda self: np.full((self._height, self._width), 0xD33470, np.int64)
        try:
            bad = call(cell, "get_thermal")
        finally:
            SafeRenderer._segment_raw = orig
        used1 = call(cell, "get_state")["frames_used"]["thermal"]
        ok2 = call(cell, "get_thermal")
        s2 = cell.finalize()
        persistent = {"reply": bad, "frames_used_before": used0, "frames_used_after": used1,
                      "frame_ids": [ok1.get("frame_id"), ok2.get("frame_id")], "render_failures": s2["render_failures"],
                      "frames_lost": s2["frames_lost"], "incidents": s2.get("render_incidents")}
        if bad.get("error") != "sensor_unavailable":
            problems.append("persistent garbage answered %s" % bad)
        if used1 != used0:
            problems.append("the failed frame was counted (%s -> %s)" % (used0, used1))
        n1, n2 = int(ok1["frame_id"][1:]), int(ok2["frame_id"][1:])
        if n2 != n1 + 1:
            problems.append("frame ids %s then %s: the failed frame took an id" % (ok1["frame_id"], ok2["frame_id"]))
        if s2["render_failures"] != 1 or s2["frames_lost"] != 1:
            problems.append("record: render_failures %s, frames_lost %s" % (s2["render_failures"], s2["frames_lost"]))
        if not any(x.get("what") == "segmentation_failed" for x in (s2.get("render_incidents") or [])):
            problems.append("the failure is not in render_incidents")
        cell.close()
    srv.close()
    ref.close()
    return not problems, {"injected": info, "frame_after": r2, "incidents": inc, "persistent": persistent,
                          "problems": problems}


# ============================================================================================== parity
def cmd_parity(a):
    problems = []
    site = a.sites[0]
    srv = PlantCellServer(config=_gcfg(), mode="hidden")
    srv.reset(site, seed=5, kind="hidden")
    w = srv.world
    tm = w.cam_models["thermal"]
    ss = int(tm.supersample)
    W, H = tm.width * ss, tm.height * ss
    plain = mujoco.Renderer(w.model, height=H, width=W)

    class _Pool(object):
        model = w.model

        def get(self, width, height):
            return plain
    same = 0
    for i in range(12):
        call(srv, "set_head", pan=0.8 * math.sin(i), tilt=0.1 + 0.3 * math.cos(0.7 * i))
        call(srv, "step", n_ticks=20)
        mujoco.mj_kinematics(w.model, w.data)
        ia, ta = bth.segmentation(w.pool, w.data, w.cam_ids["thermal"], W, H, w.scene_opt)
        ib, tb = bth.segmentation(_Pool(), w.data, w.cam_ids["thermal"], W, H, w.scene_opt)
        same += int(np.array_equal(ia, ib) and np.array_equal(ta, tb) and ia.dtype == ib.dtype)
    plain.close()
    if same != 12:
        problems.append("guarded segmentation differs from mujoco.Renderer's in %d of 12 views" % (12 - same))
    srv.close()
    # determinism: two cells, one program, the same frames
    shas, cost = [], []
    for run in range(2):
        c = PlantCellServer(config=_gcfg(), mode="hidden")
        c.reset(site, seed=11, kind="hidden")
        h = hashlib.sha256()
        for i in range(10):
            call(c, "set_head", pan=0.5 * math.sin(i), tilt=0.2)
            call(c, "step", n_ticks=15)
            t = time.perf_counter()
            r = call(c, "get_thermal")
            if i:
                cost.append(time.perf_counter() - t)
            h.update(base64.b64decode(r["thermal_ck_png_b64"]))
        shas.append(h.hexdigest())
        c.close()
    if shas[0] != shas[1]:
        problems.append("two cells gave different thermal frames")
    cost.sort()
    res = {"views_identical": same, "frames_sha256": shas, "thermal_call_s": {
        "n": len(cost), "median": round(cost[len(cost) // 2], 4), "p95": round(cost[int(0.95 * (len(cost) - 1))], 4),
        "max": round(cost[-1], 4)}, "problems": problems}
    return not problems, res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("cells", "inject", "parity"))
    ap.add_argument("sites", nargs="+")
    ap.add_argument("--missions", type=int, default=8)
    ap.add_argument("--thermal", type=int, default=10)
    ap.add_argument("--lifecycle", default="verify", choices=("verify", "stub", "late-release"))
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)
    ok, res = {"cells": cmd_cells, "inject": cmd_inject, "parity": cmd_parity}[a.cmd](a)
    print(json.dumps({k: v for k, v in res.items() if k not in ("missions_detail",)}, indent=1, default=str)[:6000])
    print("render %s%s: %s" % (a.cmd, " (%s)" % a.lifecycle if a.cmd == "cells" else "", "OK" if ok else "FAIL"))
    if a.json:
        with open(a.json, "w") as f:
            json.dump(res, f, indent=1, default=str)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
