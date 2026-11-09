#!/usr/bin/env python3
"""check_twin: check your environment twin the way the grader's gate sees it. It scores nothing.

    python3 check_twin.py [SUBMISSION]                  # default /home/user/submission
    python3 check_twin.py --settle                      # + the standard settle (the G0 blow-up check, drift, support)
    python3 check_twin.py --probes                      # + the probe protocol on your twin (implies --settle)
    python3 check_twin.py --probes Butter_1 drawer      # only these probes ('drawer' = the pull-out)
    python3 check_twin.py --overlay overlay.png         # segmentation edges of photo_cam drawn on the photo
    python3 check_twin.py --json                        # everything as one JSON document on stdout

What it does:
  1. runs twin_gate.gate(), the same code as the grader's validity gate G0, and prints PASS / FAIL with every reason,
     then each item with its status (ok, or excluded with the reasons), mass, density, friction and collision pieces;
  2. decodes photo_cam into OpenCV intrinsics (fx, fy, cx, cy; pixel (0, 0) is the centre of the top-left pixel) and
     prints the world's up direction in camera coordinates (x right, y down, z forward) and the camera's pose;
  3. --settle: the standard simulation of twin_sim.py (twin_sim_config.json) from qpos0, with excluded items removed
     first, exactly as graded: per item the largest displacement and rotation over the settle, the static flag, the
     initial penetrations, the support edges, and the blow-up check that is part of G0;
  4. --probes: from the settled state, every lift probe and the drawer pull-out of the published protocol, with the
     outcome of every other item (carried / rides, stayed, disturbed, intermediate);
  5. --overlay: renders a segmentation of the twin through photo_cam with MuJoCo (OSMesa on the CPU, or EGL when
     MUJOCO_GL=egl) and draws the item outlines on the photo.

Everything printed comes from your submission and the public rules. Nothing here reads, or can tell you anything
about, the true scene. Lengths are in your twin's units (metres; printed in mm). Exit code 0 when the gate passes
(and, with --settle, the settle does not blow up), 1 otherwise, 2 for a usage error.
"""
import argparse
import copy
import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
if os.environ.get("MUJOCO_GL") == "osmesa":
    os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")

import numpy as np  # noqa: E402

import twin_camera  # noqa: E402
import twin_gate  # noqa: E402

DEFAULT_SUB = "/home/user/submission"
DEFAULT_PHOTO = os.path.join(HERE, "photo.png")


# ---------------------------------------------------------------------------------------------------------------
# helpers

def _mm(v):
    return round(1000.0 * float(v), 3)


def _r(v, n=4):
    return None if v is None else round(float(v), n)


def camera_summary(model, cam_name):
    """photo_cam as OpenCV intrinsics, its pose, and the world's up direction in camera coordinates."""
    import mujoco
    cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
    if cid < 0:
        return None
    d = mujoco.MjData(model)
    mujoco.mj_forward(model, d)
    out = {"name": cam_name}
    try:
        K, W, H = twin_camera.decode(model, cid)
        out.update({"fx": _r(K[0, 0], 3), "fy": _r(K[1, 1], 3), "cx": _r(K[0, 2], 3), "cy": _r(K[1, 2], 3),
                    "W": int(W), "H": int(H),
                    "hfov_deg": _r(2 * math.degrees(math.atan2(W / 2.0, K[0, 0])), 3),
                    "vfov_deg": _r(2 * math.degrees(math.atan2(H / 2.0, K[1, 1])), 3)})
    except Exception as exc:  # noqa: BLE001 - twin_camera.CameraError
        out["decode_error"] = str(exc)
    R_cv, p = twin_camera.camera_frame(model, d, cid)
    up = R_cv.T @ np.array([0.0, 0.0, 1.0])
    fwd = R_cv[:, 2]
    out.update({"position_world_m": [_r(v, 6) for v in p],
                "R_world_from_cv": [[_r(v, 6) for v in row] for row in R_cv],
                "world_up_in_camera": [_r(v, 6) for v in up],
                "optical_axis_world": [_r(v, 6) for v in fwd],
                "optical_axis_below_horizontal_deg": _r(math.degrees(math.asin(max(-1.0, min(1.0, -fwd[2])))), 3),
                "roll_deg": _r(math.degrees(math.atan2(-up[0], -up[1])), 3)})
    return out


def physics_model(res, excluded):
    """The model the standard simulation runs on: the gate's grader copy with the excluded items deleted (an
    excluded item can neither support nor disturb anything). Uses the gate's own helper when it has one."""
    helper = getattr(twin_gate, "physics_model", None)
    if callable(helper):
        return helper(res)
    if not excluded:
        return copy.deepcopy(res.model)
    spec = res.spec.copy()
    for nm in excluded:
        b = spec.body(nm)
        if b is not None:
            spec.delete(b)
    return spec.compile()


def drawer_summary(model, drawer, cam_name):
    import mujoco
    if not drawer:
        return None
    b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, drawer)
    if b < 0 or int(model.body_jntnum[b]) != 1:
        return {"body": drawer}
    j = int(model.body_jntadr[b])
    d = mujoco.MjData(model)
    mujoco.mj_forward(model, d)
    axis = np.asarray(d.xaxis[j], float)
    out = {"body": drawer, "joint": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j),
           "axis_world": [_r(v, 6) for v in axis],
           "axis_tilt_from_horizontal_deg": _r(math.degrees(math.asin(max(-1.0, min(1.0, abs(axis[2]))))), 3),
           "limited": bool(model.jnt_limited[j]),
           "range_mm": [_mm(v) for v in model.jnt_range[j]] if model.jnt_limited[j] else None}
    cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
    if cid >= 0:
        R_cv, _p = twin_camera.camera_frame(model, d, cid)
        a_cam = R_cv.T @ axis
        toward = twin_gate.drawer_out_sign(model, d, b, j, cam_name) > 0     # from the drawer's geometry centre
        out["axis_in_camera"] = [_r(v, 6) for v in a_cam]
        out["pull_out_direction"] = "+axis" if toward else "-axis"
    return out


def overlay(model, photo_path, out_path, cam_name, items, drawer, scale=1.0, all_geoms=False):
    """Segmentation through photo_cam (collision geoms only unless all_geoms), outlines drawn on the photo."""
    import mujoco
    m = copy.deepcopy(model)
    W, H = (int(v) for v in m.cam_resolution[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)])
    w, h = max(16, int(round(W * scale))), max(16, int(round(H * scale)))
    m.vis.global_.offwidth = max(int(m.vis.global_.offwidth), w)
    m.vis.global_.offheight = max(int(m.vis.global_.offheight), h)
    m.vis.map.znear = min(float(m.vis.map.znear), 0.005)
    for g in range(m.ngeom):
        visual = m.geom_contype[g] == 0 and m.geom_conaffinity[g] == 0
        m.geom_group[g] = 5 if visual else 0
        if not visual:
            m.geom_rgba[g, 3] = 1.0
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    opt = mujoco.MjvOption()
    opt.geomgroup[:] = 0
    opt.geomgroup[0] = 1
    if all_geoms:
        opt.geomgroup[5] = 1
    opt.sitegroup[:] = 0
    t0 = time.time()
    r = mujoco.Renderer(m, height=h, width=w)
    try:
        r.enable_segmentation_rendering()
        r.update_scene(d, camera=cam_name, scene_option=opt)
        seg = r.render().copy()
    finally:
        r.close()
    t_render = time.time() - t0
    geom_ids = seg[:, :, 0]
    is_geom = seg[:, :, 1] == int(mujoco.mjtObj.mjOBJ_GEOM)
    root = np.asarray(m.body_rootid, int)
    item_ids = {}
    for k, nm in enumerate(items):
        b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, nm)
        if b >= 0:
            item_ids[int(b)] = k + 1
    dbody = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, drawer) if drawer else -1
    lut = np.zeros(max(1, m.ngeom), int)
    for g in range(m.ngeom):
        rb = int(root[int(m.geom_bodyid[g])])
        if rb in item_ids:
            lut[g] = item_ids[rb]
        else:
            b = int(m.geom_bodyid[g])
            in_drawer = False
            while b > 0 and dbody >= 0:
                if b == dbody:
                    in_drawer = True
                    break
                b = int(m.body_parentid[b])
            lut[g] = 100000 if in_drawer else 200000 + g     # each furniture geom keeps its own outline
    lab = np.zeros(geom_ids.shape, int)
    ok = is_geom & (geom_ids >= 0) & (geom_ids < m.ngeom)
    lab[ok] = lut[geom_ids[ok]]
    try:
        import cv2
        photo = cv2.imread(photo_path, cv2.IMREAD_COLOR)
        if photo is None:
            raise FileNotFoundError(photo_path)
        photo = cv2.cvtColor(photo, cv2.COLOR_BGR2RGB)
        if photo.shape[:2] != (h, w):
            photo = cv2.resize(photo, (w, h), interpolation=cv2.INTER_AREA)
    except ImportError:
        from PIL import Image
        photo = np.asarray(Image.open(photo_path).convert("RGB").resize((w, h)))
        cv2 = None
    img = photo.astype(np.float32)
    edge = np.zeros(lab.shape, bool)
    edge[:, 1:] |= lab[:, 1:] != lab[:, :-1]
    edge[1:, :] |= lab[1:, :] != lab[:-1, :]
    edge[:, :-1] |= edge[:, 1:].copy()
    edge[:-1, :] |= edge[1:, :].copy()
    rng = np.random.default_rng(7)
    pal = rng.integers(60, 256, size=(len(items) + 2, 3)).astype(np.float32)
    colour = np.zeros(lab.shape + (3,), np.float32)
    for k in range(1, len(items) + 1):
        colour[lab == k] = pal[k]
    colour[lab >= 200000] = (255.0, 255.0, 255.0)
    colour[lab == 100000] = (255.0, 200.0, 0.0)
    tint = (lab > 0) & (lab <= len(items))
    img[tint] = 0.75 * img[tint] + 0.25 * colour[tint]
    img[edge] = np.where(lab[edge][:, None] > 0, colour[edge], 255.0)
    img = np.clip(img, 0, 255).astype(np.uint8)
    visible = {}
    for k, nm in enumerate(items):
        mask = lab == k + 1
        n = int(mask.sum())
        visible[nm] = int(round(n / (scale * scale)))
        if n and cv2 is not None:
            ys, xs = np.nonzero(mask)
            cv2.putText(img, nm, (int(xs.mean()) - 4 * len(nm), int(ys.mean())), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45 * max(scale, 0.5), (255, 255, 255), 1, cv2.LINE_AA)
    if cv2 is not None:
        cv2.imwrite(out_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    else:
        from PIL import Image
        Image.fromarray(img).save(out_path)
    return {"out": os.path.abspath(out_path), "size": [w, h], "render_s": round(t_render, 2),
            "visible_px_full_res": visible,
            "furniture_px": int(round(int((lab >= 200000).sum()) / (scale * scale))),
            "drawer_px": int(round(int((lab == 100000).sum()) / (scale * scale))),
            "geoms": "collision and visual" if all_geoms else "collision only"}


# ---------------------------------------------------------------------------------------------------------------
# main

def run(a):
    out = {"format": "check-twin-report/1", "submission": os.path.abspath(a.submission)}
    t0 = time.time()
    vocab = a.vocab or os.path.join(HERE, "product_names.json")
    res = twin_gate.gate(a.submission, vocabulary=vocab)
    rep = res.report
    out["gate"] = {"pass": bool(res.ok), "version": rep.get("gate_version"), "reasons": rep.get("reasons", []),
                   "notes": rep.get("warnings", []), "stats": rep.get("stats", {}), "seconds": round(time.time() - t0, 2)}
    out["items"] = rep.get("items", [])
    out["excluded"] = rep.get("excluded", [])
    status = 0 if res.ok else 1
    if res.model is None:
        return out, status
    cam_name = twin_gate.DEFAULT_RULES["camera"]
    out["camera"] = camera_summary(res.model, cam_name)
    out["drawer"] = drawer_summary(res.model, rep.get("drawer"), cam_name)
    out["drawer_travel"] = rep.get("drawer_travel")
    names = [it["body"] for it in out["items"] if it.get("status") == "ok"]
    if a.overlay:
        photo = a.photo or DEFAULT_PHOTO
        try:
            out["overlay"] = overlay(res.model, photo, a.overlay, cam_name, names, rep.get("drawer"),
                                     scale=a.overlay_scale, all_geoms=a.overlay_all)
        except Exception as exc:  # noqa: BLE001 - a rendering problem must not hide the gate result
            out["overlay"] = {"error": "%s: %s" % (type(exc).__name__, str(exc)[:400]),
                              "hint": "MuJoCo renders with MUJOCO_GL=osmesa (CPU) or egl; set it before the call"}
    if not (a.settle or a.probes is not None):
        return out, status
    if not res.ok:
        out["settle"] = {"ran": False, "why": "the gate failed"}
        return out, status
    import twin_sim
    cfg = twin_sim.load_config()
    try:
        m = physics_model(res, out["excluded"])
        hold = {}
        for h in a.hold or []:
            nm, _, frame = h.partition("=")
            hold[nm] = frame or "world"
        roles = twin_sim.standardize(m, config=cfg, drawer=rep.get("drawer"), hold=hold)
    except Exception as exc:  # noqa: BLE001 - twin_sim.TwinSimError and model problems
        out["settle"] = {"ran": False, "why": "%s: %s" % (type(exc).__name__, str(exc)[:400])}
        return out, 1
    blowup = None
    if callable(getattr(twin_gate, "blowup_check", None)):
        t1 = time.time()
        try:
            chk = twin_gate.blowup_check(physics_model(res, out["excluded"]), drawer=rep.get("drawer"))
        except Exception as exc:  # noqa: BLE001
            chk = {"fail": True, "error": "%s: %s" % (type(exc).__name__, str(exc)[:400])}
        blowup = dict(chk, wall_s=round(time.time() - t1, 2))
        out["blowup_check"] = blowup
    t1 = time.time()
    st = twin_sim.settle(m, roles, config=cfg)
    if blowup is not None:
        blow = bool(blowup.get("fail"))
    else:
        blow = bool(st.get("exploded")) or not bool(st.get("finite"))
    out["settle"] = {"ran": True, "seconds_simulated": st["seconds"], "steps": st["steps"],
                     "wall_s": round(time.time() - t1, 2), "finite": st["finite"], "exploded": st["exploded"],
                     "blow_up": blow, "mujoco_warnings": st["warnings"], "items": st["items"],
                     "initial_penetration_mm": st["initial_penetration_mm"],
                     "initial_penetration_partner": st["initial_penetration_partner"],
                     "support_edges": st.get("support_edges", []),
                     "held": sorted(hold)}
    if blow:
        status = 1
    if a.probes is not None and not blow:
        want = list(a.probes)
        bodies = [b for b in want if b != "drawer"] if want else None
        drawer = ("out",) if (not want or "drawer" in want) else ()
        t2 = time.time()
        try:
            pr = twin_sim.run_probes(m, roles, qpos=st["qpos_end"], config=cfg, bodies=bodies, drawer=drawer)
        except Exception as exc:  # noqa: BLE001
            out["probes"] = {"error": "%s: %s" % (type(exc).__name__, str(exc)[:400])}
        else:
            out["probes"] = {"wall_s": round(time.time() - t2, 2), "runs": pr}
    return out, status


def print_report(out):
    g = out["gate"]
    print("twin_gate %s: %s  (%s, %.1f s)" % ("PASS" if g["pass"] else "FAIL", out["submission"], g.get("version"),
                                            g["seconds"]))
    for r in g["reasons"]:
        print("  FAIL  " + r)
    for w in g["notes"]:
        print("  note  " + w)
    c = out.get("camera")
    if c:
        if "fx" in c:
            print("photo_cam (OpenCV): fx %.3f  fy %.3f  cx %.3f  cy %.3f  (%d x %d; hfov %.2f deg, vfov %.2f deg)" % (
                c["fx"], c["fy"], c["cx"], c["cy"], c["W"], c["H"], c["hfov_deg"], c["vfov_deg"]))
        elif "decode_error" in c:
            print("photo_cam: cannot decode: " + c["decode_error"])
        print("  position (world, m): %s" % c["position_world_m"])
        print("  world up in camera coordinates (x right, y down, z forward): %s" % c["world_up_in_camera"])
        print("  optical axis %.2f deg below the horizontal, roll %.2f deg" % (c["optical_axis_below_horizontal_deg"],
                                                                             c["roll_deg"]))
    dr = out.get("drawer")
    if dr and "axis_world" in dr:
        print("drawer %s: slide axis (world) %s, in camera %s, pull-out = %s; axis tilt %.2f deg" % (
            dr["body"], dr["axis_world"], dr.get("axis_in_camera"), dr.get("pull_out_direction"),
            dr["axis_tilt_from_horizontal_deg"]))
    tr = out.get("drawer_travel")
    if tr:
        if tr.get("limited"):
            print("  travel: range %s mm, qpos0 %s mm, travel %s mm = %s x depth %s mm, %s mm more pull-out: %s" % (
                tr["range_mm"], tr["qpos0_mm"], tr["travel_mm"], tr["travel_over_depth"], tr["depth_mm"],
                tr["out_mm"], "OK" if tr["ok"] else "NOT MET"))
        else:
            print("  travel: the slide has no limits: NOT MET")
        for r in tr.get("reasons", []):
            print("    - " + r + " (C5's drawer term scores 0)")
    items = out.get("items", [])
    if items:
        print("items (%d listed, %d excluded):" % (len(items), len(out.get("excluded", []))))
        print("  %-22s %-18s %-9s %9s %9s %11s %7s" % ("body", "label", "status", "mass g", "kg/m^3", "friction",
                                                       "pieces"))
        for it in items:
            fr = it.get("friction")
            frs = "%.2f-%.2f" % tuple(fr) if fr else "-"
            print("  %-22s %-18s %-9s %9s %9s %11s %7s" % (
                it["body"], it.get("label"), it.get("status"),
                "%.1f" % (1000 * it["mass_kg"]) if "mass_kg" in it else "-",
                "%.0f" % it["density_kg_m3"] if "density_kg_m3" in it else "-", frs,
                it.get("n_collision_geoms", "-")))
            for r in it.get("reasons", []):
                print("      - " + r)
    ov = out.get("overlay")
    if ov:
        if "error" in ov:
            print("overlay: FAILED %s (%s)" % (ov["error"], ov["hint"]))
        else:
            hidden = [k for k, v in ov["visible_px_full_res"].items() if v == 0]
            print("overlay: %s (%s, %d x %d, %.1f s)%s" % (ov["out"], ov["geoms"], ov["size"][0], ov["size"][1],
                                                         ov["render_s"],
                                                         "; not visible in photo_cam: " + ", ".join(hidden)
                                                         if hidden else ""))
    st = out.get("settle")
    if st:
        if not st.get("ran"):
            print("settle: not run (%s)" % st.get("why"))
        else:
            bc = out.get("blowup_check")
            if bc is not None:
                why = []
                for k in ("reason", "reasons", "error"):
                    v = bc.get(k)
                    why += [str(x) for x in v] if isinstance(v, list) else ([str(v)] if v else [])
                print("G0 settle check (blow-up): %s%s" % ("FAIL" if bc.get("fail") else "pass",
                                                          (": " + "; ".join(why)) if why else ""))
            print("settle (standard simulation, %.1f s simulated, %.1f s wall): %s" % (
                st["seconds_simulated"], st["wall_s"],
                "BLOW-UP (G0 fails)" if st["blow_up"] else "no blow-up"))
            if st["mujoco_warnings"]:
                print("  MuJoCo warnings: %s" % st["mujoco_warnings"])
            print("  %-22s %12s %12s %7s %s" % ("item", "max drift mm", "max rot deg", "static", "initial penetration"))
            for nm, e in st["items"].items():
                pen = st["initial_penetration_mm"].get(nm)
                pens = "%.2f mm into %s" % (pen, st["initial_penetration_partner"][nm]) if pen else ""
                print("  %-22s %12.3f %12.3f %7s %s%s" % (nm, e["max_drift_mm"], e["max_rot_deg"],
                                                       "yes" if e["static"] else "NO", pens,
                                                       "  (held)" if e.get("held") else ""))
            if st["support_edges"]:
                print("  support edges (supporter -> supported, mean up-force / weight):")
                for e in st["support_edges"]:
                    print("    %s -> %s  %.2f" % (e["supporter"], e["supported"], e["weight_frac"]))
    pr = out.get("probes")
    if pr:
        if "error" in pr:
            print("probes: FAILED " + pr["error"])
        else:
            print("probes (%.1f s wall); every other item's outcome, stayed items not listed:" % pr["wall_s"])
            for name, p in pr["runs"].items():
                moved = {k: v for k, v in p["outcomes"].items() if v["outcome"] != "stayed"}
                head = "  %-28s" % name
                if not p["finite"]:
                    print(head + " NON-FINITE")
                    continue
                if not moved:
                    print(head + " every other item stayed")
                    continue
                print(head + " " + "; ".join("%s %s (|d| %.1f mm, %.1f deg)" % (
                    k, v["outcome"], float(np.linalg.norm(v["d_mm"])), v["rot_deg"]) for k, v in sorted(moved.items())))


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return [_jsonable(v) for v in x.tolist()]
    if isinstance(x, (np.floating, float)):
        v = float(x)
        return v if math.isfinite(v) else str(v)
    if isinstance(x, (np.integer,)):
        return int(x)
    return x


def main(argv=None):
    ap = argparse.ArgumentParser(description="Check an environment twin against the public gate and run the standard "
                                             "simulation on it. Scores nothing and reads no ground truth.")
    ap.add_argument("submission", nargs="?", default=DEFAULT_SUB)
    ap.add_argument("--settle", action="store_true", help="run the standard settle (part of G0: the blow-up check)")
    ap.add_argument("--probes", nargs="*", default=None, metavar="BODY",
                    help="run the probe protocol from the settled state (all probes, or the item bodies named; "
                         "'drawer' = the pull-out)")
    ap.add_argument("--hold", nargs="*", default=None, metavar="BODY[=drawer]",
                    help="hold these items kinematically (as the grader holds items that cannot rest in the true "
                         "scene); '=drawer' makes a held item follow the drawer")
    ap.add_argument("--overlay", metavar="OUT.png", help="draw photo_cam segmentation edges on the photo")
    ap.add_argument("--overlay-scale", type=float, default=1.0, help="render size relative to photo_cam (default 1)")
    ap.add_argument("--overlay-all", action="store_true", help="draw visual geoms too (default: collision geoms)")
    ap.add_argument("--photo", default=None, help="the photo (default: photo.png next to this script)")
    ap.add_argument("--vocab", default=None, help="product_names.json (default: the file next to this script)")
    ap.add_argument("--json", action="store_true", help="print one JSON document instead of the text report")
    a = ap.parse_args(argv)
    if not 0.05 <= a.overlay_scale <= 1.0:
        ap.error("--overlay-scale must be in [0.05, 1]")
    out, status = run(a)
    if a.json:
        runs = (out.get("probes") or {}).get("runs")
        if runs:
            for p in runs.values():
                p.pop("qpos", None)
        print(json.dumps(_jsonable(out), indent=1))
    else:
        print_report(out)
        if out["gate"]["pass"] and "settle" not in out:
            print("G0 also requires the standard settle not to blow up: run with --settle.")
    return status


if __name__ == "__main__":
    sys.exit(main())
