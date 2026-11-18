#!/usr/bin/env python3
"""check_twin: check your site twin the way the grader's validity gate sees it. It scores nothing.

    python3 check_twin.py [SUBMISSION]                    # default /home/user/submission
    python3 check_twin.py --settle                        # + G0's settle check and the standard settle
    python3 check_twin.py --probes                        # + every lift probe from the settled state, the gate rule
    python3 check_twin.py --probes unit_a unit_b          # only these lift probes
    python3 check_twin.py --slice slice.png               # the robot-terms map of your twin (twin_slice.py)
    python3 check_twin.py --walk routes.json              # the probe robot walks YOUR routes in YOUR twin
    python3 check_twin.py --overlay site_007 out.png      # outlines of your geoms over the photo, through photo_007
    python3 check_twin.py --render site_007 out.png       # MuJoCo's CPU render of your twin through photo_007
    python3 check_twin.py --json                          # everything as one JSON document on stdout

What it does:
  1. runs twin_gate.gate(), the same code as the grader's validity gate G0, and prints PASS / FAIL with every reason
     and every note (warnings: invalid photo cameras, hull traps, gates whose travel rule is not met, ...);
  2. one sanity line per photo camera: your intrinsics, the angle between the image's up direction and world +z
     (about 0 for a level landscape shot, about 90 deg for a portrait shot stored as shot), the optical axis
     elevation, and the share of your twin's collision geometry in front of the camera and inside its image (a
     camera that sees none of your twin usually has the wrong convention: MuJoCo cameras look along their local -z
     with +y up, twin_camera.camera_xml writes them from OpenCV poses);
  3. the units (status, mass, density, friction, pieces, extent), the gates (the travel rule in BOTH directions of
     the slide axis, each with the travel left in the opening and in the closing direction: which one opens is the
     side on which the rail continues beyond the leaf in the photos) and the column depth (how many collision geoms
     stack over one spot; the gate caps it);
  4. --settle: G0's settle check (twin_gate.settle_check: the quasi-static relaxation, then the blow-up check; a
     unit that still breaks the speed bound is excluded and the check repeated once) and the standard settle of
     twin_sim.py on the model the grader simulates (excluded units removed): per unit the initial penetration, the
     displacement during the relaxation and during the dynamic settle, the maxima, the static flag, and the support
     edges;
  5. --probes: from the settled state, the lift probes of twin_sim.py (each unit lifted by the standard drive) with
     the outcome of every other unit (carried, disturbed, intermediate; stayed units are not listed);
  6. --slice: the robot-terms map of twin_slice.py (stand / crouch / blocked cells of your twin) as a PNG;
  7. --walk: the probe robot of ../robot/ (walk_probe.py, the frozen controller) walks the routes of a JSON file
     in your twin, with every unit held still and every gate fixed at its photographed position (as in the
     grader's walk probes); per route and start variant: reached / stopped / fell, the progress, what the robot's
     body touched;
  8. --overlay / --render: MuJoCo's CPU renderer (OSMesa) through your photo_NNN camera at most 1552 px wide:
     outlines of your collision geoms over the photo (units in colour with their names, gates orange, the
     environment white), or a plain render of your twin (its own materials and textures).

Everything printed comes from your submission and the public rules. Nothing here reads, or can tell you anything
about, the true site. Lengths are in your twin's units. Exit code 0 when the gate passes (and, with --settle, G0's
settle check passes), 1 otherwise, 2 for a usage error or a broken task file. A back end that is not available
(a module or the robot folder missing) is reported as "not available" and does not change the exit code.
"""
import argparse
import copy
import importlib
import importlib.util
import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

DEFAULT_SUB = "/home/user/submission"
MAX_RENDER_WIDTH = 1552                      # renders and overlays are at most this wide (a quarter of a photo)
# --walk: the start variants and the contact standardisation of the grader's walk probes (the same numbers as the
# contract states; used only when the payload brings no shared walk module, see walk_spec)
WALK_VARIANT_DXY_M = 0.02                    # start shifted along / left of the start heading by up to this
WALK_VARIANT_DYAW_DEG = 2.0                  # heading turned by up to this
WALK_SEED = 20261113
WALK_RUN_WALL = {"per_sim_s": 3.0, "factor": 20.0, "min_s": 120.0}   # the wall guard of a walk run (the assessment's)
WALK_CONTACT = {"contype": 1, "conaffinity": 1, "condim": 3, "friction": [0.8, 0.005, 0.0001],
                "solref": [0.02, 1.0], "solimp": [0.9, 0.95, 0.001, 0.5, 2.0], "margin_m": 0.0, "gap_m": 0.0,
                "priority": 0, "solmix": 1.0}
MODE_ALIASES = {"crouch_walk": "crouch"}


def _gl_setup(want_render):
    """MuJoCo renders on the CPU through OSMesa here (no GPU): set the backend before mujoco is imported."""
    if want_render:
        os.environ.setdefault("MUJOCO_GL", "osmesa")
    if os.environ.get("MUJOCO_GL") == "osmesa":
        os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")


# ---------------------------------------------------------------------------------------------------------------
# small helpers

def _r(v, n=4):
    if v is None:
        return None
    v = float(v)
    return round(v, n) if math.isfinite(v) else v


def _err(exc):
    return "%s: %s" % (type(exc).__name__, str(exc)[:400])


def _not_available(what, exc=None):
    out = {"available": False, "why": "%s is not available" % what}
    if exc is not None:
        out["why"] += " (%s)" % _err(exc)
    return out


def _jsonable(x):
    import numpy as np
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
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return x


def photo_number(text):
    """7, '7', '007', 'site_007', 'site_007.jpg' or 'photo_007' -> 7 (None if it is not a photo number)."""
    s = os.path.basename(str(text)).strip()
    for pre in ("site_", "photo_"):
        if s.startswith(pre):
            s = s[len(pre):]
    s = s.split(".")[0]
    return int(s) if s.isdigit() else None


def _optional_module(name):
    """A payload module next to this file, or None (with the reason) when it is missing or does not import."""
    try:
        return importlib.import_module(name), None
    except Exception as exc:  # noqa: BLE001 - a missing or broken back end must not hide the gate result
        return None, exc


# ---------------------------------------------------------------------------------------------------------------
# cameras, units, gates (from the gate report)

def camera_hints(c):
    """Plain-language flags for a photo camera's sanity line (generic: they need no knowledge of the site)."""
    s = c.get("sanity") or {}
    out = []
    if not s:
        return out
    if s.get("in_view_share", 1.0) == 0.0:
        out.append("none of your twin's geometry falls inside this camera's image%s: if the photo shows part of the "
                   "site, check the camera convention (MuJoCo cameras look along their local -z with +y up)" % (
                       " (almost all of it lies behind the camera)" if s.get("in_front_share", 1.0) < 0.05 else ""))
    if s.get("up_angle_deg", 0.0) > 135.0:
        out.append("the image's up direction points down: the camera is upside down")
    return out


def camera_rows(rep):
    rows = []
    for c in rep.get("cameras", []):
        row = {"name": c.get("name"), "photo": c.get("photo"), "status": c.get("status"),
               "reasons": c.get("reasons", [])}
        for k in ("fx", "fy", "cx", "cy", "W", "H"):
            if k in c:
                row[k] = _r(c[k], 3) if k not in ("W", "H") else int(c[k])
        if "centre" in c:
            row["centre"] = [_r(v, 4) for v in c["centre"]]
        if c.get("sanity"):
            row["sanity"] = c["sanity"]
        row["hints"] = camera_hints(c)
        rows.append(row)
    return rows


def gate_rule_both(travel, rules):
    """The gate travel rule for both opening directions of the slide axis (+axis, -axis), from the gate's own
    measurements (range, photographed position, leaf width along the axis): twin_sim.gate_rule with the numbers of
    twin_sim_config.json "gate_rule" (their only source). Per direction the travel left beyond the photographed
    position in the opening direction (open_m) and in the closing direction (close_m); without the truth the closing
    side is judged as for a gate that the photos show shut."""
    out = {}
    if not travel:
        return out
    TS, _exc = _optional_module("twin_sim")
    if TS is not None and callable(getattr(TS, "gate_rule", None)):
        for sign, key in ((1, "+axis"), (-1, "-axis")):
            r = TS.gate_rule(travel, sign)
            out[key] = {"ok": bool(r.get("ok")), "reasons": list(r.get("reasons", [])), "open_m": r.get("open_m"),
                        "close_m": r.get("close_m"), "travel_over_width": r.get("travel_over_width")}
        return out
    # the rule's numbers live in twin_sim_config.json only (twin_gate needs twin_sim.py for its own note as well)
    for key in ("+axis", "-axis"):
        out[key] = {"ok": None, "reasons": ["not available: twin_sim.py (the travel rule) is missing"]}
    return out


def gate_summary(model, rep, rules):
    import mujoco
    import numpy as np
    out = []
    d = mujoco.MjData(model)
    mujoco.mj_forward(model, d)
    for g in rep.get("gates", []):
        rec = {"body": g.get("body"), "joint": g.get("joint")}
        j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, g.get("joint") or "")
        if j >= 0:
            ax = np.asarray(d.xaxis[j], float)
            ax = ax / max(float(np.linalg.norm(ax)), 1e-12)
            rec["axis_world"] = [_r(v, 6) for v in ax]
            rec["axis_tilt_from_horizontal_deg"] = _r(math.degrees(math.asin(min(1.0, abs(float(ax[2]))))), 3)
        tr = g.get("travel") or {}
        for k in ("limited", "range_m", "qpos0_m", "width_m", "leaf_centre", "opening_sign", "rail"):
            if k in tr:
                rec[k] = tr[k]
        rec["rule_by_direction"] = gate_rule_both(tr, rules)
        rec["rule_from_rail"] = {"ok": tr.get("ok"), "reasons": tr.get("reasons", [])}
        out.append(rec)
    return out


# ---------------------------------------------------------------------------------------------------------------
# --settle and --probes (twin_sim.py)

def run_settle(res, rep, a, out):
    """G0's settle check, then the standard settle: twin_sim.settle_report (the back end of this option) when the
    standard simulation provides it, else standardize + settle here. out["settle"]["units"] is per unit."""
    import twin_gate
    TS, exc = _optional_module("twin_sim")
    if TS is None or not callable(getattr(TS, "settle", None)) or not callable(getattr(TS, "standardize", None)):
        out["settle"] = _not_available("the standard settle (twin_sim.settle)", exc)
        return None
    gates = [g["body"] for g in rep.get("gates", [])]
    hold = {}
    for h in a.hold or []:
        hold[h] = "world"
    excluded = []
    if callable(getattr(twin_gate, "settle_check", None)):
        t1 = time.time()
        try:
            chk = twin_gate.settle_check(res)
        except Exception as exc:  # noqa: BLE001
            chk = {"fail": True, "reasons": ["the settle check raised %s" % _err(exc)]}
        excluded = list(chk.get("excluded") or []) if not chk.get("fail") else []
        out["blowup_check"] = dict({k: v for k, v in chk.items() if k != "passes"}, wall_s=round(time.time() - t1, 2))
    elif callable(getattr(twin_gate, "blowup_check", None)):
        t1 = time.time()
        try:
            chk = twin_gate.blowup_check(twin_gate.physics_model(res), gates=gates)
        except Exception as exc:  # noqa: BLE001
            chk = {"fail": True, "reasons": ["the settle check raised %s" % _err(exc)]}
        out["blowup_check"] = dict(chk, wall_s=round(time.time() - t1, 2))
    else:
        out["blowup_check"] = _not_available("G0's settle check (twin_gate.settle_check)")

    def physics():
        # the model the grader simulates: the gate's excluded units and those the settle check excluded removed
        if excluded:
            return twin_gate.physics_model(res, also_exclude=excluded)
        return twin_gate.physics_model(res)
    t1 = time.time()
    if callable(getattr(TS, "settle_report", None)):
        try:
            rec, sim = TS.settle_report(physics(), gates=gates, hold=hold)
        except Exception as exc:  # noqa: BLE001 - TwinSimError and model problems
            out["settle"] = {"ran": False, "why": _err(exc)}
            return None
        rec = dict(rec, ran=True, wall_s=round(time.time() - t1, 2), back_end="twin_sim.settle_report")
        out["settle"] = rec
        return sim
    try:
        cfg = TS.load_config()
        m = physics()
        roles = TS.standardize(m, config=cfg, gates=gates, hold=hold)
        st = TS.settle(m, roles, config=cfg)
    except Exception as exc:  # noqa: BLE001 - TwinSimError and model problems
        out["settle"] = {"ran": False, "why": _err(exc)}
        return None
    rec = {"ran": True, "wall_s": round(time.time() - t1, 2), "back_end": "twin_sim.settle"}
    for k, v in st.items():
        if k not in ("qpos_end", "items", "initial_penetration_mm", "initial_penetration_partner"):
            rec[k] = v
    pen, part = st.get("initial_penetration_mm", {}), st.get("initial_penetration_partner", {})
    rec["units"] = {nm: dict(e, initial_penetration_mm=pen.get(nm, 0.0), initial_penetration_into=part.get(nm))
                    for nm, e in st.get("items", {}).items()}
    rec["held"] = sorted(hold)
    out["settle"] = rec
    return m, roles, cfg, st


def run_probes(sim, a, rep, out):
    """The lift probes from the settled state (twin_sim.probes_report when present, else run_probes here) and the
    gate travel rule in both directions. out["probes"]["lift"] = {probe: {"body", "finite", "moved": {unit: outcome},
    "n_stayed"}}."""
    TS, exc = _optional_module("twin_sim")
    if TS is None or not (callable(getattr(TS, "probes_report", None)) or callable(getattr(TS, "run_probes", None))):
        out["probes"] = _not_available("the lift probes (twin_sim.probes_report)", exc)
        return
    want = list(a.probes or [])
    t2 = time.time()
    travels = {g["body"]: g.get("travel") for g in rep.get("gates", []) if g.get("travel")}
    try:
        if callable(getattr(TS, "probes_report", None)):
            pr = TS.probes_report(sim, bodies=want or None, gate_travels=travels)
            back = "twin_sim.probes_report"
        else:
            m, roles, cfg, st = sim
            runs = TS.run_probes(m, roles, qpos=st["qpos_end"], config=cfg, bodies=want or None)
            pr = {"lift": {}, "gates": {}}
            for name, r in runs.items():
                moved = {k: o for k, o in r.get("outcomes", {}).items() if o["outcome"] != "stayed"}
                pr["lift"][name] = {"body": r.get("body"), "finite": r.get("finite", True), "moved": moved,
                                    "n_stayed": len(r.get("outcomes", {})) - len(moved)}
            back = "twin_sim.run_probes"
    except Exception as exc:  # noqa: BLE001
        out["probes"] = {"error": _err(exc)}
        return
    out["probes"] = dict(pr, wall_s=round(time.time() - t2, 2), back_end=back)


# ---------------------------------------------------------------------------------------------------------------
# --slice (twin_slice.py)

def run_slice(res, a, out):
    """The robot-terms map of the twin (twin_slice.py) as a PNG and its class areas: twin_slice.check_map when the
    module provides that hook, else map_twin + render_png on the model the grader simulates (excluded units removed),
    with the reference level from the twin's own static geometry ("auto") or --slice-floor-z."""
    TL, exc = _optional_module("twin_slice")
    if TL is None:
        out["slice"] = _not_available("the robot-terms map (twin_slice.py)", exc)
        return
    t = time.time()
    try:
        if callable(getattr(TL, "check_map", None)):
            r = TL.check_map(res, out_png=a.slice)
            rec = dict(r) if isinstance(r, dict) else {}
        elif callable(getattr(TL, "map_twin", None)) and callable(getattr(TL, "render_png", None)):
            import mujoco
            import twin_gate
            m = twin_gate.physics_model(res) if res.ok else res.model
            d = mujoco.MjData(m)
            mujoco.mj_forward(m, d)
            ref = "auto" if a.slice_floor_z is None else float(a.slice_floor_z)
            sm = TL.map_twin(m, d, bounds=a.slice_bounds, reference=ref)
            rec = dict(sm.summary()) if callable(getattr(sm, "summary", None)) else {}
            P_ = getattr(sm, "params", {}) or {}
            note = "reference level: %s" % ("auto (the lowest top of your static geometry)" if ref == "auto" else
                                             "%.3f" % ref)
            if P_:
                note += "; crouch below %.2f m, stand below %.2f m, robot radius %.2f m" % (
                    P_.get("h_crouch_m", float("nan")), P_.get("h_stand_m", float("nan")),
                    P_.get("r_robot_m", float("nan")))
            TL.render_png(a.slice, sm.classes, sm.grid, title="robot-terms map of your twin", notes=[note])
            rec["reference"] = ref
        else:
            out["slice"] = _not_available("the robot-terms map (twin_slice.map_twin)")
            return
    except NotImplementedError as exc:
        out["slice"] = _not_available("the robot-terms map (twin_slice.py)", exc)
        return
    except Exception as exc:  # noqa: BLE001
        out["slice"] = {"available": True, "error": _err(exc)}
        return
    rec.update({"available": True, "wall_s": round(time.time() - t, 2), "out": os.path.abspath(a.slice)})
    out["slice"] = rec


# ---------------------------------------------------------------------------------------------------------------
# --walk (the probe robot of ../robot/walk_probe.py)

def robot_dir_candidates(a):
    sub_parent = os.path.dirname(os.path.abspath(a.submission.rstrip("/")))
    cands = []
    if a.robot:
        cands.append(a.robot)
    if os.environ.get("TWIN_ROBOT_DIR"):
        cands.append(os.environ["TWIN_ROBOT_DIR"])
    cands += [os.path.join(HERE, "robot"), os.path.join(sub_parent, "robot"), os.path.join(HERE, "..", "robot"),
              os.path.join(HERE, "..", "instrument")]
    return [os.path.abspath(c) for c in cands]


def load_walk_probe(a):
    """(module, path) of walk_probe.py, or (None, reason)."""
    tried = []
    for d in robot_dir_candidates(a):
        p = os.path.join(d, "walk_probe.py")
        tried.append(p)
        if os.path.isfile(p):
            try:
                spec = importlib.util.spec_from_file_location("walk_probe", p)
                mod = importlib.util.module_from_spec(spec)
                sys.modules["walk_probe"] = mod
                spec.loader.exec_module(mod)
            except Exception as exc:  # noqa: BLE001
                return None, "walk_probe.py at %s does not load (%s)" % (p, _err(exc))
            if not callable(getattr(mod, "run_route", None)):
                return None, "walk_probe.py at %s has no run_route()" % p
            return mod, p
    return None, "no walk_probe.py found (looked in %s)" % ", ".join(tried)


def _shared_walks(robot_dir):
    """The grader's walk module when the payload ships it (twin_walks.py next to this file, or walks.py in the robot
    folder), else None."""
    for p in (os.path.join(HERE, "twin_walk.py"), os.path.join(HERE, "twin_walks.py"),
              os.path.join(robot_dir or HERE, "walks.py")):
        if os.path.isfile(p):
            try:
                spec = importlib.util.spec_from_file_location("twin_walks", p)
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
            except Exception:  # noqa: BLE001
                continue
            if callable(getattr(mod, "prepare", None)):
                return mod
    return None


def walk_contact():
    """The walk probes' standard contact parameters: twin_sim_config.json "walk" / "contact" when the public
    configuration has them, else WALK_CONTACT."""
    TS, _exc = _optional_module("twin_sim")
    if TS is not None:
        try:
            c = TS.load_config().get("walk", {}).get("contact")
            if c:
                return dict(c)
        except Exception:  # noqa: BLE001
            pass
    return dict(WALK_CONTACT)


def walk_spec(res, robot_dir=None):
    """The twin as the walk probes see it (the contract's walk conditions): the gate's grader copy without the
    excluded units, every unit's free joint and every gate's slide joint removed (units held where they stand, gates
    fixed at their photographed position), and the standard walk contact parameters on every collision geom (the
    robot touches every collision geom; visual geoms stay inert). Uses the grader's own walk module when the payload
    ships it."""
    TS, _exc = _optional_module("twin_sim")
    if TS is not None and callable(getattr(TS, "walk_spec", None)):
        return TS.walk_spec(res), "twin_sim.walk_spec"
    W = _shared_walks(robot_dir)
    if W is not None:
        return W.prepare(res, {"contact": dict(walk_contact(), standardise=True)}), "the payload's walk module"
    sp = res.spec.copy()
    for nm in sorted(set(res.report.get("excluded") or [])):
        b = sp.body(nm)
        if b is not None:
            sp.delete(b)
    names = [u["body"] for u in res.report.get("units", []) if u.get("status") == "ok"]
    for nm in names:
        b = sp.body(nm)
        if b is not None:
            for j in list(b.joints):
                sp.delete(j)
    for g in res.report.get("gates", []):
        j = sp.joint(g["joint"])
        if j is not None:
            sp.delete(j)
    con = walk_contact()
    for g in sp.geoms:
        if int(g.contype) == 0 and int(g.conaffinity) == 0:
            continue
        g.contype, g.conaffinity = int(con["contype"]), int(con["conaffinity"])
        g.condim = int(con["condim"])
        g.friction = [float(x) for x in con["friction"]]
        g.solref = [float(x) for x in con["solref"]]
        g.solimp = [float(x) for x in con["solimp"]]
        g.margin, g.gap = float(con["margin_m"]), float(con["gap_m"])
        g.priority, g.solmix = int(con["priority"]), float(con["solmix"])
    return sp, "check_twin (units held, gates fixed, the walk contact parameters)"


def walk_variants(k):
    """k start variants in the route's own frame: number 0 is the nominal start; the others shift the start along
    and to the left of the start heading by up to +-2 cm each, turn the heading by up to +-2 deg and draw a gait
    phase (seeded: the same list every run, the grader's agent-side variant rule)."""
    import numpy as np
    out = [{"along": 0.0, "left": 0.0, "dyaw": 0.0, "phase": 0.0, "seed": 0}]
    rng = np.random.default_rng(WALK_SEED)
    for i in range(1, max(1, int(k))):
        a, l_ = rng.uniform(-WALK_VARIANT_DXY_M, WALK_VARIANT_DXY_M, 2)
        out.append({"along": round(float(a), 4), "left": round(float(l_), 4),
                    "dyaw": round(float(rng.uniform(-math.radians(WALK_VARIANT_DYAW_DEG),
                                                    math.radians(WALK_VARIANT_DYAW_DEG))), 5),
                    "phase": round(float(rng.uniform(0.0, 1.0)), 4), "seed": i})
    return out[:max(1, int(k))]


def harness_variant(route, v):
    """The harness's variant dict (world-frame start shift) of a route-frame variant."""
    p = route["waypoints"]
    if route.get("start_yaw") is not None:
        yaw = float(route["start_yaw"])
    else:
        yaw = math.atan2(float(p[1][1]) - float(p[0][1]), float(p[1][0]) - float(p[0][0]))
    c, s = math.cos(yaw), math.sin(yaw)
    return {"start_dxy": [c * v["along"] - s * v["left"], s * v["along"] + c * v["left"]],
            "start_dyaw": float(v["dyaw"]), "phase": float(v["phase"]), "seed": int(v["seed"]),
            "floor_friction_scale": 1.0}


def read_routes(path):
    with open(path) as f:
        doc = json.load(f)
    routes = doc["routes"] if isinstance(doc, dict) else doc
    if not isinstance(routes, list) or not routes:
        raise ValueError("the routes file holds no list of routes")
    out = []
    for k, r in enumerate(routes):
        if not isinstance(r, dict) or "waypoints" not in r or "modes" not in r:
            raise ValueError("route %d needs 'waypoints' and 'modes'" % k)
        r = dict(r)
        r.setdefault("id", "route_%d" % (k + 1))
        r["modes"] = [MODE_ALIASES.get(md, md) for md in r["modes"]]
        out.append(r)
    return out


_W = {}


def _walk_init(sub, vocab, robot):
    import twin_gate
    ns = argparse.Namespace(submission=sub, robot=robot)
    wp, _p = load_walk_probe(ns)
    res = twin_gate.gate(sub, vocabulary=vocab)
    _W["wp"] = wp
    _W["spec"] = walk_spec(res, robot)[0]


def _quiet_attach():
    """A warnings context that drops MuJoCo's note about attaching the robot (no asset dict) to the twin's spec."""
    import warnings
    ctx = warnings.catch_warnings()
    ctx.__enter__()
    warnings.filterwarnings("ignore", message=".*Attaching a child without asset dict.*")
    return ctx


def _walk_job(job):
    route, variant = job
    ctx = _quiet_attach()
    try:
        return _W["wp"].run_route(_W["spec"], route, variant)
    finally:
        ctx.__exit__(None, None, None)


class _Identity:
    """The identity map (the agent's routes are already in the twin's frame)."""
    s = 1.0

    @staticmethod
    def to_agent(x):
        import numpy as np
        return np.asarray(x, float)

    @staticmethod
    def dir_to_agent(d):
        import numpy as np
        return np.asarray(d, float)


def start_support(rm, wp, x, y, z_ref, window):
    """The grader's start rule: a vertical ray from z_ref + window down to z_ref - window; the support is the top of
    the first solid below the ray's origin. (z, None) or (None, 'inside_solid' | 'no_support')."""
    import mujoco
    rd = mujoco.MjData(rm)
    mujoco.mj_kinematics(rm, rd)
    merged = []
    for lo, hi in sorted(wp._solid_intervals(rm, rd, float(x), float(y))):
        if merged and lo <= merged[-1][1] + 1e-9:
            merged[-1][1] = max(merged[-1][1], hi)
        else:
            merged.append([lo, hi])
    top = z_ref + window
    for lo, hi in reversed(merged):
        if lo < top < hi:
            return None, "inside_solid"
        if hi <= top:
            return (float(hi), None) if hi >= z_ref - window else (None, "no_support")
    return None, "no_support"


def _walk_jobs(routes, variants, wp, rm, window):
    """[(route index, variant index, harness route, harness variant)] and the pre-decided results (start rule)."""
    jobs, pre = [], {}
    for ri, r in enumerate(routes):
        z_ref = float(r.get("start_z", 0.0))
        x0, y0 = (float(v) for v in r["waypoints"][0])
        zs, why = start_support(rm, wp, x0, y0, z_ref, window)
        h = {"id": r["id"], "waypoints": r["waypoints"], "modes": r["modes"], "start_z": zs}
        if r.get("start_yaw") is not None:
            h["start_yaw"] = float(r["start_yaw"])
        for vi, v in enumerate(variants):
            if zs is None:
                pre[(ri, vi)] = {"outcome": "not_reached", "reason": "start_rule", "progress": 0.0,
                                 "why": "no start support within +-%.2f of start_z %.3f: %s" % (window, z_ref, why)}
                continue
            hv = harness_variant(h, v)
            p0 = (x0 + hv["start_dxy"][0], y0 + hv["start_dxy"][1])
            d = (float(r["waypoints"][1][0]) - x0, float(r["waypoints"][1][1]) - y0)
            yaw = (h["start_yaw"] if "start_yaw" in h else math.atan2(d[1], d[0])) + hv["start_dyaw"]
            try:
                z = wp.start_ground_z(rm, p0[0], p0[1], yaw, zs)
                if not (z_ref - window - 1e-9 <= z <= z_ref + window + 1e-9):
                    pre[(ri, vi)] = {"outcome": "not_reached", "reason": "start_window", "progress": 0.0,
                                     "why": "the start height %.3f lies outside start_z %.3f +- %.2f" % (
                                         z, z_ref, window)}
                    continue
            except Exception:  # noqa: BLE001 - StartError: the harness reports it itself
                pass
            jobs.append((ri, vi, h, hv))
    return jobs, pre


def walk_section():
    """The public walk conditions: twin_sim_config.json "walk" when the configuration has it, else the mirror. The
    wall guard of a run (run_wall: max(min_s, factor x per_sim_s x the run's simulated-time limit)) is the
    assessment's own: a run that passes it stops with a time error instead of an outcome."""
    sec = {"variants": 4, "variant_seed": WALK_SEED, "start_dxy_m": WALK_VARIANT_DXY_M,
           "start_dyaw_deg": WALK_VARIANT_DYAW_DEG, "start_ray_m": 0.3, "reach_progress": 0.95,
           "contact": dict(walk_contact(), standardise=True), "procs": {"max": 8, "reserve": 0},
           "run_wall": dict(WALK_RUN_WALL), "budget_margin_s": 0.0}
    TS, _exc = _optional_module("twin_sim")
    if TS is not None:
        try:
            sec.update({k: v for k, v in (TS.load_config().get("walk") or {}).items() if k != "note"})
        except Exception:  # noqa: BLE001
            pass
    return sec


def run_walk(res, a, out, vocab):
    """The probe robot walks the agent's routes in the agent's twin under the grader's walk conditions: the
    payload's walk module (the grader's own walks.py) when it ships, else the same steps here: the twin prepared
    (walk_spec), the start rule (support within +-start_ray_m of the route's start_z, default 0), K start variants."""
    wp, where = load_walk_probe(a)
    if wp is None:
        out["walk"] = {"available": False, "why": "the probe robot is not available: " + where}
        return
    robot_dir = os.path.dirname(where)
    try:
        routes = read_routes(a.walk)
    except Exception as exc:  # noqa: BLE001
        out["walk"] = {"available": True, "error": "cannot read %s: %s" % (a.walk, _err(exc))}
        return
    sec = walk_section()
    W = _shared_walks(robot_dir)
    t = time.time()
    quiet = _quiet_attach()
    try:
        if W is not None:
            how = "the payload's walk module"
            spec = W.prepare(res, sec)
            rm = W.ray_model(spec, wp)
            mapped = []
            for r in routes:
                rr = dict(r)
                if rr.get("start_yaw") is not None and rr.get("start_yaw_rad") is None:
                    rr["start_yaw_rad"] = float(rr["start_yaw"])
                mapped.append(W.map_route(rr, _Identity(), rm, sec, wp))
            variants = W.make_variants(sec, k=a.walk_variants)
            jobs, pre = W.plan(mapped, variants, sec, wp, rm)
            got, _stats = W.run({"twin": spec}, jobs, sec, robot_dir, procs=a.procs)
            got.update(pre)
            hv = {j["key"]: j["variant"] for j in jobs}
            keyed = [((ri, vi), got[(ri, vi)], hv.get((ri, vi))) for ri in range(len(routes))
                     for vi in range(len(variants))]
        else:
            spec, how = walk_spec(res, robot_dir)
            rm = wp.build_models(spec)[1]
            variants = walk_variants(a.walk_variants)
            jobs, pre = _walk_jobs(routes, variants, wp, rm, float(sec["start_ray_m"]))
            todo = [(h, v) for _ri, _vi, h, v in jobs]
            if a.procs > 1 and len(todo) > 1:
                import multiprocessing as mp
                with mp.get_context("spawn").Pool(min(a.procs, len(todo)), initializer=_walk_init,
                                                  initargs=(os.path.abspath(a.submission), vocab, robot_dir)) as pool:
                    results = pool.map(_walk_job, todo, chunksize=1)
            else:
                results = [wp.run_route(spec, h, v) for h, v in todo]
            got = dict(pre)
            hv = {}
            for (ri, vi, _h, v), r_ in zip(jobs, results):
                got[(ri, vi)] = r_
                hv[(ri, vi)] = v
            keyed = [((ri, vi), got[(ri, vi)], hv.get((ri, vi))) for ri in range(len(routes))
                     for vi in range(len(variants))]
    except Exception as exc:  # noqa: BLE001
        out["walk"] = {"available": True, "error": "the walk failed: %s" % _err(exc), "harness": where}
        return
    finally:
        quiet.__exit__(None, None, None)
    runs = []
    summary = {}
    reach = float(sec["reach_progress"])
    for (ri, vi), res_, v in keyed:
        rid = routes[ri]["id"]
        row = {"route": rid, "variant_index": vi, "variant": v}
        row.update({k: res_.get(k) for k in ("outcome", "reason", "why", "progress", "sim_time_s", "end_xy",
                                             "max_tilt_deg", "min_base_height", "contacts_with", "displaced")})
        row["reached"] = res_.get("outcome") == "reached" or float(res_.get("progress", 0.0)) >= reach
        runs.append(row)
        s = summary.setdefault(rid, {"reached": 0, "not_reached": 0, "progress": []})
        s["reached" if row["reached"] else "not_reached"] += 1
        s["progress"].append(float(res_.get("progress", 0.0)))
    for s in summary.values():
        p = s.pop("progress")
        s["mean_progress"] = _r(sum(p) / len(p), 4)
    inst = None
    if callable(getattr(wp, "instrument_hash", None)):
        try:
            inst = wp.instrument_hash()
        except Exception:  # noqa: BLE001
            inst = None
    out["walk"] = {"available": True, "harness": where, "prepared_by": how, "instrument_sha256": inst,
                   "start_window_m": float(sec["start_ray_m"]), "reach_progress": reach,
                   "n_routes": len(routes), "n_variants": len(variants), "wall_s": round(time.time() - t, 1),
                   "summary": summary, "runs": runs}


# ---------------------------------------------------------------------------------------------------------------
# --overlay and --render (MuJoCo's CPU renderer through a photo camera)

def photo_dirs(a):
    sub_parent = os.path.dirname(os.path.abspath(a.submission.rstrip("/")))
    cands = []
    if a.photos:
        cands.append(a.photos)
    if os.environ.get("TWIN_PHOTOS_DIR"):
        cands.append(os.environ["TWIN_PHOTOS_DIR"])
    cands += [os.path.join(HERE, "photos"), os.path.join(sub_parent, "photos")]
    return [os.path.abspath(c) for c in cands]


def find_photo(a, num):
    for d in photo_dirs(a):
        for ext in (".jpg", ".jpeg", ".png", ".JPG"):
            p = os.path.join(d, "site_%03d%s" % (num, ext))
            if os.path.isfile(p):
                return p
    return None


def read_photo(path, w, h):
    """The photo as RGB uint8 (h, w, 3), decoded at a reduced size when it is 4x larger (fast for 24 MP JPEGs)."""
    import numpy as np
    try:
        import cv2
        small = cv2.imread(path, cv2.IMREAD_REDUCED_COLOR_4)
        img = small if small is not None and small.shape[1] >= w else cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if img.shape[:2] != (h, w):
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
        return img
    except ImportError:
        from PIL import Image
        return np.asarray(Image.open(path).convert("RGB").resize((w, h)))


def write_png(path, img):
    try:
        import cv2
        import numpy as np
        ok = cv2.imwrite(path, cv2.cvtColor(np.ascontiguousarray(img), cv2.COLOR_RGB2BGR))
        if not ok:
            raise OSError("cannot write %s" % path)
    except ImportError:
        from PIL import Image
        Image.fromarray(img).save(path)


def render_size(model, cid, max_width):
    W, H = (int(v) for v in model.cam_resolution[cid])
    scale = min(1.0, float(max_width) / float(W)) if W > 0 else 1.0
    return max(16, int(round(W * scale))), max(16, int(round(H * scale))), scale


def _camera_id(model, num):
    import mujoco
    name = "photo_%03d" % num
    cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, name)
    return cid, name


def segmentation(model, cam, w, h):
    """(geom id per pixel, -1 where no geom) of a MuJoCo segmentation render of the collision geoms through camera
    `cam` at w x h pixels (the camera's own intrinsics stretched over the render), and the render time."""
    import mujoco
    import numpy as np
    m = copy.deepcopy(model)
    m.vis.global_.offwidth = max(int(m.vis.global_.offwidth), w)
    m.vis.global_.offheight = max(int(m.vis.global_.offheight), h)
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
    opt.sitegroup[:] = 0
    t0 = time.time()
    r = mujoco.Renderer(m, height=h, width=w)
    try:
        r.enable_segmentation_rendering()
        r.update_scene(d, camera=cam, scene_option=opt)
        seg = r.render().copy()
    finally:
        r.close()
    ids = np.where(seg[:, :, 1] == int(mujoco.mjtObj.mjOBJ_GEOM), seg[:, :, 0], -1).astype(int)
    ids[(ids < 0) | (ids >= m.ngeom)] = -1
    return ids, time.time() - t0


def overlay(res, a, num, out_path):
    """Segmentation of the twin's collision geoms through photo_NNN, outlines drawn on the photo."""
    import mujoco
    import numpy as np
    m = res.model
    cid, cam = _camera_id(m, num)
    if cid < 0:
        return {"error": "your twin has no camera %s" % cam}
    w, h, scale = render_size(m, cid, a.max_width)
    geom_ids, t_render = segmentation(m, cam, w, h)
    rep = res.report
    units = [u["body"] for u in rep.get("units", []) if u.get("status") == "ok"]
    gates = [g["body"] for g in rep.get("gates", [])]
    root = np.asarray(m.body_rootid, int)
    unit_ids = {}
    for k, nm in enumerate(units):
        b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, nm)
        if b >= 0:
            unit_ids[int(b)] = k + 1
    gate_ids = set()
    for nm in gates:
        b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, nm)
        if b >= 0:
            gate_ids.add(int(b))
    lut = np.zeros(max(1, m.ngeom), int)
    for g in range(m.ngeom):
        rb = int(root[int(m.geom_bodyid[g])])
        if rb in unit_ids:
            lut[g] = unit_ids[rb]
            continue
        b = int(m.geom_bodyid[g])
        in_gate = False
        while b > 0:
            if b in gate_ids:
                in_gate = True
                break
            b = int(m.body_parentid[b])
        lut[g] = 100000 if in_gate else 200000 + g       # each environment geom keeps its own outline
    lab = np.zeros(geom_ids.shape, int)
    ok = geom_ids >= 0
    lab[ok] = lut[geom_ids[ok]]
    photo = find_photo(a, num)
    if photo is not None:
        img = read_photo(photo, w, h).astype(np.float32)
        note = None
    else:
        img = np.full((h, w, 3), 40.0, np.float32)
        note = "photo site_%03d not found (looked in %s): outlines drawn on grey" % (num, ", ".join(photo_dirs(a)))
    edge = np.zeros(lab.shape, bool)
    edge[:, 1:] |= lab[:, 1:] != lab[:, :-1]
    edge[1:, :] |= lab[1:, :] != lab[:-1, :]
    edge[:, :-1] |= edge[:, 1:].copy()
    edge[:-1, :] |= edge[1:, :].copy()
    rng = np.random.default_rng(7)
    pal = rng.integers(60, 256, size=(len(units) + 2, 3)).astype(np.float32)
    colour = np.zeros(lab.shape + (3,), np.float32)
    for k in range(1, len(units) + 1):
        colour[lab == k] = pal[k]
    colour[lab >= 200000] = (255.0, 255.0, 255.0)
    colour[lab == 100000] = (255.0, 160.0, 0.0)
    tint = (lab > 0) & (lab <= len(units))
    img[tint] = 0.75 * img[tint] + 0.25 * colour[tint]
    img[edge] = np.where(lab[edge][:, None] > 0, colour[edge], 255.0)
    img = np.clip(img, 0, 255).astype(np.uint8)
    visible = {}
    try:
        import cv2
    except ImportError:
        cv2 = None
    for k, nm in enumerate(units):
        mask = lab == k + 1
        n = int(mask.sum())
        visible[nm] = int(round(n / (scale * scale)))
        if n and cv2 is not None:
            ys, xs = np.nonzero(mask)
            cv2.putText(img, nm, (int(xs.mean()) - 4 * len(nm), int(ys.mean())), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (255, 255, 255), 1, cv2.LINE_AA)
    write_png(out_path, img)
    out = {"out": os.path.abspath(out_path), "camera": cam, "size": [w, h], "scale": _r(scale, 6),
           "photo": photo, "render_s": round(t_render, 2), "visible_units_px_full_res": visible,
           "environment_px_full_res": int(round(int((lab >= 200000).sum()) / (scale * scale))),
           "gate_px_full_res": int(round(int((lab == 100000).sum()) / (scale * scale)))}
    if note:
        out["note"] = note
    return out


def render(res, a, num, out_path):
    """A plain render of the twin through photo_NNN: the submitted scene with its own materials and textures when it
    compiles from its folder, else the gate's copy (no textures)."""
    import mujoco
    import numpy as np
    src = "submission"
    try:
        m = mujoco.MjModel.from_xml_path(os.path.join(os.path.abspath(a.submission), "scene.xml"))
    except Exception:  # noqa: BLE001 - fall back on the gate's copy
        m = copy.deepcopy(res.model)
        src = "the gate's copy (textures removed)"
    cid, cam = _camera_id(m, num)
    if cid < 0:
        return {"error": "your twin has no camera %s" % cam}
    w, h, scale = render_size(m, cid, a.max_width)
    m.vis.global_.offwidth = max(int(m.vis.global_.offwidth), w)
    m.vis.global_.offheight = max(int(m.vis.global_.offheight), h)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    t0 = time.time()
    r = mujoco.Renderer(m, height=h, width=w)
    try:
        r.update_scene(d, camera=cam)
        img = r.render().copy()
    finally:
        r.close()
    t_render = time.time() - t0
    out = {"out": os.path.abspath(out_path), "camera": cam, "size": [w, h], "scale": _r(scale, 6),
           "model": src, "render_s": round(t_render, 2)}
    if a.blend > 0:
        photo = find_photo(a, num)
        if photo is None:
            out["note"] = "photo site_%03d not found: no blend" % num
        else:
            ph = read_photo(photo, w, h).astype(np.float32)
            img = np.clip(a.blend * ph + (1.0 - a.blend) * img.astype(np.float32), 0, 255).astype(np.uint8)
            out["blend"] = {"photo": photo, "photo_weight": a.blend}
    write_png(out_path, img)
    return out


def run_picture(kind, res, a, out):
    num_txt, path = (a.overlay if kind == "overlay" else a.render)
    num = photo_number(num_txt)
    if num is None:
        out[kind] = {"error": "%r is not a photo (use site_NNN)" % num_txt}
        return
    try:
        out[kind] = (overlay if kind == "overlay" else render)(res, a, num, path)
    except Exception as exc:  # noqa: BLE001 - a rendering problem must not hide the gate result
        out[kind] = {"error": _err(exc), "hint": "MuJoCo renders on the CPU with MUJOCO_GL=osmesa (libOSMesa); "
                                                  "set it before the call"}


# ---------------------------------------------------------------------------------------------------------------
# main

def run(a):
    import twin_gate
    out = {"format": "check-site-twin-report/1", "submission": os.path.abspath(a.submission)}
    vocab = a.vocab or os.path.join(HERE, "class_names.json")
    t0 = time.time()
    try:
        res = twin_gate.gate(a.submission, vocabulary=vocab)
    except twin_gate.TaskError as exc:
        out["task_error"] = str(exc)
        return out, 2
    rep = res.report
    rules = dict(twin_gate.DEFAULT_RULES)
    out["gate"] = {"pass": bool(res.ok), "version": rep.get("gate_version"), "reasons": rep.get("reasons", []),
                   "notes": rep.get("warnings", []), "stats": rep.get("stats", {}),
                   "seconds": round(time.time() - t0, 2)}
    out["cameras"] = camera_rows(rep)
    out["n_photo_cameras_valid"] = rep.get("n_photo_cameras_valid", 0)
    out["min_photo_cameras"] = rules.get("min_photo_cameras")
    out["units"] = rep.get("units", [])
    out["excluded"] = rep.get("excluded", [])
    out["column_depth"] = (rep.get("stats") or {}).get("column_depth")
    status = 0 if res.ok else 1
    if res.model is None:
        return out, status
    try:
        out["gates"] = gate_summary(res.model, rep, rules)
    except Exception as exc:  # noqa: BLE001
        out["gates"] = [{"error": _err(exc)}]
    if a.overlay:
        run_picture("overlay", res, a, out)
    if a.render:
        run_picture("render", res, a, out)
    if not res.ok:
        for k, flag in (("settle", a.settle or a.probes is not None), ("slice", a.slice), ("walk", a.walk)):
            if flag:
                out[k] = {"ran": False, "why": "the gate failed"}
        return out, status
    if a.slice:
        run_slice(res, a, out)
    if a.settle or a.probes is not None:
        sim = run_settle(res, rep, a, out)
        bc = out.get("blowup_check") or {}
        if bc.get("fail"):
            status = 1
        if a.probes is not None and sim is not None and not bc.get("fail"):
            run_probes(sim, a, rep, out)
    if a.walk:
        run_walk(res, a, out, vocab)
    return out, status


def _fmt(x, k=1, unit=""):
    return "-" if x is None else ("%.*f%s" % (k, float(x), unit))


def print_report(out):
    if "task_error" in out:
        print("task error (a problem of the task's own files, not of your twin): %s" % out["task_error"])
        return
    g = out["gate"]
    print("twin_gate %s: %s  (%s, %.1f s)" % ("PASS" if g["pass"] else "FAIL", out["submission"], g.get("version"),
                                            g["seconds"]))
    for r in g["reasons"]:
        print("  FAIL  " + r)
    for w in g["notes"]:
        print("  note  " + w)
    cams = out.get("cameras", [])
    print("photo cameras: %d valid of %d (at least %s needed)" % (out.get("n_photo_cameras_valid", 0), len(cams),
                                                                   out.get("min_photo_cameras")))
    if cams:
        print("  %-10s %-7s %9s %9s %9s %9s %8s %8s %7s %7s" % ("camera", "status", "fx", "fy", "cx", "cy", "up/+z",
                                                               "axis el", "front", "view"))
    for c in cams:
        s = c.get("sanity") or {}
        print("  %-10s %-7s %9s %9s %9s %9s %8s %8s %7s %7s" % (
            c["name"], c["status"], _fmt(c.get("fx")), _fmt(c.get("fy")), _fmt(c.get("cx")), _fmt(c.get("cy")),
            _fmt(s.get("up_angle_deg"), 1, "d"), _fmt(s.get("optical_axis_elevation_deg"), 1, "d"),
            _fmt(100 * s["in_front_share"], 0, "%") if "in_front_share" in s else "-",
            _fmt(100 * s["in_view_share"], 0, "%") if "in_view_share" in s else "-"))
        for r in c.get("reasons", []):
            print("      - " + r)
        for hnt in c.get("hints", []):
            print("      ! " + hnt)
    cd = out.get("column_depth")
    if cd:
        print("column depth: at most %d collision geoms reach over one %.2f m square (at %s)" % (
            cd.get("max", 0), cd.get("cell_m") or 0.0, cd.get("at")))
    for gt in out.get("gates", []):
        if "error" in gt:
            print("gates: cannot be summarised (%s)" % gt["error"])
            continue
        print("gate %s: slide %s, axis (world) %s, tilt %s deg; range %s, photographed at %s, leaf width %s" % (
            gt["body"], gt["joint"], gt.get("axis_world"), _fmt(gt.get("axis_tilt_from_horizontal_deg"), 2),
            gt.get("range_m"), gt.get("qpos0_m"), gt.get("width_m")))
        for key, rr in sorted((gt.get("rule_by_direction") or {}).items()):
            print("  travel rule if it opens toward %s: %s; opens %s, closing side %s%s" % (
                key, "OK" if rr["ok"] else "NOT MET", _fmt(rr.get("open_m"), 3), _fmt(rr.get("close_m"), 3),
                (" (" + "; ".join(rr["reasons"]) + ")") if rr["reasons"] else ""))
        rr = gt.get("rule_from_rail") or {}
        if rr.get("ok") is not None:
            sign = gt.get("opening_sign")
            print("  from the rail in your twin: opens toward %s; rule %s" % (
                {1: "+axis", -1: "-axis"}.get(sign, "undetermined"), "OK" if rr.get("ok") else "NOT MET"))
    units = out.get("units", [])
    if units:
        print("units (%d listed, %d excluded):" % (len(units), len(out.get("excluded", []))))
        print("  %-24s %-18s %-9s %10s %9s %11s %6s %s" % ("body", "label", "status", "mass kg", "kg/m^3", "friction",
                                                         "pieces", "extent m"))
        for it in units:
            fr = it.get("friction")
            frs = "%.2f-%.2f" % tuple(fr) if fr else "-"
            ext = it.get("extent_m")
            print("  %-24s %-18s %-9s %10s %9s %11s %6s %s" % (
                it["body"], it.get("label"), it.get("status"),
                "%.3f" % it["mass_kg"] if "mass_kg" in it else "-",
                "%.0f" % it["density_kg_m3"] if "density_kg_m3" in it else "-", frs, it.get("n_collision_geoms", "-"),
                " x ".join("%.2f" % v for v in sorted(ext, reverse=True)) if ext else "-"))
            for r in it.get("reasons", []):
                print("      - " + r)
    for kind in ("overlay", "render"):
        ov = out.get(kind)
        if not ov:
            continue
        if "error" in ov:
            print("%s: FAILED %s%s" % (kind, ov["error"], (" (" + ov["hint"] + ")") if ov.get("hint") else ""))
            continue
        extra = ""
        if kind == "overlay":
            hidden = [k for k, v in ov.get("visible_units_px_full_res", {}).items() if v == 0]
            if hidden:
                extra = "; units not in this view: " + ", ".join(hidden[:12]) + (" ..." if len(hidden) > 12 else "")
        print("%s: %s (%s, %d x %d, %.1f s)%s" % (kind, ov["out"], ov["camera"], ov["size"][0], ov["size"][1],
                                                 ov["render_s"], extra))
        if ov.get("note"):
            print("  note: " + ov["note"])
    sl = out.get("slice")
    if sl:
        if not sl.get("available", True) or sl.get("ran") is False:
            print("slice: not available: %s" % sl.get("why"))
        elif "error" in sl:
            print("slice: FAILED %s" % sl["error"])
        else:
            areas = sl.get("area_m2") or {}
            print("slice (robot-terms map): %s (%.1f s)%s" % (sl.get("out"), sl.get("wall_s", 0.0), (": " + ", ".join(
                "%s %.1f m2" % (k, v) for k, v in areas.items())) if areas else ""))
    bc = out.get("blowup_check")
    st = out.get("settle")
    if bc is not None:
        if bc.get("available") is False:
            print("G0 settle check: not available: %s" % bc.get("why"))
        else:
            print("G0 settle check (blow-up): %s%s" % ("FAIL" if bc.get("fail") else "pass",
                                                      (": " + "; ".join(str(x) for x in bc.get("reasons", [])))
                                                      if bc.get("reasons") else ""))
            if bc.get("excluded"):
                print("  units excluded by the settle check (they score 0; the standard settle below runs without "
                      "them): %s" % ", ".join(bc["excluded"]))
                for u, why in sorted((bc.get("exclusion_reasons") or {}).items()):
                    print("    %s: %s" % (u, why))
    if st:
        if st.get("available") is False:
            print("settle: not available: %s" % st.get("why"))
        elif not st.get("ran"):
            print("settle: not run (%s)" % st.get("why"))
        else:
            print("standard settle (%.1f s simulated, %.1f s wall): %s" % (
                st.get("seconds", 0.0), st["wall_s"],
                "UNSTABLE" if st.get("unstable") else "stable"))
            if st.get("warnings"):
                print("  MuJoCo warnings: %s" % st["warnings"])
            units = st.get("units", {})
            print("  %-24s %9s %9s %9s %9s %9s %7s  %s" % ("unit", "relax mm", "relax deg", "settle mm", "max mm",
                                                         "max deg", "static", "initial penetration"))
            for nm, e in units.items():
                line = "  %-24s %9s %9s %9s %9s %9s %7s" % (
                    nm, _fmt(e.get("relax_drift_mm"), 2), _fmt(e.get("relax_rot_deg"), 2),
                    _fmt(e.get("settle_max_drift_mm", e.get("settle_drift_mm")), 2), _fmt(e.get("max_drift_mm"), 2),
                    _fmt(e.get("max_rot_deg"), 2), "yes" if e.get("static") else "NO")
                p = e.get("initial_penetration_mm")
                line += ("  %.1f mm into %s" % (p, e.get("initial_penetration_into") or "?")) if p else ""
                line += "  (held)" if e.get("held") else ""
                print(line)
            if st.get("support_edges"):
                print("  support edges (supporter -> supported, mean up-force / weight):")
                for e in st["support_edges"]:
                    print("    %s -> %s  %.2f" % (e["supporter"], e["supported"], e.get("weight_frac", 0.0)))
    pr = out.get("probes")
    if pr:
        if pr.get("available") is False:
            print("probes: not available: %s" % pr.get("why"))
        elif "error" in pr:
            print("probes: FAILED " + pr["error"])
        else:
            print("lift probes (%.1f s wall); the other units' outcomes, units that stayed not listed:" % pr["wall_s"])
            for name, p in (pr.get("lift") or {}).items():
                head = "  %-30s" % name
                if not p.get("finite", True):
                    print(head + " NON-FINITE")
                    continue
                moved = p.get("moved") or {}
                if not moved:
                    print(head + " every other unit stayed")
                    continue
                print(head + " " + "; ".join("%s %s (%.0f mm, %.1f deg)" % (
                    k, v["outcome"], math.sqrt(sum(x * x for x in v["d_mm"])), v["rot_deg"])
                    for k, v in sorted(moved.items())))
            for gb, g in sorted((pr.get("gates") or {}).items()):
                for key, lab in (("opening_plus_axis", "+axis"), ("opening_minus_axis", "-axis")):
                    r = g.get(key) or {}
                    print("  gate %s, if it opens toward %s: %s; opens %s, closing side %s%s" % (
                        gb, lab, "OK" if r.get("ok") else "NOT MET", _fmt(r.get("open_m"), 3),
                        _fmt(r.get("close_m"), 3),
                        (" (" + "; ".join(r.get("reasons", [])) + ")") if r.get("reasons") else ""))
    wk = out.get("walk")
    if wk:
        if not wk.get("available", True) or wk.get("ran") is False:
            print("walk: not available: %s" % wk.get("why"))
        elif "error" in wk:
            print("walk: FAILED %s" % wk["error"])
        else:
            print("walk (%d routes x %d start variants, %.0f s wall; units and gates held still; start within "
                  "+-%.2f of each route's start_z):" % (wk["n_routes"], wk["n_variants"], wk["wall_s"],
                                                       wk["start_window_m"]))
            for rid, s in wk["summary"].items():
                print("  %-20s reached %d  not reached %d  mean progress %.2f" % (
                    rid, s.get("reached", 0), s.get("not_reached", 0), s["mean_progress"]))
            for row in wk["runs"]:
                if not row.get("reached"):
                    print("    %s variant %s: %s (%s), progress %.2f, body touched: %s%s" % (
                        row["route"], row["variant_index"], row.get("outcome"), row.get("reason"),
                        row.get("progress") or 0.0, ", ".join(row.get("contacts_with") or []) or "nothing",
                        ("; " + row["why"]) if row.get("why") else ""))


def main(argv=None):
    ap = argparse.ArgumentParser(description="Check a site twin against the public gate and run the standard "
                                             "simulation, the probe robot and the renderer on it. Scores nothing "
                                             "and reads no ground truth.")
    ap.add_argument("submission", nargs="?", default=DEFAULT_SUB)
    ap.add_argument("--settle", action="store_true", help="G0's settle check and the standard settle")
    ap.add_argument("--probes", nargs="*", default=None, metavar="UNIT",
                    help="the lift probes from the settled state (all units, or the units named); implies --settle")
    ap.add_argument("--hold", nargs="*", default=None, metavar="UNIT",
                    help="hold these units still during the settle and the probes")
    ap.add_argument("--slice", metavar="OUT.png", help="the robot-terms map of your twin (twin_slice.py)")
    ap.add_argument("--slice-floor-z", type=float, default=None,
                    help="with --slice: one reference level everywhere (default: from your static geometry)")
    ap.add_argument("--slice-bounds", type=float, nargs=4, default=None, metavar=("X0", "Y0", "X1", "Y1"),
                    help="with --slice: the map's rectangle (default: your twin's collision geometry)")
    ap.add_argument("--walk", metavar="ROUTES.json", help="walk the probe robot along your routes in your twin")
    ap.add_argument("--walk-variants", type=int, default=1, metavar="K",
                    help="start variants per route (1 = the nominal start; more: +-2 cm, +-2 deg, gait phase)")
    ap.add_argument("--procs", type=int, default=1, help="processes for --walk (default 1)")
    ap.add_argument("--robot", default=None, help="the folder with walk_probe.py (default: robot/ next to this script)")
    ap.add_argument("--overlay", nargs=2, metavar=("site_NNN", "OUT.png"),
                    help="outlines of your collision geoms over photo site_NNN through camera photo_NNN")
    ap.add_argument("--render", nargs=2, metavar=("site_NNN", "OUT.png"),
                    help="a render of your twin through camera photo_NNN")
    ap.add_argument("--blend", type=float, default=0.0, help="with --render: blend the photo in (weight 0..1)")
    ap.add_argument("--max-width", type=int, default=MAX_RENDER_WIDTH,
                    help="render width in pixels (at most %d)" % MAX_RENDER_WIDTH)
    ap.add_argument("--photos", default=None, help="the folder with site_NNN.jpg (default: photos/ next to this "
                                                    "script)")
    ap.add_argument("--vocab", default=None, help="class_names.json (default: the file next to this script)")
    ap.add_argument("--json", action="store_true", help="print one JSON document instead of the text report")
    a = ap.parse_args(argv)
    if not 16 <= a.max_width <= MAX_RENDER_WIDTH:
        ap.error("--max-width must be in [16, %d]" % MAX_RENDER_WIDTH)
    if not 0.0 <= a.blend <= 1.0:
        ap.error("--blend must be in [0, 1]")
    if a.walk_variants < 1 or a.procs < 1:
        ap.error("--walk-variants and --procs must be at least 1")
    _gl_setup(bool(a.overlay or a.render))
    out, status = run(a)
    if a.json:
        print(json.dumps(_jsonable(out), indent=1))
    else:
        print_report(out)
        if out.get("gate", {}).get("pass") and "settle" not in out:
            print("G0 also requires the settle check to pass (run with --settle) and your photo cameras to pass the "
                  "camera fit described in the contract.")
    return status


if __name__ == "__main__":
    sys.exit(main())
