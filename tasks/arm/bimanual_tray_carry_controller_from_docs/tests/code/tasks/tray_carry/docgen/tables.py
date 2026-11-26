import _genenv

import json
import math
import os
import re

import numpy as np

from openarm_unit import docs

STATUS = {"stage05": "profile", "contract": "contract", "cell": "cell (OURS)", "obs": "observation (OURS)",
          "curves": "PV (read points)", "derived": "derived", "grader": "grader", "rule": "band rule",
          "runtime": "runtime"}

OBS_TORQUES = {"DM8009": (20.0, 25.0, 30.0, 35.0, 40.0), "DM4340": (9.0, 12.0, 15.0, 20.0, 27.0),
               "DM4310": (3.0, 3.5, 4.0, 5.0, 6.0, 7.0)}
OBS_HOLD_S = 1800.0
OBS_COOL_S = 600.0
OBS_AFTER_S = 60.0
OBS_TRACE_EVERY = 100


def _name(s):
    return re.sub(r"[^A-Za-z0-9_]", "_", str(s)).strip("_")


def _plain(v):
    if isinstance(v, tuple):
        return [_plain(x) for x in v]
    if isinstance(v, list):
        return [_plain(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _plain(x) for k, x in v.items()}
    return v


def _numeric(v):
    if isinstance(v, bool):
        return False
    if isinstance(v, (int, float)):
        return True
    if isinstance(v, (list, tuple)):
        return any(_numeric(x) for x in v)
    if isinstance(v, dict):
        return any(_numeric(x) for x in v.values())
    return False


def leaf(value, unit="-", status="-", source="-"):
    return {"value": _plain(value), "unit": unit, "status": status, "source": source}


def profile_table(path=None):
    with open(path or _genenv.PROFILE_PATH, encoding="utf-8") as f:
        prof = json.load(f)
    out = {}
    for e in prof["entries"] + prof["task_quantities"]:
        if "range" in e:
            v = e["range"]
        elif "value" in e and _numeric(e["value"]):
            v = e["value"]
        else:
            continue
        q = _name(e["quantity"])
        el = e.get("element") or ""
        if re.fullmatch(r"J\d(-J\d)?", el):
            out.setdefault(q, {})[_name(el)] = leaf(v, e.get("unit", "-"), STATUS["stage05"], e.get("source", "-"))
        else:
            out[q] = leaf(v, e.get("unit", "-"), STATUS["stage05"], e.get("source", "-"))
    return out


def contract_table(mod=None):
    C = mod or _genenv.contract()
    out = {}
    for k, v in vars(C).items():
        if k.startswith("_") or not k.isupper() or callable(v):
            continue
        if _numeric(v):
            out[k] = leaf(v, "-", STATUS["contract"], "contract/constants.py " + k)
    return out


def cell_table(shift=None):
    SH = shift or _genenv.shift()
    Cell = SH.Cell
    out = {}
    for k in ("FOOT_HALF", "FOOT_XY", "CUP_CLEAR", "CUP_WALL", "DECK_HALF", "DECK_THICK_HALF", "DECK_Z", "CUT_X",
              "CUT_Y_IN", "FOOTING_HALF", "WEB_WIDTH", "BLOCK_WIDTH", "MASS", "S_HALF", "F_DRAWN", "S_DRAWN", "GRIP_Y",
              "GRIP_ABOVE_FEET", "BENCH_TOP", "PITCH_DEG"):
        out[k] = leaf(getattr(Cell, k), "m (deg for PITCH_DEG, kg for MASS)", STATUS["cell"], "ref/shift.py Cell." + k)
    parts = Cell.bench_parts()
    xs, ys = [], []
    for _n, (c, _q), h in parts:
        xs += [c[0] - h[0], c[0] + h[0]]
        ys += [abs(c[1]) + h[1]]
    out["bench_outline"] = leaf({"x": [min(xs), max(xs)], "half_y": max(ys)}, "m", STATUS["cell"],
                                "ref/shift.py Cell.bench_parts() (the outline of its parts)")
    fp = Cell.fixture_parts()
    base = [p for p in fp if p[0] == "base"][0]
    out["F_plate_half"] = leaf(list(base[2]), "m", STATUS["cell"], "ref/shift.py Cell.fixture_parts() base")
    neck = [p for p in Cell.tray_parts() if p["part"] == "neck_left"][0]
    block = [p for p in Cell.tray_parts() if p["part"] == "block_left"][0]
    out["neck_half"] = leaf(list(neck["size"]), "m", STATUS["cell"], "ref/shift.py Cell.tray_parts() neck_left")
    out["block_half"] = leaf(list(block["size"]), "m", STATUS["cell"], "ref/shift.py Cell.tray_parts() block_left")
    out["gravity"] = leaf(-float(SH.G[2]), "m/s^2", STATUS["cell"], "ref/shift.py G (the maker's scene)")
    out["contact_friction"] = leaf(float(SH.W.maker_collision_class()["friction"][0]), "-", STATUS["cell"],
                                   "openarm_unit/world.py maker_collision_class() friction[0] (rest and object parts)")
    return out


def grader_table():
    import inspect
    SH = _genenv.shift()
    Q = _genenv.quantities()
    out = {}
    for k in ("G_ABS", "RELEASED_CLAMP_N", "S_FOOT_ON_TOP_TOL_M", "VISIT_TOL_M", "FOOT_ON_PLATE_TOL_M"):
        out[k] = leaf(getattr(Q, k), "-", "grader (ref/quantities.py)", "ref/quantities.py " + k)
    out["SHELF_CARRIES_N"] = leaf(SH.Monitor.SHELF_CARRIES_N, "N", "grader (ref/shift.py)",
                                  "ref/shift.py Monitor.SHELF_CARRIES_N")
    sig = inspect.signature(SH.Cell.feet_in_cups)
    for k in ("tol_xy", "tol_z"):
        out["feet_in_cups_" + k] = leaf(sig.parameters[k].default, "m", "grader (ref/shift.py)",
                                        "ref/shift.py Cell.feet_in_cups default " + k)
    return out


def curves_table(path=None):
    with open(path or os.path.join(_genenv.DATA, "vendor_curves.json"), encoding="utf-8") as f:
        d = json.load(f)
    out = {}
    for cid, c in d["curves"].items():
        out[cid] = {"points": leaf([[p["t_s"], p["T_C"]] for p in c["points"]], "s, deg C", STATUS["curves"],
                                   c["source"]),
                    "label": leaf(c["label_numbers"], "-", STATUS["curves"], c["source"])}
        for k in ("reaches_99_s", "ends_s"):
            if k in c:
                out[cid][k] = leaf(c[k], "s", STATUS["curves"], c["source"])
    return out


def runtime_publish():
    p = os.path.join(_genenv.RUNTIME, "publish.py")
    if not os.path.isfile(p):
        return None
    return _genenv.load_module("t05_runtime_publish", p)


CHILD_NUMERICS = ("OPENBLAS_CORETYPE", "OPENBLAS_L2_SIZE")
CHILD_ENV_FILES = ("procutil.py", "wire.py")


def runtime_child_numerics():
    import ast
    found = {}
    for fn in CHILD_ENV_FILES:
        p = os.path.join(_genenv.RUNTIME, fn)
        if not os.path.isfile(p):
            continue
        with open(p, encoding="utf-8") as f:
            tree = ast.parse(f.read())
        for node in ast.walk(tree):
            pairs = []
            if isinstance(node, ast.Dict):
                pairs = [(k, v) for k, v in zip(node.keys, node.values) if k is not None]
            elif isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Subscript):
                pairs = [(node.targets[0].slice, node.value)]
            elif isinstance(node, ast.Call):
                pairs = [(ast.Constant(kw.arg), kw.value) for kw in node.keywords if kw.arg]
            elif isinstance(node, ast.Tuple) and len(node.elts) == 2:
                pairs = [(node.elts[0], node.elts[1])]
            for k, v in pairs:
                if isinstance(k, ast.Constant) and k.value in CHILD_NUMERICS and isinstance(v, ast.Constant) and \
                        isinstance(v.value, (str, int)) and not isinstance(v.value, bool) and \
                        v.value not in CHILD_NUMERICS:
                    found.setdefault(k.value, set()).add(str(v.value))
    bad = {k: sorted(v) for k, v in found.items() if len(v) != 1}
    if bad:
        raise ValueError(f"the runtime gives the controller's process two values of one variable: {bad}")
    return {k: next(iter(v)) for k, v in sorted(found.items())}


def runtime_table():
    pub = runtime_publish()
    if pub is None:
        return {}
    c = pub.runtime_constants()
    out = {k: leaf(v, "-", "runtime", "runtime/publish.py runtime_constants " + k) for k, v in c.items()
           if _numeric(v)}
    num = runtime_child_numerics()
    if "OPENBLAS_L2_SIZE" in num:
        out["child_openblas_l2_size"] = leaf(int(num["OPENBLAS_L2_SIZE"]), "-", "runtime",
                                             "runtime/procutil.py, runtime/wire.py: the controller's environment, "
                                             "OPENBLAS_L2_SIZE")
    return out


def rule_table():
    with open(os.path.join(_genenv.RULES, "band_rule.json"), encoding="utf-8") as f:
        b = json.load(f)
    w = b["weights_of_record"]
    return {"weights": leaf({k: w[k] for k in ("Q", "unit", "tray_place")}, "-", "band rule",
                            "rules/band_rule.json weights_of_record")}


_OBS_CACHE = {}


def obs_editions(T=None):
    with open(_genenv.PROFILE_PATH, encoding="utf-8") as f:
        prof = json.load(f)
    c6 = [e for e in prof["entries"] if e["quantity"] == "C6"][0]["value"]
    return dict(c6)


def observation_cases():
    P = profile_table()
    h, amb, off = P["TH1"]["value"], P["S3_ambient"]["value"], P["S3_start_offset"]["value"]
    grid = lambda r: (float(r[0]), 0.5 * (float(r[0]) + float(r[1])), float(r[1]))
    eds = obs_editions()
    C = _genenv.contract()
    cases, keys = [], []
    for cls, torques in OBS_TORQUES.items():
        for tq in torques:
            for hh in grid(h):
                for a in grid(amb):
                    for o in grid(off):
                        cases.append(dict(edition=eds[cls], h=hh, T_amb=a, T_w0=a + o, T_h0=a + o,
                                          drive_torque=float(tq), ot_value=float(C.OT_VALUE_C)))
                        keys.append((cls, tq))
    return cases, keys


def observe(law=None):
    from openarm_unit import thermal
    if law is None:
        with open(os.path.join(_genenv.UNIT_PKG, "params_law.json"), encoding="utf-8") as f:
            law = json.load(f)
    tag = json.dumps(law["thermal_constants"], sort_keys=True)
    if tag in _OBS_CACHE:
        return _OBS_CACHE[tag]
    cases, keys = observation_cases()
    pub = json.load(open(os.path.join(_genenv.UNIT_PKG, "params_public.json"), encoding="utf-8"))
    for c in cases:
        peak = float(pub["editions"][c["edition"]]["peak_torque_Nm"]["value"])
        if c["drive_torque"] > peak:
            raise ValueError(f"an observed torque above the peak torque of {c['edition']}")
    r = thermal.hold_observations(law["thermal_constants"], cases, OBS_HOLD_S, OBS_COOL_S,
                                  trace_every=OBS_TRACE_EVERY)
    dt = thermal.DT
    steps = r["trace_steps"]
    rows = {}
    for i, (cls, tq) in enumerate(keys):
        tr = int(r["trip_step"][i])
        row = rows.setdefault((cls, tq), {"trip_s": [], "none": 0, "rep_at": [], "rep_max_after": [], "clear_s": [],
                                          "rep_end_hold_none": []})
        if tr < 0:
            row["none"] += 1
            k = int(np.searchsorted(steps, int(round(OBS_HOLD_S / dt))))
            row["rep_end_hold_none"].append(int(r["reported_trace"][k, i]))
            continue
        row["trip_s"].append(tr * dt)
        row["rep_at"].append(int(r["reported_at_trip"][i]))
        lo, hi = tr, tr + int(round(OBS_AFTER_S / dt))
        sel = (steps >= lo) & (steps <= hi)
        row["rep_max_after"].append(int(max(r["reported_trace"][sel, i])))
        cl = int(r["clear_allowed_step"][i])
        row["clear_s"].append((cl - tr) * dt if cl >= 0 else float("nan"))
    out = {"cases": len(cases), "rows": {}}
    for (cls, tq), row in rows.items():
        o = {"n": len(row["trip_s"]) + row["none"], "none": row["none"], "edition": eds_of(cls)}
        if row["trip_s"]:
            o.update(trip_s=[min(row["trip_s"]), max(row["trip_s"])], rep_at=[min(row["rep_at"]), max(row["rep_at"])],
                     rep_max_after=[min(row["rep_max_after"]), max(row["rep_max_after"])])
            cs = [x for x in row["clear_s"] if not math.isnan(x)]
            o["clear_s"] = [min(cs), max(cs)] if cs else None
            o["clear_none"] = len(row["clear_s"]) - len(cs)
        if row["rep_end_hold_none"]:
            o["rep_end_none"] = [min(row["rep_end_hold_none"]), max(row["rep_end_hold_none"])]
        out["rows"][f"{cls}|{tq}"] = o
    _OBS_CACHE[tag] = out
    return out


def eds_of(cls):
    return obs_editions()[cls]


def obs_key(cls, tq):
    return f"{cls}.t{_name(('%g' % tq).replace('.', 'p'))}"


def obs_table(obs=None):
    o = obs or observe()
    out = {"cases": leaf(o["cases"], "-", STATUS["obs"], "tables.observation_cases"),
           "hold_s": leaf(OBS_HOLD_S, "s", STATUS["obs"], "tables.OBS_HOLD_S"),
           "after_s": leaf(OBS_AFTER_S, "s", STATUS["obs"], "tables.OBS_AFTER_S")}
    for k, row in o["rows"].items():
        cls, tq = k.split("|")
        tq = float(tq)
        node = out.setdefault(cls, {}).setdefault(obs_key(cls, tq).split(".")[1], {})
        node["torque"] = leaf(tq, "N m", STATUS["obs"], "tables.OBS_TORQUES")
        for f in ("n", "none", "trip_s", "rep_at", "rep_max_after", "clear_s", "clear_none", "rep_end_none"):
            if row.get(f) is not None:
                node[f] = leaf(row[f], "s or deg C", STATUS["obs"], "thermal.hold_observations")
    return out


def _derived(T):
    n = T.number
    out = {}

    def d(name, value, formula, unit="-"):
        out[name] = leaf(value, unit, STATUS["derived"], formula)

    mm = lambda k: 1000.0 * n(k)
    d("foot_size_mm", [2000.0 * n("profile_cell:FOOT_HALF.0"), 2000.0 * n("profile_cell:FOOT_HALF.1"),
                       2000.0 * n("profile_cell:FOOT_HALF.2")], "2 x cell FOOT_HALF x 1000", "mm")
    d("cup_inner_mm", [2000.0 * (n("profile_cell:FOOT_HALF.0") + n("profile_cell:CUP_CLEAR")),
                       2000.0 * (n("profile_cell:FOOT_HALF.1") + n("profile_cell:CUP_CLEAR"))],
      "2 x (cell FOOT_HALF + CUP_CLEAR) x 1000", "mm")
    d("cup_clear_mm", mm("profile_cell:CUP_CLEAR"), "cell CUP_CLEAR x 1000", "mm")
    d("cup_wall_mm", [2000.0 * n("profile_cell:CUP_WALL.0"), 2000.0 * n("profile_cell:CUP_WALL.1")],
      "2 x cell CUP_WALL (thickness, height) x 1000", "mm")
    d("feet_xy_mm", [[1000.0 * n(f"profile_cell:FOOT_XY.{k}.0"), 1000.0 * n(f"profile_cell:FOOT_XY.{k}.1")]
                     for k in range(4)], "cell FOOT_XY x 1000", "mm")
    d("feet_spacing_mm", [2000.0 * abs(n("profile_cell:FOOT_XY.0.0")), 2000.0 * abs(n("profile_cell:FOOT_XY.0.1"))],
      "2 x |cell FOOT_XY[0]| x 1000", "mm")
    d("deck_mm", [2000.0 * n("profile_cell:DECK_HALF.1"), 2000.0 * n("profile_cell:DECK_HALF.0"),
                  2000.0 * n("profile_cell:DECK_THICK_HALF")], "2 x cell DECK_HALF (y, x), DECK_THICK_HALF x 1000",
      "mm")
    d("deck_top_below_grip_mm", -1000.0 * (n("profile_cell:DECK_Z") + n("profile_cell:DECK_THICK_HALF")),
      "-(cell DECK_Z + DECK_THICK_HALF) x 1000", "mm")
    d("deck_bottom_above_feet_plane_mm",
      1000.0 * (n("profile_cell:GRIP_ABOVE_FEET") + n("profile_cell:DECK_Z") - n("profile_cell:DECK_THICK_HALF")),
      "(cell GRIP_ABOVE_FEET + DECK_Z - DECK_THICK_HALF) x 1000", "mm")
    d("cut_x_mm", [1000.0 * n("profile_cell:CUT_X.0"), 1000.0 * n("profile_cell:CUT_X.1")], "cell CUT_X x 1000", "mm")
    d("cut_y_in_mm", mm("profile_cell:CUT_Y_IN"), "cell CUT_Y_IN x 1000", "mm")
    d("deck_half_len_mm", mm("profile_cell:DECK_HALF.1"), "cell DECK_HALF[1] x 1000", "mm")
    d("deck_half_wid_mm", mm("profile_cell:DECK_HALF.0"), "cell DECK_HALF[0] x 1000", "mm")
    d("footing_mm", [2000.0 * n("profile_cell:FOOTING_HALF.0"), 2000.0 * n("profile_cell:FOOTING_HALF.1")],
      "2 x cell FOOTING_HALF x 1000", "mm")
    d("web_mm", [mm("profile_cell:WEB_WIDTH"), 1000.0 * (-n("profile_cell:FOOTING_HALF.0") - n("profile_cell:CUT_X.0"))],
      "cell WEB_WIDTH x 1000; its length -(FOOTING_HALF[0] + CUT_X[0]) x 1000", "mm")
    d("block_mm", [2000.0 * n("profile_cell:block_half.0"), 1000.0 * n("profile_cell:BLOCK_WIDTH"),
                   2000.0 * n("profile_cell:block_half.2")], "the grip block: 2 x half-sizes x 1000", "mm")
    d("neck_mm", [2000.0 * n("profile_cell:neck_half.0"), 2000.0 * n("profile_cell:neck_half.1")],
      "the neck's section: 2 x half-sizes x 1000", "mm")
    d("grip_spacing_mm", 2000.0 * n("profile_cell:GRIP_Y"), "2 x cell GRIP_Y x 1000", "mm")
    d("grip_above_feet_mm", mm("profile_cell:GRIP_ABOVE_FEET"), "cell GRIP_ABOVE_FEET x 1000", "mm")
    d("grip_y_mm", mm("profile_cell:GRIP_Y"), "cell GRIP_Y x 1000", "mm")
    d("grip_tol_mm", 0.5, "grip points 360.0 +-0.5 mm, 60.0 +-0.5 mm (the drawing's tolerance)", "mm")
    d("F_plate_mm", [2000.0 * n("profile_cell:F_plate_half.0"), 2000.0 * n("profile_cell:F_plate_half.1"),
                     2000.0 * n("profile_cell:F_plate_half.2")], "2 x cell F base half-sizes x 1000", "mm")
    d("bench_top_mm", 1000.0 * n("profile_contract:BENCH_TOP_Z_M"), "contract BENCH_TOP_Z_M x 1000", "mm")
    d("bench_tol_mm", 1000.0 * n("profile_contract:BENCH_TOL_M"), "contract BENCH_TOL_M x 1000", "mm")
    d("F_tol_mm", 1000.0 * n("profile_contract:F_SURVEY_TOL.m"), "contract F_SURVEY_TOL.m x 1000", "mm")
    d("S_tol_mm", 1000.0 * n("profile_contract:S_SURVEY_TOL.m"), "contract S_SURVEY_TOL.m x 1000", "mm")
    d("S_x_m", [n("profile_contract:S_REFERENCE_POINT.0") - n("profile_cell:S_HALF.0"),
                n("profile_contract:S_REFERENCE_POINT.0") + n("profile_cell:S_HALF.0")],
      "S's plate: S_REFERENCE_POINT x -+ cell S_HALF[0]", "m")
    d("S_depth_m", 2.0 * n("profile_cell:S_HALF.0"), "2 x cell S_HALF[0]", "m")
    d("S_plate_mm", 2000.0 * n("profile_cell:S_HALF.2"), "2 x cell S_HALF[2] x 1000", "mm")
    d("S_underside_z_m", n("profile_contract:S_TOP_Z_M") - 2.0 * n("profile_cell:S_HALF.2"),
      "contract S_TOP_Z_M - 2 x cell S_HALF[2]", "m")
    d("F_front_gap_mm", 1000.0 * ((n("profile_contract:S_REFERENCE_POINT.0") - n("profile_cell:S_HALF.0")) -
                                  (n("profile_contract:F_REFERENCE_POINT.0") + n("profile_cell:DECK_HALF.0"))),
      "(S's front edge) - (F's x + the deck's half width), x 1000", "mm")
    d("clear_grow_mm", 1000.0 * n("profile_contract:CLEAR_VOLUME.grow_m"), "contract CLEAR_VOLUME.grow_m x 1000", "mm")
    d("centre_along_mm", 1000.0 * n("profile_contract:CENTRE_REGION.along_m"), "contract CENTRE_REGION x 1000", "mm")
    d("centre_fore_aft_mm", 1000.0 * n("profile_contract:CENTRE_REGION.fore_aft_m"), "contract CENTRE_REGION x 1000",
      "mm")
    d("centre_below_mm", [1000.0 * n("profile_contract:CENTRE_REGION.below_m.0"),
                          1000.0 * n("profile_contract:CENTRE_REGION.below_m.1")], "contract CENTRE_REGION x 1000",
      "mm")
    d("capture_mm", 1000.0 * n("profile_contract:CAPTURE_BOX.m"), "contract CAPTURE_BOX.m x 1000", "mm")
    d("fit_mm", 1000.0 * n("profile_contract:FIT_NOTE.m"), "contract FIT_NOTE.m x 1000", "mm")
    d("k_less_pct", 100.0 * (1.0 - n("profile_stage05:C3.0")), "(1 - profile C3 low end) x 100", "per cent")
    d("k_more_pct", 100.0 * (n("profile_stage05:C3.1") - 1.0), "(profile C3 high end - 1) x 100", "per cent")
    d("mount_mm", 1000.0 * n("profile_stage05:PL1_translation.1"), "profile PL1.translation high end x 1000", "mm")
    from openarm_unit import hs
    hsl = hs.HSLaw.from_law(T.files["law"])
    r_hs = n("profile_stage05:HS1_relative_rms")
    d("hs_rms_pct", 100.0 * r_hs, "profile HS1.relative_rms x 100", "per cent")
    d("hs_peak_pct", 100.0 * hs.amplitude(r_hs) * (hsl.harmonic_weights[0] + hsl.harmonic_weights[1]),
      "hs.amplitude(profile HS1.relative_rms) x (w1 + w2) x 100: the largest excursion of the level", "per cent")
    d("hs_period_deg", [math.degrees(n("profile_stage05:HS1_period.0")), math.degrees(n("profile_stage05:HS1_period.1"))],
      "profile HS1.period in degrees", "deg")
    d("for_ms", 1000.0 * n("profile_contract:CAPACITY_RULE.for_s"), "contract CAPACITY_RULE.for_s x 1000", "ms")
    d("hold_window_s", n("profile_contract:HOLD_WINDOW_S"), "contract HOLD_WINDOW_S", "s")
    d("trial_lift_mm", 1000.0 * n("profile_contract:TRIAL_LIFT_MAX_M"), "contract TRIAL_LIFT_MAX_M x 1000", "mm")
    d("lifted_mm", 1000.0 * n("profile_contract:LIFTED_POINT_M"), "contract LIFTED_POINT_M x 1000", "mm")
    d("pre_place_mm", 1000.0 * n("profile_contract:PRE_PLACE_ABOVE_SEAT_M"), "contract PRE_PLACE_ABOVE_SEAT_M x 1000",
      "mm")
    d("drop_mm", 1000.0 * n("profile_contract:DROP_FEET_ABOVE_REST_M"), "contract DROP_FEET_ABOVE_REST_M x 1000", "mm")
    d("seat_grow_mm", 1000.0 * n("profile_contract:S_SEAT_FRAME_GROW_M"), "contract S_SEAT_FRAME_GROW_M x 1000", "mm")
    d("on_time_pct", 100.0 * n("profile_contract:ON_TIME_SHARE_MIN"), "contract ON_TIME_SHARE_MIN x 100", "per cent")
    d("late_pct", 100.0 * n("profile_contract:LATE_TIME_STOP_SHARE"), "contract LATE_TIME_STOP_SHARE x 100",
      "per cent")
    d("cycle_ms", 1000.0 * n("profile_contract:CYCLE_S"), "contract CYCLE_S x 1000", "ms")
    if "profile_obs" in T.files:
        d("obs_hold_min", n("profile_obs:hold_s") / 60.0, "observation hold_s / 60", "min")
        d("obs_after_s", n("profile_obs:after_s"), "observation after_s", "s")
    d("stand_in_F_mm", 1000.0 * (n("profile_cell:CUP_CLEAR") + n("profile_grader:feet_in_cups_tol_xy")),
      "(cell CUP_CLEAR + Cell.feet_in_cups tol_xy) x 1000", "mm")
    d("plate_tol_mm", 1000.0 * n("profile_grader:feet_in_cups_tol_z"), "Cell.feet_in_cups tol_z x 1000", "mm")
    d("S_top_tol_mm", 1000.0 * n("profile_grader:S_FOOT_ON_TOP_TOL_M"), "ref/quantities.py S_FOOT_ON_TOP_TOL_M x 1000",
      "mm")
    st = lambda bits, lo, hi: (hi - lo) / (2 ** bits - 1)
    bits = lambda f: n(f"profile_contract:MIT_FIELDS.{f}.0")
    d("step_p", st(bits("p"), -n("profile_contract:PMAX"), n("profile_contract:PMAX")),
      "contract field_step(16, -PMAX, PMAX)", "rad")
    d("step_kp", st(bits("kp"), n("profile_contract:KP_RANGE.0"), n("profile_contract:KP_RANGE.1")),
      "contract field_step(12, KP_RANGE)", "N m/rad")
    d("step_kd", st(bits("kd"), n("profile_contract:KD_RANGE.0"), n("profile_contract:KD_RANGE.1")),
      "contract field_step(12, KD_RANGE)", "N m s/rad")
    for c in ("DM8009", "DM4340", "DM4310"):
        d(f"step_v_{c}", st(bits("v"), -n(f"profile_contract:VMAX.{c}"), n(f"profile_contract:VMAX.{c}")),
          f"contract field_step(12, +-VMAX[{c}])", "rad/s")
        d(f"step_t_{c}", st(bits("t"), -n(f"profile_contract:TMAX.{c}"), n(f"profile_contract:TMAX.{c}")),
          f"contract field_step(12, +-TMAX[{c}])", "N m")
    d("step_v_lim", 1.0 / n("profile_contract:POSFORCE_FIELDS.v_lim.1"), "1 / contract POSFORCE_FIELDS.v_lim scale",
      "rad/s")
    d("step_i_lim", 1.0 / n("profile_contract:POSFORCE_FIELDS.i_lim.1"), "1 / contract POSFORCE_FIELDS.i_lim scale",
      "per unit")
    d("step_budget_ms", 1000.0 * n("profile_contract:STEP_BUDGET_CPU_S"), "contract STEP_BUDGET_CPU_S x 1000", "ms")
    if "profile_runtime" in T.files and T.files["profile_runtime"]:
        R = "profile_runtime:"
        d("rt_cycle_ms", 1000.0 * n(R + "cycle_s"), "runtime cycle_s x 1000", "ms")
        d("rt_step_budget_ms", 1000.0 * n(R + "step_budget_cpu_s"), "runtime step_budget_cpu_s x 1000", "ms")
        d("rt_on_time_pct", 100.0 * n(R + "on_time_share_min"), "runtime on_time_share_min x 100", "per cent")
        d("rt_late_pct", 100.0 * n(R + "late_time_stop_share"), "runtime late_time_stop_share x 100", "per cent")
        d("rt_reply_line_mib", n(R + "reply_line_max_bytes") / 1048576.0, "runtime reply_line_max_bytes / 2^20", "MiB")
    d("timeout_count_us", 1e6 * n("public:frames.timeout_count_s"), "params_public frames.timeout_count_s x 1e6", "us")
    d("step_ms", 1000.0 * n("profile_contract:STEP_S"), "contract STEP_S x 1000", "ms")
    return out


GRADER_VERIFY = os.path.join(_genenv.TASK_DIR, "grader", "verify.py")
GRADER_NETAUDIT = os.path.join(_genenv.TASK_DIR, "grader", "kit88", "arm", "assembly_info", "netaudit.py")


def _module_constants(path, names):
    import ast
    with open(path, encoding="utf-8") as f:
        tree = ast.parse(f.read())
    out = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id in names:
                out[node.targets[0].id] = ast.literal_eval(node.value)
    return out


def gates_facts():
    cfg = _module_constants(GRADER_VERIFY, ("DEFAULT_CFG",))["DEFAULT_CFG"]
    na = _module_constants(GRADER_NETAUDIT, ("NETWORK_MODULES", "DYNAMIC_IMPORT_CALLS", "CODE_CALLS",
                                             "PROCESS_CALLS"))
    return {"max_submission_mb": cfg["max_submission_mb"], "network_modules": list(na["NETWORK_MODULES"]),
            "dynamic_import_calls": list(na["DYNAMIC_IMPORT_CALLS"]), "code_calls": list(na["CODE_CALLS"]),
            "process_calls": list(na["PROCESS_CALLS"])}


def gates_table():
    g = gates_facts()
    return {"max_submission_mb": leaf(float(g["max_submission_mb"]), "MB", "grader",
                                      "grader/verify.py DEFAULT_CFG max_submission_mb")}


def import_audit_words():
    g = gates_facts()
    words = lambda xs: ", ".join(xs[:-1]) + " or " + xs[-1]
    computed = [c for c in g["dynamic_import_calls"] if c not in ("spec_from_file_location", "load_module")]
    return {"modules": words(g["network_modules"]), "dynamic": words(computed),
            "dynamic_all": words(g["dynamic_import_calls"]), "code": words(g["code_calls"]),
            "process": words(g["process_calls"])}


def vendor_facts_table():
    with open(os.path.join(_genenv.DATA, "vendor_facts.json"), encoding="utf-8") as f:
        d = json.load(f)
    return {k: leaf(v["value"], v.get("unit", "-"), "fact", v["source"]) for k, v in d["facts"].items()}


def build(with_obs=True, extra=None):
    prof = {"stage05": profile_table(), "contract": contract_table(), "cell": cell_table(), "curves": curves_table(),
            "grader": grader_table(), "rule": rule_table(), "runtime": runtime_table(), "gates": gates_table()}
    if with_obs:
        prof["obs"] = obs_table()
    for k, v in (extra or {}).items():
        prof[k] = v
    T = docs.Tables(profiles=prof)
    prof["derived"] = _derived(T)
    return docs.Tables(profiles=prof)


def sources_sha256():
    import hashlib
    out = {}
    for name, p in (("params_public.json", os.path.join(_genenv.UNIT_PKG, "params_public.json")),
                    ("params_law.json", os.path.join(_genenv.UNIT_PKG, "params_law.json")),
                    ("profile_stage05.json", _genenv.PROFILE_PATH),
                    ("constants.py", os.path.join(_genenv.CONTRACT, "constants.py")),
                    ("shift.py", os.path.join(_genenv.REF, "shift.py")),
                    ("quantities.py", os.path.join(_genenv.REF, "quantities.py")),
                    ("band_rule.json", os.path.join(_genenv.RULES, "band_rule.json")),
                    ("vendor_curves.json", os.path.join(_genenv.DATA, "vendor_curves.json")),
                    ("thermal.py", os.path.join(_genenv.UNIT_PKG, "thermal.py")),
                    ("runtime/publish.py", os.path.join(_genenv.RUNTIME, "publish.py")),
                    ("runtime/frames.py", os.path.join(_genenv.RUNTIME, "frames.py"))):
        if not os.path.isfile(p):
            out[name] = None
            continue
        with open(p, "rb") as f:
            out[name] = hashlib.sha256(f.read()).hexdigest()
    out["grader gate facts (verify.py DEFAULT_CFG, netaudit lists)"] = hashlib.sha256(
        json.dumps(gates_facts(), sort_keys=True).encode()).hexdigest()
    out["runtime/wire.py"] = None
    p = os.path.join(_genenv.RUNTIME, "wire.py")
    if os.path.isfile(p):
        with open(p, "rb") as f:
            out["runtime/wire.py"] = hashlib.sha256(f.read()).hexdigest()
    return out
