import inspect
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import common as C

SH, QT, K = C.SH, C.QT, C.K
STATUS = "grader"
PENDING = "pending the freeze"


def leaf(value, unit="-", status=STATUS, source="-"):
    return {"value": C.plain(value), "unit": unit, "status": status, "source": source}


def published_table(frozen=None):
    out = {}
    for k, unit in (("G_ABS", "m/s^2"), ("RELEASED_CLAMP_N", "N"), ("S_FOOT_ON_TOP_TOL_M", "m"),
                    ("VISIT_TOL_M", "m"), ("FOOT_ON_PLATE_TOL_M", "m")):
        out[k] = leaf(getattr(QT, k), unit, source="ref/quantities.py " + k)
    out["SHELF_CARRIES_N"] = leaf(SH.Monitor.SHELF_CARRIES_N, "N", source="ref/shift.py Monitor.SHELF_CARRIES_N")
    sig = inspect.signature(SH.Cell.feet_in_cups)
    for k in ("tol_xy", "tol_z"):
        out["feet_in_cups_" + k] = leaf(sig.parameters[k].default, "m",
                                        source="ref/shift.py Cell.feet_in_cups default " + k)
    out["CUP_CLEAR"] = leaf(SH.Cell.CUP_CLEAR, "m", source="ref/shift.py Cell.CUP_CLEAR")
    sig = inspect.signature(SH.Cell.stands_on_S)
    out["stands_on_S_tol_z"] = leaf(sig.parameters["tol_z"].default, "m",
                                    source="ref/shift.py Cell.stands_on_S default tol_z")
    for k, unit in (("S_SEAT_FRAME_GROW_M", "m"), ("CAPTURE_CLAMP_N", "N"), ("RELEASE_CLAMP_N", "N"),
                    ("CLAMP_LEVER_M", "m"), ("DROP_FEET_ABOVE_REST_M", "m"), ("SPILL_DEG", "deg"),
                    ("STOP_MARGIN_DEG", "deg"), ("OT_VALUE_C", "deg C"), ("DRIVER_STAGE_C", "deg C"),
                    ("TRIAL_LIFT_MAX_M", "m"), ("LIFTED_POINT_M", "m"), ("ON_TIME_SHARE_MIN", "-"),
                    ("LATE_TIME_STOP_SHARE", "-"), ("MOVE_S", "s"), ("HOLD_S", "s"), ("HOLD_WINDOW_S", "s"),
                    ("CARRY_START_S", "s"), ("REPEAT_DIRECTION_MIN_DEG", "deg"), ("N_UNITS", "-")):
        out[k] = leaf(getattr(K, k), unit, source="contract/constants.py " + k)
    out["CAPTURE_BOX"] = leaf(dict(K.CAPTURE_BOX), "m; deg", source="contract/constants.py CAPTURE_BOX")
    for k in ("for_s", "armed_after_capture_s", "mu_printed_min"):
        out["CAPACITY_RULE_" + k] = leaf(K.CAPACITY_RULE[k], "s" if k.endswith("_s") else "-",
                                         source="contract/constants.py CAPACITY_RULE " + k)
    for k in ("closing_end", "decision_end", "decline_put_back_s", "set_down_s"):
        if k in K.SLOT:
            out["SLOT_" + k] = leaf(K.SLOT[k], "s", source="contract/constants.py SLOT " + k)
    out["SETUP_BUDGET"] = leaf(dict(K.SETUP_BUDGET), "s", source="contract/constants.py SETUP_BUDGET")
    out["STEP_BUDGET_CPU_S"] = leaf(K.STEP_BUDGET_CPU_S, "s", source="contract/constants.py STEP_BUDGET_CPU_S")
    w = C.weights_of_record()
    out["weights_Q"] = leaf(w["Q"], "-", source="rules/band_rule.json weights_of_record.Q")
    out["weights_unit"] = leaf(w["unit"], "-", source="rules/band_rule.json weights_of_record.unit")
    out["weights_tray_place"] = leaf(w["tray_place"], "-", source="rules/band_rule.json weights_of_record.tray_place")
    out["three_x_ratio"] = leaf(C.three_x_ratio(), "-", source="rules/band_rule.json three_x_rule.ratio")
    for key, f in C.floor_table().items():
        out["floor_" + key] = leaf(f["floor"], f["unit"], source=f["source"])
    keys = ("e_z_mm", "tilt_deg", "rho_mm", "e_xy_mm", "d_m_mm", "P_r_N")
    units = {k: C.floor_table()[k]["unit"] for k in keys}
    if frozen is None:
        for k in keys:
            for e in ("full_end", "zero_end", "passes"):
                out[f"{e}_{k}"] = leaf(None, units[k] if e != "passes" else "-", PENDING, "the frozen set")
        for e in ("terms_kept", "weights_renormalised", "place_passes", "anchor"):
            out[e] = leaf(None, "-", PENDING, "the frozen set")
        return out
    d = frozen["decision"]
    src = f"the frozen set {frozen.get('label')} ({frozen.get('kind')}), digest {frozen.get('digest', '')[:16]}"
    for k in keys:
        q = d["quantities"][k]
        out["full_end_" + k] = leaf(q["F"], units[k], source=src + " decision.quantities." + k + ".F")
        out["zero_end_" + k] = leaf(q["zero_end"]["Z"], units[k], source=src + " decision.quantities." + k +
                                    ".zero_end.Z")
        out["passes_" + k] = leaf(bool(q["passes"]), "-", source=src + " decision.quantities." + k + ".passes")
    out["terms_kept"] = leaf(d["terms"], "-", source=src + " decision.terms")
    out["weights_renormalised"] = leaf(d["weights"], "-", source=src + " decision.weights")
    sh = bool(frozen.get("audit", {}).get("S_height_shown", True))
    out["place_passes"] = leaf(bool(d["place_passes"]) and sh, "-", source=src + " decision.place_passes and "
                                                                             "audit.S_height_shown")
    out["anchor"] = leaf(frozen["anchor"]["raw"], "-", source=src + " anchor.raw")
    return out
