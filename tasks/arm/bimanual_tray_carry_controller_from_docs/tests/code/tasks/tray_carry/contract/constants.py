import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
TASK_DIR = os.path.dirname(HERE)
BAND_RULE_PATH = os.path.join(TASK_DIR, "rules", "band_rule.json")
PROFILE_PATH = os.path.join(HERE, "profile_stage05.json")
PRECHECK_RULE_PATH = os.path.join(TASK_DIR, "evidence", "pre-check", "rule, not shipped")
UNIT_DIR = os.path.normpath(os.path.join(TASK_DIR, "..", "..", "shared", "openarm_unit"))

TASK = "bimanual_tray_carry_controller_from_docs"

SIDES = ("left", "right")
_ARM_CLASSES = (("DM8009", "DM-J8009P"), ("DM8009", "DM-J8009P"), ("DM4340", "DM-J4340P"), ("DM4340", "DM-J4340"),
                ("DM4310", "DM-J4310"), ("DM4310", "DM-J4310"), ("DM4310", "DM-J4310"))
_GRIPPER_CLASS = ("DM4310", "DM-J4310")
MODULES = tuple(
    {"index": 8 * b + j, "side": side, "joint": j + 1, "label": f"{side} J{j + 1}",
     "class": (_ARM_CLASSES[j] if j < 7 else _GRIPPER_CLASS)[0],
     "product": (_ARM_CLASSES[j] if j < 7 else _GRIPPER_CLASS)[1],
     "kind": "arm" if j < 7 else "gripper"}
    for b, side in enumerate(SIDES) for j in range(8))
N_MODULES = len(MODULES)

PMAX = 12.5
VMAX = {"DM8009": 45.0, "DM4340": 10.0, "DM4310": 30.0}
TMAX = {"DM8009": 54.0, "DM4340": 28.0, "DM4310": 10.0}
KP_RANGE = (0.0, 500.0)
KD_RANGE = (0.0, 5.0)
MIT_FIELDS = {
    "p": (16, "+-PMAX"), "v": (12, "+-VMAX[class]"), "kp": (12, "KP_RANGE"), "kd": (12, "KD_RANGE"), "t": (12, "+-TMAX[class]")}
MIT_QUANTISATION = "truncation"
POSFORCE_FIELDS = {
    "p": ("float32", None), "v_lim": ("uint16", 100.0), "i_lim": ("uint16", 10000.0)}
V_LIM_RANGE = (0.0, 100.0)
I_LIM_RANGE = (0.0, 1.0)
REPLY_FIELDS = {"q": 16, "dq": 12, "tau": 12}
REPLY_TEMPERATURES = ("t_mos", "t_rotor")
STATUS_CODES = {"0": "disabled", "1": "enabled", "9": "under voltage", "C": "coil over temperature", "D": "communication lost"}
GRIPPER_MODES = ("mit", "posforce")
FEEDBACK_KEYS = ("cycle", "site", "modules")
MODULE_FEEDBACK_KEYS = ("q", "dq", "tau", "t_mos", "t_rotor", "status", "age")


def field_step(bits, lo, hi):
    return (hi - lo) / (2 ** bits - 1)


POSITION_STEP = field_step(16, -PMAX, PMAX)

STEP_S = 0.001
CYCLE_S = 0.004
STEPS_PER_CYCLE = round(CYCLE_S / STEP_S)
FEEDBACK_AGE_CYCLES = 1

HANDOVER_HOLD_S = 2.0
FREE_INTERVAL_S = 8.0
HOME_POSE_RAD = (0.0, 0.0, 0.0, 1.5708, 0.0, 0.0, 0.0)
MOVE_S = 1.16
HOLD_S = 0.9
TARGET_S = MOVE_S + HOLD_S
HOLD_WINDOW_S = 0.4
SLOT = {
    "closing_end": 5.0,
    "decision_end": 9.0,
    "set_down_s": 6.0,
    "retreat_end_after_Te": 9.0,
    "next_tray_after_Te": 12.0,
    "decline_put_back_s": 4.0,
    "decline_clear_s": 7.0,
    "decline_next_s": 10.0,
    "missed_clear_end": 12.0,
    "missed_next": 15.0,
}
CARRY_START_S = SLOT["decision_end"]
BETWEEN_TRAYS_S = (3.0, 3.0)
TRIAL_LIFT_MAX_M = 0.015
LIFTED_POINT_M = 0.010
PRE_PLACE_ABOVE_SEAT_M = 0.020
CLEAR_VOLUME = {"grow_m": 0.040, "z_min_m": 0.26}


def carry_end_s(n_moves):
    return CARRY_START_S + TARGET_S * n_moves


def slot_length_s(n_moves):
    return carry_end_s(n_moves) + SLOT["next_tray_after_Te"]


JOB_TARGETS = {"short": (12, 16), "medium": (24, 32)}
LAP_MOVES = 136
LAP_PAIRS_PER_JOINT = 8
LAP_CORNER_PAIRS = 12
BOX = {"x": (0.20, 0.40), "y": (-0.08, 0.08), "z": (0.36, 0.42)}
BOX_GRID = (6, 5, 3)
FIXED_FIRST = (0.22, 0.0, 0.38)
FIXED_LAST = ((0.36, 0.0, 0.40), (0.40, 0.0, 0.36))
REPEAT_TARGETS_MIN = 4
REPEAT_DIRECTION_MIN_DEG = 60.0
MOVE_MIN_M = 0.06
GENERATOR_STOP_MARGIN_DEG = 8.0
GENERATOR_SWIVEL_DEG = (20.0, 60.0)
GENERATOR_TRAY_TO_S_MIN_M = 0.015
GENERATOR_SPEED_SHARE_MAX = 0.6
GENERATOR_TORQUE_SHARE_MAX = 0.5

BENCH_TOP_Z_M = 0.2200
BENCH_TOL_M = 0.0005
F_REFERENCE_POINT = (0.220, 0.000, 0.280)
F_SURVEY_TOL = {"m": 0.0003, "deg": 0.05}
S_TOP_Z_M = 0.280
S_REFERENCE_POINT = (0.400, 0.000, 0.340)
S_SURVEY_TOL = {"m": 0.0020, "deg": 0.3}
S_SEAT_FRAME_GROW_M = 0.010
HANDLE_SPACING_M = 0.360
GRIP_ABOVE_FEET_M = 0.060
TRAY_EMPTY_KG = 0.62
TRAY_DECK_M = (0.400, 0.150)
FIT_NOTE = {"m": 0.0010, "deg": 0.5}
SITE_AMBIENT_C = (20.0, 30.0)

CENTRE_REGION = {"along_m": 0.040, "fore_aft_m": 0.005, "below_m": (0.0, 0.045)}
TRAY_CLASSES = {
    "band": {"kg": (0.9, 1.6), "jobs": ("short", "medium", "lap")},
    "proof": {"kg": (1.7, 2.2), "jobs": ("short",)},
    "decline": {"kg": (9.0, 10.0), "jobs": ("lap",), "centre_along_m": (0.030, 0.040)},
}
PER_UNIT = {"short": 3, "short_proof": 1, "medium": 1, "lap_band": 1, "decline": (1, 2)}
PRINTED_BAND_KG = "about 0.9 to 1.6 kg"
PRINTED_HEAVIER = "heavier trays, up to about 10 kg, are sometimes offered"

CLAMP_LEVER_M = 0.067
CAPTURE_BOX = {"m": 0.008, "deg": 4.0}
CAPTURE_CLAMP_N = 5.0
RELEASE_CLAMP_N = 2.0
SEAT_RAMP_S = 0.2
CAPACITY_RULE = {
    "slip_if": "F_s > mu x F_c or F_n > F_c",
    "for_s": 0.020,
    "armed_after_capture_s": 0.5,
    "mu_printed_min": 0.55,
}
DROP_FEET_ABOVE_REST_M = 0.010

SPILL_DEG = 10.0
STOP_MARGIN_DEG = 0.5
OT_VALUE_C = 100.0
DRIVER_STAGE_C = 120.0
TIMEOUT_COUNTS = 0
GRIPPER_CURRENT_BASE_A = 10.261194
EVENTS = {
    "E1": "module protection (coil over-temperature at OT_Value; driver stage at 120 deg C)",
    "E2": "slip (capacity rule) or drop",
    "E3": "spill (deck more than SPILL_DEG from level between capture and release)",
    "E4": "contact other than pads on a grip block, feet in F's cups or on S's top, and one hand's two fingers",
    "E5": "an arm joint within STOP_MARGIN_DEG of an end of its range",
}
NOT_EVENTS = ("missed capture", "late or soft set-down", "declined tray", "press on the shelf")

SETUP_BUDGET = {"cpu_s": 60.0, "wall_s": 120.0}
STEP_BUDGET_CPU_S = CYCLE_S
STEP_FATAL = {"cpu_s": 1.0, "wall_s": 5.0}
ON_TIME_SHARE_MIN = 0.90
LATE_TIME_STOP_SHARE = 0.10
VERIFIER_TIMEOUT_S = 9000.0
MEMORY_PER_SHIFT_GB = 2.0
N_UNITS = 3

REFERENCE_KP = (400.0, 400.0, 300.0, 300.0, 60.0, 60.0, 60.0)
REFERENCE_KD = (5.0, 5.0, 4.0, 4.0, 1.5, 1.5, 1.5)
LABEL_GRASP_SHARE_MAX = 0.80
LABEL_WINDING_MARGIN_K = 10.0
LABEL_DECLINE_CARRY_SHARE = 0.8
LABEL_SWIVEL_SEARCH_DEG = (0.0, 80.0)


def band_rule():
    with open(BAND_RULE_PATH, encoding="utf-8") as f:
        return json.load(f)


def weights():
    return band_rule()["weights_of_record"]


def credit(x, full, zero):
    if not zero > full:
        raise ValueError("the zero end must lie above the full end")
    if x <= full:
        return 1.0
    if x >= zero:
        return 0.0
    return (zero - x) / (zero - full)


def renormalise(w, keep):
    s = sum(w[k] for k in keep)
    if s <= 0:
        raise ValueError("no weight left")
    return {k: w[k] / s for k in keep}


def q_value(terms, scored=None):
    w = weights()["Q"]
    keep = tuple(k for k in w if (scored is None or k in scored))
    wn = renormalise(w, keep)
    return sum(wn[k] * terms[k] for k in keep)


def tray_value(completed, q, place):
    if not completed:
        return 0.0
    if place is None:
        return q
    t = weights()["tray_place"]
    return q * (t["base"] + t["place"] * place)


def youden_j(declined_decline, n_decline, declined_carry, n_carry):
    tpr = declined_decline / n_decline if n_decline else 0.0
    fpr = declined_carry / n_carry if n_carry else 0.0
    return max(0.0, tpr - fpr)


def unit_value(carry_trays, j):
    w = weights()["unit"]
    mean = sum(carry_trays) / len(carry_trays) if carry_trays else 0.0
    return w["carry_trays"] * mean + w["J"] * j


def reward(units, gates, anchor):
    raw = (1.0 if gates else 0.0) * sum(units) / len(units)
    return min(1.0, raw / anchor)
