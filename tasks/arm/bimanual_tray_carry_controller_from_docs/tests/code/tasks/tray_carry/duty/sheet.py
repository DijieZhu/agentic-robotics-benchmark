import hashlib
import importlib.util
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TASK_DIR = os.path.dirname(HERE)


def _load_constants():
    name = "t05_contract_constants"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, os.path.join(TASK_DIR, "contract", "constants.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


K = _load_constants()

SHEET_FORMAT_VERSION = 1
JOB_KINDS = ("short", "medium", "lap")
TIME_DECIMALS = 3
POSITION_DECIMALS = 6

SHEET_FIELDS = [
    {"name": "format_version", "type": "integer", "unit": None,
     "meaning": "Version of the shift sheet's format (1)."},
    {"name": "cycle_s", "type": "number", "unit": "s",
     "meaning": "The exchange cycle (0.004 s). Every time of the sheet is a multiple of it; a slot time t corresponds to "
                "the supervisor's slot_cycle t / cycle_s."},
    {"name": "time_origin", "type": "string", "unit": None,
     "meaning": "Shift times count from cycle 0, the end of the hand-over; slot times count from the cycle in which the "
                "supervisor's signal presents the slot's tray standing in F."},
    {"name": "frame", "type": "string", "unit": None,
     "meaning": "Positions are tray reference points in the cell frame of the cell sheet (m), the tray level, its handle "
                "line along y."},
    {"name": "first_presentation_s", "type": "number", "unit": "s",
     "meaning": "Shift time at which the first tray is presented (the free interval after the hand-over)."},
    {"name": "move_s", "type": "number", "unit": "s",
     "meaning": "Duration of one move of the timed reference: the reference point travels on the straight line from the "
                "previous target to the next, its path length by a trapezoid of equal thirds, the tray keeping its "
                "orientation."},
    {"name": "hold_s", "type": "number", "unit": "s",
     "meaning": "Duration of the hold that follows every move (the reference point stays at the target)."},
    {"name": "lifted_point_above_seated_m", "type": "number", "unit": "m",
     "meaning": "The carry starts at the lifted point L, this far above the reference point of the tray standing in F."},
    {"name": "slots", "type": "array of slot objects", "unit": None,
     "meaning": "The tray slots of the shift in the order in which their trays are presented."},
    {"name": "slots[].tray", "type": "integer", "unit": None,
     "meaning": "Index of the slot (0 for the first), the value of the supervisor's signal site.tray while the slot's "
                "tray is presented."},
    {"name": "slots[].planned_presentation_s", "type": "number", "unit": "s",
     "meaning": "Shift time of the presentation when every earlier tray of the sheet is carried through its whole "
                "schedule; after a declined tray or a missed capture the next trays are presented earlier, and the "
                "supervisor's signal says when."},
    {"name": "slots[].job", "type": "string, one of short | medium | lap", "unit": None,
     "meaning": "The job's length class: short (12 to 16 targets), medium (24 to 32) or the lap (the acceptance "
                "list)."},
    {"name": "slots[].pick_rest", "type": "string", "unit": None,
     "meaning": "Where the tray stands when it is presented: F, the fixture."},
    {"name": "slots[].destination_rest", "type": "string", "unit": None,
     "meaning": "Where the carried tray is set down: S, the shelf."},
    {"name": "slots[].n_targets", "type": "integer", "unit": None,
     "meaning": "Number of targets of the job n (the first fixed target and the last two fixed targets included); the "
                "carry consists of n moves, each followed by a hold."},
    {"name": "slots[].windows", "type": "object", "unit": "s (slot time)",
     "meaning": "The slot's windows: closing [0, 5.0] (both grippers capture their necks by its end); "
                "trial_lift_and_decision [5.0, 9.0]; decline_accepted [0, 9.0]; carry [9.0, T_e] with T_e = 9.0 + "
                "n (move_s + hold_s); set_down [T_e, T_e + 6.0] (lowered from the pre-place point onto the destination "
                "rest, both grippers open); retreat [T_e + 6.0, T_e + 9.0] (both hands inside the clear volume by its "
                "end; then the supervisor takes the tray off S); next_tray_s T_e + 12.0. Each window is [start, end]."},
    {"name": "slots[].declined", "type": "object", "unit": "s after the decline",
     "meaning": "The schedule after a decline sent at slot time t_d inside decline_accepted (and not after a missed "
                "capture): put_back_by_s (t_d + 4.0: the tray seated and released in F, its put-back deadline), "
                "hands_clear_by_s (t_d + 7.0: both hands inside the clear volume), next_tray_after_s (t_d + 10.0). "
                "Values are the offsets from t_d."},
    {"name": "slots[].missed_capture", "type": "object", "unit": "s (slot time)",
     "meaning": "The schedule when a hand has not captured its neck by the end of closing and no decline was sent "
                "before it: the job has failed, a later decline is ignored; both grippers open and both hands inside "
                "the clear volume by clear_by_s (12.0); next tray at next_tray_s (15.0)."},
    {"name": "slots[].targets", "type": "array of target objects", "unit": None,
     "meaning": "The timed reference of the carry, in the job's order."},
    {"name": "slots[].targets[].k", "type": "integer", "unit": None,
     "meaning": "Index of the target in the job (0 for the first)."},
    {"name": "slots[].targets[].p", "type": "array of 3 numbers", "unit": "m",
     "meaning": "The target: the tray's reference point in the cell frame (x, y, z)."},
    {"name": "slots[].targets[].move_start_s", "type": "number", "unit": "s (slot time)",
     "meaning": "Start of the move to this target (from the previous target, or from L for k = 0): 9.0 + k (move_s + "
                "hold_s)."},
    {"name": "slots[].targets[].arrive_s", "type": "number", "unit": "s (slot time)",
     "meaning": "End of the move: the reference point is at the target (move_start_s + move_s)."},
    {"name": "slots[].targets[].hold_end_s", "type": "number", "unit": "s (slot time)",
     "meaning": "End of the hold at this target (arrive_s + hold_s); the next move starts here."},
]

SLOT_KEYS = ("tray", "planned_presentation_s", "job", "pick_rest", "destination_rest", "n_targets", "windows",
             "declined", "missed_capture", "targets")
TOP_KEYS = ("format_version", "cycle_s", "time_origin", "frame", "first_presentation_s", "move_s", "hold_s",
            "lifted_point_above_seated_m", "slots")
TARGET_KEYS = ("k", "p", "move_start_s", "arrive_s", "hold_end_s")
WINDOW_KEYS = ("closing", "trial_lift_and_decision", "decline_accepted", "carry", "set_down", "retreat", "next_tray_s")

FORBIDDEN_WORDS = ("mass", "kg", "centre", "center", "label", "decline_tray", "carry_tray", "band", "proof", "class",
                   "weight", "load", "entropy", "drawn", "true", "hidden", "unit", "torque", "swivel")

TIME_ORIGIN = ("shift times from cycle 0 (the end of the hand-over); slot times from the cycle in which the slot's tray "
               "is presented")
FRAME = "cell frame of the cell sheet; tray reference point; tray level; handle line along y"


def _t(x):
    c = round(x / K.CYCLE_S)
    if abs(c * K.CYCLE_S - x) > 1e-9:
        raise ValueError(f"time {x!r} is not a multiple of the cycle")
    return round(c * K.CYCLE_S, TIME_DECIMALS)


def _p(v):
    return [round(float(a), POSITION_DECIMALS) for a in v]


def slot_entry(index, job_kind, targets, planned_presentation_s):
    if job_kind not in JOB_KINDS:
        raise ValueError(f"job: one of {JOB_KINDS}")
    n = len(targets)
    S = K.SLOT
    Te = K.carry_end_s(n)
    tg = []
    for k, p in enumerate(targets):
        ms = K.CARRY_START_S + k * K.TARGET_S
        tg.append({"k": k, "p": _p(p), "move_start_s": _t(ms), "arrive_s": _t(ms + K.MOVE_S),
                   "hold_end_s": _t(ms + K.TARGET_S)})
    return {
        "tray": int(index),
        "planned_presentation_s": _t(planned_presentation_s),
        "job": job_kind,
        "pick_rest": "F",
        "destination_rest": "S",
        "n_targets": n,
        "windows": {
            "closing": [0.0, _t(S["closing_end"])],
            "trial_lift_and_decision": [_t(S["closing_end"]), _t(S["decision_end"])],
            "decline_accepted": [0.0, _t(S["decision_end"])],
            "carry": [_t(K.CARRY_START_S), _t(Te)],
            "set_down": [_t(Te), _t(Te + S["set_down_s"])],
            "retreat": [_t(Te + S["set_down_s"]), _t(Te + S["retreat_end_after_Te"])],
            "next_tray_s": _t(Te + S["next_tray_after_Te"]),
        },
        "declined": {"put_back_by_s": _t(S["decline_put_back_s"]), "hands_clear_by_s": _t(S["decline_clear_s"]),
                     "next_tray_after_s": _t(S["decline_next_s"])},
        "missed_capture": {"clear_by_s": _t(S["missed_clear_end"]), "next_tray_s": _t(S["missed_next"])},
        "targets": tg,
    }


def build_sheet(jobs):
    slots = []
    t = K.FREE_INTERVAL_S
    for i, (kind, targets) in enumerate(jobs):
        slots.append(slot_entry(i, kind, targets, t))
        t += K.slot_length_s(len(targets))
    return {
        "format_version": SHEET_FORMAT_VERSION,
        "cycle_s": K.CYCLE_S,
        "time_origin": TIME_ORIGIN,
        "frame": FRAME,
        "first_presentation_s": _t(K.FREE_INTERVAL_S),
        "move_s": K.MOVE_S,
        "hold_s": K.HOLD_S,
        "lifted_point_above_seated_m": K.LIFTED_POINT_M,
        "slots": slots,
    }


def canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii")


def digest(obj):
    return hashlib.sha256(canonical(obj)).hexdigest()


def _walk(obj, path=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield path + "." + str(k), k, None
            yield from _walk(v, path + "." + str(k))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _walk(v, f"{path}[{i}]")
    else:
        yield path, None, obj


def validate_sheet(sheet):
    def keys(d, want, where):
        if not isinstance(d, dict) or tuple(sorted(d)) != tuple(sorted(want)):
            got = sorted(d) if isinstance(d, dict) else type(d).__name__
            raise ValueError(f"{where}: keys {got} != {sorted(want)}")
    keys(sheet, TOP_KEYS, "sheet")
    for path, key, val in _walk(sheet):
        for s in ([key] if key is not None else []) + ([val] if isinstance(val, str) else []):
            low = str(s).lower()
            for w in FORBIDDEN_WORDS:
                if w in low:
                    raise ValueError(f"{path}: the word {w!r} in {s!r}")
    if sheet["format_version"] != SHEET_FORMAT_VERSION:
        raise ValueError("format_version")
    for i, sl in enumerate(sheet["slots"]):
        keys(sl, SLOT_KEYS, f"slots[{i}]")
        keys(sl["windows"], WINDOW_KEYS, f"slots[{i}].windows")
        for k, tg in enumerate(sl["targets"]):
            keys(tg, TARGET_KEYS, f"slots[{i}].targets[{k}]")
    try:
        ref = build_sheet([(sl["job"], [t["p"] for t in sl["targets"]]) for sl in sheet["slots"]])
    except (KeyError, TypeError) as ex:
        raise ValueError(f"the sheet cannot be rebuilt from its target lists: {ex}") from ex
    if canonical(ref) != canonical(sheet):
        raise ValueError("the sheet's times or fixed fields do not follow from its target lists and the contract")
    return True


def format_document(example):
    return {
        "what": "The supervisor's shift sheet of stage 05: per tray slot the windows of the slot, the target list with "
                "times and the destination rest. It states no mass, centre or label of any tray. Produced by "
                "duty/sheet.py build_sheet (maintainers' file; the published format is worded from it).",
        "format_version": SHEET_FORMAT_VERSION,
        "fields": SHEET_FIELDS,
        "top_keys": list(TOP_KEYS),
        "slot_keys": list(SLOT_KEYS),
        "window_keys": list(WINDOW_KEYS),
        "target_keys": list(TARGET_KEYS),
        "canonical_json": "sorted keys, separators (',', ':'), ASCII; digests are sha256 of these bytes",
        "example_note": "An example of the format only: drawn from a fixed development entropy "
                        "of its own, its job lists not checked by the generator; not the shift of any unit.",
        "example": example,
    }
