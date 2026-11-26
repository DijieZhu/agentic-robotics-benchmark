"""The frame codec of the controller interface (standard library only).

  - the fields of the two frame kinds a controller sends, their ranges and bit widths, and the quantiser the modules
    apply (clip to the range, then truncation to the field's code): mit_codes, mit_effective, posforce_codes,
    posforce_effective give what a module receives for a frame;
  - the eight data bytes of each frame kind (mit_bytes, posforce_bytes) and the values of a module's state reply
    (state_fields);
  - the check of a step's reply (parse_step_reply): a reply that is an object but not a valid reply counts as a late
    cycle (ReplyMalformed); a reply that is not an object ends the controller (ReplyNotObject);
  - the feedback object of one cycle (feedback, module_feedback).

The ranges of a module are its own registers PMAX, VMAX and TMAX, as the unit's delivery record prints them.
A number field of a frame is a JSON number that is finite and not a boolean; a NaN, an infinity, a boolean, a string,
an integer beyond the float range, or a force-position p beyond the float32 range makes the reply malformed.
"""
import math
import struct

N_MODULES = 16
MIT_KEYS = ("p", "v", "kp", "kd", "t")
POSFORCE_KEYS = ("p", "v_lim", "i_lim")
MIT_BITS = {"p": 16, "v": 12, "kp": 12, "kd": 12, "t": 12}
KP_RANGE = (0.0, 500.0)                                # N m/rad
KD_RANGE = (0.0, 5.0)                                  # N m s/rad
V_LIM_RANGE = (0.0, 100.0)                             # rad/s, steps of 0.01
I_LIM_RANGE = (0.0, 1.0)                               # per unit of the module's printed current base, steps of 1e-4
V_LIM_SCALE = 100.0                                    # code = trunc(v_lim x 100), at most 10000
I_LIM_SCALE = 10000.0                                  # code = trunc(i_lim x 10000), at most 10000
FP_RAW_MAX = 10000
FLOAT32_MAX = 3.4028234663852886e38
REPLY_BITS = {"q": 16, "dq": 12, "tau": 12}
FEEDBACK_KEYS = ("cycle", "site", "modules")
SITE_KEYS = ("tray", "slot_cycle")
MODULE_FEEDBACK_KEYS = ("q", "dq", "tau", "t_mos", "t_rotor", "status", "age")
STEP_REPLY_KEYS = ("frames", "decline")


class ReplyMalformed(ValueError):
    """A step reply that is an object but not a valid reply: it counts as a late cycle."""


class ReplyNotObject(ValueError):
    """A step reply that is not an object: it ends the controller for that unit."""


# ---------------------------------------------------------------------------------------------- the quantiser
def clip(x, lo, hi):
    """max(lo, min(x, hi)) in the order of the real frame's clip (a NaN maps to lo; never reached for a checked
    field)."""
    m = hi if hi < x else x
    return m if lo < m else lo


def to_code(x, lo, hi, bits):
    """The real frame's quantiser: clip to [lo, hi], then truncation of the normalised value times 2^bits - 1."""
    x = clip(float(x), float(lo), float(hi))
    return int(math.floor((x - lo) / (hi - lo) * float((1 << bits) - 1)))


def from_code(u, lo, hi, bits):
    """The value of code u: u / (2^bits - 1) x (hi - lo) + lo."""
    return float(u) / float((1 << bits) - 1) * (hi - lo) + lo


def step_of(lo, hi, bits):
    """One code step of a field."""
    return (hi - lo) / float((1 << bits) - 1)


RANGE_REGISTERS = ("PMAX", "VMAX", "TMAX")


def register_value(dump, name):
    """The value of register `name` ("PMAX", "VMAX", "TMAX", ...) in one module's register dump as the delivery record
    prints it (the row whose parameter ends in "(name)"; its last column), as a float. KeyError if there is none."""
    tag = "(" + name + ")"
    for line in dump.splitlines():
        cols = line.split()
        if cols and line[:30].rstrip().endswith(tag):
            return float(cols[-1])
    raise KeyError(name)


def record_ranges(unit_record):
    """[(PMAX, VMAX, TMAX)] of each module of a delivery record, in its module order, read from each module's
    register_dump."""
    return [tuple(register_value(m["register_dump"], r) for r in RANGE_REGISTERS) for m in unit_record["modules"]]


def mit_ranges(pmax, vmax, tmax):
    """{field: (lo, hi)} of an MIT frame for a module with registers PMAX, VMAX, TMAX."""
    return {"p": (-pmax, pmax), "v": (-vmax, vmax), "kp": KP_RANGE, "kd": KD_RANGE, "t": (-tmax, tmax)}


def mit_codes(frame, limits):
    """(code_p, code_v, code_kp, code_kd, code_t) of an MIT frame (a dict with the five MIT_KEYS) for a module whose
    registers are limits = (PMAX, VMAX, TMAX)."""
    r = mit_ranges(*limits)
    return tuple(to_code(frame[k], r[k][0], r[k][1], MIT_BITS[k]) for k in MIT_KEYS)


def mit_effective(frame, limits):
    """What the module acts on for this MIT frame: each field clipped and truncated to its code, as a value."""
    r = mit_ranges(*limits)
    return {k: from_code(c, r[k][0], r[k][1], MIT_BITS[k]) for k, c in zip(MIT_KEYS, mit_codes(frame, limits))}


def mit_bytes(codes):
    """The eight data bytes of an MIT frame from its five codes (the maker's layout)."""
    q_u, dq_u, kp_u, kd_u, t_u = codes
    return bytes([(q_u >> 8) & 0xFF, q_u & 0xFF, dq_u >> 4, ((dq_u & 0xF) << 4) | ((kp_u >> 8) & 0xF), kp_u & 0xFF,
                  kd_u >> 4, ((kd_u & 0xF) << 4) | ((t_u >> 8) & 0xF), t_u & 0xFF])


def posforce_codes(frame):
    """(p as float32, code_v_lim, code_i_lim) of a force-position frame (a dict with the three POSFORCE_KEYS)."""
    p = struct.unpack("<f", struct.pack("<f", float(frame["p"])))[0]
    v_u = int(math.floor(clip(float(frame["v_lim"]) * V_LIM_SCALE, 0.0, float(FP_RAW_MAX))))
    i_u = int(math.floor(clip(clip(float(frame["i_lim"]), 0.0, 1.0) * I_LIM_SCALE, 0.0, float(FP_RAW_MAX))))
    return p, v_u, i_u


def posforce_effective(frame):
    """What the module acts on for this force-position frame: p as float32, the two limits truncated to their codes."""
    p, v_u, i_u = posforce_codes(frame)
    return {"p": p, "v_lim": v_u / V_LIM_SCALE, "i_lim": i_u / I_LIM_SCALE}


def posforce_bytes(codes):
    """The eight data bytes of a force-position frame: p float32 little endian, then the two codes as uint16 little
    endian."""
    p, v_u, i_u = codes
    return struct.pack("<f", p) + bytes([v_u & 0xFF, (v_u >> 8) & 0xFF, i_u & 0xFF, (i_u >> 8) & 0xFF])


def state_fields(data, limits):
    """A module's state reply (eight bytes) as values: status (the high nibble of the first byte, an integer 0..15),
    q, dq, tau (decoded with the module's registers limits = (PMAX, VMAX, TMAX)), t_mos, t_rotor (integer deg C).
    None for a frame shorter than eight bytes."""
    d = bytes(data)
    if len(d) < 8:
        return None
    pm, vm, tm = limits
    q_u = (d[1] << 8) | d[2]
    dq_u = (d[3] << 4) | (d[4] >> 4)
    t_u = ((d[4] & 0xF) << 8) | d[5]
    return {"status": d[0] >> 4, "q": from_code(q_u, -pm, pm, 16), "dq": from_code(dq_u, -vm, vm, 12),
            "tau": from_code(t_u, -tm, tm, 12), "t_mos": int(d[6]), "t_rotor": int(d[7])}


# ---------------------------------------------------------------------------------------------- a step's reply
def _number(x, where):
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        raise ReplyMalformed(f"{where}: not a number")
    try:
        f = float(x)
    except (OverflowError, ValueError):
        raise ReplyMalformed(f"{where}: beyond the float range") from None
    if not math.isfinite(f):
        raise ReplyMalformed(f"{where}: not finite")
    return f


def frame_kind(obj):
    """'mit' | 'posforce' | None (a module's place without a frame) for one entry of a reply's frames; raises
    ReplyMalformed for anything else (an object with other keys, a list, a number ...)."""
    if obj is None:
        return None
    if not isinstance(obj, dict):
        raise ReplyMalformed("a frame is an object or null")
    keys = set(obj)
    if keys == set(MIT_KEYS):
        return "mit"
    if keys == set(POSFORCE_KEYS):
        return "posforce"
    raise ReplyMalformed(f"a frame has the keys {sorted(MIT_KEYS)} or {sorted(POSFORCE_KEYS)}, not {sorted(keys)}")


def check_frame(obj, where="frame"):
    """(kind, frame with float fields) of one entry, or (None, None) for null; ReplyMalformed otherwise."""
    kind = frame_kind(obj)
    if kind is None:
        return None, None
    out = {k: _number(obj[k], f"{where}.{k}") for k in (MIT_KEYS if kind == "mit" else POSFORCE_KEYS)}
    if kind == "posforce" and abs(out["p"]) > FLOAT32_MAX:
        raise ReplyMalformed(f"{where}.p: beyond the float32 range of the frame")
    return kind, out


def parse_step_reply(rep, n_modules=N_MODULES):
    """The checked content of a step's reply: (frames, decline), frames a list of n_modules entries, each (kind,
    fields) or (None, None). Raises ReplyNotObject when rep is not an object (dict), ReplyMalformed when it is an
    object but not a valid reply: 'frames' missing or not a list of n_modules entries, an entry neither null nor a
    frame of one of the two kinds with finite numbers, 'decline' present and not a boolean. Keys other than 'frames'
    and 'decline' are ignored."""
    if not isinstance(rep, dict):
        raise ReplyNotObject(f"the reply is a {type(rep).__name__}, not an object")
    fr = rep.get("frames")
    if not isinstance(fr, list) or len(fr) != n_modules:
        raise ReplyMalformed(f"'frames' is a list of {n_modules} entries")
    dec = rep.get("decline", False)
    if not isinstance(dec, bool):
        raise ReplyMalformed("'decline' is true or false")
    return [check_frame(f, f"frames[{i}]") for i, f in enumerate(fr)], dec


# ---------------------------------------------------------------------------------------------- the feedback
def module_feedback(state, age):
    """One module's entry of the feedback from its last state reply's values (state_fields) and the reply's age in
    cycles."""
    return {"q": state["q"], "dq": state["dq"], "tau": state["tau"], "t_mos": int(state["t_mos"]),
            "t_rotor": int(state["t_rotor"]), "status": int(state["status"]), "age": int(age)}


def feedback(cycle, tray, slot_cycle, modules):
    """The feedback object of one cycle: cycle k, the supervisor's signal (the presented tray's index or None, the
    cycles since its presentation or None) and the sixteen module entries."""
    return {"cycle": int(cycle), "site": {"tray": tray, "slot_cycle": slot_cycle}, "modules": list(modules)}
