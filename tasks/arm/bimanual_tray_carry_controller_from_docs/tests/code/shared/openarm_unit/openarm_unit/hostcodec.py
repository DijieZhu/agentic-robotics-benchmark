"""Host-side codec of the joint modules' CAN frames. PUBLIC.

A port of the maker's host encoder and decoder, written from its text (the maker's openarm_can at 20d66f0,
`src/openarm/damiao_motor/dm_motor_control.cpp` and `include/openarm/damiao_motor/dm_motor_constants.hpp`; the lines are
cited at each function). Imports only `frames` and the public reader of params_public.json; nothing of the law, the
registry, the truth channel, the world, the recorder or the thermal model.

Quantisation: encoding TRUNCATES, as the maker's `double_to_uint` does (cpp:226-231):
    u = trunc(((clip(x, lo, hi) - lo) / (hi - lo)) x (2^bits - 1))
and decoding is the maker's `uint_to_double` (cpp:239-243):
    x = (u / (2^bits - 1)) x (hi - lo) + lo.
The clip is the maker's `limit_min_max` (cpp:222-224), max(lo, min(x, hi)): a NaN maps to lo, as there.

The host's mapping table (PMAX, VMAX, TMAX per class) is the maker's `MOTOR_LIMIT_PARAMS` (hpp:149-164): DM8009 12.5 /
45 / 54, DM4340 12.5 / 10 / 28, DM4310 12.5 / 30 / 10. Before 2026-07-17 the maker's table held VMAX 8 for the DM4340
(params_public.json host_profiles.host_mapping_vmax_dm4340): `host_table(vmax_dm4340=8.0)`. The numbers are read
from params_public.json (editions' mapping_default and the host table's two DM4340 values), not typed here.

The bus layer (`bus.py`) encodes its replies with `to_uint` and decodes commands with `from_uint` of this module, with
the MODULE's own registers in place of the host table.
"""
import math
import struct

from .frames import Frame
from .state import public_params

PARAM_ID = 0x7FF                 # maker cpp:65-85 (params_public.json frames.param_can_id)
OP_READ, OP_WRITE, OP_SAVE, OP_REFRESH = 0x33, 0x55, 0xAA, 0xCC     # params_public.json frames.param_ops
CMD_ENABLE, CMD_DISABLE, CMD_SAVE_ZERO, CMD_CLEAR = 0xFC, 0xFD, 0xFE, 0xFB  # maker cpp:25-35, 217-219
POSVEL_OFFSET, VEL_OFFSET, POSFORCE_OFFSET = 0x100, 0x200, 0x300   # maker cpp:42-63
KP_RANGE = (0.0, 500.0)          # maker cpp:138 (params_public.json frames.kp_range)
KD_RANGE = (0.0, 5.0)            # maker cpp:139 (params_public.json frames.kd_range)
FP_SPEED_SCALE = 100.0           # maker cpp:184-185, 191 (params_public.json frames.force_position_speed_scale)
FP_CURRENT_SCALE = 10000.0       # maker cpp:186-188, 193 (params_public.json frames.force_position_current_scale)
FP_RAW_MAX = 10000.0             # maker cpp:192, 194: "values above 10000 clamp to 10000" (frames.force_position_raw_max)
CLASSES = ("DM8009", "DM4340", "DM4310")


class CodecError(ValueError):
    """A field, a table or a frame that the maker's protocol does not allow."""


def _check_constants(p=None):
    """The constants above against params_public.json (they are the maker's, printed there with their source); the
    file must carry every one of them (a missing entry is a CodecError too). p: a parsed public parameter file
    (default: the package's)."""
    f = (public_params() if p is None else p)["frames"]
    try:
        got = (f["param_can_id"]["value"], f["param_ops"]["value"], f["special_tail"]["value"],
               f["force_position_id_offset"]["value"], tuple(f["kp_range"]["value"]), tuple(f["kd_range"]["value"]),
               f["force_position_speed_scale"]["value"], f["force_position_current_scale"]["value"],
               f["force_position_raw_max"]["value"])
    except KeyError as e:
        raise CodecError(f"params_public.json frames lacks {e}") from None
    want = (PARAM_ID, {"read": OP_READ, "refresh": OP_REFRESH, "save": OP_SAVE, "write": OP_WRITE},
            {"clear_error": CMD_CLEAR, "disable": CMD_DISABLE, "enable": CMD_ENABLE, "save_zero": CMD_SAVE_ZERO},
            POSFORCE_OFFSET, KP_RANGE, KD_RANGE, FP_SPEED_SCALE, FP_CURRENT_SCALE, FP_RAW_MAX)
    if got != want:
        raise CodecError(f"hostcodec's protocol constants differ from params_public.json: {got} != {want}")


_check_constants()


# ------------------------------------------------------------------------------------------------- the quantiser
def clip(x, lo, hi):
    """The maker's limit_min_max (cpp:222-224): std::max(lo, std::min(x, hi)); NaN -> lo."""
    m = hi if hi < x else x          # std::min(x, hi) = (hi < x) ? hi : x
    return m if lo < m else lo       # std::max(lo, m) = (lo < m) ? m : lo


def to_uint(x, lo, hi, bits):
    """The maker's double_to_uint (cpp:226-231): truncation of the normalised value times 2^bits - 1."""
    x = clip(float(x), float(lo), float(hi))
    span = hi - lo
    norm = (x - lo) / span
    return int(math.floor(norm * float((1 << bits) - 1)))


def from_uint(u, lo, hi, bits):
    """The maker's uint_to_double (cpp:239-243)."""
    span = hi - lo
    norm = float(u) / float((1 << bits) - 1)
    return norm * span + lo


def step_of(x_max, bits):
    """One code step of a field over [-x_max, x_max] (the resolution)."""
    return 2.0 * x_max / float((1 << bits) - 1)


# ----------------------------------------------------------------------------------------------- the host table
def host_table(vmax_dm4340=10.0):
    """The maker's host mapping table {class: (PMAX, VMAX, TMAX)} (hpp:149-164), with the DM4340's VMAX of the table
    edition in force: 10 since 2026-07-17, 8 before."""
    p = public_params()
    allowed = p["host_profiles"]["host_mapping_vmax_dm4340"]["value"]
    if float(vmax_dm4340) not in [float(v) for v in allowed]:
        raise CodecError(f"the maker's host table has VMAX {allowed} for the DM4340, not {vmax_dm4340}")
    ed = p["editions"]
    out = {}
    for cls, key in (("DM8009", "DM8009P"), ("DM4340", "DM4340_V1_1"), ("DM4310", "DM4310_V1_1")):
        m = ed[key]["mapping_default"]["value"]
        out[cls] = (float(m["PMAX"]), float(m["VMAX"]), float(m["TMAX"]))
    pm, _, tm = out["DM4340"]
    out["DM4340"] = (pm, float(vmax_dm4340), tm)
    return out


def limits_of(limits):
    """(PMAX, VMAX, TMAX) from a tuple, or from a class name through the current host table."""
    if isinstance(limits, str):
        return host_table()[limits]
    pm, vm, tm = limits
    return float(pm), float(vm), float(tm)


# ----------------------------------------------------------------------------------------------- command frames
def mit_data(limits, q, dq, kp, kd, tau):
    """The eight data bytes of an MIT command (cpp:136-156): q 16 bit, dq 12, kp 12 over [0, 500], kd 12 over [0, 5],
    tau 12, each truncated."""
    pm, vm, tm = limits_of(limits)
    kp_u = to_uint(kp, KP_RANGE[0], KP_RANGE[1], 12)
    kd_u = to_uint(kd, KD_RANGE[0], KD_RANGE[1], 12)
    q_u = to_uint(q, -pm, pm, 16)
    dq_u = to_uint(dq, -vm, vm, 12)
    t_u = to_uint(tau, -tm, tm, 12)
    return bytes([(q_u >> 8) & 0xFF, q_u & 0xFF, dq_u >> 4, ((dq_u & 0xF) << 4) | ((kp_u >> 8) & 0xF), kp_u & 0xFF,
                  kd_u >> 4, ((kd_u & 0xF) << 4) | ((t_u >> 8) & 0xF), t_u & 0xFF])


def pack_mit(bus, can_id, limits, q, dq, kp, kd, tau):
    """An MIT command frame on the module's command id (cpp:37-40)."""
    return Frame(int(bus), int(can_id), mit_data(limits, q, dq, kp, kd, tau))


def unpack_mit(data, limits):
    """(q, dq, kp, kd, tau) of an MIT command's eight bytes decoded with `limits` (the inverse of mit_data; what a
    module with those registers reads)."""
    d = bytes(data)
    if len(d) != 8:
        raise CodecError("an MIT frame has eight data bytes")
    pm, vm, tm = limits_of(limits)
    q_u = (d[0] << 8) | d[1]
    dq_u = (d[2] << 4) | (d[3] >> 4)
    kp_u = ((d[3] & 0xF) << 8) | d[4]
    kd_u = (d[5] << 4) | (d[6] >> 4)
    t_u = ((d[6] & 0xF) << 8) | d[7]
    return (from_uint(q_u, -pm, pm, 16), from_uint(dq_u, -vm, vm, 12), from_uint(kp_u, KP_RANGE[0], KP_RANGE[1], 12),
            from_uint(kd_u, KD_RANGE[0], KD_RANGE[1], 12), from_uint(t_u, -tm, tm, 12))


def posforce_data(q, speed, current_pu):
    """The eight data bytes of a force-position command (cpp:179-204): p as float32 little endian; speed limit uint16
    little endian = trunc(clip(speed x 100, 0, 10000)); current limit uint16 little endian = trunc(clip(clip(i, 0, 1) x
    10000, 0, 10000)), i per unit of the module's current base Imax."""
    pb = struct.pack("<f", float(q))
    v_u = int(math.floor(clip(float(speed) * FP_SPEED_SCALE, 0.0, FP_RAW_MAX)))
    i_u = int(math.floor(clip(clip(float(current_pu), 0.0, 1.0) * FP_CURRENT_SCALE, 0.0, FP_RAW_MAX)))
    return pb + bytes([v_u & 0xFF, (v_u >> 8) & 0xFF, i_u & 0xFF, (i_u >> 8) & 0xFF])


def pack_posforce(bus, command_id, q, speed, current_pu):
    """A force-position command frame on the module's command id + 0x300 (cpp:58-63)."""
    return Frame(int(bus), int(command_id) + POSFORCE_OFFSET, posforce_data(q, speed, current_pu))


def unpack_posforce(data):
    """(p, speed limit rad/s, current limit per unit) of a force-position command's eight bytes; the two limits'
    raw values above 10000 read as 10000 (cpp:184-188)."""
    d = bytes(data)
    if len(d) != 8:
        raise CodecError("a force-position frame has eight data bytes")
    p = struct.unpack("<f", d[0:4])[0]
    v_u = min(d[4] | (d[5] << 8), int(FP_RAW_MAX))
    i_u = min(d[6] | (d[7] << 8), int(FP_RAW_MAX))
    return p, v_u / FP_SPEED_SCALE, i_u / FP_CURRENT_SCALE


def special_data(cmd):
    """FF x7 + the command byte (cpp:217-219)."""
    return bytes([0xFF] * 7 + [int(cmd)])


def enable(bus, command_id):
    return Frame(int(bus), int(command_id), special_data(CMD_ENABLE))


def disable(bus, command_id):
    return Frame(int(bus), int(command_id), special_data(CMD_DISABLE))


def save_zero(bus, command_id):
    return Frame(int(bus), int(command_id), special_data(CMD_SAVE_ZERO))


def clear(bus, command_id):
    """Clear error (FB; the vendor's protocol: the maker's host has no helper for it)."""
    return Frame(int(bus), int(command_id), special_data(CMD_CLEAR))


def _param(bus, command_id, op, rid=0, value_bytes=bytes(4)):
    cid = int(command_id)
    return Frame(int(bus), PARAM_ID, bytes([cid & 0xFF, (cid >> 8) & 0xFF, op, int(rid) & 0xFF]) + bytes(value_bytes))


def refresh(bus, command_id):
    """State refresh on 0x7FF (cpp:74-85)."""
    return _param(bus, command_id, OP_REFRESH)


def read_param(bus, command_id, rid):
    """Register read on 0x7FF (cpp:65-67, 206-215)."""
    return _param(bus, command_id, OP_READ, rid)


def write_param(bus, command_id, rid, value):
    """Register write on 0x7FF (hpp:134-149), the value in the maker's layout (uint32 or float32, little endian)."""
    from .registers import pack_value
    return _param(bus, command_id, OP_WRITE, rid, pack_value(int(rid), value))


def save_param(bus, command_id):
    """Register save on 0x7FF (0xAA; the vendor's protocol as the maker's simulator emulates it, test/cli/
    damiao_simulator.py:20-22)."""
    return _param(bus, command_id, OP_SAVE)


# ---------------------------------------------------------------------------------------------------- replies
def state_data(status, module_id, limits, q, dq, tau, t_mos, t_rotor):
    """The eight bytes of a state reply (the layout decoded by cpp:88-115): D0 = (status << 4) | (id & 0xF); q 16 bit;
    dq 12 bit; torque 12 bit; T_MOS and T_Rotor as integer deg C (clipped to one byte). Truncating, as every encoder
    here."""
    pm, vm, tm = limits_of(limits)
    q_u = to_uint(q, -pm, pm, 16)
    dq_u = to_uint(dq, -vm, vm, 12)
    t_u = to_uint(tau, -tm, tm, 12)
    tmos = min(255, max(0, int(t_mos)))
    trot = min(255, max(0, int(t_rotor)))
    return bytes([((int(status) & 0xF) << 4) | (int(module_id) & 0xF), (q_u >> 8) & 0xFF, q_u & 0xFF, dq_u >> 4,
                  ((dq_u & 0xF) << 4) | ((t_u >> 8) & 0xF), t_u & 0xFF, tmos, trot])


def unpack_state(data, limits):
    """The maker's parse_motor_state_data (cpp:88-115): a dict status, id (the low nibble), q, dq, tau, t_mos, t_rotor,
    with the host table `limits` (a class name or (PMAX, VMAX, TMAX)); None for a frame shorter than eight bytes."""
    d = bytes(data)
    if len(d) < 8:
        return None
    pm, vm, tm = limits_of(limits)
    q_u = (d[1] << 8) | d[2]
    dq_u = (d[3] << 4) | (d[4] >> 4)
    t_u = ((d[4] & 0xF) << 8) | d[5]
    return {"status": d[0] >> 4, "id": d[0] & 0xF, "q": from_uint(q_u, -pm, pm, 16),
            "dq": from_uint(dq_u, -vm, vm, 12), "tau": from_uint(t_u, -tm, tm, 12), "t_mos": d[6], "t_rotor": d[7],
            "codes": (q_u, dq_u, t_u)}


def unpack_param(data):
    """The maker's parse_motor_param_data (cpp:117-133): (rid, value) for a read or write reply (0x33 / 0x55), the
    value as uint32 or float32 by the maker's layout; None for any other frame. A save reply (0xAA) gives ('save',
    data[3])."""
    d = bytes(data)
    if len(d) < 8:
        return None
    if d[2] in (OP_READ, OP_WRITE):
        from .registers import unpack_value
        return d[3], unpack_value(d[3], d[4:8])
    if d[2] == OP_SAVE:
        return "save", d[3]
    return None
