"""The joint modules' register tables at run time. PUBLIC.

Imports only the public reader of `params_public.json` (`state`); nothing of the law, the registry, the truth channel,
the world, the recorder or the thermal model.

What is here:

- the register numbers (RID) and the maker's value layout on the 0x7FF parameter frames: a value is four bytes, little
  endian, read as uint32 for RIDs 7-10, 13-16, 35-36 and as float32 for every other RID (the maker's openarm_can 20d66f0,
  `dm_motor_control.cpp:117-133` and `:259-262` `is_in_ranges`; `dm_motor_control.hpp:134-149` `pack_write_param_data`
  copies the value's bytes as they lie in memory, little endian on the maker's hosts);
- `catalogue_registers`: one module's table with the edition's catalogue figures (params_public.json
  `editions` and `registers.carried`): what a nominal unit and a test model carry. A unit's drive reports its own
  identified Flux, Rs and Ls; they are not made here: a caller may pass them;
- `RegisterTable`: the RAM copy of the sixteen (or N) modules' registers that the bus layer reads and writes;
- `write_policy`: which register writes a task profile admits: the profiles 'stage04' and 'stage05' admit each
  gripper's CTRL_MODE while that gripper is disabled, 'none' admits nothing; TIMEOUT is never writable; a CTRL_MODE
  write is refused while enabled and a write of a mode the unit does not model (2 or 3) is refused.
"""
import math
import struct

from .state import N_MODULES, RegisterSet, public_params, register_rid

# RIDs used by the bus layer by name (params_public.json registers.carried; maker dm_motor_constants.hpp:93-139)
UV_VALUE, KT_VALUE, OT_VALUE, OC_VALUE = 0, 1, 2, 3
MST_ID, ESC_ID, TIMEOUT, CTRL_MODE = 7, 8, 9, 10
DAMP, INERTIA, HW_VER, SW_VER, SN = 11, 12, 13, 14, 15
NPP, RS, LS, FLUX, GR = 16, 17, 18, 19, 20
PMAX, VMAX, TMAX = 21, 22, 23
OV_VALUE, GREF, CAN_BR = 29, 30, 35
P_M, XOUT = 80, 81

# CTRL_MODE codes (maker dm_motor_constants.hpp:39; params_public.json registers.ctrl_mode_codes, checked below)
MODE_MIT = 1
MODE_POS_VEL = 2
MODE_VEL = 3
MODE_POS_FORCE = 4
IMPLEMENTED_MODES = (MODE_MIT, MODE_POS_FORCE)      # position-velocity and velocity modes are not implemented

# Registers whose change alters the drive's torque unit or its current limit (KT and T_lim from the module's
# own registers), the routing (7, 8) or the frame mapping (21-23).
TORQUE_RIDS = frozenset((OC_VALUE, NPP, FLUX, GR, GREF))
ROUTING_RIDS = frozenset((MST_ID, ESC_ID))
MAPPING_RIDS = frozenset((PMAX, VMAX, TMAX))
LIVE_RIDS = frozenset((P_M, XOUT))


class RegisterError(ValueError):
    """A register, a value or a policy that the register tables or the runtime contract do not allow."""


def _check_constants(p=None):
    """The mode codes above against params_public.json registers.ctrl_mode_codes (the maker's, printed there with their
    source); RegisterError on any difference. p: a parsed public parameter file (default: the package's)."""
    p = public_params() if p is None else p
    got = {str(k): int(v) for k, v in p["registers"]["ctrl_mode_codes"]["value"].items()}
    want = {"MIT": MODE_MIT, "POS_VEL": MODE_POS_VEL, "VEL": MODE_VEL, "POS_FORCE": MODE_POS_FORCE}
    if got != want:
        raise RegisterError(f"the CTRL_MODE codes differ from params_public.json: {got} != {want}")


_check_constants()


def carried():
    """params_public.json registers.carried (a mapping 'RID' -> entry)."""
    return public_params()["registers"]["carried"]


def is_uint_rid(rid):
    """The maker's layout: uint32 for RIDs 7-10, 13-16, 35-36, float32 otherwise (dm_motor_control.cpp:259-262)."""
    return (7 <= rid <= 10) or (13 <= rid <= 16) or (35 <= rid <= 36)


def pack_value(rid, value):
    """The four value bytes of a parameter frame for `value` of register `rid` (little endian; uint32 or float32 by
    the maker's layout). None packs as four zero bytes."""
    if value is None:
        return bytes(4)
    if is_uint_rid(rid):
        v = int(value)
        if not 0 <= v < (1 << 32) or float(value) != v:
            raise RegisterError(f"register {rid}: {value!r} is not a uint32")
        return struct.pack("<I", v)
    return struct.pack("<f", float(value))


def unpack_value(rid, b4):
    """The value four bytes carry for register `rid` (an int for the uint32 RIDs, a float otherwise)."""
    b4 = bytes(b4)
    if len(b4) != 4:
        raise RegisterError("a register value is four bytes")
    if is_uint_rid(rid):
        return struct.unpack("<I", b4)[0]
    return struct.unpack("<f", b4)[0]


def access(rid):
    """'RW' or 'RO' of a carried register; None for a RID the unit does not carry."""
    e = carried().get(str(int(rid)))
    return None if e is None else e["access"]["value"]


# ------------------------------------------------------------------------------------------------ catalogue tables
def catalogue_registers(edition, command_id, reply_id, ctrl_mode=MODE_MIT, timeout=0, vmax=None, identified=None,
                        hw_ver=None, sw_ver=None, sn=None):
    """One module's register table with its edition's catalogue figures, as a dict RID -> value.

    edition: an edition key of params_public.json (e.g. 'DM4310_V1_1'); command_id / reply_id: ESC_ID / MST_ID;
    ctrl_mode: 1 (MIT) or 4 (force-position); timeout: TIMEOUT in counts of 50 us; vmax: the VMAX register when it
    differs from the edition's default; identified: optional {'Flux': v, 'Rs': v, 'LS': v} of the drive's own
    identified values (a unit's own, as its dump prints them); None = the catalogue figures. hw_ver, sw_ver, SN:
    None (not printed by the one real dump; a task may fill them)."""
    p = public_params()
    if edition not in p["editions"]:
        raise RegisterError(f"unknown edition {edition!r}")
    ed = p["editions"][edition]
    v = lambda k: ed[k]["value"]
    c = carried()
    m = v("mapping_default")
    regs = {
        UV_VALUE: float(c["0"]["default"]["value"]),
        KT_VALUE: float(c["1"]["default"]["value"]),
        OT_VALUE: float(c["2"]["default"]["value"]),
        OC_VALUE: float(v("OC_Value")),
        MST_ID: int(reply_id),
        ESC_ID: int(command_id),
        TIMEOUT: int(timeout),
        CTRL_MODE: int(ctrl_mode),
        DAMP: float(v("Damp_register")),
        INERTIA: float(v("rotor_inertia_kg_m2")),
        HW_VER: hw_ver,
        SW_VER: sw_ver,
        SN: sn,
        NPP: int(v("pole_pairs")),
        RS: float(v("phase_resistance_ohm")),
        LS: float(v("phase_inductance_H")),
        FLUX: float(v("flux_Wb")),
        GR: float(v("gear_ratio")),
        PMAX: float(m["PMAX"]),
        VMAX: float(m["VMAX"] if vmax is None else vmax),
        TMAX: float(m["TMAX"]),
        OV_VALUE: None,
        GREF: float(v("GREF")),
        CAN_BR: int(c["35"]["default"]["value"]),
        P_M: None,
        XOUT: None,
    }
    for name, val in (identified or {}).items():
        regs[register_rid(name)] = float(val)
    if int(ctrl_mode) not in IMPLEMENTED_MODES:
        raise RegisterError("CTRL_MODE is 1 (MIT) or 4 (force-position) on this unit")
    return regs


def nominal_register_set(gripper_mode=MODE_POS_FORCE, timeout=0, editions=None, vmax_dm4340=10.0):
    """The sixteen modules' catalogue registers in module order, with the nominal editions unless
    `editions` maps a class to an edition, the arm modules in MIT mode and the grippers in `gripper_mode`, TIMEOUT
    `timeout` on every module, and the DM4340's VMAX register at `vmax_dm4340` (nominal 10, the one
    real DM4340P dump's value; the first edition's catalogue default is 8). A
    RegisterSet (state.py)."""
    p = public_params()
    eds = dict(p["class_nominal_edition"]["value"])
    eds.update(editions or {})
    mods = []
    for m in p["conventions"]["modules"]["value"]:
        mode = gripper_mode if m["joint"] == 8 else MODE_MIT
        vmax = float(vmax_dm4340) if m["class"] == "DM4340" else None
        mods.append(catalogue_registers(eds[m["class"]], m["command_id"], m["reply_id"], mode, timeout, vmax=vmax))
    return RegisterSet(tuple(mods))


# --------------------------------------------------------------------------------------------- the RAM register table
class RegisterTable:
    """The registers of N modules in RAM (writes act in RAM; save changes nothing within an episode). Values
    are kept in their carried type (float, int or None). A RegisterSet in, a RegisterSet out."""

    def __init__(self, modules):
        mods = modules.modules if isinstance(modules, RegisterSet) else tuple(modules)
        self._m = [dict((int(k), v) for k, v in m.items()) for m in mods]

    @property
    def n(self):
        return len(self._m)

    def get(self, module, rid, default=None):
        return self._m[module].get(int(rid), default)

    def set(self, module, rid, value):
        self._m[module][int(rid)] = value

    def has(self, module, rid):
        return int(rid) in self._m[module]

    def module(self, module):
        """A copy of one module's table (RID -> value)."""
        return dict(self._m[module])

    def to_register_set(self):
        if self.n != N_MODULES:
            raise RegisterError("a RegisterSet holds sixteen modules; this table holds %d" % self.n)
        return RegisterSet(tuple(dict(m) for m in self._m))

    def to_list(self):
        """Canonical form for a snapshot: per module a list of [rid, value] sorted by RID."""
        return [[[k, m[k]] for k in sorted(m)] for m in self._m]

    @classmethod
    def from_list(cls, lst):
        return cls([{int(k): v for k, v in m} for m in lst])


# --------------------------------------------------------------------------------------------------- write policy
STAGE_WHITELISTS = ("stage04", "stage05")


class WritePolicy:
    """Which register writes are admitted: a set of (module, RID) pairs. Over every policy: TIMEOUT is never
    writable; a read-only register is never writable; CTRL_MODE only while the module is disabled and only to
    an implemented mode (1 or 4): a write of 2 or 3 is refused, so that a module never enters a mode the unit does not
    model."""

    def __init__(self, pairs=()):
        self.pairs = frozenset((int(m), int(r)) for m, r in pairs)

    def admits(self, module, rid, value, enabled):
        """(admitted, reason) for a write of `value` to register `rid` of `module` (enabled: the module's enable)."""
        rid = int(rid)
        if rid == TIMEOUT:
            return False, "timeout_never_writable"
        if access(rid) != "RW":
            return False, "read_only_or_not_carried"
        if (int(module), rid) not in self.pairs:
            return False, "not_whitelisted"
        if rid == CTRL_MODE:
            if enabled:
                return False, "ctrl_mode_while_enabled"
            if int(value) not in IMPLEMENTED_MODES:
                return False, "mode_not_implemented"
        if isinstance(value, float) and not math.isfinite(value):
            return False, "not_finite"
        return True, "whitelisted"

    def to_list(self):
        return sorted([m, r] for m, r in self.pairs)


def write_policy(profile, gripper_modules=None):
    """The default whitelist of a task profile (params_public.json frames.register_write_whitelist): 'stage04'
    and 'stage05' admit the CTRL_MODE register of the two grippers (modules 7 and 15 of the pair, or
    `gripper_modules`), while disabled: the one register write the maker's own host software makes. 'none' admits
    nothing."""
    wl = public_params()["frames"]["register_write_whitelist"]["value"]
    if profile == "none":
        return WritePolicy(())
    if profile not in wl:
        raise RegisterError(f"unknown profile {profile!r}")
    pairs = []
    for item in wl[profile]:
        if item == "gripper CTRL_MODE while disabled":
            for g in (gripper_modules if gripper_modules is not None else (7, 15)):
                pairs.append((g, CTRL_MODE))
        else:
            raise RegisterError(f"whitelist item {item!r} is not understood")
    return WritePolicy(pairs)
