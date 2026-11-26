"""Types of the shared unit and the reader of the public parameter file. PUBLIC.

`UnitState` (frozen; what makes one built unit differ from the maker's published model), with its
parts `PlantState`, `CalState`, `HSState`, `ThermalState`, `BodyInertial`, `Mount`; `SessionState`;
`HostProfile`; `RegisterSet` (what a register dump holds). Each has `to_dict` / `from_dict`; `UnitState` and
`SessionState` have `dump(path)` and `load(path)` in canonical JSON (sorted keys, floats by repr(), no whitespace) and a
`digest` = sha256 of that canonical text without the digest field. Every type is immutable (tuples and read-only
mappings) and hashable by its content, so that a state cannot change under its digest.

This module imports nothing of the law, the registry, the truth channel, the world or the recorder, and does not read
the hidden parameter file. Module order everywhere: index 0..15 = left J1..J7, left J8 (gripper), right J1..J7,
right J8. "Arm joints" (14) are the modules with J1..J7, in the same order (left J1..J7, right J1..J7).

The model's quantity labels, used in the text of this package's public modules (the same lines are the legend
`quantity_labels` of params_public.json):
- P1: the Coulomb friction level of each arm joint (the joint's frictionloss), N m
- P2: the viscous damping of each arm joint (the joint's damping), N m s/rad
- P3: the reflected rotor inertia of each joint (the joint's armature), kg m^2
- P4: the mass, centre of mass and inertia of each link as built (the body's inertial)
- P5: the items mounted on an arm (leaf bodies mount_<side>_<k>)
- P6: the gripper's friction level (finger1's frictionloss), N m
- P7: the gripper's viscous damping (the finger joints' damping), N m s/rad
- C0: the zero stored in each module before any zero routine runs, rad
- C1: the stored zero delta in force in each module (the module angle is s (q - delta)), rad
- C2: the VMAX mapping register of the two DM4340 modules of each arm, rad/s
- C3: the delivered torque per commanded torque k of each module (the actuator's gain)
- C4: the constant torque offset b of each module, N m
- C5: the module's setting registers (OT_Value, OC_Value, TIMEOUT, CTRL_MODE, can_br)
- C6: the edition of the modules of each class (DM8009, DM4340, DM4310)
- C7: the current base Imax of each module, as the module reports it at power-up, A
- S1: a session's factor on the friction levels P1 and P6 of each module
- S2: a session's step of the torque offset b of each module, N m
- S3: a session's thermal state: the lab's ambient and each module's start temperatures, deg C
- ST1: the two mechanical stops of each arm joint as built (the joint's range), rad
- PL1: where each arm's base link is bolted to the pedestal (the pose of openarm_<side>_base_link)
- HS1: how the friction level of each arm joint varies along its travel (the hold-scatter table)
- H1: the host profile: the gains, checks and settings of the reference hosts
- TH1: the heat factor h of each module (how fast its winding heats per drive torque squared)
"""
import hashlib
import json
import math
import os
from dataclasses import dataclass, replace
from types import MappingProxyType

N_MODULES = 16
N_ARM_JOINTS = 14
HS_TABLE_ENTRIES = 256
GRIPPER_MODULES = (7, 15)
ARM_MODULES = tuple(i for i in range(N_MODULES) if i not in GRIPPER_MODULES)
SIDES = ("left", "right")
STATE_FORMAT = "openarm_unit/UnitState/1"
SESSION_FORMAT = "openarm_unit/SessionState/2"          # 2: start temperatures per module
PARAMS_PUBLIC_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "params_public.json")

_PUBLIC = None


class StateError(ValueError):
    """A state whose shape is not the one stated here, or a stored digest that does not match its content."""


# ------------------------------------------------------------------------------------------------- the public parameters
def public_params():
    """The parsed params_public.json (read once per process; callers must not mutate it)."""
    global _PUBLIC
    if _PUBLIC is None:
        with open(PARAMS_PUBLIC_PATH, "rb") as f:
            raw = f.read()
        _PUBLIC = (json.loads(raw.decode("ascii")), hashlib.sha256(raw).hexdigest())
    return _PUBLIC[0]


def public_params_sha256():
    public_params()
    return _PUBLIC[1]


def is_entry(node):
    """A leaf entry of a parameter file: a dict with 'value' and 'status'."""
    return isinstance(node, dict) and "value" in node and "status" in node


def public_entry(path):
    """The leaf entry at a dotted path of params_public.json (e.g. 'editions.DM4310_V1_1.T_lim_Nm')."""
    node = public_params()
    for part in path.split("."):
        if not isinstance(node, dict) or part not in node:
            raise KeyError(path)
        node = node[part]
    if not is_entry(node):
        raise KeyError(f"{path} is not a leaf entry")
    return node


def public_value(path):
    """The value of the leaf entry at a dotted path of params_public.json."""
    return public_entry(path)["value"]


# --------------------------------------------------------------------------------------------------------------- helpers
def canonical_json(obj):
    """Canonical text: sorted keys, floats by repr() (Python's json), ASCII, no whitespace, no NaN."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def _floats(x, n=None, what="value"):
    t = tuple(float(v) for v in x)
    if n is not None and len(t) != n:
        raise StateError(f"{what}: {n} numbers expected, {len(t)} given")
    return t


def _ints(x, n=None, what="value"):
    t = tuple(int(v) for v in x)
    if n is not None and len(t) != n:
        raise StateError(f"{what}: {n} integers expected, {len(t)} given")
    return t


def _pairs(x, n, what):
    t = tuple(_floats(p, 2, what) for p in x)
    if len(t) != n:
        raise StateError(f"{what}: {n} pairs expected, {len(t)} given")
    return t


def module_side(i):
    return SIDES[i // 8]


def module_joint(i):
    """1..8 (8 = gripper)."""
    return i % 8 + 1


# ------------------------------------------------------------------------------------------------------------ registers
def _register_value(rid, v):
    """A register value in its carried type (params_public.json registers.carried[rid].type): 'float' -> float,
    'uint32' -> int (an integral number in [0, 2^32)); None stays None. A RID that is not carried is refused."""
    carried = public_params()["registers"]["carried"]
    if str(rid) not in carried:
        raise StateError(f"register {rid} is not carried by the unit")
    if v is None:
        return None
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise StateError(f"register {rid}: a value is an int, a float or None")
    kind = carried[str(rid)]["type"]["value"]
    if kind == "float":
        v = float(v)
        if not math.isfinite(v):
            raise StateError(f"register {rid}: not a finite number")
        return v
    if kind == "uint32":
        if float(v) != int(v) or not 0 <= int(v) < (1 << 32):
            raise StateError(f"register {rid}: a uint32 is an integer in [0, 2^32)")
        return int(v)
    raise StateError(f"register {rid}: unknown type {kind!r}")


@dataclass(frozen=True)
class RegisterSet:
    """The registers of the sixteen modules: per module a read-only mapping RID (int) -> value, each value in
    its register's carried type (float or int), or None. 100 and 100.0 for a float register are the same state."""
    modules: tuple

    def __post_init__(self):
        mods = tuple(MappingProxyType({int(k): _register_value(int(k), v) for k, v in sorted(
            ((int(k), v) for k, v in m.items()), key=lambda kv: kv[0])}) for m in self.modules)
        if len(mods) != N_MODULES:
            raise StateError("a RegisterSet holds the registers of 16 modules")
        object.__setattr__(self, "modules", mods)

    def __hash__(self):
        return hash(canonical_json(self.to_dict()))

    def get(self, module, rid):
        """Value of register `rid` (an int, or a name of params_public.json registers.carried) of module `module`."""
        if isinstance(rid, str):
            rid = register_rid(rid)
        return self.modules[module][rid]

    def with_value(self, module, rid, value):
        if isinstance(rid, str):
            rid = register_rid(rid)
        mods = [dict(m) for m in self.modules]
        mods[module][rid] = value
        return RegisterSet(tuple(mods))

    def to_dict(self):
        return {"modules": [{str(k): v for k, v in sorted(m.items())} for m in self.modules]}

    @classmethod
    def from_dict(cls, d):
        return cls(tuple({int(k): v for k, v in m.items()} for m in d["modules"]))


def register_rid(name):
    """RID of a carried register by its maker name (params_public.json registers.carried), or by an alias the entry
    lists (register 18 is spelled 'Ls' here, 'LS' in the maker's table). Exact names first; then a
    case-insensitive match that must be unique."""
    carried = public_params()["registers"]["carried"]
    for rid, r in carried.items():
        if r["name"] == name or name in r.get("aliases", ()):
            return int(rid)
    hits = [int(rid) for rid, r in carried.items()
            if r["name"].lower() == name.lower() or name.lower() in (a.lower() for a in r.get("aliases", ()))]
    if len(hits) == 1:
        return hits[0]
    raise KeyError(name)


# ------------------------------------------------------------------------------------------------------------ the plant
@dataclass(frozen=True)
class BodyInertial:
    """The `<inertial pos quat mass diaginertia>` of one of the maker's bodies AS BUILT."""
    body: str
    mass: float
    pos: tuple
    quat: tuple
    diaginertia: tuple

    def __post_init__(self):
        object.__setattr__(self, "mass", float(self.mass))
        object.__setattr__(self, "pos", _floats(self.pos, 3, "pos"))
        object.__setattr__(self, "quat", _floats(self.quat, 4, "quat"))
        object.__setattr__(self, "diaginertia", _floats(self.diaginertia, 3, "diaginertia"))

    def to_dict(self):
        return {"body": self.body, "mass": self.mass, "pos": list(self.pos), "quat": list(self.quat),
                "diaginertia": list(self.diaginertia)}

    @classmethod
    def from_dict(cls, d):
        return cls(d["body"], d["mass"], d["pos"], d["quat"], d["diaginertia"])


@dataclass(frozen=True)
class Mount:
    """A mounted item: a leaf body `mount_<side>_<k>` of a carrying link, an inertial, at most one primitive geom of
    class collision. `geom_type` None = no geom."""
    name: str
    parent: str
    pos: tuple
    quat: tuple
    mass: float
    com: tuple
    diaginertia: tuple
    geom_type: object = None
    geom_size: tuple = ()
    item: str = ""

    def __post_init__(self):
        if not (self.name.startswith("mount_left_") or self.name.startswith("mount_right_")):
            raise StateError("a mount is named mount_<side>_<k>")
        object.__setattr__(self, "pos", _floats(self.pos, 3, "pos"))
        object.__setattr__(self, "quat", _floats(self.quat, 4, "quat"))
        object.__setattr__(self, "mass", float(self.mass))
        object.__setattr__(self, "com", _floats(self.com, 3, "com"))
        object.__setattr__(self, "diaginertia", _floats(self.diaginertia, 3, "diaginertia"))
        object.__setattr__(self, "geom_size", _floats(self.geom_size, None, "geom_size"))
        if self.geom_type not in (None, "sphere", "capsule", "ellipsoid", "cylinder", "box"):
            raise StateError("a mount's geom is one primitive")

    def to_dict(self):
        return {"name": self.name, "parent": self.parent, "pos": list(self.pos), "quat": list(self.quat),
                "mass": self.mass, "com": list(self.com), "diaginertia": list(self.diaginertia),
                "geom_type": self.geom_type, "geom_size": list(self.geom_size), "item": self.item}

    @classmethod
    def from_dict(cls, d):
        return cls(d["name"], d["parent"], d["pos"], d["quat"], d["mass"], d["com"], d["diaginertia"],
                   d.get("geom_type"), d.get("geom_size", ()), d.get("item", ""))


@dataclass(frozen=True)
class PlantState:
    """The plant quantities of a unit as built.

    frictionloss[16]: P1 for the arm modules, P6 (finger1) for the grippers, in module order; the unit-level level
        before the session factor S1 and the hold-scatter table HS1.
    damping[16]: P2 for the arm modules, P7 (finger damping) for the grippers.
    inertials: the maker's moving bodies as built, one BodyInertial each.
    mounts: what is mounted.
    stops[14]: the two mechanical stops of each arm joint as built (ST1), (lower, upper) in the model coordinate.
    base_poses[2]: (pos 3, quat 4) of openarm_left_base_link and openarm_right_base_link as built.
    """
    frictionloss: tuple
    damping: tuple
    inertials: tuple
    mounts: tuple
    stops: tuple
    base_poses: tuple

    def __post_init__(self):
        object.__setattr__(self, "frictionloss", _floats(self.frictionloss, N_MODULES, "frictionloss"))
        object.__setattr__(self, "damping", _floats(self.damping, N_MODULES, "damping"))
        object.__setattr__(self, "inertials", tuple(self.inertials))
        object.__setattr__(self, "mounts", tuple(self.mounts))
        object.__setattr__(self, "stops", _pairs(self.stops, N_ARM_JOINTS, "stops"))
        poses = tuple((_floats(p[0], 3, "base pos"), _floats(p[1], 4, "base quat")) for p in self.base_poses)
        if len(poses) != 2:
            raise StateError("two base poses")
        object.__setattr__(self, "base_poses", poses)
        names = [m.name for m in self.mounts]
        if len(set(names)) != len(names):
            raise StateError("mount names repeat")

    def to_dict(self):
        return {"frictionloss": list(self.frictionloss), "damping": list(self.damping),
                "inertials": [b.to_dict() for b in self.inertials], "mounts": [m.to_dict() for m in self.mounts],
                "stops": [list(p) for p in self.stops],
                "base_poses": [{"pos": list(p), "quat": list(q)} for p, q in self.base_poses]}

    @classmethod
    def from_dict(cls, d):
        return cls(d["frictionloss"], d["damping"], tuple(BodyInertial.from_dict(b) for b in d["inertials"]),
                   tuple(Mount.from_dict(m) for m in d["mounts"]), d["stops"],
                   tuple((p["pos"], p["quat"]) for p in d["base_poses"]))


ZERO_SOURCES = ("as_drawn", "routine", "residual", "nominal")


@dataclass(frozen=True)
class CalState:
    """Calibration-side quantities of a module (C0, C1, C3, C4, C5, C6 via registers).

    zero_before[16]: C0, the zero stored in each module before any routine (unit state of its own; all 16 modules).
    delta[16]: C1, the stored zero IN FORCE when the unit is built (theta = s (q - delta)). For zero_mode = routine it is
        the pre-routine zero (delta = zero_before, zero_source 'as_drawn'); after the routine it is the zeros the routine
        wrote (the 'as zeroed' state, made by `UnitState.with_zeros`, zero_source 'routine');
        'residual' (a unit zeroed and then checked on a fixture) and 'nominal' (0) are the other sources.
    sign[16]: s_i, +1 on every unit. k[16]: C3. b[16]: C4 (grippers 0). imax[16]: C7, as read at power-up.
    registers: the register table of every module.
    """
    zero_before: tuple
    delta: tuple
    zero_source: str
    sign: tuple
    k: tuple
    b: tuple
    imax: tuple
    registers: RegisterSet

    def __post_init__(self):
        for name in ("zero_before", "delta", "k", "b", "imax"):
            object.__setattr__(self, name, _floats(getattr(self, name), N_MODULES, name))
        object.__setattr__(self, "sign", _ints(self.sign, N_MODULES, "sign"))
        if any(s not in (1, -1) for s in self.sign):
            raise StateError("a sign is +1 or -1")
        if self.zero_source not in ZERO_SOURCES:
            raise StateError(f"zero_source is one of {ZERO_SOURCES}")
        if not isinstance(self.registers, RegisterSet):
            raise StateError("registers is a RegisterSet")

    def to_dict(self):
        return {"zero_before": list(self.zero_before), "delta": list(self.delta), "zero_source": self.zero_source,
                "sign": list(self.sign), "k": list(self.k), "b": list(self.b), "imax": list(self.imax),
                "registers": self.registers.to_dict()}

    @classmethod
    def from_dict(cls, d):
        return cls(d["zero_before"], d["delta"], d["zero_source"], d["sign"], d["k"], d["b"], d["imax"],
                   RegisterSet.from_dict(d["registers"]))


@dataclass(frozen=True)
class HSState:
    """The hold-scatter source HS1 of the fourteen arm joints: its parameters and the stored tables of
    (1 + rho_i) over one base period, 256 entries each (never re-derived). The table's
    abscissa is the joint's MECHANICAL angle q, never the module's angle (saving a zero does not move the pattern)."""
    r: float
    period: tuple
    phases: tuple
    tables: tuple

    def __post_init__(self):
        object.__setattr__(self, "r", float(self.r))
        object.__setattr__(self, "period", _floats(self.period, N_ARM_JOINTS, "period"))
        object.__setattr__(self, "phases", _pairs(self.phases, N_ARM_JOINTS, "phases"))
        tabs = tuple(_floats(t, HS_TABLE_ENTRIES, "table") for t in self.tables)
        if len(tabs) != N_ARM_JOINTS:
            raise StateError("fourteen tables")
        object.__setattr__(self, "tables", tabs)

    def to_dict(self):
        return {"r": self.r, "period": list(self.period), "phases": [list(p) for p in self.phases],
                "tables": [list(t) for t in self.tables]}

    @classmethod
    def from_dict(cls, d):
        return cls(d["r"], d["period"], d["phases"], d["tables"])


@dataclass(frozen=True)
class ThermalState:
    """TH1: the heat factor h of each module."""
    h: tuple

    def __post_init__(self):
        object.__setattr__(self, "h", _floats(self.h, N_MODULES, "h"))

    def to_dict(self):
        return {"h": list(self.h)}

    @classmethod
    def from_dict(cls, d):
        return cls(d["h"])


@dataclass(frozen=True)
class UnitState:
    """One drawn unit. `editions` maps a class (DM8009, DM4340, DM4310) to its edition key of params_public.json
    (C6, per unit and class), held as a read-only mapping. The digest is computed from the content; `load` refuses a
    file whose stored digest differs; the hash is the digest's."""
    unit_id: str
    law_version: str
    editions: dict
    plant: PlantState
    cal: CalState
    hs: HSState
    thermal: ThermalState

    def __post_init__(self):
        ed = dict(self.editions)
        if set(ed) != {"DM8009", "DM4340", "DM4310"}:
            raise StateError("editions names the three classes")
        allowed = public_value("class_editions")
        for cls, e in ed.items():
            if e not in allowed[cls]:
                raise StateError(f"{e} is not an edition of {cls}")
        object.__setattr__(self, "editions", MappingProxyType(dict(sorted(ed.items()))))

    def __hash__(self):
        return hash(self.digest)

    def edition_of(self, module):
        """Edition key of module `module` (its class's edition)."""
        cls = public_value("conventions.modules")[module]["class"]
        return self.editions[cls]

    def body(self):
        return {"format": STATE_FORMAT, "unit_id": self.unit_id, "law_version": self.law_version,
                "editions": dict(self.editions), "plant": self.plant.to_dict(), "cal": self.cal.to_dict(),
                "hs": self.hs.to_dict(), "thermal": self.thermal.to_dict()}

    @property
    def digest(self):
        return hashlib.sha256(canonical_json(self.body()).encode("ascii")).hexdigest()

    def to_dict(self):
        d = self.body()
        d["digest"] = self.digest
        return d

    def canonical(self):
        return canonical_json(self.to_dict())

    def dump(self, path):
        path = os.fspath(path)
        text = self.canonical()
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="ascii") as f:
            f.write(text)
        os.replace(tmp, path)
        return self.digest

    @classmethod
    def from_dict(cls, d, check_digest=True):
        if d.get("format") != STATE_FORMAT:
            raise StateError("not a UnitState")
        s = cls(d["unit_id"], d["law_version"], d["editions"], PlantState.from_dict(d["plant"]),
                CalState.from_dict(d["cal"]), HSState.from_dict(d["hs"]), ThermalState.from_dict(d["thermal"]))
        if check_digest and d.get("digest") != s.digest:
            raise StateError("the stored digest does not match the content")
        return s

    @classmethod
    def load(cls, path):
        with open(path, "r", encoding="ascii") as f:
            return cls.from_dict(json.loads(f.read()))

    def with_zeros(self, delta, source="routine"):
        """The same unit with the zeros in force replaced (the 'as zeroed' state; a new digest)."""
        return replace(self, cal=replace(self.cal, delta=_floats(delta, N_MODULES, "delta"), zero_source=source))


# ---------------------------------------------------------------------------------------------------------- the session
@dataclass(frozen=True)
class SessionState:
    """One session of a unit (S1 to S3). friction_factor[16]: S1 for P1 and P6 (module order). b_step[16]: S2 (the
    grippers 0). ambient_C: the lab's ambient. start_temperatures_C[16]: the start value (T_w0, T_h0) of the winding and
    the housing node of EACH module, in deg C (start temperatures are session state per module; the two values of a pair
    may differ, e.g. in a state set by a test or a reference host)."""
    session_id: str
    friction_factor: tuple
    b_step: tuple
    ambient_C: float
    start_temperatures_C: tuple

    def __post_init__(self):
        object.__setattr__(self, "friction_factor", _floats(self.friction_factor, N_MODULES, "friction_factor"))
        object.__setattr__(self, "b_step", _floats(self.b_step, N_MODULES, "b_step"))
        object.__setattr__(self, "ambient_C", float(self.ambient_C))
        object.__setattr__(self, "start_temperatures_C", _pairs(self.start_temperatures_C, N_MODULES,
                                                                "start_temperatures_C"))
        for v in (self.ambient_C,) + tuple(x for p in self.start_temperatures_C for x in p):
            if not math.isfinite(v):
                raise StateError("temperatures are finite numbers")

    def start_winding_C(self):
        """T_w0 of the sixteen modules (module order)."""
        return tuple(p[0] for p in self.start_temperatures_C)

    def start_housing_C(self):
        """T_h0 of the sixteen modules (module order)."""
        return tuple(p[1] for p in self.start_temperatures_C)

    @classmethod
    def from_offsets(cls, session_id, friction_factor, b_step, ambient_C, offsets_K):
        """A session whose module i starts at ambient + offsets_K[i] on both nodes."""
        amb = float(ambient_C)
        offs = _floats(offsets_K, N_MODULES, "offsets_K")
        return cls(session_id, friction_factor, b_step, amb, tuple((amb + w, amb + w) for w in offs))

    @classmethod
    def nominal(cls, session_id="nominal"):
        """The nominal session: factors 1, steps 0, ambient 25 deg C, every module at ambient (S1 to S3 nominals)."""
        return cls.from_offsets(session_id, [1.0] * N_MODULES, [0.0] * N_MODULES, 25.0, [0.0] * N_MODULES)

    def body(self):
        return {"format": SESSION_FORMAT, "session_id": self.session_id, "friction_factor": list(self.friction_factor),
                "b_step": list(self.b_step), "ambient_C": self.ambient_C,
                "start_temperatures_C": [list(p) for p in self.start_temperatures_C]}

    @property
    def digest(self):
        return hashlib.sha256(canonical_json(self.body()).encode("ascii")).hexdigest()

    def __hash__(self):
        return hash(self.digest)

    def to_dict(self):
        d = self.body()
        d["digest"] = self.digest
        return d

    def dump(self, path):
        path = os.fspath(path)
        text = canonical_json(self.to_dict())
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="ascii") as f:
            f.write(text)
        os.replace(tmp, path)
        return self.digest

    @classmethod
    def from_dict(cls, d, check_digest=True):
        if d.get("format") != SESSION_FORMAT:
            raise StateError("not a SessionState of format " + SESSION_FORMAT)
        s = cls(d["session_id"], d["friction_factor"], d["b_step"], d["ambient_C"], d["start_temperatures_C"])
        if check_digest and d.get("digest") != s.digest:
            raise StateError("the stored digest does not match the content")
        return s

    @classmethod
    def load(cls, path):
        with open(path, "r", encoding="ascii") as f:
            return cls.from_dict(json.loads(f.read()))


# ------------------------------------------------------------------------------------------------------ the host profile
@dataclass(frozen=True)
class HostProfile:
    """H1: the gains, checks and settings of the reference hosts (not the unit). Arrays are per joint J1..J8 of an arm
    (the same for both arms; kp / kd for J1..J7)."""
    name: str
    kp: tuple
    kd: tuple
    v_des: float
    t_ff: float
    jump_limits: tuple
    velocity_limits: tuple
    velocity_dt_cap_s: float
    start_move_s: float
    start_move_hz: float
    host_vmax_dm4340: float
    joint_offsets: tuple
    gripper_mode: str
    gripper_fp_speed: float
    gripper_fp_limit: float
    gripper_mit_kp: float
    gripper_mit_kd: float
    recorder_profile: object = None

    def __post_init__(self):
        object.__setattr__(self, "kp", _floats(self.kp, 7, "kp"))
        object.__setattr__(self, "kd", _floats(self.kd, 7, "kd"))
        object.__setattr__(self, "jump_limits", _floats(self.jump_limits, 8, "jump_limits"))
        object.__setattr__(self, "velocity_limits", _floats(self.velocity_limits, 8, "velocity_limits"))
        object.__setattr__(self, "joint_offsets", _floats(self.joint_offsets, 8, "joint_offsets"))
        for name in ("v_des", "t_ff", "velocity_dt_cap_s", "start_move_s", "start_move_hz", "host_vmax_dm4340",
                     "gripper_fp_speed", "gripper_fp_limit", "gripper_mit_kp", "gripper_mit_kd"):
            object.__setattr__(self, name, float(getattr(self, name)))
        if self.gripper_mode not in ("force_position", "mit"):
            raise StateError("gripper_mode is 'force_position' or 'mit'")

    @classmethod
    def default(cls, gain_set="A", gripper_mode="force_position", host_vmax_dm4340=10.0, recorder_profile=None):
        """The maker's driver 0.5.1 with gain set A or B, from params_public.json host_profiles."""
        hp = public_params()["host_profiles"]
        g = hp["gain_sets"]["value"][gain_set]
        gm = hp["gripper_modes"]["value"]
        if host_vmax_dm4340 not in hp["host_mapping_vmax_dm4340"]["value"]:
            raise StateError("the host table of the DM4340 has VMAX 10 or 8")
        return cls(f"driver-{gain_set}", g["kp"], g["kd"], hp["v_des"]["value"], hp["t_ff"]["value"],
                   hp["jump_limits_rad"]["value"], hp["velocity_limits_rad_s"]["value"],
                   hp["velocity_limiter_dt_cap_s"]["value"], hp["start_move"]["value"]["duration_s"],
                   hp["start_move"]["value"]["rate_Hz"], host_vmax_dm4340, hp["joint_offsets"]["value"], gripper_mode,
                   gm["force_position"]["speed_rad_s"], gm["force_position"]["current_limit"], gm["mit"]["kp"],
                   gm["mit"]["kd"], recorder_profile)

    def to_dict(self):
        d = {}
        for k in self.__dataclass_fields__:
            v = getattr(self, k)
            d[k] = list(v) if isinstance(v, tuple) else v
        return d

    @classmethod
    def from_dict(cls, d):
        return cls(**d)
