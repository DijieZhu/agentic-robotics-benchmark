"""`OpenArmUnit`: one simulated unit of the OpenArm pair, spoken to only through its joint modules' frames, with a truth
channel for graders. PUBLIC.

PUBLIC: this module ships with the public runner of a submitted model, which builds on `OpenArmUnit.from_parts`. It
imports numpy, MuJoCo and the public modules `base`, `bus`, `hostcodec`, `registers`, `frames`, `state`, `truth`,
`hooks_api`; nothing of the law of drawing, the registry, the world, the recorder, the thermal model or the friction
profile's generator, and never the hidden parameter file. What a drawn unit needs of the hidden side enters as data or
as objects its caller passes:

- the friction profile: fourteen stored tables (a period and 256 values each) by which the friction level of each arm
  joint varies along its travel; looked up at the joint's MECHANICAL angle q (the model coordinate, not the angle a
  module reports) by linear interpolation at every step;
- heating, the coil protection and the reply noise: through `law`, an object with the methods
      law.heat(editions, h)  -> the bus layer's heat plug-in for the sixteen modules (bus.py, "THE PROTOCOLS")
      law.noise(key)         -> the bus layer's reply-noise plug-in keyed by the unit's noise key
      law.digest()           -> (optional) a hex string naming the law and its code: one of a snapshot's identities
  `build` takes it as the keyword `law`. With `law=None` the unit has no heating, no protection that heat trips and no
  reply noise: its replies carry the truncated true values and both temperature bytes the fixed public value. A
  `noise_key` without a `law` is refused.

THE UNIT'S STEP, physics step n of an exchange:
    1. bus.pre_step(): protections, then force range, ctrl and inertia entry of the sixteen carriers, the constant
       offset b into qfrc_applied at the driven dofs; the friction level of each of the 14 arm joints from its table
       at its mechanical angle q (dof_frictionloss = base level x factor(q)); the world's forces (added) and its
       switches due at step n (HookSet.pre_step)
    2. mj_step
    3. bus.post_step(): T_del, the thermal update (heat plug-in), the force-position integral, counters
    4. the plant's check (below), then event detection: torque clamps, contacts by class, stops; the world's events
       (HookSet.post_step); the per-step trace record of the fields the trace profile lists (the unit's own and the
       hooks' "<hook>.<field>" entries)
Replies of an exchange are sampled at reception, before the new commands act (bus.receive), one exchange old at the
host: the maker's driver reads each module's state reply to the command it sent. Replies come in the order of the
frames that caused them; a host matches them to modules by their bus and reply id, never by their position.

A STEP THE PLANT CANNOT INTEGRATE. After every physics step the unit reads MuJoCo's warning counters of a bad
acceleration, velocity and position (mjWARN_BADQACC, mjWARN_BADQVEL, mjWARN_BADQPOS) and of a full constraint or
contact buffer (mjWARN_CNSTRFULL, mjWARN_CONTACTFULL: the step then solved a truncated problem). MuJoCo answers the
first three by resetting the simulation to the model's reference state and running on; the state after such a step,
or after a truncated one, is no longer a state of the plant. When one of the five counters has risen since the last
step, the unit records the event `plant_invalid` (its step; `warning`, the first of the five names in the order
above that rose; `warnings`, every one that rose; `info`, MuJoCo's last info of that warning: for a bad
acceleration or velocity the index of the offending degree of freedom, for a bad position the index into qpos),
marks itself invalid and raises UnitInvalid (a UnitError; its `step`, `kind` = the event's `warning`, `kinds` and
`info`) from that exchange before any reply of it is returned. From then
on every exchange, reset, set_state, truth, truth_trace, tapped, snapshot, restore, state_digest and motion_digest of
the unit raises UnitInvalid with the same step, kind and info (nothing of the invalid state is served, and neither a
set state nor a restored snapshot revives it: a new unit is built); `events()` still drains, so that a caller can
read the event; `invalid` is None or that step, kind, kinds and info. A run in which the counters stay zero is not changed in any bit by the check: it only reads them.

THE TRUTH (`truth(fields=None)`, copies, sampled at the call: at the end of the last exchange). Fields by family:
    always:      step, exchange
    bus:         theta[16], T_ref[16], T_del[16], delivered[16] (what each joint receives: k x clamped command + b),
                 k[16], b[16], status[16], enabled[16], mode[16], T_w[16], T_h[16], steps_since_control[16],
                 command_mit[16 x 5], command_fp[16 x 3], delta[16], t_lim[16], Z[16]
    state:       q[18] / qvel[18] (the eighteen hinges, model coordinates, base.HINGES order), friction_level[18]
                 (dof_frictionloss in force), stop_margin[14] (min(q - lo, hi - q) of the 14 arm joints against their
                 as-built stops, rad), clamped[16] / clamped_peak[16] (the torque-clamp state of the last step),
                 friction_profile (the 14 stored tables: (period, 256 values) each, or None)
    kinematics:  tool_pos[2 x 3], tool_quat[2 x 4] (a unit: the tool point 0.135 m below the origin of
                 openarm_<side>_ee_base_link along its z axis, the maker's grasp offset, with that body's orientation;
                 a run built with two tool sites: the sites' frames), tool_rel_pos[3] / tool_rel_quat[4] (the right
                 tool in the left tool's frame)
    contacts:    {class pair 'a|b': normal_N, tangential_N, points} (the finger pair of each gripper excluded)
    stops:       stop_torque[18 x 2] (the torque on the lower and the upper stop of every hinge at the last solved step)
    world:       the hooks' truth
With `fields` (an iterable of field names) only the families those fields need are computed and only those fields are
returned; an unknown name is refused. The full call is a sampling call (a check, a job's end), not a per-step one; the
per-step trace (`truth_trace()`) takes the fields of `truth_profile`: names of PER_STEP_FIELDS and the per-step fields
the hooks offer, written "<hook name>.<field>"; each record carries an entry under that same name.
steps_since_control is each module's time-out counter: while a module is enabled, the physics steps since its last
control frame or, if later, its enable; while it is not enabled the counter holds its value (it counts only steps run
enabled), and an enable from not enabled sets it to zero.
What the truth reads from MuJoCo's derived arrays (delivered torques, stop torques, contact forces, the world's part)
is stored in a snapshot: right after `restore()` the truth equals the original's at the same point, until the next step.
So is the torque on each module's stops that a save zero's event reports: a save zero received in the first exchange
after a restore reports the original's value.
The truth call reads body poses at the present state (one mj_kinematics), a snapshot does so for the world's truth and
a restore recomputes every derived array (mj_forward): a world hook therefore never relies on a derived array that it
did not compute itself in the same call (hooks_api.py, "What a hook may read, and when").

THE SNAPSHOT: bytes = SNAPSHOT_MAGIC + canonical JSON {header, body}. Header: format, the identities (kind, unit, session
and world digests, code digest, the law's digest, noise, law, contacts, a numerics override, the configuration digest)
and the exchange and step counts; `restore()` refuses a snapshot whose identities differ. The configuration digest is
the sha256 of the canonical JSON of the unit's configuration: the bus layer's write policy, the contact class of every
geom, the per-step trace profile and the tool sites (a snapshot is refused by a unit configured otherwise). Body: the
physical state (below), the buffers a caller drains (pending events of the unit and of the bus layer, latched taps,
pending trace records), the edge detectors' previous sets, and the stored derived values of the truth and of the bus
layer's stop torques. Floats are written by repr(): a restore is exact.
`state_digest()` = sha256 of the canonical JSON of the PHYSICAL state only, what decides the future of the plant and
of every reply: mj_getState(mjSTATE_INTEGRATION) (warm start included), the model fields changed at run time
(dof_frictionloss; the bus layer's gain, bias, force range and inertia entries; the world's), the bus layer's state
without its event list, its frame and drop counters and the noise counters of its tap stream (these are observation:
they stay in the snapshot), the world's state, the exchange and step counts. Two runs of identical physics give the
same digest whatever their identities, whether or not their events were drained, whether or not a recorder took taps
and whether or not a frame was dropped. The edge detectors of events (contacts, stops, torque clamps) are seeded from
the state whenever a state is set (`set_state`, the pose of `reset`, the unit's construction), so that the events
that follow depend on the state alone.
`motion_digest()` = sha256 of the bytes of qpos and qvel of every dof (float64, little endian) and of the step count
(8 bytes, big endian, signed): the comparison of runs whose bus states differ by construction (a unit and a run of its
own written description: the second has no heating and public registers).

Arithmetic: on the per-step path float64 element-wise numpy (+, -, x, /, floor, min, max, abs, comparisons, indexing);
no BLAS, no reduction of floats, no transcendental function, no wall clock.
"""
import copy
import hashlib
import json
import math
import numbers
import os
from collections.abc import Mapping

import mujoco
import numpy as np

from . import base
from . import bus as busmod
from . import hostcodec as hc
from . import registers as rg
from .frames import Frame
from .hooks_api import HookError, HookSet
from .state import (GRIPPER_MODULES, ARM_MODULES, N_MODULES, N_ARM_JOINTS, HS_TABLE_ENTRIES, HostProfile, SessionState,
                    UnitState, canonical_json, public_params)
from .truth import ContactClasses, Event, StopReader, Truth, frame_pose, relative_pose, site_pose

SNAPSHOT_FORMAT = "openarm_unit/Snapshot/2"
SNAPSHOT_MAGIC = b"OPENARM-UNIT-SNAPSHOT/2\n"
DIGEST_FORMAT = "openarm_unit/PhysicalState/1"
CONFIG_FORMAT = "openarm_unit/Configuration/1"
_INTEG = int(mujoco.mjtState.mjSTATE_INTEGRATION)
PER_STEP_FIELDS = ("q", "qvel", "theta", "T_ref", "T_del", "delivered", "friction_level", "stop_margin",
                   "steps_since_control", "T_w", "T_h", "status", "enabled", "energy", "armature")
TRUTH_FAMILIES = {
    "bus": ("theta", "T_ref", "T_del", "delivered", "k", "b", "status", "enabled", "mode", "T_w", "T_h",
            "steps_since_control", "command_mit", "command_fp", "delta", "t_lim", "Z"),
    "state": ("q", "qvel", "friction_level", "stop_margin", "clamped", "clamped_peak", "friction_profile"),
    "kinematics": ("tool_pos", "tool_quat", "tool_rel_pos", "tool_rel_quat"),
    "contacts": ("contacts",),
    "stops": ("stop_torque",),
    "world": ("world",),
}
TRUTH_FIELDS = ("step", "exchange") + tuple(f for fam in TRUTH_FAMILIES.values() for f in fam)
_KIN_FIELDS = frozenset(TRUTH_FAMILIES["kinematics"])
_GETTER_FIELDS = TRUTH_FAMILIES["bus"] + TRUTH_FAMILIES["state"]
CODE_FILES = ("__init__.py", "base.py", "bus.py", "envelope.py", "frames.py", "hooks_api.py", "hostcodec.py",
              "registers.py", "state.py", "truth.py", "unit.py", "params_public.json")
MODE_NAMES = {"mit": rg.MODE_MIT, "posforce": rg.MODE_POS_FORCE}
OVERRIDE_KEYS = ("stop_compliance", "timestep", "armature", "energy")
# the warning counters the plant's check reads after every physics step, in the order in which a kind is named
PLANT_WARNINGS = ("mjWARN_BADQACC", "mjWARN_BADQVEL", "mjWARN_BADQPOS", "mjWARN_CNSTRFULL", "mjWARN_CONTACTFULL")
_PLANT_WARNING_INDEX = tuple((name, int(getattr(mujoco.mjtWarning, name))) for name in PLANT_WARNINGS)
_CODE_DIGEST = None


class UnitError(ValueError):
    """An argument or a state the unit does not accept (a snapshot of another unit, a refused override ...)."""


class UnitInvalid(UnitError):
    """The plant met a step it cannot integrate (module text, "A STEP THE PLANT CANNOT INTEGRATE"): `step`, the physics
    step at which a warning counter rose; `kind`, the first warning that rose (a name of PLANT_WARNINGS); `kinds`,
    every one that rose; `info`, MuJoCo's last info of that warning."""

    def __init__(self, step, kind, kinds=None, info=None):
        self.step = int(step)
        self.kind = str(kind)
        self.kinds = tuple(kinds) if kinds else (self.kind,)
        self.info = None if info is None else int(info)
        super().__init__(f"the plant cannot integrate physics step {self.step}: MuJoCo's warning {self.kind} rose "
                         f"(info {self.info}); the state is no longer a state of the plant and the unit serves "
                         f"nothing more (build a new unit)")


def code_digest():
    """sha256 over the files of the package that a unit runs (CODE_FILES, in that order: name, length, bytes)."""
    global _CODE_DIGEST
    if _CODE_DIGEST is None:
        h = hashlib.sha256()
        for name in CODE_FILES:
            with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), name), "rb") as f:
                b = f.read()
            h.update(name.encode("ascii") + b"\0" + str(len(b)).encode("ascii") + b"\0" + b)
        _CODE_DIGEST = h.hexdigest()
    return _CODE_DIGEST


def law_digest_of(law):
    """The digest a law provider names itself by (its `digest()`, a string), or None (no law, or no such method)."""
    if law is None:
        return None
    fn = getattr(law, "digest", None)
    if not callable(fn):
        return None
    v = fn()
    if not isinstance(v, str) or not v:
        raise UnitError("a law provider's digest() returns a non-empty string")
    return v


# ------------------------------------------------------------------------------------------- the friction profile
class FrictionProfile:
    """The per-step look-up of the fourteen stored friction-profile tables at the joints' MECHANICAL angles q.

    Value j of a table sits at q = (j / 256 + m) lambda; + - x / floor and indexing only:
        x = q / lambda;  f = x - floor(x);  p = f x 256;  j = floor(p);  t = p - j;  j0 = j mod 256;  j1 = (j0 + 1) mod 256
        factor = T[j0] + t (T[j1] - T[j0]);   level = base x factor"""

    def __init__(self, periods, tables):
        t = np.array([[float(v) for v in tab] for tab in tables], dtype=np.float64)
        if t.shape != (N_ARM_JOINTS, HS_TABLE_ENTRIES):
            raise UnitError(f"friction-profile tables of shape {(N_ARM_JOINTS, HS_TABLE_ENTRIES)} expected")
        self.tables = t
        self.periods = np.array([float(x) for x in periods], dtype=np.float64)
        if self.periods.shape != (N_ARM_JOINTS,) or not np.all(self.periods > 0.0):
            raise UnitError("fourteen positive periods")
        self._rows = np.arange(N_ARM_JOINTS)
        self._n = float(HS_TABLE_ENTRIES)
        self.flat = bool(np.all(t == 1.0))           # every factor 1: the level is the base level everywhere
        self._handout = tuple((float(lam), tuple(float(v) for v in row)) for lam, row in zip(self.periods, self.tables))

    def factor(self, q):
        x = np.asarray(q, dtype=np.float64) / self.periods
        f = x - np.floor(x)
        p = f * self._n
        j = np.floor(p)
        t = p - j
        j0 = j.astype(np.int64) % HS_TABLE_ENTRIES     # f may round to 1.0 for a tiny negative x: entry 0
        j1 = (j0 + 1) % HS_TABLE_ENTRIES
        t0 = self.tables[self._rows, j0]
        t1 = self.tables[self._rows, j1]
        return t0 + t * (t1 - t0)

    def level(self, q, base_level):
        return np.asarray(base_level, dtype=np.float64) * self.factor(q)

    def handout(self):
        """The fourteen (period, 256 values) pairs (what the truth exposes; immutable)."""
        return self._handout


# ------------------------------------------------------------------------------------------- UnitState -> plant edits
def plant_edits(state, session):
    """The plant edit mapping of base.unit_spec for a drawn unit in a named session.

    Per module i (module order = base.ACTUATORS): frictionloss = the unit's level x the session's factor; damping; range
    = the as-built stops (arm joints). The coupled finger_joint2 of each gripper gets frictionloss 0 and damping 0: the
    gripper's friction and damping are carried whole by the driven finger_joint1 (the maker's file gives both fingers
    a damping of 0.6; neither is kept). Actuators: k, and b = the unit's offset + the session's step. Inertials, base
    poses, mounts."""
    if not isinstance(state, UnitState) or not isinstance(session, SessionState):
        raise UnitError("a UnitState and a SessionState")
    p, c = state.plant, state.cal
    joints, actuators = {}, {}
    arm_pos = {m: a for a, m in enumerate(ARM_MODULES)}
    for i, (act, joint) in enumerate(base.ACTUATORS):
        e = {"frictionloss": p.frictionloss[i] * session.friction_factor[i], "damping": p.damping[i]}
        if i in arm_pos:
            lo, hi = p.stops[arm_pos[i]]
            e["range"] = [lo, hi]
        joints[joint] = e
        actuators[act] = {"k": c.k[i], "b": c.b[i] + session.b_step[i]}
    for side in base.SIDES:
        joints[f"openarm_{side}_finger_joint2"] = {"frictionloss": 0.0, "damping": 0.0}
    inertials = {b.body: {"mass": b.mass, "pos": list(b.pos), "quat": list(b.quat), "diaginertia": list(b.diaginertia)}
                 for b in p.inertials}
    bases = {side: {"pos": list(pos), "quat": list(quat)} for side, (pos, quat) in zip(base.SIDES, p.base_poses)}
    mounts = []
    for m in p.mounts:
        g = None
        if m.geom_type is not None:
            g = {"type": m.geom_type, "size": list(m.geom_size)}
        mounts.append({"name": m.name, "parent": m.parent, "pos": list(m.pos), "quat": list(m.quat),
                       "inertial": {"mass": m.mass, "pos": list(m.com), "quat": [1.0, 0.0, 0.0, 0.0],
                                    "diaginertia": list(m.diaginertia)}, "geom": g})
    return {"joints": joints, "inertials": inertials, "base": bases, "actuators": actuators, "mounts": mounts}


def _override(spec_or_model, override, stage):
    """Apply the test-only numerics override (stop compliance, a smaller time step, an inertia entry, MuJoCo's energy)
    to a spec ('spec') or a model."""
    o = dict(override)
    bad = set(o) - set(OVERRIDE_KEYS)
    if bad:
        raise UnitError(f"unknown override keys {sorted(bad)}; allowed {OVERRIDE_KEYS}")
    if stage == "spec":
        if "timestep" in o:
            spec_or_model.option.timestep = float(o["timestep"])
        if o.get("energy"):
            spec_or_model.option.enableflags |= int(mujoco.mjtEnableBit.mjENBL_ENERGY)
        return
    m = spec_or_model
    sc = o.get("stop_compliance")
    if sc is not None:
        dm = public_params()["numerics"]["stop_constraint_sensitivity_dmin_dmax"]["value"]   # [halved, doubled]
        if sc == "halved":
            v = float(dm[0])
        elif sc == "doubled":
            v = float(dm[1])
        elif sc == "nominal":
            v = None
        else:
            raise UnitError("stop_compliance is 'halved', 'doubled' or 'nominal'")
        if v is not None:
            for name in base.HINGES:
                j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
                m.jnt_solimp[j, 0] = v
                m.jnt_solimp[j, 1] = v
    for i, val in (o.get("armature") or {}).items():
        j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, base.ACTUATORS[int(i)][1])
        m.dof_armature[m.jnt_dofadr[j]] = float(val)


# ------------------------------------------------------------------------------------------------------- the unit
class OpenArmUnit:
    """One unit (module text). Construct with `build` (a drawn unit) or `from_parts` (the public runner of a model)."""

    # ------------------------------------------------------------------------------------------- construction
    @classmethod
    def build(cls, state, session, world=None, noise_key=None, contacts=True, *, law=None, policy=None,
              contact_classes=None, truth_profile=(), registry=None, _numerics_override=None):
        """The unit of `state` in `session`. world: a HookSet (default: the empty world). noise_key: 32 bytes or None
        (reply noise off). law: the hidden plug-in provider (module text) or None. policy: a registers.WritePolicy or
        a stage name ('stage04', 'stage05', 'none'; default none). contact_classes: {body or geom name: class}.
        truth_profile: per-step trace entries (PER_STEP_FIELDS and the hooks' "<hook>.<field>"; others refused).
        registry: the registry record of a registered unit (an opaque mapping); with it the test-only
        `_numerics_override` is refused. _numerics_override: test-only numerics (OVERRIDE_KEYS: stop_compliance
        'halved' | 'doubled' (dmin = dmax of the public numerics), timestep (refused with a law: the thermal step is
        1 ms), armature {module: value}, energy True (MuJoCo's energy))."""
        if not isinstance(state, UnitState) or not isinstance(session, SessionState):
            raise UnitError("build takes a UnitState and a SessionState")
        world = HookSet() if world is None else world
        if not isinstance(world, HookSet):
            raise UnitError("world is a HookSet")
        if _numerics_override is not None:
            if registry is not None:
                raise UnitError("the numerics override is test-only: refused when a registry is attached")
            if "timestep" in _numerics_override and law is not None:
                raise UnitError("a timestep override runs without the hidden law (its thermal step is 1 ms)")
        if noise_key is not None and (not isinstance(noise_key, (bytes, bytearray)) or len(noise_key) != 32):
            raise UnitError("noise_key: 32 bytes (bytes or bytearray) or None")
        if noise_key is not None and law is None:
            raise UnitError("reply noise needs the hidden law's noise plug-in (law=...)")
        if policy is None:
            policy = "none"
        if isinstance(policy, str):
            try:
                pol = rg.write_policy(policy)
            except rg.RegisterError as e:
                raise UnitError(f"policy: {e}") from None
        elif isinstance(policy, rg.WritePolicy):
            pol = policy
        else:
            raise UnitError("policy: a registers.WritePolicy or a profile name ('stage04', 'stage05', 'none')")
        if contact_classes is not None and not isinstance(contact_classes, Mapping):
            raise UnitError("contact_classes: a mapping {body or geom name: class} or None")
        ld = law_digest_of(law)
        spec = base.unit_spec(plant_edits(state, session), contacts=contacts)
        if _numerics_override is not None:
            _override(spec, _numerics_override, "spec")
        world.declare(spec)
        model = spec.compile()
        if _numerics_override is not None:
            _override(model, _numerics_override, "model")
        data = mujoco.MjData(model)
        editions = busmod.pair_modules(dict(state.editions))
        heat = noise = None
        if law is not None:
            heat = law.heat([m.edition for m in editions], list(state.thermal.h))
            if noise_key is not None:
                noise = law.noise(bytes(noise_key))
        c = state.cal
        bus = busmod.BusLayer(model, data, editions, c.registers, delta=c.delta, sign=c.sign, imax=c.imax, k=c.k,
                              b=[c.b[i] + session.b_step[i] for i in range(N_MODULES)], heat=heat, noise=noise,
                              t_amb=session.ambient_C if heat is not None else None,
                              temperatures=list(session.start_temperatures_C) if heat is not None else None,
                              policy=pol)
        base_level = [state.plant.frictionloss[i] * session.friction_factor[i] for i in range(N_MODULES)]
        profile = FrictionProfile(state.hs.period, state.hs.tables)
        ids = {"kind": "unit", "unit_digest": state.digest, "session_id": session.session_id,
               "session_digest": session.digest, "world_digest": world.digest(), "code_digest": code_digest(),
               "noise": noise_key is not None, "law": law is not None, "law_digest": ld, "contacts": bool(contacts),
               "override": None if _numerics_override is None else canonical_json(_jsonable(_numerics_override))}
        u = cls.from_parts(model, data, bus, world, base_level, profile, ids, contact_classes=contact_classes,
                           truth_profile=truth_profile)
        u.state, u.session, u.registered = state, session, registry is not None
        return u

    @classmethod
    def from_parts(cls, model, data, bus, world, friction_base, profile, ids, contact_classes=None, truth_profile=(),
                   tool_sites=None):
        """A unit on a compiled model and its bus layer (the public runner of a submitted model uses it). profile: a
        FrictionProfile or None (no friction profile); ids: the snapshot's identities (a mapping); tool_sites: None
        (the unit's tool point, module text) or the names of two sites of the model, left then right, at which the
        truth reads the tool frames. contact_classes: {body or geom name: class} or None.
        Order: the unit is put at the default pose, disabled; then world.bind(unit); then the hooks' per-step fields
        are asked once (also when no trace entry names one) and the trace plan is made. The configuration digest
        (module text) is added to `ids` (a config_digest that `ids` carries is replaced by this unit's own)."""
        u = cls.__new__(cls)
        u._init(model, data, bus, world, friction_base, profile, ids, contact_classes, truth_profile, tool_sites)
        return u

    def _init(self, model, data, bus, world, friction_base, profile, ids, contact_classes, truth_profile, tool_sites):
        self.model, self.data, self.bus, self.world = model, data, bus, world
        self.state = self.session = None
        self.registered = False
        self._ids = dict(ids)
        m = model
        self.dt = float(m.opt.timestep)
        self.hinge_jid = np.array([mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n) for n in base.HINGES], np.int64)
        if np.any(self.hinge_jid < 0):
            raise UnitError("the model lacks one of the eighteen hinges")
        self.hinge_qadr = np.array([m.jnt_qposadr[j] for j in self.hinge_jid], np.int64)
        self.hinge_dadr = np.array([m.jnt_dofadr[j] for j in self.hinge_jid], np.int64)
        self.finger2 = {}
        for s, side in enumerate(base.SIDES):
            j2 = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"openarm_{side}_finger_joint2")
            self.finger2[GRIPPER_MODULES[s]] = (int(m.jnt_qposadr[j2]), int(m.jnt_dofadr[j2]))
        arm = np.array(ARM_MODULES, dtype=np.int64)
        self.arm_qadr = bus.qadr[arm]
        self.arm_dadr = bus.dadr[arm]
        arm_jid = bus.jid[arm]
        self.arm_lo = np.array(m.jnt_range[arm_jid, 0], dtype=np.float64)
        self.arm_hi = np.array(m.jnt_range[arm_jid, 1], dtype=np.float64)
        driven = set(int(x) for x in bus.dadr)
        self.free_dofs = np.array([v for v in range(m.nv) if v not in driven], dtype=np.int64)
        self.friction_base = np.array([float(x) for x in friction_base], dtype=np.float64)
        if self.friction_base.shape != (N_MODULES,):
            raise UnitError("sixteen base friction levels")
        self.arm_base = self.friction_base[arm]
        self.profile = profile
        self._hs_active = profile is not None and not profile.flat
        m.dof_frictionloss[bus.dadr] = self.friction_base
        self.stops = StopReader(m, self.hinge_jid)
        if contact_classes is not None and not isinstance(contact_classes, Mapping):
            raise UnitError("contact_classes: a mapping {body or geom name: class} or None")
        cmap = dict(world.contact_classes())
        cmap.update(contact_classes or {})
        robot = [b for s in base.SIDES for b in base.ARM_BODIES[s]]
        robot += [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) for b in range(m.nbody)
                  if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or "").startswith("mount_")]
        self.contact_classes = ContactClasses(m, cmap, robot)
        self.ee_body = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, base.EE_BODIES[s]) for s in base.SIDES]
        self.tool_offset = tuple(float(x) for x in public_params()["conventions"]["tool_offset_m"]["value"])
        if tool_sites is None:
            self.tool_sites = None
            self.tool_site_ids = None
        else:
            names = tuple(str(x) for x in tool_sites)
            if len(names) != 2:
                raise UnitError("tool_sites: two site names, left then right")
            sid = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, n) for n in names]
            if any(x < 0 for x in sid):
                raise UnitError(f"the model lacks a tool site of {names}")
            self.tool_sites = names
            self.tool_site_ids = tuple(int(x) for x in sid)
        self.t_peak = bus.t_peak.copy()
        self.neg_t_peak = -self.t_peak
        # this layer's state (all of it in the snapshot; a unit marked invalid has no snapshot)
        self._invalid = None
        self._events = []
        self._prev_contacts = np.zeros(0, dtype=np.int64)
        self._prev_stops = np.zeros(0, dtype=np.int64)
        self._prev_clamp = np.zeros(N_MODULES, dtype=bool)
        self._prev_peak = np.zeros(N_MODULES, dtype=bool)
        self._prev_clamp_b = self._prev_clamp.tobytes()
        self._prev_peak_b = self._prev_peak.tobytes()
        self._has_world = bool(world)
        self._taps = []
        self._trace = []
        self._truth_cache = None
        self.exchange_count = 0
        self._getters = self._make_getters()
        tp = tuple(str(f) for f in truth_profile)
        if len(set(tp)) != len(tp):
            raise UnitError(f"a per-step trace entry is named twice: {tp}")
        own = tuple(f for f in tp if f in PER_STEP_FIELDS)
        hooked = tuple(f for f in tp if f not in PER_STEP_FIELDS)
        self.truth_profile = tp
        self._trace_own = own
        self._trace_on = bool(tp)
        self._hook_plan = ()
        # the configuration digest: one of the snapshot's identities
        cfg = {"format": CONFIG_FORMAT, "policy": bus.policy.to_list(),
               "contact_classes": [self.contact_classes.names[c] for c in self.contact_classes.geom_class.tolist()],
               "truth_profile": list(tp), "tool_sites": None if self.tool_sites is None else list(self.tool_sites)}
        self._ids["config_digest"] = hashlib.sha256(canonical_json(cfg).encode("ascii")).hexdigest()
        # the order: the default pose, disabled; then the world's bind; then its per-step fields, asked once
        self._set_pose(self.default_pose(), np.zeros(N_MODULES))
        world.bind(self)
        try:
            offered = world.offered_fields()
            plan = world.per_step_plan(hooked, offered=offered) if hooked else ()
        except HookError as e:
            raise UnitError(f"per-step trace entries not among {PER_STEP_FIELDS} and not offered by a hook: {e}") \
                from None
        self._hook_plan = plan
        self._clean_and_forward()          # the derived arrays at the default pose with what bind set in the model
        self._seed_detectors()
        # the plant's check: a live view of MuJoCo's warning counters and their bytes after the last step
        self._warn_view = data.warning.number
        self._warn_b = self._warn_view.tobytes()

    # ------------------------------------------------------------------------------------------------ poses
    @staticmethod
    def default_pose():
        """The default pose of reset in MODEL coordinates (params_public.json conventions.reset_pose_rad: J4 = 1.5708,
        all else 0 on both arms, grippers closed): 16 values in module order."""
        c = public_params()["conventions"]
        j = [float(x) for x in c["reset_pose_rad"]["value"]["joints"]]
        mir = (-1.0, -1.0, -1.0, 1.0, -1.0, -1.0, -1.0)           # the right arm mirrors the left: q_R = mir x q_L
        return tuple(j + [0.0] + [mir[k] * j[k] for k in range(7)] + [0.0])

    def module_q(self, bus_angles):
        """Model coordinates q of the sixteen module joints for module angles theta (theta = s (q - delta))."""
        th = np.array([float(x) for x in bus_angles], dtype=np.float64)
        if th.shape != (N_MODULES,):
            raise UnitError("sixteen module angles")
        return self.bus.delta + self.bus.sign * th

    def _set_pose(self, q16, qd16):
        d = self.data
        d.qpos[self.bus.qadr] = q16
        d.qvel[self.bus.dadr] = qd16
        for i, (qa, da) in self.finger2.items():
            d.qpos[qa] = d.qpos[self.bus.qadr[i]]
            d.qvel[da] = d.qvel[self.bus.dadr[i]]
        self._clean_and_forward()
        self._seed_detectors()

    def _clean_and_forward(self):
        d = self.data
        d.qacc_warmstart[:] = 0.0
        d.qfrc_applied[:] = 0.0
        d.xfrc_applied[:] = 0.0
        d.ctrl[:] = 0.0
        self._truth_cache = None
        self.bus.restored_stop_torque = None
        mujoco.mj_forward(self.model, d)

    def _seed_detectors(self):
        """The edge detectors of events from the state just set (after its mj_forward): the active contact pairs and
        stops of that state, and the torque-clamp flags from the bus layer's last drive torques at the present
        velocities; the events that follow then depend on the state alone."""
        d = self.data
        self._prev_contacts = self.contact_classes.active_codes(d)
        self._prev_stops = self.stops.active(d)
        fp = self.bus._fp_idx
        cl, pk = self._clamp_flags(fp, d.qvel[self.bus.dadr[fp]] if fp else None)
        self._prev_clamp, self._prev_peak = cl, pk
        self._prev_clamp_b, self._prev_peak_b = cl.tobytes(), pk.tobytes()

    # --------------------------------------------------------------------------------------------- set_state
    def set_state(self, q, qvel, world_state=None, thermal=None):
        """Model coordinates and velocities (16 module joints, the coupled fingers following; or the 18 hinges in
        base.HINGES order; or the whole qpos / qvel), a world state for the hooks, and the thermal state of every
        module ((T_w, T_h) pairs, 16; needs the hidden heat plug-in). The warm start is cleared (a defined start).
        UnitInvalid on a unit marked invalid (a set state does not revive it)."""
        self._require_valid()
        m, d = self.model, self.data
        q = np.array([float(x) for x in q], dtype=np.float64)
        v = np.array([float(x) for x in qvel], dtype=np.float64)
        if q.shape[0] == N_MODULES and v.shape[0] == N_MODULES:
            d.qpos[self.bus.qadr] = q
            d.qvel[self.bus.dadr] = v
            for i, (qa, da) in self.finger2.items():
                d.qpos[qa] = q[i]
                d.qvel[da] = v[i]
        elif q.shape[0] == len(base.HINGES) and v.shape[0] == len(base.HINGES):
            d.qpos[self.hinge_qadr] = q
            d.qvel[self.hinge_dadr] = v
        elif q.shape[0] == m.nq and v.shape[0] == m.nv:
            d.qpos[:] = q
            d.qvel[:] = v
        else:
            raise UnitError("q and qvel: 16 module joints, 18 hinges, or the whole qpos / qvel")
        if world_state is not None:
            self.world.set_world_state(self, world_state)
        if thermal is not None:
            t = [tuple(float(x) for x in p) for p in thermal]
            if len(t) != N_MODULES or any(len(p) != 2 for p in t):
                raise UnitError("thermal: sixteen (T_w, T_h) pairs")
            try:
                self.bus.set_temperatures([p[0] for p in t], [p[1] for p in t])
            except busmod.BusError as e:
                raise UnitError(str(e)) from None
        self.bus.restored_stop_torque = None
        d.qacc_warmstart[:] = 0.0
        self._truth_cache = None
        mujoco.mj_forward(m, d)
        self._seed_detectors()

    # ------------------------------------------------------------------------------------------------- reset
    def reset(self, bus_angles=None, hold=None, hold_s=None, ctrl_mode=None):
        """Put the unit at a pose and hand it over. Returns the replies of the reset's last exchange, which are the
        controller's first cycle of feedback (one exchange old, as all feedback): with `hold`, one state reply per
        module to the hold's control frames (in module order: left J1..J8, right J1..J8); without it, one state reply
        per module to a refresh. A host matches them to modules by bus and reply id.

        bus_angles: 16 module angles (theta), grippers included; None = the default pose (model coordinates,
        `default_pose`). ctrl_mode: {module: 'mit' | 'posforce'} for the two grippers (7, 15), written by register
        frames while the modules are disabled (the reset is the grader's host: its writes pass whatever the task's
        write whitelist is). hold: a HostProfile: the modules are enabled and the pose is held for hold_s seconds
        (default params_public.json conventions.reset_hold_s) by exchanges of 4 steps with the profile's joint PD
        (v_des = 0, t_ff = 0); each gripper gets frames of ITS mode with the profile's gripper settings
        (force-position: speed limit, current limit = the profile's limit / the plateau divisor of the public file;
        MIT: kp, kd); the last commands stay in force. With `hold`, UnitError when a module is not enabled after the
        enable (a latched fault that clear could not clear) or is no longer enabled at the end of the hold; the error
        names each such module and its status. Without hold: the modules stay disabled and the arms settle for
        conventions.reset_settle_s seconds; the last exchange carries a refresh of every module.
        Order: the world's reset; disable and clear every module (no physics step); the mode writes; the pose (q, qvel
        0, warm start cleared); enable (with hold); the exchanges. Time advances through the unit's own exchanges, so
        the step and exchange counts grow by the hold or the settling; every frame of the reset is answered as the bus
        layer answers it (the replies are not returned, except the last exchange's) and its events stay queued for
        events(). Thermal state is not touched (it is session state). Every argument (the mode table, the angles, the
        hold's type and time) is checked before the reset acts: a refused reset leaves the unit's state unchanged.
        UnitInvalid on a unit marked invalid (a reset does not revive it), and from the reset's own exchanges when a
        step of them is one the plant cannot integrate."""
        self._require_valid()
        B = self.bus
        conv = public_params()["conventions"]
        # the arguments, before anything acts
        modes = {}
        if ctrl_mode:
            try:
                table = dict(ctrl_mode)
            except (TypeError, ValueError):
                raise UnitError("ctrl_mode: a mapping {7 | 15: 'mit' | 'posforce'}") from None
            for mod, name in table.items():
                if isinstance(mod, str) and mod.strip().isdigit():
                    mod = int(mod)                       # a table read from JSON names its modules by strings
                if isinstance(mod, (bool, np.bool_)) or not isinstance(mod, numbers.Integral) \
                        or int(mod) not in GRIPPER_MODULES:
                    raise UnitError("ctrl_mode names the grippers (modules 7 and 15) only")
                if not isinstance(name, str) or name not in MODE_NAMES:
                    raise UnitError("a mode is 'mit' or 'posforce'")
                modes[int(mod)] = name
        if bus_angles is None:
            q16 = np.array(self.default_pose(), dtype=np.float64)
            theta = B.sign * (q16 - B.delta)
        else:
            try:
                theta = np.array([float(x) for x in bus_angles], dtype=np.float64)
            except (TypeError, ValueError):
                raise UnitError("bus_angles: sixteen module angles") from None
            if theta.shape != (N_MODULES,) or not np.all(np.isfinite(theta)):
                raise UnitError("sixteen module angles")
            q16 = None                                   # read against the zeros after the reset's frames
        if hold is not None and not isinstance(hold, HostProfile):
            raise UnitError("hold is a HostProfile")
        if hold is None:
            secs = float(conv["reset_settle_s"]["value"])
        else:
            try:
                secs = float(conv["reset_hold_s"]["value"]) if hold_s is None else float(hold_s)
            except (TypeError, ValueError):
                raise UnitError("hold_s: a number of seconds") from None
            if not math.isfinite(secs):
                raise UnitError("hold_s: a finite number of seconds")
        n_steps = int(round(secs / self.dt))
        if n_steps < 1:
            raise UnitError("a reset runs at least one physics step")
        # the reset
        self.world.reset(self)
        fr = []
        for i in range(N_MODULES):
            bi, cid = B.modules[i].bus, B.esc_id[i]
            fr += [hc.disable(bi, cid), hc.clear(bi, cid)]
        B.receive(fr, steps=50, admit="all")
        if modes:
            pairs, wf, want = [], [], {}
            for mod, name in modes.items():
                pairs.append((mod, rg.CTRL_MODE))
                wf.append(hc.write_param(B.modules[mod].bus, B.esc_id[mod], rg.CTRL_MODE, MODE_NAMES[name]))
                want[mod] = MODE_NAMES[name]
            saved = B.policy
            B.policy = rg.WritePolicy(tuple(saved.pairs) + tuple(pairs))
            try:
                B.receive(wf, steps=50, admit="all")
            finally:
                B.policy = saved
            for mod, code in want.items():
                if int(B.mode[mod]) != code:
                    raise UnitError(f"module {mod}: the mode write was not accepted")
        if q16 is None:
            q16 = self.module_q(theta)
        self._set_pose(q16, np.zeros(N_MODULES))
        if hold is None:
            last = [hc.refresh(B.modules[i].bus, B.esc_id[i]) for i in range(N_MODULES)]
            return self._run_steps(n_steps, None, last)
        B.receive([hc.enable(B.modules[i].bus, B.esc_id[i]) for i in range(N_MODULES)], steps=50, admit="all")
        self._require_enabled("after the enable")
        frames = self.hold_frames(hold, theta)
        replies = self._run_steps(n_steps, frames, frames)
        self._require_enabled("at the end of the hold")
        return replies

    def _require_enabled(self, when):
        B = self.bus
        bad = np.nonzero(~B.enabled)[0].tolist()
        if bad:
            st = B.status()
            desc = ", ".join(f"module {i} (bus {B.modules[i].bus}, reply id 0x{B.reply_id[i]:02X}): status "
                             f"0x{int(st[i]):X}" for i in bad)
            raise UnitError(f"reset: not enabled {when}: {desc}")

    def hold_frames(self, hold, theta):
        """The frames of a joint-PD hold of module angles `theta` with HostProfile `hold` (each module in its own
        mode; encoded with the module's own mapping registers)."""
        B = self.bus
        div = float(public_params()["gripper"]["plateau_divisor"]["value"])
        out = []
        for i in range(N_MODULES):
            bi, cid = B.modules[i].bus, B.esc_id[i]
            j = i % 8
            if int(B.mode[i]) == rg.MODE_POS_FORCE:
                out.append(hc.pack_posforce(bi, cid, float(theta[i]), hold.gripper_fp_speed,
                                            hold.gripper_fp_limit / div))
                continue
            if j == 7:
                kp, kd = hold.gripper_mit_kp, hold.gripper_mit_kd
            else:
                kp, kd = hold.kp[j], hold.kd[j]
            out.append(hc.pack_mit(bi, cid, B.limits(i), float(theta[i]), hold.v_des, kp, kd, hold.t_ff))
        return out

    def _run_steps(self, n_steps, frames, last_frames):
        """Exchanges of 4 steps (the last one shorter if needed) with `frames` in each and `last_frames` in the last;
        returns the replies of the last exchange."""
        done, replies = 0, []
        while done < n_steps:
            k = min(4, n_steps - done)
            is_last = done + k >= n_steps
            replies = self.exchange(last_frames if is_last else frames, steps=k)
            done += k
        return replies

    # ---------------------------------------------------------------------------------------------- exchange
    def exchange(self, frames, steps=4, taps=(), admit="all"):
        """One exchange of `steps` physics steps (1..50): the frames in the order sent (`admit` 'control_only' for a
        controller's frames: the bus layer drops every other kind), the replies sampled at reception; `taps`: step
        offsets 0..steps-1 inside the exchange at which a reply-coded sample of every module is latched (offset j =
        the state after j physics steps of this exchange, j = 0 being the reply's own instant; off the bus and its
        budget; `tapped()` drains them). Every argument is checked before any frame is received: `frames` is None or
        a list or tuple of Frame; `steps` and every tap offset are integral numbers (4.7 is refused, never truncated);
        a refused call (UnitError) leaves the unit's state unchanged. UnitInvalid (module text, "A STEP THE PLANT
        CANNOT INTEGRATE") from a step of this exchange that the plant cannot integrate, before its replies are
        returned, and from every exchange of a unit marked invalid."""
        self._require_valid()
        if frames is not None:
            if not isinstance(frames, (list, tuple)):
                raise UnitError("frames: a list or tuple of Frame, or None")
            for f in frames:
                if not isinstance(f, Frame):
                    raise UnitError("frames: a list or tuple of Frame, or None")
        steps = _integral(steps, "steps")
        if isinstance(taps, (str, bytes, bytearray)) or not hasattr(taps, "__iter__"):
            raise UnitError("taps: a sequence of integral step offsets")
        tapset = sorted(set(_integral(t, "a tap offset") for t in taps))
        if not 1 <= steps <= 50:
            raise UnitError("an exchange has 1 to 50 steps")
        if tapset and (tapset[0] < 0 or tapset[-1] >= steps):
            raise UnitError("a tap offset lies in 0 .. steps - 1")
        if admit not in busmod.ADMIT:
            raise UnitError(f"admit is one of {busmod.ADMIT}")
        try:
            replies = self.bus.receive(frames, steps, admit)
        except busmod.BusError as e:
            raise UnitError(str(e)) from None
        self._drain_bus()
        ti = 0
        for j in range(steps):
            if ti < len(tapset) and tapset[ti] == j:
                self._taps.append((self.bus.step_count, j, self.bus.coded_states("tap")))
                ti += 1
            self._step()
        self.exchange_count += 1
        return replies

    def _drain_bus(self):
        if self.bus._events:
            for e in self.bus.drain_events():
                self._events.append(Event.from_dict(e))

    def _step(self):
        m, d, B, W = self.model, self.data, self.bus, self.world
        n = B.step_count
        hw = self._has_world
        B.restored_stop_torque = None
        self._truth_cache = None
        if hw:
            d.qfrc_applied[self.free_dofs] = 0.0
            d.xfrc_applied[:] = 0.0
        B.pre_step()
        if self._hs_active:
            m.dof_frictionloss[self.arm_dadr] = self.profile.level(d.qpos[self.arm_qadr], self.arm_base)
        fp = B._fp_idx
        w0 = d.qvel[B.dadr[fp]] if fp else None
        if hw:
            W.pre_step(self, n)
        mujoco.mj_step(m, d)
        B.post_step()
        if B._events:
            self._drain_bus()
        if self._warn_view.tobytes() != self._warn_b:
            self._plant_warning(n)
        self._detect(n, fp, w0)
        if hw:
            W.post_step(self, n)
            for e in W.events():
                e = dict(e)
                self._events.append(Event(e.pop("kind"), int(e.pop("step", n)), e))
        if self._trace_on:
            r = self._per_step_record(n)
            if self._hook_plan:
                try:
                    r.update(W.per_step(self, n, self._hook_plan))
                except HookError as e:
                    raise UnitError(str(e)) from None
            self._trace.append(r)

    # ------------------------------------------------------------------------------------- the plant's check
    def _plant_warning(self, n):
        """A warning counter changed in step n: when one of PLANT_WARNINGS rose, the event `plant_invalid`, the unit
        marked invalid and UnitInvalid raised; a change of any other counter is only taken note of."""
        now = self._warn_view.copy()
        prev = np.frombuffer(self._warn_b, dtype=now.dtype)
        self._warn_b = now.tobytes()
        risen = [(name, i) for name, i in _PLANT_WARNING_INDEX if int(now[i]) > int(prev[i])]
        if not risen:
            return
        kind, i0 = risen[0]
        kinds = [name for name, _ in risen]
        info = int(self.data.warning.lastinfo[i0])
        self._invalid = {"step": int(n), "kind": kind, "kinds": kinds, "info": info}
        self._events.append(Event("plant_invalid", int(n), {"warning": kind, "warnings": list(kinds), "info": info}))
        raise UnitInvalid(n, kind, kinds, info)

    def _require_valid(self):
        v = self._invalid
        if v is not None:
            raise UnitInvalid(v["step"], v["kind"], v["kinds"], v["info"])

    @property
    def invalid(self):
        """None while the plant's check has not failed; else {step, kind, kinds, info} (module text)."""
        v = self._invalid
        return None if v is None else {"step": v["step"], "kind": v["kind"], "kinds": list(v["kinds"]),
                                       "info": v["info"]}

    # ------------------------------------------------------------------------------------------ event detection
    def _clamp_flags(self, fp, w0):
        """(limit, peak) flags of the sixteen modules from the bus layer's T_ref / T_del; w0: the velocities of the
        force-position modules `fp` at the step's start."""
        B = self.bus
        # torque clamps: the drive's reference beyond its current limit / envelope L, or the clamped command beyond
        # the peak torque; MIT modules from the bus layer's T_ref / T_del, force-position modules against L itself:
        # a force-position module's commanded current limit is a command, not a clamp, so a gripper holding at its
        # force limit raises no torque_clamped event
        tr, td = B.T_ref, B.T_del
        en = B.enabled
        cl = td != tr
        cl &= en
        pk = np.abs(td) > self.t_peak
        pk &= en
        if fp:
            for idx, i in enumerate(fp):
                if not cl[i]:
                    continue                       # not clipped: below its cap, hence below L (pk already right)
                w = float(B.sign[i]) * float(w0[idx])
                e0 = self._E_tlim(i, 0.0)
                lp = self._E_tlim(i, w) if w > 0.0 else e0
                lm = self._E_tlim(i, -w) if w < 0.0 else e0
                t = float(tr[i])
                cl[i] = t > lp or t < -lm
                tl = min(max(t, -lm), lp)
                pk[i] = tl > float(self.t_peak[i]) or tl < -float(self.t_peak[i])
        return cl, pk

    def _detect(self, n, fp, w0):
        d = self.data
        cl, pk = self._clamp_flags(fp, w0)
        cb, pb = cl.tobytes(), pk.tobytes()
        if cb != self._prev_clamp_b or pb != self._prev_peak_b:
            for which, now, prev in (("limit", cl, self._prev_clamp), ("peak", pk, self._prev_peak)):
                for i in np.nonzero(now != prev)[0].tolist():
                    self._events.append(Event("torque_clamped", n, {"module": i, "which": which, "on": bool(now[i])}))
            self._prev_clamp, self._prev_peak = cl, pk
            self._prev_clamp_b, self._prev_peak_b = cb, pb
        # contacts by class (the touching finger pair of a gripper excluded)
        codes = self.contact_classes.active_codes(d)
        if codes.tobytes() != self._prev_contacts.tobytes():
            new, old = set(codes.tolist()), set(self._prev_contacts.tolist())
            for c in sorted(new - old):
                self._events.append(Event("contact_begin", n, {"pair": list(self.contact_classes.pair_name(c))}))
            for c in sorted(old - new):
                self._events.append(Event("contact_end", n, {"pair": list(self.contact_classes.pair_name(c))}))
            self._prev_contacts = codes
        # stops: the joint-limit constraint active with more than 0.01 N m, on every one of the eighteen hinges
        st = self.stops.active(d)
        if st.tobytes() != self._prev_stops.tobytes():
            new, old = set(st.tolist()), set(self._prev_stops.tolist())
            for c in sorted(new - old):
                self._events.append(Event("stop_begin", n, {"joint": base.HINGES[c // 2],
                                                            "which": "upper" if c % 2 else "lower"}))
            for c in sorted(old - new):
                self._events.append(Event("stop_end", n, {"joint": base.HINGES[c // 2],
                                                          "which": "upper" if c % 2 else "lower"}))
            self._prev_stops = st

    def _E_tlim(self, i, w):
        """E(w) of module i with its current limit T_lim (envelope.py's form)."""
        B = self.bus
        return max(0.0, min(float(B.t_lim[i]), float(B.a1[i]) + float(B.b1[i]) * w, float(B.a2[i]) + float(B.b2[i]) * w))

    # --------------------------------------------------------------------------------------------------- truth
    def _stop_margin(self):
        q = self.data.qpos[self.arm_qadr]
        return np.minimum(q - self.arm_lo, self.arm_hi - q)

    def _make_getters(self):
        B, d, m = self.bus, self.data, self.model
        return {
            "theta": B.theta, "T_ref": lambda: B.T_ref.copy(), "T_del": lambda: B.T_del.copy(),
            "delivered": self._delivered, "k": lambda: B.k.copy(), "b": lambda: B.b.copy(), "status": B.status,
            "enabled": lambda: B.enabled.copy(), "mode": lambda: B.mode.copy(), "T_w": lambda: B.T_w.copy(),
            "T_h": lambda: B.T_h.copy(), "steps_since_control": lambda: B.steps_since.copy(),
            "command_mit": lambda: np.stack([B.p_des, B.v_des, B.kp, B.kd, B.t_ff], axis=1).copy(),
            "command_fp": lambda: np.stack([B.fp_p, B.fp_vlim, B.fp_ilim], axis=1).copy(),
            "delta": lambda: B.delta.copy(), "t_lim": lambda: B.t_lim.copy(), "Z": lambda: B.Z.copy(),
            "q": lambda: d.qpos[self.hinge_qadr].copy(), "qvel": lambda: d.qvel[self.hinge_dadr].copy(),
            "friction_level": lambda: m.dof_frictionloss[self.hinge_dadr].copy(), "stop_margin": self._stop_margin,
            "clamped": lambda: self._prev_clamp.copy(), "clamped_peak": lambda: self._prev_peak.copy(),
            "friction_profile": self.friction_profile,
        }

    # what the truth reads from MuJoCo's derived arrays: from the stored values right after a restore
    def _delivered(self):
        c = self._truth_cache
        return c["delivered"].copy() if c is not None else self.bus.delivered()

    def _stop_torque(self):
        c = self._truth_cache
        return c["stop_torque"].copy() if c is not None else self.stops.torques(self.data)

    def _contacts(self):
        c = self._truth_cache
        return copy.deepcopy(c["contacts"]) if c is not None else self.contact_classes.pair_forces(self.model,
                                                                                                    self.data)

    def _world_truth(self):
        c = self._truth_cache
        return copy.deepcopy(c["world"]) if c is not None else self.world.truth(self)

    def _tool_poses(self):
        m, d = self.model, self.data
        if self.tool_site_ids is None:
            poses = [frame_pose(d, b, self.tool_offset) for b in self.ee_body]
        else:
            poses = [site_pose(m, d, s) for s in self.tool_site_ids]
        rel_p, rel_q = relative_pose(poses[0][0], poses[0][2], poses[0][1], poses[1][0], poses[1][1])
        return {"tool_pos": np.stack([p[0] for p in poses]), "tool_quat": np.stack([p[1] for p in poses]),
                "tool_rel_pos": rel_p, "tool_rel_quat": rel_q}

    def _per_step_record(self, n):
        B, d = self.bus, self.data
        r = {"step": n}
        for f in self._trace_own:
            if f == "q":
                r[f] = d.qpos[self.hinge_qadr].copy()
            elif f == "qvel":
                r[f] = d.qvel[self.hinge_dadr].copy()
            elif f == "theta":
                r[f] = B.theta()
            elif f == "T_ref":
                r[f] = B.T_ref.copy()
            elif f == "T_del":
                r[f] = B.T_del.copy()
            elif f == "delivered":
                r[f] = B.delivered()
            elif f == "friction_level":
                r[f] = self.model.dof_frictionloss[self.hinge_dadr].copy()
            elif f == "stop_margin":
                r[f] = self._stop_margin()
            elif f == "steps_since_control":
                r[f] = B.steps_since.copy()
            elif f == "T_w":
                r[f] = B.T_w.copy()
            elif f == "T_h":
                r[f] = B.T_h.copy()
            elif f == "status":
                r[f] = B.status()
            elif f == "enabled":
                r[f] = B.enabled.copy()
            elif f == "energy":
                r[f] = d.energy.copy()          # MuJoCo's (potential, kinetic) of the state at this step's start
            elif f == "armature":
                r[f] = self.model.dof_armature[B.dadr].copy()   # the inertia entries this step ran with
        return r

    def truth_trace(self):
        """The per-step truth records since the last call (the entries of `truth_profile`, each record with its
        step; a hook's entry under its "<hook>.<field>" name); drained. UnitInvalid on a unit marked invalid."""
        self._require_valid()
        t, self._trace = self._trace, []
        return t

    def tapped(self):
        """The latched taps since the last call: (step, offset, 16 reply-coded rows of 8 bytes); drained. UnitInvalid on
        a unit marked invalid."""
        self._require_valid()
        t, self._taps = self._taps, []
        return t

    def friction_profile(self):
        """The fourteen stored friction-profile tables (period, 256 values) of the arm joints, abscissa q; None when
        the unit has none (a model run)."""
        return None if self.profile is None else self.profile.handout()

    def truth(self, fields=None):
        """The truth now (module text). fields: None (every field) or an iterable of field names (TRUTH_FIELDS): only
        the families they need are computed and only they (with step and exchange) are returned. UnitInvalid on a
        unit marked invalid."""
        self._require_valid()
        if fields is None:
            want = None
        else:
            if isinstance(fields, str):
                fields = (fields,)
            want = frozenset(str(f) for f in fields)
            bad = want - frozenset(TRUTH_FIELDS)
            if bad:
                raise UnitError(f"unknown truth fields {sorted(bad)}; known: {TRUTH_FIELDS}")
        f = {"step": int(self.bus.step_count), "exchange": self.exchange_count}
        g = self._getters
        for k in _GETTER_FIELDS:
            if want is None or k in want:
                f[k] = g[k]()
        if want is None or "stop_torque" in want:
            f["stop_torque"] = self._stop_torque()
        if want is None or "contacts" in want:
            f["contacts"] = self._contacts()
        kin = want is None or not want.isdisjoint(_KIN_FIELDS)
        wld = want is None or "world" in want
        if kin or wld:
            mujoco.mj_kinematics(self.model, self.data)   # poses at the present q (the solved step's are a step older)
        if kin:
            for k, v in self._tool_poses().items():
                if want is None or k in want:
                    f[k] = v
        if wld:
            f["world"] = self._world_truth()
        return Truth(f)

    def events(self):
        """The events since the last call, oldest first; drained (also on a unit marked invalid: its last event is
        `plant_invalid`)."""
        self._drain_bus()
        e, self._events = self._events, []
        return e

    # ------------------------------------------------------------------------------------------------ snapshot
    def _header(self):
        h = dict(self._ids)
        h.update({"format": SNAPSHOT_FORMAT, "exchange_count": self.exchange_count, "step_count": self.bus.step_count})
        return h

    def _physical(self, observation=False):
        """The physical state (what the state digest covers; module text), JSON-able. observation=True: with the bus
        layer's frame and drop counters and the noise counters of its tap stream (what a snapshot holds)."""
        m, d = self.model, self.data
        st = np.empty(mujoco.mj_stateSize(m, _INTEG), dtype=np.float64)
        mujoco.mj_getState(m, d, st, _INTEG)
        bus = self.bus.state_dict()
        del bus["events"]
        if not observation:
            del bus["frames_in"]                 # observation: in the snapshot, not in the digest
            del bus["drops"]
            bus["noise_counters"] = [e for e in bus["noise_counters"] if e[0] != "tap"]
        return _jsonable({"format": DIGEST_FORMAT, "mj_state": st, "dof_frictionloss": m.dof_frictionloss,
                          "bus": bus, "world": self.world.state(), "world_model_fields": self.world.model_fields(self),
                          "exchange_count": self.exchange_count, "step_count": int(self.bus.step_count)})

    def snapshot(self):
        """The unit's whole state as bytes (module text). UnitInvalid on a unit marked invalid."""
        self._require_valid()
        phys = self._physical(observation=True)
        own = {"events": [e.to_dict() for e in self._events], "bus_events": [dict(e) for e in self.bus._events],
               "prev_contacts": self._prev_contacts.tolist(), "prev_stops": self._prev_stops.tolist(),
               "prev_clamp": self._prev_clamp.tolist(), "prev_peak": self._prev_peak.tolist(),
               "taps": [[s, j, [r.hex() for r in rows]] for s, j, rows in self._taps],
               "trace": [{k: _jsonable(v) for k, v in r.items()} for r in self._trace]}
        if self._truth_cache is None and self._has_world:
            mujoco.mj_kinematics(self.model, self.data)      # the world's truth is read at the present q
        cache = {"delivered": self._delivered(), "stop_torque": self._stop_torque(), "contacts": self._contacts(),
                 "world": self._world_truth(), "bus_stop_torque": self._bus_stop_torques()}
        body = {"physical": phys, "unit": own, "truth_cache": cache}
        text = canonical_json({"header": _jsonable(self._header()), "body": _jsonable(body)})
        return SNAPSHOT_MAGIC + text.encode("ascii")

    def _bus_stop_torques(self):
        """The torque on each module's stops that a save zero received now would report (the bus layer's value)."""
        r = self.bus.restored_stop_torque
        if r is not None:
            return [float(x) for x in r]
        return [self.bus._stop_torque(i) for i in range(self.bus.n)]

    def state_digest(self):
        """sha256 of the canonical JSON of the physical state (module text). UnitInvalid on a unit marked invalid."""
        self._require_valid()
        return hashlib.sha256(canonical_json(self._physical()).encode("ascii")).hexdigest()

    def motion_digest(self):
        """sha256 of qpos and qvel of every dof (float64 little endian) and the step count (8 bytes, big endian).
        UnitInvalid on a unit marked invalid."""
        self._require_valid()
        d = self.data
        h = hashlib.sha256()
        h.update(np.ascontiguousarray(d.qpos, dtype="<f8").tobytes())
        h.update(np.ascontiguousarray(d.qvel, dtype="<f8").tobytes())
        h.update(int(self.bus.step_count).to_bytes(8, "big", signed=True))
        return h.hexdigest()

    def restore(self, blob):
        """Restore a snapshot of THIS unit (same identities: code, unit, session, world, law, numerics, configuration;
        refused otherwise). UnitInvalid on a unit marked invalid (a restored snapshot does not revive it)."""
        self._require_valid()
        if not isinstance(blob, (bytes, bytearray)) or not bytes(blob).startswith(SNAPSHOT_MAGIC):
            raise UnitError("not a snapshot of an OpenArmUnit (or of another snapshot format)")
        obj = json.loads(bytes(blob)[len(SNAPSHOT_MAGIC):].decode("ascii"))
        h, b = obj["header"], obj["body"]
        if h.get("format") != SNAPSHOT_FORMAT:
            raise UnitError("snapshot format")
        mine = _jsonable(self._ids)
        for k, v in mine.items():
            if h.get(k) != v:
                raise UnitError(f"the snapshot is of another unit, session, world, law, code or configuration ({k})")
        p, own, cache = b["physical"], b["unit"], b["truth_cache"]
        if p.get("format") != DIGEST_FORMAT:
            raise UnitError("snapshot state format")
        bst = [float(x) for x in cache.get("bus_stop_torque", ())]
        if len(bst) != self.bus.n:
            raise UnitError("the snapshot's stop torques do not fit this bus layer")
        m, d = self.model, self.data
        st = np.array(p["mj_state"], dtype=np.float64)
        if st.shape[0] != mujoco.mj_stateSize(m, _INTEG):
            raise UnitError("the snapshot's simulator state does not fit this model")
        mujoco.mj_setState(m, d, st, _INTEG)
        m.dof_frictionloss[:] = np.array(p["dof_frictionloss"], dtype=np.float64)
        bs = dict(p["bus"])
        bs["events"] = own["bus_events"]
        self.bus.load_state_dict(bs)
        self.world.load_state(p["world"])
        self.world.load_model_fields(self, p["world_model_fields"])
        self.exchange_count = int(p["exchange_count"])
        self._events = [Event.from_dict(e) for e in own["events"]]
        self._prev_contacts = np.array(own["prev_contacts"], dtype=np.int64)
        self._prev_stops = np.array(own["prev_stops"], dtype=np.int64)
        self._prev_clamp = np.array(own["prev_clamp"], dtype=bool)
        self._prev_peak = np.array(own["prev_peak"], dtype=bool)
        self._prev_clamp_b = self._prev_clamp.tobytes()
        self._prev_peak_b = self._prev_peak.tobytes()
        self._taps = [(int(s), int(j), [bytes.fromhex(r) for r in rows]) for s, j, rows in own["taps"]]
        self._trace = [{k: (np.array(v) if isinstance(v, list) else v) for k, v in r.items()} for r in own["trace"]]
        mujoco.mj_forward(m, d)
        self._truth_cache = {"delivered": np.array(cache["delivered"], dtype=np.float64),
                             "stop_torque": np.array(cache["stop_torque"], dtype=np.float64).reshape(
                                 len(base.HINGES), 2),
                             "contacts": cache["contacts"], "world": cache["world"]}
        self.bus.restored_stop_torque = bst


def _integral(v, what):
    """An integral number as an int (an int, a numpy integer, or a float of integral value); UnitError otherwise (a
    bool, 4.7, NaN, a string ...)."""
    if isinstance(v, (bool, np.bool_)):
        raise UnitError(f"{what}: an integral number, not {v!r}")
    if isinstance(v, numbers.Integral):
        return int(v)
    if isinstance(v, numbers.Real):
        x = float(v)
        if math.isfinite(x) and x == math.floor(x):
            return int(x)
    raise UnitError(f"{what}: an integral number, not {v!r}")


def _jsonable(v):
    """JSON-able data with exact floats (numpy arrays to lists, numpy scalars to Python scalars, tuples to lists)."""
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, np.bool_):
        return bool(v)
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        return float(v)
    if isinstance(v, (bytes, bytearray)):
        return bytes(v).hex()
    return v
