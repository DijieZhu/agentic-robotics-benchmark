"""The bus layer of the joint modules on a MuJoCo model handle (the module clamps what it commands
before the scale). PUBLIC.

PUBLIC: this module ships with a task's public runner of a submitted model. It imports numpy, MuJoCo and the
public modules `frames`, `hostcodec`, `registers`, `envelope`, `state`; nothing of the law, the registry, the truth
channel, the world, the recorder, the thermal model (`thermal.py`) or the friction profile (`hs.py`). It holds NO
constant of the thermal model and NO constant of the reply-noise law: both enter only as objects the
hidden unit passes in (`heat`, `noise`, protocols below). Without them (a submitted model's run;
params_public.json `model_run`) the replies carry the truncated true values, both temperature bytes read the fixed
public value (25 deg C) and no coil protection exists.

What one BusLayer is: N joint modules (sixteen on the pair; any N on a test model), each driving one hinge through one
`general` actuator of the compiled model, with per module: enable flag, latched fault, control mode (its CTRL_MODE
register), the last decoded command, the steps since its last control frame, the last drive torque T_del (the reply's
torque field and the heat's source), the winding and housing temperatures, the force-position integral Z, its register
table, delta, s, k, b; per bus the frame and drop counters; the noise counters; the step count. ALL of it is in
`state_dict()` (everything a snapshot holds); `load_state_dict()` restores it.

THE IN-DRIVE LAW ("clamp before scale"), in the drive's units, at every physics step
on the true state at the step's start, the last command held:

    T_ref   = kp (p_des - theta) + kd (v_des - s qdot) + t_ff,           theta = s (q - delta)
    L_plus  = min(T_lim, E(max(s qdot, 0))),  L_minus = min(T_lim, E(max(-s qdot, 0)))     (envelope.py)
    T_del   = clip(T_ref, -L_minus, +L_plus)        what the drive commands as current: the reply's torque field, the heat
    R_+-    = min(L_+-, T_peak)                     the module's whole clamp, still in the drive's units
    tau     = s k clip(T_ref, -R_minus, +R_plus) + b                         what the joint receives

carried by the joint's own actuator: gainprm[0] = 1, biasprm = [0, -k kp, 0], forcerange = [-k R_minus, +k R_plus] for
s = +1 (ends swapped for s = -1), forcelimited on, and at every step
    ctrl = s k (kp (p_des + s delta) + kd v_des + t_ff) - k kd qdot,
with the joint's inertia entry (dof_armature) raised by dt k kd while the module is not saturated at the step's start
(-R_minus < T_ref < R_plus) and left at the model's value while it is (the implicit damping device). b goes to
qfrc_applied, outside both clamps and outside the reply. The joint has NO actuatorfrcrange (base.py); the reply's T_del
is computed here from T_ref, never read back from actuator_force (at a DM4310 V1.1 stall the reply reads
7.757 N m and the joint receives k x 7.0).

THE FORCE-POSITION LAW, a module whose CTRL_MODE is 4:
    v_ref = clip(KP_APR (p_des - theta), +-v_lim);  T_ref = KP_ASR (v_ref - s qdot) + Z;
    T_del = clip(T_ref, -min(I, L_minus), +min(I, L_plus)),  I = i_lim Imax KT;   Z += dt KI_ASR (v_ref - s qdot),
the increment held only while the output is clipped AND it would drive T_ref further into that clip (conditional
integration: held when T_ref lies above the upper end and v_ref - s qdot > 0, or below the lower end and
v_ref - s qdot < 0; an opening command after a squeeze therefore always unwinds the integral). "The output" is the
drive's own command T_del, clipped by the current limit and the envelope; the peak torque clamps only what the joint
receives, as in MIT mode, so a module pressed on a stop at i_lim = 1 reports its current limit and delivers k T_peak.
The carrier is the device above with KP_ASR in the place of
kd. THE LOOP GAINS have no published value: they are the constants of params_public.json `gripper.loop_gains` (KP_APR,
KP_ASR, KI_ASR, fitted to the gripper's real closing and opening transients), the same for every module in this
mode, and the bus layer has no other source for them: a null or incomplete entry there is an error when a bus layer is
built (BusError). They are not registers: registers 25 to 27 (KP_ASR, KI_ASR, KP_APR) are not carried, and a read of
one is answered like that of any register the unit does not carry (the request echoed). A caller may still give the
three gains to the constructor (`fp_gains`); `fp_gains_source` says which source a bus layer took ('params_public' or
'given').

A DISABLED OR FAULTED module: gain, bias, ctrl and force range zero, the inertia entry at the model's value, T_ref =
T_del = 0. The offset b acts only while the module is enabled (it is part of what the module delivers: a disabled
joint is free against its own friction and damping, and falls against friction and damping only).

ONE EXCHANGE, as the unit layer drives it:
    replies = bus.receive(frames, steps, admit)      # reception in the order sent, replies sampled now
    for each of the `steps` physics steps:
        bus.pre_step()                               # (a) protections, (b) force range, ctrl, inertia entry, b
        ... (c) the unit layer: friction level from the hold-scatter table (HS1), hook forces (added to
            qfrc_applied AFTER pre_step, which ASSIGNS qfrc_applied at the N driven dofs only), hook switches ...
        mujoco.mj_step(model, data)                  # (d)
        bus.post_step()                              # (e) T_del, thermal update, Z; counters; step count
`run_exchange` does exactly this with nothing in between (tests, a plain runner).
Reception has two paths with the same results: an exchange whose frames are all well-formed control frames, each to a
module in that frame's mode, at most one per module and inside the budget, is decoded and answered in one pass (ids
through a lookup array, all MIT frames in one array operation, all replies sampled together; a command equal, bit for
bit, to the one in force leaves the per-step coefficients as they are); every other exchange is taken frame by frame
in the order below.

THE RUNTIME CONTRACT, in this order per frame:
 1. malformed (length other than 8, unknown bus, an id above 11 bits, a force-position frame whose position is not a
    finite float32: such a position cannot be executed): dropped, no reply, event frame_dropped(reason='malformed');
 2. admission: with admit='control_only' every frame that is not an MIT or a
    force-position control frame is dropped before decoding, no reply, frame_dropped(reason='not_admitted');
 3. the bus budget: at most 8 x steps frames per bus and exchange are accepted, the rest in the order sent are dropped
    (reason 'budget'); a frame counts against its bus once it passes 1 and 2, whatever happens to it next;
 4. an id no module answers (reason 'unknown_id'), a kind not implemented (position-velocity id + 0x100, velocity id +
    0x200, a 0x7FF operation other than 33 / 55 / AA / CC: reason 'unsupported');
 5. a control frame (MIT on the command id, force-position on id + 0x300): a second one to the same module in the same
    exchange is dropped (reason 'duplicate'); the module's one control slot is taken by the first control frame of
    either kind, also by one of the wrong kind; a frame of the wrong kind for the module's CTRL_MODE is ignored WITHOUT
    a reply (in both directions; event frame_dropped(reason='wrong_mode')); to a disabled or faulted module
    it is answered with its state and has no effect; otherwise the command is stored (a force-position position beyond
    PMAX is clipped to +-PMAX), the module's time-out counter reset, and the reply sampled now, before the new command
    acts;
 6. enable (FC), disable (FD), save zero (FE), clear error (FB) on the command id act and are answered with ONE state
    reply each, sampled at reception after the frame has acted (the status nibble is the new status; after a save
    zero the position reads against the new zero; the torque field is the drive's torque of the last step, as in every
    reply sampled at reception), also when the frame is refused (the unchanged state). Enable is refused while a fault
    is latched; otherwise it clears the command and, from not enabled, sets the time-out counter to zero. Save zero
    only while not enabled (also while a fault is latched). Clear returns a latched fault whose cause is gone to status
    0. These frames do not take the control slot and do not reset the counter of an enabled module;
 7. 0x7FF: refresh -> a state reply; read -> answered with the value (live registers: p_m = theta x Gr on the motor
    side, xout = theta; a RID the unit does not carry: the request echoed, value bytes as sent); write -> answered with
    the new value if the write policy admits it, else with the unchanged value (events register_write_accepted /
    register_write_refused); save -> accepted only while not enabled, answered [id, AA, 01, 0...] (the maker's
    simulator), changes nothing within an episode; while enabled no reply.
Replies are returned in the order of the frames that caused them. Under admit='control_only' only control frames are
answered (rule 2).
Time-out: with TIMEOUT = c counts of 50 us, a module leaves enable with status D at the first step at which
ceil(c / 20) steps have passed without a control frame; c = 0: off. The counter counts the physics steps the module
runs ENABLED: it holds while the module is not enabled, an enable from not enabled sets it to zero, and while enabled
only control frames reset it. truth() reports it as steps_since_control.

THE PROTOCOLS of the two hidden plug-ins (the unit passes objects; this module defines no implementation):
    heat.step(t_w, t_h, t_del, t_amb) -> (t_w', t_h')      one explicit step of both nodes of every module
    heat.latch(latched, t_w, ot_value) -> (latched', newly) the coil protection at a step's start
    heat.clear_allowed(t_w, ot_value) -> bool array         where FB may clear a coil fault (T_w < OT - hysteresis)
    noise.offsets(stream, bus, start, count, res) -> (count x 3) array of offsets added to (position, velocity,
        torque) of `count` replies on `bus`, numbered start .. start + count - 1 in that bus's `stream` ('reply' or
        'tap'); res = (count x 3) one code step of each field of each reply's module
(the thermal law and its random-number module are what the hidden implementations call; nothing of them
is imported here).

Arithmetic: float64; +, -, x, /, min, max, floor, abs, comparisons and numpy's element-wise ufuncs of these on
whole arrays; no transcendental, no BLAS, no reduction, no Python loop over modules on the per-step path. MuJoCo calls
(`mj_mulJacTVec` for the stop torque of a zero_saved event) happen at reception only.
"""
import math
import operator
import struct
from typing import NamedTuple

import mujoco
import numpy as np

from . import hostcodec as hc
from . import registers as rg
from .envelope import ModuleEnvelopes, kt_from_registers, t_lim_from_registers
from .frames import BUSES, DATA_LEN, ID_MAX, Frame
from .state import public_params

BUS_FORMAT = "openarm_unit/BusLayer/1"

# Status nibble (maker dm_motor_constants.hpp:50-60; params_public.json frames.status_codes, checked below)
ST_DISABLED, ST_ENABLED, ST_UNDER_VOLTAGE, ST_COIL, ST_COMM_LOST = 0x0, 0x1, 0x9, 0xC, 0xD
_STATUS_NAMES = {"disabled": ST_DISABLED, "enabled": ST_ENABLED, "under voltage": ST_UNDER_VOLTAGE,
                 "coil over temperature": ST_COIL, "communication lost": ST_COMM_LOST}


def _check_constants(p=None):
    """The status codes above against params_public.json frames.status_codes (the vendor's, printed there with
    their source); BusError on any difference. p: a parsed public parameter file (default: the package's)."""
    p = public_params() if p is None else p
    got = {str(name): int(str(code), 16) for code, name in p["frames"]["status_codes"]["value"].items()}
    if got != _STATUS_NAMES:
        raise BusError(f"the bus layer's status codes differ from params_public.json: {got} != {_STATUS_NAMES}")

# The force-position loop gains' names, in the order of the module text (KP_APR in 1/s, KP_ASR in N m s/rad, KI_ASR in
# N m/rad: the drive's own units at the output shaft; their values are params_public.json gripper.loop_gains).
FP_GAIN_NAMES = ("KP_APR", "KP_ASR", "KI_ASR")

ADMIT = ("all", "control_only")


class BusError(ValueError):
    """An argument the bus layer does not accept."""


_check_constants()


class ModuleSpec(NamedTuple):
    """One joint module on the model: its bus (0 left, 1 right), its MJCF hinge, its actuator, its edition key."""
    bus: int
    joint: str
    actuator: str
    edition: str


def pair_modules(editions=None):
    """The sixteen modules of the pair in module order (params_public.json conventions.modules), with the nominal
    edition of each class unless `editions` maps a class to an edition."""
    p = public_params()
    eds = dict(p["class_nominal_edition"]["value"])
    eds.update(dict(editions or {}))
    return tuple(ModuleSpec(m["bus"], m["mjcf_joint"], m["actuator"], eds[m["class"]])
                 for m in p["conventions"]["modules"]["value"])


def fp_gains_public(p=None):
    """The force-position loop gains of params_public.json gripper.loop_gains, with the label of their source
    ('params_public'). There is no fallback: a null entry, a missing gain or a gain that is not a finite positive
    number raises BusError."""
    p = public_params() if p is None else p
    v = p["gripper"]["loop_gains"]["value"]
    if not isinstance(v, dict):
        raise BusError("params_public.json gripper.loop_gains holds no gains: the bus layer has no other source")
    return _checked_fp_gains(v, "params_public.json gripper.loop_gains"), "params_public"


def _checked_fp_gains(v, what):
    out = {}
    for k in FP_GAIN_NAMES:
        x = v.get(k) if isinstance(v, dict) else None
        if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(float(x)) or float(x) <= 0.0:
            raise BusError(f"{what}: {k} must be a finite positive number, got {x!r}")
        out[k] = float(x)
    return out


def _public_reply_temperature():
    return float(public_params()["model_run"]["reply_temperature_C"]["value"])


def _timeout_steps_per_count_ratio(dt):
    """Steps per TIMEOUT count ratio: one step of dt over the vendor's 50 us time base = 20 at 1 ms."""
    cs = float(public_params()["frames"]["timeout_count_s"]["value"])
    r = int(round(dt / cs))
    if r < 1 or abs(r * cs - dt) > 1e-12:
        raise BusError("the physics step is not a whole number of TIMEOUT counts")
    return r


_SPECIAL_PREFIX = b"\xff" * 7
_SPECIAL = {hc.CMD_ENABLE: "enable", hc.CMD_DISABLE: "disable", hc.CMD_SAVE_ZERO: "save_zero", hc.CMD_CLEAR: "clear"}
_PARAM_OPS = {hc.OP_READ: "read", hc.OP_WRITE: "write", hc.OP_SAVE: "save", hc.OP_REFRESH: "refresh"}
_POW16 = float((1 << 16) - 1)
_POW12 = float((1 << 12) - 1)
_LUT_IDS = ID_MAX + 1
_FRAME_ONLY = {Frame}
_BUS_AND_ID = operator.attrgetter("bus", "can_id")
_DATA_OF = operator.attrgetter("data")
_LEN_8 = {DATA_LEN}
_MIT_SHIFT = np.array([48, 36, 24, 12, 0], dtype=np.uint64)
_MIT_MASK = np.array([0xFFFF, 0xFFF, 0xFFF, 0xFFF, 0xFFF], dtype=np.uint64)
_REPLY_SHIFT = np.array([40, 28, 16], dtype=np.uint64)
_SH56, _SH40, _SH28, _SH16 = (np.uint64(v) for v in (56, 40, 28, 16))
_U257 = np.uint64(257)
_FAST_CACHE_MAX = 64


_MISS = object()


class _FastRoute(NamedTuple):
    """What the fast path of reception knows about one sequence of (bus, id) pairs (see BusLayer._fast_route)."""
    idx: np.ndarray          # module of each frame, in the order sent
    idx_l: list
    nb: tuple                # frames on bus 0, on bus 1
    mit_mask: np.ndarray     # per frame: an MIT frame
    mit_pos: np.ndarray      # positions of the MIT frames
    mit_idx: np.ndarray      # their modules
    mit_span: np.ndarray     # their modules' decoding spans and lower ends (5 fields)
    mit_lo: np.ndarray
    has_mit: bool
    fp_pos: list             # positions of the force-position frames
    ones: np.ndarray         # status 1 for every frame
    enc: tuple               # the reply encoder's per-module constants in frame order
    bus_l: list              # reply bus and reply id per frame
    rid_l: list
    d0w: np.ndarray          # the replies' first byte at status 1 (all enabled), shifted to its place (D0 << 56)


def _same_bits(a, b):
    """True when two finite floats are the same float64 (also in the sign of a zero)."""
    return a == b and (a != 0.0 or math.copysign(1.0, a) == math.copysign(1.0, b))


class BusLayer:
    """The bus layer of N modules on (model, data). See the module text for the law, the contract and the plug-ins.

    model, data: a compiled MjModel and its MjData. modules: a sequence of ModuleSpec. registers: a RegisterSet (N = 16)
    or a sequence of N mappings RID -> value. delta, sign, imax: per module (the stored zero in force, s_i, the current
    base Imax, C7). k, b: per module; None = read from the model's actuators (gainprm[0], biasprm[0]: the file's
    statement).
    heat, noise: the hidden plug-ins or None. t_amb: the session's ambient (with heat). temperatures: per module
    (T_w, T_h) at the start (with heat; default: the ambient on both nodes). policy: a registers.WritePolicy (default:
    admits nothing). fp_gains: {'KP_APR', 'KP_ASR', 'KI_ASR'} (default: fp_gains_public()). supply_V: the supply
    (default: params_public.json supply_V)."""

    def __init__(self, model, data, modules, registers, delta=None, sign=None, imax=None, k=None, b=None, heat=None,
                 noise=None, t_amb=None, temperatures=None, policy=None, fp_gains=None, supply_V=None):
        self.model, self.data = model, data
        self.modules = tuple(ModuleSpec(*m) for m in modules)
        n = len(self.modules)
        if n == 0:
            raise BusError("at least one module")
        self.n = n
        p = public_params()
        self.dt = float(model.opt.timestep)
        self._to_ratio = _timeout_steps_per_count_ratio(self.dt)
        self.budget_per_step = int(p["frames"]["budget_frames_per_bus_per_step"]["value"])
        self.regs = rg.RegisterTable(registers)
        if self.regs.n != n:
            raise BusError(f"{self.regs.n} register tables for {n} modules")
        arr = lambda v, d: np.array([float(x) for x in (v if v is not None else [d] * n)], dtype=np.float64)
        self.delta = arr(delta, 0.0)
        self.sign = arr(sign, 1.0)
        if any(s not in (1.0, -1.0) for s in self.sign):
            raise BusError("a sign is +1 or -1")
        if imax is None:
            imax = [p["editions"][m.edition]["Imax_A"]["value"] for m in self.modules]
        self.imax = arr(imax, 0.0)
        self.editions = tuple(m.edition for m in self.modules)
        # model addresses
        jid, aid = [], []
        for m in self.modules:
            j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, m.joint)
            a = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, m.actuator)
            if j < 0 or a < 0:
                raise BusError(f"the model lacks joint {m.joint!r} or actuator {m.actuator!r}")
            if model.jnt_type[j] != mujoco.mjtJoint.mjJNT_HINGE:
                raise BusError(f"{m.joint} is not a hinge")
            if model.actuator_trntype[a] != mujoco.mjtTrn.mjTRN_JOINT or model.actuator_trnid[a, 0] != j:
                raise BusError(f"{m.actuator} does not drive {m.joint} through a joint transmission")
            if model.actuator_gear[a, 0] != 1.0:
                raise BusError(f"{m.actuator}: gear must be 1")
            jid.append(j)
            aid.append(a)
        self.jid = np.array(jid, dtype=np.int64)
        self.act = np.array(aid, dtype=np.int64)
        self.qadr = np.array([model.jnt_qposadr[j] for j in jid], dtype=np.int64)
        self.dadr = np.array([model.jnt_dofadr[j] for j in jid], dtype=np.int64)
        if len(set(aid)) != n or len(set(jid)) != n:
            raise BusError("one actuator and one hinge per module")
        self.k = arr(k, 0.0) if k is not None else np.array([model.actuator_gainprm[a, 0] for a in aid], np.float64)
        self.b = arr(b, 0.0) if b is not None else np.array([model.actuator_biasprm[a, 0] for a in aid], np.float64)
        if not np.all(self.k > 0.0):
            raise BusError("k must be positive")
        self.arm0 = np.array(model.dof_armature[self.dadr], dtype=np.float64)
        self.t_peak = np.array([float(p["editions"][e]["T_peak_Nm"]["value"]) for e in self.editions], np.float64)
        self.spos = self.sign > 0.0
        self.sk = self.sign * self.k
        g, self.fp_gains_source = (fp_gains_public(p) if fp_gains is None else (dict(fp_gains), "given"))
        self.fp_kp_apr = np.full(n, float(g["KP_APR"]))
        self.fp_kp_asr = np.full(n, float(g["KP_ASR"]))
        self.fp_ki_asr = np.full(n, float(g["KI_ASR"]))
        # plug-ins
        self.heat, self.noise = heat, noise
        self.t_amb = float(t_amb) if t_amb is not None else (None if heat is None else 25.0)
        if heat is None:
            t0 = _public_reply_temperature()
            self.T_w = np.full(n, t0)
            self.T_h = np.full(n, t0)
        else:
            if temperatures is None:
                temperatures = [(self.t_amb, self.t_amb)] * n
            if len(temperatures) != n:
                raise BusError("one (T_w, T_h) pair per module")
            self.T_w = np.array([float(t[0]) for t in temperatures], np.float64)
            self.T_h = np.array([float(t[1]) for t in temperatures], np.float64)
        self.policy = policy if policy is not None else rg.WritePolicy(())
        self.supply_V = float(p["supply_V"]["value"] if supply_V is None else supply_V)
        # run-time state
        z = lambda: np.zeros(n, dtype=np.float64)
        self.enabled = np.zeros(n, dtype=bool)
        self.fault = np.zeros(n, dtype=np.int64)
        self.cmd = np.zeros((n, 5), dtype=np.float64)          # p_des, v_des, kp, kd, t_ff (views below)
        self.p_des, self.v_des, self.kp, self.kd, self.t_ff = (self.cmd[:, j] for j in range(5))
        self.fp_p, self.fp_vlim, self.fp_ilim, self.Z = z(), z(), z(), z()
        self.steps_since = np.zeros(n, dtype=np.int64)
        self.T_ref = z()
        self.T_del = z()
        self.step_count = 0
        self.frames_in = [0 for _ in BUSES]
        self.drops = {}                           # (bus, reason) -> count
        self.noise_counters = {}                  # (stream, bus) -> next index
        self._events = []
        # None, or the stop torque of every module that a save zero reports in place of the constraint arrays'
        # value: set by a unit right after it restored a snapshot (the arrays then belong to the restored state, not
        # to the last solved step), cleared by the unit at its next step or when it sets a state
        self.restored_stop_torque = None
        self.all_pos = bool(np.all(self.spos))
        self._bus_of = [m.bus for m in self.modules]
        self._bus_arr = np.array(self._bus_of, dtype=np.int64)
        self._dirty = self._dirty_status = True
        self.neg_t_peak = -self.t_peak
        self._zeros = np.zeros(n, dtype=np.float64)
        # Reception takes the one-pass path for the common exchange (receive); False sends every exchange frame by
        # frame (the same results; a test compares the two).
        self.fast_path = True
        self._act_ix = _as_index(self.act)
        self._refresh_register_caches()
        self._attach()
        self._rebuild()

    # ------------------------------------------------------------------------------------------------ attachment
    def _attach(self):
        """The carriers' fixed fields: forcelimited on, ctrl limit off, gain and bias zero (all modules start
        disabled), ctrl, force range and b zero, the inertia entries at the model's values."""
        m, a = self.model, self.act
        m.actuator_forcelimited[a] = True
        m.actuator_ctrllimited[a] = False
        m.actuator_gainprm[a, :] = 0.0
        m.actuator_biasprm[a, :] = 0.0
        m.actuator_forcerange[a, :] = 0.0
        self.data.ctrl[a] = 0.0
        self.data.qfrc_applied[self.dadr] = 0.0
        m.dof_armature[self.dadr] = self.arm0
        self._views = _Views(self.model, self.data, a)

    def _refresh_register_caches(self):
        """Arrays the per-step path reads from the registers: mode, TIMEOUT in steps, the mapping, KT, T_lim and the
        envelope, OT and UV thresholds; the routing of ids."""
        n, R = self.n, self.regs
        self.mode = np.array([int(R.get(i, rg.CTRL_MODE, rg.MODE_MIT)) for i in range(n)], dtype=np.int64)
        c = [int(R.get(i, rg.TIMEOUT, 0) or 0) for i in range(n)]
        r = self._to_ratio
        self.timeout_steps = np.array([(ci + r - 1) // r for ci in c], dtype=np.int64)
        self.pmax = np.array([float(R.get(i, rg.PMAX)) for i in range(n)], np.float64)
        self.vmax = np.array([float(R.get(i, rg.VMAX)) for i in range(n)], np.float64)
        self.tmax = np.array([float(R.get(i, rg.TMAX)) for i in range(n)], np.float64)
        mods = [R.module(i) for i in range(n)]
        self.kt = np.array([kt_from_registers(mm) for mm in mods], np.float64)
        self.t_lim = np.array([t_lim_from_registers(mm, im) for mm, im in zip(mods, self.imax)], np.float64)
        self.i_full = self.imax * self.kt
        self.env = ModuleEnvelopes(self.editions, self.t_lim)
        self.a1, self.b1, self.a2, self.b2 = self.env.a1, self.env.b1, self.env.a2, self.env.b2
        self._a12 = np.stack([self.a1, self.a2])                # both lines of E in one array operation (pre_step)
        self._b12 = np.stack([self.b1, self.b2])
        self.ot = np.array([float(R.get(i, rg.OT_VALUE)) for i in range(n)], np.float64)
        self.uv = np.array([float(R.get(i, rg.UV_VALUE)) for i in range(n)], np.float64)
        self.gr = np.array([float(R.get(i, rg.GR)) for i in range(n)], np.float64)
        self._uv_hit = self.supply_V < self.uv
        self._uv_any = bool(self._uv_hit.any())
        self._dirty = self._dirty_status = True
        pm, vm, tm = self.pmax, self.vmax, self.tmax
        lev = np.array([_POW16, _POW12, _POW12])
        self._x3 = np.stack([pm, vm, tm], axis=1)                    # reply fields' X_MAX per module
        self._x3_2 = self._x3 + self._x3                             # X - (-X), exact
        self._lev3 = lev
        self._span5 = np.stack([pm + pm, vm + vm, np.full(n, hc.KP_RANGE[1] - hc.KP_RANGE[0]),
                                np.full(n, hc.KD_RANGE[1] - hc.KD_RANGE[0]), tm + tm], axis=1)
        self._lo5 = np.stack([-pm, -vm, np.full(n, hc.KP_RANGE[0]), np.full(n, hc.KD_RANGE[0]), -tm], axis=1)
        self._pow5 = np.array([_POW16, _POW12, _POW12, _POW12, _POW12])
        self._res = self.resolutions()                                # the reply noise's code steps per module
        self._idnib = np.array([int(R.get(i, rg.ESC_ID)) & 0xF for i in range(n)], dtype=np.int64)
        self.is_fp = self.mode == rg.MODE_POS_FORCE
        route, param_route = {}, {}
        for i, mspec in enumerate(self.modules):
            cid = int(R.get(i, rg.ESC_ID))
            bus = mspec.bus
            for off, kind in ((0, "mit"), (hc.POSFORCE_OFFSET, "posforce"), (hc.POSVEL_OFFSET, "unsupported"),
                              (hc.VEL_OFFSET, "unsupported")):
                key = (bus, cid + off)
                if key in route:
                    raise BusError(f"two modules answer id {cid + off:#x} on bus {bus}")
                route[key] = (kind, i)
            param_route[(bus, cid)] = i
        self.reply_id = [int(R.get(i, rg.MST_ID)) for i in range(self.n)]
        self.esc_id = [int(R.get(i, rg.ESC_ID)) for i in range(self.n)]
        self._route, self._param_route = route, param_route
        # The fast path's routing (receive): a lookup array over (bus, id) whose entry is 2 i for an MIT frame to a
        # module in MIT mode and 2 i + 1 for a force-position frame to a module in force-position mode; -1 for every
        # other id (those frames take the frame-by-frame path). Rebuilt with the registers; the per-exchange cache of
        # routed id sequences is cleared with it.
        lut = np.full(len(BUSES) * _LUT_IDS, -1, dtype=np.int64)
        for (bus, cid), (kind, i) in route.items():
            if 0 <= cid <= ID_MAX:
                if kind == "mit" and self.mode[i] == rg.MODE_MIT:
                    lut[bus * _LUT_IDS + cid] = 2 * i
                elif kind == "posforce" and self.mode[i] == rg.MODE_POS_FORCE:
                    lut[bus * _LUT_IDS + cid] = 2 * i + 1
        self._lut = lut
        self._fast_cache = {}

    # ------------------------------------------------------------------------------------------------- utilities
    def _event(self, kind, **fields):
        e = {"kind": kind, "step": self.step_count}
        e.update(fields)
        self._events.append(e)

    def drain_events(self):
        """The bus layer's events since the last call (of the kinds fault_set, fault_cleared, enabled, disabled,
        mode_changed, timeout, zero_saved, register_write_accepted, register_write_refused, frame_dropped), oldest
        first, each a dict with 'kind' and 'step' (the index of the next physics step when it happened)."""
        e, self._events = self._events, []
        return e

    def _drop(self, bus, reason, frame):
        self.drops[(bus, reason)] = self.drops.get((bus, reason), 0) + 1
        self._event("frame_dropped", reason=reason, bus=bus, can_id=getattr(frame, "can_id", None))

    def _status_of(self, i):
        f = int(self.fault[i])
        return f if f != 0 else (ST_ENABLED if self.enabled[i] else ST_DISABLED)

    def status(self):
        """The status nibble of every module: the latched fault's code, else 1 enabled / 0 disabled."""
        return np.where(self.fault != 0, self.fault, self.enabled.astype(np.int64))   # ST_ENABLED 1, ST_DISABLED 0

    def limits(self, i):
        """(PMAX, VMAX, TMAX) of module i's own registers."""
        return float(self.pmax[i]), float(self.vmax[i]), float(self.tmax[i])

    def resolutions(self):
        """(N x 3) one code step of position, velocity and torque of every module's replies."""
        return np.stack([2.0 * self.pmax / _POW16, 2.0 * self.vmax / _POW12, 2.0 * self.tmax / _POW12], axis=1)

    def theta(self):
        """Module angles theta = s (q - delta) at the present state (true, noise-free)."""
        return self.sign * (self.data.qpos[self.qadr] - self.delta)

    def delivered(self):
        """The torque each joint receives from its module: actuator force of the last step + b while enabled."""
        return self.data.actuator_force[self.act] + np.where(self.enabled, self.b, 0.0)

    # -------------------------------------------------------------------------------------------- classification
    def classify(self, f):
        """(kind, module, detail) of one frame without acting on it. kind: 'malformed', 'unknown_id', 'unsupported',
        'mit', 'posforce', 'special' (detail: enable / disable / save_zero / clear), 'param' (detail: read / write /
        save / refresh)."""
        if not isinstance(f, Frame) or f.bus not in BUSES or not (0 <= f.can_id <= ID_MAX) or len(f.data) != DATA_LEN:
            return "malformed", None, None
        d = f.data
        if f.can_id == hc.PARAM_ID:
            i = self._param_route.get((f.bus, d[0] | (d[1] << 8)))
            if i is None:
                return "unknown_id", None, None
            op = _PARAM_OPS.get(d[2])
            if op is None:
                return "unsupported", i, None
            return "param", i, op
        r = self._route.get((f.bus, f.can_id))
        if r is None:
            return "unknown_id", None, None
        kind, i = r
        if kind == "mit" and d[:7] == _SPECIAL_PREFIX and d[7] in _SPECIAL:
            return "special", i, _SPECIAL[d[7]]
        if kind == "posforce":
            p = struct.unpack("<f", d[0:4])[0]
            if not math.isfinite(p):
                return "malformed", i, None      # a non-finite position cannot be executed
        return kind, i, None

    # ------------------------------------------------------------------------------------------------- reception
    def receive(self, frames, steps=4, admit="all"):
        """The frames of one exchange of `steps` physics steps, in the order sent; returns the replies
        (Frame list) in the order of the frames that caused them, sampled now, before any new command acts."""
        if admit not in ADMIT:
            raise BusError(f"admit is one of {ADMIT}")
        steps = int(steps)
        if not 1 <= steps <= 50:
            raise BusError("an exchange has 1 to 50 steps")
        cap = self.budget_per_step * steps
        if self.fast_path and type(frames) in (list, tuple) and frames:
            replies = self._receive_fast(frames, cap)
            if replies is not None:
                return replies
        used = {b: 0 for b in BUSES}
        slot = np.zeros(self.n, dtype=bool)
        pend_mit, pend_fp = {}, {}
        out = []                                  # ('state', module, status, delta) or ('raw', Frame)
        route, mode, enabled = self._route, self.mode, self.enabled
        control_only = admit == "control_only"
        for f in (frames or ()):
            # 1. malformed; the fast path classifies a well-formed control frame inline (classify() is the reference)
            r = None
            if type(f) is Frame and len(f.data) == DATA_LEN and f.bus in used and f.can_id != hc.PARAM_ID:
                r = route.get((f.bus, f.can_id))
            if r is not None and r[0] == "mit" and not (f.data[:7] == _SPECIAL_PREFIX and f.data[7] in _SPECIAL):
                kind, i, detail = "mit", r[1], None
            else:
                kind, i, detail = self.classify(f)
            bus = f.bus if isinstance(f, Frame) else -1
            if kind == "malformed":
                self._drop(bus, "malformed", f)
                continue
            # 2. admission
            if control_only and kind != "mit" and kind != "posforce":
                self._drop(bus, "not_admitted", f)
                continue
            # 3. the bus budget
            if used[bus] >= cap:
                self._drop(bus, "budget", f)
                continue
            used[bus] += 1
            self.frames_in[bus] += 1
            # 4. unknown or not implemented
            if kind == "unknown_id" or kind == "unsupported":
                self._drop(bus, kind, f)
                continue
            # 5. control frames
            if kind == "mit" or kind == "posforce":
                if slot[i]:
                    self._drop(bus, "duplicate", f)
                    continue
                slot[i] = True          # the slot is taken by a control frame of either kind
                if mode[i] != (rg.MODE_MIT if kind == "mit" else rg.MODE_POS_FORCE):
                    self._drop(bus, "wrong_mode", f)          # ignored without a reply, in both directions
                    continue
                if enabled[i]:
                    out.append(("state", i, ST_ENABLED, self.delta[i]))
                    (pend_mit if kind == "mit" else pend_fp)[i] = f.data
                    self.steps_since[i] = 0
                else:
                    out.append(("state", i, self._status_of(i), self.delta[i]))
                continue
            # 6. enable, disable, save zero, clear: they act, then one state reply each (also when refused)
            if kind == "special":
                self._special(i, detail, pend_mit, pend_fp)
                out.append(("state", i, self._status_of(i), self.delta[i]))
                continue
            # 7. 0x7FF
            if detail == "refresh":
                out.append(("state", i, self._status_of(i), self.delta[i]))
            else:
                rr = self._param(i, detail, f)
                if rr is not None:
                    out.append(("raw", rr))
                route, mode = self._route, self.mode         # a register write may have changed the mode or the ids
        self._apply_pending(pend_mit, pend_fp)
        replies = self._emit(out)
        if self._dirty:
            self._rebuild()
        return replies

    def _fast_route(self, key):
        """The routing of one sequence of (bus, id) pairs for the fast path, or None when the sequence cannot take it
        (a bus or id out of range, an id that is not a control frame of the module's own mode, two frames to one
        module). Depends only on the registers (the cache is cleared with them)."""
        n = self.n
        idx, fp = [], []
        nb0 = 0
        for b, c in key:
            if b not in BUSES or not 0 <= c <= ID_MAX:
                return None
            r = int(self._lut[b * _LUT_IDS + c])
            if r < 0:
                return None
            idx.append(r >> 1)
            fp.append(bool(r & 1))
            if b == 0:
                nb0 += 1
        if len(set(idx)) != len(idx):
            return None
        idx_a = np.array(idx, dtype=np.int64)
        fp_a = np.array(fp, dtype=bool)
        mit_pos = np.nonzero(~fp_a)[0]
        mit_idx = idx_a[mit_pos]
        return _FastRoute(
            idx=idx_a, idx_l=idx, nb=(nb0, len(idx) - nb0), mit_mask=~fp_a, mit_pos=mit_pos, mit_idx=mit_idx,
            mit_span=self._span5[mit_idx], mit_lo=self._lo5[mit_idx], has_mit=bool(mit_pos.size),
            fp_pos=[j for j in range(len(idx)) if fp[j]], ones=np.full(len(idx), ST_ENABLED, dtype=np.int64),
            enc=self._enc_consts(idx_a), bus_l=[self._bus_of[i] for i in idx], rid_l=[self.reply_id[i] for i in idx],
            d0w=(((np.full(len(idx), ST_ENABLED, dtype=np.int64) & 0xF) << 4) | self._idnib[idx_a]).astype(np.uint64)
            << _SH56)

    def _receive_fast(self, frames, cap):
        """Reception of the common exchange in one pass (the same results as the frame-by-frame path below): every
        frame a well-formed control frame of its module's own mode, at most one per module, inside the budget.
        Returns the replies, or None (nothing done) when the exchange does not qualify. A command whose decoded values
        equal the ones in force, bit for bit, leaves the per-step coefficients as they are (they would be rebuilt to
        the same values)."""
        if set(map(type, frames)) != _FRAME_ONLY:
            return None
        key = tuple(map(_BUS_AND_ID, frames))
        cache = self._fast_cache
        ent = cache.get(key, _MISS)
        if ent is _MISS:
            ent = self._fast_route(key)
            if len(cache) >= _FAST_CACHE_MAX:
                cache.clear()
            cache[key] = ent
        if ent is None or ent.nb[0] > cap or ent.nb[1] > cap:
            return None
        datas = list(map(_DATA_OF, frames))
        if set(map(len, datas)) != _LEN_8:
            return None
        words = np.frombuffer(b"".join(datas), dtype=">u8")       # each frame's eight bytes as one big-endian word
        if ent.has_mit and (((words >> 56) == 0xFF) & ent.mit_mask).any():
            return None                         # possibly enable / disable / save zero / clear: frame by frame
        fpv = []
        for j in ent.fp_pos:
            p, v, cur = hc.unpack_posforce(datas[j])
            if not math.isfinite(p):
                return None                     # malformed: frame by frame
            fpv.append((ent.idx_l[j], p, v, cur))
        # --- from here on the exchange is accepted as a whole
        idx = ent.idx
        en = self.enabled[idx]
        all_en = bool(en.all())
        if all_en:
            status = ent.ones
            self.steps_since[idx] = 0
        else:
            fl = self.fault[idx]
            status = np.where(fl != 0, fl, np.where(en, ST_ENABLED, ST_DISABLED))
            self.steps_since[idx[en]] = 0
        deltas = self.delta[idx]
        pending = False
        self.frames_in[0] += ent.nb[0]
        self.frames_in[1] += ent.nb[1]
        if ent.has_mit:
            if all_en:
                pos, im, span, lo = ent.mit_pos, ent.mit_idx, ent.mit_span, ent.mit_lo
            else:
                sel = en[ent.mit_pos]
                pos, im, span, lo = ent.mit_pos[sel], ent.mit_idx[sel], ent.mit_span[sel], ent.mit_lo[sel]
            if im.size:
                pending = True
                # the five MIT fields (q 16 bit, dq, kp, kd, t_ff 12 bit each: the layout of _apply_pending)
                u = ((words[pos][:, None] >> _MIT_SHIFT) & _MIT_MASK).astype(np.float64)
                new = (u / self._pow5) * span + lo
                if not np.array_equal(new.view(np.int64), self.cmd[im].view(np.int64)):
                    self.cmd[im] = new
                    self._dirty = True
        if fpv:
            enabled = self.enabled
            for i, p, v, cur in fpv:
                if not enabled[i]:
                    continue
                pending = True
                pm = float(self.pmax[i])
                p = min(max(float(p), -pm), pm)
                if not (_same_bits(p, float(self.fp_p[i])) and _same_bits(v, float(self.fp_vlim[i]))
                        and _same_bits(cur, float(self.fp_ilim[i]))):
                    self.fp_p[i] = p
                    self.fp_vlim[i] = v
                    self.fp_ilim[i] = cur
                    self._fp_command(i)
        data = self._encode_states(idx, status, deltas, "reply", ent.enc, ent.d0w if all_en else None)
        replies = [_mk_frame(b, rid, row) for b, rid, row in zip(ent.bus_l, ent.rid_l, data)]
        if self._dirty:
            self._rebuild()
        elif pending and self._fp:
            self._rebuild_fp_entries()
        return replies

    def _rebuild_fp_entries(self):
        """What _rebuild writes for the enabled force-position modules (their per-step entries, which pre_step sets
        again at every step): c = 0, skp = s x 0, skc = s k x 0, the bias -k x 0. Used when an exchange's commands
        equal the ones in force, so that the state after reception is the frame-by-frame path's, bit for bit."""
        bp = self.model.actuator_biasprm
        for i in self._fp_idx:
            self.c[i] = 0.0
            self.skp[i] = self.sign[i] * 0.0
            self.skc[i] = self.sk[i] * 0.0
            bp[self.act[i], 1] = -self.kon[i] * 0.0

    def _special(self, i, what, pend_mit, pend_fp):
        if what == "enable":
            if self.fault[i] != 0:
                return                                           # refused while a fault is latched
            was = bool(self.enabled[i])
            self.enabled[i] = True
            self._clear_command(i)
            pend_mit.pop(i, None)
            pend_fp.pop(i, None)
            if not was:
                self.steps_since[i] = 0                          # the time-out counter starts at the enable
                self._event("enabled", module=i)
        elif what == "disable":
            was = bool(self.enabled[i])
            self.enabled[i] = False
            self._dirty = self._dirty_status = True
            if was:
                self._event("disabled", module=i)
        elif what == "save_zero":
            if self.enabled[i]:
                return                                           # ignored while enabled
            q = float(self.data.qpos[self.qadr[i]])
            self.delta[i] = q
            self._event("zero_saved", module=i, q=q, stop_torque=self._stop_torque(i))
            self._dirty = self._dirty_status = True
        elif what == "clear":
            code = int(self.fault[i])
            if code == 0:
                return
            if code == ST_COIL:
                ok = self.heat is not None and bool(np.asarray(self.heat.clear_allowed(self.T_w, self.ot))[i])
            elif code == ST_UNDER_VOLTAGE:
                ok = self.supply_V >= self.uv[i]
            else:
                ok = True                                        # communication lost: its cause is gone
            if ok:
                self.fault[i] = 0
                self.enabled[i] = False
                self._dirty = self._dirty_status = True
                self._event("fault_cleared", module=i, code=code)

    def _stop_torque(self, i):
        """The constraint torque of joint i's limit rows at the last solved step (for the zero_saved event); the
        value handed in by `restored_stop_torque` while that is set."""
        r = self.restored_stop_torque
        if r is not None:
            return float(r[i])
        m, d = self.model, self.data
        if d.nefc == 0:
            return 0.0
        mask = (d.efc_type[:d.nefc] == mujoco.mjtConstraint.mjCNSTR_LIMIT_JOINT) & (d.efc_id[:d.nefc] == self.jid[i])
        if not mask.any():
            return 0.0
        vec = np.where(mask, d.efc_force[:d.nefc], 0.0)
        res = np.zeros(m.nv)
        mujoco.mj_mulJacTVec(m, d, res, vec)
        return float(res[self.dadr[i]])

    def _clear_command(self, i):
        """Enable clears the command: kp = kd = t_ff = 0 (and the rest of it, the force-position limits and
        integral included: the maintainers' own choice)."""
        for a in (self.p_des, self.v_des, self.kp, self.kd, self.t_ff, self.fp_p, self.fp_vlim, self.fp_ilim, self.Z):
            a[i] = 0.0
        self._dirty = self._dirty_status = True

    def _apply_pending(self, pend_mit, pend_fp):
        """Decode and store the accepted control frames (vectorised over the modules; the module's own mapping)."""
        if pend_mit:
            idx = np.array(sorted(pend_mit), dtype=np.int64)
            r = np.frombuffer(b"".join(pend_mit[i] for i in idx.tolist()), dtype=np.uint8).reshape(-1, 8).astype(np.int64)
            u = np.empty((idx.shape[0], 5), dtype=np.float64)
            u[:, 0] = (r[:, 0] << 8) | r[:, 1]
            u[:, 1] = (r[:, 2] << 4) | (r[:, 3] >> 4)
            u[:, 2] = ((r[:, 3] & 0xF) << 8) | r[:, 4]
            u[:, 3] = (r[:, 5] << 4) | (r[:, 6] >> 4)
            u[:, 4] = ((r[:, 6] & 0xF) << 8) | r[:, 7]
            # x = (u / (2^bits - 1)) x (hi - lo) + lo with the module's own ranges (hostcodec.from_uint)
            self.cmd[idx] = (u / self._pow5) * self._span5[idx] + self._lo5[idx]
            self._dirty = True
        for i, data in pend_fp.items():
            p, v, cur = hc.unpack_posforce(data)
            pm = float(self.pmax[i])
            self.fp_p[i] = min(max(float(p), -pm), pm)          # a position beyond PMAX is clipped to +-PMAX
            self.fp_vlim[i] = v
            self.fp_ilim[i] = cur
            self._fp_command(i)

    # --------------------------------------------------------------------------------------------------- replies
    def _emit(self, out):
        """Encode the replies: state replies vectorised (with noise from the 'reply' stream), raw ones as built."""
        st = [(j, o) for j, o in enumerate(out) if o[0] == "state"]
        frames = [None] * len(out)
        if st:
            mods = np.array([o[1] for _, o in st], dtype=np.int64)
            status = np.array([o[2] for _, o in st], dtype=np.int64)
            deltas = np.array([o[3] for _, o in st], dtype=np.float64)
            data = self._encode_states(mods, status, deltas, "reply")
            for (j, _), i, row in zip(st, mods.tolist(), data):
                frames[j] = _mk_frame(self._bus_of[i], self.reply_id[i], row)
        for j, o in enumerate(out):
            if o[0] == "raw":
                frames[j] = o[1]
        return frames

    def _noise_offsets(self, stream, mods):
        """(len(mods) x 3) offsets from the noise plug-in, numbered per (stream, bus) in the order given."""
        off = np.zeros((len(mods), 3), dtype=np.float64)
        if self.noise is None:
            return off
        res = self._res[mods]                            # resolutions(), kept with the registers
        buses = self._bus_arr[mods]
        for b in BUSES:
            sel = np.nonzero(buses == b)[0]
            if sel.size == 0:
                continue
            start = self.noise_counters.get((stream, b), 0)
            o = np.asarray(self.noise.offsets(stream, b, start, int(sel.size), res[sel]), dtype=np.float64)
            if o.shape != (sel.size, 3):
                raise BusError("the noise plug-in returned a wrong shape")
            off[sel] = o
            self.noise_counters[(stream, b)] = start + int(sel.size)
        return off

    def _enc_consts(self, mods):
        """The reply encoder's per-module constants for the modules `mods` in that order: sign, qpos and qvel
        addresses, the fields' X_MAX and 2 X_MAX, the id nibble."""
        xm = self._x3[mods]
        return (self.sign[mods], self.qadr[mods], self.dadr[mods], xm, -xm, self._x3_2[mods], self._idnib[mods])

    def _encode_states(self, mods, status, deltas, stream, consts=None, d0w=None):
        """The eight bytes of a state reply for each of `mods` (layout hostcodec.state_data), vectorised.
        consts: _enc_consts(mods) when the caller holds them; d0w: the first byte of every reply already shifted to
        its place (D0 << 56, uint64), when the caller holds it for `status`."""
        d = self.data
        s, qa, da, xm, nxm, x2, idn = self._enc_consts(mods) if consts is None else consts
        k = len(mods)
        x = np.empty((k, 3), dtype=np.float64)
        if self.all_pos:                                 # s = +1 on every module: s x v is v
            x[:, 0] = d.qpos[qa] - deltas
            x[:, 1] = d.qvel[da]
        else:
            x[:, 0] = s * (d.qpos[qa] - deltas)
            x[:, 1] = s * d.qvel[da]
        x[:, 2] = self.T_del[mods]
        if self.noise is not None:
            x += self._noise_offsets(stream, mods)        # q + n1, w + n2, tau + n3 (adding zeros changes no code)
        # std::min(x, hi), then std::max(lo, .) with NaN -> lo: minimum (hi is no NaN and no zero) and fmax (the
        # non-NaN operand), the selections of where(hi < x, hi, x) and where(lo < c, c, lo), element by element
        c = np.fmax(np.minimum(x, xm), nxm)
        u = np.floor(((c + xm) / x2) * self._lev3).astype(np.uint64)  # (x - lo) / (hi - lo), truncated: 0 .. 2^bits - 1
        temp = (np.minimum(255, np.maximum(0, np.floor(self.T_h[mods]))).astype(np.int64) & 0xFF).astype(np.uint64)
        if d0w is None:
            d0w = (((status & 0xF) << 4) | idn).astype(np.uint64) << _SH56
        # the eight bytes as one big-endian 64-bit word per reply: D0, q (16 bit), dq and torque (12 bit each), T_MOS and
        # T_Rotor (both floor(T_h): the reply reports the housing node; temp x 257 = temp << 8 | temp)
        word = d0w | (u[:, 0] << _SH40) | (u[:, 1] << _SH28) | (u[:, 2] << _SH16) | (temp * _U257)
        blob = word.astype(">u8").tobytes()
        return [blob[8 * j: 8 * j + 8] for j in range(k)]

    def coded_states(self, stream="tap"):
        """A reply-coded sample of every module now (for a recorder's taps: off the bus, outside the frame
        budget), with noise from `stream`: a list of N eight-byte rows in module order."""
        mods = np.arange(self.n, dtype=np.int64)
        return self._encode_states(mods, self.status().astype(np.int64), self.delta.copy(), stream)

    # ------------------------------------------------------------------------------------------- parameter frames
    def _param(self, i, op, f):
        d = f.data
        rid = d[3]
        head = bytes(d[0:4])
        if op == "read":
            if rid in rg.LIVE_RIDS:
                th = float(self.theta()[i])
                val = th * float(self.gr[i]) if rid == rg.P_M else th      # p_m on the motor side, xout on the shaft
                return Frame(f.bus, self.reply_id[i], head + rg.pack_value(rid, val))
            if self.regs.has(i, rid):
                return Frame(f.bus, self.reply_id[i], head + rg.pack_value(rid, self.regs.get(i, rid)))
            return Frame(f.bus, self.reply_id[i], bytes(d))   # a RID the unit does not carry: the request echoed
        if op == "write":
            value = rg.unpack_value(rid, d[4:8])
            ok, why = self.policy.admits(i, rid, value, bool(self.enabled[i]))
            reply_id = self.reply_id[i]
            if ok:
                self.regs.set(i, rid, value)
                self._event("register_write_accepted", module=i, rid=rid, value=value)
                if rid == rg.CTRL_MODE:
                    self._event("mode_changed", module=i, mode=int(value))
                self._refresh_register_caches()
                cur = self.regs.get(i, rid)
            else:
                cur = self.regs.get(i, rid) if self.regs.has(i, rid) else None
                self._event("register_write_refused", module=i, rid=rid, value=value, reason=why)
            if cur is None and not self.regs.has(i, rid):
                return Frame(f.bus, reply_id, head + bytes(4))
            return Frame(f.bus, reply_id, head + rg.pack_value(rid, cur))
        if op == "save":
            if self.enabled[i]:
                return None                   # refused while enabled, no reply (the maker's simulator)
            return Frame(f.bus, self.reply_id[i], bytes([d[0], d[1], hc.OP_SAVE, 0x01, 0, 0, 0, 0]))
        raise BusError(op)

    # ------------------------------------------------------------------------------------------------- stepping
    def _rebuild(self):
        """The per-step coefficients from the bus state (after any event: a command, enable, disable, a fault, a saved
        zero, a register write). A module that is not enabled gets zero coefficients and a zero cap, so that the
        per-step arithmetic gives it gain, bias, ctrl, force range, T_ref and T_del zero and its inertia entry at the
        model's value without a mask. The status part runs only after status events; the command part after
        every new command."""
        if self._dirty_status:
            self._rebuild_status()
        self._dirty = False
        mitm = self._mitm
        s = self.sign
        P = np.where(mitm, self.kp, 0.0)
        D = np.where(mitm, self.kd, self._d_fp)
        # c = kp (p_des + s delta) + kd v_des + t_ff; the force-position entries are set at every step
        self.c = np.where(mitm, self.kp * (self.p_des + s * self.delta) + self.kd * self.v_des + self.t_ff, 0.0)
        self.skp = s * P
        self.skd = s * D
        self.skc = self.sk * self.c
        self.kkd = self.kon * D
        self.dtkkd = self.dt * self.kkd
        # the inertia entry of each step is arm0 + unsat x dt k kd (unsat 1 or 0): both values, element by element
        self._arm_hi = self.arm0 + 1.0 * self.dtkkd
        self._arm_lo = self.arm0 + 0.0 * self.dtkkd
        self.model.actuator_biasprm[self.act, 1] = -self.kon * P

    def _rebuild_status(self):
        self._dirty_status = False
        on = self.enabled
        mitm = on & ~self.is_fp
        fpm = on & self.is_fp
        self._mitm = mitm
        onf = np.where(on, 1.0, 0.0)
        s = self.sign
        self.kon = self.k * onf
        self._d_fp = np.where(fpm, self.fp_kp_asr, 0.0)
        self.cap = np.where(mitm, self.t_lim, np.where(fpm, np.minimum(self.t_lim, self.fp_ilim * self.i_full), 0.0))
        self.e0 = self._E(np.zeros(self.n))
        self.b_on = np.where(on, self.b, 0.0)
        self.to_active = on & (self.timeout_steps > 0)
        self.to_any = bool(self.to_active.any())
        self.model.actuator_gainprm[self.act, 0] = onf
        self._fp_idx = np.nonzero(fpm)[0].tolist()
        self._fp = [self._fp_tuple(i) for i in self._fp_idx]

    def _fp_tuple(self, i):
        """The constants of the per-step force-position law of module i (Python floats)."""
        s = float(self.sign[i])
        pp = float(self.fp_kp_asr[i]) * float(self.fp_kp_apr[i])
        return (i, int(self.act[i]), s, float(self.sk[i]), float(self.kon[i]), float(self.fp_kp_apr[i]),
                float(self.fp_kp_asr[i]), pp, pp * (float(self.fp_p[i]) + s * float(self.delta[i])),
                float(self.fp_p[i]), float(self.delta[i]), float(self.fp_vlim[i]), float(self.fp_ki_asr[i]))

    def _fp_command(self, i):
        """A new force-position command of module i: its cap min(T_lim, i_lim Imax KT), E(0) and per-step constants,
        without rebuilding the others (the same operations as _rebuild_status, on one module)."""
        if self._dirty_status or i not in self._fp_idx:
            self._dirty = self._dirty_status = True
            return
        cap = min(float(self.t_lim[i]), float(self.fp_ilim[i]) * float(self.i_full[i]))
        self.cap[i] = cap
        self.e0[i] = max(0.0, min(min(cap, float(self.a1[i]) + float(self.b1[i]) * 0.0),
                                  float(self.a2[i]) + float(self.b2[i]) * 0.0))
        self._fp[self._fp_idx.index(i)] = self._fp_tuple(i)
        self._dirty = True

    def _E(self, aw):
        """E(|w|) of every module with the module's cap in place of T_lim (envelope.py's form: max(0, min(T_lim,
        A1 + B1 w, A2 + B2 w)); the cap is T_lim, or min(T_lim, I) in force-position mode, or 0 when not enabled)."""
        return np.maximum(0.0, np.minimum(np.minimum(self.cap, self.a1 + self.b1 * aw), self.a2 + self.b2 * aw))

    def _protect(self):
        """Step (a) of ONE EXCHANGE in the module text: the protections on the state at the step's start; faults latch
        and drop enable."""
        new_c = None
        if self.heat is not None:
            _, nc = self.heat.latch(self.fault != 0, self.T_w, self.ot)
            nc = np.asarray(nc, dtype=bool) & (self.fault == 0)
            if nc.any():
                new_c = nc
        new_uv = None
        if self._uv_any:
            nu = self._uv_hit & (self.fault == 0)
            if new_c is not None:
                nu &= ~new_c
            if nu.any():
                new_uv = nu
        new_d = None
        if self.to_any:
            nd = self.to_active & (self.steps_since >= self.timeout_steps)
            if nd.any():
                new_d = nd & (self.fault == 0)
                if new_c is not None:
                    new_d &= ~new_c
                if new_uv is not None:
                    new_d &= ~new_uv
        if new_c is None and new_uv is None and new_d is None:
            return
        for code, mask in ((ST_COIL, new_c), (ST_UNDER_VOLTAGE, new_uv), (ST_COMM_LOST, new_d)):
            if mask is None:
                continue
            for i in np.nonzero(mask)[0].tolist():
                if code == ST_COMM_LOST:
                    self._event("timeout", module=i)
                was = bool(self.enabled[i])
                self.fault[i] = code
                self.enabled[i] = False
                self._event("fault_set", module=i, code=code)
                if was:
                    self._event("disabled", module=i)
        self._dirty = self._dirty_status = True

    def pre_step(self):
        """Steps (a) and (b) of ONE EXCHANGE in the module text, for the next physics step: protections, then the
        carriers of every module (force range, ctrl, inertia entry; gain and bias at events and, in force-position
        mode, at every step) and b into
        qfrc_applied at the driven dofs (ASSIGNED: hook forces are added afterwards by the unit layer)."""
        self._protect()
        if self._dirty:
            self._rebuild()
        V = self._views
        q = V.qpos[self.qadr]
        qd = V.qvel[self.dadr]
        if self._fp:
            self._fp_step(q, qd)
        t_ref = self.c - self.skp * q - self.skd * qd
        w = qd if self.all_pos else self.sign * qd
        z = self._zeros                                       # 0.0 per module (the scalar 0.0, element by element)
        lines = self._a12 + self._b12 * np.abs(w)             # A1 + B1 |w| and A2 + B2 |w|
        ew = np.maximum(z, np.minimum(np.minimum(self.cap, lines[0]), lines[1]))   # = self._E(|w|), same order
        lp = np.where(w > z, ew, self.e0)
        nlm = np.negative(np.where(w < z, ew, self.e0))
        t_del = np.minimum(np.maximum(t_ref, nlm), lp)
        rp = np.minimum(lp, self.t_peak)
        nrm = np.maximum(nlm, self.neg_t_peak)
        unsat = (t_ref < rp) & (t_ref > nrm)
        a = self._act_ix
        if self.all_pos and type(a) is slice:          # written in place (the actuators are one run of addresses)
            np.multiply(self.kon, nrm, out=V.fr_lo)
            np.multiply(self.kon, rp, out=V.fr_hi)
            np.subtract(self.skc, self.kkd * qd, out=V.ctrl_a)
        else:
            fr, d = V.forcerange, self.data
            if self.all_pos:
                fr[a, 0] = self.kon * nrm
                fr[a, 1] = self.kon * rp
            else:
                fr[a, 0] = np.where(self.spos, self.kon * nrm, -(self.kon * rp))
                fr[a, 1] = np.where(self.spos, self.kon * rp, -(self.kon * nrm))
            d.ctrl[a] = self.skc - self.kkd * qd
        V.armature[self.dadr] = np.where(unsat, self._arm_hi, self._arm_lo)   # = arm0 + unsat x dt k kd
        V.qfrc_applied[self.dadr] = self.b_on
        self._t_ref_next = t_ref
        self._t_del_next = t_del
        self._unsat = unsat

    def _fp_step(self, q, qd):
        """The force-position law of the enabled force-position modules at the step's start, in Python floats
        (two modules on the pair; the same IEEE operations as an array would use): the speed reference, its clip, and
        the coefficients P (0 while the speed reference is clipped) and c of the carrier."""
        m = self.model
        out = []
        ql, qdl, zl = q.tolist(), qd.tolist(), self.Z.tolist()
        for (i, ai, s, sk, kon, kp_apr, kp_asr, pp, ppc, p, delta, vlim, ki) in self._fp:
            z = zl[i]
            qi = ql[i]
            w = s * qdl[i]
            th = s * (qi - delta)
            e = kp_apr * (p - th)
            vref = min(max(e, -vlim), vlim)
            if vref != e:
                P, c = 0.0, kp_asr * vref + z
            else:
                P, c = pp, ppc + z
            self.c[i] = c
            self.skp[i] = s * P
            self.skc[i] = sk * c
            m.actuator_biasprm[ai, 1] = -kon * P
            out.append((i, vref, w, ki))
        self._fp_out = out

    def post_step(self):
        """Step (e) of ONE EXCHANGE in the module text, after mj_step: the step's T_del becomes the module's last
        drive torque, the thermal update (with the heat plug-in), the force-position integral (conditional
        integration), the counters, the step count."""
        self.T_ref = self._t_ref_next
        self.T_del = self._t_del_next
        if self.heat is not None:
            tw, th = self.heat.step(self.T_w, self.T_h, self.T_del, self.t_amb)
            self.T_w = np.asarray(tw, dtype=np.float64)
            self.T_h = np.asarray(th, dtype=np.float64)
        if self._fp:
            dt = self.dt
            t_ref, t_del = self.T_ref, self.T_del
            for (i, vref, w, ki) in self._fp_out:
                inc = dt * ki * (vref - w)
                tr, td = float(t_ref[i]), float(t_del[i])
                # conditional integration: held only while clipped AND driven further into that clip
                if not ((tr > td and inc > 0.0) or (tr < td and inc < 0.0)):
                    self.Z[i] = float(self.Z[i]) + inc
        self.steps_since += self.enabled                 # the time-out counter runs only while enabled
        self.step_count += 1

    def run_exchange(self, frames, steps=4, admit="all"):
        """One exchange with nothing between the bus layer's phases and mj_step (a plain runner and the tests)."""
        replies = self.receive(frames, steps, admit)
        for _ in range(int(steps)):
            self.pre_step()
            mujoco.mj_step(self.model, self.data)
            self.post_step()
        return replies

    # ---------------------------------------------------------------------------------------------- state access
    def set_temperatures(self, t_w, t_h):
        """Set both nodes of every module (a session's start, a test). Only with a heat plug-in."""
        if self.heat is None:
            raise BusError("no thermal state without the heat plug-in")
        self.T_w = np.array([float(x) for x in t_w], np.float64)
        self.T_h = np.array([float(x) for x in t_h], np.float64)

    def set_supply(self, volts):
        """The supply voltage (the optional world hook set_supply): below a module's UV_Value it latches status 9
        at the next step."""
        self.supply_V = float(volts)
        self._uv_hit = self.supply_V < self.uv
        self._uv_any = bool(self._uv_hit.any())

    def truth(self):
        """The bus layer's part of the truth, copies: module angles, T_ref, T_del, delivered torque, k, b,
        status, enable, mode, T_w, T_h, steps_since_control (each module's time-out counter: the physics steps it has
        run enabled since its last control frame or, if later, its enable; held while it is not enabled), the force-
        position integral, the commands in force."""
        return {"step": self.step_count, "theta": self.theta(), "T_ref": self.T_ref.copy(), "T_del": self.T_del.copy(),
                "delivered": self.delivered(), "k": self.k.copy(), "b": self.b.copy(), "status": self.status(),
                "enabled": self.enabled.copy(), "mode": self.mode.copy(), "T_w": self.T_w.copy(),
                "T_h": self.T_h.copy(), "steps_since_control": self.steps_since.copy(), "Z": self.Z.copy(),
                "command_mit": np.stack([self.p_des, self.v_des, self.kp, self.kd, self.t_ff], axis=1).copy(),
                "command_fp": np.stack([self.fp_p, self.fp_vlim, self.fp_ilim], axis=1).copy(),
                "delta": self.delta.copy(), "t_lim": self.t_lim.copy()}

    _ARRAYS = ("delta", "T_w", "T_h", "p_des", "v_des", "kp", "kd", "t_ff", "fp_p", "fp_vlim", "fp_ilim", "Z",
               "T_ref", "T_del")

    def state_dict(self):
        """Everything of the bus layer that a snapshot must hold, as plain JSON-able data (floats exact)."""
        a = self.act
        return {
            "format": BUS_FORMAT, "n": self.n, "step_count": self.step_count,
            "arrays": {k: [float(x) for x in getattr(self, k)] for k in self._ARRAYS},
            "enabled": [bool(x) for x in self.enabled], "fault": [int(x) for x in self.fault],
            "steps_since": [int(x) for x in self.steps_since], "supply_V": self.supply_V,
            "registers": self.regs.to_list(), "frames_in": list(self.frames_in),
            "drops": sorted([b, r, c] for (b, r), c in self.drops.items()),
            "noise_counters": sorted([s, b, c] for (s, b), c in self.noise_counters.items()),
            "events": [dict(e) for e in self._events],
            "model_fields": {
                "gainprm0": [float(x) for x in self.model.actuator_gainprm[a, 0]],
                "biasprm1": [float(x) for x in self.model.actuator_biasprm[a, 1]],
                "forcerange": [[float(x), float(y)] for x, y in self.model.actuator_forcerange[a]],
                "armature": [float(x) for x in self.model.dof_armature[self.dadr]]},
        }

    def load_state_dict(self, s):
        """Restore what state_dict() returned (into this bus layer on an equal model)."""
        if s.get("format") != BUS_FORMAT or s["n"] != self.n:
            raise BusError("not a state of this bus layer")
        self.step_count = int(s["step_count"])
        for k in self._ARRAYS:
            v = np.array(s["arrays"][k], dtype=np.float64)
            if k in ("p_des", "v_des", "kp", "kd", "t_ff"):
                getattr(self, k)[:] = v
            else:
                setattr(self, k, v)
        self.enabled = np.array(s["enabled"], dtype=bool)
        self.fault = np.array(s["fault"], dtype=np.int64)
        self.steps_since = np.array(s["steps_since"], dtype=np.int64)
        self.supply_V = float(s["supply_V"])
        self.regs = rg.RegisterTable.from_list(s["registers"])
        self.frames_in = list(s["frames_in"])
        self.drops = {(b, r): c for b, r, c in s["drops"]}
        self.noise_counters = {(st, b): c for st, b, c in s["noise_counters"]}
        self._events = [dict(e) for e in s["events"]]
        self._refresh_register_caches()
        self._dirty = self._dirty_status = True
        self._rebuild()
        mf, a = s["model_fields"], self.act
        self.model.actuator_gainprm[a, 0] = mf["gainprm0"]
        self.model.actuator_biasprm[a, 1] = mf["biasprm1"]
        self.model.actuator_forcerange[a] = np.array(mf["forcerange"], dtype=np.float64).reshape(self.n, 2)
        self.model.dof_armature[self.dadr] = mf["armature"]


class _Views:
    """Views of the fixed-size arrays of one model and its data that the per-step path reads and writes (each
    attribute access of MuJoCo's handles makes a new view of the same memory; these are made once)."""

    def __init__(self, model, data, act):
        self.qpos, self.qvel, self.qfrc_applied, self.ctrl = data.qpos, data.qvel, data.qfrc_applied, data.ctrl
        self.forcerange, self.armature = model.actuator_forcerange, model.dof_armature
        a = _as_index(act)
        if type(a) is slice:
            self.fr_lo, self.fr_hi, self.ctrl_a = self.forcerange[a, 0], self.forcerange[a, 1], self.ctrl[a]


def _as_index(a):
    """A slice for an index array that is a run of consecutive addresses (a view in place of a gather), else the
    array."""
    a = np.asarray(a, dtype=np.int64)
    if a.size and np.array_equal(a, np.arange(int(a[0]), int(a[0]) + a.size, dtype=np.int64)):
        return slice(int(a[0]), int(a[0]) + int(a.size))
    return a


def _mk_frame(bus, can_id, data, _new=object.__new__):
    """A well-formed reply Frame without the constructor's checks (the bus layer builds only well-formed replies):
    the three fields written into the instance's dictionary, as the frozen dataclass's own __init__ leaves them."""
    f = _new(Frame)
    fd = f.__dict__
    fd["bus"] = bus
    fd["can_id"] = can_id
    fd["data"] = data
    return f
