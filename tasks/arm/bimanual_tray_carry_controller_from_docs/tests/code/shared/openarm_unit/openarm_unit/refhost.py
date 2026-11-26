"""Reference hosts of the OpenArm pair: ports of the maker's public host programs, spoken through frames only. PUBLIC.

PUBLIC: this module may ship with a task (the maker's driver law is part of a public runner). It imports numpy, MuJoCo
and the public modules `frames`, `hostcodec`, `registers` and `state` (and, inside `zero_routine_result` only, the
public `unit` and `base`, to build the unit that function runs on); nothing of the law of drawing, the registry, the
world, the recorder, the thermal model or the hidden parameter file. A host knows nothing of a unit: it sees the
frames the modules answer, as a program on a real bus does.

WHAT IS PORTED (each a line-by-line port written from the maker's text; the lines are cited at each function):
- `zero_calibration(side, ...)`: the zero-position calibration script of openarm_can 1.4.1
  (github.com/enactic/openarm_can at 20d66f0, `setup/openarm-can-zero-position-calibration`), v2 sequence, either
  arm: the gripper and joint bumps onto their mechanical stops, the `interpolate` moves from the MEASURED position,
  the correction the script computes and never applies, the 2.5 s wait, `disable_all`, then `set_zero_all`.
- `DriverPD(side, profile)`: the single-arm driver of openarm_driver 0.5.1 (github.com/enactic/openarm_driver at
  5c73d83, `src/openarm_driver/driver.py`, `safety.py`, `base_safety.py`, `configs/openarm_pedestal.yaml`): joint PD
  only (velocity target 0, feed-forward 0), the joint offsets, and its three checks in its order (the joint-limit
  clip, the single-jump stop, the command-velocity limiter), the 2 s start move at 50 Hz, the gripper in
  force-position mode with its limits.
- `AlignedStart(driver)`: the alignment walk of the dora-openarm node (github.com/enactic/dora-openarm at 1663ee5,
  `src/dora_openarm/main.py:37-79`): 0.001 rad per command until every arm joint is within 0.1 rad of the request.
- `zero_routine_result(state, session, ...)`: not a port: the zero routine of both arms (right, then left) run on a
  unit built from a unit state and a session, with the record a check of the run reads (the zeros written, per bump
  the stop it ends on and what else the arm touches there, the faults, a digest). A caller may attach an observer to
  that unit (`on_build`, for instance a raw capture of the bus) before the routine's first exchange.
- `PDGravity(profile, gravity)` and `GravityFloat(gravity)`: joint PD plus a model's gravity torque at the reported
  position, and the gravity torque alone (kp = kd = 0). `GravityModel(model, ...)` computes that torque from a
  compiled MuJoCo model with `mj_forward` (nothing else).

HOW A HOST RUNS. A program on a real bus sends frames, reads the replies and sleeps. Here a SCRIPT host (the zero
routine, the driver's start and stop sequences) is a sequence of steps `Step(frames, ms)`: the frames it sends at
one point of its program and the time until its next send, in whole milliseconds. `run(unit, host)` turns each step
into exchanges of the unit (the frames in the first, at most 50 steps each) and hands the replies back; the host
reads them where its program reads its bus (`recv_all`). The replies of an exchange are the modules' states at
reception, before the new commands act: one exchange old at the host, which is where the maker's programs read them
(each send is followed by a receive before the program sleeps). A step's length is its planned period rounded up to a
whole millisecond, and 1 ms where the program plans no wait: the real period is sleep plus bus time, and the bus
time is not published (an assumption of this simulation). No clock other than these step counts exists in a host.

The host's own view of each module is the maker's `Motor` record (openarm_can `dm_motor_device.cpp`, STATE callback):
every frame of at least eight bytes that arrives on the module's reply id is decoded as a state with the HOST's
mapping table (openarm_can `dm_motor_constants.hpp:149-164`) and overwrites position, velocity, torque, both
temperatures and the status, also a register reply that arrives in that mode, as on the real library. Replies are
matched by bus and reply id, never by their position in a list.

Arithmetic: float64; +, -, x, /, min, max, floor, abs, comparisons; numpy's element-wise forms of these and
`numpy.linspace` (the maker's own interpolation); no transcendental function, no wall clock. Degrees are turned into
radians by the multiplication numpy's `deg2rad` performs (x times pi / 180), so the constants are the maker's bits.
"""
import hashlib
import math
from typing import NamedTuple

import mujoco
import numpy as np

from . import hostcodec as hc
from . import registers as rg
from .frames import Frame
from .state import HostProfile, public_params

SIDES = ("left", "right")
N_ARM = 7                         # J1..J7 on each arm; module 8 of an arm is its gripper
MAX_STEPS_PER_EXCHANGE = 50
_RAD_PER_DEG = math.pi / 180.0    # numpy's deg2rad: x * (pi / 180)
_DEG_PER_RAD = 180.0 / math.pi    # numpy's rad2deg: x * (180 / pi)


class HostError(RuntimeError):
    """A host that cannot go on: a bump that meets no stop within its guard, a refused argument."""


def deg2rad(x):
    return float(x) * _RAD_PER_DEG


def rad2deg(x):
    return float(x) * _DEG_PER_RAD


def _copysign_one(x):
    """numpy.copysign(1.0, x) for a float."""
    return math.copysign(1.0, float(x))


class Step(NamedTuple):
    """What a script host sends at one point of its program and how long it then waits (whole milliseconds, >= 1)."""
    frames: tuple
    ms: int


# ------------------------------------------------------------------------------------------- the host's motor record
def _modules():
    return public_params()["conventions"]["modules"]["value"]


def bus_of(side):
    if side not in SIDES:
        raise HostError(f"side is one of {SIDES}")
    return SIDES.index(side)


class HostMotors:
    """The maker's per-motor record on one bus (openarm_can `Motor`, `DMCANDevice::callback` in STATE mode): the
    eight modules of one arm, J1..J7 then the gripper, with their command and reply ids and the host's mapping table
    of their motor types (`host_vmax_dm4340`: the table edition the host runs, 10 since 2026-07-17, 8 before)."""

    def __init__(self, side, host_vmax_dm4340=10.0):
        self.side = side
        self.bus = bus_of(side)
        table = hc.host_table(host_vmax_dm4340)
        mods = [m for m in _modules() if m["side"] == side]
        mods.sort(key=lambda m: m["joint"])
        self.command_id = [int(m["command_id"]) for m in mods]
        self.reply_id = [int(m["reply_id"]) for m in mods]
        self.limits = [table[m["class"]] for m in mods]
        self._by_reply = {rid: j for j, rid in enumerate(self.reply_id)}
        n = len(mods)
        self.position = [0.0] * n
        self.velocity = [0.0] * n
        self.torque = [0.0] * n
        self.t_mos = [0] * n
        self.t_rotor = [0] * n
        self.error_code = [0] * n
        self.responses = [0] * n
        self.pending = []                 # replies that have arrived and that the program has not read yet

    def deliver(self, replies):
        """Replies arrive on the host's socket (they are read at the program's next receive)."""
        self.pending.extend(replies or ())

    def recv_all(self):
        """The maker's `OpenArm::recv_all` (openarm_can `src/openarm/can/socket/openarm.cpp:90-128`): every frame that
        has arrived is dispatched to the motor of its reply id; at most 64 frames per call, the rest stay queued."""
        take, self.pending = self.pending[:64], self.pending[64:]
        for f in take:
            if f.bus != self.bus:
                continue
            j = self._by_reply.get(f.can_id)
            if j is None:
                continue
            self.responses[j] += 1
            st = hc.unpack_state(f.data, self.limits[j])
            if st is None:                  # fewer than eight bytes: no state (dm_motor_device.cpp STATE case)
                continue
            self.error_code[j] = int(st["status"])
            self.position[j] = float(st["q"])
            self.velocity[j] = float(st["dq"])
            self.torque[j] = float(st["tau"])
            self.t_mos[j] = int(st["t_mos"])
            self.t_rotor[j] = int(st["t_rotor"])

    # frame builders of the maker's collections (dm_motor_device_collection.cpp)
    def mit(self, j, kp, kd, q, dq, tau):
        return hc.pack_mit(self.bus, self.command_id[j], self.limits[j], q, dq, kp, kd, tau)

    def posforce(self, j, q, speed, current_pu):
        return hc.pack_posforce(self.bus, self.command_id[j], q, speed, current_pu)

    def all_frames(self, maker):
        """One frame per motor in the collections' order (the arm's J1..J7, then the gripper)."""
        return tuple(maker(self.bus, cid) for cid in self.command_id)

    def enable_all(self):
        return self.all_frames(hc.enable)

    def disable_all(self):
        return self.all_frames(hc.disable)

    def set_zero_all(self):
        return self.all_frames(hc.save_zero)

    def refresh_all(self):
        return self.all_frames(hc.refresh)

    def set_control_mode(self, j, mode):
        """openarm_can `DMDeviceCollection::set_control_mode_one` (dm_motor_device_collection.cpp:91-96): a write of
        the CTRL_MODE register (RID 10) on 0x7FF."""
        return hc.write_param(self.bus, self.command_id[j], rg.CTRL_MODE, int(mode))


# ---------------------------------------------------------------------------------------------- running a script host
class ScriptHost:
    """A host whose program is a generator of `Step`s; it receives the replies of each step through `after`."""

    def __init__(self, motors, gen):
        self.motors = motors
        self._gen = gen
        self.result = None
        self.done = False
        self.exchanges = 0
        self.ms = 0

    def begin(self):
        return self._next(None)

    def after(self, replies):
        self.motors.deliver(replies)
        return self._next(True)

    def _next(self, send):
        try:
            st = next(self._gen) if send is None else self._gen.send(None)
        except StopIteration as e:
            self.result = e.value
            self.done = True
            return None
        st = Step(tuple(st[0]), int(st[1]))
        if st.ms < 1:
            raise HostError("a step lasts at least one millisecond")
        self.ms += st.ms
        return st


def run(unit, host, *, before_exchange=None, after_exchange=None, max_ms=None):
    """Run a script host on a unit until its program ends; returns the host's result.

    Each `Step(frames, ms)` becomes exchanges of at most 50 steps, the frames in the first; the replies of all of them
    go back to the host. The host's register writes meet the unit's own write policy, as any program's do: the
    maker's programs write each gripper's CTRL_MODE while it is disabled (`init_gripper_motor`), which is what the
    tasks' whitelists admit; under a policy that admits nothing the write is refused, the gripper keeps its mode, and
    the zero routine's gripper bump cannot end (its guard raises HostError). before_exchange(unit, frames, steps) and
    after_exchange(unit, frames, replies, steps) are called around every exchange (a bus capture, a truth reader).
    max_ms: a guard on the host's total time (HostError when passed)."""
    step = host.begin()
    while step is not None:
        frames, remaining = step.frames, step.ms
        replies = []
        first = True
        while remaining > 0:
            n = min(MAX_STEPS_PER_EXCHANGE, remaining)
            fr = frames if first else None
            if before_exchange is not None:
                before_exchange(unit, fr, n)
            rep = unit.exchange(fr, steps=n)
            if after_exchange is not None:
                after_exchange(unit, fr, rep, n)
            replies.extend(rep)
            first = False
            remaining -= n
            host.exchanges += 1
        if max_ms is not None and host.ms > max_ms:
            raise HostError(f"the host ran longer than {max_ms} ms")
        step = host.after(replies)
    return host.result


def run_parallel(unit, hosts, *, before_exchange=None, after_exchange=None):
    """Run several script hosts at once on one unit, each on its own bus (the two arms' programs run side by side,
    as two processes do on two buses). Their steps are laid on one time line: an exchange starts whenever a host
    sends and lasts until the next host sends or ends a step (at most 50 steps); the frames of every host that sends
    at the same instant go into one exchange in the order of `hosts`; each host receives the replies on its bus of
    the exchanges its step covered. Register writes meet the unit's own write policy (as in `run`). Returns the
    hosts' results."""
    hosts = list(hosts)
    buses = [h.motors.bus for h in hosts]
    if len(set(buses)) != len(buses):
        raise HostError("one host per bus")
    now = 0
    cur = [h.begin() for h in hosts]
    start = [0 if s is not None else None for s in cur]
    end = [s.ms if s is not None else None for s in cur]
    sent = [False] * len(hosts)
    buf = [[] for _ in hosts]
    while any(s is not None for s in cur):
        frames = []
        for i, s in enumerate(cur):
            if s is not None and start[i] == now and not sent[i]:
                frames.extend(s.frames)
                sent[i] = True
        nxt = min(e for e, s in zip(end, cur) if s is not None)
        n = min(MAX_STEPS_PER_EXCHANGE, nxt - now)
        fr = frames or None
        if before_exchange is not None:
            before_exchange(unit, fr, n)
        rep = unit.exchange(fr, steps=n)
        if after_exchange is not None:
            after_exchange(unit, fr, rep, n)
        for i, s in enumerate(cur):
            if s is not None:
                buf[i].extend(f for f in rep if f.bus == buses[i])
                hosts[i].exchanges += 1
        now += n
        for i, s in enumerate(cur):
            if s is not None and end[i] == now:
                cur[i] = hosts[i].after(buf[i])
                buf[i] = []
                if cur[i] is not None:
                    start[i], end[i], sent[i] = now, now + cur[i].ms, False
    return [h.result for h in hosts]


# ============================================================================== the zero-position calibration routine
# The script's constants, as it prints them (openarm-can-zero-position-calibration at 20d66f0; line numbers there).
ZR_HOLD_KP = (300.0, 300.0, 150.0, 150.0, 40.0, 40.0, 30.0)        # l.381
ZR_HOLD_KD = (2.5, 2.5, 2.5, 2.5, 0.8, 0.8, 0.8)                   # l.382
ZR_HOLD_GRIP = (10.0, 0.9)                                         # l.384
ZR_INTERP = {"kp": 52.0, "kd": 1.5, "torque_assist": 0.0, "interp_time": 2.0, "n_steps": 500}   # l.85-92
ZR_BUMP = {"step_deg": 0.2, "kp": 45.0, "kd": 1.2, "torque_bias": 0.0, "period_s": 0.005}        # l.112-113, 129
ZR_THRESH = {"gripper": (0.3, 0.3), "J1": (0.0125, 5.0), "other": (0.1, 2.0)}                  # l.102-109
ZR_FOLD_MARGIN_DEG = 5.0                                           # l.74
# MECH_LIM_V2_RIGHT / _LEFT (l.49-69): the arm rows are equal on both sides; the gripper rows differ
ZR_MECH_LIM_DEG = ((-80.0, 200.0), (-100.0, 100.0), (-90.0, 90.0), (0.0, 140.0), (-90.0, 90.0), (-45.0, 45.0),
                   (-90.0, 90.0))
ZR_JOINT_SIGN = (-1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0)              # l.77-80
J_NAMES = ("J1", "J2", "J3", "J4", "J5", "J6", "J7")
GRIP = 7                                                           # the gripper motor in this module's lists


def _mech_lim(side):
    lim = [(deg2rad(lo), deg2rad(hi)) for lo, hi in ZR_MECH_LIM_DEG]
    # the gripper rows (l.57, l.68): right [0, -pi/2], left [0, pi/2]
    lim.append((0.0, -np.pi / 2.0) if side == "right" else (0.0, np.pi / 2.0))
    return lim


class ZeroCalibration(ScriptHost):
    """The maker's zero-position calibration of one arm (see `zero_calibration`). After `run`: `console` (the script's
    own printed lines), `stops` (per bump: the joint, the travel the script prints, the host's exchange count at the
    detection, the direction of travel in the module's angle (+1 or -1), the velocity and torque the host read when
    it stopped the bump and the script's two thresholds), `phase` (what the program is doing now: 'bump <joint>', 'interpolate <joint>', 'wait', ...),
    `deltas`, `ideal_delta_map` and `arm_goal_abs` (the correction the script computes and never applies)."""

    def __init__(self, side, profile=None, save_delay_steps=4, robot_version="v2", max_bump_steps=4000):
        if side not in SIDES:
            raise HostError(f"side is one of {SIDES}")
        if robot_version != "v2":
            raise HostError("only the v2 sequence is ported")
        sd = int(save_delay_steps)
        if sd < 1:
            raise HostError("the disable-to-save interval is at least one step (1 ms)")
        profile = HostProfile.default("A") if profile is None else profile
        self.side = side
        self.save_delay_steps = sd
        self.max_bump_steps = int(max_bump_steps)
        self.console = []
        self.stops = []
        self.phase = "init"
        self.deltas = None
        self.ideal_delta_map = None
        self.arm_goal_abs = None
        super().__init__(HostMotors(side, profile.host_vmax_dm4340), self._main())

    # ------------------------------------------------------------------------------------------- primitives
    def _print(self, s):
        self.console.append(s)

    def _thresholds(self, j):
        """_hit_thresholds (l.102-109)."""
        if j == GRIP:
            return ZR_THRESH["gripper"]
        if j == 0:
            return ZR_THRESH["J1"]
        return ZR_THRESH["other"]

    def _interpolate(self, j, delta_rad, kp=None, kd=None, torque_assist=None, interp_time=None):
        """interpolate (l.85-99): 501 MIT commands from the measured position q0 to q0 + delta, one per
        interp_time / 500 seconds (rounded up to whole milliseconds), each followed by a receive."""
        kp = ZR_INTERP["kp"] if kp is None else kp
        kd = ZR_INTERP["kd"] if kd is None else kd
        torque_assist = ZR_INTERP["torque_assist"] if torque_assist is None else torque_assist
        interp_time = ZR_INTERP["interp_time"] if interp_time is None else interp_time
        M = self.motors
        name = "GRIPPER" if j == GRIP else J_NAMES[j]
        self.phase = f"interpolate {name}"
        q0 = M.position[j]
        q1 = q0 + float(delta_rad)
        n_steps = ZR_INTERP["n_steps"]
        # dt = interp_time / 500 s; the step is its period in whole milliseconds, rounded up: ceil(interp_time x 2)
        ms = max(1, -math.floor(-(float(interp_time) * 1000.0 / n_steps)))
        tau = _copysign_one(q1 - q0) * abs(torque_assist)
        for i in range(n_steps + 1):
            alpha = i / n_steps
            q = q0 + (q1 - q0) * alpha
            yield Step((M.mit(j, kp, kd, q, 0.0, tau),), ms)
            M.recv_all()

    def _sleep(self, ms, label="wait"):
        self.phase = label
        yield Step((), int(ms))

    def _bump(self, j, step_deg=None, kp=None, kd=None, torque_bias=None):
        """bump_to_limit (l.112-137): a target moved by step_deg every 5 ms until the reply shows |velocity| below
        and |torque| above the thresholds; returns the travel from the start position. The script loops for ever;
        here a bump that meets no stop within max_bump_steps commands raises HostError."""
        step_deg = ZR_BUMP["step_deg"] if step_deg is None else step_deg
        kp = ZR_BUMP["kp"] if kp is None else kp
        kd = ZR_BUMP["kd"] if kd is None else kd
        torque_bias = ZR_BUMP["torque_bias"] if torque_bias is None else torque_bias
        M = self.motors
        # JointID.GRIPPER = 0 is an alias of J1 in the script's IntEnum, so the gripper's line names "J1"
        printed = "J1" if j == GRIP else J_NAMES[j]
        self.phase = f"bump {'GRIPPER' if j == GRIP else J_NAMES[j]}"
        step_rad = deg2rad(step_deg)
        tau_cmd = _copysign_one(step_deg) * abs(torque_bias)
        q_start = M.position[j]
        q_target = q_start
        dq_th, tau_th = self._thresholds(j)
        period_ms = int(round(ZR_BUMP["period_s"] * 1000.0))
        n = 0
        while True:
            q_target += step_rad
            yield Step((M.mit(j, kp, kd, q_target, 0.0, tau_cmd),), period_ms)
            M.recv_all()
            n += 1
            if abs(M.velocity[j]) < dq_th and abs(M.torque[j]) > tau_th:
                delta_rad = M.position[j] - q_start
                delta_deg = rad2deg(delta_rad)
                self._print(f"[INFO] mechanical stop (Joint {printed}): {delta_rad:.4f} rad / {delta_deg:.2f}°")
                self.stops.append({"joint": "GRIPPER" if j == GRIP else J_NAMES[j], "printed_joint": printed,
                                   "travel_rad": float(delta_rad), "commands": n, "host_ms": self.ms,
                                   "exchange": self.exchanges, "direction": int(_copysign_one(step_deg)),
                                   "reported_velocity": float(M.velocity[j]), "reported_torque": float(M.torque[j]),
                                   "velocity_threshold": float(dq_th), "torque_threshold": float(tau_th)})
                return float(delta_rad)
            if n >= self.max_bump_steps:
                raise HostError(f"{self.side} {self.phase}: no stop within {n} commands (the script would loop on)")

    @staticmethod
    def _j4_fold_delta(mech_lim):
        """j4_fold_delta (l.155-157)."""
        return mech_lim[3][1] - np.pi / 2.0 - deg2rad(ZR_FOLD_MARGIN_DEG)

    # --------------------------------------------------------------------------------------------- sequences
    def _right(self, mech_lim):
        """_run_right_sequence (l.162-244)."""
        M = self.motors
        M.recv_all()
        q_grip_now = M.position[GRIP]
        grip_step = 0.2
        pre_target = deg2rad(-10)
        pre_delta = pre_target - q_grip_now
        if pre_delta > deg2rad(3):
            interp_t = max(2.0, abs(rad2deg(pre_delta)) / 25.0)
            yield from self._interpolate(GRIP, pre_delta, kp=10.0, kd=0.9, interp_time=interp_t)
            yield from self._sleep(500)
        d_grip = yield from self._bump(GRIP, step_deg=grip_step)
        yield from self._sleep(500)
        d_j4 = yield from self._bump(3, step_deg=-0.2)
        yield from self._sleep(500)
        yield from self._interpolate(1, deg2rad(5), interp_time=0.4)
        yield from self._sleep(500)
        d_j3 = yield from self._bump(2)
        yield from self._sleep(500)
        yield from self._interpolate(2, -mech_lim[2][1], interp_time=1.0)
        yield from self._sleep(500)
        yield from self._interpolate(1, -deg2rad(5), interp_time=0.4)
        yield from self._sleep(500)
        yield from self._interpolate(3, np.pi / 2.0)
        yield from self._sleep(500)
        d_j5 = yield from self._bump(4)
        yield from self._sleep(500)
        yield from self._interpolate(4, -mech_lim[4][1])
        yield from self._sleep(500)
        d_j6 = yield from self._bump(5)
        yield from self._sleep(500)
        yield from self._interpolate(5, -mech_lim[5][1])
        yield from self._sleep(500)
        d_j7 = yield from self._bump(6)
        yield from self._sleep(500)
        yield from self._interpolate(6, -mech_lim[6][1])
        yield from self._sleep(500)
        d_j2 = yield from self._bump(1, step_deg=-0.2)
        yield from self._sleep(500)
        yield from self._interpolate(1, deg2rad(10), interp_time=0.9, kp=280, kd=2.0)
        yield from self._sleep(500)
        j4_fold = self._j4_fold_delta(mech_lim)
        yield from self._interpolate(3, j4_fold)
        yield from self._sleep(500)
        d_j1 = yield from self._bump(0, step_deg=-0.2, kp=280, kd=2.1)
        yield from self._sleep(500)
        yield from self._interpolate(0, deg2rad(80))
        yield from self._sleep(500)
        yield from self._interpolate(3, -j4_fold)
        yield from self._sleep(500)
        yield from self._interpolate(3, -np.pi / 2.0)
        yield from self._sleep(2500)
        ideal = {6: np.pi / 2.0, 5: np.pi / 4.0, 4: np.pi / 2.0, 3: 0.0, 2: np.pi / 2.0, 1: deg2rad(-10),
                 0: deg2rad(-80)}
        return ideal, [d_j1, d_j2, d_j3, d_j4, d_j5, d_j6, d_j7, d_grip]

    def _left(self, mech_lim):
        """_run_left_sequence with robot_version 'v2' (l.247-339)."""
        M = self.motors
        M.recv_all()
        q_grip_now = M.position[GRIP]
        grip_step = -0.2
        pre_target = deg2rad(-10)
        pre_delta = pre_target - q_grip_now
        if pre_delta < -deg2rad(3):
            interp_t = max(2.0, abs(rad2deg(pre_delta)) / 25.0)
            yield from self._interpolate(GRIP, pre_delta, kp=10.0, kd=0.9, interp_time=interp_t)
            yield from self._sleep(500)
        d_grip = yield from self._bump(GRIP, step_deg=grip_step)
        yield from self._sleep(500)
        d_j4 = yield from self._bump(3, step_deg=-0.2)
        yield from self._sleep(500)
        yield from self._interpolate(1, -deg2rad(5), interp_time=0.4)
        yield from self._sleep(500)
        d_j3 = yield from self._bump(2)
        yield from self._sleep(500)
        yield from self._interpolate(2, -mech_lim[2][1], interp_time=1.0)
        yield from self._sleep(500)
        yield from self._interpolate(1, deg2rad(5), interp_time=0.4)
        yield from self._sleep(500)
        yield from self._interpolate(3, np.pi / 2.0)
        yield from self._sleep(500)
        d_j5 = yield from self._bump(4)
        yield from self._sleep(500)
        yield from self._interpolate(4, -mech_lim[4][1])
        yield from self._sleep(500)
        d_j6 = yield from self._bump(5)
        yield from self._sleep(500)
        yield from self._interpolate(5, -mech_lim[5][1])
        yield from self._sleep(500)
        d_j7 = yield from self._bump(6)
        yield from self._sleep(500)
        yield from self._interpolate(6, -mech_lim[6][1])
        yield from self._sleep(500)
        d_j2 = yield from self._bump(1)
        yield from self._sleep(500)
        yield from self._interpolate(1, -deg2rad(10), interp_time=0.9, kp=280, kd=2.0)
        yield from self._sleep(500)
        j4_fold = self._j4_fold_delta(mech_lim)
        yield from self._interpolate(3, j4_fold)
        yield from self._sleep(500)
        d_j1 = yield from self._bump(0, kp=280, kd=2.1)
        yield from self._sleep(500)
        yield from self._interpolate(0, -deg2rad(80))
        yield from self._sleep(500)
        yield from self._interpolate(3, -j4_fold)
        yield from self._sleep(500)
        yield from self._interpolate(3, -np.pi / 2.0)
        yield from self._sleep(2500)
        ideal = {6: np.pi / 2.0, 5: np.pi / 4.0, 4: np.pi / 2.0, 3: 0.0, 2: np.pi / 2.0, 1: deg2rad(10),
                 0: deg2rad(80)}
        return ideal, [d_j1, d_j2, d_j3, d_j4, d_j5, d_j6, d_j7, d_grip]

    # ------------------------------------------------------------------------------------------------- main
    def _main(self):
        """run_calibration and main (l.344-446)."""
        M = self.motors
        canport = "can1" if self.side == "left" else "can0"
        self._print(f"parser arg : Namespace(canport='{canport}', arm_side='{self.side}_arm', robot_version='v2')")
        mech_lim = _mech_lim(self.side)
        # init_arm_motors writes no register (no control modes given, l.354-359); init_gripper_motor writes the
        # gripper's CTRL_MODE = MIT (its default; openarm_can gripper_component.cpp:26-37)
        init = (M.set_control_mode(GRIP, rg.MODE_MIT),)
        self._print("Enabling...")
        self.phase = "enable"
        yield Step(init + M.enable_all(), 100)                              # enable_all(); sleep(0.1)
        self._print("Enabled...")
        yield Step(M.refresh_all(), 10)                                     # refresh_all(); sleep(0.01)
        M.recv_all()                                                        # recv_all(2000)
        initial_arm_q = [M.position[j] for j in range(N_ARM)]
        initial_grip_q = [M.position[GRIP]]
        hold = tuple(M.mit(j, ZR_HOLD_KP[j], ZR_HOLD_KD[j], initial_arm_q[j], 0.0, 0.0) for j in range(N_ARM))
        hold += (M.mit(GRIP, ZR_HOLD_GRIP[0], ZR_HOLD_GRIP[1], initial_grip_q[0], 0.0, 0.0),)
        self.phase = "hold"
        yield Step(hold, 1)                                                 # mit_control_all x2; recv_all()
        M.recv_all()
        if self.side == "right":
            ideal, deltas = yield from self._right(mech_lim)
        else:
            ideal, deltas = yield from self._left(mech_lim)
        self.deltas = [float(d) for d in deltas]
        # the ideal deltas (l.397-411). JointID.GRIPPER == JointID.J1 (both 0), so the test `j == JointID.GRIPPER`
        # is true for J1 as well: J1 is skipped and the line is printed twice
        joint_order = [0, 1, 2, 3, 4, 5, 6, 0]
        idm = {}
        for j, d in zip(joint_order, deltas):
            if j == 0:
                self._print("ideal[GRIPPER] skipped")
                continue
            q_hit = initial_arm_q[j] + d
            idm[j] = float(ZR_JOINT_SIGN[j] * (ideal[j] - q_hit))
        self.ideal_delta_map = {J_NAMES[j]: v for j, v in idm.items()}
        self.arm_goal_abs = [initial_arm_q[i] + idm.get(i, 0.0) for i in range(N_ARM)]   # never applied (l.413-416)
        self.phase = "disable"
        yield Step(M.disable_all(), self.save_delay_steps)                  # disable_all(); recv_all()
        M.recv_all()
        self.phase = "set_zero"
        yield Step(M.set_zero_all(), 1)                                     # set_zero_all(); recv_all()
        M.recv_all()
        self._print("wrote zero position to arm")
        self.phase = "finally"
        yield Step(M.disable_all(), 1)                                      # finally: disable_all()
        self._print("[INFO] Motors disabled, exiting safely.")
        self.phase = "done"
        return {"side": self.side, "stops": list(self.stops), "deltas": self.deltas,
                "ideal_delta_map": self.ideal_delta_map, "arm_goal_abs": self.arm_goal_abs,
                "console": list(self.console)}


def zero_calibration(side, profile=None, save_delay_steps=4, robot_version="v2", max_bump_steps=4000):
    """The maker's zero-position calibration of one arm as a script host (run it with `run(unit, host)`).

    side: 'left' or 'right' (the script's --arm-side; its bus is 'can1' for the left arm, 'can0' for the right).
    profile: a HostProfile; only its host mapping table edition (`host_vmax_dm4340`) is used: every gain, step and
    threshold is the script's own. save_delay_steps: the time from the `disable_all` frames to the `set_zero_all`
    frames, in whole milliseconds (default 4: one receive between them on the real bus, its length not published).
    Timing: one step per program step, the step's planned period rounded up to a whole millisecond (5 ms per bump
    command, interp_time / 500 per interpolation command, 1 ms where the program plans no wait)."""
    return ZeroCalibration(side, profile, save_delay_steps, robot_version, max_bump_steps)


# ----------------------------------------------------------------------- the routine's run on a unit, with its record
ROUTINE_ORDER = ("right", "left")                 # the right arm first, then the left; the other arm hangs disabled
from .truth import STOP_EVENT_THRESHOLD_NM as STOP_ACTIVE_NM  # noqa: E402  a stop is active above it (the events')
FAULT_EVENTS = ("fault_set", "timeout", "frame_dropped", "register_write_refused")


def _arm_bodies(model, side):
    """Body ids of one arm (every body below its base link, mounted items included) and of its two fingers."""
    base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"openarm_{side}_base_link")
    if base_id < 0:
        raise HostError(f"the model lacks openarm_{side}_base_link")
    arm = {b for b in range(model.nbody) if int(model.body_rootid[b]) == base_id}
    fingers = {mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"openarm_{side}_ee_{f}_finger")
               for f in ("inner", "outer")}
    return arm, fingers - {-1}


def arm_contacts(unit, side):
    """The contacts of one arm in the unit's last solved step: (finger_pair, others). finger_pair: the arm's two
    fingers touch each other (a closed gripper); others: the other contacts that involve a body of that arm (another
    link of it, the other arm, the pedestal, a mounted item, an object), as sorted (body, body) name pairs. Only
    contacts that enter the constraint are counted. A read of the model's data, nothing is computed."""
    m, d = unit.model, unit.data
    arm, fingers = _arm_bodies(m, side)
    n = int(d.ncon)
    pair = False
    others = set()
    for k in range(n):
        if int(d.contact.exclude[k]) != 0:
            continue
        b1 = int(m.geom_bodyid[int(d.contact.geom[k][0])])
        b2 = int(m.geom_bodyid[int(d.contact.geom[k][1])])
        if b1 not in arm and b2 not in arm:
            continue
        if b1 in fingers and b2 in fingers and b1 != b2:
            pair = True
            continue
        names = sorted(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or f"body{b}" for b in (b1, b2))
        others.add(tuple(names))
    return pair, sorted(others)


def bump_stop_active(torque_at_detection, stop_begins, joint, side, gripper=False, finger_pair=(False, False)):
    """Whether a bump ended on its stop: (active, active_at_detection). torque_at_detection: the torque of the joint's
    stop on the bumped side at the detection instant; stop_begins: the (joint, side) pairs whose stop became active
    at some step of the bump's last command (the unit's stop_begin events); for a gripper, finger_pair: whether its
    two fingers touch at the detection instant and at the end of the last command (a closed gripper's fingers are
    its stop)."""
    at_det = float(torque_at_detection) > STOP_ACTIVE_NM
    active = at_det or (joint, side) in set(stop_begins)
    if gripper:
        active = active or bool(finger_pair[0]) or bool(finger_pair[1])
    return bool(active), bool(at_det)


def zero_routine_result(state, session, law_plugins=None, *, start_pose=None, noise_key=None, policy="stage04",
                        world=None, profile=None, save_delay_steps=4, max_bump_steps=4000, contacts=True,
                        arms=ROUTINE_ORDER, registry=None, on_build=None, _numerics_override=None):
    """The maker's zero-position calibration run on a unit built from (state, session), both arms, with the record
    that a check of the run needs.

    The unit is built with `OpenArmUnit.build(state, session, world, noise_key, contacts, law=law_plugins,
    policy=policy, registry=registry, _numerics_override=_numerics_override)` (law_plugins: the provider of the
    modules' heat and reply-noise models, or None; registry: the record of a registered unit, or None;
    _numerics_override: the unit's test-only numerics, for instance {'stop_compliance': 'halved'}, or None; the unit
    refuses an override together with a registry, and the record does not name the override), set at rest at
    `start_pose` (sixteen model coordinates; default: the straight hang, every joint 0, grippers closed; at the hang
    J4 rests on its lower stop, so a start pose near it holds J4 between 0 and 5 deg), every module disabled. Then the
    routine of the right arm, then of the left, each through `run` with its own host (`zero_calibration(side,
    profile, save_delay_steps, max_bump_steps=...)`); the other arm hangs disabled meanwhile (`arms`: the arms run,
    in that order; both by default).

    on_build: None, or a callable given the built unit once, after the unit is set at the start pose and before the
    routine's first exchange (to attach an observer, for instance a raw capture of every frame on the bus); what it
    returns is ignored. It is meant to observe: it exchanges no frame, drains no event and sets no state of the unit,
    and with such an observer the record equals that of the run without one. With None nothing is called.

    Returns a mapping:
      'zeros': the sixteen zeros stored in the modules after both routines, in model coordinates (the unit's truth
        `delta`; the grippers' included);
      'bumps': one entry per bump, in the order run (right arm first): 'arm', 'joint' (the model's joint name; a
        gripper's is its finger_joint1), 'name' (J1..J7 or GRIPPER), 'module', 'side' ('lower' or 'upper': the stop
        the bump moves towards, the module's direction of travel times the module's sign), 'stop_active' (the
        joint's stop on that side carries more than 0.01 N m at the detection instant or at some step of the bump's
        last command; for a gripper the closed fingers touching each other at the detection instant or at the end of
        the last command count as its stop), 'stop_active_at_detection', 'stop_torque_Nm' (that stop's torque at the
        detection instant), 'reported_torque_Nm' and 'reported_velocity_rad_s' (what the host read when it ended the
        bump), 'threshold_Nm' and 'velocity_threshold_rad_s' (the script's), 'other_contact' (any other contact of that
        arm at the detection instant: see `arm_contacts`), 'other_contacts' (their body pairs), 'finger_contact',
        'travel_rad', 'commands';
      'faults': the unit's fault, time-out, dropped-frame and refused-write events of the run (as dictionaries), and a
        host error ({'kind': 'host_error', 'arm', 'phase', 'message'}) when a routine could not go on (a bump that met
        no stop within its guard); the run stops there and the zeros are those in force at that point;
      'console': {arm: the script's printed lines}; 'state_digest', 'motion_digest': the unit's at the end;
      'digest': sha256 of the canonical text of zeros, bumps, faults, console and the two digests (equal for two
        runs that are equal).
    The detection instant is the instant whose replies ended the bump: the state at the start of the exchange that
    carried the bump's last command (replies are sampled at reception). The stop and contact readings come from the
    unit's truth and events and from its last solved step; they do not change the run."""
    from . import base as _base
    from .state import canonical_json
    from .unit import OpenArmUnit

    arms = tuple(arms)
    if not arms or len(set(arms)) != len(arms) or any(a not in SIDES for a in arms):
        raise HostError(f"arms: one or both of {SIDES}, each once")
    if on_build is not None and not callable(on_build):
        raise HostError("on_build: None or a callable that takes the built unit")
    unit = OpenArmUnit.build(state, session, world, noise_key, contacts, law=law_plugins, policy=policy,
                             registry=registry, _numerics_override=_numerics_override)
    q0 = [0.0] * 16 if start_pose is None else [float(x) for x in start_pose]
    unit.set_state(q0, [0.0] * 16)
    if on_build is not None:
        on_build(unit)
    hinges = list(_base.HINGES)
    mods = {(m["side"], int(m["joint"])): m for m in _modules()}
    sign = [int(s) for s in state.cal.sign]
    faults, bumps, console = [], [], {}
    for side in arms:
        host = zero_calibration(side, profile, save_delay_steps, max_bump_steps=max_bump_steps)
        seen = {}

        def before(u, frames, n, host=host, seen=seen, side=side):
            if host.phase.startswith("bump"):
                pair, others = arm_contacts(u, side)
                seen[host.exchanges] = {"stop_torque": u.truth(("stop_torque",))["stop_torque"].copy(),
                                        "finger_pair": pair, "others": others, "begins": set(), "pair_end": False}

        def after(u, frames, replies, n, host=host, seen=seen, side=side):
            ev = u.events()
            faults.extend(e.to_dict() for e in ev if e.kind in FAULT_EVENTS)
            rec = seen.get(host.exchanges)
            if rec is not None:
                rec["begins"] = {(e.fields["joint"], e.fields["which"]) for e in ev if e.kind == "stop_begin"}
                rec["pair_end"] = arm_contacts(u, side)[0]

        try:
            run(unit, host, before_exchange=before, after_exchange=after)
        except HostError as e:
            faults.append({"kind": "host_error", "arm": side, "phase": host.phase, "message": str(e)})
        console[side] = list(host.console)
        for s in host.stops:
            jn = 8 if s["joint"] == "GRIPPER" else int(s["joint"][1])
            mod = mods[(side, jn)]
            name = mod["mjcf_joint"]
            k = hinges.index(name)
            wside = "upper" if s["direction"] * sign[int(mod["index"])] > 0 else "lower"
            col = 1 if wside == "upper" else 0
            rec = seen[s["exchange"] - 1]
            torque = float(abs(rec["stop_torque"][k, col]))
            active, at_det = bump_stop_active(torque, rec["begins"], name, wside, gripper=jn == 8,
                                              finger_pair=(rec["finger_pair"], rec["pair_end"]))
            bumps.append({"arm": side, "joint": name, "name": s["joint"], "module": int(mod["index"]), "side": wside,
                          "stop_active": bool(active), "stop_active_at_detection": bool(at_det),
                          "stop_torque_Nm": torque, "reported_torque_Nm": float(s["reported_torque"]),
                          "reported_velocity_rad_s": float(s["reported_velocity"]),
                          "threshold_Nm": float(s["torque_threshold"]),
                          "velocity_threshold_rad_s": float(s["velocity_threshold"]),
                          "other_contact": bool(rec["others"]), "other_contacts": [list(p) for p in rec["others"]],
                          "finger_contact": bool(rec["finger_pair"]), "travel_rad": float(s["travel_rad"]),
                          "commands": int(s["commands"])})
        if any(f["kind"] == "host_error" for f in faults):
            break
    faults.extend(e.to_dict() for e in unit.events() if e.kind in FAULT_EVENTS)
    zeros = [float(x) for x in unit.truth(("delta",))["delta"]]
    out = {"zeros": zeros, "bumps": bumps, "faults": faults, "console": console,
           "state_digest": unit.state_digest(), "motion_digest": unit.motion_digest()}
    out["digest"] = hashlib.sha256(canonical_json(out).encode("ascii")).hexdigest()
    return out


# ============================================================================================ the maker's driver
# openarm_driver 5c73d83, configs/openarm_pedestal.yaml l.92-109: the start and stop moves (positions J1..J8)
START_POSITION = {"right": (0.0, 0.0, -0.0, 1.5707963267948966, 0.0, 0.0, 0.0, -0.4),
                  "left": (-0.0, 0.0, 0.0, 1.5707963267948966, 0.0, 0.0, 0.0, 0.4)}
STOP_POSITION = {"right": (0.0, 0.0, -0.0, 1.5707963267948966, 0.0, 0.0, 0.0, 0.0),
                 "left": (-0.0, 0.0, 0.0, 1.5707963267948966, 0.0, 0.0, 0.0, 0.0)}
DRIVER_REFRESH_LOOPS = 20          # driver.py:218-221 ("iterate until commutation is stable")
DRIVER_REFRESH_SLEEP_MS = 10       # driver.py:220 time.sleep(0.01)


def driver_joint_limits(side):
    """The driver's joint limits (configs/openarm_pedestal.yaml l.17-37): J1..J7 = the model file's joint ranges,
    J8 the gripper command range [-0.4, 1.0472] on the left arm and its mirror [-1.0472, 0.4] on the right."""
    p = public_params()
    arm = [tuple(float(x) for x in r) for r in p["conventions"]["joint_ranges_rad"]["value"][side]]
    lo, hi = (float(x) for x in p["gripper"]["driver_command_range_rad"]["value"])
    j8 = (lo, hi) if side == "left" else (-hi, -lo)
    return np.array(arm + [j8], dtype=np.float64)


class CheckResult(NamedTuple):
    """base_safety.CheckResult (base_safety.py:30-39)."""
    is_safe: bool
    force_stop: bool = False
    fixed_joint_positions: object = None
    message: str = ""
    check_type: str = None


def check_joint_limits(positions, joint_limits):
    """JointPosChecker.check (safety.py:35-62): clip to the limits; not safe, not a stop."""
    positions = np.asarray(positions, dtype=float)
    low, high = joint_limits[:, 0], joint_limits[:, 1]
    limited = np.clip(positions, low, high)
    violations = (positions < low) | (positions > high)
    if np.any(violations):
        nums = [j + 1 for j in np.where(violations)[0].tolist()]
        return CheckResult(False, False, limited, f"Joint positions over limits at joints: {nums}", "joint_limits")
    return CheckResult(True, message="All joint positions within limits.", check_type="joint_limits")


def check_joint_delta(positions, last_command, delta_limits):
    """JointDeltaPosChecker.check (safety.py:77-111): a jump beyond the limit from the present command stops the
    driver (force_stop); the first joint that exceeds it is named."""
    positions = np.asarray(positions, dtype=float)
    delta = positions - last_command
    for i, d in enumerate(delta):
        if abs(d) > delta_limits[i]:
            return CheckResult(False, True, None, f"Joint {i + 1} delta {d:.4f} exceeds limit {delta_limits[i]:.4f}",
                               "joint_delta")
    return CheckResult(True, message="All joint deltas within limits.", check_type="joint_delta")


def check_joint_velocity(positions, last_command, velocity_limits, dt_s):
    """JointVelocityChecker.check (safety.py:127-163): previous + clip(delta, +-v_lim dt)."""
    dt_s = float(dt_s)
    if not math.isfinite(dt_s) or dt_s < 0.0:
        raise ValueError("Command period must be finite and non-negative.")
    positions = np.asarray(positions, dtype=float)
    previous = np.asarray(last_command, dtype=float)
    if not (positions.shape == previous.shape == velocity_limits.shape):
        raise ValueError("Joint positions, previous command, and velocity limits must have the same shape.")
    if not np.all(np.isfinite(positions)) or not np.all(np.isfinite(previous)):
        raise ValueError("Joint positions must contain only finite values.")
    delta = positions - previous
    max_delta = velocity_limits * dt_s
    limited = previous + np.clip(delta, -max_delta, max_delta)
    violations = np.abs(delta) > max_delta
    if np.any(violations):
        nums = [j + 1 for j in np.where(violations)[0].tolist()]
        return CheckResult(False, False, limited, f"Joint velocity limited at joints: {nums}", "joint_velocity")
    return CheckResult(True, message="All joint command velocities within limits.", check_type="joint_velocity")


def check_composite(positions, last_command, joint_limits, delta_limits, velocity_limits, dt_s):
    """CompositeChecker.check (base_safety.py:115-141) over the driver's default checkers in their order
    (driver.py:102-113): the joint-limit clip, the single-jump stop, the command-velocity limiter. A stop returns at
    once; a fix is carried into the next check; the last failing check's message is reported."""
    not_safe = []
    is_safe = True
    for chk in (lambda p: check_joint_limits(p, joint_limits),
                lambda p: check_joint_delta(p, last_command, delta_limits),
                lambda p: check_joint_velocity(p, last_command, velocity_limits, dt_s)):
        r = chk(positions)
        if not r.is_safe:
            not_safe.append(r)
            if r.force_stop:
                return r
            if r.fixed_joint_positions is not None:
                is_safe = False
                positions = r.fixed_joint_positions
    if is_safe:
        return CheckResult(True, message="All 3 checks passed")
    return CheckResult(False, False, positions, not_safe[-1].message, not_safe[-1].check_type)


class DriverPD:
    """openarm_driver 0.5.1 `SingleArmDriver` for one arm on its bus, with a HostProfile (gains J1..J7, the gripper
    mode and limits, the three checks' limits, the start move, the joint offsets, the host mapping table).

    Use: `run(unit, drv.construct())`, `run(unit, drv.start())`; then per command `frames = drv.send_position(p)`,
    `replies = unit.exchange(frames, steps=n)`, `drv.receive(replies)`, `drv.elapse(n)` (or `drv.tick(p, n)` on a
    unit, which does the four); `run(unit, drv.stop())`. The time a check reads is the steps the driver has been told
    of since its last command (`elapse`), times the physics step, capped at velocity_dt_cap_s (driver.py:36, 491-492):
    no wall clock. A single-jump stop latches `safety_stop_reason`; every later command is refused (no frames) until
    stop and start, and an event 'driver_stopped' is kept in `events` (drained by `drain_events`): the stop is the
    host's own decision, which the modules never see, so it is an event of the host, not of the unit. With a profile
    whose gripper mode is MIT the gripper is commanded with the profile's gripper gains (`gripper_mit_kp`,
    `gripper_mit_kd`), not with the driver file's eighth gain pair."""

    def __init__(self, side, profile=None, joint_limits=None, start_position=None, stop_position=None):
        profile = HostProfile.default("A") if profile is None else profile
        if not isinstance(profile, HostProfile):
            raise HostError("profile is a HostProfile")
        self.side = side
        self.profile = profile
        self.motors = HostMotors(side, profile.host_vmax_dm4340)
        self.gripper_posforce = profile.gripper_mode == "force_position"
        self.num_mit_motors = 7 if self.gripper_posforce else 8
        self.kps = np.array(list(profile.kp) + [profile.gripper_mit_kp], dtype=float)
        self.kds = np.array(list(profile.kd) + [profile.gripper_mit_kd], dtype=float)
        self.joint_offsets = np.array(profile.joint_offsets, dtype=float)
        self.joint_limits = driver_joint_limits(side) if joint_limits is None else np.asarray(joint_limits, float)
        self.delta_limits = np.array(profile.jump_limits, dtype=float)
        self.velocity_limits = np.array(profile.velocity_limits, dtype=float)
        if (self.velocity_limits.ndim != 1 or not np.all(np.isfinite(self.velocity_limits))
                or np.any(self.velocity_limits <= 0.0)):
            raise ValueError("Joint velocity limits must be finite and positive.")
        self.posforce_limits = (profile.gripper_fp_speed, profile.gripper_fp_limit)
        self.plateau_divisor = float(public_params()["gripper"]["plateau_divisor"]["value"])   # driver.py:524
        self.step_s = float(public_params()["conventions"]["time"]["value"]["step_s"])
        self.start_position = START_POSITION[side] if start_position is None else tuple(start_position)
        self.stop_position = STOP_POSITION[side] if stop_position is None else tuple(stop_position)
        self.latest_state = None
        self.last_command = None
        self.started = False
        self.safety_stop_reason = None
        self.steps_since_command = 0
        self.last_check = None
        self.events = []
        self.commands_sent = 0

    # --------------------------------------------------------------------------------------- bus primitives
    def receive(self, replies=()):
        """set_latest_state (driver.py:433-454): receive, then the state of the eight motors (positions minus the
        joint offsets)."""
        self.motors.deliver(replies)
        self.motors.recv_all()
        M = self.motors
        self.latest_state = {"qpos": np.array(M.position, dtype=float) - self.joint_offsets,
                             "qvel": np.array(M.velocity, dtype=float), "qtorque": np.array(M.torque, dtype=float),
                             "tmos": np.array(M.t_mos, dtype=int), "trotor": np.array(M.t_rotor, dtype=int),
                             "status": np.array(M.error_code, dtype=int)}
        return self.latest_state

    def elapse(self, steps):
        self.steps_since_command += int(steps)

    def send_position(self, position, steps_since_command=None):
        """send_position (driver.py:484-531): the checks, then MIT frames J1..J7 (J1..J8 with the gripper in MIT),
        kp and kd of the profile, target + offset, velocity 0, torque 0, and the gripper's force-position frame
        (speed limit, current limit / 4.5). Returns the frames; none while a safety stop is latched or when this
        command trips it."""
        if steps_since_command is not None:
            self.steps_since_command = int(steps_since_command)
        if self.safety_stop_reason is not None:
            return []
        elapsed_s = max(self.steps_since_command * self.step_s, 0.0)
        dt_s = min(elapsed_s, self.profile.velocity_dt_cap_s)
        r = check_composite(position, self.last_command, self.joint_limits, self.delta_limits, self.velocity_limits,
                            dt_s)
        self.last_check = r
        if not r.is_safe:
            if r.force_stop:
                self.safety_stop_reason = r.message
                self.events.append({"kind": "driver_stopped", "reason": r.message, "command": self.commands_sent})
                return []
            if r.fixed_joint_positions is not None:
                position = r.fixed_joint_positions
        target = np.array(position, dtype=float)
        M = self.motors
        frames = [M.mit(i, self.kps[i], self.kds[i], target[i] + self.joint_offsets[i], 0, 0)
                  for i in range(self.num_mit_motors)]
        if self.gripper_posforce:
            # GripperComponent::set_position (gripper_component.cpp:44-57): both limits clamped
            speed = min(max(float(self.posforce_limits[0]), 0.0), 100.0)
            torque_pu = min(max(float(self.posforce_limits[1]) / self.plateau_divisor, 0.0), 1.0)
            frames.append(M.posforce(GRIP, target[-1] + self.joint_offsets[-1], speed, torque_pu))
        self.last_command = target
        self.steps_since_command = 0
        self.commands_sent += 1
        return frames

    def tick(self, unit, position, steps):
        """One command on a unit: send_position, one exchange of `steps`, receive, elapse. Returns the replies."""
        frames = self.send_position(position)
        replies = unit.exchange(frames, steps=steps)
        self.receive(replies)
        self.elapse(steps)
        return replies

    def drain_events(self):
        e, self.events = self.events, []
        return e

    # -------------------------------------------------------------------------------------------- scripts
    def _wrap(self, gen):
        drv = self

        class _DriverScript(ScriptHost):
            def after(self, replies):
                drv.elapse(self._last_ms)
                self.motors.deliver(replies)
                return self._next(True)

            def _next(self, send):
                st = ScriptHost._next(self, send)
                self._last_ms = 0 if st is None else st.ms
                return st

        return _DriverScript(self.motors, gen)

    def construct(self):
        """The driver's constructor on the bus (driver.py:160-224): init_gripper_motor writes the gripper's CTRL_MODE
        (force-position when the profile says so, MIT otherwise: gripper_component.cpp:36), then 20 times
        sleep(0.01), refresh_all and a receive; last_command = the position read."""
        return self._wrap(self._construct())

    def _construct(self):
        M = self.motors
        mode = rg.MODE_POS_FORCE if self.gripper_posforce else rg.MODE_MIT
        init = (M.set_control_mode(GRIP, mode),) if self.gripper_posforce else ()
        # in MIT mode the gripper is one of init_arm_motors' eight, which writes no register (driver.py:174-176)
        yield Step(init, DRIVER_REFRESH_SLEEP_MS)
        for i in range(DRIVER_REFRESH_LOOPS):
            last = i == DRIVER_REFRESH_LOOPS - 1
            yield Step(M.refresh_all(), 1 if last else DRIVER_REFRESH_SLEEP_MS)
            self.receive()
            self.last_command = self.latest_state["qpos"]
        self.steps_since_command = 0
        return self.last_command.copy()

    def start(self):
        """start (driver.py:231-256): enable_all, receive, refresh_all, receive, last_command = the position read,
        then the start move; the script's result is True when it completed."""
        return self._wrap(self._start())

    def _start(self):
        M = self.motors
        self.started = False
        self.safety_stop_reason = None
        yield Step(M.enable_all(), 1)
        self.receive()
        yield Step(M.refresh_all(), 1)
        self.receive()
        self.last_command = self.latest_state["qpos"].copy()
        self.steps_since_command = 0
        ok = yield from self._smooth_move(self.start_position, self.profile.start_move_hz,
                                          self.profile.start_move_s)
        if ok is False or self.safety_stop_reason is not None:
            if self.safety_stop_reason is None:
                self.safety_stop_reason = "Start trajectory was interrupted"
            return False
        self.started = True
        return True

    def smooth_move(self, position, hz, duration):
        """smooth_move (driver.py:545-564) as a script."""
        return self._wrap(self._smooth_move(position, hz, duration))

    def _smooth_move(self, position, hz, duration):
        num_steps = int(hz * duration)
        if num_steps <= 0:
            raise ValueError(f"smooth_move step calc error: hz {hz}, duration: {duration}")
        period_ms = max(1, -math.floor(-(1000.0 / float(hz))))              # sleep(1 / hz), whole milliseconds
        for p in np.linspace(np.array(self.last_command), np.array(position, dtype=float), num_steps):
            frames = self.send_position(p)
            if not frames and self.safety_stop_reason is not None:
                return False
            yield Step(tuple(frames), period_ms)
            self.receive()
        return True

    def stop(self):
        """stop (driver.py:258-275): the stop move unless a safety stop is latched, disable_all, a receive,
        sleep(1)."""
        return self._wrap(self._stop())

    def _stop(self):
        if self.safety_stop_reason is None:
            yield from self._smooth_move(self.stop_position, self.profile.start_move_hz, self.profile.start_move_s)
        yield Step(self.motors.disable_all(), 1000)
        self.receive()
        self.started = False
        return True


# ================================================================================================ the dora node's walk
class AlignedStart:
    """The dora-openarm node's alignment after a start (main.py:37-79, defaults l.237-252): while not aligned, each
    requested position moves the driver's target by at most step_limit per joint towards it; the walk ends, and the
    request itself is sent, once every arm joint (the gripper excluded) is within `threshold` of the request. The
    present position is the driver's latest state, cast to float32 as the node casts it (the node's own refresh
    before each decision is the receive of the previous command's replies here)."""

    def __init__(self, driver, threshold=0.1, step_limit=0.001, trigger=None):
        if step_limit <= 0.0:
            raise HostError("the alignment step limit must be positive")
        self.driver = driver
        self.threshold = float(threshold)
        self.step_limit = float(step_limit)
        self.trigger = trigger
        self.align_target = None
        self.aligned = False

    def move(self, new_position):
        """One move_position request: (frames, aligned)."""
        drv = self.driver
        new_position = np.asarray(new_position)
        if self.aligned:
            return drv.send_position(new_position), True
        if self.trigger == "gripper":
            g = new_position[-1]
            gripping = g > deg2rad(-5) if drv.side == "right" else g < deg2rad(5)
            if not gripping:
                return [], False
        current = np.array(drv.latest_state["qpos"], dtype=np.float32)
        if self.align_target is None:
            self.align_target = current.copy()
        if np.all(np.abs(new_position[:-1] - current[:-1]) < self.threshold):
            frames = drv.send_position(new_position)
            self.aligned = bool(frames)
            return frames, self.aligned
        diff = new_position - self.align_target
        step_move = np.clip(diff, -self.step_limit, self.step_limit)
        next_target = self.align_target + step_move
        frames = drv.send_position(next_target)
        if frames:
            self.align_target = next_target
        return frames, False


# ======================================================================================= PD with a model's gravity
class GravityModel:
    """The gravity torque of a model at module angles, in each module's own torque units: for an arm module i,
    t_ff = s_i (g_i(q) - b_i) / k_i with q = zero_offset + s theta (so that the joint receives k t_ff + b = g), and 0
    for the grippers. model: a compiled MuJoCo model holding the pair's eighteen hinges by the maker's names (a
    submitted description, the maker's file, a unit's model); calibration: {joint name: (sign, zero_offset_rad)} of the
    arm joints (default: sign +1, offset 0); k, b: sixteen values (default 1 and 0). Gravity is read as qfrc_bias after
    `mj_forward` at zero velocity on a private MjData (nothing else of MuJoCo is called)."""

    def __init__(self, model, calibration=None, k=None, b=None):
        self.model = model
        self.data = mujoco.MjData(model)
        mods = _modules()
        self.mods = mods
        self.qadr, self.dadr, self.f2 = [], [], {}
        for m in mods:
            j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, m["mjcf_joint"])
            if j < 0:
                raise HostError(f"the model lacks joint {m['mjcf_joint']}")
            self.qadr.append(int(model.jnt_qposadr[j]))
            self.dadr.append(int(model.jnt_dofadr[j]))
        for side in SIDES:
            j2 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"openarm_{side}_finger_joint2")
            if j2 >= 0:
                self.f2[side] = int(model.jnt_qposadr[j2])
        cal = dict(calibration or {})
        self.sign = np.array([float(cal.get(m["mjcf_joint"], (1.0, 0.0))[0]) for m in mods])
        self.zero = np.array([float(cal.get(m["mjcf_joint"], (1.0, 0.0))[1]) for m in mods])
        self.k = np.ones(len(mods)) if k is None else np.array([float(x) for x in k])
        self.b = np.zeros(len(mods)) if b is None else np.array([float(x) for x in b])
        self.arm = np.array([int(m["joint"]) != 8 for m in mods])

    def joint_gravity(self, theta16):
        """g(q) in N m at the sixteen driven dofs for module angles theta16."""
        d = self.data
        th = np.array([float(x) for x in theta16], dtype=np.float64)
        q = self.zero + self.sign * th
        d.qpos[:] = self.model.qpos0
        d.qvel[:] = 0.0
        d.qpos[self.qadr] = q
        for s, side in enumerate(SIDES):
            if side in self.f2:
                d.qpos[self.f2[side]] = q[8 * s + 7]
        mujoco.mj_forward(self.model, d)
        return np.array(d.qfrc_bias[self.dadr], dtype=np.float64)

    def __call__(self, theta16):
        g = self.joint_gravity(theta16)
        t = self.sign * (g - self.b) / self.k
        return np.where(self.arm, t, 0.0)


class PDGravity:
    """Joint PD plus a model's gravity at the reported position (the law of the maker-format real logs): per command
    and arm module, kp and kd of the profile, the target, velocity 0, t_ff = gravity(theta) with theta the positions
    of the latest replies received (or given); grippers by the profile's gripper mode (force-position: the target with
    the speed limit and current limit / 4.5; MIT: the profile's gripper kp, kd). gravity: a callable of sixteen module
    angles returning sixteen feed-forward torques in the modules' units (a GravityModel, or any function)."""

    def __init__(self, profile, gravity, sides=SIDES):
        self.profile = profile
        self.gravity = gravity
        self.sides = tuple(sides)
        self.motors = {s: HostMotors(s, profile.host_vmax_dm4340) for s in SIDES}
        self.divisor = float(public_params()["gripper"]["plateau_divisor"]["value"])

    def receive(self, replies):
        for M in self.motors.values():
            M.deliver(replies)
            M.recv_all()

    def positions(self):
        return [p for s in SIDES for p in self.motors[s].position]

    def feedforward(self, theta16=None):
        return np.asarray(self.gravity(self.positions() if theta16 is None else theta16), dtype=float)

    def frames(self, targets16, theta16=None, kp=None, kd=None):
        """The frames of one command to the arms of `sides` (targets: sixteen module angles; kp, kd: 7 values to
        override the profile's)."""
        P = self.profile
        tff = self.feedforward(theta16)
        kp = P.kp if kp is None else kp
        kd = P.kd if kd is None else kd
        out = []
        for s, side in enumerate(SIDES):
            if side not in self.sides:
                continue
            M = self.motors[side]
            for j in range(N_ARM):
                i = 8 * s + j
                out.append(M.mit(j, kp[j], kd[j], float(targets16[i]), P.v_des, float(tff[i])))
            g = float(targets16[8 * s + GRIP])
            if P.gripper_mode == "force_position":
                out.append(M.posforce(GRIP, g, P.gripper_fp_speed, P.gripper_fp_limit / self.divisor))
            else:
                out.append(M.mit(GRIP, P.gripper_mit_kp, P.gripper_mit_kd, g, P.v_des, 0.0))
        return out


class GravityFloat(PDGravity):
    """kp = kd = 0 and t_ff = the model's gravity at the reported position: the arm floats on the model's gravity.
    The grippers get no frame (they keep their last command)."""

    def __init__(self, gravity, sides=SIDES, host_vmax_dm4340=10.0):
        prof = HostProfile.default("A", host_vmax_dm4340=host_vmax_dm4340)
        super().__init__(prof, gravity, sides)

    def frames(self, theta16=None):
        tff = self.feedforward(theta16)
        pos = self.positions() if theta16 is None else list(theta16)
        out = []
        for s, side in enumerate(SIDES):
            if side not in self.sides:
                continue
            M = self.motors[side]
            for j in range(N_ARM):
                i = 8 * s + j
                out.append(M.mit(j, 0.0, 0.0, float(pos[i]), 0.0, float(tff[i])))
        return out


def pd_gravity(profile, model, **kw):
    """PDGravity with a GravityModel of `model` (a compiled MjModel; calibration, k, b by keyword) or with `model`
    itself when it is already a callable."""
    g = model if callable(model) and not isinstance(model, mujoco.MjModel) else GravityModel(model, **kw)
    return PDGravity(profile, g)


def gravity_float(model, **kw):
    g = model if callable(model) and not isinstance(model, mujoco.MjModel) else GravityModel(model, **kw)
    return GravityFloat(g)


def driver_pd(profile=None, side="left", **kw):
    """The maker's driver for one arm (DriverPD)."""
    return DriverPD(side, profile, **kw)


def aligned_start(driver, **kw):
    """The dora node's alignment walk on a driver (AlignedStart)."""
    return AlignedStart(driver, **kw)
