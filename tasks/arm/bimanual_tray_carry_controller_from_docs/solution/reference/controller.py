"""The task's privileged reference AS A SUBMITTED CONTROLLER (the oracle). HIDDEN: maintainers' code with the truth in
hand; never shipped to an agent.

    class Controller:  Controller(); setup(unit_record, site, truth=None) -> {"gripper_mode": ...}; step(feedback) -> reply

The grader's runtime loads this file as the submission's controller/controller.py in its own confined process (the
solution's copy, solution/reference/controller.py, is this file byte for byte). The module imports nothing but the
standard library at import time.

WITHOUT THE TRUTH (no token, or a token whose sha256 the grader does not hold: setup receives no `truth`) it holds the
hand-over pose, as the published example does: the module angles of the first feedback with the maker's first gain
set, both grippers in force-position mode holding their first angles with a small current limit, never a decline. It
imports nothing hidden and reads nothing of the unit. The first job then fails at its capture and the shift ends with a
keep-out event at that tray's removal (the hand-over pose lies inside the set-down shelf's keep-out).

WITH THE TRUTH (a submission that carries the token whose sha256 the grader holds: setup receives the truth object
built by truth_payload() below). THE LAW IS THE INSTRUMENT'S (minimal.py), imported, not copied: setup puts the
instrument's folder named in the truth (truth['code']['ref']) on the import path, builds the unit's own simulator from
the truth (the drawn unit's state and session, its site, the cell, its reply-noise key and law, exactly as the grader
built the unit), fills minimal.Beliefs from it by minimal.beliefs_from_truth (every field from the truth, coil sparing
on) and runs minimal.MinimalReference (its law, squeeze, gains and schedule unchanged, no options) inside the
instrument's shift.run_shift on that simulator, in a thread of this process, cycle by cycle in lockstep with the wire.

WHY A SIMULATOR OF THE TRUTH IN THE CONTROLLER'S PROCESS: MinimalReference in the grader's process reads, besides the
truth at the job's start, the PER-STEP truth of the run: each grasp's clamp (the coil-sparing switch of its gripper
frames), each weld's state (the decline at the first cycle after both captures), the joint angles the hand-over left
(the plan's start) and the supervisor's missed-capture signal. None of them is on the wire, and the truth is handed
once, before the hand-over. The one way to compute through the wire the frames MinimalReference computes in the
grader's process is to compute that per-step truth from the truth handed at setup and the frames the modules
received: the unit's own simulator, stepped with exactly those frames. The replies tell which frames the modules
received (the frames of a late reply are not sent), and they are checked against the simulator's own replies (every
CHECK_EVERY-th cycle and at every cycle whose feedback carries an age other than 1; module by module: q, dq, tau,
t_mos, t_rotor, status and age, bit for bit); a difference is counted and reported on the process's standard error,
never acted on (nothing in the law reads a reply).

THE LOCKSTEP (one wire call at cycle k; the simulator's run_shift waits inside the proxy's step(c) for c <= k):
  - the frames the reference computed at the previous call (cycle c) reach the simulator at cycle c if they were
    sent: k == c + 1 and every module's reply is one cycle old (age 1); else none (a reply late by L cycles: the
    runtime sends nothing in cycles c .. c + L - 1 and calls next at k = c + L, whose feedback carries age L + 1, so
    a reply late by one cycle shows ONLY in the ages), and the late reply's decline reaches the simulator's
    supervisor at cycle k, as the runtime hands it on;
  - the simulator then runs on (its exchange, per-step truth, monitor and supervisor) to cycle k, where
    MinimalReference.step(k, site signal, replies) gives this call's sixteen frames and decline;
  - each CAN frame is written as the wire frame whose clip and truncation by the runtime give the same bytes again
    (the middle of each code step; the force-position angle as its float32), so the modules receive the bytes the
    in-process reference sends.
Between a reply and the next call the controller computes nothing (the simulator waits for the next feedback).
THE STEP BUDGET: the grader runs a controller whose token is valid with a step budget of its own (0.1 s per cycle in
place of the printed 4 ms; the late rule is a rule for submissions, the lockstep simulator what the privilege costs);
the law's own computation per cycle (MinimalReference.step, the simulator thread's CPU time) is counted apart in the
summary's 'law_cpu_s', the process CPU between two replies in 'cpu_between_replies_over_ms' (reported only).

THE TOKEN PATH (the grader's side is token_state() here; the runtime passes the truth to setup): the token file is
controller/oracle_token in the submission (solution/reference/oracle_token, copied there by solution/solve.sh); the
bytes hashed are the file's bytes with leading and trailing ASCII whitespace removed (Python bytes.strip(): the 64
lowercase hexadecimal characters of a 256-bit random value, ASCII); the grader holds their sha256 as 64 lowercase
hexadecimal characters and a newline (tests/hidden/oracle_token.sha256 in the task folder); a symbolic link, a missing
file or a mismatch gives no truth (a mismatch is reported apart).

THE TRUTH OBJECT (truth_payload; JSON): {'format': TRUTH_FORMAT, 'unit': unit_recipe(...) (the drawn UnitState and
SessionState as their own dicts with digests, the cell, the reply-noise key, the law's parameter object, the registry
record, the write policy), 'site': the true site (F's and S's true and drawn poses, the seats of every grasp event, pad
friction, ambient), 'slots': the slot list (every tray's mass, centre and label), 'jobs': per slot the generator's
targets and swivels, 'run': the run_shift options the grader runs the shift with, 'code': {'ref': the instrument's
folder}}. It holds every fact the reference reads: the joint friction per joint and direction with its profile
(state.plant, state.hs, session.friction_factor), the viscous damping, the torque scale and offset of all sixteen
modules (state.cal k, b with session.b_step), the zero residuals (cal.delta), the mounting (plant.base_poses), the
thermal constants and state (state.thermal, session temperatures, the law), each gripper's friction, each hand's pad
friction (site), F's and S's true poses and the ambient (site, session), every tray's mass, centre and label (slots),
the seat of every grasp event (site seats).
"""
import atexit
import gc
import hashlib
import heapq
import json
import os
import sys
import threading
import time

TRUTH_FORMAT = "openarm-stage05-reference-truth/1"
TOKEN_NAME = "oracle_token"
SIDES = ("left", "right")
N_MODULES = 16
GRIPPERS = (7, 15)
# the hold without the truth: the published example's numbers (the maker's first gain set of its driver configuration)
HOLD_KP = (70.0, 70.0, 70.0, 60.0, 10.0, 10.0, 10.0)
HOLD_KD = (2.75, 2.5, 2.0, 2.0, 0.7, 0.6, 0.5)
HOLD_GRIP_VLIM = 1.0
HOLD_GRIP_ILIM = 0.05
# the run_shift options a truth object may carry (the grader passes the same to run_unit)
RUN_KEYS = ("stop_at_event", "hold_profile", "max_cycles", "ambient_at_presentation", "shelf_exception", "zero_end",
            "keepout_at_removal")
KP_RANGE, KD_RANGE = (0.0, 500.0), (0.0, 5.0)
CHECK_EVERY = 4                 # the reply check's period in cycles (a difference, once there, stays: OURS, for the CPU)
SUMMARY_TAG = "REFERENCE_CONTROLLER_SUMMARY"
MISMATCH_TAG = "REFERENCE_CONTROLLER_REPLY_MISMATCH"


# ================================================================================================ the grader's side
def token_state(controller_dir, sha256_path):
    """(privileged, mismatch): controller_dir/oracle_token, a regular file (not a symbolic
    link), whose stripped bytes hash (sha256, hex) to the stripped content of sha256_path. Missing file or hash file:
    (False, False); a different hash: (False, True) (a mismatch is reported apart)."""
    p = os.path.join(controller_dir, TOKEN_NAME)
    if not os.path.isfile(p) or os.path.islink(p) or not os.path.isfile(sha256_path):
        return False, False
    try:
        with open(p, "rb") as fh:
            tok = fh.read().strip()
        with open(sha256_path) as fh:
            want = fh.read().strip()
    except OSError:
        return False, False
    got = hashlib.sha256(tok).hexdigest()
    return got == want, got != want


def unit_recipe(state, session, cell=None, noise_key=None, law=None, registry=None, policy="stage05"):
    """How the grader built the unit, as JSON (the unit's own dicts with their digests): OpenArmUnit.build(state,
    session, world=shift.make_world(site, cell=cell), noise_key, law=LawPlugins(law), registry, policy,
    truth_profile=shift.trace_fields(cell)). law None = the unit package's parameter file (shift.build_unit's
    LawPlugins()); a registry's units carry the registry's law (Registry.build)."""
    return {"state": state.to_dict(), "session": session.to_dict(), "cell": cell,
            "noise_key": None if noise_key is None else bytes(noise_key).hex(),
            "law": None if law is None else json.loads(json.dumps(law)),
            "registry": None if registry is None else json.loads(json.dumps(registry, default=str)),
            "policy": policy}


def recipe_of_registry_unit(reg, label, cell=None, session=None):
    """The recipe of Registry.build(label, world=..., policy='stage05', truth_profile=...) (noise on: its key)."""
    st, ses = reg.load_unit(label)
    s = ses[0] if session is None else next(x for x in ses if x.session_id == session)
    return unit_recipe(st, s, cell=cell, noise_key=reg.noise_key(label), law=reg.law, registry=reg.record(label))


def build_from_recipe(recipe, site, ref_dir=None):
    """The unit of a recipe on `site` (the grader's harness and the controller's simulator call the same function).
    ref_dir: the instrument's folder, when the instrument is not imported yet (default this file's folder)."""
    if "shift" not in sys.modules:
        _import_instrument(ref_dir)
    import shift as SH
    from openarm_unit import law as lawmod
    from openarm_unit.state import SessionState, UnitState
    from openarm_unit.unit import OpenArmUnit
    st = UnitState.from_dict(recipe["state"])
    ses = SessionState.from_dict(recipe["session"])
    cell = recipe.get("cell")
    nk = recipe.get("noise_key")
    return OpenArmUnit.build(st, ses, world=SH.make_world(site, cell=cell),
                             noise_key=None if nk is None else bytes.fromhex(nk),
                             law=lawmod.LawPlugins(recipe.get("law")), policy=recipe.get("policy", "stage05"),
                             truth_profile=SH.trace_fields(cell), registry=recipe.get("registry"))


def _plain(x):
    return json.loads(json.dumps(x, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))


def truth_payload(recipe, site, slots, jobs, run_kw=None, ref_dir=None):
    """The truth object handed to a token-privileged submission's setup (JSON-able). jobs: per slot the generator's
    job ({'targets', 'swivels'[, 'swivel_fixed']}) or None; run_kw: the run_shift options of RUN_KEYS the grader
    runs the shift with (the same dict goes to run_unit); ref_dir: the instrument's folder (default this file's)."""
    run_kw = dict(run_kw or {})
    bad = set(run_kw) - set(RUN_KEYS)
    if bad:
        raise ValueError(f"run options the reference cannot follow: {sorted(bad)}")
    js = []
    for j in jobs:
        if j is None:
            js.append(None)
            continue
        d = {"targets": j["targets"], "swivels": j.get("swivels")}
        if j.get("swivel_fixed") is not None:
            d["swivel_fixed"] = j["swivel_fixed"]
        js.append(d)
    return _plain({"format": TRUTH_FORMAT, "unit": recipe, "site": site, "slots": slots, "jobs": js, "run": run_kw,
                   "code": {"ref": os.path.abspath(ref_dir or os.path.dirname(os.path.abspath(__file__)))}})


def _import_instrument(ref_dir):
    if ref_dir is None:
        ref_dir = os.path.dirname(os.path.abspath(__file__))
    if ref_dir not in sys.path:
        sys.path.insert(0, ref_dir)
    if "mujoco" not in sys.modules:
        # WORK-AROUND: the runner's child environment sets MUJOCO_GL=osmesa; on a
        # host without OSMesa importing mujoco then fails inside its renderer's import (an AttributeError that
        # mujoco/__init__.py does not catch). The reference renders nothing: the renderer is switched off.
        os.environ["MUJOCO_GL"] = "disable"
    import shift                                     # noqa: F401  (imports openarm_unit first: the numerics)
    return ref_dir


# ================================================================================================ the controller
class Controller:
    """The reference as a submitted controller (see the module text). Constructed with no argument."""

    def __init__(self):
        self._twin = None
        self._hold = None

    def setup(self, unit_record, site, truth=None):
        self._twin = None
        self._hold = None
        if truth is None:
            return {"gripper_mode": {"left": "posforce", "right": "posforce"}}
        self._twin = _Twin(truth)
        return {"gripper_mode": dict(self._twin.modes)}

    def step(self, feedback):
        if self._twin is None:
            return self._hold_step(feedback)
        return self._twin.step(feedback)

    def _hold_step(self, feedback):
        mods = feedback["modules"]
        if self._hold is None:
            self._hold = [float(m["q"]) for m in mods]
        frames = []
        for i in range(N_MODULES):
            if i in GRIPPERS:
                frames.append({"p": self._hold[i], "v_lim": HOLD_GRIP_VLIM, "i_lim": HOLD_GRIP_ILIM})
            else:
                j = i % 8
                frames.append({"p": self._hold[i], "v": 0.0, "kp": HOLD_KP[j], "kd": HOLD_KD[j], "t": 0.0})
        return {"frames": frames, "decline": False}


class _Closing(Exception):
    pass


def _mid(u, lo, hi, bits):
    """The value in the middle of code u's step: the runtime's clip and truncation give the code u again."""
    return (float(u) + 0.5) / float((1 << bits) - 1) * (hi - lo) + lo


class _Twin:
    """The unit's own simulator of the truth, run by ref/shift.run_shift with the reference of record in a thread,
    in lockstep with the wire (the module text's LOCKSTEP)."""

    def __init__(self, truth):
        if not isinstance(truth, dict) or truth.get("format") != TRUTH_FORMAT:
            raise ValueError(f"the truth object is not of format {TRUTH_FORMAT}")
        _import_instrument(truth["code"]["ref"])
        import shift as SH
        import minimal as MR
        from openarm_unit import hostcodec as hc
        self.SH, self.hc = SH, hc
        site, slots, jobs = truth["site"], truth["slots"], truth["jobs"]
        self.unit = build_from_recipe(truth["unit"], site)
        u = self.unit
        B = MR.beliefs_from_truth(u, u.state, u.session, site, slots, spare_coil=True)
        self.inner = MR.MinimalReference(B, slots, jobs)
        Bu = u.bus
        self.lim = [tuple(float(x) for x in Bu.limits(i)) for i in range(N_MODULES)]
        self.key = {}
        for i in range(N_MODULES):
            b, e = int(Bu.modules[i].bus), int(Bu.esc_id[i])
            self.key[(b, e)] = (i, "m")
            self.key[(b, e + hc.POSFORCE_OFFSET)] = (i, "f")
        self.reply_key = {(int(Bu.modules[i].bus), int(Bu.reply_id[i])): i for i in range(N_MODULES)}
        self.last = [None] * N_MODULES
        self.last_cycle = [None] * N_MODULES
        self.cv = threading.Condition()
        self.k_wire = None              # the cycle of the latest wire call
        self.ages_one = True            # whether every module's reply in that call's feedback is one cycle old
        self.result = None              # (cycle, wire reply) of the call at k_wire
        self.ready = False              # the simulator waits at its first step
        self.done = False
        self.err = None
        self.closing = False
        self.pending = None             # the cycle at which a late reply's decline reaches the simulator
        self.modes = None
        self.stats = {"calls": 0, "late_seen": 0, "late_cycles": 0, "mismatch_cycles": 0, "first_mismatch": None,
                      "twin_result": None, "frames_sent_by_twin": 0, "cpu_between_replies_top": [],
                      "cpu_between_replies_over_ms": {"2.5": 0, "3.0": 0, "3.5": 0, "4.0": 0}}
        self.stats["law_cpu_s"] = {"n": 0, "total": 0.0, "max": 0.0, "over_ms": {"0.5": 0, "1.0": 0, "2.0": 0,
                                                                              "4.0": 0}}
        self._top = []                  # the largest process CPU between two replies: [(seconds, cycle)]
        self._t_reply = None
        self.thread = threading.Thread(target=self._run, args=(site, slots, jobs, dict(truth.get("run") or {})),
                                       name="reference-simulator", daemon=True)
        self.thread.start()
        with self.cv:
            while not (self.ready or self.done):
                self.cv.wait()
        if self.err is not None:
            raise self.err
        if not self.ready:
            raise RuntimeError("the reference's simulator ended before its first cycle")
        atexit.register(self._at_exit)
        gc.collect()
        gc.freeze()                     # the plan and the simulator stay out of every later collection
        self._t_reply = time.process_time()

    # -- the simulator's thread
    def _run(self, site, slots, jobs, run_kw):
        try:
            res = self.SH.run_shift(self.unit, site, slots, _Proxy(self), jobs=jobs, **run_kw)
            self.stats["twin_result"] = {k: res.get(k) for k in ("cycles", "state_digest", "motion_digest",
                                                                 "stopped")}
            self.stats["twin_result"]["events"] = res.get("events")
            self.stats["twin_result"]["declined"] = [p.get("declined") for p in res.get("presentations", [])]
        except _Closing:
            pass
        except BaseException as e:      # noqa: BLE001  (handed to the wire call that waits)
            self.err = e
        finally:
            with self.cv:
                self.done = True
                self.cv.notify_all()

    def _absorb(self, c, replies):
        """The simulator's state replies of cycle c - 1, kept raw per module (decoded only when checked)."""
        for f in replies or ():
            i = self.reply_key.get((f.bus, f.can_id))
            if i is None or len(f.data) < 8:
                continue
            self.last[i] = bytes(f.data)
            self.last_cycle[i] = c - 1

    def twin_step(self, c, sig, replies):
        """run_shift's reference.step(c, sig, replies) in the simulator: the frames the modules receive at cycle c."""
        with self.cv:
            if not self.ready:
                self.ready = True
                self.cv.notify_all()
            while not self.closing and (self.k_wire is None or self.k_wire < c):
                self.cv.wait()
            if self.closing and (self.k_wire is None or self.k_wire < c):
                raise _Closing()
            kw = self.k_wire
        due = self.pending is not None and self.pending == c
        if due:
            self.pending = None
        self._absorb(c, replies)
        if c < kw:                                     # a held late cycle: the controller was not called at c
            return [], bool(due)
        t_law = time.thread_time()
        frames, decline = self.inner.step(c, sig, replies)
        self._law_time(time.thread_time() - t_law)
        wire = {"frames": self._to_wire(frames), "decline": bool(decline)}
        with self.cv:
            self.result = (c, wire)
            self.cv.notify_all()
            while not self.closing and self.k_wire == c:
                self.cv.wait()
            kn, ages_one = self.k_wire, self.ages_one
        if self.closing and kn == c:
            kn, ages_one = c + 1, True                 # the shift's last cycle: its frames were sent
        if kn == c + 1 and ages_one:
            self.stats["frames_sent_by_twin"] += 1
            return frames, bool(decline) or due
        # late by kn - c cycles (run_unit's LATE rule): nothing sent from c to kn - 1; the decline at kn (a reply
        # late by one cycle: kn == c + 1 with every age 2)
        self.stats["late_seen"] += 1
        self.stats["late_cycles"] += kn - c
        if decline:
            self.pending = kn
        return [], bool(due)

    def _to_wire(self, frames):
        out = [None] * N_MODULES
        hc = self.hc
        for f in frames:
            i, kind = self.key[(f.bus, f.can_id)]
            d = f.data
            if kind == "m":
                pm, vm, tm = self.lim[i]
                out[i] = {"p": _mid((d[0] << 8) | d[1], -pm, pm, 16),
                          "v": _mid((d[2] << 4) | (d[3] >> 4), -vm, vm, 12),
                          "kp": _mid(((d[3] & 0xF) << 8) | d[4], KP_RANGE[0], KP_RANGE[1], 12),
                          "kd": _mid((d[5] << 4) | (d[6] >> 4), KD_RANGE[0], KD_RANGE[1], 12),
                          "t": _mid(((d[6] & 0xF) << 8) | d[7], -tm, tm, 12)}
            else:
                p, _, _ = hc.unpack_posforce(d)
                out[i] = {"p": float(p), "v_lim": (float(d[4] | (d[5] << 8)) + 0.5) / 100.0,
                          "i_lim": (float(d[6] | (d[7] << 8)) + 0.5) / 10000.0}
        return out

    # -- the wire's side
    def step(self, feedback):
        k = int(feedback["cycle"])
        with self.cv:
            if self.done:
                if self.err is not None:
                    raise self.err
                return {"frames": [None] * N_MODULES, "decline": False}
            self.k_wire = k
            self.ages_one = all(int(m.get("age", 0)) == 1 for m in feedback.get("modules") or ())
            self.result = None
            self.cv.notify_all()
            while not self.done and (self.result is None or self.result[0] != k):
                self.cv.wait()
            if self.result is None or self.result[0] != k:
                if self.err is not None:
                    raise self.err
                return {"frames": [None] * N_MODULES, "decline": False}
            wire = self.result[1]
        self.stats["calls"] += 1
        if k % CHECK_EVERY == 0 or not self.ages_one:
            self._check(k, feedback)
        self._account(k)
        return wire

    def _law_time(self, dt):
        """The law's own computation of one cycle (MinimalReference.step in the simulator's thread: that thread's CPU
        time), apart from the simulator's step and the hand-off (reported only)."""
        lt = self.stats["law_cpu_s"]
        lt["n"] += 1
        lt["total"] += dt
        if dt > lt["max"]:
            lt["max"] = dt
        for key in lt["over_ms"]:
            if dt * 1e3 > float(key):
                lt["over_ms"][key] += 1

    def _account(self, k):
        """This process's CPU since the previous reply (what the runner charges a call, less the JSON of the reply):
        the twenty largest with their cycles, and the counts above 2.5 / 3.0 / 3.5 / 4.0 ms (reported only)."""
        t = time.process_time()
        dt = t - self._t_reply
        self._t_reply = t
        over = self.stats["cpu_between_replies_over_ms"]
        for key in over:
            if dt * 1e3 > float(key):
                over[key] += 1
        if len(self._top) < 20:
            heapq.heappush(self._top, (dt, k))
        elif dt > self._top[0][0]:
            heapq.heapreplace(self._top, (dt, k))

    def _check(self, k, feedback):
        """The feedback against the simulator's own last replies (bit for bit; age as the runtime computes it)."""
        mods = feedback.get("modules") or []
        bad = []
        self.stats["checked_cycles"] = self.stats.get("checked_cycles", 0) + 1
        for i in range(N_MODULES):
            raw = self.last[i]
            r = None if raw is None else self.hc.unpack_state(raw, self.lim[i])
            m = mods[i] if i < len(mods) else None
            if r is None or m is None:
                bad.append(i)
                continue
            if not (r["q"] == m["q"] and r["dq"] == m["dq"] and r["tau"] == m["tau"]
                    and int(r["t_mos"]) == m["t_mos"] and int(r["t_rotor"]) == m["t_rotor"]
                    and int(r["status"]) == m["status"] and k - self.last_cycle[i] == m["age"]):
                bad.append(i)
        if bad:
            self.stats["mismatch_cycles"] += 1
            if self.stats["first_mismatch"] is None:
                self.stats["first_mismatch"] = {"cycle": k, "modules": bad}
                sys.stderr.write(f"{MISMATCH_TAG} " + json.dumps({"cycle": k, "modules": bad}) + "\n")
                sys.stderr.flush()

    def _at_exit(self):
        with self.cv:
            self.closing = True
            self.cv.notify_all()
        self.thread.join(timeout=3.0)
        self.stats["cpu_between_replies_top"] = sorted(([round(d, 6), c] for d, c in self._top), reverse=True)
        self.stats["gc"] = gc.get_stats()
        try:
            sys.stderr.write(f"{SUMMARY_TAG} " + json.dumps(self.stats, default=str) + "\n")
            sys.stderr.flush()
        except Exception:                              # noqa: BLE001
            pass


class _Proxy:
    """The `reference` object the simulator's run_shift drives: MinimalReference's calls, its step through the
    lockstep."""
    jobs = None

    def __init__(self, twin):
        self.t = twin

    def setup(self, unit, site, slots):
        out = self.t.inner.setup(unit, site, slots)
        self.t.modes = dict(out["gripper_mode"])
        return out

    def start(self, unit):
        return self.t.inner.start(unit)

    def slot_plan_lengths(self):
        return self.t.inner.slot_plan_lengths()

    def step(self, c, sig, replies):
        return self.t.twin_step(c, sig, replies)

    def on_trace(self, recs):
        return self.t.inner.on_trace(recs)

    def figures(self):
        return self.t.inner.figures()


def summary_from_log(text):
    """The controller's last summary line in a runner's standard-error log (dict), or None."""
    out = None
    for ln in text.splitlines():
        if ln.startswith(SUMMARY_TAG + " "):
            out = json.loads(ln[len(SUMMARY_TAG) + 1:])
    return out


def _now():
    return time.strftime("%Y-%m-%d %H:%M:%S %Z")
