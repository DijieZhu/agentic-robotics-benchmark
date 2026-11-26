"""Hidden grader for kitchen_appliance_tending (closed loop, hidden appliance units).

Runs as root in a SEPARATE verifier sandbox: no lab and no agent process exist there; the engine restores only
the declared artifact /home/user/submission. Structure and safeguards follow arm_hidden_mechanism_box's grader:
it reclaims root's PATH directories (robotics_grader.stage.secure_exec_path) before anything else, stops any lab
or agent-UID process, hands the restored submission (size-capped) to the agent account in place, starts
`bash run.sh` through nonet_exec.py (a seccomp filter: only AF_UNIX sockets) at the agent's UID, and speaks the
kit wire (Solver, per-request deadline, restarts; a ProbeLock, repeated `t` with a perturbed copy in a hidden
order, that does not count a pair in which both replies command no motion: IdleAwareProbeLock).

Episodes: the fixed hidden set tests/hidden/episodes.json, the same for every submission: `episodes_per_unit`
episodes on every hidden unit (tests/units.json, models in tests/units/), each a 64-bit seed (unit placement,
initial control states, request queue, lighting, camera offsets and sensor noise), run in the file's order
(selected and shuffled by tests/review/select_set.py, the record in tests/hidden/episode_selection.json). The
probe lock draws which steps are probed, the order of a pair and the perturbation from one stream per episode
(probe_stream): the kit's `seed_rng` keyed by a BLAKE2b hash of a domain tag, the episode's index in the set and
its hidden seed. An episode's probe schedule is therefore a function of the episode alone, not of the controller's
replies, of how many steps earlier episodes took, of the clock or of the process. The grader adds no run-to-run
randomness of its own, so a deterministic controller that stays inside the time limits (the step deadline, which
also bounds the controller's start, and the budget on the controller's own time are wall-clock) gets one result: with
one OS-entropy
stream for the whole run, where the lock probed changed from run to run, and a controller that advanced its state
on the probe copy scored between 0.278 and 0.407 in seven gradings of one fixed set. The hidden seed stays the only
secret; the tag keeps the stream apart from everything the environment draws from the seed. The lock's counts are
run-wide and are the only thing about it that is logged or written.
episode.py owns the requests and the credit (byte-identical to the agent's copy);
appliance_env.py is the environment (byte-identical too). The metric `mean_request_credit` is the mean over
episodes of the mean request credit (reached 0.5 + completed 0.5). Each episode's requests and their outcomes are
logged on stderr with the episode's index in the set (never its seed).

The budget (grader_config.json `grader_budget_s`) is the controller's own time: what the grader waits for READY, for
every reset and for every reply (Clock), plus `image_cost_s` for every image an applied action asks for, a fixed price
whatever this machine takes to draw it. The simulation and the drawing are not charged. The episode running when the
budget runs out is aborted and the later ones are not run: they score 0.

The controller's processes run only while a request is outstanding (Pauser): every process of the agent's account that
runs is stopped (SIGSTOP) once a reply has been read (or READY seen) and continued before the next request is written,
so the simulation and the drawing never share the CPUs with it and it cannot compute ahead of a request. A pause holds
only once /proc shows every thread of every process of the account stopped; processes that do not stop within
FREEZE_TIMEOUT_S, or more than MAX_PROCESSES processes of the account at a pause, abort the episode (score 0).
SIGSTOP holds only until the next SIGCONT, and a process can arrange its own (a POSIX timer that signals it), so the
pause is checked again before the next request is written: every thread that was stopped must still be there and
stopped, with the scheduler's record of its run time and of its turns on a CPU unchanged, and no thread may have
appeared. Otherwise a process of the account ran during the pause and the episode is aborted (score 0). The time it
takes to bring the account to a stop is the controller's own (a chain of short-lived processes lengthens it) and is
charged; the simulation and the drawing that follow are not.

Numerics: the environment's operational-space controller inverts two matrices with LAPACK in every physics substep, and
NumPy's OpenBLAS picks its kernels by CPU. The image pins them (OPENBLAS_CORETYPE=Haswell, OPENBLAS_L2_SIZE=256, with
NumPy's AVX-512 loops off and one BLAS thread: environment/Dockerfile, also for a controller's own Python);
tests/numerics_check.py runs a short scripted simulation and a set of NumPy, SciPy and OpenCV results, and the grader
compares their fingerprint with tests/numerics.json before grading. A machine that does not reproduce it is a task
error.

Oracle token: a submission whose .oracle_token hashes to tests/hidden/oracle_token.sha256 is the reference;
only then does every observation also carry `truth` (the unit's placement jitter, its joint values, the
microwave's flag). An untokened submission never sees it. Every path agent behaviour can reach is charged as
a score, never a crash.

Images: appliance_env.py draws the cameras with kitchen_render.py (MuJoCo's Filament renderer on Mesa's lavapipe,
the room bundle in KITCHEN_RENDER_DIR), exactly as the dev sim and the lab do. Before the first episode a render
self-test draws the three cameras of one episode (the first unit of the set at a fixed seed, not a graded episode)
and checks their shapes, that none is flat and that their brightness is sane; the metric `render_selftest` records
that it passed. A renderer failure, in the self-test or at any later image (appliance_env.RenderError: a missing
or broken room-bundle file, an error while drawing, a call that does not return), is the grader's own error: the
run ends as a task error (robotics_grader.stage.fail: no reward written, exit status 1), never as a controller fault
or a 0 for the submission. The controller's processes are stopped while an image is drawn, run at the lowest
priority if they continue themselves, and end the episode at the next request if they did. A Vulkan backend that cannot start at all aborts the grader's process inside Filament
(SIGABRT, before Python can catch anything): no reward is written either; the image's build-time render check
(environment/render_check/) proves the driver for every account that renders.
"""

import hashlib
import json
import math
import os
import pwd
import signal
import stat
import subprocess
import sys
import time
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402

import appliance_env as E  # noqa: E402
import episode as EP  # noqa: E402
from robotics_grader import (  # noqa: E402
    ProbeLock,
    Solver,
    SolverAborted,
    SolverMissing,
    agent_user,
    ratio_rewards,
    read_anchor,
    seed_rng,
    stage_config,
    write_verdict,
    zero_verdict,
)
from robotics_grader.stage import fail, secure_exec_path  # noqa: E402

STAGE = os.path.dirname(os.path.abspath(__file__))
MAX_SUBMISSION_BYTES = 200 * 1024 * 1024
MAX_SUBMISSION_FILES = 5000
NONET = os.path.join(STAGE, "nonet_exec.py")
UNITS = os.path.join(STAGE, "units.json")
TOKEN_SHA = os.path.join(STAGE, "hidden", "oracle_token.sha256")
EPISODES = os.path.join(STAGE, "hidden", "episodes.json")
STILL = 1e-9                # a reply whose six motion numbers are all within this of zero commands no motion
MAX_LINE = 1 << 22          # an observation with three images is about 1 MB; replies are short
SELFTEST_SEED = 1           # the render self-test's episode (not one of the graded seeds)
SELFTEST_SHAPES = {"left": (240, 320, 3), "right": (240, 320, 3), "wrist": (192, 256, 3)}
PROBE_TAG = b"kitchen_appliance_tending verify probe schedule 1"   # probe_stream's domain tag
NUMERICS = os.path.join(STAGE, "numerics_check.py")
NUMERICS_REF = os.path.join(STAGE, "numerics.json")
NUMERICS_TIMEOUT_S = 300
FREEZE_TIMEOUT_S = 5.0      # every thread of the agent's account must show as stopped within this
MAX_PROCESSES = 512         # processes of the agent's account at a pause (the instruction states it)
STOPPED_STATES = ("T", "t")  # a thread that is stopped (or traced and stopped)
GONE_STATES = ("Z", "X")     # a thread that has exited: it runs nothing


def log(msg):
    print("[grader] " + msg, file=sys.stderr, flush=True)


def _agent_ids():
    pw = pwd.getpwnam(agent_user())
    return pw.pw_uid, pw.pw_gid


def _agent_pids(uid):
    pids = []
    for name in os.listdir("/proc"):
        if not name.isdigit():
            continue
        try:
            with open("/proc/{}/status".format(name)) as f:
                zombie = False
                for line in f:
                    if line.startswith("State:"):
                        # a zombie runs no code; it only waits to be reaped (r5: a zombie made the
                        # bounded kill report cleanup_failed in 1 of 24 oracle runs under load)
                        zombie = line.split()[1].startswith(("Z", "X"))
                    elif line.startswith("Uid:"):
                        if int(line.split()[1]) == uid and not zombie:
                            pids.append(int(name))
                        break
        except (OSError, ValueError, IndexError):
            continue
    return pids


def kill_agent_processes(uid, rounds=20, pause=0.25):
    """Kill every process of the agent account; True when none is left (bounded)."""
    for _ in range(rounds):
        pids = _agent_pids(uid)
        if not pids:
            return True
        for pid in pids:
            try:
                os.kill(pid, signal.SIGKILL)
            except OSError:
                pass
        time.sleep(pause)
    return not _agent_pids(uid)


def _account_pids(uid):
    """PIDs owned by `uid` (the owner of /proc/<pid>), zombies included: one stat per process."""
    pids = []
    for name in os.listdir("/proc"):
        if name.isdigit():
            try:
                if os.stat("/proc/" + name).st_uid == uid:
                    pids.append(int(name))
            except OSError:
                continue
    return pids


def _thread_states(pid):
    """The state letter of every thread of `pid` (empty once the process is gone)."""
    states = []
    try:
        tids = os.listdir("/proc/{}/task".format(pid))
    except OSError:
        return states
    for tid in tids:
        try:
            with open("/proc/{}/task/{}/stat".format(pid, tid)) as f:
                states.append(f.read().rsplit(")", 1)[1].split()[0])
        except (OSError, IndexError):
            continue
    return states


def _runs(pid):
    """True while a thread of `pid` is neither stopped nor gone."""
    return any(st not in STOPPED_STATES and st not in GONE_STATES for st in _thread_states(pid))


def _alive(pid):
    """True while `pid` has a thread that has not exited."""
    return any(st not in GONE_STATES for st in _thread_states(pid))


def _thread_records(pids):
    """{(pid, tid): (state, run time, turns on a CPU)} for every thread of `pids`: the state letter from the thread's
    stat and the first and third numbers of its schedstat (ns it has run, times it was given a CPU). A kernel without
    schedstat gives the thread's user and system ticks instead. A stopped thread's record does not change."""
    records = {}
    for pid in pids:
        try:
            tids = os.listdir("/proc/{}/task".format(pid))
        except OSError:
            continue
        for tid in tids:
            base = "/proc/{}/task/{}/".format(pid, tid)
            try:
                with open(base + "stat") as f:
                    fields = f.read().rsplit(")", 1)[1].split()
                state, ran = fields[0], (fields[11], fields[12])
            except (OSError, IndexError):
                continue
            try:
                with open(base + "schedstat") as f:
                    sched = f.read().split()
                ran = (sched[0], sched[2])
            except (OSError, IndexError):
                pass
            records[(pid, int(tid))] = (state,) + ran
    return records


def _pause_broken(before, now):
    """What shows that a process ran between two readings of a paused account, or None. A thread that was stopped must
    still be there, stopped, with the same record; no thread may have appeared. A thread that had already exited
    (a zombie) may be reaped meanwhile."""
    for key, rec in now.items():
        if key not in before:
            return "{} appeared".format(_describe(key[0]))
        was = before[key]
        if rec == was or (was[0] in GONE_STATES and rec[1:] == was[1:]):
            continue
        return "{} ran".format(_describe(key[0]))
    for key, rec in before.items():
        if key not in now and rec[0] not in GONE_STATES:
            return "process {} is gone".format(key[0])
    return None


def _describe(pid):
    try:
        with open("/proc/{}/cmdline".format(pid), "rb") as f:
            return "{} {}".format(pid, f.read(120).replace(b"\0", b" ").decode("utf-8", "replace").strip())
    except OSError:
        return "{} (gone)".format(pid)


class Clock(object):
    """The controller's own time (s): what the grader waited for READY, for each reset and for each reply and what it
    took to stop the account afterwards (`timed`), and what run_episodes adds for the images an applied action asks
    for. The simulation and the drawing of the images happen outside `timed`."""

    def __init__(self):
        self.s = 0.0

    def timed(self, fn, *args, **kwargs):
        t0 = time.monotonic()
        try:
            return fn(*args, **kwargs)
        finally:
            self.s += time.monotonic() - t0


class Pauser(object):
    """Stops the agent's account between requests (longtail_manipulation_recovery's rule). `freeze` sends SIGSTOP to
    every process of the account with a running thread and confirms it thread by thread: a scan that finds one
    running signals it again and scans anew, so a fork chain is caught process by process; the pause holds when two
    consecutive scans find the same processes, none running (a stopped process cannot start another). Not reached
    within FREEZE_TIMEOUT_S: `unstoppable` names them. `thaw` continues exactly the processes this object stopped
    (a process the image itself left stopped is never continued). `count` is the number of the account's processes
    alive at the last scan."""

    def __init__(self, uid):
        self.uid = uid
        self._stopped = set()
        self.unstoppable = []
        self.count = 0
        self.peak = 0
        self._held = None          # the threads' records once the pause holds (None: nothing to compare)
        self.resumed = 0           # pauses during which a process of the account ran

    def freeze(self):
        previous = None
        deadline = time.monotonic() + FREEZE_TIMEOUT_S
        self.unstoppable = []
        self._held = None
        while True:
            pids = _account_pids(self.uid)
            running = [p for p in pids if _runs(p)]
            for pid in running:
                try:
                    os.kill(pid, signal.SIGSTOP)
                    self._stopped.add(pid)
                except OSError:
                    pass
            current = frozenset(pids)
            if not running and current == previous:
                break
            previous = None if running else current
            if time.monotonic() > deadline:
                self.unstoppable = [_describe(p) for p in running[:5]]
                break
            if running:
                time.sleep(0.0005)
        self.count = sum(1 for p in pids if _alive(p))
        self.peak = max(self.peak, self.count)
        if not self.unstoppable:
            # /proc shows a thread as stopped an instant before it leaves the CPU: the records are taken once two
            # readings a millisecond apart agree
            held = _thread_records(pids)
            while time.monotonic() <= deadline:
                time.sleep(0.001)
                again = _thread_records(_account_pids(self.uid))
                if again == held:
                    self._held = held
                    break
                held = again
            else:
                self.unstoppable = ["the account's threads keep changing"]
        return self.count

    def check_held(self):
        """Before the next request: did the pause hold? Raises SolverAborted if a process of the account ran since
        the pause was confirmed (continued by a signal it arranged for itself) or a new one appeared."""
        held, self._held = self._held, None
        if held is None:
            return
        why = _pause_broken(held, _thread_records(_account_pids(self.uid)))
        if why:
            self.resumed += 1
            raise SolverAborted("a process of the controller's account ran during a pause ({})".format(why))

    def thaw(self):
        self._held = None
        for pid in sorted(self._stopped):
            try:
                os.kill(pid, signal.SIGCONT)
            except OSError:
                pass
        self._stopped = set()

    def pause_or_abort(self):
        self.freeze()
        if self.unstoppable:
            raise SolverAborted("processes of the controller's account did not stop within {:g} s ({})".format(
                FREEZE_TIMEOUT_S, "; ".join(self.unstoppable)))
        if self.count > MAX_PROCESSES:
            raise SolverAborted("more than {} processes in the controller's account".format(MAX_PROCESSES))


def check_numerics():
    """Run numerics_check.py in a fresh interpreter (a short scripted run of the simulation and a set of NumPy, SciPy
    and OpenCV results from a Python started as a controller is) and compare its fingerprint with tests/numerics.json.
    A machine that does not reproduce the recorded one would run every controller along another trajectory, so it
    must not grade: a task error."""
    with open(NUMERICS_REF) as fh:
        want = json.load(fh)["fingerprint"]
    try:
        proc = subprocess.run([sys.executable, "-I", NUMERICS], cwd=STAGE, stdout=subprocess.PIPE,
                              stderr=subprocess.DEVNULL, timeout=NUMERICS_TIMEOUT_S, env=dict(os.environ))
        report = json.loads(proc.stdout.decode("utf-8", "replace").strip().splitlines()[-1])
        got = str(report["fingerprint"])
    except (OSError, subprocess.SubprocessError, ValueError, KeyError, IndexError) as exc:
        fail("the numerics check did not run: {}: {}".format(type(exc).__name__, str(exc)[:200]))
    if got != want:
        fail("this machine does not reproduce the recorded numerics (tests/numerics.json): got {}, recorded {}; "
             "differing parts {}; OPENBLAS_CORETYPE={!r}. Grading here would run every controller along another "
             "trajectory".format(got[:16], want[:16], report.get("differs"), os.environ.get("OPENBLAS_CORETYPE")))
    return got


def _proc_state(pid):
    """The /proc state letter of `pid` ("X" once it is gone)."""
    try:
        with open("/proc/{}/status".format(pid)) as f:
            for line in f:
                if line.startswith("State:"):
                    return line.split()[1]
    except (OSError, IndexError):
        pass
    return "X"


def _still_running(uid, bound_s=3.0):
    """SIGSTOP every live process of `uid` until a pass finds none still running (a respawn that appears meanwhile is
    stopped by the next pass) or `bound_s` elapsed; returns the PIDs still running (empty: every one is stopped)."""
    deadline = time.monotonic() + float(bound_s)
    while True:
        for pid in _agent_pids(uid):
            try:
                os.kill(pid, signal.SIGSTOP)
            except OSError:
                pass
        time.sleep(0.05)
        running = [pid for pid in _agent_pids(uid) if _proc_state(pid) not in ("T", "t", "X")]
        if not running or time.monotonic() > deadline:
            return running


def stop_agent_processes(uid):
    """kill_agent_processes, then: True when none is left or every one left is stopped. The base image's supervisord
    respawns the agent account's desktop as fast as it is killed, so in the separate verifier a process that outlives
    the bounded kill is the image's own; it is stopped (a stopped process does not exit, so it is not respawned) and
    only one still running once stopped fails the run (cleanup_failed), as before."""
    if kill_agent_processes(uid):
        return True
    running = _still_running(uid)
    log("{} process(es) of the agent account outlived the kill, {} still running after the freeze".format(
        len(_agent_pids(uid)), len(running)))
    return not running


class GuardedSolver(Solver):
    """The kit Solver, spawned through the seccomp launcher, with a reply-line cap."""

    guard = True
    nproc = 0

    def _spawn(self):
        uid, gid = _agent_ids()
        cmd = [sys.executable, "-I", NONET, "--uid", str(uid), "--gid", str(gid), "--rlimit-nproc", str(self.nproc),
               "--cwd", self.submission, "--", "bash", self.entry]
        if not self.guard:
            cmd = ["/usr/sbin/runuser", "-u", agent_user(), "--", "/usr/bin/env", "-i",
                   "PATH=/opt/venv/bin:/usr/local/bin:/usr/bin:/bin", "HOME=/home/user", "LANG=C.UTF-8",
                   "bash", self.entry]
        try:
            return subprocess.Popen(cmd, cwd=self.submission, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                    stderr=subprocess.DEVNULL, universal_newlines=True, bufsize=1,
                                    start_new_session=True, env={"PATH": "/usr/bin:/bin"})
        except OSError as exc:
            raise SolverAborted("solver could not be started: {}".format(exc))

    @staticmethod
    def _pump(proc, lines):
        while True:
            try:
                line = proc.stdout.readline(MAX_LINE + 1)
            except (ValueError, OSError, UnicodeDecodeError):
                break
            if not line:
                break
            if len(line) > MAX_LINE and not line.endswith("\n"):
                break          # oversize reply: treated as a closed wire (episode aborts)
            lines.put(line)
        lines.put(None)


def egress_self_test(uid, gid, nproc=0):
    """True if a probe started through the launcher cannot open an internet socket."""
    probe = ("import socket,sys\n"
             "bad=0\n"
             "for fam in (socket.AF_INET, socket.AF_INET6, 40):\n"
             "    try:\n"
             "        socket.socket(fam, socket.SOCK_STREAM).close(); bad=1\n"
             "    except OSError:\n"
             "        pass\n"
             "try:\n"
             "    socket.socket(socket.AF_UNIX, socket.SOCK_STREAM).close()\n"
             "except OSError:\n"
             "    bad=1\n"
             "sys.exit(3 if bad else 0)\n")
    try:
        r = subprocess.run([sys.executable, "-I", NONET, "--uid", str(uid), "--gid", str(gid),
                            "--rlimit-nproc", str(nproc), "--cwd", "/",
                            "--", sys.executable, "-I", "-c", probe],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=60,
                           env={"PATH": "/usr/bin:/bin"})
        return r.returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def agent_inet_sockets(uid):
    """Count agent-UID internet sockets with a non-loopback peer (detection fallback)."""
    n = 0
    for name in ("tcp", "tcp6", "udp", "udp6", "raw", "raw6"):
        try:
            with open("/proc/net/" + name) as f:
                next(f, None)
                for line in f:
                    parts = line.split()
                    if len(parts) < 8 or int(parts[7]) != uid:
                        continue
                    rem = parts[2].split(":")[0]
                    if rem.strip("0") == "" or rem in ("0100007F", "00000000000000000000000001000000"):
                        continue
                    n += 1
        except (OSError, ValueError, StopIteration):
            continue
    return n


def stop_lab():
    """Stop the lab service (rev-leak R14) before anything else; True when it is gone."""
    try:
        from alelab_stop import stop_all
        stop_all(agent_user=agent_user())
        return True
    except Exception as exc:  # noqa: BLE001 - a failed stop is recorded, the agent kill still runs
        log("lab stop: {}".format(type(exc).__name__))
        return False


def stage_submission(uid, gid):
    """Make the restored submission the agent account's own folder, in place at ~user/submission: the
    folder the instruction names, where run.sh is started and where the controller may write, as it can in
    the lab and the practice grader. Ownership changes never follow a symlink (lchown, a walk that does not
    follow links); size and file count are capped. Returns the path, or None if there is no usable
    submission."""
    src = os.path.join(os.path.expanduser("~" + agent_user()), "submission")
    if not os.path.isdir(src) or os.path.islink(src):
        return None
    total = files = 0
    for dp, dns, fns in os.walk(src, followlinks=False):
        for fn in fns:
            fp = os.path.join(dp, fn)
            files += 1
            if not os.path.islink(fp):
                try:
                    total += os.lstat(fp).st_size
                except OSError:
                    pass
        if files > MAX_SUBMISSION_FILES or total > MAX_SUBMISSION_BYTES:
            return None
    for dp, dns, fns in os.walk(src, followlinks=False):
        for name in dns + fns:
            os.lchown(os.path.join(dp, name), uid, gid)
    os.lchown(src, uid, gid)
    os.chmod(src, 0o700)
    return src


def oracle_token_ok(workdir):
    """True when the submission's .oracle_token (a regular file, not followed if a link, at most 256 bytes)
    hashes to the recorded value."""
    path = os.path.join(workdir, ".oracle_token")
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except OSError:
        return False
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            return False
        data = os.read(fd, 257)
    finally:
        os.close(fd)
    if len(data) > 256:
        return False
    with open(TOKEN_SHA) as f:
        want = f.read().strip()
    return hashlib.sha256(data.strip()).hexdigest() == want


class IdleAwareProbeLock(ProbeLock):
    """The kit's ProbeLock, except that a pair in which BOTH replies command no motion (the first six numbers all
    within STILL of zero) is not counted, neither in the total nor as passed: a controller that holds still on a
    unit it has given up shows nothing either way. Every other pair is the kit's (`ProbeLock.record`)."""

    def __init__(self, *args, **kwargs):
        ProbeLock.__init__(self, *args, **kwargs)
        self.idle_skipped = 0

    @staticmethod
    def still(action):
        if not isinstance(action, list) or len(action) < 6:
            return False
        for x in action[:6]:
            if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) or abs(x) > STILL:
                return False
        return True

    def record(self, action_probe, action_real):
        if self.still(action_probe) and self.still(action_real):
            self.idle_skipped += 1
            return False
        return ProbeLock.record(self, action_probe, action_real)

    def rewards(self):
        out = ProbeLock.rewards(self)
        out["probes_idle_skipped"] = float(self.idle_skipped)
        return out


def probe_stream(index, seed):
    """The probe lock's stream for episode `index` of the set (its position in the graded order), whose hidden 64-bit
    seed is `seed`: the kit's seed_rng keyed by BLAKE2b over PROBE_TAG, the index and the seed, so a function of
    (index, seed) alone. The environment and the episode draw from the seed only through NumPy generators seeded
    with its two 32-bit halves and a small tag (ApplianceEnv.reset, Episode.reset); this hash shares nothing with
    them. The kit's documented use of `entropy` is its self-test. Four other verify stages on main also pass
    per-episode entropy to seed_rng, but they take it from a separate hidden key in their fixed hidden set:
    fault_adaptive_spacecraft_docking and object_finding_and_organizing_on_messy_desk mix the set's probe_key with
    the episode's seed ((probe_key << 64) | seed); driving_safety_supervisor and hazard_bay_build use each
    episode's (mission's) own lock_key. This task needs no such key, because its only secret is the episode's
    seed: PROBE_TAG is a domain separator, not a secret."""
    digest = hashlib.blake2b(b"%s|%d|%d" % (PROBE_TAG, int(index), int(seed)), digest_size=32).digest()
    return seed_rng(entropy=int.from_bytes(digest, "little"))


def ask(lock, solver, ep, t, obs, unit, truth, call=None):
    """Step `t`'s action: one request, or a probe pair when the lock's stream says so. Every step draws once from
    the stream (lock.due); a probed step draws twice more, both before either request is written: the seed of the
    perturbation, then the order of the pair (ProbeLock.probe). `call` runs the exchange (run_episodes passes the
    one that continues the controller's processes, charges the wait to its clock and stops them again). The draws a step consumes never depend on the
    replies, and a pair that is not counted (both replies command no motion) has consumed the same draws as a
    counted one."""
    call = call or (lambda fn, *a, **k: fn(*a, **k))
    w = wire_obs(ep, obs, unit, truth)
    if lock.due():
        other = ep.env.perturbed_observation(obs, np.random.default_rng(lock.rng.seed(bits=64)))
        return call(lock.probe, solver, t, w, perturbed=wire_obs(ep, other, unit, truth))
    return call(solver.act, t, w)


def render_selftest(spec, seed=SELFTEST_SEED):
    """The three cameras of one episode, before any episode is graded: the right shapes, not flat (a standard
    deviation of at least 8 levels; the sensor noise alone gives 2) and a mean brightness of 20-235 levels. Raises
    E.RenderError. The first image of a process also pays the renderer's start (several seconds), so the graded
    episodes do not."""
    env = E.ApplianceEnv(spec, None, render=True)
    try:
        obs = env.reset(seed)
        for cam, shape in sorted(SELFTEST_SHAPES.items()):
            img = obs[cam]
            mean, std = float(img.mean()), float(img.std())
            if tuple(img.shape) != shape or img.dtype != np.uint8 or std < 8.0 or not 20.0 <= mean <= 235.0:
                raise E.RenderError("{} image: shape {}, dtype {}, mean {:.1f}, std {:.1f}".format(
                    cam, tuple(img.shape), img.dtype, mean, std))
    finally:
        env.close()


def load_episode_set(config, units, path=EPISODES):
    """The fixed hidden set: [(unit, seed)] in the graded order. `episodes_per_unit` and the units must agree
    with it (a task error otherwise, never the agent's)."""
    with open(path) as fh:
        data = json.load(fh)
    eps = [(str(e["unit"]), int(e["seed"])) for e in data["episodes"]]
    reps = int(config.get("episodes_per_unit", 2))
    counts = Counter(u for u, _s in eps)
    if set(counts) != set(units) or any(n != reps for n in counts.values()):
        raise RuntimeError("tests/hidden/episodes.json does not hold {} episodes on each of the units".format(reps))
    if any(not 0 <= s < 2 ** 64 for _u, s in eps) or len(set(s for _u, s in eps)) != len(eps):
        raise RuntimeError("tests/hidden/episodes.json: a seed outside 64 bits or a repeated seed")
    return eps


def log_requests(index, unit, sc):
    """One line per request of an episode: what was asked and what came of it (never the seed)."""
    for r in sc["requests"]:
        target = r["target"]
        target = "+".join(str(x) for x in target) if isinstance(target, (list, tuple)) else str(target)
        log("  episode {} ({}) request {}: {} {} reached={} completed={} open={} close={} -> {}".format(
            index, unit, r["id"], r["what"], target, int(bool(r["reached"])), int(bool(r["completed"])),
            r["t_open"], r["t_close"], r.get("outcome", "")))


def wire_obs(ep, obs, uid, truth):
    w = E.to_wire(obs)
    w["unit"] = uid
    if truth:
        w["truth"] = ep.env.truth()
    return w


def valid_action(action):
    if not isinstance(action, list) or len(action) != 8:
        return False
    for x in action:
        if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x):
            return False
    return 0 <= action[7] <= 7 and float(action[7]).is_integer()


def run_episodes(config, uid, egress_mode, t_start, specs, workdir, truth, episodes, clock=None, pauser=None):
    """Run `episodes` ([(unit, seed)], the fixed set in its order) over one controller. Returns (scores, aborted,
    faults, per_unit, per_request, egress_hits, budget_exhausted, lock, records); a record per episode: its index
    in the set, unit, score, reason, steps and the requests' outcomes (episode.score()). During episode n the
    probe lock draws from probe_stream(n, seed); its counts run over all episodes. `clock` carries the controller's
    own time (the budget is compared with it; `t_start` is no longer used for it) and `pauser` stops the agent's
    account between requests; without a pauser (the selection tool outside the image, which starts the controller
    as itself) nothing is stopped."""
    budget = float(config.get("grader_budget_s", 6000))
    image_cost = float(config.get("image_cost_s", 0.0))
    clock = clock if clock is not None else Clock()

    def call(fn, *args, **kwargs):
        # one exchange with the controller: its processes run only inside it, and only its wait is charged
        if pauser is not None:
            pauser.check_held()
            pauser.thaw()
        try:
            out = clock.timed(fn, *args, **kwargs)
        except BaseException:
            if pauser is not None:
                clock.timed(pauser.freeze)
            raise
        if pauser is not None:
            clock.timed(pauser.pause_or_abort)      # stopping the account is the controller's time
        return out

    lock = IdleAwareProbeLock(None, rate=config.get("probe_rate", 0.01),
                              pass_fraction=config.get("probe_pass_fraction", 0.1), tolerance=1e-4)
    scores, aborted, faults = [], 0, 0
    per_unit, per_request = {}, {}
    records = []
    egress_hits = 0
    budget_exhausted = False
    GuardedSolver.guard = egress_mode == "seccomp"
    GuardedSolver.nproc = int(config.get("solver_nproc_limit", 0))
    solver = clock.timed(GuardedSolver, step_timeout=config.get("step_timeout_s", 30), submission=workdir)
    if pauser is not None:
        clock.timed(pauser.freeze)       # READY seen: stopped until the first reset is written
    try:
        for n, (unit, seed) in enumerate(episodes):
            if clock.s > budget:
                budget_exhausted = True
                scores.append(0.0)
                per_unit.setdefault(unit, []).append(0.0)
                records.append({"index": n, "unit": unit, "score": 0.0, "reason": "controller time budget exhausted",
                                "steps": 0, "requests": []})
                log("episode {} ({}): score=0.0000 not run (controller time budget exhausted)".format(n, unit))
                continue
            lock.rng = probe_stream(n, seed)
            ep = EP.Episode(E.ApplianceEnv(specs[unit], None, render=True), seed)
            score, reason, sc = 0.0, "ok", None
            try:
                obs = ep.reset()
                call(solver.reset)
                t, done = 0, False
                while not done:
                    if clock.s > budget:
                        budget_exhausted = True
                        raise SolverAborted("controller time budget exhausted")
                    if egress_mode == "detect" and agent_inet_sockets(uid):
                        egress_hits += 1
                    action = ask(lock, solver, ep, t, obs, unit, truth, call)
                    if not valid_action(action):
                        raise SolverAborted("malformed action")
                    clock.s += image_cost * bin(int(action[7])).count("1")     # the images it asks for
                    obs, done, _info = ep.step(action)
                    t += 1
                    if ep.env.unstable():
                        faults += 1
                        raise SolverAborted("simulation unstable")
                if clock.s > budget:             # the step that ended the episode may have spent the last of it
                    budget_exhausted = True
                    raise SolverAborted("controller time budget exhausted")
                sc = ep.score()
                score = sc["credit"]
                for r in sc["requests"]:
                    per_request.setdefault(r["what"], []).append(r["credit"])
            except (SolverAborted, ValueError, TypeError, OverflowError, KeyError, AttributeError,
                    RecursionError) as exc:
                aborted += 1
                score = 0.0
                reason = "aborted: {}".format(str(exc)[:160])
            finally:
                ep.env.close()
            if not (isinstance(score, float) and math.isfinite(score)) or score < 0:
                score = 0.0
            scores.append(score)
            per_unit.setdefault(unit, []).append(score)
            steps = getattr(ep.env, "t", 0)
            log("episode {} ({}): score={:.4f} steps={} {}".format(n, unit, score, steps, reason))
            if sc is None:
                try:
                    sc = ep.score()            # the requests so far, for the log only: an aborted episode scores 0
                except Exception:  # noqa: BLE001 - an episode that never started has no requests
                    sc = {"requests": []}
            log_requests(n, unit, sc)
            records.append({"index": n, "unit": unit, "score": score, "reason": reason, "steps": steps,
                            "requests": sc["requests"]})
    finally:
        if pauser is not None:
            pauser.thaw()                # the close message is read by a running controller
        try:
            solver.close()
        except Exception:  # noqa: BLE001 - closing a broken solver must not crash the grader
            pass
    return scores, aborted, faults, per_unit, per_request, egress_hits, budget_exhausted, lock, records


def main():
    t_start = time.monotonic()
    secured = {"root_path_dirs_secured": float(len(secure_exec_path()))}   # before anything else runs
    config = stage_config()
    lab_stopped = stop_lab()
    anchor, meta = read_anchor()
    try:
        uid, gid = _agent_ids()
    except KeyError:
        write_verdict(zero_verdict(reason="no agent account", anchored=anchor is not None, extra=secured))
        return
    if not stop_agent_processes(uid):
        write_verdict(zero_verdict(reason="agent processes could not be stopped (cleanup_failed)",
                                   anchored=anchor is not None, extra=secured))
        return
    with open(UNITS) as fh:
        units = json.load(fh)
    specs = {u: E.unit_spec(c["model"], files=E.read_unit_files(os.path.join(STAGE, c["dir"])), quirks=c.get("quirks"))
             for u, c in sorted(units.items())}
    episodes = load_episode_set(config, specs)
    check_numerics()                     # a task error on a machine that computes differently
    secured["numerics_ok"] = 1.0
    t_self = time.monotonic()
    try:
        render_selftest(specs[episodes[0][0]])
    except E.RenderError as exc:
        fail("the cameras' renderer failed its self-test: {}".format(exc))
    log("render self-test passed ({:.1f} s, three cameras)".format(time.monotonic() - t_self))
    secured["render_selftest"] = 1.0
    try:
        workdir = stage_submission(uid, gid)
    except OSError as exc:
        log("staging the submission failed: {}".format(type(exc).__name__))
        workdir = None
    if workdir is None:
        write_verdict(zero_verdict(reason="no usable submission", anchored=anchor is not None, extra=secured))
        return
    truth = oracle_token_ok(workdir)
    egress_mode = "seccomp" if egress_self_test(uid, gid, int(config.get("solver_nproc_limit", 0))) else "detect"
    log("egress guard: {}; oracle token: {}; {} episodes in the fixed set".format(egress_mode, truth, len(episodes)))
    clock, pauser = Clock(), Pauser(uid)
    try:
        (scores, aborted, faults, per_unit, per_request, egress_hits, budget_exhausted,
         lock, _records) = run_episodes(config, uid, egress_mode, t_start, specs, workdir, truth, episodes,
                                        clock=clock, pauser=pauser)
    except E.RenderError as exc:
        kill_agent_processes(uid)
        fail("the cameras' renderer failed during the episodes: {}".format(exc))
    except (SolverMissing, SolverAborted) as exc:
        kill_agent_processes(uid)
        write_verdict(zero_verdict(reason=str(exc)[:200], anchored=anchor is not None, extra=secured))
        return
    cleanup_ok = stop_agent_processes(uid)
    measured = sum(scores) / len(scores) if scores else 0.0
    if egress_hits or not cleanup_ok or not math.isfinite(measured) or measured < 0:
        measured = 0.0
    extra = dict(lock.rewards(),
                 episodes_in_set=float(len(episodes)),
                 episodes_scored=float(len(scores) - aborted),
                 episodes_aborted=float(aborted),
                 sim_faults=float(faults),
                 egress_guard_seccomp=1.0 if egress_mode == "seccomp" else 0.0,
                 egress_detected=float(egress_hits),
                 cleanup_failed=0.0 if cleanup_ok else 1.0,
                 lab_stopped=1.0 if lab_stopped else 0.0,
                 budget_exhausted=1.0 if budget_exhausted else 0.0,
                 controller_seconds=round(float(clock.s), 1),
                 agent_processes_peak=float(pauser.peak),
                 pauses_not_held=float(pauser.resumed),
                 oracle_token=1.0 if truth else 0.0,
                 **secured)
    for unit, vals in sorted(per_unit.items()):
        extra["score_" + unit] = sum(vals) / len(vals)
    for what, vals in sorted(per_request.items()):
        extra["request_" + what] = sum(vals) / len(vals)
    verdict = ratio_rewards(measured, anchor, cap=config.get("scoring_cap", 1.5), higher_is_better=True,
                            metric_name="mean_request_credit", locks_ok=lock.ok, extra=extra,
                            full_at=meta.get("full_at", 1.0))
    write_verdict(verdict)
    log("measured={:.5f} anchor={} ratio={:.3f} reward={:.3f} lock={} aborted={} faults={} egress={} "
        "controller_time={:.0f}s of {:.0f}s wall={:.0f}s".format(
            measured, anchor, verdict["metrics"]["ratio"], verdict["rewards"]["reward"], lock.ok, aborted, faults,
            egress_mode, clock.s, float(config.get("grader_budget_s", 6000)), time.monotonic() - t_start))


if __name__ == "__main__":
    main()
