"""Hidden grader of longtail_manipulation_recovery (closed loop, fixed hidden failure states).

Runs as root in a SEPARATE verifier sandbox: the engine restores only /home/user/submission there. The grader owns the
kitchen simulator (one child process per state, kitchen_sim.Kitchen restored by replaying the recorded action prefix,
then the recorded MuJoCo state set exactly), starts the agent's controller at the agent's UID with no network
(nonet_exec.py, seccomp; AF_UNIX only) and asks it for one action per control step over the kit wire. A state is
recovered when the job's own success check holds for 10 consecutive control steps within the step budget.

Metric: job_balanced_recovered_fraction = the mean over jobs of each job's recovered fraction (wire_rules.job_balanced).

Isolation while grading (all stated in the instruction):
- The controller is the only thing that runs as the agent's account. The verifier image (tests/Dockerfile) starts no
  desktop session: the base image's supervisord would run one as that account and restart it when the controller
  kills it, and the new session would run the account's autostart entries and start script outside the launcher's
  filter and outside the pauses below (both routes were measured on the first form of this task). At its start the
  grader reclaims root's PATH directories (the kit's secure_exec_path), kills what runs as the account and requires
  that nothing of the account appears by itself for QUIET_S; a sandbox that fails this is a task error. At every
  pause it checks that every process of the account carries the launcher's no_new_privs mark, which the launcher sets
  before it drops to the account and which no descendant can clear: a process without it was not started through
  the launcher. That ends the episode, and that state and every later one score 0.
- The controller's processes run only while a request is outstanding: every process of the agent's account is stopped
  (SIGSTOP) from the moment a reply has been read (or READY seen) until the grader starts writing the next request,
  so kitchen builds, simulation and rendering never share the CPUs with it and it cannot compute ahead of a request.
  A pause holds only once /proc shows every thread of every process of the account stopped; one that does not stop
  within FREEZE_TIMEOUT_S ends the episode. SIGSTOP holds only until the next SIGCONT, and a process can arrange its
  own (a POSIX timer that signals it), so the pause is checked again before the next request is written: every thread
  that was stopped must still be there and stopped, with the scheduler's record of its run time and of its turns on a
  CPU unchanged, and no thread may have appeared. Otherwise a process of the account ran during the pause and the
  episode ends. The time it takes to bring the account to a stop counts as controller time.
- A request's deadline and its controller time start before the grader writes it: the write itself (an observation
  with images is larger than the pipe's buffer) is bounded by the same deadline, so a controller that reads slowly is
  charged for it and one that stops reading misses the deadline. Controller time is summed per episode and capped.
- At most MAX_QUEUED_LINES reply lines are buffered (the controller's writes block beyond that); lines that arrive
  without a request outstanding are discarded; a line longer than MAX_LINE ends the episode.
- More than MAX_PROCESSES processes of the agent's account at a pause ends the episode.
- The simulator's asset tree is root-only. Its content is hashed before the controller starts and again before every
  state; a change scores that state and every later one 0 (it can only come from the agent's account).
- The replayed prefix must reproduce the recorded joint state to PREFIX_TOL; otherwise this host does not reproduce
  the recording and the run is a task error (never a silent loss of oracle states).

A failure of the simulator or of the grader itself is a task error, not the agent's zero. A controller that crashes,
stalls, sends a malformed action or runs out of its time cap ends only its own episode, which scores 0.

Oracle token: a submission whose .oracle_token hashes to oracle_token.sha256 is told the hidden state id in the
observation; the solution replays recorded rescue actions. Any other token is ignored and reported.
"""
import fcntl
import hashlib
import hmac
import json
import os
import pwd
import queue
import re
import select
import shutil
import signal
import struct
import subprocess
import sys
import termios
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import wire_rules as W  # noqa: E402
from robotics_grader import (  # noqa: E402
    Solver, SolverAborted, SolverMissing, agent_user, fail, ratio_rewards, read_anchor, stage_config, write_verdict,
    zero_verdict,
)
from robotics_grader.stage import secure_exec_path  # noqa: E402
from robotics_grader.solver import _spawn_at_lowest_priority  # noqa: E402

STAGE = os.path.dirname(os.path.abspath(__file__))
NONET = os.path.join(STAGE, "nonet_exec.py")
HIDDEN = os.path.join(STAGE, "hidden")
ASSET_ROOT = "/opt/sim/robocasa/robocasa/models/assets"
TEX_TMP = "/tmp/robosuite_temp_tex"          # robosuite writes texture patches here (root-owned, 0700)
RC_TEMP_XML = re.compile(r"^\d+_\d+_\d+\.xml$")   # RoboCasa's per-build temp model file, removed after parsing
MAX_SUBMISSION_BYTES = 200 * 1024 * 1024
MAX_SUBMISSION_FILES = 5000
FSIZE_LIMIT = 2 * 1024 ** 3                  # largest file the controller may write
TOKEN_FILE = ".oracle_token"
TOKEN_MAX_BYTES = 4096
SIM_PYTHON = "/opt/venv/bin/python3"
PREFIX_TOL = 1e-9                            # max |qpos - recorded| after the prefix replay (every run so far: 0.0)
QUIET_S = 5.0                                # nothing of the agent's account may appear by itself for this long at the
#                                              start (longer than supervisord's longest pause before it retries a start)
FREEZE_TIMEOUT_S = 5.0                       # every thread of the account must show as stopped within this
STOPPED_STATES = ("T", "t")                  # a thread that is stopped (or traced and stopped)
GONE_STATES = ("Z", "X")                     # a thread that has exited: it runs nothing, but its process may live on
CLOSE_WRITE_S = 1.0                          # the close message is written under this deadline, never blocking
METRIC = "job_balanced_recovered_fraction"


def log(msg):
    print("[grader] " + msg, file=sys.stderr, flush=True)


class Tampered(Exception):
    """The asset tree changed while the agent's account was running."""


# ---------------------------------------------------------------- agent account handling
def _agent_ids():
    pw = pwd.getpwnam(agent_user())
    return pw.pw_uid, pw.pw_gid


def _agent_pids(uid):
    pids = []
    for name in os.listdir("/proc"):
        if not name.isdigit():
            continue
        try:
            if os.stat("/proc/" + name).st_uid == uid:
                pids.append(int(name))
        except OSError:
            continue
    return pids


def _stat_state(path):
    """The one-letter state in a /proc stat file (the field after the command name), None if it cannot be read."""
    try:
        with open(path, "rb") as f:
            stat = f.read()
    except OSError:
        return None
    rest = stat[stat.rfind(b")") + 1:].split()
    return rest[0].decode("ascii", "replace") if rest else None


def _thread_states(pid):
    """The state of every thread of `pid` (/proc/<pid>/task/*/stat); empty when the process is gone.
    /proc/<pid>/stat alone is the thread-group leader's: a process whose main thread has exited shows there as a
    zombie (Z) while its other threads run on, so the leader's state says nothing about the process."""
    try:
        tids = os.listdir("/proc/{}/task".format(pid))
    except OSError:
        return []
    states = [_stat_state("/proc/{}/task/{}/stat".format(pid, tid)) for tid in tids]
    return [state for state in states if state is not None]


def _runs(pid):
    """True while a thread of `pid` is neither stopped nor gone."""
    return any(state not in STOPPED_STATES and state not in GONE_STATES for state in _thread_states(pid))


def _alive(pid):
    """True while `pid` has a thread that has not exited."""
    return any(state not in GONE_STATES for state in _thread_states(pid))


def kill_agent_processes(uid, rounds=20, pause=0.25):
    for _ in range(rounds):
        pids = _agent_pids(uid)
        if not pids:
            return True
        for pid in pids:
            for sig in (signal.SIGKILL, signal.SIGCONT):
                try:
                    os.kill(pid, sig)
                except OSError:
                    pass
        time.sleep(pause)
    return not _agent_pids(uid)


def _describe(pid):
    try:
        with open("/proc/{}/cmdline".format(pid), "rb") as f:
            return "{} {}".format(pid, f.read(120).replace(b"\0", b" ").decode("utf-8", "replace").strip())
    except OSError:
        return "{} (gone)".format(pid)


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


def require_empty_account(uid, quiet_s=QUIET_S):
    """The grading sandbox must run nothing as the agent's account by itself (tests/Dockerfile starts no desktop
    session). Called right after the verify-start kill, before the controller exists: a process of the account that
    is there, or appears within `quiet_s`, was started by the image, and a session that an image restarts runs outside
    the controller's launcher. That sandbox is not the one this grader is written for: a task error."""
    deadline = time.monotonic() + quiet_s
    while True:
        pids = _agent_pids(uid)
        if pids:
            fail("the grading sandbox starts processes as the agent's account by itself ({}); its image must run no "
                 "session as that account: the grading sandbox is not isolated".format("; ".join(map(_describe, pids[:5]))))
        if time.monotonic() >= deadline:
            return
        time.sleep(0.1)


def outside_launcher(pid):
    """True if `pid` is a live process without the no_new_privs mark. nonet_exec.py sets the mark before it drops to
    the agent's account, every descendant inherits it and none can clear it, so a process of the account without it
    was not started through the launcher (and so runs without its filter). A zombie runs nothing and does not count;
    a kernel that does not report the mark gives no finding."""
    try:
        with open("/proc/{}/status".format(pid)) as f:
            status = f.read()
    except OSError:
        return False
    fields = dict(line.split(":", 1) for line in status.splitlines() if ":" in line)
    if fields.get("State", "").strip()[:1] in ("Z", "X"):
        return False
    return fields.get("NoNewPrivs", "").strip() == "0"


def agent_bytes(uid, roots=("/tmp", "/var/tmp", "/dev/shm", "/home")):
    total = 0
    for root in roots:
        for dp, dns, fns in os.walk(root, followlinks=False):
            for fn in fns:
                try:
                    st = os.lstat(os.path.join(dp, fn))
                except OSError:
                    continue
                if st.st_uid == uid:
                    total += st.st_size
    return total


def stage_submission(uid, gid):
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


def agent_can_read(path):
    """True if the agent's account can open `path` (checked by the account itself)."""
    runuser = next(p for p in ("/usr/sbin/runuser", "/sbin/runuser", "/usr/bin/runuser", "/bin/runuser") if os.path.exists(p))
    r = subprocess.run([runuser, "-u", agent_user(), "--", "/usr/bin/test", "-r", path],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env={"PATH": "/usr/bin:/bin"})
    return r.returncode == 0


# ---------------------------------------------------------------- asset integrity
def asset_digest():
    """sha256 over every file under the asset tree (path, mode, owner, content), RoboCasa's temp files excluded,
    plus the owner and mode of robosuite's texture directory."""
    h = hashlib.sha256()
    for dp, dns, fns in os.walk(ASSET_ROOT, followlinks=False):
        dns.sort()
        st = os.lstat(dp)
        h.update("D {} {:o} {}\n".format(os.path.relpath(dp, ASSET_ROOT), st.st_mode, st.st_uid).encode())
        for fn in sorted(fns):
            if RC_TEMP_XML.match(fn):
                continue
            fp = os.path.join(dp, fn)
            st = os.lstat(fp)
            h.update("F {} {:o} {} {}\n".format(os.path.relpath(fp, ASSET_ROOT), st.st_mode, st.st_uid,
                                                st.st_size).encode())
            if os.path.isfile(fp) and not os.path.islink(fp):
                with open(fp, "rb") as f:
                    for block in iter(lambda: f.read(1 << 20), b""):
                        h.update(block)
    st = os.lstat(TEX_TMP)
    h.update("T {:o} {}\n".format(st.st_mode, st.st_uid).encode())
    return h.hexdigest()


def prepare_sim_tree():
    """Root-only asset tree and texture directory, before any agent process exists in this sandbox."""
    if os.path.lexists(TEX_TMP) and (os.path.islink(TEX_TMP) or not os.path.isdir(TEX_TMP)):
        os.remove(TEX_TMP)
    os.makedirs(TEX_TMP, exist_ok=True)
    os.chown(TEX_TMP, 0, 0)
    os.chmod(TEX_TMP, 0o700)
    for dp, dns, fns in os.walk(ASSET_ROOT, followlinks=False):
        for name in [dp] + [os.path.join(dp, n) for n in fns]:
            st = os.lstat(name)
            if st.st_uid != 0 or st.st_mode & 0o077:
                if not os.path.islink(name):
                    os.chown(name, 0, 0)
                    os.chmod(name, st.st_mode & 0o700)


# ---------------------------------------------------------------- the controller
class GuardedSolver(Solver):
    """The kit Solver, spawned through the seccomp launcher, with a bounded reply queue, a reply-line cap, stale-line
    discarding, a process cap, and every process of the agent's account paused while no request is outstanding."""

    def __init__(self, uid, **kw):
        self.uid = uid
        self.last_reply_s = 0.0
        self._stopped = []
        self._held = None                    # the threads' records once a pause holds (None: nothing to compare)
        self.resumed = 0                     # pauses during which a process of the account ran
        self.outside = []                    # processes of the account found running outside the launcher
        self.unstoppable = []                # processes of the account that did not stop at the last pause
        super().__init__(**kw)

    # -- pausing the agent's account
    def freeze(self):
        """SIGSTOP every process of the account and confirm it thread by thread: a process counts as paused only once
        /proc shows each of its threads stopped (T, t) or exited (Z, X). Every process with a thread still running is
        signalled, whatever its leader's state: a process whose main thread has exited shows as a zombie and still
        computes in its other threads. A scan that finds one running signals it again and scans anew, so a fork
        chain is caught process by process; the pause holds when two consecutive scans find the same processes, all
        stopped (a stopped process cannot start another). Not reached within FREEZE_TIMEOUT_S: self.unstoppable.
        Returns the number of processes alive at the last scan (the process cap counts those)."""
        stopped, previous = set(), None
        deadline = time.monotonic() + FREEZE_TIMEOUT_S
        self.unstoppable = []
        self._held = None
        while True:
            pids = _agent_pids(self.uid)
            running = [p for p in pids if _runs(p)]
            for pid in running:
                try:
                    os.kill(pid, signal.SIGSTOP)
                except OSError:
                    pass
            stopped.update(pids)
            current = frozenset(pids)
            if not running and current == previous:
                break
            previous = None if running else current
            if time.monotonic() > deadline:
                self.unstoppable = [_describe(p) for p in running[:5]]
                break
            if running:
                time.sleep(0.001)
        alive = sum(1 for p in pids if _alive(p))
        self._stopped = sorted(stopped)
        for pid in self._stopped:
            if outside_launcher(pid):
                self.outside.append(_describe(pid))
                try:
                    os.kill(pid, signal.SIGKILL)
                except OSError:
                    pass
        if not self.unstoppable and not self.outside:
            # /proc shows a thread as stopped an instant before it leaves the CPU: the records are taken once two
            # readings a millisecond apart agree
            held = _thread_records(pids)
            while time.monotonic() <= deadline:
                time.sleep(0.001)
                again = _thread_records(_agent_pids(self.uid))
                if again == held:
                    self._held = held
                    break
                held = again
            else:
                self.unstoppable = ["the account's threads keep changing"]
        return alive

    def check_held(self):
        """Before the next request: did the pause hold? Raises SolverAborted if a process of the account ran since
        the pause was confirmed (continued by a signal it arranged for itself) or a new one appeared."""
        held, self._held = self._held, None
        if held is None:
            return
        why = _pause_broken(held, _thread_records(_agent_pids(self.uid)))
        if why:
            self.resumed += 1
            self._broken = True
            raise SolverAborted("a process of the controller's account ran during a pause ({})".format(why))

    def thaw(self):
        self._held = None
        for pid in self._stopped:
            try:
                os.kill(pid, signal.SIGCONT)
            except OSError:
                pass
        self._stopped = []

    def _pause_or_abort(self):
        count = self.freeze()
        if self.unstoppable:
            self._broken = True
            raise SolverAborted("processes of the controller's account did not stop within {:g}s ({})".format(
                FREEZE_TIMEOUT_S, "; ".join(self.unstoppable)))
        if self.outside:
            self._broken = True
            raise SolverAborted("a process of the controller's account runs outside the grader's launcher")
        if count > W.MAX_PROCESSES:
            self._broken = True
            raise SolverAborted("more than {} processes in the controller's account".format(W.MAX_PROCESSES))

    # -- process
    def _spawn(self):
        secure_exec_path()   # as the kit's own launcher does: nothing agent-writable is left on root's PATH
        uid, gid = _agent_ids()
        cmd = [sys.executable, "-I", NONET, "--uid", str(uid), "--gid", str(gid), "--cwd", self.submission,
               "--rlimit-fsize", str(FSIZE_LIMIT), "--", "bash", self.entry]
        try:
            return subprocess.Popen(cmd, cwd=self.submission, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                    stderr=subprocess.DEVNULL, universal_newlines=True, bufsize=1,
                                    start_new_session=True, env={"PATH": "/usr/bin:/bin"})
        except OSError as exc:
            raise SolverAborted("solver could not be started: {}".format(exc))

    def _start(self):
        # At the lowest priority, as the kit's own start: a process that continues itself during a pause runs until
        # the next request finds it, and must not slow the simulator meanwhile.
        self.proc, _idle = _spawn_at_lowest_priority(self._spawn)
        self._lines = queue.Queue(maxsize=W.MAX_QUEUED_LINES)
        threading.Thread(target=self._pump, args=(self.proc, self._lines), daemon=True).start()
        try:
            self._await_ready()
            self._pause_or_abort()
        except SolverAborted:
            self._stop(grace=0)
            raise

    def _stop(self, grace):
        self.thaw()
        super()._stop(grace)
        kill_agent_processes(self.uid)      # also what left the process group (setsid)

    def _restart(self):
        if self.restarts >= self.max_restarts:
            # The kit raises here before it stops the old process. Stop it first: with no restart left, a controller
            # that stopped reading would otherwise stay, with a full pipe, through every remaining state.
            self._stop(grace=0)
        super()._restart()

    @staticmethod
    def _pump(proc, lines):
        while True:
            try:
                line = proc.stdout.readline(W.MAX_LINE + 2)
            except (ValueError, OSError, UnicodeDecodeError):
                break
            if not line or W.line_too_long(line):
                break
            lines.put(line)                  # blocks when MAX_QUEUED_LINES are waiting: the controller's write stalls
        lines.put(None)

    def _discard_stale(self):
        """Drop every line the controller wrote without a request outstanding. Its processes are paused, so the pipe
        only drains: empty the queue until neither it nor the pipe holds anything."""
        for _ in range(2000):
            drained = False
            while True:
                try:
                    line = self._lines.get_nowait()
                except queue.Empty:
                    break
                if line is None:
                    raise SolverAborted("solver closed its stdout")
                drained = True
            try:
                pending = struct.unpack("i", fcntl.ioctl(self.proc.stdout.fileno(), termios.FIONREAD, b"\0\0\0\0"))[0]
            except (OSError, ValueError):
                pending = 0
            if not drained and not pending:
                return
            time.sleep(0.005)
        raise SolverAborted("solver keeps writing without a request")

    def _write_within(self, data, deadline):
        """Write one request against the reply's own deadline. A request with images is larger than the pipe's buffer,
        so a blocking write would wait for the controller to read it, with the controller running and not charged,
        and forever for a controller that stops reading. The pipe is written non-blocking, a chunk at a time as it
        drains, and a request still not read at the deadline ends the episode like a late reply (the restart that
        follows kills the controller; the partial request dies with its pipe)."""
        try:
            fd = self.proc.stdin.fileno()
        except (OSError, ValueError):
            raise SolverAborted("solver stdin closed")
        view = memoryview(data.encode("utf-8"))
        flags = fcntl.fcntl(fd, fcntl.F_GETFL)
        fcntl.fcntl(fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)
        try:
            while view:
                left = deadline - time.monotonic()
                if left <= 0:
                    raise SolverAborted("solver did not read the request within its {:g}s deadline".format(
                        self.step_timeout))
                if not select.select([], [fd], [], left)[1]:
                    continue
                try:
                    view = view[os.write(fd, view):]
                except BlockingIOError:
                    continue
                except OSError:
                    raise SolverAborted("solver stdin closed")
        finally:
            fcntl.fcntl(fd, fcntl.F_SETFL, flags)

    def _exchange(self, message):
        self.last_reply_s = 0.0
        self.check_held()                # a process that continued itself during the pause ends the episode
        self._discard_stale()
        data = json.dumps(message) + "\n"
        t0 = time.monotonic()            # the clock and the deadline start before the write, not after it
        deadline = t0 + self.step_timeout
        self.thaw()
        try:
            self._write_within(data, deadline)
            for _ in range(1000):
                line = self._readline(deadline, self.step_timeout).strip()
                if not line:
                    continue
                try:
                    reply = json.loads(line)
                except (ValueError, RecursionError):
                    raise SolverAborted("solver wrote non-JSON on the wire: {!r}".format(line[:120]))
                if isinstance(reply, dict):
                    return reply
                raise SolverAborted("solver reply is not an object")
            raise SolverAborted("solver wrote too many blank lines to one request")
        finally:
            try:
                self._pause_or_abort()
            finally:
                self.last_reply_s = time.monotonic() - t0      # stopping the account is the controller's time too

    def close(self):
        """The kit's close without its blocking write. The close message is written under CLOSE_WRITE_S like a
        request, and not at all once an exchange has failed (the wire is out of step, and a request that timed out
        in the write has left the pipe full: a blocking write would then never return). The process group is
        stopped either way."""
        if self._closed:
            return
        self._closed = True
        self.thaw()
        if self.proc is None:
            return
        told = False
        if self.proc.poll() is None and not self._broken and not self._dead:
            try:
                self._write_within(json.dumps({"type": "close"}) + "\n", time.monotonic() + CLOSE_WRITE_S)
                told = True
            except SolverAborted:
                pass
        self._stop(grace=10 if told else 0)


# ---------------------------------------------------------------- oracle token
def token_state(submission, digest_path):
    path = os.path.join(submission, TOKEN_FILE)
    if not os.path.lexists(path):
        return False, False
    try:
        with open(digest_path) as f:
            want = f.read().strip()
    except OSError:
        return False, True
    if os.path.islink(path) or not os.path.isfile(path) or os.path.getsize(path) > TOKEN_MAX_BYTES:
        return False, True
    try:
        with open(path, "rb") as f:
            got = hashlib.sha256(f.read()).hexdigest()
    except OSError:
        return False, True
    if hmac.compare_digest(got, want):
        return True, False
    return False, True


# ---------------------------------------------------------------- simulator child
class SimBuildFailed(Exception):
    pass


class SimWorker:
    """One restored kitchen in a child process; any failure of it is a task error (or, at build, see main)."""

    def __init__(self, spec, sim_log):
        self.log = sim_log
        self.proc = subprocess.Popen([SIM_PYTHON, "-s", "-B", os.path.join(STAGE, "sim_worker.py")], stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, stderr=self.log, universal_newlines=True, bufsize=1,
                                     cwd=STAGE, env=_sim_env())
        try:
            self._write(spec)
            msg = self._read()
        except SystemExit:
            raise SimBuildFailed("the kitchen simulator stopped while building state {}".format(spec["id"]))
        if not msg.get("ready"):
            raise SimBuildFailed("the kitchen simulator failed to build state {}: {}".format(
                spec["id"], str(msg.get("error", "?"))[:300]))
        self.prefix_max_err = float(msg["prefix_max_err"])
        self.built_s = msg["built_s"]
        if not msg.get("shape_ok"):
            fail("state {}: the rebuilt scene does not match the recording (image or assets differ)".format(spec["id"]))
        if not self.prefix_max_err <= PREFIX_TOL:
            fail("state {}: the replayed failure prefix differs from the recording by {:.3g} (tolerance {:g}); this host "
                 "does not reproduce the recorded numerics (x86-64 with AVX2/FMA and the pinned kernels are required)"
                 .format(spec["id"], self.prefix_max_err, PREFIX_TOL))

    def _write(self, obj):
        try:
            self.proc.stdin.write(json.dumps(obj) + "\n")
            self.proc.stdin.flush()
        except (OSError, ValueError) as exc:
            fail("the kitchen simulator stopped responding: {}".format(type(exc).__name__))

    def _read(self):
        line = self.proc.stdout.readline()
        if not line:
            fail("the kitchen simulator exited unexpectedly")
        try:
            msg = json.loads(line)
        except ValueError:
            fail("the kitchen simulator wrote a malformed line")
        if not isinstance(msg, dict):
            fail("the kitchen simulator wrote a malformed line")
        return msg

    def first(self):
        self._write({"first": True})
        return self._read()

    def step(self, controls, cams):
        self._write({"action": controls, "cams": cams})
        return self._read()

    def close(self):
        try:
            self._write({"close": True})
        except SystemExit:
            pass
        try:
            self.proc.stdin.close()
            self.proc.wait(timeout=30)
        except Exception:  # noqa: BLE001
            self.proc.kill()


def _sim_env():
    return {"PATH": "/opt/venv/bin:/usr/bin:/bin", "HOME": "/root", "LANG": "C.UTF-8", "PYTHONNOUSERSITE": "1",
            "MUJOCO_GL": "osmesa", "PYOPENGL_PLATFORM": "osmesa", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1", "OPENBLAS_CORETYPE": "Haswell", "NUMBA_CPU_NAME": "haswell",
            "NUMBA_CPU_FEATURES": "+avx2,+fma,+sse4.2", "NPY_DISABLE_CPU_FEATURES": "AVX512F AVX512CD AVX512_KNL "
            "AVX512_KNM AVX512_SKX AVX512_CLX AVX512_CNL AVX512_ICL AVX512_SPR AVX512FP16", "LP_NUM_THREADS": "4",
            "PYTHONHASHSEED": "0", "MESA_SHADER_CACHE_DISABLE": "true"}


# ---------------------------------------------------------------- hidden set
def load_hidden():
    path = os.path.join(HIDDEN, "states.json")
    try:
        with open(path) as f:
            spec = json.load(f)
    except (OSError, ValueError) as exc:
        fail("cannot read the hidden set: {}".format(type(exc).__name__))
    if not isinstance(spec, dict) or spec.get("format") != "longtail_manipulation_recovery hidden set 1":
        fail("the hidden set has an unexpected format")
    states = spec.get("states")
    if not isinstance(states, list) or not states:
        fail("the hidden set holds no states")
    ids = set()
    for s in states:
        for k in ("id", "task", "seed", "split", "prefix", "qpos"):
            if k not in s:
                fail("a hidden state has no '{}'".format(k))
        if s["id"] in ids:
            fail("duplicate hidden state id")
        ids.add(s["id"])
        if not (0 <= int(s["seed"]) < 2 ** 32):
            fail("a hidden seed is not a 32-bit integer")
        for k in ("prefix", "qpos"):
            if not os.path.isfile(os.path.join(HIDDEN, s[k])):
                fail("hidden state file missing: {}".format(s[k]))
    return spec


# ---------------------------------------------------------------- one state
def note(state, recovered, reason):
    """One state's outcome, logged and returned for the verdict's diagnostic."""
    line = "state {} ({}): recovered={} {}".format(state["id"], state["task"], recovered, reason)
    log(line)
    return line


def run_state(spec, state, solver, cfg, privileged, stats):
    """Returns (recovered, reason). SolverAborted ends this state only; simulator failures are task errors."""
    budget = int(spec["step_budget"])
    sim_spec = {"id": state["id"], "task": state["task"], "seed": int(state["seed"]), "split": state["split"],
                "prefix_file": os.path.join(HIDDEN, state["prefix"]), "qpos_T_file": os.path.join(HIDDEN, state["qpos"]),
                "cards_file": os.path.join(HIDDEN, state["card"]) if state.get("card") else None,
                "roles": state.get("roles"), "step_budget": budget, "max_frames": W.MAX_FRAMES,
                "privileged": bool(privileged)}
    controller_s = 0.0
    with open(os.path.join(cfg["_logdir"], "sim_{}.log".format(state["id"])), "w") as sim_log:
        sim = SimWorker(sim_spec, sim_log)
        try:
            stats["prefix_max_err"] = max(stats.get("prefix_max_err", 0.0), sim.prefix_max_err)
            reply = sim.first()
            obs = reply["obs"]
            solver.reset()   # an abort here ends this episode like any other (counted, scores 0)
            t, consecutive = 0, 0
            while t < budget:
                solver.last_reply_s = 0.0
                try:
                    action = solver.act(t, obs)
                finally:
                    controller_s += solver.last_reply_s      # an aborted request's time is the controller's too
                if controller_s > W.CONTROLLER_CAP_S:
                    raise SolverAborted("controller time cap exceeded")
                try:
                    controls, cams = W.check_action(action)
                except W.Malformed as exc:
                    raise SolverAborted(str(exc))
                except Exception as exc:  # noqa: BLE001 — anything else a reply can provoke ends this episode only
                    raise SolverAborted("unusable reply ({})".format(type(exc).__name__))
                reply = sim.step(controls, cams)
                obs = reply["obs"]
                t += 1
                consecutive = consecutive + 1 if reply["success"] else 0
                if consecutive >= W.SUCCESS_RUN:
                    return True, "ok steps={} controller_s={:.1f}".format(t, controller_s)
            return False, "ok steps={} controller_s={:.1f}".format(t, controller_s)
        finally:
            stats["controller_s"] = stats.get("controller_s", 0.0) + controller_s
            sim.close()


def main():
    secure_exec_path()       # before anything else: no directory on root's PATH is left to the agent's account
    config = stage_config()
    anchor, meta = read_anchor()
    hidden = load_hidden()
    try:
        uid, gid = _agent_ids()
    except KeyError:
        write_verdict(zero_verdict(reason="no agent account", anchored=anchor is not None))
        return
    kill_agent_processes(uid)
    require_empty_account(uid)
    try:
        workdir = stage_submission(uid, gid)
    except OSError as exc:
        log("staging the submission failed: {}".format(type(exc).__name__))
        workdir = None
    if workdir is None:
        write_verdict(zero_verdict(reason="no usable submission", anchored=anchor is not None))
        return
    prepare_sim_tree()
    for path in (os.path.join(HIDDEN, "states.json"), STAGE, ASSET_ROOT, TEX_TMP):
        if agent_can_read(path):
            fail("the agent account can read {}: the grading sandbox is not isolated".format(path))
    baseline_digest = asset_digest()
    privileged, mismatch = token_state(workdir, os.path.join(STAGE, "oracle_token.sha256"))
    if privileged:
        log("oracle token matched: privileged mode")
    logdir = os.path.join(os.environ.get("TMPDIR", "/tmp"), "simlogs")
    os.makedirs(logdir, exist_ok=True)
    config["_logdir"] = logdir
    states = hidden["states"]
    results, stats, aborted, tampered, escaped = [], {}, 0, None, None
    outcomes = []                            # one line per state, kept in the verdict's diagnostic
    try:
        solver = GuardedSolver(uid, step_timeout=W.STEP_TIMEOUT_S, submission=workdir, max_restarts=W.MAX_RESTARTS)
    except SolverMissing as exc:
        write_verdict(zero_verdict(reason=str(exc)[:200], anchored=anchor is not None))
        return
    except SolverAborted as exc:
        write_verdict(zero_verdict(reason="controller did not start: " + str(exc)[:200], anchored=anchor is not None))
        kill_agent_processes(uid)
        return
    try:
        for state in states:
            if tampered is None and asset_digest() != baseline_digest:
                tampered = state["id"]
                log("the simulator's asset tree changed before state {}: it and every later state score 0".format(
                    state["id"]))
            if tampered is not None:
                results.append((state["task"], False))
                outcomes.append(note(state, False, "assets changed"))
                continue
            if escaped is None and solver.outside:
                escaped = state["id"]
                log("processes of the agent's account ran outside the launcher ({}): state {} and every later one "
                    "score 0".format("; ".join(solver.outside[:5]), state["id"]))
            if escaped is not None:
                results.append((state["task"], False))
                outcomes.append(note(state, False, "a process ran outside the launcher"))
                continue
            try:
                recovered, reason = run_state(hidden, state, solver, config, privileged, stats)
            except SolverAborted as exc:
                recovered, reason = False, "aborted: " + str(exc)[:160]
                aborted += 1
                solver._broken = True   # kit attribute: the next reset restarts run.sh, as the instruction says
            except SimBuildFailed as exc:
                if asset_digest() != baseline_digest:
                    tampered = state["id"]
                    recovered, reason = False, "assets changed; the kitchen could not be built"
                elif agent_bytes(uid) > 1024 ** 3 and shutil.disk_usage("/").free < 1024 ** 3:
                    tampered = state["id"]
                    recovered, reason = False, "the controller's files filled the disk; the kitchen could not be built"
                else:
                    fail(str(exc))
            results.append((state["task"], bool(recovered)))
            outcomes.append(note(state, bool(recovered), reason))
    finally:
        try:
            solver.close()
        except Exception:  # noqa: BLE001
            pass
        kill_agent_processes(uid)
    measured, per_job = W.job_balanced(results)
    extra = {"states": float(len(results)), "states_recovered": float(sum(ok for _, ok in results)),
             "episodes_aborted": float(aborted), "assets_tampered": 1.0 if tampered else 0.0,
             "processes_outside_launcher": float(len(solver.outside)),
             "oracle_token_matched": 1.0 if privileged else 0.0, "oracle_token_mismatch": 1.0 if mismatch else 0.0,
             "prefix_max_err": float(stats.get("prefix_max_err", 0.0)),
             "controller_seconds": float(stats.get("controller_s", 0.0))}
    for job, frac in sorted(per_job.items()):
        extra["recovered_fraction_" + job] = float(frac)
    verdict = ratio_rewards(measured, anchor, cap=float(config.get("scoring_cap", 1.5)), higher_is_better=True,
                            metric_name=METRIC, extra=extra, full_at=float(meta.get("full_at", 1.0)))
    summary = "measured={:.4f} anchor={} reward={:.3f} aborted={} privileged={} tampered={}".format(
        measured, anchor, verdict["rewards"]["reward"], aborted, privileged, tampered)
    write_verdict(verdict, diagnostic="\n".join(outcomes + [summary]))
    log(summary)


if __name__ == "__main__":
    main()
