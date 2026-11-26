"""Hidden grader for cable_routing_through_clips (closed loop, a fixed hidden set of boards).

Runs as root in a SEPARATE verifier sandbox, whose image (tests/Dockerfile) starts no desktop session: no process of
the agent's account exists there, and the engine restores only the declared artifact /home/user/submission. Structure
and safeguards follow lab_sample_handling's grader: it reclaims root's PATH directories, clears and kills the
agent's account, hands the restored submission (size-capped) to that account in place, starts `bash run.sh` through
nonet_exec.py (a seccomp filter: only AF_UNIX sockets) at the agent's UID, at the lowest priority (the kit: the
grader's own simulation and drawing run on the same CPUs and are served first) and with at most
`solver_max_processes` processes in the account (counted here before every request, and by the kit at every
reset: more ends the episode and restarts the controller), and speaks the kit wire (Solver, per-request deadline, restarts; a
ProbeLock that does not count a pair in which both replies command no motion). During the episodes every process of the account
must carry the launcher's no_new_privs mark; one without it is killed, that episode and every later one score 0.

The cell: harness_env.py is the physics (a Panda at a bench, a harness board with clips and posts, a cable fixed at
its connector), harness_sim.py the public wire, harness_render.py the two cameras (MuJoCo's Filament on Mesa's
lavapipe, on the CPU, with the render bundle of HARNESS_RENDER_DIR). The agent has none of them: its commissioning
service (environment/commission_service/) runs byte-identical copies under another account.

A process per episode: every episode's simulation and renderer run in a child interpreter
(harness_sim.EpisodeProcess). A child that ends by itself (a native fault in the physics or the renderer: seen once
in 24 development episodes, never reproduced) is not the controller's doing: the episode is run again from its
start, with the same hidden seed and the same probe schedule, in a fresh process, and the controller's clock is set
back to the episode's start. Every such re-run is written to this log by the episode's index (never a seed) and
counted (`sim_reruns`). An episode whose second process ends by itself too, or more than `max_reruns` re-runs in a
run, is the environment's error: no reward is written (robotics_grader.stage.fail), never a 0 for the submission.

Numerics: `check_numerics` runs numerics_check.py before anything is graded and ends the run as a task error if
this machine does not reproduce tests/numerics.json.

Time: `grader_budget_s` is charged to the controller's own time (what the grader waits for READY, for each reset and
for each reply) plus `image_cost_s` for every image an applied action asks for: a fixed price, whatever this machine
takes to draw it (the first observation's two images are free). The simulation and the drawing are the grader's and
are not counted. When the budget is used up the running episode ends as it lies and later episodes score 0.

Episodes: the fixed hidden set tests/hidden/episodes.json (20 boards chosen by the pre-registered rule of
tests/review/PREREG.md; tests/review/select_set.py wrote it), the same for every submission, in the file's order.
The probe lock draws from one stream per episode (probe_stream: the kit's seed_rng keyed by a BLAKE2b hash of a
domain tag, the episode's index and its seed), so an episode's schedule is a function of the episode alone. The other
copy of a probed observation is built as an observation is (harness_sim.Cell.probe: the joints read up to 1 mrad
off; the grip site's pose one physics step behind them, as in a real observation (the kit's stale_copy rule); the
wrist sensor's noise one fresh draw around the noise-free reading; the same pictures).

Credit: an episode ends when the controller hands the job over (a ninth number 1 in its action), after
harness_env.MAX_STEPS control steps, or when a reply is late, malformed or missing (the controller is restarted for
the next episode; the kit's rule). The cell then opens the hand, lifts it, lets the cable settle and judges the
settled cable (harness_env.evaluate): the episode's credit is 1 if the whole route is made in order and the
connector was never overloaded (the mean pull on it over 0.1 s never above 40 N), else 0. `mean_route_credit` is the
mean over the set; `mean_prefix_credit` (the fraction of the route made in order from the connector, 0 if
overloaded) is reported and not scored.

Oracle token: a submission whose .oracle_token hashes to tests/hidden/oracle_token.sha256 is the reference; only
then does every observation also carry `truth` (the fixtures as mounted, the cable's points, the connector's
tension). An untokened submission never sees it. Every path agent behaviour can reach is charged as a score, never
a crash. The log names episodes by index and never prints a seed, a fixture's place or a board's layout.
"""

import base64
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402

import harness_sim as S  # noqa: E402
from robotics_grader import (  # noqa: E402
    ProbeLock,
    Solver,
    SolverAborted,
    SolverMissing,
    agent_user,
    clear_agent_account,
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
TOKEN_SHA = os.path.join(STAGE, "hidden", "oracle_token.sha256")
EPISODES = os.path.join(STAGE, "hidden", "episodes.json")
STILL = 1e-9                # a reply whose six motion numbers are all within this of zero commands no motion
SELFTEST_SEED = 1           # the render self-test's board (not a graded seed, not a development board)
PROBE_TAG = b"cable_routing_through_clips verify probe schedule 1"   # probe_stream's domain tag
MAX_LINE = 1 << 16          # replies are short (an action is at most 9 numbers)
NUMERICS = os.path.join(STAGE, "numerics_check.py")
NUMERICS_REF = os.path.join(STAGE, "numerics.json")
NUMERICS_TIMEOUT_S = 300
QUIET_S = 5.0               # after the verify-start kill nothing of the agent's account may appear by itself this long
CHECK_EVERY = 10            # control steps between two looks at the agent account's processes


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


def _describe(pid):
    try:
        with open("/proc/{}/cmdline".format(pid), "rb") as f:
            return "{} {}".format(pid, f.read(120).replace(b"\0", b" ").decode("utf-8", "replace").strip())
    except OSError:
        return "{} (gone)".format(pid)


def require_empty_account(uid, quiet_s=QUIET_S):
    """The grading sandbox must run nothing as the agent's account by itself (tests/Dockerfile starts no desktop
    session). Called right after the verify-start kill, before the controller exists: a process of the account that
    is there, or appears within `quiet_s`, was started by the image, and a session that an image restarts runs
    outside the controller's launcher. That sandbox is not the one this grader is written for: a task error."""
    deadline = time.monotonic() + quiet_s
    while True:
        pids = _agent_pids(uid)
        if pids:
            fail("the grading sandbox starts processes as the agent's account by itself ({}); its image must run no "
                 "session as that account: the grading sandbox is not isolated".format(
                     "; ".join(map(_describe, pids[:5]))))
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


def outside_processes(uid):
    """The agent account's processes that run outside the launcher, described; each is killed."""
    found = []
    for pid in _agent_pids(uid):
        if outside_launcher(pid):
            found.append(_describe(pid))
            try:
                os.kill(pid, signal.SIGKILL)
            except OSError:
                pass
    return found


def check_numerics():
    """Run numerics_check.py in a fresh interpreter (a short scripted run of the simulation; the sha256 of its
    states) and compare with tests/numerics.json. Returns the fingerprint; a machine that does not reproduce the
    recorded one would run every controller along another trajectory, so it must not grade: a task error."""
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
        fail("this machine does not reproduce the simulation's recorded numerics (tests/numerics.json): got {}, "
             "recorded {}; OPENBLAS_CORETYPE={!r}. Grading here would run every controller along another "
             "trajectory".format(got[:16], want[:16], os.environ.get("OPENBLAS_CORETYPE")))
    return got


class Clock(object):
    """The controller's own time (s): what the grader waited for READY and for each reply, and what `run_episodes`
    adds for the images it asks for. The simulation and the drawing of the images happen outside `timed`."""

    def __init__(self):
        self.s = 0.0

    def timed(self, fn, *args, **kwargs):
        t0 = time.monotonic()
        try:
            return fn(*args, **kwargs)
        finally:
            self.s += time.monotonic() - t0


class GuardedSolver(Solver):
    """The kit Solver, spawned through the seccomp launcher, with a reply-line cap."""

    guard = True
    nproc = 0

    def _spawn(self):
        secure_exec_path()               # again before every start: nothing agent-writable ahead of root's tools
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


def stage_submission(uid, gid):
    """Make the restored submission the agent account's own folder, in place at ~user/submission: the
    folder the instruction names, where run.sh is started and where the controller may write, as it can in
    commission/run_bench.py. Ownership changes never follow a symlink (lchown, a walk that does not
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
    within STILL of zero) is not counted, neither in the total nor as passed: a controller that holds still, waiting
    or finished, shows nothing either way. Every other pair is the kit's (`ProbeLock.record`)."""

    def __init__(self, *args, **kwargs):
        ProbeLock.__init__(self, *args, **kwargs)
        self.idle_skipped = 0

    @staticmethod
    def still(action):
        if not isinstance(action, list) or len(action) not in (8, 9):
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

    def counts(self):
        return (self.passed, self.total, self.idle_skipped)

    def restore(self, counts):
        self.passed, self.total, self.idle_skipped = counts


def probe_stream(index, seed):
    """The probe lock's stream for episode `index` of the set, whose hidden 64-bit seed is `seed`: the kit's seed_rng
    keyed by BLAKE2b over PROBE_TAG, the index and the seed, so a function of (index, seed) alone. The environment
    draws from the seed only through NumPy generators; this hash shares nothing with them."""
    digest = hashlib.blake2b(b"%s|%d|%d" % (PROBE_TAG, int(index), int(seed)), digest_size=32).digest()
    return seed_rng(entropy=int.from_bytes(digest, "little"))


def wire_obs(ep, obs, truth):
    w = dict(obs)
    w["job"] = ep.job
    if truth:
        w["truth"] = ep.truth()
    return w


def ask(lock, solver, ep, t, obs, truth, clock):
    """Step `t`'s action: one request, or a probe pair when the lock's stream says so. Every step draws once from
    the stream (lock.due); a probed step draws twice more, both before either request is written. The draws never
    depend on the replies. Only the wire (the requests' round trips) is charged to `clock`."""
    w = wire_obs(ep, obs, truth)
    if lock.due():
        other = ep.probe(lock.rng.seed(bits=64))
        return clock.timed(lock.probe, solver, t, w, perturbed=wire_obs(ep, other, truth))
    return clock.timed(solver.act, t, w)


def render_selftest(seed=SELFTEST_SEED):
    """The two cameras of one board, before any episode is graded: decoded, the right shapes, not flat (a standard
    deviation of at least 8 levels) and a mean brightness of 20-235 levels. Raises RuntimeError."""
    import cv2
    ep = S.EpisodeProcess(render=True)
    try:
        obs = ep.reset(seed, 3)
        for cam, spec in sorted(ep.job["cameras"].items()):
            e = obs.get(cam)
            img = cv2.imdecode(np.frombuffer(base64.b64decode(e["png"]), np.uint8), cv2.IMREAD_COLOR) if e else None
            if img is None or tuple(img.shape) != (spec["h"], spec["w"], 3) or float(img.std()) < 8.0 \
                    or not 20.0 <= float(img.mean()) <= 235.0:
                raise RuntimeError("{} image: {}".format(cam, None if img is None else (
                    tuple(img.shape), round(float(img.mean()), 1), round(float(img.std()), 1))))
    finally:
        ep.close()


def load_episode_set(config, path=EPISODES):
    """The fixed hidden set: 64-bit seeds in the graded order (a task error if it is not what the config says)."""
    with open(path) as fh:
        data = json.load(fh)
    seeds = [int(e["seed"]) for e in data["episodes"]]
    if len(seeds) != int(config.get("episodes", 20)) or len(set(seeds)) != len(seeds) \
            or any(not 0 <= s < 2 ** 64 for s in seeds):
        raise RuntimeError("tests/hidden/episodes.json does not hold {} distinct 64-bit seeds".format(
            config.get("episodes", 20)))
    return seeds


class EnvironmentError_(Exception):
    """The grader's own simulation or renderer failed: a task error, never the submission's."""


def run_one(n, seed, solver, lock, clock, config, uid, watch, truth, outside, state):
    """One attempt at episode `n` in a fresh process. Returns (result, reason, aborted). S.CellCrashed if the
    episode's process ends by itself; EnvironmentError_ if it reports a fault of its own."""
    budget = float(config.get("grader_budget_s", 3000))
    image_cost = float(config.get("image_cost_s", 0.2))
    cap = int(config.get("solver_max_processes", 64))
    ep = S.EpisodeProcess(render=True)
    reason, aborted = "ok", False
    try:
        try:
            obs = ep.reset(seed)
        except RuntimeError as exc:
            raise EnvironmentError_("reset: {}".format(str(exc)[:200]))
        try:
            clock.timed(solver.reset)
            t, done = 0, False
            while not done:
                if clock.s > budget:
                    state["budget_exhausted"] = True
                    raise SolverAborted("controller time budget exhausted")
                if watch and t % CHECK_EVERY == 0:
                    outside.extend(outside_processes(uid))
                    if outside:
                        raise SolverAborted("a process of the controller's account runs outside the launcher")
                if cap and len(_agent_pids(uid)) > cap:      # at every request (the kit's own count is every 64th)
                    solver._broken = True                    # the controller is started again before the next episode
                    raise SolverAborted("too many processes in the controller's account (more than {})".format(cap))
                if state["egress_mode"] == "detect" and agent_inet_sockets(uid):
                    state["egress_hits"] += 1
                action = ask(lock, solver, ep, t, obs, truth, clock)
                if not S.valid_action(action):
                    raise SolverAborted("malformed action")
                clock.s += image_cost * S.images_asked(action)
                state["images"] += S.images_asked(action)
                try:
                    obs, done = ep.step(action)
                except RuntimeError as exc:
                    raise EnvironmentError_("step: {}".format(str(exc)[:200]))
                t += 1
            if watch:
                outside.extend(outside_processes(uid))
        except (SolverAborted, ValueError, TypeError, OverflowError, KeyError, AttributeError, RecursionError) as exc:
            aborted = True
            reason = "aborted: {}".format(str(exc)[:160])
        try:
            res = ep.finish()                # the cable is judged as it lies, whoever ended the episode
        except RuntimeError as exc:
            raise EnvironmentError_("finish: {}".format(str(exc)[:200]))
        return res, reason, aborted
    finally:
        ep.close()


def run_episodes(config, uid, egress_mode, workdir, truth, seeds):
    """Run the fixed set over one controller. Returns (records, lock, state)."""
    lock = IdleAwareProbeLock(None, rate=config.get("probe_rate", 0.01),
                              pass_fraction=config.get("probe_pass_fraction", 0.1), tolerance=1e-6)
    clock = lock.clock = Clock()
    outside = lock.outside = []
    watch = egress_mode == "seccomp"
    state = {"budget_exhausted": False, "egress_hits": 0, "egress_mode": egress_mode, "images": 0, "reruns": 0}
    max_reruns = int(config.get("max_reruns", 2))
    budget = float(config.get("grader_budget_s", 3000))
    records = []
    GuardedSolver.guard = egress_mode == "seccomp"
    GuardedSolver.nproc = int(config.get("solver_nproc_limit", 0))
    GuardedSolver.max_processes = int(config.get("solver_max_processes", 64))
    solver = clock.timed(GuardedSolver, step_timeout=config.get("step_timeout_s", 30), submission=workdir)
    try:
        for n, seed in enumerate(seeds):
            skipped = None
            if outside:
                skipped = "a process of the controller's account ran outside the launcher"
            elif clock.s > budget:
                state["budget_exhausted"] = True
                skipped = "controller time budget exhausted"
            if skipped:
                records.append({"index": n, "credit": 0.0, "prefix": 0.0, "reason": skipped, "steps": 0, "fixtures": 0,
                                "routed": 0, "damaged": False, "aborted": False, "unstable": False, "run": False})
                log("episode {}: credit 0 not run ({})".format(n, skipped))
                continue
            for attempt in (0, 1):
                lock.rng = probe_stream(n, seed)
                mark = (clock.s, lock.counts(), state["images"])
                try:
                    res, reason, aborted = run_one(n, seed, solver, lock, clock, config, uid, watch, truth, outside,
                                                   state)
                    break
                except S.CellCrashed as exc:
                    state["reruns"] += 1
                    log("episode {}: the simulation's process ended by itself at attempt {} ({}); re-run {} of this "
                        "run".format(n, attempt + 1, str(exc)[:80], state["reruns"]))
                    if attempt == 1 or state["reruns"] > max_reruns:
                        raise EnvironmentError_("episode {}: the simulation's process ended by itself {}".format(
                            n, "twice in a row" if attempt == 1 else "more than {} times in this run".format(max_reruns)))
                    clock.s, state["images"] = mark[0], mark[2]
                    lock.restore(mark[1])
            credit = float(res.get("credit", 0.0)) if not outside else 0.0
            prefix = float(res.get("prefix_credit", 0.0)) if not outside else 0.0
            if not math.isfinite(credit) or credit < 0:
                credit = 0.0
            rec = {"index": n, "credit": credit, "prefix": prefix, "reason": reason, "steps": int(res.get("steps", 0)),
                   "fixtures": int(res.get("fixtures", 0)), "routed": int(res.get("routed", 0)),
                   "damaged": bool(res.get("damaged", False)), "aborted": aborted, "unstable": bool(res.get("unstable", False)),
                   "run": True, "handed_over": bool(res.get("handed_over", False)),
                   "peak": float(res.get("peak_mean_tension", 0.0))}
            records.append(rec)
            log("episode {}: credit {:.0f} route {}/{} damaged {} peak 0.1 s tension {:.1f} N steps {} {}{}".format(
                n, credit, rec["routed"], rec["fixtures"], rec["damaged"], rec["peak"], rec["steps"],
                "handed over" if rec["handed_over"] else "not handed over", "" if reason == "ok" else " " + reason))
    finally:
        try:
            solver.close()
        except Exception:  # noqa: BLE001 - closing a broken solver must not crash the grader
            pass
    return records, lock, state


def main():
    t_start = time.monotonic()
    secured = {"root_path_dirs_secured": float(len(secure_exec_path()))}   # before anything else runs
    config = stage_config()
    anchor, meta = read_anchor()
    try:
        uid, gid = _agent_ids()
    except KeyError:
        write_verdict(zero_verdict(reason="no agent account", anchored=anchor is not None, extra=secured))
        return
    cleared = clear_agent_account()
    secured["agent_account_cleared"] = 1.0 if cleared.get("cleared") else 0.0
    kill_agent_processes(uid)
    require_empty_account(uid)           # a task error if the sandbox still runs anything as the account by itself
    check_numerics()                     # a task error on a machine that computes the simulation differently
    secured["numerics_ok"] = 1.0
    seeds = load_episode_set(config)
    t_self = time.monotonic()
    try:
        render_selftest()
    except (RuntimeError, S.CellCrashed) as exc:
        fail("the cameras' renderer failed its self-test: {}".format(str(exc)[:200]))
    log("render self-test passed ({:.1f} s, two cameras)".format(time.monotonic() - t_self))
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
    log("egress guard: {}; oracle token: {}; {} episodes in the fixed set".format(egress_mode, truth, len(seeds)))
    try:
        records, lock, state = run_episodes(config, uid, egress_mode, workdir, truth, seeds)
    except EnvironmentError_ as exc:
        kill_agent_processes(uid)
        fail("the grader's own simulation failed: {}".format(exc))
    except (SolverMissing, SolverAborted) as exc:
        kill_agent_processes(uid)
        write_verdict(zero_verdict(reason=str(exc)[:200], anchored=anchor is not None, extra=secured))
        return
    cleanup_ok = kill_agent_processes(uid)
    credits = [r["credit"] for r in records]
    measured = sum(credits) / len(credits) if credits else 0.0
    if state["egress_hits"] or not cleanup_ok or not math.isfinite(measured) or measured < 0:
        measured = 0.0
    extra = dict(lock.rewards(),
                 episodes_in_set=float(len(seeds)),
                 episodes_full_route=float(sum(1 for r in records if r["credit"] >= 1.0)),
                 episodes_damaged=float(sum(1 for r in records if r["damaged"])),
                 episodes_aborted=float(sum(1 for r in records if r["aborted"])),
                 episodes_handed_over=float(sum(1 for r in records if r.get("handed_over"))),
                 episodes_not_run=float(sum(1 for r in records if not r["run"])),
                 mean_prefix_credit=sum(r["prefix"] for r in records) / max(1, len(records)),
                 sim_faults=float(sum(1 for r in records if r["unstable"])),
                 sim_reruns=float(state["reruns"]),
                 images_charged=float(state["images"]),
                 controller_time_s=float(lock.clock.s),
                 egress_guard_seccomp=1.0 if egress_mode == "seccomp" else 0.0,
                 egress_detected=float(state["egress_hits"]),
                 processes_outside_launcher=float(len(lock.outside)),
                 cleanup_failed=0.0 if cleanup_ok else 1.0,
                 budget_exhausted=1.0 if state["budget_exhausted"] else 0.0,
                 oracle_token=1.0 if truth else 0.0,
                 **secured)
    for k in sorted(set(r["fixtures"] for r in records if r["run"])):
        vals = [r["credit"] for r in records if r["run"] and r["fixtures"] == k]
        extra["score_{}_fixtures".format(k)] = sum(vals) / len(vals)
    verdict = ratio_rewards(measured, anchor, cap=config.get("scoring_cap", 1.5), higher_is_better=True,
                            metric_name="mean_route_credit", locks_ok=lock.ok, extra=extra,
                            full_at=meta.get("full_at", 1.0))
    write_verdict(verdict)
    for line in lock.outside[:5]:
        log("outside the launcher: {}".format(line))
    log("measured={:.5f} anchor={} ratio={:.3f} reward={:.3f} lock={} reruns={} egress={} controller_time={:.0f}s of "
        "{:.0f}s images={} wall={:.0f}s".format(
            measured, anchor, verdict["metrics"]["ratio"], verdict["rewards"]["reward"], lock.ok, state["reruns"],
            egress_mode, lock.clock.s, float(config.get("grader_budget_s", 3000)), state["images"],
            time.monotonic() - t_start))


if __name__ == "__main__":
    main()
