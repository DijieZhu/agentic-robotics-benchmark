"""Hidden grader for grocery_shelf_restocking (closed loop, hidden bays).

Runs as root in a SEPARATE verifier sandbox: no agent process exists there; the engine restores only the declared
artifact /home/user/submission. Structure and safeguards are lab_sample_handling's: it stops any agent-UID process
first, hands the restored submission (size-capped) to the agent account in place, starts `bash run.sh` through
nonet_exec.py (a seccomp filter: only AF_UNIX sockets) at the agent's UID, and speaks the kit wire (Solver,
per-request deadline, restarts; a ProbeLock, repeated `t` with a perturbed copy in a hidden order, that does not count
a pair in which both replies command no motion: IdleAwareProbeLock). Before anything else it reclaims root's PATH
directories (robotics_grader.stage.secure_exec_path).

Episodes: the fixed hidden set tests/hidden/episodes.json, the same for every submission: `episodes_per_unit`
episodes on every hidden unit (tests/units.json: a bay's design and its six SKUs), each a 64-bit seed (the bay's pose,
the planogram, the served lanes, the old stock and where it stands, the delivery in the tote, sensor noise), run in
the file's order (selected and shuffled by tests/review/select_set.py, the record in
tests/hidden/episode_selection.json). The probe lock draws which steps are probed, the order of a pair and the
perturbation from one stream per episode (probe_stream): the kit's `seed_rng` keyed by a BLAKE2b hash of a domain
tag, the episode's index in the set and its hidden seed, so an episode's probe schedule is a function of the episode
alone, and the grader adds no run-to-run randomness of its own. episode.py owns the credit and shelf_env.py,
shelf_world.py, shelf_ctl.py, shelf_sim.py and box_transport_env.py are the environment, all byte-identical to the
agent's copies. The metric `mean_episode_credit` is the mean over episodes of the episode credit (episode.py: 0.7
placement, 0.2 facing, 0.1 lanes kept).

Numerics: the operational-space controller inside the simulation calls LAPACK every physics step, and OpenBLAS picks
its kernels by CPU; a controller's own NumPy, SciPy and OpenCV arithmetic differs by CPU as well. The image pins both
(environment/Dockerfile: a .pth for every /opt/venv Python) and the launcher passes the same variables to run.sh
(nonet_exec.controller_env). `check_numerics` runs numerics_check.py before anything is graded and ends the run as a
task error on a machine that does not reproduce tests/numerics.json.

Time: `grader_budget_s` is charged to the controller's own time: what the grader waits for READY and for each reply
(`Clock`), plus `image_cost_s` for every image an applied action asks for, a fixed price whatever this machine takes
to draw it. The simulation and the drawing are the grader's and are not counted, so the budget does not depend on
how fast the grading machine simulates or renders.

Oracle token: a submission whose .oracle_token hashes to tests/hidden/oracle_token.sha256 is the reference; only
then does every observation also carry `truth` (the unit, the seed, the full simulation state). An untokened
submission never sees it. Every path agent behaviour can reach is charged as a score, never a crash.

Images: shelf_env draws the cameras and the catalog through shelf_render.py (a render process per episode), exactly as the agent's copy
does. Before the first episode a render self-test draws the four cameras and the catalog of one episode (the set's
first unit at a fixed seed, not a graded episode) and checks their shapes, that none is flat and that their
brightness is sane; the metric `render_selftest` records that it passed. A renderer failure, in the self-test or at
any later image (shelf_env.RenderError), is the grader's own error: the run ends as a task error
(robotics_grader.stage.fail: no reward written, exit status 1), never as a controller fault or a 0 for the
submission.
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

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

import episode as EP  # noqa: E402
import shelf_env as E  # noqa: E402
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
NUMERICS = os.path.join(STAGE, "numerics_check.py")
NUMERICS_REF = os.path.join(STAGE, "numerics.json")
NUMERICS_TIMEOUT_S = 300
UNITS = os.path.join(STAGE, "units.json")
TOKEN_SHA = os.path.join(STAGE, "hidden", "oracle_token.sha256")
EPISODES = os.path.join(STAGE, "hidden", "episodes.json")
STILL = 1e-9                # a 9-number reply whose six motion numbers and lift number are within this of zero
                            # commands no motion
SELFTEST_SEED = 1           # the render self-test's episode (not one of the graded seeds)
PROBE_TAG = b"grocery_shelf_restocking verify probe schedule 1"   # probe_stream's domain tag
MAX_LINE = 1 << 22          # replies are short (an action is at most 16 numbers)


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


def check_numerics():
    """Run numerics_check.py in a fresh interpreter (a short scripted run of the simulation and a set of NumPy, SciPy
    and OpenCV results computed as a controller's process computes them; sha256) and compare with tests/numerics.json.
    A machine that does not reproduce the recorded fingerprint would run every controller along another trajectory,
    so it must not grade: a task error, never the submission's."""
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
             "differs in {}. Grading here would run every controller along another trajectory".format(
                 got[:16], want[:16], report.get("differs")))
    return got


class Clock(object):
    """The controller's own time (s): what the grader waited for READY and for each reply, and what `run_episodes`
    adds for the images an applied action asks for. The simulation and the drawing happen outside `timed`."""

    def __init__(self):
        self.s = 0.0

    def timed(self, fn, *args, **kwargs):
        t0 = time.monotonic()
        try:
            return fn(*args, **kwargs)
        finally:
            self.s += time.monotonic() - t0


def images_asked(action):
    """How many images the next observation carries for this (valid) action: the set bits of its camera mask."""
    if len(action) == 1:
        return 0
    cams = int(action[9] if len(action) == 10 else action[8])
    return bin(cams & E.ALL_CAMS).count("1")


class GuardedSolver(Solver):
    """The kit Solver, spawned through the seccomp launcher, with a reply-line cap."""

    guard = True
    nproc = 0

    def _spawn(self):
        uid, gid = _agent_ids()
        cmd = [sys.executable, "-I", NONET, "--uid", str(uid), "--gid", str(gid), "--rlimit-nproc", str(self.nproc),
               "--cwd", self.submission, "--", "bash", self.entry]
        if not self.guard:
            import nonet_exec
            cmd = (["/usr/sbin/runuser", "-u", agent_user(), "--", "/usr/bin/env", "-i"]
                   + ["{}={}".format(k, v) for k, v in sorted(nonet_exec.controller_env().items())]
                   + ["bash", self.entry])
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
    the practice grader. Ownership changes never follow a symlink (lchown, a walk that does not
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
    """The kit's ProbeLock, except that a pair in which BOTH replies command no motion is not counted, neither in the
    total nor as passed: a controller that holds still, having finished or given up, shows nothing either way. No
    motion: a 9-number Cartesian step whose six motion numbers and lift number are all within STILL of zero, or the
    finish action [0]. A joint step (10 numbers) or a step that sets a posture (16) can move the arm, so it always
    counts. Every other pair is the kit's (`ProbeLock.record`)."""

    def __init__(self, *args, **kwargs):
        ProbeLock.__init__(self, *args, **kwargs)
        self.idle_skipped = 0

    @staticmethod
    def still(action):
        if isinstance(action, list) and len(action) == 1 and action[0] == 0 and not isinstance(action[0], bool):
            return True
        if not isinstance(action, list) or len(action) != 9:
            return False
        for x in action[:6] + action[7:8]:
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
    (index, seed) alone. The environment draws from the seed only through NumPy generators seeded with its two 32-bit
    halves and a small tag (shelf_world.rng_of); this hash shares nothing with them. The seed is the
    only secret: PROBE_TAG is a domain separator, not a key."""
    digest = hashlib.blake2b(b"%s|%d|%d" % (PROBE_TAG, int(index), int(seed)), digest_size=32).digest()
    return seed_rng(entropy=int.from_bytes(digest, "little"))


def ask(lock, solver, ep, t, obs, unit, truth, clock=None):
    """Step `t`'s action: one request, or a probe pair when the lock's stream says so. Every step draws once from
    the stream (lock.due); a probed step draws twice more, both before either request is written: the seed of the
    perturbation, then the order of the pair (ProbeLock.probe). The draws a step consumes never depend on the
    replies, and a pair that is not counted (both replies command no motion) has consumed the same draws as a
    counted one."""
    timed = clock.timed if clock is not None else (lambda fn, *a, **k: fn(*a, **k))
    w = wire_obs(ep, obs, unit, truth)
    if lock.due():
        other = wire_obs(ep, ep.env.perturbed_observation(obs, np.random.default_rng(lock.rng.seed(bits=64))),
                         unit, truth)
        return timed(lock.probe, solver, t, w, perturbed=other)
    return timed(solver.act, t, w)


def render_selftest(unit_props, unit, seed=SELFTEST_SEED):
    """The four cameras and the catalog of one episode, before any episode is graded: the right shapes, not flat (a
    standard deviation of at least 8 levels) and a mean brightness of 20-235 levels. Raises E.RenderError."""
    env = E.ShelfEnv(unit_props, unit, render=True)
    try:
        obs = env.reset(seed)
        imgs = [(cam, obs[cam], (h, w, 3)) for cam, (_bit, w, h, *_r) in sorted(E.CAMS.items())]
        imgs += [("catalog " + s, img, (E.CATALOG_PX, E.CATALOG_PX, 3)) for s, img in sorted(obs["catalog"].items())]
        for name, img, shape in imgs:
            mean, std = float(img.mean()), float(img.std())
            if tuple(img.shape) != shape or img.dtype != np.uint8 or std < 8.0 or not 20.0 <= mean <= 235.0:
                raise E.RenderError("{} image: shape {}, dtype {}, mean {:.1f}, std {:.1f}".format(
                    name, tuple(img.shape), img.dtype, mean, std))
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


def wire_obs(ep, obs, uid, truth):
    w = E.to_wire(obs)
    w["unit"] = uid
    if truth:
        w["truth"] = ep.env.truth()
    return w


class Run(object):
    """One episode as the grader drives it: the environment, the credit, and the end (episode.py)."""

    def __init__(self, unit_props, unit, seed):
        self.env = E.ShelfEnv(unit_props, unit, render=True)
        self.seed = seed
        self.ep = None

    def reset(self):
        obs = self.env.reset(self.seed)
        self.ep = EP.Episode(self.env.w)
        return obs

    def step(self, action):
        obs = self.env.step(action)
        done = self.ep.step_done(self.env.finished) or self.env.unstable()
        return obs, done, {}

    def score(self):
        return self.ep.score()


def valid_action(action):
    """9 numbers (a Cartesian step with the lift), 16 (the same with a null-space posture), 10 (a joint step with the
    lift) or [0] (finished), all finite; the camera bits (the 9th of 9 or 16, the 10th of 10) an integer 0-15."""
    if not isinstance(action, list) or len(action) not in (1, 9, 10, 16):
        return False
    for x in action:
        if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x):
            return False
    if len(action) == 1:
        return action[0] == 0
    cams = action[9] if len(action) == 10 else action[8]
    return 0 <= cams <= E.ALL_CAMS and float(cams).is_integer()


def run_episodes(config, uid, egress_mode, t_start, units, workdir, truth, episodes):
    """Run `episodes` ([(unit, seed)], the fixed set in its order) over one controller. The budget
    (`grader_budget_s`) is charged to the controller's own time, `lock.clock`: what the grader waited for READY (the
    first start and every restart), for each reply and for the reset, plus `image_cost_s` for every image an applied
    action asks for. `t_start` is kept for the record only. Returns (scores, aborted,
    skipped, faults, per_unit, per_part, egress_hits, budget_exhausted, lock, records) (skipped: not started within
    the budget, scored 0; a simulator error or a non-finite state aborts the episode and counts as a fault); a
    record per episode: its index in the set, unit, score, reason and steps. During episode n the probe lock draws from probe_stream(n, seed); its
    counts run over all episodes. A renderer failure (E.RenderError) is not caught here: it is the caller's task
    error."""
    budget = float(config.get("grader_budget_s", 9000))
    image_cost = float(config.get("image_cost_s", 0.1))
    clock = Clock()
    lock = IdleAwareProbeLock(None, rate=config.get("probe_rate", 0.01),
                              pass_fraction=config.get("probe_pass_fraction", 0.1), tolerance=1e-4)
    scores, aborted, faults, records = [], 0, 0, []
    per_unit, per_part = {}, {"correct": [], "face": [], "kept": []}
    egress_hits = 0
    budget_exhausted = False
    skipped = 0
    GuardedSolver.guard = egress_mode == "seccomp"
    GuardedSolver.nproc = int(config.get("solver_nproc_limit", 0))
    lock.clock = clock
    solver = clock.timed(GuardedSolver, step_timeout=config.get("step_timeout_s", 30), submission=workdir)
    try:
        for n, (unit, seed) in enumerate(episodes):
            if clock.s > budget:
                budget_exhausted = True
                skipped += 1
                scores.append(0.0)
                per_unit.setdefault(unit, []).append(0.0)
                records.append({"index": n, "unit": unit, "score": 0.0, "reason": "controller time budget exhausted",
                                "steps": 0})
                log("episode {} ({}): score=0.0000 not run (controller time budget exhausted)".format(n, unit))
                continue
            lock.rng = probe_stream(n, seed)
            ep = Run(units[unit], unit, seed)
            score, reason = 0.0, "ok"
            try:
                obs = ep.reset()
                clock.timed(solver.reset)
                t, done = 0, False
                while not done:
                    if clock.s > budget:
                        budget_exhausted = True
                        raise SolverAborted("controller time budget exhausted")
                    if egress_mode == "detect" and agent_inet_sockets(uid):
                        egress_hits += 1
                    action = ask(lock, solver, ep, t, obs, unit, truth, clock)
                    if not valid_action(action):
                        raise SolverAborted("malformed action")
                    clock.s += image_cost * images_asked(action)
                    obs, done, _info = ep.step(action)
                    t += 1
                    if ep.env.unstable():
                        faults += 1
                        raise SolverAborted("simulation unstable")
                sc = ep.score()
                score = sc["credit"]
                per_part["correct"].append(float(sc["place"]))
                per_part["face"].append(float(sc["face"]))
                per_part["kept"].append(1.0 if sc["kept"] else 0.0)
            except (mujoco.FatalError, FloatingPointError) as exc:
                faults += 1                   # the simulator stopped (a fatal error or a non-finite state)
                aborted += 1
                score = 0.0
                reason = "aborted: simulator error: {}".format(str(exc)[:140])
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
            records.append({"index": n, "unit": unit, "score": score, "reason": reason, "steps": steps})
    finally:
        try:
            solver.close()
        except Exception:  # noqa: BLE001 - closing a broken solver must not crash the grader
            pass
    return scores, aborted, skipped, faults, per_unit, per_part, egress_hits, budget_exhausted, lock, records


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
    if not stop_agent_processes(uid):
        write_verdict(zero_verdict(reason="agent processes could not be stopped (cleanup_failed)",
                                   anchored=anchor is not None, extra=secured))
        return
    check_numerics()                     # a task error on a machine that computes differently
    secured["numerics_ok"] = 1.0
    with open(UNITS) as fh:
        units = dict(sorted(json.load(fh)["units"].items()))
    episodes = load_episode_set(config, units)
    t_self = time.monotonic()
    try:
        render_selftest(units[episodes[0][0]], episodes[0][0])
    except E.RenderError as exc:
        fail("the cameras' renderer failed its self-test: {}".format(exc))
    log("render self-test passed ({:.1f} s, four cameras and the catalog)".format(time.monotonic() - t_self))
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
    try:
        (scores, aborted, skipped, faults, per_unit, per_part, egress_hits, budget_exhausted,
         lock, _records) = run_episodes(config, uid, egress_mode, t_start, units, workdir, truth, episodes)
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
                 episodes_scored=float(len(scores) - aborted - skipped),
                 episodes_aborted=float(aborted),
                 episodes_skipped=float(skipped),
                 sim_faults=float(faults),
                 egress_guard_seccomp=1.0 if egress_mode == "seccomp" else 0.0,
                 egress_detected=float(egress_hits),
                 cleanup_failed=0.0 if cleanup_ok else 1.0,
                 budget_exhausted=1.0 if budget_exhausted else 0.0,
                 oracle_token=1.0 if truth else 0.0,
                 **secured)
    for unit, vals in sorted(per_unit.items()):
        extra["score_" + unit] = sum(vals) / len(vals)
    for part, vals in sorted(per_part.items()):
        if vals:
            extra["part_" + part] = sum(vals) / len(vals)
    verdict = ratio_rewards(measured, anchor, cap=config.get("scoring_cap", 1.5), higher_is_better=True,
                            metric_name="mean_episode_credit", locks_ok=lock.ok, extra=extra,
                            full_at=meta.get("full_at", 1.0))
    write_verdict(verdict)
    log("measured={:.5f} anchor={} ratio={:.3f} reward={:.3f} lock={} aborted={} faults={} egress={} "
        "controller={:.0f}s of {:.0f}s wall={:.0f}s".format(
            measured, anchor, verdict["metrics"]["ratio"], verdict["rewards"]["reward"], lock.ok, aborted, faults,
            egress_mode, lock.clock.s, float(config.get("grader_budget_s", 9000)), time.monotonic() - t_start))


if __name__ == "__main__":
    main()
