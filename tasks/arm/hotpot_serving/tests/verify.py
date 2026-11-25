"""Hidden grader for hotpot_serving (closed loop, hidden burner-pot-and-broth units).

Runs as root in a SEPARATE verifier sandbox: no agent process exists there, the setup stage has not run there, and
the engine restores only the declared artifact /home/user/submission. It stops any agent-UID process first, hands the
restored submission (at most MAX_SUBMISSION_FILES files and MAX_SUBMISSION_BYTES bytes) to the agent account in
place, starts `bash run.sh` in it through nonet_exec.py (a seccomp filter: only AF_UNIX sockets) at the agent's UID,
and speaks the kit wire through `soup_grade.run_episodes` (the kit Solver: READY within 120 s, each reply within 30 s,
a reply line at most 4 MiB, at most 3 restarts; the probe lock, IdleAwareProbeLock). Before anything else it reclaims
root's PATH directories (robotics_grader.stage.secure_exec_path).

Episodes: the fixed hidden set tests/hidden/episodes.json, the same for every submission: `episodes_per_unit`
episodes on each hidden unit (tests/units.json: the pot's empty mass, the broth's volume and discharge coefficient,
the burner's full power, the pieces' size and friction), each a 64-bit seed (which bowl stands where, the order, the
pot's place, which dish holds which kind and the pieces on them, the sensor and image noise), run in the file's order
(selected and shuffled by tests/review/select_set.py; the record is tests/hidden/episode_selection.json). The probe schedule of an episode is a function of its index in the set and its
seed alone (soup_grade.probe_stream), so the grader adds no run-to-run randomness of its own: a deterministic
controller that stays inside the time limits gets one result. episode.py owns the credit; soup_env.py and soup_sim.py
are the environment (all three byte-identical to the agent's copies). The metric `mean_episode_credit` is the mean
over the set of each episode's credit (episode.py); an episode that is aborted or not run counts 0.

The budget (grader_config.json `grader_budget_s`) is the controller's own time: what the grader waits for READY and
for every reply (soup_grade.Clock), plus `image_cost_s` for every image an applied action asks for, a fixed price
whatever this machine takes to draw it. The simulation and the drawing are not charged. The episode running when the
budget runs out is aborted and the later ones are not run: they score 0 and the metric `episodes_scored` does not
count them. A simulator fatal error (mujoco.FatalError) or an unstable simulation scores that episode 0 and grading
goes on.

Before any agent code runs, and before the controller's clock exists, the grader clears the agent's account with the
kit (robotics_grader.account.clear_agent_account: the desktop session the sandbox image runs as that account is taken
down for good) and reads the report: an account that could not be cleared is a task error, because the controller's
confinement (no network, one process tree) could not be relied on.

Numerics: the simulation's operational-space controller calls LAPACK every physics step, and OpenBLAS picks its
kernels by CPU; on another kernel the same controller runs another trajectory. The image pins the kernels
(OPENBLAS_CORETYPE=Haswell with OPENBLAS_L2_SIZE=256, also for a Python started without the variables:
environment/Dockerfile); tests/numerics_check.py runs a short scripted simulation and the grader compares its
fingerprint with tests/numerics.json before grading. A machine that does not reproduce it is a task error.

Oracle token: a submission whose .oracle_token hashes to tests/hidden/oracle_token.sha256 is the reference; only
then does every observation also carry `truth` (the unit, the seed, the full simulation state). An untokened
submission never sees it.

Images: soup_sim.render draws the cameras with MuJoCo's own renderer (OSMesa on the CPU), exactly as the agent's copy
does. Before
the first episode a render self-test draws every camera of one episode (the set's first unit at a fixed seed, not a
graded episode) and checks their shapes, that none is flat and that their brightness is sane (metric
`render_selftest`). A renderer failure, in the self-test or at any later image (soup_env.RenderError), is the grader's
own error: the run ends as a task error (robotics_grader.stage.fail: no reward written, exit status 1), never as a
controller fault or a 0 for the submission.
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

import soup_env as E  # noqa: E402
import soup_grade as G  # noqa: E402
from soup_grade import MAX_LINE  # noqa: E402
from robotics_grader import (  # noqa: E402
    Solver,
    SolverAborted,
    SolverMissing,
    agent_user,
    ratio_rewards,
    read_anchor,
    stage_config,
    write_verdict,
    zero_verdict,
)
from robotics_grader.account import clear_agent_account  # noqa: E402
from robotics_grader.stage import fail, secure_exec_path  # noqa: E402

STAGE = os.path.dirname(os.path.abspath(__file__))
MAX_SUBMISSION_BYTES = 200 * 1024 * 1024
MAX_SUBMISSION_FILES = 5000
NONET = os.path.join(STAGE, "nonet_exec.py")
UNITS = os.path.join(STAGE, "units.json")
TOKEN_SHA = os.path.join(STAGE, "hidden", "oracle_token.sha256")
EPISODES = os.path.join(STAGE, "hidden", "episodes.json")
NUMERICS = os.path.join(STAGE, "numerics_check.py")
NUMERICS_REF = os.path.join(STAGE, "numerics.json")
NUMERICS_TIMEOUT_S = 300


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
                break          # an oversize reply: a closed wire (the episode aborts)
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


def load_episode_set(config, units, path=EPISODES):
    """The fixed hidden set: [(index, unit, seed)] in the graded order. `episodes_per_unit` and the units must agree
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
    return [(n, u, s) for n, (u, s) in enumerate(eps)]


def grader_config(config):
    out = dict(G.CONFIG)
    for k in out:
        if k in config:
            out[k] = config[k]
    return out


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
    # The kit's clearing of the account (every supervisord program switched off, every process of the account killed
    # until none comes back), called here: before any agent code and before the controller's clock exists. Its report
    # is read: outside the engine's verify stage it is skipped (a maintainer's tool), inside it must say `cleared`.
    # Solver.__init__ calls it again inside the timed start and gets this report back at once, so none of its
    # seconds is charged to the controller.
    cleared = clear_agent_account()
    if not cleared.get("cleared") and not cleared.get("skipped"):
        fail("the agent's account could not be cleared before grading ({} process(es) keep coming back: {}); the "
             "controller's confinement cannot be relied on in this sandbox".format(
                 len(cleared.get("left", [])), "; ".join(cleared.get("left", []))[:300] or "?"))
    secured["agent_account_cleared"] = 1.0 if cleared.get("cleared") else 0.0
    if not stop_agent_processes(uid):
        write_verdict(zero_verdict(reason="agent processes could not be stopped (cleanup_failed)",
                                   anchored=anchor is not None, extra=secured))
        return
    check_numerics()                     # a task error on a machine that computes the simulation differently
    secured["numerics_ok"] = 1.0
    with open(UNITS) as fh:
        units = dict(sorted(json.load(fh)["units"].items()))
    episodes = load_episode_set(config, units)
    t_self = time.monotonic()
    try:
        G.render_selftest(units[episodes[0][1]], episodes[0][1])
    except E.RenderError as exc:
        fail("the cameras' renderer failed its self-test: {}".format(exc))
    log("render self-test passed ({:.1f} s, four cameras)".format(time.monotonic() - t_self))
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
    GuardedSolver.guard = egress_mode == "seccomp"
    GuardedSolver.nproc = int(config.get("solver_nproc_limit", 0))
    egress = {"hits": 0}

    def watch():
        if egress_mode == "detect" and agent_inet_sockets(uid):
            egress["hits"] += 1

    gconfig = grader_config(config)
    clock = G.Clock()                    # the controller's own time: READY, every reply, the images' fixed price
    try:
        solver = clock.timed(GuardedSolver, step_timeout=gconfig["step_timeout_s"], submission=workdir)
        try:
            records, lock = G.run_episodes(solver, episodes, units, truth, log=log, config=gconfig,
                                           clock=clock, on_step=watch)
        finally:
            try:
                solver.close()
            except Exception:  # noqa: BLE001 - closing a broken solver must not crash the grader
                pass
    except E.RenderError as exc:
        kill_agent_processes(uid)
        fail("the cameras' renderer failed during the episodes: {}".format(exc))
    except (SolverMissing, SolverAborted) as exc:
        kill_agent_processes(uid)
        write_verdict(zero_verdict(reason=str(exc)[:200], anchored=anchor is not None, extra=secured))
        return
    cleanup_ok = stop_agent_processes(uid)
    scores = [r["credit"] for r in records]
    measured = sum(scores) / len(scores) if scores else 0.0
    if egress["hits"] or not cleanup_ok or not math.isfinite(measured) or measured < 0:
        measured = 0.0
    extra = dict(lock.rewards(),
                 episodes_in_set=float(len(episodes)),
                 episodes_scored=float(sum(1 for r in records if r["scored"])),
                 episodes_aborted=float(sum(1 for r in records if r["reason"].startswith("aborted"))),
                 episodes_not_run=float(sum(1 for r in records if r["reason"].startswith("not run"))),
                 sim_faults=float(sum(1 for r in records if r["sim_fault"])),
                 egress_guard_seccomp=1.0 if egress_mode == "seccomp" else 0.0,
                 egress_detected=float(egress["hits"]),
                 cleanup_failed=0.0 if cleanup_ok else 1.0,
                 budget_exhausted=1.0 if any("budget" in r["reason"] for r in records) else 0.0,
                 controller_seconds=round(float(clock.s), 1),
                 oracle_token=1.0 if truth else 0.0,
                 **secured)
    per_unit = {}
    for r in records:
        per_unit.setdefault(r["unit"], []).append(r["credit"])
    for unit, vals in sorted(per_unit.items()):
        extra["score_" + unit] = sum(vals) / len(vals)
    for part, vals in sorted(G.parts(records).items()):
        if vals:
            extra["part_" + part] = sum(vals) / len(vals)
    verdict = ratio_rewards(measured, anchor, cap=config.get("scoring_cap", 1.5), higher_is_better=True,
                            metric_name="mean_episode_credit", locks_ok=lock.ok, extra=extra,
                            full_at=meta.get("full_at", 1.0))
    write_verdict(verdict)
    log("measured={:.5f} anchor={} ratio={:.3f} reward={:.3f} lock={} scored={} faults={} egress={} "
        "controller_time={:.0f}s of {:.0f}s wall={:.0f}s".format(
            measured, anchor, verdict["metrics"]["ratio"], verdict["rewards"]["reward"], lock.ok,
            extra["episodes_scored"], extra["sim_faults"], egress_mode, clock.s,
            float(gconfig["grader_budget_s"] or 0), time.monotonic() - t_start))


if __name__ == "__main__":
    main()
