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
probe lock keeps its own randomness (the kit's `seed_rng`, OS entropy): which steps are probed, the order of a
pair and the perturbation. episode.py owns the requests and the credit (byte-identical to the agent's copy);
appliance_env.py is the environment (byte-identical too). The metric `mean_request_credit` is the mean over
episodes of the mean request credit (reached 0.5 + completed 0.5). Each episode's requests and their outcomes are
logged on stderr with the episode's index in the set (never its seed).

Oracle token: a submission whose .oracle_token hashes to tests/hidden/oracle_token.sha256 is the reference;
only then does every observation also carry `truth` (the unit's placement jitter, its joint values, the
microwave's flag). An untokened submission never sees it. Every path agent behaviour can reach is charged as
a score, never a crash.
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
from robotics_grader.stage import secure_exec_path  # noqa: E402

STAGE = os.path.dirname(os.path.abspath(__file__))
MAX_SUBMISSION_BYTES = 200 * 1024 * 1024
MAX_SUBMISSION_FILES = 5000
NONET = os.path.join(STAGE, "nonet_exec.py")
UNITS = os.path.join(STAGE, "units.json")
TOKEN_SHA = os.path.join(STAGE, "hidden", "oracle_token.sha256")
EPISODES = os.path.join(STAGE, "hidden", "episodes.json")
STILL = 1e-9                # a reply whose six motion numbers are all within this of zero commands no motion
MAX_LINE = 1 << 22          # an observation with three images is about 1 MB; replies are short


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


def run_episodes(config, rng, uid, egress_mode, t_start, specs, workdir, truth, episodes):
    """Run `episodes` ([(unit, seed)], the fixed set in its order) over one controller. Returns (scores, aborted,
    faults, per_unit, per_request, egress_hits, budget_exhausted, lock, records); a record per episode: its index
    in the set, unit, score, reason, steps and the requests' outcomes (episode.score())."""
    budget = float(config.get("grader_budget_s", 6000))
    lock = IdleAwareProbeLock(rng, rate=config.get("probe_rate", 0.01),
                              pass_fraction=config.get("probe_pass_fraction", 0.1), tolerance=1e-4)
    scores, aborted, faults = [], 0, 0
    per_unit, per_request = {}, {}
    records = []
    egress_hits = 0
    budget_exhausted = False
    GuardedSolver.guard = egress_mode == "seccomp"
    GuardedSolver.nproc = int(config.get("solver_nproc_limit", 0))
    solver = GuardedSolver(step_timeout=config.get("step_timeout_s", 30), submission=workdir)
    try:
        for n, (unit, seed) in enumerate(episodes):
            if time.monotonic() - t_start > budget:
                budget_exhausted = True
                scores.append(0.0)
                per_unit.setdefault(unit, []).append(0.0)
                records.append({"index": n, "unit": unit, "score": 0.0, "reason": "grader budget exhausted",
                                "steps": 0, "requests": []})
                log("episode {} ({}): score=0.0000 not run (grader budget exhausted)".format(n, unit))
                continue
            ep = EP.Episode(E.ApplianceEnv(specs[unit], None, render=True), seed)
            score, reason, sc = 0.0, "ok", None
            try:
                obs = ep.reset()
                solver.reset()
                t, done = 0, False
                while not done:
                    if time.monotonic() - t_start > budget:
                        budget_exhausted = True
                        raise SolverAborted("grader budget exhausted")
                    if egress_mode == "detect" and agent_inet_sockets(uid):
                        egress_hits += 1
                    w = wire_obs(ep, obs, unit, truth)
                    if lock.due():
                        other = ep.env.perturbed_observation(obs, np.random.default_rng(rng.seed(bits=64)))
                        action = lock.probe(solver, t, w, perturbed=wire_obs(ep, other, unit, truth))
                    else:
                        action = solver.act(t, w)
                    if not valid_action(action):
                        raise SolverAborted("malformed action")
                    obs, done, _info = ep.step(action)
                    t += 1
                    if ep.env.unstable():
                        faults += 1
                        raise SolverAborted("simulation unstable")
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
    rng = seed_rng()
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
        (scores, aborted, faults, per_unit, per_request, egress_hits, budget_exhausted,
         lock, _records) = run_episodes(config, rng, uid, egress_mode, t_start, specs, workdir, truth, episodes)
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
    log("measured={:.5f} anchor={} ratio={:.3f} reward={:.3f} lock={} aborted={} faults={} egress={}".format(
        measured, anchor, verdict["metrics"]["ratio"], verdict["rewards"]["reward"], lock.ok, aborted, faults,
        egress_mode))


if __name__ == "__main__":
    main()
