"""Hidden grader for arm_hidden_mechanism_box (closed loop, hidden boxes).

Runs as root in a SEPARATE verifier sandbox (SPEC 14.10: no lab and no agent process exist there;
the engine restores only the declared artifact /home/user/submission). It never talks to the lab.
Defensively it still stops any lab or agent-UID process first (rev-leak R14, SPEC L3), then hands the
restored submission (size-capped) to the agent account in place, at /home/user/submission, the folder the
instruction names (ownership changes never follow a symlink). It owns its own copy of
the robot environment (box_env.py, byte-identical to the agent's dev sim) and builds the hidden boxes
from tests/ref/boxes.json with mech.py, draws 64-bit episode seeds from OS entropy (kit `seed_rng`:
box placement jitter and sensor noise), drives the agent's controller over the kit wire at the agent's
UID, and scores every episode from its own simulator state (fraction of the box's stages unlocked).

Additions to the kit wire (the vendored kit stays byte-identical):
  * GuardedSolver: the solver is started through nonet_exec.py, which installs a seccomp filter
    (only AF_UNIX sockets), drops to the agent account and execs `bash run.sh` with a whitelist
    environment. Reply lines are capped at 64 KiB.
  * A start-up self-test proves the network block (`egress_guard`); if the filter cannot be
    installed, /proc/net is polled every step and any agent-UID internet socket zeroes the run.
  * Every agent-UID process is killed (bounded, fail-closed) before the solver starts and after it
    closes. One that outlives the bounded kill (the base image's supervisord respawns the account's
    desktop as fast as it is killed) is stopped; only one still running once stopped fails the run.
  * A wall-clock budget below the stage timeout: on overrun the remaining episodes count 0.
  * Every path agent behaviour can reach is charged as a score, never a crash.
"""

import json
import math
import os
import pwd
import signal
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import box_env as E  # noqa: E402
import mech  # noqa: E402
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

STAGE = os.path.dirname(os.path.abspath(__file__))
MAX_SUBMISSION_BYTES = 200 * 1024 * 1024
MAX_SUBMISSION_FILES = 5000
NONET = os.path.join(STAGE, "nonet_exec.py")
BOXES = os.path.join(STAGE, "ref", "boxes.json")
MAX_LINE = 65536


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


def episode_plan(config, rng, ids):
    reps = int(config.get("episodes_per_box", 2))
    plan = [j for j in sorted(ids) for _ in range(reps)]
    for i in range(len(plan) - 1, 0, -1):          # Fisher-Yates with the keyed grader RNG
        k = int(rng.random() * (i + 1))
        plan[i], plan[k] = plan[k], plan[i]
    return plan


def run_episodes(config, rng, uid, egress_mode, t_start, specs, workdir):
    budget = float(config.get("grader_budget_s", 6000))
    plan = episode_plan(config, rng, specs)
    lock = ProbeLock(rng, rate=config.get("probe_rate", 0.01),
                     pass_fraction=config.get("probe_pass_fraction", 0.1), tolerance=1e-4)
    scores, aborted, faults = [], 0, 0
    per_box = {}
    per_stage = {}
    egress_hits = 0
    budget_exhausted = False
    GuardedSolver.guard = egress_mode == "seccomp"
    GuardedSolver.nproc = int(config.get("solver_nproc_limit", 0))
    solver = GuardedSolver(step_timeout=config.get("step_timeout_s", 30), submission=workdir)
    try:
        for ep, box in enumerate(plan):
            if time.monotonic() - t_start > budget:
                budget_exhausted = True
                scores.append(0.0)
                per_box.setdefault(box, []).append(0.0)
                continue
            env = E.BoxEnv(specs[box], box_id=box, early_end=True)
            score = 0.0
            reason = "ok"
            try:
                obs = env.reset(rng.seed(bits=64))
                solver.reset()
                for t in range(env.max_steps):
                    if time.monotonic() - t_start > budget:
                        budget_exhausted = True
                        raise SolverAborted("grader budget exhausted")
                    if egress_mode == "detect" and agent_inet_sockets(uid):
                        egress_hits += 1
                    wire = E.to_wire(obs)
                    if lock.due():
                        other = E.to_wire(env.perturbed_observation(obs, rng))
                        action = lock.probe(solver, t, wire, perturbed=other)
                    else:
                        action = solver.act(t, wire)
                    obs, done, info = env.step(action)
                    if done:
                        break
                if env.fault is not None:
                    faults += 1
                    reason = "sim fault: " + env.fault
                    score = 0.0
                else:
                    score = env.episode_score()
                for (st, _qa), ok in zip(env.stages, env.achieved):
                    per_stage.setdefault(box + "_" + st["id"], []).append(1.0 if (ok and env.fault is None) else 0.0)
            except (SolverAborted, E.ActionError, ValueError, TypeError, OverflowError, KeyError,
                    AttributeError, RecursionError) as exc:
                aborted += 1
                score = 0.0
                reason = "aborted: {}".format(str(exc)[:160])
            finally:
                env.close()
            if not (isinstance(score, float) and math.isfinite(score)) or score < 0:
                score = 0.0
            scores.append(score)
            per_box.setdefault(box, []).append(score)
            log("episode {} ({}): score={:.4f} steps={} {}".format(ep, box, score, getattr(env, "t", 0), reason))
    finally:
        try:
            solver.close()
        except Exception:  # noqa: BLE001 - closing a broken solver must not crash the grader
            pass
    return scores, aborted, faults, per_box, per_stage, egress_hits, budget_exhausted, lock


def main():
    t_start = time.monotonic()
    config = stage_config()
    lab_stopped = stop_lab()
    rng = seed_rng()
    anchor, meta = read_anchor()
    try:
        uid, gid = _agent_ids()
    except KeyError:
        write_verdict(zero_verdict(reason="no agent account", anchored=anchor is not None))
        return
    if not stop_agent_processes(uid):
        write_verdict(zero_verdict(reason="agent processes could not be stopped (cleanup_failed)",
                                   anchored=anchor is not None))
        return
    with open(BOXES) as fh:
        cfgs = json.load(fh)
    specs = {}
    for bid, cfg in sorted(cfgs.items()):
        box = mech.Box(cfg)
        specs[bid] = {"xml": box.xml(), "meta": box.meta()}
    try:
        workdir = stage_submission(uid, gid)
    except OSError as exc:
        log("staging the submission failed: {}".format(type(exc).__name__))
        workdir = None
    if workdir is None:
        write_verdict(zero_verdict(reason="no usable submission", anchored=anchor is not None))
        return
    egress_mode = "seccomp" if egress_self_test(uid, gid, int(config.get("solver_nproc_limit", 0))) else "detect"
    log("egress guard: " + egress_mode)
    try:
        (scores, aborted, faults, per_box, per_stage, egress_hits, budget_exhausted,
         lock) = run_episodes(config, rng, uid, egress_mode, t_start, specs, workdir)
    except (SolverMissing, SolverAborted) as exc:
        kill_agent_processes(uid)
        write_verdict(zero_verdict(reason=str(exc)[:200], anchored=anchor is not None))
        return
    cleanup_ok = stop_agent_processes(uid)
    measured = sum(scores) / len(scores) if scores else 0.0
    if egress_hits or not cleanup_ok or not math.isfinite(measured) or measured < 0:
        measured = 0.0
    extra = dict(lock.rewards(),
                 episodes_scored=float(len(scores) - aborted),
                 episodes_aborted=float(aborted),
                 sim_faults=float(faults),
                 egress_guard_seccomp=1.0 if egress_mode == "seccomp" else 0.0,
                 egress_detected=float(egress_hits),
                 cleanup_failed=0.0 if cleanup_ok else 1.0,
                 lab_stopped=1.0 if lab_stopped else 0.0,
                 budget_exhausted=1.0 if budget_exhausted else 0.0)
    for box, vals in sorted(per_box.items()):
        extra["score_" + box] = sum(vals) / len(vals)
    for key, vals in sorted(per_stage.items()):
        extra["stage_" + key] = sum(vals) / len(vals)
    verdict = ratio_rewards(measured, anchor, cap=config.get("scoring_cap", 1.5), higher_is_better=True,
                            metric_name="mean_opening_progress", locks_ok=lock.ok, extra=extra,
                            full_at=meta.get("full_at", 1.0))
    write_verdict(verdict)
    log("measured={:.5f} anchor={} ratio={:.3f} reward={:.3f} lock={} aborted={} faults={} egress={}".format(
        measured, anchor, verdict["metrics"]["ratio"], verdict["rewards"]["reward"], lock.ok, aborted, faults,
        egress_mode))


if __name__ == "__main__":
    main()
