"""Hidden grader for keyed_nut_peg_assembly (closed loop, hidden nuts, fixed hidden episodes).

Runs as root in a SEPARATE verifier sandbox (SPEC 14.10: no agent process exists there; the engine
restores only the declared artifact /home/user/submission). Defensively it still stops any agent-UID
process first (rev-leak R14, SPEC L3), then hands the
restored submission (size-capped) to the agent account in place, at /home/user/submission, the folder the
instruction names (ownership changes never follow a symlink). It owns its own copy of
the robot cell (keyed_env.py; the agent never reads it: its sandbox has the cell's data sheet and a commissioning
service that runs a byte copy of this file on development kits, under an account of its own) and the credit rule
(keyed_credit.py, grader-only), plays the fixed hidden episodes of tests/ref/episodes.json in order (each: a
kit of three nuts and its work order, drawn with the public `sample_kit` from a seed, and the layout and
noise seed; the set and its pre-registered selection: tests/hidden/selection.json), drives the agent's
controller over the kit wire at the agent's UID, and scores every episode from its own simulator state
(`GradedNutEnv.episode_score`: 1 if the whole kit was assembled, every nut on its own peg, else 0). The
metric is the fraction of complete kits; the fraction of nuts assembled and the staged
`GradedNutEnv.progress` are reported as diagnostics only.

Time: every kit has a controller budget (grader_config.json controller_budget_s): the time the grader waits
for the controller (launch to READY and the reset reply, then every reply) plus a fixed charge per image the
controller asks for (image_charge_s, per camera). The simulator's own time is not charged, so the grading
machine's speed does not change a score; a kit whose controller exhausts its budget stops at that step and is
scored on the state it reached. A wall-clock cap on the whole run (wall_cap_s, below the stage timeout) is only
a safety net: the episode under way when it is reached and the ones after it score 0.

Reference-solution access: if the submission holds a file `oracle_token` whose SHA-256 equals
tests/oracle_token.sha256, every observation also carries `_gt` (the ground-truth scene state). Only
solution/ has the token; the hash is useless without it. The probe lock is recorded, never gating.

No memory between episodes: every episode starts a new controller process from a fresh copy of the
submission as restored (kept root-only), after every agent-UID process has been stopped and every
file the agent account owns in the writable places outside the submission has been removed.

Additions to the kit wire (the vendored kit stays byte-identical):
  * GuardedSolver: the solver is started through nonet_exec.py, which installs a seccomp filter
    (only AF_UNIX sockets), drops to the agent account and execs `bash run.sh` with a whitelist
    environment. Reply lines are capped at 64 KiB.
  * A start-up self-test proves the network block (`egress_guard`); if the filter cannot be
    installed, /proc/net is polled every step and any agent-UID internet socket zeroes the run.
  * Every agent-UID process is killed (bounded, fail-closed) before the solver starts and after it
    closes. One that outlives the bounded kill (the base image's supervisord respawns the account's
    desktop as fast as it is killed) is stopped; only one still running once stopped fails the run.
  * The per-kit controller budget and the wall-clock safety net above.
  * Every path agent behaviour can reach is charged as a score, never a crash.
"""

import json
import math
import os
import pwd
import shutil
import signal
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import hashlib  # noqa: E402

import keyed_credit as KC  # noqa: E402
import keyed_env as E  # noqa: E402
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

STAGE = os.path.dirname(os.path.abspath(__file__))
MAX_SUBMISSION_BYTES = 200 * 1024 * 1024
MAX_SUBMISSION_FILES = 5000
NONET = os.path.join(STAGE, "nonet_exec.py")
EPISODES = os.path.join(STAGE, "ref", "episodes.json")
TOKEN_HASH = os.path.join(STAGE, "oracle_token.sha256")
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


def image_charge(action, charges):
    """(charge in s, count) of the images an action asks for; (0, 0) for a malformed action (env.step rejects it)."""
    try:
        _mv, _g, render = E.parse_action(action)
    except (E.ActionError, TypeError, ValueError, OverflowError):
        return 0.0, 0
    return sum(float(charges[c]) for c in render), len(render)


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


def stage_submission(uid, gid):
    """Make the restored submission the agent account's own folder, in place at ~user/submission: the
    folder the instruction names, where run.sh is started and where the controller may write, as it can in
    a commissioning run. Ownership changes never follow a symlink (lchown, a walk that does not
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


def oracle_access(workdir):
    """True if the submission carries the reference solution's token (see the module docstring)."""
    p = os.path.join(workdir, "oracle_token")
    try:
        if os.path.islink(p) or not os.path.isfile(p) or os.path.getsize(p) > 256:
            return False
        with open(p, "rb") as fh:
            tok = fh.read().strip()
        with open(TOKEN_HASH) as fh:
            want = fh.read().strip()
    except OSError:
        return False
    return hashlib.sha256(tok).hexdigest() == want


SCRUB_ROOTS = ("/tmp", "/var/tmp", "/dev/shm", "/usr/local", "/run", "/var/lock")


def _agent_entries(uid, home):
    """Paths the agent account owns under the writable roots and its home (never following a symlink),
    except the submission folder and what is inside it."""
    sub = os.path.join(home, "submission")
    found = set()
    for root in list(SCRUB_ROOTS) + [home]:
        for dp, dns, fns in os.walk(root, topdown=True, followlinks=False):
            dns[:] = [d for d in dns if os.path.join(dp, d) != sub]
            for name in dns + fns:
                path = os.path.join(dp, name)
                try:
                    if os.lstat(path).st_uid == uid:
                        found.add(path)
                except OSError:
                    continue
    return found


def scrub_agent_files(uid, home, baseline):
    """Remove every file or directory the agent account owns there that was not there before grading
    started (`baseline`): what the image ships is never touched."""
    for path in sorted(_agent_entries(uid, home) - baseline, key=len):
        try:
            if os.path.isdir(path) and not os.path.islink(path):
                shutil.rmtree(path, ignore_errors=True)
            elif os.path.lexists(path):
                os.unlink(path)
        except OSError:
            pass


def fresh_submission(pristine, workdir, uid, gid):
    """Replace the submission folder with a fresh copy of the restored one, owned by the agent."""
    shutil.rmtree(workdir, ignore_errors=True)
    shutil.copytree(pristine, workdir, symlinks=True)
    for dp, dns, fns in os.walk(workdir, followlinks=False):
        for name in dns + fns:
            os.lchown(os.path.join(dp, name), uid, gid)
    os.lchown(workdir, uid, gid)
    os.chmod(workdir, 0o700)


def run_episodes(config, rng, uid, egress_mode, t_start, specs, plan, workdir, with_gt, pristine=None, gid=None,
                 baseline=None):
    budget = float(config.get("controller_budget_s", 600))
    charges = dict(config.get("image_charge_s", {"left": 0.45, "right": 0.45, "wrist": 0.35}))
    wall_cap = float(config.get("wall_cap_s", 12000))
    lock = ProbeLock(rng, rate=config.get("probe_rate", 0.01),
                     pass_fraction=config.get("probe_pass_fraction", 0.1), tolerance=1e-4)
    scores, aborted, faults = [], 0, 0
    per_nut = {}
    per_stage = {}
    progress = []
    nuts_done = []
    egress_hits = 0
    wall_cap_hit = False
    controller_s, budget_cut, images = [], 0, 0
    GuardedSolver.guard = egress_mode == "seccomp"
    GuardedSolver.nproc = int(config.get("solver_nproc_limit", 0))
    solver = None
    try:
        for ep, item in enumerate(plan):
            if solver is not None:
                try:
                    solver.close()
                except Exception:  # noqa: BLE001 - closing a broken solver must not crash the grader
                    pass
            if pristine is not None:
                stop_agent_processes(uid)
                scrub_agent_files(uid, os.path.dirname(workdir), baseline or set())
                fresh_submission(pristine, workdir, uid, gid)
            solver = None
            nut = item["id"]
            if time.monotonic() - t_start > wall_cap:
                wall_cap_hit = True
                scores.append(0.0)
                per_nut.setdefault(nut, []).append(0.0)
                log("episode {} ({}): not run, the grading wall-clock cap ({:.0f} s) was reached".format(ep, nut, wall_cap))
                continue
            env = KC.GradedNutEnv(specs[nut]["nuts"], early_end=True)
            score = 0.0
            prog = 0.0
            ndone = 0
            reason = "ok"
            spent = 0.0              # this kit's controller time: start-up, replies, image charges
            try:
                obs = env.reset(int(item["seed"]), specs[nut]["order"])
                started = time.monotonic()
                solver = GuardedSolver(step_timeout=config.get("step_timeout_s", 30), submission=workdir,
                                       max_restarts=0)
                solver.reset()
                spent += time.monotonic() - started
                for t in range(env.max_steps):
                    if time.monotonic() - t_start > wall_cap:
                        wall_cap_hit = True
                        raise SolverAborted("grading wall-clock cap reached")
                    if egress_mode == "detect" and agent_inet_sockets(uid):
                        egress_hits += 1
                    wire = E.to_wire(obs)
                    if with_gt:
                        wire["_gt"] = KC.ground_truth(env, wire)
                    other = None
                    if lock.due():
                        other = E.to_wire(env.perturbed_observation(obs, rng))
                        if with_gt:
                            other["_gt"] = wire["_gt"]
                    asked = time.monotonic()           # the controller's time starts once the request is ready
                    if other is not None:
                        action = lock.probe(solver, t, wire, perturbed=other)
                    else:
                        action = solver.act(t, wire)
                    spent += time.monotonic() - asked
                    charge, n_img = image_charge(action, charges)
                    spent += charge
                    if spent > budget:
                        budget_cut += 1
                        reason = "controller budget ({:.0f} s) used up at t={}".format(budget, t)
                        break
                    images += n_img
                    obs, done, info = env.step(action)
                    if done:
                        break
                if env.fault is not None:
                    faults += 1
                    reason = "sim fault: " + env.fault
                    score = 0.0
                else:
                    score = env.episode_score()
                    prog = env.progress()
                    ndone = env.nuts_assembled()
            except (SolverAborted, SolverMissing, E.ActionError, ValueError, TypeError, OverflowError, KeyError,
                    AttributeError, RecursionError) as exc:
                aborted += 1
                score = 0.0
                reason = "aborted: {}".format(str(exc)[:160])
            finally:
                env.close()
            if not (isinstance(score, float) and math.isfinite(score)) or score < 0:
                score = 0.0
            scores.append(score)
            per_nut.setdefault(nut, []).append(score)
            progress.append(prog)
            nuts_done.append(ndone)
            controller_s.append(spent)
            log("episode {} ({}): kit_complete={:.0f} nuts_assembled={} progress={:.3f} steps={} controller_s={:.1f} {}".format(
                ep, nut, score, ndone, prog, getattr(env, "t", 0), spent, reason))
    finally:
        try:
            if solver is not None:
                solver.close()
        except Exception:  # noqa: BLE001 - closing a broken solver must not crash the grader
            pass
    per_stage["nuts_assembled_fraction"] = [n / float(E.NUTS) for n in nuts_done]
    timing = {"controller_s_max": max(controller_s) if controller_s else 0.0,
              "controller_s_mean": sum(controller_s) / len(controller_s) if controller_s else 0.0,
              "controller_budget_cuts": float(budget_cut), "images_requested": float(images)}
    return scores, aborted, faults, per_nut, per_stage, egress_hits, wall_cap_hit, lock, progress, timing


def main():
    t_start = time.monotonic()
    config = stage_config()
    rng = seed_rng()
    anchor, meta = read_anchor()
    try:
        uid, gid = _agent_ids()
    except KeyError:
        write_verdict(zero_verdict(reason="no agent account", anchored=anchor is not None))
        return
    # The kit's clearing of the agent's account (once per verifier sandbox, a few seconds: every supervisord program
    # off, every process of the account killed until none comes back) is done here, before any kit's clock exists.
    # Solver.__init__ asks for it again inside the first kit's timed start and gets this report back at once, so
    # none of its seconds is counted against that kit's controller budget.
    clear_agent_account()
    if not stop_agent_processes(uid):
        write_verdict(zero_verdict(reason="agent processes could not be stopped (cleanup_failed)",
                                   anchored=anchor is not None))
        return
    with open(EPISODES) as fh:
        plan = json.load(fh)["episodes"]
    specs = {}
    for item in plan:
        kit = item["kit"]
        specs[item["id"]] = {"nuts": [E.check_nut_spec(sp) for sp in kit["nuts"]],
                             "order": [int(b) for b in kit["order"]]}
    try:
        workdir = stage_submission(uid, gid)
    except OSError as exc:
        log("staging the submission failed: {}".format(type(exc).__name__))
        workdir = None
    if workdir is None or not os.path.isfile(os.path.join(workdir, "run.sh")):
        write_verdict(zero_verdict(reason="no usable submission", anchored=anchor is not None))
        return
    pristine = os.path.join(tempfile.mkdtemp(prefix="keyed-pristine-"), "submission")   # root-only (umask 077)
    try:
        shutil.copytree(workdir, pristine, symlinks=True)
        os.chmod(pristine, 0o700)
    except OSError as exc:
        log("copying the submission failed: {}".format(type(exc).__name__))
        write_verdict(zero_verdict(reason="no usable submission", anchored=anchor is not None))
        return
    baseline = _agent_entries(uid, os.path.dirname(workdir))      # before any controller has run
    with_gt = oracle_access(workdir)
    if with_gt:
        log("reference-solution token present: observations carry _gt")
    egress_mode = "seccomp" if egress_self_test(uid, gid, int(config.get("solver_nproc_limit", 0))) else "detect"
    log("egress guard: " + egress_mode)
    try:
        (scores, aborted, faults, per_nut, per_stage, egress_hits, wall_cap_hit,
         lock, progress, timing) = run_episodes(config, rng, uid, egress_mode, t_start, specs, plan, workdir, with_gt,
                                                pristine=pristine, gid=gid, baseline=baseline)
    except (SolverMissing, SolverAborted) as exc:
        stop_agent_processes(uid)
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
                 wall_cap_hit=1.0 if wall_cap_hit else 0.0,
                 controller_s_max=round(timing["controller_s_max"], 3),
                 controller_s_mean=round(timing["controller_s_mean"], 3),
                 controller_budget_cuts=timing["controller_budget_cuts"],
                 images_requested=timing["images_requested"],
                 grading_wall_s=round(time.monotonic() - t_start, 1),
                 probe_lock_ok=1.0 if lock.ok else 0.0,
                 reference_token=1.0 if with_gt else 0.0,
                 mean_progress=sum(progress) / len(progress) if progress else 0.0)
    for key, vals in sorted(per_stage.items()):
        if vals:
            extra[key] = sum(vals) / len(vals)
    verdict = ratio_rewards(measured, anchor, cap=config.get("scoring_cap", 1.5), higher_is_better=True,
                            metric_name="kit_success_rate", locks_ok=True, extra=extra,
                            full_at=meta.get("full_at", 1.0))
    write_verdict(verdict)
    log("measured={:.5f} anchor={} ratio={:.3f} reward={:.3f} lock={} aborted={} faults={} egress={}".format(
        measured, anchor, verdict["metrics"]["ratio"], verdict["rewards"]["reward"], lock.ok, aborted, faults,
        egress_mode))


if __name__ == "__main__":
    main()
