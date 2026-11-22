"""Grader of earthquake_search_and_rescue (an SR-2 rescue crawler searching a collapsed building, sensing only): runs
as ROOT in the verify stage. Verification is separate (task.toml): this runs in a fresh verifier sandbox from the task
image, with only the agent's submission restored into it; tests/test.sh built the root-owned copy of the givens
(/opt/arb) before this starts.

The agent left /home/user/submission/rescue.py (solve(robot): an autonomous search-and-rescue program for the crawler:
it searches the rubble with the robot's own sensors, reports the survivor's position, places the survival kit within
their reach and brings the robot back to the deployment point). This grader:
  0. FIRST, stdlib only (`_early_lockdown`; tests/test.sh closed the stage folder already): records the upload time,
     freezes and kills every process of the agent account and keeps it stopped until the verdict is written
     (cellkit.verify_freeze), locks the stage, snapshots the submission root-owned, gates late writes, quarantines late
     entries elsewhere;
  1. checks the root-owned copy of the givens at /opt/arb (tests/test.sh builds it from setup_payload/) against
     given_hashes.json (a mismatch is a task error);
  2. gates: contract_ok = the submission checks the agent has (the givens' training/submission_checks.py, loaded from
     the root-owned copy: rescue.py is a regular file that parses and defines solve, the route gate (no weight files, no
     training calls, size cap), the import audit (static: no networking import, no computed import; runtime: every
     process of the program runs under the no-network seccomp filter (cellkit.nonet), the runner account's remote inet
     sockets are sampled), the module imports under the run regime (120 s, 120 CPU-s), >= 1 step call, no late write,
     no surviving process; before the first episode it waits until the frozen desktop is quiet and proves the
     no-network filter works (a failure is a task error);
  3. the oracle token: submission/.oracle_token whose sha256 equals hidden/oracle_token.sha256 switches the cells to the
     privileged mode (get_state_privileged, reference_run); a wrong token is ignored (stat oracle_token_mismatch);
  3b. checks that this machine computes a program's arithmetic as the reference machine does (numerics_check.py under
     the run regime's numerics pin against numerics.json; a difference is a task error);
  4. takes the `episodes` hidden site seeds pinned in hidden/seeds.json (a fixed hidden set of 8 selected by the rule of
     hidden/selection.json; a missing or short file is a task error) and runs the agent's program per site on the cell
     (rubblecell.server.RubbleCellServer, mode hidden: the building, its collapse, the obstacles, the survivor and the
     decoys drawn from the seed; sensor-only API) as `nobody` from a read-only copy under an opaque run tag, each
     episode in a process of its own (the colour renderer keeps state for a process's life); an episode
     in which the cell failed (an internal error, a physics fault, a frame it could not render) is run once more from
     scratch and the second run counts;
  5. criterion: rescue_score = the mean over the sites of the mission's score (rubblecell.server.score: located 0.45 --
     the last report that reached the team over the radio link, within 1 m of the survivor (linear to 0 at 3 m), after
     a camera or thermal frame of the program's showed them --, kit 0.25 -- the survival kit at rest within 0.8 m of the
     survivor's head or a hand, not on them --, returned 0.30 -- the robot upright within 3 m of its deployment point at
     the end, only with located > 0; a secondary collapse scores the site 0; a protective stop on the survivor
     ends the mission, which keeps what the team had, without the return and at most 0.45), times the gates. A failure on the grader's side of an episode is a TASK ERROR (`fail`); a program that crashes, hangs or
     misbehaves ends its own episode, which is scored as it stands. Each episode caps the program's own time
     (`program_time_cap_s`): the wall clock minus the cell's simulation and sensor rendering (cellkit.wire).
Untouched sandbox: no rescue.py -> contract_ok 0 -> 0. Oracle: the token + reference_run (the privileged reference
program inside the cell, with the site's global map) -> the recorded reference value -> 1.
Records: out/episodes.json, out/traj_<tag>.npz (the episode's state samples for the replay renderer, tests/review) and
logs/<tag>.log (the program's stderr) under the root-only /opt/arb-verify (outside /opt/ale, which the engine clears
before it keeps a sandbox).
"""

import ast
import json
import os
import pwd
import shutil
import stat as statmod
import subprocess
import sys
import time
import traceback

sys.dont_write_bytecode = True

STAGE_DIR = os.environ.get("ALE_STAGE_DIR", os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, STAGE_DIR)

HOME = os.environ.get("ALE_HOME", "/home/user")
AGENT_USER = os.path.basename(HOME.rstrip("/")) or "user"
SUBMISSION = os.path.join(HOME, "submission")
ROOT_PAYLOAD = os.environ.get("ARB_ROOT_PAYLOAD", "/opt/arb")
ROOT_GIVEN = os.path.join(ROOT_PAYLOAD, "given")
RUNS_ROOT = os.path.join(ROOT_PAYLOAD, "runs")
SNAPSHOT = os.path.join(RUNS_ROOT, "_submission_snapshot")
QUARANTINE = os.path.join(RUNS_ROOT, "_quarantine")
STICKY_DIRS = [d for d in ("/tmp", "/dev/shm", "/var/tmp", "/run/lock") if os.path.isdir(d)]
WATCH_ROOTS = STICKY_DIRS + [HOME]
LATE_GATE_MAX_GAP_S = 600.0

from cellkit import lockdown, runner, verify_freeze  # noqa: E402  (stdlib only)

FREEZER = [None]        # the agent account's verify_freeze.AgentFreezer: started by _early_lockdown, stopped after the verdict


def log(msg):
    print(msg, file=sys.stderr, flush=True)


def _early_lockdown():
    out = {"agent_processes_killed": 0.0, "agent_processes_left": 0.0, "stage_locked": 0.0, "stage_parent_locked": 0.0,
           "stage_lock_latency_ms": -1.0, "submission_late_writes": 0.0, "late_writes_quarantined": 0.0, "late_write_gate_active": 0.0,
           "submission_snapshot": 0.0, "saved": None, "parent": None, "t_ref": None, "late": [], "reasons": []}
    if os.getuid() != 0:
        return out
    t_ref = lockdown.stage_upload_time(STAGE_DIR)
    out["t_ref"] = t_ref
    try:
        uid = pwd.getpwnam(AGENT_USER).pw_uid
    except KeyError:
        uid = None
    if uid is not None and uid != 0:
        killed, left = lockdown.stop_and_kill_uid(uid, grace_s=8.0)
        out["agent_processes_killed"], out["agent_processes_left"] = float(killed), float(left)
        # v0.5.1: a survivor is a statistic, not a contract failure. Verification is separate (task.toml): nothing of the
        # agent's runs in this sandbox, and the image's supervisord respawns the account's desktop as fast as it is
        # killed, so a process counted here is the image's own (2026-11-08: one respawned desktop process zeroed a
        # submission before any episode ran). The freezer below keeps every process of the account stopped until the
        # verdict is written, whatever the count.
        if left:
            log("[grader] %d process(es) of the agent account (the image's respawned desktop) outlived the kill at verify start: "
                "frozen until the verdict" % left)
    # the image's init respawns the agent account's desktop within seconds of the kill, and the respawned session runs
    # the account's autostart entries: every process of the account stays SIGSTOPped (polled every 0.1 s) until the
    # verdict is written
    FREEZER[0] = verify_freeze.AgentFreezer.for_user(AGENT_USER).start()
    out["saved"] = runner.lock_stage(STAGE_DIR)
    out["stage_locked"] = 1.0 if out["saved"] is not None else 0.0
    parent = os.path.dirname(os.path.abspath(STAGE_DIR))
    if parent and parent not in ("/", os.path.abspath(HOME)) and os.path.isdir(parent):
        try:
            mode = statmod.S_IMODE(os.lstat(parent).st_mode)
            os.chmod(parent, 0o700)
            saved_mode = os.environ.get("ARB_STAGE_PARENT_MODE")
            out["parent"] = (parent, int(saved_mode, 8) if saved_mode else mode)
            out["stage_parent_locked"] = 1.0
        except (OSError, ValueError):
            pass
    gate_active = t_ref is not None
    cfg_json = os.path.join(STAGE_DIR, "config.json")
    if gate_active and os.path.isfile(cfg_json):
        try:
            gate_active = (os.lstat(cfg_json).st_ctime - t_ref) <= LATE_GATE_MAX_GAP_S
        except OSError:
            pass
    out["late_write_gate_active"] = 1.0 if gate_active else 0.0
    if os.path.isdir(SUBMISSION):
        if gate_active:
            out["late"] = lockdown.late_entries(SUBMISSION, t_ref)
            out["submission_late_writes"] = float(len(out["late"]))
            if out["late"]:
                out["reasons"].append("%d entr%s of the submission written after the agent phase ended (%s)" % (
                    len(out["late"]), "y" if len(out["late"]) == 1 else "ies", ", ".join(e[0] for e in out["late"][:5])))
        os.makedirs(RUNS_ROOT, exist_ok=True)
        os.chmod(RUNS_ROOT, 0o755)
        lockdown.snapshot_dir(SUBMISSION, SNAPSHOT)
        out["submission_snapshot"] = 1.0
    if gate_active:
        q = 0
        for root in WATCH_ROOTS:
            late = lockdown.late_entries(root, t_ref, exclude=(SUBMISSION, RUNS_ROOT, STAGE_DIR))
            if late:
                log("[grader] %d late entr%s under %s quarantined" % (len(late), "y" if len(late) == 1 else "ies", root))
                q += lockdown.quarantine(root, late, QUARANTINE)
        out["late_writes_quarantined"] = float(q)
    out["stage_lock_latency_ms"] = lockdown.process_age_ms()
    return out


EARLY = _early_lockdown() if __name__ == "__main__" else {"saved": None, "parent": None, "reasons": [], "late": [], "t_ref": None}

import hashlib  # noqa: E402

import numpy as np  # noqa: E402

from cellkit import netaudit, nonet, wire  # noqa: E402
from cellkit.seeds import load_pinned  # noqa: E402
from cellkit.seeds import seed_rng  # noqa: E402
from rubblecell.client import launcher_argv  # noqa: E402
from rubblecell.server import RubbleCellServer  # noqa: E402

HIDDEN = os.path.join(STAGE_DIR, "hidden")
# The maintainers' record (the program's stderr, episodes.json) lives outside /opt/ale: the engine deletes its stage
# folders before it keeps a sandbox, and the record is read from a kept one. Root-only (0700). A host-side run as
# another account keeps it in the stage folder, as before.
RECORD_ROOT = os.environ.get("ARB_RECORD_ROOT") or ("/opt/arb-verify" if os.getuid() == 0 else STAGE_DIR)
LOGS = os.path.join(RECORD_ROOT, "logs")
OUT = os.path.join(RECORD_ROOT, "out")
PROGRAM = "rescue"
CRITERIA = ["rescue_score"]
GATES = ["contract_ok"]
SUSPICIOUS = ("/opt/ale", "ALE_STAGE_DIR", "ALE_VERIFICATION", "hidden/seeds", "oracle_token", "reference_run", "get_state_privileged",
              "ALE_VERIFY", "/opt/ale/tests", ".ale-solution")


def record_dir(path):
    """`path` under the record root, created on demand; never a scoring input."""
    os.makedirs(RECORD_ROOT, exist_ok=True)
    if RECORD_ROOT != STAGE_DIR:
        os.chmod(RECORD_ROOT, 0o700)
    os.makedirs(path, exist_ok=True)
    return path


def program_source():
    return SNAPSHOT if os.getuid() == 0 and os.path.isdir(SNAPSHOT) else SUBMISSION


def fail(reason):
    log("[grader] TASK ERROR: " + reason)
    raise SystemExit(1)


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_json(p):
    with open(p) as f:
        return json.load(f)


def root_payload_ok(expected):
    problems = []
    owner = 0 if os.getuid() == 0 else os.getuid()
    if not os.path.isdir(ROOT_PAYLOAD):
        return False, ["%s missing (tests/test.sh builds it)" % ROOT_PAYLOAD]
    for rel, digest in expected.items():
        p = os.path.join(ROOT_PAYLOAD, rel)
        try:
            st = os.lstat(p)
        except OSError:
            problems.append("missing " + rel)
            continue
        if os.path.islink(p) or st.st_uid != owner or (st.st_mode & 0o022):
            problems.append("not a root-owned, group/world-unwritable regular file: " + rel)
            continue
        if sha256_file(p) != digest:
            problems.append("differs " + rel)
    return not problems, problems


def load_checks():
    """The submission checks: the module the agent has (~/training/submission_checks.py), loaded from the grader's
    root-owned copy of the givens after its hash was verified (root_payload_ok). One source for the grader and for the
    training launcher: what the launcher refuses, the grader refuses, and nothing else."""
    import importlib.util
    path = os.path.join(ROOT_PAYLOAD, "training", "submission_checks.py")
    spec = importlib.util.spec_from_file_location("submission_checks", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def suspicious_lines(src):
    """A statistic, never a gate: lines of the submission that name the grader's side."""
    n = 0
    for dp, _, fns in os.walk(src):
        for fn in fns:
            p = os.path.join(dp, fn)
            if fn.endswith((".py", ".sh")) and not os.path.islink(p):
                try:
                    text = open(p, encoding="utf-8", errors="replace").read()
                except OSError:
                    continue
                n += sum(1 for line in text.splitlines() if any(x in line for x in SUSPICIOUS))
    return n


class Runner(object):
    def __init__(self, cfg):
        self.acct = runner.runner_account(cfg.get("runner_account", "nobody"), AGENT_USER)
        self.name, self.uid, self.gid = self.acct if self.acct else (None, None, None)
        self.lib_dirs = {"rubblecell": os.path.join(ROOT_GIVEN, "lib", "rubblecell")}
        self.cpu_s = float(cfg.get("cpu_cap_factor", 4.0)) * float(cfg["program_time_cap_s"])
        self.killed = self.left = self.unclean_runs = self.stray_files = self.symlinks_skipped = 0
        os.makedirs(RUNS_ROOT, exist_ok=True)
        os.chmod(RUNS_ROOT, 0o755)

    def prepare(self, tag, program_dir, private=False):
        root = os.path.join(RUNS_ROOT, tag)
        if os.path.isdir(root):
            shutil.rmtree(root, ignore_errors=True)
        owner = (self.uid, self.gid) if (self.acct and not private) else None
        dirs = runner.prepare_run_dir(root, program_dir, self.lib_dirs, owner=owner, private=private)
        self.symlinks_skipped = max(self.symlinks_skipped, dirs["skipped"])
        return dirs

    def env(self, dirs, extra=None):
        # the program's environment is the one the agent's launcher builds (submission_checks.run_environment: nothing
        # of the caller's but PATH, and the numerics pin): one source
        e = load_checks().run_environment(dirs["home"], dirs["tmp"], dirs["lib"], ROOT_GIVEN)
        if extra:
            e.update(extra)
        return e

    def snapshot(self, exclude):
        return {r: runner.snapshot_tree(r, exclude) for r in WATCH_ROOTS}

    def kill_runner_processes(self, grace_s=8.0):
        if not self.acct:
            return 0
        killed, left = lockdown.stop_and_kill_uid(self.uid, grace_s=grace_s)
        self.killed += killed
        self.left += left
        if left:
            self.unclean_runs += 1
        return left

    def cleanup(self, dirs, before):
        self.kill_runner_processes()
        if before is not None and self.acct:
            for r in WATCH_ROOTS:
                after = runner.snapshot_tree(r, (dirs["root"],))
                new, modified = runner.tree_changes(before[r], after, self.uid)
                self.stray_files += len(new) + len(modified)
                runner.sweep_new_entries(r, before[r], after, self.uid, (dirs["root"],))
        shutil.rmtree(dirs["root"], ignore_errors=True)


def contract_check(cfg, rn):
    src = program_source()
    checks = load_checks()
    suspicious = suspicious_lines(src) if os.path.isdir(src) else 0
    ok, why, net = checks.check_submission(src, os.path.join(ROOT_GIVEN, "lib", "rubblecell"))
    if not ok:
        return False, why, suspicious, net
    dirs = rn.prepare("import_check", src)
    before = rn.snapshot((dirs["root"],)) if rn.acct else None
    argv = runner.wrap_limits(checks.import_check_argv(sys.executable, dirs["lib"], dirs["program"]), cpu_s=checks.IMPORT_CPU_S,
                              user=rn.name, nonet=True)
    try:
        r = subprocess.run(argv, cwd=dirs["program"], env=rn.env(dirs), capture_output=True, text=True, timeout=checks.IMPORT_WALL_S,
                           start_new_session=True)
        ok, why = r.returncode == 0, ("ok" if r.returncode == 0 else "importing %s.py failed: " % PROGRAM + (r.stderr.strip().splitlines() or ["?"])[-1][:300])
    except subprocess.TimeoutExpired:
        ok, why = False, "importing %s.py took more than %d s" % (PROGRAM, checks.IMPORT_WALL_S)
    finally:
        rn.cleanup(dirs, before)
    return ok, why, suspicious, []


def numerics_check(rn):
    """(ok, detail): the program's arithmetic on this machine against tests/numerics.json. numerics_check.py --math runs
    as a mission program does (the runner account, the run regime's environment and no-network filter) from a root-owned
    copy outside the locked stage folder."""
    ref = load_json(os.path.join(STAGE_DIR, "numerics.json")).get("math", {})
    src = os.path.join(RUNS_ROOT, "_numerics")
    shutil.rmtree(src, ignore_errors=True)
    os.makedirs(src)
    shutil.copyfile(os.path.join(STAGE_DIR, "numerics_check.py"), os.path.join(src, "numerics_check.py"))
    dirs = rn.prepare("numerics", src)
    argv = runner.wrap_limits([sys.executable] + runner.PYTHON_FLAGS + [os.path.join(dirs["program"], "numerics_check.py"), "--math"],
                              cpu_s=120, user=rn.name, nonet=True)
    r = None
    try:
        r = subprocess.run(argv, cwd=dirs["program"], env=rn.env(dirs), capture_output=True, text=True, timeout=180,
                           start_new_session=True)
        got = json.loads((r.stdout or "").strip().splitlines()[-1]) if r.returncode == 0 else None
    except (subprocess.TimeoutExpired, ValueError, IndexError):
        got = None
    finally:
        rn.cleanup(dirs, None)
        shutil.rmtree(src, ignore_errors=True)
    if got is None:
        return False, "the numerics check did not run: %s" % (((r.stderr or "").strip().splitlines() or ["?"])[-1][:300] if r is not None else "timeout")
    differs = sorted(k for k in set(ref) | set(got) if ref.get(k) != got.get(k))
    return not differs, ("ok" if not differs else "parts that differ: " + ", ".join(differs))


RUNNER_COUNTERS = ("killed", "left", "unclean_runs", "stray_files", "symlinks_skipped")


def _run_episode_forked(cfg, seed, rn, privileged, tag):
    """run_episode in a child process of its own. Why: the cell's colour renderer (Filament) keeps state for the life
    of a process (its frame counter drives the dithering noise, its objects are never all freed), so a site's frames
    would depend on the sites graded before it in the same process. The training ground forks a process per mission;
    with a process per graded episode the same site and the same requests give the same frames in training and in
    grading. The child returns the episode's summary and the runner's counters through a pipe; a task error in the
    child (exit code 1) is a task error here."""
    rfd, wfd = os.pipe()
    sys.stdout.flush()
    sys.stderr.flush()
    pid = os.fork()
    if pid == 0:
        code = 0
        try:
            os.close(rfd)
            summary = run_episode(cfg, seed, rn, privileged, tag)
            payload = json.dumps({"summary": summary, "rn": {k: getattr(rn, k) for k in RUNNER_COUNTERS}}, default=float).encode()
            with os.fdopen(wfd, "wb") as f:
                f.write(payload)
        except SystemExit as e:
            code = int(e.code) if isinstance(e.code, int) else 1
        except BaseException:  # noqa: BLE001
            log("[grader] %s: the episode's process failed\n%s" % (tag, traceback.format_exc()))
            code = 1
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(code)
    os.close(wfd)
    with os.fdopen(rfd, "rb") as f:
        data = f.read()
    _, status = os.waitpid(pid, 0)
    if not os.WIFEXITED(status) or os.WEXITSTATUS(status) != 0 or not data:
        fail("%s: the episode's process ended abnormally (status %d)" % (tag, status))
    out = json.loads(data.decode())
    for k in RUNNER_COUNTERS:
        setattr(rn, k, out["rn"][k])
    return out["summary"]


def token_state():
    """(privileged, mismatch): the submission's .oracle_token against hidden/oracle_token.sha256."""
    p = os.path.join(program_source(), ".oracle_token")
    want_p = os.path.join(HIDDEN, "oracle_token.sha256")
    if not os.path.isfile(p) or os.path.islink(p) or not os.path.isfile(want_p):
        return False, False
    try:
        tok = open(p, "rb").read().strip()
        want = open(want_p).read().strip()
    except OSError:
        return False, False
    got = hashlib.sha256(tok).hexdigest()
    return (got == want), (got != want)


def _settle_desktop(max_s=60.0, quiet_s=8.0, poll_s=0.25):
    """Wait, before the first episode's network baseline, until the frozen desktop is quiet. The image's supervisord
    respawns the agent account's desktop (Xvnc, websockify, GNOME, ...) after the verify-start kill, and a respawned
    process can open a loopback TCP connection in the moment before the freezer stops it (measured 2026-11-06 on this
    image with this grader's own lockdown: one TCP active open 1.1 s after the lockdown returned, none in the next 2
    minutes; two validate runs counted it in their first episode). Quiet = the agent account's processes and the
    namespace's outbound counters unchanged for `quiet_s`. Returns (seconds waited, counter increments meanwhile)."""
    fr = FREEZER[0]
    if os.getuid() != 0 or fr is None or fr.uid is None:
        return 0.0, 0
    t0 = time.time()
    first = last = (netaudit.net_counters(), sorted(verify_freeze.uid_processes(fr.uid)))
    quiet_since = t0
    while time.time() - t0 < max_s:
        time.sleep(poll_s)
        now = (netaudit.net_counters(), sorted(verify_freeze.uid_processes(fr.uid)))
        if now != last:
            last, quiet_since = now, time.time()
        elif time.time() - quiet_since >= quiet_s:
            break
    seen = sum(max(0, int(last[0].get(k, 0)) - int(first[0].get(k, 0))) for k in first[0] if k != "unreadable")
    return time.time() - t0, seen


def run_episode(cfg, seed, rn, privileged, tag):
    """One graded episode: a fresh cell, the program on the wire as the runner account, the cell's record. A failure on
    the grader's side (the cell's build, the program's launch, the wire, the cleanup, the scoring) is a task error
    (`fail`), never the agent's zero; whatever the program does ends at most its own episode."""
    try:
        server = RubbleCellServer(config=cfg, mode="hidden", privileged=privileged, log=log,
                                  reference_dir=os.path.join(STAGE_DIR, "reference") if privileged else None)
        server.reset(seed, "hidden")
    except Exception:  # noqa: BLE001 - the cell depends on the seed only: a build failure is the task's
        fail("%s: the cell could not be built\n%s" % (tag, traceback.format_exc()))
    record_dir(LOGS)
    t0 = time.time()
    dirs = rn.prepare(tag, program_source())          # an opaque tag: the hidden seed never reaches a path the program sees
    before = rn.snapshot((dirs["root"],)) if rn.acct else None
    argv, user = launcher_argv(sys.executable, dirs["lib"], dirs["program"], PROGRAM), rn.name
    # the program's own time (the wall clock minus the cell's simulation and rendering, cellkit.wire); the privileged
    # reference controller runs inside one call (reference_run) and has its own cap
    program_cap = float(cfg["privileged_program_time_cap_s"] if privileged else cfg["program_time_cap_s"])
    proc = None
    sampler = netaudit.SocketSampler(uid=rn.uid if rn.acct else None, pids_fn=None if rn.acct else (lambda: []),
                                     period_s=float(cfg.get("socket_sample_s", 0.5)))
    counters = netaudit.NetCounters()
    reason = None
    with counters, sampler:
        try:
            try:
                proc = wire.AgentProcess(argv, cwd=dirs["program"], user=user, env=rn.env(dirs, None), cpu_s=rn.cpu_s,
                                         stderr_path=os.path.join(LOGS, "%s.log" % tag), nonet=True)
            except Exception:  # noqa: BLE001
                fail("%s: the program could not be launched\n%s" % (tag, traceback.format_exc()))
            try:
                reason = wire.serve_episode(server, proc, program_cap, float(cfg["silence_s"]), log=log)
            except Exception:  # noqa: BLE001 - the wire turns everything the program does into an end reason
                fail("%s: the wire failed\n%s" % (tag, traceback.format_exc()))
        finally:
            if proc is not None:
                proc.terminate()
            rn.cleanup(dirs, before)             # every process of the runner account is gone before the counters are read
    try:
        if not server.done:
            server._end(reason)
        summary = server.finalize()
    except Exception:  # noqa: BLE001 - finalize() does not raise by design
        fail("%s: the episode could not be scored\n%s" % (tag, traceback.format_exc()))
    summary["episode_wall_s"] = time.time() - t0
    summary["end_reason"] = reason
    try:                                            # the state samples, for the maintainers' replay renderer
        record_dir(OUT)
        np.savez_compressed(os.path.join(OUT, "traj_%s.npz" % tag), traj=server.trajectory(), seed=seed, kind="hidden")
    except Exception:  # noqa: BLE001 - a record is never a scoring input
        log("[grader] %s: trajectory record not written" % tag)
    try:
        server.sensors.close()
    except Exception:  # noqa: BLE001
        pass
    summary["remote_sockets_seen"] = int(sampler.max_seen)
    summary["socket_samples"] = int(sampler.samples)
    summary["net_outbound"] = int(counters.total)
    summary["net_delta"] = dict(counters.delta)
    return summary


def _cell_failed(s):
    """What went wrong on the cell's side of an episode (not the program's doing): [] when nothing."""
    out = []
    if int(s.get("internal_errors", 0)) > 0:
        out.append("internal_error")
    if s.get("done_reason") == "physics_fault" or s.get("end_reason") == "physics_fault":
        out.append("physics_fault")
    if int(s.get("frames_lost", 0)) > 0:
        out.append("render_failure")
    return out


def _episode_with_retry(cfg, seed, rn, privileged, tag, metrics):
    """One episode; when the cell failed in it it is run once more from scratch (a fresh cell, a fresh program, the
    same run folder) and the second run counts. A request cannot make the cell fail: the cell refuses every non-finite
    or malformed parameter as `bad_params` (rubblecell.server._dispatch), so a program has no way to ask for a second
    run. A render failure that repeats is the grader's (a task error); an internal error or a physics fault that
    repeats is scored as it stands."""
    s_a = _run_episode_forked(cfg, seed, rn, privileged, tag)
    why = _cell_failed(s_a)
    if not why:
        return s_a
    log("[grader]   %s: the cell failed (%s): the episode is run once more" % (tag, ", ".join(why)))
    metrics["episode_retries"] = metrics.get("episode_retries", 0.0) + 1.0
    s_b = _run_episode_forked(cfg, seed, rn, privileged, tag)      # the same run folder: the program cannot tell the runs apart
    why_b = _cell_failed(s_b)
    if "render_failure" in why_b:
        fail("%s: the cell could not render frames in two runs of the episode" % tag)
    s_b["retried_after"] = why
    return s_b


def write_verdict(rewards, metrics):
    """One gated `reward` in the benchmark kit's envelope (robotics_grader.ratio_rewards against the recorded reference
    value); the gate, the criterion and every statistic go to `metrics`. In the engine (`ALE_VERIFY_PYTHON` set) the kit
    writes reward.json and verification.json, the criteria text as the `diagnostic`; the metrics keep the names and the
    filter of this grader's records on the earlier engine API (finite numbers only, '.' and '/' in a name replaced by
    '_'). Elsewhere the envelope goes to verdict.local.json in the stage folder."""
    from robotics_grader import ratio_rewards, read_anchor
    from robotics_grader import write_verdict as kit_write_verdict
    overall = float(rewards["rescue_score"])
    anchor, meta = read_anchor()
    extra = {k: float(v) for k, v in metrics.items()}
    extra.update({k: float(rewards[k]) for k in GATES + CRITERIA})
    verdict = ratio_rewards(overall, anchor, higher_is_better=meta.get("direction", "higher") == "higher",
                            metric_name=meta.get("metric", "rescue_score"), extra=extra, full_at=meta.get("full_at", 1.0))
    text = ("contract_ok %.0f (the program imports, exposes solve, passes the route gate and the network audit and steps the "
            "cell; no late write, no surviving process); rescue_score %.4f (the mean over the hidden sites of the mission's "
            "score: survivor located and reported 0.45, survival kit within reach 0.25, robot back at the deployment point 0.30; "
            "a secondary collapse scores a site 0; a protective stop on the survivor ends the mission without the return, at most "
            "0.45)") % (rewards["contract_ok"], rewards["rescue_score"])
    if os.environ.get("ALE_VERIFY_PYTHON"):
        recorded = {}
        for k, v in sorted(verdict["metrics"].items()):
            try:
                v = float(v)
            except (TypeError, ValueError):
                continue
            if v == v and abs(v) != float("inf"):
                recorded[k.replace(".", "_").replace("/", "_")] = v
        kit_write_verdict({"rewards": verdict["rewards"], "metrics": recorded}, diagnostic=text)
        return overall
    kit_write_verdict(verdict, path=os.path.join(STAGE_DIR, "verdict.local.json"))
    return overall


def _restore_modes():
    runner.unlock_stage(EARLY.get("saved"))
    mode = os.environ.get("ARB_STAGE_MODE")
    if mode:
        try:
            os.chmod(STAGE_DIR, int(mode, 8))
        except (OSError, ValueError):
            pass
    parent = EARLY.get("parent")
    if parent:
        try:
            os.chmod(parent[0], parent[1])
        except OSError:
            pass


def main():
    cfg = load_json(os.path.join(STAGE_DIR, "grader_config.json"))
    rewards = dict((k, 0.0) for k in GATES + CRITERIA)
    metrics = {"grader_exception": 0.0, "episodes": float(cfg.get("episodes", 3)),
               "root_payload_ok": 0.0, "agent_deprivileged": 0.0, "runner_uid": -1.0, "privileged_mode": 0.0, "oracle_token_mismatch": 0.0,
               "runner_processes_killed": 0.0, "runner_processes_left": 0.0, "runner_unclean_runs": 0.0,
               "stray_files_removed": 0.0, "suspicious_lines": 0.0, "remote_sockets_seen": 0.0, "net_outbound_events": 0.0,
               "network_imports": 0.0, "illegal_call_count": 0.0}
    for k in ("stage_locked", "stage_parent_locked", "stage_lock_latency_ms", "agent_processes_killed", "agent_processes_left",
              "submission_late_writes", "late_writes_quarantined", "late_write_gate_active", "submission_snapshot"):
        metrics[k] = float(EARLY.get(k, -1.0 if k == "stage_lock_latency_ms" else 0.0))
    try:
        try:
            _main_locked(cfg, rewards, metrics)
        except SystemExit:
            raise
        except Exception:  # noqa: BLE001 - a grader defect must still leave a verdict and say so
            log("[grader] EXCEPTION\n" + traceback.format_exc())
            metrics["grader_exception"] = 1.0
            raise
    finally:
        _restore_modes()
        if FREEZER[0] is not None:
            FREEZER[0].stop()                       # after the verdict (written by _main_locked); the frozen stay stopped


def _main_locked(cfg, rewards, metrics):
    t_start = time.time()
    expected = load_json(os.path.join(STAGE_DIR, "given_hashes.json"))
    ok_root, problems = root_payload_ok(expected)
    metrics["root_payload_ok"] = 1.0 if ok_root else 0.0
    if not ok_root:
        fail("the root-owned copy of the givens at %s is unusable: %s" % (ROOT_PAYLOAD, "; ".join(problems[:5])))
    rn = Runner(cfg)
    if FREEZER[0] is not None and rn.acct and rn.uid == FREEZER[0].uid:
        FREEZER[0].stop()        # no `nobody` in the image: the runner fell back to the agent account, whose programs must run
        log("[grader] runner account = the agent account: the verify-stage freezer is stopped")
    metrics["runner_uid"] = float(rn.uid) if rn.acct else -1.0
    metrics["agent_deprivileged"] = 1.0 if rn.acct else 0.0
    privileged, mismatch = token_state()
    metrics["privileged_mode"] = 1.0 if privileged else 0.0
    metrics["oracle_token_mismatch"] = 1.0 if mismatch else 0.0
    ok, why, suspicious, net = contract_check(cfg, rn)
    if ok and EARLY.get("reasons"):
        ok, why = False, "; ".join(EARLY["reasons"])
    metrics["suspicious_lines"] = float(suspicious)
    metrics["network_imports"] = float(len(net))
    log("[grader] contract_ok=%s (%s); privileged %s; runner %s; lock latency %.0f ms" % (ok, why, privileged, rn.name or "self", metrics["stage_lock_latency_ms"]))
    n = int(cfg.get("episodes", 4))
    pinned_path = os.path.join(HIDDEN, "seeds.json")
    try:
        pinned = load_pinned(pinned_path)
    except (OSError, ValueError) as e:
        fail("hidden/seeds.json is unreadable: %s" % e)
    if not pinned or len(pinned.get("seeds", [])) < n:
        fail("hidden/seeds.json must pin the %d hidden episodes (the fixed hidden set)" % n)
    ep_seeds = [int(x) for x in pinned["seeds"][:n]]
    stream = seed_rng(pinned=ep_seeds)
    metrics.update(stream.stats())
    if not ok:
        rewards["contract_ok"] = 0.0
        _finish_stats(rn, metrics, t_start)
        write_verdict(rewards, metrics)
        return
    settle_s, settle_seen = _settle_desktop()
    metrics["desktop_settle_s"] = round(settle_s, 1)
    metrics["net_settle_events"] = float(settle_seen)
    if settle_s > 0.0:
        log("[grader] the frozen desktop settled in %.1f s (%d outbound counter increments meanwhile, before any episode)" % (
            settle_s, settle_seen))
    nn_ok, nn_why = nonet.selftest(wrap=lambda a: runner.wrap_limits(a, cpu_s=60, user=rn.name),
                                   env=runner.child_env("/tmp", "/tmp", "/nonexistent", ROOT_GIVEN))
    metrics["nonet_selftest_ok"] = 1.0 if nn_ok else 0.0
    if not nn_ok:
        fail("the no-network filter for the program does not work here: " + nn_why)
    num_ok, num_why = numerics_check(rn)
    metrics["numerics_ok"] = 1.0 if num_ok else 0.0
    if not num_ok:
        fail("this machine does not compute a mission program's arithmetic as the reference machine does (%s)" % num_why)
    agent = []
    step_calls = illegal = 0
    metrics["episode_retries"] = 0.0
    for i, seed in enumerate(ep_seeds):
        log("[grader] episode %d/%d" % (i + 1, n))
        s_a = _episode_with_retry(cfg, seed, rn, privileged, "ep%02d" % (i + 1), metrics)
        log("[grader]   score %.3f parts %s voids %s stops %d lost %s sim %.1f s wall %.1f s (program %.1f s, cell %.1f s) frames %d lidar %d end %s" % (
            s_a["episode_score"], s_a["parts"], [v["what"] for v in s_a["voids"]][:3], len(s_a.get("stops", [])), s_a.get("lost"),
            s_a["sim_time_s"], s_a["episode_wall_s"], s_a["program_time_s"], s_a["cell_work_s"], s_a["camera_frames_total"],
            s_a["lidar_sweeps"], s_a["end_reason"]))
        agent.append(s_a)
        step_calls += s_a["step_calls"] + s_a["calls"].get("reference_run", 0)
        illegal += s_a["illegal_call_count"]
        metrics["remote_sockets_seen"] = max(metrics["remote_sockets_seen"], float(s_a["remote_sockets_seen"]))
        metrics["net_outbound_events"] += float(s_a.get("net_outbound", 0))
        for ck, cv in (s_a.get("net_delta") or {}).items():
            metrics["net_" + ck] = metrics.get("net_" + ck, 0.0) + float(max(0, cv))
    vals = [float(s["episode_score"]) for s in agent]          # a void already scored its episode 0 (episode.py)
    steps_gate = 1.0 if step_calls > 0 else 0.0
    clean_gate = 1.0 if rn.unclean_runs == 0 else 0.0
    # v0.5: the gate counts only what belongs to the runner account (its remote inet sockets; under the seccomp filter
    # it cannot open one); the namespace's counters (loopback and every account included) stay statistics
    net_gate = 1.0 if metrics["remote_sockets_seen"] <= 0 else 0.0
    gate = steps_gate * clean_gate * net_gate
    rewards["contract_ok"] = gate
    if not net_gate:
        log("[grader] contract_ok 0: the program held a remote inet socket during an episode (%d seen)" % metrics["remote_sockets_seen"])
    elif metrics["net_outbound_events"] > 0:
        log("[grader] note: %d namespace-wide outbound counter increments during the episodes (statistic only: loopback and "
            "other accounts count too; the program runs under the no-network filter)" % metrics["net_outbound_events"])
    rewards["rescue_score"] = float(max(0.0, min(1.0, gate * float(np.mean(vals)))))
    metrics["raw_score_mean"] = float(np.mean([s["raw_score"] for s in agent]))
    metrics["void_rate"] = float(np.mean([1.0 if s["voided"] else 0.0 for s in agent]))
    metrics["robot_lost_rate"] = float(np.mean([1.0 if s.get("lost") else 0.0 for s in agent]))
    for part in ("located", "kit", "returned"):
        metrics["part_mean." + part] = float(np.mean([s["parts"].get(part, 0.0) for s in agent]))
    errs = [s["report_error_m"] for s in agent if s.get("report_error_m") is not None]
    if errs:
        metrics["report_error_m_mean"] = float(np.mean(errs))
    metrics["seen_rate"] = float(np.mean([1.0 if s.get("seen_frames", 0) > 0 else 0.0 for s in agent]))
    metrics["secondary_collapses"] = float(sum(len(s.get("collapses", [])) for s in agent))
    metrics["protective_stops"] = float(sum(len(s.get("stops", [])) for s in agent))
    metrics["max_survivor_force_N"] = float(max(s.get("max_survivor_force_N", 0.0) for s in agent))
    metrics["mean_sim_time_s"] = float(np.mean([s["sim_time_s"] for s in agent]))
    metrics["mean_episode_wall_s"] = float(np.mean([s["episode_wall_s"] for s in agent]))
    metrics["illegal_call_count"] = float(illegal)
    metrics["step_calls"] = float(step_calls)
    metrics["camera_frames_mean"] = float(np.mean([s["camera_frames_total"] for s in agent]))
    metrics["lidar_sweeps_mean"] = float(np.mean([s["lidar_sweeps"] for s in agent]))
    metrics["thermal_frames_mean"] = float(np.mean([s["thermal_frames"] for s in agent]))
    metrics["render_wall_s_mean"] = float(np.mean([s["render_wall_s"] for s in agent]))
    metrics["program_time_s_mean"] = float(np.mean([s["program_time_s"] for s in agent]))
    metrics["cell_work_s_mean"] = float(np.mean([s["cell_work_s"] for s in agent]))
    metrics["cell_internal_errors"] = float(sum(int(s["internal_errors"]) for s in agent))
    record_dir(OUT)
    with open(os.path.join(OUT, "episodes.json"), "w") as f:
        json.dump(agent, f, indent=1, default=float)
    _finish_stats(rn, metrics, t_start)
    overall = write_verdict(rewards, metrics)
    log("[grader] rewards %s overall %.4f (verify wall %.0f s)" % ({k: round(v, 4) for k, v in rewards.items()}, overall, metrics["verify_wall_s"]))


def _finish_stats(rn, metrics, t_start):
    metrics["runner_processes_killed"] = float(rn.killed)
    metrics["runner_processes_left"] = float(rn.left)
    metrics["runner_unclean_runs"] = float(rn.unclean_runs)
    metrics["stray_files_removed"] = float(rn.stray_files)
    metrics["submission_symlinks_skipped"] = float(rn.symlinks_skipped)
    metrics["agent_processes_frozen"] = float(len(FREEZER[0].frozen)) if FREEZER[0] is not None else 0.0
    metrics["verify_wall_s"] = time.time() - t_start
    shutil.rmtree(RUNS_ROOT, ignore_errors=True)


if __name__ == "__main__":
    main()
