"""Grader of house_cleaning_with_tools (one Unitree G1 humanoid cleans a studio apartment after a dinner, with tools,
camera-only): runs as ROOT in the verify stage.

The agent left /home/user/submission/clean.py (solve(cell): a camera-only program that drives the cell's G1 through the
house cell's JSON-lines API: the trash sorted into the two bins (cans and the bottle into the recycling bin, napkins
and the apple core into the general-waste bin), the dishes to the dish tray, the books to the bookshelf, the grease off
the dining table with the sponge, soap and water, the crumbs swept with the broom and the lobby dustpan into the
general-waste bin, and the tools put back).
Verification is separate (task.toml): this runs in a fresh verifier sandbox from the task image, into which the engine
restored only the submission (root-owned); nothing else of the agent's sandbox is here (no ~/given, no process the
agent started; the task has no practice cell). This grader:
  0. FIRST, stdlib only (`_early_lockdown`; tests/test.sh closed the stage folder already): records the upload time,
     freezes and kills every process of the agent account and keeps it stopped until the verdict is written
     (cellkit.verify_freeze), locks the stage, snapshots the submission root-owned, gates late writes, quarantines late
     entries elsewhere (written for a verify in the agent's own sandbox; in the separate verifier they stop the image's
     own desktop session of the agent account, started with the sandbox, and set aside the files that session writes);
  1. checks the root-owned copy of the givens at /opt/arb against given_hashes.json (tests/test.sh builds it from
     tests/setup_payload/ and the image's robot model; the cell client library the program runs with comes from it; a
     mismatch is a task error);
  2. gates: contract_ok = clean.py is a regular file that parses and defines solve, the route gate (no model-weight
     files by extension, the size cap; grader_config.json route_gate, stated in the instruction), the network audit
     (static: no networking import, no computed import; runtime: no remote socket of the runner account's processes
     while the program runs; the namespace's outbound counters, which loopback traffic raises too, are recorded only),
     the module imports under the run regime, >= 1 step call, no late write, no surviving process; per episode the
     cell's safety watch (house_cell.physics.Safety: the robot's contact forces, objects thrown or struck; the numbers
     are in the cell manual) zeroes that episode;
  3. the oracle token: submission/.oracle_token whose sha256 equals hidden/oracle_token.sha256 switches the cells to the
     privileged mode (the calls `reference_clean` and `truth`, on the same hidden layouts; stat privileged_mode); a
     wrong token is ignored (stat oracle_token_mismatch);
  4. runs the program once per seed of the FIXED hidden scene set hidden/seeds.json (its selection rule and record:
     tests/review/hidden_set.md; agents and the oracle get the same scenes) on a fresh cell per episode
     (house_cell.server.CellServer: where every loose object lies and how it faces, the grease stain, the crumbs, which
     bin stands where, the robot's start pose and the sensor noise are drawn from the seed; camera-only API) as
     `nobody` from a read-only copy under an opaque run tag (world-readable files copied from the snapshot, so the
     runner account reads a submission the engine restored root-owned);
  5. criterion: cleaning_score = the episode mean of the cell's stage score (house_cell.scoring: the customer's
     checklist, credit over its total weight of 10: trash sorted 2, dishes cleared 1, books shelved 1, table
     degreased 2.5, floor swept 2.5, finishing 1 for the sponge rinsed in its tray, the tools in their clips, the
     faucet off and the lids closed, less the damage charges), an episode with a safety violation scoring 0, times the
     gates. A failure on the grader's side of an episode (the cell's build, the program's launch, the wire, the
     cleanup, the scoring, a camera the renderer failed to draw, a score that is not a finite number) ends the
     run as a TASK ERROR (robotics_grader.fail: a zero says the agent failed, a task error says the grading broke); a
     program that crashes, hangs or misbehaves ends its own episode, which is scored as it stands, and so does an
     internal error of the cell while it answers one of the program's requests (the program may have caused it).
     Each episode caps the program's own time (`program_time_cap_s`): the wall clock minus the cell's simulation and
     rendering (cellkit.wire), so the budget does not depend on how fast the grading machine simulates and renders.
Untouched sandbox: no clean.py -> contract_ok 0 -> 0.
Oracle: the token + reference_clean (the privileged reference cleaner inside the cell, on the hidden scenes) -> 1.
Records: out/episodes.json (every episode's record) and logs/<tag>.log (the program's stderr) under the root-only
/opt/arb-verify (outside /opt/ale, which the engine clears before it keeps a sandbox).
"""

import ast
import json
import math
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
CONFIG = os.path.join(STAGE_DIR, "grader_config.json")
CLIENT_PACKAGE = "house_client"

from cellkit import lockdown, runner, verify_freeze  # noqa: E402  (stdlib only)

FREEZER = [None]        # the agent account's verify_freeze.AgentFreezer: started by _early_lockdown, stopped after the verdict


def log(msg):
    print(msg, file=sys.stderr, flush=True)


def _uid_of(name):
    try:
        return pwd.getpwnam(name).pw_uid
    except (KeyError, TypeError):
        return None


def _proc_state(pid):
    """The /proc state letter of `pid` ("X" once it is gone)."""
    try:
        with open("/proc/%d/status" % pid) as fh:
            for line in fh:
                if line.startswith("State:"):
                    return line.split()[1]
    except (OSError, IndexError):
        pass
    return "X"


def _still_running(uid, bound_s=3.0):
    """PIDs of the account's live processes that are not stopped, after the freezer has had up to `bound_s` to stop
    them (a respawn that appears meanwhile is frozen by the next pass); empty when every one is stopped."""
    deadline = time.time() + float(bound_s)
    while True:
        FREEZER[0].freeze_now()
        time.sleep(0.05)
        running = [pid for pid in lockdown.live_uid_processes(uid) if _proc_state(pid) not in ("T", "t", "X")]
        if not running or time.time() > deadline:
            return running


def _early_lockdown():
    out = {"agent_processes_killed": 0.0, "agent_processes_left": 0.0, "stage_locked": 0.0, "stage_parent_locked": 0.0,
           "stage_lock_latency_ms": -1.0, "submission_late_writes": 0.0, "late_writes_quarantined": 0.0,
           "late_write_gate_active": 0.0, "submission_snapshot": 0.0, "saved": None, "parent": None, "t_ref": None,
           "late": [], "reasons": []}
    if os.getuid() != 0:
        return out
    t_ref = lockdown.stage_upload_time(STAGE_DIR)
    out["t_ref"] = t_ref
    uid = _uid_of(AGENT_USER)
    if uid is not None and uid != 0:
        killed, left = lockdown.stop_and_kill_uid(uid, grace_s=8.0)
        out["agent_processes_killed"], out["agent_processes_left"] = float(killed), float(left)
    # the image's init respawns the agent account's desktop within seconds of the kill, and the respawned session runs
    # the account's autostart entries: every process of the account stays SIGSTOPped (polled every 0.1 s) until the
    # verdict is written.
    FREEZER[0] = verify_freeze.AgentFreezer.for_user(AGENT_USER).start()
    if out["agent_processes_left"]:
        # The image's supervisord respawns the account's desktop as fast as it is killed, so in the separate verifier a
        # process that outlives the kill is the image's own (2026-11-07: one respawned desktop process outlived this kill
        # and set contract_ok 0 before any episode ran). It is frozen with the rest of the account; only a process that
        # is still running once the freezer had its turn voids the contract.
        running = _still_running(uid)
        if running:
            out["reasons"].append("%d process(es) of the agent account survived the kill at verify start and could not be "
                                  "stopped" % len(running))
        else:
            log("[grader] %d process(es) of the agent account (the image's respawned desktop) outlived the kill at verify "
                "start: frozen until the verdict" % int(out["agent_processes_left"]))
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

from cellkit import netaudit, wire  # noqa: E402
from house_cell import objects as HO  # noqa: E402
from house_cell import scoring as HS  # noqa: E402
from house_cell.server import CellServer  # noqa: E402

HIDDEN = os.path.join(STAGE_DIR, "hidden")
# The maintainers' record (the program's stderr, episodes.json) lives outside /opt/ale: the engine deletes its stage
# folders before it keeps a sandbox, and the record is read from a kept one. Root-only (0700).
RECORD_ROOT = os.environ.get("ARB_RECORD_ROOT", "/opt/arb-verify")
LOGS = os.path.join(RECORD_ROOT, "logs")
OUT = os.path.join(RECORD_ROOT, "out")


def record_dir(path):
    """`path` under the root-only record root, created on demand; never a scoring input."""
    os.makedirs(RECORD_ROOT, exist_ok=True)
    os.chmod(RECORD_ROOT, 0o700)
    os.makedirs(path, exist_ok=True)
    return path
PROGRAM = "clean"
CRITERIA = ["cleaning_score"]
GATES = ["contract_ok"]
SUSPICIOUS = ("/opt/ale", "ALE_STAGE_DIR", "ALE_VERIFICATION", "hidden/seeds", "oracle_token", "reference_clean",
              "house_cell", "/opt/arb/runs", "ALE_VERIFY", "/opt/ale/tests", ".ale-solution")
PARTS = tuple(HS.WEIGHTS)       # the checklist's parts (house_cell.scoring), as the record's credit_by_part names them
END_REASONS = ("finish", "eof", "sim_cap", "program_time_cap", "silence", "stalled", "protocol", "internal_error", "done")


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
    if not expected:
        return False, ["given_hashes.json lists no file"]
    for rel, digest in expected.items():
        p = os.path.join(ROOT_PAYLOAD, rel)
        try:
            st = os.lstat(p)
        except OSError:
            problems.append("missing " + rel)
            continue
        if os.path.islink(p) or not statmod.S_ISREG(st.st_mode) or st.st_uid != owner or (st.st_mode & 0o022):
            problems.append("not a root-owned, group/world-unwritable regular file: " + rel)
            continue
        if sha256_file(p) != digest:
            problems.append("differs " + rel)
    return not problems, problems


def route_gate(cfg, src):
    """(problems, suspicious lines): model-weight files by extension (grader_config.json route_gate), the size cap."""
    rg = cfg.get("route_gate", {})
    forb_ext = tuple(rg.get("forbidden_extensions", []))
    total, problems, suspicious = 0, [], 0
    for dp, _, fns in os.walk(src):
        for fn in fns:
            p = os.path.join(dp, fn)
            try:
                total += os.lstat(p).st_size
            except OSError:
                continue
            if fn.lower().endswith(forb_ext):
                problems.append("weight file %s" % os.path.relpath(p, src))
            if fn.endswith((".py", ".sh")) and not os.path.islink(p):
                try:
                    text = open(p, encoding="utf-8", errors="replace").read()
                except OSError:
                    continue
                suspicious += sum(1 for line in text.splitlines() if any(s in line for s in SUSPICIOUS))
    if total > rg.get("max_submission_mb", 60) * 1024 * 1024:
        problems.append("submission larger than %d MB" % rg.get("max_submission_mb", 60))
    return problems, suspicious


def library_digests():
    """sha256 of every .py file of the shipped library (a submission may carry byte-identical copies of it, e.g. the cell
    client)."""
    out = set()
    d = os.path.join(ROOT_GIVEN, "lib")
    for dp, _dn, fns in os.walk(d):
        for fn in fns:
            if fn.endswith(".py"):
                out.add(sha256_file(os.path.join(dp, fn)))
    return out


def library_entries():
    """({package: dir}, [module files]) of the root-owned copy of the given library (/opt/arb/given/lib): the run's lib/
    is a copy of all of it (the cell client house_client at least)."""
    lib = os.path.join(ROOT_GIVEN, "lib")
    dirs, files = {}, []
    for name in sorted(os.listdir(lib)) if os.path.isdir(lib) else []:
        p = os.path.join(lib, name)
        if os.path.islink(p) or name == "__pycache__":
            continue
        if os.path.isdir(p):
            dirs[name] = p
        elif os.path.isfile(p):
            files.append(p)
    if CLIENT_PACKAGE not in dirs:
        dirs[CLIENT_PACKAGE] = os.path.join(lib, CLIENT_PACKAGE)     # root_payload_ok has checked it is there
    return dirs, files


class Runner(object):
    def __init__(self, cfg):
        self.acct = runner.runner_account(cfg.get("runner_account", "nobody"), AGENT_USER)
        self.name, self.uid, self.gid = self.acct if self.acct else (None, None, None)
        self.lib_dirs, self.lib_files = library_entries()
        self.cpu_s = float(cfg.get("cpu_cap_factor", 4.0)) * float(cfg.get("program_time_cap_s", 3600.0))
        self.killed = self.left = self.unclean_runs = self.stray_files = self.symlinks_skipped = 0
        os.makedirs(RUNS_ROOT, exist_ok=True)
        os.chmod(RUNS_ROOT, 0o755)

    def prepare(self, tag, program_dir, private=False):
        root = os.path.join(RUNS_ROOT, tag)
        if os.path.isdir(root):
            shutil.rmtree(root, ignore_errors=True)
        owner = (self.uid, self.gid) if (self.acct and not private) else None
        dirs = runner.prepare_run_dir(root, program_dir, self.lib_dirs, owner=owner, private=private)
        for p in self.lib_files:
            dst = os.path.join(dirs["lib"], os.path.basename(p))
            shutil.copyfile(p, dst)
            os.chmod(dst, 0o644)
        self.symlinks_skipped = max(self.symlinks_skipped, dirs["skipped"])
        return dirs

    def env(self, dirs, extra=None):
        return runner.child_env(dirs["home"], dirs["tmp"], dirs["lib"], ROOT_GIVEN, extra=extra)

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
    path = os.path.join(src, PROGRAM + ".py")
    if not os.path.isfile(path) or os.path.islink(path):
        return False, "no %s.py (regular file) in the submission" % PROGRAM, 0, []
    try:
        tree = ast.parse(open(path, encoding="utf-8", errors="replace").read())
    except (SyntaxError, ValueError) as e:
        return False, "%s.py does not parse: %s" % (PROGRAM, e), 0, []
    has_solve = any(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == "solve" for n in ast.walk(tree)) or \
        any(isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "solve" for t in n.targets) for n in ast.walk(tree)) or \
        any(isinstance(n, (ast.Import, ast.ImportFrom)) and any((a.asname or a.name) == "solve" for a in n.names) for n in ast.walk(tree))
    if not has_solve:
        return False, "%s.py defines no solve" % PROGRAM, 0, []
    problems, suspicious = route_gate(cfg, src)
    if problems:
        return False, "route gate: " + "; ".join(problems[:5]), suspicious, []
    net = netaudit.scan_imports(src, allow_digests=library_digests())
    if net:
        return False, "network audit: networking import(s) %s" % ", ".join("%s: %s" % x for x in net[:5]), suspicious, net
    dirs = rn.prepare("import_check", src)
    before = rn.snapshot((dirs["root"],)) if rn.acct else None
    code = ("import sys, importlib; sys.path.insert(0, sys.argv[1]); sys.path.insert(0, sys.argv[2]); "
            "m = importlib.import_module('%s'); assert callable(m.solve)" % PROGRAM)
    argv = runner.wrap_limits([sys.executable] + runner.PYTHON_FLAGS + ["-c", code, dirs["lib"], dirs["program"]], cpu_s=120, user=rn.name)
    try:
        r = subprocess.run(argv, cwd=dirs["program"], env=rn.env(dirs), capture_output=True, text=True, timeout=120, start_new_session=True)
        ok, why = r.returncode == 0, ("ok" if r.returncode == 0 else "importing %s.py failed: " % PROGRAM + (r.stderr.strip().splitlines() or ["?"])[-1][:300])
    except subprocess.TimeoutExpired:
        ok, why = False, "importing %s.py took more than 120 s" % PROGRAM
    finally:
        rn.cleanup(dirs, before)
    return ok, why, suspicious, []


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


def episode_seeds(n):
    """(seeds, pinned): the fixed hidden scene set, hidden/seeds.json {"seeds": [...]} (tests/review/hidden_set.md: how
    it was selected); `n` episodes, the first n of the list. A missing or unreadable list is a task error."""
    pinned = os.path.join(HIDDEN, "seeds.json")
    try:
        seeds = [int(s) & ((1 << 63) - 1) for s in load_json(pinned)["seeds"]]
    except (OSError, ValueError, KeyError, TypeError) as e:
        fail("the hidden scene set %s is unreadable: %r" % (pinned, e))
    if len(seeds) < n:
        fail("the hidden scene set holds %d scenes, %d episodes are configured" % (len(seeds), n))
    return seeds[:n], True


def run_episode(cfg, seed, rn, privileged, tag):
    """One graded episode: a fresh cell, the program on the wire as the runner account, the cell's record. A failure on
    the grader's side (the cell's build, the program's launch, the wire, the cleanup, the scoring) is a task error
    (`fail`), never the agent's zero; whatever the program does ends at most its own episode."""
    t0 = time.time()
    try:
        server = CellServer(seed, privileged=privileged, nominal=False, config=cfg, log=log)
    except Exception:  # noqa: BLE001 - the cell depends on the seed only: a build failure is the task's
        fail("%s: the cell could not be built\n%s" % (tag, traceback.format_exc()))
    # the program's own time (the wall clock minus the cell's simulation and rendering, cellkit.wire); the privileged
    # reference cleaner runs inside one call (reference_clean) and has its own cap
    program_cap = float(cfg.get("privileged_program_time_cap_s", 10800.0) if privileged
                        else cfg.get("program_time_cap_s", 3600.0))
    silence = float(cfg.get("silence_s", 120.0))
    proc = dirs = before = None
    record_dir(LOGS)
    sampler = netaudit.SocketSampler(uid=rn.uid if rn.acct else None, pids_fn=None if rn.acct else (lambda: []),
                                     period_s=float(cfg.get("socket_sample_s", 0.5)))
    counters = netaudit.NetCounters()
    with counters, sampler:
        try:
            try:
                dirs = rn.prepare(tag, program_source())      # an opaque tag: the hidden seed never reaches a path the program sees
                before = rn.snapshot((dirs["root"],)) if rn.acct else None
                argv = runner.launcher_argv(sys.executable, dirs["lib"], dirs["program"], PROGRAM)
                proc = wire.AgentProcess(argv, cwd=dirs["program"], user=rn.name, env=rn.env(dirs), cpu_s=rn.cpu_s,
                                         stderr_path=os.path.join(LOGS, "%s.log" % tag))
            except Exception:  # noqa: BLE001
                fail("%s: the program could not be launched\n%s" % (tag, traceback.format_exc()))
            try:
                reason = wire.serve_episode(server, proc, program_cap, silence, log=log)
            except Exception:  # noqa: BLE001 - the wire turns everything the program does into an end reason
                fail("%s: the wire failed\n%s" % (tag, traceback.format_exc()))
        finally:
            if proc is not None:
                proc.terminate()
            if dirs is not None:
                rn.cleanup(dirs, before)              # every process of the runner account is gone before the counters are read
    try:
        if not server.done:
            server._end(reason)
        summary = server.finalize()
    except Exception:  # noqa: BLE001 - finalize() does not raise by design
        fail("%s: the episode could not be scored\n%s" % (tag, traceback.format_exc()))
    if summary.get("score_error"):
        fail("%s: the episode could not be scored: %s" % (tag, summary["score_error"]))
    if int(summary.get("render_errors", 0) or 0) > 0:
        # a camera the renderer could not draw (a renderer error or its call timeout) blinded the program: the
        # grading machine's failure, never the agent's score
        fail("%s: the renderer failed on %d camera request(s); the episode is not scored" % (
            tag, int(summary["render_errors"])))
    try:
        sc = float(summary.get("stage_score", 0.0))
    except (TypeError, ValueError):
        sc = float("nan")
    if not math.isfinite(sc):
        fail("%s: the episode's score is not a finite number (%r)" % (tag, summary.get("stage_score")))
    summary["episode_wall_s"] = time.time() - t0
    summary["end_reason"] = reason
    summary["remote_sockets_seen"] = int(sampler.max_seen)
    summary["socket_samples"] = int(sampler.samples)
    summary["net_outbound"] = int(counters.total)
    summary["net_delta"] = dict(counters.delta)
    return summary


def write_verdict(rewards, metrics):
    """One gated `reward` in the benchmark kit's envelope (robotics_grader.ratio_rewards against the recorded reference
    value); the gate, the criterion and every statistic go to `metrics`. In the engine (`ALE_VERIFY_PYTHON` set) the kit
    writes reward.json and verification.json, the criteria text as the `diagnostic`; the metrics keep the names and the
    filter of this grader's records on the earlier engine API (finite numbers only, '.' and '/' in a name replaced by
    '_'). Elsewhere the envelope goes to verdict.local.json in the stage folder."""
    from robotics_grader import ratio_rewards, read_anchor
    from robotics_grader import write_verdict as kit_write_verdict
    overall = float(rewards["cleaning_score"])
    anchor, meta = read_anchor()
    extra = {}
    for k, v in metrics.items():
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        if f == f and abs(f) != float("inf"):
            extra[k] = f
    extra.update({k: float(rewards[k]) for k in GATES + CRITERIA})
    verdict = ratio_rewards(overall, anchor, higher_is_better=meta.get("direction", "higher") == "higher",
                            metric_name=meta.get("metric", "cleaning_score"), extra=extra, full_at=meta.get("full_at", 1.0))
    text = ("contract_ok %.0f (the program imports, exposes solve, passes the route gate and the network audit and steps the "
            "cell; no late write, no surviving process); cleaning_score %.4f (episode mean of the cell's stage score: the "
            "customer's checklist credit over its total weight of 10 (trash sorted into the right bins 2, dishes on the dish "
            "tray 1, books on the shelf 1, grease off the dining table 2.5, crumbs in the general-waste bin 2.5, finishing 1: "
            "the sponge rinsed in its tray, the broom and the dustpan in their clips, the faucet off, the lids closed), on "
            "the hidden episodes, an episode with a safety violation scoring 0)") % (rewards["contract_ok"], rewards["cleaning_score"])
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
    cfg = load_json(CONFIG)
    rewards = dict((k, 0.0) for k in GATES + CRITERIA)
    metrics = {"grader_exception": 0.0, "episodes": float(cfg.get("episodes", 3)),
               "root_payload_ok": 0.0, "agent_deprivileged": 0.0, "runner_uid": -1.0, "privileged_mode": 0.0, "oracle_token_mismatch": 0.0,
               "runner_processes_killed": 0.0, "runner_processes_left": 0.0, "runner_unclean_runs": 0.0,
               "stray_files_removed": 0.0, "suspicious_lines": 0.0, "remote_sockets_seen": 0.0, "net_outbound_events": 0.0,
               "network_imports": 0.0, "illegal_call_count": 0.0, "step_calls": 0.0, "seeds_pinned": 0.0}
    for k in ("stage_locked", "stage_parent_locked", "stage_lock_latency_ms", "agent_processes_killed", "agent_processes_left",
              "submission_late_writes", "late_writes_quarantined", "late_write_gate_active", "submission_snapshot"):
        metrics[k] = float(EARLY.get(k, -1.0 if k == "stage_lock_latency_ms" else 0.0))
    try:
        try:
            _main_locked(cfg, rewards, metrics)
        except SystemExit:
            raise
        except Exception:  # noqa: BLE001 - a grader defect is a task error, and says so
            log("[grader] EXCEPTION\n" + traceback.format_exc())
            metrics["grader_exception"] = 1.0
            raise
    finally:
        _restore_modes()
        if FREEZER[0] is not None:
            FREEZER[0].stop()                   # after the verdict (written by _main_locked); the frozen stay stopped


def _mean(xs):
    xs = [float(x) for x in xs]
    return float(np.mean(xs)) if xs else 0.0


def episode_metrics(agent, metrics):
    """The record's statistics over the episodes (never a scoring input)."""
    metrics["stage_score_mean_ungated"] = _mean(s.get("stage_score", 0.0) for s in agent)
    metrics["credit_mean"] = _mean(s.get("credit", 0.0) for s in agent)
    for part in PARTS:
        cr = [float((s.get("credit_by_part") or {}).get(part, 0.0)) for s in agent]
        metrics["credit_mean." + part] = _mean(cr)
        metrics["part_rate." + part] = _mean(c / float(HS.WEIGHTS[part]) for c in cr)
    metrics["grease_removed_fraction_mean"] = _mean(s.get("grease_removed_fraction", 0.0) for s in agent)
    metrics["crumbs_binned_fraction_mean"] = _mean(s.get("crumbs_binned_fraction", 0.0) for s in agent)
    for what, names in (("trash", HO.TRASH), ("dishes", HO.DISHES), ("books", HO.BOOKS)):
        metrics[what + "_ok_mean"] = _mean(sum(1 for n in names if ((s.get("items") or {}).get(n) or {}).get("ok"))
                                           for s in agent)
    metrics["dishes_broken_mean"] = _mean(len(s.get("broken") or {}) for s in agent)
    metrics["things_on_floor_mean"] = _mean(len((s.get("damage") or {}).get("on_floor") or []) for s in agent)
    metrics["penalty_mean"] = _mean(s.get("penalty", 0.0) for s in agent)
    metrics["credit_earned_mean"] = _mean(s.get("credit_earned", s.get("credit", 0.0)) for s in agent)
    metrics["sponge_in_tray_rate"] = _mean(1.0 if (s.get("sponge") or {}).get("in_tray") else 0.0 for s in agent)
    metrics["sponge_rinsed_rate"] = _mean(1.0 if (s.get("sponge") or {}).get("rinsed") else 0.0 for s in agent)
    metrics["sponge_peak_load_mean"] = _mean((s.get("sponge") or {}).get("peak_load", 0.0) for s in agent)
    metrics["tools_clipped_mean"] = _mean(sum(1 for v in (s.get("tools_clipped") or {}).values() if v) for s in agent)
    metrics["faucet_off_rate"] = _mean(1.0 if s.get("faucet_off") else 0.0 for s in agent)
    metrics["lids_closed_mean"] = _mean(sum(1 for a in (s.get("lid_angles") or {}).values() if float(a) < HS.LID_CLOSED)
                                        for s in agent)
    for k in ("wipe_s", "sweep_s", "water_s", "soap_strokes", "soap_doses_missed", "grease_smeared"):
        metrics["process_mean." + k] = _mean((s.get("process") or {}).get(k, 0.0) for s in agent)
    for state in sorted(set(k for s in agent for k in ((s.get("process") or {}).get("crumbs") or {}))):
        metrics["crumbs_mean." + state] = _mean(((s.get("process") or {}).get("crumbs") or {}).get(state, 0) for s in agent)
    metrics["safety_violation_rate"] = _mean(0.0 if s.get("safety_ok", True) else 1.0 for s in agent)
    metrics["mean_sim_time_s"] = _mean(s.get("sim_time_s", 0.0) for s in agent)
    metrics["mean_episode_wall_s"] = _mean(s.get("episode_wall_s", 0.0) for s in agent)
    metrics["camera_frames_mean"] = _mean(s.get("camera_frames_total", 0) for s in agent)
    for cam in sorted(set(k for s in agent for k in (s.get("camera_frames") or {}))):
        metrics["camera_frames_mean." + cam] = _mean((s.get("camera_frames") or {}).get(cam, 0) for s in agent)
    metrics["render_wall_s_mean"] = _mean(s.get("render_wall_s", 0.0) for s in agent)
    metrics["program_time_s_mean"] = _mean(s.get("program_time_s", 0.0) for s in agent)
    metrics["cell_work_s_mean"] = _mean(s.get("cell_work_s", 0.0) for s in agent)
    metrics["bad_param_calls"] = float(sum(int(s.get("bad_param_calls", 0)) for s in agent))
    metrics["cell_internal_errors"] = float(sum(int(s.get("internal_errors", 0)) for s in agent))
    metrics["render_errors"] = float(sum(int(s.get("render_errors", 0)) for s in agent))
    for peak in sorted(set(k for s in agent for k in (s.get("safety_peaks") or {}))):
        metrics["safety_peak_max." + peak] = float(max([float((s.get("safety_peaks") or {}).get(peak, 0.0)) for s in agent] or [0.0]))
    for r in END_REASONS:
        metrics["end_reason." + r] = float(sum(1 for s in agent if s.get("end_reason") == r))


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
    if not ok:
        rewards["contract_ok"] = 0.0
        _finish_stats(rn, metrics, t_start)
        write_verdict(rewards, metrics)
        return
    n = int(cfg.get("episodes", 3))
    ep_seeds, pinned = episode_seeds(n)
    metrics["seeds_pinned"] = 1.0 if pinned else 0.0
    metrics["episodes"] = float(len(ep_seeds))
    agent = []
    step_calls = illegal = 0
    for i, seed in enumerate(ep_seeds):
        tag = "ep%02d" % (i + 1)
        log("[grader] episode %d/%d" % (i + 1, len(ep_seeds)))
        s_a = run_episode(cfg, seed, rn, privileged, tag)    # a grader-side failure inside is a task error
        s_a["tag"] = tag
        log("[grader]   stage score %.3f (credit %.2f of %g: %s; less %.2f for damage %s; grease %.2f removed, crumbs %.2f "
            "binned) safety %s sim %.1f s "
            "wall %.1f s (program %.1f s, cell %.1f s) frames %d end %s calls %s illegal %d bad params %d violations %s" % (
                float(s_a.get("stage_score", 0.0)), float(s_a.get("credit", 0.0)), HS.TOTAL,
                ", ".join("%s %.2f" % kv for kv in sorted((s_a.get("credit_by_part") or {}).items())),
                float(s_a.get("penalty", 0.0)), s_a.get("damage"), float(s_a.get("grease_removed_fraction", 0.0)),
                float(s_a.get("crumbs_binned_fraction", 0.0)),
                s_a.get("safety_ok"), float(s_a.get("sim_time_s", 0.0)), float(s_a.get("episode_wall_s", 0.0)),
                float(s_a.get("program_time_s", 0.0)), float(s_a.get("cell_work_s", 0.0)),
                int(s_a.get("camera_frames_total", 0)), s_a.get("end_reason"), s_a.get("calls"),
                int(s_a.get("illegal_call_count", 0)), int(s_a.get("bad_param_calls", 0)),
                [v.get("what") for v in (s_a.get("violations") or [])][:3]))
        agent.append(s_a)
        # the privileged reference cleaner steps the cell inside its one call; without the token that call was refused
        # (an illegal call) and must not pass the steps gate
        ref_calls = int((s_a.get("calls") or {}).get("reference_clean", 0)) if privileged else 0
        step_calls += int(s_a.get("step_calls", 0)) + ref_calls
        illegal += int(s_a.get("illegal_call_count", 0))
        metrics["remote_sockets_seen"] = max(metrics["remote_sockets_seen"], float(s_a.get("remote_sockets_seen", 0)))
        metrics["net_outbound_events"] += float(s_a.get("net_outbound", 0))
        for ck, cv in (s_a.get("net_delta") or {}).items():
            metrics["net_" + ck] = metrics.get("net_" + ck, 0.0) + float(max(0, cv))
    vals = [float(s.get("stage_score", 0.0)) * (1.0 if s.get("safety_ok", False) else 0.0) for s in agent]
    steps_gate = 1.0 if step_calls > 0 else 0.0
    clean_gate = 1.0 if rn.unclean_runs == 0 else 0.0
    # the runner account's remote sockets gate (loopback excluded); the namespace-wide counters (net_*) also count
    # loopback traffic and the verifier's own services, so they are recorded, not gated
    net_gate = 1.0 if metrics["remote_sockets_seen"] <= 0 else 0.0
    gate = steps_gate * clean_gate * net_gate
    rewards["contract_ok"] = gate
    if not steps_gate:
        log("[grader] contract_ok 0: the program never stepped the cell")
    if not clean_gate:
        log("[grader] contract_ok 0: %d run(s) left processes of the runner account behind" % rn.unclean_runs)
    if not net_gate:
        log("[grader] contract_ok 0: network activity during an episode (%d remote sockets seen; %d namespace outbound "
            "events)" % (metrics["remote_sockets_seen"], metrics["net_outbound_events"]))
    rewards["cleaning_score"] = float(max(0.0, min(1.0, gate * _mean(vals))))
    metrics["illegal_call_count"] = float(illegal)
    metrics["step_calls"] = float(step_calls)
    episode_metrics(agent, metrics)
    try:
        record_dir(OUT)
        with open(os.path.join(OUT, "episodes.json"), "w") as f:
            json.dump(agent, f, indent=1, default=str)
    except (OSError, TypeError, ValueError) as e:
        log("[grader] out/episodes.json not written: %r" % (e,))
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
