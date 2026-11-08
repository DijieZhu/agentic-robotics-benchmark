"""Grader of shoe_lacing_and_boxing (a two-arm finishing cell, camera-only): runs as ROOT in the verify stage.
Verification is separate (task.toml): this runs in a fresh verifier sandbox from the task image, with only the agent's
submission restored into it; tests/test.sh built the root-owned copy of the givens (/opt/arb) before this starts.

The agent left /home/user/submission/packer.py (solve(cell): a camera-only program that laces, boxes and ships pairs of
sneakers with the cell's two Pandas). This grader:
  0. FIRST, stdlib only (`_early_lockdown`; tests/test.sh closed the stage folder already): records the upload time,
     freezes and kills every process of the agent account and keeps it stopped until the verdict is written
     (cellkit.verify_freeze), locks the stage, snapshots the submission root-owned, gates late writes, quarantines late
     entries elsewhere (written for a verify in the agent's own sandbox; in the separate verifier there is nothing of the
     agent's to stop or set aside);
  1. checks the root-owned copy of the givens at /opt/arb (tests/test.sh builds it from setup_payload/ and the image's
     Panda) against given_hashes.json (the cell library and the robot model the grader compiles come from it; a mismatch
     is a task error);
  2. gates: contract_ok = packer.py is a regular file that parses and defines solve, the route gate (no weight files, no
     training calls, size cap), the network audit (static: no networking import, no computed import; runtime: no
     outbound network activity while the program runs), the module imports under the run regime, >= 1 step call, no
     late write, no surviving process; per episode `safety` (a Panda link other than the finger pads pressing more than
     30 N on the bench or an object, or an object faster than 3 m/s) voids the credit of the pair at work;
  3. the oracle token: submission/.oracle_token whose sha256 equals hidden/oracle_token.sha256 switches the cells to the
     privileged mode (the calls get_state_privileged and reference_run; stat privileged_mode); a wrong token is ignored
     (stat oracle_token_mismatch);
  4. runs the agent's program on the fixed hidden set, the `episodes` 63-bit seeds of hidden/seeds.json in their order
     (drawn once from os.urandom and selected by a pre-registered rule, hidden/selection.json; every submission meets
     the same episodes; a missing or malformed set is a task error), one episode per seed on the cell
     (shoecell.server.ShoeCellServer, mode hidden: the work order's shoes, laces and blank drawn from the seed;
     camera-only API) as `nobody` from a read-only copy under an opaque run tag;
  5. criterion: work_score = the episode mean of the cell's accepted work over the work order (per pair: lacing of each
     shoe, box erected, pair packed, lid closed, box shipped), an episode with a safety violation scoring 0, times the
     gates. A failure on the grader's side of an episode (the cell's build, the program's launch, the wire, the
     cleanup, the scoring) ends the run as a TASK ERROR (`fail`: a zero says the agent failed, a task error says the
     grading broke); a program that crashes, hangs or misbehaves ends its own episode, which is scored as it stands, and
     so does an internal error of the cell while it answers one of the program's requests (the program may have caused
     it). Each episode caps the program's own time (`program_time_cap_s`): the wall clock minus the cell's simulation
     and rendering (cellkit.wire), so the budget does not depend on how fast the grading machine simulates and renders.
Untouched sandbox: no packer.py -> contract_ok 0 -> 0. Oracle: the token + reference_run (the scripted reference
controller inside the cell) -> the recorded reference value -> 1.
Records: out/episodes.json (every episode's record) and logs/<tag>.log (the program's stderr) under the root-only
/opt/arb-verify (outside /opt/ale, which the engine clears before it keeps a sandbox).
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
        if left:
            out["reasons"].append("%d process(es) of the agent account survived the kill at verify start" % left)
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

from cellkit import netaudit, wire  # noqa: E402
from cellkit.seeds import load_pinned, seed_rng  # noqa: E402
from shoecell.client import launcher_argv  # noqa: E402
from shoecell.server import ShoeCellServer  # noqa: E402

HIDDEN = os.path.join(STAGE_DIR, "hidden")
# The maintainers' record (the program's stderr, episodes.json) lives outside /opt/ale: the engine deletes its stage
# folders before it keeps a sandbox, and the record is read from a kept one. Root-only (0700). A host-side run as
# another account keeps it in the stage folder, as before.
RECORD_ROOT = os.environ.get("ARB_RECORD_ROOT") or ("/opt/arb-verify" if os.getuid() == 0 else STAGE_DIR)
LOGS = os.path.join(RECORD_ROOT, "logs")
OUT = os.path.join(RECORD_ROOT, "out")
PROGRAM = "packer"
CRITERIA = ["work_score"]
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


def route_gate(cfg, src):
    rg = cfg.get("route_gate", {})
    forb_ext = tuple(rg.get("forbidden_extensions", []))
    calls = rg.get("forbidden_calls", [])
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
                for c in calls:
                    if c in text:
                        problems.append("%s uses %s" % (os.path.relpath(p, src), c))
                suspicious += sum(1 for line in text.splitlines() if any(s in line for s in SUSPICIOUS))
    if total > rg.get("max_submission_mb", 60) * 1024 * 1024:
        problems.append("submission larger than %d MB" % rg.get("max_submission_mb", 60))
    return problems, suspicious


def library_digests():
    """sha256 of every .py file of the shipped library (a submission may carry copies of it)."""
    out = set()
    d = os.path.join(ROOT_GIVEN, "lib", "shoecell")
    for dp, _dn, fns in os.walk(d):
        for fn in fns:
            if fn.endswith(".py"):
                out.add(sha256_file(os.path.join(dp, fn)))
    return out


class Runner(object):
    def __init__(self, cfg):
        self.acct = runner.runner_account(cfg.get("runner_account", "nobody"), AGENT_USER)
        self.name, self.uid, self.gid = self.acct if self.acct else (None, None, None)
        self.lib_dirs = {"shoecell": os.path.join(ROOT_GIVEN, "lib", "shoecell")}
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
        e = runner.child_env(dirs["home"], dirs["tmp"], dirs["lib"], ROOT_GIVEN, extra=extra)
        e["LP_NUM_THREADS"] = "2"
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
    path = os.path.join(src, PROGRAM + ".py")
    if not os.path.isfile(path) or os.path.islink(path):
        return False, "no %s.py (regular file) in the submission" % PROGRAM, 0, []
    try:
        tree = ast.parse(open(path, encoding="utf-8", errors="replace").read())
    except SyntaxError as e:
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


def hidden_set(n):
    """The hidden episodes' seeds: the fixed set of hidden/seeds.json, {"seeds": [...]}, `n` distinct 63-bit seeds run in
    their order (selected once by the rule of review/select_set.py, recorded in hidden/selection.json; the anchor is the
    reference's score on them). A pinned list is used as it stands: a pinned stream's derive() would hash it into other
    seeds. Without the set, or with another count, grading stops as a task error (never a run on other seeds)."""
    path = os.path.join(HIDDEN, "seeds.json")
    try:
        pinned = load_pinned(path)
    except (OSError, ValueError, TypeError) as e:
        fail("hidden/seeds.json unreadable: %s" % e)
    seeds = (pinned or {}).get("seeds")
    if not seeds:
        fail("hidden/seeds.json missing or holds no seed list: the hidden episodes are a fixed set")
    if len(seeds) != n or len(set(seeds)) != n or not all(0 < s < (1 << 63) for s in seeds):
        fail("hidden/seeds.json holds %d seeds; the grader runs %d distinct 63-bit seeds" % (len(seeds), n))
    return seeds


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


def run_episode(cfg, seed, rn, privileged, tag):
    """One graded episode: a fresh cell, the program on the wire as the runner account, the cell's record. A failure on
    the grader's side (the cell's build, the program's launch, the wire, the cleanup, the scoring) is a task error
    (`fail`), never the agent's zero; whatever the program does ends at most its own episode."""
    try:
        server = ShoeCellServer(ROOT_GIVEN, config=cfg, mode="hidden", privileged=privileged, log=log,
                                reference_dir=os.path.join(STAGE_DIR, "reference") if privileged else None)
        server.reset(seed)
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
                                         stderr_path=os.path.join(LOGS, "%s.log" % tag))
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
    overall = float(rewards["work_score"])
    anchor, meta = read_anchor()
    extra = {k: float(v) for k, v in metrics.items()}
    extra.update({k: float(rewards[k]) for k in GATES + CRITERIA})
    verdict = ratio_rewards(overall, anchor, higher_is_better=meta.get("direction", "higher") == "higher",
                            metric_name=meta.get("metric", "work_score"), extra=extra, full_at=meta.get("full_at", 1.0))
    text = ("contract_ok %.0f (the program imports, exposes solve, passes the route gate and the network audit and steps the "
            "cell; no late write, no surviving process); work_score %.4f (episode mean of the accepted work over the work order "
            "on the hidden episodes; a safety violation voids the pair at work)") % (rewards["contract_ok"], rewards["work_score"])
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
    n = int(cfg.get("episodes", 3))
    ep_seeds = hidden_set(n)          # 63-bit: the public draw code cannot be searched over the seed space
    metrics.update(seed_rng(pinned={"seeds": ep_seeds}).stats())   # hidden_seed_hi / _lo: the set's digest halves
    if not ok:
        rewards["contract_ok"] = 0.0
        _finish_stats(rn, metrics, t_start)
        write_verdict(rewards, metrics)
        return
    agent = []
    step_calls = illegal = 0
    for i, seed in enumerate(ep_seeds):
        log("[grader] episode %d/%d" % (i + 1, n))
        s_a = run_episode(cfg, seed, rn, privileged, tag="ep%02d" % (i + 1))
        log("[grader]   work_score %.3f pairs %s safety %s sim %.1f s wall %.1f s (program %.1f s, cell %.1f s) frames %d end %s "
            "violations %s" % (
                s_a["work_score"], [round(p["credit"], 3) for p in s_a["pairs"]], s_a["safety_ok"], s_a["sim_time_s"],
                s_a["episode_wall_s"], s_a["program_time_s"], s_a["cell_work_s"], s_a["camera_frames_total"], s_a["end_reason"],
                [v["what"] for v in s_a["violations"]][:3]))
        agent.append(s_a)
        step_calls += s_a["step_calls"] + s_a["calls"].get("reference_run", 0)
        illegal += s_a["illegal_call_count"]
        metrics["remote_sockets_seen"] = max(metrics["remote_sockets_seen"], float(s_a["remote_sockets_seen"]))
        metrics["net_outbound_events"] += float(s_a.get("net_outbound", 0))
        for ck, cv in (s_a.get("net_delta") or {}).items():
            metrics["net_" + ck] = metrics.get("net_" + ck, 0.0) + float(max(0, cv))
    vals = [float(s["work_score"]) for s in agent]          # a violation already voided the pair at work (episode.py)
    steps_gate = 1.0 if step_calls > 0 else 0.0
    clean_gate = 1.0 if rn.unclean_runs == 0 else 0.0
    net_gate = 1.0 if (metrics["remote_sockets_seen"] <= 0 and metrics["net_outbound_events"] <= 0) else 0.0
    gate = steps_gate * clean_gate * net_gate
    rewards["contract_ok"] = gate
    if not net_gate:
        log("[grader] contract_ok 0: network activity during an episode (%d outbound events, %d remote sockets seen)" % (
            metrics["net_outbound_events"], metrics["remote_sockets_seen"]))
    rewards["work_score"] = float(max(0.0, min(1.0, gate * float(np.mean(vals)))))
    metrics["work_score_mean_ungated"] = float(np.mean([s["work_score"] for s in agent]))
    metrics["safety_violation_rate"] = float(np.mean([0.0 if s["safety_ok"] else 1.0 for s in agent]))
    metrics["mean_sim_time_s"] = float(np.mean([s["sim_time_s"] for s in agent]))
    metrics["mean_episode_wall_s"] = float(np.mean([s["episode_wall_s"] for s in agent]))
    metrics["illegal_call_count"] = float(illegal)
    metrics["step_calls"] = float(step_calls)
    metrics["camera_frames_mean"] = float(np.mean([s["camera_frames_total"] for s in agent]))
    for cam in sorted(set(k for s in agent for k in s["camera_frames"])):
        metrics["camera_frames_mean." + cam] = float(np.mean([s["camera_frames"].get(cam, 0) for s in agent]))
    metrics["render_wall_s_mean"] = float(np.mean([s["render_wall_s"] for s in agent]))
    metrics["program_time_s_mean"] = float(np.mean([s["program_time_s"] for s in agent]))
    metrics["cell_work_s_mean"] = float(np.mean([s["cell_work_s"] for s in agent]))
    metrics["cell_internal_errors"] = float(sum(int(s["internal_errors"]) for s in agent))
    for key in ("lacing_a", "lacing_b", "erected", "packed", "closed", "shipped"):
        metrics["milestone_mean." + key] = float(np.mean([p["milestones"].get(key, 0.0) for s in agent for p in s["pairs"]])) if any(s["pairs"] for s in agent) else 0.0
    metrics["pairs_shipped_mean"] = float(np.mean([s["pairs_shipped"] for s in agent]))
    metrics["pairs_started_mean"] = float(np.mean([len(s["pairs"]) for s in agent]))
    record_dir(OUT)
    with open(os.path.join(OUT, "episodes.json"), "w") as f:
        json.dump([{k: v for k, v in s.items() if k not in ("events",)} for s in agent], f, indent=1, default=float)
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
