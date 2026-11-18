"""Grader of rov_subsea_manifold_leak_isolation (an inspection-class ROV, alone on the seabed, finds and isolates the
leaking branch of a subsea production manifold with its own sensors): runs as ROOT in the verify stage.  Verification is
separate (task.toml): this runs in a fresh verifier sandbox from the task image, with only the agent's submission
restored into it; tests/test.sh built the root-owned copy of the givens (/opt/arb) before this starts.

The agent left /home/user/submission/intervene.py (solve(vehicle): the vehicle's autonomous program).  This grader:
  0. FIRST, stdlib only (`_early_lockdown`; tests/test.sh closed the stage folder already): records the upload time,
     freezes and kills every process of the agent account and keeps it stopped until the verdict is written
     (cellkit.verify_freeze), locks the stage, snapshots the submission root-owned, gates late writes, quarantines late
     entries elsewhere (written for a verify in the agent's own sandbox; in the separate verifier there is nothing of
     the agent's to stop or set aside);
  1. checks the root-owned copy of the givens at /opt/arb against given_hashes.json (the client library handed to the
     programs comes from it; a mismatch is a task error);
  2. gates: contract_ok = intervene.py is a regular file that parses and defines solve, the route gate (size cap, no
     weight files, no training calls), the network audit (static: no networking import, no computed import or code, no
     program start; runtime: every process of the program runs under the no-network seccomp filter (cellkit.nonet) and
     the runner accounts' remote inet sockets are sampled), the module imports under the run regime (120 s, one core),
     >= 1 step call, no late write, no surviving process; before the first station it waits until the frozen desktop
     is quiet and proves the filter works (`nonet.selftest`; a failure is a task error);
  3. the oracle token: submission/.oracle_token whose sha256 equals hidden/oracle_token.sha256 switches the cells to
     the privileged mode (get_state_privileged, reference_run: the reference program inside the cell); a wrong token
     is ignored (stat oracle_token_mismatch);
  4. runs the program on the fixed hidden set (hidden/stations.json: six stored site packages, each pinned by sha256;
     a missing, short or mismatched set is a task error) in SLOTS (DESIGN section 13, decisions M12 / N7): up to three
     stations at a time, each slot = one cell process (station.py, single-threaded, pinned to its own core) + one
     program process (the slot's own runner account, pinned to its own core, affinity locked); stations start in their
     fixed order as slots free up (slots.SlotScheduler, queue mode: never later than two waves of three); a station
     whose cell failed (build, internal error, physics fault, lost frame, scoring inconsistency) is run once more from
     scratch after all stations have ended (at most one station; the second run counts); a failure of the harness
     itself (the program's launch, the worker, a station past its watchdog) is a TASK ERROR.  Each station's result
     depends on the station only (time passes only in step(); separate processes, directories and random streams),
     so the slot count, the cores and the completion order change nothing (`ARB_VERIFY_MAX_SLOTS=1` runs the same
     stations one at a time, same verdict);
  5. criterion: intervention_score = contract_ok x the mean over the six stations of the station score (eleven items,
     integer percentage points, rational arithmetic: subseacell.scoring), reward = clamp(score / anchor / full_at,
     0, 1) (robotics_grader.ratio_rewards against tests/anchor.json).  A failure on the grader's side ends the run as
     a TASK ERROR (`fail`: a zero says the agent failed, a task error says the grading broke); a program that
     crashes, hangs or misbehaves ends its own station, which is scored as it stands.
Untouched sandbox: no intervene.py -> contract_ok 0 -> 0.  Oracle: the token + reference_run -> the recorded reference
value -> 1.
Verdict metrics are deterministic (no wall-clock statistic: those go to the record).  Records (root-only, outside
/opt/ale which the engine clears before it keeps a sandbox): out/stations.json (every station's result and attempt),
out/plan.json (the slot plan and timeline), out/events_<tag>/ (each station's event log), logs/<tag>.log (the
program's stderr), logs/<tag>.cell.log (the cell's) under /opt/arb-verify ($ARB_RECORD_ROOT).
"""

import ast
import json
import os
import pwd
import shutil
import stat as statmod
import subprocess
import sys
import threading
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

FREEZER = [None]


def log(msg):
    print(msg, file=sys.stderr, flush=True)


def _early_lockdown():
    out = {"agent_processes_killed": 0.0, "agent_processes_left": 0.0, "stage_locked": 0.0, "stage_parent_locked": 0.0,
           "stage_lock_latency_ms": -1.0, "submission_late_writes": 0.0, "late_writes_quarantined": 0.0,
           "late_write_gate_active": 0.0, "submission_snapshot": 0.0, "saved": None, "parent": None, "t_ref": None,
           "late": [], "reasons": []}
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
            log("[grader] %d process(es) of the agent account (the image's respawned desktop) outlived the kill at "
                "verify start: frozen until the verdict" % left)
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
import math  # noqa: E402
from fractions import Fraction  # noqa: E402

import layout  # noqa: E402
import slots  # noqa: E402
from cellkit import netaudit, nonet  # noqa: E402

HIDDEN = layout.HIDDEN_DIR
RECORD_ROOT = os.environ.get("ARB_RECORD_ROOT") or ("/opt/arb-verify" if os.getuid() == 0 else os.path.join(STAGE_DIR, "_record"))
LOGS = os.path.join(RECORD_ROOT, "logs")
OUT = os.path.join(RECORD_ROOT, "out")
JOBS = os.path.join(RECORD_ROOT, "jobs")
PROGRAM = "intervene"
CRITERIA = ["intervention_score"]
GATES = ["contract_ok"]
ITEMS = ("M1", "M2", "H", "I1", "I2", "I3", "D1", "D2", "S1", "S2", "R")
SUSPICIOUS = ("/opt/ale", "ALE_STAGE_DIR", "ALE_VERIFICATION", "hidden/sites", "hidden/stations", "oracle_token",
              "reference_run", "get_state_privileged", "ALE_VERIFY", "/opt/ale/tests", ".ale-solution", "/opt/arb-verify")
RERUNNABLE_REPEAT_OK = ("physics_fault", "internal_error")   # repeated with the same program: scored as it stands


def record_dir(path):
    """`path` under the record root, created on demand; never a scoring input."""
    os.makedirs(RECORD_ROOT, exist_ok=True)
    if os.getuid() == 0:
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


def _num(v, default=0.0):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return float(default)
    return f if math.isfinite(f) else float(default)


def _record_value(o):
    if hasattr(o, "tolist"):
        return o.tolist()
    if isinstance(o, (set, frozenset, tuple)):
        return list(o)
    return str(o)


# ============================================================================================ inputs of the grader
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


def hidden_stations(n):
    """The fixed hidden set: [{"station", "name", "path", "sha256"}] for the `n` stations pinned in
    hidden/stations.json, each package checked against its sha256.  Anything short, missing or different is a task
    error."""
    p = os.path.join(HIDDEN, "stations.json")
    try:
        doc = load_json(p)
    except (OSError, ValueError) as e:
        fail("hidden/stations.json is unreadable: %s" % e)
    st = doc.get("stations") if isinstance(doc, dict) else None
    if not isinstance(st, list) or len(st) < n:
        fail("hidden/stations.json must pin the %d hidden stations" % n)
    out = []
    for i, e in enumerate(st[:n]):
        if not isinstance(e, dict):
            fail("hidden/stations.json: station %d is not an object" % (i + 1))
        rel = str(e.get("archive") or "")
        path = os.path.normpath(os.path.join(HIDDEN, rel))
        if not rel or not path.startswith(HIDDEN + os.sep) or os.path.islink(path) or not os.path.isfile(path):
            fail("hidden/stations.json: station %d's package %r is not a regular file under hidden/" % (i + 1, rel))
        want = str(e.get("sha256") or "").strip().lower()
        if len(want) != 64 or sha256_file(path) != want:
            fail("hidden/stations.json: the package of station %d (%s) does not match its recorded sha256" % (i + 1, rel))
        out.append({"station": i + 1, "name": os.path.basename(rel), "path": path, "sha256": want})
    return doc.get("set"), out


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
    if total > float(rg.get("max_submission_mb", 60)) * 1024 * 1024:
        problems.append("submission larger than %s MB" % rg.get("max_submission_mb", 60))
    return problems, suspicious


def library_digests():
    """sha256 of every .py file of the shipped client library (a submission may carry copies of it)."""
    out = set()
    d = os.path.join(ROOT_GIVEN, "lib", "subseacell")
    for dp, _dn, fns in os.walk(d):
        for fn in fns:
            if fn.endswith(".py"):
                out.add(sha256_file(os.path.join(dp, fn)))
    return out


def token_state():
    """(privileged, mismatch): the submission's .oracle_token against hidden/oracle_token.sha256."""
    p = os.path.join(program_source(), ".oracle_token")
    want_p = os.path.join(HIDDEN, "oracle_token.sha256")
    if not os.path.isfile(p) or os.path.islink(p) or not os.path.isfile(want_p):
        return False, False
    try:
        tok = open(p, "rb").read().strip()
        want = open(want_p).read().strip().lower()
    except OSError:
        return False, False
    got = hashlib.sha256(tok).hexdigest()
    return (got == want), (got != want)


# ============================================================================================ the runner accounts
class Accounts(object):
    """One runner account per slot (cellkit.runner.runner_account for each name of grader_config `runner_accounts`):
    every process of a station's program runs as its slot's account, and the clean-up after a station (kill every
    process of the uid, sweep its files) touches that slot only, never a station still running in another slot.
    Fewer existing accounts than slots: the slot count shrinks to the number that exist (at least one, falling back
    to `nobody`, then the agent account, as cellkit does); not root: the programs run as this process's account."""

    def __init__(self, cfg):
        self.names = [str(n) for n in cfg.get("runner_accounts") or []]
        self.accounts = []
        if os.getuid() == 0:
            for n in self.names:
                try:
                    pw = pwd.getpwnam(n)
                except KeyError:
                    continue
                if pw.pw_uid != 0:
                    self.accounts.append((n, pw.pw_uid, pw.pw_gid))
            if not self.accounts:
                a = runner.runner_account(cfg.get("fallback_runner_account", "nobody"), AGENT_USER)
                if a:
                    self.accounts.append(a)
        self.root = os.getuid() == 0

    def slot(self, k):
        return self.accounts[k] if self.accounts else None

    def n(self):
        return len(self.accounts) if self.root else 1


# ============================================================================================ the gates
def contract_check(cfg, acct, cpu):
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
    name, uid, gid = acct if acct else (None, None, None)
    root = os.path.join(RUNS_ROOT, "import_check")
    if os.path.isdir(root):
        shutil.rmtree(root, ignore_errors=True)
    dirs = runner.prepare_run_dir(root, src, {"subseacell": os.path.join(ROOT_GIVEN, "lib", "subseacell")},
                                  owner=(uid, gid) if acct else None)
    before = {r: runner.snapshot_tree(r, (dirs["root"], RUNS_ROOT)) for r in WATCH_ROOTS} if acct else None
    code = ("import sys, importlib; sys.path.insert(0, sys.argv[1]); sys.path.insert(0, sys.argv[2]); "
            "m = importlib.import_module('%s'); assert callable(m.solve)" % PROGRAM)
    wall_s = float(cfg.get("import_check_s", 120.0))
    argv = runner.wrap_limits([sys.executable] + runner.PYTHON_FLAGS + ["-c", code, dirs["lib"], dirs["program"]],
                              cpu_s=wall_s, user=name, nonet=True)
    argv = slots.pin_argv(argv, [cpu] if cpu is not None else [], lock=bool(cfg.get("lock_program_affinity", True)))
    env = layout.program_env(runner.child_env(dirs["home"], dirs["tmp"], dirs["lib"], ROOT_GIVEN, threads=1))
    try:
        r = subprocess.run(argv, cwd=dirs["program"], env=env, capture_output=True, text=True, timeout=wall_s,
                           start_new_session=True)
        ok = r.returncode == 0
        why = "ok" if ok else "importing %s.py failed: " % PROGRAM + (r.stderr.strip().splitlines() or ["?"])[-1][:300]
    except subprocess.TimeoutExpired:
        ok, why = False, "importing %s.py took more than %d s" % (PROGRAM, wall_s)
    finally:
        _cleanup_account(acct, before, dirs)
    return ok, why, suspicious, []


def _cleanup_account(acct, before, dirs):
    if acct:
        lockdown.stop_and_kill_uid(acct[1], grace_s=8.0)
        for r in WATCH_ROOTS:
            after = runner.snapshot_tree(r, (dirs["root"], RUNS_ROOT))
            runner.sweep_new_entries(r, before[r], after, acct[1], (dirs["root"], RUNS_ROOT))
    shutil.rmtree(dirs["root"], ignore_errors=True)


def _settle_desktop(max_s=60.0, quiet_s=8.0, poll_s=0.25):
    """Wait, before the first station, until the frozen desktop is quiet (the image's supervisord respawns the agent
    account's desktop after the verify-start kill).  Returns (seconds waited, counter increments meanwhile)."""
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


# ============================================================================================ the stations
def _fault_injection():
    """Maintainers' self-test of the re-run path (never set by the engine): ARB_TEST_INJECT_CELL_FAULT = "<station>" or
    "<station>@<sim seconds>" makes that station's FIRST attempt raise the physics fault (station.py)."""
    v = os.environ.get("ARB_TEST_INJECT_CELL_FAULT", "").strip()
    if not v:
        return None
    st, _, at = v.partition("@")
    return int(st), float(at or 5.0)


class StationRunner(object):
    def __init__(self, cfg, accounts, privileged, plan, t_deadline):
        self.cfg, self.accounts, self.privileged, self.plan = cfg, accounts, privileged, plan
        self.t_deadline = t_deadline
        self.inject = _fault_injection()
        self.watchdog_s = float(cfg["station_watchdog_s"])
        self.station_cfg = {k: cfg[k] for k in ("budgets", "program_time_cap_s", "silence_s", "program_cpu_cap_s",
                                                "socket_sample_s", "lock_program_affinity") if k in cfg}
        # the cell's own environment (the image's /opt/cellenv, root and the practice account only); this process and
        # the program use the program's environment (sys.executable)
        cp = str(cfg.get("cell_python") or "")
        self.cell_python = cp if cp and os.access(cp, os.X_OK) else sys.executable

    def run(self, station, slot):
        """One attempt of one station on `slot` (called in a worker thread of the scheduler): the station's cell
        process (station.py), pinned to the slot's cell CPU, and its result file."""
        k = station["station"]
        att = self._attempt_of(k)
        tag = "ep%02d%s" % (k, "r" if att else "")
        job = {"tag": tag, "station": k, "attempt": att, "site_path": station["path"], "site_name": station["name"],
               "privileged": self.privileged, "config": self.station_cfg, "account": self.accounts.slot(slot.index),
               "prog_cpu": slot.prog_cpu, "program_dir": program_source(), "lib_src": os.path.join(ROOT_GIVEN, "lib", "subseacell"),
               "given_dir": ROOT_GIVEN, "agent_home": HOME, "runs_root": RUNS_ROOT, "logs_dir": record_dir(LOGS),
               "out_dir": record_dir(OUT), "slot": slot.index, "program_python": sys.executable}
        if self.inject and self.inject[0] == k and att == 0:
            job["inject_fault_at_s"] = self.inject[1]
        jd = record_dir(JOBS)
        jp, rp = os.path.join(jd, tag + ".job.json"), os.path.join(jd, tag + ".result.json")
        if os.path.exists(rp):
            os.remove(rp)
        with open(jp, "w") as f:
            json.dump(job, f, indent=1)
        argv = slots.pin_argv([self.cell_python, "-s", layout.STATION_SCRIPT, jp, rp],
                              [slot.cell_cpu] if slot.cell_cpu is not None else [])
        env = dict(os.environ)
        env.update(layout.cell_env())
        t0 = time.time()
        timeout = max(60.0, min(self.watchdog_s, self.t_deadline - t0))
        with open(os.path.join(LOGS, tag + ".cell.log"), "ab") as lf:
            p = subprocess.Popen(argv, cwd=STAGE_DIR, env=env, stdin=subprocess.DEVNULL, stdout=lf, stderr=lf,
                                 start_new_session=True)
            try:
                rc = p.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(p.pid, 9)
                except OSError:
                    pass
                p.wait()
                acct = self.accounts.slot(slot.index)
                if acct:
                    lockdown.stop_and_kill_uid(acct[1], grace_s=8.0)
                return {"tag": tag, "station": k, "attempt": att,
                        "harness_error": "station %d past its watchdog (%.0f s)" % (k, timeout)}
        try:
            res = load_json(rp)
        except (OSError, ValueError):
            return {"tag": tag, "station": k, "attempt": att,
                    "harness_error": "station %d's worker left no result (exit %s)" % (k, rc)}
        res["worker_exit"] = rc
        res["slot"] = slot.index
        res["cell_cpu"], res["prog_cpu"] = slot.cell_cpu, slot.prog_cpu
        res["t_start"], res["t_end"] = t0, time.time()
        return res

    def _attempt_of(self, k):
        with _ATT_LOCK:
            n = _ATTEMPTS.get(k, 0)
            _ATTEMPTS[k] = n + 1
            return n


_ATT_LOCK = threading.Lock()
_ATTEMPTS = {}


def _rerun_if(station, res):
    return isinstance(res, dict) and not res.get("harness_error") and bool(res.get("cell_failed"))


# ============================================================================================ the verdict
def write_verdict(rewards, metrics):
    """One gated `reward` in the benchmark kit's envelope (robotics_grader.ratio_rewards against the recorded reference
    value); the gate, the criterion and every statistic go to `metrics` (finite numbers only)."""
    from robotics_grader import ratio_rewards, read_anchor
    from robotics_grader import write_verdict as kit_write_verdict
    overall = float(rewards["intervention_score"])
    anchor, meta = read_anchor()
    extra = {}
    for k, v in metrics.items():
        try:
            fv = float(v)
        except (TypeError, ValueError):
            continue
        if math.isfinite(fv):
            extra[k] = fv
    extra.update({k: float(rewards[k]) for k in GATES + CRITERIA})
    verdict = ratio_rewards(overall, anchor, higher_is_better=meta.get("direction", "higher") == "higher",
                            metric_name=meta.get("metric", "intervention_score"), extra=extra,
                            full_at=meta.get("full_at", 1.0))
    text = ("contract_ok %.0f (the program imports, exposes solve, passes the route gate and the network audit and steps "
            "the cell; no late write, no surviving process); intervention_score %.4f (the mean over the six hidden "
            "stations of the station score: marker M1 0.03 / M2 0.12, survey and no harm H 0.12, isolation I1 0.08 / I2 "
            "0.22 / I3 0.08, dropped object D1 0.07 / D2 0.05, hot stab S1 0.04 / S2 0.14, recovery R 0.05; a critical "
            "error zeroes I, S and H of its station)") % (rewards["contract_ok"], rewards["intervention_score"])
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
        return verdict
    # a maintainer's run outside the engine: the kit's envelope at $ROBOTICS_GRADER_VERDICT_PATH, else in the stage
    kit_write_verdict(verdict, path=None if os.environ.get("ROBOTICS_GRADER_VERDICT_PATH")
                      else os.path.join(STAGE_DIR, "verdict.local.json"))
    return verdict


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
    metrics = {"grader_exception": 0.0, "stations": float(cfg.get("stations", 6)), "root_payload_ok": 0.0,
               "agent_deprivileged": 0.0, "privileged_mode": 0.0, "oracle_token_mismatch": 0.0,
               "runner_processes_left": 0.0, "runner_unclean_runs": 0.0, "suspicious_lines": 0.0,
               "remote_sockets_seen": 0.0, "network_imports": 0.0, "illegal_call_count": 0.0}
    for k in ("stage_locked", "stage_parent_locked", "agent_processes_left", "submission_late_writes",
              "late_writes_quarantined", "late_write_gate_active", "submission_snapshot"):
        metrics[k] = float(EARLY.get(k, 0.0))
    record = {"early": {k: EARLY.get(k) for k in ("agent_processes_killed", "agent_processes_left", "stage_lock_latency_ms",
                                                   "late_writes_quarantined", "submission_late_writes")}}
    try:
        try:
            _main_locked(cfg, rewards, metrics, record)
        except SystemExit:
            raise
        except Exception:  # noqa: BLE001 - a grader defect must still say so
            log("[grader] EXCEPTION\n" + traceback.format_exc())
            metrics["grader_exception"] = 1.0
            raise
    finally:
        _restore_modes()
        if FREEZER[0] is not None:
            FREEZER[0].stop()


def _main_locked(cfg, rewards, metrics, record):
    t_start = time.time()
    t_deadline = t_start + float(cfg["timeout_sec"]) - float(cfg.get("deadline_margin_s", 300.0))
    expected = load_json(os.path.join(STAGE_DIR, "given_hashes.json"))
    ok_root, problems = root_payload_ok(expected)
    metrics["root_payload_ok"] = 1.0 if ok_root else 0.0
    if not ok_root:
        fail("the root-owned copy of the givens at %s is unusable: %s" % (ROOT_PAYLOAD, "; ".join(problems[:5])))
    os.makedirs(RUNS_ROOT, exist_ok=True)
    os.chmod(RUNS_ROOT, 0o755)
    acc = Accounts(cfg)
    if FREEZER[0] is not None and any(a[1] == FREEZER[0].uid for a in acc.accounts):
        FREEZER[0].stop()        # the runner fell back to the agent account, whose programs must run
        log("[grader] runner account = the agent account: the verify-stage freezer is stopped")
    metrics["agent_deprivileged"] = 1.0 if (acc.accounts or not acc.root) else 0.0
    n = int(cfg.get("stations", 6))
    k_max = int(cfg.get("slots", 3))
    env_k = os.environ.get("ARB_VERIFY_MAX_SLOTS", "").strip()
    if env_k:
        k_max = max(1, min(k_max, int(env_k)))
    k_max = max(1, min(k_max, acc.n()))
    plan = slots.plan_slots(n_jobs=n, k_max=k_max, policy=str(cfg.get("pin_policy", "auto")))
    record["plan"] = plan.as_dict()
    log("[grader] slots: %d (%s) %s; parent CPUs %s; runner accounts %s" % (
        len(plan.slots), plan.policy, [(s.cell_cpu, s.prog_cpu) for s in plan.slots], list(plan.parent_cpus),
        [a[0] for a in acc.accounts] or "self"))
    privileged, mismatch = token_state()
    metrics["privileged_mode"] = 1.0 if privileged else 0.0
    metrics["oracle_token_mismatch"] = 1.0 if mismatch else 0.0
    ok, why, suspicious, net = contract_check(cfg, acc.slot(0), plan.slots[0].prog_cpu)
    if ok and EARLY.get("reasons"):
        ok, why = False, "; ".join(EARLY["reasons"])
    metrics["suspicious_lines"] = float(suspicious)
    metrics["network_imports"] = float(len(net))
    record["contract"] = {"ok": ok, "why": why}
    log("[grader] contract_ok=%s (%s); privileged %s" % (ok, why, privileged))
    set_name, stations = hidden_stations(n)
    record["hidden_set"] = {"set": set_name, "stations": [{k: s[k] for k in ("station", "name", "sha256")} for s in stations]}
    if not ok:
        rewards["contract_ok"] = 0.0
        _write_record(record, [])
        write_verdict(rewards, metrics)
        return
    settle_s, settle_seen = _settle_desktop()
    record["desktop_settle_s"], record["net_settle_events"] = settle_s, settle_seen
    a0 = acc.slot(0)
    nn_ok, nn_why = nonet.selftest(wrap=lambda a: runner.wrap_limits(a, cpu_s=60, user=a0[0] if a0 else None),
                                   env=layout.program_env(runner.child_env("/tmp", "/tmp", "/nonexistent", ROOT_GIVEN)))
    metrics["nonet_selftest_ok"] = 1.0 if nn_ok else 0.0
    if not nn_ok:
        fail("the no-network filter for the program does not work here: " + nn_why)
    sched = slots.SlotScheduler(plan, mode=str(cfg.get("slot_mode", "queue")))
    srun = StationRunner(cfg, acc, privileged, plan, t_deadline)
    if srun.inject:
        log("[grader] MAINTAINERS' SELF-TEST: a cell fault is injected into station %d's first attempt at %.1f s" % srun.inject)
    metrics["fault_injected"] = 1.0 if srun.inject else 0.0
    tmp_saved = layout.close_shared_tmp(log=log)          # no file passes between stations that run at once (P24-5)
    record["shared_tmp_closed"] = [d for d, _ in tmp_saved]
    try:
        results = sched.map(lambda st, slot: srun.run(st, slot), stations, rerun_if=_rerun_if,
                            max_reruns=int(cfg.get("single_station_reruns", 1)), deadline=t_deadline,
                            rerun_min_s=float(cfg.get("rerun_min_s", cfg["station_watchdog_s"])))
    finally:
        layout.reopen_shared_tmp(tmp_saved)
    record["timeline"] = sched.timeline()
    attempts = {}
    for r in sched.records:
        attempts.setdefault(r.index + 1, []).append(r.attempt)
    final = []
    for st, res in zip(stations, results):
        if not isinstance(res, dict):
            fail("station %d: no result" % st["station"])
        if res.get("harness_error"):
            _write_record(record, results)
            fail("station %d: %s" % (st["station"], res["harness_error"]))
        failed = list(res.get("cell_failed") or [])
        rerun = res.get("rerun_of_attempt") is not None
        if failed:
            hard = [f for f in failed if f not in RERUNNABLE_REPEAT_OK]
            if hard:
                _write_record(record, results)
                fail("station %d: the cell failed (%s)%s" % (st["station"], ", ".join(failed),
                                                             " in two runs" if rerun else " and could not be run again"))
            log("[grader] station %d: the cell failed (%s)%s: scored as it stands" % (
                st["station"], ", ".join(failed), " again in its second run" if rerun else " (no re-run left)"))
        if "points" not in res:
            _write_record(record, results)
            fail("station %d: no score" % st["station"])
        final.append(res)
    _write_record(record, results)
    metrics["stations_rerun"] = float(sum(1 for r in final if r.get("rerun_of_attempt") is not None))
    metrics["stations_cell_failed_scored"] = float(sum(1 for r in final if r.get("cell_failed")))
    for i, r in enumerate(final):
        metrics["station_rerun.s%d" % (i + 1)] = 1.0 if r.get("rerun_of_attempt") is not None else 0.0
    step_calls = sum(int(r.get("step_calls") or 0) for r in final)
    illegal = sum(int((r.get("record") or {}).get("illegal_call_count") or 0) for r in final)
    remote = max([int(r.get("remote_sockets_seen") or 0) for r in final] + [0])
    left = sum(int(r.get("runner_processes_left") or 0) for r in final)
    unclean = sum(1 for r in final if int(r.get("runner_processes_left") or 0) > 0)
    steps_gate = 1.0 if step_calls > 0 else 0.0
    clean_gate = 1.0 if unclean == 0 else 0.0
    net_gate = 1.0 if remote <= 0 else 0.0
    gate = steps_gate * clean_gate * net_gate
    rewards["contract_ok"] = gate
    metrics.update({"step_calls": float(step_calls), "illegal_call_count": float(illegal),
                    "remote_sockets_seen": float(remote), "runner_processes_left": float(left),
                    "runner_unclean_runs": float(unclean)})
    if not net_gate:
        log("[grader] contract_ok 0: the program held a remote inet socket during a station (%d seen)" % remote)
    pts = [Fraction(int(r["points"]), 100) for r in final]
    for r in final:
        s = float(r["score"])
        if not (math.isfinite(s) and 0.0 <= s <= 1.0) or abs(s - int(r["points"]) / 100.0) > 1e-12:
            fail("station %d: the score %r is not a finite number in [0, 1] matching its points" % (r["station"], r.get("score")))
    mean = sum(pts, Fraction(0)) / len(pts)
    raw = mean if gate else Fraction(0)
    rewards["intervention_score"] = float(raw)
    metrics["intervention_points_mean"] = float(mean * 100)
    for i, r in enumerate(final):
        metrics["station_score.s%d" % (i + 1)] = float(r["score"])
        rec = r.get("record") or {}
        metrics["station_sim_s.s%d" % (i + 1)] = _num(rec.get("sim_time_s"))
        metrics["station_critical_errors.s%d" % (i + 1)] = float(len((r.get("details") or {}).get("critical_errors") or []))
    for it in ITEMS:
        metrics["item_rate." + it] = float(sum(1 for r in final if (r.get("items") or {}).get(it))) / len(final)
    reasons = [str((r.get("record") or {}).get("done_reason") or r.get("end_reason")) for r in final]
    for rsn in sorted(set(reasons)):
        metrics["end_rate." + rsn] = float(sum(1 for x in reasons if x == rsn)) / len(reasons)
    for key in ("640x480", "320x240"):
        metrics["frames_mean." + key] = float(sum(_num(((r.get("record") or {}).get("frames") or {}).get(key)) for r in final)) / len(final)
    metrics["pings_mean"] = float(sum(_num((r.get("record") or {}).get("pings")) for r in final)) / len(final)
    metrics["protective_stops_total"] = float(sum(_num((r.get("details") or {}).get("protective_stops")) for r in final))
    metrics["cell_internal_errors"] = float(sum(int(_num((r.get("record") or {}).get("internal_errors"))) for r in final))
    metrics["wire_errors"] = float(sum(1 for r in final if r.get("wire_error")))
    verdict = write_verdict(rewards, metrics)
    record["verdict"] = verdict
    record["verify_wall_s"] = time.time() - t_start
    _write_record(record, results)
    shutil.rmtree(RUNS_ROOT, ignore_errors=True)
    log("[grader] rewards %s reward %.4f stations %s (verify wall %.0f s, %d slot(s))" % (
        {k: round(v, 4) for k, v in rewards.items()}, float(verdict["rewards"]["reward"]),
        [float(r["score"]) for r in final], record["verify_wall_s"], len(plan.slots)))


def _write_record(record, results):
    try:
        record_dir(OUT)
        with open(os.path.join(OUT, "plan.json"), "w") as f:
            json.dump(record, f, indent=1, default=_record_value)
        with open(os.path.join(OUT, "stations.json"), "w") as f:
            json.dump(results, f, indent=1, default=_record_value)
    except Exception:  # noqa: BLE001
        log("[grader] the record was not written\n" + traceback.format_exc())


if __name__ == "__main__":
    main()
