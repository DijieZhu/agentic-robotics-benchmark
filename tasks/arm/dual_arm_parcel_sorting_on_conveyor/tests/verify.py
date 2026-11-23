"""Grader of dual_arm_parcel_sorting_on_conveyor (the label-facing station of a parcel sorting line: a fixed dual-arm
robot turns each arriving parcel's shipping label up before the parcel reaches the scan line; camera-driven): runs as
ROOT in the verify stage. Verification is separate (task.toml): this runs in a fresh verifier sandbox from the task
image, with only the agent's submission restored into it; tests/test.sh built the root-owned copy of the givens
(/opt/arb) before this starts.

The agent left /home/user/submission/sort.py (solve(cell): the station's program; it sees the station through its
four cameras and the robot's own joint, tool and gripper state, and commands, per arm, joint position targets or a
tool-centre-point pose that the cell's controller solves, and gripper closures). This grader:
  0. FIRST, stdlib only (`_early_lockdown`; tests/test.sh closed the stage folder already): records the upload time,
     clears the agent's account with the benchmark kit (robotics_grader.clear_agent_account: every supervisord program
     of the image switched off and supervisord reloaded, so that the account's desktop session is not started again,
     and every process of the account that this process did not start killed until none comes back), then kills
     whatever of the account is left and keeps every process of it stopped until the verdict is written
     (cellkit.verify_freeze: a second layer that finds nothing once the clearing holds), locks the stage, snapshots the submission root-owned, gates late writes, quarantines late
     entries elsewhere (written for a verify in the agent's own sandbox; in the separate verifier there is nothing of the
     agent's to stop or set aside);
  1. checks the root-owned copy of the givens at /opt/arb against given_hashes.json (the client library handed to the
     programs and ARB_GIVEN_DIR come from it; a mismatch is a task error);
  2. gates: contract_ok = sort.py is a regular file that parses and defines solve, the route gate (no weight files, no
     training calls, size cap), every .py file of the submission readable by the parser, the network audit (static: no
     networking import, no computed import, no computed code, no program start; runtime: every process of the
     program -- the import check and the shifts -- runs under the no-network seccomp filter (cellkit.nonet: no inet
     socket at all), and the runner account's remote inet sockets are sampled; the namespace's outbound counters are
     statistics only), the module imports under the run regime (120 s, 120 CPU-s), >= 1 step call, no late write, no
     surviving process; before the first shift it waits until the frozen desktop is quiet (`_settle_desktop`) and
     proves the filter works (`nonet.selftest`, a failure is a task error);
  3. the oracle token: submission/.oracle_token whose sha256 equals hidden/oracle_token.sha256 switches the cells to
     the privileged mode (the calls get_state_privileged and reference_run; stat privileged_mode); a wrong token is
     ignored (stat oracle_token_mismatch). The token also picks the set (`graded_set`): the privileged oracle is run on
     the RATED set (hidden/seeds.json: the station's rated setting, parcels at stated times, the duty of the score
     block), every other submission on the LINE set (hidden/line/seeds.json: the line setting, one parcel of each
     shift's list at a time, fed on settle, every parcel in the duty). Nothing else chooses the set; a package whose
     setting is not its set's is a task error (README.md, section "the two sets");
  4. takes the hidden shifts pinned in the set's seeds.json (the fixed hidden set; each a stored shift package
     <set>/shifts/<name>.json whose sha256 the file pins: a missing, short or mismatched set is a task error) and runs
     the agent's program once per shift, one shift after another, on a fresh cell (parcelcell.server.ParcelCellServer,
     mode hidden: the shift's belt speed and its parcels' arrival times, sizes, kinds, masses, frictions and label
     faces come from the package; camera-only API) as `nobody` from a read-only copy under an opaque run tag (the
     package's name and seed never reach a path the program sees), in a fresh process with oom_score_adj 1000
     (`choom`: when the sandbox's memory runs out the kernel stops one of the program's processes, never the grader),
     under the shift's budgets (cellkit.wire: program time = the wall clock minus the cell's simulation and rendering,
     budget = factor x the shift's duration + an allowance; one silence of at most silence_s; CPU = cpu_cap_factor x
     the budget). A shift in which the cell failed (an internal error while the shift was live, a physics fault, a
     frame it could not render) is run once more from scratch and the second run counts (`_episode_with_retry`; a
     render failure that repeats is a task error; at most one shift per verify is run again, a second shift in which
     the cell failed is a task error);
  5. criterion: facing_score = the mean over the hidden shifts of the cell's shift score (parcelcell.episode: each
     parcel's class is fixed at arrival: W = in the station's duty with its label not up, U = arrived label up, X =
     outside the duty with its label not up; credit = #(W parcels that cross the scan line readable) - #(U parcels
     that do not) - #(parcels lost, any class); shift score = max(0, credit) / #(W parcels); on the line set every
     parcel is in the duty, a parcel of a shift's list that is not done when the shift ends costs nothing, and #W
     counts the whole list), times the gates. The reward: on the line set the absolute facing_score (anchor.json
     "line": value 1.0, full_at 1.0); on the rated set the facing_score against the recorded reference value
     (anchor.json value, full_at). A failure on the grader's side of a
     shift (the cell's build, the program's launch, the wire, the cleanup, the scoring, a score that is not a finite
     number in [0, 1]) ends the run as a TASK ERROR (`fail`: a zero says the agent failed, a task error says the
     grading broke); a program that crashes, hangs or misbehaves ends its own shift, which is scored as it stands.
Untouched sandbox: no sort.py -> contract_ok 0 -> 0. Oracle (the rated set): the token + reference_run (the reference program inside
the cell) -> the recorded reference value -> 1.
Records: out/shifts.json (every shift's record), out/log_lines.txt and logs/<tag>.log (the program's stderr) under the
root-only /opt/arb-verify (outside /opt/ale, which the engine clears before it keeps a sandbox).
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
    out = {"agent_account_cleared": 0.0, "agent_account_programs_disabled": 0.0, "agent_account_clear_killed": 0.0,
           "agent_processes_killed": 0.0, "agent_processes_left": 0.0, "stage_locked": 0.0, "stage_parent_locked": 0.0,
           "stage_lock_latency_ms": -1.0, "submission_late_writes": 0.0, "late_writes_quarantined": 0.0, "late_write_gate_active": 0.0,
           "submission_snapshot": 0.0, "submission_snapshot_skipped": 0.0, "saved": None, "parent": None, "t_ref": None,
           "late": [], "reasons": []}
    if os.getuid() != 0:
        return out
    t_ref = lockdown.stage_upload_time(STAGE_DIR)
    out["t_ref"] = t_ref
    # The benchmark kit's clearing of the agent's account, before anything runs as an account (this grader starts the
    # programs with a launcher of its own, cellkit.runner, so it asks for the clearing itself, as the kit says): once
    # per sandbox, a few seconds. The verify stage needs nothing it removes: the cell renders offscreen (classic depth
    # on OSMesa, Filament colour on Vulkan through lavapipe, no display, no D-Bus), the program under test runs as
    # `nobody` (or as the agent's account, started by this process and so never touched by the clearing), and the
    # engine reads the verdict through its own guest service, which supervisord does not run.
    try:
        from robotics_grader import clear_agent_account
        rep = clear_agent_account(user=AGENT_USER)
    except Exception as exc:  # noqa: BLE001 - the kill and the freezer below still hold the account
        rep = {"cleared": False, "skipped": "the clearing raised %s" % type(exc).__name__}
    out["agent_account_cleared"] = 1.0 if rep.get("cleared") else 0.0
    out["agent_account_programs_disabled"] = float(len(rep.get("programs_disabled") or []))
    out["agent_account_clear_killed"] = float(rep.get("killed") or 0)
    if rep.get("skipped"):
        log("[grader] the agent account was not cleared: %s" % rep["skipped"])
    try:
        uid = pwd.getpwnam(AGENT_USER).pw_uid
    except KeyError:
        uid = None
    if uid is not None and uid != 0:
        killed, left = lockdown.stop_and_kill_uid(uid, grace_s=8.0)
        out["agent_processes_killed"], out["agent_processes_left"] = float(killed), float(left)
        # a survivor is a statistic, not a contract failure: verification is separate (task.toml), nothing of the
        # agent's runs in this sandbox, and the image's supervisord respawns the account's desktop as fast as it is
        # killed, so a process counted here is the image's own. The freezer below keeps every process of the account
        # stopped until the verdict is written, whatever the count.
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
        # the snapshot is where symbolic links and special files are dropped (the per-run copies of the snapshot see
        # none, so their count alone read 0)
        out["submission_snapshot_skipped"] = float(lockdown.snapshot_dir(SUBMISSION, SNAPSHOT) or 0)
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

import numpy as np  # noqa: E402

from cellkit import netaudit, nonet, wire  # noqa: E402
from cellkit.seeds import load_pinned  # noqa: E402
from cellkit.seeds import seed_rng  # noqa: E402
from parcelcell.client import launcher_argv  # noqa: E402
from parcelcell.server import ParcelCellServer  # noqa: E402

HIDDEN = os.path.join(STAGE_DIR, "hidden")
HIDDEN_LINE = os.path.join(HIDDEN, "line")


def graded_set(privileged):
    """(set name, its folder) for a submission: the rated set for the privileged oracle (its token matched), the line
    set for every other submission. The only input is the token check (`token_state`)."""
    return ("rated", HIDDEN) if privileged is True else ("line", HIDDEN_LINE)
# The maintainers' record (the program's stderr, shifts.json) lives outside /opt/ale: the engine deletes its stage
# folders before it keeps a sandbox, and the record is read from a kept one. Root-only (0700). A host-side run as
# another account keeps it in the stage folder.
RECORD_ROOT = os.environ.get("ARB_RECORD_ROOT") or ("/opt/arb-verify" if os.getuid() == 0 else os.path.join(STAGE_DIR, "_record"))
LOGS = os.path.join(RECORD_ROOT, "logs")
OUT = os.path.join(RECORD_ROOT, "out")
PROGRAM = "sort"
CRITERIA = ["facing_score"]
GATES = ["contract_ok"]
IMPORT_WALL_S = 120          # the import check's limits (instruction.md, cell_api.md): wall seconds and CPU seconds
IMPORT_CPU_S = 120
# the kernel's first choice when the sandbox's memory runs out is the program, never the grader: the import check's
# and every shift's program command start with `choom -n 1000 --` (util-linux, in the image); the program's process and
# every process it starts get oom_score_adj 1000. The prefix stands innermost (cellkit.runner.wrap_limits puts the
# account switch, the CPU limit and the no-network filter around it), so choom runs as the program's account and execs
# the program in place.
OOM_FIRST = ["choom", "-n", "1000", "--"]
# what ast.parse raises for a source it cannot read: beyond CPython's limits it is RecursionError or MemoryError, not
# SyntaxError; ValueError for other malformed input. A submission holding such a file fails the contract (closed); it
# never stops the grader.
PARSE_ERRORS = (SyntaxError, RecursionError, MemoryError, ValueError)
SUSPICIOUS = ("/opt/ale", "ALE_STAGE_DIR", "ALE_VERIFICATION", "hidden/seeds", "hidden/shifts", "oracle_token", "reference_run",
              "get_state_privileged", "ALE_VERIFY", "/opt/ale/tests", ".ale-solution", "/opt/arb-verify")
# what a shift's record (ParcelCellServer.finalize) must carry for the grader; every other key is a statistic
SUMMARY_REQUIRED = ("shift_score", "step_calls", "calls", "illegal_calls", "internal_errors", "internal_errors_live",
                    "frames", "done_reason")
# the route gate (grader_config.json `route_gate` overrides these values)
ROUTE_GATE = {"forbidden_calls": ["torch.optim", ".backward(", "jax.grad"],
              "forbidden_extensions": [".pt", ".pth", ".onnx", ".safetensors", ".ckpt", ".h5", ".pb", ".pkl", ".npz"],
              "max_submission_mb": 60}
MAX_RERUNS_PER_VERIFY = 1      # task.toml's verifier timeout is derived for the hidden shifts and one re-run


def record_dir(path):
    """`path` under the record root, created on demand; never a scoring input."""
    os.makedirs(RECORD_ROOT, exist_ok=True)
    if RECORD_ROOT != STAGE_DIR and os.getuid() == 0:
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
    """A finite float of `v`, or `default` (statistics only; never a scoring input)."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return float(default)
    return f if math.isfinite(f) else float(default)


def _dict(v):
    return v if isinstance(v, dict) else {}


def _record_value(o):
    """json.dump fallback for the maintainers' record: numpy arrays and scalars as plain values, anything else as text."""
    if hasattr(o, "tolist"):
        return o.tolist()
    if isinstance(o, (set, frozenset, tuple)):
        return list(o)
    return str(o)


def root_payload_ok(expected):
    problems = []
    owner = 0 if os.getuid() == 0 else os.getuid()
    if not os.path.isdir(ROOT_PAYLOAD):
        return False, ["%s missing (tests/test.sh builds it)" % ROOT_PAYLOAD]
    files = expected.get("files", expected) if isinstance(expected, dict) else {}
    if not files:
        return False, ["given_hashes.json lists no file"]
    for rel, digest in files.items():
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


def hidden_shifts(n, root=None, setting="rated"):
    """The fixed hidden set: [(seed, package path, tag)] for the first `n` shifts pinned in <root>/seeds.json (root:
    the set's folder, hidden/ for the rated set, hidden/line/ for the line set), each package checked against the
    sha256 the file records and its setting against the set's (`setting`: rated = no setting, line = the line
    setting). Anything short, missing or different is a task error. A PLACEHOLDER set (`"placeholder": true`, the
    stand-in until the selection runs) grades every shift it pins."""
    root = HIDDEN if root is None else root
    pinned_path = os.path.join(root, "seeds.json")
    try:
        pinned = load_pinned(pinned_path)
        doc = load_json(pinned_path)
    except (OSError, ValueError) as e:
        fail("%s is unreadable: %s" % (pinned_path, e))
    placeholder = doc.get("placeholder") is True
    if placeholder:
        n = len(doc.get("shifts") or [])
        log("[grader] PLACEHOLDER hidden set: %d sampled shifts stand in for the fixed hidden set" % n)
        if n < 1:
            fail("%s: a placeholder set must pin at least one shift" % pinned_path)
    if not pinned or len(pinned.get("seeds", [])) < n or len(doc.get("shifts") or []) < n:
        fail("%s must pin the %d hidden shifts (the fixed hidden set: seeds and shift packages)" % (pinned_path, n))
    seeds = [int(x) for x in pinned["seeds"][:n]]
    out = []
    for i, (seed, sh) in enumerate(zip(seeds, doc["shifts"][:n])):
        if not isinstance(sh, dict):
            fail("%s: shift %d is not an object" % (pinned_path, i + 1))
        rel = str(sh.get("package") or "")
        path = os.path.normpath(os.path.join(root, rel))
        if not rel or not path.startswith(root + os.sep) or os.path.islink(path) or not os.path.isfile(path):
            fail("%s: shift %d's package %r is not a regular file under its set's folder" % (pinned_path, i + 1, rel))
        if sh.get("seed") is not None and int(sh["seed"]) != seed:
            fail("%s: shift %d's seed differs from the pinned seed list" % (pinned_path, i + 1))
        want = str(sh.get("sha256") or "").strip().lower()
        if len(want) != 64 or sha256_file(path) != want:
            fail("%s: the package of shift %d (%s) does not match its recorded sha256" % (pinned_path, i + 1, rel))
        try:
            pkg_setting = load_json(path).get("setting")
        except (OSError, ValueError):
            pkg_setting = None
            fail("%s: the package of shift %d (%s) is unreadable" % (pinned_path, i + 1, rel))
        name = (pkg_setting or {}).get("name") if isinstance(pkg_setting, dict) else ("rated" if pkg_setting is None else "?")
        if (name or "rated") != setting:
            fail("%s: the package of shift %d (%s) has the setting %r, its set is %r" % (pinned_path, i + 1, rel, name, setting))
        out.append((seed, path, str(sh.get("tag") or os.path.splitext(os.path.basename(rel))[0])))
    return seeds, out, placeholder


def route_gate(cfg, src):
    rg = dict(ROUTE_GATE)
    rg.update(cfg.get("route_gate") or {})
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
    d = os.path.join(ROOT_GIVEN, "lib", "parcelcell")
    for dp, _dn, fns in os.walk(d):
        for fn in fns:
            if fn.endswith(".py"):
                out.add(sha256_file(os.path.join(dp, fn)))
    return out


class Runner(object):
    def __init__(self, cfg):
        self.acct = runner.runner_account(cfg.get("runner_account", "nobody"), AGENT_USER)
        self.name, self.uid, self.gid = self.acct if self.acct else (None, None, None)
        self.lib_dirs = {"parcelcell": os.path.join(ROOT_GIVEN, "lib", "parcelcell")}
        self.cpu_factor = float(cfg.get("cpu_cap_factor", 4.0))
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
        # the program's own environment: no numerics pin (the grader's own processes carry it, tests/test.sh)
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


def _parse_error(e):
    return str(e) if isinstance(e, SyntaxError) else "%s: %s" % (type(e).__name__, e)


def unparsable_files(src):
    """[(relpath, error text)] of the submission's .py files the parser cannot read at all (RecursionError, MemoryError,
    ValueError). The walk, the bytes and their decoding are the network audit's (cellkit.netaudit.scan_imports), and
    this runs before the audit, so that no file the audit reads reaches it unparsed: the audit catches only
    SyntaxError. A SyntaxError is left to the audit's own rule (it skips the file: no program can import it)."""
    out = []
    for dp, dns, fns in os.walk(src):
        dns[:] = sorted(d for d in dns if d != "__pycache__")
        for fn in sorted(fns):
            if not fn.endswith(".py"):
                continue
            p = os.path.join(dp, fn)
            if os.path.islink(p):
                continue
            try:
                data = open(p, "rb").read()
            except OSError:
                continue
            try:
                ast.parse(data.decode("utf-8", "replace"))
            except SyntaxError:
                continue
            except PARSE_ERRORS as e:
                out.append((os.path.relpath(p, src), _parse_error(e)[:200]))
    return out


def contract_check(cfg, rn):
    src = program_source()
    path = os.path.join(src, PROGRAM + ".py")
    if not os.path.isfile(path) or os.path.islink(path):
        return False, "no %s.py (regular file) in the submission" % PROGRAM, 0, []
    try:
        tree = ast.parse(open(path, encoding="utf-8", errors="replace").read())
    except PARSE_ERRORS as e:
        return False, "%s.py does not parse: %s" % (PROGRAM, _parse_error(e)), 0, []
    has_solve = any(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == "solve" for n in ast.walk(tree)) or \
        any(isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "solve" for t in n.targets) for n in ast.walk(tree)) or \
        any(isinstance(n, (ast.Import, ast.ImportFrom)) and any((a.asname or a.name) == "solve" for a in n.names) for n in ast.walk(tree))
    if not has_solve:
        return False, "%s.py defines no solve" % PROGRAM, 0, []
    problems, suspicious = route_gate(cfg, src)
    if problems:
        return False, "route gate: " + "; ".join(problems[:5]), suspicious, []
    bad = unparsable_files(src)
    if bad:
        return False, "a .py file the parser cannot read: %s" % "; ".join("%s (%s)" % x for x in bad[:5]), suspicious, []
    try:
        net = netaudit.scan_imports(src, allow_digests=library_digests())
    except PARSE_ERRORS as e:     # a file that parsed above and not inside the audit (a parser limit at another depth)
        return False, "a .py file the network audit's parser cannot read (%s)" % _parse_error(e)[:200], suspicious, []
    if net:
        return False, "network audit: networking import(s) %s" % ", ".join("%s: %s" % x for x in net[:5]), suspicious, net
    dirs = rn.prepare("import_check", src)
    before = rn.snapshot((dirs["root"],)) if rn.acct else None
    code = ("import sys, importlib; sys.path.insert(0, sys.argv[1]); sys.path.insert(0, sys.argv[2]); "
            "m = importlib.import_module('%s'); assert callable(m.solve)" % PROGRAM)
    argv = runner.wrap_limits(OOM_FIRST + [sys.executable] + runner.PYTHON_FLAGS + ["-c", code, dirs["lib"], dirs["program"]],
                              cpu_s=IMPORT_CPU_S, user=rn.name, nonet=True)
    try:
        r = subprocess.run(argv, cwd=dirs["program"], env=rn.env(dirs), capture_output=True, text=True, timeout=IMPORT_WALL_S,
                           start_new_session=True)
        ok, why = r.returncode == 0, ("ok" if r.returncode == 0 else "importing %s.py failed: " % PROGRAM + (r.stderr.strip().splitlines() or ["?"])[-1][:300])
    except subprocess.TimeoutExpired:
        ok, why = False, "importing %s.py took more than %d s" % (PROGRAM, IMPORT_WALL_S)
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


def _settle_desktop(max_s=60.0, quiet_s=8.0, poll_s=0.25):
    """Wait, before the first shift's network baseline, until the frozen desktop is quiet. The image's supervisord
    respawns the agent account's desktop (Xvnc, websockify, GNOME, ...) after the verify-start kill, and a respawned
    process can open a loopback TCP connection in the moment before the freezer stops it. Quiet = the agent account's
    processes and the namespace's outbound counters unchanged for `quiet_s`. Returns (seconds waited, counter
    increments meanwhile)."""
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


LOOK_CACHE = []


def _prebuild_looks(cfg, shifts, metrics):
    """Every hidden shift's parcel looks made once, before the first shift, in worker processes, into a cache folder
    of the grader's own (mode 700, root's: the program's account cannot read it); each shift's cell then starts in
    seconds (parcelcell.kitcache). A failure here is not fatal: a cell builds its looks itself when the cache lacks
    them (the same bytes either way)."""
    import tempfile
    from parcelcell import kitcache
    look = cfg.setdefault("look", {})
    if (look.get("parcel_visuals") or "kit").split(":")[0] != "kit":
        return
    t0 = time.time()
    try:
        cdir = tempfile.mkdtemp(prefix="parcel-looks-", dir="/tmp")
        os.chmod(cdir, 0o700)
        LOOK_CACHE.append(cdir)
        res = kitcache.prebuild([pkg for _seed, pkg, _tag in shifts], cdir, workers=int(cfg.get("look_workers", 4)),
                                log=log)
        look["kit_cache_dir"] = cdir
        metrics["looks_prebuild_s"] = round(time.time() - t0, 1)
        metrics["looks_prebuilt"] = float(res.get("built", 0) + res.get("hits", 0))
        metrics["looks_missing"] = float(res.get("missing", 0))
    except Exception as e:  # noqa: BLE001 - the cells make their looks themselves then
        log("[grader] the shift looks were not prebuilt (%s: %s); each cell makes its own" % (type(e).__name__, e))
        metrics["looks_prebuild_s"] = round(time.time() - t0, 1)


def _drop_look_cache():
    for d in LOOK_CACHE:
        shutil.rmtree(d, ignore_errors=True)
    del LOOK_CACHE[:]


def _new_cell(cfg, package, privileged):
    """A fresh cell on one hidden shift package (the one place the cell's construction is spelled)."""
    server = ParcelCellServer(config=cfg, mode="hidden", privileged=privileged, log=log,
                              reference_dir=os.path.join(STAGE_DIR, "reference") if privileged else None)
    server.reset(package)
    return server


def _close_cell(server):
    try:
        server.close()
    except Exception:  # noqa: BLE001 - a record of the cell's teardown is never a scoring input
        pass


def run_episode(cfg, shift, rn, privileged, tag):
    """One graded shift: a fresh cell, the program on the wire as the runner account, the cell's record. A failure on
    the grader's side (the cell's build, the program's launch, the wire, the cleanup, the scoring) is a task error
    (`fail`), never the agent's zero; whatever the program does ends at most its own shift."""
    seed, package, name = shift
    try:
        server = _new_cell(cfg, package, privileged)
    except Exception:  # noqa: BLE001 - the cell depends on the stored package only: a build failure is the task's
        fail("%s: the cell could not be built\n%s" % (tag, traceback.format_exc()))
    b = server.cfg["budgets"]        # the cell's own (grader_config.json deep-merged over the cell's defaults)
    # the program's own time (the wall clock minus the cell's simulation and rendering, cellkit.wire), per shift in
    # proportion to its duration (the cell's program_budget_s); the privileged reference program runs inside one call
    # (reference_run), whose own time the cell caps (budgets.privileged_program_time_factor x the shift's duration)
    program_cap = float(server.program_budget_s)
    if privileged:
        program_cap += float(server.privileged_budget_s)
    cpu_s = rn.cpu_factor * program_cap
    silence_s = float(b.get("silence_s", 20.0))
    start_s = float(b.get("start_s", wire.START_S))
    after_over = {"steps": int(b.get("after_over_steps", 50)), "requests": int(b.get("after_over_requests", 200)),
                  "seconds": float(b.get("after_over_s", 60.0))}
    max_line = int(b.get("request_line_bytes", wire.MAX_LINE))
    record_dir(LOGS)
    t0 = time.time()
    dirs = rn.prepare(tag, program_source())          # an opaque tag: the hidden shift never reaches a path the program sees
    before = rn.snapshot((dirs["root"],)) if rn.acct else None
    argv, user = OOM_FIRST + launcher_argv(sys.executable, dirs["lib"], dirs["program"], PROGRAM), rn.name
    proc = None
    sampler = netaudit.SocketSampler(uid=rn.uid if rn.acct else None, pids_fn=None if rn.acct else (lambda: []),
                                     period_s=float(cfg.get("socket_sample_s", 0.5)))
    counters = netaudit.NetCounters()
    reason = None
    with counters, sampler:
        try:
            try:
                proc = wire.AgentProcess(argv, cwd=dirs["program"], user=user, env=rn.env(dirs, None), cpu_s=cpu_s,
                                         stderr_path=os.path.join(LOGS, "%s.log" % tag), nonet=True, max_line=max_line)
            except Exception:  # noqa: BLE001
                fail("%s: the program could not be launched\n%s" % (tag, traceback.format_exc()))
            try:
                reason = wire.serve_episode(server, proc, program_cap, silence_s, log=log, start_s=start_s,
                                            after_over=after_over)
            except Exception:  # noqa: BLE001 - the wire turns everything the program does into an end reason
                fail("%s: the wire failed\n%s" % (tag, traceback.format_exc()))
        finally:
            if proc is not None:
                proc.terminate()
            rn.cleanup(dirs, before)             # every process of the runner account is gone before the counters are read
    try:
        if not server.done:
            server._end(reason or "crashed")
        summary = server.finalize()
    except Exception:  # noqa: BLE001 - finalize() does not raise by design
        fail("%s: the shift could not be scored\n%s" % (tag, traceback.format_exc()))
    missing = [k for k in SUMMARY_REQUIRED if not isinstance(summary, dict) or k not in summary]
    if missing:
        fail("%s: the shift's record lacks %s" % (tag, ", ".join(missing)))
    try:
        sc = float(summary["shift_score"])
    except (TypeError, ValueError):
        sc = float("nan")
    if not (math.isfinite(sc) and 0.0 <= sc <= 1.0):
        # min(1.0, nan) is 1.0: a score that is not a finite number in [0, 1] is the grader's defect, never a reward
        fail("%s: the shift's score %r is not a finite number in [0, 1]" % (tag, summary.get("shift_score")))
    summary["shift_score"] = sc
    summary["episode_wall_s"] = time.time() - t0
    summary["end_reason"] = reason
    summary["tag"] = name
    summary["program_cap_s"] = program_cap
    summary["cpu_cap_s"] = cpu_s
    try:
        summary["log_lines"] = server.log_lines(summary)
    except Exception:  # noqa: BLE001 - a record is never a scoring input
        summary["log_lines"] = []
    _close_cell(server)
    summary["remote_sockets_seen"] = int(sampler.max_seen)
    summary["socket_samples"] = int(sampler.samples)
    summary["net_outbound"] = int(counters.total)
    summary["net_delta"] = dict(counters.delta)
    return summary


def _cell_failed(s):
    """What went wrong on the cell's side of a shift (not the program's doing): [] when nothing.
      internal_error  an exception in the cell's own work while the shift was live (`internal_errors_live`): every
                      argument of every call is checked at the server's boundary first (parcelcell/server.py
                      check_args) and a call that fails it is refused with bad_params before anything runs; an
                      exception after the shift had ended never counts;
      physics_fault   MuJoCo flagged an unstable state or the state became non-finite after a tick;
      render_failure  a frame that could not be rendered (it is not charged to the program's frame budget).
    A request line the JSON parser cannot read ends the shift as `protocol`, the program's doing, and never stops the
    wire (cellkit.wire.parse_request)."""
    out = []
    live = s.get("internal_errors_live")
    if int(_num(s.get("internal_errors", 0) if live is None else live)) > 0:
        out.append("internal_error")
    if s.get("done_reason") == "physics_fault" or s.get("end_reason") == "physics_fault":
        out.append("physics_fault")
    if int(_num(_dict(s.get("frames")).get("frames_lost", 0))) > 0:
        out.append("render_failure")
    return out


def _episode_with_retry(cfg, shift, rn, privileged, tag, metrics):
    """One shift; when the cell failed in it, it is run once more from scratch (a fresh cell, a fresh program) and the
    second run counts. At most one shift of a verify is run again (MAX_RERUNS_PER_VERIFY; the verifier's timeout is
    derived for one re-run): a second shift in which the cell failed is the grader's failure (a task error). A render
    failure that repeats is the grader's (a task error); an internal error or a physics fault that repeats with the
    same program is scored as it stands."""
    s_a = run_episode(cfg, shift, rn, privileged, tag=tag)
    why = _cell_failed(s_a)
    if not why:
        return s_a
    if metrics.get("episode_retries", 0.0) >= MAX_RERUNS_PER_VERIFY:
        fail("%s: the cell failed (%s) in a second shift of this verify; at most %d shift is run again per verify "
             "(the verifier's timeout is derived for that)" % (tag, ", ".join(why), MAX_RERUNS_PER_VERIFY))
    log("[grader]   %s: the cell failed (%s): the shift is run once more" % (tag, ", ".join(why)))
    metrics["episode_retries"] = metrics.get("episode_retries", 0.0) + 1.0
    s_b = run_episode(cfg, shift, rn, privileged, tag=tag + "r")
    why_b = _cell_failed(s_b)
    if "render_failure" in why_b:
        fail("%s: the cell could not render frames in two runs of the shift" % tag)
    s_b["retried_after"] = why
    return s_b


def write_verdict(rewards, metrics, contract_why=None, setting="line"):
    """One gated `reward` in the benchmark kit's envelope (robotics_grader.ratio_rewards against the recorded reference
    value); the gate, the criterion and every statistic go to `metrics`. In the engine (`ALE_VERIFY_PYTHON` set) the kit
    writes reward.json and verification.json, the criteria text as the `diagnostic`; the metrics are finite numbers only
    ('.' and '/' in a name replaced by '_'). Elsewhere the envelope goes to verdict.local.json in the stage folder.
    `contract_why`: the static gates' reason when they failed the contract; the diagnostic then names it."""
    from robotics_grader import ratio_rewards, read_anchor
    from robotics_grader import write_verdict as kit_write_verdict
    overall = float(rewards["facing_score"])
    anchor, meta = read_anchor()
    if setting != "rated":
        # the line set: the absolute score (anchor.json "line": the value 1.0 = every parcel of every list turned)
        line = meta.get("line") if isinstance(meta.get("line"), dict) else {}
        anchor = None if line.get("value") is None else float(line["value"])
        meta = dict(meta, full_at=float(line.get("full_at", 1.0)))
    extra = {}
    for k, v in metrics.items():
        try:
            extra[k] = float(v)
        except (TypeError, ValueError):
            continue
    extra.update({k: float(rewards[k]) for k in GATES + CRITERIA})
    verdict = ratio_rewards(overall, anchor, higher_is_better=meta.get("direction", "higher") == "higher",
                            metric_name=meta.get("metric", "facing_score"), extra=extra, full_at=meta.get("full_at", 1.0))
    if setting == "rated":
        text = ("contract_ok %.0f (the program imports, exposes solve, passes the route gate and the network audit and steps the "
                "cell; no late write, no surviving process); facing_score %.4f (the mean over the hidden shifts of the shift "
                "score max(0, credit) / #W parcels, credit = W parcels (in the station's duty, label not up at arrival) "
                "crossing the scan line readable - U parcels (label up at arrival) not crossing readable - parcels lost, any "
                "class)") % (rewards["contract_ok"], rewards["facing_score"])
    else:
        text = ("contract_ok %.0f (the program imports, exposes solve, passes the route gate and the network audit and steps the "
                "cell; no late write, no surviving process); facing_score %.4f (the mean over the hidden shifts of the shift "
                "score max(0, credit) / #W parcels of the shift's list, credit = W parcels (label not up at arrival) "
                "crossing the scan line readable - U parcels (label up at arrival) done and not readable - parcels that fell "
                "or were damaged; a parcel not done when the shift ends costs nothing)") % (
                    rewards["contract_ok"], rewards["facing_score"])
    if contract_why and not rewards["contract_ok"]:
        text += "; the contract failed: %s" % str(contract_why)[:300]
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
    kit_write_verdict(verdict, path=os.path.join(STAGE_DIR, "verdict.local.json"), diagnostic=text)
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
    metrics = {"grader_exception": 0.0, "shifts_graded": float(cfg.get("shifts_graded", 8)),
               "root_payload_ok": 0.0, "agent_deprivileged": 0.0, "runner_uid": -1.0, "privileged_mode": 0.0, "oracle_token_mismatch": 0.0,
               "runner_processes_killed": 0.0, "runner_processes_left": 0.0, "runner_unclean_runs": 0.0,
               "stray_files_removed": 0.0, "suspicious_lines": 0.0, "remote_sockets_seen": 0.0, "net_outbound_events": 0.0,
               "network_imports": 0.0, "illegal_call_count": 0.0, "hidden_set_placeholder": 0.0}
    for k in ("agent_account_cleared", "agent_account_programs_disabled", "agent_account_clear_killed",
              "stage_locked", "stage_parent_locked", "stage_lock_latency_ms", "agent_processes_killed", "agent_processes_left",
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
        _drop_look_cache()
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
    n = int(cfg.get("shifts_graded", 8))
    setting, root = graded_set(privileged)
    metrics["graded_set_line"] = 1.0 if setting == "line" else 0.0
    log("[grader] graded set: %s" % setting)
    ep_seeds, shifts, placeholder = hidden_shifts(n, root, setting)
    metrics["hidden_set_placeholder"] = 1.0 if placeholder else 0.0
    metrics["shifts_graded"] = float(len(shifts))
    stream = seed_rng(pinned=ep_seeds)
    metrics.update(stream.stats())
    if not ok:
        rewards["contract_ok"] = 0.0
        _finish_stats(rn, metrics, t_start)
        write_verdict(rewards, metrics, contract_why=why, setting=setting)
        return
    settle_s, settle_seen = _settle_desktop()
    metrics["desktop_settle_s"] = round(settle_s, 1)
    metrics["net_settle_events"] = float(settle_seen)
    if settle_s > 0.0:
        log("[grader] the frozen desktop settled in %.1f s (%d outbound counter increments meanwhile, before any shift)" % (
            settle_s, settle_seen))
    nn_ok, nn_why = nonet.selftest(wrap=lambda a: runner.wrap_limits(a, cpu_s=60, user=rn.name),
                                   env=runner.child_env("/tmp", "/tmp", "/nonexistent", ROOT_GIVEN))
    metrics["nonet_selftest_ok"] = 1.0 if nn_ok else 0.0
    if not nn_ok:
        fail("the no-network filter for the program does not work here: " + nn_why)
    _prebuild_looks(cfg, shifts, metrics)
    agent = []
    step_calls = illegal = 0
    metrics["episode_retries"] = 0.0
    for i, shift in enumerate(shifts):
        tag = "ep%02d" % (i + 1)
        log("[grader] shift %d/%d" % (i + 1, len(shifts)))
        s_a = _episode_with_retry(cfg, shift, rn, privileged, tag, metrics)
        log("[grader]   score %.3f credit %s (W %s readable %s, U %s not readable %s, X %s, lost %s) stops %s sim %.1f s wall %.1f s "
            "(program %.1f s of %.0f, cell %.1f s) frames %s/%s end %s/%s" % (
                s_a["shift_score"], s_a.get("credit"), s_a.get("n_W"), s_a.get("W_readable"), s_a.get("n_U"),
                s_a.get("U_not_readable"), s_a.get("n_X"), s_a.get("lost"), _dict(s_a.get("safety")).get("protective_stop_count"),
                _num(s_a.get("t_end")), s_a["episode_wall_s"], _num(s_a.get("program_time_s")), s_a["program_cap_s"],
                _num(s_a.get("cell_work_s")), _dict(s_a.get("frames")).get("used"), _dict(s_a.get("frames")).get("budget"),
                s_a.get("done_reason"), s_a["end_reason"]))
        agent.append(s_a)
        step_calls += int(_num(s_a["step_calls"])) + int(_num(_dict(s_a.get("calls")).get("reference_run", 0)))
        illegal += int(_num(s_a["illegal_calls"]))
        metrics["remote_sockets_seen"] = max(metrics["remote_sockets_seen"], float(s_a["remote_sockets_seen"]))
        metrics["net_outbound_events"] += float(s_a.get("net_outbound", 0))
        for ck, cv in (s_a.get("net_delta") or {}).items():
            metrics["net_" + ck] = metrics.get("net_" + ck, 0.0) + float(max(0, cv))
    vals = [float(s["shift_score"]) for s in agent]          # each checked finite and in [0, 1] (run_episode)
    steps_gate = 1.0 if step_calls > 0 else 0.0
    clean_gate = 1.0 if rn.unclean_runs == 0 else 0.0
    # the gate counts only what belongs to the runner account (its remote inet sockets; under the seccomp filter it
    # cannot open one); the namespace's counters (loopback and every account included) stay statistics
    net_gate = 1.0 if metrics["remote_sockets_seen"] <= 0 else 0.0
    gate = steps_gate * clean_gate * net_gate
    rewards["contract_ok"] = gate
    if not net_gate:
        log("[grader] contract_ok 0: the program held a remote inet socket during a shift (%d seen)" % metrics["remote_sockets_seen"])
    elif metrics["net_outbound_events"] > 0:
        log("[grader] note: %d namespace-wide outbound counter increments during the shifts (statistic only: loopback and "
            "other accounts count too; the program runs under the no-network filter)" % metrics["net_outbound_events"])
    mean = float(np.mean(vals)) if vals else 0.0
    rewards["facing_score"] = float(max(0.0, min(1.0, gate * mean)))
    for i, s in enumerate(agent):
        metrics["shift_score.ep%02d" % (i + 1)] = float(s["shift_score"])
    for key in ("credit", "n_W", "n_U", "n_X", "W_readable", "U_not_readable", "lost", "max_entry_delay_s", "t_end",
                "program_time_s", "program_budget_s", "cell_work_s", "tick_wall_s", "build_wall_s", "episode_wall_s"):
        metrics[key + "_mean"] = float(np.mean([_num(s.get(key)) for s in agent])) if agent else 0.0
    n_w = sum(_num(s.get("n_W")) for s in agent)
    metrics["W_readable_rate"] = float(sum(_num(s.get("W_readable")) for s in agent) / n_w) if n_w else 0.0
    metrics["lost_total"] = float(sum(_num(s.get("lost")) for s in agent))
    for why in ("fall", "damaged", "not_across"):
        metrics["lost_by." + why] = float(sum(_num(_dict(s.get("lost_by")).get(why)) for s in agent))
    for why in ("not_set_down", "in_station"):
        metrics["not_done." + why] = float(sum(_num(_dict(s.get("not_done")).get(why)) for s in agent))
    metrics["protective_stops_mean"] = float(np.mean([_num(_dict(s.get("safety")).get("protective_stop_count")) for s in agent])) if agent else 0.0
    metrics["locked_out_rate"] = float(np.mean([1.0 if _dict(s.get("safety")).get("locked_out") else 0.0 for s in agent])) if agent else 0.0
    reasons = [str(s.get("end_reason")) for s in agent]
    for r in sorted(set(reasons)):
        metrics["end_rate." + r] = float(sum(1 for x in reasons if x == r)) / max(1, len(reasons))
    metrics["illegal_call_count"] = float(illegal)
    metrics["step_calls"] = float(step_calls)
    metrics["bad_params_total"] = float(sum(_num(s.get("bad_params")) for s in agent))
    metrics["frames_used_mean"] = float(np.mean([_num(_dict(s.get("frames")).get("used")) for s in agent])) if agent else 0.0
    metrics["frames_budget_mean"] = float(np.mean([_num(_dict(s.get("frames")).get("budget")) for s in agent])) if agent else 0.0
    metrics["render_wall_s_mean"] = float(np.mean([_num(_dict(s.get("frames")).get("render_wall_s")) for s in agent])) if agent else 0.0
    cams = sorted(set(k for s in agent for k in _dict(_dict(s.get("frames")).get("by_camera")) if isinstance(k, str)))
    for k in cams:
        metrics["frames_mean." + k] = float(np.mean([_num(_dict(_dict(s.get("frames")).get("by_camera")).get(k, 0)) for s in agent]))
    metrics["cell_internal_errors"] = float(sum(int(_num(s.get("internal_errors"))) for s in agent))
    try:                                            # the maintainers' record; never a scoring input
        record_dir(OUT)
        with open(os.path.join(OUT, "shifts.json"), "w") as f:
            json.dump(agent, f, indent=1, default=_record_value)
        with open(os.path.join(OUT, "log_lines.txt"), "w") as f:
            for s in agent:
                f.write("%s (%s, end %s)\n" % (s.get("tag"), s["shift"]["sha256"][:12] if isinstance(s.get("shift"), dict) else "?",
                                               s.get("end_reason")))
                for line in s.get("log_lines") or []:
                    f.write(line + "\n")
    except Exception:  # noqa: BLE001
        log("[grader] shifts.json not written\n" + traceback.format_exc())
    _finish_stats(rn, metrics, t_start)
    overall = write_verdict(rewards, metrics, setting=setting)
    log("[grader] rewards %s overall %.4f (verify wall %.0f s)" % ({k: round(v, 4) for k, v in rewards.items()}, overall, metrics["verify_wall_s"]))


def _finish_stats(rn, metrics, t_start):
    metrics["runner_processes_killed"] = float(rn.killed)
    metrics["runner_processes_left"] = float(rn.left)
    metrics["runner_unclean_runs"] = float(rn.unclean_runs)
    metrics["stray_files_removed"] = float(rn.stray_files)
    metrics["submission_symlinks_skipped"] = float(max(rn.symlinks_skipped, int(EARLY.get("submission_snapshot_skipped", 0.0))))
    metrics["agent_processes_frozen"] = float(len(FREEZER[0].frozen)) if FREEZER[0] is not None else 0.0
    metrics["verify_wall_s"] = time.time() - t_start
    shutil.rmtree(RUNS_ROOT, ignore_errors=True)


if __name__ == "__main__":
    main()
