"""Grader of freeflyer_uncrewed_station_caretaking (a cabin free-flyer alone in an uncrewed three-module station: from
the station's telemetry it works out which return inlet is blocked, takes the soft sheet off, presses it onto a
holding screen, puts the loose pouches on pouch pads and docks in time with charge left). Runs as ROOT in the verify
stage. Verification is separate (task.toml): this runs in a fresh verifier sandbox started from the task image, with
only the agent's submission restored into it; tests/test.sh built the root-owned copy of the givens (/opt/arb) and
named the asset root before this starts.

The agent left /home/user/submission/caretake.py (solve(robot): the robot's autonomous program; robot is the client
library's Robot, ~/given/lib/cabinkit). This grader:
  0. FIRST, stdlib only (`_early_lockdown`; tests/test.sh closed the stage folder already): records the upload time,
     freezes and kills every process of the agent account and keeps it stopped until the verdict is written
     (cellkit.verify_freeze), locks the stage, snapshots the submission root-owned, gates late writes, quarantines late
     entries elsewhere (in the separate verifier there is nothing of the agent's to stop or set aside);
  1. the task's own inputs, each a TASK ERROR when broken (a zero says the agent failed, a task error says the grading
     broke): the runtime copy of the task's numbers (tests/cabincell/data/site_spec.json against its sha256 file) and
     tests/grader_config.json; the cell's numeric pins in this process's environment equal to the runtime key
     cell.recipe.env (the same bits on every CPU); the BLAS kernel pin complete (OPENBLAS_CORETYPE never without OPENBLAS_L2_SIZE=256,
     ``blas_policy``) and the BLAS self-test in child processes of this environment (``blas_selftest``), both before
     numpy is loaded here; the root-owned copy of the givens at /opt/arb against given_hashes.json (the client
     library handed to the programs comes from it); the fixed hidden set of tests/hidden/selection.json (its `set`:
     exactly layout.sites site packages, each pinned by sha256, with the reference's energy on the site); the three
     asset components of the cell (robot, station, robot_author) in the asset root $CABINCELL_ASSET_ROOT (tests/test.sh
     sets it, default /opt/task-assets): an archive present there must have the sha256 pinned in
     cabincell/data/<component>/ASSETS.json, an unpacked component folder must match that manifest file by file, and a
     component that is neither there nor archived is a task error; the no-network filter's self-test (cellkit.nonet;
     run before the import check, so that a broken filter never reads as the agent's failed import);
  2. gates (checks of the submission's FORM only): contract_ok = caretake.py is a regular file that parses and defines
     solve, the route gate (no weight files, no training calls, at most cell.contract.max_submission_mb MB), every .py
     file readable by the parser, the network audit (static: no networking import, no computed import, no computed
     code, no program start; runtime: every process of the program -- the import check and every site's run -- runs
     under the no-network seccomp filter, and the runner account's remote inet sockets are sampled), the module imports
     under the run regime (cell.contract.import_check: 120 s wall, 120 CPU-s, as `nobody`, no network), on EVERY site
     at least cell.contract.min_steps_per_site calls that advanced the simulated time (step() or any call that
     consumed simulated time, counted from the cell's record of the run: `sim_time_calls`), no late
     write, no surviving process. No caretake.py at all is the honest zero (robotics_grader.zero_verdict);
  3. the oracle token: submission/.oracle_token whose sha256 equals hidden/oracle_token.sha256 switches the cells to the
     privileged mode (the call reference_run: the privileged reference program inside the cell, tests/reference/);
     a wrong token is ignored (stat oracle_token_mismatch);
  4. runs the program per hidden site, in the set's order, each on a FRESH cell (cabincell.sensors_render.make_server:
     CabinCellServer with the frame renderer bound; the cell, its render twin and its renderer are released before the
     next site) as `nobody` from a read-only copy under an opaque run tag (the package's name never reaches a path the
     program sees), oom_score_adj 1000 (`choom`: when memory runs out the kernel stops the program, never the grader),
     on the cellkit wire with cabincell.wire_guard.GuardedProcess, the program-time and silence budgets of the runtime
     spec (cell.budget.*); a site on which the CELL failed (its record's exit `cell_error`, or an internal error of the
     cell during the run) is run once more from scratch and the second run counts; at most cell.fault_rerun runs of a
     verify are run again: a second cell failure (on the same site or another) is a TASK ERROR;
  5. criterion: caretaking_score = contract_ok x the mean over the sites of the site score (cabincell.grade.grade:
     R x (D + U1 + H + U2 + V + K + L + E), every threshold from tests/grader_config.json, the energy line from the
     reference's energy on the site recorded in the set); reward = clamp(caretaking_score / anchor / full_at, 0, 1)
     (tests/anchor.json). A failure on the grader's side of a run (the cell's build, the program's launch, the wire, the
     record, the grading, a site score that is not a finite number in [0, 1]) is a TASK ERROR; whatever the program does
     (crash, hang, silence, garbage on the wire) ends at most its own run, which is scored on the world it left.
     The verdict's metrics hold finite numbers only.
Untouched sandbox: no caretake.py -> zero_verdict -> 0. Oracle: the token + reference_run -> the recorded reference
value -> 1.
Records: out/episodes.json (every site's summary), out/<tag>.record.json (the cell's record, the record format)
and logs/<tag>.log (the program's stderr) under the root-only /opt/arb-verify (outside /opt/ale, which the engine
clears before it keeps a sandbox); host-side runs as another account keep them under $ARB_RECORD_ROOT or the stage.

Adapted from the tests/verify.py of the quadruped plant inspection task of this benchmark: the early lockdown, the root payload check, the route gate, the parser and network audits, the import check, the token,
the desktop settle, the runner, the per-run wire and clean-up, the gates and the verdict writer are that file's.
Changed for this task: the program is caretake.py on cabinkit; every run setting is read by key from
the runtime site_spec (never written here) except the route gate's lists, which are the sibling's grader_config
values; the hidden set comes from hidden/selection.json; the asset root check; the no-network self-test before the
import check; the per-site gate on simulated time (every call that advanced the simulated time counts, from the
cell's record, so that a program that moves the robot with blocking calls only is not failed for writing no
step()); the cell is CabinCellServer with the
frame renderer; grading is cabincell.grade; a cell failure is the record's exit cell_error or a cell internal error;
the honest zero.
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
        # a survivor is a statistic, not a contract failure: verification is separate (task.toml), nothing of the
        # agent's runs in this sandbox; the freezer below keeps every process of the account stopped until the verdict
        if left:
            log("[grader] %d process(es) of the agent account outlived the kill at verify start: frozen until the verdict" % left)
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

import gc  # noqa: E402
import hashlib  # noqa: E402
import math  # noqa: E402
from pathlib import Path  # noqa: E402

from cellkit import netaudit, nonet, wire  # noqa: E402

import cabincell  # noqa: E402,F401  (stdlib only at import: puts the cell's _lib and its own folder on sys.path)
import assetroot as AR  # noqa: E402  (tests/cabincell/assetroot.py, stdlib only)
from cabincell import grade as GR  # noqa: E402  (stdlib only)

HIDDEN = os.path.join(STAGE_DIR, "hidden")
SELECTION = os.path.join(HIDDEN, "selection.json")
REFERENCE_DIR = os.path.join(STAGE_DIR, "reference")
# The maintainers' record (the program's stderr, the cell's records, episodes.json) lives outside /opt/ale: the engine
# deletes its stage folders before it keeps a sandbox, and the record is read from a kept one. Root-only (0700).
RECORD_ROOT = os.environ.get("ARB_RECORD_ROOT") or ("/opt/arb-verify" if os.getuid() == 0 else STAGE_DIR)
LOGS = os.path.join(RECORD_ROOT, "logs")
OUT = os.path.join(RECORD_ROOT, "out")
PROGRAM = "caretake"
CRITERIA = ["caretaking_score"]
GATES = ["contract_ok"]
RUNNER_ACCOUNT = "nobody"                          # the program runs as nobody (fallback: the agent account)
# the asset root of the cell's three components (cabincell/assetroot.py reads the same variable); tests/test.sh exports
# it with this default, the image's root-owned copy (environment/Dockerfile)
ASSET_ENV = AR.ENV
ASSET_ROOT_DEFAULT = "/opt/task-assets"
ASSET_COMPONENTS = ("robot", "station", "robot_author")   # the maker's robot mesh for the frames is required too
# Route gate: the sibling task's lists (its tests/grader_config.json `route_gate`), the size cap from the runtime spec.
FORBIDDEN_EXTENSIONS = (".pt", ".pth", ".onnx", ".safetensors", ".ckpt", ".h5", ".pb", ".pkl", ".npz")
FORBIDDEN_CALLS = ("torch.optim", ".backward(", "jax.grad")
CPU_CAP_FACTOR = 4.0          # the program's CPU seconds per site = this x cell.budget.program_compute_s (sibling's value)
SOCKET_SAMPLE_S = 0.5
# the kernel's first choice when the sandbox's memory runs out is the program, never the grader (as in the sibling
# task); util-linux's choom is in the image; a host without it runs the program without the prefix (stat)
OOM_FIRST = ["choom", "-n", "1000", "--"] if shutil.which("choom") else []
PARSE_ERRORS = (SyntaxError, RecursionError, MemoryError, ValueError)
SUSPICIOUS = ("/opt/ale", "ALE_STAGE_DIR", "ALE_VERIFICATION", "hidden/selection", "hidden/sites", "oracle_token",
              "reference_run", "ALE_VERIFY", "/opt/ale/tests", ".ale-solution", "/opt/task-assets", "CABINCELL_ASSET_ROOT")
# what a run's record (CabinCellServer.finalize, the record format) must carry for the grader
RECORD_REQUIRED = ("schema", "times", "metrics", "state", "hashes", "budgets_used")
# runtime keys the verifier reads (tests/cabincell/data/site_spec.json; values never written here)
RUNTIME_KEYS = ("layout.sites", "cell.fault_rerun", "cell.contract.import_check", "cell.contract.max_submission_mb",
                "cell.contract.min_steps_per_site", "cell.contract.python_flags", "cell.recipe.env",
                "cell.budget.program_compute_s", "cell.budget.silence_s")


def record_dir(path):
    """`path` under the record root, created on demand; never a scoring input."""
    os.makedirs(RECORD_ROOT, exist_ok=True)
    if RECORD_ROOT != STAGE_DIR:
        try:
            os.chmod(RECORD_ROOT, 0o700)
        except OSError:
            pass
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
    if isinstance(v, bool):
        return 1.0 if v else 0.0
    try:
        f = float(v)
    except (TypeError, ValueError):
        return float(default)
    return f if math.isfinite(f) else float(default)


def _dict(v):
    return v if isinstance(v, dict) else {}


def _mean(xs):
    xs = [float(x) for x in xs]
    return sum(xs) / len(xs) if xs else 0.0


def _p95(xs):
    xs = sorted(float(x) for x in xs)
    if not xs:
        return 0.0
    return xs[min(len(xs) - 1, int(math.ceil(0.95 * len(xs))) - 1)]


def _clock(v):
    """A mission-clock value of the record as a finite float, or None (missing, not a number, not finite)."""
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    f = float(v)
    return f if math.isfinite(f) else None


def sim_time_calls(record):
    """(n, t_end) of one run, read from the cell's record (the record format; the cell's code is frozen, so the gate
    uses only what the record already holds). t_end = times.program_end: the mission clock (simulated seconds from the
    program's start, the robot API's conventions.time) when the program's run ended, before the settle. n = the number of the
    program's calls that advanced the simulated time, as far as the record shows them: the record lists every call
    except step() and the four queries (`calls`, each with the clock when the call returned) and counts the step()
    calls without their clock (metrics.calls_step). Simulated time passes only inside calls (the robot API's client.time), so
    every rise of the clock along 0, t_1, ..., t_k, t_end was made by at least one call of that interval (the listed
    call that closes it, or a step() before it): n = the number of rises is a lower bound of the calls that advanced
    the simulated time, and n >= 1 exactly when the clock moved at all on the site. A clock value that is missing or
    not a finite number counts as no rise (the unfavourable reading). The runtime minimum
    (cell.contract.min_steps_per_site) is 1: for it the lower bound decides the gate exactly; a larger minimum would
    need the clock of each step(), which the record does not hold."""
    rec = _dict(record)
    prev, n = 0.0, 0
    for e in rec.get("calls") or []:
        t = _clock(e.get("t")) if isinstance(e, dict) else None
        if t is not None and t > prev:
            n, prev = n + 1, t
    t_end = _clock(_dict(rec.get("times")).get("program_end"))
    if t_end is not None and t_end > prev:
        n += 1
    return n, (t_end if t_end is not None else 0.0)


def finite_metrics(m):
    """Only finite float values under string names (the verdict's metrics may hold nothing else)."""
    out = {}
    for k, v in m.items():
        if isinstance(v, bool):
            out[str(k)] = 1.0 if v else 0.0
            continue
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        if math.isfinite(f):
            out[str(k)] = f
    return out


def _record_value(o):
    """json.dump fallback for the maintainers' record: numpy arrays and scalars as plain values, anything else as text."""
    if hasattr(o, "tolist"):
        return o.tolist()
    if isinstance(o, (set, frozenset, tuple)):
        return list(o)
    return str(o)


# ------------------------------------------------------------------------------------------------- the task's inputs
def load_inputs():
    """(runtime, config): the runtime copy of the task's numbers (sha256 self-check) and the grader's thresholds; a
    missing key or a broken file is a task error."""
    try:
        RT = GR.load_runtime()
        C = GR.load_config()
    except Exception as e:  # noqa: BLE001 - the task's own files
        fail("the task's numbers are unusable (tests/cabincell/data/site_spec.json or tests/grader_config.json): %s" % e)
    missing = [k for k in RUNTIME_KEYS if k not in RT["values"]]
    if missing:
        fail("the runtime site_spec lacks the verifier's keys %s" % missing)
    return RT, C


def pins_ok(rt_values):
    """The cell's numeric pins (runtime key cell.recipe.env) must be this process's environment, as
    tests/test.sh exports them before the interpreter starts: [] or the differing names."""
    want = rt_values["cell.recipe.env"]
    return sorted(k for k, v in _dict(want).items() if os.environ.get(k) != str(v))


BLAS_L2 = ("OPENBLAS_L2_SIZE", "256")
# The self-test's child: level-3 products of every dtype with 64 result rows and inner dimension 512 (the block size
# that overruns the pinned Haswell kernel's stack buffer when the Zen 4/5 block-size override of OpenBLAS 0.3.33-0.3.34,
# numpy 2.5.2-2.5.5, is on), a 700 x 700 solve and a 400 x 700 SVD; core dumps off (PR_SET_DUMPABLE = 0).
BLAS_PROBE = ("import ctypes;ctypes.CDLL(None).prctl(4,0,0,0,0);import numpy as n;"
              "[(n.ones((64,512),t)@n.ones((512,64),t)==512).all() or exit(3) for t in (n.float64,n.float32,n.complex128,n.complex64)];"
              "a=n.random.default_rng(0).random((700,700))+30*n.eye(700);n.linalg.solve(a,n.ones(700));n.linalg.svd(a[:400])")


def blas_policy() -> str:
    """'' when the BLAS kernel pin is complete, else the reason: OPENBLAS_CORETYPE is never set without
    OPENBLAS_L2_SIZE=256 (with the kernel forced and the L2 size unset, the forced table gets 512-row blocks on AMD Zen 4
    and Zen 5 hosts and the Haswell kernel writes past its stack buffer; a host of another kind would not show it)."""
    core = os.environ.get("OPENBLAS_CORETYPE")
    if core and os.environ.get(BLAS_L2[0]) != BLAS_L2[1]:
        return "OPENBLAS_CORETYPE=%s is set with %s=%r (want %s)" % (core, BLAS_L2[0], os.environ.get(BLAS_L2[0]), BLAS_L2[1])
    return ""


def blas_selftest(tries: int = 2, timeout_s: float = 300.0) -> str:
    """'' when ``tries`` fresh child processes with this process's environment run BLAS_PROBE to the end, else the
    reason (exit codes; a negative code is the signal)."""
    codes = []
    for _ in range(tries):
        try:
            codes.append(subprocess.run([sys.executable, "-I", "-B", "-c", BLAS_PROBE], stdout=subprocess.DEVNULL,
                                        stderr=subprocess.DEVNULL, timeout=timeout_s).returncode)
        except subprocess.TimeoutExpired:
            codes.append("timeout")
    return "" if all(c == 0 for c in codes) else "child exit codes %s (negative = signal)" % codes


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


def hidden_set(n):
    """The fixed hidden set: [{tag, label, path, sha256, seed, ref_energy_wh}] from the `set` of hidden/selection.json, in
    episode order. Exactly `n` (layout.sites) entries; each package a regular file under hidden/ whose sha256 is the
    recorded one; seed an integer or null (null: the cell's own seed of the package, server.noise_seed_for); the
    reference's energy on the site a finite positive number (the e_elec line of E). Anything else is a task error."""
    try:
        doc = load_json(SELECTION)
    except (OSError, ValueError) as e:
        fail("hidden/selection.json is unreadable: %s" % e)
    entries = doc.get("set") if isinstance(doc, dict) else None
    if not isinstance(entries, list) or len(entries) != n:
        fail("hidden/selection.json must hold the fixed hidden set as `set`: a list of %d site entries (layout.sites)" % n)
    out, tags = [], set()
    hidden_abs = os.path.abspath(HIDDEN)
    for i, e in enumerate(entries):
        if not isinstance(e, dict):
            fail("hidden/selection.json: set entry %d is not an object" % (i + 1))
        rel = e.get("package")
        if not isinstance(rel, str) or not rel:
            fail("hidden/selection.json: set entry %d has no package path" % (i + 1))
        path = os.path.normpath(os.path.join(hidden_abs, rel))
        if not path.startswith(hidden_abs + os.sep) or os.path.islink(path) or not os.path.isfile(path):
            fail("hidden/selection.json: the package of entry %d (%r) is not a regular file under hidden/" % (i + 1, rel))
        want = str(e.get("sha256") or "").strip().lower()
        if len(want) != 64 or sha256_file(path) != want:
            fail("hidden/selection.json: the package of entry %d (%s) does not match its recorded sha256" % (i + 1, rel))
        if any(o["sha256"] == want for o in out):
            fail("hidden/selection.json: the package of entry %d (%s) is in the set twice" % (i + 1, rel))
        seed = e.get("seed")
        if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
            fail("hidden/selection.json: the seed of entry %d is neither an integer nor null" % (i + 1))
        ref_e = e.get("ref_energy_wh")
        if isinstance(ref_e, bool) or not isinstance(ref_e, (int, float)) or not math.isfinite(float(ref_e)) or float(ref_e) <= 0.0:
            fail("hidden/selection.json: entry %d has no finite positive ref_energy_wh (the reference's energy on the "
                 "site, the energy line of E)" % (i + 1))
        label = str(e.get("tag") or "")
        if label and label in tags:
            fail("hidden/selection.json: the tag %r is used twice" % label)
        tags.add(label)
        # the run tag is opaque (ep01 ...): the set's own tag is a label of the maintainers' record only, it never
        # reaches a path the program sees
        out.append(dict(tag="ep%02d" % (i + 1), label=label, path=path, sha256=want, seed=seed, ref_energy_wh=float(ref_e),
                        index=i))
    return out


def assets_ok():
    """The cell's asset components in the asset root ($CABINCELL_ASSET_ROOT, default ASSET_ROOT_DEFAULT): for each of
    ASSET_COMPONENTS, against its pin file tests/cabincell/data/<component>/ASSETS.json --
      an archive <root>/<archive name> that is present must have the pinned sha256;
      an unpacked folder <root>/<top> that is present must match the manifest file by file (assetroot.verify_tree:
      every listed file with its sha256 and size, no other file);
      a component with neither is missing.
    Any problem is a task error naming the component, the archive and the pinned sha256. When every component is
    unpacked in the root, the cell reads the root itself; otherwise the archived ones are unpacked (after the check)
    into a root-only asset root under the record root, the unpacked ones linked there, and the cell reads that one.
    Returns the statistics; sets $CABINCELL_ASSET_ROOT for the cell (assetroot searches it first)."""
    root = os.environ.get(ASSET_ENV) or ASSET_ROOT_DEFAULT
    problems, plan = [], {}
    for comp in ASSET_COMPONENTS:
        try:
            man = AR.manifest(comp)
        except AR.AssetError as e:
            fail("the task's asset pin file for %r is unusable: %s" % (comp, e))
        a = man["archive"]
        top = a["top"]
        want = "%s (sha256 %s, data/%s/ASSETS.json)" % (a["name"], a["sha256"], comp)
        arc = os.path.join(root, a["name"])
        unp = os.path.join(root, top)
        archived = False
        if os.path.lexists(arc):
            if os.path.islink(arc) or not os.path.isfile(arc):
                problems.append("%s: %s is not a regular file; expected %s" % (comp, arc, want))
                continue
            got = sha256_file(arc)
            if got != a["sha256"]:
                problems.append("%s: the archive %s has sha256 %s; expected %s" % (comp, arc, got, want))
                continue
            archived = True
        if os.path.isdir(unp):
            probs = AR.verify_tree(Path(unp), man)
            if probs:
                problems.append("%s: the unpacked folder %s does not match its manifest (%d problems, first %s); expected %s"
                                % (comp, unp, len(probs), probs[:3], want))
                continue
            plan[comp] = ("unpacked", unp, man)
        elif archived:
            plan[comp] = ("archive", arc, man)
        else:
            problems.append("%s: neither the archive %s nor its unpacked folder %s/ is in the asset root %s"
                            % (comp, want, top, root))
    if problems:
        fail("the cell's task assets are missing or altered: " + "; ".join(problems))
    stats = {"assets_ok": 1.0, "assets_unpacked_here": float(sum(1 for v in plan.values() if v[0] == "archive"))}
    if all(v[0] == "unpacked" for v in plan.values()):
        effective = root
    else:
        effective = record_dir(os.path.join(RECORD_ROOT, "assets"))
        for comp, (kind, p, man) in sorted(plan.items()):
            dst = os.path.join(effective, man["archive"]["top"])
            if kind == "unpacked":
                if not os.path.lexists(dst):
                    os.symlink(p, dst)
            else:
                try:
                    AR.unpack(Path(p), Path(effective), man)
                except AR.AssetError as e:
                    fail("the cell's task asset %r could not be unpacked: %s" % (comp, e))
    os.environ[ASSET_ENV] = effective
    stats["asset_root_is_default"] = 1.0 if os.path.abspath(root) == ASSET_ROOT_DEFAULT else 0.0
    return stats


# ------------------------------------------------------------------------------------------------- the gates
def route_gate(src, max_mb):
    total, problems, suspicious = 0, [], 0
    for dp, _, fns in os.walk(src):
        for fn in fns:
            p = os.path.join(dp, fn)
            try:
                total += os.lstat(p).st_size
            except OSError:
                continue
            if fn.lower().endswith(FORBIDDEN_EXTENSIONS):
                problems.append("weight file %s" % os.path.relpath(p, src))
            if fn.endswith((".py", ".sh")) and not os.path.islink(p):
                try:
                    text = open(p, encoding="utf-8", errors="replace").read()
                except OSError:
                    continue
                for c in FORBIDDEN_CALLS:
                    if c in text:
                        problems.append("%s uses %s" % (os.path.relpath(p, src), c))
                suspicious += sum(1 for line in text.splitlines() if any(s in line for s in SUSPICIOUS))
    if total > max_mb * 1024 * 1024:
        problems.append("submission larger than %d MB" % max_mb)
    return problems, suspicious


def library_digests():
    """sha256 of every .py file of the shipped client library (a submission may carry copies of it)."""
    out = set()
    d = os.path.join(ROOT_GIVEN, "lib", "cabinkit")
    for dp, _dn, fns in os.walk(d):
        for fn in fns:
            if fn.endswith(".py"):
                out.add(sha256_file(os.path.join(dp, fn)))
    return out


class Runner(object):
    def __init__(self, program_compute_s):
        self.acct = runner.runner_account(RUNNER_ACCOUNT, AGENT_USER)
        self.name, self.uid, self.gid = self.acct if self.acct else (None, None, None)
        self.lib_dirs = {"cabinkit": os.path.join(ROOT_GIVEN, "lib", "cabinkit")}
        self.cpu_s = CPU_CAP_FACTOR * float(program_compute_s)
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


def _parse_error(e):
    """The text of a parser failure: a SyntaxError's own message, the others with their name."""
    return str(e) if isinstance(e, SyntaxError) else "%s: %s" % (type(e).__name__, e)


def unparsable_files(src):
    """[(relpath, error text)] of the submission's .py files the parser cannot read at all (RecursionError, MemoryError,
    ValueError; as in the sibling task): the walk and decoding are the network audit's, so that no file
    the audit reads reaches it unparsed. A SyntaxError is left to the audit's own rule (no program can import it)."""
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


def contract_check(rt_values, rn):
    """(ok, why, suspicious lines, network imports, present): the static gates and the import check (module doc 2)."""
    src = program_source()
    path = os.path.join(src, PROGRAM + ".py")
    if not os.path.isfile(path) or os.path.islink(path):
        return False, "no %s.py (regular file) in the submission" % PROGRAM, 0, [], False
    try:
        tree = ast.parse(open(path, encoding="utf-8", errors="replace").read())
    except PARSE_ERRORS as e:
        return False, "%s.py does not parse: %s" % (PROGRAM, _parse_error(e)), 0, [], True
    has_solve = any(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == "solve" for n in ast.walk(tree)) or \
        any(isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "solve" for t in n.targets) for n in ast.walk(tree)) or \
        any(isinstance(n, (ast.Import, ast.ImportFrom)) and any((a.asname or a.name) == "solve" for a in n.names) for n in ast.walk(tree))
    if not has_solve:
        return False, "%s.py defines no solve" % PROGRAM, 0, [], True
    problems, suspicious = route_gate(src, float(rt_values["cell.contract.max_submission_mb"]))
    if problems:
        return False, "route gate: " + "; ".join(problems[:5]), suspicious, [], True
    bad = unparsable_files(src)
    if bad:
        return False, "a .py file the parser cannot read: %s" % "; ".join("%s (%s)" % x for x in bad[:5]), suspicious, [], True
    try:
        net = netaudit.scan_imports(src, allow_digests=library_digests())
    except PARSE_ERRORS as e:
        return False, "a .py file the network audit's parser cannot read (%s)" % _parse_error(e)[:200], suspicious, [], True
    if net:
        return False, "network audit: networking import(s) %s" % ", ".join("%s: %s" % x for x in net[:5]), suspicious, net, True
    ic = _dict(rt_values["cell.contract.import_check"])
    wall_s, cpu_s = float(ic.get("wall_s", 120)), float(ic.get("cpu_s", 120))
    dirs = rn.prepare("import_check", src)
    before = rn.snapshot((dirs["root"],)) if rn.acct else None
    code = ("import sys, importlib; sys.path.insert(0, sys.argv[1]); sys.path.insert(0, sys.argv[2]); "
            "m = importlib.import_module('%s'); assert callable(m.solve)" % PROGRAM)
    argv = runner.wrap_limits(OOM_FIRST + [sys.executable] + runner.PYTHON_FLAGS + ["-c", code, dirs["lib"], dirs["program"]],
                              cpu_s=cpu_s, user=rn.name, nonet=True)
    try:
        r = subprocess.run(argv, cwd=dirs["program"], env=rn.env(dirs), capture_output=True, text=True, timeout=wall_s,
                           start_new_session=True)
        ok = r.returncode == 0
        why = "ok" if ok else "importing %s.py failed: " % PROGRAM + (r.stderr.strip().splitlines() or ["?"])[-1][:300]
    except subprocess.TimeoutExpired:
        ok, why = False, "importing %s.py took more than %d s" % (PROGRAM, wall_s)
    finally:
        rn.cleanup(dirs, before)
    return ok, why, suspicious, [], True


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
    """Wait, before the first run's network baseline, until the frozen agent account is quiet (its processes and the
    namespace's outbound counters unchanged for `quiet_s`; the sibling's rule). Returns (seconds waited, counter
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


# ------------------------------------------------------------------------------------------------- the runs
class Context(object):
    """What every site's run needs (built once per verify)."""

    def __init__(self, rn, privileged, runtime, config, launcher):
        from cabincell import sensors_render as SR
        from cabincell import server as SV
        from cabincell.wire_guard import GuardedProcess
        self.rn, self.privileged, self.runtime, self.config = rn, privileged, runtime, config
        self.launcher = launcher
        self.SR, self.SV, self.GuardedProcess = SR, SV, GuardedProcess
        # the per-site gate: at least this many calls that advanced the simulated time (sim_time_calls)
        self.min_time_calls = int(runtime["values"]["cell.contract.min_steps_per_site"])
        self.max_reruns = int(runtime["values"]["cell.fault_rerun"])


def _close_cell(server):
    try:
        server.close()                       # the renderer's twin and its MjrContext (the GL context is per process)
    except Exception:  # noqa: BLE001 - the cell's teardown is never a scoring input
        pass
    server.cell = None
    server.renderer = None
    gc.collect()


def run_site(ctx, site, tag):
    """One run of the program on one hidden site: a fresh cell with the frame renderer, the program on the wire as the
    runner account, the cell's record. A failure on the grader's side (the cell's build, the program's launch, the
    wire, the record) is a task error (`fail`), never the agent's zero; whatever the program does ends at most its own
    run. Returns {record, end_reason, ...} (not graded yet: a run in which the cell failed is not counted)."""
    rn = ctx.rn
    t0 = time.time()
    try:
        server = ctx.SR.make_server(privileged=ctx.privileged, reference_dir=REFERENCE_DIR if ctx.privileged else None, log=log)
        server.reset(site["path"], seed=site["seed"], label=tag)
    except Exception:  # noqa: BLE001 - the cell depends on the stored package and the task's files only
        fail("%s: the cell could not be built\n%s" % (tag, traceback.format_exc()))
    build_wall = time.time() - t0
    b = server.cell.budgets
    program_cap, silence = float(b.program_compute_s), float(b.silence_s)
    record_dir(LOGS)
    dirs = rn.prepare(tag, program_source())          # an opaque tag: the hidden site never reaches a path the program sees
    before = rn.snapshot((dirs["root"],)) if rn.acct else None
    argv = OOM_FIRST + ctx.launcher(sys.executable, dirs["lib"], dirs["program"], PROGRAM)
    proc, reason = None, None
    sampler = netaudit.SocketSampler(uid=rn.uid if rn.acct else None, pids_fn=None if rn.acct else (lambda: []),
                                     period_s=SOCKET_SAMPLE_S)
    counters = netaudit.NetCounters()
    t1 = time.time()
    with counters, sampler:
        try:
            try:
                proc = ctx.GuardedProcess(argv, cwd=dirs["program"], user=rn.name, env=rn.env(dirs), cpu_s=rn.cpu_s,
                                          stderr_path=os.path.join(LOGS, "%s.log" % tag), nonet=True)
            except Exception:  # noqa: BLE001
                fail("%s: the program could not be launched\n%s" % (tag, traceback.format_exc()))
            try:
                reason = wire.serve_episode(server, proc, program_cap, silence, log=log)
            except Exception:  # noqa: BLE001 - the wire turns everything the program does into an end reason
                fail("%s: the wire failed\n%s" % (tag, traceback.format_exc()))
        finally:
            if proc is not None:
                proc.terminate()
            rn.cleanup(dirs, before)          # every process of the runner account is gone before the counters are read
    rc = proc.proc.returncode
    if server.done_reason in ("eof", None) and rc not in (None, 0):   # the cell's own wire driver (server.run_wire)
        server.done_reason = "crashed"
    try:
        record = server.finalize(exit_code=rc)
    except Exception:  # noqa: BLE001 - finalize() runs the settle and writes the record; it does not raise by design
        fail("%s: the run's record could not be made\n%s" % (tag, traceback.format_exc()))
    missing = [k for k in RECORD_REQUIRED if not isinstance(record, dict) or k not in record]
    if missing:
        fail("%s: the run's record lacks %s" % (tag, ", ".join(missing)))
    frame_times = list(getattr(server.renderer, "frame_wall_s", None) or [])
    out = dict(record=record, tag=tag, end_reason=reason, done_reason=server.done_reason, exit_code=rc,
               reference_ran=bool(ctx.privileged and server.reference_result is not None),
               build_wall_s=build_wall, run_wall_s=time.time() - t1, program_used_s=_num(getattr(server, "program_used", 0.0)),
               cell_work_s=_num(getattr(server, "work_wall", 0.0)), internal_errors=int(getattr(server, "internal_errors", 0)),
               remote_sockets_seen=int(sampler.max_seen), socket_samples=int(sampler.samples),
               net_outbound=int(counters.total), net_delta=dict(counters.delta),
               colour_frame_s=[s for c, s in frame_times if c in ("FIC", "FWC", "AWC")],
               depth_frame_s=[s for c, s in frame_times if c in ("FDC", "ADC")])
    _close_cell(server)
    del server
    gc.collect()
    out["site_wall_s"] = time.time() - t0
    return out


def _cell_failed(run):
    """What went wrong on the cell's side of a run (not the program's doing): [] when nothing. The record's exit
    `cell_error` (an internal error that ended the run: the cell's server ends a run on an exception in an action call,
    a render failure included) or any internal error of the cell during the run (every argument is checked at the
    call boundary first, cabincell/api.py: a request the checks refuse answers INVALID_ARGS, never an exception)."""
    rec = run["record"]
    out = []
    if _dict(rec.get("times")).get("exit") == "cell_error":
        out.append("cell_error")
    if max(int(_num(_dict(rec.get("metrics")).get("internal_errors"), 0)), int(run.get("internal_errors", 0))) > 0:
        out.append("internal_error")
    return out


def _site_with_retry(ctx, site, metrics):
    """One site; when the cell failed in its run, the site is run once more from scratch (a fresh cell, a fresh
    program) and the second run counts. At most ctx.max_reruns runs of a verify are run again (the verifier's timeout
    is derived for one): a second cell failure -- on the rerun of the same site or on any later site -- is the
    grader's failure (a task error)."""
    tag = site["tag"]
    run_a = run_site(ctx, site, tag)
    why = _cell_failed(run_a)
    if not why:
        return run_a
    if metrics["site_reruns"] >= ctx.max_reruns:
        fail("%s: the cell failed (%s) in a second run of this verify; at most %d run is run again per verify"
             % (tag, ", ".join(why), ctx.max_reruns))
    log("[grader]   %s: the cell failed (%s): the site is run once more" % (tag, ", ".join(why)))
    metrics["site_reruns"] += 1.0
    run_b = run_site(ctx, site, tag + "r")
    why_b = _cell_failed(run_b)
    if why_b:
        fail("%s: the cell failed in both runs of the site (%s; then %s)" % (tag, ", ".join(why), ", ".join(why_b)))
    run_b["retried_after"] = why
    return run_b


def grade_site(ctx, site, run):
    """The site's verdict (cabincell.grade.grade on the counted run's record); a grader error is a task error."""
    from cabincell import server as SV
    try:
        site_d, _members, sha = SV.open_site(site["path"])
    except Exception as e:  # noqa: BLE001
        fail("%s: the site package could not be read: %s" % (site["tag"], e))
    if sha != site["sha256"]:
        fail("%s: the site package changed during the verify" % site["tag"])
    try:
        g = GR.grade(run["record"], site_d, site["ref_energy_wh"], ctx.config, ctx.runtime)
    except Exception:  # noqa: BLE001 - a site without its hidden truth, or a grader defect: never the agent's zero
        fail("%s: the run could not be graded\n%s" % (site["tag"], traceback.format_exc()))
    sc = g.get("score")
    if isinstance(sc, bool) or not isinstance(sc, (int, float)) or not math.isfinite(float(sc)) or not 0.0 <= float(sc) <= 1.0:
        fail("%s: the site score %r is not a finite number in [0, 1]" % (site["tag"], sc))
    return g


def site_summary(site, run, g):
    """The maintainers' summary of one counted run (episodes.json; never a scoring input except `score`)."""
    rec = run["record"]
    m = _dict(rec.get("metrics"))
    t = _dict(rec.get("times"))
    n_time, _t_end = sim_time_calls(rec)       # the per-site gate's count (_main_locked); program_end_s is its clock
    return dict(tag=site["tag"], label=site["label"], index=site["index"], package_sha256=site["sha256"], site_id=rec.get("site_id"),
                time_calls=int(n_time),
                score=float(g["score"]), raw=g.get("raw"), R=g.get("R"), items=g.get("items"), binary=g.get("binary"),
                exit=t.get("exit"), program_end_s=t.get("program_end"), end_s=t.get("end"), end_reason=run["end_reason"],
                done_reason=run["done_reason"], exit_code=run["exit_code"], steps=int(_num(m.get("calls_step"), 0)),
                reference_ran=run["reference_ran"], energy_wh=rec.get("energy_wh"), ref_energy_wh=site["ref_energy_wh"],
                budgets_used=rec.get("budgets_used"), hashes=rec.get("hashes"), run=rec.get("run"),
                cell_cpu_per_sim_s=m.get("cpu_per_sim_s"), cell_rss_mb=m.get("rss_mb"), build_wall_s=run["build_wall_s"],
                run_wall_s=run["run_wall_s"], site_wall_s=run.get("site_wall_s"), program_used_s=run["program_used_s"],
                cell_work_s=run["cell_work_s"], colour_frame_p95_s=_p95(run["colour_frame_s"]),
                depth_frame_p95_s=_p95(run["depth_frame_s"]), frames_rendered=len(run["colour_frame_s"]) + len(run["depth_frame_s"]),
                remote_sockets_seen=run["remote_sockets_seen"], socket_samples=run["socket_samples"],
                net_outbound=run["net_outbound"], retried_after=run.get("retried_after"),
                grade_metrics={k: v for k, v in _dict(g.get("metrics")).items() if not str(k).startswith("cell_")})


# ------------------------------------------------------------------------------------------------- the verdict
def write_verdict(rewards, metrics, contract_why=None, zero_reason=None):
    """One gated `reward` in the benchmark kit's envelope (robotics_grader.ratio_rewards against the recorded reference
    value, or robotics_grader.zero_verdict for an absent submission); the gate, the criterion and every statistic go to
    `metrics` (finite numbers only). In the engine (`ALE_VERIFY_PYTHON` set) the kit writes reward.json and
    verification.json ('.' and '/' in a metric name replaced by '_'); elsewhere the envelope goes to verdict.local.json
    in the stage folder."""
    from robotics_grader import ratio_rewards, read_anchor, zero_verdict
    from robotics_grader import write_verdict as kit_write_verdict
    overall = float(rewards["caretaking_score"])
    anchor, meta = read_anchor()
    extra = finite_metrics(metrics)
    extra.update({k: float(rewards[k]) for k in GATES + CRITERIA})
    text = ("contract_ok %.0f (caretake.py parses, defines solve, passes the route gate, the parser and network audits "
            "and the import check, and advances the simulated time on every site; no late write, no surviving process); "
            "caretaking_score %.4f (contract_ok x the mean over the hidden sites of the site score R x (D + U1 + H + U2 + "
            "V + K + L + E): diagnosis 0.14, removal 0.10, docking 0.10, hold 0.14, ventilation 0.08, the sheet fixed "
            "on a holding screen 0.18, the pouches on pouch pads 0.18, economy 0.08; R 1.0, else 0.85)") % (
                rewards["contract_ok"], rewards["caretaking_score"])
    if zero_reason:
        verdict = zero_verdict(zero_reason, anchored=anchor is not None, extra=extra)
        text = verdict.get("diagnostic") or text
    else:
        verdict = ratio_rewards(overall, anchor, higher_is_better=meta.get("direction", "higher") == "higher",
                                metric_name=meta.get("metric", "caretaking_score"), extra=extra,
                                full_at=meta.get("full_at", 1.0))
        if contract_why and not rewards["contract_ok"]:
            text += "; the contract failed: %s" % str(contract_why)[:300]
    recorded = {}
    for k, v in sorted(_dict(verdict.get("metrics")).items()):
        try:
            v = float(v)
        except (TypeError, ValueError):
            continue
        if math.isfinite(v):
            recorded[k.replace(".", "_").replace("/", "_") if os.environ.get("ALE_VERIFY_PYTHON") else k] = v
    if os.environ.get("ALE_VERIFY_PYTHON"):
        kit_write_verdict({"rewards": verdict["rewards"], "metrics": recorded}, diagnostic=text)
    else:
        kit_write_verdict({"rewards": verdict["rewards"], "metrics": recorded},
                          path=os.path.join(STAGE_DIR, "verdict.local.json"), diagnostic=text)
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
    rewards = dict((k, 0.0) for k in GATES + CRITERIA)
    metrics = {"grader_exception": 0.0, "root_payload_ok": 0.0, "agent_deprivileged": 0.0, "runner_uid": -1.0,
               "privileged_mode": 0.0, "oracle_token_mismatch": 0.0, "runner_processes_killed": 0.0,
               "runner_processes_left": 0.0, "runner_unclean_runs": 0.0, "stray_files_removed": 0.0, "suspicious_lines": 0.0,
               "remote_sockets_seen": 0.0, "net_outbound_events": 0.0, "network_imports": 0.0, "site_reruns": 0.0,
               "submission_present": 0.0, "oom_first": 1.0 if OOM_FIRST else 0.0}
    for k in ("stage_locked", "stage_parent_locked", "stage_lock_latency_ms", "agent_processes_killed", "agent_processes_left",
              "submission_late_writes", "late_writes_quarantined", "late_write_gate_active", "submission_snapshot"):
        metrics[k] = float(EARLY.get(k, -1.0 if k == "stage_lock_latency_ms" else 0.0))
    try:
        try:
            _main_locked(rewards, metrics)
        except SystemExit:
            raise
        except Exception:  # noqa: BLE001 - a grader defect must say so (a task error, exit 1)
            log("[grader] EXCEPTION\n" + traceback.format_exc())
            metrics["grader_exception"] = 1.0
            raise
    finally:
        _restore_modes()
        if FREEZER[0] is not None:
            FREEZER[0].stop()                       # after the verdict (written by _main_locked); the frozen stay stopped


def _main_locked(rewards, metrics):
    t_start = time.time()
    RT, C = load_inputs()
    V = RT["values"]
    n = int(V["layout.sites"])
    metrics["episodes"] = float(n)
    bad_pins = pins_ok(V)
    if bad_pins:
        fail("the cell's numeric pins differ from the runtime key cell.recipe.env (tests/test.sh exports them): %s"
             % ", ".join("%s=%r (want %r)" % (k, os.environ.get(k), V["cell.recipe.env"][k]) for k in bad_pins))
    why = blas_policy()
    if why:
        fail("the BLAS kernel pin is incomplete (tests/test.sh exports OPENBLAS_CORETYPE together with OPENBLAS_L2_SIZE): " + why)
    t_blas = time.time()
    why = blas_selftest()
    if why:
        fail("the BLAS self-test failed in the cell's environment: " + why)
    log("[grader] BLAS kernel pin complete; self-test ok (%.1f s)" % (time.time() - t_blas))
    expected = load_json(os.path.join(STAGE_DIR, "given_hashes.json"))
    ok_root, problems = root_payload_ok(expected)
    metrics["root_payload_ok"] = 1.0 if ok_root else 0.0
    if not ok_root:
        fail("the root-owned copy of the givens at %s is unusable: %s" % (ROOT_PAYLOAD, "; ".join(problems[:5])))
    sites = hidden_set(n)
    metrics.update(assets_ok())
    rn = Runner(V["cell.budget.program_compute_s"])
    if FREEZER[0] is not None and rn.acct and rn.uid == FREEZER[0].uid:
        FREEZER[0].stop()        # no `nobody` in the image: the runner fell back to the agent account, whose programs must run
        log("[grader] runner account = the agent account: the verify-stage freezer is stopped")
    metrics["runner_uid"] = float(rn.uid) if rn.acct else -1.0
    metrics["agent_deprivileged"] = 1.0 if rn.acct else 0.0
    nn_ok, nn_why = nonet.selftest(wrap=lambda a: runner.wrap_limits(a, cpu_s=60, user=rn.name),
                                   env=runner.child_env("/tmp", "/tmp", "/nonexistent", ROOT_GIVEN))
    metrics["nonet_selftest_ok"] = 1.0 if nn_ok else 0.0
    if not nn_ok:
        fail("the no-network filter for the program does not work here: " + nn_why)
    privileged, mismatch = token_state()
    metrics["privileged_mode"] = 1.0 if privileged else 0.0
    metrics["oracle_token_mismatch"] = 1.0 if mismatch else 0.0
    ok, why, suspicious, net, present = contract_check(V, rn)
    metrics["submission_present"] = 1.0 if present else 0.0
    if ok and EARLY.get("reasons"):
        ok, why = False, "; ".join(EARLY["reasons"])
    metrics["suspicious_lines"] = float(suspicious)
    metrics["network_imports"] = float(len(net))
    log("[grader] contract_ok=%s (%s); privileged %s; runner %s; %d hidden sites; lock latency %.0f ms" % (
        ok, why, privileged, rn.name or "self", n, metrics["stage_lock_latency_ms"]))
    if not ok:
        rewards["contract_ok"] = 0.0
        _finish_stats(rn, metrics, t_start)
        write_verdict(rewards, metrics, contract_why=why, zero_reason=None if present else why)
        return
    settle_s, settle_seen = _settle_desktop()
    metrics["desktop_settle_s"] = round(settle_s, 1)
    metrics["net_settle_events"] = float(settle_seen)
    sys.path.insert(0, os.path.join(ROOT_GIVEN, "lib"))           # the root-owned, hash-checked client library
    from cabinkit import client as CK
    ctx = Context(rn, privileged, RT, C, CK.launcher_argv)
    metrics["python_flags_as_runtime"] = 1.0 if " ".join(CK.PYTHON_FLAGS) == str(V["cell.contract.python_flags"]) else 0.0
    summaries = []
    time_ok_all = True
    for i, site in enumerate(sites):
        log("[grader] site %d/%d (%s)" % (i + 1, n, site["tag"]))
        run = _site_with_retry(ctx, site, metrics)
        g = grade_site(ctx, site, run)
        s = site_summary(site, run, g)
        # the per-site gate: calls that advanced the simulated time, step() or any other (blocking)
        # call, counted from the record (sim_time_calls); the oracle's reference_run makes its calls through the same
        # call boundary, so they are in its record like a program's
        s["time_ok"] = s["time_calls"] >= ctx.min_time_calls
        time_ok_all = time_ok_all and s["time_ok"]
        try:                                         # the maintainers' record; never a scoring input
            record_dir(OUT)
            ctx.SV.write_record(os.path.join(OUT, "%s.record.json" % run["tag"]), run["record"])
        except Exception:  # noqa: BLE001
            log("[grader] %s: record not written" % run["tag"])
        run["record"] = None
        log("[grader]   score %.3f items %s exit %s (%s) end %.1f s, program end %.1f s, calls advancing the time %d "
            "(step() %d), wall %.0f s (build %.1f s, cell %.0f s, program %.1f s)" % (
                s["score"], s["items"], s["exit"], s["end_reason"], _num(s["end_s"]), _num(s["program_end_s"]),
                s["time_calls"], s["steps"], _num(s["site_wall_s"]), s["build_wall_s"], s["cell_work_s"], s["program_used_s"]))
        summaries.append(s)
        metrics["remote_sockets_seen"] = max(metrics["remote_sockets_seen"], float(s["remote_sockets_seen"]))
        metrics["net_outbound_events"] += float(s["net_outbound"])
        for ck, cv in (run.get("net_delta") or {}).items():
            metrics["net_" + ck] = metrics.get("net_" + ck, 0.0) + float(max(0, _num(cv)))
    vals = [float(s["score"]) for s in summaries]           # each checked finite and in [0, 1] (grade_site)
    # the metric keeps its first name steps_gate; it is the gate on simulated time
    steps_gate = 1.0 if time_ok_all else 0.0
    clean_gate = 1.0 if rn.unclean_runs == 0 else 0.0
    # the gate counts only what belongs to the runner account (its remote inet sockets; under the seccomp filter it
    # cannot open one); the namespace's counters (loopback and every account included) stay statistics
    net_gate = 1.0 if metrics["remote_sockets_seen"] <= 0 else 0.0
    gate = steps_gate * clean_gate * net_gate
    rewards["contract_ok"] = gate
    contract_why = None
    if not steps_gate:
        contract_why = "fewer than %d call(s) that advanced the simulated time on site(s) %s" % (
            ctx.min_time_calls, ", ".join(s["tag"] for s in summaries if not s["time_ok"]))
    elif not clean_gate:
        contract_why = "a process of the program outlived its run"
    elif not net_gate:
        contract_why = "the program held a remote inet socket during a run (%d seen)" % metrics["remote_sockets_seen"]
    if contract_why:
        log("[grader] contract_ok 0: " + contract_why)
    elif metrics["net_outbound_events"] > 0:
        log("[grader] note: %d namespace-wide outbound counter increments during the runs (statistic only)" % metrics["net_outbound_events"])
    rewards["caretaking_score"] = float(max(0.0, min(1.0, GR.mean_score(vals, gate > 0.0))))
    metrics["steps_gate"], metrics["clean_gate"], metrics["net_gate"] = steps_gate, clean_gate, net_gate
    metrics["site_score_mean"] = _mean(vals)
    for s in summaries:
        metrics["site_score." + s["tag"]] = s["score"]
        metrics["steps." + s["tag"]] = float(s["steps"])                 # literal step() calls (statistic)
        metrics["time_calls." + s["tag"]] = float(s["time_calls"])       # calls that advanced the simulated time (the gate)
    for k in GR.ITEMS:
        metrics["item_mean." + k] = _mean([_num(_dict(s["items"]).get(k)) for s in summaries])
    for k in ("U1", "H", "U2", "V", "K", "R", "D_inlet", "D_item", "D_onset", "E_gate"):
        metrics["rate." + k] = _mean([_num(_dict(s["binary"]).get(k)) for s in summaries])
    metrics["R_mean"] = _mean([_num(s["R"]) for s in summaries])
    exits = [str(s["exit"]) for s in summaries]
    for r in sorted(set(exits)):
        metrics["exit_rate." + r] = float(sum(1 for x in exits if x == r)) / max(1, len(exits))
    for key in ("program_end_s", "end_s", "energy_wh", "build_wall_s", "run_wall_s", "site_wall_s", "program_used_s",
                "cell_work_s", "cell_cpu_per_sim_s", "cell_rss_mb", "colour_frame_p95_s", "depth_frame_p95_s"):
        metrics[key + "_mean"] = _mean([_num(s.get(key)) for s in summaries])
    for key in ("cell_cpu_per_sim_s", "colour_frame_p95_s", "depth_frame_p95_s", "site_wall_s", "cell_rss_mb"):
        metrics[key + "_max"] = max([_num(s.get(key)) for s in summaries] or [0.0])
    for key in ("colour_frames", "depth_frames", "capture_calls", "light_requests", "forbidden_calls"):
        metrics["budget_mean." + key] = _mean([_num(_dict(s["budgets_used"]).get(key)) for s in summaries])
    try:                                            # the maintainers' record; never a scoring input
        record_dir(OUT)
        with open(os.path.join(OUT, "episodes.json"), "w") as f:
            json.dump(summaries, f, indent=1, default=_record_value)
    except Exception:  # noqa: BLE001
        log("[grader] episodes.json not written\n" + traceback.format_exc())
    _finish_stats(rn, metrics, t_start)
    overall = write_verdict(rewards, metrics, contract_why=contract_why)
    log("[grader] rewards %s overall %.4f (verify wall %.0f s)" % ({k: round(v, 4) for k, v in rewards.items()}, overall,
                                                                 metrics["verify_wall_s"]))


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
