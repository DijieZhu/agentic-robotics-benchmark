"""Grader of humanoid_final_assembly_25dof_ur10e (a UR10e-class arm with a wide parallel gripper builds a humanoid robot
from five modules onto its base, which a stand holds: two legs, two arms, the head; camera-only): runs as ROOT in the
verify stage.

The agent left /home/user/submission/assemble.py (solve(cell): a program that builds the robot with the cell's arm and
gripper). Verification is separate (task.toml): this runs in a fresh verifier sandbox from the task image, into which
the engine restored only the submission (root-owned). This grader:
  0. FIRST, stdlib only (`_early_lockdown`): records the upload time, freezes and kills every process of the agent
     account and keeps it stopped until the verdict is written (`arm.verify_freeze`), locks the stage, snapshots the
     submission root-owned, gates late writes, quarantines late entries elsewhere;
  1. checks the root-owned copy of the givens at /opt/arb against given_hashes.json (tests/test.sh builds it from the
     image's task assets and tests/setup_payload/; a mismatch is a task error), then unpacks the cell's hidden kit (gt/:
     the kit index, the collision model, the mating graph, the fixtures' design and meshes, the frozen numbers, the
     render twin's body map; never in the repository) into this root-only stage folder (`fetch_hidden_gt`): from a
     maintainer's local copy assets/gt.tar.xz when it is present (an engine asset root), else downloaded from the
     benchmark's asset dataset (grader_config.json "hidden_gt"); either way its sha256 and its file count must be the
     pinned ones; cell_roots.bind() points the cell's code at gt/ and the given robot models;
  2. checks that netguard.py holds in this sandbox and puts a root-owned copy where the runner account can read it:
     every agent process starts through it (no AF_INET / AF_INET6 socket); the wire to the cell is a pipe pair;
  3. gates: contract_ok = the submission's size (summed before anything is copied or parsed), assemble.py is a regular
     file that parses and defines solve, the route gate, the code check (a .py file that the parser refuses for a
     reason other than a syntax error, that is not UTF-8 text or that declares another source encoding fails it; a file
     with a syntax error is not read; netaudit's static import scan), the module imports under the run regime, >= 1 step call, the network check at run time (no socket
     of the runner account in /proc/net/{tcp,udp}{,6} in any sample of an episode, and the socket refusal active; the
     namespace's outbound counters are recorded, not judged), no late write, no surviving process;
  4. the oracle token: submission/.oracle_token whose sha256 equals hidden/oracle_token.sha256 switches the cells to
     the privileged mode (get_state_privileged, describe_privileged; stat privileged_mode) and to the nominal layout
     (grader_config.json oracle_nominal_layout); a wrong token is ignored (stat oracle_token_mismatch);
  5. runs the program once per episode (grader_config.json episodes: 3) on the cell (arm.assembly_humanoid.episode:
     each episode a FRESH layout drawn from OS entropy, its seed never shown and never stored where the program can
     read it) as `nobody` from a read-only copy, launched by arm.assembly_humanoid.client; the cell's two cameras render
     through the given render layer in a worker process (render_seam/twin_worker.py): a failed frame is retried once
     on a fresh worker, then a control frame is rendered on another; a failed control frame is a task error;
  6. criterion: assembly_score = contract_ok x the mean over the episodes of stage_e x (1, or
     scoring.safety_violation_factor when the episode has a counted safety violation), stage_e = the weight of the
     scope's mates LOCKED at their design bolt position on a module that was not scrapped / the scope's total weight;
     the scope is grader_config.json scope.counted_mates (the release state counts all five mates, W = 7.0);
  7. (a record only) every episode's summary under /opt/arb-verify/out (root-only).
Untouched sandbox: no assemble.py -> contract_ok 0 -> 0. Reference: the token + the reference program -> 1.
"""

import ast
import codecs
import io
import json
import math
import os
import pwd
import re
import resource
import shutil
import stat as statmod
import subprocess
import sys
import tempfile
import threading
import time
import tokenize
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
os.environ.setdefault("ARB_AGENT_RENDER_DIR", os.path.join(ROOT_PAYLOAD, "render"))
SNAPSHOT = os.path.join(RUNS_ROOT, "_submission_snapshot")
QUARANTINE = os.path.join(RUNS_ROOT, "_quarantine")
STICKY_DIRS = [d for d in ("/tmp", "/dev/shm", "/var/tmp", "/run/lock") if os.path.isdir(d)]
WATCH_ROOTS = STICKY_DIRS + [HOME]
LATE_GATE_MAX_GAP_S = 600.0

from arm.assembly_cell import runner  # noqa: E402  (stdlib only)
from arm.assembly_six import lockdown  # noqa: E402  (stdlib only)
from arm import verify_freeze  # noqa: E402  (stdlib only)

FREEZER = [None]


def log(msg):
    print(msg, file=sys.stderr, flush=True)


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
    them; empty when every one is stopped."""
    deadline = time.time() + float(bound_s)
    while True:
        FREEZER[0].freeze_now()
        time.sleep(0.05)
        running = [pid for pid in lockdown.live_uid_processes(uid) if _proc_state(pid) not in ("T", "t", "X")]
        if not running or time.time() > deadline:
            return running


def submission_bytes(src):
    """the submission's size as the route gate sums it: every file's lstat size under `src`"""
    total = 0
    for dp, _dn, fns in os.walk(src):
        for fn in fns:
            try:
                total += os.lstat(os.path.join(dp, fn)).st_size
            except OSError:
                continue
    return total


def route_limit_bytes(cfg=None):
    """grader_config.json route_gate.max_submission_mb in bytes (read from the stage folder when no cfg is given)"""
    try:
        if cfg is None:
            with open(os.path.join(STAGE_DIR, "grader_config.json")) as f:
                cfg = json.load(f)
        return float((cfg.get("route_gate") or {}).get("max_submission_mb", 60)) * 1024 * 1024
    except (OSError, ValueError, TypeError, AttributeError):
        return 60.0 * 1024 * 1024


def snapshot_submission(src, dst, max_bytes):
    """(copied, total): the root-owned snapshot of `src` into `dst`, made only when the summed size is at most
    `max_bytes`; a larger submission is not copied at all (the route gate then refuses it)"""
    total = submission_bytes(src)
    if total > max_bytes:
        return False, total
    lockdown.snapshot_dir(src, dst)
    return True, total


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
    FREEZER[0] = verify_freeze.AgentFreezer.for_user(AGENT_USER).start()
    if out["agent_processes_left"]:
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
        copied, total = snapshot_submission(SUBMISSION, SNAPSHOT, route_limit_bytes())
        out["submission_snapshot"] = 1.0 if copied else 0.0
        out["submission_bytes"] = float(total)
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

import cell_roots  # noqa: E402
from arm.assembly_cell import wire  # noqa: E402
from arm.assembly_humanoid import render_iface as RI  # noqa: E402
from arm.assembly_humanoid.client import launcher_argv  # noqa: E402
from arm.assembly_humanoid.wire_guard import GuardedAgentProcess, STDERR_CAP_BYTES, drain_capped  # noqa: E402
from arm.assembly_info import netaudit  # noqa: E402

GT = os.path.join(STAGE_DIR, "gt")
ROBOT_DIR = os.path.join(ROOT_GIVEN, "robot")
HIDDEN = os.path.join(STAGE_DIR, "hidden")
WORKER = os.path.join(STAGE_DIR, "render_seam", "twin_worker.py")
RECORD_ROOT = os.environ.get("ARB_RECORD_ROOT", "/opt/arb-verify")
LOGS = os.path.join(RECORD_ROOT, "logs")
OUT = os.path.join(RECORD_ROOT, "out")
CRITERIA = ["assembly_score"]
GATES = ["contract_ok"]
MATES = ("leg_R", "leg_L", "arm_L", "arm_R", "head")
SUSPICIOUS = ("/opt/ale", "ALE_STAGE_DIR", "ALE_VERIFICATION", "oracle_token", "ALE_VERIFY", "tests/gt", "/opt/ale/tests",
              ".ale-solution", "verifier/gt", "/opt/arb-verify", "twin_map")


def record_dir(path):
    os.makedirs(RECORD_ROOT, exist_ok=True)
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


# ------------------------------------------------------------------------------------------------ the cell's hidden kit
def fetch_hidden_gt(cfg, stage_dir=None, local=None):
    """The cell's hidden kit (gt/) is not in the repository: one tarball, pinned in grader_config.json ("hidden_gt": its
    URL at an asset-dataset commit, its sha256 and its file count). A maintainer's local copy assets/gt.tar.xz in this
    stage folder (an engine asset root) or ARB_HIDDEN_GT_TARBALL is used when present; otherwise it is downloaded here at
    verify start. Either way it must have the pinned sha256, every member must be under gt/ and be a file or a folder,
    and gt/ must hold the pinned number of files: anything else is a task error. Returns (files, seconds, source)."""
    import tarfile
    import urllib.request
    stage_dir = stage_dir or STAGE_DIR
    gt = os.path.join(stage_dir, "gt")
    spec = cfg["hidden_gt"]
    url, want = spec["url"], spec["sha256"]
    tries, backoff = max(1, int(spec.get("tries", 4))), float(spec.get("backoff_s", 5.0))
    t0 = time.time()
    tarball = os.path.join(stage_dir, ".gt.tar.xz")
    part = os.path.join(stage_dir, ".gt.part")
    for p in (part, gt):
        shutil.rmtree(p, ignore_errors=True)
    if os.path.exists(tarball):
        os.remove(tarball)
    local = local if local is not None else os.path.join(stage_dir, "assets", "gt.tar.xz")
    env = os.environ.get("ARB_HIDDEN_GT_TARBALL")
    if env:
        src, source = env, "maintainer copy"
    elif os.path.isfile(local) and not os.path.islink(local):
        src, source = local, "local copy"
    else:
        src, source = tarball, "download"
        for k in range(tries):
            try:
                with urllib.request.urlopen(url, timeout=120) as r, open(tarball, "wb") as fh:
                    for block in iter(lambda: r.read(1 << 20), b""):
                        fh.write(block)
                break
            except Exception as e:  # noqa: BLE001 - retried, then a task error
                log("[grader] the cell's kit: download %d/%d failed: %r" % (k + 1, tries, e))
                if os.path.exists(tarball):
                    os.remove(tarball)
                if k == tries - 1:
                    fail("could not download the cell's kit from %s: %r" % (url, e))
                time.sleep(backoff * (k + 1))
    got = sha256_file(src)
    if got != want:
        fail("the cell's kit (%s) has sha256 %s; grader_config.json pins %s" % (source, got, want))
    os.makedirs(part, mode=0o700)
    with tarfile.open(src, "r:xz") as tf:
        bad = [m.name for m in tf.getmembers() if not (m.isfile() or m.isdir()) or not (m.name == "gt" or m.name.startswith("gt/"))]
        if bad:
            fail("the cell's kit tarball holds unexpected members: %s" % bad[:5])
        tf.extractall(part, filter="data")
    os.rename(os.path.join(part, "gt"), gt)
    os.chmod(gt, 0o700)
    shutil.rmtree(part, ignore_errors=True)
    if os.path.exists(tarball):
        os.remove(tarball)
    n = sum(len(fns) for _dp, _dn, fns in os.walk(gt))
    if n != int(spec["files"]):
        fail("the cell's kit unpacked to %d files; grader_config.json expects %d" % (n, int(spec["files"])))
    dt = time.time() - t0
    log("[grader] the cell's kit (%s): %d files, sha256 %s..., in %.1f s" % (source, n, got[:12], dt))
    return n, dt, source


# ------------------------------------------------------------------------------------------------ netguard
def netguard_folder(stage_dir=None):
    src = os.path.join(stage_dir or STAGE_DIR, "netguard.py")
    try:
        chk = subprocess.run([sys.executable, "-I", "-S", src, "selftest"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             universal_newlines=True, timeout=60)
        ok, text = chk.returncode == 0, chk.stdout.strip()
    except (OSError, subprocess.SubprocessError) as e:
        ok, text = False, repr(e)
    if not ok:
        fail("netguard does not hold in this sandbox: %s" % text[-300:])
    base = "/run" if os.path.isdir("/run") and os.access("/run", os.W_OK) else None
    folder = tempfile.mkdtemp(prefix="arb-netguard-", dir=base)
    os.chmod(folder, 0o755)
    guard = os.path.join(folder, "netguard.py")
    shutil.copyfile(src, guard)
    os.chmod(guard, 0o644)
    return folder, guard


def guarded(guard, program_dir, argv):
    if not guard:
        return list(argv)
    return [sys.executable, "-I", "-S", guard, "exec", program_dir] + list(argv)


# ------------------------------------------------------------------------------------------------ the contract
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


PARSE_ERRORS = (SyntaxError, RecursionError, MemoryError, ValueError, OverflowError)


def _parse_error(e):
    return str(e) if isinstance(e, SyntaxError) else "%s: %s" % (type(e).__name__, str(e)[:200])


UTF8_NAMES = ("utf-8", "utf-8-sig")


def refused_reason(data):
    """None when the bytes of a .py file are read by the code check or skipped as a syntax error, else the reason the
    check fails on them: a byte-order mark at its start (the shared import scan skips such a file while Python imports
    it); a null byte; a declared source encoding other than UTF-8 (tokenize.detect_encoding on the bytes: a cookie in
    the first two lines; an unknown codec counts); bytes that are not UTF-8 text; a parser refusal other than
    SyntaxError (an expression nested or chained beyond the parser's limits). A file with a SYNTAX ERROR is not read
    (Python cannot import it either)."""
    if data.startswith(codecs.BOM_UTF8):
        return "begins with a byte-order mark"
    if b"\x00" in data:
        return "holds a null byte"
    try:
        enc, _lines = tokenize.detect_encoding(io.BytesIO(data).readline)
    except SyntaxError as e:
        head = data.split(b"\n", 2)[:2]
        if any(tokenize.cookie_re.match(ln.decode("latin-1")) for ln in head):
            return "declares a source encoding that is not UTF-8 (%s)" % str(e)[:200]
        return "is not UTF-8 text (%s)" % str(e)[:200]
    if codecs.lookup(enc).name not in UTF8_NAMES:
        return "declares the source encoding %s (only UTF-8 is read)" % enc
    try:
        text = data.decode(enc)
    except UnicodeDecodeError as e:
        return "is not UTF-8 text (%s)" % str(e)[:200]
    try:
        ast.parse(text)
    except SyntaxError:
        return None
    except PARSE_ERRORS as e:
        return "does not parse (%s)" % _parse_error(e)
    return None


def unparsable_files(src, allow_digests=()):
    """[(relpath, reason)] of the .py files under `src` that the code check refuses (refused_reason: a byte-order mark
    at the start, a null byte, another declared source encoding, bytes that are not UTF-8 text, a parser refusal other than a syntax error; a
    file with a syntax error is not read and not refused); the files netaudit.scan_imports reads, chosen as it chooses
    them (symbolic links, unreadable files, __pycache__ and files byte for byte a file of the given library skipped)"""
    allow = set(allow_digests or ())
    out = []
    for dp, dns, fns in os.walk(src):
        dns[:] = [d for d in dns if d != "__pycache__"]
        for fn in sorted(fns):
            if not fn.endswith(".py"):
                continue
            p = os.path.join(dp, fn)
            if os.path.islink(p):
                continue
            try:
                with open(p, "rb") as f:
                    data = f.read()
            except OSError:
                continue
            if hashlib.sha256(data).hexdigest() in allow:
                continue
            why = refused_reason(data)
            if why is not None:
                out.append((os.path.relpath(p, src), why))
    return out


def library_digests():
    out = set()
    for pkg in ("arm", "common", "render"):
        d = os.path.join(ROOT_GIVEN, "lib", pkg)
        for dp, _dn, fns in os.walk(d):
            for fn in fns:
                if fn.endswith(".py"):
                    out.add(sha256_file(os.path.join(dp, fn)))
    return out


class Runner(object):
    def __init__(self, cfg):
        self.acct = runner.runner_account(cfg.get("runner_account", "nobody"), AGENT_USER)
        self.name, self.uid, self.gid = self.acct if self.acct else (None, None, None)
        self.lib_dirs = {p: os.path.join(ROOT_GIVEN, "lib", p) for p in ("arm", "common", "render")}
        self.cpu_s = float(cfg.get("cpu_cap_factor", 4.0)) * float(cfg["wall_cap_s"])
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
        for k in ("OPENBLAS_CORETYPE", "OPENBLAS_L2_SIZE", "NPY_DISABLE_CPU_FEATURES"):
            if os.environ.get(k):
                e[k] = os.environ[k]
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


IMPORT_TAIL_BYTES = 4096


def import_check_run(argv, cwd, env, log_path, timeout_s):
    """(the return code, or None past the time-out; the last line of the program's standard error): the import check's
    process with its standard output discarded and its standard error through the episodes' capped writer
    (wire_guard.drain_capped, STDERR_CAP_BYTES) into `log_path`; only the last IMPORT_TAIL_BYTES bytes are held for
    the reason"""
    log_file = open(log_path, "wb")
    proc = subprocess.Popen(argv, cwd=cwd, env=env, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,  # noqa: S603
                            stderr=subprocess.PIPE, start_new_session=True)
    box = {}

    def drain():
        box.update(drain_capped(proc.stderr.fileno(), log_file, STDERR_CAP_BYTES, tail_bytes=IMPORT_TAIL_BYTES))
    th = threading.Thread(target=drain, daemon=True)
    th.start()
    try:
        rc = proc.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
        rc = None
    th.join(timeout=10.0)
    try:
        proc.stderr.close()
    except OSError:
        pass
    lines = [ln for ln in box.get("tail", b"").decode("utf-8", "replace").strip().splitlines() if ln.strip()]
    return rc, (lines[-1] if lines else "?")


def contract_check(cfg, rn, guard):
    src = program_source()
    max_mb = cfg.get("route_gate", {}).get("max_submission_mb", 60)
    if submission_bytes(src) > route_limit_bytes(cfg):
        return False, "route gate: submission larger than %d MB" % max_mb, 0, []
    path = os.path.join(src, "assemble.py")
    if not os.path.isfile(path) or os.path.islink(path):
        return False, "no assemble.py (regular file) in the submission", 0, []
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            tree = ast.parse(f.read())
    except PARSE_ERRORS as e:
        return False, "assemble.py does not parse: %s" % _parse_error(e), 0, []
    has_solve = any(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == "solve" for n in ast.walk(tree)) or \
        any(isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "solve" for t in n.targets) for n in ast.walk(tree)) or \
        any(isinstance(n, (ast.Import, ast.ImportFrom)) and any((a.asname or a.name) == "solve" for a in n.names) for n in ast.walk(tree))
    if not has_solve:
        return False, "assemble.py defines no solve", 0, []
    problems, suspicious = route_gate(cfg, src)
    if problems:
        return False, "route gate: " + "; ".join(problems[:5]), suspicious, []
    allow = library_digests()
    bad = unparsable_files(src, allow)
    if bad:
        return False, "code check: %s %s%s" % (bad[0][0], bad[0][1], (" and %d more file(s)" % (len(bad) - 1)) if len(bad) > 1 else ""), suspicious, []
    try:
        net = netaudit.scan_imports(src, allow_digests=allow)
    except PARSE_ERRORS as e:
        return False, "code check: a file does not parse (%s)" % _parse_error(e), suspicious, []
    if net:
        return False, "network audit: networking import(s) %s" % ", ".join("%s: %s" % x for x in net[:5]), suspicious, net
    dirs = rn.prepare("import_check", src)
    before = rn.snapshot((dirs["root"],)) if rn.acct else None
    code = ("import sys, importlib; sys.path.insert(0, sys.argv[1]); sys.path.insert(0, sys.argv[2]); "
            "m = importlib.import_module('assemble'); assert callable(m.solve)")
    argv = guarded(guard, dirs["program"], [sys.executable] + runner.PYTHON_FLAGS + ["-c", code, dirs["lib"], dirs["program"]])
    argv = runner.wrap_limits(argv, cpu_s=120, user=rn.name)
    try:
        rc, last = import_check_run(argv, dirs["program"], rn.env(dirs), os.path.join(record_dir(LOGS), "import_check.log"), timeout_s=120)
        if rc is None:
            ok, why = False, "importing assemble.py took more than 120 s"
        else:
            ok, why = rc == 0, ("ok" if rc == 0 else "importing assemble.py failed: " + last[:300])
    finally:
        rn.cleanup(dirs, before)
    return ok, why, suspicious, []


TOKEN_READ_BYTES = 1024


def token_state():
    """(privileged, mismatch): the submission's .oracle_token against hidden/oracle_token.sha256."""
    p = os.path.join(program_source(), ".oracle_token")
    want_p = os.path.join(HIDDEN, "oracle_token.sha256")
    if not os.path.isfile(p) or os.path.islink(p) or not os.path.isfile(want_p):
        return False, False
    try:
        with open(p, "rb") as f:
            tok = f.read(TOKEN_READ_BYTES).strip()
        with open(want_p) as f:
            want = f.read().strip()
    except OSError:
        return False, False
    got = hashlib.sha256(tok).hexdigest()
    return (got == want), (got != want)


# ------------------------------------------------------------------------------------------------ the cell
def render_deadlines(cfg):
    """the render worker's deadlines, bound to the budget's inputs (grader_config.json render_first_frame_s and
    render_frame_s, the budget's first-frame and frame times): the first frame of a worker session max(300 s, 5 x the
    first-frame time), every later frame and the worker's close max(60 s, 10 x the frame time); without those keys the
    floors"""
    return {"first_frame": max(RI.DEADLINE_FLOORS_S["first_frame"], 5.0 * float(cfg.get("render_first_frame_s") or 0.0)),
            "frame": max(RI.DEADLINE_FLOORS_S["frame"], 10.0 * float(cfg.get("render_frame_s") or 0.0))}


def cell_config(cfg, scratch):
    """the cell's configuration of record (arm/assembly_humanoid/cell_config.json) with the grader's overrides: the
    render worker of the given layer (its scratch folder under the root-only record root, its deadlines) and the
    budgets"""
    from arm.assembly_humanoid import config as CF
    gl = {"python": sys.executable, "worker": WORKER, "scratch": scratch,
          "env": {"ARB_TWIN_STAGE": STAGE_DIR, "ARB_TWIN_GT": GT, "ARB_GIVEN_DIR": ROOT_GIVEN,
                  "ARB_TWIN_LIB": os.path.join(ROOT_GIVEN, "lib"), "ARB_AGENT_RENDER_DIR": os.path.join(ROOT_PAYLOAD, "render"),
                  "PYTHONDONTWRITEBYTECODE": "1", "LP_NUM_THREADS": str(cfg.get("render_threads", 8)),
                  "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
          "deadlines_s": render_deadlines(cfg)}
    if os.environ.get("ARB_TWIN_SAVE_STATE") == "1":
        gl["env"]["ARB_TWIN_SAVE_STATE"] = "1"
    over = json.loads(json.dumps(cfg.get("cell_overrides") or {}))
    over.setdefault("cameras", {})["given_layer"] = gl
    c = CF.load_config(overrides=over)
    # the grader's worker block REPLACES any such block of the cell configuration (a deep merge would carry that
    # block's host environment, e.g. its Vulkan driver file, into the worker)
    c["cameras"]["given_layer"] = gl
    return c


def episode_value(summary, scope, factor):
    """(stage_e, value_e, rows): the scope's mates LOCKED at their design bolt position on a module that was not
    scrapped (the coupling's own `counted` reading), their weight over the scope's total weight; value = stage x (1, or
    the factor with a counted safety violation)"""
    recs = {r["id"]: r for r in summary["mates"]}
    if sorted(recs) != sorted(MATES):
        fail("the cell reports mates %s, the task has %s" % (sorted(recs), sorted(MATES)))
    W = float(sum(float(recs[m]["weight"]) for m in scope))
    got = float(sum(float(recs[m]["weight"]) for m in scope if recs[m]["counted"]))
    stage = got / W if W > 0 else 0.0
    value = stage * (1.0 if summary["safety_ok"] else factor)
    rows = [{"id": m, "weight": float(recs[m]["weight"]), "in_scope": m in scope, "counted": bool(recs[m]["counted"]),
             "locked": bool(recs[m]["locked"]), "design_ok": recs[m]["design_ok"], "k": recs[m]["k"],
             "void": recs[m]["void"], "state": recs[m]["state"]} for m in MATES]
    return stage, value, rows


def runner_socket_count(uid, proc_net="/proc/net"):
    """the inet sockets (tcp, tcp6, udp, udp6 of the network namespace) whose owner, the uid column, is `uid`"""
    n = 0
    for name in ("tcp", "tcp6", "udp", "udp6"):
        try:
            with open(os.path.join(proc_net, name)) as fh:
                next(fh, None)
                for line in fh:
                    f = line.split()
                    try:
                        n += 1 if len(f) >= 8 and int(f[7]) == int(uid) else 0
                    except ValueError:
                        continue
        except OSError:
            continue
    return n


class RunnerSockets(object):
    """`with RunnerSockets(uid, period_s):` samples runner_socket_count(uid) every period_s (no sampling for uid None:
    a run as the grader's own account); .max_seen, .samples"""

    def __init__(self, uid=None, period_s=0.5, proc_net="/proc/net"):
        self.uid, self.period_s, self.proc_net = uid, float(period_s), proc_net
        self.max_seen = self.samples = 0
        self._stop = threading.Event()
        self._t = None

    def _run(self):
        while True:
            try:
                n = runner_socket_count(self.uid, self.proc_net)
            except Exception:  # noqa: BLE001
                n = 0
            self.samples += 1
            self.max_seen = max(self.max_seen, n)
            if self._stop.wait(self.period_s):
                break

    def __enter__(self):
        if self.uid is not None:
            self._t = threading.Thread(target=self._run, daemon=True)
            self._t.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._t is not None:
            self._t.join(timeout=5.0)
        return False


def network_gate(metrics):
    """1.0 when the run regime's socket refusal is active (netguard) and no sample of any episode saw a socket of the
    runner account (the uid column, or a remote socket among its processes' descriptors); the namespace's outbound
    counters (net_*) are recorded and judge nothing"""
    ok = (float(metrics.get("netguard_active", 0.0)) == 1.0 and float(metrics.get("runner_sockets_seen", 0.0)) <= 0
          and float(metrics.get("remote_sockets_seen", 0.0)) <= 0)
    return 1.0 if ok else 0.0


def run_episode(cfg, cell_cfg, k, seed, rn, privileged, guard, inputs):
    from arm.assembly_humanoid import episode as EP
    t_build = time.time()
    ep = EP.build_episode(seed=seed, nominal=bool(privileged and cfg.get("oracle_nominal_layout", True)), cfg=cell_cfg,
                          privileged=privileged, renderer=str(cfg.get("renderer", "given_layer")),
                          robot_models_dir=ROBOT_DIR, inputs=inputs, log=log)
    build_s = time.time() - t_build
    server = ep.server
    record_dir(LOGS)
    wire_error, t0 = None, time.time()
    dirs = rn.prepare("agent_ep%02d" % k, program_source())
    before = rn.snapshot((dirs["root"],)) if rn.acct else None
    argv = guarded(guard, dirs["program"], launcher_argv(sys.executable, dirs["lib"], dirs["program"]))
    wall_cap = float(cfg["privileged_wall_cap_s"] if privileged else cfg["wall_cap_s"])
    proc = None
    sampler = netaudit.SocketSampler(uid=rn.uid if rn.acct else None, pids_fn=None if rn.acct else (lambda: []),
                                     period_s=float(cfg.get("socket_sample_s", 0.5)))
    counters = netaudit.NetCounters()
    sockets = RunnerSockets(uid=rn.uid if rn.acct else None, period_s=float(cfg.get("socket_sample_s", 0.5)))
    try:
        with counters, sampler, sockets:
            try:
                proc = GuardedAgentProcess(argv, cwd=dirs["program"], user=rn.name, env=rn.env(dirs, None), cpu_s=rn.cpu_s,
                                           stderr_path=os.path.join(LOGS, "agent_ep%02d.log" % k))
                reason = wire.serve_episode(server, proc, wall_cap, float(cfg["silence_s"]), log=log)
            finally:
                if proc is not None:
                    proc.terminate()
                rn.cleanup(dirs, before)
    except RI.RenderInfrastructureError as e:
        fail("episode %d: the cell's camera renderer failed (the frame, its retry on a fresh worker and a control frame "
             "on another): %s" % (k, e))
    except Exception:  # noqa: BLE001
        wire_error = traceback.format_exc()
        reason = "wire_error"
        if not server.done:
            server._end(reason)
    if wire_error:
        log("[grader] episode %d: wire error\n%s" % (k, wire_error))
    summary = server.finalize()
    summary["episode_wall_s"] = time.time() - t0
    summary["build_s"] = build_s
    summary["end_reason"] = reason
    summary["build_report"] = {"times_s": ep.report["times_s"], "rss_kb": ep.report["rss_kb"]}
    summary["remote_sockets_seen"] = int(sampler.max_seen)
    summary["socket_samples"] = int(sampler.samples)
    summary["runner_sockets_seen"] = int(sockets.max_seen)
    summary["runner_socket_samples"] = int(sockets.samples)
    summary["net_outbound"] = int(counters.total)
    summary["net_delta"] = dict(counters.delta)
    del ep, server
    import gc
    gc.collect()
    return summary


# ------------------------------------------------------------------------------------------------ the verdict
def safety_factor(cfg):
    v = (cfg.get("scoring") or {}).get("safety_violation_factor")
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(float(v)) or not 0.0 <= float(v) <= 1.0:
        fail("grader_config.json scoring.safety_violation_factor must be a number in [0, 1]")
    return float(v)


def scope_of(cfg):
    sc = (cfg.get("scope") or {}).get("counted_mates")
    if not isinstance(sc, list) or not sc or any(m not in MATES for m in sc) or len(set(sc)) != len(sc):
        fail("grader_config.json scope.counted_mates must be a non-empty list of distinct mates of %s" % (MATES,))
    return [m for m in MATES if m in sc]


def write_verdict(rewards, metrics, scope):
    from robotics_grader import ratio_rewards, read_anchor
    from robotics_grader import write_verdict as kit_write_verdict
    overall = float(rewards["assembly_score"])
    anchor, meta = read_anchor()
    extra = {}
    for k, v in metrics.items():
        try:
            v = float(v)
        except (TypeError, ValueError):
            continue
        if math.isfinite(v):
            extra[k.replace(".", "_").replace("/", "_")] = v
    extra.update({k: float(rewards[k]) for k in GATES + CRITERIA})
    verdict = ratio_rewards(overall, anchor, higher_is_better=meta.get("direction", "higher") == "higher",
                            metric_name=meta.get("metric", "assembly_score"), extra=extra, full_at=meta.get("full_at", 1.0))
    text = ("contract_ok %.0f (the program imports, exposes solve, passes the route gate and the network audit and steps the "
            "cell; no late write, no surviving process); assembly_score %.4f (episode mean of the counted mates' weight over "
            "the scope's weight (%s) on %d fresh hidden episodes, an episode with a counted safety violation keeping %s %% "
            "of its score)") % (rewards["contract_ok"], rewards["assembly_score"], ", ".join(scope), int(metrics.get("episodes", 0)),
                                "%.10g" % (100.0 * float(metrics.get("safety_violation_factor", 0.0))))
    recorded = {}
    for k, v in sorted(verdict["metrics"].items()):
        try:
            v = float(v)
        except (TypeError, ValueError):
            continue
        if math.isfinite(v):
            recorded[k] = v
    if os.environ.get("ALE_VERIFY_PYTHON"):
        kit_write_verdict({"rewards": verdict["rewards"], "metrics": recorded}, diagnostic=text)
        return overall
    kit_write_verdict({"rewards": verdict["rewards"], "metrics": recorded}, path=os.environ.get("ROBOTICS_GRADER_VERDICT_PATH")
                      or os.path.join(STAGE_DIR, "verdict.local.json"), diagnostic=text)
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


def memory_metrics(metrics):
    """the verify process's own peak, its children's (the agent's programs, the render workers), the container's"""
    metrics["verify_maxrss_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    metrics["children_maxrss_mb"] = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024.0
    for name, key in (("memory.peak", "cgroup_memory_peak_mb"), ("memory.current", "cgroup_memory_current_mb")):
        try:
            with open(os.path.join("/sys/fs/cgroup", name)) as f:
                metrics[key] = int(f.read().strip()) / 1048576.0
        except (OSError, ValueError):
            pass
    try:
        with open("/sys/fs/cgroup/memory.events") as f:
            for ln in f:
                k, v = ln.split()
                if k in ("max", "oom", "oom_kill"):
                    metrics["cgroup_memory_events_" + k] = float(v)
    except (OSError, ValueError):
        pass


def main():
    cfg = load_json(os.path.join(STAGE_DIR, "grader_config.json"))
    rewards = dict((k, 0.0) for k in GATES + CRITERIA)
    metrics = {"grader_exception": 0.0, "episodes": float(cfg.get("episodes", 3)),
               "root_payload_ok": 0.0, "agent_deprivileged": 0.0, "runner_uid": -1.0, "privileged_mode": 0.0, "oracle_token_mismatch": 0.0,
               "runner_processes_killed": 0.0, "runner_processes_left": 0.0, "runner_unclean_runs": 0.0, "netguard_active": 0.0,
               "stray_files_removed": 0.0, "suspicious_lines": 0.0, "agent_wire_errors": 0.0, "remote_sockets_seen": 0.0,
               "runner_sockets_seen": 0.0, "net_outbound_events": 0.0,
               "network_imports": 0.0, "illegal_call_count": 0.0}
    for k in ("stage_locked", "stage_parent_locked", "stage_lock_latency_ms", "agent_processes_killed", "agent_processes_left",
              "submission_late_writes", "late_writes_quarantined", "late_write_gate_active", "submission_snapshot",
              "submission_bytes"):
        metrics[k] = float(EARLY.get(k, -1.0 if k == "stage_lock_latency_ms" else 0.0))
    guard_dir = [None]
    try:
        _main_locked(cfg, rewards, metrics, guard_dir)
    finally:
        if guard_dir[0]:
            shutil.rmtree(guard_dir[0], ignore_errors=True)
        _restore_modes()
        if FREEZER[0] is not None:
            FREEZER[0].stop()


def _main_locked(cfg, rewards, metrics, guard_dir):
    t_start = time.time()
    expected = load_json(os.path.join(STAGE_DIR, "given_hashes.json"))
    ok_root, problems = root_payload_ok(expected)
    metrics["root_payload_ok"] = 1.0 if ok_root else 0.0
    if not ok_root:
        fail("the root-owned copy of the givens at %s is unusable: %s" % (ROOT_PAYLOAD, "; ".join(problems[:5])))
    factor = safety_factor(cfg)
    metrics["safety_violation_factor"] = factor
    scope = scope_of(cfg)
    metrics["scope_mates"] = float(len(scope))
    metrics["scope_release_state"] = 1.0 if bool((cfg.get("scope") or {}).get("release_state")) else 0.0
    gt_files, gt_s, gt_source = fetch_hidden_gt(cfg)
    metrics["hidden_gt_files"] = float(gt_files)
    metrics["hidden_gt_fetch_s"] = float(gt_s)
    metrics["hidden_gt_local"] = 1.0 if gt_source != "download" else 0.0
    cell_roots.bind(GT, ROOT_GIVEN)
    guard_dir[0], guard = netguard_folder()
    metrics["netguard_active"] = 1.0
    rn = Runner(cfg)
    if FREEZER[0] is not None and rn.acct and rn.uid == FREEZER[0].uid:
        FREEZER[0].stop()
        log("[grader] runner account = the agent account: the verify-stage freezer is stopped")
    metrics["runner_uid"] = float(rn.uid) if rn.acct else -1.0
    metrics["agent_deprivileged"] = 1.0 if rn.acct else 0.0
    privileged, mismatch = token_state()
    metrics["privileged_mode"] = 1.0 if privileged else 0.0
    metrics["oracle_token_mismatch"] = 1.0 if mismatch else 0.0
    ok, why, suspicious, net = contract_check(cfg, rn, guard)
    if ok and EARLY.get("reasons"):
        ok, why = False, "; ".join(EARLY["reasons"])
    metrics["suspicious_lines"] = float(suspicious)
    metrics["network_imports"] = float(len(net))
    log("[grader] contract_ok=%s (%s); privileged %s; runner %s under netguard; scope %s" % (
        ok, why, privileged, rn.name or "self", ",".join(scope)))
    if not ok:
        rewards["contract_ok"] = 0.0
        _finish_stats(rn, metrics, t_start)
        write_verdict(rewards, metrics, scope)
        return
    from arm.assembly_humanoid import episode as EP
    scratch = record_dir(os.path.join(RECORD_ROOT, "render"))
    cell_cfg = cell_config(cfg, scratch)
    inputs = EP.load_inputs(cell_cfg)
    n = int(cfg.get("episodes", 3))
    agent, vals, step_calls, illegal = [], [], 0, 0
    for k in range(1, n + 1):
        seed = EP.os_seed()                          # a fresh hidden layout per episode; never shown
        log("[grader] episode %d/%d" % (k, n))
        s = run_episode(cfg, cell_cfg, k, seed, rn, privileged, guard, inputs)
        stage, value, rows = episode_value(s, scope, factor)
        s["stage_scope"], s["episode_value"], s["mate_rows"] = stage, value, rows
        log("[grader]   stage %.3f (%s) safety_ok %s value %.3f sim %.1f s wall %.1f s frames %d end %s scrapped %s "
            "counted violations %s" % (stage, " ".join("%s:%s" % (r["id"], r["state"]) for r in rows), s["safety_ok"], value,
                                        s["sim_time_s"], s["episode_wall_s"], s["camera_frames_total"], s["end_reason"],
                                        sorted(s["scrapped"]), sorted(k for k, v in s["violations_counted"].items() if v)))
        agent.append(s)
        vals.append(value)
        step_calls += s["step_calls"]
        illegal += s["illegal_call_count"]
        metrics["agent_wire_errors"] += 1.0 if s["end_reason"] == "wire_error" else 0.0
        metrics["remote_sockets_seen"] = max(metrics["remote_sockets_seen"], float(s["remote_sockets_seen"]))
        metrics["runner_sockets_seen"] = max(metrics["runner_sockets_seen"], float(s["runner_sockets_seen"]))
        metrics["net_outbound_events"] += float(s.get("net_outbound", 0))
        for ck, cv in (s.get("net_delta") or {}).items():
            metrics["net_" + ck] = metrics.get("net_" + ck, 0.0) + float(max(0, cv))
    steps_gate = 1.0 if step_calls > 0 else 0.0
    clean_gate = 1.0 if rn.unclean_runs == 0 else 0.0
    net_gate = network_gate(metrics)
    gate = steps_gate * clean_gate * net_gate
    rewards["contract_ok"] = gate
    if not net_gate:
        log("[grader] contract_ok 0: a socket of the runner account during an episode, or the socket refusal not active")
    rewards["assembly_score"] = float(max(0.0, min(1.0, gate * float(np.mean(vals)))))
    episode_metrics(agent, metrics)
    metrics["illegal_call_count"] = float(illegal)
    metrics["step_calls"] = float(step_calls)
    record_dir(OUT)
    with open(os.path.join(OUT, "episodes.json"), "w") as f:
        json.dump([{k: v for k, v in s.items() if k not in ("events", "coupling_events")} for s in agent], f, indent=1, default=float)
    _finish_stats(rn, metrics, t_start)
    overall = write_verdict(rewards, metrics, scope)
    log("[grader] rewards %s overall %.4f (verify wall %.0f s)" % ({k: round(v, 4) for k, v in rewards.items()}, overall, metrics["verify_wall_s"]))


def episode_metrics(agent, metrics):
    def mean(xs):
        xs = [float(x) for x in xs]
        return float(np.mean(xs)) if xs else 0.0
    metrics["stage_scope_mean_ungated"] = mean(s["stage_scope"] for s in agent)
    metrics["stage_all_mates_mean"] = mean(s["stage_score"] for s in agent)
    metrics["mates_locked_design_mean"] = mean(s["n_locked_design"] for s in agent)
    metrics["locked_off_mean"] = mean(len(s["locked_off"]) for s in agent)
    metrics["scrapped_units_mean"] = mean(len(s["scrapped"]) for s in agent)
    metrics["safety_violation_rate"] = mean(0.0 if s["safety_ok"] else 1.0 for s in agent)
    for rule in sorted(set(k for s in agent for k in s["violations_counted"])):
        metrics["violation_episodes." + rule] = float(sum(1 for s in agent if s["violations_counted"].get(rule)))
    metrics["mean_sim_time_s"] = mean(s["sim_time_s"] for s in agent)
    metrics["max_sim_time_s"] = max([float(s["sim_time_s"]) for s in agent] or [0.0])
    metrics["mean_episode_wall_s"] = mean(s["episode_wall_s"] for s in agent)
    metrics["max_episode_wall_s"] = max([float(s["episode_wall_s"]) for s in agent] or [0.0])
    metrics["mean_build_s"] = mean(s["build_s"] for s in agent)
    metrics["camera_frames_mean"] = mean(s["camera_frames_total"] for s in agent)
    for cam in ("top", "wrist"):
        metrics["camera_frames_mean." + cam] = mean(s["camera_frames"].get(cam, 0) for s in agent)
    metrics["render_wall_s_mean"] = mean(s["render_wall_s"] for s in agent)
    metrics["render_failures"] = float(sum(int(s["render_failures"]) for s in agent))
    metrics["render_retries"] = float(sum(int(s.get("render_retries", 0)) for s in agent))
    metrics["render_controls"] = float(sum(int(s.get("render_controls", 0)) for s in agent))
    metrics["end_finish_episodes"] = float(sum(1 for s in agent if s["end_reason"] == "finish"))
    mx = {}
    for s in agent:
        for kk, vv in (s["safety"].get("max") or {}).items():
            if isinstance(vv, (int, float)) and not isinstance(vv, bool):
                mx[kk] = max(mx.get(kk, 0.0), float(vv))
    for kk, vv in mx.items():
        metrics["watch_max." + kk] = vv
    for m in MATES:
        metrics["mate_rate." + m] = mean(1.0 if any(r["id"] == m and r["counted"] for r in s["mate_rows"]) else 0.0 for s in agent)
    return metrics


def _finish_stats(rn, metrics, t_start):
    metrics["runner_processes_killed"] = float(rn.killed)
    metrics["runner_processes_left"] = float(rn.left)
    metrics["runner_unclean_runs"] = float(rn.unclean_runs)
    metrics["stray_files_removed"] = float(rn.stray_files)
    metrics["submission_symlinks_skipped"] = float(rn.symlinks_skipped)
    metrics["agent_processes_frozen"] = float(len(FREEZER[0].frozen)) if FREEZER[0] is not None else 0.0
    metrics["verify_wall_s"] = time.time() - t_start
    memory_metrics(metrics)
    shutil.rmtree(RUNS_ROOT, ignore_errors=True)


if __name__ == "__main__":
    main()
