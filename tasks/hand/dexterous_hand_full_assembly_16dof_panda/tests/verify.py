"""Grader of dexterous_hand_full_assembly_16dof_panda (a four-digit, 16-joint robot hand built from its bolt-on
modules onto the palm's bottom shell on a raised fixture, camera-only): runs as ROOT in the verify stage. Built from
the tensegrity strut assembly task's grader (tensegrity_strut_assembly_3bar_panda, itself built on the quadruped
assembly task's: a kit downloaded at verify start); the physics is this task's arm/assembly_hand.

The scored mates are the scope of grader_config.json (every mate of the mating graph: the nine units). The cell's
cameras render through arm/assembly_hand/agent_render.py (grader_config.json camera_layer.enabled; with it false the
cell runs with its cameras off and every get_camera call answers camera_unavailable).

The agent left /home/user/submission/assemble.py (solve(cell): a camera-only program that builds the hand with the
cell's Panda). Verification is separate (task.toml): this runs in a fresh verifier sandbox from the task image, into
which the engine restored only the submission (root-owned); nothing else of the agent's sandbox is here. This grader:
  0. FIRST, stdlib only (`_early_lockdown`, the six-servo pattern; tests/test.sh closed the stage folder already): records
     the upload time, freezes and kills every process of the agent account and keeps it stopped until the verdict is
     written (`arm.verify_freeze`), locks the stage, checks the submission's caps (entries, size, path length) by one
     walk that only stats -- above a cap nothing of it is copied or read and the verdict carries the cap's reason --,
     then snapshots the submission root-owned, gates late writes, quarantines
     late entries elsewhere (written for a verify in the agent's own sandbox; in the separate verifier nothing of the
     agent is left, and they stop the image's own desktop session of the agent account, started with the sandbox, and
     set aside the files that session writes);
  1. checks the root-owned copy of the givens at /opt/arb against given_hashes.json (tests/test.sh builds it from the
     image's task assets and tests/setup_payload/; the Panda model and the runner's library come from it; a mismatch is
     a task error), then fetches the cell's kit (gt/: kit.json, mating_graph.json, joints.json, meshes/; never in the
     image) from the benchmark's asset dataset at the commit pinned in grader_config.json -- or, before it is
     published, from the local copy tests/assets/gt.tar.xz (tests/test.sh sets ARB_HIDDEN_GT_TARBALL) -- checks its
     sha256 and file count and unpacks it into this root-only stage folder (`fetch_hidden_gt`; a failure or a mismatch
     is a task error);
  2. gates: contract_ok = assemble.py is a regular file that parses and defines solve, the route gate (no weight files,
     no training calls, size cap), the network audit (static: no networking import, no computed import; runtime: no
     outbound network activity while the program runs), the module imports under the run regime, >= 1 step call, no late
     write, no surviving process; every .py file is parsed in one child process under the limits of grader_config.json
     parse_limits (the grader itself never parses the submission's sources; a file that does not fit is a contract
     failure with its own reason); per episode `safety` (the cell's four rules: R1 a Panda link other than the two finger
     pad bodies pressing on anything above safety.panda_force_N; R2 a unit loaded above damage.part_force_N for 0.2 s,
     by contact or through its fastening; R3 a product joint against its mechanical stop above stops.torque_Nm for 0.2 s;
     R4 a unit's centre of mass faster than safety.part_speed_m_s) multiplies that episode's score by
     scoring.safety_violation_factor;
  3. the oracle token: submission/.oracle_token whose sha256 equals hidden/oracle_token.sha256 switches the agent's cells to
     the privileged mode (the nominal layout and the call `reference_assemble`; stat privileged_mode); a wrong token is
     ignored (stat oracle_token_mismatch), and so is, without being opened, a token that is not a regular file or is
     larger than 4096 bytes;
  4. draws `episodes` hidden 63-bit episode seeds from OS entropy (hidden/seeds.json pins them) and runs the agent's
     program per episode on the hand cell (`arm.assembly_hand.server.HandInfoServer`: the palm's bottom shell on the
     fixture, the nine loose units in the nests of the digit tray and the parts tray, the fixture and the trays placed
     and turned, the finger chains' and the thumb's angles in their nests and the flat parts' directions drawn per
     episode; camera-only API) as `nobody` from a read-only copy;
  5. criterion: assembly_score = the episode mean of the scope's fastened mates' weight / the scope's total weight
     (from the kit's mating graph), an episode with a safety violation keeping scoring.safety_violation_factor of it,
     times the gates;
  6. (a record only) every agent episode's replay trace under /opt/arb-verify/out/trace/agent_epNN (episode_trace.py;
     root-only, outside /opt/ale, so the verifier sandbox still has it when a run keeps it: `[sandbox_retention]
     verifier = "keep"`).
Untouched sandbox: no assemble.py -> contract_ok 0 -> 0.
Oracle: the token + `reference_assemble` (the privileged reference program inside the cell, on the nominal layout,
after one call per camera) -> every mate of the scope on every episode -> 1.
"""

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
# the scenery of the agent cameras' renderer (arm/assembly_hand/agent_render.py, once the cell has it), from the same root-owned copy
os.environ.setdefault("ARB_AGENT_RENDER_DIR", os.path.join(ROOT_PAYLOAD, "render"))
SNAPSHOT = os.path.join(RUNS_ROOT, "_submission_snapshot")
QUARANTINE = os.path.join(RUNS_ROOT, "_quarantine")
STICKY_DIRS = [d for d in ("/tmp", "/dev/shm", "/var/tmp", "/run/lock") if os.path.isdir(d)]
WATCH_ROOTS = STICKY_DIRS + [HOME]
LATE_GATE_MAX_GAP_S = 600.0

from arm.assembly_cell import runner  # noqa: E402  (stdlib only)
from arm.assembly_six import lockdown  # noqa: E402  (stdlib only)
from arm import verify_freeze  # noqa: E402  (stdlib only)

FREEZER = [None]        # the agent account's verify_freeze.AgentFreezer: started by _early_lockdown, stopped after the verdict
SCOPE = ["digits"]      # grader_config.json's scope name (set in _main_locked): "digits" (the four digit seats) | "default" (every mate)


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
    them (a respawn that appears meanwhile is frozen by the next pass); empty when every one is stopped."""
    deadline = time.time() + float(bound_s)
    while True:
        FREEZER[0].freeze_now()
        time.sleep(0.05)
        running = [pid for pid in lockdown.live_uid_processes(uid) if _proc_state(pid) not in ("T", "t", "X")]
        if not running or time.time() > deadline:
            return running


# The caps on the submission's shape, checked by one walk that only stats (`submission_scan`) BEFORE anything of the
# submission is copied or read: at verify start (`_early_lockdown`, on the submission as restored, before the late-write
# walk and the snapshot) and again in the contract check (on the snapshot). Above a cap nothing is copied and nothing is
# read: the verdict carries the cap's reason (contract_ok 0). The entry and size caps are grader_config.json
# route_gate.max_entries / max_submission_mb; the path cap is this constant: the longest path inside the submission, in
# bytes (the grader's copies put at most 35 bytes in front of it, and Linux refuses paths of 4096 bytes and more).
MAX_PATH_BYTES = 4000
# A real oracle token is 64 hex digits and a line end: a .oracle_token larger than this, or one that is not a regular
# file, is a mismatch without being opened (token_state).
TOKEN_MAX_BYTES = 4096


def submission_caps(cfg):
    """The submission caps of grader_config.json (route_gate) and the path cap, checked: a missing or wrong value
    raises (in _main_locked that is a task error, before the line)."""
    rg = cfg.get("route_gate") or {}
    caps = {"max_submission_mb": int(rg.get("max_submission_mb", 60)), "max_entries": int(rg.get("max_entries", 10000)),
            "max_path_bytes": int(MAX_PATH_BYTES),
            "forbidden_extensions": tuple(str(x) for x in rg.get("forbidden_extensions", []))}
    for k in ("max_submission_mb", "max_entries", "max_path_bytes"):
        if not caps[k] > 0:
            raise ValueError("grader_config.json: %s = %r is not a positive number" % (k, caps[k]))
    caps["cap_bytes"] = caps["max_submission_mb"] * 1024 * 1024
    return caps


def submission_scan(src, p, prune_pycache=True):
    """One walk of the submission that only stats: os.scandir and lstat, nothing is opened, and no folder's list of names
    is built before its entries are counted. Returns the sum of the sizes of the entries that are not folders (lstat: a
    sparse file counts by its apparent size), the number of entries (files, folders, links and special files together),
    the weight-file problems (names ending in a forbidden extension), the regular .py / .sh files (relpaths, the network
    audit's order: folders top-down in the order os.walk visits them, each folder's files by name), and whether a cap was
    passed: `over_entries` (the walk stops at once when the count passes p["max_entries"]), `over_bytes` (the size above
    p["cap_bytes"]; once passed, no further size is taken), `over_path` (a path inside the submission longer than
    p["max_path_bytes"] bytes: that folder is not entered). Iterative (no recursion, whatever the depth), one folder open
    at a time; what it holds is bounded by the caps. `prune_pycache` skips __pycache__ folders as the run folder does (the
    contract check); the walk at verify start counts every entry."""
    forb_ext = tuple(p.get("forbidden_extensions") or ())
    max_entries, cap_bytes = int(p["max_entries"]), int(p["cap_bytes"])
    max_path = int(p.get("max_path_bytes", MAX_PATH_BYTES))
    total, entries, weight, py, sh = 0, 0, [], [], []
    over_entries = over_path = False
    stack = [""]                                    # relative folder paths still to visit ("" = src itself)
    while stack and not over_entries:
        rel_dir = stack.pop()
        top = os.path.join(src, rel_dir) if rel_dir else src
        subdirs, files = [], []
        try:
            it = os.scandir(top)
        except OSError:
            continue                                # a folder that cannot be listed is skipped, as os.walk does
        with it:
            while True:
                try:
                    e = next(it)
                except StopIteration:
                    break
                except OSError:
                    break
                try:
                    is_dir = e.is_dir()             # follows a link, as os.walk's folder list does
                except OSError:
                    is_dir = False
                if is_dir and prune_pycache and e.name == "__pycache__":
                    continue
                entries += 1
                if entries > max_entries:
                    over_entries = True
                    break
                rel = os.path.join(rel_dir, e.name) if rel_dir else e.name
                if len(os.fsencode(rel)) > max_path:
                    over_path = True                # not entered, not listed for the audit
                    continue
                if is_dir:
                    try:
                        if not e.is_symlink():
                            subdirs.append(e.name)
                    except OSError:
                        pass
                else:
                    files.append(e.name)
        if over_entries:
            break
        for fn in sorted(files):                    # bounded: at most max_entries names in all
            path = os.path.join(top, fn)
            rel = os.path.join(rel_dir, fn) if rel_dir else fn
            if total <= cap_bytes:
                try:
                    st = os.lstat(path)
                except OSError:
                    continue
                total += st.st_size
                mode = st.st_mode
            else:
                mode = None                         # past the size cap: no further stat
            if forb_ext and fn.lower().endswith(forb_ext):
                weight.append("weight file %s" % rel)
            if fn.endswith((".py", ".sh")) and mode is not None and statmod.S_ISREG(mode):
                (py if fn.endswith(".py") else sh).append(rel)
        stack.extend(os.path.join(rel_dir, d) if rel_dir else d for d in reversed(subdirs))
    return {"bytes": total, "entries": entries, "weight": weight, "py": py, "sh": sh,
            "over_entries": over_entries, "over_bytes": total > cap_bytes, "over_path": over_path}


def cap_reason(scan, p):
    """The contract reason of the first cap a walk (submission_scan) passed, or None: the entry count, then the size,
    then the path length."""
    if scan["over_entries"]:
        return "route gate: submission has more than %d files and folders" % p["max_entries"]
    if scan["over_bytes"]:
        return "route gate: submission larger than %d MB" % p["max_submission_mb"]
    if scan["over_path"]:
        return "route gate: submission has a path longer than %d bytes" % p["max_path_bytes"]
    return None


def _early_caps():
    """The caps for the walk at verify start (grader_config.json beside this file), or None when they cannot be read
    (then nothing of the submission is walked or copied here, and _main_locked stops with a task error before the line)."""
    try:
        with open(os.path.join(STAGE_DIR, "grader_config.json")) as fh:
            return submission_caps(json.load(fh))
    except Exception:  # noqa: BLE001 - the task's own fault: reported loudly by _main_locked (grading_params)
        return None


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
    # supervisord (the image's PID 1) respawns the agent account's desktop within seconds of the kill, and the
    # respawned session runs the account's autostart entries: every process of the account stays SIGSTOPped (polled every
    # 0.1 s; a stopped process does not exit, so it is not respawned) until the verdict is written
    FREEZER[0] = verify_freeze.AgentFreezer.for_user(AGENT_USER).start()
    if out["agent_processes_left"]:
        # The image's supervisord respawns the account's desktop as fast as it is killed, so in the separate verifier a
        # process that outlives the kill is the image's own (once, one respawned desktop process outlived this kill
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
    caps = _early_caps() if os.path.isdir(SUBMISSION) else None
    if caps is not None:
        # the caps first: one walk that only stats (nothing opened, nothing copied); above a cap the late-write walk
        # and the snapshot are skipped, so nothing of the submission is copied, and the verdict carries the cap's
        # reason (_grade_submission: the token is checked as always -- lstat first, at most TOKEN_MAX_BYTES read --;
        # no source audit, no import check, no episode)
        try:
            scan = submission_scan(SUBMISSION, caps, prune_pycache=False)
            out["cap_reason"] = cap_reason(scan, caps)
            if out["cap_reason"]:
                log("[grader] the submission is beyond a cap (%s; the walk at verify start counted %d entr%s, %d bytes): "
                    "nothing of it is copied or read" % (out["cap_reason"], scan["entries"],
                                                         "y" if scan["entries"] == 1 else "ies", scan["bytes"]))
        except Exception:  # noqa: BLE001 - depends on the submission: a verdict with grader_exception 1 (_main_locked)
            out["exception"] = traceback.format_exc()
    if caps is not None and not out.get("cap_reason") and not out.get("exception"):
        try:
            if gate_active:
                out["late"] = lockdown.late_entries(SUBMISSION, t_ref)
                out["submission_late_writes"] = float(len(out["late"]))
                if out["late"]:
                    out["reasons"].append("%d entr%s of the submission written after the agent phase ended (%s)" % (
                        len(out["late"]), "y" if len(out["late"]) == 1 else "ies", ", ".join(e[0] for e in out["late"][:5])))
            os.makedirs(RUNS_ROOT, exist_ok=True)
            os.chmod(RUNS_ROOT, 0o755)
            skipped = lockdown.snapshot_dir(SUBMISSION, SNAPSHOT)
            out["submission_snapshot"] = 1.0
            if skipped:
                log("[grader] the submission's snapshot dropped %d entr%s that %s not a regular file or a directory (symbolic "
                    "links, FIFOs, devices, sockets): the gates and the program never see %s" % (
                        skipped, "y" if skipped == 1 else "ies", "is" if skipped == 1 else "are", "it" if skipped == 1 else "them"))
        except Exception:  # noqa: BLE001 - depends on the submission: a verdict with grader_exception 1 (_main_locked)
            out["exception"] = traceback.format_exc()
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

from arm.assembly_cell import wire  # noqa: E402
from arm.assembly_hand.server import HandInfoServer  # noqa: E402
from arm.assembly_info import netaudit  # noqa: E402
from arm.assembly_info.client import launcher_argv  # noqa: E402
from common.seeds import seed_rng  # noqa: E402
import episode_trace  # noqa: E402  (the maintainers' replay record; never a scoring input)

GT = os.path.join(STAGE_DIR, "gt")                               # the cell's kit: kit.json, mating_graph.json, joints.json, meshes/ (fetch_hidden_gt)
PANDA_DIR = os.path.join(ROOT_GIVEN, "panda")                    # the given robot model (the Panda with its hand)
HIDDEN = os.path.join(STAGE_DIR, "hidden")
# The maintainers' record (the agent program's stderr, the replay traces, episodes.json) lives outside /opt/ale: the
# engine deletes its stage folders before it keeps a sandbox, and the record is read from a kept one (the verifier
# sandbox: run with `[sandbox_retention] verifier = "keep"`). Root-only (0700).
RECORD_ROOT = os.environ.get("ARB_RECORD_ROOT", "/opt/arb-verify")
LOGS = os.path.join(RECORD_ROOT, "logs")
OUT = os.path.join(RECORD_ROOT, "out")


def record_dir(path):
    """`path` under the root-only record root, created on demand; never a scoring input."""
    os.makedirs(RECORD_ROOT, exist_ok=True)
    os.chmod(RECORD_ROOT, 0o700)
    os.makedirs(path, exist_ok=True)
    return path
CRITERIA = ["assembly_score"]
GATES = ["contract_ok"]
# Every process of the program (the import check's, each episode's launcher and whatever it starts) begins with
# oom_score_adj 1000, so that the kernel's memory killer takes the program first when the verifier's one memory cgroup
# (the grader, the cell's render session and the program) is full: `choom -n 1000 --` (util-linux), innermost, after
# runuser and prlimit, sets it as the runner account and execs the interpreter (the same pid, the environment
# untouched; raising one's own value needs no privilege; a child inherits it). The value is also kept: while the
# program runs, the root sampler that walks the runner account's processes (netaudit.SocketSampler, oom_floor) KILLS
# (SIGKILL) any of them that reads below 1000 -- a legitimate program never lowers its own value, so one found below it
# has lowered it, and a stopped process is a crashed process (the episode goes on under its crash rule) -- and counts
# the kills (runner_oom_adj_killed, a record only). The launch chain's own helpers (runuser / prlimit / choom), which
# run as the account for a few milliseconds before choom writes 1000, are told apart by their executable name and are
# never killed. The grader's own source-audit child starts the same way (it is root, so the account sampler never sees
# it). tests/test.sh lowers the grader's own value where the container allows it.
OOM_ADJ = 1000
CHOOM = "/usr/bin/choom" if os.path.isfile("/usr/bin/choom") else shutil.which("choom")


def oom_first(argv):
    return ([CHOOM, "-n", str(OOM_ADJ), "--"] if CHOOM else []) + list(argv)


def own_oom_score_adj():
    try:
        with open("/proc/self/oom_score_adj") as fh:
            return fh.read().strip()
    except OSError:
        return "?"
SUSPICIOUS = ("/opt/ale", "ALE_STAGE_DIR", "ALE_VERIFICATION", "hidden/bench_plan", "hidden/seeds", "oracle_token", "verify/gt",
              "ALE_VERIFY", "tests/gt", "/opt/ale/tests", ".ale-solution", "verifier/gt")


def program_source():
    return SNAPSHOT if os.getuid() == 0 and os.path.isdir(SNAPSHOT) else SUBMISSION


def fail(reason):
    log("[grader] TASK ERROR: " + reason)
    raise SystemExit(1)


_WS = b" \t\n\r\x0b\x0c"     # the ASCII white space bytes.strip() removes


def sha256_file(p, strip=False, max_bytes=None):
    """sha256 of a file's bytes, read in pieces of at most 1 MB (streaming: the memory it holds does not grow with the
    file). With `max_bytes` the file is opened without following a link and without blocking, must be a regular file of
    at most `max_bytes` bytes on the open descriptor and is read at most that far; otherwise None. With `strip` (only
    together with `max_bytes`) the leading and trailing ASCII white space is left out of the hash, as bytes.strip()
    would (a run of white space is held back until a later byte shows it is not trailing: at most `max_bytes`)."""
    h = hashlib.sha256()
    if max_bytes is None:
        with open(p, "rb") as f:
            for b in iter(lambda: f.read(1 << 20), b""):
                h.update(b)
        return h.hexdigest()
    fd = os.open(p, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | getattr(os, "O_CLOEXEC", 0))
    with os.fdopen(fd, "rb", buffering=0) as f:
        st = os.fstat(f.fileno())
        if not statmod.S_ISREG(st.st_mode) or st.st_size > max_bytes:
            return None
        left, started, pending = int(max_bytes), not strip, b""
        while True:
            b = f.read(min(1 << 20, left + 1))
            if not b:
                break
            left -= len(b)
            if left < 0:
                return None                         # grew past the bound after the check
            if not strip:
                h.update(b)
                continue
            if not started:
                b = b.lstrip(_WS)
                if not b:
                    continue
                started = True
            body = b.rstrip(_WS)
            if body:
                h.update(pending)
                h.update(body)
                pending = b[len(body):]
            else:
                pending += b
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


def fetch_hidden_gt(cfg):
    """The cell's kit (gt/: kit.json with the units, their frames and masses, mating_graph.json, joints.json with the
    digit modules' joints as simulated, meshes/ with the units' visual meshes and carved collision pieces) is not in the
    repository: one tarball in the benchmark's asset
    dataset, pinned in grader_config.json ("hidden_gt": the URL at a dataset commit, its sha256 and its file count). It
    is downloaded here at verify start (the agent's sandbox never held it and the image has no copy), checked and
    unpacked into gt/ of this root-only stage folder. A failed download, a hash mismatch, a member outside gt/ or another
    file count is a task error: nothing is scored without the exact kit. ARB_HIDDEN_GT_TARBALL (the local copy
    tests/assets/gt.tar.xz that tests/test.sh points at before the kit is published, or a maintainer's local copy; checked
    the same way) replaces the download. Returns (files, seconds)."""
    import tarfile
    import urllib.request
    spec = cfg["hidden_gt"]
    url, want = spec["url"], spec["sha256"]
    t0 = time.time()
    tarball = os.path.join(STAGE_DIR, ".gt.tar.xz")
    part = os.path.join(STAGE_DIR, ".gt.part")
    for p in (part, GT):
        shutil.rmtree(p, ignore_errors=True)
    if os.path.exists(tarball):
        os.remove(tarball)
    local = os.environ.get("ARB_HIDDEN_GT_TARBALL")
    if local:
        shutil.copyfile(local, tarball)
    else:
        tries = 4
        for k in range(tries):
            try:
                with urllib.request.urlopen(url, timeout=120) as r, open(tarball, "wb") as fh:
                    for block in iter(lambda: r.read(1 << 20), b""):
                        fh.write(block)
                break
            except Exception as e:  # noqa: BLE001 - retried, then a task error
                log("[grader] the cell's kit: download %d/%d failed: %r" % (k + 1, tries, e))
                if k == tries - 1:
                    fail("could not download the cell's kit from %s: %r" % (url, e))
                time.sleep(5 * (k + 1))
    got = sha256_file(tarball)
    if got != want:
        fail("the cell's kit %s has sha256 %s; grader_config.json pins %s" % (local or url, got, want))
    os.makedirs(part, mode=0o700)
    with tarfile.open(tarball, "r:xz") as tf:
        bad = [m.name for m in tf.getmembers() if not (m.isfile() or m.isdir()) or not (m.name == "gt" or m.name.startswith("gt/"))]
        if bad:
            fail("the cell's kit tarball holds unexpected members: %s" % bad[:5])
        tf.extractall(part, filter="data")
    os.rename(os.path.join(part, "gt"), GT)
    os.chmod(GT, 0o700)
    shutil.rmtree(part, ignore_errors=True)
    os.remove(tarball)
    n = sum(len(fns) for _dp, _dn, fns in os.walk(GT))
    if n != int(spec["files"]) or not os.path.isfile(os.path.join(GT, "kit.json")):
        fail("the cell's kit unpacked to %d files (kit.json %s); grader_config.json expects %d"
             % (n, os.path.isfile(os.path.join(GT, "kit.json")), int(spec["files"])))
    dt = time.time() - t0
    log("[grader] the cell's kit: %d files, sha256 %s..., in %.1f s" % (n, got[:12], dt))
    return n, dt


def grading_params(cfg):
    """Every configuration value the submission's part of the grading reads, checked once (a missing key or a value that
    is not a positive number raises: _main_locked calls this before the line, so such a fault is a task error)."""
    rg = cfg.get("route_gate") or {}
    pl = cfg["parse_limits"]
    tc = cfg.get("episode_trace") or {}
    caps = submission_caps(cfg)             # the same values the walk at verify start used
    p = {"wall_cap_s": float(cfg["wall_cap_s"]), "privileged_wall_cap_s": float(cfg["privileged_wall_cap_s"]),
         "silence_s": float(cfg["silence_s"]), "socket_sample_s": float(cfg.get("socket_sample_s", 0.5)),
         "episodes": int(cfg.get("episodes", 3)), "max_submission_mb": caps["max_submission_mb"],
         "max_entries": caps["max_entries"], "max_path_bytes": caps["max_path_bytes"],
         "forbidden_extensions": caps["forbidden_extensions"],
         "forbidden_calls": [str(x) for x in rg.get("forbidden_calls", [])],
         "parse_as_gib": float(pl["address_space_gib"]), "parse_file_s": float(pl["per_file_s"]),
         "parse_total_s": float(pl["total_s"]),
         "trace_agent": bool(tc.get("agent")), "trace_every_ticks": int(tc.get("every_ticks", 4)),
         "cameras_live": cameras_live(cfg)}
    for k in ("wall_cap_s", "privileged_wall_cap_s", "silence_s", "socket_sample_s", "episodes", "max_submission_mb",
              "max_entries", "parse_as_gib", "parse_file_s", "parse_total_s", "trace_every_ticks"):
        if not (math.isfinite(p[k]) and p[k] > 0):
            raise ValueError("grader_config.json: %s = %r is not a positive number" % (k, p[k]))
    p["cap_bytes"] = p["max_submission_mb"] * 1024 * 1024
    p["parse_as_bytes"] = int(p["parse_as_gib"] * (1 << 30))
    return p


def submission_cap_bytes(cfg):
    return grading_params(cfg)["cap_bytes"]


def _vm_kb(field, pid="self"):
    try:
        with open("/proc/%s/status" % pid) as fh:
            for line in fh:
                if line.startswith(field + ":"):
                    return int(line.split()[1])
    except (OSError, ValueError, IndexError):
        pass
    return -1


# Python's parser takes a few hundred times a dense source's size in memory (measured: 6 MB of `x = 1` lines, 2.4 GB),
# so a submission under the size cap can need more memory to parse than the verifier has. The grader itself does no work
# on the submission's bytes at all: it stats, copies and hashes them, and one child process per verify
# (`source_audit`; root, the grader's own code, never the submission's: it reads, decodes, scans and parses, it executes
# nothing) does every decode, split, string check and parse. Per .py / .sh file the child returns, as one JSON line:
# the declared-encoding problem (a .py that declares a text encoding other than UTF-8 / ASCII / Latin-1, decided before
# the body is decoded), the network audit's hits (netaudit.tree_hits), the parser's error, whether assemble.py defines
# solve, the route gate's forbidden-call strings found (streamed, both .py and .sh) and the count of suspicious lines.
# Its address space is limited (prlimit --as, grader_config.json parse_limits address_space_gib), each file has its own
# wall-clock limit (per_file_s, from the line that announces the file) and the whole audit has a wall-clock limit
# (total_s); it starts at oom_score_adj 1000 like the program. A file that does not fit (MemoryError from the
# address-space limit, the per-file time, or the child killed while it parses the file) is a contract failure with its
# own reason (`<file>: cannot be parsed within the grader's limits (<n> GiB, <n> s)`), and the whole audit past total_s
# its own reason, not a networking import. The parser's own nesting limit ("too complex", RecursionError) is a parse
# error of the file, as a syntax error is.
SOURCE_AUDIT_CODE = r"""
import ast, hashlib, io, json, os, sys
sys.dont_write_bytecode = True
sys.path.insert(0, sys.argv[1])
from arm.assembly_info import netaudit
req = json.loads(sys.stdin.buffer.read())
root, allow, solve_file = req["root"], set(req["allow"]), req["solve_file"]
FORBIDDEN = list(req.get("forbidden_calls") or [])
SUSPICIOUS = list(req.get("suspicious") or [])
MAX_HITS = int(req.get("max_hits") or 5)          # hits sent per file (their count apart): what the grader shows
MAX_HIT_CHARS = int(req.get("max_hit_chars") or 200)


def emit(d):
    sys.stdout.write(json.dumps(d) + "\n")
    sys.stdout.flush()


def defines_solve(tree):
    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == "solve":
            return True
        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "solve" for t in n.targets):
            return True
        if isinstance(n, (ast.Import, ast.ImportFrom)) and any((a.asname or a.name) == "solve" for a in n.names):
            return True
    return False


def string_scan(data, enc):
    # the route gate's string checks: decode with `enc` (errors replaced) and stream line by line -- no list of all
    # lines and no second copy of the whole text held at once -- counting the forbidden-call strings present (a set)
    # and the lines that hold a suspicious pattern.
    calls, susp = set(), 0
    wrap = io.TextIOWrapper(io.BytesIO(data), encoding=enc, errors="replace")
    try:
        for line in wrap:
            for c in FORBIDDEN:
                if c in line:
                    calls.add(c)
            if SUSPICIOUS and any(s in line for s in SUSPICIOUS):
                susp += 1
    finally:
        wrap.detach()
    return sorted(calls), susp


emit({"ready": len(req["files"])})
for k, rel in enumerate(req["files"]):
    emit({"begin": k})
    res = {"k": k}
    is_py = rel.endswith(".py")
    try:
        path = os.path.join(root, rel)
        with open(path, "rb") as fh:
            data = fh.read(os.lstat(path).st_size)
    except OSError:
        res["hits"] = ["cannot be read"]
        emit(res)
        continue
    exempt = is_py and hashlib.sha256(data).hexdigest() in allow        # byte-identical to a shipped library file
    if is_py and not exempt:
        enc = netaudit.encoding_problem(rel, data)                      # decided from the first two lines, before any decode
        if enc is not None:
            res["enc_bad"] = enc[1]
            del data
            emit(res)
            continue
    try:
        calls, susp = string_scan(data, "utf-8" if (not is_py or exempt) else netaudit.scan_encoding(data))
        res["calls"], res["susp"] = calls, susp
    except MemoryError:
        res["limit"] = "MemoryError"
        del data
        emit(res)
        continue
    if not is_py or (exempt and rel != solve_file):
        if is_py:
            res["hits"] = []
        del data
        emit(res)
        continue
    tree = None
    try:
        tree = ast.parse(data)
    except MemoryError as e:
        if str(e):
            res["parse_error"] = ["MemoryError", str(e)[:300]]
        else:
            res["limit"] = "MemoryError"
    except SyntaxError as e:
        res["parse_error"] = ["SyntaxError", str(e)[:300]]
    except Exception as e:
        res["parse_error"] = [type(e).__name__, ""]
    del data
    if tree is not None:
        try:
            hits = [] if exempt else netaudit.tree_hits(rel, tree)
            if rel == solve_file:
                res["solve"] = defines_solve(tree)
            res["n_hits"] = len(hits)
            res["hits"] = [str(h[1])[:MAX_HIT_CHARS] for h in hits[:MAX_HITS]]     # the file is the line's k
        except MemoryError:
            res.pop("hits", None)
            res["limit"] = "MemoryError"
        except RecursionError:
            res["parse_error"] = ["RecursionError", ""]
        del tree
    emit(res)
emit({"end": True})
"""
AUDIT = {}      # the last source audit's record (files, wall, the child's peak); a record only
# the network audit's hits the grader holds and names in its reason (the first NET_SHOWN, each at most NET_HIT_CHARS
# characters); the child counts every hit (network_imports) and sends at most NET_SHOWN per file without the file's name
# (it is the line's index), so what the grader holds does not grow with the number of hits or with their names
NET_SHOWN = 5
NET_HIT_CHARS = 200


def _limits_text(p):
    gib = p["parse_as_gib"]
    return "%s GiB, %s s" % (("%d" % gib) if gib == int(gib) else ("%g" % gib),
                             ("%d" % p["parse_file_s"]) if p["parse_file_s"] == int(p["parse_file_s"]) else ("%g" % p["parse_file_s"]))


def _total_text(p):
    return ("%d" % p["parse_total_s"]) if p["parse_total_s"] == int(p["parse_total_s"]) else ("%g" % p["parse_total_s"])


def source_audit(src, p, allow, files, solve_file="assemble.py"):
    """{"results": {relpath: result}, "limit": (relpath, what) | None, "total_limit": bool, "files": n}: every regular
    .py / .sh file of `src` (the relpaths in `files`, the network audit's order) handled in one child process
    (SOURCE_AUDIT_CODE) that decodes, scans and parses them -- the grader never touches their bytes -- under the
    address-space limit p["parse_as_bytes"], the per-file wall-clock limit p["parse_file_s"] and the whole-audit
    wall-clock limit p["parse_total_s"]. The grader only reads the child's JSON lines. A child that does not start
    (no ready line within 60 s) is a task error."""
    import select
    import signal
    AUDIT.clear()
    AUDIT.update({"files": len(files), "wall_s": 0.0, "child_peak_kb": -1, "limit": None, "total_limit": False})
    out = {"results": {}, "limit": None, "total_limit": False, "files": len(files)}
    if not files:
        return out
    argv = [sys.executable, "-I", "-B", "-c", SOURCE_AUDIT_CODE, STAGE_DIR]
    if shutil.which("prlimit"):
        argv = ["prlimit", "--as=%d" % p["parse_as_bytes"], "--core=0", "--"] + argv
    t0 = time.time()
    total_deadline = t0 + p["parse_total_s"]
    try:
        err_path = os.path.join(record_dir(LOGS), "source_audit_stderr.log")     # made before the line in a graded run
    except OSError:
        err_path = os.devnull
    with open(err_path, "wb") as err:
        try:
            proc = subprocess.Popen(oom_first(argv), cwd="/", stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=err,
                                    env={"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "LANG": "C.UTF-8"},
                                    start_new_session=True)
        except OSError as e:
            fail("the source audit's process could not be started: %s" % e)
        req = json.dumps({"root": src, "files": files, "allow": sorted(allow), "solve_file": solve_file,
                          "forbidden_calls": p["forbidden_calls"], "suspicious": list(SUSPICIOUS),
                          "max_hits": NET_SHOWN, "max_hit_chars": NET_HIT_CHARS}).encode()
        try:
            proc.stdin.write(req)
            proc.stdin.close()
        except OSError:
            pass
        fd, buf, ready, current, done = proc.stdout.fileno(), b"", False, None, False
        deadline = min(t0 + 60.0, total_deadline)       # the ready window, never past the whole-audit limit
        killed = None
        while not done:
            now = time.time()
            if now >= total_deadline:
                killed = "total"
                break
            if now >= deadline:
                killed = "time"
                break
            r, _w, _x = select.select([fd], [], [], min(deadline - now, total_deadline - now, 1.0))
            if not r:
                continue
            chunk = os.read(fd, 1 << 20)
            if not chunk:
                break
            buf += chunk
            *lines, buf = buf.split(b"\n")     # one split per piece read (no copy of the rest per line)
            for line in lines:
                msg = json.loads(line.decode("utf-8"))
                if "ready" in msg:
                    ready = True
                elif "begin" in msg:
                    current = int(msg["begin"])
                    deadline = min(time.time() + p["parse_file_s"], total_deadline)
                elif "k" in msg:
                    out["results"][files[int(msg["k"])]] = msg
                    current = None
                    if msg.get("limit") and out["limit"] is None:
                        out["limit"] = (files[int(msg["k"])], msg["limit"])
                elif msg.get("end"):
                    done = True
        if not done:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except OSError:
                pass
        try:
            _pid, status, ru = os.wait4(proc.pid, 0)
            proc.returncode = os.waitstatus_to_exitcode(status)
            AUDIT["child_peak_kb"] = int(ru.ru_maxrss)
        except ChildProcessError:
            proc.wait()
        proc.stdout.close()
    AUDIT["wall_s"] = time.time() - t0
    if not ready:
        fail("the source audit's process did not start (exit %s; its stderr: %s)" % (proc.returncode, err_path))
    if killed == "total" and out["limit"] is None:
        out["total_limit"] = True
        out["limit"] = ("(the source audit)", "the whole-audit time limit")
    elif not done and out["limit"] is None:
        what = "the per-file time limit" if killed == "time" else "the process ended (exit %s)" % proc.returncode
        out["limit"] = (files[current] if current is not None else "(the .py / .sh files)", what)
    AUDIT["limit"] = out["limit"]
    AUDIT["total_limit"] = out["total_limit"]
    if out["limit"]:
        log("[grader] source audit: %s did not fit within the limits (%s; whole audit %s s): %s" % (
            out["limit"][0], _limits_text(p), _total_text(p), out["limit"][1]))
    return out


def route_problems(scan, results, p):
    """The route gate's problems and suspicious-lines count, built from the lstat walk (`scan`) and the child's per-file
    results (`results`) -- the grader reads no submission byte here. Problems: the weight files (names) then the
    forbidden-call strings the child found (`.py` and `.sh`). Suspicious: the sum of the child's per-file counts (the
    caller reports -1 where the audit did not run to this stage)."""
    problems = list(scan["weight"])
    suspicious = 0
    for rel in scan["py"] + scan["sh"]:
        res = results.get(rel) or {}
        for c in res.get("calls") or []:
            problems.append("%s uses %s" % (rel, c))
        suspicious += int(res.get("susp", 0))
    return problems, suspicious


def library_digests():
    """sha256 of every .py file of the shipped library (a submission may carry copies of it)."""
    out = set()
    for pkg in ("arm", "common", "render"):
        d = os.path.join(ROOT_GIVEN, "lib", pkg)
        for dp, _dn, fns in os.walk(d):
            for fn in fns:
                if fn.endswith(".py"):
                    out.add(sha256_file(os.path.join(dp, fn)))
    return out


def blas_pin_stats():
    """Statistics only, never a scoring input: is the numerics' pin in effect. blas_pin_env: this process's
    environment holds runner.BLAS_PIN (tests/test.sh sets it before the interpreter starts); blas_pin_child_env: the
    programs' environment (runner.child_env) does; blas_l2_size_256: both of them hold OPENBLAS_L2_SIZE=256 (the
    Haswell kernel's built-in dgemm blocking, safe on Zen 4 / Zen 5 CPUs) 1, else 0; blas_kernel_haswell: every
    OpenBLAS loaded in this process reports the Haswell kernel 1, another kernel 0, not readable -1;
    blas_numpy_simd_capped: numpy dispatches none of the SIMD targets the pin disables 1, it does 0, not readable -1."""
    pin = runner.BLAS_PIN
    child = runner.child_env("/", "/", "/", "/")
    out = {"blas_pin_env": 1.0 if all(os.environ.get(k) == v for k, v in pin.items()) else 0.0,
           "blas_pin_child_env": 1.0 if all(child.get(k) == v for k, v in pin.items()) else 0.0,
           "blas_l2_size_256": 1.0 if os.environ.get("OPENBLAS_L2_SIZE") == "256" and child.get("OPENBLAS_L2_SIZE") == "256" else 0.0,
           "blas_kernel_haswell": -1.0, "blas_numpy_simd_capped": -1.0}
    try:
        import ctypes
        import re
        with open("/proc/self/maps") as fh:
            libs = sorted(set(re.findall(r"(/\S*openblas\S*\.so\S*)", fh.read())))
        names = []
        for lib in libs:
            h = ctypes.CDLL(lib)
            for sym in ("scipy_openblas_get_corename64_", "scipy_openblas_get_corename", "openblas_get_corename64_",
                        "openblas_get_corename"):
                f = getattr(h, sym, None)
                if f is not None:
                    f.restype = ctypes.c_char_p
                    names.append((f() or b"").decode("ascii", "replace").strip().lower())
                    break
        if names:
            out["blas_kernel_haswell"] = 1.0 if all(n == "haswell" for n in names) else 0.0
    except Exception:  # noqa: BLE001 - a record only
        pass
    try:
        try:
            from numpy._core._multiarray_umath import __cpu_features__ as feats
        except ImportError:
            from numpy.core._multiarray_umath import __cpu_features__ as feats
        out["blas_numpy_simd_capped"] = 0.0 if any(feats.get(k) for k in pin["NPY_DISABLE_CPU_FEATURES"].split()) else 1.0
    except Exception:  # noqa: BLE001 - a record only
        pass
    return out


class Runner(object):
    def __init__(self, cfg):
        self.acct = runner.runner_account(cfg.get("runner_account", "nobody"), AGENT_USER)
        self.name, self.uid, self.gid = self.acct if self.acct else (None, None, None)
        self.lib_dirs = {p: os.path.join(ROOT_GIVEN, "lib", p) for p in ("arm", "common", "render")}
        self.cpu_s = float(cfg.get("cpu_cap_factor", 4.0)) * float(cfg["wall_cap_s"])
        self.killed = self.left = self.unclean_runs = self.stray_files = self.symlinks_skipped = self.oom_killed = 0
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

    def sampler(self, period_s):
        """The root sampler of the runner account's processes (remote sockets; oom_score_adj kept at OOM_ADJ)."""
        return netaudit.SocketSampler(uid=self.uid if self.acct else None, pids_fn=None if self.acct else (lambda: []),
                                      period_s=period_s, oom_floor=OOM_ADJ if self.acct else None)

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


def audit_reason(shown, n_enc):
    """the contract reason of the source audit's hits: `shown` = the first hits [(file, what)], of which the first
    `n_enc` are helpers that declare a disallowed text encoding (a source-audit reason); the others are the network
    audit's (networking imports, files that cannot be parsed)"""
    parts = []
    if n_enc:
        parts.append("source audit: %s" % ", ".join("%s %s" % x for x in shown[:n_enc]))
    if len(shown) > n_enc:
        parts.append("network audit: networking import(s) %s" % ", ".join("%s: %s" % x for x in shown[n_enc:]))
    return "; ".join(parts)


def episode_gates(step_calls, unclean_runs, left, net_outbound, remote_sockets):
    """the three gates the episodes decide (at least one step call; no network activity; no process of the run
    account still running after the kill at an episode's end) and, for each gate that is 0, its reason, in the
    instruction's order (a step call, no network activity, no process left)"""
    steps_gate = 1.0 if step_calls > 0 else 0.0
    clean_gate = 1.0 if unclean_runs == 0 else 0.0
    net_gate = 1.0 if (remote_sockets <= 0 and net_outbound <= 0) else 0.0
    reasons = []
    if not steps_gate:
        reasons.append("no step call over the episodes (the program must make at least one step call)")
    if not net_gate:
        reasons.append("network activity during an episode (%d outbound events, %d remote sockets seen)" % (net_outbound, remote_sockets))
    if not clean_gate:
        reasons.append("%d process(es) of the run account still running %.0f s after the kill at an episode's end (%d episode(s))" % (
            left, Runner.kill_runner_processes.__defaults__[0], unclean_runs))
    return steps_gate, clean_gate, net_gate, reasons


def contract_check(cfg, rn):
    """(ok, reason, suspicious lines (-1: not computed), network-audit hits). The grader itself only stats, copies and
    hashes the submission's files; the one lstat walk (`submission_scan`) gives the size, the entry count, the weight
    files and the .py / .sh list, and the source-audit child does every decode, string check and parse. The order:
    assemble.py is a regular file; the entry count, the submission's size and its paths' length (stat only, nothing
    opened; at verify start the same walk ran on the submission as restored, before the snapshot); the source audit
    (a file beyond the parse limits, or the whole audit beyond its limit, fails here with its own reason, counted in no
    other key); a .py that declares a disallowed text encoding; assemble.py parses and defines solve; the route gate
    (weight files, forbidden-call strings); the network audit's hits (a helper file that does not parse is a hit); the
    import check under the run regime."""
    p = grading_params(cfg)
    src = program_source()
    path = os.path.join(src, "assemble.py")
    if not os.path.isfile(path) or os.path.islink(path):
        return False, "no assemble.py (regular file) in the submission", -1, []
    scan = submission_scan(src, p)          # one walk that only stats: sizes, entries, paths, weight files, the .py / .sh list
    reason = cap_reason(scan, p)            # before any read: the entry-count, the size or the path-length reason
    if reason:
        return False, reason, -1, []
    files = scan["py"] + scan["sh"]
    audit = source_audit(src, p, library_digests(), files)
    log("[grader] source audit: %d source file(s) handled in a separate process (%s; whole audit %s s): %.1f s, its peak "
        "%.0f MB; the grader's own peak (VmHWM) %.0f MB" % (AUDIT.get("files", 0), _limits_text(p), _total_text(p),
                                      AUDIT.get("wall_s", 0.0), AUDIT.get("child_peak_kb", -1) / 1024.0, _vm_kb("VmHWM") / 1024.0))
    if audit["total_limit"]:
        return False, "the source audit did not finish within %s s" % _total_text(p), -1, []
    if audit["limit"]:
        return False, "%s: cannot be parsed within the grader's limits (%s)" % (audit["limit"][0], _limits_text(p)), -1, []
    bad_enc = [(rel, audit["results"][rel]["enc_bad"]) for rel in files if (audit["results"].get(rel) or {}).get("enc_bad")]
    a = audit["results"].get("assemble.py") or {}
    if a.get("enc_bad"):
        return False, "assemble.py: %s" % a["enc_bad"], -1, []
    if a.get("parse_error"):
        kind, msg = a["parse_error"]
        if kind == "SyntaxError":
            return False, "assemble.py does not parse: %s" % msg, -1, []
        return False, "assemble.py cannot be parsed: %s" % kind, -1, []
    if not a.get("solve"):
        return False, "assemble.py defines no solve", -1, []
    problems, suspicious = route_problems(scan, audit["results"], p)
    if problems:
        return False, "route gate: " + "; ".join(problems[:5]), suspicious, []
    net, n_hits = [], 0                     # net: the first NET_SHOWN hits (the reason names them); n_hits counts all
    n_enc = 0                               # of them, the first n_enc are encoding hits: a source-audit reason
    for rel, reason in bad_enc:             # a helper that declares a disallowed encoding: counted as the audit's hits are
        if len(net) < NET_SHOWN:
            net.append((rel, reason))
            n_enc += 1
        n_hits += 1
    for rel in scan["py"]:
        res = audit["results"].get(rel) or {}
        if res.get("enc_bad"):
            continue                        # already listed above
        if res.get("parse_error"):
            if len(net) < NET_SHOWN:
                net.append((rel, "cannot be parsed (%s)" % res["parse_error"][0]))
            n_hits += 1
        else:
            shown = res.get("hits") or []
            net += [(rel, str(h)) for h in shown[:max(0, NET_SHOWN - len(net))]]
            n_hits += int(res.get("n_hits", len(shown)))
    AUDIT["n_hits"] = n_hits                # network_imports (the child sends at most NET_SHOWN hits per file, counts all)
    if net:
        return False, audit_reason(net[:5], n_enc), suspicious, net
    dirs = rn.prepare("import_check", src)
    before = rn.snapshot((dirs["root"],)) if rn.acct else None
    code = ("import sys, importlib; sys.path.insert(0, sys.argv[1]); sys.path.insert(0, sys.argv[2]); "
            "m = importlib.import_module('assemble'); assert callable(m.solve)")
    argv = runner.wrap_limits(oom_first([sys.executable] + runner.PYTHON_FLAGS + ["-c", code, dirs["lib"], dirs["program"]]), cpu_s=120,
                              user=rn.name)
    try:
        with rn.sampler(p["socket_sample_s"]) as smp:
            try:
                r = subprocess.run(argv, cwd=dirs["program"], env=rn.env(dirs), capture_output=True, text=True, timeout=120,
                                   start_new_session=True)
                ok, why = r.returncode == 0, ("ok" if r.returncode == 0 else "importing assemble.py failed: " + (r.stderr.strip().splitlines() or ["?"])[-1][:300])
            except subprocess.TimeoutExpired:
                ok, why = False, "importing assemble.py took more than 120 s"
        rn.oom_killed += smp.oom_killed
    finally:
        rn.cleanup(dirs, before)
    return ok, why, suspicious, []


def token_state(src=None):
    """(privileged, mismatch): the submission's .oracle_token against hidden/oracle_token.sha256. The token is stat'ed
    first (lstat): none -> (False, False); not a regular file (a link, a folder, a FIFO, a device, a socket) or larger
    than TOKEN_MAX_BYTES -> a mismatch, without being opened; otherwise it is hashed by the streaming routine
    (sha256_file: opened without following a link and without blocking, checked again on the open file, read at most
    TOKEN_MAX_BYTES; its surrounding white space left out, as the token file holds the token and a line end)."""
    p = os.path.join(src or program_source(), ".oracle_token")
    want_p = os.path.join(HIDDEN, "oracle_token.sha256")
    if not os.path.isfile(want_p):
        return False, False
    try:
        st = os.lstat(p)
    except OSError:
        return False, False                         # no token
    if not statmod.S_ISREG(st.st_mode) or st.st_size > TOKEN_MAX_BYTES:
        return False, True                          # a mismatch, never opened
    try:
        got = sha256_file(p, strip=True, max_bytes=TOKEN_MAX_BYTES)
        with open(want_p) as fh:                    # the task's own digest file (64 hex digits)
            want = fh.read().strip()
    except OSError:
        return False, False
    if got is None:
        return False, True
    return (got == want), (got != want)


def cameras_live(cfg):
    """the cell's cameras on (grader_config.json camera_layer.enabled; off: every get_camera call answers
    camera_unavailable)"""
    return bool((cfg.get("camera_layer") or {}).get("enabled", True))


def layout_nominal(summary):
    """the episode ran on the nominal layout (arm/assembly_hand/layout.py records it as the layout's `nominal`)"""
    return bool((summary.get("layout") or {}).get("nominal"))


def run_episode(cfg, seed, rn, privileged, kit_info, plan, trace_tag=None):
    p = grading_params(cfg)                 # checked before the line (_main_locked): cannot raise here
    try:
        server = HandInfoServer(GT, PANDA_DIR, plan=plan, config=cfg, mode="hidden", privileged=privileged, render=p["cameras_live"],
                                log=log, kit_info=kit_info)
        server.reset(seed)
    except Exception as e:  # noqa: BLE001 - before the program starts: the grader's fault, never the submission's
        log(traceback.format_exc())
        fail("the cell of an episode could not be built: %s" % type(e).__name__)
    trace = None
    if trace_tag and p["trace_agent"]:
        try:
            trace = episode_trace.EpisodeTrace(server, every_ticks=p["trace_every_ticks"], log=log)
        except Exception as e:  # noqa: BLE001 - a record only
            log("[grader] episode trace not attached: %r" % (e,))
    wire_error, t0 = None, time.time()         # the record folders were made before the line (_main_locked)
    # the run folder (the program's working directory, HOME, TMPDIR and PYTHONPATH lie in it) is named by the episode's
    # index, never by its seed: the seed drives the public camera draw
    dirs = rn.prepare(trace_tag or "agent_run", program_source())
    before = rn.snapshot((dirs["root"],)) if rn.acct else None
    argv, user = oom_first(launcher_argv(sys.executable, dirs["lib"], dirs["program"])), rn.name
    # the privileged reference assembler runs inside one call (reference_assemble): its own, longer wall-clock cap
    wall_cap = p["privileged_wall_cap_s"] if privileged else p["wall_cap_s"]
    proc = None
    sampler = rn.sampler(p["socket_sample_s"])
    counters = netaudit.NetCounters()
    try:
        with counters, sampler:
            try:
                # the program's stderr record is named by the episode's index as its run folder is: the program can
                # read the name of its own open file (/proc/self/fd/2), and the seed drives the public camera draw
                proc = wire.AgentProcess(argv, cwd=dirs["program"], user=user, env=rn.env(dirs, None), cpu_s=rn.cpu_s,
                                         stderr_path=os.path.join(LOGS, "%s.log" % (trace_tag or "agent_run")))
                reason = wire.serve_episode(server, proc, wall_cap, p["silence_s"], log=log)
            finally:
                if proc is not None:
                    proc.terminate()
                rn.cleanup(dirs, before)             # every process of the runner account is gone before the counters are read
    except Exception:  # noqa: BLE001
        wire_error = traceback.format_exc()
        reason = "wire_error"
        if not server.done:
            server._end(reason)
    if wire_error:
        log("[grader] seed %d: wire error\n%s" % (seed, wire_error))
    summary = server.finalize(None)
    summary["episode_wall_s"] = time.time() - t0
    summary["end_reason"] = reason
    if trace is not None:
        try:
            trace.detach()
            summary["trace_dir"] = trace.save(os.path.join(OUT, "trace"), trace_tag, summary,
                                              extra={"seed": int(seed), "privileged": bool(privileged)})
        except Exception as e:  # noqa: BLE001 - a record only
            log("[grader] episode trace not written: %r" % (e,))
    rn.oom_killed += int(sampler.oom_killed)
    summary["oom_adj_killed"] = int(sampler.oom_killed)
    summary["remote_sockets_seen"] = int(sampler.max_seen)
    summary["socket_samples"] = int(sampler.samples)
    summary["net_outbound"] = int(counters.total)
    summary["net_delta"] = dict(counters.delta)
    return summary


def write_verdict(rewards, metrics):
    """One gated `reward` in the benchmark kit's envelope (robotics_grader.ratio_rewards against anchor.json: value 1.0 =
    the oracle, full_at 1.0, so reward = assembly_score x the gates); the gate, the criterion and every statistic go to
    `metrics`. In the engine (`ALE_VERIFY_PYTHON` set) the kit writes reward.json and verification.json, the criteria
    text as the `diagnostic`; the metrics keep the names and the filter of this grader's records on the earlier engine
    API (finite numbers only, '.' and '/' in a name replaced by '_'). Elsewhere the envelope goes to
    ROBOTICS_GRADER_VERDICT_PATH, else verdict.local.json in the stage folder."""
    from robotics_grader import ratio_rewards, read_anchor
    from robotics_grader import write_verdict as kit_write_verdict
    overall = float(rewards["assembly_score"])
    anchor, meta = read_anchor()
    extra = {k: float(v) for k, v in metrics.items()}
    extra.update({k: float(rewards[k]) for k in GATES + CRITERIA})
    verdict = ratio_rewards(overall, anchor, higher_is_better=meta.get("direction", "higher") == "higher",
                            metric_name=meta.get("metric", "assembly_score"), extra=extra, full_at=meta.get("full_at", 1.0))
    text = ("contract_ok %.0f (the program imports, exposes solve, passes the route gate and the network audit and steps the "
            "cell; no late write, no surviving process); assembly_score %.4f (episode mean of the scored mates' weight / %g "
            "(scope %s) on the hidden episodes, an episode with a safety violation keeping %g of it)") % (
                rewards["contract_ok"], rewards["assembly_score"], float(metrics.get("scope_weight", 0.0)), SCOPE[0],
                float(metrics.get("safety_violation_factor", 0.0)))
    if os.environ.get("ALE_VERIFY_PYTHON"):
        recorded = {}
        for k, v in sorted(verdict["metrics"].items()):
            try:
                v = float(v)
            except (TypeError, ValueError):
                continue
            if math.isfinite(v):
                recorded[k.replace(".", "_").replace("/", "_")] = v
        kit_write_verdict({"rewards": verdict["rewards"], "metrics": recorded}, diagnostic=text)
        return overall
    kit_write_verdict(verdict, path=os.environ.get("ROBOTICS_GRADER_VERDICT_PATH")
                      or os.path.join(STAGE_DIR, "verdict.local.json"))
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
               "stray_files_removed": 0.0, "suspicious_lines": -1.0, "agent_wire_errors": 0.0, "remote_sockets_seen": 0.0, "net_outbound_events": 0.0,
               "network_imports": 0.0, "illegal_call_count": 0.0, "source_parse_limit": 0.0, "runner_oom_adj_killed": 0.0}
    for k in ("stage_locked", "stage_parent_locked", "stage_lock_latency_ms", "agent_processes_killed", "agent_processes_left",
              "submission_late_writes", "late_writes_quarantined", "late_write_gate_active", "submission_snapshot"):
        metrics[k] = float(EARLY.get(k, -1.0 if k == "stage_lock_latency_ms" else 0.0))
    try:
        _main_locked(cfg, rewards, metrics)
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
    gt_files, gt_s = fetch_hidden_gt(cfg)
    metrics["hidden_gt_files"] = float(gt_files)
    metrics["hidden_gt_fetch_s"] = float(gt_s)
    metrics["hidden_gt_local_copy"] = 1.0 if os.environ.get("ARB_HIDDEN_GT_TARBALL") else 0.0
    sc = cfg.get("scope") or {}
    SCOPE[0] = str(sc.get("name", "default"))
    graph = load_json(os.path.join(GT, "mating_graph.json"))
    weights = {m["id"]: float(m["weight"]) for m in graph["mates"]}
    scope_ids = list(sc.get("mates") or [m["id"] for m in graph["mates"]])
    unknown = [i for i in scope_ids if i not in weights]
    if unknown:
        fail("grader_config.json scope names mates the kit's mating graph does not have: %s" % unknown)
    metrics["scope_mates"] = float(len(scope_ids))
    metrics["scope_weight"] = float(sum(weights[i] for i in scope_ids))
    metrics["graph_weight"] = float(sum(weights.values()))
    factor = float((cfg.get("scoring") or {}).get("safety_violation_factor", 0.0))
    metrics["safety_violation_factor"] = factor
    metrics["camera_layer_enabled"] = 1.0 if cameras_live(cfg) else 0.0
    kit_info = load_json(os.path.join(STAGE_DIR, "kit_info.json"))    # the units' materials and colours (kept beside this file)
    plan_p = os.path.join(HIDDEN, "bench_plan.json")
    plan = load_json(plan_p) if os.path.isfile(plan_p) else None       # without a bench plan: the reference's default steps
    metrics["bench_plan"] = 1.0 if plan is not None else 0.0
    rn = Runner(cfg)
    if FREEZER[0] is not None and rn.acct and rn.uid == FREEZER[0].uid:
        FREEZER[0].stop()        # no `nobody` in the image: the runner fell back to the agent account, whose programs must run
        log("[grader] runner account = the agent account: the verify-stage freezer is stopped")
    metrics["runner_uid"] = float(rn.uid) if rn.acct else -1.0
    metrics["agent_deprivileged"] = 1.0 if rn.acct else 0.0
    pinned = os.path.join(HIDDEN, "seeds.json")
    stream = seed_rng(pinned=pinned if os.path.isfile(pinned) else None)
    metrics.update(stream.stats())
    try:
        grading_params(cfg)                 # every configuration value the submission's part reads, checked here
    except (KeyError, TypeError, ValueError, AttributeError) as e:
        fail("grader_config.json: a value the grading needs is missing or wrong: %s: %s" % (type(e).__name__, e))
    try:
        record_dir(LOGS)
        record_dir(OUT)
    except OSError as e:
        fail("the maintainers' record folders under %s cannot be made: %s" % (RECORD_ROOT, e))
    log("[grader] memory killer: the grader's oom_score_adj %s; every process of the program starts at %d (%s); the root "
        "sampler kills any process of the program found below %d" % (
            own_oom_score_adj(), OOM_ADJ, "choom" if CHOOM else "no choom: a process below the floor is killed by the sampler",
            OOM_ADJ))
    # Up to here the task's own inputs (config and every value the grading reads from it, givens, kit, mating graph,
    # runner account, seeds, the record folders): a fault is a task error (no verdict, exit 1). From here on everything
    # depends on the submission (its token, the contract check: source audit, route gate, network audit, import check;
    # the episodes from the program's launch on; the scoring of their records): any exception there, other than a task
    # error raised by fail(), gives a verdict with grader_exception 1, contract_ok 0 and reward 0
    # (_grader_exception_verdict). The cell's build at an episode's start stays a task error.
    try:
        _grade_submission(cfg, rewards, metrics, rn, t_start, stream, scope_ids, weights, factor, kit_info, plan)
    except Exception:  # noqa: BLE001 - a verdict in every case
        _grader_exception_verdict(rewards, metrics, rn, t_start, traceback.format_exc())


def _grader_exception_verdict(rewards, metrics, rn, t_start, tb):
    """The verdict when the submission-dependent part raised: grader_exception 1, contract_ok 0, reward 0; the
    exception's type goes to the grader's log, its traceback to the maintainers' record (root-only), nothing to the
    program. Exits as every other verdict does."""
    lines = [l for l in tb.strip().splitlines() if l.strip()]
    kind = (lines[-1].split(":")[0].strip() if lines else "") or "?"
    try:
        with open(os.path.join(record_dir(LOGS), "grader_exception.log"), "a") as fh:
            fh.write(tb)
    except OSError:
        pass
    log("[grader] contract_ok 0: the grader raised %s while it graded the submission (grader_exception 1, reward 0; the "
        "traceback is in the maintainers' record)" % kind)
    for k in list(rewards):
        rewards[k] = 0.0
    for k in list(metrics):
        if not isinstance(metrics[k], (int, float)):
            metrics.pop(k)
    metrics["grader_exception"] = 1.0
    try:
        rn.kill_runner_processes()                  # no process of the program outlives the verdict
        _finish_stats(rn, metrics, t_start)
    except Exception:  # noqa: BLE001 - the verdict is written all the same
        metrics["verify_wall_s"] = time.time() - t_start
    write_verdict(rewards, metrics)


def _grade_submission(cfg, rewards, metrics, rn, t_start, stream, scope_ids, weights, factor, kit_info, plan):
    if EARLY.get("exception"):
        raise RuntimeError("the submission's walk or snapshot at verify start failed:\n" + EARLY["exception"])
    privileged, mismatch = token_state()
    metrics["privileged_mode"] = 1.0 if privileged else 0.0
    metrics["oracle_token_mismatch"] = 1.0 if mismatch else 0.0
    if EARLY.get("cap_reason"):
        # beyond a cap at verify start: nothing was copied (no snapshot), and nothing runs on the submission -- no
        # source audit, no import check, no episode; the reason as the contract check words it, after its first check
        a = os.path.join(SUBMISSION, "assemble.py")
        ok, suspicious, net = False, -1, []
        why = ("no assemble.py (regular file) in the submission" if (not os.path.isfile(a) or os.path.islink(a))
               else EARLY["cap_reason"])
    else:
        ok, why, suspicious, net = contract_check(cfg, rn)
    if ok and EARLY.get("reasons"):
        ok, why = False, "; ".join(EARLY["reasons"])
    metrics["suspicious_lines"] = float(suspicious)
    metrics["network_imports"] = float(AUDIT["n_hits"] if net and "n_hits" in AUDIT else len(net))
    metrics["source_parse_limit"] = 1.0 if AUDIT.get("limit") else 0.0
    # before the episodes only a failed check is a verdict (contract_ok 0 with its reason); a passed one leaves
    # contract_ok to the episodes' gates (rules 12 to 14), which are logged with their reasons after the episodes
    log("[grader] %s; privileged %s; runner %s; lock latency %.0f ms" % (
        ("contract check passed before the episodes (contract_ok follows from the episodes' gates)" if ok else "contract_ok 0: %s" % why),
        privileged, rn.name or "self", metrics["stage_lock_latency_ms"]))
    if not ok:
        rewards["contract_ok"] = 0.0
        _finish_stats(rn, metrics, t_start)
        write_verdict(rewards, metrics)
        return
    n = grading_params(cfg)["episodes"]
    ep_seeds = stream.derive("episodes").seeds(n, bits=63)     # 63-bit: the public draw code cannot be searched over the seed space
    agent = []
    step_calls = illegal = 0
    for i, seed in enumerate(ep_seeds):
        log("[grader] episode %d/%d" % (i + 1, n))
        s_a = run_episode(cfg, seed, rn, privileged, kit_info, plan, trace_tag="agent_ep%02d" % (i + 1))
        got = {r["id"] for r in s_a["mates"] if r["achieved"]}
        s_a["scope_achieved"] = sorted(i_ for i_ in scope_ids if i_ in got)
        s_a["scope_score"] = sum(weights[i_] for i_ in s_a["scope_achieved"]) / metrics["scope_weight"]
        metrics["episode_traces_written"] = metrics.get("episode_traces_written", 0.0) + (1.0 if s_a.get("trace_dir") else 0.0)
        pk, su = s_a["damage"]["peak"], s_a["damage"]["sustained_0.2s"]
        log("[grader]   scope score %.3f (%d / %d scope mates; all mates %d / %d) safety %s sim %.1f s wall %.1f s camera calls %d "
            "frames %s renderer %s nominal %s end %s R1 %.2f N, R2 %.2f N (0.2 s), R3 %.3f N m (0.2 s), R4 %.2f m/s; violations %s" % (
                s_a["scope_score"], len(s_a["scope_achieved"]), len(scope_ids), s_a["n_achieved"], len(s_a["weights"]), s_a["safety_ok"],
                s_a["sim_time_s"], s_a["episode_wall_s"], int(s_a["calls"].get("get_camera", 0)), s_a["camera_frames"], s_a.get("renderer"),
                layout_nominal(s_a), s_a["end_reason"],
                pk.get("R1_N", 0.0), su.get("R2_N", 0.0), su.get("R3_Nm", 0.0), pk.get("R4_m_s", 0.0), [v["what"] for v in s_a["violations"]][:3]))
        agent.append(s_a)
        step_calls += s_a["step_calls"] + s_a["calls"].get("reference_assemble", 0)
        illegal += s_a["illegal_call_count"]
        metrics["agent_wire_errors"] += 1.0 if s_a["end_reason"] == "wire_error" else 0.0
        metrics["remote_sockets_seen"] = max(metrics["remote_sockets_seen"], float(s_a["remote_sockets_seen"]))
        metrics["net_outbound_events"] += float(s_a.get("net_outbound", 0))
        for ck, cv in (s_a.get("net_delta") or {}).items():
            metrics["net_" + ck] = metrics.get("net_" + ck, 0.0) + float(max(0, cv))
    vals = [float(s["scope_score"]) * (1.0 if s["safety_ok"] else factor) for s in agent]
    steps_gate, clean_gate, net_gate, gate_reasons = episode_gates(step_calls, rn.unclean_runs, rn.left, metrics["net_outbound_events"],
                                                                   metrics["remote_sockets_seen"])
    gate = steps_gate * clean_gate * net_gate
    rewards["contract_ok"] = gate
    for why in gate_reasons:
        log("[grader] contract_ok 0: %s" % why)
    if gate:
        log("[grader] contract_ok 1: the episodes' gates passed (%d step calls, no process left after the kill, no network activity)" % step_calls)
    rewards["assembly_score"] = float(max(0.0, min(1.0, gate * float(np.mean(vals)))))
    metrics["scope_score_mean_ungated"] = float(np.mean([s["scope_score"] for s in agent]))
    metrics["stage_score_all_mates_mean"] = float(np.mean([s["stage_score"] for s in agent]))
    metrics["scope_mates_achieved_mean"] = float(np.mean([len(s["scope_achieved"]) for s in agent]))
    metrics["scope_mates_achieved_min"] = float(min(len(s["scope_achieved"]) for s in agent))
    metrics["scope_all_mates_rate"] = float(np.mean([1.0 if len(s["scope_achieved"]) == len(scope_ids) else 0.0 for s in agent]))
    metrics["mates_achieved_mean"] = float(np.mean([s["n_achieved"] for s in agent]))
    metrics["mates_achieved_min"] = float(min(s["n_achieved"] for s in agent))
    metrics["mates_fastened_mean"] = float(np.mean([s["n_fastened"] for s in agent]))
    metrics["all_mates_rate"] = float(np.mean([1.0 if s["all_mates"] else 0.0 for s in agent]))
    metrics["safety_violation_rate"] = float(np.mean([0.0 if s["safety_ok"] else 1.0 for s in agent]))
    metrics["mean_sim_time_s"] = float(np.mean([s["sim_time_s"] for s in agent]))
    metrics["mean_episode_wall_s"] = float(np.mean([s["episode_wall_s"] for s in agent]))
    metrics["illegal_call_count"] = float(illegal)
    metrics["step_calls"] = float(step_calls)
    metrics["dropped_parts_mean"] = float(np.mean([len(s["dropped_parts"]) for s in agent]))
    metrics["lost_parts_mean"] = float(np.mean([len(s["lost_parts"]) for s in agent]))
    # the four rules' readings (R1, R4: the largest tick; R2, R3: the largest 0.2 s-sustained; pad grip: reported only)
    metrics["max_R1_panda_force_N"] = float(max(s["damage"]["peak"].get("R1_N", 0.0) for s in agent))
    metrics["max_R2_part_force_sustained_N"] = float(max(s["damage"]["sustained_0.2s"].get("R2_N", 0.0) for s in agent))
    metrics["max_R3_stop_torque_sustained_Nm"] = float(max(s["damage"]["sustained_0.2s"].get("R3_Nm", 0.0) for s in agent))
    metrics["max_R4_part_speed_m_s"] = float(max(s["damage"]["peak"].get("R4_m_s", 0.0) for s in agent))
    metrics["max_pad_force_N"] = float(max(s["damage"]["peak"].get("pad_N", 0.0) for s in agent))
    metrics["camera_calls_mean"] = float(np.mean([s["calls"].get("get_camera", 0) for s in agent]))
    metrics["camera_frames_mean"] = float(np.mean([s["camera_frames_total"] for s in agent]))
    for cam in ("top", "wrist"):
        metrics["camera_frames_mean." + cam] = float(np.mean([s["camera_frames"].get(cam, 0) for s in agent]))
    metrics["render_wall_s_mean"] = float(np.mean([s["render_wall_s"] for s in agent]))
    # which renderer produced the agent's frames (None when no frame was taken): the oracle asks each camera once first
    metrics["renderer_filament_rate"] = float(np.mean([1.0 if s.get("renderer") == "filament" else 0.0 for s in agent]))
    metrics["nominal_layout_rate"] = float(np.mean([1.0 if layout_nominal(s) else 0.0 for s in agent]))
    for mid in agent[0]["weights"]:
        metrics["mate_rate." + mid] = float(np.mean([1.0 if any(r["id"] == mid and r["achieved"] for r in s["mates"]) else 0.0 for s in agent]))
    metrics["fastened_off_design_rate"] = float(sum(len(s.get("off_design", [])) for s in agent)) / float(len(agent))
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
    metrics["runner_oom_adj_killed"] = float(rn.oom_killed)
    metrics["agent_processes_frozen"] = float(len(FREEZER[0].frozen)) if FREEZER[0] is not None else 0.0
    metrics.update(blas_pin_stats())        # after the episodes: every OpenBLAS the cell loaded is in the process by now
    log("[grader] numerics pin: env %.0f, programs' env %.0f, L2 size 256 %.0f, OpenBLAS kernel Haswell %.0f, numpy SIMD capped %.0f" % (
        metrics["blas_pin_env"], metrics["blas_pin_child_env"], metrics["blas_l2_size_256"], metrics["blas_kernel_haswell"],
        metrics["blas_numpy_simd_capped"]))
    metrics["verify_wall_s"] = time.time() - t_start
    shutil.rmtree(RUNS_ROOT, ignore_errors=True)


if __name__ == "__main__":
    main()
