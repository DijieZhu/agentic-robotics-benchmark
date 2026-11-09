"""Verify-start isolation, the unprivileged native CAD extraction and the grading child (stdlib only; adapted from the
arm tasks: robotic_arm_cad_design_from_photos, bimanual_cad_design_from_photos)."""
import json
import os
import pwd
import shutil
import signal
import stat as statmod
import subprocess
import sys
import tempfile
import time

sys.dont_write_bytecode = True
STAGE_DIR = os.environ.get("ALE_STAGE_DIR", os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, STAGE_DIR)
HOME = os.environ.get("ALE_HOME", "/home/user")
AGENT_USER = os.path.basename(HOME.rstrip("/")) or "user"
SUBMISSION = os.path.join(HOME, "submission")
SNAP_ROOT = os.environ.get("ARB_SNAPSHOT_ROOT", "/opt/arb-verify")
SNAPSHOT = os.path.join(SNAP_ROOT, "submission")
LATE_GATE_MAX_GAP_S = 600.0
FREECADCMD = os.environ.get("ARB_FREECADCMD", "/opt/freecad/usr/bin/freecadcmd")
SYSTEM_PATH = "/usr/sbin:/usr/bin:/sbin:/bin"
EXTRACT_USER = "nobody"
DOCUMENT = "drone.FCStd"
MCP_TOOL_PREFIX = "mcp__freecad__"

from arm.assembly_six import lockdown  # noqa: E402  (stdlib only; vendored)
from arm import verify_freeze  # noqa: E402  (stdlib only; vendored)


def log(msg):
    print("[grader] " + str(msg), file=sys.stderr, flush=True)


def lock_stage(root):
    """chmod 0700 every directory and 0600 every file under `root` (root only); the modes to restore, or None."""
    if os.getuid() != 0:
        return None
    saved = []
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in [""] + filenames:
            p = os.path.join(dirpath, name) if name else dirpath
            try:
                mode = statmod.S_IMODE(os.lstat(p).st_mode)
            except OSError:
                continue
            saved.append((p, mode))
            try:
                os.chmod(p, 0o700 if not name else 0o600)
            except OSError:
                pass
    return saved


def unlock_stage(saved):
    for p, mode in saved or []:
        try:
            os.chmod(p, mode)
        except OSError:
            pass


def _submission_root_ok():
    """(ok, reason): the submission root must be a real directory inside the agent's home, never a symbolic link. A
    missing root is ok (graded as missing)."""
    try:
        st = os.lstat(SUBMISSION)
    except FileNotFoundError:
        return True, None
    except OSError as exc:
        return False, "the submission root %s cannot be inspected (%s)" % (SUBMISSION, exc)
    if statmod.S_ISLNK(st.st_mode):
        try:
            target = os.readlink(SUBMISSION)
        except OSError:
            target = "?"
        return False, "the submission root %s is a symbolic link (to %s): not read" % (SUBMISSION, target)
    home = os.path.realpath(HOME)
    real = os.path.realpath(SUBMISSION)
    if os.path.commonpath([home, real]) != home:
        return False, "the submission root %s resolves outside %s (%s): not read" % (SUBMISSION, HOME, real)
    return True, None


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
    """SIGSTOP every live process of `uid` until a pass finds none still running (a respawn that appears meanwhile is
    stopped by the next pass) or `bound_s` elapsed; returns the PIDs still running (empty: every one is stopped)."""
    deadline = time.time() + float(bound_s)
    while True:
        for pid in lockdown.live_uid_processes(uid):
            try:
                os.kill(pid, signal.SIGSTOP)
            except OSError:
                pass
        time.sleep(0.05)
        running = [pid for pid in lockdown.live_uid_processes(uid) if _proc_state(pid) not in ("T", "t", "X")]
        if not running or time.time() > deadline:
            return running


def _early_lockdown(freezer=None):
    """Record the stage upload time, stop and kill every process of the agent account (in a separate verifier, the
    image's own desktop; a survivor is frozen, and only one still running then voids the run), freeze the account,
    lock the stage, snapshot the submission into a root-only folder and count the entries written after the stage
    upload began (the late-write gate)."""
    out = {"agent_processes_killed": 0.0, "agent_processes_left": 0.0, "agent_processes_running": 0.0,
           "stage_locked": 0.0, "stage_parent_locked": 0.0,
           "submission_late_writes": 0.0, "late_write_gate_active": 0.0, "submission_snapshot": 0.0,
           "submission_root_rejected": 0.0, "saved": None, "parent": None, "reasons": [], "t_ref": None}
    root_ok, why = _submission_root_ok()
    if not root_ok:
        out["submission_root_rejected"] = 1.0
        out["reasons"].append(why)
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
            # The image's supervisord respawns the account's desktop as fast as it is killed, so in the separate
            # verifier a process that outlives the kill is the image's own. It is frozen (a stopped process cannot exit,
            # so it is not respawned); only one still running once frozen voids the run.
            out["agent_processes_running"] = float(len(_still_running(uid)))
            log("%d process(es) of the agent account outlived the kill at verify start, %d still running after the "
                "freeze" % (left, out["agent_processes_running"]))
    if freezer is not None:
        freezer.start()
    out["saved"] = lock_stage(STAGE_DIR)
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
    cfg_json = os.path.join(STAGE_DIR, "grader_config.json")
    if gate_active and os.path.isfile(cfg_json):
        try:
            gate_active = (os.lstat(cfg_json).st_ctime - t_ref) <= LATE_GATE_MAX_GAP_S
        except OSError:
            pass
    out["late_write_gate_active"] = 1.0 if gate_active else 0.0
    if root_ok and os.path.isdir(SUBMISSION):
        if gate_active:
            late = lockdown.late_entries(SUBMISSION, t_ref)
            out["submission_late_writes"] = float(len(late))
            if late:
                out["reasons"].append("%d entr%s of the submission written after the agent phase ended (%s)" % (
                    len(late), "y" if len(late) == 1 else "ies", ", ".join(e[0] for e in late[:5])))
        os.makedirs(SNAP_ROOT, exist_ok=True)
        os.chmod(SNAP_ROOT, 0o700)
        lockdown.snapshot_dir(SUBMISSION, SNAPSHOT)
        out["submission_snapshot"] = 1.0
    return out


def _restore(early):
    try:
        if early.get("saved") is not None:
            unlock_stage(early["saved"])
        if early.get("parent"):
            os.chmod(early["parent"][0], early["parent"][1])
        mode = os.environ.get("ARB_STAGE_MODE")
        if mode:
            os.chmod(STAGE_DIR, int(mode, 8))
    except OSError:
        pass


def tool_usage(stats):
    """Stats from the solver trajectory: total tool calls, FreeCAD MCP calls, its code calls, Bash calls, and the number of
    distinct non-MCP tools used. Not a gate."""
    path = os.environ.get("ALE_TRAJECTORY_PATH")
    if not path or not os.path.isfile(path):
        return "no solver trajectory"
    try:
        with open(path) as fh:
            traj = json.load(fh)
    except (OSError, ValueError) as exc:
        return "the solver trajectory is unreadable (%s)" % str(exc)[:120]
    names = [str(tc.get("function_name", "")) for st in traj.get("steps", []) for tc in (st.get("tool_calls") or [])]
    mcp = [n for n in names if n.startswith(MCP_TOOL_PREFIX)]
    other = sorted(set(n for n in names if not n.startswith(MCP_TOOL_PREFIX)))
    stats["tool_calls"] = float(len(names))
    stats["mcp_tool_calls"] = float(len(mcp))
    stats["execute_code_calls"] = float(sum(1 for n in mcp if n.startswith(MCP_TOOL_PREFIX + "execute_code")))
    stats["bash_calls"] = float(sum(1 for n in names if n == "Bash"))
    stats["non_mcp_tool_kinds"] = float(len(other))
    return "%d tool calls, %d to the FreeCAD MCP server; other tools: %s" % (len(names), len(mcp), ", ".join(other)[:300] or "none")


def find_document(sub_dir):
    """The submitted document: drone.FCStd, else the only *.FCStd in the submission root (any case)."""
    if not os.path.isdir(sub_dir):
        return None, "no submission folder"
    p = os.path.join(sub_dir, DOCUMENT)
    if os.path.isfile(p) and not os.path.islink(p):
        return p, None
    cands = [f for f in os.listdir(sub_dir) if f.lower().endswith(".fcstd") and os.path.isfile(os.path.join(sub_dir, f))
             and not os.path.islink(os.path.join(sub_dir, f))]
    if len(cands) == 1:
        return os.path.join(sub_dir, cands[0]), "document read from %s (expected %s)" % (cands[0], DOCUMENT)
    return None, "no %s in the submission (%d FreeCAD documents found)" % (DOCUMENT, len(cands))


def extract(doc_path, timeout_s, stats):
    """fcstd_extract.py under freecadcmd as an unprivileged account, in a private work folder, with a time limit; returns
    (that folder's output, the reader's last words)."""
    work = tempfile.mkdtemp(prefix="arb_extract_")
    os.makedirs(os.path.join(work, "home"))
    for src, name in ((doc_path, "doc.FCStd"), (os.path.join(STAGE_DIR, "fcstd_extract.py"), "fcstd_extract.py")):
        shutil.copyfile(src, os.path.join(work, name))
    cmd_py = ("import sys; sys.argv = ['fcstd_extract.py', %r, %r]; exec(open(%r).read())" % (
        os.path.join(work, "doc.FCStd"), os.path.join(work, "out"), os.path.join(work, "fcstd_extract.py")))
    env = {"HOME": os.path.join(work, "home"), "XDG_CONFIG_HOME": os.path.join(work, "home", ".config"),
           "XDG_CACHE_HOME": os.path.join(work, "home", ".cache"), "XDG_DATA_HOME": os.path.join(work, "home", ".local", "share"),
           "PATH": "/usr/bin:/bin", "LANG": "C.UTF-8", "QT_QPA_PLATFORM": "offscreen", "OMP_NUM_THREADS": "1"}
    argv = [FREECADCMD, "-c", cmd_py]
    if os.getuid() == 0:
        uid = pwd.getpwnam(EXTRACT_USER)
        for dirpath, dirnames, filenames in os.walk(work):
            for n in [dirpath] + [os.path.join(dirpath, f) for f in dirnames + filenames]:
                os.chown(n, uid.pw_uid, uid.pw_gid)
        os.chmod(work, 0o700)
        runuser = shutil.which("runuser", path=SYSTEM_PATH) or "/usr/sbin/runuser"
        # runuser resets HOME to the account's (nobody: /nonexistent); FreeCAD needs a writable one for its settings
        argv = [runuser, "-u", EXTRACT_USER, "--", "/usr/bin/env"] + ["%s=%s" % kv for kv in sorted(env.items())] + argv
    t = time.time()
    try:
        r = subprocess.run(argv, cwd=work, env=env, capture_output=True, text=True, timeout=timeout_s)
        stats["extract_exit"] = float(r.returncode)
        tail = (r.stdout or "")[-600:] + (r.stderr or "")[-600:]
    except subprocess.TimeoutExpired:
        stats["extract_exit"] = -1.0
        tail = "timed out after %d s" % timeout_s
    stats["extract_s"] = time.time() - t
    return os.path.join(work, "out"), tail


def grade_child(extract_dir, purchased_root, cfg, stats):
    """grade.py in a child process with a time limit (an OCC call that hangs on a broken shape cannot be interrupted
    from Python): (its result or None, its last words)."""
    work = tempfile.mkdtemp(prefix="arb_grade_")
    out = os.path.join(work, "result.json")
    argv = [sys.executable, "-B", os.path.join(STAGE_DIR, "grade.py"), extract_dir, os.path.join(STAGE_DIR, "reference"),
            purchased_root, out, "--config", os.path.join(STAGE_DIR, "grader_config.json")]
    env = dict(os.environ, ARB_GRADER_CPUS=str(int(cfg.get("grader_cpus", 8))), OMP_NUM_THREADS="1",
               OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    timeout = int(cfg.get("grade_timeout_s", 1080))
    t = time.time()
    try:
        r = subprocess.run(argv, cwd=STAGE_DIR, env=env, capture_output=True, text=True, timeout=timeout)
        stats["grade_child_exit"] = float(r.returncode)
        tail = (r.stderr or "")[-1500:]
    except subprocess.TimeoutExpired:
        stats["grade_child_exit"] = -1.0
        return None, "the grade did not finish within %d s" % timeout
    finally:
        stats["grade_child_s"] = time.time() - t
    if not os.path.isfile(out):
        return None, "the grader child wrote no result: " + tail
    with open(out) as fh:
        return json.load(fh), tail
