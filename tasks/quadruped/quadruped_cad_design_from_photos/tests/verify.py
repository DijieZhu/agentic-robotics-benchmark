"""Grader of quadruped_cad_design_from_photos (information tier, hardware-design stage, FreeCAD MCP version). Root,
verify stage, agent gone. A task-local fork of the arm task's verify.py (the same structure; the quadruped's modules).

Where it runs: a separate verifier sandbox (task.toml `environment_mode = "separate"`), started fresh from the task image.
The engine restores only the declared artifact /home/user/submission into it (written by root, before this folder is
uploaded), so nothing the agent left anywhere else is present. setup.sh does not run there and nothing it staged is read:
the givens the grader uses are this folder's own copies (given/). The lockdown below works the same in a shared sandbox.
In the verifier, the agent account's processes are the image's own desktop, and the restored submission is root-owned
and older than the stage upload, so the late-write gate has nothing to flag there (the engine copied the submission
when the agent phase ended). given_hashes.json is the maintainers' pin of the staged givens (the payload's 139 files);
verify does not read it.

  0. FIRST THING, stdlib only (`_early_lockdown`, as in the arm task): record when the engine began uploading this
     folder, freeze and kill every process of the agent account (a survivor that cannot be frozen voids the run: task
     error), lock the stage,
     take a root-owned 0700 snapshot of the submission that every later read uses, and zero the gate cad_valid (stat
     submission_final = 0) when an entry of the submission was written at or after the stage upload began. A submission
     root that is a symbolic link, or that resolves outside the agent's home, is not read at all. The agent account stays
     stopped until the verdict is written (`arm/verify_freeze.py`): supervisord respawns the desktop it runs as that account.
  1. Tool use from the engine's solver trajectory (ATIF, ALE_TRAJECTORY_PATH), stats only: how many calls went to the
     task's FreeCAD MCP server, to its code tools and to the harness's own tools (nothing is scored on it).
  2. The submitted FreeCAD document (snapshot `quadruped.FCStd`, else the only *.FCStd) is read by FreeCAD's headless
     interpreter as an unprivileged account (nobody, a private HOME / XDG folder) with a time limit (fcstd_extract.py):
     the top-level Assembly's components in world coordinates (printed parts by label, the bought actuator cores and
     foot pads by shape against given/core.step and given/foot.step) and its joints.
  3. quad_convert.py infers the mechanism (parts by tag, links by Fixed joints, the revolute tree from the grounded
     body, the driving core of every joint, validity per leg), aligns the body with the hidden robot's (6 DoF) and
     writes it in the reference's own form (quad-cad-gt/1). A document that cannot be read, whose grounded group does
     not stand for the robot's body (it must carry more of the body's tags than of any other link's), with a closed
     loop or a joint of another type scores cad_valid = 0 with the reason recorded; a wrong leg only invalidates that
     leg's items, and a body part missing from the grounded group only costs its own items (GRADER_SPEC 4).
  4. quad_grade.grade on that folder against reference/ (the hidden robot; joint senses = the driving cores' +z): the
     gates cad_valid and public_model_reuse; the reported checks legs_valid, cores_mounted, cad_json_consistent; the
     criteria kinematic_fidelity, feature_fidelity, shape_fidelity, mass_properties, component_fit, interference_free
     with the shape floor and the per-leg foot-point floors (weights, bands and floors in grader_config.json).
     design_score = the weighted mean; the verdict's single gated `reward` comes from the benchmark kit
     (robotics_grader.ratio_rewards against anchor.json: value 1.0 = the oracle, full_at 1.0), so reward = design_score.
No agent code runs here (the document is data that FreeCAD reads). Knobs: grader_config.json. Hooks for host runs, as
the arm's: ALE_STAGE_DIR, ALE_HOME, ARB_SNAPSHOT_ROOT, ARB_FREECADCMD, ARB_SEEDS_PIN="hi:lo"; and the verdict file of a
run outside the engine, ROBOTICS_GRADER_VERDICT_PATH (default verdict.local.json in the stage folder).
"""
import json
import math
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
EXTRACT_USER = "nobody"
DOCUMENT = "quadruped.FCStd"
MCP_TOOL_PREFIX = "mcp__freecad__"
DIAGNOSTIC_ITEM_CHARS = 150
MAX_DIAGNOSTIC_CHARS = 2000          # the earlier ale_verify API's diagnostic cap, kept: the recorded text is unchanged
RESERVED_METRICS = ("reward", "ratio", "anchor_recorded")

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
    out = {"agent_processes_killed": 0.0, "agent_processes_left": 0.0, "agent_processes_running": 0.0,
           "stage_locked": 0.0, "stage_parent_locked": 0.0,
           "submission_late_writes": 0.0, "late_write_gate_active": 0.0, "submission_snapshot": 0.0, "submission_root_rejected": 0.0,
           "saved": None, "parent": None, "reasons": [], "t_ref": None}
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
            # verifier a process that outlives the kill is the image's own (2026-11-07: one respawned desktop process
            # outlived this kill and voided a run elsewhere). It is frozen (a stopped process cannot exit, so it is not
            # respawned); only one still running once frozen voids the run.
            out["agent_processes_running"] = float(len(_still_running(uid)))
            log("[grader] %d process(es) of the agent account outlived the kill at verify start, %d still running "
                "after the freeze" % (left, out["agent_processes_running"]))
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
    try:
        names = [str(tc.get("function_name", "")) for st in traj.get("steps", []) for tc in (st.get("tool_calls") or [])]
    except (AttributeError, TypeError) as exc:
        return "the solver trajectory has an unexpected form (%s)" % str(exc)[:120]
    mcp = [n for n in names if n.startswith(MCP_TOOL_PREFIX)]
    other = sorted(set(n for n in names if not n.startswith(MCP_TOOL_PREFIX)))
    stats["tool_calls"] = float(len(names))
    stats["mcp_tool_calls"] = float(len(mcp))
    stats["execute_code_calls"] = float(sum(1 for n in mcp if n.startswith(MCP_TOOL_PREFIX + "execute_code")))
    stats["bash_calls"] = float(sum(1 for n in names if n == "Bash"))
    stats["non_mcp_tool_kinds"] = float(len(other))
    return "%d tool calls, %d to the FreeCAD MCP server; other tools: %s" % (len(names), len(mcp), ", ".join(other)[:300] or "none")


def find_document(sub_dir):
    """The submitted document: quadruped.FCStd, else the only *.FCStd in the submission root (any case)."""
    if not os.path.isdir(sub_dir):
        return None, "no submission folder"
    p = os.path.join(sub_dir, DOCUMENT)
    if os.path.isfile(p) and not os.path.islink(p):
        return p, None
    cands = sorted(f for f in os.listdir(sub_dir) if f.lower().endswith(".fcstd") and os.path.isfile(os.path.join(sub_dir, f))
                   and not os.path.islink(os.path.join(sub_dir, f)))
    if len(cands) == 1:
        return os.path.join(sub_dir, cands[0]), "document read from %s (expected %s)" % (cands[0], DOCUMENT)
    return None, "no %s in the submission (%d FreeCAD documents found)" % (DOCUMENT, len(cands))


def _kill_group(proc):
    """SIGKILL the extractor's whole process group (runuser forks: killing it alone would orphan FreeCAD)."""
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (OSError, ProcessLookupError):
        try:
            proc.kill()
        except OSError:
            pass


def _strip_links(root):
    """Remove every symbolic link or special file the unprivileged reader left in its output (root reads it next)."""
    removed = 0
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        for name in list(dirnames) + filenames:
            p = os.path.join(dirpath, name)
            try:
                st = os.lstat(p)
            except OSError:
                continue
            if statmod.S_ISLNK(st.st_mode) or not (statmod.S_ISREG(st.st_mode) or statmod.S_ISDIR(st.st_mode)):
                try:
                    os.unlink(p)
                    removed += 1
                except OSError:
                    pass
                if name in dirnames:
                    dirnames.remove(name)
    return removed


def extract(doc_path, classes, timeout_s, stats):
    """fcstd_extract.py under freecadcmd as an unprivileged account, in a private work folder; returns (its output
    folder, the tail of its log). `classes` = [[class, STEP path]] of the bought components (quad_task.component_classes):
    the STEPs are copied into the work folder (the stage is closed to that account) and passed as a JSON file."""
    import quad_task as Q
    work = tempfile.mkdtemp(prefix="arb_extract_")
    os.makedirs(os.path.join(work, "home"))
    copies = [(doc_path, "doc.FCStd"), (os.path.join(STAGE_DIR, "fcstd_extract.py"), "fcstd_extract.py")]
    pairs = []
    for cls, step in classes:
        name = "class_%s.step" % "".join(c if c.isalnum() else "_" for c in str(cls))
        copies.append((step, name))
        pairs.append([str(cls), os.path.join(work, name)])
    for src, name in copies:
        shutil.copyfile(src, os.path.join(work, name))
    with open(os.path.join(work, "classes.json"), "w") as fh:
        json.dump({"classes": pairs, "tag_pattern": Q.TAG_PATTERN}, fh)
    out_dir = os.path.join(work, "out")
    cmd_py = ("import sys; sys.argv = ['fcstd_extract.py', %r, %r, %r]; exec(open(%r).read())" % (
        os.path.join(work, "doc.FCStd"), os.path.join(work, "classes.json"), out_dir, os.path.join(work, "fcstd_extract.py")))
    env = {"HOME": os.path.join(work, "home"), "XDG_CONFIG_HOME": os.path.join(work, "home", ".config"),
           "XDG_CACHE_HOME": os.path.join(work, "home", ".cache"), "XDG_DATA_HOME": os.path.join(work, "home", ".local", "share"),
           "PATH": "/usr/bin:/bin", "LANG": "C.UTF-8", "QT_QPA_PLATFORM": "offscreen"}
    argv = [FREECADCMD, "-c", cmd_py]
    if os.getuid() == 0:
        uid = pwd.getpwnam(EXTRACT_USER)
        for dirpath, dirnames, filenames in os.walk(work):
            for n in [dirpath] + [os.path.join(dirpath, f) for f in dirnames + filenames]:
                os.chown(n, uid.pw_uid, uid.pw_gid)
        os.chmod(work, 0o700)
        runuser = shutil.which("runuser", path="/usr/sbin:/usr/bin:/sbin:/bin") or "/usr/sbin/runuser"
        # runuser resets HOME to the account's (nobody: /nonexistent); FreeCAD needs a writable one for its settings
        argv = [runuser, "-u", EXTRACT_USER, "--", "/usr/bin/env"] + ["%s=%s" % kv for kv in sorted(env.items())] + argv
    t = time.time()
    # the reader's console output goes to a root-owned temporary file (FreeCAD prints progress text without bound; only
    # the tail is kept), its whole process group is killed at the time limit
    with tempfile.TemporaryFile() as console:
        proc = subprocess.Popen(argv, cwd=work, env=env, stdin=subprocess.DEVNULL, stdout=console, stderr=subprocess.STDOUT,
                                start_new_session=True)
        try:
            proc.wait(timeout=timeout_s)
            stats["extract_exit"] = float(proc.returncode)
            tail = None
            _kill_group(proc)                   # anything the reader left running in its group stops here
        except subprocess.TimeoutExpired:
            _kill_group(proc)
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                pass
            stats["extract_exit"] = -1.0
            stats["extract_timed_out"] = 1.0
            tail = "timed out after %d s" % timeout_s
            try:                                # a record written while being killed is not trusted
                os.remove(os.path.join(out_dir, "model.json"))
            except OSError:
                pass
        if tail is None:
            size = console.seek(0, os.SEEK_END)
            console.seek(max(0, size - 1200))
            tail = console.read().decode("utf-8", errors="replace")
    stats["extract_s"] = time.time() - t
    stats["extract_entries_removed"] = float(_strip_links(work))
    return out_dir, tail


def _finite_stats(d):
    """The numeric, finite entries of a statistics dict as floats (the engine's record refuses NaN and infinity)."""
    out = {}
    for k, v in (d or {}).items():
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            try:
                import numpy as np
                if not isinstance(v, (np.integer, np.floating)):
                    continue
            except ImportError:
                continue
        f = float(v)
        if math.isfinite(f):
            out[str(k)] = f
    return out


def design_score(rewards, weights, criteria):
    """The weighted mean of the criteria (quad_grade's formula: math.fsum on both sides, so a perfect run is exactly 1)."""
    sw = math.fsum(float(weights[c]) for c in criteria)
    if sw <= 0:
        return 0.0
    return max(0.0, min(1.0, math.fsum(float(weights[c]) * float(rewards[c]) for c in criteria) / sw))


def diagnostic_text(rewards, details, checks, criteria, stats):
    """One line per check, the leg verdicts and one per criterion, each item capped, the whole within
    MAX_DIAGNOSTIC_CHARS."""
    parts = []
    for k in checks:
        parts.append("%s %.4f: %s" % (k, float(rewards[k]), " | ".join(str(d) for d in details.get(k, []))[:DIAGNOSTIC_ITEM_CHARS]))
    if details.get("legs_valid"):
        parts.append("legs_valid %d: %s" % (int(stats.get("legs_valid", 0)), " | ".join(str(d) for d in details["legs_valid"])[:DIAGNOSTIC_ITEM_CHARS]))
    for k in criteria:
        parts.append("%s %.4f: %s" % (k, float(rewards[k]), " | ".join(str(d) for d in details.get(k, []))[:DIAGNOSTIC_ITEM_CHARS]))
    return "; ".join(parts)[:MAX_DIAGNOSTIC_CHARS]


def _finite_metrics(metrics):
    """`metrics` in name order, finite numbers only: ale_verify.write refuses NaN and infinity, so a statistic that is
    not a finite number stays out of the engine's record instead of failing the verdict."""
    out = {}
    for k, v in sorted(metrics.items()):
        try:
            x = float(v)
        except (TypeError, ValueError):
            continue
        if math.isfinite(x):
            out[k] = x
    return out


def write_verdict(checks, criteria, rewards, weights, stats, details):
    """One gated `reward` in the benchmark kit's envelope (robotics_grader.ratio_rewards against anchor.json: value 1.0 =
    the oracle, full_at 1.0, so reward = design_score); the checks, the criteria and every statistic go to `metrics`, the
    per-check and per-criterion diagnostics into the verdict's `diagnostic` text (the run record keeps them). In the
    engine (ALE_VERIFY_PYTHON set) the kit writes /logs/verifier/reward.json and verification.json through ale_verify;
    a host-side run writes the plain envelope to ROBOTICS_GRADER_VERDICT_PATH, else to verdict.local.json in the stage
    folder."""
    from robotics_grader import ratio_rewards, read_anchor
    from robotics_grader import write_verdict as kit_write
    overall = design_score(rewards, weights, criteria)
    anchor, meta = read_anchor()
    metric = meta.get("metric", "design_score")
    extra = {k: v for k, v in _finite_stats(stats).items() if k not in RESERVED_METRICS and k != metric}
    extra.update({k: float(rewards[k]) for k in list(checks) + list(criteria)})
    verdict = ratio_rewards(overall, anchor, higher_is_better=meta.get("direction", "higher") == "higher",
                            metric_name=metric, extra=extra, full_at=meta.get("full_at", 1.0))
    text = diagnostic_text(rewards, details, checks, criteria, stats)
    if os.environ.get("ALE_VERIFY_PYTHON"):
        recorded = {"rewards": verdict["rewards"], "metrics": _finite_metrics(verdict["metrics"])}
        kit_write(recorded, diagnostic=text[:MAX_DIAGNOSTIC_CHARS])
        return overall, verdict
    local = os.environ.get("ROBOTICS_GRADER_VERDICT_PATH") or os.path.join(STAGE_DIR, "verdict.local.json")
    kit_write(verdict, path=local, diagnostic=text)
    return overall, verdict


def _budget(stats, key, stage, limits):
    """Record whether a stage stayed within its budget (grader_config.json time_limits_s; statistics, not a gate)."""
    lim = limits.get(stage)
    if lim is not None and key in stats:
        stats["over_budget_" + stage] = 1.0 if stats[key] > float(lim) else 0.0


def main():
    t0 = time.time()
    freezer = verify_freeze.AgentFreezer.for_user(AGENT_USER)
    early = _early_lockdown(freezer)
    stats = {k: v for k, v in early.items() if isinstance(v, float)}
    stats["lockdown_s"] = time.time() - t0
    try:
        if early["agent_processes_running"] > 0:
            raise RuntimeError("%d agent process(es) survived the verify-start kill and could not be stopped: run void"
                               % early["agent_processes_running"])
        import numpy as np
        import quad_task as Q
        import quad_convert
        import quad_grade as G
        cfg = Q.load_config(STAGE_DIR)          # grader_config.json over the converter's defaults (quad_grade merges its own)
        checks = list(G.CHECKS)
        criteria = list(G.CRITERIA)
        weights = dict(G.DEFAULT_CONFIG["weights"])
        weights.update(cfg.get("weights", {}))
        gates = list(cfg.get("gates", G.DEFAULT_CONFIG["gates"]))
        limits = dict(G.DEFAULT_CONFIG.get("time_limits_s", {}))
        limits.update(cfg.get("time_limits_s", {}))
        rewards = {k: 0.0 for k in checks + criteria}
        details = {}
        details["tool_usage"] = [tool_usage(stats)]
        sub_dir = SNAPSHOT if early["submission_snapshot"] else SUBMISSION
        if early["submission_root_rejected"] > 0:
            sub_dir = tempfile.mkdtemp(prefix="arb_rejected_submission_")
        pin = os.environ.get("ARB_SEEDS_PIN")
        if pin:
            hi, lo = [int(x) for x in pin.split(":")]
        else:
            hi, lo = int.from_bytes(os.urandom(4), "big"), int.from_bytes(os.urandom(4), "big")
        stats["hidden_seed_hi"], stats["hidden_seed_lo"] = float(hi), float(lo)
        rng = np.random.default_rng(np.random.SeedSequence([hi, lo]))
        t1 = time.time()
        given_dir = os.path.join(STAGE_DIR, "given")
        ref_dir = os.path.join(STAGE_DIR, "reference")
        Q.load_spec(ref_dir)                      # the task's own folders: a broken one raises (task error, not a 0)
        prepared = G.prepare(ref_dir, given_dir, cfg)
        classes = Q.component_classes(given_dir)
        if sorted(c for c, _p in classes) != sorted(Q.COMPONENT_CLASSES):
            raise RuntimeError("given/ lacks the STEP of a bought component (%s found)" % [c for c, _p in classes])
        stats["load_s"] = time.time() - t1
        stats["grader_exception"] = 0.0
        doc, note = find_document(sub_dir)
        conv = {"ok": False, "errors": [note or "no document"], "stats": {}, "details": {}}
        conv_dir = tempfile.mkdtemp(prefix="arb_converted_")
        if doc is not None:
            out_dir, tail = extract(doc, classes, int(cfg.get("extract_timeout_s", limits.get("extract", 900))), stats)
            _budget(stats, "extract_s", "extract", limits)
            t2 = time.time()
            try:
                conv = quad_convert.convert(out_dir, ref_dir, given_dir, conv_dir, cfg)
            except Q.SpecError:
                raise
            except Exception:  # noqa: BLE001 - a document that crashes the conversion earns 0, not a task error
                import traceback
                conv = {"ok": False, "errors": ["conversion failed: " + traceback.format_exc()[-400:]], "stats": {}, "details": {}}
                stats["grader_exception"] = 1.0
            stats["convert_s"] = time.time() - t2
            _budget(stats, "convert_s", "convert", limits)
            if not conv["ok"] and stats.get("extract_exit", 0.0) != 0.0:
                conv["errors"].append("FreeCAD reader: " + tail[-300:])
            if note:
                details.setdefault("cad_valid", []).append(note)
        for k, v in _finite_stats(conv.get("stats", {})).items():
            stats["doc_" + k] = v
        stats["doc_legs_valid"] = float(conv.get("legs_valid", 0) or 0)
        details["mechanism"] = conv.get("details", {})
        if conv.get("warnings"):
            details["mechanism_warnings"] = [str(w)[:300] for w in conv["warnings"][:12]]
        if conv["ok"]:
            t3 = time.time()
            try:
                res = G.grade(conv_dir, ref_dir, given_dir, cfg, rng, log=log, prepared=prepared)
                rewards.update({k: float(res["rewards"][k]) for k in checks + criteria})
                details.update({k: list(details.get(k, [])) + list(v) for k, v in res["details"].items()})
                stats.update(_finite_stats(res["stats"]))
                if res["stats"].get("grader_exception"):
                    stats["grader_exception"] = 1.0
            except Exception:  # noqa: BLE001
                import traceback
                log("exception while grading the submission:\n" + traceback.format_exc())
                rewards = {k: 0.0 for k in checks + criteria}
                details["cad_valid"] = ["grader exception on this submission: " + traceback.format_exc()[-400:]]
                stats["grader_exception"] = 1.0
            stats["grade_s"] = time.time() - t3
            _budget(stats, "grade_s", "grade", limits)
        else:
            details["cad_valid"] = list(details.get("cad_valid", [])) + [str(e)[:400] for e in conv["errors"][:20]]
        final = 0.0 if (early["submission_late_writes"] > 0 or early["submission_root_rejected"] > 0) else 1.0
        stats["submission_final"] = final
        if final < 1.0:
            rewards["cad_valid"] = 0.0
            details["cad_valid"] = ["submission_final: " + "; ".join(early["reasons"])] + list(details.get("cad_valid", []))
        gate = 1.0
        for g in gates:
            gate *= float(rewards[g])
        for c in criteria:
            rewards[c] = float(rewards[c]) * (1.0 if gate > 0 else 0.0)
        stats["verify_s"] = time.time() - t0
        _budget(stats, "verify_s", "verify", limits)
        stats["agent_processes_frozen"] = float(len(freezer.frozen))
        overall, _verdict = write_verdict(checks, criteria, rewards, weights, stats, details)
        log("checks {} legs valid {:.0f} criteria {} design_score {:.4f} ({:.0f} s)".format(
            {g: rewards[g] for g in checks}, stats.get("legs_valid", 0.0), {c: round(rewards[c], 4) for c in criteria},
            overall, stats["verify_s"]))
    finally:
        _restore(early)
        freezer.stop()


if __name__ == "__main__":
    main()
