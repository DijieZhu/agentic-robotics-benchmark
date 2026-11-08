"""Grader of robotic_arm_cad_design_from_photos (information tier, hardware-design stage, FreeCAD MCP version). Root,
verify stage, agent gone.

Where it runs: a separate verifier sandbox (task.toml `environment_mode = "separate"`), started fresh from the task image.
The engine restores only the declared artifact /home/user/submission into it (written by root, before this folder is
uploaded), so nothing the agent left anywhere else is present. setup.sh does not run there and nothing it staged is read:
the givens the grader uses are this folder's own copies (given/). The lockdown below works the same in a shared sandbox.
In the verifier, the agent account's processes are the image's own desktop, and the restored submission is root-owned
and older than the stage upload, so the late-write gate has nothing to flag there (the engine copied the submission
when the agent phase ended).

  0. FIRST THING, stdlib only (`_early_lockdown`, as in the predecessor task): record when the engine began uploading this
     folder, freeze and kill every process of the agent account (a survivor that cannot be frozen voids the run: task
     error), lock the stage,
     take a root-owned 0700 snapshot of the submission that every later read uses, and zero the gate cad_valid (stat
     submission_final = 0) when an entry of the submission was written at or after the stage upload began. A submission
     root that is a symbolic link, or that resolves outside the agent's home, is not read at all. The agent account stays
     stopped until the verdict is written (`arm/verify_freeze.py`): supervisord respawns the desktop it runs as that account.
  1. Tool use from the engine's solver trajectory (ATIF, ALE_TRAJECTORY_PATH), stats only: how many calls went to the
     task's FreeCAD MCP server, to its code tools and to the harness's own tools (the instruction asks for the MCP server;
     nothing is scored on it).
  2. The submitted FreeCAD document (snapshot `arm.FCStd`) is read by FreeCAD's headless interpreter as an unprivileged
     account with a time limit (fcstd_extract.py): world-frame shapes of the assembly's components and its joints.
  3. fcstd_convert.py infers the mechanism (parts by tag label, servos by shape, links by Fixed joints, the revolute
     chain from the grounded base), aligns the base with the unit's, and writes the tier submission (assembly.json +
     parts in link frames). A document that cannot be read or whose mechanism differs from the photographed unit's
     scores cad_valid = 0 with the reason recorded.
  4. cad_photo_grade.grade on that submission against reference/ (the hidden unit's GT, joint senses = the driving
     servos' +z): checks cad_valid, public_model_reuse, cad_json_consistent, servos_mounted, the first three gating every
     criterion (task_grading.py: servos_mounted is reported, not a gate; the floors softened); criteria kinematic_fidelity,
     feature_fidelity, shape_fidelity, mass_properties, component_fit, interference_free (weights in grader_config.json)
     with the photographed-arm floors; no template-relative credit (the agent was given no templates). design_score =
     the weighted mean; the verdict's single gated `reward` comes from the benchmark kit (robotics_grader.ratio_rewards
     against anchor.json: value 1.0 = the oracle, full_at 1.0), so reward = design_score.
No agent code runs here (the document is data that FreeCAD reads). Knobs: grader_config.json.
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
DOCUMENT = "arm.FCStd"
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
    """The submitted document: arm.FCStd, else the only *.FCStd in the submission root (any case)."""
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


def extract(doc_path, servo_step, timeout_s, stats):
    """fcstd_extract.py under freecadcmd as an unprivileged account, in a private work folder; returns that folder."""
    work = tempfile.mkdtemp(prefix="arb_extract_")
    os.makedirs(os.path.join(work, "home"))
    for src, name in ((doc_path, "doc.FCStd"), (servo_step, "servo.step"), (os.path.join(STAGE_DIR, "fcstd_extract.py"), "fcstd_extract.py")):
        shutil.copyfile(src, os.path.join(work, name))
    cmd_py = ("import sys; sys.argv = ['fcstd_extract.py', %r, %r, %r]; exec(open(%r).read())" % (
        os.path.join(work, "doc.FCStd"), os.path.join(work, "servo.step"), os.path.join(work, "out"), os.path.join(work, "fcstd_extract.py")))
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
    try:
        r = subprocess.run(argv, cwd=work, env=env, capture_output=True, text=True, timeout=timeout_s)
        stats["extract_exit"] = float(r.returncode)
        tail = (r.stdout or "")[-600:] + (r.stderr or "")[-600:]
    except subprocess.TimeoutExpired:
        stats["extract_exit"] = -1.0
        tail = "timed out after %d s" % timeout_s
    stats["extract_s"] = time.time() - t
    return os.path.join(work, "out"), tail


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


def write_verdict(G, checks, rewards, weights, stats, details):
    """One gated `reward` in the benchmark kit's envelope (robotics_grader.ratio_rewards against anchor.json: value 1.0 =
    the oracle, full_at 1.0, so reward = design_score); the checks, the criteria and every statistic go to `metrics`, the
    per-check and per-criterion diagnostics into the verdict's `diagnostic` text (the run record keeps them). In the
    engine (ALE_VERIFY_PYTHON set) the kit writes /logs/verifier/reward.json and verification.json through ale_verify;
    a host-side run writes the plain envelope to ROBOTICS_GRADER_VERDICT_PATH, else verdict.local.json in the stage
    folder."""
    from robotics_grader import ratio_rewards, read_anchor
    from robotics_grader import write_verdict as kit_write
    crit = G.CRITERIA
    overall = sum(float(rewards[c]) * weights[c] for c in crit) / sum(weights[c] for c in crit)
    anchor, meta = read_anchor()
    extra = {k: float(v) for k, v in stats.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}
    extra.update({k: float(rewards[k]) for k in list(checks) + list(crit)})
    verdict = ratio_rewards(overall, anchor, higher_is_better=meta.get("direction", "higher") == "higher",
                            metric_name=meta.get("metric", "design_score"), extra=extra,
                            full_at=meta.get("full_at", 1.0))
    text = "; ".join("%s %.4f: %s" % (k, float(rewards[k]), " | ".join(str(d) for d in details.get(k, []))[:150])
                     for k in list(checks) + list(crit))
    if os.environ.get("ALE_VERIFY_PYTHON"):
        recorded = {"rewards": verdict["rewards"], "metrics": _finite_metrics(verdict["metrics"])}
        # text[:2000]: the diagnostic cap of the earlier ale_verify API, kept so the recorded text is unchanged
        kit_write(recorded, diagnostic=text[:2000])
        return overall
    kit_write(verdict, path=os.environ.get("ROBOTICS_GRADER_VERDICT_PATH") or os.path.join(STAGE_DIR, "verdict.local.json"),
              diagnostic=text)
    return overall


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
        import cad_photo_grade as G
        import fcstd_convert
        import task_grading
        checks = list(task_grading.ALL_CHECKS)
        with open(os.path.join(STAGE_DIR, "grader_config.json")) as fh:
            cfg = json.load(fh)
        weights = dict(G.DEFAULT_WEIGHTS)
        weights.update(cfg.get("weights", {}))
        rewards = {k: 0.0 for k in checks + list(G.CRITERIA)}
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
        given = G.load_given(os.path.join(STAGE_DIR, "given"))
        gates = task_grading.apply(G, given, cfg)
        ref_dir = os.path.join(STAGE_DIR, "reference")
        ref = G.load_reference(ref_dir, given)
        cases = G.make_cases(rng, n_fk=int(cfg.get("fk_configs", 64)), n_int=int(cfg.get("interference_configs", 64)),
                             n_decl=int(cfg.get("declared_range_configs", 16)), n_mjcf=int(cfg.get("mjcf_configs", 16)))
        stats["load_s"] = time.time() - t1
        stats["grader_exception"] = 0.0
        doc, note = find_document(sub_dir)
        conv = {"ok": False, "errors": [note or "no document"], "stats": {}, "details": {}}
        conv_dir = tempfile.mkdtemp(prefix="arb_converted_")
        if doc is not None:
            out_dir, tail = extract(doc, os.path.join(STAGE_DIR, "given", "servo", "servo.step"),
                                    int(cfg.get("extract_timeout_s", 600)), stats)
            t2 = time.time()
            try:
                conv = fcstd_convert.convert(out_dir, ref_dir, conv_dir)
            except Exception:  # noqa: BLE001 - a document that crashes the conversion earns 0, not a task error
                import traceback
                conv = {"ok": False, "errors": ["conversion failed: " + traceback.format_exc()[-400:]], "stats": {}, "details": {}}
                stats["grader_exception"] = 1.0
            stats["convert_s"] = time.time() - t2
            if not conv["ok"] and stats.get("extract_exit", 0.0) != 0.0:
                conv["errors"].append("FreeCAD reader: " + tail[-300:])
            if note:
                details.setdefault("cad_valid", []).append(note)
        for k, v in conv.get("stats", {}).items():
            stats["doc_" + k] = float(v)
        details["mechanism"] = conv.get("details", {})
        if conv["ok"]:
            t3 = time.time()
            try:
                g_rewards, gstats, g_details = G.grade(conv_dir, given, ref, cases, log=log, templates=None)
                rewards.update(g_rewards)
                details.update({k: list(details.get(k, [])) + list(v) for k, v in g_details.items()})
                for k, v in gstats.items():
                    if isinstance(v, (int, float)) and not isinstance(v, bool):
                        stats[k] = float(v)
            except Exception:  # noqa: BLE001
                import traceback
                log("exception while grading the submission:\n" + traceback.format_exc())
                details["cad_valid"] = ["grader exception on this submission: " + traceback.format_exc()[-400:]]
                stats["grader_exception"] = 1.0
            stats["grade_s"] = time.time() - t3
        else:
            details["cad_valid"] = list(details.get("cad_valid", [])) + conv["errors"][:20]
        final = 0.0 if (early["submission_late_writes"] > 0 or early["submission_root_rejected"] > 0) else 1.0
        stats["submission_final"] = final
        if final < 1.0:
            rewards["cad_valid"] = 0.0
            details["cad_valid"] = ["submission_final: " + "; ".join(early["reasons"])] + list(details.get("cad_valid", []))
        gate = 1.0
        for g in gates:
            gate *= float(rewards[g])
        for c in G.CRITERIA:
            rewards[c] = float(rewards[c]) * (1.0 if gate > 0 else 0.0)
        stats["verify_s"] = time.time() - t0
        stats["agent_processes_frozen"] = float(len(freezer.frozen))
        details.setdefault("servos_mounted", []).append("reported, not a gate (task_grading.py)")
        overall = write_verdict(G, checks, rewards, weights, stats, details)
        log("checks {} criteria {} overall {:.4f} ({:.0f} s)".format({g: rewards[g] for g in checks},
                                                                   {c: round(rewards[c], 4) for c in G.CRITERIA}, overall, stats["verify_s"]))
    finally:
        _restore(early)
        freezer.stop()


if __name__ == "__main__":
    main()
