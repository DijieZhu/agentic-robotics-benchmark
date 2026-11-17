"""Grader of robotic_arm_robot_description_from_photos (the robot-description stage of the robotic-arm full-stack
workflow). Root, verify stage, in a SEPARATE verifier sandbox (task.toml):
a fresh sandbox from the task image into which the engine restored only /home/user/submission; the agent's sandbox is
gone, setup.sh has not run here.

  0. FIRST THING, stdlib only (`_early_lockdown`; test.sh has already chmod 0700 the stage and its parent): record when
     the engine began uploading this folder (earliest ctime of its files), freeze and kill every process of the agent
     account (a survivor that cannot be frozen voids the run: task error), keep every process of the account SIGSTOPped
     until the verdict is
     written (`verify_freeze.AgentFreezer`: the image's supervisord respawns its desktop services as the agent account
     within seconds of the kill; a stopped process cannot exit, so nothing is respawned), lock the stage (0700 / 0600,
     restored at the end), take a root-owned 0700 snapshot of the submission that every later read uses, and zero the
     gate model_loads (stat submission_final = 0) when an entry of the submission was written at or after the stage
     upload began. A submission root that is a symbolic link, or that resolves outside the agent's home, is not read
     (stat submission_root_rejected = 1); a tree above 200 MB or 20 000 entries is not copied (stat
     submission_oversize = 1); both grade as an empty folder. Whatever happens, a verdict is written.
  1. The photographed arm's ground truth from gt/ (robot.xml + assets, calibration.json, calibration_gt.json).
  2. Per-run hidden draws from OS entropy (stats hidden_seed_hi / hidden_seed_lo): 200 FK readings + 64 clean readings
     (Halton over the true firmware box in bus angles, true self / table contact rejected), 48 downstream targets, 5 jaw
     readings, the inertial / collision subsets, the IK seed.
  3. rd_info_grade.grade: gate model_loads x the criteria fk_tcp 0.30, fk_axes 0.15, downstream_reach 0.20, gripper 0.10,
     limits_actuators 0.05, inertial 0.10, collision 0.10; description_score = the weighted mean.
  4. The benchmark kit's envelope (robotics_grader.ratio_rewards against anchor.json: value 1.0 = the oracle, full_at
     1.0): one gated `reward` = description_score; the gate, the seven criteria and every stat go to `metrics`.
In the separate verifier the only processes of the agent account are the base image's desktop session and nothing is
written late (the engine restores the submission before it uploads tests/): the kill, the lock, the snapshot and the
late-write gate stay as defence in depth. The grader reads only tests/ (gt/, grader_config.json, anchor.json, the
grading modules, the kit) and the restored submission; tests/given_hashes.json is the maintainers' pin of the givens
(the agent's home copies do not exist here, so no statistic reads them).
No agent code runs here (MJCF, meshes and JSON are data; build.py is never executed). Knobs: grader_config.json.
"""
import json
import math
import os
import pwd
import signal
import stat as statmod
import sys
import time

sys.dont_write_bytecode = True
STAGE_DIR = os.environ.get("ALE_STAGE_DIR", os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, STAGE_DIR)
sys.path.insert(0, os.path.join(STAGE_DIR, "rd_info_check"))
HOME = os.environ.get("ALE_HOME", "/home/user")
AGENT_USER = os.path.basename(HOME.rstrip("/")) or "user"
SUBMISSION = os.path.join(HOME, "submission")
SNAP_ROOT = os.environ.get("ARB_SNAPSHOT_ROOT", "/opt/arb-verify")
SNAPSHOT = os.path.join(SNAP_ROOT, "submission")
LATE_GATE_MAX_GAP_S = 600.0
SUBMISSION_BYTES_CAP = 200 * 1024 * 1024          # = rd_info_contract.CAPS["max_submission_bytes"] (the gate's cap)
SUBMISSION_ENTRIES_CAP = 20000

import lockdown  # noqa: E402  (stdlib only; vendored)
import verify_freeze  # noqa: E402  (stdlib only; vendored)

# started right after the verify-start kill, stopped after the verdict (a no-op unless root and the account exists)
FREEZER = verify_freeze.AgentFreezer.for_user(AGENT_USER)


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
    """(ok, reason): the submission root must be a real directory inside the agent's home, never a symbolic link."""
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


def _submission_measure(root, max_entries=SUBMISSION_ENTRIES_CAP):
    """(entries, apparent bytes, allocated bytes) of the tree below `root` by lstat, never following a link; stops
    counting once max_entries is exceeded."""
    n, apparent, allocated = 0, 0, 0
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        for name in dirnames + filenames:
            n += 1
            if n > max_entries:
                return n, apparent, allocated
            try:
                st = os.lstat(os.path.join(dirpath, name))
            except OSError:
                continue
            if statmod.S_ISREG(st.st_mode):
                apparent += int(st.st_size)
                allocated += int(getattr(st, "st_blocks", 0)) * 512
    return n, apparent, allocated


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


def _early_lockdown():
    out = {"agent_processes_killed": 0.0, "agent_processes_left": 0.0, "agent_processes_running": 0.0,
           "stage_locked": 0.0, "stage_parent_locked": 0.0,
           "submission_late_writes": 0.0, "late_write_gate_active": 0.0, "submission_snapshot": 0.0, "submission_root_rejected": 0.0,
           "submission_oversize": 0.0, "saved": None, "parent": None, "reasons": [], "t_ref": None}
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
        try:
            FREEZER.start()                   # whatever supervisord respawns from here on stays stopped until the verdict
        except Exception:  # noqa: BLE001 - the freezer never takes the verdict down (agent_processes_frozen then reads 0)
            pass
        if left:
            # The image's supervisord respawns the account's desktop as fast as it is killed, so in the separate
            # verifier a process that outlives the kill is the image's own. It is frozen (a stopped process cannot exit,
            # so it is not respawned); only one still running once frozen voids the run.
            out["agent_processes_running"] = float(len(_still_running(uid)))
            log("%d process(es) of the agent account outlived the kill at verify start, %d still running after the "
                "freeze" % (left, out["agent_processes_running"]))
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
        n, apparent, allocated = _submission_measure(SUBMISSION)
        if n > SUBMISSION_ENTRIES_CAP or max(apparent, allocated) > SUBMISSION_BYTES_CAP:
            out["submission_oversize"] = 1.0
            out["reasons"].append("the submission holds %s entries / %d apparent bytes (caps %d entries, %d bytes): not copied, "
                                  "graded as an empty folder" % (n if n <= SUBMISSION_ENTRIES_CAP else "> %d" % SUBMISSION_ENTRIES_CAP,
                                                                 apparent, SUBMISSION_ENTRIES_CAP, SUBMISSION_BYTES_CAP))
            return out
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


def _finite_metrics(metrics):
    """`metrics` in name order, finite numbers only: ale_verify.write refuses NaN and infinity."""
    out = {}
    for k, val in sorted(metrics.items()):
        try:
            x = float(val)
        except (TypeError, ValueError):
            continue
        if math.isfinite(x):
            out[k] = x
    return out


def write_verdict(GR, rewards, weights, stats, details):
    """One gated `reward` (the kit's envelope) = description_score; the gate, the criteria and the stats go to
    `metrics`, the per-criterion diagnostics into the verdict's `diagnostic` text. In the engine (ALE_VERIFY_PYTHON set)
    the kit writes /logs/verifier/reward.json and verification.json through ale_verify; a host-side run writes the plain
    envelope to ROBOTICS_GRADER_VERDICT_PATH, else verdict.local.json in the stage folder."""
    from robotics_grader import ratio_rewards, read_anchor
    from robotics_grader import write_verdict as kit_write
    crit = GR.CRITERIA
    overall = sum(float(rewards[c]) * weights[c] for c in crit) / sum(weights[c] for c in crit)
    anchor, meta = read_anchor()
    extra = {k: float(val) for k, val in stats.items() if isinstance(val, (int, float)) and not isinstance(val, bool)}
    extra.update({k: float(rewards[k]) for k in list(GR.GATES) + list(crit)})
    verdict = ratio_rewards(overall, anchor, higher_is_better=meta.get("direction", "higher") == "higher",
                            metric_name=meta.get("metric", "description_score"), extra=extra, full_at=meta.get("full_at", 1.0))
    text = "; ".join("%s %.4f: %s" % (k, float(rewards[k]), (" | ".join(str(d) for d in details.get(k, [])))[:220])
                     for k in list(GR.GATES) + list(crit))
    if os.environ.get("ALE_VERIFY_PYTHON"):
        recorded = {"rewards": verdict["rewards"], "metrics": _finite_metrics(verdict["metrics"])}
        kit_write(recorded, diagnostic=text[:2000])
        return overall
    kit_write(verdict, path=os.environ.get("ROBOTICS_GRADER_VERDICT_PATH") or os.path.join(STAGE_DIR, "verdict.local.json"),
              diagnostic=text)
    return overall


def _zero_verdict(reason, stats):
    """All rewards 0 (the lockdown itself failed): still a verdict, never a task error."""
    import rd_info_grade as GR
    rewards = {k: 0.0 for k in list(GR.GATES) + list(GR.CRITERIA)}
    details = {"model_loads": [reason[:1500]]}
    return write_verdict(GR, rewards, dict(GR.DEFAULT_WEIGHTS), dict(stats, lockdown_exception=1.0), details)


def main():
    t0 = time.time()
    try:
        early = _early_lockdown()
    except Exception:  # noqa: BLE001 - never a task error without a verdict
        import traceback
        log("the verify-start lockdown raised:\n" + traceback.format_exc())
        _zero_verdict("the verify-start lockdown failed: " + traceback.format_exc()[-800:],
                      {"lockdown_s": time.time() - t0, "agent_processes_frozen": float(len(FREEZER.frozen))})
        FREEZER.stop()
        return
    stats = {k: v for k, v in early.items() if isinstance(v, float)}
    stats["lockdown_s"] = time.time() - t0
    try:
        if early["agent_processes_running"] > 0:
            raise RuntimeError("%d agent process(es) survived the verify-start kill and could not be stopped: run void"
                               % early["agent_processes_running"])
        import numpy as np
        import rd_info_grade as GR
        import rd_info_gate as GT
        with open(os.path.join(STAGE_DIR, "grader_config.json")) as fh:
            cfg = json.load(fh)
        weights = dict(GR.DEFAULT_WEIGHTS)
        weights.update(cfg.get("weights", {}))
        sub_dir = SNAPSHOT if early["submission_snapshot"] else SUBMISSION
        if early["submission_root_rejected"] > 0 or early["submission_oversize"] > 0:   # never read: graded as an empty folder
            import tempfile
            sub_dir = tempfile.mkdtemp(prefix="arb_rejected_submission_")
        pin = os.environ.get("ARB_SEEDS_PIN")
        if pin:
            hi, lo = [int(x) for x in pin.split(":")]
        else:
            hi, lo = int.from_bytes(os.urandom(4), "big"), int.from_bytes(os.urandom(4), "big")
        stats["hidden_seed_hi"], stats["hidden_seed_lo"] = float(hi), float(lo)
        rng = np.random.default_rng(np.random.SeedSequence([hi, lo]))
        t1 = time.time()
        gcfg = cfg.get("grader", {})
        ref = GR.load_reference(os.path.join(STAGE_DIR, "gt"), gcfg)
        cases = GR.make_cases(ref, rng, gcfg)
        stats["draw_s"] = time.time() - t1
        stats["grader_exception"] = 0.0
        t2 = time.time()
        try:
            gate = GT.gate_model_loads(sub_dir)
            rewards, gstats, details = GR.grade(sub_dir, ref, cases, gcfg, log=log, gate=gate)
        except Exception:  # noqa: BLE001 - a submission that crashes the grader earns 0, not a task error
            import traceback
            log("exception while grading the submission:\n" + traceback.format_exc())
            rewards = {k: 0.0 for k in list(GR.GATES) + list(GR.CRITERIA)}
            gstats, details = {}, {"model_loads": ["grader exception on this submission: " + traceback.format_exc()[-400:]]}
            stats["grader_exception"] = 1.0
        stats["grade_s"] = time.time() - t2
        for k, v in gstats.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                stats[k] = float(v)
        final = 0.0 if (early["submission_late_writes"] > 0 or early["submission_root_rejected"] > 0
                        or early["submission_oversize"] > 0) else 1.0
        stats["submission_final"] = final
        if final < 1.0:                       # the deliverable must be final when the session ends: model_loads = 0
            rewards["model_loads"] = 0.0
            details["model_loads"] = ["submission_final: " + "; ".join(early["reasons"])] + list(details.get("model_loads", []))
        gate_v = 1.0
        for g in GR.GATES:
            gate_v *= float(rewards[g])
        for c in GR.CRITERIA:
            rewards[c] = float(rewards[c]) * (1.0 if gate_v > 0 else 0.0)
        stats["verify_s"] = time.time() - t0
        stats["agent_processes_frozen"] = float(len(FREEZER.frozen))
        overall = write_verdict(GR, rewards, weights, stats, details)
        log("gate {} criteria {} overall {:.4f} ({:.0f} s)".format({g: rewards[g] for g in GR.GATES},
                                                                  {c: round(rewards[c], 4) for c in GR.CRITERIA}, overall, stats["verify_s"]))
    finally:
        _restore(early)
        FREEZER.stop()                        # after the verdict; the stopped processes are never resumed


if __name__ == "__main__":
    main()
