"""Grader of robotic_arm_environment_twin_from_real_photo. Root, verify stage, in a SEPARATE verifier sandbox
(task.toml): a fresh sandbox from the task image into which the engine restored only /home/user/submission; the
agent's sandbox is gone and setup.sh has not run here.

  0. FIRST THING, stdlib only (`_early_lockdown`; test.sh has already chmod 0700 the stage and its parent): record when
     the engine began uploading this folder (earliest ctime of its files), kill every process of the agent account (a
     survivor that cannot be frozen voids the run: task error) and keep the account SIGSTOPped until the verdict is
     written (arm/verify_freeze.py: the image's supervisord respawns its desktop as that account), lock the stage (0700 / 0600,
     restored at the end), and take a root-owned 0700 snapshot of the submission that every later read uses. Gate G1
     (submission integrity) is 0 when an entry of the submission was written at or after the upload began, when the
     submission root is a symbolic link, not a directory, resolves outside the agent's home or was itself replaced
     after the session, or when the folder holds more than 5000 entries or 300 MB (then it is not copied at all).
  1. The worker: a separate process (twin/part2.worker_main) as the unprivileged account `nobody` where it exists,
     under an address-space limit and a hard wall-clock timeout planned from ALE_VERIFY_DEADLINE_SECONDS, on private
     copies of the grading code, the truth and the submission. It builds the truth (gt/), self-tests it (the true twin
     through the same pipeline: the gate with no exclusions, G0's settle check, every criterion exactly 1, its settle
     within 1 mm, its support edges and every unambiguous probe outcome equal to the truth files, C3's ray cast
     agreeing with the depth reference), then grades the submission: the gate and G0's settle check (G0), part 1
     (alignment R_pp + s*, C1 inventory, C2 pose and shape, C3 environment, C6 scale), the standard settle (C4 static
     physics), the probes (C5 interaction). Each stage runs under its own alarm; a timeout, a memory error or an
     exception in a stage scores that stage's criteria 0 with a diagnostic, and so does a worker killed at the hard
     timeout. A truth that fails its self-test (or a worker that dies before finishing it) is a task error.
  2. twin_score = G1 x G0 x (0.15 C1 + 0.25 C2 + 0.10 C3 + 0.20 C4 + 0.25 C5 + 0.05 C6) (weights:
     grader_config.json); the verdict goes through the benchmark kit (robotics_grader.ratio_rewards against
     anchor.json: value 1.0 = the oracle, full_at 1.0, so reward = twin_score), finite metrics only.
No agent code runs here (MJCF, meshes, images and JSON are data). The grader renders nothing (CPU only).
Knobs: grader_config.json; the public rules: setup_payload/twin_gate.py and twin_sim_config.json (the agent's copies).
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
SUBMISSION_BYTES_CAP = 300 * 1024 * 1024        # = twin_gate.DEFAULT_RULES["max_bytes"]
SUBMISSION_ENTRIES_CAP = 5000                   # = twin_gate.DEFAULT_RULES["max_entries"]
CRITERIA = ("inventory", "pose_shape", "environment", "static_physics", "interaction", "scale")
LABEL = {"inventory": "C1", "pose_shape": "C2", "environment": "C3", "static_physics": "C4", "interaction": "C5",
         "scale": "C6"}
PAYLOAD_FILES = ("twin_gate.py", "twin_sim.py", "twin_camera.py", "twin_sim_config.json", "product_names.json")
PROGRESS_MAX_BYTES = 64 * 1024 * 1024

from arm.assembly_six import lockdown  # noqa: E402  (stdlib only; vendored)
from arm.assembly_cell import runner  # noqa: E402  (stdlib only; vendored)
from arm import verify_freeze  # noqa: E402  (stdlib only; vendored)

# started right after the verify-start kill, stopped after the verdict (a no-op unless root and the account exists)
FREEZER = verify_freeze.AgentFreezer.for_user(AGENT_USER)


def log(msg):
    print("[grader] " + str(msg), file=sys.stderr, flush=True)


def _submission_root_ok():
    """(ok, reason): the submission root must be a real directory inside the agent's home, never a symbolic link."""
    try:
        st = os.lstat(SUBMISSION)
    except FileNotFoundError:
        return True, None
    except OSError as exc:
        return False, "the submission root %s cannot be inspected (%s)" % (SUBMISSION, exc)
    if statmod.S_ISLNK(st.st_mode):
        return False, "the submission root %s is a symbolic link: not read" % SUBMISSION
    if not statmod.S_ISDIR(st.st_mode):
        return False, "the submission root %s is not a directory" % SUBMISSION
    home = os.path.realpath(HOME)
    real = os.path.realpath(SUBMISSION)
    if os.path.commonpath([home, real]) != home:
        return False, "the submission root %s resolves outside %s: not read" % (SUBMISSION, HOME)
    return True, None


def _submission_measure(root, max_entries=SUBMISSION_ENTRIES_CAP):
    """(entries, apparent bytes, allocated bytes) below `root` by lstat, never following a link; stops counting once
    max_entries is exceeded."""
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
           "submission_late_writes": 0.0, "late_write_gate_active": 0.0, "submission_snapshot": 0.0,
           "submission_root_rejected": 0.0, "submission_root_violation": 0.0, "submission_oversize": 0.0,
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
            log("%d process(es) of the agent account outlived the kill at verify start, %d still running after the "
                "freeze" % (left, out["agent_processes_running"]))
        try:
            FREEZER.start()               # whatever supervisord respawns from here on stays stopped until the verdict
        except Exception:  # noqa: BLE001 - the freezer never takes the verdict down
            pass
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
    cfg_json = os.path.join(STAGE_DIR, "grader_config.json")
    if gate_active and os.path.isfile(cfg_json):
        try:
            gate_active = (os.lstat(cfg_json).st_ctime - t_ref) <= LATE_GATE_MAX_GAP_S
        except OSError:
            pass
    out["late_write_gate_active"] = 1.0 if gate_active else 0.0
    if root_ok and os.path.isdir(SUBMISSION):
        if gate_active:
            try:
                rst = os.lstat(SUBMISSION)
                if max(rst.st_ctime, rst.st_mtime) >= t_ref:
                    out["submission_root_violation"] = 1.0
                    out["reasons"].append("the submission directory itself was replaced or changed after the agent "
                                          "phase ended")
            except OSError:
                pass
        n, apparent, allocated = _submission_measure(SUBMISSION)
        if n > SUBMISSION_ENTRIES_CAP or max(apparent, allocated) > SUBMISSION_BYTES_CAP:
            out["submission_oversize"] = 1.0
            out["reasons"].append("the submission holds %s entries / %d bytes (caps %d entries, %d bytes): not "
                                  "copied" % (n if n <= SUBMISSION_ENTRIES_CAP else "> %d" % SUBMISSION_ENTRIES_CAP,
                                              max(apparent, allocated), SUBMISSION_ENTRIES_CAP, SUBMISSION_BYTES_CAP))
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
            runner.unlock_stage(early["saved"])
        if early.get("parent"):
            os.chmod(early["parent"][0], early["parent"][1])
        mode = os.environ.get("ARB_STAGE_MODE")
        if mode:
            os.chmod(STAGE_DIR, int(mode, 8))
    except OSError:
        pass


def _g1(early):
    return 0.0 if (early["submission_late_writes"] > 0 or early["submission_root_rejected"] > 0
                   or early["submission_root_violation"] > 0 or early["submission_oversize"] > 0) else 1.0


# ---------------------------------------------------------------------------------------------------------------
# the worker

def _budget(cfg, t0):
    """(seconds the worker may run, where the number came from): the verify deadline the engine passes, less the
    time already spent and a reserve for the verdict; the configured default outside the engine."""
    rt = cfg["runtime"]
    raw = os.environ.get("ALE_VERIFY_DEADLINE_SECONDS")
    try:
        total, src = float(raw), "ALE_VERIFY_DEADLINE_SECONDS"
    except (TypeError, ValueError):
        total, src = float(rt["default_budget_s"]), "runtime.default_budget_s"
    left = total - (time.time() - t0) - float(rt["reserve_s"])
    return max(float(rt["min_worker_s"]), min(left, float(rt["max_worker_s"]))), src, total


def _copy_file(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copyfile(src, dst)


def _prepare_work(cfg, sub_dir):
    """A private work folder: code/ (the grading modules and the public payload modules), gt/ (the truth files),
    submission/ (regular files only), out/, home/; returns (work, job dict)."""
    work = tempfile.mkdtemp(prefix="arb_twin_verify_")
    code = os.path.join(work, "code")
    runner.copy_regular_files(os.path.join(STAGE_DIR, "twin"), os.path.join(code, "twin"))
    for nm in PAYLOAD_FILES:
        _copy_file(os.path.join(STAGE_DIR, "setup_payload", nm), os.path.join(code, "setup_payload", nm))
    _copy_file(os.path.join(STAGE_DIR, "grader_config.json"), os.path.join(code, "grader_config.json"))
    gt = cfg["gt"]
    job = {"format": "twin-worker-job/1"}
    for key in ("depth_ref", "depth_class", "support_truth", "probe_truth"):
        dst = os.path.join(work, "gt", os.path.basename(gt[key]))
        _copy_file(os.path.join(STAGE_DIR, gt[key]), dst)
        job[key] = dst
    job["gt_twin"] = os.path.join(work, "gt", "twin")
    runner.copy_regular_files(os.path.join(STAGE_DIR, gt["twin"]), job["gt_twin"])
    job["submission"] = os.path.join(work, "submission")
    if sub_dir is not None and os.path.isdir(sub_dir):
        runner.copy_regular_files(sub_dir, job["submission"])
    else:
        os.makedirs(job["submission"])
    for d in ("out", "home"):
        os.makedirs(os.path.join(work, d))
    job.update({"progress": os.path.join(work, "out", "progress.json"), "code": code,
                "vocabulary": os.path.join(code, "setup_payload", "product_names.json"),
                "grader_config": os.path.join(code, "grader_config.json")})
    return work, job


def _read_progress(path):
    """The worker's progress document, read without following links and within a size cap (None if absent or bad)."""
    try:
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except OSError:
        return None
    try:
        if not statmod.S_ISREG(os.fstat(fd).st_mode):
            return None
        data = os.read(fd, PROGRESS_MAX_BYTES + 1)
    finally:
        os.close(fd)
    if len(data) > PROGRESS_MAX_BYTES:
        return None
    try:
        doc = json.loads(data.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        return None
    return doc if isinstance(doc, dict) else None


def run_worker(cfg, sub_dir, budget_s, stats):
    """Run the worker; returns (progress document or None, how it ended, the tail of its log)."""
    import signal as _signal
    rt = cfg["runtime"]
    work, job = _prepare_work(cfg, sub_dir)
    try:
        job["deadline_wall"] = time.time() + budget_s - float(rt["worker_margin_s"])
        job["stage_caps_s"] = dict(rt["stage_caps_s"])
        job["address_space_mb"] = int(rt["worker_address_space_mb"])
        job_path = os.path.join(work, "job.json")
        with open(job_path, "w") as f:
            json.dump(job, f)
        acct = runner.runner_account(rt.get("worker_account", "nobody"), None)
        # no MUJOCO_GL: the worker renders nothing, so no GL library is loaded
        env = {"HOME": os.path.join(work, "home"), "PATH": "/usr/bin:/bin", "LANG": "C.UTF-8",
               "PYTHONDONTWRITEBYTECODE": "1", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
               "MKL_NUM_THREADS": "1", "TMPDIR": os.path.join(work, "home")}
        boot = ("import sys; sys.path.insert(0, %r); from twin import part2; sys.exit(part2.worker_main(%r))"
                % (job["code"], job_path))
        argv = [sys.executable, "-I", "-B", "-c", boot]
        stats["worker_deprivileged"] = 0.0
        if acct is not None:
            for dirpath, dirnames, filenames in os.walk(work):
                for n in [dirpath] + [os.path.join(dirpath, x) for x in dirnames + filenames]:
                    os.lchown(n, acct[1], acct[2])
            os.chmod(work, 0o700)
            from robotics_grader.stage import secure_exec_path
            secure_exec_path()
            ru = shutil.which("runuser", path="/usr/sbin:/usr/bin:/sbin:/bin") or "/usr/sbin/runuser"
            argv = [ru, "-p", "-u", acct[0], "--"] + argv
            stats["worker_deprivileged"] = 1.0
            stats["worker_uid"] = float(acct[1])
        t = time.time()
        log("worker: %s, budget %.0f s, address space %d MB" % ("as " + acct[0] if acct else "as this account",
                                                                 budget_s, job["address_space_mb"]))
        with open(os.path.join(work, "out", "worker.log"), "wb") as logf:
            if acct is not None:
                os.lchown(os.path.join(work, "out", "worker.log"), acct[1], acct[2])
            proc = subprocess.Popen(argv, cwd=os.path.join(work, "home"), env=env, stdout=logf, stderr=subprocess.STDOUT,
                                    start_new_session=True)
            how = "exit"
            try:
                proc.wait(timeout=budget_s)
            except subprocess.TimeoutExpired:
                how = "timeout"
                try:
                    os.killpg(proc.pid, _signal.SIGKILL)
                except OSError:
                    pass
                proc.wait()
        stats["worker_s"] = time.time() - t
        stats["worker_exit"] = float(proc.returncode if proc.returncode is not None else -999)
        stats["worker_timed_out"] = 1.0 if how == "timeout" else 0.0
        doc = _read_progress(job["progress"])
        try:
            with open(os.path.join(work, "out", "worker.log"), "rb") as f:
                f.seek(0, 2)
                f.seek(max(0, f.tell() - 3000))
                tail = f.read().decode("utf-8", "replace")
        except OSError:
            tail = ""
        if how == "timeout":
            how = "killed at the hard timeout (%.0f s)" % budget_s
        elif proc.returncode != 0:
            how = "exit code %d" % proc.returncode
        else:
            how = "finished"
        return doc, how, tail
    finally:
        shutil.rmtree(work, ignore_errors=True)


# ---------------------------------------------------------------------------------------------------------------
# the verdict

def _finite_metrics(metrics):
    """`metrics` in name order, finite numbers only: ale_verify.write refuses NaN and infinity."""
    out = {}
    for k, v in sorted(metrics.items()):
        try:
            x = float(v)
        except (TypeError, ValueError):
            continue
        if math.isfinite(x):
            out[k] = x
    return out


def _unit(x):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return 0.0
    return min(1.0, max(0.0, x)) if math.isfinite(x) else 0.0


def write_verdict(overall, crit, stats, diagnostics):
    """One gated `reward` (the kit's envelope) = twin_score; the gates, the criteria and the stats go to `metrics`, the
    diagnostics into the verdict's `diagnostic` text. In the engine (ALE_VERIFY_PYTHON set) the kit writes
    /logs/verifier/reward.json and verification.json through ale_verify; a host-side run writes the plain envelope to
    ROBOTICS_GRADER_VERDICT_PATH, else verdict.local.json in the stage folder."""
    from robotics_grader import ratio_rewards, read_anchor
    from robotics_grader import write_verdict as kit_write
    anchor, meta = read_anchor()
    extra = {k: float(v) for k, v in stats.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}
    extra.update({c: float(crit[c]) for c in CRITERIA})
    extra["twin_score"] = float(overall)
    verdict = ratio_rewards(overall, anchor, higher_is_better=meta.get("direction", "higher") == "higher",
                            metric_name=meta.get("metric", "twin_score"), extra=extra, full_at=meta.get("full_at", 1.0))
    text = "; ".join("%s %s %.4f: %s" % (k, LABEL.get(k, ""), float(crit.get(k, stats.get(k, 0.0))),
                                         str(diagnostics.get(k, ""))[:240]) for k in ("G1", "G0") + CRITERIA)
    if os.environ.get("ALE_VERIFY_PYTHON"):
        recorded = {"rewards": verdict["rewards"], "metrics": _finite_metrics(verdict["metrics"])}
        kit_write(recorded, diagnostic=text[:2000])
        return verdict
    kit_write(verdict, path=os.environ.get("ROBOTICS_GRADER_VERDICT_PATH") or os.path.join(STAGE_DIR, "verdict.local.json"),
              diagnostic=text)
    return verdict


def _zero_verdict(reason, stats):
    crit = {c: 0.0 for c in CRITERIA}
    stats = dict(stats, G1=0.0, G0=0.0)
    return write_verdict(0.0, crit, stats, {"G1": reason[:1500]})


def main():
    t0 = time.time()
    try:
        early = _early_lockdown()
    except Exception:  # noqa: BLE001 - never a task error without a verdict
        import traceback
        log("the verify-start lockdown raised:\n" + traceback.format_exc())
        _zero_verdict("the verify-start lockdown failed: " + traceback.format_exc()[-800:],
                      {"lockdown_s": time.time() - t0, "agent_processes_frozen": float(len(FREEZER.frozen)),
                       "lockdown_exception": 1.0})
        FREEZER.stop()
        return
    stats = {k: v for k, v in early.items() if isinstance(v, float)}
    stats["lockdown_s"] = time.time() - t0
    try:
        from robotics_grader import fail
        if early["agent_processes_running"] > 0:
            fail("%d agent process(es) survived the verify-start kill and could not be stopped: run void"
                 % early["agent_processes_running"])
        with open(os.path.join(STAGE_DIR, "grader_config.json")) as fh:
            cfg = json.load(fh)
        weights = cfg["weights"]
        crit = {c: 0.0 for c in CRITERIA}
        diag = {}
        g1 = _g1(early)
        stats["G1"] = g1
        if g1 < 1.0:
            diag["G1"] = "; ".join(early["reasons"])
            stats["G0"] = 0.0
            stats["verify_s"] = time.time() - t0
            stats["agent_processes_frozen"] = float(len(FREEZER.frozen))
            write_verdict(0.0, crit, stats, diag)
            log("G1 = 0: %s" % diag["G1"])
            return
        sub_dir = SNAPSHOT if early["submission_snapshot"] else (SUBMISSION if os.path.isdir(SUBMISSION) else None)
        budget, src, total = _budget(cfg, t0)
        stats["verify_budget_s"] = total
        stats["worker_budget_s"] = budget
        log("time plan: %.0f s from %s; worker %.0f s" % (total, src, budget))
        doc, how, tail = run_worker(cfg, sub_dir, budget, stats)
        if doc is None or not isinstance(doc.get("selftest"), dict):
            fail("the grader's worker produced no self-test result (%s): %s" % (how, tail[-1200:]))
        st = doc["selftest"]
        if not st.get("ok"):
            fail("the truth's self-test failed (%s): %s %s" % (how, st.get("reason"), str(st.get("trace", ""))[-800:]))
        if doc.get("task_error"):
            fail("task error while grading: %s" % doc["task_error"].get("reason"))
        for k in ("truth_build_and_selftest_s", "settle_worst_mm", "probe_entries_checked", "c3_gt_agree_frac"):
            if isinstance(st.get(k), (int, float)):
                stats["selftest_" + k] = float(st[k])
        if isinstance(doc.get("peak_rss_mb"), (int, float)):
            stats["worker_peak_rss_mb"] = float(doc["peak_rss_mb"])
        sub = doc.get("submission") if isinstance(doc.get("submission"), dict) else {}
        g0 = 1.0 if _unit(sub.get("G0")) == 1.0 else 0.0
        stats["G0"] = g0
        sub_crit = sub.get("criteria") if isinstance(sub.get("criteria"), dict) else {}
        sub_diag = sub.get("diagnostics") if isinstance(sub.get("diagnostics"), dict) else {}
        for c in CRITERIA:
            if c in sub_crit:
                crit[c] = _unit(sub_crit[c])
                if c in sub_diag:
                    diag[c] = str(sub_diag[c])
            else:
                diag[c] = "not scored: the worker %s before this criterion" % ("stopped" if how == "finished" else how)
        g0_info = sub.get("g0") if isinstance(sub.get("g0"), dict) else {}
        if g0 < 1.0:
            reasons = list(g0_info.get("reasons") or []) + list((g0_info.get("settle") or {}).get("reasons") or [])
            diag["G0"] = "; ".join(str(r) for r in reasons[:4]) or "the gate did not complete (%s)" % how
        for name, row in (sub.get("stages") or {}).items():
            if isinstance(row, dict):
                if isinstance(row.get("s"), (int, float)):
                    stats["stage_%s_s" % name] = float(row["s"])
                stats["stage_%s_ok" % name] = 1.0 if row.get("ok") else 0.0
        for k, v in (sub.get("stats") or {}).items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                stats[k] = float(v)
        tot = sum(float(weights[c]) for c in CRITERIA)
        overall = g1 * g0 * sum(float(weights[c]) * crit[c] for c in CRITERIA) / tot
        stats["verify_s"] = time.time() - t0
        stats["agent_processes_frozen"] = float(len(FREEZER.frozen))
        write_verdict(overall, crit, stats, diag)
        log("G1 %.0f G0 %.0f %s twin_score %.4f (%.0f s, worker %s)" % (
            g1, g0, " ".join("%s %.3f" % (LABEL[c], crit[c]) for c in CRITERIA), overall, stats["verify_s"], how))
    finally:
        _restore(early)
        FREEZER.stop()                    # after the verdict; the stopped processes are never resumed


if __name__ == "__main__":
    main()
