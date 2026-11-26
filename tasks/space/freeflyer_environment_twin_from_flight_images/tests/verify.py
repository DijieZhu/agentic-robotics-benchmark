"""Grader of the flight-twin task (an environment twin of one real flight of the cabin free-flyer). Root, verify
stage, in a SEPARATE verifier sandbox: a fresh sandbox from the task image into which the engine restored only
/home/user/submission; the agent's sandbox is gone and setup.sh has not run here.

  0. FIRST THING, stdlib only (`_early_lockdown`; test.sh has already chmod 0700 the stage and its parent): record when
     the engine began uploading this folder (earliest ctime of its files), kill every process of the agent account (a
     survivor that cannot be frozen voids the run: task error) and keep the account SIGSTOPped until the verdict is
     written (arm/verify_freeze.py), lock the stage (0700 / 0600, restored at the end), and take a root-owned 0700
     snapshot of the submission that every later read uses. The integrity gate G1 is 0 when an entry of the
     submission was written at or after the upload began, when the submission root is a symbolic link, not a
     directory, resolves outside the agent's home or was itself replaced after the session, when the folder holds
     more entries or bytes than the configuration's caps (then it is not copied at all), or when it holds a symbolic
     link or a special file (the probe library's submission_size rule, the same one the agent's check tool applies).
  1. The worker (twin/worker.py): a separate process as the unprivileged account of the configuration's runtime where
     it exists, under an address-space limit and a hard wall-clock timeout planned from ALE_VERIFY_DEADLINE_SECONDS, on
     private copies of the grading code, the configuration, the truth, the reference twin, the givens it reads and the
     submission. It grades the reference first (every term exactly 1, or a task error), then the submission; when G1
     is 0 it still grades the reference (on an empty submission) before the zero is written.
  2. twin_score = G1 x the mean over the flights of [G0_f x A_f], A_f the flight's score under the configuration's
     aggregation (both aggregations are computed and reported), from the flight's H and D under the configuration's
     score.hazard_aggregation and score.depth_aggregation (every registered form of each is computed, the verdict
     carries the forms and the counts of rule P, the maintainers' grade record the flight's score under every pair);
     with key_frames.named false (rule L) the scored moments are the truth's own index and cameras.json declares a
     pose for every given frame (twin/grade.py); the verdict goes through the benchmark kit against
     anchor.json (reward = twin_score), with metrics from the whitelist (twin/metrics.py) and finite only, and nothing
     that depends on the clock (stage times and memory go to stderr and, if ARB_TWIN_TIMING_PATH is set, to that
     file), so a rerun writes a byte-identical verdict. With ARB_TWIN_GRADE_PATH set, the whole grade is written there
     as well (maintainers only).
A worker that times out, runs out of memory, fails a self-test or meets an internal error makes the verification
void: a task error, no score. Rule Z7 is the submission's: a flight whose ray scene the worker's child process does not
build inside the scene stage's own limit (runtime.scene_time_limit_s) or whose rays it does not cast inside the ray
stage's own limit (runtime.rays_time_limit_s), or which does not fit inside the worker's address space
(runtime.scene_memory_limit_mb), scores 0 under rule Z7, and the run ends with a verdict. No agent code runs here
(MJCF, meshes and JSON are data). The grader renders nothing.
Knobs: grader_config.json (generated from the public entries of the task's table of numbers; no default in code).
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
PROGRESS_MAX_BYTES = 64 * 1024 * 1024           # a bound on what is read back from the worker, not a rule
# the worker's floating-point code paths: numpy and OpenBLAS pinned to x86-64-v3, the L2 size always beside the kernel
FP_PINNING = {"NPY_DISABLE_CPU_FEATURES": "X86_V4 AVX512_ICL AVX512_SPR", "OPENBLAS_CORETYPE": "Haswell",
              "OPENBLAS_L2_SIZE": "256"}
# the metrics the benchmark kit's ratio_rewards adds itself
KIT_METRICS = ("ratio", "anchor_recorded")
# stats that depend on the clock or the machine: never in the verdict, only in the log
NONDETERMINISTIC = ("lockdown_s", "verify_s", "verify_budget_s", "worker_budget_s", "worker_s", "worker_peak_rss_mb",
                    "selftest_s", "submission_s")

from arm.assembly_six import lockdown  # noqa: E402  (stdlib only; vendored)
from arm.assembly_cell import runner  # noqa: E402  (stdlib only; vendored)
from arm import verify_freeze  # noqa: E402  (stdlib only; vendored)

FREEZER = verify_freeze.AgentFreezer.for_user(AGENT_USER)


def log(msg):
    print("[grader] " + str(msg), file=sys.stderr, flush=True)


def _read_config_doc():
    """The configuration document (stdlib only: the lockdown reads its caps before anything else runs)."""
    with open(os.path.join(STAGE_DIR, "grader_config.json"), "rb") as fh:
        return json.loads(fh.read().decode("utf-8"))


def _caps(doc):
    v = doc["values"]
    rt = doc["runtime"]
    megabyte = int(v["contract.megabyte_bytes"])        # one MB of every size cap, as the table states it
    return int(v["caps.files"]), int(round(float(v["caps.megabytes"]) * megabyte)), float(rt["late_gate_max_gap_s"])


def _submission_root_ok():
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


def submission_measure(root, max_entries):
    """(entries, bytes of regular files, irregular entries) below root by lstat, never following a link; the same
    count as twinprobe.gate.tree_stats (files and folders, root excluded); stops once max_entries is exceeded."""
    n, total, irregular = 0, 0, []
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        for name in list(dirnames):
            n += 1
            p = os.path.join(dirpath, name)
            if os.path.islink(p):
                irregular.append(os.path.relpath(p, root) + " (symbolic link)")
                dirnames.remove(name)
        for name in filenames:
            n += 1
            p = os.path.join(dirpath, name)
            try:
                st = os.lstat(p)
            except OSError:
                irregular.append(os.path.relpath(p, root) + " (unreadable)")
                continue
            if statmod.S_ISLNK(st.st_mode):
                irregular.append(os.path.relpath(p, root) + " (symbolic link)")
            elif not statmod.S_ISREG(st.st_mode):
                irregular.append(os.path.relpath(p, root) + " (not a regular file)")
            else:
                total += int(st.st_size)
        if n > max_entries:
            break
    return n, total, irregular


def _still_running(uid, bound_s=3.0):
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


def _proc_state(pid):
    try:
        with open("/proc/%d/status" % pid) as fh:
            for line in fh:
                if line.startswith("State:"):
                    return line.split()[1]
    except (OSError, IndexError):
        pass
    return "X"


def _early_lockdown():
    out = {"agent_processes_killed": 0.0, "agent_processes_left": 0.0, "agent_processes_running": 0.0,
           "stage_locked": 0.0, "stage_parent_locked": 0.0,
           "submission_late_writes": 0.0, "late_write_gate_active": 0.0, "submission_snapshot": 0.0,
           "submission_root_rejected": 0.0, "submission_root_violation": 0.0, "submission_oversize": 0.0,
           "submission_irregular": 0.0, "saved": None, "parent": None, "reasons": [], "t_ref": None,
           "config_error": None}
    root_ok, why = _submission_root_ok()
    if not root_ok:
        out["submission_root_rejected"] = 1.0
        out["reasons"].append(why)
    is_root = os.getuid() == 0
    if is_root:
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
                out["agent_processes_running"] = float(len(_still_running(uid)))
                log("%d process(es) of the agent account outlived the kill at verify start, %d still running after "
                    "the freeze" % (left, out["agent_processes_running"]))
            try:
                FREEZER.start()
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
    try:
        max_entries, max_bytes, late_gap = _caps(_read_config_doc())
    except Exception as exc:  # noqa: BLE001 - recorded; main() makes it a task error before anything is graded
        out["config_error"] = "grader_config.json cannot be read for the caps: %s" % str(exc)[:300]
        return out
    gate_active = is_root and out["t_ref"] is not None
    cfg_json = os.path.join(STAGE_DIR, "grader_config.json")
    if gate_active and os.path.isfile(cfg_json):
        try:
            gate_active = (os.lstat(cfg_json).st_ctime - out["t_ref"]) <= late_gap
        except OSError:
            pass
    out["late_write_gate_active"] = 1.0 if gate_active else 0.0
    if root_ok and os.path.isdir(SUBMISSION):
        if gate_active:
            try:
                rst = os.lstat(SUBMISSION)
                if max(rst.st_ctime, rst.st_mtime) >= out["t_ref"]:
                    out["submission_root_violation"] = 1.0
                    out["reasons"].append("the submission directory itself was replaced or changed after the agent "
                                          "phase ended")
            except OSError:
                pass
        n, total, irregular = submission_measure(SUBMISSION, max_entries)
        if n > max_entries or total > max_bytes:
            out["submission_oversize"] = 1.0
            out["reasons"].append("the submission holds %s entries and %d bytes (caps %d entries, %d bytes): not "
                                  "copied" % (n if n <= max_entries else "more than %d" % max_entries, total,
                                              max_entries, max_bytes))
            return out
        if irregular:
            out["submission_irregular"] = float(len(irregular))
            out["reasons"].append("the submission holds symbolic links or special files: %s" % ", ".join(irregular[:5]))
        if gate_active:
            late = lockdown.late_entries(SUBMISSION, out["t_ref"])
            out["submission_late_writes"] = float(len(late))
            if late:
                out["reasons"].append("%d entr%s of the submission written after the agent phase ended (%s)" % (
                    len(late), "y" if len(late) == 1 else "ies", ", ".join(e[0] for e in late[:5])))
        if is_root:
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
                   or early["submission_root_violation"] > 0 or early["submission_oversize"] > 0
                   or early["submission_irregular"] > 0) else 1.0


# ---------------------------------------------------------------------------------------------------------------
# the worker

def _budget(rt, t0):
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


def _copy_submission(src, dst):
    """The submission's work copy: runner.copy_regular_files (folders and regular files only; symbolic links, special
    files and __pycache__ skipped), except that a regular file this process cannot read is left out, so that the
    validity gate meets a missing file (rule Z2) and the run ends with a verdict. Returns the relative paths left out
    (named in the maintainers' grade record only)."""
    left_out = []
    os.makedirs(dst, exist_ok=True)
    for dirpath, dirnames, filenames in os.walk(src, followlinks=False):
        rel = os.path.relpath(dirpath, src)
        target = dst if rel == "." else os.path.join(dst, rel)
        keep = []
        for d in dirnames:
            p = os.path.join(dirpath, d)
            if d == "__pycache__" or os.path.islink(p):
                continue
            os.makedirs(os.path.join(target, d), exist_ok=True)
            keep.append(d)
        dirnames[:] = keep
        for f in filenames:
            p = os.path.join(dirpath, f)
            if os.path.islink(p) or not statmod.S_ISREG(os.lstat(p).st_mode):
                continue
            try:
                with open(p, "rb") as fh:
                    data = fh.read()
            except OSError:
                left_out.append(os.path.normpath(os.path.join(rel, f)))
                continue
            with open(os.path.join(target, f), "wb") as fh:
                fh.write(data)
    return sorted(left_out)


def _prepare_work(doc, sub_dir):
    """A private work folder: code/twin (the grading modules), stage/ (the configuration, the truth with the reference,
    the given files the grader reads), submission/ (regular files only, readable ones only: _copy_submission), out/,
    home/; returns (work, job, the submission's files left out of the work copy)."""
    lay = doc["layout"]
    work = tempfile.mkdtemp(prefix="arb_flight_twin_verify_")
    code = os.path.join(work, "code")
    stage = os.path.join(work, "stage")
    runner.copy_regular_files(os.path.join(STAGE_DIR, "twin"), os.path.join(code, "twin"))
    _copy_file(os.path.join(STAGE_DIR, "grader_config.json"), os.path.join(stage, "grader_config.json"))
    src = os.path.join(STAGE_DIR, lay["truth_dir"])
    if os.path.isdir(src):
        runner.copy_regular_files(src, os.path.join(stage, lay["truth_dir"]))
    rels = [lay["ray_table"]] + [lay[k].format(flight=f) for f in doc["values"]["flights.names_neutral"]
                                 for k in ("frames_csv", "estimate_csv")]
    for rel in rels:
        p = os.path.join(STAGE_DIR, lay["givens_dir"], rel)
        if os.path.isfile(p):
            _copy_file(p, os.path.join(stage, lay["givens_dir"], rel))
    job = {"format": "flight-twin-worker-job/1", "stage": stage, "code": code,
           "grader_config": os.path.join(stage, "grader_config.json"), "submission": os.path.join(work, "submission")}
    left_out = []
    if sub_dir is not None and os.path.isdir(sub_dir):
        left_out = _copy_submission(sub_dir, job["submission"])
    else:
        os.makedirs(job["submission"])
    for d in ("out", "home"):
        os.makedirs(os.path.join(work, d))
    job["progress"] = os.path.join(work, "out", "progress.json")
    return work, job, left_out


def _read_progress(path):
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


def _stop_worker(proc):
    """SIGKILL the worker's process group by the id recorded when it was started (start_new_session: group id = its
    pid); never an id looked up later, never 0, 1 or -1."""
    pid = proc.pid
    if isinstance(pid, int) and pid > 1000:
        try:
            os.killpg(pid, signal.SIGKILL)
        except OSError:
            pass
    elif isinstance(pid, int) and pid > 1 and pid != os.getpid():
        try:
            proc.kill()
        except OSError:
            pass


def run_worker(doc, sub_dir, budget_s, stats):
    """Run the worker; returns (progress document or None, how it ended, the tail of its log)."""
    rt = doc["runtime"]
    work, job, left_out = _prepare_work(doc, sub_dir)
    stats["work_copy_left_out"] = left_out
    if left_out:
        log("%d regular file(s) of the submission cannot be read and are left out of the work copy: %s"
            % (len(left_out), ", ".join(left_out[:5])))
    try:
        job["deadline_wall"] = time.time() + budget_s - float(rt["worker_margin_s"])
        job["stage_caps_s"] = dict(rt["stage_caps_s"])
        job["address_space_mb"] = int(rt["worker_address_space_mb"])
        job_path = os.path.join(work, "job.json")
        with open(job_path, "w") as f:
            json.dump(job, f)
        acct = runner.runner_account(rt["worker_account"], None)
        env = {"HOME": os.path.join(work, "home"), "PATH": "/usr/bin:/bin", "LANG": "C.UTF-8",
               "PYTHONDONTWRITEBYTECODE": "1", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
               "MKL_NUM_THREADS": "1", "TMPDIR": os.path.join(work, "home")}
        env.update(FP_PINNING)
        boot = ("import sys; sys.path.insert(0, %r); from twin import worker; sys.exit(worker.worker_main(%r))"
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
        log("worker: %s, budget %.0f s, address space %d MB" % ("as " + acct[0] if acct else "as this account",
                                                                 budget_s, job["address_space_mb"]))
        t = time.time()
        with open(os.path.join(work, "out", "worker.log"), "wb") as logf:
            if acct is not None:
                os.lchown(os.path.join(work, "out", "worker.log"), acct[1], acct[2])
            proc = subprocess.Popen(argv, cwd=os.path.join(work, "home"), env=env, stdout=logf,
                                    stderr=subprocess.STDOUT, start_new_session=True)
            how = "exit"
            try:
                proc.wait(timeout=budget_s)
            except subprocess.TimeoutExpired:
                how = "timeout"
                _stop_worker(proc)
                proc.wait()
        stats["worker_s"] = time.time() - t
        doc_out = _read_progress(job["progress"])
        try:
            with open(os.path.join(work, "out", "worker.log"), "rb") as f:
                f.seek(0, 2)
                f.seek(max(0, f.tell() - 3000))
                tail = f.read().decode("utf-8", "replace")
        except OSError:
            tail = ""
        if how == "timeout":
            how = "killed at the hard timeout"
        elif proc.returncode != 0:
            how = "exit code %d" % proc.returncode
        else:
            how = "finished"
        return doc_out, how, tail
    finally:
        shutil.rmtree(work, ignore_errors=True)


# ---------------------------------------------------------------------------------------------------------------
# the verdict

def _diagnostic(g1, early, grade, flights):
    parts = ["G1 %.0f%s" % (g1, (": " + "; ".join(early["reasons"])[:600]) if early["reasons"] else "")]
    if grade is not None:
        for f in flights:
            r = grade["flights"][f]
            parts.append("%s: G0 %.0f, score %.4f (sum %.4f, product %.4f); D %.4f, H %.4f%s" % (
                f, r["G0"], r["terms"]["score"], r["terms"]["sum"], r["terms"]["product"], r["terms"]["D"],
                r["terms"]["H"], (" | " + "; ".join(r["reasons"])[:500]) if r["reasons"] else ""))
    return " || ".join(parts)[:2000]


def write_verdict(overall, metrics, diagnostic, doc):
    from robotics_grader import ratio_rewards, read_anchor
    from robotics_grader import write_verdict as kit_write
    from robotics_grader import fail
    anchor, meta = read_anchor()
    v = doc["values"]
    if anchor != v["score.anchor"] or meta.get("full_at") != v["score.full_at"]:
        fail("anchor.json (value %r, full_at %r) is not the configuration's score.anchor / score.full_at"
             % (anchor, meta.get("full_at")))
    verdict = ratio_rewards(overall, anchor, higher_is_better=meta.get("direction", "higher") == "higher",
                            metric_name=meta.get("metric", "twin_score"), extra=dict(metrics),
                            full_at=meta.get("full_at"))
    from twin.metrics import whitelist
    allowed = whitelist(v["flights.names_neutral"]) | {"twin_score", meta.get("metric", "twin_score")}
    kept = {}
    for k, x in sorted(verdict["metrics"].items()):
        try:
            f = float(x)
        except (TypeError, ValueError):
            continue
        if math.isfinite(f) and (k in allowed or k in KIT_METRICS):
            kept[k] = f
    if os.environ.get("ALE_VERIFY_PYTHON"):
        kit_write({"rewards": verdict["rewards"], "metrics": kept}, diagnostic=diagnostic)
        return verdict
    verdict["metrics"] = kept
    kit_write(verdict, path=os.environ.get("ROBOTICS_GRADER_VERDICT_PATH") or os.path.join(STAGE_DIR,
                                                                                         "verdict.local.json"),
              diagnostic=diagnostic)
    return verdict


def _timing(stats, extra=None):
    t = {k: v for k, v in stats.items() if k in NONDETERMINISTIC}
    t.update(extra or {})
    log("timing: " + json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in sorted(t.items())}))
    path = os.environ.get("ARB_TWIN_TIMING_PATH")
    if path:
        try:
            with open(path, "w") as fh:
                json.dump(t, fh, indent=1, sort_keys=True)
        except OSError:
            pass


def main():
    t0 = time.time()
    from robotics_grader import fail
    try:
        early = _early_lockdown()
    except Exception:  # noqa: BLE001
        import traceback
        log("the verify-start lockdown raised:\n" + traceback.format_exc())
        FREEZER.stop()
        fail("the verify-start lockdown failed: " + traceback.format_exc()[-800:])
        return
    stats = {k: v for k, v in early.items() if isinstance(v, float)}
    stats["lockdown_s"] = time.time() - t0
    try:
        if early["agent_processes_running"] > 0:
            fail("%d agent process(es) survived the verify-start kill and could not be stopped: run void"
                 % early["agent_processes_running"])
        if early["config_error"]:
            fail(early["config_error"])
        doc = _read_config_doc()
        sys.path.insert(0, os.path.join(STAGE_DIR))
        from twin import config as config_mod
        from twin.metrics import metrics_of
        try:
            config_mod.from_document(doc)
        except config_mod.ConfigError as exc:
            fail("the configuration is not usable: %s" % exc)
        flights = list(doc["values"]["flights.names_neutral"])
        g1 = _g1(early)
        stats["G1"] = g1
        stats["agent_processes_frozen"] = float(len(FREEZER.frozen))
        sub_dir = SNAPSHOT if early["submission_snapshot"] else (SUBMISSION if os.path.isdir(SUBMISSION) else None)
        if g1 < 1.0:
            sub_dir = None                # nothing of the submission is read; the worker still grades the reference
        budget, src, total = _budget(doc["runtime"], t0)
        stats["verify_budget_s"] = total
        stats["worker_budget_s"] = budget
        log("time plan: %.0f s from %s; worker %.0f s" % (total, src, budget))
        out, how, tail = run_worker(doc, sub_dir, budget, stats)
        if out is None or not isinstance(out.get("selftest"), dict):
            fail("the grader's worker produced no self-test result (%s): %s" % (how, tail[-1200:]))
        st = out["selftest"]
        if not st.get("ok"):
            fail("the reference or the truth failed the self-test (%s): %s" % (how, str(st.get("reason"))[:1500]))
        if g1 < 1.0:
            m = metrics_of(None, stats, flights)
            m["twin_score"] = 0.0
            write_verdict(0.0, m, _diagnostic(g1, early, None, flights), doc)
            log("G1 = 0: %s" % "; ".join(early["reasons"]))
            stats["verify_s"] = time.time() - t0
            _timing(stats)
            return
        if out.get("task_error"):
            fail("the verification is void (%s): %s" % (how, str(out["task_error"].get("reason"))[:1500]))
        if how != "finished" or not out.get("done") or not isinstance(out.get("submission"), dict):
            fail("the worker was %s before the submission was graded: the verification is void" % how)
        grade = out["submission"]
        timing = dict(out.get("timing") or {})
        if isinstance(out.get("peak_rss_mb"), (int, float)):
            timing["worker_peak_rss_mb"] = float(out["peak_rss_mb"])
        grade_path = os.environ.get("ARB_TWIN_GRADE_PATH")
        if grade_path:
            rec = {"submission": grade, "selftest": st}
            if stats.get("work_copy_left_out"):
                rec["work_copy_left_out"] = list(stats["work_copy_left_out"])
            try:
                with open(grade_path, "w") as fh:
                    json.dump(rec, fh, indent=1, sort_keys=True)
            except (OSError, TypeError, ValueError):
                pass
        overall = g1 * float(grade["twin_score"])
        m = metrics_of(grade, stats, flights)
        m["twin_score"] = overall
        write_verdict(overall, m, _diagnostic(g1, early, grade, flights), doc)
        stats["verify_s"] = time.time() - t0
        _timing(stats, timing)
        log("G1 %.0f %s twin_score %.4f (%s)" % (g1, " ".join("%s %.4f" % (f, grade["flights"][f]["terms"]["score"])
                                                            for f in flights), overall, how))
    finally:
        _restore(early)
        FREEZER.stop()


if __name__ == "__main__":
    main()
