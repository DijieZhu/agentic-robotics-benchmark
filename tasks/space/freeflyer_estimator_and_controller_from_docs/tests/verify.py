"""Grader of the estimator-and-controller task (the flight software of the cabin free-flyer: a state estimator and a
controller written from documents, records and photographs). Root, verify stage, in a SEPARATE verifier sandbox: a
fresh sandbox from the task image into which the engine restored only /home/user/submission; the agent's sandbox is
gone and setup.sh has not run here.

  0. FIRST THING, stdlib only (`_early_lockdown`; test.sh has already closed the stage and its parent): record when the
     engine began uploading this folder (earliest ctime of its files), kill every process of the agent account (a
     survivor that cannot be frozen voids the run) and keep the account SIGSTOPped until the verdict is written
     (arm/verify_freeze.py), lock the stage, and take a root-owned 0700 snapshot of the submission that every later
     read uses. The snapshot is taken only when the folder is inside interface.submission_cap (regular files; symbolic
     links, special files and __pycache__ are not copied): an over-cap submission scores 0 in every session and nothing
     is flown. A submission written after the agent phase ended, or replaced, is not read either (the house's late-write
     gate): its reward is 0.
  1. The static import audit (grader/audit.py) on the snapshot's controller folder: every import names the standard
     library, a library of interface.libraries or a module of the submission.
  2. The flights (grader/fly.py): each hidden session of the hidden set is flown once by the hidden world, the
     submitted controller behind the runner (tests/runner: the wire, its sandbox and its rules); the grader hands the
     node the world's records of the blocks' ends before the world runs. The task's solution holds an access token
     (grader/access.py): a controller folder whose token's sha256 equals the stage's copy receives the session's
     registered reads with its set-up's session paper; any other folder is flown exactly as without it, and a wrong
     or malformed token is no error.
  3. The score (grader/grade.py, measures.py, score.py): each session from the world's true state and records and the
     runner's record; reward = the gates x the mean of the sessions (score.reward). The verdict goes through the
     engine's own writer (grader/verdict.py): the reward is the formula's reward (in [0, 1] by construction; the
     reference scores 1.0 by the scored set), the metrics are the summary of score.grader_output (the gates, per
     session its score and per kind its term; finite only), nothing per job, nothing that depends on the clock (times
     go to stderr only).
The grader's work folder (the runner's run folders with their read-only copies of the submission, and so of a token
it holds) is removed at the end of the run whatever happened in it, also when the grader does not run as root: what the
runner made read-only is unlocked first (never through a link).
Every object built from a session's records is built inside the guard (grader/guard.py); a guard that trips is the
grader's own cause and the verification is void: a task error, no score. The system-call filter's failure to install,
a missing or malformed hidden set, configuration or record, and the verify deadline are task errors as well.
Knobs: grader_config.json (generated from the value of the public entries of the task's table of numbers and the
grader's frame; no knob has a default in code).
"""
import json
import os
import pwd
import shutil
import signal
import stat as statmod
import sys
import tempfile
import time

sys.dont_write_bytecode = True
STAGE_DIR = os.environ.get("ALE_STAGE_DIR", os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, STAGE_DIR)
HOME = os.environ.get("ALE_HOME", "/home/user")
AGENT_USER = os.path.basename(HOME.rstrip("/")) or "user"
SUBMISSION = os.path.join(HOME, "submission")
# the numerics' code paths of the world (house numerics; the runner hands the same pins to the controller)
FP_PINNING = {"NPY_DISABLE_CPU_FEATURES": "AVX512F,AVX512CD,AVX512_KNL,AVX512_KNM,AVX512_SKX,AVX512_CLX,AVX512_CNL,"
                                          "AVX512_ICL",
              "OPENBLAS_CORETYPE": "Haswell", "OPENBLAS_L2_SIZE": "256"}
for _k, _v in FP_PINNING.items():
    os.environ.setdefault(_k, _v)
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

from arm.assembly_six import lockdown  # noqa: E402  (stdlib only; vendored)
from arm.assembly_cell import runner as stage_runner  # noqa: E402  (stdlib only; vendored)
from arm import verify_freeze  # noqa: E402  (stdlib only; vendored)

FREEZER = verify_freeze.AgentFreezer.for_user(AGENT_USER)


def log(msg):
    print("[grader] " + str(msg), file=sys.stderr, flush=True)


def _read_config_doc():
    with open(os.path.join(STAGE_DIR, "grader_config.json"), "rb") as fh:
        return json.loads(fh.read().decode("utf-8"))


def _caps(doc):
    """(entries cap, bytes cap, late-write gap) from interface.submission_cap and the frame's megabyte and gap."""
    cap = doc["values"]["interface.submission_cap"]
    rt = doc["runtime"]
    return (int(cap["files"]), int(round(float(cap["megabytes"]) * int(rt["megabyte_bytes"]))),
            float(rt["late_gate_max_gap_s"]), os.path.join(rt["snapshot_root"], "submission"), rt["snapshot_root"])


def _submission_root_ok():
    try:
        st = os.lstat(SUBMISSION)
    except FileNotFoundError:
        return True, None
    except OSError as exc:
        return False, "the submission root cannot be inspected (%s)" % exc
    if statmod.S_ISLNK(st.st_mode):
        return False, "the submission root is a symbolic link: not read"
    if not statmod.S_ISDIR(st.st_mode):
        return False, "the submission root is not a directory"
    home = os.path.realpath(HOME)
    real = os.path.realpath(SUBMISSION)
    if os.path.commonpath([home, real]) != home:
        return False, "the submission root resolves outside the agent's home: not read"
    return True, None


def submission_measure(root, max_files):
    """(regular files, their bytes) below root by lstat, never following a link; symbolic links, special files and
    __pycache__ are not counted (they are not copied: interface.submission_cap's unit); stops once max_files is passed."""
    n, total = 0, 0
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        dirnames[:] = [d for d in dirnames if d != "__pycache__" and not os.path.islink(os.path.join(dirpath, d))]
        for name in filenames:
            p = os.path.join(dirpath, name)
            try:
                st = os.lstat(p)
            except OSError:
                continue
            if statmod.S_ISREG(st.st_mode):
                n += 1
                total += int(st.st_size)
        if n > max_files:
            break
    return n, total


def _proc_state(pid):
    try:
        with open("/proc/%d/status" % pid) as fh:
            for line in fh:
                if line.startswith("State:"):
                    return line.split()[1]
    except (OSError, IndexError):
        pass
    return "X"


def _still_running(uid, bound_s=3.0):
    deadline = time.time() + float(bound_s)
    while True:
        for pid in lockdown.live_uid_processes(uid):
            # inside the verifier's sandbox the agent's ids are small: every int id above 1 is stopped (never 0 or 1)
            if isinstance(pid, int) and not isinstance(pid, bool) and pid > 1:
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
           "stage_locked": 0.0, "submission_late_writes": 0.0, "late_write_gate_active": 0.0,
           "submission_snapshot": 0.0, "submission_root_rejected": 0.0, "submission_root_violation": 0.0,
           "submission_oversize": 0.0, "saved": None, "parent": None, "reasons": [], "t_ref": None,
           "config_error": None, "snapshot": None}
    root_ok, why = _submission_root_ok()
    if not root_ok:
        out["submission_root_rejected"] = 1.0
        out["reasons"].append(why)
    is_root = os.getuid() == 0
    if is_root:
        out["t_ref"] = lockdown.stage_upload_time(STAGE_DIR)
        try:
            uid = pwd.getpwnam(AGENT_USER).pw_uid
        except KeyError:
            uid = None
        if uid is not None and uid != 0:
            killed, left = lockdown.stop_and_kill_uid(uid, grace_s=8.0)
            out["agent_processes_killed"], out["agent_processes_left"] = float(killed), float(left)
            if left:
                out["agent_processes_running"] = float(len(_still_running(uid)))
            try:
                FREEZER.start()
            except Exception:  # noqa: BLE001 - the freezer never takes the verdict down
                pass
        out["saved"] = stage_runner.lock_stage(STAGE_DIR)
        out["stage_locked"] = 1.0 if out["saved"] is not None else 0.0
        parent = os.path.dirname(os.path.abspath(STAGE_DIR))
        if parent and parent not in ("/", os.path.abspath(HOME)) and os.path.isdir(parent):
            try:
                mode = statmod.S_IMODE(os.lstat(parent).st_mode)
                os.chmod(parent, 0o700)
                saved_mode = os.environ.get("ARB_STAGE_PARENT_MODE")
                out["parent"] = (parent, int(saved_mode, 8) if saved_mode else mode)
            except (OSError, ValueError):
                pass
    try:
        max_files, max_bytes, late_gap, snapshot, snap_root = _caps(_read_config_doc())
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
        n, total = submission_measure(SUBMISSION, max_files)
        if n > max_files or total > max_bytes:
            out["submission_oversize"] = 1.0
            out["reasons"].append("the submission holds %s regular files and %d bytes (interface.submission_cap: %d "
                                  "files, %d bytes): not copied" % (n if n <= max_files else "more than %d" % max_files,
                                                                    total, max_files, max_bytes))
            return out
        if gate_active:
            late = lockdown.late_entries(SUBMISSION, out["t_ref"])
            out["submission_late_writes"] = float(len(late))
            if late:
                out["reasons"].append("%d entr%s of the submission written after the agent phase ended" % (
                    len(late), "y" if len(late) == 1 else "ies"))
        if is_root:
            os.makedirs(snap_root, exist_ok=True)
            os.chmod(snap_root, 0o700)
            lockdown.snapshot_dir(SUBMISSION, snapshot)
            out["submission_snapshot"] = 1.0
            out["snapshot"] = snapshot
    return out


def _restore(early):
    try:
        if early.get("saved") is not None:
            stage_runner.unlock_stage(early["saved"])
        if early.get("parent"):
            os.chmod(early["parent"][0], early["parent"][1])
        mode = os.environ.get("ARB_STAGE_MODE")
        if mode:
            os.chmod(STAGE_DIR, int(mode, 8))
    except OSError:
        pass


def _entry_gates(early):
    """the gates read at the grader's entry: the submission inside its caps (interface.submission_cap) and the
    house's integrity gate (not written after the agent phase, not replaced, not a link)."""
    caps = 0.0 if early["submission_oversize"] > 0 else 1.0
    integrity = 0.0 if (early["submission_late_writes"] > 0 or early["submission_root_rejected"] > 0
                        or early["submission_root_violation"] > 0) else 1.0
    return caps, integrity


def _deadline(cfg, t0):
    raw = os.environ.get("ALE_VERIFY_DEADLINE_SECONDS")
    try:
        total = float(raw)
    except (TypeError, ValueError):
        total = cfg.verify_timeout_s()
    return t0 + total - float(cfg.runtime["reserve_s"])


def _stage_path(cfg, key):
    return os.path.join(STAGE_DIR, cfg.layout[key])


def _unlock_tree(top):
    """give the owner read, write and search permission on top and on every folder below it, by lstat, never following
    a link (the runner's copies of the submission are read-only folders, which an account that is not root cannot
    empty); a folder that cannot be changed or listed is left as it is."""
    todo = [top]
    while todo:
        d = todo.pop()
        try:
            st = os.lstat(d)
        except OSError:
            continue
        if not statmod.S_ISDIR(st.st_mode):
            continue
        try:
            if statmod.S_IMODE(st.st_mode) & statmod.S_IRWXU != statmod.S_IRWXU:
                os.chmod(d, statmod.S_IMODE(st.st_mode) | statmod.S_IRWXU)
            with os.scandir(d) as it:
                for e in it:
                    if e.is_dir(follow_symlinks=False):
                        todo.append(e.path)
        except OSError:
            pass


def _remove_work(work):
    """remove the grader's work folder (the run folders: the runner's copies of the submission, and with them of a token
    it holds, the controllers' homes and logs). As root the tree goes at once; as another account what the runner made
    read-only is unlocked first and the tree removed again. Never raises; True when nothing is left."""
    try:
        shutil.rmtree(work, ignore_errors=True)
        if os.path.lexists(work):
            _unlock_tree(work)
            shutil.rmtree(work, ignore_errors=True)
        if os.path.lexists(work):
            log("the work folder could not be removed completely")
            return False
        return True
    except Exception:  # noqa: BLE001 - the removal never takes the verdict down
        return False


def write_verdict(out, diagnostic):
    from grader import verdict as VD
    if os.environ.get("ALE_VERIFY_PYTHON"):
        return VD.write(out["reward"], out["metrics"], diagnostic=diagnostic)
    return VD.write(out["reward"], out["metrics"], diagnostic=diagnostic,
                    path=os.environ.get("ARB_VERDICT_PATH") or os.path.join(STAGE_DIR, "verdict.local.json"))


def run(cfg, early, t0):
    """the gates, the flights and the score; every failure inside is a GraderVoid (a task error)."""
    from grader import access as AC
    from grader import audit as AU
    from grader import fly as FL
    from grader import grade as GR
    from grader.guard import Guard, GraderVoid
    guard = Guard(deadline=_deadline(cfg, t0))
    hidden = guard.run("hidden_set", lambda: GR.load_hidden_set(
        json.loads(open(_stage_path(cfg, "hidden_set"), "rb").read().decode("utf-8")), cfg))
    guard.run("bands", cfg.bands)
    caps, integrity = _entry_gates(early)
    if caps < 1.0 or integrity < 1.0:
        return GR.zero_run(cfg, {"caps": caps, "integrity": integrity}), early["reasons"]
    want = guard.run("token", AC.read_hash, _stage_path(cfg, "token_sha256"))
    sub = early["snapshot"] if early["submission_snapshot"] else (SUBMISSION if os.path.isdir(SUBMISSION) else None)
    prog = os.path.join(sub, os.path.relpath(cfg.layout["submission_dir"], "submission")) if sub else None
    work = tempfile.mkdtemp(prefix=cfg.runtime["work_prefix"])
    try:
        os.chmod(work, 0o700)
        if prog is None or not os.path.isdir(prog):
            prog = os.path.join(work, "no_controller")          # nothing submitted: every start is a set-up failure
            os.makedirs(prog)
        reads_at_setup = AC.holds_token(prog, cfg.layout["token_file"], want)
        log("registered reads at set-up: %s" % ("yes" if reads_at_setup else "no"))
        audit_ok, findings = guard.run("audit", AU.audit, prog, cfg.libraries())
        paths = {k: _stage_path(cfg, k) for k in ("world_root", "world_table", "sessions_dir", "seed_key", "runner_dir")}
        results = {}
        for name in cfg.sessions():
            entry = hidden[name]
            rec = guard.run(name + ".fly", FL.fly_session, paths, entry, prog, os.path.join(work, "run_" + name),
                            cfg.address_space_bytes(), cfg.recovery_s(), reads_at_setup)
            results[name] = guard.run(name + ".score", GR.grade_session, rec, cfg, entry["scored"])
            log("%s: flown, score %.4f" % (name, results[name]["score"]))
        out = guard.run("reward", GR.grade_run, results, cfg, caps > 0.0, audit_ok)
    finally:
        _remove_work(work)
    out["metrics"]["gate_integrity"] = integrity
    reasons = list(early["reasons"]) + ["audit: " + f for f in findings[:5]]
    return out, reasons


def main():
    t0 = time.time()
    from grader.verdict import fail
    try:
        early = _early_lockdown()
    except Exception:  # noqa: BLE001
        import traceback
        FREEZER.stop()
        fail("the verify-start lockdown failed: " + traceback.format_exc()[-800:])
        return
    try:
        if early["agent_processes_running"] > 0:
            fail("%d agent process(es) survived the verify-start kill and could not be stopped: run void"
                 % early["agent_processes_running"])
        if early["config_error"]:
            fail(early["config_error"])
        from grader import config as CFG
        from grader.guard import GraderVoid
        try:
            cfg = CFG.from_document(_read_config_doc())
        except CFG.ConfigError as exc:
            fail("the configuration is not usable: %s" % exc)
        try:
            out, reasons = run(cfg, early, t0)
        except (GraderVoid, CFG.ConfigError) as exc:
            fail("the verification is void: %s" % exc)
        except Exception as exc:  # noqa: BLE001 - any other failure is the grader's own: a task error, never a score
            fail("the verification is void: %s: %s" % (type(exc).__name__, str(exc)[:400]))
        diag = "reward %.4f; gates %s%s" % (out["reward"], ", ".join("%s %.0f" % kv for kv in sorted(out["gates"].items())),
                                            ("; " + "; ".join(reasons)[:800]) if reasons else "")
        write_verdict(out, diag)
        log("reward %.4f (%.0f s)" % (out["reward"], time.time() - t0))
    finally:
        _restore(early)
        FREEZER.stop()


if __name__ == "__main__":
    main()
