"""Grader of tensegrity_requirements_replica_from_photos (requirements stage, a three-strut tensegrity robot). Root,
verify stage, the agent gone. No agent code runs here: the submission is three static files (YAML / JSON).

Separate verification (task.toml: [verifier] environment_mode = "separate"). The engine releases the agent's sandbox,
starts a fresh verifier sandbox from the task image and restores into it only the declared artifact
/home/user/submission (regular files and directories, owned by root; links and special files are never copied), then
uploads this folder. Everything read here is in this folder (gt/, grader_config.json, anchor.json, reqgrade/, the
kit robotics_grader/), in the image (/opt/venv) or in that restored submission; setup.sh does not run in the
verifier and none of the agent's files besides the three deliverables is read.

  0. FIRST THING, stdlib only (`_early_lockdown`; test.sh has already chmod 0700 the stage and its parent): record when
     the engine began uploading this folder (the earliest ctime of its files), freeze and kill every process of the
     agent account (a survivor is frozen with SIGSTOP passes for up to 3 s; only one still RUNNING after the freeze
     voids the run -> task error: in the separate verifier the image's supervisord respawns the account's desktop as
     fast as it is killed), lock the stage (0700 / 0600, restored at the end). Gate G1 (submission integrity): an entry
     of the submission written at or after the upload began, or a submission ROOT that is a symbolic link, not a
     directory, or a directory replaced or changed after the agent phase (lstat, never followed), zeroes every
     criterion, and nothing behind such a root is read. Then a root-owned 0700 snapshot that every later read uses:
     the three deliverables only, each a regular file opened with O_NOFOLLOW | O_NONBLOCK and read up to
     MAX_FILE_BYTES + 1 bytes (a sparse or oversize file costs at most that and still reads as oversize); notes/ and
     everything else stay behind.
  1. G0 structure and G2 no hedging (reqgrade.load + reqgrade.gates, the hidden copy of the public schema_check.py):
     a failure zeroes every criterion. Anything the structure rules cannot even read is a G0 problem, never a task
     error.
  2. The criteria (weights: grader_config.json `weights`; v1.1 of the scoring):
     net_and_geometry_replication C1 0.25, requirements_sheet C2 0.05, derived_quantities C3 0.03 (reqgrade.c1 / c2 /
     c3), bom_buildability C4 0.37 (kit acceptance) and bom_procurement C5 0.30 (reqgrade.c4 / c5 over the matcher
     reqgrade.match and the authors' cost function reqgrade.cost). Truth: gt/ (ledger, price table, neutral lists;
     freeze.json, frozen for v1.0 and again for v1.1).
     requirements_score = G0 G1 G2 x the weighted mean of the five, on [0, 1]; the verdict's single gated `reward`
     comes from the benchmark kit (robotics_grader.ratio_rewards against anchor.json: value 1.0 = the privileged
     reference, full_at 1.0), so reward = requirements_score. The criteria, the gates and every statistic ride along in
     `metrics` (finite numbers only, no timings: the same submission gives the same verdict byte for byte); the
     per-criterion diagnostics are escaped (a lone surrogate never reaches a UTF-8 writer) and then cut to 380
     characters each and 2000 in all.

Interface of the grading modules (reqgrade/) that this file calls:
  reqgrade.load.SUBMISSION_FILES  {"requirements": "requirements.yaml", "bom": "bom.json", "cable_net": "cable_net.json"}
  reqgrade.load.MAX_FILE_BYTES    int (the snapshot below reads MAX_FILE_BYTES + 1 bytes of each deliverable)
  reqgrade.load.load_submission(sub_dir) -> (docs, problems)        docs keyed as SUBMISSION_FILES; problems: [str]
  reqgrade.gates.check(docs) -> (g0_problems, g2_problems)          lists of str
  reqgrade.c1 / c2 / c3 / c4 / c5 .score(docs, gt, cfg) -> (value in [0, 1], {"diagnostic": str, "stats": {str: num}})
where gt = {stem: parsed JSON of every gt/*.json} with gt["freeze"] = None until freeze.json exists, cfg = the whole
grader_config.json. Knobs: grader_config.json.
"""
import json
import math
import os
import pwd
import signal
import stat as statmod
import sys
import time
import types

sys.dont_write_bytecode = True
STAGE_DIR = os.environ.get("ALE_STAGE_DIR", os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, STAGE_DIR)
HOME = os.environ.get("ALE_HOME", "/home/user")
AGENT_USER = os.path.basename(HOME.rstrip("/")) or "user"
SUBMISSION = os.path.join(HOME, "submission")
SNAP_ROOT = os.environ.get("ARB_SNAPSHOT_ROOT", "/opt/arb-verify")
SNAPSHOT = os.path.join(SNAP_ROOT, "submission")
LATE_GATE_MAX_GAP_S = 600.0

CRITERIA = ("net_and_geometry_replication", "requirements_sheet", "derived_quantities", "bom_buildability",
            "bom_procurement")
MODULES = {"net_and_geometry_replication": "c1", "requirements_sheet": "c2", "derived_quantities": "c3",
           "bom_buildability": "c4", "bom_procurement": "c5"}
# The verify-start snapshot: the deliverables reqgrade.load reads (load.SUBMISSION_FILES), each read up to
# load.MAX_FILE_BYTES + 1 bytes. Stdlib only, so the lockdown imports nothing of the grader; main() checks both against
# reqgrade/load.py before any grading (a mismatch is a task error).
DELIVERABLES = ("requirements.yaml", "bom.json", "cable_net.json")
SNAPSHOT_MAX_BYTES = 2 * 10 ** 6 + 1
DIAG_PER_CRITERION = 380
DIAG_TOTAL = 2000


def log(msg):
    print("[grader] " + msg, file=sys.stderr, flush=True)


# ------------------------------------------------------------------------- vendored lockdown helpers (stdlib only)
# Function bodies copied from the published sister task bimanual_requirements_replica_from_photos (benchmark main
# b7b55ac): tests/arm/assembly_six/lockdown.py (sha256 741b9735...0068: live_uid_processes, stop_and_kill_uid,
# stage_upload_time, late_entries) and tests/arm/assembly_cell/runner.py (sha256 e2699fe0...ab2a: chmod_tree,
# lock_stage, unlock_stage). Kept in this file because this task ships no other stage-lock package; they are reached
# through the two namespaces `lockdown` and `runner` below, so the grader's own _early_lockdown reads exactly as in the
# sister task (and the benchmark's survivor test, which runs it against stand-ins for these two names, applies as is).
def live_uid_processes(uid):
    """PIDs of every live (non-zombie) process owned by `uid` (Linux /proc)."""
    pids = []
    for name in os.listdir("/proc"):
        if not name.isdigit():
            continue
        try:
            with open("/proc/{}/status".format(name)) as fh:
                u, state = None, None
                for line in fh:
                    if line.startswith("Uid:"):
                        u = int(line.split()[1])
                    elif line.startswith("State:"):
                        state = line.split()[1]
                    if u is not None and state is not None:
                        break
            if u == int(uid) and not (state or "").startswith("Z"):
                pids.append(int(name))
        except (OSError, ValueError, IndexError):
            continue
    return pids


def _signal_all(pids, sig):
    for pid in pids:
        try:
            os.kill(pid, sig)
        except OSError:
            pass


def stop_and_kill_uid(uid, grace_s=5.0, settle_s=0.05):
    """Freeze, then kill, every live process of `uid`. Returns (killed, remaining)."""
    seen = set()
    deadline = time.time() + float(grace_s)
    remaining = []
    while True:
        pids = live_uid_processes(uid)
        if not pids:
            return len(seen), 0
        seen.update(pids)
        _signal_all(pids, signal.SIGSTOP)
        again = live_uid_processes(uid)
        if set(again) - set(pids):
            _signal_all(again, signal.SIGSTOP)      # newcomers frozen too; scan once more before killing
            seen.update(again)
            if time.time() > deadline:
                _signal_all(again, signal.SIGKILL)
                time.sleep(settle_s)
                remaining = live_uid_processes(uid)
                return len(seen), len(remaining)
            continue
        _signal_all(again, signal.SIGKILL)
        time.sleep(settle_s)
        remaining = live_uid_processes(uid)
        if not remaining:
            return len(seen), 0
        if time.time() > deadline:
            return len(seen), len(remaining)


def stage_upload_time(stage_dir):
    """Earliest ctime (s) of a regular file under `stage_dir`, or None. Call BEFORE anything chmods the files."""
    t = None
    for dirpath, _, filenames in os.walk(stage_dir):
        for fn in filenames:
            try:
                st = os.lstat(os.path.join(dirpath, fn))
            except OSError:
                continue
            if statmod.S_ISREG(st.st_mode) and (t is None or st.st_ctime < t):
                t = st.st_ctime
    return t


def late_entries(root, t_ref, exclude=(), skip_uids=(0,)):
    """Entries under `root` (files, dirs, links; `root` itself excluded) whose ctime or mtime >= t_ref, as a sorted list
    of (relpath, kind, ctime, mtime). Entries owned by a uid in `skip_uids` (root) are skipped."""
    out = []
    if t_ref is None or not os.path.isdir(root):
        return out
    ex = tuple(os.path.abspath(e) for e in exclude)
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        if any(os.path.abspath(dirpath).startswith(e) for e in ex):
            dirnames[:] = []
            continue
        for name in filenames + dirnames:
            p = os.path.join(dirpath, name)
            if any(os.path.abspath(p).startswith(e) for e in ex):
                continue
            try:
                st = os.lstat(p)
            except OSError:
                continue
            if st.st_uid in skip_uids:
                continue
            if max(st.st_ctime, st.st_mtime) >= t_ref:
                kind = "dir" if statmod.S_ISDIR(st.st_mode) else ("link" if statmod.S_ISLNK(st.st_mode) else "file")
                out.append((os.path.relpath(p, root), kind, st.st_ctime, st.st_mtime))
    return sorted(out)


def chmod_tree(root, dir_mode, file_mode):
    os.chmod(root, dir_mode)
    for dirpath, dirnames, filenames in os.walk(root):
        for d in dirnames:
            os.chmod(os.path.join(dirpath, d), dir_mode)
        for f in filenames:
            os.chmod(os.path.join(dirpath, f), file_mode)


def lock_stage(root):
    """chmod 0700 every directory and 0600 every file under `root` (root only). Returns the modes to restore,
    or None when not root."""
    if os.getuid() != 0:
        return None
    saved = []
    for dirpath, dirnames, filenames in os.walk(root):
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


lockdown = types.SimpleNamespace(live_uid_processes=live_uid_processes, stop_and_kill_uid=stop_and_kill_uid,
                                 stage_upload_time=stage_upload_time, late_entries=late_entries)
runner = types.SimpleNamespace(chmod_tree=chmod_tree, lock_stage=lock_stage, unlock_stage=unlock_stage)


# ------------------------------------------------------------------------------------------------ the lockdown
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


def _snapshot(src, dst):
    """Root-owned 0700 / 0600 copy of the three DELIVERABLES of `src`: each opened with O_NOFOLLOW | O_NONBLOCK (a
    symbolic link or a FIFO never blocks or redirects the read), kept only if it is a regular file, and read up to
    SNAPSHOT_MAX_BYTES (a sparse file is never written out at its apparent size; an oversize deliverable still reads
    as oversize). Everything else (notes/, links, special files) stays behind. Returns the number copied."""
    import shutil
    if os.path.isdir(dst):
        shutil.rmtree(dst, ignore_errors=True)
    os.makedirs(dst)
    kept = 0
    for name in DELIVERABLES:
        try:
            fd = os.open(os.path.join(src, name), os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        except OSError:
            continue
        try:
            if not statmod.S_ISREG(os.fstat(fd).st_mode):
                continue
            with open(os.path.join(dst, name), "wb") as out:
                left = SNAPSHOT_MAX_BYTES
                while left > 0:
                    blk = os.read(fd, min(left, 1 << 20))
                    if not blk:
                        break
                    out.write(blk)
                    left -= len(blk)
            kept += 1
        finally:
            os.close(fd)
    runner.chmod_tree(dst, 0o700, 0o600)
    return kept


def _early_lockdown():
    out = {"agent_processes_killed": 0.0, "agent_processes_left": 0.0, "agent_processes_running": 0.0,
           "stage_locked": 0.0, "stage_parent_locked": 0.0,
           "submission_late_writes": 0.0, "submission_root_violation": 0.0, "late_write_gate_active": 0.0,
           "submission_snapshot": 0.0, "saved": None, "parent": None, "reasons": [], "t_ref": None}
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
            # the image's supervisord respawns the account's desktop as fast as it is killed: a process that outlives
            # the kill is the image's own; it is frozen, and only one still running once frozen voids the run
            out["agent_processes_running"] = float(len(_still_running(uid)))
            log("%d process(es) of the agent account outlived the kill at verify start, %d still running after the "
                "freeze" % (left, out["agent_processes_running"]))
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
    try:
        rst = os.lstat(SUBMISSION)
    except OSError:
        rst = None
    root_problem = None
    if rst is not None:
        if statmod.S_ISLNK(rst.st_mode):
            root_problem = "the submission root is a symbolic link"
        elif not statmod.S_ISDIR(rst.st_mode):
            root_problem = "the submission root is not a directory"
        elif gate_active and max(rst.st_ctime, rst.st_mtime) >= t_ref:
            root_problem = "the submission directory itself was replaced or changed after the agent phase ended"
    if root_problem:
        out["submission_root_violation"] = 1.0
        out["submission_late_writes"] += 1.0
        out["reasons"].append(root_problem)
    if rst is not None and statmod.S_ISDIR(rst.st_mode) and not root_problem:
        if gate_active:
            late = lockdown.late_entries(SUBMISSION, t_ref)
            out["submission_late_writes"] += float(len(late))
            if late:
                out["reasons"].append("%d entr%s of the submission written after the agent phase ended (%s)" % (
                    len(late), "y" if len(late) == 1 else "ies", ", ".join(repr(e[0]) for e in late[:5])))
        os.makedirs(SNAP_ROOT, exist_ok=True)
        os.chmod(SNAP_ROOT, 0o700)
        _snapshot(SUBMISSION, SNAPSHOT)
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


# ------------------------------------------------------------------------------------------------ the verdict
def _fin(x):
    """A finite float, or None (a value the engine's record would refuse)."""
    if isinstance(x, bool):
        return float(x)
    try:
        v = float(x)
    except (TypeError, ValueError, OverflowError):
        return None
    return v if math.isfinite(v) else None


def _escape(text):
    """A lone surrogate (from an agent string or a non-UTF-8 file name) would make every UTF-8 writer raise."""
    return str(text).encode("utf-8", "backslashreplace").decode("utf-8")


def _cut(text, n):
    text = str(text)
    return text if len(text) <= n else (text[:n - 3] + "..." if n >= 3 else text[:max(0, n)])


def write_verdict(rewards, weights, stats, diagnostics, gates_ok=True):
    """One gated `reward` (the kit's envelope); the criteria and statistics go to `metrics` (finite numbers, in name
    order), the per-criterion diagnostics into the verdict's `diagnostic` text (escaped, then cut). In the engine
    (ALE_VERIFY_PYTHON set) the kit writes /logs/verifier/reward.json and verification.json through ale_verify; a
    host-side run writes the plain envelope to ROBOTICS_GRADER_VERDICT_PATH, else verdict.local.json here."""
    from robotics_grader import ratio_rewards, read_anchor
    from robotics_grader import write_verdict as kit_write
    wsum = sum(float(weights[c]) for c in CRITERIA)
    overall = sum(float(rewards[c]) * float(weights[c]) for c in CRITERIA) / wsum if gates_ok else 0.0
    overall = min(max(overall, 0.0), 1.0)
    anchor, meta = read_anchor()
    extra = {}
    for k in sorted(stats):
        v = stats[k]
        if isinstance(v, (int, float)):
            f = _fin(v)
            if f is not None:
                extra[k] = f
    for c in CRITERIA:
        extra[c] = _fin(rewards[c]) or 0.0
    verdict = ratio_rewards(overall, anchor, higher_is_better=meta.get("direction", "higher") == "higher",
                            metric_name=meta.get("metric", "requirements_score"), extra=extra,
                            full_at=meta.get("full_at", 1.0))
    verdict["metrics"] = {k: verdict["metrics"][k] for k in sorted(verdict["metrics"])}
    text = "; ".join("%s %.4f: %s" % (c, float(rewards[c]), _cut(_escape(diagnostics.get(c, "")), DIAG_PER_CRITERION))
                     for c in CRITERIA)
    text = _cut(_escape(text), DIAG_TOTAL)
    if os.environ.get("ALE_VERIFY_PYTHON"):
        recorded = {"rewards": verdict["rewards"], "metrics": verdict["metrics"]}
        kit_write(recorded, diagnostic=text)
        return overall
    kit_write(verdict, path=os.environ.get("ROBOTICS_GRADER_VERDICT_PATH") or os.path.join(STAGE_DIR, "verdict.local.json"),
              diagnostic=text)
    return overall


def read_gt(gt_dir):
    """{stem: parsed JSON} of every gt/*.json; freeze = None until gt/freeze.json exists."""
    gt = {}
    for fn in sorted(os.listdir(gt_dir)):
        if fn.endswith(".json"):
            with open(os.path.join(gt_dir, fn)) as fh:
                gt[fn[:-5]] = json.load(fh)
    gt.setdefault("freeze", None)
    return gt


def main():
    t0 = time.time()
    early = _early_lockdown()
    stats = {k: v for k, v in early.items() if isinstance(v, float)}
    log("lockdown %.3f s" % (time.time() - t0))
    try:
        if early["agent_processes_running"] > 0:
            raise RuntimeError("%d agent process(es) survived the verify-start kill and could not be stopped: run void"
                               % early["agent_processes_running"])
        with open(os.path.join(STAGE_DIR, "grader_config.json")) as fh:
            cfg = json.load(fh)
        weights = cfg["weights"]
        if tuple(cfg.get("criteria", CRITERIA)) != CRITERIA:
            raise RuntimeError("grader_config.json names the criteria %r, verify.py %r" % (cfg.get("criteria"), CRITERIA))
        stats["config_provisional"] = 1.0 if cfg.get("provisional") else 0.0
        sub = SNAPSHOT if early["submission_snapshot"] else SUBMISSION
        rewards = {c: 0.0 for c in CRITERIA}
        diagnostics = {}
        stats["gate_g0_structure"] = -1.0
        stats["gate_g2_no_hedging"] = -1.0
        if early["submission_late_writes"] > 0:
            stats["gate_g1_submission_integrity"] = 0.0
            diagnostics = {c: "G1: " + "; ".join(early["reasons"]) for c in CRITERIA}
            return write_verdict(rewards, weights, stats, diagnostics, gates_ok=False)
        stats["gate_g1_submission_integrity"] = 1.0
        from reqgrade import load as LD
        from reqgrade import gates as GT
        if sorted(DELIVERABLES) != sorted(LD.SUBMISSION_FILES.values()) or SNAPSHOT_MAX_BYTES != LD.MAX_FILE_BYTES + 1:
            raise RuntimeError("the verify-start snapshot (DELIVERABLES, SNAPSHOT_MAX_BYTES) disagrees with "
                               "reqgrade/load.py's SUBMISSION_FILES / MAX_FILE_BYTES + 1")
        try:
            docs, problems = LD.load_submission(sub)
            g0, g2 = (list(problems), []) if problems else GT.check(docs)
        except Exception as e:  # noqa: BLE001  (agent-written values the structure rules cannot read fail G0)
            docs, g0, g2 = {}, ["the submission cannot be checked: %s: %r" % (type(e).__name__, str(e)[:200])], []
        stats["gate_g0_structure"] = 0.0 if g0 else 1.0
        if g0:
            stats["g0_problems"] = float(len(g0))
            diagnostics = {c: "G0 structure: " + "; ".join(str(p) for p in g0[:6]) for c in CRITERIA}
            return write_verdict(rewards, weights, stats, diagnostics, gates_ok=False)
        stats["gate_g2_no_hedging"] = 0.0 if g2 else 1.0
        if g2:
            stats["g2_problems"] = float(len(g2))
            diagnostics = {c: "G2 no hedging: " + "; ".join(str(p) for p in g2[:6]) for c in CRITERIA}
            return write_verdict(rewards, weights, stats, diagnostics, gates_ok=False)
        # reqgrade.load also checks the sha256 pins of freeze.json; the plain reader serves until that module exists
        gt = LD.load_gt(STAGE_DIR) if hasattr(LD, "load_gt") else read_gt(os.path.join(STAGE_DIR, "gt"))
        stats["freeze_present"] = 0.0 if gt["freeze"] is None else 1.0
        import importlib
        for c in CRITERIA:
            t1 = time.time()
            mod = importlib.import_module("reqgrade." + MODULES[c])
            value, det = mod.score(docs, gt, cfg)
            value = _fin(value)
            if value is None or not 0.0 <= value <= 1.0:
                raise RuntimeError("criterion %s returned %r, not a number in [0, 1]" % (c, value))
            rewards[c] = value
            diagnostics[c] = det.get("diagnostic", "")
            for k, v in sorted((det.get("stats") or {}).items()):
                stats[k] = v
            log("%s %.4f in %.3f s" % (c, value, time.time() - t1))
        log("verify %.3f s" % (time.time() - t0))
        return write_verdict(rewards, weights, stats, diagnostics)
    finally:
        _restore(early)


if __name__ == "__main__":
    main()
