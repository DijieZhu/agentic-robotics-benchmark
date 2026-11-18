"""Grader of racecar_time_trial_on_unseen_track (a tenth-scale car's race day at venues it has never seen): runs as ROOT
in the verify stage. Verification is separate (task.toml): this runs in a fresh verifier sandbox from the task image,
with only the agent's submission restored into it (root-owned); nothing else of the agent's sandbox is here.

The agent left /home/user/submission/drive.py (solve(car): the car's program for one venue's day: practice with its
mapping period, two timed heats, the obstacle test). This grader:
  0. FIRST, stdlib only (`_early_lockdown`, the house pattern; tests/test.sh closed the stage folder already): records
     the upload time, freezes and kills every process of the agent account and keeps it stopped until the verdict is
     written (cellkit.verify_freeze), locks the stage (every file 0600, every folder 0700), snapshots the submission
     root-owned (every later read of the submission uses the snapshot), gates late writes, quarantines late entries;
  1. checks the vendored grader against tests/grader_manifest.json (the frozen cell, generator, contract and scoring
     trees of the freeze recorded in tests/hidden/selection.json, the client library, the house's process kit; a changed
     file is a task error), the cell's configuration tests/grader_config.json (byte-identical to the cell's own), the
     anchor (tests/anchor.json; its full_at must be the configuration's) and the hidden set (ttverify.hiddenset:
     tests/hidden/selection.json, the venues' world files and reference records, pinned by sha256 and layout digest;
     a malformed set is a task error whose message names no value of it);
  2. proves the run regime holds in this sandbox (ttverify.regime: the account `nobody`, no new privileges, no
     capabilities, two seccomp filters, the limits, the stage closed to the program; the house filter refuses an inet
     socket and the killing filter kills; no new process can be made, a thread can; the documented tools
     import in the program's environment; a failure is a task error);
  3. gate submission_ok: the scoring package's static audit (sketch.scoring.audit: drive.py defines solve, no weight or
     model files, no training calls, no process, network or foreign-function module, no os process call, no computed
     import, no non-literal exec / eval / compile, <= 60 MB; byte-identical copies of the given client library are
     exempt) and one import of drive.py under the regime (120 s, 120 CPU s); a failed gate skips the venues;
  4. runs the program once per hidden venue in the vendored cell (sketch.referee.Venue, the token's sha256 from the
     hidden side: only the reference program's token unlocks the privileged calls) over the cell's own wire
     (sketch.wire.server.serve: a new process in its own process group, the budgets of the configuration) under the
     regime: account, limits, filters, a fresh run directory with a root-owned copy of the submission and of carkit;
     a venue whose cell failed (internal_error, physics_fault) is run once more and the second run counts; after
     each run survivors are counted, the account is killed, its writes under the shared folders swept;
  5. scores with the vendored scoring package (sketch.scoring.report: per venue 0.35 fast + 0.35 consecutive + 0.30
     obstacle test against the frozen reference; the mean times the gates submission_ok, step_called, clean_run (every
     venue has a result, no cell failure, no survivor, no unkillable process), no_network (no process ended by the
     killing filter, no remote inet socket seen) and no_late_write (no late entry, the snapshot unchanged)); the
     reward = clip(score / anchor / full_at) through the benchmark kit, which must agree with the score record's;
  6. (a record only) the results, the score record and the program's stderr per venue under the root-only
     /opt/arb-verify.
Untouched sandbox: no drive.py -> submission_ok false -> 0. Oracle: the reference program and its token -> its own
numbers on every venue -> 1.
"""

import json
import math
import os
import pwd
import shutil
import stat as statmod
import sys
import time
import traceback

sys.dont_write_bytecode = True

STAGE_DIR = os.environ.get("ALE_STAGE_DIR", os.path.dirname(os.path.abspath(__file__)))
GRADER_ROOT = os.path.join(STAGE_DIR, "grader")              # the vendored trees: sketch/ and carkit/
sys.path.insert(0, STAGE_DIR)

HOME = os.environ.get("ALE_HOME", "/home/user")
AGENT_USER = os.path.basename(HOME.rstrip("/")) or "user"
SUBMISSION = os.path.join(HOME, "submission")
RUN_ROOT = os.environ.get("ARB_RUN_ROOT", "/opt/arb-tt")
RUNS_ROOT = os.path.join(RUN_ROOT, "runs")
SNAPSHOT = os.path.join(RUN_ROOT, "submission")
QUARANTINE = os.path.join(RUN_ROOT, "quarantine")
RECORD_ROOT = os.environ.get("ARB_RECORD_ROOT", "/opt/arb-verify")
LOGS = os.path.join(RECORD_ROOT, "logs")
OUT = os.path.join(RECORD_ROOT, "out")
STICKY_DIRS = [d for d in ("/tmp", "/dev/shm", "/var/tmp", "/run/lock") if os.path.isdir(d)]
WATCH_ROOTS = STICKY_DIRS + [HOME]
LATE_GATE_MAX_GAP_S = 600.0
METRIC = "race_day_score"

from cellkit import lockdown, runner, verify_freeze  # noqa: E402  (stdlib only)

FREEZER = [None]        # the agent account's verify_freeze.AgentFreezer: started by _early_lockdown, stopped after the verdict


def log(msg):
    print(msg, file=sys.stderr, flush=True)


class TaskError(RuntimeError):
    """The grading broke: no verdict is written (the engine records a failed verification, never the agent's zero)."""


def fail(reason):
    raise TaskError(reason)


def _early_lockdown():
    """The house's verify-start lockdown (pump_skid_startup_and_operation v0.6, unchanged in substance)."""
    out = {"agent_processes_killed": 0.0, "agent_processes_left": 0.0, "stage_locked": 0.0, "stage_parent_locked": 0.0,
           "stage_lock_latency_ms": -1.0, "submission_late_writes": 0.0, "late_writes_quarantined": 0.0,
           "late_write_gate_active": 0.0, "submission_snapshot": 0.0, "saved": None, "parent": None, "t_ref": None,
           "late": [], "reasons": []}
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
        # verification is separate: a process of the agent account here is the image's own respawned desktop; it is a
        # statistic, frozen with the rest of the account until the verdict (house rule since 2026-11-08)
        if left:
            log("[grader] %d process(es) of the agent account (the image's respawned desktop) outlived the kill at "
                "verify start: frozen until the verdict" % left)
    FREEZER[0] = verify_freeze.AgentFreezer.for_user(AGENT_USER).start()
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
        os.makedirs(RUN_ROOT, exist_ok=True)
        os.chmod(RUN_ROOT, 0o755)
        lockdown.snapshot_dir(SUBMISSION, SNAPSHOT)
        out["submission_snapshot"] = 1.0
    if gate_active:
        q = 0
        for root in WATCH_ROOTS:
            late = lockdown.late_entries(root, t_ref, exclude=(SUBMISSION, RUN_ROOT, STAGE_DIR))
            if late:
                log("[grader] %d late entr%s under %s quarantined" % (len(late), "y" if len(late) == 1 else "ies", root))
                q += lockdown.quarantine(root, late, QUARANTINE)
        out["late_writes_quarantined"] = float(q)
    out["stage_lock_latency_ms"] = lockdown.process_age_ms()
    return out


EARLY = _early_lockdown() if __name__ == "__main__" else {"saved": None, "parent": None, "reasons": [], "late": [], "t_ref": None}

from cellkit import netaudit  # noqa: E402  (stdlib only)
from ttverify import manifest  # noqa: E402  (stdlib only)
from ttverify.regime import Regime  # noqa: E402  (stdlib only)
from ttverify.regime import TaskError as RegimeError  # noqa: E402


def program_source():
    """The root-owned snapshot of the submission (taken by the lockdown), or the submission itself on a host run."""
    return SNAPSHOT if os.path.isdir(SNAPSHOT) else SUBMISSION


def record_dir(path):
    os.makedirs(path, exist_ok=True)
    try:
        os.chmod(RECORD_ROOT, 0o700)
    except OSError:
        pass
    return path


def _dump(path, obj):
    record_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, sort_keys=True, default=str)


def load_json(p):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def _settle_desktop(max_s=60.0, quiet_s=8.0, poll_s=0.25):
    """The house's wait (pump skid v0.6) until the frozen desktop of the agent account is quiet before the network
    baseline: its processes and the namespace's outbound counters unchanged for `quiet_s`."""
    fr = FREEZER[0]
    if os.getuid() != 0 or fr is None or fr.uid is None:
        return 0.0, 0
    t0 = time.time()
    first = last = (netaudit.net_counters(), sorted(verify_freeze.uid_processes(fr.uid)))
    quiet_since = t0
    while time.time() - t0 < max_s:
        time.sleep(poll_s)
        now = (netaudit.net_counters(), sorted(verify_freeze.uid_processes(fr.uid)))
        if now != last:
            last, quiet_since = now, time.time()
        elif time.time() - quiet_since >= quiet_s:
            break
    seen = sum(max(0, int(last[0].get(k, 0)) - int(first[0].get(k, 0))) for k in first[0] if k != "unreadable")
    return time.time() - t0, seen


# ------------------------------------------------------------------------------------------------ one venue
def run_venue(k, entry, rn, cell_cfg, priv_sha, carkit_dir):
    """One venue: a fresh cell, a fresh process of the program over the cell's wire under the regime, the cell's
    record. A failure on the grader's side (the cell's build, the launch, the cleanup) is a task error."""
    import copy

    from carkit.client import launcher_argv
    from sketch.examples import check
    from sketch.referee import Venue
    from sketch.wire import runner as wire_runner
    from sketch.wire import server as wire_server
    world = copy.deepcopy(entry["world"])
    if check.world_digest(world) != entry["layout_digest"]:
        fail("venue %d: the world changed after it was loaded" % k)
    try:
        venue = Venue(world, cell_cfg, priv_token_sha256=priv_sha)
    except Exception:  # noqa: BLE001 - the cell depends on the world only: a build failure is the task's
        fail("venue %d: the cell could not be built\n%s" % (k, traceback.format_exc()[-1500:]))
    dirs = rn.prepare(program_source(), carkit_dir)
    env = rn.program_env(wire_runner.child_env)(dirs["home"], dirs["tmp"], dirs["lib"])
    argv = rn.wrap(launcher_argv(sys.executable, dirs["lib"], "drive"), cpu_s=float(venue.caps["cpu"]) + 60.0)
    before = rn.snapshot(dirs)
    stats = {}
    err_path = os.path.join(record_dir(LOGS), "venue%02d_%s.stderr" % (k, dirs["tag"]))
    sampler = netaudit.SocketSampler(uid=rn.uid, pids_fn=None if rn.uid is not None else (lambda: []), period_s=0.5)
    counters = netaudit.NetCounters()
    t0 = time.time()
    clean = {}
    try:
        with counters, sampler:
            try:
                wire_server.serve(venue, argv, cwd=dirs["program"], env=env, stderr_path=err_path, stats=stats, log=log)
            finally:
                clean = rn.cleanup(dirs, before)          # survivors, then every process of the account is gone
    except RegimeError as e:
        fail("venue %d: %s" % (k, e))
    res = venue.result()
    if res.get("layout_digest") != entry["layout_digest"]:
        fail("venue %d: the cell's record names another layout" % k)
    attempt = rn.note_exit(stats.get("exit_code"))
    info = {"k": k, "world_id": entry["world_id"], "wall_s": round(time.time() - t0, 3), "end_reason": res["end_reason"],
            "exit_code": stats.get("exit_code"), "requests": stats.get("requests", 0), "cpu_s": round(float(stats.get("cpu_s", 0.0)), 3),
            "network_attempt": bool(attempt), "remote_sockets_seen": int(sampler.max_seen), "socket_samples": int(sampler.samples),
            "net_outbound": int(counters.total), "net_delta": dict(counters.delta), "run_tag": dirs["tag"], **clean}
    return res, info


def run_venue_with_retry(k, entry, rn, cell_cfg, priv_sha, carkit_dir):
    from sketch.scoring import CELL_FAILURES
    res, info = run_venue(k, entry, rn, cell_cfg, priv_sha, carkit_dir)
    info["retried"] = False
    if res["end_reason"] in CELL_FAILURES:
        log("[grader] venue %d: the cell failed (%s); the venue runs once more" % (k, res["end_reason"]))
        first = info
        res, info = run_venue(k, entry, rn, cell_cfg, priv_sha, carkit_dir)
        info["retried"] = True
        info["first_run"] = first
    return res, info


# ------------------------------------------------------------------------------------------------ the verdict
def write_verdict(score, metrics, text, expect_reward=None):
    """One gated `reward` in the benchmark kit's envelope (robotics_grader.ratio_rewards against tests/anchor.json); the
    gates, the criterion and every statistic go to `metrics` (finite numbers only). With `expect_reward` (the score
    record's own reward) the two computations must agree before anything is written."""
    from robotics_grader import ratio_rewards, read_anchor
    from robotics_grader import write_verdict as kit_write_verdict
    anchor, meta = read_anchor()
    extra = {}
    for k, v in metrics.items():
        try:
            v = float(v)
        except (TypeError, ValueError):
            continue
        if math.isfinite(v):
            extra[k.replace(".", "_").replace("/", "_")] = v
    verdict = ratio_rewards(float(score), anchor, higher_is_better=meta.get("direction", "higher") == "higher",
                            metric_name=meta.get("metric", METRIC), extra=extra, full_at=meta.get("full_at", 1.0))
    if expect_reward is not None and abs(float(verdict["rewards"]["reward"]) - float(expect_reward)) > 1e-5:
        fail("the kit's reward %.6f and the score record's %.6f disagree" % (verdict["rewards"]["reward"], expect_reward))
    recorded = {}
    for k, v in sorted(verdict["metrics"].items()):
        try:
            v = float(v)
        except (TypeError, ValueError):
            continue
        if math.isfinite(v):
            recorded[k] = v
    env = {"rewards": verdict["rewards"], "metrics": recorded}
    if os.environ.get("ALE_VERIFY_PYTHON"):
        kit_write_verdict(env, diagnostic=text[:1900])
    else:
        kit_write_verdict(env, path=os.environ.get("ROBOTICS_GRADER_VERDICT_PATH") or os.path.join(RECORD_ROOT, "verdict.local.json"),
                          diagnostic=text[:1900])
    return float(verdict["rewards"]["reward"])


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
    code = 0
    try:
        _main_locked()
    except TaskError as e:
        log("[grader] TASK ERROR: %s" % e)
        code = 1
    except Exception:  # noqa: BLE001
        log("[grader] TASK ERROR (grader exception):\n%s" % traceback.format_exc())
        code = 1
    finally:
        _restore_modes()
        if FREEZER[0] is not None:
            FREEZER[0].stop()                       # after the verdict; the frozen stay stopped
        shutil.rmtree(RUN_ROOT, ignore_errors=True)
    return code


def _main_locked():
    t_start = time.time()
    vcfg = load_json(os.path.join(STAGE_DIR, "verify_config.json"))
    problems = manifest.check(STAGE_DIR)
    if problems:
        fail("the vendored grader differs from its manifest: %s" % "; ".join(problems[:5]))
    sys.path.insert(0, GRADER_ROOT)
    from robotics_grader import read_anchor
    from sketch.examples import check
    from sketch.referee import load_config
    from sketch.scoring import report
    from sketch.scoring.audit import audit_submission, submission_digest, tree_sha256
    from ttverify import hiddenset

    metrics = {"grader_exception": 0.0}
    for k in ("stage_locked", "stage_parent_locked", "stage_lock_latency_ms", "agent_processes_killed",
              "agent_processes_left", "submission_late_writes", "late_writes_quarantined", "late_write_gate_active",
              "submission_snapshot"):
        metrics[k] = float(EARLY.get(k, -1.0 if k == "stage_lock_latency_ms" else 0.0))
    # 1. the cell's configuration, the anchor, the hidden set
    cfg_path = os.path.join(STAGE_DIR, "grader_config.json")
    with open(cfg_path, "rb") as f, open(os.path.join(GRADER_ROOT, "sketch", "referee", "grader_config.json"), "rb") as g:
        if f.read() != g.read():
            fail("tests/grader_config.json is not the cell's configuration")
    cell_cfg = load_config(cfg_path)
    anchor, meta = read_anchor()
    if abs(float(meta.get("full_at", 1.0)) - float(cell_cfg["scoring"]["full_at"])) > 1e-12 or meta.get("metric") != METRIC:
        fail("tests/anchor.json disagrees with the configuration (full_at or metric)")
    try:
        hs = hiddenset.load(os.path.join(STAGE_DIR, "hidden"), check)
    except ValueError as e:
        fail("the hidden set: %s" % e)
    venues = hs["venues"]
    metrics["hidden_set_venues"] = float(len(venues))
    metrics["hidden_set_placeholder"] = 1.0 if hs["placeholder"] else 0.0
    metrics["hidden_set_files_pinned"] = float(hs["n_pinned"])
    if hs["placeholder"]:
        log("[grader] the hidden set is the development state's PLACEHOLDER (not the hidden set)")
    # 2. the run regime
    from sketch.wire.runner import child_env
    try:
        rn = Regime(vcfg, RUNS_ROOT, AGENT_USER, log=log)
    except RegimeError as e:
        fail(str(e))
    child_env = rn.program_env(child_env)              # the cell's environment + program_env_extra
    if FREEZER[0] is not None and rn.uid is not None and rn.uid == FREEZER[0].uid:
        fail("the runner account is the agent account")
    metrics["runner_uid"] = float(rn.uid) if rn.uid is not None else -1.0
    settle_s, settle_seen = _settle_desktop()
    metrics["desktop_settle_s"] = round(settle_s, 1)
    metrics["net_settle_events"] = float(settle_seen)
    try:
        st = rn.selftest(child_env, STAGE_DIR)
    except RegimeError as e:
        fail(str(e))
    metrics["regime_selftest_ok"] = 1.0
    log("[grader] run regime: %s" % json.dumps(st.get("probe"), sort_keys=True))
    log("[grader] run regime: no new process (%s); the documented tools import (%s)" % (st.get("noproc"), st.get("tools")))
    carkit_dir = os.path.join(GRADER_ROOT, "carkit")
    # 3. the submission gate
    src = program_source()
    digest_start = submission_digest(src)
    live_start = submission_digest(SUBMISSION)
    aud = audit_submission(src, cell_cfg["route_gate"], exempt_sha256=tree_sha256(carkit_dir))
    reasons = list(aud["reasons"])
    metrics["submission_files"] = float(aud["files"])
    metrics["submission_bytes"] = float(aud["bytes"])
    metrics["audit_warnings"] = float(len(aud["warnings"]))
    imp = None
    if aud["ok"]:
        ok, why, imp = rn.import_check(src, carkit_dir, child_env, wall_s=float(vcfg["limits"]["import_check_s"]),
                                       cpu_s=float(vcfg["limits"]["import_cpu_s"]))
        metrics["import_check_s"] = float(imp["wall_s"])
        if not ok:
            reasons.append(why)
    submission_ok = not reasons
    metrics["submission_ok"] = 1.0 if submission_ok else 0.0
    log("[grader] submission_ok=%s%s" % (submission_ok, "" if submission_ok else " (%s)" % "; ".join(reasons[:4])))
    # 4. the venues
    results, infos = [], []
    if submission_ok:
        for k, entry in enumerate(venues, 1):
            log("[grader] venue %d/%d" % (k, len(venues)))
            res, info = run_venue_with_retry(k, entry, rn, cell_cfg, hs["priv_token_sha256"], carkit_dir)
            results.append(res)
            infos.append(info)
            _dump(os.path.join(OUT, "results", "venue%02d.json" % k), res)
            log("[grader]   end %s laps %s T_best %s N_best %s avoid %s touches %d crashes %d wall %.0f s exit %s survivors %d%s" % (
                res["end_reason"], res["stats"]["laps_by_session"], res["time_trial"]["T_best_s"], res["time_trial"]["N_best"],
                json.dumps(res["avoid"], sort_keys=True), res["stats"]["touches"], res["stats"]["crashes"], info["wall_s"],
                info["exit_code"], info["survivors"], " NETWORK ATTEMPT" if info["network_attempt"] else ""))
    # 5. the gates and the score
    rs = rn.stats
    no_network = rs["network_attempts"] == 0 and all(i["remote_sockets_seen"] <= 0 for i in infos)
    clean_procs = rs["survivors"] == 0 and rs["left"] == 0
    digest_end = submission_digest(src)
    no_late = not EARLY.get("reasons") and digest_end == digest_start and submission_digest(SUBMISSION) == live_start
    gates = {"submission_ok": submission_ok, "step_called": True, "clean_run": clean_procs, "no_network": no_network,
             "no_late_write": no_late}
    refs = [v["reference"] for v in venues]
    rep = report(results, refs, cell_cfg, gates=gates, anchor_value=float(anchor))
    score = rep["score"]
    _dump(os.path.join(OUT, "score.json"), rep)
    _dump(os.path.join(OUT, "runs.json"), {"infos": infos, "import_check": imp, "audit": aud, "regime": rs,
                                           "survivor_cmdlines": rn.survivor_cmdlines})
    for g, v in score["gates"].items():
        metrics["gate_" + g] = 1.0 if v else 0.0
    metrics["gate"] = float(score["gate"])
    metrics["time_trial_score"] = float(score["time_trial_score"])
    for row, ref in zip(score["venues"], refs):
        i = venues.index(next(v for v in venues if v["world_id"] == row["world_id"])) + 1
        for key in ("fast", "cons", "avoid_static", "avoid_dynamic", "avoid", "venue"):
            metrics["venue%02d_%s" % (i, key)] = float(row[key])
        if row["virtual_rank"] is not None:
            metrics["venue%02d_virtual_rank" % i] = float(row["virtual_rank"])
    for st_ in rep["statistics"]:
        if st_.get("result") is None and "end_reason" not in st_:
            continue
        i = [v["world_id"] for v in venues].index(st_["world_id"]) + 1
        for key in ("touches", "crashes", "pickups", "out_of_bounds", "warnings", "forbidden_calls", "bad_calls",
                    "wall_displacement_m", "program_s", "desk_s", "N_best", "T_best_s", "T_clean_static_s", "T_clean_dynamic_s"):
            v = st_.get(key)
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                metrics["venue%02d_%s" % (i, key)] = float(v)
    for i in infos:
        metrics["venue%02d_wall_s" % i["k"]] = float(i["wall_s"])
        metrics["venue%02d_retried" % i["k"]] = 1.0 if i["retried"] else 0.0
    metrics["runner_runs"] = float(rs["runs"])
    metrics["runner_survivors"] = float(rs["survivors"])
    metrics["runner_processes_killed"] = float(rs["killed"])
    metrics["runner_processes_left"] = float(rs["left"])
    metrics["network_attempts"] = float(rs["network_attempts"])
    metrics["remote_sockets_seen"] = float(max([i["remote_sockets_seen"] for i in infos] or [0]))
    metrics["net_outbound_events"] = float(sum(i["net_outbound"] for i in infos))
    metrics["stray_files_removed"] = float(rs["stray_files"])
    metrics["submission_symlinks_skipped"] = float(rs["symlinks_skipped"])
    metrics["agent_processes_frozen"] = float(len(FREEZER[0].frozen)) if FREEZER[0] is not None else 0.0
    metrics["verify_wall_s"] = round(time.time() - t_start, 1)
    text = ("%s %.4f = gate %d x the mean over the %d venue(s) of 0.35 fast + 0.35 consecutive + 0.30 obstacle test against "
            "the reference program's own numbers; gates: %s%s%s") % (
                METRIC, score["time_trial_score"], score["gate"], len(venues),
                ", ".join("%s %d" % (g, int(v)) for g, v in score["gates"].items()),
                "" if submission_ok else "; submission: " + "; ".join(reasons[:3]),
                "; PLACEHOLDER hidden set (development state)" if hs["placeholder"] else "")
    reward = write_verdict(score["time_trial_score"], metrics, text, expect_reward=score["reward"])
    rn.finish()
    log("[grader] %s %.6f reward %.4f (verify wall %.0f s)" % (METRIC, score["time_trial_score"], reward, metrics["verify_wall_s"]))


if __name__ == "__main__":
    sys.exit(main())
