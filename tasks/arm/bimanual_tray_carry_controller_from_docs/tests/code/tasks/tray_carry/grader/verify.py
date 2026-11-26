import ast
import hashlib
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

HERE = os.path.dirname(os.path.abspath(__file__))
STAGE_DIR = os.environ.get("ALE_STAGE_DIR", HERE)
KIT = STAGE_DIR if os.path.isdir(os.path.join(STAGE_DIR, "robotics_grader")) else os.path.join(HERE, "kit88")
for _p in (KIT, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

HOME = os.environ.get("ALE_HOME", "/home/user")
AGENT_USER = os.path.basename(HOME.rstrip("/")) or "user"
SUBMISSION = os.environ.get("ARB_SUBMISSION", os.path.join(HOME, "submission"))
ROOT_PAYLOAD = os.environ.get("ARB_ROOT_PAYLOAD", "/opt/arb")
RUNS_ROOT = os.environ.get("ARB_RUNS_ROOT", os.path.join(ROOT_PAYLOAD, "runs"))
SNAPSHOT = os.path.join(RUNS_ROOT, "_submission_snapshot")
QUARANTINE = os.path.join(RUNS_ROOT, "_quarantine")
STICKY_DIRS = [d for d in ("/tmp", "/dev/shm", "/var/tmp", "/run/lock") if os.path.isdir(d)]
WATCH_ROOTS = STICKY_DIRS + [HOME]
LATE_GATE_MAX_GAP_S = 600.0
HIDDEN = os.path.join(STAGE_DIR, "hidden")
FROZEN_SET = os.environ.get("ARB_FROZEN_SET", os.path.join(HIDDEN, "frozen_set.json"))
REGISTRY = os.environ.get("ARB_REGISTRY", os.path.join(HIDDEN, "registry"))
TOKEN_SHA = os.environ.get("ARB_TOKEN_SHA256_FILE", os.path.join(HIDDEN, "oracle_token.sha256"))
RECORD_ROOT = os.environ.get("ARB_RECORD_ROOT", "/opt/arb-verify")
DEFAULT_CFG = {"format": "stage05-grader-config/1", "runner_account": "nobody", "max_submission_mb": 80,
               "units_side_by_side": 1}
ORACLE_STEP_BUDGET_S = 0.1
ORACLE_INSTRUMENT_ROOT = os.environ.get("ARB_ORACLE_INSTRUMENT_ROOT", "/opt/arb-oracle-instrument")
REF_DIR = os.path.join(os.path.dirname(HERE), "ref")

from arm.assembly_cell import runner
from arm.assembly_six import lockdown
from arm import verify_freeze

FREEZER = [None]


def log(msg):
    print(msg, file=sys.stderr, flush=True)


def _early_lockdown():
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
        if left:
            out["reasons"].append("%d process(es) of the agent account survived the kill at verify start" % left)
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
                    len(out["late"]), "y" if len(out["late"]) == 1 else "ies",
                    ", ".join(e[0] for e in out["late"][:5])))
        os.makedirs(RUNS_ROOT, exist_ok=True)
        os.chmod(RUNS_ROOT, 0o755)
        lockdown.snapshot_dir(SUBMISSION, SNAPSHOT)
        out["submission_snapshot"] = 1.0
    if gate_active:
        q = 0
        for root in WATCH_ROOTS:
            late = lockdown.late_entries(root, t_ref, exclude=(SUBMISSION, RUNS_ROOT, STAGE_DIR))
            if late:
                log("[grader] %d late entr%s under %s quarantined" % (len(late), "y" if len(late) == 1 else "ies",
                                                                       root))
                q += lockdown.quarantine(root, late, QUARANTINE)
        out["late_writes_quarantined"] = float(q)
    out["stage_lock_latency_ms"] = lockdown.process_age_ms()
    return out


EARLY = _early_lockdown() if __name__ == "__main__" else {"saved": None, "parent": None, "reasons": [], "late": [],
                                                          "t_ref": None}

from arm.assembly_info import netaudit


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_json(p):
    with open(p) as f:
        return json.load(f)


def config():
    p = os.path.join(STAGE_DIR, "grader_config.json")
    cfg = dict(DEFAULT_CFG)
    if os.path.isfile(p):
        cfg.update(load_json(p))
    if os.environ.get("ARB_UNITS_SIDE_BY_SIDE"):
        cfg["units_side_by_side"] = int(os.environ["ARB_UNITS_SIDE_BY_SIDE"])
    return cfg


def program_source(submission=None):
    sub = submission or SUBMISSION
    src = SNAPSHOT if (submission is None and os.getuid() == 0 and os.path.isdir(SNAPSHOT)) else sub
    return os.path.join(src, "controller")


def tree_size(root):
    total = 0
    for dp, _, fns in os.walk(root):
        for fn in fns:
            try:
                total += os.lstat(os.path.join(dp, fn)).st_size
            except OSError:
                pass
    return total


def static_check(cfg, src):
    out = {"controller_file": False, "size_cap": None, "static_import_audit": None, "why": None, "network_hits": []}
    path = os.path.join(src, "controller.py")
    if not os.path.isfile(path) or os.path.islink(path):
        out["why"] = "no controller/controller.py (regular file) in the submission"
        return out
    try:
        tree = ast.parse(open(path, encoding="utf-8", errors="replace").read())
    except SyntaxError as e:
        out["why"] = "controller.py does not parse: %s" % e
        return out
    has = any(isinstance(n, ast.ClassDef) and n.name == "Controller" for n in ast.walk(tree)) or any(
        isinstance(n, (ast.Import, ast.ImportFrom)) and any((a.asname or a.name) == "Controller" for a in n.names)
        for n in ast.walk(tree))
    if not has:
        out["why"] = "controller.py defines no class Controller"
        return out
    out["controller_file"] = True
    out["size_cap"] = tree_size(src) <= float(cfg.get("max_submission_mb", 80)) * 1024 * 1024
    if not out["size_cap"]:
        out["why"] = "the controller folder is larger than %s MB" % cfg.get("max_submission_mb", 80)
    net = netaudit.scan_imports(src)
    out["network_hits"] = [list(x) for x in net]
    out["static_import_audit"] = not net
    if net and out["why"] is None:
        out["why"] = "static import audit: %s" % ", ".join("%s: %s" % tuple(x) for x in net[:5])
    return out


def token_state(src):
    return _reference_controller().token_state(src, TOKEN_SHA)


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


def _reference_controller():
    mod = sys.modules.get("reference_controller")
    if mod is not None:
        return mod
    import importlib.util
    path = os.path.join(REF_DIR, "reference_controller.py")
    spec = importlib.util.spec_from_file_location("reference_controller", path)
    if spec is None or not os.path.isfile(path):
        raise FileNotFoundError("the token path's module is missing: " + path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["reference_controller"] = mod
    try:
        spec.loader.exec_module(mod)
    except BaseException:
        sys.modules.pop("reference_controller", None)
        raise
    return mod


def truth_for(ctx, unit, ref_dir=None):
    RC = _reference_controller()
    rec = RC.recipe_of_registry_unit(ctx.reg, ctx.label, cell=ctx.cell)
    truth = RC.truth_payload(rec, ctx.site, ctx.slots, ctx.jobs, ref_dir=ref_dir)
    return truth, "reference_controller.truth_payload"


def grade_one(args):
    import common as C
    import gtrace as TR
    import score as SC
    import units as UN
    import runtime as RT
    t0, c0 = time.perf_counter(), time.process_time()
    ctx = UN.UnitContext(args["registry"], args["label"])
    u = ctx.build()
    rec_pub, site_pub, pub_src = ctx.public(u)
    truth, tsrc = (None, "not privileged")
    budget = None
    if args.get("privileged"):
        truth, tsrc = truth_for(ctx, u, ref_dir=args.get("oracle_ref_dir"))
        if truth is not None:
            budget = ORACLE_STEP_BUDGET_S
    tr = TR.GradeTrace()
    rec = RT.run_unit(u, ctx.site, ctx.slots, ctx.jobs, args["program"], args["run_root"], rec_pub, site_pub,
                      truth=truth, recorder=tr, runner_user=args.get("runner_user", "nobody"), step_budget_s=budget)
    t_run = time.perf_counter() - t0
    g = SC.grade_unit(rec, tr, ctx.slots, ctx.jobs, ctx.site, args["form"], check_events=args.get("check_events",
                                                                                                    True))
    if args.get("keep_trace"):
        tr.save(args["keep_trace"])
    rec_small = {k: v for k, v in rec.items() if k != "shift"}
    rec_small["shift_summary"] = None if rec.get("shift") is None else {
        k: rec["shift"].get(k) for k in ("cycles", "planned_cycles", "stopped", "events", "tray_in_work",
                                         "state_digest", "motion_digest", "presentations", "sim_s",
                                         "load_over_capacity_max", "winding_margin_min_K")}
    out = {"label": ctx.label, "identity": ctx.identity(), "public_data": pub_src, "truth": tsrc,
           "oracle_run": budget is not None,
           "instrument_dir": None if truth is None else truth["code"]["ref"],
           "step_budget": rec.get("step_budget") or {"rule": "printed" if budget is None else "oracle",
                                                     "step_budget_cpu_s": budget},
           "record": rec_small, "grade": g, "run_wall_s": t_run, "wall_s": time.perf_counter() - t0,
           "grader_cpu_s": time.process_time() - c0, "trace_steps": len(tr.arrays()["step"])}
    return C.plain(out)


def grade_submission(program, frozen_path=FROZEN_SET, registry=REGISTRY, cfg=None, privileged=False, run_root=None,
                     static=None, side_by_side=None, check_events=True, keep_traces=None, require_kind=None,
                     oracle_copy_root=None):
    import common as C
    import frozen as FZ
    import score as SC
    cfg = dict(DEFAULT_CFG, **(cfg or {}))
    st = static_check(cfg, program) if static is None else static
    fs = FZ.load(frozen_path, registry_root=registry, require_kind=require_kind)
    form = SC.form_of(fs)
    anchor = float(fs["anchor"]["raw"])
    out = {"static": st, "frozen_set_digest": fs["digest"], "frozen_set_kind": fs["kind"], "anchor": anchor,
           "units": [], "oracle_copy": None}
    if not (st["controller_file"] and st["size_cap"] and st["static_import_audit"]):
        g = SC.gates([], dict(st, no_late_write=not EARLY.get("reasons")))
        out.update(gates=g, raw=0.0, reward=0.0, why=st.get("why"))
        return out
    run_root = run_root or os.path.join(RUNS_ROOT, "units")
    oracle_ref_dir = None
    if privileged and oracle_copy_root:
        import oracle_copy as OC
        oc = OC.make(oracle_copy_root, C.TASK_DIR, C.K.UNIT_DIR, forbid_inside=(HERE, STAGE_DIR, HIDDEN, registry))
        oracle_ref_dir = oc["ref_dir"]
        out["oracle_copy"] = {"root": oc["root"], "ref_dir": oc["ref_dir"], "n_files": oc["n_files"],
                              "digest": oc["digest"]}
    jobs = []
    for k, u in enumerate(fs["units"]):
        jobs.append({"label": u["label"], "registry": registry, "program": program,
                     "run_root": os.path.join(run_root, "u%d_%s" % (k + 1, u["label"])), "form": form,
                     "privileged": privileged, "runner_user": cfg.get("runner_account", "nobody"),
                     "check_events": check_events, "oracle_ref_dir": oracle_ref_dir,
                     "keep_trace": None if not keep_traces else os.path.join(keep_traces, u["label"] + ".npz")})
    n_par = int(side_by_side if side_by_side is not None else cfg.get("units_side_by_side", 1))
    if n_par > 1:
        import multiprocessing as mp
        with mp.get_context("spawn").Pool(min(n_par, len(jobs))) as pool:
            res = pool.map(grade_one, jobs)
    else:
        res = [grade_one(j) for j in jobs]
    out["units"] = res
    g = SC.gates([r["record"] for r in res], dict(st, no_late_write=not EARLY.get("reasons")))
    raw, reward = SC.raw_and_reward([r["grade"]["unit_u"] for r in res], g["gates"], anchor)
    out.update(gates=g, raw=raw, reward=reward)
    return C.plain(out)


def metrics_of(result):
    m = {"raw": result.get("raw", 0.0), "anchor": result.get("anchor", 0.0), "gates": result["gates"]["gates"]}
    for k, v in result["gates"]["each"].items():
        if v is not None:
            m["gate_" + k] = 1.0 if v else 0.0
    m["gates_not_run"] = float(sum(1 for v in result["gates"]["each"].values() if v is None))
    for i, u in enumerate(result.get("units", [])):
        g = u["grade"]
        p = "unit%d_" % (i + 1)
        m[p + "unit_u"] = g["unit_u"]
        m[p + "carry_mean"] = g["carry_mean"]
        if g.get("J"):
            m[p + "J"], m[p + "TPR"], m[p + "FPR"] = g["J"]["J"], g["J"]["TPR"], g["J"]["FPR"]
        tr = g.get("trays", [])
        m[p + "completed"] = float(sum(1 for t in tr if t["completed"]))
        m[p + "carry_trays"] = float(len(tr))
        m[p + "V_mean"] = (sum(t["V"] for t in tr) / len(tr)) if tr else 0.0
        rt = u["record"].get("runtime") or {}
        m[p + "on_time_share"] = float(rt.get("on_time_share", 0.0))
        m[p + "late_cycles"] = float(rt.get("late_cycles", 0))
        se = g.get("shift_end") or {}
        m[p + "ended_early"] = 0.0 if se.get("kind") in (None, "end") else 1.0
        ec = g.get("events_check") or {}
        if ec.get("agree") is not None:
            m[p + "events_agree"] = 1.0 if ec["agree"] else 0.0
        m[p + "wall_s"] = u["wall_s"]
        m[p + "oracle_run"] = 1.0 if u.get("oracle_run") else 0.0
        sb = (u.get("step_budget") or {}).get("step_budget_cpu_s")
        if isinstance(sb, (int, float)):
            m[p + "step_budget_s"] = float(sb)
    return {k: float(v) for k, v in m.items() if isinstance(v, (int, float)) and math.isfinite(float(v))}


def write_verdict(result, extra):
    from robotics_grader import ratio_rewards
    from robotics_grader import write_verdict as kit_write_verdict
    metrics = metrics_of(result)
    metrics.update({k: float(v) for k, v in extra.items()})
    anchor = result["anchor"]
    verdict = ratio_rewards(result["raw"], anchor, metric_name="raw_score", extra=metrics, full_at=1.0)
    if abs(verdict["rewards"]["reward"] - result["reward"]) > 1e-12:
        raise RuntimeError("the kit's reward %r differs from the rule's %r" % (verdict["rewards"]["reward"],
                                                                                result["reward"]))
    text = ("gates %.0f (controller.py defines Controller, size cap, static import audit, setup on every unit, a call "
            "answered, >= 90 %% of each unit's cycles on time, no late write, no process left); raw %.4f = gates x the "
            "mean over the three units of unit_u; reward = min(1, raw / anchor %.4f)") % (
        result["gates"]["gates"], result["raw"], anchor)
    not_run = [k for k, v in result["gates"]["each"].items() if v is None]
    if not_run:
        text += "; not run: " + ", ".join(not_run)
    if any(u.get("oracle_run") for u in result.get("units", [])):
        text += ("; THE RUN WAS THE ORACLE'S (a controller whose token is valid), with a step budget of %g s of CPU per "
                 "cycle in place of the printed one") % ORACLE_STEP_BUDGET_S
    if os.environ.get("ALE_VERIFY_PYTHON"):
        kit_write_verdict({"rewards": verdict["rewards"], "metrics": {
            k.replace(".", "_").replace("/", "_"): v for k, v in verdict["metrics"].items()}}, diagnostic=text)
    else:
        kit_write_verdict(verdict, path=os.environ.get("ROBOTICS_GRADER_VERDICT_PATH")
                          or os.path.join(RECORD_ROOT, "verdict.local.json"), diagnostic=text)
    return verdict


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
    t_start = time.time()
    cfg = config()
    extra = {"grader_exception": 0.0, "root_payload_ok": 0.0, "privileged_mode": 0.0, "oracle_token_mismatch": 0.0}
    for k in ("stage_locked", "stage_parent_locked", "stage_lock_latency_ms", "agent_processes_killed",
              "agent_processes_left", "submission_late_writes", "late_writes_quarantined", "late_write_gate_active",
              "submission_snapshot"):
        extra[k] = float(EARLY.get(k, -1.0 if k == "stage_lock_latency_ms" else 0.0))
    try:
        gh = os.path.join(STAGE_DIR, "given_hashes.json")
        if os.path.isfile(gh):
            ok_root, problems = root_payload_ok(load_json(gh))
            extra["root_payload_ok"] = 1.0 if ok_root else 0.0
            if not ok_root:
                log("[grader] TASK ERROR: the root-owned givens are unusable: " + "; ".join(problems[:5]))
                raise SystemExit(1)
        src = program_source()
        privileged, mismatch = token_state(src)
        extra["privileged_mode"] = 1.0 if privileged else 0.0
        extra["oracle_token_mismatch"] = 1.0 if mismatch else 0.0
        st = static_check(cfg, src)
        log("[grader] static check: %s; privileged %s" % (st.get("why") or "ok", privileged))
        kind = None if os.environ.get("ARB_ALLOW_DEVELOPMENT_SET") == "1" else "of_record"
        result = grade_submission(src, cfg=cfg, privileged=privileged, static=st, require_kind=kind,
                                  oracle_copy_root=ORACLE_INSTRUMENT_ROOT if os.getuid() == 0 else None)
        os.makedirs(os.path.join(RECORD_ROOT, "out"), exist_ok=True)
        os.chmod(RECORD_ROOT, 0o700)
        with open(os.path.join(RECORD_ROOT, "out", "units.json"), "w") as f:
            json.dump(result, f, indent=1, default=float)
        extra["verify_wall_s"] = time.time() - t_start
        write_verdict(result, extra)
        log("[grader] raw %.4f anchor %.4f reward %.4f (verify wall %.0f s)" % (
            result["raw"], result["anchor"], result["reward"], extra["verify_wall_s"]))
    except SystemExit:
        raise
    except Exception:
        log("[grader] grader exception\n" + traceback.format_exc())
        raise
    finally:
        shutil.rmtree(RUNS_ROOT, ignore_errors=True) if os.getuid() == 0 else None
        _restore_modes()
        if FREEZER[0] is not None:
            FREEZER[0].stop()


if __name__ == "__main__":
    main()
