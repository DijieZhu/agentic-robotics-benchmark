"""Grader part 2: the pipeline one twin goes through, the truth's self-test and the worker run by verify.py.

    truth = truth_mod.Truth(gt, cfg, vocabulary)
    res = grade_twin(folder, truth, cfg, guard)       # JSON-ready: G0, criteria, availability, details, stages
    st = selftest(truth, cfg)                         # the true twin through grade_twin + the reproduction checks

Stages (each under Guard: an alarm at runtime.stage_caps_s[stage] or the time left):
  gate -> g0_settle -> align (G0 needs the camera fit) -> core (solids, matching, zones) -> q -> the criteria in
  criteria.order (one stage each, named C1 .. C7).
A timeout, a MemoryError or an exception in a stage scores that stage's criteria 0 with a diagnostic (gate, g0_settle
and align: G0 = 0; core: every criterion; q: C2 and every q-weighted term through q = 0; a criterion: itself), except
in the stages of runtime.timeout_is_task_error (the walks, C6): no outcome may come from the
wall clock there, so a timeout is a task error (the episode is void), as is a walk run that exceeds its wall guard or
a walk budget that runs out (walks.WalkTimeError). TruthError and twin_gate.TaskError propagate: they are task errors.
A criterion that is not available (no module or no truth file) is "not implemented"; with criteria.require_all it is a
task error at the self-test. Weight changes a criterion asks for come from the truth alone (its module's
reweight(truth, cfg)), so a criterion whose stage fails is weighted as in the self-test. No clock value enters a
result's criteria, diagnostics or stats (the stage seconds stay under "stages", which the verdict leaves out).
"""
import json
import math
import os
import signal
import sys
import time

import numpy as np

from . import align, config, load, part1, raycast, registry
from . import truth as truth_mod
from .truth import TruthError

FORMAT = "site-twin-grade/1"


class StageTimeout(Exception):
    pass


class Guard:
    """Run stages under a wall-clock alarm: min(cap of the stage, time left before the deadline). enabled=False runs
    without alarms (the self-test; the deadline and the caps still reach ctx.deadline, so the walk pool plans its
    budget)."""

    def __init__(self, deadline=None, caps=None, enabled=True):
        self.deadline = deadline
        self.caps = dict(caps or {})
        self.enabled = enabled

    def left(self):
        return float("inf") if self.deadline is None else self.deadline - time.time()

    def run(self, name, fn, *args):
        if not self.enabled:
            return fn(*args)
        stage_cap = float(self.caps.get(name, float("inf")))
        cap = min(stage_cap, self.left())
        if cap <= 0.5:
            raise StageTimeout("no time was left for stage %s before the verify deadline" % name)
        if not math.isfinite(cap):
            return fn(*args)
        # the message names the configured cap only (never a time that depends on the clock)
        why = ("stage %s timed out (its cap of %.0f s)" % (name, stage_cap) if cap >= stage_cap
               else "stage %s ran out of the verify's time" % name)

        def _alarm(_sig, _frm):
            raise StageTimeout(why)

        old = signal.signal(signal.SIGALRM, _alarm)
        signal.setitimer(signal.ITIMER_REAL, cap)
        try:
            return fn(*args)
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0.0)
            signal.signal(signal.SIGALRM, old)


def _err(exc):
    if isinstance(exc, StageTimeout):
        return "timeout: %s" % exc
    if isinstance(exc, MemoryError):
        return "out of memory (the grader's address-space limit) on this twin"
    return "%s: %s" % (type(exc).__name__, str(exc)[:300])


def _peak_mb():
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def _align_stage(tw, truth, cfg):
    try:
        scene = raycast.static_scene(tw)
    except Exception as exc:  # noqa: BLE001 - an environment that cannot be ray-cast: no refinement
        scene = None
        err = _err(exc)
    else:
        err = None
    al = align.align_twin(tw, truth, cfg, scene)
    if err:
        al.info.setdefault("step3", {})["scene_error"] = err
    return al, scene


class WalkTaskError(RuntimeError):
    """A walk stage ran out of wall time (a stage alarm, a run past its wall guard, the walk budget): the verifier was
    too slow, the episode is void. A task error, never a score."""


def _task_error_stages(cfg):
    return set((cfg.get("runtime") or {}).get("timeout_is_task_error", []))


def grade_twin(folder, truth, cfg, guard, record=None, keep=None, selftest=False):
    """G0 and every criterion of one twin folder. record(res) is called after every stage with the partial result;
    keep (a dict) receives the in-memory objects (twin, ctx) for the self-test."""
    from . import walks
    entries = registry.criteria(cfg)
    names = [e.key for e in entries]
    res = {"format": FORMAT, "G0": 0.0, "g0": {}, "criteria": {}, "available": {}, "diagnostics": {},
           "details": {}, "stages": {}, "stats": {}, "weights": {e.key: e.weight for e in entries}}
    TG = load.payload()["twin_gate"]
    strict = _task_error_stages(cfg)

    def stage(name, fn, *args):
        t = time.time()
        try:
            out = guard.run(name, fn, *args)
            res["stages"][name] = {"ok": True, "s": time.time() - t, "peak_rss_mb": _peak_mb()}
            return True, out
        except TG.TaskError:
            raise
        except TruthError:
            raise
        except walks.WalkTimeError as exc:
            raise WalkTaskError("stage %s: %s" % (name, exc))
        except StageTimeout as exc:
            if name in strict:
                raise WalkTaskError("stage %s: %s (no outcome may come from the wall clock)" % (name, exc))
            res["stages"][name] = {"ok": False, "s": time.time() - t, "error": _err(exc), "peak_rss_mb": _peak_mb()}
            return False, None
        except BaseException as exc:  # noqa: BLE001 - the twin's fault or the grader's limits: this stage scores 0
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            res["stages"][name] = {"ok": False, "s": time.time() - t, "error": _err(exc), "peak_rss_mb": _peak_mb()}
            return False, None
        finally:
            if record:
                record(res)

    def zero_from(keys, why):
        for c in keys:
            res["criteria"].setdefault(c, 0.0)
            res["diagnostics"].setdefault(c, why)

    for e in entries:
        ok, why = registry.available(e, truth)
        res["available"][e.key] = {"ok": ok, "why": why}
    # G0: the gate
    ok, tw = stage("gate", load.load_twin, folder, truth.vocabulary)
    if not ok:
        res["g0"] = {"gate": False, "reasons": [res["stages"]["gate"]["error"]]}
        zero_from(names, "G0: " + res["stages"]["gate"]["error"])
        return res
    res["g0"] = {"gate": bool(tw.ok), "reasons": tw.reasons[:20], "warnings": tw.report.get("warnings", [])[:20],
                 "excluded": tw.excluded(), "n_units": tw.n_units,
                 "n_photo_cameras_valid": tw.report.get("n_photo_cameras_valid", 0)}
    res["stats"].update({"n_units": float(tw.n_units), "n_excluded": float(len(tw.excluded())),
                         "n_photo_cameras": float(tw.report.get("n_photo_cameras_valid", 0))})
    if keep is not None:
        keep["twin"] = tw
    if not tw.ok:
        zero_from(names, "G0 gate: " + "; ".join(tw.reasons[:4]))
        return res
    # G0: the settle check
    ok, chk = stage("g0_settle", load.g0_settle, tw)
    if not ok:
        res["g0"]["settle"] = {"fail": True, "reasons": [res["stages"]["g0_settle"]["error"]]}
        zero_from(names, "G0 settle check: " + res["stages"]["g0_settle"]["error"])
        return res
    res["g0"]["settle"] = {k: chk.get(k) for k in ("fail", "reasons", "max_energy_gain_J", "energy_bound_J",
                                                    "max_speed_excess_m_s", "max_speed_excess_item", "fail_time_s",
                                                    "fail_phase", "excluded", "exclusion_reasons")}
    if chk["fail"]:
        zero_from(names, "G0 settle check: " + "; ".join(chk["reasons"][:3]))
        return res
    if chk.get("excluded"):
        res["g0"]["excluded"] = tw.excluded()
        res["g0"]["excluded_by_settle"] = list(chk["excluded"])
        res["stats"]["n_excluded"] = float(len(tw.excluded()))
        res["stats"]["n_excluded_by_settle"] = float(len(chk["excluded"]))
    # G0: the camera fit
    ok, out = stage("align", _align_stage, tw, truth, cfg)
    if not ok:
        res["g0"]["align"] = {"fail": res["stages"]["align"]["error"]}
        zero_from(names, "G0 camera fit: " + res["stages"]["align"]["error"])
        return res
    al, scene = out
    res["details"]["alignment"] = al.summary()
    if al.fail:
        res["g0"]["align"] = {"fail": al.fail}
        zero_from(names, "G0 camera fit: " + al.fail)
        return res
    res["G0"] = 1.0
    res["stats"].update({"s_star": float(al.s), "up_tilt_deg": float(al.up_tilt_deg),
                         "refined": 1.0 if al.refined else 0.0, "n_camera_inliers": float(len(al.inliers))})
    record and record(res)
    ctx = registry.Ctx(cfg, truth, tw, selftest=selftest)
    ctx.align = al
    if scene is not None:
        ctx._cache["_static_scene"] = scene
    if keep is not None:
        keep["ctx"] = ctx
    # the core
    ok, cinfo = stage("core", part1.core, ctx)
    if not ok:
        zero_from(names, "the matching stage failed: " + res["stages"]["core"]["error"])
        return res
    res["details"]["matching"] = ctx.matching.summary()
    res["details"]["zones"] = ctx.zones.summary()
    res["stats"].update({"n_pairs": float(cinfo["n_pairs"]), "n_removed_ignore": float(cinfo["n_removed"])})
    ok, _qi = stage("q", part1.q_stage, ctx)
    if not ok:
        ctx.q, ctx.q_detail = {}, {}
        res["details"]["q_error"] = res["stages"]["q"]["error"]
    # the criteria
    for e in entries:
        if not res["available"][e.key]["ok"]:
            res["criteria"][e.key] = 0.0
            res["diagnostics"][e.key] = res["available"][e.key]["why"]
            continue
        ctx.deadline = None if guard.deadline is None else min(guard.deadline, time.time() + float(
            guard.caps.get(e.key, float("inf"))))
        ok, r = stage(e.key, registry.run_criterion, e, ctx)
        if not ok:
            res["criteria"][e.key] = 0.0
            res["diagnostics"][e.key] = "%s failed: %s" % (e.key, res["stages"][e.key]["error"])
            ctx.results[e.key] = {"value": 0.0, "ungated": 0.0, "detail": {}, "failed": True}
            continue
        if isinstance(r.get("timing"), dict):
            res["stages"][e.key]["timing"] = {k: v for k, v in sorted(r.pop("timing").items())
                                              if isinstance(v, (int, float)) and not isinstance(v, bool)}
        else:
            r.pop("timing", None)
        ung = r["value"]
        if e.gate is not None:
            gv = float(res["criteria"].get(e.gate, 0.0))
            r["value"] = ung * gv
            r["gate"] = {"by": e.gate, "value": gv}
        r["ungated"] = ung
        ctx.results[e.key] = r
        res["criteria"][e.key] = float(r["value"])
        res["details"][e.key] = {"detail": r["detail"], "ungated": ung, "gate": r.get("gate")}
        if r["diagnostic"]:
            res["diagnostics"][e.key] = r["diagnostic"]
        if r["reweight"]:
            res["details"][e.key]["reweight"] = r["reweight"]
        record and record(res)
    res["weights"] = weights_after_reweight(entries, truth, cfg, res, ctx.results)
    return res


def weights_after_reweight(entries, truth, cfg, res, results):
    """The criterion weights after the weight changes the criteria ask for. A module with reweight(truth, cfg) is asked
    directly (the change depends on the truth only, so it holds even when the criterion's stage failed: request CP-R9);
    otherwise the "reweight" of its result is used. The deltas must sum to 0 and keep every weight >= 0."""
    w = {e.key: e.weight for e in entries}
    for e in entries:
        if not res["available"].get(e.key, {}).get("ok"):
            continue
        mod = registry.module(e)
        if callable(getattr(mod, "reweight", None)):
            rw = mod.reweight(truth, cfg) or {}
            got = (results.get(e.key) or {}).get("reweight") or {}
            if got and got != rw:
                raise TruthError("criterion %s asks for the weight change %s in its result and %s from the truth" % (
                    e.key, got, rw))
        else:
            rw = (results.get(e.key) or {}).get("reweight") or {}
        if rw and abs(sum(float(x) for x in rw.values())) > 1e-12:
            raise TruthError("criterion %s: reweight deltas must sum to 0: %s" % (e.key, rw))
        for k, dv in sorted(rw.items()):
            if k in w:
                w[k] += float(dv)
    if any(v < -1e-12 for v in w.values()):
        raise TruthError("a criterion's reweight makes a weight negative: %s" % w)
    return {k: max(0.0, v) for k, v in w.items()}


def twin_score(res, cfg, g1=1.0):
    """G1 x G0 x the weighted mean of the criteria; with criteria.require_all false, the criteria that are not
    available are left out of the weights (dev only)."""
    w = res.get("weights") or {e.key: e.weight for e in registry.criteria(cfg)}
    keys = [k for k in w if res.get("available", {}).get(k, {}).get("ok", True)]
    tot = sum(float(w[k]) for k in keys)
    s = sum(float(w[k]) * float(res["criteria"].get(k, 0.0)) for k in keys) / tot if tot > 0 else 0.0
    return float(g1) * float(res.get("G0", 0.0)) * s


# ---------------------------------------------------------------------------------------------------------------
# the self-test of the truth

def selftest(truth, cfg, deadline=None, caps=None):
    """The true twin through grade_twin (the same code as every submission) and the reproduction checks. Raises
    TruthError on any failure. No stage alarms; `deadline` (wall clock) and the stage caps reach the criteria as
    ctx.deadline, so the walk pool plans its budget (and a walk budget that runs out is a task error)."""
    t0 = time.time()
    keep = {}
    caps = dict(caps if caps is not None else (cfg.get("runtime") or {}).get("stage_caps_s", {}))
    res = grade_twin(truth.paths["twin"], truth, cfg, Guard(deadline, caps, enabled=False), keep=keep, selftest=True)
    if res["G0"] != 1.0:
        raise TruthError("the true twin fails G0: %s" % {k: v for k, v in res["diagnostics"].items()})
    if res["g0"].get("excluded"):
        raise TruthError("the gate excludes true units: %s" % res["g0"]["excluded"])
    ctx = keep["ctx"]
    al = ctx.align
    if not (al.map.is_identity() and al.s == 1.0):
        raise TruthError("the true twin's alignment is not the identity: %s" % al.map.as_dict())
    unavailable = {k: v["why"] for k, v in res["available"].items() if not v["ok"]}
    if unavailable and cfg["criteria"]["require_all"]:
        raise TruthError("criteria not available: %s" % unavailable)
    bad = {k: res["criteria"].get(k) for k, v in res["available"].items() if v["ok"] and res["criteria"].get(k) != 1.0}
    if bad:
        raise TruthError("the true twin does not score 1 in every criterion: %s; %s" % (
            bad, {k: res["diagnostics"].get(k) for k in bad}))
    out = {"criteria": dict(res["criteria"]), "available": {k: v["ok"] for k, v in res["available"].items()},
           "cameras": part1.check_cameras(truth, cfg), "zones_checked": part1.check_zones(truth, ctx),
           "surface_max_err_m": part1.check_surfaces(truth, ctx, cfg), "c3": part1.check_c3(truth, cfg)}
    if truth.exists("sfm_selftest"):
        out["sfm"] = align.selftest_sfm(truth, cfg)
    elif cfg["criteria"]["require_all"]:
        raise TruthError("the SfM-frame alignment self-test file (gt sfm_selftest) is missing")
    for e in registry.criteria(cfg):
        if not res["available"][e.key]["ok"]:
            continue
        mod = registry.module(e)
        if hasattr(mod, "selftest"):
            ctx.name, ctx.sec = e.key, cfg.get(e.key, {})
            out["module_" + e.key] = mod.selftest(ctx, ctx.results.get(e.key))
    out["stages"] = res["stages"]
    out["seconds"] = time.time() - t0
    return out


# ---------------------------------------------------------------------------------------------------------------
# the worker (a separate process started by verify.py)

def _clean(x):
    """JSON-safe copy: numpy scalars and arrays to Python, non-finite floats to None, tuple keys joined."""
    if isinstance(x, dict):
        return {("|".join(map(str, k)) if isinstance(k, tuple) else str(k)): _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set)):
        return [_clean(v) for v in (sorted(x) if isinstance(x, set) else x)]
    if isinstance(x, np.ndarray):
        return _clean(x.tolist())
    if isinstance(x, (np.floating, float)):
        f = float(x)
        return f if math.isfinite(f) else None
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return x


class Progress:
    def __init__(self, path):
        self.path = path
        self.doc = {"format": "site-twin-worker-progress/1", "selftest": None, "submission": None, "done": False}

    def write(self):
        self.doc["peak_rss_mb"] = _peak_mb()
        tmp = self.path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(_clean(self.doc), f, allow_nan=False)
        os.replace(tmp, self.path)


def worker_main(job_path):
    """Entry of the worker process: limits itself, builds the truth, self-tests it, grades the submission; every step
    is written to the progress file. Exit 0 when done, 3 on a truth failure."""
    import resource
    with open(job_path) as f:
        job = json.load(f)
    lim = int(job["address_space_mb"]) * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_AS, (lim, lim))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    prog = Progress(job["progress"])
    prog.write()
    deadline = float(job["deadline_wall"])
    caps = job.get("stage_caps_s", {})
    t0 = time.time()
    try:
        st_cap = float(caps.get("selftest", float("inf")))
        st_deadline = min(deadline, t0 + st_cap) if math.isfinite(st_cap) else deadline

        def build():
            cfg = config.load(job["grader_config"])
            truth = truth_mod.Truth(job["gt"], cfg, vocabulary=job["vocabulary"])
            st = selftest(truth, cfg, deadline=st_deadline, caps=caps)
            return cfg, truth, st
        cfg, truth, st = Guard(deadline, {"selftest": st_cap}).run("selftest", build)
        prog.doc["selftest"] = dict(ok=True, truth_build_and_selftest_s=time.time() - t0, **st)
    except BaseException as exc:  # noqa: BLE001 - a truth that cannot be built or checked is a task error
        if isinstance(exc, KeyboardInterrupt):
            raise
        import traceback
        prog.doc["selftest"] = {"ok": False, "reason": _err(exc), "trace": traceback.format_exc()[-1500:],
                                "s": time.time() - t0}
        prog.write()
        return 3
    prog.write()

    def record(res):
        prog.doc["submission"] = res
        prog.write()

    try:
        res = grade_twin(job["submission"], truth, cfg, Guard(deadline, caps), record=record)
    except BaseException as exc:  # noqa: BLE001 - only task errors reach here (vocabulary, truth)
        if isinstance(exc, KeyboardInterrupt):
            raise
        import traceback
        prog.doc["task_error"] = {"reason": _err(exc), "trace": traceback.format_exc()[-1500:]}
        prog.write()
        return 3
    res["twin_score_g1_1"] = twin_score(res, cfg)
    prog.doc["submission"] = res
    prog.doc["done"] = True
    prog.write()
    return 0


if __name__ == "__main__":  # pragma: no cover - verify.py starts the worker through a bootstrap
    sys.exit(worker_main(sys.argv[1]))
