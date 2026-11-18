"""C6 walk predictivity.

The probe robot walks every kept route of gt/routes.json in the graded twin (units held where submitted, gates
fixed, standard contacts: walks.prepare) under the K agent-side variants of the config; each run is scored against
the STORED true outcome:
    truth reached:      1 if the run reaches (progress >= reach_progress), else 0
    truth not reached:  0 if the run reaches, else band(|progress - progress_true|, C6.progress_band)
A route's raw score is the mean over its variants. With C6.unit_routes_need
"paired", a route whose deciding units (routes.json feature.units) are named scores its raw score only when every one
of them is PAIRED in the matching (ctx.matching.t2a) with an agent unit that takes part in the walks (status ok: an
excluded unit is deleted from the walk model), else 0: a route outcome is a statement about a thing, and a twin
without that thing has said nothing about it. How well the unit is modelled is C1's and C2's question (no true
positive is asked for), whether the robot's run ends as on the site is this criterion's. Routes
with no deciding unit (the building, the gates, open floor) are scored as they are; "none" scores every route raw. A
deciding unit that is not a counted true unit is a TruthError when the routes are loaded. BA = 0.5 (mean over the
"open" routes + mean over the "blocked" routes); the ungated value is max(0, 2 BA - 1) (the orchestrator multiplies it
by C1).

Self-test (the true twin graded with ctx.selftest): every kept route is walked ONCE at the nominal start; the run must
reproduce the stored outcome and progress (sec.selftest_progress_tol), and the route condition must hold on every
route (every deciding unit paired with the true twin's own unit); any mismatch is a TruthError (a broken
environment: another MuJoCo or numpy build, a changed instrument, a stale truth), never a silent change of the route
set.

No outcome comes from the wall clock: a run that passes its wall guard, or a walk budget that
runs out, raises walks.WalkTimeError, which the orchestrator turns into a task error (the episode is void), in the
self-test and for a submission alike.
"""
import math
import os
import sys

from . import walks
from .truth import TruthError

FORMAT = "site-routes/1"


def _log(msg):
    sys.stderr.write("[grader] C6 %s\n" % msg)
    sys.stderr.flush()


def _robot_dir(ctx):
    return os.path.join(ctx.payload_dir, "robot")


def load_routes(ctx):
    """The kept routes, checked against the instrument and the variant set they were built for (TruthError)."""
    doc = ctx.truth.load("routes")
    sec = ctx.sec
    if not isinstance(doc, dict) or doc.get("format") != FORMAT:
        raise TruthError("routes.json: format must be %r" % FORMAT)
    rd = _robot_dir(ctx)
    if not os.path.isfile(os.path.join(rd, "walk_probe.py")):
        raise TruthError("the walk-probe instrument is missing from the payload (%s)" % rd)
    have = walks.instrument_hash(rd)
    want = (doc.get("instrument") or {}).get("sha256")
    if have != want:
        raise TruthError("routes.json was built with instrument %s, the payload has %s" % (str(want)[:16], have[:16]))
    built = (doc.get("instrument") or {}).get("agent_variants") or []
    have_v = walks.make_variants(sec)
    if built[:len(have_v)] != have_v:
        # the oracle scores exactly 1 only if its own variants are in the truth's variant set (a prefix is fine:
        # the variant draws are sequential, so a smaller K gives the first K of the same list)
        raise TruthError("routes.json was built for other agent-side variants than the config's (rebuild the routes)")
    routes = doc.get("routes") or []
    groups = {"open": 0, "blocked": 0}
    for r in routes:
        if r.get("group") not in groups or r.get("outcome") not in ("reached", "not_reached"):
            raise TruthError("routes.json: route %s has no valid group / outcome" % r.get("id"))
        if (r["group"] == "open") != (r["outcome"] == "reached"):
            raise TruthError("routes.json: route %s: group %s does not match outcome %s" % (r["id"], r["group"],
                                                                                           r["outcome"]))
        groups[r["group"]] += 1
        units = deciding_units(r)
        if units:
            # the deciding units must be counted true units (only those have counted pairs)
            bad = [u for u in units if u not in ctx.truth.unit or ctx.truth.unit[u].ignore]
            if bad:
                raise TruthError("routes.json: route %s names deciding units that are not counted true units: %s" % (
                    r["id"], ", ".join(map(str, bad[:6]))))
    need = int(sec["min_routes_per_group"])
    if min(groups.values()) < need:
        raise TruthError("routes.json keeps %d open and %d blocked routes (at least %d each)" % (
            groups["open"], groups["blocked"], need))
    return doc, routes


def walk_twin(ctx, routes, variants, deadline=None):
    """Prepare the graded twin, map the routes, run every (route, variant): (mapped routes, {key: result}, stats)."""
    sec = ctx.sec
    rd = _robot_dir(ctx)
    WP = walks.instrument(rd)
    spec = walks.prepare(ctx.twin, sec)
    rm = walks.ray_model(spec, WP)
    mapped = [walks.map_route(r, ctx.align.map, rm, sec, WP) for r in routes]
    jobs, pre = walks.plan(mapped, variants, sec, WP, rm)
    res, stats = walks.run({"twin": spec}, jobs, sec, rd, deadline=deadline, log=_log)
    res.update(pre)
    stats["pre_decided"] = len(pre)
    return mapped, res, stats


def _gate_value(ctx):
    g = ((ctx.cfg.get("criteria") or {}).get(ctx.name or "C6") or {}).get("gate")
    r = ctx.results.get(g) if g else None
    return None if r is None else float(r.get("value", 0.0))


def deciding_units(route):
    """The route's deciding units (routes.json feature.units: true unit ids), [] when the building, a gate or open
    floor decides it."""
    units = (route.get("feature") or {}).get("units") or []
    if not isinstance(units, list) or not all(isinstance(u, str) for u in units):
        raise TruthError("routes.json: route %s: feature.units must be a list of unit ids" % route.get("id"))
    return list(units)


def units_paired(ctx):
    """{true unit id: agent body} of the counted pairs of the matching whose agent unit takes part in the walks (status
    ok; walks.prepare deletes the excluded units). A true unit without such a pair is missing from the twin."""
    M = getattr(ctx, "matching", None)
    if M is None:
        return {}
    twin = getattr(ctx, "twin", None)
    excluded = set(twin.excluded()) if twin is not None else set()
    return {str(t): a for t, a in sorted(M.t2a.items()) if a not in excluded}


def score(ctx):
    sec = ctx.sec
    _doc, routes = load_routes(ctx)
    gv = _gate_value(ctx)
    if sec.get("skip_when_gate_zero") and not ctx.selftest and gv == 0.0:
        # the orchestrator multiplies by the gate (C1): the walks cannot change a zero; skip their cost
        return {"value": 0.0, "detail": {"skipped": "the gate criterion is 0", "n_routes": len(routes)},
                "diagnostic": "walks skipped: C1 = 0 makes C6 0"}
    variants = walks.make_variants(sec)
    if ctx.selftest:
        variants = variants[:1]                  # the true twin: once, at the nominal start
    mapped, res, stats = walk_twin(ctx, routes, variants, ctx.deadline)
    need = sec["unit_routes_need"]
    paired = units_paired(ctx) if need == "paired" else {}
    per_route = []
    groups = {"open": [], "blocked": []}
    not_run = {h: 0 for h in walks.NOT_RUN}
    unheld = []
    for ri, (r, m) in enumerate(zip(routes, mapped)):
        truth = {"outcome": r["outcome"], "progress": float(r["progress"])}
        runs = []
        for vi in range(len(variants)):
            x = res[(ri, vi)]
            v = walks.variant_score(x, truth, sec)
            row = walks.public(x)
            row.update({"variant": vi, "reached": walks.reached(x, sec), "score": v})
            if x.get("how") in not_run:
                not_run[x["how"]] += 1
            runs.append(row)
        raw = sum(rw["score"] for rw in runs) / len(runs)
        units = deciding_units(r)
        units_pair = {u: paired.get(u) for u in units} if need == "paired" else None
        held = all(v is not None for v in units_pair.values()) if need == "paired" else None   # vacuous: no units
        s = raw if held is not False else 0.0
        if held is False:
            unheld.append(r["id"])
        groups[r["group"]].append(s)
        per_route.append({"id": r["id"], "group": r["group"], "feature": r.get("feature"),
                          "true_outcome": r["outcome"], "true_progress": truth["progress"],
                          "start": m["start"], "score_raw": raw, "deciding_units": units, "units_paired": units_pair,
                          "condition": held, "score": s, "runs": runs})
    if ctx.selftest and unheld:
        raise TruthError("the true twin does not have every deciding unit of these routes paired with its own unit"
                         ": %s" % ", ".join(unheld[:6]))
    mean = {g: (sum(v) / len(v) if v else float("nan")) for g, v in groups.items()}
    if not all(math.isfinite(x) for x in mean.values()):
        raise TruthError("C6: a route group is empty")
    ba = 0.5 * (mean["open"] + mean["blocked"])
    value = max(0.0, 2.0 * ba - 1.0)
    lost = [p["id"] for p in per_route if p["score"] < 1.0]
    diag = "BA %.3f (open %.3f, blocked %.3f) over %d routes x %d variants" % (ba, mean["open"], mean["blocked"],
                                                                            len(routes), len(variants))
    if unheld:
        diag += "; %d route(s) scored 0: a deciding unit is missing from the twin (not paired)" % len(unheld)
    if lost:
        diag += "; below 1: %s" % ", ".join(lost[:6]) + (" +%d" % (len(lost) - 6) if len(lost) > 6 else "")
    if not_run["error"]:
        diag += "; %d runs broke the walk harness (not reached)" % not_run["error"]
    # the detail is deterministic (byte-identical grades): the pool size and the wall clock go to stderr only
    _log("timing: %s" % {k: (round(v, 2) if isinstance(v, float) else v) for k, v in sorted(stats.items())})
    detail = {"BA": ba, "open_mean": mean["open"], "blocked_mean": mean["blocked"], "n_routes": len(routes),
              "n_variants": len(variants), "selftest_nominal_only": bool(ctx.selftest), "not_run": not_run,
              "unit_routes_need": need, "routes_without_their_units": unheld,
              "routes": per_route, "runs": {"jobs": stats.get("jobs"), "pre_decided": stats.get("pre_decided")}}
    return {"value": value, "detail": detail, "diagnostic": diag[:240], "_stats": stats}


def selftest(ctx, result):
    """The true twin's nominal runs reproduce the stored outcome and progress of every kept route (TruthError)."""
    if result is None or result.get("failed"):
        raise TruthError("C6 did not run on the true twin")
    sec = ctx.sec
    tol = float(sec["selftest_progress_tol"])
    bad = []
    rows = result["detail"]["routes"]
    for p in rows:
        run0 = p["runs"][0]
        same = (run0["reached"] == (p["true_outcome"] == "reached")) and (
            abs(float(run0["progress"]) - float(p["true_progress"])) <= tol)
        if not same or run0.get("how") in walks.NOT_RUN:
            bad.append("%s: stored %s %.3f, now %s %.3f (%s)" % (p["id"], p["true_outcome"], p["true_progress"],
                                                                 run0.get("outcome"), float(run0["progress"]),
                                                                 run0.get("reason")))
    if bad:
        raise TruthError("the true twin does not reproduce its stored walks: " + "; ".join(bad[:6]))
    unheld = [p["id"] for p in rows if p.get("condition") is False]
    if unheld:
        raise TruthError("the true twin does not have every deciding unit of these routes paired with its own unit"
                         ": %s" % ", ".join(unheld[:6]))
    return {"routes": len(rows), "nominal_reproduced": len(rows),
            "unit_routes_condition_held": sum(1 for p in rows if p.get("deciding_units") and p.get("condition"))}
