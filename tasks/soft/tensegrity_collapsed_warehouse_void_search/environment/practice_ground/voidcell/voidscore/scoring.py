"""The score of one void-search episode from the cell's own truth record (the rule: tests/README.md, Scoring; hidden side only).

Input: the episode's record (voidcell.server.VoidCellServer.finalize(): `reports` with the true position at each
report, the 10 Hz truth log `log10` [t, x, y, heading, busy], the executor's `executor_history` (every finished
command with t_start, t_end, aborted), the accepted `commands`, the `end` state) and the site's own search points and
pad. Nothing the program sends is an input except its reports (when) and what the robot physically did. Every
threshold is read from acceptance.yaml (`load_acceptance`); no number of it is spelled here.

    acc = load_acceptance()
    sc = score_record(rec, site, acc)       # {"S", "F", "c", "points", "retrieval", ...}
    server.scorer = scorer_for(site, acc)   # the cell calls it from finalize(): rec["score"], rec["episode_score"]

Rules (acceptance.yaml's header states them in words):
  * a command (primitive or set_targets) executes from its acceptance to its end; one still running at the episode's
    end executes to +inf. The window [t - W, t] of a report at t is clear when no command interval (t_start, t_end)
    meets it: t_start < t and t_end > t - W, or the report found a command running (`busy`: a command accepted at the
    same instant before the report);
  * the window must lie inside the episode (t >= W) and every 10 Hz truth sample of it must be within the still
    tolerance of the position at t;
  * the last accepted report of a point counts; a refused report (beyond the per-point budget) changes nothing;
  * F needs the end position within the retrieval radius of the pad and no command executing at the end.
The end reason (finish, the program's exit, the time limit, a budget, a protocol breach) does not change the rules:
the reports made until then count.
"""

import hashlib
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ACCEPTANCE = os.path.join(HERE, "acceptance.yaml")
POINTS = (1, 2, 3)
COMMAND_KINDS = ("primitive", "set_targets")

__all__ = ["ACCEPTANCE", "load_acceptance", "credit", "command_intervals", "dwell", "score_record", "scorer_for",
           "check_cell_constants", "check_site", "AcceptanceError"]


class AcceptanceError(RuntimeError):
    """The acceptance file, a site or the cell's announced constants disagree: the task's fault, never a score."""


def load_acceptance(path=ACCEPTANCE):
    """acceptance.yaml as a dict, with its sha256 under '_sha256' and its path under '_path'."""
    import yaml
    with open(path, "rb") as fh:
        raw = fh.read()
    acc = yaml.safe_load(raw.decode("utf-8"))
    if not isinstance(acc, dict) or acc.get("version") != 2:
        raise AcceptanceError("acceptance file %s: not version 2" % path)
    w = acc["weights"]
    acc["weights"] = {int(k): float(v) for k, v in w.items()}
    if sorted(acc["weights"]) != list(POINTS):
        raise AcceptanceError("acceptance file: weights must name the points %s" % (POINTS,))
    acc["_sha256"] = hashlib.sha256(raw).hexdigest()
    acc["_path"] = os.path.abspath(path)
    return acc


def credit(d, acc):
    """clip(1 - (d - full) / (zero - full), 0, 1)."""
    full, zero = float(acc["credit_radius_m"]["full"]), float(acc["credit_radius_m"]["zero"])
    return min(1.0, max(0.0, 1.0 - (float(d) - full) / (zero - full)))


def command_intervals(rec):
    """[(t_start, t_end, kind, aborted)] of every command that executed: the executor's finished commands in order,
    plus the one still running at the end (accepted, not in the history) with t_end = +inf."""
    hist = [h for h in (rec.get("executor_history") or []) if h.get("kind") in COMMAND_KINDS]
    out = [(float(h["t_start"]), float(h["t_end"]), str(h["kind"]), bool(h.get("aborted"))) for h in hist]
    acc_cmds = [c for c in (rec.get("commands") or []) if c.get("kind") in COMMAND_KINDS and c.get("accepted")]
    if len(acc_cmds) > len(hist):
        if len(acc_cmds) != len(hist) + 1:
            raise AcceptanceError("record: %d accepted commands but %d finished ones (at most one may run at the end)"
                                  % (len(acc_cmds), len(hist)))
        last = acc_cmds[-1]
        out.append((float(last["t"]), math.inf, str(last["kind"]), False))
    return out


def dwell(t, xy, busy, intervals, log10, acc):
    """(valid, reasons, max_dev_m) of the dwell before a report at time t with true position xy."""
    W = float(acc["dwell"]["window_s"])
    tol = float(acc["dwell"]["still_tolerance_m"])
    et, el = float(acc["eps"]["time_s"]), float(acc["eps"]["length_m"])
    reasons = []
    if t < W - et:
        reasons.append("window_before_episode_start")
    if busy or any(t0 < t - et and t1 > t - W + et for (t0, t1, _k, _a) in intervals):
        reasons.append("command_in_window")
    dev = 0.0
    n = 0
    for row in log10:
        ts = float(row[0])
        if t - W - et <= ts <= t + et:
            n += 1
            dev = max(dev, math.hypot(float(row[1]) - xy[0], float(row[2]) - xy[1]))
    if n == 0:
        reasons.append("no_truth_samples")
    if dev > tol + el:
        reasons.append("moved_in_window")
    return (not reasons), reasons, dev


def check_site(site, acc):
    """The site's search points carry the acceptance weights (the generator writes them; a mismatch is a task
    error)."""
    sp = {int(p["id"]): p for p in site["search_points"]}
    if sorted(sp) != list(POINTS):
        raise AcceptanceError("site: search points %s, not %s" % (sorted(sp), POINTS))
    for i in POINTS:
        if abs(float(sp[i]["weight"]) - acc["weights"][i]) > 1e-9:
            raise AcceptanceError("site: point %d weight %s differs from the acceptance file's %s" % (
                i, sp[i]["weight"], acc["weights"][i]))
    return sp


def score_record(rec, site, acc):
    """The episode's score: S = F x sum(w_i c_i) with the per-point details (hidden side; the practice debrief
    reports a subset)."""
    sp = check_site(site, acc)
    pad = site["pad"]["xy"]
    intervals = command_intervals(rec)
    log10 = rec.get("log10") or []
    reports = rec.get("reports") or []
    per = {}
    rows = []
    for k, r in enumerate(reports):
        i = int(r["i"])
        row = {"k": k, "t": float(r["t"]), "i": i, "accepted": bool(r["accepted"])}
        if r["accepted"]:
            xy = (float(r["xy"][0]), float(r["xy"][1]))
            valid, why, dev = dwell(row["t"], xy, bool(r.get("busy")), intervals, log10, acc)
            d = math.hypot(xy[0] - float(sp[i]["xy"][0]), xy[1] - float(sp[i]["xy"][1]))
            c = (1.0 if valid else 0.0) * credit(d, acc)
            row.update(d_m=d, dwell_valid=bool(valid), dwell_failed=why, dwell_max_dev_m=dev, c=c)
            per[i] = k
        rows.append(row)
    for row in rows:
        row["counted"] = bool(row["accepted"] and per.get(row["i"]) == row["k"])
    c = {i: (rows[per[i]]["c"] if i in per else 0.0) for i in POINTS}
    end = rec.get("end") or {}
    exy = end.get("xy") or [math.nan, math.nan]
    d_pad = math.hypot(float(exy[0]) - float(pad[0]), float(exy[1]) - float(pad[1]))
    running = bool(end.get("busy")) or any(math.isinf(t1) for (_t0, t1, _k, _a) in intervals)
    rr = float(acc["retrieval"]["radius_m"])
    within = bool(d_pad <= rr + float(acc["eps"]["length_m"]))
    retrieved = bool(within and not running)
    F = float(acc["retrieval"]["factor_retrieved"] if retrieved else acc["retrieval"]["factor_not_retrieved"])
    total = sum(acc["weights"][i] * c[i] for i in POINTS)
    S = F * total
    if not (math.isfinite(S) and 0.0 <= S <= 1.0 + 1e-9):
        raise AcceptanceError("score %r outside [0, 1]" % (S,))
    return {"S": min(1.0, max(0.0, S)), "F": F, "c": {str(i): c[i] for i in POINTS},
            "weighted": {str(i): acc["weights"][i] * c[i] for i in POINTS},
            "points": rows,
            "retrieval": {"retrieved": retrieved, "within_radius": within, "command_running_at_end": running,
                          "d_pad_m": d_pad},
            "end_reason": rec.get("done_reason"), "t_end": end.get("t"),
            "acceptance_version": acc["version"], "acceptance_sha256": acc.get("_sha256")}


def scorer_for(site, acc):
    """The callable the cell's finalize() takes (record -> score dict with 'S')."""
    check_site(site, acc)

    def _score(rec):
        return score_record(rec, site, acc)
    return _score


def check_cell_constants(mission_mod, budgets_mod, acc):
    """The constants the cell announces in mission() and enforces as budgets equal the acceptance file's
    (voidcell/_lib/mission.py and budgets.py carry their own copies). Raises AcceptanceError when one differs."""
    pairs = [("mission.DWELL_S", mission_mod.DWELL_S, acc["dwell"]["window_s"]),
             ("mission.CREDIT_RADIUS_M[0]", mission_mod.CREDIT_RADIUS_M[0], acc["credit_radius_m"]["full"]),
             ("mission.CREDIT_RADIUS_M[1]", mission_mod.CREDIT_RADIUS_M[1], acc["credit_radius_m"]["zero"]),
             ("mission.STILL_TOLERANCE_M", mission_mod.STILL_TOLERANCE_M, acc["dwell"]["still_tolerance_m"]),
             ("mission.RETRIEVAL_RADIUS_M", mission_mod.RETRIEVAL_RADIUS_M, acc["retrieval"]["radius_m"]),
             ("budgets.DEFAULTS.reports_per_point", budgets_mod.DEFAULTS["reports_per_point"], acc["reports_per_point"])]
    bad = ["%s = %r, acceptance %r" % (n, a, b) for n, a, b in pairs if abs(float(a) - float(b)) > 1e-12]
    if bad:
        raise AcceptanceError("the cell's constants differ from the acceptance file: " + "; ".join(bad))
    return [n for n, _a, _b in pairs]
