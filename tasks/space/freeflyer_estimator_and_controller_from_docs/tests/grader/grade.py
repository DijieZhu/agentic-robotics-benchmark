"""One session and the whole run, from flown session records (records.py) to the reward and its summary.

    res = grade_session(record, cfg, scored_ids)        -> {"score", "terms", "runner"} ; raises on any defect
    out = grade_run(results_by_session, cfg, inside_caps, audit_passes)  -> {"reward", "gates", "metrics"}

Every exception raised here is the grader's own cause (a defective record, configuration or hidden set); the caller
runs both inside the guard, which voids the verification.
"""
from . import measures as ME
from . import records as RC
from . import score as SC


def load_hidden_set(doc, cfg):
    """the hidden set (the acceptance sessions: the world's session, its tight mode, its third-block list, its seed and
    its scored set); a missing or malformed one is the grader's own defect."""
    if not isinstance(doc, dict) or doc.get("format") != "estimator-controller-hidden-set/1":
        raise ValueError("the hidden set is not of the form estimator-controller-hidden-set/1")
    ss = doc.get("sessions")
    names = cfg.sessions()
    if not isinstance(ss, dict) or sorted(ss) != sorted(names):
        raise ValueError("the hidden set does not hold exactly the sessions of sessions.names")
    out = {}
    for n in names:
        e = ss[n]
        for k, t in (("world_session", str), ("tight_mode", str), ("list", int), ("seed", str), ("scored", list)):
            if not isinstance(e.get(k), t) or isinstance(e.get(k), bool):
                raise ValueError("the hidden set's %s has no %s of the right form" % (n, k))
        if not e["scored"] or not all(isinstance(x, str) for x in e["scored"]) or len(set(e["scored"])) != len(e["scored"]):
            raise ValueError("the hidden set's %s has no scored set of distinct job ids" % n)
        out[n] = dict(e)
    return out


def check_recoveries(chk, recovery_s, step_s):
    """monitor.hand_back and the invalid plan's recovery: the built-in law's recovery after a hand-back or an invalid
    plan takes the mode's monitor.recovery_s off the block's budget (the world's record of it: recovery.t_from,
    t_to); a record whose recovery is shorter or longer than that, by more than one step, contradicts the rule the
    world applies and voids the verification."""
    for (bn, jid), r in chk["job_records"].items():
        if r["status"] not in ("handback", "invalid_plan") or r.get("third_handback"):
            continue
        rv = r.get("recovery")
        if not isinstance(rv, dict):
            raise RC.RecordError("job %s: a hand-back or an invalid plan without the world's record of its recovery" % jid)
        R = recovery_s[r.get("mode")] if r.get("mode") in recovery_s else None
        if R is None:
            raise RC.RecordError("job %s: no mode for its recovery" % jid)
        took = float(rv["t_to"]) - float(rv["t_from"])
        if abs(took - R) > step_s + 1e-6:
            raise RC.RecordError("job %s: its recovery took %.3f s, not the mode's monitor.recovery_s" % (jid, took))


def grade_session(record, cfg, scored_ids):
    chk = RC.check(record, cfg.step_s())
    if not SC.overspeed_consistent(chk["truth"], chk["ended"], cfg.overspeed()):
        raise RC.RecordError("the truth log and the world's end of the session disagree on safety.overspeed_m_s")
    check_recoveries(chk, cfg.recovery_s(), cfg.step_s())
    failed = SC.failed_setup(chk["runner"])
    rows = ME.session_jobs(chk, cfg)
    cells = SC.cell_table(cfg.bands())
    s, terms = SC.session_score(rows, set(scored_ids), cfg.weights(), cells, cfg.overshoot_terms(), session_failed=failed)
    runner = {"start_outcomes": chk["start_outcomes"], "steps": chk["runner"]["steps"],
              "runner": chk["runner"]["runner"]}
    return {"score": s, "terms": terms, "runner": runner, "failed_setup": failed}


def grade_run(results, cfg, inside_caps, audit_passes):
    names = cfg.sessions()
    gates, share = SC.gate_values([results[n]["runner"] for n in names], inside_caps, audit_passes,
                                  cfg.on_time_share_gate())
    rew = SC.reward(gates, [results[n]["score"] for n in names])
    metrics = SC.summary(names, {n: (results[n]["score"], results[n]["terms"]) for n in names}, gates, share, rew)
    return {"reward": rew, "gates": gates, "metrics": metrics}


def zero_run(cfg, known_gates):
    """the run's verdict when a gate decided at the grader's entry is 0 (an over-cap submission scores 0 in every
    session, interface.submission_cap; a late write or a replaced submission): nothing is flown; known_gates: the gates
    read before the flight ({name: 0 or 1}); the others are not computed and not written."""
    if all(float(v) > 0.0 for v in known_gates.values()):
        raise ValueError("zero_run without a gate at 0")
    names = cfg.sessions()
    metrics = SC.summary(names, {n: (0.0, {}) for n in names}, dict(known_gates), None, 0.0)
    del metrics["on_time_share"]
    return {"reward": 0.0, "gates": dict(known_gates), "metrics": metrics}
