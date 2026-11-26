"""The scoring formula (the table's score.* entries), on the measures of measures.py.

  L (score.L)           1 at the full end, 0 at the zero end, linear between
  a cell                (kind, mode, measure) with its full end FE and zero end Z (score.bands); dropped when Z < 2 FE
                        (score.zero_end_rule: its L is 1); its completion limit is Z, or 2 FE where dropped, except the
                        view's optical-centre displacement, whose completion limit is its zero end in every mode
                        (score.completed)
  completed / captured  the world finished the job (event accepted, window closed inside the budget, end-point check
                        passed, no hand-back: status 'finished') and each of its own measures is inside its completion
                        limit (score.completed); a hold is tested in the same way and also needs completed(job before)
  leg                   completed x L(end_pos) x L(end_att) x (0.8 + 0.2 O), O = L(overshoot)          (score.leg, score.O)
  slew                  completed x L(end_att) x L(end_pos) x (0.8 + 0.2 O)                            (score.slew)
  hold                  completed(job before) x L(pos_rms) x L(att_rms) x L(speed_rms) x L(rate_rms)   (score.hold)
  view                  captured x L(pointing) x L(centre) x L(rate_rms)                               (score.view)
  refusals              J x (share of the accepted feasible jobs completed), J = share of requests refused + share of
                        feasible jobs accepted - 1, at least 0                                         (score.refusals)
  session               the weighted mean over the kinds present (score.weights; a kind without scored jobs drops out
                        and the weights are renormalised); a job that is not finished scores 0 and stays in the mean; a
                        session whose set-up overran or failed scores 0                               (score.session)
  reward                gates x the mean of the acceptance sessions                                    (score.reward)
Only the jobs of the session's scored set (score.scored_set, frozen with the hidden set) enter the kinds' means; every
request enters the refusal term.
"""
import math

from .config import CENTRE, KIND_WEIGHT, MEASURES, REFUSAL_WEIGHT


class ScoreError(Exception):
    pass


def L(x, fe, z):
    """score.L: 1 at the full end FE, 0 at the zero end Z, linear between (a zero end at or below the full end is a
    step at the full end)."""
    if z <= fe:
        return 1.0 if x <= fe else 0.0
    return float(min(1.0, max(0.0, (z - x) / (z - fe))))


def dropped(fe, z):
    """score.zero_end_rule: the measure does not separate and leaves the product when Z < 2 FE."""
    return z < 2.0 * fe


def completion_limit(kind, measure, fe, z):
    """score.completed: the zero end, or twice the full end where the band rule dropped the measure; the view's
    optical-centre displacement keeps its zero end in every mode."""
    if (kind, measure) == CENTRE:
        return z
    return 2.0 * fe if dropped(fe, z) else z


def cell_table(bands):
    """{(kind, mode, measure): (FE, Z, dropped, completion limit)} from score.bands' checked cells."""
    out = {}
    for key, (fe, z) in bands.items():
        kind, mode, m = key.split("/")
        out[(kind, mode, m)] = (fe, z, dropped(fe, z), completion_limit(kind, m, fe, z))
    return out


def job_terms(kind, mode, measures, cells, ov):
    """(own completion, the product of the job's L terms) for a job the world finished; ov: the factor (a + b O) of
    score.leg and score.slew as the configuration reads them from the formulas' text."""
    own = True
    Lv = {}
    for m in MEASURES[kind]:
        if (kind, mode, m) not in cells:
            raise ScoreError("no band for the cell %s/%s/%s" % (kind, mode, m))
        fe, z, dr, cl = cells[(kind, mode, m)]
        x = measures[m]
        if x > cl:
            own = False
        Lv[m] = 1.0 if dr else L(x, fe, z)
    if kind == "leg":
        prod = Lv["end_pos"] * Lv["end_att"] * (ov[0] + ov[1] * Lv["overshoot"])
    elif kind == "slew":
        prod = Lv["end_att"] * Lv["end_pos"] * (ov[0] + ov[1] * Lv["overshoot"])
    elif kind == "hold":
        prod = Lv["pos_rms"] * Lv["att_rms"] * Lv["speed_rms"] * Lv["rate_rms"]
    else:
        prod = Lv["pointing"] * Lv["centre"] * Lv["rate_rms"]
    return own, prod


def job_scores(rows, cells, ov):
    """per job of a scored kind: {id: (completed, score)}; a hold is conditioned on the completion of the job of a
    scored kind before it in its block (score.hold), a job that is not finished scores 0 (its status is the rule that
    zeroed it: budget, invalid plan, hand-back, watchdog, protocol error, the session's end, a refusal)."""
    out = {}
    for r in rows:
        if r["kind"] not in MEASURES:
            continue
        if r["measures"] is None:
            # a job the world did not finish (measures.MEASURED): 0, kept in the mean
            comp, s = False, 0.0
        else:
            comp, prod = job_terms(r["kind"], r["mode"], r["measures"], cells, ov)
            s = prod if comp else 0.0
        if r["kind"] == "hold" and r["before"] is not None and not out.get(r["before"], (False, 0.0))[0]:
            comp, s = False, 0.0                     # score.hold: completed(job before)
        if not math.isfinite(s):
            raise ScoreError("job %s: a score that is not finite" % r["id"])
        out[r["id"]] = (comp, s)
    return out


def session_score(rows, scored_ids, cfg_weights, cells, ov, session_failed=False):
    """(score, terms {kind weight name: term}) of one flown session; scored_ids: the session's scored set (feasible
    job ids)."""
    w = cfg_weights
    feas = [r for r in rows if r["kind"] in MEASURES and r["id"] in scored_ids]
    missing = set(scored_ids) - {r["id"] for r in rows if r["kind"] in MEASURES}
    if missing:
        raise ScoreError("the scored set names jobs the session does not hold: %s" % sorted(missing)[:5])
    reqs = [r for r in rows if r["kind"] == "request"]
    if session_failed:
        terms = {KIND_WEIGHT[k]: 0.0 for k in sorted({r["kind"] for r in feas})}
        if reqs:
            terms[REFUSAL_WEIGHT] = 0.0
        return 0.0, terms
    js = job_scores(rows, cells, ov)
    terms = {}
    for k in MEASURES:
        v = [js[r["id"]][1] for r in feas if r["kind"] == k]
        if v:
            terms[KIND_WEIGHT[k]] = sum(v) / len(v)
    if reqs:
        refused = sum(1 for r in reqs if r["status"] == "refused") / float(len(reqs))
        accepted = [r for r in feas if r["status"] != "refused_feasible"]
        acc_share = len(accepted) / float(len(feas)) if feas else 0.0
        J = refused + acc_share - 1.0
        done = (sum(1 for r in accepted if js[r["id"]][0]) / float(len(accepted))) if accepted else 0.0
        terms[REFUSAL_WEIGHT] = max(0.0, J) * done
    tot = sum(w[k] for k in terms)
    if not tot > 0.0:
        raise ScoreError("the kinds present carry no weight")
    s = sum(w[k] * terms[k] for k in terms) / tot
    if not (math.isfinite(s) and -1e-12 <= s <= 1.0 + 1e-12):
        raise ScoreError("a session score outside [0, 1]: %r" % s)
    return min(1.0, max(0.0, s)), terms


# ----------------------------------------------------------------------------------------------- the gates
def gate_values(sessions_runner, submission_inside_caps, audit_passes, on_time_gate):
    """score.gates, each 0 or 1, from the runner's records of every session (a list of the records' runner parts) and
    the two checks made on the submission at the grader's entry:
      start_came_up  score.gates (1), read as what it measures: at least one start of the controller came up in
                   the run (interface.overrun: a start that does not come up is a set-up failure of its session only)
      caps         the submission is inside interface.submission_cap
      audit        the static import audit passes
      on_time      the share of the run's steps answered inside the processor budget is at least
                   interface.on_time_share_gate (a skipped call is not on time)
      survivors    no process of the controller was alive at the end of any session
    The system-call filter's installation is not a gate value: a runner record with a task error voids the run."""
    ups = 0
    answered = late = skipped = 0
    stray = 0
    for r in sessions_runner:
        ups += sum(1 for o in r["start_outcomes"] if o == "ok")
        st = r["steps"]
        answered += int(st["steps_answered"])
        late += int(st["late_steps"])
        skipped += int(st["step_calls_skipped"])
        stray += int(r["runner"]["stray_processes"])
    n_all = answered + skipped
    share = (answered - late) / float(n_all) if n_all else None
    return {"start_came_up": 1.0 if ups > 0 else 0.0,
            "caps": 1.0 if submission_inside_caps else 0.0,
            "audit": 1.0 if audit_passes else 0.0,
            "on_time": 1.0 if (share is not None and share >= on_time_gate) else 0.0,
            "survivors": 1.0 if stray == 0 else 0.0}, share


def reward(gates, session_scores):
    g = 1.0
    for v in gates.values():
        g *= float(v)
    if not session_scores:
        raise ScoreError("no session score")
    m = sum(session_scores) / float(len(session_scores))
    r = g * m
    if not (math.isfinite(r) and 0.0 <= r <= 1.0 + 1e-12):
        raise ScoreError("a reward outside [0, 1]: %r" % r)
    return min(1.0, r)


def failed_setup(runner_rec):
    """the session's set-up overran or failed (setup.overrun, setup.failure: that session scores 0); a task error of the
    runner (task_error.*) is not a score and raises."""
    sf = runner_rec.get("session_failed")
    te = runner_rec.get("task_error")
    if te:
        raise ScoreError("the runner reports a task error: %s" % (str(te)[:300],))
    if sf is None:
        return False
    if sf.startswith("task_error"):
        raise ScoreError("the runner reports a task error: %s" % sf[:300])
    if sf.startswith("setup.overrun") or sf.startswith("setup.failure"):
        return True
    raise ScoreError("the runner reports a session failure the grader does not know: %s" % sf[:300])


def summary(session_names, results, gates, share, rew):
    """score.grader_output: the reward, the gates, and per session its score and per kind its term; no per-job value,
    no measure, no band end, no hidden parameter, no job id."""
    m = {"reward": rew, "on_time_share": share if share is not None else 0.0}
    for k, v in gates.items():
        m["gate_" + k] = v
    for name in session_names:
        s, terms = results[name]
        m[name + "_score"] = s
        for k, v in sorted(terms.items()):
            m[name + "_" + k] = v
    return m


def overspeed_consistent(truth, ended, limit):
    """the truth log against the world's own end: a true speed above safety.overspeed_m_s must have ended the session
    at that step (else the world's records contradict its truth)."""
    import numpy as np
    sp = np.linalg.norm(truth["v"], axis=1)
    over = np.nonzero(sp > limit)[0]
    if len(over) == 0:
        return ended is None or ended[0] != "overspeed"
    if ended is None or ended[0] != "overspeed":
        return False
    return abs(float(truth["t"][over[0]]) - float(ended[1])) < 1e-6
