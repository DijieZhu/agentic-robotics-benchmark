"""The records of one flown session, as the grader reads them, and their checks.

A flown session record is a dict:
  session, tight_mode, list, seed       which hidden session and variant was flown
  blocks        the world's blocks as flown: name, mode, start_pose {position, attitude}, budget_s and the job list
                (id, kind, mode, goal; a request with request_kind and request_shape)
  job_records   the world's record of every job it reached (World.jobs_out without the setpoints): block, id, kind,
                status, t_start, and where they apply t_event, t_window_end, handback_reason, recovery, hold_target
  block_records the world's record of every block it flew (World.block_records): name, t_start, t_end, budget_s,
                ended_by
  ended         None, or [cause, t] when the world ended the session (safety.overspeed_m_s, safety.keep_in,
                safety.contact)
  truth         the world's true state per step: t (n), p (n, 3) body origin, q (n, 4) attitude, v (n, 3), w (n, 3)
                body rate relative to the cabin
  runner        the runner's record (node.close() of the wire, with the outcome of every start): session_failed,
                task_error, runner {stray_processes, ...}, steps {steps_answered, step_calls_skipped, late_steps, ...},
                start_outcomes
The three-file form the runner's flights leave (a result file, a truth log, the runner's report) is read into this
dict by read_files. A record that is missing a field, holds a field of the wrong form, a non-finite number, a status
the world does not write, or two records that contradict each other raises RecordError: the verification is void (a
task error, never a score).
"""
import json
import math
import os

import numpy as np

STATUSES = ("finished", "handback", "budget", "watchdog", "protocol", "ended", "not_started", "refused",
            "refused_feasible", "accepted_request_not_flown", "invalid_plan")
FEASIBLE = ("leg", "slew", "hold", "view")
ENDED_BY = ("budget", "jobs_done", "third_handback", "watchdog", "protocol", "ended")
TRUTH_KEYS = {"t": 1, "p": 3, "q": 4, "v": 3, "w": 3}


class RecordError(Exception):
    pass


def _need(d, key, types, what):
    if not isinstance(d, dict) or key not in d:
        raise RecordError("%s has no field %r" % (what, key))
    x = d[key]
    if types is not None and not isinstance(x, types):
        raise RecordError("%s field %r is %s, not %s" % (what, key, type(x).__name__, types))
    return x


def _num(x, what):
    if isinstance(x, bool) or not isinstance(x, (int, float, np.integer, np.floating)) or not math.isfinite(float(x)):
        raise RecordError("%s is not a finite number: %r" % (what, x))
    return float(x)


def _vec(x, n, what):
    try:
        a = np.asarray(x, float).reshape(-1)
    except (TypeError, ValueError):
        raise RecordError("%s is not a vector" % what)
    if a.shape != (n,) or not np.all(np.isfinite(a)):
        raise RecordError("%s is not a finite vector of %d" % (what, n))
    return a


def check_truth(tr):
    if not isinstance(tr, dict):
        raise RecordError("the truth log is not a mapping")
    out = {}
    n = None
    for k, dim in TRUTH_KEYS.items():
        if k not in tr:
            raise RecordError("the truth log has no %r" % k)
        try:
            a = np.asarray(tr[k], float)
        except (TypeError, ValueError):
            raise RecordError("the truth log's %r is not numeric" % k)
        if dim == 1:
            ok = a.ndim == 1
        else:
            ok = a.ndim == 2 and a.shape[1] == dim
        if not ok:
            raise RecordError("the truth log's %r has the shape %s" % (k, a.shape))
        if not np.all(np.isfinite(a)):
            raise RecordError("the truth log's %r holds a number that is not finite" % k)
        if n is None:
            n = len(a)
        elif len(a) != n:
            raise RecordError("the truth log's arrays differ in length")
        out[k] = a
    if n < 2:
        raise RecordError("the truth log holds fewer than two steps")
    if not np.all(np.diff(out["t"]) > 0.0):
        raise RecordError("the truth log's times do not increase")
    nq = np.linalg.norm(out["q"], axis=1)
    if not np.all(np.abs(nq - 1.0) < 1e-6):
        raise RecordError("the truth log's attitudes are not unit quaternions")
    return out


def check(rec, step_s):
    """the checked record (a new dict: truth as arrays, job records by (block, id)); RecordError on any defect."""
    if not isinstance(rec, dict):
        raise RecordError("the flown session's record is not a mapping")
    for k in ("session", "tight_mode", "seed"):
        _need(rec, k, str, "the record")
    blocks = _need(rec, "blocks", list, "the record")
    if not blocks:
        raise RecordError("the record holds no block")
    names = []
    ids = set()
    for b in blocks:
        nm = _need(b, "name", str, "a block")
        names.append(nm)
        _need(b, "mode", str, "block " + nm)
        sp = _need(b, "start_pose", dict, "block " + nm)
        _vec(_need(sp, "position", None, "a start pose"), 3, "block %s start position" % nm)
        q = _vec(_need(sp, "attitude", None, "a start pose"), 4, "block %s start attitude" % nm)
        if not np.linalg.norm(q) > 0:
            raise RecordError("block %s start attitude has norm 0" % nm)
        if _num(_need(b, "budget_s", None, "block " + nm), "block %s budget" % nm) <= 0:
            raise RecordError("block %s has no budget" % nm)
        for j in _need(b, "jobs", list, "block " + nm):
            jid = _need(j, "id", str, "a job")
            kind = _need(j, "kind", str, "job " + jid)
            if kind not in FEASIBLE + ("request",):
                raise RecordError("job %s has the kind %r" % (jid, kind))
            _need(j, "goal", dict, "job " + jid)
            if (nm, jid) in ids:
                raise RecordError("job %s appears twice in block %s" % (jid, nm))
            ids.add((nm, jid))
    if len(set(names)) != len(names):
        raise RecordError("two blocks share a name")
    jr = _need(rec, "job_records", list, "the record")
    by = {}
    order = {}
    for r in jr:
        key = (_need(r, "block", str, "a job record"), _need(r, "id", str, "a job record"))
        if key not in ids:
            raise RecordError("a job record names a job the blocks do not hold: %s" % (key,))
        if key in by:
            raise RecordError("two job records for %s" % (key,))
        st = _need(r, "status", None, "job record %s" % (key,))
        if st not in STATUSES:
            raise RecordError("job record %s has the status %r" % (key, st))
        _num(_need(r, "t_start", None, "job record %s" % (key,)), "t_start")
        by[key] = r
        order.setdefault(key[0], []).append(key[1])
    for b in blocks:
        want = [j["id"] for j in b["jobs"]]
        got = order.get(b["name"], [])
        if got and got != want[:len(got)]:
            raise RecordError("the job records of %s are not in the block's order" % b["name"])
        if got and len(got) != len(want):
            raise RecordError("block %s was reached but not every job of it has a record" % b["name"])
    br = _need(rec, "block_records", list, "the record")
    brn = []
    for r in br:
        nm = _need(r, "name", str, "a block record")
        if _need(r, "ended_by", None, "block record " + nm) not in ENDED_BY:
            raise RecordError("block record %s ended by %r" % (nm, r.get("ended_by")))
        t0 = _num(_need(r, "t_start", None, "block record " + nm), "t_start")
        t1 = _num(_need(r, "t_end", None, "block record " + nm), "t_end")
        if t1 < t0:
            raise RecordError("block record %s ends before it starts" % nm)
        _num(_need(r, "budget_s", None, "block record " + nm), "budget_s")
        brn.append(nm)
    if brn != names[:len(brn)]:
        raise RecordError("the block records are not the blocks in their order")
    reached = set(order)
    if not reached <= set(brn):
        raise RecordError("a block has job records but no block record")
    ended = rec.get("ended", "missing")
    if ended == "missing":
        raise RecordError("the record has no field 'ended'")
    if ended is not None:
        if not isinstance(ended, (list, tuple)) or len(ended) != 2 or not isinstance(ended[0], str):
            raise RecordError("the session's end is not [cause, t]")
        _num(ended[1], "the session's end time")
    truth = check_truth(_need(rec, "truth", dict, "the record"))
    runner = _need(rec, "runner", dict, "the record")
    sf = runner.get("session_failed", "missing")
    if sf == "missing" or not (sf is None or isinstance(sf, str)):
        raise RecordError("the runner's record has no session_failed of the right form")
    if "task_error" not in runner:
        raise RecordError("the runner's record has no task_error")
    rs = _need(runner, "runner", dict, "the runner's record")
    _num(_need(rs, "stray_processes", None, "the runner's stats"), "stray_processes")
    steps = _need(runner, "steps", dict, "the runner's record")
    for k in ("steps_answered", "step_calls_skipped", "late_steps"):
        _num(_need(steps, k, None, "the step report"), k)
    so = _need(runner, "start_outcomes", list, "the runner's record")
    blk_by = {r["name"]: r for r in br}
    for (bn, jid), r in by.items():
        b = blk_by.get(bn)
        if b is None:
            raise RecordError("job %s of block %s has no block record" % (jid, bn))
        t_lo = b["t_start"] - 1e-6
        if r["status"] == "finished":
            kind = r.get("kind")
            if kind in ("leg", "slew", "view"):
                te = _num(_need(r, "t_event", None, "finished job " + jid), "t_event")
                tw = _num(_need(r, "t_window_end", None, "finished job " + jid), "t_window_end")
                if not (t_lo <= r["t_start"] <= te <= tw):
                    raise RecordError("finished job %s: its times are out of order" % jid)
            else:
                tw = _num(_need(r, "t_window_end", None, "finished hold " + jid), "t_window_end")
            if tw > b["t_start"] + b["budget_s"] + step_s + 1e-6:
                raise RecordError("job %s is finished after its block's budget" % jid)
    return {"session": rec["session"], "tight_mode": rec["tight_mode"], "seed": rec["seed"], "list": rec.get("list"),
            "blocks": blocks, "job_records": by, "block_records": br, "ended": ended, "truth": truth,
            "runner": runner, "start_outcomes": so}


# ----------------------------------------------------------------------------------------------- the three files
def read_files(result_json, truth_npz, runner_json, blocks):
    """a flown session's record from the three files the runner's flights leave: the result file (session, seed,
    tight_mode, list, ended, block_records), the truth log (npz: t, p, q, v, w) and the runner's report (its close()
    record with world_job_records and world_block_records); blocks: the world's blocks as flown. A file that is missing
    or does not parse raises RecordError."""
    try:
        with open(result_json, "rb") as fh:
            res = json.loads(fh.read().decode("utf-8"))
        with open(runner_json, "rb") as fh:
            rep = json.loads(fh.read().decode("utf-8"))
        with np.load(truth_npz, allow_pickle=False) as z:
            truth = {k: np.asarray(z[k]) for k in TRUTH_KEYS if k in z.files}
    except (OSError, ValueError, UnicodeDecodeError, KeyError) as exc:
        raise RecordError("a file of the flown session cannot be read: %s: %s" % (type(exc).__name__, exc))
    if not isinstance(res, dict) or not isinstance(rep, dict):
        raise RecordError("a file of the flown session is not a JSON object")
    runner = {k: rep.get(k) for k in ("session_failed", "task_error", "runner", "steps")}
    runner["start_outcomes"] = rep.get("start_outcomes")
    if runner["start_outcomes"] is None and isinstance(rep.get("log"), dict):
        runner["start_outcomes"] = [s.get("outcome") for s in rep["log"].get("starts") or []]
    for k in ("session_failed", "task_error"):
        if k not in rep:
            del runner[k]
    return {"session": res.get("session"), "tight_mode": res.get("tight_mode"), "list": res.get("list"),
            "seed": res.get("seed"), "blocks": blocks, "job_records": rep.get("world_job_records"),
            "block_records": rep.get("world_block_records"), "ended": res.get("ended", "missing"),
            "truth": truth, "runner": runner}


def exists_all(paths):
    missing = [p for p in paths if not os.path.isfile(p)]
    if missing:
        raise RecordError("files of the flown session are missing: %s" % ", ".join(os.path.basename(p) for p in missing))
