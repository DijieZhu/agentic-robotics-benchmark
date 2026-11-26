"""The measures of every job on the hidden world's true state (score.measured_on), in m, rad, m/s and rad/s.

  leg   end_pos    mean over [t_event + a, t_event + b] (score.end_window_s) of |p - goal position|
        end_att    mean over the same window of the angle between q and the goal attitude
        overshoot  the largest positive excursion beyond the goal along the move (unit vector from the pose of the job
                   before to the goal), from the job's start to the end window's close (score.O)
  slew  end_att, end_pos as a leg; overshoot: the largest positive turn beyond the goal about the move's rotation axis
  hold  pos_rms, att_rms about the hold's target (the goal of the job before, score.hold), speed_rms, rate_rms over the
        hold's duration
  view  pointing  mean angle between the imaging camera's axis (body +x) and the goal axis over the capture
                  [t_event, t_event + jobs.capture_s]
        centre    mean distance of the optical centre (jobs.camera_mount_m) from the goal centre over the capture
        rate_rms  over the capture
A window is the truth log's samples with a <= t <= b. Measures exist only for a job the world finished (status
'finished', with its accepted event for a leg, slew or view); every other job scores 0 by its status. The pose of the
job before follows the world's rule (plan.start_tolerance): the goal of the last job of the block that was flown, by
the controller or by a recovery of the built-in controller; a refused job, an accepted request in a mode that is not in
the paper's list of modes and a job that was not reached leave it as it was; the block's start pose for the first.
"""
import math

import numpy as np

from . import rot

NOT_FLOWN = ("refused", "refused_feasible", "accepted_request_not_flown", "not_started")
# only a job the world finished is measured; a job of any other status scores 0 by the rule that set it
# (budget: jobs.finished; invalid_plan: plan.*; handback: monitor.hand_back; watchdog, protocol: the block's end;
# ended: the session's end; refused_feasible, not_started: score.refusals, jobs.finished)
MEASURED = ("finished",)


class MeasureError(Exception):
    pass


def _win(T, a, b):
    i = int(np.searchsorted(T, a, side="left"))
    j = int(np.searchsorted(T, b, side="right"))
    if j <= i:
        raise MeasureError("a measuring window [%.3f, %.3f] holds no sample of the truth log" % (a, b))
    return i, j


def _angles(Q, gq):
    d = np.abs(Q @ np.asarray(gq, float))
    return 2.0 * np.arccos(np.minimum(1.0, d))


def leg_slew(tr, rec, kind, prev, end_window):
    T = tr["t"]
    gp = np.asarray(rec["goal"]["position"], float)
    gq = rot.qnormalize(rec["goal"]["attitude"])
    te = float(rec["t_event"])
    i, j = _win(T, te + end_window[0], te + end_window[1])
    out = {"end_pos": float(np.mean(np.linalg.norm(tr["p"][i:j] - gp, axis=1))),
           "end_att": float(np.mean(_angles(tr["q"][i:j], gq)))}
    a, b = _win(T, float(rec["t_start"]), te + end_window[1])
    if kind == "leg":
        u = gp - prev[0]
        n = float(np.linalg.norm(u))
        u = u / n if n > 1e-9 else np.zeros(3)
        ex = (tr["p"][a:b] - gp) @ u
        out["overshoot"] = float(max(0.0, float(ex.max())))
    else:
        rv = rot.qlog(rot.qmul(gq, rot.qconj(prev[1])))
        n = float(np.linalg.norm(rv))
        ax = rv / n if n > 1e-9 else np.zeros(3)
        ov = [float(rot.qlog(rot.qmul(q, rot.qconj(gq))) @ ax) for q in tr["q"][a:b]]
        out["overshoot"] = float(max(0.0, max(ov)))
    return out


def hold(tr, rec, target, duration_s):
    T = tr["t"]
    gp, gq = target
    t0 = float(rec["t_start"])
    i, j = _win(T, t0, t0 + float(duration_s))
    P, Q, Vv, W = tr["p"][i:j], tr["q"][i:j], tr["v"][i:j], tr["w"][i:j]
    return {"pos_rms": float(math.sqrt(np.mean(np.sum((P - gp) ** 2, 1)))),
            "att_rms": float(math.sqrt(np.mean(_angles(Q, gq) ** 2))),
            "speed_rms": float(math.sqrt(np.mean(np.sum(Vv ** 2, 1)))),
            "rate_rms": float(math.sqrt(np.mean(np.sum(W ** 2, 1))))}


def view(tr, rec, mount, capture_s):
    T = tr["t"]
    g = rec["goal"]
    c = np.asarray(g["centre"], float)
    ax = np.asarray(g["axis"], float)
    te = float(rec["t_event"])
    i, j = _win(T, te, te + float(capture_s))
    R = rot.qmats(tr["q"][i:j])
    cam_axis = R[:, :, 0]
    ang = np.arccos(np.clip(cam_axis @ ax, -1.0, 1.0))
    centre = tr["p"][i:j] + R @ np.asarray(mount, float)
    W = tr["w"][i:j]
    return {"pointing": float(np.mean(ang)), "centre": float(np.mean(np.linalg.norm(centre - c, axis=1))),
            "rate_rms": float(math.sqrt(np.mean(np.sum(W ** 2, 1))))}


def _event_time(r):
    """the accepted event's time (t_event) of a job the world finished."""
    return r.get("t_event")


def session_jobs(chk, cfg):
    """every job of every block in order, with its measures where it was finished: a list of dicts (block, mode, id,
    kind, status, measures or None, before: the id of the job of a scored kind before it, for a hold)."""
    mount = cfg.mount()
    ew = cfg.end_window()
    cap = cfg.capture_s()
    tr = chk["truth"]
    out = []
    for b in chk["blocks"]:
        prev = (np.asarray(b["start_pose"]["position"], float), rot.qnormalize(b["start_pose"]["attitude"]))
        before = None
        for j in b["jobs"]:
            r = chk["job_records"].get((b["name"], j["id"]))
            status = r["status"] if r is not None else "not_started"
            kind = j["kind"]
            row = {"block": b["name"], "mode": b["mode"], "id": j["id"], "kind": kind, "status": status,
                   "request_kind": j.get("request_kind"), "measures": None, "before": None}
            if kind == "hold":
                row["before"] = before
                if r is not None and r.get("hold_target") is not None:
                    ht = r["hold_target"]
                    if (np.linalg.norm(np.asarray(ht["position"], float) - prev[0]) > 1e-9 or
                            rot.qangle(rot.qnormalize(ht["attitude"]), prev[1]) > 1e-9):
                        raise MeasureError("hold %s: the world's target is not the goal of the job before" % j["id"])
                if status in MEASURED:
                    dur = float(j["goal"]["duration_s"])
                    if abs(dur - cfg.hold_s()) > 1e-9:
                        raise MeasureError("hold %s lasts %r s, not jobs.hold_s" % (j["id"], dur))
                    row["measures"] = hold(tr, r, prev, dur)
            elif kind in ("leg", "slew", "view"):
                if status in MEASURED:
                    if _event_time(r) is None:
                        raise MeasureError("job %s is finished without an accepted event" % j["id"])
                    rr = dict(r, goal=j["goal"], t_event=_event_time(r))
                    row["measures"] = (view(tr, rr, mount, cap) if kind == "view" else
                                       leg_slew(tr, rr, kind, prev, ew))
                before = j["id"]
            for k, x in (row["measures"] or {}).items():
                if not math.isfinite(x):
                    raise MeasureError("job %s: the measure %s is not finite" % (j["id"], k))
            out.append(row)
            if kind != "hold" and status not in NOT_FLOWN and not (
                    r is not None and status == "protocol" and r.get("protocol_error_call") == "plan"):
                prev = rot.body_goal(j, mount)
    return out
