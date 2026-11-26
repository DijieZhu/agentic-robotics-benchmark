"""The builder of synthetic flown sessions for the grader's unit tests: records in the form the world and the runner
leave them (grader/records.py), with a truth log whose every measure is set by the test, not by physics.

A session is described by blocks of jobs; each job names its kind, its status as the world would record it and the
errors its truth shows. The timeline follows the world's (a 16 ms step clock; plan calls take no time; an event comes
after the move; a leg's or slew's end window, a view's capture and a hold's duration close on the clock; a hand-back
or an invalid plan is followed by the built-in law's recovery of the mode's recovery time, flown perfectly to the
job's goal; a job that was not reached has a record 'not_started'), so that a grader that ignored the rule a status
stands for would find a perfect truth to credit.

    s = Session(cfg_values)                                   # the configuration's values (camera mount, step, ...)
    b = s.block("block_1", "nominal", budget_s=600.0)
    b.leg("j01", end_pos=0.01)                                # finished; errors in m and rad
    b.hold("j02", pos_rms=0.02)
    b.request("j03", "mode_not_available", refused=True)
    rec = s.record()                                          # the flown session record
Truth values per job: leg / slew: end_pos (m), end_att (rad), overshoot (m for a leg, rad for a slew); hold: pos_rms,
att_rms, speed_rms, rate_rms; view: pointing (rad), centre (m: the displacement when pointing is 0), rate (rad/s).
"""
import math

import numpy as np

MOVE_S = 3.0          # the synthetic move's duration
EVENT_AFTER_S = 0.512  # the event this long after the move's end


def qmul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([aw * bx + ax * bw + ay * bz - az * by, aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw, aw * bw - ax * bx - ay * by - az * bz])


def qexp(rv):
    rv = np.asarray(rv, float)
    th = float(np.linalg.norm(rv))
    if th < 1e-15:
        return np.array([0.0, 0.0, 0.0, 1.0])
    s = math.sin(0.5 * th) / th
    return np.array([rv[0] * s, rv[1] * s, rv[2] * s, math.cos(0.5 * th)])


def qmat(q):
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def qfrom_mat(R):
    t = R[0, 0] + R[1, 1] + R[2, 2]
    s = math.sqrt(t + 1.0) * 2
    q = np.array([(R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s, 0.25 * s])
    return q / np.linalg.norm(q)


def slerp(a, b, f):
    d = float(np.dot(a, b))
    if d < 0:
        b, d = -b, -d
    if d > 0.9999999:
        q = a + f * (b - a)
        return q / np.linalg.norm(q)
    th = math.acos(d)
    return (math.sin((1 - f) * th) * a + math.sin(f * th) * b) / math.sin(th)


def _perp(u):
    u = np.asarray(u, float)
    u = u / np.linalg.norm(u)
    a = np.array([0.0, 0.0, 1.0]) if abs(u[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    p = np.cross(u, a)
    return p / np.linalg.norm(p), u


class Block(object):
    def __init__(self, session, name, mode, budget_s, start_pose):
        self.s = session
        self.name = name
        self.mode = mode
        self.budget_s = float(budget_s)
        self.start = (np.asarray(start_pose[0], float), np.asarray(start_pose[1], float))
        self.jobs = []
        self.ended_by = None

    def _add(self, kind, jid, **kw):
        self.jobs.append(dict(kind=kind, id=jid, **kw))
        return self

    def leg(self, jid, d=(0.3, 0.0, 0.0), status="finished", **err):
        return self._add("leg", jid, d=d, status=status, err=err)

    def slew(self, jid, turn=(0.0, 0.0, 0.35), status="finished", **err):
        return self._add("slew", jid, turn=turn, status=status, err=err)

    def hold(self, jid, status="finished", duration_s=None, **err):
        return self._add("hold", jid, status=status, err=err, duration_s=duration_s)

    def view(self, jid, offset=(0.5, 0.2, 0.0), axis=(0.0, 1.0, 0.0), status="finished", **err):
        return self._add("view", jid, offset=offset, axis=axis, status=status, err=err)

    def request(self, jid, request_kind, refused=True, shape="leg"):
        st = "refused" if refused else ("accepted_request_not_flown" if request_kind == "mode_not_available"
                                        else "finished")
        return self._add("request", jid, request_kind=request_kind, status=st, shape=shape, err={})


class Session(object):
    def __init__(self, values, name="session_x", tight_mode="quiet"):
        self.values = values
        self.dt = float(values["interface.step_period_ms"]) / 1000.0
        self.mount = np.asarray(values["jobs.camera_mount_m"], float)
        self.ew = values["score.end_window_s"]
        self.capture = float(values["jobs.capture_s"])
        self.hold_s = float(values["jobs.hold_s"])
        self.R = values["monitor.recovery_s"]
        self.name = name
        self.tight = tight_mode
        self.blocks = []
        self.ended = None
        self.runner = {"session_failed": None, "task_error": None,
                       "runner": {"stray_processes": 0, "starts": 1, "deaths": ["process.end.session_end (close)"]},
                       "steps": None, "start_outcomes": ["ok"]}
        self.late_steps = 0
        self.skipped = 0
        self.overspeed_at = None

    def block(self, name, mode, budget_s=900.0, start_pose=None):
        if start_pose is None:
            start_pose = ((10.0 + len(self.blocks), -9.0, 4.3), (0.0, 0.0, 0.0, 1.0))
        b = Block(self, name, mode, budget_s, start_pose)
        self.blocks.append(b)
        return b

    # ------------------------------------------------------------------------------------------- the timeline
    def _grid(self, t):
        return round(t / self.dt) * self.dt

    def record(self):
        dt = self.dt
        T, P, Q, Vv, W = [], [], [], [], []
        t = 0.0
        pose = (np.asarray(self.blocks[0].start[0], float), np.asarray(self.blocks[0].start[1], float))

        def emit(t_to, fn):
            nonlocal t
            n = int(round((t_to - t) / dt))
            for k in range(1, n + 1):
                tk = t + k * dt
                p, q, v, w = fn(tk)
                T.append(tk); P.append(np.asarray(p, float)); Q.append(np.asarray(q, float))
                Vv.append(np.asarray(v, float)); W.append(np.asarray(w, float))
            t = t + n * dt

        def still(p, q):
            return lambda tk: (p, q, np.zeros(3), np.zeros(3))

        emit(20.0, still(*pose))                                 # the start hold
        blocks, jrs, brs = [], [], []
        session_over = False
        for bi, b in enumerate(self.blocks):
            if session_over:
                blocks.append(self._paper_block(b, []))
                continue
            if bi > 0:
                emit(t + 5.008, still(*b.start))                 # the gap: the built-in law at the next start pose
            pose = b.start
            t_b = t
            B = b.budget_s
            jobs_paper = []
            block_over = False
            handbacks = 0
            ended_by = None
            prev = b.start
            before_ok = None
            for j in b.jobs:
                paper, goal_pose = self._job_paper(j, prev)
                jobs_paper.append(paper)
                rec = {"block": b.name, "mode": b.mode, "id": j["id"], "kind": j["kind"], "job_mode": paper["mode"],
                       "goal": paper["goal"], "t_start": t, "status": None, "events": []}
                if j["kind"] == "request":
                    rec["request_kind"] = j["request_kind"]
                jrs.append(rec)
                if block_over:
                    rec["status"] = "not_started"
                    continue
                st = j["status"]
                if st == "not_started":
                    rec["status"] = "not_started"
                    block_over = True
                    ended_by = ended_by or "budget"
                    continue
                if st in ("refused", "refused_feasible", "accepted_request_not_flown"):
                    rec["status"] = st
                    continue
                if st == "invalid_plan":
                    rec["status"] = "invalid_plan"
                    rec["invalid_rule"] = "plan.limit_rule"
                    t0 = t
                    emit(t + math.ceil(self.R[b.mode] / dt - 1e-9) * dt, still(*goal_pose))
                    rec["recovery"] = {"t_from": t0, "t_to": t, "recovery_s": self.R[b.mode], "after_budget_s": 0.0}
                    prev = goal_pose
                    continue
                if j["kind"] == "hold":
                    self._fly_hold(j, rec, prev, emit, lambda: t)
                else:
                    self._fly_move(j, rec, prev, goal_pose, emit, lambda: t)
                if st == "finished":
                    rec["status"] = "finished"
                elif st == "handback":
                    handbacks += 1
                    rec["status"] = "handback"
                    rec["handback_reason"] = j["err"].get("reason", "end_point")
                    rec["t_handback"] = t
                    if handbacks >= 3:
                        rec["third_handback"] = True
                        block_over = True
                        ended_by = "third_handback"
                    else:
                        t0 = t
                        emit(t + math.ceil(self.R[b.mode] / dt - 1e-9) * dt, still(*goal_pose))
                        rec["recovery"] = {"t_from": t0, "t_to": t, "recovery_s": self.R[b.mode], "after_budget_s": 0.0}
                elif st in ("budget", "watchdog", "protocol"):
                    rec["status"] = st
                    block_over = True
                    ended_by = st
                    if st == "protocol":
                        rec["protocol_error_t"] = t
                elif st == "ended":
                    rec["status"] = "ended"
                    block_over = True
                    ended_by = "ended"
                    session_over = True
                    vbad = np.array([float(self.values["safety.overspeed_m_s"]) * 1.2, 0.0, 0.0])
                    emit(t + dt, lambda tk: (goal_pose[0], goal_pose[1], vbad, np.zeros(3)))
                    self.ended = ["overspeed", t]
                else:
                    raise ValueError("status %r" % st)
                if j["kind"] != "hold":
                    prev = goal_pose
            t_end = t
            if ended_by is None:
                ended_by = "jobs_done"
            if ended_by == "budget":
                t_end = max(t, t_b + B)
                if t_end > t:
                    emit(self._grid(t_end) + 0.0, still(*prev))
            brs.append({"name": b.name, "mode": b.mode, "t_start": t_b, "t_end": t_end, "elapsed_s": t_end - t_b,
                        "budget_s": B, "ended_by": ended_by, "recovery_after_budget": None})
            blocks.append(self._paper_block(b, jobs_paper))
        if not session_over:
            emit(t + 1.008, still(*pose))          # the time after the last block; after the session's end nothing
        n = len(T)
        steps = self.runner.get("steps") or {"steps_answered": n - self.skipped, "step_calls_skipped": self.skipped,
                                             "late_steps": self.late_steps}
        runner = dict(self.runner, steps=steps)
        truth = {"t": np.asarray(T), "p": np.asarray(P), "q": np.asarray(Q), "v": np.asarray(Vv), "w": np.asarray(W)}
        return {"session": self.name, "tight_mode": self.tight, "list": 1, "seed": "acceptance", "blocks": blocks,
                "job_records": jrs, "block_records": brs, "ended": self.ended, "truth": truth, "runner": runner}

    def _paper_block(self, b, jobs_paper):
        if not jobs_paper:
            jobs_paper = [self._job_paper(j, b.start)[0] for j in b.jobs]
        return {"name": b.name, "mode": b.mode, "start_pose": {"position": list(b.start[0]), "attitude": list(b.start[1])},
                "budget_s": b.budget_s, "jobs": jobs_paper}

    def _job_paper(self, j, prev):
        kind = j["kind"]
        if kind == "leg" or (kind == "request" and j.get("shape") == "leg"):
            gp = prev[0] + np.asarray(j.get("d", (0.3, 0.1, 0.0)), float)
            goal = {"position": list(gp), "attitude": list(prev[1])}
            pose = (gp, np.asarray(prev[1], float))
        elif kind == "slew":
            gq = qmul(qexp(j["turn"]), prev[1])
            goal = {"position": list(prev[0]), "attitude": list(gq)}
            pose = (np.asarray(prev[0], float), gq)
        elif kind == "hold":
            dur = j.get("duration_s") or self.hold_s
            goal = {"duration_s": float(dur)}
            pose = prev
        else:
            a = np.asarray(j.get("axis", (0.0, 1.0, 0.0)), float)
            a = a / np.linalg.norm(a)
            up, _ = _perp(a)
            c = prev[0] + np.asarray(j.get("offset", (0.5, 0.2, 0.0)), float)
            goal = {"centre": list(c), "axis": list(a), "up": list(up)}
            ez = -up
            ey = np.cross(ez, a)
            Rb = np.stack([a, ey, ez], 1)
            qb = qfrom_mat(Rb)
            pose = (c - Rb @ self.mount, qb)
        paper = {"id": j["id"], "kind": kind, "mode": j.get("mode", "nominal") if kind != "request" else
                 ("docking" if j["request_kind"] == "mode_not_available" else "nominal"), "goal": goal}
        if kind == "request":
            paper["request_kind"] = j["request_kind"]
            paper["request_shape"] = j.get("shape", "leg")
        return paper, pose

    def _fly_move(self, j, rec, prev, goal_pose, emit, now):
        dt = self.dt
        e = j["err"]
        kind = j["kind"] if j["kind"] != "request" else ("view" if j.get("shape") == "view" else "leg")
        t0 = now()
        p0, q0 = np.asarray(prev[0], float), np.asarray(prev[1], float)
        gp, gq = goal_pose
        u = gp - p0
        nu = float(np.linalg.norm(u))
        u = u / nu if nu > 1e-9 else np.array([1.0, 0.0, 0.0])
        perp, _ = _perp(u)
        rv = np.zeros(3)
        if kind == "slew":
            from_q = qmul(gq, np.array([-q0[0], -q0[1], -q0[2], q0[3]]))
            th = 2 * math.acos(min(1.0, abs(from_q[3])))
            rv = np.asarray(j["turn"], float)
            ax = rv / np.linalg.norm(rv)
            aperp, _ = _perp(ax)
        else:
            ax = np.array([0.0, 0.0, 1.0])
            aperp = np.array([1.0, 0.0, 0.0])
        t_move_end = t0 + MOVE_S
        bump = (t_move_end - 1.0, t_move_end - 0.5)
        ov = float(e.get("overshoot", 0.0))
        epos = float(e.get("end_pos", 0.0)) * perp
        eatt = qexp(float(e.get("end_att", 0.0)) * aperp)

        def move(tk):
            f = min(1.0, max(0.0, (tk - t0) / MOVE_S))
            p = p0 + f * (gp - p0)
            q = slerp(q0, gq, f)
            if bump[0] <= tk <= bump[1] and ov > 0:
                if kind == "leg":
                    p = gp + ov * u
                elif kind == "slew":
                    q = qmul(qexp(ov * ax), gq)
            return p, q, (gp - p0) / MOVE_S, np.zeros(3)

        emit(t_move_end, move)
        if kind == "view":
            pe = float(e.get("pointing", 0.0))
            a_cam = np.asarray(rec["goal"]["axis"], float)
            k, _ = _perp(a_cam)
            qv = qmul(qexp(pe * k), gq)
            dcv = float(e.get("centre", 0.0)) * _perp(a_cam)[0]
            wr = float(e.get("rate", 0.0)) * a_cam

            def settle(tk):
                return gp + dcv, qv, np.zeros(3), wr
            closes = self.capture
        else:
            def settle(tk):
                return gp + epos, qmul(eatt, gq), np.zeros(3), np.zeros(3)
            closes = float(self.ew[1])
        t_event = t0 + MOVE_S + EVENT_AFTER_S
        if j.get("event", True) is not False and j["status"] not in ("watchdog",):
            rec["t_event"] = t_event
        emit(t_event, settle)
        rec["t_motion_start"] = t0
        rec["t_plan_end"] = t_move_end
        if j["status"] == "handback" and e.get("reason") == "tolerance":
            rec.pop("t_event", None)
            return
        if j["status"] == "handback":
            return
        if j["status"] in ("budget", "protocol", "ended", "watchdog"):
            emit(t_event + 0.512, settle)
            return
        n = int(math.ceil((closes - 1e-9) / dt))
        emit(t_event + n * dt, settle)
        rec["t_window_end"] = now()

    def _fly_hold(self, j, rec, prev, emit, now):
        e = j["err"]
        t0 = now()
        dur = float(j.get("duration_s") or self.hold_s)
        tp, tq = np.asarray(prev[0], float), np.asarray(prev[1], float)
        d = np.array([float(e.get("pos_rms", 0.0)), 0.0, 0.0])
        qa = qmul(qexp(np.array([0.0, float(e.get("att_rms", 0.0)), 0.0])), tq)
        vh = np.array([0.0, float(e.get("speed_rms", 0.0)), 0.0])
        wh = np.array([0.0, 0.0, float(e.get("rate_rms", 0.0))])
        rec["t_motion_start"] = t0
        rec["hold_target"] = {"position": list(tp), "attitude": list(tq)}
        if j["status"] in ("budget", "protocol", "ended", "watchdog", "handback"):
            emit(t0 + 0.512, lambda tk: (tp + d, qa, vh, wh))
            return
        emit(t0 + dur, lambda tk: (tp + d, qa, vh, wh))
        rec["t_window_end"] = now()


def bands_fixture():
    """round synthetic bands in the form of score.bands (m, rad, m/s, rad/s); some cells drop by the rule Z < 2 FE:
    the legs' and slews' overshoot in the quiet mode, the holds' speed RMS, the view's centre and rate RMS."""
    cells = {}
    for mode in ("nominal", "quiet", "precision"):
        cells["leg/%s/end_pos" % mode] = (0.02, 0.075)
        cells["leg/%s/end_att" % mode] = (0.004, 0.08)
        cells["leg/%s/overshoot" % mode] = (0.03, 0.05) if mode == "quiet" else (0.01, 0.075)
        cells["slew/%s/end_att" % mode] = (0.003, 0.08)
        cells["slew/%s/end_pos" % mode] = (0.02, 0.075)
        cells["slew/%s/overshoot" % mode] = (0.006, 0.01) if mode == "quiet" else (0.003, 0.08)
        cells["hold/%s/pos_rms" % mode] = (0.03, 0.075)
        cells["hold/%s/att_rms" % mode] = (0.003, 0.08)
        cells["hold/%s/speed_rms" % mode] = (0.007, 0.0105)
        cells["hold/%s/rate_rms" % mode] = (0.0005, 0.002)
    cells["view/nominal/pointing"] = (0.0035, 0.17)
    cells["view/nominal/centre"] = (0.038, 0.033)
    cells["view/nominal/rate_rms"] = (0.0022, 0.0043)
    return {"cells": {k: {"full_end": fe, "zero_end": z} for k, (fe, z) in cells.items()}}
