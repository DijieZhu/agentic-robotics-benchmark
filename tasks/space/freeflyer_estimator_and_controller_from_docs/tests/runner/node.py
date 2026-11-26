"""The controller over the wire, as the node the session calls (node.setup(paper), node.plan(job, frame) and
node.step(frame)); this object puts the wire's rules between the session and the controller's process. Every rule it
applies is named by its key (PROTOCOL.json lists them) and counted in log["rules"]:

  * the set-up allowance (setup.allowance): every start of the controller's process in a session, the first and each
    restart, draws on the session's ONE allowance of interface.setup_cpu_s and interface.setup_wall_s, summed over the
    starts (the interpreter's start, the import, Controller() and setup(session) all count); when it is used up, a
    set-up overrun (setup.overrun): the session scores 0 and no further start is made; a start that cannot go on, or a
    setup that raises or replies malformed, is a set-up failure (setup.failure): the session scores 0;
  * the restart rule (restart): after the process has ended it is started again before the next call, fresh:
    Controller() and setup with the same session paper; nothing of its earlier state is handed to it;
  * plan: interface.plan_cpu_s / interface.plan_wall_s (plan.overrun: an invalid plan, the block goes on); the reply's
    form; the validity rules (plan.validity), whose verdict is a notice of the next frame the controller receives; the
    start pose of a plan (plan.start_pose);
  * step: the wall rule per stretch (interface.block_wall_rule, wall.stretches); the late-reply rule
    (interface.late_reply; the lists of a skipped frame arrive with the next frame, step.skipped_frames); the reply's
    form; a null command (step.null_command); the event rule (interface.event_rule); the watchdog's count
    (interface.watchdog_s); the step-time report;
  * a protocol error (an exception, a reply refused by its form, a reply stream that cannot go on) at a plan call or at
    a job's step ends the block (protocol_error.block); in the check-out while the controller is in control it hands
    the rest of the check-out to the built-in controller (protocol_error.outside_jobs); while the controller is not in
    control it costs its notice only (protocol_error.not_in_control);
  * the end of a block's stretch (block.end): at the first frame the session sends outside the block, for whatever
    cause the block ended (the frame's block fields; the runner's own end; where the frame cannot show it, the
    session's record of the block's end, which the grader hands the node with read_block_ends_from).

The node answers every plan call to the session with its verdict (plan.verdict): valid (the plan as checked, its
attitudes normalised); refused with its reason; invalid with the rule's key and the setpoint's index; protocol error
(the block ends at once). A step's protocol error is answered with a non-finite wrench, which ends the block (or, in
the check-out, hands it to the built-in controller) at that step.
"""
import collections
import math

import numpy as np

from . import clock as CK
from . import config as CF
from . import rules as RL
from . import stats as ST
from . import wire as WR

NONE_REPLY = {"force": None, "torque": None, "event": "none"}


def plain(x):
    if isinstance(x, dict):
        return {str(k): plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [plain(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    return x


def _nan_reply():
    n = float("nan")
    return {"force": [n, n, n], "torque": [n, n, n], "event": "none"}


class WireNode(object):
    def __init__(self, program_dir, run_root, cfg=None, sandbox_required=True, mem_cap_bytes=None, stderr_path=None,
                 extra_read_dirs=()):
        self.cfg = cfg or CF.get()
        self.runner = WR.Runner(program_dir, run_root, self.cfg, stderr_path=stderr_path,
                                sandbox_required=sandbox_required, mem_cap_bytes=mem_cap_bytes,
                                extra_read_dirs=extra_read_dirs)
        self.R = self.cfg.plan_rules()
        self.clock = CK.BlockClock(self.cfg)
        self.wd = CK.Watchdog(self.cfg)
        self.stats = ST.StepStats(self.cfg)
        self.paper = None
        self.session_failed = None
        self.task_error = None
        self.notices = []
        self.pending = None
        self.skip = 0
        self.job = None
        self.prev_goal = None
        self.block_index = 0
        self.block_dead = set()
        self.block_source = None
        self.allowance_used = {"cpu_s": 0.0, "wall_s": 0.0}
        self.rules_applied = collections.Counter()
        self.log = {"plans": [], "events": [], "protocol": [], "late": [], "restarts": [], "overruns": [],
                    "budgets": [], "setup": None, "starts": [], "cuts": [], "block_ends": [],
                    "rules": self.rules_applied}

    # ------------------------------------------------------------------------------------------------ the rules
    def _rule(self, key):
        """count one application of the rule named key (a key of PROTOCOL.json)."""
        self.rules_applied[key] += 1

    def _fail_session(self, key, why):
        """the session scores 0 (setup.overrun, setup.failure) or the run is a task error (task_error.*): no further
        call of the controller."""
        self._rule(key)
        if not self.session_failed:
            self.session_failed = "{}: {}".format(key, why)

    def _end_block(self, key, t, why):
        """the block ends at this call (protocol_error.block, interface.watchdog_s): its stretch ends here."""
        self._rule(key)
        if self.clock.in_block():
            self.block_dead.add(self.clock.block["index"])
            self.log["block_ends"].append({"t": t, "block": self.clock.block["index"], "rule": key, "why": why})
            self.clock.end_block(why)

    def _notice(self, n):
        """a notice for the next frame the controller receives (frame.notices)."""
        self.notices.append(n)

    def read_block_ends_from(self, source):
        """the grader hands the node the session's records of the blocks' ends (block.end): an object whose attribute
        block_records, or a callable that returns them, is the list of the blocks flown so far, each with t_start,
        t_end and ended_by; the node reads these three fields only, to end a block's stretch where the frame cannot
        show the block's end."""
        self.block_source = source

    def _session_block_end(self):
        """the session's record of the end of the block in progress, or None."""
        src = self.block_source
        if src is None or not self.clock.in_block():
            return None
        recs = src() if callable(src) else getattr(src, "block_records", None)
        eps = self.cfg.eps("time_eps_s")
        for r in reversed(list(recs or [])):
            if abs(float(r["t_start"]) - self.clock.block["t_start"]) <= eps:
                return {"t_start": float(r["t_start"]), "t_end": float(r["t_end"]), "ended_by": r.get("ended_by")}
        return None

    def _verdict(self, rec, outcome, reply=None, **fields):
        """the plan call's answer to the session (plan.verdict): the reply with the node's verdict."""
        self._rule("plan.verdict")
        v = {"outcome": outcome}
        v.update(fields)
        rec["verdict"] = dict(v)
        out = dict(reply or {})
        out["verdict"] = v
        return out

    def _protocol(self, call, t, what):
        self.log["protocol"].append({"call": call, "t": t, "what": what, "block": self.clock.block and
                                     self.clock.block["index"]})
        self._notice({"kind": "protocol_error", "call": call, "what": what})

    # ------------------------------------------------------------------------------------------------ starts
    def _use(self, srec, cpu_s, wall_s):
        self.allowance_used["cpu_s"] += float(cpu_s or 0.0)
        self.allowance_used["wall_s"] += float(wall_s or 0.0)
        srec["used_cpu_s"] = srec.get("used_cpu_s", 0.0) + float(cpu_s or 0.0)
        srec["used_wall_s"] = srec.get("used_wall_s", 0.0) + float(wall_s or 0.0)

    def _start_and_setup(self):
        """one start of the controller's process on the session's set-up allowance: (ok, info)."""
        cpu_cap, wall_cap = self.cfg.setup_caps
        cpu_left = cpu_cap - self.allowance_used["cpu_s"]
        wall_left = wall_cap - self.allowance_used["wall_s"]
        self._rule("setup.allowance")
        srec = {"start": len(self.log["starts"]) + 1, "cpu_left_s": cpu_left, "wall_left_s": wall_left}
        self.log["starts"].append(srec)
        if cpu_left <= 0.0 or wall_left <= 0.0:
            srec["outcome"] = "no start: the set-up allowance is used up"
            self._fail_session("setup.overrun", "the set-up allowance is used up before start %d" % srec["start"])
            return False, srec
        ok, info = self.runner.start(wall_left, cpu_left)
        self._use(srec, info.get("cpu_s"), info.get("wall_s"))
        if not ok:
            srec["outcome"] = info.get("error")
            if info.get("task_error"):
                self.task_error = plain(info)
                self._fail_session("task_error.sandbox_missing", info.get("error"))
            elif info.get("status") in ("cpu", "timeout"):
                self._fail_session("setup.overrun", "start %d: %s" % (srec["start"], info.get("error")))
            else:
                self._fail_session("setup.failure", "start %d: %s" % (srec["start"], info.get("error")))
            return False, srec
        st, res, cpu, wall, why = self.runner.call("setup", {"session": plain(self.paper)},
                                                   cpu_left - info["cpu_s"], max(0.0, wall_left - info["wall_s"]))
        self._use(srec, cpu, wall)
        srec["setup_status"] = st
        if st in ("cpu", "timeout"):
            srec["outcome"] = "setup over the allowance"
            self._fail_session("setup.overrun", "start %d: setup over what was left of the allowance (%s)" %
                               (srec["start"], why))
            return False, srec
        if st == "error":
            self._rule("reply.exception")
            srec["outcome"] = "setup raised: {}".format(res)
            self._fail_session("setup.failure", "start %d: setup raised: %s" % (srec["start"], res))
            return False, srec
        if st != "ok":
            srec["outcome"] = "{}: {}".format(st, why)
            self._fail_session("setup.failure", "start %d: %s: %s" % (srec["start"], st, why))
            return False, srec
        try:
            prof = RL.parse_setup(res)
        except ValueError as e:
            self._rule("reply.malformed")
            srec["outcome"] = "malformed setup reply: {}".format(e)
            self._fail_session("setup.failure", "start %d: a malformed setup reply: %s" % (srec["start"], e))
            return False, srec
        srec["outcome"] = "ok"
        return True, {"mass_profile": prof, "start": srec, "sandbox": info.get("sandbox")}

    def _ensure(self):
        """True when the controller runs (started again after its process ended); False when the session has failed."""
        if self.session_failed:
            return False
        if not self.runner.dead:
            return True
        self._rule("restart")
        ok, info = self._start_and_setup()
        self.log["restarts"].append({"ok": ok, "info": plain(info), "deaths": list(self.runner.stats["deaths"]),
                                     "starts": self.runner.stats["starts"]})
        return ok

    # ------------------------------------------------------------------------------------------------ frames
    def _deliver(self, frame):
        fr = dict(frame)
        if self.pending is not None:
            for k in ("inertial", "registrations", "fixes"):
                fr[k] = self.pending[k] + list(frame.get(k) or [])
            self.pending = None
        fr["notices"] = self.notices
        self.notices = []
        self._rule("frame.notices")
        return plain(fr)

    def _hold_lists(self, frame):
        if self.pending is None:
            self.pending = {"inertial": [], "registrations": [], "fixes": []}
        for k in self.pending:
            self.pending[k].extend(plain(frame.get(k) or []))

    def _frame_block(self, frame):
        t = float(frame["t"])
        inc = bool(frame.get("in_control"))
        n_stretches = len(self.clock.stretches)
        was_in_block = self.clock.in_block()
        self.clock.on_frame(frame, self._session_block_end())
        if was_in_block and not self.clock.in_block() and self.clock.block.get("end_read_from") != "the runner's own end":
            self._rule("block.end")
            self.log["block_ends"].append({"t": t, "block": self.clock.block["index"], "rule": "block.end",
                                           "why": self.clock.block["ended"],
                                           "read_from": self.clock.block.get("end_read_from")})
        if len(self.clock.stretches) != n_stretches:
            self._rule("wall.stretches")
        if self.clock.block is not None and self.clock.block["index"] != self.block_index:
            self.block_index = self.clock.block["index"]
            b = self.paper["blocks"][self.block_index - 1]
            self.prev_goal = (np.asarray(b["start_pose"]["position"], float), RL.qnormalize(b["start_pose"]["attitude"]))
            self.wd.on_block_start(t)
            self.wd.in_control_before = inc
            self.job = None
        else:
            self.wd.on_frame(t, inc)

    def _watch(self, t, in_control, applied, in_job):
        """the runner's watchdog count at a step; a trip at a job's step ends the block there."""
        if self.wd.on_step(t, in_control, applied, in_job):
            self._end_block("interface.watchdog_s", t, "the watchdog's count")

    # ------------------------------------------------------------------------------------------------ setup
    def setup(self, paper, *ignored):
        """the session's set-up: the paper goes to the controller; further arguments of the session's call are not
        passed on."""
        self.paper = plain(paper)
        self.log["budgets"] = CK.paper_budgets(self.paper, self.cfg)
        if not all(b["agree"] for b in self.log["budgets"]):
            self.task_error = {"error": "a block budget on the paper differs from jobs.block_budget",
                               "budgets": self.log["budgets"]}
            self._rule("task_error.budget_differs")
        ok, info = self._start_and_setup()
        self.log["setup"] = plain(info)
        if not ok:
            return None
        if info.get("mass_profile"):
            return {"mass_profile": info["mass_profile"]}
        return None

    # ------------------------------------------------------------------------------------------------ plan
    def plan(self, job, frame):
        t = float(frame["t"])
        self._frame_block(frame)
        bi = self.block_index
        rec = {"job_id": job["id"], "t": t, "block": bi}
        self.log["plans"].append(rec)
        self.job = None
        if self.session_failed:
            self._rule("session_failed.calls")
            rec["outcome"] = "session failed"
            return self._verdict(rec, "protocol_error", what="the session has failed: " + str(self.session_failed))
        if bi in self.block_dead:
            self._rule("protocol_error.block")
            rec["outcome"] = "the block was ended by a protocol error"
            return self._verdict(rec, "protocol_error", what="the block was ended by a protocol error")
        if not self._ensure():
            self._rule("session_failed.calls")
            rec["outcome"] = "session failed"
            return self._verdict(rec, "protocol_error", what="the session has failed: " + str(self.session_failed))
        fr = self._deliver(frame)
        cpu_cap, wall_cap = self.cfg.plan_caps
        st, res, cpu, wall, why = self.runner.call("plan", {"job": plain(job), "frame": fr}, cpu_cap, wall_cap)
        self._rule("charge.since_previous_reply")
        rec.update({"status": st, "cpu_s": cpu, "wall_s": wall})
        if st in ("cpu", "timeout"):
            rule = "interface.plan_cpu_s" if st == "cpu" else "interface.plan_wall_s"
            self._rule("plan.overrun")
            rec.update({"outcome": "overrun: invalid plan", "rule": rule, "process_ended": bool(self.runner.dead)})
            self.log["overruns"].append({"call": "plan", "job_id": job["id"], "rule": rule, "why": why})
            self._notice({"kind": "plan", "job_id": job["id"], "valid": False, "rule": rule, "setpoint": None})
            self._after_invalid(job)
            return self._verdict(rec, "invalid", {"setpoints": []}, rule=rule, setpoint=None)
        if st == "error":
            self._rule("reply.exception")
            what = "an exception in plan: {}".format(res)
        elif st != "ok":
            # the reply stream could not go on (the process has been ended)
            what = "{}: {}".format(st, why)
        else:
            what = None
            try:
                form, val = RL.parse_plan(res, self.cfg.refusal_reasons, self.R["max_setpoints"], self.R["zero_eps"])
            except ValueError as e:
                self._rule("reply.malformed")
                what = "malformed plan reply: {}".format(e)
        if what is not None:
            self._protocol("plan", t, what)
            self._end_block("protocol_error.block", t, "a protocol error in a plan reply")
            rec["outcome"] = "protocol error: the block ends"
            return self._verdict(rec, "protocol_error", what=what)
        if form == "refuse":
            self._rule("plan.refusal")
            rec["outcome"] = "refused: " + val
            return self._verdict(rec, "refused", {"refuse": val}, reason=val)
        kind = job["kind"]
        release = self.paper["release_modes"]
        if job["mode"] not in release:
            self._rule("plan.mode_not_listed")
            rec["outcome"] = "accepted, mode not listed: not flown"
            return self._verdict(rec, "valid", {"setpoints": [] if val is None else [list(s) for s in val]})
        if kind == "hold":
            self._rule("plan.hold")
            self.job = {"id": job["id"], "kind": "hold", "t_last": t, "accepted": False}
            self._notice({"kind": "plan", "job_id": job["id"], "valid": True})
            rec["outcome"] = "valid (hold)"
            return self._verdict(rec, "valid", {"setpoints": []})
        goal = CK.body_goal(job, self.cfg.camera_mount)
        lim = self.cfg.mode_limits(job["mode"])
        self._rule("plan.start_pose")
        self._rule("plan.validity")
        if val is None:
            verdict = ("plan.max_setpoints", None, "more than the cap")
        else:
            verdict = RL.check_plan(val, lim, self.prev_goal, goal, kind, self.R)
        self.prev_goal = goal
        if verdict:
            rule, idx, detail = verdict
            rec.update({"outcome": "invalid", "rule": rule, "setpoint": idx, "detail": detail})
            self._notice({"kind": "plan", "job_id": job["id"], "valid": False, "rule": rule, "setpoint": idx})
            return self._verdict(rec, "invalid", {"setpoints": []}, rule=rule, setpoint=idx)
        rec["outcome"] = "valid"
        self._notice({"kind": "plan", "job_id": job["id"], "valid": True})
        self.job = {"id": job["id"], "kind": kind, "t_last": float(val[-1][0]), "accepted": False}
        return self._verdict(rec, "valid", {"setpoints": val})

    def _after_invalid(self, job):
        """plan.start_pose: an invalid plan is followed by a recovery to the job's goal."""
        if job["kind"] != "hold" and job["mode"] in self.paper["release_modes"]:
            self.prev_goal = CK.body_goal(job, self.cfg.camera_mount)

    # ------------------------------------------------------------------------------------------------ step
    def _step_protocol(self, t, in_control, in_job, what):
        self._protocol("step", t, what)
        self.wd.on_step(t, in_control, False, False)
        if in_job:
            self._end_block("protocol_error.block", t, "a protocol error at a job's step")
            return _nan_reply()
        if in_control:
            # the check-out: the session hands its rest to the built-in controller
            self._rule("protocol_error.outside_jobs")
            return _nan_reply()
        # not in control: the reply is not applied anyway; it costs its notice only
        self._rule("protocol_error.not_in_control")
        return NONE_REPLY

    def step(self, frame):
        t = float(frame["t"])
        self._frame_block(frame)
        in_control = bool(frame.get("in_control"))
        blk = frame.get("block") or {}
        in_job = in_control and frame.get("job_id") not in (None, "checkout") and blk.get("budget_s") is not None
        if self.session_failed:
            self._rule("session_failed.calls")
            self.wd.on_step(t, in_control, False, False)
            return NONE_REPLY
        if in_job and self.block_index in self.block_dead:
            # a block the runner has ended (a protocol error in a plan reply): the session ends it at this step
            self._rule("protocol_error.block")
            self.wd.on_step(t, in_control, False, False)
            return _nan_reply()
        if self.skip > 0:
            self.skip -= 1
            self._hold_lists(frame)
            self.stats.skip()
            self._rule("step.skipped_frames")
            self._rule("step.on_time_share")
            self._watch(t, in_control, False, in_job)
            return NONE_REPLY
        if not self._ensure():
            self._rule("session_failed.calls")
            self.wd.on_step(t, in_control, False, False)
            return NONE_REPLY
        fr = self._deliver(frame)
        wall_cap = self.clock.allowance_left(t)
        self._rule("interface.block_wall_rule")
        st, res, cpu, wall, why = self.runner.call("step", {"frame": fr}, None, max(wall_cap, 0.0))
        self._rule("charge.since_previous_reply")
        self.clock.charge(wall)
        budget = self.cfg.step_cpu_s
        late = cpu > budget
        self.stats.answered(t, cpu, wall, late)
        if st == "timeout":
            in_block = self.clock.in_block()
            self.log["cuts"].append({"t": t, "in_block": in_block, "block": self.clock.block and
                                     self.clock.block["index"], "allowance_s": wall_cap, "in_control": in_control})
            if in_block:
                self._protocol("step", t, "interface.block_wall_rule: the stretch's wall allowance is used up")
                self._end_block("protocol_error.block", t, "a cut at the stretch's wall allowance")
                self.wd.on_step(t, in_control, False, False)
                return _nan_reply()
            self.wd.on_step(t, in_control, False, False)
            if in_control and frame.get("job_id") == "checkout":
                # the check-out: the process has been ended (it is started again before the next call) and the rest
                # of the check-out goes to the built-in controller, as for a protocol error there
                self._protocol("step", t, "interface.block_wall_rule: the stretch's wall allowance is used up; the rest "
                                          "of the check-out goes to the built-in controller")
                self._rule("wall.cut_checkout")
                return _nan_reply()
            # outside a block no job is lost: the process has been ended and is started again before the next call
            self._rule("wall.cut_outside_blocks")
            return NONE_REPLY
        if st == "error":
            self._rule("reply.exception")
            return self._step_protocol(t, in_control, in_job, "an exception in step: {}".format(res))
        if st != "ok":
            # the reply stream could not go on (the process has been ended)
            return self._step_protocol(t, in_control, in_job, "{}: {}".format(st, why))
        try:
            F, T, ev = RL.parse_step(res)
        except ValueError as e:
            self._rule("reply.malformed")
            return self._step_protocol(t, in_control, in_job, "malformed step reply: {}".format(e))
        applied = F is not None
        if not applied:
            self._rule("step.null_command")
        if late:
            n = int(math.ceil((cpu - budget) / budget))
            self.skip = n
            self._rule("interface.late_reply")
            self.log["late"].append({"t": t, "cpu_ms": cpu * 1000.0, "skipped": n, "in_control": in_control})
            applied = False
        kind = self.job["kind"] if (in_job and self.job is not None and frame.get("job_id") == self.job["id"]) else None
        acc, reason = RL.event_verdict(ev, kind, t, self.job["t_last"] if kind else 0.0,
                                       bool(self.job and self.job["accepted"]), applied, self.cfg.eps("time_eps_s"))
        if ev in ("done", "ready"):
            self._rule("interface.event_rule")
            self.log["events"].append({"t": t, "event": ev, "job_id": frame.get("job_id"), "accepted": acc,
                                       "reason": reason})
            if acc:
                self.job["accepted"] = True
            else:
                self._notice({"kind": "event_ignored", "job_id": frame.get("job_id"), "event": ev, "t": t,
                              "reason": reason})
        self._watch(t, in_control, applied, in_job)
        if not applied:
            return NONE_REPLY
        return {"force": F.tolist(), "torque": T.tolist(), "event": ev}

    # ------------------------------------------------------------------------------------------------ end
    def close(self):
        rs = self.runner.close()
        self._rule("process.end.session_end")
        return {"runner": rs, "steps": self.stats.report(self.wd), "session_failed": self.session_failed,
                "task_error": self.task_error, "block_clock_inconsistent": self.clock.inconsistent,
                "blocks_ended_by_the_runner": sorted(self.block_dead), "set_up_allowance_used": dict(self.allowance_used),
                "starts": len(self.log["starts"]), "stretches": self.clock.stretches,
                "watchdog_resets": len(self.wd.resets), "rules_applied": dict(self.rules_applied),
                "log_counts": {k: (len(v) if isinstance(v, list) else None) for k, v in self.log.items()}}
