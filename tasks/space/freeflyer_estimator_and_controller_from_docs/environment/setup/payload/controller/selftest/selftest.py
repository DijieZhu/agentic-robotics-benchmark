#!/usr/bin/env python3
"""Self-test of a cabin free-flyer controller: the wire, the plan rules and the events, against a TOY plant.

    python3 selftest.py PATH/TO/submission/controller [--json FILE] [--printed FILE] [--log FILE] [--quiet]

What it does. It starts the controller folder (controller.py with class Controller) the way the acceptance does and
talks to it over the same wire: the files in wire/ are the acceptance runner's code (JSON lines over standard
streams, the processor and wall accounting, the sandbox, the reply forms, the plan validity rules, the event rule,
the late-reply rule, the set-up allowance and the restart rule, the block clock and budget, the wall rule per
stretch, the watchdog's count, the step-time report). It flies one short toy
session: a start hold, a check-out, three blocks (legs, slews, holds, views and two requests to refuse; the third
block after a mode change) and the time between them, then it prints what each rule made of the controller's
replies.

What it is not. The plant is a toy rigid body with toy sensors that use the frame's message formats and
conventions (cabin frame, body frame, quaternions x y z w, inertial samples in the inertial unit's own frame and
not corrected for offsets, fixes of a navigation camera's pose stamped at the image's time, registrations, the
propulsion telemetry, the camera's busy flag, the block's budget and time left). Its mass, inertia, sensor mounting,
offsets and noise, camera mountings, fix timing, fan speeds, mode limits, zones and every duration of the toy
session are arbitrary toy values: nothing in it describes the robot, the cabin or the acceptance sessions, and how
well the controller holds the toy body says nothing about the robot. Use it for the protocol, the budgets, the
rules and the signs (does a positive force along body x push the toy body along its x axis?).

What the toy session does differently from the acceptance's sessions. It has no supervisor: no hand-backs and no
recoveries; a job whose plan is invalid (an overrun of a hold's plan call included) puts the toy body at the job's
goal at once, where the acceptance's built-in controller flies a recovery that takes the mode's recovery time off the
block's budget. Between blocks the toy body rests for a toy time instead of being flown to the next block's start.
Its zones and its check-out box are toy boxes. Like the acceptance's session it acts on the wire's verdict on every
plan call (valid, refused, invalid, protocol error), ends a block at its budget, after its last job, at a protocol
error or at the watchdog, and hands the wire its record of each block's end, so a block's stretch of the wall rule
ends with the block.

The toy session's paper. setup(session) receives a paper with the keys of the acceptance's session paper: session,
release, release_modes, checkout_mode, start_pose, zone_file, battery_slots ({"occupied": ..., "of": ...}), equipment,
time_window ({"planned_minutes": ...}), blocks (each with its name, mode, start_pose, its ordered jobs and budget_s)
and window_s. Every value in it is a toy value. Two keys are the toy's own: its boxes are given inline under "zones"
(the toy has no zone file), and "toy" is true.

The toy values of the rules. The wire's own numbers (the processor and wall budgets, the step period, the watchdog,
the reply cap, the plan rules' tolerances, the budget factor) are the acceptance's. The values that belong to the
robot are toy values here, marked "toy" in wire/runner_config.json: the modes' speed and acceleration limits, the
imaging camera's mounting and capture time, the fan levels and the impeller speeds. A controller that plans with
the printed limits will see the toy limits judge its plans and the toy budgets time its blocks. To judge its plans
with the printed values instead, write them into a JSON file by their keys, for example

    {"plan.speed_limit.nominal": {"linear_m_s": ..., "angular_rad_s": ...}, "jobs.camera_mount_m": [...], ...}

and give it with --printed FILE (the keys that may be given are the ones marked "toy").

Options
  --json FILE     write the full report as JSON
  --printed FILE  values by key that replace the toy values of the rules (see above)
  --log FILE      where the controller's error stream (its prints) goes (default: a file next to the run folder)
  --run-dir DIR   where the run folder is made (default: a new folder under the system's temporary folder)
  --quiet         print the summary only

Exit status: 0 the controller ran the toy session without a protocol error, an overrun or a failed setup; 1
otherwise; 2 it could not be started. Run it with the python3 of the image: standard library and numpy.
"""
import argparse
import json
import math
import os
import shutil
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from wire import config as CF  # noqa: E402
from wire import clock as CK  # noqa: E402
from wire import node as ND  # noqa: E402
from wire import rules as RL  # noqa: E402
from wire import procs as PR  # noqa: E402

# ------------------------------------------------------------------------------------------------ the toy (all ours)
TOY = {
    "mass_kg": 4.0, "inertia_diag": [0.04, 0.05, 0.06],
    "command_delay_steps": 1,
    "imu_interval_s": 0.006, "imu_jitter_s": 0.001,
    "imu_axes_in_body": [0.7071067811865476, 0.0, 0.0, 0.7071067811865476],     # the toy unit turned 90 deg about x
    "imu_position_m": [0.02, 0.0, 0.03],
    "gyro_offset_rad_s": [0.002, -0.001, 0.0005], "accel_offset_m_s2": [0.05, -0.02, 0.01],
    "gyro_noise_rad_s": 0.001, "accel_noise_m_s2": 0.005,
    "nav_camera_position_m": [0.0, 0.08, 0.0], "nav_camera_axes_in_body": [0.0, 0.0, 0.0, 1.0],
    "image_interval_s": 0.25, "registration_offset_s": 0.012, "registration_delivery_s": 0.035,
    "fix_delay_s": 0.55, "fix_noise_m": 0.004, "fix_noise_rad": 0.003, "every_nth_image_not_localised": 9,
    "impeller_rad_s_per_level": {"1": 100.0, "2": 150.0, "3": 200.0}, "impeller_ramp_rad_s2": 40.0,
    "ready_within_rad_s": 5.0,
    "zone_half_m": 3.0,
    "start_hold_s": 3.0, "checkout_s": 5.0, "checkout_box_m": 0.5, "between_s": 2.0, "hold_s": 3.0,
    "start_pose": {"position": [1.0, 2.0, 1.5], "attitude": [0.0, 0.0, 0.0, 1.0]},
    "battery_slots": {"occupied": 1, "of": 2},
    "seed": 12345,
}


def qmul(a, b):
    return RL.qmul(a, b)


def qconj(q):
    return np.array([-q[0], -q[1], -q[2], q[3]])


def qrot(q, v):
    """the vector v of the axes q (x y z w) in the frame q is given in."""
    p = np.array([v[0], v[1], v[2], 0.0])
    return qmul(qmul(q, p), qconj(q))[:3]


def qz(deg):
    h = math.radians(deg) / 2.0
    return [0.0, 0.0, math.sin(h), math.cos(h)]


def toy_session(cfg):
    """the toy session paper: three blocks of toy jobs around the toy start pose (the request j04 names a mode that is
    not in the paper's release_modes; the request j08 has its goal outside the toy zone), with the fields of the
    acceptance's paper under the same keys and toy values, each block's budget by the block budget's formula and the
    toy session's planned minutes."""
    sp = TOY["start_pose"]
    x0, y0, z0 = sp["position"]
    q0 = sp["attitude"]
    b1 = {"name": "block_1", "mode": "nominal", "start_pose": sp, "jobs": [
        {"id": "j01", "kind": "leg", "mode": "nominal", "goal": {"position": [x0 + 0.3, y0, z0], "attitude": q0}},
        {"id": "j02", "kind": "hold", "mode": "nominal", "goal": {"duration_s": TOY["hold_s"]}},
        {"id": "j03", "kind": "slew", "mode": "nominal", "goal": {"position": [x0 + 0.3, y0, z0], "attitude": qz(20.0)}},
        {"id": "j04", "kind": "leg", "mode": "docking", "goal": {"position": [x0 + 0.3, y0 + 0.3, z0],
                                                                   "attitude": qz(20.0)}},
        {"id": "j05", "kind": "leg", "mode": "nominal", "goal": {"position": [x0, y0 + 0.2, z0 + 0.1],
                                                                   "attitude": qz(20.0)}}]}
    sp2 = {"position": [x0, y0 + 0.2, z0 + 0.1], "attitude": qz(20.0)}
    b2 = {"name": "block_2", "mode": "nominal", "start_pose": sp2, "jobs": [
        {"id": "j06", "kind": "view", "mode": "nominal", "goal": {"centre": [x0 + 0.1, y0 + 0.3, z0 + 0.1],
                                                                   "axis": [1.0, 0.0, 0.0], "up": [0.0, 0.0, 1.0]}},
        {"id": "j07", "kind": "view", "mode": "nominal", "goal": {"centre": [x0 + 0.1, y0 + 0.3, z0 + 0.1],
                                                                   "axis": [0.9396926207859084, 0.3420201433256687, 0.0],
                                                                   "up": [0.0, 0.0, 1.0]}},
        {"id": "j08", "kind": "view", "mode": "nominal", "goal": {"centre": [x0 + TOY["zone_half_m"] + 1.0, y0, z0],
                                                                   "axis": [1.0, 0.0, 0.0], "up": [0.0, 0.0, 1.0]}}]}
    sp3 = {"position": [x0, y0, z0], "attitude": q0}
    b3 = {"name": "block_3", "mode": "quiet", "start_pose": sp3, "jobs": [
        {"id": "j09", "kind": "leg", "mode": "quiet", "goal": {"position": [x0 + 0.1, y0, z0], "attitude": q0}},
        {"id": "j10", "kind": "hold", "mode": "quiet", "goal": {"duration_s": TOY["hold_s"]}}]}
    paper = {"session": "toy_session", "release": "toy release", "release_modes": ["nominal", "quiet", "precision"],
             "checkout_mode": "nominal", "start_pose": sp, "zones": {"keep_in": [[[x0 - TOY["zone_half_m"], y0 -
                                                                                    TOY["zone_half_m"], z0 - TOY["zone_half_m"]],
                                                                                   [x0 + TOY["zone_half_m"], y0 +
                                                                                    TOY["zone_half_m"], z0 + TOY["zone_half_m"]]]],
                                                                     "keep_out": [], "note": "toy zones (not a zone file)"},
             "zone_file": "toy zone file: none (the toy boxes are given inline under 'zones')",
             "battery_slots": dict(TOY["battery_slots"]),
             "equipment": "toy equipment: the toy body as this file defines it",
             "blocks": [b1, b2, b3], "window_s": 3.0, "toy": True}
    for i, b in enumerate(paper["blocks"]):
        before = paper["checkout_mode"] if i == 0 else paper["blocks"][i - 1]["mode"]
        b["budget_s"] = CK.block_budget(b, cfg, paper["release_modes"], before)[0]
    # the toy session's planned minutes: its start hold, its check-out, the blocks' budgets and the rests between them
    planned_s = (TOY["start_hold_s"] + TOY["checkout_s"] + sum(float(b["budget_s"]) for b in paper["blocks"])
                 + TOY["between_s"] * (len(paper["blocks"]) - 1))
    paper["time_window"] = {"planned_minutes": round(planned_s / 60.0, 6)}
    return paper


# ------------------------------------------------------------------------------------------------ the toy plant
class ToyPlant(object):
    def __init__(self, cfg, start):
        self.cfg = cfg
        self.rng = np.random.default_rng(TOY["seed"])
        self.dt = cfg.step_period_s
        self.m = TOY["mass_kg"]
        self.J = np.diag(TOY["inertia_diag"])
        self.p = np.asarray(start["position"], float)
        self.q = RL.qnormalize(start["attitude"])
        self.v = np.zeros(3)
        self.w = np.zeros(3)
        self.a_body = np.zeros(3)
        self.dw = np.zeros(3)
        self.queue = [np.zeros(6)] * TOY["command_delay_steps"]
        self.hist = []
        self.t = 0.0
        self.next_imu = TOY["imu_interval_s"]
        self.next_image = TOY["image_interval_s"]
        self.image_no = 0
        self.pending = []           # (arrival, kind, entry)
        self.level = cfg.v("mode.fan_level")["nominal"]
        self.speed = [float(TOY["impeller_rad_s_per_level"][str(self.level)])] * 2
        self.last_cmd = np.zeros(6)
        self.camera_busy_until = -1.0
        self.q_imu = RL.qnormalize(TOY["imu_axes_in_body"])
        self.q_nav = RL.qnormalize(TOY["nav_camera_axes_in_body"])

    def teleport(self, pose):
        """the toy's stand-in for the built-in controller between blocks and after an invalid plan (toy: at once)."""
        self.p = np.asarray(pose[0], float).copy()
        self.q = RL.qnormalize(pose[1])
        self.v = np.zeros(3)
        self.w = np.zeros(3)

    def advance(self, cmd):
        """one step of the toy dynamics with the command issued now (applied after the toy delay)."""
        self.queue.append(np.asarray(cmd, float))
        u = self.queue.pop(0)
        F, T = u[:3], u[3:]
        dt = self.dt
        self.a_body = F / self.m
        self.dw = np.linalg.solve(self.J, T - np.cross(self.w, self.J @ self.w))
        t0 = self.t
        self.v = self.v + qrot(self.q, self.a_body) * dt
        self.p = self.p + self.v * dt
        self.w = self.w + self.dw * dt
        self.q = RL.qnormalize(qmul(self.q, RL.qexp(self.w * dt)))
        self.t = t0 + dt
        self.hist.append((self.t, self.p.copy(), self.q.copy()))
        if len(self.hist) > 400:
            del self.hist[:100]
        self.last_cmd = np.asarray(cmd, float)
        target = float(TOY["impeller_rad_s_per_level"][str(self.level)])
        for i in (0, 1):
            d = target - self.speed[i]
            self.speed[i] += max(-TOY["impeller_ramp_rad_s2"] * dt, min(TOY["impeller_ramp_rad_s2"] * dt, d))

    def pose_at(self, t):
        best = None
        for h in self.hist:
            if h[0] <= t + 1e-12:
                best = h
        if best is None:
            return self.p.copy(), self.q.copy()
        return best[1], best[2]

    def frame(self, in_control, job_id, mode, budget, left):
        """the frame of the toy at its time: what arrived since the previous frame."""
        t = self.t
        ins = []
        while self.next_imu <= t + 1e-12:
            ts = self.next_imu
            r = np.asarray(TOY["imu_position_m"], float)
            acc_b = self.a_body + np.cross(self.dw, r) + np.cross(self.w, np.cross(self.w, r))
            rate = qrot(qconj(self.q_imu), self.w) + TOY["gyro_offset_rad_s"] + \
                TOY["gyro_noise_rad_s"] * self.rng.standard_normal(3)
            acc = qrot(qconj(self.q_imu), acc_b) + TOY["accel_offset_m_s2"] + \
                TOY["accel_noise_m_s2"] * self.rng.standard_normal(3)
            ins.append({"stamp": round(ts, 9), "rate": rate.tolist(), "accel": acc.tolist()})
            self.next_imu = ts + TOY["imu_interval_s"] + TOY["imu_jitter_s"] * (self.rng.random() - 0.5)
        while self.next_image <= t + 1e-12:
            ti = self.next_image
            self.image_no += 1
            n = self.image_no
            p, q = self.pose_at(ti)
            pc = p + qrot(q, TOY["nav_camera_position_m"]) + TOY["fix_noise_m"] * self.rng.standard_normal(3)
            qc = RL.qnormalize(qmul(RL.qexp(TOY["fix_noise_rad"] * self.rng.standard_normal(3)), qmul(q, self.q_nav)))
            if qc[3] < 0:
                qc = -qc
            lost = n % TOY["every_nth_image_not_localised"] == 0
            self.pending.append((ti + TOY["registration_delivery_s"], "registrations",
                                 {"camera_id": n, "stamp": round(ti + TOY["registration_offset_s"], 9)}))
            self.pending.append((ti + TOY["fix_delay_s"], "fixes",
                                 {"camera_id": n, "stamp": round(ti, 9), "landmarks": 0 if lost else 9 + (7 * n) % 45,
                                  "position": None if lost else pc.tolist(), "attitude": None if lost else qc.tolist()}))
            self.next_image = ti + TOY["image_interval_s"]
        regs, fixes, keep = [], [], []
        for arr, kind, e in sorted(self.pending, key=lambda x: x[0]):
            if arr <= t + 1e-12:
                (regs if kind == "registrations" else fixes).append(e)
            else:
                keep.append((arr, kind, e))
        self.pending = keep
        target = float(TOY["impeller_rad_s_per_level"][str(self.level)])
        states = ["ready" if abs(s - target) <= TOY["ready_within_rad_s"] else "ramping" for s in self.speed]
        noz = [int(max(0, min(255, round(128 + 100.0 * self.last_cmd[i % 6])))) for i in range(12)]
        return {"t": round(t, 9), "in_control": bool(in_control), "job_id": job_id, "mode": mode,
                "fan_level_commanded": int(self.level), "inertial": ins, "registrations": regs, "fixes": fixes,
                "propulsion": {"impeller_speed": [float(s) for s in self.speed], "module_state": states,
                               "nozzle_command": noz},
                "camera": {"busy": bool(t < self.camera_busy_until - 1e-12)},
                "block": {"budget_s": budget, "time_left_s": left}}


# ------------------------------------------------------------------------------------------------ the toy session
def empty_lists(fr):
    fr = dict(fr)
    fr["inertial"], fr["registrations"], fr["fixes"] = [], [], []
    return fr


class ToySession(object):
    def __init__(self, cfg, node):
        self.cfg = cfg
        self.node = node
        self.paper = toy_session(cfg)
        self.plant = ToyPlant(cfg, self.paper["start_pose"])
        self.mode = "nominal"
        self.cmd = np.zeros(6)
        self.last_applied_t = 0.0
        self.jobs = []
        self.phases = []
        self.block_records = []
        self.ended = None

    def reply_wrench(self, rep):
        """the wrench a reply commands; None for a protocol error (a non-finite wrench); no new command keeps the last."""
        if rep is None or rep.get("force") is None:
            return self.cmd
        w = np.concatenate([np.asarray(rep["force"], float).reshape(3), np.asarray(rep["torque"], float).reshape(3)])
        if not np.all(np.isfinite(w)):
            return None
        self.last_applied_t = self.plant.t
        return w

    def outside(self):
        z = self.paper["zones"]["keep_in"][0]
        return bool(np.any(self.plant.p < np.asarray(z[0])) or np.any(self.plant.p > np.asarray(z[1])))

    def idle(self, dur, job_id, phase):
        """steps out of control (the start hold, between blocks): frames sent, replies not applied, the toy at rest."""
        t_end = self.plant.t + dur
        n = 0
        while self.plant.t < t_end - 1e-9:
            self.node.step(self.plant.frame(False, job_id, self.mode, None, None))
            self.plant.v[:] = 0.0
            self.plant.w[:] = 0.0
            self.plant.advance(np.zeros(6))
            n += 1
        self.phases.append({"phase": phase, "steps": n})

    def checkout(self):
        p0 = self.plant.p.copy()
        t_end = self.plant.t + TOY["checkout_s"]
        out, n = False, 0
        while self.plant.t < t_end - 1e-9:
            rep = self.node.step(self.plant.frame(not out, "checkout", self.mode, None, None))
            if not out:
                w = self.reply_wrench(rep)
                if w is None:
                    out = "protocol error"
                    w = np.zeros(6)
                self.cmd = w
            self.plant.advance(self.cmd if not out else np.zeros(6))
            if not out and np.linalg.norm(self.plant.p - p0) > TOY["checkout_box_m"]:
                out = "left the toy box"
            n += 1
        self.phases.append({"phase": "checkout", "steps": n, "node_out_of_control": out or None})

    def block(self, b, mode_before):
        B = float(b["budget_s"])
        t0 = self.plant.t
        if b["mode"] != mode_before:
            self.mode = b["mode"]
            self.plant.level = self.cfg.v("mode.fan_level")[b["mode"]]
        prev = (np.asarray(b["start_pose"]["position"], float), RL.qnormalize(b["start_pose"]["attitude"]))
        self.plant.teleport(prev)
        self.last_applied_t = self.plant.t
        over = None
        for job in b["jobs"]:
            rec = {"block": b["name"], "id": job["id"], "kind": job["kind"], "mode": job["mode"], "t_start": self.plant.t}
            self.jobs.append(rec)
            if over or self.ended or self.plant.t - t0 >= B - 1e-9:
                rec["status"] = "not_started"
                continue
            fr = self.plant.frame(True, job["id"], self.mode, B, B - (self.plant.t - t0))
            ans = self.node.plan(job, fr)
            verdict = (ans.get("verdict") or {}).get("outcome")
            plog = self.node.log["plans"][-1]
            rec["plan"] = {k: plog.get(k) for k in ("outcome", "rule", "setpoint", "detail", "cpu_s", "wall_s")
                           if plog.get(k) is not None}
            should_refuse = job["mode"] not in self.paper["release_modes"] or (
                job["kind"] == "view" and np.any(np.abs(np.asarray(job["goal"]["centre"]) -
                                                        np.asarray(self.paper["start_pose"]["position"])) >
                                                 TOY["zone_half_m"]))
            if ans.get("refuse"):
                rec["status"] = "refused (%s)" % ans["refuse"]
                rec["should_refuse"] = bool(should_refuse)
                continue
            if should_refuse:
                rec["accepted_a_request_to_refuse"] = True
            if job["mode"] not in self.paper["release_modes"]:
                rec["status"] = "accepted, mode not listed: not flown"
                continue
            if verdict == "protocol_error":
                rec["status"] = "protocol error in the plan reply: the block ends"
                over = "protocol"
                continue
            if verdict == "invalid" and job["kind"] == "hold":
                rec["status"] = "invalid hold: not flown (toy: the body stays at the hold's pose)"
                continue
            if job["kind"] == "hold":
                sps = None
            else:
                sps = ans.get("setpoints") or []
                if verdict != "valid":
                    rec["status"] = "invalid plan: not flown (toy: the body is put at the goal)"
                    goal = CK.body_goal(job, self.cfg.camera_mount)
                    self.plant.teleport(goal)
                    prev = goal
                    continue
            res = self.fly(job, rec, sps, prev, B, t0)
            rec["status"] = res
            if res in ("budget", "watchdog", "protocol", "left the toy zones"):
                over = res
                if res == "left the toy zones":
                    self.ended = res
            if job["kind"] != "hold":
                prev = CK.body_goal(job, self.cfg.camera_mount)
        self.phases.append({"phase": b["name"], "budget_s": B, "used_s": self.plant.t - t0, "ended_by": over})
        ended_by = over or ("budget" if self.plant.t - t0 >= B - 1e-9 else "jobs_done")
        self.block_records.append({"name": b["name"], "t_start": t0, "t_end": self.plant.t, "ended_by": ended_by})
        self.idle(TOY["between_s"], None, "after " + b["name"])

    def fly(self, job, rec, sps, prev, B, t0):
        kind = job["kind"]
        t_job = self.plant.t
        win_end = t_job + float(job["goal"]["duration_s"]) if kind == "hold" else None
        first = True
        goal = prev if kind == "hold" else CK.body_goal(job, self.cfg.camera_mount)
        while True:
            el = self.plant.t - t0
            if el >= B - 1e-9:
                return "budget" if (win_end is None or self.plant.t < win_end - 1e-9) else "finished"
            if win_end is not None and self.plant.t >= win_end - 1e-9:
                return "finished"
            fr = self.plant.frame(True, job["id"], self.mode, B, B - el)
            if first:
                fr = empty_lists(fr)
                first = False
            rep = self.node.step(fr)
            w = self.reply_wrench(rep)
            if w is None:
                rec["protocol_error_t"] = self.plant.t
                return "protocol"
            if self.plant.t - self.last_applied_t > self.cfg.watchdog_s + 1e-9:
                return "watchdog"
            ev = (rep or {}).get("event") or "none"
            if ev in ("done", "ready") and win_end is None and sps:
                if self.plant.t >= sps[-1][0] - 1e-9 and ((ev == "done" and kind in ("leg", "slew")) or
                                                          (ev == "ready" and kind == "view")):
                    rec["t_event"] = self.plant.t
                    e_p = float(np.linalg.norm(self.plant.p - goal[0]))
                    e_a = math.degrees(RL.qangle(self.plant.q, goal[1]))
                    rec["toy_error_at_event"] = {"position_m": e_p, "attitude_deg": e_a}
                    if kind == "view":
                        win_end = self.plant.t + float(self.cfg.v("jobs.capture_s"))
                        self.plant.camera_busy_until = win_end
                    else:
                        win_end = self.plant.t + float(self.cfg.v("score.end_window_s")[1])
            self.cmd = w
            self.plant.advance(w)
            if self.outside():
                return "left the toy zones"
            if kind == "hold" and self.plant.t >= win_end - 1e-9:
                rec["toy_drift_at_end"] = {"position_m": float(np.linalg.norm(self.plant.p - goal[0])),
                                           "attitude_deg": math.degrees(RL.qangle(self.plant.q, goal[1]))}

    def run(self):
        self.node.setup(self.paper)
        if self.node.session_failed:
            self.ended = "setup failed"
            return
        self.idle(TOY["start_hold_s"], None, "start hold")
        self.checkout()
        before = self.paper["checkout_mode"]
        for b in self.paper["blocks"]:
            if self.ended:
                break
            self.block(b, before)
            before = b["mode"]


# ------------------------------------------------------------------------------------------------ the report
def config_with(printed):
    base = os.path.join(HERE, "wire", "runner_config.json")
    if not printed:
        return CF.Config(base), []
    with open(base) as f:
        raw = json.load(f)
    with open(printed) as f:
        given = json.load(f)
    used = []
    for k, v in given.items():
        if k not in raw["entries"] or not raw["entries"][k].get("toy"):
            raise SystemExit("--printed: %s is not one of the toy values" % k)
        raw["entries"][k] = {"value": v, "unit": raw["entries"][k]["unit"], "toy": False, "from": "--printed"}
        used.append(k)
    if "jobs.capture_s" in given:
        raw["derived"]["fixed_term_view_s"] = {"value": float(given["jobs.capture_s"]), "from": "jobs.capture_s"}
    fd, path = tempfile.mkstemp(suffix=".json")
    with os.fdopen(fd, "w") as f:
        json.dump(raw, f)
    cfg = CF.Config(path)
    os.remove(path)
    return cfg, used


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("controller_dir")
    ap.add_argument("--json")
    ap.add_argument("--printed")
    ap.add_argument("--log")
    ap.add_argument("--run-dir")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()
    prog = os.path.abspath(a.controller_dir)
    if os.path.isfile(prog):
        prog = os.path.dirname(prog)
    if not os.path.isfile(os.path.join(prog, "controller.py")):
        print("no controller.py in %s" % prog)
        return 2
    cfg, used = config_with(a.printed)
    base = a.run_dir or tempfile.mkdtemp(prefix="selftest_")
    run_root = os.path.join(base, "run")
    log = a.log or os.path.join(base, "controller_stderr.log")
    node = ND.WireNode(prog, run_root, cfg=cfg, sandbox_required=False, stderr_path=log)
    S = ToySession(cfg, node)
    node.read_block_ends_from(S)
    S.run()
    rep = node.close()
    L = node.log
    out = {"toy": True, "printed_values_used": used,
           "toy_values": sorted(k for k, v in cfg.E.items() if v.get("toy")),
           "sandbox": rep["runner"].get("sandbox"), "setup": L.get("setup"), "session_failed": rep["session_failed"],
           "block_budgets": L["budgets"], "phases": S.phases, "jobs": S.jobs, "plans": L["plans"],
           "events": L["events"], "protocol_errors": L["protocol"], "late_steps": L["late"],
           "overruns": L["overruns"], "restarts": L["restarts"], "starts": L["starts"],
           "set_up_allowance_used": rep["set_up_allowance_used"], "cuts": L["cuts"], "block_ends": L["block_ends"],
           "stretches": rep["stretches"], "rules_applied": rep["rules_applied"], "steps": rep["steps"],
           "processes_left_in_the_group_at_the_end": rep["runner"].get("stray_processes"),
           "deaths_of_the_controller_process": rep["runner"].get("deaths"),
           "controller_log": log, "ended": S.ended}
    bad = bool(rep["session_failed"] or L["protocol"] or L["overruns"] or L["restarts"] or L["cuts"])
    st = rep["steps"]
    summary = [
        "toy session: %s" % ("ended: " + S.ended if S.ended else "flown to its end"),
        "sandbox: %s (required at the acceptance)" % json.dumps(out["sandbox"]),
        "setup: %s" % ("failed: " + str(rep["session_failed"]) if rep["session_failed"] else "ok"),
        "plans: " + ", ".join("%s %s" % (p["job_id"], p.get("outcome") + (" [" + p["rule"] + "]" if p.get("rule") else ""))
                              for p in L["plans"]),
        "jobs: " + ", ".join("%s %s" % (j["id"], j.get("status")) for j in S.jobs),
        "events: %d accepted, %d ignored %s" % (sum(1 for e in L["events"] if e["accepted"]),
                                               sum(1 for e in L["events"] if not e["accepted"]),
                                               sorted(set(e["reason"] for e in L["events"] if not e["accepted"]))),
        "protocol errors: %d %s" % (len(L["protocol"]), [p["what"] for p in L["protocol"]][:3]),
        "steps answered %d, late %d, skipped calls %d, on-time share %s (gate %s)" % (
            st["steps_answered"], st["late_steps"], st["step_calls_skipped"],
            None if st["on_time_share"] is None else round(st["on_time_share"], 4), st["on_time_gate"]),
        "processor per step (ms): %s" % (None if not st.get("cpu_ms") else
                                         {k: round(v, 3) for k, v in st["cpu_ms"].items() if k != "t_of_largest_s"}),
        "longest run without an applied command: %s (watchdog %s s)" % (
            st["longest_run_without_applied_command"]["s"], st["longest_run_without_applied_command"]["watchdog_s"]),
        "starts of the controller's process: %d (restarts %d); the one set-up allowance used: %.2f s of processor of "
        "%g, %.2f s of wall of %g" % (len(L["starts"]), len(L["restarts"]), rep["set_up_allowance_used"]["cpu_s"],
                                      cfg.setup_caps[0], rep["set_up_allowance_used"]["wall_s"], cfg.setup_caps[1]),
        "wall rule: %d cuts (%d in a block, %d outside blocks); processes of its group left at the end: %s" % (
            len(L["cuts"]), sum(1 for c in L["cuts"] if c["in_block"]), sum(1 for c in L["cuts"] if not c["in_block"]),
            out["processes_left_in_the_group_at_the_end"]),
    ]
    print("\n".join(summary))
    if not a.quiet:
        for j in S.jobs:
            if j.get("toy_error_at_event") or j.get("toy_drift_at_end"):
                print("  toy %s %s: %s" % (j["id"], j["kind"], json.dumps(j.get("toy_error_at_event") or
                                                                          j.get("toy_drift_at_end"))))
        print("the controller's prints: %s" % log)
    if a.json:
        with open(a.json, "w") as f:
            json.dump(out, f, indent=1, default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x))
    if not a.run_dir:
        PR.unlock_tree(run_root)
        shutil.rmtree(base, ignore_errors=True)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
