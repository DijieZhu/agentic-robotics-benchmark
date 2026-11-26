"""The block's budget and clock (jobs.block_budget, jobs.spin_up_term, jobs.block_clock), the wall allowance of a
stretch of step replies (interface.block_wall_rule in its running form) and the runner's count of the watchdog
(interface.watchdog_s). Numbers from config.Config; the budget is computed from the session paper alone (published
quantities and the job list)."""
import math

import numpy as np

from . import rules as RL


def view_body_pose(c, a, u, mount):
    """the body pose that puts the imaging camera's optical centre at c looking along a with u up."""
    a = np.asarray(a, float)
    u = np.asarray(u, float)
    ez = -u
    ey = np.cross(ez, a)
    R = np.stack([a, ey, ez], 1)
    return np.asarray(c, float) - R @ np.asarray(mount, float), _qfrom_mat(R)


def _qfrom_mat(R):
    t = R[0, 0] + R[1, 1] + R[2, 2]
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        q = [(R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s, 0.25 * s]
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        q = [0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s, (R[2, 1] - R[1, 2]) / s]
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        q = [(R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s, (R[0, 2] - R[2, 0]) / s]
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        q = [(R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s, (R[1, 0] - R[0, 1]) / s]
    q = RL.qnormalize(np.array(q))
    return q if q[3] >= 0 else -q


def body_goal(job, mount):
    """the body pose that meets a job's goal (a view: the pose the camera mounting gives)."""
    if job["kind"] == "view":
        g = job["goal"]
        return view_body_pose(g["centre"], g["axis"], g["up"], mount)
    return np.asarray(job["goal"]["position"], float), RL.qnormalize(job["goal"]["attitude"])


def trap(x, A, V):
    if x <= V * V / A:
        return 2.0 * math.sqrt(x / A)
    return x / V + V / A


def block_budget(block, cfg, release_modes, mode_before=None):
    """B = S + sum over every announced job of (k t_j + F_j) (jobs.block_budget) from the paper's block; returns
    (B, per-job [t_j, F_j], S). mode_before: the mode in force before the block (the paper's previous block's mode;
    None = no mode change)."""
    mode = block["mode"]
    k = cfg.budget_factor
    mount = cfg.camera_mount
    prev_p = np.asarray(block["start_pose"]["position"], float)
    prev_q = RL.qnormalize(block["start_pose"]["attitude"])
    S = cfg.spin_up_s(mode_before, mode) if (mode_before is not None and mode_before != mode) else 0.0
    total = S
    terms = []
    for job in block["jobs"]:
        jm = job["mode"] if job["mode"] in release_modes else mode
        v, a, w, al = cfg.mode_limits(jm)
        if job["kind"] == "hold":
            t = 0.0
            F = cfg.fixed_term("hold", job["goal"]["duration_s"])
        else:
            p, q = body_goal(job, mount)
            d = float(np.linalg.norm(p - prev_p))
            th = RL.qangle(prev_q, q)
            t = max(trap(d, a, v), trap(th, al, w))
            F = cfg.fixed_term(job["kind"])
            prev_p, prev_q = p, q
        terms.append([t, F])
        total += k * t + F
    return total, terms, S


def paper_budgets(paper, cfg):
    """the runner's own budget of every block of a session paper, and whether each agrees with the paper's."""
    out = []
    before = paper.get("checkout_mode", "nominal")
    for b in paper["blocks"]:
        B, terms, S = block_budget(b, cfg, paper["release_modes"], before)
        out.append({"name": b["name"], "runner_s": B, "paper_s": b["budget_s"], "spin_up_s": S,
                    "agree": abs(B - float(b["budget_s"])) <= cfg.eps("budget_match_s")})
        before = b["mode"]
    return out


class BlockClock(object):
    """the block clock read from the frames (jobs.block_clock): a block's clock starts at its first step and runs in
    world time; a frame inside a block carries block.budget_s and block.time_left_s = budget - elapsed. Also the
    stretches the wall rule is applied to (rule wall.stretches): each block is a stretch from its first frame to the
    frame at which the block ends, and the first frame the session sends outside the block opens the next stretch,
    whatever ended the block (rule block.end): its clock reaching B (read from the frame's own fields: the block's
    budget and time left give its start and its end, and every frame from t_start + B on, a recovery's tail after the
    budget included, is outside the block); the step or plan call at which the runner ends it (a protocol error, a
    cut, the watchdog's count at a job's step); the first frame of the next block (a new budget and start); and,
    where the frame cannot show it (the block's last job over before B, the third hand-back, the session's own
    watchdog: the frames of a recovery inside a block and of the gap after it are alike), the session's record of the
    block's end (its start and end time and why it ended), which the grader hands the node. Outside blocks, the start
    hold with the check-out, each gap between blocks and the time after the last block are stretches of their own.
    Each stretch has its own slack."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.block = None            # {"index", "t_start", "B", "ended"}
        self.n_blocks = 0
        self.stretch = None          # {"kind", "t0", "wall", "block"}
        self.stretches = []          # every stretch opened: kind, block, t0, t_end, wall used, why it ended
        self.inconsistent = []

    def on_frame(self, fr, session_end=None):
        """read one frame; session_end = None, or the session's record of a block's end {"t_start", "t_end",
        "ended_by"} (applied only to the block it names, by its start time)."""
        t = float(fr["t"])
        blk = fr.get("block") or {}
        B, left = blk.get("budget_s"), blk.get("time_left_s")
        eps = self.cfg.eps("time_eps_s")
        if B is not None:
            elapsed = float(B) - float(left)
            same = self.block is not None and abs((t - elapsed) - self.block["t_start"]) <= eps and \
                abs(float(B) - self.block["B"]) <= eps
            if not same:
                self.n_blocks += 1
                self.block = {"index": self.n_blocks, "t_start": t - elapsed, "B": float(B), "ended": None}
                self._open("block", self.block["t_start"], self.block["index"], "the next block starts")
            else:
                exp = self.block["B"] - (t - self.block["t_start"])
                if abs(exp - float(left)) > eps:
                    self.inconsistent.append((t, exp, float(left)))
        if self.block is not None and not self.block["ended"]:
            if t >= self.block["t_start"] + self.block["B"] - eps:
                self.block["ended"] = "the block's clock reached its budget"
                self.block["end_read_from"] = "the frame's block fields"
            elif session_end is not None and abs(float(session_end["t_start"]) - self.block["t_start"]) <= eps and \
                    t >= float(session_end["t_end"]) - eps:
                self.block["ended"] = "the session ended the block (%s)" % session_end.get("ended_by")
                self.block["end_read_from"] = "the session's block record"
        if not self.in_block() and (self.stretch is None or self.stretch["kind"] == "block"):
            b = self.block or {}
            why = "%s [%s]" % (b["ended"], b.get("end_read_from")) if b.get("ended") else "the session starts"
            self._open("other", t, None, why)
        return self.stretch

    def in_block(self):
        """True while the frames belong to a block that has not ended."""
        return self.block is not None and not self.block["ended"]

    def end_block(self, why):
        """the block ends at this frame (the runner ended it, or its watchdog count tripped): its stretch ends here
        and the next frame opens the stretch of the gap."""
        if self.in_block():
            self.block["ended"] = why
            self.block["end_read_from"] = "the runner's own end"

    def _open(self, kind, t0, index, why_previous_ended):
        if self.stretch is not None:
            self.stretches[-1].update(t_end=t0, ended_by=why_previous_ended)
        self.stretch = {"kind": kind, "t0": t0, "wall": 0.0, "block": index}
        self.stretches.append({"kind": kind, "block": index, "t0": t0, "t_end": None, "ended_by": None,
                               "wall_s": 0.0})

    def allowance_left(self, t):
        """wall seconds this step reply may still take (interface.block_wall_rule, the running form): the stretch's
        world time up to the end of this step, plus the stretch's slack, minus the wall its replies have used."""
        s = self.stretch
        return (t + self.cfg.step_period_s - s["t0"]) + self.cfg.wall_slack_s - s["wall"]

    def charge(self, wall):
        self.stretch["wall"] += float(wall)
        self.stretches[-1]["wall_s"] = self.stretch["wall"]


class Watchdog(object):
    """the runner's count of the watchdog (interface.watchdog_s, safety.watchdog_stop; rule interface.watchdog_s): the
    time since the last applied command, counted from the instant control returns to the controller (a block's first
    frame; the first frame in control after a stretch out of control, i.e. after a recovery the built-in controller
    flew), checked at every step of a job; a reply given while not in control does not count. Also the longest run of
    consecutive steps in control without an applied command (a frame out of control ends a run)."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.last_applied_t = None
        self.in_control_before = False
        self.run_steps = 0
        self.longest_steps = 0
        self.longest_s = 0.0
        self.trips = []
        self.resets = []

    def on_block_start(self, t):
        self.last_applied_t = t
        self.resets.append((t, "block start"))

    def on_frame(self, t, in_control):
        """control returns to the controller at the first frame in control after frames out of control."""
        if in_control and not self.in_control_before and self.last_applied_t is not None:
            self.last_applied_t = t
            self.resets.append((t, "control returns"))
        self.in_control_before = bool(in_control)

    def on_step(self, t, in_control, applied, in_job):
        if not in_control:
            self.run_steps = 0
            return False
        if applied:
            self.last_applied_t = t
            self.run_steps = 0
        else:
            self.run_steps += 1
            self.longest_steps = max(self.longest_steps, self.run_steps)
            self.longest_s = max(self.longest_s, self.run_steps * self.cfg.step_period_s)
        if in_job and self.last_applied_t is not None and \
                t - self.last_applied_t > self.cfg.watchdog_s + self.cfg.eps("time_eps_s"):
            self.trips.append(t)
            return True
        return False
