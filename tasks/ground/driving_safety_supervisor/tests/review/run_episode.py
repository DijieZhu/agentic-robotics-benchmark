"""Grader-exact episodes of a submission (a maintainer's tool; the grader never runs it).

Runs as root inside a container of the task image, from a root-only copy of tests/ (as the separate verifier stages
it), against a submission at <agent home>/submission (a solve.sh run, say). It does what verify.main does before and
around the episodes - secure_exec_path, the token, own_submission, the netguard self-test and entry, the bridge as the
simulator's account from a copy in /run, one SimClient and one Solver for the whole list - and runs every episode
through verify.grade_episode itself (its own lock stream, the layout recorded, aborts as zeros). The specs are a list
of episodes in the set's form ({id, town, kind, variant, seed, distractors, lock_key[, layout_sha256]}); a missing
layout digest is recorded, not checked.

What it adds, on the grading side only: a recorder around the simulator (per step: the ego's speed, the applied
control, the truth so far, and, in a token run, the smallest box-to-box clearance from the ego to every actor of the
privileged block, sized as the expert sizes them: the larger of the box and solution/footprints.json, and the
walkers within FAST_RADIUS_M of the car that move faster than FAST_WALKER_MPS over the ground), the time of every
reply, and one result line per episode.

    ALE_STAGE_DIR=/opt/ale/tests ALE_HOME=/home/user python3 -B review/run_episode.py --specs SPECS.json --out DIR
        [--footprints /task/solution/footprints.json] [--steps]

DIR/results.jsonl: per episode its spec, truth, score, steps, layout, staging, probes (pairs, moved), the smallest
clearance overall / to props / to vehicles / to walkers (m, real requests only), the fastest walker near the car
(`walker_fast`: [m/s, id, blueprint, distance m, time s]) and the longest run of consecutive steps one walker near the
car moved faster than FAST_WALKER_MPS (`walker_launch`: [steps, peak m/s, id, blueprint, distance m at the peak,
time s at the peak], or None), the slowest reply (s), wall time.
DIR/steps_<id>.jsonl.gz (with --steps): the per-step record.
"""

import argparse
import gzip
import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
TESTS = os.path.dirname(HERE)
sys.dont_write_bytecode = True
sys.path.insert(0, TESTS)

import verify  # noqa: E402
from episode_score import score  # noqa: E402
from robotics_grader import ProbeLock, Solver, SolverAborted, SolverMissing, stage_config  # noqa: E402
from robotics_grader.stage import agent_home, secure_exec_path  # noqa: E402

EGO_HALF_L, EGO_HALF_W = 2.45, 0.92
#: People walk at 0.9-1.7 m/s in every staging; one near the car going faster than this over the ground has been
#: launched by the physics (a crowd member another walked into, say). Its speed is the block's (vx, vy).
FAST_WALKER_MPS, FAST_RADIUS_M = 4.0, 25.0


def load_footprints(path):
    try:
        with open(path) as handle:
            table = json.load(handle).get("footprints", {})
    except (OSError, ValueError):
        return {}
    out = {}
    for name, fp in table.items():
        (x0, x1), (y0, y1) = fp["x"], fp["y"]
        ex, ey = fp["box"]["extent"][:2]
        lx, ly = fp["box"]["location"][:2]
        out[name] = (max(ex, lx - x0), max(ex, x1 - lx), max(ey, ly - y0), max(ey, y1 - ly))
    return out


def _corners(cx, cy, yaw, hx, hy):
    c, s = math.cos(yaw), math.sin(yaw)
    return [(cx + c * dx - s * dy, cy + s * dx + c * dy) for dx, dy in ((hx, hy), (hx, -hy), (-hx, -hy), (-hx, hy))]


def _seg_point(p, a, b):
    ax, ay = a
    bx, by = b
    dx, dy = bx - ax, by - ay
    n = dx * dx + dy * dy
    u = 0.0 if n < 1e-12 else max(0.0, min(1.0, ((p[0] - ax) * dx + (p[1] - ay) * dy) / n))
    return math.hypot(p[0] - ax - u * dx, p[1] - ay - u * dy)


def _inside(p, poly):
    sign = None
    for a, b in zip(poly, poly[1:] + poly[:1]):
        cross = (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
        if abs(cross) < 1e-12:
            continue
        if sign is None:
            sign = cross > 0
        elif (cross > 0) != sign:
            return False
    return True


def box_clearance(a, b):
    """Distance between two convex quadrilaterals (0 when they touch or overlap)."""
    if any(_inside(p, b) for p in a) or any(_inside(p, a) for p in b):
        return 0.0
    best = float("inf")
    for poly, other in ((a, b), (b, a)):
        for p in poly:
            for q, r in zip(other, other[1:] + other[:1]):
                best = min(best, _seg_point(p, q, r))
    return best


class Recorder(object):
    """The simulator, recording every step (grading side: it sees the state before the wire's filter)."""

    def __init__(self, sim, footprints, keep_steps):
        self.sim, self.fp, self.keep = sim, footprints, keep_steps
        self.steps = []
        self.min = {"all": None, "prop": None, "vehicle": None, "walker": None}
        self.speed_limits = []
        self.walker_fast = None      # [m/s, id, blueprint, distance, time]: the fastest walker near the car
        self.walker_launch = None    # [steps, peak m/s, id, blueprint, distance, time]: the longest fast run
        self._fast = {}              # walker id -> its current run of fast steps

    def _walkers(self, state, now):
        """Walkers within FAST_RADIUS_M of the car: the fastest, and runs of consecutive steps faster than
        FAST_WALKER_MPS per walker (real requests only: the probe's pasted road user is not in this block)."""
        priv = state.get("privileged")
        if not priv:
            return None
        fast, top = set(), None
        for a in priv.get("actors", ()):
            if a.get("type") != "walker":
                continue
            dist = math.hypot(a["x"], a["y"])
            if dist > FAST_RADIUS_M:
                continue
            speed = math.hypot(a.get("vx", 0.0), a.get("vy", 0.0))
            if top is None or speed > top[0]:
                top = [round(speed, 2), a["id"], round(dist, 2)]
            if self.walker_fast is None or speed > self.walker_fast[0]:
                self.walker_fast = [round(speed, 2), a["id"], a.get("blueprint"), round(dist, 2), now]
            if speed <= FAST_WALKER_MPS:
                continue
            fast.add(a["id"])
            run = self._fast.setdefault(a["id"], [0, 0.0, a["id"], a.get("blueprint"), None, None])
            run[0] += 1
            if speed > run[1]:
                run[1], run[4], run[5] = round(speed, 2), round(dist, 2), now
            if self.walker_launch is None or (run[0], run[1]) > (self.walker_launch[0], self.walker_launch[1]):
                self.walker_launch = list(run)
        for ident in [i for i in self._fast if i not in fast]:
            del self._fast[ident]
        return top

    def _clearances(self, state):
        priv = state.get("privileged")
        if not priv:
            return None
        ego = _corners(0.0, 0.0, 0.0, EGO_HALF_L, EGO_HALF_W)
        out = {}
        for a in priv.get("actors", ()):
            if math.hypot(a["x"], a["y"]) > 25.0:
                continue
            fp = self.fp.get(a.get("blueprint", ""))
            if fp is None:
                fp = (a["length"] / 2.0, a["length"] / 2.0, a["width"] / 2.0, a["width"] / 2.0)
            back, front, left, right = fp
            yaw = math.radians(a["yaw"])
            c, s = math.cos(yaw), math.sin(yaw)
            ox, oy = (front - back) / 2.0, (right - left) / 2.0
            box = _corners(a["x"] + c * ox - s * oy, a["y"] + s * ox + c * oy, yaw, (front + back) / 2.0,
                           (left + right) / 2.0)
            gap = box_clearance(ego, box)
            kind = a["type"]
            if kind not in out or gap < out[kind][0]:
                out[kind] = (gap, a["id"], a.get("blueprint"))
        return out

    def _note(self, state, mode=None, control=None):
        truth = state.get("truth", {})
        gaps = self._clearances(state)
        if gaps:
            for kind, (gap, _ident, _bp) in gaps.items():
                for key in (kind, "all"):
                    if key in self.min and (self.min[key] is None or gap < self.min[key][0]):
                        self.min[key] = (round(gap, 3), _ident, _bp, truth.get("time_s"))
        walker = self._walkers(state, truth.get("time_s"))
        if state.get("privileged"):
            limit = state["privileged"].get("speed_limit")
            if not self.speed_limits or self.speed_limits[-1] != limit:
                self.speed_limits.append(limit)
        if self.keep:
            self.steps.append({"time_s": truth.get("time_s"), "speed": state["ego"]["speed"],
                               "mode": mode, "control": control, "completion": truth.get("route_completion"),
                               "collisions": truth.get("collisions"), "hazard": truth.get("hazard_collisions"),
                               "harsh": truth.get("harsh_brakes"), "red": truth.get("red_lights"),
                               "red_light": (state.get("privileged") or {}).get("red_light"),
                               "gaps": {k: [round(v[0], 3), v[1]] for k, v in (gaps or {}).items()},
                               "walker": walker})

    def reset(self, spec):
        state = self.sim.reset(spec)
        self._note(state)
        return state

    def step(self, mode, control):
        state = self.sim.step(mode, control)
        self._note(state, mode, [round(v, 3) for v in control])
        return state


class TimedSolver(Solver):
    """The kit's Solver, timing every reply."""

    def __init__(self, *args, **kwargs):
        self.max_s, self.replies = 0.0, 0
        super().__init__(*args, **kwargs)

    def act(self, t, obs):
        asked = time.monotonic()
        try:
            return super().act(t, obs)
        finally:
            self.max_s = max(self.max_s, time.monotonic() - asked)
            self.replies += 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--specs", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--footprints", default=os.path.join(os.path.dirname(TESTS), "solution", "footprints.json"))
    parser.add_argument("--steps", action="store_true")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    with open(args.specs) as handle:
        episodes = json.load(handle)
    footprints = load_footprints(args.footprints)

    secure_exec_path()
    config = stage_config()
    submission = os.path.join(agent_home(), "submission")
    privileged, mismatch = verify.token_state(submission)
    verify.own_submission(submission)
    folder = verify.guarded_entry(config.get("guard_python", "/opt/venv/bin/python3"))
    lock = ProbeLock(None, rate=config.get("probe_rate", 0.02),
                     pass_fraction=config.get("probe_pass_fraction", 0.25))
    t0 = time.time()
    sim, sim_log, bridge_folder = verify.start_simulator(config)
    sys.stderr.write("[run] simulator up in {:.1f}s; privileged={} mismatch={}\n".format(
        time.time() - t0, privileged, mismatch))
    results = open(os.path.join(args.out, "results.jsonl"), "a")
    try:
        try:
            solver = TimedSolver(step_timeout=config.get("step_timeout_s", 10), submission=folder)
        except (SolverMissing, SolverAborted) as exc:
            sys.stderr.write("[run] no solver: {}\n".format(exc))
            return 1
        try:
            for index, ep in enumerate(episodes):
                t1 = time.time()
                rec = Recorder(sim, footprints, args.steps)
                total0, passed0 = lock.total, lock.passed
                solver.max_s = 0.0
                outcome, layout = verify.grade_episode(rec, solver, lock, ep, index, privileged)
                truth = outcome.truth
                line = {"id": ep.get("id"), "spec": {k: ep.get(k) for k in ("town", "kind", "variant", "seed",
                                                                               "distractors", "lock_key")},
                        "aborted": outcome.aborted, "note": outcome.note,
                        "truth": {k: v for k, v in truth.items() if k != "staging"},
                        "staging": truth.get("staging"), "score": round(score(outcome), 6),
                        "layout": layout, "probes": [lock.total - total0, lock.passed - passed0],
                        "clearance": {k: v for k, v in rec.min.items()},
                        "walker_fast": rec.walker_fast, "walker_launch": rec.walker_launch,
                        "speed_limits": rec.speed_limits, "max_reply_s": round(solver.max_s, 3),
                        "wall_s": round(time.time() - t1, 1), "privileged": privileged}
                results.write(json.dumps(line) + "\n")
                results.flush()
                if args.steps:
                    with gzip.open(os.path.join(args.out, "steps_{}.jsonl.gz".format(ep.get("id", index))), "wt") as fh:
                        for row in rec.steps:
                            fh.write(json.dumps(row) + "\n")
                sys.stderr.write("[run] {} {} {} {} score={:.4f} {} rc={} hz={} col={} harsh={} pace={} steps={} "
                                 "clear={} probes={} reply={:.2f}s wall={:.1f}s\n".format(
                                     index, ep.get("id"), ep["town"], ep["variant"], line["score"],
                                     truth.get("termination", outcome.note), truth.get("route_completion"),
                                     truth.get("hazard_collisions"), truth.get("collisions"),
                                     truth.get("harsh_brakes"), truth.get("pace"), truth.get("steps"),
                                     rec.min["all"][0] if rec.min["all"] else None, line["probes"],
                                     solver.max_s, line["wall_s"]))
        finally:
            solver.close()
    finally:
        results.close()
        sim.close()
        import shutil
        shutil.rmtree(folder, ignore_errors=True)
        if bridge_folder:
            shutil.rmtree(bridge_folder, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
