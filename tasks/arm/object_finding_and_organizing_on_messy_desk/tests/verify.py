"""The grader. Runs as ROOT in the verify stage, in a fresh verifier sandbox that holds only the
submission (closed_loop).

The grader owns the simulator (an Isaac Sim process it launches from this folder's env.py)
and drives the agent's run.sh step by step over the ALE wire. The score is computed from
the simulator's privileged state only; nothing the solver prints is a scoring input.

The hidden episodes are a fixed set (episodes.json in this folder): the same 64-bit seeds,
in the same order, for every submission, none of them a practice seed. Each episode runs in
a simulator process of its own, so an episode's world does not depend on the ones before it,
and its probe schedule comes from a stream keyed by the set's hidden key and the episode's
seed, so the same submission meets the same probes every time. A probed step is asked twice with the same
`t` (real and perturbed proprioception, hidden order); only the real reply reaches the
simulator. An aborted episode (deadline, crash, malformed action) scores 0 and is never
dropped.

The time budget is the controller's own, never the simulator's: per episode the grader sums
the time it waits for the controller's replies plus a fixed charge per hi-res camera frame
asked for (the simulator's PNG encoding of that frame, at a nominal cost, so that the charge
does not depend on the machine either). An episode whose controller exhausts the budget is cut
there: the steps it never ran earn nothing. A slower simulator therefore only makes the run
longer; it never changes a score.

The reference program's channel: a submission whose `.oracle_token` (a regular file, not a
link) hashes to oracle_token.sha256 in this folder gets the simulator's ground truth in every
observation (`privileged`: every object's identity, pose and state, the containers' poses, the
open request's object, the true proprioception). It exists only to show that the episodes can
be done; nothing else about the run changes, and the metrics record `privileged_mode` and
`oracle_token_mismatch`.
"""

import hashlib
import json
import math
import os
import stat
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from messy_desk_score import EpisodeScorer  # noqa: E402
from robotics_grader import (  # noqa: E402
    ProbeLock,
    Solver,
    SolverAborted,
    SolverMissing,
    fail,
    ratio_rewards,
    read_anchor,
    seed_rng,
    stage_config,
    stage_dir,
    write_verdict,
    zero_verdict,
)
from robotics_grader.stage import agent_home  # noqa: E402
from sim_client import SimClient, SimError  # noqa: E402

ACTION_DIM = 8          # plus an optional ninth element: the hi-res camera bitmask
HIRES_BITS = 15         # 1 agentview, 2 left_view, 4 eye_in_hand, 8 oblique (the env reads these four bits)
TOKEN_FILE = ".oracle_token"
TOKEN_MAX_BYTES = 4096
# what of each object the privileged block carries (the simulator's own state, as scored)
OBJECT_KEYS = ("id", "model", "cls", "pos", "quat", "scale", "state", "in_container", "on_table", "held",
               "contact", "resting")


def _action(reply):
    if not isinstance(reply, (list, tuple)):
        raise SolverAborted("malformed action: {!r}".format(reply)[:160])
    try:
        a = [float(v) for v in reply]
    except (TypeError, ValueError, OverflowError):
        raise SolverAborted("malformed action: {!r}".format(reply)[:160])
    if len(a) not in (ACTION_DIM, ACTION_DIM + 1) or not all(math.isfinite(v) for v in a):
        raise SolverAborted("action must be {} finite numbers, plus an optional ninth (the hi-res "
                            "camera bitmask)".format(ACTION_DIM))
    return a


def _hires_frames(action):
    """How many hi-res camera frames the action's optional ninth element asks for."""
    if len(action) <= ACTION_DIM:
        return 0
    return bin(int(round(action[ACTION_DIM])) & HIRES_BITS).count("1")


def _perturbed(obs, rng):
    """A copy of the observation that an input-reading policy must answer differently:
    proprioception shifted, pixels untouched (re-encoding every frame would cost more
    than it proves). The step index stays the same."""
    p = dict(obs)
    p["joint_pos"] = [v + rng.uniform(-0.15, 0.15) for v in obs["joint_pos"]]
    ee = list(obs["ee_pose"])
    for i in range(3):
        ee[i] += rng.uniform(-0.04, 0.04)
    p["ee_pose"] = ee
    p["gripper_width"] = max(0.0, obs["gripper_width"] + rng.uniform(-0.02, 0.02))
    return p


def _request_log(state):
    """The request events of one step, for the per-episode diagnostic line on the grader's own
    stderr (the solver never sees it): when each request opened, what it asked for, how it closed."""
    out = []
    for e in state.get("events") or []:
        kind = e.get("event", "")
        if kind.startswith("request_"):
            item = {"t": e.get("t"), "id": e.get("request"), "event": kind[len("request_"):]}
            if kind == "request_opened":
                item.update(text=(state.get("request") or {}).get("text"), absent=e.get("absent"))
            out.append(item)
    return out


def load_episodes(path=None):
    """The fixed hidden episode set: (seeds, probe key). A missing or malformed set is a task error."""
    path = path or os.path.join(stage_dir(), "episodes.json")
    try:
        with open(path) as f:
            spec = json.load(f)
        seeds = [int(s) for s in spec["seeds"]]
        key = int(spec["probe_key"])
    except (IOError, OSError, ValueError, KeyError, TypeError) as exc:
        fail("cannot read the hidden episode set {}: {}".format(path, exc))
    if not seeds or any(not 0 <= s < (1 << 64) for s in seeds) or len(set(seeds)) != len(seeds):
        fail("the hidden episode set must hold distinct 64-bit seeds")
    return seeds, key


def token_state(submission, digest_path=None):
    """(privileged, mismatch) for the submission's .oracle_token against oracle_token.sha256.

    Only a regular file counts (opened without following a link, at most TOKEN_MAX_BYTES); a
    token that is there but does not match is recorded as a mismatch and otherwise ignored."""
    digest_path = digest_path or os.path.join(stage_dir(), "oracle_token.sha256")
    path = os.path.join(submission, TOKEN_FILE)
    try:
        with open(digest_path) as f:
            want = f.read().split()[0].strip().lower()
    except (IOError, OSError, IndexError):
        return False, False
    try:
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0))
    except OSError:
        return False, os.path.lexists(path)
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode) or st.st_size > TOKEN_MAX_BYTES:
            return False, True
        data = os.read(fd, TOKEN_MAX_BYTES + 1)
    finally:
        os.close(fd)
    ok = hashlib.sha256(data.strip()).hexdigest() == want
    return ok, not ok


def privileged_block(state, containers, obs):
    """The ground truth the reference program plans with: every present object as the simulator
    scores it, the containers' poses this episode, the open request with its object (None when the
    named model is absent), and the true proprioception (the perturbed copy of a probed step keeps
    this block, so a program that reads its proprioception from here plans the same either way)."""
    req = state.get("request") or None
    request = None
    if req and req.get("status") == "open":
        request = {k: req.get(k) for k in ("id", "model", "object", "text", "opened_t", "deadline_t")}
    # an object the world set down during this step has no pose yet (the simulator reads poses before its events)
    objects = [{k: o.get(k) for k in OBJECT_KEYS} for o in state.get("objects") or []
               if o.get("present") and o.get("pos") is not None]
    return {"t": state.get("t"), "objects": objects, "request": request, "containers": containers,
            "ee_pose": list(obs["ee_pose"]), "joint_pos": list(obs["joint_pos"]),
            "gripper_width": obs["gripper_width"], "ik_ok": state.get("ik_ok")}


def _containers(info):
    return {n: {"xy": [float(c["center_xy"][0]), float(c["center_xy"][1])], "yaw": float(c["yaw"])}
            for n, c in (info.get("containers") or {}).items()}


def _start_sim(config):
    env_script = os.path.join(stage_dir(), "env.py")
    try:
        sim = SimClient(env_script, ready_timeout=float(config.get("sim_ready_timeout_s", 1500)),
                        attempts=int(config.get("sim_start_attempts", 3)),
                        log_path=os.path.join(stage_dir(), "sim.log"))
    except SimError as exc:
        fail("simulator failed to start: {}".format(exc))
    print("[grader] simulator up in {:.0f}s ({} start attempt(s))".format(sim.startup_s, sim.start_attempts),
          file=sys.stderr, flush=True)
    return sim


def run_episodes(config, seeds, probe_key, solver, privileged=False):
    horizon = int(config.get("horizon", 20000))
    budget = float(config.get("controller_budget_s", 3000))
    hires_charge = float(config.get("hires_charge_s", 0.1))
    lock = ProbeLock(None, rate=config.get("probe_rate", 0.01),
                     pass_fraction=config.get("probe_pass_fraction", 0.2))

    scores, summaries, aborted, cut = [], [], 0, 0
    for episode, seed in enumerate(seeds):
        # one probe stream per episode, keyed by the set's hidden key and the episode's seed: the same
        # probes for the same submission every time, whatever ran before and in whatever order
        rng = seed_rng(entropy=(probe_key << 64) | seed)
        lock.rng = rng
        sim = _start_sim(config)
        try:
            try:
                obs, state = sim.reset(seed, horizon)
                containers = _containers(sim.info()) if privileged else None
            except SimError as exc:
                fail("simulator reset failed: {}".format(exc))
            scorer = EpisodeScorer(horizon)
            started = time.monotonic()
            spent = 0.0          # the controller's time this episode: its replies plus the hi-res charge
            hires_frames = 0
            requests = []
            try:
                solver.reset()
                for t in range(horizon):
                    if privileged:
                        obs["privileged"] = privileged_block(state, containers, obs)
                    if lock.due():
                        perturbed = _perturbed(obs, rng)
                        asked = time.monotonic()
                        action = lock.probe(solver, t, obs, perturbed=perturbed)
                    else:
                        asked = time.monotonic()
                        action = solver.act(t, obs)
                    spent += time.monotonic() - asked
                    action = _action(action)
                    frames = _hires_frames(action)
                    hires_frames += frames
                    spent += hires_charge * frames
                    try:
                        obs, state = sim.step(action)
                    except SimError as exc:
                        fail("simulator step failed: {}".format(exc))
                    scorer.add_step(state)
                    requests += _request_log(state)
                    if spent > budget:
                        cut += 1
                        print("[grader] episode {} cut at t={} (controller budget {:.0f}s used up)".format(
                            episode, t + 1, budget), file=sys.stderr, flush=True)
                        break
            except SolverAborted as exc:
                aborted += 1
                scores.append(0.0)
                summaries.append({"aborted": str(exc)[:160], "steps": scorer.steps, "controller_s": round(spent, 1)})
                print("[grader] episode {} aborted at t={} ({}); scored 0".format(
                    episode, scorer.steps, exc), file=sys.stderr, flush=True)
                continue
            scores.append(scorer.score())
            summaries.append(dict(scorer.summary(), controller_s=round(spent, 1), hires_frames=hires_frames,
                                  episode_wall_s=round(time.monotonic() - started, 1)))
            print("[grader] episode {}: score={:.4f} {}".format(episode, scorer.score(),
                  json.dumps(scorer.summary(), sort_keys=True)), file=sys.stderr, flush=True)
            # Printed after the episode ends, so a failed hidden episode can be replayed offline.
            print("[grader] episode {}: requests={}".format(episode, json.dumps(requests)),
                  file=sys.stderr, flush=True)
        finally:
            sim.close()
    return scores, summaries, aborted, cut, lock


def main():
    config = stage_config()
    anchor, anchor_meta = read_anchor()
    seeds, probe_key = load_episodes()
    submission = os.path.join(agent_home(), "submission")
    # read before any of the submission's code runs
    privileged, mismatch = token_state(submission)
    if privileged or mismatch:
        print("[grader] {}".format("privileged mode (oracle token)" if privileged else "oracle token mismatch: ignored"),
              file=sys.stderr, flush=True)
    flags = {"privileged_mode": float(privileged), "oracle_token_mismatch": float(mismatch)}

    try:
        solver = Solver(step_timeout=config.get("step_timeout_s", 30), submission=submission)
    except (SolverMissing, SolverAborted) as exc:
        # An untouched sandbox, or a submission that never comes up, is a real all-zero
        # verdict, not a task error; and no simulator is launched for it.
        write_verdict(zero_verdict(reason=str(exc), anchored=anchor is not None, extra=flags))
        return

    try:
        scores, summaries, aborted, cut, lock = run_episodes(config, seeds, probe_key, solver, privileged=privileged)
    finally:
        solver.close()

    measured = sum(scores) / len(scores) if scores else 0.0
    extra = dict(lock.rewards(), episodes_aborted=float(aborted), episodes_cut=float(cut), **flags)
    for i, s in enumerate(scores):
        extra["episode_{}_score".format(i)] = float(s)
    for i, s in enumerate(summaries):
        extra["episode_{}_controller_s".format(i)] = float(s.get("controller_s", 0.0))
    for key in ("requests_success", "requests_failed", "requests_delivered", "requests_expired", "absent_correct",
                "absent_wrong", "claims_wrong", "first_holds", "objects_lost"):
        extra["total_" + key] = float(sum(s.get(key, 0) for s in summaries))
    for key in ("organization", "request_rate", "request_progress"):
        vals = [s[key] for s in summaries if key in s]
        extra["mean_" + key] = float(sum(vals) / len(vals)) if vals else 0.0
    verdict = ratio_rewards(measured, anchor, cap=config.get("scoring_cap", 1.5),
                            higher_is_better=anchor_meta.get("direction", "higher") == "higher",
                            metric_name="mean_episode_score", locks_ok=lock.ok, extra=extra,
                            full_at=anchor_meta.get("full_at", 1.0))
    write_verdict(verdict)
    print("[grader] measured={:.4f} anchor={} ratio={:.3f} reward={:.3f} lock={} aborted={} cut={} privileged={}".format(
        measured, anchor, verdict["metrics"]["ratio"], verdict["rewards"]["reward"], lock.ok, aborted, cut,
        privileged), file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
