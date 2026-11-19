"""Pantry jar refilling: the grader's episode loop over the closed_loop wire (used by tests/verify.py).

- The wire: READY, then reset / act / close, one JSON object per line; an action is {"l": [...], "r": [...],
  "cams": mask} (`jar_env.py`). A malformed action, a missed deadline or a closed wire aborts the episode (credit 0);
  the solver restarts the controller at the next reset (at most 3 restarts).
- The probe lock (`IdleAwareProbeLock`): a step is asked twice with the same `t`, the true and a perturbed
  observation (`JarEnv.perturbed_observation`) in a hidden order, and only the true reply is applied. Which steps,
  the order and the perturbation come from one stream per episode (`probe_stream`), a function of the episode's index
  in the set and its seed alone.
- Images are rendered for the cameras an action's `cams` asks for (the next observation carries them).
- `truth` (`JarEnv.truth()`) is added to every observation only in a token run (the caller decides).
- The budget (`grader_budget_s`) is the controller's own time, kept by a `Clock`: what the grader waits for READY, for
  each reset and for each reply, plus `image_cost_s` for every image an applied action asks for (a fixed price; the
  first observation's five images are free). The simulation and the drawing of the images are not charged, so the
  budget does not depend on the grader's machine or its load. The episode running when it runs out is aborted and
  the episodes after it are not run; both score 0 and neither counts as scored. A simulator fatal error
  (`mujoco.FatalError`) or an unstable simulation ends that episode at 0 and grading goes on. A renderer failure
  (`jar_env.RenderError`) propagates: it is the grader's own error, not the controller's.
"""

import hashlib
import math
import time

import mujoco
import numpy as np

import episode as EP
import jar_env as E
import jar_sim as S
from robotics_grader import ProbeLock, SolverAborted, seed_rng

STILL = 1e-9                # an arm's 7-number step whose six motion numbers are all within this of zero: no motion
PROBE_TAG = b"pantry_jar_refilling verify probe schedule 1"   # probe_stream's domain tag
CONFIG = {"probe_rate": 0.01, "probe_pass_fraction": 0.1, "step_timeout_s": 30, "grader_budget_s": None,
          "image_cost_s": 0.2}
MAX_LINE = 1 << 22          # a reply line longer than 4 MiB closes the wire (the episode aborts); an action is ~300 B
SELFTEST_SEED = 1           # the render self-test's episode (not a graded seed)


class Clock(object):
    """The controller's own time (s): what the grader waited for READY and for each reply (`timed`), and what
    `run_episodes` adds for the images an applied action asks for. The simulation and the drawing of the images
    happen outside `timed`."""

    def __init__(self):
        self.s = 0.0

    def timed(self, fn, *args, **kwargs):
        t0 = time.monotonic()
        try:
            return fn(*args, **kwargs)
        finally:
            self.s += time.monotonic() - t0


def flat_action(action):
    """An action as one list of numbers for the kit's comparison: left, right, cams."""
    if isinstance(action, dict):
        return list(action.get("l", [])) + list(action.get("r", [])) + [action.get("cams", 0)]
    return action


class IdleAwareProbeLock(ProbeLock):
    """The kit's ProbeLock on the two-arm action, except that a pair in which BOTH replies command no motion is not
    counted, neither in the total nor as passed: a controller that holds still, having finished or given up, shows
    nothing either way. An action commands no motion when each arm's list has 7 numbers whose first six are within
    STILL of zero; a joint step (8) or a step that sets a posture (14) can move the arm, so it always counts.

    Which controllers fail it (fewer than `pass_fraction` of the counted pairs differ): any whose reply does not
    depend on the arms' joints, hand poses or wrist forces, the parts the perturbed copy changes (images are not
    perturbed). A replayed action stream does; so does a controller that commands a fixed absolute joint target (an
    8-number joint step to the same q whatever the observation) for most of its moving steps, or one that moves the
    hands by steps computed only from images. Cartesian steps toward a target pose computed from the observed hand
    pose pass."""

    def __init__(self, *args, **kwargs):
        ProbeLock.__init__(self, *args, **kwargs)
        self.idle_skipped = 0

    @staticmethod
    def still(action):
        if not isinstance(action, dict):
            return False
        for s in ("l", "r"):
            a = action.get(s)
            if not isinstance(a, list) or len(a) != 7:
                return False
            for x in a[:6]:
                if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) or abs(x) > STILL:
                    return False
        return True

    def record(self, action_probe, action_real):
        if self.still(action_probe) and self.still(action_real):
            self.idle_skipped += 1
            return False
        return ProbeLock.record(self, flat_action(action_probe), flat_action(action_real))

    def rewards(self):
        out = ProbeLock.rewards(self)
        out["probes_idle_skipped"] = float(self.idle_skipped)
        return out


def probe_stream(index, seed):
    """The probe lock's stream for episode `index` of the run, whose seed is `seed`: the kit's seed_rng keyed by
    BLAKE2b over PROBE_TAG, the index and the seed. The environment draws from the seed only through NumPy generators
    seeded with its two 32-bit halves and a small tag; this hash shares nothing with them."""
    digest = hashlib.blake2b(b"%s|%d|%d" % (PROBE_TAG, int(index), int(seed)), digest_size=32).digest()
    return seed_rng(entropy=int.from_bytes(digest, "little"))


def wire_obs(env, obs, unit_id, truth):
    w = E.to_wire(obs)
    w["unit"] = unit_id
    if truth:
        w["truth"] = env.truth()
    return w


def ask(lock, solver, env, t, obs, unit_id, truth, clock=None):
    """Step `t`'s action: one request, or a probe pair when the lock's stream says so. Every step draws once from
    the stream (lock.due); a probed step draws twice more before either request is written (the perturbation's seed,
    then the pair's order), so the draws never depend on the replies. Only the requests' round trips are charged to
    `clock`."""
    clock = Clock() if clock is None else clock
    w = wire_obs(env, obs, unit_id, truth)
    if lock.due():
        other = env.perturbed_observation(obs, np.random.default_rng(lock.rng.seed(bits=64)))
        pw = wire_obs(env, other, unit_id, truth)
        return clock.timed(lock.probe, solver, t, w, perturbed=pw)
    return clock.timed(solver.act, t, w)


def render_selftest(unit, unit_id, seed=SELFTEST_SEED):
    """Every camera of one episode: the right shape, not flat (std at least 8 levels), mean brightness 20-235.
    Raises E.RenderError."""
    env = E.JarEnv(unit, unit_id, render=True)
    try:
        obs = env.reset(seed)
        for cam in E.CAMS:
            img = obs[cam]
            w, h, _fovy = S.camera_size(cam)
            mean, std = float(img.mean()), float(img.std())
            if tuple(img.shape) != (h, w, 3) or img.dtype != np.uint8 or std < 8.0 or not 20.0 <= mean <= 235.0:
                raise E.RenderError("{} image: shape {}, mean {:.1f}, std {:.1f}".format(cam, tuple(img.shape),
                                                                                       mean, std))
    finally:
        env.close()


def run_episodes(solver, episodes, units, truth, log=None, config=CONFIG, clock=None, on_step=None):
    """Run `episodes` ([(index in the run, unit id, seed)], in order) over one controller. Returns (records, lock); a
    record per episode: index, unit, seed, credit, reason, `scored` (it ran to its end), steps and the score's parts.

    The budget (config `grader_budget_s`) is charged to the controller's own time, `clock` (`lock.clock` afterwards;
    the caller starts it around the solver's start, so the wait for READY is in it): the waits for each reset and each
    reply, plus config `image_cost_s` for every image an applied action asks for. When it is used up the episode
    running is aborted and the rest are skipped (credit 0, not scored). A simulator fatal error or an unstable
    simulation ends that episode (credit 0, not scored, `sim_fault`). A renderer failure (E.RenderError) propagates:
    it is the grader's own error. `on_step()`, if given, runs before every step (the grader's egress watch)."""
    budget = config.get("grader_budget_s")
    image_cost = float(config.get("image_cost_s", 0.0))
    clock = Clock() if clock is None else clock

    def over():
        return budget is not None and clock.s > budget

    lock = IdleAwareProbeLock(None, rate=config["probe_rate"], pass_fraction=config["probe_pass_fraction"],
                              tolerance=1e-4)
    lock.clock = clock
    records = []
    for n, uid, seed in episodes:
        lock.rng = probe_stream(n, seed)
        env = E.JarEnv(units[uid], uid, render=True)
        ep = EP.Episode(env, seed)
        rec = {"index": n, "unit": uid, "seed": int(seed), "credit": 0.0, "reason": "ok", "scored": False,
               "sim_fault": False}
        t0 = time.monotonic()
        if over():
            rec.update(reason="not run: controller time budget exhausted", steps=0, wall=0.0)
            records.append(rec)
            continue
        try:
            obs = ep.reset()
            clock.timed(solver.reset)
            t, done = 0, False
            while not done:
                if over():
                    raise SolverAborted("controller time budget exhausted")
                if on_step is not None:
                    on_step()
                action = ask(lock, solver, env, t, obs, uid, truth, clock)
                clock.s += image_cost * bin(E.check_action(action)["cams"]).count("1")    # the images it asks for
                obs, done, _info = ep.step(action)
                t += 1
                if env.unstable():
                    rec["sim_fault"] = True
                    raise SolverAborted("simulation unstable")
            sc = ep.score()
            rec.update(credit=float(sc["credit"]), score=sc, scored=True)
        except mujoco.FatalError as exc:
            rec.update(credit=0.0, sim_fault=True, reason="simulator fatal error: {}".format(str(exc)[:160]))
        except (SolverAborted, ValueError, TypeError, OverflowError, KeyError, AttributeError, RecursionError) as exc:
            rec.update(credit=0.0, reason="aborted: {}".format(str(exc)[:160]))
        finally:
            env.close()
        rec.update(steps=int(env.t), wall=round(time.monotonic() - t0, 1))
        if log:
            log("episode {} ({}, {}): credit={:.4f} steps={} {}".format(n, uid, seed, rec["credit"], rec["steps"],
                                                                        rec["reason"]))
        records.append(rec)
    return records, lock


def parts(records):
    """Per scored episode, over its listed jars: the fraction done, the fraction holding exactly the listed pieces,
    the fraction closed again, and whether the episode was halved."""
    out = {"done": [], "filled": [], "sealed": [], "halved": []}
    for r in records:
        sc = r.get("score")
        if not r.get("scored") or not sc:
            continue
        jars = list(sc["jars"].values())
        out["done"].append(sum(1.0 for j in jars if j["done"]) / len(jars))
        out["filled"].append(sum(1.0 for j in jars if j["filled"]) / len(jars))
        out["sealed"].append(sum(1.0 for j in jars if j["sealed"]) / len(jars))
        out["halved"].append(1.0 if sc["penalized"] else 0.0)
    return out
