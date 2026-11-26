"""Pipe spool push-fit assembly: an episode and its credit.

An episode starts with the pipes and the elbows on the bench (spool_sim). It ends after EPISODE_STEPS
control steps; or once three joints are home, both grippers are open at least OPEN m, both grip sites are at least
CLEAR m from every spool part and every part is slower than STILL m/s, for HOLD steps in a row; or at an unstable
simulation (credit 0).

The credit, counted from the simulation at the end:
- JOINT for each joint (at most three) that is home (the pipe's end within 1 mm of the socket's stop), was made in the
  air (spool_sim: no part of its connected spool touched anything but the grippers and the other spool parts while it
  engaged or while its pipe advanced), with every spool part's centre of mass above the bench top;
- SHAPE when the spool is the work order's: A's two ends in the two elbows, B's one end in one of them, every joint
  counting (three), B's elbow on A's banded end, and the twist (spool_sim.World.twist) within TWIST_TOL deg of the
  order's.
An untouched run scores 0 (the parts stay where they lie).
"""

import spool_sim as S

EPISODE_STEPS = 6000            # 10 simulated minutes
HOLD = 20
OPEN = 0.07
CLEAR = 0.08
STILL = 0.02
JOINT, SHAPE = 0.2, 0.4         # per joint (three), and once
TWIST_TOL = 5.0


def twist_error(measured, wanted):
    return abs((float(measured) - float(wanted) + 180.0) % 360.0 - 180.0)


class Episode:
    def __init__(self, env, seed):
        self.env, self.seed = env, seed

    def __getattr__(self, k):
        return getattr(self.env, k)

    def reset(self):
        self.hold = 0
        self.done = False
        return self.env.reset(self.seed)

    def finished(self):
        w = self.env.w
        if sum(1 for j in w.joints() if j[3]) < 3:
            return False
        if any(a.gripper() < OPEN for a in w.arms.values()):
            return False
        if any(w.hand_clearance(s) < CLEAR for s in w.arms):
            return False
        return w.part_speed() < STILL

    def step(self, action):
        obs = self.env.step(action)
        if self.env.unstable():
            self.done = True
        self.hold = self.hold + 1 if self.finished() else 0
        if self.hold >= HOLD or self.env.t >= EPISODE_STEPS:
            self.done = True
        return obs, self.done, {}

    def score(self):
        env, w = self.env, self.env.w
        unstable = env.unstable()
        on_bench = bool(w.on_bench())
        joints = w.joints()
        counted = [j for j in joints if j[3] and not j[4]] if on_bench else []
        shape = w.twist()
        out = {"joints": [{"socket": int(k), "pipe": p, "end": int(sg), "home": bool(h), "on_support": bool(dt),
                           "depth_mm": round(1000 * w.depth(k), 2)} for k, p, sg, h, dt in joints],
               "joints_counted": len(counted), "on_bench": on_bench, "steps": int(env.t), "unstable": unstable,
               "order_twist_deg": int(w.episode["twist_deg"]), "shape": shape,
               "peak_push": [round(float(x), 1) for x in w.peak_push]}
        credit = JOINT * min(3, len(counted))
        shaped = bool(shape is not None and len(counted) == 3 and shape["banded_in_b_elbow"]
                      and twist_error(shape["twist_deg"], w.episode["twist_deg"]) <= TWIST_TOL)
        out["shape_ok"] = shaped
        credit += SHAPE * shaped
        if unstable:
            credit = 0.0
        out["credit"] = round(float(credit), 4)
        return out


def hold_action():
    return {"l": [0.0] * 6 + [-1.0], "r": [0.0] * 6 + [-1.0], "cams": 0}


def run(env, seed, policy=None):
    """An episode with `policy(obs) -> action` (default: hold still): returns the score."""
    ep = Episode(env, seed)
    obs = ep.reset()
    done = False
    while not done:
        obs, done, _ = ep.step(hold_action() if policy is None else policy(obs))
    return ep.score()
