"""Cabinet door hanging: an episode and its credit.

An episode starts with the old door hung on two old pins (door_sim). It ends after EPISODE_STEPS control steps; or
once the new door hangs with a pin of the unit's class home in each hinge, the cup has let go, the gripper is open at
least OPEN m and both tools are at least CLEAR m in front of the door's front face, for HOLD steps in a row; or when
the old door falls off (its centre of mass below the table top less 5 cm: door_sim.World.old_fell; credit 0); or at an
unstable simulation (credit 0).

The credit, counted from the simulation at the end (only the new door counts):
- PINNED per hinge (two hinges): the door hangs on the opening (its frame within 2 mm of the hinge axis and the hung
  height, its leaf within 2 deg of upright and 3 deg of closed: door_sim.World.door_hung) and a pin of the unit's
  class is home in that hinge (on the axis within 1.5 mm, tilted at most 3 deg, its head at most 3 mm above the
  hinge's top face). A pin of another class does not count.
- SWINGS: both hinges pinned, then the swing test (door_sim.World.swing_test: the arms are parked out of the way, a
  test hand turns the door about the hinge axis with at most SWING_TORQUE N m toward 70 deg for 2 s, back to closed,
  and lets it settle) opens it at least SWING_MIN_DEG, and afterwards the door still hangs closed and both hinges are
  still pinned.
An untouched run scores 0 (the new door stays in its rack).
"""

import door_sim as S

EPISODE_STEPS = 6000            # 10 simulated minutes
HOLD = 20
OPEN = 0.07
CLEAR = 0.08
PINNED, SWINGS = 0.25, 0.5      # per hinge, and once
SWING_TORQUE = 0.3
SWING_MIN_DEG = 60.0


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
        if w.held is not None or not w.door_hung() or not all(w.hinges_pinned()):
            return False
        if w.arms["l"].gripper() < OPEN:
            return False
        face = -S.DOOR_T / 2                                   # the closed door's front face, in the carcass frame
        return all(w.to_carcass(a.pos())[0] < face - CLEAR for a in w.arms.values())

    def step(self, action):
        obs = self.env.step(action)
        if self.env.unstable() or self.env.w.old_fell:
            self.done = True
        self.hold = self.hold + 1 if self.finished() else 0
        if self.hold >= HOLD or self.env.t >= EPISODE_STEPS:
            self.done = True
        return obs, self.done, {}

    def score(self):
        env, w = self.env, self.env.w
        unstable = env.unstable()
        lost = unstable or w.old_fell
        hung = bool(w.door_hung())
        pinned = [bool(hung and p) for p in w.hinges_pinned()]
        out = {"door_hung": hung, "pinned": pinned, "pins_home": [w.pin_home(k) for k in range(w.npin)],
               "pin_class": list(w.pin_class), "cup_holding": w.held is not None,
               "cup_peak": [round(float(x), 2) for x in w.cup_peak], "steps": int(env.t), "unstable": unstable,
               "old_fell": bool(w.old_fell)}
        credit = PINNED * sum(pinned)
        swing = None
        if all(pinned) and not lost:
            swing = w.swing_test(torque=SWING_TORQUE)
            swung = bool(swing["opened_deg"] >= SWING_MIN_DEG and swing["door_hung"] and all(swing["hinges_pinned"]))
            credit += SWINGS * swung
            out["swings"] = swung
        else:
            out["swings"] = False
        out["swing"] = swing
        if lost:
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

