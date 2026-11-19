"""The grader's credit rule for keyed_nut_peg_assembly (grader-only; the commissioning service in the agent's
sandbox has none of it).

keyed_env.py (here, and as a byte copy in the commissioning service) is the robot cell: physics, robot,
controller, sensors, cameras and the two safety stops (an unstable simulation; a ruined nut). It has no
ground-truth accessor and judges no assembly. This module adds, for the grader only:

  * `GradedNutEnv`: `keyed_env.NutEnv` plus the credit bookkeeping of every step and the early end of an
    episode once its kit is complete. The rule is the one instruction.md states: a nut is ASSEMBLED when it
    sits on ITS peg's collar (bottom within SEAT_TOL of the collar top, tilt below SEAT_TILT) for
    RELEASE_STEPS consecutive steps with the fingers open at least RELEASE_OPEN and not touching it, and it
    still sits there at the end; a kit is complete when all its nuts are assembled and none was ruined.
    `episode_score` is 1.0 for a complete kit, else 0.0 (also after a fault). `nuts_assembled` and
    `progress` are diagnostics only.
  * `ground_truth(env, wire_obs)`: the scene state attached as `_gt` to the observations of the token-holding
    reference solution only (verify.py), with the real observation's proprioception copied in, so that the
    reference answers a repeated step identically whatever copy it is shown (it ignores the perturbed copy).

The arithmetic is the one keyed_env.py carried before the split (PR #47 maintainer review, 2026-11-12), in
the same order, so a submission meets the same credit as under the single-file version.
"""

import math

import keyed_env as E

LIFT_HEIGHT = 0.03           # nut bottom above the table while held (stage 1)
SEAT_TOL = 0.003             # nut bottom within this height of the collar top (stage 3)
SEAT_TILT = math.radians(5.0)
RELEASE_OPEN = 0.06          # finger opening (m) counted as released
RELEASE_STEPS = 10
CREDIT_ON_PEG = 0.35
CREDIT_DEPTH = 0.80


class GradedNutEnv(E.NutEnv):
    """NutEnv with the grader's credit rule (see the module docstring)."""

    def reset(self, seed, order):
        self.credit = [0.0] * len(self.specs)
        self.release_count = [0] * len(self.specs)
        self.depth_now = 0.0
        return super().reset(seed, order)

    def stage_now(self, i=0):
        """Instantaneous stage of nut i: 0 none, 1 lifted while held, 2 on its target peg (then
        self.depth_now is the fraction of the groove travelled), 3 seated on it."""
        p, R = self.nut_state(i)
        tilt = math.acos(max(-1.0, min(1.0, R[2, 2])))
        self.depth_now = 0.0
        if self.on_peg(i):
            h = p[2] - (E.TABLE_Z + E.COLLAR_H)
            self.depth_now = max(0.0, min(1.0, (E.GROOVE_TOP - h) / E.GROOVE_TOP))
            if h <= SEAT_TOL and tilt <= SEAT_TILT:
                return 3
            return 2
        held = len(self._finger_touch(i)[1]) >= 2
        if held and p[2] - E.TABLE_Z >= LIFT_HEIGHT:
            return 1
        return 0

    def assembled(self, i):
        """Nut i was assembled (credit 1) and still sits on its target peg's collar, not ruined."""
        return self.credit[i] >= 1.0 and not self.broken[i] and self.stage_now(i) == 3

    def _post_step(self):
        super()._post_step()                 # safety stops: fault, ruined nuts, the step limit
        if self.fault is not None:
            return
        d = self.data
        opening = float(d.qpos[self.gq[0]] - d.qpos[self.gq[1]])
        for i in range(self.n):
            if self.broken[i]:
                continue                     # ruined: its credit stays where it was
            st = self.stage_now(i)
            if st >= 1:
                self.credit[i] = max(self.credit[i], 0.25)
            if st >= 2:
                self.credit[i] = max(self.credit[i], CREDIT_ON_PEG + (CREDIT_DEPTH - CREDIT_ON_PEG) * self.depth_now)
            if st == 3 and opening >= RELEASE_OPEN and not self._finger_touch(i)[0]:
                self.release_count[i] += 1
            else:
                self.release_count[i] = 0
            if self.release_count[i] >= RELEASE_STEPS:
                self.credit[i] = 1.0
        if self.early_end and all(self.assembled(i) for i in range(self.n)):
            self.done = True

    def episode_score(self):
        """1.0 if the kit is complete at the end: every nut assembled and still seated on its own peg, and no
        nut ruined; else 0.0 (also after a fault)."""
        if self.fault is not None:
            return 0.0
        return 1.0 if all(self.assembled(i) for i in range(self.n)) and not any(self.broken) else 0.0

    def nuts_assembled(self):
        """Diagnostic: number of nuts assembled and still seated on their target pegs."""
        if self.fault is not None:
            return 0
        return sum(1 for i in range(self.n) if self.assembled(i))

    def progress(self):
        """Diagnostic only (not scored): the mean over the nuts of the best stage each reached (0 after a fault):
        0.25 lifted LIFT_HEIGHT above the table while both fingers held it; 0.35 + 0.45 x d on its own peg, d =
        how far its bottom came down from resting on the pin towards the collar (the best d counts); 1.0
        assembled. A ruined nut keeps the value it had before."""
        if self.fault is not None:
            return 0.0
        return float(sum(self.credit)) / self.n


def actual_groove(env, i):
    """What the reference may know of nut i as it really is, on its own peg (the drawing's values when the episode
    has no tolerances): `risers`, the nut-frame angles (rad, unwrapped along the groove) of the middles of its
    entrance riser and of the risers after the first and the second turn; `stalls`, the heights (m above the collar)
    at which its bottom comes to rest after each of them; `pin_top`, the height (m above the collar) at which it
    rests on the pin; `handle`, the handle's real direction (rad) against the nut frame's +x."""
    u = env.unit
    g = env.specs[i]["groove"]
    wb = E.wall_boundaries(u["nuts"][i] if u else None)
    zb = E.level_boundaries(u["nuts"][i] if u else None)

    def middle(sec):                 # the channel of sectors (sec, sec + 1) lies between boundaries sec and sec + 2
        lo = wb[sec % E.NSEC] + 360.0 * (sec // E.NSEC)
        hi = wb[(sec + 2) % E.NSEC] + 360.0 * ((sec + 2) // E.NSEC)
        return math.radians((lo + hi) / 2.0)

    e, k1, k2 = int(g["entrance"]), int(g["turns"][0]), int(g["turns"][1])
    peg = u["pegs"][env.target_peg[i]] if u else None
    pin_top = E.PIN_Z + E.PIN_HH + ((float(peg["pin_z"]) + float(peg["pin_hh"])) if peg else 0.0)
    ceilings = [zb[E.TURN_LAYERS[0][1] + 1], zb[E.TURN_LAYERS[1][1] + 1], zb[E.RISERS[2][1] + 1]]
    return {"risers": [middle(e), middle(e + k1), middle(e + k1 + k2)],
            "stalls": [max(0.0, pin_top - c) for c in ceilings], "pin_top": pin_top,
            "handle": math.radians(float(u["nuts"][i]["handle"])) if u else 0.0}


def ground_truth(env, wire_obs):
    """The reference solution's `_gt`: nut poses, grooves (the drawing's and, under `actual`, the nut as it really
    is), target pegs, peg poses, and `proprio`, the real observation's q / ee_pos / ee_quat (the reference acts on
    these, never on a perturbed copy)."""
    nuts = []
    for i in range(env.n):
        p, R = env.nut_state(i)
        nuts.append({"nut_pos": p.tolist(), "nut_quat": E.quat_from_mat(R).tolist(),
                     "groove": env.specs[i]["groove"], "actual": actual_groove(env, i),
                     "target_peg": env.target_peg[i],
                     "colour": E.colour_name(env.specs[i]["rgba"]), "stage": int(env.stage_now(i)),
                     "credit": float(env.credit[i]), "broken": bool(env.broken[i]),
                     "assembled": bool(env.assembled(i))})
    return {"nuts": nuts, "pegs": [[float(v) for v in pp] for pp in env.peg_poses], "t": int(env.t),
            "fault": env.fault,
            "proprio": {k: wire_obs[k] for k in ("q", "ee_pos", "ee_quat")}}
