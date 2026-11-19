"""Lab sample handling: an episode and its credit.

An episode ends after EPISODE_STEPS control steps, LID_HOLD steps after the lid is shut (once it has been open:
in some episodes it starts closed), or at an unstable simulation (credit 0).

The credit, from the simulation at the end. The job is the centrifuge run, so the credit is the run's:
- The run counts only if the centrifuge could be started on it: the lid is shut, and every tube in the rotor is
  one of the run's tubes (the worklist's spun tubes, the tubes already in the rotor), seated at the bottom of its
  hole, with one of them seated in the opposite position (k and k + 4), their masses within BALANCE_G. A rotor with
  a tube that has no partner, an unbalanced pair or a tube that does not belong in it must not be spun.
- A run that counts has the share of the required pairs it holds: the balanced pairs in the rotor over the pairs
  the worklist needs (half the spun tubes plus the tubes already in the rotor). A complete run has share 1.
- The credit is that share times (RUN + COLD x the fraction of the cold tubes seated in their named cold-rack
  positions): a complete run with every cold tube in place scores 1, a complete run with none RUN.
- A run that does not count scores 0, whatever else was done: the cold tubes alone are worth nothing.
An episode in which a tube not to be moved has moved scores 0: another person's tube in the cold rack, a tube that
was already in the rotor, an incoming tube on neither list. An untouched episode scores 0.

(Until 2026-11-18 the credit was a sum: 0.2 for the cold tubes, 0.6 for the pairs, 0.2 for the lid shut over a
complete rotor. It paid 0.2 for the cold tubes of an episode whose run never happened and 0.8 for a rotor loaded
under an open lid.)
"""

import lab_sim as S

EPISODE_STEPS = 20000           # 33 simulated minutes (every load needs a turn to the port; the reference took up
                                # to 16125 steps over 48 benches)
LID_HOLD = 20                   # control steps the lid stays shut before the episode ends
LID_SHUT = 0.02                 # rad, the lid's angle counted as shut
LID_OPENED = 0.5                # rad, the lid's angle counted as having been opened
BALANCE_G = 1.0
RUN, COLD = 0.80, 0.20          # a complete run; the cold tubes, with it


class Episode:
    def __init__(self, env, seed):
        self.env, self.seed = env, seed

    def __getattr__(self, k):
        return getattr(self.env, k)

    def reset(self):
        self.lid_steps = 0
        self.lid_opened = False
        self.done = False
        return self.env.reset(self.seed)

    def step(self, action):
        obs = self.env.step(action)
        if self.env.unstable():
            self.done = True
        ang = self.env.p.lid_angle()
        self.lid_opened = self.lid_opened or ang > LID_OPENED
        self.lid_steps = self.lid_steps + 1 if (self.lid_opened and ang < LID_SHUT) else 0
        if self.lid_steps >= LID_HOLD or self.env.t >= EPISODE_STEPS:
            self.done = True
        return obs, self.done, {}

    def score(self):
        env = self.env
        p, names, ep, src = env.p, env.names, env.episode, env.src
        slots = S.slot_names()
        cold = {slots[i]: bool(S.seated(p, names[i], env.cold[n])) for i, n in ep["cold"].items()}
        # the run's tubes: the worklist's spun tubes and the ones already in the rotor
        rotor = {names[i]: env.masses[i] for i in ep["spin"]}
        rotor.update({b: p.extra_mass[b] for b in p.pre.values()})
        # what stands in the rotor: any tube of the bench seated in a hole
        everyone = [n for n in names if n is not None] + list(p.pre.values()) + list(p.foreign.values())
        in_hole = {}
        for h in range(S.NHOLES):
            for b in everyone:
                if S.seated(p, b, h):
                    in_hole[h] = b
                    break
        half = S.NHOLES // 2
        pairs = []
        for k in range(half):
            i, j = in_hole.get(k), in_hole.get(k + half)
            pairs.append(bool(i in rotor and j in rotor and abs(rotor[i] - rotor[j]) * 1000 <= BALANCE_G))
        need = len(rotor) // 2
        n_pairs = min(sum(pairs), need)
        shut = bool(p.lid_angle() < LID_SHUT)
        balanced = all(pairs[h % half] for h in in_hole)      # every tube in the rotor is half of a counted pair
        startable = bool(shut and balanced and n_pairs >= 1)
        run = n_pairs / need if startable else 0.0
        lid = bool(shut and n_pairs == need)
        stay = [i for i, n in enumerate(names) if n is not None and i not in ep["cold"] and i not in ep["spin"]]
        moved = ([slots[i] for i in stay if not S.seated(p, names[i], src[i])]
                 + [f"rotor {h + 1}" for h, b in p.pre.items() if not S.seated(p, b, h)]
                 + [f"cold {c}" for c, b in p.foreign.items() if not S.seated(p, b, env.cold[c])])
        unstable = env.unstable()
        credit = run * (RUN + COLD * sum(cold.values()) / len(cold))
        if unstable or moved:
            credit = 0.0
        return {"credit": round(float(credit), 4), "cold": cold, "pairs": pairs, "need": need, "lid": lid,
                "startable": startable, "run": round(float(run), 4),
                "moved": moved, "in_hole": {str(h + 1): b for h, b in in_hole.items()}, "unstable": unstable,
                "steps": int(env.t)}


def hold_action():
    return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0]


def run(env, seed, policy=None):
    """An episode with `policy(obs) -> action` (default: hold still): returns the score."""
    ep = Episode(env, seed)
    obs = ep.reset()
    done = False
    while not done:
        obs, done, _ = ep.step(hold_action() if policy is None else policy(obs))
    return ep.score()
