"""Lab sample handling: an episode and its credit.

An episode ends after EPISODE_STEPS control steps, LID_HOLD steps after the lid is shut (once it has been open:
in some episodes it starts closed), or at an unstable simulation (credit 0). The credit, from the simulation at the end:
- the cold tubes, each seated in its named cold-rack position: COLD in all, split evenly;
- the rotor's required pairs (half the worklist's spun tubes plus the tubes already in the rotor): PAIRS in all,
  split evenly; a pair counts when opposite positions (k and k + 4) hold two of those tubes, both seated, their
  masses within BALANCE_G;
- the lid shut over a complete rotor (every required pair counted): LID.
An episode in which a tube not to be moved has moved scores 0: another person's tube in the cold rack, a tube that
was already in the rotor, an incoming tube on neither list. A tube elsewhere counts nothing; an untouched episode
scores 0.
"""

import lab_sim as S

EPISODE_STEPS = 20000           # 33 simulated minutes (every load needs a turn to the port; the reference took up
                                # to 16125 steps over 48 benches)
LID_HOLD = 20                   # control steps the lid stays shut before the episode ends
LID_SHUT = 0.02                 # rad, the lid's angle counted as shut
LID_OPENED = 0.5                # rad, the lid's angle counted as having been opened
BALANCE_G = 1.0
COLD, PAIRS, LID = 0.20, 0.60, 0.20


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
        # the tubes that belong in the rotor: the worklist's spun tubes and the ones already there
        rotor = {names[i]: env.masses[i] for i in ep["spin"]}
        rotor.update({b: p.extra_mass[b] for b in p.pre.values()})
        in_hole = {}
        for h in range(S.NHOLES):
            for b in rotor:
                if S.seated(p, b, h):
                    in_hole[h] = b
                    break
        pairs = []
        for k in range(S.NHOLES // 2):
            i, j = in_hole.get(k), in_hole.get(k + S.NHOLES // 2)
            pairs.append(bool(i is not None and j is not None and abs(rotor[i] - rotor[j]) * 1000 <= BALANCE_G))
        need = len(rotor) // 2
        n_pairs = min(sum(pairs), need)
        lid = bool(p.lid_angle() < LID_SHUT and n_pairs == need)
        stay = [i for i, n in enumerate(names) if n is not None and i not in ep["cold"] and i not in ep["spin"]]
        moved = ([slots[i] for i in stay if not S.seated(p, names[i], src[i])]
                 + [f"rotor {h + 1}" for h, b in p.pre.items() if not S.seated(p, b, h)]
                 + [f"cold {c}" for c, b in p.foreign.items() if not S.seated(p, b, env.cold[c])])
        unstable = env.unstable()
        credit = (COLD * sum(cold.values()) / len(cold) + PAIRS * n_pairs / need + LID * lid)
        if unstable or moved:
            credit = 0.0
        return {"credit": round(float(credit), 4), "cold": cold, "pairs": pairs, "need": need, "lid": lid,
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
