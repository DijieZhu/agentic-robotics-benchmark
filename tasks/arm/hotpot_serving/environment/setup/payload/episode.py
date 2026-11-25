"""Hotpot serving: an episode and its credit.

An episode ends after EPISODE_STEPS control steps; or once the pot has been lifted off its ring during the episode
and is back on it (ON_RING), the burner is off, and both hands are open at least OPEN m and at least CLEAR m from the
pot's rim, for HOLD consecutive steps; or at an unstable simulation (credit 0).

The credit follows the order, counted from the simulation at the end. A bowl is DONE when both hold:
- it holds exactly the pieces the order lists for it: of each kind exactly that kind's count (a piece is in a bowl
  when its centre is inside the bowl's wall, above its floor and below its rim), and every piece in it is cooked
  (World.cooked), not overcooked (World.overcooked) and untouched (World.touched: no contact of an arm's, a hand's or
  a finger's geom with it during the episode);
- it holds the broth the order lists for it within SOUP_TOL_ML.
The episode's credit is the number of bowls done over three.

The credit is halved (PENALTY) if, at the end, any of these holds:
- more than SPILL_MAX_ML of broth has been spilled (it fell onto the counter, from the pot or from a bowl);
- a piece is neither in the pot, nor in a bowl, nor on a dish;
- a bowl is tilted more than FALLEN deg;
- the pot is not on its ring: its centre more than RING_TOL m from the ring's centre, tilted more than POT_TILT deg,
  or its bottom more than 1 cm above the ring;
- the burner is on.
An untouched run scores 0 (every bowl's order asks for pieces and broth).
"""

import soup_sim as S

EPISODE_STEPS = 8000            # 13 1/3 simulated minutes
HOLD = 20
CLEAR = 0.12
OPEN = 0.07
SOUP_TOL_ML = 20.0
SPILL_MAX_ML = 30.0
FALLEN = 30.0
RING_TOL = 0.03
POT_TILT = 10.0
PENALTY = 0.5


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
        if not (w.pot_lifted and w.pot_on_ring(RING_TOL, POT_TILT)) or w.burner_on():
            return False
        for a in w.arms.values():
            if a.gripper() < OPEN or w.distance_to_rim(a.pos()) < CLEAR:
                return False
        return True

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
        order = env.episode["order"]
        n = len(w.pieces)
        places = [w.piece_place(i) for i in range(n)]
        good = [w.cooked(i) and not w.overcooked(i) and not w.touched[i] for i in range(n)]
        bowls = {}
        for nm, want in order.items():
            inb = [i for i in range(n) if places[i] == nm]
            n_in = {kd: sum(1 for i in inb if w.kinds[i] == kd) for kd in S.KINDS}
            i_ok = bool(n_in == {kd: int(want["items"].get(kd, 0)) for kd in S.KINDS} and all(good[i] for i in inb))
            ml = w.bowl_soup_ml(nm)
            s_ok = bool(abs(ml - want["soup_ml"]) <= SOUP_TOL_ML)
            bowls[nm] = {"done": bool(i_ok and s_ok), "items_right": i_ok, "soup_right": s_ok,
                         "items_in": n_in, "items_listed": dict(want["items"]),
                         "raw": sum(1 for i in inb if not w.cooked(i)),
                         "overcooked": sum(1 for i in inb if w.overcooked(i)),
                         "touched": sum(1 for i in inb if w.touched[i]),
                         "soup_ml": round(ml, 1), "soup_listed_ml": int(want["soup_ml"]),
                         "tilt": round(w.tilt_deg(w.bowls[nm]), 2)}
        credit = sum(1.0 for b in bowls.values() if b["done"]) / len(bowls)
        spilled = w.spilled_ml()
        stray = sum(1 for p in places if p is None)
        fallen = [nm for nm in bowls if w.tilt_deg(w.bowls[nm]) > FALLEN]
        pot_ok = w.pot_on_ring(RING_TOL, POT_TILT)
        burner = w.burner_on()
        penalized = bool(spilled > SPILL_MAX_ML or stray or fallen or not pot_ok or burner)
        if penalized:
            credit *= PENALTY
        unstable = env.unstable()
        if unstable:
            credit = 0.0
        return {"credit": round(float(credit), 4), "bowls": bowls, "spilled_ml": round(spilled, 1),
                "pieces_stray": stray, "pieces_touched": int(sum(w.touched)),
                "pieces_in_pot": sum(1 for p in places if p == "pot"),
                "pieces_on_dishes": sum(1 for p in places if p == "dish"),
                "fallen": fallen, "pot_on_ring": bool(pot_ok), "burner_on": burner, "pot_lifted": bool(w.pot_lifted),
                "pot_soup_ml": round(w.pot_soup_ml(), 1), "broth_c": round(w.broth_c(), 1), "penalized": penalized,
                "unstable": unstable, "steps": int(env.t)}


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

