"""Pantry jar refilling: an episode and its credit.

An episode ends after EPISODE_STEPS control steps; or once every listed jar has been opened and is closed again
(its own lid in the thread and seated) with both hands open and at least CLEAR m from every jar's top for HOLD steps;
or at an unstable simulation (credit 0).

The credit is the mean over the listed jars of two halves, counted from the simulation at the end:
- REFILL: the pieces of the jar's kind inside it over the pieces listed; 0 if it holds more than listed or any piece
  of another kind;
- SEALED: the jar was opened during the episode and its own lid is back in the thread, seated, with the largest
  closing torque since the lid last entered the thread within TORQUE_WINDOW (N m).
An untouched run scores 0 (nothing refilled, nothing opened).
The episode's credit is halved if a jar not on the list was opened, a jar fell over (more than FALLEN deg), or a piece
left the counter.
"""


EPISODE_STEPS = 12000           # 20 simulated minutes
HOLD = 20
CLEAR = 0.12
REFILL, SEALED = 0.5, 0.5
TORQUE_WINDOW = (0.5, 2.5)        # hand-tight; 2.5 N m is about what turning hard at the seat reaches
FALLEN = 45.0
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
        listed = self.env.episode["listed"]
        if not all(w.opened[j] and w.seated(j) for j in listed):
            return False
        for a in w.arms.values():
            if a.gripper() < 0.07:
                return False
            p = a.pos()
            if any(((p - w.neck_top(j)) ** 2).sum() ** 0.5 < CLEAR for j in range(w.nj)):
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
        ep = env.episode
        kinds = [jar["kind"] for jar in env.unit["jars"]]
        where = [w.piece_in_jar(i) for i in range(len(w.pieces))]
        jars = {}
        for j in ep["listed"]:
            k = kinds[j]
            listed_n = ep["refill"][k]
            inside = [ep["pieces"][i]["kind"] for i in range(len(w.pieces)) if where[i] == j]
            right = sum(1 for x in inside if x == k)
            refill = 0.0 if (len(inside) > right or right > listed_n) else right / listed_n
            opened = bool(w.opened[j])
            sealed = bool(opened and w.seated(j) and TORQUE_WINDOW[0] <= w.seat_peak[j] <= TORQUE_WINDOW[1])
            moved = float(((w.jar_pos(j)[:2] - w.jar_start[j][:2]) ** 2).sum() ** 0.5)
            jars[k] = {"refill": round(refill, 4), "sealed": sealed, "opened": opened, "seated": bool(w.seated(j)),
                       "pieces_in": right, "wrong_in": len(inside) - right, "seat_peak": round(w.seat_peak[j], 3),
                       "tilt": round(w.jar_tilt_deg(j), 2), "moved": round(moved, 4),
                       "credit": round(REFILL * refill + SEALED * sealed, 4)}
        credit = sum(v["credit"] for v in jars.values()) / len(jars)
        unlisted_opened = [kinds[j] for j in range(w.nj) if j not in ep["listed"] and w.opened[j]]
        fallen = [kinds[j] for j in range(w.nj) if w.jar_tilt_deg(j) > FALLEN]
        off = sum(1 for i in range(len(w.pieces)) if not w.piece_on_counter(i))
        penalized = bool(unlisted_opened or fallen or off)
        if penalized:
            credit *= PENALTY
        unstable = env.unstable()
        if unstable:
            credit = 0.0
        return {"credit": round(float(credit), 4), "jars": jars, "unlisted_opened": unlisted_opened, "fallen": fallen,
                "pieces_off_counter": off, "penalized": penalized, "unstable": unstable, "steps": int(env.t)}


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
