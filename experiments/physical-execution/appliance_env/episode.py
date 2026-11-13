"""Appliance tending, P2: one episode = one unit, a queue of requests, the design's two-stage credit.

Wraps an ApplianceEnv (attribute access passes through, so `reference.Driver(Episode(...))` works) and
adds what the grader will own in P4:

  * an initial state drawn per episode (burners on or off, the oven preheated or not, the microwave
    running or not), set before the first action;
  * K requests in sequence from the v1 list (design section 10, owner decisions 2026-11-12): burner on /
    off (cooktop, range), preheat and turn the oven off (range, oven), range door open, microwave
    start / stop. Each request is
    valid in the state it opens on; one is open at a time; it closes when completed or at its deadline;
  * credit per request (design section 4): **reached** 0.5 once the target state has held 3 consecutive
    steps; **completed** 0.5 once it has held 20 consecutive steps with the hand at least 0.10 m from
    the control while every other control keeps the discrete state it had when the request opened (doing
    one request must not change anything else). A counted stage stays counted.

    ep = Episode(ApplianceEnv(spec, assets), seed)
    obs = ep.reset()                  # obs["request_id"], ["request_text"], ["request_deadline_t"]
    obs, done, info = ep.step(action)
    ep.score()                        # {"credit": mean request credit, "requests": the per-request record}
"""

import numpy as np

DEADLINE = 600                    # control steps per request (60 s at 10 Hz)
REACH_HOLD = 3
COMPLETE_HOLD = 20
CLEAR = 0.10                      # m, gripper site to the control
K_MAX = 3
BURNER_ON_P = 0.3                 # chance a burner starts on
RUNNING_P = 0.5                   # chance the microwave starts running
PREHEATED_P = 0.4                 # chance the oven (range or wall oven) starts on, its knob at 30-50 % of range
PREHEATED_RANGE = (0.30, 0.50)
LOC_TEXT = {"rear_left": "rear left", "rear_center": "rear centre", "rear_right": "rear right",
            "front_left": "front left", "front_center": "front centre", "front_right": "front right",
            "center": "centre", "left": "left", "right": "right"}


class Episode(object):

    def __init__(self, env, seed, k_max=K_MAX):
        self.env = env
        self.seed = int(seed)
        self.k_max = k_max

    def __getattr__(self, name):          # everything else is the environment's
        return getattr(self.__dict__["env"], name)

    # ---- set-up ---------------------------------------------------------------------
    def reset(self):
        env = self.env
        env.reset(self.seed)
        self.rng = np.random.default_rng([self.seed & 0xFFFFFFFF, self.seed >> 32, 23])
        u = env.unit
        for loc in sorted(u.knobs):
            if self.rng.random() < BURNER_ON_P:
                hi = env.model.joint("u_knob_%s_joint" % loc).range[1]
                env.set_joint("knob_%s_joint" % loc, float(self.rng.uniform(0.8, 1.6)) * (1 if hi > 0 else -1))
        if "start_button" in u.buttons and self.rng.random() < RUNNING_P:
            u.turned_on = True
        if u.temp_knob is not None and self.rng.random() < PREHEATED_P:
            _qa, lo, hi = u.temp_knob
            f = float(self.rng.uniform(*PREHEATED_RANGE))
            env.set_joint("knob_temp_joint", lo + f * (hi - lo) if lo >= 0 else hi - f * (hi - lo))
        self.requests = []
        self.current = None
        self.t_open = 0
        self._open_next()
        return self._obs(env.observation())

    def discrete(self):
        """Every control's discrete state: what 'nothing else changed' compares."""
        s = self.env.unit_state()
        out = {}
        for k, v in s.items():
            if k.startswith("burner_on_") or k in ("preheated", "turned_on", "door_open", "door_closed"):
                out[k] = bool(v)
        return out

    def valid_requests(self):
        u, kind = self.env.unit, self.env.spec["kind"]
        st = self.discrete()
        out = []
        if kind in ("stovetop", "stove"):
            for loc in sorted(u.knobs):
                on = st["burner_on_" + loc]
                out.append(("burner_off" if on else "burner_on", loc))
        if kind in ("stove", "oven") and u.temp_knob is not None:
            out.append(("oven_off", "temp") if st["preheated"] else ("preheat", "temp"))
        if kind == "stove" and u.door is not None and st.get("door_closed", False):
            out.append(("door_open", "door"))
        if kind == "microwave" and "start_button" in u.buttons and "stop_button" in u.buttons:
            out.append(("stop", "stop_button") if st["turned_on"] else ("start", "start_button"))
        return out

    def _open_next(self):
        if len(self.requests) >= self.k_max:
            self.current = None
            return
        options = self.valid_requests()
        if not options:
            self.current = None
            return
        what, target = options[int(self.rng.integers(len(options)))]
        text = {"burner_on": "Turn on the %s burner.", "burner_off": "Turn off the %s burner."}.get(what)
        if text:
            text = text % LOC_TEXT.get(target, target.replace("_", " "))
        else:
            text = {"preheat": "Preheat the oven.", "oven_off": "Turn off the oven.", "door_open": "Open the oven door.",
                    "start": "Start the microwave.", "stop": "Stop the microwave."}[what]
        r = {"id": len(self.requests), "what": what, "target": target, "text": text, "t_open": self.env.t,
             "deadline_t": self.env.t + DEADLINE, "before": self.discrete(), "reached": False,
             "completed": False, "run_reach": 0, "run_complete": 0, "t_close": None}
        self.requests.append(r)
        self.current = r

    # ---- per step -------------------------------------------------------------------
    def control_pos(self, r):
        d, m = self.env.data, self.env.model
        if r["what"] in ("burner_on", "burner_off", "preheat", "oven_off"):
            return d.geom_xpos[m.geom("u_knob_%s_main" % r["target"]).id]
        if r["what"] in ("start", "stop"):
            return d.geom_xpos[self.env.unit.buttons[r["target"]]]
        return d.geom_xpos[self.env.unit.handle]

    def target_holds(self, r):
        st = self.discrete()
        w = r["what"]
        if w == "burner_on":
            return st["burner_on_" + r["target"]]
        if w == "burner_off":
            return not st["burner_on_" + r["target"]]
        if w == "preheat":
            return st["preheated"]
        if w == "oven_off":
            return not st["preheated"]
        if w == "door_open":
            return st["door_open"]
        if w == "start":
            return st["turned_on"]
        return not st["turned_on"]

    def others_unchanged(self, r):
        own = {"burner_on": "burner_on_" + r["target"], "burner_off": "burner_on_" + r["target"],
               "preheat": "preheated", "oven_off": "preheated", "door_open": "door_", "start": "turned_on", "stop": "turned_on"}[r["what"]]
        now = self.discrete()
        for k, v in r["before"].items():
            if k.startswith(own) or (own == "turned_on" and k.startswith("door_")):
                continue           # the microwave's own door state is not touched by a button press
            if now.get(k) != v:
                return False
        return True

    def step(self, action):
        obs, done, info = self.env.step(action)
        r = self.current
        if r is not None:
            holds = self.target_holds(r)
            r["run_reach"] = r["run_reach"] + 1 if holds else 0
            if r["run_reach"] >= REACH_HOLD:
                r["reached"] = True
            clear = np.linalg.norm(self.env.data.site_xpos[self.env.site] - self.control_pos(r)) >= CLEAR
            ok = holds and clear and self.others_unchanged(r)
            r["run_complete"] = r["run_complete"] + 1 if ok else 0
            if r["run_complete"] >= COMPLETE_HOLD:
                r["completed"] = True
            if r["completed"] or self.env.t >= r["deadline_t"]:
                r["t_close"] = self.env.t
                self._open_next()
        done = self.current is None
        return self._obs(obs), done, info

    def _obs(self, obs):
        """The open request, flat (the lab's reply schema has no optional objects): id, text and
        deadline step, or -1 / "" / -1 while none is open."""
        r = self.current
        obs["request_id"] = -1 if r is None else r["id"]
        obs["request_text"] = "" if r is None else r["text"]
        obs["request_deadline_t"] = -1 if r is None else r["deadline_t"]
        return obs

    def score(self):
        rs = self.requests
        credit = [0.5 * r["reached"] + 0.5 * r["completed"] for r in rs]
        return {"credit": float(np.mean(credit)) if credit else 0.0,
                "requests": [{k: r[k] for k in ("id", "what", "target", "text", "t_open", "t_close", "reached",
                                                "completed")} for r in rs]}

