"""Kitchen appliance tending: the episode logic. One episode = one unit, a queue of requests, a two-stage credit.

Wraps an ApplianceEnv (attribute access passes through, so `reference.Driver(Episode(...))` works) and adds:

  * an initial state drawn per episode (burners on or off, the oven preheated or not, the microwave
    running or not), set before the first action;
  * K requests in sequence: burner on / off and, on a cooktop, compositional requests over several
    burners (two on, two off, all off) and burner settings (low / medium / high: BURNER_BANDS); preheat,
    turn the oven off and oven heat settings (low / medium / high: OVEN_BANDS); microwave start / stop. A
    request's category (simple, compositional, setting) is drawn first by CATEGORY_WEIGHTS over the
    categories valid now. Each request is valid in the state it opens on; one is open
    at a time; it closes when completed or at its deadline (DEADLINE steps, DEADLINE_COMPOUND for a
    compositional request);
  * credit per request, decided when it closes:
      **reached** 0.5: when the request closes (completed, or at its deadline) the target state (every named
      control's, for a compositional request) has held for the last REACH_HOLD (3) consecutive steps; a state
      that was passed through and left earns nothing;
      **completed** 0.5: the target state has held COMPLETE_HOLD (20) consecutive steps with the gripper site at
      least CLEAR (0.10 m) from every named control while every other control keeps the discrete state it had
      when the request opened (`discrete`, `others_unchanged`; compared during those 20 steps only); the
      request closes at that step.

    ep = Episode(ApplianceEnv(spec, assets), seed)
    obs = ep.reset()                  # obs["request_id"], ["request_text"], ["request_deadline_t"]
    obs, done, info = ep.step(action)
    ep.score()                        # {"credit": mean request credit, "requests": the per-request record}
"""

import numpy as np

DEADLINE = 1000                   # control steps per request (100 s at 10 Hz); a knob may need a long turn
DEADLINE_COMPOUND = 1500          # a compositional request (several burners)
# A request's category is drawn first, by these weights over the categories valid now, then a request within it.
CATEGORY_WEIGHTS = {"simple": 0.3, "compound": 0.35, "level": 0.35}
LEVELS = ("low", "medium", "high")
# A burner's setting (Unit.burner_level, 0..1 over its on-range) and the oven's temperature (normalized
# temperature knob, preheated from 0.25) in thirds.
BURNER_BANDS = {"low": (0.0, 1 / 3), "medium": (1 / 3, 2 / 3), "high": (2 / 3, 1.0)}
OVEN_BANDS = {"low": (0.25, 0.5), "medium": (0.5, 0.75), "high": (0.75, 1.0)}


def band_of(value, bands):
    """The band a value falls in (the top band includes 1.0), or None."""
    if value is None:
        return None
    for name, (lo, hi) in bands.items():
        if lo <= value < hi or (hi >= 1.0 and value >= hi):
            return name
    return None


REACH_HOLD = 3
COMPLETE_HOLD = 20
CLEAR = 0.10                      # m, gripper site to the control
K_MAX = 3
BURNER_ON_P = 0.3                 # chance a burner starts on
RUNNING_P = 0.5                   # chance the microwave starts running
PREHEATED_P = 0.4                 # chance the oven starts on, its knob at 30-50 % of range
PREHEATED_RANGE = (0.30, 0.50)
LOC_TEXT = {"rear_left": "rear left", "rear_center": "rear centre", "rear_right": "rear right",
            "front_left": "front left", "front_center": "front centre", "front_right": "front right",
            "center": "centre", "left": "left", "right": "right"}


def request_texts():
    """Every request text an episode can open, and "" (no request open): the lab's reply schema lists exactly
    these (its strings are enums)."""
    locs = sorted(set(LOC_TEXT.values()))
    out = [""]
    for want in ("on", "off"):
        out += ["Turn %s the %s burner." % (want, x) for x in locs]
    for want in ("on", "off"):
        out += ["Turn %s the %s and %s burners." % (want, a, b) for a in locs for b in locs if a != b]
    out += ["Turn the %s burner to %s." % (x, lv) for x in locs for lv in LEVELS]
    out += ["Set the oven to %s heat." % lv for lv in LEVELS]
    return out + ["Turn off all the burners.", "Preheat the oven.", "Turn off the oven.",
                  "Start the microwave.", "Stop the microwave."]


REQUEST_TEXTS = frozenset(request_texts())
COMPOUND = ("burners_on", "burners_off", "all_off")


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
        env.unit.update(env.model, env.data, env.gripper_geoms)   # flames and indicators show the initial state
        self.requests = []
        self.current = None
        self.t_open = 0
        self._open_next()
        return self._obs(env.observation())

    def discrete(self):
        """Every control's discrete state: what 'nothing else changed' compares. A burner on or off, the oven
        preheated or not, the microwave running or not, a door open (at least 0.90 of its range) and closed (at
        most 0.005 of its range). Not part of it: a burner's setting, the oven's temperature beyond preheated,
        the oven's timer knob."""
        s = self.env.unit_state()
        out = {}
        for k, v in s.items():
            if k.startswith("burner_on_") or k in ("preheated", "turned_on", "door_open", "door_closed"):
                out[k] = bool(v)
        return out

    def compound_requests(self):
        """Cooktop requests over several burners, each valid now: two burners on, two burners off, all off."""
        u, kind = self.env.unit, self.env.spec["kind"]
        if kind not in ("stovetop", "stove"):
            return []
        st = self.discrete()
        locs = sorted(u.knobs)
        on = [x for x in locs if st["burner_on_" + x]]
        off = [x for x in locs if not st["burner_on_" + x]]
        out = []
        if len(off) >= 2:
            out.append(("burners_on", tuple(off[i] for i in self.rng.choice(len(off), 2, replace=False))))
        if len(on) >= 2:
            out.append(("burners_off", tuple(on[i] for i in self.rng.choice(len(on), 2, replace=False))))
            out.append(("all_off", tuple(locs)))
        return out

    def level_requests(self):
        """A burner to a setting it is not at now (an off burner too), the oven to a heat it is not at now."""
        u, kind = self.env.unit, self.env.spec["kind"]
        d = self.env.data
        out = []
        if kind in ("stovetop", "stove"):
            for loc in sorted(u.knobs):
                now = band_of(u.burner_level(d, loc), BURNER_BANDS)
                out += [("burner_level", (loc, lv)) for lv in LEVELS if lv != now]
        if kind in ("stove", "oven") and u.temp_knob is not None:
            now = band_of(u.temperature(d), OVEN_BANDS)
            out += [("oven_level", ("temp", lv)) for lv in LEVELS if lv != now]
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
        if kind == "microwave" and "start_button" in u.buttons and "stop_button" in u.buttons:
            out.append(("stop", "stop_button") if st["turned_on"] else ("start", "start_button"))
        return out

    def _open_next(self):
        if len(self.requests) >= self.k_max:
            self.current = None
            return
        cats = {"simple": self.valid_requests(), "compound": self.compound_requests(), "level": self.level_requests()}
        names = [c for c in ("simple", "compound", "level") if cats[c]]
        options = []
        if names:
            w = np.array([CATEGORY_WEIGHTS[c] for c in names])
            options = cats[names[int(self.rng.choice(len(names), p=w / w.sum()))]]
        if not options:
            self.current = None
            return
        what, target = options[int(self.rng.integers(len(options)))]
        name = lambda x: LOC_TEXT.get(x, x.replace("_", " "))  # noqa: E731
        text = {"burner_on": "Turn on the %s burner.", "burner_off": "Turn off the %s burner."}.get(what)
        if text:
            text = text % name(target)
        elif what in ("burners_on", "burners_off"):
            text = "Turn %s the %s and %s burners." % ("on" if what == "burners_on" else "off", name(target[0]),
                                                       name(target[1]))
        elif what == "all_off":
            text = "Turn off all the burners."
        elif what == "burner_level":
            text = "Turn the %s burner to %s." % (name(target[0]), target[1])
        elif what == "oven_level":
            text = "Set the oven to %s heat." % target[1]
        else:
            text = {"preheat": "Preheat the oven.", "oven_off": "Turn off the oven.",
                    "start": "Start the microwave.", "stop": "Stop the microwave."}[what]
        assert text in REQUEST_TEXTS, text           # the lab's reply schema must know every text
        r = {"id": len(self.requests), "what": what, "target": target, "text": text, "t_open": self.env.t,
             "deadline_t": self.env.t + (DEADLINE_COMPOUND if what in COMPOUND else DEADLINE),
             "before": self.discrete(), "reached": False, "completed": False, "run_reach": 0, "run_complete": 0,
             "t_close": None}
        self.requests.append(r)
        self.current = r

    # ---- per step -------------------------------------------------------------------
    def control_pos(self, r):
        """The positions of the request's controls (one for a simple request)."""
        d, m = self.env.data, self.env.model
        if r["what"] in COMPOUND:
            return [d.geom_xpos[m.geom("u_knob_%s_main" % x).id] for x in r["target"]]
        if r["what"] in ("burner_level", "oven_level"):
            return [d.geom_xpos[m.geom("u_knob_%s_main" % r["target"][0]).id]]
        return [self._control_pos(r)]

    def _control_pos(self, r):
        d, m = self.env.data, self.env.model
        if r["what"] in ("burner_on", "burner_off", "preheat", "oven_off"):
            return d.geom_xpos[m.geom("u_knob_%s_main" % r["target"]).id]
        return d.geom_xpos[self.env.unit.buttons[r["target"]]]

    def target_holds(self, r):
        st = self.discrete()
        w = r["what"]
        if w == "burner_on":
            return st["burner_on_" + r["target"]]
        if w == "burner_off":
            return not st["burner_on_" + r["target"]]
        if w == "burner_level":
            loc, lv = r["target"]
            return band_of(self.env.unit.burner_level(self.env.data, loc), BURNER_BANDS) == lv
        if w == "oven_level":
            return band_of(self.env.unit.temperature(self.env.data), OVEN_BANDS) == r["target"][1]
        if w == "burners_on":
            return all(st["burner_on_" + x] for x in r["target"])
        if w in ("burners_off", "all_off"):
            return not any(st["burner_on_" + x] for x in r["target"])
        if w == "preheat":
            return st["preheated"]
        if w == "oven_off":
            return not st["preheated"]
        if w == "start":
            return st["turned_on"]
        return not st["turned_on"]

    def others_unchanged(self, r):
        """Every control the request does not name keeps the discrete state it had when the request opened
        (`discrete`). A burner request (simple, compositional or setting) compares the other burners and a
        door; an oven request a door and any burner; a microwave request nothing (its door is not compared,
        and opening the door stops it)."""
        if r["what"] in ("burner_level", "oven_level"):
            own = {"burner_on_" + r["target"][0]} if r["what"] == "burner_level" else {"preheated"}
            now = self.discrete()
            return all(now.get(k) == v for k, v in r["before"].items() if k not in own)
        if r["what"] in COMPOUND:
            own = {"burner_on_" + x for x in r["target"]}
            now = self.discrete()
            return all(now.get(k) == v for k, v in r["before"].items() if k not in own)
        own = {"burner_on": "burner_on_" + r["target"], "burner_off": "burner_on_" + r["target"],
               "preheat": "preheated", "oven_off": "preheated", "start": "turned_on", "stop": "turned_on"}[r["what"]]
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
            hand = self.env.data.site_xpos[self.env.site]
            clear = min(np.linalg.norm(hand - p) for p in self.control_pos(r)) >= CLEAR
            ok = holds and clear and self.others_unchanged(r)
            r["run_complete"] = r["run_complete"] + 1 if ok else 0
            if r["run_complete"] >= COMPLETE_HOLD:
                r["completed"] = True
            if r["completed"] or self.env.t >= r["deadline_t"]:
                r["t_close"] = self.env.t
                r["reached"] = r["run_reach"] >= REACH_HOLD      # decided at the close: the state held then
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

    @staticmethod
    def request_credit(r):
        return 0.5 * r["reached"] + 0.5 * r["completed"]

    def score(self):
        """The mean request credit and, per request: its type and target, open and close steps, reached,
        completed, its credit and its outcome ("completed", "reached" or "missed")."""
        rs = self.requests
        credit = [self.request_credit(r) for r in rs]
        out = []
        for r in rs:
            q = {k: r[k] for k in ("id", "what", "target", "text", "t_open", "t_close", "reached", "completed")}
            q["credit"] = self.request_credit(r)
            q["outcome"] = "completed" if r["completed"] else ("reached" if r["reached"] else "missed")
            out.append(q)
        return {"credit": float(np.mean(credit)) if credit else 0.0, "requests": out}
