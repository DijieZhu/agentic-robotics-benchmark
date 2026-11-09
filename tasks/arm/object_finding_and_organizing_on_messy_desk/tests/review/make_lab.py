"""make_lab.py <tests/env.py> <out messy_desk_lab.py>: derive the agent's desk lab from the grader's simulator.

    python3 tests/review/make_lab.py tests/env.py environment/setup/payload/messy_desk_lab.py

The lab keeps the scene (desk, robot, containers, objects, cameras, physics, the action rule and the IK) byte for
byte and drops the episode: the world state machine (arrivals, requests, disturbances, container service, the
base-zone rescue), the detectors and values the scorer reads, the privileged state, the finger-gain setter and the
live score. Every edit asserts that its anchor occurs exactly once, so a changed env.py fails loudly."""
import ast
import re
import sys

src = open(sys.argv[1]).read()


def sub_once(old, new):
    global src
    n = src.count(old)
    assert n == 1, (n, old[:80])
    src = src.replace(old, new)


def cut_method(name, next_marker):
    """Remove `def name(...)` up to (not including) `next_marker` (the next def/comment line, indented)."""
    global src
    start = src.index("    def %s(" % name)
    end = src.index(next_marker, start + 1)
    src = src[:start] + src[end:]


# --- module docstring ---------------------------------------------------------------------------------------------
doc_end = src.index('"""', 3) + 3
src = '''"""messy_desk_lab - the desk lab: the task's scene in NVIDIA Isaac Sim 4.5, with a stdio server.

The same desk, robot, containers, objects, cameras, physics and action rule as the graded episodes, without the
episode: nothing arrives, no request opens, nothing is misplaced or nudged, no container is emptied and nothing is
scored. You lay out the desk yourself (`reset` sets out a starting desk the way an episode starts: the containers
with their seeded jitter, the two initial piles and a few loose objects; `place` puts any object anywhere) and look
at it through the observation your controller will get, from the same sensors: each desk's fixed cameras carry
their own mounting error and the depth is a real sensor's (see the task card). It runs under the Isaac interpreter
(/opt/isaac/bin/python3, Python 3.10), never under the task's py3.12 `python3`.

Run as a script it serves one JSON object per line on stdin/stdout:

    {"cmd": "info"}                         -> {"ok": true, "info": {...}}  calibration, geometry, this desk's container poses
    {"cmd": "reset", "seed": <int>}         -> {"ok": true, "obs": {...}, "robot": {...}}
    {"cmd": "step", "action": [8 floats, optional 9th: hi-res bitmask]}   -> {"ok": true, "obs": {...}, "robot": {...}}
    {"cmd": "place", "model": "006_mustard_bottle", "instance": 0, "pose": [x, y, z, yaw], "clear": true}
                                            -> {"ok": true, "obs": {...}, "robot": {...}}
    {"cmd": "close"}                        -> {"ok": true}, then exit

`obs` has exactly the shape a controller receives from the grader; its `request` is always null (the lab opens
none) and the `mark` element of an action does nothing here. `robot` is the arm's own status: `ik_ok` is false when
the last delta had no inverse-kinematics solution within one step's joint jump (see the task card's Control).
With MESSY_DESK_FREECAM=WxH in the environment, `{"cmd": "freecam", "pos": [x, y, z], "yaw": .., "pitch": ..}`
returns a JPEG from a free camera anywhere in the room, for looking at the desk yourself.
"""'''.replace("- the desk lab", "— the desk lab") + src[doc_end:]

# --- constants: the world state machine's parameters go --------------------------------------------------------------
sub_once('''BASE_RESCUE = 0.26         # nearer than this nothing reaches or sees an object: the world sets it back on the desk
BASE_ZONE_REST_STEPS = 20    # steps at rest beside the base before the world sets an object back on the desk
HORIZON_DEFAULT = 20000

# world state machine parameters (per control step)
ARRIVAL_MEAN_STEPS = 500
LOOSE_CAP = 6
PRESENT_CAP = 22
CONTAINER_CAP = 9
REQUEST_SLOT_STEPS = 5000    # one request opens at a random time in each 5000-step window (4 per 20000-step episode)
REQUEST_DEADLINE_STEPS = 1200
REQUEST_ABSENT_P = 0.2
DISTURB_MEAN_STEPS = 2500
MAX_INSTANCES_PER_MODEL = 2
''', '''MAX_INSTANCES_PER_MODEL = 2
''')
sub_once('''REST_SPEED = 0.08          # piles of convex-decomposed meshes jitter at a few cm/s
HOLD_LIFT = 0.03           # "held": both fingers on the object and its centre this far above where it last rested
ON_TABLE_Z_MAX = 0.16      # centre height of anything resting on the table top (tallest YCB upright ~0.25)
LOST_VALUE = -1            # an object knocked off the table counts as misfiled for the rest of the episode
REBASE_TIMEOUT = 100       # steps a nudged object may keep moving before its baseline is re-read anyway
''', '''REST_SPEED = 0.08          # piles of convex-decomposed meshes jitter at a few cm/s
''')

# --- __init__ tail -------------------------------------------------------------------------------------------------
sub_once('''        self.horizon = HORIZON_DEFAULT
        self.rng = None
        self._sensor_key = 0
        self.t = 0
''', '''        self.rng = None
        self._sensor_key = 0
        self.t = 0
''')

# --- spawn: no baseline / value bookkeeping ---------------------------------------------------------------------------
sub_once('''        obj["present"], obj["removed"], obj["frozen"], obj["rebase_t"] = True, False, 0, None
        obj["rest_z"] = float(xyz[2])      # re-read once it rests (see HOLD_LIFT)
        if getattr(self, "baseline", None) is not None:
            self.baseline[obj["id"]] = 0
''', '''        obj["present"] = True
''')

# --- reset: the starting desk only -------------------------------------------------------------------------------------
sub_once('''    def reset(self, seed, horizon=None):
        np = self.np
        self.rng = np.random.default_rng(int(seed) & ((1 << 64) - 1))
        self.horizon = int(horizon or HORIZON_DEFAULT)
        self.t = 0
        self.events = []
        self.request = None
        self.requests_done = []
        self.touched = set()
''', '''    def reset(self, seed):
        """A starting desk the way an episode starts (containers, piles, loose objects drawn from `seed`)."""
        np = self.np
        self.rng = np.random.default_rng(int(seed) & ((1 << 64) - 1))
        self.t = 0
''')
sub_once('''        for o in self.objects:
            o["removed"], o["frozen"], o["state"], o["on_table"], o["rebase_t"] = False, 0, "absent", False, None
            self._park(o)
''', '''        for o in self.objects:
            self._park(o)
''')
sub_once('''        self._settle_until_rest()
        self.next_arrival = int(self.rng.exponential(ARRIVAL_MEAN_STEPS))
        self.request_slot = 0
        self.next_request = self._request_time(0)
        self.next_disturb = int(self.rng.exponential(DISTURB_MEAN_STEPS)) + 500
        self._mount_cameras(seed)
        for _ in range(RESET_WARMUP_FRAMES):   # the RTX renderer needs frames to converge
            self.world.step(render=True)
        self._update_states()
        # v(0): container membership only. Nothing is held at reset, so "resting" is not required
        # here; demanding it would let a pile that is still jittering drift upward later.
        self.baseline = {o["id"]: self._value(o, require_rest=False) for o in self.objects}
        _log("reset seed=%d settled_steps=%d present=%d max_speed=%.3f" % (
            int(seed) & 0xFFFF, self._settled_steps, sum(o["present"] for o in self.objects),
            max([o.get("speed", 0.0) for o in self.objects if o["present"]] or [0.0])))
        return self.observe(), self.privileged()
''', '''        self._settle_until_rest()
        self._mount_cameras(seed)
        for _ in range(RESET_WARMUP_FRAMES):   # the RTX renderer needs frames to converge
            self.world.step(render=True)
        _log("reset settled_steps=%d present=%d" % (self._settled_steps, sum(o["present"] for o in self.objects)))
        return self.observe(), self.robot()
''')

# --- step: physics only ------------------------------------------------------------------------------------------------
sub_once('''        mark = int(round(a[7]))          # 0 none, 1 "done: delivered", 2 "done: no such object"
        self._apply_arm(dpos, drot, grip)
''', '''        # a[7], the mark, closes a request in an episode; the lab opens none, so it does nothing here
        self._apply_arm(dpos, drot, grip)
''')
sub_once('''        self.t += 1
        self._update_states()
        self._world_events(mark)
        return self.observe(hires=self._hires_capture(hires)), self.privileged()
''', '''        self.t += 1
        return self.observe(hires=self._hires_capture(hires)), self.robot()
''')

# --- the scorer's detectors, the world's events, requests, the gains setter, the privileged state: gone ---------------
start = src.index("    # -- state ----")
end = src.index("    def lab(")
src = src[:start] + src[end:]
cut_method("gains", "    # -- outputs ----")

# --- lab -> place ---------------------------------------------------------------------------------------------------
sub_once('''    def lab(self, model, instance=0, pose=None, clear=True):
        """Development aid, available to everyone who runs this file: optionally clear the
        desk, then place one instance of `model` at `pose` (x, y, z[, yaw]) and let it settle; a
        yaw of null sets it down the way an arrival is (resting pose and yaw rule of `_desk_yaw`).
        The world state machine is paused (no arrivals, requests or disturbances) until the
        next `reset`. Returns an observation and the privileged state like `step`."""
        if self.rng is None:
            self.reset(0)
        if clear:
            for o in self.objects:
                if o["present"]:
                    self._park(o)
            self.request = None           # a request stays open when only one object is placed
        self.next_arrival = self.next_request = self.next_disturb = 10 ** 12
        if model is not None:''', '''    def place(self, model, instance=0, pose=None, clear=True):
        """Optionally clear the desk (`clear`), then put instance `instance` (0 or 1, the two sizes) of
        `model` at `pose` (x, y, z[, yaw]) and let it settle; a yaw of null sets it down the way an
        arrival is (standing in its resting pose, its long side away from the nearest container
        wall). A model of null only clears. Returns an observation like `step`."""
        if self.rng is None:
            self.reset(0)
        if clear:
            for o in self.objects:
                if o["present"]:
                    self._park(o)
        if model is not None:''')
sub_once('''        self._settle_until_rest()
        for _ in range(RESET_WARMUP_FRAMES):   # RTX needs frames to converge after a teleport
            self.world.step(render=True)
        self._update_states()
        return self.observe(), self.privileged()
''', '''        self._settle_until_rest()
        for _ in range(RESET_WARMUP_FRAMES):   # RTX needs frames to converge after a teleport
            self.world.step(render=True)
        return self.observe(), self.robot()
''')

# --- observe: no request ------------------------------------------------------------------------------------------------
sub_once('''        req = None
        if self.request and self.request["status"] == "open":
            req = {"text": self.request["text"], "deadline_t": self.request["deadline_t"]}
        obs = {"t": self.t, "request": req, "rgb": rgb, "depth": depth,''', '''        obs = {"t": self.t, "request": None, "rgb": rgb, "depth": depth,''')

# --- privileged() -> robot() ------------------------------------------------------------------------------------------
start = src.index("    def privileged(self):")
end = src.index("    def info(self):")
src = src[:start] + '''    def robot(self):
        """The arm's own status (what its controller knows about itself), never the objects'."""
        return {"t": self.t, "ik_ok": getattr(self, "_ik_ok", True)}

''' + src[end:]
sub_once('''                "pos_step_max": POS_STEP_MAX, "rot_step_max": ROT_STEP_MAX, "horizon": self.horizon,''',
         '''                "pos_step_max": POS_STEP_MAX, "rot_step_max": ROT_STEP_MAX,''')

# --- serve(): no scorer, no requests, no score, no gains ------------------------------------------------------------------
start = src.index("    env = MessyDeskEnv(headless=True)")
end = src.index('            elif cmd == "freecam":')
src = src[:start] + '''    env = MessyDeskEnv(headless=True)
    out.write(json.dumps({"ok": True, "ready": True}) + "\\n"); out.flush()
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
            cmd = msg.get("cmd")
            if cmd == "info":
                reply = {"ok": True, "info": env.info()}
            elif cmd == "reset":
                obs, robot = env.reset(msg.get("seed", 0))
                reply = {"ok": True, "obs": obs, "robot": robot}
            elif cmd == "step":
                obs, robot = env.step(msg["action"])
                reply = {"ok": True, "obs": obs, "robot": robot}
            elif cmd == "place":
                obs, robot = env.place(msg.get("model"), msg.get("instance", 0), msg.get("pose"), msg.get("clear", True))
                reply = {"ok": True, "obs": obs, "robot": robot}
''' + src[end:]
src = src.replace("class MessyDeskEnv:", "class MessyDeskLab:")
src = src.replace("    env = MessyDeskEnv(headless=True)", "    env = MessyDeskLab(headless=True)")

sub_once('''        """Run physics until every present object has been at rest for `quiet` consecutive
        steps (or `max_steps`), so that the baseline v(0) is taken on a settled scene and an
        untouched episode cannot drift upward on its own."""''', '''        """Run physics until every present object has been at rest for `quiet` consecutive
        steps (or `max_steps`)."""''')

# --- checks: no trace of the episode left ------------------------------------------------------------------------------
for word in ("privileged", "baseline", "_world_events", "request_", "force_request", "EpisodeScorer", "score\"", "live()",
             "ARRIVAL", "DISTURB", "REQUEST_", "CONTAINER_CAP", "LOOSE_CAP", "PRESENT_CAP", "HOLD_LIFT", "_value(",
             "def gains", "horizon", "self.request", "self.events", "frozen", "rebase"):
    hits = [i + 1 for i, l in enumerate(src.splitlines()) if word in l]
    assert not hits, (word, hits[:5])
tree = ast.parse(src)
open(sys.argv[2], "w").write(src)
print("wrote", sys.argv[2], len(src), "bytes,", src.count("\n"), "lines")
