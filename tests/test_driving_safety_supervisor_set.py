"""driving_safety_supervisor: the fixed hidden set, the grader's checks of it, and what the agent is given.

The payload's copies of the bridge, the protocol, the client, the guard and the probe sprites are the grader's; the
hidden set is 24 episodes, one per graded cell in grading order, with distinct seeds in [2**32, 2**62), distinct lock
keys and a layout digest each, every one the first kept candidate of its cell in the committed selection record,
kept by the pre-registered rule, selected with the files that are committed now, by a selection that shares no seed,
lock key or id with the one it superseded; nothing agent-visible holds a value
of the set or names the reference's token; the practice grader refuses the graded towns; a malformed set is a task
error whose message names no value of it, and so is a layout the simulator builds differently; the mock's layout
digest follows the spec alone; footprints.json covers every prop a staging spawns, and the car; shoulder_pass's sign
keeps the lane centre wherever the shoulder leaves the car room beside it, with the same draws either way; the keep
rule drops a run in which a walker near the car was launched, and the review recorder finds such a launch; a red light
counts once for a crossing; the expert's replies and memory do not depend on the order of a probe pair.
"""
import contextlib
import copy
import hashlib
import io
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

TASK = Path(__file__).resolve().parents[1] / "tasks/driving/driving_safety_supervisor"
TESTS = TASK / "tests"
PAYLOAD = TASK / "environment/setup/payload"


# the task's module names, which other tasks' tests may import from their own folders: imported here with those
# set aside, and taken out of sys.modules again afterwards
_NAMES = ("dev_batch", "drive_bridge", "drive_protocol", "drive_variants", "episode_score", "robotics_grader",
          "run_episode", "select_set", "sim_client", "solver", "verify")


def _load():
    saved_path, saved_flag = list(sys.path), sys.dont_write_bytecode
    saved = {n: sys.modules.pop(n) for n in list(sys.modules) if n.split(".")[0] in _NAMES}
    sys.dont_write_bytecode = True  # (no __pycache__ in the task folder)
    try:
        for d in ("solution", "tests/review", "tests"):
            sys.path.insert(0, str(TASK / d))
        import drive_bridge  # noqa: E402
        import drive_variants  # noqa: E402
        import run_episode  # noqa: E402
        import select_set  # noqa: E402
        import solver  # noqa: E402
        import verify  # noqa: E402
        return drive_bridge, drive_variants, run_episode, select_set, solver, verify
    finally:
        sys.path[:] = saved_path
        sys.dont_write_bytecode = saved_flag
        for n in [n for n in sys.modules if n.split(".")[0] in _NAMES]:
            del sys.modules[n]
        sys.modules.update(saved)


drive_bridge, drive_variants, run_episode, select_set, solver, verify = _load()
SET = json.loads((TESTS / "hidden/episodes.json").read_text())
SELECTION = json.loads((TESTS / "hidden/selection.json").read_text())
CONFIG = json.loads((TESTS / "grader_config.json").read_text())


class CopiesTests(unittest.TestCase):
    def test_payload_copies_are_the_graders(self):
        for name in ("drive_bridge.py", "drive_protocol.py", "sim_client.py", "netguard.py"):
            self.assertEqual((PAYLOAD / "drive" / name).read_bytes(), (TESTS / name).read_bytes(), name)
        ours = sorted(p.name for p in (PAYLOAD / "drive/probe_sprites").iterdir())
        theirs = sorted(p.name for p in (TESTS / "probe_sprites").iterdir())
        self.assertEqual(ours, theirs)
        for name in ours:
            self.assertEqual((PAYLOAD / "drive/probe_sprites" / name).read_bytes(),
                             (TESTS / "probe_sprites" / name).read_bytes(), name)


class SetTests(unittest.TestCase):
    def test_format_cells_order_seeds_keys(self):
        self.assertEqual(SET["format"], verify.SET_FORMAT)
        cells = drive_variants.graded_cells(CONFIG["towns"])
        episodes = SET["episodes"]
        self.assertEqual([(e["town"], e["variant"]) for e in episodes], cells)
        self.assertEqual(len(episodes), 24)
        self.assertEqual(sum(1 for c in cells if c == ("Town05", "shoulder_pass")), 3)
        kind_of = {v: k for k, vs in drive_variants.VARIANTS.items() for v in vs}
        for e in episodes:
            self.assertEqual(e["kind"], kind_of[e["variant"]])
            self.assertIs(type(e["seed"]), int)
            self.assertTrue(1 << 32 <= e["seed"] < 1 << 62)
            self.assertIn(e["distractors"], (0, 1, 2))
            for field in ("lock_key", "layout_sha256"):
                self.assertRegex(e[field], r"\A[0-9a-f]{64}\Z")
        for field in ("seed", "lock_key", "id"):
            self.assertEqual(len({e[field] for e in episodes}), 24, field)
        self.assertEqual(len(verify.load_set(CONFIG)), 24)

    def test_set_is_the_selections_first_kept_per_cell(self):
        self.assertEqual(SELECTION["format"], select_set.SEL_FORMAT)
        by_id = {c["id"]: c for c in SELECTION["candidates"]}
        for k, e in enumerate(SET["episodes"]):
            c = by_id[e["id"]]
            self.assertEqual(c["cell"], k)
            self.assertEqual(c["selection"], "kept")
            for field in ("town", "variant", "kind", "seed", "distractors", "lock_key"):
                self.assertEqual(c[field], e[field], field)
            self.assertEqual(c["run"]["layout"], e["layout_sha256"])
            self.assertEqual(select_set.keep_clauses(c), [])
            earlier = [d for d in SELECTION["candidates"] if d["cell"] == k and d["draw"] < c["draw"]]
            for d in earlier:          # every earlier candidate of the cell ran and failed the rule
                self.assertEqual(d["status"], "ran")
                self.assertTrue(select_set.keep_clauses(d))
                self.assertEqual(d["selection"], "dropped")
        self.assertEqual(SELECTION["set"]["episodes_sha256"],
                         hashlib.sha256((TESTS / "hidden/episodes.json").read_bytes()).hexdigest())

    def test_supersedes_the_first_selection(self):
        sup = SELECTION["supersedes"]
        path = TASK / sup["file"]
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), sup["sha256"])
        old = json.loads(path.read_text())["candidates"]
        seeds, keys, ids = ({c[f] for c in old} for f in ("seed", "lock_key", "id"))
        for c in SELECTION["candidates"]:
            self.assertNotIn(c["seed"], seeds)
            self.assertNotIn(c["lock_key"], keys)
            self.assertNotIn(c["id"], ids)
            self.assertGreater(c["round"], sup["last_round"])

    def test_selected_with_the_committed_files(self):
        now = select_set.hashes(select_set.FROZEN)
        changed = [p for p in select_set.FROZEN if SELECTION["frozen"][p] != now[p]]
        self.assertEqual(changed, [], "a file the set was selected with has changed: select the set again")


class AgentVisibleTests(unittest.TestCase):
    def test_no_value_of_the_set_and_no_token(self):
        needles = set()
        for e in SET["episodes"]:
            needles.update((str(e["seed"]), e["lock_key"], e["layout_sha256"]))
        needles.add((TASK / "solution/reference/oracle_token").read_text().strip())
        needles.add((TESTS / "oracle_token.sha256").read_text().split()[0])
        needles.update((".oracle_token", "oracle_token"))
        files = [TASK / "instruction.md"] + [p for p in (TASK / "environment").rglob("*") if p.is_file()]
        for path in files:
            data = path.read_bytes()
            for needle in needles:
                self.assertNotIn(needle.encode(), data, "{} holds a value it must not".format(path))

    def test_practice_grader_refuses_graded_towns(self):
        with tempfile.TemporaryDirectory() as tmp:
            sub = os.path.join(tmp, "submission")
            os.makedirs(sub)
            Path(sub, "run.sh").write_text("#!/bin/bash\necho READY\n")
            env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
            proc = subprocess.run([sys.executable, "-B", str(PAYLOAD / "practice_grader/grade.py"), "--towns", "Town04",
                                   "--backend", "mock", "--submission", sub], capture_output=True, text=True,
                                  timeout=60, env=env)
        self.assertEqual(proc.returncode, 2, proc.stderr[-400:])
        self.assertIn("invalid choice", proc.stderr)


class GraderChecksTests(unittest.TestCase):
    def _fails(self, fn):
        err = io.StringIO()
        with contextlib.redirect_stderr(err), self.assertRaises(SystemExit):
            fn()
        return err.getvalue()

    def _values(self, spec):
        out = set()
        for e in spec["episodes"] if isinstance(spec, dict) and isinstance(spec.get("episodes"), list) else ():
            if isinstance(e, dict):
                out.update(str(e.get(k)) for k in ("seed", "lock_key", "layout_sha256", "id") if e.get(k) is not None)
        return {v for v in out if len(v) > 3}

    def test_malformed_sets_are_task_errors_naming_no_value(self):
        good = copy.deepcopy(SET)
        cases = []

        def case(fn):
            spec = copy.deepcopy(good)
            fn(spec)
            cases.append(spec)
        case(lambda s: s.update(format="driving_safety_supervisor fixed set 0"))
        case(lambda s: s["episodes"].pop())
        case(lambda s: s["episodes"].__setitem__(slice(0, 2), [s["episodes"][1], s["episodes"][0]]))
        case(lambda s: s["episodes"][3].update(kind="occluded_pedestrian"))
        case(lambda s: s["episodes"][4].update(seed=12345))
        case(lambda s: s["episodes"][5].update(seed=True))
        case(lambda s: s["episodes"][6].update(seed=s["episodes"][2]["seed"]))
        case(lambda s: s["episodes"][7].update(distractors=3))
        case(lambda s: s["episodes"][8].update(lock_key=s["episodes"][8]["lock_key"].upper()))
        case(lambda s: s["episodes"][9].update(lock_key=s["episodes"][1]["lock_key"]))
        case(lambda s: s["episodes"][10].update(layout_sha256="0" * 63))
        case(lambda s: s["episodes"][11].pop("id"))
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "episodes.json")
            for spec in cases:
                with open(path, "w") as f:
                    json.dump(spec, f)
                message = self._fails(lambda: verify.load_set(CONFIG, path))
                self.assertIn("TASK ERROR", message)
                for value in self._values(good):
                    self.assertNotRegex(message, r"(?<![0-9a-f]){}(?![0-9a-f])".format(re.escape(value)))
            with open(path, "w") as f:
                f.write("{not json")
            self.assertIn("TASK ERROR", self._fails(lambda: verify.load_set(CONFIG, path)))

    def test_layout_check(self):
        class FakeSim(object):
            def __init__(self, layout):
                self.layout = layout

            def reset(self, spec):
                return {"truth": {"layout": self.layout}}

            def step(self, mode, control):
                return {}
        want = SET["episodes"][0]["layout_sha256"]
        verify.LayoutChecked(FakeSim(want), want, 0).reset({})
        message = self._fails(lambda: verify.LayoutChecked(FakeSim("f" * 64), want, 0).reset({}))
        self.assertIn("another layout", message)
        self.assertNotIn(want, message)
        self._fails(lambda: verify.LayoutChecked(FakeSim(None), want, 0).reset({}))

    def test_mock_layout_digest(self):
        def layout(seed, kind="construction_zone"):
            return drive_bridge.MockBackend(None).reset({"town": "Town01", "kind": kind, "seed": seed})["truth"]["layout"]
        self.assertEqual(layout(5), layout(5))
        self.assertNotEqual(layout(5), layout(6))
        self.assertNotEqual(layout(5), layout(5, "occluded_pedestrian"))


class FootprintTests(unittest.TestCase):
    def test_every_staged_prop_and_the_car(self):
        table = json.loads((TASK / "solution/footprints.json").read_text())
        measured = set(table["footprints"])
        unmeasured = set(re.findall(r"static\.prop\.[a-z0-9_]+", table["source"].split("No entry:")[1]))
        spawned = set()
        for name in ("drive_variants.py", "drive_bridge.py"):
            spawned |= set(re.findall(r"static\.prop\.[a-z0-9_]+", (TESTS / name).read_text()))
        self.assertTrue(spawned)
        self.assertEqual(spawned - measured - unmeasured, set())
        self.assertIn("vehicle.lincoln.mkz_2020", measured)


class ShoulderSignTests(unittest.TestCase):
    """drive_variants._shoulder_pass on stand-ins for CARLA's objects: a straight lane along x, 3.5 m wide, with a
    shoulder of a given width to its right (y grows to the right, as in CARLA)."""

    def test_sign_sizes_are_the_measured_ones(self):
        table = json.loads((TASK / "solution/footprints.json").read_text())["footprints"]
        for name, half in drive_variants.SIGN_HALF_RIGHT_M.items():
            fp = table[name]
            self.assertAlmostEqual(half, max(fp["box"]["location"][1] + fp["box"]["extent"][1], fp["y"][1]), 3)
        ego = table["vehicle.lincoln.mkz_2020"]
        self.assertAlmostEqual(drive_variants.EGO_WIDTH_M, 2 * max(ego["box"]["extent"][1], ego["y"][1]), 3)

    def _stage(self, shoulder, seed, sign=None):
        class V(object):
            def __init__(self, x=0.0, y=0.0, z=0.0):
                self.x, self.y, self.z = x, y, z

        class Rotation(object):
            def __init__(self, pitch=0.0, yaw=0.0, roll=0.0):
                self.pitch, self.yaw, self.roll = pitch, yaw, roll

        class Transform(object):
            def __init__(self, location, rotation=None):
                self.location, self.rotation = location, rotation or Rotation()

            def get_forward_vector(self):
                return V(1.0, 0.0)

            def get_right_vector(self):
                return V(0.0, 1.0)

        LaneType = type("LaneType", (), {"Driving": "driving", "Shoulder": "shoulder", "Parking": "parking"})

        class Lane(object):
            def __init__(self, width, kind, right=None):
                self.lane_width, self.lane_type, self.lane_id, self.right = width, kind, -1, right
                self.transform = Transform(V())

            def get_right_lane(self):
                return self.right

            def get_left_lane(self):
                return None

        lane = Lane(3.5, LaneType.Driving, Lane(shoulder, LaneType.Shoulder) if shoulder else None)

        class Actor(object):
            def __init__(self, name, tf):
                self.type_id, self.tf = name, tf

            def apply_control(self, control):
                pass

        class Library(object):
            def filter(self, pattern):
                if pattern == "vehicle.*":
                    return [type("BP", (), {"id": "vehicle.audi.a2", "get_attribute": lambda self, k: "4"})()]
                return [pattern] if sign is None or pattern == sign else []

        b = type("Builder", (), {})()
        b.carla = type("carla", (), {"LaneType": LaneType, "Transform": Transform, "Rotation": Rotation,
                                     "Location": V, "VehicleControl": lambda self=None, **kw: kw})
        b.rng = __import__("random").Random(seed)
        b.world = type("World", (), {"get_blueprint_library": lambda self: Library()})()
        b.map = type("Map", (), {"get_waypoint": lambda self, loc: lane})()
        spawned = []
        b.b = type("Backend", (), {"_spawn": lambda self, name, tf: spawned.append(Actor(name, tf)) or spawned[-1]})()
        b.hazards, b.staging = [], {}
        b._offset = lambda wp, forward, right, z=0.0: Transform(V(forward, right, z))
        drive_variants._shoulder_pass(b, None, 0, lane)
        sign_actor = [a for a in spawned if a.type_id in drive_variants.SIGN_HALF_RIGHT_M][0]
        return sign_actor.tf.location, b.staging, b.rng.getstate()

    def test_centred_where_the_shoulder_leaves_room(self):
        for seed in range(12):
            wide, staged, after = self._stage(3.5, seed)
            self.assertEqual((wide.x, wide.y), (-8.0, 0.0))           # the author's place
            self.assertEqual(staged["sign_lateral"], 0.0)
            self.assertGreaterEqual(staged["sign_room"], drive_variants.SIGN_SPARE_M)
            narrow, staged_n, after_n = self._stage(2.4, seed, "static.prop.trafficwarning")
            self.assertTrue(-0.9 <= narrow.y <= -0.7)                  # a centred trailer leaves 0.88 m there
            self.assertLess(staged_n["sign_room"], drive_variants.SIGN_SPARE_M)
            _, _, after_w = self._stage(3.5, seed, "static.prop.trafficwarning")
            self.assertEqual(after_n, after_w)                         # the same draws either way
        _, staged, _ = self._stage(3.5, 0, "static.prop.trafficwarning")
        self.assertAlmostEqual(staged["sign_room"], 1.75 + 3.5 - 1.436 - 1.836, 2)


class ShoulderClearTests(unittest.TestCase):
    """drive_variants._shoulder_clear on a straight stand-in lane along x (3.5 m, a 3.5 m shoulder to its right):
    a map object standing in the shoulder beside the stretch of a pass rules the spot out."""

    def _lane(self, x):
        test = self
        LaneType = type("LaneType", (), {"Driving": "driving", "Shoulder": "shoulder", "Parking": "parking"})
        V = type("V", (), {"__init__": lambda self, x=0.0, y=0.0: setattr(self, "x", x) or setattr(self, "y", y)})

        class Transform(object):
            location = V(x, 0.0)

            def get_forward_vector(self):
                return V(1.0, 0.0)

            def get_right_vector(self):
                return V(0.0, 1.0)

        class Lane(object):
            lane_width, lane_type, transform = 3.5, LaneType.Driving, Transform()

            def get_right_lane(self):
                return type("Shoulder", (), {"lane_width": 3.5, "lane_type": LaneType.Shoulder})()

            def previous(self, d):
                return [test._lane(x - d)]

            def next(self, d):
                return [test._lane(x + d)]
        self.b = type("Builder", (), {"carla": type("carla", (), {"LaneType": LaneType})})()
        return Lane()

    def test_poles_in_the_shoulder(self):
        wp = self._lane(0.0)
        clear = lambda *points: drive_variants._shoulder_clear(self.b, wp, drive_variants._obstacle_grid(points))  # noqa: E731
        self.assertTrue(clear())
        self.assertFalse(clear((20.0, 1.75 + 0.58, 0.05)))     # a speed-limit sign 0.58 m into the shoulder, 20 m on
        self.assertFalse(clear((-25.0, 1.75 + 2.0, 0.05)))     # ... 25 m before
        self.assertTrue(clear((20.0, 1.75 + 3.45, 0.05)))      # on the kerb line
        self.assertTrue(clear((55.0, 2.5, 0.05)))              # past the stretch
        self.assertTrue(clear((20.0, -3.0, 0.05)))             # across the road


class WalkerLaunchTests(unittest.TestCase):
    """The keep rule's walker clause, and review/run_episode.Recorder, which records what it reads."""

    def _run(self, launch, **extra):
        truth = {"termination": "completed", "collisions": {}, "route_length_m": 80.0, "time_s": 12.0}
        run = {"truth": truth, "clearance": {"all": [1.0, 7, "static.prop.box01", 3.0]}, "walker_launch": launch}
        run.update(extra)
        return {"run": run}

    def test_keep_clause(self):
        self.assertEqual(select_set.keep_clauses(self._run(None)), [])
        self.assertEqual(select_set.keep_clauses(self._run([1, 21.0, 5, "walker.pedestrian.0001", 3.0, 4.0])), [])
        failed = select_set.keep_clauses(self._run([2, 15.0, 5, "walker.pedestrian.0001", 3.0, 4.0]))
        self.assertEqual(len(failed), 1)
        self.assertIn("walker 5", failed[0])
        no_record = self._run(None)
        del no_record["run"]["walker_launch"]
        self.assertTrue(select_set.keep_clauses(no_record))

    def test_recorder(self):
        def state(t, speeds):
            actors = [{"id": i, "type": "walker", "blueprint": "walker.pedestrian.0001", "x": x, "y": 1.0, "yaw": 0.0,
                       "vx": v, "vy": 0.0, "length": 0.5, "width": 0.5} for i, (x, v) in speeds.items()]
            return {"truth": {"time_s": t}, "ego": {"speed": 0.0}, "privileged": {"actors": actors, "speed_limit": 8.33}}
        steps = [{1: (8.0, 1.4), 2: (30.0, 20.0)},       # 2 is fast but beyond 25 m
                 {1: (8.0, 15.0), 2: (30.0, 20.0)},      # 1 launched ...
                 {1: (6.5, 9.0)},                        # ... for three steps
                 {1: (5.6, 8.0), 3: (4.0, 30.0)},        # 3: one fast step (a person put back by hand)
                 {1: (5.6, 0.0), 3: (4.0, 0.0)}]
        sim = type("Sim", (), {})()
        feed = iter(steps)
        sim.reset = lambda spec: state(0.0, next(feed))
        sim.step = lambda mode, control, n=[0]: n.__setitem__(0, n[0] + 1) or state(0.1 * n[0], next(feed))
        rec = run_episode.Recorder(sim, {}, True)
        rec.reset({})
        for _ in range(4):
            rec.step(0, [0.0, 0.0, 0.0])
        self.assertEqual(rec.walker_launch[0], 3)
        self.assertEqual(rec.walker_launch[1:3], [15.0, 1])
        self.assertEqual(rec.walker_fast[:2], [30.0, 3])
        self.assertEqual(rec.steps[1]["walker"], [15.0, 1, 8.06])


class RedLightCountTests(unittest.TestCase):
    """CarlaBackend._red_light_check on stand-ins for CARLA's objects."""

    class State(object):
        Red, Yellow, Green = "red", "yellow", "green"

    def _backend(self):
        test = self

        class Light(object):
            def __init__(self, ident):
                self.id, self.state = ident, test.State.Red

            def get_state(self):
                return self.state

        class Ego(object):
            light = None

            def is_at_traffic_light(self):
                return self.light is not None

            def get_traffic_light(self):
                return self.light

        class Waypoint(object):
            def __init__(self, junction):
                self.is_junction = junction

        class Map(object):
            junction = False

            def get_waypoint(self, loc):
                return Waypoint(self.junction)

        b = type("Backend", (), {})()
        b.carla = type("carla", (), {"TrafficLightState": test.State})
        b.ego, b.map = Ego(), Map()
        b.nominal = type("Nominal", (), {"traffic_light_manager": lambda self: False})()
        b.tracker = type("Tracker", (), {"time": 0.0, "red_lights": 0})()
        b.red_candidate, b.red_counted = None, None
        return b, Light

    def _tick(self, b, at_light, in_junction, speed):
        b.ego.light = at_light
        b.map.junction = in_junction
        b.tracker.time += 0.05
        return drive_bridge.CarlaBackend._red_light_check(b, None, speed)

    def test_once_per_crossing(self):
        b, Light = self._backend()
        first, second = Light(7), Light(8)
        self._tick(b, first, False, 7.0)               # at the line, red
        for _ in range(12):                            # moving through the junction, still in its trigger
            self._tick(b, first, True, 5.0)
        self.assertEqual(b.tracker.red_lights, 1)
        for _ in range(4):                             # through, beyond the light
            self._tick(b, None, True, 5.0)
        self._tick(b, None, False, 5.0)                # out of the junction: the crossing is over
        self.assertIsNone(b.red_counted)
        self._tick(b, second, False, 7.0)
        self._tick(b, second, True, 7.0)               # the next junction's red light counts again
        self.assertEqual(b.tracker.red_lights, 2)

    def test_green_and_standing_do_not_count(self):
        b, Light = self._backend()
        light = Light(9)
        self.assertTrue(self._tick(b, light, False, 0.0))   # held at red: excused
        light.state = self.State.Green
        for _ in range(10):
            self._tick(b, light, True, 6.0)
        self.assertEqual(b.tracker.red_lights, 0)


class ProbeOrderTests(unittest.TestCase):
    """E0: with a probe pair every 4 steps, the expert's replies and carried memory are the same whichever request
    of the pair comes first (a synthetic road: a box in the lane, an oncoming car, a red light at 70 m)."""

    def _episode(self, order, steps=120, lane=3.5):
        e = solver.Expert()
        e.reset()
        X, Y, YAW, V, wheel = 0.0, 0.0, 0.0, 0.0, 0.0
        out = []
        for t in range(steps):
            c, s = math.cos(YAW), math.sin(YAW)

            def loc(x, y):
                dx, dy = x - X, y - Y
                return round(dx * c + dy * s, 2), round(-dx * s + dy * c, 2)
            route, i0 = [], int(X // 2)
            for i in range(max(0, i0 - 5), i0 + 45):
                lx, ly = loc(2.0 * i, 0.0)
                llx, lly = loc(2.0 * i, -lane)
                left = {"x": llx, "y": lly, "width": lane, "same_direction": False}
                route.append({"x": lx, "y": ly, "width": lane, "junction": 72.0 <= 2.0 * i <= 90.0, "left": left,
                              "lefts": [left], "rights": []})
            bx, by = loc(45.0, 0.2)
            ox, oy = loc(160.0 - 0.9 * t, -lane)
            sx, sy = loc(70.0, 0.0)
            priv = {"actors": [{"id": 100, "type": "prop", "blueprint": "static.prop.box01", "x": bx, "y": by,
                                "z": 0.3, "yaw": round(math.degrees(-YAW), 1), "vx": 0.0, "vy": 0.0, "length": 0.65,
                                "width": 0.65, "height": 0.7},
                               {"id": 900, "type": "vehicle", "blueprint": "vehicle.audi.a2", "x": ox, "y": oy,
                                "z": 0.0, "yaw": round(math.degrees(math.pi - YAW), 1), "vx": round(-9.0 * c, 2),
                                "vy": round(9.0 * s, 2), "length": 3.7, "width": 1.8, "height": 1.5,
                                "reverse": False}],
                    "route": route, "lights": [{"id": 5, "x": sx, "y": sy, "state": "red" if t < 90 else "green"}],
                    "red_light": False, "speed_limit": 8.33}
            obs = {"t": t, "speed": round(V, 3), "nominal": [0.75, 0.0, 0.0], "privileged": priv}
            if t % 4 == 1:
                fake = copy.deepcopy(obs)
                fake["privileged"]["actors"].append({"id": -1, "type": "walker", "x": 9.6, "y": 0.3, "yaw": 0.0,
                                                     "vx": 0.0, "vy": 0.0, "length": 0.6, "width": 0.6})
                if order == "probe-first":
                    probe_reply, a = e.act(t, fake), e.act(t, obs)
                else:
                    a, probe_reply = e.act(t, obs), e.act(t, fake)
                out.append(["probe", [float(x) for x in probe_reply]])
            else:
                a = e.act(t, obs)
            a = [float(x) for x in a]
            out.append([a, json.dumps(e.mem, sort_keys=True, default=str)])
            thr, st, br = (0.75, 0.0, 0.0) if a[0] == 0 else (a[1], a[2], a[3])
            for _ in range(2):
                wheel += (st - wheel) * (1 - math.exp(-0.05 / 0.09))
                V = max(0.0, V + (3.0 * thr - 8.0 * br - 0.3) * 0.05)
                YAW += V * math.tan(0.815 * 1.22 * wheel) / 2.9 * 0.05
                X += V * math.cos(YAW) * 0.05
                Y += V * math.sin(YAW) * 0.05
        return out

    def test_both_orders_alike(self):
        self.assertEqual(self._episode("probe-first"), self._episode("real-first"))


if __name__ == "__main__":
    unittest.main()
