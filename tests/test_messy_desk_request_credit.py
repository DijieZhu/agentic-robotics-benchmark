"""Request milestones must not reward repeated holds, historical contact, a mere touch or a guessed mark."""
import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace
import unittest


TASK = Path(__file__).resolve().parents[1] / "tasks/arm/object_finding_and_organizing_on_messy_desk"


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, TASK / "tests" / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


score = load("messy_desk_credit_score", "messy_desk_score.py")
env = load("messy_desk_credit_env", "env.py")


class RequestCreditTests(unittest.TestCase):
    def setUp(self):
        self.scorer = score.EpisodeScorer(horizon=1)

    def event(self, kind, rid=0):
        self.scorer.add_step({"objects": [], "events": [{"event": kind, "request": rid}]})

    def test_hold_is_immediate_once_and_delivery_tops_up(self):
        self.event("request_opened")
        self.assertEqual(self.scorer.score(), 0)
        self.event("request_first_hold")
        self.assertEqual(self.scorer.request_progress(), 0.5)
        self.assertEqual(self.scorer.request_points[0], 1)
        self.assertEqual(self.scorer.score(), 0)  # S = O * R: no sorting work brought, no organization (O = 0).
        for _ in range(10):
            self.event("request_first_hold")
        self.assertEqual(self.scorer.request_progress(), 0.5)
        self.assertEqual(self.scorer.counts["first_holds"], 1)
        self.event("request_delivered")
        self.assertEqual(self.scorer.request_progress(), 1)
        self.assertEqual(self.scorer.score(), 0)
        self.assertEqual(self.scorer.request_points[0], 2)
        self.event("request_first_hold")
        self.assertEqual(self.scorer.request_progress(), 1)

    def test_expiry_retains_only_earned_progress(self):
        for held in (False, True):
            with self.subTest(held=held):
                self.setUp()
                self.event("request_opened")
                if held:
                    self.event("request_first_hold")
                self.event("request_expired")
                self.assertEqual(self.scorer.request_points[0], 1 if held else 0)
                self.assertEqual(self.scorer.request_progress(), 0.5 if held else 0)
                self.assertEqual(self.scorer.request_rate(), 0)

    def test_wrong_mark_costs_a_point(self):
        for wrong in score.WRONG_MARK_EVENTS:
            for held in (False, True):
                with self.subTest(wrong=wrong, held=held):
                    self.setUp()
                    self.event("request_opened")
                    if held:
                        self.event("request_first_hold")
                    self.event(wrong)
                    self.assertEqual(self.scorer.request_points[0], (1 if held else 0) - score.WRONG_MARK_COST)
                    self.assertEqual(self.scorer.request_progress(), 0)   # floored at 0
                    self.assertEqual(self.scorer.request_rate(), 0)

    def test_guessing_absent_does_not_pay(self):
        # one correct and four wrong "no such object" marks: the one-in-five absent rate
        for rid, kind in enumerate(["request_absent_correct"] + ["request_absent_wrong"] * 4):
            self.event("request_opened", rid)
            self.event(kind, rid)
        self.assertEqual(self.scorer.request_progress(), 0)
        self.setUp()
        self.event("request_opened", 0)
        self.event("request_delivered", 0)
        self.event("request_opened", 1)
        self.event("request_absent_wrong", 1)
        self.assertEqual(self.scorer.request_progress(), 0.25)   # (2 - 1) / 4

    def test_absence_and_open_requests_use_one_bounded_average(self):
        self.event("request_opened", 0)
        self.event("request_absent_correct", 0)
        self.event("request_opened", 1)
        self.assertEqual(self.scorer.request_progress(), 0.5)
        self.event("request_first_hold", 1)
        self.assertEqual(self.scorer.request_progress(), 0.75)
        self.event("request_delivered", 1)
        self.assertEqual(self.scorer.request_progress(), 1)

    def test_empty_and_organization_only_score_zero(self):
        # S = O * R (2026-11-08): sorting earns nothing while no request is served.
        self.assertEqual(self.scorer.request_progress(), 0)
        self.scorer.add_step({"objects": [{"id": 0, "present": True, "value": 1, "baseline": 0}]})
        self.assertEqual(self.scorer.organization(), 1)
        self.assertEqual(self.scorer.score(), 0)

    def test_score_is_organization_times_request_progress(self):
        # a sorted arrival (+1 of 1) and an arrival left loose (0 of 1): O(t) = 1/2 at the close and at the end
        loose = [{"id": 0, "present": True, "value": 1, "baseline": 0}, {"id": 1, "present": True, "value": 0, "baseline": 0}]
        self.scorer = score.EpisodeScorer(horizon=4)
        self.scorer.add_step({"objects": loose, "events": [{"event": "request_opened", "request": 0}]})
        self.scorer.add_step({"objects": loose, "events": [{"event": "request_first_hold", "request": 0}]})
        self.scorer.add_step({"objects": loose, "events": [{"event": "request_delivered", "request": 0}]})
        self.scorer.add_step({"objects": loose, "events": [{"event": "request_opened", "request": 1}]})
        self.assertEqual(self.scorer.organization(), 0.5)
        self.assertEqual(self.scorer.request_progress(), 0.5)      # (2 + 0) / 4: the second request is open
        self.assertEqual(self.scorer.score(), 0.25)
        self.assertEqual(self.scorer.live()["score_running"], 0.25)
        # a wrong mark lowers R and with it S; with R > 0 a tidier desk always scores more
        self.scorer.add_step({"objects": loose, "events": [{"event": "request_absent_wrong", "request": 1}]})
        self.assertEqual(self.scorer.request_progress(), 0.25)
        tidy = [dict(loose[0]), dict(loose[1], value=1)]
        before = self.scorer.score()
        self.scorer.add_step({"objects": tidy, "events": []})
        self.assertGreater(self.scorer.score(), before)

    def test_a_cut_episode_is_scaled_by_the_steps_that_ran(self):
        self.scorer = score.EpisodeScorer(horizon=4)
        sorted_ = [{"id": 0, "present": True, "value": 1, "baseline": 0}]
        self.scorer.add_step({"objects": sorted_, "events": [{"event": "request_opened", "request": 0}]})
        self.scorer.add_step({"objects": sorted_, "events": [{"event": "request_absent_correct", "request": 0}]})
        self.assertEqual(self.scorer.score_unscaled(), 1.0)
        self.assertEqual(self.scorer.score(), 0.5)                  # 2 of 4 steps ran

    def world(self, target_state="held"):
        target = {"id": 0, "present": True, "state": target_state, "in_container": None}
        other = {"id": 1, "present": True, "state": "held", "in_container": None}
        return SimpleNamespace(
            t=10, rng=None, events=[], objects=[target, other], touched={0},
            request={"id": 0, "object": 0, "status": "open", "deadline_t": 100},
            requests_done=[], next_request=math.inf, next_arrival=math.inf,
            next_disturb=math.inf,
        )

    def test_current_target_hold_only_and_once_per_request(self):
        world = self.world("on_table")
        env.MessyDeskEnv._world_events(world, 0)
        self.assertEqual(world.events, [])  # Historical contact and another held object do not count.
        world.objects[0]["state"] = "held"
        env.MessyDeskEnv._world_events(world, 0)
        self.assertEqual([e["event"] for e in world.events], ["request_first_hold"])
        world.events.clear()
        world.objects[0]["state"] = "on_table"
        env.MessyDeskEnv._world_events(world, 0)
        world.objects[0]["state"] = "held"
        env.MessyDeskEnv._world_events(world, 0)
        self.assertEqual(world.events, [])
        world.request = {"id": 1, "object": 0, "status": "open", "deadline_t": 100}
        env.MessyDeskEnv._world_events(world, 0)
        self.assertEqual(world.events[0]["request"], 1)

    def test_first_hold_on_closing_step_is_not_dropped(self):
        world = self.world()
        env.MessyDeskEnv._world_events(world, 1)  # Still held: the delivery claim is wrong.
        self.assertEqual([e["event"] for e in world.events],
                         ["request_first_hold", "request_claim_wrong"])
        self.scorer.add_step({"objects": [], "events": world.events})
        self.assertEqual(self.scorer.request_points[0], 0)   # the hold's point, minus the wrong mark
        self.assertEqual(self.scorer.request_progress(), 0)

    def test_a_touch_is_not_a_hold(self):
        world = SimpleNamespace(t=10, request={"id": 0, "object": 0, "status": "open", "deadline_t": 100})
        world._is_target = lambda o: env.MessyDeskEnv._is_target(world, o)
        touched = {"id": 0, "cls": "A", "present": True, "removed": False, "contact": True, "held": False,
                   "resting": True, "in_container": None}
        self.assertEqual(env.MessyDeskEnv._value(world, touched), 0)
        lifted = dict(touched, held=True, resting=False)
        self.assertEqual(env.MessyDeskEnv._value(world, lifted), 1)
        in_pile = dict(touched, in_container="gray")             # touched where it lies: not placed, not held
        self.assertEqual(env.MessyDeskEnv._value(world, in_pile), 0)
        self.assertEqual(env.MessyDeskEnv._value(world, dict(in_pile, contact=False)), -1)



class DeskLabTests(unittest.TestCase):
    """The agent gets the desk lab, not the episode (2026-11-07): the lab is tests/review/make_lab.py's derivation of
    the grader's env.py (the same scene, robot, cameras, physics and action rule), and nothing agent-visible carries
    the world's state machine, the scorer or a practice grader."""

    def test_the_lab_is_the_derivation_of_the_graders_env(self):
        import subprocess
        import sys
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "lab.py"
            subprocess.run([sys.executable, "-I", str(TASK / "tests/review/make_lab.py"), str(TASK / "tests/env.py"),
                            str(out)], check=True, capture_output=True)
            self.assertEqual(out.read_bytes(), (TASK / "environment/setup/payload/messy_desk_lab.py").read_bytes(),
                             "re-run tests/review/make_lab.py after changing tests/env.py")

    def test_nothing_agent_visible_plays_or_scores_an_episode(self):
        payload = TASK / "environment/setup/payload"
        names = sorted(p.relative_to(payload).as_posix() for p in payload.rglob("*") if p.is_file())
        self.assertEqual(names, ["example_controller/controller.py", "example_controller/run.sh", "lab_client.py",
                                 "messy_desk_lab.py", "task_card.md"])
        lab = (payload / "messy_desk_lab.py").read_text()
        for needle in ("_world_events", "_open_request", "force_request", "EpisodeScorer", "messy_desk_score",
                       "REQUEST_ABSENT_P", "ARRIVAL_MEAN_STEPS", "DISTURB_MEAN_STEPS", "CONTAINER_CAP", "baseline",
                       "def privileged", "def _value", "def gains"):
            self.assertNotIn(needle, lab)


class SensorTests(unittest.TestCase):
    """The cameras are real sensors (2026-11-07): each desk's fixed cameras carry a mounting error and depth is noisy
    with missing edge pixels. Both are drawn from a stream of the episode's seed of their own, never from the world's
    generator, so the world (and the privileged reference, which reads no pixels) is the same episode as before."""

    def test_the_sensors_never_draw_from_the_worlds_stream(self):
        import ast
        text = (TASK / "tests/env.py").read_text()
        found = set()
        for node in ast.walk(ast.parse(text)):
            if isinstance(node, ast.FunctionDef) and node.name in ("_mount_cameras", "_depth_sensor"):
                src = ast.get_source_segment(text, node)
                self.assertNotIn("self.rng", src, node.name)
                self.assertIn("default_rng([self._sensor_key, SENSOR_STREAM", src, node.name)
                found.add(node.name)
        self.assertEqual(found, {"_mount_cameras", "_depth_sensor"})
        self.assertIn("self._mount_cameras(seed)", text)

    def test_info_reports_the_nominal_poses_only(self):
        text = (TASK / "tests/env.py").read_text()
        info = text[text.index("    def info(self):"):text.index("    def close(self):")]
        self.assertIn("CAMERA_NOMINAL[name]", info)

    @unittest.skipUnless(importlib.util.find_spec("numpy"), "numpy is not installed here")
    def test_depth_noise_is_deterministic_and_of_the_documented_size(self):
        import numpy as np
        fake = SimpleNamespace(np=np, _sensor_key=12345, t=7)
        z = np.ones((200, 200), np.float32)
        z[:, 100:] = 1.5                        # a 0.5 m jump down the middle
        z[:5, :5] = 0.0                         # pixels that see nothing
        a = env.MessyDeskEnv._depth_sensor(fake, z, 0)
        b = env.MessyDeskEnv._depth_sensor(fake, z, 0)
        self.assertTrue(np.array_equal(a, b))
        self.assertFalse(np.array_equal(a, env.MessyDeskEnv._depth_sensor(fake, z, 1)))
        self.assertTrue(np.all(a[:5, :5] == 0))
        interior = a[10:190, 10:90]
        self.assertAlmostEqual(float(np.std(interior)), env.DEPTH_NOISE_1M, delta=0.2 * env.DEPTH_NOISE_1M)
        edge = a[:, 99:101]
        self.assertGreater(float(np.mean(edge == 0)), 0.3)
        self.assertLess(float(np.mean(edge == 0)), 0.7)
        # a plane seen at the steepest slope the fixed cameras meet (inverse depth falling 0.006 per metre a pixel,
        # 0.8 m to 6 m away) loses nothing; with edges off (the hand camera) not even the step does
        slope = np.tile((1.0 / np.linspace(1.25, 1.25 - 0.006 * 199, 200)).astype(np.float32), (200, 1))
        self.assertEqual(int(np.sum(env.MessyDeskEnv._depth_sensor(fake, slope, 2) == 0)), 0)
        self.assertEqual(int(np.sum(env.MessyDeskEnv._depth_sensor(fake, z, 3, edges=False)[5:, 5:] == 0)), 0)


if __name__ == "__main__":
    unittest.main()
