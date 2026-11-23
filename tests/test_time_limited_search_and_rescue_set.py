"""time_limited_search_and_rescue: the fixed hidden set, its pre-registered rule, the copies, the cameras and the stated numbers.

The grader flies the agent's program on the eight sites of tests/hidden/sites.json. They are chosen by the rule in
tests/review/selection.md (one site per stratum of mirror x open station x flange size, spread over the privileged
pilot's route length) from 96 fixed candidates. The simulator and the client the verifier uses are copies of the ones
the agent's sandbox runs; the numbers the instruction states are the ones the code enforces.

Standard library only: the task's Python files are read as text, never imported.
"""

import hashlib
import json
import re
import unittest
from pathlib import Path

TASK = Path(__file__).resolve().parents[1] / "tasks/uav/time_limited_search_and_rescue"
HIDDEN = TASK / "tests/hidden"
INSTRUCTION = (TASK / "instruction.md").read_text()


def number(pattern, text):
    return float(re.search(pattern, text, re.M).group(1).replace(",", "").replace("_", ""))


class HiddenSet(unittest.TestCase):
    def setUp(self):
        self.sites = json.loads((HIDDEN / "sites.json").read_text())["seeds"]
        self.candidates = json.loads((HIDDEN / "candidates.json").read_text())
        self.selection = json.loads((HIDDEN / "selection.json").read_text())

    def test_candidates_are_the_registered_range(self):
        script = (TASK / "tests/review/select_sites.py").read_text()
        base, count = int(number(r"^BASE, COUNT, RULE = (\d+)", script)), len(self.candidates)
        self.assertEqual([c["seed"] for c in self.candidates], list(range(base, base + count)))
        self.assertEqual(count % 96, 0)
        self.assertEqual(self.selection["candidates"], count)
        self.assertIn("39000000 + k", (TASK / "tests/review/selection.md").read_text())

    def test_rule_reproduces_the_set(self):
        done = [c for c in self.candidates if c["delivered"]]
        self.assertEqual(self.selection["reference_completed"], len(done))
        strata = [[m, s, f] for m in (0, 1) for s in (0, 1) for f in (0, 1)]
        chosen = []
        for i, stratum in enumerate(strata):
            members = sorted((c for c in done if c["stratum"] == stratum), key=lambda c: (c["actions"], c["seed"]))
            self.assertTrue(members)
            chosen.append(members[int((i + .5) / 8 * len(members))]["seed"])
        self.assertEqual(chosen, self.sites)
        self.assertEqual(len(set(self.sites)), 8)

    def test_strata_follow_the_seed_bits(self):
        for c in self.candidates:
            self.assertEqual(c["stratum"][:2], [(c["seed"] >> 1) & 1, (c["seed"] >> 2) & 1])

    def test_commissioning_sites_cannot_be_candidates(self):
        service = (TASK / "environment/setup/runtime/service.py").read_text()
        self.assertIn("secrets.randbits(63) | (1 << 63)", service)
        self.assertTrue(all(seed < 2 ** 32 for seed in self.sites))

    def test_anchor_is_the_reference_on_the_set(self):
        anchor = json.loads((TASK / "tests/anchor.json").read_text())
        self.assertEqual(anchor["metric"], "rescue_progress")
        self.assertEqual(len(anchor["measurements"]), 8)
        self.assertAlmostEqual(anchor["value"], sum(anchor["measurements"]) / 8)
        self.assertGreaterEqual(anchor["value"], .98)
        self.assertEqual(anchor["full_at"], .98)


class CopiesAndNumbers(unittest.TestCase):
    def test_verifier_runs_the_sandbox_simulator_and_client(self):
        for folder, copy in (("environment/setup/runtime", "tests/runtime"),
                             ("environment/setup/payload", "tests/setup_payload")):
            names = sorted(p.name for p in (TASK / folder).iterdir() if p.is_file())
            self.assertEqual(names, sorted(p.name for p in (TASK / copy).iterdir() if p.is_file()))
            for name in names:
                self.assertEqual(hashlib.sha256((TASK / folder / name).read_bytes()).hexdigest(),
                                 hashlib.sha256((TASK / copy / name).read_bytes()).hexdigest(), name)

    def test_oracle_token_matches_its_digest(self):
        token = (TASK / "solution/oracle/oracle_token").read_bytes().strip()
        self.assertEqual(hashlib.sha256(token).hexdigest(), (TASK / "tests/oracle_token.sha256").read_text().split()[0])
        self.assertNotIn("_gt", INSTRUCTION)

    def test_truth_is_token_gated_and_never_in_commissioning(self):
        service = (TASK / "tests/runtime/service.py").read_text()
        self.assertIn('self.truth = bool(truth) and mode == "graded"', service)
        setup = (TASK / "environment/setup/setup_runtime.py").read_text()
        self.assertNotIn("--truth", setup)
        self.assertIn('"--mode", "commissioning"', setup)

    def test_budgets_in_the_instruction_are_the_enforced_ones(self):
        service = (TASK / "tests/runtime/service.py").read_text()
        world = (TASK / "tests/runtime/rescue.py").read_text()
        config = json.loads((TASK / "tests/grader_config.json").read_text())
        self.assertEqual(number(r"^LIMIT = (\d+)", world), 4400)
        self.assertIn("4,400 actions", INSTRUCTION)
        self.assertEqual(number(r"^REQUESTS = (\d+)", service), 9000)
        self.assertIn("9,000 calls", INSTRUCTION)
        self.assertEqual(number(r"^COMMISSIONING_ACTIONS = (\d+)", service), 16000)
        self.assertIn("**16,000 actions**", INSTRUCTION)
        self.assertEqual(number(r"^COMMISSIONING_SITES = (\d+)", service), 40)
        self.assertIn("**40 sites**", INSTRUCTION)
        self.assertEqual(config["program_seconds"], 900)
        self.assertIn("**900 s of its own time**", INSTRUCTION)
        self.assertEqual(config["mission_wall_seconds"], 2100)
        self.assertIn("wall limit of 2,100 s", INSTRUCTION)
        # the verifier's limit covers eight flights at the wall limit and the start of the verifier
        self.assertEqual(number(r"^\[verifier\]\ntimeout_sec = (\d+)", (TASK / "task.toml").read_text()), 8 * 2100 + 1200)

    def test_session_time_is_one_number(self):
        toml = (TASK / "task.toml").read_text()
        script = (TASK / "environment/setup/payload/session_time.sh").read_text()
        self.assertEqual(number(r"^\[agent\]\ntimeout_sec = (\d+)", toml), 7200)
        self.assertEqual(number(r"^LIMIT_MIN=(\d+)", script), 120)
        self.assertIn("2 hours of wall-clock time", INSTRUCTION)
        self.assertIn("/run/session", (TASK / "environment/setup/setup_runtime.py").read_text())
        self.assertIn('environment_mode = "separate"', toml)

    def test_progress_weights_in_the_instruction(self):
        world = (TASK / "tests/runtime/rescue.py").read_text()
        table = re.search(r"^SCORE = \{(.+?)\}", world, re.M | re.S).group(1)
        weights = {k: float(v) for k, v in re.findall(r'"(\w+)": (\.\d+)', table)}
        self.assertEqual(weights, {"entered_wing": .05, "mezzanine": .04, "located": .06, "grasped": .15,
                                   "lifted": .10, "cargo_return": .25})
        self.assertEqual(number(r"^ROOMS = (\.\d+)", world), .03)
        self.assertAlmostEqual(sum(weights.values()) + .03, .68)
        # reaching the litter is worth little; the handoff is the largest single step
        self.assertAlmostEqual(weights["entered_wing"] + weights["mezzanine"] + weights["located"], .15)
        self.assertGreater(1 - sum(weights.values()) - .03, max(weights.values()))
        for text in ("| 0.05 |", "| + 0.04 |", "| + 0.06 |", "| + 0.15 |", "| + 0.10 |", "| + 0.25 |",
                     "| + up to 0.03 |", "at most 0.68"):
            self.assertIn(text, INSTRUCTION)

    def test_cameras_are_one_worker_process_per_site(self):
        """A picture is a function of the site and the state: the world draws through a worker process of its
        own (rescue_render.Cameras), never through a renderer that lives in the service's process."""
        world = (TASK / "tests/runtime/rescue.py").read_text()
        self.assertIn("self.renderer = Cameras(seed) if render else None", world)
        self.assertNotIn("mujoco.Renderer", world)
        render = (TASK / "tests/runtime/rescue_render.py").read_text()
        self.assertIn('"--worker"', render)
        self.assertIn("setd(self.view, 0)", render)             # Filament's temporal dithering is switched off
        look = json.loads((TASK / "tests/runtime/rescue_look.json").read_text())
        self.assertFalse(look["render"].get("shadows", False))
        self.assertNotIn("mark_extras", look)
        self.assertNotIn("without_extras", look)

    def test_render_bundle_holds_every_map_the_look_names(self):
        listed = {line.split("  ", 1)[1].strip() for line in (TASK / "environment/assets.sha256").read_text().splitlines()
                  if line and not line.startswith("#")}
        look = json.loads((TASK / "tests/runtime/rescue_look.json").read_text())
        self.assertIn("render/" + look["ibl"]["file"], listed)
        for name, rule in look["classes"].items():
            if rule.get("map"):
                self.assertIn("render/%s.png" % rule["map"], listed, name)
        # floor markings are worn paint in the model's own colours: one map per colour of rescue_scene.py
        scene = (TASK / "tests/runtime/rescue_scene.py").read_text()

        def colour(name):
            r, g, b = re.search(r"^%s = \(([\d.]+), ([\d.]+), ([\d.]+), 1\)" % name, scene, re.M).groups()
            return "%02x%02x%02x" % tuple(int(round(255 * float(v))) for v in (r, g, b))
        for name in ("PURPLE", "BLUE", "AMBER", "WHITE", "TEAL", "LIME"):
            self.assertIn("render/paint_%s.png" % colour(name), listed, name)
        self.assertIn("render/mat_%s.png" % colour("TEAL"), listed)

    def test_dressing_rule_is_stated_and_its_models_are_pinned(self):
        listed = {line.split("  ", 1)[1].strip() for line in (TASK / "environment/assets.sha256").read_text().splitlines()
                  if line and not line.startswith("#")}
        for name in ("look.tar", "mark_base.png", "norm_metal.png", "norm_wall.png", "norm_floor.png", "norm_rubber.png"):
            self.assertIn("render/" + name, listed)
        self.assertIn("tar -xf /opt/task-assets/payload/render/look.tar", (TASK / "environment/Dockerfile").read_text())
        rule = (TASK / "tests/runtime/rescue_dressing.py").read_text()
        self.assertIn("LANE_HALF, LOW_TOP, TURN_R = 0.72, 0.80, 1.25", rule)
        for words in ("within 0.72 m of a lane", "taller than 0.80 m stands within 1.25 m", "every piece of stock and furniture",
                      "both cameras sit clear of the rotor blur", "through the guide ring, from above"):
            self.assertIn(words, " ".join(INSTRUCTION.split()))
        world = (TASK / "tests/runtime/rescue.py").read_text()
        self.assertIn('startswith("look_")', world)            # drawn models are not contact pieces
        self.assertIn('obstacles.append(("prop", name))', rule)
        for name in ("rescue_dressing.py", "rescue_marks.py", "rescue_render.py", "rescue_look.json", "rescue_scene.py", "rescue.py"):
            self.assertEqual((TASK / "tests/runtime" / name).read_bytes(), (TASK / "environment/setup/runtime" / name).read_bytes(), name)

    def test_render_bundle_is_roots_and_checked_at_build(self):
        docker = (TASK / "environment/Dockerfile").read_text()
        self.assertIn("chmod -R u=rwX,go= /opt/task-assets", docker)
        self.assertIn("! runuser -u user -- ls /opt/task-assets", docker)
        self.assertIn("render_check.py", docker)
        self.assertIn("An answer takes the aircraft about 0.24 s", INSTRUCTION)

    def test_grader_logs_no_seed(self):
        grader = (TASK / "tests/grader.py").read_text()
        self.assertNotIn("authoritative-journal", grader)
        for line in grader.splitlines():
            if "log(" in line:
                self.assertNotIn("seed", line)


if __name__ == "__main__":
    unittest.main()
