"""kitchen_appliance_tending: the agent practises with exactly the environment the lab and the grader run.

The environment (appliance_env.py), the episode logic (episode.py) and the cameras' renderer beside them
(kitchen_render.py, kitchen_fx.py, kitchen_scene.py, kitchen_look.json) live in four byte-identical copies: tests/
(the grader), environment/setup/payload/ (the agent's home), environment/setup/lab/ (the lab service) and
solution/oracle/ (the reference). environment/render_check/ holds one more copy of the environment and the renderer for the image's
build-time render check (the image's build context is environment/ less setup/). The vendored grading kit is the
canonical one. The practice units' models and the room bundle are task assets (docs/task-assets.md): listed in
environment/assets.sha256, not in git, and the renderer reads the bundle where the image puts it.

Standard library only (the host-side suite has no numpy): nothing of the task is imported.
"""

import hashlib
import json
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / "tasks/arm/kitchen_appliance_tending"
COPIES = ("tests", "environment/setup/payload", "environment/setup/lab", "solution/oracle")
SHARED = ("appliance_env.py", "episode.py", "kitchen_render.py", "kitchen_fx.py", "kitchen_scene.py",
          "kitchen_look.json")
RENDER_CHECK = ("appliance_env.py", "kitchen_render.py", "kitchen_fx.py", "kitchen_scene.py", "kitchen_look.json")
BUNDLE = ("render/room.npz", "render/room.json", "render/ibl.ktx", "render/sky.ktx")
RENDER_DIR = "/opt/task-assets/payload/render"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Copies(unittest.TestCase):

    def test_the_four_copies_are_byte_identical(self):
        for name in SHARED:
            grader = TASK / "tests" / name
            self.assertTrue(grader.is_file(), name)
            for copy in COPIES[1:]:
                self.assertEqual(sha256(TASK / copy / name), sha256(grader), "{} differs in {}".format(name, copy))

    def test_the_render_check_copy_is_byte_identical(self):
        for name in RENDER_CHECK:
            self.assertEqual(sha256(TASK / "environment/render_check" / name), sha256(TASK / "tests" / name), name)

    def test_the_vendored_kit_is_the_canonical(self):
        canonical = {p.name: sha256(p) for p in (ROOT / "shared/robotics_grader").glob("*.py")}
        vendored = {p.name: sha256(p) for p in (TASK / "tests/robotics_grader").glob("*.py")}
        self.assertEqual(vendored, canonical)

    def test_the_look_parses(self):
        look = json.loads((TASK / "tests/kitchen_look.json").read_text())
        for key in ("exposure_scale", "ibl", "lamps", "headlight", "room", "dressing", "model_materials", "burner_fx",
                    "indicator", "filament_numeric", "filament_text", "render"):
            self.assertIn(key, look)


class Budget(unittest.TestCase):
    """The numbers the instruction states about grading are the ones the grader and the practice grader use."""

    def setUp(self):
        self.config = json.loads((TASK / "tests/grader_config.json").read_text())
        self.instruction = (TASK / "instruction.md").read_text()
        self.verify = (TASK / "tests/verify.py").read_text()
        self.practice = (TASK / "environment/setup/payload/practice_grader/grade.py").read_text()

    def test_the_budget_is_controller_time_with_a_fixed_image_price(self):
        self.assertEqual(self.config["grader_budget_s"], 6000)
        self.assertEqual(self.config["image_cost_s"], 0.03)
        self.assertIn("time budget of 6000 s", self.instruction)
        self.assertIn("a fixed price of 0.03 s for every image", self.instruction)
        self.assertNotIn("time.monotonic() - t_start > budget", self.verify)      # wall time is not the budget
        self.assertIn("IMAGE_COST = 0.03", self.practice)
        self.assertIn("BUDGET_PER_EPISODE = 6000.0 / 18", self.practice)
        card = (TASK / "environment/setup/payload/task_card.md").read_text()
        self.assertIn("costs 0.03 s of the controller's 6000 s budget", card)

    def test_the_image_bill_of_a_full_run_is_as_stated(self):
        price = self.config["image_cost_s"]
        self.assertIn("between 54,000 and 63,000 steps", self.instruction)
        for steps, images, text in ((54000, 3, "4,860"), (63000, 3, "5,670"), (54000, 1, "1,620"), (63000, 1, "1,890")):
            self.assertEqual("{:,}".format(round(steps * images * price)), text)
            self.assertIn(text, self.instruction)

    def test_the_pause_and_the_process_cap_are_stated(self):
        self.assertIn("MAX_PROCESSES = 512", self.verify)
        self.assertIn("FREEZE_TIMEOUT_S = 5.0", self.verify)
        self.assertIn("more than 512 processes of your account", self.instruction)
        self.assertIn("do not stop within 5 s", self.instruction)
        self.assertIn("more than 512 processes of your account", self.practice)

    def test_the_verifier_timeout_covers_the_worst_case(self):
        toml = (TASK / "task.toml").read_text()
        timeout = int(re.search(r"\[verifier\]\s*\ntimeout_sec = (\d+)", toml).group(1))
        self.assertGreaterEqual(timeout, 3 * self.config["grader_budget_s"])

    def test_the_probe_pair_rule_is_the_kits(self):
        # since the kit's value-only lock, two replies of different lengths do not count as differing
        self.assertIn("return len(aa) == len(bb) and max(", self.practice)
        self.assertIn("different count of numbers do not count as", self.instruction)


class Numerics(unittest.TestCase):

    def test_the_pin_is_in_the_image_and_checked_before_grading(self):
        docker = (TASK / "environment/Dockerfile").read_text()
        for needle in ('"OPENBLAS_CORETYPE", "Haswell"', '"OPENBLAS_L2_SIZE", "256"', '"OPENBLAS_NUM_THREADS", "1"',
                       '"NPY_DISABLE_CPU_FEATURES", "X86_V4 AVX512_ICL AVX512_SPR"', '"OPENCV_IPP", "avx2"',
                       "kitchen_numerics_pin.pth", "-W error::ImportWarning"):
            self.assertIn(needle, docker)
        ref = json.loads((TASK / "tests/numerics.json").read_text())
        self.assertEqual(len(ref["fingerprint"]), 64)
        verify = (TASK / "tests/verify.py").read_text()
        self.assertIn("check_numerics()", verify)
        instruction = (TASK / "instruction.md").read_text()
        for name in ("OPENBLAS_CORETYPE=Haswell", "OPENBLAS_L2_SIZE=256", "OPENBLAS_NUM_THREADS=1", "OMP_NUM_THREADS=1",
                     'NPY_DISABLE_CPU_FEATURES="X86_V4 AVX512_ICL AVX512_SPR"', "OPENCV_IPP=avx2"):
            self.assertIn(name, instruction)

    def test_nothing_tells_agents_to_import_from_outside_the_submission(self):
        skeleton = (TASK / "environment/setup/payload/helpers/wire_skeleton.py").read_text()
        self.assertNotIn('sys.path.insert(0, "/home/user/helpers")', skeleton)


class Assets(unittest.TestCase):

    def _listed(self):
        lines = (TASK / "environment/assets.sha256").read_text().splitlines()
        return [line.split("  ", 1)[1] for line in lines if re.match(r"^[0-9a-f]{64}  ", line)]

    def test_the_practice_units_and_the_bundle_are_listed_not_in_git(self):
        listed = self._listed()
        units = json.loads((TASK / "environment/setup/payload/units.json").read_text())
        for uid, unit in units.items():
            self.assertFalse((TASK / "environment/setup/payload" / unit["dir"]).exists(), uid)
            self.assertIn(unit["dir"] + "/model.xml", listed, uid)
        for rel in BUNDLE:
            self.assertIn(rel, listed)
        allowed = tuple(u["dir"] + "/" for u in units.values()) + ("render/",)
        self.assertEqual([rel for rel in listed if not rel.startswith(allowed)], [])

    def test_no_hidden_unit_is_an_asset(self):
        listed = "\n".join(self._listed())
        for uid, unit in json.loads((TASK / "tests/units.json").read_text()).items():
            self.assertNotIn(unit["dir"].split("/")[-1], listed, uid)

    def test_the_renderer_reads_the_bundle_where_the_image_puts_it(self):
        dockerfile = (TASK / "environment/Dockerfile").read_text()
        self.assertIn("ENV KITCHEN_RENDER_DIR=" + RENDER_DIR + "\n", dockerfile)
        src = (TASK / "tests/kitchen_render.py").read_text()
        self.assertIn('os.environ.get("KITCHEN_RENDER_DIR", "{}")'.format(RENDER_DIR), src)
        self.assertIn("mesa-vulkan-drivers", dockerfile)

    def test_setup_stages_the_practice_units_into_the_home(self):
        setup = (TASK / "environment/setup/setup.sh").read_text()
        self.assertIn('cp -r /opt/task-assets/payload/units/. "$HOME_DIR/units/"', setup)
        self.assertNotIn("payload/render", setup)


if __name__ == "__main__":
    unittest.main()
