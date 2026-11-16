"""kitchen_appliance_tending: the agent practises with exactly the environment the lab and the grader run.

The environment (appliance_env.py), the episode logic (episode.py) and the cameras' renderer beside them
(kitchen_render.py, kitchen_scene.py, kitchen_look.json) live in four byte-identical copies: tests/ (the grader),
environment/setup/payload/ (the agent's home), environment/setup/lab/ (the lab service) and solution/oracle/ (the
reference). environment/render_check/ holds one more copy of the environment and the renderer for the image's
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
SHARED = ("appliance_env.py", "episode.py", "kitchen_render.py", "kitchen_scene.py", "kitchen_look.json")
RENDER_CHECK = ("appliance_env.py", "kitchen_render.py", "kitchen_scene.py", "kitchen_look.json")
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
        for key in ("exposure_scale", "ibl", "lamps", "headlight", "room", "dressing", "model_materials", "flame",
                    "indicator", "filament_numeric", "filament_text", "render"):
            self.assertIn(key, look)


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
