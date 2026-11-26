"""pipe_spool_push_fit_assembly: the agent practises with exactly the simulator and the renderer the grader runs, and
what the instruction says about the session clock, the episodes, the pictures and the numerics pin holds in the files.

The simulator, the environment, the episode logic and what the cameras draw live in three byte-identical copies:
tests/ (the grader), environment/setup/payload/ (the agent's home) and solution/oracle/ (the reference).

Standard library only (the host-side suite has no numpy): nothing of the task is imported.
"""

import hashlib
import json
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / "tasks/arm/pipe_spool_push_fit_assembly"
COPIES = ("tests", "environment/setup/payload", "solution/oracle")
SHARED = ("spool_env.py", "spool_sim.py", "episode.py",
          "spool_render.py", "spool_scene.py", "spool_parts.py", "spool_geom.py", "spool_looks.py", "spool_look.json")
PIN = {"OPENBLAS_CORETYPE": "Haswell", "OPENBLAS_L2_SIZE": "256",
       "NPY_DISABLE_CPU_FEATURES": "X86_V4 AVX512_ICL AVX512_SPR",
       "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "OPENCV_IPP": "avx2"}
EPISODES = 12


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def toml_int(text, table, key):
    body = text[text.index("[{}]".format(table)):]
    return int(re.search(r"^{}\s*=\s*(\d+)".format(key), body, re.M).group(1))


class Copies(unittest.TestCase):

    def test_the_three_copies_are_byte_identical(self):
        for name in SHARED:
            grader = TASK / "tests" / name
            self.assertTrue(grader.is_file(), name)
            for copy in COPIES[1:]:
                self.assertEqual(sha256(TASK / copy / name), sha256(grader), "{} differs in {}".format(name, copy))

    def test_the_instruction_names_every_drawn_file(self):
        text = (TASK / "instruction.md").read_text()
        for name in SHARED:
            self.assertIn("`{}`".format(name), text, name)


class SessionTime(unittest.TestCase):

    def test_script_instruction_and_task_agree_on_two_hours(self):
        script = (TASK / "environment/setup/payload/session_time.sh").read_text()
        self.assertEqual(int(re.search(r"^LIMIT_MIN=(\d+)", script, re.M).group(1)) * 60,
                         toml_int((TASK / "task.toml").read_text(), "agent", "timeout_sec"))
        text = (TASK / "instruction.md").read_text()
        self.assertIn("## Session time", text)
        self.assertIn("bash /home/user/session_time.sh", text)


class HiddenSet(unittest.TestCase):

    def test_twelve_episodes_one_per_hidden_unit_from_the_latest_draw(self):
        eps = json.loads((TASK / "tests/hidden/episodes.json").read_text())["episodes"]
        units = json.loads((TASK / "tests/units.json").read_text())["units"]
        self.assertEqual(len(eps), EPISODES)
        self.assertEqual(sorted(e["unit"] for e in eps), sorted(units))
        self.assertEqual(len({int(e["seed"]) for e in eps}), EPISODES)
        rec = json.loads((TASK / "tests/hidden/episode_selection.json").read_text())
        last = rec["draws"][-1]
        self.assertEqual(last["status"], "made")
        self.assertTrue(all(d["status"] == "void" and d.get("void_reason") for d in rec["draws"][:-1]))
        kept = {(c["unit"], int(c["seed"])) for c in rec["candidates"] if c["draw"] == last["draw"] and c["status"] == "kept"}
        self.assertTrue({(e["unit"], int(e["seed"])) for e in eps} <= kept)

    def test_the_frozen_hashes_are_the_files(self):
        rec = json.loads((TASK / "tests/hidden/episode_selection.json").read_text())
        for rel, digest in rec["frozen"].items():
            self.assertEqual(sha256(TASK / rel), digest, rel)

    def test_no_practice_unit_is_a_hidden_unit(self):
        hidden = json.loads((TASK / "tests/units.json").read_text())["units"]
        practice = json.loads((TASK / "environment/setup/payload/units.json").read_text())["units"]
        for p in practice.values():
            self.assertNotIn(p, list(hidden.values()))


class Pictures(unittest.TestCase):

    def test_the_look_rules_read_no_hidden_property_but_the_bore(self):
        scene = (TASK / "tests/spool_scene.py").read_text()
        parts = (TASK / "tests/spool_parts.py").read_text()
        for key in ("stop", "f_ins", "torque", "elbow_mass", "pipe_kg_m"):
            self.assertNotIn('unit["{}"]'.format(key), scene, key)
            self.assertNotIn('unit["{}"]'.format(key), parts, key)

    def test_the_bundle_is_pinned_and_the_image_reads_it(self):
        pin = (TASK / "environment/assets.sha256").read_text()
        self.assertRegex(pin, r"# revision: [0-9a-f]{40}")
        for name in ("ibl.ktx", "room.json", "room.npz", "textures.npz", "props.npz"):
            self.assertIn("render/" + name, pin)
        docker = (TASK / "environment/Dockerfile").read_text()
        self.assertIn("SPOOL_RENDER_DIR=/opt/task-assets/payload/render", docker)
        self.assertIn("mesa-vulkan-drivers", docker)


class Numerics(unittest.TestCase):

    def test_the_image_pins_what_the_launcher_passes(self):
        docker = (TASK / "environment/Dockerfile").read_text()
        launcher = (TASK / "tests/nonet_exec.py").read_text()
        for key, value in PIN.items():
            self.assertIn(key, docker)
            self.assertIn(value, docker)
            self.assertIn(key, launcher)


if __name__ == "__main__":
    unittest.main()
