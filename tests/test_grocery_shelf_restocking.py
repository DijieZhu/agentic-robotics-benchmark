"""grocery_shelf_restocking: the agent practises with exactly the simulator the grader runs, and what the instruction
says about the session clock, the controller's time budget and the numerics pin holds in the files.

The world, the environment, the controller and the episode logic live in three byte-identical copies: tests/ (the
grader), environment/setup/payload/ (the agent's home) and solution/oracle/ (the reference).

Standard library only (the host-side suite has no numpy): nothing of the task is imported.
"""

import hashlib
import json
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / "tasks/arm/grocery_shelf_restocking"
COPIES = ("tests", "environment/setup/payload", "solution/oracle")
SHARED = ("box_transport_env.py", "episode.py", "shelf_ctl.py", "shelf_env.py", "shelf_sim.py", "shelf_world.py")
PIN = {"OPENBLAS_CORETYPE": "Haswell", "OPENBLAS_L2_SIZE": "256",
       "NPY_DISABLE_CPU_FEATURES": "X86_V4 AVX512_ICL AVX512_SPR",
       "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "OPENCV_IPP": "avx2"}


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


class SessionTime(unittest.TestCase):

    def test_script_instruction_and_task_agree_on_two_hours(self):
        script = (TASK / "environment/setup/payload/session_time.sh").read_text()
        self.assertEqual(int(re.search(r"^LIMIT_MIN=(\d+)", script, re.M).group(1)) * 60,
                         toml_int((TASK / "task.toml").read_text(), "agent", "timeout_sec"))
        text = (TASK / "instruction.md").read_text()
        self.assertIn("## Session time", text)
        self.assertIn("bash /home/user/session_time.sh", text)
        self.assertIn("stopped 2 hours of wall-clock time", text)

    def test_the_stamp_is_roots(self):
        setup = (TASK / "environment/setup/setup.sh").read_text()
        self.assertIn("install -d -o root -g root -m 755 /run/session", setup)
        self.assertIn("chmod 0444 /run/session/start", setup)
        self.assertIn("STAMP=/run/session/start", (TASK / "environment/setup/payload/session_time.sh").read_text())


class Budget(unittest.TestCase):

    def test_the_numbers_in_the_instruction_are_the_graders(self):
        config = json.loads((TASK / "tests/grader_config.json").read_text())
        text = (TASK / "instruction.md").read_text()
        self.assertIn("budget of {} s of its own time".format(config["grader_budget_s"]), text)
        self.assertIn("plus {} s for every image".format(config["image_cost_s"]), text)
        self.assertIn("within {} s".format(config["step_timeout_s"]), text)
        practice = (TASK / "environment/setup/payload/practice_grader/grade.py").read_text()
        self.assertIn("IMAGE_COST = {}".format(config["image_cost_s"]), practice)
        self.assertIn("BUDGET_PER_EPISODE = {}.0 / 12".format(config["grader_budget_s"]), practice)

    def test_the_verifier_timeout_covers_simulation_and_budget(self):
        config = json.loads((TASK / "tests/grader_config.json").read_text())
        timeout = toml_int((TASK / "task.toml").read_text(), "verifier", "timeout_sec")
        worst = 12 * 15000 * 0.045 + config["grader_budget_s"] + 60
        self.assertGreaterEqual(timeout, worst * 1.3)

    def test_the_grader_charges_the_controllers_own_time(self):
        verify = (TASK / "tests/verify.py").read_text()
        self.assertIn("if clock.s > budget:", verify)
        self.assertNotIn("time.monotonic() - t_start > budget", verify)
        self.assertIn("clock.s += image_cost * images_asked(action)", verify)


class Numerics(unittest.TestCase):

    def test_the_pin_is_the_same_in_the_image_the_launcher_and_the_practice_grader(self):
        docker = (TASK / "environment/Dockerfile").read_text()
        launcher = (TASK / "tests/nonet_exec.py").read_text()
        practice = (TASK / "environment/setup/payload/practice_grader/grade.py").read_text()
        text = (TASK / "instruction.md").read_text()
        for key, value in PIN.items():
            self.assertIn('os.environ.setdefault("{}", "{}")'.format(key, value), docker)
            self.assertIn('"{}": "{}"'.format(key, value), launcher)
            self.assertIn('"{}": "{}"'.format(key, value), practice)
            self.assertIn(key, text)
        self.assertIn("os.execvpe(cmd[0], cmd, controller_env())", launcher)

    def test_the_grader_checks_the_fingerprint_before_grading(self):
        verify = (TASK / "tests/verify.py").read_text()
        main = verify[verify.index("def main():"):]
        self.assertLess(main.index("check_numerics()"), main.index("run_episodes("))
        ref = json.loads((TASK / "tests/numerics.json").read_text())
        self.assertEqual(len(ref["fingerprint"]), 64)
        self.assertNotIn("launcher!=pth", ref["math"].values())

    def test_the_reference_has_no_wall_clock_limit_of_its_own(self):
        wire = (TASK / "solution/oracle/wire_world.py").read_text()
        self.assertNotIn("THINK_S", wire)
        self.assertNotIn("timeout=", wire[wire.index("def act("):])


class Rules(unittest.TestCase):

    def test_whole_run_zeros_and_resources_are_stated(self):
        text = (TASK / "instruction.md").read_text()
        for needle in ("cannot stop every process your submission left running", "internet socket",
                       "4 CPUs and 12 GB", "at most 5000 files and 200 MiB", "prints `READY` within 120 s"):
            self.assertIn(needle, text)
        task = (TASK / "task.toml").read_text()
        self.assertEqual(toml_int(task, "environment", "cpus"), 4)
        self.assertEqual(toml_int(task, "environment", "memory_mb"), 12288)


if __name__ == "__main__":
    unittest.main()
