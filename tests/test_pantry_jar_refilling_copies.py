"""pantry_jar_refilling: the agent practises with exactly the simulator the grader runs, and what the task says about
its session clock, its cameras and its records holds in the files.

The simulator (jar_sim.py), its cameras' renderer and look (jar_render.py, jar_look.json), the environment
(jar_env.py) and the episode logic (episode.py) live in three byte-identical copies: tests/ (the grader),
environment/setup/payload/ (the agent's home) and solution/oracle/ (the reference). The render bundle is a task asset
(docs/task-assets.md): listed in environment/assets.sha256, not in git, read where the image puts it.

Standard library only (the host-side suite has no numpy): nothing of the task is imported.
"""

import hashlib
import json
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / "tasks/arm/pantry_jar_refilling"
COPIES = ("tests", "environment/setup/payload", "solution/oracle")
SHARED = ("jar_sim.py", "jar_render.py", "jar_look.json", "jar_env.py", "episode.py")
BUNDLE = ("render/ibl.ktx", "render/room.json", "render/room.npz")
RENDER_DIR = "/opt/task-assets/payload/render"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Copies(unittest.TestCase):

    def test_the_three_copies_are_byte_identical(self):
        for name in SHARED:
            grader = TASK / "tests" / name
            self.assertTrue(grader.is_file(), name)
            for copy in COPIES[1:]:
                self.assertEqual(sha256(TASK / copy / name), sha256(grader), "{} differs in {}".format(name, copy))

    def test_the_frozen_list_holds_the_renderer(self):
        src = (TASK / "tests/review/select_set.py").read_text()
        frozen = src[src.index("ORACLE_MODULES = ("):src.index("PER_UNIT")]
        for name in ("jar_render.py", "jar_look.json", "tests/jar_render.py", "tests/jar_look.json", "tests/jar_grade.py"):
            self.assertIn('"{}"'.format(name), frozen)

    def test_the_look_parses_and_draws_no_shadow(self):
        look = json.loads((TASK / "tests/jar_look.json").read_text())
        for key in ("exposure_scale", "ibl", "lamps", "headlight", "room", "model_materials", "filament_numeric",
                    "filament_text", "render"):
            self.assertIn(key, look)
        self.assertFalse(look["render"]["shadows"])
        self.assertFalse(any(lamp.get("shadow") for lamp in look["lamps"]))
        for part in ("jar", "label", "lid", "tray", "piece_box", "piece_capsule"):
            self.assertNotIn("gain", look["model_materials"][part], "the task's own parts keep the model's colours")
            self.assertNotIn("srgb", look["model_materials"][part], "the task's own parts keep the model's colours")


class Assets(unittest.TestCase):

    def test_the_bundle_is_listed_and_fetched(self):
        listed = [line.split("  ", 1)[1] for line in (TASK / "environment/assets.sha256").read_text().splitlines()
                  if line and not line.startswith("#")]
        self.assertEqual(sorted(listed), sorted(BUNDLE))
        docker = (TASK / "environment/Dockerfile").read_text()
        self.assertIn("fetch_assets.py /opt/task-assets/assets.sha256 /opt/task-assets/payload", docker)
        self.assertIn("ENV JAR_RENDER_DIR=" + RENDER_DIR, docker)
        self.assertIn("mesa-vulkan-drivers libvulkan1", docker)
        self.assertIn('RENDER_DIR = os.environ.get("JAR_RENDER_DIR", "{}")'.format(RENDER_DIR),
                      (TASK / "tests/jar_render.py").read_text())
        self.assertEqual(docker.strip().splitlines()[-1], "WORKDIR /")

    def test_the_bundle_is_not_staged_into_the_agent_home(self):
        setup = (TASK / "environment/setup/setup.sh").read_text()
        self.assertNotIn("/opt/task-assets", setup)
        self.assertFalse((TASK / "environment/setup/payload/render").exists())


class Records(unittest.TestCase):

    def test_the_graders_episode_line_names_no_seed(self):
        """The grader's log is published with a run's records: an episode's line names its index and its unit."""
        src = (TASK / "tests/jar_grade.py").read_text()
        lines = [line for line in src.splitlines() if 'log("episode ' in line]
        self.assertEqual(len(lines), 1)
        self.assertIn('log("episode {} ({}): credit={:.4f} steps={} {}".format(n, uid, rec["credit"], rec["steps"], rec["reason"]))',
                      lines[0])

    def test_session_time_is_the_engines_limit(self):
        """The instruction's session limit, the script's and the engine's agent timeout are one number; the stamp the
        script reads is written by setup.sh into a folder that is root's (the agent's account reads it and can
        neither change nor replace it)."""
        toml = (TASK / "task.toml").read_text()
        agent = toml[toml.index("[agent]"):]
        timeout = int(re.search(r"^timeout_sec\s*=\s*(\d+)", agent, re.M).group(1))
        script = (TASK / "environment/setup/payload/session_time.sh").read_text()
        limit = int(re.search(r"^LIMIT_MIN=(\d+)$", script, re.M).group(1))
        self.assertEqual(limit * 60, timeout)
        self.assertEqual(re.search(r"^STAMP=(\S+)$", script, re.M).group(1), "/run/session/start")
        setup = (TASK / "environment/setup/setup.sh").read_text()
        for line in ("install -d -o root -g root -m 755 /run/session", "date -u +%s > /run/session/start",
                     "chown root:root /run/session/start && chmod 0444 /run/session/start"):
            self.assertIn(line, setup)
        self.assertNotIn("HOME_DIR/.session", setup, "a stamp in the agent's home could be replaced by the agent")
        instruction = " ".join((TASK / "instruction.md").read_text().split())
        self.assertIn("bash /home/user/session_time.sh", instruction)
        self.assertIn("2 hours", instruction)


if __name__ == "__main__":
    unittest.main()
