"""cross_embodiment_adaptation: what the texts state is what the code does, and a mission leaves nothing behind.

No simulator is needed: the checks read the task's files, and the isolation module is run on a scratch tree.
"""
import ast
import hashlib
import importlib.util
import json
import os
import re
import tomllib
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT/"tasks/quadruped/cross_embodiment_adaptation"


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _constants(path, names):
    out = {}
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Assign):
            several = isinstance(node.targets[0], ast.Tuple)
            targets = node.targets[0].elts if several else [node.targets[0]]
            values = node.value.elts if several and isinstance(node.value, ast.Tuple) else [node.value]
            for target, value in zip(targets, values):
                if isinstance(target, ast.Name) and target.id in names:
                    out[target.id] = ast.literal_eval(value)
    return out


class Texts(unittest.TestCase):
    def setUp(self):
        self.instruction = " ".join((TASK/"instruction.md").read_text().split())
        self.datasheet = " ".join((TASK/"environment/setup/payload/go1_datasheet.md").read_text().split())
        self.ranges = _constants(TASK/"tests/runtime/props.py", {"RANGES"})["RANGES"]

    def test_every_range_is_in_the_datasheet(self):
        for key, (lo, hi) in self.ranges.items():
            digits = 3 if key in ("duct_clearance_m", "guide_rise_m") else 2
            signed = key in ("ballast_x_m", "odom_yaw_drift_deg_s")
            text = "[{}, {}]".format(*("{:{s}.{d}f}".format(v, s="+" if signed else "", d=digits) for v in (lo, hi)))
            self.assertIn(text, self.datasheet, key)

    def test_the_instruction_states_the_load_and_the_fittings(self):
        lo, hi = self.ranges["payload_kg"]
        self.assertIn("between {:.2f} and {:.2f} kg".format(lo, hi), self.instruction)
        lo, hi = self.ranges["ballast_x_m"]
        self.assertIn("between {:.2f} and {:.2f} m ahead of the trunk origin".format(lo, hi), self.instruction)
        lo, hi = self.ranges["duct_clearance_m"]
        self.assertIn("between {:.3f} and {:.3f} m above the floor".format(lo, hi), self.instruction)
        lo, hi = self.ranges["guide_rise_m"]
        self.assertIn("crest between {:.3f} and {:.3f} m".format(lo, hi), self.instruction)

    def test_the_instruction_states_the_sensing_and_the_changes(self):
        lo, hi = self.ranges["odom_yaw_drift_deg_s"]
        self.assertIn("between {:.2f} and +{:.2f} degrees per second".format(lo, hi), self.instruction)
        lo, hi = self.ranges["odom_scale"]
        self.assertIn("a factor between {:.2f} and {:.2f} (per site), laid along".format(lo, hi), self.instruction)
        lo, hi = self.ranges["uav_scale"]
        self.assertIn("a factor between {:.2f} and {:.2f}".format(lo, hi), self.instruction)
        noise = _constants(TASK/"tests/runtime/props.py", {"ENCODER_NOISE_RAD", "ENCODER_BIAS_RAD", "GYRO_NOISE_RAD_S",
                                                            "GYRO_BIAS_RAD_S", "GRAVITY_NOISE"})
        self.assertIn("`joint_rad` (noise {:.3f} rad, a constant bias within {:.3f} rad per joint)".format(
            noise["ENCODER_NOISE_RAD"], noise["ENCODER_BIAS_RAD"]), self.instruction)
        self.assertIn("`gyro_body_rad_s` (noise {:.2f} rad/s, a constant bias within {:.3f} rad/s per axis)".format(
            noise["GYRO_NOISE_RAD_S"], noise["GYRO_BIAS_RAD_S"]), self.instruction)
        self.assertIn("(noise {:.2f} per axis)".format(noise["GRAVITY_NOISE"]), self.instruction)
        scene = _constants(TASK/"tests/runtime/articulated.py", {"PALLET_X", "PALLET_Y", "PALLET_Z", "BEACON"})
        self.assertIn("(magenta plastic, 1.20 m by 0.80 m by {:.3f} m, stood on its short edge so that it is {:.2f} m wide "
                      "and {:.2f} m tall)".format(scene["PALLET_X"], scene["PALLET_Y"], scene["PALLET_Z"]), self.instruction)
        self.assertEqual(scene["PALLET_X"], .144)
        self.assertIn("Touching the pallet or a bracket ends the mission.", self.instruction)
        self.assertIn("always stands {:.2f} m east and {:.2f} m north of the dock's centre".format(*scene["BEACON"]),
                      self.instruction)
        for phrase in ("the pallet stands in the middle rack's other opening", "the dock was moved",
                       "one bollard was taken away, or one was added"):
            self.assertIn(phrase, self.instruction)

    def test_the_assets_are_pinned(self):
        listing = (TASK/"environment/assets.sha256").read_text()
        self.assertRegex(listing, r"# revision: [0-9a-f]{40}\n")
        self.assertIn("# prefix: tasks/quadruped/cross_embodiment_adaptation/payload", listing)
        dockerfile = (TASK/"environment/Dockerfile").read_text()
        self.assertIn("fetch_assets.py /opt/task-assets/assets.sha256", dockerfile)
        for url in ("polyhaven.org", "githubusercontent", "github.com/google-deepmind"):
            self.assertNotIn(url, dockerfile)

    def test_budgets(self):
        ticks = _constants(TASK/"tests/runtime/articulated.py", {"MAX_TICKS", "MAX_SCOUT", "MAX_LOOKS"})
        self.assertIn("{:,} physics ticks".format(ticks["MAX_TICKS"]), self.instruction)
        self.assertIn("There are {} moves".format(ticks["MAX_SCOUT"]), self.instruction)
        self.assertIn("allows {} exposures".format(ticks["MAX_LOOKS"]), self.instruction)
        pad = _constants(TASK/"tests/runtime/service.py", {"PAD_EPISODES", "PAD_TICKS", "REQUESTS"})
        self.assertIn("**{} episodes and {:,} physics ticks**".format(pad["PAD_EPISODES"], pad["PAD_TICKS"]),
                      self.instruction)
        self.assertIn("{:,} requests".format(pad["REQUESTS"]), self.instruction)

    def test_session_time(self):
        toml = tomllib.loads((TASK/"task.toml").read_text())
        self.assertEqual(toml["agent"]["timeout_sec"], 7200)
        sites = len(json.loads((TASK/"tests/hidden/sites.json").read_text())["seeds"])
        config = json.loads((TASK/"tests/grader_config.json").read_text())
        self.assertGreaterEqual(toml["verifier"]["timeout_sec"], sites*config["mission_wall_seconds"]+1800)
        self.assertIn("out of sixteen", self.instruction)
        self.assertEqual(sites, 16)
        self.assertIn("**2 hours**", self.instruction)
        script = (TASK/"environment/setup/payload/session_time.sh").read_text()
        self.assertIn("120 - used", script)
        self.assertIn("/run/session/start", script)
        setup = (TASK/"environment/setup/setup_runtime.py").read_text()
        self.assertIn('Path("/run/session")', setup)
        self.assertIn("stamp.chmod(0o444)", setup)

    def test_no_rule_the_grader_cannot_check(self):
        for phrase in ("offline target rollouts", "pretrained walking policy", "weight updates"):
            self.assertNotIn(phrase, self.instruction)
        self.assertIn("any model or simulation of your own", self.instruction)

    def test_the_whole_submission_zero_is_stated(self):
        self.assertIn("the whole submission scores zero", self.instruction)


class Copies(unittest.TestCase):
    def test_runtime_and_payload_copies_are_identical(self):
        for a, b in (("environment/setup/runtime", "tests/runtime"), ("environment/setup/payload", "tests/setup_payload")):
            names = sorted(p.name for p in (TASK/a).iterdir() if p.is_file())
            self.assertEqual(names, sorted(p.name for p in (TASK/b).iterdir() if p.is_file()))
            for name in names:
                self.assertEqual((TASK/a/name).read_bytes(), (TASK/b/name).read_bytes(), name)

    def test_the_oracle_token(self):
        token = (TASK/"solution/oracle/token.txt").read_text().strip()
        recorded = (TASK/"tests/hidden/oracle_token.sha256").read_text().split()[0]
        self.assertEqual(hashlib.sha256(token.encode()).hexdigest(), recorded)
        for folder in ("environment", "tests"):
            for path in (TASK/folder).rglob("*"):
                if path.is_file() and path.stat().st_size < 1 << 20:
                    self.assertNotIn(token.encode(), path.read_bytes(), path)

    def test_truth_and_pad_are_gated_by_mode(self):
        service = (TASK/"tests/runtime/service.py").read_text()
        self.assertRegex(service, r'if self\.mode == "graded":\s+allowed\["advance"\] = set\(\)\s+'
                                  r'allowed\["look"\] = \{"detent"\}\s+allowed\["truth"\] = \{"token"\}')
        self.assertIn('if self.mode != "commissioning":\n                    raise ValueError("pad is a commissioning call',
                      service)
        self.assertIn("hashlib.sha256(token.encode()).hexdigest() != self.token_sha256", service)

    def test_the_graded_set_is_unchanged(self):
        sites = json.loads((TASK/"tests/hidden/sites.json").read_text())["seeds"]
        candidates = json.loads((TASK/"tests/hidden/candidates.json").read_text())["seeds"]
        self.assertEqual([candidates.index(s) for s in sites], [0, 1, 2, 3, 4, 10, 11, 16, 5, 6, 7, 8, 9, 12, 13, 14])
        anchor = json.loads((TASK/"tests/anchor.json").read_text())
        self.assertEqual((anchor["value"], anchor["full_at"]), (1.0, 0.98))


class Isolation(unittest.TestCase):
    """tests/isolation.py on a scratch tree owned by the test's own account."""
    def setUp(self):
        import tempfile
        self.iso = _load(TASK/"tests/isolation.py", "cea_isolation")
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.top = os.path.realpath(self.tmp.name)
        self.iso.mounts = lambda: [self.top]

    def test_purge_removes_what_the_account_owns_and_keeps_the_named_paths(self):
        top = Path(self.top)
        (top/"home/sub/deep").mkdir(parents=True)
        (top/"home/sub/deep/note").write_text("gait trial 1")
        (top/"home/.profile").write_text("x")
        os.symlink("/etc/passwd", top/"home/link")
        (top/"program").mkdir()
        (top/"program/run.sh").write_text("#!/bin/sh\n")
        removed = self.iso.purge_files(os.getuid(), keep=(str(top/"program"),), keep_dirs=(str(top/"home"),))
        self.assertEqual(removed, 5)
        self.assertEqual(sorted(p.name for p in top.iterdir()), ["home", "program"])
        self.assertEqual(list((top/"home").iterdir()), [])
        self.assertTrue((top/"program/run.sh").is_file())
        self.assertTrue(os.path.exists("/etc/passwd"))

    def test_another_accounts_files_stay(self):
        (Path(self.top)/"a").write_text("x")
        self.assertEqual(self.iso.purge_files(os.getuid()+1), 0)
        self.assertTrue((Path(self.top)/"a").exists())

    def test_restore_puts_changed_files_back(self):
        path = Path(self.top)/"shared.log"
        path.write_text("before\n")
        info = path.stat()
        snapshot = {str(path): (info.st_size, info.st_mtime_ns, b"before\n"),
                    str(Path(self.top)/"big"): (10, 1, None)}
        self.assertEqual(self.iso.restore({str(path): snapshot[str(path)]}), (0, []))
        path.write_text("before\nleft by a mission\n")
        restored, failed = self.iso.restore(snapshot)
        self.assertEqual((restored, failed), (1, [str(Path(self.top)/"big")]))
        self.assertEqual(path.read_text(), "before\n")

    def test_the_grader_isolates_every_mission(self):
        grader = (TASK/"tests/grader.py").read_text()
        mission = grader[grader.index("def run_mission("):grader.index("def account_home(")]
        self.assertLess(mission.index("stop_agent_processes(uid)"), mission.index("purge_files(uid"))
        self.assertIn("purge_ipc(uid)", mission)
        self.assertIn("restore(snapshot)", mission)
        self.assertRegex(grader, re.compile(r"check_numerics\(\)"))


if __name__ == "__main__":
    unittest.main()
