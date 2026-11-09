"""object_finding_and_organizing_on_messy_desk: the privileged reference channel and the fixed hidden episode set.

The grader hands the simulator's ground truth only to a submission whose .oracle_token (a regular file, never a
link) hashes to tests/oracle_token.sha256; nothing agent-visible names the channel; the hidden episodes are a fixed
set of distinct 64-bit seeds; and an episode's probe stream depends only on the set's key and the episode's seed.
"""
import hashlib
import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

TASK = Path(__file__).resolve().parents[1] / "tasks/arm/object_finding_and_organizing_on_messy_desk"


def load_verify():
    sys.path.insert(0, str(TASK / "tests"))
    try:
        spec = importlib.util.spec_from_file_location("messy_desk_verify", TASK / "tests" / "verify.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(TASK / "tests"))


verify = load_verify()


class TokenTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.sub = os.path.join(self.tmp.name, "submission")
        os.makedirs(self.sub)
        self.token = b"0123456789abcdef" * 4
        self.digest = os.path.join(self.tmp.name, "oracle_token.sha256")
        with open(self.digest, "w") as f:
            f.write(hashlib.sha256(self.token).hexdigest() + "\n")

    def tearDown(self):
        self.tmp.cleanup()

    def state(self):
        return verify.token_state(self.sub, digest_path=self.digest)

    def test_no_token_is_neither(self):
        self.assertEqual(self.state(), (False, False))

    def test_matching_token_is_privileged(self):
        with open(os.path.join(self.sub, ".oracle_token"), "wb") as f:
            f.write(self.token + b"\n")
        self.assertEqual(self.state(), (True, False))

    def test_wrong_token_is_a_mismatch(self):
        with open(os.path.join(self.sub, ".oracle_token"), "wb") as f:
            f.write(b"f" * 64)
        self.assertEqual(self.state(), (False, True))

    def test_a_link_is_never_followed(self):
        real = os.path.join(self.tmp.name, "real_token")
        with open(real, "wb") as f:
            f.write(self.token)
        os.symlink(real, os.path.join(self.sub, ".oracle_token"))
        self.assertEqual(self.state(), (False, True))

    def test_a_directory_or_a_huge_file_is_a_mismatch(self):
        os.makedirs(os.path.join(self.sub, ".oracle_token"))
        self.assertEqual(self.state(), (False, True))
        os.rmdir(os.path.join(self.sub, ".oracle_token"))
        with open(os.path.join(self.sub, ".oracle_token"), "wb") as f:
            f.write(self.token + b" " * (verify.TOKEN_MAX_BYTES + 10))
        self.assertEqual(self.state(), (False, True))

    def test_the_reference_token_matches_the_digest(self):
        token = (TASK / "solution/privileged/oracle_token").read_bytes().strip()
        self.assertEqual(len(token), 64)                      # 256 bits, hex
        want = (TASK / "tests/oracle_token.sha256").read_text().split()[0]
        self.assertEqual(hashlib.sha256(token).hexdigest(), want)

    def test_nothing_agent_visible_names_the_channel(self):
        token = (TASK / "solution/privileged/oracle_token").read_text().strip()
        digest = (TASK / "tests/oracle_token.sha256").read_text().split()[0]
        seeds = [str(s) for s in json.loads((TASK / "tests/episodes.json").read_text())["seeds"]]
        visible = [TASK / "instruction.md"] + [p for p in (TASK / "environment").rglob("*") if p.is_file()]
        for p in visible:
            text = p.read_text(errors="replace")
            for needle in [token, digest, ".oracle_token", "oracle_token", "privileged_mode", '"privileged"'] + seeds:
                self.assertNotIn(needle, text, "{} mentions {}".format(p.relative_to(TASK), needle[:16]))


class EpisodeSetTests(unittest.TestCase):
    def test_fixed_set_is_distinct_64_bit(self):
        seeds, key = verify.load_episodes(str(TASK / "tests/episodes.json"))
        self.assertGreaterEqual(len(seeds), 1)
        self.assertEqual(len(set(seeds)), len(seeds))
        self.assertTrue(all(2 ** 32 <= s < 2 ** 64 for s in seeds))     # none a seed anyone would type into the lab
        self.assertTrue(0 < key < 2 ** 64)

    def test_probe_stream_depends_on_key_and_seed_only(self):
        seeds, key = verify.load_episodes(str(TASK / "tests/episodes.json"))
        a = verify.seed_rng(entropy=(key << 64) | seeds[0])
        b = verify.seed_rng(entropy=(key << 64) | seeds[0])
        self.assertEqual([a.random() for _ in range(50)], [b.random() for _ in range(50)])
        if len(seeds) > 1:
            c = verify.seed_rng(entropy=(key << 64) | seeds[1])
            d = verify.seed_rng(entropy=(key << 64) | seeds[0])
            self.assertNotEqual([c.random() for _ in range(5)], [d.random() for _ in range(5)])


class PrivilegedBlockTests(unittest.TestCase):
    def test_block_carries_the_truth_of_the_open_request(self):
        state = {"t": 7, "ik_ok": True,
                 "objects": [{"id": 3, "model": "025_mug", "cls": "C", "present": True, "removed": False,
                              "pos": [0.5, 0.1, 0.04], "quat": [1, 0, 0, 0], "scale": 1.0, "state": "on_table",
                              "in_container": None, "on_table": True, "held": False, "contact": False, "resting": True,
                              "value": 0, "baseline": 0},
                             {"id": 4, "model": "024_bowl", "cls": "C", "present": False, "removed": True}],
                 "request": {"id": 2, "model": "025_mug", "object": 3, "text": "the mug", "opened_t": 5,
                             "deadline_t": 1205, "status": "open"}}
        obs = {"ee_pose": [0.46, 0.0, 0.39, 0.0, 1.0, 0.0, 0.0], "joint_pos": [0.0] * 7, "gripper_width": 0.08}
        cont = {"red": {"xy": [0.25, 0.0], "yaw": 0.01}}
        blk = verify.privileged_block(state, cont, obs)
        self.assertEqual([o["id"] for o in blk["objects"]], [3])           # present objects only
        self.assertNotIn("value", blk["objects"][0])                        # the score's terms stay out
        self.assertEqual(blk["request"]["object"], 3)
        self.assertEqual(blk["containers"], cont)
        self.assertEqual(blk["ee_pose"], obs["ee_pose"])
        state["request"]["status"] = "delivered"
        self.assertIsNone(verify.privileged_block(state, cont, obs)["request"])

    def test_a_probe_keeps_the_true_block(self):
        obs = {"ee_pose": [0.46, 0.0, 0.39, 0.0, 1.0, 0.0, 0.0], "joint_pos": [0.0] * 7, "gripper_width": 0.08,
               "privileged": {"ee_pose": [0.46, 0.0, 0.39, 0.0, 1.0, 0.0, 0.0]}}
        p = verify._perturbed(obs, verify.seed_rng(entropy=1))
        self.assertNotEqual(p["ee_pose"][:3], obs["ee_pose"][:3])
        self.assertIs(p["privileged"], obs["privileged"])


if __name__ == "__main__":
    unittest.main()
