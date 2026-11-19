"""keyed_nut_peg_assembly: the fixed hidden set, its pre-registered selection and the anchor.

The hidden kits (tests/ref/episodes.json) are the first 10, in drawn order, of the pre-registered os.urandom
candidates (tests/hidden/selection.json) that the frozen reference solution completed when each ran once through the
grader (tests/review/select_set.py, tests/review/regrade.sh). When a file the reference or the grader runs changes,
the rule runs again on the same candidates as an amendment of the record (its frozen sha256 committed before any run
on the changed files); the set in force is the latest amendment's. The files the reference and the grader run are
unchanged since (their sha256 in that record). The anchor is the reference's recorded full verify of the set, which
repeated every kept selection run. The grader's simulator carries no ground-truth accessor and no score; the credit
rule is the grader's own module. The commissioning service's side is in test_keyed_nut_peg_assembly_commissioning.py.

Standard library only (the host-side suite has no numpy or mujoco): the task's Python files are read as text or syntax
trees, never imported.
"""

import ast
import hashlib
import json
import re
import unittest
from pathlib import Path

TASK = Path(__file__).resolve().parents[1] / "tasks/arm/keyed_nut_peg_assembly"
SELECTION = json.loads((TASK / "tests/hidden/selection.json").read_text())
EPISODES = json.loads((TASK / "tests/ref/episodes.json").read_text())["episodes"]
ANCHOR = json.loads((TASK / "tests/anchor.json").read_text())
#: The record in force: the latest amendment, or the selection itself.
LATEST = SELECTION["amendments"][-1] if SELECTION.get("amendments") else SELECTION
#: An amendment is committed before its runs (pre-registration); until they are recorded, the set's checks wait.
PENDING = "selection_run" not in LATEST or "anchor_verify" not in LATEST
WAIT = "the latest amendment is pre-registered and has not run yet"
#: The candidates in force: those of the latest amendment that drew fresh ones of its own (an amendment that leaves
#: the kits as they are runs on the candidates in force before it), else the selection's.
CANDIDATES = next((a["candidates"] for a in reversed(SELECTION.get("amendments", [])) if a.get("candidates")),
                  SELECTION["candidates"])
#: The takt in force (keyed_env.MAX_STEPS), read from the simulator's source.
TAKT = int(re.search(r"^MAX_STEPS = (\d+)", (TASK / "tests/keyed_env.py").read_text(), re.M).group(1))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class HiddenSet(unittest.TestCase):
    def test_candidates_are_unique_63_bit_seeds(self):
        pools = [SELECTION["candidates"]] + [a["candidates"] for a in SELECTION.get("amendments", []) if "candidates" in a]
        seeds = []
        for cands in pools:
            self.assertGreaterEqual(len(cands), 16)
            self.assertEqual([c["i"] for c in cands], list(range(len(cands))))
            self.assertEqual(len({c["id"] for c in cands}), len(cands))
            seeds += [c["kit_seed"] for c in cands] + [c["layout_seed"] for c in cands]
        self.assertEqual(len(set(seeds)), len(seeds))
        self.assertTrue(all(0 < s < 2 ** 63 for s in seeds))
        ids = [c["id"] for cands in pools for c in cands]
        self.assertEqual(len(set(ids)), len(ids), "two candidate pools share an id")

    @unittest.skipIf(PENDING, WAIT)
    def test_set_is_the_first_ten_kept_in_drawn_order(self):
        runs = LATEST["selection_run"]["runs"]
        self.assertEqual([r["id"] for r in runs], [c["id"] for c in CANDIDATES])
        kept = [r["id"] for r in runs if r["kept"]]
        self.assertEqual(LATEST["hidden_set"], kept[:SELECTION["set_size"]])
        self.assertEqual(len(LATEST["hidden_set"]), 10)
        for r in runs:
            if r["kept"]:
                self.assertEqual((r["kit_complete"], r["nuts_assembled"], r["drop_reasons"]), (1, 3, []))
            else:
                self.assertEqual(r["kit_complete"], 0)
                self.assertTrue(r["drop_reasons"])

    @unittest.skipIf(PENDING, WAIT)
    def test_episodes_are_the_selected_candidates(self):
        cands = {c["id"]: c for c in CANDIDATES}
        self.assertEqual([e["id"] for e in EPISODES], LATEST["hidden_set"])
        for e in EPISODES:
            self.assertEqual(e["seed"], cands[e["id"]]["layout_seed"])
            nuts = e["kit"]["nuts"]
            self.assertEqual(len(nuts), 3)
            self.assertEqual([n["id"] for n in nuts], ["%s-%d" % (e["id"], k) for k in range(3)])
            self.assertEqual(sorted(e["kit"]["order"]), [1, 2, 3])

    def test_frozen_files_unchanged(self):
        for rel, digest in LATEST["frozen"].items():
            self.assertEqual(sha256(TASK / rel), digest, "%s changed since the selection: record an amendment" % rel)

    def test_amendments_name_what_changed(self):
        before = SELECTION["frozen"]
        for a in SELECTION.get("amendments", []):
            changed = sorted(k for k in set(before) | set(a["frozen"]) if before.get(k) != a["frozen"].get(k))
            self.assertEqual(a["changed"], changed)
            self.assertTrue(a["what"].strip() and a["rule"].strip())
            before = a["frozen"]

    @unittest.skipIf(PENDING, WAIT)
    def test_anchor_is_the_recorded_full_verify(self):
        av = LATEST["anchor_verify"]
        self.assertTrue(av["every_episode_repeats_its_selection_run"])
        self.assertEqual([e["id"] for e in av["episodes"]], LATEST["hidden_set"])
        runs = {r["id"]: r for r in LATEST["selection_run"]["runs"]}
        for e in av["episodes"]:
            self.assertEqual((e["kit_complete"], e["steps"]), (runs[e["id"]]["kit_complete"], runs[e["id"]]["steps"]))
        self.assertEqual(ANCHOR["metric"], "kit_success_rate")
        self.assertEqual(ANCHOR["value"], av["measured"])
        self.assertLessEqual(ANCHOR["full_at"], ANCHOR["value"])
        self.assertGreaterEqual(ANCHOR["full_at"], 0.95)

    @unittest.skipIf(PENDING, WAIT)
    def test_reference_keeps_the_takt(self):
        steps = [r["steps"] for r in LATEST["selection_run"]["runs"] if r["kept"]]
        self.assertEqual(TAKT, 850)
        self.assertTrue(all(0 < n < TAKT for n in steps), steps)


class Grader(unittest.TestCase):
    def test_sim_has_no_truth_or_score(self):
        tree = ast.parse((TASK / "tests/keyed_env.py").read_text())
        env = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "NutEnv")
        methods = {n.name for n in env.body if isinstance(n, ast.FunctionDef)}
        for name in ("gt", "episode_score", "nuts_assembled", "progress", "assembled", "stage_now"):
            self.assertNotIn(name, methods)

    def test_grader_scores_with_the_credit_module(self):
        verify = (TASK / "tests/verify.py").read_text()
        self.assertIn("KC.GradedNutEnv(", verify)
        self.assertIn("KC.ground_truth(env, wire)", verify)
        self.assertNotIn("E.NutEnv(", verify)
        config = json.loads((TASK / "tests/grader_config.json").read_text())
        self.assertNotIn("grader_budget_s", config)
        self.assertEqual(config["controller_budget_s"], 600)


if __name__ == "__main__":
    unittest.main()
