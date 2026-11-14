"""pump_skid_startup_and_operation: the commissioning service (v0.6) and the fixed hidden set.

The agent's sandbox holds a commissioning service (environment/commission_service/): the series' cell on development
sites, for the agent's commissioning shifts. It must run the grader's own cell (a byte copy of tests/skidcell/, less
the client), on sites that are neither graded sites nor candidates of the hidden set's selection, under the graded
episode's budgets and safety rules, and never answer ground truth, a score or a debrief. The given client is the one
the grader launches programs with. The hidden set is the one its latest recorded selection kept.

Standard library only (the host-side suite has no numpy): the service is read as text and a syntax tree, not run.
"""

import ast
import json
import unittest
from pathlib import Path

TASK = Path(__file__).resolve().parents[1] / "tasks/arm/pump_skid_startup_and_operation"
SERVICE = TASK / "environment/commission_service"
SITES = json.loads((SERVICE / "sites.json").read_text())
SELECTION = json.loads((TASK / "tests/hidden/selection.json").read_text())
SEEDS = json.loads((TASK / "tests/hidden/seeds.json").read_text())
GRADER = json.loads((TASK / "tests/grader_config.json").read_text())
PUBLIC = json.loads((TASK / "environment/setup/payload/given/cell/cell_config.json").read_text())


class CommissioningService(unittest.TestCase):
    def test_runs_the_graders_cell(self):
        mine = sorted(p.name for p in (SERVICE / "skidcell").glob("*.py"))
        graders = sorted(p.name for p in (TASK / "tests/skidcell").glob("*.py") if p.name != "client.py")
        self.assertEqual(mine, graders)
        for name in graders:
            self.assertEqual((SERVICE / "skidcell" / name).read_bytes(), (TASK / "tests/skidcell" / name).read_bytes(),
                             "commission_service/skidcell/%s differs from tests/skidcell/%s" % (name, name))

    def test_development_sites_are_not_graded_sites(self):
        dev = set(int(s) for s in SITES["seeds"])
        self.assertEqual(len(dev), len(SITES["seeds"]))
        graded = set(int(s) for s in SEEDS["seeds"])
        candidates = set(int(s) for s in SELECTION["candidates"])
        for a in SELECTION.get("amendments", []):
            candidates |= set(int(s) for s in a.get("candidates", []))
        self.assertFalse(dev & graded, "a development site is a graded site")
        self.assertFalse(dev & candidates, "a development site is a candidate of the hidden set's selection")

    def test_budgets_and_safety_are_the_graded_episodes(self):
        cfg = SITES["config"]
        for k in ("sim_cap_s", "program_time_cap_s", "silence_s", "max_ticks_per_step", "camera_budget_frames"):
            self.assertEqual(cfg[k], GRADER[k], k)
        self.assertEqual(cfg["safety"], GRADER["safety"])
        self.assertEqual(SITES["program_time_cap_s"], GRADER["program_time_cap_s"])
        self.assertEqual(SITES["silence_s"], GRADER["silence_s"])
        c = PUBLIC["commissioning"]
        self.assertEqual((c["development_sites"], c["shifts"], c["open_at_a_time"]), (len(SITES["seeds"]), SITES["shifts"], SITES["open"]))
        self.assertEqual(PUBLIC["safety"], GRADER["safety"])

    def test_answers_no_truth_score_or_debrief(self):
        src = (SERVICE / "commissiond.py").read_text()
        tree = ast.parse(src)
        called = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                called.add(node.func.attr)
        for forbidden in ("truth", "finalize", "summary", "get_state_privileged", "reference_run"):
            self.assertNotIn(forbidden, called, "the commissioning service calls %s()" % forbidden)
        self.assertIn('mode="hidden", privileged=False', src)
        public = None
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "PUBLIC" for t in node.targets):
                public = ast.literal_eval(node.value)
        self.assertEqual(public, ("describe", "get_state", "set_control", "command", "command_rail", "step", "get_camera",
                                  "log_reading", "finish"))

    def test_client_copies(self):
        given = (TASK / "environment/setup/payload/given/lib/skidcell/client.py").read_bytes()
        self.assertEqual(given, (TASK / "tests/skidcell/client.py").read_bytes())
        self.assertEqual(given, (TASK / "tests/setup_payload/given/lib/skidcell/client.py").read_bytes())


class HiddenSet(unittest.TestCase):
    def test_the_set_is_the_latest_selections(self):
        runs = SELECTION["selection_run"]
        latest = runs
        for a in SELECTION.get("amendments", []):
            if "selection_run" in a:
                latest = a["selection_run"]
        self.assertEqual([int(s) for s in SEEDS["seeds"]], [int(s) for s in latest["hidden_set" if "hidden_set" in latest else "kept"][:8]])


if __name__ == "__main__":
    unittest.main()
