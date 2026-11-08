"""shoe_lacing_and_boxing: the privileged reference's token, the fixed hidden set, its selection and the anchor.

The grader gives the privileged calls (the reference controller inside the cell) only to a submission whose
.oracle_token hashes to tests/hidden/oracle_token.sha256, and that digest lives only there. The hidden episodes are a
fixed set (tests/hidden/seeds.json): the first 6, in drawn order, of a pre-registered batch of os.urandom seeds on
which the frozen reference scores 1.0 and whose scene starts physical (no contact deeper than 1 mm, the lid tray flat
inside the infeed tray's rim, nothing moving off in 1 s idle; tests/hidden/selection.json, the rule in
tests/review/select_set.py), and verify.py runs that list as it stands. The cell's three copies (the grader's, the
agent's givens and the grader's copy of the givens) are byte-identical and pinned by tests/given_hashes.json, and the
lid tray's delivery spot keeps even the biggest lid inside the rim. The files the set was selected with (the reference, the cell, the grader, the
image's inputs) are unchanged since (their sha256 in the selection record), or changed only by a recorded amendment
that touched the cameras' renderer alone and after which the reference's full verify on the amended image repeated
every selection run (score, simulated time, ticks and per-pair detail). The anchor is the lowest work_score of the
recorded full verifies of the privileged submission on the set, with a full_at under which those verifies still read
reward 1.0. Nothing the agent sees names the token, the set or its selection; the cell library's own mentions of its
privileged calls (the PR author's public cell code, byte-identical to the grader's) are pinned as they are.

Standard library only (the host-side suite has no numpy): verify.py is read as a syntax tree, not imported.
"""

import ast
import hashlib
import importlib.util
import json
import math
import re
import sys
import unittest
from pathlib import Path

TASK = Path(__file__).resolve().parents[1] / "tasks/arm/shoe_lacing_and_boxing"
HIDDEN = TASK / "tests/hidden"


def _module(name, path):
    saved = sys.dont_write_bytecode
    sys.dont_write_bytecode = True  # (no __pycache__ in the task folder)
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    finally:
        sys.dont_write_bytecode = saved


seeds_mod = _module("shoe_lacing_cellkit_seeds", TASK / "tests/cellkit/seeds.py")
select_set = _module("shoe_lacing_select_set", TASK / "tests/review/select_set.py")
SELECTION = json.loads((HIDDEN / "selection.json").read_text())
SEEDS = json.loads((HIDDEN / "seeds.json").read_text())
ANCHOR = json.loads((TASK / "tests/anchor.json").read_text())
CONFIG = json.loads((TASK / "tests/grader_config.json").read_text())
AMENDMENTS = SELECTION.get("amendments", [])
# what an amendment may change: the cameras' renderer (the colour from a render twin, the depth from a depth twin; the
# physics model, the reference and the scoring untouched), the given-file pins that follow it, and the image's inputs
RENDER_ONLY = ("tests/shoecell/cameras.py", "tests/shoecell/physics.py", "tests/shoecell/look.py",
               "tests/given_hashes.json", "environment/Dockerfile")


def current_frozen():
    return AMENDMENTS[-1]["frozen"] if AMENDMENTS else SELECTION["frozen"]


def recorded_images():
    return [SELECTION["image"]["id"]] + [am["image"]["id"] for am in AMENDMENTS]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def task_files():
    return sorted(p for p in TASK.rglob("*") if p.is_file() and "__pycache__" not in p.parts)


def agent_visible():
    return [TASK / "instruction.md"] + sorted(p for p in (TASK / "environment").rglob("*") if p.is_file())


class TokenTests(unittest.TestCase):
    def test_the_reference_token_matches_the_digest(self):
        token = (TASK / "solution/reference/oracle_token").read_bytes().strip()
        self.assertRegex(token.decode(), r"^[0-9a-f]{64}$")  # 256 bits, hex
        digest = (HIDDEN / "oracle_token.sha256").read_text().split()[0]
        self.assertEqual(hashlib.sha256(token).hexdigest(), digest)

    def test_the_digest_lives_only_under_tests_hidden(self):
        digest = (HIDDEN / "oracle_token.sha256").read_text().split()[0]
        holders = [p.relative_to(TASK).as_posix() for p in task_files() if digest in p.read_text(errors="replace")]
        self.assertEqual(holders, ["tests/hidden/oracle_token.sha256"])

    def test_the_token_lives_only_in_the_solution(self):
        token = (TASK / "solution/reference/oracle_token").read_text().strip()
        holders = [p.relative_to(TASK).as_posix() for p in task_files() if token in p.read_text(errors="replace")]
        self.assertEqual(holders, ["solution/reference/oracle_token"])


class HiddenSetTests(unittest.TestCase):
    def kept(self):
        return [c for c in sorted(SELECTION["candidates"], key=lambda c: c["draw"]) if c["status"] == "kept"]

    def test_seeds_json_is_the_episode_count_of_distinct_63_bit_seeds(self):
        pinned = seeds_mod.load_pinned(str(HIDDEN / "seeds.json"))
        seeds = pinned["seeds"]
        self.assertEqual(len(seeds), CONFIG["episodes"])
        self.assertEqual(CONFIG["episodes"], 6)  # instruction.md: "6 episodes"
        self.assertIn("(6 episodes)", (TASK / "instruction.md").read_text())
        self.assertEqual(len(set(seeds)), len(seeds))
        self.assertTrue(all(isinstance(s, int) and 2 ** 32 <= s < 2 ** 63 for s in seeds))
        self.assertEqual(set(SEEDS) - {"seeds", "source"}, set())

    def test_the_set_is_the_selection_s_kept_seeds_in_drawn_order(self):
        kept = self.kept()
        self.assertEqual([c["seed"] for c in kept], SEEDS["seeds"])
        self.assertEqual(SELECTION["set"]["kept"], [c["draw"] for c in kept])
        self.assertEqual(SELECTION["set"]["seeds_sha256"], sha256(HIDDEN / "seeds.json"))
        k, d, a = select_set.classify(SELECTION, CONFIG["episodes"])  # the rule, applied again to the record
        self.assertEqual([c["draw"] for c in k], SELECTION["set"]["kept"])
        self.assertEqual([c["draw"] for c in d], SELECTION["set"]["dropped"])
        self.assertEqual([c["draw"] for c in a], SELECTION["set"]["after_cutoff"])
        for c in kept:
            r = c["result"]
            self.assertTrue(select_set.passes(r))
            self.assertEqual(r["work_score"], 1.0)
            self.assertTrue(r["safety_ok"])
            st = r["start_state"]
            self.assertTrue(st["ok"], st.get("why_not"))
            self.assertEqual(st["deep"], {})
            self.assertLess(st["lid_tilt_deg"], select_set.START_TILT_DEG)
            self.assertEqual(st["lid_rim_contact_N"], 0.0)
            self.assertGreater(st["lid_inside_rim_mm"], 0.0)
            self.assertEqual(st["lace_in_lid"], {})
            self.assertLessEqual(max(st["moved_mm"].values()), select_set.START_MOVE_MM)
        for c in d:
            self.assertFalse(select_set.passes(c["result"]))
            self.assertTrue(c.get("reason"))

    def test_every_candidate_s_start_state_was_checked(self):
        """Rule item 4: a candidate passes only with work_score 1.0 and a physical start (start_state)."""
        self.assertEqual(SELECTION["format"], select_set.SEL_FORMAT)
        for c in SELECTION["candidates"]:
            r = c["result"]
            st = r["start_state"]
            for key in ("deep", "lid_tilt_deg", "lid_rim_contact_N", "lid_inside_rim_mm", "lace_in_lid", "moved_mm", "ok",
                        "why_not"):
                self.assertIn(key, st)
            self.assertEqual(st["idle_s"], select_set.START_IDLE_S)
            self.assertEqual(set(st["moved_mm"]), {"shoe0", "shoe1", "lid", "blank"})
            self.assertEqual(st["ok"], not st["why_not"])
            self.assertEqual(r["passes"], select_set.passes(r))
            if not st["ok"]:
                self.assertFalse(select_set.passes(r))

    def test_every_candidate_was_drawn_before_any_ran_and_ran_once(self):
        cands = sorted(SELECTION["candidates"], key=lambda c: c["draw"])
        self.assertEqual([c["draw"] for c in cands], list(range(len(cands))))
        self.assertEqual(len({c["seed"] for c in cands}), len(cands))
        self.assertTrue(all(2 ** 32 <= c["seed"] < 2 ** 63 for c in cands))
        drawn = {b["batch"]: b["drawn_utc"] for b in SELECTION["batches"]}
        for c in cands:
            self.assertIn(c["status"], ("kept", "dropped", "after_cutoff"))
            r = c["result"]
            self.assertEqual(r["seed"], c["seed"])
            self.assertLess(drawn[c["batch"]], r["started_utc"])  # (ISO 8601 UTC strings order as times)
            self.assertTrue(r["frozen_ok"])
            self.assertEqual(r["image_id"], SELECTION["image"]["id"])
        self.assertEqual(sum(b["count"] for b in SELECTION["batches"]), len(cands))

    def test_the_rule_is_the_one_recorded_at_the_draw(self):
        """select_set.py (the rule, its pass criterion included) and run_candidate.sh as they were when the candidates were
        drawn: their sha256 in every batch of the selection record."""
        for b in SELECTION["batches"]:
            self.assertEqual(sorted(b["tool_sha256"]), sorted(select_set.TOOL))
            for rel, digest in b["tool_sha256"].items():
                self.assertEqual(sha256(TASK / rel), digest, "{} changed since batch {} was drawn".format(rel, b["batch"]))

    def test_the_pinned_list_is_run_as_it_stands(self):
        """verify.py reads the list itself (hidden_set); a pinned stream's derive() would hash it into other seeds."""
        tree = ast.parse((TASK / "tests/verify.py").read_text())
        funcs = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
        calls = {n.func.id for n in ast.walk(funcs["_main_locked"]) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
        self.assertIn("hidden_set", calls)
        attrs = {n.func.attr for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
        self.assertNotIn("derive", attrs)
        self.assertIn('"seeds.json"', ast.get_source_segment((TASK / "tests/verify.py").read_text(), funcs["hidden_set"]))
        path = str(HIDDEN / "seeds.json")
        self.assertEqual(seeds_mod.seed_rng(pinned=path).seeds(6, bits=63), SEEDS["seeds"])
        self.assertNotEqual(seeds_mod.seed_rng(pinned=path).derive("episodes").seeds(6, bits=63), SEEDS["seeds"])

    def test_the_hollowing_check_is_recorded_as_the_rule_computes_it(self):
        k, d, a = select_set.classify(SELECTION, CONFIG["episodes"])
        self.assertEqual(SELECTION["set"]["hollowing"], select_set.hollowing(k, d, a))


class FrozenTests(unittest.TestCase):
    def test_the_files_the_set_was_selected_with_are_unchanged(self):
        """... or as the last amendment recorded them (test_every_amendment_...)."""
        frozen = current_frozen()
        for rel, digest in sorted(frozen.items()):
            self.assertEqual(sha256(TASK / rel), digest, "{} changed since the selection (or its last amendment): the "
                             "set and the anchor were measured with the frozen file".format(rel))
        self.assertEqual(sorted(frozen), select_set.frozen_paths(str(TASK)))

    def test_every_amendment_changed_only_the_renderer_and_the_reference_repeated_every_run(self):
        prev = SELECTION["frozen"]
        for am in AMENDMENTS:
            changed = sorted(p for p in set(prev) | set(am["frozen"]) if prev.get(p) != am["frozen"].get(p))
            self.assertEqual(changed, am["changed"])
            self.assertTrue(set(changed) <= set(RENDER_ONLY), set(changed) - set(RENDER_ONLY))
            self.assertTrue(am.get("what"))
            runs = [v for v in ANCHOR["verifies"] if v["run"] == am["check"]["run"]]
            self.assertEqual(len(runs), 1, "the amendment's check names one recorded verify")
            v = runs[0]
            self.assertEqual(v["image_id"], am["image"]["id"])
            self.assertEqual([e["seed"] for e in v["episodes"]], SEEDS["seeds"])
            for e in v["episodes"]:
                self.assertTrue(e["equals_selection_run"] and e["same_trajectory"], e["seed"])
            prev = am["frozen"]

    def test_every_reference_and_cell_file_is_frozen(self):
        frozen = set(current_frozen())
        for folder in ("tests/reference", "solution/reference", "tests/shoecell", "tests/cellkit"):
            for p in (TASK / folder).iterdir():
                if p.is_file():
                    self.assertIn(p.relative_to(TASK).as_posix(), frozen)
        for rel in ("solution/solve.sh", "tests/verify.py", "tests/test.sh", "tests/grader_config.json",
                    "tests/given_hashes.json", "tests/hidden/oracle_token.sha256", "environment/Dockerfile"):
            self.assertIn(rel, frozen)


class CellTests(unittest.TestCase):
    """The cell the agent practises with is the one the grader runs, and its lid tray is delivered inside the rim."""

    def test_the_three_copies_of_the_cell_are_byte_identical(self):
        grader = TASK / "tests/shoecell"
        for copy in (TASK / "environment/setup/payload/given/lib/shoecell", TASK / "tests/setup_payload/given/lib/shoecell"):
            names = sorted(p.name for p in grader.glob("*.py"))
            self.assertEqual(sorted(p.name for p in copy.glob("*.py")), names)
            for n in names:
                self.assertEqual(sha256(grader / n), sha256(copy / n), "{} differs in {}".format(n, copy.relative_to(TASK)))

    def test_given_hashes_pin_the_grader_s_copy_of_the_givens(self):
        given = json.loads((TASK / "tests/given_hashes.json").read_text())
        payload, agent = TASK / "tests/setup_payload", TASK / "environment/setup/payload"
        files = sorted(p.relative_to(payload).as_posix() for p in payload.rglob("*") if p.is_file())
        self.assertEqual(files, sorted(p.relative_to(agent).as_posix() for p in agent.rglob("*") if p.is_file()))
        self.assertEqual(sorted(k for k in given if not k.startswith("given/panda/")), files)
        for rel in files:
            self.assertEqual(given[rel], sha256(payload / rel), rel)
            self.assertEqual(sha256(agent / rel), sha256(payload / rel), rel)

    def test_the_lid_spot_keeps_the_biggest_lid_inside_the_rim(self):
        """episode.py's lid draw against the infeed tray's near rim (inner face 4 mm in from its edge) for the biggest
        batch: shoe 0.21 x 0.08 m, box L + 5 cm by 2 W + 2 cm, lid 2 cm larger (board 3 mm, clearance 4 mm and wall
        3 mm per side); and the cell manual's nominal spot."""
        src = (TASK / "tests/shoecell/episode.py").read_text()
        m = re.search(r'lid = \{"pos": \[tx \+ float\(rng\.uniform\(-([0-9.]+), ([0-9.]+)\)\), '
                      r'ty - ([0-9.]+) \+ float\(rng\.uniform\(-([0-9.]+), ([0-9.]+)\)\)\],\s*'
                      r'"yaw": float\(rng\.choice\(\[0\.0, math\.pi\]\) \+ rng\.uniform\(-([0-9.]+), ([0-9.]+)\)\)\}', src)
        self.assertIsNotNone(m, "the lid draw of episode.py changed its form")
        jx, off, jy, jyaw = float(m.group(1)), float(m.group(3)), float(m.group(4)), float(m.group(6))
        self.assertEqual((m.group(1), m.group(4), m.group(6)), (m.group(2), m.group(5), m.group(7)))
        server = (TASK / "tests/shoecell/server.py").read_text()
        cx, cy, hx, hy = map(float, re.search(r'"infeed": \{"centre": \[([0-9.]+), ([0-9.]+)\], "half": \[([0-9.]+), ([0-9.]+)\]\}',
                                              server).groups())
        lo, wo = 0.21 + 0.05 + 0.020, 2 * 0.080 + 0.02 + 0.020
        reach_y = wo / 2 * math.cos(jyaw) + lo / 2 * math.sin(jyaw)
        clear_near = (cy - off - jy - reach_y) - (cy - hy + 0.004)
        self.assertGreaterEqual(clear_near, 0.003)
        reach_x = lo / 2 * math.cos(jyaw) + wo / 2 * math.sin(jyaw)
        self.assertGreaterEqual((cx - jx - reach_x) - (cx - hx + 0.004), 0.003)   # (the -x rim; +x is symmetric)
        manual = (TASK / "environment/setup/payload/given/cell/cell_manual.md").read_text()
        self.assertIn("the lid tray (near (%.2f, %s), either way round)" % (cx, ("%.3f" % (cy - off)).rstrip("0")), manual)


class AnchorTests(unittest.TestCase):
    def test_the_anchor_is_the_lowest_recorded_verify_of_the_set(self):
        verifies = ANCHOR["verifies"]
        self.assertGreaterEqual(len(verifies), 1)
        self.assertEqual(ANCHOR["value"], min(v["work_score"] for v in verifies))
        for v in verifies:
            self.assertEqual([e["seed"] for e in v["episodes"]], SEEDS["seeds"])
            mean = sum(e["work_score"] for e in v["episodes"]) / len(v["episodes"])
            self.assertAlmostEqual(v["work_score"], mean, places=12)
            self.assertIn(v["image_id"], recorded_images())
            self.assertEqual(v["seeds_sha256"], SELECTION["set"]["seeds_sha256"])
        self.assertIn(recorded_images()[-1], [v["image_id"] for v in verifies])   # a verify on the current image

    def test_full_at_follows_the_verifies_against_the_selection_runs(self):
        runs = {c["seed"]: c["result"] for c in SELECTION["candidates"] if c["status"] == "kept"}
        identical = all(e["work_score"] == runs[e["seed"]]["work_score"] for v in ANCHOR["verifies"] for e in v["episodes"])
        for v in ANCHOR["verifies"]:
            for e in v["episodes"]:
                self.assertEqual(e["equals_selection_run"], e["work_score"] == runs[e["seed"]]["work_score"])
        if identical:
            self.assertEqual(ANCHOR["full_at"], 0.98)
        else:
            self.assertLess(ANCHOR["full_at"], 0.98)
        for v in ANCHOR["verifies"]:  # the oracle's own verifies read reward 1.0 (half-up, two decimals)
            reward = min(1.0, max(0.0, v["work_score"] / ANCHOR["value"] / ANCHOR["full_at"]))
            self.assertGreaterEqual(reward, 0.995)


class AgentVisibleTests(unittest.TestCase):
    # the PR author's public cell code names its privileged calls (the grader runs this exact code; the calls answer
    # "forbidden" to a program without the token): pinned here as it stands, so no other agent-visible text starts to
    PRIVILEGED = re.compile(r"\bprivileged|reference_run|reference program's token", re.IGNORECASE)
    PINNED = {"environment/setup/payload/given/cell/cell_api.md": 1,
              "environment/setup/payload/given/lib/shoecell/__init__.py": 1,
              "environment/setup/payload/given/lib/shoecell/client.py": 3,
              "environment/setup/payload/given/lib/shoecell/local.py": 1,
              "environment/setup/payload/given/lib/shoecell/server.py": 12,
              "environment/setup/payload/practice/grade.py": 1}

    def test_nothing_agent_visible_names_the_token_the_set_or_its_selection(self):
        token = (TASK / "solution/reference/oracle_token").read_text().strip()
        digest = (HIDDEN / "oracle_token.sha256").read_text().split()[0]
        needles = [token, digest, "oracle_token", "privileged_mode", "seeds.json", "selection.json", "select_set",
                   "run_candidate", "regrade", "fixed set", "fixed hidden", "hidden set", "anchor", SELECTION["format"],
                   SEEDS["source"][:40]] + [str(c["seed"]) for c in SELECTION["candidates"]]
        for p in agent_visible():
            text = p.read_text(errors="replace")
            for needle in needles:
                self.assertNotIn(needle, text, "{} names {}".format(p.relative_to(TASK), needle[:24]))
        self.assertNotIn("fresh seeds", (TASK / "instruction.md").read_text())

    def test_the_cell_library_s_mentions_of_its_privileged_calls_are_the_pinned_ones(self):
        found = {}
        for p in agent_visible():
            n = sum(1 for line in p.read_text(errors="replace").splitlines() if self.PRIVILEGED.search(line))
            if n:
                found[p.relative_to(TASK).as_posix()] = n
        self.assertEqual(found, self.PINNED)


if __name__ == "__main__":
    unittest.main()
