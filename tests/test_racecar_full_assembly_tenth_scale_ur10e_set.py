"""racecar_full_assembly_tenth_scale_ur10e: the fixed hidden set, the grader's checks of it, and what the agent is given.

The grader reads the set straight from tests/hidden/episodes.json (format racecar-asm-set/1: four episodes, one per
stratum of arm/assembly_car/layout.py, each a seed and a layout digest) and compares the digest at every reset; a
malformed set, and a layout the cell builds differently, are task errors whose messages name no value of the set; every
agent process starts through netguard; the hidden kit is pinned by sha256 and file count (a maintainer's local copy
tests/assets/gt.tar.xz, when present, has that sha256); the oracle token's sha256 is the one under tests/hidden/; once
the set is frozen it holds four distinct seeds in [0, 2**63), one per stratum, the selection record keeps each of them,
and nothing agent-visible (instruction.md, environment/) holds a value of the set or the token. The safety rule (v1.7,
the owner's decision of 2026-11-16): an episode with a safety violation keeps grader_config.json
scoring.safety_violation_factor of its stage score (one factor per episode), factor 0 is the earlier rule, and every
agent-visible text that states the rule prints that factor and none says the episode is zeroed. The robot-contact limit
(v1.8, the owner's decision of 2026-11-17): one number, grader_config.json safety.robot_force_N = 150 (was 30), which
the grader checks and records, the cell's watch takes through the server, the cell's and the server's defaults repeat,
and every agent-visible text prints.

Stdlib only: the grader's set functions are taken from tests/verify.py by name (it imports numpy and MuJoCo at module
level, which the host venv does not have). Tests that need the frozen set skip until it exists
(tests/review/select_set.py writes tests/hidden/episodes.json and selection.json).
"""
import ast
import hashlib
import json
import math
import os
import re
import tempfile
import unittest
from pathlib import Path

SLUG = "racecar_full_assembly_tenth_scale_ur10e"
_HERE = Path(__file__).resolve()
_CANDIDATES = [_HERE.parents[1] / "tasks" / "ground" / SLUG, _HERE.parents[2] / SLUG]
TASK = next((p for p in _CANDIDATES if (p / "tests" / "verify.py").is_file()), _CANDIDATES[0])
TESTS = TASK / "tests"
VERIFY = TESTS / "verify.py"
HIDDEN = TESTS / "hidden"
SET_PATH = HIDDEN / "episodes.json"
SELECTION_PATH = HIDDEN / "selection.json"
CONFIG = json.loads((TESTS / "grader_config.json").read_text())
TEXT_EXT = (".md", ".txt", ".json", ".py", ".sh", ".toml", ".yaml", ".yml", ".xml", ".csv", ".mtl", ".cfg", ".ini")


class TaskError(Exception):
    pass


def _fail(reason):
    raise TaskError(reason)


def _grader(names, consts=("SET_FORMAT", "STRATA", "SEED_LIMIT")):
    """the named functions and module constants of tests/verify.py, executed in a stdlib namespace (fail -> TaskError)"""
    tree = ast.parse(VERIFY.read_text())
    body = [n for n in tree.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in consts for t in n.targets)]
    body += [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    have = {n.name for n in body if isinstance(n, ast.FunctionDef)}
    assert have == set(names), "verify.py lacks %s" % (set(names) - have)
    ns = {"json": json, "math": math, "os": os, "re": re, "fail": _fail}
    exec(compile(ast.Module(body=body, type_ignores=[]), str(VERIFY), "exec"), ns)
    return ns


G = _grader(["load_hidden_set", "check_layout_digest"])


def _layout_strata():
    tree = ast.parse((TESTS / "arm" / "assembly_car" / "layout.py").read_text())
    for n in tree.body:
        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "STRATA" for t in n.targets):
            return tuple(ast.literal_eval(n.value))
    raise AssertionError("layout.py has no STRATA")


def _agent_visible_texts():
    """(relative path, text) of every agent-visible text file: instruction.md and environment/**"""
    out = []
    p = TASK / "instruction.md"
    if p.is_file():
        out.append(("instruction.md", p.read_text(errors="replace")))
    env = TASK / "environment"
    for dp, dns, fns in os.walk(env):
        dns[:] = [d for d in dns if d != "__pycache__"]
        for fn in fns:
            q = Path(dp) / fn
            if q.suffix.lower() in TEXT_EXT or fn.startswith(("Dockerfile", "requirements")):
                out.append((str(q.relative_to(TASK)), q.read_text(errors="replace")))
    return out


def _bounded(value):
    return re.compile(r"(?<![0-9A-Za-z])" + re.escape(value) + r"(?![0-9A-Za-z])")


class GraderSetChecks(unittest.TestCase):
    """the grader's own set loader and digest check, on synthetic sets"""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)
        self.digests = [hashlib.sha256(b"layout %d" % i).hexdigest() for i in range(4)]
        self.seeds = [6151231234567890123, 4242, 9000000000000000001, 1 << 50]
        self.good = [{"seed": s, "stratum": st, "layout_digest": d} for s, st, d in zip(self.seeds, G["STRATA"], self.digests)]

    def _put(self, doc):
        (self.dir / "episodes.json").write_text(doc if isinstance(doc, str) else json.dumps(doc))

    def _set(self, eps):
        return {"format": G["SET_FORMAT"], "episodes": eps}

    def test_format_and_strata_are_the_cells(self):
        self.assertEqual(G["SET_FORMAT"], "racecar-asm-set/1")
        self.assertEqual(tuple(G["STRATA"]), _layout_strata())
        self.assertEqual(G["SEED_LIMIT"], 1 << 63)

    def test_a_valid_set_is_read_in_order(self):
        self._put(self._set(self.good))
        eps = G["load_hidden_set"]({"episodes": 4}, str(self.dir))
        self.assertEqual([e["seed"] for e in eps], self.seeds)
        self.assertEqual([e["stratum"] for e in eps], list(G["STRATA"]))

    def test_a_malformed_set_is_a_task_error_naming_no_value(self):
        g = self.good

        def with0(**kw):
            return [dict(g[0], **kw)] + g[1:]
        variants = {
            "missing file": None, "not json": "{oops", "wrong format": {"format": "x/1", "episodes": g},
            "no list": {"format": G["SET_FORMAT"], "episodes": {"a": 1}}, "three episodes": self._set(g[:3]),
            "five episodes": self._set(g + [dict(g[0], seed=77)]), "float seed": self._set(with0(seed=1.0)),
            "bool seed": self._set(with0(seed=True)), "negative seed": self._set(with0(seed=-1)),
            "seed of 2**63": self._set(with0(seed=1 << 63)), "string seed": self._set(with0(seed=str(g[0]["seed"]))),
            "repeated seed": self._set([g[0], dict(g[1], seed=g[0]["seed"])] + g[2:]),
            "unknown stratum": self._set(with0(stratum="yaw0lead0")),
            "two in one stratum": self._set([g[0], dict(g[1], stratum=g[0]["stratum"])] + g[2:]),
            "upper-case digest": self._set(with0(layout_digest=g[0]["layout_digest"].upper())),
            "short digest": self._set(with0(layout_digest=g[0]["layout_digest"][:63])),
            "no digest": self._set([{"seed": g[0]["seed"], "stratum": g[0]["stratum"]}] + g[1:]),
            "not an object": self._set([7] + g[1:]),
        }
        values = [str(s) for s in self.seeds] + self.digests + [d.upper() for d in self.digests] + [self.digests[0][:63]]
        for name, doc in variants.items():
            with self.subTest(name):
                if doc is None:
                    p = self.dir / "episodes.json"
                    if p.exists():
                        p.unlink()
                else:
                    self._put(doc)
                with self.assertRaises(TaskError) as cm:
                    G["load_hidden_set"]({"episodes": 4}, str(self.dir))
                msg = str(cm.exception)
                self.assertFalse([v for v in values if v in msg], msg)

    def test_a_different_layout_is_a_task_error_naming_no_digest(self):
        e = self.good[1]
        self.assertIs(G["check_layout_digest"](2, e, e["layout_digest"]), True)
        with self.assertRaises(TaskError) as cm:
            G["check_layout_digest"](2, e, "f" * 64)
        self.assertNotIn(e["layout_digest"], str(cm.exception))
        self.assertNotIn("f" * 64, str(cm.exception))
        self.assertIs(G["check_layout_digest"](2, e, "anything", nominal=True), False)


class GraderWiring(unittest.TestCase):
    """the grader's main path reads the set, checks every reset's digest and starts every agent process through netguard"""

    @classmethod
    def setUpClass(cls):
        tree = ast.parse(VERIFY.read_text())
        cls.fns = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}

    def _calls(self, fn):
        return [ast.unparse(n.func) for n in ast.walk(self.fns[fn]) if isinstance(n, ast.Call)]

    def test_the_set_is_read_from_hidden_and_drives_the_episodes(self):
        src = ast.unparse(self.fns["_main_locked"])
        self.assertIn("load_hidden_set(cfg, HIDDEN)", src)
        self.assertIn("for k, ep in enumerate(episodes, 1)", src)
        self.assertNotIn("seed_rng", src)                 # no OS-entropy draw: the set is fixed

    def test_every_reset_is_compared(self):
        calls = self._calls("run_episode")
        self.assertIn("server.reset", calls)
        self.assertIn("check_layout_digest", calls)
        src = ast.unparse(self.fns["run_episode"])
        self.assertLess(src.index("server.reset("), src.index("check_layout_digest("))
        self.assertLess(src.index("check_layout_digest("), src.index("wire.AgentProcess("))

    def test_agent_processes_start_through_netguard(self):
        self.assertIn("guarded", self._calls("run_episode"))
        self.assertIn("guarded", self._calls("contract_check"))
        self.assertIn("netguard_folder", self._calls("_main_locked"))
        g = ast.unparse(self.fns["guarded"])
        self.assertIn('"-I", "-S", guard, "exec", program_dir', g.replace("'", '"'))
        self.assertTrue((TESTS / "netguard.py").is_file())

    def test_the_kit_is_fetched_with_its_pins(self):
        src = ast.unparse(self.fns["fetch_hidden_gt"])
        for needle in ("assets", "gt.tar.xz", "sha256_file(src)", 'spec["files"]', "gt/", 'filter="data"'):
            self.assertIn(needle, src.replace("'", '"'))


class ConfigAndToken(unittest.TestCase):
    def test_grader_config_pins(self):
        self.assertEqual(CONFIG["episodes"], 4)
        self.assertIs(CONFIG["oracle_nominal_layout"], False)
        hg = CONFIG["hidden_gt"]
        self.assertRegex(hg["sha256"], r"\A[0-9a-f]{64}\Z")
        self.assertIsInstance(hg["files"], int)
        self.assertGreater(hg["files"], 0)
        self.assertTrue(hg["url"].endswith("/tasks/ground/%s/verifier/gt.tar.xz" % SLUG), hg["url"])
        self.assertEqual(CONFIG["scope"]["premounted"], [])
        self.assertIs(CONFIG["scoring"]["count_subassembly"], True)

    def test_the_givens_and_the_kit_are_pinned_at_one_published_dataset_commit(self):
        # published 2026-11-15, the folder moved to tasks/ground/ on 2026-11-17: the asset list and hidden_gt.url name
        # the same 40-hex commit of the asset dataset,
        # never the placeholder of forty zeros, and the kit sits in the task's own verifier/ folder next to payload/
        head = {}
        for ln in (TASK / "environment" / "assets.sha256").read_text(encoding="utf-8").splitlines():
            if ln.startswith("#"):
                k, _, v = ln[1:].partition(":")
                head[k.strip()] = v.strip()
        rev = head.get("revision", "")
        self.assertRegex(rev, r"\A[0-9a-f]{40}\Z")
        self.assertNotEqual(rev, "0" * 40, "the asset list holds the placeholder revision")
        self.assertEqual(head.get("prefix"), "tasks/ground/%s/payload" % SLUG)
        want = "%s/datasets/%s/resolve/%s/tasks/ground/%s/verifier/gt.tar.xz" % (
            head.get("endpoint", "https://huggingface.co").rstrip("/"), head.get("repo"), rev, SLUG)
        self.assertEqual(CONFIG["hidden_gt"]["url"], want)
        self.assertNotIn("placeholder", CONFIG["hidden_gt"].get("note", ""))

    def test_the_local_kit_copy_has_the_pinned_sha256(self):
        p = TESTS / "assets" / "gt.tar.xz"
        if not p.is_file():
            self.skipTest("no development copy tests/assets/gt.tar.xz here (an asset root, never in git)")
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for b in iter(lambda: f.read(1 << 20), b""):
                h.update(b)
        self.assertEqual(h.hexdigest(), CONFIG["hidden_gt"]["sha256"])

    def test_the_oracle_token(self):
        tok = (TASK / "solution" / "reference" / "oracle_token").read_bytes().strip()
        self.assertRegex(tok.decode(), r"\A[0-9a-f]{64}\Z")
        self.assertEqual((HIDDEN / "oracle_token.sha256").read_text().strip(), hashlib.sha256(tok).hexdigest())

    def test_nothing_agent_visible_holds_the_token_or_names_the_hidden_files(self):
        tok = (TASK / "solution" / "reference" / "oracle_token").read_text().strip()
        sha = (HIDDEN / "oracle_token.sha256").read_text().strip()
        texts = _agent_visible_texts()
        self.assertTrue(texts, "no agent-visible text file found")
        for rel, text in texts:
            for v in (tok, sha, CONFIG["hidden_gt"]["sha256"]):
                self.assertIsNone(_bounded(v).search(text), rel)
            for name in ("oracle_token", "hidden/episodes", "hidden/selection", "hidden/bench_plan"):
                self.assertNotIn(name, text, rel)


class SafetyRule(unittest.TestCase):
    """v1.7: an episode with a safety violation keeps the configured share of its stage score"""

    @classmethod
    def setUpClass(cls):
        cls.g = _grader(["safety_factor", "episode_values", "rule_words"], consts=())
        cls.factor = CONFIG["scoring"]["safety_violation_factor"]
        cls.fns = {n.name: n for n in ast.parse(VERIFY.read_text()).body if isinstance(n, ast.FunctionDef)}

    @staticmethod
    def _ep(stage, hard_ok):
        return {"stage_score": stage, "hard_ok": hard_ok}

    def test_the_factor_is_one_number_of_the_grader_configuration(self):
        self.assertEqual(self.factor, 0.3)
        self.assertEqual(self.g["safety_factor"](CONFIG), 0.3)

    def test_a_bad_factor_is_a_task_error(self):
        for bad in (None, -0.1, 1.5, float("nan"), float("inf"), True, "0.3"):
            cfg = {"scoring": {"count_subassembly": True}} if bad is None else {"scoring": {"safety_violation_factor": bad}}
            with self.assertRaises(TaskError, msg=repr(bad)):
                self.g["safety_factor"](cfg)
        for good in (0, 0.0, 0.3, 1):
            self.assertEqual(self.g["safety_factor"]({"scoring": {"safety_violation_factor": good}}), float(good))

    def test_one_factor_per_episode_and_clean_episodes_unchanged(self):
        ev = self.g["episode_values"]
        eps = [self._ep(1.0, False), self._ep(1.0, True), self._ep(9.0 / 11.5, False), self._ep(0.0, False)]
        self.assertEqual(ev(eps, 0.3), [1.0 * 0.3, 1.0, (9.0 / 11.5) * 0.3, 0.0])
        self.assertEqual(ev([self._ep(0.5, True)], 0.3), [0.5])

    def test_factor_zero_is_the_earlier_rule_bit_for_bit(self):
        ev = self.g["episode_values"]
        stages = [0.21739130434782608, 0.5217391304347826, 0.391304347826087, 0.2608695652173913, 1.0, 0.0]
        for bits in ([False] * 6, [True] * 6, [True, False, True, False, False, True]):
            eps = [self._ep(s_, b) for s_, b in zip(stages, bits)]
            old = [float(e["stage_score"]) * (1.0 if e["hard_ok"] else 0.0) for e in eps]
            self.assertEqual([repr(v) for v in ev(eps, 0.0)], [repr(v) for v in old])
        self.assertEqual(self.g["rule_words"](0.0), "an episode with a hard safety violation scoring 0")
        self.assertEqual(self.g["rule_words"](0.3), "an episode with a hard safety violation keeping 30 % of its score")

    def test_four_violating_episodes_keep_the_factor_of_their_mean(self):
        stages = [2.5 / 11.5, 6.0 / 11.5, 4.5 / 11.5, 3.0 / 11.5]
        vals = self.g["episode_values"]([self._ep(s_, False) for s_ in stages], self.factor)
        self.assertAlmostEqual(sum(vals) / 4.0, 0.3 * sum(stages) / 4.0, places=15)
        self.assertEqual(round(sum(vals) / 4.0, 3), 0.104)

    def test_the_grader_scores_with_the_rule(self):
        src = ast.unparse(self.fns["_main_locked"])
        self.assertIn("factor = safety_factor(cfg)", src)
        self.assertIn("vals = episode_values(agent, factor)", src)
        self.assertIn("metrics['safety_violation_factor'] = factor", src)
        self.assertLess(src.index("safety_factor(cfg)"), src.index("contract_check("))

    def test_the_agent_visible_texts_print_the_factor(self):
        pct = "%.0f %%" % (100.0 * self.factor)
        given = TASK / "environment" / "setup" / "payload" / "given" / "cell"
        texts = {"instruction.md": (TASK / "instruction.md").read_text(), "cell_manual.md": (given / "cell_manual.md").read_text(),
                 "cell_api.md": (given / "cell_api.md").read_text()}
        flat = {k: " ".join(v.split()) for k, v in texts.items()}
        sentence = "A safety violation does not end the episode, but an episode with one keeps only %s of its score." % pct
        self.assertIn(sentence, flat["instruction.md"])
        self.assertIn(sentence, flat["cell_manual.md"])
        self.assertIn("keeps only %s of its score" % pct, flat["cell_api.md"])
        for name, text in flat.items():
            for old in ("zero that episode", "zeroes the episode", "scores nothing", "does not zero the episode"):
                self.assertNotIn(old, text, name)
        public = json.loads((given / "cell_config.json").read_text())
        self.assertEqual(public["scoring"]["safety_violation_factor"], self.factor)


def _dict_entry(tree, var, key):
    """the AST node of `var = {..., key: <node>, ...}` at module level"""
    for n in tree.body:
        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == var for t in n.targets) and isinstance(n.value, ast.Dict):
            for k, v in zip(n.value.keys, n.value.values):
                if isinstance(k, ast.Constant) and k.value == key:
                    return v
    raise AssertionError("%s has no %r" % (var, key))


class ContactLimit(unittest.TestCase):
    """v1.8: the robot-contact limit is one number of the grader configuration, 150 N"""

    @classmethod
    def setUpClass(cls):
        cls.g = _grader(["contact_limit"], consts=())
        cls.limit = CONFIG["safety"]["robot_force_N"]
        cls.fns = {n.name: n for n in ast.parse(VERIFY.read_text()).body if isinstance(n, ast.FunctionDef)}
        cls.car = TESTS / "arm" / "assembly_car"

    def test_the_limit_is_one_number_of_the_grader_configuration(self):
        self.assertEqual(self.limit, 150.0)
        self.assertEqual(self.g["contact_limit"](CONFIG), 150.0)
        self.assertEqual(CONFIG["damage"]["part_force_N"], 150.0)        # the unit-pressed rule, unchanged
        self.assertEqual(CONFIG["damage"]["ticks"], 20)

    def test_a_bad_limit_is_a_task_error(self):
        for bad in (None, 0, 0.0, -30.0, float("nan"), float("inf"), True, "150"):
            cfg = {"safety": {}} if bad is None else {"safety": {"robot_force_N": bad}}
            with self.assertRaises(TaskError, msg=repr(bad)):
                self.g["contact_limit"](cfg)
        for good in (30, 30.0, 150.0):
            self.assertEqual(self.g["contact_limit"]({"safety": {"robot_force_N": good}}), float(good))

    def test_the_grader_reads_and_records_it(self):
        src = ast.unparse(self.fns["_main_locked"])
        self.assertIn("limit = contact_limit(cfg)", src)
        self.assertIn("metrics['robot_contact_limit_N'] = limit", src)
        self.assertLess(src.index("contact_limit(cfg)"), src.index("contract_check("))

    def test_the_cell_takes_it_from_the_configuration(self):
        server = ast.parse((self.car / "server.py").read_text())
        cls = [n for n in server.body if isinstance(n, ast.ClassDef) and n.name == "CarInfoServer"][0]
        cc = ast.unparse([n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "cell_config"][0])
        self.assertIn("'robot_force_N': float(c['safety']['robot_force_N'])", cc)
        self.assertIn("float(W['robot_force_N'])", ast.unparse(ast.parse((self.car / "cell.py").read_text())))
        desc = ast.unparse([n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "describe"][0])
        self.assertIn("'robot_force_N': float(self.cfg['safety']['robot_force_N'])", desc)

    def test_the_defaults_are_the_same_number(self):
        watch = _dict_entry(ast.parse((self.car / "cell.py").read_text()), "CELL_CONFIG", "watch")
        self.assertEqual(ast.literal_eval(watch)["robot_force_N"], self.limit)
        safety = _dict_entry(ast.parse((self.car / "server.py").read_text()), "DEFAULT_CAR_CONFIG", "safety")
        expr = {ast.literal_eval(k): ast.unparse(v) for k, v in zip(safety.keys, safety.values)}
        self.assertEqual(expr["robot_force_N"], "CELL_CONFIG['watch']['robot_force_N']")

    def test_the_agent_visible_texts_print_it(self):
        given = TASK / "environment" / "setup" / "payload" / "given" / "cell"
        flat = {name: " ".join(p.read_text().split()) for name, p in (("instruction.md", TASK / "instruction.md"),
                                                                       ("cell_manual.md", given / "cell_manual.md"))}
        rule = "a robot link other than the gripper's four pad boxes pressing with more than %.0f N on the chassis" % self.limit
        for name, text in flat.items():
            self.assertIn(rule, text, name)
            self.assertNotIn("more than 30 N", text, name)
        public = json.loads((given / "cell_config.json").read_text())
        self.assertEqual(public["safety"]["robot_force_N"], self.limit)
        copy = json.loads((TESTS / "setup_payload" / "given" / "cell" / "cell_config.json").read_text())
        self.assertEqual(copy["safety"]["robot_force_N"], self.limit)


class FrozenSet(unittest.TestCase):
    """the frozen set itself (skips until tests/review/select_set.py has written tests/hidden/episodes.json)"""

    def setUp(self):
        if not SET_PATH.is_file():
            self.skipTest("tests/hidden/episodes.json not written yet (tests/review/select_set.py selects it)")
        self.set = json.loads(SET_PATH.read_text())

    def test_the_set_passes_the_graders_own_loader(self):
        eps = G["load_hidden_set"](CONFIG, str(HIDDEN))
        self.assertEqual(len(eps), 4)
        self.assertEqual(sorted(e["stratum"] for e in eps), sorted(G["STRATA"]))
        self.assertEqual(len({e["seed"] for e in eps}), 4)

    def test_the_selection_keeps_every_episode(self):
        if not SELECTION_PATH.is_file():
            self.skipTest("tests/hidden/selection.json not written yet")
        sel = json.loads(SELECTION_PATH.read_text())
        cands = sel.get("candidates") or []
        seeds = {c.get("seed") for c in cands if isinstance(c, dict)}
        for e in self.set["episodes"]:
            self.assertIn(e["seed"], seeds)
        kept = [c for c in cands if isinstance(c, dict) and c.get("seed") in {e["seed"] for e in self.set["episodes"]}]
        for c in kept:
            if "selection" in c:
                self.assertEqual(c["selection"], "kept", c.get("seed"))

    def test_nothing_agent_visible_holds_a_value_of_the_set(self):
        values = []
        for e in self.set["episodes"]:
            values += [str(e["seed"]), e["layout_digest"]]
            if len("%x" % e["seed"]) >= 8:
                values.append("%x" % e["seed"])
        if SELECTION_PATH.is_file():
            for c in json.loads(SELECTION_PATH.read_text()).get("candidates") or []:
                if isinstance(c, dict) and isinstance(c.get("seed"), int):
                    values.append(str(c["seed"]))
        for rel, text in _agent_visible_texts():
            for v in values:
                self.assertIsNone(_bounded(v).search(text), "%s holds a value of the hidden set" % rel)


if __name__ == "__main__":
    unittest.main()
