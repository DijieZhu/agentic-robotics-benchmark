"""rov_final_assembly_and_trim_single_arm: the fixed hidden set, the grader's checks of it, its formula, and what the agent
is given.

The grader reads the set straight from tests/hidden/episodes.json (format rov-asm-set/1: four episodes E1..E4, each a
layout seed, a dome end, a work order, the vehicle's per-unit deviation and a layout digest) and compares the digest at
every reset; a malformed set, and a layout the cell builds differently, are task errors whose messages name no value of
the set; every agent process starts through netguard; the hidden kit is pinned by sha256 and file count (the development
copy tests/assets/gt.tar.xz, when present, has that sha256); the reference token's sha256 is the one under
tests/hidden/; the score block of grader_config.json is the trim table's acceptance; the anchor saturates at 0.97 on a
hardware-free metric; once the set is frozen it holds four distinct seeds in [0, 2**63), one per stratum in order, each
dome end twice, deviations inside the trim table's published range, and nothing agent-visible (instruction.md,
environment/) holds a value of the set or the token.

Stdlib only: the grader's set functions are taken from tests/verify.py by name (it imports numpy and MuJoCo at module
level, which the host venv does not have). Tests that need the frozen set skip until it exists (the hidden-set package
writes tests/hidden/episodes.json and selection.json).
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

SLUG = "rov_final_assembly_and_trim_single_arm"
_HERE = Path(__file__).resolve()
# in the benchmark repository: tests/<this file> beside tasks/underwater/<slug>; in the development tree: <slug>-dev/repo_tests/
_CANDIDATES = [_HERE.parents[1] / "tasks" / "underwater" / SLUG, _HERE.parents[2] / SLUG]
TASK = next((p for p in _CANDIDATES if (p / "tests" / "verify.py").is_file()), _CANDIDATES[0])
TESTS = TASK / "tests"
VERIFY = TESTS / "verify.py"
HIDDEN = TESTS / "hidden"
SET_PATH = HIDDEN / "episodes.json"
SELECTION_PATH = HIDDEN / "selection.json"
CONFIG = json.loads((TESTS / "grader_config.json").read_text())
ANCHOR = json.loads((TESTS / "anchor.json").read_text())
TABLE = json.loads((TESTS / "trim_table.json").read_text())
TEXT_EXT = (".md", ".txt", ".json", ".py", ".sh", ".toml", ".yaml", ".yml", ".xml", ".csv", ".mtl", ".cfg", ".ini")
CONSTS = ("SET_FORMAT", "STRATA", "DOME_ENDS", "SEED_LIMIT", "WATER_T_RANGE_C", "WATER_SP_RANGE", "UNIT_RE")


class TaskError(Exception):
    pass


def _fail(reason):
    raise TaskError(reason)


def _grader(names, consts=CONSTS):
    """the named functions and module constants of tests/verify.py, executed in a stdlib namespace (fail -> TaskError)"""
    tree = ast.parse(VERIFY.read_text())
    body = [n for n in tree.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in consts for t in n.targets)]
    body += [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    have = {n.name for n in body if isinstance(n, ast.FunctionDef)}
    assert have == set(names), "verify.py lacks %s" % (set(names) - have)
    ns = {"json": json, "os": os, "re": re, "math": math, "fail": _fail}
    exec(compile(ast.Module(body=body, type_ignores=[]), str(VERIFY), "exec"), ns)
    return ns


G = _grader(["load_hidden_set", "check_layout_digest", "_finite"])


def _module_tuple(path, name):
    tree = ast.parse(path.read_text())
    for n in tree.body:
        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in n.targets):
            return ast.literal_eval(n.value)
    raise AssertionError("%s has no %s" % (path.name, name))


def _token():
    toks = re.findall(r"(?<![0-9a-f])[0-9a-f]{64}(?![0-9a-f])", (TASK / "solution" / "solve.sh").read_text())
    assert len(toks) == 1, "solution/solve.sh holds %d tokens" % len(toks)
    return toks[0]


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


def _episode(i, seed, stratum, dome, digest):
    waters = [(8.2, 0.2), (13.3, 17.7), (6.3, 34.1), (28.3, 37.7)]
    devs = [(0.244, -6.36, 3.78), (-0.243, -4.40, 6.90), (-0.198, -6.73, 7.91), (-0.226, -6.99, 4.77)]
    t, s = waters[i]
    dm, dx, dy = devs[i]
    return {"seed": seed, "stratum": stratum, "dome_end_deg": dome, "layout_digest": digest,
            "work_order": {"unit": "ROV-B07-%04d" % (2000 + i), "delivery_site": {"water_temperature_C": t, "salinity_psu": s}},
            "deviation": {"dm_kg": dm, "dx_mm": dx, "dy_mm": dy}}


class GraderSetChecks(unittest.TestCase):
    """the grader's own set loader and digest check, on synthetic sets"""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)
        self.digests = [hashlib.sha256(b"layout %d" % i).hexdigest() for i in range(4)]
        self.seeds = [6151231234567890123, 4242, 9000000000000000001, 1 << 50]
        self.good = [_episode(i, s, st, d, g) for i, (s, st, d, g) in
                     enumerate(zip(self.seeds, G["STRATA"], (0, 180, 0, 180), self.digests))]
        self.rng = TABLE["per_unit_deviation"]

    def _put(self, doc):
        (self.dir / "episodes.json").write_text(doc if isinstance(doc, str) else json.dumps(doc))

    def _set(self, eps):
        return {"format": G["SET_FORMAT"], "episodes": eps}

    def test_format_strata_and_dome_ends(self):
        self.assertEqual(G["SET_FORMAT"], "rov-asm-set/1")
        self.assertEqual(tuple(G["STRATA"]), ("E1", "E2", "E3", "E4"))
        self.assertEqual(tuple(G["DOME_ENDS"]), (0, 180))
        self.assertEqual(G["SEED_LIMIT"], 1 << 63)
        # the layout's strata are the dome ends (arm/assembly_rov/layout.py), the server's hidden strata the four E's
        self.assertEqual(tuple(_module_tuple(TESTS / "arm" / "assembly_rov" / "layout.py", "STRATA")), ("dome0", "dome180"))
        self.assertEqual(tuple(_module_tuple(TESTS / "arm" / "assembly_rov" / "server.py", "HIDDEN_STRATA")), tuple(G["STRATA"]))
        self.assertEqual(tuple(_module_tuple(TESTS / "arm" / "assembly_rov" / "server.py", "DOME_ENDS")), tuple(G["DOME_ENDS"]))

    def test_a_valid_set_is_read_in_order(self):
        self._put(self._set(self.good))
        eps = G["load_hidden_set"]({"episodes": 4}, str(self.dir), self.rng)
        self.assertEqual([e["seed"] for e in eps], self.seeds)
        self.assertEqual([e["stratum"] for e in eps], list(G["STRATA"]))
        self.assertEqual(eps[1]["work_order"], self.good[1]["work_order"])
        self.assertEqual(eps[2]["deviation"], self.good[2]["deviation"])

    def test_a_malformed_set_is_a_task_error_naming_no_value(self):
        g = self.good

        def with0(**kw):
            return [dict(g[0], **kw)] + g[1:]
        wo = g[0]["work_order"]
        variants = {
            "missing file": None, "not json": "{oops", "wrong format": {"format": "x/1", "episodes": g},
            "no list": {"format": G["SET_FORMAT"], "episodes": {"a": 1}}, "three episodes": self._set(g[:3]),
            "five episodes": self._set(g + [dict(g[0], seed=77)]), "float seed": self._set(with0(seed=1.0)),
            "bool seed": self._set(with0(seed=True)), "negative seed": self._set(with0(seed=-1)),
            "seed of 2**63": self._set(with0(seed=1 << 63)), "string seed": self._set(with0(seed=str(g[0]["seed"]))),
            "repeated seed": self._set([g[0], dict(g[1], seed=g[0]["seed"])] + g[2:]),
            "unknown stratum": self._set(with0(stratum="E5")),
            "strata out of order": self._set([g[1], g[0]] + g[2:]),
            "dome end 90": self._set(with0(dome_end_deg=90)), "dome end as text": self._set(with0(dome_end_deg="0")),
            "three of one dome end": self._set(with0(dome_end_deg=180)),
            "upper-case digest": self._set(with0(layout_digest=g[0]["layout_digest"].upper())),
            "short digest": self._set(with0(layout_digest=g[0]["layout_digest"][:63])),
            "no digest": self._set([{k: v for k, v in g[0].items() if k != "layout_digest"}] + g[1:]),
            "no work order": self._set(with0(work_order=None)),
            "no water": self._set(with0(work_order={"unit": wo["unit"]})),
            "water out of range": self._set(with0(work_order=dict(wo, delivery_site={"water_temperature_C": 61.0, "salinity_psu": 0.2}))),
            "salinity out of range": self._set(with0(work_order=dict(wo, delivery_site={"water_temperature_C": 8.2, "salinity_psu": -1.0}))),
            "water as text": self._set(with0(work_order=dict(wo, delivery_site={"water_temperature_C": "8.2", "salinity_psu": 0.2}))),
            "unit with a path": self._set(with0(work_order=dict(wo, unit="../../x"))),
            "no deviation": self._set(with0(deviation=None)),
            "deviation outside the range": self._set(with0(deviation={"dm_kg": 0.30, "dx_mm": 0.0, "dy_mm": 0.0})),
            "deviation as text": self._set(with0(deviation={"dm_kg": "0.1", "dx_mm": 0.0, "dy_mm": 0.0})),
            "not an object": self._set([7] + g[1:]),
        }
        values = ([str(s) for s in self.seeds] + self.digests + [d.upper() for d in self.digests] + [self.digests[0][:63]] +
                  ["8.2", "13.3", "17.7", "34.1", "28.3", "37.7", "0.244", "-0.243", "-6.36", "6.9"])
        for name, doc in variants.items():
            with self.subTest(name):
                if doc is None:
                    p = self.dir / "episodes.json"
                    if p.exists():
                        p.unlink()
                else:
                    self._put(doc)
                with self.assertRaises(TaskError) as cm:
                    G["load_hidden_set"]({"episodes": 4}, str(self.dir), self.rng)
                msg = str(cm.exception)
                self.assertFalse([v for v in values if _bounded(v).search(msg)], msg)

    def test_a_different_layout_is_a_task_error_naming_no_digest(self):
        e = self.good[1]
        self.assertIs(G["check_layout_digest"](2, e, e["layout_digest"]), True)
        with self.assertRaises(TaskError) as cm:
            G["check_layout_digest"](2, e, "f" * 64)
        self.assertNotIn(e["layout_digest"], str(cm.exception))
        self.assertNotIn("f" * 64, str(cm.exception))
        self.assertIs(G["check_layout_digest"](2, e, "anything", nominal=True), False)


class GraderWiring(unittest.TestCase):
    """the grader's main path reads the set, checks every reset's digest, starts every agent process through netguard,
    gives no submission the honest zero and computes the score as DESIGN v1.2 section 3.1 says"""

    @classmethod
    def setUpClass(cls):
        tree = ast.parse(VERIFY.read_text())
        cls.fns = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}

    def _calls(self, fn):
        return [ast.unparse(n.func) for n in ast.walk(self.fns[fn]) if isinstance(n, ast.Call)]

    def test_the_set_is_read_from_hidden_and_drives_the_episodes(self):
        src = ast.unparse(self.fns["_main_locked"])
        self.assertIn("load_hidden_set(cfg, HIDDEN", src)
        self.assertIn("for k, ep in enumerate(episodes, 1)", src)
        self.assertNotIn("seed_rng", src)                 # no OS-entropy draw: the set is fixed
        self.assertIn("check_trim_config(cfg, table)", src)

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
        self.assertIn("launcher_argv", self._calls("run_episode"))
        self.assertIn("from arm.assembly_rov.cell_client_ext import launcher_argv", VERIFY.read_text())

    def test_no_submission_is_the_honest_zero(self):
        src = ast.unparse(self.fns["_main_locked"])
        self.assertLess(src.index("submission_missing()"), src.index("fetch_hidden_gt(cfg)"))
        self.assertIn("zero_verdict", ast.unparse(self.fns["write_zero_verdict"]))

    def test_the_formula(self):
        ns = {"np": None}
        tree = ast.parse(VERIFY.read_text())
        body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ("contract_gate", "assembly_score")]

        class _Np(object):
            @staticmethod
            def mean(xs):
                xs = list(xs)
                return sum(xs) / len(xs)
        ns["np"] = _Np
        exec(compile(ast.Module(body=body, type_ignores=[]), str(VERIFY), "exec"), ns)
        gate, score = ns["contract_gate"], ns["assembly_score"]
        self.assertEqual(gate(1, 0, 0, 0)[0], 1.0)
        for args in ((0, 0, 0, 0), (3, 1, 0, 0), (3, 0, 2, 0), (3, 0, 0, 5)):
            self.assertEqual(gate(*args)[0], 0.0, args)
        self.assertEqual(score(1.0, [1.0] * 4, [True] * 4), 1.0)
        self.assertAlmostEqual(score(1.0, [1.0, 11.0 / 15.5, 1.0, 1.0], [True] * 4), (3 + 11.0 / 15.5) / 4)
        self.assertEqual(score(1.0, [1.0] * 4, [True, True, False, True]), 0.75)
        self.assertEqual(score(0.0, [1.0] * 4, [True] * 4), 0.0)

    def test_the_kit_is_fetched_with_its_pins(self):
        src = ast.unparse(self.fns["fetch_hidden_gt"])
        for needle in ("assets", "gt.tar.xz", "sha256_file(src)", 'spec["files"]', "gt/", 'filter="data"'):
            self.assertIn(needle, src.replace("'", '"'))


class ConfigAnchorAndToken(unittest.TestCase):
    def test_grader_config_pins(self):
        self.assertEqual(CONFIG["episodes"], 4)
        self.assertIs(CONFIG["oracle_nominal_layout"], False)
        hg = CONFIG["hidden_gt"]
        self.assertRegex(hg["sha256"], r"\A[0-9a-f]{64}\Z")
        self.assertIsInstance(hg["files"], int)
        self.assertGreater(hg["files"], 0)
        self.assertTrue(hg["url"].endswith("/tasks/underwater/%s/verifier/gt.tar.xz" % SLUG), hg["url"])
        self.assertEqual((CONFIG["sim_cap_s"], CONFIG["wall_cap_s"], CONFIG["camera_budget_frames"], CONFIG["max_ticks_per_step"]),
                         (1200.0, 2400.0, 600, 500))
        self.assertEqual(CONFIG["fixture"]["max_turns"], 4)
        self.assertIs(CONFIG["layout"]["lmag"], False)              # rulings J8 / M10
        self.assertEqual(CONFIG["physics"], {})                      # the designed physics (ruling M8)

    def test_the_score_block_is_the_trim_tables(self):
        sc, acc = CONFIG["score"], TABLE["acceptance"]
        self.assertEqual(sc["trim_weight"], 4.5)
        self.assertEqual(sc["total_weight"], 15.5)
        t = sc["trim"]
        self.assertEqual(t["net_buoyancy_window_N"], acc["net_buoyancy_N"])
        self.assertEqual(t["angle_window_deg"], acc["pitch_abs_deg"])
        self.assertEqual(t["angle_window_deg"], acc["roll_abs_deg"])
        self.assertEqual((t["share_net_buoyancy"], t["share_pitch"], t["share_roll"]),
                         (acc["score_shares"]["net_buoyancy"], acc["score_shares"]["pitch"], acc["score_shares"]["roll"]))
        self.assertEqual(CONFIG["fixture"]["load_cells"]["bias_N"], CONFIG["layout"]["cell_bias_max_N"])

    def test_the_anchor(self):
        self.assertEqual(ANCHOR["metric"], "assembly_score")
        self.assertEqual(ANCHOR["direction"], "higher")
        self.assertEqual(ANCHOR["value"], 1.0)
        self.assertEqual(ANCHOR["full_at"], 0.97)
        hw = {"time", "runtime", "walltime", "wallclock", "latency", "ms", "msec", "millis", "milliseconds", "fps", "throughput",
              "frequency", "freq", "hz", "khz", "mhz", "ghz"}
        toks = re.findall(r"[A-Za-z]+", re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", ANCHOR["metric"]))
        self.assertFalse([x for x in toks if x.lower() in hw])
        self.assertTrue(ANCHOR["source"].strip())
        self.assertIsNone(re.search(r"paper|reported|published", ANCHOR["source"], re.I))

    def test_the_local_kit_copy_has_the_pinned_sha256(self):
        p = TESTS / "assets" / "gt.tar.xz"
        if not p.is_file():
            self.skipTest("no development copy tests/assets/gt.tar.xz here (an asset root, never in git)")
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for b in iter(lambda: f.read(1 << 20), b""):
                h.update(b)
        self.assertEqual(h.hexdigest(), CONFIG["hidden_gt"]["sha256"])

    def test_the_reference_token(self):
        tok = _token()
        self.assertEqual((HIDDEN / "oracle_token.sha256").read_text().strip(), hashlib.sha256(tok.encode()).hexdigest())
        sh = (TASK / "solution" / "solve.sh").read_text()
        self.assertIn(".oracle_token", sh)
        self.assertTrue((TASK / "solution" / "assemble.py").is_file())
        self.assertIn('_call("reference_assemble")', (TASK / "solution" / "assemble.py").read_text().replace("'", '"'))

    def test_nothing_agent_visible_holds_the_token_or_names_the_hidden_files(self):
        tok = _token()
        sha = (HIDDEN / "oracle_token.sha256").read_text().strip()
        texts = _agent_visible_texts()
        self.assertTrue(texts, "no agent-visible text file found")
        for rel, text in texts:
            for v in (tok, sha, CONFIG["hidden_gt"]["sha256"]):
                self.assertIsNone(_bounded(v).search(text), rel)
            for name in ("oracle_token", "hidden/episodes", "hidden/selection"):
                self.assertNotIn(name, text, rel)


class FrozenSet(unittest.TestCase):
    """the frozen set itself (skips until the hidden-set package has written tests/hidden/episodes.json)"""

    def setUp(self):
        if not SET_PATH.is_file():
            self.skipTest("tests/hidden/episodes.json not written yet (the hidden-set package selects it)")
        self.set = json.loads(SET_PATH.read_text())

    def test_the_set_passes_the_graders_own_loader(self):
        eps = G["load_hidden_set"](CONFIG, str(HIDDEN), TABLE["per_unit_deviation"])
        self.assertEqual(len(eps), 4)
        self.assertEqual([e["stratum"] for e in eps], list(G["STRATA"]))
        self.assertEqual(len({e["seed"] for e in eps}), 4)
        self.assertEqual(sorted(e["dome_end_deg"] for e in eps), [0, 0, 180, 180])

    def test_the_selection_records_the_set(self):
        if not SELECTION_PATH.is_file():
            self.skipTest("tests/hidden/selection.json not written yet")
        sel = json.loads(SELECTION_PATH.read_text())
        rec = sel.get("set") or {}
        digest = rec.get("episodes_sha256") or rec.get("sha256")
        if digest is None:
            self.skipTest("selection.json carries no sha256 of episodes.json")
        self.assertEqual(digest, hashlib.sha256(SET_PATH.read_bytes()).hexdigest())

    def test_nothing_agent_visible_holds_a_value_of_the_set(self):
        values = []
        for e in self.set["episodes"]:
            values += [str(e["seed"]), e["layout_digest"], e["work_order"]["unit"]]
            if len("%x" % e["seed"]) >= 8:
                values.append("%x" % e["seed"])
        for rel, text in _agent_visible_texts():
            for v in values:
                self.assertIsNone(_bounded(v).search(text), "%s holds a value of the hidden set" % rel)


if __name__ == "__main__":
    unittest.main()
