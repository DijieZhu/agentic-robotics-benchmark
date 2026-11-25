"""cable_routing_through_clips: the commissioning service, the hidden set's rule, and what the instruction states.

The agent has no simulator of the cell. Its sandbox runs a commissioning service (environment/commission_service/)
under another account: the grader's own simulator and renderer (byte copies) on three development boards that are
not graded boards, for a stated window of bench time, answering the public observation only. The graded set is the
one the pre-registered rule gives (tests/review/PREREG.md), and the numbers the instruction states are the code's.

Standard library only (the host-side suite has no numpy): files are read as text and syntax trees, not run.
"""

import ast
import hashlib
import json
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / "tasks/arm/cable_routing_through_clips"
SERVICE = TASK / "environment/commission_service"
PAYLOAD = TASK / "environment/setup/payload"
HIDDEN = TASK / "tests/hidden"
BENCH = json.loads((SERVICE / "bench.json").read_text())
CONFIG = json.loads((TASK / "tests/grader_config.json").read_text())
INSTRUCTION = " ".join((TASK / "instruction.md").read_text().split())
SIMULATOR = ("harness_env.py", "harness_sim.py", "harness_render.py", "harness_scene.py", "harness_geom.py",
             "harness_parts.py", "harness_looks.py", "harness_look.json")
GRADED_KEY = "cable_routing_through_clips/graded/v%d/%d"
DEV_KEY = "cable_routing_through_clips/dev/v1/%d"


def constants(path, names):
    out = {}
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in names:
                    try:
                        out[target.id] = ast.literal_eval(node.value)
                    except ValueError:
                        pass
    return out


def key_seed(text):
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big")


class Copies(unittest.TestCase):
    def test_service_and_oracle_run_the_graders_simulator(self):
        for name in SIMULATOR:
            self.assertEqual((SERVICE / name).read_bytes(), (TASK / "tests" / name).read_bytes(),
                             "commission_service/%s differs from tests/" % name)
        self.assertEqual((TASK / "solution/oracle/harness_env.py").read_bytes(), (TASK / "tests/harness_env.py").read_bytes())
        files = sorted(p.relative_to(SERVICE).as_posix() for p in SERVICE.rglob("*")
                       if p.is_file() and "__pycache__" not in p.parts)
        self.assertEqual(files, sorted(SIMULATOR + ("bench.json", "commissiond.py")),
                         "the service's folder holds the daemon, the simulator and the renderer; nothing hidden")

    def test_agent_home_holds_no_simulator_and_no_truth(self):
        names = sorted(p.relative_to(PAYLOAD).as_posix() for p in PAYLOAD.rglob("*") if p.is_file() and "__pycache__" not in p.parts)
        self.assertEqual(names, ["NOTICE.md", "commission/__init__.py", "commission/client.py", "commission/run_bench.py",
                                 "helpers/wire_skeleton.py", "session_time.sh"])
        for p in PAYLOAD.rglob("*.py"):
            text = p.read_text()
            for word in ("harness_env", "mujoco", "truth", "tension_mean", "evaluate("):
                self.assertNotIn(word, text, "%s names %s" % (p.name, word))


class Service(unittest.TestCase):
    def setUp(self):
        self.src = (SERVICE / "commissiond.py").read_text()
        self.tree = ast.parse(self.src)

    def test_development_boards_follow_the_registered_key(self):
        fn = [n for n in self.tree.body if isinstance(n, ast.FunctionDef) and n.name == "board_seed"]
        ns = {"hashlib": hashlib}
        exec(compile(ast.Module(body=fn, type_ignores=[]), "board_seed", "exec"), ns)
        self.assertEqual(BENCH["boards"], 3)
        self.assertEqual(BENCH["board_fixtures"], 3)
        dev = [ns["board_seed"](j) for j in range(BENCH["boards"])]
        self.assertEqual(dev, [key_seed(DEV_KEY % j) for j in range(3)])
        self.assertIn("not 0 <= board < BOARDS", self.src)           # the service stages no other board
        graded = set(int(e["seed"]) for e in json.loads((HIDDEN / "episodes.json").read_text())["episodes"])
        selection = json.loads((HIDDEN / "episode_selection.json").read_text())
        version = int(re.search(r"/v(\d+)/", selection["key"]).group(1))
        candidates = set(key_seed(GRADED_KEY % (version, c["k"])) for c in selection["candidates"])
        numerics = int(json.loads((TASK / "tests/numerics.json").read_text())["seed"])
        selftest = constants(TASK / "tests/verify.py", ("SELFTEST_SEED",))["SELFTEST_SEED"]
        self.assertFalse(set(dev) & (graded | candidates), "a development board is a graded board or a candidate")
        self.assertFalse({numerics, selftest} & (graded | candidates | set(dev)))

    def test_answers_no_truth_no_credit_no_debrief(self):
        imported = set()
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Import):
                imported |= {a.name for a in node.names}
            elif isinstance(node, ast.ImportFrom):
                imported.add(node.module)
        self.assertEqual(imported, {"fcntl", "hashlib", "json", "os", "resource", "select", "signal", "socket", "sys",
                                    "time", "numpy", "harness_sim"})
        used = {node.attr for node in ast.walk(self.tree) if isinstance(node, ast.Attribute)}
        for forbidden in ("truth", "credit", "evaluate", "env", "cable_points", "damaged"):
            self.assertNotIn(forbidden, used, "the service touches .%s" % forbidden)
        # the settle at an episode's end is run and its verdict dropped: the call is a statement, never a value
        calls = [n for n in ast.walk(self.tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                 and n.func.attr == "finish"]
        stmts = [n.value for n in ast.walk(self.tree) if isinstance(n, ast.Expr)]
        self.assertEqual(len(calls), 1)
        self.assertIn(calls[0], stmts)
        for word in ("truth", "credit", "routed", "prefix"):
            self.assertNotIn('"%s"' % word, self.src)

    def test_window_is_the_stated_one(self):
        self.assertEqual((BENCH["window_steps"], BENCH["reset_steps"], BENCH["image_steps"], BENCH["max_benches"]),
                         (20000, 100, 2, 2))
        self.assertIn("The window is **20000 control steps** in all; staging a board costs 100, every picture costs 2",
                      INSTRUCTION)
        self.assertIn("At most two boards at a time", INSTRUCTION)


class HiddenSet(unittest.TestCase):
    def setUp(self):
        self.set = [int(e["seed"]) for e in json.loads((HIDDEN / "episodes.json").read_text())["episodes"]]
        self.sel = json.loads((HIDDEN / "episode_selection.json").read_text())
        self.version = int(re.search(r"/v(\d+)/", self.sel["key"]).group(1))

    def test_set_is_what_the_registered_rule_gives(self):
        cand = self.sel["candidates"]
        self.assertEqual([c["k"] for c in cand], list(range(len(cand))), "every candidate walked is recorded, from 0")
        kept, counts, open_seen = [], {3: 0, 4: 0}, 0
        for c in cand:
            n = c.get("fixtures")
            if c.get("generator") != "ok":
                want = "no episode"
            elif n not in counts:
                want = "skipped: %d fixtures" % n
            elif counts[n] >= {3: 12, 4: 8}[n]:
                want = "class full"
            else:
                open_seen += 1
                want = "kept" if c.get("credit") == 1.0 else "reference failed"
                if want == "kept":
                    counts[n] += 1
                    kept.append(c["k"])
            self.assertEqual(c["outcome"], want, "candidate %d" % c["k"])
        self.assertEqual(counts, {3: 12, 4: 8})
        self.assertEqual(cand[-1]["outcome"], "kept", "the walk stops at the twentieth kept candidate")
        self.assertEqual(self.sel["kept"], kept)
        self.assertEqual(self.set, [key_seed(GRADED_KEY % (self.version, k)) for k in kept])
        self.assertEqual(len(set(self.set)), 20)
        self.assertGreaterEqual(len(kept) / open_seen, 0.7, "below 0.7 the rule says: improve the reference and walk again")
        self.assertEqual(CONFIG["episodes"], 20)

    def test_token_and_set_are_not_in_an_image_or_the_agents_files(self):
        token = (TASK / "solution/oracle_token").read_text().strip()
        self.assertEqual(hashlib.sha256(token.encode()).hexdigest(), (HIDDEN / "oracle_token.sha256").read_text().strip())
        for folder in (TASK / "environment",):
            for p in folder.rglob("*"):
                if p.is_file() and p.suffix in (".py", ".json", ".sh", ".md", ".txt", "") and p.stat().st_size < 2_000_000:
                    text = p.read_text(errors="replace")
                    self.assertNotIn(token, text)
                    for s in self.set:
                        self.assertNotIn(str(s), text, "%s holds a graded seed" % p)
        for d in ("environment", "tests"):          # no instruction of an image names the hidden folder or the token
            code = [line for line in (TASK / d / "Dockerfile").read_text().splitlines() if not line.lstrip().startswith("#")]
            self.assertFalse([line for line in code if "hidden" in line or "oracle_token" in line])

    def test_grader_log_names_no_seed(self):
        src = (TASK / "tests/verify.py").read_text()
        for node in ast.walk(ast.parse(src)):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in ("log", "fail"):
                names = {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}
                self.assertNotIn("seed", names, "a log or fail call formats a seed (line %d)" % node.lineno)


class Statements(unittest.TestCase):
    def test_instruction_states_the_codes_numbers(self):
        c = constants(TASK / "tests/harness_env.py", (
            "MAX_STEPS", "MAX_DPOS", "MAX_DROT", "TENSION_LIMIT", "TENSION_WINDOW", "MOUNT_TOL", "POST_TOL", "SEATED_Z",
            "POST_R", "POST_H", "POST_CAP_R", "PIN_H", "PIN_R", "BOOT_STIFFNESS", "BOOT_TRAVEL", "GRIP_FORCE",
            "GROOVE_BELOW_SITE", "BOARD_T", "WRENCH_NOISE", "OVERHEAD_POS", "SETTLE_S", "CLIP_GAP_OVER", "LEAD_LEN"))
        self.assertEqual((c["MAX_STEPS"], c["MAX_DPOS"], c["MAX_DROT"]), (1500, 0.03, 0.2))
        self.assertIn("Without it the episode ends after 1500 steps (150 s)", INSTRUCTION)
        self.assertIn("at most 0.03 m long", INSTRUCTION)
        self.assertIn("at most 0.2 rad", INSTRUCTION)
        self.assertEqual((c["TENSION_LIMIT"], c["TENSION_WINDOW"]), (40.0, 0.1))
        self.assertIn("averaged over any 0.1 s, exceeds 40 N", INSTRUCTION)
        self.assertEqual((c["MOUNT_TOL"], c["POST_TOL"], c["SEATED_Z"]), (0.020, 0.010, 0.013))
        self.assertIn("mounted within 20 mm of their printed places", INSTRUCTION)
        self.assertIn("within 10 mm of the post's surface", INSTRUCTION)
        self.assertIn("lower than 13 mm above the board", INSTRUCTION)
        self.assertEqual((c["POST_R"], c["POST_H"], c["POST_CAP_R"], c["PIN_R"], c["PIN_H"]), (0.012, 0.030, 0.018, 0.002, 0.020))
        self.assertIn("12 mm radius and 30 mm height with a black cap of 18 mm radius", INSTRUCTION)
        self.assertIn("two steel pins 4 mm thick and 20 mm high", INSTRUCTION)
        self.assertEqual(c["CLIP_GAP_OVER"], (0.0043, 0.0047))
        self.assertIn("4.3 to 4.7 mm wider than the cable", INSTRUCTION)
        self.assertEqual((c["BOOT_STIFFNESS"], c["BOOT_TRAVEL"], c["GRIP_FORCE"]), (3200.0, 0.007, 60.0))
        self.assertIn("The boot gives 7 mm at 3.2 kN/m", INSTRUCTION)
        self.assertIn("Each finger presses with up to 60 N", INSTRUCTION)
        self.assertEqual((c["GROOVE_BELOW_SITE"], c["BOARD_T"]), (0.017, 0.012))
        self.assertIn("17 mm plus the cable's radius above the board's top", INSTRUCTION)
        self.assertIn("its top is `z = 0.012`", INSTRUCTION)
        self.assertEqual(c["WRENCH_NOISE"], (0.3, 0.03))
        self.assertIn("0.3 N and 0.03 N m standard deviation per axis", INSTRUCTION)
        self.assertEqual(c["OVERHEAD_POS"], (0.49, 0.0, 1.05))
        self.assertIn("fixed 1.05 m above the board's centre", INSTRUCTION)
        self.assertEqual(c["SETTLE_S"], 2.0)
        self.assertIn("lifts it 15 cm and waits 2 s", INSTRUCTION)

    def test_instruction_states_the_graders_rules(self):
        self.assertEqual((CONFIG["grader_budget_s"], CONFIG["image_cost_s"], CONFIG["step_timeout_s"]), (3600, 0.2, 30))
        self.assertIn("Your controller has **3600 s of its own time** for the whole set", INSTRUCTION)
        self.assertIn("**0.2 s for every picture you ask for**", INSTRUCTION)
        self.assertIn("within 30 s", INSTRUCTION)
        self.assertEqual(CONFIG["solver_max_processes"], 64)
        self.assertIn("with at most 64 processes of your account at a time", INSTRUCTION)
        self.assertEqual((CONFIG["probe_rate"], CONFIG["probe_pass_fraction"]), (0.01, 0.1))
        self.assertIn("identical in more than 90 % of the pairs that command any motion scores 0", INSTRUCTION)
        self.assertIn("a fixed hidden set of 20 boards with 3 or 4 fixtures each", INSTRUCTION)
        self.assertIn("**1 if the whole route is made in order and the connector was not damaged, otherwise 0**", INSTRUCTION)
        toml = (TASK / "task.toml").read_text()
        self.assertIn("timeout_sec = 7200", toml)
        self.assertIn("You have 2 hours from the start of the session", INSTRUCTION)
        self.assertIn("LIMIT_MIN=120", (PAYLOAD / "session_time.sh").read_text())
        self.assertIn("cpus = 4", toml)
        self.assertIn("this machine has 4 CPUs and 12 GB", INSTRUCTION)

    def test_job_sheet_carries_no_clip_axis(self):
        src = (TASK / "tests/harness_env.py").read_text()
        fn = src[src.index("def job_sheet(ep):"):src.index("def barb_length")]
        self.assertNotIn("axis", fn.replace("to_world_angle(nom[\"connector\"][\"heading\"])", ""))
        self.assertNotIn("clip_axis_tolerance", INSTRUCTION)
        self.assertIn("the axis does not have to be measured", INSTRUCTION)

    def test_stamp_is_roots(self):
        setup = (TASK / "environment/setup/setup.sh").read_text()
        self.assertIn("install -d -o root -g root -m 755 /run/session", setup)
        self.assertIn("chmod 0444 /run/session/start", setup)


if __name__ == "__main__":
    unittest.main()
