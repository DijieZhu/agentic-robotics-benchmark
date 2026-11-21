"""lab_sample_handling: the commissioning service, what the agent holds, and the credit rule's statement.

The agent has no simulator of the bench. Its sandbox runs a commissioning service (environment/commission_service/)
under another account: the grader's own simulator (byte copies) on development units that are not scored units, on
development worklists that are not scored episodes, for a stated window of bench time, answering the public
observation only: no ground truth, no credit, no word on how a worklist went. The agent's home holds the
equipment's sheets, the service's client and a runner; none of them holds the simulator or says how to do the job.

Standard library only (the host-side suite has no numpy): the service is read as text and a syntax tree, not run.
"""

import ast
import hashlib
import json
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / "tasks/arm/lab_sample_handling"
SERVICE = TASK / "environment/commission_service"
PAYLOAD = TASK / "environment/setup/payload"
BENCH = json.loads((SERVICE / "bench.json").read_text())
INSTRUCTION = (TASK / "instruction.md").read_text()
SHEETS = (PAYLOAD / "task_card.md").read_text()
SIMULATOR = ("lab_sim.py", "lab_env.py", "box_transport_env.py", "lab_render.py", "lab_look.json")


def constants(path, names):
    """Module-level literal constants of a Python file (also from tuple assignments)."""
    out = {}
    for node in ast.parse(path.read_text()).body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id in names:
                out[target.id] = ast.literal_eval(node.value)
            elif isinstance(target, ast.Tuple) and isinstance(node.value, ast.Tuple):
                for t, v in zip(target.elts, node.value.elts):
                    if isinstance(t, ast.Name) and t.id in names:
                        out[t.id] = ast.literal_eval(v)
    return out


def flat(text):
    return " ".join(text.split())


class CommissioningService(unittest.TestCase):
    def test_runs_the_graders_simulator(self):
        for name in SIMULATOR:
            grader = (TASK / "tests" / name).read_bytes()
            self.assertEqual((SERVICE / name).read_bytes(), grader, "commission_service/%s differs from tests/" % name)
            self.assertEqual((TASK / "solution/oracle" / name).read_bytes(), grader, "solution/oracle/%s differs" % name)
        self.assertEqual((TASK / "solution/oracle/episode.py").read_bytes(), (TASK / "tests/episode.py").read_bytes())
        files = sorted(p.relative_to(SERVICE).as_posix() for p in SERVICE.rglob("*")
                       if p.is_file() and "__pycache__" not in p.parts)
        self.assertEqual(files, ["bench.json", "box_transport_env.py", "commissiond.py", "lab_env.py", "lab_look.json",
                                 "lab_render.py", "lab_sim.py"],
                         "the service's folder holds the daemon, the simulator and the units; never the credit")

    def test_development_units_are_not_scored_units(self):
        dev = BENCH["units"]
        scored = json.loads((TASK / "tests/units.json").read_text())["units"]
        self.assertEqual(sorted(dev), ["d0", "d1", "d2"])
        self.assertFalse(set(dev) & set(scored))
        c = constants(TASK / "tests/lab_sim.py", ("UNIT_RANGES", "TURN_PROPS", "FIT_PROPS", "CALIB", "CALIB_CAMS"))
        ranges = c["UNIT_RANGES"]
        self.assertEqual(sorted(ranges), sorted(c["TURN_PROPS"] + c["FIT_PROPS"]))
        self.assertEqual({k: list(ranges[k]) for k in c["TURN_PROPS"]}, BENCH["draw"]["ranges"])
        self.assertEqual({k: list(ranges[k]) for k in c["FIT_PROPS"]}, BENCH["fit_draw"]["ranges"])
        for uid, props in list(dev.items()) + list(scored.items()):
            self.assertEqual(sorted(props), sorted(list(ranges) + ["calib"]))
            for k, (lo, hi) in ranges.items():
                self.assertTrue(lo <= props[k] <= hi, "%s.%s is outside the stated range" % (uid, k))
            self.assertEqual(sorted(props["calib"]), sorted(c["CALIB_CAMS"]))
            for cam, err in props["calib"].items():
                mm, deg = c["CALIB"]["wrist" if cam == "wrist" else "fixed"]
                self.assertEqual(len(err), 6)
                self.assertTrue(all(abs(x) <= deg for x in err[:3]) and all(abs(x) <= mm for x in err[3:]),
                                "%s.calib.%s is outside the stated bound" % (uid, cam))
        # no bench has a scored unit's fit, and the scored units' values are the record's (tests/review/unit_fit.py)
        fit = json.loads((TASK / "tests/hidden/selection_fit.json").read_text())
        for uid, props in scored.items():
            self.assertTrue(fit["recheck"][uid]["kept"])
            self.assertEqual({k: props[k] for k in list(c["FIT_PROPS"]) + ["calib"]}, fit["values"]["scored"][uid])
        for uid, props in dev.items():
            self.assertEqual({k: props[k] for k in list(c["FIT_PROPS"]) + ["calib"]}, fit["values"]["bench"][uid])
            for s in scored.values():
                self.assertFalse(all(abs(props[k] - s[k]) <= 0.02 * (ranges[k][1] - ranges[k][0])
                                     for k in c["FIT_PROPS"]))
        self.assertEqual((TASK / "tests/units.json").read_text(), (TASK / "solution/oracle/units.json").read_text())
        # the three strata of every property are covered (a Latin hypercube: the units span the stated ranges)
        for k, (lo, hi) in ranges.items():
            strata = sorted(min(2, int(3 * (dev[u][k] - lo) / (hi - lo))) for u in dev)
            self.assertEqual(strata, [0, 1, 2], k)

    def test_development_episodes_are_not_scored_episodes(self):
        src = (SERVICE / "commissiond.py").read_text()
        fn = [n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == "episode_seed"]
        self.assertEqual(len(fn), 1)
        ns = {"hashlib": hashlib}
        exec(compile(ast.Module(body=fn, type_ignores=[]), "episode_seed", "exec"), ns)
        seed = ns["episode_seed"]
        self.assertEqual(seed("d0", 0), int.from_bytes(hashlib.blake2b(
            b"lab_sample_handling commissioning|d0|0", digest_size=8).digest(), "little"))
        self.assertEqual(BENCH["episodes"], 5000)
        self.assertIn("not 0 <= number < EPISODES", src)             # the service stages no other worklist
        self.assertIn("(0 to 4999)", flat(INSTRUCTION))
        dev = set(seed(u, n) for u in BENCH["units"] for n in range(BENCH["episodes"]))
        self.assertEqual(len(dev), BENCH["episodes"] * len(BENCH["units"]))
        hidden = TASK / "tests/hidden"
        graded = set(int(e["seed"]) for e in json.loads((hidden / "episodes.json").read_text())["episodes"])
        selection = json.loads((hidden / "episode_selection.json").read_text())
        candidates = set(int(c["seed"]) for c in selection["candidates"])
        recheck = set(int(s) for s in json.loads((hidden / "selection.json").read_text())["recheck_seeds"])
        recheck |= set(int(s) for s in json.loads((hidden / "selection_fit.json").read_text())["recheck_seeds"])
        numerics = {int(json.loads((TASK / "tests/numerics.json").read_text())["seed"])}
        self.assertEqual(len(graded), 12)
        self.assertFalse(dev & graded, "a development episode is a graded episode")
        self.assertFalse(dev & candidates, "a development episode is a candidate of the hidden set's selection")
        self.assertFalse(dev & recheck, "a development episode is a recheck episode of the units' selection")
        self.assertFalse(dev & numerics)

    def test_answers_no_truth_no_credit_no_debrief(self):
        src = (SERVICE / "commissiond.py").read_text()
        tree = ast.parse(src)
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported |= {a.name for a in node.names}
            elif isinstance(node, ast.ImportFrom):
                imported.add(node.module)
        self.assertEqual(imported, {"fcntl", "hashlib", "json", "math", "os", "resource", "select", "signal", "socket",
                                    "sys", "time", "numpy", "lab_env"})
        used = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
        # what the service touches of a bench: reset, step, the observation and its perturbed copy, the cameras'
        # request, the step count, the fault flag and the lid's angle (the public end rule); never the bench's
        # state, its draw or the credit
        for forbidden in ("truth", "score", "seated", "masses", "fills", "names", "src", "cold", "pre", "foreign",
                          "extra_mass", "episode", "data", "model", "qpos", "qvel", "rotor_q", "rotor_angle",
                          "tube_pos", "target_hole", "scale_raw", "credit", "pairs", "run", "startable"):
            self.assertNotIn(forbidden, used, "the commissioning service reads %s" % forbidden)
        for word in ("import episode", "episode.py\")", "S.seated", "BALANCE", "RUN,", "COLD"):
            self.assertNotIn(word, src)
        public = None
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "PUBLIC" for t in node.targets):
                public = ast.literal_eval(node.value)
        self.assertEqual(public, ("status", "reset", "step", "look", "probe", "close"))
        # the replies' keys: nothing but the observation, the end's public reason and the window's count
        keys = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Dict):
                keys |= {k.value for k in node.keys if isinstance(k, ast.Constant) and isinstance(k.value, str)}
        self.assertEqual(keys, {"id", "result", "error", "detail", "obs", "unit", "episode", "steps_left", "done",
                                "ended", "closed", "window_steps", "steps_used", "resets", "reset_steps",
                                "max_benches", "benches_open", "units", "episodes"})

    def test_the_observation_carries_no_truth(self):
        tree = ast.parse((TASK / "tests/lab_env.py").read_text())
        cls = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "LabEnv"][0]
        for name in ("observation", "perturbed_observation", "step", "reset"):
            fn = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name][0]
            called = {n.func.attr for n in ast.walk(fn) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
            self.assertNotIn("truth", called, "LabEnv.%s calls truth()" % name)

    def test_window_and_end_rules_are_the_stated_ones(self):
        ep = constants(TASK / "tests/episode.py", ("EPISODE_STEPS", "LID_HOLD", "LID_SHUT", "LID_OPENED"))
        self.assertEqual((BENCH["episode_steps"], BENCH["lid_hold"], BENCH["lid_shut"], BENCH["lid_opened"]),
                         (ep["EPISODE_STEPS"], ep["LID_HOLD"], ep["LID_SHUT"], ep["LID_OPENED"]))
        self.assertEqual((BENCH["window_steps"], BENCH["reset_steps"], BENCH["max_benches"], BENCH["silence_s"]),
                         (48000, 300, 2, 120))
        text = flat(INSTRUCTION)
        for words in ("48,000 control steps in all", "80 minutes of bench time at 10 Hz", "Staging a bench costs 300 steps",
                      "Two connections", "a third is answered `busy`", "silent for 120 s is closed",
                      "`commissioning_window_used`", "units `d0`, `d1` and `d2`", "none of them a scored unit",
                      "no credit, and no word on how a worklist went",
                      "It does not end because a worklist is complete",
                      "**At grading** the service is not there"):
            self.assertIn(words, text)
        runner = (PAYLOAD / "commission/run_bench.py").read_text()
        grader = json.loads((TASK / "tests/grader_config.json").read_text())
        self.assertIn("READY_TIMEOUT, STEP_TIMEOUT, MAX_LINE = 120.0, %.1f, 1 << 22" % grader["step_timeout_s"], runner)
        self.assertIn("PROBE_RATE, IMAGE_COST, BUDGET_PER_EPISODE = %s, %s, %.1f / 12" % (
            grader["probe_rate"], grader["image_cost_s"], grader["grader_budget_s"]), runner)

    def test_image_closes_the_service_to_the_agent(self):
        docker = (TASK / "environment/Dockerfile").read_text()
        self.assertIn("--shell /usr/sbin/nologin labbench", docker)
        self.assertIn("COPY commission_service /opt/labbench-commission", docker)
        self.assertIn("chmod -R u+rwX,go-rwx /opt/labbench-commission", docker)
        for probe in ("! runuser -u user -- ls /opt/labbench-commission",
                      "! runuser -u user -- cat /opt/labbench-commission/bench.json",
                      "! runuser -u user -- cat /opt/labbench-commission/lab_sim.py",
                      '[ -z "$(find /opt/labbench-commission ! -user labbench)" ]',
                      '[ -z "$(find /opt/labbench-commission -perm /077)" ]'):
            self.assertIn(probe, docker)
        instructions = [line.split()[0] for line in docker.splitlines()
                        if re.match(r"^(FROM|RUN|COPY|ADD|ENV|WORKDIR|USER|ENTRYPOINT|CMD)\b", line)]
        self.assertEqual(instructions[-1], "WORKDIR")
        self.assertIn("\nWORKDIR /\n", docker)
        setup = (TASK / "environment/setup/setup.sh").read_text()
        self.assertIn("runuser -u labbench --", setup)
        self.assertIn("/usr/bin/env -i", setup)
        self.assertNotIn("cp -a", setup)
        # the verifier's image holds no service: nothing of it is there when a controller is graded
        self.assertNotIn("labbench", (TASK / "tests/Dockerfile").read_text())
        self.assertNotIn("commission", (TASK / "tests/test.sh").read_text())


class AgentSide(unittest.TestCase):
    def test_agent_has_no_simulator(self):
        files = sorted(p.relative_to(PAYLOAD).as_posix() for p in PAYLOAD.rglob("*")
                       if p.is_file() and "__pycache__" not in p.parts)
        self.assertEqual(files, ["NOTICE.md", "commission/__init__.py", "commission/client.py",
                                 "commission/run_bench.py", "helpers/wire_skeleton.py", "session_time.sh",
                                 "task_card.md"])
        for rel in files:
            text = (PAYLOAD / rel).read_text()
            for word in ("import mujoco", "MjSpec", "MjModel", "mj_step", "LabEnv", "lab_sim", "lab_env", "episode.py",
                         "_osc_torque", "solref", "condim", "units.json", "practice", "bench.json"):
                self.assertNotIn(word, text, "%s holds %s" % (rel, word))

    def test_instruction_names_only_what_the_agent_has(self):
        for name in ("task_card.md", "commission/client.py", "commission/run_bench.py", "helpers/wire_skeleton.py",
                     "session_time.sh"):
            self.assertIn(name, INSTRUCTION)
            self.assertTrue((PAYLOAD / name).is_file(), name)
        for gone in ("lab_sim.py", "lab_env.py", "episode.py", "units.json", "practice", "box_transport_env.py"):
            self.assertNotIn(gone, INSTRUCTION)

    def test_session_time_is_the_engines_limit(self):
        """The instruction's session limit, the script's and the engine's agent timeout are one number; the stamp the
        script reads is written by setup.sh into a folder that is root's (the agent's account reads it and can
        neither change nor replace it)."""
        toml = (TASK / "task.toml").read_text()
        agent = toml[toml.index("[agent]"):]
        timeout = int(re.search(r"^timeout_sec\s*=\s*(\d+)", agent, re.M).group(1))
        script = (PAYLOAD / "session_time.sh").read_text()
        limit = int(re.search(r"^LIMIT_MIN=(\d+)$", script, re.M).group(1))
        self.assertEqual(limit * 60, timeout)
        self.assertEqual(timeout, 2 * 3600)
        text = flat(INSTRUCTION)
        for words in ("You have 2 hours of wall-clock time", "This session is stopped 2 hours of wall-clock time after it started",
                      "bash /home/user/session_time.sh", "How much of the 2 hours is left".lower()):
            self.assertIn(words.lower(), text.lower())
        stamp = re.search(r"^STAMP=(\S+)$", script, re.M).group(1)
        self.assertEqual(stamp, "/run/session/start")
        setup = (TASK / "environment/setup/setup.sh").read_text()
        for line in ("install -d -o root -g root -m 755 /run/session", "date -u +%s > /run/session/start",
                     "chown root:root /run/session/start && chmod 0444 /run/session/start"):
            self.assertIn(line, setup)
        self.assertNotIn("HOME_DIR/.session", setup, "a stamp in the agent's home could be replaced by the agent")

    def test_sheets_state_the_series_ranges(self):
        ranges = constants(TASK / "tests/lab_sim.py", ("UNIT_RANGES",))["UNIT_RANGES"]
        lo, hi = ranges["rotor_friction"]
        for text, name in ((flat(INSTRUCTION), "instruction.md"), (flat(SHEETS), "task_card.md")):
            for words in ("%.2f-%.2f N m" % (lo, hi), "%g-%g deg" % ranges["handle_play"],
                          "%.3f-%.3f kg m^2" % ranges["rotor_inertia"],
                          "| %.1f-%.1f mm |" % ranges["bore_clearance"], "| %.2f-%.2f |" % ranges["bore_friction"],
                          "| %.2f-%.2f N m |" % ranges["lid_friction"]):
                self.assertIn(words, text, "%s does not state the range %s" % (name, words))
        calib = constants(TASK / "tests/lab_sim.py", ("CALIB",))["CALIB"]
        self.assertEqual(calib, {"wrist": (2.0, 0.5), "fixed": (5.0, 0.5)})
        for text in (flat(INSTRUCTION), flat(SHEETS)):
            self.assertIn("up to 2 mm", text)
            self.assertIn("up to 5 mm", text)
            self.assertIn("0.5 deg", text)
            self.assertIn("calibration on file", text)

    def test_payload_states_equipment_not_a_method(self):
        """What the agent is given says what the equipment is and what its sensors read. It does not say how to do
        the job nor how to spend the commissioning window."""
        texts = {"instruction.md": INSTRUCTION}
        for p in sorted(PAYLOAD.rglob("*")):
            if p.is_file() and "__pycache__" not in p.parts and p.name != "wire_skeleton.py":
                texts[p.relative_to(PAYLOAD).as_posix()] = p.read_text()
        for name, text in texts.items():
            low = flat(text).lower()
            for words in ("you should", "we recommend", "recommended", "a good way", "best way", "strategy",
                          "plan your", "spend the window", "spend your window", "first weigh", "weigh every",
                          "weigh each", "tip:", "hint", "step 1", "how to turn", "re-grip", "regrip",
                          "tune", "calibrate your", "identify the unit by", "estimate the play by"):
                self.assertFalse(re.search(r"(?<![a-z])%s(?![a-z])" % re.escape(words), low),
                                 "%s says %r" % (name, words))


class Credit(unittest.TestCase):
    def test_the_rule_is_stated(self):
        c = constants(TASK / "tests/episode.py", ("RUN", "COLD", "BALANCE_G", "LID_SHUT"))
        self.assertEqual((c["RUN"], c["COLD"], c["BALANCE_G"], c["LID_SHUT"]), (0.8, 0.2, 1.0, 0.02))
        text = flat(INSTRUCTION)
        for words in ("The run counts only if the centrifuge could be started on it",
                      "the lid is shut (its angle under 0.02 rad)",
                      "every tube seated in the rotor is half of a balanced pair",
                      "the rotor holds at least one balanced pair",
                      "the balanced pairs in the rotor divided by the pairs the worklist needs",
                      "The credit is the share times (0.8 + 0.2 x the fraction of the cold tubes seated in their named "
                      "cold-rack positions).",
                      "A run that does not count scores 0",
                      "within 6 mm of the hole's floor",
                      "their masses within 1 g",
                      "If a tube that was not yours to move is no longer seated where it started, the episode scores 0."):
            self.assertIn(words, text)
        src = (TASK / "tests/episode.py").read_text()
        self.assertIn("credit = run * (RUN + COLD * sum(cold.values()) / len(cold))", src)
        self.assertIn("startable = bool(shut and balanced and n_pairs >= 1)", src)
        self.assertIn("run = n_pairs / need if startable else 0.0", src)
        seated = (TASK / "tests/lab_sim.py").read_text()
        self.assertIn('depth_in > h.get("depth", HOLE_DEPTH) - 0.006', seated)


class Grader(unittest.TestCase):
    def test_the_account_is_cleared_before_the_check_and_outside_the_clock(self):
        """The kit's clearing takes seconds in an image with a desktop session. It runs before the empty-account
        check and before the controller's clock exists, so it is never charged to the controller's time budget
        (Solver.__init__ calls it again inside the timed start and gets the first call's report back)."""
        src = (TASK / "tests/verify.py").read_text()
        main = src[src.index("\ndef main():"):]
        clear, kill, check = (main.index("cleared = clear_agent_account()"), main.index("kill_agent_processes(uid)"),
                              main.index("require_empty_account(uid)"))
        self.assertTrue(clear < kill < check < main.index("check_numerics()"))
        run = src[src.index("\ndef run_episodes("):src.index("\ndef main():")]
        self.assertIn("clock.timed(GuardedSolver", run)
        self.assertNotIn("clear_agent_account", run)
        self.assertNotIn("clear_account=False", src)                  # the kit's own call stays on


    def test_a_controllers_arithmetic_is_pinned_in_both_images_and_checked(self):
        """The same submission must score the same on every grading host: both images carry one pin (NumPy's
        dispatch groups by this numpy's names, BLAS on one thread, OpenCV's IPP), the instruction names it, and the
        numerics check hashes a controller's math beside the simulation's."""
        agent, grader = (TASK / "environment/Dockerfile").read_text(), (TASK / "tests/Dockerfile").read_text()
        pin = agent[agent.index("ENV OPENBLAS_CORETYPE=Haswell"):agent.index("[image] numerics pin OK")]
        self.assertIn(pin, grader)
        for line in ('os.environ.setdefault("NPY_DISABLE_CPU_FEATURES", "X86_V4 AVX512_ICL AVX512_SPR")',
                     'os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")', 'os.environ.setdefault("OMP_NUM_THREADS", "1")',
                     'os.environ.setdefault("OPENCV_IPP", "avx2")', "-W error::ImportWarning",
                     "set(e['NPY_DISABLE_CPU_FEATURES'].split()) == set(D) - {'X86_V3'}"):
            self.assertIn(line, pin)
        self.assertNotIn('"AVX512_SKX"', pin)                         # numpy 2.5 rejects the old names and disables nothing
        text = flat(INSTRUCTION)
        for words in ("`OPENBLAS_CORETYPE=Haswell`", "`OPENBLAS_NUM_THREADS=1`", "`NPY_DISABLE_CPU_FEATURES`",
                      "`OPENCV_IPP=avx2`", "A variable that you set yourself is kept"):
            self.assertIn(words, text)
        ref = json.loads((TASK / "tests/numerics.json").read_text())
        self.assertEqual(set(ref), {"fingerprint", "simulation", "math", "steps", "seed", "what"})
        self.assertTrue({"cv2", "scipy", "big", "trig_float64", "exp_float32", "lin_781"} <= set(ref["math"]))
        h = hashlib.sha256()
        h.update(ref["simulation"].encode())
        h.update(json.dumps(ref["math"], sort_keys=True).encode())
        self.assertEqual(h.hexdigest(), ref["fingerprint"])
        check = (TASK / "tests/numerics_check.py").read_text()
        self.assertIn('CONTROLLER_ENV = {"PATH": "/opt/venv/bin:/usr/local/bin:/usr/bin:/bin", "HOME": "/tmp", "LANG": "C.UTF-8"}', check)
        self.assertIn('"PATH=/opt/venv/bin:/usr/local/bin:/usr/bin:/bin", "HOME=/home/user", "LANG=C.UTF-8"',
                      (TASK / "tests/verify.py").read_text())


if __name__ == "__main__":
    unittest.main()
