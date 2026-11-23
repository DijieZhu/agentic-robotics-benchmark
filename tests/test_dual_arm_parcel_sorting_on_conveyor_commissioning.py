"""dual_arm_parcel_sorting_on_conveyor: the commissioning service, the run regime copies and the fixed hidden set.

The agent's sandbox holds a commissioning service (environment/commission_service/): the station's cell on development
shifts, for the agent's commissioning runs. It must run the grader's own cell (a byte copy of tests/parcelcell/, less
the client) through the grader's own wire loop (a byte copy of tests/cellkit/wire.py), on shifts that are neither
graded shifts nor candidates of the hidden set's selection, under the graded shift's budgets and safety rules, and
never answer ground truth, a score or a debrief. The given client is the one the grader launches programs with, and
~/commission/run_shift.py runs a program under the grader's regime. The rated set (the privileged oracle's) is the
one its latest recorded selection kept; the line set (every other submission's) is drawn by its registered rule;
neither set may be a placeholder.

Standard library only (the host-side suite has no numpy): the service is read as text and a syntax tree, not run.
"""

import ast
import hashlib
import json
import os
import unittest
from pathlib import Path

TASK = Path(os.environ.get("ARB_TASK_DIR") or (Path(__file__).resolve().parents[1] / "tasks/arm/dual_arm_parcel_sorting_on_conveyor"))
SERVICE = TASK / "environment/commission_service"
PAYLOAD = TASK / "environment/setup/payload"
SITES = json.loads((SERVICE / "sites.json").read_text())
SELECTION = json.loads((TASK / "tests/hidden/selection.json").read_text())
SEEDS = json.loads((TASK / "tests/hidden/seeds.json").read_text())
LINE = json.loads((TASK / "tests/hidden/line/seeds.json").read_text())
GRADER = json.loads((TASK / "tests/grader_config.json").read_text())
PUBLIC_CFG = PAYLOAD / "given/cell/cell_config.json"
CELL_BLOCKS = ("world", "cameras", "sensors", "look", "budgets", "safety", "score", "line")
NO_PIN = ("OPENBLAS_CORETYPE", "OPENBLAS_L2_SIZE", "NPY_DISABLE_CPU_FEATURES")
PINS = 'OPENBLAS_CORETYPE=Haswell OPENBLAS_L2_SIZE=256 NPY_DISABLE_CPU_FEATURES="X86_V4 AVX512_ICL AVX512_SPR"'


def files(root, skip=()):
    out = {}
    for dp, dns, fns in os.walk(root):
        dns[:] = [d for d in dns if d != "__pycache__"]
        for fn in fns:
            rel = os.path.relpath(os.path.join(dp, fn), root)
            if rel not in skip:
                out[rel] = Path(dp, fn).read_bytes()
    return out


def sha(b):
    return hashlib.sha256(b).hexdigest()


def assigned(tree, name):
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise KeyError(name)


class CommissioningService(unittest.TestCase):
    def test_runs_the_graders_cell_and_wire(self):
        mine = files(SERVICE / "parcelcell")
        graders = files(TASK / "tests/parcelcell", skip=("client.py",))
        self.assertEqual(sorted(mine), sorted(graders), "the service's cell files differ in name from the grader's")
        for rel, data in graders.items():
            self.assertEqual(mine[rel], data, "commission_service/parcelcell/%s differs from tests/parcelcell/%s" % (rel, rel))
        self.assertNotIn("client.py", mine)
        for name in ("__init__.py", "wire.py"):
            self.assertEqual((SERVICE / "cellkit" / name).read_bytes(), (TASK / "tests/cellkit" / name).read_bytes(),
                             "commission_service/cellkit/%s differs from tests/cellkit/%s" % (name, name))
        self.assertEqual(sorted(p.name for p in (SERVICE / "cellkit").iterdir() if p.name != "__pycache__"), ["__init__.py", "wire.py"])

    def test_development_shifts_are_not_graded_shifts(self):
        dev = [int(s) for s in SITES["seeds"]]
        self.assertEqual(len(set(dev)), len(dev))
        self.assertEqual(dev, [int(s["seed"]) for s in SITES["sites"]])
        graded = set(int(s) for s in SEEDS["seeds"]) | set(int(s["seed"]) for s in SEEDS.get("shifts", [])) | \
            set(int(s) for s in LINE["seeds"]) | set(int(s["seed"]) for s in LINE.get("shifts", []))
        def seed_of(c):
            return int(c["seed"]) if isinstance(c, dict) else int(c)
        candidates = set(seed_of(s) for s in SELECTION.get("candidates") or [])
        for a in SELECTION.get("amendments") or []:
            candidates |= set(seed_of(s) for s in a.get("candidates", []))
        self.assertFalse(set(dev) & graded, "a development shift is a graded shift")
        self.assertFalse(set(dev) & candidates, "a development shift is a candidate of the hidden set's selection")
        hidden_sha = set(s["sha256"] for s in SEEDS.get("shifts", []) + LINE.get("shifts", []))
        for s in SITES["sites"]:
            raw = (SERVICE / s["package"]).read_bytes()
            self.assertEqual(sha(raw), s["sha256"], s["package"])
            self.assertNotIn(s["sha256"], hidden_sha)
            meta = json.loads(raw)["meta"]
            self.assertEqual((meta["label"], int(meta["seed"]), int(meta["k"])), ("line-dev", int(s["seed"]), int(s["k"])))
        for root, doc in ((TASK / "tests/hidden", SEEDS), (TASK / "tests/hidden/line", LINE)):
            for s in doc.get("shifts", []):
                meta = json.loads((root / s["package"]).read_bytes())["meta"]
                self.assertNotIn(meta.get("label"), ("dev", "line-dev"))

    def test_budgets_and_safety_are_the_graded_shifts(self):
        cfg = SITES["config"]
        for block in CELL_BLOCKS:
            self.assertEqual(cfg.get(block), GRADER.get(block), "sites.json config.%s differs from grader_config.json" % block)
        self.assertEqual((SITES["shifts"], SITES["open"]), (12, 2))
        if PUBLIC_CFG.is_file():
            pub = json.loads(PUBLIC_CFG.read_text())
            c = pub["commissioning"]
            self.assertEqual((c["development_shifts"], c["shifts"], c["open_at_a_time"]),
                             (len(SITES["sites"]), SITES["shifts"], SITES["open"]))
            if "safety" in pub:
                self.assertEqual(pub["safety"], GRADER["safety"])
        else:
            self.fail("environment/setup/payload/given/cell/cell_config.json is missing")

    def test_answers_no_truth_score_or_debrief(self):
        src = (SERVICE / "commissiond.py").read_text()
        tree = ast.parse(src)
        called = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                f = node.func
                called.add(f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", ""))
        for forbidden in ("truth", "finalize", "summary", "get_state_privileged", "reference_run", "log_lines", "so_far",
                          "parcel_states", "parcel_state", "contacts", "crossings", "shift_features", "draw_shift"):
            self.assertNotIn(forbidden, called, "the commissioning service calls %s()" % forbidden)
        for word in ("truth", "privileged=True", "allow_reset", "mode=\"practice\""):
            self.assertNotIn(word, src.replace("no ground truth", "").replace("no truth", ""), word)
        self.assertIn('mode="hidden", privileged=False', src)
        public = assigned(tree, "PUBLIC")
        self.assertEqual(public, ("describe", "get_state", "step", "get_camera", "finish"))
        server = ast.parse((TASK / "tests/parcelcell/server.py").read_text())
        self.assertEqual(public, assigned(server, "PUBLIC_METHODS"))
        self.assertIn("wire.serve_episode(", src)

    def test_client_copies(self):
        given = (PAYLOAD / "given/lib/parcelcell/client.py").read_bytes()
        self.assertEqual(given, (TASK / "tests/parcelcell/client.py").read_bytes())
        self.assertEqual(given, (TASK / "tests/setup_payload/given/lib/parcelcell/client.py").read_bytes())
        self.assertEqual(sorted(p.name for p in (PAYLOAD / "given/lib/parcelcell").iterdir() if p.name != "__pycache__"),
                         ["__init__.py", "client.py"], "the given library holds the client only")

    def test_setup_payload_is_a_byte_copy_and_hashed(self):
        a = files(PAYLOAD)
        b = files(TASK / "tests/setup_payload")
        self.assertEqual(sorted(a), sorted(b))
        for rel in a:
            self.assertEqual(a[rel], b[rel], rel)
        gh = json.loads((TASK / "tests/given_hashes.json").read_text())["files"]
        for rel, data in a.items():
            self.assertEqual(gh.get(rel), sha(data), "given_hashes.json: %s" % rel)
        # the image's task assets under given/ (environment/assets.sha256) reach the agent home and /opt/arb too
        fetched = {}
        for line in (TASK / "environment/assets.sha256").read_text().splitlines():
            if line.strip() and not line.startswith("#"):
                h, rel = line.split(None, 1)
                if rel.strip().startswith("given/"):
                    fetched[rel.strip()] = h
        self.assertFalse(set(fetched) & set(a), "a given file is both in the setup payload and in the task assets")
        for rel, h in fetched.items():
            self.assertEqual(gh.get(rel), h, "given_hashes.json: %s (task asset)" % rel)
        self.assertEqual(set(gh), set(a) | set(fetched))


class RunRegime(unittest.TestCase):
    def test_run_shift_runs_the_program_as_the_grader_does(self):
        rs = ast.parse((PAYLOAD / "commission/run_shift.py").read_text())
        nonet = ast.parse((TASK / "tests/cellkit/nonet.py").read_text())
        self.assertEqual(assigned(rs, "NONET_SRC"), assigned(nonet, "NONET_SRC"), "run_shift's no-network filter differs")
        verify = ast.parse((TASK / "tests/verify.py").read_text())
        self.assertEqual(assigned(rs, "OOM_FIRST"), assigned(verify, "OOM_FIRST"))
        self.assertEqual(float(assigned(rs, "CPU_FACTOR")), float(GRADER["cpu_cap_factor"]))
        src = (PAYLOAD / "commission/run_shift.py").read_text()
        for key in ("HOME", "TMPDIR", "TMP", "TEMP", "LANG", "MPLBACKEND", "PYTHONPATH", "ARB_GIVEN_DIR", "MUJOCO_GL",
                    "PYOPENGL_PLATFORM", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "LP_NUM_THREADS"):
            self.assertIn('"%s"' % key, src, key)
        for pin in NO_PIN:
            self.assertNotIn(pin, src, "run_shift pins %s for the program; the grader does not" % pin)
        runner = (TASK / "tests/cellkit/runner.py").read_text()
        for pin in NO_PIN:
            self.assertNotIn(pin, runner, "the program's environment carries %s" % pin)
        self.assertIn("launcher_argv(", src)

    def test_numerics_pins_in_every_grader_process(self):
        self.assertIn("export " + PINS, (TASK / "tests/test.sh").read_text())
        setup = (TASK / "environment/setup/setup.sh").read_text()
        self.assertIn(PINS, " ".join(setup.split()))
        docker = (TASK / "environment/Dockerfile").read_text()
        env_lines = [l for l in docker.splitlines() if l.startswith("ENV ")]
        for pin in NO_PIN:
            self.assertFalse(any(pin in l for l in env_lines), "the image's ENV sets %s" % pin)


class Reference(unittest.TestCase):
    """The reference program (tests/reference) is reached only through the oracle token's privileged path."""

    def test_reference_reached_only_through_the_token(self):
        ref = TASK / "tests/reference"
        ctl = ast.parse((ref / "controller.py").read_text())
        self.assertTrue(any(isinstance(n, ast.FunctionDef) and n.name == "run" for n in ctl.body))
        self.assertTrue((ref / "refctl").is_dir())
        src = (TASK / "tests/verify.py").read_text()
        tree = ast.parse(src)
        builds = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "ParcelCellServer"]
        self.assertEqual(len(builds), 1, "verify.py builds a cell in more than one place")
        self.assertIn('reference_dir=os.path.join(STAGE_DIR, "reference") if privileged else None', src)
        self.assertIn("privileged, mismatch = token_state()", src)
        server = ast.parse((TASK / "tests/parcelcell/server.py").read_text())
        self.assertEqual(assigned(server, "PRIVILEGED_METHODS"), ("get_state_privileged", "reference_run"))
        sh = (TASK / "tests/test.sh").read_text()
        self.assertLess(sh.index('chmod 700 "$STAGE"'), sh.index('exec "${ALE_VERIFY_PYTHON:-python3}"'))

    def test_the_agent_sandbox_holds_nothing_of_it(self):
        ref = TASK / "tests/reference"
        ref_sha = set(sha(b) for b in files(ref).values())
        token = (TASK / "solution/reference/oracle_token").read_bytes().strip()
        self.assertEqual((TASK / "tests/hidden/oracle_token.sha256").read_text().strip(), sha(token))
        for rel, data in files(TASK / "environment").items():
            self.assertNotIn(sha(data), ref_sha, "environment/%s is a file of the reference" % rel)
            self.assertNotIn(token, data, "environment/%s holds the oracle token" % rel)
            self.assertNotIn("refctl", rel)
        for line in (TASK / "environment/Dockerfile").read_text().splitlines():
            if line.startswith(("COPY", "ADD")):
                for word in ("tests", "solution", "reference"):
                    self.assertNotIn(word, line, line)
        svc = (SERVICE / "commissiond.py").read_text()
        self.assertNotIn("reference_dir", svc)


class HiddenSet(unittest.TestCase):
    def test_the_set_is_the_latest_selections(self):
        self.assertIsNot(SEEDS.get("placeholder"), True, "tests/hidden/seeds.json is still the PLACEHOLDER set")
        runs = SELECTION["selection_run"]
        latest = runs
        for a in SELECTION.get("amendments", []):
            if "selection_run" in a:
                latest = a["selection_run"]
        kept = latest["hidden_set" if "hidden_set" in latest else "kept"]
        n = int(GRADER["shifts_graded"])
        self.assertEqual([int(s) for s in SEEDS["seeds"]][:n], [int(s) for s in kept][:n])
        self.assertGreaterEqual(len(SEEDS["shifts"]), n)
        for s in SEEDS["shifts"][:n]:
            self.assertEqual(sha((TASK / "tests/hidden" / s["package"]).read_bytes()), s["sha256"])


class LineSet(unittest.TestCase):
    def test_the_line_set_is_drawn_by_its_rule(self):
        self.assertIsNot(LINE.get("placeholder"), True, "tests/hidden/line/seeds.json is still the PLACEHOLDER set")
        n = int(GRADER["shifts_graded"])
        self.assertGreaterEqual(len(LINE["shifts"]), n)
        self.assertTrue((TASK / "tests/hidden/line/selection_rule.txt").is_file())

    def test_every_line_package_is_a_line_shift_and_pinned(self):
        for s in LINE["shifts"]:
            raw = (TASK / "tests/hidden/line" / s["package"]).read_bytes()
            self.assertEqual(sha(raw), s["sha256"])
            self.assertEqual(json.loads(raw)["setting"]["name"], "line")
        for s in SEEDS["shifts"]:
            self.assertNotIn("setting", json.loads((TASK / "tests/hidden" / s["package"]).read_bytes()))

    def test_only_the_token_selects_the_rated_set(self):
        src = (TASK / "tests/verify.py").read_text()
        tree = ast.parse(src)
        fn = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "graded_set"]
        self.assertEqual(len(fn), 1)
        self.assertIn('return ("rated", HIDDEN) if privileged is True else ("line", HIDDEN_LINE)', src)
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "graded_set"]
        self.assertEqual(len(calls), 1)
        self.assertIn("setting, root = graded_set(privileged)", src)


class Frozen(unittest.TestCase):
    def test_generator_and_scoring_config_are_frozen(self):
        """tests/frozen.json pins the files that define a shift and its score, the cell and the reference (the line
        set is drawn on them; the rated set was selected on the cell recorded in its block rated_set and carried to
        this cell by its identity record); tests/review/freeze.py recomputes them (stdlib only)."""
        import importlib.util
        spec = importlib.util.spec_from_file_location("parcel_freeze", str(TASK / "tests/review/freeze.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        self.assertEqual(mod.problems(str(TASK / "tests")), [])
        frozen = json.loads((TASK / "tests/frozen.json").read_text())
        self.assertEqual(frozen["versions"]["generator"], SELECTION.get("generator", frozen["versions"]["generator"]))
        rated = frozen["rated_set"]
        self.assertEqual(rated["selected_on_cell"], SELECTION.get("frozen") or {})
        self.assertTrue(rated["identity"]["identical"])
        self.assertEqual(sorted(rated["identity"]["packages"]), sorted(s["package"].split("/")[-1] for s in SEEDS["shifts"]))


class HiddenValuesStayHidden(unittest.TestCase):
    def test_no_hidden_shift_value_in_what_the_agent_gets(self):
        """No hidden shift's seed, package digest or the selection's salt appears anywhere in the agent's sandbox
        (environment/: the setup payload, the commissioning service, the build context) or in the instruction, and no
        hidden shift's belt speed appears as a number in the agent's own files (the setup payload) or the
        instruction."""
        import re
        every = {rel: data for rel, data in files(TASK / "environment").items()}
        every["instruction.md"] = (TASK / "instruction.md").read_bytes()
        mine = {rel: data for rel, data in files(PAYLOAD).items()}
        mine["instruction.md"] = every["instruction.md"]
        exact, speeds = [], []
        if SELECTION.get("salt_hex"):
            exact.append(SELECTION["salt_hex"])
        for root, doc in ((TASK / "tests/hidden", SEEDS), (TASK / "tests/hidden/line", LINE)):
            for s in doc.get("shifts", []):
                pkg = json.loads((root / s["package"]).read_bytes())
                exact += [str(int(s["seed"])), s["sha256"]]
                speeds += [repr(float(seg[1])) for seg in pkg["belt_speed"]]
        for rel, data in every.items():
            for n in exact:
                self.assertNotIn(n.encode(), data, "%s holds the hidden value %s" % (rel, n))
        for rel, data in mine.items():
            for v in speeds:
                pat = re.compile((r"(?<![\d.])%s(?!\d)" % re.escape(v)).encode())
                self.assertIsNone(pat.search(data), "%s holds a hidden shift's belt speed %s" % (rel, v))


class EngineersInformation(unittest.TestCase):
    """The agent gets what an engineer gets: the robot's model files carry the maker's nominal dynamics and no servo
    gain, the cameras' poses are the installation drawing's (the installed poses live only on the grader's side and in
    the commissioning service, which the agent account cannot read), the torque readings are noisy."""

    def _mine(self):
        mine = {rel: data for rel, data in files(PAYLOAD).items() if not rel.lower().endswith((".stl", ".ttf"))}
        mine["instruction.md"] = (TASK / "instruction.md").read_bytes()
        return mine

    def test_the_given_models_hold_no_true_mass_and_no_gain(self):
        import re
        true = (TASK / "tests/parcelcell/assets/robot/dual_arm.xml").read_text()
        masses = re.findall(r'<inertial[^>]*\bmass="([0-9.eE+-]+)"', true)
        self.assertGreater(len(masses), 10)
        gains = set(re.findall(r'\bkp="([0-9.]+)"', true)) | set(re.findall(r'\bkv="([0-9.]+)"', true))
        self.assertTrue(gains)
        for rel, data in self._mine().items():
            for m in masses:
                for fmt in ("%.6g", "%.5g", "%.4g"):
                    v = fmt % float(m)
                    if len(v.replace(".", "").lstrip("0")) >= 4:
                        self.assertNotIn(v.encode(), data, "%s holds the true link mass %s" % (rel, v))
            if rel.endswith(".xml"):
                self.assertNotIn(b'kp="', data, rel)
                self.assertNotIn(b'kv="', data, rel)
                self.assertNotIn(b"<position ", data, rel)
        for dp, _dn, fns in os.walk(TASK / "environment"):
            self.assertNotIn("nominal_dynamics.json", fns)
        self.assertTrue((TASK / "tests/review/docs/nominal_dynamics.json").is_file())

    def test_no_installed_camera_pose_in_the_agents_files(self):
        inst = GRADER["installation"]["cameras"]
        self.assertEqual(set(inst), {"infeed", "head", "wrist_left", "wrist_right"})
        vals = [v for e in inst.values() for v in e["d_pos_m"] + e["d_rotvec_rad"] + [e["focal_scale"]]]
        for rel, data in self._mine().items():
            for v in vals:
                self.assertNotIn(repr(v).encode(), data, "%s holds an installation value %r" % (rel, v))
        self.assertEqual(SITES["config"]["installation"], GRADER["installation"])   # the service: the same cameras
        pub = json.loads(PUBLIC_CFG.read_text())
        self.assertNotIn("installation", pub)
        self.assertIn("installation_tolerance", pub["cameras"])
        self.assertEqual(pub["budgets"]["frames_per_shift_s"], GRADER["budgets"]["frames_per_shift_s"])
        self.assertEqual(pub["sensors"]["tau_noise_rel"], GRADER["sensors"]["tau_noise_rel"])


if __name__ == "__main__":
    unittest.main()
