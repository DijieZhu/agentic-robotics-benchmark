"""keyed_nut_peg_assembly: the commissioning service, the data sheet, the tolerances and the takt.

The agent's sandbox holds no simulator it can read. It holds the cell's data sheet (environment/setup/payload/:
datasheet.md, cell_data.py, the commissioning client and runner, the wire helper) and a commissioning service
(environment/commission_service/) that runs the grader's own simulator (a byte copy of tests/keyed_env.py) on
development kits, for a fixed number of episodes, under an account of its own. The service must run the graded
episode's limits, on kits that are neither graded kits nor candidates of the graded set's selection, and never answer
ground truth, a score or a completion verdict. The data sheet's numbers are the simulator's, and so are the
tolerances within which every real kit and the cell on the day differ from the drawing: each range the simulator
draws from is stated in the data sheet. The takt is 85 s per kit. The data sheet states the fit and what the wrist
reads in each contact, not how to put a nut on a shaft.

Standard library only (the host-side suite has no numpy or mujoco): the task's Python files are read as text or syntax
trees, never imported.
"""

import ast
import json
import re
import unittest
from pathlib import Path

TASK = Path(__file__).resolve().parents[1] / "tasks/arm/keyed_nut_peg_assembly"
SERVICE = TASK / "environment/commission_service"
PAYLOAD = TASK / "environment/setup/payload"
KITS = json.loads((SERVICE / "kits.json").read_text())
SELECTION = json.loads((TASK / "tests/hidden/selection.json").read_text())
EPISODES = json.loads((TASK / "tests/ref/episodes.json").read_text())["episodes"]
GRADER = json.loads((TASK / "tests/grader_config.json").read_text())
LATEST = SELECTION["amendments"][-1] if SELECTION.get("amendments") else SELECTION
INSTRUCTION = (TASK / "instruction.md").read_text()
DATASHEET = (PAYLOAD / "datasheet.md").read_text()

#: The constants the data sheet's module shares with the simulator: the same name, the same expression.
SHARED = ("NUTS", "MAX_STEPS", "MAX_DPOS", "MAX_DROT", "TABLE_Z", "TABLE_HALF", "ROBOT_BASE",
          "WORKSPACE_LO", "WORKSPACE_HI", "NSEC", "LAYER", "NLAYER", "R_HOLE", "R_GROOVE", "R_NUT", "NUT_H",
          "HANDLE_LEN", "HANDLE_W", "HANDLE_Z", "SHAFT_R", "PEG_H", "COLLAR", "COLLAR_H", "PIN_Z", "PIN_R", "PIN_W",
          "PIN_HH", "TURN_LAYERS", "RISERS", "PEG_REGION", "PEG_MIN_SPACING", "NUT_REGION", "NUT_PEG_MIN_DIST",
          "NUT_MIN_SPACING", "HANDLE_CLEAR", "NUT_YAW_CENTER", "NUT_YAW_SPREAD", "ENGAGE_XY", "GROOVE_TOP",
          "FORCE_ANGLE_TOL", "FORCE_Z_TOL", "FORCE_STEPS", "GRIP_KP", "RGB_NOISE", "WRENCH_NOISE", "WRENCH_BIAS",
          "INSTAB_ROBOT_QVEL", "INSTAB_NUT_VEL", "HOME_Q", "NULLSPACE_Q", "CAMERAS", "WRIST", "CAMERA_NAMES",
          "OBS_KEYS", "NUT_COLOURS", "TOL", "NUT_FRICTION", "NUT_MASS", "PAD_FRICTION_MODEL", "PAD_TORSION_MODEL",
          "PEG_FRICTION", "JOINT_FRICTION")
#: The helpers it shares with the simulator: the same name, the same body.
SHARED_FUNCTIONS = ("look_at", "camera_intrinsics", "fixed_camera_matrices", "wrap_angle", "groove_cells",
                    "wall_boundaries", "level_boundaries", "sample_unit")


def assignments(path):
    """{name: syntax tree of the value} of a module's top-level simple assignments."""
    out = {}
    for node in ast.parse(Path(path).read_text()).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            out[node.targets[0].id] = ast.dump(node.value)
    return out


def functions(path):
    """{name: syntax tree of the body, docstring dropped} of a module's top-level functions."""
    out = {}
    for node in ast.parse(Path(path).read_text()).body:
        if isinstance(node, ast.FunctionDef):
            body = node.body[1:] if ast.get_docstring(node) is not None else node.body
            out[node.name] = (ast.dump(node.args), [ast.dump(n) for n in body])
    return out


class CommissioningService(unittest.TestCase):
    def test_runs_the_graders_cell(self):
        # the cell, its cameras' renderer and the look's numbers: byte copies of the grader's, so that the service
        # and the grader draw the same image of the same state
        for name in ("keyed_env.py", "keyed_render.py", "keyed_look.json"):
            self.assertEqual((SERVICE / name).read_bytes(), (TASK / "tests" / name).read_bytes(),
                             "commission_service/%s differs from tests/%s" % (name, name))
        self.assertEqual(sorted(p.name for p in SERVICE.iterdir() if p.name != "__pycache__"),
                         ["commissiond.py", "keyed_env.py", "keyed_look.json", "keyed_render.py", "kits.json"])

    def test_development_kits_are_not_graded_kits(self):
        kits = KITS["kits"]
        self.assertEqual(len(kits), 16)
        self.assertEqual(len({k["kit_seed"] for k in kits}), 16)
        dev_seeds = {k["kit_seed"] for k in kits} | {k["layout_seed"] for k in kits}
        candidate_seeds = {c["kit_seed"] for c in SELECTION["candidates"]} | {c["layout_seed"] for c in SELECTION["candidates"]}
        graded_seeds = {e["seed"] for e in EPISODES}
        self.assertFalse(dev_seeds & candidate_seeds, "a development seed is a candidate's")
        self.assertFalse(dev_seeds & graded_seeds, "a development seed is a graded episode's")
        graded = [sorted(json.dumps(n["groove"], sort_keys=True) for n in e["kit"]["nuts"]) for e in EPISODES]
        for k in kits:
            nuts = k["kit"]["nuts"]
            self.assertEqual(len(nuts), 3)
            self.assertEqual(sorted(k["kit"]["order"]), [1, 2, 3])
            self.assertEqual([n["id"] for n in nuts], ["%s-%d" % (k["name"], i) for i in range(3)])
            self.assertNotIn(sorted(json.dumps(n["groove"], sort_keys=True) for n in nuts), graded,
                             "%s has a graded kit's grooves" % k["name"])

    def test_development_kits_are_the_pre_registered_ones(self):
        dev = LATEST["development_kits"]
        cands = {c["id"]: c for c in dev["candidates"]}
        seed0, layout0 = dev.get("seed0", 7001), dev.get("layout_seed0", 11001)
        for c in dev["candidates"]:
            self.assertEqual((c["kit_seed"], c["layout_seed"]), (seed0 + c["i"], layout0 + c["i"]))
        names = [k["name"] for k in KITS["kits"]]
        for k in KITS["kits"]:
            self.assertEqual((k["kit_seed"], k["layout_seed"]), (cands[k["name"]]["kit_seed"], cands[k["name"]]["layout_seed"]))
        if "selection_run" in dev:                   # once the reference has run them: the first 16 it completed
            runs = dev["selection_run"]["runs"]
            self.assertEqual([r["id"] for r in runs], [c["id"] for c in dev["candidates"]])
            kept = [r["id"] for r in runs if r["kept"]]
            self.assertEqual(names, kept[:16])
            self.assertEqual(dev["kits"], names)
            for r in runs:
                if r["kept"]:
                    self.assertEqual((r["kit_complete"], r["nuts_assembled"], r["drop_reasons"]), (1, 3, []))
                    self.assertLess(r["steps"], 850)
        else:                                        # pre-registered, not run yet: the first 16 candidates
            self.assertEqual(names, [c["id"] for c in dev["candidates"]][:16])

    def test_window_and_limits(self):
        self.assertEqual((KITS["episodes"], KITS["open"]), (32, 2))
        self.assertEqual(KITS["controller_budget_s"], GRADER["controller_budget_s"])
        self.assertEqual(KITS["image_charge_s"], GRADER["image_charge_s"])
        self.assertEqual(KITS["probe_rate"], GRADER["probe_rate"])
        self.assertGreaterEqual(KITS["silence_s"], GRADER["step_timeout_s"])
        for name, text in (("instruction.md", INSTRUCTION), ("datasheet.md", DATASHEET)):
            flat = " ".join(text.split())
            for words in ("32 episodes", "16 development kits", "at most 2 at a time",
                          "%d s without a request" % KITS["silence_s"]):
                self.assertTrue(words in flat, "%s does not say %r" % (name, words))
        runner = (PAYLOAD / "commission/run_kit.py").read_text()
        self.assertIn("CONTROLLER_BUDGET = %.1f" % GRADER["controller_budget_s"], runner)
        charges = GRADER["image_charge_s"]
        self.assertIn('IMAGE_CHARGE = {"left": %s, "right": %s, "wrist": %s}' % (
            charges["left"], charges["right"], charges["wrist"]), runner)
        self.assertIn("STEP_TIMEOUT = %.1f" % GRADER["step_timeout_s"], runner)

    def test_answers_no_truth_score_or_verdict(self):
        src = (SERVICE / "commissiond.py").read_text()
        tree = ast.parse(src)
        used = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
        # what the service touches of the cell: its episode calls, its step count and limit, its fault, the work
        # order's colours (to name a ruined nut), the cameras; never its state or the grader's credit
        for forbidden in ("nut_state", "pin_in_wall", "on_peg", "peg_poses", "layout", "specs", "cells", "broken",
                          "force_count", "target_peg", "data", "model", "wrench_bias", "credit", "release_count",
                          "episode_score", "nuts_assembled", "progress", "assembled", "stage_now", "ground_truth",
                          "depth_now", "_finger_touch", "unit", "wall_rel", "levels", "pin_mid", "pin_height",
                          "joint_friction", "wrench_drift"):
            self.assertNotIn(forbidden, used, "the commissioning service reads %s" % forbidden)
        for word in ("keyed_credit", "GradedNutEnv", "kit_complete", "_gt"):
            self.assertNotIn(word, src)
        self.assertIn('E.NutEnv(entry["kit"]["nuts"], early_end=True)', src)
        public = None
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "PUBLIC" for t in node.targets):
                public = ast.literal_eval(node.value)
        self.assertEqual(public, ("reset", "step", "close"))

    def test_image_closes_the_service_to_the_agent(self):
        docker = (TASK / "environment/Dockerfile").read_text()
        self.assertIn("COPY commission_service /opt/keyedcell-commission", docker)
        self.assertIn("chmod -R u+rwX,go-rwx /opt/keyedcell-commission", docker)
        for probe in ("! runuser -u user -- ls /opt/keyedcell-commission",
                      "! runuser -u user -- cat /opt/keyedcell-commission/keyed_env.py",
                      "! runuser -u user -- cat /opt/keyedcell-commission/kits.json"):
            self.assertIn(probe, docker)
        instructions = [line.split()[0] for line in docker.splitlines()
                        if re.match(r"^(FROM|RUN|COPY|ADD|ENV|WORKDIR|USER|ENTRYPOINT|CMD)\b", line)]
        self.assertEqual(instructions[-1], "WORKDIR")
        self.assertIn("\nWORKDIR /\n", docker)
        setup = (TASK / "environment/setup/setup.sh").read_text()
        self.assertIn("runuser -u keyedcell --", setup)
        self.assertNotIn("keyed_env", setup)


class AgentSide(unittest.TestCase):
    def test_agent_has_no_simulator(self):
        files = sorted(p.relative_to(PAYLOAD).as_posix() for p in PAYLOAD.rglob("*")
                       if p.is_file() and "__pycache__" not in p.parts)
        self.assertEqual(files, ["cell_data.py", "commission/client.py", "commission/run_kit.py", "datasheet.md",
                                 "helpers/wire_skeleton.py"])
        for rel in files:
            text = (PAYLOAD / rel).read_text()
            for word in ("import mujoco", "MjSpec", "MjModel", "mj_step", "NutEnv", "SCENE_XML", "nut_body_xml",
                         "_osc_torque", "sample_kit", "sample_nut_spec", "sample_layout", "solref", "condim",
                         "keyed_env"):
                self.assertNotIn(word, text, "%s holds %s" % (rel, word))

    def test_data_sheet_numbers_are_the_cells(self):
        cell, sheet = assignments(TASK / "tests/keyed_env.py"), assignments(PAYLOAD / "cell_data.py")
        for name in SHARED:
            self.assertIn(name, cell)
            self.assertEqual(sheet.get(name), cell[name], "cell_data.%s is not the simulator's" % name)
        cell_f, sheet_f = functions(TASK / "tests/keyed_env.py"), functions(PAYLOAD / "cell_data.py")
        for name in SHARED_FUNCTIONS:
            self.assertEqual(sheet_f.get(name), cell_f[name], "cell_data.%s is not the simulator's" % name)

    def test_takt(self):
        cell = (TASK / "tests/keyed_env.py").read_text()
        self.assertIn("\nNUTS = 3 ", cell)
        self.assertIn("\nMAX_STEPS = 850 ", cell)
        for name, text in (("instruction.md", INSTRUCTION), ("datasheet.md", DATASHEET)):
            flat = " ".join(text.split())
            self.assertTrue("850 steps" in flat and "85 s" in flat, "%s does not state the takt" % name)
            self.assertFalse(re.search(r"\b(2400|1200|1000)\b|\b100 s\b", flat), "%s still names an earlier takt" % name)
        self.assertTrue("ends at 850 steps" in " ".join(INSTRUCTION.split()), "the instruction's ending rule")
        self.assertIn("(85 s)", KITS["window"])

    def test_every_tolerance_is_stated(self):
        tree = ast.parse((TASK / "tests/keyed_env.py").read_text())
        consts = {}
        for node in tree.body:
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) \
                    and node.targets[0].id in ("TOL", "NUT_FRICTION", "NUT_MASS"):
                consts[node.targets[0].id] = eval(compile(ast.Expression(node.value), "tol", "eval"), {"__builtins__": {}})
        tol = consts["TOL"]
        sheet = " ".join(DATASHEET[DATASHEET.index("## 11. Tolerances"):].split())
        stated = {
            "wall_deg": "+-%g degree" % tol["wall_deg"],
            "level": "+-%g mm" % (tol["level"] * 1000),
            "hole_r": "| 12.8 mm | +-%g mm" % (tol["hole_r"] * 1000),
            "groove_r": "| 19 mm | +-%g mm" % (tol["groove_r"] * 1000),
            "handle_deg": "+-%g degrees" % tol["handle_deg"],
            "colour": "+-%g |" % tol["colour"],
            "shaft_r": "| 12.3 mm | +-%g mm" % (tol["shaft_r"] * 1000),
            "pin_r": "| 17.2 mm | +-%g mm" % (tol["pin_r"] * 1000),
            "pin_w": "| 3 mm / 4 mm | +-%g mm each" % (2 * tol["pin_w"] * 1000),
            "pin_hh": "| 3 mm / 4 mm | +-%g mm each" % (2 * tol["pin_hh"] * 1000),
            "pin_z": "| 37.5 mm | +-%g mm" % (tol["pin_z"] * 1000),
            "pad_friction": "%g to %g (the pads' torsional friction in proportion)" % tol["pad_friction"],
            "joint_friction": "x %g to %g" % tol["joint_friction"],
            "wrench_drift": "+-%g N and +-%g N m per minute" % (tol["wrench_drift"][0] * 60, tol["wrench_drift"][1] * 60),
            "cam_fixed_pos": "| section 7 | +-%g mm / +-%g degrees" % (tol["cam_fixed_pos"] * 1000, tol["cam_fixed_rot_deg"]),
            "cam_fixed_rot_deg": "| section 7 | +-%g mm / +-%g degrees" % (tol["cam_fixed_pos"] * 1000, tol["cam_fixed_rot_deg"]),
            "cam_wrist_pos": "mount | +-%g mm / +-%g degrees" % (tol["cam_wrist_pos"] * 1000, tol["cam_wrist_rot_deg"]),
            "cam_wrist_rot_deg": "mount | +-%g mm / +-%g degrees" % (tol["cam_wrist_pos"] * 1000, tol["cam_wrist_rot_deg"]),
            "cam_focal": "+-%g %% / +-%g pixel" % (tol["cam_focal"] * 100, tol["cam_principal_px"]),
            "cam_principal_px": "+-%g %% / +-%g pixel" % (tol["cam_focal"] * 100, tol["cam_principal_px"]),
            "light": "+-%g %%" % (tol["light"] * 100),
        }
        self.assertEqual(sorted(stated), sorted(tol), "a tolerance the data sheet's check does not know")
        for key, text in stated.items():
            self.assertTrue(text in sheet, "the data sheet does not state the tolerance %s as %r" % (key, text))
        self.assertTrue("%g to %g |" % consts["NUT_FRICTION"] in sheet, "the nuts' friction range")
        self.assertTrue("%g to %g kg" % consts["NUT_MASS"] in sheet, "the nuts' mass range")
        flat = " ".join(INSTRUCTION.split())
        for words in ("section 11", "tolerances", "never told", "published camera calibration"):
            self.assertTrue(words in flat, "the instruction does not say %r" % words)

    def test_the_fit_is_stated(self):
        """The nut's fit on its shaft, the hand's coupled fingers and what follows from them are in the data sheet
        (section 12) and the instruction, with the simulator's numbers."""
        tree = ast.parse((TASK / "tests/keyed_env.py").read_text())
        consts = {}
        for node in tree.body:
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) \
                    and node.targets[0].id in ("TOL", "R_HOLE", "SHAFT_R"):
                consts[node.targets[0].id] = eval(compile(ast.Expression(node.value), "fit", "eval"), {"__builtins__": {}})
        clear = (consts["R_HOLE"] - consts["SHAFT_R"]) * 1000
        spread = (consts["TOL"]["hole_r"] + consts["TOL"]["shaft_r"]) * 1000
        fit = "%g mm of radial clearance on the drawing" % round(clear, 3)
        real = "%g to %g mm on real parts" % (round(clear - spread, 3), round(clear + spread, 3))
        self.assertEqual((fit, real), ("0.5 mm of radial clearance on the drawing", "0.35 to 0.65 mm on real parts"))
        sheet = " ".join(DATASHEET.split())
        self.assertIn("## 12. The fit", DATASHEET)
        tail = " ".join(DATASHEET[DATASHEET.index("## 12. The fit"):].split())
        for words in (fit, real, "no lead-in chamfer", "cannot be put over a shaft from image positions alone",
                      "What the wrist reads, the nut on the shaft end",
                      "What the wrist reads, the hole over the shaft", "What the wrist reads, a nut that binds"):
            self.assertTrue(words in tail, "the data sheet's section 12 does not say %r" % words)
        for words in ("a cylinder of radius %g mm" % (consts["SHAFT_R"] * 1000), "locating fit", "**Compliance.**",
                      "The two fingers are one mechanism"):
            self.assertTrue(words in sheet, "the data sheet does not say %r" % words)
        flat = " ".join(INSTRUCTION.split())
        for words in (fit, "(%s)" % real, "no lead-in chamfer", "from image positions alone", "section 12",
                      "shaft of radius %g mm" % (consts["SHAFT_R"] * 1000), "The two fingers move together"):
            self.assertTrue(words in flat, "the instruction does not say %r" % words)
        cell = (TASK / "tests/keyed_env.py").read_text()
        self.assertIn("mjEQ_JOINT", cell)
        self.assertIn('eq.name1 = "r_g_finger_joint1"', cell)
        self.assertIn('eq.name2 = "r_g_finger_joint2"', cell)

    def test_payload_states_readings_not_a_method(self):
        """What the agent is given says what the parts are and what the sensors read; it does not say how to put a
        nut on its shaft (the round-3 data sheet did: the rim against the shaft's side with its distance, a slide
        over the shaft end, "both ways work") or how to spend the commissioning window."""
        texts = {"instruction.md": INSTRUCTION}
        for p in sorted(PAYLOAD.rglob("*")):
            if p.is_file() and "__pycache__" not in p.parts:
                texts[p.relative_to(PAYLOAD).as_posix()] = p.read_text()
        for name, text in texts.items():
            flat = " ".join(text.split())
            for words in ("rim", "40.3", "40.4", "40.5", "Both ways", "both ways", "found by touch", "by touch",
                          "slid over", "spiral", "two directions", "teach-in and single measurements first",
                          "plan your episodes"):
                self.assertFalse(re.search(r"(?<![A-Za-z])%s(?![A-Za-z])" % re.escape(words), flat),
                                 "%s says %r" % (name, words))

    def test_instruction_names_only_what_the_agent_has(self):
        for name in ("datasheet.md", "cell_data.py", "commission/client.py", "commission/run_kit.py",
                     "helpers/wire_skeleton.py"):
            self.assertIn(name, INSTRUCTION)
            self.assertTrue((PAYLOAD / name).is_file(), name)
        for gone in ("keyed_env.py", "practice_kit", "practice_bench", "dev sim", "sample_kit"):
            self.assertNotIn(gone, INSTRUCTION)


class Cameras(unittest.TestCase):
    """The cameras' renderer (tests/keyed_render.py with tests/keyed_look.json) and its render bundle."""

    RENDER = (TASK / "tests/keyed_render.py").read_text()
    LOOK = json.loads((TASK / "tests/keyed_look.json").read_text())
    ASSETS = (TASK / "environment/assets.sha256").read_text()
    DOCKERFILE = (TASK / "environment/Dockerfile").read_text()

    def test_bundle_is_pinned_and_complete(self):
        self.assertRegex(self.ASSETS, r"(?m)^# revision: [0-9a-f]{40}$")
        self.assertIn("# prefix: tasks/arm/keyed_nut_peg_assembly/payload\n", self.ASSETS)
        listed = set(re.findall(r"(?m)^[0-9a-f]{64}  (\S+)$", self.ASSETS))
        names = ast.literal_eval(re.search(r"(?m)^BUNDLE_FILES = (\(.*\))$", self.RENDER).group(1))
        self.assertEqual(listed, {"render/" + n for n in names})
        self.assertEqual(self.LOOK["ibl"]["file"], "ibl.ktx")
        self.assertIn('os.environ.get("KEYED_RENDER_DIR", "/opt/task-assets/payload/render")', self.RENDER)
        self.assertIn("ENV KEYED_RENDER_DIR=/opt/task-assets/payload/render", self.DOCKERFILE)

    def test_bundle_is_the_cells_not_the_agents(self):
        # fetched into a root-owned folder that the agent account cannot read; nothing stages it into the home
        self.assertIn("chown -R root:keyedcell /opt/task-assets && chmod -R u=rwX,g=rX,o= /opt/task-assets",
                      self.DOCKERFILE)
        self.assertIn("! runuser -u user -- ls /opt/task-assets", self.DOCKERFILE)
        self.assertNotIn("task-assets", (TASK / "environment/setup/setup.sh").read_text())

    def test_the_lamps_tolerance_reaches_every_light(self):
        env = (TASK / "tests/keyed_env.py").read_text()
        self.assertIn('self.lamp_gain = 1.0 + float(self.unit["light"])', env)
        self.assertIn('self.gain = float(getattr(env, "lamp_gain", 1.0)) * float(look["exposure_scale"])', self.RENDER)
        self.assertIn('self.ibl_intensity = float(look["ibl"]["intensity"]) * self.gain', self.RENDER)
        self.assertIn('intensity=float(l["candela"]) * self.gain', self.RENDER)
        self.assertGreater(len(self.LOOK["lamps"]), 0)
        self.assertTrue(all(l.get("shadow") for l in self.LOOK["lamps"]))

    def test_images_keep_their_sizes_and_noise(self):
        env = (TASK / "tests/keyed_env.py").read_text()
        self.assertIn("noise = self.rng.normal(0.0, RGB_NOISE, rgb.shape)", env)
        self.assertGreaterEqual(int(self.LOOK["render"]["supersample"]), 1)
        self.assertIn("W, H = ep.size[cam]", self.RENDER)                 # the model's own camera resolutions
        self.assertIn("mujoco.mjv_updateCamera(m, d, cam, scn)", self.RENDER)   # the model's own cameras of the day

    def test_nothing_is_made_anew_per_episode(self):
        # an image must not depend on the episodes the process drew before: an episode writes data into the
        # process's renderables, materials, meshes and textures and makes none (tests/review/rendering.md)
        tree = ast.parse(self.RENDER)
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "_Filament")
        body = {n.name: ast.get_source_segment(self.RENDER, n) for n in cls.body if isinstance(n, ast.FunctionDef)}
        for name in ("_activate", "_nut_upload", "draw"):
            for made in ("create_renderable", "create_mesh", "create_texture", "create_light", "ModelResources(",
                         "ModelRenderables(", "Material(", "set_material(", "set_mesh(", "add_renderable("):
                self.assertNotIn(made, body[name].replace("self._lamps(ep)", ""), "%s: %s" % (name, made))
        self.assertIn("self._parts(ep)", body["_activate"])
        self.assertIn("if self.parts is not None:", body["_parts"])
        review = (TASK / "tests/review/test_draw_determinism.py").read_text()
        self.assertIn('if args.child == "R":', review)
        self.assertIn("reload_fail", review)

    def test_the_account_is_cleared_before_the_first_kits_clock(self):
        verify = (TASK / "tests/verify.py").read_text()
        main = verify[verify.index("\ndef main():"):]
        self.assertIn("clear_agent_account()", main)
        self.assertLess(main.index("clear_agent_account()"), main.index("run_episodes("))
        loop = verify[verify.index("\ndef run_episodes("):verify.index("\ndef main():")]
        self.assertNotIn("clear_agent_account", loop)

    def test_data_sheet_says_what_the_images_show(self):
        self.assertIn("**What the images show.**", DATASHEET)
        self.assertIn("what the images show", INSTRUCTION)


if __name__ == "__main__":
    unittest.main()
