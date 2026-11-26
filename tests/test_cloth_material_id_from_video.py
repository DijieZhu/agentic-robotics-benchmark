"""cloth_material_id_from_video: the copies, the hidden set's shape, the pin, the recordings' list and what the
instruction says hold in the files.

The planning simulator (cphys.py) and the scored skills (ctasks.py) live in three byte-identical copies: tests/ (the
grader), environment/setup/payload/ (the agent's home) and environment/gen/ (the renderer); the hidden model of the
real fabric (creal.py) in two (tests/, environment/gen/), and never in the agent's home. The recordings are a task
asset (docs/task-assets.md): listed in environment/assets.sha256, not in git.

Standard library only (the host-side suite has no numpy): nothing of the task is imported.
"""

import hashlib
import json
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / "tasks/arm/cloth_material_id_from_video"
SCORED = tuple("ABCDEFGHIJ")
SKILLS = ("slide", "overhang")
PIN = "X86_V4 AVX512_ICL AVX512_SPR"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Copies(unittest.TestCase):

    def test_the_shared_modules_are_byte_identical(self):
        for name in ("cphys.py", "ctasks.py"):
            for copy in ("environment/setup/payload", "environment/gen"):
                self.assertEqual(sha256(TASK / copy / name), sha256(TASK / "tests" / name), "{} differs in {}".format(name, copy))
        self.assertEqual(sha256(TASK / "environment/gen/carm.py"), sha256(TASK / "environment/setup/payload/carm.py"))
        self.assertEqual(sha256(TASK / "environment/gen/creal.py"), sha256(TASK / "tests/creal.py"))

    def test_the_looks_do_not_follow_the_materials(self):
        scene = (TASK / "environment/gen/cscene.py").read_text()
        body = scene[scene.index("def fabric_looks"):]
        body = body[:body.index("\ndef ", 1)] if "\ndef " in body[1:] else body
        for word in ("material", "young", "thickness", "mass"):
            self.assertNotIn(word + "[", body)
            self.assertNotIn("fabrics.json", body)
        self.assertIn("DRAW_THICKNESS = 0.0015", (TASK / "environment/gen/render_clips.py").read_text())

    def test_nothing_hidden_is_in_the_agents_home(self):
        payload = TASK / "environment/setup/payload"
        names = sorted(p.name for p in payload.rglob("*") if p.is_file())
        self.assertEqual(names, ["carm.py", "cio.py", "cphys.py", "ctasks.py", "session_time.sh"])
        for p in payload.rglob("*.py"):
            text = p.read_text()
            for word in ("creal", "STATIC_EXCESS", "V_STRIBECK", "RHO_AIR"):
                self.assertNotIn(word, text, "{} names the hidden model".format(p.name))


class HiddenSet(unittest.TestCase):

    def setUp(self):
        self.ref = json.loads((TASK / "tests/ref/fabrics.json").read_text())
        self.sel = json.loads((TASK / "tests/ref/selection.json").read_text())

    def test_ten_fabrics_with_four_tests_per_skill(self):
        self.assertEqual(tuple(sorted(self.ref["fabrics"])), SCORED)
        for k, f in self.ref["fabrics"].items():
            fams = [t["fam"] for t in f["tests"]]
            self.assertEqual({s: fams.count(s) for s in SKILLS}, {s: 4 for s in SKILLS}, k)
            self.assertEqual(sorted(f["reference"]), ["damping", "mu", "poisson", "thickness", "young"])
        self.assertEqual(list(self.ref["practice"]), ["P"])

    def test_the_selection_record_matches_the_set(self):
        cands = {c["candidate"]: c for c in self.sel["candidates"]}
        self.assertEqual(sorted(self.sel["ids"]), list(SCORED))
        order = [self.sel["ids"][k] for k in SCORED]
        self.assertEqual(order, sorted(order), "the scored fabrics are the accepted candidates in order")
        for k in SCORED:
            c = cands[self.sel["ids"][k]]
            self.assertTrue(c["accepted"])
            self.assertEqual(c["tests"], self.ref["fabrics"][k]["tests"])
            for s in SKILLS:
                kept = [r for r in c["examined"][s] if r["kept"]]
                self.assertEqual(len(kept), 4)
                self.assertTrue(all(r["q"] >= 1.0 for r in kept), "a kept test is one the reference scores 1.0 on")
        for c in self.sel["candidates"]:
            if not c["accepted"]:
                self.assertTrue(c["reason"])

    def test_the_reference_is_the_solution(self):
        sol = json.loads((TASK / "solution/ref/params.json").read_text())["fabrics"]
        self.assertEqual(sol, {k: f["reference"] for k, f in self.ref["fabrics"].items()})

    def test_the_anchor_is_the_reference_at_full(self):
        a = json.loads((TASK / "tests/anchor.json").read_text())
        self.assertEqual((a["metric"], a["direction"], a["value"], a["full_at"]), ("material_transfer_score", "higher", 1.0, 1.0))

    def test_the_renderer_gets_the_same_fabrics(self):
        gen = json.loads((TASK / "environment/gen/fabrics.json").read_text())["fabrics"]
        self.assertEqual(sorted(gen), sorted(SCORED + ("P",)))
        for k in SCORED:
            self.assertEqual(gen[k]["material"], self.ref["fabrics"][k]["material"])


class Pin(unittest.TestCase):

    def test_the_pin_is_set_where_the_arithmetic_runs(self):
        docker = (TASK / "environment/Dockerfile").read_text()
        self.assertIn('ENV NPY_DISABLE_CPU_FEATURES="{}"'.format(PIN), docker)
        self.assertIn("-W error::ImportWarning", docker)
        self.assertIn('NPY_DISABLE_CPU_FEATURES="{}"'.format(PIN), (TASK / "tests/test.sh").read_text())
        self.assertIn(PIN, (TASK / "instruction.md").read_text())
        fp = json.loads((TASK / "tests/numerics.json").read_text())
        self.assertRegex(fp["fingerprint"], r"^[0-9a-f]{64}$")
        self.assertEqual(sorted(fp["parts"]), ["cphys", "creal", "numpy"])

    def test_the_grader_has_no_clock(self):
        verify = (TASK / "tests/verify.py").read_text()
        self.assertNotIn("budget_s", verify)
        self.assertNotIn("budget_s", (TASK / "tests/grader_config.json").read_text())
        self.assertIn("check_numerics()", verify)


class Recordings(unittest.TestCase):

    def test_every_clip_of_every_fabric_is_listed(self):
        lines = [ln for ln in (TASK / "environment/assets.sha256").read_text().splitlines() if ln and not ln.startswith("#")]
        listed = {ln.split(None, 1)[1].strip() for ln in lines}
        want = {"data/cameras.json"}
        for fid in SCORED + ("P",):
            want.add("data/fabrics/{}/fabric.json".format(fid))
            for i in range(7):
                for name in ("clip.json", "clamp_500hz.npy", "arm_q.npy", "cam0.mp4", "cam1.mp4", "cam2.mp4"):
                    want.add("data/fabrics/{}/clip{:02d}/{}".format(fid, i, name))
        self.assertEqual(listed, want)
        self.assertFalse(any(p.suffix in (".mp4", ".npy") for p in TASK.rglob("*")), "recordings stay out of git")

    def test_the_instruction_matches(self):
        text = (TASK / "instruction.md").read_text()
        for phrase in ("ten fabrics", "7 clips", "product of its two skill means", "| 3 mm |", "| 6 mm |", "session_time.sh",
                       "comes with no\n  answer", "no score feedback"):
            self.assertIn(phrase, text)
        self.assertIsNone(re.search(r'"young":\s*[0-9]', text), "the instruction prints no material")
        tasks = (TASK / "tests/ctasks.py").read_text()
        self.assertIn('TOL = {"slide": 0.003, "overhang": 0.006}', tasks)
        self.assertIn("LIMIT_MIN=480", (TASK / "environment/setup/payload/session_time.sh").read_text())
        toml = (TASK / "task.toml").read_text()
        self.assertIn("timeout_sec = 28800", toml)


if __name__ == "__main__":
    unittest.main()
