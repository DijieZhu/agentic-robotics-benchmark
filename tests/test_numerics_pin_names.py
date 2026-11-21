"""The numerics pins name NumPy's CPU features by names the task's own numpy knows.

`NPY_DISABLE_CPU_FEATURES` takes NumPy's AVX-512 loops out, so that a submitted program computes the same on an AVX2
grading host and on an AVX-512 one (np.arctan2, exp, log, power and others round differently there). NumPy renamed its
dispatch groups: numpy 2.2 knows AVX512F, AVX512_SKX, ...; numpy 2.5 knows X86_V4 instead and rejects the old names
with an ImportWarning that nobody sees, disabling nothing (measured on a Zen 5 host with numpy 2.2.5, 2.2.6, 2.5.3 and
2.5.4; the same submission of one task graded 8 of 12 episodes on an AVX2 host and 7 of 12 on that one). A pin
written for the other numpy is silent and has no effect, so it is checked here for every task: a literal value of
the variable in a file the image or the grader runs must fit the numpy that the task pins.

pantry_jar_refilling carried the old names with numpy 2.5.4; its image, its fingerprint and its instruction are
checked below. Standard library only: no numpy, no Docker.
"""

import json
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TASKS = ROOT / "tasks"
VALUE = re.compile(r"NPY_DISABLE_CPU_FEATURES[\"']?\s*(?:=|:|,)\s*[\"']?([A-Z][A-Z0-9_]*(?:[ ,]+[A-Z][A-Z0-9_]*)*)")
NUMPY = re.compile(r"^numpy==([0-9]+)\.([0-9]+)\.[0-9]+\s*$", re.M)
OLD = {"AVX512F", "AVX512CD", "AVX512_KNL", "AVX512_KNM", "AVX512_SKX", "AVX512_CLX", "AVX512_CNL", "AVX512FP16"}
NEW = {"X86_V4"}
# Files that run in the image or in the grader. A record of a past run (a selection log, a review tool) is no pin.
RUN_SUFFIXES = (".py", ".sh", ".pth")
SKIP_PARTS = ("review", "hidden")


def task_dirs():
    return sorted(p for p in TASKS.glob("*/*") if (p / "task.toml").is_file())


def numpy_minor(task):
    """(major, minor) of the numpy the task pins in a requirements file, None when it pins none or several."""
    found = set()
    for req in list(task.rglob("requirements*.txt")):
        found.update((int(a), int(b)) for a, b in NUMPY.findall(req.read_text(errors="replace")))
    return found.pop() if len(found) == 1 else None


def pinned_values(task):
    out = []
    for path in sorted(task.rglob("*")):
        rel = path.relative_to(task)
        if not path.is_file() or any(part in SKIP_PARTS for part in rel.parts):
            continue
        if not (path.name == "Dockerfile" or path.suffix in RUN_SUFFIXES):
            continue
        text = path.read_text(errors="replace")
        if "NPY_DISABLE_CPU_FEATURES" not in text:
            continue
        for value in VALUE.findall(text):
            names = set(re.split(r"[ ,]+", value.strip()))
            if names & (OLD | NEW):
                out.append((str(rel), names))
    return out


class Names(unittest.TestCase):

    def test_every_literal_pin_fits_the_tasks_numpy(self):
        seen = 0
        for task in task_dirs():
            version = numpy_minor(task)
            for rel, names in pinned_values(task):
                seen += 1
                where = "{}/{}".format(task.name, rel)
                self.assertIsNotNone(version, "{} pins NumPy's CPU features, but the task pins no single numpy".format(where))
                if version >= (2, 5):
                    self.assertFalse(names & OLD, "{}: numpy {}.{} rejects {} and then disables nothing; its names are "
                                     "X86_V4 AVX512_ICL AVX512_SPR".format(where, version[0], version[1], sorted(names & OLD)))
                    self.assertIn("X86_V4", names, where)
                elif version <= (2, 2):
                    self.assertFalse(names & NEW, "{}: numpy {}.{} does not know X86_V4 and then keeps AVX512F and "
                                     "AVX512_SKX on".format(where, version[0], version[1]))
                    self.assertIn("AVX512F", names, where)     # a value may run over several source lines
                else:
                    self.fail("{}: numpy {}.{} is not measured; check which names it knows (numpy._core."
                              "_multiarray_umath.__cpu_dispatch__ on an AVX-512 host) and extend this test".format(
                                  where, version[0], version[1]))
        self.assertGreater(seen, 20, "the scan found too few pins: the pattern no longer matches the tasks' files")

    def test_no_run_file_of_a_numpy_2_5_task_builds_the_pin_from_old_names(self):
        """A pin assembled at run time (a list of names joined into the variable) escapes the literal scan: a file
        that sets the variable in a numpy 2.5 task names none of numpy 2.2's groups at all."""
        for task in task_dirs():
            version = numpy_minor(task)
            if version is None or version < (2, 5):
                continue
            for path in sorted(task.rglob("*")):
                rel = path.relative_to(task)
                if not path.is_file() or any(part in SKIP_PARTS for part in rel.parts):
                    continue
                if not (path.name == "Dockerfile" or path.suffix in RUN_SUFFIXES):
                    continue
                text = path.read_text(errors="replace")
                if "NPY_DISABLE_CPU_FEATURES" not in text:
                    continue
                code = "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("#"))
                self.assertIsNone(re.search(r"[\"'][^\"'\n]*\bAVX512(?:F|_SKX)\b", code),
                                  "{}/{} names AVX512F or AVX512_SKX next to the pin".format(task.name, rel))

    def test_the_scan_reads_both_spellings(self):
        self.assertEqual(VALUE.findall('ENV NPY_DISABLE_CPU_FEATURES="X86_V4 AVX512_ICL AVX512_SPR"'), ["X86_V4 AVX512_ICL AVX512_SPR"])
        self.assertEqual(VALUE.findall('"NPY_DISABLE_CPU_FEATURES": "X86_V4,AVX512_ICL"'), ["X86_V4,AVX512_ICL"])
        self.assertEqual(VALUE.findall('os.environ.setdefault("NPY_DISABLE_CPU_FEATURES", "AVX512F AVX512_SKX")'), ["AVX512F AVX512_SKX"])


class PantryJarRefilling(unittest.TestCase):
    TASK = TASKS / "arm/pantry_jar_refilling"
    PIN = {"OPENBLAS_CORETYPE": "Haswell", "OPENBLAS_L2_SIZE": "256", "NPY_DISABLE_CPU_FEATURES": "X86_V4 AVX512_ICL AVX512_SPR",
           "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "OPENCV_IPP": "avx2"}

    def test_the_image_sets_every_default_and_fails_on_an_unknown_name(self):
        docker = (self.TASK / "environment/Dockerfile").read_text()
        for key, value in self.PIN.items():
            self.assertIn("'os.environ.setdefault(\"{}\", \"{}\")'".format(key, value), docker)
        self.assertIn("'    os.environ.setdefault(\"OPENCV_CPU_DISABLE\", \"AVX512-SKX\")'", docker)
        self.assertIn("jar_numerics_pin.pth", docker)
        self.assertNotIn("AVX512_SKX\"", docker, "the names of numpy 1.x / 2.2 are gone from the .pth")
        check = next(line for line in docker.splitlines() if "numerics pin OK" in line)
        self.assertIn("env -i PATH=/opt/venv/bin:/usr/bin:/bin HOME=/tmp /opt/venv/bin/python3 -I -W error::ImportWarning", check,
                      "the build starts a Python without the variables and fails on numpy's warning for an unknown name")
        self.assertIn("set(e['NPY_DISABLE_CPU_FEATURES'].split()) == set(D) - {'X86_V3'}", check)
        self.assertEqual(numpy_minor(self.TASK), (2, 5))

    def test_the_fingerprint_covers_a_controllers_arithmetic(self):
        ref = json.loads((self.TASK / "tests/numerics.json").read_text())
        self.assertEqual(sorted(ref), ["fingerprint", "math", "seed", "simulation", "steps", "what"])
        self.assertTrue({"trig_float64", "exp_float64", "big", "scipy", "cv2"} <= set(ref["math"]))
        self.assertNotIn("absent", ref["math"].values(), "SciPy and OpenCV were importable when the reference was written")
        check = (self.TASK / "tests/numerics_check.py").read_text()
        self.assertIn('CONTROLLER_ENV = {"PATH": "/opt/venv/bin:/usr/local/bin:/usr/bin:/bin", "HOME": "/tmp", "LANG": "C.UTF-8"}', check)
        self.assertIn("env=dict(CONTROLLER_ENV)", check)
        verify = (self.TASK / "tests/verify.py").read_text()
        self.assertIn('"PATH=/opt/venv/bin:/usr/local/bin:/usr/bin:/bin"', verify, "run.sh starts with this PATH")
        self.assertIn('json.load(fh)["fingerprint"]', verify)

    def test_the_instruction_names_the_pin(self):
        text = (self.TASK / "instruction.md").read_text()
        for name in ("OPENBLAS_CORETYPE=Haswell", "OPENBLAS_L2_SIZE=256", "OPENBLAS_NUM_THREADS=1", "OMP_NUM_THREADS=1",
                     "NPY_DISABLE_CPU_FEATURES", "OPENCV_IPP=avx2"):
            self.assertIn(name, text)


if __name__ == "__main__":
    unittest.main()
