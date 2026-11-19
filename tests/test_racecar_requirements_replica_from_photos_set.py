"""racecar_requirements_replica_from_photos: the grader's copies of what the agent is given, the pins, the reference
solution and what the agent is given.

The grader's procedure.py and schema_check.py are byte copies of the agent's and the ones grader_config.json pins;
schemas.json holds the agent's three schemas. tests/given_hashes.json pins every given file: a file that is in the
payload has its digest and size, and a given image that the published form fetches instead (it is then not in the
payload) is named with the same digest by the image list that setup.sh checks (environment/setup/given_images.sha256)
and, once wired, by the asset list (environment/assets.sha256). The reference solution's BOM is the stored optimum
and its sketch the hidden key quantities. Nothing agent-visible (instruction.md, environment/setup/payload/**) holds
the optimum's price, a digest of a hidden file or the name of a hidden file. Stdlib only.
"""
import hashlib
import json
import unittest
from pathlib import Path

TASK = Path(__file__).resolve().parents[1] / "tasks/ground/racecar_requirements_replica_from_photos"
TESTS = TASK / "tests"
GT = TESTS / "gt"
PAYLOAD = TASK / "environment/setup/payload"
IMAGE_LIST = TASK / "environment/setup/given_images.sha256"
ASSET_LIST = TASK / "environment/assets.sha256"
SCHEMA_FILES = {"requirements.yaml": "requirements.schema.json", "bom.json": "bom.schema.json",
                "kinematic_sketch.json": "kinematic_sketch.schema.json"}
HIDDEN_FILE_NAMES = ("car_truth.json", "key_quantities.json", "reference_sheet.json", "optimum.json",
                     "grader_config.json")
TEXT_SUFFIXES = (".md", ".json", ".py", ".yaml")


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _list(path):
    out = {}
    for line in Path(path).read_text(encoding="ascii").splitlines():
        if not line or line.startswith("#"):
            continue
        digest, rel = line.split("  ", 1)
        out[rel] = digest
    return out


def _visible_text():
    files = [TASK / "instruction.md"] + sorted(p for p in PAYLOAD.rglob("*") if p.is_file())
    return {p: p.read_text(encoding="utf-8") for p in files if p.suffix in TEXT_SUFFIXES}


class CopiesTests(unittest.TestCase):
    def test_grader_modules_are_the_payloads_and_pinned(self):
        cfg = _json(TESTS / "grader_config.json")
        for name, key in (("procedure.py", "procedure_sha256"), ("schema_check.py", "schema_check_sha256")):
            self.assertEqual((TESTS / name).read_bytes(), (PAYLOAD / name).read_bytes(), name)
            self.assertEqual(_sha(TESTS / name), cfg[key], name)

    def test_schemas_json_holds_the_agents_schemas(self):
        schemas = _json(TESTS / "schemas.json")
        self.assertEqual(set(schemas), set(SCHEMA_FILES))
        for name, file in SCHEMA_FILES.items():
            self.assertEqual(schemas[name], _json(PAYLOAD / "schemas" / file), name)

    def test_given_hashes_pin_every_given_file(self):
        given = _json(TESTS / "given_hashes.json")
        images = _list(IMAGE_LIST)
        in_payload = {p.relative_to(PAYLOAD).as_posix() for p in PAYLOAD.rglob("*") if p.is_file()}
        self.assertTrue(in_payload <= set(given), sorted(in_payload - set(given))[:5])
        for rel, rec in given.items():
            if rel in in_payload:
                self.assertEqual(_sha(PAYLOAD / rel), rec["sha256"], rel)
                self.assertEqual((PAYLOAD / rel).stat().st_size, rec["size"], rel)
            else:
                self.assertEqual(images.get(rel), rec["sha256"], "%s is neither in the payload nor listed" % rel)

    def test_image_list_names_every_photo_and_page(self):
        images = _list(IMAGE_LIST)
        given = _json(TESTS / "given_hashes.json")
        self.assertEqual(sorted(r for r in images if r.startswith("photos/")),
                         ["photos/PH-%02d.jpg" % i for i in range(1, 34)])
        self.assertEqual(len([r for r in images if r.startswith("catalogue/")]), 27)
        self.assertEqual({r for r in given if r.endswith((".png", ".jpg"))}, set(images))
        for rel, digest in images.items():
            self.assertEqual(given[rel]["sha256"], digest, rel)
        if ASSET_LIST.is_file():
            self.assertEqual(_list(ASSET_LIST), images)


class SolutionTests(unittest.TestCase):
    def test_reference_bom_is_the_stored_optimum(self):
        opt = _json(GT / "optimum.json")
        bom = _json(TASK / "solution/submission/bom.json")
        self.assertEqual(bom, opt["car"])
        self.assertEqual(list(bom), _json(TESTS / "grader_config.json")["slot_order"])

    def test_reference_sketch_is_the_key_quantities(self):
        kq = _json(GT / "key_quantities.json")["sketch"]
        sketch = _json(TASK / "solution/submission/kinematic_sketch.json")
        self.assertEqual(set(sketch), set(kq))
        for k, v in kq.items():
            self.assertEqual(sketch[k]["nominal"], v, k)


class VisibleTests(unittest.TestCase):
    def test_no_hidden_value_or_name_is_visible(self):
        opt = _json(GT / "optimum.json")
        price = opt["J_usd"]
        needles = {"%.2f" % price, repr(price), opt["passing_set"]["sha256"]}
        needles |= {_sha(p) for p in GT.glob("*.json")}
        needles |= set(HIDDEN_FILE_NAMES)
        for path, text in _visible_text().items():
            for n in needles:
                self.assertNotIn(n, text, "%s holds %r" % (path.relative_to(TASK), n))


if __name__ == "__main__":
    unittest.main()
