"""rov_cad_design_from_photos: the task folder's layout and the separation of what the agent gets from what the grader
holds. The top level is the Harbor layout; no file of tests/ (the reference parts, the purchased modules' pins, the
grader) or solution/ (the reference build) has the bytes of a file of environment/ (the image's build context); the
asset list names the agent's givens only (photos/photo_001.jpg onwards without a gap, the parts list, and a STEP file,
a datasheet and a drawing per purchased module) under this task's folder of the asset dataset; the STEP files it lists
are the ones the grader's pins name, byte for byte; and no listed file has the bytes of a published file of the
vehicle that the grader refuses (tests/public_blocklist.json), nor of a file of tests/ or solution/. Stdlib only.
"""
import hashlib
import json
import re
import unittest
from pathlib import Path

TASK = Path(__file__).resolve().parents[1] / "tasks/underwater/rov_cad_design_from_photos"
TOP = {"task.toml", "instruction.md", "environment", "tests", "solution"}
PHOTO = re.compile(r"^photos/photo_(\d{3})\.jpg$")
PURCHASED = re.compile(r"^purchased/([A-Z0-9-]+)/([A-Z0-9-]+)(\.step|_datasheet\.md|_drawing\.png)$")
HEADER = ("endpoint", "repo", "revision", "prefix")


def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _files(root):
    return sorted(p for p in root.rglob("*") if p.is_file() and "__pycache__" not in p.parts)


def _asset_list():
    head, files = {}, []
    for line in (TASK / "environment" / "assets.sha256").read_text().splitlines():
        if not line.strip():
            continue
        if line.startswith("#"):
            key, _, value = line[1:].partition(":")
            head[key.strip()] = value.strip()
            continue
        digest, rel = line.split(None, 1)
        files.append((digest, rel.strip()))
    return head, files


class TaskLayout(unittest.TestCase):
    def test_top_level_is_the_harbor_layout(self):
        self.assertEqual({p.name for p in TASK.iterdir()} - {".ale-cache"}, TOP)

    def test_hidden_and_reference_files_are_not_in_the_build_context(self):
        env = {_sha(p): p for p in _files(TASK / "environment")}
        hidden = _files(TASK / "tests") + _files(TASK / "solution")
        self.assertGreater(len(hidden), 50)
        both = sorted("%s = %s" % (p.relative_to(TASK), env[_sha(p)].relative_to(TASK)) for p in hidden
                      if _sha(p) in env)
        self.assertEqual(both, [])

    def test_asset_list_names_the_givens_only(self):
        head, files = _asset_list()
        self.assertEqual(sorted(head), sorted(HEADER))
        self.assertEqual(head["prefix"], "tasks/underwater/rov_cad_design_from_photos/payload")
        self.assertRegex(head["revision"], r"^[0-9a-f]{40}$")
        rels = [rel for _d, rel in files]
        self.assertEqual(rels, sorted(rels))
        photos = [int(PHOTO.match(r).group(1)) for r in rels if PHOTO.match(r)]
        self.assertEqual(photos, list(range(1, len(photos) + 1)))
        self.assertGreater(len(photos), 100)
        bought = {}
        for r in rels:
            m = PURCHASED.match(r)
            if m:
                self.assertEqual(m.group(1), m.group(2), r)
                bought.setdefault(m.group(1), set()).add(m.group(3))
        self.assertEqual({pn: kinds for pn, kinds in bought.items()},
                         {pn: {".step", "_datasheet.md", "_drawing.png"} for pn in bought})
        rest = [r for r in rels if not PHOTO.match(r) and not PURCHASED.match(r)]
        self.assertEqual(rest, ["parts_list.md"])
        self.assertTrue(all(re.match(r"^[0-9a-f]{64}$", d) for d, _r in files))
        self.assertEqual(len({d for d, _r in files}), len(files))

    def test_listed_step_files_are_the_graders_pins(self):
        pins = json.loads((TASK / "tests" / "reference" / "purchased_pins.json").read_text())["types"]
        _head, files = _asset_list()
        listed = {rel: d for d, rel in files if rel.endswith(".step")}
        pinned = {"purchased/" + t["step"]: t["sha256"] for t in pins.values()}
        self.assertEqual(listed, pinned)

    def test_no_listed_file_is_a_published_or_hidden_file(self):
        blocked = set(json.loads((TASK / "tests" / "public_blocklist.json").read_text())["files"].values())
        self.assertGreater(len(blocked), 10)
        hidden = {_sha(p) for p in _files(TASK / "tests") + _files(TASK / "solution")}
        _head, files = _asset_list()
        self.assertEqual(sorted(rel for d, rel in files if d in blocked), [])
        self.assertEqual(sorted(rel for d, rel in files if d in hidden), [])


if __name__ == "__main__":
    unittest.main()
