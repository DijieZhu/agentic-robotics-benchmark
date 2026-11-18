"""tensegrity_cad_design_from_photos: what the image and the asset dataset may carry.

The grader's reference (the original parts, their features, the "seen" marks) and the oracle live only in tests/ and
solution/. The image is built from environment/ (less setup/) and fetches the agent-visible payload listed in
environment/assets.sha256; setup.sh stages that payload into the agent home, and the separate verifier reads the
purchased components' STEP files from the same fetched payload. So:
  * no non-empty file of tests/ or solution/ has the sha256 of a file of environment/ (empty files carry no content);
  * assets.sha256 names only the task's payload folder in the dataset (tasks/soft/<slug>/payload), only files under
    photos/ and purchased/<part number>/, and none of its files is a file of tests/ or solution/;
  * the purchased STEPs the grader pins (tests/reference/purchased_pins.json) are payload files with the same sha256;
  * no payload file is a public file of this robot that the grader blocks (tests/public_blocklist.json);
  * the task folder holds the five Harbor entries and nothing else.
Stdlib only.
"""
import hashlib
import json
import re
import unittest
from pathlib import Path

SLUG = "tensegrity_cad_design_from_photos"
TASK = Path(__file__).resolve().parents[1] / "tasks" / "soft" / SLUG
EMPTY = hashlib.sha256(b"").hexdigest()
PART_NUMBER = r"[A-Z0-9]+(?:-[A-Z0-9]+)+"
PAYLOAD_PATH = re.compile(r"^(photos/photo_\d{3}\.jpg|purchased/(%s)/\2(\.step|_datasheet\.md|_drawing\.png))$"
                          % PART_NUMBER)


def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _files(root):
    return {p.relative_to(TASK).as_posix(): _sha(p) for p in sorted(root.rglob("*"))
            if p.is_file() and "__pycache__" not in p.parts}


def _asset_list():
    head, files = {}, {}
    for line in (TASK / "environment" / "assets.sha256").read_text().splitlines():
        if not line.strip():
            continue
        if line.startswith("#"):
            key, _, value = line[1:].partition(":")
            head[key.strip()] = value.strip()
            continue
        digest, rel = line.split(None, 1)
        files[rel.strip().lstrip("*")] = digest.lower()
    return head, files


class TaskLayout(unittest.TestCase):
    def test_top_level_is_the_harbor_layout(self):
        self.assertEqual(sorted(p.name for p in TASK.iterdir()),
                         ["environment", "instruction.md", "solution", "task.toml", "tests"])

    def test_hidden_files_are_not_in_the_build_context(self):
        hidden = _files(TASK / "tests")
        hidden.update(_files(TASK / "solution"))
        self.assertIn("tests/reference/families.json", hidden)
        self.assertIn("solution/build_oracle.py", hidden)
        context = {}
        for rel, digest in _files(TASK / "environment").items():
            context.setdefault(digest, []).append(rel)
        hits = {rel: context[d] for rel, d in hidden.items() if d != EMPTY and d in context}
        self.assertEqual(hits, {})

    def test_asset_list_holds_only_the_payload(self):
        head, files = _asset_list()
        self.assertEqual(head.get("prefix"), "tasks/soft/%s/payload" % SLUG)
        self.assertRegex(head.get("revision", ""), r"^[0-9a-f]{40}$")
        self.assertTrue(files)
        self.assertEqual([rel for rel in files if not PAYLOAD_PATH.match(rel)], [])
        photos = sorted(rel for rel in files if rel.startswith("photos/"))
        self.assertEqual(photos, ["photos/photo_%03d.jpg" % k for k in range(1, len(photos) + 1)])
        hidden = set(_files(TASK / "tests").values()) | set(_files(TASK / "solution").values())
        self.assertEqual(sorted(rel for rel, d in files.items() if d in hidden), [])

    def test_pinned_purchased_steps_are_payload_files(self):
        _, files = _asset_list()
        pins = json.loads((TASK / "tests" / "reference" / "purchased_pins.json").read_text())
        self.assertTrue(pins["types"])
        for kind, pin in pins["types"].items():
            self.assertEqual(files.get("purchased/" + pin["step"]), pin["sha256"], kind)

    def test_no_blocked_public_file_in_the_payload(self):
        _, files = _asset_list()
        blocked = set(json.loads((TASK / "tests" / "public_blocklist.json").read_text())["files"].values())
        self.assertTrue(blocked)
        self.assertEqual(sorted(rel for rel, d in files.items() if d in blocked), [])


if __name__ == "__main__":
    unittest.main()
