"""freeflyer_cad_design_from_photos: the task folder's layout and the separation of what the agent gets from what the
grader holds. The top level is the Harbor layout; no file of tests/ (the standard, the graded samples, the grader) or
solution/ (the reference build) has the bytes of a file of environment/ (the image's build context); the asset list
names the photo set only, photos/photo_001.jpg onwards without a gap, under this task's folder of the asset dataset; and
no listed photograph, nor a staged local copy of one, has the bytes of a file of the robot's published material that
the grader refuses (tests/public_blocklist.json). Stdlib only.
"""
import hashlib
import json
import re
import unittest
from pathlib import Path

TASK = Path(__file__).resolve().parents[1] / "tasks/space/freeflyer_cad_design_from_photos"
TOP = {"task.toml", "instruction.md", "environment", "tests", "solution"}
PHOTO = re.compile(r"^photos/photo_(\d{3})\.jpg$")
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

    def test_asset_list_names_the_photo_set_only(self):
        head, files = _asset_list()
        self.assertEqual(sorted(head), sorted(HEADER))
        self.assertEqual(head["prefix"], "tasks/space/freeflyer_cad_design_from_photos/payload")
        self.assertRegex(head["revision"], r"^[0-9a-f]{40}$")
        rels = [rel for _d, rel in files]
        self.assertEqual([r for r in rels if not PHOTO.match(r)], [])
        self.assertEqual([int(PHOTO.match(r).group(1)) for r in rels], list(range(1, len(rels) + 1)))
        self.assertTrue(all(re.match(r"^[0-9a-f]{64}$", d) for d, _r in files))
        self.assertEqual(len({d for d, _r in files}), len(files))

    def test_no_photograph_is_a_published_file_of_the_robot(self):
        blocked = set(json.loads((TASK / "tests" / "public_blocklist.json").read_text())["files"].values())
        self.assertGreater(len(blocked), 10)
        _head, files = _asset_list()
        self.assertEqual(sorted(rel for d, rel in files if d in blocked), [])
        local = TASK / "environment" / "local_assets"
        if local.is_dir():   # a validation copy before the upload: the staged photographs themselves
            have = {p.relative_to(local).as_posix(): _sha(p) for p in _files(local)}
            self.assertEqual(have, {rel: d for d, rel in files})
            self.assertEqual(sorted(r for r, d in have.items() if d in blocked), [])


if __name__ == "__main__":
    unittest.main()
