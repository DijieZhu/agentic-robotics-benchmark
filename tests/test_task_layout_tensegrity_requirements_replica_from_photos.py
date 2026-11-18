"""tensegrity_requirements_replica_from_photos: the Harbor layout and what may reach the agent's image.

The grader's files (tests/) and the reference submission (solution/) must not reach the image or the agent home: no
file of either is byte-identical to a file of the image's build context (environment/ less setup/), of the small
givens (environment/setup/payload/) or of the photo set the image fetches (environment/assets.sha256). The only
exception is the public structure check and its schemas, which the grader carries as byte copies of the agent's
(tests/reqgrade/schema_check.py, tests/reqgrade/schemas/<name>.json); the hidden contract tables are not in the
payload. The asset list holds the photo set only, under this task's prefix. tests/given_hashes.json, the maintainers'
pin of the givens, equals the payload and the asset list. The manifest keeps the sizes the task was validated with.
Stdlib only, nothing is downloaded: the photos themselves are not in git, only their hashes.
"""
import hashlib
import json
import os
import re
import tomllib
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SLUG = "tensegrity_requirements_replica_from_photos"
TASK = REPO / "tasks" / "soft" / SLUG
PHOTO = re.compile(r"^photos/photo_(\d{3})\.jpg$")
PUBLIC_COPY = re.compile(r"^tests/reqgrade/(schema_check\.py|schemas/[A-Za-z0-9_.-]+\.json)$")


def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _tree(root):
    out = {}
    for d, dirs, files in os.walk(root):
        dirs[:] = sorted(x for x in dirs if x != "__pycache__")
        for f in sorted(files):
            p = os.path.join(d, f)
            out[os.path.relpath(p, root).replace(os.sep, "/")] = _sha(p)
    return out


def _asset_list():
    head, entries = {}, {}
    for line in (TASK / "environment" / "assets.sha256").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        if line.startswith("#"):
            key, _, value = line[1:].partition(":")
            head[key.strip()] = value.strip()
            continue
        digest, rel = line.split(None, 1)
        entries[rel.strip().lstrip("*")] = digest.lower()
    return head, entries


class TaskLayout(unittest.TestCase):
    def test_top_level_is_the_harbor_layout(self):
        names = {p.name for p in TASK.iterdir()} - {".ale-cache"}
        self.assertEqual(names, {"task.toml", "instruction.md", "environment", "tests", "solution"})

    def test_manifest(self):
        cfg = tomllib.loads((TASK / "task.toml").read_text(encoding="utf-8"))
        self.assertEqual(cfg["schema_version"], "1.4+ale.1")
        self.assertEqual(cfg["task"]["name"], "grounded-intelligence-initiative/" + SLUG)
        self.assertEqual(cfg["metadata"]["platform"], "soft")
        self.assertEqual(cfg["artifacts"], ["/home/user/submission"])
        self.assertEqual((cfg["environment"]["cpus"], cfg["environment"]["memory_mb"]), (4, 8192))
        self.assertEqual((cfg["agent"]["timeout_sec"], cfg["agent"]["network_mode"]), (21600, "no-network"))
        self.assertEqual((cfg["verifier"]["environment_mode"], cfg["verifier"]["timeout_sec"]), ("separate", 1200))
        ve = cfg["verifier"]["environment"]
        self.assertEqual((ve["cpus"], ve["memory_mb"], ve["gpus"]), (2, 4096, 0))

    def test_no_bytecode_in_the_task(self):
        found = [str(p.relative_to(TASK)) for p in TASK.rglob("*") if p.name == "__pycache__" or p.suffix == ".pyc"]
        self.assertEqual(found, [])

    def test_asset_list_is_the_photo_set_only(self):
        head, entries = _asset_list()
        conf = tomllib.loads((REPO / "task-assets.toml").read_text(encoding="utf-8"))["huggingface"]
        self.assertEqual(head.get("prefix"), "tasks/soft/%s/payload" % SLUG)
        self.assertEqual(head.get("repo"), conf["dataset"])
        self.assertEqual(head.get("endpoint"), conf["endpoint"].rstrip("/"))
        self.assertRegex(head.get("revision", ""), r"^[0-9a-f]{40}$")
        numbers = []
        for rel, digest in entries.items():
            m = PHOTO.match(rel)
            self.assertIsNotNone(m, "not a photo of the set: %s" % rel)
            self.assertRegex(digest, r"^[0-9a-f]{64}$")
            numbers.append(int(m.group(1)))
        self.assertEqual(sorted(numbers), list(range(1, len(numbers) + 1)))
        self.assertIn("photos/photo_%03d.jpg" % len(numbers), (TASK / "instruction.md").read_text(encoding="utf-8"))

    def test_grader_and_reference_never_reach_the_image(self):
        hidden = {"tests/" + r: h for r, h in _tree(TASK / "tests").items()}
        hidden.update({"solution/" + r: h for r, h in _tree(TASK / "solution").items()})
        visible = {}
        for rel, digest in _tree(TASK / "environment").items():
            if not rel.startswith("setup/"):
                visible.setdefault(digest, []).append("environment/" + rel)
        for rel, digest in _tree(TASK / "environment" / "setup" / "payload").items():
            visible.setdefault(digest, []).append("payload/" + rel)
        for rel, digest in _asset_list()[1].items():
            visible.setdefault(digest, []).append("assets/" + rel)
        bad = []
        for rel, digest in sorted(hidden.items()):
            if digest not in visible:
                continue
            m = PUBLIC_COPY.match(rel)
            if m and visible[digest] == ["payload/" + m.group(1)]:
                continue
            bad.append((rel, visible[digest]))
        self.assertEqual(bad, [])
        payload = set(_tree(TASK / "environment" / "setup" / "payload"))
        self.assertFalse([r for r in payload if "contract" in r.split("/")], "a hidden contract table in the payload")

    def test_given_hashes_pin_the_payload_and_the_photo_set(self):
        pin = json.loads((TASK / "tests" / "given_hashes.json").read_text(encoding="utf-8"))
        self.assertEqual(pin.get("format"), "given-hashes/2")
        expect = dict(_tree(TASK / "environment" / "setup" / "payload"))
        photos = _asset_list()[1]
        self.assertFalse(set(expect) & set(photos))
        expect.update(photos)
        self.assertEqual(pin["files"], expect)
        self.assertEqual(set(pin["sizes"]), set(expect))
        for rel in _tree(TASK / "environment" / "setup" / "payload"):
            self.assertEqual(pin["sizes"][rel], (TASK / "environment" / "setup" / "payload" / rel).stat().st_size, rel)


if __name__ == "__main__":
    unittest.main()
