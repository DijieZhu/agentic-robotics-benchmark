"""freeflyer_estimator_and_controller_from_docs: the wiring of the task's given flight records.

The unit's earlier flight records (106 files under payload/unit/records/ of the agent's home, about half a gigabyte)
are not in git: the image fetches them at build from the benchmark's asset dataset, pinned by commit and checked file
by file, and setup.sh copies them into the agent's home beside the givens kept in git (environment/setup/payload/).
These tests guard that wiring: the vendored fetcher is byte-identical to shared/task_assets/fetch_assets.py; the
Dockerfile ends with the fetch block of docs/task-assets.md, after every toolchain layer; setup.sh stages
/opt/task-assets/payload into the home; the records are not in the task folder and no local asset root is; and the
asset list names the dataset of task-assets.toml, the task's own prefix and exactly the 106 records (one digest over
its rows), none of which is also a file kept in git.

Before the pin (PINNED = False) the branch carries no asset list: its checks are skipped, and a list whose revision is
the placeholder of forty zeros is accepted. The commit that pins the dataset commit (scripts/task_assets.py list
--revision <commit>, which writes environment/assets.sha256) sets PINNED = True: from then on the list must exist and
its revision must be a 40-hex commit other than the placeholder. The rows do not depend on the revision, so ROWS_SHA256
stays as it is. Stdlib only; nothing is downloaded, imported from the task folder or written.
"""
import hashlib
import os
import re
import tomllib
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SLUG = "freeflyer_estimator_and_controller_from_docs"
TASK = REPO / "tasks" / "space" / SLUG
ENV = TASK / "environment"
PAYLOAD = ENV / "setup" / "payload"
LIST = ENV / "assets.sha256"
FETCHER = REPO / "shared" / "task_assets" / "fetch_assets.py"
PLACEHOLDER = "0" * 40
# False until the commit that pins the dataset commit; that commit sets it to True
PINNED = True
# the given records: their number and the sha256 of the asset list's rows ("<sha256>  <path>" lines, as
# scripts/task_assets.py list writes them, each ending with a newline)
ROWS = 106
ROWS_SHA256 = "a77a83da7693a254423148502e36c0fcc0e090679704ae3ec5b25e63d7974fed"
RECORDS_PREFIX = "payload/unit/records/"


def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _instructions(text):
    """The Dockerfile's instructions, comments and blank lines removed first, continuation lines joined."""
    out, cur = [], []
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        cont = s.endswith("\\")
        cur.append(s[:-1].strip() if cont else s)
        if not cont:
            out.append(" ".join(" ".join(cur).split()))
            cur = []
    if cur:
        out.append(" ".join(" ".join(cur).split()))
    return out


def _fetch_block():
    m = re.search(r"```dockerfile\n(.*?)```", (REPO / "docs" / "task-assets.md").read_text(encoding="utf-8"), re.S)
    assert m, "docs/task-assets.md has no Dockerfile fence"
    return m.group(1)


def _asset_list():
    head, rows = {}, []
    for ln in LIST.read_text(encoding="utf-8").splitlines():
        if ln.startswith("#"):
            key, _, value = ln[1:].partition(":")
            head[key.strip()] = value.strip()
        elif ln.strip():
            rows.append(ln)
    return head, rows


class AssetWiring(unittest.TestCase):
    def test_fetcher_is_the_shared_one(self):
        self.assertEqual((ENV / "fetch_assets.py").read_bytes(), FETCHER.read_bytes())

    def test_dockerfile_ends_with_the_fetch_block_after_the_toolchain(self):
        text = (ENV / "Dockerfile").read_text(encoding="utf-8")
        self.assertTrue(text.endswith(_fetch_block()), "the Dockerfile does not end with the fetch block")
        ins = _instructions(text)
        self.assertEqual(ins[-2], "COPY fetch_assets.py assets.sha256 /opt/task-assets/")
        self.assertTrue(ins[-1].startswith("RUN python3 /opt/task-assets/fetch_assets.py /opt/task-assets/assets.sha256 "
                                           "/opt/task-assets/payload"), ins[-1])
        self.assertEqual(sum("fetch_assets.py" in i for i in ins), 2)
        self.assertTrue(ins[0].startswith("FROM "))
        self.assertGreaterEqual(sum(i.startswith("RUN ") for i in ins[:-2]), 3)

    def test_setup_stages_the_fetched_files_into_the_home(self):
        text = (ENV / "setup" / "setup.sh").read_text(encoding="utf-8")
        self.assertIn('for root in "$SETUP_DIR/payload" /opt/task-assets/payload; do', text)
        self.assertIn('cp -r "$root/." "$HOME_DIR/"', text)
        self.assertIn("payload/unit/records", text)

    def test_the_records_are_not_in_the_task_folder(self):
        self.assertFalse((PAYLOAD / "payload" / "unit" / "records").exists())
        self.assertIn("payload/payload/unit/records/", (ENV / "setup" / ".gitignore").read_text(encoding="utf-8"))
        for root in (ENV / "assets", TASK / "tests" / "assets", TASK / "solution" / "assets"):
            self.assertFalse(root.exists(), "a local asset root: %s" % root.relative_to(TASK))
        top = {p.name for p in TASK.iterdir()} - {".ale-cache"}
        self.assertEqual(top, {"task.toml", "instruction.md", "environment", "tests", "solution"})

    def test_the_asset_list(self):
        if not LIST.exists():
            if PINNED:
                self.fail("environment/assets.sha256 is missing after the pin")
            self.skipTest("before the pin: the branch carries no asset list yet")
        head, rows = _asset_list()
        cfg = tomllib.loads((REPO / "task-assets.toml").read_text(encoding="utf-8"))["huggingface"]
        self.assertEqual(head.get("endpoint", "").rstrip("/"), cfg["endpoint"].rstrip("/"))
        self.assertEqual(head.get("repo"), cfg["dataset"])
        self.assertEqual(head.get("prefix"), "tasks/space/%s/payload" % SLUG)
        self.assertRegex(head.get("revision", ""), r"^[0-9a-f]{40}$")
        if PINNED:
            self.assertNotEqual(head.get("revision"), PLACEHOLDER, "the asset list still names the placeholder")
        self.assertEqual(len(rows), ROWS)
        paths = []
        for ln in rows:
            digest, _, rel = ln.partition("  ")
            self.assertRegex(digest, r"^[0-9a-f]{64}$")
            self.assertTrue(rel.startswith(RECORDS_PREFIX) and ".." not in rel.split("/"), rel)
            paths.append(rel)
        self.assertEqual(len(set(paths)), ROWS)
        self.assertEqual(hashlib.sha256("".join(ln + "\n" for ln in rows).encode("utf-8")).hexdigest(), ROWS_SHA256)
        for want in (RECORDS_PREFIX + "about_the_records.md", RECORDS_PREFIX + "index.csv"):
            self.assertIn(want, paths)
        kept = [rel for rel in paths if (PAYLOAD / rel).exists()]
        self.assertEqual(kept, [], "a listed record is also a file kept in git")


if __name__ == "__main__":
    unittest.main()
