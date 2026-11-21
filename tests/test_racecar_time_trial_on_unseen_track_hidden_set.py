"""racecar_time_trial_on_unseen_track: the repository-level test that guards the hashes of the hidden set.

The hidden set was drawn by a pre-registered selection rule (tests/hidden/selection.prereg.json; the rule is quoted in
tests/hidden/selection.json under rule, one citation reworded by the record's recorded change 2). Its item 8, FROZEN
AFTER SELECTION, has the same text in both and says that the six world files, the six reference records,
tests/anchor.json, tests/review/observability.json, the completed selection.json and the oracle token's sha256 are
frozen by sha256, and that "a repository-level pytest guards the hashes". This is that test, and it checks nothing
else:

(a) every file of the record's frozen_after_selection_sha256 table exists and has the sha256 the table gives; the
    table is read from the record, so the test follows the record, and the table holds the world file and the
    reference record of every venue the record selected, the anchor, the observability record and the token file;
(b) tests/hidden/oracle_token.sha256 holds the token sha256 the record gives (oracle_token_sha256);
(c) the anchor is one of the table's files, so (a) guards it;
(d) the record itself and the pre-registration are pinned here by sha256 (SELECTION_SHA256, PREREG_SHA256), so that
    the table cannot be edited silently: a deliberate, dated amendment or recorded change of the record changes
    SELECTION_SHA256 in this file in the same pull request, where it is visible;
(e) the files under tests/hidden/ are exactly the table's, the record and the pre-registration (no stray and no
    missing file).

File names and hashes only: no value of the hidden set appears here (the hashes of the hidden files are public in the
record already). Standard library only; nothing of the task is imported or run.
"""
import hashlib
import json
import re
import unittest
from pathlib import Path, PurePosixPath

TASK = Path(__file__).resolve().parents[1] / "tasks/ground/racecar_time_trial_on_unseen_track"
HIDDEN = TASK / "tests/hidden"
RECORD = "tests/hidden/selection.json"
PREREG = "tests/hidden/selection.prereg.json"
TOKEN = "tests/hidden/oracle_token.sha256"
ANCHOR = "tests/anchor.json"
OBSERVABILITY = "tests/review/observability.json"

#: (d) the completed selection record, as merged (benchmark pull request #104)
SELECTION_SHA256 = "93416b3e4ce488653b73200d883a44f06914cbc42721562f8ac68fffa9e07a31"
#: (d) the pre-registration as it stands: the record's prereg entry gives its sha256 at registration, and the record's
#: recorded change 2 (2026-11-18, path strings and machine names only) gives this sha256 since
PREREG_SHA256 = "f8a8327b25d1c690a377e882cb0032dc04c43a2b55462b8b2622402939ee6c9b"

_HEX64 = re.compile(r"[0-9a-f]{64}")


def sha256(rel):
    return hashlib.sha256((TASK / rel).read_bytes()).hexdigest()


def record():
    try:
        return json.loads((TASK / RECORD).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise AssertionError(f"{RECORD} cannot be read: {exc}")


def table():
    return record()["frozen_after_selection_sha256"]


def prereg_sha256_in_force(rec):
    """The pre-registration's sha256 as the record last states it (a recorded change names the file's new sha256)."""
    now = rec["prereg"]["sha256"]
    for change in rec.get("recorded_changes", []):
        if "preregistration" in change:
            now = change["preregistration"]["sha256_now"]
    return now


class HiddenSetHashes(unittest.TestCase):
    def assert_sha256(self, rel, want):
        self.assertTrue((TASK / rel).is_file(), f"{rel} is missing")
        self.assertEqual(sha256(rel), want, f"{rel} differs from its pinned sha256")

    def test_a_every_file_of_the_frozen_table_has_its_sha256(self):
        frozen = table()
        self.assertTrue(frozen, "the record's frozen_after_selection_sha256 table is empty")
        for rel, want in sorted(frozen.items()):
            with self.subTest(rel):
                path = PurePosixPath(rel)
                self.assertTrue(path.parts[0] == "tests" and ".." not in path.parts and not path.is_absolute(), rel)
                self.assertRegex(want, _HEX64, rel)
                self.assert_sha256(rel, want)

    def test_a_the_table_holds_what_item_8_names(self):
        rec = record()
        ids = [ent["world_id"] for ent in rec["selected"]]
        self.assertEqual(len(ids), 6, "item 8 names six world files and six reference records")
        self.assertEqual(len(set(ids)), len(ids), "a venue is selected twice")
        want = {ANCHOR, OBSERVABILITY, TOKEN}
        want |= {f"tests/hidden/{wid}.json" for wid in ids}
        want |= {f"tests/hidden/reference/{wid}.json" for wid in ids}
        self.assertEqual(set(rec["frozen_after_selection_sha256"]), want)

    def test_b_the_token_file_holds_the_records_token_sha256(self):
        want = record()["oracle_token_sha256"]
        self.assertRegex(want, _HEX64, "the record's oracle_token_sha256")
        self.assertTrue((TASK / TOKEN).is_file(), f"{TOKEN} is missing")
        self.assertEqual((TASK / TOKEN).read_bytes(), (want + "\n").encode("ascii"),
                         f"{TOKEN} does not hold the record's token sha256")

    def test_c_the_anchor_is_in_the_table(self):
        self.assertIn(ANCHOR, table(), f"the record's table does not freeze {ANCHOR}")

    def test_d_the_record_and_the_preregistration_are_pinned(self):
        self.assert_sha256(RECORD, SELECTION_SHA256)
        self.assert_sha256(PREREG, PREREG_SHA256)
        self.assertEqual(prereg_sha256_in_force(record()), PREREG_SHA256, "the record names another pre-registration")

    def test_e_the_hidden_folder_holds_exactly_the_expected_files(self):
        want = {rel for rel in table() if rel.startswith("tests/hidden/")} | {RECORD, PREREG}
        have = {p.relative_to(TASK).as_posix() for p in HIDDEN.rglob("*") if p.is_file() or p.is_symlink()}
        self.assertEqual(sorted(have - want), [], "stray file(s) under tests/hidden/")
        self.assertEqual(sorted(want - have), [], "missing file(s) under tests/hidden/")


if __name__ == "__main__":
    unittest.main()
