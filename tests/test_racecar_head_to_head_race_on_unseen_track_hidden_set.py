"""racecar_head_to_head_race_on_unseen_track: the repository-level test that guards the hashes of the hidden set.

The hidden set was drawn by a pre-registered selection rule (tests/hidden/selection.prereg.json; the same rule texts
are quoted in tests/hidden/selection.json under rule_texts). Its item 10, FROZEN AFTER SELECTION, says that the six
world files, the six reference records, the anchor, the observability record, the completed selection record and the
sha256 of the reference's token are frozen by sha256, and that "a repository-level test guards the hashes". This is
that test, and it checks nothing else:

(a) every file of the record's frozen_after_selection_sha256 table exists and has the sha256 the table gives; the
    table is read from the record, so the test follows the record, and the table holds the world file and the
    reference record of every venue the record selected and the observability record;
(b) tests/hidden/oracle_token.sha256 holds the token sha256 the record gives (oracle_token_sha256);
(c) the pre-registration announced the anchor (tests/anchor.json) as frozen, the record's table does not list it, and
    this test pins it instead (ANCHOR_SHA256);
(d) the record itself and the pre-registration are pinned here by sha256 (SELECTION_SHA256, PREREG_SHA256), so that
    the table cannot be edited silently: a deliberate, dated amendment of the record changes SELECTION_SHA256 in this
    file in the same pull request, where it is visible;
(e) the files under tests/hidden/ are exactly the table's, the token file, the record and the pre-registration (no
    stray and no missing file).

File names and hashes only: no value of the hidden set appears here (the hashes of the hidden files are public in the
record already). Standard library only; nothing of the task is imported or run.
"""
import hashlib
import json
import re
import unittest
from pathlib import Path, PurePosixPath

TASK = Path(__file__).resolve().parents[1] / "tasks/ground/racecar_head_to_head_race_on_unseen_track"
HIDDEN = TASK / "tests/hidden"
RECORD = "tests/hidden/selection.json"
PREREG = "tests/hidden/selection.prereg.json"
TOKEN = "tests/hidden/oracle_token.sha256"
ANCHOR = "tests/anchor.json"
OBSERVABILITY = "tests/review/observability.json"

#: (d) the completed selection record, as merged (benchmark pull request #142)
SELECTION_SHA256 = "d18e883631eac788a4ffba09998126a6af0039a6a960d9f837e785c3831f3c0e"
#: (d) the pre-registration, the part of the record written before the first candidate venue (the record's prereg
#: entry gives the same sha256)
PREREG_SHA256 = "2722c5133df5f28fd4f2e3a87a3d55cdf8a94791d74e61b1f61c4f4f7522e565"
#: (c) the anchor: frozen by the pre-registration's item 10, not in the record's table, pinned here
ANCHOR_SHA256 = "91e89d14e52086884459795f28ea3c50ed94166fe8d8851a0411e280ec7ce799"

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

    def test_a_the_table_holds_the_selected_venues_and_the_observability_record(self):
        rec = record()
        ids = [ent["world_id"] for ent in rec["selected"]]
        self.assertEqual(len(ids), 6, "item 10 names six world files and six reference records")
        self.assertEqual(len(set(ids)), len(ids), "a venue is selected twice")
        want = {OBSERVABILITY}
        want |= {f"tests/hidden/{wid}.json" for wid in ids}
        want |= {f"tests/hidden/reference/{wid}.json" for wid in ids}
        self.assertEqual(set(rec["frozen_after_selection_sha256"]), want)

    def test_b_the_token_file_holds_the_records_token_sha256(self):
        want = record()["oracle_token_sha256"]
        self.assertRegex(want, _HEX64, "the record's oracle_token_sha256")
        self.assertTrue((TASK / TOKEN).is_file(), f"{TOKEN} is missing")
        self.assertEqual((TASK / TOKEN).read_bytes(), (want + "\n").encode("ascii"),
                         f"{TOKEN} does not hold the record's token sha256")

    def test_c_the_anchor_is_pinned(self):
        self.assert_sha256(ANCHOR, ANCHOR_SHA256)

    def test_d_the_record_and_the_preregistration_are_pinned(self):
        self.assert_sha256(RECORD, SELECTION_SHA256)
        self.assert_sha256(PREREG, PREREG_SHA256)
        self.assertEqual(record()["prereg"]["sha256"], PREREG_SHA256, "the record names another pre-registration")

    def test_e_the_hidden_folder_holds_exactly_the_expected_files(self):
        want = {rel for rel in table() if rel.startswith("tests/hidden/")} | {TOKEN, RECORD, PREREG}
        have = {p.relative_to(TASK).as_posix() for p in HIDDEN.rglob("*") if p.is_file() or p.is_symlink()}
        self.assertEqual(sorted(have - want), [], "stray file(s) under tests/hidden/")
        self.assertEqual(sorted(want - have), [], "missing file(s) under tests/hidden/")


if __name__ == "__main__":
    unittest.main()
