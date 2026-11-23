"""dual_arm_parcel_sorting_on_conveyor: the salts of the development shifts and of the two hidden sets.

The task's maintainer notes (tests/README.md, section 6) say that the development shifts
(environment/commission_service/sites.json) share no seed, salt or label with either hidden set, and that the
repository test of this task checks it. test_dual_arm_parcel_sorting_on_conveyor_commissioning.py checks the seeds,
the labels and the package digests; this file checks the salts, and nothing else:

(a) the three salts (sites.json, tests/hidden/selection.json for the rated set, tests/hidden/line/selection.json for
    the graded line set; each `salt_hex`) are 16 bytes each and pairwise different;
(b) neither hidden set's salt appears in what the agent's image and sandbox are built from (environment/ and
    instruction.md), as hex in either case or as its 16 raw bytes.

No value of a hidden set is written here. Standard library only; nothing of the task is imported or run.
"""

import json
import os
import re
import unittest
from pathlib import Path

TASK = Path(os.environ.get("ARB_TASK_DIR") or (Path(__file__).resolve().parents[1] / "tasks/arm/dual_arm_parcel_sorting_on_conveyor"))
SITES = "environment/commission_service/sites.json"
RATED = "tests/hidden/selection.json"
LINE = "tests/hidden/line/selection.json"

_HEX32 = re.compile(r"[0-9a-f]{32}")


def salt(rel):
    try:
        value = json.loads((TASK / rel).read_text(encoding="utf-8"))["salt_hex"]
    except (OSError, ValueError, KeyError) as exc:
        raise AssertionError(f"{rel}: no salt_hex ({exc})")
    if not isinstance(value, str) or not _HEX32.fullmatch(value):
        raise AssertionError(f"{rel}: salt_hex is not 16 bytes of lower-case hex")
    return value


def agent_side_files():
    yield "instruction.md", (TASK / "instruction.md").read_bytes()
    for dp, dns, fns in os.walk(TASK / "environment"):
        dns[:] = [d for d in dns if d != "__pycache__"]
        for fn in fns:
            p = Path(dp, fn)
            yield p.relative_to(TASK).as_posix(), p.read_bytes()


class Salts(unittest.TestCase):
    def test_a_the_three_salts_differ(self):
        dev, rated, line = salt(SITES), salt(RATED), salt(LINE)
        self.assertNotEqual(dev, rated, "the development shifts share the rated set's salt")
        self.assertNotEqual(dev, line, "the development shifts share the line set's salt")
        self.assertNotEqual(rated, line, "the two hidden sets share a salt")

    def test_b_no_hidden_salt_on_the_agents_side(self):
        needles = []
        for rel in (RATED, LINE):
            s = salt(rel)
            needles += [(rel, s.encode()), (rel, s.upper().encode()), (rel, bytes.fromhex(s))]
        for name, data in agent_side_files():
            for rel, needle in needles:
                self.assertNotIn(needle, data, f"{name} holds the salt of {rel}")


if __name__ == "__main__":
    unittest.main()
