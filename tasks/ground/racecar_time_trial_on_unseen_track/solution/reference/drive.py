"""The privileged reference program: entry point `solve(car)` like any submission.

The 64-hex-digit token comes from the file `oracle_token` next to this file (written into the run's copy of the
program by the maintainer's runner, never kept in the repository) or from the `token` argument of an in-process run.
The program's statistics go to stderr as one JSON line (`S7_STATS {...}`).
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.append(HERE)

import s7ref  # noqa: E402

LAST_STATS = {}


def read_token(path=None):
    with open(path or os.path.join(HERE, "oracle_token"), "r", encoding="ascii") as f:
        return f.read().strip()


def solve(car, token=None, params=None):
    stats = s7ref.run(car, token or read_token(), params=params)
    LAST_STATS.clear()
    LAST_STATS.update(stats)
    print("S7_STATS " + json.dumps(stats, sort_keys=True, default=str), file=sys.stderr)
    return stats
