"""The site generator's robot-free constant-force fixture (the hold-force rule of the sites): a blocker
candidate on a framed-inlet bench, a constant force at one of its handles for 40 s, removed or not.

  ``bench``    the bench (flat wall, one return-inlet frame, the unit's fan network, the soft sheet, pull directions)
  ``fixture``  the tier conditions, one candidate's fixture record (station_site/2 ``fixture_records.blocker``),
               candidate draws, the command line

Review tool of the hidden side (``tests/review/``): the cell does not import it; the generator does.
``tests/`` and ``tests/cabincell/_lib`` are put on ``sys.path`` here (``cabincell.site`` for the item families and the
placement draws).
"""
from __future__ import annotations

import os
import sys

VERSION = "1.0.0"
HERE = os.path.dirname(os.path.abspath(__file__))
TESTS = os.path.normpath(os.path.join(HERE, "..", "..", ".."))
LIB = os.path.join(TESTS, "cabincell", "_lib")
for _p in (LIB, TESTS):
    if _p not in sys.path:
        sys.path.insert(0, _p)
