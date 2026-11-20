"""Site parts of the caretaking cell (hidden side): the loose items, the scene of one site description,
the natural placement and its measures.

Shared by the cell and the site generator: both build a site's scene with ``build.build`` and measure the
items with ``build.measure``, so the generator's settled state and the cell's state are states of the same model.

Modules:
  ``spec``       the runtime copy of the task's site_spec (``tests/cabincell/data/site_spec.json``, sha256 self-check)
  ``items``      the three soft-sheet families and the tool pouch: register classes, size / mass ranges, registered
                 colours, station_site/2 item entries
  ``placement``  natural placement of a blocker, r_face ranges by unit size, the analytic replica of the suction
                 sampler's sealed area (bit-identical to the scene's sampler at seal 1), the exact geometric overlap,
                 the unit network's hold, joint (placement, r_face) sampling, pouches at an opening end by hold
  ``build``      station_site/2 -> the station builder's site (stowage units, pad plate, unit flows), the scene, the
                 robot-free wiring, in-scene measures (sealed area and hold per item), settling

Import as ``cabincell.site`` (``tests/`` on ``sys.path``) -- not as a top-level ``site`` (the standard library's).
The station library and the vendored simulation package (``tests/cabincell/_lib``) are put on ``sys.path`` here.
"""
from __future__ import annotations

import os
import sys

VERSION = "1.0.0"
HERE = os.path.dirname(os.path.abspath(__file__))
CELL = os.path.dirname(HERE)
LIB = os.path.join(CELL, "_lib")
if LIB not in sys.path:
    sys.path.insert(0, LIB)
