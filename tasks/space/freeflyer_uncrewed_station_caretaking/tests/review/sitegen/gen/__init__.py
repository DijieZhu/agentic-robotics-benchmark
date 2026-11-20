"""The site generator of freeflyer_uncrewed_station_caretaking (hidden side, review only; the site
rules; the site package format).

One candidate of a cell (module x tier of the hidden-set rule) is the first *attempt* drawn from its seed stream that passes
every generation condition; attempt j of candidate k draws from ``sha256(label / <M>x<T> / k / j)`` (the candidate's
own seed is ``sha256(label / <M>x<T> / k)``, site_spec ``layout.seed_rule``).  An attempt goes through stages that
are ordered by cost, so most rejections are cheap:

  A  draw     the description: units, the blocker (natural placement + r_face jointly), the pouches, the stowage
              units, lighting, robot, inter-module flows, the lost-item register; network / replica windows; a
              telemetry pre-draw on the replica's sealed areas                                   (``draw``, ``register``)
  B  screen   the scene at t = 0: the station part of check_site and the pre-grasp reach on the built poses
  C  fixture  the robot-free 40 s constant-force fixture (``review.sitegen.fixture``)
  D  settle   600 s robot-free settle; the settled state (``state0``); in-scene sealed areas and holds in the windows
  E  telemetry the 72 h log, configuration record and advisory log on the in-scene areas (``telemetry_synth``)
  F  check    check_site on the settled state: station part + pre-grasp reach + pouch handle / end rules +
              stowage-unit siting + the advisory arrangement (+-3 sigma)                (``checks``)
  G  dry run  2400 s robot-free run from state0 in a fresh rig: sheet sealed area change < 1 %, displacement <= 3 mm;
              pouch displacement <= 1 mm                                                      (``settle``)
  H  pack     the deterministic site package ``<site_id>.site.tar.xz``                     (``mission``, ``sitepack``)

Modules: ``cells`` (labels, seeds, the cell table), ``draw``, ``register``, ``pregrasp``, ``checks``, ``settle``,
``mission``, ``sitepack`` (the site package writer), ``pipeline`` (stages, cost accounting, the
command line), ``freeze`` (the generator's sha256 over its input set).

Every number is read by key from the runtime site_spec (``tests/cabincell/data/site_spec.json``, sha256 self-check by
``cabincell.site.spec``); constants that are generator design choices are marked [C] where they are defined.
``tests/``, ``tests/cabincell`` and ``tests/cabincell/_lib`` are put on ``sys.path`` here.
"""
from __future__ import annotations

import os
import sys

VERSION = "1.0.0"
HERE = os.path.dirname(os.path.abspath(__file__))
TESTS = os.path.normpath(os.path.join(HERE, "..", "..", ".."))
CELL = os.path.join(TESTS, "cabincell")
LIB = os.path.join(CELL, "_lib")
for _p in (CELL, LIB, TESTS):
    if _p not in sys.path:
        sys.path.insert(0, _p)
