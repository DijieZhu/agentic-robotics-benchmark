"""cabincell: the grader's simulation cell of freeflyer_uncrewed_station_caretaking (the hidden side; agents get only the
client library ``cabinkit``).

One run = one site package + one program.  The cell builds the station scene of the site (station library, the stowage
units, the loose items, the dock) with the reference robot (asset root), wires it (suction at 50 Hz, item aerodynamics
and the robot's jets, the station model and service, the lighting richness of the localisation, the dock), puts the
task firmware on the robot and answers the program's calls; it records the world state for the grader.

Modules:
  ``server``          CabinCellServer: site -> cell, the call boundary (whitelist, forbidden count), the run's end
                      (firmware stop + settle), the record; in-process and wire drivers
  ``api``             the calls of the robot API on top of the task firmware (TIME_UP, budgets, camera frames and
                      the evidence predicate's inputs, report, mission, station calls)
  ``budgets``         the per-site budgets of the runtime site_spec (cell.budget.*)
  ``events``          the recorder: world state every cell.record_period_s, contact events at every physics step,
                      keep-out time, docking, the record and its hashes
  ``wire_guard``      the guard on the program's request lines (cellkit.wire.AgentProcess subclass)
  ``firmware``        the task firmware (imported as the top-level package ``firmware``)
  ``site``            the site parts and the scene builder (``cabincell.site``)
  ``station_service``, ``telemetry_synth``  the station side
  ``predicates``, ``grade``                 the grader; the cell measures the record with predicates.py
  ``assetroot``       the verified asset root of the robot and station meshes

Importing the package puts the vendored libraries (``_lib``: spacekit 0.4.0 and the station library) and the cell
directory (``firmware``, ``assetroot``) on sys.path.  Standard library only at import time.
"""
from __future__ import annotations

import os
import sys

CELL_VERSION = "1.0.0"
__version__ = CELL_VERSION

HERE = os.path.dirname(os.path.abspath(__file__))
LIB_DIR = os.path.join(HERE, "_lib")
DATA_DIR = os.path.join(HERE, "data")
TESTS_DIR = os.path.dirname(HERE)

for _p in (HERE, LIB_DIR):                      # LIB first: the vendored spacekit / station win
    if _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)
if TESTS_DIR not in sys.path:
    sys.path.append(TESTS_DIR)
