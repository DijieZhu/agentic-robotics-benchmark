"""practice_measure.py -- the rehearsal service's own measurement table for the cell.

The cell builds its recorder with ``predicates.measure_params(config['values'])`` and takes ``config`` from
``cabincell.grade.load_config()`` when it is given none (tests/cabincell/server.py).  The rehearsal service's copy of
the cell has no grade.py and no grader configuration (the rehearsal service never reads the
grader's measurement keys).  The service passes this table to every cell it builds and registers this module under
the name ``cabincell.grade`` before the cell is imported, so that the copy's ``from . import grade`` resolves to it.

The values are the SIMULATION's own definitions, not the grader's:
  * seal band: the suction model's seal box (spacekit.suction.SuctionFaces: no seal beyond a 12 mm gap, the value
    cabincell.site.placement.G_MAX mirrors) -- an item counts as lying on a face while the face's suction can act on it;
  * covered fraction of an inlet: the same band over the opening area (runtime site_spec world.inlet.opening_area_m2);
  * finger cage: the simulation package's own caged test of a closed hand (spacekit.primitives, the perch check:
    palm x in [-135, -30] mm, |y| <= 30 mm; the closing plane +-30 mm);
  * loss of control: kept only because the recorder needs a value; the rehearsal debrief does not use it.
The debrief (rehearsal_debrief.py) reports categories only (on which kind of face, in the hand, floating, elsewhere);
no area, threshold or distance of this table is ever returned.  Standard library only.
"""
from __future__ import annotations

import json
import os

SCHEMA = "caretaking.rehearsal_measure/1"
SEAL_GAP_MM = 12.0          # spacekit SuctionFaces g_max = 0.012 m (cabincell.site.placement.G_MAX)
CAGE_MM = {"x": [-135, -30], "y_abs_max": 30, "plane_max": 30}      # spacekit.primitives caged test
LOSS_UNUSED = {"disp_m": 0.10, "min_s": 5.0}                       # not used by the debrief


def _opening_area_m2() -> float:
    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "cabincell", "data", "site_spec.json"), "rb") as fh:
        v = json.loads(fh.read().decode("ascii"))["values"]
    return float(v["world.inlet.opening_area_m2"])


def values() -> dict:
    """The table under the key names the cell's predicates.measure_params reads."""
    return {"grader.seal.max_gap_mm": SEAL_GAP_MM,
            "grader.covered": {"max_gap_m": SEAL_GAP_MM / 1000.0, "opening_area_m2": _opening_area_m2()},
            "grader.in_gripper_cage_mm": dict(CAGE_MM),
            "grader.loss_of_control": dict(LOSS_UNUSED)}


def load_config(path=None) -> dict:
    """The cell's ``config`` (the shape of grade.load_config's result; ``path`` is ignored)."""
    return {"schema": SCHEMA, "values": values(), "sha256": None, "contract_version": None,
            "source": "practice_measure.py (the simulation's own definitions)"}
