"""Grader of freeflyer_cad_design_from_photos: one extraction of the submitted FreeCAD document (fcstd_extract.py's
output: model.json + one world-frame BREP per solid) against the maker's published appearance meshes and robot
description (reference/, read by makerstd/), in millimetres. Root, verify stage, agent gone; reads data only.

    python3 -B grade.py EXTRACT_DIR OUT_JSON [--config grader_config.json] [--stage DIR]

The rules are in cadgrade/run.py's docstring (the instruction states them); every number is in grader_config.json.
OUT_JSON: {"ok", "design_score", "criteria", "floor", "checks", "zero_case", "tags", "hinges", "structure",
"alignment", "statistics", "diagnostics", "timing"}, or {"ok": false, "task_error", "error"} when the grader itself
fails (verify.py then fails the run as a task error, never a silent 0).
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from cadgrade.run import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main(sys.argv))
