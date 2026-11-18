"""plantcell: the simulated plant building and the inspection robot of quadruped_hazardous_plant_inspection (the
grader's cell; agents get only plantcell.client).

Importing the package puts the cell's vendored library (_lib/, written by the maintainers' assembly tool: the
mission world, facade, evidence and scoring, the instruments and sensors, the site loader and the gait firmware) on
sys.path. The cell's data (the acceptance file, the robot model, the cell configuration) are in data/.
"""

import json
import os
import sys

CELL_VERSION = "2.0.0-dev"
__version__ = CELL_VERSION

HERE = os.path.dirname(os.path.abspath(__file__))
LIB_DIR = os.path.join(HERE, "_lib")
DATA_DIR = os.path.join(HERE, "data")


def _setup():
    layout = os.path.join(LIB_DIR, "LAYOUT.json")
    if not os.path.isfile(layout):
        return
    with open(layout) as f:
        roots = json.load(f).get("import_roots", ["."])
    for rel in reversed(roots):
        p = os.path.normpath(os.path.join(LIB_DIR, rel))
        if p not in sys.path:
            sys.path.insert(0, p)


_setup()
