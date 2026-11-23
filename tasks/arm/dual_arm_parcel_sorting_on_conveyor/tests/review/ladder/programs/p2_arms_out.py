"""Degenerate program 2 (hold the arms out over the belt): one arm after the other, each hand is moved into the
parcels' path in the work window (tool down, about 6 cm above the belt, fingers closed) and both are held there to
the shift's end. Expected shift score 0: parcels jam against the hands or are turned by them.

The two postures were taught offline on the given robot and station models (IK, straight joint paths from the start
posture checked free of contact), as a commissioning engineer teaches points."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _reach import go  # noqa: E402

WIRE_RUN = "pending: run through the full graded path (verify) once the packaging exists"
RIGHT_IN = [-0.7842, 2.1067, 0.7271, 1.9460, -1.9143, -1.5400, -1.1825, -1.4767, -2.6183, 1.4518, 1.0210, -1.0042, -1.5500, 0.4595]
BOTH_IN = [1.1889, 2.5537, -0.7313, 1.4502, -0.3383, 0.1863, -1.0235, -1.4767, -2.6183, 1.4518, 1.0210, -1.0042, -1.5500, 0.4595]


def solve(cell):
    info = cell.describe()
    go(cell, RIGHT_IN, grip=[1.0, 1.0], ticks=60)
    st = go(cell, BOTH_IN, grip=[1.0, 1.0], ticks=60)
    n = info["budgets"]["max_ticks_per_step"]
    while not st["shift_over"]:
        st = cell.step(n=n)
