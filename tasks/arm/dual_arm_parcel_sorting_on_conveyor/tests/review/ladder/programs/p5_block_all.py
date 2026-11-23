"""Degenerate program 5 (block every parcel): the right hand, fingers closed and the tool down, is moved to about 3 cm
above the belt across the parcels' path upstream in the work window and held there: a dam that no parcel passes.
Expected shift score 0: parcels pile up against it and never reach the scan line (lost).

The posture was taught offline on the given robot and station models (IK, a straight joint path from the start
posture checked free of contact)."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _reach import go  # noqa: E402

WIRE_RUN = "pending: run through the full graded path (verify) once the packaging exists"
DAM = [-0.7842, 2.1067, 0.7271, 1.9460, -1.9143, -1.5400, -1.1825, -1.1647, -2.9494, -0.2029, 1.1112, -0.5340, -0.7401, -0.4997]


def solve(cell):
    info = cell.describe()
    st = go(cell, DAM, grip=[0.0, 1.0], ticks=60)
    n = info["budgets"]["max_ticks_per_step"]
    while not st["shift_over"]:
        st = cell.step(n=n)
