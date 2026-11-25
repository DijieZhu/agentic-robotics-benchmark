"""The privileged reference program of tensegrity_collapsed_warehouse_void_search, run inside the cell by
`reference_run` (the oracle token's call only; VoidCellServer.reference_run loads this file and calls run(server)).

    run(server) -> {"summary": {...}, "rows": [...], "events": [...]}

It drives the robot through the station's own calls (voidcell.client.connect_handler on the cell's handle: the
wire's JSON round trip, the same argument checks, budgets, records and time limit as an agent's program) and adds
the privileged truth for planning: the true node positions (pose and the bottom triple it passes), the site's true
occupancy (which mouth is open, the plug, every box). It never takes a camera frame, so its result does not depend
on the renderer. Its modules are in _lib/voidref/ (planner, controller, monitor, prim_table.json); nothing of them is
in the cell's folder or in the practice ground's copy of the cell.

The summary (kept by the cell's record as record["reference"]) carries the run's facts for the selection
predicates R1-R7 (the selection tools, not published): the inline score, the primitive counts (all, under the slab), stuck
and end-triangle events, the strut-axis and tendon penetration, the planning wall time.
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
LIB = os.path.join(HERE, "_lib")


def run(server, log=None):
    if LIB not in sys.path:
        sys.path.insert(0, LIB)
    from voidcell.client import connect_handler
    from voidref.controller import Reference
    station = connect_handler(server.handle)
    return Reference(server, station, log=log).run()
