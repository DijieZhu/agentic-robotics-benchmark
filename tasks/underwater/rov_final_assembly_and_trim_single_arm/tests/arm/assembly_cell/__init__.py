"""assembly_cell: the grader-owned assembly cell of the arm chain's assembly stage (CHAIN_PLAN 4.3).

    layout.py       seed -> hidden layout (tray slot assignment, jitter, fixture jitter, friction)
    cell_builder.py A2 givens + model collection Panda + layout -> MjSpec / MjModel of the cell (collision-model gate)
    predicates.py   sleeve_seat / part_seat / horn_mate predicates, latch, snapped welds
    controller.py   Panda joint / Cartesian (DLS) position control, gripper, rate limits, stiffness scale
    server.py       CellServer: owns MjModel / MjData, the API (describe, get_state, set_control, command,
                    step, get_camera, finish), safety monitor, held detection, assembly record
    functional.py   the functional-articulation check (powers the seated servos, drives hidden configurations)
    wire.py         JSON-lines RPC (agent = caller, grader = server) and the unprivileged spawn
    client.py       Cell proxy for assemble.py::solve(cell) + the launcher (python -m assembly_cell.client)
    grading.py      per-episode metrics -> criteria (public formulas; the practice grader uses the same code)

Units on the API: metres / radians / quaternions wxyz (MuJoCo order); the A2 files are mm.
"""

CELL_VERSION = "arm-assembly-cell/1.0.0"
