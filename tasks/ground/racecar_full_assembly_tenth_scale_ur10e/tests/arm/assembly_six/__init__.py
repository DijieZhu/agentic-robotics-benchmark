"""assembly_six: the assembly cell of the six-servo desktop arm (assembly sibling task
`arm_assemble_six_servo_arm_panda`, CHAIN_PLAN 4.3 "different arm").

Own namespace of the six-servo assembly lane (nothing under `arm/assembly_cell/` is edited; the generic
modules `predicates`, `controller`, `wire`, `client`, `runner`, `rungs`, `grading` are imported from there):

    source.py        the arm's public CAD source (named only there; NOT vendored into a task) -> a neutral in-memory
                     description of the arm: links, joints, printed parts, six identical servos
    derive.py        (not vendored) deterministic builders of the bank layers: hardware-design/ (parts/*.stl in the link
                     frames, assembly.json in the chain schema) and derived/a2_assembly/ (parts_manifest.json,
                     mating_graph.json with the `elbow` and `full` scopes, visual + CoACD collision meshes)
    layout.py        seed -> hidden layout (trays, jitter, fixture, friction) for THIS arm's parts and the
                     pre-assembled groups
    cell_builder.py  MjSpec build of the cell: fixture + pre-assembled proximal, loose parts, pre-assembled
                     distal compound, six-servo horn hinges at the servo's own horn axis, blue printed parts
    functional.py    the functional-articulation check with a pre-assembled joint prefix
    server.py        CellServer subclass (this arm's builder, layout, describe, damage watch)
    lockdown.py      verify-start lockdown helpers (stdlib; grader side only)
    local.py         LocalCell for practice / QC

The public API on the wire, the predicates, the latch and the grading are the shared cell's.
"""

CELL_VERSION = "arm-assembly-cell-six/1.0.0"
