"""shoecell: the shoe lacing and boxing work cell (PUBLIC CODE: the agent's development copy and the grader run the
same package; only the seeds, the hidden reference value and the reference program differ).

Modules
    client      the agent-side proxy of the cell and the program launcher (JSON-lines wire on stdio)
    physics     scene builders (bench, arms, shoes, laces, carton, cradle) and the compiled MuJoCo model
    control     the per-arm joint / Cartesian controller (damped least-squares differential IK)
    cameras     RGB-D rendering, intrinsics and extrinsics, the depth sensor model
    episode     per-episode randomisation, arrivals, faults, milestones and the work score
    server      the cell API served over the wire (public calls; privileged calls for the reference program)
"""

CELL_VERSION = "shoe-lacing-and-boxing-cell/0.1.0"
