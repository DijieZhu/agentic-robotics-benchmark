"""Hidden grader modules of the site-twin task (numpy, scipy, mujoco).

load (snapshot -> gated twin), truth (the hidden truth files), align (camera fit + structure refinement), surfaces
(unit solids in the site frame), match (surface F-score + Hungarian), zones, support (b_vert), raycast, geom2d,
registry (criteria + ctx), part1 (the geometric core), part2 (pipeline, self-test, worker), score_c1 .. score_c7.
The public gate, camera, simulation and slice modules are imported from the payload mirror (tests/setup_payload, or
$TWIN_PAYLOAD_DIR) by load.payload(); there is no other fallback.
"""
