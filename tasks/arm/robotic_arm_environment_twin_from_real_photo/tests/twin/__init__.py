"""Hidden grader modules of the environment-twin task (numpy, scipy, mujoco).

part 1: load (snapshot -> gated model), solids, align, match, score_c1, score_c2, score_c3, score_c6, part1 (runs
them in order). The public gate, camera and simulation modules are imported from the payload mirror
(tests/setup_payload, or $TWIN_PAYLOAD_DIR) by load.payload(); there is no other fallback.
"""
