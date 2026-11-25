"""The verify stage's process kit (grader side, stdlib only): how the program under test is run and served.

  * `wire`: the JSON-lines RPC between the program (caller) and the grader's ShoeCellServer (server), with the
    program-time cap;
  * `runner`: the run regime of the program (account, per-run directory, minimal environment, launcher, clean-up);
  * `lockdown`: the verify-start lockdown (kill the agent account, snapshot the submission, late-write gate);
  * `verify_freeze`: keeps the agent account stopped until the verdict is written;
  * `netaudit`: the static and runtime network audit of the program;
  * `nonet`: the no-network seccomp filter every process of the program runs under (v0.5), and its self-test;
  * `seeds`: the hidden episode seeds (v0.5: the fixed hidden set pinned in tests/hidden/seeds.json).

Adapted from the arm assembly task's copies (robotic_arm_full_assembly_so101_panda) by way of
furniture_assembly_from_drawings (the wire's program-time cap, the hardened framing and reply deadline).
"""
