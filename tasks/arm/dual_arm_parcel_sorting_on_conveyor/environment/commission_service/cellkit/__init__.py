"""The verify stage's process kit (grader side, standard library only): how the program under test is run and served.

  * `wire`: the JSON-lines RPC between the program (caller) and the grader's ParcelCellServer (server), with the
    per-shift program-time budget, the silence cap and the shift's end reasons;
  * `runner`: the run regime of the program (account, per-run directory, minimal environment, launcher, clean-up);
  * `lockdown`: the verify-start lockdown (kill the agent account, snapshot the submission, late-write gate);
  * `verify_freeze`: keeps the agent account stopped until the verdict is written;
  * `netaudit`: the static and runtime network audit of the program;
  * `nonet`: the no-network seccomp filter every process of the program runs under, and its self-test;
  * `seeds`: the hidden shifts' seed list (the fixed hidden set pinned in tests/hidden/seeds.json).

Adapted from the benchmark's shared verify kit: the wire gained the per-shift budget in proportion to the shift's
duration, a separate start window, a request-line guard and this task's end reasons.
"""
