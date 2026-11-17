"""The verify stage's process kit (grader side, stdlib only): how the program under test is run and served.

  * `wire`: the JSON-lines RPC between the program (caller) and the grader's CellServer (server);
  * `runner`: the run regime of the program (account, per-run directory, minimal environment, launcher, clean-up);
  * `lockdown`: the verify-start lockdown (kill the agent account, snapshot the submission, late-write gate);
  * `verify_freeze`: keeps the agent account stopped until the verdict is written;
  * `netaudit`: the static and runtime network audit of the program.

Adapted from the arm assembly tasks' copies (robotic_arm_full_assembly_so101_panda: arm/assembly_cell/{wire,runner}.py,
arm/assembly_six/lockdown.py, arm/assembly_info/netaudit.py, arm/verify_freeze.py). This copy is
furniture_assembly_from_drawings's (tests/cellkit at 149ea28) unchanged, except the launcher's client package and
program module (runner.BOOTSTRAP: house_client, clean).
"""
