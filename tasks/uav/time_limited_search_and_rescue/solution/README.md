# Reference solution

`solve.sh` writes the reference submission: `run.sh`, `pilot.py` and the oracle token.

`oracle/pilot.py` is a privileged pilot. When the submission's `.oracle_token` hashes to `tests/oracle_token.sha256`,
the grader starts every graded site's service with a `_gt` block in its answers: the aircraft's true pose and
velocity, the litter's pose and the site's layout (grid, open doorways, hoist wells, the litter's room, the flange
size, the tool stands, the receiving stations and which one is open). The pilot plans a route on that, closes its
position loop on the true pose and reads no image. It completes every graded site in about 450 of the 4,400 actions.

It shows that each graded site can be completed inside the stated budgets with the stated aircraft. It is not a
solution an agent could write (an agent's program never receives `_gt`), and no score band was derived from an
unprivileged pilot.

`PILOT_MUTANT=<name>` degrades one part of the plan; the review's score table uses it (`tests/review/mutants.md`).
