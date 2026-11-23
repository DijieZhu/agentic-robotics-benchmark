# Author solution (not part of the participant sandbox)

`solve.sh` installs the privileged oracle (`oracle/`) as the submission. It holds a token whose sha256 only the
grader knows (`tests/hidden/oracle_token.sha256`), so the graded robot service answers its `truth` call: the
site as built (racks, openings, both fittings with their clearances, the dock), the Go1's launch pose and the
site's physical properties. It plans its route on that truth and walks it with the robot's own commands: an
analytical trot from twelve joint targets, with the stance height and the trunk's pitch held by feedback on the
robot's joint angles and gravity reading, low under the duct and high over the deck's guides.

It shows that every graded site can be completed; it is not a participant program. No participant program can
obtain the truth: the commissioning service has no such call, and a graded service started without the token's
hash (or asked with any other token) answers it like an unknown call.

The earlier reference that used only the public observations (UAV survey, cap registration, a nominal gait) is
kept as review material in `tests/review/public_reference/`; it assumes the nominal load and clearances and is
one of the review's controls (`tests/review/README.md`).
