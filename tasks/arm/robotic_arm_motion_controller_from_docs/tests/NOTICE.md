# NOTICE (grader-side; never staged into the agent's home)

The arm of this task is the design `H_asm2` (the arm ground-truth generator gt-arm/1.1.0, family `arm_family_v1`; the
arm the published task `robotic_arm_full_assembly_so101_panda` builds), a modified derivative of the SO-ARM101
follower arm by TheRobotStudio / Hugging Face LeRobot (https://github.com/TheRobotStudio/SO-ARM100, commit
5f6d2b876a53a4872e405b991dd925556c9e38a4, Apache-2.0): its printed parts are B-rep stretches, pocket spacers and
cut-outs of the repository's STEP parts, re-levelled in the generator's frame. Its MuJoCo description (tests/design/robot.xml,
the grader's model) follows the layout of MuJoCo Menagerie `robotstudio_so101` (commit c96a32d, Apache-2.0). The servo
is a parametric stand-in of the Feetech STS3215 class, named `BUS-SERVO-12V-A`. Modification statement
(Apache-2.0 section 4(b)): every unit of an acceptance run is the design with this task's unit layers
(`arm/motion_ctl/unit.py`: installation and calibration-record error, joint-frame errors, axis tilts, axis-distance
scales, wrist-to-tip offset, mass spread, a series compliance with a hysteresis band behind each servo's encoder, servo
gain and friction spread) and the generator's servo firmware layer (`arm/gt_arm/servo.py`, extended with
the goal-speed and acceleration registers in `arm/motion_ctl/plant.py`), all generated from numbers. The desktop cells
(`arm/motion_ctl/cells.py`) are primitives generated from numbers. CITATION: Knight, Kooijmans, Cadene et al.,
"Standard Open SO-100 & SO-101 Arms", 2024.

The documents (kinematics sheet, drawings, mass table, datasheet, bus manual, record format, cell documentation,
acceptance specification) are written for this task from the design's numbers and the stated tolerances.

Per the owner's rule the givens, the instruction and the public card never name the arm, its repositories, the model
collection or the servo vendor; this file carries the attribution.
