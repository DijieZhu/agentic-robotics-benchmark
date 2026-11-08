# Fleet requirements

The fleet: **three flying robots** (`/World/Fleet/UAV_1`, `UAV_2`, `UAV_3`) and **one ground vehicle**
(`/World/Fleet/UGV`), built from the catalog (`catalog/`) and the libraries. Each requirement below is one or more
of the grader's checks; the numbers are the ones it uses.

**R1 — At rest where they belong.** At the start the flying robots stand on the three pads (UAV_1 on the pad nearest
the mat, UAV_2 on the pad behind it on the -x side, UAV_3 on the one on the +x side) and the vehicle in its bay, each
facing its marking's arrow (+y). Nothing moves until something acts on it; the ropes may settle.

**R2 — Each flying robot carries a manipulator** from the arm catalog (at least four driven joints and a gripper),
declared with `arb:arm_model`.

**R3 — The ropes and the rig.** Each flying robot carries one of the bucket's three ropes (`facts.md`: 1.50 m from
hook to ring, 0.30 kg/m), hooked to one of its bodies, its other end on one of the bucket's rings (rings at least 100°
apart around the rim), built as `interface.md` says: a chain of at least 8 rigid links. The grader puts the real
bay's rig through the same moves as yours, by holding the robots and moving them, and compares:

- slack, the rope sags like a rope (a 1.2 m chord of the 1.5 m rope hangs about as deep as the real one);
- taut, it is straight and 1.50 m long (±3 cm), and it stretches no more than 8 cm under any of the moves;
- the three robots rising together bring the bucket up level (bottom ≥ 0.40 m, tilt ≤ 5°);
- one robot moved 0.3 m outward shifts the bucket and tilts it as the real rig does (±4 cm, ±2°);
- stepped 0.4 m sideways, the bucket swings with the real rig's period (±8 %) and does not die out unnaturally
  (its swing after 10 s is between 25 % and 160 % of its first seconds');
- one robot lowered 0.8 m tips the bucket as the real one tips (±5°); over the pan, 0.8 m, it pours and the fire
  goes out (the hazard rules);
- a 1 m move in half a second and a dead stop leaves the bucket swinging, not flying (never faster than 4 m/s);
- lowered back to the floor it stands upright with the ropes slack.

**R4 — Take-off mass.** Every flying robot's total mass as simulated (airframe, arm, sensors, everything but the rope)
must be at least what the catalog says those parts weigh, and together with its share of the load (the bucket's
11.2 kg plus the three ropes, divided by three, times a dynamic factor of 1.3) must not exceed its airframe's maximum
take-off mass. Declare the airframe with `arb:airframe_model`.

**R5 — Fire watch.** Each flying robot carries a thermal camera from the catalog that sees the whole fire pan from
the water drop's approach hover: the robot at (2.60, -0.40, 2.50) m, oriented as it stands on its pad, the pan at
least 25 px across in the image.

**R6 — The ground vehicle** is a wheeled base from the catalog (declared with `arb:base_model`) carrying an arm from
the catalog (`arb:arm_model`) and its sensors; the arm's and the sensors' catalog masses must not exceed the base's
payload. Nothing fixes the vehicle to the floor; it stands on its wheels.

**R7 — Working the valve.** The arm's rated payload, acting at the lever's grip (0.22 m from the stem), must give at
least the valve's breakaway torque; and with the vehicle standing on the white square (oriented as parked) the arm
must be able to bring one of its links within 6 cm of the lever's grip.

**R8 — Reading the gauge.** A camera on the vehicle, as it stands parked in its bay, sees the pressure gauge's dial
from the front, at least 40 px across the dial.

**R9 — Honest sensors.** Every camera that counts is a `UsdGeom.Camera` with `arb:sensor_model` naming a catalog
entry, and the camera's focal length and apertures are that entry's (within 2 %). Its mass counts toward the robot's.
In the bay mission each camera that counts streams at its catalog resolution, up to four cameras and one lidar a robot
(`control.md`, The sensors).

**R10 — State.** Each robot's root prim carries `arb:disabled` (bool, false at the start); the hazard rules in
`bay_brief.md` drive it, with `arb:steam_active` on `/World/Facility/SteamLine` and `arb:fire_active` on
`/World/Facility/FirePan`, all written by the stage's own OmniGraph logic.

**R11 — Seeing each other.** Each flying robot carries a lidar from the catalog (declared with `arb:sensor_model`): in
the bay mission its scan is how the fleet measures where the others are; the robots' own positions drift
(`control.md`). Place every robot's sensors for what it must see: the others, the bucket below, the pan ahead, the
lever, the dial, the tables' tops and the crate. A sensor rides on the body it is mounted on.

**R12 — The program.** The fleet does the whole job under your program, `control.md`: the valve closed, the fire
put out (a synchronised take-off, a level lift, a carry with little swing and little spill, the pour), every part off
the tables and into the crate; nothing lost to the hazards, nothing driven or flown into, the robots still at the end,
within the mission's time, and the same way every time, in the real bay (`control.md`).
