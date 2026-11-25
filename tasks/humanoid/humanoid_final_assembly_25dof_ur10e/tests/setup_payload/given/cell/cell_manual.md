# Final-assembly station: integrator's description of the assembly cell (commissioning document)

This document describes the station as it was commissioned and as the grader's cell simulates it. Every number in it
is the cell's own. Where the cell behaves in a way a careful engineer would not expect, it says so.

## 1. Station scope

This station receives the pelvis-and-torso assembly (battery and covers fitted) clamped on its stand, and
five finished modules. It seats each module on its interface. Wiring, the interface screws themselves and
every later inspection belong to other stations; the cell fastens a module that is held on its seat (section
7). The 41 interface screws are not in the cell: they are not in the modules, not
on the drawings and not in the photographs.

The station's mates: each leg, with its hip-roll unit, goes with its flat bracket flange onto a hip output flange of
the base; each arm goes with its shoulder ring into a shoulder socket of the torso; the head goes with its bracket onto
the neck's output. Nothing in this document gives a seat's pose, a bolt position, a direction of insertion, an order
of work or a way of holding a module. The table says where each kind of fact can be read.

| what | where it can be read |
|---|---|
| what the finished robot looks like; which module belongs where | drawing 6; photographs 17 to 20, 14 and 16 |
| a leg's seat: the output flange's face, its axis and its place on the base | drawing 1 (views A and B, section D-D) and drawing 6; photograph 4 |
| a leg's bolt position | drawing 1 (views A, C and E and its table of angles); photographs 4, 8, 13 and 14 |
| an arm's seat: the socket's axis, faces, diameters and depths | drawing 2 (views A and B, section E-E), drawing 7 and drawing 6; photographs 2 and 5 |
| an arm's bolt position | drawing 2 (views A, D and F and its tables of angles) read together with the arm's delivered joint angles (`module.json`; the table of drawing 6); photographs 5, 9, 10, 15 and 16 |
| the head's seat and bolt position | drawing 3; photographs 6, 12 and 17 to 20 |
| the base on its stand, the stand in the cell | drawing 4; photographs 2 and 3; the stand's datum board |
| where each module rests and how it is turned there | drawing 5; photographs 1, 7, 9 and 11; the datum boards; `items` of `describe()` |
| the joint angles each module is delivered with | `module.json`; the table of drawing 6 |
| how the output rotors stand at the start | section 6; drawings 1 and 3 show the hole patterns at that clock |

## 2. Frames, units, datum boards

Units on the API and in the configuration: metres, radians, kilograms, newtons; quaternions w x y z. A key or a
table that names another unit is in that unit. The mesh files
are in millimetres; the drawings in millimetres and degrees.

The cell frame: its origin lies on the arm's base axis at the arm's mounting face (the top of its pedestal), z up;
the floor is at z = -0.85 m. The base faces the cell's +x; its left is the cell's +y. Nominally its
waist axis stands 0.75 m from the arm's base axis on the cell's -y side, and the soles of the
finished robot hang 0.1 m above the floor: the robot stays clamped at the pelvis and never
stands on the floor. The cell frame and the arm's base pose are exact.

The unit frame (the frame of a unit's mesh files, of its `module.json` and of its collision pieces): z up with the
unit upright as it is presented, x along the long side of the smallest rectangle that encloses its footprint, the
centre of mass on the +x side, the origin at that rectangle's centre at the height of the unit's lowest point. Every
body of a unit has this frame at the delivered joint angles. The frame is made from the unit's own shape as delivered:
it says nothing about where the unit sits on the base.

The drawings define their own datums (A, B and C), their sign rules, a board's corner and axes, and the order and
direction of the turns of a module's axes. This manual does not define them a second time: read them on the sheets
(`~/given/drawings/README.md`).

Datum boards: seven printed checker boards, each a flat plate with a border around the pattern.

| board | on | squares | square, mm | border, mm | moves |
|---|---|---|---|---|---|
| `floor_board` | the floor | 7 by 5 | 40 | 20 | fixed in the cell |
| `tool_board` | the rack table | 5 by 4 | 30 | 10 | fixed in the cell |
| `stand_board` | the stand's floor plate | 4 by 4 | 40 | 10 | with the stand |
| `leg_rack_board` | the leg rack's bench | 4 by 4 | 40 | 10 | with the leg rack |
| `arm_rack_R_board` | the right arm's rack | 4 by 4 | 30 | 10 | with its rack |
| `arm_rack_L_board` | the left arm's rack | 4 by 4 | 30 | 10 | with its rack |
| `head_nest_board` | the head's nest | 4 by 5 | 30 | 10 | with the nest |

`datums` of `describe()` repeats this table and gives, for the boards that are fixed in the cell, every inner
corner of the pattern in the cell frame (exact). A board on a fixture that is placed anew per episode moves with its
fixture; where it sits on the fixture is on drawings 4 and 5.

## 3. Arm, gripper, fingertips, controller

The arm is a 6-DoF UR10e-class arm with a wide two-finger parallel gripper. Its base stands at the cell's origin. `describe()` gives its joint names and
ranges, its torque limits and its home posture (0, -1.5708, 1.5708, -1.5708, -1.5708, 0 rad), at which every episode starts with the gripper
open. The models are in `~/given/robot/`; the given library builds the arm, the gripper and the controller exactly as the
cell does, but for the field of the wrist camera (section 9).

Controller. The physics runs at 0.5 ms per step and the control tick is 10 ms. The
arm is driven by computed torque, recomputed every physics step: it holds its posture against gravity and tracks the
commanded configuration. Joint targets are rate-limited to 1.5 rad/s. Cartesian targets go through the
cell's differential inverse kinematics on the tool point; the targets move at most 0.5 m/s and 1.5 rad/s. The
stiffness scale (0.2 to 1.0) scales the joint position gains. The controller has no
integral term and does not know what the hand carries: with a module in the hand the tool settles away from its
commanded pose. Measured in the cell with the arm settled at rest: with a leg in the hand the tool stood up to about
7 mm from its commanded pose, with an arm up to about 2.8 mm, with the head up to about
0.8 mm; with the hand empty, at the same commanded configurations, 0.0 mm.
`get_state()` gives the pose reached and the commanded configuration.

The line's handling rule for a module in the hand is at most 0.25 m/s and 0.5 metres per
second squared at the tool. The cell does not enforce it.

Gripper. A wide parallel gripper with two fingers: the opening runs from closed to
133.6 mm (the gap between the pad faces at their middle) and moves at 0.1 m/s. Each finger
carries a flat gripping face 27 mm across and 65 mm long; the pads' friction
coefficient is 0.7. The force register sets the force per pad, 10 to
125 N (100 N unless commanded). On a part that cannot centre itself between the pads
the mean pad force read 3.5 to 5.5 % below the command, a single pad up to
12.8 % below; with an arm in the hand a pad read from about 9 % below the
command to about 6 % above it, and at the moment a module leaves its fixture a pad can read about
10 % above the command: the register's exact force is not promised. The fingertips are the
gripper's own; they carry no extension.

The tool point lies 196.2 mm from the flange on the tool axis. Tool axes: z is the direction out of the
flange, y the closing axis, x across the pads. The grasp centre between the pad faces is not the tool point; the
gripper code in the given library gives it for every opening.

Above the root of the gripping faces the finger links stand inside the faces' line. A part that reaches between the
fingers beyond the faces' root meets the links before the pads close on it: on a test block 120 mm wide
held with its middle at the faces' middle, the links met the block's upper edges at an opening of
112 mm and the pads never touched it.

A module is not clamped rigidly between the pads. When it leaves its supports its weight can turn it and make it slip
in the grip: in the cell's own trials an arm turned by up to 7.6 degrees as it left its rack and
the centre of its ring moved by up to 11 mm, while in another hold the left arm turned by about
1.8 degrees; a leg turned by up to about 7.3 degrees and slipped by up to
3.5 mm after it left its station. The pose of a module in the hand is not its pose in the rack.

The force and torque sensor sits at the flange and reads in the flange frame; it carries the weight of the gripper
and of whatever the gripper holds, white noise (0.3 N and 0.02 N m rms per reading) and a bias
that is chosen at random per episode (up to 1 N and 0.05 N m per axis).

## 4. Stand and clamp

The pelvis rests with its underside on the stand's saddle pads and with its rear wall on a rear button; the tips of
the toggle clamps press on its front walls (drawing 4; photographs 2 and
3). In the cell the pelvis does not move, whatever presses on it.

Where the operator leaves the base in the clamp differs from run to run by up to ±1 mm along the
cradle, which is the base's left-right axis, and by up to ±0.3 degrees about the vertical through the
rear button. The stand's datum board is on the
stand, not on the base: it gives the stand's place, and the base lies on the stand within that repeat.

The torso is not held rigidly. It turns on the waist joint, which is unpowered and has a holding friction of
1.54 N m. A brace with a tine on either side of the torso stands behind it; the tines
do not touch the torso (the gaps are between 0.8 and 2.1 mm). Under a sideways push the
torso turns about the waist until a tine stops it, by up to about 2.1 degrees, and it stays where it
stops. It comes back only under an opposite moment above the waist's holding friction. A shoulder socket, and the neck's output
with it, is therefore where the torso was left, not where the drawing puts it on a square torso.

## 5. Racks: how each module is presented

| unit | role | mass, kg | bodies | joints | presented on |
|---|---|---|---|---|---|
| `base` | the base | 10.294 | 5 | 4 | the stand (`stand`) |
| `leg_R` | right leg | 6.971 | 6 | 5 | its station of the leg rack (`leg_station_R`) |
| `leg_L` | left leg | 6.971 | 6 | 5 | its station of the leg rack (`leg_station_L`) |
| `arm_R` | right arm | 4.435 | 6 | 5 | its arm rack (`arm_rack_R`) |
| `arm_L` | left arm | 4.433 | 6 | 5 | its arm rack (`arm_rack_L`) |
| `head` | head | 1.209 | 2 | 1 | the head's nest (`head_nest`) |

`items` of `describe()` names each unit's fixture and gives the fixture's nominal place rounded to
0.01 m, with the bounds of its placement per episode. The exact resting place of a module in its fixture
is on drawing 5, relative to the fixture's datum board. The fixtures' shapes are in `~/given/fixtures/`, at
their nominal places.

The leg rack: a bench that carries a station for each leg and the rack's datum board. A leg stands in its station on
its own sole, in a sole nest, leaning against a corner stop at the top of its shank. The stations are passive: no
clamp, no spring. The stations lean the same way; they are not mirror images of each other. A leg resting in its
station presses on it with up to 91 N.

The arm racks stand on the rack table. An arm hangs in its rack turned as it is mounted on the robot, at its
delivered joint angles: its shoulder ring's flange lies in a half cradle with a shoulder 0.2 mm off
either face of the flange, and the housing at its elbow lies on rest pads.

The head's nest stands on the rack table. The head stands upright on a boss inside a locating skirt.

Each module rests in the attitude that drawing 5 gives. Per episode the fixtures are placed anew
(section 10); a module is never displaced inside its fixture.

## 6. Modules: bodies, joints, unpowered behaviour, delivered joint state, output rotors

The base and the modules are rigid bodies on hinge joints: 25 joints in all, 4 of them in
the base (the waist and the output rotors of the hips and of the neck) and 21 in the modules.
`module.json` gives every body, every joint's axis, range and actuator type, and each body's mass, centre of mass
and inertia; the three output rotors carry none: their actuator's reflected inertia acts at the joint.

No joint is powered. A joint resists with its holding friction, with damping and with its actuator's reflected
inertia, by actuator type:

| actuator type | holding friction, N m | damping, N m s/rad | reflected inertia, kg m squared |
|---|---|---|---|
| A | 0.61 | 0 | 0.0246 |
| B | 0.27 | 0 | 0.0324 |
| C | 1.54 | 0 | 0.065 |
| D | 0.72 | 0.378 | 0.0343 |
| E | 1.21 | 1.18 | 0.1 |

The ankle's pitch and roll are driven by actuators of type A through a linkage. At the ankle pitch joint that makes
2.46 N m of holding friction and 0.2008 kg m squared of reflected inertia, at the
ankle roll joint 0.98 N m and 0.0315 kg m squared, in place of type A's own values.
These are the values the cell simulates. The data pages in `~/given/datasheets/` give the same values for every actuator
type; the ankle joints' values are given only here. A joint loaded beyond its holding friction moves and stays where
it stops. Nothing in this station returns a joint to an angle.

Joint ranges are those of `module.json`. For the hip roll and hip yaw joints the cell's ranges are narrower than
the joints' own travel, because the shells meet inside it: hip roll -20.5 to 7.5 degrees on
the right and -7.5 to 20.5 on the left, hip yaw -38 to 43.5 on the
right and -43.5 to 38 on the left.

Delivered joint state. Each module is delivered, rests in its rack and is drawn on drawing 6 at the
joint angles that `module.json` lists as delivered. In the hand and after its mate a joint keeps its angle only
as long as its holding friction holds it: in the cell's runs the knee of a mounted leg stood 3 to
5 degrees flexed: a knee flexes while its leg is carried; after the release no joint of a mounted module
moved. The cell reads a mate only at the module's interface against
its seat; the angles of the joints beyond the interface count for nothing.

Output rotors. A leg mates on the output rotor of a hip pitch actuator, the head on the output rotor of the neck yaw
actuator. These rotors are joints of the base and unpowered like every other joint. At the start of every episode the
right hip output stands at -15.00 degrees, the left hip output at -15.00 degrees, the neck
output at 30.00 degrees and the waist at 0.00 degrees of their joints (the delivered
angles in the base's `module.json`): the line lines the outputs up at commissioning so that their threaded holes
lie under the module's screw pattern at the bolt position that counts. This statement is the primary information.
The wrist camera can check it, but its image alone does not resolve the neck output's clock to the fastening window.
In the mesh files the neck output's end face has no holes: its threaded holes lie under that face, and drawing
3 shows them where they are. In a camera frame the neck output's clock shows only by the ring of
six screw heads on that face.

On the right shoulder the clearance holes of the torso shell are turned against the threaded holes behind them; on
the left they are in line. Drawing 2 dimensions both sets. An arm's bolt position refers to the
threaded holes, on either side.

## 7. Fastening

Each interface watches its own module and nothing else. After every control tick the cell reads the module's
interface against its seat: the axial distance along the seat's axis (zero with the module's face home on the seat's
face, positive while the module is short of home, negative past it; the windows take its size), the lateral distance
from that axis, the tilt
between the axes, and the phase about the axis measured from the nearest bolt position; and the speed and the angular
speed of the module's flange relative to the body that carries the seat, at the centre of the bolt circle on the
seating face. On an output rotor the carrying body is the rotor: the phase is read against the rotor wherever the
rotor stands.

| seat class | seat | bolt positions | pitch, deg | enter: axial, mm | enter: lateral, mm | enter: tilt, deg | enter: phase, deg | lock: axial, mm | lock: tilt, deg |
|---|---|---|---|---|---|---|---|---|---|
| leg | a leg's flat flange with its shallow pilot on a hip output rotor | 6 | 60 | 2 | 0.6 | 1.5 | 2.5 | 0.5 | 0.3 |
| arm | an arm's ring in a shoulder socket of the torso shell | 12 | 30 | 3 | 1 | 1.5 | 2 | 0.5 | 0.5 |
| head | the head's bracket on the boss of the neck's output rotor | 6 | 60 | 2 | 0.5 | 2 | 3 | 0.5 | 0.5 |

A module inside the enter window of its seat is a candidate at the bolt position nearest to it. Nothing happens to
a candidate: no force, no sign. The lock condition is: inside the enter window, the axial distance and the tilt inside
the lock window, the phase still within the enter window of the bolt position it became a candidate at, and the
module's flange slower than 5 mm/s and 2 degrees per second relative to
its seat. When the condition has held for 50 control ticks without a break (0.5 s),
the cell fastens the module exactly where it is: it is not moved to a nominal pose, and a gap or a tilt inside the
lock window stays. A lock means inside the window and slower than those speeds for that time; it does not mean at
rest, and it can happen while the module is held or still moving slowly. A candidate that leaves
2 times the enter window is free again.

The cell applies no force to a module before the lock: it does not pull a module in, it does not hold it, and it
does not rescue a module that is let go without a lock. It fastens at whichever bolt position the module is held at.
For each module a single bolt position counts, the position at which the drawings and the photographs show the
finished robot; at any other position the module is fastened for good and counts nothing. Nothing can be unfastened.
An interface never takes the module of the mirrored interface. A module that has touched the floor is never fastened
again. Nothing reports a candidate, a lock or a scrapped module.

Entry behaviour measured on the cell's seats, with a spring in place of the hand (20 kN/m at the seat's
centre and 50 N m/rad, the module's weight carried) aimed at the seat with one offset at a time:

| seat class | aimed off the axis: locked up to, mm | not from, mm | aimed tilted: locked up to, deg | not from, deg | aimed off phase: locked up to, deg | not from, deg | held short of the face: locked up to, mm | not from, mm | lateral play when seated, mm |
|---|---|---|---|---|---|---|---|---|---|
| leg | 0.5 | 0.75 | 0.5 | 1 | 2.4 | 2.6 | 0.45 | 0.55 | 0.30 |
| arm | 0.25 | 0.4 | 0.25 | 0.5 | 1.9 | 2.1 | 0.45 | 0.55 | 0.26 |
| head | 0.4 | 0.6 | 0.5 | 1 | 2.9 | 3.1 | 0.45 | 0.55 | 0.21 |

Jamming is not monotonic in the offset; the table gives what was tried.

A module let go before its lock, in the same trials. A leg let go with its flange home on the output, or at most
0.15 mm short of the face, hung on its pilot cocked 0.09 to 0.29 degrees and
was fastened 0.51 to 0.53 s later. Let go 0.25 mm or more short of the face it
fell off its pilot to the floor. An arm let go before its lock cocked 0.88 to 0.91
degrees in its socket, backed out to 0.86 to 1.06 mm from home and was never fastened.
The head let go on its boss settled onto its seat and was fastened 0.32 to 0.52 s later.

## 8. Collision model

The cell does not collide the surfaces of the mesh files. Every body collides as a set of convex pieces: 2962 pieces on
31 bodies. They are the pieces of `collision.npz`, in the unit frame, and a simulation of your own
that uses them collides as the grader's cell does. Pieces flagged exact follow the true surface: the interfaces (the
pilots, the seat faces, the sockets, the bosses, with the clearances the drawings give) and the tiled faces near the
hips and the shoulders. Pieces flagged rest are those a module rests on in its fixture.

Elsewhere the pieces differ from the mesh files, both ways. The collision surface can stand outside the surface of
the mesh files: a convex piece bridges the hollows of its part. Measured with rays from outside, per body:

| body | outside the surface of the mesh files: 99 percent of the rays within, mm | at most, mm |
|---|---|---|
| `head_root` | 0.7 | 2.5 |
| `left_ankle_pitch_link` | 1.9 | 2.6 |
| `left_ankle_roll_link` | 4.2 | 7.3 |
| `left_arm_root` | 1.0 | 2.6 |
| `left_elbow_link` | 2.4 | 3.5 |
| `left_hip_pitch_link` | 4.2 | 8.6 |
| `left_hip_pitch_rotor` | 1.5 | 2.0 |
| `left_hip_roll_link` | 3.2 | 4.6 |
| `left_hip_yaw_link` | 5.3 | 12.4 |
| `left_knee_link` | 9.0 | 13.8 |
| `left_shoulder_pitch_link` | 4.5 | 7.2 |
| `left_shoulder_roll_link` | 0.3 | 1.7 |
| `left_shoulder_yaw_link` | 4.4 | 8.1 |
| `left_wrist_yaw_link` | 2.3 | 3.5 |
| `neck_pitch_link` | 8.1 | 13.2 |
| `neck_yaw_rotor` | 0.0 | 0.0 |
| `pelvis_link` | 4.4 | 7.6 |
| `right_ankle_pitch_link` | 1.9 | 2.6 |
| `right_ankle_roll_link` | 4.3 | 7.5 |
| `right_arm_root` | 1.0 | 2.5 |
| `right_elbow_link` | 2.6 | 3.5 |
| `right_hip_pitch_link` | 3.9 | 9.0 |
| `right_hip_pitch_rotor` | 1.5 | 2.0 |
| `right_hip_roll_link` | 3.4 | 4.6 |
| `right_hip_yaw_link` | 6.0 | 13.4 |
| `right_knee_link` | 9.1 | 17.7 |
| `right_shoulder_pitch_link` | 4.5 | 7.4 |
| `right_shoulder_roll_link` | 0.3 | 1.8 |
| `right_shoulder_yaw_link` | 4.2 | 7.9 |
| `right_wrist_yaw_link` | 2.0 | 2.5 |
| `waist_yaw_link` | 21.3 | 34.3 |

Beside the brace's tines, pieces of the torso stand outside its true surface and stop the torso's turn a little
earlier than the true surface would (section 4).

The surface of the mesh files can also stand outside the pieces, and what stands outside does not collide. Where
pieces were cut back for the travel of the joints and for the entry of a module into its seat, the cut took away at
most 1.749 mm. Screws and other fasteners have no piece at all, and neither have some bought parts
(actuators, circuit boards, the fan), most of which lie inside the shells. Measured per module at the delivered joint
angles, against all of the module's pieces together, as the distance from a point of the mesh surface to the nearest
piece:

| module | seen from outside: 99 percent of the rays within, mm | seen from outside: at most, mm | anywhere on parts that have pieces: at most, mm | anywhere on parts without pieces: at most, mm |
|---|---|---|---|---|
| leg | 0.8 | 5.5 | 9.1 | 17.0 |
| arm | 0.6 | 11.4 | 22.8 | 27.4 |
| head | 1.0 | 2.9 | 1.5 | 35.5 |
| base | 1.8 | 5.6 | 5.6 | 35.2 |

"Seen from outside" is the surface that a straight line from outside the module meets first (the rays of the table
above; a ray that meets a piece first counts as zero). The largest values of the last two columns lie
inside the modules, behind the modules' own surface; only the head's and the base's values of the column "anywhere on
parts that have pieces" lie on the outside.

Inside a unit its bodies collide with each
other, except the pairs `module.json` lists as excluded. The stand, the racks, the nest, the rack table, the floor,
the camera portal with its beam and posts, and the housing of the overhead camera are colliding bodies; a contact
with them counts under the safety rules like any other. The contact parameters are in `~/given/cell/cell_config.json`.

## 9. Cameras, depth, render layer and its cost

The overhead camera (`top`): 1280 by 960 pixels, nominal vertical field of view
60 degrees, looking straight down, the top of the image towards the cell's +x. Nominally the front of
its lens is at (0.15, -0.02, 1.89) m. Per episode its mount is moved by up to
±30 mm along each axis of the cell and turned by up to 3 degrees about an axis that is
chosen at random as well and can point any way; each of its focal lengths (along the image's width and along its height) differs
by up to ±3 % from the nominal one, and its principal point lies up to ±2 % of the image's
width and of its height from the image's centre. At the height of the units a
pixel covers 2.0 to 2.8 mm.

The wrist camera (`wrist`): 640 by 480 pixels, horizontal field of view
80 degrees (vertical 64.37), on the hand beside the tool axis, looking between the
fingers; `cameras` of `describe()` gives the lens' place and its aim point in the flange frame. Its frames are
rendered at this nominal mount and field. The lens' place and the aim point fix the optical axis only: the camera's
whole pose on the hand, the roll of its image included, is that of the wrist camera in the arm model that the given
library builds; that model's camera carries a vertical field of 70 degrees, and the cell sets it to
64.37 degrees (80 horizontal) before it compiles.

The frames are pinhole images: no lens distortion is applied. The nominal lens of either camera has square pixels and
its principal point at the image's centre, midway between the centres of the outermost pixels along the width and
along the height. The overhead camera's mount and lens of an episode are not reported.

Depth: a 16-bit image registered to the colour pixels, the distance along the optical axis rounded to
steps of 1 mm, 0 where there is no reading. The cell cuts no range out of it:
every finite reading above zero and below 65 m is returned.

What the frames show. The overhead camera sees the whole cell, every unit and every datum board; a board's square is
11.8 pixels or more across. Its frame may show parts of the bay's structure at its edges; none of it lies
between the camera and a board or a rack. The wrist camera's frames are exposed two stops above
the overhead camera's. In the wrist camera's colour frame the right shoulder socket's 12 clearance
holes show as bright marks on a dark face; of the left socket's holes seven show as dark marks and
five are weak. The depth frame resolves the clearance holes of either socket, 2 to
21 mm behind the face. It does not resolve the screw heads on the hip flanges and on the neck's boss,
which lie within 2 mm of their face; in the colour frame these heads differ from their face by
colour, and some are as bright as the face. Until the arms are mounted the torso is hollow between the shoulder
sockets: through a socket the camera sees through the torso, and the socket has no bottom in the depth frame.

What the cameras do not show. With a leg within a few millimetres of its seat, and with an arm's ring inside its
socket, neither camera sees either mating face. The head's seating face is hidden in its nest.

Frames of the same state are not identical: between frames rendered again of an unchanged state, about
0.03 % of the values differed by more than 8 grey levels.

The render layer. The cell renders its cameras with MuJoCo's Filament renderer on the CPU, in the room of
`~/render/`; depth comes from MuJoCo's own renderer on the same geometry. The given library holds this layer for a
twin of your own (`~/given/cell/cell_api.md`, "Rendering your own frames"). MuJoCo's own renderer gives a different picture than the
cell's cameras: no room, its own lights, flat colours. Image processing tuned on its frames may behave differently on
the cell's frames.

Cost. The wall cap allows about 4.4 s of wall clock for a frame and about 51 s more for
the first frame of an episode, because the renderer's session opens then; what a frame really takes depends on the
machine, and all of it counts against the episode's wall clock. The layer's session holds about 5.2 GB
of memory.

## 10. What varies per episode

Every episode has a layout of its own, chosen anew at random; nothing in the given files predicts it. Each amount is
chosen at random, evenly between minus and plus the value of the table, each on its own. A shift runs along the cell's x and along the cell's y; a turn is about
the vertical line through the point named.

| what | shift along each of x and y, mm | turn about the vertical, deg | the vertical passes through |
|---|---|---|---|
| the stand, with the base | ±15 | ±2 | the waist axis |
| the base in its clamp | ±1, along the cradle only (the base's left-right axis) | ±0.3 | the rear button |
| the leg rack: bench, stations and board together | ±12.5 | ±5 | the mid point of the stations |
| each arm rack, on its own | ±20 | ±5 | the rack's footprint centre |
| the head's nest | ±20 | ±10 | the nest's footprint centre |

What else varies per episode: the overhead camera's mount and lens (section 9) and the bias of the force
and torque sensor (section 3). What does not vary: which module stands where, a module's place inside its fixture, the joint
angles of the modules, the start of the output rotors and of the waist, the rack table, the floor board, the camera
portal and the arm.

## 11. Safety rules and budgets

The cell checks four rules after every control tick.

* Arm contact (in the events: `robot_contact`). Any link of the arm or of its gripper on any body that is not part
  of the arm or of its gripper (a module, the base, a fixture, the floor, anything else in the cell): the normal forces
  between that link and that body, summed, above 150 N on any tick. One exemption: the gripper's
  pads on a module, which carry the grip. A pad on the base, on a fixture, on the floor or on anything else counts like
  any other link.
* Part force. A module on a single other body: the normal forces between the module and that body, summed, above
  150 N for 20 ticks without a break (0.2 s). Not
  counted: the gripper's pads; a link of the arm or of its gripper (the rule above reads it); contacts inside a module; a body the
  module is fastened to, directly or through other fastened modules and the base. Counted: a module in its rack, on
  the base or on a module it is not fastened to, on a fixture, on the floor. A fastened module is still watched.
* Part speed. A module's centre of mass faster than 3 m/s.
* Floor contact. Any body of a module touching the floor. The module is scrapped: it is never fastened again and
  does not count.

A violation does not end the episode. Each rule counts at most once per episode, and an episode with any counted
violation keeps 30 % of its score. `step()` returns an event for a violation, at most once per
rule and pair of bodies in 0.5 s.

Budgets per episode: 1293 s of simulated time, after which the episode ends; 8074 s
of wall clock, rendering included; 120 s of silence between the cell's reply and the program's next
request; 450 camera frames over both cameras; at most 500 ticks per
`step()` call; 32296 s of CPU time for each process of your program (four times the
wall cap), past which the process is killed, and for the program's own process that ends the episode with what it
achieved. The grading machine has 8 CPUs and 24 GB of memory, which your program
shares with the cell's physics and its camera renderer. The wall cap is built on about
4.2 s of wall clock per simulated second on a machine of that size and on about 4.4 s for a
frame (section 9): the wall clock can end an episode before the simulated time does.

## 12. What a stopped session gets

Your session. It is stopped after 6 hours of wall clock. A stopped session is graded on what the
submission folder holds at that moment. At verify start every process of your account is stopped, and the submission
is copied; a file written after that voids the gate.

An episode. It ends when the program calls `finish()`, when the entry function returns or raises, when the
simulated time reaches its cap, or when the wall clock or the silence limit is passed. The grader then reads which
mates the cell has fastened; a module that is seated without a lock at that moment counts nothing. A program that
crashes or breaks the wire ends its own episode with what it achieved, and the other episodes still run. When the
frame budget is used up, a `get_camera()` call is refused: through the given proxy it raises an error that the
program can catch (`~/given/cell/cell_api.md`, "Error replies"), and the episode goes on if it does.
