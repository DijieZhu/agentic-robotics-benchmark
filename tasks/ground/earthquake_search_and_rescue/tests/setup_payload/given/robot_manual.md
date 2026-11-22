# SR-2 rescue crawler: operator and integrator manual

The SR-2 is a tracked response robot for collapsed structures. This manual gives what an integrator needs to write
its autonomous mode: dimensions, configurations, mobility, the arm, the sensors and their mounting, the radio link,
the survival kit and the safety rules. The program interface is in `robot_api.md`.

All lengths in metres, angles in radians (degrees in brackets), times in seconds.

## 1. Frames

* **Chassis frame**: origin at the centre of the chassis spine, x forward, y left, z up. On level ground the origin is
  0.137 m above the ground.
* **Map frame**: the robot's chassis frame at deployment, moved down to the ground: origin on the ground under the
  chassis centre where the team set the robot down, x along its initial heading, y to its left, z up. The odometry
  pose and the survivor report use it.
* **Optical frames** of the cameras: x right, y down, z forward (pinhole model, square pixels, principal point at the
  image centre). Each frame reply gives the camera's pose in the chassis frame (`T_chassis_cam`, 4 x 4).
* **Lidar frame**: axes of the chassis frame, origin at the lidar's optical centre (`T_chassis_lidar`).

## 2. Dimensions and configurations

| item | value |
|---|---|
| main tracks | 0.72 long (sprocket and idler radius 0.09, axles 0.54 apart), 0.08 wide; the lower run carries the robot on its middle 0.34 m on hard level ground |
| track gauge (centre to centre) | adjustable 0.30 to 0.50 (telescopic cross beams), 0.05 m/s |
| overall width | gauge + 0.216: **0.516 at gauge 0.30**, **0.716 at gauge 0.50** (the flippers ride outboard of the tracks) |
| flippers | one pair at the front axle, one at the rear axle, each pair moved together; 0.30 long, tapering from radius 0.085 to 0.045, belts driven with the main tracks; 60 deg/s |
| flipper angle | 0 = pointing straight out (forward for the front pair, backward for the rear pair), positive = tip raised; range -80 deg .. +190 deg; about 170 deg = folded back over the track |
| overall length | 0.72 with both pairs folded, 1.32 with both pairs straight out |
| height | **0.555 with the mast down** (top of the lidar), plus the mast extension (0 .. 0.45, 0.10 m/s): 1.005 fully raised |
| ground clearance | 0.06 under the spine |
| mass | about 55 kg |

The arm, stowed, stays inside this envelope (its forearm and gripper show at the bottom of the mast camera's image).

## 3. Mobility

Driving is commanded as a forward speed v (up to 0.6 m/s) and a yaw rate w (up to 1.2 rad/s); the controller sets the
two belts to v -/+ w x gauge / 2 (ramped at 1.5 m/s^2). A tracked vehicle skid-steers: the actual turn rate is lower
than commanded, depends on the ground, and is much lower at the narrow gauge (roughly half). Tracks slip on loose
debris; the odometry (section 6) measures the motion actually made.

Rated capabilities (dry concrete and rubble, robot in good order):

| obstacle | capability |
|---|---|
| slope, driving up or down the fall line | up to about 35 deg (traction-limited; less on loose rubble) |
| side slope | stable to about 40 deg at the narrow gauge and 55 deg at the wide gauge (static); turning on a side slope is not advised |
| step, flippers straight out | up to about 0.25 |
| step, flipper technique | up to about 0.33: approach square at low speed with the front flippers raised steeply (about 65 deg) until their tips reach over the step's edge; then lower the front flippers onto the step's top (about 20 deg), which levers the nose up, and press the rear flippers down (about -40 deg) to push the tail up and keep the centre of gravity forward, while driving on slowly; level the flippers once on top |
| step down | any of the above heights, flippers straight out, slowly |
| gap (trench, hole) | up to about 0.55, square to the gap, slowly, with both flipper pairs straight out or a little below (about -8 deg) so that the flippers' belts carry the robot over the edges (a 1.32 m footprint); about 0.25 with the flippers folded |
| passage width | gauge + 0.216 plus a margin |
| overhead clearance | 0.555 + the mast extension, plus a margin |

Driving the front of the tracks into a wall or a high obstacle makes the robot rear up; keep doing it and it goes over
backwards. A tracked robot turns on the spot by sliding its tracks sideways: it needs room around it (a robot with its
flippers straight out sweeps a circle of about 0.67 m radius, folded about 0.45 m).

## 4. The arm and the gripper

A five-axis arm on a turret at chassis (0.165, 0, 0.095):

| joint | axis | range | |
|---|---|---|---|
| q1 turret | yaw, + = to the left | -175 deg .. 175 deg | |
| q2 shoulder | pitch; 0 = upper arm pointing forward, horizontal; + = raising it | -20 deg .. 190 deg | shoulder at chassis z 0.17 |
| q3 elbow | pitch, relative to the upper arm | -186 deg .. 186 deg | upper arm 0.34 |
| q4 wrist pitch | relative to the forearm | -120 deg .. 120 deg | forearm 0.34 |
| q5 wrist roll | about the tool axis | -180 deg .. 180 deg | tool centre point 0.22 beyond the wrist pitch axis |

The tool axis lies in the turret's vertical plane at the angle `pitch = q2 + q3 + q4` above the horizontal. Joint rate
45 deg/s. Stowed: `[0, 184 deg, -184 deg, -55 deg, 0]` (upper arm back along the deck, forearm folded over it, the gripper tucked down behind the tracks' noses). `arm_to(p, pitch, roll)` moves the tool centre point along a
straight line to point p of the chassis frame at a given tool pitch and wrist roll (the controller solves the inverse
kinematics every tick; it answers `unreachable` when no solution exists); `set_arm(q)` moves the joints directly. The
arm's links collide with the world: under a low slab, keep the elbow low.

The parallel gripper opens 0 .. 0.16 (0.08 m/s); its jaws close along the gripper's y axis, which at roll 0 is
horizontal and perpendicular to the arm's plane. **Grip rule**: closing the jaws on a movable object until both pads
press on it (4 N or more each) grasps it; from then on it moves with the gripper. Opening the jaws releases it. An
object heavier than 8 kg cannot be lifted (the grip slips at once), and a held object that snags so that the grip has
to carry more than about 260 N for 0.2 s slips out. Things fixed in the structure cannot be grasped.

## 5. The survival kit

An orange pouch (water, a hand-held radio, a foil blanket), 0.17 x 0.09 x 0.07, 1.2 kg, in a cradle on the rear deck:
its bottom centre at chassis (-0.275, 0.055, 0.065), long side along x. A latch holds it while driving and opens when
the gripper grasps the kit. To deliver it, set it down on the floor next to the survivor (within their reach: within
0.8 m of their head or a hand) and open the gripper; do not put it on them.

## 6. Sensors

| sensor | mounting | data |
|---|---|---|
| mast RGB-D camera | pan-tilt head on the mast: at mast 0 and pan / tilt 0 the optical centre is at chassis (-0.129, -0.030, 0.275), looking forward | 640 x 480 colour, vertical field of view 55 deg; depth 0.2 .. 10 m, noise about 2 mm + 0.25 % x z^2 |
| thermal camera | same head, 4.5 cm to the right of the RGB-D camera, same direction | 160 x 120, vertical field of view 42 deg, radiometric: kelvin x 100 per pixel, NETD 0.05 K |
| lamps | two LED lamps on the head, aimed with the cameras | the head's view is lit even in dark voids |
| front driving camera | spine nose, chassis (0.305, 0, 0.055), pitched 12 deg down | 320 x 240 colour, vertical field of view 80 deg |
| rear driving camera | spine tail, looking back, pitched 12 deg down | 320 x 240 colour, 80 deg |
| gripper camera | on the gripper palm, looking along the tool axis | 320 x 240 colour + depth (0.07 .. 3 m) |
| lidar | on the mast top (does not pan): at mast 0 its optical centre is at chassis (-0.235, -0.050, 0.390), rising with the mast | 16 channels from -15 deg to +15 deg, 0.5 deg azimuth steps (720 per channel), 0.3 .. 25 m, noise 1 cm + 0.2 % of the range, about 2 % dropouts; one sweep per request |
| odometry | lidar-inertial | the robot's pose (x, y, yaw) in the map frame; drift about 1 % of the distance travelled and a slow heading drift. `pose.z` is the chassis' height above the ground at the deployment point (barometric and inertial; noise about 1 cm, no drift) |
| IMU | chassis | roll, pitch (noise about 0.002 rad), body rates (`imu.gyro`, chassis frame, noise about 0.005 rad/s) |
| velocity estimate | lidar-inertial | `velocity`: forward, lateral (noise about 0.01 m/s) and yaw rate (about 0.005 rad/s), chassis frame |
| joint and belt encoders | | flipper angles, gauge, mast, head, arm joints, gripper opening, belt speeds |
| radio | antenna on the mast | link up / down and its signal strength (dBm) |

The colour cameras expose automatically for where they are: in a dark crawl space the image is brightened (the lamps'
pool of light dominates it, and the street seen through an opening from inside comes out overexposed); outdoors they
expose for daylight.

Surfaces have the temperatures they would have on site: the shaded interior of the building at the day's air
temperature (8 to 24 C, a little scatter between objects), sun-heated outdoor surfaces warmer (metal in the sun much
warmer), the survivor's skin about 33-35 C and their clothes about 27-31 C. Appliances can still be warm.

What a person is to each sensor: an adult lying on the floor, on their side, on their back or face down, in everyday
clothes whose colours, like their skin tone, differ from site to site. Every sensor sees that same body: the colour
cameras its clothes, skin and hair; the depth camera and the lidar its surface; the thermal camera its bare skin (face,
neck, hands) at about 33-35 C, hair about 3 C below the skin, clothes at about 27-31 C. Debris can cover part of
them. A shop mannequin has a person's shape, plain plastic all over, and the room's temperature.

## 7. The radio link

The team's base station stands at the deployment point. The link's signal falls with distance and with every wall and
slab between the base station and the robot; below about -85 dBm the link is down (`get_state()["link"]`). A survivor
report made while the link is down is queued on board and sent the moment the link comes back. The team only holds
what reached it.

## 8. Safety rules

* **The survivor: protective stop.** The robot, the kit or anything the robot moves must never press on the
  survivor. The robot watches the contact itself: more than 60 N for 30 ms trips its protective stop. Every drive
  stops for good, `step` reports `protective_stop` and the mission ends there; the team keeps what it had already
  received, the robot does not come back, and the mission scores at most 0.45. There is no warning before the stop:
  the gripper's `load_N` rises only once the contact is there. Approach slowly; leave room; carry the kit high
  enough to clear the person.
* **Load-bearing debris.** Never move debris that something rests on: a slab, a beam, a stack above it. Such a prop
  really carries its load (a slab section of 180 kg bears on a crate with most of a kilonewton), so brushing against
  it or bumping into it does not shift it far. The robot can still shift it: its tracks push with more than its own
  weight, and a prop that is pushed on gives way slowly, a few centimetres within a minute. Once the robot,
  what it holds or debris it pushes has touched the prop or the slab on it, moving either by more than 5 cm brings
  the structure down (a secondary collapse, `collapse` in `step`'s events): the team aborts the mission and it
  scores 0. Before you take hold of anything or push through anything, check what bears on it.
* **The robot.** It is lost if it overturns (tilted more than 70 deg for 2 s) or drops into an opening in the floor
  (a collapsed floor section into the basement): the mission ends there.

## 9. The training ground

The team trains at a training ground that runs in your sandbox as a service (the robot and its physics exactly as on
a mission; the site plans are the trainers', not yours). Its sites: `lanes`, the standard test lanes (a street, a
facade spill, then one obstacle of each kind in a row: a beam across a doorway, a step, a narrow breach, a floor gap,
a crawl space under a dropped slab, with a training dummy at the end), and training rubble sites built like the real
ones (`run_mission.py --list`). A mission there is the same as a graded one: the same budgets, the same start on the
street. The training ground is for commissioning the program on the robot, and it gives no answer sheet: nobody
tells you where the person was, how far off your report was, where the kit ended up relative to them, or a score.
At the end you get the robot's own mission log: how the mission ended, the simulated time, how many reports were
made and how many reached the team, whether the kit left its cradle, the events (`protective_stop`, `collapse`,
`robot_overturned`, ...) and the budgets used. Judge a mission from what your program sensed and logged: its
frames, its map, its odometry. The training ground allows 80 missions in total and 2 at a time
(`/run/rubblecell/training_status.json`); each mission log is also kept in `/run/rubblecell/logs/`.
