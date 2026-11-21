# Equipment sheets: the bench, the robot, the numbers

What the manuals and drawings of this bench state. Lengths are in metres unless a line says mm; angles in degrees
unless a line says rad. A value given with a range is drawn inside that range, per unit or per episode as the line
says; the actual value is not told, and your controller has to find it or cope with it. A value without a range is
the same on every bench.

## 1. Frames and the table

- World frame: z up, metres, radians, seconds; quaternions are (w, x, y, z).
- The table top is at z = 0.80 and spans x from -0.15 to 0.75 and y from -0.80 to 0.80.

## 2. The robot

- A Franka Panda with its base at (0, 0, 0.80), facing +x. Its model is the Panda of robosuite 1.5.2, whose files are
  in the image: `robots/panda/robot.xml` and `grippers/panda_gripper.xml` under
  `/opt/venv/lib/python3.12/site-packages/robosuite/models/assets/`. The gripper is mounted on the body `right_hand`.
- `ee_pos` and `ee_quat` are the pose of the gripper model's site `grip_site` (between the finger pads).
- Start posture (joint angles, rad): 0, -0.35, 0, -2.30, 0, 1.95, 0.785 (pi/4).
- Control at 10 Hz; each control step is 50 physics steps of 2 ms.
- **Cartesian step** (8 or 15 numbers): at the start of the step the hand's target is set to its pose now, moved by
  (dx, dy, dz) and turned by the rotation vector (drx, dry, drz), both in the world frame. Steps longer than 0.03 m
  or 0.2 rad are shortened to that length. An operational-space impedance controller (the arm's inertia and
  gravity compensated) pulls the hand toward the target for the step: it commands an acceleration of 400 s^-2 times
  the position error and 200 s^-2 times the orientation error, critically damped. A weak pull in its null space
  moves the joints toward the posture last given (the start posture until one is given). The hand is compliant:
  under a load, and while it holds a tube, it lags the target by millimetres.
- **Joint step** (9 numbers): a gravity-compensated joint position controller (200 N m/rad on the error, 20 N m s/rad
  on the velocity) moves toward the given angles; each joint's target is at most 0.1 rad from its angle now.
- **Gripper**: `grip` runs from -1 to +1; the fingers' opening is 0.04 x (1 - grip) metres (80 mm at -1, 40 mm at 0,
  closed at +1). The fingers are position-controlled and take a few steps to arrive.
- `wrench`: the wrist's force (N) and torque (N m) in the world frame, with noise of 0.3 N and 0.03 N m.
- An arm joint faster than 8 rad/s counts as an unstable simulation.

## 3. The cameras

| Camera | Size | Where |
|---|---|---|
| `left`, `right` | 320 x 240, vertical field of view 50 deg | fixed at (1.15, +-0.55, 1.35), looking at (0.5, 0, 0.8) |
| `wrist` | 256 x 192, vertical field of view 75 deg | on the hand; part of the gripper shows at the image's bottom edge |
| `rack` | 640 x 480, vertical field of view 50 deg | fixed at (-0.09, 0.3625, 0.9555), looking at (0.36, 0.3625, 0.8755): along +x at the incoming rack's nominal place |

Every observation carries each camera's pose `T_world_<cam>` (4 x 4, OpenCV convention: x right, y down, z forward)
and intrinsics `K_<cam>` (a pixel's centre at its index + 0.5). Images carry pixel noise (sd 2 levels per channel).
There is no depth image.

**The poses are the calibration on file, not the cameras' true poses.** Each bench was calibrated once, and
`T_world_<cam>` is that calibration: the camera's true pose with a fixed error that is the unit's own (the same in
every episode on that unit, different from unit to unit and from every commissioning bench, not told):

| Camera | Error of the pose on file, in the camera's own frame |
|---|---|
| `wrist` (the hand-eye calibration: the camera's place on the hand) | up to 2 mm along each axis and 0.5 deg about each axis |
| `left`, `right`, `rack` (fixed) | up to 5 mm along each axis and 0.5 deg about each axis |

The intrinsics are exact, and so are `ee_pos`, `ee_quat` and `q`. A point located through a pose on file is
therefore off by a few millimetres: 2-4 mm from the wrist camera 0.2 m away, up to about 10 mm from a fixed camera.
What can be seen and felt on site tells the error: the fingers and whatever the hand holds are in the wrist image
at a place the kinematics give; the hand is in the fixed cameras' images where `ee_pos` says it is; a hole's rim
and floor are felt in `wrench`.

The pictures show the bench where it stands, in a bay of a laboratory: the room's wall benches, shelves, glassware,
floor and ceiling lights are in them, and the `rack` camera looks past the incoming rack into the room. Nothing of
the room is part of the task, and nothing of it stands on the bench: what you handle is described in sections 4 to
8, in the colours named there. The light is the bay's ceiling light at a fixed exposure; parts shade each other
softly where they meet, and nothing casts a sharp shadow.

## 4. The tubes

- 15 mL conical tubes: body 17 mm across with a rounded tip, 120 mm long including a blue cap 19 mm across and
  12 mm tall. Opaque; they all look alike.
- Mass: 15 g empty plus the liquid, 3-13 mL (1 mL = 1 g). The centre of mass is about 35 mm above the tip.
- The tubes to be spun (the worklist's and the ones already in the rotor) can always be paired with every opposite
  pair within 1 g: the two tubes of a pair differ by 0.2-0.8 g, tubes of different pairs by 1.2 g or more.

## 5. The incoming rack

- One row of ten positions A1-A10, 45 mm apart, each an orange sleeve: bore 18.5 mm (the tube plus 0.75 mm all
  round), 50 mm deep, wall 10 mm, rim at z = 0.865. A tube's tip rests at z = 0.8155.
- Nominally A1 is at (0.36, 0.16) and A10 at (0.36, 0.565), the row along +y. Per episode the rack's centre sits up
  to 30 mm off in x and in y, and the rack is turned up to 10 deg about it.
- 8-10 positions hold tubes; the others are empty.

## 6. The cold rack

- 3 x 3 positions, 45 mm apart, orange sleeves: bore 21 mm (the tube plus 2 mm all round), 50 mm deep, wall 10 mm,
  rim at z = 0.865.
- Rows A, B, C run along the rack's own x, columns 1, 2, 3 along its own y. Nominally the rack is unturned and B2
  (its centre) is at (0.20, 0.64). Per episode the centre sits up to 40 mm off in x and in y, and the rack is turned
  up to 20 deg.
- 2-4 positions already hold other people's tubes.

## 7. The centrifuge

- **Cabinet:** a square chamber, inner half width 0.18 m, walls 10 mm thick. Nominally its centre (the rotor's
  axis) is at (0.45, -0.10) and it is unturned. Per episode the centre sits up to 40 mm off in x and in y, and the
  cabinet is turned up to 20 deg about it. The chamber's walls end 30 mm above the rotor's top.
- **Rotor:** fixed-angle, eight positions 45 deg apart on a vertical hinge. "The rotor's top" is the plane of the
  hole entrances.
  - Each hole is a bore 100 mm deep, wall 10 mm, tilted 45 deg: its entrance is the upper end, and the bore runs
    down and away from the axis. Its width is the unit's own: the tube plus 0.6-1.0 mm all round (18.2-19.0 mm
    across; the table below). In a tight bore a tube that enters off its axis or is let go early stops
    short of the floor: it stands proud at the port, it does not count as seated, and the rotor cannot be turned
    past it.
  - The entrances lie on a circle of radius 65-75 mm around the axis, at z = 0.90-0.92. Both are drawn per episode.
  - Positions are numbered 1-8 counter-clockwise seen from above. Position 1 is marked by a green dot (12 mm across)
    on the rotor's bowl, 18 mm outside its entrance.
  - The bowl between the holes is a thin cover that is only drawn: what a hand or a tube can touch are the bores'
    walls, the hub and the knob.
  - The hub is a cylinder of radius 35 mm whose top is 10 mm below the rotor's top.
  - The rotor's start angle differs per episode. 0-2 positions already hold tubes from another job.
- **Shield and loading port:** a 2 mm plate fixed to the cabinet (it turns with it), yellow, over the hole ring.
  - Its underside is 23 mm above the rotor's top; a seated tube's cap passes under it.
  - It spans from 48 mm from the rotor's axis (the hub and the knob stay open) to 6 mm inside the hole circle.
  - One loading port cuts through it: a slot 24 mm wide, centred on the azimuth 160 deg about the rotor's axis in
    the cabinet's frame (the cabinet unturned: 160 deg from +x, counter-clockwise).
- **Ratchet:** the rotor turns counter-clockwise only (seen from above), on 2 deg teeth: it holds wherever it
  stops, and a turn the other way does not move it.
- **Knob:** a bar 70 mm x 12 mm x 18 mm lying on the hub's top, centred on the axis, 22.5 deg from position 1's
  direction (between two holes' planes). Its top is 8 mm above the rotor's top.
- **Lid:** a plate 0.36 m x 0.36 m, 12 mm thick, 0.4 kg, hinged along the chamber's -y edge at the height of the
  walls' tops (it turns with the cabinet). A handle bar 15 mm across and 100 mm long runs parallel to the hinge,
  45 mm beyond the plate's free edge, its axis in the plate's mid-plane (6 mm above the hinge's plane when the
  lid is shut), held by two flat arms 112 mm apart. The lid starts either open (1.75 rad, at its stop past vertical) or shut
  (0 rad). It counts as shut at an angle under 0.02 rad. The hinge's friction is the unit's own (the table
  below). The lid's weight makes 0.7 N m about the hinge when the lid lies flat: on a hinge with less friction a
  lid let go near the horizontal falls shut, on a stiffer one it stays where it is let go, at any angle.
- **What differs from unit to unit** (one centrifuge is one unit; a unit's values are its own and are not told):

  | Property | Range over the series |
  |---|---|
  | Friction of the rotor's hinge | 0.10-0.45 N m |
  | Play between the knob and the rotor (the bar turns on its own hinge within +-play/2 before it drives the rotor) | 0-20 deg |
  | Rotor inertia | 0.010-0.040 kg m^2 |
  | Radial clearance of the rotor's bores around a tube | 0.6-1.0 mm |
  | Friction of the bores against a tube (coefficient) | 0.35-0.58 |
  | Friction of the lid's hinge | 0.30-1.20 N m |
  | Calibration error of each camera's pose on file | section 3's table |

## 8. The scale

- A bench balance, base 90 x 90 x 30 mm, with one upright tube holder on top: bore 18.5 mm, 50 mm deep, its rim at
  z = 0.884. Nominally its centre is at (0.54, 0.32); per episode it sits up to 30 mm off in x and in y.
- `scale_g` in every observation is its display: the load on the holder in grams, in 0.1 g steps. An empty scale
  shows 0.0. The display follows the load with a 0.4 s lag and carries a little noise (sd 0.03 g). Anything that
  pushes on the holder or on the tube in it, the hand included, shows in the reading.

## 9. How a bench is laid out

Each episode's bench is drawn uniformly over the ranges above. A draw is used only if it keeps every incoming
position and every cold position at least 60 mm from the cabinet, every cold position 70 mm from every incoming
position, and the scale 100 mm from the cabinet and from every incoming position, and if the open lid's handle is
within 0.60 m of the robot's base on the table.
