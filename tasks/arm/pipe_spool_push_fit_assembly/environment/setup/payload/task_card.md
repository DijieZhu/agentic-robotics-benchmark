# Task card: the cell, the parts, the numbers

`spool_sim.py` is the exact simulation the grader runs, `spool_env.py` the exact environment and `episode.py` the
exact credit and end. Read their docstrings and constants. This card collects the numbers. Where a place is given as
nominal, the episode draws the real one around it (`spool_sim.draw_episode`); your controller has to find it.

## Frames and the robots

- World frame: z up, metres, radians, seconds; quaternions are (w, x, y, z). The bench top is at z = 0.80 and spans
  x -0.28 to 0.80, y -0.80 to 0.80.
- Two Franka Pandas (robosuite 1.5.2 model) with their bases at (0, 0.28, 0.80) (`l`, the left arm) and
  (0, -0.28, 0.80) (`r`), both facing +x. Start posture `HOME_Q` for both (the hands point down).
- Control: 10 Hz (50 physics steps of 2 ms). Each arm takes one of the three action shapes in the instruction
  (`spool_env.py`): an operational-space step (at most 0.03 m and 0.2 rad), the same with a null-space posture, or a
  joint step (at most 0.1 rad a joint). The operational-space controller compensates the arm's own weight, not what
  the hand holds.
- Both arms have the Panda gripper, driven as the Franka Hand grasps: `grip` -1 = fully open (80 mm between the
  fingers) ... +1 = closed. Each finger is a position servo (4000 N/m) limited to 70 N, aimed 30 mm past closed at
  `grip` +1, so a closed hand squeezes what it holds with 70 N per finger. The finger pads' friction is 1.0
  (torsional 0.01 m). The tool frame (`ee_pos`, `ee_quat`) is the grip site between the fingertips; at `HOME_Q` its
  x axis is the world's +y (the fingers close along it) and its z axis points down.
- Wrist force sensor (both arms): the force and torque at the wrist, in the world frame, with noise of 0.3 N and
  0.03 N m.

## The parts

- **Pipes.** Copper, 22 mm outside diameter, the ends chamfered 1.5 mm. Pipe A is 0.22-0.30 m long, pipe B 0.19-0.23 m
  (per episode). Both lie loose on the bench roughly along y (each turned by up to 30 deg about z), their axes 11 mm
  above the bench; nothing holds them, and a pipe that is pushed rolls. Nominal centres: A (0.40, 0.24), B (0.58,
  0.18); each episode moves each by up to 15 mm in x and y. A carries a blue band 12 mm wide whose centre is 25 mm from
  one end, its **banded end**; which end is drawn per episode.
- **Elbows.** Two identical 90 deg push-fit elbows, light grey, on the bench. Nominal corners (seen from above):
  E0 (0.42, -0.24), E1 (0.62, 0.44); each episode moves each by up to 15 mm and turns it by any angle about z. Each
  elbow either lies flat (its plane horizontal, the corner's centre 15 mm above the bench) or stands on one socket's
  mouth (that socket's axis pointing down, the corner's centre 50 mm above the bench, the other socket level along
  the elbow's turn). Per episode at least one elbow stands: both with probability 1/2, otherwise one of the two, either
  alike; a standing elbow stands on either socket alike. An elbow: a corner block (38 x 38 x 30 mm) and two sockets at 90 deg, each a barrel 30 mm across whose
  mouth is 50 mm from the corner's centre; the elbow is 30 mm thick everywhere.
- **A socket.** The bore is 22 mm + twice the unit's clearance across and 30 mm deep from the mouth. 4 mm inside the
  mouth sits the O-ring; deeper, at the unit's stop depth, the stop. Nothing on the outside shows the stop depth.

## The push-fit joint (`spool_sim.World._joints`)

- **Engaging.** A pipe end pressed against a socket's O-ring goes past it only when the end's axial push on the
  O-ring is above the unit's insertion force and the pipe's axis is within 1 deg and 0.5 mm of the socket's axis.
  Otherwise it rests on the O-ring however hard it is pushed.
- **Sliding.** Past the O-ring the pipe moves in the socket only along the socket's axis, against a friction equal to
  the insertion force, up to the stop, and turns about the axis against the unit's turning torque.
- **Home.** The pipe's end within 1 mm of the stop. A home joint locks (the fitting's grip ring): the pipe cannot be
  drawn back any more; it still turns against the turning torque.
- **Coming apart.** A pipe that is not home and is drawn back until its end is less than 3.5 mm inside comes out.
- **Made in the air.** A joint is marked as made on a support if, at the moment its pipe goes past the O-ring or at
  any moment while its pipe advances in the socket (faster than 1 mm/s), any part connected to it (through joints)
  touches anything other than the two grippers and the other spool parts: the bench, a robot's arm links. A
  joint made on a support does not count. The mark goes only when the joint comes apart, which a home joint cannot.

## What is hidden per unit

| Property | Range |
|---|---|
| Insertion force | 10-25 N |
| Stop depth (from the mouth) | 20-28 mm |
| Turning torque of a joint | 0.5-1.2 N m |
| Bore clearance (radial) | 0.2-0.4 mm |
| Elbow mass | 0.08-0.16 kg |
| Pipe mass per metre | 0.45-0.75 kg/m |

## The work order and the credit

- The observation's `order` is `{"twist_deg": phi}`, phi 0 or 180.
- The spool: an elbow on A's banded end (call it the B-elbow), B's one end in the B-elbow's other socket, the other
  elbow on A's other end, its free socket turned phi from the B-elbow's B-socket about A's axis pointing from the
  B-elbow to the other elbow (both sockets' outward axes projected on the plane normal to A; the angle runs
  counter-clockwise from the B-socket's to the free socket's when A's axis points at you; 0: the same way, 180:
  opposite ways).
- Credit: 0.2 per joint that is home, made in the air, with every spool part's centre of mass above the bench top
  (at most three); 0.4 more when the spool has the order's shape (the three joints count, the B-elbow on A's banded
  end, the twist within 5 deg).
- The episode ends after 6000 control steps; or after 20 steps in a row with three joints home, both grippers open at
  least 70 mm, both grip sites at least 8 cm from every spool part and every part slower than 2 cm/s; or at an
  unstable simulation (an arm joint faster than 8 rad/s or a part faster than 5 m/s), which scores 0.

## Cameras

| Camera | Size | Where |
|---|---|---|
| `overhead` | 640 x 480, 62 deg | at (0.30, 0, 2.10) looking at (0.48, 0, 0.80) |
| `front` | 640 x 480, 50 deg | at (-0.35, 0, 1.60) looking at (0.55, 0, 0.90) |
| `side` | 640 x 480, 50 deg | at (0.45, -0.95, 1.25) looking at (0.47, 0, 0.95) |
| `wrist_l`, `wrist_r` | 320 x 240, 75 deg | on each hand, looking along the fingers |

Each observation carries every camera's pose and intrinsics; the images only when the previous action asked for them
(the first observation has all five). Images have pixel noise of 2 levels. There is no depth image.
