# Task card: the counter, the jars, the numbers

`jar_sim.py` is the exact simulation the grader runs, `jar_env.py` the exact environment and `episode.py` the exact
credit and end. Read their docstrings and constants. This card collects the numbers. Where a place is given as nominal,
the episode draws the real one around it (`jar_sim.draw_episode`). Your controller has to find it.

## Frames and the robots

- World frame: z up, metres, radians, seconds; quaternions are (w, x, y, z). The counter top is at z = 0.80.
- Two Franka Pandas (robosuite 1.5.2 model) with their bases at (0, 0.28, 0.80) (`l`, the left arm) and
  (0, -0.28, 0.80) (`r`), both facing +x. Start posture `HOME_Q` for both.
- Control: 10 Hz (50 physics steps of 2 ms). Each arm takes one of the three action shapes in the instruction
  (`jar_env.py`): an operational-space step (at most 0.03 m and 0.2 rad), the same with a null-space posture, or a joint
  step (at most 0.1 rad a joint). Joint angles in an action are clipped to the joints' limits.
- Gripper: `grip` -1 = fully open (80 mm between the fingers) ... +1 = closed. The finger pads' friction is 2.0.
- Wrist force sensor: the force and torque at the wrist, in the world frame, with noise of 0.3 N and 0.03 N m.

## The jars

- Three per unit, one of each kind. The label band (36 mm tall, centred at 55 % of the jar's height) has the kind's
  colour: `sugar` white (0.96, 0.95, 0.90), `mint` green (0.25, 0.75, 0.40), `berry` purple (0.75, 0.20, 0.55). The
  jar's body is pale blue.
- Places: x = 0.55 +- 0.02, y = 0.17, 0 and -0.17, each +- 0.01; which jar stands at which place is drawn per episode.
- A jar is an open container: a 4 mm wall, a 12 mm floor, a shoulder 4 mm deep at the top, and a neck as wide as the
  lid's inside (lid radius - 4 mm outside, lid radius - 8 mm inside: the mouth) and `turns` x 5 mm tall.
- Per unit (hidden for the scored units; `units.json` gives the practice units'):

  | Property | Key in `units.json` | Range |
  |---|---|---|
  | Lid radius | `lid_r` | 0.027, 0.029, 0.031, 0.033 or 0.035 m; the unit's three differ |
  | Body radius | `jar_r` | `lid_r` - (0.001-0.003) m |
  | Height (to the shoulder) | `jar_h` | 0.10-0.14 m |
  | Mass (with its contents) | `mass` | 0.25-0.90 kg |
  | Friction of the jar's surface (jar on counter) | `mu` | 0.4-0.8 |
  | The thread's running torque | `tau` | 0.20-0.80 N m |
  | Turns to free the lid | `turns` | 1, 1.5, 2 or 2.5 |

## The lids and the thread

- All lids are red, 20 mm tall, 30 g, friction 1.0; they differ only in radius.
- In the thread a lid turns on the jar's axis and rises 5 mm per turn as it turns counter-clockwise (seen from above).
  Turning it takes the running torque `tau`. At the seat (fully closed) it stops; the torque the seat takes is the
  closing torque.
- After `turns` turns counter-clockwise the lid comes free (a free body).
- A free lid catches on its own jar's thread when all of these hold: its centre is within 3 mm of the neck's axis;
  its bottom is between 2 mm below and 4 mm above the neck's top; it is tilted less than 8 deg from the jar's axis; it
  turns clockwise relative to the jar (faster than 0.05 rad/s); it is within 0.3 rad of the thread's start. A lid that
  has just come free must first leave the neck (more than 3 mm off the axis, or its bottom more than 4 mm above the
  neck's top) before it can catch again. A lid never catches on another jar's thread.
- Seated: the lid within 0.02 rad of the seat. Closing torque window: 0.5-2.5 N m (the largest since the lid last
  caught counts).

## The tray and the pieces

- The tray: inner size 150 x 110 mm, centred at (0.32, -0.42), floor 4 mm, rim 6 mm tall.
- `sugar`: 16 mm cubes, 4 g, friction 0.8. `mint` and `berry`: capsules 24 mm long and 14 mm across, 3 g, friction
  0.8. Each in its kind's colour.
- The pieces lie on a 4 x 3 grid of cells about 36 x 35 mm (each within 4 mm of its cell's centre), capsules
  lengthwise along y (+- 0.3 rad).
- The tray holds, for each listed kind, the listed count plus 1-2 spare pieces, and 1-2 pieces of the unlisted kind.
- A piece is inside a jar when its centre is within the wall (closer to the jar's axis than `jar_r` - 4 mm) and between
  6 mm above the jar's bottom and the shoulder (`jar_h`). It is in the tray when its centre is over the tray's inside
  and below z = 0.83. It is off the counter when its centre is below z = 0.79 or outside x -0.15 to 0.85 or y -0.80 to
  0.80.

## Cameras

| Name | Size | Pose |
|---|---|---|
| `overhead` | 640 x 480, vertical field of view 55 deg | at (0.62, 0, 1.90) looking at (0.50, 0, 0.80) |
| `front` | 640 x 480, 45 deg | at (1.20, 0, 1.30) looking at (0.50, 0, 0.88) |
| `tray` | 640 x 480, 40 deg | at (0.44, -0.42, 1.22) looking at (0.32, -0.42, 0.80) |
| `wrist_l`, `wrist_r` | 320 x 240, 75 deg | on each hand (robosuite's eye-in-hand camera) |

Every observation carries each camera's pose `T_world_<cam>` (OpenCV convention: x right, y down, z forward) and
intrinsics `K_<cam>` (f = (H / 2) / tan(fov / 2), principal point (W / 2, H / 2)). Images carry pixel noise
(standard deviation 2 levels). There is no depth image.

## The episode

- At most 12000 control steps (20 simulated minutes). It ends earlier after 20 steps in which every listed jar has been
  opened and is seated with its own lid, both grippers are open at least 70 mm and both gripper sites are at least
  0.12 m from every jar's top (the neck's top centre).
- Unstable: an arm joint faster than 8 rad/s, or a jar, lid or piece faster than 5 m/s. The episode ends and scores 0.
- The refill list names two of the three jars by kind, 2-4 pieces each.
- Credit (`episode.py`): a listed jar is done when it holds exactly the listed pieces of its kind and nothing else,
  and is closed again (opened in the episode, its own lid seated, closing torque 0.5-2.5 N m). The episode's credit is
  the done jars over the listed jars, halved if the unlisted jar is not as it was, a jar is tilted more than 45 deg or
  a piece is off the counter.
