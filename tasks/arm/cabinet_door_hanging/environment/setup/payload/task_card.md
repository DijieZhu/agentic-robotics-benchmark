# Task card: the cell, the cabinet, the numbers

`door_sim.py` is the exact simulation the grader runs, `door_env.py` the exact environment and `episode.py` the exact
credit and end. Read their docstrings and constants. This card collects the numbers. Where a place is given as
nominal, the episode draws the real one around it (`door_sim.draw_episode`); your controller has to find it.

## Frames and the robots

- World frame: z up, metres, radians, seconds; quaternions are (w, x, y, z). The table top is at z = 0.80; the table
  ends at x = 0.55. Beyond it is the wall cabinet; nothing is below the cabinet's door.
- Two Franka Pandas (robosuite 1.5.2 model) with their bases at (0, 0.28, 0.80) (`l`, the left arm) and
  (0, -0.28, 0.80) (`r`), both facing +x. Start posture `HOME_Q` for both (the hands point down).
- Control: 10 Hz (50 physics steps of 2 ms). Each arm takes one of the three action shapes in the instruction
  (`door_env.py`): an operational-space step (at most 0.03 m and 0.2 rad), the same with a null-space posture, or a
  joint step (at most 0.1 rad a joint).
- Left arm, the Panda gripper: `grip` -1 = fully open (80 mm between the fingers) ... +1 = closed. Each finger is a
  position servo (1000 N/m, at most 20 N) toward 0.02 (1 - `grip`) m from the centre, so a closed grip squeezes the
  12 mm grip post with about 6 N per finger. The finger pads' friction is 2.0 (torsional 0.01 m). Its tool frame
  (`ee_pos`, `ee_quat`) is the grip site between the fingertips; at `HOME_Q` its x axis is the world's +y (the fingers
  close along it) and its z axis points down.
- Right arm, the vacuum cup tool: an L-shaped bracket on the flange, a post 90 mm down from the flange, then a cup of
  radius 30 mm whose face is 45 mm in front of the post, facing along the hand's x axis (forward when the hand points
  down). Its tool frame is the cup face's centre, its z axis along the direction the cup faces. `grip` > 0 asks for
  vacuum, `grip` <= 0 releases it. A camera on the post looks the way the cup faces (`wrist_r`).
- Wrist force sensor (both arms): the force and torque at the wrist, in the world frame, with noise of 0.3 N and
  0.03 N m.

## The vacuum cup

- **Seal.** With vacuum on, the cup seals on the door when all of these hold: its face centre is within 3 mm of the
  door's front face plane; its axis is within 10 deg of the door's normal; the whole cup (radius 30 mm) is on the
  leaf. From then on it holds the door through a compliant joint (time constant 0.02 s).
- **Break.** The seal breaks when vacuum is switched off, or when the load it holds stays above 150 N (force) or
  15 N m (moment about the cup face's centre) for 20 ms. The door is then let go (it falls when nothing else holds
  it). With vacuum still on, the cup reseals as soon as the seal conditions hold again.
- The observation's `vacuum` (right arm) is true while the cup holds a door (either one).

## The cabinet and the doors

- **The carcass frame:** its origin is on the hinge axis at the hung door's bottom edge; its z axis is vertical, its
  x axis points into the cabinet (the hung door's front face faces -x). Nominal origin (0.589, 0.180, 0.86) and yaw 0;
  each episode moves it by up to 8 mm in x, y and z and turns it by up to 2 deg about z.
- The opening has its hinges on the left (+y) side; the door leaf extends from 14 mm to 414 mm on the -y side of the
  axis. The carcass's side panels start 15 mm behind the axis; the door overlays the right panel, which stops it when
  closed. The carcass is wood coloured; its knuckles are steel grey.
- **The new door:** 0.40 m wide, 0.45 m tall, 18 mm thick, pale beige with brass knuckles; mass 1.5-3.5 kg (per
  unit); a dark handle (24 x 16 x 120 mm, 0.1-0.4 kg per unit) on its front face, 40 mm from the free edge, at
  mid-height. Its frame (`door_sim` body `door`): origin on its knuckles' axis at its bottom edge, the leaf toward its
  -y, its front face at x = -9 mm. Hung, the door frame coincides with the carcass frame. Only this door counts.
- **The old door** (`door_sim` body `old`): the same model, mass and handle, dark brown with dark knuckles. At the
  start it hangs in the opening (its frame placed on the carcass frame) on two old pins.
- **The new door's rack:** the door stands upright on the table in a channel (two rails 40 mm tall, 1 mm clear of
  the door on each side), its front face toward the robots (-x), its frame's origin nominally at (0.38, 0.10) on the
  table; each episode moves it by up to 8 mm in x and y.
- **The parking rack:** the same channel, empty, with 8 mm clear of a door on each side, nominally at (0.50, -0.36)
  (a door frame's origin standing in it); each episode moves it by up to 8 mm in x and y. It is a place for the old
  door; any place where the old door stays up will do.
- **Other things with contact** (`door_parts.contact_shapes`; none is needed, none is in the job's way): on the
  table a spirit level (x -0.257..-0.224, y 0.42..0.78, 52 mm tall), a mallet (head at x -0.235, y -0.70..-0.61,
  handle toward x -0.05, y -0.73), a punch (x -0.20..-0.07, y -0.56..-0.585), a tape measure (x -0.25..-0.176,
  y -0.50..-0.435, 32 mm tall) and a pencil (x -0.15..-0.02, y -0.47..-0.455); a tool bag on the floor (0.41 x 0.23
  x 0.30 m at x 0.05, y -1.12, turned 20 deg); the rail, two brackets and a spacer between the cabinet's back and the
  wall; on the cup tool three bellows rings behind the pad (radius 28.5 mm), a fitting and a hose with clips on the
  post's back. The hinge leaves, the felt pieces and the room in the pictures have no contact.
- **The old door must not fall.** When its centre of mass comes below z = 0.75 (5 cm under the table top) the
  episode ends and scores 0.

## The hinges

- Two per opening: the bottom one centred 0.05, 0.06 or 0.07 m above the door's bottom edge, the top one 0.38, 0.39 or
  0.40 m (per unit).
- Each hinge is a 75 mm stack of 3 or 5 knuckles (per unit; the same for both hinges), alternately carcass and door,
  carcass knuckles at both ends. A door knuckle is 1 mm shorter than its slot at each end. The door's knuckles are
  brass coloured.
- Bore = the pin's radius + the radial clearance (0.2-0.5 mm per unit). Every knuckle's outer radius is 9.0 mm,
  whatever its bore.
- A door hangs only on its pins: with no pin it falls out of the knuckles.

## The pins

- Three classes by diameter: S 5 mm, M 6.5 mm, L 8 mm. A unit's hinges take one class (per unit). A pin of a smaller
  class goes into a larger bore but does not count; a pin of a larger class does not go in.
- A pin: a shaft 71 mm long below its head (rounded tip); a head of radius (pin radius + 3 mm), 3 mm thick; an arm
  from the head to a square grip post 12 mm wide and 36 mm tall whose centre is 50 mm from the pin's axis and 27 mm
  above the head's underside; 80 g in all. In the stands the arm points toward -x.
- **Stands:** six, at x = 0.45 and y = 0.30, 0.38, 0.46, 0.54, 0.62 and 0.70, each a tube 125 mm tall (inner radius
  pin radius + 1.5 mm, outer radius pin radius + 8 mm) whose top carries the pin's head (at z = 0.925). Two hold pins
  of each class; which stand holds which is drawn per episode, the same way for every unit.
- **The detent.** While its tip is inside a hinge's bore and it is within 2 mm of the axis, a pin of the unit's class
  slides only under a push of about 6 N (it never drops through under its own weight). A pin of a smaller class has
  no detent: it slides through freely, never clicks, and rests on its head. When its head comes within 0.7 mm of its seat (the
  top knuckle's top face) it clicks in: its head's centre is held there, it still turns about its axis, and it comes
  out only when pulled up harder than 40 N.
- **The old pins:** two, one home in each hinge of the old door at the start (head on the top knuckle, arm toward
  -x). The same shape as the others but 4 mm across (radius 2 mm, head radius 5 mm): thinner than every class, so
  they slide freely in any bore (no detent) and never count. Pulled out, put them anywhere on the table.
- **Home** (for the credit): on the hinge's axis within 1.5 mm, tilted at most 3 deg, its head at most 3 mm above
  the top knuckle's top face.

## Units

| Property | Key in `units.json` | Range |
|---|---|---|
| Knuckles per hinge | `n_kn` | 3 or 5 |
| Pin class | `pin_class` | `S`, `M` or `L` |
| Radial clearance | `clear` | 0.0002-0.0005 m |
| Bottom hinge centre above the door's bottom | `hb` | 0.05, 0.06 or 0.07 m |
| Top hinge centre | `ht` | 0.38, 0.39 or 0.40 m |
| Door mass | `door_mass` | 1.5-3.5 kg |
| Handle mass | `handle_mass` | 0.1-0.4 kg |

`units.json` gives the two practice units' values. The scored units' are not given.

## Cameras

| Camera | Where | Image |
|---|---|---|
| `overhead` | (0.25, 0, 2.05) looking at (0.42, 0, 0.80), vertical fov 70 deg | 640 x 480 |
| `front` | (-0.30, 0, 1.65) looking at (0.60, 0, 1.05), fov 50 deg | 640 x 480 |
| `side` | (0.22, 0.58, 1.22) looking at (0.59, 0.18, 1.07), fov 42 deg | 640 x 480 |
| `wrist_l` | on the left hand (robosuite's eye-in-hand camera), fov 75 deg | 320 x 240 |
| `wrist_r` | on the cup tool's post, looking the way the cup faces, fov 75 deg | 320 x 240 |

Every observation gives each camera's exact pose and intrinsics. Images carry noise of 2 levels (of 255). No depth.

## Limits

- Unstable (the episode scores 0): an arm joint faster than 8 rad/s, or a door or a pin faster than 5 m/s.
- The old door falling (its centre of mass below z = 0.75) ends the episode with 0.
- An episode: at most 6000 control steps (10 simulated minutes).
