# Task card: the bench, the units, the numbers

`lab_sim.py` is the exact simulation the grader runs, `lab_env.py` the exact environment and `episode.py` the
exact credit. Read their docstrings and constants. This card collects the numbers. Where a place is given as
nominal, the episode draws the real one around it (`lab_sim.bench_draw`). Your controller has to find it.

## Frames and the robot

- World frame: z up, metres, radians, seconds; quaternions are (w, x, y, z). The table top is at z = 0.80 and spans
  x from -0.15 to 0.75 and y from -0.80 to 0.80.
- A Franka Panda (robosuite 1.5.2 model) with its base at (0, 0, 0.80), facing +x. Start posture `HOME_Q`
  (`box_transport_env.py`).
- Control: 10 Hz (50 physics steps of 2 ms). The three action shapes are in the instruction and `lab_env.py`.
- Gripper: `grip` -1 = fully open (80 mm between the fingers) ... +1 = closed.

## The tubes

- 15 mL conical tubes: 17 mm across, 120 mm long including a blue cap 19 mm across and 12 mm tall.
- Mass: 15 g empty plus the liquid (1 mL = 1 g), 3-13 mL.
- The tubes are opaque, so the liquid does not show. They all look alike: a tube is known by where it stands.
- The tubes to be spun (the worklist's and the ones already in the rotor) can always be paired with every
  opposite pair within 1 g.

## The incoming rack

- One row of ten positions A1-A10, 45 mm apart, with orange rings. Nominally A1 is at (0.36, 0.16) and A10 at
  (0.36, 0.565), the row along +y.
- Each episode the rack's centre sits up to 30 mm off in x and in y, and the rack turns up to 10 deg about it.
- Holes are 50 mm deep and 0.75 mm wider than the tube all round, with their rims at z = 0.865. A tube's tip rests
  at z = 0.8155.
- 8-10 positions hold tubes; the others are empty.

## The cold rack

- 3 x 3 positions, 45 mm apart, with orange rings.
  - Rows A, B, C run along the rack's own x, columns 1, 2, 3 along its own y.
  - Nominally the rack is unturned and B2 (its centre) is at (0.20, 0.64).
  - Each episode the centre sits up to 40 mm off in x and in y, and the rack turns up to 20 deg.
- Holes are 50 mm deep and 2 mm wider than the tube all round, with their rims at z = 0.865.
- 2-4 positions already hold other people's tubes. Leave them where they are.

## The centrifuge

- **Cabinet:** a square chamber with an inner half-width of 0.18 m.
  - Nominally its centre (the rotor's axis) is at (0.45, -0.10) and it is unturned. Each episode the centre sits up
    to 40 mm off in x and in y, and the cabinet turns up to 20 deg about it.
  - The chamber's walls end 0.03 m above the rotor's top.
- **Rotor:** a fixed-angle rotor on a damped vertical hinge, with eight holes 45 deg apart.
  - Each hole is tilted 45 deg with its top leaning toward the axis. Holes are 100 mm deep and 1 mm wider than the
    tube all round.
  - The hole entrances lie on a circle of radius 65-75 mm around the axis, at a height of z = 0.90-0.92. Both are
    drawn per episode.
  - Positions are numbered 1-8 counter-clockwise seen from above. Position 1 is marked by a green dot on the
    rotor's bowl just outside it.
  - The rotor's start angle differs per episode.
  - 0-2 positions already hold tubes from another job. They are spun too, so leave them where they are; the
    worklist gives their masses.
- **Shield and loading port:** a 2 mm plate fixed to the cabinet (it turns with it), yellow, over the hole ring.
  - Its underside is 23 mm above the rotor's top; a seated tube's cap passes under it.
  - It spans from 48 mm from the rotor's axis (the hub and the knob stay open) to 6 mm inside the hole circle.
  - One loading port cuts through it: a slot 24 mm wide, centred on the azimuth 160 deg about the rotor's axis in
    the cabinet's frame (the cabinet unturned: 160 deg from +x, counter-clockwise). A hole is loaded only when
    the rotor brings it to the port.
- **Ratchet:** the rotor turns counter-clockwise only (seen from above), on 2 deg teeth: it holds wherever it
  stops, and a turn the other way does not move it.
- **Knob:** a bar 70 mm x 12 mm x 18 mm on the hub, lying between two holes' planes (22.5 deg from position 1's
  direction). With play, the bar turns on its own hinge within +-play/2 before it drives the rotor. Inside the
  shield's opening a hand grips the bar up to about 18 mm from the axis (farther out, the fingers meet the shield).
- **Lid:** hinged along the chamber's -y edge (it turns with the cabinet), with a handle bar beyond its free edge.
  - It starts either open (1.75 rad, at its stop past vertical) or shut (0 rad). Friction holds it where it is
    left.
  - It counts as shut at an angle under 0.02 rad.

## The scale

- A bench balance with one upright tube holder on top.
  - Base: 90 x 90 x 30 mm.
  - Holder: like an incoming-rack hole, 50 mm deep and 0.75 mm wider than the tube all round, its rim at
    z = 0.884.
  - Nominally its centre is at (0.54, 0.32). Each episode it sits up to 30 mm off in x and in y.
- `scale_g` in every observation is its display. It shows the load on the holder in grams, in 0.1 g steps.
  - Tare: an empty scale shows 0.0.
  - The display follows the load with a 0.4 s lag and carries a little noise.
  - Anything that pushes on the holder or on the tube in it, the hand included, shows in the reading.

## The bench draw

- Each episode's bench is drawn uniformly over the ranges above. A draw is used only if it keeps every incoming
  position and cold position at least 60 mm from the cabinet, every cold position 70 mm from every incoming
  position, and the scale 100 mm from the cabinet and from every incoming position. The open lid's handle must
  also stay within 0.60 m of the robot's base on the table.
- `lab_sim.bench_draw` is the exact rule.

## Cameras

| Camera | Size | Where |
|---|---|---|
| `left`, `right` | 320 x 240, fovy 50 | fixed at (1.15, +-0.55, 1.35), looking at (0.5, 0, 0.8) |
| `wrist` | 256 x 192, fovy 75 | on the hand; part of the gripper shows at the image's bottom edge |
| `rack` | 640 x 480, fovy 50 | fixed 0.45 m on the robot's side of the incoming rack's nominal place, 8 cm above the tubes' middle, looking along +x at the row |

Images carry pixel noise (sd 2 per channel). Poses and intrinsics come with every observation.

## The units

A unit is one centrifuge's rotor. The scored units span these ranges; the practice units (`units.json`) sit at
their low end:

| Property | Practice units | Scored units |
|---|---|---|
| Hinge friction | 0.10-0.15 N m | 0.10-0.45 N m |
| Knob play | 0-4 deg | 0-20 deg |
| Rotor inertia | 0.010-0.015 kg m^2 | 0.010-0.040 kg m^2 |

Everything else is drawn per episode, not per unit.
