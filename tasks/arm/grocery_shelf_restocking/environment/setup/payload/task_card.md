# Task card: the scene's numbers

Metres, radians unless stated; world z up. `shelf_world.py` is the source of every number here.

## The robot

- A Franka Panda (robosuite 1.5.2's model) with its parallel gripper; fingers open to 80 mm.
- Its base sits on a vertical lift at world (0, 0, `lift`), facing +x. The lift's height runs 0.05-1.30 m and its
  set point moves at most 0.01 m per control step (0.1 m/s). An episode starts with the lift at 0.60 m and the arm
  in its start posture.
- The controller: box transport's operational-space controller at 10 Hz (`shelf_ctl.LiftCtl`, `box_transport_env`).
  `shelf_sim.Kin` is a kinematic IK you may use.
- The gripper site (`ee_pos`) is between the fingertips. The hand is 0.21 m across along the fingers' axis and
  0.064 m thick; its palm starts 0.031 m behind the site; link 6 reaches 0.285 m behind it.

## The bay

- Frame: x into the bay, y to the left as seen from the aisle, z up; origin on the floor at the front plane (the
  lips' outer face), centred on the lanes. Nominal: front plane at world x = 0.50, centred on y = 0, facing the
  robot. Per episode: shifted by up to 0.02 m in x and in y, turned by up to 5 deg about z.
- Three boards; the bottom board's top at 0.26-0.34 m. Each board is 0.02 m thick; above board i the headroom (to the
  underside of the next board, or of the bay's top) is the shelf's product height plus 0.32-0.35 m.
- A lip on each board's front: 0.02 m tall, 0.01 m thick. Usable depth (lip's inner face to the back panel):
  0.24-0.28 m.
- Five lanes per shelf, 0.115-0.13 m apart; lane 2 on the bay's centre line, lane 0 on the right (-y). The bay's
  sides stand 0.17 m beyond the outer lanes' centre lines.
- A lane's slots: slot 0's front face 12 mm behind the lip's inner face, each next slot 12 mm behind the one in front
  (`shelf_world.slot_x`); a lane holds as many slots as have their centre at least 0.06 m in front of the back panel
  (`shelf_world.capacity`: 2-4 by the product's depth and the bay's depth). A restocked lane ends with every slot
  full.

## The products

Each shelf has one size class and one height (its two SKUs share it). In about half the bays two shelves have the
same class and the same height. Each dimension is 0.92-1.05 times the nominal, drawn per bay (the height once per
class, the footprint per SKU). No two SKUs of a bay share a look (body colour, band colour, logo colour and shape).

| Class | Shape | Size (nominal) | Mass |
|---|---|---|---|
| box | box | 0.05 deep x 0.07 wide x 0.14 tall | 0.25-0.40 kg |
| can | cylinder | 0.066 diameter x 0.12 tall | 0.33-0.42 kg |
| bottle | cylinder | 0.064 diameter x 0.19 tall, with a cap | 0.45-0.65 kg |
| tall box | box | 0.06 x 0.06 x 0.20 | 0.35-0.55 kg |

A box's printed front (its logo) is its -x face in its own frame: a box faces the aisle when its yaw equals the
bay's. No product is more than 0.07 m wide across a box's width or a cylinder's diameter.

The date code (`shelf_world.add_code`): two seven-segment digits, each 7 mm tall and 4 mm wide with 0.8 mm strokes,
2 mm apart, dark (0.08, 0.08, 0.10) on a light patch 11.5 x 10 mm (0.96, 0.96, 0.93) standing 2.4 mm proud of the
side. Its place is drawn for each unit (`shelf_world.draw_code_at`, from a random stream of its own; a world's `meta`
records it as `code_at`): on a box, one of its four sides, the patch at least 2 mm in from the side's vertical edges;
on a can or bottle, any angle about its axis; the patch's centre at a height in the unit's upper 70 % (a bottle's
body, not its cap) and its top at least 40 mm under the unit's top; never over the logo (2 mm clear of the logo's
square on a box's front; on a can or bottle, level with the logo, at least atan((logo half + patch half) / radius) +
0.15 rad round from it). Old units of a box SKU stand face out; cans, bottles and every unit in the tote
stand at any turn (a tote box faces +x or -x within 8 deg).

## The cart and the tote

- The cart's top is at 0.55 m. The tote (inner 0.34 x 0.27, walls 0.04 m tall) is centred at world (0.22, -0.55).
- The tote's units stand upright in a 4 x 2 grid of slots (0.085 m apart along x, 0.17 m along y): the delivery,
  then extras (units of the bay's other SKUs, not part of the delivery) until zero or one slot is free.

## The cameras

| Camera | Size | Vertical field of view | Mount |
|---|---|---|---|
| `left`, `right` | 400 x 300 | 60 deg | the lift's carriage, at (0.15, ±0.40, 0.42) from the base, looking at (0.66, 0, 0.28) from the base |
| `wrist` | 256 x 192 | 75 deg | the hand |
| `overview` | 640 x 480 | 55 deg | fixed at world (-0.55, 0.55, 1.75), looking at (0.66, 0, 0.85) |

Images carry noise (2 levels). Poses and intrinsics come with every observation.

## The episode

- A unit is a bay design and its six SKUs; an episode on it draws the bay's pose, `today` (20-70), the planogram
  (each lane's SKU), three lanes to restock of three different SKUs, two of them on one shelf and one on another
  (each with one old unit up to its capacity; the delivery fills it once its expired units are gone), the other
  lanes' stock (faced, in date order), the tote's contents, every unit's code and the noise.
- Graded episodes (and the practice units' `seeds`) are those in which the tote alone is not enough: with every
  unit known, no sequence that sets a unit down in a lane only at its final place restocks every lane, while a
  strategy that reads a unit only once it is out of its lane can restock them all.
- Codes: an old unit's day number is today to today + 14, or, expired, today - 9 to today - 1; a delivered unit's
  is mostly today + 2 to today + 24, but may be earlier than an old unit's. In at least two of the three lanes the
  codes matter (an expired unit, or a delivered unit earlier than an old one).
- The episode ends on `[0]`, after 15000 control steps, or at an unstable simulation.
