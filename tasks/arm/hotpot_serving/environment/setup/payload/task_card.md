# Task card: the counter, the hotpot, the numbers

`soup_sim.py` is the exact simulation the grader runs, `soup_env.py` the exact environment and `episode.py` the exact
credit and end. Read their docstrings and constants. This card collects the numbers. Where a place is given as
nominal, the episode draws the real one around it (`soup_sim.draw_episode`). Your controller has to find it.

## Frames and the robots

- World frame: z up, metres, radians, seconds; quaternions are (w, x, y, z). The counter top is at z = 0.80.
- Two Franka Pandas (robosuite 1.5.2 model) with their bases at (0, 0.28, 0.80) (`l`, the left arm) and
  (0, -0.28, 0.80) (`r`), both facing +x. Start posture `HOME_Q` for both.
- Control: 10 Hz (50 physics steps of 2 ms). Each arm takes one of the three action shapes in the instruction
  (`soup_env.py`): an operational-space step (at most 0.03 m and 0.2 rad), the same with a null-space posture, or a
  joint step (at most 0.1 rad a joint). Joint angles in an action are clipped to the joints' limits.
- Gripper: `grip` -1 = fully open (80 mm between the fingers) ... +1 = closed. Each finger is a position servo
  (`GRIP_KP`, 4000 N/m) whose target at grip +1 is 3 cm past closed (`GRIP_PAST`), limited to 70 N (`GRIP_FORCE`): a
  closed hand squeezes what it holds with up to 70 N. The finger pads' friction is 2.0, their torsional friction
  0.01 m (`PAD_TORSION`).
- Wrist force sensor: the force and torque at the wrist, in the world frame, with noise of 0.3 N and 0.03 N m.

## The burner, the knob and the pot

- The burner: a ring of radius 0.14 m, 4 mm high, centred at (0.45, 0) (`STOVE_XY`). It heats the pot's broth with
  the knob's share of the unit's `burner_kw` while the pot is on the ring (`Liquid.burner_power`).
- The knob (`KNOB_*`): at (0.25, -0.22) (`KNOB_XY`), a cap of radius 18 mm and height 20 mm on a 6 mm base, a fin
  8 x 32 x 24 mm on top (thin along x at angle 0); a hinge about z from 0 (off) to pi/2 (`KNOB_MAX`, full power),
  with friction so it stays where it is turned; below 0.05 rad (`KNOB_OFF`) the burner is off. The observation
  carries its angle (`knob`).
- The pot (`POT_*`): inside radius 0.125 m, inside depth 0.095 m from the floor's top to the rim, walls 6 mm, floor
  8 mm. It stands on the ring, its centre within 1 cm of the ring's centre, turned about z by at most 0.05 rad.
- Two grip tabs (`TAB_HALF`, `TAB_Z`): blocks 32 mm along x, 60 mm along y and 32 mm along z, sticking out from the wall
  on the pot's +y and -y sides (their outer ends about 0.19 m from the pot's centre), their centres 65 mm above the
  floor's top.
- The pot's empty mass is the unit's `pot_mass` (its centre of mass on its axis, 40 % of its height up).

## The broth and its heat

- The broth (`RHO`): 1030 kg/m^3, clear (its surface drawn in `SOUP_RGBA`, half transparent), the unit's `soup_l`
  litres in the pot at the start, at 80 C (`BROTH_T0`).
- Outflow over a rim: q = (2/3) Cd sqrt(2 g) d^1.5 m^3/s per metre of rim, d the height of the surface above that
  point of the rim, Cd the unit's `cd`; the surface is level and holds the vessel's volume at the vessel's pose
  (`Vessel`, `Liquid` in `soup_sim.py`). Updated at 100 Hz; the stream's parcels fly ballistically.
- Heat (`Liquid._heat`, 100 Hz): the pot's broth, at most 100 C (`BOIL`), gains the burner's power and loses
  9 W/K x (T - 20 C) (`POT_LOSS`, `ROOM`) and the heat flowing into the pieces under it; its heat capacity is
  4000 J/(kg K) (`BROTH_C`) at 1 kg per litre. A piece under the pot's broth (at least 30 % of its volume, taken as
  a sphere of its volume) warms as dT/dt = (T_broth - T) / tau, tau the kind's `tau` times the unit's `piece_scale`
  squared, and takes mass x 3600 J/(kg K) (`PIECE_C`) x dT/dt; anywhere else it cools toward 20 C with a 600 s time
  constant (`AIR_TAU`).
- Doneness (`Liquid.cooked`, `Liquid.overcooked`): cooked once the core has reached the kind's `t_cook`; overcooked
  once it has spent more than the kind's `window` under the pot's broth since. The colour (`Liquid.piece_rgba`) moves
  from the kind's raw to its cooked colour as the hottest core temperature so far goes from 20 C to `t_cook`, then to
  its overcooked colour over the window.

## The dishes and the pieces

- Four dishes (`DISH_*`), nominal centres `DISH_XY` (0.36, 0.34), (0.52, 0.34), (0.36, 0.49), (0.52, 0.49), each
  within 1 cm: inner radius 50 mm, a 4 mm floor and a 2 mm lip (3 mm thick), 0.12 kg; a flat grip tab 44 x 24 x 8 mm
  sticking out toward -x, its centre 22 mm above the dish's underside (`DISH_TAB`, `DISH_TAB_Z`). Which dish holds
  which kind is drawn per episode (`dish_of`).
- The pieces (`KINDS`, `PIECE`), sizes times the unit's `piece_scale`, friction the unit's `piece_mu`:

| Kind | Shape (half sizes, m) | Density (kg/m^3) | tau (s) | t_cook (C) | window (s) | RGB raw / cooked / overcooked |
|---|---|---|---|---|---|---|
| `fishball` | sphere 0.012 | 1080 raw, 960 cooked | 90 | 72 | 270 | (0.80, 0.80, 0.78) / (0.97, 0.96, 0.92) / (0.80, 0.72, 0.52) |
| `shrimp` | capsule radius 0.0075, half length 0.008 (lying) | 1070 | 50 | 63 | 240 | (0.55, 0.58, 0.60) / (0.96, 0.52, 0.36) / (0.70, 0.33, 0.22) |
| `tofu` | box 0.012 | 550 | 150 | 70 | 360 | (0.96, 0.90, 0.70) / (0.90, 0.72, 0.38) / (0.62, 0.42, 0.20) |
| `mushroom` | ellipsoid 0.014, 0.014, 0.008 | 700 | 60 | 65 | 240 | (0.86, 0.80, 0.68) / (0.55, 0.42, 0.30) / (0.30, 0.22, 0.15) |

- Under a surface a piece gets buoyancy (its submerged volume taken as that of a sphere of its volume) and a drag of
  0.6 N s/m at full immersion (`DRAG`); a fish ball's extra lift grows with its doneness.
- The order (`soup_sim.draw_episode`): per bowl 2-3 pieces (`PIECES_PER_BOWL`), each of a kind at random, no kind more
  than 4 in all (`PIECES_PER_KIND_MAX`), and 100-250 ml of broth in steps of 10 ml. Each dish holds the order's count
  of its kind and one more (`PIECE_EXTRA`).
- A piece any robot geom has touched during the episode (`World.touched`, checked every 2 ms substep) spoils the bowl
  it ends in.

## The ladle

- A slotted skimmer (`LADLE_*`): a dish of radius 45 mm whose floor is five 14 mm bars with 4 mm slots, a 15 mm rim
  round its back and sides, open at the front over 127 deg; a handle 25 cm x 18 mm x 10 mm leaving the back of the rim
  and rising at 70 deg from the dish's plane; 0.15 kg. In its own frame the front is +x and the handle leaves at -x.
- It starts standing in the pot, its dish on the pot's floor 3.5 cm to the -y side of the pot's centre, its front
  toward +y and its handle leaning toward -y (`LADLE_IN_POT`, `LADLE_YAW`).

## The bowls

- White, blue and yellow (`BOWL_*`): inside radius 40 mm at the floor, 65 mm at the rim, 60 mm deep inside, a 6 mm
  floor, 0.35 kg. Places: x = 0.72, y = 0.15, 0 and -0.15, each within 1 cm; which bowl stands where is drawn per
  episode.

## The units (hidden for the scored units; `units.json` gives the practice units')

| Property | Key in `units.json` | Range |
|---|---|---|
| The pot's empty mass | `pot_mass` | 1.4-2.4 kg |
| Broth at the start | `soup_l` | 2.2-3.0 l |
| Discharge coefficient | `cd` | 0.35-0.75 |
| The burner's full power | `burner_kw` | 2.0-3.5 kW |
| The pieces' size scale | `piece_scale` | 0.85-1.15 |
| The pieces' friction | `piece_mu` | 0.4-0.8 |

## Cameras

| Camera | Pose | Size, vertical field of view |
|---|---|---|
| `overhead` | at (0.40, 0, 1.80), looking at (0.55, 0, 0.80) | 640 x 480, 60 deg |
| `front` | at (1.30, -0.10, 1.30), looking at (0.52, -0.05, 0.85) | 640 x 480, 50 deg |
| `wrist_l`, `wrist_r` | on each hand (the Panda model's `eye_in_hand` camera) | 320 x 240, 75 deg |

Every observation carries each camera's pose and intrinsics.

## Credit and end (`episode.py`)

The numbers in the instruction's "Credit" section, as constants: `SOUP_TOL_ML`, `SPILL_MAX_ML`, `FALLEN`, `RING_TOL`,
`POT_TILT`, `EPISODE_STEPS`, `HOLD`, `CLEAR`, `OPEN`.
