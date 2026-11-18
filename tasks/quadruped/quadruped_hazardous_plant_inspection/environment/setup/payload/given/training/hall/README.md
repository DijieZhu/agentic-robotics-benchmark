# The training hall

The team's training hall is practice site `hall`: one large hall behind an airlock like a plant building's, built to
measure the robot and to try a program's parts one at a time. Unlike a plant building, everything about it is known
and written down here, and its drawing is complete and correct: no sheet is missing. A hall mission lasts at most
900 s of simulated time.

| file | content |
|---|---|
| `hall_plan.png` | the hall's general arrangement drawing as a picture (north is +y) |
| `hall_plan.json` | the hall's documents as data: what `robot.mission()` returns in the hall (drawing, link plan, work order, round sheet, modification record, the owner's statements), without the entries the robot adds at run time (`budgets`, `catalog`) and without the items' `weight` |
| `calibration_sheet.json` | everything below as data: the instruments' set values with `position` (as the work order has it) and `face_pos` (the reference point of the record photo's rules), the plates of the plate board and the posts, the check source, the survey mark, the marker plates, the lanes, the floor lines, the link nodes |

All coordinates are in the site frame: origin at the centre of the deployment pad, x into the hall, y to the left, z
up; metres; directions in degrees counter-clockwise from +x.

## 1. The instrument wall

The south wall carries every instrument type of the catalogue, bay by bay, the instruments of a bay at mounting
heights of 0.5, 1.0 and 1.5 m, each with its tag plate where the catalogue's section 8 puts it. The last bay holds
two instruments set up as the plant has them elsewhere: a quarter-turn valve on a vertical pipe, and a lamp panel
mounted high under a canopy. All hold these values in every hall mission:

<!-- BEGIN GENERATED hall_instruments -->
| tag (the text of its plate) | instrument | bay | position x, y, z (m): exact, of the point the work order's rounded position means | reference point of the photo rules x, y, z (m) | faces (deg from +x) | set value |
|---|---|---|---|---|---|---|
| `PI-211` | dial gauge, 63 mm case | 1 | 3.60, -10.254, 0.50 | 3.600, -10.254, 0.500 | 90 | 1.5 bar (scale 0 to 6 bar) |
| `PI-212` | dial gauge, 63 mm case | 1 | 4.60, -10.254, 1.00 | 4.600, -10.254, 1.000 | 90 | 5 bar (scale 0 to 10 bar) |
| `PI-213` | dial gauge, 63 mm case | 1 | 5.60, -10.254, 1.50 | 5.600, -10.254, 1.500 | 90 | 12 bar (scale 0 to 16 bar) |
| `PI-221` | dial gauge, 100 mm case | 2 | 6.80, -10.241, 0.50 | 6.800, -10.241, 0.500 | 90 | 6.25 bar (scale 0 to 25 bar) |
| `PI-222` | dial gauge, 100 mm case | 2 | 7.80, -10.241, 1.00 | 7.800, -10.241, 1.000 | 90 | 1 bar (scale -1 to 3 bar) |
| `TI-223` | dial gauge, 100 mm case | 2 | 8.80, -10.241, 1.50 | 8.800, -10.241, 1.500 | 90 | 84 °C (scale 0 to 120 °C) |
| `PI-231` | dial gauge, 160 mm case | 3 | 10.00, -10.221, 0.50 | 10.000, -10.221, 0.500 | 90 | 4.2 bar (scale 0 to 6 bar) |
| `PI-232` | dial gauge, 160 mm case | 3 | 11.00, -10.221, 1.00 | 11.000, -10.221, 1.000 | 90 | 3.2 bar (scale 0 to 16 bar) |
| `PI-233` | dial gauge, 160 mm case | 3 | 12.00, -10.221, 1.50 | 12.000, -10.221, 1.500 | 90 | 18.75 bar (scale 0 to 25 bar) |
| `LG-241` | level gauge | 4 | 13.20, -9.895, 0.50 | 13.167, -9.895, 0.500 | 90 | 22 % |
| `LG-242` | level gauge | 4 | 14.20, -9.895, 1.00 | 14.167, -9.895, 1.000 | 90 | 55 % |
| `LG-243` | level gauge | 4 | 15.20, -9.895, 1.50 | 15.167, -9.895, 1.500 | 90 | 86 % |
| `HV-251` | quarter-turn valve | 5 | 16.40, -10.230, 0.50 | 16.400, -10.148, 0.500 | 90 | 0° (open) |
| `HV-252` | quarter-turn valve | 5 | 17.40, -10.230, 1.00 | 17.400, -10.148, 1.000 | 90 | 45° (throttled) |
| `HV-253` | quarter-turn valve | 5 | 18.40, -10.230, 1.50 | 18.400, -10.148, 1.500 | 90 | 90° (closed) |
| `V-261` | rising-stem valve | 6 | 19.60, -10.220, 0.50 | 19.600, -10.220, 0.920 | 90 | 12 % open |
| `V-262` | rising-stem valve | 6 | 20.60, -10.220, 1.00 | 20.600, -10.220, 1.420 | 90 | 50 % open |
| `V-263` | rising-stem valve | 6 | 21.60, -10.220, 1.50 | 21.600, -10.220, 1.920 | 90 | 88 % open |
| `LCP-271` | lamp panel | 7 | 22.80, -10.250, 0.50 | 22.800, -10.250, 0.495 | 90 | green on, red off, amber on |
| `LCP-272` | lamp panel | 7 | 23.80, -10.250, 1.00 | 23.800, -10.250, 0.995 | 90 | green off, red on, amber on, white off |
| `LCP-273` | lamp panel | 7 | 24.80, -10.250, 1.50 | 24.800, -10.250, 1.495 | 90 | green on, red on, amber off, white on, green off, red off |
| `TS-281` | temperature point | 8 | 26.00, -10.187, 0.50 | 26.000, -10.191, 0.500 | 90 | 45 °C; painted steel, emissivity 0.93 |
| `TS-282` | temperature point | 8 | 27.00, -10.187, 1.00 | 27.000, -10.191, 1.000 | 90 | 75 °C; powder-coated steel, emissivity 0.91 |
| `TS-283` | temperature point | 8 | 28.00, -10.187, 1.50 | 28.000, -10.191, 1.500 | 90 | 102 °C; oxidised cast iron, emissivity 0.94 |
| `HV-291` | quarter-turn valve: on a vertical pipe (0°: the lever along the pipe) | 9 | 29.40, -10.230, 1.00 | 29.400, -10.142, 1.000 | 90 | 30° (throttled) |
| `LCP-292` | lamp panel: mounted high under a canopy, to be read from below in its shade | 9 | 31.00, -10.250, 1.90 | 31.000, -10.250, 1.895 | 90 | green on, amber off, red on, white on |
<!-- END GENERATED hall_instruments -->

The hall's work order lists all of them, so each can be reported and the debrief tells how the report fared, with
each instrument's position and the reference point of the photo rules. A hall mission has no score: the `weight`
entries of the hall's documents mean nothing.

## 2. The plate board, the posts and the room plates

For a program that reads tag plates the hall has plates that hang alone, with no instrument beside them and no item
in the work order:

<!-- BEGIN GENERATED hall_plates -->
A plate board on the west wall, at the end of the straight line on the floor (section 6), carries sixteen tag plates in four rows from 0.4 to 1.6 m above the floor: every letter of the plant's tag prefixes, every digit and the hyphen at least twice, three pairs of tags one character apart, four plates on dark steel. Walking the straight shows the board head-on from 0.9 to 28.9 m. Beside it stand two posts, each with a plate turned from square to the straight (45° and 60°).

| text of the plate | centre of its front x, y, z (m) | faces (deg from +x) | behind it | where |
|---|---|---|---|---|
| `PI-1047` | 2.137, -6.540, 1.60 | 0 | grey board | plate board, row 1 column 1 |
| `PI-1048` | 2.137, -6.180, 1.60 | 0 | grey board | plate board, row 1 column 2 |
| `TI-2385` | 2.137, -5.820, 1.60 | 0 | grey board | plate board, row 1 column 3 |
| `LG-3906` | 2.143, -5.460, 1.60 | 0 | dark steel | plate board, row 1 column 4 |
| `LG-3956` | 2.137, -6.540, 1.20 | 0 | grey board | plate board, row 2 column 1 |
| `HV-4172` | 2.143, -6.180, 1.20 | 0 | dark steel | plate board, row 2 column 2 |
| `V-5629` | 2.137, -5.820, 1.20 | 0 | grey board | plate board, row 2 column 3 |
| `V-5620` | 2.137, -5.460, 1.20 | 0 | grey board | plate board, row 2 column 4 |
| `LCP-6813` | 2.143, -6.540, 0.80 | 0 | dark steel | plate board, row 3 column 1 |
| `TS-7254` | 2.137, -6.180, 0.80 | 0 | grey board | plate board, row 3 column 2 |
| `HV-8031` | 2.137, -5.820, 0.80 | 0 | grey board | plate board, row 3 column 3 |
| `TI-9640` | 2.137, -5.460, 0.80 | 0 | grey board | plate board, row 3 column 4 |
| `LCP-2597` | 2.137, -6.540, 0.40 | 0 | grey board | plate board, row 4 column 1 |
| `TS-0368` | 2.137, -6.180, 0.40 | 0 | grey board | plate board, row 4 column 2 |
| `PI-7412` | 2.143, -5.820, 0.40 | 0 | dark steel | plate board, row 4 column 3 |
| `LG-1859` | 2.137, -5.460, 0.40 | 0 | grey board | plate board, row 4 column 4 |
| `TS-5530` | 2.875, -7.325, 1.00 | 45 | post | post 1: the plate turned 45° from square to the straight |
| `HV-9061` | 2.868, -4.680, 1.00 | -60 | post | post 2: the plate turned 60° from square to the straight |

Room plates, number and name, as they hang beside the main doors of a building's rooms (facility notes, section 2):

| text of the plate | centre x, y, z (m) | faces (deg from +x) | hangs in |
|---|---|---|---|
| `100 AIRLOCK` | 2.139, -0.880, 1.60 | 0 | the training hall (room 200) |
| `200 TRAINING HALL` | 1.861, -0.880, 1.60 | 180 | the airlock (room 100) |
<!-- END GENERATED hall_plates -->

The board and the posts are the place to find out at which size and angle your reader reads a tag right, where it
starts to read a wrong one, and how well a plate's position and facing come out of its border line: the table gives
the truth for every plate, and the hall's debrief gives the robot's true pose (section 6). The instrument wall adds
plates at three heights beside every type of instrument, and the canopy's lamp panel one at 1.9 m in the shade.

## 3. The check source and the survey mark

<!-- BEGIN GENERATED hall_source -->
| what | x, y, z (m) | value |
|---|---|---|
| check source | 31.20, -0.60, 0.30 | 25 Gy/h at 1 m; inverse square in the open; the shield blocks attenuate behind them |
| survey mark `SM-01` | 27.20, -0.60 (on the floor) | 1.569 Gy/h at the mark's centre, 0.30 m above the floor |
| background | everywhere | 0.0067 Gy/h |
<!-- END GENERATED hall_source -->

The check source stands in a cave of shield blocks that is open towards the survey mark. In the hall's work order it
is the item of the "unlisted" source: here its position is known, and the debrief of a `report_source` shows how a
program's estimate compares with it.

## 4. The lanes

Two rows of dead-end lanes, each about 1.1 m wide between dividing walls, each with one obstacle across it. The
south row is entered from its mouth towards -y, the north row towards +y; a robot that has crossed the obstacle
stands in the lane's end and comes back the same way.

<!-- BEGIN GENERATED hall_lanes -->
| lane | what it holds | size | rating | lane between x (m) | mouth at y (m) | obstacle at y (m) | walk in |
|---|---|---|---|---|---|---|---|
| `L01` | door sill | 0.03 m | free | 5.03 to 6.17 | 5.40 | 3.80 | towards -y |
| `L02` | door sill | 0.045 m | unrated | 6.23 to 7.37 | 5.40 | 3.80 | towards -y |
| `L03` | door sill | 0.052 m | needs_high_step | 7.43 to 8.57 | 5.40 | 3.80 | towards -y |
| `L04` | door sill | 0.057 m | needs_high_step | 8.63 to 9.77 | 5.40 | 3.80 | towards -y |
| `L05` | door sill | 0.07 m | unrated | 9.83 to 10.97 | 5.40 | 3.80 | towards -y |
| `L06` | hose on the floor | 0.03 m | free | 11.03 to 12.17 | 5.40 | 3.80 | towards -y |
| `L07` | hose on the floor | 0.045 m | unrated | 12.23 to 13.37 | 5.40 | 3.80 | towards -y |
| `L08` | hose on the floor | 0.052 m | needs_high_step | 13.43 to 14.57 | 5.40 | 3.80 | towards -y |
| `L09` | hose on the floor | 0.062 m | unrated | 14.63 to 15.77 | 5.40 | 3.80 | towards -y |
| `L10` | cable trench cover, a steel plate strip 0.2 m deep with square edges: its height | 0.052 m | needs_high_step | 15.83 to 16.97 | 5.40 | 3.80 | towards -y |
| `L11` | pipe on the floor, square to the lane | 0.03 m | free | 17.03 to 18.17 | 5.40 | 3.80 | towards -y |
| `L12` | pipe on the floor, square to the lane | 0.045 m | unrated | 18.23 to 19.37 | 5.40 | 3.80 | towards -y |
| `L13` | pipe on the floor, square to the lane | 0.052 m | needs_high_step | 19.43 to 20.57 | 5.40 | 3.80 | towards -y |
| `L14` | pipe on the floor, square to the lane | 0.06 m | needs_high_step | 20.63 to 21.77 | 5.40 | 3.80 | towards -y |
| `L15` | pipe on the floor, square to the lane | 0.07 m | unrated | 21.83 to 22.97 | 5.40 | 3.80 | towards -y |
| `L16` | pipe on the floor, turned 20° from square | 0.052 m | needs_high_step | 23.03 to 24.17 | 5.40 | 3.80 | towards -y |
| `L17` | pipe on the floor, turned 20° from square | 0.06 m | needs_high_step | 24.23 to 25.37 | 5.40 | 3.80 | towards -y |
| `L18` | kerb | 0.04 m | unrated | 25.43 to 26.57 | 5.40 | 3.80 | towards -y |
| `L19` | kerb | 0.052 m | needs_high_step | 26.63 to 27.77 | 5.40 | 3.80 | towards -y |
| `L20` | kerb | 0.07 m | unrated | 27.83 to 28.97 | 5.40 | 3.80 | towards -y |
| `L21` | kerb | 0.11 m | barrier | 29.03 to 30.17 | 5.40 | 3.80 | towards -y |
| `L22` | nothing (level floor) | - | free | 30.23 to 31.37 | 5.40 | 3.80 | towards -y |
| `L23` | ramped cable protector | 0.03 m | free | 5.03 to 6.17 | 6.80 | 8.40 | towards +y |
| `L24` | ramped cable protector | 0.05 m | unrated | 6.23 to 7.37 | 6.80 | 8.40 | towards +y |
| `L25` | ramped cable protector | 0.07 m | unrated | 7.43 to 8.57 | 6.80 | 8.40 | towards +y |
| `L26` | ramped cable protector | 0.09 m | unrated | 8.63 to 9.77 | 6.80 | 8.40 | towards +y |
| `L27` | low rack: clearance under it | 0.42 m | free | 9.83 to 10.97 | 6.80 | 8.40 | towards +y |
| `L28` | low rack: clearance under it | 0.34 m | unrated | 11.03 to 12.17 | 6.80 | 8.40 | towards +y |
| `L29` | low rack: clearance under it | 0.31 m | unrated | 12.23 to 13.37 | 6.80 | 8.40 | towards +y |
| `L30` | low rack: clearance under it | 0.29 m | needs_crouch | 13.43 to 14.57 | 6.80 | 8.40 | towards +y |
| `L31` | low rack: clearance under it | 0.26 m | unrated | 14.63 to 15.77 | 6.80 | 8.40 | towards +y |
| `L32` | pipe bridge 0.8 m long (pipes in layers on cross beams, as in the plant's rooms): clearance under it | 0.29 m | needs_crouch | 15.83 to 16.97 | 6.80 | 8.40 | towards +y |
| `L33` | ramp up to a platform | 8° | free | 17.03 to 18.17 | 6.80 | 8.40 | towards +y |
| `L34` | ramp up to a platform | 12° | slow | 18.23 to 19.37 | 6.80 | 8.40 | towards +y |
| `L35` | ramp up to a platform | 15° | slow | 19.43 to 20.57 | 6.80 | 8.40 | towards +y |
| `L36` | ramp up to a platform | 18° | unrated | 20.63 to 21.77 | 6.80 | 8.40 | towards +y |
| `L37` | gap between two walls: clear width | 0.65 m | unrated | 21.83 to 22.97 | 6.80 | 8.40 | towards +y |
| `L38` | gap between two walls: clear width | 0.55 m | tight | 23.03 to 24.17 | 6.80 | 8.40 | towards +y |
| `L39` | gap between two walls: clear width | 0.48 m | tight | 24.23 to 25.37 | 6.80 | 8.40 | towards +y |
| `L40` | gap between two walls: clear width | 0.42 m | unrated | 25.43 to 26.57 | 6.80 | 8.40 | towards +y |
| `L41` | wet patch: its friction | friction 0.3 | slow | 26.63 to 27.77 | 6.80 | 8.40 | towards +y |
| `L42` | wet patch: its friction | friction 0.25 | slow | 27.83 to 28.97 | 6.80 | 8.40 | towards +y |
| `L43` | wet patch: its friction | friction 0.2 | slow | 29.03 to 30.17 | 6.80 | 8.40 | towards +y |
| `L44` | wet patch: its friction | friction 0.15 | slow | 30.23 to 31.37 | 6.80 | 8.40 | towards +y |

Lanes `L10` and `L32` hold the two obstacles that a building can have inside a room, away from any door (facility notes, section 3): a cable trench cover, which `high_step` crosses, and a pipe bridge, which `crouch_walk` passes under. The lanes are also where to measure what a `blocked` costs (robot manual, section 5): walk into lane `L03` in `walk`, for instance, and time the `walk` replies until a command is taken again.
<!-- END GENERATED hall_lanes -->

The ratings are the plant's own classes for what stands on a floor, set against this robot's manual:

| rating | meaning |
|---|---|
| free | within what plain `walk` is rated for |
| needs_high_step | above what `walk` takes, within what `high_step` is rated for |
| needs_crouch | too low for `walk` and `high_step`, within what `crouch_walk` is rated for |
| slow | passable at reduced speed (ramps), or slippery (wet patches) |
| tight | passable straight through, no room to turn |
| barrier | beyond every mode |
| unrated | between or beyond the rated sizes: the lanes to find out what the robot does there |

## 5. Marker plates

The hall's grid marker plates, all in place:

<!-- BEGIN GENERATED hall_markers -->
| plate | marker id | x, y (m) | centre above the floor (m) | faces (deg from +x) | on |
|---|---|---|---|---|---|
| `GM-01` | 300 | 6.200, -10.371 | 0.25 | 90 | wall |
| `GM-02` | 301 | 9.400, -10.371 | 0.25 | 90 | wall |
| `GM-03` | 302 | 12.600, -10.371 | 0.25 | 90 | wall |
| `GM-04` | 303 | 15.800, -10.371 | 0.25 | 90 | wall |
| `GM-05` | 304 | 19.000, -10.371 | 0.25 | 90 | wall |
| `GM-06` | 305 | 22.200, -10.371 | 0.25 | 90 | wall |
| `GM-07` | 306 | 25.400, -10.371 | 0.25 | 90 | wall |
| `GM-08` | 307 | 2.129, -8.000 | 0.60 | 0 | wall |
| `GM-09` | 308 | 2.129, -4.000 | 0.60 | 0 | wall |
| `GM-10` | 309 | 2.129, 3.000 | 0.60 | 0 | wall |
| `GM-11` | 310 | 31.871, -8.000 | 0.60 | 180 | wall |
| `GM-12` | 311 | 31.871, -3.500 | 0.60 | 180 | wall |
| `GM-13` | 312 | 14.000, -0.996 | 0.60 | 90 | island |
| `GM-14` | 313 | 14.000, -3.504 | 0.60 | -90 | island |
| `GM-15` | 314 | 9.996, -2.250 | 0.60 | 180 | island |
<!-- END GENERATED hall_markers -->

## 6. Floor lines and the link

<!-- BEGIN GENERATED hall_lines -->
| what | where |
|---|---|
| straight line on the floor, a tick every metre | from (3.0, -6.0) to (31.0, -6.0): 28 m |
| closed line on the floor around the pallet island | 1 m outside the island x 10.0 to 18.0, y -3.5 to -1.0: 29 m round |
| base station aerial | (0.62, -1.07), 1.9 m above the floor; reaches 12 m; it covers the airlock, the pad included, and through the open inner door a strip of the hall in front of the airlock |
| repeater `RP1` | (31.7, -5.0), 2.0 m above the floor; in service; reaches 9 m |
<!-- END GENERATED hall_lines -->

The hall's debrief contains the robot's true pose (t, x, y, yaw) twice a second, on top of what every practice debrief
has: walking the straight or the loop and comparing that with the odometry your program read gives the odometry's
error for this robot.
