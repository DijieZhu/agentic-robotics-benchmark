# Instrument catalogue of the plant

Every type of instrument a round sheet can list, as the plant fitted it: faces and scales as printed, sizes,
conventions, the plant lettering with the tag plates and room plates engraved in it, the grid marker plates, the
emissivity of the surfaces whose temperature is taken, and for each type where its tag plate hangs and which point
of it the record photo's rules are measured to (section 8). The same facts are in `catalog.json` for programs; the
pictures in this folder are the printed faces themselves, to the scale stated with each.
`robot.mission()["catalog"]` carries a short form of this catalogue, with the whole lettering.

Which face, case size, window or lamp row a listed instrument has is on the round sheet (the item's `scale`,
`case_mm`, `window_m`, `n_lamps` and `colors`, `surface`).

## 1. Dial gauges

One round dial, white with black ticks and numerals and a black pointer, in a black case. The printed face is the
same for every case size; the unit is printed on the face.

<!-- BEGIN GENERATED dials -->
| scale | numbered division | smallest division | face as printed |
|---|---|---|---|
| 0 to 6 bar | 1 | 0.2 | `dial_0-6bar.png` |
| 0 to 10 bar | 1 | 0.2 | `dial_0-10bar.png` |
| 0 to 16 bar | 2 | 0.5 | `dial_0-16bar.png` |
| 0 to 25 bar | 5 | 1 | `dial_0-25bar.png` |
| -1 to 3 bar | 1 | 0.2 | `dial_-1-3bar.png` |
| 0 to 120 °C | 20 | 5 | `dial_0-120C.png` |

| case diameter (mm) | dial diameter (mm) |
|---|---|
| 63 | 56.7 |
| 100 | 90 |
| 160 | 144 |

| item | as built |
|---|---|
| scale | 270° clockwise, from the low end at the 7:30 position to the high end at the 4:30 position; the 90 degrees at the bottom (6 o'clock) |
| pointer and face | black pointer; white, black ticks and numerals; accuracy class 1.0 (printed on the dial) |
| bezel | the bezel lip stands 4 % of the case diameter in front of the dial |
| tag plate | below the case, centred (tag plate centre 15 mm + half the plate height below the case's lower edge), parallel to the dial |
| pictures | dial images are 1024 px across the printed dial (the same face for every case size) |
<!-- END GENERATED dials -->

The pointer turns about the dial's centre and points at the value. A dial's scale is identified by its range and unit
on the round sheet.

## 2. Tubular level gauges

A vertical glass tube beside a tank or a level pot, with a white scale plate to its right. The liquid stands in the
tube; its top is read against the scale.

<!-- BEGIN GENERATED level -->
| window (m) | scale plate, width × height (m) | scale as printed |
|---|---|---|
| 0.40 | 0.05 × 0.45 | `level_scale_400mm.png` |
| 0.44 | 0.05 × 0.49 | `level_scale_440mm.png` |

| item | as built |
|---|---|
| scale | 0-100 % on a white plate to the right of the glass tube; 0 % at the bottom |
| ticks | 21 ticks every 5 %; majors every 10 % (0.45 of the plate width), minors 0.28; numerals every 20 % |
| plate | window + 2 x 25 mm (the 0 % and 100 % ticks lie 25 mm inside the plate's ends, at the window's ends) |
| glass tube | 24 mm outside diameter; its axis 28 mm left of the plate's left edge; a white strip behind the tube |
| liquids | amber, green, oil, water |
| tag plate | right of the scale plate, 20 mm beyond its edge, its lower edge level with the window's bottom |
| pictures | 2400 px per m |
<!-- END GENERATED level -->

## 3. Quarter-turn valves

A valve in a pipe, worked by a lever that turns through a quarter of a turn.

<!-- BEGIN GENERATED lever -->
| item | as built |
|---|---|
| lever | 0.18 m long, red |
| reading | angle between lever and pipe in degrees: 0 in line (open), 90 across (closed) |
| lever plane | parallel to the valve's face, in front of the pipe |
| pipe | horizontal or vertical; the round sheet does not state which |
| tag plate | on the valve body below the stem (horizontal pipe: on the side away from the closing lever); vertical pipe: beside the pipe on the closing side, below the swept quadrant; parallel to the lever plane |
| drawing | `valve_lever_positions.png` |
<!-- END GENERATED lever -->

## 4. Rising-stem valves

A gate valve with a handwheel. Its stem rises as the valve opens; a pointer on the stem moves along an indicator
plate fixed beside it.

<!-- BEGIN GENERATED handwheel -->
| item | as built |
|---|---|
| indicator | vertical scale 0-100 % on a white plate beside the rising stem, 0 % (SHUT) at the bottom, 100 % (OPEN) at the top; an orange pointer on the stem slides over the plate's left half |
| plate | 36 × 130 mm; the scale from 0 % to 100 % is 100 mm long |
| ticks | 21 ticks every 5 %, centred on the plate; majors at 0, 25, 50, 75, 100 % (0.62 of the plate width), minors 0.38; numerals 0, 50, 100; legends OPEN (top) and SHUT (bottom) |
| pointer | 18 × 7 mm, 2.7 mm in front of the plate |
| handwheel | 250 mm diameter |
| tag plate | right of the indicator plate, 20 mm beyond its edge, centred on the plate's height |
| plate as printed | `valve_handwheel_indicator.png` (6000 px per m) |
<!-- END GENERATED handwheel -->

## 5. Lamp panels

A local panel with one row of indicator lamps.

<!-- BEGIN GENERATED lamps -->
| lens colour | legend under the lamp | meaning when lit |
|---|---|---|
| green | RUN | running |
| red | TRIP | tripped (fault) |
| amber | ALARM | alarm |
| white | POWER | power on |

| item | as built |
|---|---|
| lamps per panel | 3, 4, 5, 6 |
| lens | 23 mm diameter in a 33 mm bezel; lamps 65 mm apart, centre to centre |
| panel | light grey (RAL 7035); 200 mm high; width: n x pitch + 50 mm |
| order | left to right as seen from the front; the round sheet lists the colours in that order |
| lit and unlit | a lit lens glows in its colour; an unlit lens is a dark tint of it |
| legend strip | below the lens row, one legend word per lamp |
| tag plate | below the panel, centred |
| panel fronts | `lamp_panel_3.png`, `lamp_panel_4.png`, `lamp_panel_5.png`, `lamp_panel_6.png` (4000 px per m (drawings of the front; lenses shown unlit)) |
<!-- END GENERATED lamps -->

The pictures show panel fronts with the lenses unlit; a panel's own colours, in order, are on the round sheet.

## 6. The plant lettering, tag plates, room plates and grid marker plates

Tag plates and room plates are engraved in one lettering, the plant's own. A tag plate carries its instrument's tag
and nothing else: no code, no number besides. An instrument is the listed one only if its plate says so.

<!-- BEGIN GENERATED tags -->
| plate | size | colours (sRGB) | border line | lettering | pictures |
|---|---|---|---|---|---|
| instrument tag plate | 190.8 × 79.2 mm, 2 mm thick | plate (245, 196, 0) (signal yellow), lettering and border line (15, 15, 15) | 1.6 mm wide, its outer edge 4 mm inside the plate's edge | the tag on one line, capitals 25 mm high, drawn with a round pen of 3.59 mm, one character every 20 mm; at most 8 characters | `tag_plate_example.png` (another tag, 4 px per mm), `tag_plate_drawing.png` |
| room plate | 400 × 100 mm | plate (238, 238, 230), lettering and border line (15, 15, 15) | 1.6 mm wide, its outer edge 4 mm inside the plate's edge | the room's number in capitals 45 mm high, beginning 14 mm inside the left edge, then its name in capitals 20 mm high (on two lines when it is long), both centred on the plate's height | `room_plate_drawing.png` |
| grid marker plate | 349.8 × 145.2 mm | white, black marker | - | a coded marker, AprilTag 36h11 (OpenCV: `DICT_APRILTAG_36h11`), 110 mm side, at the plate's left end: its centre lies 102 mm to the left of the plate's centre, as seen facing the plate, at mid-height | `grid_marker_example.png` (another id, 4 px per mm) |

| the plant lettering |  |
|---|---|
| glyphs | 37: the capitals A to Z, the digits 0 to 9 and the hyphen; `lettering.png` shows them (160 px of cap height; each glyph in its 4 x 8 cell (grey)) |
| a glyph | polylines on a cell 4 units wide and 8 units high (a capital's height), origin at the cell's lower left corner, x to the right, y up; drawn with a round pen 1.15 units across: every point within half the pen of a polyline is ink |
| a line of text | one cell every 6.4 units (fixed pitch), whatever the character; a space advances one pitch without ink |
| on a tag plate | 1 unit = 3.125 mm: cells 12.5 mm wide and 25 mm high, one every 20 mm. The row of cells (not the ink) is centred on the plate: with n characters the first cell's left edge lies ((n − 1) × 20 + 12.5) / 2 mm to the left of the plate's centre, and the cells' lower edge 12.5 mm below it |
| as data | `catalog.json`, `lettering`: `grid_units`, `pen_units`, `pitch_units`, `glyphs` (per character its polylines as lists of [x, y] in units), `tag_grammar`, `tag_plate`, `room_plate`; the same table is in `robot.mission()["catalog"]["lettering"]` |

A tag is one to three capital letters, a hyphen, two to five digits (`^[A-Z]{1,3}-[0-9]{2,5}$`), 8 characters at most. In a frame of the robot's inspection camera (focal length 4040.8 px) the capitals of a tag plate are 101.0 / d px high at d metres and its characters 80.8 × cos a / d px apart when the plate is seen a degrees off its normal; the lettering is legible from 14 px of cap height and 8 px of pitch, up to 60° off the normal (robot manual, section 9).

The position, height and facing that the drawing gives for a grid marker plate (`grid_markers[].pos`, `height`, `facing_deg`) are those of the plate's centre, not of the marker's.
<!-- END GENERATED tags -->

- The lettering is given as numbers so that a program can draw any text at any size itself: every glyph of a tag
  plate is the same polylines scaled to its cell, the cells stand at a fixed pitch whatever the character, and the
  row of cells is centred on the plate. `tag_plate_drawing.png` shows the plate with its dimensions.
- Letters and digits never share a place in a tag: the part before the hyphen holds letters only, the part after it
  digits only.
- The border line is what a camera finds of a plate first and best: its four corners, with the plate's size, give
  the plate's position and the direction it faces (robot manual, section 9), and the plate is parallel to its
  instrument's face at the place section 8 gives.
- The survey mark's disc on the floor carries its tag in the same lettering (round procedure, section 4).

## 7. Temperature points

<!-- BEGIN GENERATED emissivity -->
| surface type on the round sheet (`surface`) | emissivity (8-14 um) |
|---|---|
| painted steel | 0.93 |
| powder-coated steel | 0.91 |
| oxidised cast iron | 0.94 |

A temperature point is a heated component with one flat face, the measured surface. The round sheet names the component (`component`); the table below gives every kind there is. The building's rooms and their walls are at about 22 °C.
<!-- END GENERATED emissivity -->

<!-- BEGIN GENERATED thermal_kinds -->
| `component` on the round sheet | the hot face is | its width × height (m) | reference point | centre of the tag plate | on the practice ground |
|---|---|---|---|---|---|
| `<tag> heater terminal box` | the front plate of a heater terminal box on a wall | 0.34 × 0.28 | the centre of that face: 0.004 m behind the recorded position | 0.295 m to the right of, 0.007 m in front of the reference point | yes |
| `<tag> heater` | the front plate of a heater terminal box on a tank | 0.28 × 0.24 | the centre of that face: 0.004 m behind the recorded position | 0.265 m to the right of, 0.007 m in front of the reference point | yes |
| `<switchgear> cubicle <n>` | the door of a switchgear cubicle | 0.50 × 1.40 | the centre of that face: 0.050 m above and 0.124 m behind the recorded position | 0.050 m below, 0.375 m to the right of, 0.127 m in front of the reference point | no |

The round sheet's `units` has one entry for every temperature point: the component's name, as in `component`. The position the round sheet records for a temperature point lies in front of the hot face, at the tag plate's depth for a cubicle door.
<!-- END GENERATED thermal_kinds -->

The round sheet names the surface type of each temperature point (`surface`). How the thermal camera's reading is
corrected with the emissivity is in the robot manual, section 7.1. Other heated equipment can stand next to a
temperature point, also equipment of the same kind with a tag plate of its own: the reading wanted is that of the
component whose tag plate carries the round sheet's tag.

## 8. Position, reference point and tag plate of each type

Three points of an instrument matter to a program: the **position** that the round sheet records for it, the
**reference point** that the record photo's rules are measured to (distance, angle off the face, size: round
procedure, section 3), and the centre of its **tag plate**, which tells that the instrument is the listed one.

<!-- BEGIN GENERATED geometry -->
| instrument | the position the work order and the debrief give is | reference point of the photo rules | centre of the tag plate |
|---|---|---|---|
| dial gauge | the centre of the dial | the same point | 0.086 / 0.105 / 0.135 m below it (63 / 100 / 160 mm case), in the dial's plane |
| level gauge | the glass tube's axis, at the middle of the window | the middle of tube and scale plate taken together, at mid-window: 0.033 m to the right of the position | 0.160 m to the right of it; its lower edge is level with the window's bottom, so its centre is 0.160 m below (0.40 m window) and 0.180 m below (0.44 m window) it |
| quarter-turn valve | the pipe's axis at the valve | the lever's pivot on the valve stem, the same for every lever angle: 0.082 m in front of the position on the hall's horizontal pipes, 0.088 m on its vertical pipe; the distance grows with the pipe's size (in the plant: 5 to 14 cm) | horizontal pipe: 0.126 m below and 0.034 m behind the pivot; vertical pipe: 0.063 m below, 0.201 m to the right of, 0.033 m behind the pivot |
| rising-stem valve | the valve body, at the height of the pipe's axis | the centre of the indicator plate, 0.42 m straight above the position | 0.133 m to the right of it, at the plate's height |
| lamp panel | the panel | the centre of the panel's front, 5 mm below the position | 0.152 m below it |
| temperature point | a point just in front of the hot face | the centre of the hot face; per kind of component in section 7 (a wall box: 4 mm behind the position) | beside the component, to the right; per kind of component in section 7 (a wall box: 0.295 m to the right of the reference point) |

"Right", "left", "in front" and "behind" are as seen by someone who faces the instrument. A tag plate is parallel to its instrument's face, its long side horizontal.
<!-- END GENERATED geometry -->

<!-- BEGIN GENERATED rounding -->
The round sheet records a position to the nearest 0.1 m in x, y and height, and the direction a face looks to the nearest 5°: for an instrument that is still at its recorded place the recorded facing is right to within 5°.
<!-- END GENERATED rounding -->
Position, height and facing are recorded for the place the instrument was recorded at (facility notes, section 3),
also on a sheet of the drawing that is missing. For an instrument that was moved or relocated since, they are that
old place's and not the instrument's, the facing included. In the training hall's calibration sheet both points of
every instrument are given exactly (`position`, `face_pos`).

## 9. Files in this folder

| file | content |
|---|---|
| `catalog.json` | this catalogue as data: every figure of the tables above; the lettering under `lettering` (every glyph as polylines), the plates under `tags`, the legible sizes under `legibility`, the figures of section 8 under `positions` |
| `lettering.png` | every glyph of the plant lettering |
| `tag_plate_example.png`, `tag_plate_drawing.png` | an instrument tag plate as engraved (an example: another tag), and its dimensioned drawing |
| `room_plate_drawing.png` | the dimensioned drawing of a room plate |
| `grid_marker_example.png` | a grid marker plate as printed (an example: another id) |
| `dial_*.png` | the dial faces as printed, one per scale |
| `level_scale_*.png` | the level gauges' scale plates as printed |
| `valve_lever_positions.png` | a drawing of the quarter-turn valve's lever positions |
| `valve_handwheel_indicator.png` | the rising-stem valve's indicator plate as printed |
| `lamp_panel_*.png` | drawings of the lamp panel fronts, by number of lamps |
