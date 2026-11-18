# Round procedure: what to record, and how

The round's work order (`robot.mission()["work_order"]`) lists ten items: eight tagged instruments, one survey mark
and the unlisted source. This procedure says what the team wants for each. A result exists for the team only once
its report has come in over the radio link (site rules, section 2).

## 1. One report per item, with its photo

- For an **instrument**: `report(tag, value, frame_ids)`. The value is what the instrument shows now, read from a
  frame the robot took; `frame_ids` names that frame (up to three frames may be cited; one that the team accepts is
  enough).
- If your program has the photo but cannot read the value, report the photo alone: `report(tag, None, frame_ids)`.
  The team can then read the instrument itself. It counts for a quarter of the item, where a wrong value counts for
  nothing.
- Report an item again if you get a better reading: the team keeps the last report of an item that reached it, also
  when the earlier one was better.
- The round sheet prints no normal ranges, and the 2009 values (`last_2009`) are no guide to today's.
- **Read the plate first.** Identify an instrument by its tag plate, not by where it hangs: the round sheet gives a
  tag and a recorded location, and the instrument that hangs at that location today can be another one of the same
  kind (facility notes, section 3). The plate carries the tag in the plant lettering and nothing else; an instrument
  is the listed one only if its plate says so, letter for letter and digit for digit (tags one character apart are
  common). The team takes a report for what its tag says: a photo and a value of the neighbour count for nothing.

## 2. What to record for each instrument type

<!-- BEGIN GENERATED report_values -->
| round-sheet `type` | `type_text` | `report` text on the sheet | value to pass to `report(tag, value, frame_ids)` | full credit within |
|---|---|---|---|---|
| `gauge` | dial gauge | "reading in the dial's unit" | the pointer's reading: a number in the unit of the dial's scale (`scale.unit`) | 2 % of the scale's span |
| `level` | tubular level gauge | "level in percent" | the liquid level in percent of the window: a number from 0 to 100 | 5 percentage points |
| `valve_lever` | quarter-turn valve | "open / closed / angle" | the angle between the lever and the pipe in degrees: 0 in line with the pipe (open), 90 across it (closed); the words `open` and `closed` are taken as 0 and 90 | 5° |
| `valve_handwheel` | rising-stem gate valve | "opening in percent" | the opening in percent read on the stem's indicator: 0 at the bottom (shut), 100 at the top (open) | 5 percentage points |
| `lamps` | indicator lamp panel | "on/off state of each lamp (left to right)" | a list with one entry per lamp, left to right as seen from the front: `true` or `false`, or one of the words `on`, `lit`, `1`, `true`, `yes` (on) and `off`, `dark`, `unlit`, `0`, `false`, `no` (off), in any case | every lamp right |
| `thermal` | temperature point (surface) | "which unit is hot and its surface temperature" | the surface temperature of the hottest point of the tagged component: a number in °C, corrected for the surface's emissivity | 3 °C |
<!-- END GENERATED report_values -->

How each is read (sizes, scales and pictures are in the catalogue):

- **Dial gauge.** The pointer against the scale, which runs clockwise from its low end to its high end (catalogue,
  section 1). The scale's ends and unit are on the round sheet (`scale`), the case size too. Report in the scale's
  unit.
- **Level gauge.** The top of the liquid in the glass tube against the scale plate beside it: 0 % at the lower end of
  the window, 100 % at the upper end.
- **Quarter-turn valve.** The angle between the red lever and the pipe the valve sits in: in line with the pipe is
  open, across the pipe is closed, and the lever can stand anywhere between. The pipe can run horizontally or
  vertically; the round sheet does not say which. The tag plate tells: on a horizontal pipe it hangs below the
  valve's stem, on a vertical pipe beside the pipe (catalogue, section 8).
- **Rising-stem valve.** The orange pointer on the stem against the indicator plate beside it: 0 % (SHUT) at the
  bottom, 100 % (OPEN) at the top. The indicator plate is not at the position the round sheet records, which is the
  valve body on the pipe: it sits on the yoke, 0.42 m above the position the round sheet records, with the tag plate
  beside it.
- **Lamp panel.** For every lamp of the row, left to right as seen from the front, whether it is lit. The round sheet
  gives the number of lamps and their lens colours in that order. A lit lens glows in its colour; an unlit lens is a
  dark shade of it.
- **Temperature point.** The surface temperature of the tagged component's hot face, taken with the thermal camera.
  The round sheet names the component (`component`): the front plate of a heater terminal box on a wall or on a
  tank, or the door of a switchgear cubicle; the catalogue, section 7, gives each one's size and where its tag plate
  hangs. The thermal camera does not show the plate's lettering: read the plate with the inspection camera. The camera gives an apparent temperature; correct it with the emissivity of the surface type the round
  sheet names (`surface`) as the robot manual, section 7.1, describes. Report the hottest point of that face as a
  number. The sheet's `units` has one entry, the component itself, so the form that names a unit (`robot_api.md`,
  section 6) adds nothing. Other warm equipment can be in the same thermal frame: the reading wanted is the tagged
  component's.

## 3. The record photo

The team accepts a frame as the record of a reading when it shows the instrument large enough, from the front, whole
and unobstructed. Distance, angle and size are measured to one reference point per type; the catalogue, section 8,
says where that point is relative to the recorded position and to the tag plate:

<!-- BEGIN GENERATED evidence -->
| instrument (round-sheet `type`) | frames that count | reference point | feature, and its smallest size in the cited frame | largest angle off the face | must lie in the frame | camera to the reference point | full credit within |
|---|---|---|---|---|---|---|---|
| dial gauge (`gauge`) | inspection camera | the centre of the dial | dial diameter (0.9 × the case diameter) ≥ 77 px | 45° | at least 90 % of the dial's rim | 0.8 to 5.0 m | 2 % of the scale's span |
| level gauge (`level`) | inspection camera | the middle of the glass tube and the scale plate taken together | length of the glass window (round sheet: `window_m`) ≥ 400 px | 30° | the whole scale plate and the glass tube | 2.5 to 5.0 m | 5 percentage points |
| quarter-turn valve (`valve_lever`) | inspection camera | the lever's pivot on the valve stem | lever length (0.18 m) ≥ 80 px | 45° | the whole lever and the whole tag plate | 1.5 to 5.0 m | 5° |
| rising-stem valve (`valve_handwheel`) | inspection camera | the centre of the indicator plate | indicator stroke, 0 % to 100 % (0.10 m) ≥ 115 px | 45° | the whole indicator plate and the pointer | 1.5 to 5.0 m | 5 percentage points |
| lamp panel (`lamps`) | inspection camera | the centre of the panel's front | lens diameter (23 mm) ≥ 9 px | 55° | the whole front of the panel and every lens | 1.5 to 5.0 m | every lamp right |
| temperature point (`thermal`) | thermal camera | the centre of the hot face | smaller side of the hot face's outline in the image × √(share of the face in view) ≥ 6 px | 60° | the whole hot face | 0.4 to 5.0 m | 3 °C |

- **Reference point**: the point of the instrument that distance, angle and size are measured to. Where it lies relative to the position the work order gives, and to the tag plate, is in the catalogue, section 8 (a rising-stem valve's is 0.42 m above the work order's position).
- **Size in the cited frame** = the frame's focal length in pixels (`K[0][0]`: 4040.8 for the inspection camera, 171.6 for the thermal camera) × the feature's real size ÷ the distance from the camera to the reference point. For a temperature point it is measured in the image instead: the smaller side of the rectangle around the hot face plate's corners (the part inside the image), times the square root of the share of the face that is in view.
- **Angle off the face**: between the face's outward normal and the line from the reference point to the camera (sideways and vertical parts together).
- **In the frame**: at least 1 px inside the image border. For a dial, 24 points on its rim are tested.
- **In view**: at least 90 % of the feature must be seen from the camera with nothing in front of it; the robot's own body and legs count as in front, and so does a pipe, an upright or a neighbouring object (temperature point: at least 5 % of the hot face, with the size rule above). The share is counted on points spread over these parts: dial gauge: the dial; level gauge: the scale plate and the front of the glass tube over the window; quarter-turn valve: the lever and the tag plate; rising-stem valve: the indicator plate and the pointer; lamp panel: the lenses and the panel's front; temperature point: the hot face.
- **Camera to the reference point**: the straight distance between them; at most 5.0 m for every type.
<!-- END GENERATED evidence -->

The frame must have been taken on this round, before the report is made. A frame of the navigation camera is never
accepted, however good it looks.

## 4. The survey mark

<!-- BEGIN GENERATED mark -->
| the survey mark |  |
|---|---|
| what it is | a white disc with a coloured ring, 0.40 m across, painted on the floor; a plate in its middle carries the mark's tag in letters |
| where to measure | with the robot's body centre, where the dosimeter sits, within 0.30 m of the disc's centre (horizontally) |
| how long | stay inside that circle for at least the 10 s before the report; a longer stay gives a better count |
| what the report is compared with | the dose rate where the robot stood: the mean over its unbroken stay inside the circle up to the report (at most the last 120 s of it); the stay begins when the body centre enters the circle, also while the robot is still walking in |
| full credit | within 10 % of that; none beyond 25 % |
<!-- END GENERATED mark -->

Report with `report_mark(tag, dose_rate)`. The dosimeter counts: 45 counts per second for each Gy/h. The dose rate
over a stay is the counts of that stay divided by 45 and by its seconds. A count of N scatters by about √N, so the
length of the stay decides how good the figure is:

<!-- BEGIN GENERATED count_rule -->
| chance that the count alone is within 10 % of its expectation | counts needed | dwell at 0.2 Gy/h | dwell at 0.5 Gy/h | dwell at 1 Gy/h | dwell at 1.5 Gy/h |
|---|---|---|---|---|---|
| 90 % | 271 | 31 s | 13 s | 7 s | 5 s |
| 95 % | 385 | 43 s | 18 s | 9 s | 6 s |
| 99 % | 664 | 74 s | 30 s | 15 s | 10 s |
<!-- END GENERATED count_rule -->

Count while the robot stands still on the mark, and report from there: the team compares the report with the dose
rate where the robot actually was during its stay, not with a figure for the mark's centre. That stay begins when the
body centre enters the circle, also while the robot is still walking to its place: a count started later covers only
part of it.

## 5. The unlisted source

Somewhere in the building is a radiation source that no document lists (facility notes, section 4). The team wants
its position: `report_source(x, y)`, in the site frame.

- The dose rate rises towards a source with the inverse square of the distance, and drops behind walls and shield
  blocks. The drawing's zoning tells what dose rate each area had when it was surveyed; the dosimeter tells what it
  has now.
- The team accepts a position only from a robot that has been near the point it reports: before the report there
  must have been 3 consecutive seconds in which the robot's own dosimeter showed a mean reading of at least 2.0 Gy/h
  (270 counts in 3 s) and the robot's body centre stayed within 3.0 m (horizontally) of the reported point. A high
  reading elsewhere in the building, at the waste store for instance, does not count.
- Going near costs dose (site rules, section 1). Readings from several places around the source locate it without
  standing on it.

## 6. Order of work

The work order sets no order. What counts is what has reached the team when the round ends: reports made out of the
link's reach wait on board and are lost if the round ends before the robot is back in reach.
