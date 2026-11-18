# Conventions

This page fixes the names, units and definitions used by the three deliverables (`requirements.yaml`, `bom.json`,
`cable_net.json`) and by the files in `schemas/`. It says what each quantity means and how it is written. It states no
measured value, no count and no part of the robot: those are yours to recover from the evidence.

## Names

**Struts and ends.** The robot has three struts. A strut is called by the colour of the covers on its end caps:
`red`, `green` or `blue`. The struts are alike except for that colour and for the index given to their controllers. Each
strut has two ends:

- end `U` is the end nearest the USB connector of the control board that the strut carries;
- end `D` is the other end.

A **node** is one end of one strut, written `{"strut": "<colour>", "end": "<U|D>"}`.

**Parts of a strut.** The definitions on this page use these words:

| word | meaning |
|---|---|
| rod | the straight round bar that runs through the strut; its centre line is the **rod axis** |
| end cap | the rigid made part fixed on an end of the rod; it carries the hook posts |
| hook post | a short round post on an end cap, on which cables and strain sensors are hooked |
| cover | the soft coloured part fitted over an end cap; its outer surface is part of a sphere |
| motor module | a housing clamped on the rod that holds a gearmotor and the winch on its shaft |
| winch | the spool on a gearmotor shaft that winds a cable |
| electronics enclosure | the housing clamped on the rod that holds the main board |
| main board | the circuit board described by `board/fab_data.json` |
| control board | the bought microcontroller board mounted on the main board |
| breakout board | any other bought circuit board mounted on the main board |
| motor driver | a breakout board that drives gearmotors |
| controller | a control board running the firmware `software/controller/onboard.ino` |
| host software | `software/host/run_tensegrity.py` |
| actuated cable | a cable wound by a winch |
| strain sensor | an elastic capacitive sensor with one of the outlines in `sensors/`: a **short** sensor has the outline of `short_sensor_cut.dxf`, a **long** sensor that of `long_sensor_cut.dxf` |

**Indices.** `motor_index`, `sensor_index` and `controller_index` are the indices that the host software itself uses.

**Designators.** Connectors and pads of the main board are written with the designators of `board/fab_data.json`: a
placement as `X<nn>`, a pad as `X<nn>.<pad>`. The main boards of the struts are alike, so a lead of one kind has the
same designator on every strut.

**Source ids.** `source_ids` (and `inputs` of a derived entry) name the evidence an entry rests on:

| id | source |
|---|---|
| `PH-<nnn>` | `photos/photo_<nnn>.jpg` |
| `FW` | `software/controller/onboard.ino` |
| `HOST` | `software/host/run_tensegrity.py` |
| `CFG` | `software/host/tracking_config.json` |
| `CAL` | the files in `software/host/calibration/` |
| `FAB` | `board/fab_data.json` |
| `DXF` | the cut files in `sensors/` |
| `VD-<nn>` | `vendor/VD-<nn>_*.md` |

An entry cites at most 3 sources: the ones that carry the fact.

## Units

- `requirements.yaml`: every entry carries its `unit`. `schemas/units.json` lists the accepted units by dimension with
  their factors; any unit of the quantity's dimension is accepted. Enumerations take the unit `none`.
- `cable_net.json`: lengths under `geometry` are plain numbers in metres. The schema refuses a length above
  1 m, as a guard against millimetres written by mistake.
- `bom.json`: an attribute is in the unit its key ends with (`_mm`, `_v`, `_a`, `_ohm`, `_mah`).

## Which source to trust

When sources disagree:

- **What the hardware is:** the photographed object comes first, then the fabrication data of the main board, then the
  vendor's data of a part you have identified, then constants about the hardware that are written in the software.
- **What the software does:** the code decides, not its comments.
- A source that does not mention a thing does not contradict a source that does.

## `cable_net.json`

| field | meaning |
|---|---|
| `controllers[]` | one record per controller: `controller_index`, and `strut` = the strut that carries that controller |
| `cables[]` | one record per actuated cable: `motor_index` = the motor that winds it; `winch` = the node at which that motor and its winch sit; `hook` = the node at which the far end of the cable is hooked; `motor_connector` = the placement on the main board into which the lead of that motor plugs |
| `sensors[]` | one record per strain sensor: `sensor_index`; `between` = the two nodes it is hooked to, in either order; `connector_pad` = the pad, on the main board that reads the sensor, through which its signal (not its return) reaches the capacitance inputs |
| `spare_inputs[]` | at most one record per strut: `strut`, and `connector_pad` = a signal pad of a sensor connector on that strut's main board that carries no strain sensor in the assembled robot |
| `geometry` | the key dimensions below |
| `counts` | the counts below |

A record or a field that you leave out earns nothing and is not an error. Two records with the same index (for
`spare_inputs`: the same strut) are an error that fails the whole submission.

**Key dimensions** (`geometry`). All are lengths of the assembled strut. Where a dimension can be taken at both ends of
the strut or on several like parts, give the mean.

| name | definition |
|---|---|
| `strut_overall_length` | Length of the assembled strut along the rod axis, covers on: the distance between the two planes perpendicular to the rod axis that touch the outermost point of the cover at each end. |
| `node_spacing` | Distance between the centres of the two spheres to which the outer surfaces of the covers at the two ends of the strut belong. |
| `motor_module_outer_face_station` | Distance along the rod axis from the mid-length point of the rod to the outer end face of a motor module. The outer end face is the flat face of the module's housing, perpendicular to the rod axis, against which the neighbouring end cap sits. |
| `winch_axis_offset` | Perpendicular distance between the rod axis and the rotation axis of a winch (the axis of the gearmotor shaft that carries it). |
| `envelope_diameter` | Diameter of the smallest cylinder coaxial with the rod that contains the whole assembled strut, covers and everything mounted on the strut included: twice the largest distance of any point of the strut from the rod axis. |
| `cover_sphere_radius` | Radius of the sphere to which the outer surface of a cover belongs. |
| `rod_insertion_in_end_cap` | Length of rod inside an end cap, along the rod axis: from the end face of the rod to the face of the end cap at which the rod enters it (the face of the end cap nearest the middle of the strut). |
| `winch_flange_diameter` | Largest outside diameter of a winch about its rotation axis (the diameter over its flanges). |
| `hook_circle_radius` | Radius of the circle, centred on the rod axis, that passes through the axes of the hook posts of an end cap: the distance from the rod axis to the axis of a hook post. |
| `electronics_bay_length` | Length of the closed electronics enclosure along the rod axis, between its two outermost end faces. |

**Counts** (`counts`).

| name | definition |
|---|---|
| `hooks_per_end_cap` | Number of hook posts on one end cap. |

## `requirements.yaml`

`items[]` holds one entry per requirement that the evidence fixes.

| field | meaning |
|---|---|
| `id` | `REQ-<label>`, unique |
| `quantity` | a name of `schemas/quantities.json`, which gives every name its unit, its domain and its definition |
| `comparator` | `"="` for a fact of the robot as built (what it is, what it has, what it is set to); `"<="` or `">="` for a limit, in the direction in which the limit holds. `"=="` is read as `"="`, `"<"` as `"<="`, `">"` as `">="` |
| `bound` | a number; for an enumeration, one of its tokens |
| `unit` | the unit of `bound` |
| `source_ids` | the sources the entry rests on |
| `note` | free text, optional |

A quantity may appear twice with different comparators (a lower and an upper limit). The same quantity twice with the
same comparator is an error that fails the whole submission.

Write comparators in quotes: unquoted, YAML reads `=` and `>` as syntax. Write numbers in plain decimal notation,
without an exponent.

`derived[]` holds the derived quantities: `id` (`DER-<label>`), `quantity`, `value`, `unit`, and optionally `inputs`
(ids of your entries and source ids) and `derivation` (free text). The same derived quantity twice is an error that
fails the whole submission.

**Derived quantities.** The formulas are fixed; their inputs are yours to find.

| name | unit | formula | meaning |
|---|---|---|---|
| `short_sensor_strain_at_min_command` | `none` | `(L_min - p_short) / p_short` | Strain of a short strain sensor when its cable is at the shortest commanded length. |
| `short_sensor_strain_at_max_command` | `none` | `(L_max - p_short) / p_short` | Strain of a short strain sensor when its cable is at the longest commanded length. |
| `short_sensor_sheets_for_order` | `count` | `ceil(n_short * R / s_short)` | Number of short-sensor cut sheets the order needs. |
| `long_sensor_sheets_for_order` | `count` | `ceil(n_long * R / s_long)` | Number of long-sensor cut sheets the order needs. |
| `winch_turns_over_command_range` | `none` | `(L_max - L_min) / (pi * d)` | Turns of a winch between the shortest and the longest commanded cable length. |
| `encoder_counts_over_command_range` | `count` | `(L_max - L_min) / (pi * d) * G * N` | Encoder counts of one motor between the shortest and the longest commanded cable length. |
| `pwm_8bit_at_speed_cap` | `count` | `floor(D * c / S)` | Drive value the controller firmware passes to the motor driver when the host commands its largest speed. |
| `mean_short_sensor_sensitivity` | `pF/mm` | `mean of m_i over the short strain sensors` | Mean calibration slope of the short strain sensors (the slopes of the calibration files are in pF per mm). |

| symbol | meaning |
|---|---|
| `L_min` | shortest cable length the host software commands |
| `L_max` | longest cable length the host software commands |
| `p_short` | distance between the centres of the attachment holes at the two ends of one short strain sensor, from its cut file |
| `n_short, n_long` | number of short and of long strain sensors in one robot |
| `s_short, s_long` | number of sensors in one cut sheet of each kind (one cut file is one sheet) |
| `R` | number of robots ordered (`robots` of `bom.json`) |
| `d` | winch drum diameter the host software uses to turn winch turns into cable length |
| `N` | encoder counts per revolution of the motor shaft, the constant of the software as it stands |
| `G` | the quantity `motor_gearbox_ratio` of the vocabulary |
| `c` | largest speed command the host software sends |
| `S, D` | full-scale speed command and full-scale drive value in the expression with which the controller firmware turns a speed command into a drive value |
| `m_i` | slope of the calibration line of sensor `i` in the calibration file that the host software loads |

Inputs that come from the host software are the values its constructor sets; a value that a key press changes while the
script runs does not count. `ceil` rounds up and `floor` rounds down to a whole number.

**How values are compared.** Linear between full and no credit.

| kind of quantity | full credit | no credit |
|---|---|---|
| a quantity whose domain is integer or enumeration | equal | not equal |
| a gearbox ratio | absolute error <= 0.05 | absolute error >= 0.30 |
| a derived number of encoder counts | relative error <= 0.1 % | relative error >= 0.3 % |
| any other number | relative error <= 1 % | relative error >= 5 % |

The tolerances of the key dimensions are given with the task, by size class.

## `bom.json`

**Scope.** List what is mounted on the robot: bought parts and parts made in the lab. What comes as part of a bought
item (the components on a bought board, the gearbox of a gearmotor) belongs to that item's line. Out of scope, and
ignored if listed, are
the categories `wire`, `tape`, `heat_shrink`, `solder`, `print_material`, `casting_material`, `charger`, `tool`, `consumable` (described in `schemas/categories.json`).

| field | meaning |
|---|---|
| `robots` | the size of the order |
| `line_id` | `L<label>`, unique |
| `kind` | `purchased` or `made` |
| `category` | a category of `schemas/categories.json` |
| `vendor`, `part_number` | maker (or seller) and its part number, for a bought part |
| `description` | free text |
| `attributes` | the attributes that `schemas/categories.json` lists for the category |
| `tags` | for a made part: the paper tags, as written in the photos, of every instance of this part on the photographed strut |
| `qty_per_strut` | pieces on one strut |
| `qty_per_robot_extra` | pieces that one robot needs beyond those on its struts |
| `source_ids` | the sources the line rests on |

**Quantities.** The order is counted as `robots` x (3 x `qty_per_strut` + `qty_per_robot_extra`).

**One part, one line.** Parts that are the same part (the same bought item, or made parts of the same shape) belong on
one line. Lines with the same category and the same attributes are merged and their quantities added, so splitting a
part over several lines does no harm.

**Categories and attributes.** A line is recognised by its category and by what the table names for that category. An
attribute you leave out cannot be recognised; attributes beyond those listed are ignored. `schemas/categories.json` has
the tokens of every token attribute. Part numbers and other strings are compared without regard to case, spaces and
hyphens.

| category | what it is | recognised by |
|---|---|---|
| `battery_cell` | A rechargeable cell, or several cells sold as one item, with its lead and plug. | `cells_in_series` (integer), `connector_positions` (integer), `capacity_mah` (number) |
| `bearing` | A rolling or plain bearing. | `bore_mm` (number), `outer_diameter_mm` (number), `width_mm` (number) |
| `cable_assembly` | A bought, ready-made cable with a connector fitted on one end or on both. | `connector_series` (string), `positions` (integer), `terminated_ends` (integer), `length_mm` (number) |
| `capacitor` | A capacitor soldered on a board you list as a made part. | `package` (string), `value` (string) |
| `control_board` | A bought microcontroller board. | `vendor`, `part_number` |
| `diode` | A diode soldered on a board you list as a made part. | `package` (string), `type` (string) |
| `dowel_pin` | A plain cylindrical pin. | `diameter_mm` (number), `length_mm` (number) |
| `fuse` | A fuse or resettable fuse. | `package` (string), `rating_a` (number) |
| `gearmotor` | A motor with a gearbox, with or without an encoder. | `gear_ratio` (number), `rated_voltage_v` (number), `encoder` (token), `encoder_counts_per_rev` (integer) |
| `led` | A light-emitting diode soldered on a board you list as a made part. | `package` (string), `colour` (string) |
| `line_guide` | A bought ring, eyelet or tube that guides a cable. | `inner_diameter_mm` (number) |
| `made_part` | A part made in the lab (printed, cast, cut or a bare circuit board) that carries a paper tag in the photos. | `tags` |
| `magnet` | A permanent magnet. | `diameter_mm` (number), `thickness_mm` (number) |
| `motor_driver_board` | A bought breakout board whose function is to drive motors. | `vendor`, `part_number` |
| `nut` | A threaded nut. | `thread` (string), `style` (string) |
| `pin_header` | An unshrouded strip of pins soldered to a board. | `positions` (integer), `rows` (integer), `orientation` (token) |
| `radio_module` | A bought wireless module or breakout board. | `vendor`, `part_number` |
| `resistor` | A resistor soldered on a board you list as a made part. | `package` (string), `resistance_ohm` (number) |
| `retaining_ring` | A retaining ring or clip on a shaft. | `shaft_diameter_mm` (number) |
| `rod` | Round bar stock, solid or hollow, cut to length. | `diameter_mm` (number), `length_mm` (number), `material` (string) |
| `screw` | A screw with a head. | `thread` (string), `length_mm` (number), `head` (token) |
| `sensor_board` | A bought breakout board whose function is to measure something. | `vendor`, `part_number` |
| `servo` | A bought servo actuator. | `vendor`, `part_number` |
| `set_screw` | A headless screw. | `thread` (string), `length_mm` (number) |
| `shaft_collar` | A collar clamped or screwed on a shaft. | `bore_mm` (number), `outer_diameter_mm` (number) |
| `socket_header` | A strip of receptacles soldered to a board, into which the pins of another board plug. | `positions` (integer), `rows` (integer), `orientation` (token) |
| `spacer` | A spacer or standoff, threaded or not, that holds parts apart. | `length_mm` (number), `outer_diameter_mm` (number), `inner_diameter_mm` (number), `thread` (string) |
| `spring` | A coil spring. | `outer_diameter_mm` (number), `free_length_mm` (number) |
| `strain_sensor` | An elastic strain sensor made in the lab from one of the given cut outlines. | `outline` (string) |
| `switch` | A switch worked by hand. | `actuation` (token), `terminals` (integer) |
| `threaded_insert` | A threaded metal insert set into a plastic part. | `thread` (string), `installation` (token), `length_mm` (number) |
| `transistor` | One transistor, or several in one package, soldered on a board you list as a made part. | `package` (string), `channel` (token) |
| `voltage_regulator_module` | A bought voltage regulator or converter board. | `vendor`, `part_number` |
| `washer` | A flat or spring washer. | `inner_diameter_mm` (number), `outer_diameter_mm` (number) |
| `wire_to_board_connector` | The board-side part of a keyed wire-to-board connector family: a shrouded header soldered to a board. | `series` (string), `positions` (integer), `mount` (token), `entry` (token) |

- Write a thread designation the way its standard does and without spaces: a unified thread as `<size>-<threads per inch>`, a metric thread as `M<diameter>x<pitch>`.
- Write a package by its usual case code or package name.
