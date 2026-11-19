# Submission contract

You deliver three files in `/home/user/submission/`. This page is their contract: what each file holds, the
closed lists of ids, every rule its structure must meet, and the bands that the scoring in `instruction.md`
relies on.

| file | what it is | section |
|---|---|---|
| `requirements.yaml` | the requirements sheet: the facets of the order and your derived values | 2 |
| `bom.json` | the bill of materials: one SKU for each purchase line | 3 |
| `kinematic_sketch.json` | the kinematic sketch of the reference unit | 4 |

Each file is a regular file (not a link), UTF-8 without a byte-order mark, of at most 256 kB (262144 bytes).
The JSON schemas in `schemas/` are the machine-readable form of the shapes. `schema_check.py` checks a submission
against them (section 7). A file that breaks a rule of sections 1 to 5 zeroes the whole result.

The lists of ids on this page are closed: no other key is accepted, and where a section says *exactly*, every key
is required.

You do not deliver the car record or the catalogue. You need both to run `procedure.py` (`quantities.md`,
sections 2 and 3): your derived values and your choice of a BOM rest on them.

## 1. Sources

A `sources` field is a list of at most 16 source ids. A source id names one given document or picture:

| kind | ids | what an id names |
|---|---|---|
| `B` | `B-01` to `B-16` | a numbered item of brief.md; Appendix A and Appendix D are cited by the item that invokes them |
| `BL` | `BL-01` to `BL-10` | an entry of build_log.md |
| `PH` | `PH-01` to `PH-33` | a photo |
| `DS` | `DS-01` to `DS-20` | a datasheet page |
| `ER` | `ER-1` to `ER-3` | an erratum notice |
| `LR` | `LR-1` to `LR-4` | a lot report |

Only the ids listed here exist. `catalogue/quote.md` and `acceptance.json` are not sources: cite the pages they
point to and the brief item they repeat.

Cite the place where the value itself can be read:

- a reading of the scale: the photograph that shows the display with that reading;
- a value that the brief or the build log states in words: the item or the entry that states it;
- a value of Appendix A: the brief item that the row or the scenario of the value names. B-10, which calls on
  the appendix as a whole, is accepted as well;
- a value of a lot: the lot report that prints the samples of that lot;
- a value corrected by an erratum: the erratum notice.

## 2. `requirements.yaml`

```yaml
facets:
  - {id: <facet id>, value: <number>, unit: "<unit string>", sources: [<source id>, ...]}
  - ...
derived:
  <derived id>: {value: <number>, unit: "<unit string>"}
  ...
```

A mapping with exactly the two keys `facets` and `derived`.

### 2.1 `facets`

A list of 1 to 20 entries, each a mapping with exactly `id`, `value`, `unit` and `sources`. An `id` is one of the
20 facet ids below, and no id appears twice. `value` is a finite number. `unit` is the unit string of the facet,
written exactly as in the table.

| id | unit | tolerance | what it is | its source |
|---|---|---|---|---|
| `pod_mass_g` | `g` | 1 | Mass of the customer's sensor pod, as the scale reads it. | the photograph of the reading |
| `as_found_mass_g` | `g` | 1 | Mass of the reference unit, as the scale reads it with the whole car on it. | the photograph of the reading |
| `as_found_front_axle_g` | `g` | 1 | Reading of the scale under the two front wheels of the reference unit, its rear wheels standing on a block level with the scale's platform. | the photograph of the reading |
| `steering_lock_deg` | `deg` | 0.05 | Steering lock that the reference unit is set to. | the build-log entry |
| `replica_count` | `count` | 0 | Number of replicas ordered. | the brief item |
| `eco_usd` | `USD` | 0.005 | Engineering-change fee for one changed purchase line. | the brief item |
| `scan_plane_min_mm` | `mm` | 0.05 | Lower limit of the league's band for the height of the LiDAR scan plane. | the brief item |
| `scan_plane_max_mm` | `mm` | 0.05 | Upper limit of the league's band for the height of the LiDAR scan plane. | the brief item |
| `ground_clearance_min_mm` | `mm` | 0.05 | Lower limit of the ground clearance. | the brief item |
| `steering_torque_ratio_max` | `1` | 0.005 | Upper limit of the ratio of the steering torque needed at standstill to the stall torque of the servo. | the brief item |
| `lock_time_max_s` | `s` | 0.005 | Upper limit of the time from full lock to full lock. | the brief item |
| `mass_max_kg` | `kg` | 0.005 | Upper limit of the mass of a car. | the brief item |
| `lap_time_max_s` | `s` | 0.005 | Upper limit of the lap time on the reference track. | the brief item |
| `cell_lot_capacity_factor` | `1` | 0.005 | Capacity factor that the brief's lot policy gives for the arrival lot of the battery SKU fitted to the reference unit. | the lot report of the lot |
| `cell_lot_resistance_factor` | `1` | 0.005 | Resistance factor that the brief's lot policy gives for the arrival lot of the battery SKU fitted to the reference unit. | the lot report of the lot |
| `tyre_lot_friction` | `1` | 0.005 | Friction on ground class M that the brief's lot policy gives for the arrival lot of the tyre SKU fitted to the reference unit. | the lot report of the lot |
| `lidar_lot_range_m` | `m` | 0.005 | Range, on the target class of the stop zone of the acceptance, that the brief's lot policy gives for the arrival lot of the LiDAR SKU fitted to the reference unit. | the lot report of the lot |
| `compute_power_w` | `W` | 0.05 | Supply power of the compute SKU fitted to the reference unit, as corrected by the maker's erratum. | the erratum notice |
| `power_board_fuse_a` | `A` | 0.05 | Fuse rating of the power distribution board SKU fitted to the reference unit, as corrected by the maker's erratum. | the erratum notice |
| `schedule_duration_s` | `s` | 0.5 | Total duration of the session programme of the acceptance: the driving and the standing time of all its blocks, repeats included. | the brief item |

A facet counts when its value differs from the reference value by no more than the tolerance, its unit is the
unit string of the table, and at least one of its sources is a source the value can be read from (section 1). A
facet that you leave out does not count. A facet that you submit and that does not count lowers the result as
well, because every submitted facet adds to the divisor (section 6).

### 2.2 `derived`

A mapping with exactly the 7 derived ids below, each a mapping with exactly `value` and `unit`. A derived value
is your prediction of a quantity of `procedure.py` for a replica built to **your own BOM**: the reference unit,
with the SKUs of your `bom.json` and the pod, in the context of the acceptance (Appendix A, with the lot table of
B-12), under the conditions of the table.

| id | unit | conditions | what it is |
|---|---|---|---|
| `mass_kg` | `kg` | `{"config": {"body_shell": false}}` | Mass of a replica with your BOM and the pod. |
| `front_axle_load_n` | `N` | `{}` | Static load on the front axle of that replica. |
| `scan_plane_height_mm` | `mm` | `{}` | Height of the LiDAR scan plane of that replica. |
| `ground_clearance_mm` | `mm` | `{}` | Ground clearance of that replica: the height of its lowest point, wheels and tyres left out of account. |
| `envelope_height_mm` | `mm` | `{}` | Height of the highest point of that replica. |
| `steering_torque_nm` | `N.m` | `{"surface": "M"}` | Torque the steering servo of that replica must give to turn the front wheels at standstill. |
| `lock_time_s` | `s` | `{"surface": "M"}` | Time that replica takes from full lock to full lock. |

The grader computes the same quantity with the car record of the real car, your BOM and the lots as they
arrived, and compares.

## 3. `bom.json`

```json
{"motor": "<SKU>", "pinion": "<SKU>", "motor_controller": "<SKU>", "battery": "<SKU>", "lidar": "<SKU>",
 "deck_post": "<SKU>", "compute": "<SKU>", "power_board": "<SKU>", "tyres": "<SKU>", "steering_servo": "<SKU>"}
```

A mapping with exactly the ten purchase lines as keys. Each value is the code of an SKU that the quote lists for
that line.

| purchase line | item | pieces per car | SKUs on the quote |
|---|---|---|---|
| `motor` | 540-size brushed motor | 1 | 4 |
| `pinion` | motor pinion | 1 | 4 |
| `motor_controller` | motor controller | 1 | 3 |
| `battery` | NiMH stick pack | 1 | 6 |
| `lidar` | 2D scanning LiDAR | 1 | 5 |
| `deck_post` | deck post | 4 | 3 |
| `compute` | compute module | 1 | 4 |
| `power_board` | power distribution board | 1 | 2 |
| `tyres` | wheel and tyre, pre-mounted | 4 | 4 |
| `steering_servo` | steering servo | 1 | 5 |

The file may name any combination of quoted SKUs. Whether the car can be built and whether it is acceptable is
judged by the pass items, not by the structure check.

## 4. `kinematic_sketch.json`

```json
{"<sketch id>": {"nominal": <number>, "tolerance": <number>, "sources": ["<source id>", "..."]}, "...": {}}
```

A mapping with exactly the 33 sketch ids below, each a mapping with exactly `nominal`, `tolerance` and
`sources`. The sketch describes the reference unit as it was photographed; Appendix D of the brief defines every
quantity. `nominal` is your value, in the unit that ends the id (`_mm`: millimetres, `_deg`: degrees; an id that
ends in `_teeth` is a count and takes an integer). `tolerance` is a finite number, not negative, in the same
unit: how far the true value may lie from your nominal, as you judge it. `sources` lists what you read the value
from.

| id | group | unit | kind |
|---|---|---|---|
| `deck_top_height_mm` | heights | `mm` | length |
| `deck_thickness_mm` | heights | `mm` | length |
| `lowest_point_height_mm` | heights | `mm` | length |
| `highest_point_height_mm` | heights | `mm` | length |
| `lidar_top_height_mm` | heights | `mm` | length |
| `antenna_top_height_mm` | heights | `mm` | length |
| `post_lower_end_height_mm` | heights | `mm` | length |
| `rear_body_post_top_height_mm` | heights | `mm` | length |
| `deck_length_mm` | layout | `mm` | length |
| `deck_width_mm` | layout | `mm` | length |
| `deck_front_edge_x_mm` | layout | `mm` | length |
| `lidar_x_mm` | layout | `mm` | length |
| `compute_x_mm` | layout | `mm` | length |
| `motor_controller_x_mm` | layout | `mm` | length |
| `post_spacing_x_mm` | layout | `mm` | length |
| `post_spacing_y_mm` | layout | `mm` | length |
| `battery_bay_length_mm` | layout | `mm` | length |
| `kingpin_ground_offset_mm` | steering | `mm` | length |
| `caster_angle_deg` | steering | `deg` | angle |
| `steering_arm_mm` | steering | `mm` | length |
| `bellcrank_output_arm_mm` | steering | `mm` | length |
| `bellcrank_input_arm_mm` | steering | `mm` | length |
| `servo_saver_arm_mm` | steering | `mm` | length |
| `toe_in_per_side_deg` | steering | `deg` | angle |
| `wheelbase_mm` | chassis | `mm` | length |
| `track_front_mm` | chassis | `mm` | length |
| `track_rear_mm` | chassis | `mm` | length |
| `overall_length_mm` | chassis | `mm` | length |
| `overall_width_mm` | chassis | `mm` | length |
| `pinion_teeth` | gears | `count` | count |
| `spur_teeth` | gears | `count` | count |
| `bevel_pinion_teeth` | gears | `count` | count |
| `ring_gear_teeth` | gears | `count` | count |

## 5. Units and numbers

The unit strings are plain ASCII and are compared character by character:

`mm`, `deg`, `count`, `g`, `kg`, `N`, `N.m`, `s`, `USD`, `1`, `V`, `K`, `m`, `W`, `A`

Every number is finite; an integer is at most 2^53 in magnitude. The YAML file may use aliases, but with its
aliases expanded it has at most 20000 nodes; mapping keys are strings; no key appears twice in a mapping, in
YAML or in JSON. `NaN` and `Infinity` are not numbers.

In `requirements.yaml`, write every unit string in quotes, as the example of section 2 does: YAML reads an
unquoted `1` as a number, and a unit that is not a string breaks the schema. Write every number as a plain
decimal number, such as `1250` or `12.5`, without an exponent: YAML reads `1.25e3` as a string and not as a
number. YAML merge keys (`<<`) are not accepted.

## 6. Bands of the scoring

`instruction.md` names the five items and their weights. Their bands:

- **`kinematic_replication`.** Each quantity of the sketch earns a score between 0 and 1 from the error of its
  `nominal` against the real car. A length: 1 when the error is at most the larger of 1.5 % of the true value
  and 1.0 mm; 0 when it is at least the larger of 6 % and 4.0 mm; in a straight line between. An
  angle: 1 up to 0.5 degrees, 0 from 3 degrees. A count: 1 when equal, else 0. The score of a group is
  the mean over its quantities, and the item is the weighted sum of the groups: heights 0.30, layout
  0.25, steering 0.35, chassis 0.05, gears 0.05. `tolerance` and `sources` are not scored.
- **`requirements_sheet`.** 2 x (facets that count) / (20 + facets you submit).
- **`derived_requirements`.** Each derived value earns 1 when its relative error against the grader's value is
  at most 3 %, 0 when it is at least 15 %, in a straight line between; the item is the mean of the
  seven. A value with another unit string than the table's earns 0, and so does a value whose counterpart on
  the real car is not finite (a lock time when the servo of your BOM stalls).
- **`bom_acceptance`.** 1 when the real car with your BOM passes all 17 pass items of Appendix A, else 0.
- **`bom_economy`.** With *J* the price of the cheapest acceptable car and *J_eff* the smallest value, over all
  acceptable cars, of the car's price plus 45 USD for every purchase line in which the car differs from your
  BOM: the item is 1 − (*J_eff* − *J*) / 45, held between 0 and 1.

## 7. `schema_check.py`

`/opt/venv/bin/python3 schema_check.py` checks `/home/user/submission/` (or the folder you name) and prints
every structure problem it finds: a file that is missing, not a regular file, too large, not UTF-8 or that does
not parse; a number that is not finite; a mismatch with a schema of `schemas/`; a facet id that appears twice.
It prints `Structure OK` when it finds none. It reads the structure only: it does not look at any value, and it
says nothing about any score.
