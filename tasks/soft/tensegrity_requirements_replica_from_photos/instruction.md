# Specify eight replicas of a three-strut tensegrity robot that has been taken apart

A lab built one small rolling robot, a tensegrity: three struts held apart by a net of actuated cables and elastic
strain sensors. A partner now orders eight robots identical to it (24 struts). The robot can no longer be measured.
For a teardown record its net was cut away and the loose leads between its parts were taken off; then a colleague
photographed one of its struts on the electronics bench, complete and stage by stage as it was taken apart, down to
every made part on a cutting mat. Neither the net (cables, strain sensors) nor those leads are in any photo. Besides
the photos there are the controller firmware, the host software with its configuration and calibration files, the
fabrication data of the main board, the cut files of the strain sensors, and excerpts of vendors' public data on
families of bought parts. The struts are alike except for the colour of the covers on their end caps (red, green,
blue) and the index given to their controllers.

You are the requirements engineer. Recover from this evidence what the replicas need: how the net joined the struts,
and through which connector or pad of the main boards each cable's motor and each strain sensor is wired; the key
dimensions of a strut; every part mounted on the robot, bought or made, and how many of each; the requirements that
the evidence fixes and the quantities that follow from them. Write it down traced to the evidence. You are scored on
how well it matches the real robot.

## What you have (`/home/user/`)

| path | content |
|---|---|
| `photos/photo_001.jpg` .. `photos/photo_320.jpg` | the colleague's photo set of one strut, taken with a hand-held phone; no camera pose and no calibration was recorded |
| `photo_index.md` | for every photo: the scene, the teardown stage, the lens and the scale references on the mat; the nominal sizes of the scale references |
| `software/controller/onboard.ino` | the controller firmware: the one copy of its source in the lab's repository, shared by the three struts (its network name and password are replaced by placeholders) |
| `software/host/run_tensegrity.py` | the host control script |
| `software/host/tracking_config.json` | the configuration file of the camera tracking software that runs beside the host script |
| `software/host/calibration/calibration.json`, `new_calibration.json` | two calibration files of the strain sensors |
| `board/fab_data.json`, `board/fab_data.md` | the fabrication data of the main board (outline, placements, pads, which pads share copper; no names of parts, pins or nets) and the description of its format |
| `sensors/short_sensor_cut.dxf`, `sensors/long_sensor_cut.dxf` | the cut file of one sheet of each kind of strain sensor |
| `vendor/VD-01_*.md` .. `vendor/VD-08_*.md` | excerpts of the vendors' own public data, one family of bought parts per file: microcontroller boards (`VD-01`); dual motor driver boards (`VD-02`); capacitive touch sensing boards (`VD-03`); inertial measurement boards (`VD-04`); current and power monitor boards (`VD-05`); micro metal gearmotors (`VD-06`); wire-to-board connectors and pin and socket headers (`VD-07`); screws, threaded inserts, spacers and standoffs (`VD-08`). A file lists comparable parts side by side and does not say which of them, if any, is on the robot; it gives no prices |
| `docs/conventions.md` | the names, units and definitions the deliverables use, the order of trust between sources, the formulas of the derived quantities, the scope and categories of the BOM |
| `schemas/` | the JSON schemas of the three deliverables, the vocabulary of requirement quantities (`quantities.json`), the BOM categories (`categories.json`) and the unit table (`units.json`) |
| `schema_check.py` | the structure check of your submission (gates G0 and G2 below; it knows no answer): `python3 schema_check.py` |

There is no checker of the real robot and no way to query it: it exists only in the grader.

## What you deliver (`/home/user/submission/`)

- `requirements.yaml`: the traced requirements sheet. `items`: one entry per requirement that the evidence fixes
  (`id`, `quantity` = a name of `schemas/quantities.json`, `comparator`, `bound`, `unit`, `source_ids`, optional
  `note`). `derived`: the derived quantities that `docs/conventions.md` defines, computed from the inputs you
  recovered (`id`, `quantity`, `value`, `unit`, optional `inputs` and `derivation`).
- `bom.json`: the bill of materials of the order. `robots` = the size of the order, and `lines`: one line per part
  mounted on the robot (`line_id`, `kind`, `category` = a category of `schemas/categories.json`, `description`,
  `qty_per_strut`, `qty_per_robot_extra`, `source_ids`; for a bought part its `vendor`, its `part_number` and the
  `attributes` of its category; for a made part the paper `tags` of its instances in the photos).
- `cable_net.json`: the net and the strut. `controllers`: which strut carries which controller. `cables`: for every
  actuated cable the node of its motor and winch, the node its far end is hooked to, and the connector of the main
  board its motor is plugged into. `sensors`: for every strain sensor its two nodes and the pad of the main board that
  reads it. `spare_inputs`: the sensor inputs that carry no sensor. `geometry`: the key dimensions, in metres.
  `counts`.
- `notes/` (optional): your method, code and intermediate estimates. Recorded, never scored.

`docs/conventions.md` gives the meaning of every field and `schemas/` the exact form. The shape, with placeholders:

```yaml
items:
  - {id: REQ-001, quantity: <name>, comparator: "=", bound: <number or token>, unit: <unit>, source_ids: [<id>, <id>]}
derived:
  - {id: DER-001, quantity: <name>, value: <number>, unit: <unit>, inputs: [<id>], derivation: "<text>"}
```

```json
{"robots": 8, "lines": [
  {"line_id": "L001", "kind": "purchased", "category": "<category>", "vendor": "<vendor>", "part_number": "<number>",
   "attributes": {"<attribute>": "<value>"}, "qty_per_strut": 0, "qty_per_robot_extra": 0, "source_ids": ["<id>"]},
  {"line_id": "L002", "kind": "made", "category": "made_part", "tags": ["<tag>"], "qty_per_strut": 0,
   "qty_per_robot_extra": 0, "source_ids": ["<id>"]}]}
```

```json
{"controllers": [{"controller_index": 0, "strut": "<colour>"}],
 "cables": [{"motor_index": 0, "winch": {"strut": "<colour>", "end": "<U|D>"},
             "hook": {"strut": "<colour>", "end": "<U|D>"}, "motor_connector": "<placement>"}],
 "sensors": [{"sensor_index": 0, "between": [{"strut": "<colour>", "end": "<U|D>"},
                                             {"strut": "<colour>", "end": "<U|D>"}], "connector_pad": "<pad>"}],
 "spare_inputs": [{"strut": "<colour>", "connector_pad": "<pad>"}],
 "geometry": {"<dimension>": 0.0}, "counts": {"<count>": 0}}
```

## How it is scored

`reward` = G0 x G1 x G2 x the weighted mean of five criteria, each in [0, 1]. A gate is 1 or 0:

- **G0 structure**: the three files are there, parse and pass `schema_check.py`.
- **G1 submission integrity**: nothing in `/home/user/submission` may be written after your session ends, and the
  folder itself must stay the one your session wrote: a symbolic link, or a folder replaced or renamed into place
  after the session, scores 0. Any process you leave running is stopped when grading starts.
- **G2 no hedging**: fails when `cable_net.json` holds two records with the same index (two spare inputs of one
  strut), when `requirements.yaml` holds one quantity twice with the same comparator or one derived quantity twice, or
  when a list is longer than 60 items, 20 derived entries or 60 BOM lines. `schema_check.py` checks this too. Not
  gated, only scored: a record, a field or an entry left out; a lower and an upper limit on one quantity; one part on
  several BOM lines; entries and lines that the evidence does not support.

| criterion | weight | what is compared |
|---|---|---|
| `net_and_geometry_replication` | 0.25 | `cable_net.json` against the real robot, in three groups: topology (0.10 of the criterion), key dimensions (0.85), counts (0.05) |
| `requirements_sheet` | 0.05 | the `items` of `requirements.yaml` against a hidden reference sheet: the requirements that this evidence fixes, valued on the real robot |
| `derived_quantities` | 0.03 | the `derived` entries against the values that the public formulas give with the real inputs |
| `bom_buildability` | 0.37 | how many corrections your order (`bom.json`) needs before the robot can be built from it |
| `bom_procurement` | 0.30 | what your order costs against what the exact order costs, at the lab's real purchase prices |

**Topology**, in four blocks. A block is the mean over its records, and a record you leave out earns nothing.

| block | share | records | a record earns the block when |
|---|---|---|---|
| sensor leads | 0.40 | every strain sensor and every spare input | its `connector_pad` is right |
| motor leads | 0.15 | every actuated cable | its `motor_connector` is right |
| ends | 0.35 | every actuated cable and every strain sensor | both of its nodes are right (`winch` and `hook`, or the pair `between`; strut and end) and its lead field of the rows above is right too: nodes count only in a record whose lead is traced |
| controllers | 0.10 | every controller | its `strut` is right |

The key dimensions, the reference items of the requirements sheet and the derived quantities each carry a weight of
1, 2 or 3 according to how much the evidence asks for; which entry carries which weight is not published. Between a
full-credit point and a no-credit point a score is linear.

**Key dimensions**: each by its absolute error; the full-credit and no-credit points follow from what the photo set
resolves for the dimension. For the dimensions of one small part (`winch_axis_offset`, `cover_sphere_radius`,
`rod_insertion_in_end_cap`, `winch_flange_diameter`, `hook_circle_radius`) the full-credit point is 0.087 to 0.18 mm
and the no-credit point 0.348 to 0.72 mm, depending on the dimension; for the others (`strut_overall_length`,
`node_spacing`, `motor_module_outer_face_station`, `envelope_diameter`, `electronics_bay_length`) the full-credit
point is 0.21 to 0.343 mm and the no-credit point 0.84 to 1.372 mm. A dimension you leave out earns nothing.
**Counts**: exact.

**`requirements_sheet`**: an item matches a reference item when it has the same `quantity`, a comparator of the same
kind (`=` and `==`; `<=` and `<`; `>=` and `>`) and a `bound` that earns credit under the table "How values are
compared" of `docs/conventions.md`. A match none of whose `source_ids` is of a kind the reference item rests on (all
photos are one kind, every other id is its own) keeps 0.5 of its credit. Recall is the weighted mean credit of the
reference items. Precision is taken over every item you submit: a match counts with its credit; an item on a quantity
whose value a given file states outright is not on the reference sheet, and is left out of the count when its value is
right and counted as wrong when it is not; every other item counts as wrong (a wrong value or comparator, or a
quantity that the evidence does not fix: the vocabulary is wider than the evidence). The score is the harmonic mean of
precision and recall, 0 when either is 0.

**`derived_quantities`**: each derived quantity of `docs/conventions.md` scores under the same table against the value
that its formula gives with the real inputs. One you leave out, or give in a unit of another dimension, earns nothing.

**`bom_buildability`**: `bom.json` is judged as an order, by the number of corrections it needs before the robot can
be built from it. The order has to cover every bought part that the rebuild needs, and a ready-made cable assembly
that connects bought parts is such a part; leads that the lab crimps itself are not graded. Lines in an out-of-scope
category are ignored, and so are lines for a few items whose listing is a matter of bookkeeping (which items is not
published): such lines are neither rewarded, nor counted, nor priced. Lines of one category with the same identifying
fields are merged and their quantities added. The merged lines are assigned one to one to the parts of the real robot.
A line is recognised as a part when its category is the part's and what `docs/conventions.md` names under "recognised
by" is right: for a bought board the vendor and part number (where a vendor sells one board with and without header
strips fitted under two part numbers, either is accepted); for a made part a non-empty subset of the part's paper
tags; for any other category those attributes of the category that the grader compares (it does not compare every
listed attribute, and which it compares is not published). Integers, tokens and strings must be equal after the
normalisation that `docs/conventions.md` promises (common writing variants of a vendor name, a connector series, a
thread or a package are tolerated as well); a length or a diameter must be within what the close-ups resolve, 0.083 to
0.102 mm on small hardware and on a rod's diameter and 1.39 mm on a rod's length; a gear ratio within 1; any other
number equal.

Every recognised part whose quantity is wrong is one correction; a quantity is right when `qty_per_strut` and
`qty_per_robot_extra` make the number of pieces that the order needs of the part, counted as `docs/conventions.md`
states under "Quantities". In each category the parts that no line is recognised as and the lines that are recognised
as no part are counted, and whichever count is larger is added: a line with a wrong field and the part it stands for
are one correction, because putting that field right mends the order. A part listed under a category other than its
own counts once there, as a line that is no part, and once in its own category, as a part without a line: two
corrections. With n corrections the score is `1-n/2`, never below zero: one correction leaves 0.5 of the criterion, 2
or more leave nothing. There is no partial credit per line, and every line added on speculation that is recognised as
no part is a correction.

**`bom_procurement`**: the bought parts of your order of 8 robots are priced with the lab's real purchase prices, pack
sizes and quantity discounts, which are not given to you. D = (the sum, over the bought parts of the real robot, of
the absolute difference between what your recognised line costs and what the exact quantity costs, a part without a
recognised line counting as not bought; plus the cost of every merged line that is recognised as no part, priced as
the cheapest real part of its category at your quantity, or at nothing when the real robot has no bought part of that
category) / the cost of the exact order. Full credit at D <= 0.5 %, no credit at D >= 5 %.

The real values were settled by the order of trust of `docs/conventions.md`; it decides when sources disagree:

- **What the hardware is:** the photographed object comes first, then the fabrication data of the main board, then the
  vendor's data of a part you have identified, then constants about the hardware that are written in the software.
- **What the software does:** the code decides, not its comments.
- A source that does not mention a thing does not contradict a source that does.

## Notes

- Tools: Python 3.12 (`/opt/venv`, on PATH) with numpy, scipy, OpenCV (`cv2`, headless), scikit-image, matplotlib,
  Pillow, PyYAML, jsonschema, trimesh and ezdxf. No network, no OCR engine. The photo set is large: process the photos
  programmatically and view only the images you need.
- Units: each file's units are in `docs/conventions.md` (`geometry` is in metres; `schemas/units.json` lists the units
  accepted in `requirements.yaml`).
- Do not modify the given files in your home directory: only `/home/user/submission/` is graded.
- Environment facts (harness limits): a single tool call whose input runs above roughly 25-30k tokens is dropped by
  the harness (it is never executed and no error comes back), and a response that reaches the 64k output-token limit
  is lost and, after at most a few retries, ends the session with whatever is in `/home/user/submission/` at that
  moment. Write large files in several parts, keep every single response well below that limit, write a first complete
  submission early and refine it, and run `schema_check.py` after each part.
