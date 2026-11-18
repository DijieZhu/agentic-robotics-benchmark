# Specify eight upgraded replicas of an inspection quadruped that is no longer in the lab

A lab built one 12-actuator, belt-reduced quadruped for walkway inspections (the "reference unit"). An inspection
contractor now orders eight replicas with the same kinematics, the same printed parts and the same printed stops,
fitted with new actuator modules, a new battery pack, new drivers and a new hub for a heavier, longer duty: the
contractor's sensor pod rides on the deck. The reference unit is on a long loan to a partner lab and cannot be measured
again. What is left: a colleague's photo set of it on the lab bench, the build log, the vendors' datasheet and erratum
pages, a distributor quote, three incoming-inspection reports and the customer's brief.

You are the requirements engineer. Recover from this evidence what the replicas need (the unit's key dimensions, stops
and module housings, where its masses sit, the payload, the loads the customer's acceptance duty puts on every joint and
on the power system, the real capability of the modules and packs you would buy), choose the actuator modules, the
pack, the drivers and the hub, and write it down traced to the evidence. You are scored on the OUTCOME in the real world
of that unit: how well your sketch replicates the real reference unit, how close your derived loads are to the real
ones, whether your BOM passes the customer's acceptance procedure on the real unit with the real delivered lots, and
what it costs.

## What you have (`/home/user/`)

| path | content |
|---|---|
| `brief.md` | the customer brief, clauses `[B-01]..[B-16]`, including the ACCEPTANCE PROCEDURE (B-14, public and exact, including the models of a module lot's and of a cell lot's capability), the deliverables and Appendix D (the definitions and angle conventions of every graded quantity) |
| `build_log.md` | the colleague's notes `[BL-01]..[BL-13]` on the build and on the photo session |
| `photos/PH-01.jpg .. PH-36.jpg`, `photos/photos.json` | the survey photo set as the colleague left it. `photos.json` lists per photo its public label and the focal-length class of the phone camera that took it (`wide` = the main camera, `normal` = the 2x camera). Camera poses and intrinsics are NOT known and the lenses are not calibrated |
| `datasheets/DS-*.png` | vendor pages (images) of the actuator modules, battery packs, drivers and the hub, plus one erratum sheet of each vendor that issued one; typical values in each vendor's own units at the vendor's voltages |
| `price_list.md` | the distributor quote: every orderable part id, role, description, housing class, list price and lead time, and the lots the modules and the packs ship from |
| `lot_reports/LR-*.png` | three incoming-inspection reports (scanned pages, images only): two of actuator-module lots, one of a battery cell lot |
| `schemas/` | JSON schemas of the three deliverables (`requirements`, `bom`, `kinematic_sketch`) with their unit table, and of the inputs the public grading modules take: `quad_model` (a replica model), `layer` (the brief's values), `catalogue` (a transcription of the pages and the quote) and `lots` (a description of the delivered lots) |
| `grader_public/` | the grader's own public modules (`quad/req_info/q1/`: the acceptance procedure, the replica-model builder `quadmodel.py`, the lot model, the optimum search (the cheapest accepted BOM), the engineering-change repair, every criterion) and `grader_config.public.json`. They contain no vendor, price, lot or unit number: they take a replica model, a layer, a catalogue and a lots description as inputs, and carry nothing of the real unit |
| `schema_check.py` | the structure gate G0 on your submission (no ground truth involved) |

There is no checker of the real unit and no way to query it: the real unit, its lots and the vendors' own numbers exist
only in the grader.

## What you deliver (`/home/user/submission/`)

- `requirements.yaml`: the traced requirements sheet. One item per requirement the documents or the photos fix (`id`
  REQ-..., `quantity` from the schema's vocabulary, `comparator`, `bound`, `unit`, `source_id` = the clause (`B-07`,
  `BL-03`), the photo (`PH-35`), the page (`DS-TK-2C14`), the lot report (`LR-yyyy-nnnn` or `LR`) or `derived`), plus
  the 16 derived items of brief B-16 for YOUR BOM (source `derived`, with a `derivation` note):
  `proof_peak_torque:hip_roll`, `proof_peak_torque:hip_pitch`, `proof_peak_torque:knee`, `duty_rms_torque:hip_roll`,
  `duty_rms_torque:hip_pitch`, `duty_rms_torque:knee`, `standup_peak_torque:knee`, `joint_peak_speed:hip_pitch`,
  `joint_peak_speed:knee`, `supply_peak_current`, `mission_charge`, `drive_peak_current:knee`,
  `drive_rms_current:knee`, `robot_mass`, `loaded_com_x`, `peak_normal_force`.
- `bom.json`: `battery_family` (one of the brief's pack families) and one line per part: three `actuator` lines, one
  per joint type (`hip_roll`, `hip_pitch`, `knee`, as `joint_type`) with `count` 4 (one module per leg); one `battery`
  line with `count` 1; one `driver` line whose `count` is the boards the 12 joints' channels need; one `hub` line with
  `count` 1. Every `part_id` from `price_list.md`.
- `kinematic_sketch.json` (SI: m, rad; `units` = `{"length": "m", "angle": "rad"}`): `geometry` = the eight
  Appendix-D lengths (`pitch_axis_spacing`, `roll_axis_spacing`, `pitch_axis_drop`, `leg_offset`, `hip_spacer`,
  `thigh_length`, `shank_length`, `foot_radius`), each `{nominal, tol}`; `spacer_legs` = the legs that carry spacer
  rings; `stops` = the printed stops `[lower, upper]`
  of each joint type in the Appendix-D angle conventions; `housing_class` = per joint type the class of the can its
  housing takes as replicated (`R52`, `R38` or `R62`); `pack_bay` = the bay that carries the pack (`nose_box` or
  `belly_bay`); `modules` = per joint type the module of your BOM's actuator line.
- `notes/` (optional): your method, code, intermediate estimates, your replica model and catalogue transcription.
  Recorded, never scored.

## How it is scored

Gates (multiply every criterion): **G0 structure** (the files parse and pass `schema_check.py`; with the grader's
catalogue and layer also the driver count and the battery family) and **G1 submission integrity** (nothing in
`/home/user/submission` may be written after your session ends, and the folder itself must stay the one your session
wrote: a symbolic link, or a folder replaced or renamed into place after the session, scores 0; any process you leave
running is stopped when grading starts). Then five criteria, each in [0, 1], weighted:

| criterion | weight | what is measured | band (linear between full and zero credit) |
|---|---|---|---|
| `kinematic_replication` | 0.30 | your sketch vs the real reference unit (Appendix D): lengths group (0.55 of the criterion) = the eight lengths, stops group (0.30) = the six stops, classes group (0.15) = the three housing classes, the spacer legs (as a set) and the pack bay, each exact | lengths: full <= the larger of 0.5 mm and 0.25 % of the true value, zero >= the larger of 4 mm and 2 %; stops: full <= 1 deg, zero >= 6 deg |
| `requirements_sheet` | 0.10 | facet F1 between your non-derived items and a hidden reference sheet valued in the real world: the brief's customer requirements (incl. the payload band, whose top is read off the kitchen scale) and the pod's box length, width and height as B-04 defines the box (`payload_box_length`, `payload_box_width`, `payload_box_height`); an item matches a reference item only with the same `quantity` AND the same `comparator` (`<` and `<=` count as the same comparator, a reversed one does not match); an item traced to a source other than the reference item's clause(s) keeps half its weight; invented or wrong-valued items lower the precision | brief items (incl. the payload band): relative error full <= 2 %, zero >= 4 %; the pod's box: full <= 1 %, zero >= 6 % |
| `derived_requirements` | 0.20 | the 16 derived items of B-16 vs their values in the real world with YOUR BOM installed on the real unit and the real delivered lots (B-14 on the real unit over the real acceptance duty); a missing item scores 0 | relative error, normalised by max(abs(true value), floor) with floors 0.05 N m, 0.2 rad/s, 0.2 A, 0.05 Ah, 0.01 kg, 1 N and 0.1 m (so `loaded_com_x` has an absolute band: full <= 3 mm, zero >= 15 mm): full <= 3 %, zero >= 15 % |
| `bom_acceptance` | 0.10 | the acceptance procedure of B-14 run on the real unit with the weakest expected units of the real delivered module lot and cell lot (B-14 g, i): the unit is accepted (1) iff all 16 items pass (the 12 actuator joints, the battery, the drivers, the hub, the budget), else 0 | pass / fail |
| `bom_economy` | 0.30 | `max(0, 1 - (J_eff - J*) / 60 USD) x taper`: `J*` = the cheapest BOM that passes the acceptance procedure on the real unit with the real lots (recomputed live by the grader; a vendor without an inspection report counted as B-10 says, here and in `bom_acceptance`); `J_eff` = your BOM after the grader's deterministic repair: while an item fails, the first failing line in the order knee, hip_pitch, hip_roll, battery, drivers, hub is replaced by the cheapest line that makes its items pass (the battery family is never changed) and the brief's engineering-change cost is added per replaced line; 60 USD = the brief's engineering-change cost per line (B-13): a BOM one engineering change dearer than `J*` earns nothing; taper = 1 if `J_eff` <= the B-12 ceiling, else `max(0, 1 - (J_eff - ceiling) / (0.1 x ceiling))` | full at `J_eff` = `J*`, zero at `J*` + 60 USD |

`overall` = the weighted mean. The acceptance procedure and every formula above are public (brief B-14,
`grader_public/`); their inputs (the unit's geometry, housings and mass layout, the payload, the vendors' numbers, the
lots' real capability) are not: they are behind the photos, the build log, the datasheet and erratum pages, the quote
and the lot reports.

## Notes

- Tools: Python 3.12 (`/opt/venv`, on PATH) with numpy, scipy, OpenCV (`cv2`, headless), scikit-image, MuJoCo 3.14,
  matplotlib, Pillow, PyYAML, jsonschema, trimesh. No network, no OCR engine. The photos and pages are large: process the
  photos programmatically and view only the images you need; read the document pages as images.
- Units: SI in the files (brief B-03: the schemas' unit table lists accepted equivalents such as mm, g, min, rpm, mAh,
  lbf.in, and an item's unit must convert to the SI unit of its quantity).
- The public modules run on your own estimates. `grader_public/quad/req_info/q1/` needs a replica model (format
  `quad-model/1`: your Appendix-D geometry, the printed parts and fixed components as rigid inertials, the module
  seats, the pack bay and the pod mount; `quadmodel.py` builds its dynamics model, B-14 a), a layer (`quad-layer/1`:
  the brief's values), a catalogue transcription of the pages and the quote (`quad-catalogue/1`) and a lots description
  (`quad-lots/1`: your reading of the delivered lots); their schemas are in `schemas/`. With them
  `procedure.Procedure(model, layer)` gives the acceptance verdict (`evaluate`) and the derived items (`derived`) of a
  BOM, and `optimum.optimal_bom` the cheapest BOM it accepts, all for YOUR estimates.
- Do not modify the given files in your home directory; the grader reads its own copies.
- Environment facts (harness limits): a single tool call whose input runs above roughly 25-30k tokens is dropped by the
  harness (it is never executed and no error comes back), and a response that reaches the 64k output-token limit is lost
  and, after at most a few retries, ends the session with whatever is in `/home/user/submission/` at that moment. Write
  large files in several parts (one function group per write), keep every single response well below that limit, write
  a first complete submission early and refine it, and run `schema_check.py` after each part.
