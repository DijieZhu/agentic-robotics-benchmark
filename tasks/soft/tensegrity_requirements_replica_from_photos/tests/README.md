# tensegrity_requirements_replica_from_photos: notes for maintainers and validators

The agent-facing contract is `instruction.md`; the names, units, definitions, order of trust and formulas the
deliverables use are in `environment/setup/payload/docs/conventions.md`. This page records what the robot of the task
is made of, what the hidden standard consists of and which of the designers' files each part of it comes from, how it
is rebuilt, where each gate and criterion is implemented, what the neutral and W0 lists mean, how the grader's own QC
is re-run, and what is known to be rough. State of 2026-11-17: scoring v1.1, frozen before the one model episode that
is to be read. v1.0 was frozen before any episode; the first episode on it was cut off by a failure of the model
channel and does not count, and v1.1 was decided after it (section 9 says plainly what that means). The hidden truth,
the photos and every given file are the same bytes in v1.0 and v1.1; what changed is how the same answers are scored.
Who designed the robot, where every source was taken from and under which licence is in `NOTICE.md`. Neither file is
staged into the agent's sandbox: `tests/` exists only in the verifier.

Words: "the designers" = the people who designed and published the robot; "the maintainers' build" = the scripts and
records from which the hidden standard, the givens and the photos were generated (not part of this repository). A band "a / b" = full credit up to a, none from b, linear between.

## 1. What the robot of this task is made of

The photos show the designers' published whole-strut mesh (sha256 `8dc4caff...4614`, 1 118 331 triangles), rendered
triangle for triangle as one strut on a bench, complete and stage by stage as it is taken apart. Every value of the
hidden standard is read from the designers' own files (that mesh, the main board's board file, the controller
firmware, the host control script, the tracking configuration, two calibration files, the two sensor cut outlines,
the bill-of-materials workbook) or from data the vendors of the bought parts publish themselves. Nothing of the robot
is modelled, redrawn or completed; there is no hidden variant and no written-up instance layer (no fictional catalogue,
erratum, lot report or load case). The cables, the strain sensors and the leads have no geometry in the sources: they
are in no photo, and the cable net is recovered from document evidence only (section 7).

## 2. The task in five lines

* Story: a lab built one three-strut tensegrity rolling robot; a partner orders eight identical robots (24 struts);
  the robot was taken apart and its net cut away before one strut was photographed.
* Givens (`/home/user/`): 320 photos (`photos/photo_001.jpg` .. `photo_320.jpg`) and `photo_index.md`; the firmware,
  the host script, the tracking configuration and two calibration files (`software/`); the main board's fabrication
  data without any name (`board/fab_data.json`, `fab_data.md`); two sensor cut outlines (`sensors/`); eight vendor
  family pages (`vendor/VD-01` .. `VD-08`); `docs/conventions.md`; `schemas/` and `schema_check.py`.
* Deliverables (`/home/user/submission/`): `requirements.yaml`, `bom.json`, `cable_net.json` (and an optional,
  unscored `notes/`).
* Resources: 4 CPUs, 8 GB, 6 hours, no network, no desktop; a separate, static verifier (2 CPUs, 4 GB) that runs no
  submitted code.
* What earns the score (v1.1): an order from which the robot can be built and which costs what the exact order costs
  (0.67 of the total), and measuring on perspective photos (0.2125). Tracing signals across four files, the count, the
  requirement items and the derived quantities share the remaining 0.1175: they are read off the given text files.
  Visible contradictions are ruled on by the public order of trust. A fact that a given file states outright is never
  graded (the W0 list).

## 3. The grade and where each part is implemented

`reward = requirements_score = G0 x G1 x G2 x (0.25 C1 + 0.05 C2 + 0.03 C3 + 0.37 C4 + 0.30 C5)` (v1.1; v1.0 was
0.45 / 0.10 / 0.05 / 0.30 / 0.10); the privileged reference scores exactly 1.0 (`anchor.json`). Every knob is in
`grader_config.json` (the weights under `weights`); the graded list, the bands and the tiers are in `gt/freeze.json`,
which overrides the ledger's drafts.

| part | weight | what it checks | where |
|---|---|---|---|
| G0 structure | gate | the three files are present as regular files of at most 2 000 000 bytes, parse (YAML / JSON within the node, nesting, alias and integer-length budgets), validate against their schemas (names of the vocabulary and unit table, order size 8, at most 3 source ids) | `reqgrade/gates.py` running `reqgrade/schema_check.py`, the byte copy of the agent's public check; limits in `grader_config.json` `limits` |
| G1 submission integrity | gate | nothing of the submission written at or after the upload of the verify stage; the submission root is a directory that was not replaced, never a link | `verify.py` `_early_lockdown` (the lockdown helpers are copied from the published sister task, see the file's comment) |
| G2 no hedging | gate | no second record for a motor, sensor or controller index or for a strut's spare input; no quantity twice with the same comparator class; no derived quantity twice; at most 60 requirement items, 20 derived, 60 BOM lines | `reqgrade/schema_check.py` |
| C1 `net_and_geometry_replication` | 0.25 | `cable_net.json`: topology 0.10, key dimensions 0.85, counts 0.05 (below) | `reqgrade/c1.py`; `grader_config.json` `c1` (`c1.groups`, `c1.topology_blocks`, `c1.tier_weights`; the bands in `gt/freeze.json` `bands`) |
| C2 `requirements_sheet` | 0.05 | the non-derived items of `requirements.yaml` against 8 reference items: harmonic mean of tier-weighted recall and of precision over every submitted item | `reqgrade/c2.py`; `c2` (`c2.source_class_miss_factor`) |
| C3 `derived_quantities` | 0.03 | 8 closed-form quantities (formulas public in `conventions.md`, inputs not): tier-weighted mean of band credit | `reqgrade/c3.py`; `c3` |
| C4 `bom_buildability` | 0.37 | kit acceptance: how many corrections the order `bom.json` needs before the robot can be built from it, against the 35 parts mounted on the robot (27 bought, 8 made): `max(0, 1 - corrections / 2)` | `reqgrade/match.py`, `reqgrade/c4.py`; `c4` (`c4.kit.zero_at_corrections`, `c4.neutral_rules`, `c4.normalise`, `c4.discriminating`; accepted values and tolerances in `gt/freeze.json`) |
| C5 `bom_procurement` | 0.30 | the agent's order for 8 robots priced with the designers' own cost function against the exact order | `reqgrade/match.py`, `reqgrade/cost.py`, `reqgrade/c5.py`; `c5` (`c5.band`) |

Details a maintainer needs when reading a verdict:

* **C1 topology** is scored in four blocks, each the mean over its records; a missing record earns nothing in its
  block. `sensor_harness` 0.40 (9 sensors and 3 spare inputs: the `connector_pad`), `motor_harness` 0.15 (6 cables:
  the `motor_connector`), `ends` 0.35 (6 cables and 9 sensors: both nodes right, strut colour and end U / D, **and**
  the harness field of the same record right), `controllers` 0.10 (3 records: the strut colour). Shares of the total:
  0.0100 / 0.00375 / 0.00875 / 0.0025 (the topology group is 0.025 of the total). The designators and pads are the
  neutral ones of `fab_data.json`; a pad is compared as a number when it is one.
* **C1 dimensions**: 10 lengths in metres, each scored by its own frozen band a / b in millimetres, tier-weighted
  3 : 2 : 1 (W3 / W2 / W1). The full-credit points were measured, not chosen: a = max(3 px, 2 sigma) of the ledger's
  measuring route on the delivered photos (0.087-0.343 mm; the maintainers' precision budget). The no-credit point is
  b = 4 a since v1.1 (v1.0: 6.5 a): 0.348-0.720 mm for the five dimensions of one small part, 0.840-1.372 mm for the
  five strut-scale lengths (per row: 1.004, 1.032, 0.980, 0.520, 1.372, 0.716, 0.616, 0.348, 0.720, 0.840 mm for
  `C1D-01` .. `C1D-04`, `C1D-06` .. `C1D-11`). The full-credit points did not move: a measurement within the 2 sigma
  of the budget still scores 1; what changed is that a rough estimate earns nothing. Share of the total 0.2125.
  **C1 counts**: 1 count, exact (0.0125).
* **C2**: an item matches a reference item when the quantity is the same, the comparator is of the same class and the
  bound is in the band (integers and enumerations exact; the gearbox ratio absolute 0.05 / 0.30; other numbers
  relative 1 % / 5 %); a match none of whose source ids is of an accepted class keeps half its credit. Precision
  counts every submitted non-derived item: a W0 name with a right value is left out, with a wrong value it counts as
  wrong; a graded name with a wrong value or comparator and a name that no given material fixes count as wrong.
* **C3**: integers exact; relative 1 % / 5 %; the encoder count over the command range 0.1 % / 0.3 % (so that the
  software's nominal gear ratio scores nothing); a missing entry scores 0.
* **C4, the matcher** (`reqgrade/match.py`, shared with C5; unchanged in v1.1 but for the four points marked):
  lines of an out-of-scope category or on the neutral list leave first; lines of the same part merge and add their
  quantities; the rest is assigned one-to-one to the truth lines. A truth line is recognised by vendor and part number
  (the bought boards), by the paper tags of its instances in the photos (made parts) or by the discriminating
  attributes of its category within tolerance (everything else; numeric tolerances frozen from the photo budget and
  unchanged in v1.1: screw length 0.102 mm, rod diameter 0.083 mm, rod length 1.39 mm, spacer length 0.087 mm; gear
  ratio within 1). v1.1: (1) **fold**: a line that takes part, was assigned to no truth line and is recognised by a
  truth line that another line already holds is merged into that line and the quantities are added; it is not a
  correction. This keeps the promise of `conventions.md` that splitting a part over several lines does no harm (the
  same screw written with two lengths inside the tolerance, a board under both accepted part numbers). (2) A quantity
  is right when it makes the same order as the truth's, `robots x (3 x qty_per_strut + qty_per_robot_extra)` pieces:
  the same order written per robot is no correction. (3) Three **accepted values** (`gt/freeze.json`
  `c4_accepted_values`): the two XH wire-to-board lines (`C4-17`, `C4-21`) also accept series EH and the PH line
  (`C4-18`) also accepts PA, because the vendor page lists each pair at one pitch without an outline that tells them
  apart and the photos were checked for XH against PH only. (4) **Series names** are normalised further
  (`c4.normalise`: the suffix words `style` and `type`, a trailing pitch, size or position count such as
  `JST SH-style`, `SH 1.0mm`, `SH 6-pin`; a thread class after UNC / UNF such as `4-40 UNC-2B`; more package aliases
  such as `SOIC-8N`); a wrong series or thread stays wrong.
* **C4, the score** (`reqgrade/c4.py`; v1.1, kit acceptance): corrections = (truth lines recognised with a wrong
  quantity) + the sum over the categories of max(truth lines of the category that no line is recognised as, lines of
  the category that take part and are recognised as no truth line). `C4 = max(0, 1 - corrections / N)` with
  N = `c4.kit.zero_at_corrections` = 2 (a missing key is a task error, there is no default): one correction leaves
  0.5, two or more leave 0. Within one category a line with a wrong field and the part it stands for are one
  correction; a part written under another category is two (one in each category); a line added on speculation is
  one. No line earns partial credit. The per-line F0.5 of v1.0 (`c4_f_beta_score`, with `c4.f_beta` and
  `c4.line_scores`) and `c4_kit_complete` are reported, never scored; the line scores still order the assignment.
  New metrics: `c4_corrections`, `c4_wrong_quantity_lines`, `c4_unrecognised_truth_lines`, `c4_unrecognised_lines`,
  `c4_lines_folded`.
* **C5**: money distance D = (the sum over the 27 priced truth lines of |cost of the agent's units - cost of the exact
  units|, a line nothing is matched to counting as bought 0, plus every unrecognised line that takes part and whose
  category has a priced truth line, priced as the cheapest truth line of that category at the agent's quantity) /
  the exact order's cost (3 747.804 USD for 8 robots). `C5 = 1 - clip((D - 0.005) / 0.045, 0, 1)`: full credit up to
  0.5 %, none from 5 %. The algorithm and the band are those of v1.0; it reads the matcher's folded lines. In practice
  a near-binary block: any wrong or missing board zeroes it. C4 and C5 judge the same order, so one wrong judgement
  costs in both (a cheap line with a wrong count: reward 0.815; any board or the gearmotor wrong: 0.515). The price
  table is never given to the agent.
* The verdict's `metrics` carry the criteria, the gates and the statistics (`c1_topology_sensor_harness`,
  `c1_dimensions_full`, `c2_precision`, `c4_corrections`, `c5_distance`, ...); the `diagnostic` text has one clause
  per criterion (for C4: the number of corrections, its three parts and their categories). The per-record, per-item
  and per-line detail is not in the verdict; it is the `detail` that `reqgrade.c1.score`, `c2.score` and `c3.score`
  return and the `match` of `reqgrade.c4.evaluate` (C4 and C5). A read-out script of the maintainers' build prints it.

## 4. The hidden standard, file by file

Hashes are those of the files in this folder: v1.1 of the scoring, frozen on 2026-11-17, and one text-only amendment of
2026-11-18 that reworded notes, comments and doc strings for publication (no value, key or code changed; the three
pins of `gt/freeze.json` were renewed for the reworded truth files, and the recorded verdicts re-grade identically).
`load.py` checks the three pins of `gt/freeze.json` (ledger, neutral lists, price table) on every verify run and
raises a task error when one differs. The values in the three truth files are those of v1.0. v1.1 changed `gt/freeze.json`,
`grader_config.json`, `reqgrade/c4.py` and `reqgrade/match.py`, the doc strings of `verify.py` and `reqgrade/c5.py`,
and `solution/answer/bom.json` (one line fewer); `gt/freeze.json` `v1_0` keeps the sha256 of the two v1.0 files it
replaced. The ledger's own row counts below are its drafts; the freeze grades 83 rows (C1T 21, C1D 10, C1N 1, C2 8,
C3 8, C4 35).

| file | sha256 | what it is |
|---|---|---|
| `tests/gt/ledger.json` | `55ad0b2e7b5bdd4e91ee38a8bff7a2aeca12cf4ef18b4259b50fab0ce2301d56` | the truth ledger: 118 rows (graded C1T 21, C1D 11, C1N 2, C2 8, C3 8, C4 36; 7 neutral BOM rows; 25 W0 rows), each with its value, real sources (file, sha256, how read), contradictions and ruling, route in the givens, tier |
| `tests/gt/price_table.json` | `edfc721637c3a213d04d212eb15b9c74b06ef6b0db477fb8678eb78ed481ac09` | the designers' purchase table: per row the pack size, unit price, quantity-discount tiers, quantity per strut and the priced-scope flag; the totals for 1 / 2 / 4 / 8 / 10 robots |
| `tests/gt/neutral_lists.json` | `682bb7bfb446aca082b48c83910ca2d370c0357adad5b8a68c41f2df3b60634e` | the BOM neutral list (7 rows), the out-of-scope categories, the W0 list (25 quantities with their truths) and the vocabulary names that no given material fixes |
| `tests/gt/freeze.json` | `a4287637b22cbe33b39634941dd1ddb94aa9cda6c2c1623517850bebff69b22c` | the freeze (v1.1): the graded list (83 rows), the three ungraded rows and why, the measured bands (zero point 4 a), tiers, route classes and the route audit with its one-step route floor, the matching attributes and tolerances, extra accepted part numbers and accepted series, the sha256 pins of the three files above, the hashes of v1.0 |
| `tests/grader_config.json` | `62dcf6bb2afbebd3cd8d2ee074d0e926668d464ad1009fcabcad46cd01040a9b` | every knob (v1.1): weights, limits, C1 groups and blocks, tier weights, band rules, the C4 category / attribute maps, normalisation, neutral rules, the kit rule's zero point, the reported F-beta, the C5 band |
| `tests/anchor.json` | `e1c99f7ef34b134b48e84336ad7ce3e71ca38f4e47cf24af3069f387e5826b2b` | the reward anchor: value = full_at = 1.0 (the privileged reference's own score) |
| `tests/given_hashes.json` | `b20b2a94cce5855203dc86696475b56db4210e6e0ce051bcbb769caf2694cbc5` | the maintainers' pin of the givens (26 payload files and 320 photos, with sizes); not read at verify time |
| `tests/verify.py` | `ec2dc5525c7f26f80df61d5584d9076bfac589eabcbeec03bb0e0ac5e6db1fae` | the grader's entry: verify-start lockdown and gate G1, snapshot of the three deliverables, gates G0 / G2, the five criteria, the verdict |
| `tests/test.sh` | `a50e8cbf964e9214a94dc292911f361d59c75882eca25bea1df93cb84ab611ef` | the verify stage's shell entry: closes the staged folder, pins the BLAS threads to 1, starts verify.py |
| `tests/reqgrade/__init__.py` | `fe600eb4c8a874a363590ea661b99b48204c5229ac1e91402ead8f3028d263fb` | package note |
| `tests/reqgrade/load.py` | `ffe67b9a82d713f0483d07994db0129023e01d1b5b9b9c50a651b83f4c8bda6f` | reads the submission (through schema_check), the truth and the settings; checks freeze.json's sha256 pins on every run |
| `tests/reqgrade/gates.py` | `d5a7dfd4df82bf9b680b3f5976de400ff352042cae6ee556b4fb6e6164daeaac` | gates G0 (structure) and G2 (no hedging): runs the copy of the public check |
| `tests/reqgrade/schema_check.py` | `38176ad28b51a0df5972c09d221542047c6539f64a605dac48f29b3ba4245e6c` | byte copy of the public `schema_check.py` of the payload |
| `tests/reqgrade/bands.py` | `e5c4cb6ab79621984a76d8f73b3e4118a041bed9c9425ce1a71e866bd44f7ee2` | band functions and unit conversion |
| `tests/reqgrade/c1.py` | `5e414042d5c8a7e4d293305c4d5126539167e762878d420773df752eb8725a71` | C1 net_and_geometry_replication |
| `tests/reqgrade/c2.py` | `5f72c9135c5311078f68db9f363cfab87f5fe13dfab4868185bc227e8bb12d93` | C2 requirements_sheet |
| `tests/reqgrade/c3.py` | `35c0825eff71799cac671dc4607f79d47bdb5d5eefaa84377b2bc413e47ab69c` | C3 derived_quantities |
| `tests/reqgrade/match.py` | `99ac3fd114f2435f45b237f36c188ab7896a0a471e12906de72ea41487747dc3` | the BOM matcher shared by C4 and C5 (truth lines, merging, recognition, assignment, fold) |
| `tests/reqgrade/c4.py` | `509efd27304c922e95819b233f6c92f4b4f2114cc50806147429608ad913d520` | C4 bom_buildability (kit acceptance over the matcher: corrections; the per-line F-beta reported only) |
| `tests/reqgrade/c5.py` | `1e981b97f9ab0a6f605226c24ceb828c1b9742fc7cc066eaf6aaa84ed5245f25` | C5 bom_procurement (money distance over the matcher) |
| `tests/reqgrade/cost.py` | `e0fb4ee1df9001236f732822bbffd867f670c6e57969bdf1867421283156c87f` | the designers' cost function, driven by the price table |
| `tests/reqgrade/schemas/bom.schema.json` | `b5bf04841bb455ecf445067f886288eb1c8a4adb5eab9b7c470e0817ba101429` | byte copy of the public schema |
| `tests/reqgrade/schemas/cable_net.schema.json` | `e1e4b90f3ce82707ae090b0e801a5585a95b1fe027f8de740f3974fd75c48922` | byte copy of the public schema |
| `tests/reqgrade/schemas/requirements.schema.json` | `596f2a114666571261cac710f884837b86ddef6a1e841eb2c9063058025e49b5` | byte copy of the public schema |
| `tests/reqgrade/schemas/categories.json` | `76f7ac92c6e38f9fbf1797c037f3c88c62a5c1aa8b813d750caffc6ac2257e43` | byte copy of the public BOM categories |
| `tests/reqgrade/schemas/quantities.json` | `03097ff4ed9a4cc66d5043b84bf5fdbe80bf95e15c465850c6e6878b13a931fd` | byte copy of the public vocabulary (66 names) |
| `tests/reqgrade/schemas/units.json` | `5871cd5d111dba376f4990fcc6afade571dd3738a70f0d8dd64b9fa3cdf833b1` | byte copy of the public unit table |
| `tests/reqgrade/schemas/contract/ledger_map.json` | `554f24ba9aef5d90b8827c4d232fa1706b240472098019e425dd8da0e23283f4` | HIDDEN contract table: ledger keys and categories -> the public names; the two ruled-out rows |
| `tests/reqgrade/schemas/contract/quantity_classes.json` | `bc66df81f62ac312894db6d707905ef09ef78f4b8a278e741eacea1ef227586b` | HIDDEN contract table: which vocabulary names are graded (8), W0 (27) or fixed by no given material (31); tolerance classes |
| `tests/robotics_grader/__init__.py` | `d75f3c6293445af3277b05fe00df38ef7378645651ff3cfbad7f8bcfe1f56604` | the benchmark's shared verify kit (byte-identical to `shared/robotics_grader/`) |
| `tests/robotics_grader/scoring.py` | `9353682eb3536df264b3cb5ec3bf2be24a410f88af5b20cf86cb31796930a83b` | kit |
| `tests/robotics_grader/seeds.py` | `b795f63404d14f0c90ca0098582e80afabd251396b1f4d6ababf4e7f6037b5e1` | kit |
| `tests/robotics_grader/selftest.py` | `7a4be7513643147c2f91f866a24b6a5253a3203180b4724ca54a188fd0c2027f` | kit |
| `tests/robotics_grader/solver.py` | `288a8c1f3a1dfd61d3b2c9240c40f095411c6129af867b00fc6a0d8a77ab2d84` | kit |
| `tests/robotics_grader/stage.py` | `a97eefbc9ee2f4c67b26eecdfc76d1ac84cf4615a2f8ab2fc63e75c581eb0c2f` | kit |
| `solution/solve.sh` | `eb99373f072aa6a57ca73213cc7090888a7cd5fa6171e0af1d1e945e2fcdc9c5` | the privileged reference's script: copies `answer/` into the submission folder |
| `solution/answer/requirements.yaml` | `0828915153f920b1add3c0ed1e1951efa2246c40fbb9ceeb7cc5a420daaa7802` | reference: 8 requirement items and 8 derived quantities, transcribed from the ledger |
| `solution/answer/bom.json` | `4d1534f286668d2b7fa962e7b6a6512a161e247c58d74b18c3d79da19f3e9330` | reference: 35 lines for 8 robots (the graded rows), transcribed from the ledger |
| `solution/answer/cable_net.json` | `d218ae819b51759a94689f367937b87d7a0f749733b943258568dc522c2808e9` | reference: 3 controllers, 6 cables, 9 sensors, 3 spare inputs, 10 dimensions, 1 count |

### 4.1 Which of the designers' files each truth file is read from

The upstream locations, commits and licences of these files are in `NOTICE.md`. None of them is in this repository
or in the image. sha256 in full, as recorded in each ledger row's `sources`; those cited for the CAD sibling task's
`tests/reference/families.json` and `features.json` are the hashes of the two files when the ledger was built (later
text-only amendments of those files changed their hashes, not a value).

| key in the ledger | the designers' (or vendor's) file | sha256 | what is read from it |
|---|---|---|---|
| `mesh_ply` | the whole-strut mesh `Will_strut_scaled_centered.ply` | `8dc4caff1efa26468660f30d5e2d91f7d927d6b287480ecd3b06b351f88a4614` | the 10 key dimensions and the count; the fastener, spacer and rod sizes; what is mounted and how many (through the pieces and part families the CAD sibling task split from it without filling any hole: its `tests/reference/families.json` `a91fcf80...9b92`, `features.json`, `parts/*.ply`) |
| `board_brd`, `board_sch` | the main board's EAGLE board and schematic files (`v3 Tensegrity Motherboard.brd` / `.sch`) | `9b82a9fbc8f01d505b7e2dcada6e47cdb6cbc13acdec08d7c172d485822b156c`, `d7ca816ab816ad368820fb9625beff33ab618fd56985fd5f3b381a00d539457d` | nets, footprints, pad positions, connector positions; which board end a connector is on; the BOM rows soldered on the board |
| `firmware` | the controller firmware (`onboard.ino.ino`) | `b19b59eb75c6cdbbc8cabdd78a21de4c9f8f66fe62679e892010c93cea9f5ed2` | motor table, driver and encoder pins, electrode re-map, telemetry order, which IMU is read |
| `host` | the host control script (`run_tensegrity.py`) | `14a53af9e3c2e432b4df8b250caec9858d5aa03a15f3a81011bf0c9e63ae2ed7` | per-controller unpacking, constants (inputs of the derived quantities), the calibration file in use; the withheld description block (lines 308-316) as a cross-check |
| `tracking_config` | the tracking configuration (`data_cfg.json`) | `5e6db514515d42610f6ed6865a7403d39478e08c62cd7ffd64f896172584a393` | the nine node pairs, the node colours; the two rod numbers that contradict the object |
| `calibration_new` | `new_calibration.json` | `5ca8e4ab7cd217bc4d35a0f85235678412fd5512693f61b3a7ad0dff57fc8e5a` | the slopes of the calibration the script loads |
| `dxf_short`, `dxf_long` | the cut outlines of one sheet of short and of long strain sensors (`sensors.DXF`, twice) | `e8cbb80050fecdc968319c99dd3b11d7b9e0ce867cbea6616d4d67ec242149bb`, `a3033959c3db3ccb34f64b0e45380c3469bfaca5317717b38ec10a0f5414d58a` | sensors per sheet, hole pitch, planform (W0 inputs of the derived quantities) |
| `bom_xlsx` | the designers' bill-of-materials workbook (`BOM.xlsx`) | `3dea5cc931d8758edf7f7f91b350ff498b703577aa277af90cf918284448c613` | the C4 rows and quantities per strut; pack sizes, unit prices and discount tiers (the price table, formulas evaluated and checked against the workbook's cached values); the out-of-scope supplies |
| `vendor_motor_page` | the gearmotor vendor's product page (capture) | `bd881913ff9ef94ab073eba9ede3b0d619efa1ab1c770ef65d211b87253d171b` | the exact gear ratio, the pin table of the encoder connector |
| `vendor_nano_cad_zip` | the controller board vendor's CAD package | `56477d319fa577bf70df52f949b6b4ea7fa4e128cdc9f1a0ba256a319f50fb16` | which end of the controller board carries the USB receptacle (the definition of end U) |

Cross-checks kept on the hidden side only (they decide no value; they agree with the ledger): the assembly guide
(colour rule of the net, where the long and short sensors plug in; sha256 `912c44cc...e49e`), the vendors' own board
files of the capacitive sensing board and the motor driver board and the controller core's pin table (a trace
self-check of the maintainers' build re-walks the cable net on the neutral export: 51 of 51 items equal), and
four purchase invoices (nine ledger rows carry an `invoice_check`: seller, part number, unit price; nothing else of
an invoice is recorded anywhere).

`gt/freeze.json` adds what was measured on the delivered photo set (the maintainers' photo metrics and photo
checks), eleven decisions on the vendor fact pack (`vendor_rulings`: the facts behind the vendor pages, each read from
a public page), the accepted values of v1.1 and the results of the precision budget, the answer-line scan and the
route audit (`inputs` pins these three by sha256). Route classes (`route_classes`; `route` in each ledger row):
R0 = the value is written in a given file; R1 = one step from a given file; R2 = traced across several files,
judged qualitatively on the photos, or resting on a judgement without direct evidence; R3 = measured on the
photos, or a visible contradiction settled by the order of trust.

The trust order inside the ledger (for whoever rebuilds it; the agent sees the two-sentence order of
`conventions.md`): mesh > board file > vendor data of an identified part > firmware, host script and tracking
configuration > assembly guide > bill of materials > paper and web. "Not drawn" is not a statement: what the mesh
lacks (resistors, the second line-guide ring, the cables) does not count as absent. That is the ledger's reading. The
agent has only the public order, in which the photographed object comes first, and a careful reader of the photos
would judge the resistors not fitted: so v1.1 takes their row out of the graded list instead of grading either
reading (section 5).

### 4.2 What reaches the agent and what is withheld

Given, cropped but never rewritten (`NOTICE.md` lists all ten edits): the firmware (the
two network string constants and one given name in a comment replaced by placeholders), the host script (the
nine-line description block that says which strut carries which sensors, motors and IMU deleted, nothing else), the
tracking configuration and the two calibration files (byte copies), the two cut outlines (two watermark text entities
and the "last saved by" account name removed from each). The board is given as neutral fabrication data exported
item by item from the board file: outline, 23 placements `X01`..`X23` with their pads, 36 nets `N001`..`N036` as
connectivity; no part number, package or library name, pin name, net name, designator or board text
(an item-by-item export of the maintainers' build; the map back to the designators stays with the maintainers).
Withheld, because they are the hidden standard or name it: the bill-of-materials workbook, the assembly guide, the
papers and web pages, the simulation models, every name in the board file, the description block of the host script,
the designers' real photos, every file of the maintainers' download of the designers' CAD package, and every grader module
except `schema_check.py` and the public schemas.

## 5. The neutral list and the W0 list (`gt/neutral_lists.json`)

| list | content | effect of writing it |
|---|---|---|
| BOM neutral list (`bom_neutral`, 7 rows; rules in `grader_config.json` `c4.neutral_rules`) | parts that are on the robot but that the givens cannot settle honestly: the battery extension lead (no route in the givens), the bought line-guide ring (the workbook says 2 per strut, the mesh shows 1), the short and long strain sensors (their counts are stated in two given files), the header strips that ship with the bought boards, a 3-pin block and one unidentified small piece of the mesh that are not in the workbook | the line leaves before matching: not rewarded, not counted as a correction of C4, not priced in C5. One exception: the unidentified piece (`NEU-07`) has no attribute a rule could recognise, so a line written for it takes part like any other line |
| two neutral rules of v1.1 (`c4.neutral_rules` only; no row in `gt/neutral_lists.json`, which is unchanged) | `NEU-08`: every line of category `resistor`. The main board's chip resistors (ledger row `C4-31`) are not on the photographed board: in the photos the board area between the two 15-position socket rows is empty while the transistor beside it is drawn, and the mesh draws no resistor. `NEU-09`: every `cable_assembly` line whose `positions` is absent or not 6, that is any ready-made lead other than the motor lead. No lead is in any photo, and whether the sensor and battery leads were bought ready-made or crimped in the lab cannot be settled from the givens. The 6-position motor lead (`C4-27`) stays graded: the gearmotor page gives its connector and the main board has the two matching headers | the same. The instruction says in general terms that the order has to cover ready-made cable assemblies that connect bought parts and that leads crimped in the lab are not graded; it names no line |
| BOM out of scope (`bom_out_of_scope`; public in `schemas/categories.json`) | wire, tape, heat shrink, solder, print and casting material, chargers, tools, consumables | the same |
| requirements W0 list (`requirements_w0`, 25 quantities) | quantities whose truth is in the ledger but which a given file states outright (port, gains, tolerances, command limits, the software's constants, field and channel counts, sensors per robot and per sheet, hole pitches, planforms, ...): inputs, never graded | right value: left out of C2's precision; wrong value: counts as wrong |
| names with no source (`no_source_quantity_candidates`; the classes are in the hidden `reqgrade/schemas/contract/quantity_classes.json`) | vocabulary names that no given material fixes (mass, endurance, speed, payload, slope, cable stiffness, ...) | counts as wrong in C2's precision |

The public vocabulary (`schemas/quantities.json`, 66 names) mixes the three classes without marking them. Three
ledger rows are ruled out of the graded list (`gt/freeze.json` `ungraded`, with the reasons): `C1D-05`
(`line_exit_offset`: the mesh gives no hole axis to measure), `C1N-02` (`line_exits_per_motor_module`: the mesh draws
the ring at one end only) and, since v1.1, `C4-31` (the chip resistors, rule `NEU-08` above). The graded BOM is 35
rows; the privileged reference has 35 lines and no resistor line.

## 6. Rebuilding the hidden standard

Everything is regenerated by the maintainers' build from the designers' files (neither is in this repository; where
to get the designers' files: `NOTICE.md`); no value is typed by hand. Python 3.12 with the task image's pins for the grading stack
(PyYAML 6.0.3 with libyaml, jsonschema 4.26.0, numpy 2.5.3, scipy 1.18.1); run with `-I -B`.

| step | how | writes | check |
|---|---|---|---|
| neutral board data | item-by-item export of the board file | payload `board/fab_data.json`, a hidden map back to the designators | a scan of the 347 names and texts of the board files: 0 hits |
| given software, calibrations, cut outlines | crop of the designers' files | payload `software/`, `sensors/`, a manifest of the ten edits | byte-exact except the listed edits; AST of the host script equal but for the deleted block |
| the ledger | read from the designers' and vendors' files | `tests/gt/ledger.json`, `price_table.json`, `neutral_lists.json` | 16 checks: built twice and byte-identical, every graded row sourced, the two totals, the trace self-check, the route audit |
| vendor fact pack | 147 public captures | the facts behind the payload's vendor pages | every fact against its capture |
| contract | from the ledger | payload `docs/conventions.md`, `schemas/`, the two hidden contract tables | a contract check |
| photos | the CAD sibling task's photo set (287 JPEG, byte for byte) plus 33 photos rendered with the same scene | the photo set of the asset dataset (not in git; manifest sha256 `45cd6e39...4788`) | every graded count, dimension and attribute against the photos |
| budget, scan, audit, freeze | precision budget, answer-line scan, route audit, then the freeze, in this order (the freeze refuses an audit that was run on another configuration) | `tests/gt/freeze.json`, `tests/grader_config.json` | gate of v1.1: no graded row answered by one line of a given file (R0 = 0) and the one-step route floor at most 0.10: the reference cut down to the rows that are one step from a given file scores 0.0437 through `verify.py` (the same file scored 0.3100 under v1.0). The row-weighted share one step away, 0.2596, is reported only (v1.0 gated on it: at most 0.30, measured 0.2167) |
| agent-visible text | generated from the frozen configuration | `instruction.md`, payload `photo_index.md`, `board/fab_data.md`, `vendor/VD-0*.md` | its 23 built-in checks; every number of the scoring section read from the frozen config. v1.1 regenerated the scoring section of `instruction.md` only; the ten payload texts are the bytes of v1.0 |
| privileged reference | transcribed from the ledger | `solution/answer/` | the ledger transcribed: graded rows only, no W0 item, no neutral line |
| image lists | from the payload and the photo set | `environment/assets.sha256`, `tests/given_hashes.json`, `environment/Dockerfile` | 9 provenance checks |

After the freeze only two kinds of change are allowed: the repair of a grader defect (rescored offline and marked as
a rescore; a score recorded before the repair stays as recorded) and a pre-registered lever (a new configuration
needs a new episode). v1.1 is a recorded departure from that rule, not an instance of it: it was decided after an
episode that does not count and with that submission in view (section 9, last item). Its levers and their triggers
were registered before the episode that is read: the kit rule's zero point is the one key `c4.kit.zero_at_corrections`;
the zero points of the dimensions (b = 4 a) and the weights are set in the freeze step of the build, and the route
audit is run again after a change of the weights.

## 7. The cable net: how it is recoverable, and the premise it rests on

No cable, sensor or lead has geometry, and none is drawn. The net follows from six steps over the givens: the
tracking configuration gives the nine node pairs and the node colours; the host script says which sensors are short
and long and which motor shortens which sensor's span; the firmware gives each controller's two motors, their driver
channels and encoder pins and the electrode re-map; the fabrication data together with the vendors' pin tables gives
the pad of each electrode, the connector of each motor and the board end of each connector; and the last step joins
them under one premise: **a lead is plugged in at the board end nearest to its own strut end**. Under that premise
the solution is unique (controller 0 / 1 / 2 = red / green / blue; the odd nodes are at end D, the end away from the
USB receptacle); without it two colourings remain. The premise has no direct evidence in the givens (the leads are in
no source geometry; their lengths are only in withheld files); it is supported by the design itself (each board end
carries one sensor connector and one motor connector) and by six independent files of the designers on the hidden
side. The score treats it accordingly: the `ends` block is 0.00875 of the total (0.071 under v1.0) and is earned only
on records whose harness field is right. The maintainers' fairness list records, for each graded row, the given
files, photos and vendor families that carry its evidence, and the twelve facts that support the premise.

## 8. Re-running the grader's own QC

* **Oracle and untouched** (`ale validate`): expected untouched 0.0, oracle 1.0 with every criterion exactly 1.0
  (v1.0: twice, engine 0.1.1). Every change under `tests/` or `solution/` changes the content digest, so the tree is
  validated again on its final bytes before an episode. Until the photo set is in the asset dataset,
  `environment/assets.sha256` carries a placeholder revision (40 zeros) and a validation copy that points the two
  header lines at a local mirror of the photo set is what builds; after the upload the revision is pinned and the
  public tree is validated once more.
* **Host QC** (the maintainers' case generator, seed 20261117, and their QC runner): 279 deterministic
  edits of the reference, each scored twice through `verify.py` on a copy of `tests/` (558 runs). Expected on v1.1:
  * the reference 1.0 with each of the five criteria exactly 1.0; not submitted, the empty skeleton, 10 G0 and 6 G2
    counter-examples 0;
  * 30 benign rewrites exactly 1.0: the 14 of v1.0; EH written for XH (twice) and PA for PH; `JST SH-style`; the
    resistor line written, or written with another package; an extra 4-position lead; three cases of one part split
    over two lines that the fold joins (screws at two lengths inside the tolerance, the spacer at two lengths inside
    the tolerance, the controller board under both accepted part numbers); five more writing forms (`JST SH 1.0mm`,
    `SH 6-pin`, `JST-PH 2.0`, `4-40 UNC-2B`, `SOIC-8N`) and one line written per robot;
  * each of the 191 mutants loses credit only in its own criterion (largest difference from the independent
    formula 1.24e-14, and the formula's correction count equals `c4_corrections` case by case); every one of the 104
    single-line BOM mutants has C4 exactly 0.5, plus its C5 money;
  * the 11 rows of the maintainers' BOM table, computed apart from the grader, within 5e-5, with the same correction counts
    (0, 1, 1, 8, 1, 2, 1, 2, 10, 13, 35); one and two corrections on unpriced lines 0.815 and 0.630; the
    counter-example to the neutral rule `NEU-09` (an extra 6-position lead of another series is not neutral: one
    correction, and it is priced) 0.690374;
  * 10 half-finished submissions above 0 and reduced only where they are incomplete;
  * the four BOM hedges below the reference, with C4 not above the matching single wrong answer and a break-even
    confidence p* = (hedge - counterpart) / (reference - counterpart) of at most 0.5; the five other hedges no
    better than their counterpart by more than 0.01 (the largest is the count hedge of C2, +0.0033);
  * two full passes give byte-identical results (the maintainers keep their sha256); the 263
    submissions of the v1.0 case set score the same with and without the fold; on the host the grader is not root, so
    the lockdown is skipped.
  The maintainers' unit tests of the grading modules pass on v1.1 (216 and 123 tests).
* **Through the real verifier** (one container
  of the task image per submission, root, 2 CPUs, 4 GB, no network, `tests/test.sh` as it is): the expected values
  are those of the host QC. On v1.0, 26 cases agreed with the host in rewards, diagnostic and every scoring metric
  (only the six lockdown metrics differ); the 27th, the G1 case (a deliverable rewritten after the stage upload),
  exists only as root and scores 0. 30 adversarial submissions (sparse files, lone
  surrogates, alias and merge-key bombs, alias cycles, long integers, long lists, deep nesting, FIFOs, links, late
  writes) all ended as a recorded 0 within 0.5 s and 100 MB. Verify took 0.15-0.64 s; the container peaked at 526 MB
  with the image's desktop session. The v1.1 run of this step uses the maintainers' case generator as it is.
* **Determinism across CPUs** (measured on v1.0 and not repeated on v1.1, whose changes to the grader are integer
  counts and one division): the same image on a second CPU generation gave the same rewards, diagnostics and scoring
  metrics on 5 cases. A verdict file written as root is never byte-identical between two runs: it records `t_ref` (a
  time stamp) and `agent_processes_killed` (how many processes of the image's desktop session the lockdown stopped).
* **Leak scans** (the maintainers' scanner and its self-test): names (0 hits over every agent-visible byte, the photos'
  metadata and every path of the image), withheld answer sentences (0), every hidden number against every
  agent-visible number (each hit carries a waiver with its public meaning), vendor pages against the fact pack, and
  the intersection of `tests/` and `solution/` with the image, the agent home, the payload and the photo set (empty
  but for `schema_check.py` and the six public schemas). Re-run them whenever an agent-visible byte changes.
* Repository tests: `tests/test_task_layout_tensegrity_requirements_replica_from_photos.py` (the Harbor layout, the
  asset list, nothing of `tests/` or `solution/` in the build context, the payload or the photo set) and
  `tests/test_verify_survivor_tensegrity_requirements_replica_from_photos.py`.

## 9. Known limits

* **No human has done this task with these materials**; one model episode has (the frontier check of the pull
  request that added the task). The solvability evidence is the privileged reference at 1.0, the degenerate submissions at 0 and the grader's QC.
  That a skilled engineer finishes in six hours is an estimate. The difficulty is measured by model episodes, not
  predicted; no pilot, probe or baseline ladder was run.
* **The nearest-end premise** of section 7 is a judgement an engineer would make, not a reading. If an episode traces
  every harness right and gets every end wrong, the premise is what failed, and the pre-registered relaxation is to
  give the guide's one-line colour rule.
* The designers' mesh carries raised lettering on two bought boards (the controller board's logo and chip marking,
  the driver board's name and pin names). The views in which it is legible at 5x were withheld; the chip marking is
  half legible in a few 2x photos. It leads at most to the family of the board, which the firmware's libraries give
  as well; it does not give the part number.
* Four attributes could not be shown at the required resolution (the outline and the pin count of the controller
  board and of the motor driver board, after the views with legible lettering were withheld). Both boards are
  recognised by vendor and part number, so no matching condition depends on those views.
* Accepted alternatives are frozen: the controller board with or without headers, two part numbers of the motor
  driver board (the workbook links one, the designers' invoice names the other) and, since v1.1, series EH for the
  two XH lines and PA for the PH line (section 3). The instruction tells the agent in general terms that a board is
  accepted with and without header strips; it does not mention the series pairs, which can only turn an answer that
  would be wrong into a right one.
* **The hedge residual of C5** (C5 is near-binary by design and weighs 0.30 in v1.1). Listing several candidates for
  cheap parts, each at a small quantity, scores 0.1366 more than writing one candidate each that is wrong (the QC's
  fastener case: 0.4666 against 0.3300; the whole margin is C5 money, the residual of v1.0 was 0.0396). It gives up
  the kit criterion entirely (C4 = 0 in both), so it pays only when the agent's chance of writing the single
  candidates right is below 0.20 (break-even 0.204); with the true quantities on every candidate the hedge gains
  nothing. The other three BOM hedges of the QC gain nothing or lose.
* **The spacer's length** has the thinnest margin among the lines that the kit rule makes decisive. The length in
  the mesh, 11.002 mm (the workbook's part for this line is a metric spacer of 11 mm), is 0.11 mm from the nearest
  inch size, 7/16 in: 2.6 sigma on one 5x close-up, 4.4 sigma on the three close-ups together (sigma 0.043 mm; the
  matching tolerance 0.087 mm is 2 sigma), while the outer diameter separates the two by 6.7 sigma. The vendor page
  of fasteners lists this length in neither system, so the line rests on the measurement. 7/16 in is not accepted;
  on an otherwise exact order this line alone is worth 0.44 of the reward.
* The photo set is shared with the CAD sibling task `tensegrity_cad_design_from_photos` (287 of the 320 photos are
  its set); the two tasks' sandboxes have no network.
* The scale references, the bench and the room are generated or come from the project's licensed art; the scenario
  text calls the pictures a colleague's phone photos, while they are renders of the designers' mesh.
* `environment/assets.sha256` has a placeholder revision until the upload (section 8).
* **v1.1 was triggered by an episode that does not count.** The first episode on v1.0 was cut off by an
  infrastructure failure after about 35 minutes of work. By the pre-registered rules it is no reading and triggers no
  lever. The maintainers decided to harden the task before the one episode that is read. The structure of v1.1 (the
  kit rule, the five weights, the 4 a zero points) was therefore decided after that episode and with the rescoring of
  its unfinished submission in view: two registered levers were used although their triggers had not been met,
  and the weight of C5 was moved against the registered direction. What was taken from that submission line by line
  served only the fairness fixes, all of them relaxations (the resistor line and the other leads neutral, the series
  of one pitch accepted, the series writing forms); the kit rule, its zero point and the weights treat the 35 lines
  alike. Two thirds of the weight now sit on one order, so a single episode's reading spreads more than under v1.0.
  The amendment with the hashes, the levers and their triggers was dated and registered before the episode that is
  read.
