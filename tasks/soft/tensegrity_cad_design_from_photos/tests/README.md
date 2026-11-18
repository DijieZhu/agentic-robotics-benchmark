# tensegrity_cad_design_from_photos: notes for maintainers and validators

The agent-facing contract is `instruction.md`. This page records how the task is graded, what every file of `tests/`
is for, how the hidden reference is laid out, what the configuration keys mean, what "frozen" covers and how to check
it, how to grade one document again, what determinism is required and was measured, what the grader's own QC showed,
and what is known to be rough (state of 2026-11-17, frozen before any model episode; one change to the functional
item's cylinder search on 2026-11-18, frozen again: 2.4, 6; a change of text only after that: 6). Where every piece
of geometry comes from, and under which terms, is in `NOTICE.md`. Neither file reaches the agent: `tests/` exists
only in the verifier's sandbox.

Words. "The original" = the robot's own parts as its authors published them (`reference/`). A band `a / b` gives full
credit for an error up to `a`, none from `b`, linear in between; `rising(x; a, b)` is 0 up to `a`, 1 from `b`, linear
in between. L = the normal-resolution limit the photo set was planned for: 0.040 mm for small parts, 0.067 mm for
large ones.

## 1. What the robot of this task is made of

One strut of a three-strut, cable-driven tensegrity rolling robot. The 13 made parts the agent models (tags P01 to
P13, 8 designs) are graded against the authors' own geometry: 8 pieces of their published whole-strut mesh, kept
triangle for triangle (`reference/parts/`). The two purchased component types are the vendor's own STEP of the
gearmotor and the mesh's rod piece converted triangle for triangle into a faceted solid. Nothing of the robot is
modelled, redrawn or completed anywhere in this task, and there is no hidden variant. The photographs the agent gets
are renders of the same mesh. The cables, strain sensors and wires have no geometry in the sources and appear nowhere.

## 2. The grade

### 2.1 Path

`test.sh` (root; closes the stage folder) -> `verify.py`: (0) the lockdown of `lockdown_runtime.py`: stop, kill and
freeze the agent account's processes (in the separate verifier these are the image's own desktop), lock the stage,
snapshot `/home/user/submission` into a root-only folder, zero the score if an entry of the submission was written
after the stage upload began; (1) tool-use statistics from the trajectory (never a gate); (2) `fcstd_extract.py` under
FreeCAD 1.1.4's headless interpreter as `nobody`, with a time limit: every solid of the Assembly in world coordinates
as one BREP each, the labels, the Grounded, Fixed and Revolute joints; (3) `grade.py` in a child process with a time
limit and a CPU limit; (4) `public_model_reuse`; then the verdict through the benchmark kit (`robotics_grader`).
No agent code runs: the document is data that FreeCAD reads, and the assembly is not solved again.

    design_score = 0.25 S + phi x (0.20 H + 0.15 K + 0.10 G + 0.15 P + 0.10 C + 0.05 M)
    phi          = rising(S; 0.30, 0.80)                      (the shape floor)
    reward       = design_score x public_model_reuse          (anchor.json: value 1.0, full_at 1.0)

### 2.2 The seven criteria (every number is `grader_config.json`'s)

Each tagged part is first fitted rigidly on its own onto its design (`strutgrade/fit.py`), wherever it sits in the
document: S, H and K are graded in that fit. Then the whole strut is fitted rigidly onto the original
(`strutgrade/align.py`: no scale, no mirror, no base part; every fitted part and every recognised purchased instance
has a say; identical parts are assigned to installed positions one to one inside their family). P, C and M use that
alignment; G is measured inside the submission and uses it only to know which part plays which role. Size classes:
small = P06 to P13, large = P01 to P05.

| Criterion | Weight | What is measured | Bands | Mean over |
|---|---:|---|---|---|
| S shape (`items_shape.py`) | 0.25 | per Body: 0.7 band(e) + 0.3 rising(IoU; 0.40, 0.95); e = the worse of the two one-way 90th percentiles of the tangent-disk surface distance (original -> submission, submission -> original), on the surface the photographs let one locate; IoU on the whole part | e: 0.12 / 0.60 mm small, 0.20 / 1.00 mm large | the 13 Bodies, weights of 2.3 |
| H holes (`items_holes.py`) | 0.20 | per hole step: band(axis offset) x band(radius difference) x band(axis angle), after a one-to-one assignment of the original's steps to the concave cylinders found on the submitted part | offset as S; radius 0.08 / 0.35 mm; angle 2 / 10 deg | the 64 steps of the strut |
| K flat faces (`items_planes.py`) | 0.15 | per graded flat face: the median, over its seen points, of the distance to the nearest submitted surface facing the same way (normals within 30 deg; 5 mm when none within 5 mm) | as S | the 179 graded faces |
| G functional geometry (`items_functional.py`) | 0.10 | 11 quantities between the submission's own parts (2.4) | 2.4 | their shares |
| P placement and joints (`items_placement.py`) | 0.15 | 0.5 made parts + 0.3 purchased instances + 0.2 joints (2.5) | 2.5 | - |
| C fit and clearance (`items_fit.py`) | 0.10 | 0.4 purchased (contact x intrusion) + 0.3 made (intrusion) + 0.3 spool sweep clearance (2.5) | 2.5 | - |
| M mass properties (`items_mass.py`) | 0.05 | 0.3 total mass + 0.35 centre of mass + 0.35 principal moments (largest relative error) | 2 / 25 %; 0.5 / 5.0 mm; 4 / 50 % | - |

The floor: H, K, G, P, C and M are multiplied by phi. Its lower end 0.30 is the most a document can reach on S through
solid overlap alone; its upper end 0.80 is the weighted shape score of the whole strut with every face moved 0.25 mm
(the worse sign: 0.8210), rounded down to 0.05. A bounding-box or convex-hull strut therefore keeps only 0.25 S.

### 2.3 What is graded, and the weights

Fixed at the freeze from `reference/features.json` (what the original has) and `reference/seen.json` (what the 287
photographs let a careful solver locate). The weight of a Body in S is 1 + its graded hole steps + its graded flat
faces; a missing or uncredited Body scores 0 and keeps its weight.

| Body | design | class | hole steps (position / radius graded) | flat faces graded / in the original | weight | share of S |
|---|---|---|---:|---:|---:|---:|
| P01 | F-EA | large | 14 / 8 | 34 / 97 | 49 | 19.1 % |
| P02 | F-EB | large | 6 / 6 | 43 / 85 | 50 | 19.5 % |
| P03 | F-PB | large | 4 / 4 | 2 / 14 | 7 | 2.7 % |
| P04, P05 | F-MA | large | 11 / 11 each | 18 / 52 each | 30 each | 11.7 % each |
| P06, P07 | F-MB | small | 4 / 4 each | 7 / 16 each | 12 each | 4.7 % each |
| P08, P09 | F-WI | small | 4 / 3 each | 3 / 3 each | 8 each | 3.1 % each |
| P10, P11 | F-EC | small | 1 / 1 each | 21 / 31 each | 23 each | 9.0 % each |
| P12, P13 | F-CO | small | 0 | 1 / 1 each | 2 each | 0.8 % each |
| strut | | | 64 / 56 | 179 / 402 | 256 | |

* Hole steps: every step of every hole, bore and counterbore counts once (64 steps in 48 holes). All 64 are graded
  for position; the radius of 8 is not graded because hardware (an insert, a screw, a set screw) covers the mouth in
  every photograph that shows it (6 on P01, 1 on each spool). Depth is never measured.
* Flat faces: a face is in the original's list when it is an edge-connected set of coplanar triangles of at least
  2 mm2 that is flat within 0.005 mm and is neither a strip of a faceted curved surface nor part of a fitted
  cylinder, sphere or cone (402 per strut; the rule and its numbers are in `features.json` `rules`). It is graded
  when at least 60 % of its samples are seen, at least 60 % of its boundary is visible in the photographs that see
  it and, where a sharp edge of it is in view, a contrast-to-noise ratio of at least 5 holds over at least 60 % of
  the sharp-edge length that two or more photographs show: 179 per strut. Without the edge condition there would be
  360; with an edge share of 80 % there would be 92. The rule was fixed before any score under it was seen
  (`seen.json` `plane_reading`).
* Surface: only reference samples marked seen are distance targets, and only submitted samples whose nearest
  reference sample is seen count the other way (seen share of the surface: F-EA 95.7 %, F-EB 99.2 %, F-PB 87.9 %,
  F-MA 75.1 %, F-MB 96.5 %, F-WI 93.9 %, F-EC 92.4 %, F-CO 99.9 %).

### 2.4 Functional geometry (weight 0.10)

Each reference primitive (a sphere, a bore, a post, a drum profile) is carried onto the submitted part by that part's
own fit and looked for there, within 3 mm and within 50 % of its radius, by least squares on the part's surface
samples; the quantity is then measured between the submission's own primitives. A quantity whose primitive is not
found scores 0 for that quantity only. A bore or post whose last search window is not round as a whole (rms above
max(0.05 mm, 3 % of the radius)) because a flat, a key or another surface lies beside its wall inside the window (a
D-shaped bore, a flatted post) is fitted on its wall alone: the points whose normals are within 7 deg of the radial
direction (half the holes criterion's smoothness angle) and that lie within that limit of one circle, when they are
more than half of the window's points; a hole of a few flat sides, a square hole or a slot is still not found
(`items_functional.cylinder_wall`, added on 2026-11-18 after the first measured episode; section 6). Axial distances
and turns use the recognised rod's axis (without a rod: the line through the two end-cap bores); turns are relative
to the electronics datum, the common direction of the radial holes of P01 and P02 (one of them suffices; a half whose
shape score is below 0.2 gives none).

| Quantity (`functional.items[].name`) | instances | reference | share of G | band |
|---|---|---:|---:|---|
| `node_sphere_centre_distance` | P12 with P13 | 290.800 mm | 15 % | 0.6 / 4.0 mm |
| `node_sphere_radius` | P12, P13 | 17.500 mm | 5 % | 0.12 / 0.6 mm |
| `winch_axis_offset_from_rod_axis` | P08, P09 | 9.200 mm | 15 % | 0.15 / 1.0 mm |
| `drum_mid_to_node_centre_axial` | P08 with P12, P09 with P13 | 21.25 mm | 10 % | 0.3 / 2.0 mm |
| `drum_diameter` | P08, P09 | 6.350 mm | 5 % | 0.1 / 0.6 mm |
| `post_mid_to_node_centre_axial` | the 4 posts of P10 with P12, of P11 with P13 | 2.0 mm | 10 % | 0.3 / 2.0 mm |
| `post_axis_offset_from_cap_bore_axis` | the 8 posts | 10.000 mm | 5 % | 0.15 / 1.0 mm |
| `winch_azimuth_minus_electronics_datum_mod180` | P08, P09 | 86.84 deg, 4.97 deg | 10 % | 2 / 12 deg |
| `twist_between_winches` | P08 with P09 | 81.86 deg | 5 % | 2 / 12 deg |
| `post_pattern_phase_minus_electronics_datum_mod90` | P10, P11 | 0.00 deg | 10 % | 2 / 12 deg |
| `motor_encl_A_mid_to_rod_mid_axial` | P04, P05 | 103.9 mm | 10 % | 0.5 / 4.0 mm |

A share is split equally over the quantity's instances (20 in all). The two turns of the motor modules are what the
original has (3.16 deg and 4.97 deg off a right angle and off zero), not round numbers.

### 2.5 Placement, fit, mass

* **Made parts (0.5 of P).** Per reference instance: band(displacement of the design's centroid; 0.5 / 4.0 mm) x
  (0.7 + 0.3 band(orientation; 1.5 / 12 deg)); the 13 averaged. Orientation is taken modulo the design's symmetry
  group (`families.json`): end cap C4 about its axis; end cap cover continuous about its axis; spool continuous and
  end for end; the others none.
* **Purchased instances (0.3 of P).** Gearmotor: completeness x band(centroid; 0.4 / 3.0 mm) x (0.7 + 0.3
  band(orientation; 1.5 / 12 deg)). Rod: band(lateral offset; 0.2 / 1.5 mm) x band(axial position; 0.5 / 4.0 mm) x
  band(axis angle; 0.3 / 3 deg), end for end and any turn about its axis free. Weights 2 each (two gearmotors, the
  rod).
* **Joints (0.2 of P).** 0.6: the share of pieces reached from a Grounded component through unsuppressed Fixed joints
  (made parts and purchased placements half each; a spool reached through its Revolute counts). 0.4: per spool, an
  unsuppressed Revolute to its end's gearmotor (or to a component fixed to it) whose axis lies on the motor's output
  axis: band(distance; 0.2 / 1.0 mm) x band(angle; 1 / 8 deg), the worse of the joint's two frames. A spool held by
  a Fixed joint loses only this share. A document without an Assembly gets no joint credit.
* **C purchased (0.4).** Per placement: band(the smallest distance to each made part that is near it in the
  original, beyond the original's own distance; 0.1 / 2.0 mm), the pairs averaged, x band(volume shared with made
  parts beyond what the original shares, plus all it shares with extra solids, as a share of its own volume;
  0.05 / 5 %).
* **C made (0.3).** Per instance: the same intrusion band on its own volume. The original's own overlaps are exempt
  (`features.json` `supports.own_overlaps_in_scope`), with 0.5 mm3 of slack.
* **C clearance (0.3).** Per spool: a cylinder of the original flange radius (6.350 mm) and spool width (9.500 mm)
  on the submitted spool's axis, against everything except that spool and its gearmotor: band(excess overlap;
  0.5 / 40 mm3).
* **Who pays for an overlap.** See 10.3.
* **M.** Made parts at their material density (`families.json` `materials`: printed PLA 1.24, SLA resin 1.20,
  silicone 1.08, board laminate 1.85 g/cm3), purchased instances at their data-sheet mass spread over the vendor
  solid (gearmotor 11 g, rod 6.483 g), extra solids at 1.24 g/cm3. Out-of-scope hardware counts on neither side.

### 2.6 Zeros, local rules, task errors

Three conditions zero the whole document inside the grade (`strutgrade/gates.py`; `cad_valid` 0):

1. `unreadable`: no extraction, the document does not open, the extraction did not finish in time, more than 4000
   shape leaves or 6000 solids at the reader, or more than 2000 usable solids (`gates.max_solids`);
2. `no_solid`: the Assembly (or, without one, the visible objects) holds no usable solid;
3. `too_few_fitted`: fewer than 3 pieces can be fitted (credited tagged parts plus recognised purchased instances).

Outside the grade, `verify.py` also gives 0 when nothing was submitted (no `strut.FCStd`; a single other `*.FCStd`
in the submission root is read instead), when the submission was written to after the agent phase ended or its root
is a symbolic link or resolves outside the agent's home (`submission_final` 0), and when a file of the submission
has the sha256 of a public file of this robot (`public_model_reuse` 0; `public_blocklist.json`).

Everything else is local and costs only the piece concerned (each with a diagnostic and a statistic):

* a tagged leaf (label P01 to P13) that is not a PartDesign Body of exactly one valid closed solid gets no credit;
  its valid solids stay as extra solids;
* a tag on more than one leaf (also one Body reached twice through links): none is credited;
* invalid solids, open shells and loose faces of untagged leaves are left out;
* purchased components are recognised by full shape, never by label (volume within 1 %, then a rigid registration:
  exact on the B-rep vertex set for an unchanged import, else ICP);
* extra solids (untagged and not recognised, or demoted by the rules above) count in the mass at the printed-part
  density and as obstacles in fit and clearance; they are not fitted and take no part in the alignment;
* no Assembly: the visible solids are graded, without joint credit.

A grading item that raises on a document scores 0 for that item and is counted (`grader_item_errors`). A failure of
the grader itself is a task error, never a score: a reference file that does not match its sha256 or its triangle
count, a vendor STEP that does not match its pin, a `seen.json` bitmap that does not match the grader's own samples,
an exception outside every item, a grading child that crashes, writes nothing or runs out of time, a process of the
agent account that is still running after the verify-start freeze.

## 3. Files of `tests/`

| File | Role | In the freeze (section 6) |
|---|---|---|
| `test.sh` | entry of the verify stage: closes the stage folder and its parent, pins BLAS threads, runs `verify.py` | code |
| `verify.py` | the order of 2.1, the verify-level zeros of 2.6, the verdict (one gated `reward`; checks, criteria and statistics as metrics; a diagnostic text of at most 2000 characters) | code |
| `lockdown_runtime.py` | verify-start isolation (kill, freeze, stage lock, snapshot, late-write gate), the unprivileged extraction, the grading child, tool-use statistics | code |
| `fcstd_extract.py` | the only contact with FreeCAD: reads the document into `model.json` (format `strut-fcstd-extract/1`) and one BREP per solid; lists the document's Python proxy classes without executing anything | code |
| `grade.py` | the grading child: base, seven items, aggregation; turns the grader's own failures into task errors | code |
| `strutgrade/__init__.py` | the context, the item contract, `BASE_DEFAULTS`, `resolve_config`, `run_base`, `run_item` | code |
| `strutgrade/detenv.py` | deterministic environment: thread counts, CPU affinity, the OpenBLAS kernel, the seeds | code |
| `strutgrade/reference.py` | loads `reference/` and the pinned vendor STEPs; every mismatch is a task error | code |
| `strutgrade/geom.py` | rigid motions, B-rep access, tessellation, exact booleans (manifold3d), seeded surface sampling, the tangent-disk distance | code |
| `strutgrade/gates.py` | the three whole-document zeros and the local rules | code |
| `strutgrade/recognise.py` | recognition of purchased parts by full shape | code |
| `strutgrade/fit.py` | per-part fit with candidate poses | code |
| `strutgrade/align.py` | whole-strut alignment without a base part | code |
| `strutgrade/primitives.py` | flat faces, concave cylinders, spheres and radial profiles on a triangle mesh (numpy only); the same code built the reference tables and reads the submission | code |
| `strutgrade/items_*.py` | the seven criteria (2.2) | code |
| `grader_config.json` | every number of the grader (section 5) | configuration |
| `reference/` | the hidden reference (section 4) | reference list |
| `anchor.json` | the reward scale: metric `design_score`, value 1.0 (the oracle's own score), `full_at` 1.0 | no |
| `public_blocklist.json` | sha256 of 29 public files of this robot (23 distinct): the published whole-strut mesh, six print files of a public dataset, the 22 STL files of the authors' CAD package | no |
| `env.py` | says that this is an artifact task: no simulator or controller is scored | no |
| `arm/` | vendored lockdown helpers (`verify_freeze.py`, `assembly_six/lockdown.py`, `assembly_cell/runner.py`), byte-identical to the ones in `uav_cad_design_from_photos` | no |
| `robotics_grader/` | the benchmark's shared verify kit, byte-identical to `shared/robotics_grader/` | no |
| `.gitignore`, `.gitattributes` | `verdict.local.json` and `__pycache__/` ignored; `*.ply`, `*.bits`, `*.FCStd` binary | no |
| `README.md`, `NOTICE.md` | this page and the sources | no |

`solution/` holds the oracle: `solve.sh` runs `build_oracle.py` under the image's `freecadcmd`; it builds the
reference document natively at oracle time from `solution/reference/` (`oracle.json`: tags, families, instance poses,
purchased placements, the joint table; `parts/`: the 8 design meshes, byte-identical to `tests/reference/parts/`) and
the vendor STEPs staged in the agent's home. Every made part is its original mesh turned triangle for triangle into a
faceted solid (one planar face per triangle, nothing merged), every purchased part the vendor's STEP imported
unchanged; one PartDesign Body per tag, one Assembly with P01 grounded, a Fixed joint to every other component and a
Revolute joint without limits for each spool. It checks every input's sha256 and that nothing moves on recompute,
solve, save, reopen and a second solve. The document is about 49 MB and is not in the repository.

## 4. The reference (`reference/`)

Frame of everything: the strut frame, rod axis = +z, rod midpoint = origin, millimetres. A pose is a 4 x 4 row-major
matrix mapping a file's coordinates to the strut frame.

* `parts/F-<key>.ply`: the 8 designs, each one piece of the authors' mesh (binary PLY, 28 604 triangles and 544 632
  bytes together). The grader takes them as meshes, without repair.
* `families.json`: `families[]` (key, file, sha256, bytes, source piece, triangles, volume, area, size class and its
  margin, symmetry group, material, tags), `instances[]` (tag, family, end, source piece, pose, how the pose was
  registered), `materials` (densities and where they come from), `symmetry_tests`, and the flip between the two motor
  modules. Family keys: F-EA electronics enclosure half that carries the board (P01), F-EB its cover (P02), F-PB the
  bare main board (P03), F-MA motor enclosure A (P04, P05), F-MB motor enclosure B (P06, P07), F-WI spool (P08,
  P09), F-EC end cap (P10, P11), F-CO end cap cover (P12, P13). Pairs are rigid copies of one mesh (residual at most
  3.7e-5 mm, determinant +1): there is no mirrored part.
* `features.json`: per design `holes.steps[]` (id, hole group, axis, radius, two ends, the triangles), `holes.holes[]`
  (the steps of one hole), `planes[]` (id, outward normal, offset, area, flatness, the triangles); `functional`
  (`primitives` per design, `values`: the 11 quantities with instances and reference values, `derived`);
  `supports` (for each purchased placement the made parts near it with contact area and smallest distance, the
  original's own overlaps, the spool sweep reference); `rules` (the hole and flat-face rules with their numbers);
  `weights_upper_bound` (the weights if everything were graded: 479).
* `seen.json` and `seen/F-<key>.{dense,query}.bits`: what the photographs let one locate. `designs.<key>.samples`
  names the two bitmaps (numpy `packbits`, bit order little, in the grader's own sampling order) with the number of
  samples and the hash of the point set they mark; `hole_steps[]` (`position_gradable`, `radius_gradable`),
  `planes[]` (`gradable`); `rules` states the visibility rules in words. The grader reads only `designs`; the
  lists `functional` (which photographs support each quantity) and `plane_reading` are records. The dense samples
  are at least 20 points per mm2 (20 000 to 500 000 per design), the query samples 2 per mm2 (4 000 to 60 000).
* `purchased.json`: the two types (part number, vendor, name, data-sheet mass, local frame, symmetry, anchor
  constituent, weight in P) and the three placements GM1, GM2, ROD1 (local frame -> strut frame, how registered).
* `purchased_pins.json`: for each type the vendor STEP's path under the purchased root, its sha256 and bytes, and for
  each of its solids the volume and centroid. The grader reads the STEPs from the image's own fetched payload
  (`purchased_root`, `/opt/task-assets/payload/purchased`), never from the agent's home.
* `reference.json`: the reference list written at the freeze: sha256 and bytes of the 29 files above, the sha256 of
  the frozen `grader_config.json`, the sampling keys and, per design, the number, seed and point-set hash of the dense
  and query samples as the grader draws them, each equal to what `seen.json` was written against. The grader does not
  read this file.

## 5. Configuration (`grader_config.json`)

Bands are `[full credit, zero credit]` for errors and `[zero, full]` for rising quantities; lengths mm, volumes mm3,
angles degrees unless a key says otherwise. The base's defaults (`strutgrade.BASE_DEFAULTS`) and each item module's
defaults are overlaid by this file; the frozen file names every number (`_defaults_used` is empty). It overrides
two base defaults: `sampling.disk_mm` 0.25 (default 0.15) and `fit.trim` 1.0 (default 0.9). A grade run without this
file is therefore not the frozen grader.

| Section | Keys |
|---|---|
| top level | `status` (`frozen`; reported by every graded run as `config_frozen`), `status_note`, `extract_timeout_s` 1800, `grade_timeout_s` 1800, `grader_cpus` 4, `purchased_root`, `provenance` (what changed against the design values and why) |
| `aggregate` | `weights`, `floor` [0.30, 0.80], `floored` (the six criteria the floor multiplies), `allow_placeholders` false (a missing item module is a task error) |
| `detenv` | `grader_cpus`: threads of the exact booleans and the CPU affinity |
| `gates` | `max_solids` 2000, `min_volume_mm3`, `recheck_validity`, `min_fittable` 3 |
| `mesh` | tessellation of submitted solids (`deflection_mm` 0.01, `angle_rad` 0.1, the retries), welding, hole filling, `exact64` (try the float64 manifold of the raw tessellation first), `repair_rounds` |
| `sampling` | `seed` 20261117, dense and query densities and limits, `disk_mm` (the tangent disk of the surface distance) |
| `recognition` | purchased parts: `volume_rel` 0.01, `exact_tol_mm`, the ICP fallback and its acceptance, `attach_mm` |
| `fit` | per-part fit: starts (`spin_step_deg` 15), coarse and fine ICP, `window` 0.05 (poses within this of the best shape score stay candidates), `feature_tie` 0.01, `refine_symmetric` 24, `trim` and `coarse_trim` |
| `shape_minimal` | the base's built-in shape score (used only if the shape item were missing) |
| `align` | whole-strut alignment: `points_per_part` 64, `robust_mm` 0.25 and `cap_mm` 2.0 (the loss), `rounds` 5, `shape_min` 0.2 (below it a part contributes its centroid only), `land_mm` |
| `shape`, `holes`, `planes` | bands by size class, `quantile` 0.9, `distance_cap_mm` 3.0; the hole finder's three passes and the matching (`match_cap_factor`, `cost_clip`); `normal_deg` 30, `search_mm` 5, `missing_mm` 5 |
| `functional` | capture range, fitting thresholds, `datum_shape_min` 0.2 and `items[]` (name, block, weight, band) |
| `placement` | the three shares, the bands of 2.5, `type_weights`, `joints` (which gearmotor each spool belongs to) |
| `fit_clearance` | the three shares, `contact_mm`, `intrusion_rel`, `slack_mm3` 0.5, `sweep_mm3`, `blame_tie` 0.05 |
| `mass` | the three bands and their weights, `extra_material` |

## 6. What is frozen, and how to check it

Frozen on 2026-11-17 18:09:17 UTC, before any model episode: the configuration, the reference list, the photo list and
the grader code. Frozen again on 2026-11-18 06:54:30 UTC after one change of grader code, the functional item's
cylinder search (`strutgrade/items_functional.py`, 2.4): the configuration, the 29 reference files and their samples,
the photo list and the instruction are the same bytes; `reference.json` differs only in its freeze time. Frozen a
third time on 2026-11-18 11:32:22 UTC after a change of text only: comments, docstrings and note strings of the grader
code, the configuration and the reference tables (and three key names of records the grader does not read: two in
`seen.json` `plane_reading`, one in `purchased.json`'s instances) lost references to development records; every code
file has the same syntax tree without its docstrings, every value the grader reads is unchanged, the reference samples
are the same point sets, and the oracle, the empty submission and the measured episode's document grade bit for bit as
before. Later that day nine more docstrings were reworded in the same way (the same syntax trees, the same grades);
the code list hash below includes them, and no other identity moved. Any change to one of them is a new version of the
task and has to be accepted again (oracle 1.0, nothing submitted 0, the QC of section 9); scores of episodes recorded
before stay as they are; a re-grade under another configuration is marked as such and is not a reading. `README.md`
and `NOTICE.md` are not part of the freeze.

| What | Identity |
|---|---|
| `grader_config.json` | sha256 `531746a4c7c61ade4ee3c579a63ec950876ef7a871a95450c66a2021f7164a4e` |
| `reference/reference.json` (pins the 29 reference files) | sha256 `dd9d02e41e2c9a1aa3e7a295b2306afac9f266d10d88916573c9d45aa9d250d8` |
| the 21 code files (`fcstd_extract.py`, `grade.py`, `lockdown_runtime.py`, `strutgrade/*.py`, `test.sh`, `verify.py`) | list hash `bbd11a3711f171dd5baac9cbbe1b1eb77b566f3bd3627ce2a3d57e8bb51d21c6` |
| the 287 photographs of the payload | list hash `2077cdc55000a83c318bca1f36c24e562ccd17bad2ec5d6d0dbb4b64965b430d` |
| `instruction.md` as generated from this configuration | sha256 `4b443e63a8eddd39c52f8fe0e5dd461aad2e3c0c2be2d3f1922e5152ce1bce8a` |

A list hash is the sha256 of the lines `<sha256> <name>` in byte order of the names. From the task folder:

    for f in $(ls tests/fcstd_extract.py tests/grade.py tests/lockdown_runtime.py tests/strutgrade/*.py \
               tests/test.sh tests/verify.py | LC_ALL=C sort); do
      printf '%s %s\n' "$(sha256sum "$f" | cut -d' ' -f1)" "$f"; done | sha256sum
    grep '  photos/' environment/assets.sha256 | awk '{sub("photos/", "", $2); print $1 " " $2}' \
      | LC_ALL=C sort -k2 | sha256sum

Every graded run reports `config_frozen` 1 in its metrics. The instruction prints the bands, weights and the floor
from the configuration, so a new configuration needs the instruction generated again.

## 7. Grading one document again

**On a host, without the lockdown** (not root: the same extraction and the same grading code, run as the calling
user). Needed: FreeCAD 1.1.4 (`freecadcmd`), a `python3` with the packages of `environment/requirements.txt` (numpy,
scipy, trimesh, manifold3d, cadquery-ocp at those versions), and the payload's `purchased/` folder with the two
vendor STEPs pinned in `reference/purchased_pins.json`.

    RUN=$(mktemp -d)                                   # a private folder of its own
    cp -r tests "$RUN/stage"
    mkdir -p "$RUN/home/submission" && cp /path/to/strut.FCStd "$RUN/home/submission/strut.FCStd"
    cd "$RUN/stage"
    env -u ALE_VERIFY_PYTHON ALE_HOME="$RUN/home" ARB_FREECADCMD=/path/to/freecadcmd \
        ARB_PURCHASED_ROOT=/path/to/payload/purchased ROBOTICS_GRADER_VERDICT_PATH="$RUN/verdict.json" \
        OPENBLAS_CORETYPE=Haswell PYTHONDONTWRITEBYTECODE=1 TMPDIR="$RUN" bash test.sh

`$RUN/verdict.json` holds `rewards.reward`, `metrics` (`design_score`, the seven criteria, `shape_floor`,
`cad_valid`, `public_model_reuse`, `config_frozen`, timings, counts) and the `diagnostic` text; the grading child's full
result (every part, hole step, flat face and functional quantity) is `$RUN/arb_grade_*/result.json`, the extraction
`$RUN/arb_extract_*/out/`. Leave out the document to get the "nothing submitted" verdict. Cautions: `test.sh` sets
the stage folder and its parent to mode 0700 (the parent stays so when not root), hence the private folder; and never
run it as root on a shared machine: as root, `verify.py` kills and freezes every process of the account named after
the last component of `ALE_HOME`. The oracle document scores 1.0 this way (4 cores: 92 s on a moderately loaded
host, 321 s on one loaded far beyond its cores).

The two steps can also be run by hand, on a copy of `tests/`:

    /path/to/freecadcmd -c "import sys; sys.argv = ['fcstd_extract.py', '/abs/strut.FCStd', '/abs/out']; exec(open('/abs/stage/fcstd_extract.py').read())"
    OPENBLAS_CORETYPE=Haswell python3 -B /abs/stage/grade.py /abs/out /abs/stage/reference /path/to/payload/purchased /abs/result.json

**In the task image, the real path** (what the QC of section 9 used; the engine's separate verifier does the same in
a sandbox it starts itself). Copy the document first and stage `tests/` after it: the late-write gate compares the
submission's change times with the stage's.

    docker run -d --rm --name strut-verify --network none --cpus 4 --memory 16g -e VNC_RESOLUTION=1440x900 \
        -v "$RUN/home/submission:/home/user/submission" -v "$RUN/stage:/opt/ale/tests" <task image>
    until docker exec strut-verify test -f /tmp/gnome-ready; do sleep 1; done
    docker exec -u root -w /opt/ale/tests -e ALE_HOME=/home/user -e ALE_STAGE_DIR=/opt/ale/tests \
        strut-verify bash test.sh
    docker exec -u root strut-verify chown -R "$(id -u):$(id -g)" /opt/ale/tests
    docker stop strut-verify                           # the verdict is $RUN/stage/verdict.local.json

`ALE_VERIFY_PYTHON` is left unset, so the kit writes the verdict into the stage folder instead of the engine's log
folder; the numbers come from the same code. The oracle document scores 1.0 this way with `late_write_gate_active`
1 and `stage_locked` 1 (verify 90 s at a host load near 47, 138 s at a load above 200). Numbers reported for this
task are the image's (section 8).

`ale validate` and episodes of this task are run with `--set desktop_resolution=1440x900` (see `task.toml`).

## 8. Determinism

Required: the same document graded twice in the task image gives rewards within 1e-9; on two CPU generations within
1e-6. How: one base seed (`sampling.seed`); every other seed is derived from it and from what it is for (a design
key, a solid, a stage), never from the order of grading; the reference samples are drawn by the grader itself and
checked against `seen.json` by point-set hash; fits, assignments and tie-breaks run in a fixed order; BLAS threads 1,
the boolean kernel's threads and the CPU affinity pinned to `grader_cpus`, `OPENBLAS_CORETYPE=Haswell` set before
numpy is imported (`strutgrade/detenv.py`).

Measured: four documents (the oracle, a face-offset strut, the swapped-angle strut, the convex-hull strut) graded
twice in the image: rewards bit-identical, the grading child's results identical word for word apart from timings;
the same four on a second CPU generation (Zen 4 and Zen 5): bit-identical, all 91 non-timing metrics equal; the
612 MB document of 10.4 twice under different host load: identical. These pairs were measured on the grading code
that was then frozen; after the freeze every one of the 69 QC documents scored bit for bit what it had scored before.

Host against image: a host environment with manifold3d 3.5.5 and numpy 2.5.4 (the image has 3.5.3 and 2.5.3) gave
bit-identical scores on 67 of 68 documents and a fit item that differed by 2.5e-4 on one (a boolean volume). Report
the image's numbers.

## 9. The grader's own QC (summary of the mutant table)

69 documents, every one derived from the original parts by rigid motion, relabelling, another way of importing or
saving, face offset, scaling, exact morphological opening and closing, convex hull, bounding box or mirroring, each
graded through the real verify stage in the task image (one container at a time, no network, 4 CPUs, 16 GB) on the
frozen configuration: 68 as expected, 1 known margin (10.3), no failure. These are tests of the grader, not
measurements of how hard the task is. The documents are not shipped (they hold the graded geometry).

| Group | Documents | Result |
|---|---|---|
| benign (17) | the oracle; two other ways of importing the purchased parts (FreeCAD's own STEP import into an App::Part; one feature per type, linked once per instance); the whole strut moved rigidly; the rod end for end and turned; the references of the Fixed joints swapped; the Revolute frames flipped; tags swapped inside every family; spools, covers turned about their axes, end caps a quarter turn; the same holes written four ways (analytic, B-spline, 12-sided and 36-sided prisms); the oracle saved from the GUI with joint view providers attached, reopened and solved again (two documents) | all 1.0000, every criterion >= 0.999 |
| ruled (2) | a spool turned end for end; the two motor modules each turned to the other's angle | 0.9991 (only C, 0.991: 10.6); 0.9269 (G 0.850, P 0.655, M 0.874: 10.7) |
| whole-document zeros (5) | nothing submitted; an empty document; no solid; an unreadable file; fewer than three fittable pieces | all 0 |
| degenerate (2) | every part its bounding box; every part its convex hull (in place, purchased parts and joints complete) | 0.0098 = 0.25 x S 0.039; 0.0208 = 0.25 x S 0.083; floor 0 in both |
| face offsets (8) | every face moved by +2 L / -2 L; +6 L / -6 L; +12 L / -12 L; +0.25 mm / -0.25 mm | 0.9454 / 0.9672 (required >= 0.90); 0.6203 / 0.6210; 0.0755 / 0.0220; 0.7901 / 0.8075 |
| scaling (6) | each part scaled along its own axes by +0.5 % / -0.5 %; +1 %; +2 %; the whole strut by 1 %; 2 % | 0.9960 / 0.9965 (required >= 0.95); 0.9725; 0.8356; 0.9382; 0.6761 |
| detail removal (6) | every design filtered: identity; gaps narrower than 1 mm closed; narrower than 2 mm closed; walls thinner than 1 mm removed; thinner than 2 mm; thinner than 2.3 mm (end caps) | 1.0000; 1.0000; 0.9747; 0.9762; 0.7969; 0.9443; every removed hole step and deeply removed flat face scores 0 (43 features); named parts under their caps except the row of 10.1 |
| structure (6) | P01 as its convex hull; P01 as two solids; P03 as a plain board without holes; an end cap mirrored; a motor enclosure A mirrored; only one electronics half fittable | 0.8604; 0.8294; 0.9865 (10.3); 0.9933 (10.6); 0.9042 (10.7); 0.8876; the other parts' terms stay >= 0.99 except as noted in section 10 |
| single changes (17) | a motor module moved 3 mm along the rod; turned 10 deg; an end cap with its cover moved 2 mm; a spool 1 mm off its motor axis; drum diameter +0.5 mm; a cover's outer radius +0.5 mm; a post moved 1 mm; a hole moved 0.5 mm; a hole radius +0.215 mm; a flat face moved 0.6 mm; a part missing; a gearmotor missing; local rules (a tagged leaf that is not a Body, a Body used twice through a second link, an untagged block); no Assembly; all joints suppressed; purchased parts moved 1 mm; turned 5 deg | 0.9628; 0.9761; 0.9838; 0.9740; 0.9964; 0.9961; 0.9995; 0.9988; 0.9984; 0.9996; 0.9557; 0.9438; 0.7977; 0.9700; 0.9707; 0.9741; 0.9964 |

In the 21 rows whose expectation is "only its own terms" (the single changes, P01 as two solids, one electronics
half, the two ruled rows) no sub-score outside the row's own list moves by more than 0.005.

Also measured: `ale validate` of the frozen task: untouched 0.0, oracle 1.0 (verify stage 91 s). Verify time of the
64 QC documents that reach the grade, at a host load averaging 47: 84 to 341 s, median 118 s (extraction 6 to 148 s,
median 31 s; container memory peak 1.1 to 2.8 GB); the oracle and its two copies saved from the GUI 87 to 90 s. Very
large documents: 10.4.

## 10. Known limits and margins

1. **One named detail-removal row is not graded, by rule.** Removing a detail must cost score when the removed detail
   is graded content: seen in the photographs and readable there. Closing every gap narrower than 2 mm on motor
   enclosure A (which fills its clamp slot) leaves that part's shape score at 1.000. The slot's deep walls appear in
   no photograph; its two seen walls do not show readable sharp edges on the black print, so they are not in the
   graded flat-face list (the 60 % edge rule of 2.3, fixed before any score under it was seen); those two walls are
   3.4 % of the part's seen area, inside the 10 % that the 90th-percentile statistic leaves out, and the part's IoU
   stays at 0.958, above the full-credit point. The other four named rows stay under their caps: end cap with walls
   thinner than 2.3 mm removed 0.384 (cap 0.70), spool with walls thinner than 1 mm removed 0.257 and 0.259 (0.50),
   P01 and motor enclosure B with walls thinner than 2 mm removed 0.293 and 0.274 (0.50; these two only from the
   mesh harness of 10.2). This is recorded in the configuration's `status_note`.
2. **Ten detail-removal combinations cannot be built as FreeCAD solids.** For 10 of the 37 (filter, design)
   combinations the exact morphological result is not a valid single solid after save and reopen (gaps < 1 mm
   closed: F-EA, F-EB, F-EC, F-PB; gaps < 2 mm closed: F-EA, F-PB; identity: F-EA, F-MA; walls < 2 mm removed: F-EA,
   F-MB). Their documents keep the original part; their readings come from a development mesh harness that runs the
   same base and the same S, H and K code without FreeCAD: the closing and identity combinations 1.000, F-EA 0.293,
   F-MB 0.274.
3. **Who pays for an overlap (fit item).** An overlap between two pieces beyond the original's own is charged to the
   piece further from the original: the one with the lower quality (a made part: its chosen pose's shape score times
   its placement score; a purchased instance: its placement score) carries it alone; when the two qualities are
   within `fit_clearance.blame_tie` (0.05) both carry it. The rule is deterministic and does not depend on the order
   of the parts. The quality does not look at holes: a part modelled as its convex hull pays alone (its shape score
   is low), but a flat plate modelled without its holes ties with its neighbour. Measured on P03 (the main board)
   replaced by a plain board: the board wraps P01's posts (116.6 mm3 beyond the original), both pay, P01's fit term
   is 0.915 and the board's 0.660; the design score is 0.9865 instead of 0.9867 if the board paid alone. A rule that
   put the parts' own hole and flat-face scores into the quality fixed this row but moved other rows' fit terms by
   up to 1.0 (bounding box, convex hull, +12 L offset), so it was withdrawn before the freeze.
4. **Very large documents.** The extraction has 1800 s, the grading child 1800 s, the verifier 3600 s; the extractor
   writes one BREP per solid and no STEP. Measured in the task image after the freeze (4 CPUs, 16 GB): a document
   with every triangle of the original parts split in 4 plus 8 extra solids (0.32 M faces, 195 MB) verifies in 255 s
   (extraction 123 s), container memory peak 3.75 GB; one split in 16 (1.74 M faces, 612 MB, an order of magnitude
   beyond any reasonable submission) twice, at host load averaging 29 and 47: extraction 512 s and 612 s, grading
   552 s and 647 s, verify 1065 s and 1261 s, the same score both times (0.95: the extra solids zero the mass item,
   the six other criteria are 1.000; face count itself costs nothing); container memory peak 12.7 and 12.3 GB of
   16 GB (process heap 10.1 to 10.2 GB, the rest page cache). Under the earlier 900 s extraction limit the same
   document once ran out of time under load and scored 0. The two limits add up to the verifier's 3600 s, so a
   document that used both fully would end as a verifier time-out rather than a score; an extraction that runs out
   of time is "unreadable" (0), a grading child that runs out of time is a task error.
5. **What the photographs do not show is not graded.** Surface samples that are not marked seen take no part in the
   surface distances (2.3; a quarter of motor enclosure A's surface, mostly inside its cavities); IoU is still taken
   on the whole part. 223 of the 402 flat faces and the radius of 8 hole steps are not graded. The depth of blind
   holes is never graded. Two functional quantities of the +z spool (its axis offset and its drum's axial position)
   face the mat in every pose and have no supporting photograph; they are graded all the same, because both ends
   carry the same motor module design, the instruction says so, and the two ends' reference values differ by 7e-6 mm
   and 1.0e-5 mm.
6. **Near-symmetric parts and candidate poses.** Poses of a part whose shape score is within 0.05 of its best stay
   candidates; among them the part's own holes and flat faces decide, then the pose nearest the strut's consensus.
   The bare main board fits four ways equally in shape, so its pose comes from its holes and windows, and a board
   without them takes the pose the strut gives it (it keeps its orientation credit). The spool is treated as
   continuous about its axis and reversible end for end in placement (its surfaces agree within 0.004 mm at the
   90th percentile), and its small features (counterbore, set-screw hole, line hole) are graded at the phase the fit
   prefers; but the original spool has its counterbore for the gearbox boss on the motor side only, so a spool put
   on end for end intrudes by 2.87 mm3 and loses 0.009 in C (0.9991). The end cap is nearly mirror-symmetric (its
   mirror image equals a turn of 33.6 deg within the bands): a mirrored end cap keeps shape 1.000 and loses only its
   orientation credit and the phase of its posts (0.9933).
7. **Motor modules at each other's angle.** The whole-strut alignment then follows the +z module; the position loss
   lands on the three electronics parts and the -z module rather than being shared (0.9269). With a mirrored motor
   enclosure A, the sweep clearance of the spool that turns inside it is counted as that part's own loss.
8. **Curved surfaces are not graded patch by patch.** Domes, drums, spheres and cylinders are graded through S by
   area and, where they carry function, through G; only flat faces are graded one by one.
9. **The original's own interferences are exempt.** The authors' assembly overlaps itself in places (the deepest:
   the -z gearmotor 5.49 mm inside motor enclosure B); C measures only what exceeds the original's own overlap or
   distance for the same pair.
10. **Given data.** The rod the agent imports is the mesh's rod piece as a 36-sided prism (3.175 mm across corners,
    3.163 mm across flats), and its mass is its volume at 2.70 g/cm3; the densities of the printed parts (1.24 and
    1.20 g/cm3) are design values that could not be checked against the makers' pages. M weighs 0.05.
11. **Texts that lag the frozen numbers.** `grade.py`'s docstring still gives the floor as 0.30 -> 0.85, and
    `BASE_DEFAULTS` holds the two superseded defaults of section 5. The configuration is the authority; the files
    were left as they are because changing them would be a new version.
12. **No evidence from a solver.** The evidence is the oracle through the real path, the degenerate submissions and
    the QC above, plus a precision budget for the photo set. Nobody, human or model, had worked the task from the
    photographs when it was frozen.
