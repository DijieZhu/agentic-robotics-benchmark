# rov_cad_design_from_photos: grading notes for maintainers

The agent never sees this folder. It holds the grader, the hidden truth of the made parts and the numbers of the
grading. This note says where each of them is, what the grader does, how it differs from the grader it was ported
from, what it is known not to catch, and how to grade a document by hand. Sources, credits and the departures of the
photographed vehicle from the vendor's models are in `NOTICE.md`.

## What is hidden, and where

| Path | What it is |
|---|---|
| `reference/reference.json` | The reference (schema `rov-cadp-reference/1`), in the vehicle frame, millimetres. The 31 made parts by tag with their family (16 families; the parts of one family are rigid copies of one solid, none is a mirror image), material, density and scored features; the 8 purchased types with their solids (545 constituents), the solid that seeds an instance (the anchor) and the datasheet mass; the placement of the 20 purchased instances; one flow body per thruster (`rotors`; no propeller instance: `prop` is null); what each purchased instance rests on (`supports`); the 154 items that are in the photographs only (`unscored`); the mass properties. `intrinsic_interference` lists the overlaps the vendor's CAD has itself, for information: the grader computes its own baseline, pair by pair, from the reference geometry. |
| `reference/parts/Pnn.brep` | The true solid of every made part at its installed place: the vendor's B-rep, unchanged. The hidden truth. Nothing the agent gets is derived from these by anything but photography. |
| `reference/purchased_pins.json` | For every purchased type: its STEP file below the payload root, the file's sha256, and which solid of the file is which constituent. The grader reads the purchased solids from the image's own payload (`/opt/task-assets/payload/purchased`); a file that differs from its pin is a task error, not a zero. |
| `grader_config.json` | Every number of the grader: weights, bands, the shape floor, tolerances, limits, time limits. Bands are `[full credit, zero credit]` for errors and `[zero, full]` for rising quantities. |
| `public_blocklist.json` | sha256 of the public files of this vehicle that an agent could copy (the vendor's public downloads of the vehicle, its retrofit kit and its components: STEP, IGES, SolidWorks, STL and archive files; the sibling assembly task's public meshes of the vehicle's parts). A submission folder that holds one of them scores 0. |
| `anchor.json` | The benchmark kit's reference value: the reference document scores 1.0, so the reward is the design score. |
| `../solution/reference/` | `oracle.json` (tags, families, the grounded part, the placement of every purchased instance) and `parts/Pnn.brep`, byte-identical to `reference/parts/`. `../solution/solve.sh` builds the reference document from them and from the payload at solve time; no finished document is stored. |

What the agent gets is `instruction.md` and the payload only (`photos/`, `purchased/<part number>/` with a STEP file, a
datasheet and a drawing each, `parts_list.md`), fetched into the image at build time and staged in the agent's home by
`environment/setup/setup.sh`.

What is scored on the made parts: 310 features (226 holes and bores whose wall closes, 84 half-open round walls) with
320 cylindrical steps, on 24 of the 31 instances. A feature is a chain of coaxial concave faces with at least one
cylindrical step; a hole has every cylindrical step covering at least 300 degrees, a half-open feature at least one
covering 175 degrees; the smallest radius read is 0.5 mm, there is no largest. Conical faces are not scored (the 370
concave cones of the 31 instances are moulding draft, small edge breaks, drill points and one countersink pair; the
photographs cannot give them), so `reference.json` lists no conical step. The extractor finds 13 more features (9
holes, 4 half-open, 16 cylindrical steps); no photograph shows them readably, so they were taken out of scoring before
the freeze and stand in `reference.json` `features_withdrawn` (see "Known limits").

## What the verifier does

1. `test.sh` closes the staged folder, then `verify.py` (as root, in a separate verifier sandbox that holds only the
   restored `/home/user/submission`) stops the agent account's processes, snapshots the submission and gates late
   writes (`lockdown_runtime.py`, `arm/`).
2. `fcstd_extract.py` runs under headless FreeCAD as the unprivileged `nobody` account with a time limit
   (`extract_timeout_s`). It reads `rov.FCStd` (of several Assemblies that are not nested in another one, the one
   created first), writes every solid of the Assembly in world coordinates as an exact BREP, with FreeCAD's own validity
   verdict per solid, and lists the joints. The Assembly is not solved again.
3. `grade.py` runs as a child process with a time limit (`grade_timeout_s`). Its module docstring states the rules:
   what is graded, the local rules, the alignment on P01, the recognition of the purchased modules, the assignment
   inside families and types, the six criteria and the shape floor.
4. `verify.py` applies the public-file check and writes the verdict through the kit (`robotics_grader/`):
   `reward = design_score`.

The whole document scores 0 in seven cases and in no other. The instruction lists the same seven cases with the same
limits; its list is numbered 1 to 7 and begins with P01, which is Z5 in this table:

| Id | Case | Where it is decided |
|---|---|---|
| Z1 | No document: `rov.FCStd` is absent and the submission folder does not hold exactly one other `.FCStd`; or the folder is a symbolic link or resolves outside the agent's home | `lockdown_runtime.find_document`, `_early_lockdown` |
| Z2 | The document cannot be read: FreeCAD cannot open it; more than 4000 shape leaves or more than 6000 solids among what is read; the reading does not finish within `extract_timeout_s` (1500 s) | `fcstd_extract.py` (`MAX_LEAVES`, `MAX_SOLIDS`, the model written first that says the extraction did not finish) |
| Z3 | More solids in the unit than `gates.max_solids` (4000) | `grade.py`, `Submission` |
| Z4 | No usable solid | `grade.py`, `Submission` |
| Z5 | P01 is missing or unusable for the alignment: on two leaves, several solids in its leaf, or none valid | `grade.py`, `Submission` and `align` |
| Z6 | An entry of the submission was written at or after the moment the verifier's stage upload began | `lockdown_runtime._early_lockdown` |
| Z7 | The submission folder holds an unchanged public file of this vehicle | `verify.py` `public_reuse`, `public_blocklist.json` |

A failure of the grader itself (the reference, a payload file that does not match its pin, an exception outside a
grading item, a child that crashes or exceeds `grade_timeout_s`, a surviving agent process) fails the run as a task
error and is never a 0. A grading item that raises scores 0 and is counted (`grader_item_errors`).

The limit of 4000 solids is deliberate. A complete submission holds 732 solids in the Assembly (31 made, 701
purchased), and three trial imports of the electronics module left inside the Assembly add 1266 more. Measured in the
task image on 8 CPUs: the reference document's whole verify takes 583 to 617 s; a document of 3999 solids is graded
(0.9: its extra copies ruin the mass, nothing else) with 910 to 1008 s of extraction and 600 to 705 s of grading, both
within 0.8 of their limits; a document of 4001 solids scores 0.

## Ported from the UAV task, and what differs

`tests/` is the grader of `uav_cad_design_from_photos` (benchmark main `b7b55ac`), ported. 12 files are byte-identical
(`test.sh`, `fcstd_extract.py`, `env.py`, `anchor.json`, `.gitignore`, `.gitattributes`, `arm/`). Four files differ, in
65 places, each tied to one of the entries below; `public_blocklist.json` is this vehicle's own list. The grading logic
is the UAV task's, with the exceptions of entries 4, 7, 8, 9 and 10. `robotics_grader/` is not part of the port: it is
the benchmark's kit, a byte copy of `shared/robotics_grader` at benchmark main `4731ef5` (which re-vendored the kit into
every task and added `clear_agent_account`; this grader starts no agent code, so the kit's account clearing never runs
here).

1. **The reference's schema name** (`grade.py`, 1 place, and the module docstring, entry 12): `rov-cadp-reference/1`.
   The reference itself has no conical step, and its `rotors` have no propeller instance (a thruster carries its
   propeller inside the module): the flow body takes the place of the UAV task's swept disc.
2. **Numbers and the document's name** (`grader_config.json`, 4 places; `lockdown_runtime.py`, 2; `verify.py`, 1):
   `extract_timeout_s` 600 -> 1500 and `grade_timeout_s` 1080 -> 1500; `gates.max_solids` 2000 -> 4000;
   `installation.type_weights` empty, so that every purchased instance weighs the same (the UAV rule, the square root
   of the mass over 10 g capped at 3, gives 3 for every type here); `mass.extra_material` `HDPE`; the document is
   `rov.FCStd`.
3. **This platform's public files** (`verify.py`, 2 places; `public_blocklist.json`).
4. **Unchanged vendor solids that fail the validity check are kept** (`grade.py`, 7 places: `vendor_exact`, the
   statistic `invalid_purchased_kept`). An untagged closed solid that fails the validity check
   (FreeCAD's `isValid` in the extraction, or the BRepCheck recheck on the BREP that is read back) is still used when it
   is an unchanged vendor solid, that is when it registers exactly on its B-rep vertex set (`recognition.volume_rel`,
   `recognition.exact_tol_mm`) with a constituent of a pinned purchased type. The reason: the CAD kernel reports some
   vendor solids of this vehicle invalid as shipped (a circuit board of the electronics module) or once they are placed
   (the thruster housing at most poses that are not axis-aligned; a second board). Without the rule the grader dropped
   15 solids of the reference itself and found only 17 of its 20 instances (0.911). The reference document now scores
   1.0 with `invalid_purchased_kept` = 9. Tagged parts keep the rule unchanged: a made part whose solid is invalid
   (since entry 9: also with the margin) gets no credit, and an invalid untagged solid that registers on no constituent
   is left out as before.
5. **`alignment.refine_candidates` 6 -> 24** (`grader_config.json`, 1 place): every
   principal-axis start is refined. P01 is nearly four-fold symmetric; with 6 candidates a slightly imperfect P01
   could be aligned half a turn off (a quality-control document read 0.138 instead of 0.925).
6. **The grade child runs on OpenBLAS's Haswell kernels, with `OPENBLAS_L2_SIZE=256`**
   (`verify.py`, 1 place; `grade.py`, 2 places). numpy and scipy each carry their own
   OpenBLAS, which picks its kernels by the CPU it finds: SkylakeX on a CPU with 512-bit vector units, Haswell on
   others. The kernels' sums differ in the last bits, and the grader's tessellations follow them (a solid is meshed
   in its principal frame, an eigendecomposition). Measured in the task image on a CPU with
   512-bit units, under the kernel that CPU selects and under Haswell, on 18 documents (the reference, the four
   reference-solution documents and 13 quality-control documents): 490 of the reference's 576 tessellations and 558
   of the documents' 575 changed; no solid changed between a closed manifold and its convex hull; the reference and
   the reference-solution documents scored 1.0 under both; 11 of the 18 documents' scores moved, by up to 0.0006
   (single instances by up to 0.053: the fit of a light whose penetration lies near the full-credit end of its band).
   `verify.py` therefore sets `OPENBLAS_CORETYPE=Haswell` and `OPENBLAS_L2_SIZE=256` at module level and the grade
   child inherits both: the same 18 documents score bit for bit the same whether the calling environment sets
   SkylakeX, Haswell alone, both variables or nothing.
   Why the second variable: the OpenBLAS that numpy 2.5.2 to 2.5.5 bundle (0.3.34; the image has numpy 2.5.3) widens
   the double-precision block of every kernel table, the forced Haswell one included, from 256 to 512 on AMD Zen 4
   and 5 CPUs, and the Haswell kernel then overruns a fixed stack buffer once a product's inner dimension passes
   about 340 with 24 or more result rows: the process dies, or the product is wrong. `OPENBLAS_L2_SIZE=256` keeps the
   block at 256. Measured in the task image on such a CPU: under the pin alone the block is 512 and 9 of 18 probe
   cases crash (products of 64 x 512 by 512 x 64 and of 700 x 700, a 700 x 700 solve, SVDs, eigh, a 40-column
   least-squares fit); with both variables the block is 256 and all 18 are right. The grader itself never reaches the
   fault: grading the reference document, every product with an inner dimension above 256 has a 3 x 3 result (the
   covariance of two point sets in the exact registration of solids on their B-rep vertices and in the alignment,
   `kabsch`, up to 17 662 points; that of P01's surface samples for the alignment's starting frames, up to 204 869
   surface samples; recorded with an interposed BLAS library). With both variables those sums are added in blocks of
   256 instead of 512, which moves their last bits: of the 102 documents graded under the pin alone, 61 score bit for
   bit the same with both variables, the others move by at most 5.2e-12 (single instances by at most 6.6e-10), and
   the reference and the four reference-solution documents still score 1.0. On a CPU that keeps the block at 256
   under the pin alone (Intel; AMD before Zen 4) the pin alone already sums this way (from the library's source).
   `grade.py` refuses the pin without `OPENBLAS_L2_SIZE=256` before any numerics, as a task error that says so, and
   probes the settings in force once per grade in a fresh process (exact products in four number types, a 700 x 700
   solve, an SVD, a least-squares fit; about 0.2 s), so that a kernel that faults fails the run with a message
   instead of a crash in the middle of a grade.
   The FreeCAD reader runs in an environment of its own and loads no BLAS (measured on the reference document), and
   numpy's own 512-bit code paths changed nothing (the same 18 documents with them disabled), so neither is pinned.
   The pin fixes the kernels, not every difference between machines: the CAD kernel's own arithmetic may still differ
   in the last bits on another CPU (one machine was measured).
7. **Hull union** (`grade.py`, 2 places). A made part whose tessellation is no closed manifold of
   its own volume enters the volume measures as its convex hull, as in the UAV task. Its IoU now takes the union from
   that hull as well as the intersection, so the IoU is at most 1. The UAV grader took the union from the tessellation's
   own volume, so a part whose hull covers more of the reference than its own volume read an IoU above 1 and got the IoU
   term of its shape score in full (a test part, a side plate with a large bore whose mesh is left open: 1.2248
   before, 0.5096 now). The part record carries `convex_hull` and the statistic `parts_on_convex_hull` counts such
   parts; none of the 102 documents graded with the pin has a part on its convex hull (the 99 quality-control documents
   that hold a document, the two at the solid limit and the reference). Entry 10 narrows this entry: the hull serves the
   part's own IoU and nothing else.
8. **A cylinder the original does not have lowers the step beside it** (`grade.py`, 13 places; `grader_config.json`, 1
   place: `feature_surplus`). After the one-to-one matching,
   which is unchanged, a step's score is divided by one plus the number of the part's concave cylinders that lie inside
   all three zero-credit limits of the step (axis offset below 1.5 mm, radius error below 0.6 mm, axis angle below 10
   degrees) and that the true part does not have. The faces of one cylinder (split at the seam, or in two along it)
   count as one (`feature_surplus`: coaxial within 0.01 mm and 0.05 degrees, of the same radius within 0.01 mm, touching
   along the axis within 0.05 mm; two holes on one axis line, through two walls, are two). A cylinder is the true part's
   when one of its faces is matched to a scored step with a score above 0; when it is a piece of such a matched
   cylinder: it lies inside the full-credit limits of the matched face's own axis and radius (0.25 mm, 0.15 mm, 2
   degrees) and overlaps or touches it along the axis (the two halves of a bore that a non-uniform scale left as two
   cylinders a few hundredths of a millimetre apart are one hole); when it lies inside the full-credit limits of any
   concave surface of the true part (a cylinder or a cone, scored or not: it stands where the original has a surface);
   or when it is matched one-to-one, inside the zero-credit limits, to a concave cylinder of the true part that no
   scored step takes (a feature withdrawn from scoring or one never scored; not a cone, since edge breaks and drill
   points stand beside nearly every hole). A part modelled as the original, a draft modelled as a straight bore
   included, is never charged. The part record and the statistics carry `features_surplus_faces`. Why, and what it
   changes: "Known limits", H1.
9. **A tolerance margin before a tagged solid is called invalid** (`grade.py`, 3 places; `grader_config.json`, 1 place:
   `validity.tagged_vertex_margin_mm`). As in the UAV task, the solid of a tagged
   leaf is checked again (BRepCheck) on the BREP that is read back. FreeCAD writes that BREP at 15 significant digits,
   which can move a vertex just past its tolerance: a solid that FreeCAD's `isValid()` passed, as the instruction has
   the agent check, could be turned down, and a P01 turned down zeroes the document. A solid that fails is checked once
   more as a copy with every vertex tolerance raised by 1e-7 mm (geometry and volume unchanged) and is used, as that
   copy, when the copy passes (statistic `tags_valid_with_margin`); a solid that fails both ways is invalid as before
   (a made part P17 with a face turned inside out still gets no credit). A constructed example of that case is graded
   (`p01_fc_valid_grader_invalid`), and so are a P01 and three plates offset by -0.10 mm as the precision series' offset
   process first wrote them (`p01_offset_m10_fc_valid_grader_invalid` 1.0000000, `tagged_fc_valid_grader_invalid_m10`
   0.9999995).
10. **The convex hull serves the part's own IoU only** (`grade.py`, 9 places). A
    submitted solid whose tessellation is no closed manifold of its own volume is no obstacle in fit or in the flow
    clearance, and its own penetration is not measured (it is not charged; the part record says so): its hull, which
    fills the hollows of a shell, pressed into the neighbours and stood in the thrusters' flow. Statistic
    `hull_solids_not_obstacles`, and a diagnostic line that begins with `Hull:`. Measured on the reference with the
    tessellation of single parts made open (in the task image): P23 costs 0.0039 (0.0116
    before), P17 0.0024 (0.0149 before, when its two thrusters lost their clearance), P25 0.0039 (0.0069 before, when
    the DVL lost its fit). The precision document `offset_+0.50`, with P23 and P24 on their hull, scores 0.76084
    (0.74725 before).
11. **The reference's own numbers stay out of the verdict** (`verify.py`, 3 places).
    `mass_ref_g` (the reference's total mass) and `holes_total` (its count of scored features) stay in the
    grade's result and are not copied into the verdict's `metrics`, which are published with a run's log.
12. **The module docstrings in this task's words** (`grade.py`, 11 places; `verify.py`, 1 place).
    They state this grader's rules, entries 1 to 10 among them; the docstring places of entries 1, 4, 6
    and 7 are counted here.

Not changed, and therefore still worded for the UAV task: the reader's format name `uav-fcstd-extract/2` (the reader is
byte-identical) and the prefix of the grader's temporary folder. Where a docstring and this note disagree about this
task, this note and `grader_config.json` hold.

The flow body (the sixth criterion, `thruster flow clearance`): per thruster, a cylinder on the propeller axis with
the propeller's tip radius (38.1 mm) from 10 mm beyond the module's nose-cone end to 25 mm beyond its duct exit, moved
with the submitted thruster and tested against everything else, the other flow bodies included; what the reference has
itself inside a body is subtracted (`reference_overlap_exact_mm3` of each `rotors` record lists it, for information).
The 10 mm at the nose-cone end is on purpose: with 25 mm there, the bodies of a horizontal thruster and of the vertical
thruster beside it overlapped in the reference, and a 2 mm error of one thruster's place zeroed the clearance of both.

## Known limits

- **A hole the vehicle does not have is charged (H1).** The rule: after the one-to-one
  matching, the score of a reference step is divided by one plus the number of the submission's concave cylindrical
  faces that lie inside the step's three zero-credit limits and are matched to no geometry of the reference. "Geometry
  of the reference" is every hole and round wall of the true part: the scored ones, the ones withdrawn from scoring and
  the ones that were never scored, so a submission that models the true part is never charged. The instruction states
  the rule in the feature row. Why: before this rule the feature criterion did not charge a submission's unmatched
  faces, and an uncertain hole could be hedged with candidate holes at no cost. Measured before the rule, on the whole
  vehicle: every scored hole of the 20 parts with holes shifted by 0.8 mm loses 0.0712 of the
  total; the same document with two more holes per hole, one of them on the true place, scores 0.9997 (a gain of 0.0709,
  above the 0.05 at which this hedge was not to be left open; 0.028 on the nine plates alone); a direction-blind hedge
  (every through hole cut as 19 slices of its radius, each with its own offset) costs 0.0006 when the estimate is right
  and recovers 0.060 / 0.113 / 0.137 when every feature is off by 0.8 / 1.2 / 1.5 mm. With the rule, measured in the
  task image: the whole-vehicle hedge (the quality-control document `hedged_holes_whole_vehicle`, two more holes per
  scored hole of the 20 parts) scores 0.9190, 0.0099 below the same vehicle with every scored hole shifted 0.8 mm
  (0.9288, measured on the reference with the targets moved); the nine-plate hedge `hedged_holes` scores 0.9682, 0.0039
  below `holes_shifted_0.8mm`; the sliced hedge (`sliced_hedge`) scores 0.8565 with a right estimate, and its net gain
  at 0.8 / 1.2 / 1.5 mm is -0.216 / -0.162 / -0.124; the reference and the four reference-solution documents score 1.0,
  and 102 of the 107 quality-control documents graded before the rule score bit for bit as before (the 5 that changed
  are the documents with such cylinders or with a part on its convex hull, entries 8 and 10). The pieces of one bore are
  one hole: a second bore inside the full-credit limits of the bore that takes the step costs nothing, which leaves a
  small hedge. Measured in the task image on `holes_shifted_0.8mm` (every scored hole of the nine plates 0.8 mm off), a
  second bore beside each, 0.10 / 0.24 mm across its axis in a random direction, gains 0.0015 / 0.0036 (0 when the
  estimate is right); 0.50 mm across it costs 0.0149. The anisotropically scaled precision documents keep a few charged
  cylinders, the pieces of large half-round seats that the scale moved 0.27 to 0.43 mm apart: `scale_0.5` -0.0026,
  `scale_1` -0.0003, `scale_2` -0.0003 against their grades before the rule.
- **Only P01 aligns (H2).** One asymmetric error on P01 moves every other part with it. Measured on the whole
  vehicle: one outer edge of P01 moved out by 1 mm lowers the total by 0.00008 (the trimmed point-to-plane fit
  absorbs it). P01 is nearly four-fold symmetric: four of its poses lie within 0.01 mm of residual of each other,
  and the one taken is the one under which the other parts and the modules land nearest their places. It was the
  right one in all six variants tried (P01 turned by 180 degrees about each of its three principal axes, mirrored
  across two planes, one end 5 mm longer). A P01 that is modelled turned by 180 degrees about its normal or its short
  axis, or mirrored across its long axis, pulls the whole vehicle about 0.4 mm off its place and costs 0.027 to
  0.028; turned about its long axis or mirrored across its short axis it costs nothing.
- **P01 is a gate (H3).** Several solids in P01's Body, or P01 on two leaves, zero the document, as in the UAV task.
  Kept as it is: the agent can check it before it finishes, and the instruction states it first, with the way to check.
  Since entry 9 a P01 that passes that check in FreeCAD is not turned down by the 15-digit rounding of the BREP it is
  read back from.
- **Completeness is by count of solids (H4).** Deleting the electronics module's small internals costs that instance
  most of its credit. Intended; the instruction says: delete nothing.
- **The two centre-plate families (H5).** The two shapes have the same envelope and the same scored radii and counts;
  installed at each other's ends, all four instances lose shape and features (at most 4 of 25 features coincide under
  any in-plane motion), and the four vertical thrusters that rest on them lose their fit (0.012 of the total; the
  whole document scores 0.907). Intended; the photographs must show the distinguishing details.
- **A vendor overlap "corrected" by the agent (H6).** The vendor's CAD has its own interferences (the tube clamp's
  half bore against the battery tube, about 1.2 mm radially). Fit subtracts them, but a bore adjusted to the tube's
  file misses the radius band. The instruction says: model as photographed.
- **Features no photograph shows are not scored.** The black-part rule of the photo set (in frame with the axis at
  most 60 degrees off the view, an unoccluded opening at least 15 px across, an opening-to-surface contrast of 20 or
  a sharp rim, and the part's own surface neither too dark nor too bright) was measured on the finished photographs:
  12 of the 259 shape features fail it, and their 13 instance features (`reference.json` `features_withdrawn`, each
  with its reason) were taken out of scoring before the freeze; the geometry of their parts is unchanged and
  they still count in the shape. They are: the half-round seat of tube clamp half P06 (no photograph sees its axis
  within 60 degrees); three 5.4 mm holes of P01, two of P03 and two of P20, seen only in low oblique close-ups where
  the opening is as dark as the part's side; the half-open wall of P02 (radius 5 mm; one photograph, contrast 6.5);
  the half-open wall U02 of P17 and P18 (radius 20.6 mm; contrast at most 4 in 21 photographs); two 4.2 mm holes of
  P21 (sharp and contrasted, but the part's surface has more than 2 % of its pixels at 8 or below, 2.24 % with the
  openings taken out). The two 5.4 mm holes of P04 and of P19 (the twin instances of P03 and P20) pass the rule and
  stay scored. A submission that models the withdrawn features loses nothing and gains nothing in the feature
  criterion. The two holes of P21 are clear in photo_153 (contrast 31): they are withdrawn by the
  letter of the rule and stay withdrawn, which leaves unscored 2 cylindrical steps that the photographs could give.
- **Features finer than their best photograph.** The photo set's margin rule (a band's full-credit end at least 1 px
  and its zero end at least 4 px in the photographs where the quantity is best seen) holds for the medians of the
  photo classes. Feature by feature it does not: for 42 of the 310 scored features the best photograph that passes
  the black-part rule is coarser than 0.15 mm/px (0.162 to 0.725; over all 310 the finest is 0.071 and the median
  0.102), so the radius band's full-credit end is 0.21 to 0.93 px there and its zero end at most 3.7 px; for 8
  features the axis-offset band's full-credit end is below 1 px as well. The coarsest is the half-round seat of tube
  clamp half P05 (0.725 mm/px, a stage photograph: photo_073, meant as that part's end view, looks across the seat's
  axis, 92 degrees off it). No photograph is added: the set is frozen at 156, most of these features are
  holes of standard sizes, and a long arc is fitted far below a pixel.
- **Thin walls.** Nine instances are thin-walled (the two cradle clips P07 and P08, the four corner fairings P13 to P16,
  the two centre fairing caps P23 and P24, the DVL bracket P25: mean walls of 1.4 to 2.0 mm). For them the full-credit
  end of the volume overlap (0.95) stands for an error of 0.04 to 0.05 mm normal to the wall, 0.2 to 0.5 px in their
  best photographs: the photographs cannot give it (the zero end, 0.25, they can). The 40 % of the shape score that is
  the overlap is therefore not fully reachable on these nine. Counted in at that error, they leave the first condition
  of the precision budget (every face offset by 0.25 mm) standing: 0.9161 outward and 0.9375 inward (0.9181 and 0.9417
  without them; the line is 0.85).
- **Places the photographs give only roughly.** The scoring is not changed for any of them. The DVL bracket P25 and the
  DVL: the truth's bottom plate is the vendor's plate, without the two 5.5 mm holes that the bracket's installation
  guide has the user drill, so the bracket cannot be placed hole on hole; its place has to be measured (the two screw
  heads under the plate in photo_089, the side views), within the 4 mm the pose itself is known to (`NOTICE.md`, section
  4). The four ballast weights are turned by 1.014 degrees about their screw (the reference vehicle's pose): 1.0 mm
  between the two ends of a 57.89 mm block, about 3 px, and nothing in the photographs suggests it; placed square, the
  four keep their orientation credit: with the centroids brought together, a block turned by one degree still lies on
  the reference block within the surface tolerance of the orientation measure (90th percentile 0.30 mm against 0.33 mm),
  which reads as no orientation error (see "Measurement resolution"). The two round foam blocks P28 and P29 stand at an
  inferred pose (5 mm), touch nothing in the reference (`supports` is empty) and, installed, show one face of P29 in one
  photograph (photo_089); placing them by symmetry is estimated to cost 0.003 to 0.008 of the total. The four corner
  foam blocks P09 to P12 are hidden under the corner fairings when installed (seen through the vent slots only); their
  place follows from the fairing's inside and the centre plates. The gripper's extension through its clamp and its roll
  are a chosen state (2 mm) that only the photographs give.
- **Measurement resolution.** Two measures are coarser than the bands the instruction states, both in the submission's
  favour. Orientation: a module that already lies on the reference module within the surface tolerance, once its
  centroid is brought onto the reference one, reads an orientation error of zero (`Grading._orientation_error`; the
  tolerance is the module's own sampling residual plus the larger of 0.1 mm and 0.002 of its size); for the smallest
  modules this covers a turn of about one degree (the ballast weights: about 1.1 degrees), beyond the 0.5 degree
  full-credit end. Contact: the distance between a module and what it rests on is measured on the 0.2 mm tessellation
  (`mesh.deflection_mm`), so a gap of a few tenths of a millimetre can read as contact (measured: an exact gap of 0.25
  mm read as contact, where the band gives full credit to 0.1 mm), and the same tessellation can report a few cubic
  millimetres of penetration where the exact solids do not meet, far below the penetration band's full-credit end. In
  the frontier episode the two together were worth about 0.0001 of the total.
- **An orientation error of 180 degrees is a marker**, not a measured angle: the registration that measures a module's
  orientation did not land on the reference surface (an error beyond its basin), which reads as no orientation credit.
  In every case seen the true angle was beyond the band's zero end (45, 84 and 90 degrees in the frontier episode); a
  pose between 0.5 and 12 degrees whose registration failed would lose more than the written band gives.
- **The vehicle can be recognised.** Its outline and the combination of its modules in the photographs (an open
  frame of black plate, blue fairings, eight thrusters, a 4-inch enclosure with a dome and a 3-inch one) tell a
  reader who knows this class of vehicle which one it is; the registered markings of third-party components inside
  the electronics module (`NOTICE.md`, section 2) only confirm it and stay. The task does not rest on a
  hidden identity: the agent has no network, what a model remembers of the public dimensions is far coarser than the
  bands, and `public_blocklist.json` stops an unchanged public file.
- **A truncated document may still open.** FreeCAD reads the entries of a document in order: a document cut in half
  opens with the objects whose shapes it still holds and is graded as an incomplete design, not as Z2.

## The instruction and the numbers

`instruction.md`, `parts_list.md` and the eight datasheets are generated, never edited by hand: every number they
print is read from `grader_config.json`, the constants of `fcstd_extract.py`, `reference.json`, `task.toml` and the
measurements made when the task was built, the photo table from the photo set's captions, and the list of
whole-document zeros has one text per case of the table above. When a number of `grader_config.json`, a caption or
a purchased module changes, they are generated again and checked: the numbers are parsed out of the instruction
again and compared with their sources.

Statistics worth reading in a result (`metrics`): `shape_floor`; `alignment_residual_mm`, `alignment_snapped`,
`alignment_candidates_tied`; `n_solids`, `n_tagged`, `tags_invalid`, `tags_duplicated`, `tags_reassigned`;
`purchased_instances_assigned`, `invalid_purchased_kept`, `ignored_bad_solids`, `tags_valid_with_margin`;
`parts_on_convex_hull`, `hull_solids_not_obstacles`; `holes_found`, `features_surplus_faces`; `installation_poses`,
`connected_parts_fraction`, `connected_modules_fraction`, `p01_grounded`; `fit_purchased`, `fit_manufactured`;
`mass_error_rel`, `com_error_mm`, `inertia_error_rel`; `grader_item_errors`; the tool-use counts. `holes_total` and
`mass_ref_g` are numbers of the reference: they are in the grade's result (`grade.json`) only (entry 11).

## Grading a document by hand

Both steps need the payload on disk (`purchased/<part number>/<part number>.step`) and FreeCAD 1.1.4. FreeCAD writes
`__pycache__` next to every script it runs: run copies of the scripts from a work folder, never from this folder.

```bash
W=/tmp/rov_grade && mkdir -p "$W/home" && cp tests/fcstd_extract.py "$W/"
# 1. extract: every solid of the document as a BREP, the joints, model.json (the electronics module alone takes minutes)
HOME="$W/home" QT_QPA_PLATFORM=offscreen freecadcmd -c \
  "import sys; sys.argv = ['fcstd_extract.py', '/path/to/rov.FCStd', '$W/extract']; exec(open('$W/fcstd_extract.py').read())"
# 2. grade: an interpreter with the task image's Python packages (the CAD kernel bindings, numpy, scipy, trimesh,
#    manifold3d); 8 CPUs; the result is one JSON file. Loading the reference takes 3 to 4 minutes. OPENBLAS_CORETYPE and
#    OPENBLAS_L2_SIZE as verify.py sets them for the grade child (entry 6 above), else the numbers follow the kernels
#    this CPU selects; grade.py refuses the first without the second.
ARB_GRADER_CPUS=8 OPENBLAS_CORETYPE=Haswell OPENBLAS_L2_SIZE=256 \
  python3 -B tests/grade.py "$W/extract" tests/reference /path/to/payload/purchased "$W/grade.json" \
  --config tests/grader_config.json
```

`grade.json` holds `design_score`, `criteria`, `checks`, `statistics`, one record per tag (`parts`), per purchased
instance (`purchased`) and per thruster (`rotors`), and `diagnostics`. Without a document the score is 0 (`verify.py`
handles that case; `grade.py` needs an extraction).

The whole verify stage also runs on a host, as the calling account (the root-only lockdown steps are skipped). Stage a
copy of this folder and point it at a fake home that holds `submission/rov.FCStd`:

```bash
cp -r tests "$W/stage" && cd "$W/stage" && \
ALE_STAGE_DIR="$W/stage" ALE_HOME="$W/fakehome" ARB_FREECADCMD=/path/to/freecadcmd \
  ARB_PURCHASED_ROOT=/path/to/payload/purchased ARB_SNAPSHOT_ROOT="$W/snap" ALE_VERIFY_PYTHON= \
  PYTHONDONTWRITEBYTECODE=1 PATH="/path/to/that/interpreter/bin:/usr/bin:/bin" bash test.sh
```

The verdict is `verdict.local.json` in the staged folder (the kit's plain envelope: `rewards.reward`, `metrics`,
`diagnostic`). The reference document itself is built from a copy of `solution/` with
`ALE_HOME=<a home that holds the payload> ARB_FREECADCMD=/path/to/freecadcmd bash solve.sh`; it must score 1.0, and an
untouched home must score 0.0 (`ale validate` checks both). In the task image, mount the stage and the submission and
run `test.sh` as root with `--network none` on 8 CPUs.
