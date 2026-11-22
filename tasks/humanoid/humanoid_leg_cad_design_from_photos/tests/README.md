# humanoid_leg_cad_design_from_photos: grading notes for maintainers

The agent never sees this folder. It holds the grader, the hidden truth of the made parts and the numbers of the
grading; this note says where each of them is, what a reader of a result should know and how to grade a document by
hand. Sources and credits are in `NOTICE.md`.

## What is hidden, and where

| Path | What it is |
|---|---|
| `reference/reference.json` | The reference (schema `humanoid-cad-reference/1`), in the frame of the given shell file (millimetres). The 19 made parts by tag with their family (the parts of one family are rigid copies of one solid; 18 families), material, density, region of the placement floor (`segment`), rigid body and graded holes: 224 graded holes with 382 graded items in 18 parts, of which 62 items are the hidden steps of a hole that carries a threaded insert (position and direction only); 6 holes that lie wholly behind an insert are listed as `features_not_scored`. A graded item is one surface: where the authors' CAD stores one cone (a countersink) as two half-faces, the two are one item, whose radius is the largest over them, as an ordinary countersink of one face reads. The grader does not read the two steps of graded holes listed under `steps_not_scored`, each with its reason: the two flanks of a narrow notch deep inside one bore of `P01`, photographed nowhere and implied by no given file (its twin in the other bore was never graded). The 10 given component types with their solids, the solid that seeds a piece, the flags (`no_volume`: the shell; `axis_only`; `ball_centre_only`: the rod end; `light` and `sphere_like` per solid) and the points the mechanism reads (`skeleton`). The placement of each of the 19 given pieces (140 solids, 110 of them light, 13 of them balls), its region, and which items it rests on (`supports`). The mechanism block: 5 rigid bodies, 4 axes with a point and a direction each, 64 poses, 4 test points on the last body and the 9 given pieces that locate it, the 29 membership items, 2 loops with their pieces, 25 loop poses and the true output angles. The hardware that is on neither side (`unscored`, 199 units). The overlaps the leg has itself are not stored: the grader computes them from the reference geometry, pair by pair. |
| `reference/parts/Pnn.brep` | The true solid of every made part, in the reference frame: the hidden truth. Nothing in the agent's files is derived from these by anything but photography. |
| `reference/purchased_pins.json` | For every given component type: the path of its file below the payload root (STEP; the shell is a BREP file), its sha256, and which solid of the file is which constituent. The grader reads the given solids from the image's own payload (`/opt/task-assets/payload`); a file that differs from its pin is a task error, not a zero. |
| `grader_config.json` | Every number of the grader: weights, bands, the placement floor, tolerances, limits, time limits. Bands are `[full credit, zero credit]` for errors and `[zero, full]` for rising quantities. Keys whose value is a word name a rule that the code implements; another word is a task error. |
| `grade.py`, `mech.py` | The grader and its mechanism criterion. The module docstrings state the rules. FreeCAD's assembly solver is never run, here or in the reference document's build. |
| `public_blocklist.json` | sha256 of the public source files of the robot's model that could be copied into a submission (145 files, 91 distinct digests). A submission that contains one of them scores 0. |
| `anchor.json` | The benchmark kit's reference value: the design score of the reference document is 1.0, so the reward is the design score. |
| `../solution/reference/` | `oracle.json` (tags, families, the grounded component, the placement of every given piece, the rigid groups, the Revolute joints and the ball joints, one to a loop, that leave the loops open) and `parts/Pnn.brep`, byte-identical to `reference/parts/`. `../solution/solve.sh` builds the reference document from them and from the payload at solve time; no finished document is stored. |

What the agent gets is the payload only (photographs, `given/`, `parts_list.md`, `hardware_sheet.md`), fetched into the
image at build time and staged in the agent's home by `environment/setup/setup.sh`, and `instruction.md`.

## What the verifier does

1. `test.sh` closes the staged folder, then `verify.py` (as root, in a separate verifier sandbox that holds only the
   restored `/home/user/submission`) stops the agent account's processes, snapshots the submission and gates late writes
   (`lockdown_runtime.py`, `arm/`).
2. `fcstd_extract.py` runs under headless FreeCAD as the unprivileged `nobody` account with a time limit
   (`extract_timeout_s`). It reads `leg.FCStd`, writes every solid of the Assembly in world coordinates as an exact BREP
   and lists the joints with their kind, their two components, their two frames and whether they are suppressed, and the
   grounded components. It records whether a solid is closed and runs no validity check.
3. `grade.py` runs as a child process with a time limit (`grade_timeout_s`), on fixed numeric kernels (`verify.py` sets
   them). It applies the three whole-document zeros and the local rules, recognises the given pieces by their shape,
   aligns the document on the shell, assigns pieces and tagged parts to installed positions, and computes the six
   criteria and the placement floor.
4. `verify.py` applies the public-file check and writes the verdict through the kit (`robotics_grader/`): `reward =
   design_score`. `metrics` holds the checks, the six criteria and the statistics (finite numbers only; the timings go
   to stderr).

The whole document scores 0 only when it cannot be read (more than `gates.max_solids` solids and an extraction that does
not finish in time included), when it holds no closed solid, or when it cannot be aligned: the shell is missing, changed
or present more than once, and the other given pieces of the shell's rigid body (every type of `alignment.thigh_types`
but the shell: the two joint actuators and the flanged bearing) are not all there, each imported unchanged and once. A
failure of the grader itself (the reference, a payload file that does not match its pin, an exception outside a grading
item, a child that crashes or times out) fails the run as a task error and is never a 0.

## What a reader of a result should know

- **The alignment decides everything else.** `alignment_mode` 1 = the shell registered exactly on its vertex set (the
  configured mode); 3 = the fallback without the shell, a least-squares fit on the given pieces of the shell's rigid
  body; 2 = the least-squares mode with the shell, used only when the configuration asks for it. `alignment_residual_mm`
  is the residual of that step. `segment_fit_rms_mm_<region>`, `segment_fit_shift_mm_<region>` and
  `segment_fit_turn_deg_<region>` are the residual, shift and turn of a rigid fit of each region's exactly recognised
  given pieces onto the reference: they tell an error of the datum (every region shifted alike) from errors of the
  parts. Nothing is scored by them. The mode of record is `shell` (`alignment.mode`), chosen on three families of
  documents, each the reference document with given pieces moved (the table below: rewards measured under each mode).
  Under `shell` a misplaced piece costs that piece only, and a misplaced shell costs every part alike, as an equal
  offset of every made part would; under `thigh_lsq` one misplaced actuator drags the whole leg.

| Family: what is moved | `shell` | `thigh_lsq` |
|---|---|---|
| A: the shell alone, 0.5 / 1 / 3 mm | 0.9598 / 0.8556 / 0.5977 | 0.9982 / 0.9711 / 0.7948 |
| B: the `ACT-K` piece alone, 1 / 3 / 6 mm | 0.9949 / 0.9846 / 0.9789 | 0.8949 / 0.6677 / 0.3683 |
| C: the four pieces of `alignment.thigh_types`, 1 mm each (three seeds) | 0.8477 / 0.8515 / 0.8586 | 0.7753 / 0.7916 / 0.8774 |

- **The placement floor and its weighting.** `floor_thigh`, `floor_shin`, `floor_foot` and `floor_leg` are the factors
  of the three regions and of the whole leg under the configured weighting (`floor.weight`). `raw_features`,
  `raw_installation`, `raw_mechanism`, `raw_mass` and `raw_fit` are the criteria before any factor.
  `reward_floor_count`, `reward_floor_sqrt_area` and `reward_floor_area` are the design score under each of the three
  weightings of a region's parts (by number, by the square root of the surface area, by the surface area); only the
  configured one is the reward. Read them together: a low reward with high raw criteria means that a region's large
  parts are out of place, and the three rewards show how much of that is the weighting.
- **The shell's side of the fit is invisible.** The shell file is damaged: its faces enclose no usable volume (FreeCAD
  reads a negative volume), so the shell takes no part in overlaps, penetration and inertia (`no_volume`). Its mass is a
  point mass at the centre of mass its README states. A made part pushed into the shell is therefore not seen by the fit
  criterion; it is seen by the shape criterion, which grades the part where it is installed.
  `shell_samples_inside_made_share` is the share of the shell's surface samples that lie inside made parts of the
  submission, and `shell_samples_inside_made_share_reference` the same share in the reference; both are reported, not
  scored, and both are reproducible: the containment test runs in chunks of points (bounded memory), and the few points
  whose two rays disagree are cast again in a direction drawn from a generator seeded per mesh and chunk, whose shared
  state is restored after each call. A document graded twice gives the same value of every metric.
- **Readings of record.** Grading is reproducible bit for bit in the task image: a document reads the same in every run;
  on a machine without AVX-512, outside the image but with the image's FreeCAD build, Python packages and C library
  version, documents read the image's values exactly. A grade computed outside the image, against another C library, can
  differ in the last bits of the geometry kernel's numbers and so by a few millionths in the score. The readings of
  record of the maintainers' checks are therefore those taken in the task image, compared leaf by leaf with tolerance 0;
  a host's readings are compared with that host's own. `verify.py` holds numpy's own SIMD loops at their AVX2 level on
  every machine (`NPY_DISABLE_CPU_FEATURES`), beside the BLAS kernels it fixes: numpy's AVX-512 loops of a few functions
  the grader calls give other last bits.
- **Two benign changes that cost a little.** A flanged bearing turned about its own axis changes the score by at most
  1e-4: its balls and the seat they sit in overlap a little more or less at some angles (fit). The radius of a hole that
  carries a threaded insert is ignored in the Holes criterion only: a smaller hole leaves more material, which the fit
  criterion sees where it meets a given piece or another part.
- **The mechanism.** `mech_joint_<axis>`, `mech_joint_<axis>_offset_mm` and `mech_joint_<axis>_angle_deg` per axis;
  `mech_foot_median_mm`, `mech_foot_p90_mm`, `mech_foot_fit_rms_mm` and `mech_foot_fit_dropped` (pieces left out of the
  fit that locates them) for the test points; `mech_membership_mean`; `mech_chain_<loop>_p90_deg` per loop;
  `mech_closure_<tag>_mm` per bar; `mech_n_groups`, `mech_n_fixed`, `mech_n_revolute`, `mech_n_empty_groups_merged` and
  `mech_frame_mismatches` for what the document's joints were read as. The result file of `grade.py` has a `mechanism`
  record with the group that stands for each body and the reason when an axis was not found. Transmission and closure
  are computed from the given pieces' positions alone: a document without any joint can score on them.
- **Given pieces.** `purchased_instances_found`, `purchased_instances_assigned`, `purchased_instances_surplus_removed`
  (pieces beyond the leg's number, taken out of the grading), `purchased_surplus_ties_broken` (removals of equal cost
  decided by the nearest installed position), `purchased_solids_sphere_like` (bearing balls matched by volume and
  place), `purchased_solids_unattached`, `base_instances_found` and `base_grounded` for the shell.
- **Tags and holes.** `tags_invalid`, `tags_duplicated`, `tags_reassigned`; `holes_total`, `holes_found`,
  `insert_items`; `grader_item_errors` (a grading item that raised scores 0 and is counted here); the tool-use counts.
- **Mass.** The reference weighs 5629.0 g: 3652.6 g of made parts (volume times nominal density) and the given pieces at
  the masses of their data sheets. Screws and threaded inserts are on neither side.

## The instruction and the numbers

The criteria table of `instruction.md` (section "How it is graded") prints values of `grader_config.json`, and so do the
paragraphs around it (the recognition of the given solids, the whole-document zeros, the placement floor) and the
grading-weight column of the component table (`installation.type_weights`). The photo table holds the captions of the
photo set and the component table lists the given components. When one of these values, a caption or a given component
changes, `instruction.md` must be changed to match. No test in this repository compares the two; to check by hand, find
the weights, bands, shares and tolerances of the criteria table, the recognition tolerances (`recognition.volume_rel`,
`recognition.attach_mm`, `recognition.light_volume_mm3`), the solid and time limits of the whole-document zeros, the two
ends of the placement floor (`floor.p90_mm`) and the grading weights under their keys in `grader_config.json`, where
bands are written as in the table above, shares as fractions and time limits in seconds. Some shares are not in the
config and are fixed in the code: the equal split of the connection term of Installation between the made parts and the
given pieces (`grade.py`); the equal split of the test-point score between the median and the 90th percentile, and the
rule by which badly fitting pieces are left out of the fit that locates the test points (`mech.py`). The data sheets in
the payload state the points the mechanism reads (an actuator's axis and output end, the rod end's ball centre and
shoulder distance, the bore centre of the 12 mm pillow block); the shoulder distance is also
`mechanism.rod_end_seat_mm`.

## Grading a document by hand

Both steps need the payload on disk (`given/thigh_shell/thigh_shell.brep`, `given/<part number>/<part number>.step`) and
FreeCAD 1.1.4. FreeCAD writes `__pycache__` next to every script it runs: run copies of the scripts from a work folder,
never from this folder.

```bash
W="$(mktemp -d)" && mkdir -p "$W/home" && cp tests/fcstd_extract.py "$W/"
# 1. extract: every solid of the document as a BREP, the joints, model.json
HOME="$W/home" QT_QPA_PLATFORM=offscreen freecadcmd -c \
  "import sys; sys.argv = ['fcstd_extract.py', '/path/to/leg.FCStd', '$W/extract']; exec(open('$W/fcstd_extract.py').read())"
# 2. grade: an interpreter with the task image's Python packages (the CAD kernel bindings, numpy, scipy, trimesh,
#    manifold3d); the kernel settings of verify.py; the result is one JSON file
OPENBLAS_CORETYPE=Haswell OPENBLAS_L2_SIZE=256 NPY_DISABLE_CPU_FEATURES="X86_V4 AVX512_ICL AVX512_SPR" \
  ARB_GRADER_CPUS=8 python3 -B tests/grade.py \
  "$W/extract" tests/reference /path/to/payload "$W/grade.json" --config tests/grader_config.json
```

`grade.json` holds `design_score`, `criteria`, `checks`, `statistics`, one record per tag (`parts`) and per given piece
(`purchased`), the `mechanism` record and `diagnostics`. Without a document the score is 0 (`verify.py` handles that
case; `grade.py` needs an extraction).

The whole verify stage also runs on a host, as the calling account (the root-only lockdown steps are skipped). Stage a
copy of this folder and point it at a stand-in home that holds `submission/leg.FCStd`:

```bash
cp -r tests "$W/stage" && cd "$W/stage" && \
ALE_STAGE_DIR="$W/stage" ALE_HOME="$W/standin_home" ARB_FREECADCMD=/path/to/freecadcmd ARB_PURCHASED_ROOT=/path/to/payload \
  ARB_SNAPSHOT_ROOT="$W/snap" ALE_VERIFY_PYTHON= PYTHONDONTWRITEBYTECODE=1 \
  PATH="/path/to/that/interpreter/bin:/usr/bin:/bin" bash test.sh
```

The verdict is `verdict.local.json` in the staged folder (the kit's plain envelope: `rewards.reward`, `metrics`,
`diagnostic`). The reference document itself is built from a copy of `solution/` with `ALE_HOME=<a home that holds the
payload> ARB_FREECADCMD=/path/to/freecadcmd bash solve.sh`; it must score 1.0, and an untouched home must score 0.0
(`ale validate` checks both).
