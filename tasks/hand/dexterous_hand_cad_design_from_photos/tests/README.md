# dexterous_hand_cad_design_from_photos: grading notes for maintainers

The agent never sees this folder. It holds the grader, the hidden truth of the hand and every number of the grading.
This note says what each file is, how the verify stage runs, how to run it by hand, how the reference is rebuilt and
what the photographs locate. Sources and credits are in `NOTICE.md`.

## The files of this folder

| Path | What it is |
|---|---|
| `test.sh` | Entry of the verify stage. Closes the staged folder to every other account, fixes one math thread per process and starts `verify.py`. |
| `verify.py` | The verify stage: lockdown, the document reader, the grade child, the public-file check, the verdict. Its module docstring lists the steps. |
| `lockdown_runtime.py`, `arm/` | The lockdown shared with the sibling CAD tasks (byte copies): stopping and freezing the agent account's processes, the root-owned snapshot of the submission, the gate against late writes, the launch of the reader and of the grade child. |
| `fcstd_extract.py` | Runs under headless FreeCAD as the unprivileged `nobody` account. Reads `hand.FCStd` as data: every solid of the Assembly in world coordinates as an exact B-rep, the labels, the joints with their frames, angles and limits. No validity check of a solid. |
| `grade.py` | The grade child. The rigid part of the grading: recognition of the given pieces, the fit of every made part onto the true part of its shape, the alignment on the palm, the criteria shape, holes and assembly, the shape floor and the weighted sum. Its module docstring states the rules. |
| `handtree.py` | The mechanism part of the grading, called by `grade.py`: rigid groups, the palm, hinges, digits, links, the servo that drives a hinge, limits, closed chains, the poses; the criteria kinematics, mass, fit and interference per digit, and the digit floors. |
| `hand_task.py` | The reader of the mechanism's truth (`reference.json`, block `mechanism`) and the names the mechanism module and the reference builder share. |
| `grader_config.json` | Every number of the grader: weights, bands, floors, tolerances, sample counts, time limits. Bands are `[full credit, zero credit]` for errors and `[zero, full]` for rising quantities. |
| `anchor.json` | The benchmark kit's reference value: the design score of the reference document is 1.0, so the reward is the design score. |
| `public_blocklist.json` | sha256 of the public source files of the hand's model. A submission folder that holds one of them scores 0. |
| `reference/reference.json` | The reference (schema `hand-cad-reference/1`): the 27 shapes of the made parts (`kinds` K01 to K27, each with its tags, material, density, size class, weight, mesh and graded holes), the 52 tagged parts with their poses at the rest pose, the 8 given types and the 53 placements of their pieces, the palm's members, the mass per link, and the mechanism (schema `hand-cad-mechanism/1`: 20 links, 22 hinges of which 13 are independent with their true limits, 3 closed chains, 4 fingertip points). Every hole and every hinge carries the mark `located`, every tagged part and every placement the mark `place_located` (the last section). |
| `reference/parts/Knn.brep` | The true solid of every shape in its own frame: the hidden truth. Three shapes also have `Knn.mesh.npz`, a closed triangle mesh of the same solid that the grader uses for volume measures where the kernel's own tessellation of that shape is not closed. |
| `reference/samples.npz` | Surface samples of every shape (points, normals) with the mark `located` per sample: only located surface enters the shape criterion's distance and the gap of the fit criterion. |
| `reference/pairs.json` | The contacts and overlaps the hand has itself, pair by pair (schema `hand-reference-pairs/1`): exempt in the fit and interference criteria. |
| `reference/purchased_pins.json` | For every given type the path of its STEP file below the payload root and its sha256. The grader reads the given solids from the image's own payload (`/opt/task-assets/payload`); a file that differs from its pin is a task error, not a zero. |
| `robotics_grader/` | The benchmark's grading kit, a byte copy of the repository's `shared/robotics_grader`. |
| `README.md`, `NOTICE.md` | This note; sources, credits and the licence text. |

Outside this folder: `../solution/` holds `solve.sh`, `build_oracle.py` and `reference/` (`oracle.json`, the plan of the
reference document, and `parts/Knn.brep`, byte-identical to the `.brep` files of `reference/parts/`). The agent gets
the payload only (the photographs, `purchased/`, `parts_list.md`), fetched into the image at build time and staged in
the agent's home by `../environment/setup/setup.sh`, and `../instruction.md`.

## What the verify stage does

It runs as root in a separate verifier sandbox, started fresh from the task image, that holds only the restored
`/home/user/submission`. No code of the agent runs: the document is data that FreeCAD reads.

1. `test.sh` closes the staged folder; `verify.py` records when the upload of this folder began, stops and freezes the
   agent account's processes (in the verifier these are the image's own desktop), takes a root-owned snapshot of the
   submission and sets the score to 0 when an entry of the submission was written after the upload began.
2. `fcstd_extract.py` reads the snapshot's `hand.FCStd` under `freecadcmd` in a private work folder, with the time limit
   `extract_timeout_s`.
3. `grade.py` runs as a child process with the time limit `grade_timeout_s` on `grader_cpus` CPUs and one math thread,
   with the numeric kernels of its linear algebra fixed (the two `OPENBLAS_` variables set in `verify.py`), so that a
   score does not depend on the CPU of the grading machine. It grades the extraction against `reference/` and the
   payload's given STEP files and calls `handtree.py` for the mechanism.
4. `verify.py` applies the public-file check and writes the verdict through the kit: `reward = design_score`. The
   metrics are finite numbers; the diagnostic is at most 2000 characters.

The poses of the fingertip and interference items are drawn afresh at every grading. A verdict records the stream in
its metrics `pose_seed_hi` and `pose_seed_lo`; `ARB_SEEDS_PIN=<hi>:<lo>` in the environment reproduces it.

## Zeros and task errors

The whole document scores 0 only when it is missing or cannot be read (no `hand.FCStd` that is a regular file, a
document that FreeCAD cannot open, no Assembly with components, more than `gates.max_solids` solids, more than 4000
objects in the Assembly's tree (`MAX_LEAVES` of `fcstd_extract.py`), or a reader that does not finish in time), when
its Assembly holds no closed solid, when it holds no usable palm (no palm group, or none whose members the alignment
can read), when the submission root is a symbolic link or resolves outside the agent's home, when the submission
folder holds a file of `public_blocklist.json`, or when the submission was written after the session. The
instruction lists the same cases in one paragraph. Everything else is local: it costs the part, the hinge or the
digit concerned and is reported in the metrics.

A failure of the grader itself ends the run as a task error and is never a 0: the reference or a payload file that
cannot be used, an exception outside a grading item, a grade child that crashes or runs out of time, a process of the
agent account that cannot be stopped. A grading item that fails on a document scores 0 for that item and is counted
in `grader_item_errors`.

## Reading a result

`metrics` holds, beside `design_score` and the two checks `cad_valid` and `public_model_reuse`: the seven criteria after
the floors (`shape`, `kinematic`, `features`, `assembly`, `mass`, `fit`, `interference`) and before them
(`<criterion>_raw`); `shape_floor`, `digit_fk_<digit>` and `digit_valid_<digit>` for `index`, `middle`, `ring` and
`thumb`; the alignment (`alignment_members` and `alignment_inliers`: the members are the given pieces of the alignment's
types and the made parts of the palm's shapes that sit in a palm group, each of a type or shape with at least one
located installed place in the reference; the inliers are the members that the one-to-one assignment within their type
or shape pairs with a located place and that lie within `alignment.inlier_mm` of it after the fit, so that a type or
shape with fewer located places than submitted pieces leaves its surplus pieces counted as members but never paired or
fitted: the reference document reads 18 members and 17 inliers, because one of its three spacers of one shape, P50 to
P52, has no located place; `alignment_residual_mm`, `alignment_runner_up_ratio`); what was recognised (`n_tagged`,
`tags_invalid`, `tags_duplicated`, `parts_on_hull`,
`purchased_instances_found`, `purchased_instances_assigned`, `purchased_instances_left_over`); the mechanism as read
(`palm_groups`, `palm_split`, `digit_candidates`, `unassigned_candidates`, `first_hinge_surplus`, per hinge the three
flags `hinge_<digit>_<hinge>_present`, `hinge_<digit>_<hinge>_missing` and `hinge_<digit>_<hinge>_undriven`,
`hinges_disagreeing` (the sets of joints between two groups that disagree: their hinge is missing) and
`edges_disagreeing` (the joints beyond the first in those sets), `links_tied`, `kinematic_<digit>`,
`loop_closed_<finger>`, `pin_marked`, `servo_unfixed`, `floating_groups`, `tip_modules_surplus`, `joints_nonfinite`);
the holes (`hole_steps_total` and `hole_steps_found`: the graded steps of the reference and those of them that score
above one half; `hole_steps_unread`: the submitted steps left over where a hole has more eligible walls than the true
hole has steps; `hole_steps_extra`: the submitted steps counted as extras, each of which adds to the divisor of its
part's hole score; `hole_steps_at_unlocated`: the submitted steps paired with a step of a true hole that is not
located, neither credited nor counted); what the reference marks not located (`hinges_not_located` and
`hinge_items_not_graded`: the hinges and the kinematic items they take out, both read from the reference and the same
in every verdict; `placement_items_not_graded` and `installation_items_not_graded`: the tagged parts and the given
placements whose installed place is not located); `fingertip_median_mm_<digit>` and `fingertip_p90_mm_<digit>` (-1
for a digit that is not graded); `submission_final`; the exit codes of the two steps, `extract_exit` and
`grade_child_exit` (-1 after a time-out, -2 when that step did not run); the tool-use counts.

Every verdict carries the same metric names, whatever the document: 220 for this reference on a host and one more in
the task image (the time the stage's upload began); the engine adds its exact reward. A statistic that was not
measured stands at its default: a count or a flag 0, `kinematic_<digit>` 0, `alignment_residual_mm` -1 without an
alignment. The diagnostic's grade line names the parts that are graded on their convex hull. The names of hinges and
links in the metrics are the grader's own and appear in no file the agent gets.

## Running the verify stage by hand

The stage runs on a host as the calling account; the root-only lockdown steps are then skipped. It needs FreeCAD
1.1.4, an interpreter with the task image's Python packages (the CAD kernel bindings, numpy, scipy, trimesh,
manifold3d) and the payload on disk (`purchased/<part number>/<part number>.step`). FreeCAD writes `__pycache__` next
to every script it runs, so stage a copy of this folder and never run from the folder itself; `ALE_HOME` is a folder
that stands for the agent's home and holds `submission/`.

```bash
W=/tmp/hand_grade && mkdir -p "$W/agent/submission" "$W/tmp" && cp -r tests "$W/stage"
cp /path/to/hand.FCStd "$W/agent/submission/hand.FCStd"
cd "$W/stage" && ALE_STAGE_DIR="$W/stage" ALE_HOME="$W/agent" ARB_FREECADCMD=/path/to/freecadcmd \
  ARB_PURCHASED_ROOT=/path/to/payload ARB_SNAPSHOT_ROOT="$W/snap" ALE_VERIFY_PYTHON= TMPDIR="$W/tmp" \
  PYTHONDONTWRITEBYTECODE=1 PATH="/path/to/that/interpreter/bin:/usr/bin:/bin" bash test.sh
```

The verdict is `verdict.local.json` in the staged folder (the kit's plain envelope: `rewards.reward`, `metrics`,
`diagnostic`); the timings go to stderr. Without a document the reward is 0. The reference document takes about five
minutes to verify in the task image with 8 CPUs, most of it in the mechanism module (the true poses and the
collisions), and several times longer on a loaded host.

`grade.py` can also be run alone on an extraction:

```bash
ARB_GRADER_CPUS=8 python3 -B tests/grade.py /path/to/extraction tests/reference /path/to/payload /tmp/grade.json \
  --config tests/grader_config.json
```

The result holds `design_score`, `criteria`, `criteria_raw`, `checks`, `statistics`, one record per tag (`parts`), per
given piece (`purchased`) and per digit (`digits`), and `diagnostics`. A part's record carries `hole_steps_unread` and
`hole_steps_extra` when it has such steps, and `place_located: false` in place of a placement score when its installed
place is not located; a given piece's record carries `place_located: false` in place of an installation score in the
same case. A digit's `items` hold `null` for a component (axes, lengths, range) that is left with nothing to grade.
The diagnostics name the parts with extra steps and the parts with steps left over.

## How the reference is rebuilt

The reference document is not stored. `../solution/solve.sh` builds it at solve time from `../solution/reference/` and
the payload in the agent's home: a PartDesign Body per tag around the true solid of its shape, every given STEP
imported unchanged and placed, one Assembly with the Grounded, Fixed and Revolute joints and the limits of the
independent hinges, saved at the rest pose. The builder checks that nothing moves on recompute, solve, save and reopen
and writes its report beside the document. By hand, from a copy of `solution/`:

```bash
ALE_HOME=/path/to/a/folder/that/holds/the/payload ARB_FREECADCMD=/path/to/freecadcmd bash solve.sh
```

The document must score 1.0 on every criterion, and an untouched home must score 0.0 (`ale validate` checks both).

The reference data of `reference/` and `../solution/reference/` are built from the release CAD by the task's
development tools, which are not part of this repository, in this order: one solid per part family, taken out of the
release CAD entity for entity; tags and shapes (the shapes ordered by the size of their bounding box, largest first);
the given types in their documented frames; the poses of every part and piece; the graded holes with their classes;
the hinges, the closed chains and the fingertip points; the limits of the independent hinges by collision sweeps of
the true solids; the hand's own contacts and overlaps; the surface samples; the masses; last, the located marks, from
the delivered photographs. Two builds from scratch give the same bytes. After any rebuild these must hold before the
task is used:

- the `.brep` files of `reference/parts/` and `../solution/reference/parts/` are byte-identical
  (`for f in tests/reference/parts/*.brep; do cmp "$f" "solution/reference/parts/$(basename "$f")"; done`); the three
  `.mesh.npz` files are in `reference/parts/` only;
- every given STEP of the payload has the sha256 that `reference/purchased_pins.json` records;
- the reference document validates at 1.0 and an untouched home at 0.0;
- `../instruction.md` is generated again: its tables and numbers print `grader_config.json`, the captions of the photo
  set, the parts list and the reference's size classes, and no test in this repository compares them.

## The photographs and what they locate

The payload's photographs are the delivered set: the 363 photographs of the final photo plan, rendered at full
quality and checked by the photo audit. `../environment/assets.sha256` holds their digests; its revision line is a
placeholder until the asset dataset is uploaded. The captions and the scale references of the photo table in
`../instruction.md` are those of the delivered set, where the column of scale references follows what each
photograph shows.

What the photographs do not locate is not graded, and the reference carries the marks that say so. They were
computed from the delivered photographs and their recorded cameras, with the true geometry, before any submission:

- `reference/samples.npz`, `located` per surface sample: the surface is seen without obstruction in two photographs
  from clearly different directions, or in one photograph with the face's boundary in view, at a resolution that
  fits the bands. A reading test on the delivered set found the edges of each class (outlines, inner edges) readable
  in some kinds of photograph and not in others: a sample at an edge that the photographs locate as an edge only
  through kinds of photograph in which its class did not read loses its mark; a sample at an edge that no photograph
  locates as an edge keeps the mark of the surface rule above. A sample that is not located enters neither the shape
  distance nor the fit's gap.
- `reference/reference.json`, `located` per hole (`kinds.<shape>.features`): the hole's mouth is seen in enough
  photographs. A hole that is not located has no hole item; a submitted wall at it is neither credited nor an extra.
  Every hole and step stays in the reference, so that the grader can tell such a wall from an extra.
- `located` per hinge (`mechanism.hinges`): each end of the axis is seen from two directions, or the hinge turns on
  a pin whose bores in both parts it joins are located holes of parts whose places are located. A hinge that is not
  located has no axis item, no range item and no key length that ends at it.
- `place_located` per tagged part (`parts`) and per given placement (`placements`): the installed part or piece is
  seen in two photographs of the assembled hand or of a stage, or its place follows from located bores or from its
  counterpart in the finger that was taken apart. A place that is not located has no placement or installation item
  and no piece of the submission is paired with it in the alignment (a submitted piece of its type or shape is still
  counted among the alignment's members; see "Reading a result").

A mark that is absent counts as located; a mark that is neither true nor false is a task error. The marks of this
reference leave out 4 of the 202 holes, none of the 22 hinges and the installed place of 1 of the 52 tagged parts
and of none of the 53 given pieces; the metrics named under "Reading a result" count what a verdict left out. After
a new photo set or a new audit the marks are written again and the reference document is validated again; the
instruction states the rule and none of these counts.
