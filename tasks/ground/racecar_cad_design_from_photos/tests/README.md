# racecar_cad_design_from_photos: grading notes for maintainers

The agent never sees this folder. It holds the grader, the hidden truth of the made parts and the numbers of the
grading; this note says where each of them is and how to grade a document by hand. Sources and credits are in
`NOTICE.md`.

## What is hidden, and where

| Path | What it is |
|---|---|
| `reference/reference.json` | The reference (schema `racecar-cad-reference/1`): the 23 made parts by tag with their family (parts of one family are rigid copies of one solid), material, density and graded holes; the 16 given component types with their solids, the solid that seeds an instance, the spin-free flag, the flag of the types that are the same end for end (`flip_symmetric`: either end up is the same placement) and the light flag; the placement of every given piece in the car frame; which items each piece rests on (`supports`); the hardware that is on neither side (`unscored`). The overlaps the car has itself are not stored: the grader computes them from the reference geometry, pair by pair. |
| `reference/parts/Pnn.brep` | The true solid of every made part, in the car frame: the hidden truth. Nothing in the agent's files is derived from these by anything but photography. |
| `reference/purchased_pins.json` | For every given component type: the path of its STEP file below the payload root, its sha256, and which solid of the file is which constituent. The grader reads the given solids from the image's own payload (`/opt/task-assets/payload`); a file that differs from its pin is a task error, not a zero. |
| `grader_config.json` | Every number of the grader: weights, bands, the shape floor, tolerances, limits, time limits. Bands are `[full credit, zero credit]` for errors and `[zero, full]` for rising quantities. |
| `public_blocklist.json` | sha256 of the public source files of the car's model. A submission that contains one of them scores 0. |
| `anchor.json` | The benchmark kit's reference value: the design score of the reference document is 1.0, so the reward is the design score. |
| `../solution/reference/` | `oracle.json` (tags, families, the grounded component, the placement of every given piece) and `parts/Pnn.brep`, byte-identical to `reference/parts/`. `../solution/solve.sh` builds the reference document from them and from the payload at solve time; no finished document is stored. |

What the agent gets is the payload only (photographs, `base/`, `purchased/`, `parts_list.md`), fetched into the image at
build time and staged in the agent's home by `environment/setup/setup.sh`, and `instruction.md`.

## What the verifier does

1. `test.sh` closes the staged folder, then `verify.py` (as root, in a separate verifier sandbox that holds only the
   restored `/home/user/submission`) stops the agent account's processes, snapshots the submission and gates late
   writes (`lockdown_runtime.py`, `arm/`).
2. `fcstd_extract.py` runs under headless FreeCAD as the unprivileged `nobody` account with a time limit
   (`extract_timeout_s`). It reads `racecar.FCStd`, writes every solid of the Assembly in world coordinates as an exact
   BREP and lists the joints. It records whether a solid is closed and runs no validity check: the given solids of
   this car fail FreeCAD's and the CAD kernel's validity check after a rigid move or a BREP round trip while being
   closed and usable.
3. `grade.py` runs as a child process with a time limit (`grade_timeout_s`). Its module docstring states the rules:
   what is graded, the three whole-document zeros, the local rules, the alignment on the base assembly, the
   recognition of the given pieces, the assignment inside families, the five criteria and the shape floor.
4. `verify.py` applies the public-file check and writes the verdict through the kit (`robotics_grader/`):
   `reward = design_score`. `metrics` holds the checks, the five criteria and the statistics (finite numbers only; the
   timings go to stderr), the diagnostic at most 2000 characters.

The whole document scores 0 only when it cannot be read (more than `gates.max_solids` solids and an extraction that
does not finish in time included), when it holds no closed solid, or when the tub of the base assembly is missing or
present more than once. A failure of the grader itself (the reference, a payload file that does not match its pin, an
exception outside a grading item, a child that crashes or times out) fails the run as a task error and is never a 0.

## The instruction and the numbers

The criteria table of `instruction.md` (section "How it is graded") prints values of `grader_config.json`, and so do the
paragraphs around it (the recognition of the given solids, the whole-document zeros, the shape floor) and the
grading-weight column of the component table (`installation.type_weights`). The photo table holds the captions of the
photo set and the component table lists the given components. When one of these values, a caption or a given component
changes, `instruction.md` must be changed to match. No test in this repository compares the two; to check by hand, find
the weights, bands, shares and tolerances of the criteria table, the two recognition tolerances
(`recognition.volume_rel`, `recognition.attach_mm`), the solid and time limits of the whole-document zeros, the two ends
of the shape floor and the grading weights under their keys in `grader_config.json`, where bands are written as in the
table above, shares as fractions and time limits in seconds. One share is not in the config: the equal split of the
connection term of Installation between the made parts and the given pieces is fixed in `grade.py`.

Statistics worth reading in a result (`metrics`): `shape_floor` and `raw_features`, `raw_installation`, `raw_mass`,
`raw_fit` (the four criteria before the floor); `alignment_mode` (1 = the tub registered exactly on its vertex set),
`alignment_residual_mm`, `base_instances_found`, `base_completeness`, `base_grounded`; `purchased_instances_found`,
`purchased_instances_assigned`, `purchased_solids_unattached`; `holes_total`, `holes_found`; `tags_invalid`,
`tags_duplicated`, `tags_reassigned`; `connected_parts_fraction`, `connected_modules_fraction`;
`grader_item_errors`; the tool-use counts.

## Grading a document by hand

Both steps need the payload on disk (`base/base_assembly.step`, `purchased/<part number>/<part number>.step`) and
FreeCAD 1.1.4. FreeCAD writes `__pycache__` next to every script it runs: run copies of the scripts from a work
folder, never from this folder.

```bash
W=/tmp/racecar_grade && mkdir -p "$W/home" && cp tests/fcstd_extract.py "$W/"
# 1. extract: every solid of the document as a BREP, the joints, model.json
HOME="$W/home" QT_QPA_PLATFORM=offscreen freecadcmd -c \
  "import sys; sys.argv = ['fcstd_extract.py', '/path/to/racecar.FCStd', '$W/extract']; exec(open('$W/fcstd_extract.py').read())"
# 2. grade: an interpreter with the task image's Python packages (the CAD kernel bindings, numpy, scipy, trimesh,
#    manifold3d); 8 CPUs; the result is one JSON file
ARB_GRADER_CPUS=8 python3 -B tests/grade.py "$W/extract" tests/reference /path/to/payload "$W/grade.json" \
  --config tests/grader_config.json
```

`grade.json` holds `design_score`, `criteria`, `checks`, `statistics`, one record per tag (`parts`) and per given piece
(`purchased`), and `diagnostics`. Without a document the score is 0 (`verify.py` handles that case; `grade.py` needs an
extraction).

The whole verify stage also runs on a host, as the calling account (the root-only lockdown steps are skipped). Stage a
copy of this folder and point it at a fake home that holds `submission/racecar.FCStd`:

```bash
cp -r tests "$W/stage" && cd "$W/stage" && \
ALE_STAGE_DIR="$W/stage" ALE_HOME="$W/fakehome" ARB_FREECADCMD=/path/to/freecadcmd ARB_PURCHASED_ROOT=/path/to/payload \
  ARB_SNAPSHOT_ROOT="$W/snap" ALE_VERIFY_PYTHON= PYTHONDONTWRITEBYTECODE=1 \
  PATH="/path/to/that/interpreter/bin:/usr/bin:/bin" bash test.sh
```

The verdict is `verdict.local.json` in the staged folder (the kit's plain envelope: `rewards.reward`, `metrics`,
`diagnostic`). The reference document itself is built from a copy of `solution/` with
`ALE_HOME=<a home that holds the payload> ARB_FREECADCMD=/path/to/freecadcmd bash solve.sh`; it must score 1.0, and an
untouched home must score 0.0 (`ale validate` checks both).
