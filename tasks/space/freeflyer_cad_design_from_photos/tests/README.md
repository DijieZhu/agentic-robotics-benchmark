# freeflyer_cad_design_from_photos: the grader, for maintainers

The agent re-models the cabin free-flyer in FreeCAD 1.1 from 119 photographs and saves one document,
`/home/user/submission/unit.FCStd`: an Assembly whose members carry the tags P01 to P12, with a Grounded joint on P01,
Fixed joints for P02 to P06 and six Revolute joints at the arm's and the fingers' hinges. This folder grades that
document against the robot's published appearance meshes and robot description, exactly as their makers give them.
Nothing here is in the agent's image: the engine stages `tests/` only into the separate verifier sandbox. Real names,
sources and licences are in `NOTICE.md`; the rules as the agent reads them are in `../instruction.md`.

## How a run is graded

1. `test.sh` (root, in the verifier sandbox) closes the stage folder to every other account before any interpreter
   starts, then runs `verify.py` under the image's Python (`ALE_VERIFY_PYTHON`, else `python3`).
2. `verify.py`, step 0 (`lockdown_runtime.py`, the UAV CAD task's file as it was before that task's #119: only
   `grade_child` differs, and this grader does not call it): stop and freeze every process of the agent account (the
   image's own desktop runs as that account), lock the stage, snapshot the submission into a root-only folder and gate
   late writes. A submission root that is a symbolic link or lies outside the agent home is not read.
3. The document (the snapshot's `unit.FCStd`, else the only other `*.FCStd`, with a note) is checked as a zip archive
   (central directory, every member's CRC; a cut-off or damaged file scores 0), then read by headless FreeCAD as
   `nobody` with `fcstd_extract.py` (the UAV reader byte for byte; 600 s; at most 4000 shape leaves) into one BREP per
   solid plus a JSON model of members, Labels, types and joints.
4. `grade.py` (a child process, single-threaded, numpy only, CPU features and the BLAS kernel pinned; 1080 s) runs
   `cadgrade/`: it loads and checks the standard (`makerstd/`, `reference/`), reads the members and tags, aligns the
   whole unit rigidly onto the standard, matches the two pairs (P02 / P03, P04 / P05) to their installed places, and
   scores `surface`, `joint_axes` and `structure` with the floor. The rules are stated in `cadgrade/run.py`'s
   docstring. The child's pins (`CHILD_ENV` in `verify.py`): one thread, `NPY_DISABLE_CPU_FEATURES=X86_V4,AVX512_ICL`,
   and the kernel `OPENBLAS_CORETYPE=Haswell` together with a second variable, `OPENBLAS_L2_SIZE=256`. Without it,
   the OpenBLAS bundled with numpy 2.5.2-2.5.5 enlarges the pinned kernel's block size on CPUs that report a 1 MB L2
   cache with AVX-512, and the kernel then overruns a buffer (a crash, or a wrong number with exit 0). Before the child
   starts, `verify.py` checks that rule: the kernel pin set while `OPENBLAS_L2_SIZE` is unset, empty, not a positive
   whole number or 1024 ends the run as a task error, and the grade is not started.
5. `public_model_reuse`: a submission file whose sha256 is in `public_blocklist.json` (37 published files of the robot:
   meshes, textures, description) zeroes the score.
6. The verdict goes through the benchmark kit (`robotics_grader/`, byte-identical to the repository's
   `shared/robotics_grader`): `reward` = `design_score` against `anchor.json` (value 1.0, full_at 1.0). Metrics are
   finite numbers only; timings go to stderr and to a root-only record, so two runs of one document give the same
   verdict.

A grader input that is missing or differs from its pin is a task error (the run is void), never a silent 0.

`design_score = public_model_reuse x submission_final x (0.65 surface + floor x (0.25 joint_axes + 0.10 structure))`,
floor rising linearly from 0 at `surface` 0.15 to 1 at 0.55.

## Files

| file or folder | what it is |
|---|---|
| `test.sh`, `verify.py` | the verify stage (above) |
| `lockdown_runtime.py`, `arm/` | the lockdown: `arm/` (the stdlib stage helpers it imports) are byte copies from the UAV CAD task; `lockdown_runtime.py` is that task's file as of before its #119 (`grade_child` differs and is not called here) |
| `fcstd_extract.py` | the document reader, the UAV CAD task's file byte for byte |
| `grade.py`, `cadgrade/` | the grade child: `reference.py` (the standard and the graded samples), `submission.py` (members, tags, solids, meshing), `align.py` (whole-unit alignment), `geom.py` (distances), `score.py` (surface, joint axes, structure), `run.py` (the rules), `util.py` |
| `grader_config.json` | every number of the grader (groups below) |
| `anchor.json` | the reward anchor: design_score, higher is better, 1.0 = the reference build |
| `makerstd/` | the reader of the makers' files (stdlib + numpy): COLLADA meshes, the description texts, the placement, the sample codec; `solution/makerstd/` is the same bytes |
| `reference/maker/` | the standard: 10 appearance meshes (`meshes/`, three of them xz-compressed), the two description files (`description/`), the launch file and the gripper plugin source the placement reads two values from (`context/`), and `pins.json` (sha256 of every file as published, of the uncompressed bytes for the xz files) |
| `reference/placement.json` | each tag's pose in configuration Z and the six joint axes, derived from the pinned texts (every number with its file and line); `verify.py` derives it again at every run and compares |
| `reference/seen.json`, `reference/seen/*.bin` | the graded samples: per mesh file, the surface points seen in at least three of the delivered photographs |
| `public_blocklist.json` | the sha256 of the 37 published files a submission must not contain |
| `NOTICE.md` | sources, licences, real names, what the project authored |
| `robotics_grader/` | the benchmark kit (verdict, rewards, stage helpers) |
| `.gitattributes`, `.gitignore` | byte-exact files never converted at checkout; the local verdict and caches never committed |

## grader_config.json, group by group

- `document`, `extract_timeout_s` 600, `grade_timeout_s` 1080, `verifier_timeout_s` 3600 (= task.toml), `grader_cpus` 1.
- `weights` surface 0.65 / joint_axes 0.25 / structure 0.10; `floor` the ramp [0.15, 0.55] of surface multiplying the
  other two.
- `surface`: the pooled 90th percentile, full credit to 0.6 mm, none from 6 mm. `part_weights`: P01-P03 3, P04-P05
  0.5, P06-P08 1.5, P09-P12 0.75 (17.5 in all). `pairs`: the two interchangeable pairs.
- `joint_axes`: offset band [0.3, 3] mm weighted 0.6, angle band [0.5, 5] deg weighted 0.4; `revolute_pairs`: the six
  hinges, by the tags on either side. `grounded` P01 and `fixed_parts` P02-P06 for `structure`.
- `sampling`: the seed, the dense density on the standard (2 / mm2) and the query density on the submission
  (0.25 / mm2), the tangent disk, per-tag minima and maxima. `gates`: at most 6000 solids, minimum volume, validity
  re-checked. `mesh`: the meshing of submitted solids (deflection, retries, weld, volume check, hole filling).
- `alignment` (the staged ICP: starts, subsets, trimming, bounds, the tie band and the placement tie-break) and
  `search` (the nearest-point search caps; distances are exact up to them, and a tag whose 90th percentile depends
  on points beyond them reports a lower bound, which is then at or past the zero line, so it scores exactly 0).
- `reference`: where the standard, the placement and the graded samples are.

Changing any value changes the task: the bands and the floor were fixed before the first model episode.

## The standard and its pins

The standard is the makers' files as published, converted by nothing: the reader takes the meshes' triangles and node
transforms, and the poses from the description in configuration Z (the arm stowed, the gripper closed as the gripper
plugin's start values give it). `makerstd.standard.load` checks every file against `reference/maker/pins.json`
(decompressing the xz files), derives `placement.json` again from the four pinned texts and compares, and only then
reads the meshes. Odd things of the makers' files are kept as they are and listed in `NOTICE.md` (the arm base about
14 mm off the core module, one propulsion module turned by 3.14 rather than pi, stray triangles in the bumper file,
zero-area triangles in the gripper's proximal pieces).

## How the graded samples were made

`reference/seen.json` states the rule and the format. Per mesh file, points are drawn uniformly by area at 0.25 / mm2
(seeded, quantised by `makerstd/seen_codec.py`); a point counts as seen in a photograph when it is inside the frame,
is the first surface on the ray through the centre of its pixel within 0.25 mm (a z-buffer at the photograph's own
resolution, every part instance, the side skins and the bench props occluding), and is viewed at least 20 deg off its
tangent plane. Points seen in at least three delivered photographs are graded; the two instances of a pair pool their
sightings. The camera table is pinned by `cameras_sha256`, the photo set by `payload_manifest_sha256`. The tables are
an output of the photo set: if a photograph changes, they are made again and the task is validated again.

## The image and its assets

The image is `environment/` (less `setup/`), built FROM the benchmark's container base: the UAV CAD task's recipe
(venv, FreeCAD 1.1.4 from its sha256-pinned AppImage, the openbox desktop at 1440x900 running FreeCAD only, the
FreeCAD MCP server), an import check of the grading packages, and the photo set. `environment/assets.sha256` lists the
119 photographs (`photos/photo_NNN.jpg`, the same path in the dataset's folder `tasks/space/<task>/payload/` and in the
agent home); `environment/stage_assets.py` puts exactly those files into `/opt/task-assets/payload`, each checked
against its sha256, and `setup.sh` copies them into `/home/user/photos`. The build argument `FFCAD_ASSETS` chooses where
they come from:

- `hub`: `fetch_assets.py` (the repository's canonical copy) downloads them from the asset dataset at the commit the
  list pins; refused while the list pins the placeholder commit (40 zeros);
- `local`: the same files placed in the build context at `environment/local_assets/<path>`, for builds before the
  upload (never committed);
- `auto` (the default): local when every listed file is there, else hub.

## After the upload

While `assets.sha256` pins the placeholder revision the task builds only with `local_assets/`. Once the photo set is
in the dataset:

1. upload `photos/` to `tasks/space/freeflyer_cad_design_from_photos/payload/` of the dataset named in the repository's
   `task-assets.toml` at that moment (read it right before uploading);
2. pin the list: `python3 scripts/task_assets.py list tasks/space/freeflyer_cad_design_from_photos <payload folder>
   --revision <dataset commit>`; the file's sha256 lines must not change, only `# revision:`;
3. remove `environment/local_assets/` from every copy of the task, run `python3 environment/fetch_assets.py
   environment/assets.sha256 <empty folder>` once to see the files arrive, then `scripts/task_assets.py check`;
4. validate again with `--set desktop_resolution=1440x900` (the engine starts desktops at 1024x768 otherwise) and
   check `resources.effective.desktop_resolution` in both episodes' `lock.json`: untouched 0.0, oracle 1.0.

## Measured (fresh containers of this image, no network, 2026-11-18)

- The reference build (`solution/solve.sh`): 1.5-3.5 min; the document 36.7 MB, 12 members, 2193 solids, all valid
  and closed. Its grade: reward 1.0, every tag's 90th percentile 0.20-0.24 mm (full credit to 0.6 mm), every hinge
  within 0.004 mm and 0.002 deg; verify 131 s on 8 CPUs (read 54 s, grade 76 s), container peak 3.1 GB.
- The slowest graded submission measured: 182 s on 1 CPU. An empty submission folder scores 0.0 in under a second.
- Grading is deterministic: the same document graded twice gives the same metrics bit for bit, and so do three CPU
  families (Zen 3, Zen 4 and Zen 5, 2026-11-19, with the pins above): the reference build and a submission of convex
  hulls give byte-identical result files on all three.
