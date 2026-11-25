# tests/review/: maintainer tools (never run by verify.py)

Hidden side only, like the rest of tests/. Nothing in this folder is imported or executed by `tests/test.sh` or
`tests/verify.py`; nothing here reaches the agent, the payload or the practice ground. Written for the grader's own QC
on 2026-11-17 (task version 1.0); the QC scripts, the generators `w2gen` and `w3gen` and the degenerate submissions
D0-D9 are version 1.1's (2026-11-19). Version 2.0 added the generator of the heaps (`sitegen/w4gen/`), replaced the
calibration site's generator by the one that places the two heaps, added the QC of the heaps and of the plates (Q7,
Q8, Q9) and the degenerate submission D10, and brought the other QC scripts to site family W4 and the plate
backend.

| Folder | What | Source of the bytes |
|---|---|---|
| `variants/` | the maintainer variants of the privileged reference (`qc_reference.py`) and their plans with expected values written before the runs (`plans.json`): QC Q2 / Q3 through the real verify.py. The plans are version 1.0's (credit 0.33 / 0.66 m, weights 0.15 / 0.25 / 0.60); the plans in use, with version 1.1's credit and weights, which version 2.0 keeps, are in `qc/variants/` (copies of the maintainers' files) | written here (originals) |
| `qc/` | the QC scripts: `qc_verify.py` (runs the real verify.py on the frozen hidden set: oracle, oracle without / with a wrong token, a program folder, or a variant through the stage's `reference_dir`), `check_runs.py` (recomputes every episode's score from the cell's record by its own reading of the scoring rules and judges the runs), `q1_frames.py` (a fixed 20-frame sequence for the two-process frame check), `q4_interface.py` (bad arguments, reply whitelist, mission without layout, refusals, privileged calls, the pose switch, no debrief), `q6_recompute.py` (every G predicate recomputed from the stored site packages, the heaps' on the compiled model, and the reference's admission rows), `variant_inproc.py` (a development check of one variant in one cell), `q5_v11_extra.py` (the hidden quantities against the agent-visible files and the image's file list), `d9_practice_debrief.py` (D9 on the practice path with the real debrief), `q7_plate_depth.py` (Q7: the heaps' depth in a site's plate bundle against rays at the compiled height field, by the one-sided measure), `q8_heap_trials.py` (Q8: privileged trials against every heap of a site in that site's own compiled world), `q9_plate_binding.py` (Q9: every pinned site has its plate bundle, the bundle names the pinned package and every file hashes as pinned; the bake's own checks from the bundles) | copies of the maintainers' QC scripts |
| `sitegen/w2gen/` | the W2 site generator (the writer side: `family.py`, `mjcf.py`, `predicates.py`, `grid.py`, `envelope.py`, `views.py`, `cli.py`) | copy of the maintainers' W2 generator |
| `sitegen/w3gen/` | the W3 site generator of version 1.1 (the rubble under the slab included; it imports w2gen) | copy of the maintainers' W3 generator |
| `sitegen/w4gen/` | the W4 site generator of version 2.0 (w4gen 0.3.0: W3 with heaps on the open floor; it imports w3gen and w2gen unchanged): `family.py` (the placement), `library.py` (the reader of the heap library's pieces), `mjcf.py` (a heap's inline height field), `measure.py` and `heapchecks.py` (what is measured on the compiled model), `predicates.py` (G1-G24), `cli.py` | copy of the maintainers' W4 generator |
| `sitegen_cal/calgen.py` | the calibration site's generator (calgen 0.4.0; it imports w2gen, w3gen and w4gen) | copy of the maintainers' calibration generator |
| `degenerate/` | the degenerate submissions D0-D4 (programs) and D8, D9, D10 (plans of the maintainer variant: they need the true pose; D10 rolls the robot at the heap nearest to the pad and comes with a helper that stages it on a development site), `run_verify.py` (a development hidden set) and `practice_sync.py` (refreshes the practice ground's copy of the cell) | copy of the maintainers' folder of degenerate submissions |

`SHA256SUMS` lists every file of this folder with its sha256 (the table above names the source of each folder). The
copies are kept for reading and review: they expect the maintainers' layout of folders around them
(`DEV = <this file>/../..`) and are not run from here. The selection, packaging and oracle-verify tools are NOT here
either: the admission rule and the selection tools are kept by the maintainers and the tools are not published
(`tests/README.md` section 8 describes how the set was selected and packed), and rebuilding a package from its seed
is done by the maintainers too (the generators' copies here read the capability tables, the heap library with its
registry and the art tools, which are not published). The heap library, the scene builder and the tools that bake
the plates are NOT here: they read the maintainers' art packs. The leak scan and its word lists are NOT here (the
word lists name the prototype and the art packs; the maintainers keep them and they are not published).

How the variants run (the maintainers' switch): `qc_verify.py --submission oracle --variant NAME` copies `tests/` to a
scratch stage, replaces the stage's `reference/reference.py` by `qc/variants/qc_reference.py` and writes the plan as
`reference/qc_variant.json`; the submission is the oracle's own (`solution/solve.sh`, the real token), so the cell
runs the variant as its privileged `reference_run`. Everything else (verify.py, the cell, the scoring, the frozen
hidden set) is the task's bytes. A variant moves a search point or the pad only in the cell's in-memory copy of the
site and only after the robot has stopped: a search point and the pad are not physical objects (site.xml has no geom
for them), so the world and the trajectory stay the same and only the distance the scoring measures changes.
