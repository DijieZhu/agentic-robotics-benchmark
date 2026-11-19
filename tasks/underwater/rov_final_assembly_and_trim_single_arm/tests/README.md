# tests/ (grader side)

Maintainers' notes on the grader of this task. Started by the cell freeze (P18a); the hidden-set package (P20) and the final
assembly complete this file.

## The episode's vehicle (the per-unit deviation)

Every hidden episode carries a per-unit deviation `{dm_kg, dx_mm, dy_mm}` of its base vehicle. There is one definition,
used by the cell (`arm/assembly_rov/server.py` `vehicle_kit`), by the trim truth (`arm/assembly_rov/trim_eval.py`
`TrimTable.base_row`) and by the reference harnesses: **this vehicle = the trim table's base vehicle (its tabled mass and
centre of mass, `trim_table.json`) plus the deviation**: mass + `dm_kg`, centre of mass + (`dx_mm`, `dy_mm`, 0) in the
vehicle frame B; the inertia about the centre of mass, the displacement and the centre of buoyancy as tabled. The cell
puts this mass and centre of mass on its copy of the hidden kit's `vehicle` body, so the load cells weigh exactly the
vehicle the trim is graded on.

## The cell's data files

`cell_layout.json` (the frozen cell layout), `fixture_facts.json` (the fixtures' facts), `trim_sheet_printed.json` and
`cell_public.json` (the printed trim sheet's columns and the cell manual's trim facts, read by the reference program) are
byte-for-byte copies of the development files; their sha256 are pinned by the maintainers' cell freeze record.

## The pinned numerics

The cell's physics is bit-for-bit the same on every x86-64 CPU only with three variables set before any interpreter
starts:

    export OPENBLAS_CORETYPE=Haswell OPENBLAS_L2_SIZE=256 NPY_DISABLE_CPU_FEATURES="X86_V4 AVX512_ICL AVX512_SPR"

Without them NumPy's bundled OpenBLAS picks its kernels per CPU (SkylakeX on a host with 512-bit vector units, Haswell on
others) and NumPy its SIMD paths, and the same episode's state differs in the last bits between two such hosts from the
first checkpoint on (the outcome, the events and the scores agree; the state hashes do not). `OPENBLAS_L2_SIZE=256`
keeps the pinned Haswell kernels' own block sizes: without it the OpenBLAS 0.3.34 that NumPy 2.5.3 bundles enlarges
them on AMD Zen 4 / 5 hosts, and its Haswell float64 matrix kernel then overruns a fixed buffer, so a float64 product
or solve with an inner dimension above about 300 crashes the process or returns wrong numbers (in any process under
the pin, an agent's program included); a process whose matrix products all have inner dimensions of 256 or less
computes the same bits with or without it. The task image sets all three (`environment/Dockerfile`, its last `ENV`)
and `test.sh` exports them again before it starts the verifier. Set the same three whenever the cell runs outside the
image (a maintainer's direct run, a re-selection, a re-run of the hidden set on another machine). The development tree
reads the values from these two files (`-dev/tools/pinned_env.py`), so they are changed in both places or not at all;
a change is a change of the cell's physics.

## The fixed hidden set (`hidden/episodes.json`, `hidden/selection.json`, `review/select_set.py`)

The grader plays four fixed episodes, one per stratum in the order E1 (fresh water, a heavy unit), E2 (brackish, a
light unit), E3 (cold sea water, light), E4 (warm sea water, light); the dome end of the electronics enclosure is 0 deg
in two of them and 180 deg in the other two. `hidden/episodes.json` (format `rov-asm-set/1`) holds per episode the
layout seed, the stratum, the dome end, the work order (unit, water temperature and salinity, one decimal), the
per-unit deviation (`dm_kg`, `dx_mm`, `dy_mm`), the layout digest, and where the episode came from (generation,
trim candidate, layout candidate, round; verify does not read these). `verify.py` reads the file with
`load_hidden_set` (any malformed entry is a task error whose message names no value) and, after building each
episode's cell, compares the cell's layout digest with the recorded one (`check_layout_digest`): a different layout --
the kit, the layout code, the fixtures or the layout block of `grader_config.json` changed after the selection -- is a
task error, never a silently different episode.

The set was selected by a pre-registered rule (rule_version 3, `-dev/review/selection_rule.json` and
`-dev/review/PREREGISTRATION.md`; their sha256 are in `hidden/selection.json`; rule_version 3 replaced rule_version 2,
and that one rule_version 1, before any generation was opened -- the last revision changed only what the precondition
of the opening counts as a false lock of the slow-release probe -- and the superseded files are kept byte for byte in
`-dev/review/superseded/`), on seeds sealed and committed before
any candidate was drawn (commit and reveal: the opened generation's seeds are published in `selection.json`; anyone
recomputes their sha256 against the commitments printed in the rule). `review/select_set.py` carries the rule out
and decides nothing; `hidden/selection.json` records every candidate it drew, every condition it checked, every run
and the acceptance. To check the record: `python3 -I -B review/select_set.py check` (the freeze) and
`python3 -I -B review/select_set.py selftest` (the stream code's known-answer test and the tool's own tests); the
tool's docstring lists the steps and what each needs (the development folder's readings library of the trim study,
the view-coverage tool, Docker with the task image).

What a maintainer must know:

- **After the set is fixed, a change of the kit, the layout code or its configuration, the cell's physics settings
  (sleep included), the readings of the hard rules H1 / H2 / H3, the coupling, the positioner or the weighing model,
  or of the trim truth (the trim table, `trim_eval.py`, the printed sheet's numbers, the hole map, the blocked holes,
  the conflict rule, the load-cell stations) is a class S amendment: the set is selected again with the next sealed
  generation, and the quality checks that use the set (the mutant / benign cases and the run on a second CPU) are run
  again.** A change of only the reference program or its planner, the server / verify plumbing or the renderer's look
  that changes no layout digest and no physics or grading quantity is class V: the four episodes are each played
  once more by the reference program and the set stays if all four pass. Agent-visible prose, drawings and photos
  that change no number the public route or the layout reads are class N. The main session classifies; in doubt,
  class S. Never replace one episode by another candidate of the same generation.
- **The reference vehicle's two published ballast maps are not this task's acceptance.** The digest's fresh-water
  map and its sea-water map (and the vendor's 5- and 10-block maps, the cell manual's standard map) are cheap
  readings the selection requires to fail: on every episode the fresh-water map, the vendor maps and the standard map
  score 0 and the sea-water map at most 0.7 (lenient window). The trim the grader accepts is the window of
  `trim_table.json` for this unit, in this water, with the weights that actually lock.
- **There is no nominal-layout retreat.** `grader_config.json` keeps `oracle_nominal_layout` false. If the twelve
  layout candidates of a generation run out before every stratum has an episode, no set is written and the case goes
  to the main session (the rule's fallback FB-L); the precedents' fallback of grading the oracle on the nominal
  layout applies only after a separate ruling.
- The selection's image runs used the task image built from `environment/` (its local asset folder, pinned file by
  file by `assets.sha256`; tag and image id in `selection.json`) and the assembled givens (`setup_payload/`,
  `given_hashes.json`, which the verifier checks `/opt/arb` against); all of these are frozen in `selection.json`.
  The cell compiles the arm and the gripper from the givens, so `given/robot` must keep the bytes of the development
  robot models `-dev/build/robot/tree` (its digest is frozen in `selection.json`); a different robot model is a change
  of the cell's physics (class S). A later change of only the givens' prose, drawings or photos is classified like any
  other change (class N when it changes no number the public route or the layout reads); `select_set.py check` then
  reports it, which by itself does not void the set.
- Run the cell outside the image with the three pinned numerics variables (above); the selection's host-side checks
  (the trim stage, the layouts, the views) refuse to run without them.

## The maintainers' record and the amendments after the acceptance

verify.py keeps a record of every agent episode under the root-only `/opt/arb-verify` (`ARB_RECORD_ROOT`): the agent
program's stderr (`logs/agent_epNN.log`) and the replay trace (`out/trace/agent_epNN/`). Neither is a scoring input.
Since work package P32b the record has limits, set as constants of `episode_trace.py` (not grading knobs, so not in
`grader_config.json`): the trace keeps the frames of the first 20 camera requests and then one of every 20, at most
150 MB of PNGs, at most 40 000 joint-state records and 20 000 events, at most 200 MB of files per episode; the stderr
log of an episode is kept to its first 5 MB and then ends with one line saying how many bytes were dropped (checked
every 0.1 s while the program runs). `meta.json` lists every camera request and why a frame was not kept.

Changes after the set was accepted are entries of `hidden/selection.json`'s `amendments` (`review/select_set.py
amend`), each with the frozen values it changes; `review/select_set.py check` reports a frozen value that changed as
explained when the last entry recording it gives exactly its value now, and fails on any other difference.
