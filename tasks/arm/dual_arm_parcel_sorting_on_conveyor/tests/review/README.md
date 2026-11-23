# Review tooling of the label-facing station

Maintainers' tools that certify the grader and keep the agent's documents bound to the cell. Nothing here runs at
verify. Every tool but `freeze.py` and `quota.py` (standard library only) imports the cell package (`tests/parcelcell`)
and needs the task's Python (3.12 with the pinned MuJoCo, numpy, scipy, Pillow, OpenCV); a tool that takes `--out`
writes its results to that folder, outside the task folder.
Run them with `PYTHONDONTWRITEBYTECODE=1 python3 -B ...` and a GL backend (`MUJOCO_GL=egl` with a GPU, `osmesa` on
the CPU); leave no `__pycache__` behind. The cell's default look is the filament backend: a tool that renders needs the
unpacked render bundle (`PARCELCELL_RENDER_DIR`, else `/opt/task-assets/render` as in the image) and a Vulkan driver
(Mesa's lavapipe on the CPU); the physics and the score do not depend on the look. The robot's mesh files are task
assets, not in the task folder: before a tool builds a cell (on a host, or in a container of the task image that holds
a copy of `tests/`), put them where `tests/test.sh` puts them for the grader's cell, from the image's fetched copy
(`/opt/task-assets/payload/given/robot/`) or from the asset repository at the commit pinned in
`environment/assets.sha256`: `meshes/*.STL` into `parcelcell/assets/robot/meshes/` and `gripper_2f85/assets/*.stl` into
`parcelcell/assets/gripper_2f85/assets/` (the given model files under `environment/setup/payload/given/robot/` read the
same files at `meshes/` and `gripper_2f85/assets/`).

| file | what it does |
|---|---|
| `common.py` | paths, the grader's configuration, `LocalCell` (the client's API on an in-process server), one shift of a program in process, the review shifts (labelled seeds "review": one per belt-speed class), the line review shifts (`line_seed("review-line", k)`: slow running, medium indexing, fast running), a tooling IK |
| `docs/make_docs.py` | renders `instruction.md` and `given/cell/` (`cell_manual.md`, `cell_api.md`, `cell_config.json`) from `docs/templates/` and `docs/facts.py` |
| `docs/facts.py` | every number of the documents, computed from the cell (configuration, a built cell, the generator) and `docs/measured.json` |
| `docs/measured.json` | values only a measurement gives, each written by its tool (gripper sheet, the observability study, timing, the picture's text); the sandbox's `nproc` and `os.cpu_count()` are filled by hand from the task image |
| `docs/make_models.py` | writes `given/robot/` and `given/station/` from the cell's own build with the maker's nominal dynamics (`docs/nominal_dynamics.json`), torque inputs instead of the servos, the cameras' drawing poses, and checks them against the cell (exact kinematics, every dynamic value a nominal one) |
| `docs/nominal_dynamics.json` | hidden: the factors between the given models' nominal dynamics and the cell's robot (one per link mass, link inertia, joint damping, friction loss and armature), drawn once by `draw_installed.py dynamics` |
| `draw_installed.py` | the station as installed against its documents, drawn once from a recorded seed: `cameras` writes `grader_config.json` "installation" (each camera's fixed error against the drawing, within `cameras.installation_tolerance`; the commissioning service gets it through `sites.json`), `dynamics` writes `docs/nominal_dynamics.json` |
| `docs/test_docs.py` | the binding (pytest): shipped = rendered, no unbound number in a template, the manual's numbers read back equal the configuration, the public configuration holds no grader-only key, the given models agree with the cell, the development shifts are the rule's, the text rules, the publication gate |
| `ladder/run_ladder.py`, `ladder/programs/`, `ladder/controls.py` | the degenerate ladder: programs written for the wire and state-writing grader controls on three review shifts of either setting (`--setting rated` or `line`) |
| `mutants.py` | a mutant for every sentence of the credit rule, with the verdict written before it runs |
| `quota.py` | the lexicographically first subset under quotas (selection, development shifts) |
| `gen_shifts.py` | the development shifts (their rules and salts: `line-dev` gives the commissioning service's, `dev` the rated setting's), candidate lists of either setting, the never-queue replay, the line set's slots (`LINE_SLOTS`) |
| `draw_line_set.py` | the graded line set by its registered rule (`hidden/line/selection_rule.txt`): `rule` writes the rule once, before the draw; `draw` draws the salt and each slot's first draw of the frozen line generator, no quota and no admission (`hidden/line/selection.json`, `seeds.json`, `shifts/`); `check` draws the set again from the recorded salt and compares the bytes |
| `select_hidden.py` | the rated set's selection by its registered rule (`hidden/selection_rule.txt`): candidates drawn per slot, the quotas, the frozen reference run twice per candidate as the grader drives the oracle (admission, identity), every candidate and reason recorded in `hidden/selection.json` |
| `freeze.py` | the frozen generator and scoring configuration (`tests/frozen.json`): `check` fails when a file that defines a shift or its score, the cell, the reference or a version changed |
| `observability.py` | the waybill in pixels in each camera over the places a parcel can be, the share visible and the contrast |
| `gripper_sheet.py` | the gripper's closing and opening times, openings, squeeze and hold, measured in the cell |
| `timing.py` | wall seconds per simulated second, the cost of a frame per camera, the verify-timeout arithmetic |

## What the agent gets and what the cell is

The cell (the truth) keeps its robot, physics, belt and scoring. What the agent gets is what an engineer gets: the
robot's model files carry the maker's nominal dynamics (masses and inertias within 10 %, joint damping, friction loss
and armature within 30 % of the cell's; never closer than a quarter of the tolerance) and no servo gain (torque inputs
at the drives' limits); the pads' and the belt's friction are the data sheet's ranges; the cameras' poses are the
installation drawing's and their optics the data sheet's, while the cell renders every image (colour and depth) from
the installed cameras (`grader_config.json` "installation", grader-only); the torque readings of the public state
carry the drives' sensor noise (`sensors.tau_noise_rel`, `sensors.tau_offset_rel`; seeded per shift, deterministic),
while the privileged state and the safety watch read the true torques. A new draw (`draw_installed.py`) changes the
cell's images and the given models: regenerate the models and documents, then renew the hidden sets as the section
"The hidden sets" below says.

## The documents and the cell cannot drift

Edit text in `docs/templates/`, never in the generated files. A template holds no number of its own: each value is
a `{{key}}` of `docs/facts.py`, and `make_docs.py` refuses to write while a template holds a digit that its
`LITERALS` list does not explain (section numbers, product and format names, the gripper command's ends). After any
change of `tests/grader_config.json`, the cell package or a measurement:

    python3 -B tests/review/gripper_sheet.py --out OUT --write-measured
    python3 -B tests/review/observability.py --out OUT --write-measured
    python3 -B tests/review/docs/make_docs.py --write
    python3 -B tests/review/docs/make_models.py --write
    cd tests/review/docs && python3 -B -m pytest -q -p no:cacheprovider test_docs.py

then copy `environment/setup/payload/` to `tests/setup_payload/` and refresh `tests/given_hashes.json` (the
packaging's files). `make_docs.py --check` exits 1 when a shipped file differs from its render. Values that need a
measurement in the task image or the final look render as visible markers `[TO BE MEASURED: ...]` and
`[TO BE WRITTEN ON THE FINAL LOOK: ...]`; the publication gate of `test_docs.py` fails while any is left.

## The degenerate ladder

    python3 -B tests/review/ladder/run_ladder.py --out OUT --jobs 6            # in process, three review shifts
    python3 -B tests/review/ladder/run_ladder.py --out OUT --jobs 6 --setting line      # three line review shifts
    python3 -B tests/review/ladder/run_ladder.py --out OUT --wire-kit tests --only p1_hold_start    # over the wire

Programs (`ladder/programs/`, each `solve(cell)` for the wire; postures taught offline on the given models): `p0`
returns at once; `p1` holds the start posture (do nothing); `p2` holds both hands in the parcels' path; `p3` shoves
the parcels against the near lip (the guards keep them on the belt); `p4` random joint targets; `p5` a hand as a dam across the belt; `p6` reads the privileged truth
every second and touches nothing (privileged cell); `e0` the agent's example program. Controls (`ladder/controls.py`,
state writes, in process only): `c1` turns every W parcel label up before the work window (must score 1), `c2` turns
every parcel label down (must score 0), `c3` turns every X parcel (outside the station's duty) label up and nothing
else (must score 0: turning an X parcel earns nothing), `c4` turns every W and every X parcel label up (must score 1:
the X parcels add nothing and cost nothing), `c5` turns the first W parcel label up (credit exactly 1), `c6` turns two
W parcels label up and one U parcel label down (credit exactly 1: a spoiled U parcel costs 1), `c7` turns the first W
parcel label up and holds it still for 6 s (on the line setting late: credit exactly 0), `c8` turns two W parcels
label up and holds one U parcel still for 6 s (on the line setting credit exactly 1: a late U parcel costs 1); the
rated setting has no due time, so there `c7` and `c8` give credit 1 and 2. On an indexing line the controls move a
parcel the belt holds in the zone past it after half a second (a state write), so that the list goes on. Expected:
every program, `c2` and `c3` score 0 on every shift (`p1`, `p6`, `e0`: credit exactly 0; on the line setting `p0`
too: a parcel not done costs nothing), `c1` and `c4` score 1.

Last run, rated setting (the cell's default filament backend, plain parcel looks; the rated setting's population and
duty: cartons whose label is underneath or on the downstream face are the station's work; the measured arrival gap;
review shifts k = 1 slow 0.051 m/s 15 parcels (W 4, U 6, X 5), k = 2 medium 0.186 m/s 21 parcels (W 8, U 5, X 8),
k = 5 fast 0.255 m/s 23 parcels (W 7, U 6, X 10), 600 s each): all 16 entries as expected. Shift scores 0 for p0-p6,
e0, c2 and c3 on all three shifts; credits p0 -21 / -26 / -29 (all lost), p1 0 / 0 / 0 (no parcel touches the parked
robot), p2 -21 / -21 / -29 (parcels jam against the hands; 15 / 17 / 23 never cross), p3 -5 / -1 / 0 (the station's
guards keep the shoved parcels on the belt; some are crushed against the near lip and lost as damaged, one falls), p4
0 / 0 / 0 (six protective stops, then the lock-out), p5 -21 / -24 / -29 (parcels held back: 15 / 20 / 23 never
across), p6 0 / 0 / 0 and e0 0 / 0 / 0, c1 1.0 / 1.0 / 1.0 (credit 4 / 8 / 7), c2 credit -6 / -5 / -6, c3 credit
0 / 0 / 0 (turning X parcels earns nothing), c4 1.0 / 1.0 / 1.0 (credit 4 / 8 / 7: the X parcels add nothing and cost
nothing), c5 and c6 credit 1, c7 credit 1 and c8 credit 2 on every shift.

Last run, line setting (the same backend and looks; every parcel in the duty, no X; line review shifts k = 0 slow
running 0.092 m/s 23 parcels (W 16, U 7), k = 1 medium indexing 0.239 m/s 48 parcels (W 32, U 16), k = 2 fast running
0.279 m/s 53 parcels (W 39, U 14), at most 600 s each): all 16 entries as expected. Shift scores 0 for p0-p6, e0, c2,
c3 and c7 on all three shifts; credits p0, p1, p2, p5, p6 and e0 0 on every shift (nothing lost; what is not done
costs nothing), p3 -1 / 0 / -6 (parcels crushed against the near lip and lost as damaged: 1 / 0 / 2; on k = 2 four U
parcels not readable), p4 0 / 0 / 0 (six protective stops, then the lock-out), c1 1.0 / 1.0 / 1.0 (credit 16 / 32 /
39), c2 credit -7 / -16 / -14, c3 credit 0 (no X parcel on the line), c4 as c1, c5 and c6 credit 1, c7 credit 0
(late), c8 credit 1 (the late U parcel costs 1) on every shift.

The tooling uses the plain parcel looks (`REVIEW_PARCEL_VISUALS=kit` for the graded look);
physics and score do not depend on them. Through the wire (`--wire-kit tests`, the client's
launcher in a child process, k = 1; a run on an earlier population, not repeated on the ones above): p0 -17, p1 0, p5 -17,
e0 0, as in process then. Each program carries `WIRE_RUN`:
pending its run through the full graded path (verify: the run account, the seccomp filter, the import check).

## Mutants of the credit rule

    python3 -B tests/review/mutants.py --out OUT

56 mutants, each with its expected verdict written in the file before it runs. 39 on the rated setting's sentences:
the five readable clauses (physical constructions: a tilted parcel tipping across the line, a parcel 6 cm in the air,
kicked to slide, set spinning, a low parcel rubbing under the closed fingers, a parcel stacked on another, a squeeze
entered into the watch; and the thresholds 20 / 25 / 25.01 / 30 deg, 1.99 / 2.0 rad/s, 0.099 / 0.1 m/s on constructed
crossing records), the first downstream crossing (first airborne, then calm), a backward crossing (judged at the later
downstream one), a parcel moved past the line without crossing (lost), the shift's end, `finish()` and an early return
(lost), a parcel off the near edge and one dropped to the floor (fall), a pass-through parcel knocked over, the class
fixed at arrival (a parcel turned right after its arrival either way keeps its class; one that never gets far from the
entry), the credit arithmetic and its floor, a shift without W parcels, the lock-out (parcels still judged, commands
refused), and benign variants that must score as the truth (yaw, the belt's edge, side by side with a gap, turned
exactly label up, the belt end); and the duty's classes: an X parcel turned exactly label up beside an untouched W
parcel (credit 0), an untouched X parcel (costs nothing), an X mailer dropped to the floor (lost, -1), a label-up
mailer knocked over (U not readable, -1), the duty as one configuration entry (the same carton X under the default
duty and W when the duty lists its label position) and the label position taken in the world frame after the arrival's
yaw. 17 on the line setting's sentences (mutants N*): every parcel in the duty (W or U, no X); a parcel not done when
the shift ends costs nothing (the shift's duration runs out before a W parcel reaches the line, `finish()` with a U
parcel still upstream, a parcel held in the station so that the next is never set down: both W parcels count in the
divisor); the feed on settle (the next parcel at the tick after the one before it crossed the scan line, fell or was
damaged); the indexing belt (it stands while the parcel's centre is in the zone and runs again once the parcel is
past; a parcel left there stalls the list); the credit arithmetic (one of four W parcels turned: 1/4; a spoiled U
parcel costs 1); the shift's end at the tick its list's last parcel is settled; and the due time (a W parcel held 6 s
is late and earns nothing, held 4 s it is readable; a late U parcel costs 1; an indexing halt counts against the rated
cycle; a parcel riding free crosses well inside its due time). Last run: 56 of 56 give the verdict their sentence
demands. A note from the constructions: a parcel kicked or set spinning on the belt loses contact with the belt
surface for a moment, so its record shows `not_on_belt` beside `sliding` or `turning`.

## The hidden sets

    python3 -B tests/review/freeze.py check                  # the frozen files (tests/frozen.json)
    python3 -B tests/review/draw_line_set.py check           # the line set is its rule's draw for the recorded salt
    python3 -B -s tests/review/select_hidden.py rule         # the rated set's rule as the tool implements it
    python3 -B -s tests/review/select_hidden.py run --tests tests --reference tests/reference --salt draw \
        --label hidden --out OUT --rule-file tests/hidden/selection_rule.txt --jobs 4
    python3 -B tests/review/gen_shifts.py line-dev --out DEV --salt <sites.json salt_hex>   # the development shifts

Two sets (`tests/README.md` section 6): the LINE set (`hidden/line/`) is the graded one, the RATED set (`hidden/`) is
the oracle's. The line set's rule (`hidden/line/selection_rule.txt`) was written, and its sha256 recorded, before the
draw; `draw_line_set.py draw` drew the salt once from the system's entropy and took each slot's first draw of the
frozen line generator on the frozen files of `tests/frozen.json`, with no quota and no admission, and wrote
`hidden/line/selection.json` (the salt, the rule file's sha256, the eight slots), `seeds.json` and `shifts/`; nothing
was run on a shift before the set was frozen.

The rated set's rule (`hidden/selection_rule.txt`) was written before its draw. `select_hidden.py run` runs inside the
task image (the grading environment, the grader's numerics settings of `test.sh`), on the frozen files: it draws the
salt once from the system's entropy, draws each slot's candidates in k order, keeps the first that meets the quotas
and whose two reference runs (fresh processes) are admitted and identical, and writes `selection.json` (the salt, the
rule file's sha256, the sha256 of every frozen file before and after, every candidate with its class counts, natural
end, quota result, run digests, admission result and reasons), `seeds.json` and `shifts/`. Nothing else influenced the
choice, and no candidate was run with a camera-driven program before the set was frozen. The rated set was selected on
the cell files recorded in `frozen.json` "rated_set" and carried to the present files by the identity check recorded
there (the reference's records on its eight shifts identical field by field on both cells, the cell's version string
apart; measured before one `describe()` text was reworded, see the record). The anchor (`anchor.json`) is the
reference's facing score on the rated set in one fresh full verify of the oracle in the task image. A change to any
file `freeze.py` pins needs the rated set's identity check again, a new draw of the line set and a new anchor. The
line development rule (`gen_shifts.py line-dev`) gives shifts equal to the commissioning service's `sites.json`
(checked by `test_docs.py`).

## Observability, gripper sheet, timing

    python3 -B tests/review/observability.py --out OUT [--write-measured]
    python3 -B tests/review/gripper_sheet.py --out OUT [--write-measured]
    MUJOCO_GL=osmesa python3 -B tests/review/timing.py --out OUT --tag osmesa

Observability (`measured.json` "observability_faces": a study on the cameras' drawing poses, not printed in the
documents; the installed cameras' coverage is checked by the same study on the cameras as installed, the robot parked, the rated setting's population, 200 cartons per label position and 200 mailers): a waybill on
the top face is shown by the `infeed` camera before the work window and by the `head` camera inside it (100 %), one
on the downstream face by the left palm camera at the start posture (100 %); one on the upstream face, on either side
face or underneath by no camera before the work window (0 %). The waybill's size in pixels per camera and place: `cell_manual.md`
section 6 (from the published calibration; `measured.json` "observability" is not printed in the documents). The waybill's tone against
the parcel faces under the final look is part of the look's own fit (the exposure keeps a white waybill
distinguishable on the palest carton). Gripper (the configuration's current physics
settings): closing 0.36 s, opening 0.52 s, 85.2 mm open, a 50 mm block squeezed with about 94 N in total at
command 1 (the drive at its limit), every block up to 3 kg (the rated setting's heaviest parcel; the line's parcels weigh at most 2 kg) held for 3 s. Timing in the task image (a container on the build host, 8 CPUs, kit looks, the default filament backend with the
one-lamp look): at host load 100-145 on 48 cores a frame up to 1.03 s p95 per camera; at host load 12-15 0.050 wall s
per simulated second, a frame 0.17-0.23 s p95 (colour, colour + depth), first frame 4.0 s; on a shift of the rated set
(24 parcels) at host load 60-200, 0.076 wall s per simulated second and a frame 0.31-0.42 s p95, first frame 5.9 s; on
the graded set's fastest shift (59 parcels at 0.337 m/s) at host load about 180, the 59 parcels' looks in 258 s with 4
worker processes, a cell built from the look cache in 2.1 s, its first frame 12.0 s, 0.108 wall s per simulated second
and a frame 0.327 s p50, 0.448-0.581 s p95 per camera. task.toml's verify time limit (21200 s) is the arithmetic of
the 8 graded line shifts with these last costs (`tests/README.md` section 7), x 1.35 (re-measure on the grading host
if it is busier).
