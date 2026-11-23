# dual_arm_parcel_sorting_on_conveyor: the grader (maintainers' notes)

Nothing in this folder reaches the agent: the engine uploads `tests/` only for the verify stage, into a separate
verifier sandbox. The agent's side is `environment/setup/payload/` (staged into its home) and the commissioning
service in the image (`environment/commission_service/`, readable by its own account only).

## 1. Layout

| path | what it is |
|---|---|
| `test.sh` | the verify stage's entry: closes the stage folder, builds the root-owned copy of the givens at `/opt/arb`, pins threads and numerics, runs `verify.py` |
| `verify.py` | the grader (its docstring lists every step and gate) |
| `cellkit/` | the process kit: `wire` (the JSON-lines RPC, budgets, end reasons), `runner` (the run regime), `lockdown`, `verify_freeze`, `netaudit`, `nonet`, `seeds` |
| `parcelcell/` | the station's cell: physics, robot, belt and parcels, cameras, the tool-pose controller, safety watch, score, the wire server, the client |
| `reference/` | the privileged reference program (`controller.py`, `run(server)`, and its package `refctl/`; `README.txt` for a maintainer), run inside the cell by `reference_run` |
| `grader_config.json` | every number of the cell and the runner (deep-merged over `parcelcell.config.DEFAULTS`) |
| `hidden/` | the RATED set (section 6): `selection_rule.txt` (its registered rule, written before the draw), `selection.json` (the draw: every candidate and reason), `seeds.json` (seeds and shift packages with their sha256), `shifts/*.json`; `oracle_token.sha256` |
| `hidden/line/` | the LINE set, the graded one (section 6): `selection_rule.txt` (its registered rule, written before the draw), `selection.json` (the draw: the salt and the eight slots), `seeds.json` (seeds and shift packages with their sha256), `shifts/*.json` |
| `anchor.json` | the reference value on the rated set (the oracle's facing score in one fresh full verify in the task image) and `full_at`; its block `line`: the line set's absolute scale (value 1.0, full_at 1.0) |
| `frozen.json` | the frozen generator and scoring configuration and the frozen cell (sha256 of each file, versions; `review/freeze.py check`) |
| `given_hashes.json` | sha256 of every file of `/opt/arb` |
| `setup_payload/` | a byte copy of `environment/setup/payload/` (the verifier never runs setup) |
| `robotics_grader/` | a byte copy of the benchmark's shared grader kit (the clearing of the agent's account, the verdict) |
| `review/` | review tooling (nothing runs it at verify) |

Byte copies that must stay equal (the repository test of this task checks them): `environment/commission_service/
parcelcell/` = `parcelcell/` less `client.py`; `environment/commission_service/cellkit/{__init__,wire}.py` =
`cellkit/{__init__,wire}.py`; `environment/setup/payload/given/lib/parcelcell/client.py` = `parcelcell/client.py`;
`setup_payload/` = `environment/setup/payload/`. After any change to one of them, renew the copies and then
`given_hashes.json`. The robot's mesh files (the dual-arm platform's links and the gripper's) are not in this folder or in `environment/`, and the agent's copy of the gripper's licence file (`given/robot/gripper_2f85/LICENSE`) is not in the setup payload: they are task assets on the benchmark's asset repository at the commit pinned in `environment/assets.sha256` (with the agent cameras' render bundle), fetched at the image's build to `/opt/task-assets/payload`; `setup.sh` stages them into the agent home, `test.sh` into `/opt/arb` and into this folder's `parcelcell/assets/`, the image into the commissioning service's cell. `given_hashes.json` pins them at their paths in the agent home.

## 2. How a shift is driven, end to end

1. `test.sh` (root, cwd the stage folder): `chmod 700` the stage folder and its parent; rebuild `/opt/arb` from
   `setup_payload/` and the image's task assets (`/opt/task-assets/payload`: the robot's mesh files and the gripper's licence) (root-owned, read-only), and copy the mesh files into this folder's `parcelcell/assets/`; export `HOME=/root`, `PYTHONNOUSERSITE=1`, `MUJOCO_GL=osmesa`,
   `LP_NUM_THREADS=8`, BLAS / OpenMP 2 and the numerics pins (section 5); `exec python3 -s verify.py`.
2. `verify.py`, before numpy is imported: the upload time from the stage's ctimes; the agent's account cleared by the
   benchmark kit (`robotics_grader.clear_agent_account`: every supervisord program of the image off, supervisord
   reloaded, every process of the account killed until none comes back; stats `agent_account_cleared`,
   `agent_account_programs_disabled`, `agent_account_clear_killed`); whatever of the account is left stopped and
   killed, then kept SIGSTOPped until the verdict; the stage locked (0700 / 0600); the submission
   snapshotted root-owned (regular files only); late writes (after the upload) zero the contract, late entries
   elsewhere are quarantined.
3. `/opt/arb` checked file by file against `given_hashes.json` (a mismatch is a task error).
4. The token: `submission/.oracle_token` hashed against `hidden/oracle_token.sha256`; a match builds every cell
   privileged and picks the rated set; every other submission gets the line set (`verify.graded_set`, section 6).
5. The contract: `sort.py` parses and defines `solve`; the route gate (no weight files, no training calls, at most
   60 MB); every `.py` file readable by the parser; the static network audit (no networking module, no computed import
   or code, no program start; files byte-equal to the given library exempt); the import check (`sort` imported as
   `nobody` under the no-network filter, `choom -n 1000`, 120 s wall, 120 CPU s).
6. The hidden shifts: the set's `seeds.json` (`hidden/` or `hidden/line/`), each package checked against its sha256
   and its setting against the set's (missing, short, different or of the other setting: a task error). The desktop is left to settle; the no-network filter proves itself once (`nonet.selftest`).
7. Per shift, one after another, in a fresh cell and a fresh process:
   * the cell is built from the package (`ParcelCellServer(config, mode="hidden", privileged=...)`, `reset(package)`):
     the shift's duration fixes its program-time budget (`program_time_factor` x duration + `program_time_extra_s`)
     and its frame budget (`frames_per_shift_s` x duration);
   * a run folder under an opaque tag (`/opt/arb/runs/epNN`): a read-only copy of the snapshot, the client library,
     an empty HOME and TMPDIR owned by `nobody`;
   * the program: `runuser -p -u nobody -- prlimit --cpu=<4 x budget> --core=0 -- python -I -B -c <no-network filter>
     -- choom -n 1000 -- python -B -E -s -c <bootstrap> <lib> sort` (or `bash run.sh`), with the minimal environment
     of `cellkit.runner.child_env` (no numerics pin, threads 2);
   * `cellkit.wire.serve_episode` answers requests until the shift ends (section 3); the program's stderr goes to
     `/opt/arb-verify/logs/epNN.log`;
   * every process of `nobody` is stopped and killed, new files of `nobody` in `/tmp`, `/dev/shm`, `/var/tmp`,
     `/run/lock` and the agent home are swept, the run folder removed; the shift is finalized (`finalize()`: the
     record and its score) and the cell closed;
   * a cell fault (an internal error while the shift was live, a physics fault, a lost frame) runs the shift once
     more from scratch; at most one shift per verify; a repeated render failure, or a fault in a second shift, is a
     task error.
8. `facing_score` = contract gate x the mean of the shift scores; the verdict through `robotics_grader`
   (`ratio_rewards`: on the line set against `anchor.json`'s block `line`, value 1.0 and full_at 1.0, so the reward is
   the absolute score; on the rated set against `anchor.json`'s value and full_at); the records in
   `/opt/arb-verify/out/` (`shifts.json`, `log_lines.txt`).

The oracle's `sort.py` reads `describe()`, one frame of every camera, then calls `reference_run()`: the cell imports
`reference/controller.py` and calls `run(server)`; the reference acts through the cell's own command path (the speed
limit, the safety watch and the damage rule apply), its own time capped by `budgets.privileged_program_time_factor`
(10) x the shift's duration; the wire's cap in privileged mode is the shift's program budget plus that.

A damaged parcel shows it (both settings, the graded cell and the commissioning service's alike): every exposure keeps
the set of parcels the safety watch has listed as damaged by its tick (`cameras.CameraSystem.damage_fn`), and before a
frame of that exposure is drawn each newly damaged parcel gets its crushed look (`parcelkit/crush.py`: its base colour
map darkened and greyed, dark irregular creases over it, pale fold lines or scuffs; a deterministic overlay seeded by
the shift's digest and the slot). The classic backend rewrites the parcel's texture in the model and uploads it again;
the filament backend rewrites it in the render twin and makes the session's renderables again at the next frame (about
0.15-0.35 s, once per damaged parcel). Measured on 20 parcels (filament): the face's mean grey level falls to 0.45-0.78
of the intact one and its edge share rises 1.5-24 times (the head camera separates every crushed face from every
intact one by its share of dark pixels below 0.6 of the intact median: at most 0.07 intact, at least 0.2 crushed; at
the infeed camera a white board drops least). Drawing only: no physics field, no geometry, no depth changes; the record lists the
switches (`frames.crushed`, only when there is one).

## 3. Budgets and end reasons (`cellkit/wire.py`)

Program time = the shift's wall clock minus the cell's own time inside `step` (simulation) and `get_camera`
(rendering, PNG and base64 included). Per shift:

| limit | value | end reason |
|---|---|---|
| program-time budget | 1.0 x duration + 60 s | `program_time` |
| one silence (reply sent to next request) | 20 s | `stalled` |
| start window (program start to its first request) | 120 s, inside the budget | `stalled` |
| a reply the program does not take | within the silence cap | `stalled` |
| request line | 4 MiB, one JSON object with a `method` | `protocol` |
| the program exits or closes the wire while the shift is live | | `crashed` |
| `finish()` (the client calls it when `solve` returns or raises; in process, `finalize()` of an open shift) | | `finished` |
| the shift's duration elapsed, or (line setting) its list's last parcel settled (the cell) | | `shift_end` |
| after the shift is over | 50 steps, 200 requests, 60 s | the cell's reason |
| CPU (all threads) | 4 x the budget | the program is killed: `crashed` |

A cell fault ends the shift with the cell's own reason (`physics_fault`, `internal_error`). The cell, the wire, the
commissioning service and the documents use the same names (the wire's `CELL_REASONS` only maps a legacy name). Whatever ends it, the
parcels already across the scan line keep their judgement; every other parcel is lost on the rated setting and not
done (no cost, no credit) on the line setting (`parcelcell/episode.py`). The request-line guard: any
parser failure (a line nested beyond the parser's limit included) is `protocol`; method names are cut to 64
characters, ids the reply cannot echo become null. Blank lines do not extend a silence.

## 4. Commissioning service (`environment/commission_service/`)

`commissiond.py` under its own account `parcelcell` (mode 700), started by `setup.sh` in a restart loop, socket
`/run/parcelcell/commission.sock` (666), status `/run/parcelcell/commission_status.json` (644), ledger and log in its
`state/`. One connection = one shift: the first request is `reset(site)` (site 1 .. 8); then the cell's public calls
only (`describe`, `get_state`, `step`, `get_camera`, `finish`; a second `reset` answers `one_shift_per_connection`,
anything else `forbidden`, counted), served by the grader's own `wire.serve_episode` (a byte copy) under the graded
shift's budgets; any end but `finished` sends `{"id": null, "event": "shift_over", "reason": ...}` and closes. Window:
12 shifts in the session, 2 at a time (a third connection's first request is answered `busy` and it is closed). It never answers truth, a score or a debrief.

Development shifts (`sites.json`, packages in `shifts/`; the rule is `tests/review/gen_shifts.py`'s line
development rule): line shifts under the label `line-dev` with the development set's own salt
(`shiftgen.line_seed("line-dev", k, mix=salt)`, `sites.json` `salt_hex`), shared with nothing a graded set uses.
Sites 1-3 are bench shifts (300 s at most) with stated contents, each the first candidate of its own k range that
holds exactly them: bench 1 (slow belt, running) one rigid parcel with its label underneath; bench 2 (medium,
running) a soft parcel with its label underneath and a rigid parcel with its label on a side face; bench 3 (slow,
indexing) a flat rigid parcel (below 0.11 m) with its label not up, a label-up parcel and a soft parcel with its
label not up. Sites 4-8 are the first draws (k = 4001 .. 4005) of five full shifts: slow running, medium running,
fast running, slow indexing, fast indexing. Their parcel looks are made at the image's build (the service's look
cache, `parcelcell.kitcache`), so a development shift starts in seconds.

`~/commission/run_shift.py` (agent side) runs a whole program as the grader does: a read-only copy, the grader's
launcher and minimal environment, the CPU cap, the no-network filter (the same code) and `choom`; it relays the wire to
the service and prints how the shift ended, frames and protective stops. No score.

## 5. Numerics

Every process the grader starts for itself (the verify process, which also runs every cell and the privileged
reference in-process) and the commissioning service run with `OPENBLAS_CORETYPE=Haswell OPENBLAS_L2_SIZE=256
NPY_DISABLE_CPU_FEATURES="X86_V4 AVX512_ICL AVX512_SPR"` (`test.sh`, `setup.sh`). `OPENBLAS_L2_SIZE=256` guards the
kernel pin on AMD Zen 4 / Zen 5 hosts, where numpy 2.5.x's OpenBLAS otherwise raises its block size for a forced
kernel and the Haswell kernel can crash or compute wrong float64 products with inner dimensions above about 300. The
program under test gets no pin (its minimal environment; the image's ENV sets none either), in the grader and in
`run_shift.py` alike. The parcels' looks (`parcelcell/parcelkit`, generated from each shift's package at the cell's
build) are byte-identical across processes under the pins; without them numpy dispatches AVX-512 kernels on hosts
that have them and the generated textures differ: every cell that renders for a program must run under the pins.

## 6. The two sets: the line set is graded, the rated set is the oracle's

The cell runs two settings, chosen by the shift package (`Shift.setting`; `parcelcell/world.py`, `episode.py`):

* the LINE setting (`"setting": {"name": "line", ...}` in the package; `grader_config.json` block `line`): the
  station's job. One parcel of the shift's list at a time, fed on settle (the next one is set down at the tick after
  the one before it crossed the scan line, fell or was damaged) on the belt's centre line at y -1.15..-1.00 with any
  yaw; rigid and soft parcels half and half, rigid 0.16-0.30 x 0.14-0.26 x 0.07-0.16 m, soft 0.16-0.30 x 0.14-0.26 x
  0.05-0.10 m, 0.2-2 kg, surface friction 0.3-1.5, the centre of mass up to 4 cm off; the label on any face whose two
  sides are at least 8 cm, uniform over those faces; belt 0.05-0.35 m/s in three classes; running or indexing (the
  belt stands while the station's parcel has its centre in y -0.15..0.10). Every parcel is in the duty (W = label
  not up, U = label up); a shift lasts at most 600 s and its list holds floor(600 / (2.025 / v + 4)) parcels (13 at
  0.05 m/s, 34 at 0.15, 49 at 0.25, 61 at 0.35); each parcel is due at the scan line by its set-down time + (0.95 -
  its set-down y) / v + 5 s (the station's rated cycle; an indexing halt counts against it) and a later crossing is
  not readable ("late": a W parcel earns nothing, a U parcel counts as made unreadable); a parcel not done when the
  shift ends costs nothing. The agent's documents describe this setting only.
* the RATED setting (packages without a setting): the station's rated job, on which the reference program is
  validated: parcels at stated arrival times with the generator's spacing, landing aligned against the near guide,
  cartons 0.16-0.30 x 0.16-0.30 x 0.11-0.18 m up to 3 kg and mailers, the duty of the score block (cartons with the
  label underneath or on the downstream face). It behaves exactly as the cell always has: the reference's records on
  its eight shifts are identical field by field, the cell's version string apart.

Which set a submission is graded on (`verify.graded_set`): the privileged oracle (its token's sha256 equals
`hidden/oracle_token.sha256`) runs the RATED set, `hidden/` (eight frozen shifts, its registered rule and draw below),
and its reward is its facing score against `anchor.json`'s value (1.0, full_at 0.98): this is what `ale validate`
checks. It shows that the job can be done: the reference turns 98-99 % of its in-duty parcels there. Every other
submission runs the LINE set, `hidden/line/`, and its reward is its facing score itself (`anchor.json` block `line`,
value 1.0, full_at 1.0): an absolute scale whose top, every parcel of every list turned and across readable in
time, no program has reached, so the scale leaves room above the reference. On the graded set the reference does
not do the job (below): its proof stays on the rated set. No input but the token chooses the set; a
package of the other setting in a set's folder is a task error; nothing in an unprivileged program's reach (its
files, its environment, the wire) can select the rated set or reach the reference.

The line set (`hidden/line/`): eight shifts drawn by its registered rule `hidden/line/selection_rule.txt` (written,
and its sha256 recorded, before the draw: 8cab6b5e6fb4f8850dd88265a870bfb6e05e580d82fe01b127371710ada710a7) with
`review/draw_line_set.py` on the frozen files (`frozen.json`, `review/freeze.py check`): slots 1 slow running, 2 medium
running, 3 fast running, 4 medium indexing, 5 fast running, 6 slow indexing, 7 medium running, 8 fast running; one
salt of 16 bytes from the system's entropy (`selection.json` `salt_hex`); each slot's shift the frozen line
generator's first draw for `line_seed("line-hidden-<slot>", 0, mix=salt)`; no quota, no admission, nothing run on a
shift before the set was frozen (`draw_line_set.py check` draws it again from the salt and compares the bytes). The
set (347 parcels, 237 W, 110 U; label positions at set-down; flat cartons = rigid parcels lower than 0.11 m):

| slot | belt class, mode | belt (m/s) | parcels | W | U | rigid / soft | flat cartons | label at set-down: top / underneath / downstream / upstream / near / far |
|---|---|---|---|---|---|---|---|---|
| 1 | slow, running | 0.116 | 27 | 21 | 6 | 15 / 12 | 3 | 6 / 6 / 2 / 6 / 2 / 5 |
| 2 | medium, running | 0.219 | 45 | 28 | 17 | 24 / 21 | 15 | 17 / 14 / 8 / 0 / 5 / 1 |
| 3 | fast, running | 0.279 | 53 | 33 | 20 | 24 / 29 | 9 | 20 / 7 / 8 / 4 / 5 / 9 |
| 4 | medium, indexing | 0.228 | 46 | 34 | 12 | 21 / 25 | 9 | 12 / 15 / 6 / 5 / 4 / 4 |
| 5 | fast, running | 0.276 | 52 | 34 | 18 | 32 / 20 | 18 | 18 / 9 / 6 / 6 / 5 / 8 |
| 6 | slow, indexing | 0.068 | 17 | 11 | 6 | 9 / 8 | 4 | 6 / 4 / 1 / 1 / 2 / 3 |
| 7 | medium, running | 0.240 | 48 | 31 | 17 | 23 / 25 | 10 | 17 / 16 / 4 / 3 / 4 / 4 |
| 8 | fast, running | 0.337 | 59 | 45 | 14 | 33 / 26 | 15 | 14 / 17 / 7 / 4 / 11 / 6 |
| all | | | 347 | 237 | 110 | 181 / 166 | 83 | 110 / 88 / 42 / 29 / 38 / 40 |

A parked robot (no program) scores 0 on every shift of the set: credit 0, nothing touched or lost (`review/gen_shifts.py`
replay); the example program scores 0 in the task image.

The reference on the graded set, as information (after the freeze; nothing was chosen by it): the frozen reference
through the privileged call, every method of its package switched on (the side-face modules, the end-face roll and
squeeze flip for upstream faces, the squeeze flip for soft parcels: options side "module", minus_y "roll_squeeze",
mailers "work"), the cell's physics and scoring with the cameras off:

| shift | score | W readable / W | U not readable | lost | protective stops |
|---|---|---|---|---|---|
| line_01 | 0.000 | 0 / 21 | 0 | 0 | 0 |
| line_02 | 0.000 | 0 / 28 | 0 | 0 | 0 |
| line_03 | 0.000 | 0 / 33 | 0 | 0 | 0 |
| line_04 | 0.000 | 0 / 34 | 0 | 0 | 0 |
| line_05 | 0.000 | 0 / 34 | 0 | 0 | 0 |
| line_06 | 0.000 | 0 / 11 | 0 | 0 | 0 |
| line_07 | 0.000 | 0 / 31 | 0 | 0 | 0 |
| line_08 | 0.000 | 0 / 45 | 0 | 0 | 0 |
| mean | 0.000 | 0 / 237 | 0 | 0 | 0 |

It turns nothing there: its scheduler admits a job only when the next parcel of the list cannot reach the work zone
before the job's end, and it takes every parcel not yet set down as waiting at the set-down line, so on the line,
where the next parcel comes only once the present one is settled, it lets every parcel through.
Of the set's 110 U parcels it left 92 readable; the other 18 were not done (never set down, or in the station when an
indexing shift ended: its belt stood behind a parcel left in the zone).

Evidence that every parcel class of the line set can be turned at least sometimes. Per class (W parcels: kind, flat or
not, label position at set-down): on the graded set (the run above) and on 60 single-parcel line shifts per class
drawn with a seed label of their own (shared with no graded or development shift; the bench profile with one parcel of
the kind, the belt running, belt class and speed as the generator draws them; binned by the parcel's class; 900
shifts), where the reference runs with every method on and its admission off (scheduler options t_job 0 and
deadline_abort off: with one parcel nothing comes behind it). Each cell gives n, label up at the scan line, of those
in time, readable:

| class (W: arrives label not up) | graded set: n, label up, in time, readable | single-parcel line shifts: n, label up, in time, readable |
|---|---|---|
| rigid-flat downstream | 17, 0, 0, 0 | 60, 8, 5, 5 |
| rigid-flat far_side | 9, 0, 0, 0 | 60, 2, 1, 1 |
| rigid-flat near_side | 7, 0, 0, 0 | 60, 0, 0, 0 |
| rigid-flat underneath | 19, 0, 0, 0 | 60, 1, 0, 0 |
| rigid-flat upstream | 8, 0, 0, 0 | 60, 0, 0, 0 |
| rigid-tall downstream | 14, 0, 0, 0 | 60, 15, 8, 8 |
| rigid-tall far_side | 20, 0, 0, 0 | 60, 6, 1, 1 |
| rigid-tall near_side | 19, 0, 0, 0 | 60, 2, 2, 2 |
| rigid-tall underneath | 17, 0, 0, 0 | 60, 8, 0, 0 |
| rigid-tall upstream | 10, 0, 0, 0 | 60, 0, 0, 0 |
| soft downstream | 11, 0, 0, 0 | 60, 2, 2, 2 |
| soft far_side | 11, 0, 0, 0 | 60, 4, 0, 0 |
| soft near_side | 12, 0, 0, 0 | 60, 2, 2, 2 |
| soft underneath | 52, 0, 0, 0 | 60, 0, 0, 0 |
| soft upstream | 11, 0, 0, 0 | 60, 0, 0, 0 |

Records of this station's methods on the rated setting's aligned landings (yaw within 8 degrees of a quarter turn,
against the near guide): label underneath 296 of 300 (98.7 %) by the reference's half turn; downstream face 296 of
300 (98.7 %) by its quarter roll; upstream face about 88 % (265 of 300 strict) by the end-face module's roll and
squeeze flip; a near or far side face 84-88 % (264 of 300; holdout 253 of 300) by the side-face module; flat cartons
below 0.10 m about 66 % by the end-face module; outside this station's records, a truth-reading scripted routine on
a comparable two-arm station model turned 4 of 10 soft parcels. Label-up parcels need nothing: the parked robot leaves
every one readable (the physics check).

Classes with a readable turn on record at the line's set-down (the single-parcel shifts): rigid-flat downstream;
rigid-flat far_side; rigid-tall downstream; rigid-tall far_side; rigid-tall near_side; soft downstream; soft
near_side.
Classes with no success on record at the line's set-down: rigid-flat near_side; rigid-flat underneath; rigid-flat
upstream; rigid-tall underneath; rigid-tall upstream; soft far_side; soft underneath; soft upstream.
Where a class shows label-up crossings but nothing readable, the turned parcels crossed after their due time
(the reference's half turn of a tall carton with the label underneath takes longer than the rated cycle allows) or
crossed unsettled.

A program's turns on the graded set itself, as information (one episode of a frontier model on 2026-11-21, graded as
above: facing score 0.0128, 3 of 237 W parcels readable; per parcel from a re-run of its submission through the
grader's wire and cell on the build host (outside the task image), which gave the episode's shift scores, mean frame
count and step count exactly; the parcel's `i` counted from 0 in the shift's list): readable turns of a soft parcel
with the label underneath (line_07, parcel 41) and of two tall rigid cartons with the label on the far side (line_01,
parcel 17; line_08, parcel 2). Three parcels with the label underneath were turned label up but crossed the scan line
after their due time (6.7-12.3 s against the belt's own ride, where 5 s is allowed): a flat rigid carton (line_05,
parcel 42) and two soft parcels (line_01, parcel 9; line_05, parcel 15). The two lists above are the reference's; with
this program, soft underneath has a readable turn on record at the line's set-down too.

The rated set (`hidden/`): eight shifts, selected by its registered rule `hidden/selection_rule.txt` (written, and its
sha256 recorded, before the draw) with `review/select_hidden.py` inside the task image on the frozen files: slots
1-8 = slow, medium, fast, medium, fast, slow, medium, fast; one salt of 16 bytes drawn once from the system's entropy
(`selection.json` `salt_hex`); each slot's candidates `labelled_seed("hidden-<slot>", k, mix=salt)`, k = 0, 1, ...,
drawn by the frozen generator in the slot's belt class; the first candidate that meets the quotas (at least 6 W
parcels, of them at least 3 with the label underneath and 2 on the downstream face; 2 label-up cartons; 2 mailers; 3
cartons outside the duty; the natural end at most 600 s; 2 W cartons longer than 0.26 m along the belt) and whose two
reference runs leave every W and U parcel readable, lose or damage nothing, have no stop and no limiter clip and give
the identical record, is kept: 37 candidates, 9 met the quotas, 8 admitted (`selection.json`). The set: 15-24 parcels
a shift (159 in all), 58 W parcels; each shift lasts 600 s. `verify.py` makes every graded shift's parcel looks once
before the first shift (worker processes, a root-only cache). A parked robot (no program) scores 0 on every shift of
either set. The development shifts (`environment/commission_service/sites.json`) share no seed, salt or label with
either set (the repository test of this task checks it).

## 7. Verify time (task.toml carries the arithmetic)

Measured in a container of the task image on the build host (8 CPUs, 16 GB; the grader's environment: the cameras'
colour by the filament backend on lavapipe, depth by the classic renderer on OSMesa, 8 rasteriser threads, the
one-lamp look; host load average about 180 on 48 cores), on the graded set's fastest shift (59 parcels at 0.337 m/s):
the 59 parcels' looks made with 4 worker processes in 258 s (4.4 s a parcel); a cell builds from the look cache in
2.1 s, its first frame (the renderer's start included) 12.0 s; a new frame p50 0.327 s, p95 0.535 s over the four
cameras (per camera p95: head colour 0.448 s, infeed colour + depth 0.556 s, wrist_left colour 0.581 s, wrist_right
colour + depth 0.510 s); the simulation 0.108 wall s per simulated second through the server; peak RSS of one cell
2.66 GB. Switching a damaged parcel to its crushed look costs 0.16-0.34 s once. With every graded shift at its full
600 s, its full frame and program-time budgets, one re-run and the 347 parcels' looks, the derivation in task.toml
gives 21200 s (x 1.35). The oracle's branch (the rated set) took 1259 s in one fresh full verify in this image (the
159 parcels' looks 483 s, 88-100 s a shift). On this host at load 12-18 a frame cost 0.19-0.27 s p95 and at load
100-145 up to 1.03 s p95: re-measure on the grading host if it is busier than at the measurement.

## 8. Records

`/opt/arb-verify/` (root-only, outside `/opt/ale`, which the engine clears before it keeps a sandbox): `logs/epNN.log`
(the program's stderr), `out/shifts.json` (every shift's record: per parcel its class, scan crossing, readability and
the failed clauses; safety; frames; budgets used; end reasons) and `out/log_lines.txt`. Never a scoring input.
