# `dexterous_hand_full_assembly_16dof_panda`: notes for maintainers

This folder is the grader. It never reaches the agent: the engine uploads it into a separate verifier sandbox after the
agent phase. The sources and credits of the hand, the cell and the room are in `NOTICE.md` beside this file.

## What the robot in this task is made of

The hand is an open-source four-digit robot hand of 16 joints with 13 identical servos (each finger's last joint has no
servo: it follows the joint before it through a four-bar loop), built from its designers' nine bolt-on modules: three
identical finger modules, the thumb module, the electronics stack, the upper shell, the dorsal cover, the power hub and
the press-fit cap, set onto the palm's bottom shell, which stands dorsal side up on a raised fixture. Every unit is the
designers' released CAD geometry, tessellated: nothing is modelled again, scaled or simplified. Each servo is drawn from
its maker's CAD model at the designers' placement and collides as the designers' one-solid servo. The collision pieces
are convex pieces cut from the true solids. The digit modules keep their joints, unpowered and held by friction (0.02 N
m), with the stops and four-bar loops of the released solids. The Panda is the MuJoCo Menagerie model with its stock
gripper and its own pads.

## The episode and the grade (`verify.py`)

1. Lockdown before anything else (stdlib only): the agent account's processes are stopped and killed, the stage is
   closed; then one walk of the submission as restored that only stats (os.scandir and lstat, iterative, nothing opened)
   checks the three caps of the route gate below. Every entry counts here (files, folders, symbolic links, special
   files, `__pycache__` and its content), and the walk stops as soon as the entry count passes its cap. Above a cap
   nothing of the submission is copied or read: no late-write walk, no snapshot, no source audit, no import check, no
   episode, and the verdict carries the cap's reason (`contract_ok` zero, reward zero, written as every other contract
   failure). Within the caps the submission is snapshotted root-owned, late writes into the submission void the
   contract, late entries elsewhere are quarantined.
2. `/opt/arb` is checked file by file against `given_hashes.json` (`tests/test.sh` builds it from the image's task
   assets and `setup_payload/`); the hidden kit is fetched (or, before its upload, read from `assets/gt.tar.xz`) and
   checked against the digest and file count of `grader_config.json` `hidden_gt`.
3. The contract gate: `assemble.py` is a regular file that parses and defines `solve`; the route gate (no weight files,
   no training calls, a size cap, an entry cap, a path cap); the network audit (no networking or computed import; no
   outbound network activity while a program runs); the import under the run regime (at most 120 s of wall clock and 120
   s of CPU); at least one `step` (or the reference submission's `reference_assemble`); no surviving process (every
   process of the run account is killed when an episode ends; one still running 8 s after that kill breaks the gate).
   Before the episodes the log says only whether the checks before them passed
   (`contract check passed before the episodes ...` or `contract_ok 0: <reason>`); a zero from an episode gate is logged
   after the episodes with its rule (`contract_ok 0: no step call over the episodes ...`,
   `contract_ok 0: network activity during an episode ...`,
   `contract_ok 0: <n> process(es) of the run account still running ... after the kill ...`; in this order, the
   instruction's). The gates fail closed: sizes, the entry count and the paths' length are taken (stat only) before any
   file is copied or read, and a submission larger than `route_gate.max_submission_mb` (60 MB; the sum of the sizes of
   every entry that is not a folder, a sparse file by its apparent size), with more than `route_gate.max_entries` (10000
   files and folders), or with a path inside it longer than 4000 bytes (`verify.py` `MAX_PATH_BYTES`; the grader's
   copies of the submission must stay below the kernel's path limit) is refused with the reason
   `route gate: submission larger than 60 MB`, `route gate: submission has more than 10000 files and folders` or
   `route gate: submission has a path longer than 4000 bytes` (the entry count first, then the size, then the path
   length when several apply; a missing `assemble.py` is named before all three); only regular files are read (the
   snapshot drops symbolic links, FIFOs, devices and sockets). The grader itself does no work on the submission's bytes
   at all: it stats, copies and hashes them, and every decode, split, string check and parse happens in one child
   process per verify (`verify.py` `source_audit`, root, `oom_score_adj` 1000; it reads, decodes, scans and parses, it
   executes nothing). The child handles every regular `.py` and `.sh` file and returns, per file: a `.py` that declares
   a text encoding other than UTF-8, ASCII or Latin-1 (the coding line of its first two lines, decided before the body
   is decoded); the network audit's hits; the parser's error; whether `assemble.py` defines `solve`; the route gate's
   forbidden-call strings (streamed, `.py` and `.sh` alike); and the count of suspicious lines. Of the network audit's
   hits it sends at most 5 per file (each name at most 200 characters) and counts them all (`network_imports`): the
   grader holds the first 5 of them and names those. The child's address space is limited to `grader_config.json`
   `parse_limits.address_space_gib` (12 GiB), each file to `parse_limits.per_file_s` (300 s) of wall clock, and the
   whole audit to `parse_limits.total_s` (1800 s) of wall clock. A file beyond the per-file limit is a contract failure
   with its own reason (`<file>: cannot be parsed within the grader's limits (12 GiB, 300 s)`; the record
   `source_parse_limit` is set, and the file is not counted in `network_imports`), and an audit beyond `total_s` has its
   own reason (`the source audit did not finish within 1800 s`). Measured in the image: 40 MB of one-statement lines are
   refused at the address-space limit, a 10 MB module holding one list of digits takes 4.7 GiB in the child and passes.
   A `.py` that declares a disallowed text encoding is a contract failure
   (`declares the text encoding <name>; sources must be UTF-8, ASCII or Latin-1`; for a helper the reason is a
   source-audit reason, `source audit: <file> declares ...`, counted in `network_imports` as the audit's hits are);
   `.sh` files declare nothing and are read as UTF-8 with replacement. A `.py` file other than `assemble.py` that cannot
   be read or does not parse is a network-audit hit (`cannot be parsed (<error>)`). Any other exception in the part of
   the grader that depends on the submission (the token, the contract check, the episodes from the program's launch on,
   the scoring) writes a verdict with `grader_exception` set, `contract_ok` zero and reward zero (the traceback goes to
   `/opt/arb-verify/logs/grader_exception.log`); a fault of the task's own inputs (config and every value the grading
   reads from it, givens, kit, the record folders, the cell's build at an episode's start) stays a task error (a nonzero
   exit, no verdict): those are read and made before the line. Every process of the program starts with `oom_score_adj`
   1000 (`choom` inside runuser and prlimit; the memory killer's first choice in the verifier's one memory cgroup), and
   the root sampler that walks the program account's processes every `socket_sample_s` (0.5 s) KILLS (SIGKILL) any of
   them found below that value: a legitimate program never lowers its own value, and a process that does is stopped at
   the next sampler pass; a stopped process is a crashed process (the episode goes on under its crash rule), and the
   kills are a record only (`runner_oom_adj_killed`; no gate). The launch chain's own helpers (runuser / prlimit /
   choom), which run for a few milliseconds at the inherited value before choom writes the value, are told apart by
   their executable name and are never killed. A program that lowers its own value and fills the memory within one
   sampler pass can still make the grader the largest process and end the verify without a verdict; that residual is
   accepted. `tests/test.sh` lowers the grader's own value to -500 where the container allows it. `suspicious_lines` (a
   record only) is `-1` where it is not computed: no `assemble.py`, a submission above a cap, a file beyond the parse
   limits, an `assemble.py` that declares a disallowed encoding, does not parse or defines no `solve`, and a
   `grader_exception` verdict whose exception came before the route gate.
4. `episodes` (3) seeds drawn from OS entropy; per episode a fresh `HandInfoServer` (`arm/assembly_hand/server.py`) on a
   drawn layout (fixture, trays, chain angles, flat parts' directions, cameras, lights, F/T bias), checked for reach
   (`reach_check`: a failed draw is drawn again, at most 8 draws, the nominal layout last); the program runs as `nobody`
   from a read-only copy in a run folder named by the episode's index (`agent_ep01`, ...; never by the seed, which
   drives the public camera draw). An agent's episode ends at 1200 s simulated, 2400 s of wall clock (the privileged
   reference: `privileged_wall_cap_s`, 7200 s), 120 s of silence or 9600 s of CPU per process (`cpu_cap_factor` 4 times
   the wall cap); after 600 camera frames further frames are refused.
5. Episode score = the LOCKED mates' weight / 10.5 (four digits 1.5 each; the stack, the upper shell, the dorsal cover
   and the power hub 1.0 each; the cap 0.5), times `scoring.safety_violation_factor` (0.3) when the episode had a safety
   violation. `assembly_score` = the episode mean times the contract gate; `reward` = `assembly_score` (`anchor.json`:
   value 1.0, full_at 1.0).
6. A replay trace of every agent episode is written under `/opt/arb-verify/out/trace/` (root-only; never a scoring
   input), with `episodes.json` and the program's stderr under `/opt/arb-verify/logs/`.

The verifier's timeout in `task.toml` (14400 s) covers the worst case term by term: 1800 s (the source audit) + 120 s
(the import check) + 3 x (8 layout draws x 3 s to build and settle a cell + 2400 s) = 9192 s, which leaves 5208 s for
the sandbox's start, the copy of the givens and the kit's fetch. The reference submission's verify took 1003 s in the
validate of record.

## Fastening (`arm/assembly_hand/coupling.py`; the numbers in the kit's `mating_graph.json`)

| kind | units | how a unit is caught and fastened |
|---|---|---|
| `module_slide` | the four digits | a finger base at least 8 mm inside its channel, between both walls and on the floor, is guided across its channel only (the robot pushes it home against the bay wall); the thumb stands against its wall (no guide); a digit locks at rest within 0.5 mm of its wall, its mounted joint ranges set at the lock |
| `part_seat` | stack, upper shell, dorsal cover, power hub | caught once it is within 3 mm above home and nearly at rest; then centred across the seat (from at most 2 mm off, at 10 mm/s), never moved along the insertion; locks at rest at most 0.5 mm above home while its seat and the parts fastened under it carry at least 0.5 of its weight along the insertion over the hold ticks (`part_seat.support_share_min`; the gripper and friction on side faces do not count, so a plate hung between side faces or held in the window does not lock and one pressed on its seat while held does); the stack turned half a turn locks as LOCKED_OFF (counts nothing, counts as fastened for the order rules) |
| `pressed_seat` | the cap | a press fit (3 N of friction on its rail); locks at rest within 0.5 mm of home, no support rule |

A mate locks only while every unit it waits for is fastened and the tool access of its screws is free of every other
unit; a unit in the way stops a mate silently; a fastened unit cannot be taken off. The thumb waits for the index
finger; the stack for the four digits; the upper shell for the stack; the cover for the upper shell; the hub for the
cover; the cap for the hub. In the complete builds of the cell's last package the plates lay at most 0.20 mm (the
stack), 0.24 mm (the upper shell, which comes to rest on one point of its seat: it is let go above its seat and locks by
settling inside the lock window, without a push), 0.01 mm (the cover) and 0.00 mm (the hub) above home at their lock
tick, and the cap 0.22 mm.

## The hidden kit

`gt.tar.xz` (91 files, digest `9f72744d49e4181b4dbaa4249c43c2c040c105d02024a242b8116c72cf3e236c`): `kit.json` (units,
links, frames, masses), `mating_graph.json` (mates, weights, design poses, insertion axes and depths, the order rules,
the tool access volumes), `joints.json` (the joints as simulated), `meshes/visual/` (the render set the cameras draw)
and `meshes/collision/` (the convex pieces; `collision.json` records the two slots, the three side faces set back by
0.05 mm and the servo's five slabs). It is never in the image. Until it is on the benchmark's asset dataset
`grader_config.json` `hidden_gt.url` is a placeholder and `tests/test.sh` points the grader at `assets/gt.tar.xz`
(ignored by git); the maintainers' build tool rebuilds it byte for byte. `hidden/bench_plan.json` is the reference's
plan (order, grasps, release heights, push poses, park angles, planner seed); `hidden/manual_gt/photos_gt.json` the
picture manual's camera plan; the third file under `hidden/` holds the digest of the reference submission's token
(`verify.py` `token_state` names it).

## The reference submission

`solution/solve.sh` installs `solution/reference/assemble.py` and, beside it in the submission folder, the secret token
(a hidden file; `verify.py` `token_state` names it). With the token the cells are privileged: the nominal layout, and
the call `reference_assemble`, which runs `arm/assembly_hand/reference.py` inside the cell through the cell's own
control ticks and safety watch (exact unit poses, the cell's own model for planning and collision checks). It takes one
frame per camera first, so that every validation also goes through the agent cameras' path. On the nominal layout it
fastens every mate in every episode in about 208 s simulated and 332 s of wall clock per episode, with no safety event
(`validate-92ed8b66`: the whole verify 1003 s; 709-1446 s over the last four validates, by host load). A wrong token is
ignored (the verdict's token-mismatch record is set) and the privileged calls answer `forbidden`. The token is stat'ed
before it is opened: a token file that is not a regular file, or is larger than 4096 bytes (`verify.py`
`TOKEN_MAX_BYTES`), is a mismatch without being opened; a smaller one is opened without following a link and hashed in
pieces, its surrounding white space left out.

## The four safety rules (`grader_config.json` `safety`, `damage`, `stops`)

| rule | what counts | limit |
|---|---|---|
| R1 robot contact | a Panda link other than the two finger (pad) bodies pressing on anything, per tick | 20 N |
| R2 unit loaded | (a) the normal forces between a unit and anything but the pads, contacts inside what is fastened together left out; (b) the net force a fastened or engaged unit's fastening carries | 25 N for 0.2 s |
| R3 joint stop | the torque a product joint carries against its mechanical stop | 0.20 N m for 0.2 s |
| R4 unit speed | a unit's centre of mass faster than the limit (a unit that falls off the bench counts here) | 3 m/s |

Each number is `max(proposal, min(cap, round_up(1.5 x the reference's largest reading)))`, frozen after the first full
reference. The validate of record reads, at most, R1 0 N, R2 6.26 N, R3 0.083 N m and R4 0.46 m/s; each number is
certified by a probe just below and just above it in the grader's QC table. The drop watch (`drop`: 0.25 m/s) is a
statistic only: parts that the reference lets go just above their seats count as "dropped" in the reference submission's
statistics (`dropped_parts_mean` 2).

## Running the verify stage by hand

```
docker run --rm --cpus 8 --memory 24g --network none -u root \
  -v <a copy of this folder>:/opt/ale/tests -v <submission folder>:/home/user/submission \
  -v <the payload folder>:/opt/task-assets/payload:ro -e ARB_RECORD_ROOT=/rec -v <record folder>:/rec \
  -e ROBOTICS_GRADER_VERDICT_PATH=/rec/verdict.json <the task image> bash -c 'cd /opt/ale/tests && bash test.sh'
```

Copy the submission before the stage folder: the late-write gate compares the submission's times with the stage's upload
time. Set `episodes` in the copy's `grader_config.json` to shorten a run. The grader's QC table (degenerate programs
through this path, the order rules with the reference's own steps and with jigs, the safety probes, a unit off the
bench, the reach check on fresh layouts, the cap's press fit, the run folder's names, the hidden paths) is kept with the
maintainers' build tools, each row's expected verdict registered before it ran.

## Known limits

- The reference runs on the nominal layout only; drawn layouts are checked for reach, not built.
- Parts let go from above land with a short contact transient, and solids that are line to line at home overlap while a
  plate is centred. The clipping audit reads both by the contact model: at a landing a solid pair may overlap by at most
  1.25 times v x tau / e above its own overlap at home (the speed at first contact, the pair's time constant); between a
  unit's first contact with its seat and its lock an overlap may reach the unit's own offset from home plus 0.05 mm;
  anything else stays within 0.05 mm of the pair's overlap in the finished hand. Over the cell package's ten distinct
  runs the audit found no defect: the impact peaks were at most 0.63 of their limits, and the smallest margin of a
  capture row was 0.051 mm (the upper shell's plate against the thumb base servo). A landing can read far above the R2
  number for a tick; R2 counts only readings sustained for 0.2 s.
- The thumb's side joint is stopped by the collision pieces towards the index finger at -35 deg with the index finger
  straight and at -9.2 deg with the index finger's sideways joint at -44.9 deg; its mounted range ends at -36.1 deg.
- Camera colour frames differ slightly between processes (Filament's soft shadows); depth comes from MuJoCo on the same
  geometry.
- Filament prints `PanicLog` lines about its handle arena into the verify log; they are notices, not failures.
- The merged STL files of the two digit modules have more than 200,000 faces (MuJoCo's limit); the cell manual names the
  per-link files for MuJoCo.
- The grader does not bound what a program writes at run time: the output of the import is captured whole, an episode's
  stderr log has no size limit, and the walk of the watched folders after a run holds every new entry. The first and the
  third end in a verdict by the rules above (the program is the memory killer's first choice; a grader exception is a
  zero); the second can fill the sandbox's disk.
- The room is the baked hardware lab bay of the payload's `render/` folder and the picture manual is the full render of
  25 photographs; the camera checks, the frame times and the comparison of the given render layer with the grader's
  frames were run again on them.
