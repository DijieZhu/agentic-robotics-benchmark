# racecar_requirements_replica_from_photos: grading notes for maintainers

The agent never sees this folder. The grader is static and deterministic: it reads the three submission files as
data (YAML and JSON, never executed), evaluates the public procedure over every combination of the catalogue on
the real car, and scores the submission against the hidden truth. There is no judge and no played party.

## What runs

`test.sh` (root, cwd = this staged folder, in the separate verifier sandbox; `task.toml` sets
`environment_mode = "separate"`) closes the folder, pins every numerical library to one thread and runs
`python3 -I -B verify.py`. `verify.py`:

1. **Lockdown (gate G1).** Freezes and kills the agent account's processes (in the separate verifier only the
   image's own respawned desktop; a process that cannot be stopped fails the verify), locks this folder, copies the
   three submission files into a root-only snapshot and sets `gate_g1_submission_integrity` to 0 when the
   submission root is not a directory or an entry is linked, special, larger than 256 kB or changed after this
   folder arrived. The code is carried over unchanged from an earlier requirements task of this platform, apart
   from the three file names, the snapshot path and the metric name.
2. Calls `grade.grade_stage(STAGE, snapshot, g1=...)` and writes the verdict through the kit
   (`robotics_grader.ratio_rewards` with `anchor.json`, then `write_verdict`). A `grade.WorldError` ends the run as
   a **task error** (`robotics_grader.fail`), never as a score.

## Files

| file | what | origin |
|---|---|---|
| `test.sh`, `verify.py`, `grade.py`, `anchor.json`, `NOTICE.md`, this `README.md` | the verifier and its notes | hand-written |
| `procedure.py` | byte copy of the agent's public procedure (`rc-procedure/2`); its sha256 is pinned in `grader_config.json`, and `grade.py` executes exactly the bytes it hashed | copied |
| `schema_check.py` | byte copy of the agent's structure check, pinned in the same way; the grader reads the submission through it, so the agent's check and gate G0 take one loading path | copied |
| `schemas.json` | `{file name: schema}` of the agent's three submission schemas, with the SKU codes of every purchase line and the source ids of this instance | generated |
| `grader_config.json` | weights, C1 groups and bands, C3 band, change fee, limits, tolerances, block size, the two pins, the closed lists, the world file names | generated |
| `robotics_grader/` | byte copy of the repository's `shared/robotics_grader` | copied |
| `gt/catalogue.json`, `gt/world.json` | the hidden world: the catalogue as the procedure reads it (page values as corrected by the errata); the 17 pass items, the conditions of the 7 derived quantities, the lots as they arrived, cycles and scenarios | the world solve |
| `gt/car_truth.json`, `gt/key_quantities.json` | the real car: its car record (`rc-car/1`, with the pod) and the 33 quantities of the sketch | the truth build |
| `gt/reference_sheet.json` | the 20 reference facets: value, unit, tolerance, sources | generated |
| `gt/optimum.json` | the cheapest acceptable car, its price and a digest of the acceptable set | generated; recomputed at every verify |
| `given_hashes.json` | sha256 of every given file, the maintainers' pin | generated |

The generators (the truth build, the world solve, the photo set, the document pages, the stage builder) are the
task's development tools and are not part of this repository.

## Gate G0 (structure; a failure scores 0, never a task error)

The three files exist, are regular files of at most 256 kB, are UTF-8 and parse (YAML with a node budget of
20000, alias cycles, over-long integers, duplicate keys and keys that are not strings refused; strict JSON:
duplicate keys, `NaN` and `Infinity` refused); every number is finite and every integer at most 2^53 in
magnitude; each file matches its schema (JSON Schema 2020-12: the closed key sets, the facet ids, the source-id
pattern of this instance, an integer `nominal` for a count, `tolerance` not negative); no facet id appears twice;
every line of the BOM holds an SKU of that line. Whether the BOM can be built is not a structure rule: it is the
pass item `FIT`. The diagnostic (at most 2000 characters) names the submission's own content and public constants
only.

## Recomputation (any difference is a task error)

The grader evaluates the 17 pass items on every car of the catalogue (`itertools.product` order over
`SLOT_ORDER`, block by block through `procedure.evaluate_block`) with the real car record and the arrival lots,
takes the acceptable set, its cheapest car and that car's price J* (ties: the earlier car), and compares them
with `gt/optimum.json`: the price within 1e-12 relative, the car exactly, the acceptable set by its count and the
sha256 of its canonical list. The number of cars must equal `cars` of the config. These checks run whatever the
submission is. Messages name files and checks, never a value.

## Criteria

`replica_score` = G0 x G1 x (0.30 C1 + 0.10 C2 + 0.20 C3 + 0.10 C4 + 0.30 C5):

- C1 `kinematic_replication`: per sketch quantity, a length scores 1 for an error of at most
  max(1.5 % of the truth, 1.0 mm) and 0 from max(6 %, 4.0 mm), linear between; an angle 0.5 / 3 degrees; a
  count must be equal. Group means weighted heights 0.30, layout 0.25, steering 0.35, chassis 0.05, gears 0.05.
  `tolerance` and `sources` are not scored.
- C2 `requirements_sheet`: a facet counts when its value is within the reference tolerance, its unit is the
  reference unit string and one of its sources is a reference source; C2 = 2 x counted / (20 + submitted).
- C3 `derived_requirements`: the mean over the 7 predictions of ramp((zero − e) / (zero − full)), e the relative
  error against the procedure's value for the real car with the agent's BOM and the arrival lots, full 3 %, zero
  15 %; a wrong unit string or a procedure value that is not finite scores 0.
- C4 `bom_acceptance`: 1 when all pass items hold for the agent's BOM on the real car, else 0.
- C5 `bom_economy`: J_eff = min over the acceptable cars of price + ECO x (lines that differ from the agent's
  BOM), ECO = 45 USD (brief B-14); C5 = clip(1 − (J_eff − J*) / ECO, 0, 1).

The reference submission (`solution/submission/`: the sketch equal to `gt/key_quantities.json`, the facets equal
to the reference sheet, the cheapest acceptable car as the BOM, and the procedure's own values as the predictions)
scores exactly 1.0 on every criterion.

## Metrics

Always the same key set, finite floats only: `gate_g0_structure`, the five criteria, `c1_group:<group>`,
`sketch_score:<id>`, `facets_counted`, `facets_submitted`, `facet_counted:<id>`, `derived_score:<id>`,
`derived_relative_error:<id>`, `acceptance:<pass item>`, `bom_cost_usd`, `effective_cost_usd`,
`optimum_cost_usd`, `eco_lines`, `bom_lines_differing_from_optimum`, `passing_cars`, `cars_enumerated`;
`verify.py` adds the lockdown metrics and the kit adds `ratio`, `anchor_recorded` and `replica_score`. The metrics
hold the optimum's price and the per-item results; they reach the public task log with the trajectory.

## Running it by hand

In the task image: `ROBOTICS_GRADER_VERDICT_PATH=/tmp/verdict.json bash test.sh` in the staged folder, as root.
`task.toml` gives the verifier's machine, its time limit and how the limit was derived from the measured time of
one verify.
