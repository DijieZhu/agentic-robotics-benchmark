# The task template

`task/` is a complete robotics task. It passes `ale lint` and `ale validate` as shipped,
except that its anchor is null, so the oracle scores 0 until a measured value is recorded.
Copy it from a checkout of this repository, rename it, and change one thing at a time from
a state you know is green.

The folder contract is upstream's, in Harbor's task format (`task.toml`, `environment/`,
`tests/`, `solution/`):
https://github.com/AgentsLastExam/ale/blob/main/docs/specs/task-authoring.md (the engine is
pinned at `vendor/ale`, and `vendor/ale/docs/specs/` holds the specs of the pinned version).
This file says what each piece in the template is for and what to change.

This README lives outside `task/` on purpose. `ale lint` admits only `task.toml`,
`instruction.md`, `environment/`, `tests/`, `solution/` and `.ale-cache` at a task's top
level (a top-level `tools/` fails it), and the template must itself lint clean.

## The files

```
task/
├── task.toml            strict: an unknown key fails. [task].name is
│                        "grounded-intelligence-initiative/<slug>"; [metadata].platform;
│                        resources, budgets, network, artifacts. No admission bar in here.
├── instruction.md       the prompt. The only thing the tested agent is told.
├── environment/         the agent's machine
│   ├── Dockerfile       the image, built from environment/ without setup/: FROM the ALE
│   │                    base at an exact version, py3.12 /opt/venv, agent-UID import self-check
│   ├── requirements.txt the one list of Python dependencies; the Dockerfile installs it
│   └── setup/           not in the image; uploaded to /opt/ale/setup when setup runs
│       ├── setup.sh     trusted root, before the agent: stages everything under payload/
│       │                into the agent home (no file names in the script)
│       └── payload/
│           ├── template_env.py   the agent's own copy of the environment (byte-identical
│           │                     to tests/env.py; a test holds the two together)
│           └── practice_grader/  a self-test on visible seeds; prints the raw metric, never a score
├── tests/               the grader. Absent while the agent works; uploaded root-only to
│   │                    /opt/ale/tests at verify time
│   ├── test.sh            `exec "${ALE_VERIFY_PYTHON:-python3}" verify.py`
│   ├── verify.py          computes the metric under the grader's own physics, writes the verdict
│   ├── env.py             the authoritative environment copy the grader imports
│   ├── anchor.json        the metric and its direction, the hidden denominator, the saturation point
│   ├── grader_config.json grader knobs: episodes, probe_rate, step_timeout_s, scoring_cap
│   └── robotics_grader/   the vendored kit (canonical copy: ../shared/; byte-equality in CI)
└── solution/            the reference solution. `ale validate` uploads it to ~/.ale-solution and
    │                    runs it in place of the agent: same unprivileged user, timeout and network
    ├── solve.sh           leaves the reference submission in /home/user/submission/
    └── solver.py          the reference controller that solve.sh copies there
```

The template uses no assets, no MCP servers and no Skills. Add them if your task needs
them; the authoring spec says how.

- Large agent-visible files (photo sets, scans, meshes) come from the benchmark's Hugging
  Face dataset while the image builds: `environment/assets.sha256` pins them and the
  vendored `environment/fetch_assets.py` fetches them
  ([`docs/task-assets.md`](../docs/task-assets.md)). This repository does not use
  upstream's `ale assets` folders (`environment/assets/`, `tests/assets/`,
  `solution/assets/`).
- An MCP server is a program the image installs, declared in `task.toml` as
  `[[environment.mcp_servers]]` (`name`, `transport`, `command`, `args`). There is no `cwd`
  or environment key: a server that needs either starts through a wrapper script in the
  image.
- Skills are built into the image, and `[environment].skills_dir` names their folder.

## What the engine provides at verify time

Verification runs in a **separate** sandbox, the recommended default this template ships
(`[verifier] environment_mode = "separate"`): once the agent is done, the engine releases
the agent's sandbox, provisions a fresh verifier from the task image that the agent never
touched, restores only the declared `artifacts` into it, uploads `tests/` to
`/opt/ale/tests` (root-only, like all of `/opt/ale`) and runs `bash test.sh` there as root,
with that folder as cwd and the network open. Shared verification (omit the key) instead
runs `test.sh` in the agent's own sandbox, where a program the agent planted on root's PATH
(the base image leaves `/usr/local/bin` agent-writable and ahead of `/usr/bin`) would run
as root before the grader and bypass it; separate verification closes that, so prefer it for
every task. Two consequences of the fresh sandbox: `environment/setup/setup.sh` does **not**
run in the verifier, and nothing the agent left outside the declared `artifacts` is there —
so anything verify reads must be staged by the task (in `tests/`, or a declared artifact),
not assumed from setup. Before the script starts, the engine

- stages its own `ale_verify` package into the first interpreter that is Python 3.12 or
  newer, `python3`, else `python3.12`, and names that interpreter in `ALE_VERIFY_PYTHON`.
  Tasks neither declare nor vendor the package. `tests/test.sh` runs
  `exec "${ALE_VERIFY_PYTHON:-python3}" verify.py`; in the template's image the interpreter is the
  `/opt/venv` `python3`, first on `PATH`. An image with neither fails every verify stage
  before the grader starts, which is why the venv is 3.12;
- sets `ALE_STAGE_DIR` (`/opt/ale/tests`), `ALE_HOME` (the agent's home, `/home/user`),
  `ALE_EPISODE_ID` and, when the verify phase has a deadline, `ALE_VERIFY_DEADLINE_SECONDS`
  (the seconds left in it);
- writes its own files into `tests/`, replacing any file the task ships under the same
  name: `config.json` (the Judge settings, `ALE_VERIFY_CONFIG_PATH`; this is why the
  grader's knobs live in `grader_config.json`), `instruction.md`
  (`ALE_TASK_INSTRUCTION_PATH`) and `trajectory.json` (the agent's trajectory,
  `ALE_TRAJECTORY_PATH`, when there is one). The Judge tools log to `llm-judge.jsonl` and
  `agent-judge.jsonl` in the same folder, which the engine collects afterwards. Never ship
  your own file under any of these names.

The result is Harbor's: `/logs/verifier/reward.json` holds `{"reward": x}` with `x` a
finite number in [0, 1], and `verification.json` beside it holds the complete result, with
the same `reward`. The engine empties `/logs/verifier` before every attempt and gives it to
the framework alone. It reads only `reward`, rounded half-up to two decimals (`0.994`
becomes `0.99`, `0.995` becomes `1.0`), and keeps `verification.json` in the episode's run
folder. A nonzero exit, an exception or a timeout fails verification even after a result
was written: the episode then has no score, which is not the same as a zero.

Everything else the grader needs travels inside `tests/`. The vendored `robotics_grader`
package is the shared machinery: the solver wire and agent-UID spawn (`Solver`), hidden
seeds (`seed_rng`), the probe lock (`ProbeLock`), ratio scoring (`ratio_rewards`), the
honest zero (`zero_verdict`) and the verdict (`write_verdict`). In the engine,
`write_verdict(verdict, diagnostic=text)` hands the verdict to `ale_verify.write`, which
writes both files: `reward` at the top, the verdict's `metrics` beside it (with the
unrounded reward as `reward_exact`), and the diagnostic text when there is one.
`write_verdict(verdict, path=local_file)` writes the kit's plain envelope to that file
instead, for a run outside the engine (a host-side dry run, the kit's selftest);
`os.environ.get("ALE_VERIFY_PYTHON")` tells the two cases apart.

## The closed_loop wire

`Solver` speaks one JSON object per line to the agent's `run.sh`, launched as `bash run.sh`
with cwd `/home/user/submission` at the agent's UID. `READY` must arrive within
`READY_TIMEOUT_FACTOR` (4) step deadlines of launch, 120 s with the template's
`step_timeout_s: 30` (`tests/grader_config.json`), and every reply within one step deadline.
That deadline is a **single per-request budget**, fixed the moment the request is written
(and the `READY` wait likewise): blank lines, a partial line or a trickle of output are all
charged against the same `step_timeout` and cannot extend it — and a flood of blank lines is
capped — so a solver can never stall verification by dribbling output. Missing the deadline
aborts the episode exactly as a slow reply does.

```
{"type": "reset"}                        -> {"ok": true}
{"type": "act", "t": 0, "obs": <json>}   -> {"action": <json>}
{"type": "close"}                        -> {"ok": true}, then exit
```

`t` is the step index inside the episode, owned by the grader's loop and restarting at 0
after each `reset`. Nothing on the wire marks a probe. `ProbeLock.probe(solver, t, obs)`
asks a probed step twice with the same `t`, the real observation and a perturbed copy in a
hidden order, applies only the real reply and compares the two. The instruction tells the
agent exactly this, so an honest solver needs no special case. The lock catches a policy
that ignores its input; it is not a proof of honesty (a solver can notice a repeated `t`
and vary its answer on purpose), and isolation, not the lock, keeps the score honest. The
practice grader speaks the same wire, deadlines and repeated-`t` queries included.

After an aborted episode (a missed deadline, a crash, a malformed or missing action) the
next `reset` stops the solver's whole process group and launches `run.sh` again, which must
print `READY` again: a late reply left on the wire must not answer the next request. After
`max_restarts` (3) restarts, or a restart that fails, every remaining episode aborts at
once. Say so in the instruction, and charge an aborted episode a real cost (the template
charges what standing still would cost), never drop it: a dropped episode lets a weak
policy abort its bad starts and average only its good ones.

## How validation gates

`ale validate` runs the task twice, each time in a fresh sandbox.

1. Untouched: nobody touches the sandbox. `tests/` must write a real result with `reward`
   0.0; `untouched_nonzero` fails the run otherwise. A crash fails it too
   (`untouched_not_completed`), which is why the grader catches `SolverMissing` and writes
   `zero_verdict(...)`.
2. Oracle: `solution/solve.sh` runs in place of the agent. `tests/` must then score
   `reward` 1.0; `oracle_not_full` fails the run otherwise.

Both passes compare `reward` after the engine's two-decimal rounding and ignore every other
value in the result. There is no per-task bar to declare. Seed-to-seed variance is absorbed
in scoring:

```
ratio  = clamp(measured / anchor, 0, cap)     # inverted for lower-is-better; cap = 1.5
reward = clamp(ratio / full_at, 0, 1)         # the single gated reward key
```

`anchor` (`tests/anchor.json` `value`) is the reference implementation's own measured
value from a real `ale validate` run, never a published number, and never on an
agent-readable path. It starts null: the first run measures, reports the value in the
oracle episode's `verification.json` (`metrics`), scores 0 and fails with
`oracle_not_full`. That is the bootstrap, not a broken task.

`full_at` (same file) is where the reward saturates to 1.0, in ratio units: the worst ratio
an honest on-reference run lands on under fresh hidden seeds. Size it from a measured
spread. This template measured a 1.16x spread over 12 independent 60-episode runs (1.43x at
10 episodes, which made the gate a coin flip), so the worst honest ratio is about 0.86 and
`full_at: 0.8` clears it with margin. Too high and an honest oracle fails its own gate by
luck; too low and a mediocre solution reads as perfect.

Everything that is not the gated reward (the raw metric, the capped `ratio`, probe and
abort counts) goes in the verdict's `metrics`, which `write_verdict` records in
`verification.json` and the engine never gates on.

## What a task can be

The agent leaves any artifact in `/home/user/submission/`; `tests/` scores it however the
author decides, on hidden seeds, after the agent is gone; `solution/` produces the
reference artifact that scores full. Four shapes that fit:

- a controller or agent process queried over the stdio wire (what this template ships);
- a plan or planner that the grader executes in a simulator (success rate, path cost);
- a design artifact (URDF, mesh, parameters, generated code) that the grader loads and
  measures;
- any program or produced file with a deterministic check.

Two minimal `verify.py` skeletons for the non-wire shapes. `ratio_rewards` turns one
measured value into the verdict, `zero_verdict` is the honest zero an absent or unusable
submission earns, `write_verdict` hands the verdict to the engine. A planner program runs
through `deprivileged([...])`, never as root.

```python
# read a file and score: a design artifact, or any produced file with a deterministic check
import os
from robotics_grader import ratio_rewards, read_anchor, stage_config, write_verdict, zero_verdict
config, (anchor, meta) = stage_config(), read_anchor()          # grader_config.json, anchor.json
path = os.path.join(os.environ.get("ALE_HOME", "/home/user"), "submission", "arm.urdf")
if not os.path.isfile(path):                                     # absent or unusable: an honest 0
    write_verdict(zero_verdict(reason="no arm.urdf", anchored=anchor is not None)); raise SystemExit
measured = reachable_volume(path)                                # your check, in your simulator
write_verdict(ratio_rewards(measured, anchor, cap=config.get("scoring_cap", 1.5), metric_name="reachable_volume",
                            higher_is_better=meta.get("direction", "higher") == "higher", full_at=meta.get("full_at", 1.0)))
```

```python
# run a plan on hidden seeds and score
import json, os
from robotics_grader import ratio_rewards, read_anchor, seed_rng, stage_config, write_verdict, zero_verdict
config, rng, (anchor, meta) = stage_config(), seed_rng(), read_anchor()
path = os.path.join(os.environ.get("ALE_HOME", "/home/user"), "submission", "plan.json")
if not os.path.isfile(path):
    write_verdict(zero_verdict(reason="no plan.json", anchored=anchor is not None)); raise SystemExit
episodes = int(config.get("episodes", 20))
succeeded = sum(execute(json.load(open(path)), seed=rng.seed()) for _ in range(episodes))  # your simulator
write_verdict(ratio_rewards(succeeded / episodes, anchor, cap=config.get("scoring_cap", 1.5), higher_is_better=True,
                            metric_name="success_rate", full_at=meta.get("full_at", 1.0)))
```

## The three verify patterns

One template; the pattern is an authoring choice inside `verify.py`, not a `task.toml`
field.

| Pattern | The agent produces | verify.py does |
|---|---|---|
| closed_loop (as shipped) | a controller process behind a stdio wire | owns the simulator, steps the controller at the agent's UID by step index `t`, asks probed steps twice with the same `t` (real and perturbed, hidden order; nothing on the wire marks the probe) |
| open_loop | answers, plans or designs on disk from the agent phase | reads them and re-simulates or measures under its own dynamics; rejects answers that do not start where they claim |
| artifact_rollout | a frozen checkpoint, a planner program, generated code | loads or runs it inside `deprivileged(...)` (deserializing or executing as root hands the grader over), rolls it out on hidden seeds |

## Turning it into your task

1. Name. `task.toml`'s `[task].name` is `grounded-intelligence-initiative/<slug>`, and the
   slug must be collection-unique (`^[a-z0-9][a-z0-9_-]*$`, preferably snake_case). Do it
   first: the slug is the Task Name registered in the project's sheet, and once the task
   sits under `tasks/` the domain lint checks that the folder carries the same name. The
   org part stays `grounded-intelligence-initiative`; a folder scaffolded with
   `ale new-task` gets `agentslastexam/<folder>` and must change it. `[metadata].platform`
   is one of the benchmark's platforms (`tasks/platforms.json`, listed in the manifest's
   comment) and names the `tasks/<platform>/` folder the task lands in. A new key needs a
   maintainer.
2. Budgets and network. Size `[environment]` (`cpus`, `memory_mb`, and `gpus` when the task
   needs them), `[agent].timeout_sec` (the agent's whole budget, training included) and
   `[verifier].timeout_sec` (every scored episode plus probes). A GPU task names, in
   `gpu_types` in `[environment]` and `[verifier.environment]`, the models it has been
   tested on, cheapest first: Daytona tries them in that order, and a host without one of
   them refuses the task before it starts. It leaves the key out only when every model
   tested runs it. Test the H100 and Daytona's RTX 4090, RTX 5090, RTX PRO 6000 and H200 by
   running what uses the GPU on each: the simulator's start, a frame compared with one from
   a model already validated, and the agent's own GPU libraries. A model can lack a
   library's kernels, render differently or run too slowly for the budgets, and rendering
   also needs the graphics part of the host's NVIDIA driver, which compute-only installs
   leave out. Keep `network_mode = "no-network"` unless the task needs a host: `allowlist`
   names it in `allowed_hosts`, and `public` needs `[metadata].network_justification`. Setup
   has no field; the engine bounds it at 1800 s.
3. Image. Make `environment/Dockerfile` your environment: the simulator and every stable
   dependency, FROM the ALE base at an exact version; Python packages go in
   `environment/requirements.txt`. Keep the py3.12 `/opt/venv` with a world-traversable
   interpreter and the agent-UID import self-check at the end. A prebuilt image can instead
   be named in `[environment].docker_image` (at an exact tag), with the Dockerfile deleted.
   Large agent-visible files (photo sets, scans, meshes) are fetched at build time from the
   benchmark's Hugging Face dataset, not committed:
   [`docs/task-assets.md`](../docs/task-assets.md).
4. Environment. Replace `tests/env.py` with your simulator, imported from the image, and
   `environment/setup/payload/template_env.py` with whatever dev material the agent gets.
   While both files exist they must stay byte-identical (this repository's
   `tests/test_template_task.py`). `environment/setup/setup.sh` stages every payload file
   without naming it, so renames need no script edit.
5. Solution. `solution/solve.sh` leaves the reference artifact in `/home/user/submission/`:
   run the real reference method (train here if the method trains, inside the agent-phase
   budget), or copy in a hand-made file. Whatever scores full under your `tests/`.
6. Grader. Make `tests/verify.py` score your artifact (`tests/test.sh` stays as it is).
   The shipped `verify.py` drives a controller over the wire (adapt `run_episodes` to your
   physics);
   for a design file, read it and measure; for a plan, execute it on hidden seeds. Keep
   what must not vary: an absent or unusable submission is `zero_verdict`, never a crash;
   the abort handling and the probe lock when there is a wire; the kit's scoring. Leave
   the vendored `robotics_grader/` untouched. CI diffs it against the canonical copy and
   fails the pull request on a difference.
7. Instruction. Describe the deliverable (what to leave in `/home/user/submission/`) and the
   grading protocol. The anchor is what stays hidden, not the protocol.
8. Anchor. Set `metric` (the metric key your grader reports) and `direction` (`higher` or
   `lower`) in `tests/anchor.json`. Leave `value` null and run `ale validate`. It fails
   (`validation.json` says `oracle_not_full`); read the measured value off the oracle
   episode (`<runs>/validate-*/grounded-intelligence-initiative.<slug>-oracle-*/verification.json`,
   key `metrics`), write it in, run several more times to see the spread, set `full_at`,
   and repeat until validate prints
   `ok  ... untouched={'reward': 0.0}, oracle={'reward': 1.0}`.

The domain lint (`scripts/lint_domain.py`) enforces what the engine cannot know:
hardware-invariant metrics, a positive metric scale, `full_at` presence, agent-visible-surface
isolation, image hygiene, anchor provenance, a registered platform with its
`tasks/<platform>/` placement, and the cheap static manifest checks.
