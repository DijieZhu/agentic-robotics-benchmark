# Contributing a task

The path: get the engine, start from the template, build the task locally, validate it,
measure its difficulty with a frontier model, then open a pull request that adds your folder
at `tasks/<platform>/<name>/`. Sign-up, roles, the task stages (expert validation, formal
benchmarking) and how contributions are credited are in the project's contribution guidelines
([English](https://docs.google.com/document/d/1s1E3ymIBLujWf2l7rhu4g9Nl4orBfooK0aDWOdum7-Y/view),
[中文](https://docs.google.com/document/d/1gX8v-w-idIPLjzoWgRskT0llE7cKXqg9sVLB0cWfZk0/view));
the short version of this page is the repository's [`CONTRIBUTING.md`](../CONTRIBUTING.md).

What a task is and the bar it has to clear are the engine's own specs. Read them once:

- Task authoring: https://github.com/AgentsLastExam/ale/blob/main/docs/specs/task-authoring.md
- Task quality standard: https://github.com/AgentsLastExam/ale/blob/main/docs/specs/task-quality-standard.md

What the robotics domain adds, the file-by-file tour of the template and the "Turning it
into your task" checklist are in [`templates/README.md`](../templates/README.md). Where this
page disagrees with it, it wins.

A robotics task here is an agentic problem an autonomous agent solves inside a sealed
sandbox. It is handed an `instruction.md` and an environment image, works however it likes
(train a controller, write a planner, design a part, generate code, tune parameters), and
leaves any artifact in `/home/user/submission/`. A hidden grader scores that artifact on
fresh hidden seeds. Your job is to author the whole package so that it validates: an
untouched sandbox scores reward 0.0 and the reference artifact 1.0, after the engine rounds
`reward` half-up to two decimals.

## 1. Prerequisites

You need Docker and the ALE engine at the commit this repository pins, the `vendor/ale`
submodule; upstream `main` moves on, and a task is gated on the pinned commit. `uv` and
`just` (`uv tool install rust-just`) drive it. Clone this repository with its submodule (or
run `git submodule update --init vendor/ale` in an existing clone) and run every `ale` command
from `vendor/ale`:

```bash
git clone --recurse-submodules https://github.com/Grounded-Intelligence-Initiative/agentic-robotics-benchmark.git
cd agentic-robotics-benchmark/vendor/ale
just bootstrap
uv run ale --help
images/build.sh container-ubuntu22-base ale-qemu-runner   # once; the base takes a while
```

`images/build.sh` pulls each image that ghcr.io already has at the engine's version (`0.1.1`
at the pin) and builds it from the checkout otherwise. Tasks build on
`container-ubuntu22-base:0.1.1`; the engine's Docker provider runs `ale-qemu-runner` to set up
each sandbox's network rules. If your reference method trains, plan for a single GPU of 24 GB
or less; training happens only in the oracle phase, never during scoring.

## 2. Get the template

Copy [`templates/task/`](../templates/task/) from your checkout to wherever you work. Its
name in the project's sheet, the folder name and the part of `[task] name` in `task.toml`
after `grounded-intelligence-initiative/` are one string that stays the task's identity for
good: preferably a short `snake_case` name (lowercase letters, digits, underscores; the
engine accepts `^[a-z0-9][a-z0-9_-]*$`).

Rename the folder to that name first and set `name = "grounded-intelligence-initiative/<name>"`
under `[task]` in `task.toml`. No file needs an executable bit to run: the engine uploads
every stage file with mode 0644 (directories 0755) and runs `setup.sh`, `test.sh` and
`solve.sh` through `bash`, so a copy that lost its Unix permissions (Windows, some GUI tools)
still runs. A program a stage starts itself must be run through its interpreter, or made
executable by the stage.

The engine admits only `task.toml`, `instruction.md`, `environment/`, `tests/`, `solution/`
and `.ale-cache` at a task's top level; any other entry, `tools/` included, fails lint. Every
file in the folder is part of its identity, except the engine's asset roots
(`environment/assets/`, `tests/assets/`, `solution/assets/`) and `.ale-cache`, so keep notes
and scratch files outside it.

The template validates as shipped, except that its anchor is null (step 6). Start from that
state and change one thing at a time.

## 3. The build loop

Work through "Turning it into your task" in [`templates/README.md`](../templates/README.md).
The short form:

1. Name: the folder and the slug in `task.toml`'s `[task] name` are the same
   (`^[a-z0-9][a-z0-9_-]*$`).
2. Image: `environment/Dockerfile` is your environment, final stage `FROM` an exact version of
   an official ALE base (`ghcr.io/agentslastexam/container-ubuntu22-base:0.1.1`; lint refuses
   `latest` and an untagged base); Python packages go in `environment/requirements.txt`. Keep
   the py3.12 `/opt/venv` and the agent-UID import self-check. A prebuilt stack can name
   `[environment].docker_image` instead.
3. Environment: `tests/env.py` is the authoritative copy the grader imports;
   `environment/setup/payload/template_env.py` is the dev copy the agent gets.
4. Oracle: `solution/solve.sh` leaves the reference artifact in `/home/user/submission/`, by
   running the real method or by copying in a hand-made file.
5. Grader: `tests/verify.py` scores your artifact. Do not touch `tests/robotics_grader/`;
   CI compares it byte for byte with the canonical `shared/robotics_grader/` and fails the
   pull request on a difference. An absent or unusable submission is `zero_verdict`, never a
   crash.
6. Instruction: the deliverable and the grading protocol. For a controller on the wire, keep
   the paragraph about the step index `t`: a step may be queried more than once with the same
   `t`, answer each request from the observation it carries, and repeated queries do not
   advance the episode. Nothing on the wire marks a probe.
7. Anchor: section 6 below.

After each change, from the engine checkout (`<task_dir>` is your folder, `<runs>` a scratch
directory):

```bash
uv run ale lint <task_dir>
uv run ale validate <task_dir> --runs-dir <runs>
```

`ale validate` runs the task twice. Untouched: nobody touches the sandbox, and `tests/` must
write a real result whose `reward` is 0.0 (`untouched_nonzero` otherwise). A crash fails it
too (`untouched_not_completed`), which is why the grader treats an absent artifact as
`zero_verdict(...)`. Oracle: `solution/solve.sh` runs in place of the agent, and `reward`
must be 1.0 (`oracle_not_full` otherwise). The engine rounds `reward` half up to two
decimals before it compares: 0.995 counts as 1.0 and 0.004 as 0.0.

Nothing you print to stdout is a scoring input. The grader computes the metric under its own
physics and writes the result to `/logs/verifier`, which the engine empties and gives to root
alone before scoring, so your solution can neither pre-create nor overwrite it.

**Verify in a separate sandbox.** Set `[verifier] environment_mode = "separate"` (with a
`[verifier.environment]` giving the verifier its `cpus`/`memory_mb`), as the template and
every task here do. Shared verification runs the grader as root in the agent's own sandbox,
where the base image leaves `/usr/local/bin` agent-writable and ahead of `/usr/bin` on root's
PATH: a program the agent dropped there would run as root before the grader and score itself
1.0. A separate verifier is fresh, with only your declared `artifacts` restored — so stage
anything verify needs in `tests/` or an artifact, because `setup.sh` does not run there. If
your grader drives a solver over the wire (closed_loop), each request has a single deadline
(`step_timeout_s`) fixed when it is sent: blank or partial output is charged against it and
cannot extend it, so a stalling submission aborts its episode instead of running out the
verify budget.

Files too large for git (photo sets, document scans, meshes the agent receives) go to the
benchmark's Hugging Face dataset and are fetched while the image builds, pinned by commit and
sha256: [`task-assets.md`](task-assets.md). Hidden material never goes there.

From the repository root, `uv run python scripts/lint_domain.py <task_dir>` runs the domain
lint the pull request CI runs (section 5, plus placement once the folder sits under
`tasks/`).

## 4. What the agent sees, and what it must not

| Agent sees (during its phase) | Hidden (staged only when its stage runs) |
|---|---|
| `instruction.md`, the built image, `environment/setup/payload/` (dev env, practice grader) | `tests/` (grader, `env.py`, `anchor.json`), `solution/`, and `task.toml` itself |

There is no "hide this" flag. A stage's folder is absent from the sandbox until that stage
runs, and `/opt/ale` is root-only while the solver runs unprivileged. The one thing you
police yourself: the practice grader in `environment/setup/payload/` prints the raw metric
only, never a score, a ratio or the anchor, and it never imports `tests/`.

The repository is public, so a person can read every anchor. That is fine: the benchmark
measures what the agent does inside a no-network sandbox that does not contain `tests/`, and
the agent is what is being measured.

## 5. Metric discipline

Two mistakes silently corrupt a task. Neither `ale lint` nor `ale validate` checks them; the
domain lint does, on every pull request. The metric is declared in `tests/anchor.json` and
nowhere else: `metric` is the key your grader reports, `direction` is `higher` or `lower`.

- The metric is hardware-invariant. Never score wall-clock time, throughput or FPS; they
  depend on the machine. Use outcome metrics (success rate, collision rate, SPL, ATE, episode
  reward). If the natural headline of your problem is a time, convert it to a pass/fail
  real-time threshold.
- The metric is on a positive scale. A negative anchor inverts the ratio, so a better result
  scores lower. Report a negative reward as a positive cost with the anchor's `direction` set
  to `lower`.

## 6. The anchor bootstrap

The scoring denominator is the reference implementation's own measured value, never a figure
printed elsewhere. You cannot know it until you run, so:

1. Leave `tests/anchor.json`'s `value` as `null` and run `ale validate`. It prints `FAIL`
   and the run's `validation.json` names the reason, `oracle_not_full`; that is expected.
   The oracle episode's `verification.json` carries the measured value (`result.json` holds
   only the rounded `reward`):

   ```bash
   python3 -c 'import glob,json; [print(p, json.load(open(p))["metrics"]) for p in
     sorted(glob.glob("<runs>/validate-*/grounded-intelligence-initiative.<slug>-oracle-*/verification.json"))]'
   ```

   The measured value is the key `tests/anchor.json` names as `metric`; `ratio`,
   `match_lock_ok`, `anchor_recorded`, `probes_*`, `episodes_aborted` and `reward_exact` are
   bookkeeping. The grader's own stderr is not printed by `ale validate`; it is in the same
   folder's `trace.execution.jsonl`, in the verify-phase `command_finished` event.
2. Write the value into `anchor.json` as `value`, and say where it came from in `source`.
3. Run the oracle several more times, look at the spread of `ratio`, and set `full_at` (the
   ratio at which the reward saturates to 1.0) to the worst ratio an honest on-reference run
   lands on. Too high and an honest oracle fails its own gate by luck; too low and a mediocre
   solution reads as perfect. The template's own spread and method are in
   `templates/README.md`.
4. Repeat until `ale validate` prints `ok  ... untouched={'reward': 0.0}, oracle={'reward': 1.0}`.

## 7. Measure the difficulty

`ale validate` proves the task runs and scores honestly; it says nothing about difficulty.
Run frontier models on it with the engine's `ale run` (for example
`uv run ale run <task_dir> --agent claude-code --model <model> --runs-dir <runs>`) and read
the verdicts. The task's score is the verdict's `reward` (0 to 1; the capped `ratio` rides
along in `verification.json`'s `metrics`), and current frontier models should stay roughly
below 0.5 under the agreed configuration and run budget. When a model saturates the task,
raise the engineering or perception challenge and test again; do not suppress scores with
arbitrary penalties, and keep the task completable by a human expert under the stated
conditions. Before reading a failed run as a model limitation, fix what is not the task:
environment failures, missing dependencies, unclear instructions. Keep the run instructions
and the results for the pull request, and record each run's trajectories as
`skills/agent-trajectory-archiving/SKILL.md` describes (`scripts/trajectories.py`: one trajectory
per task and agent, with the agent's whole session, its score and its cost).

Three limits of the `claude-code` harness and the engine lowered scores without any error
at the previous pin (`90a7c1c`). Re-checked in the code of the current pin (`1cdbc02`,
2026-11-03; paths relative to `vendor/ale/packages/ale-run/src/ale/run/`), the stream cut-off
moved from 300 s to 600 s, the `PATH` limit is gone, and the 30-minute request bound and the
CLI's response cap hold. Check for each before trusting a run:

- **The stream cut-off is 600 s now, not 300 s.** The harness sets
  `BUN_CONFIG_HTTP_IDLE_TIMEOUT=600` and `CLAUDE_STREAM_IDLE_TIMEOUT_MS=600000` itself
  (`harnesses/claude_code/adapter.py:30`, `:173-182`), and the gateway writes an SSE keep-alive
  comment to the CLI after every 15 s of upstream silence (`gateway/server.py:51-63`,
  `:444-453`). A stream on which the CLI sees no byte for 600 s still ends the episode
  (`API Error: Stream idle timeout`). The harness's values win over an image `ENV` or an
  `[environment].env` entry of the same names (`guestd/main.py:112-113`,
  `protocol/episode_protocol.py:1750`), so the earlier workaround, those two variables in an
  evaluation copy of the Dockerfile, no longer changes anything.
- **The agent's `PATH` keeps the image's.** The harness puts its own `~/.local/bin` in front
  of the `PATH` the agent account inherits from the image (`harnesses/_npm.py:70-76`, used at
  `harnesses/claude_code/adapter.py:126`), so an image with `ENV PATH=/opt/venv/bin:$PATH` (the
  template's) gives the agent the venv's `python3`. At the previous pin the harness replaced
  `PATH` with its own directories plus `/usr/local/bin:/usr/bin:/bin`. A `ModuleNotFoundError`
  in the agent's transcript is still the first thing to look for.
- **One model request still ends after 30 minutes, and the response cap is the CLI's.** The
  gateway ends any single model request after 1800 s in total (`gateway/server.py:49`). The
  engine sets no output-token cap: the harness does not set `CLAUDE_CODE_MAX_OUTPUT_TOKENS`
  (`harnesses/claude_code/adapter.py:150-200`) and the gateway forwards the CLI's own
  `max_tokens` unchanged (`gateway/server.py:649-670`). The pinned CLI is still Claude Code
  2.1.227 (`harnesses/claude_code/adapter.py:29`); at the previous pin it capped one response
  at 64k output tokens. A model that thinks for longer than that before acting gets
  back a response with no text and no tool call, and can loop on it until the agent budget
  runs out with nothing in `/home/user/submission/`. Look for `stop_reason: max_tokens` and
  calls with a `latency_ms` near 1800000 in the episode's `trace.transport.jsonl`. An
  `ENV CLAUDE_CODE_MAX_OUTPUT_TOKENS=...` in an evaluation copy of the Dockerfile still
  reaches the CLI, and helps only up to what fits in 30 minutes. On the spacecraft task
  (2026-10-15, previous pin), Fable 5.1 and Fable 5 never acted at 64k, 100k or 128k; Opus 5.5
  and Opus 5 did. Treat such a zero as an evaluation failure, not as difficulty.

## 8. Submit

Open a pull request into this repository that adds exactly one folder,
`tasks/<platform>/<name>/`, where `<platform>` is `[metadata].platform` from your `task.toml`
(one of the keys in `tasks/platforms.json`) and `<name>` the slug in its `[task] name`. Push
the branch to this repository (the project initiator grants write access after you sign up
and send your GitHub ID) or to a fork. The pull request
template asks for the `ok` line of your `ale validate` run, your frontier-model results and a
few confirmations. Nothing else belongs in the pull request: no notes, no `__pycache__/`, no
`.ale-cache/`, no `assets/`. To revise the task under review, push to the same branch.

## 9. What happens on the pull request

CI (`.github/workflows/pr-checks.yml`) runs `ale lint` over `tasks/` and `templates/`, the
domain lint (metric, anchor, visible surface, image hygiene, placement), the kit comparison,
the registry identity cross-check (a new slug is a notice) and the test suite. It needs no
secrets, so it runs on forks too. It does not build your image or run `ale validate`.

A maintainer then reads `tests/` and `solution/`, re-runs `ale lint` and `ale validate` on
your branch, corrects the anchor on the branch if the measurement disagrees, and merges. The
task is live at its path from that moment and enters the registry at the maintainers' next
export; the website displays the registry once that export is installed there. Expert
validation works from a task version (the pull request's branch and commit); the guidelines
describe that stage and the formal benchmarking runs.

## Common pitfalls

- A venv the agent account cannot traverse fails only at solve time, as a misleading
  `ModuleNotFoundError`. The agent UID must be able to traverse (not merely read) everything
  the venv points at; keep the Dockerfile's agent-UID import self-check.
- `task.toml` is strict: an unknown key fails lint, and it has no parameters. Grader knobs go
  in `tests/grader_config.json`, never in `tests/config.json`, which the engine overwrites
  with its own verification settings at verify time.
- `ale validate` does not print a stage's stderr. It is in `trace.execution.jsonl`.
- Do not edit run outputs to make a task pass. Author the inputs; the verdict is computed.
- The folder name, the slug in `task.toml`'s `[task] name` and the `tasks/<platform>/` parent
  (`[metadata].platform`) must agree; the domain lint refuses a pull request where they do
  not.
