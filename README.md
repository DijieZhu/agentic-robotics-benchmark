# Agentic Robotics Benchmark

**A benchmark of agentic robotics tasks for autonomous agents.** Each task drops an autonomous
agent into a sandboxed robotics problem (develop a controller, plan a manipulation sequence,
design a part or a mechanism, write code-as-policy, tune a planner, …) and scores **whatever
artifact it leaves** in `/home/user/submission/` — a controller process, a plan or planner, a
URDF / mesh / parameter design, generated code, a report — with a **hidden grader on hidden
seeds**. `tests/` scores the artifact however the task author decides; `solution/` is the
reference artifact that scores full. There is one kind of task; `[metadata].platform`
(one of the fixed keys in [`tasks/platforms.json`](tasks/platforms.json)) is the taxonomy axis.

The tasks run on the upstream **ALE engine**
([AgentsLastExam/ale](https://github.com/AgentsLastExam/ale)), pinned as the submodule at
`vendor/ale`; nothing here reimplements it. Upstream's
[task-authoring.md](https://github.com/AgentsLastExam/ale/blob/main/docs/specs/task-authoring.md)
and
[task-quality-standard.md](https://github.com/AgentsLastExam/ale/blob/main/docs/specs/task-quality-standard.md)
are the authority on the task format and the quality bar; this repository documents only what
the robotics domain adds. The website, <https://agentic-robotics-benchmark.org>, displays
the tasks; it takes no uploads and no submissions.

This repository is public and complete: every task folder is here whole, `solution/` and
`tests/` included. The anchor is hidden from the *tested agent* by the engine's stage
isolation while it works, not from people.

*[中文说明](README.zh-CN.md)*

## One task = one folder

A task is a self-contained folder in the engine's Harbor-based format that the engine runs in
three stages (`setup` → `agent` → `verify`); `solution/` stands in for the agent under
`ale validate`. Its `task.toml` (schema `1.4+ale.1`) names it
`grounded-intelligence-initiative/<slug>`, the slug being the folder name. What the robotics
domain adds (the three verify patterns, scoring, the vendored kit, the probe wire) is in
[`templates/README.md`](templates/README.md), next to the file-by-file tour of the template.

```
tasks/<platform>/<slug>/
├── task.toml         the configuration (strict): name, platform, resources, timeouts, network,
│                     artifacts; invisible to the agent
├── instruction.md    the prompt — the only thing the tested agent is told
├── environment/      the image: Dockerfile, final stage FROM an exact version of an ALE base
│   └── setup/        setup.sh, trusted root, before the agent: stages payload/ into the agent home
├── solution/solve.sh produces the reference artifact; `ale validate` runs it in place of the agent
└── tests/            test.sh -> the grader (verify.py), anchor.json, grader_config.json, and the
                      vendored robotics_grader/ package; ABSENT while the agent works
```

What keeps the grader and the anchor away from the agent is **timing, not a flag**: a
stage's folder reaches the sandbox only when that stage runs (`tests/` at `/opt/ale/tests`
for verification, `solution/` at `~/.ale-solution` in an oracle episode only), `/opt/ale` is
root-only, and the agent's solver runs at the agent's UID. Nothing the agent prints is ever a
scoring input; the grader computes the metric from its own simulator state.

## Scoring

```
ratio  = clamp(measured / anchor, 0, cap)     # inverted for lower-is-better metrics; cap = 1.5
reward = clamp(ratio / full_at, 0, 1)         # the task's score, the one value the engine reads
```

The **anchor** is the reference implementation's own measured value from a real run —
never a printed figure from elsewhere — and `full_at` is the ratio at which the reward
saturates, sized from the metric's measured seed-to-seed spread. `reward` (0 to 1) is the
task's score. The grader hands its verdict to the kit's `write_verdict`, which writes it
through the engine's `ale_verify.write`: `/logs/verifier/reward.json` holds `reward`, rounded
half-up to two decimals, and `verification.json` beside it the complete result (the raw
metric, the capped `ratio`, the unrounded `reward_exact`, the grader's diagnostic). Metrics
are always outcome metrics on a positive scale, never wall-clock time or throughput.

`ale validate` runs every task twice, in fresh sandboxes. An untouched sandbox must score
`reward` 0.0, and the oracle must then score 1.0, both after the two-decimal rounding; no
other value is gated. Either miss fails validation (`untouched_nonzero`, `oracle_not_full`;
an episode that does not complete fails as `untouched_not_completed` or
`oracle_not_completed`; exit 2). A null anchor fails the same way by design;
`harness/ale_onboard.py` reads the run records and reports it as `needs_anchor` with the
measured value.

## Contributing a task

A contribution is a pull request into this repository that adds one task folder at
`tasks/<platform>/<name>/`, where `<name>` is the task's registered name (`task.toml` says
`name = "grounded-intelligence-initiative/<name>"`) and `<platform>` its
`[metadata].platform`. The project's contribution guidelines cover
sign-up, roles, the task stages and credit:
[English](https://docs.google.com/document/d/1s1E3ymIBLujWf2l7rhu4g9Nl4orBfooK0aDWOdum7-Y/view) and [中文](https://docs.google.com/document/d/1gX8v-w-idIPLjzoWgRskT0llE7cKXqg9sVLB0cWfZk0/view).

Build the task from `templates/task/`, gate it with `ale lint` and `ale validate`, measure its
difficulty with frontier models (`ale run`; the target is a score roughly below 0.5), then
open the pull request. CI runs `ale lint`, the domain lint, the kit's selftest and byte
comparison, the asset check, the registry identity cross-check and the test suite on it
(the list is in [`CONTRIBUTING.md`](CONTRIBUTING.md)); a maintainer re-runs `ale validate`,
merges, and exports the registry. The short
version is [`CONTRIBUTING.md`](CONTRIBUTING.md); the whole path, from the template to a merged
task, is [`docs/contributing-a-task.md`](docs/contributing-a-task.md).

## Layout

```
vendor/ale/                     upstream engine (submodule, pinned commit) — do not reimplement it
tasks/<platform>/<slug>/        the tasks, one folder each
tasks/platforms.json            the platform vocabulary: the `[metadata].platform` keys, with labels
tasks/README.md                 a note: which task is the worked example
templates/task/                 ONE template, gates green as shipped (null anchor by design)
templates/README.md             the task contract, file by file (source of truth for the task form)
identity-history.json           past (spec_hash, content_digest) pairs per task (registry v2)
shared/robotics_grader/         canonical verify-stage kit; each task vendors a byte-identical copy
                                at tests/robotics_grader/
shared/task_assets/             canonical asset fetcher; a task with large files vendors it as
                                environment/fetch_assets.py
task-assets.toml                the Hugging Face dataset that holds the tasks' large files
harness/
  ale_onboard.py                `ale-onboard`: instantiate the template + run the real gates
scripts/
  export_registry.py            registry v2 from the engine's own loader (task identity)
  lint_domain.py                domain lint (what `ale lint` cannot know), placement included
  task_assets.py                every task's asset list against task-assets.toml (check / sync)
  trajectories.py               model runs' trajectories: inventory, curate, export, scan, upload, sample
schemas/task-registry.schema.json   the registry's contract (owned here; the export is tested against it)
skills/onboard-*/SKILL.md       agent-facing onboarding skills (router + one per verify pattern)
skills/agent-trajectory-archiving/SKILL.md   recording model runs' trajectories (scripts/trajectories.py)
tests/                          pytest suite
docs/                           architecture, security model, contributing,
                                evidence/ (real verdicts, isolation audit), dev/ (design records)
.github/workflows/pr-checks.yml the checks every pull request gets
```

## Commands

```bash
uv sync                                                    # host-side tooling (py3.11)

# the pinned engine's own venv, once per checkout (what upstream's `just bootstrap` runs)
cd vendor/ale && VIRTUAL_ENV= uv sync --frozen --all-groups
# once per machine while upstream has not published ale-qemu-runner:0.1.1, which the
# docker provider runs to install each sandbox's network rules
cd vendor/ale && DOCKER_HOST=unix:///var/run/docker.sock images/build.sh ale-qemu-runner

# the engine's own gates, from the pinned submodule (clear VIRTUAL_ENV, pin the docker socket)
cd vendor/ale && VIRTUAL_ENV= uv run ale lint ../../tasks && VIRTUAL_ENV= uv run ale lint ../../templates
cd vendor/ale && VIRTUAL_ENV= DOCKER_HOST=unix:///var/run/docker.sock \
    uv run ale validate ../../tasks/<platform>/<slug> --runs-dir /tmp/ale-runs

# authoring gate: lint -> validate (untouched 0.0 / oracle 1.0) -> honest outcome
uv run ale-onboard <spec>.yaml --json                                      # instantiate + gate
uv run ale-onboard --gate-only tasks/<platform>/<slug> --json
#   built | needs_anchor (measured value reported) | needs_input | gave_up | error

# domain lint (platform and placement, metric discipline, anchor provenance, visible surface, image hygiene)
# + the task asset lists + the shared kit's self-test (no docker)
uv run python scripts/lint_domain.py .
python3 scripts/task_assets.py check
cd shared && python3 -m robotics_grader.selftest

# run a task with a real agent (the difficulty test; the results are in <runs>)
cd vendor/ale && VIRTUAL_ENV= DOCKER_HOST=unix:///var/run/docker.sock \
    uv run ale run ../../tasks/<platform>/<slug> --agent claude-code --model <model> --runs-dir <runs>
# then keep one trajectory per task and agent: skills/agent-trajectory-archiving/SKILL.md
python3 scripts/trajectories.py inventory --runs-dir <runs> --out inventory.jsonl

# the task registry (task identity from the engine loader; refuses a dirty tree; contract in
# schemas/task-registry.schema.json). platforms come from tasks/platforms.json; the website
# installs this export with its own sync script and only displays it
cd vendor/ale && VIRTUAL_ENV= uv run python ../../scripts/export_registry.py \
    --tasks ../../tasks --history ../../identity-history.json --out registry.json

uv run pytest -q
```

The worked example validates in well under a minute once its image is built.

## Access

Running a task needs the **ALE engine** and Docker. The engine repository
([AgentsLastExam/ale](https://github.com/AgentsLastExam/ale)) is public; it is not on PyPI
and has no license file yet, so reference it, never vendor its code into this
repository. Use the pinned submodule: `cd vendor/ale && uv sync --frozen --all-groups` (what
upstream's `just bootstrap` runs; needs `uv`), then `uv run ale --help`. Task images build
`FROM ghcr.io/agentslastexam/container-ubuntu22-base:0.1.1`, which is published and needs no
login. The engine's docker provider also runs `ghcr.io/agentslastexam/ale-qemu-runner:0.1.1`,
which upstream has not published yet: `images/build.sh ale-qemu-runner` in the engine checkout
builds it once per machine (the script pulls a published version and builds one that is not).
A contributor runs `ale lint` and `ale validate` themselves before opening a pull request; a
maintainer re-runs them before merging.

## Docs

- [`docs/architecture.md`](docs/architecture.md) — engine stages, stage isolation, the UID
  boundary, the verdict, and the registry export.
- [`docs/security-model.md`](docs/security-model.md) — what stage isolation hides from the
  tested agent, and what a verdict can and cannot prove.
- [`docs/contributing-a-task.md`](docs/contributing-a-task.md) — the contributor's path from
  the template to a merged task.
- [`docs/dev/`](docs/dev/) — dated design records; their 2026-10-15 notes say what no longer
  holds (the website upload channel and the leaderboard bridge are retired), and the task
  form they describe predates the 2026-11-03 move to `task.toml`.

## History

This repository was created on 2026-09-07 with a fresh history when the project was renamed
from "ALE Robotics" to the Agentic Robotics Benchmark. It replaces two archived repositories:
`ale-robotics-benchmark-private` (the maintainers' task repository; the pre-ALE task form and
its harness are at its tag `legacy-harness-final`) and `ale-robotics-benchmark` (the public
mirror, tag `legacy-v0.5-final`). Nothing here revives either.

On 2026-11-03 the tasks moved to the engine's Harbor-based format (`vendor/ale` at
`1cdbc02`): `task.yaml` became `task.toml`, `image/` became `environment/`, `setup/` became
`environment/setup/`, `verify/` became `tests/` and `oracle/` became `solution/`. Records from
before that day (`docs/evidence/`, `docs/dev/`) use the old names.

## Conventions

Everything written into this repository — code, comments, docs, task files — is **English**;
it is read by tested agents and benchmark users worldwide. Where a document and the code
disagree, the code wins: say so and fix the document.

## License

Apache-2.0 (see [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE)). The upstream engine under
`vendor/ale` is a separate project with its own terms.
