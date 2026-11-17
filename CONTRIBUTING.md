# Contributing a task

A task is one self-contained folder. Contributing one means getting a pull request merged
that adds it at `tasks/<platform>/<name>/`. The project's contribution guidelines, with
sign-up, the three roles (development, validation, benchmarking), the task stages and how
contributions are credited, are two Google Docs with the same content:
[English](https://docs.google.com/document/d/1s1E3ymIBLujWf2l7rhu4g9Nl4orBfooK0aDWOdum7-Y/view)
and [中文](https://docs.google.com/document/d/1gX8v-w-idIPLjzoWgRskT0llE7cKXqg9sVLB0cWfZk0/view).
This page covers what happens in this repository.

## Before you start

1. Sign up in the project's coordination sheet (linked from the guidelines) and tell the
   project initiator your GitHub ID; they give you write access to this repository (the sheet
   alone does not).
2. Register the task in the sheet. Its Task Name is the task's permanent `name`, identical in
   the sheet, in the folder name and after the organization in `task.toml`'s `[task]` table
   (`name = "grounded-intelligence-initiative/<name>"`): preferably a short `snake_case`
   string (lowercase letters, digits, underscores; the engine accepts `^[a-z0-9][a-z0-9_-]*$`).
   A retired task keeps its name; never reuse one.
3. Pick its platform, one of the keys in [`tasks/platforms.json`](tasks/platforms.json):
   `arm`, `hand`, `quadruped`, `humanoid`, `uav`, `driving`, `ground`, `underwater`, `space`,
   `soft`, `surgical`, `swarm`, `subsystems`. The most specific one wins: a legged robot is
   `quadruped` or `humanoid`; a car-like vehicle whose task is the driving itself, on a road or
   a track and at any scale, is `driving`; any other wheeled or tracked ground robot (an AMR or
   AGV, a UGV, a rover, a wheeled mobile manipulator) is `ground`. A task that fits none needs
   a maintainer first. The other platforms a task involves go in its card (step 9).

You need Docker and the ALE engine at the commit this repository pins: the `vendor/ale`
submodule (upstream `main` moves on; a task is gated on the pinned commit). Clone with the
submodule (or run `git submodule update --init vendor/ale` in an existing clone):

```bash
git clone --recurse-submodules https://github.com/Grounded-Intelligence-Initiative/agentic-robotics-benchmark.git
cd agentic-robotics-benchmark/vendor/ale  # the pinned engine; run every `ale` command from here
uv sync --frozen --all-groups           # the engine's venv (upstream's `just bootstrap`; needs uv)
uv run ale --help
images/build.sh ale-qemu-runner         # once per machine (upstream has not published 0.1.1 yet)
```

Task images build `FROM ghcr.io/agentslastexam/container-ubuntu22-base:0.1.1`, which is
published; Docker pulls it on the first build. The engine's docker provider also runs
`ghcr.io/agentslastexam/ale-qemu-runner` at the engine's version to install each sandbox's
network rules. Without that image every episode ends in `env_error` ("could not route the
sandbox's access to the host"); `images/build.sh` pulls a published version and builds one
that is not.

## Build and test it

Copy `templates/task/` from a checkout of this repository, rename the folder to the task's
name and set `name` in `task.toml`'s `[task]` table to `grounded-intelligence-initiative/`
followed by the same string. What each file is for, and what the robotics domain requires
beyond the engine's own specs, is in [`templates/README.md`](templates/README.md). The long
form of the whole path, including the anchor bootstrap, is
[`docs/contributing-a-task.md`](docs/contributing-a-task.md).

Keep the template's `[verifier] environment_mode = "separate"`: the grader then scores in a
fresh sandbox the agent never touched, with only your declared `artifacts` restored, instead
of as root beside the agent's files (where a program planted on root's PATH would run as root
before the grader). Stage anything verify needs in `tests/` or an artifact — `setup.sh` does
not run in a separate verifier.

Gate it from `vendor/ale` until both lines are green:

```bash
uv run ale lint <task_dir>
uv run ale validate <task_dir> --runs-dir <runs>     # reward: untouched 0.0, oracle 1.0, rounded
```

Validation shows that the task runs; it says nothing about difficulty. Measure that with a
frontier model (`uv run ale run <task_dir> --agent <harness> --model <model>`) and iterate:
the task's score is the episode's `reward` (0 to 1; the episode's `verification.json` keeps
it with the run's `ratio` and the raw metric), and current frontier models should stay
roughly below 0.5 under the agreed configuration and budget. Raise the engineering or
perception challenge rather than adding arbitrary penalties, and fix environment failures,
missing dependencies and unclear instructions before reading a failed run as a model
limitation. The pinned `claude-code` harness and engine have limits of their own that lower
scores without an error (a model stream silent for 600 s is cut off; the CLI caps a response
at 64k tokens and the gateway a request at 30 minutes, so a model that thinks longer never
acts); see [`docs/contributing-a-task.md`](docs/contributing-a-task.md) §7. Keep the run
instructions and the results of both.

## Open the pull request

1. Push a branch (to this repository, or to a fork) that adds exactly one folder,
   `tasks/<platform>/<name>/`, where `<platform>` is `[metadata].platform` in your
   `task.toml` and `<name>` the part of `[task].name` after the organization. Nothing else
   goes in the pull request.
2. Open the pull request. Its template asks for the `ok` line of your `ale validate` run,
   your frontier-model results and a few confirmations.
3. Once CI passes, hand the task version (the branch and its commit) to the task's validator,
   as the guidelines describe.

## What CI checks on every pull request

`.github/workflows/pr-checks.yml`, no secrets needed, so it runs on forks too:

- the pinned engine's `ale lint` over `tasks/` and `templates/`;
- the domain lint (`scripts/lint_domain.py`): a hardware-invariant metric on a positive scale
  (`metric` and `direction` in `tests/anchor.json`), an anchor with a positive `value`, a
  `full_at` inside the scoring cap and a real `source`, nothing under
  `environment/setup/payload/` that reaches the grader, image hygiene, the organization
  `grounded-intelligence-initiative` in `[task].name`, a `[metadata].platform` from
  `tasks/platforms.json`, and the placement of the folder at `tasks/<platform>/<name>/`;
- the kit's selftest (`cd shared && python3 -m robotics_grader.selftest`, on the system
  python: the kit is stdlib-only), and `tests/robotics_grader/` byte-identical to
  `shared/robotics_grader/` (copy the canonical kit over yours and re-validate if this fails);
- for a task with large files in the Hugging Face dataset, its `environment/assets.sha256`
  naming the dataset in [`task-assets.toml`](task-assets.toml) and its
  `environment/fetch_assets.py` identical to `shared/task_assets/fetch_assets.py`
  (`python3 scripts/task_assets.py check`; [`docs/task-assets.md`](docs/task-assets.md));
- the registry identity cross-check for tasks the registry already knows (a new slug is a
  notice, not a failure);
- the pytest suite.

CI does not build images or run `ale validate`; that stays with you before the pull request
and with a maintainer before the merge.

## What the maintainers do

1. Read `tests/` and `solution/`. This is contributor code, and `tests/` computes every
   future score.
2. Check out the branch and re-run the gate from `vendor/ale`:

   ```bash
   VIRTUAL_ENV= uv run ale lint ../../tasks/<platform>/<name>
   VIRTUAL_ENV= DOCKER_HOST=unix:///var/run/docker.sock \
       uv run ale validate ../../tasks/<platform>/<name> --runs-dir <runs>
   cd ../.. && uv run python scripts/lint_domain.py tasks/<platform>/<name>
   ```

   If the measured anchor differs from `tests/anchor.json`, push a correction to the branch
   rather than asking the contributor to redo it.
3. Merge. The task is live at its path; nothing relocates or rewrites it.
4. Export the registry from a clean checkout of `main` (`scripts/export_registry.py`) and
   commit the updated `identity-history.json`. The website installs that export and only
   displays it.

## Contributing to the tooling

`harness/`, `scripts/`, `shared/`, `skills/` and `tests/` are ordinary Python; `uv sync` and
`uv run pytest -q`. Edit `shared/robotics_grader/` and copy it into every task's
`tests/robotics_grader/`, never the other way round. Everything written into this repository
is English. Where a document and the code disagree, the code wins: fix the document.
