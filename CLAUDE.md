# CLAUDE.md · working conventions for agentic-robotics-benchmark

> **The Agentic Robotics Benchmark is an agentic-robotics benchmark: autonomous agents solve
> robotics problems inside a sealed sandbox.** Each task drops an agent into a robotics
> problem (develop a controller, plan a manipulation sequence, design a part, write
> code-as-policy, tune a planner, …) and scores **whatever artifact it leaves** in
> `/home/user/submission/` with a hidden grader on hidden seeds. The contract is
> artifact-general: `tests/` scores the artifact however the author decides, `solution/` is the
> reference artifact that scores full. Never frame the benchmark as "running a policy"; that
> is one shape among many. There is **one kind of task**; `[metadata].platform` (one of the
> fixed keys in `tasks/platforms.json`) is the taxonomy axis. The tasks run on the upstream
> `AgentsLastExam/ale` engine (its Harbor-based form since 2026-11-03), pinned as a submodule
> at `vendor/ale`. **Never reimplement anything the engine does.** This repository is public.
> The user-visible name is always the full "Agentic Robotics Benchmark"; "ALE
> Robotics" is the name it had before 2026-09-07 and appears only where history is explained.

## Read this first

- **Authority for the task shape and quality: the upstream specs** in
  `vendor/ale/docs/specs/` (public: https://github.com/AgentsLastExam/ale/blob/main/docs/specs/):
  `task-authoring.md` (the folder and `task.toml`), `verification.md` (the result files),
  `episode-protocol.md` (the stages and `ale validate`), `sandbox-image.md` (images, upload
  modes) and `task-quality-standard.md`. Where a spec is unclear, the engine code under
  `vendor/ale/packages/` decides. Our docs state only the robotics deltas and link there;
  never paraphrase the specs at length.
- **Our deltas:** `templates/README.md` (the task contract, file by file: the verify
  patterns, the closed_loop wire, scoring); `tasks/README.md` is only a note on the worked
  example. **Design records** (dated; their 2026-10-15 notes say what no longer
  holds, and their task form predates the 2026-11-03 move to `task.toml`):
  `docs/dev/20260902-engine-run-submission.md` (task form, registry; its leaderboard
  bridge is retired) and `docs/dev/20260902-contribute-download-upload.md` (its website
  download and upload channel are retired).
- **Docs vs code: code wins.** If a document (this file, `PROGRESS.md`, `docs/`) disagrees
  with the code or `git log`, trust the code, say so, and fix the document.
- **Everything written into the repository is English** — code, comments, docstrings, task
  files, docs, commit messages. Conversation with the owner may be in Chinese.
- Two repositories are **archived, not maintained**: `ale-robotics-benchmark-private` (the
  maintainers' repository this one grew out of; the pre-ALE task form and its harness sit at
  its tag `legacy-harness-final`) and `ale-robotics-benchmark` (the public mirror, tag
  `legacy-v0.5-final`). Do not revive them, do not cite their paths (`ale-domain/`,
  `contrib-staging/`), and do not re-introduce their vocabulary (promote, publish, mirror).

## Workflow rules (every session)

1. **Start by reading `PROGRESS.md`** (repo root): current state, last stopping point, known
   gotchas. Restate them before touching anything.
2. **Before every hand-off, update `PROGRESS.md`** (state / next step / gotchas) and commit.
3. **Disk is the truth**: git + `PROGRESS.md` + this file. Session context is a volatile cache.
4. One milestone per named session; `/clear` between milestones.
5. Large changes start in plan mode; a human approves before execution.

## The task form (there is exactly one)

```
tasks/<platform>/<slug>/
├── task.toml          strict (an unknown key fails): top-level schema_version = "1.4+ale.1"
│                      and artifacts, [task] name = "grounded-intelligence-initiative/<slug>",
│                      [metadata] platform, [environment] resources, [agent] timeout_sec +
│                      network_mode, [verifier] timeout_sec (+ environment_mode = "separate")
├── instruction.md     the prompt — the only thing the tested agent is told
├── environment/       Dockerfile; build context = environment/ less setup/; final stage FROM
│   │                  ghcr.io/agentslastexam/container-ubuntu22-base:0.1.1 (an exact version);
│   │                  py3.12 /opt/venv (installs requirements.txt, the ONE Python dep list)
│   │                  + agent-UID import self-check
│   └── setup/         setup.sh: trusted root, before the agent, cwd /opt/ale/setup; stages
│                      payload/ (practice grader, dev copies) into the agent home
├── solution/solve.sh  the reference implementation; `ale validate` runs it in place of the
│                      agent (staged at ~/.ale-solution, run as the agent, same timeout)
└── tests/             test.sh (`exec "${ALE_VERIFY_PYTHON:-python3}" verify.py`) -> verify.py (the grader),
                       env.py, anchor.json, grader_config.json, robotics_grader/ (vendored kit);
                       staged root-only at /opt/ale/tests — ABSENT while the agent works
```

- Identity = `task.toml`'s `[task].name`, `grounded-intelligence-initiative/<slug>` (the
  GitHub / Hugging Face organization that owns the benchmark, then the slug). The slug is the
  folder name, the registry `slug` and the sheet's Task Name. `[metadata].platform` is the only
  metadata field anything here reads: one of the fixed keys in `tasks/platforms.json` (this
  repository owns the vocabulary; the registry export hands it to the website, which only
  displays it). Older tasks may still carry retired keys there, which nothing reads. A landed
  task sits at exactly `tasks/<platform>/<slug>/` (`scripts/lint_domain.py` battery 7 checks
  the key and the placement; a contribution's pull request puts the folder there directly).
  The metric and its direction live only in `tests/anchor.json` (`metric`, `direction`).
  There is no task-kind field and no evaluation-mode field: the three verify patterns
  (open_loop / closed_loop / artifact_rollout) are **authoring choices inside
  `tests/verify.py`**, not configuration fields. Do not reintroduce retired metadata keys
  from the archived form.
- A task's top level holds only `task.toml`, `instruction.md`, `environment/`, `tests/`,
  `solution/` and `.ale-cache` (`ale lint` refuses anything else, `tools/` included). An MCP
  server is a `[[environment.mcp_servers]]` table naming a program in the image (a wrapper
  script when it needs its own directory or variables); maintainers' review tooling goes in
  `tests/review/`, which rides in the verify upload and which nothing runs.
- The artifact contract (owner, 2026-09-05): the agent leaves ANY artifact in
  `/home/user/submission/` (declared in `task.toml`'s `artifacts`) — a controller process, a
  plan or planner, a URDF / mesh / parameter design, generated code, a report; `tests/` scores
  it however the author decides; `solution/` is simply the reference artifact (a hand-made
  file is fine). Wording everywhere says "submission" / "artifact", never "policy", where the
  artifact is meant ("code-as-policy" as a task-kind example is fine). Shapes + skeletons:
  `templates/README.md` "What a task can be".
- Hidden material is hidden by **stage isolation**: `tests/` reaches the sandbox only for
  verification (`/opt/ale/tests`) and `solution/` only in an oracle episode
  (`~/.ale-solution`); `/opt/ale` is root-only (`chown 0:0`, `chmod go-rwx`), and the solver
  runs at the agent's UID (`robotics_grader.Solver` / `deprivileged` via `runuser`).
  Verification is shared (in the solver's sandbox, as root) unless the task sets
  `[verifier] environment_mode = "separate"`, which restores only the declared `artifacts`
  into a fresh sandbox; a task keeps the mode it has. Proof (recorded on the pre-Harbor
  engine; the mechanism is unchanged except the result path, `/logs/verifier`, reset before
  every attempt, which was checked in the 1cdbc02 code, not by a run):
  `docs/evidence/isolation-audit.md`. The repository
  being public changes nothing here: the anchor is hidden from the tested agent while it
  works, not from people.
- The engine writes `config.json`, `instruction.md` and `trajectory.json` into the staged
  `tests/` at verify time, replacing a task file of that name — grader knobs live in
  `tests/grader_config.json` and are read with `stage_config()`.
- Uploads ignore host modes (files 0644, directories 0755): the engine runs `setup.sh`,
  `solve.sh` and `test.sh` through `bash`, and anything else a stage starts from its own
  folder goes through its interpreter.
- There are no `params`, variants or instruction templates (removed upstream);
  `instruction.md` is used verbatim.

## Gates

- **Engine gates** (from `vendor/ale`, clear `VIRTUAL_ENV`, pin the docker socket):
  `ale lint ../../tasks` and `ale lint ../../templates` (never the repository root: the
  engine would walk into `vendor/ale` and refuse the top-level `skills/` as a removed
  concept), then `ale validate <task_dir>`. Validate runs the task twice, in fresh
  sandboxes: an **untouched** sandbox must score `reward` **0.0**, then the **oracle**
  (`solution/solve.sh`) runs in place of the agent and must score `reward` **1.0**, both
  after the engine rounds `reward` half-up to two decimals; nothing else in the result is
  gated. Both are the engine's own gates: `untouched_nonzero` / `oracle_not_full` (an episode
  that does not complete: `untouched_not_completed` / `oracle_not_completed`), exit 2. Never
  describe the oracle rule as a warning or as "domain policy"; that framing is obsolete.
- `ale-onboard` classifies from the run records, not the exit status: a null-anchor run
  also fails with `oracle_not_full` and must read as `needs_anchor` (measured value
  reported), never as an error. Outcomes: `built` | `needs_anchor` | `needs_input` |
  `gave_up` (cannot be scored fairly) | `error`.
- `scripts/lint_domain.py`: hardware-invariant metric (`tests/anchor.json` names it, with a
  `direction` of higher or lower), positive metric scale, `full_at` present, visible-surface
  isolation (`environment/setup/payload/` never reads the anchor or grader; the practice
  grader prints the raw metric only), image hygiene, anchor provenance (never a published
  figure), `task.toml` completeness, the organization `grounded-intelligence-initiative` in
  `[task].name`, a `[metadata].platform` from `tasks/platforms.json` and placement at
  `tasks/<platform>/<slug>/`.
- `scripts/task_assets.py check`: every task's `environment/assets.sha256` names the dataset
  in `task-assets.toml`, and its `environment/fetch_assets.py` is byte-identical to
  `shared/task_assets/fetch_assets.py`.
- The worked example (`tasks/space/fault_adaptive_spacecraft_docking`) validates in well
  under a minute once its image is built (`validate-44ad164a` on 1cdbc02); the template has
  the same machinery and stops at `needs_anchor` until its anchor is recorded.

## Scoring (the kit owns it; do not re-derive)

```
ratio  = clamp(measured / anchor, 0, cap=1.5)   # inverted for direction: lower
reward = clamp(ratio / full_at, 0, 1)           # the task's score, the one value the engine reads
```

- `tests/anchor.json`: `metric` + `direction` (higher | lower) = the task's metric, declared
  nowhere else; `value` = the reference implementation's own measured value from a
  real run (never a printed figure; null while bootstrapping — that is `needs_anchor`, not an
  error); `full_at` = the ratio at which reward saturates, sized from measured seed spread.
- Metric must be an **outcome on a positive scale**; never wall-clock time / throughput / FPS
  / frequency. A negative reward becomes a positive cost with the anchor's `direction` lower; the kit
  refuses a non-positive anchor.
- `rewards` carries only `reward` (the task's score, 0 to 1); the raw metric, capped `ratio`,
  `match_lock_ok`, `anchor_recorded`, probe counts ride in the verdict's `metrics`.
- The result goes out through the kit, never by hand: `write_verdict(verdict, diagnostic=text)`.
  In the engine (no `path`) it calls `ale_verify.write`, which writes
  `/logs/verifier/reward.json` (`{"reward": r}`, rounded half-up to two decimals, the only
  value the engine reads) and `verification.json` beside it (the complete result: `metrics`,
  the unrounded `metrics.reward_exact`, `diagnostic`). With `path=` (the selftest, a host-side
  run) it writes the plain `{"rewards", "metrics"}` envelope there. Code that must tell the two
  apart tests `os.environ.get("ALE_VERIFY_PYTHON")`.

## Kit vendoring rule

`shared/robotics_grader/` is the **canonical** verify-stage kit (stdlib-only). Every task
carries a **byte-identical vendored copy** at `tests/robotics_grader/`; CI diffs each copy
against the canonical one and fails the pull request on a difference. Nothing re-vendors a
task after it lands, so the copy a maintainer validated is the copy that scores. Edit
`shared/` and copy — never edit a vendored copy in place. Self-test:
`cd shared && python3 -m robotics_grader.selftest`.

## Contributing: one pull request

- The project's contribution guidelines (sign-up in the coordination sheet, the three roles,
  task stages, credit) are two Google Docs with the same content, linked from
  `CONTRIBUTING.md`:
  [English](https://docs.google.com/document/d/1s1E3ymIBLujWf2l7rhu4g9Nl4orBfooK0aDWOdum7-Y/view)
  and [中文](https://docs.google.com/document/d/1gX8v-w-idIPLjzoWgRskT0llE7cKXqg9sVLB0cWfZk0/view).
  Link them as `/view` URLs, never `/edit`. This repository documents what happens here.
- The sheet's Task Name, the folder name and the slug in `task.toml`'s `[task].name`
  (`grounded-intelligence-initiative/<slug>`) are one string (preferably `snake_case`).
  Contributors get write access from the project initiator after signing up,
  push a branch (here or on a fork) adding
  `tasks/<platform>/<name>/`, and open the pull request; CI
  (`.github/workflows/pr-checks.yml`: no secrets, so fork pull requests run it) is the same
  for everyone. No staging area, no promote step, no public mirror: the repository is public.
- Before the pull request a contributor runs `ale lint` + `ale validate` and measures
  difficulty with frontier models (`ale run`); the target is a frontier-model `reward` roughly
  below 0.5, reached by raising the engineering or perception challenge, never by arbitrary
  penalties. Check runs for the pinned stack's own limits first (`docs/contributing-a-task.md`
  §7: a stream silent for 600 s is cut off; the CLI caps a response at 64k tokens and the
  gateway a request at 30 minutes, which leave long-thinking models with empty turns). The
  pinned engine (`bfb7505`) keeps the image's `/opt/venv/bin` on the agent's `PATH`;
  `PROGRESS.md` §8.
- Maintainer flow: read `tests/` and `solution/`, check out the branch, re-run `ale lint` +
  `ale validate` from `vendor/ale`, fix the anchor on the branch if the measurement
  disagrees, merge, then export the registry (`scripts/export_registry.py`) and commit
  `identity-history.json`.

## The registry export, and the website

- `scripts/export_registry.py`: `ale-robotics-task-registry/v2` from the engine's own loader
  (`spec_hash`, `content_digest`), metric/direction/anchor/full_at/cap from `tests/`,
  `platforms` from `tasks/platforms.json`; merges `identity-history.json`; **refuses a dirty
  tree**. Runs inside the engine venv (re-execs itself if needed). Its contract is
  `schemas/task-registry.schema.json`, owned here and tested against the export: change both
  in one commit. The protocol name (`ale-robotics-task-registry/v2`) and the `ALE_*`
  environment variables are runtime identifiers, not brand text: they did not change with the
  rename and must not be edited casually.
- The website (<https://agentic-robotics-benchmark.org>) is a display-only front end (owner
  decision 2026-10-15): it installs the registry export and shows the tasks, and its
  submission page links the two guideline documents. It takes no uploads, no run or score
  submissions, and has no leaderboard. Nothing in this repository depends on the website or
  reads from it; never reintroduce such a dependency.

## Commands

```bash
uv sync
cd vendor/ale && VIRTUAL_ENV= uv sync --frozen --all-groups     # the engine venv (upstream's `just bootstrap`)
cd vendor/ale && DOCKER_HOST=unix:///var/run/docker.sock images/build.sh ale-qemu-runner
#   once per host while upstream has not published ale-qemu-runner:0.1.1 (the docker provider needs it)
cd vendor/ale && VIRTUAL_ENV= uv run ale lint ../../tasks && VIRTUAL_ENV= uv run ale lint ../../templates
cd vendor/ale && VIRTUAL_ENV= DOCKER_HOST=unix:///var/run/docker.sock \
    uv run ale validate <task_dir> --runs-dir /tmp/ale-runs
uv run ale-onboard <spec>.yaml --json
uv run ale-onboard --gate-only tasks/<platform>/<slug> --json
uv run python scripts/lint_domain.py .
python3 scripts/task_assets.py check     # every task's asset list vs task-assets.toml (sync to fix)
cd shared && python3 -m robotics_grader.selftest
cd vendor/ale && VIRTUAL_ENV= uv run python ../../scripts/export_registry.py --tasks ../../tasks \
    --history ../../identity-history.json --out registry.json
#   the website installs registry.json with its own sync script and only displays it
uv run pytest -q
```

## Discipline

- **Task data vs framework code**: `tasks/` and `templates/` are data; `harness/`, `scripts/`,
  `shared/`, `skills/` are code. Author the inputs; never hand-edit run outputs to make a gate
  pass.
- Onboarding goes through `skills/onboard-task/SKILL.md` (router → `onboard-open-loop` /
  `onboard-closed-loop` / `onboard-artifact-rollout`); `tests/test_skills.py` asserts the
  skills carry no retired vocabulary.
- Anti-cheat is **architectural isolation, not detection**: the grader is unreachable by the
  agent, seeds live only in grader memory, nothing the agent prints is a scoring input. The
  probe lock (closed_loop: the same step `t` is queried twice, true and perturbed observation
  in random order with no marker on the wire, only the true reply is applied) and
  re-simulation (open_loop) sit on top of that; they do not replace it.
- No secrets in the repo (`.env`, tokens, keys). No hardware-dependent scored metric. No
  100 MB+ blobs in git. Large agent-visible files live in the Hugging Face dataset named once
  in `task-assets.toml`; tasks pin them in `environment/assets.sha256` and vendor
  `shared/task_assets/fetch_assets.py` byte-identically as `environment/fetch_assets.py`,
  which downloads them at image build. Upstream's `ale assets` is a different mechanism and
  is not used. Never hand-edit the `endpoint` / `repo` lines: edit `task-assets.toml` and run
  `scripts/task_assets.py sync` (`docs/task-assets.md`).
- `tasks/space/fault_adaptive_spacecraft_docking` (contributed in #1, merged 2026-10-15 after
  a maintainer pass) is the worked example of a complete task. It is too easy as merged:
  on 2026-10-15 Opus 5.5 scored reward 1.0 twice (ratio 1.26 and 1.32 against the reference),
  so raising its difficulty is open work. The dummy
  `uav_hover` (formerly `drone_hover`) was removed on 2026-10-15: a task frontier models solve
  trivially is neither a real example nor a task anyone could submit (owner decision).

## Gotchas (details in PROGRESS.md)

- `uv venv --python 3.12` puts its managed interpreter under `/root/.local/share/uv`, which
  the agent account cannot traverse → the venv silently fails for the solver UID with a
  misleading `ModuleNotFoundError`. Set `UV_PYTHON_INSTALL_DIR=/opt/uv-python` and keep the
  Dockerfile's agent-UID import self-check. **Rule: the agent UID must traverse (not only
  read) everything the venv points at.**
- The engine stages `ale_verify` into the image's `python3`, else its `python3.12` (3.12+),
  else the framework's own Python, which lacks the task's packages, and names the one it used
  in `ALE_VERIFY_PYTHON`, set for the verify stage only:
  `tests/test.sh` runs `exec "${ALE_VERIFY_PYTHON:-python3}" verify.py`, never a bare `python3`.
- The desktop's size is a run setting at `bfb7505` (`desktop_resolution`, default 1024x768):
  the docker provider overrides an image's `VNC_RESOLUTION`, and a task cannot declare it. The
  three CAD tasks are 1440x900: run them with `--set desktop_resolution=1440x900`.
- `reward` is rounded half-up to two decimals before any gate reads it: an oracle at 0.995
  passes as 1.0 and an untouched run at 0.004 reads as 0.0. The unrounded value is
  `metrics.reward_exact` in the episode's `verification.json`.
- The docker provider runs `ghcr.io/agentslastexam/ale-qemu-runner:<engine version>` to set a
  sandbox's network rules. Without it every episode ends in `env_error` at provision
  ("could not route the sandbox's access to the host"); build it once per host
  (`images/build.sh ale-qemu-runner`).
- A stage's stdout and stderr are recorded in the episode's execution trace
  (`trace.execution.jsonl`, `command_finished`), not in a file of their own — put what the
  grader has to say into `write_verdict`'s `diagnostic`, which `verification.json` keeps.
- A null anchor is not an error; it is `needs_anchor` with the measured value. The engine
  prints `FAIL  <name>: untouched={...}, oracle={'reward': 0.0}` (exit 2) and names the
  code, `oracle_not_full`, in the run's `validation.json`; read the records.
- When auditing isolation, mount permissions must match a real run, otherwise you measure
  your own audit tool.
- Two docker contexts on the dev machine → always `DOCKER_HOST=unix:///var/run/docker.sock`.
- Judge a background command by its log tail and container state, not the notification's
  exit code. Kill long runs by `pgrep`-selected pid, never a broad `pkill -f`.
- `export_registry.py` refuses a dirty tree on purpose: `content_digest` covers every regular
  file, including ignored `__pycache__/`. `--allow-dirty` is for throwaway exports only: it
  stamps `provenance.source_commit` `<sha>-dirty` + `dirty: true` and never writes the
  canonical `identity-history.json`.
