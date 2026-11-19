# Architecture — how a task runs and what the repository publishes

> Audience: maintainers and contributors. The task contract is
> [`templates/README.md`](../templates/README.md); the trust boundary is
> [`security-model.md`](security-model.md); the design record is
> [`dev/20260902-engine-run-submission.md`](dev/20260902-engine-run-submission.md) (dated: its
> leaderboard bridge is retired).
> The engine's own normative specs live in `vendor/ale/docs/specs/` (`task-authoring.md`,
> `episode-protocol.md`, `verification.md`, `sandbox-image.md`). Where this page and the
> engine disagree, the engine wins.

## 1. What the benchmark measures

The Agentic Robotics Benchmark is an agentic-robotics benchmark built on the upstream ALE
engine. Each task drops an autonomous agent into a sandboxed robotics problem — develop a
controller, plan a manipulation sequence, write code-as-policy, tune a planner — and scores
the artifact it leaves behind
with a hidden grader on hidden seeds. The agent's deliverable is whatever the task's
`instruction.md` asks for (a solver behind `run.sh`, answer files, a frozen checkpoint)
under a declared artifact path, normally `/home/user/submission/`.

Nothing here is a harness of our own. The **upstream ALE engine** (submodule `vendor/ale`)
builds the sandbox, runs the agent, captures the artifact, and runs the task's grader; this
repository owns the **tasks**, the **authoring gate** and the **task registry** it
publishes.

## 2. The task folder

The folder is Harbor's task format, as the engine at `vendor/ale` implements it:

```
tasks/<platform>/<slug>/
├── task.toml         strict configuration: top-level schema_version "1.4+ale.1" and
│                     artifacts, [task] name, [metadata] platform, [environment] resources,
│                     [agent] and [verifier] timeouts, [agent] network_mode
├── instruction.md    the prompt — the only thing the tested agent is told
├── environment/      the image: Dockerfile (context environment/ without setup/; final stage
│   │                 FROM an exact version of an official ALE base), requirements.txt
│   └── setup/        setup.sh: trusted root, before the agent; stages setup/payload/ into
│                     the agent home
├── solution/solve.sh the reference implementation; `ale validate` runs it in place of the agent
└── tests/            test.sh -> verify.py (the grader), env.py (authoritative environment),
                      anchor.json, grader_config.json, robotics_grader/ (vendored kit)
```

Identity is `task.toml`'s `[task] name`, `grounded-intelligence-initiative/<slug>`, whose slug
is the folder name. Everything the grader needs — including the anchor — travels **inside
the folder**. The three verify patterns (open_loop, closed_loop, artifact_rollout) are
authoring choices inside `tests/verify.py`, not configuration fields.

## 3. Engine stages and stage isolation

One episode, as the engine runs it (`vendor/ale/docs/specs/episode-protocol.md`):

```
provision   build environment/Dockerfile (or use [environment].docker_image) plus the
   │        engine's own layer; start the sandbox; /opt/ale is drwx------ root:root
   │
setup       environment/setup/ is uploaded to /opt/ale/setup; setup.sh runs as root, egress
   │        open, cwd = that folder, and stages payload/ into /home/user (practice grader,
   │        dev copies). tests/ and solution/ are NOT in the sandbox.
   │
agent       the harness (claude-code, codex-cli, …) works as the image's agent account
   │        under [agent].network_mode = "no-network" (the sandbox reaches only the model
   │        gateway), within [agent].timeout_sec. It sees instruction.md, the image,
   │        /home/user. It leaves the declared artifact.
   │        [ale validate: solution/solve.sh runs here instead, as the SAME unprivileged
   │         user; solution/ is uploaded to ~/.ale-solution only for that episode]
   │
evidence    trajectory.json (agent transcript, ATIF) from the harness's own logs; declared
   │        artifacts are copied out only for separate verification or when a run asks
   │
verify      shared verification (the default) scores in the same sandbox; with [verifier]
   │        environment_mode = "separate" the solver sandbox is released and a fresh one
   │        receives only the declared artifacts. /logs/verifier is emptied and given to
   │        root alone; tests/ is uploaded NOW to /opt/ale/tests; egress opens; ale_verify is
   │        staged into the image's python3 (>= 3.12, named in ALE_VERIFY_PYTHON);
   │        `bash test.sh` -> verify.py runs as root, cwd = /opt/ale/tests. The grader owns
   │        the simulator, spawns the agent's solver AT THE AGENT'S UID
   │        (robotics_grader.Solver / deprivileged via runuser), draws seeds from OS entropy
   │        in memory, computes the metric from its own state, writes the result.
   │
teardown    sandbox destroyed by default (retention is a run policy; a kept sandbox first
            loses every stage, the solution and the verifier directory)
```

**Stage isolation** is the whole hiding mechanism: a stage folder reaches the sandbox only
when its stage runs, so `tests/` (grader, anchor, config) and `solution/` are simply absent
while the agent works. Two more barriers hold even for what does exist: `/opt/ale` is
root-only, and the solver runs at the agent's UID (the kernel refuses to let one UID trace
another's process), so the agent's code cannot read the grader's memory. It does start with a
copy of the verify stage's environment (`runuser` keeps it), so a grader keeps no secret in
`os.environ`. Nor can it write
the result: before every verification attempt the engine recreates `/logs/verifier` empty
and readable by root alone. Real-run proof, made on the engine before the Harbor migration
and re-checked in the 1cdbc02 code on 2026-11-03:
[`evidence/isolation-audit.md`](evidence/isolation-audit.md).

The engine's untouched/oracle **double pass** (`ale validate`) runs this protocol twice with
fresh sandboxes: the untouched episode must complete with `reward` 0.0 and the oracle
episode with `reward` 1.0, both after the engine rounds `reward` half up to two decimals.
Either miss fails validation (`untouched_nonzero`, `oracle_not_full`; exit 2).
`harness/ale_onboard.py` reads the same records back and tells a null anchor
(`needs_anchor`, the measured value reported) apart from a broken oracle.

## 4. The verdict and the episode record

`robotics_grader.write_verdict({"rewards": {...}, "metrics": {...}}, diagnostic=text)`
records through the engine's `ale_verify.write`, which atomically writes
`/logs/verifier/reward.json` (`{"reward": r}`, `r` rounded half up to two decimals) and
`verification.json` beside it: the complete result, with the kit's `metrics` (the unrounded
reward among them as `reward_exact`) and the `diagnostic` text. The engine reads
`reward.json`, rejects a `verification.json` whose `reward` differs, and keeps only `reward`
as the episode's score. Given a `path`, `write_verdict` writes the plain envelope to that file
instead, for runs outside a sandbox.

- `rewards` — only the gated key, `reward ∈ [0, 1]`. Under validate it must be 0.0
  (untouched) and 1.0 (oracle) after rounding.
- `metrics` — everything worth keeping but never gated: the raw metric (e.g.
  `mean_docking_quality`), the capped `ratio` (reported alongside the reward), `match_lock_ok`,
  `anchor_recorded`, probe and abort counts. They land in `verification.json`.

The engine persists one episode directory `runs/<run_id>/<org>.<slug>-<hex8>/` (under
`ale validate`, `<org>.<slug>-untouched-<hex8>/` and `<org>.<slug>-oracle-<hex8>/`):

| file | what it carries |
|---|---|
| `lock.json` | `task.{name,spec_hash,content_digest,non_verifier_digest,source}`, `framework.{commit,version}`, `agent.{harness,model,version,family}`, `image.*`, `resources.effective`, `verification.mode`, `ale_verify.content_hash` |
| `result.json` | `episode_id`, `status` (`completed`, `agent_error`, `timeout`, …), `rewards{reward}` (present whenever verification completed, including after an agent that timed out, failed or hit a limit; absent otherwise), `metrics{}` (engine diagnostics; empty for these tasks), `phases[]`, `sandboxes[]`, `failure{}` |
| `verification.json` | the verifier's complete result: here the kit's `reward`, `metrics` and `diagnostic` |
| `trajectory.json`, `trace.execution.jsonl`, `trace.transport.jsonl` | agent transcript, engine trace (each stage's stdout and stderr included), model calls |

Run level: `ledger.db` (the only enumeration that also lists interrupted episodes) and, for
`ale validate`, `validation.json`.

**Task identity** is the engine's: `spec_hash` hashes the loaded spec (`task.toml` fields +
`instruction.md` + engine defaults); `content_digest` is a tree digest of every regular file
in the folder (`solution/` included, exec bits included) except the asset roots
(`environment/assets/`, `tests/assets/`, `solution/assets/`) and `.ale-cache`. Editing one
character of `instruction.md` changes `spec_hash`; a stray `__pycache__` changes
`content_digest`. We never re-implement either — `scripts/export_registry.py` reads both from
the engine loader and keeps past pairs in `identity-history.json` at the repository root.

## 5. Scoring

```
ratio  = clamp(measured / anchor, 0, cap)     # anchor / measured for direction: lower; cap = 1.5
reward = clamp(ratio / full_at, 0, 1)         # the gated key
```

- `anchor` (`tests/anchor.json` `value`) is the reference implementation's own measured
  value from a real run — never a printed figure from elsewhere. It starts null: the first
  run measures, reports the value in `metrics`, and scores 0 (`needs_anchor`).
- `full_at` is where the reward saturates, in ratio units — the worst ratio a genuinely
  on-reference run lands on under fresh hidden seeds, sized from a measured spread.
- `cap` (`tests/grader_config.json` `scoring_cap`, default 1.5) bounds how much beating the
  reference can dominate an aggregate.
- Metrics are outcome metrics on a positive scale, hardware-invariant by policy
  (`scripts/lint_domain.py`); a non-positive anchor is refused by the kit because it would
  invert the ordering.

The canonical kit is `shared/robotics_grader/` (`Solver`, `ProbeLock`, `seed_rng`,
`ratio_rewards`, `zero_verdict`, `write_verdict`, `deprivileged`, `clear_agent_account`); every task vendors a
byte-identical copy under `tests/robotics_grader/` because the engine stages exactly the
`tests/` folder and nothing else.

## 6. What the repository publishes: the task registry

```
 tasks/<platform>/<slug>/                      (this repo, committed, clean tree)
     │
     ├─ ale lint / ale validate  ──►  harness/ale_onboard.py  ──►  built | needs_anchor | …
     │
     ├─ ale run <task> --agent <harness> --model <model>  ──►  runs/<run_id>/<episode>/…
     │        (the difficulty test and the formal benchmarking runs; the engine's own records)
     │
     └─ scripts/export_registry.py  (engine loader: spec_hash, content_digest;
              tests/anchor.json: metric, direction, anchor, full_at; grader_config: cap;
              tasks/platforms.json: platforms; identity-history)
              ──►  registry.json  (ale-robotics-task-registry/v2,
                                   contract: schemas/task-registry.schema.json)
              ──►  the website installs it and displays the tasks
```

The repository is public: every task folder is here whole, `solution/` and `tests/` included,
and there is no separate mirror. The website is a display-only front end (owner decision
2026-10-15): it shows the tasks from the registry and links the contribution guidelines. It
takes no uploads and no run or score submissions, has no leaderboard, and nothing here reads
from it. A result is the engine's own episode record (`lock.json`, `result.json`,
`verification.json`) for a task identity the registry names.

## 7. Repository components

```
vendor/ale/                  the engine (submodule, pinned) — never reimplemented here
tasks/  templates/task/      the tasks and the one template (flat layout); identity-history.json at the root
tasks/platforms.json         the platform vocabulary ([metadata].platform keys and labels)
shared/robotics_grader/      canonical verify-stage kit (stdlib-only)
harness/ale_onboard.py       ale-onboard: instantiate + gate (lint -> validate -> outcome)
scripts/export_registry.py   registry v2 (engine loader identity; dirty-tree refusal)
scripts/lint_domain.py       domain lint (what `ale lint` cannot know)
schemas/                     the registry's JSON Schema (owned here; the export is tested against it)
skills/onboard-*/            agent-facing onboarding skills
tests/                       pytest
```

## 8. Operational notes

- Host-side tooling runs on uv (`pyproject.toml`, py3.11). The engine has its own venv under
  `vendor/ale`; run engine commands from there with `VIRTUAL_ENV=` cleared and
  `DOCKER_HOST=unix:///var/run/docker.sock` pinned.
- Task images: py3.12 `/opt/venv` with a world-traversable interpreter
  (`UV_PYTHON_INSTALL_DIR=/opt/uv-python`) and an agent-UID import self-check at the end of
  the Dockerfile. At verify time the engine takes `python3`, else `python3.12`, and refuses
  one older than 3.12.
- `[agent].network_mode = "no-network"` during the agent phase means any lazy first-use
  download must be baked into the image.
- `ale validate` does not print a stage's stderr. The episode's `trace.execution.jsonl` keeps
  it (the stage's `command_finished` event), and the kit's `diagnostic` text is in
  `verification.json`.
- The worked example validates in well under a minute once its image is built
  (`validate-44ad164a` on 1cdbc02); the template has the same machinery and is the place to
  test it.
