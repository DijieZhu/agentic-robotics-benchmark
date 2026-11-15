# PROGRESS.md · progress log for agentic-robotics-benchmark

> **What this is:** the progress source of truth for this repository, maintained by people
> and agents alike. It records milestone state, the last stopping point, known gotchas and the
> next step. **Disk is the truth** (git + this file + `CLAUDE.md`); a session's context is a
> volatile cache. Read this file first; update it (state / next step / gotchas) and commit
> before every hand-off. Where it disagrees with the code or `git log`, the code wins: fix
> this file.
>
> Last updated: 2026-11-09 (§11: `vendor/ale` @ `bfb7505`, a task names its GPU models).
> Before that: 2026-11-07 (§10: the asset dataset is renamed `ale-robotics-assets`);
> 2026-11-03 (§8: the tasks move to the Harbor-based engine, `vendor/ale` @ `1cdbc02`).

## 0. 2026-09-07 — created by the rebrand from "ALE Robotics"

Origin: the maintainers' repository `ale-robotics-benchmark-private` at `86c2285` (its
history, including the Chinese progress log, stays there, archived) and the public mirror
`ale-robotics-benchmark` (archived, tag `legacy-v0.5-final`). This repository starts with a
fresh history and replaces both: it is public and carries every task folder whole, `oracle/`
and `verify/` included.

- **Layout flattened.** `ale-domain/tasks/` → `tasks/`, `ale-domain/templates/` →
  `templates/`, `ale-domain/identity-history.json` → `identity-history.json`,
  `ale-domain/_evidence/` → `docs/evidence/`, `ale-domain/README.md` → `tasks/README.md`.
  Verified with the engine loader: `drone_hover`'s `spec_hash` and `content_digest` are
  unchanged (a folder's location is not part of its identity), so `identity-history.json`
  still describes the tree.
- **Staging retired.** Both contribution channels (a pull request opened by hand; the
  website upload whose bot opens the same pull request) land a task directly at
  `tasks/<direction>/<slug>/`. Deleted: `contrib-staging/`, `scripts/promote_contribution.py`,
  `.github/workflows/promote-on-merge.yml`, `scripts/publish_public.py` and its tests.
  `.github/workflows/pr-checks.yml` now runs the full check set on the final path, needs no
  secret (fork pull requests run it), and its registry identity cross-check treats a slug the
  history does not know as a notice, not a failure. `scripts/lint_domain.py` gained battery 7
  (placement: `tasks/<metadata.direction>/<name>/`); `harness/ale_onboard.py` and the lint
  walk only `tasks/` and `templates/`, never the repository root (that would descend into
  `vendor/ale`). `.github/PULL_REQUEST_TEMPLATE.md` is new.
- **Renamed.** `pyproject` name `agentic-robotics-benchmark`; the download is
  `agentic-robotics-task-template.zip` with folder `agentic-robotics-task-template/`;
  `export_registry.py` release name "Agentic Robotics Benchmark" (`provenance.tasks_repo`
  follows the directory name); `GETTING-STARTED.md` rewritten for the two channels and the
  new domain `agentic-robotics-benchmark.org`; README, README.zh-CN, CLAUDE.md, CONTRIBUTING.md
  (new), docs and skills rewritten for the brand and the flat paths. Added `LICENSE`
  (Apache-2.0, from the old public mirror) and `NOTICE`.
- **Unchanged on purpose** (runtime identifiers stay until a separate decision): `ALE_*`
  environment variables, the protocol names `ale-engine-run/v1` and
  `ale-robotics-task-registry/v2`, the console scripts `ale-onboard` / `ale-export`,
  `harness/export_run.py`'s exporter and benchmark names (they travel in bundle manifests
  the server reads), the engine pin `vendor/ale` @ `90a7c1c`, and every file of
  `tasks/control/drone_hover` (its `task.yaml` and `verify/verify.py` still carry three
  comments saying "ALE Robotics domain policy"; editing them would change the task's
  `content_digest`, so they go when the task goes: owner decision, delete `drone_hover` when
  the first community task is merged).
- **Not carried over** (owner decision; they live in the archived private repository):
  the Chinese `PROGRESS.md` history, `docs/prompts/` (session hand-off briefs),
  `docs/dev/archive/` (the pre-ALE snapshot and audits), `docs/tree_diagram/` (the taxonomy
  visualisation source; the workspace repository keeps the taxonomy markdown).
- **Verified at creation:** `uv run pytest -q` 225 passed, 1 skipped (the website registry
  cross-check, which runs when the website checkout is next to this repository); `ale lint ../../tasks` and
  `ale lint ../../templates` ok from `vendor/ale`; `scripts/lint_domain.py .` ok;
  `robotics_grader` selftest passed; the template's vendored kit byte-identical to
  `shared/`; the workflow file parses; a clean registry export reproduces
  `identity-history.json` byte for byte and the template zip builds (22 entries); ruff reports
  the same nine pre-existing style notes as before (non-blocking in CI).

## 1. 2026-09-08 — pushed; the pending contribution re-opened here

- `main` pushed to `https://github.com/Grounded-Intelligence-Initiative/agentic-robotics-benchmark`
  after the owner signed off the privacy checklist; the first `pr-checks` run on `main` is
  green (engine lint over `tasks/` + `templates/`, identity cross-check, static checks).
- The contribution `fault_adaptive_spacecraft_docking` (old repository PR #14, uploaded on
  the website by Jiankai-Sun, revision 2) is re-opened here as https://github.com/Grounded-Intelligence-Initiative/agentic-robotics-benchmark/pull/1 on branch
  `contrib/fault_adaptive_spacecraft_docking`: commit 1 = the 20 uploaded files byte for
  byte at `tasks/control/fault_adaptive_spacecraft_docking/`; commit 2 = `verify/robotics_grader/`
  re-vendored from `shared/` (the upload carried the 2026-09-05 kit). Old PR #14 closed with
  a pointer. Still to do by a maintainer before merge: `ale validate` (the anchor was a
  dry-run measurement; the oracle runs three hours).

## 2. 2026-09-08 — website live; both channels rehearsed

- The website (`agentic-robotics-benchmark-website`, formerly `ale-robotics-website`) is live
  on `https://agentic-robotics-benchmark.org` with the two-channel Contribute page, the
  download built from this repository at `b04359f`, and the registry exported from the same
  commit; `ale-robotics.org` and its `www` 301 to the new domain.
- Dress rehearsal with a template-derived task `rehearsal_reach_20260908` (uav/control;
  anchor `0.0813` measured by `ale validate` on the maintainer's machine, oracle exactly 1.0):
  channel A (pull request #2, opened by hand) and channel B (pull request #3, opened by the
  website's bot after the upload precheck) produced byte-identical trees at
  `tasks/control/rehearsal_reach_20260908/` (20 files, same modes, nothing outside the
  folder), and `pr-checks.yml` passed on both. Both were closed without merging and their
  branches deleted, as planned.
- Website follow-ups found by the rehearsal and fixed the same day: the `www` alias of the
  new domain (Caddyfile `WWW_DOMAIN`), and a 500 on `POST /api/v1/challenges` caused by the
  rename creating a second release row (the website's seed now keys the release by version
  and merges duplicates).

## 3. 2026-09-08 — the paper-onboarding skill link removed

- `.claude/skills/onboard_paper` was a symlink to `skill/`, the paper-reproduction
  onboarding skill of the project's first months; its target had been deleted long before
  the rebrand copied the dangling link over. Removed (owner decision: the benchmark is not
  about reproducing papers, so no replacement). The repository has no `.claude/` directory
  now; the agent-facing skills are the ones under `skills/onboard-*/`, routed by
  `skills/onboard-task/SKILL.md`. `docs/research/agentic-robotics-survey.md` pointed at
  the old `skills/SKILL.md`; it then pointed at the router (the survey left this repository
  on 2026-10-17).

## 4. 2026-10-12 — `drone_hover` is now `tasks/uav/uav_hover`; tasks are placed by platform

- Owner decisions: `tasks/control/drone_hover` becomes `tasks/uav/uav_hover`
  (`name: uav_hover`), a dummy task that frontier models complete easily, kept only to show
  what a task's file structure looks like. This replaces "delete it with the first merged
  community task". `tasks/README.md` is now one paragraph saying so: there is no open-loop /
  closed-loop distinction any more.
- A landed task sits at `tasks/<metadata.platform>/<name>/` (was
  `tasks/<metadata.direction>/<name>/`). Changed with it: `scripts/lint_domain.py` battery 7,
  the instantiate target of `harness/ale_onboard.py`, the template download's getting-started
  text (`scripts/build_template_zip.py`), the comment in `templates/task/task.yaml`, the pull
  request template, the comments in `pr-checks.yml`, README, README.zh-CN, CONTRIBUTING,
  CLAUDE.md, docs and skills. The tests now refuse a direction-named folder.
- Identity: the new `name` gives the task a new `spec_hash` and `content_digest`, so the
  registry identity cross-check reports `uav_hover` as a new task; `identity-history.json`
  keeps the `drone_hover` record. `image/` is untouched, so the image identity (and an image
  already built from it) carries over; its Dockerfile still says `drone-hover` in two log
  lines and one comment.
- Verified locally (the `pr-checks.yml` steps): `ale lint ../../tasks` and
  `../../templates` ok; registry export + identity cross-check ok (`uav_hover` new);
  `robotics_grader` selftest and the vendored-kit diff ok; `uv run pytest -q` all passed with
  the engine venv present (the template-zip test, which refuses a dirty tree, passed on a
  clean throwaway commit); `scripts/lint_domain.py .` ok; ruff reports the same nine
  pre-existing notes.

## 5. 2026-10-15 — the benchmark owns its taxonomy; `direction` and `metric` metadata retired

- Owner decisions: this repository is self-sustaining, and the website is only a front end
  that displays what this repository exports. The platform vocabulary is
  `tasks/platforms.json`, fixed keys `arm` (robotic arm), `hand` (dexterous hand),
  `quadruped`, `humanoid`, `uav`, `driving` (autonomous driving), `underwater` (UUV/AUV),
  `space` (space robot), `soft`, `surgical`, `swarm`, `subsystems`. `metadata.direction` is
  removed: platform is the only taxonomy field. `metadata.metric` is removed: upstream ALE
  does not define it (its metadata is "optional descriptive values; no runtime behavior"), and
  the metric and its direction are declared in `verify/anchor.json` only, where the grader
  reads them.
- `scripts/lint_domain.py`: battery 7 refuses a `metadata.platform` outside the vocabulary;
  the metric checks (named, not hardware-dependent, `direction` higher or lower) read
  `verify/anchor.json`; the two "task.yaml and anchor.json agree" checks are gone; stale
  `direction` / `metric` metadata is ignored, like `genre` / `paper`.
- `scripts/export_registry.py`: `platforms` comes from `tasks/platforms.json`; a task whose
  platform is not a key is refused; `metric_name` / `metric_direction` / `metric_label` come
  from `verify/anchor.json`; no per-task `direction`, no `directions`, no `category_tree`, no
  `hardware_invariant` (a self-declaration nothing checked; the lint's token rule is what
  enforces invariance); `--carry-over` is removed; `tasks/platforms.json` is held to the same
  clean-tree rule as the task folders, since it goes into the document. `schema_version` stays
  `ale-robotics-task-registry/v2` on purpose: the website's sync treats any other version as
  a legacy merge and its seed refuses unknown versions, whereas a v2 document without
  `direction` fails the website's current schema closed (nothing is written).
- `harness/ale_onboard.py`: the spec's `platform` must be a key (the `misc` fallback is gone),
  and the spec's metric name and direction are written into the copied `verify/anchor.json`
  (a name without a direction is `higher`, never the template's `lower`). The gates check that
  the grader reports the anchor's metric: an oracle record without it is `needs_input` (the
  anchor's name is what the export publishes and the server's recompute looks up). An anchor
  still at `TODO` names no metric, and `needs_anchor` then asks for the name with the value.
- Template: `metadata: {platform: TODO}` with the key list; the anchor's `metric` is `TODO`
  (the template grader's `mean_goal_distance` is an example, not a default);
  `templates/README.md`, the template-zip guide, README, README.zh-CN, CONTRIBUTING,
  CLAUDE.md, docs and skills follow; the two 2026-09-02 design records carry a dated note. The
  pointers that called `tasks/README.md` the task contract now point at `templates/README.md`,
  and the upload docs carry an interim note (the website still runs the old form: open the
  pull request yourself). The test that read the sibling website checkout is deleted (the
  dependency pointed the wrong way).
- `uav_hover`: metadata reduced to `platform: uav`; its anchor gains `"direction": "higher"`
  (its grader already hard-codes higher), so its `spec_hash` and `content_digest` change again
  (still not in `identity-history.json`; CI reports it as a new task).
- Verified locally (the `pr-checks.yml` steps): `ale lint ../../tasks` and `../../templates` ok
  (`tasks/platforms.json` is invisible to the engine); registry export ok (`platforms` = the 12
  keys; no `direction`, `directions` or `category_tree`) and identity cross-check ok (`uav_hover`
  new); `robotics_grader` selftest and the vendored-kit diff ok; `uv run pytest -q` all passed
  (the template-zip test on a clean throwaway commit); `scripts/lint_domain.py .` ok; ruff the
  same nine pre-existing notes.
- Still bound to the website (not changed here, candidates for a later pass): website policy
  copied into this repository's text (upload URL, size limits, the precheck rule list); the
  JSON Schemas of the registry and of `ale-engine-run/v1` exist only in the website
  (`packages/contracts/schemas`); `provenance.tasks_repo` is the clone's directory name; the
  `episode_id` pattern in `harness/export_run.py` mirrors the server's.

## 6. 2026-10-15 — display-only website; the spacecraft task is the worked example

- Owner decisions: the website is a display-only front end. It shows the tasks, and its
  submission page links the contribution guidelines (two Google Docs, English and Chinese,
  `/view` links). It takes no uploads and no run or score submissions, and the leaderboard is
  removed for now. The registry and submission-bundle JSON Schemas are decoupled from the
  website. `ale-export` is removed. Contribution follows the guidelines: sign up, register
  the Task Name (= folder = `task.yaml` `name`), measure difficulty with frontier models
  (target roughly below 0.5), and open a pull request.
- #6 (kit). The grader's seed stream is BLAKE2b keyed by OS entropy. Before, splitmix64 let
  a solver rebuild the stream from probe perturbations and compute the next episode's seed.
  - After an aborted episode the solver is restarted: at most 3 times, with its process group
    stopped and READY required again. A stray reply line no longer cascades into every later
    episode.
  - The template charges an aborted episode what standing still would cost. It used to drop
    aborted episodes, so a weak policy could abort its bad starts and score 1.0.
  - Malformed actions abort an episode instead of crashing the grader.
- #1 merged (merge commit, contributor history kept) after a maintainer pass on its branch.
  - The agent held a byte-identical simulator copy, and 31-bit LCG seeding let it
    brute-force each episode's seed. Episodes now use 64-bit, keyed-hash seeding.
  - The instruction now matches the grader.
  - The anchor was re-measured: 40 fresh oracle runs, mean 0.7440 (sd 0.063); `full_at` went
    from 0.90 to 0.75. With 0.90, 2 of 31 fresh runs failed `ale validate`.
  - Eight fresh validates passed. The `contrib/fault_adaptive_spacecraft_docking` branch is
    kept because the guidelines link to it.
- This change:
  - `tasks/uav/uav_hover` removed. A dummy that frontier models solve cannot be a real
    example or a submittable task. `tasks/README.md` names the spacecraft task as the worked
    example.
  - Removed: `harness/{export_run,bundle_crypto,redact,trajectory,integrity}.py`, their
    tests and fixtures, `docs/submission-format.md`, `scripts/build_template_zip.py` and
    its tests, and the `zstandard` and `pyrage` dependencies.
  - `schemas/task-registry.schema.json` is the registry's contract, tested against the
    export.
  - `provenance.tasks_repo` is the origin's `owner/name`.
  - Docs follow the guidelines: one path, a frontier-model difficulty test, and no website
    channel. The security model now lists the limits that matter (open egress during verify;
    the probe lock is not a proof of honesty).
- Measured on 2026-10-13, on the previous version of the task: Fable 5.1 1.0, Opus 5.5 1.0,
  Opus 5 0.84, Fable 5 0.74. The 2026-10-15 runs on the merged version use an evaluation copy
  whose image only adds harness workarounds (outside the repository, `arb-eval`): the stream
  idle timeout, a `.pth` so the harness's hard-coded `PATH` still finds the venv's numpy, and
  for the later runs `CLAUDE_CODE_MAX_OUTPUT_TOKENS`. Results (n = 1 per configuration):
  - Opus 5.5: reward 1.0 twice (`mean_docking_quality` 0.940 and 0.982; ratio 1.26 and 1.32;
    46/48 and 48/48 docked).
  - Opus 5: the first run ended in an API error before writing `run.sh`. The second hit the
    3 h agent timeout (no score); its leftover submission, scored afterwards in the sandbox,
    reaches reward 1.0 (0.740, ratio 0.99).
  - Fable 5.1 and Fable 5: empty submissions, reward 0. Every long turn hit the response
    cap (64k, then 100k) or the gateway's 30-minute request limit (at 128k) with no text and
    no tool call, so this is an evaluation-stack failure, not difficulty.
  - Conclusion: the task does not stump frontier models; its difficulty must go up (results
    posted on #1 for its author). About $104 at list prices.

## 7. 2026-10-21 — task assets: one configuration file names the asset repository

Tasks with large agent-visible files fetch them at image build from the Hugging Face dataset
`grounded-intelligence-initiative/ale-robotics-assets` (first used by the open pull
request #13). The dataset is now named once, in `task-assets.toml`; `scripts/task_assets.py`
writes it into every task's `image/assets.sha256` (`sync`), checks it in CI (`check`) and
writes a new task's list (`list`); `shared/task_assets/fetch_assets.py` is the canonical
fetcher that each such task vendors byte-identically (it also takes an `endpoint`, so a mirror
works). Renaming the dataset = edit `task-assets.toml`, run `sync`, commit; every task with
assets then gets a new content digest (re-export the registry after the merge). Usage:
`docs/task-assets.md`.

- Uses: `robotic_arm_cad_design_from_photos` (#15, merged; dataset commit `ccf1b9b`) and the open #13
  (`robotic_arm_replica_requirements_from_photos`), both on the canonical fetcher and the synced header.
- Next: export the registry for #15's new slug (`identity-history.json`).

## 8. 2026-11-03 — the tasks move to the Harbor-based engine

Upstream ALE adopted Harbor's task format. `vendor/ale` moves from `90a7c1c` to upstream
`main` `1cdbc02` (engine version 0.1.1), which no longer loads a `task.yaml`, so every task,
the template, the kit and the docs move with it. What a task measures does not change: the
instructions, the images' software, the graders' numbers, the anchors, the hidden data and
the agent-visible payloads are the same.

- **The task folder.** `task.yaml` → `task.toml`: `schema_version = "1.4+ale.1"`, strict
  (an unknown key fails); `[task] name = "grounded-intelligence-initiative/<slug>"`, the slug
  still the folder name, the registry `slug` and the Task Name; `[metadata] platform`;
  resources in `[environment]`; `timeouts.agent` / `timeouts.verify` → `[agent].timeout_sec`
  / `[verifier].timeout_sec`; `network: {mode: block}` →
  `[agent].network_mode = "no-network"`; separate verification →
  `[verifier] environment_mode = "separate"` + `[verifier.environment]`;
  `tools/mcp/*.toml` → `[[environment.mcp_servers]]`. `image/` →
  `environment/` (build context: `environment/` less `setup/`); the base
  `container-ubuntu22-base:latest` → `:0.1.1`, since `ale lint` refuses `latest` and an
  untagged `FROM`. `setup/run.sh` + `setup/payload/` → `environment/setup/setup.sh` +
  `environment/setup/payload/`, staged at `/opt/ale/setup` and run there as root (a fixed
  1800 s bound; no timeout field). `verify/` → `tests/` (`run.sh` → `test.sh`), staged
  root-only at `/opt/ale/tests`. `oracle/run.sh` → `solution/solve.sh`, staged at
  `~/.ale-solution` and run there as the agent. A task's top level holds only `task.toml`,
  `instruction.md`, `environment/`, `tests/`, `solution/` and `.ale-cache`; review tooling
  that sat in a task's `tools/` moved to `tests/review/`. `params`, variants and instruction
  templates are gone upstream (no task used them).
- **The result.** The engine reads `/logs/verifier/reward.json` (`{"reward": x}`, x finite in
  [0, 1], rounded half-up to two decimals) and keeps `verification.json` beside it. The kit's
  `write_verdict(verdict, path=None, diagnostic=None)` takes the same `{"rewards", "metrics"}`
  verdict as before. In the engine it calls `ale_verify.write`, which writes both files
  (`verification.json` carries `metrics`, the unrounded `metrics.reward_exact` and the
  grader's `diagnostic`); with a `path` it writes the plain envelope there (the selftest, a
  host-side run). `ale_verify.Verification` / `CheckResult`, `ALE_VERDICT_PATH` and
  `ALE_VERIFICATION_PATH` are gone.
- **The gates.** `ale lint` and `ale validate` still run from `vendor/ale`, whose venv is set
  up with `uv sync --frozen --all-groups`. Validate gates `reward` alone, after the rounding:
  untouched 0.0, oracle 1.0 (`untouched_nonzero`, `oracle_not_full`; an episode that does not
  complete fails as `untouched_not_completed` / `oracle_not_completed`; exit 2). The kit is
  vendored at `tests/robotics_grader/`, and `pr-checks.yml` compares that path.
- **Unchanged on purpose.** The Hugging Face asset flow (`environment/assets.sha256` + the
  vendored `environment/fetch_assets.py`, at image build, pinned by dataset commit and per-file
  sha256); upstream's `ale assets` (one gated dataset per source repository, pulled before the
  build) is a different mechanism and is not adopted. Each task's verification mode (shared,
  or separate where a task already was). Processes started by setup outlive it, shared
  verification runs in the solver's sandbox as root, separate verification restores only the
  declared `artifacts`, and `tests/` and `solution/` are absent while the agent works.
- **Identity.** Every task's `spec_hash` and `content_digest` change (the name gains the
  organization, and every file moved), so the registry identity cross-check fails for every
  known slug until `identity-history.json` records the new identities: the history was
  re-exported from the clean converted tree and pushed with the migration.
- **Validated so far.** `tasks/space/fault_adaptive_spacecraft_docking`, converted, on
  `1cdbc02`: run `validate-44ad164a` passed, untouched `reward` 0.0 ("scoring 0: no
  submission at /home/user/submission/run.sh") and oracle 1.0 (ratio 1.02,
  `mean_docking_quality` 0.759, 48 episodes, 690 probes, match lock ok).
  `tasks/arm/arm_hidden_mechanism_box` (setup-started lab service, separate verification, the
  in-loop Filament renderer): `validate-e683055d` passed, untouched 0.0, oracle 1.0 (8
  episodes, 110/110 probes). Both runs used the tree before the follow-up edits (comments,
  the kit's host-run fallback); the engine-path code they exercised is unchanged. The other
  tasks and the template have not been validated on the new engine.
- **Agent-run limits** (Open items). At `1cdbc02` the claude-code adapter raises the CLI's two
  stream idle timers from 300 s to 600 s, and the gateway writes an SSE keep-alive after 15 s
  of upstream silence; the harness's values win over an image `ENV` of the same names, so the
  evaluation copies' timer workaround no longer changes anything. The harness now keeps the
  sandbox's own `PATH` instead of a fixed `/usr/local/bin:/usr/bin:/bin`. Neither change has
  been measured on our tasks yet. The CLI's response cap and the gateway's 1800 s request
  limit are unchanged (`docs/contributing-a-task.md` §7).

Gotchas found on the way:

- **`ale-qemu-runner`.** The docker provider runs `ghcr.io/agentslastexam/ale-qemu-runner` at
  the engine's version to install each sandbox's iptables rules, and upstream has not
  published `0.1.1`. Without it both episodes end in `env_error` at provision ("could not
  route the sandbox's access to the host: … manifest unknown") and validate reports
  `untouched_not_completed` / `oracle_not_completed`. Build it once per machine:
  `cd vendor/ale && images/build.sh ale-qemu-runner` (the script pulls a published version
  and builds one that is not). The base `container-ubuntu22-base:0.1.1` is published.
- **`ALE_VERIFY_PYTHON`.** The engine stages `ale_verify` into `python3`, else `python3.12`
  (3.12+ required), and names the one it used in `ALE_VERIFY_PYTHON`, set for the verify
  stage only. `tests/test.sh` runs `exec "${ALE_VERIFY_PYTHON:-python3}" verify.py` (the
  default only matters for a run by hand), never a bare `python3`. A grader that must tell an
  engine run from a host-side run tests `os.environ.get("ALE_VERIFY_PYTHON")`; outside the
  engine `write_verdict` needs a `path` or `ROBOTICS_GRADER_VERDICT_PATH`. This replaces the
  `python3` line under Gotchas below.
- **Stage cleanup before retention.** Before the engine keeps or snapshots a sandbox it
  deletes `/opt/ale/setup`, `/opt/ale/tests`, the solution and `/logs/verifier`, so anything a
  grader leaves for maintainers must live outside `/opt/ale` (the so101 assembly grader now
  writes its replay record to the root-only `/opt/arb-verify`).
- **Artifacts and stopped episodes.** Run-level `artifacts.collect` defaults to `none` (was
  `host`): set `--set artifacts.collect=host` to keep submissions on the host. An agent that
  stops early (timeout, agent_error, a limit) is still verified, and its `result.json` carries
  both the failure status and `rewards`. The agent deadline starts at harness launch, after
  the CLI install and the network seal.
- **Rounding.** `reward` is rounded half-up to two decimals before validate or a run reads it:
  an oracle at 0.995 passes as 1.0 and an untouched run at 0.004 reads 0.0, where the old
  gate wanted exactly 0 and exactly 1. The unrounded value is `metrics.reward_exact` in the
  episode's `verification.json`; `full_at` is sized as before.
- The engine writes `config.json`, `instruction.md` and `trajectory.json` into the staged
  `tests/`, replacing a task file of that name (grader knobs stay in `grader_config.json`).
- Uploads ignore host modes (files 0644, directories 0755): stage entries run through `bash`,
  and a stage that starts another of its own files calls the interpreter.

## 9. 2026-10-21 — spacecraft hardening validated on PR #10

- PR #10 (`hardening/fault-adaptive-spacecraft-docking-stateful-valves`) now composes three
  terminal-estimation challenges without extending the 400-step horizon: six physical-nozzle
  shroud vortex cells with public hybrid load integrators and one hidden common initial parity;
  a radially oscillating docking boom whose current extension must be tracked; and a redundant
  boom encoder whose displacement polarity is hidden initially and independently re-selected
  at the blackout-producing re-key. The camera can identify the polarity only after aligning
  its delayed timestamp, so the blackout requires an explicit unresolved sensor hypothesis.
- Capture tolerances are 0.085 m position, 0.060 m/s velocity, 0.020 rad yaw and 0.015 rad/s
  yaw rate for 30 consecutive steps. The episode remains 400 steps; failures and collisions
  are genuine closed-loop outcomes, not time-budget or malformed-output penalties.
- The reference controller identifies the mux from timestamp-aligned camera/radius samples,
  tracks the six hybrid wake modes, predicts their transitions inside its allocator, and uses
  the measured boom radius plus differentiated radial rate. Four fresh oracle calibrations
  measured 0.4432074102, 0.3634887656, 0.5031788493 and 0.5221913496 (mean 0.4580165937,
  population sd 0.0618725521). The anchor uses that mean and `full_at: 0.75`; a final
  `ale validate` passed with untouched reward 0 and oracle reward 1 (23/48, raw 0.4824985816).
- A fresh GPT-6.1 Sol interactive trial saw the final specification from the start, explicitly
  implemented the encoder-polarity calibration, and was recovered after the 30-minute agent
  budget. Independent hidden replay completed all 48 episodes: 10 docks, 6 target collisions,
  match lock passed, raw 0.2217035448. A second scored replay produced 12 docks, 7 collisions,
  raw 0.2625658218, ratio 0.5732670506 and reward 0.7643560675. This no longer saturates the
  benchmark; its success rate is 21--25%, versus the previous Opus 5.5 46--48/48 result.
- Not yet validated on the Harbor-based engine (§8): PR #10's branch pins `vendor/ale` at
  `90a7c1c`, and §8's `validate-44ad164a` on `1cdbc02` ran the version before this hardening.

## 10. 2026-11-07 — the asset dataset is renamed `ale-robotics-assets`

The Hugging Face dataset that serves the tasks' large agent-visible files was renamed from
`grounded-intelligence-initiative/agentic-robotics-last-exam` to
`grounded-intelligence-initiative/ale-robotics-assets` (the Hub answers the old name with a redirect).
Following section 4 of `docs/task-assets.md`: `task-assets.toml` names the new dataset and
`scripts/task_assets.py sync` rewrote the `repo` line of every task's `environment/assets.sha256` and
re-vendored the fetcher (its docstring names the dataset). File lists, hashes and pinned dataset commits
are unchanged, since a Hub rename keeps the repository's history. Every task with assets has a new
content digest, so the registry is re-exported after the merge.

## 11. 2026-11-09 — the engine pin lets a task name its GPU models

`vendor/ale` moves from `1cdbc02` to upstream `main` `bfb7505` (engine version still 0.1.1,
eleven merged pull requests). AgentsLastExam/ale#108 lets a task name the GPU models it accepts
with Harbor's `gpu_types`, in `[environment]` and `[verifier.environment]`; without it every
model is accepted. Every provider gives a sandbox only an accepted model, or refuses the task
before it starts. `1cdbc02` refused the key as an unsupported Harbor field.

- Each GPU task names the models it was tested on (2026-11-09), cheapest first, since Daytona
  tries them in that order (RTX 4090 $0.57/h, RTX 5090 $0.74/h, H100 $2.27/h). The other
  Daytona models were tested only where the cheap ones fell short, and the H100 on the team's
  own machine; pull request #71 holds every run.
  - `hand/hot3d_sharpa_retarget` names none: it validated on the RTX 4090, RTX 5090, RTX PRO
    6000, H200 and H100.
  - `driving/driving_safety_supervisor` and `uav/aerial_search_geolocation`:
    `["RTX-4090", "H100"]`. Their agents' PyTorch (2.7.1, built for CUDA 12.6) has no kernels
    for the RTX 5090. On the H100 the simulators render the same frames as on the RTX 4090,
    1.4 (aerial) to 2-5 (driving) times slower on a loaded host.
  - `swarm/hazard_bay_build`: `["RTX-4090", "RTX-5090"]`. On the H100 its render canary
    matched the RTX 5090's, but the missions did not run on the loaded host; not listed until
    they do.
  - `arm/robotic_arm_environment_twin_from_real_photo`: `["RTX-4090", "RTX-5090"]`, in
    `[environment]` only (its verifier has no GPU). OVRTX crashes the H100 at its first
    render (`ERROR_DEVICE_LOST`).
  - `arm/object_finding_and_organizing_on_messy_desk`: `["RTX-4090"]`. Isaac Sim 4.5 crashes
    at start on the RTX 5090, and on the H100 its RGB cameras come out about 30 % darker than
    on the RTX 4090 (depth identical).
- Rendering needs the graphics half of the host's NVIDIA driver (Vulkan, EGL, OptiX), which
  compute-only server installs leave out: on such a host every rendering task fails whatever
  the model. The H100 results above were taken with the matching driver's graphics libraries
  added to the test containers.
- **Identity.** Every task's `spec_hash` changes, because the engine's task specification now
  carries `gpu_types` (empty when a task names none); without that key every hash equals the
  recorded one. The five tasks whose `task.toml` changed also get a new `content_digest`. The
  registry identity cross-check fails for every known slug until `identity-history.json` is
  re-exported after the merge.
- Nothing else here changes: `ale lint`, `lint_domain.py`, `task_assets.py check`, the kit
  selftest and `pytest` pass as they did on `1cdbc02`. The limits in
  `docs/contributing-a-task.md` §7 and the verify-stage egress in `docs/security-model.md` §4
  hold at `bfb7505`; §7's code references now point there. The docker provider now requests
  every NVIDIA driver capability, graphics included (AgentsLastExam/ale#104). Verification
  falls back to the framework's own Python when the image has no 3.12+
  (AgentsLastExam/ale#100), which lacks the task's packages, so a grader that imports them
  still needs the image's own.
- The desktop's size became a run setting (AgentsLastExam/ale#107): the docker provider starts
  every desktop at the run's `desktop_resolution` (default 1024x768), whatever the image's
  `VNC_RESOLUTION` says, and a task cannot declare its own. The three CAD tasks
  (`robotic_arm_cad_design_from_photos`, `uav_cad_design_from_photos`,
  `bimanual_cad_design_from_photos`) were designed, validated and measured at 1440x900, and the
  views their agents get are screenshots of the desktop: run them with
  `--set desktop_resolution=1440x900` (their `task.toml` says so). Every other desktop task
  uses the base's 1024x768, which the default keeps.
- AgentsLastExam/ale#111 keeps Daytona sandbox names unique for long task names: before it,
  concurrent episodes of one task on Daytona collided and could delete each other's sandboxes.
- **RTX PRO 6000** (2026-11-10, the team's RTX PRO 6000 Blackwell Workstation Edition host, `bfb7505`;
  each task copied with `gpu_types` narrowed to that model, as in #71):
  - `hazard_bay_build`: `validate-0e73675f`, oracle 1.0. The render canary matched the RTX 5090's
    (RGB std 74.0 vs 73.1, 8852 hot thermal pixels, depth 0.762, lidar 1.0), and both missions ran
    with acceptance 0.956, as on the RTX 4090; verify 50 min. Now listed.
  - `robotic_arm_environment_twin_from_real_photo`: the setup's OVRTX warm-up (471 s of its
    1440 s) and two OVRTX renders of the oracle's twin from the agent's account (run `c5f4886c`,
    reward 1.0; bit-identical, 7 s each at 1920x1080 and 64 spp). Now listed, in `[environment]`.
  - `hot3d_sharpa_retarget` validates there too (`validate-24a8b005`, oracle 1.0, on a GPU shared
    with other jobs); it names no models.
  - Not listed: `driving_safety_supervisor` and `aerial_search_geolocation` (the agents' PyTorch
    has no sm_120 kernels, as on the RTX 5090), and `object_finding_and_organizing_on_messy_desk`
    (Isaac Sim 4.5 starts, but the RGB cameras come out about 30 % darker than on the RTX 4090, as
    on the H100).
- Next: run `hazard_bay_build`'s missions on an H100 on a quiet host, and add `H100` if they
  pass.

## Open items

- Website, display-only pull request: remove the backend (API, database service, sign-in,
  upload, review, admin), the leaderboard and the run pages; `/submit` links the two
  guidelines; the registry is validated against this repository's schema; install a fresh
  export made from a clean `main`. The production database volume is left untouched on the
  server.
- After this merges: export the registry from a clean `main`, commit `identity-history.json`
  (the spacecraft task is new; the `drone_hover` record stays as history), and install the
  export in the website.
- Review PR #10's hybrid-wake, telescoping-boom and encoder-mux hardening (§9). The measured
  GPT-6.1 Sol reward is 0.764 rather than the aspirational sub-0.5 target, but it no longer
  saturates and only docks 21--25% of hidden episodes. Any further reduction should improve
  the reference policy or add a new observable capability gap, not tighten penalties.
- Pull request #4 (`ball_physics_stepper`) must re-vendor the kit from #6 (CI's byte
  comparison), and should move to 64-bit, keyed-hash seeding if its agent holds a simulator
  copy.
- The guidelines' example link (a `task.yaml` on the `contrib/` branch, in the Google Doc)
  should point at the converted example's `task.toml` on `main`; the doc is the owner's.
- Engine (upstream) issues that skew agent runs:
  - the claude-code harness ends a model stream after 600 s without a byte (300 s before
    1cdbc02; the gateway's 15 s keep-alive covers silence inside a stream, not before its
    first byte);
  - the CLI caps a response at 64k tokens and the gateway ends a request after 1800 s in total
    (`gateway/server.py`), which leaves long-thinking models with empty turns;
  - egress is opened before verify while a closed_loop solver runs.

## 7. 2026-10-23 — messy-desk PR #12 diagram refinement

- Redrew the object and request automata as compact, color-coded Mermaid flowcharts
  in the task's scoring README and PR description. Kept state names, object values,
  outcome branches and request cycling visible; moved qualifications into captions.
- The object diagram is a lifecycle overview: transit, arrivals, disturbances and instance
  reuse are omitted explicitly. The request diagram groups the two successful outcomes
  and the incorrect-claim / expiry outcomes. No simulator or scoring behavior changed.
- Checked Mermaid parsing and rendered both diagrams locally; task lint passes.
- Follow-up: publish tightly cropped, opaque white PNGs with editable Mermaid sources
  to keep the backgrounds stable in dark mode. Keep the object layout; compact the
  request diagram vertically with Idle above the new-request transition and no Closed node.
- Moved transition labels beside the arrows, removed opaque label backgrounds, and
  reduced label emphasis so the arrow paths remain continuous and unobstructed.
- Validation remains pending: stabilize oracle grasping, measure its anchor, run
  `ale validate`, and measure frontier-model difficulty before marking the PR ready.

## 8. 2026-10-23 — first-target-hold request credit

- Request points award +1 for the first current target hold per request, topped up
  to 2 points by verified delivery; a correct absence declaration also earns 2 points. Repeated
  holds add nothing, and failure/expiry preserves only the first-hold credit already earned.
- R divides total points by twice the number of opened requests, including an ongoing one, so
  partial credit is immediate. The separate request success-rate diagnostic remains available.
- Updated the environment event, hidden/practice scorer copies, instructions, formula and
  diagrams. Historical contact is insufficient; the target must currently be in held state.
  The existing loss predicate remains centre z < -0.2 m; diagrams explicitly say falls off
  table. Releases onto the table or into a container are not loss events.
- Focused request-credit tests cover deduplication, completion, failure, absence, current
  holds and copy parity. Full Isaac rollout validation and anchor measurement remain pending.

## 9. 2026-10-23 — messy desk: dev-wire live score, exact depth, oracle grasping (Claude session)

- Practice tools: every `step`/`reset` reply of the simulator carries the running score (the same
  `EpisodeScorer`), `{"cmd": "score"}` adds per-object value/baseline, `{"cmd": "request", ...}`
  opens a request on demand; `practice_grader/grade.py --live N --score-file PATH`. Verified end
  to end with a dummy wire agent (obs keys as documented, untouched = 0, marks: claim_wrong /
  absent_correct / ignored without a request). Known gap: a request opened by the dev command
  is not counted by the dev scorer until the next `step` (its event rides the `request` reply).
- Simulator: `anti_aliasing: 0`. The default DLSS accumulated previous frames into the depth
  images: ghost depth under moving fingers and a skirt of floating points around every object,
  which inflated measured widths by ~2 cm and hid flat objects. `BASE_SCALE` master chef can
  0.62, tuna can 0.72 (both exceeded the 8 cm hand); instance factors 0.92 / 1.00.
- Oracle: containers are fitted from the static cameras only (the clipped hand-camera rim view
  biased the red container's fit and leaked its walls as loose clusters); `plan_grasp` = pinch,
  else rim grasp for hollow wide objects; `pick` re-centres from the hand camera 10 cm above the
  object; silhouette pixels dropped before back-projection. Grasp validation on single objects
  is running; refs.json must be rebuilt from `calib_collect2.py` output once grasps are stable.
- Frontier run decided: Opus 5.5, claude-code harness, subscription auth, 1 episode first.

## 10. 2026-10-23 — messy desk: scene physics fixes found while debugging the oracle (Claude session)

- Every YCB object used to spawn lying on its side: the stock `Axis_Aligned` meshes are authored
  with their height along local Y and `_spawn` only applied a yaw. Desk arrivals now stand
  upright (Rx(-90)); scissors, marker and both clamps (local Y = their length) lie flat. Desk
  spawns are set down 4 mm above the desk (`half_h` from the USD bound) instead of dropped.
- The stock table is flat only for |x| < 0.709 and then rises into a 1.2 cm rim, which tipped
  over objects that arrived at x > 0.71 and appeared as a 15 cm "object" strip in the hand camera.
  The room is now shifted +0.10 m in x (`ROOM_OFFSET_X`): the whole reachable desk is flat.
- Scales: every model offers a pinch of at most ~6.8-7.2 cm in any resting pose (cracker box
  0.42, sugar 0.73, pudding 0.63, gelatin 0.80, potted meat 0.70, mustard 0.71, foam brick 0.92,
  tomato soup can 0.92: at 6.8 cm the 8 cm hand kept only 6 mm per side and missed).
- Measured: the open fingertips stop on the desk at TCP z = 0.028 (TCP_Z_MIN unchanged); the
  visual finger mesh suggests 1.2 cm, the collision geometry is longer. Thin objects lying flat
  (marker, scissors) are pinched with the fingertips at the desk; a pinch along a long object
  wedges it between the fingers instead (lifted with the hand open).
- Desk arrivals point their long side away from the nearest container wall (+-0.6 rad): with the
  20 cm Franka hand, a long object lying along a wall could not be pinched in place (Monte Carlo:
  81 % -> 95 % pinchable).
- `sim_client`: a start attempt that hangs (Isaac's renderer occasionally deadlocks on the first
  frame of a fresh process, seen twice today) is killed and retried, 3 x 480 s.
- Instruction: draft banner removed; interpreters stated by absolute path (the claude-code harness
  hard-codes PATH without /opt/venv/bin); request example matches the env's text.
- Oracle: perception subtracts the desk profile; hand-camera component chosen by size x proximity;
  width check keeps a 14 mm margin (the hand-camera cloud loses ~7 mm per side); tall objects are
  grasped mid-body; rim grasps go 22 mm deep; the behaviour loop restarts after any exception.

## 11. 2026-10-23 — messy desk: arm motion, reach, piles, oracle search (Claude session)

Opus 5.5 frontier run in flight (`ale run ... --agent claude-code --model claude-opus-5-5 --set
agent.version=2.1.309`; the harness default 2.1.227 is rejected by the model). `ale run` uploads
the task's `verify/` folder from the local path when the verify stage starts, so `verify/` stays at
commit 2278f3f until that run is scored; the changes below live in `setup/payload/` and `oracle/`.

- Arm motion (env): each step's IK joint targets are interpolated over the 6 physics substeps.
  Applied at once, the stiff drives covered the delta in 1-2 substeps and the jerk flung held
  objects out of the fingers (12 of 42 test grasps dropped in transit, e.g. the sugar box slid
  3.7 cm during a 3 cm/step lift). A 5 cm delta now completes ~96 % within the step.
- Reach (env): the arm cannot reach down at r < ~0.25 m around |y| = 0.15 (IK refuses every
  step); arrivals keep 0.26 m from the base (was 0.22). At x = 0.19-0.22 a hand turned 90 degrees
  hits the container walls.
- Initial piles (env): objects are set down on top of the pile near the container's centre line
  and settled one by one (were thrown in from 0.25-0.65 m and rolled to the ends). A top-down
  Franka hand (0.205 m wide) fits inside a 0.28 m container only within +-2-3 cm of its centre
  line: 20 % of pile objects were reachable, now 36 % (1.3x longer containers would give 47 %,
  not worth shrinking the desk strips).
- `lab` with a null yaw sets an object down the way arrivals are (dev aid, task card updated).
- Oracle: references rebuilt for the current scales (161 static, 166 hand-camera views; the old
  LOO split was buggy: "_p" in "potted"/"pitcher" excluded whole models); hand-camera + 0.7 x
  static distances fused (LOO class 166/166); narrowest-pinch yaw; a pinch-position search for
  clamps/scissors; gentle placing near the centre line; container inventory (what it put where,
  which initial piles are still unseen); request search digs the reachable top object, delivers it
  straight to red when it is the target, and never claims absence while a container is blocked.
- Grasp test (42 arrival-like spawns, both instances of every model): 18/42 -> 33/42.
- Frontier run 1 (Opus 5.5, old env, `be35f322`): the agent was still debugging at the 4 h limit
  (569 tool calls) and the engine does not verify a timed-out agent phase (status `timeout`, no
  metrics). The instruction now states the hard 4 h limit and that a cut-off session is not graded.
  Its last submission is kept in the run's `artifacts/submission`.
- Domain lint: the practice grader's live line no longer reads a `"score"` key (the lint's
  leaky-output pattern); only the two expected null-anchor problems remain.

## Gotchas

- `export_registry.py` refuses a dirty tree on purpose: `content_digest` covers every
  regular file, ignored `__pycache__/` included. Export from a clean commit; `--allow-dirty`
  is for throwaway exports and never writes the canonical `identity-history.json`.
- The template and the worked example each validate in well under a minute.
- Engine commands run from `vendor/ale` with `VIRTUAL_ENV=` cleared and
  `DOCKER_HOST=unix:///var/run/docker.sock` pinned (two docker contexts on the dev machine).
  `uvx ruff`, not `uv run ruff` (ruff is not a project dependency).
- `uv venv --python 3.12` puts its managed interpreter under `/root/.local/share/uv`, which
  the agent account cannot traverse; task images set `UV_PYTHON_INSTALL_DIR=/opt/uv-python`
  and keep the Dockerfile's agent-UID import self-check. The agent UID must traverse (not
  only read) everything the venv points at.
- The engine stages `ale_verify` into the image's `python3`, else `python3.12` (3.12+), else
  its own Python, which lacks the task's packages, and names it in `ALE_VERIFY_PYTHON`
  (sections 8 and 11).
- `ale validate` does not keep a stage's stderr; it is in the episode's
  `trace.execution.jsonl`.
- A null anchor is not an error: the engine fails the run with `oracle_not_full` and
  `harness/ale_onboard.py` reads that as `needs_anchor` with the measured value.
- A task that hands the agent a copy of its simulator must seed episodes with 64 bits and
  draw hidden values through a keyed hash; an LCG or a 31-bit seed gives the episode away.

### 2026-10-23 — provisional anchor
- Oracle a5c3982 measured through the task's verify stage (task image, network off): episode 0 S = 0.641 (O 0.72, R 0.56); episodes 1–2 running. Anchor recorded as 0.641058, full_at 0.8 (template default), marked provisional.
- Opus 5.5 run 1 (old env build, timed out): submission graded offline on practice seed 11, 20000 steps: S = 0.353 (O 0.61, R 0.10). Run 2 on the current build in progress.
- Open: full_at from the 3-episode spread; official `ale validate` once the GPU is free.
- Agent budget lowered 4 h → 1 h (template default; the 4 h was a scaffold-time choice, not a benchmark requirement). Opus runs 1–2 are kept and reported as extended trials with a 4 h budget. Official `ale validate` queued to start when run 2 releases the GPU.
- 2026-10-23 (owner decisions): four requests per episode (one per 5000-step window, `3a698f5`); O scores the episode's final state, and mark 3 (*done: episode*) ends the episode once its last request has closed (`a1321ca`; earlier done marks are ignored so all four requests count in R). The oracle sends mark 3 after its fourth request when nothing is left to sort. Anchor must be re-measured on this build (old-env oracle: 0.641 / 0.436 / 0.482, per-episode sd ~0.11). Validate is gated on a new anchor. Oracle pick work in progress (hard objects: mustard, bleach, clamps, scissors, pitcher, tuna).
- Superseded the final-state score the same day (`d266c95`, owner): each request is a find-and-organize task judged at its close; O = mean of organization snapshots at every request close plus the last step; no episode-level done mark. While a request is open its object belongs to red only (+1 in red or held, -1 in any other container, 0 on the table); its baseline is re-read at the opening and restored the step after the close. Oracle: after the target is in red (or absence is settled) it puts back what it dug out and sorts loose objects, then marks (FINISH_MARGIN 350 steps before the deadline).
- 2026-10-23 evening: `ale validate` passed (validate-1c441e93): untouched 0, oracle 1.0; oracle measured 0.434 (0.419 / 0.539 / 0.344) against the provisional anchor 0.520 (full_at 0.6). Anchor set to 0.4341, full_at 0.75 (`30b3ff7`). Opus run 2 (4 h extended trial) graded offline on the current grader: 0.555 (0.629 / 0.429 / 0.607) -> reward 1.0, so the task is too easy against this oracle. Oracle strengthened with 3D hand-fit checks and tilted grasps near container walls (`c788106`): seed 23 diagnostic 0.486 -> 0.625. Running: Opus run 3 on the 1 h budget (started 19:31 UTC); oracle re-measure mv_c (3 episodes) for the anchor; then a confirmation validate on the final anchor.
- 2026-10-23 night: oracle c788106 re-measured through the verify stage on 3 hidden seeds: 0.4219 (0.163 / 0.507 / 0.596; 2 of 12 requests won, 7 expired). The tilted grasps did not move the hidden-seed mean (earlier oracle 0.434). Pooled per-episode sd ~0.16 over six oracle episodes, so a 3-episode mean varies by ~22% of the anchor: anchor set to 0.4219 with full_at 0.6 (`f93a8f9`, chosen before run 3's score was read). Opus 5.5 run 3 on the 1 h budget: agent ended after 22 min; mean_episode_score 0.200 (0.162 / 0.277 / 0.161; episode 1 cut by the 45 min wall budget), engine reward 0.615 against the old anchor, 0.79 against the new one. It never held a requested object (0 first holds, 8 wrong absence claims of 12). The grader now prints each episode's full seed and request log after the episode ends (stderr only), so a failed hidden episode can be replayed; the env's sim.log keeps only 16 bits of the seed, so the oracle's 0.163 episode cannot be replayed. Confirmation `ale validate` on f93a8f9 running (frozen copy).
- 2026-10-24 00:00 UTC: confirmation `ale validate` on f93a8f9 passed (validate-b76c111b): untouched 0.0, oracle 1.0 (mean_episode_score 0.426 = 0.315 / 0.584 / 0.379 against the anchor 0.4219, ratio 1.01). The grader's new per-episode line shows two of its twelve requests were ABSENT requests that expired: the oracle never claims absence while a container it could not dig out may still hide the object, which under the current scoring costs the same as a wrong claim. Oracle failure census on four visible seeds (281 grasps): desk grasps lift 70%, in-container grasps 24% (geometric ceiling with the Franka hand 20.5 cm long in a 28 x 22 cm interior: 32% of the floor upright, 52% with tilts); 11 of 16 requests lost (5 grasp execution, 3 no feasible grasp, 2 perception misses, 2 decision logic). Owner decision: enlarge the containers to interior 36 x 24 cm (CONTAINER_SCALE 1.43 x 1.375 x 1.30, nominal x 0.41; reachable floor 73% with tilts). Staged in the scratch copy, reach test and probe episodes to follow; anchor, Opus 1 h run, task card, instruction, README and hero images must then be redone. Dev oracle (A/B on seeds 31/58/77/90 running): grasp height from the top right under the grasp point, re-centring around the anchor only, delivered mark when the target went into red unconfirmed; next: mark absent instead of expiring.
- 2026-10-24 night (`ccab991`): the oracle's request success was capped by geometry: the Franka hand body (20.5 cm wide, 6.2 cm above the fingertips) fits a 28 x 24 cm container only within +-3.4 / +-1.35 cm of the centre line for low objects; tilted grasps reached their plan within 0.5 cm yet lifted 2 of 16. Owner decisions: red delivery container right in front of the base (0.25, 0; 0.237 x 0.272 m), the three class containers behind it (gray (0.44, -0.31), lightblue (0.575, 0), blue (0.44, 0.31); 0.326 x 0.272 m), walls 0.19 -> 0.12 m, a fourth task camera `oblique` (the scene image's viewpoint), and hi-res frames on request (optional ninth action element, camera bitmask; 1920x1080 RGB-D of those cameras in the same observation; the hi-res render products run every frame, about 9 ms per step, and are only encoded when asked: about 0.1 s and 3 MB per camera; switching an Isaac render product off and on again stops it for good, so they stay on). Free arrival spots 174 -> 406, all class-container floor within reach. Arrivals never spawn the open request's model. Oracle v3: mark 20 steps before the deadline whatever the arm is doing, stuck-arm recovery then give the object up, loose objects only within r 0.40-0.76, no grasp retries (0 of 23 lifted), clamps told apart by length, failed spots and stalled containers move the dig on, tilted grasps compensate the fingertip offset, carry height 0.38. Four visible seeds (31/58/77/90) on the new layout: 0.572 / 0.627 / 0.623 / 0.416 (mean 0.56), 8 of 16 requests won (was 6 of 16), container grasps 41% lifted (was 24%); seed 90 lost its second half to a loose clamp pushed to r = 0.80, hence the 0.76 limit. Every probe episode now records a composed video and one clip per request (host folder `video/`, git-ignored). Running: hidden-seed oracle measurement on the frozen copy of `ccab991` (3 episodes); then anchor, `ale validate`, Opus 1 h, PR numbers.
