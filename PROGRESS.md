# PROGRESS.md · progress log for agentic-robotics-benchmark

> **What this is:** the progress source of truth for this repository, maintained by people
> and agents alike. It records milestone state, the last stopping point, known gotchas and the
> next step. **Disk is the truth** (git + this file + `CLAUDE.md`); a session's context is a
> volatile cache. Read this file first; update it (state / next step / gotchas) and commit
> before every hand-off. Where it disagrees with the code or `git log`, the code wins: fix
> this file.
>
> Last updated: 2026-11-03 (§8: the tasks move to the Harbor-based engine, `vendor/ale` @
> `1cdbc02`). Before that: 2026-10-21 (§7: task assets: `task-assets.toml`; §9: PR #10
> hardens the spacecraft task with hybrid wake cells, a telescoping port and an
> encoder-polarity mux; fresh GPT-6.1 Sol no longer saturates it).

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
`grounded-intelligence-initiative/agentic-robotics-last-exam` (first used by the open pull
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
- The engine stages `ale_verify` into `python3`, else `python3.12` (3.12+ required), and
  names it in `ALE_VERIFY_PYTHON` (section 8).
- `ale validate` does not keep a stage's stderr; it is in the episode's
  `trace.execution.jsonl`.
- A null anchor is not an error: the engine fails the run with `oracle_not_full` and
  `harness/ale_onboard.py` reads that as `needs_anchor` with the measured value.
- A task that hands the agent a copy of its simulator must seed episodes with 64 bits and
  draw hidden values through a keyed hash; an LCG or a 31-bit seed gives the episode away.
