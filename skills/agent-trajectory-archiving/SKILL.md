---
name: agent-trajectory-archiving
description: Record, select, export, secret-scan and upload the trajectories of model runs (`ale run` with a real agent) on this benchmark's tasks with scripts/trajectories.py. One trajectory per task and agent (the task's latest version, its lowest-scoring episode); each kept folder holds the episode as the engine wrote it plus run.json (summary, score, tokens and cost, the task text, a per-step digest), trajectory.jsonl.gz, steps.bin (the pictures the agent saw), a poster and optional replays, under a root index. Every path, remote and run directory is a parameter. Use it right after every model run.
---

# Agent trajectory archiving

A model run of a task (`ale run` with a real agent, the difficulty test of
`docs/contributing-a-task.md` §7) leaves one folder per episode under the run's `--runs-dir`. This
skill turns those folders into an **archive**: one trajectory per task and agent, each readable
without the engine, with a root index. Everything goes through one script,
`scripts/trajectories.py` (stdlib only; Pillow, ffmpeg and rclone are used when present):

| subcommand | does |
|---|---|
| `inventory` | every model episode under a runs directory, one JSON line each |
| `curate` | one trajectory per task and agent, written to `index.csv` |
| `export` | the kept episodes into the archive, plus `run.json` and the derived files and the root index |
| `scan` | configured secrets and token patterns; exit status 1 on a hit |
| `upload` | rclone copy of the archive to any destination, scanned first, checked after |
| `sample` | resource samples of one run's own episode containers, started beside `ale run` |

`python3 scripts/trajectories.py <subcommand> --help` lists every option.

## 1. Parameters: nothing is fixed

| what | flag | environment variable |
|---|---|---|
| the runs directory (`ale run --runs-dir`) | `--runs-dir` | `ARB_RUNS_DIR` |
| the archive (a local folder) | `--archive` | `ARB_TRAJECTORIES` |
| the upload destination: any rclone destination, `remote:path` or a local path | `--dest` | `ARB_TRAJECTORIES_DEST` |
| files whose secret values the scan looks for (the engine's `vendor/ale/.env`, an `rclone.conf`) | `--secrets-from` | `ARB_SECRET_FILES` (separated by `:`) |
| environment variables whose values the scan looks for | `--secret-env` | |
| the benchmark checkout that resolves task commits, anchors and platforms | `--repo` | (this checkout) |
| list prices, USD per million tokens | `--prices` | (built-in table) |
| replays, stills and robot rules | `--extras` | |

The rclone remote (a cloud drive, a bucket, a server) is set up with `rclone config` on the
machine that uploads; its credentials stay in that machine's `rclone.conf`. Never commit an
archive, a run or a credential to this repository: `runs/` and `*.mp4` are ignored, and a
trajectory carries the agent's whole session.

## 2. Before the run

- **Run configuration** (`ale run --config run.toml`): `[artifacts] collect = "host"` keeps the
  submission in the episode's `artifacts/`, so the archive holds what the grader scored.
  `[logging] native_logs = "debug"` keeps the harness's native logs even after a clean conversion
  (with the default `"minimal"` the engine deletes them then; it always keeps them when the
  conversion fails, which is when the export needs them).
- **Resource sampler**: the engine records no resource peaks. Start `sample` beside `ale run`
  and stop it by its pid (never by a pattern):

  ```bash
  RUNS=/abs/path/to/runs RUN_ID=<run id>
  python3 scripts/trajectories.py sample --run-dir "$RUNS/$RUN_ID" --parent $$ >> "sampler-$RUN_ID.log" 2>&1 &
  SAMPLER=$!
  (cd vendor/ale && VIRTUAL_ENV= DOCKER_HOST=unix:///var/run/docker.sock uv run ale run ../../tasks/<platform>/<slug> \
      --agent claude-code --model <model> --config run.toml --runs-dir "$RUNS" --run-id "$RUN_ID")
  kill "$SAMPLER"
  ```

  Every 10 s it records docker stats (memory, CPU, processes, network and block I/O) of the
  containers whose `ale.episode` label names an episode folder of this run, their writable layer
  every 60 s, and nvidia-smi for the GPUs they lease, into `<run dir>/resources.jsonl`. It is
  read-only: it never starts, stops, execs into or changes a container, and it never queries
  another run's containers. `--parent` makes it exit with the launcher.

## 3. After the run

```bash
python3 scripts/trajectories.py inventory --runs-dir "$RUNS" --out inventory.jsonl
python3 scripts/trajectories.py curate inventory.jsonl --out index.csv         # --task, --overrides: section 4
python3 scripts/trajectories.py export --runs-dir "$RUNS" --index index.csv --archive "$ARCHIVE"
python3 scripts/trajectories.py scan "$ARCHIVE" --secrets-from vendor/ale/.env --context 60
python3 scripts/trajectories.py upload --archive "$ARCHIVE" --dest "$DEST" --secrets-from vendor/ale/.env
```

`export --only <task> ...` and `upload --only <task> ...` redo some tasks; `export` always rebuilds
the root index from every exported folder, `upload --only` leaves the destination's index alone.
`upload --prune` then removes destination folders `index.csv` no longer lists (a task's older
version); `--dry-run` shows what it would copy and remove. Tell whoever ran the task the folder,
the reward, the agent's time and the cost.

The export reads files and writes only inside the archive; it starts no container and uses no GPU.
On a shared machine run it on the cores you were given (`taskset -c <cores>`).

## 4. Which trajectory is kept (`curate`)

- **Tasks that count**: the folders `tasks/<platform>/<task>/` of the checkout (`--tasks-dir`),
  plus `--task NAME` for a task in an open pull request; `--any-task` keeps every task.
- **Model episodes only**: validation runs (a run-level `validation.json`; `ale validate` names
  them `validate-*`) and the engine's own `oracle` and `nop` harnesses are not agent
  trajectories; `inventory --all` lists them with a `skip` reason.
- **The latest version**: the content digest (`lock.json` `task.content_digest`) of the task's most
  recently started episode. Episodes of earlier versions are not kept.
- **One per task and agent** (`<harness>_<model>_effort-<effort>`, from `lock.json` `agent`): of
  that version, the **lowest-scoring** episode (`--pick highest` or `latest` otherwise; equal
  scores: the most recent). Episodes that ended on the agent's side (`completed`, `timeout`,
  `budget_exceeded`, `refused`) come first; one that ended in `env_error` or `agent_error` counts
  only when no other has a score, and one without a score only when none has one.
- **Overrides** (`--overrides FILE`, JSON; the flags `--task`, `--rename`, `--exclude-run` do the
  same for single values):

  ```json
  {
    "tasks": ["<task in an open pull request>"],
    "rename": {"<earlier name>": "<current name>"},
    "exclude_runs": ["<run id another run superseded>"],
    "regrades": {"<run id> or <run id>/<episode id>": {"verification": "<path>", "score": 0.5, "note": "why"}},
    "notes": {"<run id>": "a note for the index"}
  }
  ```

  `rename`: a task measured under an earlier name goes under its current name (the folder is
  renamed, the records stay as the engine wrote them, the note names the earlier name).
  `regrades`: the same submission scored again (for example after a grader fix): its score
  replaces the engine's, `verification` (relative to the episode folder, or absolute) is the
  regrade's result file and goes into the archive, and the note says why. A key without
  `/<episode id>` applies to every episode of the run.
- A round is **one episode per task**: run each model once, not three times.

`index.csv` columns: `task, agent, score, status, run_date, task_commit, run, episode, regrade,
notes, folder`.

## 5. The archive

```
<archive>/
  index.csv                 the selection (curate)
  index.json                counts per platform and run setting, the shards, tasks without a replay
  index/<platform>.json     one row per kept trajectory
  <task>/<harness>_<model>_effort-<effort>/<run id>__<episode id>/
      the episode folder as the engine wrote it (section 6), and
      run.json  trajectory.jsonl.gz  steps.bin  pics/  poster.jpg  resources.jsonl  replay.mp4
```

`export --derived-only` writes only the derived files (when the episode's own records are
uploaded another way); `--link` hard-links the episode's files instead of copying them;
`--prune` removes archive folders `index.csv` no longer lists. The archive and the runs
directory must not contain each other.

## 6. What the engine keeps per episode

Spec: `vendor/ale/docs/specs/trace.md`. The archive keeps the folder as written (symlinks are
kept as links, never followed; `__pycache__/` is left out):

| file | holds |
|---|---|
| `trajectory.json` | Harbor ATIF v1.8: the instruction, model messages and exposed reasoning, tool calls and results, pictures as media parts whose `source.path` points into `blobs/`, subagents |
| `blobs/` | pictures (screenshots of GUI tasks), audio and large tool outputs the trajectory references |
| `trace.transport.jsonl` | every gateway model call: model, tokens, cache reads and writes, latency, retries, refusals |
| `trace.execution.jsonl` | engine phases, setup and verify commands with their output (the grader's log is the verify phase's `command_finished` stderr) |
| `result.json` | status, rewards, failure, phase timings, the native session id |
| `verification.json` | the grader's full result (every metric, `metrics.reward_exact`, the diagnostic) |
| `lock.json` | provenance: task commit and content digest, image, engine commit, harness, model, effort, limits |
| `logs/` | the harness's native logs when kept (Claude Code: `logs/claude-code/.claude-config/projects/*/<session>.jsonl`, and `.ale-harness/stdout.log`, whose result line has its own cost figure) |
| `artifacts/` | the collected submission (`[artifacts] collect = "host"`) |
| `attempts/<n>/` | earlier attempts a retry replaced, complete |

Failed episodes (`agent_error`, `env_error`, `timeout`) keep their status in `result.json` and
are kept like the others. GUI tasks keep their screenshots in `blobs/`; when a task's visual
evidence lives elsewhere (a screen recording, a screenshot folder in the submission), it rides in
the same folder through `artifacts/`.

## 7. `run.json` (format 3)

One document per kept episode, next to the engine's records.

| field | holds | from |
|---|---|---|
| `format` | 3, the version of this layout | |
| `slug` | `<task>--<episode suffix>`, e.g. `aerial_search_geolocation--4f730246` | episode id |
| `task`, `bench` | the task's current name, its platform key (`tasks/platforms.json`) | `index.csv`, `lock.json` `task.source.path`, else the checkout |
| `mode` | `lock.json` `agent.family` (`autonomous`): the benchmark has one setting per task | |
| `run`, `agent`, `model`, `effort` | e.g. `claude_code-opus_5_5-high`, `claude-code`, `claude-opus-5-5`, `high` | `lock.json` `agent` |
| `state`, `success` | `success` = 1 when the ratio reaches the task's `full_at` (reward 1.0), else 0 | grader metrics, `tests/anchor.json` |
| `ended_by` | `infra` for an `env_error` (or an `agent_error` on a refused connection), else null | `result.json` |
| `reward` | every grader metric, `reward` (rounded half-up to two decimals, what the engine reads), `reward_exact`, `success`, `infra_error`, every named reward of the result, and the resource fields `peak_memory_gb`, `peak_gpu_memory_gb`, `peak_processes`, `peak_disk_gb` (null and `resource_monitor_missing` 1 without samples) | `verification.json`, a regrade's record |
| `wall_s` | the agent phase, seconds | `result.json` phases |
| `tokens` | `in` (cache reads and writes included), `cached`, `out`, `usd`, `cache_write`, `uncached_in`, `calls`, `failed_calls`, `replayed_calls`, `unlinked_calls` and `unlinked_usd` (calls no step links to, such as context compaction), `unpriced_calls`, `models`, the price used, `usd_other` (the harness's own cost figure); judge calls of the verifier apart under `judge` | `trace.transport.jsonl`, the harness's result line |
| `task_text` | the instruction the agent got | `trajectory.json` step 1 (a blob pointer resolved) |
| `replay`, `poster` | the replay's file name or null; the poster's file name | section 9 |
| `steps_bin` | `{file, size, mime, count, mimes}` of `steps.bin` | the pictures the agent saw |
| `trajectory` | `{file, size, bytes, lines, source, layout}` of `trajectory.jsonl.gz` | `trajectory.json`, or the native session log when the conversion failed |
| `steps[]` | one per model call: `id` (its first ATIF step), `t` (seconds since the agent phase began), `say` (the agent's text, in full), `calls[]`, `out`, `tok`, `sim`, `from` (agent, user, system, tool-program), `atif` (the ATIF step ids), `at` | ATIF steps merged per call |
| `calls[]` | `kind` (`robot`, `look`, `edit`, `read`, `shell`, `web`, `other`), `label` (the tool's description, a file name, `server: tool`), `code` (the command or arguments, over 6,000 characters cut in the middle), `tool`, `id`, `exit`, `error` | tool calls |
| `out` | `text` (the results, over 3,000 characters cut in the middle; the full text stays in the trajectory), `exit` (known exit codes), `pics` (`[offset, length, mime]` ranges of `steps.bin`), `pic_files`, `chars`, `errors` | tool results |
| `tok` | that call's `in`, `cached`, `out`, `usd`, `cache_write`, `latency_ms`, `stop_reason`; `think` null (thinking tokens are inside `out`) | transport trace |
| `sim`, `sim_total` | simulator submit and reset counts, only for a task's MCP robot tools; null where the agent's simulator commands run inside its own programs | |
| `step_times` | `recorded` | |
| `ale` | run and episode id, status, failure, start and end, engine commit and version, harness version, native session, every phase with its duration | `result.json`, `lock.json` |
| `score` | `metric`, `direction`, `measured`, `anchor`, `full_at`, `cap`, `ratio`, the rule, `anchor_source`, the grader's `diagnostic` and `criteria`, the tail of the grader's log, the `regrade` and the engine's own score when one replaced it | `verification.json`, `tests/anchor.json` at the task commit |
| `task_info` | task commit, path, the name it was measured under, content digest, `site` (`<platform>/<task>`) | `lock.json` |
| `notes`, `folder`, `files` | the index notes, the folder's path in the archive, what each file in the folder is | |
| `replays`, `poster_info` | what each replay shows and where it came from; where the poster came from | `--extras` |
| `resources` | limits (cpus, memory, GPUs, network, agent and verify time limits of the version that ran); peaks and the sample count when sampled, else the reason | `lock.json`, `task.toml`, `resources.jsonl` |
| `digest` | its source, step count, kinds, the robot rule and the agent's own scripts that counted as robot, the truncation rule | |
| `pictures` | every picture once: sha256, size, mime, width and height, its file, its byte range in `steps.bin`, its source record, the steps that show it | |
| `provenance` | the whole `lock.json` | |
| `format_note`, `exported` | where this guide is; when and by which commit of the script the file was written | |

`trajectory.jsonl.gz`: line 1 `{"atif_header": ...}` (the ATIF document without its steps), then
one line per root ATIF step as written, then one line per subagent
`{"subagent_trajectory": ...}`; large texts and pictures stay in `blobs/`, referenced by path.
When the conversion failed it is the native session log, verbatim.

## 8. The index

`index.json`: `v` 3, `total`, `suites` (trajectories per platform), `solved` (successes per
platform), `settings` (trajectories per `run` label), `shards` (`index/<platform>.json`),
`pending_replays` (tasks without a replay), `layout`, `generated_at`, `generated_by`.
`index/<platform>.json`: `{"suite": <platform>, "runs": [...]}`, one row per trajectory: `slug, task,
bench, run, model, agent, effort, success, state, wall_s, steps, kinds, usd, poster, site`,
`ended_by` when set, and `reward, ratio, task_commit, task_digest, notes, folder, replay, status,
ale_run, episode, date, metric, anchor, full_at, pictures, tokens_in, tokens_out`. A row's
`poster` and `replay` are relative to its `folder`.

## 9. Rules the export follows

- **Cost** = list price per token type times the transport trace's tokens: input, output, cache
  reads, cache writes (at the 5-minute rate; the trace does not split writes by TTL). The built-in
  table holds Anthropic's first-party prices of October 2026; `--prices FILE` replaces or adds
  models (`{"<model>": {"input": .., "output": .., "cache_read": .., "cache_write_5m": ..,
  "cache_write_1h": ..}}`). Partner platforms (Amazon Bedrock, Vertex AI) bill separately. A call
  the trace recorded without cache figures is not priced (`unpriced_calls`). The harness's own
  figure (Claude Code's `total_cost_usd`) is kept in `tokens.usd_other` and never used for `usd`:
  it prices with the harness's own table, which can map a model to another model's prices (the
  note names the model it used).
- **Pictures**: from `blobs/image/`, from a text blob holding a picture as base64 JSON, or from
  the native log; the last two are also decoded into `pics/` so every picture has a file.
  `steps.bin` holds each distinct picture once, in its original bytes.
- **Conversion failed** (`trajectory.json` `extra.ale.incomplete_reason` `conversion_failed`):
  the digest is built from the native session log
  (`logs/claude-code/.claude-config/projects/*/<session>.jsonl`) and `trajectory.jsonl.gz` is that
  log verbatim.
- **Poster**: the replay's frame at `poster_at` when a replay has one, else the last picture the
  agent saw, else the task's still from `--extras` (described in `poster_info`); converted to
  JPEG with Pillow or ffmpeg when present, else kept in its own format.
- **Replays**: only the kept episode's **own submission**, replayed by the task's own episode
  runner on **development episodes, never hidden (graded) ones**. They are given in the extras
  file and copied into the folder:

  ```json
  {
    "replays": {"<task>": [{"file": "replay.mp4", "local": "<path>", "what": "what it shows",
                            "poster_at": 12.5, "source": "where it came from", "readme": "<notes file>"}]},
    "stills": {"<task>": {"local": "<path>", "what": "what it shows", "source": "where it came from"}},
    "robot": {"<task>": "<regex>"}
  }
  ```

  Relative paths are relative to the extras file. Tasks without a replay are listed in
  `index.json` `pending_replays`.
- **robot** = the task's own simulator, robot, CAD or practice tools: its MCP servers (viewer tools
  are `look`), and shell commands that run an interpreter on code naming them, or on a script the
  agent wrote that uses them; a script the agent never showed counts as `shell`. The rule per task
  is `ROBOT` in the script; a task without an entry uses the names in its
  `environment/setup/payload/` (top-level folders and the Python and shell files in them);
  `--robot TASK=REGEX` or the extras file's `robot` replace an entry. A new task with tools the
  payload does not name (an application in the image, say) needs an entry.
- **Resources**: the `peak_*` fields come from the run's `resources.jsonl`, and the episode's
  samples are copied into the folder; without samples they are null with `resources.reason`.

## 10. The secret scan

The engine redacts the credentials it knows before anything reaches disk and keeps the model
credential out of the sandbox (`lock.json` `credential_exposed_to_agent: false`); the scan is the
second line. It searches every file (inside `.gz` too) for:

- **configured secrets**: in each `--secrets-from` file (dotenv, ini, `rclone.conf`), every value
  of 16 characters or more whose key names a key, token, secret, password or credential, and the
  string values of such keys inside a JSON value; and the value of each `--secret-env` variable;
- **token patterns**: AWS access keys and `aws_secret_access_key`, Anthropic and OpenAI keys,
  GitHub and Hugging Face tokens, Google API keys and OAuth tokens, Slack tokens, private key
  blocks.

Pictures, videos and `steps.bin` are skipped (`--include-media` scans them too): compressed bytes
match patterns by chance. A hit prints the file, the kind and the byte offset, never the secret;
`--context N` prints N bytes around it with every secret redacted. `upload` refuses to start on
any hit. A hit that is not a secret (a pattern inside unrelated text) is passed with
`--accept <path relative to the archive>` after you looked at it. A real secret is never
uploaded: drop the episode from the selection (`--exclude-run`), have the secret rotated, and say
so.

## 11. After every model run

1. When the episode finishes: `inventory`, `curate`, `export`, `scan`, `upload` (section 3), with
   `--prune` once a task's newer version replaced an older one.
2. Launch runs with the sampler beside them (section 2); a launcher of your own does the same.
3. GUI tasks: check that the screenshots are in `blobs/` and referenced by `trajectory.json`.
4. Delete a run folder only after its upload was checked (`upload` runs `rclone check` per
   folder).
