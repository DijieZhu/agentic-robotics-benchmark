# Task assets: large agent-visible files

Photo sets, document scans, meshes and other inputs that are too large for git live in the benchmark's Hugging Face
dataset. A task fetches them while its image is built, pinned to one dataset commit and checked file by file. The
dataset is named in exactly one place, [`task-assets.toml`](../task-assets.toml) at the repository root, so moving
the assets (the dataset renamed, a mirror) is one edit and one command (section 4).

The dataset is [`grounded-intelligence-initiative/ale-robotics-assets`](https://huggingface.co/datasets/grounded-intelligence-initiative/ale-robotics-assets).
It was renamed from `grounded-intelligence-initiative/agentic-robotics-last-exam` on 2026-11-07; the Hub redirects the
old name, but every task now names the new one.

Only agent-visible material goes there: the dataset is public and its files end up in the image the agent runs in.
The grader's ground truth and the oracle's reference artifact stay in the task's `tests/` and `solution/` in git
(they reach the sandbox only when their stage runs; see [`security-model.md`](security-model.md)).

## 1. The pieces

| file | what it is |
|---|---|
| `task-assets.toml` | the asset repository: `[huggingface] dataset` (`<owner>/<name>`) and `endpoint` (the Hub, or a mirror serving the same `/resolve/` paths) |
| `shared/task_assets/fetch_assets.py` | the canonical download script (stdlib only) |
| `tasks/<platform>/<name>/environment/fetch_assets.py` | a byte-identical copy (the image's build context is the task's `environment/` folder, less `setup/`) |
| `tasks/<platform>/<name>/environment/assets.sha256` | the task's asset list: a header, then one `<sha256>  <relative path>` line per file |
| `tasks/<platform>/<name>/tests/fetch_assets.py`, `tests/assets.sha256` | the same two files for a separate verifier image that fetches assets too (a task with a `tests/Dockerfile`, whose build context is `tests/`) |
| `scripts/task_assets.py` | `sync`, `check`, `list`, `url` (below) |

An asset list looks like this:

```text
# endpoint: https://huggingface.co
# repo: grounded-intelligence-initiative/ale-robotics-assets
# revision: d2a07b8c123c6cf0d500d1c8737802c50e3240c9
# prefix: tasks/arm/robotic_arm_replica_requirements_from_photos/payload
cf8cb99a7adf0631aff2d5f99cb966cf91c0c8d78c363c09f54eccd09104ec3e  photos/PH-01.jpg
...
```

- `endpoint` and `repo` are copied from `task-assets.toml` by `sync`. Never edit them by hand; CI fails a task whose
  lines differ from the file.
- `revision` is the dataset commit the task was validated with. A task never follows a branch, so later uploads for
  other tasks never change it.
- `prefix` is the task's folder in the dataset, always `tasks/<platform>/<name>/payload`.

The dataset mirrors the `tasks/` tree: `tasks/<platform>/<name>/payload/` holds the files with the paths they get in
the agent's home, and `shared/` holds assets several tasks use. Its README is the dataset card and lists the tasks.

## 2. Wiring a task

In `environment/Dockerfile`, after the toolchain layers:

```dockerfile
# ---- task assets: fetched at the dataset commit pinned in assets.sha256, checked file by file (docs/task-assets.md)
COPY fetch_assets.py assets.sha256 /opt/task-assets/
RUN python3 /opt/task-assets/fetch_assets.py /opt/task-assets/assets.sha256 /opt/task-assets/payload \
    && chmod -R a+rX /opt/task-assets
```

In `environment/setup/setup.sh`, stage them into the agent home next to `environment/setup/payload/`:

```bash
if [ -d /opt/task-assets/payload ]; then
  cp -r /opt/task-assets/payload/. "$HOME_DIR/"
  for entry in /opt/task-assets/payload/*; do chown -R "$AGENT_USER:$AGENT_USER" "$HOME_DIR/$(basename "$entry")"; done
fi
```

The files are not in git: keep them out of `environment/setup/payload/` (an `environment/setup/.gitignore` helps
while you author).

Files the agent home never gets (a simulator release, a scene archive, model weights, mirrored byte for byte from
their sources) are fetched by the image step that unpacks or moves them, into a scratch folder that the same step
removes, so no layer keeps the download. A step that needs only the files under one folder of the list fetches a
filtered copy of it, for example
`grep -E '^(#|[0-9a-f]{64}  carla/)' assets.sha256 > carla.sha256` (`tasks/driving/driving_safety_supervisor` and
`tasks/uav/aerial_search_geolocation` work this way).

## 3. Adding or updating a task's assets

1. Put the files in a local folder with the paths the agent should see (for example `upload/photos/photo_01.jpg`).
2. Upload that folder to the dataset under the task's prefix (a maintainer of the dataset's organisation, or a pull
   request on the dataset):

   ```bash
   hf upload grounded-intelligence-initiative/ale-robotics-assets upload \
       tasks/<platform>/<name>/payload --repo-type dataset --commit-message "<name>: payload"
   ```

   Add the task's row to the dataset card's task table in the same or a following commit. Note the commit hash the
   upload prints (or read it on the dataset's history page).
3. Write the task's asset list from the same local folder, pinned at that commit. This also vendors the fetcher:

   ```bash
   python3 scripts/task_assets.py list tasks/<platform>/<name> upload --revision <40-hex commit>
   python3 scripts/task_assets.py url tasks/<platform>/<name>      # where the image will fetch from
   ```

   A separate verifier image that fetches files too gets its own list in `tests/`: the same `list` command with
   `--context tests`.

4. Rebuild and validate (`ale validate`): the image build downloads every file and fails on any hash mismatch.

Replacing a file is the same procedure: upload, re-run `list` with the new commit, validate.

## 4. Moving every task's assets (a renamed dataset, a mirror)

1. Rename the dataset on Hugging Face, or set up the mirror. A Hub rename keeps the repository's history, so every
   pinned `revision` stays valid (the Hub also redirects the old name, but tasks do not rely on that).
2. Edit `task-assets.toml`: `dataset` (and `endpoint` for a mirror).
3. Run the sync and commit the result together with the configuration change:

   ```bash
   python3 scripts/task_assets.py sync     # rewrites endpoint / repo in every asset list
   python3 scripts/task_assets.py check    # what CI runs
   ```

Nothing else changes: file lists, hashes and pinned commits stay as they are. Because each task's `assets.sha256` is
part of its content, every task with assets gets a new content digest. Its image is rebuilt on the next run, and the
maintainer re-exports the registry after the merge (`identity-history.json`, as after any task change). Re-validate one
task to confirm the new location serves the files.

## 5. What CI checks

`python3 scripts/task_assets.py check` runs on every pull request (`.github/workflows/pr-checks.yml`). It fails when:

- a task's `endpoint` or `repo` line differs from `task-assets.toml`;
- its `environment/fetch_assets.py` is missing or differs from `shared/task_assets/fetch_assets.py`;
- the same for a separate verifier image's `tests/assets.sha256` and `tests/fetch_assets.py`, where a task has them;
- its `prefix` is not `tasks/<platform>/<name>/payload`, its `revision` is not a 40-hex commit, or a line is
  malformed.

CI does not download anything; the downloads happen in the image build that `ale validate` runs.

## 6. This flow and the engine's `ale assets`

The engine has an asset mechanism of its own (`vendor/ale/docs/specs/task-assets.md`): files in a task's
`environment/assets/`, `tests/assets/` and `solution/assets/` stay out of git, and `ale assets pull` / `push`
synchronize them through Git LFS with one gated Hugging Face dataset per source repository,
`<namespace>/<repository name>`; they are pulled before the build, and running a task never downloads them. This
repository keeps the flow above: one dataset, named in `task-assets.toml`, whose files the image build fetches itself,
pinned per task by a dataset commit and a sha256 per file. Moving to the engine's mechanism would mean a new dataset
for this repository (`grounded-intelligence-initiative/agentic-robotics-benchmark`) holding every task's files.
`.gitignore` lists the engine's asset roots and `environment/setup/assets/`, so that nothing placed there reaches git.
