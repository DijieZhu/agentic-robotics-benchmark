# Appliance tending, P1: the environment prototype

Phase P1 of `docs/dev/20261112-appliance-tending-task-design.md`. Results are in that document's
§10. Nothing here is a task folder; the task is built in P4.

| File | What it does |
|---|---|
| `appliance_env.py` | The prototype environment: plain MuJoCo 3.16, a fixed Panda (robosuite 1.5.2 model files only), the operational-space controller, two fixed cameras and a wrist camera, the wrist F/T sensor (all after `tasks/arm/arm_hidden_mechanism_box/tests/box_env.py`), and one RoboCasa appliance model loaded as a **unit**. RoboCasa's appliance predicates are re-implemented in `Unit`. |
| `inventory.py` | Loads every stove, stovetop, microwave and oven model standalone and records joints, burner sites, buttons, sizes and moving masses. |
| `reach.py` | Kinematic reach study: IK to every control (pre-contact point and contact point) with an approach direction, over a mount. |
| `check_predicates.py` | Threshold checks of the re-implemented predicates, and physical microwave button presses through the controller. |
| `perf.py` | Build time, control steps per second, seconds per image, peak memory. |
| `smoke.py` | Builds units and saves a contact sheet of the three camera images. |
| `reference.py` | P2: the privileged reference's primitives (knob, button, doors) behind a collision-checked IK planner; acts only through the 8-number action. |
| `run_reference.py` | P2: runs primitives as trials (fresh episode, initial state, one primitive) and writes JSON lines; `--video` saves the cameras. |
| `batch.sh` | P2: every candidate model x the primitives x seeds, sharded over parallel processes. |
| `summarize.py` | P2: success per kind and primitive, failure stages, fully reliable models. |
| `episode.py` | P2: the grader-side episode around the environment: initial state, three requests, the two-stage credit. |
| `run_episodes.py` | P2: whole episodes with the reference (or `--untouched`), one JSON line each. |
| `batch_episodes.sh` | P2: reference episodes on every candidate model x seeds, sharded. |
| `oracle_main.py` | P3: the reference as a submission: speaks the grading wire, builds its own copy of the scene from the token truth (`reference.Shadow`), runs the primitive for each request. |
| `wire_grade.py` | P3: grades a submission over the wire with the kit's `Solver` and `ProbeLock` (prototype of the task's grader and practice grader); `--truth` makes it a token run. |
| `sync_task.py` | P3: copies the canonical code and the selected unit models into `tasks/arm/kitchen_appliance_tending/` (byte-identical copies). |
| `baseline/` | P3: the practice-model baseline: `perceive.py` (placement and knob angles by render-and-compare against the fixed cameras), `baseline_main.py` (the reference's primitives without the truth). |

## Set-up (outside the repository)

The RoboCasa fixture assets are the v1.0 "lightwheel" set (`download_kitchen_assets.py`, key
`fixtures_lw`; archive `fixtures_lightwheel.zip`, 529,745,465 bytes, sha256
`62c1c87776edda9b7c42dda7b00781a1e5a7f90acc3e49d06c39c9545cbd6c3e`). Its README names the source:
the Hugging Face dataset `nvidia/PhysicalAI-Robotics-Manipulation-Objects-Kitchen-MJCF`, licence
CC-BY-4.0. Extract it outside the repository:

```bash
mkdir -p ~/arb-data/robocasa-assets/dl ~/arb-data/robocasa-assets/extracted
curl -L -o ~/arb-data/robocasa-assets/dl/fixtures_lightwheel.zip https://utexas.box.com/shared/static/idbncsadpnaz1jfl4i6m8qejawk7p9pi.zip
```

```bash
cd ~/arb-data/robocasa-assets/extracted && python3 -I -m zipfile -e ../dl/fixtures_lightwheel.zip .
```

A Python 3.12 environment with the box task's pins:

```bash
uv venv --python 3.12 ~/arb-data/venv-p1
```

```bash
VIRTUAL_ENV=~/arb-data/venv-p1 uv pip install mujoco==3.16.0 numpy==2.5.4 scipy==1.18.1 opencv-python-headless==5.0.0.93 pillow==13.0.0 && VIRTUAL_ENV=~/arb-data/venv-p1 uv pip install --no-deps robosuite==1.5.2
```

`run_reference.py --video` also needs `imageio` (`VIRTUAL_ENV=~/arb-data/venv-p1 uv pip install imageio==2.37.0`).

## Runs

Outputs go to `../outputs/p1/` (git-ignored). Rendering uses `MUJOCO_GL=osmesa`.

```bash
~/arb-data/venv-p1/bin/python -I inventory.py --assets ~/arb-data/robocasa-assets/extracted/fixtures --out ../outputs/p1/inventory.json
```

```bash
~/arb-data/venv-p1/bin/python -I reach.py --assets ~/arb-data/robocasa-assets/extracted/fixtures --front-x 0.55 --robot-z 0.65 --out ../outputs/p1/reach.json
```

```bash
MUJOCO_GL=osmesa ~/arb-data/venv-p1/bin/python -I check_predicates.py --assets ~/arb-data/robocasa-assets/extracted/fixtures
```

```bash
MUJOCO_GL=osmesa OMP_NUM_THREADS=1 ~/arb-data/venv-p1/bin/python -I perf.py --assets ~/arb-data/robocasa-assets/extracted/fixtures --out ../outputs/p1/perf.json stovetops/Stovetop002 microwaves/Microwave011 ovens/Oven048 stoves/Stove026
```

```bash
MUJOCO_GL=osmesa ~/arb-data/venv-p1/bin/python -I smoke.py --assets ~/arb-data/robocasa-assets/extracted/fixtures --out ../outputs/p1/smoke stovetops/Stovetop002 microwaves/Microwave011 ovens/Oven048 stoves/Stove026
```

```bash
bash batch.sh ../outputs/p2/batch3 32 0 1 2 && ~/arb-data/venv-p1/bin/python -I summarize.py ../outputs/p2/batch3/*.jsonl
```

```bash
bash batch_episodes.sh ../outputs/p2/ep_batch1 40 0 1 2 3 4
```
