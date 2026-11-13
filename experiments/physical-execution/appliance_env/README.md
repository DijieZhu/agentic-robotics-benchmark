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
