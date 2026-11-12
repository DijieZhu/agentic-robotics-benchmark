# Small-box transport component tests

Final revised protocol on 2026-11-11: **45/45 passed**: 30/30 development and
regression cases plus 15/15 new held-out cases. See the
[audited results report](</Users/dijie/Desktop/ALE Robotic Benchmark/simulator-lab/outputs/box_v2_evaluation/report.md>)
for the frozen source hashes, per-mass metrics, retained failures, and video.
This result covers preloaded-box transport only.

The wide tray rim grasp is retained as an experiment, but is not the preferred
single-arm work object. These tests instead use RoboCasa's native rigid
`lightwheel/tupperware/Tupperware036` asset, scaled to approximately
94 x 68 x 46 mm. The Panda grasps both opposite side walls across the short
dimension, near the combined centre of mass. Native empty-container mass is
approximately 8.6 g.

## Scope

- `empty_box_transport.py`: empty-container grasp, lift, transport, placement,
  release, and withdrawal. No payload capability is claimed by this mode.
- `preloaded_box_transport.py`: the scene initializer places a small rigid
  item inside the box, before controller execution. The robot transports the
  loaded box; it does not load the item in this component test.
  The final generator uses the fixed native
  `objaverse/boxed_food/boxed_food_0` asset at object scale 0.45 (approximately
  23 x 59 x 36 mm including registry scaling). This flat package avoids fruit
  rolling to the edge during initialization. Seeds vary scene initialization,
  not object identity.
- `loaded_box_transport.py`: an experimental robot-loading sequence. This
  combined loading-and-transport sequence has **not** passed validation and
  should not be reported as working.

The controller uses private simulation poses and contact state. This remains a
privileged feasibility reference in one fixed kitchen layout/style, not a
sensor-only policy or a complete long-horizon ALE task.

## Physical setup and acceptance

For a requested ballast mass, the payload's mass and rotational inertia are
scaled by the same factor before the first controller action. Model constants
are recomputed with scratch simulation data; live object positions and
velocities are unchanged. Geometry and contact parameters, gripper force limits,
and robot actuator limits are unchanged. The mass sweep is a fixed-geometry
ballast stress test, not a material model of a particular packaged food.
At 200 g its bounding volume implies approximately 4.2 g/cm3: a dense ballast
load, rather than the nominal food suggested by the asset texture.

There are no welds, suction attachments, or runtime object teleports. The sampler
supplies XY jitter and orientation. Before reset applies placements, the custom
scene initializer specifies a payload bottom 8 mm above the carrier's bounding
box bottom, just above its physical floor. Normal reset physics settles it.
This avoids a drop from above the rim. Invalid initial containment is reported
as a failure; no scene is silently resampled based on success.

Loaded acceptance retains the tray test's thresholds: at least 0.08 m physical
lift, at least 0.30 m airborne displacement while grasped, no payload containment
violations at any physics step from the first controller action through final
withdrawal (including approach and grasp), final placement within 0.035 m of the target,
correct support contact, gripper released and more than 0.25 m away, and at least
20 stable control steps. Linear speed must be below 0.025 m/s and container
angular speed below 0.10 rad/s. The nominal horizontal transport is 0.42 m.

## Run

```sh
conda activate arb-robocasa
cd "/Users/dijie/Desktop/ALE Robotic Benchmark/simulator-lab"
python -I empty_box_transport.py --seed 0
python -I preloaded_box_transport.py --seed 0 --payload-mass-g 200
```

The fixed evaluation matrix uses 50, 100, and 200 g payloads over development
seeds 0–4 and 100–104, followed by new held-out seeds 200–204 with the frozen
controller. Seeds 100–104 became development data after revealing a generator
failure; they are not counted as held out after that fix:

```sh
python -I loaded_tray_batch.py --script preloaded_box_transport.py --seeds 0 1 2 3 4 100 101 102 103 104 --masses-g 50 100 200 --label box_development
python -I loaded_tray_batch.py --script preloaded_box_transport.py --seeds 200 201 202 203 204 --masses-g 50 100 200 --label box_holdout
```

The runner retains all failed cases and uses a fresh process per trial. It
refuses mixed controller versions, verifies the actual mass, and records hashes
of the entry script and all controller dependencies. Every case keeps its video,
trace, source snapshots, console log, and result. Batch directories contain
`summary.json` and `summary.md`. No failed seed is silently retried or omitted.

The final run partitions the same frozen matrix into three concurrent batches,
one per mass, with all 15 seeds in each. `report_box_transport.py --batches ...
--output-dir ...` audits the complete 45-pair matrix and writes its aggregate
report. It checks actual masses, source hashes, recorded acceptance metrics,
trace completeness and containment, video presence, and three RGB-D artifacts.

Environment: Python 3.11, RoboCasa 1.0.1 at
`3443cdf3dd20f9ae8ed41adba442059b506891d0`, robosuite 1.5.2 at
`5ce6643f3092639d08f7b0f90ed1c6a84f50552c`, and MuJoCo 3.3.1 on Apple Silicon
macOS. Simulator source files were not modified for this experiment.

## Initial component checks

On 2026-11-11, empty-box seed 0 passed with a 0.120 m lift, 0.421 m transport,
0.331 degree maximum tilt, and 70 stable end steps. The 200 g preloaded-box seed 0
probe passed with a 0.119 m lift, 0.420 m transport, 0.398 degree maximum tilt,
zero retention violations, and 70 stable end steps. The multi-seed matrix is
recorded separately; these two probes alone do not establish reliability.
That initial loaded probe used a small apple and is excluded from the final
fixed-package evaluation. Earlier fruit initialization failures, the aborted
small-package diagnostic batch, and tray grasp failures remain in `outputs/`.
