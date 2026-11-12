# Loaded tray transport feasibility test

Historical experiment: the wide tray is no longer the preferred single-arm
object. Follow-up heavy-load tests exposed unreliable rim contacts. The current
small-box side-grasp tests are documented in
[BOX_TRANSPORT_TEST.md](</Users/dijie/Desktop/ALE Robotic Benchmark/simulator-lab/BOX_TRANSPORT_TEST.md>).
The light-load evidence below is retained; it does not establish reliable heavy
tray transport. Its original source snapshots remain beside the recorded video.

Run in the existing `arb-robocasa` environment:

```sh
conda activate arb-robocasa
python -I "/Users/dijie/Desktop/ALE Robotic Benchmark/simulator-lab/loaded_tray_transport.py" --seed 0
```

This is a privileged scripted reference in a fixed RoboCasa kitchen (layout 1,
style 1), not a sensor-only controller or a complete ALE benchmark task. It loads
an apple into a native tray, pinches the tray's raised end rim, lifts it, carries
it 0.42 m along the counter, places it, releases it, and withdraws.

The tray is RoboCasa's `objaverse/tray/tray_2` asset, with the registry's scale
multiplied by 0.8. Geometry, collision parameters, mass density, and robot actuator
limits are not modified. There are no runtime object teleports, welds, suction,
or attachment constraints. The apple is the small native-physics development
payload (approximately 3.3 g in seed 0), so this establishes feasibility for a
light load; larger loads, multiple items, and other seeds remain unvalidated.

The script reuses controller utilities from the adjacent `auto_pick_place.py`.
RoboCasa resets with its native base-frame arm controller; the script refreshes
the controller references before switching to world-frame control. Base actions
align the robot with the work area and compensate the existing joint friction.

## Acceptance

- Apple physically loaded and in contact with the tray.
- Tray grasped by both finger pads and lifted at least 0.08 m without counter contact.
- At least 0.30 m of loaded horizontal displacement while held and airborne.
- Payload bounding box remains inside a conservative tray-frame interior at every
  physics step from tray pickup through final verification. No retention violations.
- Tray placed within 0.035 m of the delivery point and 0.012 m of its initial support height.
- Tray contacts the counter; payload contacts the tray; gripper is released and over 0.25 m away.
- Tray and payload linear speeds below 0.025 m/s, and tray angular speed below
  0.10 rad/s, for at least 20 consecutive control steps at the end.

## Evidence

Every attempt writes `outputs/loaded_tray_seed0_TIMESTAMP/` containing
`loaded_tray.mp4`, `trace.jsonl`, `result.json`, initial RGB and metric depth,
a final image, and source snapshots. Results include source hashes, object masses,
tray dimensions, physics sample count, retention violations, measured displacement,
tilt, final velocities, and gripper clearance. Failed attempts retain their evidence.

Use `python` for this offscreen test, not `mjpython`. On macOS the process needs
access to the native graphics services, even though no interactive viewer opens.

## Verified result (2026-11-11)

Seed 0 completed in two fresh processes after correcting the rim grasp and
separating vertical position feedback from the horizontal transport speed cap.
The confirmation run also checked final payload speed and tray angular speed.

- Tray mass: 0.184 kg; dimensions: approximately 0.197 x 0.302 x 0.056 m.
- Maximum tray lift: 0.112 m; loaded horizontal transport: 0.421 m.
- Payload retention violations: 0 across 38,925 physics steps.
- Maximum tray tilt: 5.32 degrees; final delivery error: 0.00151 m.
- Final gripper clearance: 0.321 m; stable success: 77 control steps.
- Evidence: `outputs/loaded_tray_seed0_20261111_223542_507516/`.
