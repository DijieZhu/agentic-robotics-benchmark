# What is in this home directory

- `photos/`: the photo session of the arm (21 photos, `photos.json`, `bus_log.json` with the servo readings of every photo,
  `README.md` = how the photos were taken).
- `parts/`: the published printed parts P01-P10 as printed (STL, mm, print frames), `base_in_world.stl` (the published base in
  the arm's world frame), `servo.stl` (the servo's vendor mesh), `parts.json`.
- `servo_datasheet.md`, `servo_datasheet.png`, `servo_drawing.png`: the servo (BUS-SERVO-12V-A).
- `servo_config.json`: the EEPROM dump of the six servos of this arm.
- `masses.json`: the slicer's mass estimates of the published parts, the servo mass, the weighed total of this arm.
- `build_notes.md`: the builder's notes (no dimensions).
- `scale_refs.md`: the nominal sizes of the references that were on the platform in some photos.
- `robot_contract.md`, `contract/zero_pose.png`, `contract/tcp_definition.png`: the files you deliver and their conventions.
- `check_submission.py`, `rd_info_check/`: the contract gate (the grader's own gate code; no ground truth involved).
- `submission/`: where your description goes.
