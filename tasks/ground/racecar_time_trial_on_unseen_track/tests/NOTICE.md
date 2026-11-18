# Notices

- `grader/` (the cell, the venue builder, the referee, the wire, the contract's checker, the scoring), `carkit/` (also
  in the agent's givens), `ttverify/`, `verify.py`, `test.sh` and the reference program in `../solution/reference/`
  were written for this benchmark. The car's geometry and constants come from the owner's own model of his car (the
  same model as this platform's requirements, CAD and assembly tasks); the drive, tyre, LiDAR, IMU, duct and floor
  models are this task's own and are documented to the agent at datasheet level only.
- `cellkit/` is the benchmark's process kit, copied unchanged from `tasks/arm/pump_skid_startup_and_operation/tests/`;
  `ttverify/netkill.py` follows the structure of its `nonet.py` filter. `robotics_grader/` is the benchmark's kit
  (`shared/robotics_grader/`).
- The league's rules given to the agent are this series' own indoor league. Their procedures (closed practice with a
  mapping period, two timed heats, the obstacle test, touch and crash by a 5 cm displacement, the staff's handling
  after a crash, open walls marked by tape) follow the public general and competition rules of real tenth-scale
  autonomous racing events; the ratios that place the zero points of the scoring (1.33 and 0.43 times the reference
  run's own numbers) come from the field of a real 2026 time trial. No rule text is copied.
- The agent's givens: the documents, the lab logs and the client library were written or recorded for this benchmark
  (the lab logs by the frozen cell's own lab rig). The twelve venue photographs are path-traced stills of two public
  example venues of this task's own venue generator (seeds registered for the purpose, never a hidden venue), rendered
  with the series' hall model and art the owner supplied; the car's meshes are exported from the owner's own model of
  his car (the series' assembly kit), with neutral part names; what in them derives from the makers' CAD files of that
  model keeps those makers' terms. The lab logs, the photographs and the meshes are served from the benchmark's asset
  dataset at the commit pinned in `../environment/assets.sha256`.
- Python packages of the image: MuJoCo (Apache-2.0), numpy and scipy (BSD-3-Clause), jsonschema (MIT), Pillow (MIT-CMU),
  OpenCV (Apache-2.0).
