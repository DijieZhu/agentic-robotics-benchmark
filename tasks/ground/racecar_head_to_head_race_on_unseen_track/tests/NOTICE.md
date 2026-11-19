# Notices

- `grader/` (the cell, the heat cell, the venue builder and its head-to-head additions, the referee, the wire, the
  contract's checker, the scoring), `carkit/` (also in the agent's givens), `h2hverify/`, `verify.py`, `test.sh` and
  the reference program in `../solution/reference/` were written for this benchmark; the files this task shares with
  `racecar_time_trial_on_unseen_track` are that task's, unchanged. The car's geometry and constants come from the
  owner's own model of his car (the same model as this platform's requirements, CAD and assembly tasks); the drive,
  tyre, LiDAR, IMU, duct and floor models are this task's own and are documented to the agent at datasheet level only.
  The other cars are the same car, driven by the same pure-pursuit program as the sibling task's slow cars.
- `cellkit/` is the benchmark's process kit, copied unchanged from `tasks/arm/pump_skid_startup_and_operation/tests/`;
  `h2hverify/netkill.py` follows the structure of its `nonet.py` filter. `robotics_grader/` is the benchmark's kit
  (`shared/robotics_grader/`).
- The league's rules given to the agent are this series' own indoor league. Their procedures (closed practice with a
  mapping period, heats of several cars from a staggered grid, laps to a set number or a time cap, overtakes, contact
  and its fault, warnings and a stop by the staff after a crash, the staff's handling of the cars) follow the public
  general and competition rules of real tenth-scale autonomous racing events and their head-to-head formats; where
  the league departs from them, the rules say so. No rule text is copied.
- The agent's givens: the documents, the lab logs and the client library were written or recorded for this benchmark
  (the lab logs by the shared cell's own lab rig). The eleven venue photographs are path-traced stills of two public
  example venues of the venue generator (seeds registered for the purpose, never a hidden venue), rendered with the
  series' hall model and art the owner supplied; the car's meshes are exported from the owner's own model of his car
  (the series' assembly kit), with neutral part names; what in them derives from the makers' CAD files of that model
  keeps those makers' terms. The five scrutineering photographs are path-traced renders, made for this task, of the
  owner's model of his car with the league's marker box on it, and two of the car's given files
  (`car/meshes/compute_unit.obj`, `car/meshes/textures/controller_board.png`) had lettering removed.
- Python distributions of the image: MuJoCo (Apache-2.0), numpy and scipy (BSD-3-Clause), jsonschema (MIT), Pillow
  (MIT-CMU), OpenCV (Apache-2.0).
