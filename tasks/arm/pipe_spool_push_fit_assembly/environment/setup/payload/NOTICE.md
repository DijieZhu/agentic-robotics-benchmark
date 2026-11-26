# NOTICE - sources

- **The robots** are the Franka Panda model of robosuite 1.5.2 (MIT), installed unmodified in the task image;
  `spool_sim.py` loads its model files (robosuite itself is not imported).
- **Everything else in the cell** (the bench, the pipes, the push-fit elbows) is built from MuJoCo
  primitives and one convex mesh generated in `spool_sim.py`; no third-party assets.
