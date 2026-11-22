# NOTICE - sources

- **The robots** are the Franka Panda model of robosuite 1.5.2 (MIT), installed unmodified in the task image;
  `door_sim.py` loads its model files (robosuite itself is not imported). The right arm's vacuum cup tool is built
  from MuJoCo primitives in `door_sim.py`.
- **Everything else in the cell** (the table, the wall cabinet and its hinges, the door, the rack, the pins and their
  stands) is built from MuJoCo primitives in `door_sim.py`; no third-party assets.
