# NOTICE - sources

- **The robots** are the Franka Panda model of robosuite 1.5.2 (MIT), installed unmodified in the task image;
  `soup_sim.py` loads its model files (robosuite itself is not imported). `soup_sim.py` sets the fingers' position
  gain and force limit (4000 N/m, 70 N, the Franka Hand's continuous grasp force) after loading them.
- **Everything else on the counter** (the burner and its knob, the braiser and its tabs, the broth, the dishes and the
  pieces, the ladle, the bowls) is built from MuJoCo primitives in `soup_sim.py`; no third-party assets.
- **The cameras' images** are drawn by MuJoCo's own renderer (Apache-2.0) on Mesa's OSMesa.
