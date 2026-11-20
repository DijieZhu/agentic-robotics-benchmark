# NOTICE - sources

- **The robots** are the Franka Panda model of robosuite 1.5.2 (MIT), installed unmodified in the task image;
  `jar_sim.py` loads its model files (robosuite itself is not imported).
- **Everything else on the counter** (the jars, their lids and labels, the tray, the refill pieces) is built from
  MuJoCo primitives in `jar_sim.py`; no third-party assets.
- **The kitchen in the cameras' images** (`/opt/task-assets/payload/render`, drawn by `jar_render.py`; scenery only,
  nothing of it is simulated): the house and kitchen of the Unity Asset Store pack "ArchVizPRO Interior Vol.6"
  (meshes decimated, base-colour maps), the counter's oak top from Poly Haven's `oak_veneer_05` (CC0), and an image
  light from a path-traced probe of that kitchen under Poly Haven's sky `kloofendal_48d_partly_cloudy_puresky` (CC0).
  The renderer is MuJoCo 3.16's Filament module (Apache-2.0) on Mesa's lavapipe.
