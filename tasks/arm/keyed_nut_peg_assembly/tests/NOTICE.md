# NOTICE - sources of what the cameras draw

- **The robot** is the Franka Panda model of robosuite 1.5.2 (MIT), installed unmodified in the task image.
- **The laboratory around the cell** (the cameras' render bundle, fetched into `/opt/task-assets/payload/render` at
  the dataset commit pinned in `environment/assets.sha256`; it is not in the agent's home) is a reduced derivative of
  two Unity Asset Store packs licensed to the benchmark's owner for distribution with the benchmark: the office scene
  of **HQ ArchViz Loft Office Modular** (the room, its desks and windows) and workshop furniture of **House Props
  Furniture Vol 2** (shelving, bins, totes, a trolley, recoloured to grey and black so that nothing in the room
  carries a nut's colour). Only what the cameras can see is kept: meshes decimated, base-colour maps reduced to 512
  px (the floor and the bench top 1024 px), one roughness and one metallic value per material. The cell's bench,
  pedestal and camera gantry in the bundle are original to this task.
- **The image light** (`render/ibl.ktx`) is a path-traced light probe of that room over the bench, prefiltered with
  Filament's cmgen; the sky behind the windows in the probe is Poly Haven's `kloofendal_48d_partly_cloudy_puresky`
  (CC0).
- The pegs, the nuts and their grooves are drawn from the simulation's own geometry (`keyed_env.py`, original to
  this task).
