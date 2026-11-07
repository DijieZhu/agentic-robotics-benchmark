# NOTICE - sources of this task's assets (verify / maintainers only; never shown to the agent)

The objects, the kitchen and its clutter come from the **RoboCasa** asset set (S. Nasiriany et al., RoboCasa,
https://robocasa.ai). They are the source of two things: the objects' collision pieces in `ref/objects.json`, and
what the agent's videos show (`data/objects/<ID>/clipNN/cam*.mp4`). The asset set is the Hugging Face dataset
`robocasa/robocasa-assets` at revision `1b92c3d02ca4354984fec961357db0bff7b32166` (`objaverse.zip`, `textures.zip`),
licence **CC-BY-4.0**. This notice is the attribution the licence asks for.

- **Objects:** A `objaverse/kettle/kettle_13`, B `objaverse/condiment/condiment_2`, C `objaverse/jug/jug_0`, and the
  practice object P `objaverse/cereal/cereal_1`.
- **Clutter:** `objaverse/` cutting_board_0, lemon_0, mug_1, wine_1, bowl_0, teapot_0, tomato_0, apple_1.
- **Textures:** the counters `textures/marble/` granite, marble_2, dark_marble, granite_2. The backsplashes
  `textures/tiles/` black_square_tiles, concrete_tiles, flower_ceramic, orange_tiles. The cabinet front
  `textures/prev/wood4`.
- **Modifications:**
  - Each object is scaled uniformly and re-centred: bottom at z = 0, centred in x and y. The scales are A 0.1625,
    B 0.1725 and C 0.156. For P the maintainers reconstructed the scale for the re-render, about 0.168.
  - The collision pieces are RoboCasa's convex decomposition, as given.
  - The masses, centres of mass, inertia tensors and friction coefficients are this task's own; they are not
    RoboCasa's.
- **The robot** in the videos and in `rarm.py` is the Franka Panda model of robosuite 1.5.2 (MIT), installed unmodified
  in the task image.
- **The videos** were re-rendered by the maintainers on 2026-10-29. The renderer is MuJoCo 3.16's Filament, lit with
  an image light distilled from an OVRTX path-traced render of the same kitchen. The camera degradations are fitted
  to the statistics of the author's original videos (MuJoCo's native renderer). The physics, the probe paths, the arm
  motion and the cameras are the author's.
- **Where they live:** the 120 videos and the 80 data arrays of `data/objects/` are in the benchmark's Hugging Face
  dataset `grounded-intelligence-initiative/ale-robotics-assets` under
  `tasks/arm/robocasa_object_dynamics_from_video/payload/`, fetched at the commit pinned in `environment/assets.sha256`
  (docs/task-assets.md). Do not add them back to git; to change one, upload it, re-run `scripts/task_assets.py list`
  with the new commit, and re-validate.
