# NOTICE (grader-side; never staged into the agent's home)

The task's art is visual only: it shows in the photographs, in the frames of the fleet's sensors and in the maintainers'
videos, and nothing of it has physics.

- **The plant's hall round the bay** (`/opt/assets/arb/hall/`: `bay_hall.usda`, its prototypes `bay_hall_protos.usdc`,
  `textures/`, and `hall_manifest.json`, which names the source prefab of every prototype) was built for this task from
  meshes and textures of five Unity Asset Store packs that the benchmark's owner licensed for redistribution:
  "FactoryWarehouse Interior Modular Pack" 1.0.1, "Factory Interior Factory Props Vol 1 - BUNDLE" and "Warehouse Props
  Vol 1" 1.0 (CyberneticWalrus), "Warehouse - Abandoned Factory District" 1.1 (ScansFactory) and "Military Base
  Military Camp Environment ... Military Props" 1.0 (FreshCan3D). The FactoryWarehouse pack's demo hall is centred on
  the bay; a pipe bridge with a walkway, tanks, a process vessel, switchgear, racks, stores and safety signs stand round
  it, all outside the bay's volume. The materials are the packs' maps as UsdPreviewSurface.
- **The airframes' look layer** (`/opt/assets/arb/airframes/look/`, loaded by the `real` variant of each airframe's
  `look` variant set; `environment/arb_assets/make_airframes.py`): the propellers, motor bells, arm tubes, clamps,
  plates, dampers and skid sleeves of the industrial hexacopter of the Unity Asset Store pack "Realistic Drones"
  (licensed the same way), fitted to the catalog's four layouts. The quads' and the X8's frame plate and the
  propellers' blur disc were made for this task.
- **The photographs** `given/photos/P01.jpg` to `P12.jpg` were rendered for this task with Isaac Sim 6.2's RTX
  renderer (path traced) from the real bay in that hall, through the cameras of `given/photos/cameras.json`.

These files are not in git. They are in the benchmark's Hugging Face dataset
`grounded-intelligence-initiative/agentic-robotics-last-exam` under `tasks/swarm/hazard_bay_build/payload/`
(`given/photos/`, `arb/hall/`, `arb/airframes/look/`), fetched by the image build at the commit pinned in
`environment/assets.sha256` and checked file by file (`docs/task-assets.md`). To change one, upload it, re-run
`scripts/task_assets.py list` with the new commit and re-validate.

NVIDIA's Isaac Sim assets in `/opt/assets/isaac` (`environment/isaac_assets.sha256`) carry their own licences in their
folders.
