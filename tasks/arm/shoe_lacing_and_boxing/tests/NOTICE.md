# NOTICE (grader-side; never staged into the agent's home)

The two arms are the MuJoCo Menagerie `franka_emika_panda` model (Apache-2.0, its LICENSE ships inside the folder),
fetched by `environment/Dockerfile` at commit f3b6013629c7b94f8ee93fd977b161fa8350f9bc and pinned file by file in
`environment/panda_assets.sha256`. At build time `environment/obj2stl.py` turns the model's visual OBJ meshes into
binary STL (same geometry) and rewrites the mesh references; the result is checked file by file against
`environment/panda_converted.sha256`. The cell builds both arms from that folder with MjSpec at cell start
(`tests/shoecell/cell.py`: the pedestals, the gripper's attach weld, the lacing hook on the second arm's flange). The
agent gets the converted folder in `~/given/panda/`.

The sneaker shell with its eyelet flaps, the lace and its aglets, the lacing cradle, the die-cut box blank, the lid
tray, the bench and the infeed tray are original to this task (`tests/shoecell/`, byte-identical to the agent's
`~/given/lib/shoecell/`). No third-party shoe, lace or packaging model or dataset is used. Prior art the task was
designed against (none of it reused): UniLace (Unity and Obi rope, a bimanual shoe-lacing simulator), Luo and Demiris
(RA-L 2024, simulated lacing with a tension limit), LapGym's rope threading (SOFA), RLBench's `put_shoes_in_box`, pi0 /
RL-100 box folding and EmbodiedSWE-Bench's packing tasks.

The agent cameras' render bundle (`render/shoecell_render_v2.tar` in the benchmark's Hugging Face asset dataset,
pinned by `environment/assets.sha256`, unpacked to `/opt/task-assets/render`) holds the factory hall and its props
(CyberneticWalrus' FactoryWarehouse Interior Modular Pack, Warehouse Props Vol 1 and Factory Props Vol 1, Unity Asset
Store, licensed to the benchmark's owner for redistribution), CC0 textures from ambientCG (leather, knit, rubber,
paper, cardboard, metal) and Poly Haven (oak_veneer_01) for the cell's parts, the sneaker skin template
(original to this task, modelled on the task's shoe shell) and an image light rendered with NVIDIA OVRTX from that
scene.
