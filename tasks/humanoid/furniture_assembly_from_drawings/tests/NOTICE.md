# NOTICE (grader-side; never staged into the agent's home)

The two robots are the MuJoCo Menagerie `unitree_g1` model (`g1_with_hands.xml`; BSD-3-Clause, its LICENSE ships inside
the folder), fetched by `environment/Dockerfile` at commit f3b6013629c7b94f8ee93fd977b161fa8350f9bc and pinned file by
file in `environment/g1_assets.sha256`. The cell replaces the model's floating base and actuators (base joints, servos,
gravity compensation) and uses the primitive colliders of Menagerie's `g1_mjlab.xml`; `tests/furniture_cell/robot.py`
builds it with MjSpec at cell start. The agent gets the unmodified folder in `~/given/robot/unitree_g1/`.

The B930 bookcase, its drawing set (`tests/furniture_cell/drawings.py`, rendered with matplotlib), the room and its
fixed furniture as the physics knows them (the base cabinet, the couch, the plant and the rug, built from primitive
shapes), the panel cart and the kitting table are original to this task. No third-party furniture model, drawing or
dataset is part of the physics or the drawings.

What the cameras see of the room, its decor (the oak floor, the plaster walls and ceiling, the skirting, the window's
frame, glass and board, the door leaf, the pictures, the clock, the ceiling panels, the couch, the plants, the rug and
the buildings across the street), is drawn from meshes and textures of two Unity Asset Store packs that the benchmark's
owner licensed for redistribution, "Office and Police Station Pack" and "Office With Conference Room": the props as
the packs model them, re-placed in this room, and the shell (walls, ceiling, skirting, window) built for this room
from the packs' textures. The base maps are downsized to 512 px. The image light (`ibl.ktx`) is the mean of three light
probes of the same room path-traced with NVIDIA OVRTX, the sunlit patches and the lamps' emissive surfaces painted out
(the cameras' renderer draws their light with lights of its own), prefiltered with Filament's cmgen. The decor is not in
git: its files are in the benchmark's Hugging Face dataset `grounded-intelligence-initiative/ale-robotics-assets`
under `tasks/humanoid/furniture_assembly_from_drawings/payload/render/room_decor/`, fetched by the image build at the
commit pinned in `environment/assets.sha256` and checked file by file (`docs/task-assets.md`); to change one, upload it,
re-run `scripts/task_assets.py list` with the new commit and re-validate. The cameras' colour renderer is MuJoCo's
Filament module (Apache-2.0, part of the mujoco wheel). The robots' legs as the cameras draw them while a robot moves
(`furniture_cell/walk_core.py`: footsteps planned from the base motion, leg inverse kinematics on the G1 model's own leg
chain) are original to this task: no motion-capture or third-party animation data.
Prior art the task was designed against (none of it reused): the Humanoid IKEA Assembly Challenge (IROS 2026, one G1,
IKEA UTTER table, learned policies), EmbodiedSWE-Bench (coding agents with privileged simulator state), Manual2Skill /
Manual2Skill++ (VLM-read IKEA manuals, fixed-base arms) and IKEA-Manual / IKEA Furniture Assembly environments.
