# NOTICE - sources of this task's assets (verify / maintainers only; never shown to the agent)

The logs of the practice rigs (the task's assets, `payload/practice_rigs.tar`) and of the hidden rigs (the verifier's
archive, `verifier/hidden_rigs.tar`, pinned in `tests/grader_config.json`) were generated once with `tests/gen/` from:

- **The site**: a warehouse hall and 34 kinds of warehouse and factory props (crates, cartons, pallets, tool
  cabinets, carts, hand trucks, pallet jacks, forklifts, a reach truck, racks, conveyor pieces, wire partitions, a
  wrapping machine, a scale, a table) from the maintainers' licensed warehouse and factory art packs, converted to
  USD with their materials and maps. They appear in the task only as rendered camera frames and as LiDAR returns.
- **The robot**: Clearpath Husky A200 meshes (base, top chassis, top plate, user rail, wheels, 510 mm sensor arch)
  from Clearpath Robotics' `husky_description` (BSD 3-Clause). They appear in the task only as the LiDAR's
  self-occlusion (the returns the body blocks).
- **Sensor models named in the task** (figures from their datasheets, listed in `tests/gen/README.md`): FLIR
  Blackfly S BFS-U3-63S4C, Velodyne VLP-16, Bosch BMI088. No vendor file is part of the task.
- **Renderers**: Blender 5.3 (EEVEE) for the camera frames; Embree for the LiDAR rays.

`tests/review/author_calibrator_v1.py` and `author_rigsim_v1.py` are the task author's first reference calibrator
and generator, kept as review material.
