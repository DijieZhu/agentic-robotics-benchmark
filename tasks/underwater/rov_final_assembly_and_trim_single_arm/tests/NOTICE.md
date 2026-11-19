# NOTICE (grader side; never staged into the agent's home)

This file names the real sources behind the task's generic surface, carries their attributions and licences, and
records for maintainers the few places where the task departs from those sources and the limits of its simulation.
The agent-visible material (`instruction.md`, everything under `environment/setup/payload/` and the asset payload the
image stages into the agent home) names no platform, maker, product, part number, simulator, dataset or author other
than the registered exemptions listed in section 8: the vehicle is "an inspection-class ROV", the units carry the
generic names of the underwater tasks' naming table, the arm is "UR10e-class" as in the benchmark's published
assembly tasks, and the gripper is "the wide gripper". How the grader scores and how the fixed hidden set was chosen
are in `README.md`. The benchmark's owner has stated that the sources below may be used and distributed with the
benchmark.

## 1. The vehicle

The vehicle is the **BlueROV2 R4 in its Heavy (8-thruster) configuration by Blue Robotics**, with the vendor's
accessories of the underwater tasks' `mission` configuration: the Newton subsea gripper on the starboard side, the
Ping360 scanning imaging sonar on the starboard side plate, a Water Linked DVL A50 on its stainless bracket on the
port side, four Lumen lights, no payload skid. It was measured once from the vendor's public R4 STEP model, the
official Heavy retrofit parts, the accessories' public models and the official guides into the underwater tasks'
frozen reference-vehicle digest, version v3 (`spec_values.json` sha256
`358676915495f3ea4e719154634557669b14a69a7554dce6a134cb80bf3b9bbd`, `vehicle_parts_manifest.json`
`99b937ff958bf88e28a866059a887fcfeb3981b80e6a5b677da5056fbf448abe`); every digest-derived quantity of the cell, the kit,
the trim table and the documents is read from it, none is typed in. The four lower guard brackets have no public CAD:
they are generated parts of the digest (mass +-30 %, position 1 mm).

| Agent-visible | Real source | What was done to it |
|---|---|---|
| `given/parts/base_vehicle.stl` | the vehicle as the station receives it: the frame's seven plates, the two 4-inch cradles and the 3-inch clamp, the battery tube with its battery, the four horizontal T200 thrusters, the Newton gripper with its mount, the DVL A50 with its bracket, the four lower guard brackets | the digest's decimated mesh; six screw heads flattened (section 3) |
| `given/parts/electronics_enclosure.stl` | the vendor's 4-inch watertight enclosure (acrylic tube and dome, end cap, flanges, the electronics inside) with the six M3 x 16 screws that hold the two cradles | outside the tube the high-resolution mesh, inside the decimated one; not decimated further |
| `given/parts/thruster_a.stl`, `thruster_b.stl` | the vendor's T200 thruster with its CW and its CCW propeller | the vendor's CAD as it is, both in one frame; the two propellers are **not** mirror images (blade angles and widths differ), which the datasheet page states |
| `given/parts/thruster_guard.stl` | the Heavy retrofit's vertical-thruster guard plate | as it is; the collision cut-out is carved wider than the drawn one (cell manual) |
| `given/parts/ballast_block.stl` | the vendor's 200 g stainless ballast weight | as it is; the block's outline is turned by about 1 deg against the file's axes in the vendor's model and kept so |
| the trim sheet, the water density chart, the propeller datasheet, the seven drawings | the digest | drawn for this task from the measured geometry and the trim table; none is a copy of a maker's document |
| `given/manual/photo_01..16.jpg` | the high-resolution meshes | rendered for this task (Filament on CPU, the factory hall of section 5) with physical scale references (steel rules, an ID-1 card); no camera poses in the files |

The hidden kit (`gt/`, `assets/gt.tar.xz`, sha256 pinned in `grader_config.json`) holds the units' visual meshes and
collision pieces rebuilt for this task from the high-resolution meshes (convex pieces, boxes and spheres that add no
material outside a unit's outline, less the carvings the cell manual lists), made with numpy and trimesh.

## 2. What the trim acceptance takes from the vendor, and what not

- The trim table (`trim_table.json`; the printed trim sheet) is generated from the digest: mass to 1 g (the ballast
  unit to 0.1 g), displacement to 0.1 cm3, centres to 0.1 mm. Its printed in-water weights and net buoyancies are formed
  at the reference density 997.0 kg/m3, the vendor's weighing assumption recorded in the digest.
- **The vendor's own figures are not used anywhere visible**: the vendor's product pages give the subsea buoyancy foam
  block as 209 g with a net buoyancy of +488 g and the ballast weight (nominal 200 g) as 165 g in water. The volumes
  these imply are 1.7 % and 4 % larger than the digest's B-rep volumes; trimming with them would overstate the net
  buoyancy by about 0.6 N. The trim sheet prints the digest's values (209 g / +474.3 g for the corner foam block,
  192.8 g / 167.7 g in water for a ballast unit of block and screw).
- **The per-unit deviation acts on mass and centre of mass only**: displacement and centre of buoyancy are taken as
  tabled (an acceptance convention printed on the sheet). On a real line the unmodelled cables that explain such a
  deviation would move their displacement too (a smaller net effect than their mass moment); the convention ignores
  that. The documents describe the range only as the statistical range of this batch's factory weighing record and
  tell no physical story for it (ruling N5).
- Seawater density: the TEOS-10 equation of state (`trim_eval.rho_teos10`) is transcribed in pure Python from GSW-C
  3.6.23 as shipped in the `gsw` Python package's source distribution (`SA_from_SP` at 0 E 0 N, `pt0_from_t`,
  `CT_from_pt` and the 75-term specific-volume polynomial); it equals `gsw` 3.6.23 bit for bit on two test grids. The
  Gibbs SeaWater (GSW) Oceanographic Toolbox: Copyright (c) 2011, SCOR/IAPSO WG127; Copyright 2017-2020 Eric Firing and
  contributors; BSD-style licence (http://www.TEOS-10.org). The water density chart the agent reads is drawn from the
  same function.

## 3. The two appearance-only changes, and two touch-ups of the photographs

The fasteners of the digest have no collision bodies, so neither change alters the physics:

1. **Four M4 socket heads under ballast holes C4 and C5.** The heads of the 3-inch clamp's screws stand 1.47 mm below
   the keel plate's lower face inside the footprints of C4 and C5 (the digest lists them as tolerated contacts). In
   `base_vehicle.stl`, the high-resolution mesh, the photographs and the drawings they are cut flush with the plate
   (1.50 mm removed), so C4 and C5 stay usable holes, as in the cell.
2. **Two M5 button heads on the port side plate's outer face** (x -167 and x +185 mm of the six there) lie under the
   port guard's inner edge with a 4.0 mm overlap; they are cut flush with the plate's outer face (|y| 169.025 mm,
   2.75 mm removed). The digest's vehicle has no counterpart of these six heads on the starboard plate; whether the
   vendor's STEP has none there either was not checked. It affects only the appearance.

In the photographs only: the flush caps of the six cut heads are painted in the plate's colour (their hexagon sockets
stay, as in the STL), and a raised maker's mark on the top of the scanning sonar's head (a unit of the next station)
is covered by a thin conforming cap of the same material.

## 4. The robot

- **Arm**: the MuJoCo Menagerie model `universal_robots_ur10e` (derived from the ROS-Industrial URDF), BSD-3-Clause,
  Copyright 2018 ROS Industrial Consortium; byte copy of the benchmark's published assembly tasks' folder (its licence
  in `given/robot/universal_robots_ur10e/LICENSE`). The folder keeps its upstream name because the vendored
  `ur_robot.py` looks it up by that name.
- **Gripper**: the 2F-140 MJCF of `personalrobotics/geodude_assets` (`robotiq_2f140`, derived from the
  ros-industrial `robotiq_2f_140_gripper_visualization` URDF), BSD-2-Clause, Copyright (c) 2013, ROS-Industrial. Its
  seven STL meshes and its LICENSE are byte copies (`given/robot/wide_gripper/`); `wide_gripper.xml` is its `2f140.xml`
  with the comments removed, the model name changed to `wide_gripper` and the default class to `wide` (every numeric
  array of the compiled model bit for bit equal). The upstream README is not shipped (vendor names, a product link).
  The cell drives it with this task's re-tuned actuation and pads (`arm/assembly_rov/gripper_wide.py`).
- The cell mounts the gripper on the arm's flange (`ur_robot.py`) and drives the arm with the precedents' computed-torque
  controller (`ur_controller.py`).

## 5. Code, scenery and software

- **Grader code**: the benchmark's assembly-cell modules `arm/assembly_cell`, `assembly_info`, `assembly_six`,
  `render`, `render_filament`, `room`, `common`, `robotics_grader`, `episode_trace.py`, `verify_freeze.py`, `test.sh`
  and `arm/{chain_model, frames}.py` are vendored byte for byte from the published quadruped assembly task
  (`quadruped_full_assembly_12dof_ur10e`), and `arm/assembly_rov/{ur_robot, ur_controller, ur_ik, motion, rests,
  kitpose}.py` from its `arm/assembly_quad/`, with four registered changes (the gripper module's name in
  `ur_controller.py` and `motion.py`; the pinned numerics exported in `test.sh`, `README.md`; the datum boards' print
  texture decoded as sRGB in `arm/assembly_info/cell_render.py`, as the room's and the bench top's colour maps are:
  without it MuJoCo's Filament read the print as linear and lifted the dark squares) and one registered rewrite: `episode_trace.py` keeps the record within limits (camera frames thinned and capped, qpos records, events
  and each episode's trace capped, the agent's stderr log capped; ruling P26-2, its docstring says what was added),
  registered by the sha256 of its source and of the rewrite in the development folder's `tools/vendor_sync.py`;
  `netguard.py` is the benchmark's `driving_safety_supervisor` copy;
  `environment/fetch_assets.py` is the benchmark's `shared/task_assets/` copy. The cell, the coupling, the positioner,
  the fixtures, the trim truth, the server, the reference program and `verify.py` were written for this task.
- **The agent's library** (`given/lib/`): this task's own files are byte copies of `tests/`; the vendored precedent
  files are the published assembly task's agent-side rewrite as merged with its given Filament layer
  (`render/filament/`, `render/room.py`, the Filament-by-default `arm/assembly_info/cell_render.py`), with three
  registered renames (the keyword `anchor` -> `align` in `render/textures.py` and `render/scale_refs.py`, the default
  camera `overview` -> `top` in `arm/assembly_info/client.py`) and the boards' sRGB line of `cell_render.py`. The
  cell's camera session for the agent's own twin (`arm/assembly_rov/agent_render.py`, `twin_session.py`,
  `agent_look.json`) is derived from this task's `tests/arm/assembly_rov/` (`agent_render.py` by counted edits,
  `twin_session.py` the render twin's functions it reads, unchanged; the look without its provenance keys) by the
  development folder's `tools/agentview/port_agent_render.py`.
- **The agent cameras' scenery** (`render/`: what the cell's top and wrist cameras show around the cell, at verify and
  in the agent's session): the demo film's marine-vehicle workshop set, baked with Cycles for the live renderer. Its
  hall is the FactoryWarehouse Interior Modular Pack 1.0.1's (CyberneticWalrus, Unity Asset Store), without the
  Archviz Laboratory wall row, with a grey roller door; its props come from CyberneticWalrus's Warehouse Props Vol. 1
  (pallet rack, cable drums, shipping crate, small box, work table, wire cage and panel, site office, floor scale,
  cart with scanners) and Factory Props Vol. 1 (stacking boxes and bin, drawer cabinets, pallet) (Unity Asset Store;
  licensed for redistribution; rebuilt without Unity; the packs' author marks in-painted); the windows, the test
  tank, the pressure vessel, the crane, the bollards, the floor lines, the shelf items, the transport cases, the cable
  coil, the hose, the chain and the two pendant light bars are procedural (geometry and textures generated by code).
  The bench top is a butcher block cut from the ArchVizPRO Vol. 6 wood scan (`3D MATERIAL/Wood`, Unity Asset Store),
  colour-matched to the film's bench. `render/room/room.json` keeps only what the renderer reads (no `provenance`,
  `source_scene`, `unity` or `set` keys, no Unity names, ids or intensities); the attribution is this paragraph.
- **The photographs' setting**: the published quadruped assembly task's baked factory hall and bench-top maps, built
  from the FactoryWarehouse Interior Modular Pack 1.0.1 (CyberneticWalrus, Unity Asset Store) and the wall row of the
  Archviz Laboratory Pack 2.0 (Unity Asset Store; licensed for redistribution; rebuilt without Unity), the bench top
  Poly Haven's `oak_veneer_01` (CC0).
- **Software in the image**: MuJoCo 3.14.0 (Apache-2.0) with its Filament renderer (Apache-2.0) on Mesa's lavapipe
  and OSMesa (MIT), Python 3.12, numpy and scipy (BSD-3-Clause), OpenCV headless (Apache-2.0), trimesh (MIT),
  manifold3d (Apache-2.0).

## 6. Simulation limits (recorded, not hidden from maintainers)

- **Residual in-hand creep** (ruling M8): while the arm moves, a held unit slowly slides along the fingers' hinge axis
  by an amount that grows with the pad contacts' time constant minus the physics step (an empirical regularity of the
  gate trials; the mechanism inside the solver was not traced); with the cell's 0.5 ms step and pad time constant
  0.001 s it is 1.11 mm over 120 s of continuous carrying for the electronics enclosure and 0.155 / 0.195 mm over 20 s
  for a ballast weight, far inside every capture window. It is a property of the simulation, not of the station, and
  the agent's documents do not mention it.
- **Sleeping loose units**: loose units at rest are put to sleep by MuJoCo's sleeping (velocity tolerance 1e-5 m/s;
  the positioner and the robot never sleep), a simulation setting that keeps a control tick within its budget
  (rulings P08-1 / P14-1). A sleeping unit reads 0 N in the second hard rule until it is woken; a unit at rest only
  presses with its own weight, far below the rule's force.
- **RGB frames**: the Filament RGB frames of the agent cameras are not bit-for-bit reproducible between runs (in fresh
  processes up to about 0.5 % of a frame's pixels differ by a few levels, from the renderer's own parallel job
  system); depth, segmentation, physics and scores are bit for bit, and the grader never reads RGB.
- **The second hard rule's reading near its threshold** (ruling CAL-9 after the calibration episode of 2026-11-16):
  a program that keeps a unit pressed with the arm in full-stiffness position control (in that episode an open finger
  of a blind grasp landing on a thruster standing in its nest and held there for about 4 s) produces a sustained
  contact force on a plateau of about 150-152 N or 184-187 N, presumably where the arm's joint torque limits cap the
  press (not traced further). With `part_force_N` = 150 N over `part_force_window_s` = 0.2 s, whether such a run scores
  nothing can come down to millinewtons: in that episode one run sustained 149.993 N, 6.6 mN under the threshold, and
  was not zeroed; another sustained 151.40 N and was. The reading is deterministic (bit for bit between runs) and the
  rule, `describe()` and the cell manual agree, so this is a sensitivity of the reading, not a defect; the reference
  program presses with about 8 N at the same step. Moving the threshold off the plateau would change a hard rule
  (class S: a new selection, QC and validation); the owner chose not to.

## 7. Agent-visible text

The instruction and the cell's documents are generated from templates, the placeholder table (every printed number
bound to a file it is read from at build time) and 36 registered text overrides plus one generated-file override
(the caption of photo 14); the list is the development folder's `docs_src/text_overrides.json` (maintainers only).
The 36th (P34, ruling CAL-5) adds the tray board's axes to the cell manual's table "Datum boards" ("its x along the
tray's y, its y along the tray's -x", bound from the cell's fixture definition); sheet 7 of the drawings
(`trays_and_racks.png`) draws the board's own x and y from its origin since the same revision. Since P36 (ruling
CAL-12) the origin mark in the same sheet's "Enclosure rack, seen from the side" (the rack's x-z view) labels its
vertical arrow z; it had printed y. Since the agent-view change of 2026-11-18 the instruction, the cell manual and the
cell API describe the given render layer (the cell cameras' own Filament session for the agent's twin, and a view of
any scene in the same room), say that MuJoCo's own renderer draws a different picture than the cell's cameras, and
name the workshop instead of the factory hall.

## 8. Registered exemptions of the leak scan

What the agent sees passes the underwater tasks' shared word list. The registered exemptions (development folder,
`tools/leak_scan.py`, each with its ruling): the tool names MuJoCo and trimesh in prose; the `UR10e-class` wording and
the arm's folder, MJCF and mesh names; the two licences' copyright lines; the arm and old-gripper identifiers in the
vendored control library, its comments' physics constants and geometry conventions and its prose; the arm model's
upstream README; the key names `unity_name`, `anchor_unity_xz` and `floor_y_unity` of the earlier hall bundle's
`room.json` (the workshop bundle's `room.json` has none); the CPU feature
names in the Dockerfile's pinned numerics; the mesh library's stamp in the gripper STL headers; one `"model": "ur10e"`
value in `gripper_wide.py`; the sense-of-rotation wording of the propulsion drawing and the datasheet page.
