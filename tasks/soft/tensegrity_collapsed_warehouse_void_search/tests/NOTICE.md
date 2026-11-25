# NOTICE (grader side; never staged into the agent's sandbox)

## The robot

The robot of this task is the open-source three-strut, cable-driven tensegrity rolling robot of the Yale Faboratory
(Rebecca Kramer-Bottiglio, Yale University) and the PRACSYS lab (Kostas Bekris, Rutgers University), the fourth
version of their three-bar platform: W. R. Johnson III, P. Meng, N. Chen, L. Cimatti, A. Vercoutere, M. Aanjaneya,
R. Kramer-Bottiglio, K. E. Bekris, "An Open-Source, Reproducible Tensegrity Robot that can Navigate Among
Obstacles", IEEE Robotics and Automation Letters (IEEE Xplore document 11474858; arXiv:2511.05798). Project site:
https://tensegrity.yale.edu/ . (The citation is the one of the sibling task `tensegrity_cad_design_from_photos`.)

In this task the robot rolls through a simulated ruin. Its body is the designers' own published whole-strut mesh,
three times; its mass, inertia, cable net and elastic constants come from the designers' own simulation model; its
controller, primitives and symmetry table follow the designers' control code. No part of the robot was modelled,
redrawn, rebuilt or completed for this task, and there is no hidden variant of it. The cables and the stretch sensors
are tendons of the simulation (a length between two attachment points); they, the sensor strips and the wires have no
geometry in the sources and are in no file and no camera frame of the task. The heaps on the open floor and the
dressing of the sites are scene objects made from the maintainers' art packs (below); none of them is a part of the
robot.

## Sources

sha256 of each file as taken from its source. The licence column states what the source itself says, as the
maintainers recorded it on 2026-11-17; nothing here is a legal opinion.

| # | Source | Taken from | File, sha256 | What the source says about its licence | Use in this task |
|---|---|---|---|---|---|
| 1 | the designers' whole-strut mesh | https://github.com/shiyoung77/tensegrity_perception , branch `ros`, commit `726ed401` | `mesh/Will_strut_scaled_centered.ply`, 21 243 165 bytes, `8dc4caff1efa26468660f30d5e2d91f7d927d6b287480ecd3b06b351f88a4614` (1 118 331 triangles, 494 edge-connected pieces) | the repository has no licence file; the `package.xml` of its `ros` branch says `<license>MIT</license>` | the robot's visual meshes for the search camera, `tests/voidcell/data/meshes/dec/` (five binary STL files, 6 640 320 bytes together, the mesh regrouped by material; table below) and their byte copy in the practice ground of the image; every measurement of `tests/voidcell/_lib/notes/sim_base_data/strut_geometry.json` and `collision_hull_elec.json` (the fitted collision bodies: a sphere of radius 17.500 mm on each end-cap cover, a cylinder of radius 20.000 mm on each motor housing, a convex hull of 31 vertices on the electronics enclosure; the hooks and the line guide the tendons attach to) |
| 2 | the designers' simulation model of this robot | the data set of "CableRobotGraphSim" (N. Chen, W. R. Johnson III, R. Kramer-Bottiglio, K. Bekris, M. Aanjaneya, L4DC 2026, PMLR 331:169-182, arXiv:2602.21331): its data-set archive `crgs.zip` (a Google Drive download) and its repository https://github.com/nchen9191/cable-robot-graph-sim (clone at commit `7ea068d4`) | `crgs_3_bar_dataset/dataset_0/3bar_new_platform_all_cables.xml`, `6fb60be75c5cc6ac295026c5f99c680507ed695948d7c140c5c236d1535b0d00`; the repository's `mujoco_physics_engine/xml_models/3bar_new_platform_all_cables.xml`, `cca986571d80096795924e9c6248df048d0c1b03c83f386b12c70e1fe5238e2b` (the default friction and contact values) | the repository has no licence file; the data set states no licence | the strut's mass (0.2429 kg) and inertia, the 15 tendons and which nodes each joins, the sensors' rest lengths, the tendons' stiffness and damping and the contact constants (carried from the model's decimetre-gram units to SI at Earth gravity by a stated similarity rule), the attachment sites as the reference of the hook-assignment rule. Kept as data in `sim_base_data/authors_model.json` and `params.json`; no file of the data set or the repository is shipped |
| 3 | the designers' control code, firmware and calibration | https://github.com/ChemStud24/tensegrity , branch `rss25` (commit `4b79620`): `src/run_tensegrity_Astar.py`, `src/symmetry_reduction_utils.py`, `onboard/onboard.ino/onboard.ino.ino`, `calibration/new_calibration.json` (`5ca8e4ab7cd217bc4d35a0f85235678412fd5512693f61b3a7ad0dff57fc8e5a`), `calibration/new_platform_primitive_dict.pkl` (`0655fdb90c041e6acfa2e1e9cb0eb2aa4e5e22317456ec02ca3c64626671b911`), `calibration/new_platform_transformation_table.pkl` (`a937e10e9f168fb86ee4ac83ce052249f32ae53a91799b54898693bb26cb0a9d`); branch `hybrid_mppi` (commit `d1e58a4`): the simulator's `pid.py` and `tensegrity_mjc_simulation.py` | as named | the repository has no licence file; its `package.xml` says `<license>TODO</license>`. A Zenodo deposit of its `master` branch (record 20347587) is marked CC BY 4.0 | the cable-length PID with its gains, limits and stop rule, the eleven primitives' target states, the symmetry mapping by the bottom triple, the rest phase between primitives, the slack rule, the 100 Hz rate, the telemetry's fields and units, the sensors' calibration constants (`authors_calibration.json`). The simulation base implements these after that code in its own files. The two pickled tables were read as data, opcode by opcode, without unpickling, and serve only the base's validation (`authors_tables.json`): the handbook's primitive table is this simulated robot's own measurement, not the designers' table |
| 4 | the designers' tracking configuration | the repository and branch of row 1 | `data_cfg.json`, `5e6db514515d42610f6ed6865a7403d39478e08c62cd7ffd64f896172584a393` | as row 1 | the search camera's intrinsics (607.686, 607.987, 326.550, 250.662 px at 640 x 480) and depth unit (4000 counts per metre), which the camera's data sheet prints as this unit's rated values; which end-cap colour belongs to which strut |
| 5 | the designers' bill of materials and assembly guide (version 3) | https://tensegrity.yale.edu/ | `BOM.xlsx`, `3dea5cc931d8758edf7f7f91b350ff498b703577aa277af90cf918284448c613`; `Guide_to_assembly_a_tensegrity_robot_V3.pdf`, `912c44ccae2bd04e7d734ca0d4771612b2ed40e37c9bd8a30f138cbe300de49e` | neither the site nor the files carry a licence or copyright statement | the vendors and part numbers of the three purchased parts the handbook names; the fishing line's rating; the guide's rule on which hook a line or a sensor may hang from (the base's hook assignment is one of the 64 assignments it allows). No text or figure of either is in the task |
| 6 | the gearmotor's data | Pololu, https://www.pololu.com/product/5192/specs | - | a vendor's public product page | gear ratio, no-load speed and current, stall torque and current, encoder resolution of the simulated motors |
| 7 | the stretch sensors' cut outlines | figshare data set "Impact-resistant, autonomous robots inspired by tensegrity architecture", DOI 10.6084/m9.figshare.32273583 | - | CC BY 4.0 (the data set's licence field) | a cross-check only: the hole spacings (65 and 130 mm) equal the sensors' rest lengths of row 2. No file of it is used |
| 8 | the designers' CAD package ("tribar-2" on GrabCAD), as downloaded by the benchmark's maintainers | https://grabcad.com/library/tribar-2 | - | GrabCAD's site terms: non-commercial internal use, no redistribution | **no file of the package is in this task**, and nothing of the task was derived from it |

### The visual meshes, file by file

The mesh of row 1, split into its edge-connected pieces without any repair, grouped into five materials and written
per group in the strut frame (`tests/voidcell/data/meshes/MESHES.json` holds this record). No piece was dropped or
reshaped; one group was decimated because its 1 061 293 triangles of small electronic parts are a few pixels in the
search camera's frames.

| File (`tests/voidcell/data/meshes/dec/`) | Content | Triangles | Bytes | sha256 | Decimation |
|---|---|---:|---:|---|---|
| `sock.stl` | the end-cap covers | 13 136 | 656 884 | `7e4aada84dcc6a82cea650d8aa76d4d4e9c3a4e147b41743bba9a75f04f140b3` | none |
| `endcap.stl` | the end caps | 12 680 | 634 084 | `74db392f73225147ed7d34fd676214840e3b35239836c56b545e6ad184fa6837` | none |
| `housing.stl` | the black printed parts | 31 078 | 1 553 984 | `9fb0e6ab23bcf7d15f73246d4b00816a68b304864d9d7357ffd7471751fa4f8d` | none |
| `rod.stl` | the aluminium rod | 144 | 7 284 | `65fd8a3767398062ad92881f948a118b4c24d4e08416ac771523df8d0b5d1e40` | none |
| `other.stl` | the electronics and hardware | 75 760 | 3 788 084 | `1849857316eb73acf7cfd1d3cb5fcf1b8f942dfac41848462de0de71546c4dd2` | `fast_simplification` 0.2.0, target reduction 0.94, aggressiveness 7 |

Hero and demonstration pictures of the robot, when they are made, use the full-resolution mesh.

## What is real and what this task adds

`README.md`, section 5, lists it item by item. In short: the robot, its off-board control from one fixed RGB-D
camera, its primitives and its way of navigating among obstacles are the designers'; the collapsed warehouse, the
camera on a mast, pose estimation from raw frames, the void out of the camera's view with one open mouth and two
plugged ones, the heaps on the open floor, the dressed picture with its light and litter, the 20 s dwell for an air
reading, the coloured chips, the noise figures and the time limit are this task's.
How far each primitive moves the simulated robot is a property of the simulation base, which carries declared
assumptions where no measurement of the real robot is public (the stop band, the hook assignment, the line's
stiffness, the rest between primitives); the designers' published primitive table is not reproduced by it and is not
given to the agent.

## Code and tools

* The simulation base `ysim` 0.3.1 (`tests/voidcell/_lib/ysim/`, locked by `ysim.lock.json`) was written for this
  platform's tasks. It runs on MuJoCo 3.14.0 (Google DeepMind, Apache-2.0): the physics; the classic renderer, off
  screen through Mesa's OSMesa (MIT), for the robot's depth and mask in the search camera's frames; and MuJoCo's
  physically based renderer (Google's Filament, Apache-2.0, as MuJoCo builds it in) on Mesa's software Vulkan driver
  lavapipe (MIT), for the robot's colour (`tests/voidcell/_lib/plate_robot.py`).
* The image installs from PyPI, by `environment/requirements.txt`: mujoco 3.14.0 (Apache-2.0), numpy 2.5.4
  (BSD-3-Clause and the licences of its bundled parts), scipy 1.18.1 (BSD), pillow 13.0.0 (MIT-CMU),
  opencv-python-headless 5.0.0.93 (Apache-2.0), pyyaml 6.0.3 (MIT).
* The decimation of `other.stl` used fast_simplification 0.2.0 (MIT); the mesh handling of the base's asset build
  uses trimesh (MIT); the calibration site's plan (`site_plan.png`) was drawn with matplotlib.
* `tests/cellkit/` is the benchmark's verify-stage process kit, byte-identical to the copy in
  `quadruped_hazardous_plant_inspection`; `tests/robotics_grader/` is byte-identical to `shared/robotics_grader/`.
  `tests/verify.py`, `tests/test.sh`, the practice service (`environment/practice_ground/practiced.py`), the
  Dockerfile and `setup.sh` are built on that task's files.
* Everything else is original to this task: the site generators (`tests/review/sitegen/`: `w2gen` for the open
  ground, `w3gen` for the slab, the three mouths, the void and the rubble under the slab, `w4gen` for the heaps;
  `tests/review/sitegen_cal/`) and the stored site packages, the calibration site, the cell (`tests/voidcell/` apart
  from the base; the plate backend `plate_*.py` with its look `agent_look.json` among it), the scoring and the debrief
  (`tests/voidscore/`), the reference program (`tests/reference/`), the documents, the client library and the
  example.

## The dressed sites: art packs, renderers and tools

The search camera's picture of a site is a plate computed offline (`README.md`, "What the camera's picture is made
of"). A site's plate bundle holds images and fields of that one camera view: the picture, its radiance, two depth
plates, the sun's share and visibility, the fog, a few small light probes. **No mesh, no texture and no other file of
an art pack is in the task's folder, in the image or in a plate bundle**, and no file name, folder name or metadata
value of a site package or of a plate bundle names a pack, a piece, a renderer or a tool. This file is the one place
of the task that names them.

The scene builder dresses the sites with pieces of art packs of the benchmark's maintainers, licensed for
redistribution:

| Pack | Form | What its pieces are used for |
|---|---|---|
| ScansFactory, "Warehouse - Abandoned Factory District" | the Unity pack (version 1.1), rebuilt without Unity, and the Unreal Engine project (UE 5.1+), exported with Unreal Engine 5.4.4 | photo-scanned pieces of the building: shell and trusses, walls, slabs, racking, ducts and cables, cargo, blocks and bricks, floors, scanned rubble, fallen roof; its ground scans are the loose-ground patches (at their native scale) and the scattered stones at the foot of the rubble heaps |
| ScansFactory, "Industrial Abandoned Buildings Modular Environment" | the Unreal Engine project (UE 5.1+), exported with Unreal Engine 5.4.4 | blocks, broken concrete, a slab, scanned rubble; the bricks of the rubble heaps |
| "Mud Eroded Environment" | an Unreal Engine project | one scanned mud pile, cut and scaled down: the mounds of fill |
| CyberneticWalrus, "Warehouse Props Vol 1" and "Factory Interior Props Vol 1" | Unity packs, rebuilt without Unity | cargo and racking |
| "Post Apocalyptic Destroyed Buildings" (version 1.4) | a Unity pack, rebuilt without Unity | wall pieces, broken concrete, fallen roof and broken steel |
| "Chernobyl" (version 1.1) | a Unity pack, rebuilt without Unity | blocks and slabs |
| KitBash3D, "Refineries" | tiling materials | floor materials |

The heaps are pieces of the maintainers' heap library, built from the scans above (a mound of fill is one scanned
mud pile, cut and scaled; a rubble heap is scanned bricks set into a pile over scanned ground stones); each piece's
mesh is the source of the picture, of the depth and, as a height field, of the collision (`README.md`, section 8).
The thin litter, the dust in the air and the light are the scene builder's own.

Renderers and tools of the plates, none of which is in the task or in the image: the picture and its light passes
are path traced with NVIDIA's OVRTX renderer (Omniverse RTX) from scenes written as USD; the depth plates are
rendered with MuJoCo's classic renderer on the task image's own Mesa, from a two-sided copy of the dressed scene; the
light probes are filtered with `cmgen`, a tool of Google's Filament (Apache-2.0). Which renderer, version and machine
made each file of a bundle is in that bundle's provenance record, which the maintainers keep outside the bundle.

## Names

By the maintainers' rule the instruction, the payload, every reply of the station, the practice ground's debriefs and
everything the agent's account can read in the image name neither the robot nor its designers, their schools, labs,
papers, design names or repositories. The documents call the robot the "RT-3 search tensegrity"; the three purchased
parts keep their real vendors and part numbers (Pololu 5192; McMaster-Carr 8974K19; Arduino Nano 33 IoT, ABX00027,
with its STMicroelectronics LSM6DS3). The cell's files do carry such names (the base's notes and provenance strings),
in `tests/` and in the practice ground's byte copy of the cell, which only the practice service's own account can
read. The art packs, the heap library's pieces and the renderers of the plates are named nowhere but here: not in
the instruction, the payload, a site package, a plate bundle or the asset list. This file carries the attribution.

## Distribution

Two things of this task are public.

In the benchmark's repository, the task's folder: the five mesh files above (pieces of the mesh of row 1, four
groups at full resolution) twice, under `tests/voidcell/data/meshes/` and under
`environment/practice_ground/voidcell/data/meshes/`; the tables measured on that mesh; and the data taken from the
sources of rows 2 to 4. The mesh is also the source of the geometry that the sibling task
`tensegrity_cad_design_from_photos` grades. The sources of rows 1 to 3 carry no licence file. The benchmark's
maintainers decided for version 1.1 to publish the folder with these files, as they did for the sibling tasks, which
use the same mesh; version 2.0 adds no file of the robot's sources to them.

In the benchmark's asset dataset, the plate bundles: the practice ground's four as files under this task's
`payload/` prefix (listed in `environment/assets.sha256` and fetched when the image is built), the hidden sites' ten
as one tarball under its `verifier/` prefix (pinned in `tests/grader_config.json` and fetched by the grader). They
are images and fields computed from the dressed scenes, one camera view per site; they hold no mesh and no texture
of an art pack. The bundles are not in git. The benchmark's maintainers decided to publish them there: the image
build and the grader fetch them at the dataset commit that `environment/assets.sha256` and
`tests/grader_config.json` pin, and a local copy `tests/assets/plates.tar.xz` (not in git), when it exists, replaces
the grader's download (`README.md`, section 2).

If a later decision changes what may be shipped, this file changes with it.
