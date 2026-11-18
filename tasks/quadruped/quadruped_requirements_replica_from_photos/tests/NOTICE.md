# NOTICE (grader-side; never staged into the agent's home)

## The reference unit

The 12-actuator, belt-reduced quadruped of this task is a hidden VARIANT of the ODRI Solo12 open quadruped of the Open
Dynamic Robot Initiative (New York University and Max Planck Gesellschaft), built on that robot's printed parts:

- `open-dynamic-robot-initiative/robot_properties_solo`, commit `2201f454f4caffa0cad307fa8cbd1d64dd844008`
  (the robot description: kinematic layout, joint arrangement and link conventions), BSD-3-Clause,
  "Copyright (c) 2019, New York University and Max Planck Gesellschaft." Full licence text:
  `LICENSES/robot_properties_solo.BSD-3-Clause.txt`.
- `open-dynamic-robot-initiative/open_robot_actuator_hardware`, commit `66af1522b4fba0ec4a1d7790e66f5e4652208d30`
  (the 12-DOF quadruped's mechanics and the belt-reduced 3-DOF leg: the printed parts' STL files and the STEP files of
  the foot, the pulleys and the motor shaft), BSD-3-Clause, "Copyright (c) 2019, New York University and Max Planck
  Gesellschaft." Full licence text: `LICENSES/open_robot_actuator_hardware.BSD-3-Clause.txt`.

**Modification statement.** The photos of the reference unit are rendered from the second repository's printed-part
STL and STEP files (the body structure of `mechanics/quadruped_robot_12dof_v1` and `_v1.1`, the leg of
`mechanics/leg_3dof_v1`, the foot of `mechanics/foot_contact_switch_v1`, the pulleys and the shaft of
`mechanics/actuator_module_v1`), placed on the first repository's kinematic layout and MODIFIED by the task's own
ground-truth generator (maintainer side, not published with the task): parts are cut at a plane and extended (the
trunk's centre plates, the upper-leg shells, the lower legs), moved or relieved, and the foot is scaled; printed
parts of the task's own design are added (offset hip adapters, pivot arms, stop blocks, spacer rings
on two legs, a nose box with its lid, a cradle, a mount plate). The maintainer's asset manifest lists every source
file used with its sha256 at the commits above. No source file ships in the task: neither the STL or STEP files nor
any mesh made from them is in the agent's files, the image, this folder or the task's asset dataset; only photos
rendered from the modified parts are. The architecture (a trunk with four identical 3-DOF legs, hip_roll axes along
the trunk, the hip_pitch module in the shoulder shell, the knee module in the thigh shell with a belt to the knee) and
the look of the printed structure are the source's; every dimension, stop, mass, housing class, the spacer rings on
two legs, the payload and the power system are drawn per instance and differ from the source's published values by
construction (the generator's laws keep every graded length and stop away from them). The joint and leg names are the
task's own (`hip_roll`, `hip_pitch`, `knee`; `fl`, `fr`, `rl`, `rr`). The source's published values are used only on
the maintainer side, for the leak scans and the recall dummy of the local checks.

Per the owner's rule the givens, the instruction and the public card never name the robot, its project, its
institutions or its suppliers; this file and the maintainer's own records carry the attribution.

## Vendors, documents and photos

The vendors (Wrenfield Drives, Quillon Motion, Ardent Cells, Torvik Controls), their part ids and every datasheet,
erratum, quote and lot-report number are fictional, and so is every marking on the boards, labels and packs in the
photos. `gt/catalogue.json` is the generator-side truth of the pages; no agent-visible code carries these numbers. The
pages and reports are synthesised from numbers by the task's `docs_synth_q` page renderers (matplotlib, DejaVu fonts).

The photos were rendered with Blender Cycles in a lab room built from the Archviz Laboratory Pack [2.0] (Unity Asset
Store, licensed by the maintainer; the pack is not redistributed, only photos rendered with it). Sky: 'Schadowplatz' by
Greg Zaal, Poly Haven, CC0 (polyhaven.com/a/schadowplatz). Bench top: 'Oak Veneer 01' by Jenelle van Heerden, Poly
Haven, CC0 (polyhaven.com/a/oak_veneer_01). The scale references, the kitchen scale, the pod's enclosure and the
service stand are generated from numbers by the task's bench scene.

Two sensors in the photos are rendered from their makers' own models; neither model file ships in the task, only
photos rendered from them, and neither product is named in an agent-visible file:

- the depth camera on the nose box: Intel RealSense D435, the mesh `realsense2_description/meshes/d435.dae` of
  `IntelRealSense/realsense-ros` (branch `ros2-master`),
  `https://raw.githubusercontent.com/IntelRealSense/realsense-ros/ros2-master/realsense2_description/meshes/d435.dae`,
  the maker's own model; in the asset manifest as `stage0:components/glb/camera_intel_realsense_d435.glb`
  (sha256 `c12af60a29b01d2398e760b1dac32146dc1b2f47f166b81dfb4c3052ca79a0cc`);
- the lidar in the customer's pod: Livox Mid-360, the maker's STEP model `mid-360-asm.stp`,
  `https://terra-1-g.djicdn.com/65c028cd298f4669a7f0e40e50ba1131/Mid360/mid-360-asm.stp`, the maker's own model;
  in the asset manifest as `stage0:components/glb/lidar_livox_mid360.glb`
  (sha256 `baba1f3855876339969ad9b2e431b595eacecbbf69abd2a68e260dd6827bd2db`).

The other bought parts in the photos (motors, belts, bearings, encoders, connectors, boards, wiring, packs, screws)
are modelled by the maintainer.

The verify stage carries byte copies of the benchmark's `shared/robotics_grader` kit and of two stdlib helpers of the
benchmark's arm tasks (the verify-start lockdown and the stage lock), in `stage_lock/`.
