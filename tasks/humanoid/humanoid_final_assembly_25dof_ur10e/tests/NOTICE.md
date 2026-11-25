# NOTICE (grader-side; never staged into the agent's home)

The robot built in this task is Asimov 1 by Menlo Research (repository menloresearch/asimov-1, commit
55fca52; hardware under CERN-OHL-S-2.0, software under GPL-2.0). Every module the cell
handles is made of the makers' own CAD (their full-body STEP), grouped by the bodies of the makers' simulation model.
The picture meshes differ from the CAD in these ways only: marks are levelled on their own surfaces, broken faces are
triangulated again on their own surfaces, and bought parts are split along their real moving interfaces. The robot is
not changed and no variant of it is built. The givens, the instruction and the public card name neither the robot that
is built nor its makers nor a vendor of its parts (the arm's and the gripper's own model folders carry their makers'
names and licences, as the next paragraph says); this file carries the attribution.

The actuators are units of ENCOS; the holding friction the cell simulates is a conservative reading
of their published test data. The arm is the model library's universal_robots_ur10e (BSD-3-Clause) and the
gripper is the benchmark's wide gripper model (BSD-2-Clause); the makers' marks on their meshes are
levelled in every copy this task uses, and their licences are in the given folder of the arm's and the gripper's models.

## What differs from the makers' assembly manual

- The leg's interface: the makers mount the leg's hub on the hip-roll output; this station mounts the leg with its
  hip-roll unit on the hip-pitch output, because the leg with its shells cannot be brought onto the makers' interface
  as a rigid module.
- The arm's threaded plates are fitted in the torso before this station; the torso and the pelvis arrive joined, with
  the battery and the covers.
- The head arrives as a module that ends at the neck-yaw output; the makers' own head step is further up.
- The 41 interface screws are not in the kit:
  - leg_R: U1028, U1095, U1096, U1097, U1098, U1099
  - leg_L: U1115, U1116, U1117, U1118, U1119, U1120
  - arm_R: U0869, U0875, U0876, U0877, U0878, U0879, U0880, U0881, U0882, U0883, U0884, U0885
  - arm_L: U0817, U0818, U0819, U0820, U0821, U0822, U0823, U0824, U0825, U0826, U0827, U0828
  - head: U1161, U1162, U1163, U1164, U1165
- The arms are delivered at joint angles that differ from the CAD's stored pose; the hip roll and hip yaw ranges are
  the CAD's free travel less a margin, narrower than the makers' simulation model.
- The right shoulder's shell holes are turned 5.53 degrees against the threaded holes in the
  makers' CAD; the CAD is used as it is.
- The stand, the racks, the nest, the camera portal, the datum boards and the gripper's fingertips are station
  equipment of this task.

## Where the kit, the room and the scenery come from, and their terms

The kit: every module the cell handles is made from the makers' full-body CAD named above, through the benchmark's picture meshes of it, and keeps the makers' terms named above; the surface finishes of those picture meshes are procedural or built on CC0 scans from ambientCG. The stand, the racks, the nest, the camera portal, the datum boards and the fingertips were made by code for this task. The room: the cameras' frames show the cell in a baked room package (meshes, light maps, a generated floor detail texture and an image light) made from the benchmark's own workshop bay scene under that bay's own lights. The scenery of that scene: its shell, its services and its work fixtures were made by code for the benchmark; its furniture and props come from art packs supplied by the benchmark's owner (recorded as office_police_station, factory_interior_props_bundle, archviz_laboratory, house_props_furniture_vol2 and warehouse_props_vol1; Unity format, read without Unity) and from Poly Haven (CC0); its surface textures are CC0 sets from Poly Haven and ambientCG and material sets of the owner's ArchVizPRO Interior volumes; its sky is a Poly Haven image (CC0) with the passers-by taken out. The terms: the CC0 material carries no condition; the owner's packs are used as the owner supplies them and are licensed for redistribution with the benchmark; the arm and gripper models keep the licences named above.
