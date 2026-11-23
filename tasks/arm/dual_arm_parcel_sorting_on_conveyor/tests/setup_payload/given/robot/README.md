# The robot's model files

MuJoCo models (MJCF) of the station's dual-arm robot with the maker's data, for your own kinematics, collision checks
and simulation. They hold no station and no parcel (the station's geometry: `../station/`).

| file | content |
|---|---|
| `dual_arm_with_grippers.xml` | the robot: the arm model with a gripper attached at each wrist flange site (its coupler joints locked: each finger is one rigid linkage driven by its driver joint), the tool centre point sites, the finger pads, the palm cameras flush in the grippers' palms (no housing, no collision shape), the head camera's housing, the cameras' drawing poses and data-sheet fields of view, and a keyframe `start` (the start posture of `describe()["robot"]["start_posture"]`) |
| `dual_arm.xml`, `meshes/*.STL` | the arm alone (column, head, two arms of seven joints; no gripper) |
| `gripper_2f85/` | the gripper alone: the MuJoCo Menagerie model of the Robotiq 2F-85 (`2f85.xml`, `assets/`), BSD-2-Clause (`gripper_2f85/LICENSE`, copyright 2013 ROS-Industrial): its geometry and kinematics as published, its dynamic values the station's nominal data (below) |

What the files are exact in, and what they are not:

* Exact: the kinematics (bodies, joint axes and ranges, the flange and tool centre point sites), the collision shapes,
  the torque limits (`forcerange`) and the meshes.
* Nominal: every link's mass and inertia and every joint's damping, friction loss and armature are the maker's
  nominal data; the station's robot differs from them within the tolerances of `cell_manual.md` section 4.
* Not in the files: the drives' servo gains. Each arm actuator here is a torque input (newton metres, limited to the
  drive's torque limit), not the station's position servo (`cell_manual.md` section 4 describes the servos).
* The finger pads' friction coefficient here is the middle of the data sheet's range (`cell_manual.md` section 5).
* The cameras' poses are the installation drawing's and their fields of view the data sheet's; the installed cameras
  differ within the tolerances of `cell_manual.md` section 6.

Frames and names:

* World frame: the column's base at the origin, z up, the robot facing +x (the belt runs along y in front of it).
* Joints `Left_<j>_Joint` and `Right_<j>_Joint` with `<j>` = `Shoulder_Pitch`, `Shoulder_Roll`, `Shoulder_Yaw`,
  `Elbow_Pitch`, `Wrist_Yaw`, `Wrist_Pitch`, `Wrist_Roll`: the order of every 14-vector on the wire is left arm then
  right arm, each in this order. The actuators carry the joints' names.
* The right arm is a mirror image of the left about the x-z plane: a mirror-symmetric posture has the same
  `Elbow_Pitch` angle on both arms and the opposite angle at every other joint (the ranges are not symmetric).
* Sites `Left_flange` and `Right_flange`: the wrist flanges, z axis = the tool axis. The grippers are attached there,
  their bodies, joints and geoms prefixed `L_` and `R_`. Sites `Left_tcp` and `Right_tcp`: the tool centre points
  between the finger pads (`describe()["robot"]["T_flange_tcp"]`).
* Cameras: `head_cam` (the `head` camera), `Left_wrist_cam` and `Right_wrist_cam` (`wrist_left`, `wrist_right`: palm
  cameras, flush in each gripper base's palm between the finger roots, looking along the tool axis; no housing and no
  collision shape of their own); the head camera's housing `head_camera_housing` on the column's body. MuJoCo cameras
  look along their -z axis with +y up; the cell publishes every camera pose in OpenCV axes (`describe()["cameras"]`).
* Geom groups: 1 robot visuals, 2 gripper visuals (no contact), 3 collision shapes, 4 sites.
* Units: metres, radians, kilograms. The timestep is the cell's physics step.

The `<option>` block is the arm model's own with the cell's `impratio` (its timestep is the cell's physics step); the
cell's simulation of the station and the parcels (the belt's drive, the station's and the parcels' contact
parameters) is not in these files.
