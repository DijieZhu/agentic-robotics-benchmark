# NOTICE (grader-side; never staged into the agent's home)

Vehicle geometry, textures and part data: the benchmark owner's own tenth-scale car model and the CAD resources it was built from; the owner confirmed they may be redistributed with this task.

The cell's hidden kit (`tests/gt/`, unpacked at verify start from the asset dataset, or from a maintainer's local copy
`tests/assets/gt.tar.xz`) holds the car's visual meshes (decimated from the model, two product-label textures replaced
by unlabelled copies) and the simplified, subtract-only collision pieces made from the model's dimensions; the
differences between collision and appearance are documented to the agent in the cell manual.

The arm and the gripper are the MuJoCo Menagerie models `universal_robots_ur10e` (BSD-3-Clause) and `robotiq_2f85`
(BSD-2-Clause), commit c96a32d28fb5da84da38c1da4d749e7a13212855, with their licences in `given/robot/` (the files of the
benchmark's quadruped assembly task, `quadruped_full_assembly_12dof_ur10e`, unchanged); the cell mounts the gripper on
the real flange face (`arm/assembly_quad/ur_robot.py`) and drives the arm with its own computed-torque controller
(`ur_controller.py`). The cell's structure (a workbench with the base on a stand, the other parts loose on the bench or
in a tray, an industrial arm that builds the product, a coupling state machine per slot) follows the benchmark's
assembly tasks; their grader modules `arm/assembly_cell`, `assembly_info`, `assembly_six`, `assembly_quad`, `render`,
`render_filament`, `room`, `common`, `robotics_grader` are vendored here byte for byte, and `netguard.py` is the copy of
`driving_safety_supervisor`'s.

The agent cameras' scenery is the quadruped assembly task's baked factory hall and bench-top maps, unchanged (built
from the FactoryWarehouse Interior Modular Pack 1.0.1 (CyberneticWalrus) and the wall row of the Archviz Laboratory
Pack 2.0 (Unity Asset Store; licensed for redistribution; rebuilt without Unity), the bench top Poly Haven's
oak_veneer_01 (CC0)).

The givens, the instruction and the public card name neither the car platform the model follows nor its vendors or
the benchmark's model sources; this file carries the attribution.

## The car and the makers' CAD files in it

The task's car is the benchmark maintainer's own Blender model of a tenth-scale autonomous race car of the F1TENTH /
RoboRacer kind (final scene `car model_6.blend`; its part-level scene `car model_2.blend`, from which every part is
taken), the same model as that of the race car's CAD-stage task `racecar_cad_design_from_requirements`. The model is used
as it is: no vertex of it is changed. What this task derives from it (the six part files, the photographs of the picture
manual, the outlines of the drawings, the hidden kit's visual meshes and the collision pieces made from the model's
dimensions) is computed from the model by rule.

The model contains these makers' CAD files, which the maintainer placed in it; this task's units derive from them as
listed:

| What | Source file in the model's `Resources/` | In this task |
|---|---|---|
| The rolling chassis: the maker's CAD of a Tamiya TT-01 Type E four-wheel-drive touring car chassis, with its wheels and its 7.2 V NiMH stick battery | `Chassis TT-01 Type E.stp` | `roller.stl` (the chassis tub with drivetrain, suspension, steering and the kit's electronics, the body posts, the four wheel hubs), `wheel.stl`, `battery_pack.stl`; their meshes in the kit and in the photographs |
| The compute module: an NVIDIA Jetson Xavier NX developer kit (module and carrier board), scaled by 0.886 in the model | `Top3518_3668_A02_3509_A01_0204.stp` (with the module alone in `P3668_3D_CAD_STEP_Model.stp`) | `compute_module.stl`; its meshes in the kit and in the photographs |
| The LiDAR: a Hokuyo UST-10LX, scaled by 0.816 in the model | `Hokuyo UST-10LX.SLDPRT` / `.fbx` | `lidar.stl`; its meshes in the kit and in the photographs |
| The platform deck: a platform deck part (`Platform_Deck_V3`), scaled non-uniformly (0.817, 0.894, 0.817) in the model | `Platform_Deck_V3.SLDPRT` / `.fbx` | the deck plate of `deck_assembly.stl` |
| The antenna mount (scaled by 0.906) and the two WiFi antennas (scaled by 0.581) on the deck | `antenna_mount.SLDPRT` / `.fbx`, `WiFi Antenna.STEP` | parts of `deck_assembly.stl` |

The motor controller and the power distribution board on the deck assembly are textured boxes in the model
(photographs of real boards on thin boxes). The rear wing frame on the rear body posts, the deck standoffs and the
spacer pins are modelled primitives.

## Where the files are, and their terms

The large givens (the units' part files, the drawings, the picture manual, the arm and gripper models and the cameras'
scenery) are served by the benchmark's asset dataset `grounded-intelligence-initiative/ale-robotics-assets` under
`tasks/ground/racecar_full_assembly_tenth_scale_ur10e/payload/`, at the commit pinned in `environment/assets.sha256`;
the hidden kit `gt.tar.xz` is `tasks/ground/racecar_full_assembly_tenth_scale_ur10e/verifier/gt.tar.xz` there, at the
commit pinned in `grader_config.json` (`hidden_gt.url`). The photographs are renders of the model above, the part files
and the kit's meshes are decimated from it, and what derives from the makers' CAD files in the table above keeps their
makers' terms. The arm and gripper models keep their BSD licences (their LICENSE files next to them); the cameras'
scenery keeps the terms named above. The dataset card's licence note says the same.
