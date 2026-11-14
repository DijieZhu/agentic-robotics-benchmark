# NOTICE (grader-side; never staged into the agent's home)

The quadruped of this task is a twelve-actuator robot modelled on an open quadruped of the Open Dynamic Robot Initiative
(https://github.com/open-dynamic-robot-initiative, BSD-3-Clause; the actuator module after
https://github.com/open-dynamic-robot-initiative/open_robot_actuator_hardware, BSD-3-Clause): the nominal design of a
parametric quadruped generator. Its units are built from the generator's solids (printed parts, the actuator module
`VK-BDJ-0945`, the foot); the kit (`tests/gt/`, fetched from the asset dataset at verify start) and its carved collision
pieces were made with CoACD (MIT) and manifold3d (Apache-2.0). The actuator's datasheet, drawing and STEP files are the
generator's documents.

The arm and the gripper are the MuJoCo Menagerie models `universal_robots_ur10e` (BSD-3-Clause) and `robotiq_2f85`
(BSD-2-Clause), commit c96a32d28fb5da84da38c1da4d749e7a13212855, with their licences in `given/robot/`; the cell mounts
the gripper on the real flange face (`arm/assembly_quad/ur_robot.py`) and drives the arm with its own computed-torque
controller (`ur_controller.py`). The cell's structure (a workbench with the base on a fixture or stand, every other part
loose on the bench, an industrial arm that builds the robot, a coupling state machine) follows the benchmark's arm
assembly tasks (`robotic_arm_full_assembly_so101_panda`, whose grader modules `arm/assembly_cell`, `assembly_six`,
`assembly_info`, `render`, `render_filament`, `room`, `robotics_grader` are vendored here).

The agent cameras' scenery is the factory hall built from the FactoryWarehouse Interior Modular Pack 1.0.1
(CyberneticWalrus) and the wall row of the Archviz Laboratory Pack 2.0 (Unity Asset Store; licensed for redistribution;
rebuilt without Unity), the bench top Poly Haven's oak_veneer_01 (CC0).

The givens, the instruction and the public card name neither the robot the design follows nor the benchmark, the model
collection or the actuator vendor; this file carries the attribution.
