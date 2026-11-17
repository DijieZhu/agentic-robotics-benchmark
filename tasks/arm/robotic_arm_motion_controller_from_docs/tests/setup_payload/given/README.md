# DA-6 desktop arm: documentation set

Documentation of the lab's DA-6 desktop arm (a five-axis arm with a gripper, six BUS-SERVO-12V-A bus servos), its installation in three desktop cells and the acceptance of its motion controller. Start with `acceptance/acceptance_spec.md` (what the controller must do and how it is accepted), then `controller/` (the interface), `robot/`, `servo/` and `unit/` (the arm, its servos and the bus, the unit calibration record) and `cells/` (where it works).

## robot/ (the arm: kinematics, drawings, link CAD, collision envelope, mass properties)

- `robot/README.md`: index of the folder
- `robot/kinematics_sheet.md`: DA-6 desktop arm - kinematics sheet
- `robot/mass_properties.md`: DA-6 desktop arm - mass properties
- `robot/cad/README.md`: the link CAD files, their frames and the collision envelope
- `robot/cad/base.stl`: link CAD of the base (STL, link frame)
- `robot/cad/forearm.stl`: link CAD of the forearm (STL, link frame)
- `robot/cad/gripper.stl`: link CAD of the gripper (STL, link frame)
- `robot/cad/jaw.stl`: link CAD of the jaw (STL, link frame)
- `robot/cad/shoulder.stl`: link CAD of the shoulder (STL, link frame)
- `robot/cad/upper_arm.stl`: link CAD of the upper arm (STL, link frame)
- `robot/cad/wrist.stl`: link CAD of the wrist (STL, link frame)
- `robot/cad/collision/README.md`: DA-6 desktop arm - collision envelope
- `robot/cad/collision/base.stl`: collision envelope of the base (convex pieces, STL, link CAD frame)
- `robot/cad/collision/forearm.stl`: collision envelope of the forearm (convex pieces, STL, link CAD frame)
- `robot/cad/collision/gripper.stl`: collision envelope of the gripper (convex pieces, STL, link CAD frame)
- `robot/cad/collision/jaw.stl`: collision envelope of the jaw (convex pieces, STL, link CAD frame)
- `robot/cad/collision/pieces.json`: index of the collision envelope's pieces (JSON)
- `robot/cad/collision/shoulder.stl`: collision envelope of the shoulder (convex pieces, STL, link CAD frame)
- `robot/cad/collision/upper_arm.stl`: collision envelope of the upper arm (convex pieces, STL, link CAD frame)
- `robot/cad/collision/wrist.stl`: collision envelope of the wrist (convex pieces, STL, link CAD frame)
- `robot/drawings/general_arrangement_front.png`: general arrangement, front view
- `robot/drawings/general_arrangement_side.png`: general arrangement, side view
- `robot/drawings/general_arrangement_top.png`: general arrangement, top view
- `robot/drawings/gripper.png`: gripper drawing: TCP and jaw
- `robot/drawings/needle.png`: dispensing-needle drawing: the needle tip relative to the TCP
- `robot/figures/zero_pose.png`: the zero pose with the base, DH and TCP frames

## servo/ (the BUS-SERVO-12V-A servo and the bus)

- `servo/README.md`: index of the folder
- `servo/bus_interface.md`: DA-6 desktop arm: bus interface manual
- `servo/servo_datasheet.md`: BUS-SERVO-12V-A smart serial-bus servo

## unit/ (the units: the calibration record of each unit)

- `unit/README.md`: index of the folder
- `unit/calibration_record.md`: DA-6 desktop arm: unit calibration record
- `unit/example_calibration_record.json`: the calibration record of a demonstration unit

## cells/ (the three desktop cells, the scene description, the clamped tools)

- `cells/README.md`: index of the folder
- `cells/cell_manual.md`: integration manual of the three desktop cells: objects, frames, scene description, survey tolerance, setup arguments, clamped tools (part, needle and holder, the jaw's pad gap), reach
- `cells/example_scene_bins.json`: scene description of an example bin placement installation
- `cells/example_scene_dispense.json`: scene description of an example adhesive dispensing installation
- `cells/example_scene_shelf.json`: scene description of an example shelf picking installation

## acceptance/ (the site acceptance test of the motion controller)

- `acceptance/README.md`: index of the folder
- `acceptance/acceptance_spec.md`: site acceptance test: the three runs, one on each unit (home pose, calls, failures, a contact ending the cell), the jobs, the measurements, the scoring, the gates
- `acceptance/example_jobs_bins.json`: job list of the example bin placement installation, with the cell's `tool`
- `acceptance/example_jobs_dispense.json`: job list of the example adhesive dispensing installation, with the cell's `tool`
- `acceptance/example_jobs_shelf.json`: job list of the example shelf picking installation, with the cell's `tool`

## controller/ (the controller interface, an example and the self-test)

- `controller/README.md`: index of the folder
- `controller/check_controller.py`: Interface self-test of a DA-6 motion controller: the protocol and the timing, nothing else.
- `controller/controller_interface.md`: DA-6 cells: motion controller interface specification
- `controller/example_controller.py`: Example for the DA-6 motion controller interface: it shows the calls and the conversions; it does NOT move the arm.
