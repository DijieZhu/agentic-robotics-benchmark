# Write the motion controller for the lab's DA-6 desktop arm

You are the controls engineer of a small lab. The lab runs a DA-6 desktop arm (six bus servos: base yaw, shoulder
pitch, elbow pitch, wrist pitch, wrist roll, jaw) in three desktop cells:

- **shelf**: reach into the compartments of a shelf organiser and onto a fixture block with the bare gripper;
- **bins**: place a part that is clamped in the gripper into open bins and onto a fixture;
- **dispense**: run a dispensing needle, clamped in the gripper, along lines and arcs over a tilted board, and park
  it over a purge cup.

Your job is the arm's motion controller: the program that receives each job and drives the servos so that the tool
gets where it must, on time, without touching anything and without passing through singular configurations.

The arm is not available to you. You have only what the vendor and the lab hand over, in `/home/user/given/`
(start with `given/README.md`):

- `robot/`: the kinematics sheet (modified DH table, joint types, axis directions, limits, the TCP), the
  general-arrangement, gripper and needle drawings, the links' CAD (STL) and their collision envelope
  (`robot/cad/collision/`, the convex pieces on which contact and clearance are judged), and the mass table;
- `servo/`: the servo datasheet and the bus interface manual (the 10 ms bus cycle, registers, latency);
- `unit/`: the format of the unit's calibration record and an example record of another unit;
- `cells/`: the cells' documentation and the format of the scene description you receive at runtime, with examples;
- `acceptance/`: the lab's acceptance specification: the job types, tolerances, time targets and exactly how the
  acceptance run is scored;
- `controller/`: the controller interface, a minimal example controller and `check_controller.py`, a self-test of
  the interface against a toy plant.

No robot description file comes with the arm and no unit's true parameters are known: every unit differs from the
drawings within the tolerances the documents state. Each acceptance runs on a unit and in cells you have never seen;
the unit's calibration record, the scene of each cell and every job are given to your controller at runtime.

## What to deliver

`/home/user/submission/controller/controller.py` with `class Controller` (`__init__(workdir)`, `setup(unit, cell,
tool)`, `plan(job, state)` returning `"ok"` or `"infeasible"` (`state`: the bus readings where the arm stands),
`step(state)` returning the bus command), plus any files it needs, all inside `/home/user/submission/controller/` (at
most 80 MB). The interface, the budgets and the rules are in `given/controller/controller_interface.md`. The
controller runs on the lab's controller PC: an unprivileged account, no network, only the files of
`submission/controller/` (the documents are not there), Python 3.12 with the packages of this machine (numpy, scipy,
pinocchio, trimesh, ...).

## How it is accepted

The acceptance (`given/acceptance/acceptance_spec.md`) runs on three units of the arm. Each run gives the controller
one unit and the three cells in turn, about 14 jobs each, every job once, starting where the previous one ended. A
contact with anything stops the arm and fails the cell: that job and every later job of the cell score 0. Each job is
scored on what the arm actually did: the tool's position and direction, the needle's path, the time, the clearance to
the cell and the distance from singular configurations; refusing a job that can be done scores 0, and so does
attempting one that cannot.

## Notes

- Session time: you have 6 hours. Leave a working controller in `submission/controller/` well before the end; an
  episode cut by the time limit is not graded.
- Run `python3 given/controller/check_controller.py /home/user/submission/controller` before you finish: it checks
  that the controller imports, answers every call in the expected format and keeps the bus cycle's CPU budget. It does
  not know the arm and says nothing about how well the controller will do.
- You may build any model or simulator of your own from the documents; nothing of it is graded except the controller's
  behaviour at acceptance.
