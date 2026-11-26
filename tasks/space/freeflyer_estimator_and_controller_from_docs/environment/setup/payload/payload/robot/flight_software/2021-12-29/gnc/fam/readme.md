\page fam Force Allocation Module (FAM)

The force allocation module (FAM) converts the force and torques output by control into 
PM speeds and nozzle positions.

# Inputs

* `gnc/ctl/command`: the control command which the FAM follows, containing force and torque.

## Outputs

* `hw/pm/command`: The commands for the PM to execute to obtain the desired force and torque.

