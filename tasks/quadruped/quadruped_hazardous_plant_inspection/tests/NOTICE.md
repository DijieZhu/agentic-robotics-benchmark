# NOTICE (grader-side; never staged into the agent's home)

The quadruped of this task is a locomotion model of the open twelve-actuator torque-controlled quadruped of the Open
Dynamic Robot Initiative (the Solo12 class; https://github.com/open-dynamic-robot-initiative/robot_properties_solo,
commit 2201f454f4caffa0cad307fa8cbd1d64dd844008, BSD-3-Clause, Copyright (c) 2019, New York University and Max
Planck Gesellschaft): link frames, joint axes, masses, centres of mass and inertia tensors of its 13 links, the IMU
offset, the motor gear ratio and rotor inertia were taken from that repository's description files, and its visual
meshes (base, hip modules, upper and lower legs) are used unmodified under neutral file names. The actuator limits
follow the same initiative's actuator documentation (open_robot_actuator_hardware, BSD-3-Clause) and Grimminger et
al., arXiv 1910.00093. The joint friction, damping and reflected inertia are the series' own (the quadruped assembly
task's cell). The sensor head, the lidar mount, the payload and the robot's collision primitives are this task's own.

The robot's on-board gait (the firmware the cell runs; the agent only commands it) is a learned controller trained for
this task with brax's PPO (Apache-2.0, used as a library) in an environment whose structure follows MuJoCo
Playground 0.2.0's quadruped joystick environment (DeepMind, Apache-2.0); its gait clock, commands, observations,
rewards, constraints, joint PD loop, mode presets, state machine and protective behaviours are this task's own. No code
or weights of published controllers of this robot were used.

Everything else is original to this task: the plant site generator and its stored site packages, the instruments and
their faces, tag plates and legends (drawn at generation time, with Pillow and, for the plates' text since version 2,
the task's own lettering), the dose field and the dosimeter model,
the cell (`tests/plantcell/`), the reference program and its readers (`tests/reference/`), the documents and the
instrument catalogue. The verify kit (`cellkit/`, `robotics_grader/`) comes from the benchmark's shared code.

The buildings' look in the robot's cameras (since version 2) comes from the render bundle `qinspect_look_v1`: 62
files (21 surface textures, 10 prop meshes in 12 mesh files with 13 albedo maps, 12 safety signs, 4 effect maps)
and a manifest of their sha256. The task image fetches it from the benchmark's asset dataset at the revision pinned in
`environment/assets.sha256` and keeps it in a folder that only root and the practice ground's account can read. Its
textures, meshes and signs derive from the maintainers' licensed art library: the packs "Chernobyl" 1.1 and
VP.Studio's "Nuclear power plant" 1.0 (Unity), KitBash3D's "Refineries 4K", CyberneticWalrus's "Factory
Interior + Props Vol 1" and "Warehouse Props Vol 1" and ScansFactory's "Warehouse - Abandoned Factory District"
1.1 (surface textures re-toned, one prop mesh decimated); the four effect maps are made by the bundle's builder. Apart from this
bundle, no third-party plant model, drawing or dataset is used.

The givens, the instruction, every reply of the cell and the public card name neither the robot the model follows nor
its makers, the training tools or the benchmark; this file carries the attribution.

## Licence of the robot description (BSD-3-Clause)

This notice covers every copy of the robot description's files that the task ships: the model and meshes in
`tests/plantcell/data/robot/` and their byte copy in the practice ground of the task image
(`environment/practice_ground/plantcell/data/robot/`, readable by the practice service's account only). Those copies
carry no licence file of their own, so that nothing the image holds names the robot's makers; this file, distributed
with the task, reproduces the copyright notice, the conditions and the disclaimer.

```
BSD 3-Clause License

Copyright (c) 2019, New York University and Max Planck Gesellschaft.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```
