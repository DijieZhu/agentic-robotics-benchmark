# NOTICE - source of the ground truth of this task (tests / solution only; never shown to the agent)

The photographed robot of this task (its ground truth in `reference/`, the oracle's inputs in `../solution/`) is a
**modified derivative** of the open-hardware 12-DoF quadruped **Solo12** of the Open Dynamic Robot Initiative:

- printed parts, assembly and actuator design: https://github.com/open-dynamic-robot-initiative/open_robot_actuator_hardware,
  commit `66af1522b4fba0ec4a1d7790e66f5e4652208d30` (`mechanics/quadruped_robot_12dof_v1/`, `mechanics/leg_3dof_v1/`,
  `mechanics/actuator_module_v1/`, and the core parts of `mechanics/biped_6dof_v1/cad_files/STEP/biped_6dof_v1.STEP`);
- kinematics and inertial data: https://github.com/open-dynamic-robot-initiative/robot_properties_solo, commit
  `2201f454f4caffa0cad307fa8cbd1d64dd844008` (`src/robot_properties_solo/resources/xacro/`).

What was derived and how it was modified: the thirteen printed part designs were re-modelled as parametric B-rep solids
from the published STL meshes and placed with the published assembly; the bought actuator core and the foot pad were
re-modelled as stand-ins with fictional part numbers. The robot was then modified by this task's generator: changed leg
segment lengths and body length, lightening cut-outs, a new colour, and photographed with the benchmark's render layer
(Blender Cycles). The task never names the source, and the names of the copyright holders and contributors are not used to
endorse or promote it. Both repositories carry the following licence, reproduced here as it requires:

```text
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

The source repositories' copyright statements also read "Copyright (c) 2019-2021, Max Planck Gesellschaft and New York
University" (open_robot_actuator_hardware README) and "Copyright(c) 2018-2021 Max Planck Gesellschaft, New York University"
(robot_properties_solo readme).

Other material:
- the photo scenes use procedural, self-made assets (bench, cutting mat, checker targets, rulers, ID-1 card, paper tags)
  inside a lab room built from the project's licensed Unity asset pack (Archviz Laboratory Pack); the Cycles world light is
  the HDRI "Schadowplatz" by Greg Zaal (Poly Haven, CC0), as shipped in that pack. Only the rendered photos are distributed.
- the public-model digest list (`given/public_blocklist.json`) holds sha256 digests of the source repositories' mesh and
  CAD files and vertex fingerprints of the re-modelled nominal parts; it contains no geometry.
- FreeCAD 1.1.4 (LGPL-2.0-or-later, https://github.com/FreeCAD/FreeCAD) is installed unmodified in the task image from its
  release AppImage, with the FreeCAD MCP server neka-nat/freecad-mcp @ `78bf023` (MIT, https://github.com/neka-nat/freecad-mcp)
  vendored unmodified (`environment/vendor/freecad-mcp/`).
