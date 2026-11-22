# Sources of humanoid_leg_cad_design_from_photos

This note says where the task's robot leg and its other material come from. The agent never sees this folder; everything
the agent sees uses generic names and invented part numbers only.

## The robot

The leg of this task is the right leg module of Asimov 1, an open-source humanoid robot by Menlo Research, as published
in the public repository `menloresearch/asimov-1` on GitHub (commit `55fca52b9f82906c00cc70738ca8a38e4be3caaa`). The
repository states its licences in `HARDWARE-LICENSE.txt` (CERN-OHL-S-2.0, the mechanical design) and
`SOFTWARE-LICENSE.txt` (GPL-2.0). The task uses the authors' full-body STEP assembly (sha256
`35903b20bffd0e21d709acc75a30d0ee98323fa6c76b59fefffd181471422ba7`) and their single-part STEP files of the leg module.
The CAD is used as it is: no hidden variant, no redesign, no rebuilt geometry, no stand-in for any component.

What the task derives from the authors' CAD:

| Derived material | Where | How |
|---|---|---|
| The true solids of the 19 made parts | `reference/parts/` (and, byte-identical, `../solution/reference/parts/`) | Each solid of the authors' CAD at the pose it has in the full-body assembly, moved with the whole leg by one translation into the frame of the given shell file. The treatments below apply. |
| The given components (10 types, 19 pieces) | the payload: `given/` | The authors' own solids, each type written to one file in a documented frame, with neutral product names and file headers. The joint actuators are the envelope solids of the authors' files (their detailed models are loose faces without a volume). The outer thigh shell is the authors' solid as it is, see below. |
| The photographs | the payload: `photos/` | Renders of the authors' CAD: the leg complete in three poses, taken apart stage by stage, and each made part on a cutting mat. |
| The documents (`given/thigh_shell/README.md`, data sheets, drawings, `parts_list.md`, `hardware_sheet.md`) | the payload | Written for this task from the measured geometry; none is a copy of a maker's document. |

Treatments of derived copies; the authors' files themselves are never modified:

- **One part united from three solids.** The printed steel part with the tag P11 is three solids in the authors' file
  that touch face to face; the real part is one printed piece. The reference holds the union of the three as one solid:
  the outer surface is unchanged, and the faces between the three solids are gone. This is the only boolean operation
  applied to the authors' geometry in this task.
- **The two bars graded on their fabrication shape.** The bars with the tags P17 and P18 have plain bores in the
  authors' part files and cut threads in the full-body assembly. The reference holds the plain shape of the part files,
  because the threads are inside blind bores that no photograph shows; the outline is the same.
- **One part taken from the assembly.** The authors' single-part file for the printed steel part with the tag P06 holds
  the left leg's part; the reference uses the right leg's solid of the full-body assembly.
- **The given shell is damaged, and is given as it is.** The authors' solid of the outer thigh shell is closed, but the
  trimming of some of its faces is broken: a band along its upper rim does not display in FreeCAD, and its volume and
  inertia read wrongly. No face is repaired, removed or re-fitted. The solid is carried as an OpenCASCADE BREP file
  because a STEP round trip turns it into another shape, and it is moved by a pure translation, so that the origin of
  the file is the corner of the shell's bounding box with the smallest coordinates, rounded to whole millimetres; the
  axes stay parallel to the leg's own. The grader treats the shell as a surface without a volume.
- **What the authors' CAD has of imperfection stays.** The lower rod ends are not in line with their bars, the bars
  close their loops with a small residual, and some neighbouring solids overlap slightly. None of this is corrected; the
  grader exempts the reference's own overlaps pair by pair, and its bands are wider than these residuals.

Screws, nuts, washers, threaded inserts (199 units in the leg module) and the wiring are in neither the given files nor
the reference. The inserts stay in the printed parts in the photographs, as they do on the real leg. A hole of the
reference that carries an insert is graded on position and direction only, because the insert hides its wall.

A per-face geometric lettering check of every given solid and of every made part of the reference found no lettering:
the leg module carries no maker's or product name in its geometry.

## Masses

The grader's masses are physical ones, not those of the authors' simulation model. Made parts: the volume of the
reference solid times the nominal density of its material (`parts_list.md`). Joint actuators: the catalogue masses of
the units in the public data sheet of the actuator maker, ENCOS (434 g, 485 g and 382 g for `RJ-5378`, `RJ-5669` and
`RJ-5660`). Pillow blocks, D-shaft and set-screw collar: catalogue weights. The masses of three given components are
estimates, flagged `mass_is_estimate` in the reference: the rod end and the flanged ball bearing (the volume of the
model as steel) and the outer thigh shell (a volume estimate of the damaged solid times the density of its material).
The data sheets say which masses are computed.

## Documents written for this task

The data sheets and drawings of the given components are ours. The mounting holes of the joint actuators are not in the
envelope solids; the hole groups printed in the sheets of `RJ-5669` and `RJ-5660` were read from the authors' detailed
models and are listed only where they meet the holes of the made parts they are bolted to: the housing groups hole for
hole, the output flange groups up to a turn about the axis, because the output stands at a joint angle in the authors'
assembly. The detailed model of `RJ-5378` does not hold its mounting holes; its sheet carries no hole table, and no
pattern was derived from the parts it is fastened to. No torque, speed or gear ratio is given.

## Invented names

The part numbers in the agent's files (`RJ-5378`, `RJ-5669`, `RJ-5660`, `RE-06M`, `PB-12`, `PB-06`, `DS-0650`, `SC-06`,
`FB-23`), the paper tags and the components' generic names are invented for the benchmark. No robot, product or maker
name appears in any file the agent sees, in the file headers, in the photographs or on the given solids.

## Third-party software

- FreeCAD 1.1.4 (LGPL-2.0-or-later, https://github.com/FreeCAD/FreeCAD) is installed in the task image from the official
  release AppImage, pinned by sha256 in `environment/Dockerfile` and extracted there. It is not the complete release:
  nine workbench folders are removed from its module folder (AddonManager, CAM, OpenSCAD, Robot, Test, Web, BIM, Plot
  and Idf), and the addon of the MCP server below is added to that folder. The remaining files keep the contents of the
  release.
- The FreeCAD MCP server neka-nat/freecad-mcp (MIT, https://github.com/neka-nat/freecad-mcp) is vendored unmodified in
  `environment/vendor/freecad-mcp/` with its `LICENSE`, at the commit that folder records.
- The grader's Python packages are those of the benchmark's shared base image (`environment/requirements.txt`).

## Background of the photographs

The photographs are rendered with Blender Cycles in a set built for this robot's tasks, a robot integration bay put
together from the project's licensed art assets, public CC0 material and props generated by code. In the rendered meshes
only, and never in a file the grader or the agent reads, the broken faces of the outer thigh shell are tessellated on
the authors' own surfaces so that the shell shows no gap, and the rod ends are split into ball and housing so that they
can follow the leg's poses. None of the set's assets is shipped with the task: only the rendered photographs are.

## Where the files are

The payload (photographs, `given/`, `parts_list.md`, `hardware_sheet.md`) is served by the benchmark's asset dataset
under `tasks/humanoid/humanoid_leg_cad_design_from_photos/payload/`, at the commit pinned in
`environment/assets.sha256`. The hidden truth is in this folder and in `../solution/reference/` and is never part of the
payload. The public source files of the robot's model are listed by digest in `public_blocklist.json`; a submission that
contains one of them scores 0.
