# NOTICE (grader-side; never staged into the agent's home)

## The robot

The robot of this task is the open-source three-strut, cable-driven tensegrity rolling robot of the Yale Faboratory
(Rebecca Kramer-Bottiglio, Yale University) and the PRACSYS lab (Kostas Bekris, Rutgers University), the fourth
version of their three-bar platform ("Tribar"; W. R. Johnson III, P. Meng, N. Chen, L. Cimatti, A. Vercoutere,
M. Aanjaneya, R. Kramer-Bottiglio, K. Bekris, "An Open-Source, Reproducible Tensegrity Robot that can Navigate Among
Obstacles", IEEE RA-L, arXiv 2025-11). Every module of the task is the authors' published whole-strut mesh
`Will_strut_scaled_centered.ply` from the Rutgers pose-tracking repository https://github.com/shiyoung77/tensegrity_perception
(branch `ros`, commit 726ed401; sha256 `8dc4caff1efa26468660f30d5e2d91f7d927d6b287480ecd3b06b351f88a4614`,
1 118 331 triangles, strut 304.8 mm; the repository carries no licence file), split into its 494 edge-connected pieces
without filling any hole and grouped into modules after the authors' assembly guide (V3, pages 22-24: the order of the
modules, the colour pairs of the end-cap covers). Nothing of the robot is modelled, redrawn or completed. The masses are those of the authors' simulation model
(the lid's share is an estimate, 31.2 g); the purchased parts carry the vendors and part numbers of the authors' bill of
materials (`given/purchased/purchased_parts.md`); the colours of the green and blue covers are estimates (the red one
is the CAD sibling task's). The benchmark's maintainers' GrabCAD download of the authors' files is
not used here (its print STLs are the same geometry as the published mesh); nothing of it is in the task. The payload
(its given module STLs are the exact geometry the CAD sibling task grades) and the hidden kit are distributed through
the benchmark's asset dataset, with this file as their notice.

## What is real and what this task adds

| Element | Real or added |
|---|---|
| The shape of every part, the modules, the order of assembly, how the modules join (slide on the rod, the lid on the bay, a cap on each rod end), every design position and turn | Real: the authors' mesh and assembly guide |
| Three struts told apart by red / green / blue cap covers, two of one colour per strut; the turn of an end cap about the rod does not matter | Real: the guide (page 24) and the four-fold cap |
| A robot arm cell building the struts | Added (in reality by hand, about 20 minutes a strut); the precedents' task type |
| The aluminium rod and the electronics-bay base pre-assembled, the silicone covers already on the caps | Added division of labour: real steps, given to an upstream worker |
| "At rest in the window for 0.5 s fastens" instead of screws; end caps also fasten | Added (the precedents' mechanism; in reality a cap is only pushed onto the rod end) |
| The sensor jumpers (guide step 7, plugged before the lid goes on) | Not there: no geometry in the sources, none made. The end state is three mechanically complete struts, not electrically complete ones |
| The cables, the strain sensors, the wires, the fishing line | Not there: no geometry in the sources |
| The battery leads (a 36 mm rigid rod-like piece in the mesh) | Drawn, do not collide (in reality a soft lead; in the mesh it runs through the base's solid) |
| The pieces no ray reaches when their module stands alone (591 561 of the cell's 3 354 993 triangles) | Left out of the agent cameras' renders only (`arm/assembly_strut/render_omit.json`, sha256 `0e63b1b03e828e92d8f5e5429fd2e00724980e326f9007f0cf295a87cc7d85fa`, pinned in `grader_config.json`); no piece changed; the given STLs and every review render keep all triangles |
| The mesh's own interferences: the main board's header pins (piece c029) reach up to 0.71 mm (49.8 mm^3) into the lid; the two lid screws' threads overlap their inserts by 0.68-0.69 mm^3 | Real (the authors' mesh), left as they are; the lid's collision piece is carved around the strut core |
| The end cap's mouth face and the motor module's outer face are coplanar in the mesh: when both are fastened they touch at about 0.6 N | Real geometry; inside one fastened cluster, so it does not count for the part-force rule |
| Two end-cap releases of the reference program (the +z caps of the first and third struts): the rod's cross load peaks near 5 N and 7.4 N for 10-15 ticks while a pad's end face slides off the locked cap | Simulation artefact of the reference's grasp, below the 20-tick window of the rod rule; deterministic; accepted |
| The cradle, the two trays and their nests, the datum boards, the two cameras, the per-episode placement, the four hard rules and their values | Added cell equipment and rules (the precedents' way; the values frozen from the reference program's force record by a rule written beforehand) |
| The motor modules lie in their nests turned 6-16 deg off a straight transfer; the lids lie either way round | Added (the hidden layout family) |
| The safety events name the cell's units (`M3_2`, ...) | Added; the names map to fixed colours but appear only after a violation |

## The cell, the room and the tools

The Franka Panda is the MuJoCo Menagerie `franka_emika_panda` model (Apache-2.0, commit c96a32d, as in
`robotic_arm_full_assembly_so101_panda`), with the original flat finger pads. The cell's structure and its grader
modules (`arm/assembly_cell`, `assembly_six`, `assembly_info`, `render`, `render_filament`, `room`, `robotics_grader`,
`common`) are vendored from the benchmark's arm assembly task `robotic_arm_full_assembly_so101_panda`; `verify.py`, the
kit's download and the reach check follow `quadruped_full_assembly_12dof_ur10e`. The agent cameras' RGB is rendered by
MuJoCo's Filament module on CPU Vulkan in a university lab rebuilt from the Archviz Laboratory Pack 2.0 (Unity Asset
Store; the benchmark's maintainers' art pack, licensed for redistribution; rebuilt without Unity, the lab's central island replaced by
the cell's steel bench), its lightmaps baked with Blender Cycles; the bench top's light grey laminate is generated by
code. Depth, the photo pipeline and the depth-sensor model are the shared render layer (MuJoCo backend, OSMesa). The
collision pieces were made with CoACD (MIT) and manifold3d (Apache-2.0); the mesh handling uses trimesh (MIT). The
picture manual is rendered from the same triangles with Blender Cycles.

The givens, the instruction and the image name neither the robot's designers nor their schools, labs, papers, design
names or repositories; this file carries the attribution.
