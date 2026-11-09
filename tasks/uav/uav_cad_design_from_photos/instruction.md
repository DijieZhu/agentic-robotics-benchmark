# Recover the CAD of an AGIPIX research quadrotor

The editable CAD of the lab's AGIPIX research quadrotor was lost. A colleague photographed the complete unit, took it
apart layer by layer while photographing each stage, and photographed every made part on a cutting mat. The unit is no
longer available for measurement. Re-create its mechanical design in FreeCAD from those photographs and the purchased
components' documents.

The unit is built from cut carbon-fibre plates, 3D-printed parts and machined aluminium landing legs, around bought
components: four motors with propellers, a lidar, a depth camera, a compute stack, a flight controller, an ESC, a power
board, a battery and small hardware. Infer every made part's shape, dimensions and installed position from the evidence.

## How to work

Use the FreeCAD MCP server to inspect and edit the running FreeCAD session. Its `execute_code` tool runs Python inside
that session. Other tools remain available for image measurement, calculations and scripts. The Python environment is
`/opt/venv/bin/python3`; the native headless CAD executable is `/opt/freecad/usr/bin/freecadcmd`.

Save early and periodically. You have six hours. A single model response is limited to 64k output tokens; do not wait
until the end to create the deliverable.

## Given files

`photos/photo_001.jpg` through `photos/photo_155.jpg`. Teardown photos show every made part next to its paper tag
(P01 to P51), first one side up, then turned over. Parts with the same shape were photographed together, each with its
own tag. Close-ups show hole patterns and other details. The stage photos show the unit as it was taken apart; each
caption says what had been removed so far.

| photos | what they show | scale references |
|---|---|---|
| `photo_001`-`002` | P01, first side up | cutting mat A2 (10 mm grid), checker card, steel rule |
| `photo_003`-`004` | P01, turned over | cutting mat A2 (10 mm grid), checker card, steel rule |
| `photo_005`-`006` | P02, first side up | cutting mat A2 (10 mm grid), checker card |
| `photo_007`-`008` | P02, turned over | cutting mat A2 (10 mm grid), checker card |
| `photo_009`-`010` | P03, first side up | cutting mat A3 (10 mm grid), checker card |
| `photo_011`-`012` | P03, turned over | cutting mat A3 (10 mm grid), checker card |
| `photo_013`-`014` | P04, first side up | cutting mat A3 (10 mm grid), checker card, steel rule |
| `photo_015`-`016` | P04, turned over | cutting mat A3 (10 mm grid), checker card, steel rule |
| `photo_017`-`018` | P05, P06, first side up | cutting mat A3 (10 mm grid), checker card |
| `photo_019`-`020` | P05, P06, turned over | cutting mat A3 (10 mm grid), checker card |
| `photo_021`-`022` | P07, first side up | cutting mat A3 (10 mm grid), checker card |
| `photo_023`-`024` | P07, turned over | cutting mat A3 (10 mm grid), checker card |
| `photo_025`-`026` | P08, first side up | cutting mat A3 (10 mm grid), checker card |
| `photo_027`-`028` | P08, turned over | cutting mat A3 (10 mm grid), checker card |
| `photo_029`-`030` | P09, first side up | cutting mat A3 (10 mm grid), checker card |
| `photo_031`-`032` | P09, turned over | cutting mat A3 (10 mm grid), checker card |
| `photo_033`-`034` | P10, first side up | cutting mat A3 (10 mm grid), checker card |
| `photo_035`-`036` | P10, turned over | cutting mat A3 (10 mm grid), checker card |
| `photo_037`-`038` | P11-P14, first side up | cutting mat A3 (10 mm grid), checker card, steel rule |
| `photo_039`-`040` | P11-P14, turned over | cutting mat A3 (10 mm grid), checker card, steel rule |
| `photo_041`-`042` | P15, P16, first side up | cutting mat A3 (10 mm grid), checker card |
| `photo_043`-`044` | P15, P16, turned over | cutting mat A3 (10 mm grid), checker card |
| `photo_045`-`046` | P17, P18, first side up | cutting mat A3 (10 mm grid), checker card |
| `photo_047`-`048` | P17, P18, turned over | cutting mat A3 (10 mm grid), checker card |
| `photo_049`-`050` | P19-P22, first side up | cutting mat A3 (10 mm grid), checker card |
| `photo_051`-`052` | P19-P22, turned over | cutting mat A3 (10 mm grid), checker card |
| `photo_053`-`054` | P23-P26, first side up | cutting mat A3 (10 mm grid), checker card |
| `photo_055`-`056` | P23-P26, turned over | cutting mat A3 (10 mm grid), checker card |
| `photo_057`-`058` | P27-P30, first side up | cutting mat A3 (10 mm grid), checker card |
| `photo_059`-`060` | P27-P30, turned over | cutting mat A3 (10 mm grid), checker card |
| `photo_061`-`062` | P31, first side up | cutting mat A3 (10 mm grid), checker card |
| `photo_063`-`064` | P31, turned over | cutting mat A3 (10 mm grid), checker card |
| `photo_065`-`066` | P32, first side up | cutting mat A3 (10 mm grid), checker card |
| `photo_067`-`068` | P32, turned over | cutting mat A3 (10 mm grid), checker card |
| `photo_069`-`070` | P33-P36, first side up | cutting mat A3 (10 mm grid), checker card |
| `photo_071`-`072` | P33-P36, turned over | cutting mat A3 (10 mm grid), checker card |
| `photo_073`-`074` | P37-P40, first side up | cutting mat A3 (10 mm grid), checker card |
| `photo_075`-`076` | P37-P40, turned over | cutting mat A3 (10 mm grid), checker card |
| `photo_077`-`078` | P41, first side up | cutting mat A3 (10 mm grid), checker card |
| `photo_079`-`080` | P41, turned over | cutting mat A3 (10 mm grid), checker card |
| `photo_081`-`082` | P42, first side up | cutting mat A3 (10 mm grid), checker card |
| `photo_083`-`084` | P42, turned over | cutting mat A3 (10 mm grid), checker card |
| `photo_085`-`086` | P43, first side up | cutting mat A3 (10 mm grid), checker card |
| `photo_087`-`088` | P43, turned over | cutting mat A3 (10 mm grid), checker card |
| `photo_089`-`090` | P44-P47, first side up | cutting mat A3 (10 mm grid), checker card |
| `photo_091`-`092` | P44-P47, turned over | cutting mat A3 (10 mm grid), checker card |
| `photo_093`-`094` | P48-P51, first side up | cutting mat A3 (10 mm grid), checker card |
| `photo_095`-`096` | P48-P51, turned over | cutting mat A3 (10 mm grid), checker card |
| `photo_097`-`102` | assembled unit standing on the bench, propellers on | checker card, steel rule |
| `photo_103`-`104` | propellers removed | checker card, steel rule |
| `photo_105`-`107` | P03 and P19-P22 removed | checker card, steel rule |
| `photo_108`-`109` | compute stack and P33-P36 removed | checker card, steel rule |
| `photo_110`-`111` | unit turned over, resting on two wooden blocks | checker card, steel rule |
| `photo_112`-`113` | turned over; battery removed | checker card, steel rule |
| `photo_114`-`116` | turned over; P04, P37-P40 and four standoffs removed | checker card, steel rule |
| `photo_117`-`118` | turned over; depth camera, P10, P42 and P43 removed | checker card, steel rule |
| `photo_119`-`121` | standing again; lidar with its four dampers and P05-P09 removed | checker card, steel rule |
| `photo_122`-`124` | P02, P11-P14 and P27-P30 removed | checker card, steel rule |
| `photo_125`-`128` | close-up of P01, first side up | cutting mat A2 (10 mm grid), steel rule |
| `photo_129`-`135` | close-up of P02, first side up | cutting mat A2 (10 mm grid), steel rule |
| `photo_136` | close-up of P04, first side up | cutting mat A3 (10 mm grid), steel rule |
| `photo_137` | close-up of P06, first side up | cutting mat A3 (10 mm grid), steel rule |
| `photo_138` | close-up of P05, first side up | cutting mat A3 (10 mm grid), steel rule |
| `photo_139` | close-up of P12-P14, first side up | cutting mat A3 (10 mm grid), steel rule |
| `photo_140` | close-up of P14, first side up | cutting mat A3 (10 mm grid), steel rule |
| `photo_141` | close-up of P23, first side up | cutting mat A3 (10 mm grid), steel rule |
| `photo_142` | close-up of P27, first side up | cutting mat A3 (10 mm grid), steel rule |
| `photo_143` | close-up of P44, first side up | cutting mat A3 (10 mm grid), steel rule |
| `photo_144` | close-up of P20, P21, first side up | cutting mat A3 (10 mm grid), steel rule |
| `photo_145` | close-up of P19, first side up | cutting mat A3 (10 mm grid), steel rule |
| `photo_146` | close-up of P22, first side up | cutting mat A3 (10 mm grid), steel rule |
| `photo_147` | close-up of P23, P24, first side up | cutting mat A3 (10 mm grid), steel rule |
| `photo_148` | close-up of P42, first side up | cutting mat A3 (10 mm grid), steel rule |
| `photo_149` | close-up of P51, first side up | cutting mat A3 (10 mm grid), steel rule |
| `photo_150` | close-up of P48, P50, first side up | cutting mat A3 (10 mm grid), steel rule |
| `photo_151` | close-up of P48, first side up | cutting mat A3 (10 mm grid) |
| `photo_152` | close-up of P49, first side up | cutting mat A3 (10 mm grid), steel rule |
| `photo_153` | close-up of P51, first side up | cutting mat A3 (10 mm grid), steel rule |
| `photo_154` | close-up of P50, first side up | cutting mat A3 (10 mm grid), steel rule |
| `photo_155` | close-up of P49, first side up | cutting mat A3 (10 mm grid), steel rule |

`purchased/<part number>/` for each purchased component: `<part number>.step` (its geometry in its own frame; an
assembled module has several solids in one file), `<part number>_datasheet.md` (mass, frame, mounting interface) and
`<part number>_drawing.png`.

| part number | component | in the unit |
|---|---|---:|
| `TMOTOR-2306.3-1700KV` | T-Motor 2306.3 1700 KV brushless motor | 4 |
| `AZURE-SFP5148-CCW` | Azure Power SFP 5148 tri-blade propeller, counter-clockwise | 2 |
| `AZURE-SFP5148-CW` | Azure Power SFP 5148 tri-blade propeller, clockwise | 2 |
| `TMOTOR-F55A-PROII-4IN1` | T-Motor F55A Pro II 4-in-1 ESC | 1 |
| `POWER-BOARD` | Power distribution board with XT60 plug | 1 |
| `XT60-A` | XT60 connector, female housing | 2 |
| `XT60-B` | XT60 connector, male housing | 1 |
| `M3-HEX-STANDOFF-7` | M3 hex standoff, 7 mm | 2 |
| `M3-HEX-STANDOFF-15` | M3 hex standoff, 15 mm | 4 |
| `CUBE-ORANGE-CARRIER` | CubePilot Cube Orange flight controller on its carrier board | 1 |
| `HADRON-ORIN-NX-STACK` | Connect Tech Hadron carrier with NVIDIA Jetson Orin NX module and active heatsink | 1 |
| `TATTU-6S-4500` | Tattu 4500 mAh 6S LiPo battery pack with lead and XT60 | 1 |
| `LIVOX-MID-360` | Livox MID-360 lidar | 1 |
| `INTEL-D455` | Intel RealSense D455 depth camera | 1 |
| `M3-RUBBER-DAMPER` | M3 rubber vibration damper (stud and threaded hole) | 4 |

Scale references: the cutting mats (A3 and A2) have a 10 mm grid; the checker card has ten by seven 12 mm squares on a
white border; the steel rule is graduated in millimetres. The camera is perspective and uncalibrated, and file names
do not encode its pose. Materials: carbon-fibre laminate 1.60 g/cm³ for P01 to P14, aluminium 2.70 g/cm³ for P44 to
P47, printed polymer 1.24 g/cm³ for all other made parts. Purchased masses are in their datasheets.

## Deliverable

Save `/home/user/submission/drone.FCStd` with:

1. One valid, closed B-rep solid in a PartDesign Body for each tag P01 to P51, with the Body's **Label** exactly its
   paper tag. Use each tag once: one Body per tag, not reused through links. Parts photographed together have the same
   shape; give each its own Body and tag. Which of several identical parts sat where is not graded: the grader matches
   identical parts to the installed positions itself. Mirror-image parts are different parts.
2. Every purchased component in the table, imported unchanged from its STEP file, as separate shape objects in its
   installed position. Keep all solids of a file together and in their relative placement; do not fuse, simplify or
   re-model them. The propellers sit on their motors; which hand goes on which motor is part of the design.
3. One native FreeCAD Assembly containing all of these, with P01 grounded and Fixed joints connecting the rigid
   structure and the mounted components. Only what is inside the Assembly is graded.
4. The assembly saved in the photographed assembled configuration, propellers on. Origin and overall orientation are
   arbitrary; units are millimetres. Do not include a ground plane, mat, photo scene, alternative versions or duplicate
   solids.

Screws, nuts, wiring and cables (the lidar's cable included) and straps are outside the modelling scope; do not model
them (anything extra counts as an extra solid, see below). Only the saved document is evaluated. A submission that
contains an unchanged copy of a published CAD or mesh export of this platform scores 0.

## How it is graded

The verifier opens the saved document with FreeCAD (nothing in it runs), exports every solid in world coordinates
and reads the joints. It grades what is inside the Assembly: its members, the members of App::Part containers and
folders inside it, and links. Objects outside the Assembly are ignored. A document without an Assembly is graded from
its visible solids, with no credit for joints.

Only three things score zero: the document cannot be read; it holds no solid; P01 is missing or unusable for the
alignment (on two objects, several solids, or none valid). Everything else is local and reported. A tagged part that is
not a PartDesign Body of exactly one valid closed solid gets no credit, and neither does a tag on more than one object (a
Body reused through several links counts as its tag used twice); their solids count as extra solids. Invalid solids,
open shells and loose faces are left out. Extra solids, meaning anything neither matched to a tag nor recognised as a
purchased component, count in the mass at 1.24 g/cm³ and as obstacles in fit and rotor clearance.

The unit is aligned rigidly on P01, with no scaling or mirroring. When P01 alone fits several ways, the way under which
the other parts and components land nearest their places wins. Nothing is resized or moved into place. Purchased
components are recognised by their full shape, never by label: each solid must be the vendor geometry (volume within
1 %, then a rigid registration), and the solids of one STEP file must keep their relative placement, so import them
together and unchanged. Identical made parts are matched to installed positions one-to-one, so it does not matter which
identical unit carries which tag. Mirror-image parts are different parts.

| Criterion | Weight | What is measured |
|---|---:|---|
| Manufactured shape | 0.30 | Each of the 51 parts where it is installed. Surface error, 90th percentile: full credit to 0.35 mm, none from 5 mm (60 %). Solid overlap (IoU): none below 0.25, full from 0.95 (40 %). |
| Functional features | 0.25 | Every existing hole, bore, counterbore or countersink step and slot end, matched one-to-one to the part's concave cylindrical or conical faces: axis offset (full to 0.25 mm, none from 1.5 mm) × radius error (full to 0.15 mm, none from 0.6 mm) × axis angle (full to 2°, none from 10°). Averaged per part, then over the parts that have such features. |
| Installation | 0.20 | 80 %: every purchased instance scores its centroid position (full to 0.6 mm, none from 12 mm) × (0.7 + 0.3 × orientation, full to 0.5°, none from 12°) × the share of its solids present. A pose that is a symmetric equivalent of the original counts as exact. A propeller is measured on its motor: on the motor's axis, the hand is its part number, the spin phase is free. Instance weights: lidar, battery, compute stack 3; flight controller, depth camera 2.5; motor 2; ESC 1.5; propeller, power board 1; XT60, damper 0.5; standoff 0.25. 20 %: the made parts (half) and purchased instances (half, weighted) connected to the grounded part through unsuppressed Fixed joints. |
| Mass properties | 0.10 | Total mass: full to 2 %, none from 25 % (30 %). Centre of mass: full to 1 mm, none from 15 mm (35 %). Principal moments of inertia, largest relative error: full to 4 %, none from 50 % (35 %). Made parts at their material density, purchased components at their datasheet mass, extra solids at 1.24 g/cm³. Cables and screws are on neither side. |
| Component fit | 0.10 | 60 %: every purchased instance, weighted as above, scores its contact with what it rests on in the original (full when no farther than there, none from 2 mm farther) × its penetration into made parts and extra solids (full to 0.05 %, none from 5 % of its volume). 40 %: each made part's penetration into the other parts and extra solids, on the same band of its volume. Overlaps the original design has itself are exempt. |
| Rotor clearance | 0.05 | Each propeller's swept disc (radius 63 mm, over the propeller's own height), carried by its motor's pose, against everything else in the unit except that motor and the propeller on it. Full to 0.05 mm³ of overlap, none from 200 mm³. |

**Shape floor:** features, installation, mass, fit and clearance are multiplied by a factor that rises linearly from 0 at
a mean shape score of 0.15 to 1 at 0.55. Empty space, boxes or a few correct parts therefore collect nothing beyond their
shape credit. design_score = 0.30 shape + 0.25 features + 0.20 installation + 0.10 mass + 0.10 fit + 0.05 clearance;
the reward is the design score.

The choice of shell or MCP tools does not affect your score. Geometry and assembly correctness do.
