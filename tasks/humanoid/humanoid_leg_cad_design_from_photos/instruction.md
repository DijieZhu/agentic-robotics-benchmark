# Recover the CAD of a humanoid robot's right leg

The lab needs a second right leg for its humanoid robot, but the CAD of the leg's made parts is lost. What is left are
the models of the purchased components and the file of one made part, the outer shell of the thigh, and that file is
damaged. A colleague took the right leg off the robot, photographed it complete in three poses, took it apart stage by
stage while photographing every stage, and photographed each made part on a cutting mat from both sides. The leg is in
pieces now and cannot be measured again. Re-create its mechanical design in FreeCAD from those photographs and the
documents of the given components.

The leg has 19 made parts to re-create, in 18 different shapes (printed polymer, printed steel, machined aluminium and
steel bar). Infer each one's shape, dimensions and installed position from the evidence, put every given component where
it is installed, and build the assembly with the joints the leg moves about. Reproduce the leg as photographed, not a
corrected version of it.

## How to work

Use the FreeCAD MCP server to inspect and edit the running FreeCAD session. Its `execute_code` tool runs Python inside
that session. Other tools remain available for image measurement, calculations and scripts. The Python environment is
`/opt/venv/bin/python3`; the headless CAD executable is `/opt/freecad/usr/bin/freecadcmd`.

Save early and periodically. You have six hours. A single model response is limited to 64k output tokens; do not wait
until the end to create the deliverable.

`execute_code` stops waiting after 90 seconds unless you give it a longer timeout; a call that reports a timeout keeps
running, so look at the document before you repeat it. The sandbox has 16 GB of memory for everything you run, FreeCAD
included: work through the photographs a few at a time. The photographs are large (2016 x 1512 pixels, 117 MB in all):
look at reduced copies or crops (Pillow is in /opt/venv) and open a full-size file only where a detail needs it; a
session that loads many full-size images may not be able to continue. Nothing in the environment evaluates a design
against the reference.

## Given files (in `/home/user/`)

`photos/photo_001.jpg` to `photos/photo_196.jpg`. Teardown photos show every made part next to its paper tag on a
cutting mat, first one side up, then turned over, each side once from high above and once at a slant. Where two tags
share a mat, each part lies next to its own tag; `parts_list.md` says which parts have the same shape. The tags run from
P01 to P19. Some hollow parts were also photographed looking into their open side. Close-ups show holes and other small
details of the tagged parts. In some close-ups a part stands in a wooden block or leans on a wooden riser, and a checker
card may stand on a wooden post or ledge. The wood, the cards, the rules, the paper tags and the mat are the
photographer's aids: none of them is part of the leg and none is to be modelled. Some close-ups of black parts were
exposed longer so that their holes show; in those photos the black parts look grey and the mat and the paper look
lighter than elsewhere. The photos of the complete leg show it in three poses. In the reference pose it hangs straight
from a hoist hook; in poses B and C it lies on the bench with its joints turned. The lifting block by which it hangs is
a workshop fixture bolted to the leg's upper end: it is not part of the leg and is not to be modelled. Front, rear, left
and right in the captions are the robot's own sides. The lower end is the end that hangs lowest in the reference pose.
The stage photos show the leg as it was taken apart. Each caption names what was taken off in that stage; everything
taken off in an earlier stage stays off. Screws taken out along the way are not named. Close views of a stage show given
components and tagged parts that are still in place; a caption says so when a close view shows what the stage took off
instead. The photos of the complete robot are for orientation only: they show where the leg sits on the robot. The
camera is perspective and uncalibrated, and file names do not encode its pose.

| photos | what they show | scale references |
|---|---|---|
| `photo_001` | P01, first side up, seen from high above | cutting mat A2 (10 mm grid), checker card, steel rule |
| `photo_002` | P01, first side up, seen at a slant | cutting mat A2 (10 mm grid), checker card, steel rule |
| `photo_003` | P01, turned over, seen from high above | cutting mat A2 (10 mm grid), checker card, steel rule |
| `photo_004` | P01, turned over, seen at a slant | cutting mat A2 (10 mm grid), checker card, steel rule |
| `photo_005` | P02, first side up, seen from high above | cutting mat A3 (10 mm grid), checker card |
| `photo_006` | P02, first side up, seen at a slant | cutting mat A3 (10 mm grid), checker card |
| `photo_007` | P02, turned over, seen from high above | cutting mat A3 (10 mm grid), checker card, steel rule |
| `photo_008` | P02, turned over, seen at a slant | cutting mat A3 (10 mm grid), checker card, steel rule |
| `photo_009` | P03, first side up, seen from high above | cutting mat A3 (10 mm grid), checker card |
| `photo_010` | P03, first side up, seen at a slant | cutting mat A3 (10 mm grid), checker card |
| `photo_011` | P03, turned over, seen from high above | cutting mat A3 (10 mm grid), checker card |
| `photo_012` | P03, turned over, seen at a slant | cutting mat A3 (10 mm grid), checker card |
| `photo_013` | P04, first side up, seen from high above | cutting mat A3 (10 mm grid), checker card |
| `photo_014` | P04, first side up, seen at a slant | cutting mat A3 (10 mm grid), checker card |
| `photo_015` | P04, turned over, seen from high above | cutting mat A3 (10 mm grid), checker card |
| `photo_016` | P04, turned over, seen at a slant | cutting mat A3 (10 mm grid), checker card |
| `photo_017` | P05, first side up, seen from high above | cutting mat A3 (10 mm grid), checker card, steel rule |
| `photo_018` | P05, first side up, seen at a slant | cutting mat A3 (10 mm grid), checker card, steel rule |
| `photo_019` | P05, turned over, seen from high above | cutting mat A3 (10 mm grid), checker card, steel rule |
| `photo_020` | P05, turned over, seen at a slant | cutting mat A3 (10 mm grid), checker card, steel rule |
| `photo_021` | P06, first side up, seen from high above | cutting mat A3 (10 mm grid), checker card, steel rule |
| `photo_022` | P06, first side up, seen at a slant | cutting mat A3 (10 mm grid), checker card, steel rule |
| `photo_023` | P06, turned over, seen from high above | cutting mat A3 (10 mm grid), checker card, steel rule |
| `photo_024` | P06, turned over, seen at a slant | cutting mat A3 (10 mm grid), checker card, steel rule |
| `photo_025` | P07, first side up, seen from high above | cutting mat A3 (10 mm grid), checker card |
| `photo_026` | P07, first side up, seen at a slant | cutting mat A3 (10 mm grid), checker card |
| `photo_027` | P07, turned over, seen from high above | cutting mat A3 (10 mm grid), checker card, steel rule |
| `photo_028` | P07, turned over, seen at a slant | cutting mat A3 (10 mm grid), checker card, steel rule |
| `photo_029` | P08, first side up, seen from high above | cutting mat A3 (10 mm grid), checker card, steel rule |
| `photo_030` | P08, first side up, seen at a slant | cutting mat A3 (10 mm grid), checker card, steel rule |
| `photo_031` | P08, turned over, seen from high above | cutting mat A3 (10 mm grid), checker card |
| `photo_032` | P08, turned over, seen at a slant | cutting mat A3 (10 mm grid), checker card |
| `photo_033` | P09, first side up, seen from high above | cutting mat A3 (10 mm grid), checker card |
| `photo_034` | P09, first side up, seen at a slant | cutting mat A3 (10 mm grid), checker card |
| `photo_035` | P09, turned over, seen from high above | cutting mat A3 (10 mm grid), checker card, steel rule |
| `photo_036` | P09, turned over, seen at a slant | cutting mat A3 (10 mm grid), checker card, steel rule |
| `photo_037` | P10, first side up, seen from high above | cutting mat A3 (10 mm grid), checker card |
| `photo_038` | P10, first side up, seen at a slant | cutting mat A3 (10 mm grid), checker card |
| `photo_039` | P10, turned over, seen from high above | cutting mat A3 (10 mm grid), checker card, steel rule |
| `photo_040` | P10, turned over, seen at a slant | cutting mat A3 (10 mm grid), checker card, steel rule |
| `photo_041` | P11, first side up, seen from high above | cutting mat A3 (10 mm grid), checker card, steel rule |
| `photo_042` | P11, first side up, seen at a slant | cutting mat A3 (10 mm grid), checker card, steel rule |
| `photo_043` | P11, turned over, seen from high above | cutting mat A3 (10 mm grid), checker card, steel rule |
| `photo_044` | P11, turned over, seen at a slant | cutting mat A3 (10 mm grid), checker card, steel rule |
| `photo_045` | P12, first side up, seen from high above | cutting mat A3 (10 mm grid), checker card |
| `photo_046` | P12, first side up, seen at a slant | cutting mat A3 (10 mm grid), checker card |
| `photo_047` | P12, turned over, seen from high above | cutting mat A3 (10 mm grid), checker card |
| `photo_048` | P12, turned over, seen at a slant | cutting mat A3 (10 mm grid), checker card |
| `photo_049` | P13, first side up, seen from high above | cutting mat A3 (10 mm grid), checker card, steel rule |
| `photo_050` | P13, first side up, seen at a slant | cutting mat A3 (10 mm grid), checker card, steel rule |
| `photo_051` | P13, turned over, seen from high above | cutting mat A3 (10 mm grid), checker card, steel rule |
| `photo_052` | P13, turned over, seen at a slant | cutting mat A3 (10 mm grid), checker card, steel rule |
| `photo_053` | P14, first side up, seen from high above | cutting mat A3 (10 mm grid), checker card |
| `photo_054` | P14, first side up, seen at a slant | cutting mat A3 (10 mm grid), checker card |
| `photo_055` | P14, turned over, seen from high above | cutting mat A3 (10 mm grid), checker card, steel rule |
| `photo_056` | P14, turned over, seen at a slant | cutting mat A3 (10 mm grid), checker card, steel rule |
| `photo_057` | P19, first side up, seen from high above | cutting mat A3 (10 mm grid), checker card, steel rule |
| `photo_058` | P19, first side up, seen at a slant | cutting mat A3 (10 mm grid), checker card, steel rule |
| `photo_059` | P19, turned over, seen from high above | cutting mat A3 (10 mm grid), checker card, steel rule |
| `photo_060` | P19, turned over, seen at a slant | cutting mat A3 (10 mm grid), checker card, steel rule |
| `photo_061` | P15, P16, first side up, seen from high above | cutting mat A3 (10 mm grid), checker card, steel rule |
| `photo_062` | P15, P16, first side up, seen at a slant | cutting mat A3 (10 mm grid), checker card, steel rule |
| `photo_063` | P15, P16, turned over, seen from high above | cutting mat A3 (10 mm grid), checker card, steel rule |
| `photo_064` | P15, P16, turned over, seen at a slant | cutting mat A3 (10 mm grid), checker card, steel rule |
| `photo_065` | P17, P18, first side up, seen from high above | cutting mat A3 (10 mm grid), checker card |
| `photo_066` | P17, P18, first side up, seen at a slant | cutting mat A3 (10 mm grid), checker card |
| `photo_067` | P17, P18, turned over, seen from high above | cutting mat A3 (10 mm grid), checker card |
| `photo_068` | P17, P18, turned over, seen at a slant | cutting mat A3 (10 mm grid), checker card |
| `photo_069` | P03, its open end seen at a slant | cutting mat A3 (10 mm grid) |
| `photo_070` | P03, its open end seen at a slant | cutting mat A3 (10 mm grid), checker card (partly in frame) |
| `photo_071` | P01, looking into its open side | cutting mat A2 (10 mm grid) |
| `photo_072` | P01, lying with its open side down | cutting mat A2 (10 mm grid), checker card (partly in frame), steel rule |
| `photo_073`-`074` | P04, looking into its open side | cutting mat A3 (10 mm grid) |
| `photo_075`-`076` | P06, looking into its open side | cutting mat A3 (10 mm grid) |
| `photo_077`-`078` | P13, looking into its open side | cutting mat A3 (10 mm grid) |
| `photo_079` | reference pose: complete leg hanging straight from the hoist hook by a lifting block (workshop fixture, not part of the leg), seen from the front | checker card, steel rule standing upright |
| `photo_080` | reference pose: complete leg hanging straight from the hoist hook by a lifting block (workshop fixture, not part of the leg), seen from the front right | checker card, steel rule standing upright |
| `photo_081` | reference pose: complete leg hanging straight from the hoist hook by a lifting block (workshop fixture, not part of the leg), seen from the right | checker card, steel rule standing upright |
| `photo_082` | reference pose: complete leg hanging straight from the hoist hook by a lifting block (workshop fixture, not part of the leg), seen from the rear | checker card, steel rule standing upright |
| `photo_083` | reference pose: complete leg hanging straight from the hoist hook by a lifting block (workshop fixture, not part of the leg), seen from the left | checker card, steel rule standing upright |
| `photo_084` | reference pose: complete leg hanging straight from the hoist hook by a lifting block (workshop fixture, not part of the leg), seen from the front left | checker card, steel rule standing upright |
| `photo_085`-`086` | reference pose: complete leg hanging straight from the hoist hook by a lifting block (workshop fixture, not part of the leg), close view of the lower end | - |
| `photo_087`-`088` | pose B: complete leg lying on the bench | checker card, steel rule |
| `photo_089` | pose B: complete leg lying on the bench, seen from the front | checker card, steel rule |
| `photo_090` | pose B: complete leg lying on the bench, seen from the front left | checker card, steel rule |
| `photo_091`-`092` | pose B: complete leg lying on the bench, close view of the lower end | - |
| `photo_093` | pose C: complete leg lying on the bench, propped with wooden wedges, seen from the front right | checker card, steel rule |
| `photo_094`-`095` | pose C: complete leg lying on the bench, propped with wooden wedges, seen from the right | checker card, steel rule |
| `photo_096` | pose C: complete leg lying on the bench, propped with wooden wedges, seen from the rear right | checker card, steel rule |
| `photo_097` | pose C: complete leg lying on the bench, propped with wooden wedges, close view of the lower end | - |
| `photo_098` | pose C: complete leg lying on the bench, propped with wooden wedges, close view of the lower end | steel rule |
| `photo_099`-`100` | stage 1: P17, P18 and two of the four rod ends RE-06M removed | checker card, steel rule |
| `photo_101`-`102` | stage 2: P04, P12, P14, the remaining two rod ends RE-06M, the two pillow blocks PB-06, the D-shaft DS-0650 and the two set-screw collars SC-06 removed | checker card, steel rule |
| `photo_103`-`104` | stage 3: two of the four pillow blocks PB-12 removed | checker card, steel rule |
| `photo_105`-`106` | stage 4: P11, P13 and the remaining two pillow blocks PB-12 removed | checker card, steel rule |
| `photo_107`-`108` | stage 5: P05, P15 and P16 removed | checker card, steel rule |
| `photo_109`-`110` | stage 6: the two joint actuators RJ-5660 removed | checker card, steel rule |
| `photo_111`-`112` | stage 7: P01, P09 and P10 removed | checker card, steel rule |
| `photo_113`-`114` | stage 8: P06 removed | checker card, steel rule |
| `photo_115`-`116` | stage 9: P02 and the outer thigh shell removed | checker card, steel rule |
| `photo_117`-`118` | stage 10: P08 and P19 removed | checker card, steel rule |
| `photo_119`-`120` | stage 11: P07, the joint actuator RJ-5669 and the flanged ball bearing FB-23 removed | checker card, steel rule |
| `photo_121`-`122` | stage 5, close view of the two joint actuators RJ-5660 | - |
| `photo_123` | reference pose: complete leg hanging straight from the hoist hook by a lifting block (workshop fixture, not part of the leg), close view of two of the rod ends RE-06M | steel rule standing upright |
| `photo_124` | reference pose: complete leg hanging straight from the hoist hook by a lifting block (workshop fixture, not part of the leg), close view of two of the rod ends RE-06M | - |
| `photo_125` | stage 2, close view of the four pillow blocks PB-12 | - |
| `photo_126` | stage 2, close view of the four pillow blocks PB-12 | steel rule |
| `photo_127`-`128` | stage 2, close view of what was removed, with P04 taken off: P12, P14, two of the four rod ends RE-06M, the two pillow blocks PB-06, the D-shaft DS-0650 and the two set-screw collars SC-06 | steel rule |
| `photo_129`-`131` | complete robot hanging from the hoist, shown for orientation only: this task is about its right leg | - |
| `photo_132` | close-up of P01, set down on another side | cutting mat A2 (10 mm grid), checker card (partly in frame), small checker card |
| `photo_133`-`134` | close-up of P01, set down on another side | cutting mat A2 (10 mm grid), small checker card |
| `photo_135` | close-up of P01, set down on another side | small checker card in the plane of the nearest opening |
| `photo_136` | close-up of P01, first side up | cutting mat A2 (10 mm grid), small checker card in the plane of the nearest opening |
| `photo_137` | close-up of P01, turned over | cutting mat A2 (10 mm grid), small checker card in the plane of the nearest opening |
| `photo_138`-`139` | close-up of P02, first side up | cutting mat A3 (10 mm grid), small checker card in the plane of the nearest opening |
| `photo_140` | close-up of P02, turned over | cutting mat A3 (10 mm grid), small checker card in the plane of the nearest opening, steel rule |
| `photo_141` | close-up of P02, turned over | cutting mat A3 (10 mm grid), small checker card in the plane of the nearest opening |
| `photo_142` | close-up of P03, set down on another side | cutting mat A3 (10 mm grid), small checker card |
| `photo_143` | close-up of P03, first side up | cutting mat A3 (10 mm grid), small checker card |
| `photo_144` | close-up of P03, first side up | cutting mat A3 (10 mm grid), checker card, small checker card in the plane of the nearest opening |
| `photo_145`-`146` | close-up of P03, first side up | cutting mat A3 (10 mm grid), small checker card |
| `photo_147` | close-up of P05, first side up | cutting mat A3 (10 mm grid), checker card (partly in frame), small checker card in the plane of the nearest opening |
| `photo_148` | close-up of P05, first side up | cutting mat A3 (10 mm grid), checker card (partly in frame), small checker card in the plane of the nearest opening, steel rule |
| `photo_149` | close-up of P06, set down on another side | cutting mat A3 (10 mm grid), checker card (partly in frame), small checker card, steel rule |
| `photo_150` | close-up of P06, first side up | cutting mat A3 (10 mm grid), checker card, small checker card, steel rule |
| `photo_151` | close-up of P06, first side up | cutting mat A3 (10 mm grid), checker card (partly in frame), small checker card, steel rule |
| `photo_152` | close-up of P07, first side up | cutting mat A3 (10 mm grid), checker card (partly in frame), small checker card in the plane of the nearest opening |
| `photo_153` | close-up of P07, turned over | cutting mat A3 (10 mm grid), small checker card in the plane of the nearest opening |
| `photo_154` | close-up of P08, first side up | cutting mat A3 (10 mm grid), small checker card in the plane of the nearest opening |
| `photo_155` | close-up of P08, turned over | cutting mat A3 (10 mm grid), checker card (partly in frame), small checker card in the plane of the nearest opening |
| `photo_156`-`157` | close-up of P09, first side up | cutting mat A3 (10 mm grid), checker card (partly in frame), small checker card in the plane of the nearest opening |
| `photo_158`-`159` | close-up of P10, first side up | cutting mat A3 (10 mm grid), checker card (partly in frame), small checker card in the plane of the nearest opening |
| `photo_160` | close-up of P12, first side up | cutting mat A3 (10 mm grid), small checker card in the plane of the nearest opening |
| `photo_161` | close-up of P13, set down on another side | cutting mat A3 (10 mm grid), checker card (partly in frame), small checker card, steel rule |
| `photo_162` | close-up of P13, first side up | cutting mat A3 (10 mm grid), checker card (partly in frame), small checker card in the plane of the nearest opening |
| `photo_163`-`164` | close-up of P13, first side up | cutting mat A3 (10 mm grid), checker card (partly in frame), small checker card, steel rule |
| `photo_165` | close-up of P14, first side up | cutting mat A3 (10 mm grid), checker card (partly in frame), small checker card in the plane of the nearest opening |
| `photo_166` | P15, held on edge in a slot cut in a wooden block | cutting mat A3 (10 mm grid), small checker card in the plane of the nearest opening |
| `photo_167`-`170` | close-up of P15, first side up | cutting mat A3 (10 mm grid), checker card (partly in frame), small checker card in the plane of the nearest opening |
| `photo_171` | P16, held on edge in a slot cut in a wooden block | cutting mat A3 (10 mm grid), small checker card |
| `photo_172`-`174` | close-up of P16, first side up | cutting mat A3 (10 mm grid), small checker card in the plane of the nearest opening, steel rule |
| `photo_175` | close-up of P16, first side up | cutting mat A3 (10 mm grid), checker card (partly in frame), small checker card in the plane of the nearest opening, steel rule |
| `photo_176`-`179` | P17, standing on end in a hole drilled in a wooden block | cutting mat A3 (10 mm grid), small checker card in the plane of the nearest opening |
| `photo_180`-`181` | P18, standing on end in a hole drilled in a wooden block | small checker card in the plane of the nearest opening |
| `photo_182` | P18, standing on end in a hole drilled in a wooden block | cutting mat A3 (10 mm grid), small checker card in the plane of the nearest opening |
| `photo_183` | P18, standing on end in a hole drilled in a wooden block | small checker card in the plane of the nearest opening |
| `photo_184` | close-up of P19, first side up | small checker card in the plane of the nearest opening |
| `photo_185` | close-up of P19, first side up | cutting mat A3 (10 mm grid), small checker card in the plane of the nearest opening |
| `photo_186` | close-up of P01, set down on another side | cutting mat A2 (10 mm grid), small checker card in the plane of the nearest opening |
| `photo_187`-`188` | close-up of P01, set down on another side | cutting mat A2 (10 mm grid), small checker card |
| `photo_189` | close-up of P06, set down on another side | cutting mat A3 (10 mm grid), small checker card |
| `photo_190`-`191` | close-up of P12, turned over | cutting mat A3 (10 mm grid), small checker card in the plane of the nearest opening |
| `photo_192` | close-up of P01, first side up | cutting mat A2 (10 mm grid), small checker card in the plane of the nearest opening |
| `photo_193` | close-up of P02, turned over | cutting mat A3 (10 mm grid), small checker card in the plane of the nearest opening, steel rule |
| `photo_194` | close-up of P05, first side up | cutting mat A3 (10 mm grid), small checker card in the plane of the nearest opening |
| `photo_195` | close-up of P03, first side up | cutting mat A3 (10 mm grid), small checker card |
| `photo_196` | close-up of P01, turned over | cutting mat A2 (10 mm grid), small checker card |

Scale references in the photographs: cutting mat A3: 420 x 297 mm, 10 mm grid, every fifth line bold, the bold lines
numbered in centimetres along two margins; cutting mat A2: 594 x 420 mm, same grid and numbering; checker card: ten by
seven squares of 12 mm inside a 12 mm white border, 144 x 108 mm overall; small checker card: ten by seven squares of 4
mm inside a 4 mm white border, 48 x 36 mm overall, in the close-ups raised on a wooden riser, a standing wooden holder
or a wooden post, or lying on the mat (riser, holder and post are props, not parts); where a row of the table calls it
'small checker card in the plane of the nearest opening', its printed side lies in the plane of the opening of the hole
that is nearest to it in the picture, and in the other close-ups it gives the scale at its own place only; steel rule:
300 mm scale graduated in millimetres and numbered every centimetre.

`given/thigh_shell/thigh_shell.brep` with `given/thigh_shell/README.md`: the outer thigh shell, the one made part whose
file survived, in its own frame; the README states the frame, the mass, the centre of mass and what is wrong with the
file. `given/<part number>/` for each purchased component: `<part number>.step` (its geometry in its own frame; a
component supplied assembled has several solids in one file), `<part number>_datasheet.md` (mass, frame, size and what
is known of its mounting interface; the points of a component that the grading uses are named there) and `<part
number>_drawing.png`.

| part number | component | in the leg | files | grading weight |
|---|---|---:|---|---:|
| - | outer thigh shell, printed (file damaged) | 1 | `given/thigh_shell/` | 0 |
| `RJ-5378` | joint actuator `*` | 1 | `given/RJ-5378/` | 3 |
| `RJ-5669` | joint actuator `*` | 1 | `given/RJ-5669/` | 3 |
| `RJ-5660` | joint actuator `*` | 2 | `given/RJ-5660/` | 3 |
| `RE-06M` | ball-joint rod end `**` | 4 | `given/RE-06M/` | 1.5 |
| `PB-12` | pillow block, 12 mm bore | 4 | `given/PB-12/` | 1 |
| `PB-06` | pillow block, 6 mm bore | 2 | `given/PB-06/` | 0.5 |
| `DS-0650` | D-shaft, 6 mm, 50 mm long | 1 | `given/DS-0650/` | 0.5 |
| `SC-06` | set-screw collar `*` | 2 | `given/SC-06/` | 0.25 |
| `FB-23` | flanged ball bearing `*` | 1 | `given/FB-23/` | 0.5 |

Grading weight: the weight of one piece of the component in the installation and fit criteria. The shell has weight 0:
the document is aligned on it, so its own position is not graded. `*`: as orientation, only the direction of the
component's axis is graded (the z axis of the file for the joint actuators and the flanged ball bearing, the axis of the
bore for the set-screw collar), with the sense the axis has in the file; a turn about that axis is not graded. `**`:
only the position of the ball centre is graded, the orientation of the piece is not.

`parts_list.md`: the made parts by tag, with pieces, material and density: polyamide PA12, printed 1.01 g/cm³ (P01-P05,
P12), stainless steel 316L, printed 7.95 g/cm³ (P06, P11, P15, P16), aluminium 7075, machined 2.81 g/cm³ (P07-P10, P13,
P14, P19), stainless steel bar 7.90 g/cm³ (P17, P18). `hardware_sheet.md`: the kinds and sizes of the screws and
threaded inserts seen in the photographs. The masses of the given components are in their data sheets and in the shell's
README. No drawing, dimension or hole table of a made part is given, and no document states which tagged part sits
where, what it is fastened to or how the leg moves; the stage captions name only what was taken off together.

## Deliverable

Save `/home/user/submission/leg.FCStd` with:

1. One PartDesign Body for each tag P01 to P19, with the Body's **Label** exactly the paper tag and exactly one closed
   solid in it. Use each tag once: one Body per tag, not reused through links. Parts on one line of `parts_list.md` have
   the same shape; give each its own Body and tag. Which of them sits where is not graded.
2. Every given component of the table, each piece imported unchanged from its file (in Python: `Part.read(path)` as the
   Shape of one `Part::Feature`) and placed where it is installed. Change nothing but the Placement of the whole piece:
   keep the solids of a file in their places relative to each other, and do not fuse, repair, simplify, scale or
   re-model them. Put in as many pieces of each component as the leg has (19 pieces of 10 components, the shell
   included), and no more: every piece beyond that number displaces the best-placed piece of the same component from the
   grading, and its solids count as extra solids.
3. Everything positioned relative to the given shell. The grader compares your document with the reference in the frame
   of the shell: where the shell stands in your document decides where every made part and every other given piece has
   to be. The origin and the overall orientation of the document are free; units are millimetres.
4. One native FreeCAD Assembly that contains all of the above, with the component that holds the shell grounded. Connect
   parts and pieces that are rigidly fastened to each other with Fixed joints, so that they form rigid groups; the
   members of one App::Part container count as fixed to each other. Build one flat Assembly: an Assembly placed inside
   it counts as one rigid component, and the joints between its own members are not read. Wherever two rigid groups turn
   relative to each other when the leg is in use, put one Revolute joint between them on that axis: the axis is the z
   axis of the joint's frame, in either sense. The two frames of a joint should coincide; where they do not, the frame
   on the group nearer to the shell is read. Connect whatever is left to a neighbouring part with a joint of any kind.
   Only what is inside the Assembly, directly or through links, is read; grounding a component does not connect it to
   anything.
5. Any closed loop of connections left open. Where parts and pieces would form a closed loop, connect each of them at
   one end only: that counts as connected, and the motion that such a loop transmits is computed by the grader itself,
   from the positions of your given pieces. The grader never runs FreeCAD's solver. The solver can report a failure on a
   closed loop, and also on a model without any loop in some overall orientations; neither matters. What counts is where
   the parts are when the file is saved. Creating or editing a joint can move parts, even when the solver reports a
   failure: check the placements after working on joints and put every part that has moved back where it was before you
   save. Only the saved placements are graded.
6. The Assembly saved in the reference pose: the pose of the photographs whose captions say "reference pose". The saved
   pose is the zero position of every joint, and nothing is turned back into the reference pose: a leg saved in another
   pose is graded where its parts are. The sense of rotation of a joint and its angle limits are not read.
7. Holes that carry a threaded insert modelled as round holes of the insert's body diameter (`hardware_sheet.md`), with
   the wider countersink or counterbore at the mouth where the part has one. The insert's own seat is graded on its
   position and direction only; a wider step at the mouth is graded like any other step. The made parts count with the
   densities of `parts_list.md`, the given components with the masses of their data sheets.
8. Nothing else: no ground plane, mat or photo scene, no screws, nuts, washers, inserts or wiring, no alternative
   versions and no duplicate solids. Keep everything in this one file: only `/home/user/submission` reaches the
   verifier, and a link to an object of a document saved elsewhere is empty there.

Screws, nuts, washers, threaded inserts, the wiring and whatever else is neither a tagged part nor in a given file
appear in the photographs but are outside the modelling scope: anything extra counts as an extra solid (see below).

## How it is graded

The verifier runs in a separate sandbox after your session. It opens the saved document with FreeCAD (nothing in it is
executed, and the Assembly is never solved), exports every solid in world coordinates and reads the joints: the kind of
each, whether it is suppressed, the two components it connects and its two frames, and which components are grounded. It
grades what is inside the Assembly: its members, the members of App::Part containers and folders inside it, and links.
Objects outside the Assembly are ignored, the shell included: left outside the Assembly, it counts as missing. A
document without an Assembly is graded from its visible solids, with no credit for joints. A solid counts when it is
closed (`Shape.isClosed()`) and has a volume; its geometric validity is not checked. A solid whose surface cannot be
meshed as a closed volume, or whose closed mesh differs from the solid's own volume by more than half, is replaced by
its convex hull in the volume measures (overlap and penetration).

The whole document scores 0 in three cases only: it is missing or cannot be read (a document with more than 3000 solids,
or one that FreeCAD needs more than 20 minutes to open and export, counts as unreadable; the complete leg has 159
solids); it holds no closed solid; or it cannot be aligned, because the Assembly holds neither the shell, imported
unchanged and exactly once, nor, without the shell, the given components fastened to the same rigid group as the shell.
Those components take the shell's place when each of them is in the Assembly, imported unchanged and exactly once; a
single one of them is not enough. Besides, a submission folder that holds an unchanged copy of a public source file of
this robot's CAD model scores 0, and so does one that is still written to after your session has ended.

Everything else is local and reported. A tag that is not on a PartDesign Body of exactly one closed solid and nothing
else gets no credit, and neither does a tag found on more than one object (a Body reused through several links counts as
its tag used twice); the solids concerned count as extra solids. Open shells and loose faces are left out. Extra solids
are all solids that are neither a credited tagged part nor part of a given piece that the grading uses: they count in
the mass (a stray or left-over copy of a given solid at the density of its component, anything else at 1.01 g/cm³) and
are obstacles in the fit criterion. A given component that is in the document more often than in the leg loses pieces
until the leg's number is left: one piece at a time, each time the piece whose removal lowers the component's
installation score most, which is the best-placed one; where several removals cost the same, the piece at the installed
position nearest to the surplus piece goes. The solids of the pieces taken out are extra solids.

Your document is aligned with the reference rigidly, without scaling or mirroring, on the shell alone: the pose of the
shell in your document is taken onto its pose in the reference. After that nothing is resized, turned or moved into
place, neither a made part nor a given piece. The shell is recognised by its exact geometry, which only an unchanged
import has. Where the shell cannot serve and the given components of its rigid group take its place (see above), the
alignment is a least-squares rigid fit on those components; with no shell in the Assembly at all, nothing counts as
connected, because connection is counted from the component that holds the shell. Given components are recognised by
their full shape, never by a label: each solid must be the given geometry (volume within 1 %, then a rigid
registration), and the solids of one file must keep their places relative to each other within 1 mm; one that does not
is an extra solid and leaves its piece incomplete. Solids of a given file smaller than 20 mm³ count for the completeness
and the mass of their piece only. The balls of the flanged ball bearing, which have no orientation of their own, are
matched by their volume and their place in the bearing. The pieces of one component are matched to its installed
positions one-to-one, and so are the tagged parts of one shape (one line of `parts_list.md`): it does not matter which
of them carries which tag.

| Criterion | Weight | What is measured |
|---|---:|---|
| Shape | 0.25 | Each of the 19 made parts where it is installed, against the true part there. Surface distance, both ways, 90th percentile: full credit up to 0.35 mm, none from 5 mm (60 % of the part's score). Volume overlap (intersection over union): none up to 0.25, full from 0.95 (40 %). A missing part scores 0. Mean over the 19 parts. |
| Holes | 0.20 | The round holes of the made parts: a graded hole is one concave cylindrical wall, with a radius up to 30 mm, that covers at least 300° about its axis; in a stepped hole (a counterbore) each such step is graded, whether or not the other steps close, and a countersink cone is graded where every cylindrical step of its hole is such a wall. Every conical face of such a hole is graded like a step: a countersink, a chamfer at a mouth or at a step, and the conical end of a blind hole. They are matched one-to-one to the concave cylindrical or conical faces of your part: axis offset (full up to 0.25 mm, none from 1.5 mm) × radius error (full up to 0.15 mm, none from 0.6 mm) × axis angle (full up to 2°, none from 10°). A face of yours that is no exact cylinder counts when it fits one within 6 % of the radius (at least 0.02 mm). In a hole that carries a threaded insert, the steps the insert hides (those with a radius up to the insert's body radius plus 0.5 mm) are one graded item without the radius term: it is matched on axis offset and axis angle to a concave wall of yours whose radius lies between 0.4 times the insert's body radius and the body radius plus 0.5 mm; wider steps at the mouth of such a hole are graded as usual. Mean per hole, per part, then over the parts that have such holes; each part's value is multiplied by the smaller of its own placement factor and the factor of its region (see the placement floor below). Walls that do not close (open slots, half-round recesses) are no holes; they belong to the shape. |
| Installation | 0.15 | 80 %: every given piece scores its position (the centroid of the piece: full up to 0.6 mm, none from 12 mm) × (0.7 + 0.3 × orientation, full up to 0.5°, none from 12°) × the share of its solids present, with the weights of the table above. A pose that is a symmetric equivalent of the original counts as exact; a component marked `*` is graded on its axis as the table note says (turned end for end, it is 180° off); a component marked `**` is graded on the position of its ball centre in place of the centroid, and has no orientation term. 20 %: the made parts (half) and the given pieces (half, weighted) that are connected to the component that holds the shell through unsuppressed joints of any kind, directly or through other components; that component must be grounded, and grounding a component does not connect it. |
| Mechanism | 0.20 | Five parts. Axes (30 %): for every axis about which two rigid bodies of the true leg turn against each other, the Revolute joint between the two rigid groups of your document that stand for those bodies scores the distance of a point of the true axis from your axis (full up to 0.3 mm, none from 3 mm) × the angle between the two axes (full up to 0.5°, none from 5°). A group stands for a body when it holds most of that body's tagged parts; given pieces do not decide it. Fixed joints form the groups, a group that holds neither a tagged part nor a given piece is merged into a neighbour, and joints of other kinds only connect. Both bodies in one group, or no chain of Revolute joints from one group to the other: 0 for that axis. Several Revolute joints directly between the same two groups: the worst one counts. Revolute joints in series from one group to the other, through further groups: the best one counts, divided by their number. A body none of whose tagged parts is in the document has no group: the axes on its two sides are then looked for, in order, among the Revolute joints between the groups of its neighbours. Reach (20 %): test points on the body at the far end of the leg are carried through 64 fixed poses, once about the axes of your document found by the rules above and once about the true ones (an axis for which no Revolute joint was found stays rigid). The points are put on your leg by a least-squares rigid fit of the given pieces of that body and the ball centres of the two rod ends nearest to it, repeated once without the pieces that fit far worse than the others. Per pose the largest distance counts; half of the score is the median over the poses (full up to 1.5 mm, none from 15 mm), half the 90th percentile (full up to 3.5 mm, none from 35 mm). Membership (15 %): every tagged part and given piece of the bodies that those axes join (the shell is left out, and so are the members of a loop of point 5 that belong to none of these bodies) is carried through the same poses twice, both times about the true axes: with the body it belongs to in the true leg, and with the group it is in in your document. A group that stands for a body moves as that body; any other group moves as the nearest such group it is joined to by joints of any kind, with the found axes passed on the way taken into account; a group joined to none does not move. Median over the poses of the distance between the two: full up to 1.5 mm, none from 15 mm; times the share of the body's own axes for which a Revolute joint was found. Transmission (25 %): the motion that a loop of point 5 transmits, computed from the positions of given pieces as saved and from nothing else: the axis of each joint actuator `RJ-5660` (the z axis of its file, pointing to its output end), the ball centres of the rod ends `RE-06M`, and the lines through the bore centres of pillow blocks `PB-12` whose bores are in line; the data sheets name these points. The distance between the two ball centres of a loop is kept as saved. Over a fixed set of poses the grader solves the angle that the output of each of these actuators has to turn from the saved pose and compares it with the true leg: 90th percentile of the difference, full up to 1°, none from 10°, per actuator, then the mean. A pose that your geometry cannot reach counts as 10°; a loop with a piece missing scores 0. Closure (10 %): for each tagged part that carries a rod end at each of its two ends, the distance between the two ball centres against the length of your part along its long axis plus twice 14 mm (ball centre to shoulder, in the rod end's data sheet): full up to 0.5 mm of difference, none from 4 mm. The joints of your document play no part in Transmission and Closure. With more than 2000 Revolute joints in the document no joint is read for this criterion: axes, reach and membership score 0. |
| Mass properties | 0.10 | Total mass: full up to 2 %, none from 25 % (30 %). Centre of mass: full up to 1 mm, none from 15 mm (35 %). Principal moments of inertia, largest relative error: full up to 4 %, none from 50 % (35 %). Made parts count with their material's density, the solids of a given component with the density that gives the component its documented mass, the shell as a point mass at the centre of mass of its README, extra solids as stated above. The items outside the modelling scope are on neither side. |
| Fit | 0.10 | 60 %: every given piece, weighted as above, scores its contact with what it rests on in the leg (full up to 0.1 mm farther apart than there, none from 2 mm farther) × its penetration into made parts and extra solids (full up to 0.05 %, none from 5 % of its own volume). 40 %: each made part's penetration into the other made parts, the given pieces and extra solids, on the same band of the part's own volume. What the leg itself has of overlap between the same two items is exempt, and 0.5 mm³ more per pair. A missing part or piece scores 0. The shell is a thin surface in this measure: penetration into the shell is not measured. |

**Placement floor:** every made part has a placement factor: 1 while the surface distance of the Shape row is at most
2.5 mm, falling linearly to 0 at 6 mm; a missing part has 0. The made parts fall into three regions along the leg, by
where they are installed; which part lies in which region is not given. The factor of a region is the mean of the
factors of its parts, each part weighted by the square root of its surface area. Shape is multiplied by nothing. In
Holes each part's value is multiplied by the smaller of its own factor and its region's. In Installation, Mechanism and
Fit every item is multiplied by the factor of the region of its part or piece, an item that concerns several regions by
the mean of their factors. Mass properties is multiplied by the same weighted mean taken over all made parts. Correct
given components around crude or misplaced made parts therefore collect little beyond the shape credit. design_score =
0.25 shape + 0.20 holes + 0.15 installation + 0.20 mechanism + 0.10 mass properties + 0.10 fit; the reward is the design
score.

Whether you work through the MCP server or the shell does not affect your score. Geometry and assembly correctness do.
