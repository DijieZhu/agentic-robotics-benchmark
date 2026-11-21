# Recover the CAD of a cabin free-flyer

The editable CAD of the lab's engineering unit of the cabin free-flyer, a cube-shaped free-flying robot for the cabin of
a crewed station, was lost. A colleague photographed the complete unit, then took it apart module by module,
photographing every stage and every part on a cutting mat. The modules themselves were not opened, and the unit is no
longer available for measurement. Re-create its mechanical design in FreeCAD from those photographs.

The unit is a core module between two propulsion modules. Each propulsion module carries a corner-bumper set of four
pieces and, on the assembled unit, a side skin. A perching arm with a gripper is stowed in the top-aft bay. There are no
drawings, datasheets or component files: every shape, dimension and position has to come from the photographs.

## How to work

Use the FreeCAD MCP server `freecad` to inspect and edit the running FreeCAD 1.1 session; its `execute_code` tool runs
Python inside that session. Your other tools stay available for looking at the photographs, measuring in them,
calculations and scripts. In the shell, Python with numpy, scipy, Pillow, matplotlib and trimesh (no OpenCV) is
`/opt/venv/bin/python3`, and the headless FreeCAD executable is `/opt/freecad/usr/bin/freecadcmd`. There is no network
access; the sandbox has 8 CPUs and 16 GB of memory.

## Given files

`/home/user/photos/photo_001.jpg` to `photo_119.jpg`, all taken with one phone. The table lists them by number. The mat
is always listed, the checker card when all its inner corners are in view, the steel rule when at least 40 mm of its
scale is; a reference followed by numbers is in view in those photographs only.

| photos | what they show | scale references in view |
|---|---|---|
| 001-008 | the assembled unit, configuration Z, side skins on | A2 mat 594 x 420 mm; steel rule 300 mm (001-006 only) |
| 009-011 | stage S1: both side skins removed | A2 mat 594 x 420 mm; steel rule 300 mm (009, 011 only) |
| 012-014 | stage S2: P03 and P05 removed as well; the unit lies on P02 | A2 mat 594 x 420 mm; steel rule 300 mm (012-013 only) |
| 015-017 | stage S3: P06-P12 removed as well; the unit still lies on P02 | A2 mat 594 x 420 mm; steel rule 300 mm |
| 018-020 | the assembled unit at pose A (not recorded), side skins on | A2 mat 594 x 420 mm; steel rule 300 mm (018 only) |
| 021-023 | the assembled unit at pose B (not recorded), side skins on | A2 mat 594 x 420 mm; steel rule 300 mm (021-022 only) |
| 024-025 | P08-P12 seen from close, gripper closed (as in configuration Z); the unit at another pose (not recorded), side skins on | A2 mat 594 x 420 mm |
| 026-027 | P08-P12 seen from close, gripper open; the unit at another pose (not recorded), side skins on | A2 mat 594 x 420 mm |
| 028-033 | P01 on the mat, resting face 1 of 3 | A2 mat 594 x 420 mm; checker card, 12 mm squares; steel rule 300 mm (028-030, 032-033 only) |
| 034-039 | P01 on the mat, resting face 2 of 3 | A2 mat 594 x 420 mm; checker card, 12 mm squares; steel rule 300 mm |
| 040-045 | P01 on the mat, resting face 3 of 3 | A2 mat 594 x 420 mm; checker card, 12 mm squares; steel rule 300 mm (041-042, 044-045 only) |
| 046-048, 116-119 | P02 on the mat, outer face up | A2 mat 594 x 420 mm; checker card, 12 mm squares |
| 049-051 | P03 on the mat, outer face up | A2 mat 594 x 420 mm; checker card, 12 mm squares |
| 052-054 | P02 on the mat, inner face up | A2 mat 594 x 420 mm; checker card, 12 mm squares |
| 055-057 | P03 on the mat, inner face up | A2 mat 594 x 420 mm; checker card, 12 mm squares |
| 058-060 | P02 on the mat, standing on its edge | A2 mat 594 x 420 mm; checker card, 12 mm squares; steel rule 300 mm (060 only) |
| 061-063 | P03 on the mat, standing on its edge | A2 mat 594 x 420 mm; checker card, 12 mm squares; steel rule 300 mm |
| 064-069 | P04 and P05 (four pieces each) on the mat, first side up | A2 mat 594 x 420 mm; checker card, 12 mm squares; steel rule 300 mm (064-066, 068-069 only) |
| 070-075 | P04 and P05 (four pieces each) on the mat, turned over | A2 mat 594 x 420 mm; checker card, 12 mm squares; steel rule 300 mm |
| 076-081 | P06-P08 on the mat, first side up | A3 mat 420 x 297 mm; checker card, 12 mm squares; steel rule 300 mm |
| 082-087 | P06-P08 on the mat, turned over | A3 mat 420 x 297 mm; checker card, 12 mm squares; steel rule 300 mm |
| 088-093 | P09-P12 on the mat, first side up | A3 mat 420 x 297 mm; checker card, 12 mm squares; steel rule 300 mm |
| 094-099 | P09-P12 on the mat, turned over | A3 mat 420 x 297 mm; checker card, 12 mm squares; steel rule 300 mm (094, 098-099 only) |
| 100 | close-up of P06, first side up (layout of 076-081) | A3 mat 420 x 297 mm; steel rule 300 mm |
| 101 | close-up of P07, first side up (layout of 076-081) | A3 mat 420 x 297 mm; steel rule 300 mm |
| 102, 105 | close-up of P09, first side up (layout of 088-093) | A3 mat 420 x 297 mm; steel rule 300 mm |
| 103 | close-up of P10, first side up (layout of 088-093) | A3 mat 420 x 297 mm; steel rule 300 mm |
| 104 | close-up of P09, turned over (layout of 094-099) | A3 mat 420 x 297 mm; steel rule 300 mm |
| 106 | close-up of P11, turned over (layout of 094-099) | A3 mat 420 x 297 mm; steel rule 300 mm |
| 107-110, 112 | close-up of P01, resting face 2 of 3 (layout of 034-039) | A2 mat 594 x 420 mm; steel rule 300 mm |
| 111, 113 | close-up of P01, resting face 3 of 3 (layout of 040-045) | A2 mat 594 x 420 mm; steel rule 300 mm |
| 114-115 | close-up of P02, outer face up (layout of 046-048, 116-119) | A2 mat 594 x 420 mm; steel rule 300 mm |

Configuration Z is the configuration the document is saved in; the stage photographs (009-017) still show it, and each
stage keeps what the stage before it removed. Poses A and B and the pose of 024-027 are other configurations of the
assembled unit; their angles were not recorded, and neither was the opening of the gripper in 026-027.

The phone took plain perspective pictures without lens distortion: with a wide lens for the unit, stage and teardown
photographs, with a longer lens for the close-ups and the gripper photographs. It is not calibrated, where it was held
is not recorded, and exposure and white balance vary a little.

Scale references, with their nominal sizes:

- cutting mats: A2 (594 x 420 mm) and A3 (420 x 297 mm), 2 mm thick, with a printed 10 mm grid inside a 15 mm margin,
  every fifth line bold, numbered in cm; the unit and the parts rest on the mat's top face;
- checker card: 10 x 7 squares of 12 mm (9 x 6 inner corners) inside a 12 mm white border, 0.5 mm thick, flat on the
  mat;
- steel rule: a 300 mm scale with 1 mm and 10 mm graduations, 316 x 25 x 0.9 mm, on the mat or on the bench beside it;
- paper tags: 34 x 18 mm, one next to each part in the teardown photographs (one for the four pieces of a corner-bumper
  set). A tag reads the letter P and two digits, P01 to P12; its font draws the digit 1 as a plain vertical stroke.

## Deliverable

Save one FreeCAD document as `/home/user/submission/unit.FCStd`: one Assembly (FreeCAD's Assembly workbench) whose
members carry the twelve tags. Only this document is graded, as saved, and nothing in it is executed. If `unit.FCStd` is
absent and the submission folder holds exactly one other `.FCStd` file, that file is read instead, with a note.

| tags | class | part weight (each) |
|---|---|---:|
| P01 | core module | 3 |
| P02, P03 | propulsion module | 3 |
| P04, P05 | corner-bumper set (its four pieces under one tag) | 0.5 |
| P06 to P08 | arm piece | 1.5 |
| P09 to P12 | finger piece | 0.75 |

1. **Tags.** Each tag is the Label of one member of the Assembly, written exactly (case matters; spaces at either end
   are ignored). The members of a group folder (App::DocumentObjectGroup) inside the Assembly are members too, and
   hidden members count like visible ones; objects outside the Assembly are not members (a member may link to one in the
   same document). A member may be a PartDesign Body, a Part feature, an App::Part holding several solids (nested
   containers included) or an App::Link to one of these. The member's own Label is read, not the Label of the object a
   link points to, and FreeCAD renames a Label that already exists in the document (a second `P06` becomes `P001`):
   check the Labels of the Assembly's members before you finish.
2. **Solids.** Every valid closed solid under a tagged member belongs to that tag: one or more per tag, in millimetres.
   Valid and closed are what FreeCAD's `isValid()` and `isClosed()` report for the solid. P02 and P03 are the two
   propulsion modules and P04 and P05 the two corner-bumper sets; the two of a pair have the same shape, and their tags
   are interchangeable.
3. **Joints.** P01 itself carries a Grounded joint. P02 to P06, the five parts that do not move relative to P01, are
   joined to it by Fixed joints, directly or through one another. The arm and the fingers have six Revolute joints in
   all, one at each hinge, each joining the two tagged parts on either side of it; which parts these are, and where the
   hinges lie, is for you to read from the photographs. A Revolute joint's axis is the z axis of its joint frame through
   the frame's origin; its direction and the order of the joint's two references do not matter.
4. **Configuration.** Save the unit assembled, every part in its installed place, in configuration Z: the configuration
   of photographs 001-008 (the arm folded in and the fingers closed).

## How it is graded

Your document is compared with the photographed unit itself, called the original below. The grader opens the saved
document with FreeCAD and reads every part where its saved placement puts it; it does not solve the assembly again. Of
several Assemblies in a document, only the first top-level one is read. Without an Assembly, the document's visible
top-level objects are read as the members: `surface` is graded, `joint_axes` and `structure` are 0.

Only five things score 0 for the whole submission, each with its reason:

1. no document: the submission folder holds no `unit.FCStd` and not exactly one other `.FCStd` file. Symbolic links are
   not followed: a linked submission folder, or one outside your home folder, is not read, and a linked document does
   not count;
2. a document that cannot be read: it does not open, it is cut off or damaged, reading it takes longer than 600 s, or it
   holds more than 4000 shape objects (Bodies, features and links to them, inside the Assembly) or more than
   6000 solids;
3. no valid closed solid on any tag;
4. a file anywhere in the submission folder whose bytes equal a published mesh, texture or description file of this
   robot;
5. anything in the submission folder created or modified after the session ended (leave no process writing there).

Everything else costs only the item it touches and is reported:

- Invalid solids, open shells and loose faces are left out.
- A tag carried by more than one member is void: it earns no surface credit, the joints that name it do not count, and
  it counts neither as joined to P01 nor as present.
- Members whose Label is not a tag are ignored; their solids belong to no tag.
- A member is identified by its Label whether or not it holds a valid closed solid: its joints and its fixed connection
  count; its surface and its presence need at least one valid closed solid.
- A suppressed Fixed or Revolute joint counts as absent.
- A part or joint that the grader fails to evaluate scores 0 for that item only.

Before anything is measured, the grader aligns your whole unit rigidly onto the original, with no scaling and no
mirroring, so the document's origin and orientation are free. The one alignment serves every part and every joint, and a
document that holds only some of the parts is aligned on those. Among nearly equal fits the one that puts the tagged
parts nearest their places wins. The two tags of a pair are then matched to the pair's two installed places one to one
by position, so either tag of a pair may sit at either place, and the choice of fit does not depend on which is written
where.

design_score = 0.65 x surface + floor x (0.25 x joint_axes + 0.10 x structure); the reward is the design score (full
marks 1.0).

| Criterion | Weight | What is measured |
|---|---:|---|
| `surface` | 0.65 | Every tagged part where it is installed in configuration Z. Two sets of distances are pooled, each taken at points spread evenly over its surface: from the surfaces of the tag's valid closed solids to the original part's surface, and from the points of the original part's surface that the photographs show (each seen in at least three photographs) to the tag's solids. The 90th percentile of the pooled distances earns full credit up to 0.6 mm and none from 6 mm, linearly between. `surface` is the mean over the twelve tags with the part weights above (17.5 in all); a tag that is missing, void or without a valid closed solid scores 0. |
| `joint_axes` | 0.25 | For each of the six hinges of the arm and the fingers, the unsuppressed Revolute joint between the two tagged parts on either side of it scores 0.6 x band(offset; full to 0.3 mm, none from 3 mm) + 0.4 x band(angle; full to 0.5 deg, none from 5 deg); a band is 1 up to its first value, 0 from its second, linear between. The offset is the distance from the original joint's origin to your axis line; the angle is the angle between the two lines. Both joint frames of a joint are read and their scores averaged (they coincide in a solved assembly). When several Revolute joints join the same two tags, the best-scoring one counts. A Revolute joint between two members that do not meet at a hinge of the original unit is ignored. A hinge without a joint scores 0; `joint_axes` is the mean of the six. |
| `structure` | 0.10 | (the share of P02 to P06 joined to P01) x (the share of the twelve tags present on exactly one member with at least one valid closed solid). Joined means connected with P01's member through Grounded joints and unsuppressed Fixed joints only, never through a Revolute joint. P01 must itself carry a Grounded joint: without it no part counts as joined, and another grounded member does not stand in for it. |

**Floor.** `joint_axes` and `structure` are multiplied by a factor that rises linearly from 0 at a `surface` score of
0.15 to 1 at 0.55; `surface` itself is not.

Not graded: volume, mass, hole features as separate items, joint limits, colours and printing, the side skins, fasteners
and wiring, and whatever no photograph shows.

## Modelling conventions

They follow from how `surface` is measured, and careful work loses credit without them.

- **What is seen is graded.** From the original's side, only the surface the photographs show is measured against your
  solids. That includes what is seen through the core module's open frame and through the propulsion modules' grilles:
  the batteries, trays and boards seen there are to be built. What no photograph shows is not graded and need not be
  built. From your side, every face of your solids is measured against the whole surface of the original part, seen or
  not, so a face that part does not have (a cap over an opening, a filled-in recess) counts by its distance from the
  part's nearest surface.
- **Thin walls.** A plate or shell whose thickness the photographs do not show: build it no thicker than 0.5 mm. The
  original's surface there is a single sheet, and both faces of your wall are measured against it: on an otherwise exact
  part a 0.5 mm wall with one face on the photographed surface keeps full surface credit, and a thicker wall puts its
  other face that much further off.
- **No union.** Each solid's whole boundary counts as surface, the faces of an enclosed cavity included, and the solids
  of one tag are not united: a face of one solid buried inside another solid of the same tag counts as surface too, so
  fuse such solids or keep them apart.
- **Side skins.** The side skins seen on the assembled unit are not tagged parts and are not graded: leave them out.

## Practical notes

- You have six hours. Save a complete document early and keep saving it: what is in the submission folder when the
  session ends is what is graded.
- A single reply of yours can be at most 64k tokens long, and a reply that reaches the limit is lost: write long scripts
  in parts.
- An image viewer may show a photograph scaled down; crop a detail to a file to see it at full resolution.
- Keep scratch files out of the submission folder, for example in `/home/user/work/`.
- Before you finish, reopen the saved document and check what will be read: the twelve Labels, each member's solids and
  where they sit, and the joints.
- Which tools you use does not affect your score.
