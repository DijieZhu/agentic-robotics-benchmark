# Recover the CAD of one strut of a three-strut tensegrity robot

The lab's small rolling robot is a tensegrity: three identical struts held apart by a net of lines and elastic
sensors. The editable CAD of the strut is lost. To free one strut the net was cut away; a colleague then photographed
the complete strut on the electronics bench, took it apart step by step while photographing each stage, and
photographed every made part on a cutting mat. The strut is no longer available for measurement. Re-create its
mechanical design in FreeCAD from those photographs and the purchased components' documents.

The strut is an aluminium rod. In the middle it carries a two-piece printed enclosure around a custom main board with
plug-in electronics; near each end a motor module (printed pieces around a geared motor that turns a small printed
line spool), then a printed end cap with a moulded silicone cover. Both ends are built from the same parts: the parts
of a motor module are arranged identically at the two ends, and each module, end cap and cover is as far from the
rod's middle as its twin. How each motor module and each end cap is turned about the rod is part of what you recover.
Infer every made part's shape, dimensions and installed position from the evidence.

## How to work

Use the FreeCAD MCP server to inspect and edit the running FreeCAD session. Its `execute_code` tool runs Python inside
that session. Other tools remain available for image measurement, calculations and scripts. The Python environment is
`/opt/venv/bin/python3`; the native headless CAD executable is `/opt/freecad/usr/bin/freecadcmd`. There is no network
access.

Save early and periodically. You have six hours. A single model response is limited to 64k output tokens; do not wait
until the end to create the deliverable.

## Given files

Everything is in `/home/user/`: `photos/`, `purchased/`, and the empty folders `submission/` and `work/` (for scratch
files).

`photos/photo_001.jpg` through `photos/photo_287.jpg`. The teardown photos show every made part beside its paper tag
(P01 to P13), in two to four arrangements on the cutting mat. P04 and P05, P06 and P07, P08 and P09, P10 and P11, P12
and P13 are pairs of identical parts, photographed together, each part with its own tag; the photos of one part alone
(with its tag at 2x, in close-up at 5x) show only one part of each pair. The stage photos show the strut as it was
taken apart: the stages follow one another, and each caption says what was removed after the stage before it. Where a
row holds photos of two lenses, the lens column names the lens of most of them and, in brackets, the photos taken with
the other. The last column names the scale references that lay on the mat for those photos; a close photo has only
some of them in the frame.

| photos | what they show | lens | scale references on the mat |
|---|---|---|---|
| `photo_001`-`005` | P01, P02: arrangement 1 on the mat | 2x | mat, checker card, small card, rule |
| `photo_006`-`010` | P01, P02: arrangement 2 on the mat | 2x | mat, checker card, small card, rule |
| `photo_011`-`015`, `052` | P01, P02: arrangement 3 on the mat | 2x | mat, checker card, small card, rule |
| `photo_016`-`019` | P04, P05: arrangement 1 on the mat | 2x | mat, checker card, small card, rule |
| `photo_020`-`023` | P04, P05: arrangement 2 on the mat | 2x | mat, checker card, small card, rule |
| `photo_024`-`027` | P04, P05: arrangement 3 on the mat | 2x | mat, checker card, small card, rule |
| `photo_028`-`031` | P04, P05: arrangement 4 on the mat | 2x | mat, checker card, small card, rule |
| `photo_032`-`034` | P03: arrangement 1 on the mat | 2x | mat, checker card, small card, rule |
| `photo_035`-`037` | P03: arrangement 2 on the mat | 2x | mat, checker card, small card, rule |
| `photo_038`-`041`, `077`-`078`, `112`-`113` | all thirteen parts laid out together | 2x (1x: `112`-`113`) | mat, checker card, small card, rule |
| `photo_042`, `152`-`157`, `231`-`234` | P01 alone: resting position 1 | 5x (2x: `042`) | mat, small card, rule, standing small card |
| `photo_043`-`044`, `164`-`169`, `196`, `220` | P02 alone: resting position 1 | 5x (2x: `043`-`044`) | mat, small card, rule, standing small card |
| `photo_045`-`046`, `170`-`175` | P02 alone: resting position 2 | 5x (2x: `045`-`046`) | mat, small card, rule, standing small card |
| `photo_047`, `176`-`177`, `197`-`199`, `221` | P03 alone: resting position 1 | 5x (2x: `047`) | mat, small card, rule, standing small card |
| `photo_048`, `178`-`179`, `200`-`204`, `222`-`224` | P03 alone: resting position 2 | 5x (2x: `048`) | mat, small card, rule, standing small card |
| `photo_049`, `184`-`187`, `216`-`219`, `229`-`230` | P04 alone: resting position 2 | 5x (2x: `049`) | mat, small card, rule, standing small card |
| `photo_050`-`051`, `188`-`190` | P04 alone: resting position 3 | 5x (2x: `050`-`051`) | mat, small card, rule, standing small card |
| `photo_053`-`056`, `099`-`100` | P10-P13: arrangement 1 on the mat | 2x | mat, small card, rule |
| `photo_057`-`060`, `101`-`103` | P10-P13: arrangement 2 on the mat | 2x | mat, small card, rule |
| `photo_061`-`064`, `104` | P10-P13: arrangement 3 on the mat | 2x | mat, small card, rule |
| `photo_065`-`068`, `105`-`106` | P06-P09: arrangement 1 on the mat | 2x | mat, small card, rule |
| `photo_069`-`072`, `107`-`108` | P06-P09: arrangement 2 on the mat | 2x | mat, small card, rule |
| `photo_073`-`076`, `109`-`111` | P06-P09: arrangement 3 on the mat | 2x | mat, small card, rule |
| `photo_079`-`080`, `114`-`116`, `147` | P06 alone: resting position 1 | 5x (2x: `079`-`080`) | mat, small card, rule, standing small card |
| `photo_081`-`082`, `117`-`119`, `148`-`149` | P06 alone: resting position 2 | 5x (2x: `081`-`082`) | mat, small card, rule, standing small card |
| `photo_083`-`084`, `120`-`122`, `150` | P06 alone: resting position 3 | 5x (2x: `083`-`084`) | mat, small card, rule, standing small card |
| `photo_085`-`086`, `123`-`125` | P08 alone: resting position 1 | 5x (2x: `085`-`086`) | mat, small card, rule, standing small card |
| `photo_087`, `126`-`128` | P08 alone: resting position 2 | 5x (2x: `087`) | mat, small card, rule, standing small card |
| `photo_088`-`089`, `129`-`131` | P08 alone: resting position 3 | 5x (2x: `088`-`089`) | mat, small card, rule, standing small card |
| `photo_090`-`091`, `132`-`134` | P10 alone: resting position 1 | 5x (2x: `090`-`091`) | mat, small card, rule, standing small card |
| `photo_092`-`093`, `135`-`137`, `151` | P10 alone: resting position 2 | 5x (2x: `092`-`093`) | mat, small card, rule, standing small card |
| `photo_094`-`095`, `138`-`140` | P10 alone: resting position 3 | 5x (2x: `094`-`095`) | mat, small card, rule, standing small card |
| `photo_096`-`097`, `141`-`143` | P12 alone: resting position 1 | 5x (2x: `096`-`097`) | mat, small card, rule, standing small card |
| `photo_098`, `144`-`146` | P12 alone: resting position 2 | 5x (2x: `098`) | mat, small card, rule, standing small card |
| `photo_158`-`163`, `191`-`195` | P01 alone: resting position 2 | 5x | mat, small card, rule, standing small card |
| `photo_180`-`183`, `205`-`215`, `225`-`228`, `235` | P04 alone: resting position 1 | 5x | mat, small card, rule, standing small card |
| `photo_236`-`244` | assembled strut lying on the mat: resting position 1 | 1x | mat, checker card, small card, rule |
| `photo_245`-`248` | assembled strut lying on the mat: resting position 2 | 1x | mat, checker card, small card, rule |
| `photo_249`-`253` | assembled strut, one end half in view: resting position 1 | 2x | mat, checker card, small card, rule |
| `photo_254`-`256` | assembled strut, one end half in view: resting position 2 | 2x | mat, checker card, small card, rule |
| `photo_257`-`258` | assembled strut seen obliquely from beyond one end | 5x | mat, checker card, small card, rule, standing small card |
| `photo_259`-`260` | assembled strut seen obliquely from beyond the other end | 5x | mat, checker card, small card, rule, standing small card |
| `photo_261`-`263`, `279` | stage 1: P12 and P13 taken off | 2x (1x: `263`) | mat, checker card, small card, rule |
| `photo_264`-`266`, `280`-`281` | stage 2: P10 and P11 pulled off the rod ends | 5x (1x: `266`) | mat, checker card, small card, rule |
| `photo_267`-`269`, `282`-`283` | stage 3: P06 and P07 unscrewed and taken off | 5x (1x: `269`) | mat, checker card, small card, rule |
| `photo_270`-`271` | stage 4: P08 and P09 taken off the motor shafts | 2x (1x: `271`) | mat, checker card, small card, rule |
| `photo_272`-`273` | stage 5: P02 unscrewed and lifted off | 2x (1x: `273`) | mat, checker card, small card, rule |
| `photo_274`-`275` | stage 6: P04 and P05 slid off the rod ends, each with its gearmotor and battery cells still inside | 2x (1x: `275`) | mat, checker card, small card, rule |
| `photo_276`-`277` | stage 7: the plug-in boards unplugged from P03 (one small board stays under it) | 2x (1x: `277`) | mat, checker card, small card, rule |
| `photo_278` | stage 8: P01, with P03 still in it, slid off the rod; only the rod is left | 1x | mat, checker card, small card, rule |
| `photo_284` | the mat grid filling the frame, seen obliquely (scene: P10-P13, arrangement 2) | 5x | mat, small card, rule |
| `photo_285` | the mat grid filling the frame, seen obliquely (scene: P06-P09, arrangement 1) | 5x | mat, small card, rule |
| `photo_286`-`287` | the mat grid filling the frame, seen obliquely (scene: P04, P05, arrangement 2) | 1x | mat, checker card, small card, rule |

`purchased/<part number>/` for each purchased component: `<part number>.step` (its geometry in its own frame), `<part
number>_datasheet.md` (mass, frame, shaft and faces) and `<part number>_drawing.png` (dimensioned views).

| part number | component | in the strut |
|---|---|---:|
| `MCMASTER-8974K19` | McMaster-Carr 8974K19: aluminium rod, 1/8 in diameter x 12 in long, used uncut | 1 |
| `POLOLU-5192` | Pololu 5192: 150:1 Micro Metal Gearmotor HPCB 6V with 12 CPR Encoder, Back Connector | 2 |

Scale references, with their nominal sizes. Cutting mat: A3 (420 x 297 mm), 2 mm thick, with a 10 mm grid whose every
fifth line is bold and centimetre numbers along two margins; the parts rest on its top face. Checker card: 10 x 7
squares of 12 mm inside a 6 mm white border, 132 x 96 mm, 1 mm thick. Small checker card: 10 x 7 squares of 2 mm
inside a 3 mm white border, 26 x 20 mm, 1 mm thick; in the scenes of one part alone and in the end views one such card
also stands on its long edge against a plain block (30 x 14 x 24 mm), leaning back 8 deg. Steel rule: 160 x 18 mm, 0.5
mm thick, with a 150 mm scale graduated in millimetres (numbered every centimetre) on one edge and in half millimetres
on the other. Paper tags: 30 x 14 mm.

One hand-held phone took all photos with three lenses whose focal lengths are in the ratio 1 : 2 : 5 (1x, 2x and 5x in
the table). The camera is not calibrated and no camera pose was recorded. Every photo is tilted and rolled: none looks
square onto the mat or onto a part. Exposure and white balance vary a little, and file names do not encode the pose.

Materials (solid density): printed polymer 1.24 g/cm3 for P01, P02 and P04 to P09; printed resin 1.2 g/cm3 for P10 and
P11; silicone 1.08 g/cm3 for P12 and P13; board laminate 1.85 g/cm3 for P03. Purchased masses are in their datasheets.

## Deliverable

Save `/home/user/submission/strut.FCStd` with:

1. One valid, closed B-rep solid in a PartDesign Body for each tag P01 to P13, with the Body's Label exactly its paper
   tag. Use each tag once: one Body per tag, not reused through links. Which part of an identical pair sits at which
   end is not graded.
2. Every purchased component of the table, imported unchanged from its STEP file, as a separate shape object in its
   installed position. Do not fuse, simplify or re-model it.
3. One native FreeCAD Assembly containing all of these: a Grounded joint on P01, Fixed joints for everything rigidly
   attached, and each line spool joined to its motor by a Revolute joint about the motor's output shaft, without
   limits. Only what is inside the Assembly is graded.
4. The assembly saved as photographed assembled, covers on. The spools' turning position is free. Origin and overall
   orientation are arbitrary; units are millimetres. Do not include a ground plane, the mat, alternative versions or
   duplicate solids.

Outside the modelling scope (do not model them; anything extra counts as an extra solid, see below): screws, set
screws, spacers and the threaded inserts (the inserts stay in the printed parts in every photo; the hole an insert
sits in belongs to the printed part); the line guide; the plug-in boards and everything soldered to the main board
(P03 is the bare board only); the battery cells with their leads; all wiring, the lines and the elastic sensors.

Only `strut.FCStd` is read. Anything else in the submission folder must be your own work: a copy of an existing mesh
or CAD model from elsewhere scores zero.

## How it is graded

The verifier opens the saved document with FreeCAD, exports every solid in world coordinates and reads the joints.
Nothing in the document runs and the assembly is not solved again: the saved placements count. It grades what is
inside the Assembly: its members, the members of App::Part containers and folders inside it, and links. A document
without an Assembly is graded from its visible solids, with no credit for joints.

A tagged part that is not a PartDesign Body of exactly one valid closed solid gets no credit, and neither does a tag
on more than one object (a Body reused through links counts as its tag used twice); their solids count as extra
solids. Invalid solids, open shells and loose faces are left out. Purchased components are recognised by their full
shape, never by label: volume within 1 %, then a rigid registration onto the given solid. Extra solids, meaning
anything neither credited to a tag nor recognised as a purchased component, count in the mass at 1.24 g/cm3 and as
obstacles in fit and clearance.

Each tagged part is first fitted rigidly, on its own, onto the original part, wherever it sits in your document: its
shape, its holes and its flat faces are graded in that fit, so a misplaced part keeps that credit. Where a part fits
several ways almost equally well, the way that suits its own holes and flat faces is taken, then the one nearest its
place in the strut. Where the parts sit is graded separately, after the whole strut has been fitted rigidly (no
scaling, no mirroring) onto the original: every fitted part and every recognised component takes part in that fit, and
no single part is its base. Identical parts are matched to their places one-to-one. Size classes: P06 to P13 small,
P01 to P05 large. Bands are written full credit / no credit: an error up to the first number earns full credit, from
the second none, linear in between.

| Criterion | Weight | What is measured |
|---|---:|---|
| Shape | 0.25 | Each of the 13 parts in its own fit. Surface error (70 %): the 90th percentile of the distance from the original's surface to yours, and of the distance from your surface to the original's, whichever is worse, taken on the surface that the photographs let one locate: 0.12 / 0.6 mm small, 0.2 / 1 mm large. Solid overlap of the whole part (IoU, 30 %): none below 0.4, full from 0.95. The parts are averaged with weights of one plus the number of their graded hole steps and flat faces; a missing or uncredited part scores zero and keeps its weight. |
| Holes | 0.2 | Every step of every hole, bore and counterbore of the original that the photographs let one locate, matched one-to-one to the concave cylindrical faces of your part in its own fit: axis offset (the distance of the original step's middle from your hole's axis, combined with any gap along that axis between the two; same bands as the surface error) x radius error (0.08 / 0.35 mm; counted only where a photograph shows the hole's mouth clear, not where an insert or a screw sits in it) x axis angle (2 / 10 deg). Depth is not measured. An unmatched step scores zero. Averaged over all graded steps of the strut. |
| Flat faces | 0.15 | Every flat face of the original of 2 mm2 or more that the photographs let one locate (rib tops, pocket floors, steps, window edges, tabs, end faces; the facets of a rounded surface are not flat faces). From every point of such a face, the distance to the nearest surface of your part that faces the same way (outward normals within 30 deg; 5 mm when there is none within 5 mm); the median over the face is its error, on the same bands as the surface error. So a rib left out is read as its height, a pocket not cut as its depth, a face out of place as its offset. Averaged over all such faces of the strut. |
| Functional geometry | 0.1 | Relations between your own parts, measured on your geometry: the quantities of the second table, weighted by their shares. |
| Placement | 0.15 | 50 %: each made part's pose in the fitted strut: position of its centroid (0.5 / 4 mm) x (0.7 + 0.3 x orientation, 1.5 / 12 deg). 30 %: each gearmotor in the same way (0.4 / 3 mm; 1.5 / 12 deg) and the rod by its axis: sideways offset (0.2 / 1.5 mm) x position along the axis (0.5 / 4 mm) x angle (0.3 / 3 deg); the purchased components weigh the same. A pose that is a symmetric equivalent of the original counts as exact: any turn of a cover, a spool or the rod about its own axis, a spool or the rod end for end, a quarter turn of an end cap. 20 %: 60 % of it for the made parts (half) and purchased components (half) connected to the grounded part through unsuppressed Fixed joints, a spool through its Revolute joint; 40 % of it for the spools: each needs an unsuppressed Revolute joint to its gearmotor, or to a part fixed to that gearmotor, whose axis lies on the motor's output axis: distance (0.2 / 1 mm) x angle (1 / 8 deg). A spool held by a Fixed joint loses only that share. |
| Fit and clearance | 0.1 | 40 %: each purchased component's contact with the made parts next to it in the original (its smallest distance to each may exceed the original's by 0.1 mm for full credit, none from 2 mm) x its penetration into made parts and extra solids (0.05 / 5 % of its volume). 30 %: each made part's penetration into the other made parts and extra solids, on the same band of its volume. 30 %: each spool turns freely: the cylinder swept by the original spool's flanges, carried on your spool's axis, against everything except that spool and its gearmotor; overlap 0.5 / 40 mm3. Overlaps that the original design has itself are exempt, and an overlap is charged to the piece that is further from the original. |
| Mass properties | 0.05 | Of everything in scope, in the fitted strut: total mass (2 / 25 %; 30 %), centre of mass (0.5 / 5 mm; 35 %) and principal moments of inertia, largest relative error (4 / 50 %; 35 %). Made parts at their material density, purchased components at their datasheet mass spread evenly over the given solid, extra solids at 1.24 g/cm3. What is outside the modelling scope counts on neither side. |

Functional geometry. The surface behind each quantity (a sphere, a bore, a post, a drum) is looked for on your own
part, where the original's lies in that part's fit: within 3 mm, and within 50 % of its radius. A quantity whose
surface is not found scores zero for that quantity only. Distances along the rod and turns about the rod use the axis
of the imported rod, or without it the line through the two end caps' bores. The middle enclosure's direction about
the rod is taken from its two halves; one suffices, a half whose shape score is below 0.2 gives none, and with neither
the quantities relative to the middle enclosure score zero. A drum is the cylinder between a spool's flanges; the
posts are the round pegs under an end cap's dome.

| quantity | share | band |
|---|---:|---|
| distance between the centres of the two covers' outer spheres | 15 % | 0.6 / 4 mm |
| radius of each cover's outer sphere | 5 % | 0.12 / 0.6 mm |
| distance of each spool's axis from the rod's axis | 15 % | 0.15 / 1 mm |
| distance along the rod from the middle of each spool's drum to the centre of its end's cover sphere | 10 % | 0.3 / 2 mm |
| diameter of each spool's drum | 5 % | 0.1 / 0.6 mm |
| distance along the rod from the middle of each end-cap post to the centre of its end's cover sphere | 10 % | 0.3 / 2 mm |
| distance of each end-cap post's axis from the axis of the end cap's bore | 5 % | 0.15 / 1 mm |
| direction about the rod in which each spool's axis lies off the rod's axis, relative to the middle enclosure, modulo a half turn | 10 % | 2 / 12 deg |
| angle about the rod between the directions of the two spools' axes | 5 % | 2 / 12 deg |
| turn about the rod of each end cap's pattern of posts, relative to the middle enclosure, modulo a quarter turn | 10 % | 2 / 12 deg |
| distance along the rod from the rod's middle to the middle of the larger printed piece of each motor module | 10 % | 0.5 / 4 mm |

Shape floor: holes, flat faces, functional geometry, placement, fit and clearance, and mass are multiplied by a factor
that rises linearly from zero at a weighted mean shape score of 0.3 (the most that solid overlap alone can give) to
one at 0.8. design_score = 0.25 shape + factor x (0.2 holes + 0.15 flat faces + 0.1 functional geometry + 0.15
placement + 0.1 fit and clearance + 0.05 mass); the reward is the design score.

Only three things score zero as a whole: the document cannot be read (one with more than 2000 solids counts as
unreadable); it holds no solid; fewer than 3 pieces, credited tagged parts and recognised purchased components
together, can be fitted. A submission folder written to after the session has ended is void. Everything else is local,
as described above.

## Notes

- There is no checker, template or practice grader in this environment. Nothing in this text or in the purchased
  components' documents states a dimension of a made part.
- The choice of shell or MCP tools does not affect your score. Geometry and assembly correctness do.
