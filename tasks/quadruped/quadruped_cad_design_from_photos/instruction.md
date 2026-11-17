# Re-model a modified 12-DoF quadruped robot in FreeCAD from a photo set

You are the mechanical designer of a lab that builds small quadruped robots: a body and four legs with three joints each
(hip abduction / adduction, hip flexion / extension, knee), every joint driven by the same bought belt-drive actuator unit,
every structural part 3D-printed. One robot was modified by a student (leg segment lengths, the body length, lightening
cut-outs) and its CAD is lost. Before the robot was shipped to a partner lab, a colleague took it apart and photographed
everything on the workbench with a phone: the 42 printed parts laid out on a cutting mat in seven groups, each part turned
over once, a few close-ups, then the re-assembled robot standing on the mat, in two other poses, and two of its legs in front
of a checker board. The photos are all there is: nobody can measure the robot again. Re-create the robot in FreeCAD so it can
be re-printed and simulated: the 42 printed parts as solids around the bought actuator units, assembled with the robot's
joints.

Your work is graded against the photographed robot's ground truth, which only the grader has: how well your mechanism, your
parts' features and shapes, your links' masses and centres of mass match the robot's, and whether the result works (the
actuator units fit their housings and drive their links, no self-collision over the robot's range of motion).

## How you work

- Do the work in FreeCAD through the FreeCAD MCP server `freecad`: it drives the FreeCAD 1.1 session running on this
  machine (documents, objects, views, and `execute_code` / `execute_code_async`, which run Python inside that session,
  with FreeCAD's modules: Part, PartDesign, Sketcher, Assembly, Import, Qt; its state persists between calls). Build the
  parts, place the actuator units and foot pads, set up the assembly and save the document there, as a designer working in
  FreeCAD would.
- Your other tools (reading files and images, a shell) are there for everything around it: looking at the photos and the
  components' documents, cropping a detail, notes and checks. There is no internet access. In the shell, the image's Python
  with its scientific packages (numpy, scipy, Pillow, trimesh and others) is `/opt/venv/bin/python3`; the `python3` on
  your `PATH` may be a bare interpreter.
- `get_view` and the screenshots the object tools return show FreeCAD's 3D view of the session. `Read` shows a photo
  scaled down to fit; to look at a detail at full resolution, crop it to a file first and read the crop. Keep scratch
  files under `/home/user/work/`.

## Given inputs (in `/home/user/`)

**`photos/photo_01.jpg` ... `photo_132.jpg`**: the fixed photo set of the robot. A paper tag printed with the part's id
(`P01` ... `P42`) lies next to every part in the teardown photos. The colleague bagged the parts group by group as they came
off and laid out one bag at a time.

| photos | what the photographer recorded |
|---|---|
| 01-12 | group 1, parts P01-P04 (from the body): first side up, six directions (01-06), then turned over (07-12) |
| 13-24 | group 2, parts P05-P12 (from the body): first side / turned over |
| 25-36 | group 3, parts P13-P20 (the four modules mounted across the body, opened): first side / turned over |
| 37-48 | group 4, parts P21-P25 (the front-left leg): first side / turned over |
| 49-60 | group 5, parts P26-P31 (the front-right leg): first side / turned over |
| 61-72 | group 6, parts P32-P37 (the hind-left leg): first side / turned over |
| 73-84 | group 7, parts P38-P42 (the hind-right leg): first side / turned over |
| 85-104 | close-ups, each subject next to the checker card: P03 (85-86, 101), P09 (87-88), P08 (89-90), P31 (91), P26 (92), P41 (93), P17 (94), P30 (95), P40 (96), P37 with P20, P15, P32 and P35 (97), P24 (98-99), P11 (100), P13 (102), P29 (103), P33 (104); 94-96 show an opened module half with an actuator unit lying in it as mounted, 97 a joint put together |
| 105-112 | the re-assembled robot standing on the mat, configuration **Z**: every actuator commanded to its zero position (the first four from the A4 board's printed side) |
| 113-120 | configuration A: a crouched stand (joint values not recorded) |
| 121-128 | configuration B: the body resting on a box, legs free (joint values not recorded) |
| 129-132 | the robot on its box, the front-right leg (129-130) and the hind-left leg (131-132) stretched straight out in front of the A4 checker board, the other legs folded (joint values not recorded) |

The printed parts are yellow. In the teardown photos the lower legs keep their glued foot pads, and the parts P05-P12 that
hold a ball bearing keep it pressed in (a bought 25 x 20 x 4 mm bearing: not modelled, not placed).

All photos come from one phone camera (wide lens, about 20-30 mm 35 mm-equivalent; not calibrated; no camera pose was
recorded; exposure and white balance vary a little). The actuator units are installed, and the robot's tether cable leaves
the body, in the assembled photos. Scale references, physical objects in the scene, with their nominal sizes:

- cutting mat: grey, A2 (600 x 450 mm), printed grid every 10 mm with every fifth line bold, numbered in cm; 2 mm thick, the
  parts lie on its top face, and the assembled robot stands on it;
- checker card: 10 x 7 squares of 12 mm (9 x 6 inner corners), 6 mm white margin, on a 132 x 96 x 1 mm card lying flat on
  the mat in every teardown and close-up photo;
- A4 checker board: 10 x 7 squares of 25 mm (9 x 6 inner corners), 10 mm white margin, on a 270 x 195 x 3 mm board standing
  beside the robot;
- two steel rulers (300 mm scale with 1 mm ticks, 316 x 25 x 0.9 mm), one on the mat, one on the bench;
- an ID-1 card (bank-card size, 85.6 x 53.98 mm, 0.76 mm thick) on the bench.

**`actuator/`**: the bought actuator unit, the same part at every joint: `datasheet_VK-BDJ-0945.png`,
`drawing_VK-BDJ-0945.png` (dimensioned outline), `VK-BDJ-0945.step` (vendor model) and `VK-BDJ-0945_keepout.step` (the space
the unit's moving parts sweep, with clearance: the envelope the grader intersects with your parts). It is a motor, an encoder
and a two-stage 9:1 timing-belt reduction without a housing: your printed parts are its housing and carry its bearing seats,
the stator mount, the encoder mount and the tensioner screws. In the unit's own frame, the origin is on the output axis at the
output pulley's mid-plane, +z is the output axis pointing toward the motor's rotor side (the motor's positive rotation is
right-handed about +z), and +x points from the output axis toward the motor axis. Datasheet numbers: output, centre shaft and
motor axes at x = 0 / 69.9 / 113.8 mm on y = 0; output pulley d 29.0 with an 18 x 10 mm (R1) rectangular through socket and
two M3 inserts at y = +-8 mm, faces at z = +-8.2 mm; output bearings 32 x 25 x 4 at z -8.25..-4.25 and 4.25..8.25; stator
spigot d 30.0, face at z = 10.45, 4 x M3 on a d 25 circle; rotor d 44.4 at z 16.85..29.65; mass 95 g, centre of mass
(81.54, 0, 11.95) mm. The driven link mates on either face of the output pulley (a key into the socket, two M3 screws); the
output turns with that link.

**`foot/`**: the foot pad, one per leg: `datasheet_VK-FP-2734-7.png`, `drawing_VK-FP-2734-7.png`, `VK-FP-2734-7.step`: a
rubber strip on a d 27 foot ring, outer d 34, 7 mm wide; frame: origin at the ring centre on the ring's mid-plane, +z = the ring
axis, +x toward the strip's gap; mass 2 g.

## Deliverable: `/home/user/submission/quadruped.FCStd`

One FreeCAD document, saved there, holding:

1. **42 PartDesign Bodies**, one per printed part, each labelled with the id on its paper tag (`P01` ... `P42`), each one valid
   solid, in millimetres. One Body per physical part: four identical parts are four Bodies, each with its own label (do not
   reuse one Body through links).
2. **The 12 actuator units and the 4 foot pads**: `actuator/VK-BDJ-0945.step` imported 12 times and `foot/VK-FP-2734-7.step`
   4 times, each copy placed where that unit or pad is mounted.
3. **One Assembly** (FreeCAD's Assembly workbench: an Assembly object with its joints) holding every part, unit and pad as a
   direct member (no sub-assemblies): one Grounded joint on a body part; Fixed joints for everything rigidly attached (parts
   screwed together, each unit to the parts that house it, each pad to its lower leg); one Revolute joint for every joint of
   the robot between the two parts it connects, with both angle limits enabled (Min / Max angle) at the range the joint moves
   through without self-collision, measured from the saved pose. Only Grounded, Fixed and Revolute joints.
4. **Saved in configuration Z**, the pose of photos 105-112.

The grader reads the document as saved; anything not saved in `quadruped.FCStd` is not seen.

## How it is graded

The grader reads your document with FreeCAD and works out your mechanism from it: parts by their labels, actuator units and
foot pads by their shape, links = parts, units and pads held together by Fixed joints, the Revolute joints between them, a
tree from the grounded body. It places your body onto the robot's body (by the four hip abduction units and the body parts'
shapes) and compares everything with the robot at configuration Z. A joint's driving unit is the unit in the joint's parent
link whose output axis lies on the joint axis; a joint's positive sense is that unit's +z, and its zero is the saved pose. A
leg's foot point is the origin of its foot pad. Each leg is judged on its own: a leg is valid when its three links carry the
same parts as the robot's (the same tags), its joints connect them in the robot's order, each joint lies on its driving
unit's output axis (within 1 deg and 0.5 mm) with the child resting on its parent or the unit (gap at most 0.2 mm), and its
lower leg carries a foot pad. The items of an invalid leg score 0; the rest of the robot is still scored.

**Gates** (each 0 or 1, multiplied into every score):

- `cad_valid`: the document opens and has one Assembly; the grounded group is the body; the mechanism is a tree without
  loops and with no other joint types; the parts convert into a valid rigid-body simulation model; the document was not
  written after the session ended.
- `public_model_reuse`: no part is a published model of a robot.

**Criteria** (weights), every band absolute against the robot: band(e; full, zero) = 1 at |e| <= full, 0 at |e| >= zero,
linear between.

- `kinematic_fidelity` (0.30): each joint's driving output centre at configuration Z (0.3 / 3 mm); key lengths: the body's
  hip spacing and, per leg, the hip offsets, the thigh and shank lengths and the knee and foot lateral offsets (0.3 / 3 mm); each
  leg's foot point over 64 random poses inside the robot's range (median 0.5 / 8 mm, 90th percentile 1 / 15 mm); how much of
  the robot's range your limits cover.
- `feature_fidelity` (0.25): cut-outs (position 0.3 / 3 mm, area 10 / 50 %), each part's extents (0.3 / 3 mm), the openings of
  the body's centre plates (0.3 / 3 mm).
- `shape_fidelity` (0.25): each part against the robot's after a rigid alignment: 90th-percentile surface distance
  (0.3 / 2.5 mm) and volume IoU (1 - IoU 0.20 / 0.75).
- `mass_properties` (0.10): each link's mass (solid volume x 1050 kg/m3 for the printed parts, plus the datasheet masses of
  the units and pads it carries; 2 / 15 %) and centre of mass at configuration Z (0.5 / 6 mm).
- `component_fit` (0.05): how well each unit is free of part material (keep-out), seated (its bearing seats, stator mount
  and screws carried by its housing) and engaged (the driven link's key in its output socket).
- `interference_free` (0.05): no collision between the links of a leg, or between a leg and the body, over the robot's range
  and your declared range.

Floors: the mean over the parts of band(the part's 90th-percentile surface distance; 2.5, 6.0 mm), a missing part counting
0, multiplies `kinematic_fidelity` and `feature_fidelity`; that times band(the leg's median foot-point error; 15, 60 mm)
multiplies each leg's `mass_properties`, `component_fit` and `interference_free` items (the body's items take the mean over
the legs). The photographed robot itself scores 1.0 on everything.

## Practical notes

- FreeCAD's Revolute angle limits are angles between the joint's two frames; set the joint frames so that the saved pose
  reads 0, or account for the offset. A joint that turns freely through a full turn may be declared -180 / +180 deg.
- Session time: this session is stopped after 6 hours of wall-clock time from its start. A session stopped by that limit
  is not graded at all, so a late finish scores nothing. Note the time when you begin (for example print
  `datetime.datetime.now()` through `execute_code`), save a complete document early and keep saving it, and end the
  session yourself (stop working and say that you are done) at least 15 minutes before the limit.
- Environment fact (harness limit): a response that reaches the 64k output-token limit is lost and, after at most a few
  retries, ends the session with whatever is in `/home/user/submission/` at that moment. Keep every single response
  well below that limit.
