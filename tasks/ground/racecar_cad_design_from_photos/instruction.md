# Recover the CAD of a tenth-scale autonomous race car

The lab wants to build a second unit of its tenth-scale four-wheel-drive autonomous race car, but the CAD of the car's
made parts is lost: only the solids of the purchased components and of the rolling base assembly are left. A colleague
photographed the complete car, took it apart stage by stage while photographing every stage, and photographed each made
part on a cutting mat, from both sides. The car is in pieces now and cannot be measured again. Re-create its mechanical
design in FreeCAD from those photographs and the documents of the given components.

The car has 23 made parts in 14 different shapes (moulded, cut and printed polymer). Infer each one's shape, dimensions
and installed position from the evidence. This is a display car: reproduce it exactly as photographed, also where parts
seem to run into each other or to hang free. The reference is the car as it stands in the photographs, not a corrected
version of it.

## How to work

Use the FreeCAD MCP server to inspect and edit the running FreeCAD session. Its `execute_code` tool runs Python inside
that session. Other tools remain available for image measurement, calculations and scripts. The Python environment is
`/opt/venv/bin/python3`; the headless CAD executable is `/opt/freecad/usr/bin/freecadcmd`.

Save early and periodically. You have six hours. A single model response is limited to 64k output tokens; do not wait
until the end to create the deliverable.

The given files are large: the base assembly has 146 solids and the compute module 697. Import each STEP file as one
object holding all its solids (in Python: `Part.read(path)` as the Shape of one `Part::Feature`), not as one object per
solid, and do not list or print the document's objects one by one. Give `execute_code` a timeout of several minutes for
these two imports; a call that reports a timeout keeps running, so look at the document before importing again. Nothing
in the environment evaluates a design against the reference.

## Given files (in `/home/user/`)

`photos/photo_001.jpg` to `photos/photo_087.jpg`. Teardown photos show every made part next to its paper tag, first one
side up, then turned over. Parts with the same shape were photographed together, each with its own tag. The tags run
from P01 to P23. Close-ups show holes and other small details of the tagged parts. The photos of the complete car show
it standing on the bench, from the side each caption names. The stage photos show the car as it was taken apart. Each
caption names what was taken off in that stage; everything taken off in an earlier stage stays off. The last stage shows
the car turned over, in the state its caption names. The camera is perspective and uncalibrated, and file names do not
encode its pose.

| photos | what they show | scale references |
|---|---|---|
| `photo_001`-`002` | P01, first side up | cutting mat A3 (10 mm grid), checker card, steel rule |
| `photo_003`-`004` | P01, turned over | cutting mat A3 (10 mm grid), checker card, steel rule |
| `photo_005`-`006` | P02, P03, first side up | cutting mat A3 (10 mm grid), checker card, steel rule |
| `photo_007`-`008` | P02, P03, turned over | cutting mat A3 (10 mm grid), checker card, steel rule |
| `photo_009`-`010` | P04, first side up | cutting mat A2 (10 mm grid), checker card, steel rule |
| `photo_011`-`012` | P04, turned over | cutting mat A2 (10 mm grid), checker card, steel rule |
| `photo_013`-`014` | P05, P06, first side up | cutting mat A3 (10 mm grid), checker card |
| `photo_015`-`016` | P05, P06, turned over | cutting mat A3 (10 mm grid), checker card |
| `photo_017`-`018` | P07, P08, first side up | cutting mat A3 (10 mm grid), checker card |
| `photo_019`-`020` | P07, P08, turned over | cutting mat A3 (10 mm grid), checker card |
| `photo_021`-`022` | P09, P10, first side up | cutting mat A3 (10 mm grid), checker card |
| `photo_023`-`024` | P09, P10, turned over | cutting mat A3 (10 mm grid), checker card |
| `photo_025`-`026` | P11, P12, first side up | cutting mat A3 (10 mm grid), checker card |
| `photo_027`-`028` | P11, P12, turned over | cutting mat A3 (10 mm grid), checker card |
| `photo_029`-`030` | P13, first side up | cutting mat A3 (10 mm grid), checker card |
| `photo_031`-`032` | P13, turned over | cutting mat A3 (10 mm grid), checker card |
| `photo_033`-`034` | P14, P15, first side up | cutting mat A3 (10 mm grid), checker card |
| `photo_035`-`036` | P14, P15, turned over | cutting mat A3 (10 mm grid), checker card |
| `photo_037`-`038` | P16, first side up | cutting mat A3 (10 mm grid), checker card |
| `photo_039`-`040` | P16, turned over | cutting mat A3 (10 mm grid), checker card |
| `photo_041`-`042` | P17, P18, first side up | cutting mat A3 (10 mm grid), checker card |
| `photo_043`-`044` | P17, P18, turned over | cutting mat A3 (10 mm grid), checker card |
| `photo_045`-`046` | P19, first side up | cutting mat A3 (10 mm grid), checker card |
| `photo_047`-`048` | P19, turned over | cutting mat A3 (10 mm grid), checker card |
| `photo_049`-`050` | P20, P21, first side up | cutting mat A3 (10 mm grid), checker card |
| `photo_051`-`052` | P20, P21, turned over | cutting mat A3 (10 mm grid), checker card |
| `photo_053`-`054` | P22, P23, first side up | cutting mat A3 (10 mm grid), checker card |
| `photo_055`-`056` | P22, P23, turned over | cutting mat A3 (10 mm grid), checker card |
| `photo_057` | complete car on the bench, seen from the front | checker card, steel rule standing upright |
| `photo_058` | complete car on the bench, seen from the rear | checker card, steel rule standing upright |
| `photo_059` | complete car on the bench, seen from its left side | checker card, steel rule standing upright |
| `photo_060` | complete car on the bench, seen from its right side | checker card, steel rule standing upright |
| `photo_061` | complete car on the bench, seen from above | checker card, steel rule standing upright |
| `photo_062` | complete car on the bench, three-quarter view from the front left | checker card, steel rule standing upright |
| `photo_063`-`064` | stage 1: 2D scanning LiDAR, compute module, motor controller, power distribution board and both Wi-Fi antennas removed, with the spacer pins belonging to them | checker card, steel rule |
| `photo_065`-`066` | stage 2: P04, P16 and the remaining spacer pins removed | checker card, steel rule |
| `photo_067`-`068` | stage 3: P01, P13 and the front foam bumper removed | checker card, steel rule |
| `photo_069`-`070` | stage 4: P14, the two front dampers and the two front wheel sets removed | checker card, steel rule |
| `photo_071` | stage 4, rear end of the car: the two rear dampers are still in place | checker card (partly in frame), steel rule |
| `photo_072` | stage 4, rear end of the car: the two rear dampers are still in place | checker card, steel rule |
| `photo_073`-`074` | stage 5: P05, the front differential and the two front drive shafts removed | checker card, steel rule |
| `photo_075`-`076` | stage 6: P07, P09, P11 and P22 removed | checker card, steel rule |
| `photo_077`-`078` | stage 7: car turned over and resting on two wooden blocks, in the state after stage 3 | checker card, steel rule |
| `photo_079` | close-up of P02, P03, first side up | cutting mat A3 (10 mm grid), steel rule |
| `photo_080` | close-up of P03, turned over | cutting mat A3 (10 mm grid), steel rule |
| `photo_081` | close-up of P02, turned over | cutting mat A3 (10 mm grid), steel rule |
| `photo_082`-`084` | close-up of P04, first side up | cutting mat A2 (10 mm grid), steel rule |
| `photo_085` | close-up of P05, first side up | cutting mat A3 (10 mm grid), steel rule |
| `photo_086` | close-up of P07, first side up | cutting mat A3 (10 mm grid), steel rule |
| `photo_087` | close-up of P08, turned over | cutting mat A3 (10 mm grid), steel rule |

Scale references in the photographs: cutting mat A3: 420 x 297 mm, 10 mm grid, every fifth line bold, the bold lines
numbered in centimetres along two margins; cutting mat A2: 594 x 420 mm, same grid and numbering; checker card: ten by
seven squares of 12 mm inside a 12 mm white border, 144 x 108 mm overall; steel rule: 300 mm scale graduated in
millimetres and numbered every centimetre.

`base/base_assembly.step` with `base/README.md`: the rolling base assembly, the part of the car that is given complete,
in the car's own frame. `purchased/<part number>/` for each other given component: `<part number>.step` (its geometry in
its own frame; a component supplied assembled has several solids in one file), `<part number>_datasheet.md` (mass,
frame, size, mounting interface) and `<part number>_drawing.png`.

| part number | component | in the car | files | grading weight |
|---|---|---:|---|---:|
| `GLH-C2C4` | rolling base assembly | 1 | `base/` | 0 |
| `GLH-X2BG` | front wheel set `*` | 2 | `purchased/GLH-X2BG/` | 2 |
| `GLH-6SN7` | damper | 4 | `purchased/GLH-6SN7/` | 1 |
| `GLH-1PSW` | differential `*` | 2 | `purchased/GLH-1PSW/` | 2 |
| `GLH-A697` | front drive shaft `*` | 2 | `purchased/GLH-A697/` | 1 |
| `GLH-9VJD` | front foam bumper | 1 | `purchased/GLH-9VJD/` | 1 |
| `EST-012L` | 2D scanning LiDAR | 1 | `purchased/EST-012L/` | 3 |
| `WNL-QCNP` | compute module | 1 | `purchased/WNL-QCNP/` | 3 |
| `PRD-C0MV` | motor controller | 1 | `purchased/PRD-C0MV/` | 1.5 |
| `RTN-MEHX` | power distribution board | 1 | `purchased/RTN-MEHX/` | 1 |
| `GLH-HZ2S` | Wi-Fi antenna | 2 | `purchased/GLH-HZ2S/` | 1 |
| `SP-2.6-7.42` | spacer pin SP-2.6-7.42 | 2 | `purchased/SP-2.6-7.42/` | 0.25 |
| `SP-2.8-17.43` | spacer pin SP-2.8-17.43 | 8 | `purchased/SP-2.8-17.43/` | 0.25 |
| `SP-2.8-27.02` | spacer pin SP-2.8-27.02 | 3 | `purchased/SP-2.8-27.02/` | 0.25 |
| `SP-2.8-48.20` | spacer pin SP-2.8-48.20 | 4 | `purchased/SP-2.8-48.20/` | 0.25 |
| `SP-4.0-14.15` | spacer pin SP-4.0-14.15 | 4 | `purchased/SP-4.0-14.15/` | 0.25 |

Grading weight: the weight of one piece in the installation and fit criteria below (the base assembly has none: the car
is aligned on it). `*`: the turn about the component's own axis of rotation, the z axis of its file, is not graded.

`parts_list.md`: the made parts by tag, with pieces, material and density: injection-moulded polymer 1.10 g/cm³
(P01-P03, P05-P15, P17-P23), polymer plate (cut) 1.20 g/cm³ (P04), 3D-printed polymer 1.24 g/cm³ (P16). The masses of
the given components are in `base/README.md` and in the datasheets. No drawing, dimension or hole table of a made part
is given, and no document states which tagged part sits where or what it is fastened to; the stage captions name only
what was taken off together.

## Deliverable

Save `/home/user/submission/racecar.FCStd` with:

1. One PartDesign Body for each tag P01 to P23, with the Body's **Label** exactly the paper tag and exactly one closed
   solid in it. Use each tag once: one Body per tag, not reused through links. Parts photographed together have the same
   shape; give each its own Body and tag. Which of several identical parts sat where is not graded.
2. `base/base_assembly.step` imported unchanged and moved only as a whole, and every other given component of the table,
   each piece imported unchanged from its STEP file and placed where it is installed (38 pieces of 15 components). Keep
   the solids of a file in their places relative to each other; do not fuse, simplify, scale or re-model them.
3. One native FreeCAD Assembly that contains all of these, with the base assembly grounded and every made part and every
   given piece connected by a joint to what it is mounted on. The kind of joint does not matter: only the connection is
   graded. Only what is inside the Assembly, directly or through links, is graded.
4. The Assembly saved in the state of the complete car in the photographs. The saved placements are graded as they are;
   the Assembly is not solved again. Creating a joint lets FreeCAD's solver move the parts until the two joint frames
   coincide: check the placements after adding joints, before the last save. Origin and overall orientation are free;
   units are millimetres. Do not include a ground plane, mat, photo scene, alternative versions or duplicate solids.
   Keep everything in this one file: only `/home/user/submission` reaches the verifier, and a link to an object of a
   document saved elsewhere is empty there.

Screws, nuts and washers, the ball studs with their sockets, the wiring with its plugs, the rear wing frame and whatever
else is neither a tagged part nor in a given file appear in the photographs but are outside the modelling scope: do not
model them (anything extra counts as an extra solid, see below).

## How it is graded

The verifier runs in a separate sandbox after your session. It opens the saved document with FreeCAD (nothing in it is
executed), exports every solid in world coordinates and reads the joints. It grades what is inside the Assembly: its
members, the members of App::Part containers and folders inside it, and links. Objects outside the Assembly are ignored,
the base assembly included: left outside the Assembly, it counts as missing. A document without an Assembly is graded
from its visible solids, with no credit for joints. A solid counts when it is closed (`Shape.isClosed()`) and has a
volume; its geometric validity is not checked. A solid whose surface cannot be meshed as a closed volume of its own size
is replaced by its convex hull in the volume measures (overlap and penetration).

The whole document scores 0 in three cases only: it is missing or cannot be read (a document with more than 3000 solids,
or one that FreeCAD needs more than 20 minutes to open and export, counts as unreadable; the complete car has 975
solids); it holds no closed solid; or the tub of the base assembly, as imported from `base/base_assembly.step`, is
missing or present more than once, so that the car cannot be aligned. Besides, a submission folder that holds an
unchanged copy of a public source file of this car's CAD model scores 0, and so does one that is still written to after
your session has ended.

Everything else is local and reported. A tag that is not on a PartDesign Body of exactly one closed solid and nothing
else gets no credit, and neither does a tag found on more than one object (a Body reused through several links counts as
its tag used twice); the solids concerned count as extra solids. Open shells and loose faces are left out. Extra solids
are all solids that are neither a credited tagged part nor part of a recognised given piece: they count in the mass (a
stray or left-over copy of a given solid at the density of a component that holds this solid, anything else at 1.10
g/cm³) and are obstacles in the fit criterion. A given piece that is in the document more often than in the car is extra
as well: the solids of the copies left over after the one-to-one matching are extra solids.

Your document is aligned with the reference rigidly, without scaling or mirroring, on the base assembly alone: the pose
of its tub in your document is taken onto its pose in the reference. After that nothing is resized, turned or moved into
place, neither a made part nor a given piece. Given components are recognised by their full shape, never by a label:
each solid must be the given geometry (volume within 1 %, then a rigid registration), and the solids of one file must
keep their places relative to each other within 1 mm; one that does not is an extra solid and leaves its piece
incomplete, in the base assembly too. The base assembly is put together first, then the other components. The pieces of
one component are matched to its installed positions one-to-one, and so are the tagged parts of one shape (one line of
`parts_list.md`): it does not matter which of them carries which tag.

| Criterion | Weight | What is measured |
|---|---:|---|
| Shape | 0.30 | Each of the 23 made parts where it is installed, against the true part there. Surface distance, both ways, 90th percentile: full credit up to 0.35 mm, none from 5 mm (60 % of the part's score). Volume overlap (intersection over union): none up to 0.25, full from 0.95 (40 %). A missing part scores 0. Mean over the 23 parts. |
| Holes | 0.25 | The round holes of the made parts: a graded hole is one concave cylindrical wall, with a radius up to 12 mm, that covers at least 300° about its axis; in a stepped hole (a counterbore) each such step is graded, whether or not the other steps close, and a countersink cone is graded where every cylindrical step of its hole is such a wall. They are matched one-to-one to the concave cylindrical or conical faces of your part: axis offset (full up to 0.25 mm, none from 1.5 mm) × radius error (full up to 0.15 mm, none from 0.6 mm) × axis angle (full up to 2°, none from 10°). A hole that is slightly out of round is graded as the cylinder fitted to its wall, so a round hole of its mean radius scores like the exact one; a face of yours that is no exact cylinder counts when it fits one within 6 % of the radius (at least 0.02 mm). Mean per part, then over the parts that have such holes. Walls that do not close (open slots, half-round recesses) are no holes; they belong to the shape. |
| Installation | 0.20 | 80 %: every given piece scores its centroid position (full up to 0.6 mm, none from 12 mm) × (0.7 + 0.3 × orientation, full up to 0.5°, none from 12°) × the share of its solids present, with the weights of the table above. A pose that is a symmetric equivalent of the original counts as exact (a spacer pin turned about its axis or end for end, for instance); for the components marked `*` only the axis counts, with the sense in which the z axis of the file points along it (turned end for end, such a component is 180° off). 20 %: the made parts (half) and the given pieces (half, weighted) that are connected to the grounded base assembly through unsuppressed joints of any kind, directly or through other components; grounding a component does not connect it. |
| Mass properties | 0.10 | Total mass: full up to 2 %, none from 25 % (30 %). Centre of mass: full up to 1 mm, none from 15 mm (35 %). Principal moments of inertia, largest relative error: full up to 4 %, none from 50 % (35 %). Made parts count with their material's density, the solids of a given component with the density that gives the component its documented mass, extra solids as stated above. The items outside the modelling scope are on neither side. |
| Fit | 0.15 | 60 %: every given piece, weighted as above, scores its contact with what it rests on in the car (full up to 0.1 mm farther apart than there, none from 2 mm farther) × its penetration into made parts and extra solids (full up to 0.05 %, none from 5 % of its own volume). 40 %: each made part's penetration into the other made parts, the given pieces (every solid of the base assembly included) and extra solids, on the same band of the part's own volume. What the car itself has of overlap between the same two items is exempt, and 0.5 mm³ more per pair. A missing part or piece scores 0. |

**Shape floor:** holes, installation, mass properties and fit are multiplied by a factor that rises linearly from 0 at a
mean shape score of 0.12 to 1 at 0.55. Correct given components around crude made parts therefore collect little beyond
the shape credit. design_score = 0.30 shape + 0.25 holes + 0.20 installation + 0.10 mass properties + 0.15 fit; the
reward is the design score.

Whether you work through the MCP server or the shell does not affect your score. Geometry and assembly correctness do.
