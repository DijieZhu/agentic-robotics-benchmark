# NOTICE (grader side; never staged into the agent's home)

This file names the real sources behind the task's generic surface, says what the agent's files and the photographs
are made of, and records for maintainers every place where they depart from the vendor's models or where a pose is
less than exact. The agent-visible material (`instruction.md`, the payload: photographs, `purchased/`,
`parts_list.md`) names no platform, maker, product, part number, simulator, dataset or author: the vehicle is "an
inspection-class ROV", the purchased modules carry generic names and invented part numbers. How the grader scores is
in `README.md`. The benchmark's owner has stated that the sources below may be used and distributed with the
benchmark.

## 1. The vehicle and its sources

The vehicle is the **BlueROV2 R4 in its Heavy (8-thruster) configuration by Blue Robotics**, with the accessories of the
underwater tasks' `mission` configuration: the Newton subsea gripper, the Ping360 scanning imaging sonar, a Water Linked
DVL A50 on its stainless bracket, four Lumen lights, four 200 g ballast weights, no payload skid. It is the one vehicle
of the platform's four tasks, measured once into one frozen description of the reference vehicle that the four tasks
share (its parts manifest, sha256 `99b937ff958bf88e28a866059a887fcfeb3981b80e6a5b677da5056fbf448abe`, and its table of
values, sha256 `358676915495f3ea4e719154634557669b14a69a7554dce6a134cb80bf3b9bbd`). No hidden variant of it exists.

The truth of this task is the vendor's B-rep at the reference vehicle's poses: no solid was rebuilt, repaired,
simplified or redesigned. One change was made to the vendor's solids: the maker's emblem was taken off six heat pads of
the electronics module (section 2); every other face is the vendor's. The solids come from these public files of the
vendor and of the DVL's maker (sha256 of the files as downloaded):

| File | sha256 | What is taken from it |
|---|---|---|
| `BLUEROV2-R4-PUBLIC.STEP` | `f1ec8be6bccdd2d91b9931f43992c46c148b77c4de032fca79a7eb68db0a3deb` | the frame's seven plates, the tube cradles, clips and clamp halves, the four corner fairings and corner foam blocks, both enclosures with their contents, the eight thrusters (both propeller hands), the ballast weight, the fasteners |
| `BROV2-HEAVY-P-FAIRING-X1G.STEP` | `298cf4b8d052e1af80ae68786b9a174e72a2a53b3e5c4d0a215a4cf0144c865c` | the two centre fairing caps of the Heavy retrofit |
| `BROV2-HEAVY-P-BUOYANCY-X1B1.STEP` | `9b635531cac39d387f1ceddb1211b8f9a078516c6f4a6f83ad9e93ebc8c680a8` | the two round foam blocks |
| `BROV2-HEAVY-P-GUARD-X1B.STEP` | `663b86be613aae53e2d97821fe48e0e327ea473303dfa77ee2871cf5e34a9375` | the two vertical-thruster guard plates |
| `NEWTON-MOUNT-ASM-X2A.STEP` | `63755688cbc92e5861d71947eab507f3e57e574c366ceb567799d7368be6834b` | the two gripper clamp halves |
| `PING360-M-FRAME-MOUNT-R1.STEP` | `0a8eb4a9a5e247ea7a8ad6ec0dcc521f50ed51ba2641703466d77ab4a7ab57f6` | the sonar's frame mount |
| `WL-52046-2_DVL_Bracket_BROV2.step` | `21e2d7e737c66e19677d8b6e6fe4c24b38e0959084850805ecb11fc1efe7d8d1` | the DVL bracket |
| `LUMEN-ASM-R3-PUBLIC.STEP` | `2682f8646778c2a61c2d8de5403b5e812e120c4f43d6d9e4613efea607161f60` | the light |
| `NEWTON-GRIPPER-ASSEM-R3-PUBLIC.STEP` | `6bd6a51c0579f41165467ff001fb835fc238cb8d30b18ac9dbae15ab96e71b2d` | the gripper |
| `WL-21035-4-S--DVL-A50--PUBLIC.stp` | `8e473181b1a776986f6867b514a60851d7ca5736162e6bfe18c0673882ed381d` | the DVL |
| `BATTERY-LI-4S-18AH-R1.STEP` | `d79912e904b2ce58eff8c9df99e770873bafcb706285fc0da51521b2468b21de` | the battery pack inside the battery enclosure |

## 2. What the agent's files are made of

**The made parts** (31 instances, 16 shapes, tags P01 to P31) are in no file the agent gets: their solids are
`reference/parts/Pnn.brep`. P01 is the frame's bottom plate; the order of the other tags is a seeded hash, checked to
be unrelated to the stage a part leaves in, to its place along, across and up the vehicle, and to its mass. Six parts
that the vendor also sells separately (the two tube clamp halves, the two gripper clamp halves, the sonar's frame
mount, the DVL bracket) are among the made parts.

**The purchased modules** (8 types, 20 instances, 701 solids):

| Agent-visible | What it is | Solids |
|---|---|---:|
| `EV-411` electronics enclosure | the vendor's 4-inch watertight enclosure with dome, end cap and flanges and everything inside it (tray, boards, the six speed controllers of the R4 model, camera and tilt servo, the depth sensor and the penetrators on its end cap) | 422 |
| `BV-311` battery enclosure | the 3-inch enclosure with the battery pack and the fittings of its end caps | 40 |
| `TD-76-CW`, `TD-76-CCW` ducted thruster | the T200 thruster with its clockwise and its counter-clockwise propeller; 13 of the 14 solids are congruent between the two, only the propeller differs | 14 |
| `LD-15` LED light | the Lumen light | 23 |
| `GR-62` single-axis gripper | the Newton gripper | 24 |
| `DV-4B` Doppler velocity log | the DVL A50 | 7 |
| `BW-19` ballast weight | the 200 g stainless ballast weight | 1 |

- The grouping rule: a module is what leaves the vehicle with it. A fastener that engages a made part or a frame plate
  is in no module (the six M3 x 16 screws that hold the clips on the cradles, the four M5 x 16 screws of the
  accessories and the gripper clamp's thumb screw among them): given with a module, it would hand over the hole
  pattern of the made part it engages. Port fittings go with the enclosure they sit on.
- Each type is one STEP file in a frame defined by the module's own geometry (the datasheet states it), written by the
  CAD kernel (AP214, millimetres) with a neutral header and generic product names. The files are pinned by sha256
  (`reference/purchased_pins.json`). No module frame equals the vehicle frame; the frames of the two port-side
  vertical thrusters are parallel to it, with another origin (kept: the thruster's frame is the one the sibling
  assembly task uses).
- **The one change to the vendor's solids: the maker's round emblem is taken off the six heat pads of `EV-411`**
  (solids 128 to 133, the thermal pads of the six speed controllers). Each pad carried the vendor's emblem (the same
  figure as on the scanning sonar's head) as a zero-height imprint on its outer face: two crescent pockets 0.0001 mm
  deep, 13.48 mm across, that a CAD program shows as face edges. The figure's 16 faces were removed and the pad's face
  closed into one plane by the CAD kernel's defeaturing, before the STEP file was written: each pad keeps its
  bounding box, gains 0.0082 mm³ and has 10 faces instead of 26, and the six stay congruent. No other solid of any
  module was touched (checked solid by solid: faces, volume and vertex set as before). The emblem has no height and a
  pad has one material, so no photograph showed it; the photographs were not rendered again.
- **"No other vendor face changed"** is judged on the surfaces (at most 1e-6 mm apart) and the vertex sets (1e-9 mm).
  Twelve solids of the STEP files are listed below. Eleven differ from the vendor's by more than 1e-9 in relative
  volume (adaptive integration): the kernel's STEP reader rebuilt edges, the surfaces and vertices are the vendor's.
  The grader and the agent read the same files, so the grading is not affected.

  | Module, solid (in the grader's reading order) | What it is | Relative volume difference | What changed |
  |---|---|---|---|
  | `EV-411` 44 | a lead (red) | 2.1e-6 | 6 degenerate edges dropped |
  | `EV-411` 45 | a lead (white) | 1.1e-6 | 2 degenerate edges dropped |
  | `EV-411` 141 | the camera module's case | 7.9e-9 | edge curves recomputed |
  | `EV-411` 390 to 393 | four M3 x 31 standoffs | 3.2e-8 | edge curves recomputed |
  | `EV-411` 324 | the flight-controller board | 5.1e-14 | nothing: the vendor's solid is invalid for the kernel as shipped, and stays so |
  | `TD-76-CW` 1 and 3 | nose cone, duct housing | 1.1e-8, 7.4e-8 | edge curves recomputed; a few faces changed surface type (same geometry within 4.6e-7 mm) |
  | `TD-76-CCW` 0 and 2 | nose cone, duct housing | 1.3e-8, 5.0e-8 | as above |

- **Four markings of third-party components stay as modelled** inside `EV-411`: on solid 163 (a sub-module of the tether
  interface board) the raised type designation `LX200V20`, 0.10 mm high; on solid 317 (the memory card) the raised
  `16GB` and an arrow, 0.01 mm high; on solid 351 (the single-board computer) the markings of four chips, sunk 0.010 mm
  into their top faces, characters 0.4 to 1.3 mm high (`BCM54213PE B1KMLG TA1913 P21 009-20 1W`, `MXL7704-P4 F4KT12`,
  `91D77 D9WHV 778K`, and the chip maker's logo `VLI` with `VL805-Q6`); on solid 134 (the 5 V regulator module) the
  inductor value `6R8` as a zero-height imprint. None is the name of this vehicle or of its maker, and none is on the
  platform's word list; no face was changed and no solid removed for them. The photo set's quality check confirms that
  none can be read in any photograph: seen through the clear tube, the characters are at most 9.1 px (the type
  designation) and 6.8 px (the card) high in the best case, and the three photographs where the first reaches 9 px show
  no readable character at full size; the chip markings are at most 4.5 px high (photo_094, an upper bound by the same
  ray cast over the 19 photographs that hold the enclosure's insides), and the imprint has no height to show.
- **Solids the CAD kernel calls invalid**: the flight-controller board as shipped, a second board of the electronics
  module and the thruster's duct housing once placed (at most poses that are not axis-aligned). They are closed and
  usable; the grader keeps them when they are unchanged (`README.md`, entry 4 of the differences).
- **The documents** (datasheets, drawings, `parts_list.md`) are written for this task from the measured geometry; none
  is a copy of a maker's document. A datasheet describes the module in its own frame only: envelope, outer diameters,
  its own holes, its mounting face. A module's mass is the sum of the reference vehicle's masses of its items ("as
  weighed for this unit" is the task's fiction); the electronics enclosure's also holds the wiring inside the tube and
  the two speed controllers and two terminal blocks of the Heavy kit, which have a mass and no geometry. The made parts'
  densities are the reference vehicle's materials: HDPE 0.960, aluminium 2.700, polycarbonate 1.200, buoyancy foam
  0.305, stainless steel 8.000 g/cm³.
- The thruster's mounting face is 49.00 mm from the propeller axis (the outer plane of the duct housing, flush with
  the four threaded inserts). The inserts' inner ends are at 43.16 mm; that number is in no agent-visible text.
- **Invented names**: the part numbers `EV-411`, `BV-311`, `TD-76-CW`, `TD-76-CCW`, `LD-15`, `GR-62`, `DV-4B`, `BW-19`
  and the modules' generic names are invented for the benchmark.

## 3. What the photographed vehicle is made of

The `mission` configuration has 643 manifest items; 639 have geometry and all of them stand in the photo scene at the
reference vehicle's poses:

- 600 items are the vendor's closed solids, 843 solids in all: the 31 made parts, the 701 solids of the purchased
  modules, 111 fasteners.
- 1 item is the vendor's surface model of the scanning sonar's head (open surfaces; the vendor publishes no solid).
- **38 solids were generated by this project**: the 8 L-shaped guard brackets (the vendor publishes no CAD of them)
  and 30 screws drawn to ISO dimensions (16 of the guards, 4 of the centre fairing caps, 4 of the DVL bracket, 6 of
  the sonar's frame mount).

The fasteners, the 38 generated solids and the sonar's head are in the photographs only: in no given file, not in the
reference, on neither side of the mass. The instruction puts them outside the modelling scope.

There are no cable runs: the vendor's model has none. The lead stubs the vendor's model has itself are where they
are. The electronics enclosure shows what the vendor's model holds (tray, boards, speed controllers with their lead
stubs, the depth sensor's four leads, a short piece of the tether lead) and no harness; nothing was added, covered or
frosted, and no camera position was chosen to hide the inside. The two speed controllers and two terminal blocks of
the Heavy kit have no geometry and are in no photograph. The instruction says that the photographed vehicle has no
cables fitted.

## 4. Poses

Of the 643 items, 439 stand at their pose in the vendor's R4 assembly; 204 were placed after the vendor's
installation guides (123 on a hole pattern, 36 inferred, 22 from a guide's text, 14 read from a figure, 8 by the trim
solution, 1 by the trim setting).

Of the 51 delivered instances (31 made, 20 purchased), 27 stand at the vendor's assembly pose, 17 on vendor hole
positions (within 0.5 mm: the gripper clamp halves, the sonar's frame mount, the guard plates, the four vertical
thrusters, the lights, the ballast weights), and **7 are known to 1 to 5 mm only**: the two centre fairing caps (1 mm,
inferred), the two round foam blocks (5 mm, inferred), the DVL bracket and the DVL (4 mm, read from a figure), the
gripper (2 mm: its extension through the clamp and its roll are a chosen state). The guard plates' height follows
from the rule the generated brackets were drawn by (0.5 mm). The battery's place along its tube is the trim setting,
and which four ballast holes are used is the trim solution.

The bottom plate P01 is the vendor's plate as published: it has not the two 5.5 mm holes that the DVL bracket's
installation guide has the user drill for the bracket's two M5 screws. The photographs therefore show the two screw
heads under a plate whose flat-lay shows no hole at that place, and the bracket's place can be measured in the
photographs but not found hole on hole (the bracket's two holes are 5.0 and 22.9 mm from the nearest hole axes of the
plate). The truth was not changed for it: the plate is the part the whole vehicle is aligned on, and a changed plate
would need new photographs, a new reference and new quality-control documents. No agent-visible text mentions it. The
four ballast weights stand turned by 1.014 degrees about their screws: the reference vehicle's pose, taken from the
frame of the vendor's mesh.

All 51 are graded alike, with the same bands (owner's decision): the photographs are renders of exactly this truth,
so what the agent sees and what it is graded against are the same places.

## 5. The vendor CAD's own overlaps

The vendor's CAD is kept with its own overlaps; the grader subtracts, pair by pair, what the reference has itself.
The 22 pairs between delivered items (exact volumes; the three marked pairs by the grader's meshes, where the exact
boolean closed its volume balance on a wrong value):

| Pair | mm³ | What it is |
|---|---:|---|
| battery enclosure, tube clamp half P05 | 484.5 | the 3-inch clamp against the tube: about 1.2 mm of radial interference |
| foam block P11, front centre plate P19 | 811.3 | a foam block pressed onto the plate |
| corner fairing P16, centre fairing cap P24 | 494.1 | the cap's wall inside the corner fairing's (about 0.3 mm) |
| P13 and P24, P14 and P23, P15 and P23 (mesh values) | 490.3, 488.4, 488.6 | as above |
| DVL, DVL bracket P25 | 113.4 | the back plate in the bracket |
| corner fairing P16, front centre plate P19 | 60.5 | |
| rear centre plate P04, a horizontal thruster | 45.1 | |
| side plates P17 and P18, guard plates P31 and P30 | 44.3 each | |
| rear centre plate P04, tube cradle P26 | 22.4 | |
| the four corner foam blocks, their corner fairings | 20.3 each | press fits |
| four thrusters' threaded inserts, the side plates | 14.2 each | |
| two lights, the side plates | 4.8 each | |

Not an overlap: the battery enclosure's two cord handles, two set screws and two locking cords reach 1.4 to 2.9 mm
below the upper face of the bottom plate, where the plate has two openings; the common volume is 0.

## 6. The photographs

- The photographs are renders. The room, the bench, the materials and the light recipe are those of the sibling assembly
  task's demonstration (`rov_final_assembly_and_trim_single_arm`, in the look the owner approved), an OpenUSD scene
  rendered with OVRTX path tracing. Three things were changed for this task: the vehicle is taken apart (every manifest
  item with geometry is its own object, placed per photograph; the made parts tessellated from the truth B-reps, the
  rest from the reference vehicle's high-resolution meshes); photo props were added (light grey cutting mats with a 10
  mm grid, the checker card, steel rules, paper tags, two wooden blocks, a white reflector and flags that stay out of
  frame); the cameras are hand-held views (perspective, no pose recorded, no geometric distortion added). The assembly
  cell's fixtures and its robot arm are not loaded. The room's art assets are named in that task's `NOTICE.md`; none is
  shipped with this task, only the rendered photographs are.
- The finish is phone-like (white balance and gain per photo, sensor noise, a light sharpening, JPEG, no metadata).
- The set: 156 photographs (64 teardown, 17 extra views, 35 close-ups, 8 of the complete vehicle, 32 of the stages),
  2016 x 1512 or 2400 x 1800 pixels, rendered in one batch with one renderer setup (OVRTX 0.6.0, eight steps of 32
  samples, super-sampling 2). Ten of them (photo_071, 073, 078, 103, 106, 111, 116, 143, 144, 156) were rendered with
  the renderer's texture streaming switched off, because with it the renderer crashed on those views; the setting
  changes no visible detail (measured on a test view: 0.6 % mean difference of the raw image). Two exposure rules were
  applied after the render, as a change of exposure only: a cap so that no segment of a steel rule is brighter than
  240 (photo_016) and a floor on the mean brightness of the low hall views (photo_084, 093, 104, 108). photo_022 was
  rendered again with the tag of P11 moved clear of its part.
- **Material changes made for the photographs only** (made when this task's first stills were judged; the
  demonstration's own layers are untouched): the bench top's material is built from a photographic wood scan of the
  project's licensed art library (the Wood material of ArchVizPRO Vol. 6, an asset the owner supplied), colour-matched
  to the demonstration's oiled beech top, because the procedural top showed its pattern at the photographs' scale; the
  black HDPE's machining-mark normal map is weakened and the plates' edge shading sharpened (the true plates have
  straight cut edges); each HDPE part carries a wear texture of its own (fine scratches and fine dust, denser near edges
  and holes; the area-averaged albedo changes by less than 1 %); the fairings' blue is slightly less saturated and
  rougher; the foam blocks are cream-coloured, as the vehicle's documentation describes them, with a fine pore texture;
  the cutting mat is lighter and carries light marks of use. No geometry of a made part or of a purchased module was
  changed.
- **Photo props as photographed**: the steel rules' figures are centred on their centimetre lines, and the rules of
  the flat-lays and close-ups have a brushed body; the top-down views of the complete vehicle carry a 300 mm rule, the
  flat-lays of the small parts a 150 mm rule; the checker card is left out where it would not fit (some flat-lays and
  stage photographs, the turned-over vehicle). The edge views (photo_079, 081) carry a 300 mm rule flat on the mat. The
  turned-over vehicle (photo_089) rests on two stacks of two wooden blocks; its only scale reference is a 150 mm rule
  lying flat on the turned-up bottom plate P01 (no checker card); photo_097 (2400 x 1800) is a third view of the
  ballast weights. The feature close-ups (photo_125, 143, 144, 152 to 156) are low oblique views along the axes of
  holes that no other photograph shows well.
- **What the captions say about the props, corrected after a check of the finished photographs** (the photographs are
  unchanged; the captions are generated from the photo plan and from the visibility maps of the finished photographs).
  In the 28 upright views with a leaning checker card, the card stands beyond an end of the vehicle. It is at the depth
  of the near side plate (0.95 to 1.05 of that plate's depth along the optical axis) in 6 of them (photo_082, 083, 092,
  102, 112, 118) and nearer or farther in the other 22 (0.40 to 0.91 or 1.44 to 1.80). The first captions called every
  such card level with the near face of the vehicle; the captions now say "at about the same distance from the camera as
  the near face of the vehicle" for the 6 only and give no depth for the others. In photo_104 and photo_108 the card of
  the plan lies under the turned-over vehicle and is not seen (2 and 0 visible pixels): their captions no longer name
  it.
- **How the parts lie in the teardown photographs.** A part's first lay is the stable rest nearest to the way it
  sits on the vehicle: 18 of the 31 parts lie exactly the way up they are installed, 4 within 33 degrees of it
  (P22, P23, P24, P25), and the 9 that stand on edge on the vehicle (P02, P07, P08, P17, P18, P26 to P29) lie on a
  face. The instruction says so, because an attentive reader could infer it from the parts whose way up is obvious.
  The close-ups use the same two lays. photo_073, meant as the end view of tube clamp half P05, looks across the axis
  of its half-round seat (92 degrees off it), so the seat's outline is not in that photograph.
- **The maker's round emblem on the top of the scanning sonar's head is flattened in the photo scene** (the head is in
  the photographs only; no cover or cap was added, because the vehicle has none): a thin skin that exists only in the
  photo scene follows the head's top cap, with the cap's own material and texture coordinates, and covers the emblem;
  the head's own mesh is unchanged. The photo set's quality check requires that no maker's mark can be recognised in
  any photograph.
- **The disassembly order is the photographed order, not a verified removal sequence.** Each stage is photographed
  as the state after the removal. On the CAD, 5 of the 10 stages pass a straight-line removal sweep (with at most
  0.4 mm of easing); the others are blocked, for these reasons:

  | Stage | Removed | Blocked by | Why |
  |---|---|---|---|
  | 2 | the sonar's head and frame mount P22, the six fairings, the six foam blocks | P22 against side plate P17; on the starboard side the centre fairing cap P23 against the corner fairings P14 and P15 and the foam blocks P09 and P12 | P22 stands on a hole pattern (0.5 mm); the cap's pose is inferred (1 mm); the vendor's own wall overlaps and press fits |
  | 3 | the two guard plates, the four vertical thrusters | the guards against the thrusters | the guards' height follows from the generated brackets (0.5 mm), the thrusters stand on hole patterns; the thrusters' photo meshes are merged and not manifold, so the sweep may judge too strictly |
  | 5 | the DVL and its bracket P25 | the bracket against side plate P18 and the DVL's connector | both poses are read from a figure (4 mm) |
  | 7 | the battery enclosure, tube clamp half P06 | the enclosure against the other clamp half P05 and the bottom plate | the vendor's own 1.2 mm radial interference of the clamp |
  | 8 | the four horizontal thrusters | three of them against a centre plate each | merged thruster meshes (see stage 3) |

  Stages 1 (the lights), 4 (the electronics enclosure and its clips), 6 (the ballast weights), 9 (the gripper with its
  clamp halves) and 10 (the centre plates, the tube cradles, the last clamp half) pass. Three rules hold in any case:
  the four centre plates leave in one stage, the six fairings in one stage, the six foam blocks in one stage.
- The paper tags and the captions do not tell the two centre-plate families apart: their tags are P03, P04 (rear) and
  P19, P20 (front) by the hash, the four leave in one stage, and no caption or file name holds a function or position
  word for a made part. Which shape is at which end can only be read from the stage photographs.

- **What the photographs cannot give is not scored.** The black-part rule of the photo set was measured on the
  finished photographs; 13 reference features of 8 instances that no photograph shows readably (the half-round seat
  of P06, three holes of P01, two each of P03, P20 and P21, the half-open walls of P02, P17 and P18) were taken out of
  the feature criterion before the freeze. Their parts' geometry is the vendor's, unchanged, and is scored
  in the shape like the rest; the list with each feature's reason is `reference.json` `features_withdrawn`, the
  reasons in short are in `README.md` ("Known limits").

## 7. Public availability

The vehicle is a public product. The vendor publishes the STEP files of section 1; the sibling assembly task's public
assets hold a mesh of the whole vehicle (157 thousand triangles) and the guard plate's mesh at its original
resolution. The agent has no network; `public_blocklist.json` stops an unchanged copy of such a file in the
submission folder. A copy in another format or a re-meshed one is not stopped. A model may remember the vehicle's
general layout; it cannot remember the sub-millimetre outlines of its plates.

## 8. Third-party software

- FreeCAD 1.1.4 (LGPL-2.0-or-later, https://github.com/FreeCAD/FreeCAD) is installed unmodified in the task image
  from its release AppImage, pinned by sha256 in `environment/Dockerfile`.
- The FreeCAD MCP server neka-nat/freecad-mcp at commit `78bf0230a03324b56c9a0b7f55150fb5be2a0d42` (MIT,
  https://github.com/neka-nat/freecad-mcp) is vendored unmodified in `environment/vendor/freecad-mcp/` with its
  `LICENSE`.
- The grader is the benchmark's own, ported from the UAV task `uav_cad_design_from_photos` (`README.md`); its Python
  packages are those of the benchmark's shared base image (`environment/requirements.txt`).

## 9. Where the files are

The payload (photographs, `purchased/`, `parts_list.md`) is served by the benchmark's asset dataset under
`tasks/underwater/rov_cad_design_from_photos/payload/`, at the commit pinned in `environment/assets.sha256`. The
hidden truth is in this folder and in `../solution/reference/` and is never part of the payload.
