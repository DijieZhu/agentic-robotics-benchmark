# NOTICE (grader-side; never staged into the agent's home)

The grippers are the MuJoCo Menagerie `robotiq_2f85` model (BSD-2-Clause, copyright 2013 ROS-Industrial; its LICENSE
and a SOURCE.txt with the commit ship inside `parcelcell/assets/gripper_2f85/`), used unmodified; the cell attaches it
at both wrist flanges at build time, locks its two coupler joints (each finger becomes one rigid linkage on its driver),
sets the finger pad boxes' contact values and puts a palm camera into its base (`parcelcell/robot.py`,
`parcelcell/cell.py`). The agent gets the model's geometry and kinematics in `~/given/robot/gripper_2f85/` (with its
LICENSE) and the combined model as the cell assembles it, both with the station's nominal dynamic data instead of the
cell's own values (`~/given/robot/README.md`).

The parcel kit (`parcelcell/parcelkit/`) is original to this task; its fonts (`parcelkit/fonts/`, Roboto Condensed) are
under the Apache License 2.0 (`parcelkit/fonts/LICENSE-fonts.txt`). The carrier on the waybills is fictional.

The dual-arm platform's model (`parcelcell/assets/robot/`: the MJCF, and its meshes, which come with the task's
fetched assets, `environment/assets.sha256`, and are copied there at grading time; generic names) is licensed to the
benchmark for redistribution.

The agent cameras' render bundle (`render/parcelcell_render_v3.tar`, one tar fetched with the task's assets at the
dataset commit pinned in `environment/assets.sha256`; the image unpacks it to `/opt/task-assets/render`, root-owned
and readable besides root only by the commissioning service's account, so that neither the agent's account nor the
program under test can read it, and checks it file by file against its manifest) is what the cameras' renderer (the
filament backend, `parcelcell/look_filament.py`) draws besides the parcels: every static mesh an agent camera sees,
simplified beyond 6 m, with textures of at most 512 px, and the robot's drawn parts. Its hall, the hall's dressing,
the line around the station and the cell's fence are meshes and textures of two Unity Asset Store packs licensed to
the benchmark's owner for distribution with the benchmark, recomposed into a new hall for this task: "Factory Interior
Factory Props Vol 1 - BUNDLE" (CyberneticWalrus: the wall, roof and column modules, rafters, ducts and cable trays,
the line's belt conveyors, pallets, cartons and wrapped loads, a forklift, fire and first-aid equipment and safety
signs) and "Warehouse Props Vol 1" 1.0 (CyberneticWalrus: the racks, cartons and crates, the mezzanine and its office
booth, pallet trucks, a floor scale, cable drums, and the cell's wire-mesh fence). Parts of them were repainted (the
conveyor frames graphite, the rack uprights and the columns grey). Painted out: the forklift's model numbers, the
pallet trucks' embossed maker name, the floor scale's maker print and the extinguisher's maker label. Left out: the
packs' props printed with brand names (sacks, a stretch wrapper, tool chests). A shipping label, a rating plate,
pallet stamps and the packs' maker-like name printed on the forklift and the pallet trucks remain in pack textures and
cannot be read at the bundle's sizes. Its free surface sets are CC0: Poly Haven's `smooth_concrete_floor` and
ambientCG's `Concrete034` in the floor and the yard apron, Poly Haven's `painted_metal_shutter` on the dock doors'
shutters and ambientCG's `Metal009` on brushed-steel parts. Made for this task (their textures too, apart from the CC0
sets above): the station's drawn parts (the conveyor section with its frame, drive, rails and photo-eye, the scan
gantry, the operator panel, the robot's riser and floor plate, the light curtain, the floor ducts, the task-light
frame with its lamps and the infeed camera's mount), and in the hall the dock doors, the floor tiles with their
markings and wear, the luminaires, the skylights, the conveyor stands and transitions, the roll cages and the
background sorter; the dock wall is the first pack's wall modules above a lower part made here in that pack's wall
maps. The image light (`ibl.ktx`) is a cube probe of this scene path-traced with NVIDIA OVRTX at the work area, with
the robot and the parcels hidden, prefiltered with Filament's cmgen. The depth files (`depth_near.npz`,
`depth_near.json`) hold the 50 static meshes the cameras see within their depth range, apart from the belt top and the
hall floor, which the cell has as geometry; the cell adds these meshes (among them the conveyor's frame members, rails
and photo-eye, the robot's riser, the light curtain, the fence's near panels and posts, the gantry post) to its depth
pass. The robot's drawn parts (`robot.npz`) are the dual-arm platform's and the gripper's meshes above, split into
finish classes, with parting lines, screw heads, the head camera's housing and the cameras' bezels and windows added
for this task.

Everything else is original to this task: the simulated station and its conveyor, the parcels and their labels, the
shift generator, the cameras and their sensor models, the tool-pose controller, the safety watch, the score, the wire
server and client (`parcelcell/`), the commissioning service (`environment/commission_service/`), the reference program
(`reference/`) and the documents. `robotics_grader/` is the repository's shared kit (`shared/robotics_grader/`), byte
for byte; `cellkit/` is adapted from the benchmark's shared verify kit (its wire gained the per-shift budget in
proportion to the shift's duration, a separate start window, a request-line guard and this task's end reasons). No
third-party dataset is used.

## Accepted facts (maintainer's record; not in the agent's documents)

These were found by the independent check and accepted as they are:

* The wall time a `step` takes grows with the number of parcels on the belt (measured on this cell through the server's
  call path, on the build host under heavy load: a median of about 78 ms for `step(n=50)` with none, 88 ms with one and
  92 ms with two on three rated shifts; 67 ms with none and 84 ms with four on a constructed shift, not monotone in
  between), and `program_time_left` against the program's own clock shows the cell's own time exactly: a program could
  estimate how many parcels are on the belt from timing. The cameras show them anyway; padding replies to a constant
  time is not usual practice.
* No line signal is on the wire: no belt speed and no photo-eye. A program finds the shift's belt speed and the
  arrivals from its cameras (two timed frames of the infeed camera give the speed; polling it catches the arrivals).
  What else the wire tells, checked field by field: the robot's own state (joints, tools, grippers, stops), the
  budgets' counters, the frames, and the following.
* The shift's duration (in `describe()`) is a constant: 600 s for every graded and every full development shift, 300 s
  for a bench shift, whatever the belt speed and the arrivals, so it tells nothing about the last arrival or the number
  of parcels (on the line setting a full shift's list length follows from the belt speed by a public rule:
  `describe()["shift"]["list"]`, the manual's section 7). On the rated setting (the oracle's set, `hidden/`) the
  generator keeps only draws whose last parcel has crossed the scan line and left the belt by then (last arrival + the
  belt's travel from the entry to the scan line + 40 s at most the duration) and draws again otherwise: on the slow
  belts about 16 % of draws are drawn again, which lowers the mean parcel count there a little (17.7 against 18.0;
  shifts of 21 parcels about 11 % against 14.3 %) and raises the mean belt speed there a little (about 0.096 against
  0.090 m/s); on the medium and fast belts none of 2000 draws each was drawn again. On the line setting the generator's
  first draw is taken and nothing is redrawn.
* The tool-pose level's solve and the measured tool poses are the cell's own work: their wall time is not charged to
  the program, so they add nothing to the timing channel above.
* `step` with `n` above the per-call cap runs the cap (50 ticks) and says so in `ticks` (documented), instead of being
  refused.
* A parcel that crossed the scan line readable and is lost afterwards keeps its reading and still counts as lost (net 0
  for a W parcel, -1 for a U parcel); the manual says so.
* "Rests on the belt" at the scan line is judged over a short window, not from one tick's contact flag: the parcel
  touched the belt surface in at least one tick of the last 0.5 s (the crossing tick included) and its lowest point is
  at most 10 mm above the belt surface. On an earlier version of the cell a soft mailer riding untouched rocked on its
  pad and lost its belt contact for up to 8 ticks (0.16 s) at a time, and with a one-tick test 3 of 651 label-up parcels
  of a run with no program at all (three untouched mailers) were judged unreadable, which cost credit to every program,
  one that does nothing included. With the window, every label-up parcel replayed with the robot parked is readable:
  1324 of 1324 over 200 of the present generator's shifts (no selection) and the label-up single-parcel shifts
  (`reference/README.txt`), and 2213 of 2213 over 200 line shifts; a parcel in the air, tumbling, resting on the robot
  or on another parcel at the line is still unreadable.
* On the rated setting (the oracle's set, `hidden/`) a parcel's class (W / U / X, from its kind and its label's
  position against the station's duty, `score.duty`) is fixed at its arrival on the belt; nothing the robot does
  changes it. The duty is public (describe): a program may leave every X parcel alone, which is the intended
  behaviour. On the line setting the class (W or U) is likewise fixed as the parcel is set down; every parcel is in
  the duty (`line.score.duty`; the manual's section 8, "A shift and its score") and there is no X class.
* On the rated setting (the oracle's set, `hidden/`) the arrival gap is one rule for every pair of parcels, drawn as
  belt distance: half the two footprint diagonals + the entry's clearance and its margin (0.16 m) + the belt's travel
  in `shifts.arrival_gap_min.t_s` + `d_m` (the station's rated job) + a capped exponential extra, so the arrival times
  say nothing about a parcel's class (they depend on the footprints and the belt speed, which the cameras show
  anyway). The values are the configuration's (`t_s` 10 s, `d_m` 0.20 m); with them the reference's evaluation on this
  cell had every in-duty parcel readable and nothing lost or damaged in 106 of 120 of the generator's own shifts with
  at least 6 in-duty parcels (`reference/README.txt`); on slow belts the distance terms make gaps long (up to
  about 74 s at 0.05 m/s); every shift still lasts its fixed duration (above). On the line setting there is no arrival
  gap: the parcels are fed on settle (the next one is set down at the first tick after the one before it is settled,
  once the set-down place is clear).
* The conveyor's middle leg's near post and the robot's pedestal box overlap by 5 mm in x (both static: they never
  make a contact); the drawn station keeps them apart (the drawn pedestal is 10 mm inside its box, 5 mm clear of the
  post).
* Frames of the filament backend are not promised to be identical across processes: with its dithering off (or the
  renderer failed closed), a few pixels at object edges can still differ between two processes that draw the same
  state (5 to 7 pixels of a frame in the runs that showed it; the cause was not isolated). The physics never depends
  on a frame, so the episode record of a program that takes no frame is identical in every process; a re-grade of a
  program that acts on its camera images is a sample, not a replay. The classic backend's frames are byte-identical.
