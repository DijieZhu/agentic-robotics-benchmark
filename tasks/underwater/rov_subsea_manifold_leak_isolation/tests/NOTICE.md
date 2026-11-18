# rov_subsea_manifold_leak_isolation · sources and attributions (grading side only)

This file is the one place where the task names the real sources behind its fictional surface. Nothing here is
handed to the agent; the agent-visible documents use generic names only.

## The vehicle

The reference vehicle is Blue Robotics' BlueROV2 (R4) in its Heavy configuration (eight thrusters) with the vendor's
accessories (single-jaw gripper, scanning imaging sonar, Water Linked DVL, four lights, depth sensor), rebuilt from the
vendors' public CAD models and documentation into the underwater tasks' shared reference digest (masses, centres,
displacement, thruster geometry and curves, sensor mounts and specifications).

## Hydrodynamic coefficients (hidden-world draw)

The vehicle's water-interaction coefficients are one draw, made once for the task world, between two published sets for
this vehicle class, with a multiplicative jitter:

- Wu (2018), "6-DoF Modelling and Control of a Remotely Operated Vehicle", MSc thesis, Flinders University.
- von Benzon et al. (2022), J. Mar. Sci. Eng. 10(12), 1898 (Table A1).

## Software

- MuJoCo 3.14.0 (Apache-2.0): physics and the classic renderer (OSMesa / llvmpipe) of the vehicle cell.
- numpy, scipy, Pillow, OpenCV (headless), PyYAML: the programs' and the cell's numerics and image handling.
- The process kit (`cellkit/`) and the grading kit (`robotics_grader/`) are the benchmark's own.

## Static scenery of the sites (amendment 02 of the hidden selection, 2026-11-17)

Every scored station and the sister practice site carry static scenery as world geometry, present at the same place
in the physics (collision), the vehicle's camera and the sonar: scattered boulders (eight closed rock meshes, at most
0.55 m tall), the production line with its jumper spool and pipe supports, a flexible test line (outer diameter
0.15 m, polymer sheath) at the end of the test header, and concrete mattresses. The site generator lays them out from
each station's own seed, outside geometric keep-out zones (the mud mats plus 1 m, the panel's working band, the laydown
area, the dropped object's surroundings, 4 m around the cage centre and 7 m in front of its entrance, the cage-to-panel
passage, the reference program's transit corridors), every piece within 45 m of the structure centre; the work order's
chart gives each line's as-laid route. The hidden set was selected again under the same labels with the scenery in the
world (`tests/hidden/selection.json`, amendment 02): the same six cells and candidate numbers, six new station
packages. The rock meshes and the three tiling textures of the scenery are derived from the maintainers' licensed 3D
art library (decimated, closed, albedo baked).

## Simulation limits (recorded, not hidden from maintainers)

- The camera is rendered with the classic rasteriser (albedo and depth) plus a depth-based water model (the lamps,
  per-channel attenuation, back-scatter, particles). A later look pass changes only the albedo and structure-attached
  detail of this path; a physically based renderer is not used for the vehicle's camera (it would light the lamps twice
  and lose bitwise determinism).
- Contact and friction of the valve hinges, the gripper rule and the hot stab follow grader-side world rules
  (fixed 2 ms physics step); they are calibrated to the panel standard's numbers, not to a measured panel.

## Known deviations (recorded at release)

- Water density: the reference vehicle's data use 1024.8 kg/m3, while the cell's water column is described as
  salinity 35 at about 10 degC; the two differ by about 0.2 % (some 30 g of buoyancy), below datasheet accuracy. The
  documents print each figure where it is used.
- The scanning sonar's head slews to a new sector without taking time (simulation simplification).
- Valve turning by rolling the vehicle: with the current near the working limit and a gust during the turn, the flow
  load on the vehicle's broad side can exceed the grip and the handle slips (14 of 400 turns, 3.5 %, in the
  development test matrix at its high-drag, 0.30 m/s corner). This is the world's behaviour, not a numerical artefact; the design target
  "no slip" was not reached and is recorded as such. The gust factor range (0.10-0.20) is kept.
- The hot-stab receptacle's wall pieces are corrected in place by the cell (a handedness error of the panel
  generator's wall staves); the generator itself is fixed when the sites are next regenerated.
- Valve-handle grasping by a frozen non-privileged pilot program (a design-gate check; non-privileged pilots are not
  part of the release acceptance): the pilot frozen at the phase-1 gate grasped the target handle within three
  approaches in 29 and 28 of 30 tries at the two 0.30 m/s corners of the gate's world, and in 12-23 of 30 (40-77 %;
  19-23 of 30, 63-77 %, with the dropped object in the laydown area) in the finished world, below the gate's target of
  26 of 30, with no wrong grasp in 300 runs. The finished world is harder than the gate's (the dropped object can take
  the DVL's bottom lock above it, the neighbouring valve columns are real, the DVL lock in front of the panel is
  poorer). Feasibility on the finished world rests on the privileged reference program (100 points on every station).
- The verifier's time limit (`task.toml`, 9600 s) comes from unit costs measured on the phase-1 gate's station world.
  Measured again on the six hidden stations in the task image (2026-11-17, three slots), the cell costs up to 0.94 s of
  wall clock per simulated second while the dropped object on the mud mat in front of the panel is awake (the vehicle,
  the marker or the stab within reach of it: the stand's contacts with the mat and its water loads are then computed),
  against 0.29 s in the derivation and about 0.25-0.34 s at the other measured positions (cage, transit); the static
  scenery changes these costs by 1-3 %. The reference program's stations take 290-650 s of wall clock (1048 s for the
  six). A program that kept the vehicle next to the dropped object for most of its 2400 s of simulated time and also
  spent most of its frame, ping and own-time budgets would need about 3900 s for that station (3916 s with every budget
  used up at the dearest position). That is within the station's watchdog, which is one wave's share of the limit,
  (9600 - 250) / 2 = 4675 s, so such a station ends by its own budgets and is scored. Two waves of such stations (about
  7830 s) still fit in the 9050 s that the fixed steps leave before the deadline; what does not fit is the
  single-station re-run on top of them. It starts only while the average share, (9600 - 250) / 3 rounded up to
  3117 s, remains, so if both waves are at this worst case and a cell failure calls for a re-run, the re-run is not
  started: that station is scored as it stands (a physics fault or an internal error of the cell) or the verify ends
  as a task error (any other cell failure).
- Line ends: the work order's chart gives each scenery line's as-laid route to its last node, at the edge of the
  surveyed area (44 m from the structure centre), and the documents say the line continues beyond it; the simulated
  seabed ends the line at that node, inside the 60 m bound, where a vehicle flying out that far can see it end.
