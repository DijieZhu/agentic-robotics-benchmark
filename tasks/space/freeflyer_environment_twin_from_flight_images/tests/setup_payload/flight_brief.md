# Flight brief: the module as found

## Your role and what the twin is for

You rehearse flights of the cabin free-flyer before they are flown. The rehearsal runs in a simulator against a
collision twin of the module, and the robot's own hazard check decides there, as it does in flight, which straight
moves are clear. A twin that holds only the bare module lets the rehearsal fly through stowage; a twin with phantom
geometry stops moves that would have been clear. So the twin must hold the module as the robot found it during the
recorded session, at the right places and sizes, in metres.

It is a rehearsal twin: at a moment of the flight the rehearsal asks it, move by move, whether a straight move is
clear and where it stops. One wrong obstacle verdict there (a move found clear that is blocked, a move blocked that is
clear, or a stop far from the real one) makes that moment unusable for the rehearsal, however good the twin is
elsewhere. Walls and racks count as much as stowage: a move that runs into a wall the twin places wrongly is as wrong
as one that runs into missing stowage.

The module is prepared for each session. The twin describes the module during this flight.

## The flight

There is one recorded flight, `flight_a`: the robot flew through the module, recording its forward wide camera (1068
frames in `flights/flight_a/frames/`), its inertial unit and its on-board pose estimate. Its forward depth camera
recorded too; that record is withheld and is what your twin is scored against. The grader scores 24 moments of the
flight and does not say which, and no given file names them. So you declare the wide camera's pose at every given
frame of the flight (1068 poses). The flight scores H x D, with H the hazard term and D the depth term that
`twin_contract.md` defines; D is the product of Dc and Ds, the content's and the structure's terms, and one wrong
obstacle verdict at a scored moment loses that moment.

## The robot and its sensors

- The robot is a cube of about 0.32 m edge (half edge 0.16 m about its body origin). Another unit of the same kind has
  the same size and shape.
- The forward wide camera: grey frames of 1280 x 960 pixels, one byte per pixel, through a wide-angle lens; its model
  and mounting are in `sensors/camera_sheet.json`. The mounting is the maker's engineering position with an idealised
  orientation, as the maker's file says.
- The forward depth camera sits beside the wide camera, about 0.0767 m from it (`sensors/wide_to_depth.json`, the
  transform the grader uses). Its image is 224 x 171 pixels; along the central row its rays span about -30.1 to 28.9
  degrees, along the central column about -22.7 to 21.9 degrees. `sensors/depth_rays.npy` holds its ray per pixel.
- The inertial unit: its mounting in the body frame is the maker's (`sensors/camera_sheet.json`); `imu.csv` holds its
  samples in the unit's own axes, as recorded.
- The on-board estimate (`onboard_estimate.csv`) is the robot's own estimate of its body position and attitude in its
  own map frame, written as recorded, with the estimator's own health fields beside each sample. It is an estimate.

## Time

Every `t_s` in the flight's files is a recorded stamp, in seconds after the recorded stamp of the first wide frame.
The maintainers measured, against the depth camera's stamps of the same records, that a wide frame's exposure lies
0.067 s before its recorded stamp, with a spread of about 0.006 s. Wherever the grader needs the on-board estimate at
a frame, it reads the estimate at the frame's recorded stamp minus 0.067 s.

## The module model

`module_shell/shell.obj` is the maker's mesh of the bare module, its geometry unchanged, in metres, in the robot's map
frame (the frame of the on-board estimate). It holds no stowage. How well it fits the real module, measured by the
maintainers on this flight: after the model is placed on the measured walls by one best rigid fit, the measured
surfaces that lie less than 0.1 m in front of or behind the placed model (the part of what the scoring treats as
structure that lies that close to it) lie typically within 5 cm of it, and nine in ten within 9 cm (the median and the
nine-in-ten distance, each rounded up to a whole centimetre).

## The zone record

`module_shell/zone_record.json` is the maker's dated zone record of the module, as given: 4 keep-out boxes from a file
dated 2020-09-04, and the 2 keep-in boxes of its keep-in file that contain the flight's estimated positions, from a
file dated 1530672755 (Unix seconds, as written in the maker's file: 2018-07-04 in UTC). Its boxes are in the frame of
the on-board estimate.

## What counts as the environment

Everything solid in the module during the flight that the depth camera measured is the environment. The scoring splits
it with the module model placed by the maintainers' one best rigid fit, measuring along each depth pixel's ray, with
b(r) = max(0.06 m, 0.04 r) the band at the measured range r:

- **as-found content**: a measured surface at least 3 b(r) + 0.03 m in front of the placed module model (stowage,
  bags, equipment, anything fixed in place during the flight that stands off the bare module);
- **structure**: a measured surface less than 0.1 m in front of the placed module model, or behind it (the module's
  own walls, racks and fittings as the model has them);
- pixels between the two lines are not scored.

## What is not scored

- anything that moved during the flight;
- another free-flyer, whether it moves or not. Geometry you place inside its own outline neither earns nor costs;
  geometry larger than its outline is judged like any other extra geometry;
- pixels where the depth camera recorded no return, and measured ranges beyond 5 m or closer than 0.2 m;
- pixels whose ray does not meet the placed module model;
- pixels whose measured range or class the maintainers could not establish firmly: range edges, surfaces seen at a
  grazing angle, pixels whose value depends on the exact placement of the depth camera, and surfaces, content and
  structure alike, that the given wide frames do not show from viewpoints far enough apart. A surface is scored only
  where a twin inside the contract's limits can reproduce the depth record, so fine detail that no such twin
  reproduces neither earns nor costs. A pixel or a move is scored only where the maintainers' own twin of this flight,
  built from the withheld record, reproduces the record under the grader. A move is scored only where the depth record
  is unambiguous about what the move meets. Which moments, which pixels and which moves are scored is not published;
- texture and appearance, mass and friction, and any registration to a frame other than your twin's own.

## The hazard check

The robot's hazard check is the maker's documented one (occupancy cells around the depth camera's points, inflated by
the robot's radius, and straight moves checked against them; `twin_contract.md` defines it exactly and
`sensors/probe_card.json` holds its numbers). In the flight software this mapper is switched off by default; the
rehearsal runs it on the twin's depth frames.
