# Site brief: the goods delivery bay

## The customer and the job

A facility-services team is preparing a small four-legged robot for night rounds in the goods delivery bay of a
large building. On a round the robot walks the lanes of the bay, looks under the trailers, checks what stands where
and whether the cage gates are shut. The rounds are never tried first on the real robot. They are rehearsed in
simulation, on a digital twin of the bay as it was found, and only rounds that work in the twin are sent to the
robot.

The bay cannot be surveyed again before the first night. A surveyor walked one loop through the hall and took
44 colour photos from a tripod (`photos/`; what is known about the camera is in `photo_info.json`).
The photos are all there is: no laser scan, no floor plan, no measurements, no second visit. You build the twin
from them.

## What the twin is used for, and what that requires

- **The robot walks through it.** The robot is small: about 0.34 m wide, 0.32 m tall
  when it walks and 0.25 m when it crouches (`robot/robot_card.md`). Whether it gets through
  somewhere is decided by the lowest part of everything in the bay, up to the robot's own height: the gaps between
  things, the undersides of trailer beds, decks and overhanging loads, the walls. A wall that is missing in the
  twin is a way through that does not exist on site (a wire-mesh wall is a wall); a trailer modelled as one solid
  block hides a way through that does exist. The team walks the robot through the twin with the controller in
  `robot/`, and the outcome must be the one the robot would have on site.
- **Loose things are moved between rounds.** Goods come and go, and the team rearranges the twin to rehearse the
  next night. Every loose unit in the bay is therefore its own rigid body that can be moved freely, with a class
  label. Two units fused into one body would be moved together; a unit left out is a thing the robot walks into.
- **The twin holds the surveyed state.** When the simulation starts, every unit stays where the photos show it:
  what stands keeps standing, what lies on something keeps lying on it, nothing sinks into the floor, hangs in the
  air or slides away.
- **What rests on what.** Lifting a unit must carry along exactly what is loaded on it, and nothing else.
- **The cage gates slide.** The robot's rounds include the mesh cages. Each cage gate named below is one body on
  one straight slide along its rail, with a limited travel that suits a gate of its size.
- **Where things are, in metres.** The robot has a fixed size, so the twin must be at true scale; things of
  standard size in the photos tell you the scale. The twin is tied to the site through the photos: for every photo
  you used, the twin contains that photo's camera, where you estimate it stood (`twin_contract.md`).

## Units, environment, gates

- **Units**: every loose thing that a person, a pallet truck or a forklift moves as one piece, with at least two
  of its three extents 0.3 m or more, that stands in the surveyed area and not inside a cage
  or a storage rack (a thing that leans against a rack or a wall from outside is not inside it). Vehicles and
  trailers are units too. What counts as one unit:
  - a pallet with its collars is one unit (a pallet box); a bare pallet is a unit;
  - a trolley or a cart is one unit, and the goods stacked on it are another;
  - sacks, bags, cardboard and the like that lie loosely on a unit are not part of it (see below).

  Label each unit with a class from `class_names.json`. The list names more classes than the bay contains; it does
  not say which classes are present, or how many of each.
- **Environment** (static): the floor, the walls, the pillars, the raised dock, the raised walkway, the storage
  rack with what is stored in it, the cage walls, every door or gate that is not named below, and fixed
  installations.
- **Gates**: the two sliding wire-mesh gates of the storage cages, and nothing else. The two cages stand side by
  side behind one end wall of the hall, the wall with the yellow direction signs above an opening. Between the two
  cage fronts a concrete pillar with black-and-yellow hazard stripes and a black downpipe stands in the wall line.
  Each cage front is one wire-mesh leaf that hangs from a rail on the wall above it, one leaf on each side of that
  pillar:
  - the gate on the side of the pillar towards the opening under the yellow signs;
  - the gate on the other side of the pillar, towards the corridor that ends at a blue door.

  Every other door, shutter or gate in the photos, the large sliding door of the hall included, is environment.
- **Not units**, whatever their size: small or soft things such as buckets, pots, sacks, bags, loose cardboard,
  packaging and hoses. Leave them out. The assessment does not count the ones it knows of, for or against you, but
  a modelled thing that it cannot match counts as a unit that is not there.

## The surveyed area

The surveyed area is the floor that the photos show from at least two positions: the hall, from the face of the
raised dock at one end to the cage fronts at the other and from one long wall to the raised walkway along the
other, and the front part of the side room that opens off the hall. The top of the dock and the rooms behind it,
the insides of the cages and of the rack, the openings and recesses in the end wall beside the cages and the
corridor that leads to the blue door are not part of it. It is assessed in three parts:

- **the dock end**: the end of the hall in front of the raised dock, the floor under the dock's overhang included,
  as far into the hall as the first free-standing pillar and the near corner of the side room's opening;
- **the side room and its forecourt**: the side room with the storage rack, and the strip of hall floor in front
  of its opening;
- **the main hall and the walkway**: the rest of the hall floor, up to the cage fronts, and the raised walkway
  along the long wall opposite the side room.

A unit belongs to the part in which the centre of its footprint lies; a unit whose centre lies outside the surveyed
area is not counted. Every part counts the same, however many units it holds: a twin of one part alone earns that
part's share only. Places that the photos do not show from two positions are not assessed.

## What the team does not need

Textures and colours are welcome (they help you compare a render with a photo) but play no part in acceptance.
Not needed and not assessed: wheels that turn, brakes, doors of vehicles, what is stored inside the cages and the
rack, the ceiling and the rest of the building above 2.2 m, small unevenness of the floor, exact
masses, friction coefficients and contact softness (the team's simulator applies its own standard values, as long
as each unit's mass is physically plausible for its size).

Chains, tape barriers, cables and hoses get **no collision geometry**: leave them out of the twin, or give them
visual geoms only. The robot's rounds pass under a hanging chain, and a chain modelled as a solid would be a wall
that the bay does not have.

The full technical contract (files, frames, the cameras, the rules for units and gates, the standard simulation,
the probes, the robot's walks and how the twin is assessed) is in `twin_contract.md`.
