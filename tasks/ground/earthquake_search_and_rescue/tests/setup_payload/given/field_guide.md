# Field guide: searching collapsed buildings with a robot

The team's notes for robot operators and integrators. They describe what earthquake rubble of a mid-rise building of
reinforced-concrete frames with brick infill looks like from a small robot, and the rules the team works by.

## 1. How these buildings come down

* **Standing bays.** Some rooms keep their ceiling (about 3 m): walls cracked, plaster and bricks on the floor,
  furniture thrown about, heaps of rubble where a wall or part of the ceiling came down.
* **Pancaked bays.** The floors above drop onto the one below and come to rest on whatever is stiff enough: broken
  columns, wall stubs, heavy furniture. Underneath stays a **void**, a crawl space typically 0.75 to 0.95 m high, often
  dark and dusty. Survivors are found in such voids, in pockets next to something sturdy (a wardrobe, a table, a wall).
* **Filled bays.** Where the floors above broke up completely, the room is filled with rubble up to the ceiling; a
  stair core usually fills like this too. There is no way through.
* **The facade.** Rubble spills onto the pavement in front of the building in mounds; shop windows become openings,
  some blocked by fallen facade panels or heavy rubble. The street in front is cordoned off with concrete road barriers
  along the far kerb and across both ends; the team's vehicles stand inside the cordon.
  Beside the deployment point the team has its post: a generator with its fuel cans, work lights on tripods, tool
  boxes and crates, a stretcher, barrier boards and cones. Rubble, pieces of broken wall and slab and bent
  reinforcement lie on the pavement along the facade and in some rooms; a crushed car stands on the street. All of
  that is solid and stays where it is: drive round it.

Between rooms there are doorways and breaches in the walls. In the rubble they come in all states: open, partly
blocked by a fallen concrete block (a narrow slot), blocked by debris that can be moved (a timber beam that fell across
a doorway), raised by a broken slab wedged in the doorway (a step up and down), or closed for good by heavy rubble.
Floors can be broken through to the basement: a **floor gap** where a strip of the floor slab dropped, half a metre
wide and deep enough to swallow a robot. Not every way that looks open goes anywhere: dead ends are common.

## 2. Rules

* **Load-bearing debris stays where it is.** Before moving anything, look at what rests on it. Debris that a slab, a
  beam or a stack bears on is part of the structure now: pull it out and the load comes down (a secondary collapse)
  on whoever is underneath, the robot included. Move only debris that carries nothing: a beam lying free across a
  doorway, a pipe, a board.
* **Never drive onto people.** A robot is heavy; approach a person slowly, stop well short of them. The robot's
  protective stop trips on contact with a person, and a stopped robot stays where it is.
* **Know where you came from.** The way in is the way out: the robot has to come back, and its report only reaches
  the team where the radio link holds.

## 3. Finding a person

* **Thermal imaging** is the best detector at short range in the dark: a person's skin shows at 32 to 35 C, clothing
  a few degrees cooler, dust and debris on them cooler still; only the uncovered parts show. A small warm patch (a
  face, a hand) can be all there is.
* Not everything warm is a person: appliances can stay warm long after the power is gone (a boiler, a water heater:
  flat, box-shaped, evenly warm), and outdoors, metal and asphalt in the sun can be warmer than a person.
* Not everything that looks like a person is one: shop mannequins (cold, smooth, uniform colour), heaps of clothes.
  Combine the camera and the thermal camera: the shape of a person and the warmth of one.
* Victims lie on the floor, usually along a wall, often with their lower legs under a broken piece of slab that rests
  on rubble either side of them, sometimes with a low cupboard standing beyond their feet or their head. That piece
  of slab is wedged where it lies; do not push against it or the rubble under it.
* **Report the middle of the person's torso.** The team plans its tunnel from it.
* **Supplies.** Put the kit down next to the person's head or a free hand, where they can take it: never on them.

## 4. Practical notes for the robot

* Crawl spaces need the mast down and the arm stowed; the arm can work in a void if its elbow is kept low.
* Narrow slots are taken at the narrow track gauge, square to the slot, slowly; the narrow gauge turns poorly, so turn
  before the slot, not in it.
* Steps up need the flipper technique (robot manual, section 3); steps down are taken slowly with the flippers out.
* Floor gaps are crossed square to the gap with the flippers out (a little below straight), slowly and without
  stopping on the gap.
* Lidar and depth see the geometry; the colour cameras and the thermal camera tell what it is. The lamps light the
  view in the dark.
* The odometry drifts by about 1 % of the distance travelled: a long search accumulates error in where you think the
  person is and where the way out is.
