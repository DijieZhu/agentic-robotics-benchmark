# The bay

A maintenance bay on the floor of a process plant's hall. A steam line runs along its back wall; one of its flanges
leaks and throws a plume of steam into the bay. A pan of oil has caught fire on the floor. In the middle of the bay a
small assembly station is still in use: two tables with parts on them and a crate for the parts. Nobody goes in while
the steam and the fire are there. The plant wants a robot fleet that can, from its marked pads:

1. close the leaking line's valve and watch its pressure gauge,
2. put the fire out with a bucket of water from the tank, carried by three flying robots on ropes,
3. then clear the assembly station: every part off the tables and into the crate.

Your job is the simulation they will develop this in: a twin of the bay as it is, with its rig and its hazards, a
fleet you design to meet `requirements.md`, and the program that runs that fleet through the three jobs above,
rehearsed in your twin and graded in the real bay (`control.md`). The photographs in `photos/`, `facts.md`, the account below of the plant's own tests and the offline
libraries are all there is to know about the bay. Nobody can tell you more.

## Where things are

The world frame is fixed by the calibration mat on the floor: a flat board of 7 × 5 black and white squares of
100 mm (700 × 500 mm), long side along the world x axis, its centre at x = 0.00 m, y = -0.60 m, on the floor (z = 0).
+z points up, +y points from the flying robots' pads toward the back wall (the steam line). Everything else you take
from the photographs. `photos/cameras.json` gives each photograph's camera intrinsics (pinhole, no distortion) and
nothing else; the cameras' poses are yours to recover.

The photographs were taken before the fleet existed: **there is no robot in them**. The floor markings below show where
the robots will stand, and the bucket's three ropes lie on the floor with their hooks free, waiting for the flying
robots' hooks.

The photographs show the bay on an earlier day. On the day of the job the parts lie wherever they were left on the two
tables, some of them on their side, the vehicle stands near its parking bay, not quite as it was parked, the valve's
lever may have been left partly turned, and the pan may stand up to 0.45 m (along x and along y) from where the
photographs show it; everything else stands as photographed. Your twin is the bay as photographed; the fleet's program
meets the bay of the day (`control.md`).

Painted on the floor: three square launch pads with an arrow (the flying robots, one per pad, facing the arrow), a
larger rectangular parking bay with an arrow (the ground vehicle, facing the arrow), and a white square in front of
the valve (where the vehicle stands to work the valve).

## What is in the bay

- **The steam line**: a pipe along the back wall on supports, with a tee dropping a branch to a quarter-turn valve
  (a lever on the valve body), a pressure gauge on a stub, and further along a flange that leaks: the plume rises
  from it and drifts into the bay. What the photographs show is only its dense core; the hazard volume is wider
  (`facts.md` gives its measured extent round the leaking flange).
- **The fire pan**: a round pan on the floor, burning.
- **The water tank**: an open tank, full, and a bucket standing beside it. The bucket has three lifting rings on its
  rim, 120° apart, and three ropes already on them, lying on the floor, their hooks free.
- **The assembly station**: two plain tables (A the taller one), six parts on them, and an open plastic parts crate
  from the plant's stock beside them, big enough for all six.
- The hall around the bay, which you may model or leave out.

## How the rig and the valve behave

The plant's people tested both; this is what they saw (`facts.md` has what they measured):

- **The valve**: its lever is turned to its stop, a quarter turn. The leak stops about 3 s after the lever reaches
  the stop and the plume thins out. Turned back, the leak returns at once.
- **The rig**: the plant's test rig, three overhead hooks that move together, lifts the bucket on its ropes. Slack
  ropes lie on the floor and hang in bows; taut ropes are straight, with no stretch to speak of. With the hooks level
  and evenly spaced round the bucket it comes up level and hangs under the hooks' centre; pulled out of true by one
  hook, it shifts and leans a little toward that hook. Carried sideways with a start and a stop it swings like a
  pendulum, about 1.6 s a swing, and settles slowly over many swings. With one hook let down most of a metre that
  rope goes slack and the bucket tips about 25° toward the other two; taken up again it levels. Brought to a dead
  stop the ropes hold; set down, it stands upright.
- **The pour**: carried over the pan and tipped that way, the bucket pours and the fire goes out (the hazard rules
  say when).

## The hazard rules

These are the rules the twin must implement (in OmniGraph, through the state attributes `interface.md` names):

- The steam is **active** from the start (`arb:steam_active` true). Closing the valve stops it 3 s after the lever
  reaches its closed stop; opening the valve again brings it back at once.
- A **flying robot inside the plume's volume while the steam is active is lost**: within 1 s its `arb:disabled`
  becomes true, and it stays true. Outside the plume, or inside it while the steam is off, nothing happens.
- The **fire is active** from the start (`arb:fire_active` true). A **ground vehicle within 2 m (horizontally) of the
  pan's centre while the fire is active is lost**: within 2 s its `arb:disabled` becomes true, and it stays true.
- **Water puts the fire out**: the bucket tipped past 15° from upright, with its body within 0.6 m (horizontally) of
  the pan's centre and below 2.5 m, pours; one second of pouring in all puts the fire out for good
  (`arb:fire_active` false, and it stays false). The pan is then harmless.
- For these rules the bucket counts as full throughout (its authored mass does not change). The bay mission
  (`control.md`) is where the grader itself takes the water's loss into account.
