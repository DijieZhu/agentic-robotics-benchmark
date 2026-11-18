# Lab runs of the car

Fourteen runs of this car in the team's lab hall, recorded before the season. Each file holds, tick by tick, the
commands that were sent, everything the car's sensors returned, in the same form as at a venue, and the pose of
the car from the lab's overhead tracker. The format is described in `../interface/car_api.md`, section 10;
`carkit.read_log(path)` reads a file.

The lab is not a venue. Its floor grips more than any venue floor so far (`../league/venues.md`), its duct walls
have no gaps and no openings, and a venue has no overhead tracker.

## The lab

**Frame.** The lab frame is the frame of the overhead tracker: origin at the centre of the lab's oval, x along the
long axis of the oval, y to the left of x, yaw counter-clockwise from x. All positions below are in metres in this
frame. A tracker sample is the pose of the car frame (`../car/datasheet.md`, section 1) in the lab frame, to
0.1 mm and 0.0001 rad; take it as exact.

**Hall.** The inner faces of the hall's walls are at x = -6.2 and x = 17.0, y = -5.6 and y = 5.6. Along the walls
stand a cabinet (0.5 x 1.2 m, 1.9 m high, centre at (-5.9, 4.7)), a bench on four thin dark legs (at x = -5.95 and
-5.35, y = -5.3 and -3.9), a rack (0.5 x 2.0 m, 2.0 m high, centre at (16.7, -4.3)) and a crate (0.6 x 0.6 m,
0.5 m high, centre at (16.6, 5.2)).

**Floor.** A glossy polyurethane sports floor, level, the same everywhere (`polyurethane_sports_floor` in the
headers). To the scanning sensor it behaves like the hard, shiny floors of venues. The grip of this car's tyres
on it is 1.00: full braking decelerates the car at 9.8 m/s^2.

**Oval.** A small oval between duct walls, driven counter-clockwise. Its centreline is two straights of 3.2 m, at
y = -2.0 and y = +2.0 from x = -1.6 to x = +1.6, joined by two half circles of 2.0 m radius about (-1.6, 0) and
(+1.6, 0); it is 18.97 m long. The track is 1.5 m wide: the faces of the walls are 0.75 m from the centreline on
both sides. The walls are ducts of 0.33 m diameter, laid end to end without gaps, all the way round on both
sides. The ducts are orange and yellow with black ribs, except one: the outer wall is plain black from
(3.95, 1.73) in the eastern bend, along the northern straight, to (-3.20, 2.44) in the western bend.

**Open floor.** East of the oval, from about x = 5 to the eastern wall, the floor is free. The circles, the
steering steps and the spin were driven there. The straight-line runs were driven along the northern wall, on the
line y = 4.35.

**Carpet tile.** For one run a carpet tile was laid on the floor near the southern wall: 2.0 m x 1.5 m, 6 mm
thick, its edges parallel to the walls, covering x = 6.2 to 8.2 and y = -5.05 to -3.55. Its grip was not measured.

**Between runs** the car was switched off and on again.

## The runs

Every run begins with one second at rest and ends at rest. "Speed target" and "steering" are the commands of
`step`. Where a run follows a line, an operator's line follower steered the car on the tracker's pose; the
commands it sent are in the file like all others.

| file | length | what was driven |
|---|---|---|
| `circle_r10_ramp.jsonl` | 31.2 s | on the open floor, steering held at +0.2523 rad (a circle of 1.0 m radius to the left); speed target 1.0 m/s, then raised slowly (0.12 m/s per second) to 3.9 m/s and held 2 s; then speed target 0 with the steering held |
| `circle_r18_ramp.jsonl` | 36.6 s | the same to the right at -0.1423 rad (1.8 m radius), speed target 1.5 to 5.0 m/s |
| `circle_r26_ramp.jsonl` | 34.1 s | the same to the left at +0.0988 rad (2.6 m radius), speed target 2.5 to 5.7 m/s |
| `circle_r13_steps.jsonl` | 24.5 s | steering held at +0.1958 rad (1.3 m radius); speed targets 1.5, 2.5 and 3.2 m/s for 6 s each, then 3.8 m/s for 4 s; then speed target 0 |
| `steer_steps_slow.jsonl` | 24.5 s | speed target 2.0 m/s; steering steps of 0.15, 0.25, 0.35 and 0.4189 rad to the left, then the same to the right, each held until the car had turned through half a turn, 0.8 s straight between them |
| `steer_steps_fast.jsonl` | 9.2 s | speed target 3.2 m/s; steering steps of 0.07, 0.10 and 0.13 rad to the left, each held through a quarter turn, 0.3 s straight between them |
| `launch_top_speed_brake.jsonl` | 6.2 s | from rest with the largest speed target (8 m/s) along the hall until the car ran at its top speed, then speed target 0 to a stop |
| `speed_steps_reverse.jsonl` | 10.9 s | on a straight line: speed targets 2.0, 4.0 and 2.0 m/s in steps, speed target 0, then 2 s of reverse at -1.5 m/s and a stop |
| `oval_laps_2mps.jsonl` | 33.1 s | three laps of the oval on its centreline, speed target 2.0 m/s; then on at 1.5 m/s to a stop |
| `oval_laps_3mps.jsonl` | 28.2 s | four laps of the oval on its centreline, speed target 3.2 m/s; then on at 1.5 m/s to a stop |
| `oval_laps_fast.jsonl` | 27.0 s | five laps of the oval on its centreline, speed target 4.6 m/s on the straights and 4.0 m/s in the bends; then on at 1.5 m/s to a stop |
| `spin.jsonl` | 6.5 s | on the open floor: straight at 4.2 m/s, turn-in to the left, 1.6 s of full lock with the largest speed target, then speed target 0 with the steering straight while the car was still cornering: the car spun |
| `carpet_tile.jsonl` | 18.9 s | along the southern wall at 3.0 m/s square on to the tile, speed target 0 as soon as the car was on it, to a stop on the tile; away again and off its far edge at 2.5 m/s; round; back across the tile on a diagonal at 2.0 m/s |
| `duct_touch_slow.jsonl` | 6.2 s | in the oval, from a start beside the outer wall and turned 0.18 rad toward it: straight on at 0.5 m/s into the wall, 0.2 s more after the first contact, then steered off the wall and stopped |

## What was measured afterwards

- In none of the runs but the last did the car touch anything.
- After `duct_touch_slow` the duct stood 2.4 cm from where it had lain, at the place where it had moved most. By
  the league's rules (clause S-13) that is a touch.
