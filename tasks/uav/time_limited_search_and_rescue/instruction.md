# Civilian Rescue: Warehouse Handoff

A small indoor aircraft has to recover a casualty (a clothed training mannequin under a blanket) that lies on a rigid basket litter somewhere in a
two-storey warehouse wing, and deliver it intact to the **receiving station that is accepting this incident**. The
radio does not reach inside the wing, so the aircraft flies on its own: you write its flight program.

Leave the program in `/home/user/submission/`, started by `run.sh`. After your session a grader flies it, once, on
each of **8 sites you have not seen**. The sites differ in where the doorways are open, where the hoist wells and the
litter are, which fitting the litter carries, which station is open, the room sizes and the aircraft's trim.

## The site

A dispatch arrow near the entrance points toward the open station. At receiving, a lowered shutter bears a large
cross; an open doorway exposes a horizontal deck. The incident stays fixed during a flight. You may inspect either
approach, change your mind, or use any safe route.

Entry safety briefing: center on the violet apron, settle at approximately 0.71 m, cross the inspection slot facing
into it, and climb only after the aircraft has cleared the frame.

## The aircraft

`act([right, forward, up, yaw_left, grip])` takes five numbers in [-1, 1] and advances the flight by 0.2 s. Maximum
nominal setpoint increments are 0.32 m on each horizontal axis, 0.19 m vertically and 30 degrees in yaw. Horizontal
axes use the heading at the **start** of the action; zero holds. Translation gains differ from nominal by a few
percent (fixed for a site). Forward travel drifts sideways by about 0.03 m per metre when unloaded and about
0.13 m per metre when carrying the litter; the side is fixed for a site. Launch is at 1.30 m facing the wing;
altitude limits are 0.39-4.69 m above the launch floor.

Every answer (`observe()` and `act(...)`) holds:

| Field | Content |
|---|---|
| `images` | Two 256-by-256 RGB PNGs (base64). Both cameras turn with the aircraft and have a 94-degree vertical field of view (focal length 119.4 pixels). Camera 0 looks down from 0.46 m forward and 0.145 m below center; right/up in the image are body right/forward. Camera 1 looks forward, pitched 6.9 degrees down, from 0.17 m forward and 0.045 m above center. An attached board can obscure the down view. |
| `sensors.altitude_m` | Barometric height above the launch floor: noise 0.015 m (1 sigma) plus a slow drift that stays within 0.06 m. |
| `sensors.flow_m` | Optical-flow odometer: `[right, forward]` displacement during the last action, in the body frame at that action's start. Scale error up to 2 % per axis (fixed for a site), noise 0.003 m plus 1 % of the displacement (1 sigma). |
| `sensors.gyro_z_rad_s` | Mean yaw rate during the last action: bias up to 0.004 rad/s (fixed for a site), noise 0.01 rad/s. |
| `sensors.accel_m_s2` | Mean gravity-compensated acceleration during the last action, `[right, forward, up]`: noise 0.05 m/s^2. |
| `frame_id`, `steps_remaining`, `done`, `requests_remaining` | Actions flown, actions left, whether the flight has ended, calls left. |

There is no pose, no map and no depth. `observe()` returns the current answer again and passes no time.

The pictures show the site as it is built and furnished: plastered walls with a painted dado (blue-grey downstairs,
green upstairs) and a timber skirting, a concrete floor downstairs and a tiled floor upstairs, windows in the
outer walls with the yard behind them, daylight on floors and walls where the sun comes in, timber door frames,
ceiling lamps, paper on the floors, and the markings and fixtures in the colours this document names. The colours
are worn paint, darker and greyer than a display colour, and a door apron's paint is rubbed through in places.
Three things are drawn and are not structure: the windows are openings in the picture but closed to flight (a
wall is solid over its whole face), a doorway has a frame with a head at 2.10 m and wall drawn above it (the opening
is drawn at its true width; structurally it is open up to the slab), and the yard outside. The aircraft's own
rotors are not in its pictures: both cameras sit clear of the rotor blur. A picture depends on the site and on the
aircraft's state alone: the same state gives the same picture, here and in grading.

**Stock and furniture** stand in the rooms, along walls: cartons, drawer chests, small tables, chairs, barrels,
buckets, a gurney, wall cabinets and posters. Each is solid. They follow one rule at every site. Lanes are the
straight lines that join the centres of two rooms through their doorway, the staging row's centre line, the ways
to the inspection slot, the cargo door, the turning pocket, both receiving sides, the tool stands and the launch
point, the litter's own length and the way from it to its room's centre; a hoist well is free over its whole
bore. Nothing stands within 0.72 m of a lane, and nothing taller than 0.80 m stands within 1.25 m of a room's
centre, of the turning pocket's centre or of a receiving pad's centre.

## Equipment card

| Equipment | Physical contract |
|---|---|
| Litter and heads | Basket litter (orange shell, aluminium rail): 1.80 by 0.44 m; violet end grips 1.62 m apart. Its lifting receiver stands over the casualty's thighs at the litter's centre: a cyan ring, 8.2 or 18.6 cm across, on a post, and 0.05 m above it a dark guide ring with a green band (bore 0.28 m) carried by two hoops. The cyan ring is the flange the head grips; it can be seen from the side and, through the guide ring, from above. Small and large split collets fit the corresponding flange. The clamp preserves the actual pickup yaw. |
| Gripper | Above +0.5 closes, below -0.5 releases, otherwise holds; grip acts before movement. Mount or return a head on its own stand at 0.430 +/- 0.035 m, within 0.085 m horizontally. Latch within 0.135 m of the fitting, aircraft center 0.306 +/- 0.060 m above board center (about 0.40 m above the floor). Steady to at most 0.30 m/s and below 0.30 rad/s. A mismatch does not attach. |
| Wing | Room pitches 2.60-2.80 m; storeys 2.60 m apart, slab 0.22 m thick. Amber hoist thresholds lead to square bores of 1.08 or 2.34 m; back-wall guides are 1.40 m apart. Ordinary white doorways clear 1.48 m. The violet inspection slot is 1.02 m wide with sill/lintel at 0.43/0.89 m. The blue cargo shutter retracts on a held lift of 0.54 m above the local floor. |
| Transfer | The turning pocket is 4.20 by 4.60 m. Jambs give 1.10 m width over 0.56 m depth; lime toe strips extend 1.24 m toward their approach. |
| Handoff | The green deck is 2.50 by 1.48 m. Release with board center no higher than 0.195 m above the floor, speed and yaw rate no greater than 0.34 m/s and 0.34 rad/s. For full credit the released board must physically rest entirely on the deck, center within 0.28 m of the blue target center and at most 0.095 m high, for 0.25 s. The settling limit is 0.16 for the norm of its linear and angular velocities (m/s, rad/s). |

## What ends a flight

- **Contact.** The aircraft, the mounted head and, once it is carried, the whole litter must clear every structural
  surface (walls, door and slot frames, slabs, shutters, jambs) and every piece of stock and furniture. The first
  contact ends the flight.
- **An unsafe release.** Releasing the litter higher than 0.195 m above the floor, or faster than 0.34 m/s or
  0.34 rad/s, ends the flight.
- **The handoff.** The flight ends when the board has rested on the open station's deck as the card says.
- **The budgets.** 4,400 actions, or 9,000 calls (observations count).

A flight that has ended keeps the progress it had reached.

## Scoring

Each of the 8 flights earns its rescue progress; your score is their mean (a mean of 0.98 or more is full score).

| Reached during the flight | Progress |
|---|---|
| Entered the wing (crossed the entrance line without the litter) | 0.05 |
| Reached the upper storey (0.70 m or more above its floor, inside the wing) | + 0.04 |
| Reached the litter: before picking it up, flown to within 0.62 m of its center on both horizontal axes, 0.85 m or more above that storey's floor | + 0.06 |
| Latched the litter (the mounted head fits its fitting) | + 0.15 |
| Lifted it 0.54 m above its floor | + 0.10 |
| Carried it out of the wing across the entrance line | + 0.25 |
| Rooms searched: the share of the 4 marked rescue bays flown over (within 0.47 m of the bay's center on both horizontal axes, 0.85 m or more above its floor) | + up to 0.03 |
| **Handed off on the open station's deck** | **1.00 in total** |

The order of visits is not scored. Without a handoff a flight earns at most 0.68: reaching the litter is worth 0.15 in all, and the handoff is the largest single step.

## Your program

- `/home/user/submission/run.sh` is started once per site with `bash run.sh`, from a read-only copy of the folder,
  as an unprivileged account with a fresh empty home and no network. Other files in the folder come along (at most
  2,000 files, 64 MB).
- The aircraft is behind the Unix socket named by the environment variable `ROBOT_SOCKET`. The client you have here
  is at the same path in the grading sandbox: `/opt/visual_robotics/client/client.py` (`Client().observe()`,
  `Client().act(action)`). Python is `/opt/venv/bin/python3` with numpy, scipy, Pillow, mujoco and trimesh; nothing
  else can be installed there.
- Per site the program may use **900 s of its own time** (wall time minus the time the aircraft takes to answer),
  inside a wall limit of 2,100 s. When either runs out the flight is stopped and keeps its progress. The program is
  stopped 10 s after a flight has ended.
- The grading sandbox has 2 CPUs and 4 GB of memory, as this one. An answer takes the aircraft about 0.24 s (its two
  pictures are drawn on the CPU); the first answer of a site takes about 10 s.
- Nothing carries over between sites: every site starts the program anew, with an empty home; the temporary
  directories, shared memory and message queues are emptied, and nowhere else is writable. `/home/user` is not readable there; what the program needs must be in the submission folder,
  which is its working directory.
- The program's environment is minimal: `PATH=/opt/venv/bin:/usr/bin:/bin`, one thread for OpenBLAS and OpenMP, and
  `OPENBLAS_CORETYPE=HASWELL` with NumPy's AVX-512 paths switched off (`NPY_DISABLE_CPU_FEATURES`). This sandbox sets
  the same two numerics variables, so a program computes the same numbers here and on every grading machine.

## Commissioning aircraft (this sandbox)

The same aircraft is available here on **commissioning sites drawn at random** from the same kind of wing; none of
them is graded. Use it to learn how the aircraft, its cameras and sensors, the fixtures and the passages behave.

- Calls: `observe()`, `act(action)` and `new_site()`, which abandons the current site and starts at the launch
  point of a new one. As MCP tools: `robot_observe`, `robot_act`, `robot_new_site`. From code:
  `/home/user/robot/client.py`.
- Budget for the whole session: **16,000 actions** and **40 sites**. Answers carry
  `commissioning_actions_remaining` and `commissioning_sites_remaining`. A flight here ends by the same rules as a
  graded one; after it, only `observe()` and `new_site()` do anything.
- The aircraft reports what it senses and nothing else: no pose, no site map, no progress and no score, here or in
  grading. `done` tells you that a flight has ended, not why.

## Session time

- This session is stopped 2 hours of wall-clock time after it started. What is in `/home/user/submission/` at that
  moment is graded as it is: keep a complete, runnable submission there from early on and replace it as you improve
  it.
- The only clock for this limit is the wall clock. `bash /home/user/session_time.sh` prints the minutes used and
  the minutes left, from a time stamp written when your sandbox was set up (moments before your session started, so
  you have slightly more than it says, never less); `date` shows the same clock. Run the script when you begin,
  again before every decision that depends on the time left, and whenever you pick the work up from notes or from a
  summary of what was done. Do not estimate the time used from how much you have done.
- The commissioning budget (actions and sites) and a flight's 4,400 actions are counts, not time; neither tells you
  how much of the session is left.
- This sandbox has 2 CPUs and 4 GB of memory, no GPU and no network.
