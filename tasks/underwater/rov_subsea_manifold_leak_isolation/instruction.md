# Isolate the leaking branch of a subsea manifold with an autonomous inspection-class ROV

After a storm the six-slot production manifold of a shallow gas field, in 45 to 75 m of water, has lost its control
umbilical. The platform sees that the manifold is losing pressure and that gas reaches the surface, and nothing more.
Before it had to leave, the support vessel set a deployment cage with an acoustic modem on the sea bed near the
manifold; in the cage sits a battery-powered inspection-class ROV with a leak marker in its jaws. The acoustic link
carries no video and no joystick: whatever is done down there is done by the program the vehicle carries. The storm
also dropped a piece of deck equipment onto the mud mat in front of the manifold's intervention panel.

All six wells are live. A healthy well that is shut in at sea-bed temperature forms a hydrate plug and takes weeks to
bring back, so closing everything is not an option: the leaking branch has to be found, and only that branch isolated.

You are the operator's intervention engineer. **Write the vehicle's program.** The vehicle has no manipulator arm: its
one-axis gripper holds a valve handle and the whole vehicle rolls about the gripper axis to turn it. It knows the site
only through its own sensors: a camera with lamps, a mechanically scanned sonar, a DVL that can lose its bottom lock,
a depth sensor, an attitude sensor and a navigation estimate that drifts. Nothing tells it where the manifold really
is, which branch leaks, where a handle stands or what it has achieved.

On each dive the program has to:

1. **find the manifold**, which lies some metres away from its charted position;
2. **find out which of the six branches leaks** (the plume of bubbles is bent and spread by the tidal current) and
   **put the leak marker down on the roof grating, inside the marker zone of that branch**;
3. **survey the intervention panel as found**: take camera frames of the tag plates of its valve positions;
4. **close the valves of the isolation set**, that is the valves of the leaking branch that are open as found, and
   move no other valve;
5. **move the dropped object** away from the panel into the laydown area, holding it by one of its lifting eyes;
6. once the isolated segment has bled down, **take the hot stab out of its parking port and latch it in the test port
   of the leaking branch**;
7. **bring the vehicle back into the cage** and to rest there.

The order is yours. Physics sets the only constraints: the marker is in the jaws until it is put down; a hot stab
cannot be latched into a test port that is under pressure; a dropped object that stands in front of a port has to go
before that port can be reached; and contact work at the panel is not reliable while the current there is above 0.30
m/s, which at some sites it is when the dive starts.

## What you produce

`/home/user/submission/intervene.py` defining `solve(vehicle)`, with any helper modules and small data files next to
it. `vehicle` is the proxy of the given client library: every method is one request to the vehicle, and simulated time
passes only inside `vehicle.step(dt)` and `vehicle.sonar_scan(...)`. There is no report to write: nothing the program
prints, returns or stores is assessed, only what it does to the site.

**How a graded dive runs it:** one fresh process per site, under an unprivileged account that is not yours, started as
`python3 -B -E -s` from a read-only copy of your submission folder (regular files only), with an empty writable `HOME`
and `TMPDIR` of its own, no network, and an environment that holds nothing of yours (`PATH`, `HOME`, `TMPDIR`, `LANG`,
`PYTHONPATH` = the given library, `ARB_GIVEN_DIR` = a read-only copy of `~/given`, and settings that hold the
numerical libraries to one thread). Nothing the program writes is kept from one site to the next. Reach the given
files through `os.environ.get("ARB_GIVEN_DIR", os.path.expanduser("~/given"))`. What the program prints is not kept
from a graded dive (the launcher sends `print` to the standard error stream); in a practice run you see it.

**The program runs on one CPU core.** It cannot change its own CPU affinity (the call fails), and more threads or
processes do not make it faster. Its own computing is limited per dive (program time, below); the vehicle's simulation
and the rendering of its sensors do not count against it.

**The contract** (a submission that breaks it earns nothing at any site): `intervene.py` is a regular file that parses
and defines `solve`; importing it under this regime takes at most 120 s (do heavy precomputation inside `solve`, not
at module level); the folder is at most 60 MB and holds no file ending in `.pt`, `.pth`, `.onnx`, `.safetensors`,
`.ckpt`, `.h5`, `.pb`, `.pkl` or `.npz`, and no training call (`torch.optim`, `.backward(`, `jax.grad`); your own
`.py` files import no networking, process or foreign-function module (`socket`, `ssl`, `urllib`, `http`, `requests`,
`asyncio`, `selectors`, `ctypes`, `cffi`, `subprocess`, `pty`, `multiprocessing.connection`, `logging.handlers` and
the like), load no module under a computed name, run no computed code (`exec`, `eval` or `compile` of anything but a
string literal) and start no other program (`os.system`, `os.exec*`, `os.spawn*`, ...); over the six sites the program
makes at least one `step` call, leaves no process behind and holds no network socket; and nothing is written into the
submission folder after your session has ended.

Your submission must be final when your session ends: the folder is then copied out of this sandbox and run in a fresh
one that holds nothing else of yours.

## What you are given

| path | content |
|---|---|
| `~/given/vehicle_manual.md`, `~/given/figures/`, `~/given/gripper_outline.json` | the vehicle at datasheet level: arrangement and mass properties, thrusters and rated capability, the autopilot (modes, what it does when it loses the bottom lock, when its thrusters saturate and when the vehicle is held by a structure), every sensor, the gripper with its grasp rule, slip thresholds and outline data, the leak marker, the protective behaviours, the battery, operating guidance, and what the manual does not give |
| `~/given/vehicle_api.md` | the programming interface: the 17 calls with their arguments, replies, units and frames, the budgets, the reasons a dive ends, the errors |
| `~/given/field/panel_standard.md`, `~/given/field/figures/`, `~/given/field/panel_codes.json`, `~/given/field/panel_codes_bytes.json` | the standard of the intervention panel: handles, stops and torque class, tag plates with their square codes (the code book, also packed as bytes for a custom marker dictionary), ports and the hot stab, guard rail, marker zones, laydown area |
| `~/given/field/isolation_philosophy.md` | the process of a branch, the isolation set, what must never happen and why, the operating limits, the deployment cage, and every condition by which a dive is assessed |
| `~/given/field/manifold_ga.md`, `~/given/field/manifold_ga.json` | the type general arrangement of the manifolds: structure, slots, the leak segment and flanges of every branch, roof grating, panel side |
| `~/given/field/metocean_summary.md` | tide, current near the bed, gusts, visibility, water |
| `~/given/field/dropped_object_register.md` | the stands that can have gone overboard and their lifting eyes |
| `~/given/field/leak_detection_note.md`, `~/given/field/samples/` | what such a seep looks like to the camera and to the sonar, with sample frames |
| `~/given/training/tank_drawing.md`, `~/given/training/training_panel_calibration.md`, `~/given/training/sister_site_ga.md`, `~/given/training/figures/` | the two practice facilities (below) |
| `~/given/lib/subseacell/` | the client library (`subseacell.client`): the `vehicle` proxy, `CellError`, the frame and sonar decoders, the program launcher, `connect_practice` |
| `~/given/example/intervene.py` | a minimal program that runs and earns nothing: the starting point |
| `~/given/training/run_session.py` | runs your program in one practice run |

Each site's own documents come at run time from `vehicle.mission()`, the work order: the chart (nominal position and
heading of the manifold, and the as-laid routes of the two lines that are tied in to it), the cage (exact), the
as-designed panel layout, the hand-over state table, the tide forecast, a telemetry summary, the dropped-object report
and the budgets.

Tools: Python 3.12 at `/opt/venv/bin/python3` (call it by that path) with numpy 2.5.4, scipy 1.18.1, Pillow 13.0.0,
OpenCV 5.0.0.93 (`cv2`, headless build, with its marker module `cv2.aruco` and its pose solvers) and PyYAML 6.0.3.
These are the packages a graded dive has too; nothing else can be installed. No network in your session.

## How you are graded

Your program is run once at each of **6 sites** that you do not have, the same sites for every submission: manifolds
of this type, each with its own position, panel as built, valve positions as found, leak, tide, water and dropped
object. A dive ends when `solve` returns or calls `finish()`, when the simulated time of the dive is used up, when the
vehicle is disabled or surfaces by itself (protective behaviours, below), or when the program exceeds its program time
or its silence limit, breaks the protocol or exits. However a dive ends, the site is assessed as it is at that moment.
The assessment reads the true state of the site at the end of the dive and what truly happened during it, as the
vehicle cell recorded it; nothing your program reports is an input. A dive in which the vehicle cell itself failed is
run once more from scratch, and the second run counts; nothing a program sends can bring that about.

### Budgets of a dive

| budget | per dive | when it is used up |
|---|---|---|
| simulated time | 2400 s | the dive ends |
| program time: the wall clock of the dive minus the vehicle cell's own work inside `step`, `camera_frame` and `sonar_scan`, on one CPU core | 1200 s | the dive ends |
| silence: from a reply to your next request | 180 s | the dive ends |
| camera frames | 600 frame units: a 640 x 480 frame costs 1.00, a 320 x 240 frame 0.40 | a further frame is refused; the dive goes on |
| sonar pings | 16000 | a scan whose pings do not all fit is refused; the dive goes on |
| collision stops | 6 | the 6th disables the vehicle and ends the dive |

`vehicle.mission()["budget"]` repeats these numbers and `vehicle.state()["budget"]` reports what has been used.

### The eleven items

A dive is assessed by 11 items. Each is earned in full or not at all, and the points of the earned items add up to the
result for the site, at most 1.00.

| item | points | part of the work | in short |
|---|---|---|---|
| M1 | 0.03 | marking | the leak marker lies at rest on the roof grating |
| M2 | 0.12 | marking | ... inside the marker zone of the leaking branch |
| H | 0.12 | survey and no harm | panel survey complete, no critical error, vehicle not disabled |
| I1 | 0.08 | isolation | at least one valve of the isolation set closed |
| I2 | 0.22 | isolation | every valve of the isolation set closed |
| I3 | 0.08 | isolation | ... and no misoperation |
| D1 | 0.07 | dropped object | the dropped object moved clear of the panel by a lifting eye |
| D2 | 0.05 | dropped object | ... and at rest inside the laydown area |
| S1 | 0.04 | hot stab | the hot stab taken out and latched in the leaking branch's test port or a parking port |
| S2 | 0.14 | hot stab | the hot stab latched in the leaking branch's test port |
| R | 0.05 | return | survey complete, vehicle at rest in the cage |

The terms:

- A valve is **closed** when its handle stands at 80 deg or more from its open stop.
- A valve is **open** when its handle stands at 10 deg or less. Anywhere below 80 deg the valve still passes.
- The **isolation set** of a site is made of those of the three valves BV, PV and TV of the leaking branch that pass
  as found. It has one, two or three members.
- A **critical error** is a valve outside the isolation set whose passing state differs from its state as found for
  0.2 s or longer: an open valve turned to 80 deg or beyond, or a closed valve brought below 80 deg. It counts at any
  time of the dive, also when the valve is turned back afterwards, and also when the change is still there at the end
  of the dive. The state of the process is evaluated every 0.1 s, and each evaluation stands until the next. It makes
  no difference what moved the handle: the jaws, the hull or a held object.
- A **misoperation** is a valve outside the isolation set that ends the dive more than 10 deg away from its position
  as found.
- The **panel survey** is complete when at least 15 of the 18 valve positions have been imaged.
- A valve position is **imaged** when one camera frame taken by your program shows its tag plate so that all of the
  following hold for that frame. They are judged from where the camera really was and from what was really in the way,
  not from what the program reads in the frame; frames of either size count:
  - the four outer corners of the plate's square code are inside the image, at least 20 px from its edges;
  - the camera is at most 1.3 m from the centre of the code (slant range);
  - the shortest edge of any of the code's 8 by 8 cells, as projected into the image, is at least 4 px long;
  - the line of sight is within 60 deg of the normal of the plate;
  - the lamps are at level 0.3 or higher and the centre of the code is inside the beam of at least one lamp, that is
    within 67.5 deg of that lamp's axis;
  - of 5 by 5 points spread evenly over the code, at least 20 are in free view from the camera. The jaws, anything
    held in them, the guard rail and the dropped object block the view; marine growth on the plate does not.
- An object is **at rest** when its speed stays below 0.01 m/s during the last 5 s of the dive, the end of the dive
  included. In a dive shorter than that nothing is at rest.
- The vehicle is **at rest** when its speed stays below 0.02 m/s during the same last 5 s. A vehicle that hovers in a
  current is not that still: to end a dive at rest in the cage, seat the vehicle on the cage deck
  (`vehicle_manual.md`, section 8.3).
- The vehicle is **disabled** by its collision stop number 6.

The items:

- **M1, 0.03.** During the last 5 s of the dive the leak marker lies at rest on the roof grating of the manifold,
  touching it, and is not in the jaws.
- **M2, 0.12.** M1 holds and the marker's centre of mass is, seen from above, inside the marker zone of the leaking
  branch (`field/panel_standard.md`, section 7). There is one marker: where it lies is the program's statement of
  which branch leaks.
- **H, 0.12.** The panel survey is complete, there was no critical error, and the vehicle was not disabled.
- **I1, 0.08.** At the end of the dive at least one valve of the isolation set is closed.
- **I2, 0.22.** At the end of the dive every valve of the isolation set is closed, so that the leak segment is cut off
  from its well and from both headers.
- **I3, 0.08.** I2 holds and there is no misoperation.
- **D1, 0.07.** At the end of the dive the point of the dropped object nearest to the panel face, its lifting eyes
  included, is at least 1.2 m from the face, and at some time in the dive the jaws held one of its lifting eyes. An
  object that was pushed away by the hull, or that moved by itself, does not earn the item.
- **D2, 0.05.** D1 holds, the object is at rest, and its centre of mass is inside the laydown area
  (`field/panel_standard.md`, section 7).
- **S1, 0.04.** The hot stab has left the port it was parked in, and at the end of the dive it is latched in the test
  port of the leaking branch or in either parking port.
- **S2, 0.14.** At the end of the dive the hot stab is latched in the test port of the leaking branch.
- **R, 0.05.** The panel survey is complete, and during the last 5 s of the dive the vehicle is inside the cage (the
  origin of its body frame inside the cage's box, above the deck) and at rest.

Critical error and totals:

- A critical error at a site removes these items at that site, whatever else was achieved: I1, I2, I3, S1, S2 and H.
  The other items are not affected, and nothing is carried to another site. Closing a valve to see whether the bubbles
  stop, and reopening it when they do not, is a critical error.
- The result for a site is the sum of the points of the items earned there. There are no negative points, and no
  single failure cancels the rest: a disabled vehicle, a dive that runs out of time or an automatic ascent only end
  the dive.
- Your result is the mean of the six site results, and zero when the contract above is broken. The reward is this
  result compared with the measured value of a reference program on the same sites: that value is not disclosed, and
  the full reward is reached slightly below it. The reference program steers by the true positions of the vehicle and
  of the structure, which yours does not have; what it decides, it decides from its own sonar scans, camera frames and
  vehicle state, as yours must.

`field/isolation_philosophy.md` states the same conditions once more, with the process they belong to.

## Rules of the vehicle cell

Details are in `vehicle_manual.md` and `vehicle_api.md`; the essentials:

- The vehicle answers the 17 documented calls. Any other request is answered `forbidden` and counted; it changes
  nothing. No reply tells where an object is, how a handle stands, what a pressure is or what has been earned:
  `state()` is the vehicle's own state only.
- The gripper holds by a rule, not by friction (`vehicle_manual.md`, section 6). Closing the jaws on a blade handle (a
  valve's, the marker's, the hot stab's) that lies inside the documented window attaches the blade to the jaws;
  closing them with the tool point within 35 mm of the centre of a lifting eye hangs the object from the jaws;
  anywhere else the jaws close empty and you try again. Opening the jaws releases what they hold. `state()["gripper"]`
  says `open`, `closed_empty` or `stalled_on_object`, never what is held. A handle stays where it is left: open is
  blade horizontal, closed is blade vertical, and closing is clockwise seen facing the panel.
- The hot stab latches only when it is pushed home with the mating force of the port, which grows with the pressure
  behind a test port (`field/panel_standard.md`, section 5); a segment that is still connected to a pressure source
  stays at line pressure whatever else is done.
- The firmware protects the vehicle; a protective stop costs time and nothing else:

| behaviour | trigger | what happens |
|---|---|---|
| collision stop | contact force on the vehicle body above 150 N, or a contact impulse above 12 N s within 0.1 s (the jaws and what they hold do not count) | thrusters off for 2 s, then depth and attitude keeping for 8 s, motion calls refused meanwhile; the stop counter rises by one; stop number 6 disables the vehicle and ends the dive |
| out of bounds | more than 60 m horizontally from the charted centre of the manifold, or more than 15 m above the sea bed, for 10 s | the vehicle surfaces; the dive ends |
| undervoltage | battery terminal voltage below 12.0 V for 5 s | the vehicle surfaces; the dive ends |
| grasp slip | side force on a held blade above 90 N, or bending moment above 12 N m, for 0.2 s | the blade is released; nothing is damaged and no points are taken |
| hot stab side load | side force above 60 N on a hot stab that is guided in a port | the stab leaves the guide and has to be withdrawn and inserted again; nothing is damaged |

## What you can rely on in a site's documents

The operator states this of every site (details in `~/given/field/`):

- **The chart is nominal.** The manifold lies 2 to 8 m from its charted position and is turned by up to 15 deg against
  its charted heading. The routes of the two lines in the chart are nominal in the same way: they were surveyed
  together with the manifold's charted position, so they are offset and turned with it, and they do not say where the
  manifold or the lines really are. The cage's position, heading and deck height in the work order are exact.
- **The panel as built can differ from the as-designed layout** in the order of its columns and of its rows. The tag
  plates on the panel prevail: read the square codes. On up to 4 valve plates of a panel the code is covered by marine
  growth; the painted tag next to it is not.
- **The hand-over state table can be out of date** in up to 2 entries. The handles show the positions as found.
- **Any of the six branches can be the leaking one.** Nothing in the work order points to it, and neither does the
  place of the dropped object.
- **The tide forecast** gives the time of slack water within 10 min and the direction of the flood within 20 deg.
  Every dive has a window for contact work at the panel within its simulated time; the work order does not say when it
  opens.
- **What does not vary** is the vehicle. The same vehicle dives at every graded site and in both practice facilities,
  and it moves through the water in the same way everywhere: what a practice run shows about how it handles holds for
  the graded dives.

## Practice: the training tank and the sister site

Two practice facilities run in this sandbox as a service whose files you cannot read. They run the vehicle of the
graded dives, with the same firmware, sensors, gripper rule and protective behaviours, behind the same calls.

- **The training tank**: 24 runs, each of at most 600 s of simulated time. A basin with a segment of the standard
  panel (handles of known torque, a parking port, a test port whose pressure you set, a hot stab), a bubbler, a flow
  generator, three turbidity levels, two stands of the dropped-object register, a leak marker, a code board and a
  cage. You choose the settings of each run, and the tank gives back what its instruments measured: the tracked pose
  of the vehicle 5 times a second, the encoder angles of the handles, the lamps of the two ports and the set points
  (`~/given/training/tank_drawing.md`, `~/given/training/training_panel_calibration.md`).
- **The sister site**: 6 runs, each of at most 1800 s of simulated time. A pipeline end termination on open sea bed
  with a two-valve panel and one seep, in a tide and a water that are drawn anew for every run: a whole dive from cage
  to cage (`~/given/training/sister_site_ga.md`). It gives back nothing but the replies to the vehicle's own calls.
- **What they are not.** Neither facility is a graded site and neither assesses anything: there are no points, no
  items and no debrief. The tank has no manifold, no process, no tide and no out-of-date document; the sister site's
  structure, panel and process are not a manifold's. A graded site is never available for practice.
- At most 2 practice runs can be open at a time; a further one is refused as `busy` and uses up nothing. The other
  limits of a practice run are those of a graded dive.

`/opt/venv/bin/python3 ~/given/training/run_session.py tank` (or `sister`, or `status` for the runs left) starts one
run, calls `solve(vehicle)` of `~/submission/intervene.py` and writes what the run gave back into `~/practice_runs/`;
the script's first lines list the tank settings (also `training/training_panel_calibration.md`, section 6). It runs
the program under your own account and not under the regime of a graded dive: a program that reads a file of your home
directory by its path works there and fails when graded. You can also drive the vehicle step by step from a script of
your own with `subseacell.client.connect_practice("tank")`.

Plan the runs: a practice run takes about a third of its simulated time in wall clock for the simulation alone, more
with the frames it takes (a quarter to half a second each) and your program's own computing. A tank run that uses all
of its simulated time takes about 4 minutes, a sister-site run about 9; the whole allowance spent on full-length runs,
2 at a time, would come to about 1.2 hours. End a practice run (`finish()`) as soon as it has shown what you wanted.

## Notes

- Camera frames, sonar scans and the sample pictures are meant to be processed by programs; look at an image yourself
  only when you need to.
- Do not modify the given files in your home directory; the graded dives use their own copies.
- Environment facts (harness limits): a single tool call whose input runs above roughly 25-30k tokens is dropped by
  the harness (it is never executed and no error comes back), and a response that reaches the 64k output-token limit
  is lost and, after at most a few retries, ends the session with whatever is in `/home/user/submission/` at that
  moment. Write large files in several parts (one function group per write), keep every single response well below
  that limit, write a first complete submission early and refine it.
- Session time: this session is stopped after 4 hours of wall-clock time from its start. A session stopped by that
  limit is graded on whatever the submission folder holds at that moment, so a file left half-written then is what
  runs. Note the start time with `date` when you begin, keep a complete submission in place from early on, and end the
  session yourself (stop working and say that you are done) at least 15 minutes before the limit.

You have 4 hours. Save your work in `/home/user/submission/` as you go.