# Void search in a collapsed warehouse with a rolling tensegrity robot steered from one overhead camera

An earthquake has brought down the racking hall of a warehouse. Under the fallen slabs there are voids that may hold
people, and nobody can go in. The rescue team has lowered a mast through a hole in the roof; the search camera at its
head, a fixed RGB-D camera, looks down on a patch of the ruin. Onto that patch the team lowers its **RT-3 search
tensegrity** on a rope: three struts held apart by cables and elastic stretch sensors, a third of a metre long and
under a kilogram, which rolls over the floor by winding and unwinding six of its cables. The robot carries no camera
and no computer. A program on the team's PC outside the building watches the camera's frames, receives the robot's
telemetry over the radio, and sends the robot rolling primitives and cable-length targets.

You are the team's robotics engineer. **Write that program.** In each episode the robot is set down on the pad of a
site your program has never seen, and the program has to:

1. take the robot to **three search points**, whose coordinates the mission gives, and at each one **stand still for
   the dwell time and then report the point** (an air sampler on the robot takes one reading during the stay; the
   sampler is not modelled, only where the robot stands and that it stands still);
2. do this also for the third point, which lies in a **crawl space under a collapsed slab** at the back wall, **where
   the camera does not see**: there the program steers the robot blind;
3. **bring the robot back** to the retrieval area around the pad, where the team can hook it.

## What you are given

| path | content |
|---|---|
| `~/given/robot_handbook.md` | the robot: dimensions, the numbering of struts, nodes, cables and sensors, the end-cap colours, the rest shape and the faces it stands on, the cable controller, the eleven rolling primitives with their target states and the symmetry table, this robot's measured table of every primitive on a flat floor, what is not guaranteed, every telemetry field with its noise, the IMU frames |
| `~/given/station_api.md` | the programming interface: every call, its reply fields, errors, budgets, how your program is run, the practice ground's calls and debrief, the wire |
| `~/given/site_briefing.md` | the team's briefing: what is known about a site and what is not, what the ruin is made of, what the camera sees |
| `~/given/camera_datasheet.md` | the search camera: intrinsics, frames, the depth encoding and its noise, the calibration of its pose |
| `~/given/calibration_site/` | the practice ground's calibration site: `calibration_site.md` (every dimension), `site_plan.json` (the same as data), `site_plan.png` (the plan) |
| `~/given/lib/voidcell/` | the client library (`voidcell.client`): the `station` proxy, the launcher, `connect_practice` |
| `~/example/operate.py`, `~/training/run_mission.py` | a minimal program that runs and scores nothing; the tool that runs your submission in a practice episode |

A site's own data (the site rectangle, the pad, the three search points, the camera's calibration, the budgets) come
at run time from `station.mission()`.

<!-- BEGIN GENERATED tools -->
`/opt/venv/bin/python3` (first on `PATH`) is the interpreter that runs your program in a practice run and in a scored episode, with numpy 2.5.4, scipy 1.18.1, OpenCV (`cv2`, headless) 5.0.0.93, Pillow 13.0.0, PyYAML 6.0.3. No network in your session.
<!-- END GENERATED tools -->

## What you leave

`/home/user/submission/operate.py` defining `solve(station)`, with any helper modules and data files next to it.
`station` is the proxy of the given client library; every method is one request to the station, and simulated time
passes only inside `station.step(dt)`.

<!-- BEGIN GENERATED regime -->
**How a scored episode runs it:** one fresh process per episode, under an unprivileged account that is not yours, started as `python -B -E -s` from a read-only copy of your submission folder (regular files only), with an empty writable `HOME` and `TMPDIR` of its own, no network, the numerical libraries held to 2 threads, and an environment that holds nothing of yours (`PATH`, `HOME`, `TMPDIR`, `LANG`, `PYTHONPATH` = the given library, `ARB_GIVEN_DIR` = a read-only copy of `~/given`; the full list is in `~/given/station_api.md`, section 8). Nothing the program writes is kept from one episode to the next. `~/training/run_mission.py` runs it the same way, from read-only copies too; only the account differs: there it runs under your own, so a program that reads a file of your home directory by its path works there and fails in a scored episode. Reach the given files through `os.environ.get("ARB_GIVEN_DIR", os.path.expanduser("~/given"))`. `print` goes to the standard error stream: you see it in a practice run, not in a scored episode.
<!-- END GENERATED regime -->

<!-- BEGIN GENERATED contract -->
**The contract** (a submission that breaks it scores 0 on every episode): `operate.py` is a regular file that parses and defines `solve`; importing it under this regime takes at most 120 s of wall clock and 120 s of CPU time (do heavy precomputation inside `solve`, not at module level); the folder is at most 60 MB and holds no file ending in `.pt`, `.pth`, `.onnx`, `.safetensors`, `.ckpt`, `.h5`, `.pb`, `.pkl`, `.npz`, and no training call; your own `.py` files import no networking, process or foreign-function module (`socket`, `ssl`, `urllib`, `http`, `requests`, `asyncio`, `selectors`, `ctypes`, `cffi`, `subprocess`, `pty`, `multiprocessing.connection`, `logging.handlers` and the others listed in `~/given/station_api.md`, section 8), load no module under a computed name, run no computed code (`exec`, `eval` or `compile` of anything but a string literal) and start no other program (`os.system`, `os.exec*`, `os.spawn*`, ...); over the episodes the program makes at least one `step` call, leaves no process behind and holds no network socket.
<!-- END GENERATED contract -->

Your submission must be final when your session ends: the folder is then copied out of this sandbox and run in a fresh
one that holds nothing else of yours.

## How it is graded

Your program is run on **10 hidden sites**, one episode each: patches of the same ruin, each with its own layout,
openings, ground, lighting, camera pose and search points, none of which you have here. An episode ends when `solve`
returns or calls `finish()`, when the simulated time limit is reached, or when your program exceeds its program time,
CPU time or silence budget, breaks the protocol or exits. However an episode ends, the reports accepted until then are
scored, and the retrieval is judged where the robot is at that instant. A camera frame or a report beyond its budget
is refused and changes nothing, and so is a call with an argument the station cannot accept. An episode in which the
station itself failed while the episode was running is run once more from scratch, and the second run counts; nothing
a program sends can bring that about.

<!-- BEGIN GENERATED budgets -->
| budget | per episode |
|---|---|
| simulated time | 1820 s (calibration site: 1400 s) |
| program time: the wall clock minus the station's own work inside `step`, `get_frame`, `telemetry`, `pose` | 600 s |
| silence: from the start, or from a reply, to your next request | 180 s |
| CPU time of your program, all threads together | 2400 s |
| camera frames (`get_frame()`) | 600 |
| reports (`report_point(i)`) | 3 per search point |
| cable-length targets (`set_targets`) | 100 to 240 mm each |
<!-- END GENERATED budgets -->

<!-- BEGIN GENERATED score -->
An episode's score is S = F × (0.05 c₁ + 0.15 c₂ + 0.80 c₃), with one credit c per search point:

| search point | weight | credit c, from 0 to 1 |
|---|---|---|
| 1 | 0.05 | dwell × distance |
| 2 | 0.15 | dwell × distance |
| 3 | 0.80 | dwell × distance |

- **Position.** The robot's position is the centroid of its six nodes (the centres of its end caps), projected on the floor. Dwell, distance and retrieval are judged on the robot's true position, never on anything your program sends.
- **Report.** A `report_point(i)` call that the station accepts (each point at most 3 times). Only the last accepted report of a point counts, also when an earlier one was better.
- **Dwell**: 1 when all three hold, else 0: the report comes at least 20 s after the episode's start; no command was executing at any moment of the 20 s before the report (a command is a primitive or a `set_targets`, from its acceptance to its end; a `stop` is not a command, and a refused command never ran); and over those 20 s the robot's position, sampled 10 times a second, was never more than 0.02 m from where it is at the report.
- **Distance**: 1 when the robot's position at the report is within 0.20 m (horizontally) of the search point, falling linearly to 0 at 0.40 m.
- **Retrieval factor F**: 1.0 when, at the instant the episode ends, the robot's position is within 0.60 m of the pad's centre and no command is executing; otherwise 0.85.
<!-- END GENERATED score -->

Nothing is subtracted anywhere: touching the rubble costs nothing, the robot cannot be damaged, and a report that
earns no credit costs only what it replaces. A scored episode tells your program nothing about its credit: the reply
to a report says only that the report was accepted.

<!-- BEGIN GENERATED reference -->
Your result is the mean of S over the 10 episodes (0 if the contract is broken), compared with the measured value of a reference program on the same sites; full credit is reached slightly below that value. The reference program was given each site's true layout and the robot's true pose at every instant; it looks at no camera frame and reads no telemetry, and it commands the same primitives through the same cable controller as your program. On every scored site it stopped within the full-credit distance of each of the three search points, dwelt, reported, and brought the robot back to the retrieval area, in at most 63 % of the simulated time.
<!-- END GENERATED reference -->

<!-- BEGIN GENERATED pose_source -->
The station gives your program **no pose of the robot**: `mission()["pose_source"]` is `"estimated"` and `pose()` answers `not_available`. Your program finds the robot in the camera frames while it is in view, and keeps track of it from the primitives it has commanded and from the telemetry while it is not.
<!-- END GENERATED pose_source -->

## What you can rely on at a site

The team states this of every site, scored or practice (details in `~/given/site_briefing.md`):

- **The site.** A rectangle of floor inside the ruin, enclosed by a perimeter wall; the site frame has x east, y
  north, z up and its origin at the south-west floor corner. The camera's mast stands at the south side, and the north
  side is the warehouse's back wall. The floor is level concrete: no steps, no kerbs, no slopes. What stands on it is
  rubble that the robot cannot move: blocks, fallen panels resting on supports, crates, rubble heaps and mounds of
  fill. Nothing on a site moves but the robot, and the light does not change during an episode.
- **What the mission gives is right.** The site rectangle, the pad's centre and the three search points are surveyed
  coordinates in the site frame; the camera's intrinsics are exact and its pose is calibrated to the accuracy stated
  in the camera's datasheet. Neither the pad nor a search point is marked on the floor.
- **What the camera sees.** The camera's picture covers the site's floor from the south wall to about 4 m north of it;
  the site reaches further north, and that part lies under the slab. The pad and search points 1 and 2 lie in the
  open, in the camera's view. The crawl space and search point 3 are out of its view altogether. Of a covered opening
  the camera sees the face that is turned to it and at most a narrow strip of floor inside.
- **Openings.** The rubble is passed through openings: open gaps between blocks, and covered openings under fallen
  panels. They differ in width and in clear height, and some of them are too narrow or too low for the robot.
- **The way to point 2.** Between the pad and search point 2 a ridge of rubble crosses the whole width of the site.
- **The way to point 3.** One collapsed slab covers the north part of the site, from its front edge to the back wall
  and from the west wall to the east wall. Under it lies a crawl space: straight passages between fallen racking,
  parallel to the site's walls, which meet at right angles; a passage can end blind. Fallen material can stand against
  one side of a passage and narrow it for a short stretch; what stays open beside it is as wide as the openings under
  the slab's front edge. Search point 3 lies in a chamber of the crawl space in one of the site's two back corners:
  the back wall and the east or west wall of the site are two of the chamber's walls, the floor between the point and
  these two walls is clear, and nothing stands within 0.6 m of the point. The crawl space is entered from the south,
  through openings under the slab's front edge. There are three of them, alike in width, height and cover; one leads
  in, the other two are blocked by fallen material a short way inside, and the camera cannot see that from outside.
  From the opening that leads in, the way to search point 3 is several metres long and turns more than once.
- **Heaps.** Heaps stand on the open floor, each a rubble heap (bricks and lumps of broken concrete) or a mound of
  fill (dumped earth). Each stands against a wall, against the ridge or against the slab's front edge; none
  stands in the crawl space. A heap is between 0.14 and 0.30 m high at its top and at most 0.60 m long in x and in y,
  with an irregular outline; a mound of fill has sloping sides, a rubble heap can have upright faces. The robot cannot
  get over a heap: go around it. Between a heap and anything else that stands on the floor, the floor is either less
  than 0.15 m wide, far too narrow for the robot, or at least 0.60 m wide. No heap stands within 0.45 m of the pad's
  centre or within 0.40 m of search point 1 or 2, and in front of each opening under the slab the floor is free of
  heaps over the opening's width and 0.70 m southwards. The camera sees every heap.
- **The ground.** Patches of loose ground lie on the concrete, flush with it: dust and gravel, each visible as an area
  of another colour. On loose ground the robot does not move as the handbook's table says. Scattered over the floor
  lie flat chips of plastic, a few centimetres across, in the three colours of the robot's end caps; they do not
  affect the robot. Thin litter lies on the floor as well (loose pages, flattened cardboard, film, strapping, spilled
  powder, grit, and the earth and grit strewn around the foot of a heap): it is flat and does not affect the robot.
- **The robot and the link.** The robot is the handbook's robot. At the start it stands in its rest shape on one of
  its rollable faces, near the pad's centre, at any heading. The radio link holds everywhere on a site, under the
  slab too: commands arrive and the telemetry keeps coming.

What the team does not know of a site, and no given file tells: its layout in numbers, where its heaps stand, which
of its openings the robot can pass, which opening under the slab leads into the crawl space, how the passages of the
crawl space run and where they are narrowed, and how the robot moves on its loose ground.

## The practice ground

The team's practice ground runs in this sandbox as a service whose files you cannot read. It runs the same robot,
station and rules as the scored episodes on four sites, none of them a scored site:

<!-- BEGIN GENERATED practice -->
| number | name | what | simulated time limit |
|---|---|---|---|
| 1 | `cal` | the calibration site | 1400 s |
| 2 | `p1` | a practice site | 1820 s |
| 3 | `p2` | a practice site | 1820 s |
| 4 | `p3` | a practice site | 1820 s |

Allowance: 30 practice episodes in your whole session, at most 2 running at the same time.
<!-- END GENERATED practice -->

The calibration site is surveyed completely, and `~/given/calibration_site/` tells everything in it: a flat floor with
painted grid lines and four surveyed crosses; a ridge with six openings side by side (three open gaps of different
widths, a low covered slot, a covered mouth that ends at a back wall, a covered passage); behind the passage a roofed
room, and from that room a roofed passage, narrowed at one place by a block against its west wall, to a roofed chamber
in the site's north-east corner, none of which the camera can see into, with the site's third search point in the
chamber; a free-standing wall; a strip of gravel and a strip of dust; two crates, a rubble heap, a mound of fill and
six plastic chips. Its page states the dimensions and nothing else: what the robot can pass is for you to find out.
The three practice sites are sites like the scored ones.

`/opt/venv/bin/python3 ~/training/run_mission.py --site cal` runs `~/submission` in one practice episode the way a
scored episode runs it, returns when the episode is over (also when your program hangs: the silence limit ends the
episode) and prints the trainers' debrief; `--list` shows the sites and the episodes left. You can also drive the
robot step by step from a script of your own with `voidcell.client.connect_practice(site)`. Every episode draws its
own set-down and its own sensor noise.

The debrief of an episode on the calibration site gives the true positions of the robot's six nodes twice a second,
under the roofs too, and the true displacement of every primitive that ran. The debrief of an episode on a practice
site gives, for every report, the true distance to the search point, whether the dwell counted and which condition
failed, and the credit; whether the robot was retrieved; the periods in which it was stuck; the budget use; the
events. It gives no total, no layout and no path. A scored episode has no debrief.

<!-- BEGIN GENERATED costs -->
With one episode running, the station takes about 0.23 s of wall clock per simulated second (rolling or standing alike) and about 0.44 s per camera frame. An episode that uses all 1820 s of simulated time therefore takes about 6.9 minutes for the simulation alone, a full calibration-site episode about 5.3, and all 600 frames add about 4.4 minutes, before your program's own computing. The whole allowance spent on full-length episodes with every frame taken, 2 at a time, would come to about 2.8 hours: end a practice episode (`finish()`) as soon as it has shown what you wanted. The sandbox has 8 CPUs and 16 GB of memory, which your program shares with the station, in a practice run as in a scored episode; inside either, the numerical libraries are held to 2 threads.
<!-- END GENERATED costs -->

## Notes

- Camera frames and the calibration site's plan are meant to be processed by programs; look at an image yourself only
  when you need to.
- Do not modify the given files in your home directory; the scored episodes use their own copies.
- Environment facts (harness limits): a single tool call whose input runs above roughly 25-30k tokens is dropped by the
  harness (it is never executed and no error comes back), and a response that reaches the 64k output-token limit is lost
  and, after at most a few retries, ends the session with whatever is in `/home/user/submission/` at that moment. Write
  large files in several parts (one function group per write), keep every single response well below that limit, write
  a first complete submission early and refine it.
<!-- BEGIN GENERATED session -->
- Session time: this session is stopped after 4 hours of wall-clock time from its start. A session stopped by that limit is graded on whatever the submission folder holds at that moment, so a file left half-written then is what runs. Note the start time with `date` when you begin, keep a complete submission in place from early on, and end the session yourself (stop working and say that you are done) at least 15 minutes before the limit.

You have 4 hours. Save your work in `/home/user/submission/` as you go.
<!-- END GENERATED session -->
