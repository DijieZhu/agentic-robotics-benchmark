# Assemble a twelve-actuator quadruped from its loose units with a 6-DoF arm, guided by its picture manual and the cell's cameras

You are the assembly engineer of a small legged-robot lab. The lab has a quadruped's units ready: the body with its four
hip ab/ad actuator modules, four shoulder modules (the hip flexion actuators: two plain ones and two with the hip
adapter), four hip ab/ad supports, four upper legs (with the knee actuators) and four lower legs. You have what an
assembly engineer really has: every unit as an STL file, the actuator module's datasheet, drawing and STEP model, the
lab's picture manual of this robot (25 photographs with scale references in view, captions without dimensions) and a
robot workcell that you see only through its cameras. The body is bolted on a stand in the cell; the sixteen other units
lie loose on the bench. Today's job is the whole robot: program the cell's 6-DoF arm and its parallel gripper so that it
builds the quadruped the way the manual shows: every shoulder's key into its hip ab/ad output the right way round, every
support over its shoulder onto the body, every lower leg onto its upper leg's knee output and every leg onto its
shoulder's hip flexion output.

Nothing gives the answer directly: there is no mating graph, no design pose, no insertion axis, no camera calibration,
no unit pose and no "fastened" signal. The units lie in different places and turned differently about the vertical in
every run (each on the same face every time), the stand stands slightly differently, every output pulley starts at a
slightly different angle, and the cell's cameras are mounted and calibrated differently in every run.

## What you produce (in `/home/user/submission/`)

`assemble.py` exposing `solve(cell) -> None` (plus any helper modules, optional `run.sh`). The grader runs it once per
hidden episode (3 episodes) in its own physics cell. It talks to the cell through `cell` (API in
`~/given/cell/cell_api.md`): the robot's own state, a noisy force/torque sensor at the flange and two RGB-D cameras (an
overhead camera that sees the whole bench and a wrist camera); no unit poses, no mate status. Per episode: 1200 s of
simulated time, 2400 s of wall clock (camera rendering counts), 120 s of silence at most (counted from the program's
start or the cell's reply to a request until the program's next request: the cell's own time inside a request does not
count, and any request, `get_state()` being the cheapest, resets it), 600 camera frames in total.

How the program is run: a fresh process under a dedicated unprivileged account that is not yours, from a per-run
directory with a read-only copy of your submission folder (regular files only), an empty writable HOME / TMPDIR, a
minimal environment (`PATH`, `HOME`, `TMPDIR`, `LANG`, `MUJOCO_GL`, `PYTHONPATH` = the given library `~/given/lib`,
`ARB_GIVEN_DIR` = a read-only copy of `~/given`) and the launcher `python3 -B -E -s -c <bootstrap> <lib> assemble` (or
`bash run.sh`). Resolve given files through `os.environ.get("ARB_GIVEN_DIR", os.path.expanduser("~/given"))`, write only
under the run's HOME / TMPDIR. A program that crashes ends that episode with what it achieved; the other episodes still
run. Your submission must be final when your session ends: at verify start every process of your account is stopped
and killed, the submission is snapshotted root-owned, a submission entry written after that time voids the contract
gate, and a surviving process of yours or of the run account voids it too.

## What you are given

| path | content |
|---|---|
| `~/given/kit.json` | the packing list: the units `body`, `shoulder_plain`, `shoulder_adapter`, `support`, `upper_leg`, `lower_leg` with role, quantity, mass to 1 g (a kitchen scale), centre of mass (from the CAD), size, material; the actuator module |
| `~/given/parts/*.stl` | every unit as delivered, millimetres, resting on its most likely stable face on z = 0, its footprint's smallest enclosing rectangle centred on the origin with its long side along x; visual tessellation |
| `~/given/actuator/` | the actuator module (every hip ab/ad, hip flexion and knee joint is one): datasheet and dimensioned drawing (PNG), its STEP model and its keep-out envelope (STEP) |
| `~/given/manual/photo_01.jpg` .. `photo_25.jpg`, `photos.json` | the picture manual: the kit, an exploded view, the body on its stand, one leg's four steps before / after, the finished robot at its zero pose from several sides, close-ups; `photos.json` lists per photo the caption, the size, a focal-length class and the scale references in view with their nominal sizes; the camera poses and the lens are not known |
| `~/given/cell/cell_manual.md` | the integrator's cell description: bench, robot base pose (exact), the three datum checkerboards (their exact poses), the stand, how the units lie, cameras with resolution, nominal field of view and approximate mounts, depth-sensor and force-sensor noise figures, the gripper, the fastening rules, the unpowered joints, safety rules, budgets |
| `~/given/cell/cell_api.md` | the cell's programming manual (the wire API) |
| `~/given/robot/` | the robot's MuJoCo models: `universal_robots_ur10e/` (BSD-3-Clause) and `robotiq_2f85/` (BSD-2-Clause), with their licences |
| `~/given/lib/` | the cell's client and helper code: `arm.assembly_info.client` (the proxy, `decode_frame`), `arm.assembly_quad` (the cell's own arm model builder `ur_robot`, its controller `ur_controller` and the gripper model `gripper_2f85`), `arm.assembly_cell` / `arm.assembly_six` (physics building blocks of an earlier cell), `render` (the camera rendering layer) |
| `~/render/` | the scenery of the cell cameras' renderer (the factory hall as meshes and baked textures, the bench top's scanned wood maps); nothing about the robot being built |

Tools: Python 3.12 (`/opt/venv`, on PATH) with numpy, scipy, OpenCV (`cv2`, headless), scikit-image, MuJoCo 3.14,
trimesh, manifold3d, Pillow, matplotlib. No network in your session.

## How you are graded (verify runs 3 hidden episodes)

Gates (multiply the score): `contract_ok` = `assemble.py` is a regular file that parses and defines `solve`, passes the
route gate (no weight files, no training calls, submission <= 60 MB), imports under the run regime above, makes at least
one `step` call over the episodes, causes no network activity during any episode (the grader reads the container's
outbound network counters before and after each episode and samples the program's sockets), and your own `.py` files
import no networking module (`socket`, `ssl`, `urllib`, `http`, `requests`, `asyncio`, `ctypes`, `subprocess`, ...),
load no module under a computed name, execute no computed code (`exec` / `eval` / `compile` of a non-literal) and start
no other program (`os.system`, `os.exec*`, `os.spawn*`, ...), and nothing of the submission was written after your
session ended. The code check parses your `.py` files (`~/given/lib/arm/assembly_info/netaudit.py`) and counts only the
built-ins `exec(...)`, `eval(...)` and `compile(...)` called by their bare names with a non-literal argument, so a
method of the same name, such as MuJoCo's `spec.compile()`, is fine. Per episode `safety`: a robot link other than the
gripper's four pad boxes pressing more than 30 N on the stand, the body, the bench or a unit, a unit's centre of mass
faster than 3 m/s, or a unit pressed on the environment or on another unit above 150 N for 0.2 s zeroes that episode.

`assembly_score` = the mean over the episodes of the robot's completed fraction: the weight of the slots the cell has
fastened (each shoulder on its hip ab/ad output 1.5, counted only in its design key orientation; each support 1.0; each
leg on its hip flexion output 1.5; each lower leg on its knee output 1.5) over the robot's total weight 22.0.
`overall` = `assembly_score` x the gates. Stats (not scored): per-slot rates, shoulders fastened the other way round,
frames used per camera, sim / wall time, illegal calls, safety events, dropped units, network counters.

## Rules of the cell (details in `cell_manual.md` / `cell_api.md`)

* Every joint of the robot being built is one actuator module whose output pulley has a rectangular socket; a unit's
  key goes into it (the key fits two ways, 180 deg apart). A key is captured once at least a quarter of its height is
  inside the socket (the lead-in counts), its axis within about 2 mm and 6 deg of the socket's and its turn about the
  axis within 6 deg of a key orientation; it gets that far only from close to its socket: pushed in by the arm at
  stiffness 0.5, a shoulder's key from about 0.3-0.4 mm across the socket and 1 deg of turn, an upper leg's from about
  1.5 mm and 3 deg (`cell_manual.md` has the measured window). Each output pulley starts up to about 2 deg off its
  index, so measure or search the pulley's turn. The cell then holds the unit there and pulls it home (the screws
  being driven, about 10 mm/s), and fastens it when it rests at home for 50 consecutive 10 ms ticks. A support is
  captured on the body up to 8 mm short of home over its shoulder's journal; it needs its shoulder fastened first. A
  unit may still be held while it fastens. The cell never tells you about any fastening.
* Sub-assemblies built on the bench may be mounted later: a lower leg fastened on its upper leg goes on with it.
* The actuators are unpowered: every joint of the robot being built turns against about 0.10 N m of friction, so
  whatever pushes the robot moves it, and a leg hanging from its joint swings when it is bumped.
* The keys and the sockets collide as drawn: 0.3 mm of play per side, and the drawing's R1.2 lead-in modelled as a
  45 deg x 1.2 mm chamfer on both faces of every output pulley; see `cell_manual.md`, "Collision model".
* Calls other than the documented public ones answer `{"error": "forbidden"}` and are counted.

## No practice

There is no practice cell, no answer key and no checker: everything known about this robot is in its kit files, the
actuator documents and its manual, and the only run of your program on the cell is the graded one.

## Notes

- Photos and camera frames are meant to be processed programmatically; view only the images you need.
- The cell frame, the robot base pose and the three datum boards' poses are exact; everything else about the cameras,
  the stand and the units' poses must be measured in each episode.
- Do not modify the given files in your home directory; the grader reads its own copies.
- Environment facts (harness limits): a single tool call whose input runs above roughly 25-30k tokens is dropped by the
  harness (it is never executed and no error comes back), and a response that reaches the 64k output-token limit is lost
  and, after at most a few retries, ends the session with whatever is in `/home/user/submission/` at that moment. Write
  large files in several parts (one function group per write), keep every single response well below that limit, write
  a first complete submission early and refine it.
- Session time: this session is stopped after 6 hours of wall-clock time from its start. A session stopped by that limit
  is graded on whatever the submission folder holds at that moment, so a file left half-written then is what runs. Note
  the start time with `date` when you begin, keep a complete submission in place from early on, and end the session
  yourself (stop working and say that you are done) at least 15 minutes before the limit.
