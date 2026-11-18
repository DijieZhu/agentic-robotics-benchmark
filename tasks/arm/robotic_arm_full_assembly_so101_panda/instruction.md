# Assemble a six-servo desktop arm from its loose parts with a Panda, guided by its picture manual and the cell's cameras

You are the assembly engineer of a small robot-arm shop. The shop has printed the ten plastic parts of a six-servo
desktop arm (one servo per joint: base yaw, shoulder pitch, elbow pitch, wrist pitch, wrist roll, gripper jaw) and
bought its six bus servos. You have what an assembly engineer really has: the printed parts as the STL files that went
to the printer (each in its own print orientation), the servo's CAD file and datasheet, the shop's picture manual of
this arm (25 photographs with scale references in view, captions without dimensions) and a robot workcell that you see
only through its cameras. Only the arm's base is screwed on the cell's fixture; the other fifteen parts lie loose on the
bench. Today's job is the whole arm: program the cell's Franka Panda so that it builds the arm from the base, the way
the manual shows: every servo into its pocket the right way round, every bracket clamped over its servo, every link
mounted on its servo's horn in the orientation the manual shows for the arm's calibration pose.

Nothing gives the answer directly: there is no mating graph, no design pose, no insertion axis, no camera calibration,
no part pose and no "seated" signal. The arm is a modified build (it differs from any arm you may know), the parts lie
in different places and orientations in every run, and the cell's cameras are mounted and calibrated differently in
every run.

## What you produce (in `/home/user/submission/`)

`assemble.py` exposing `solve(cell) -> None` (plus any helper modules, optional `run.sh`). The grader runs it once per
hidden episode (3 episodes) in its own physics cell. It talks to the cell through `cell` (API in
`~/given/cell/cell_api.md`): the robot's own state, a noisy force/torque sensor and two RGB-D cameras (an overhead camera that sees the whole bench and a wrist camera); no part poses,
no mate status. Per episode: 1200 s of simulated time, 2400 s of wall clock (camera rendering counts), 120 s of silence
at most between two requests, 600 camera frames in total.

How the program is run: a fresh process under a dedicated unprivileged account that is not yours, from a per-run
directory with a read-only copy of your submission folder (regular files only), an empty writable HOME / TMPDIR, a
minimal environment (`PATH`, `HOME`, `TMPDIR`, `LANG`, `MUJOCO_GL`, `PYTHONPATH` = the given library `~/given/lib`,
`ARB_GIVEN_DIR` = a read-only copy of `~/given`) and the launcher `python3 -B -E -s -c <bootstrap> <lib> assemble` (or
`bash run.sh`). Resolve given files through `os.environ.get("ARB_GIVEN_DIR", os.path.expanduser("~/given"))`, write only
under the run's HOME / TMPDIR. A program that crashes ends that episode with what it achieved; the other episodes still
run. Your submission must be final when your session ends: the submission folder is then copied out of your sandbox
and graded in a fresh sandbox that holds nothing else of yours (no other file, no process you started), and a
surviving process of the run account voids the contract gate.

## What you are given

| path | content |
|---|---|
| `~/given/kit.json` | the packing list: parts `P01`..`P10` with role names, quantity, mass to 1 g (a kitchen scale), material; the servos |
| `~/given/parts/P01.stl` .. `P10.stl` | the printed parts, millimetres, each in its PRINT frame (largest flat face down on z = 0, centred in x / y); visual tessellation |
| `~/given/parts/servo.stl`, `servo_hornless.stl` | the bus servo as the vendor's CAD, in the servo's own frame (the datasheet defines it) |
| `~/given/servo/` | the servo datasheet (`servo_datasheet.md`, datasheet and dimensioned-drawing pages as PNG) |
| `~/given/manual/photo_01.jpg` .. `photo_25.jpg`, `photos.json` | the picture manual: the kit, an exploded view, the seven assembly steps before / after, the finished arm in its calibration pose (every servo at its zero bus reading) and in another pose, a close-up of the elbow; `photos.json` lists per photo the caption, the size, a focal-length class and the scale references in view with their nominal sizes; the camera poses and the lens are not known |
| `~/given/cell/cell_manual.md` | the integrator's cell description: bench, robot base pose (exact), the three datum checkerboards (their exact poses), the fixture, how the parts lie, cameras with resolution, nominal field of view and approximate mounts, depth-sensor and force-sensor noise figures, gripper, fastening rules, the unpowered servos, safety rules, budgets |
| `~/given/cell/cell_api.md` | the cell's programming manual (the wire API) |
| `~/given/panda/` | the robot's MuJoCo model (Apache-2.0) |
| `~/given/lib/` | the cell's client and helper code: `arm.assembly_info.client` (the proxy, `decode_frame`), `arm.assembly_cell` / `arm.assembly_six` (the cell's physics building blocks, predicates, controller), `render` (the camera rendering layer; it renders like the cell's cameras: `arm.assembly_full.agent_render` is the cell cameras' own render session for your twin of the cell, `render.filament.view` renders any scene of yours the same way; MuJoCo's own renderer, `mujoco.Renderer` or the layer's `"mujoco"` backend, draws a different picture than the cell's cameras (no hall, MuJoCo's own lights, flat colours, no reflections), so image processing tuned on its frames may behave differently on the cell's frames, and the given layer is the way to see what the cell's cameras would see; `cell_api.md`, "Rendering your own frames") |
| `~/render/` | the scenery of the cell cameras' renderer, which the given render layer uses too (the factory hall as meshes and baked textures, the bench top's scanned wood maps); nothing about the arm |

Tools: Python 3.12 (`/opt/venv`, on PATH) with numpy, scipy, OpenCV (`cv2`, headless), scikit-image, MuJoCo 3.14,
trimesh, manifold3d, Pillow, matplotlib. No network in your session.

## How you are graded (verify runs 3 hidden episodes)

Gates (multiply the score): `contract_ok` = `assemble.py` is a regular file that parses and defines `solve`, passes the
route gate (no weight files, no training calls, submission <= 60 MB), imports under the run regime above, makes at least
one `step` call over the episodes, causes no network activity during any episode (the grader reads the container's
outbound network counters before and after each episode and samples the program's sockets), and your own `.py` files
import no networking module (`socket`, `ssl`, `urllib`, `http`, `requests`, `asyncio`, `ctypes`, `subprocess`, ...),
load no module under a computed name, execute no computed code (`exec` / `eval` / `compile` of a non-literal) and start
no other program (`os.system`, `os.exec*`, `os.spawn*`, ...). Per episode `safety`: a Panda link other than the
fingertip pads pressing more than 30 N on the fixture, the bench or a part, a part faster than 3 m/s, or a part pressed
on the environment or on another part above 150 N for 0.2 s zeroes that episode.

`assembly_score` = the mean over the episodes of the arm's completed fraction: the weight of the mates the cell has
fastened (each of the six servo seats and three brackets 1.0, each of the six horn mates 1.5; a horn mate counts only
when its link is fastened within 3 deg of the design rotation) over the arm's total weight 18.0. `overall` =
`assembly_score` x the gates. Stats (not scored): per-mate rates, fastened forks off the design rotation, frames used per
camera, sim / wall time, illegal calls, safety events, dropped parts, network counters.

## Rules of the cell (details in `cell_manual.md` / `cell_api.md`)

* A servo seat or a bracket is fastened when the part sits at its design pose in its carrier within 1.5 mm / 3 deg for
  50 consecutive 10 ms ticks while it moves slower than 5 mm/s relative to the carrier. A fork is fastened wherever it
  RESTS on a seated servo's horn (axis within 3 deg and 1.5 mm, axial offset <= 1 mm, moving slower than 5 mm/s and
  turning slower than 2 deg/s relative to the servo for 50 consecutive ticks), at the rotation about the horn axis it
  rests at; a fastened fork cannot be mounted again. A part may still be held while it fastens. The cell never tells you
  about any fastening.
* A part that is on its feature (a servo or bracket in its pocket at most 3 mm short of home, a fork on its horn within
  a few mm and degrees) is held there until it fastens; further out nothing holds a part. A bracket clamps a seated
  servo; a link goes on a seated servo's horn. Sub-assemblies built on the bench may be mounted later.
* The servos are unpowered: the joints of the half-built arm turn against about 0.4 N m of gearbox friction within the
  arm's joint limits, so whatever pushes the arm moves it.
* Each servo's collision shape is one box over its case footprint from the rear disc's face to the horn-side flange,
  slightly shrunk (a servo resting horn up stands about 3 mm higher than its case bottom would); the horn disc and the
  rear disc are visual only (`cell_manual.md`, "Collision model").
* Calls other than the documented public ones answer `{"error": "forbidden"}` and are counted.

## No practice

There is no practice cell, no answer key and no checker: everything known about this arm is in its kit files and its
manual, and the only run of your program on the cell is the graded one.

## Notes

- Photos and camera frames are meant to be processed programmatically; view only the images you need.
- The cell frame, the robot base pose and the three datum boards' poses are exact; everything else about the cameras,
  the fixture and the parts' poses must be measured in each episode.
- Do not modify the given files in your home directory; the grader reads its own copies.
- Environment facts (harness limits): a single tool call whose input runs above roughly 25-30k tokens is dropped by the
  harness (it is never executed and no error comes back), and a response that reaches the 64k output-token limit is lost
  and, after at most a few retries, ends the session with whatever is in `/home/user/submission/` at that moment. Write
  large files in several parts (one function group per write), keep every single response well below that limit, write
  a first complete submission early and refine it.
- Session time: this session is stopped after 6 hours of wall-clock time from its start. A session stopped by that limit
  is not graded at all, so a late finish scores nothing. Note the start time with `date` when you begin, keep a complete
  submission in place from early on, and end the session yourself (stop working and say that you are done) at least
  15 minutes before the limit.
