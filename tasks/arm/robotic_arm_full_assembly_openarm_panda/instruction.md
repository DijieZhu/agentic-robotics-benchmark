# Assemble both arms of an OpenArm pair from their loose units with two Pandas, guided by its picture manual and the cell's cameras

You are the assembly engineer of a robot-arm shop. The shop builds the OpenArm 2.0, a pair of 7-DoF arms hanging from
a gallows. The gallows stands on the bench with the two J1 motors on top of it; everything else of the two arms lies
loose on the bench: per arm nine units (the J1 link, the J2 motor, the J2 link, the upper arm with the J3 and J4 motors,
the forearm with the J5 motor, two cable holders that go over J4, the wrist with J6 and the J7 motor, and the gripper).
You have what an assembly engineer really has: every unit's CAD (its look and the exact shapes the cell simulates for
it), the arm's design (where every unit sits in the assembled arm, its joints and interfaces), the shop's picture manual
of the build (27 photographs), and a workcell with two Franka Pandas that you see only through its cameras. Today's job
is the whole pair: program the cell's two Pandas so that they build both arms the way the manual shows: every unit onto
its motor's output flange in the right rotation (the flanges' index marks), the J2 motor into its yoke, the cable
holders over J4.

Nothing in the cell tells you where anything is: no part pose, no camera calibration, no "fastened" signal. The units
lie in different places and orientations in every run, the gallows stands a little differently, and the cell's cameras
are mounted and calibrated differently in every run.

## What you produce (in `/home/user/submission/`)

`assemble.py` exposing `solve(cell) -> None` (plus any helper modules, optional `run.sh`). The grader runs it once per
hidden episode (3 episodes) in its own physics cell. It talks to the cell through `cell` (API in
`~/given/cell/cell_api.md`): the two robots' own states, their noisy force/torque sensors and three RGB-D cameras (an
overhead camera over the whole bench and one camera on each Panda's hand); no part poses, no mate status. Per episode:
1800 s of simulated time, 3000 s of wall clock (camera rendering counts), 120 s of silence at most between two requests,
900 camera frames in total.

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
| `~/given/kit.json` | the kit: the 18 units (role, arm, mass, centre of mass, files, their inner bodies: the motors' output flanges (rotors) and other hinged parts, with axes and index marks), the arms' joints (axes, limits, motors, the unpowered friction), the design (every unit's pose in the pair frame at the zero posture, the pair frame on the gallows), the mates (carrier and part, the interface frames, the insertion direction, tolerances, weights, prerequisites), the fastening rules, the riser blocks |
| `~/given/parts/<unit>.stl` | each unit's look, metres, in the unit's own frame (the frame `kit.json` uses for it) |
| `~/given/collision/<unit>/*.stl` | the convex pieces the cell simulates for each unit, in the same frame: these, not the look, are what touches |
| `~/given/manual/photo_01.jpg` .. `photo_27.jpg`, `photos.json` | the picture manual: the kit on the bench, each arm's units, the nine steps of the right arm before / after (the left arm mirrors it), close-ups of J4_D, the cable holders and the flanges' index marks, the finished pair; `photos.json` lists per photo the caption, the size, a focal-length class and the scale references in view; the camera poses and the lens are not known |
| `~/given/cell/cell_manual.md` | the integrator's cell description: the bench, the two robots' base poses (exact), the datum checkerboards (their exact poses), the gallows, how the units lie, the cameras (resolution, nominal field of view, approximate mounts), depth-sensor and force-sensor noise figures, the grippers, the fastening rules, the powered and unpowered joints, safety rules, budgets |
| `~/given/cell/cell_api.md` | the cell's programming manual (the wire API) |
| `~/given/panda/` | the robots' MuJoCo model (Apache-2.0) |
| `~/given/lib/` | the cell's client code: `arm.assembly_openarm.client` (the proxy `PairCell`, `decode_frame`), `arm.assembly_cell.controller` (the Panda controller the cell runs), `render` (the camera rendering layer with the camera model's post-processing code; it renders like the cell's cameras: `arm.assembly_openarm.agent_render` is the cell cameras' own render session for your twin of the cell, `render.filament.view` renders any scene of yours the same way; MuJoCo's own renderer, `mujoco.Renderer` or the layer's `"mujoco"` backend, draws a different picture than the cell's cameras (no hall, MuJoCo's own lights, flat colours, no reflections), so image processing tuned on its frames may behave differently on the cell's frames, and the given layer is the way to see what the cell's cameras would see; `cell_api.md`, "Rendering your own frames") |
| `~/render/` | the scenery of the cell cameras' renderer, which the given render layer uses too (the factory hall as meshes and baked textures, the bench top's scanned wood maps); nothing about the arms |

Tools: Python 3.12 (`/opt/venv`, on PATH) with numpy, scipy, OpenCV (`cv2`, headless), scikit-image, MuJoCo 3.17,
trimesh, manifold3d, Pillow, matplotlib. No network in your session.

## How you are graded (verify runs 3 hidden episodes)

Gates (multiply the score): `contract_ok` = `assemble.py` is a regular file that parses and defines `solve`, passes the
route gate (no weight files, no training calls, submission <= 60 MB), imports under the run regime above, makes at least
one `step` call over the episodes, causes no network activity during any episode (the grader reads the container's
outbound network counters before and after each episode and samples the program's sockets), and your own `.py` files
import no networking module (`socket`, `ssl`, `urllib`, `http`, `requests`, `asyncio`, `ctypes`, `subprocess`, ...),
load no module under a computed name, execute no computed code (`exec` / `eval` / `compile` of a non-literal) and start
no other program (`os.system`, `os.exec*`, `os.spawn*`, ...). Per episode `safety`: a Panda link other than its fingertip pads pressing more than 30 N on the gallows,
the bench, a unit or the other Panda, a unit faster than 3 m/s, or a unit pressed on the environment or on another unit
above 150 N for 0.2 s zeroes that episode.

`assembly_score` = the mean over the episodes of the pair's completed fraction: the weight of the mates the cell has
fastened over the pair's total weight 22.0 (per arm: each of the six flanges, J1-J5 and J7, 1.5; the J2 motor's seat
1.0; each cable holder 0.5; a flange counts only when it is bolted in its design rotation). `overall` = `assembly_score`
x the gates. Stats (not scored): per-mate rates, flanges bolted off the design rotation, frames used per camera, sim /
wall time, illegal calls, safety events, dropped units, network counters.

## Rules of the cell (details in `cell_manual.md` / `cell_api.md`)

* A unit on its feature is held there by the cell (within 4 mm across and 8 deg; a flange also within 4 mm along its
  axis, its rotation about the axis left free; the J2 motor at most 3 mm short of home in its yoke), so it stays on when
  it is let go; it fastens when it rests within its tolerance (1.5 mm / 3 deg; a flange's faces within 1.5 mm) for 50
  consecutive 10 ms ticks, moving slower than 5 mm/s and 2 deg/s relative to its carrier. It may still be held while it
  fastens. A flange is bolted at the nearest of its bolt positions (every 60 deg about the axis) and counts only on its
  design position (the manual's index marks). The forearm fastens onto J4's flange and J4_D at once: J4_D must be within
  the tolerance too. The cell never tells you about any fastening.
* J1, J2 and J3 of each arm hold the angle they were fastened at while the arm is built (position-held motors, up to
  10 / 10 / 5 N m). J4-J7 and J4_D's bearing are unpowered: they turn against their back-drive friction (J4 about
  0.1 N m, J4_D's bearing about 0.02 N m), so whatever pushes them moves them.
* Every unit collides as its given convex pieces (`~/given/collision/`); the given looks are drawn only.
* Calls other than the documented public ones answer `{"error": "forbidden"}` and are counted.

## No practice

There is no practice cell, no answer key and no checker: everything known about the pair is in its kit files and its
manual, and the only run of your program on the cell is the graded one.

## Notes

- Photos and camera frames are meant to be processed programmatically; view only the images you need.
- The cell frame, the robots' base poses and the datum boards' poses are exact; everything else about the cameras, the
  gallows and the units' poses must be measured in each episode.
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
