# Assemble the three struts of a small tensegrity robot from their modules with a Panda, guided by its picture manual and the cell's cameras

You are the assembly engineer of a small robotics lab that builds a rolling tensegrity robot: three identical struts,
later tied together with cables. A strut is an aluminium rod with an electronics bay at its middle, a motor module on
each side of the bay and an end cap with a silicone cover on each rod end. The lab has put the printed and bought parts
together into modules by hand: three strut cores (the bay's base half with its boards, on the rod), six motor modules,
three bay lids and six end caps with covers (two red, two green, two blue). You have what an assembly engineer really
has: every module as an STL file, the packing list, the bought parts' part numbers, the lab's picture manual of a strut
(25 photographs with scale references in view, captions without dimensions) and a robot workcell that you see only
through its cameras. The three strut cores are clamped side by side in the cell's cradle; the fifteen loose modules lie
in the nests of two parts trays. Today's job is all three struts: program the cell's Franka Panda so that it finishes
each strut the way the manual shows: a motor module on each side of the bay at its place along the rod and at its turn
about it, the lid on the bay, an end cap on each rod end, both caps of a strut with covers of one colour. Cables,
sensors and wires are not part of this job.

Nothing gives the answer directly: there is no mating graph, no design pose, no camera calibration, no module pose and
no "fastened" signal. The trays stand and are turned differently in every run, the modules lie in different nests, the
motor modules at turns about their axes that are not their mounting turn, the lids either way round, the cradle stands
slightly differently, and the cell's cameras are mounted and calibrated differently in every run.

## What you produce (in `/home/user/submission/`)

`assemble.py` exposing `solve(cell) -> None` (plus any helper modules, optional `run.sh`). The grader runs it once per
hidden episode (3 episodes) in its own physics cell. It talks to the cell through `cell` (API in
`~/given/cell/cell_api.md`): the robot's own state, a noisy force/torque sensor at the flange and two RGB-D cameras (an
overhead camera that sees the whole bench and a wrist camera); no module poses, no mate status. Per episode: 1200 s of
simulated time, 2400 s of wall clock (camera rendering counts), 120 s of silence at most between two requests, 600
camera frames in total.

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
| `~/given/kit.json` | the packing list: the module types `M0` strut core, `M1` motor module, `M2` bay lid, `M3` end cap with cover; per type the count, model file(s), nominal mass (design values, not weighed; the lid's is an estimate), centre of mass, size, material, cover colours, and the rule that fixes each model file's frame |
| `~/given/parts/*.stl` | the modules as built, millimetres, each in its own frame fixed by its own geometry (not its mounting frame): `M0_strut_core`, `M1a_motor_module` (with a small cable guide ring) and `M1b_motor_module` (without; one design, either fits either end), `M2_bay_lid`, `M3_end_cap`; visual tessellation, not watertight |
| `~/given/purchased/purchased_parts.md` | the bought parts inside the modules: makers and part numbers; no drawings |
| `~/given/manual/photo_01.jpg` .. `photo_25.jpg`, `photos.json` | the picture manual: the modules laid out, the strut taken apart along its rod, the four assembly steps before / after, the finished strut from several sides and along the rod, close-ups, three finished struts in the cradle; `photos.json` lists per photo the caption, the size, a focal-length class and the scale references in view with their nominal sizes; the camera poses and the lens are not known |
| `~/given/cell/cell_manual.md` | the integrator's cell description: bench, robot base pose (exact), the three datum checkerboards (their exact poses), the cradle and the strut cores in it, the trays and their nests, what changes from episode to episode, cameras with resolution, nominal field of view and approximate mounts, depth-sensor and force-sensor noise figures, the gripper, the fastening rules, the collision model, safety rules, budgets |
| `~/given/cell/cell_api.md`, `cell_config.json` | the cell's programming manual (the wire API); the cell's public configuration (limits, budgets, rule numbers) |
| `~/given/panda/` | the robot's MuJoCo model (Apache-2.0) |
| `~/given/lib/` | the cell's client and helper code: `arm.assembly_info.client` (the proxy, `decode_frame`), `arm.assembly_cell` (the arm's controller) and `arm.assembly_six` (physics building blocks of an earlier cell), `render` (the camera rendering layer) |
| `~/render/` | the scenery of the cell cameras' renderer (the laboratory as meshes and baked textures, the bench top's maps); nothing about the robot being built |

Tools: Python 3.12 (`/opt/venv`, on PATH) with numpy, scipy, OpenCV (`cv2`, headless), scikit-image, MuJoCo 3.14,
trimesh, manifold3d, Pillow, matplotlib. No network in your session.

## How you are graded (verify runs 3 hidden episodes)

Gates (multiply the score): `contract_ok` = `assemble.py` is a regular file that parses and defines `solve`, passes the
route gate (no weight files, no training calls, submission <= 60 MB), imports under the run regime above, makes at least
one `step` call over the episodes, causes no network activity during any episode (the grader reads the container's
outbound network counters before and after each episode and samples the program's sockets), and your own `.py` files
import no networking module (`socket`, `ssl`, `urllib`, `http`, `requests`, `asyncio`, `ctypes`, `subprocess`, ...),
load no module under a computed name, execute no computed code (`exec` / `eval` / `compile` of a non-literal) and start
no other program (`os.system`, `os.exec*`, `os.spawn*`, ...). Per episode `safety` (any of the four zeroes that
episode): the robot other than its fingertip pads pressing more than 15 N on the cradle, a tray, the bench, a module or
a strut core; a module pressed on its surroundings or on another module above 30 N for 0.2 s; a rod loaded across its
axis above 4.0 N for 0.2 s on either side of its bay; a module faster than 3 m/s.

`assembly_score` = the mean over the episodes of the completed fraction: the weight of the mates that the cell has
fastened and that count (each of the six motor modules 1.5, each of the three lids 1.0, each of the six end caps 1.0)
over the total weight 18.0. `overall` = `assembly_score` x the gates. A fastened module counts nothing in three cases: a
motor module fastened 1.5-3.0 mm from its design place along the rod; a motor module fastened more than 5 deg from its
design turn about the rod (the wrong way round is 180 deg off); a strut's second end cap whose cover colour differs
from its first.

## Rules of the cell (details in `cell_manual.md` / `cell_api.md`)

* A motor module or an end cap is taken onto a rod end once the mouth of its bore is within 2.0 mm of the rod's axis,
  its bore axis within 5 deg of the rod's and the rod end at least 3 mm inside the bore. The cell then pulls it towards
  the rod's axis: held, it slides along the rod and turns about it almost freely; released by the gripper, it tilts
  until its bore rests on the rod, whose friction holds it (its turn moves by up to about 30 deg as it settles).
  Released with less than about 40 mm of the rod in its bore, a module may tilt so far that the cell lets it go again
  (with less than about 13 mm it always does); it then falls off the rod if less than about 14 mm (an end cap) or 10 mm
  (a motor module) of the rod is inside. A motor module taken on the other way round is not let go in this way.
* It is fastened where it rests (slower than 5 mm/s and 2 deg/s relative to its strut for 50 consecutive 10 ms ticks; it
  may still be held), at the place and turn it has then, once a motor module is within 3.0 mm of its design place along
  the rod or an end cap within 1.5 mm of home (the rod end at the bottom of its bore). A fastened module cannot be moved
  or fitted again, and the cell never tells you about any fastening.
* An end cap is taken on only when its rod end's motor module is fastened and counts; a lid only when both motor modules
  of its strut are, within 4 mm across, 3 mm short of home and 8 deg of its seated orientation (not turned end for end);
  the cell then pulls the lid home and fastens it at rest. So a motor module that is fastened without counting also
  blocks its end cap and its strut's lid.
* An end cap's turn about the rod is free; a strut's first fastened end cap sets its colour; any strut may take any
  colour, and either kind of motor module fits either end of any rod.
* In the collision model a bore is a clear channel 7.2 mm in diameter (the rod is 3.175 mm): contact does not centre a
  module on the rod. A motor module's battery lead, the thin tubes around the bores and hardware enclosed in a module
  are drawn but do not collide (`cell_manual.md`, "Collision model").
* Calls other than the documented public ones answer `{"error": "forbidden"}` and are counted.

## Notes

- No practice: there is no practice cell, no answer key and no checker. Everything known about this robot is in its kit
  files and its manual, and the only run of your program on the cell is the graded one.
- Photos and camera frames are meant to be processed programmatically; view only the images you need. Do not modify
  the given files in your home directory; the grader reads its own copies.
- The cell frame, the robot base pose and the three datum boards' poses are exact; everything else about the cameras,
  the cradle, the trays and the modules' poses must be measured in each episode.
- Environment facts (harness limits): a single tool call whose input runs above roughly 25-30k tokens is dropped by the
  harness (it is never executed and no error comes back), and a response that reaches the 64k output-token limit is lost
  and, after at most a few retries, ends the session with whatever is in `/home/user/submission/` at that moment. Write
  large files in several parts (one function group per write), keep every single response well below that limit, write
  a first complete submission early and refine it.
- Session time: this session is stopped after 6 hours of wall-clock time from its start. A session stopped by that limit
  is graded on whatever the submission folder holds at that moment, so a file left half-written then is what runs. Note
  the start time with `date` when you begin, keep a complete submission in place from early on, and end the session
  yourself (stop working and say that you are done) at least 15 minutes before the limit.
