# Assemble a tenth-scale autonomous race car from its rolling chassis and eight loose units with a 6-DoF arm, guided by its picture manual, its drawings and the cell's cameras

You are the assembly engineer of a small autonomous-racing workshop. The workshop's car is a tenth-scale,
four-wheel-drive touring-class car with an autonomy deck. Its rolling chassis (the chassis tub with drivetrain,
suspension, steering and the kit's own electronics, four deck standoffs standing on it, four bare wheel hubs) is
clamped on a work stand in a robot workcell. What is still missing lies ready in the cell: the NiMH stick pack, the
deck assembly (the deck plate with the antenna mount, the motor controller and the power distribution board already on
it), the compute module, the 2D scanning LiDAR and four wheels. You have what an assembly engineer really has: the
rolling chassis and every unit as an STL file, a packing list, five drawings with a few dimensions each (images), the
workshop's picture manual of this car (14 photographs with scale references in view, captions without dimensions) and
a robot workcell that you see only through its cameras. Today's job is the whole car: program the cell's 6-DoF arm and
its parallel gripper so that it builds the car the way the manual shows: the stick pack in its bay, the deck assembly
on its four standoffs, the compute module and the LiDAR on the deck the right way round, and a wheel on every hub.

Nothing gives the answer directly: there is no mating graph, no design pose beyond the few dimensions on the drawings,
no insertion axis, no camera calibration, no unit pose, no hub angle and no "fastened" signal. In every run the work
stand, the deck shelf and the parts tray stand slightly differently, the wheels and the stick pack lie in different
places and turned differently on the bench, every wheel hub starts at its own random angle, and the cell's cameras are
mounted and calibrated differently.

## What you produce (in `/home/user/submission/`)

`assemble.py` exposing `solve(cell) -> None` (plus any helper modules, data files, optional `run.sh`). The grader runs
it once per hidden episode (4 episodes) in its own physics cell. It talks to the cell through `cell` (API in
`~/given/cell/cell_api.md`): the robot's own state, a noisy force/torque sensor at the flange and two RGB-D cameras
(an overhead camera that sees the whole bench and a wrist camera); no unit poses, no hub angles, nothing about what is
fastened. Per episode: 650 s of simulated time, 3100 s of wall clock (camera rendering counts), 120 s of silence
at most (counted from the program's start or the cell's reply to a request until the program's next request: the
cell's own time inside a request does not count, and any request, `get_state()` being the cheapest, resets it), 450
camera frames in total.

How the program is run: a fresh process under a dedicated unprivileged account that is not yours, from a per-run
directory with a read-only copy of your submission folder (regular files only), an empty writable HOME / TMPDIR, a
minimal environment (`PATH`, `HOME`, `TMPDIR`, `LANG`, `MUJOCO_GL`, `PYTHONPATH` = the given library `~/given/lib`,
`ARB_GIVEN_DIR` = a read-only copy of `~/given`) and the launcher `python3 -B -E -s -c <bootstrap> <lib> assemble` (or
`bash run.sh`). The process cannot open an Internet socket. Resolve given files through
`os.environ.get("ARB_GIVEN_DIR", os.path.expanduser("~/given"))`, write only under the run's HOME / TMPDIR. A program
that crashes ends that episode with what it achieved; the other episodes still run. Your submission must be final when
your session ends: at verify start every process of your account is stopped and killed, the submission is snapshotted
root-owned, a submission entry written after that time voids the contract gate, and a surviving process of yours or of
the run account voids it too.

## What you are given

| path | content |
|---|---|
| `~/given/kit.json` | the packing list: the loose units `wheel` (four), `battery_pack`, `deck_assembly`, `compute_module`, `lidar` and the rolling chassis `roller`, each with role, quantity, mass, centre of mass, size, material and how it is presented in the cell (loose on the bench, on the deck shelf, in a nest of the parts tray); an index of the drawings and of the manual; the car model's appearance (appearance only): each material's base colour, metallic and roughness, and the material of every triangle of the part files, so that a twin built from the part files renders the car as the cell's cameras do (`cell_api.md`, "Rendering your own frames"). It lists no mates and no poses |
| `~/given/appearance/` | the images of the car model's printed faces (the deck's boards), each with its textured mesh: that material's triangles of the part file, with texture coordinates (`kit.json`, `materials`) |
| `~/given/parts/*.stl` | six files in millimetres. Each kind of loose unit in the attitude in which it is presented in the cell (the wheel on its inner face, the stick pack lying flat, the deck plate level with its top face up, the compute module and the LiDAR upright), its lowest point on z = 0, its footprint's smallest enclosing rectangle centred on the origin with its long side along x. `roller.stl`: the rolling chassis with its deck standoffs and its wheel hubs, in the car's own frame. The files show the car as it is drawn; the cell's collision shapes differ from them as `cell_manual.md` lists |
| `~/given/drawings/*.png` | five drawings, as images: `assembly_layout.png` (the deck on its four standoffs; where the compute module and the LiDAR stand on the deck), `hub_and_wheel.png` (wheel hex, rim pocket, the four axles' directions, the seat and stop that the cell simulates), `battery_bay.png`, `tray.png`, `stand_and_shelf.png` (the fixtures, their datum boards and what sits where on them). The printed dimensions are exact; there are few of them |
| `~/given/manual/photo_01.jpg` .. `photo_14.jpg`, `photos.json` | the picture manual: the kit, the rolling chassis on its stand, the car after each build step, a wheel hub close up, the finished car from four sides with a checkerboard card beside it, a close-up of the deck. `photos.json` lists per photo the caption, the size, a focal-length class and the scale references in view with their nominal sizes; the camera poses and the lens are not known. The finished car in the photographs carries a few parts that this station does not fit; the captions name them |
| `~/given/cell/cell_manual.md` | the integrator's cell description: bench, robot base pose (exact), the five datum boards, the fixtures' nominal places and how much they and the loose units vary, cameras with resolution, nominal field of view and approximate mounts, depth-sensor and force-sensor noise figures, the gripper, the fastening rules with every window and the entry behaviour measured in the cell, every difference between the cell's collision shapes and the drawn car, what the stick pack does in its bay and what a wheel does on a hub, the unpowered hubs, safety rules, budgets |
| `~/given/cell/cell_api.md`, `cell_config.json` | the cell's programming manual (the wire API); the cell's configuration as far as it is public |
| `~/given/robot/` | the robot's MuJoCo models: `universal_robots_ur10e/` (BSD-3-Clause) and `robotiq_2f85/` (BSD-2-Clause), with their licences |
| `~/given/lib/` | the cell's client and helper code: `arm.assembly_info.client` (the proxy, `decode_frame`), `arm.assembly_quad` (the cell's own arm model builder `ur_robot`, its controller `ur_controller` and the gripper model `gripper_2f85`; the package is named after the cell it was first built for), `arm.assembly_cell` / `arm.assembly_six` (physics building blocks of an earlier cell), `render` (the camera rendering layer; it renders like the cell's cameras: `arm.assembly_car.agent_render` is the cell cameras' own render session for your twin of the cell, `render.filament.view` renders any scene of yours the same way; MuJoCo's own renderer, `mujoco.Renderer` or the layer's `"mujoco"` backend, draws a different picture than the cell's cameras (no hall, MuJoCo's own lights, flat colours, no reflections), so image processing tuned on its frames may behave differently on the cell's frames, and the given layer is the way to see what the cell's cameras would see; `cell_api.md`, "Rendering your own frames") |
| `~/render/` | the scenery of the cell cameras' renderer, which the given render layer uses too (the factory hall as meshes and baked textures, the bench top's scanned wood maps); nothing about the car |

Tools: Python 3.12 (`/opt/venv`, on PATH) with numpy, scipy, OpenCV 5.0 (`cv2`, headless; this build has no hand-eye
calibration: `cv2.calibrateHandEye` and `cv2.calibrateRobotWorldHandEye` are absent), scikit-image, MuJoCo 3.14, mink,
Pinocchio (`pinocchio`), CoACD (`coacd`), CadQuery (`cadquery`), trimesh, manifold3d, Pillow, matplotlib. No network
in your session.

## How you are graded (verify runs 4 hidden episodes)

Gates (multiply the score; a failed gate makes the whole run score 0): `contract_ok` = `assemble.py` is a regular file
that parses and defines `solve`, passes the route gate (no weight files: `.pt`, `.pth`, `.onnx`, `.safetensors`,
`.ckpt`, `.h5`, `.pb`, `.pkl`, `.npz`; no training calls; submission <= 60 MB), imports under the run regime above,
makes at least one `step` call over the episodes, causes no network activity during any episode (the grader reads the
container's outbound network counters before and after each episode and samples the program's sockets), and your own
`.py` files import no networking module (`socket`, `ssl`, `urllib`, `http`, `requests`, `asyncio`, `ctypes`,
`subprocess`, ...), load no module under a computed name, execute no computed code (`exec` / `eval` / `compile` of a
non-literal) and start no other program (`os.system`, `os.exec*`, `os.spawn*`, ...), and nothing of the submission was
written after your session ended. The code check is static: it parses your `.py` files
(`~/given/lib/arm/assembly_info/netaudit.py`) and finds those imports and calls by their names; of `exec(...)`,
`eval(...)` and `compile(...)` it counts only the built-ins called by their bare names with a non-literal argument, so
a method of the same name, such as MuJoCo's `spec.compile()`, is fine.

Per episode, two things are safety violations: a robot link other than the gripper's four pad boxes pressing with more
than 150 N on the chassis, a fixture, the bench or a unit; and a unit pressed on the environment or on another unit
with more than 150 N for 0.2 s. A safety violation does not end the episode, but an episode with one keeps only
30 % of its score. One violation or several: the episode keeps the same 30 %. A dropped unit is not a safety
violation: a loose unit whose centre of mass moves faster than 3 m/s, or that falls off the bench, is scrapped. It
can no longer be fastened and no longer counts; the episode goes on with the other units. If the scrapped unit is the
deck assembly, the compute module and the LiDAR fastened on it stop counting with it.

`assembly_score` = the mean over the episodes of the car's completed fraction: the weight of what the cell has
fastened at the end of the episode over the car's total weight 11.5, kept at 30 % in an episode with a safety
violation. The weights: the stick pack in its bay 1.0; the deck assembly on its standoffs 2.0; the compute module
on the deck 1.0; the LiDAR on the deck 1.5, counted only in the orientation the manual shows; each wheel on a hub
1.5. The reward is `assembly_score` x the gate. Stats (not scored): per-place rates, a LiDAR fastened in another
orientation, scrapped units, what stopped counting with a scrapped deck, frames used per camera, sim / wall time,
illegal calls, safety events, the largest sustained force, network counters.

## Rules of the cell (details in `cell_manual.md` / `cell_api.md`)

* **Fastening.** A unit is captured once it is inside the capture window of its place: a few millimetres and degrees
  (across: the stick pack 3.2 mm, the deck 3.0 mm, the compute module and the LiDAR 2.5 mm, a wheel 2.0 mm;
  `cell_manual.md` has every window and what was measured). The cell then holds the unit there and pulls it home (the
  screws or the wheel nut being driven, at most 10 mm/s), and fastens it when it rests at home for 50 consecutive
  10 ms ticks. A unit may still be held while it fastens. The cell never tells you about any fastening, and nothing
  can be unfastened.
* **Orientation.** The stick pack, the deck assembly and the compute module are captured only the way round the manual
  shows. The LiDAR is captured and fastened in four orientations a quarter turn apart; only the one the manual shows
  counts, and in the other three it is fastened for good and counts nothing. A wheel counts in every orientation of
  the hex, and any wheel fits any hub.
* **Sub-assemblies.** The compute module and the LiDAR mount on the deck plate wherever the deck assembly is. On a
  deck that is still on its shelf they count as on the car, as long as the deck assembly is not scrapped by the end of
  the episode. No place needs another one fastened first: the order is yours, within what the car physically allows
  and what the fixtures beside the car leave room for.
* **The wheel hubs are unpowered.** Each hub turns freely against about 0.04 N m of friction and starts every run at
  its own random angle. A wheel pushed onto a hub stops 2.50 mm short of home unless the turn of its hex pocket is
  within 6 deg of an orientation of the hex; pushing harder does not help, and from about 6 N of push the hub
  turns with the wheel. Measure the turn or search for it.
* **The stick pack drops into its bay.** The gripper cannot follow it down between the bay's walls: it has to be let
  go above the bay, 22 mm or more above its seat (`cell_manual.md` has what was measured).
* **The cell's collision shapes are not the drawn shapes.** The part files show a display model whose parts touch or
  pass through each other. The cell simulates simpler shapes with material taken away: fasteners, leads and plugs do
  not collide, 9 parts are cut back for clearance, the deck standoffs end under the deck, the rim's hex pocket is a
  round seat with a stop. `cell_manual.md`, "Collision model", lists every difference with its size.
* Calls other than the documented public ones answer `{"error": "forbidden"}` and are counted.

## No practice

There is no practice cell, no answer key and no checker: everything known about this car is in its kit files, its
drawings and its manual, and the only run of your program on the cell is the graded one.

## Notes

- Photos, drawings and camera frames are meant to be processed programmatically; view only the images you need.
- The cell frame, the robot base pose and the bench board's pose are exact, the tool plate's within its mounting
  tolerance; everything else about the cameras, the fixtures, the units' poses and the hubs' angles must be measured
  in each episode.
- Do not modify the given files in your home directory; the grader reads its own copies.
- Machine: your session's sandbox has 8 CPUs and 24 GB of memory, no GPU and no network. The grading machine has
  the same size, and there your program shares it with the cell's physics and its camera renderer, which take about
  7 GB between them.
- Environment facts (harness limits): a single tool call whose input runs above roughly 25-30k tokens is dropped by
  the harness (it is never executed and no error comes back), and a response that reaches the 64k output-token limit
  is lost and, after at most a few retries, ends the session with whatever is in `/home/user/submission/` at that
  moment. Write large files in several parts (one function group per write), keep every single response well below
  that limit, write a first complete submission early and refine it.
- Session time: this session is stopped after 6 hours of wall-clock time from its start. A session stopped by that
  limit is graded on whatever the submission folder holds at that moment, so a file left half-written then is what
  runs. Note the start time with `date` when you begin, keep a complete submission in place from early on, and end the
  session yourself (stop working and say that you are done) at least 15 minutes before the limit.
