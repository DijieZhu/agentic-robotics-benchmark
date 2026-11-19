# Run the final-assembly station of an inspection-class ROV with a 6-DoF arm: ballast each vehicle for the water on its work order, then fit its electronics enclosure, its vertical thrusters and its thruster guards

You are the manufacturing engineer of a small-batch line that builds an inspection-class remotely operated vehicle
(ROV). At the line's station for propulsion, electronics and trim a 6-DoF arm with a wide parallel gripper works on
one vehicle at a time. The vehicle arrives clamped in a turnover positioner that stands on three load cells: the open
frame with its battery enclosure, its four horizontal ducted thrusters, its own gripper and its Doppler velocity log
already on it. With every vehicle comes a work order that names the water the vehicle is delivered to: a temperature
and a salinity. The station makes the vehicle ready for that water. It fastens ballast blocks under the bottom plate
so that the finished vehicle floats level and slightly light there, and it fits the electronics enclosure, the four
vertical thrusters and the two thruster guards. Buoyancy foam, fairings, the sonar head, the lights and the harness
follow at the next station.

You have what a manufacturing engineer really has: the vehicle as it arrives and every unit as an STL file, a packing
list, 7 drawings, the vehicle's trim sheet, a water-density chart and a propeller data sheet (all images), the shop's
picture manual (16 photographs with scale references in view), the integrator's description of the cell, and the
robot's models and control library. Today's job is the station's program: for whatever vehicle is on the fork and
whatever water its work order names, it ballasts the vehicle and fits the enclosure, the thrusters and the guards the
way the drawings and the manual show.

Nothing gives the answer directly. There is no ballast plan for the vehicle on the fork, no water density, no mass and
no centre of mass of that vehicle (every vehicle differs from the trim sheet's table within the range the sheet
states), no mating graph, no design pose beyond the dimensions on the drawings, no camera calibration, no unit pose,
no word on which thruster carries which propeller, and no "fastened" signal. In every run the vehicle and its work
order are different ones, the thruster tray and the racks stand slightly differently, the thrusters stand in other
nests of the tray with their propellers at other angles, the enclosure lies turned differently on its rack, and the
cell's cameras are mounted and calibrated differently.

## What you produce (in `/home/user/submission/`)

`assemble.py` exposing `solve(cell) -> None` (plus any helper modules, data files, optional `run.sh`). The grader runs
it once per hidden episode (4 episodes) in its own cell. It talks to the cell through `cell` (API in
`~/given/cell/cell_api.md`): the run's work order, the robot's own state, a noisy force/torque sensor at the flange,
the positioner's index position and the three load cells under it, and two RGB-D cameras (an overhead camera that sees
the whole bench and a wrist camera); no unit poses, nothing about what is fastened, nothing about the trim. Per
episode: 1200 s of simulated time, 2400 s of wall clock (camera rendering counts), 120 s of silence at most (counted
from the program's start or the cell's reply to a request until the program's next request: the cell's own time inside
a request does not count, and any request, `get_state()` being the cheapest, resets it), 600 camera frames in total, 4
turns of the positioner.

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
| `~/given/kit.json` | the packing list: the units `electronics_enclosure`, `thruster_a`, `thruster_b`, `thruster_guard`, `ballast_block` and the vehicle as it arrives, `base_vehicle`, each with file, quantity, mass, size, material and how it is presented in the cell; an index of the drawings, the documents and the manual. It lists no mates, no places and no poses, and it does not say which thruster file carries which propeller |
| `~/given/parts/*.stl` | 6 files in millimetres. `base_vehicle.stl`: the vehicle as it arrives from upstream, in the vehicle frame (x forward, y to port, z up). `electronics_enclosure.stl`: x along the tube axis, +x towards the dome. `thruster_a.stl`, `thruster_b.stl`: the two kinds of vertical thruster, both ducted thrusters, in one common frame (origin on the duct's axis, level with the middle of the mounting holes; +z from the nose cone to the outlet; the mounting face on the -y side). `thruster_guard.stl`, `ballast_block.stl`. The files show the vehicle as it is drawn; the cell's collision shapes differ from them as `cell_manual.md` lists |
| `~/given/drawings/*.png` | 7 drawings, as images: `general_arrangement.png` (three views, the station datums, where the units of this station sit), `keel_hole_map.png` (the ballast holes of the bottom plate seen from below: names, coordinates, which holes are free and which are blocked, the block's outline), `propulsion_layout.png` (the thruster places: for each the direction in which its propeller turns and the way its nose cone points), `enclosure_mounting.png`, `guard_and_brackets.png`, `positioner_and_cells.png` (the positioner, its two index positions, the load cells' stations, the turn column), `trays_and_racks.png` (the ballast box, the thruster tray, the enclosure rack, the guard racks). The printed dimensions are exact; there are few of them |
| `~/given/docs/trim_sheet.png` | the vehicle's mass and buoyancy table, as an image: for every item its mass in air, its centre of mass, its displacement (given in one of three forms: a volume, a mass in water or a net buoyancy; the header names the reference water of the last two), its centre of buoyancy and the station that fits it; the acceptance definition with its windows, its sign conventions and the point where the thrust acts; the range within which a vehicle of this batch differs from the table; the batch's standard ballast plan, with the one vehicle and the one water for which it holds |
| `~/given/docs/water_density_chart.png` | water density against salinity, one curve per water temperature, and for nearly fresh water against temperature, one curve per salinity, as an image; there is no formula |
| `~/given/docs/propeller_datasheet.png` | the two kinds of propeller: the direction in which each turns in forward thrust, seen from the nose cone, and what a right-handed and a left-handed blade look like; the same convention as `propulsion_layout.png` |
| `~/given/docs/station_scope.md`, `work_order_example.json` | what is fitted upstream, at this station and at the next station; the form of a work order (the example's values belong to no graded run) |
| `~/given/manual/photo_01.jpg` .. `photo_16.jpg`, `photos.json` | the picture manual: the vehicle as it arrives, deck up and keel up, the loose units of this station, the vehicle after each step of this station, propellers of the horizontal thrusters close up, both ends of the electronics enclosure, the standard ballast plan seen from below, the finished vehicle in four views. `photos.json` lists per photo the caption, the size, a focal-length class and the scale references in view with their nominal sizes; the camera poses and the lens are not known. The finished vehicle in the photographs carries the parts of the next station; the captions name them |
| `~/given/cell/cell_manual.md` | the integrator's cell description: bench, robot base pose (exact), the gripper's fingers, the datum boards, the positioner with the vehicle's pose (exact), the turning procedure, the load cells with their noise and the conditions under which a reading is valid, the ballast box (exact), the nominal places of the thruster tray and the racks and how much they vary, cameras with resolution, nominal field of view and approximate mounts, depth-sensor and force-sensor noise figures, the fastening rules with every window and the entry behaviour measured in the cell, every difference between the cell's collision shapes and the drawn vehicle, safety rules, budgets |
| `~/given/cell/cell_api.md`, `cell_config.json` | the cell's programming manual (the wire API); the cell's configuration as far as it is public |
| `~/given/robot/` | the robot's MuJoCo models: the arm (`universal_robots_ur10e/`) and the wide gripper (`wide_gripper/`), each with its licence |
| `~/given/lib/` | the cell's client and helper code: `arm.assembly_rov.cell_client_ext` (this station's proxy and the launcher), `arm.assembly_info.client` (the general camera-cell proxy that it extends, `decode_frame`), `arm.assembly_rov` (the cell's own arm model builder, its controller and the gripper module with its measured tables), `arm.assembly_cell` / `arm.assembly_six` (physics building blocks of an earlier cell), `render` (the camera rendering layer; it renders like the cell's cameras: `arm.assembly_rov.agent_render` is the cell cameras' own render session for your twin of the cell, `render.filament.view` renders any scene of yours the same way; MuJoCo's own renderer, `mujoco.Renderer` or the render layer's `mujoco` backend, draws a different picture than the cell's cameras (no workshop, MuJoCo's default lights, flat colours, no reflections), so image processing tuned on its frames may behave differently on the cell's frames; `cell_api.md`, "Rendering your own frames") |
| `~/render/` | the scenery of the cell cameras' renderer, which the given render layer uses too (the workshop around the cell as meshes and baked textures, the bench top's maps); nothing about the vehicle |

Tools: Python 3.12 (`/opt/venv`, on PATH) with numpy, scipy, OpenCV (`cv2`, headless), scikit-image, MuJoCo 3.14,
trimesh, manifold3d, Pillow, matplotlib. No network in your session.

## How you are graded (verify runs 4 hidden episodes)

Gates (multiply the score; a failed gate makes the whole run score zero): `contract_ok` = `assemble.py` is a regular
file that parses and defines `solve`, passes the route gate (no weight files: `.pt`, `.pth`, `.onnx`, `.safetensors`,
`.ckpt`, `.h5`, `.pb`, `.pkl`, `.npz`; no training calls; submission <= 60 MB), imports under the run regime above,
makes at least one `step` call over the episodes, causes no network activity during any episode (the grader reads the
container's outbound network counters before and after each episode and samples the program's sockets), and your own
`.py` files import no networking module (`socket`, `ssl`, `urllib`, `http`, `requests`, `asyncio`, `ctypes`,
`subprocess`, ...), load no module under a computed name, execute no computed code (`exec` / `eval` / `compile` of a
non-literal) and start no other program (`os.system`, `os.exec*`, `os.spawn*`, ...), and nothing of the submission was
written after your session ended. The code check parses your `.py` files (`~/given/lib/arm/assembly_info/netaudit.py`)
and counts only the built-ins `exec(...)`, `eval(...)` and `compile(...)` called by their bare names with a
non-literal argument, so a method of the same name, such as MuJoCo's `spec.compile()`, is fine.

Per episode, two things zero that episode: a robot link other than the gripper's pads pressing with more than 30 N on
the vehicle, the positioner, a fixture, the bench or a unit; and a unit pressed on the environment or on another unit
with a contact force of more than 150 N for 0.2 s. Only contact forces count for the second: the force with which the
cell itself holds or fastens a captured unit does not. `cell_manual.md`, "Safety", says how the contact forces of each
rule are added up. Dropping a unit does not zero the episode but scraps the unit: a loose unit whose centre of mass
moves faster than 3 m/s, or that falls off the bench, or that is on the vehicle and not fastened when the positioner
starts to turn, can no longer be fastened and no longer counts; the episode goes on with the other units. The dome of
the electronics enclosure must not be touched: anything that presses on it with more than 20 N for 0.05 s scraps the
enclosure.

`assembly_score` = the mean over the episodes of

    ( the weights of the places that count + 4.5 x trim ) / 15.5

The weights: the electronics enclosure 3.0; each vertical thruster 1.5; each thruster guard 1.0. A place counts when
the cell has fastened a unit on it at the end of the episode, in the orientation that counts and, for a vertical
thruster, with the propeller that the place needs, and the unit is not scrapped. Ballast blocks carry no weight of
their own: they count through `trim`.

`trim` is a functional acceptance of the vehicle as it leaves the factory, a number from zero to one. The grader
computes it for: the base vehicle of this run (with its own mass and centre of mass; its displaced volume and its
centre of buoyancy as on the trim sheet), every unit of this station and every unit of the next station at its
trim-sheet values, and the ballast blocks that are fastened on the keel when the episode ends, each where it is
fastened. Rule T1: a unit of this station that your program did not fit is fitted by rework and enters at its
trim-sheet values all the same, so `trim` depends on the ballast alone. In the water of the work order, of density rho
as `water_density_chart.png` gives it for the work order's temperature and salinity:

    W = g x (sum of the masses)       B = rho x g x (sum of the displaced volumes)       dB = B - W
    d = B x (r_B - p_T) - W x (r_G - p_T)
    pitch = atan2(-d_x, d_z)          roll = atan2(d_y, d_z)

    trim = 0.4 x [ 0.2 N <= dB <= 3.0 N ]
         + 0.3 x [ the first, and |pitch| <= 3.0 deg ]
         + 0.3 x [ the first, and |roll| <= 3.0 deg ]

r_G and r_B are the centre of mass and the centre of buoyancy of the whole vehicle in the vehicle frame, p_T is the
point at which the thrust of the vertical thrusters acts when they hold the vehicle's depth, g is the standard
gravity; the trim sheet prints p_T, g, the sign conventions and the same windows. Every set of fastened blocks that
meets the windows scores the same: there is no reference plan to match. A block that is not fastened when the episode
ends is not on the vehicle.

What the ballast alone is worth: a program that ballasts the vehicle and fits nothing else can reach 4.5 / 15.5 = 0.29
of an episode and no more. The ballast box and the vehicle on the positioner stand at exact, documented places.

The reward is `assembly_score` x the gate. Stats (not scored): per-place rates, units fastened in an orientation or
with a propeller that does not count, scrapped units, the fastened blocks and dB, pitch and roll per episode, turns of
the positioner, frames used per camera, sim / wall time, illegal calls, safety events, the largest sustained contact
force, network counters.

## Rules of the cell (details in `cell_manual.md` / `cell_api.md`)

* **Fastening.** A unit is captured once it is inside the capture window of its place, a few millimetres and degrees
  (across: the electronics enclosure 2.0 mm, a vertical thruster 2.5 mm, a guard 3.0 mm, a ballast block 3.0 mm;
  `cell_manual.md` has every window and what was measured), and at rest there relative to the vehicle for 0.1 s. A
  unit that slides or falls through its window is not captured. The cell then holds the unit there and pulls it home
  (the screws being driven, at most 10 mm/s), and fastens it when it rests at home for 50 consecutive 10 ms ticks. A
  unit may still be held while it fastens. The cell never tells you about any capture or fastening, and nothing can be
  unfastened. No place has a physical key: nothing guides a unit into its window.
* **Units that do not stay by themselves.** A vertical thruster and a guard do not stay on their places without being
  held. Hold such a unit still inside its window for at least 1.5 s before you let it go. Once captured it does not
  fall when it is let go; let go before that, it falls.
* **Orientation and propellers.** The electronics enclosure is fastened either way round, and a vertical thruster in 4
  orientations and with either kind of propeller; only the way round that the drawings show counts, and a thruster
  only on a place that needs its propeller. Fastened otherwise, a unit stays fastened for good and counts nothing. A
  guard is captured one way only.
* **The positioner.** Every run starts with the vehicle at `keel_up`. `fixture_index` turns it over, at most 4 times
  per run and only while the robot is outside the turn column. When a turn starts, every unit on the vehicle that is
  not fastened is scrapped by the station; what is fastened turns with the vehicle.
* **Weighing.** `get_state()` carries the readings of the three load cells under the positioner, in newtons, with
  noise. They are valid at an index position with the positioner at rest, with nothing but the vehicle and what is on
  it loading the fork, 0.5 s after the last outside force is gone. The cells stand on the bench and do not turn with
  the vehicle.
* **The cell's collision shapes are not the drawn shapes.** The cell simulates simpler shapes, with material taken
  away almost everywhere: fasteners do not collide, the guards' cut-outs and a few faces and edges beside the places
  are cut back. Round parts are solid prisms, and small holes in the plates are filled. `cell_manual.md`, "Collision
  model", lists every difference with its size.
* **Transparent parts.** The acrylic tube and dome of the electronics enclosure give no depth reading in the depth
  images. The acrylic tube of the battery enclosure, a part of the vehicle as it arrives, gives none either.
* Calls other than the documented public ones answer `{"error": "forbidden"}` and are counted.

## No practice

There is no practice cell, no answer key and no checker: everything known about this vehicle and this station is in
the given files, and the only run of your program on the cell is the graded one.

## Notes

- Photos, drawings and camera frames are meant to be processed programmatically; view only the images you need.
- The cell frame, the robot base pose, the bench board's pose, the vehicle's pose on the positioner and the ballast
  box are exact, the tool plate's pose within its mounting tolerance; everything else about the cameras, the thruster
  tray, the racks and the units' poses must be measured in each episode.
- Do not modify the given files in your home directory; the grader reads its own copies.
- Machine: your session's sandbox has 8 CPUs and 24 GB of memory, no GPU and no network. The grading machine has the
  same size, and there your program shares it with the cell and its camera renderer, which take about 8 GB between
  them.
- Environment facts (harness limits): a single tool call whose input runs above roughly 25-30k tokens is dropped by
  the harness (it is never executed and no error comes back), and a response that reaches the 64k output-token limit
  is lost and, after at most a few retries, ends the session with whatever is in `/home/user/submission/` at that
  moment. Write large files in several parts (one function group per write), keep every single response well below
  that limit, write a first complete submission early and refine it.
- Session time: this session is stopped after 6 hours of wall-clock time from its start. A session stopped by that
  limit is graded on whatever the submission folder holds at that moment, so a file left half-written then is what
  runs. Note the start time with `date` when you begin, keep a complete submission in place from early on, and end the
  session yourself (stop working and say that you are done) at least 15 minutes before the limit.
