# Build a humanoid robot from five finished modules on its clamped base with one industrial arm, guided by dimension drawings, a picture manual and the cell's cameras

You are the automation engineer of the last mechanical station of a small-batch line that builds a humanoid robot.
The station receives the robot's base, which is the pelvis and the torso with the battery and the covers fitted,
clamped on a stand, and five finished modules in their racks: a right leg, a left leg, a right arm, a left
arm and the head. A 6-DoF UR10e-class arm with a wide two-finger parallel gripper seats each module on its interface on the base: a leg on a hip output flange,
an arm in a shoulder socket, the head on the neck's output. Wiring, the interface screws themselves and every later
inspection belong to other stations; the cell fastens a module that is held on its seat (`~/given/cell/cell_manual.md`, section
7).

You have what the engineer of such a station has: the base and every module as mesh files with their joints, masses
and the collision pieces the cell simulates, a packing list, seven dimension drawings (images), the
line's picture manual (20 photographs, captions without dimensions), the data pages of the actuators, the
station's manual and its programming manual, the models of the arm and the gripper, the cell's own client and camera
rendering code, and the cell itself, which you see only through its two cameras and only in the graded
run. Your job: program the arm so that it mounts all five modules the way the drawings and the
photographs show the finished robot.

Nothing gives the answer directly: there is no list of which interface takes which module beyond what the pictures
show, no seat pose as numbers, no bolt position, no direction of insertion, no order of work, no way of holding a
module, no camera calibration, no module pose and no "fastened" signal. The facts about where and how each module
sits are on the drawings and in the photographs; `~/given/cell/cell_manual.md`, section 1, says which sheet and which
photograph carries which kind of fact, and no more. In every run the stand, the leg rack, the arm racks and the head's
nest stand slightly differently, the base sits slightly differently in its clamp, and the overhead camera is mounted
and calibrated differently.

## What you produce (in `/home/user/submission/`)

`assemble.py` exposing `solve(cell)` (plus any helper modules, data files, optional `run.sh`). The grader runs it
once per hidden episode (3 episodes, each on a layout chosen anew at random) in its own physics cell. The program
talks to the cell through its one argument (the API: `~/given/cell/cell_api.md`): the arm's own state, a noisy force and torque
sensor at the flange and two RGB-D cameras (an overhead camera and a wrist camera); no module poses,
nothing about what is fastened. Per episode: 1293 s of simulated time, 8074 s of wall
clock (camera rendering counts), 120 s of silence at most (counted from the program's start or the
cell's reply to a request until the program's next request; any request resets it), 450 camera
frames in total. The wall clock can end an episode before the simulated time does: the wall cap is
built on about 4.2 s of wall clock per simulated second (`~/given/cell/cell_manual.md`, section 11, has it
beside the allowance for a frame). Each process of the program may use at most 32296 s of
CPU time per episode (four times the wall cap); a process that passes it is killed, and when it is
the program's own process the episode ends with what it achieved.

How the program is run: a fresh process under a dedicated unprivileged account that is not yours, from a per-run
directory with a read-only copy of your submission folder (regular files only), an empty writable home and temporary
folder, a minimal environment (the search path, the home and temporary folders, the locale, the off-screen rendering
setting, the Python path set to a read-only copy of the given library, and `ARB_GIVEN_DIR` set to a read-only copy
of `~/given`) and the launcher of `~/given/cell/cell_api.md`. The process cannot open an Internet socket. Resolve given files
through `ARB_GIVEN_DIR` and write only under the run's home and temporary folders. A program that crashes ends that
episode with what it achieved; the other episodes still run. Your submission must be final when your session ends: at
verify start every process of your account is stopped and killed and the submission is snapshotted; a submission entry
written after that time voids the contract gate, and so does a surviving process of yours.

## What you are given

| path | content |
|---|---|
| `~/given/kit.json` | the packing list: the base and the five modules, each with its role, quantity, mass, centre of mass, size, the folder of its files and, in the field `presented`, the fixture that presents it (`stand` for the base, `rack:<id>` for a module, with the fixture's id of the given fixtures file). It lists no interface, no seat pose and no direction of insertion |
| `~/given/parts/<unit>/` | per unit a folder: `module.json` (its bodies, each joint's axis, range, actuator type and the angle it is delivered at, each body's mass, centre of mass and inertia, and the frame rule of the unit's files), one mesh file per body in millimetres (`<body>.stl`) and `collision.npz`, the convex pieces the cell's physics uses for that unit (the same pieces as the grader's) |
| `~/given/fixtures/` | the stand, the leg rack, the arm racks, the head's nest, the rack table, the camera portal and the datum boards at their nominal places in the cell frame: one description file and their meshes |
| `~/given/drawings/` | seven dimension drawings as images (`drawing_1.png` to `drawing_7.png`) and a note on how to read them. Their values are exact and are given only as pictures |
| `~/given/manual/` | the picture manual: 20 photographs (`photo_01.jpg` to `photo_20.jpg`) of the cell, the base, the modules, the interfaces close up, states of the build and the finished robot in four views, with `photos.json` (per photograph its caption, its size and the scale references in view) and a note. The camera poses and the lens are not known |
| `~/given/datasheets/` | one data page per actuator type (types A to E), as images: outline, mass, the hole pattern of its output flange where the model has one, and the holding friction, the damping and the reflected inertia of the unpowered actuator that the cell simulates. Read a page from crops at full size, like a drawing |
| `~/given/cell/cell_manual.md` | the integrator's description of the station in twelve sections: scope and where to read what, frames and datum boards, arm and gripper, stand and clamp, racks, modules and their unpowered joints, fastening with every window and what was measured in the cell, the collision model, cameras and the render layer, what varies per episode, safety rules and budgets, what a stopped session gets |
| `~/given/cell/cell_api.md`, `~/given/cell/cell_config.json` | the cell's programming manual (the seven calls); the cell's public configuration |
| `~/given/robot/` | the models of the arm and of the gripper, with their licences |
| `~/given/lib/` | the cell's client and helper code: the proxy and the frame decoder (`arm/assembly_humanoid/`, with `arm/assembly_cell/client.py` and `arm/assembly_info/client.py`), the cell's own arm model builder, controller and gripper code (`arm/assembly_rov/`), and the render layer of the cell's cameras (`render/`, entered through `arm/assembly_humanoid/agent_render.py`). MuJoCo's own renderer gives a different picture than the cell's cameras (no room, its own lights, flat colours), so image processing tuned on its frames may behave differently on the cell's frames; the given layer is the way to see what the cell's cameras would see (`~/given/cell/cell_api.md`, "Rendering your own frames"). Other folders and files of the library are shared code of other cells. |
| `~/render/` | the scenery of the cell cameras' renderer (the room as meshes and baked textures) and the render twin of the cell; nothing about seats or poses |

Tools: the Python environment on the search path with numpy, scipy, OpenCV (headless), scikit-image, MuJoCo
3.14.0, mink, Pinocchio, CoACD, CadQuery, trimesh, Pillow and matplotlib. No network in your
session.

## How you are graded (verify runs 3 hidden episodes)

The gate (it multiplies the score; a failed gate makes the whole run score zero): `assemble.py` is a regular file
that parses and defines the entry function; the submission passes the route gate (no weight files, no training calls,
at most 60 MB); your own program files pass the code check (they import no networking module, load no
module under a computed name, execute no computed code and start no other program); the program imports under the run
regime above within 120 s of wall clock and 120 s of CPU time, makes at least one
`step()` call over the episodes and holds no Internet socket during any episode (the run regime refuses such
sockets, the grader checks that this refusal is in force, and during each episode it samples the sockets owned by the
account that runs the program); and nothing of the submission was written after your session ended. Both checks of
the submission's files work on fixed lists, given here in full.

The route gate reads every file of the submission folder. A weight file is any file whose name ends, in any letter
case, in one of `.pt`, `.pth`, `.onnx`, `.safetensors`, `.ckpt`, `.h5`, `.pb`, `.pkl`, `.npz`, whatever the file holds, a copy of a given file included: the archive format of numpy
and the format of pickle are among them, so data of your own must have another format. A training call is one of the strings `torch.optim`, `.backward(`, `jax.grad` anywhere in the text of a file
whose name ends in one of `.py`, `.sh`: the match is on plain text, so a comment or a string that holds one fails
too. The size is the sum over all files of the folder.

The code check is static: it parses your files whose names end in `.py` and finds imports and calls by
their names, whether or not they would run (`~/given/lib/arm/assembly_info/netaudit.py` is the check; a file that is byte for byte a file of the
given library is not read). It fails on an import of a module of this list, or of anything inside such a module:
`socket`, `_socket`, `ssl`, `_ssl`, `urllib`, `urllib2`, `urllib3`, `http`, `httplib`, `requests`, `httpx`, `aiohttp`, `ftplib`, `smtplib`, `imaplib`, `poplib`, `nntplib`, `telnetlib`, `websocket`, `websockets`, `paramiko`, `pycurl`, `xmlrpc`, `asyncio`, `socketserver`, `grpc`, `zmq`, `huggingface_hub`, `boto3`, `botocore`, `wget`, `ctypes`, `_ctypes`, `cffi`, `selectors`, `multiprocessing.connection`, `multiprocessing.managers`, `logging.handlers`, `subprocess`, `pty`, `_posixsubprocess`; on a call of `__import__`, `import_module`, `find_spec` without a string literal as its leading argument, or a call of
`__import__`, `import_module`, `find_spec`, `spec_from_file_location`, `load_module` whose literal argument names a module of that list; on a call of the built-in `exec`, `eval`, `compile`
by its bare name without a string literal as its leading argument (a method of the same name on an object, such as the
compile method of a MuJoCo model specification, is fine); and on a program started by a call named
`system`, `popen`, `execv`, `execve`, `execvp`, `execvpe`, `execl`, `execle`, `execlp`, `execlpe`, `spawnv`, `spawnve`, `spawnvp`, `spawnvpe`, `spawnl`, `spawnle`, `spawnlp`, `spawnlpe`, `posix_spawn`, `posix_spawnp`, `fork_exec`, made by its bare name or as an attribute of `os`, `posix`, `_posixsubprocess`. A file with a syntax
error is not read by the code check: Python cannot import it either. A file fails the check, whether or not the
program imports it, when it begins with a byte-order mark or when the parser refuses it for another reason: bytes that
are not UTF-8 text, a declared source encoding other than UTF-8, a null byte, or an expression chained
or nested so deeply that the parser runs out of its stack (a nesting of more than 200 brackets is a
syntax error).
`run.sh` is not parsed: it is read for the training-call strings only, and the programs it starts run under the
same account, the same socket refusal and the same socket check.

Per episode the cell counts every mate that it has fastened at the bolt position the drawings and photographs show,
with these weights: a leg 1.5, an arm 1.5, the head 1.0; the total is 7.0. A module that
the cell fastened at another bolt position stays fastened for good and counts nothing. A module of which any body
touches the floor is scrapped: it can no longer be fastened and no longer counts, and the contact is a safety
violation.

Safety violations, per episode: a link of the arm or gripper pressing with more than 150 N
on anything, except the gripper's pads on a module; a module pressed on any single other body with more than
150 N for 0.2 s; a module's centre of mass moving faster than
3 m/s; a module touching the floor. A violation does not end the episode, but an episode with
a violation keeps only 30 % of its score, whether it had a single violation or several.

The score is the mean over the episodes of the counted weight over 7.0, kept at 30 % in an
episode with a violation, times the gate. Statistics that are not scored: per-mate rates, modules fastened at another
bolt position, scrapped modules, frames used, simulated and wall time, illegal calls, safety events and the largest
forces.

## Rules of the cell (details in `~/given/cell/cell_manual.md` and `~/given/cell/cell_api.md`)

* **Fastening.** The cell fastens a module when its interface stays inside the lock window of its seat and moves
  slower than 5 mm/s and 2 degrees per second relative to the seat for
  0.5 s without a break. The windows are a few millimetres and degrees at most, the lock window a
  fraction of a millimetre and of a degree (`~/given/cell/cell_manual.md`, section 7, has each of them and what was
  measured). A module may be held while it is fastened, and
  it need not be at rest. Before that moment the cell applies no force to a module: it does not pull it in, it does
  not hold it, and it does not rescue a module that is let go without being fastened. The cell never tells you about
  a fastening, and nothing can be unfastened.
* **Bolt positions.** A leg's flange fits its output at 6 bolt positions, an arm's ring fits its
  socket at 12, the head's bracket fits the neck's output at 6. The cell
  fastens at whichever position the module is held at; for each module a single position counts.
* **Only its own module.** Each interface takes its own module only. A module presented to the mirrored interface of
  the other side is never fastened there.
* **Nothing is powered.** Every joint of the robot is unpowered and holds by its holding friction only. The modules'
  joints, the output rotors of the hips and the neck and the waist move when they are loaded beyond their holding
  friction and stay where they stop. The torso can turn on its brace under a sideways push and stays where it stops (`~/given/cell/cell_manual.md`, section
  4).
* **The collision shapes are the given pieces.** The cell simulates the convex pieces of `collision.npz`, which
  are exact at the interfaces and can differ from the surfaces of the mesh files elsewhere (`~/given/cell/cell_manual.md`, section
  8, with the sizes).
* Calls other than the documented public calls are refused and counted (`~/given/cell/cell_api.md` says how the proxy reports a
  refused call).

## No practice

There is no practice cell, no answer key and no checker: everything known about this robot and this station is in the
given files, and the only run of your program on the cell is the graded run.

## Notes

- Photographs, drawings and camera frames are meant to be processed programmatically; view only the images you need.
  A drawing is 4200 by 2970 pixels and its smallest lettering is about
  16 pixels high: read a drawing from crops at full size, because a whole sheet shown reduced does
  not show its smallest text.
- The cell frame and the arm's base pose are exact; the places of the fixed datum boards are exact; everything else
  about the cameras, the fixtures and the modules must be measured in each episode.
- Do not modify the given files in your home directory; the grader reads its own copies.
- Machine: your session's sandbox has 8 CPUs and 24 GB of memory, no GPU and no network.
  The grading machine has the same size, and there your program shares it with the cell's physics and its camera
  renderer (`~/given/cell/cell_api.md`, "Rendering your own frames", has what they take).
- Session time: this session is stopped after 6 hours of wall-clock time from its start. A session
  stopped by that limit is graded on whatever the submission folder holds at that moment, so a file left half-written
  is what runs. Keep a complete submission in place from early on.
