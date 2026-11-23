# Build a four-digit robot hand from its bolt-on modules with a Panda, guided by its picture manual and the cell's cameras

You are the automation engineer of a small final-assembly cell. The product is a robot hand with four digits, three
fingers and a thumb, and a servo actuator in each of its thirteen driven joints. The stations before yours build each
finger and the thumb as a module and put the boards together; your cell does the last step: it puts nine units onto the
palm base, the bottom shell of the palm with the digits' wall screws in it, which is clamped on a raised fixture with
the back of the hand up. The units are three identical finger modules, the thumb module, the electronics stack, the
upper shell, the dorsal cover, the power hub and the press-fit cap; they lie loose in the nests of two trays. You have
what a cell engineer really has: every unit as it arrives from its station (a model file, a packing list with masses and
the joints' data), the list of the bought parts inside them, the product's picture manual (25 photographs with scale
references in view, captions without dimensions), the cell's commissioning documents and the arm's model. You see the
cell only through its two cameras, a force / torque sensor at the wrist and the arm's own joint state. Today's job:
program the cell's Franka Panda so that it builds the hand the way the manual shows it. The palm pad and every cable are
fitted at the next station and are not part of this job. Screws are not handled by the robot either: they are drawn on
the units from the start, and the cell counts a unit's screws as driven once the unit is at rest in its place (the rules
are below).

Nothing gives the answer directly: there is no mating graph, no design pose, no camera calibration, no unit pose and no
"fastened" signal. In every run the fixture and the trays stand and are turned slightly differently, each finger
module's chain lies swung to another angle in its nest and so does the thumb's, the upper shell, the dorsal cover, the
power hub and the press-fit cap lie either way round, and the cell's cameras are mounted and calibrated differently (a
layout in which the equipment would stand too close together or a unit could not be gripped is drawn again, within the
same stated bounds). The digits' joints are not powered while the hand is built: a joint turns under a light touch and
stays where it is left.

## What you produce (in `/home/user/submission/`)

`assemble.py` exposing `solve(cell) -> None` (plus any helper modules, optional `run.sh`). The grader runs it once per
hidden episode (3 episodes) in its own physics cell. It talks to the cell through `cell` (API in
`~/given/cell/cell_api.md`): the robot's own state, a noisy force / torque sensor at the flange and two RGB-D cameras
(an overhead camera that sees the whole working area and a wrist camera); no unit poses, no mate status. Per episode:
1200 s of simulated time, 2400 s of wall clock (camera rendering counts), 120 s of silence at most between two requests,
600 camera frames in total (a further `get_camera` call is refused) and 9600 s of CPU time for each process of your
program (4 times the wall-clock cap). An episode that runs into the simulated-time cap, the wall-clock cap, the silence
limit or the CPU limit ends there and is scored on what the cell has fastened at that moment; the other episodes still
run.

How the program is run: a fresh process under a dedicated unprivileged account that is not yours, from a per-run
directory with a read-only copy of your submission folder (regular files only), an empty writable HOME / TMPDIR, a
minimal environment (`PATH`, `HOME`, `TMPDIR`, `LANG`, `MUJOCO_GL`, `PYTHONPATH` = the given library `~/given/lib`,
`ARB_GIVEN_DIR` = a read-only copy of `~/given`) and the launcher `python3 -B -E -s -c <bootstrap> <lib> assemble` (or
`bash run.sh`). Resolve given files through `os.environ.get("ARB_GIVEN_DIR", os.path.expanduser("~/given"))`, write only
under the run's HOME / TMPDIR. A program that crashes ends that episode with what it achieved; the other episodes still
run. Your submission must be final when your session ends: the submission folder is then copied out of your sandbox and
graded in a fresh sandbox that holds nothing else of yours (no other file, no process you started).

Memory: the verifier has 24 GiB, shared by the grader, the cell's own camera renderer and your program. Every process of
your program is marked as the first to be stopped when that memory runs out (the kernel's `oom_score_adj` 1000): the
kernel then kills your program's processes, and the episode ends as a crash with what it achieved. A process that lowers
its own mark is killed at the grader's next look (0.5 s apart) and counts as a crashed process, not as a broken gate. A
program that lowers the mark and fills the memory faster than that can end the whole verify without a verdict, and a run
without a verdict has no score.

## What you are given

| path | content |
|---|---|
| `~/given/kit.json` | the packing list: the eight kinds of unit (the palm base and the seven kinds of loose unit); per kind the count, its model file(s), the mass as built (design values, not weighed), the centre of mass, the size, the materials with their colours, the rule that fixes each model file's frame, and the box of the shape the cell collides beside the box of the drawn shape; for the finger module and the thumb module the links and the joints as the cell simulates them (axes, stops, friction, damping, armature, the start pose, the closed four-bar loop of a finger's last joint and its law), with the product's published joint ranges beside them |
| `~/given/parts/*.stl` | the units as built, millimetres, each in its own frame fixed by its own geometry (not its mounting frame): `palm_base.stl`, `finger_module.stl`, `thumb_module.stl`, `electronics_stack.stl`, `upper_shell.stl`, `dorsal_cover.stl`, `power_hub.stl`, `press_cap.stl`; the triangles the cell's cameras draw, fasteners included; visual tessellation, not watertight |
| `~/given/parts/finger_module/`, `~/given/parts/thumb_module/` | each digit module once more, one STL file per link and material, with a MuJoCo model (`finger_module.xml`, `thumb_module.xml`) that loads as it is and moves as the cell's module does (links, joints, stops, friction, the closed loop); these models carry no collision geometry |
| `~/given/purchased/purchased_parts.md` | the bought parts inside the units by generic name and part number, the pieces per unit and their masses; no drawings |
| `~/given/manual/photo_01.jpg` .. `photo_25.jpg`, `photos.json` | the picture manual: the units laid out, every unit turned the way it goes on, each unit before and after it goes on, the finished hand from six sides, close-ups of the seats; `photos.json` lists per photo the caption, the size, a focal-length class and the scale references in view with their nominal sizes; the camera poses and the lens are not known |
| `~/given/cell/cell_manual.md` | the integrator's cell description: bench, robot base pose (exact), the three datum checkerboards (exact), the fixture with the palm base on it, the trays and their nests and how each unit lies in its nest, what changes from episode to episode, cameras with their nominal mounts and how far a mount is off, depth-sensor and force-sensor noise figures, the gripper, the joints of the digit modules, the fastening rules, the collision model, safety rules, budgets |
| `~/given/cell/cell_api.md`, `cell_config.json`, `describe_example.json` | the cell's programming manual (the wire API); the cell's public configuration (limits, budgets, rule numbers, the fixture's and the nests' pieces as the cell simulates them); the cell's `describe()` reply |
| `~/given/panda/` | the robot's MuJoCo model (Apache-2.0) |
| `~/given/lib/` | the cell's client and helper code: `arm.assembly_info.client` (the proxy, `decode_frame`), `arm.assembly_cell` (the arm's controller), `arm.assembly_hand` and `render` (the renderer of the cell's cameras, for a twin of your own); `arm.assembly_six` and the rest of `arm.assembly_cell` and `arm.assembly_info` came with earlier cells and describe none of this cell's units, trays or nests |
| `~/render/` | the scenery of the cell cameras' renderer (the room as meshes and baked textures, the bench top's maps); nothing about the hand |

Tools: Python 3.12 (`/opt/venv`, on PATH) with numpy 2.5, scipy 1.18, OpenCV 5.0 (`cv2`, headless), scikit-image 0.26,
MuJoCo 3.14, trimesh, manifold3d, Pillow, matplotlib. Your session has 8 CPUs and 16 GB of memory and no network.

## How you are graded (verify runs 3 hidden episodes)

`overall` = `assembly_score` x `contract_ok`. `contract_ok` is a gate, one or zero: it is one only while every rule of
the list below holds, and a single broken rule makes it zero and the whole run score zero, whatever the episodes built.

1. `assemble.py` is a regular file in the submission folder. The copy that is graded holds regular files only: symbolic
   links, FIFOs, devices and sockets are dropped, so a link named `assemble.py` counts as missing and a helper that is a
   link is not there when the program is imported.
2. Size: the entries of the submission folder that are not folders are together at most 60 MB (62,914,560 bytes), by
   their sizes as the file system reports them: a sparse file counts by its apparent size, and a hard-linked file once
   for each of its names.
3. Entries: at most 10000 entries in the submission folder, folders included. Every entry counts: files, folders,
   symbolic links, special files, and `__pycache__` with what is in it.
4. Paths: no path inside the submission folder is longer than 4000 bytes (the path from the submission folder down, as
   the file system holds it).
5. Source audit: every `.py` and `.sh` file is decoded, scanned and parsed in a separate process with at most 12 GiB of
   address space, at most 300 s for one file and at most 1800 s for all of them, in wall clock on the verifier. A file
   beyond the memory or its time, or an audit beyond the total, breaks this rule. Python's parser needs several hundred
   times the size of a dense source in memory: measured there, 40 MB of one-statement lines did not fit (refused after
   33 s), and a 10 MB file holding one list of digits took 4.7 GiB and 36 s and passed. Keep bulk data in a data file
   (`.npy`, `.json`), not in a `.py` file.
6. Encodings: a `.py` file declares, in its first two lines, no text encoding other than UTF-8, ASCII or Latin-1 (a
   UTF-8 byte-order mark is fine; no declaration means UTF-8). `.sh` files are read as UTF-8 whatever they declare.
7. `assemble.py` parses as Python 3.12 reads its bytes and holds a definition of `solve` (a `def`, an assignment or an
   import of that name).
8. No weight files: no file whose name ends, in any letter case, in one of the nine extensions `.pt`, `.pth`, `.onnx`,
   `.safetensors`, `.ckpt`, `.h5`, `.pb`, `.pkl`, `.npz`.
9. No training calls: none of the three strings `torch.optim`, `.backward(`, `jax.grad` anywhere in a `.py` or `.sh`
   file, comments and string literals included.
10. No networking and no other program in your code. None of your own `.py` files (a byte copy of a file of the given
    library is exempt) imports a networking module, loads a module under a computed name (`__import__`, `import_module`,
    `find_spec` with an argument that is not a string literal), executes computed code (`exec`, `eval`, `compile` of
    something that is not a string literal) or starts another program (a call of one of the 21 names `system`, `popen`,
    `execv`, `execve`, `execvp`, `execvpe`, `execl`, `execle`, `execlp`, `execlpe`, `spawnv`, `spawnve`, `spawnvp`,
    `spawnvpe`, `spawnl`, `spawnle`, `spawnlp`, `spawnlpe`, `posix_spawn`, `posix_spawnp`, `fork_exec` on `os`, `posix`
    or `_posixsubprocess`, or as a plain name). The networking modules are the 41 names of `NETWORK_MODULES` in
    `~/given/lib/arm/assembly_info/netaudit.py`, a submodule counting for its package: `socket`, `_socket`, `ssl`,
    `_ssl`, `urllib`, `urllib2`, `urllib3`, `http`, `httplib`, `requests`, `httpx`, `aiohttp`, `ftplib`, `smtplib`,
    `imaplib`, `poplib`, `nntplib`, `telnetlib`, `websocket`, `websockets`, `paramiko`, `pycurl`, `xmlrpc`, `asyncio`,
    `socketserver`, `grpc`, `zmq`, `huggingface_hub`, `boto3`, `botocore`, `wget`, `ctypes`, `_ctypes`, `cffi`,
    `selectors`, `multiprocessing.connection`, `multiprocessing.managers`, `logging.handlers`, `subprocess`, `pty`,
    `_posixsubprocess`. Passing one of these names as a literal to one of the import calls above, or to
    `spec_from_file_location` or `load_module`, counts as importing it. A helper `.py` file that cannot be read or
    parsed, or that declares another encoding, counts as a hit.
11. `import assemble` succeeds under the run regime above and `solve` is callable, within 120 s of wall clock and 120 s
    of CPU time.
12. Your program makes at least one `step` call over the episodes.
13. No network activity during an episode: the grader reads the container's outbound counters before and after each
    episode (TCP connections opened, UDP datagrams and ICMP echo requests sent, both IP versions) and samples the
    sockets of your program's processes every 0.5 s.
14. Every process of the run account is killed when an episode ends; a process that is still running 8 s after that kill
    breaks the gate.
15. Nothing in the submission folder is written after your session has ended, and no process of yours outlives it.
16. The grader itself must get through your submission: if it fails with an error while it handles the submission, the
    run scores zero. No ordinary program causes this.

The size, the entry count and the path length are checked first, by looking at the entries of the submission folder
only, before anything of it is copied or read: above one of them nothing of your program runs (no import, no episode).
When several rules are broken the grader names the first of: no `assemble.py`; the entry count; the size; the path
length; the limits of the source audit; an encoding of `assemble.py`; `assemble.py` does not parse or defines no
`solve`; weight files and training strings; networking and other programs in the code; the import; a late write or a
surviving process; then the rules about the episodes (a `step` call, no network activity, no process left).

`assembly_score` = the mean over the episodes of the completed fraction: the weight of the units that the cell has
fastened and that count (each of the four digit modules 1.5; the electronics stack, the upper shell, the dorsal cover
and the power hub 1.0 each; the press-fit cap 0.5) over the total weight 10.5. A fastened unit counts nothing in one
case: the electronics stack fastened half a turn about the vertical away from the way the manual shows it (it seats that
way round as well); like every fastened unit it cannot be taken off again.

Safety, per episode: an episode in which any of the four rules below is broken goes on, and its completed fraction is
multiplied by 0.3 (it keeps 30 % of it).
* R1: any part of the robot other than its two fingers (whose every face counts as a pad) pressing on anything with more
  than 20 N in any tick.
* R2: a unit loaded with more than 25 N for 0.2 s, either by what it touches (the pads' own grip is not counted) or,
  once the cell guides it or has fastened it, through its fastening: a push on a fastened unit, by the pads as well,
  counts.
* R3: a joint of a digit module carrying more than 0.2 N m against one of its mechanical stops for 0.2 s.
* R4: a unit's centre of mass faster than 3 m/s, the speed of a free fall of 0.46 m. A unit that falls from the bench to
  the floor (0.994 m) passes it, so such a fall is a violation.

## Rules of the cell (details in `cell_manual.md` / `cell_api.md`)

* "Home" below is the place and attitude that the picture manual shows for a unit on the hand; it is documented nowhere
  else. A unit is fastened once it has stayed at rest at home (slower than 5 mm/s and 2 deg/s relative to the palm base
  for 50 consecutive 10 ms ticks; it may still be held). The cell never tells you about any fastening, and a fastened
  unit cannot be moved or taken off again.
* A digit module's base goes into its place in the palm base and has to stand against the wall at the end of it (within
  0.5 mm) to be fastened. The cell guides a finger module's base across its channel once it is 8 mm inside and never
  carries it: your program moves it home (the thumb module's base is not guided at all). The last 3 mm before the wall
  count only for a base that came from further out; a base that first appears there (lowered from above) is not
  fastened.
* The electronics stack, the upper shell, the dorsal cover and the power hub are fastened only where they lie on their
  seats: at most 0.5 mm above home, with the seat carrying at least 0.5 of the part's weight (the gripper and friction
  on side faces do not count: a part held in the air or hung between side faces is not fastened). The cell centres such
  a part across its seat once it is close (within 2 mm across, 5 deg and 3 mm above home) and nearly at rest; it never
  moves a part towards its seat.
* The press-fit cap is held by its fit: once its rim is in its opening it stays where it is put and resists a push along
  the fit with 3 N of friction; it is fastened at rest within 0.5 mm of home.
* Two order rules, both physical. A unit is fastened only after the units that it bears on or is screwed onto, and after
  the units whose screws, or whose way onto the hand, it would close off. And a unit is fastened only while no other
  unit, fastened or loose, lies where the screwdriver has to pass to reach its screws; a unit in the way stops a
  fastening without any message, until it is taken away.
* A digit module is a chain of unpowered joints with 0.02 N m of friction and mechanical stops (`kit.json`); nothing in
  the cell holds a joint except its own friction, what it rests on and what your program does.
* The cell collides convex pieces cut from the units' true shapes. They are not given; `cell_manual.md` ("Collision
  model") says where they differ from the given model files: every screw, the small wire clips and hardware enclosed in
  a unit are drawn but do not collide, two fits are opened to slots and three side faces are set back by 0.05 mm.
* Calls other than the documented public ones answer `{"error": "forbidden"}` and are counted.

## Notes

- No practice: there is no practice cell, no answer key and no checker. Everything known about this hand is in its kit
  files and its manual, and the only run of your program on the cell is the graded one.
- What is graded is the submission folder exactly as it stands when your session ends, at the latest 6 hours after its
  start: nothing written after that counts, and a later write into the folder breaks the gate (the rule on late writes
  in the list above).
- Photos and camera frames are meant to be processed programmatically; view only the images you need. Do not modify the
  given files in your home directory; the grader reads its own copies.
- The cell frame, the robot base pose and the bench board's place are exact, and so are the fixture board's and the palm
  base's places on the fixture; the hand plate is mounted within a stated tolerance. Everything else about the cameras,
  the fixture, the trays and the units' poses must be measured in each episode.
- A twin of the cell that you build yourself looks like the cell's camera frames only when it is pictured with the given
  renderer (`cell_api.md`, "Picturing a model of your own"); MuJoCo's own renderer draws a different picture (no room,
  other lights and materials).
- Environment facts (harness limits): a single tool call whose input runs above roughly 25-30k tokens is dropped by the
  harness (it is never executed and no error comes back), and a response that reaches the 64k output-token limit is lost
  and, after at most a few retries, ends the session with whatever is in `/home/user/submission/` at that moment. Write
  large files in several parts (one function group per write), keep every single response well below that limit, write a
  first complete submission early and refine it.
- Session time: this session is stopped after 6 hours of wall-clock time from its start. A session stopped by that limit
  is graded on whatever the submission folder holds at that moment, so a file left half-written then is what runs. Note
  the start time with `date` when you begin, keep a complete submission in place from early on, and end the session
  yourself (stop working and say that you are done) at least 15 minutes before the limit.
