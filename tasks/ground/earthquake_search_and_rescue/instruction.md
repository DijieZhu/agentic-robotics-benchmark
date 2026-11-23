# Earthquake search and rescue with a tracked rescue robot

A strong earthquake has brought down part of a three- to four-storey building of shops and flats. Neighbours say
someone is trapped inside. Your urban search-and-rescue team has an **SR-2 rescue crawler**: a tracked robot with
front and rear flippers, a track gauge it can narrow or widen, a telescopic sensor mast (RGB-D camera, thermal camera,
16-channel lidar, lamps) and a five-axis arm with a gripper. It carries a **survival kit** (water and a hand-held radio)
in a cradle on its rear deck.

Inside the rubble the robot's radio link to the team drops after a few walls, so nobody can drive it in there: it must
search on its own. You are the team's robotics engineer. **Write the robot's autonomous search-and-rescue program.**
On each mission the team sets the robot down on the street in front of a collapsed building it has never seen, and
your program has to:

1. search the building with the robot's own sensors: work through the rubble, clear what blocks the way and can be
   moved, take the passages the robot can take in a suitable configuration, and avoid what would bring more of the
   building down;
2. **find the survivor**, see them with a camera, and **report their position** in the robot's map frame;
3. **place the survival kit within the survivor's reach** (within 0.8 m of their head or a hand, not on them);
4. **bring the robot back** to where the team set it down.

## What you leave

`/home/user/submission/rescue.py` defining `solve(robot)`, plus any helper modules next to it (pure Python; numpy,
scipy, Pillow and OpenCV are installed). The grader imports it and calls `solve(robot)` once per mission with a
`robot` object of the given client library (`~/given/lib/rubblecell/client.py`): describe, sensors (cameras, lidar,
thermal camera, state), motion (drive, flippers, gauge, mast, head, arm, gripper), `step` (simulated time passes only
there) and `report_survivor`. Everything is in `~/given/robot_api.md` and `~/given/robot_manual.md`;
`~/given/field_guide.md` is the team's field guide to collapsed buildings. `~/example/rescue.py` is a minimal program
that runs.

## What a submission must satisfy

Before any mission the grader checks the submission folder. **A submission that fails one of these checks scores 0 on
every site.** The checks are the code of `~/training/submission_checks.py` (the grader runs its own copy of that
file); `/opt/venv/bin/python3 ~/training/run_mission.py --check` applies all of them to `~/submission` without using
a training mission.

1. `rescue.py` is a regular file that parses as Python and defines `solve`.
2. The folder holds no file whose name ends in `.pt`, `.pth`, `.onnx`, `.safetensors`, `.ckpt`, `.h5`, `.pb`, `.pkl`
   or `.npz`; no `.py` or `.sh` file contains the text `torch.optim`, `.backward(` or `jax.grad` (the text is
   searched as it stands, comments and strings included); the folder is at most 60 MB.
3. No `.py` file imports a networking or process-starting module: `socket`, `ssl`, `urllib`, `http`, `requests`,
   `httpx`, `aiohttp`, `ftplib`, `smtplib`, `imaplib`, `poplib`, `nntplib`, `telnetlib`, `websocket(s)`, `paramiko`,
   `pycurl`, `xmlrpc`, `asyncio`, `socketserver`, `grpc`, `zmq`, `huggingface_hub`, `boto3`, `botocore`, `wget`,
   `ctypes`, `cffi`, `selectors`, `multiprocessing.connection`, `multiprocessing.managers`, `logging.handlers`,
   `subprocess`, `pty` (the full list is `FORBIDDEN_MODULES` in the file). None imports a module under a computed
   name (`__import__`, `importlib.import_module` or `find_spec` with anything but a string literal), runs computed
   code (`exec`, `eval` or `compile` of anything but a string literal) or starts another program (`os.system`,
   `os.popen`, `os.exec*`, `os.spawn*`, `os.posix_spawn*`). A file that is a byte-for-byte copy of a file of the
   given client library is exempt.
4. `import rescue` succeeds within 120 s of wall clock and 120 CPU-seconds in the run regime below.
5. Over the missions the program calls `step` at least once, leaves no process running when a mission ends and
   holds no remote network socket. Nothing may be written into `/home/user/submission/` after your session ends.

The run regime, the same for the import check and for every mission: `python -B -E -s` as an unprivileged account,
the working directory a **read-only** copy of your submission folder (regular files only; symbolic links are
dropped), so a program that writes a file next to itself fails with a `PermissionError`; an empty writable `HOME`
and `TMPDIR`; no file kept from one mission to the next; no network; the client library on `PYTHONPATH`.

Your program's arithmetic is pinned, so that it computes the same on the training ground and on the grading
machine, whatever their CPUs: NumPy's linear algebra on one fixed set of kernels and one thread
(`OPENBLAS_CORETYPE=Haswell`, `OPENBLAS_L2_SIZE=256`, `OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1`), NumPy without
its AVX-512 loops (`NPY_DISABLE_CPU_FEATURES="X86_V4 AVX512_ICL AVX512_SPR"`) and OpenCV's IPP at AVX2
(`OPENCV_IPP=avx2`). `run_mission.py` sets the same variables, and so does this sandbox's shell. The grader checks
before the first mission that its machine reproduces a fixed set of NumPy, SciPy and OpenCV results under this pin
and refuses to grade otherwise. If your program changes these variables or starts its own threads, its results may
depend on the machine. The robot's sensors are part of the pin: the same robot state gives the same sensor data
on every machine.

## How it is graded

The grader runs your program on **8 hidden sites**: collapsed buildings with their own layout, damage, obstacles and
survivor position, none of which you have here. Each mission ends when your program returns or calls `finish()`, when
the robot is lost, at a protective stop or a secondary collapse (below), or after **1800 s of simulated time**. Your
program's own computing time (wall clock minus the robot cell's simulation and sensor rendering) is capped at
**1200 s per mission**, its CPU time at 4800 s and its silence between two requests at 180 s; sensor budgets per
mission: 700 camera frames, 1800 lidar sweeps, 1800 thermal frames.

Each mission scores, on a 0 to 1 scale:

| part | weight | credit |
|---|---|---|
| survivor located | 0.45 | your last survivor report that **reached the team over the radio link** (a report is sent at once when the link is up, else queued until it is): full credit within 1.0 m of the survivor (horizontal distance to the middle of their torso), falling linearly to 0 at 3.0 m; only if, before that report, a camera or thermal frame your program took showed the survivor from 6 m or closer (camera to the middle of their torso) and large enough to make out: at least 40 of the 160 x 120 pixels of that view (0.2 % of the frame; frames of the larger cameras count at the same 160 x 120 scale) |
| survival kit | 0.25 | at the end, the kit has been put down (it is neither in the gripper nor in its cradle) and lies at rest within 0.8 m (horizontally) of the survivor's head or a hand, at floor level (its centre at most 0.75 m above the floor and at most 0.45 m below it), and does not lie on them (pressing on them with more than 2 N) |
| robot returned | 0.30 | at the end the robot is upright (tilted less than 60°) within 3 m (horizontally) of its deployment point; counted only when the survivor-located part is above 0 |

Three things end a mission early:

- **Protective stop.** If the robot, the kit or anything the robot moves presses on the survivor with more than
  60 N for 30 ms, the robot's protective stop trips: every drive stops for good and the mission ends there
  (`protective_stop` in `step`'s events). The mission keeps what the team had at that moment (a report already
  received, a kit already put down within reach), gets no credit for the return, and scores at most 0.45.
- **Secondary collapse: the mission scores 0.** Load-bearing debris is debris that a slab or beam rests on. Once the
  robot, what it holds or debris it pushes has touched such a prop or the slab on it, moving either of them by more
  than 5 cm brings the structure down and the team aborts the mission (`collapse`). The load is real: the slab's
  weight bears on the prop, so brushing against it or bumping into it does not shift it that far; keeping on
  pushing into it does (it gives way slowly, within a minute of pushing), and so does pulling it out.
- **Robot lost.** A robot that overturns (tilted more than 70° for 2 s) or drops into an opening in the floor is
  lost: the mission ends there and keeps only what the team had already received.

Your score is the mean over the 8 missions, compared with a reference program's measured value on the same sites
(the reference value is not disclosed; full credit is reached slightly below its level). The reference program was
written with the sites' plans; your program has only what the robot senses.

## The training ground

The team's training ground is running in this sandbox (`~/given/robot_manual.md` section 9): the standard **test
lanes** (one obstacle of each kind in a row, a training dummy at the end) and six **training rubble sites** built by
the same generator as the graded ones (none of them is a graded site). It is a place to commission your program on
the real robot and physics, not a rehearsal with an answer sheet: it never tells you where the person was, how far
off your report was, where the kit ended up relative to them, or a score. What you learn there is what your program
itself senses and logs.

`/opt/venv/bin/python3 ~/training/run_mission.py --site lanes` first applies the grader's submission checks (above)
to `~/submission`, then runs it in one mission under the grader's run regime (as your account instead of the
grader's unprivileged one) and prints the robot's mission log: how the mission ended, the simulated time, how many
reports were made and how many reached the team, the events, the budgets used. `--list` shows the sites and the
missions left. You can also drive the robot step by step from your own scripts with
`rubblecell.client.connect_training(site)`. The training ground allows **80 missions** in total, 2 at a time. It
runs the same robot and physics as the grader; a mission can take several minutes of wall-clock time.

## Your sandbox

8 CPU cores, 16 GB of memory, no GPU, no network. `nproc` prints 1 because the image sets `OMP_NUM_THREADS=1` (the
numerics pin above); the cores are there. The training ground's robot cell runs on the same cores (its sensor rendering uses several threads),
so two training missions at once share them with whatever you run.

## Session time

- This session is stopped 4 hours of wall-clock time after it started. What is in `/home/user/submission/` at that
  moment is graded as it is: keep a complete submission that passes `run_mission.py --check` there from early on and
  replace it as you improve it.
- The only clock for this limit is the wall clock. `bash /home/user/session_time.sh` prints the minutes used and the
  minutes left, from a time stamp written when your sandbox was set up (moments before your session started, so you
  have slightly more than it says, never less); `date` shows the same clock. Run the script when you begin, again
  before every decision that depends on the time left, and whenever you pick the work up from notes or from a
  summary of what was done. Do not estimate the time used from how much you have done.
- Three other counts in this task are not your session's time: a mission's 1800 s of simulated time (a few minutes
  to a quarter of an hour of wall clock), a mission's 1200 s of program time, and the training ground's 80
  missions. None of them says how much of the 4 hours is left.
