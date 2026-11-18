# Take a tenth-scale autonomous race car through a time-trial day at venues it has never seen: map the track in practice, set timed laps in two heats, pass the obstacle test

You are the software engineer of a small team in a regional indoor league for tenth-scale autonomous race cars.
The team's car is a four-wheel-drive touring-class chassis with an autonomy deck. It senses with a 2D scanning
LiDAR, an inertial unit and the speed of its motor; it has no camera. A round of the league goes from hall to
hall. At each venue a temporary track, walled with flexible air ducts, lies on the bare floor of the hall, and the
car does, unattended, what every team's car does on a time-trial day: a practice session in which it maps a track
it has never seen, two timed heats, and an obstacle test with boxes and slower cars on the track (the rules call
it the avoidance test). Nobody touches the car or its program from the first session to the last. Your job is the
program that drives it.

Nothing gives the answer directly. There is no drawing of any track, no map, no position of the car, no value for
the grip of a floor, and no simulator of the car or of a venue among your files. You have what a team of this
league has: the league's rules, its guide to the venues of past seasons with photographs, the car's datasheet,
logs of the car driven in the team's lab, and the client library through which a program talks to the car. Every
venue of the round is different: another hall floor with another grip, another layout, other places where the
floor differs locally, where a wall is open or dark, and where things stand beside the track.

## What you produce (in `/home/user/submission/`)

`drive.py` exposing `solve(car) -> None` (plus any helper modules and small data files of your own). The grader
runs it at several hidden venues, once per venue, each time as a new process that lives through the whole day at
that venue: a closed practice of 480 s whose first 120 s are a mapping period with a speed limit of 2.0 m/s, two
heats of 300 s each, and an avoidance test of 180 s. The program talks to the car through `car`, a client of five
calls (`/home/user/given/interface/car_api.md`). Each `car.step(steer, speed)` moves the day on by one tick of
25 ms and returns what the car's sensors gave in that tick (one revolution of the scanning sensor, 1081 ranges;
five inertial samples; the speed from the motor; the steering echo) and the organiser's messages. No call returns
a position, a map, what the car touched or a score.

Budgets of the program at a venue (`car_api.md`, section 6): on the track 15 ms of its own time per tick, summed
over a session; at the desk before the four sessions 60 s, 300 s, 120 s and 60 s; 120 s from its start to its first
call; 180 s of silence at most; CPU time at most 4 times these wall-clock budgets. A program that exceeds a
budget, raises an error or falls silent ends its day at that venue with what it had achieved; the other venues
are still run.

How the program is run: a fresh process per venue under a dedicated unprivileged account that is not yours, in a
read-only copy of your submission folder (regular files only), with an empty writable HOME / TMPDIR, a minimal
environment and the launcher of the given library (`python -B -E -s`, your folder as the working directory, `carkit`
importable). The numeric libraries run single-threaded. A process of the program may use at most 12 GiB of address
space and write files of at most 1 GiB. It cannot start another process: every attempt fails and no process is made.
It cannot open a network socket: creating a socket other than a Unix-domain one ends the process at once and fails
the gate below. The given files are not there at a venue: whatever the program needs from them must be in the
submission folder. Write only under the run's HOME / TMPDIR; what the program leaves elsewhere is removed after
every venue. Before the first venue the grader imports `drive.py` once under the same regime; the import must finish
within 120 s and 120 s of CPU time. A venue at which the grader's own simulation fails is run once more from
scratch. Your submission must be final when your session ends: only `/home/user/submission` is carried over to the
grading, a submission entry written after that time fails the gate below, and so does a process of your program that
outlives its day at a venue.

## What you are given

| path | content |
|---|---|
| `/home/user/given/league/rules.md` | the league's rules of a time-trial day, clauses S-01 to S-31: the track, the four sessions, touch and crash, what the staff do after a crash, the two time-trial tables, the avoidance test, the scoring with its formulas, the organiser's messages |
| `/home/user/given/league/venues.md`, `league/photos/` | the league's guide to its venues: what a venue sheet says and how far each entry can be trusted, floors and the range of grip, local patches of the floor, kinds of walls, gaps, open walls, shared walls, the track's range of sizes, what stands beside the track; 12 photographs of earlier venues with captions in the guide. None of them is a venue of your round |
| `/home/user/given/car/datasheet.md`, `car/meshes/` | the car: frames, geometry and mass, steering, drive, the three sensors with their rates and limits, timing; its outer geometry. It gives no dynamics model: no model file, no tyre curve, no controller gains |
| `/home/user/given/lab_logs/` | `README.md` and 14 logs (`*.jsonl`, about 60 MB) of this car driven in the team's lab, on a floor that grips more than any venue floor: every command sent, everything the sensors returned in the same form as at a venue, and the car's pose from the lab's overhead tracker |
| `/home/user/given/interface/car_api.md` | the program interface: how a program is run, the five calls, `describe()`, the observation, the messages, the budgets, the errors |
| `/home/user/given/interface/carkit/` | the client library (standard library only; uses numpy when present): the `Car` client, a decoder of the messages, the beam geometry of the scan, a reader of lab logs |
| `/home/user/given/interface/stub/stub.py` | a protocol stub: it starts a program the way a venue does and answers it with the rows of one lab log. It checks the protocol and the timing of a program and simulates nothing |

Tools: Python 3.12 in `/opt/venv` (`/opt/venv/bin/python3`) with numpy, scipy, OpenCV (`cv2`, headless), Pillow
and MuJoCo; the same environment runs your program at the venues. No network in your session. Nothing you are
given simulates the car or a venue; what you build for yourself with these tools is yours to judge.

## How you are graded (the grader runs a day at each hidden venue)

Gate (multiplies the score; a failed gate makes the whole run score 0): `drive.py` is a regular file in the
submission folder that defines `solve` at its top level; every file of the submission is a regular file and
together they are at most 60 MB; no file has a weights or model extension (`.pt`, `.pth`, `.onnx`, `.safetensors`,
`.ckpt`, `.h5`, `.pb`, `.pkl`, `.npz`); no `.py` file holds the text `torch.optim`, `.backward(` or `jax.grad`;
the program imports under the run regime above; at some venue at least one `step` is answered; no network
activity; no process of the program outlives its day at a venue; nothing of the submission was written after your
session ended. The code check is static and reads every `.py` file of the submission, imported or not
(byte-identical copies of the files of `carkit` are exempt). A file fails it when it does not parse, or when it
imports a module that starts processes (`subprocess`, `multiprocessing`, `pty`, `pexpect`, `sh`, `plumbum`,
`concurrent.futures.process`, or a process pool of `concurrent.futures`), a networking module (`socket`, `ssl`,
`http`, `urllib`, `urllib3`, `requests`, `httpx`, `aiohttp`, `ftplib`, `smtplib`, `telnetlib`, `poplib`,
`imaplib`, `nntplib`, `socketserver`, `xmlrpc`, `websocket`, `websockets`, `paramiko`, `asyncssh`, `grpc`, `zmq`)
or a foreign-function module (`ctypes`, `cffi`, `_ctypes`); when it calls a function of `os` that starts or
detaches a process (`os.system`, `os.popen`, `os.fork`, `os.exec*`, `os.spawn*`, `os.posix_spawn`, ...); when it
imports under a computed name or loads a module by path; or when it calls the built-in `exec`, `eval` or `compile`
with an argument that is not a literal (a method of the same name, such as `re.compile(...)`, is fine).

`race_day_score` is the mean over the venues of the venue score of the rules (clauses S-25 to S-30):

    venue = 0.35 * fast + 0.35 * cons + 0.30 * avoid

- `fast`: the fastest lap of the two heats that no crash and no pick-up interrupted;
- `cons`: the largest number of consecutive uninterrupted laps in one heat;
- `avoid`: the fastest clean lap of each of the two phases of the avoidance test, half of the part per phase.

Each part is scored against the organiser's reference run on the same venue: a reference team's car of the same
build, driven under the same rules by the organiser's own program, which works from a survey of the track as built
and of its floor and is told at every moment where its car is and, in the avoidance test, where the boxes and the
slow cars are. Full marks on a part are the level of that run on the same venue, with a small margin. The zeros of
`fast` and `cons` are that run's numbers scaled by how the middle of the field compared with the winner at the
league's rounds so far: the level of a mid-field team only if the reference run is as good as that winner. The
formulas are in the rules; the reference numbers of a venue are not published. There are no negative scores.
`race_day_score` already carries the gate; the reward is `race_day_score` on a scale on which a score a little below
the reference run's own level is full marks. Reported but not scored: touches, crashes, pick-ups and warnings per
session, laps per session, how far walls were pushed, program time, calls outside the interface.

## Rules of the day (in full in `rules.md`; the calls and messages in `car_api.md`)

* **Sessions.** `practice`, `tt1`, `tt2`, `avoid`, in this order, each starting with the car at rest on the same
  start position. A session can be closed early, none can be skipped or repeated. One process runs all four, so
  what the program learns in practice it still has in the avoidance test; nothing is kept between venues.
* **Mapping period.** In the first 120 s of practice the ground speed may not stay above 2.0 m/s for 0.5 s; each
  such spell is a warning, and the third warning ends the practice session.
* **Touch and crash.** The walls are ducts lying loose on the floor. A contact that moves a wall or a box by less
  than 5 cm is a touch, and so is any contact with a slow car; a touch costs nothing in practice and in the heats.
  A crash is a wall or a box pushed 5 cm or more (pushes of the same object add up), three or more wheels beyond
  the tape of an open wall, a rollover, or the whole car outside the track. After a crash the staff stop the car,
  carry it by hand to the side of the track near the last place where it drove freely, repair the track and
  release the car 20 s after the stop; the clocks keep running, and the lap is interrupted.
  `car.request_pickup()` asks for the same when the car is stuck.
* **Open walls.** Some walls have openings whose boundary is a tape line on the floor that the scanning sensor
  cannot see. The tape lies straight between the track-side edges of the two duct ends.
* **Laps.** A lap runs from one crossing of the timing line in the driving direction to the next, with the whole
  track driven in between. The organiser times it and reports it in the message `lap`.
* **Avoidance test.** Cardboard boxes stand on the track for the whole session; for its last 90 s one or two slow
  cars drive on it as well. A clean lap touches nothing, never stands still for more than 1 s and, in that last
  phase, overtakes at least one slow car.
* **Messages.** The organiser tells the car when a lap was timed, when the staff stopped it and released it, when
  it was warned, when a session entered its next phase and when it ended; never where the car is, what it
  touched or why it was stopped.
* Calls other than the five documented ones answer the error `forbidden` and are counted.

## No practice

There is no practice venue, no simulator of a venue and no answer key in your session: the first track your
program drives on is the first venue of the graded round, and its only practice there is the practice session of
the day. The protocol stub replays lab logs; it tells you whether a program speaks the protocol and keeps its
budgets, nothing about how it drives.

## Notes

- The lab logs and the photographs are meant to be processed programmatically; view only the images you need.
- Do not modify the given files in your home directory; the grader uses its own copies.
- Machine: your session's sandbox has 8 CPUs and 16 GB of memory, no GPU and no network. The grading machine has
  the same size, and there your program shares it with the venue's own computation.
- Environment facts (harness limits): a single tool call whose input runs above roughly 25-30k tokens is dropped by
  the harness (it is never executed and no error comes back), and a response that reaches the 64k output-token limit
  is lost and, after at most a few retries, ends the session with whatever is in `/home/user/submission/` at that
  moment. Write large files in several parts (one function group per write), keep every single response well below
  that limit, write a first complete submission early and refine it.
- Session time: this session is stopped after 4 hours of wall-clock time from its start. A session stopped by that
  limit is graded on whatever the submission folder holds at that moment, so a file left half-written then is what
  runs. Note the start time with `date` when you begin, keep a complete submission in place from early on, and end the
  session yourself (stop working and say that you are done) at least 15 minutes before the limit.
