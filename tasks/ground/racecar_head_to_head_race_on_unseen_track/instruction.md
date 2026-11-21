# Race a tenth-scale autonomous car head-to-head at venues it has never seen: map the track in practice, then start last on a four-car grid in two heats and overtake cleanly

You are the software engineer of a small team in a regional indoor league for tenth-scale autonomous race cars.
The team's car is a four-wheel-drive touring-class chassis with an autonomy deck. It senses with a 2D scanning
LiDAR, an inertial unit and the speed of its motor; it has no camera. A round of the league goes from hall to
hall. At each venue a temporary track, walled with flexible air ducts, lies on the bare floor of the hall, and the
car does, unattended, what every team's car does on a head-to-head day: a practice session in which it maps a
track it has never seen, and two heats in which it races three other cars on that track. In both heats it starts
at rest from the last slot of the grid, behind the three others. Nobody touches the car or its program from the
first session to the last. Your job is the program that drives it.

Nothing gives the answer directly. There is no drawing of any track, no map, no position of the car or of any
other car, no value for the grip of a floor, and no simulator of the car, of another car or of a venue among your
files. You have what a team of this league has: the league's rules, its guide to the venues of past seasons with
photographs, its scrutineering photographs of a car of the league, the car's datasheet, logs of the car driven in
the team's lab, and the client library through which a program talks to the car. Every venue of the round is
different: another hall floor with another grip, another layout, other places where the floor differs locally,
where a wall is open or dark, and where things stand beside the track; and in every heat three other cars on lines
and at speeds of their own.

## What you produce (in `/home/user/submission/`)

`drive.py` exposing `solve(car) -> None` (plus any helper modules and small data files of your own). The grader
runs it at several hidden venues, once per venue, each time as a new process that lives through the whole day at
that venue: a closed practice of 480 s whose first 120 s are a mapping period with a speed limit of 2.0 m/s, and
two heats of at most 300 s each against three other cars. The program talks to the car through `car`, a client of
five calls (`/home/user/given/interface/car_api.md`). Each `car.step(steer, speed)` moves the day on by one tick
of 25 ms and returns what the car's sensors gave in that tick (one revolution of the scanning sensor, 1081 ranges;
five inertial samples; the speed from the motor; the steering echo) and the organiser's messages. No call returns
a position, a map, what the car touched, anything about another car or a score: the other cars are in the scan,
and nowhere else.

Budgets of the program at a venue (`car_api.md`, section 6): on the track 15 ms of its own time per tick, summed
over a session, which is 288 s in practice and 180 s in a heat however early the heat ends; at the desk before the
three sessions 60 s, 300 s and 120 s; 120 s from its start to its first call; 180 s of silence at most; CPU time
at most 4 times these wall-clock budgets. A program that exceeds a budget, raises an error or falls silent ends
its day at that venue with what it had achieved; the other venues are still run. A heat that is cut short in this
way is judged as it stands, and a heat that is never opened scores zero.

How the program is run: a fresh process per venue under a dedicated unprivileged account that is not yours, in a
read-only copy of your submission folder (regular files only), with an empty writable HOME / TMPDIR, a minimal
environment and the launcher of the given library (`python -B -E -s`, your folder as the working directory, `carkit`
importable). The numeric libraries run single-threaded. A process of the program may use at most 12 GiB of address
space and write files of at most 1 GiB. It cannot start another process: every attempt fails and no process is made.
It cannot open a network socket: creating a socket other than a Unix-domain one ends the process at once and fails
the gate below. The given files are not there at a venue: whatever the program needs from them must be in the
submission folder. Write only under the run's HOME / TMPDIR; what the program leaves elsewhere is removed after
every venue. Before the first venue the grader imports `drive.py` once under the same regime; the import must finish
within 120 s and 120 s of CPU time. A venue at which the grader's own simulation fails is run once more from its
beginning. If it fails there a second time, the gate below fails. Your submission must be final when your session
ends: only `/home/user/submission` is carried over to the grading, a submission entry written after that time fails
the gate below, and so does a process of your program that outlives its day at a venue.

## What you are given

| path | content |
|---|---|
| `/home/user/given/league/rules.md` | the league's rules of a head-to-head day, clauses S-01 to S-38: the track, the three sessions, touch and crash, what the staff do after a crash, the heats (laps, grid and start, the other cars, standing, overtakes, contact and fault, warnings, stops), the scoring with its formulas, the organiser's messages |
| `/home/user/given/league/venues.md`, `league/photos/` | the league's guide to its venues: what a venue sheet says and how far each entry can be trusted, floors and the range of grip, local patches of the floor, kinds of walls, gaps, open walls, shared walls, the track's range of sizes, what stands beside the track, the heats, the scrutineering photographs; 11 photographs of earlier venues with captions in the guide. None of them is a venue of your round. Also 5 scrutineering photographs of a car of the league, with scale references in the picture. They are the only place where the size of the marker box and its place on the car can be found |
| `/home/user/given/car/datasheet.md`, `car/meshes/` | the car: frames, geometry and mass, steering, drive, the three sensors with their rates and limits, timing; its outer geometry. It gives no dynamics model: no model file, no tyre curve, no controller gains |
| `/home/user/given/lab_logs/` | `README.md` and 14 logs (`*.jsonl`, about 60 MB) of this car driven in the team's lab, on a floor that grips more than any venue floor: every command sent, everything the sensors returned in the same form as at a venue, and the car's pose from the lab's overhead tracker. The car is alone in every one of them |
| `/home/user/given/interface/car_api.md` | the program interface: how a program is run, the five calls, `describe()`, the observation, the messages, the budgets, the errors |
| `/home/user/given/interface/carkit/` | the client library (standard library only; uses numpy when present): the `Car` client, a decoder of the messages, the beam geometry of the scan, a reader of lab logs |
| `/home/user/given/interface/stub/stub.py` | a protocol stub: it starts a program the way a venue does and answers it with the rows of one lab log. It checks the protocol and the timing of a program and simulates nothing |

Tools: Python 3.12 in `/opt/venv` (`/opt/venv/bin/python3`) with numpy, scipy, OpenCV (`cv2`, headless), Pillow
and MuJoCo; the same environment runs your program at the venues. No network in your session. Nothing you are
given simulates the car, another car or a venue; what you build for yourself with these tools is yours to judge.

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

`head_to_head_score` is the mean over the venues of the venue score of the rules (clauses S-34 to S-37). The venue
score is the mean of its two heats, and a heat scores

    heat = classified * conduct * (0.5 * place + 0.5 * pass)

- `classified`: 1 when, at the end of the heat, the car has completed at least `N - 1` laps and has been given
  fewer than three warnings; otherwise 0, and the heat scores zero. `N` is the number of laps of a heat on the
  venue sheet (`heat_laps` in `describe()`); at past venues it lay between 7 and 18. A heat
  ends when the first of the four cars completes `N` laps, and after 300 s at the latest;
- `conduct`: `max(0, 1 - K / 3)`, with `K` the warnings of the heat. Every crash of the car and every car crash at
  its fault is one warning; the third ends the heat, which then scores zero;
- `place`: the number of other cars beaten, divided by 3. One of the other cars is beaten when the car has
  overtaken it cleanly at least once more than it was re-overtaken by it, and it is behind the car at the end;
- `pass`: `clip(C / (C_ref - 1), 0, 1)`. `C` is the number of the car's clean overtakes minus the number of times
  it was re-overtaken, over the whole heat and all three cars; `C_ref` is the same number of the organiser's
  reference run in the same heat of the same venue.

The reference run is a reference team's car of the same build, driven under the same rules from the same slot of
the same grids against the same other cars by the organiser's own program, which works from a survey of the track
as built and of its floor and is told at every moment where its car and every other car are, and the line each of
the other cars drives. Full marks on `pass` are one overtake short of that run's balance; `C_ref` is not
given to you. In every heat of every venue that run's balance is at least 5: it lapped cars. The organiser set `N`
with that run in view: in both of its heats it completed the `N` laps within 240 s of the 300 s a heat may last.
No part of the score reads a lap time: a car that is fast on an empty track and overtakes nobody scores zero. The
formulas are in the rules. There are no negative scores. `head_to_head_score` already carries the gate; the reward
is `head_to_head_score` on a scale on which a score a little below the reference run's own level is full marks.
Reported but not scored: touches, crashes, pick-ups and warnings per session, laps per session, how far walls were
pushed, program time, calls outside the interface.

## Rules of the day (in full in `rules.md`; the calls and messages in `car_api.md`)

* **Sessions.** `practice`, `heat1`, `heat2`, in this order. Practice starts with the car at rest on the start
  position, a heat with the car at rest in the last slot of the grid. A session can be closed early, none can be
  skipped or repeated. One process runs all three, so what the program learns in practice it still has in the
  heats; nothing is kept between venues.
* **Mapping period.** In the first 120 s of practice the ground speed may not stay above 2.0 m/s for 0.5 s; each
  such spell is a warning, and the third warning ends the practice session.
* **Grid and start.** Four slots lie one behind the other along the track, 1.5 m apart, the first level with the
  start position of practice; the rear slots can lie in the bend before the start straight. The three other cars
  stand in the first three, the fastest in front, each on the line it is going to drive. The car stands in the
  last slot, 4.5 m behind the first along the track and 0.15 m to one side of the track's centreline, which need
  not lie midway between the walls. There is no start signal: the clock of the heat runs from its first tick, and
  the other cars set off at once.
* **The other cars.** Cars of the same build as yours, each with the league's marker box of bright cardboard
  standing upright at its rear; the rest of such a car lies below the scan plane but for its dark sensor head. A
  scan shows little of such a car but its marker box, and the car is larger than its box: how large the marker box
  is, where it stands on the car, and how far the car reaches ahead of the box and beyond its sides is printed in
  none of your files. These facts are only in the league's scrutineering photographs (`league/photos/`;
  `venues.md`, section 9), which carry scale references of stated sizes. Each drives a fixed line of its own, lap
  after lap, at a fixed fraction of the race pace of that line: the speed at which a car of this build can drive
  the line on the floor as it is, so it is lower where the floor grips less. The fractions lie
  between 0.45 and 0.75; neither they nor the lines are published. Such a car does not make way,
  does not defend and does not change its line for a car beside it or behind it. It slows down, and stops if it
  has to, for something close ahead on its line; behind a slower car it follows and does not overtake. Its own
  staff put it back when it is pushed off its line or stuck.
* **Overtakes.** They are counted along the track at the reference points (the midpoint of the wheelbase). The car
  overtakes another car when, coming from behind, it leads it by at least half a car length, 0.194 m; it is
  re-overtaken when that car does the same to it, whatever the cause, also while the car is stopped by the staff.
  Laps count: lapping a car is an overtake. An overtake is clean when the two cars are in no contact from 3 s
  before it to 2 s after it, the other car is not being handled by its staff in that time, and the car is on the
  track and not stopped by the staff. A lead that comes from the staff setting one of the two cars down is not a
  clean overtake either. An overtake that is not clean counts for nothing, and letting that car by to overtake it
  again only brings the count back to where it was: a car first overtaken with contact counts as beaten only once
  the car has lapped it cleanly.
* **Touch and crash.** The walls are ducts lying loose on the floor. A contact that moves a wall by less than
  5 cm is a touch and costs nothing. A crash is a wall pushed 5 cm or more (pushes of the same wall add up), three
  or more wheels beyond the tape of an open wall, a rollover, or the whole car outside the track. After a crash
  the staff stop the car, carry it by hand to the side of the track near the last place where it drove freely,
  repair the track and release the car 20 s after the stop; the clocks keep running, and the other cars drive on.
  In a heat every crash is a warning as well. `car.request_pickup()` asks for the same stop when the car is stuck;
  it brings no warning. It takes none away either: a crash before the staff have set the car down, and a contact
  with another car begun before then and judged a car crash at the car's fault, are warned all the same.
* **Contact with another car.** Contacts with the same car less than 1 s apart are one contact. It is a car crash
  when, within 1.5 s of the first touch, the other car is 0.3 m or more (about one and a half car widths) to the
  side of where it would have driven had it been left alone, or when its staff have to put it back before 1.5 s
  have passed since the last touch; otherwise it is a touch. What that car does of its own accord, such as
  swinging across its line in a bend, is no push. The fault is the car's, with two exceptions; in both, at the
  first touch, the car is more than half a car length ahead of that car on the track and has completed no overtake
  of it in the 1 s before: the car had overtaken that car, or that car had driven further in the heat and was
  coming up to lap the car. A touch brings no warning, but an overtake with the contact in its interval is not
  clean. A car crash at the car's fault is a warning and a stop of 20 s, after which the car is set down at least
  1.5 m behind that car.
* **Standing still.** In a heat a car that has moved 0.5 m or more from its place on the grid and then stays below
  0.05 m/s for more than 3 s is picked up by the staff (a stop of 20 s, no warning), unless it is stopped already
  or a standing car is within 2 m ahead of it; the 3 s begin again from zero once neither is the case.
* **Open walls.** Some walls have openings whose boundary is a tape line on the floor that the scanning sensor
  cannot see. The tape lies straight between the track-side edges of the two duct ends.
* **Laps.** A lap runs from one crossing of the timing line in the driving direction to the next, with the whole
  track driven in between. All four cars start behind the line, so the first crossing completes no lap. The
  organiser reports the car's own laps in the message `lap`.
* **Leaving the track.** The third time at a venue that the whole car is outside the track, the session in
  progress ends at once and the sessions that remain at that venue are void.
* **Messages.** The organiser tells the car when a lap of its own was timed, when the staff stopped it and
  released it, when it was warned, when practice became free and when a session ended; never where the car is,
  what it touched, why it was stopped, or anything about another car.
* Calls other than the five documented ones answer the error `forbidden` and are counted.

## No practice

There is no practice venue, no simulator of a venue, no other car and no answer key in your session: the first
track your program drives on is the first venue of the graded round, its only practice there is the practice
session of the day, and the first cars it meets are those of heat 1. The protocol stub replays lab logs; it tells
you whether a program speaks the protocol and keeps its budgets, nothing about how it drives.

## Notes

- The lab logs and the photographs are meant to be processed programmatically; view only the images you need. The
  scrutineering photographs are there to be measured: each holds scale references whose sizes the venue guide
  gives.
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
