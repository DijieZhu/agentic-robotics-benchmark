# Aerial search and geolocation

A ground team sends a quadrotor ahead with a short list of things to look for in the area
around it, for example *"a red fire truck"*, *"a round hay bale"*, *"a brown and white
horse"*. The drone has to fly out to each one, close enough to confirm it from right above
it, and send back where it is, so the team can go straight there. Some items on the list
may not be in the area at all, and saying so confidently is useful too.

Build a controller that flies these missions and leave it runnable at:

    /home/user/submission/run.sh

## How it is graded

The grader flies your controller through **24 missions**, 8 in each of three scenes **you
do not have here**: two city scenes and a tropical island. The two practice scenes in
this sandbox are a suburban park and a city downtown. Where targets stand, which they are
and where the drone starts are drawn at random at grading time; memorising this sandbox's
scenes does not help.

Each mission gives you **three target descriptions**, each of a different kind of thing.
Each target is either in the scene, at ground level **40 to 120 m** (horizontally)
from where you start, or it is a **decoy** that is not in the scene at all. The targets of
a mission lie in different directions from the start (at least 40 degrees apart as seen
from it) and 25 to 160 m from each other, so they are seldom all in view at once and
buildings or trees may hide them. A mission has at most one decoy, and about half of the
missions have none. Targets fit the scene: street vehicles, scooters, street furniture and
dogs in the city scenes; tents, a beach table, two-wheelers and farm animals on the island.
Each is a kind of thing the scene does not otherwise contain, so a description fits only
the target, and a decoy is a kind from the same scene's list. You start in open air at
least 10 m (usually 10 to 40 m) above the highest of the targets. Targets in the scene are
on the ground or up to about 0.7 m above it.

A mission ends when your controller sends **Stop**, when the drone **collides** with
anything (including the ground), or when it has spent its **budget of 1000 actions** (Stop
is free). Your **last report** for each target when the mission ends is final.

A target in the scene counts only if the drone **went down and looked at it**: at some point
during the mission the drone was within **3 m of it horizontally, no more than 6 m above
it**, and with the target in the down camera's view, that is, no farther from it
horizontally than the down camera (1 m under the drone) is above it. The grader measures
this from the true positions. Spotting it from afar is not enough.

Each target earns:

| your final report | target in the scene, visited | in the scene, not visited | decoy |
|---|---|---|---|
| found, within 3 m of it | 1 | 0 | 0 |
| found, 3 to 10 m from it | falls linearly from 1 to 0 | 0 | 0 |
| found, 10 m or more from it | 0 | 0 | 0 |
| declared absent | **-1** | **-1** | 1 |
| no report | 0 | 0 | 0 |

A mission scores the mean of its three targets. If that is positive, it is multiplied by
`1 - 0.2 * min(1, metres flown / 1000)` (energy), and halved if the drone collided. A
negative mission score is kept as it is. Your metric is the mean over the 24 missions
(floored at 0 for the whole run), compared with a reference controller's measured value
on the same kind of hidden missions (the reference value is not disclosed; full credit is
reached somewhat below its level). Guessing that targets are absent loses on average.

## What to build

`run.sh` starts a long-lived process that speaks one JSON object per line on
stdin/stdout. The grader launches it as `bash run.sh` with the working directory
`/home/user/submission`, under your own user account.

```
                                       <- READY   (your first stdout line, within 120 s)
{"type": "reset"}                      -> {"ok": true}    (a new mission starts)
{"type": "act", "t": 0, "obs": {...}}  -> {"action": [move, s1, x1, y1, z1, s2, x2, y2, z2, s3, x3, y3, z3]}
{"type": "close"}                      -> {"ok": true}, then exit
```

`action` is a flat list of 13 numbers: the move, then your current report for each target,
in the order of `obs["targets"]`:

- `move` is one integer:

  | code | move | effect |
  |---|---|---|
  | 0 | stop | end the mission now |
  | 1 | move forward | 1 m along the current heading |
  | 2 | move left | 1 m to the left |
  | 3 | move right | 1 m to the right |
  | 4 | move up | climb 1 m |
  | 5 | move down | descend 1 m |
  | 6 | turn left | yaw 15 degrees counter-clockwise (seen from above) |
  | 7 | turn right | yaw 15 degrees clockwise |

- for each target `s, x, y, z`: `s` is 0 (not located yet), 1 (found at `x, y, z`) or 2
  (declared absent; `x, y, z` ignored). `x, y, z` is the target's position in the
  **mission start frame**, the same frame as `obs["odom"]`: metres forward, right and down
  from where the mission started, along the starting heading.

You may also reply with just `[move]`: your previous report then stands. Every report
replaces the previous one, so keep sending what you know.

Moves are swept for collisions: a move that would pass through geometry ends the mission.
Keep stdout clean; it is the wire. Send diagnostics to stderr.

- **Deadlines.** `READY` within 120 s of launch (load your models first, then print it)
  and every reply within 30 s. A missed deadline, a crash, non-JSON output, or a reply
  that is not 1 or 13 valid numbers aborts the mission (scored 0). After an aborted
  mission the grader stops your process and launches `bash run.sh` again for the next one,
  so it must work when started a second time (no fixed ports, PID files or locks left
  behind); after three such restarts every remaining mission aborts.
- **Repeated `t`.** `t` is the step index, restarting at 0 after each `reset`. A step may
  be queried more than once with the same `t`; repeated queries do not advance the
  mission, so count steps by `t`, never by requests, and answer every request from the
  observation it carries, including its target descriptions. The grader checks that your
  reply depends on the targets you are given: on about 4 % of steps it asks the same `t`
  twice, once with the true observation and once with the same images but other target
  descriptions, in a hidden order, and applies only the reply to the true observation. If
  fewer than 25 % of these pairs get two different replies over the whole run, the run
  scores 0 regardless of its metric; a controller that ignores the targets, replays a
  sequence or returns constants fails this.
- **Aborted missions** score 0 and count in the mean. Grading itself (scene start-up and
  the 24 missions) has a 4-hour wall-clock budget.
- **No IP networking.** Your controller and everything it starts run without IP sockets,
  loopback included (`socket(AF_INET|AF_INET6)` fails with EACCES). Unix-domain sockets,
  pipes, files and the GPU work.
- One GPU is shared between the simulator and your controller.

## Observation

`obs` is a JSON object:

| key | content |
|---|---|
| `targets` | three `{"description": "..."}`, in natural language, in a fixed order for the mission |
| `rgb` | `{"front", "left", "right", "down"}`: base64 PNG, 8-bit RGB, 256 x 256 |
| `depth` | the same four keys: base64 PNG, 16-bit single channel, **centimetres** along the pixel's ray (65535 = no return, or 655.35 m or more) |
| `odom` | `{"position": [forward, right, down], "yaw": rad}`: your pose in the mission start frame (yaw clockwise-positive, 0 = the starting heading) |
| `steps_used`, `budget` | moves spent so far (Stop excluded) and the budget, 1000 |

The true positions of the drone and of the targets are never part of the observation.

Cameras (all pinhole, 90 degree field of view, square images, principal point at the
image centre, so the focal length in pixels is half the image width), all mounted 1 m
below the body's centre:

| camera | looks | image axes |
|---|---|---|
| front | along the heading | right = body right, down = body down |
| left | 90 degrees left of the heading | right = body forward |
| right | 90 degrees right of the heading | right = body backward |
| down | straight down | up = body forward, right = body right |

The body frame is forward / right / down; the drone flies level (moves never tilt it).

## Your sandbox

- **Interpreters.** `/opt/venv/bin/python3` is Python 3.12 with NumPy, SciPy, Pillow,
  OpenCV, PyTorch (CUDA) and Hugging Face Transformers. Call it by that absolute path in
  `run.sh`; a `python3` found on your shell's `PATH` may be a different interpreter.
  There is no network access, so nothing can be downloaded.
- **Models.** `/opt/models/clip-vit-base-patch16` (CLIP) and
  `/opt/models/owlv2-base-patch16-ensemble` (OWLv2 open-vocabulary detection), loadable
  with `from_pretrained(<path>)`. You may use them, combine them, or do without them.
- **Practice scenes.** `/opt/aerialsim/practice/pool.json` lists, for the two practice
  scenes, the points where targets may stand and the kinds of target (with their
  descriptions) used there. The held-out city scenes use city kinds from the same list
  (minus any kind their own scenery already has); the island uses tents, a beach table,
  two-wheelers, a pickup, a camper van and farm animals.
- **The simulator and the mission rules, for your own experiments.**
  `/home/user/aerial/sim_client.py` runs the same simulator the grader uses, with the
  ground truth visible; `/home/user/aerial/sar_protocol.py` holds the mission rules
  (assembly, observation, reply format, termination, scoring) shared with the grader:

  ```python
  import json, random, sys
  sys.path.insert(0, "/home/user/aerial")
  from sar_protocol import assemble_missions, load_pool, run_mission
  from sim_client import SimClient
  pool = load_pool("/opt/aerialsim/practice/pool.json")
  missions = assemble_missions(pool, random.Random(0), per_map=2)
  with SimClient(log_path="/home/user/sim.log") as sim:   # starts Unreal (slow, once)
      outcome = run_mission(sim, missions[0], my_controller_fn, budget=1000)
      print(outcome.record())
  ```

  `SimClient.reset(...)` / `.step("forward")` drive the drone directly if you want frames
  and ground truth for your own tests.
- **Practice grader.** `/opt/venv/bin/python3 /home/user/practice_grader/grade.py
  [--missions 3] [--seed 1]` runs your submission over the real wire, deadlines, socket
  restriction and repeated-`t` queries on practice missions, and prints the raw metric. It
  is not your score: the graded scenes are different.

## Time

Your session has a hard limit of **1 hour** of wall-clock time from its start (run `date`
now and keep track). Starting a scene takes a few seconds, and every simulated step costs
real time, so plan your experiments. Keep a working controller in
`/home/user/submission` as you go: whatever is there when the session ends is what the
grader runs.
