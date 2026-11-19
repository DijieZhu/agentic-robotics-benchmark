# Process a centrifuge worklist

You are a lab-automation engineer. A Franka Panda at a lab bench has to work through a worklist of capped 15 mL
sample tubes. It sees through cameras, feels with a wrist force sensor and can weigh on a bench scale. Write the
controller that does the job. It runs closed loop on the bench and is graded on centrifuge units and benches you
have not seen. The bench is simulated, and the simulator is not in your sandbox: you have the equipment's sheets
and a commissioning window on three other centrifuges of the same series. You have 2 hours of wall time.

**The bench.** Nothing stands in exactly the same place twice:
- the incoming rack (one row, positions A1-A10);
- the cold rack (3 x 3, A1-C3);
- the centrifuge (its cabinet, its lid, its eight-position rotor and the shield over the rotor);
- the scale (one tube holder on a balance).

Each episode places each of these within a few centimetres of a nominal place, turned by up to 10-20 deg. The
rotor's hole circle and height also vary. `task_card.md` gives the nominal places and the ranges; your cameras
have to find the real ones.

The tubes are opaque and all look alike: nothing shows how full a tube is, and a tube is known only by where it
stands. 1 mL of sample weighs 1 g.

**The job.** Each episode's worklist names:
- `cold`: two or three incoming tubes and the cold-rack position each goes to, e.g. `{"A3": "B2", "A7": "C1"}`;
- `spin`: the incoming tubes to centrifuge;
- `preloaded`: zero to two tubes already seated in the rotor from another job, by rotor position, with their
  masses, e.g. `{"3": 22.41}`. They are spun too.

Load the rotor so that the spun tubes and the preloaded ones stand in opposite pairs, each pair's two masses
within 1 g. Positions k and k + 4 are opposite; the positions are numbered 1-8 counter-clockwise from the one
marked with a green dot. Such a pairing always exists, but the masses are known only by weighing. Which pair goes
to which positions is up to you, except that the preloaded tubes stay where they are. A fixed shield covers the
rotor's holes except at one loading port, so a tube goes in only at the hole the rotor has brought to the port.
Then close the lid: the job is the centrifuge run, and a rotor under an open lid cannot be started.

Some positions are not yours to touch:
- other people's tubes in the cold rack;
- incoming tubes on neither list;
- the preloaded tubes.

Leave all of these where they are. The lid may start open or shut.

**What is hidden per unit.** The rotor turns on its hub one way only: a ratchet lets it turn counter-clockwise
(seen from above) and holds it against the other way. You turn it by hand, by gripping the knob bar on its hub
inside the shield and moving it along its arc. A hole carried past the port comes back only after a full turn.
How the rotor answers a turn is a property of each centrifuge unit, and no unit's values are told:

| Property | Range over the series |
|---|---|
| Friction of the rotor's hinge | 0.10-0.45 N m |
| Play between the knob and the rotor | 0-20 deg |
| Rotor inertia | 0.010-0.040 kg m^2 |

A grip commanded to turn 45 deg may turn the rotor by much less, and a short grip may not turn it at all. The six
scored units and the three commissioning units were drawn separately over these ranges: a commissioning unit shows
what the series is like, not what a scored unit's values are.

## Commissioning

There is no simulator of the bench in your sandbox, and no model of it other than what `task_card.md` states. What
you can do before you hand in is what an integrator can do before a method goes live: work on a spare bench for a
limited time.

- **The benches.** A commissioning service holds three development benches, units `d0`, `d1` and `d2`: other
  centrifuges of the same series (above), none of them a scored unit, with worklists of their own (none of them a
  scored episode's). It runs under another account, and its files are closed to yours. You reach it through its
  socket only, `/run/labbench/commission.sock`; `commission/client.py` is a client for it.
- **What it gives.** The observation a controller gets when it is graded (below), and nothing else: no pose of
  anything, no mass, no fill, no rotor angle, no unit property, no credit, and no word on how a worklist went. What
  happened on the bench you judge as your controller must: from the cameras, the scale and the wrist.
- **Its requests.**
  - `reset(unit, episode)` stages a bench: development worklist `episode` (0 to 4999) of that unit. The same
    unit and number stage the same bench again (the same places, tubes, fills and noise).
  - `step(action)` runs one control step and returns the observation.
  - `look(cams)` returns the cameras' images now, without a step.
  - `probe()` returns a perturbed copy of the last observation, as the grader's repeated requests carry (below).
  - `status()` returns the window's count.
- **How an episode ends there.** As a graded one does: after 20000 steps, 20 steps after the lid is shut, or at an
  unstable simulation. It does not end because a worklist is complete, and the service does not say whether it is.
- **The window.** 144,000 control steps in all, which is 4 hours of bench time at 10 Hz (a spare bench for half a
  working day; a graded episode is at most 20,000 steps). Staging a bench costs 300 steps (half a minute to set
  it up). The count is kept by the service and survives reconnects; `/run/labbench/commission_status.json` shows
  it. Once it is used up, every `reset` and `step` answers `commissioning_window_used`. Nothing gives steps back.
- **Benches at a time.** Two connections, each with one bench; a third is answered `busy`. A connection silent for
  120 s is closed. The benches run on your sandbox's CPUs; a control step takes about 30 ms of wall time
  (more when images are drawn), on one bench as on two.
- **At grading** the service is not there: your controller meets the scored benches with what is in its folder.

`commission/run_bench.py` runs your submission folder on a development bench the way the grader runs it (a fresh
copy, `bash run.sh`, the grading wire with its repeated requests and its deadlines). It reports how the episode
ended, the steps, your controller's time against the grading budget and the repeated-request check; it cannot say
how the worklist went. `--images DIR` saves what the cameras show at the start and at the end.

## What is in your home directory

| Path | What it is |
|---|---|
| `task_card.md` | The equipment sheets: the robot and its control, the cameras, the tubes, the racks, the centrifuge, the scale; sizes, nominal places, and the range of everything that varies. |
| `commission/client.py` | Client of the commissioning service (`Bench`: `status`, `reset`, `step`, `look`, `probe`, `close`). |
| `commission/run_bench.py` | Runs your submission on a development bench over the grading wire. |
| `helpers/wire_skeleton.py` | Optional boilerplate for the wire (no control logic). |

Python is `/opt/venv/bin/python3` (MuJoCo 3.16, numpy, scipy, OpenCV, Pillow; robosuite 1.5.2's model files, the
Panda's among them). There is no network. The sandbox has 4 CPUs and 12 GB of memory, while grading too. Every
`/opt/venv` Python runs NumPy's linear algebra on one fixed set of kernels (`OPENBLAS_CORETYPE=Haswell` with
`OPENBLAS_L2_SIZE=256`, which the image also sets when the variables are missing), here and in the graded run.

## Credit

Each episode's credit (0 to 1) is computed from the simulation when the episode ends. The job is the centrifuge
run, so the credit is the run's.

A tube is **seated** in a hole (of a rack or of the rotor) when its tip is inside the bore and within 6 mm of the
hole's floor. The **run's tubes** are the worklist's `spin` tubes and the preloaded ones. A **balanced pair** is
two of the run's tubes seated in opposite rotor positions, their masses within 1 g.

1. **The run counts only if the centrifuge could be started on it.** All three must hold:
   - the lid is shut (its angle under 0.02 rad);
   - every tube seated in the rotor is half of a balanced pair;
   - the rotor holds at least one balanced pair.

   So a run does not count with the lid open, with a tube that has no partner opposite it (a preloaded tube whose
   partner has not been loaded is one), with a pair more than 1 g apart, or with a tube in the rotor that is not
   one of the run's (a cold tube, say).
2. **A run that counts has a share:** the balanced pairs in the rotor divided by the pairs the worklist needs
   (half the number of the run's tubes). A complete run has share 1.
3. **The credit is the share times (0.8 + 0.2 x the fraction of the cold tubes seated in their named cold-rack
   positions).** A complete run with every cold tube in place scores 1, a complete run with no cold tube in place
   0.8, and two of three pairs under a shut lid with every cold tube in place 0.67.
4. **A run that does not count scores 0**, whatever else was done. Cold tubes in their places are worth nothing
   without a run.
5. **If a tube that was not yours to move is no longer seated where it started, the episode scores 0.**

A tube anywhere else counts nothing.

An episode ends:
- after 20000 control steps (33 simulated minutes);
- 20 steps after the lid is shut, once it has been open (its angle under 0.02 rad, after having been over
  0.5 rad);
- at an unstable simulation (an arm joint faster than 8 rad/s), which scores 0.

The score is the mean credit over the graded episodes.

## Observation and action

The controller runs at 10 Hz. The observation is a JSON object:

- **Robot:** `t` (step), `unit` (the unit's id), `q` and `dq` (7 joint angles and velocities), `ee_pos` and
  `ee_quat` (the gripper site, wxyz), `gripper` (the fingers' opening, m), `wrench` (the wrist's force and torque
  in the world frame, with noise: 0.3 N, 0.03 N m).
- **The scale:** `scale_g`, its display in grams (see `task_card.md`).
- **Cameras:**
  - `T_world_<cam>`: 4 x 4 pose in the OpenCV convention (x right, y down, z forward);
  - `K_<cam>`: intrinsics, a pixel's centre at its index + 0.5;
  - `<cam>`: the image, `{"dtype": "uint8", "shape": [H, W, 3], "b64": ...}`.

  The four cameras are `left` and `right` (320 x 240, fixed, over the bench), `wrist` (256 x 192, on the hand)
  and `rack` (640 x 480, fixed, facing the incoming rack). An image is empty (`[0, 0, 3]`) unless the previous
  action asked for it; the first observation has all four. There is no depth image.
- **The worklist:** `worklist`, `{"cold": {incoming position: cold position}, "spin": [incoming positions],
  "preloaded": {rotor position: grams}}`.

The action is a JSON list in one of three shapes:

- 8 numbers `[dx, dy, dz, drx, dry, drz, grip, cams]`: the operational-space controller moves the gripper site's
  target by (dx, dy, dz) m (clipped to 0.03 m) and turns it by the rotation vector (drx, dry, drz) rad (clipped to
  0.2 rad), both in the world frame. `grip` runs from -1 (fully open, 80 mm) to +1 (closed).
- 15 numbers: the same, then seven joint angles. These are the posture the controller's null space pulls the arm
  toward from now on (the start posture until you set one).
- 9 numbers `[q1..q7, grip, cams]`: a joint step. A gravity-compensated joint PD moves toward q, each joint's
  target at most 0.1 rad from its current angle. q also becomes the null-space posture.

`cams` is a bitmask of the images the next observation carries: 1 left, 2 right, 4 wrist, 8 rack. Images are
large (the rack image alone is about 1.2 MB on the wire), so ask for one only when you need it.

## Deliverable and grading

Put your controller in `/home/user/submission/` with a `run.sh`. Everything it needs must be inside that folder (at
most 5000 files and 200 MiB; a larger folder scores 0), and the controller may write there. It is graded in a fresh
sandbox: nothing else of your home directory is there (no `task_card.md`, no `commission/`: copy what you use into
the folder), the commissioning service does not run, there is no desktop session, and the only processes of your
account are the ones `run.sh` starts. A process of your account that was started any other way ends the run: that
episode and every later one score 0.

The grader starts the controller with `bash run.sh` in that folder, as your user, with only local (AF_UNIX) sockets,
and speaks one JSON object per line on its stdin and stdout:

- your controller prints `READY` within 120 s;
- `{"type": "reset"}` → `{"ok": true}` (an episode starts);
- `{"type": "act", "t": t, "obs": {...}}` → `{"action": [...]}`, within 30 s;
- `{"type": "close"}` → `{"ok": true}`, then exit.

Stdout carries only these messages; write logs to stderr. `run.sh` starts with only `PATH`, `HOME` and `LANG` set:
export anything else you need there (to render with MuJoCo, `MUJOCO_GL=osmesa`).

**Probes.** About 1 % of steps are asked twice with the same `t`: the real observation and a slightly perturbed copy
(joints by up to 1 mrad with the poses recomputed, force by up to 0.2 N; images and the scale unchanged), in a
hidden order. Only the reply to the real one is applied. A pair in which both replies command no motion (8 numbers whose
first six are all zero, within 1e-9) is not counted either way. If fewer than 10 % of the other pairs get replies that
differ (by more than 1e-4 in some number), the whole run scores 0: a controller that ignores its input, or answers a
repeated `t` from a cache, fails. Which steps are probed depends only on the episode, not on your replies.

**Grading.**
- There are 12 episodes: 2 for each of 6 hidden units, a fixed hidden set (each episode's seed is the same for
  every submission, so are its bench, worklist, fills, rotor start angle, lid start and noise), in a hidden order
  (the observation names the unit).
- Before the first episode the grader draws one test image of each camera, which takes a few seconds of its budget.
- A crash, a late reply, a malformed action or an unstable simulation scores the episode 0. After 3 restarts the
  remaining episodes score 0.
- Your controller's time budget is 9000 s in all: the time it takes to print `READY` and to answer each request,
  summed over the run, plus 0.2 s for every image you ask for (a fixed price, whatever the grader's machine takes to
  draw it; the four images of an episode's first observation are free). The simulation itself is not counted. Once
  the budget is used up, the episode under way and the episodes not yet started score 0. `commission/run_bench.py`
  reports the same sum for one episode (`controller_seconds`).
- Your controller is one process for the whole run, so it may carry what it learns about a unit, or a rotor, from
  one episode to the next.
