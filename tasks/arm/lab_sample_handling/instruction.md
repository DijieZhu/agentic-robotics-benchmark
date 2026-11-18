# Process a centrifuge worklist

You are a lab-automation engineer. A Franka Panda at a lab bench has to work through a worklist of capped 15 mL
sample tubes. It sees through cameras, feels with a wrist force sensor and can weigh on a bench scale. Write the
controller that does the job. It runs closed loop against a simulator and is graded on centrifuge units and
benches you have not seen. You have 2 hours of wall time.

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
Then close the lid.

Some positions are not yours to touch:
- other people's tubes in the cold rack;
- incoming tubes on neither list;
- the preloaded tubes.

Leave all of these where they are. The lid may start open or shut.

**What is hidden per unit.** The rotor turns on its hub one way only: a ratchet lets it turn counter-clockwise
(seen from above) and holds it against the other way. You turn it by hand, by gripping the knob bar on its hub
inside the shield and moving it along its arc. A hole carried past the port comes back only after a full turn.
How the rotor answers a turn is a property of each centrifuge unit:

- the hinge's friction;
- the play between the knob and the rotor;
- the rotor's inertia.

A grip commanded to turn 45 deg may turn the rotor by much less, and a short grip may not turn it at all.

**The practice units are not like the scored ones.** The three practice units in `units.json` sit at the low end
of each range. The scored units span the whole range, and most of them lie beyond anything the practice units
show:

| Property | Practice units | Scored units |
|---|---|---|
| Hinge friction | 0.10-0.15 N m | 0.10-0.45 N m |
| Knob play | 0-4 deg | 0-20 deg |
| Rotor inertia | 0.010-0.015 kg m^2 | 0.010-0.040 kg m^2 |

A controller tuned only on the practice units will meet rotors that answer a turn differently.

## What is in your home directory

| Path | What it is |
|---|---|
| `lab_sim.py` | The simulator of the bench: the world, the robot and its controller, the centrifuge, the racks, the scale, the tubes, the cameras, and each episode's draw. The grader runs this exact file. |
| `lab_env.py` | The environment your controller talks to: `LabEnv(unit).reset(seed)`, `step(action)`, the observation. The grader runs this exact file. |
| `episode.py` | An episode and its credit. The grader runs this exact file. |
| `box_transport_env.py` | The robot and table definitions `lab_sim.py` builds on. |
| `units.json` | Three practice units (p0-p2): rotor properties to develop against (the low end of the ranges). |
| `task_card.md` | The bench's numbers in one place: nominal places, ranges, sizes. |
| `practice_grader/grade.py` | Runs your controller over the wire, as the grader does, on the practice units; prints the raw credit. |
| `helpers/wire_skeleton.py` | Optional boilerplate for the wire (no control logic). |

Python is `/opt/venv/bin/python3` (MuJoCo 3.16, numpy, scipy, OpenCV, Pillow). There is no network. The sandbox has
4 CPUs and 12 GB of memory, while grading too. Every `/opt/venv` Python runs NumPy's linear algebra on one fixed set
of kernels (`OPENBLAS_CORETYPE=Haswell`, which the image also sets when the variable is missing), here and in the
graded run, so the simulator computes the same on every machine.

## Credit

Each episode's credit (0 to 1) is computed from the simulation when the episode ends:

- the cold tubes, each seated in its named position: 0.2 in all, split evenly;
- the rotor's pairs (as many as half the spun and preloaded tubes): 0.6 in all, split evenly. A pair counts when
  two opposite positions hold two of those tubes, both seated at the bottom of their holes, their masses within
  1 g;
- the lid shut over a complete rotor (every pair counted): 0.2.

If a tube that was not yours to move has moved, the episode scores 0. A tube anywhere else counts nothing. So
does an unbalanced pair, a pair with an empty position or with a cold tube, and a lid shut over an incomplete
rotor.

An episode ends:
- after 20000 control steps (33 simulated minutes);
- 20 steps after the lid is shut, once it has been open (its angle under 0.02 rad);
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
sandbox: nothing else of your home directory is there (no `lab_sim.py`, no `units.json`: copy what you use into the
folder), there is no desktop session, and the only processes of your account are the ones `run.sh` starts. A
process of your account that was started any other way ends the run: that episode and every later one score 0.

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
  the budget is used up, the episode under way and the episodes not yet started score 0. The practice grader reports
  the same sum (`controller_seconds`).
- Your controller is one process for the whole run, so it may carry what it learns about a unit, or a rotor, from
  one episode to the next.
