# Build the simulation of a hazard bay, design its robot fleet, and program the fleet through the job

A maintenance bay in a process plant has to be rebuilt in simulation, in NVIDIA Isaac Sim, together with the robot
fleet that will work in it, so that the fleet's programs can be developed and tested before anything flies or drives
there. There is no model of the bay: you have photographs, a few measured facts, an account of how its valve and
its lifting rig behave, a list of what the fleet must be able to do, and a catalog of hardware to build the fleet from.
Deliver one USD stage that behaves like the real bay — the right things in the right places, the bucket on its three
ropes hanging, swinging and tipping as the real rig does, the hazards doing what the brief says — holding a fleet
you designed that meets the requirements; and deliver the program that runs that fleet through the plant's whole job
in the real bay (the valve closed, the fire put out with the bucket the three flying robots carry, the station cleared
by the vehicle), from what the robots' own sensors tell them: the control of every robot and the planning that
sequences them.

You have **8 hours** of wall time for the whole job. Have a usable scene and a program in
`/home/user/submission/` long before the time is up, and improve them from there.

## What you have

| path | what |
|---|---|
| `~/given/bay_brief.md` | the bay: what is in it, what happens there, how the valve and the rig behave, the world frame, the hazard rules |
| `~/given/requirements.md` | what the fleet must be able to do; the checks your design has to pass |
| `~/given/control.md` | the bay mission: your program's wire, what the robots sense, what they command, how it is scored |
| `~/given/facts.md` | the few measured numbers nobody can see in a picture (scale readings, the ropes, a nameplate) |
| `~/given/photos/` | photographs of the bay (`cameras.json`: the cameras' intrinsics; nothing else) |
| `~/given/catalog/` | airframes, sensors, ground vehicles and arms you may build the fleet from, with their data (`catalog.json`: the same, as data) |
| `~/given/interface.md` | how the grader reads your scene: stage settings, the prim paths it looks for, the ropes, the rules |
| `~/given/check_structure.py` | checks a scene against `interface.md` (paths, files, ropes, graph nodes); it scores nothing |
| `~/given/frame_reader.py` | reads the bay mission's sensor frames (`control.md`, The sensors); copy it into your submission to use it |
| `/opt/assets/isaac/` | a part of NVIDIA's Isaac Sim asset library, offline (robots, objects, environments) |
| `/opt/assets/arb/` | the task's own assets: four multirotor airframes; two NVIDIA Flow volume effects (a fire, a steam plume) for the pictures, if you want them |

Nothing in `~/given` runs the bay mission: rehearse your program in your own twin (`control.md` says what the grader
simulates and measures).

Software: Isaac Sim 6.2 and Isaac Lab 3.0 with Python 3.12 in `/opt/venv` (run Python as `/opt/venv/bin/python3`;
it is not on your `PATH`), PyTorch, NumPy, OpenCV, one NVIDIA GPU with RTX rendering, so you can render your own
scene and compare it with the photographs, as the grader renders your robots' sensors in the bay mission. Isaac Sim's
settings already point its asset root at `/opt/assets/isaac`. There is no network.

## What you deliver

In `/home/user/submission/`:

- `scene.usda` (or `scene.usd`): the bay and the fleet as one USD stage, following `interface.md`, together with
  every file it needs, referenced by relative path, plus whatever it references from `/opt/assets/`. Everything the
  bay does must be in the stage itself: geometry, physics (the ropes are chains of rigid bodies), and OmniGraph action
  graphs for the hazard rules. The grader runs none of your scene's code.
- `run.sh` and whatever it starts: your fleet's program for the bay mission, speaking the wire in `control.md`
  (`/opt/venv/bin/python3` with NumPy, OpenCV, PyTorch and `pxr` is there; the GPU is shared with the simulator's
  renderer). The grader runs it as a separate process, as your account,
  in a read-only copy of your submission folder, with no network. Grading happens in a fresh copy of this environment
  that holds only `/home/user/submission/`: `~/given` and anything else you leave in your home are not there, so put
  everything `run.sh` needs inside the submission (copy what you use from `~/given`); `/opt/venv` and `/opt/assets/`
  are there as now.

The submission may hold only regular files and directories, 2 GiB at most in all: a symbolic link or any other kind of
file in it, and the whole submission scores 0.

## How it is graded

After your session ends, every process of yours is stopped and the submission is copied. The grader checks the
scene against `interface.md` (as `check_structure.py` does; a scene that fails it scores 0), then opens it in its own
Isaac Sim process, several times, steps it at 120 Hz and measures it. Each check scores between 0 and 1 by how close
the measured value is to the real bay's or to what the brief requires. `acceptance_score` is the scene's score times
the mission's: the scene's score is the weighted mean of the first five sections below (the weights in brackets), the
mission's is the last section's score.

- **Layout (0.08)**: where the bay's furniture and fixtures are and how big they are, which objects stand on the
  tables and where (an object counts when it has the library model's shape), the fleet on its pads.
- **Physics (0.04)**: nothing moves when the scene starts; the objects' and the bucket's masses.
- **The steam line and the hazards (0.04)**: the grader turns the valve, carries robots into and out of the plume,
  drives the ground vehicle up to the fire, and reads the state attributes your logic maintains.
- **The rig (0.20)**: the grader holds the three flying robots and moves them through the moves of `requirements.md`
  R3 — slack ropes, the lift, a one-sided pull, a sideways step, one rope lowered, the pour over the pan, a dead
  stop, the set-down — and compares how the bucket and the ropes behave with the real rig under the same moves.
- **The fleet's design (0.10)**: the requirements in `requirements.md`, each checked from the stage and the catalog.
- **The mission (it multiplies the scene's score)**: your program runs the fleet, as dynamic bodies under its
  commands, through the whole job in the real bay (the grader's own stage of the bay as it really is, with your four
  robots in it as your stage builds them and the plant's rope rig), twice, on two days of its own; the grader renders
  what the robots' sensors see, keeps the bay's rules and measures only the outcomes: the steam released, the fire out, the parts in the crate, the time, the
  lift, the swing, the water spilt, the losses, the collisions, the separation, and how alike the two runs were
  (`control.md`).

The amounts, angles, points, directions and timings of these tests were drawn once at random and are kept hidden: they
are the same for every submission. So are the mission's two days, drawn once from the ranges `control.md` gives: the
water in the bucket, the wind, where and how the six parts lie on the tables, where the vehicle stands, how far the
valve's lever is turned and where the burning pan stands. Measurements come from the simulation itself (body poses,
masses, joint motion) and from the state attributes the interface declares, never from anything else the scene, the
controller or the plan reports about itself.
