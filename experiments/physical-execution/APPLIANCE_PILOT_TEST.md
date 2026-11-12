# Appliance primitive feasibility pilot

Pilot for the proposed multi-appliance tending task. It asks one question: can a
Panda on the Omron base physically operate RoboCasa's appliance controls through
ordinary actions. It is not a benchmark task.

## Scope

`appliance_primitives_pilot.py` runs four primitives, each in its own fresh
episode with no task objects:

- `stove_knob`: pinch one stove knob from above and turn it until RoboCasa's
  `Stove.is_burner_on` holds.
- `microwave_start`: press the start button with closed fingertips, then
  withdraw. Judged by `Microwave.get_state()["turned_on"]` and
  `Microwave.gripper_button_far`.
- `oven_door`: open the oven door from fully closed. The hand drags the handle
  outward from above; past a few degrees the door's own weight opens it. Judged
  by `Fixture.is_open` (at least 0.90 of the joint range).
- `oven_rack`: hook the rack's front lip with closed fingertips and slide the
  rack out to at least 0.95 of its travel (the threshold of RoboCasa's
  `SlideOvenRack`). The scene starts with the door open, as that task does.

Every outcome must hold for 20 consecutive control steps with the hand at least
0.10 m from the control.

The controller is a privileged scripted reference: it reads fixture geometry,
joint values and contacts from the simulator. No joint is set and nothing is
teleported after reset. The mobile base is moved through base actions, using the
velocity mapping of `auto_pick_place.align_base`.

Seeds vary RoboCasa's default base-placement deviation (0.15 m, 0.05 m) and
arm-joint noise. `--fixed-spawn` removes both; every seed is then the same
episode.

## Run

```sh
conda activate arb-robocasa
cd "/Users/dijie/Desktop/ALE Robotic Benchmark/simulator-lab"
python -I appliance_primitives_pilot.py --primitive all --seeds 0 1 2 3 4
python -I appliance_primitives_pilot.py --primitive oven_door --seed 0 --layout 2 --style 1
```

The default layout is 2. The oven primitives need a layout with an `Oven`
fixture; RoboCasa lists the 45 layouts without one in
`Kitchen.OVEN_EXCLUDED_LAYOUTS` (layout 1 is among them). Every layout has a
microwave and a stove or stovetop.

Each trial writes `outputs/appliance_<primitive>_seed<k>_<time>/` with
`result.json`, `trace.jsonl` and a snapshot of the script. With rendering it also
writes the initial camera images and a video. Failed trials are kept.

`--no-render` runs physics only. On Linux it also keeps MuJoCo from loading an
OpenGL backend, so it runs where no EGL or OSMesa library is installed.

## Results (2026-11-12, physics only)

Run in a Linux aarch64 virtual machine with `--no-render`, Python 3.11.17,
MuJoCo 3.3.1, the lab's RoboCasa `3443cdf` and robosuite `5ce6643` sources.
Script SHA256 `81ec148fad7a8f062e990bd5247cb272c1463bb0cbe877953613e135509e68d4`.
All 68 trials are in `outputs/appliance_pilot_headless_20261112/` with
`summary.md` and `summary.json`.

| Phase | Scenes | stove_knob | microwave_start | oven_door | oven_rack | Total |
| --- | --- | --- | --- | --- | --- | --- |
| Development | layout 2, style 1, seeds 0-4 | 5/5 | 5/5 | 5/5 | 5/5 | 20/20 |
| Holdout | layout 2, style 1, seeds 100-104 | 5/5 | 5/5 | 5/5 | 5/5 | 20/20 |
| Generalization | layout 2 styles 2-5; layouts 4, 7, 9 style 1; seed 0 | 5/7 | 6/7 | 6/7 | 0/7 | 17/28 |

The controller was tuned on the development seeds only, then frozen. Holdout and
generalization trials were each run once with the frozen script.

Passing trials took 355 to 1226 control steps at 20 Hz. Physics-only stepping
measured about 145 control steps per second on 4 cores, with about 1.9 GB of
resident memory. This is not a camera-enabled measurement.

## What the pilot established

- In the kitchen it was tuned for, all four controls can be operated reliably
  from varied spawn poses.
- Pressing the microwave button, opening the oven door and turning a stove knob
  mostly transfer to other appliance models and layouts without change.
- The rack hook does not transfer: it failed in all seven unseen scenes (lip not
  found on other rack models, hook slipping, reach limits). The rack step needs a
  different method or must be left out of the task.

## Physical findings that affect the task design

- The gripper's finger force is limited to 20 N per finger. A pinch turns the
  stovetop knob (1.0 N m joint friction) slowly: about 375 steps for 0.7 rad.
- The oven door is not gravity compensated (6.6 kg in layout 2). It opens by its
  own weight past a few degrees, so opening needs only a short drag. Closing it
  means lifting it and has not been tested.
- An open oven door blocks the robot. The base must stand beside the door's
  sweep, and in layout 2 a kitchen island limits how far back it can stand.
- In layout 1 the stove's knobs are on the rear panel behind the cooktop; in
  layout 2 they are on the front edge of the stovetop. Reach differs strongly by
  layout.

## Limits

- Not run on macOS and not rendered. The camera and video path of the script
  mirrors `auto_pick_place.py` but is untested.
- Generalization used one seed per scene.
- Only the forward direction of each control was tested: knob on, button start,
  door open, rack out. Knob off, door close, rack in, opening the microwave door
  and loading an item are untested.
- The controller is privileged. Nothing here shows that the controls can be
  operated from camera observations alone.

## Housekeeping

A first test run inside this folder left one temporary file that RoboCasa could
not delete: `_to_delete/1794461718_3276358_10.xml` (moved out of
`robocasa/robocasa/models/assets/objects/lightwheel/pot/Pot057/`). It is safe to
delete. Later runs used a copy of RoboCasa outside this folder.
