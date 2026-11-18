# Robot card

The rounds are walked by a small four-legged robot with twelve torque-controlled joints and a fixed walking
controller. This folder holds the files that walk the hidden routes in your twin: the robot model
(`robot/model/robot.xml` and its meshes), the controller (`locomotion/`) and the route runner (`walk_probe.py`).
They need Python, numpy and MuJoCo only. The licence of the robot description travels with the model:
`robot/LICENSE-robot-description-BSD-3-Clause.txt`.

## Size and mass

| item | value |
|---|---|
| mass | 3.15 kg (body, legs, battery, a camera head at the front and a sensor puck on top) |
| body | 0.45 m long, 0.22 m wide, 0.06 m thick; legs 0.16 m thigh + 0.16 m shank, round feet 34 mm |
| footprint while walking (legs included) | about 0.60 m long, 0.34 m wide |
| highest point, standing / walking | 0.32 m above the floor (body centre at 0.24 m) |
| highest point, crouch walking | 0.25 m above the floor (body centre at 0.17 m) |

## How a run works

A route is a list of floor points (x, y in the scene's frame, metres) with a gait per segment: `walk` (0.25 m/s) or
`crouch` (the body lowered; 0.12 m/s commanded, about 0.08 m/s achieved). A third gait, `high_step`, exists but the
assessment does not use it. The route runner:

- replaces the scene's `<option>` with its own simulation (1 ms steps, the implicitfast integrator, elliptic
  friction cones, the Newton solver with 50 iterations; everything else at MuJoCo's defaults) and attaches the
  robot;
- stands the robot at the first point, on the surface that vertical rays find there: under the robot's centre the
  top of a solid with at least 0.40 m of free height above it, the one closest to the route's `start_z` (default
  0); under each of the four feet the highest surface from 5 cm below to 10 cm above it. Without such a surface the
  run ends at once as fell (`no_start_surface`); with geometry in the standing robot's space (0.10 to 0.33 m above
  the surface, at the centre or over a foot) it ends at once as stopped (`start_blocked`);
- steers along the points (pure pursuit on the robot's position, 0.30 m ahead) and switches the gait from segment
  to segment;
- ends the run as
  - **reached**: the robot's centre is within 0.15 m of the last point and it has covered the route to within
    0.5 m of its end, before the time limit of twice the route's nominal duration (the segment lengths divided by
    the commanded speeds) plus 5 s;
  - **fell**: the trunk tilts by more than 50 degrees, or rests on something under it for 0.25 s, or the simulation
    goes unstable;
  - **stopped**: the time limit runs out, or the robot gains less than 0.15 m along the route in 6 s.

A result holds the outcome and its reason, the **progress** (the furthest point reached along the route as a share
of its length; 1 when reached), the simulated time, the end position and the names of the scene's geoms that the
robot's trunk touched.

Only collision geometry counts: what the robot meets is what the scene's collision geoms say (a mesh collides as
its convex hull).

## What it gets through

Measured on simple test scenes, 20 perturbed starts each:

| situation | always gets through | never gets through | in between (outcome not predictable) |
|---|---|---|---|
| passage between two solid objects | gap >= 0.34 m | gap <= 0.28 m | 0.29-0.33 m |
| under an overhang, walking | underside >= 0.33 m | <= 0.31 m | about 0.32 m |
| under an overhang, crouching | underside >= 0.26 m | <= 0.25 m | - |
| a raised strip or a low object to step over | height <= 0.02 m | - | 0.04 m and more |
| a solid wall or object on the route | - | always | - |
| a floor of slabs with joints up to 1 cm, slopes up to 1 degree | always | - | - |

The robot is blind: it does not see what lies ahead and does not plan around it. A step, a kerb or a pallet lying
across its route is not a reliable barrier (it sometimes gets over, usually falls), which is why no scored route is
decided by one.

## The scored walks

The conditions under which the assessment walks the routes (units held, gates fixed, the same contact parameters
on every geom, the start window, the starts, when a run counts as reached) are in `twin_contract.md`, section 9.
`check_twin.py --walk routes.json` applies them to routes of your own. `walk_probe.py` by itself walks a scene file
exactly as it is: free units can be pushed, gates can slide, the scene's own contact parameters apply, and the
robot meets only geoms whose `conaffinity` has its first bit set (MuJoCo's default).

## Running it

    python3 walk_probe.py --twin scene.xml --routes routes.json [--variants K] [--procs N] --out results.json

`routes.json` is a list (or `{"routes": [...]}`) of `{"id": "r1", "waypoints": [[x, y], ...], "modes": ["walk" |
"crouch", one per segment]}` with an optional `"start_yaw"` (radians; default along the first segment) and
`"start_z"` (the height near which the start surface is looked for, default 0). `--variants K` walks each route
from K starts: the nominal one and K - 1 seeded ones (the start moved by up to 5 cm, turned by up to 3 degrees,
another gait phase, the scene's friction scaled by 0.92 to 1.08), a wider set than the scored walks use. Results
are reproducible for the same MuJoCo and numpy builds.
