# Interface: what your scene must look like to the grader

The grader opens `/home/user/submission/scene.usda` (or `scene.usd`) in its own Isaac Sim 6.2 process, after your
session has ended, and measures it. It runs none of your scene's Python: everything the bay does must be in the stage
itself: geometry, physics (rigid bodies, colliders, masses, materials, joints, drives, limits) and, where a rule needs
logic, OmniGraph action graphs. Your fleet's program (`run.sh` + its code, `control.md`) is the one piece of your code the
grader runs, as a separate process behind a wire.

## Stage

- Metres, kilograms, seconds; `upAxis = "Z"`; `metersPerUnit = 1`; `kilogramsPerUnit = 1`; default prim `/World`.
- One physics scene at `/World/PhysicsScene`, gravity 9.81 m/s² along -Z. The grader steps physics at 120 Hz.
- Files: the root layer, plus anything it references, sublayers or loads as a texture, must live under
  `/home/user/submission/` (referenced by **relative** path; the submission is copied before it is graded) or under
  the offline libraries `/opt/assets/isaac/` and `/opt/assets/arb/` (referenced by absolute path). Nothing else is
  reachable at grading time. No behaviour scripts (`OmniScriptingAPI`) anywhere in the stage.
- At t = 0 everything is at rest: the lever open, the parts on their tables, the bucket on the floor with its ropes
  lying slack, the robots on their pads. Nothing may be set moving by the scene itself.

## Required prims

| path | what |
|---|---|
| `/World/Facility/Floor` | a static collider whose top is the floor plane z = 0 |
| `/World/Facility/Tables/Table_A`, `Table_B` | the two tables (static colliders) |
| `/World/Facility/Crate` | the parts crate beside the station (static; its bounding box is where the parts go) |
| `/World/Facility/Items/<name>` | one rigid body per part on the tables; `<name>` is up to you |
| `/World/Facility/SteamLine` | the steam line; this prim carries `arb:steam_active` (bool) |
| `/World/Facility/SteamLine/Handle` | the valve's lever, a rigid body on a revolute joint (open at rest, a quarter turn to its closed stop) |
| `/World/Facility/SteamLine/Plume` | the hazard volume: geometry (any; it may be invisible, no collision needed) whose bounding box is the volume |
| `/World/Facility/SteamLine/Gauge/Dial` | the gauge's dial face (a flat disc; its largest extent is the dial's diameter) |
| `/World/Facility/FirePan` | the pan and its flames; this prim carries `arb:fire_active` (bool) |
| `/World/Facility/Tank` | the water tank |
| `/World/Facility/Bucket/Body` | the bucket, a rigid body (full: its mass includes the water), its three rings part of it |
| `/World/Facility/Pads/Pad_1`, `Pad_2`, `Pad_3`, `UGV_Bay`, `Valve_Square` | the floor markings (geometry; the grader takes their centres) |
| `/World/Fleet/UAV_1`, `UAV_2`, `UAV_3`, `UGV` | the robots: each one floating articulation (exactly one `ArticulationRootAPI` below it, nothing fixing it to the world) with everything it carries; each of these four prims carries `arb:disabled` (bool) |
| `/World/Fleet/UAV_1/Rope`, `UAV_2/Rope`, `UAV_3/Rope` | that robot's rope (below) |

How the valve or a robot is put together below these prims (which bodies, which joints, what the articulation root
is) is yours to decide. Other prims (lights, the hall, your helpers, NVIDIA Flow volumes for the fire and the steam)
may go anywhere under `/World`; the scene checks measure bodies and the state attributes and never render, so the
bay's looks are yours to decide. In the bay mission the grader renders the real bay and your four robots as composed
for the robots' sensors: a robot's looks (meshes, materials) are what the other robots' cameras see (`control.md`).

## The ropes

Each flying robot's rope is a **chain of rigid bodies** under `/World/Fleet/UAV_k/Rope`: at least 8 links, each a
rigid body with a collider, joined link to link by joints (spherical, or D6 with the translations locked) into one
chain. The first link is joined to a body of the robot (the **hook** joint); the last link is joined to
`/World/Facility/Bucket/Body` at one of its rings (the **ring** joint). The hook joint carries
`physics:excludeFromArticulation = true`: the links are plain rigid bodies, not links of the robot's articulation
(PhysX cannot hang a heavy body off a light leaf link of an articulation; a chain built that way blows up). No
distance joints anywhere in the rigging; the bucket is joined to nothing but its three ropes. The grader measures the
rope as it hangs, sags, carries and swings, against the real bay's rig (`requirements.md` R3); the bay mission
(`control.md`) flies on the plant's own ropes, hooked where your hook joint holds yours.

## Declarations

- `arb:airframe_model` (string) somewhere under each flying robot, `arb:base_model` under the vehicle, `arb:arm_model`
  under each robot: a catalog name. The named model's asset must be what the robot is built from (the grader checks
  the referenced files).
- `arb:sensor_model` (string) on each sensor prim that counts: a catalog name; cameras are `UsdGeom.Camera` prims with
  that entry's intrinsics; a lidar is any prim with the entry's name, and its scan is in its prim's frame. What a robot
  senses in the bay mission (`control.md`, The sensors) is what it carries here: its first four cameras and its first
  lidar stream.
- Put every declaration and every sensor prim outside instanceable prims: the grader does not read inside instances.

## State attributes

`arb:steam_active`, `arb:fire_active` and the four `arb:disabled` are plain `bool` attributes on the prims above, true /
true / false at the start, maintained by your OmniGraph logic as `bay_brief.md`'s hazard rules say. The grader reads
them in its scene checks; in the bay mission it keeps the rules itself (`control.md`).

## OmniGraph rules

Action graphs (`evaluator_name = "execution"`) are allowed for the hazard rules. The grader steps the simulation the
normal way (timeline playing, physics stepped), so graphs triggered by `isaacsim.core.nodes.OnPhysicsStep` (in an
on-demand graph) or `omni.graph.action.OnPlaybackTick` run. Node types are limited to an allowlist
(`check_structure.py` has it): events, reads of prims, attributes and articulation state, arithmetic, comparison,
logic, vectors and matrices, variables, `WritePrimAttribute`. No script node, no Python, no timeline control, no node
that writes a transform. `WritePrimAttribute` may write joint drive attributes (`drive:*`) and your own `arb:*`
attributes, with the attribute name a constant; nothing else.

`~/given/check_structure.py <scene>` checks this page's rules (paths, files, ropes, declarations, graph nodes) without
scoring anything. The grader runs the same check first; a scene that fails it scores 0.

## In the bay mission

The graded mission takes from your stage only the four robots, each the whole subtree under `/World/Fleet/<robot>`
as composed; it drops the ropes, any joint from a robot to something outside its own subtree, and everything outside
the four robots (physics materials a robot binds are kept wherever under `/World` they are defined). Build each robot
so that it works on its own in another bay. Each robot keeps the colliders of the library models it declares
(`arb:airframe_model`, `arb:base_model`, `arb:arm_model`): it references each model's file in the library (not a copy
of it), and none of the model's colliders is removed, switched off (`physics:collisionEnabled`), made a trigger or
shrunk. Add your own colliders beside them if you like; a robot that breaks this rule scores 0 on the mission's
no-collisions check (`check_structure.py` notes it). Your collision groups and filtered pairs act there among your own
robots only, and the grader reports every contact of a robot's body whatever your scene's contact-report settings.
