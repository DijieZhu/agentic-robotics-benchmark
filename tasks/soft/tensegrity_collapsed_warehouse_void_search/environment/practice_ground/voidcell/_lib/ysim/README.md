# ysim 0.3.1: simulation base of the three-strut cable-driven tensegrity (MuJoCo 3.14, SI units)

**What the simulated robot is made of:** three rigid struts that are the authors' own strut mesh
(`Will_strut_scaled_centered.ply`, three copies), with the authors' own masses and inertia, collision shapes fitted
to that mesh, and the authors' own tendon net (6 motor-wound lines, 6 short + 3 long stretch sensors) attached to
the real hooks and line guides found in the mesh (which hook carries which tendon is the builder's free choice in the
authors' assembly guide: ASSUMPTION `hook_roles`). Nothing of the robot is modelled, redrawn or completed; the
stretch sensors have no 3D shape and exist only as hook-to-hook lengths.

Parameter provenance: `params.py` (one entry per value); validation, limits and what each version changed: the
changelog and the known limits below. Data tables: `../notes/sim_base_data/`. Large meshes (visual only, not shipped with this copy): `$YSIM_ASSETS` or
`/tmp/ysim/work/ysim_assets/` (`paths.SCRATCH`). The physics model (`ModelConfig(visual=False)`)
needs nothing outside this copy.

## Quick start

```python
import sys; sys.path.insert(0, "<the directory that holds ysim/>")
from ysim import model, sim, gaits
S = sim.Simulator(model.ModelConfig(visual=False))   # visual=True loads the authors' mesh for rendering
S.stand()                                            # rest state: all six short sensors at 200 mm, settled
r = S.run_primitive("roll_100_100")                  # any of gaits.NAMES (11 published primitives)
print(r["dist"], r["dtheta_deg"], r["t_total"], r["bottom_start"], r["bottom_end"], r["rest_reached"])
tel = S.telemetry()                                  # what the robot reports: sensor lengths, capacitance, IMU, encoders
```

Run every Python process as `python -I -B ...` with `OPENBLAS_CORETYPE=Haswell` and `OPENBLAS_L2_SIZE=256`
set before numpy loads; physics is single-threaded and bit-reproducible (see `health.py`, tests).

## One primitive, start to end (0.3.1; what an executor builds on)

| step | what happens | source |
|---|---|---|
| precondition | the robot rests on a rollable face: the three lowest end caps are one of `gaits.ROLLABLE_FACES` (025, 035, 124, 125, 034, 134); otherwise `run_primitive` / `start_states` raise `gaits.NotRollableFace` (a `ValueError`) | the authors' symmetry table |
| mapping | the primitive's base target states are mapped once from that face (`gaits.SYMMETRY`); both members of a pair in `gaits.FACE_PAIRS` map alike | rss25 `run_tensegrity_Astar.py` 556-561 |
| own states | `[1]*6` at the primitive's ranges, the two (roll) or three (turn) published states, `[1]*6` at its ranges; PID at 100 Hz on the short sensors | lines 166-168, 401-428 |
| rest phase | `[1]*6` at RANGE 100: every short sensor back to 200 mm (the authors' re-plan phase) | lines 438-443, 471-477; RA-L Fig. 5 |
| hold | braked for `params.settle_after` = 0.911 s (executed in 10 ms ticks: 0.92 s) | ASSUMPTION: the authors' re-planning delay, one 7 Hz pose period + the planner's mean 0.768 s (RA-L lines 208-209, 391). ysim's reading: the rest phase completes every time, then the hold; on the robot the two overlap and a roll that follows a roll can cut an unfinished rest phase short (rss25 lines 548-553; ASSUMPTION `end_of_primitive`) |
| result | `dist, dx, dy, dtheta_deg` (start frame of the authors' SE(2)), `t_total`, `bottom_start / bottom_end`, `rollable_end`, `end_triangle`, `rest_reached`, `timeouts`, `top_z_end`, `short_sensors_end_mm`, `se2_before_rest_phase`, energy, peak line tension | `sim.Simulator.run_primitive` |

Execution mode (`run_primitive(name, mode=...)`, default `params.execution_mode` = `tracked`):

| mode | stopping band | slack rule | source |
|---|---|---|---|
| `tracked` (default) | 5 % of the range for every target and the rest phase | the authors' simulator rule: a motor asked to pay out a slack line gets 0 that tick (not latched); a state ends when every motor is done or zeroed | ASSUMPTION `track_tol` (with the authors' band no forward roll of this model reaches the next face: 0 / 108; the same targets at a uniform 15 % band: 21 / 108; 5 % is the widest of 15 / 10 / 7 / 5 % at which all 108 do) |
| `hardware` | 15 % (35 % for rolls with a range >= 130; a turn and the rest phase keep the last roll's band) | none | the authors' control code rss25 |
| `authors_sim` | 10 % or 10 mm, min length 90/100/80 mm, k_p 6, no rest phase | as tracked | hybrid_mppi `get_3bar_prims_se2_data` (diagnostic) |

Faces (deterministic in every execution measured; 0.3.1 re-measured, unchanged from 0.3.0, in the base's validation, not shipped with this copy): a forward
roll lands on the face class the authors' `next_nodes` predicts, always on the pair member 125 / 134 / 035
(025 -> 125, 124 -> 134, 034 -> 035, 125 -> 134, 134 -> 035, 035 -> 125); a turn keeps the class: from 025 / 124 /
034 it moves to the pair partner (035 / 125 / 134), from 035 / 125 / 134 it stays. So after its first primitive
the robot lives on 125 / 134 / 035; 124 and 034 are reached only through the authors' reverse rolls, 025 only at
the start. Read the face with `S.bottom()` after the primitive (or predict its class with `gaits.NEXT_NODES`; the
mapping needs only the class). Displacement tables must be read per start face (the 0.3.1 validation, not shipped).

## Modules

| module | content |
|---|---|
| `params.py` | the one parameter table (SI, provenance per value, Rule S for the elastic members, material rule for the line, sensitivity of every ASSUMPTION); `params.dump()` writes `params.json` |
| `assets.py` | dev tool: split the authors' mesh, material groups (full + decimated), collision fits, hook / guide features -> `strut_geometry.json`, `collision_hull_elec.json` |
| `geometry.py` | tendon net topology (authors' MJCF, via `authors_model.json`), hook roles (the default rule, 0.3.0's rule, the 64 assignments the assembly guide allows, `assignment_distance` = the metric that chose the rule), feature points per strut |
| `model.py` | MJCF generator (`ModelConfig`, incl. `hook_rule` / `hook_assignment`), initial poses, `Ids` name tables, `line_stiffness()` (k = EA / L) |
| `motor.py` | Pololu #5192 gearmotor on the 6.35 mm spool: vendor torque-speed line, hold, encoder, energy; sensitivity knobs |
| `controller.py` | the authors' off-board PID (`compute_command`) on the short-sensor lengths; the authors' simulator slack rule |
| `gaits.py` | the 11 published primitives (RA-L Table I), symmetry maps, `ROLLABLE_FACES`, `FACE_PAIRS`, `NotRollableFace` |
| `sim.py` | `Simulator`: physics + motors + PID at 100 Hz + executor (own states + rest phase) + telemetry + `state_hash()` |
| `metrics.py` | SE(2) exactly as the authors' table (`reformat_dictionary.py`) |
| `validate.py` | full-SE(2) validation against the published table, per primitive and start face |
| `health.py` | rest, 10-minute run, timestep study with the same-dt floor, determinism, real-time factor |
| `render.py` | classic-renderer recorder / tracking camera (inspection media) |
| `world.py` | blockout world records (box, portal, tunnel, step, ramp, friction patch, bumps, relief hfield), MJCF export, top-view plot, tendon-cut test, site-record import of an earlier family |
| `camera.py` | fixed overhead RGB-D camera (authors' calibration by default), renderer-agnostic backend interface, classic backend |
| `capability.py` | capability protocols (the 0.3.0 extension, completed for 0.3.1; `CAPABILITY_REV` "0.3.1-sheet1"): placement (`place`, `place_on_face` = the 07 drop start), `set_floor_friction`, collision envelope, tick-hook tracker, the two measuring devices (`privileged_steer_plan`: true pose after every primitive, PRIVILEGED; `dead_reckon_plan`: a fixed open-loop sequence from a believed start), `set_targets_mm` / `primitive_shapes_mm` (one target shape in mm through the same PID, normalised as the 07 cell does), the `trial_*` protocols. The capability sheet measured with them is FINAL for 0.3.1 (not shipped with this copy); `STRAIGHT = "roll_100_140"` |

## Worlds and the overhead camera

```python
from ysim import world as wd, camera as cm, capability as cap
w = wd.World("demo").add(wd.portal("door", (0, 0), 0.0, 2.0, 0.3, 0.8, 0.50, 0.40), wd.friction_patch("dust", -1, 1, 1, 2, "dust"))
S = cap.new_sim(w, visual=True)                     # = sim.Simulator(w.model_config(visual=True))
cap.place(S, (0, -0.8), 90.0); S.stand()            # robot centroid at (0, -0.8), rolling toward +y
cam = cm.OverheadCamera(S, cm.CameraSpec.authors_lab(target_xy=(0, 0)))   # 7 Hz exposures (qpos snapshots)
S.run_primitive("roll_120_140"); f = cam.latest()   # Frame: rgb uint8, depth float32 m (0 = no hit)
w.plot_top("top.png"); w.to_json("world.json")      # records for art replacement; wd.from_concept07(json) for 07 sites
```
Options: `ModelConfig(lights=False)` (no ysim key / fill light, headlight off: the site brings its own lighting);
`S.start_states(states, raw=True)` executes raw motor targets without the face check (recovery sequences).
Rendering: an OSMesa or EGL library and `MUJOCO_GL` set before Python starts (camera.py and render.py set
`MESA_SHADER_CACHE_DISABLE`).

## Rebuild the assets

```bash
OPENBLAS_CORETYPE=Haswell OPENBLAS_L2_SIZE=256 python -I -B -c \
  "import sys; sys.path.insert(0, '<the directory that holds ysim/>'); from ysim import assets; assets.build()"
```
(writes the asset dir, `notes/sim_base_data/strut_geometry.json` and `collision_hull_elec.json`; the tests check
the recorded hashes). `notes/sim_base_data/authors_model.json` is copied from the authors' MJCF by
a development script of ysim 0.3.0 (not shipped with this copy; the source and its sha256 are inside the file).

## Tests

```bash
OPENBLAS_CORETYPE=Haswell OPENBLAS_L2_SIZE=256 PYTHONDONTWRITEBYTECODE=1 python -I -B -m pytest -p no:cacheprovider -q \
  <ysim_tests>          # fast set (the base's own test folder; not shipped with this copy)
... -m slow    # validation regression and the 10-minute run
```

## Changelog

**0.3.1 + capability sheet** (2026-11-17; the robot is unchanged: model.py, sim.py, params.py,
geometry.py, gaits.py, controller.py, motor.py, metrics.py, assets.py byte-identical to the 0.3.1 install, version
stays "0.3.1"). The capability sheet (not shipped with this copy) is re-issued on the final model
(status FINAL for 0.3.1): every envelope key of the 07 task with definition, seeded trials,
mean +- sd or k / n and confidence, the open-loop dispersion and the dead-end
pocket. `capability.py` (`CAPABILITY_REV` "0.3.1-sheet1"): `STRAIGHT = "roll_100_140"` (0.3.1's straightest roll;
`STRAIGHT_V030` keeps "roll_100_120"); `set_targets_mm` normalises as the 07 cell's set_targets (100 mm min length,
RANGE 100 mm; the 0.3.0 extension had 140 mm); `trial_prim(..., reverse=True)` (the reverse-roll table by start face);
`trial_pocket` stops entering after two primitives touched the plug (was: stuck or 10 primitives, which let the robot
grind and turn against the plug) and chooses the set_targets reverse shapes per current face (`rev_by_face`).
CLASS_A / CLASS_B / FACE_NEXT_* re-checked on 0.3.1 (unchanged). Tests (not shipped with this copy) check the sheet's
layout, that it was measured on exactly these robot files (sha256), that STRAIGHT is the sheet's straightest roll, and
the set_targets normalisation.

**0.3.1** (2026-11-17; fix round 2 after the second independent verification). The model
changes in exactly two declared places; with `ModelConfig(hook_rule="guide_side_v030")` and
`run_primitive(settle_after=1.0)` 0.3.0's recorded executions come out unchanged (a test of the base, not shipped).
- Hook roles are an ASSUMPTION (`params.hook_roles`), set by a stated geometric rule: the assembly guide leaves the
  hooks free (the line on any hook, the long sensor opposite it, the short sensors on the other two: 8 per end, 64 per
  strut, `geometry.allowed_end_assignments`); the rule takes the allowed assignment whose attachment sites lie nearest
  the authors' simulation model (`geometry.assignment_distance`, M = 10.68 mm; runner-up 11.77 mm; 0.3.0's rule
  15.60 mm with 12 nearer). Result: S_out on the hook nearest the line guide, S_out -> L -> S_in -> I at +90 deg about
  the strut's odd -> even axis at both ends (the authors' order at all six ends). Even ends as in 0.3.0; odd ends: I and
  L exchanged. The initial pose's roll per strut is now the closed-form fit of all ten attachment sites.
- Consequences (not criteria): forward rolls on the faces the robot lives on turn -24.6...+0.7 deg (0.3.0:
  -31.4...-9.1 deg); the straightest roll is now roll_100_140 (+0.7 deg); full-SE(2) validation 6 / 66 cells (0.3.0:
  0 / 66), none from the faces the robot lives on; the authors' 15 % band (tracked) tips 21 / 108 rolls (0.3.0: 3);
  peak line tension up to 59.5 N in 10 minutes (0.3.0: 32.5 N).
- Across the 64 assignments the guide allows (the model's real uncertainty): mean heading per forward
  roll on the live faces -25.7...+1.2 deg (chosen -12.2), distance 0.225-0.265 m; rolls tipping with a uniform 15 %
  band 0-18 of 18 (chosen 3, 0.3.0's 0); with the authors' code verbatim some rolls tip in 16 of 64 (at most 13 / 18);
  all 64 rest stably.
- The braked hold after the rest phase is an ASSUMPTION `params.settle_after` = 0.911 s (the authors' re-planning
  delay: one 7 Hz pose period + the planner's mean 0.768 s); 0.3.0 held 1.0 s unregistered, with a citation that is
  not in the RA-L text. `end_of_primitive` is an ASSUMPTION too: "the rest phase completes every time, then the hold" is
  a reading of the planning delay (the authors' loop cuts an unfinished rest phase short when a roll follows a roll and
  the plan comes early, rss25 lines 548-553).
- The stopping band stays 5 % (same ladder rule; 15 / 10 / 7 / 5 % tip 21 / 80 / 100 / 108 of 108); the hook choice is
  named among its candidate causes.
- Provenance: the slack rule (not a verbatim copy of the authors' test), the premise of `mesh_end_to_node`; the
  timestep re-checked on 0.3.1. Sensitivities of every ASSUMPTION and of the Rule-S members (+-30 %) re-measured on
  0.3.1 (not shipped with this copy): the long-sensor stiffness x1.3 stops roll_100_100 / roll_100_120 tipping.
- Statements: the model's cw differs qualitatively from the table (the table's cw flips and translates 0.166 m, the
  model's turns in place); translation errors are quoted for forward rolls only (0.04-0.10 m in 0.3.1).
- API: `run_primitive(settle_after=None)` and `capability.run_shapes(settle_after=None)` default to
  `params.settle_after`; `ModelConfig.hook_rule` / `hook_assignment`; `geometry.hook_roles(geo, rule, assignment)`,
  `strut_features(geo, hook_rule, hook_assignment)`, `HOOK_RULES`, `DEFAULT_HOOK_RULE`, `allowed_end_assignments`,
  `assignment_distance`; `model.initial_pose(feat, end_map, fit)`. Result dictionaries, primitive names, targets,
  ranges, PID, motors, telemetry, camera and world API unchanged; faces and transitions unchanged.
- The capability sheet was still SUPERSEDED at this point (re-issued in the entry above).

**0.3.0 + capability extension, NOT completed** (2026-11-17; the model is unchanged by it): the
re-measurement of the capability sheet on 0.3.0 was stopped half-way because the model was
about to change (0.3.1). What it left: `capability.py` extended with protocols for the 07 task's sites
(PRIM_TABLE per start face, open-loop dispersion, blind covered galleries, the route under a slab,
the dead-end pocket with set_targets recovery, creep, the 07 per-site floor friction), the two labelled measuring
devices and `place_on_face`; `capability.STRAIGHT = "roll_100_120"` (0.3.0's straightest roll). It imports and its
unit tests pass, but it is unfinished: **the sheet was not re-issued** - the capability sheet file
still holds the 0.2.0 measurements and is marked SUPERSEDED (now by 0.3.1); the same content is
kept in an archived copy (not shipped). (An earlier version of this entry said the sheet was current again; that was not so.)

**0.3.0** (2026-11-17; the fix round after the first independent verification)
- End of a primitive as on the real robot: after its own states the robot is driven back to the rest shape
  (`[1]*6` at RANGE 100, the authors' re-plan phase), then 1 s settling; every primitive ends at rest
  (top of the robot 0.231-0.243 m after any of the 11; 0.2.0 left it folded, 0.127-0.219 m after rolls).
- The slack latch of 0.2.0 (ours) is removed; mode `tracked` uses the authors' simulator slack rule (zero that
  tick, no latch; a state ends when all motors are done or zeroed).
- Fishing line: material rule k = EA / (L0 + 8.3 mm) per cable from its current free length, EA = 5.56 kN
  (100 lb braided UHMWPE, ASSUMPTION with range 0.5x-2x); about 26 kN/m at rest instead of the Rule-S 1000 N/m.
- Timestep 0.25 ms (was 1 ms), chosen by a study against 0.125 ms with a same-dt perturbation floor.
- The authors' 15 % band re-tested with the stiffer line: still no roll reaches the next face (0 / 108);
  the 5 % band stays a declared ASSUMPTION (re-derived by the same ladder rule: 3 / 72 / 97 / 108 of 108 at
  15 / 10 / 7 / 5 %).
- Validation in full SE(2) (distance, heading and direction of travel), per primitive and start face, 4 executions
  per cell (`validate.py`; its results are not shipped with this copy).
- Sensitivities for battery voltage, spool layering, motor hold, rotor inertia, line EA and the integrator.
- No run-time dependence on scratch: node and site data in `authors_model.json`, collision hull in
  `collision_hull_elec.json` (a test of the base, not shipped with this copy).
- `mesh_end_to_node` declared with its evidence (micro-USB socket at the mesh +z end); three citations corrected
  (line capacity 0.225 m, friction sources, decimation input 1 061 293 faces).
- `start_states` raises `gaits.NotRollableFace` on a bottom triple outside the authors' table (no silent mapping);
  `ModelConfig(lights=False)` documented and tested.

**0.2.0**: worlds, overhead RGB-D camera, capability sheet (superseded by the 0.3.0 model change).
**0.1.0**: the robot model.

## Known limits

No public calibration data of the real robot exists; the validation reference (the authors' primitive table) is
itself a simulation output. Which hook carries which tendon is the builder's free choice (ASSUMPTION `hook_roles`);
across the 64 assignments the guide allows the per-roll heading changes a lot (params.py, `hook_roles`): that spread is the
model's real uncertainty. The model needs a stopping band three times tighter than the authors' code to roll
(ASSUMPTION `track_tol`). The model's cw turns in place, the table's cw flips to another face. Tendons do not collide
with struts or with world geometry (capability.Tracker counts the samples in which a line cuts a world box). Battery
discharge is not modelled; rotor inertia only as a sensitivity. With `visual=True` the asset directory defaults to
scratch: set `YSIM_ASSETS` or `ModelConfig(assets=...)` elsewhere (the physics model needs only the git tables).
