# NOTICE (grader-side; never staged into the agent's home)

The givens, the instruction, every reply of the cell and of the rehearsal facility, and every picture the agent sees
name neither the robot the cell follows, nor its makers, nor the station whose modules the scene follows, nor the
benchmark: they use neutral names only (the table below). This file carries the attribution and the record of what
the task's world leaves out (the task's rule: sources are written here only).

## Sources

- **The robot.** The cabin free-flyer of this task follows NASA's Astrobee free-flying robot (NASA Ames Research
  Center). Everything a person or the agent can see of the robot -- in every camera frame, in the figures of the robot
  manual -- is the robot's visible mesh as NASA published it with the Astrobee Robot Software's media
  (`astrobee_media`, folder `astrobee_freeflyer`: body, propulsion modules with the default skin, bumpers, the arm's and
  the gripper's links), used as given: only converted in format, units and frame, triangulated and split by material
  (asset component `robot_author`, `tests/cabincell/data/robot_author/ASSETS.json`). The robot's collision model, its
  mass properties and the parts nobody sees are this project's reconstruction from NASA's public documents (the
  flight software's parameters, the payload interface drawings, NASA technical reports), asset component `robot`.
  The Astrobee Robot Software (github.com/nasa/astrobee) is distributed under the Apache License 2.0.
- **The station.** The three mission modules and the rehearsal module follow the interior of the International Space
  Station's laboratory segment as NASA's public 3D models show it: the visual meshes of NASA 3D Resources' station
  interior model and the coarse module shells of the Astrobee media (`astrobee_iss`), registered module by module into
  one frame. The textures were neutralised (signage, part numbers, insignia and facility names removed or painted
  over; asset component `station`, look `station_look_v05`). The collision world (wall boxes fitted to the interior,
  the crew handrails with their stand-offs), the return inlets, the stowage units, the dock's collision geometry and
  every item are this task's own.
- **Everything else is original to this task**: the cell (`tests/cabincell/`: the task firmware, the station services,
  the telemetry synthesis, the suction and seal models of the inlets and the stowage units, the frame renderer, the
  records), the site generator and its stored site packages, the reference program (`tests/reference/`), the
  grader, the rehearsal facility, the documents, the figures and the catalogue photographs. The verify kit
  (`cellkit/`, `robotics_grader/`) is the benchmark's shared code. The simulator is MuJoCo 3.14.0 (Apache License 2.0).

## A fictional station

The station of this task is fictional. Its ventilation architecture (one cabin air unit per module with two or three
parallel return inlets of a pad or element build, the return-loop pressure telemetry, the advisories and their
limits), the stowage units with their holding screen and pouch pads, the flight rules and the station requests, the
lost-item register, the crew's items and every site are invented for the task and do not describe the real station's
systems or operations. The real station's modules lend the scene its shape and look only.

## Known and not modelled

- A free-flyer of this class flying within about 0.5-0.6 m of a wall is drawn toward the wall by its own fan intake
  (reference configuration key `prop.wall_suction_rule`): not modelled; the manual does not state it.
- The fan power model draws its largest power with every nozzle closed (an open question of the reference
  configuration): kept as it is; the manual prints the measured power by mode.
- The aft depth camera's bar detector and the depth frames see the collision proxies of the arm and the fingers (at
  most 5.2 mm from the maker's finger surfaces), the colour frames the maker's meshes.

## Neutral names (agent-visible) and what they stand for

| neutral name | real counterpart |
|---|---|
| the cabin free-flyer (CFR-1) | Astrobee free flyer (NASA Ames) |
| laboratory module L1 / L2 / L3 | US Laboratory (Destiny) / Japanese Experiment Module pressurised module (Kibo) / Columbus |
| hub H2 | Node 2 (Harmony) |
| the station | International Space Station, US Orbital Segment |
| cabin air system / cabin air unit (CAU) | ECLSS cabin ventilation / a module's cabin air fan (fictional unit) |
| return inlet (pad-type, element-type) | cabin ventilation return air grille (fictional builds) |
| crew handrail | ISS crew handrail |
| forward imaging / forward wide / aft wide / forward depth / aft depth camera (FIC, FWC, AWC, FDC, ADC) | SciCam / NavCam / DockCam / HazCam / PerchCam |
| flight modes OFF / STANDARD / AGILE / HUSH / FINE | flight modes off / nominal / difficult / quiet / precision |
| perching arm, tendon gripper | perching arm, underactuated tendon gripper |
| dock, berth 1 / 2 | docking station, berth 1 / 2 |
| forward / aft lamp | forward / aft flashlight |
| localisation modes MAP / FIDUCIAL | MappedLandmarks / ARTags |
| zone file (keep-in / keep-out boxes) | the flight software's zones file |

No agent-visible file carries the real names.
