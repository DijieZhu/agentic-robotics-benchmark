# The session paper: its format and filled examples

setup(session) receives the session paper: one JSON object with the fields below. The acceptance hands out one paper per session; each block's budget on it is the value of the block budget's formula on that block's list (acceptance_specification.md, The block's budget), and the runner checks it against its own computation.

Paths on the task's pages (payload/..., controller/...) are relative to the folder that holds payload and controller; the zone_file of a session paper is a path relative to payload.

## The fields

- `sessions.paper_fields`:
  - session name: an identifier that begins with the session's name
  - software release in force (the paper also carries the list of the modes available to the session's jobs, and the monitor's window that holds under this release)
  - the mode of each block
  - the check-out's mode (nominal, fan level 2)
  - start pose (nominal; the built-in controller holds the robot within 0.10 m and 10 degrees of it until the check-out begins)
  - zone file in force
  - battery slots occupied
  - equipment: as prepared for the campaign (see the preparation photographs)
  - the session's time window
  - per block: its name, its start pose, the ordered job list and the block's budget
  - unit: fields of the session paper that setup receives

Each field under the key with which setup receives it:

| field | key or keys |
|---|---|
| session name: an identifier that begins with the session's name | `session` |
| software release in force (the paper also carries the list of the modes available to the session's jobs, and the monitor's window that holds under this release) | `release`, `release_modes`, `window_s` |
| the mode of each block | `blocks[].mode` |
| the check-out's mode (nominal, fan level 2) | `checkout_mode` |
| start pose (nominal; the built-in controller holds the robot within 0.10 m and 10 degrees of it until the check-out begins) | `start_pose` |
| zone file in force | `zone_file` |
| battery slots occupied | `battery_slots` |
| equipment: as prepared for the campaign (see the preparation photographs) | `equipment` |
| the session's time window | `time_window` |
| per block: its name, its start pose, the ordered job list and the block's budget | `blocks` |

The keys with their nesting: `session`, `release`, `release_modes`, `checkout_mode`, `start_pose` {`position`, `attitude`}, `zone_file`, `battery_slots` {`occupied`, `of`}, `equipment`, `time_window` {`planned_minutes`}, `blocks` [{`name`, `mode`, `start_pose`, `jobs`, `budget_s`}], `window_s`.

Forms: `start_pose` and each block's `start_pose` are {`position`: x, y, z in m in the cabin frame, `attitude`: a quaternion x, y, z, w}; `release` names the software release in force by its version, as the maker names its releases; `release_modes` is the list of the modes available to the session's jobs (a job that names a mode outside it is a request of the mode kind, refusal.margin); `window_s` is the monitor's window that holds under this release (monitor.window_s in the acceptance specification); `zone_file` is the path of the zone file in force in the payload; `battery_slots` is {`occupied`: slots occupied, `of`: slots}; `time_window` is {`planned_minutes`: the planned length of the session in minutes}; `blocks` lists the blocks in their order.

## A job on the paper

- `interface.session_paper_jobs`: the session paper lists, for each block, every job in order (id, kind, mode, goal, hold duration) and the block's budget in seconds; the budget is the value of jobs.block_budget on that list and can be recomputed from it
  - unit: rule
- `jobs.kinds` = `["leg", "slew", "hold", "view"]`
  - unit: kinds
- `jobs.hold_s` = `20.0`
  - unit: s, the duration of every hold in every session
- `jobs.view_goal`:
  - `centre`: m, cabin frame: where the imaging camera's optical centre is to be
  - `axis`: unit vector, cabin frame: where the imaging camera's axis (the body's +x axis) is to point
  - `up`: unit vector, cabin frame, at right angles to axis: where the body's -z axis is to point
  - unit: the fields of a view's goal; the body pose that meets it puts the optical centre (jobs.camera_mount_m) at centre, the body's +x axis along axis and its -z axis along up

A job is {`id`, `kind`, `mode`, `goal`}. The goal of a leg or a slew is {`position`, `attitude`}: the body pose to reach; the goal of a hold is {`duration_s`}: the hold keeps the pose that meets the goal of the job before; the goal of a view is {`centre`, `axis`, `up`} as jobs.view_goal says. A request that must be refused is shown on the paper as an ordinary job; the paper has no field that says which jobs are requests.

## The filled examples

The files below are papers of the examples' own: every key of the acceptance's paper, filled with identifiers and values of their own. They are not sessions of the acceptance and say nothing of them: their start poses and the goals of their legs and slews are body poses of the unit's given records (payload/unit/records), and a view's goal (its centre, axis and up) is derived from such a body pose through the imaging camera's mounting (jobs.camera_mount_m), as jobs.view_goal defines a view's goal; every pose but a zone-kind request's lies inside the zone file the example names with the printed margin (refusal.margin), and the zone-kind request's breaks a face of it; their blocks, orders and requests are the examples' own; each block's budget is the value of jobs.block_budget on that block's list, as the runner computes it. The self-test (controller/selftest) flies a toy paper of the same keys.

- `payload/acceptance/example_papers/example_1.json`: session `example_1`, release `0.14.3`, zone file `cabin/zone_files/2020-06-22/cabinflyer/resources/zones/station.bin`; blocks `ex1_block_a` (nominal), `ex1_block_b` (nominal), `ex1_block_c` (quiet).
  - the job `ex1-a_07` names the mode `docking`, which is not in the paper's `release_modes`: a request of the mode kind (mode_not_available).
  - the job `ex1-b_03` has a goal whose body pose, with the body's bounding sphere, breaks a face of the zone file the example names by at least the margin of refusal.margin: a request of the zone kind (outside_zones).
- `payload/acceptance/example_papers/example_2.json`: session `example_2`, release `0.15.2`, zone file `cabin/zone_files/2020-06-22/cabinflyer/resources/zones/station.bin`; blocks `ex2_block_a` (nominal), `ex2_block_b` (nominal), `ex2_block_c` (precision).
  - the job `ex2-a_03` has a goal whose body pose, with the body's bounding sphere, breaks a face of the zone file the example names by at least the margin of refusal.margin: a request of the zone kind (outside_zones).
  - the job `ex2-c_03` names the mode `off`, which is not in the paper's `release_modes`: a request of the mode kind (mode_not_available).
