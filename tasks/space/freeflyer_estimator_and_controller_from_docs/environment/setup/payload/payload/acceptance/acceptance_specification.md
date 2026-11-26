# Acceptance specification

What the acceptance does with the submitted controller, every rule that ends a job, a block, a session or the run, the block budget's formula, the plan rules, every printed range of the sensors and the unit, and the score. The interface page (controller/interface.md) gives the wire, the frame and the replies; the session paper's format (session_paper_format.md, beside this page) gives the paper that setup receives.

How to read an entry. Each rule or number is printed under its key in backticks: its value as the task's table holds it (numbers in full, as JSON where the value is made of numbers), then its unit or meaning (unit:), then the sentences that belong to it (also:). The same key means the same entry on every page of the task.

Paths on the task's pages (payload/..., controller/...) are relative to the folder that holds payload and controller; the zone_file of a session paper is a path relative to payload.

## The sessions

The acceptance flies the submitted controller on hidden sessions. Each session is announced to the controller by its session paper (see session_paper_format.md); the same controller folder flies every session, and its process is started afresh for each session (interface.restart).

- `sessions.acceptance_count` = `6`
  - unit: hidden sessions
- `sessions.names` = `["session_1", "session_2", "session_3", "session_4", "session_5", "session_6"]`
  - unit: neutral names
- `sessions.blocks`:
  - block 1: legs, slews and holds in the nominal mode
  - block 2: survey views in the nominal mode
  - block 3: legs, slews and holds in the quiet mode (three sessions) or the precision mode (three sessions), after a mode change
  - unit: blocks per session
- `sessions.jobs_told`: about 34 jobs per session; between 2 and 6 of them are requests that must be refused
  - unit: text

## The flow of a session

In order: the start hold by the built-in controller (frames are delivered, the controller is not in control); the check-out; block 1; the robot is brought by the built-in controller to the start pose of the next block and held there (the controller is not in control); block 2; the same transfer; block 3 after a mode change; then the time after the last block, until the session ends. Frames are delivered at every step of all of these stretches.

- `sessions.start_hold_s` = `20.0`
  - unit: s of hold by the built-in controller at the start of a session, frames delivered, the node not in control
- `interface.frame_fields`:
  - t
  - in_control
  - job_id
  - mode
  - fan_level_commanded
  - inertial[].stamp
  - inertial[].rate
  - inertial[].accel
  - registrations[].camera_id
  - registrations[].stamp
  - fixes[].camera_id
  - fixes[].stamp
  - fixes[].landmarks
  - fixes[].position
  - fixes[].attitude
  - propulsion.impeller_speed[2]
  - propulsion.module_state[2]
  - propulsion.nozzle_command[12]
  - camera.busy
  - block.budget_s
  - block.time_left_s
  - unit: field names of a step frame; no estimate, no supervisor timer
- `jobs.block_clock`: the block's clock starts at the first step of the block (for block 3 the mode command is issued at that instant), runs in world time and stops for nothing; plan calls take no world time; when it reaches B the block ends and the built-in controller takes over
  - unit: rule

## The check-out

The check-out follows the start hold. The controller is in control in it; nothing in it is scored. Its mode is the one the paper's check-out field names, with its fan level as printed below. The rules that act in it are printed in their own sections, as their texts say: the zone rule (safety.keep_in) is tested at every step of the check-out; the limit on a stretch without fixes takes the nominal mode's value there (sessions.outage_cap_s); the supervisor is not active in it (monitor.active); the watchdog does not run there and the box applies instead (interface.watchdog_s); a protocol error or a cut at the wall rule there hands the rest of the check-out to the built-in controller (safety.protocol_error, interface.overrun).

- `sessions.checkout_s` = `90.0`
  - unit: s of unscored check-out at the start of each session
- `sessions.checkout_box` = `{"position_m": 0.3, "attitude_deg": 30.0}`
  - unit: the check-out ends early when the robot, as the supervisor measures it, leaves this box about the start pose; nothing is scored and nothing is lost but the remaining check-out time
- `mode.fan_level` = `{"nominal": 2, "quiet": 1, "precision": 3}`
  - unit: fan level of each mode in the releases of the acceptance sessions (nominal was 3 before 2020-10-23)

## What ends a job: the supervisor, the hand-back, an invalid plan, the budget, a request

During every job of a block the supervisor compares the plan with the robot's true pose as it measures it, with its measurement noise; it hands control back to the built-in controller when the error stays outside the mode's tolerance for longer than the window, and it checks the end point at the event done. What a hand-back, an invalid plan, the budget's end and an accepted request cost is printed with each rule.

- `monitor.error`: norm of (plan at the present time minus the robot's true position as the supervisor measures it); angle between plan attitude and the true attitude as measured; the supervisor's measurement adds supervisor.noise
  - unit: definition
- `supervisor.noise` = `{"position_sd_m": 0.01, "attitude_sd_deg": 0.5}`
  - unit: independent Gaussian error of each check of the supervisor (62.5 checks per second)
- `monitor.tolerance.nominal` = `{"position_m": 0.25, "attitude_deg": 20.0, "end_point_m": 0.1}`
  - unit: limits
- `monitor.tolerance.quiet` = `{"position_m": 0.1, "attitude_deg": 10.0, "end_point_m": 0.1}`
  - unit: limits
- `monitor.tolerance.precision` = `{"position_m": 0.1, "attitude_deg": 10.0, "end_point_m": 0.1}`
  - unit: limits
- `monitor.window_s` = `{"releases_from_2021_08_26": 3.0, "earlier_releases": 1.0}`
  - unit: s the error may stay outside the limit without interruption
- `monitor.window_field`: the window is the motion manager's maximum tolerance time; the flight-mode block's own 'time' entry (1.0, a synchronisation check) is not the monitor's window
  - unit: rule
- `monitor.active`: during every job of a block, not during the check-out
  - unit: rule
- `monitor.end_point_check`: at the event done, once accepted (interface.event_rule): the robot's true position as the supervisor measures it (supervisor.noise) against the job's goal position; a measured position farther from the goal position than the mode's end_point_m (monitor.tolerance) fails the check, and a failed check is a hand-back (monitor.hand_back): control returns to the built-in controller; the job in progress scores 0; the built-in controller brings the robot to rest at that job's goal pose, which takes the recovery time monitor.recovery_s off the block's budget; the controller then flies the next job; it counts toward the third hand-back in a block, which ends the block and its remaining jobs score 0
  - unit: rule
- `monitor.hand_back`: control returns to the built-in controller; the job in progress scores 0; the built-in controller brings the robot to rest at that job's goal pose, which takes the recovery time monitor.recovery_s off the block's budget; the controller then flies the next job; the third hand-back in a block ends the block and its remaining jobs score 0
  - unit: rule
- `monitor.recovery_s` = `{"nominal": 25, "precision": 17, "quiet": 30}`
  - unit: s taken from the block's budget for each hand-back, per mode of the block: the largest time the built-in controller needs in that mode's cases flown in the hidden world, rounded up to a whole second
  - also: an invalid plan costs the mode's recovery time off the block's budget, as a hand-back does (the built-in controller brings the robot to rest at the job's goal); the session applies it
  - also: a hold made invalid by a plan call's overrun is followed by the same recovery, flown at the hold's pose
- `jobs.finished`: a job is finished if its fixed window (end window, hold, capture) has closed when the block's clock reaches B; a job that is not finished or not started then scores 0 and, if it counts toward the score, stays in the mean; a refused request takes no time
  - unit: rule
  - also: a job ends, and the next job of the block's list is announced at once by its plan call (plan calls take no world time), when: a leg's or slew's end window closes (the upper end of score.end_window_s after its accepted done); a view's capture closes (jobs.capture_s after its accepted ready); a hold's duration has passed; the built-in controller's recovery after a hand-back or an invalid plan ends; or, for a refused job and an accepted request of the mode kind, at once; a leg, slew or view whose event is never accepted is flown until the block ends, unless a hand-back ends it first
- `safety.accepted_request`:
  - `outside_zones`: the plan is flown as any other; when the body leaves the zones the session ends
  - `mode_not_available`: the plan is not flown; the job scores 0 and counts as a request that was not refused; the block goes on
  - unit: what accepting a request costs

## What ends a block

A block ends when its clock reaches its budget (jobs.block_clock), after its last job, at the third hand-back (monitor.hand_back), at a protocol error at a plan call or at a job's step, at the watchdog, or at a cut by the wall rule inside the block; after a set-up overrun or a set-up failure each block ends at once. The rules, with their numbers:

- `safety.protocol_error`: a malformed or non-finite reply ends the block
  - unit: rule
  - also: an exception the controller raises in plan or step and a reply read whole whose form is refused (a non-finite number, a number beyond the float range, a wrong shape, an unknown event, an unknown refusal reason) are protocol errors; at a plan call or a job's step the block ends, elsewhere a notice; the process lives on with its state
  - also: while the controller is not in control a refused reply costs its notice only
  - also: in the check-out, while the controller is in control, a protocol error or a cut at the wall rule hands the rest of the check-out to the built-in controller
  - also: after a set-up overrun or a set-up failure the session has no controller: every plan call is answered as a protocol error, each block ends at once, and the session scores 0
- `interface.watchdog_s` = `0.32`
  - unit: s without an applied command before the propulsion stops
  - also: the count runs from the instant control returns to the controller (a block's start; after a recovery the built-in controller flew) and does not run while the built-in controller flies
  - also: the watchdog runs in the jobs of the scored blocks only; in the check-out the controller's box applies (leaving it hands the check-out to the built-in controller)
- `safety.watchdog_stop`: ends the block
  - unit: rule
- `interface.block_wall_rule`: at every step of a stretch (each block, from its first step to the step at which it ends; the start hold with the check-out; each gap between blocks; the time after the last block), the wall-clock time of the stretch's step replies so far may not exceed the stretch's world time so far plus 5 s; a call that would exceed it is cut there: in a block it is a protocol error and the block ends, outside a block the controller's process is ended and started again (interface.restart) and no job is lost
  - unit: rule
  - also: a block's stretch ends with the block, at the first frame outside it, whatever ended the block (its budget, its last job, the third hand-back, a protocol error, the watchdog); a recovery's tail after the budget belongs to the gap
- `interface.late_reply`: a step that uses more than its processor budget is not applied: the previous command stays, and the next n step calls are skipped, n = overrun divided by the step budget, rounded up (counted in step calls, not in world time)
  - unit: rule
  - also: the frames of skipped calls are not lost: their inertial, registration and fix lists arrive with the next frame

## What ends a session

A session ends early at the overspeed rule, at the zone rule (against the zone file in force that the session paper names) and at a contact with the cabin. A session whose set-up overran or failed scores nothing; the other sessions are flown. The rules, with their numbers:

- `safety.overspeed_m_s` = `0.5`
  - unit: m/s true linear speed; ends the session
- `safety.keep_in`: the body's bounding sphere inside the cabin's keep-in zones and outside its keep-out zones; a breach ends the session; the rule is tested at every step of the check-out and of the blocks, and not while the built-in controller holds the robot at the start or brings it to the start of a block
  - unit: rule
- `safety.body_radius_m` = `0.28`
  - unit: m, radius of the bounding sphere used by the zone rule
- `safety.contact`: a contact of the robot with the cabin ends the session
  - unit: rule
- `interface.setup_cpu_s` = `60.0`
  - unit: s of processor time per session
- `interface.setup_wall_s` = `180.0`
  - unit: s of wall clock per session
- `interface.restart`: The controller's process is started once per session. When it has ended (it exited, or the runner ended it because its reply stream could not go on: a reply line over interface.reply_line_cap_bytes, the stream closed, a line the client did not write, a call cut at its wall allowance or at a processor cap reached while waiting) it is started again before the next call, fresh: Controller() and setup(session) again with the same session paper, nothing of its earlier state handed to it. Every start of a session (the first and each restart) draws on the session's one set-up allowance, interface.setup_cpu_s of processor time and interface.setup_wall_s of wall clock summed over the starts (the interpreter's start, the import of controller.py, Controller() and setup all count); when it is used up: a set-up overrun (interface.overrun), that session scores 0 and no further start is made. The mass profile of the first setup stands; a later setup reply is checked for its form only.
  - unit: rule
- `interface.overrun`:
  - `setup (processor or wall)`: every start of the session (the first and each restart) draws on one allowance of interface.setup_cpu_s and interface.setup_wall_s summed over the starts; a start that would use more than what is left is cut there, and a restart that is needed when nothing is left is not made: a set-up overrun, that session scores 0; the other sessions are flown
  - `setup failure`: a start that cannot go on (the import of controller.py or Controller() raises, the process ends at its start) or a setup that raises or replies malformed: a set-up failure, that session scores 0; the other sessions are flown
  - `plan (processor or wall)`: the job is treated as one with an invalid plan: it scores 0 and the block goes on; the controller's process is ended only when the cap was reached while the runner waited for the reply
  - `step, processor`: the reply is not applied, the previous command stays, and the next n step calls are skipped, n = overrun divided by the step budget, rounded up
  - `step, wall (the running bound of its stretch)`: in a block the block ends, as for a protocol error; outside a block the process is ended and started again (interface.block_wall_rule); in the check-out the rest of the check-out goes to the built-in controller
  - unit: what an overrun costs

## What ends the run, and the deliverable's rules

The whole run scores nothing unless every condition of score.gates holds; the submission's caps and its integrity rule are the deliverable's rules (the integrity rule is not one of the conditions of score.gates, and breaking it too makes the whole run score nothing).

- `score.gates`:
  - a start of the controller comes up (it imports, builds Controller() and answers setup) in at least one session
  - the submission is inside its caps
  - the static import audit passes
  - at least 90 % of steps answered inside the processor budget
  - no process of the controller survives
  - the system-call filter installs (else a task error, not a score)
  - unit: list
  - also: the static import audit reads every .py file of the submitted controller folder without importing or running it; it passes when every import statement names a module whose top-level name is in the standard library of the verifying interpreter, is a library of interface.libraries, or is the name of a .py file or of a folder inside that folder; a relative import passes; a .py file that does not parse fails the audit
- `interface.on_time_share_gate` = `0.9`
  - unit: share of steps answered inside the processor budget, over the whole run
  - also: a skipped call counts as not on time
- `interface.submission_cap` = `{"files": 200, "megabytes": 16}`
  - unit: caps on the submission folder (regular files; symbolic links, special files and __pycache__ are not copied)
  - also: an over-cap submission scores 0 in every session
- `interface.submission_integrity`: the submission folder is final when the working session ends: if anything in it is written, renamed, linked or has its permissions changed after that, if the folder itself is replaced or changed after that, or if it is a symbolic link, is not a folder or leads outside the home folder, the reward is 0 and no session is flown
  - unit: rule
- `score.reward`: gates x mean of the six sessions
  - unit: formula

## The jobs

What the jobs are, what a view's goal means, and which requests must be refused:

- `jobs.kinds` = `["leg", "slew", "hold", "view"]`
  - unit: kinds
- `jobs.min_leg_m` = `0.15`
  - unit: m, smallest leg (above the end-point tolerance, so that 'done at once' fails)
- `jobs.min_slew_deg` = `10.0`
  - unit: degrees, smallest slew
- `jobs.min_view_step_deg` = `20.0`
  - unit: degrees, smallest turn between consecutive views (recorded runs: 20 to 35)
- `jobs.view_step_deg` = `[20.0, 40.0]`
  - unit: degrees between consecutive stations
- `jobs.hold_s` = `20.0`
  - unit: s, the duration of every hold in every session
- `jobs.capture_s` = `1.7`
  - unit: s the imaging camera needs for one picture after the event ready
- `jobs.camera_mount_m` = `[0.118, 0.0, -0.096]`
  - unit: m, optical centre of the imaging camera in body axes
- `jobs.view_pivot_radius_m` = `0.152`
  - unit: m, distance of that optical centre from the body origin
- `jobs.view_goal`:
  - `centre`: m, cabin frame: where the imaging camera's optical centre is to be
  - `axis`: unit vector, cabin frame: where the imaging camera's axis (the body's +x axis) is to point
  - `up`: unit vector, cabin frame, at right angles to axis: where the body's -z axis is to point
  - unit: the fields of a view's goal; the body pose that meets it puts the optical centre (jobs.camera_mount_m) at centre, the body's +x axis along axis and its -z axis along up
- `interface.session_paper_jobs`: the session paper lists, for each block, every job in order (id, kind, mode, goal, hold duration) and the block's budget in seconds; the budget is the value of jobs.block_budget on that list and can be recomputed from it
  - unit: rule
- `refusal.reasons` = `["outside_zones", "mode_not_available"]`
  - unit: the printed list
- `refusal.margin`: a request of the zone kind breaks a zone face by at least 0.05 m with the body's bounding sphere at the body pose that meets the goal (for a view: the pose the camera mounting gives); a feasible job keeps at least 0.05 m. A request of the mode kind names a mode that is not in the session paper's list of modes (release_modes)
  - unit: rule

## The block's budget

Each block's budget is the value of the formula below on that block's list, computed from the paper alone; the paper carries it and it can be recomputed from the paper. The four hard limits of each mode that the formula and the plan rules use:

- `jobs.block_budget`: B = S + sum over every job announced for the block of (k x t_j + F_j). t_j = max(trap(d_j, a, v), trap(theta_j, alpha, omega)) with trap(x, A, V) = 2 sqrt(x / A) if x <= V^2 / A, else x / V + V / A; d_j and theta_j are the distance and the turn angle between the body pose that meets the goal of the job before (here the last job before it in the block's list that is not a hold, whether it is a request or not; the block's start pose if there is none) and the body pose that meets this job's goal; v, a, omega, alpha are the four hard limits of the job's mode (a job that names a mode that is not in the session paper's list of modes takes the block's mode); a hold has t_j = 0. F_j = 2.5 s for a leg or slew (its end window), the duration for a hold, 1.7 s for a view (its capture). S = jobs.spin_up_term for a block that starts with a mode change, else 0. k = jobs.budget_factor.
  - unit: formula, seconds
- `jobs.budget_factor` = `1.35`
  - unit: k of jobs.block_budget
- `jobs.spin_up_term`: S = |n_new - n_old| / 200 + 2.0 s with n = 2000, 2500, 2800 rpm for fan levels 1, 2, 3 (4.5 s into the quiet mode, 3.5 s into the precision mode)
  - unit: formula, seconds
- `plan.speed_limit.nominal` = `{"linear_m_s": 0.2, "angular_rad_s": 0.1745}`
  - unit: limit on the plan's speeds
- `plan.speed_limit.quiet` = `{"linear_m_s": 0.02, "angular_rad_s": 0.0174}`
  - unit: limit on the plan's speeds
- `plan.speed_limit.precision` = `{"linear_m_s": 0.02, "angular_rad_s": 0.1745}`
  - unit: limit on the plan's speeds
- `plan.acceleration_limit` = `{"nominal": {"linear_m_s2": 0.0175, "angular_rad_s2": 0.1745}, "quiet": {"linear_m_s2": 0.002, "angular_rad_s2": 0.0174}, "precision": {"linear_m_s2": 0.01, "angular_rad_s2": 0.1745}}`
  - unit: hard limits of the mode's block in the flight-mode file of the release in force, undivided

## The plan and its validity rules

Every job is announced to plan(job, frame) before it is flown; the reply accepts it with a plan or refuses it. The rules a plan is checked by, in their order, what the supervisor does between setpoints, and when an event is accepted:

- `interface.plan_reply`:
  - accept with a list of setpoints, each a list [t, position, velocity, acceleration, attitude, body rate, angular acceleration]: t a number (s), attitude a list of 4 numbers (x, y, z, w), every other entry a list of 3 numbers
  - refuse with one reason of refusal.reasons
  - unit: forms
- `interface.plan_cpu_s` = `5.0`
  - unit: s of processor time per job
- `interface.plan_wall_s` = `20.0`
  - unit: s of wall clock per job
- `plan.max_setpoints` = `2000`
  - unit: setpoints
  - also: a plan of a leg, slew or view holds 1 to plan.max_setpoints setpoints: an empty plan breaks plan.max_setpoints (a hold's plan is not checked and may be empty)
- `plan.limit_rule`: every setpoint of a plan: the norms of velocity, acceleration, body rate and angular acceleration are at most the four hard limits of the job's mode (plan.speed_limit, plan.acceleration_limit), relative tolerance 1e-6; checked when plan returns, before the job starts; a plan that breaks it is invalid
  - unit: rule
- `plan.consistency`: setpoint times increase strictly and are at most 1.0 s apart; a setpoint integrated forward at its constant acceleration to the time of the next one agrees with it within 2 mm and 0.2 degrees in pose and 2 mm/s and 0.2 deg/s in velocity and body rate
  - unit: rule
- `plan.start_tolerance` = `{"position_m": 0.001, "attitude_deg": 0.1}`
  - unit: first setpoint against the body pose that meets the goal of the job before (the block's start pose for the first job)
  - also: the job before = the last job before it in the block that was not refused and was flown, by the controller or by a recovery of the built-in controller (after a hand-back or an invalid plan); a refused job, a refused request and an accepted request in a mode that is not in the session paper's list of modes leave the start pose as it was; for the first such job, the block's start pose
- `plan.end_tolerance`:
  - `leg, slew` = `{"position_m": 0.001, "attitude_deg": 0.1}`
  - `view`: the pose of the last setpoint is not checked
  - `every kind`: the last setpoint has zero velocity, acceleration, body rate and angular acceleration
  - unit: last setpoint against the job's goal
  - also: zero means at most 1e-12
- `plan.check_order`: plan.max_setpoints, plan.limit_rule, plan.consistency (the times, then the forward integration), plan.start_tolerance, plan.end_tolerance (the rates, then a leg's or slew's last pose); the first broken one is named with its setpoint index; a plan's attitudes are normalised before the check and an attitude of norm zero is a malformed reply
  - unit: rule
  - also: an attitude of norm zero means a norm of at most 1e-12 (the printed zero)
- `plan.interpolation`: between setpoints the monitor integrates the latest setpoint forward at constant acceleration; from the time of the last setpoint on, the reference is the last setpoint, held; in a hold the reference is the goal pose of the job before
  - unit: rule
- `interface.event_rule`: the event done or ready is accepted only when all five conditions hold, checked in this order: it is in an applied reply (on time, well formed, with a command); its frame is one of the job being flown; no event was accepted for this job before; it is of the job's kind (done for a leg or slew, ready for a view); its frame's time is at or after the time of the last setpoint of the job's accepted plan. An ignored event is named with its reason (step_not_applied, no_job, already_accepted, wrong_kind, before_last_setpoint) in the next frame's notices.
  - unit: rule

## What the sensors give: the printed ranges and statements

The frame's conventions, and the printed ranges and statements about the sensors, each as its own text says; what is drawn inside a range is not printed.

- `interface.conventions`:
  - `cabin_frame`: fix poses, job goals and zone files are in the cabin frame
  - `body_frame`: force and torque replies are in the body frame
  - `quaternion`: order x, y, z, w; the orientation of the named thing's axes in the cabin frame
  - `fix_pose`: the pose of the navigation camera, not of the body; the camera's mounting in the body is in the unit file
  - `inertial_samples`: in the inertial unit's own frame, as its driver publishes them (after the device's internal averaging, no offset removed); the unit's mounting in the body is in the unit file
  - `stamps`: one on-board clock; an inertial stamp is the driver's read time; a fix's stamp is the time of its image, not of its arrival; a registration's stamp is the time the localiser received the image (8 to 84 ms after the image's own time in the unit's records)
  - unit: conventions printed on the interface page
- `interface.registration_fields`:
  - `camera_id`: the image's number
  - `stamp`: the time the localiser received the image, as the maker's code sets it (8 to 84 ms after the image's own time in the unit's records)
  - unit: one entry of registrations[]: the localiser has started on this image
- `told.inertial_interval_ms` = `[2, 60]`
  - unit: ms between two inertial samples; the generator never leaves this range (a binding test)
- `told.inertial_noise` = `{"rate_mrad_s_max": 2.4, "accel_mm_s2_max": 20.0}`
  - unit: white-noise standard deviation per sample and axis with the impellers running, upper limits
- `told.accel_offset` = `{"per_axis_max_m_s2": 0.4, "change_per_minute_max_mm_s2": 6.0}`
  - unit: constant part of the accelerometer reading per axis, and how far it may differ from its value one minute earlier
- `told.gyro_offset_mrad_s_max` = `5.0`
  - unit: mrad/s per axis, constant part of the gyro reading apart from the cabin's rotation
- `told.cabin_rotation`:
  - `rate_mrad_s` = `[1.09, 1.17]`
  - `axis`: within 15 degrees of the cabin frame's negative y axis
  - unit: the cabin frame turns in inertial space at this rate, as measured on the unit's records; a gyro at rest in the cabin reads it, and the turning frame's terms act on motion relative to the cabin
- `told.fix_attempt_interval_s` = `[0.3, 4.6]`
  - unit: s between two images the localiser takes, outside the localiser's pauses; one image is processed at a time
- `told.fix_delay_s` = `[0.4, 3.2]`
  - unit: s from an image's time to the arrival of its result
- `told.fix_landmarks_min` = `10`
  - unit: smallest landmark count of a successful fix
- `told.fix_noise` = `{"landmarks_20_or_more": {"position_mm_per_axis": [1, 3], "attitude_mrad": [0.4, 1.4]}, "fewer_than_20": {"position_mm_per_axis": [10, 20], "attitude_mrad": [4, 9]}}`
  - unit: standard deviation of an ordinary fix
- `told.fix_gross_error` = `{"position_m": [0.1, 0.35], "attitude_deg": [5, 20], "share_max_fewer_than_20_landmarks": 0.25, "share_max_20_or_more": 0.01}`
  - unit: grossly wrong fixes: the whole error of such a fix against the truth lies in these ranges (it is not a displacement added on top of an ordinary fix's noise); the two shares are upper limits of the probability that a fix of that class is grossly wrong (the share realised in one session scatters about it)
- `told.fix_after_outage` = `{"stretch_s_min": 5, "probability_max_fewer_than_20_landmarks": 0.998}`
  - unit: the first successful fix after a stretch of at least stretch_s_min seconds without one (counted between the image times of two successive successful fixes) has fewer than 20 landmarks far more often than the other successful fixes do; probability_max_fewer_than_20_landmarks is an upper limit of the probability that such a fix has fewer than 20 landmarks
- `told.fixes_per_minute` = `{"outside_scored_blocks": [4, 33], "inside_scored_blocks": [14, 66]}`
  - unit: successful fixes per minute: outside the scored blocks (start hold, check-out, between blocks), read over the whole of a session outside its blocks, the session follows its day; inside a scored block the upper limit on a stretch without fixes (sessions.outage_cap_s) makes the rate higher, and its range is printed apart
- `sessions.outage_cap_s` = `{"nominal": 5, "precision": 5, "quiet": 0}`
  - unit: s, per mode of the block: the longest stretch between the image times of two successive successful fixes that touches a scored block, the stretch that is running when the block starts included (the results arrive later by the fix delay); a cap of 0 s means that no attempt fails: every attempt whose image time lies inside the block, and the last attempt before the block's start, succeeds; the check-out takes the nominal mode's cap

## What the unit and the cabin are like: the printed ranges and statements

The robot's response, its delays, the cabin's air and the impellers, and which of these the maintainers wrote:

- `told.response_ratio` = `[1.1, 2.6]`
  - unit: delivered over documented force or torque per signed axis and fan level
  - also: the ratio of a signed axis applies to the whole documented force or torque on that axis (what the documents' model of the unit gives for the command in force), and the sign of that whole documented force or torque decides which signed axis applies: while the unit holds a steady force or torque on an axis, a change of command that does not reverse its sign meets the ratio of the holding side, whichever way the change goes
  - also: each nozzle's command moves in whole steps: the maker's allocation turns the commanded force and torque into a whole count from 0 to 255 for each nozzle (8 bits), so a small change of command can leave every nozzle's count as it was
- `told.force_response_level3` = `{"+x": [1.5, 1.64], "-x": [1.36, 1.47], "+y": [1.53, 1.66], "-y": [1.49, 1.61], "+z": [1.51, 1.65], "-z": [1.32, 1.44]}`
  - unit: delivered over documented force per signed axis at the highest fan level: for each signed axis a range that holds the unit's response in every session (a narrower statement than told.response_ratio for these six values)
- `told.lowest_level_response`: the unit's response at the lowest fan level (quiet) equals its response at the middle fan level (nominal), per signed axis, in force and in torque; this relation is written by the maintainers (the given records hold only one short stretch at the lowest level, in which the built-in controller flew one short move), not measured
  - unit: statement printed in the documents: how the response at the lowest fan level relates to the middle level's
- `told.lag_command_to_motion_ms` = `[40, 80]`
  - unit: ms: the pure delay that best aligns the gyro's answer with the commanded torque (the inertial unit's own filter included)
- `told.steady_force_N` = `0.025`
  - unit: N per axis, upper bound of the slowly varying force on the robot
- `told.steady_torque_Nm` = `0.001`
  - unit: N m per axis, upper bound
- `told.steady_force_change_mN_per_min` = `3.0`
  - unit: mN per cabin axis: the part of the steady force that is fixed in the cabin at any instant differs from its value one minute earlier by at most this; the other part is fixed in the body's axes and constant over a session, so the sum as the body feels it also changes when the robot turns
- `told.arm_mass_kg` = `[0.68, 1.0]`
  - unit: kg, the documents' figures for the arm, rounded outward (the description file's links sum to 0.688 kg; the documents' prose says about 1 kg)
- `told.mode_change_s` = `{"between_levels": 4.0, "from_rest": 15.0}`
  - unit: s, upper bounds of the impellers' speed change
- `world.ready_thresholds_rad_s` = `{"enter": 10.0, "leave": 30.0}`
  - unit: rad/s
- `told.models_statement`: The robot's response at the two higher fan levels, its delays, the inertial unit's timing, noise and offsets, the cabin's rotation, and the delay, landmark counts and noise of the fixes are measured on this unit's real flights. How sparse the fixes are in a session follows one real day of this unit, through the ratio between the built-in estimator's corrections and the fixes measured on this unit's flights; for the two latest sessions that sparsity is the maintainers' choice, bounded by the same ratio. The supervisor's measurement noise, the upper limit on a stretch without fixes inside a scored block, the share and size of grossly wrong fixes within their measured limits, the recovery time after a hand-back, the slow change of the cabin's air, the response and the inertial noise at the lowest fan level, and the use of earlier sensor statistics for the two latest sessions are the maintainers' choices, each bounded by this unit's real sessions.
  - unit: sentence

## The score

Every measure is taken on the hidden world's true state. The terms, their formulas and their parts:

- `score.measured_on`: the true state of the hidden world
  - unit: rule
- `score.weights` = `{"legs": 0.25, "slews": 0.15, "holds": 0.15, "views": 0.35, "refusals": 0.1}`
  - unit: weights of a session
- `score.leg`: completed x L(end position error) x L(end attitude error) x (0.8 + 0.2 O)
  - unit: formula
- `score.slew`: completed x L(end attitude error) x L(end position error) x (0.8 + 0.2 O)
  - unit: formula
- `score.hold`: completed(job before) x L(position RMS about the hold's target pose) x L(attitude RMS about it) x L(speed RMS) x L(rate RMS); the target pose is the goal of the job before (as plan.start_tolerance defines it); in completed(job before) the job before is the last leg, slew or view before the hold in its block's list, whatever became of it (a request is not one of them; where there is none, the factor is 1); a hold that is not itself completed (score.completed) scores 0
  - unit: formula
- `score.view`: captured x L(pointing error) x L(optical-centre displacement) x L(rate RMS in the capture)
  - unit: formula
- `score.L`: 1 at the full end, 0 at the zero end, linear between, and clamped: 1 for a measure at or below the full end, 0 for one at or above the zero end; a measure whose cell does not enter the product (its zero end below twice its full end, score.completed) counts as 1 in its kind's formula
  - unit: definition
- `score.O`: L(overshoot beyond the goal along the move)
  - unit: definition
- `score.end_window_s` = `[0.5, 2.5]`
  - unit: s after done over which the true end errors are averaged; the controller keeps holding
- `score.measures`:
  - `end_pos`: m: a leg's or slew's end position error, the mean over the end window (score.end_window_s after the accepted done) of the distance of the body origin from the goal position
  - `end_att`: rad: its end attitude error, the mean over the same window of the angle between the body's attitude and the goal attitude
  - `overshoot`: a leg's in m, the largest excursion of the body origin beyond the goal along the move (the line from the body pose that meets the goal of the job before, the block's start pose for the first job, to this goal); a slew's in rad, the largest turn beyond the goal about the move's rotation axis; each from the job's start to the end window's close, 0 if there is none
  - `pos_rms`: m: a hold's RMS distance of the body origin from the hold's target position over the hold
  - `att_rms`: rad: a hold's RMS angle between the body's attitude and the target attitude over the hold
  - `speed_rms`: m/s: a hold's RMS speed of the centre of mass relative to the cabin over the hold
  - `rate_rms`: rad/s: the RMS angular rate of the body relative to the cabin, over the hold for a hold and over the capture for a view
  - `pointing`: rad: a view's mean angle, over the capture (jobs.capture_s from the accepted ready), between the imaging camera's axis (the body's +x axis) and the goal's axis
  - `centre`: m: a view's mean distance, over the capture, of the imaging camera's optical centre (jobs.camera_mount_m) from the goal's centre
  - unit: the measures of the cells of score.bands, each with its unit; all are taken on the hidden world's true state
- `score.bands` = `{"cells": {"hold/nominal/att_rms": {"full_end": 0.0017305410275683774, "zero_end": 0.08726646259971647}, "hold/nominal/pos_rms": {"full_end": 0.03323235504624486, "zero_end": 0.075}, "hold/nominal/rate_rms": {"full_end": 0.00045285747959079593, "zero_end": 0.0012337825297712732}, "hold/nominal/speed_rms": {"full_end": 0.007003210988598642, "zero_end": 0.009366869260475124}, "hold/precision/att_rms": {"full_end": 0.002446996976940676, "zero_end": 0.08726646259971647}, "hold/precision/pos_rms": {"full_end": 0.040335987111628085, "zero_end": 0.075}, "hold/precision/rate_rms": {"full_end": 0.0006294909953286212, "zero_end": 0.0012357682034256586}, "hold/precision/speed_rms": {"full_end": 0.008318879827604861, "zero_end": 0.012148531157801304}, "hold/quiet/att_rms": {"full_end": 0.0010338066699514409, "zero_end": 0.08726646259971647}, "hold/quiet/pos_rms": {"full_end": 0.011383891788986307, "zero_end": 0.075}, "hold/quiet/rate_rms": {"full_end": 0.0003344672730756743, "zero_end": 0.0009334840647439746}, "hold/quiet/speed_rms": {"full_end": 0.0037061272447133694, "zero_end": 0.008283432855541692}, "leg/nominal/end_att": {"full_end": 0.0022512023541130565, "zero_end": 0.08726646259971647}, "leg/nominal/end_pos": {"full_end": 0.04514305389948954, "zero_end": 0.075}, "leg/nominal/overshoot": {"full_end": 0.035451918661079024, "zero_end": 0.075}, "leg/precision/end_att": {"full_end": 0.00215536908968003, "zero_end": 0.08726646259971647}, "leg/precision/end_pos": {"full_end": 0.030311313066614855, "zero_end": 0.075}, "leg/precision/overshoot": {"full_end": 0.03751547661154336, "zero_end": 0.03375306338975648}, "leg/quiet/end_att": {"full_end": 0.0013812003678192227, "zero_end": 0.08726646259971647}, "leg/quiet/end_pos": {"full_end": 0.015309447761539747, "zero_end": 0.075}, "leg/quiet/overshoot": {"full_end": 0.009441126933934488, "zero_end": 0.012671035752214323}, "slew/nominal/end_att": {"full_end": 0.002034278465868498, "zero_end": 0.08726646259971647}, "slew/nominal/end_pos": {"full_end": 0.03745494296899089, "zero_end": 0.075}, "slew/nominal/overshoot": {"full_end": 0.0026236770634682025, "zero_end": 0.0015211870496902623}, "slew/precision/end_att": {"full_end": 0.002394049658961959, "zero_end": 0.08726646259971647}, "slew/precision/end_pos": {"full_end": 0.03337956925248774, "zero_end": 0.075}, "slew/precision/overshoot": {"full_end": 0.002129470305113701, "zero_end": 0.001113598981634917}, "slew/quiet/end_att": {"full_end": 0.0010553256053358897, "zero_end": 0.08726646259971647}, "slew/quiet/end_pos": {"full_end": 0.01247146041933409, "zero_end": 0.075}, "slew/quiet/overshoot": {"full_end": 0.000877279284199407, "zero_end": 0.01696046045606303}, "view/nominal/centre": {"full_end": 0.049625183404483825, "zero_end": 0.033}, "view/nominal/pointing": {"full_end": 0.0029591104530330816, "zero_end": 0.17453292519943295}, "view/nominal/rate_rms": {"full_end": 0.002801918767410978, "zero_end": 0.0022007497671787018}}}`
  - unit: every full and zero end in numbers, each in the unit of its measure (score.measures)
- `score.completed`: a leg or slew is completed if its event done was accepted, its end window closed inside the block's budget, the end-point check passed, no hand-back occurred, and each of its own measures is inside its completion limit; a view is captured if its event ready was accepted, the capture closed inside the budget and each of its own measures is inside its completion limit; the completion limit of a measure is its zero end, or, where the band rules dropped the measure from the product (zero end below twice the full end), twice its full end, except the optical-centre displacement of a view, whose completion limit is always its zero end (0.033 m); a hold is tested in the same way: it is completed if it was flown for its whole duration inside the block's budget with no hand-back and each of its own measures is inside its completion limit; a measure is inside its completion limit when it is at most that limit
  - unit: rule
- `score.refusals`: J x (share of accepted feasible jobs completed), J = share of requests refused + share of feasible jobs accepted - 1, and 0 where J is below 0; the feasible jobs are those that count toward the score, the requests every request of the session; a refused feasible job and a flown request score 0 themselves
  - unit: formula
- `score.session`: weighted mean over the kinds of the jobs that count (a kind without jobs drops out, weights renormalised); jobs that count and are lost to a hand-back, a stop or the end of the session stay in the mean with 0; a session whose setup overran scores 0
  - unit: formula
  - also: not every announced job need count toward the score: the kinds' means and the refusal term's shares of feasible jobs run over the jobs that count, and which jobs count is not told; every request counts in the refusal term
- `score.grader_output`: the grader writes the reward and a summary per session and kind; no per-job truth, no band end that is not in the acceptance specification, no hidden parameter
  - unit: rule

The cells of score.bands, each with the unit of its measure (score.measures), both ends, whether the cell enters the product, and its completion limit, derived here from score.bands by the public rule of score.completed (a cell whose zero end is below twice its full end does not enter the product and its completion limit is twice its full end; otherwise the limit is the zero end; the optical-centre displacement of a view always has its zero end as its limit):

| cell | unit | full end | zero end | in the product | completion limit |
|---|---|---|---|---|---|
| `hold/nominal/att_rms` | rad | 0.0017305410275683774 | 0.08726646259971647 | yes | 0.08726646259971647 |
| `hold/nominal/pos_rms` | m | 0.03323235504624486 | 0.075 | yes | 0.075 |
| `hold/nominal/rate_rms` | rad/s | 0.00045285747959079593 | 0.0012337825297712732 | yes | 0.0012337825297712732 |
| `hold/nominal/speed_rms` | m/s | 0.007003210988598642 | 0.009366869260475124 | no | 0.014006421977197285 |
| `hold/precision/att_rms` | rad | 0.002446996976940676 | 0.08726646259971647 | yes | 0.08726646259971647 |
| `hold/precision/pos_rms` | m | 0.040335987111628085 | 0.075 | no | 0.08067197422325617 |
| `hold/precision/rate_rms` | rad/s | 0.0006294909953286212 | 0.0012357682034256586 | no | 0.0012589819906572424 |
| `hold/precision/speed_rms` | m/s | 0.008318879827604861 | 0.012148531157801304 | no | 0.016637759655209722 |
| `hold/quiet/att_rms` | rad | 0.0010338066699514409 | 0.08726646259971647 | yes | 0.08726646259971647 |
| `hold/quiet/pos_rms` | m | 0.011383891788986307 | 0.075 | yes | 0.075 |
| `hold/quiet/rate_rms` | rad/s | 0.0003344672730756743 | 0.0009334840647439746 | yes | 0.0009334840647439746 |
| `hold/quiet/speed_rms` | m/s | 0.0037061272447133694 | 0.008283432855541692 | yes | 0.008283432855541692 |
| `leg/nominal/end_att` | rad | 0.0022512023541130565 | 0.08726646259971647 | yes | 0.08726646259971647 |
| `leg/nominal/end_pos` | m | 0.04514305389948954 | 0.075 | no | 0.09028610779897908 |
| `leg/nominal/overshoot` | m | 0.035451918661079024 | 0.075 | yes | 0.075 |
| `leg/precision/end_att` | rad | 0.00215536908968003 | 0.08726646259971647 | yes | 0.08726646259971647 |
| `leg/precision/end_pos` | m | 0.030311313066614855 | 0.075 | yes | 0.075 |
| `leg/precision/overshoot` | m | 0.03751547661154336 | 0.03375306338975648 | no | 0.07503095322308671 |
| `leg/quiet/end_att` | rad | 0.0013812003678192227 | 0.08726646259971647 | yes | 0.08726646259971647 |
| `leg/quiet/end_pos` | m | 0.015309447761539747 | 0.075 | yes | 0.075 |
| `leg/quiet/overshoot` | m | 0.009441126933934488 | 0.012671035752214323 | no | 0.018882253867868976 |
| `slew/nominal/end_att` | rad | 0.002034278465868498 | 0.08726646259971647 | yes | 0.08726646259971647 |
| `slew/nominal/end_pos` | m | 0.03745494296899089 | 0.075 | yes | 0.075 |
| `slew/nominal/overshoot` | rad | 0.0026236770634682025 | 0.0015211870496902623 | no | 0.005247354126936405 |
| `slew/precision/end_att` | rad | 0.002394049658961959 | 0.08726646259971647 | yes | 0.08726646259971647 |
| `slew/precision/end_pos` | m | 0.03337956925248774 | 0.075 | yes | 0.075 |
| `slew/precision/overshoot` | rad | 0.002129470305113701 | 0.001113598981634917 | no | 0.004258940610227402 |
| `slew/quiet/end_att` | rad | 0.0010553256053358897 | 0.08726646259971647 | yes | 0.08726646259971647 |
| `slew/quiet/end_pos` | m | 0.01247146041933409 | 0.075 | yes | 0.075 |
| `slew/quiet/overshoot` | rad | 0.000877279284199407 | 0.01696046045606303 | yes | 0.01696046045606303 |
| `view/nominal/centre` | m | 0.049625183404483825 | 0.033 | no | 0.033 |
| `view/nominal/pointing` | rad | 0.0029591104530330816 | 0.17453292519943295 | yes | 0.17453292519943295 |
| `view/nominal/rate_rms` | rad/s | 0.002801918767410978 | 0.0022007497671787018 | no | 0.005603837534821956 |

Cells that do not enter the product by that rule: `hold/nominal/speed_rms`, `hold/precision/pos_rms`, `hold/precision/rate_rms`, `hold/precision/speed_rms`, `leg/nominal/end_pos`, `leg/precision/overshoot`, `leg/quiet/overshoot`, `slew/nominal/overshoot`, `slew/precision/overshoot`, `view/nominal/centre`, `view/nominal/rate_rms` (11 of 33).

## What is not scored

Nothing of the start hold, the check-out, the transfers between blocks or the time after the last block is scored: the controller is not in control in the start hold and the transfers, nothing in the check-out is scored, and the score's terms are taken over the jobs of the blocks. The rules that end a session or the run act in every stretch where their texts say so.
