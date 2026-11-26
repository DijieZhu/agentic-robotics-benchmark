# Interface page: the controller, the wire, the frame and the replies

How the submitted controller is run and what passes between it and the acceptance's runner. Entries are printed as in the acceptance specification (payload/acceptance/acceptance_specification.md): the key in backticks, its value, its unit or meaning (unit:) and its sentences (also:).

Paths on the task's pages (payload/..., controller/...) are relative to the folder that holds payload and controller; the zone_file of a session paper is a path relative to payload.

## The deliverable and its sandbox

The deliverable is a folder holding controller.py; the runner builds its class Controller with no argument at each start of the controller's process (interface.restart), in a read-only copy of the folder, with HOME and TMPDIR naming an empty writable folder of its own. The instruction names where the folder goes.

- `interface.class`: Controller with setup(session), plan(job, frame), step(frame)
  - unit: names
- `interface.libraries`:
  - python standard library
  - numpy
  - scipy
  - unit: names
- `interface.submission_cap` = `{"files": 200, "megabytes": 16}`
  - unit: caps on the submission folder (regular files; symbolic links, special files and __pycache__ are not copied)
  - also: an over-cap submission scores 0 in every session
- `interface.submission_integrity`: the submission folder is final when the working session ends: if anything in it is written, renamed, linked or has its permissions changed after that, if the folder itself is replaced or changed after that, or if it is a symbolic link, is not a folder or leads outside the home folder, the reward is 0 and no session is flown
  - unit: rule
- `interface.sandbox`:
  - `processors` = `4`
  - `address_space_per_process_GiB` = `8`
  - `threads_per_numerics_library` = `1`
  - `network`: none (no socket of any family)
  - `reads`: its own controller folder's read-only copy, the interpreter's installation and the system's libraries
  - `writes`: its own HOME and TMPDIR
  - unit: processors and memory of the controller's sandbox
  - also: the controller's process runs as python3 -I -B: the interpreter reads no PYTHON environment variable and no user site folder and writes no bytecode, and the controller folder's copy is put first on the import path
  - also: a system-call filter, inherited by every process the controller starts, refuses with an error (the process goes on): every socket; sending a signal to any process, itself and its own children included; tracing or reading another process's memory; new namespaces; leaving the process group or the session; mounts and changes of the root folder; io_uring; and the kernel's bpf, performance-counter, key-ring, user-fault, page-moving and file-handle calls; system calls of another architecture are refused as well. Starting threads and child processes and waiting for them is not refused
  - also: the 4 processors are a quota of the acceptance's machine, shared with the acceptance's own processes; a count of processors made inside the sandbox reads the host's and may be larger; the step budget counts processor time over all threads and children, so more processors give no more budget
  - also: at the acceptance the controller's process runs under an unprivileged account of its own, without supplementary groups (neither root nor the agent's account); in the self-test it runs under the account that starts the self-test

## The wire

JSON lines over the standard streams: the runner writes one request line, the controller's process answers one reply line. The reply line is written by the runner's client around the method's return value; the controller's own prints go to the error stream. The request lines and the reply envelope, with their keys as they pass on the wire (one example of each is in controller/example_messages):

- setup: {`op`, `session`}
- plan: {`op`, `job`, `frame`}
- step: {`op`, `frame`}
- end: {`op`}
- the reply envelope: {`ok`, `result`} with the method's return value under `result`; an exception in the controller is answered with {`ok`: false, `error`: its type and message}.

The replies:

- setup: null, or {`mass_profile`: {`mass_kg`, `com_m`, `inertia`}} (`mass_kg` positive; the sizes of `com_m` and `inertia` as in the example message reply_setup_mass_profile.json).
- plan: {`setpoints`: a list of setpoints} to accept (a hold may send an empty list), or {`refuse`: one reason of refusal.reasons} to refuse; in the words of interface.plan_reply: accept with a list of setpoints, each a list [t, position, velocity, acceleration, attitude, body rate, angular acceleration]: t a number (s), attitude a list of 4 numbers (x, y, z, w), every other entry a list of 3 numbers.
- step: {`force`, `torque`, `event`}, force and torque in body axes; in the words of interface.command_fields: event: the string "none", "done" or "ready" (left out: "none"; any other value, null included, is a malformed reply); force and torque both null are no new command.

- `interface.setup_reply`: nothing, or a mass profile (mass, centre of mass, inertia) to be sent to the robot before the first block
  - unit: form
  - also: the mass and the inertia of a profile that is sent are taken by the robot's built-in controller; the robot's force allocation keeps the centre of mass of the world configuration's inertia entry on file (cabinflyer/config/worlds/station.config, the same in every edition given) whether a profile is sent or not: the centre of mass of a profile that is sent does not move it
- `interface.plan_reply`:
  - accept with a list of setpoints, each a list [t, position, velocity, acceleration, attitude, body rate, angular acceleration]: t a number (s), attitude a list of 4 numbers (x, y, z, w), every other entry a list of 3 numbers
  - refuse with one reason of refusal.reasons
  - unit: forms
- `interface.command_fields`:
  - force[3] N body axes
  - torque[3] N m body axes
  - event: the string "none", "done" or "ready" (left out: "none"; any other value, null included, is a malformed reply)
  - unit: field names of a step reply
  - also: force and torque both null: not an applied reply (the previous command stays, the watchdog runs, an event in it is ignored as step_not_applied)
- `refusal.reasons` = `["outside_zones", "mode_not_available"]`
  - unit: the printed list
- `interface.reply_line_cap_bytes` = `1048576`
  - unit: bytes per reply line

## The frame

plan(job, frame) and step(frame) receive the frame of their time. While the controller is not in control (the start hold, a recovery of the built-in controller, the transfers between blocks) frames are delivered with in_control false and its replies are not applied.

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
- `interface.frame_field_meanings`:
  - `camera.busy`: true while the imaging camera takes a view's picture: from the first frame after an accepted ready until jobs.capture_s after that ready; else false
  - `propulsion.module_state`: per impeller, ready or ramping: it becomes ramping when the impeller's speed is further from the speed of the commanded fan level than world.ready_thresholds_rad_s's leave value, and ready again when it is within its enter value
  - `block.budget_s`: in the frames of a block's jobs, the block's budget B (jobs.block_budget); null in every other frame
  - `block.time_left_s`: in the frames of a block's jobs, B less the world time since the block's first step; null in every other frame
  - `job_id`: in the frames of a job (its plan call's frame and its steps), that job's id as the session paper gives it; in the check-out's frames, over the whole of sessions.checkout_s, the string "checkout" (also after the robot has left the check-out's box and the built-in controller holds it); null in every other frame (the start hold, a recovery of the built-in controller, the transfers between blocks, the time after the last block)
  - `propulsion.impeller_speed`: rad/s: the speed of each of the two impellers at the step
  - unit: what six fields of a step frame mean
- `interface.inertial_fields`:
  - `stamp`: s
  - `rate`: rad/s, three axes of the inertial unit
  - `accel`: m/s^2, three axes of the inertial unit
  - unit: one entry of inertial[]
- `interface.registration_fields`:
  - `camera_id`: the image's number
  - `stamp`: the time the localiser received the image, as the maker's code sets it (8 to 84 ms after the image's own time in the unit's records)
  - unit: one entry of registrations[]: the localiser has started on this image
- `interface.fix_fields`:
  - `camera_id`: the image's number, the same as in its registration notice
  - `stamp`: time of the image
  - `landmarks`: number of matched landmarks; 0 means the image was not localised and no pose follows
  - `position`: m, cabin frame
  - `attitude`: quaternion
  - unit: one entry of fixes[]
- `interface.frame_notices`:
  - `field`: notices[]
  - `entry_forms`:
    - `plan`: {kind: plan, job_id, valid: true} or {kind: plan, job_id, valid: false, rule: plan.max_setpoints | plan.limit_rule | plan.consistency | plan.start_tolerance | plan.end_tolerance | interface.plan_cpu_s | interface.plan_wall_s, setpoint: index or null}
    - `event_ignored`: {kind: event_ignored, job_id or null, event: done | ready, t: the frame's time, reason: step_not_applied | no_job | already_accepted | wrong_kind | before_last_setpoint}
    - `protocol_error`: {kind: protocol_error, call: plan | step, what: text}
  - `where`: a list in every frame the controller receives (plan and step), empty in most; a notice is given once, in the first frame sent after its cause
  - unit: one more field of every frame the controller receives (plan and step): what the runner says about the controller's last replies
- `interface.not_on_the_wire`: The built-in estimator's output is not on the wire: it is the part this deliverable replaces, not a withheld document. Optical-flow and marker measurements are not on the wire either.
  - unit: sentence of the interface page
- `interface.conventions`:
  - `cabin_frame`: fix poses, job goals and zone files are in the cabin frame
  - `body_frame`: force and torque replies are in the body frame
  - `quaternion`: order x, y, z, w; the orientation of the named thing's axes in the cabin frame
  - `fix_pose`: the pose of the navigation camera, not of the body; the camera's mounting in the body is in the unit file
  - `inertial_samples`: in the inertial unit's own frame, as its driver publishes them (after the device's internal averaging, no offset removed); the unit's mounting in the body is in the unit file
  - `stamps`: one on-board clock; an inertial stamp is the driver's read time; a fix's stamp is the time of its image, not of its arrival; a registration's stamp is the time the localiser received the image (8 to 84 ms after the image's own time in the unit's records)
  - unit: conventions printed on the interface page
- `world.ready_thresholds_rad_s` = `{"enter": 10.0, "leave": 30.0}`
  - unit: rad/s
- `sessions.start_hold_s` = `20.0`
  - unit: s of hold by the built-in controller at the start of a session, frames delivered, the node not in control
- `safety.protocol_error`: a malformed or non-finite reply ends the block
  - unit: rule
  - also: an exception the controller raises in plan or step and a reply read whole whose form is refused (a non-finite number, a number beyond the float range, a wrong shape, an unknown event, an unknown refusal reason) are protocol errors; at a plan call or a job's step the block ends, elsewhere a notice; the process lives on with its state
  - also: while the controller is not in control a refused reply costs its notice only
  - also: in the check-out, while the controller is in control, a protocol error or a cut at the wall rule hands the rest of the check-out to the built-in controller
  - also: after a set-up overrun or a set-up failure the session has no controller: every plan call is answered as a protocol error, each block ends at once, and the session scores 0

## Steps, budgets and caps

- `interface.step_period_ms` = `16.0`
  - unit: ms between two step calls (62.5 Hz), constant
- `interface.design_rate_hz` = `62.5`
  - unit: Hz, the design rate of the control loop
- `interface.step_cpu_ms` = `13.5`
  - unit: ms of processor time per step
  - also: a call is charged everything the controller's process group used since the previous reply (threads that exited, waited-for children, live members)
- `interface.late_reply`: a step that uses more than its processor budget is not applied: the previous command stays, and the next n step calls are skipped, n = overrun divided by the step budget, rounded up (counted in step calls, not in world time)
  - unit: rule
  - also: the frames of skipped calls are not lost: their inertial, registration and fix lists arrive with the next frame
- `interface.on_time_share_gate` = `0.9`
  - unit: share of steps answered inside the processor budget, over the whole run
  - also: a skipped call counts as not on time
- `interface.plan_cpu_s` = `5.0`
  - unit: s of processor time per job
- `interface.plan_wall_s` = `20.0`
  - unit: s of wall clock per job
- `interface.setup_cpu_s` = `60.0`
  - unit: s of processor time per session
- `interface.setup_wall_s` = `180.0`
  - unit: s of wall clock per session
- `interface.block_wall_rule`: at every step of a stretch (each block, from its first step to the step at which it ends; the start hold with the check-out; each gap between blocks; the time after the last block), the wall-clock time of the stretch's step replies so far may not exceed the stretch's world time so far plus 5 s; a call that would exceed it is cut there: in a block it is a protocol error and the block ends, outside a block the controller's process is ended and started again (interface.restart) and no job is lost
  - unit: rule
  - also: a block's stretch ends with the block, at the first frame outside it, whatever ended the block (its budget, its last job, the third hand-back, a protocol error, the watchdog); a recovery's tail after the budget belongs to the gap
- `interface.watchdog_s` = `0.32`
  - unit: s without an applied command before the propulsion stops
  - also: the count runs from the instant control returns to the controller (a block's start; after a recovery the built-in controller flew) and does not run while the built-in controller flies
  - also: the watchdog runs in the jobs of the scored blocks only; in the check-out the controller's box applies (leaving it hands the check-out to the built-in controller)
- `interface.overrun`:
  - `setup (processor or wall)`: every start of the session (the first and each restart) draws on one allowance of interface.setup_cpu_s and interface.setup_wall_s summed over the starts; a start that would use more than what is left is cut there, and a restart that is needed when nothing is left is not made: a set-up overrun, that session scores 0; the other sessions are flown
  - `setup failure`: a start that cannot go on (the import of controller.py or Controller() raises, the process ends at its start) or a setup that raises or replies malformed: a set-up failure, that session scores 0; the other sessions are flown
  - `plan (processor or wall)`: the job is treated as one with an invalid plan: it scores 0 and the block goes on; the controller's process is ended only when the cap was reached while the runner waited for the reply
  - `step, processor`: the reply is not applied, the previous command stays, and the next n step calls are skipped, n = overrun divided by the step budget, rounded up
  - `step, wall (the running bound of its stretch)`: in a block the block ends, as for a protocol error; outside a block the process is ended and started again (interface.block_wall_rule); in the check-out the rest of the check-out goes to the built-in controller
  - unit: what an overrun costs
- `interface.restart`: The controller's process is started once per session. When it has ended (it exited, or the runner ended it because its reply stream could not go on: a reply line over interface.reply_line_cap_bytes, the stream closed, a line the client did not write, a call cut at its wall allowance or at a processor cap reached while waiting) it is started again before the next call, fresh: Controller() and setup(session) again with the same session paper, nothing of its earlier state handed to it. Every start of a session (the first and each restart) draws on the session's one set-up allowance, interface.setup_cpu_s of processor time and interface.setup_wall_s of wall clock summed over the starts (the interpreter's start, the import of controller.py, Controller() and setup all count); when it is used up: a set-up overrun (interface.overrun), that session scores 0 and no further start is made. The mass profile of the first setup stands; a later setup reply is checked for its form only.
  - unit: rule
- `interface.verify_timeout_s` = `18000`
  - unit: s; by the formula registered before its numbers (per session: the one set-up allowance, the plan calls' wall caps, the session's world time, 5 s per stretch, twice the grader's time per step; six sessions; 600 s fixed; rounded up to 600 s), not from the reference
  - also: if the verification reaches this limit it is void: a task error of the run, not a score (no reward is written)

## Plans and events

- `plan.max_setpoints` = `2000`
  - unit: setpoints
  - also: a plan of a leg, slew or view holds 1 to plan.max_setpoints setpoints: an empty plan breaks plan.max_setpoints (a hold's plan is not checked and may be empty)
- `plan.check_order`: plan.max_setpoints, plan.limit_rule, plan.consistency (the times, then the forward integration), plan.start_tolerance, plan.end_tolerance (the rates, then a leg's or slew's last pose); the first broken one is named with its setpoint index; a plan's attitudes are normalised before the check and an attitude of norm zero is a malformed reply
  - unit: rule
  - also: an attitude of norm zero means a norm of at most 1e-12 (the printed zero)
- `interface.event_rule`: the event done or ready is accepted only when all five conditions hold, checked in this order: it is in an applied reply (on time, well formed, with a command); its frame is one of the job being flown; no event was accepted for this job before; it is of the job's kind (done for a leg or slew, ready for a view); its frame's time is at or after the time of the last setpoint of the job's accepted plan. An ignored event is named with its reason (step_not_applied, no_job, already_accepted, wrong_kind, before_last_setpoint) in the next frame's notices.
  - unit: rule

The other plan rules (their numbers, the hard limits of each mode) and the block budget's formula are in the acceptance specification.

## The session paper

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
- `interface.session_paper_jobs`: the session paper lists, for each block, every job in order (id, kind, mode, goal, hold duration) and the block's budget in seconds; the budget is the value of jobs.block_budget on that list and can be recomputed from it
  - unit: rule

The paper's keys with their nesting: `session`, `release`, `release_modes`, `checkout_mode`, `start_pose` {`position`, `attitude`}, `zone_file`, `battery_slots` {`occupied`, `of`}, `equipment`, `time_window` {`planned_minutes`}, `blocks` [{`name`, `mode`, `start_pose`, `jobs`, `budget_s`}], `window_s`. Their forms and two filled examples: payload/acceptance/session_paper_format.md.

## The self-test

- `interface.self_test`: a toy rigid body with toy sensors in the same message formats and conventions, with arbitrary noise and no calibrated value of this robot; for checking protocol and signs only
  - unit: what controller/selftest holds

How it is run (its own opening text, controller/selftest/selftest.py): python3 selftest.py PATH/TO/submission/controller [--json FILE] [--printed FILE] [--log FILE] [--quiet]

What its toy session does differently from the acceptance's sessions, in its opening text's own words: It has no supervisor: no hand-backs and no recoveries; a job whose plan is invalid (an overrun of a hold's plan call included) puts the toy body at the job's goal at once, where the acceptance's built-in controller flies a recovery that takes the mode's recovery time off the block's budget. Between blocks the toy body rests for a toy time instead of being flown to the next block's start. Its zones and its check-out box are toy boxes. Like the acceptance's session it acts on the wire's verdict on every plan call (valid, refused, invalid, protocol error), ends a block at its budget, after its last job, at a protocol error or at the watchdog, and hands the wire its record of each block's end, so a block's stretch of the wall rule ends with the block.

Its wire is the acceptance runner's code; its paper has the keys of the acceptance's paper and two keys of its own (zones: its toy boxes, given inline since the toy has no zone file; toy: true); the values that belong to the robot are toy values there. To have its rules judge plans with the printed values, give them to its printed-values option in a JSON file by their keys; the keys it takes and their printed values, from the entries of the acceptance specification:

```json
{
 "jobs.camera_mount_m": [
  0.118,
  0.0,
  -0.096
 ],
 "jobs.capture_s": 1.7,
 "jobs.spin_up_term": "S = |n_new - n_old| / 200 + 2.0 s with n = 2000, 2500, 2800 rpm for fan levels 1, 2, 3 (4.5 s into the quiet mode, 3.5 s into the precision mode)",
 "mode.fan_level": {
  "nominal": 2,
  "quiet": 1,
  "precision": 3
 },
 "plan.acceleration_limit": {
  "nominal": {
   "linear_m_s2": 0.0175,
   "angular_rad_s2": 0.1745
  },
  "quiet": {
   "linear_m_s2": 0.002,
   "angular_rad_s2": 0.0174
  },
  "precision": {
   "linear_m_s2": 0.01,
   "angular_rad_s2": 0.1745
  }
 },
 "plan.speed_limit.nominal": {
  "linear_m_s": 0.2,
  "angular_rad_s": 0.1745
 },
 "plan.speed_limit.precision": {
  "linear_m_s": 0.02,
  "angular_rad_s": 0.1745
 },
 "plan.speed_limit.quiet": {
  "linear_m_s": 0.02,
  "angular_rad_s": 0.0174
 }
}
```

## The example controller and the example messages

controller/example_controller/controller.py answers every call with a valid reply and nothing else: it refuses every job and commands a zero force and torque. It scores nothing by construction (its opening text says why), and the self-test flies it to the end without a protocol error. controller/example_messages holds one message of each kind, each captured from a run of the toy self-test; their values are the toy session's. The accepted plan among them (reply_plan_accept.json) is a probe's plan, shown for its form only: one setpoint at the start pose of a view that opens a block, every rate zero, which the runner judged valid (its notice is frame_notice_plan_valid.json); the messages:

- `frame_notice_event_ignored.json`: a request whose frame carries the notice of an ignored event (its reason)
- `frame_notice_plan_invalid.json`: a request whose frame carries the notice of an invalid plan (its rule and setpoint index)
- `frame_notice_plan_valid.json`: a request whose frame carries the notice of a valid plan (that of the plan in reply_plan_accept.json)
- `frame_notice_protocol_error.json`: a request whose frame carries the notice of a protocol error
- `reply_plan_accept.json`: a plan reply: accepted with its setpoints (a probe's plan, shown for its form only: one setpoint at the start pose of a view that opens a block, every rate zero; the runner judged it valid: its notice is frame_notice_plan_valid.json)
- `reply_plan_refuse.json`: a plan reply: refused with a reason
- `reply_setup_mass_profile.json`: the setup reply: a mass profile (toy numbers)
- `reply_setup_null.json`: the setup reply: nothing
- `reply_step_command.json`: a step reply: a command with the event none
- `reply_step_command_with_event.json`: a step reply: a command with the event done
- `reply_step_null_command.json`: a step reply: the null command (force and torque both null)
- `request_end.json`: the end request at the session's end
- `request_plan.json`: a plan request (the job and the frame of that time)
- `request_setup.json`: the setup request (its session paper is the toy session's)
- `request_step.json`: a step request (a frame while the controller is in control)
