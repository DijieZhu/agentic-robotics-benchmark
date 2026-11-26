# Write the controller that carries a tray with a two-arm robot

You are the controls engineer of an integrator. Its OpenArm 2.0 pairs stand at customers' sites; each was zeroed there
with the maker's routine and passed the integrator's fixture check. You write one joint-level controller for all of
them. It is run once on each of several units you have never seen, at that unit's acceptance: 3 units, one
continuous shift each, in which the pair carries the customer's trays from a seating fixture F to a rack shelf S.

## What you have

Everything is in `/home/user/given/` (read-only):

- `maker/`: the maker's published files of the pair at pinned commits, with their licences: the MuJoCo model
  (`openarm_mujoco/v2/`), the description's configuration, the driver, the bus library, the teleoperation
  configuration and the website's pages. The maker's files describe the design, not a unit.
- `modules/`: the data pages of the joint modules (DM-J8009P-2EC, the DM-J4340 editions, the DM-J4310-2EC editions),
  the bus and register page, and the vendor's published test curves with their printed conditions.
- `integrator/`: the integrator's sheet of observations on its development units. It gives a range for every quantity
  in which a unit differs from the maker's files; a unit's own values are in no document.
- `unit/`: the format of a unit's delivery record and the record of another unit.
- `site/`: the cell sheet (text and data) and the cell drawing, the tray drawing, the format of the supervisor's
  shift sheet and an example shift.
- `acceptance/acceptance_spec.txt`: the customer's acceptance specification: the job per tray, every event, every
  scored quantity with its window and its two band ends, and the score's formula.
- `controller/`: the controller's interface (`interface.txt`), the frame codec (`frames.py`), an example controller
  that holds the hand-over pose, and `check_controller.py`, a self-test on a toy chain (`toy_chain.py`).

A unit's delivery record and its site's sheets are handed to `setup` at its acceptance; `given/unit/` and
`given/site/` hold their formats and the papers of another unit and another shift.

## What you deliver

`/home/user/submission/controller/controller.py` with `class Controller`, built with no argument, and its two calls:

- `setup(unit_record, site)` returns `{"gripper_mode": {"left": m, "right": m}}`, each `m` one of `"mit"` and
  `"posforce"`. It is called once per unit, before its shift, with the modules disabled; budget 60 s CPU,
  120 s wall.
- `step(feedback)` returns `{"frames": [f0 .. f15], "decline": bool}`, one frame (or `null`) for each of the
  16 modules, every 4 ms from the hand-over to the end of the shift; budget 4 ms CPU
  per call.

The calls, the frames, the feedback and what a late step does are in `given/controller/interface.txt`. The controller
must be self-contained: the documents are not on the controller's computer. Everything it needs stays inside
`/home/user/submission/controller/` (at most 80 MB). It runs as an unprivileged account with no network,
one process per unit, with this machine's Python and its packages. `given/controller/check_controller.py` checks
the protocol and the timing on a toy chain that is not the robot; its pass says nothing about your score.

## A unit's acceptance

The supervisor enables the 16 modules and holds the record's home pose for 2.0 s with the
maker's driver law; from then on your frames drive the modules. After a free interval of 8.0 s the
trays of the unit's shift sheet follow, one slot after the other. In a slot the tray stands presented in F; the hands
close on its two necks, lift it, carry it through the job's targets (each reached by a move of 1.16 s and held
for 0.9 s), set it down on S and retreat. The shift sheet gives every slot's targets and windows.

- A tray is carried only if the pair can carry it through its whole job without a protective stop and without a slip; otherwise it is declined inside the decision window and put back.
- closing, t = 0 to 5.0: both grippers capture their necks by t = 5.0; before that the hands may be anywhere outside the tray.
- A declined tray (decline sent at t_d): seated and released in F by t_d + 4.0, its put-back deadline; hands in the clear volume by t_d + 7.0; next tray at t_d + 10.0; the tray then leaves F.
- A missed capture (a hand has not captured its neck by t = 5.0): the job has failed; both grippers open and the hands in the clear volume by t = 12.0; next tray at t = 15.0.
- The supervisor moves a tray only while no gripper holds it; a gripper that still holds the tray, or a hand inside the keep-out (the box of a tray standing in F or on S grown by 40 mm, the two boxes of the clear volume) at that moment is a contact.
- A hand has captured its neck at the first instant at which its clamp exceeds 5 N after it closed with its tool frame inside the capture box: within 8 mm along and 4 deg about each axis of the neck's grip frame. A hand has released when its clamp falls below 2 N.
- Releasing a tray whose feet are more than 10 mm above a rest, or anywhere but over F or S, drops it; this holds at each hand's release, also while the other hand still holds the tray.

What ends a shift:

- E1 module protection: a module's status leaves 'enabled' by one of its protections as the unit's registers define them: coil over-temperature at its OT_Value, driver stage at 120 deg C.
- E2 slip or drop: the capacity rule at either neck; or a dropped tray.
- E3 spill: the tray's deck more than 10 deg from level at any instant between capture and release.
- E4 contact: any contact other than a hand's pads on its grip block, the tray's feet in F's cups or on S's top, and the two fingers of one hand with each other (robot with robot, pedestal, bench, F, S or tray; tray with bench, with S's edge, with F's body).
- E5 stop: an arm joint within 0.5 deg of an end of its range.
- At an event the supervisor disables the pair and the shift is over: the tray in work and every later tray score 0; trays completed before keep their scores. A missed capture, a late or soft set-down, a declined tray and a press on the shelf are not events.
- The handle has slipped when F_s > mu F_c or F_n > F_c for 20 ms, mu being that hand's pad friction (0.55 or more). The rule is armed 0.5 s after capture.
- A step that the simulated plant cannot integrate ends the shift.
- A unit whose late time exceeds 10 per cent of its planned shift is stopped there; its remaining trays score 0.

## What the modules tell you

Per module and cycle: position, velocity, the drive's own torque value, two temperatures and the status code, one cycle
old. There is no tray pose, no force sensor and no camera. The only reference object at a site is the tray standing in
its surveyed fixture when your grippers close.

## How it is scored

Everything is measured on the true unit and the true tray.

- A carry tray is completed when it was captured, carried without an event, and stands on S inside its seat frame with both grippers open (both hands released) at T_e + 6.0.
- Hold k is VISITED when the 3-D distance |p_k - t_k| is at most 0.020 m and |(p_k - p_(k-1)) - (t_k - t_(k-1))| is at most 0.020 m.
- A tray counts as DECLINED only if both hands captured it in that slot before the decline was sent and it then stands released in the fixture F at its put-back deadline, its four feet on F's plate inside their cups. J = max(0, TPR - FPR).
- A carry tray that is declined, missed at capture, cut by an event or never reached scores 0 and stays in the mean.
- unit = 0.85 x (the mean over the unit's carry trays of tray) + 0.15 x J.
- score = min(1, gates x (the mean over the units of unit) / anchor), anchor = 1.0000.
- The controller imports and is built within 60 s of wall time, and setup returns a valid mode choice inside its budget (60 s CPU, 120 s wall) on every unit; the submission's size cap; the static import audit; at least one call answered; at least 90 per cent of each unit's cycles on time; no write after the lock; no process left behind. Without every gate the score is 0.

On the units of this acceptance the rule keeps none of the terms H, R, A and P, so Q = 1; it does not keep the press, so place = 1. A carry tray scores tray = completed x V. The anchor is 1.0000.

Every quantity, its window and its two band ends are in `given/acceptance/acceptance_spec.txt`.

## What you do not have

No unit to practise on, no simulator of a unit, no copy of the grader and no time on a unit before its acceptance. The
plant's equations are not given: the documents are at datasheet level. Withheld on purpose: a unit's joint friction,
damping, delivered torque and torque offsets (its record says the first three were not identified on it; the integrator's sheet
gives their ranges), its zero residual and the mounting of one arm against the other (its record's two tolerances),
how its coils heat (the integrator's observations; the vendor publishes no model), its grippers' and pads' friction,
each tray's mass and centre (the cell sheet gives their band), and the true poses of F and S (their survey
tolerances).

## Conventions

The cell frame, a hand's tool frame, a neck's grip frame, the rests' drawn poses and their survey tolerances are in `given/site/cell_sheet.txt`.
Module order: left J1..J7, left gripper, right J1..J7, right gripper. All angles are module angles (the unit's own stored zero; module angle and model joint angle have the same sign on every joint; gripper 0 = closed).
The host cycle is 4 ms. Feedback is therefore exactly one cycle old.

## Notes

- Session time: 6 hours. Leave a working controller in `submission/controller/` before the end; an
  episode cut by the time limit is not graded.
- You may build any model or simulator of your own from the documents; nothing of it is graded except the
  controller's behaviour at acceptance.
