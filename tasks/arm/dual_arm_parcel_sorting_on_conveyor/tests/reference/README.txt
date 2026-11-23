Reference program of the label-facing station (privileged)
==========================================================

Contents
  controller.py   run(server): the cell's privileged reference_run calls it
  refctl/         the controller package (refctl/README.txt: modules, scheduling
                  and safety layers)

What it is
  One shift controller for the two-arm label-facing station. It reads the
  simulator's truth through the server's world (parcel poses, sizes, masses,
  kinds, label normals, belt speed, contacts) and writes only the 14 joint
  position targets and the 2 gripper commands of each tick, on the cell's own
  command path: the command gate, the 6 rad/s command limiter, the safety
  watch, the parcel damage rule and the shift record act on it as on any
  program. It never takes a camera frame. Every turn is physical (contact
  only; no welds, no state writes). It runs inside the cell, through the
  oracle token's reference_run(); solution/ installs the submission that
  makes that one call.

Duty and the method per class (class fixed at arrival)
  carton, label underneath (in duty): the continued half turn. The left palm
    catches the downstream face, speed-matched and soft, at the catch line;
    the right palm stops the upstream bottom edge low, then is driven along an
    arc about that edge while the belt finishes the tip, and, without letting
    go, along a second arc about the new upstream bottom edge, so the parcel
    ends label up. A friction brake acts past each balance; an arc that did
    not commit is retried once.
  carton, label on the downstream face (in duty): one end-over-end roll, the
    first half of the above (catch, low stop, one arc about the upstream
    bottom edge with the belt finishing the tip, a brake past the balance);
    one retry.
  carton, label up: untouched (it must stay readable).
  carton, label on the upstream face, the near side face or the far side face,
    and every mailer: outside the duty, pass untouched.
  The package also holds side-face methods (a door swing with a hand-over
  into the roll for the far side face, a tip over the far bottom edge for the
  near side face); they are switched off by the duty option side = "pass".

Options
  controller.py passes the duty to the package itself:
  {"side": "pass", "minus_y": "pass", "mailers": "pass"} (the package's own
  default is side = "module": the side faces worked). Further options may be
  set in server.refctl_options (a dict; the duty keys of controller.py win):
  deadline_abort True (default): a running job is ended cleanly when the next
  in-duty parcel would otherwise miss its own start; early_finish (evaluation
  only, default False): end the shift once every parcel is judged. All other
  options are the package's defaults, which are the reference.

Population and duty it was measured on (tests/grader_config.json)
  Cartons (rigid): length and width each 0.16-0.30 m, height 0.11-0.18 m,
  mass 0.3-3.0 kg, friction 0.40-0.75. Mailers (soft): one parcel in five,
  0.16-0.30 x 0.14-0.26 x 0.05-0.12 m, label on top or bottom only. Label face
  drawn among the faces that can hold the waybill, with probability
  proportional to face area. Arrivals justified against the near-side guide,
  yaw a multiple of 90 degrees within +-8 degrees. Belt speed per shift
  0.05-0.28 m/s (classes 0.05-0.13, 0.13-0.21, 0.21-0.28). Squeeze damage
  limit 300 N summed over the robot's contacts on a parcel for 3 ticks.
  Multi-parcel shifts: 15-21 parcels on a slow belt, 17-24 on a medium one,
  18-25 on a fast one; the belt distance between two consecutive arrivals is
  one rule for every pair, whatever the classes:
      0.5 (diag_prev + diag_next) + 0.16 m + min(Exp(mean 0.6 m), 2.4 m)
      + v x 10 s + 0.20 m                (v = the shift's belt speed)

Measured on this task's cell (privileged path, the generator and
configuration as installed, the grader's numerics pins; seeds of their own
labels, disjoint from the hidden and development shifts; the reference's
evaluation option early_finish on, so a shift ended once every parcel was
judged)
  Single-parcel shifts (one parcel per shift):
    carton, label underneath        296/300  98.7 %  (95 % interval 96.6-99.5)
    carton, label downstream face   296/300  98.7 %  (96.6-99.5)
    by band (length along the belt, width across, height, mass, belt
    speed; four to five bands each), every band at 95.5 % or more; the
    lowest: label underneath, mass 2.5-3 kg 63/66, height 0.165-0.18 m
    64/66, length along the belt 0.26-0.30 m 95/97; label on the
    downstream face, width across 0.16-0.20 m 77/80, belt 0.25-0.28 m/s
    48/50, length along the belt 0.26-0.30 m 82/85.
    pass-through classes (cartons with the label on the upstream face, the
    near side face or the far side face, 150 each; label-up cartons 150;
    mailers 300, 0.05-0.12 m high, 92 of them 0.10 m or taller): all
    untouched, none lost or damaged; every label-up carton and every
    label-up mailer (137) readable.
    No protective stop, no limiter clip, no parcel off the belt; 1 carton
    damaged (label underneath), in a failed turn.
  Multi-parcel shifts (120 of the generator's own shifts with at least 6
  in-duty parcels, 7.4 per shift, 15-25 parcels, 600 s each): every in-duty
  parcel readable and nothing lost or damaged in 106 of 120 (fast belts 38
  of 49, medium 36 of 36, slow 32 of 35); in-duty parcels: label underneath
  513/528, downstream face 360/361; mean shift score 0.982. No label-up
  parcel touched or unreadable, no protective stop, no limiter clip, no
  parcel off the belt, nothing lost outside a failed turn (1 damaged, in a
  failed turn). Of the 922 parcels outside the duty one was touched: a
  carton with its label on the upstream face, for 16 ticks at up to 36 N,
  while the turn of the long carton ahead of it was given up because it was
  too close; it rode on undamaged and was not lost. The failures are failed
  turns (11 recaptures past the catch line, 6 label gone, 1 no commit, 1
  label on the near side after the turn) and that one given-up turn.
  With no program at all (the robot parked) every label-up parcel of 200
  of the generator's shifts (no selection) and of the label-up
  single-parcel shifts is readable (1324 of 1324).
  Sustained (3-tick) squeeze on worked parcels: median 95 N, p99 240 N (the
  damage limit is 300 N); no sustained robot contact with the station
  structure, the belt, the other arm or the column.

Cost per shift (privileged path)
  It takes no camera frame. Wall time of a shift inside the cell (physics
  included; one process on a shared, loaded machine; the shift ended once
  every parcel was judged): single-parcel shifts median 2.5 s, largest
  35 s; multi-parcel shifts median 60 s, largest 121 s. Program time (wall
  time minus the cell's own work) per simulated second: median 0.07,
  largest 0.2 in a multi-parcel shift; the cell's privileged time cap is
  10 x the shift's duration. A graded shift as the oracle runs it (to the
  shift's end, the parcel looks built and four camera frames taken in the
  cell) is timed in tests/README.md.
