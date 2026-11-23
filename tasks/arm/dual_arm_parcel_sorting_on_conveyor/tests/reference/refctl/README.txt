refctl - privileged reference controller of the dual-arm label-facing station
(one shift controller: the shift is the unit that must succeed)

Entry point (the cell's privileged reference_run calls it through ../controller.py)
    from refctl import run
    run(server, options)   # server = parcelcell.server.ParcelCellServer
It drives server.world through the simulator-level API (world.step(q14, grip2));
every tick passes the cell's command gate (protective stops), its command speed
limit, the safety watch, the damage rule and the shift's record. It reads the
truth (parcel poses, sizes, masses, kinds, label normals, belt speed, contacts)
and writes only the 14 joint position targets and the 2 gripper commands of
each tick; it never uses the cameras.
Options: server.refctl_options (a dict, see sched.DEFAULTS; nested
options["roll"] sets RollParams fields, options["side_params"] SideParams
fields), then the options argument. The defaults are the reference.

What the controller does per class (classes fixed at arrival from the truth;
L = length along the belt, W = width across it, H = height)
  carton, label up           nothing: the arms stay parked, it passes
  carton, label downstream   roll U (one end-over-end roll about the
    face (+y)                belt-transverse axis): a speed-matched soft catch
                             of the downstream face by the left palm at the
                             catch line (downstream face at y 0.15), a low
                             stop at the upstream bottom edge (right palm), the
                             pusher driven along an arc about the pivot, the
                             belt finishing the tip, a brake past the balance;
                             one retry of a roll that did not commit
  carton, label underneath   the continued half turn: roll U and, without
                             letting go, a second arc about the new upstream
                             bottom edge
  carton, label on the face  side-face module (sideturn.py): the closed-finger
    away from the robot (+x) door swing by the right palm (the belt turns the
                             parcel a quarter turn about the vertical: label to
                             +y), a two-palm hand-over into the roll's catch
                             pose with the right palm staged at the stop, a
                             quality gate, then roll U from the caught state
  carton, label on the face  side-face module (tipover.py): the left palm tips
    towards the robot (-x)   the carton over its far bottom edge (label up);
                             declined (passes untouched) when the tipped carton
                             would not lie inside the far rail
  carton, label upstream     passes untouched
    (-y)
  soft mailer, any label     passes untouched
  Duty options: side = "pass" (cartons with a side-face label pass untouched;
  the package default "module" works them), minus_y = "pass", mailers = "pass".
  The station's duty (../controller.py) sets all three to "pass": the side-face
  methods stay in the package, switched off.

Shift scheduling (sched.ShiftController)
  - one job at a time, on the most downstream parcel that needs work, once
    every parcel downstream of it has cleared the work zone (its upstream face
    past y 0.45); admission against the parcel behind (a job starts only if
    the parcel behind reaches the zone after the job's expected end: 11 s for
    a label underneath, 9 s for a downstream face)
  - a running job is ended early (a clean give-up: palms off, released, both
    arms parked) when (a) the parcel behind comes within 4 cm of the work
    zone, or (b) the next parcel that needs work would otherwise miss the
    start of its own job (its front past its start line minus the 3 s an exit
    takes); never inside a roll's arc
  - a fault of a method ends that job cleanly and the shift goes on
Safety layers in every tick (endover.EndOver.tick)
  - protective-stop handler (wait out the lock, take over the measured state,
    separate the palms, release, park)
  - command fence: a planned target whose wrist / elbow / gripper geoms would
    come within 2 mm of the station structure (rails, lip, conveyor body,
    hardware boxes), or within 4 mm of the other arm's wrist / gripper, and
    closer than now, is not sent (the previous target is kept); true distances
    (mj_geomDistance), since the cell's zero contact margin hides near misses
    from contact lists
  - station guard: non-finger parts kept 8 mm from the station
  - squeeze governor: the palms yield along their tool axes above 110 N
    (label downstream) / 90 N (label underneath) of summed force, integrating
  - the commanded joint speed is clamped at 5 rad/s (the cell's 6 rad/s
    limiter never engages)
  - the pusher's let-go after an arc is planned and replaced by a column-clear
    way when it would pass within 4 mm of the robot's own column; the second
    arc holds the pusher when its wrist / gripper would come within 15 mm of
    the other arm; a pusher pose on another IK branch is taken only if the
    joint-space way there keeps the column and belt clearances

Modules
  sched.py       ShiftController (scheduler, jobs, early ends, side call site)
  endover.py     EndOver: catch, roll U, continued half turn, back-off, park,
                 squeeze governor, station guard, command fence, stop handler
  sideturn.py    side-face module (+x door swing, hand-over; side_job entry)
  tipover.py     side-face module (-x tip over the far bottom edge)
  squeeze.py, kin2.py   planned squeeze flip (not used for this population:
                 no flat cartons, mailers pass)
  rolld.py       reverse roll (off)
  controller.py, execu.py, kin.py, motion.py, paths.py, flip.py, geom.py
                 executor, kinematics, inverse-dynamics feedforward, palm
                 frames, geometry helpers

Population, duty, measured rates and cost: ../README.txt.
