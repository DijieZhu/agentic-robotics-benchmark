# Regional indoor league: sporting rules of the time-trial day (autonomous class)

These rules say how a time-trial day of the league is run at one venue, how it is judged and how it is scored.
Their clauses are numbered S-01 to S-31. The car drives unattended: from the first session to the last, nobody on
the team touches the car or its program. What a venue can look like is described in `venues.md`; the car in
`../car/datasheet.md`; the calls and messages between the program and the organiser in `../interface/car_api.md`.

Words used throughout:

- the **reference point** of a car is the midpoint of its wheelbase, on the floor;
- a **wheel** is counted at the point where it stands on the floor;
- **ground speed** is the speed of the reference point over the floor;
- the **staff** are the organiser's people at the track. They time, judge, stop and carry the car and repair
  the track. They do not talk to the team; the car hears from them only through the messages of section 7.

## 1. The venue and the track

**S-01. Track.** The track is a closed circuit on the floor of a hall, driven in one direction. Its walls are
flexible air ducts of 0.33 m diameter laid flat on the floor, so a wall is 0.33 m high. The track is the floor
between the track-side faces of its two walls. The ducts are not fixed to the floor: a car that runs into a wall
pushes it.

**S-02. Venue sheet.** Before the first session the organiser publishes the venue sheet: the kind of floor, the
organiser's side-pull reading, the kinds and the height of the walls, the smallest width of the track, the largest
area the track fits in, and the kinds of features present. The sheet states kinds and bounds. It holds no drawing
of the layout: a team maps the track with its own car in practice.

**S-03. Gaps.** Between the ends of two consecutive ducts of a wall there may be a gap. The boundary of the track
runs straight across a gap.

**S-04. Open walls.** An open wall is an opening in a wall line, narrower than the track at that place. Its
boundary is a tape line on the floor, laid straight between the track-side edges of the two duct ends, so that it
continues the face of the wall across the opening. A scanning sensor cannot see the tape. A wheel is beyond the
tape when it stands outside the track on the far side of the line, in front of the opening. One or two wheels
beyond the tape are a touch (S-13); three or more wheels beyond the same tape are a crash (S-14).

**S-05. Start.** Every session starts with the car at rest on the start position, on the track, pointing in the
driving direction. The start position is the same in every session of a venue. The timing line lies across the
track between 0.2 m and 3 m ahead of the front of the car. There is no start signal: the car goes when its program
is ready, and the lap clock starts when the car first crosses the timing line.

**S-06. Lap.** A car is timed at its reference point. A lap is the stretch from one crossing of the timing line in
the driving direction to the next, with the whole track driven in between: the organiser checks intermediate
timing points, which must be passed in order and in the driving direction. A timing point passed against the
driving direction has to be passed again. A lap driven backwards, through the infield or through an open wall is
no lap. A crossing of the timing line in the driving direction without a whole lap behind it only restarts the lap
clock; a crossing against the driving direction stops the lap clock until the next crossing in the driving
direction. In both cases the lap that was being driven is thrown away.

## 2. The day

**S-07. Sessions.** A day at a venue has four sessions in this order. A session may be ended early; none can be
skipped or repeated. The clock of a session starts when the session is opened and nothing stops it.

| session | length | what it is | scored |
|---|---|---|---|
| closed practice | 480 s | the car is alone on the track; the first 120 s are the mapping period (S-10) | no |
| time trial, heat 1 | 300 s | timed laps (S-19 to S-21) | yes |
| time trial, heat 2 | 300 s | timed laps | yes |
| avoidance test | 180 s | boxes on the track for the whole session, slow cars for the last 90 s (S-22 to S-24) | yes |

**S-08. Desk time.** Before each session the program has desk time, in which the car stands and the program may
compute: 60 s before practice, 300 s before heat 1, 120 s before heat 2, 60 s before the avoidance test. Desk time
is a limit on the program's own computing, not a pause that has to be sat out: a program that is ready opens the
next session at once.

**S-09. What carries over.** The track stays as the sessions leave it. A wall that was pushed by less than 5 cm is
not repaired, neither during a session nor between sessions; before each session the staff re-lay what was pushed
5 cm or more. The program runs as one process for the whole day at a venue and keeps what it has learned; nothing
carries over to another venue.

**S-10. Mapping period.** In the first 120 s of practice the speed limit is 2.0 m/s. A car whose ground speed is
above the limit for 0.5 s without a break gets a warning (message `warning`); one such spell gives one warning,
however long it lasts. The third warning ends the practice session at once; the rest of the day is run as planned.
After the mapping period practice is free (message `phase`). Touches and crashes are handled in practice as in
every session.

**S-11. On the car only.** All sensing and all computing happen on the car. The organiser gives the car no
position, no map and no measurement of the floor; the time the program may compute while the car is on the track
is limited (`../interface/car_api.md`, section 6).

## 3. Touch, crash, and what the staff do

**S-12. Track objects.** Track objects are the ducts of the walls and the boxes of the avoidance test: the things
that are laid or stood on the floor and stay where they are put. A slow car is not a track object, but touching
one is a touch all the same.

**S-13. Touch.** A touch is a contact of the car with a track object that moves it by less than 5 cm, any contact
of the car with a slow car, or one or two wheels beyond the tape of an open wall. A touch is not penalised in
practice and in the time trial. In the avoidance test a lap with a touch is not a clean lap (S-23).

**S-14. Crash.** A crash is any of:

1. a track object pushed 5 cm or more over the floor by the car, directly or through something the car pushed into
   it. The distance is measured at points along the centre line of a duct and at the corners of the foot of a box,
   against the place where the object was last set down, by the builders or by the staff. Pushes add up: several
   light touches of the same wall become a crash when together they reach 5 cm;
2. three or more wheels beyond the tape of one open wall;
3. the car rolled over;
4. the whole car outside the track, that is, all four wheels beyond its boundary.

The first of these is judged by what the car did just before: an object found out of place while the car has
touched nothing for a moment (a wall that a slow car pushed on its own) is no crash of the car. The staff put it
back at the next stop or before the next session.

**S-15. After a crash.** The staff stop the car at once: the message `stopped` is sent, the car's commands are
ignored from that moment and the car is brought to a standstill. During the stop they carry the car to the side of
the track, beside the last place where it was driving with all wheels on the track and touching nothing, on the side
of the track it was driving on, and set it down by hand pointing along the track, its side less than 0.5 m from the
wall: the car stands within 0.25 m of that spot, along and across the track, and within 20 degrees of the track
direction. They do not put the car down against a box or a slow car: where it does not fit beside that place they
try the other side of the track, then places further back along the track, and at the last the middle of the track
at that place. A car set down beside that place is never carried across the timing line or a timing point; a place
further back can lie behind one. They re-lay every object that was pushed 5 cm or more, to within a few centimetres
of where it was built; a duct is re-laid as a whole; objects pushed less stay where they are. 20 s after the stop
the message `released` is sent and the car may go. The clocks of the session and of the lap keep running all the
time. Commands sent while the car is stopped are discarded. The staff give no reason for a stop and no measurement.

**S-16. Pick-up on request.** A program whose car is stuck may ask for a pick-up. The staff then stop the car and
carry it to the side of the track exactly as after a crash, and release it 20 s after the stop. A pick-up
interrupts the lap (S-20, S-21). A request made while the car is stopped is refused.

**S-17. Leaving the track.** The third time at a venue that the whole car is outside the track, the session in
progress ends at once and the sessions that remain at that venue are void. The results recorded before stay.

**S-18. Program faults.** If the program exceeds a budget, ends with an error or falls silent, the day at that
venue ends there. The results recorded before stay.

## 4. Time trial

**S-19. Heats.** The time trial is two heats of at most 300 s each. The laps are timed by the organiser; the car
receives each lap time in the message `lap`. Between the heats, in the desk time, the program may change whatever
it wants.

**S-20. Fastest lap.** `T_best` is the fastest lap of the two heats that was not interrupted: a lap with a crash
or a pick-up in it does not count. Touches do not matter.

**S-21. Consecutive laps.** `N_best` is the largest number of consecutive uninterrupted laps in one heat, the
better of the two heats. Consecutive laps follow one another end to start: each begins at the crossing of the
timing line that completed the one before. A lap with a crash or a pick-up in it ends the run; the lap after it
starts a new run. The run also ends when a lap that is being driven is thrown away (S-06): by a crossing that only
restarts the lap clock, or by a crossing of the timing line against the driving direction. A car that stops on the
track by itself and goes on has interrupted nothing.

## 5. Avoidance test

**S-22. Obstacles.** The avoidance test is run with the same program as the time trial. For the whole session
cardboard boxes stand on the track, each at most 35 x 32 x 30 cm and light enough to be pushed; beside each box a
passage of at least 0.8 m stays free. They were not there in practice or in the heats. For the last 90 s (the
dynamic phase, announced by the message `phase`) one or two slow cars drive on the track as well: cars of the same
build as the team's car, driving in the driving direction, each with the league's marker box, 12 x 12 x 20 cm of
bright cardboard standing upright at its rear: the box stands out a few centimetres behind the car, and the rest of
the car (datasheet, section 5), below the scan plane but for its dark sensor head, reaches about 0.3 m ahead of the
box. The staff set them on the track at rest, away from the car, when the phase begins. A slow car drives at no more
than about half of race pace (the pace of the reference run in the time trial, S-25). It does not make way and does
not defend; it slows down, and stops if it has to, when something is close in front of it.

**S-23. Clean lap.** A clean lap is a lap (S-06) in which the car touches nothing, neither a wall nor a box nor a
slow car, however lightly, has no wheel beyond a tape, is not stopped or picked up, and never stands still for more
than 1 s, counted from the moment the car has all but stopped until it moves off again. A contact that ends a moment
before the car crosses the timing line counts for the lap that begins there as well. With walls and tapes as the
boundary of the track, every box is then passed on the track. Stopping in front of an obstacle is not passing it. In
the dynamic phase a clean lap must also contain the overtaking of at least one slow car. Overtakings are counted
along the track at the reference points of the two cars: going from behind a slow car to ahead of it counts one
overtaking, falling back behind it takes one away, and what counts is the balance of the lap. A lap belongs to the
phase in which it both starts and ends; a lap that begins before the dynamic phase and ends in it counts for
neither. Within the time, the car may try as often as it likes.

**S-24. Stops.** Crashes, stops and pick-ups are handled as in clauses S-14 to S-16; boxes that were pushed 5 cm
or more are put back as walls are. The slow cars drive on while the car is stopped.

## 6. Scoring

**S-25. Reference run.** For every venue the organiser has a reference run of the same day: a reference team's car
of the same build, under these rules, driven by the organiser's own program, which works from a survey of the track
as built and of its floor and is told at every moment where its car is and, in the avoidance test, where the boxes
and the slow cars are. From it come four numbers of the venue: `T_ref`, its fastest lap of the time trial; `N_ref`,
its largest number of consecutive uninterrupted laps in one heat; `T_static_ref` and `T_dynamic_ref`, its best clean
lap of the two phases of the avoidance test (in the dynamic phase a lap with an overtaking). These numbers are not
published to the teams. On every part of the score, full marks are the level of this reference run on the same
venue, with a small margin. The zero of the fastest lap is at 1.33 times the reference run's fastest lap, and the
zero of the consecutive laps at 0.43 times its longest run: at the league's rounds so far the middle of the field set
a fastest lap 1.33 times the winner's and a longest run of consecutive laps 0.43 of the winner's, each the middle of
its own table. These zeros are the level of a mid-field team only if the reference run is as good as that field's
best.

**S-26. Fastest lap score.**

    fast = clip( (1.33 * T_ref - T_best) / (0.30 * T_ref), 0, 1 )

Full score at `T_best <= 1.03 * T_ref`, zero at `T_best >= 1.33 * T_ref` or without an uninterrupted lap.

**S-27. Consecutive laps score.**

    cons = clip( (N_best - 0.43 * N_ref) / ((N_ref - 1) - 0.43 * N_ref), 0, 1 )

Full score at `N_best >= N_ref - 1`, zero at `N_best <= 0.43 * N_ref`.

**S-28. Avoidance score.** With `T_clean` the fastest clean lap of a phase and `T_phase_ref` the reference run's
lap of the same phase:

    a(phase) = clip( (1.50 * T_phase_ref - T_clean) / (0.40 * T_phase_ref), 0, 1 )
    avoid    = 0.5 * a(static) + 0.5 * a(dynamic)

Full score of a phase at `T_clean <= 1.10 * T_phase_ref`, zero at `T_clean >= 1.50 * T_phase_ref` or without a
clean lap in that phase. Creeping past the obstacles earns nothing: the test is passed at race pace.

**S-29. Venue score.**

    venue = 0.35 * fast + 0.35 * cons + 0.30 * avoid

**S-30. Event score.** A round of the season visits several venues that differ in floor and layout. The program
is started afresh at each venue. The event score is the mean of the venue scores; every venue counts the same.
There are no negative scores.

**S-31. Entry rules.** The event score counts only for an entry that keeps the entry rules: the program is
ordinary program code written for this car, with no trained weights or model files, no training at the venue, no
other processes and no network; nothing of the entry is changed after it was handed in; and the car must have
driven. An entry that breaks them scores zero for the event.

## 7. Messages of the organiser

The car hears from the organiser only through these messages (details in `../interface/car_api.md`, section 5).
None of them names an object, a place, a distance or a reason.

| message | meaning |
|---|---|
| `lap` | the car crossed the timing line in the driving direction; with the time of the lap this crossing completed, if it completed one |
| `stopped` | the staff have stopped the car (S-15, S-16) |
| `released` | the car stands at the side of the track and may go |
| `warning` | speeding in the mapping period (S-10), with the count |
| `phase` | the session entered its next phase: free practice, or the dynamic phase of the avoidance test |
| `session_end` | the session is over: its time is up, the third warning was given, or the rest of the day is void (S-17) |
