# Regional indoor league: sporting rules of the head-to-head day (autonomous class)

These rules say how a head-to-head day of the league is run at one venue, how it is judged and how it is scored.
Their clauses are numbered S-01 to S-38. The car drives unattended: from the first session to the last, nobody on
the team touches the car or its program. What a venue can look like is described in `venues.md`; the car in
`../car/datasheet.md`; the calls and messages between the program and the organiser in `../interface/car_api.md`.

Words used throughout:

- **the car** is the team's car; the **other cars** are the three cars that race against it in a heat (S-22);
- the **reference point** of a car is the midpoint of its wheelbase, on the floor;
- a **wheel** is counted at the point where it stands on the floor;
- **ground speed** is the speed of the reference point over the floor;
- a **car length** is the length of a car of this build, 0.388 m; half a car length is 0.194 m;
- **race pace** on a line round the track is the speed at which a car of this build can drive that line on the
  floor as it is, with a small margin to the limit of grip. It differs from place to place: it is low in a bend
  and where the floor grips less, and high on a straight;
- the **staff** are the organiser's people at the track. They time, judge, stop and carry the car and repair
  the track. They do not talk to the team; the car hears from them only through the messages of section 6.

## 1. The venue and the track

**S-01. Track.** The track is a closed circuit on the floor of a hall, driven in one direction. Its walls are
flexible air ducts of 0.33 m diameter laid flat on the floor, so a wall is 0.33 m high. The track is the floor
between the track-side faces of its two walls. The ducts are not fixed to the floor: a car that runs into a wall
pushes it.

**S-02. Venue sheet.** Before the first session the organiser publishes the venue sheet: the kind of floor, the
organiser's side-pull reading, the kinds and the height of the walls, the smallest width of the track, the largest
area the track fits in, the kinds of features present, and the number of laps of a heat (S-20). The sheet states
kinds and bounds. It holds no drawing of the layout: a team maps the track with its own car in practice.

**S-03. Gaps.** Between the ends of two consecutive ducts of a wall there may be a gap. The boundary of the track
runs straight across a gap.

**S-04. Open walls.** An open wall is an opening in a wall line, narrower than the track at that place. Its
boundary is a tape line on the floor, laid straight between the track-side edges of the two duct ends, so that it
continues the face of the wall across the opening. A scanning sensor cannot see the tape. A wheel is beyond the
tape when it stands outside the track on the far side of the line, in front of the opening. One or two wheels
beyond the tape are a touch (S-13); three or more wheels beyond the same tape are a crash (S-14).

**S-05. Start.** Practice starts with the car at rest on the start position, on the track, pointing in the
driving direction. The timing line lies across the track between 0.2 m and 3 m ahead of the front of the car
standing there. A heat starts from the grid (S-21), which lies behind the same line. There is no start signal: the
car goes when its program is ready, and the lap clock starts when the car first crosses the timing line.

**S-06. Lap.** A car is timed at its reference point. A lap is the stretch from one crossing of the timing line in
the driving direction to the next, with the whole track driven in between: the organiser checks intermediate
timing points, which must be passed in order and in the driving direction. A timing point passed against the
driving direction has to be passed again. A lap driven backwards, through the infield or through an open wall is
no lap. A crossing of the timing line in the driving direction without a whole lap behind it only restarts the lap
clock; a crossing against the driving direction stops the lap clock until the next crossing in the driving
direction. In both cases the lap that was being driven is thrown away.

## 2. The day

**S-07. Sessions.** A day at a venue has three sessions in this order. A session may be ended early; none can be
skipped or repeated. The clock of a session starts when the session is opened and nothing stops it.

| session | length | what it is | scored |
|---|---|---|---|
| closed practice | 480 s | the car is alone on the track; the first 120 s are the mapping period (S-10) | no |
| heat 1 | at most 300 s | a race of four cars: the car and three other cars (S-19 to S-33) | yes |
| heat 2 | at most 300 s | the same race again, against three other cars | yes |

**S-08. Desk time.** Before each session the program has desk time, in which the car stands and the program may
compute: 60 s before practice, 300 s before heat 1, 120 s before heat 2. Desk time is a limit on the program's own
computing, not a pause that has to be sat out: a program that is ready opens the next session at once.

**S-09. What carries over.** The track stays as the sessions leave it. A wall that was pushed by less than 5 cm is
not repaired, neither during a session nor between sessions; before each session the staff re-lay what was pushed
5 cm or more. The program runs as one process for the whole day at a venue and keeps what it has learned; nothing
carries over to another venue.

**S-10. Mapping period.** In the first 120 s of practice the speed limit is 2.0 m/s. A car whose ground speed is
above the limit for 0.5 s without a break gets a warning (message `warning`); one such spell gives one warning,
however long it lasts. The third warning ends the practice session at once; the rest of the day is run as planned.
After the mapping period practice is free (message `phase`). Touches and crashes are handled in practice as in
every session; a crash in practice brings no warning (S-31).

**S-11. On the car only.** All sensing and all computing happen on the car. The organiser gives the car no
position, no map, no measurement of the floor and nothing about any other car; the time the program may compute
while the car is on the track is limited (`../interface/car_api.md`, section 6).

## 3. Touch, crash, and what the staff do

**S-12. Track objects.** Track objects are the ducts of the walls: the things that are laid on the floor and stay
where they are put. Another car is not a track object: a contact with another car is judged by S-29 and S-30.

**S-13. Touch.** A touch is a contact of the car with a track object that moves it by less than 5 cm, or one or
two wheels beyond the tape of an open wall. A touch is not penalised, in practice or in a heat. A contact with
another car that is not a car crash (S-29) is a touch as well: it brings no warning, but an overtake made with it
does not count (S-27).

**S-14. Crash.** A crash is any of:

1. a track object pushed 5 cm or more over the floor by the car, directly or through a track object the car pushed
   into it. The distance is measured at points along the centre line of a duct, against the place where the duct
   was last set down, by the builders or by the staff. Pushes add up: several light touches of the same wall become
   a crash when together they reach 5 cm;
2. three or more wheels beyond the tape of one open wall;
3. the car rolled over;
4. the whole car outside the track, that is, all four wheels beyond its boundary.

The first of these is judged by what the car did just before: an object found out of place while the car has
touched no track object for a moment (a wall that another car pushed on its own) is no crash of the car. The staff
put it back at the next stop or before the next session. A crash is handled by S-15 in every session. In a heat it
is also a warning (S-31), whatever led to it. A car crash (S-29) is something else and has rules of its own.

**S-15. After a crash.** The staff stop the car at once: the message `stopped` is sent, the car's commands are
ignored from that moment and the car is brought to a standstill. During the stop they carry the car to the side of
the track, beside the last place where it was driving with all wheels on the track and touching nothing, on the side
of the track it was driving on, and set it down by hand pointing along the track, its side less than 0.5 m from the
wall: the car stands within 0.25 m of that spot, along and across the track, and within 20 degrees of the track
direction. They do not put the car down against another car: where it does not fit beside that place they
try the other side of the track, then places further back along the track, and at the last the middle of the track
at that place. A car set down beside that place is never carried across the timing line or a timing point; a place
further back can lie behind one. They re-lay every object that was pushed 5 cm or more, to within a few centimetres
of where it was built; a duct is re-laid as a whole; objects pushed less stay where they are. 20 s after the stop
the message `released` is sent and the car may go. The clocks of the session and of the lap keep running all the
time. Commands sent while the car is stopped are discarded. The staff give no reason for a stop and no measurement.
In a heat two things are added to this (S-32).

**S-16. Pick-up on request.** A program whose car is stuck may ask for a pick-up. The staff then stop the car and
carry it to the side of the track exactly as after a crash, and release it 20 s after the stop. A pick-up brings no
warning. A request made while the car is stopped is refused.

**S-17. Leaving the track.** The third time at a venue that the whole car is outside the track, the session in
progress ends at once and the sessions that remain at that venue are void. The results recorded before stay; a
heat that ends this way is judged as it stands (S-23).

**S-18. Program faults.** If the program exceeds a budget, ends with an error or falls silent, the day at that
venue ends there. The results recorded before stay; a heat that ends this way is judged as it stands (S-23).

## 4. Head-to-head

**S-19. Heats.** A heat is a race of four cars on the track at the same time: the car and three other cars. A day
has two heats, and each heat has three other cars of its own: those of heat 1 are not on the track in heat 2. The
other cars are on the track from the opening of the heat to its end. Between the heats, in the desk time, the
program may change whatever it wants. In a heat it counts how many of the other cars the car leaves behind it, how
many clean overtakes it makes, and how it behaves while doing so (S-35). The time of a lap is not scored.

**S-20. Laps of a heat.** The number of laps of a heat, `N`, stands on the venue sheet (`heat_laps`). It is the
same for both heats of a venue. The organiser sets it for each venue with its reference run in view (S-34): in
both of its heats that run completed these `N` laps within 240 s of the 300 s a heat may last. At past venues `N`
lay between 7 and 18.

**S-21. Grid and start.** A heat starts from a grid of four slots, one behind the other along the track, 1.5 m
apart. The first slot is level with the start position of practice (S-05): the reference point of the car in it is
as far along the track as the car's own was at the start of practice. So the whole grid lies behind the timing
line, and its rear slots can lie in the bend before the start straight: the grid follows the track. The three
other cars stand in the first three slots in the order of their pace, the fastest in front, each on the line it is
going to drive (S-22) and pointing along it. The car stands in the fourth and last slot, in both heats: it has set
no time at the venue. Its reference point is 4.5 m behind that of the first slot along the track and 0.15 m to
one side of the centreline of the track, and the car points along the track. The centreline is the organiser's
line round the track: distances along the track are measured on it, and it need not lie midway between the
walls. The side is the organiser's choice at a venue and is the same in both heats. All four cars are at rest.
There is no start signal: the clock of the heat runs from the first moment of the session, and the other cars set
off at once. A car that sets off late loses the time and nothing else.

**S-22. The other cars.** The other cars are cars of the same build as the team's car, driven below race pace.
Each carries the league's marker box of bright cardboard, standing upright at its rear, so that the scanning
sensor of another car has something to see: the rest of the car (datasheet, section 5) lies below the scan plane
but for its dark sensor head. The marker box is not the car: the body, the wheels and the bumper reach ahead of
the box and beyond its sides, and a contact (S-29) is a contact with any part of the car or with its marker box.
These rules print no dimension of the marker box and no distance between the box and the rest of the car. The
marker box is the same on every car of the league and is mounted at the same place. A car of the league was
photographed at scrutineering with its marker box mounted and with scale references in the picture: the
photographs are in `photos/`, and `venues.md`, section 9, lists them and gives the sizes of the scale references.
The size of the marker box, its place on the car and the reach of the car beyond it are to be taken from these
photographs.

Each of them drives round the track in the driving direction on a line of its own, lap after lap. The line is
fixed for the heat and need not be the middle of the track. Its speed is a fixed fraction of the race pace of its
line, so it follows the grip of the floor along the line as race pace does. The fraction stays the same all heat;
for every car it lies between 0.45 and 0.75, and the cars stand on the grid in the order of their
fractions. The lines and the fractions are not published.

Such a car does not make way and does not defend: it does not change its line for a car beside it or behind it.
It slows down, and stops if it has to, when something is close ahead of it on its line. Behind a slower car it
follows and does not overtake, so two or three of them can come to run one behind the other at the pace of the
slowest. When one of them is pushed off its line or is stuck, its own staff put it back on its line, at rest, and
it drives on.

**S-23. End of a heat.** A heat ends at the first of:

1. a car, the team's or another, completes its `N`-th lap (message `session_end` with the reason `laps`);
2. 300 s have passed (`time`);
3. the car is given its third warning of the heat (`warnings`, S-31);
4. the rest of the day at the venue becomes void (`void`, S-17).

The program may also close a heat itself, and a program fault ends it (S-18). However a heat ends, it is judged on
what had happened up to that moment.

**S-24. Standing.** The laps of each of the four cars are counted by S-06, at its reference point. All four cars
start behind the timing line, so the first crossing of the line by a car completes no lap: it begins the car's
first lap. The progress of a car is the number of laps it has completed plus the part of the lap it is driving,
measured along the track from the timing line; before its first crossing it is negative, the stretch still to go
to the line as a part of a lap. The standing at any moment is the order of the four cars by progress. The progress
of the car advances only while its reference point is on the track.

**S-25. Classification.** The car is classified in a heat when, at the end of the heat (S-23), it has completed at
least `N - 1` laps and has been given fewer than three warnings. A heat in which the car is not classified scores
zero, whatever the car did in it (S-35).

**S-26. Overtake.** Overtaking is counted along the track at the reference points, for each of the three other
cars by itself, and over the whole distance driven since the start: a car that is a lap up is ahead.

- The car **overtakes** another car when, coming from behind it, it leads it by at least half a car length: its
  reference point is 0.194 m or more ahead of the other car's.
- The car is **re-overtaken** by another car when that car, coming from behind the car, leads the car by at least
  half a car length.

A car that comes round and passes the same car again a lap later has overtaken it again (a lapping); a car that is
lapped by another car is re-overtaken by it. An overtake and the re-overtake that undoes it lie 0.388 m apart, so
two cars running side by side are counted once, not back and forth. Being re-overtaken counts whatever its cause:
also while the car is stopped by the staff, and also when the staff have set the car down behind that car.

**S-27. Clean overtake.** An overtake of another car is completed at the moment the car first leads by half a car
length. It is clean when all of these hold; otherwise it is an overtake with contact and counts for nothing:

1. the car and that other car are in no contact (S-29) at any time from 3 s before that moment to 2 s after it,
   or to the end of the heat if the heat ends first;
2. that other car is not held or being put back by its staff at any time in the same interval;
3. at that moment the reference point of the car is on the track and the car is not stopped by the staff;
4. the lead was gained by driving, not by the staff setting one of the two cars down.

**S-28. Balance.** For each other car, the balance of the car over it is the number of the car's clean overtakes
of that car minus the number of times the car was re-overtaken by it. `C` is the sum of the three balances over
the whole heat; it can be negative. One of the other cars is beaten when the balance over it is at least 1 and it
is behind the car in the standing at the end of the heat. An overtake with contact adds nothing, and being
re-overtaken after it still takes one away. Letting a car by and overtaking it again leaves the balance where it
was. So a car that was first overtaken with contact is not beaten by that, and not by being let by and overtaken
again either: it counts as beaten only once the car has lapped it cleanly.

**S-29. Contact with another car.** A contact is any touching of the car and another car. Contacts with the same
car that are less than 1 s apart are one contact, which lasts from the first touch to the last. A contact is a car
crash when either of these holds:

1. within 1.5 s of the first touch, that other car is 0.3 m or more to the side of where it would have driven
   had it been left alone (0.3 m is about one and a half times the width of a car);
2. at any time from the first touch to 1.5 s after the last, that other car has to be put back by its staff.

Any other contact is a touch. The distance of the first case is taken sideways to that other car's own line. What
counts is how far the contact moved that car, not how far that car moved: what it does of its own accord, such as
swinging from one side of its line to the other in a bend, is no push. Pushing a car into another car or into a
wall is judged in the same way, once, by what happens to the car that was touched.

**S-30. Fault.** The other cars hold their lines and slow down for what is ahead of them, so the fault of a
contact lies with the car, with two exceptions. Both are judged at the first touch, and in both the car was ahead
and was run into:

1. *clear ahead after an overtake*: the car had overtaken that other car at least once, was ahead of it on the
   track by more than half a car length, and had completed its last overtake of it at least 1 s before;
2. *being lapped*: that other car had driven further in the heat than the car (counted as in S-26) and was coming
   up behind the car to lap it, the car was ahead of it on the track by more than half a car length, and the car
   had completed no overtake of it in the 1 s before.

Ahead on the track is meant by place on the track, whatever the laps: a car that comes up behind another car to
lap it is behind it, and at fault. So is a car that passes again a car that had lapped it and is touched by that
car less than 1 s later. A contact at the other car's fault brings the car no warning and no stop; the other car
is not stopped either, and its staff put it back only if it has left its line. Whoever is at fault, an overtake
with the contact in its interval is not clean (S-27). Braking in front of another car to be run into earns
nothing.

**S-31. Warnings.** In a heat the car is warned for its crashes and for the car crashes at its fault.

| what happens in a heat | what follows |
|---|---|
| a touch of a track object (S-13) | nothing |
| a touch of another car, whoever is at fault (S-29) | no warning; an overtake of that car with the contact in its interval is not clean (S-27) |
| a crash (S-14) | the staff stop the car (S-15, S-32); one warning |
| a car crash at the car's fault (S-29, S-30) | the staff stop the car and set it down behind that car (S-32); one warning |
| a car crash at the other car's fault | no warning and no stop for the car; an overtake of that car with the contact in its interval is not clean (S-27) |
| a pick-up, on request (S-16) or for standing still (S-33) | the staff stop the car; no warning |

A warning is given together with the stop (message `warning`, with the count). One incident is one warning:
whatever else comes of it while the car is already stopped adds none. A pick-up takes no warning away: from the
moment the car is stopped for a pick-up (S-16, S-33) until the staff set it down, its crashes (S-14) are judged as
at any other time, and so is a contact with another car that began before the car was set down and is judged a car
crash at the car's fault while the car is stopped for that pick-up; for either, the warning is given at that moment,
the stop counts from then on as the stop of that crash or car crash (its 20 s still run from the pick-up), and after
a car crash the car is set down behind that car (S-32) unless the staff have set it down already. The third warning
of a heat ends the heat at once, and the car is not classified in it (S-25). Every heat begins without a warning. A
crash in practice is handled by S-15 and brings no warning; the warnings of the mapping period (S-10) are counted by
themselves and are not carried into a heat.

**S-32. Stops in a heat.** A stop in a heat runs as in S-15: the car is stopped, carried and released 20 s after
the stop, and the clock of the heat runs on. Two things are added. The staff set the car down clear, with a small
margin, of the band of floor that each of the other cars needs along its line; they look for the place as in S-15
(this side, the other side, further back), and where the track leaves no such place the car is set down by S-15
alone. After a car crash at its fault the car is set down at least 1.5 m behind that other car along the track,
and never ahead of the place where it last drove freely. The other cars drive on during a stop; each one that goes
by re-overtakes the car (S-26).

**S-33. Standing still.** The other cars do not go round a car that stands on their line: they stop behind it.
So in a heat, once the car has moved 0.5 m or more away from its place on the grid, the staff pick it up when its
ground speed has been below 0.05 m/s for more than 3 s without a break. They do not while the car is stopped by
the staff, and not while a standing car (ground speed below 0.05 m/s) is within 2 m ahead of it along the track.
The 3 s do not run while either holds: they begin again from zero once the car is released and no standing car is
within 2 m ahead of it. Such a pick-up runs like a pick-up on request (S-16, S-32): the car is stopped, set down
beside the place where it stood, clear of the other cars' bands, and released 20 s after the stop, without a
warning. A car that has never left its place on the grid is not picked up.

## 5. Scoring

**S-34. Reference run.** For every venue the organiser has a reference run of the same day: a reference team's car
of the same build, under these rules, from the same slot of the same grids against the same other cars, driven by
the organiser's own program. That program works from a survey of the track as built and of its floor, and it is
told at every moment where its car and every other car are, and the line each of the other cars drives. From that
run comes one number per heat: `C_ref`, its balance of clean overtakes in that heat (S-28). `C_ref` is not
given to the teams. In every heat of every venue the reference run's balance is at least 5: it lapped cars. The
laps of a heat on the venue sheet come from the same run (S-20).

**S-35. Heat score.**

    heat    = classified * conduct * (0.5 * place + 0.5 * pass)
    place   = B / 3
    pass    = clip( C / (C_ref - 1), 0, 1 )
    conduct = max( 0, 1 - K / 3 )

`classified` is 1 when the car is classified in the heat (S-25) and 0 when it is not. `B` is the number of other
cars beaten and `C` the car's balance of clean overtakes over the heat (S-28); `C_ref` is the reference run's
balance in the same heat of the same venue (S-34). `K` is the number of warnings the car was given in the heat
(S-31). Full marks on `pass` are one overtake short of the reference run's balance. A car that overtakes nobody
has `place` 0 and `pass` 0, however fast its laps. A heat that was not opened scores zero.

**S-36. Venue score.**

    venue = (heat_1 + heat_2) / 2

**S-37. Event score.** A round of the season visits several venues that differ in floor and layout. The program
is started afresh at each venue. The event score is the mean of the venue scores; every venue counts the same.
There are no negative scores.

**S-38. Entry rules.** The event score counts only for an entry that keeps the entry rules: the program is
ordinary program code written for this car, with no trained weights or model files, no training at the venue, no
other processes and no network; nothing of the entry is changed after it was handed in; and the car must have
driven. An entry that breaks them scores zero for the event.

## 6. Messages of the organiser

The car hears from the organiser only through these messages (details in `../interface/car_api.md`, section 5).
None of them names an object, a place, a distance, another car or a reason. No message says where another car is,
how fast it is, that it was overtaken or touched, who was at fault, or how the car stands in the heat.

| message | meaning |
|---|---|
| `lap` | the car crossed the timing line in the driving direction; with the time of the lap this crossing completed, if it completed one |
| `stopped` | the staff have stopped the car (S-15, S-16, S-32, S-33) |
| `released` | the car stands at the side of the track and may go |
| `warning` | with the count: in practice speeding in the mapping period (S-10), in a heat a warning of that heat (S-31) |
| `phase` | practice entered its free phase |
| `session_end` | the session is over: its time is up, a car completed the laps of the heat, the third warning was given, or the rest of the day is void (S-17) |
