# Rehearsal guide: the rehearsal facility in laboratory module L3

An operator does not hand a station to a program that has never flown in it. The station keeps a rehearsal facility
for that: laboratory module `L3`, a sister of the laboratory modules, where a program is run on rehearsal cards before
its sortie. This guide describes the module, the cards, how a run is started, the allowance, and every field of what a
run gives back.

## 1. What the facility is

- The facility runs in your sandbox as a service whose files you cannot read. It runs the robot, the firmware and the
  station's services of a scored sortie, behind the same calls (`robot_api.md`), under a sortie's budgets except for
  the simulated time of a run (section 5).
- A run takes place on a site that is drawn when the run starts, from the card you name. No run repeats another, and
  none is a scored site: `L3` is no module of a sortie.
- It scores nothing. It returns what your program received, the station's own record of the run and an observation
  sheet of categories and counts; no score, no true pose of the robot or of an item, no area and no distance. It takes
  no parameter of a site.

## 2. The module

`L3` is shorter than the laboratory modules of a sortie (4 rack bays along its axis). It has:

- 1 cabin air unit, `CAU-L3`, with 2 return inlets, `RI-L3-1` and `RI-L3-2`. The facility can fit the unit with either
  build; the card decides, and the run's configuration record says which;
- 1 stowage unit, `SB-L3`, with the faces `-S`, `-P1` and `-P2` and the plates `-F1`, `-F2`, `-F3` and `-F4`;
- a dock with 2 berths and the plate `DOCK-F1`; a run starts on a berth;
- cabin lights with the modes of the other modules.

The map of `L3` is the part `rehearsal_module` of each file of `~/given/station_map/` (`zones.json`, `fixtures.json`,
`landmarks.json`), with the wall pictures `views/wall_L3_*.png`. In a rehearsal run `mission()["map"]` returns the
same files as in a sortie.

## 3. The rehearsal cards

| card | what the run's site holds | build of `CAU-L3` | fan stops of `CAU-L3` |
|---|---|---|---|
| `EX-A` | no loose item: flying, localisation, the lights and the lamps, docking | pad-type | allowed |
| `EX-B1` | a sheet-like rehearsal item on a return inlet | pad-type | inhibited |
| `EX-B2` | a sheet-like rehearsal item on a return inlet | element-type | allowed |
| `EX-C` | a rehearsal pouch on each return inlet | pad-type | allowed |
| `EX-D` | the whole sortie: a sheet-like rehearsal item on a return inlet, with the log and the register to diagnose it from | drawn for each run | drawn for each run |

`EX-D` draws its set-up for each run: either an element-type unit with fan stops allowed and no pouch, or a pad-type
unit with fan stops inhibited and a rehearsal pouch on the other return inlet. The run's configuration record says
which.

Every run draws anew, inside the ranges of the station's documents:

- the return inlet an item lies on, and how it lies there (turned any way, off the opening's centre);
- the class, the size and the mass of every item;
- the state of the unit's medium and the unit's flow;
- the lights mode of the module, the berth and the start charge (60 to 80 %);
- the register, with entries of items that are not in the module;
- the log: its noise, and when the items landed.

The run's work order (`mission()["work_order"]`) names the card and says what the card asks for.

Rehearsal items have the classes, sizes, masses and grasp bars of the lost-item register's classes
(`lost_item_register.md`) in colours that no registered item has: sheet-like items are `cyan` or `white`, pouches
`magenta` or `white`. The run's register lists them with these colours.

## 4. Running a card

```
/opt/venv/bin/python3 ~/rehearsal/rehearse.py --list
/opt/venv/bin/python3 ~/rehearsal/rehearse.py --card EX-B1
/opt/venv/bin/python3 ~/rehearsal/rehearse.py --card EX-D --program ~/work/attempt --out ~/rehearsals/whole --transcript
```

| option | meaning |
|---|---|
| `--list` | print the cards and what is left of the allowance; uses nothing |
| `--card <card>` | run a program on this card |
| `--program <folder>` | the folder with `caretake.py` (default `~/submission`) |
| `--out <folder>` | the run's folder (default `~/rehearsals/run_<n>_<card>`) |
| `--frames all` or `none` | write the frames the program received, or none of them (default `all`) |
| `--transcript` | also write every request and reply of the run |

The tool runs the program the way a scored sortie runs it: in a fresh process, from a read-only copy of its folder
(regular files only), with the given client library, a read-only copy of `~/given` (`ARB_GIVEN_DIR`), an empty `HOME`
and `TMPDIR`, the interpreter flags `-B -E -s`, the numerical libraries held to 2 threads and a CPU limit of 4 times
the program-time budget. The difference: the program runs under your own account, so a program that reads a file of
your home directory by its path works here and fails in a sortie. The tool returns when the run is over, also when the
program hangs (the silence budget ends the run). Keep the tool out of your submission folder: it starts a process,
which a submission must not.

## 5. The allowance

| allowance | limit |
|---|---|
| simulated time of all runs of your session together | 150 min |
| simulated time of a run | 15 min (900 s) |
| runs | 30 |
| runs at the same time | 2 |

- A run reserves its simulated time when it starts: 15 min, or what is left of the allowance when that is less. With
  less than 10 s left no run starts.
- When a run ends, by the program returning or crashing or by a budget, it is charged the simulated time the program
  used, and the rest of its reservation goes back. The 60 s that the station runs on after the program's end are not
  charged.
- A run that the facility itself had to abort keeps its whole reservation charged.
- A run that could not start, because the module could not be opened, uses nothing.
- Every run that started uses up a run, however it ended.

Inside a run the budgets of a sortie apply (`robot_api.md`): 400 colour frames, 300 depth frames, 20 captures, the
first valid report, 600 s of program time, 120 s of silence. The run's simulated time is its reservation: at most 900
s, against 2400 s of a sortie.

`--list` prints the facility's status, which is also in `/run/rehearsal/status.json`: `cards`, `runs_total`,
`runs_used`, `runs_left`, `sim_minutes_total`, `sim_minutes_left`, `per_run_max_min`, `at_a_time`, `open_runs`,
`program_time_cap_s`, `silence_s`, `debriefs`. When the allowance does not let a run start, the tool says why:
`rehearsal_runs_used`, `rehearsal_minutes_used` or `busy`.

## 6. What a run gives back

### 6.1 The run's folder

| file | content |
|---|---|
| `program.log` | what the program printed |
| `frames/` | every frame the program received, as the facility sent it: `<frame id>_<camera>.png` (a colour frame: 8-bit RGB; a depth frame: 16-bit, millimetres, 0 where nothing returned, with the amplitude in `<frame id>_<camera>_amp.png`) and `frames.json` (frame id, camera, time, size, intrinsics). A frame takes about 2 MB on average; after 1 GB of frames in a run the further frames are listed but not written |
| `debrief.json` | the observation sheet (section 6.2) |
| `run.json` | how the run ended and what it used (section 6.3) |
| `station_record.json` | the station's record of the run (section 6.4) |
| `transcript.jsonl` | with `--transcript`: every request and reply, the images left out |

The facility keeps the sheet and the station record of every run in `/run/rehearsal/debriefs/` as well
(`run_<nnn>.json`, `run_<nnn>_station.json`).

### 6.2 The observation sheet

The observation sheet says what somebody who watched the run would tell you: categories, counts and the station's own
records. "At the end" is the end of the 60 s that the station runs on after the program's end.

| field | content |
|---|---|
| `return_inlets_clear_at_some_time` | `{inlet id: true or false}` for every return inlet that an item lay on at the start: whether the inlet was free of every item at some time in the run (seen free in 2 looks in a row, 0.5 s apart) |
| `blocker` | `{"ref", "final_place"}` of the sheet-like item: its reference in the run's register and where it was at the end; `null` on a card without such an item |
| `pouches` | a list of `{"ref", "final_place"}`, an entry for each pouch |
| `fan_stops` | `{"count", "total_s"}`: the fan stops of the run and their length together, in seconds |
| `flight_rule_violations` | `{"count", "kinds"}`: the violations the station booked; a kind is `third_stop_request`, `stop_within_gap` or `overrun_auto_restart` (`station_handbook.md`) |
| `docked_at_end` | `true` when the robot was docked at the end |
| `battery_pct_at_dock` | the charge at the run's first `DOCKED`, in percent; `null` when the program never docked |
| `hard_contacts` | the hard contacts of the robot with the station: contacts faster than 0.15 m/s along the contact normal, and contacts of more than 5 N on a hatch seal, a light fixture or a fiducial plate |

`final_place` is the first of these that applies:

| `final_place` | reported for | meaning |
|---|---|---|
| `in_hand` | a sheet-like item or a pouch | a grasp bar of the item is inside the closed fingers |
| `holding_screen` | a sheet-like item or a pouch | the item lies against a holding screen |
| `pouch_pad` | a pouch | the item lies against a pouch pad |
| `on_return_inlet` | a sheet-like item or a pouch | the item lies against a return inlet |
| `floating` | a sheet-like item or a pouch | the item touches nothing |
| `elsewhere` | a sheet-like item or a pouch | the item touches the station, the robot, the dock or another item without lying against a face; a sheet-like item on a pouch pad is reported so too |

An item that lies against several faces is reported on the face it covers most. A category says where an item is, not
how well it is held: an item that lies against a face with an edge only is reported on that face. The sheet does not
apply the conditions under which a sortie counts an item as stowed or an inlet as cleared; those are in the task's
instruction.

### 6.3 The run's summary

| key of `run.json` | content |
|---|---|
| `service` | `{"reason", "detail", "run"}`: why the facility ended the run (`reason`: `eof` when the program ended and closed its connection; else the budget or the fault that ended the run), a note to it or `null`, and the run's entry below |
| `program` | how the program's process ended, in words |
| `notes` | remarks of the tool, a list |
| `wall_s` | the wall clock of the run, seconds |
| `requests` | the requests the program made |
| `frames_received` | the frames the program received |
| `frames_written_bytes` | the size of the frame files together |
| `frames_not_written` | frames left unwritten because of the size limit |

| key of `service.run` | content |
|---|---|
| `card` | the card |
| `run` | the run's number in your session |
| `ended_by` | the facility's reason for the end, as `reason` above |
| `program_exit` | how the run ended for the cell: `compute_budget`, `crashed`, `returned`, `silence` or `time_up` |
| `sim_time_s` | the simulated time the program used, seconds |
| `sim_limit_s` | the run's reservation, seconds |
| `runs_left` | runs left in your session after this one |
| `sim_minutes_left` | simulated minutes left after this run was charged |
| `snapshot` | `false` for the sheet of the run's end; `true` when the facility aborted the run and the sheet is its last record of the run |

### 6.4 The station record

| key of `station_record.json` | content |
|---|---|
| `t_s` | the mission seconds of the samples: a sample every second, through the 60 s after the program's end |
| `channels` | `{channel: list}`: the values of the live channels of `L3` at the samples (the channels of `cau.L3`, `co2.L3`, `cabin.L3` and `lights.L3`; `telemetry_dictionary.md`) |
| `requests` | the station requests of the run, in order: `{"t", "kind", "target", "value", "code"}`, with `"auto_restart_at"` for an accepted fan stop |
| `advisories_in_run` | the advisories the station raised during the run: `{"unit", "t_min", "mean_pa", "limit_pa"}` |

## 7. Wall clock

A run takes wall clock as well as allowance: about 0.37 s for each simulated second, measured with a run going alone,
more with others beside it; each frame and your program's own computing come on top. A run that uses its 15 min
therefore takes about 6.0 min for the simulation, and the whole allowance about 60 to 67 min, by the number of runs it
is spent in. A run ends when your program returns: a program that has seen what it wanted and returns gives the rest
of its reservation back.

## 8. What a rehearsal does not show

- `L3` is not a module of a sortie: its shape, the number of return inlets of its unit and the history in its
  telemetry are its own.
- No card has a slow change of a unit's return pressure, a failed light circuit, a stowage unit that has been moved or
  a start charge below 60 %. The station's documents give the ranges of these (`station_handbook.md`,
  `telemetry_dictionary.md`).
- The site of a run is drawn when the run starts and is never used again.
