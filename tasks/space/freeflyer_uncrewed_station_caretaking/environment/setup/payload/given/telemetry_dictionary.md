# Telemetry dictionary: channels, log, configuration record, advisories

The station monitors the cabin air of its modules and a few other quantities and keeps them as telemetry. This
dictionary says what each channel measures, how often and how well, how the log is laid out, what the configuration
record holds, how the station raises an advisory, and which characteristics of the channels are known. The cabin air
system itself is described in `station_handbook.md`; the calls that return the telemetry are in `robot_api.md`.

## 1. Where the telemetry is

| what | where | content |
|---|---|---|
| the log | `mission()["log"]` | the 72 h before the sortie, a row every 1 min |
| live telemetry | `station_telemetry()` | the latest value of every channel, refreshed at 1 Hz |
| configuration record | `mission()["config_record"]` | for each cabin air unit its build, clean reference pressure, advisory limit and whether its fan may be stopped; for each module the commanded lights mode |
| advisory log | `mission()["advisory_log"]`; live: the channel `advisory_log` | the advisories the station has raised |
| a sample | `~/given/telemetry_sample/` | a log of `L3` from a crewed period, with its record (section 7) |

`<module>` in a channel name is `L2`, `H2` or `L1`. The channels of a cabin air unit carry the module's id:
`cau.L2.return_dp_a` belongs to `CAU-L2`.

## 2. The log

`mission()["log"]` is a dict of columns: `t_min` and a list for each logged channel, 4320 rows each.

- `t_min` counts the minutes of the log, from 0 to 4319. A row holds the mean of the minute that begins at its
  `t_min`.
- The sortie begins a row period after the last row: mission second 0 is minute 4320 on the log's axis. `report` takes
  the onset time in minutes on this axis.
- Logged channels, 7 for each module: `cau.<module>.return_dp_a`, `cau.<module>.return_dp_b`,
  `cau.<module>.fan_speed`, `cau.<module>.fan_current`, `co2.<module>`, `cabin.<module>.temp`,
  `cabin.<module>.dew_point`.
- Not in the log: `cau.<module>.state`, `cau.<module>.stops_used`, `lights.<module>`, the hatch surveys
  `imd.*.last_survey` and the channel `advisory_log`. They exist live only; the advisories of the logged time are in
  `mission()["advisory_log"]`.

## 3. The channels

`station_telemetry()` returns `{"t", "channels"}`: the mission second of the latest sample and the latest value of
every channel. The live channels continue the log: they are the same sensors, with the same offsets.

| channel | unit | in the log | live | step of the value | accuracy | what it is |
|---|---|---|---|---|---|---|
| `cau.<module>.return_dp_a`, `cau.<module>.return_dp_b` | Pa | mean of the minute, every 1 min | 1 Hz | 0.1 | noise of a value 1 Pa (one sigma); each sensor has an offset of its own of up to 0.5 Pa and wanders slowly, by about 0.5 Pa (one sigma) over the log | the unit's return pressure, measured by its sensors a and b |
| `cau.<module>.fan_speed` | % of design | every 1 min | 1 Hz | 0.1 | 1 % | the speed of the unit's fan |
| `cau.<module>.fan_current` | A | every 1 min | 1 Hz | 0.01 | 1 % of the value | the current of the supply channel that feeds the fan; the channel's other loads are in it |
| `cau.<module>.state` | - | no | at a change | - | - | `running` or `stopped` |
| `cau.<module>.stops_used` | - | no | at a change | - | - | the stops of this unit used in the sortie |
| `co2.<module>` | mmHg | the analyser's last value | the analyser's last value | 0.01 | 5 % of the value | partial pressure of CO2. The station has 1 analyser, which samples the modules in turn: a module gets a new value every 20 min |
| `cabin.<module>.temp`, `cabin.<module>.dew_point` | degC | every 1 min | a new value every 1 min | 0.1 | 0.5 degC | cabin temperature and dew point |
| `lights.<module>` | - | no | at a change | - | - | the commanded lights mode, `general` or `night` |
| `imd.L2-H2.last_survey`, `imd.H2-L1.last_survey` | m/s, days | no | fixed | - | - | `{"speed_m_s", "age_days"}`: the air speed through the open hatch between the named modules at the last manual survey with an anemometer, and the age of that survey |
| `advisory_log` | - | no | at a change | - | - | the list of advisories, the logged ones and those raised since (section 5) |

## 4. The configuration record

`mission()["config_record"]["units"]["CAU-<module>"]` holds, for each cabin air unit:

| key | content |
|---|---|
| `build` | `"pad-type"` or `"element-type"`: the build of the unit's return inlets (`station_handbook.md`) |
| `clean_reference_pa` | the clean reference pressure: the unit's return pressure at its last service, with the medium as it was then and nothing on its inlets, entered to 0.1 Pa |
| `advisory_limit_pa` | the advisory limit: the clean reference pressure × 1.40 for a pad-type unit, × 1.35 for an element-type unit |
| `fan_stop` | `"allowed"` or `"inhibited"`: whether the station takes stop requests for the unit's fan |
| `inhibit_reason` | the reason of an inhibit, else `None` |

`mission()["config_record"]["lights"]` gives the commanded lights mode of each module at the start of the sortie.

## 5. Advisories

The station's health monitor watches the return pressure of each cabin air unit. For every row it takes the mean of
the unit's 2 sensors and averages it over the last 10 rows (10 min). When this mean rises above the unit's advisory
limit, the monitor raises an advisory:

    {"unit": "CAU-<module>", "t_min": <the last row of the averaged rows>, "mean_pa": <the mean>, "limit_pa": <the limit>}

It raises the next advisory for the same unit only after the mean has fallen more than 1 Pa below the limit and has
risen above the limit again. The monitor keeps watching during the sortie, on the minutes of the live channels; an
advisory raised then appears in the channel `advisory_log`, with a `t_min` that continues the log's axis (4320 and
up).

An advisory says that a unit's mean return pressure is above the unit's limit. It says neither why nor since when: a
mean over 10 min follows a step late, and a slow rise crosses the limit long after it began.

## 6. Known characteristics

### 6.1 Items on return inlets

An item that lands on a return inlet covers part of the opening, and the unit's return pressure steps up
(`station_handbook.md`). Both sensors of the unit show the step, from a row to the next or, when the landing falls
inside a row's minute, over 2 row intervals; the level stays up for as long as the item lies there. Steps of the items
of the lost-item register:

| item on an inlet | build of the unit | unit with 3 return inlets | unit with 2 return inlets |
|---|---|---|---|
| a sheet-like item | `pad-type` | 6 to 13 Pa | 10 to 18 Pa |
| a sheet-like item | `element-type` | 32 to 53 Pa | 32 to 65 Pa |
| a pouch | `pad-type` | 0.5 to 3.2 Pa | 0.8 to 5 Pa |

Pouches lie on inlets of pad-type units only, so the table has no row for a pouch on an element-type unit. The step of
a pouch is of the size of the noise of a single value (1 Pa). A pouch does not raise an advisory by itself, and the
moment it landed is one more step in the log of its unit.

### 6.2 Slow changes

- **Filter loading.** The medium of a unit loads with dust, and the unit's return pressure creeps up on both sensors
  alike: rises of 6 to 12 Pa within 60 h occur.
- **Sensor drift.** A unit's sensor can drift by as much (6 to 12 Pa within 60 h) while the unit's other sensor stays
  where it was.

Either raises the mean that the health monitor watches (section 5), and a slow rise can carry that mean over the
unit's advisory limit.

### 6.3 What an advisory tells

The monitor compares a mean with a limit and knows nothing else. An advisory can stand on a unit whose return inlets
are all free (section 6.2), and a unit with an item on a return inlet can be without an advisory: whether the step
carries the mean over the limit depends on the step and on how far below the limit the unit was.

### 6.4 The fan channels and a fan stop

- `fan_speed` reads the design speed for as long as the fan runs: the unit's speed controller holds it, whatever the
  state of the medium and of the inlets.
- `fan_current` follows the air power that the fan delivers, on top of the channel's other loads: a change of the
  unit's flow shows in it, within the accuracy of the channel.
- During a fan stop (`station_handbook.md`) the live return pressure of the unit falls toward zero and `fan_speed`
  falls with it, with a time constant of about 3 s; `state` reads `stopped` and `stops_used` has counted the stop.
  After the restart they return the same way.

### 6.5 The other channels

- `co2.<module>`: no source of CO2 is on board while the station is uncrewed.
- `imd.*.last_survey`: the surveys are 60 to 110 days old.
- `lights.<module>`: the commanded mode. A module whose light circuit has failed reports its commanded mode like any
  other (`station_handbook.md`).

### 6.6 Crewed periods

These occur only while a crew is on board; the sample has each of them (section 7):

- **A change of the filter medium**: both sensors of the unit step down.
- **A temporary occlusion**: something covers an inlet for a while and is taken away again; both sensors step up and
  later down.

While the station is uncrewed nobody services a unit, and nobody takes an item off an inlet.

## 7. The sample

`~/given/telemetry_sample/l3_72h.csv` is a log of the rehearsal module `L3` from a crewed period: 4320 rows, a row
every 1 min, as text: a header row with the column names, then the rows. Its columns are `t_min` and the 7 logged
channels of `L3`. `CAU-L3` has 2 return inlets.

`~/given/telemetry_sample/l3_72h_record.json` holds what belongs to the sample: the configuration record and the
advisory log in the form `mission()` returns them, and the events the crew logged:

| what | when (`t_min` of the sample) |
|---|---|
| configuration record: `CAU-L3`, `pad-type`, clean reference pressure 40.9 Pa, advisory limit 57.3 Pa | the whole sample |
| a change of the filter medium | 589 |
| a temporary occlusion of `RI-L3-1` by a crew item | from 3511 to 3744 |
| an advisory on `CAU-L3`: mean 57.52 Pa over the limit of 57.3 Pa | 3519 |

The occluding item was a crew item, not an item of the lost-item register: the step table of section 6.1 is for the
register's items and does not bind this occlusion.
