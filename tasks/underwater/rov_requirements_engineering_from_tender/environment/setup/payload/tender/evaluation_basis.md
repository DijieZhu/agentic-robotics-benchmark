# The evaluation basis

`tender/evaluation_basis.py` is the calculation procedure of tender specification TS-01 rev C. Part A of TS-01
states the site duties, the twelve needs `N-01` to `N-12` and the Owner's policy; Part C (evaluation) and Part D
(factory acceptance) say how the needs are judged; this module is the exact, executable form of those calculations.
Whoever evaluates the bids uses this same file. This page describes it. Where the page and the code differ, the code
counts.

## 1. What it holds, and what it does not

It holds the formulas of the twelve needs, the policy constants of Part A (`POLICY`), the closed dictionary of
facility quantities of Part D (`FACILITY_QUANTITIES`) and the density of pure water.

It holds no site data, no page value of the catalogue and no fact about any unit. Every condition of the site, every
page value and every fact about a unit is an argument, and nothing has a default. Without the conditions of the site
and the facts of a unit it computes nothing.

```python
import sys
sys.path.insert(0, "/home/user/tender")
import evaluation_basis as eb

margins = eb.rows(vehicle, site)            # {"N-01": ..., ..., "N-12": ...}
values = eb.site_values(vehicle, site)      # the eight `site` values of evaluation.json
detail = eb.row_details(vehicle, site)      # intermediate values of N-03 and N-04
q = eb.facility_value(quantity, conditions, vehicle)   # one facility quantity of Part D
```

`vehicle` and `site` are records (section 3) that you fill in.

## 2. Conventions

- **Units are in the names**: `_kg`, `_l` (litre), `_n` (newton), `_m`, `_mm`, `_deg`, `_c` (degC), `_v`, `_a`, `_w`,
  `_wh`, `_ohm`, `_mohm`, `_bar`, `_min`, `_mps` (m/s), `_m3s` (m3/s), `_pct`, `_frac` (0 to 1), `_msw` (metres of
  sea water), `_kgf` (as thruster pages print it), `_rpm`, `_nm` (N m), `_kg_m3` (kg/m3).
- **A margin is signed**: zero or more meets the need, less than zero does not.
- **No defaults.** Every key of a record and every argument must be given. A missing key raises `KeyError`, an
  unknown value of an enumeration raises `ValueError`. Extra keys of a record are ignored.
- **The same result on every machine.** The elementary operations are IEEE double arithmetic; `sin`, `cos`, `tan`,
  `asin` and powers are computed to 56 significant digits and rounded once, so each is the correctly rounded value
  whatever the machine and its maths library; tables are read with the written-out linear interpolation `interp`.
  The module reads no file, no network and no clock.
- **The policy constants are read from `POLICY`** when a function is called.

## 3. Records

A record is a plain `dict`. The four key lists are `THRUSTER_RECORD_KEYS`, `PACK_RECORD_KEYS`,
`VEHICLE_RECORD_KEYS` and `SITE_RECORD_KEYS`.

**Thruster page**: the full-throttle table by bus voltage, as a thruster datasheet prints it.

| key | what it is |
|---|---|
| `supply_v` | list of the bus-voltage columns [V], increasing |
| `thrust_kgf` | full-throttle forward thrust of one thruster in each column [kgf] |
| `current_a` | full-throttle current of one thruster in each column [A] |
| `speed_rpm` | full-throttle shaft speed in each column [rpm] |
| `reverse_ratio` | reverse thrust over forward thrust at the same throttle [1], one constant for the page |
| `thrust_power_exp` | exponent of the part-throttle law: thrust = full thrust x (current / full current) ** `thrust_power_exp` [1] |
| `prop_pitch_m` | geometric pitch of the propeller [m] |
| `power_at_1kgf_w` | electrical power of one thruster that produces 1 kgf in the 12 V column [W]; the power at a thrust of t kgf is `power_at_1kgf_w` x t ** (1 / `thrust_power_exp`) |

**Battery pack page**: one pack of one cell lot, with its temperature tables on the printed nodes.

| key | what it is |
|---|---|
| `capacity_ah` | nominal capacity [Ah] |
| `capacity_factor` | delivered capacity of the lot over the nominal capacity, at the reference temperature [1] |
| `retention_temp_c` | temperature nodes of the capacity-retention table [degC], increasing |
| `retention_frac` | capacity retention at those nodes [1] |
| `r_ss_23c_ohm` | steady-state DC resistance of the pack, after 60 s at load, at 23 degC [Ohm] |
| `r_factor_temp_c` | temperature nodes of the resistance-factor table [degC], increasing |
| `r_factor` | steady-state resistance at the node over the resistance at 23 degC [1] |
| `continuous_current_a` | continuous current rating of the pack [A] |
| `ocv_soc_frac` | state-of-charge nodes of the open-circuit-voltage table [1], increasing |
| `ocv_v` | open-circuit voltage of the pack at those nodes, at `ocv_ref_temp_c` [V] |
| `ocv_ref_temp_c` | temperature of the open-circuit-voltage table [degC] |
| `ocv_drift_v_per_c` | drop of the open-circuit voltage of the pack for each degC below `ocv_ref_temp_c` [V/degC] |

**Vehicle**: one unit in one state (as tendered, as delivered, or as it was on a test day).

| key | what it is |
|---|---|
| `thruster` | thruster page record of the four horizontal thrusters |
| `front_thruster_angle_deg` | angle between the axis of a front horizontal thruster and the x axis of the vehicle [deg] |
| `rear_thruster_angle_deg` | angle between the axis of a rear horizontal thruster and the x axis of the vehicle [deg] |
| `thrust_system_coeff` | system coefficient of the axis thrust of this unit in this state [1] |
| `surge_drag_lin_n_per_mps` | surge drag without tether, linear coefficient [N s/m] |
| `surge_drag_quad_n_per_mps2` | surge drag without tether, quadratic coefficient [N s2/m2] |
| `pack` | battery pack page record of the pack fitted |
| `current_preset_a` | factory preset of the battery current, a limit in the firmware [A] |
| `hotel_load_w` | hotel electrical load with the lights at 100 % [W] |
| `lights_load_w` | the part of `hotel_load_w` that the lights draw at 100 % [W] |
| `camera_port` | `"flat"` or `"dome"` |
| `camera_hfov_air_deg` | horizontal field of view of camera and lens in air [deg] |
| `pressure_part_ratings_msw` | rated depth of every pressure part as built [msw] |
| `air_mass_kg` | mass in air [kg] |
| `displaced_volume_l` | displaced volume [L] |
| `tank_water_density_kg_m3` | density of the water of the trim tank in which the unit was weighed [kg/m3] |
| `ballast_piece_mass_kg` | mass in air of one ballast piece [kg] |
| `ballast_piece_volume_l` | displaced volume of one ballast piece [L] |
| `removable_ballast_pieces` | removable ballast pieces fitted; 0 when the ballast cannot be removed |
| `spare_ballast_pieces` | loose spare ballast pieces supplied with the unit |
| `bg_m` | vertical distance from the centre of buoyancy down to the centre of gravity [m] |
| `static_pitch_trim_deg` | static pitch trim, nose down positive [deg] |
| `active_pitch_moment_nm` | pitch moment the thrusters can add, at the 12 V column [N m] |
| `jaw_reach_m` | forward distance from the centre of gravity to the tip of the gripper jaw [m] |
| `tether_length_m` | tether length [m] |
| `tether_diameter_mm` | tether diameter [mm] |
| `wet_part_min_temps_c` | lowest operating water temperature of every wet part [degC] |
| `link_test_length_m` | tether length over which the data link is tested [m] |
| `sonar_min_range_m` | minimum range of the imaging sonar [m] |
| `dvl_min_altitude_m` | lowest altitude at which the velocity log keeps bottom lock [m] |
| `gripper_opening_mm` | opening of the gripper jaw [mm] |
| `depth_sensor_range_bar` | full-scale pressure of the depth sensor [bar] |

**Site**: the conditions of the inspection, each as TS-01 Part A defines it.

| key | what it is |
|---|---|
| `water_temp_c` | bottom water temperature of the inspection-month cast at the intake station [degC]; not rounded; the density and both battery needs use it |
| `working_depth_m` | design water level minus the lowest point of the inspection scope [m] |
| `launch_path_m` | tether length from the zero of the payout counter, along the payout route, to the lowest point of the inspection scope [m] |
| `face_reach_m` | reach along the dam face on one side [m] |
| `design_flow_m3s` | design discharge, rounded up to 0.5 m3/s (`policy_design_flow_m3s`) [m3/s] |
| `rack_width_m` | width of the trash rack [m] |
| `rack_height_m` | height of the trash rack that is scanned [m] |
| `open_height_m` | height of the rack that is open to the flow [m] |
| `secchi_m` | Secchi statistic, rounded down to 0.1 m (`policy_secchi_m`) [m] |
| `min_water_temp_c` | lowest water temperature of the policy, rounded down to 0.1 degC (`policy_min_water_temp_c`) [degC] |
| `crane_swl_kg` | safe working load of the Owner's lifting appliance [kg] |
| `rigging_kg` | mass of the lifting gear [kg] |
| `survey_altitude_m` | survey altitude above the bottom [m] |
| `sample_max_mm` | largest sample to be retrieved [mm] |

## 4. The policy constants

`POLICY` holds the constants of TS-01 Part A under these keys. Part A states each of them; the values are in the
code.

| key | what it is |
|---|---|
| `depth_factor` | rated depth must be at least `depth_factor` x working depth (N-02) |
| `msw_rho` | density behind the unit msw of depth ratings [kg/m3] |
| `thrust_factor` | available thrust at the rack must be at least `thrust_factor` x drag (N-03) |
| `reserve` | state of charge left at the end of the dive (N-03, N-04) |
| `v_min` | lowest bus voltage [V] (N-03, N-04) |
| `slack` | tether length must be at least `slack` x path (N-05) |
| `standoff_per_secchi` | inspection standoff = `standoff_per_secchi` x Secchi statistic |
| `overlap` | overlap of neighbouring video strips |
| `buoy_lo`, `buoy_hi` | lower and upper edge of the net-buoyancy window in the site water, as fractions of the air mass (N-01) |
| `pitch_lim_deg` | pitch from horizontal with the sample held [deg] (N-06) |
| `sample_kg` | weight of the sample in water, expressed in kg (N-06) |
| `cd_tether` | drag coefficient of the tether in service |
| `contingency_s` | contingency time at the rack [s] |
| `scan_mps` | scan speed along a strip [m/s] |
| `strip_turn_s` | time for each change of strip [s] |
| `transit_mps` | transit speed, vertical and along the dam face [m/s] |
| `transit_vert_extra_w` | extra electrical power in vertical transit [W] |
| `transit_face_extra_w` | extra electrical power in transit along the dam face [W] |
| `rack_extra_w` | extra electrical load at the rack [W] |
| `eta_dist` | efficiency of the power distribution, applied to the usable energy of the pack |
| `n_flat_port` | refractive index used for a flat camera port |

## 5. Site quantities

- **Density.** `rho(t_c, p_dbar)` is the density of pure water [kg/m3] at the temperature `t_c` [degC] and the
  pressure `p_dbar` [dbar]: the 1980 international equation of state of seawater at zero salinity, with the
  temperature converted to the 1968 scale. There is no salinity term, and conductivity records are not used.
  `site_density(water_temp_c, working_depth_m)` is `rho` at the working depth, with the pressure taken as 1 dbar for
  each metre of depth.
- **Flow speed in front of the rack.** `rack_flow_speed_mps(design_flow_m3s, rack_width_m, open_height_m)` =
  design discharge / (rack width x open height).
- **Standoff.** `standoff_m(secchi_m)` = `standoff_per_secchi` x Secchi statistic.
- **The three roundings of Part A.** `policy_design_flow_m3s(max_plant_flow_m3s)` rounds the statistic of Part A up
  to 0.5 m3/s, `policy_secchi_m(lower_quartile_m)` rounds down to 0.1 m, `policy_min_water_temp_c(lowest_c)` rounds
  down to 0.1 degC. Each is an exact decimal rounding of the value first rounded to 0.000001. The battery needs use
  the bottom water temperature `water_temp_c`, which is not rounded.

## 6. The twelve needs

`rows(vehicle, site)` returns the margin of each need for one unit on one site record. The functions behind it can
be called one by one.

**N-01, trim on site.** `n01_net_buoyancy_kg(air_mass_kg, displaced_volume_l, water_density_kg_m3)` = density x
volume / 1000 - air mass, in kg, positive when the unit floats. The window is [`buoy_lo`, `buoy_hi`] x air mass, in
the site water. The margin is the distance to the nearer edge of the window, positive inside. N-01 has two named
rules:

- `n01_trim_reachable(vehicle, water_density_kg_m3)` is the rule of Part C. It takes the best margin over the
  ballast states that the unit's own kit reaches: j removable pieces taken out, for j from 0 to
  `removable_ballast_pieces`, or j spare pieces added, for j from 1 to `spare_ballast_pieces`. Taking a piece out
  lowers the air mass by `ballast_piece_mass_kg` and raises the net buoyancy by `ballast_piece_mass_kg` -
  `ballast_piece_volume_l` x density / 1000; adding a piece does the opposite. The window of a state is taken on
  the air mass of that state.
- `n01_as_delivered(air_mass_kg, displaced_volume_l, water_density_kg_m3)` is the rule of Part D. It takes the unit
  as it is ballasted at the works: nothing is trimmed on site and spare pieces do not count.

`rows` uses `n01_trim_reachable`.

**N-02, depth rating.** `n02_depth_margin_m(pressure_part_ratings_msw, working_depth_m, water_density_kg_m3)` = the
lowest rating among the pressure parts as built x `msw_rho` / site density - `depth_factor` x working depth, in m.

**N-03, station keeping at the rack.** `n03_station_thrust(vehicle, site)` = available astern thrust / drag at the
rack - `thrust_factor`. The vehicle faces the rack and thrusts astern against the design flow, at the end of the
dive.

1. `end_of_dive_bus(vehicle, water_temp_c)` gives the bus under a full astern demand. With E the open-circuit
   voltage `pack_ocv_v` at the state of charge `reserve` and the water temperature, and R the resistance
   `pack_r_ss_ohm` at the water temperature, the battery current is the smallest of the continuous rating of the
   pack, the factory preset and the current that pulls the bus down to `v_min`, (E - `v_min`) / R; when two of them
   are equal the first in that order is the one named. If the full-throttle demand of the four horizontal thrusters
   plus the hotel current is smaller still, the current is that demand (evaluated at the resulting bus voltage, in
   six passes). The bus voltage is E - current x R.
2. `thrust_at_current(vehicle, "surge_astern", bus voltage, battery current)` gives the static thrust. The hotel
   current, `hotel_load_w` / bus voltage, comes first; the four horizontal thrusters share the rest equally; the
   thrust of one thruster is its full thrust at that voltage x (its current / its full current) **
   `thrust_power_exp`, never more than the full thrust; the axis thrust in N is `thrust_system_coeff` x
   `axis_factor` x that thrust in kgf x G. `axis_factor("surge_astern", vehicle)` = 2 cos(front angle) + 2
   `reverse_ratio` cos(rear angle): the front pair runs forward and the rear pair in reverse. For `"surge_ahead"` it
   is 2 `reverse_ratio` cos(front angle) + 2 cos(rear angle). The shaft speed is the full-throttle speed x the
   square root of thrust / full thrust.
3. `inflow_factor(thruster, rpm, flow_mps)` = 1 - flow speed / (`prop_pitch_m` x rpm / 60), not below 0. The flow
   speed is the whole speed of the oncoming flow (`rack_flow_speed_mps`); it is not multiplied by the cosine of the
   thruster angle. Available thrust = static thrust x inflow factor.
4. `drag_n(vehicle, flow_mps, water_density_kg_m3, open_height_m)` = linear coefficient x v + quadratic coefficient
   x v ** 2 + 0.5 x density x `cd_tether` x tether diameter in m x v ** 2 x open height: the surge drag of the
   vehicle plus the drag of the tether in service over the open height of the rack.

**N-04, energy for one dive.** `n04_energy_reserve(vehicle, site)` = (usable energy - energy needed) / usable
energy; minus infinity when nothing is usable.

1. Field of view in water: `water_hfov_deg(camera_port, hfov_air_deg)`. A dome port keeps the value in air; a flat
   port gives 2 asin(sin(value in air / 2) / `n_flat_port`).
2. Strips: swath = 2 x standoff x tan(field of view in water / 2); number of strips = rack width / (swath x (1 -
   `overlap`)), rounded up to a whole number.
3. Time at the rack = strips x (rack height / `scan_mps` + `strip_turn_s`) + `contingency_s`.
4. Power at the rack = `hotel_load_w` + `hold_power_w(vehicle, drag at the rack, flow speed)` + `rack_extra_w`.
   `hold_power_w` is the electrical power of the four horizontal thrusters that hold the drag against the flow, from
   the 12 V column of the thruster page: the thrust t of one thruster solves t = drag / (`thrust_system_coeff` x
   `axis_factor("surge_astern", vehicle)` x G x inflow factor at the shaft speed of t), by iteration from t = 0.5
   kgf, with the inflow factor not below 0.05 and at most 60 passes; the power is 4 x `power_at_1kgf_w` x t ** (1 /
   `thrust_power_exp`).
5. Energy needed, in Wh = (power at the rack x time at the rack + (`hotel_load_w` + `transit_vert_extra_w`) x 2 x
   working depth / `transit_mps` + (`hotel_load_w` + `transit_face_extra_w`) x 2 x dam-face reach / `transit_mps`)
   / 3600.
6. Usable energy = `pack_energy_wh(pack, power at the rack, water_temp_c, v_min, reserve)` x `eta_dist`.

**N-05, tether reach.** `n05_tether_margin_m(tether_len_m, launch_path_m, face_reach_m)` = tether length - `slack`
x (`launch_path_m` + `face_reach_m`). `launch_path_m` is the tether length from the zero of the payout counter,
along the payout route, to the lowest point of the inspection scope.

**N-06, pitch with the sample.** `n06_pitch_authority(vehicle)` = (W x `bg_m` x sin(`pitch_lim_deg` -
`static_pitch_trim_deg`) + `active_pitch_moment_nm`) / (`sample_kg` x G x `jaw_reach_m`) - 1, with W = air mass x G.
The static trim is nose down positive and the pitch limit is taken from horizontal.

**N-07, water temperature of the wet parts.** `n07_wet_temperature_margin_c(min_water_temp_c,
wet_part_min_temps_c)` = lowest water temperature of the policy - the highest of the lower operating limits of the
wet parts.

**N-08, launch mass.** `n08_launch_mass_margin_kg(crane_swl_kg, air_mass_kg, rigging_kg)` = safe working load -
air mass - lifting gear.

**N-09, work near the structure.** `n09_near_structure_margin_m(sonar_min_range_m, standoff_m, dvl_min_altitude_m,
survey_altitude_m)` = the smaller of standoff - minimum sonar range and survey altitude - minimum altitude of the
velocity log.

**N-10, link test.** `n10_link_test_margin_m(link_test_length_m, tether_len_m)` = tested length - tether length.

**N-11, gripper opening.** `n11_gripper_opening_margin_mm(gripper_opening_mm, sample_max_mm)` = jaw opening -
largest sample.

**N-12, depth range.** `n12_depth_range_margin_m(depth_sensor_range_bar, working_depth_m, water_density_kg_m3)` =
full scale of the depth sensor, in metres of the site water, - working depth.

`k4_near_structure`, `k5_link_test`, `k6_gripper_opening` and `k7_depth_range` are the yes-or-no forms of N-09 to
N-12: true when the margin is zero or more.

**The eight `site` values.** `site_values(vehicle, site)` returns the values that `evaluation.json` asks for:

| key | value |
|---|---|
| `as_tendered_net_buoyancy_n` | net buoyancy of the unit as it is, without trimming, in the site water, in N (`n01_net_buoyancy_kg` x G) |
| `depth_rating_margin_m` | margin of N-02 |
| `station_thrust_ratio` | margin of N-03 + `thrust_factor`: available thrust / drag |
| `dive_energy_reserve_frac` | margin of N-04 |
| `tether_reach_margin_m` | margin of N-05 |
| `pitch_authority_ratio` | margin of N-06 + 1 |
| `wet_temperature_margin_c` | margin of N-07 |
| `launch_mass_margin_kg` | margin of N-08 |

`row_details(vehicle, site)` returns the intermediate values of N-03 and N-04: the site density, the flow speed, the
standoff, the tether length needed, the bus voltage and battery current at the end of the dive and the limit that
binds, the available thrust, the drag, the field of view in water, the strips, the time and power at the rack, the
energy needed and the usable energy.

## 7. Thruster pages and battery pages

- **Thruster page.** `thr_full(thruster, bus_v)` reads the full-throttle thrust, current and shaft speed of one
  thruster by linear interpolation in the voltage columns; below the first column and above the last it takes the
  end column.
- **Battery.** There is one battery model, for the needs and for the facility quantities alike.
  `pack_ocv_v(pack, soc, temp_c)` = open-circuit voltage from the table at that state of charge -
  `ocv_drift_v_per_c` x (`ocv_ref_temp_c` - temperature). `pack_retention(pack, temp_c)` reads the retention table.
  `pack_r_ss_ohm(pack, temp_c)` = `r_ss_23c_ohm` x resistance factor at that temperature. The temperature tables are
  read by linear interpolation between the printed nodes, and outside the nodes they take the end node.
- **`pack_energy_wh(pack, load_w, temp_c, cutoff_v, soc_end)`** is the one energy function: the energy in Wh that
  the pack delivers at the constant power `load_w`, from full charge, at `temp_c`, until the terminal voltage falls
  below `cutoff_v` or the state of charge reaches `soc_end`. The terminal voltage at constant power P is V = (E +
  sqrt(E ** 2 - 4 P R)) / 2, with E = `pack_ocv_v` and R = `pack_r_ss_ohm`; where E ** 2 is less than 4 P R there is
  no operating point and V is 0. The charge, `capacity_ah` x `capacity_factor` x retention at `temp_c`, is cut into
  400 equal steps from full charge to `soc_end`; V is evaluated at the ends of the steps; the energy is the
  trapezoid sum over the steps whose two ends are both at or above `cutoff_v`, plus the part of the first failing
  step up to the point where V crosses `cutoff_v`, found by linear interpolation of V between the two ends of that
  step. If V is below `cutoff_v` already at full charge the energy is 0.
- **`interp(x, xp, fp)`**: `xp` increasing; x below the first node gives the first value, x at or above the last
  node gives the last value, x on a node gives the value of that node; otherwise slope x (x - `xp[j]`) + `fp[j]`,
  with slope = (`fp[j+1]` - `fp[j]`) / (`xp[j+1]` - `xp[j]`).

## 8. The facility quantities of Part D

`FACILITY_QUANTITIES` is the closed dictionary in which an acceptance clause is written: each quantity with its unit
and the names of its conditions. `facility_value(quantity, conditions, vehicle)` returns the value of any of them
for a unit at the supplier's works. `conditions` is a `dict` with exactly the condition names of the quantity; a
numeric condition is a finite number; `axis` is one of `AXES`, `limiter` one of `LIMITER`, `medium` one of `MEDIUM`.
`submission_contract.md` gives the comparators that Part D allows for each quantity.

| quantity | unit | conditions | value |
|---|---|---|---|
| `air_mass_kg` | kg | none | `air_mass_kg` of the unit |
| `tank_net_buoyancy_n` | N | none | net buoyancy in the supplier's trim tank, positive when the unit floats: `tank_net_buoyancy_n(air_mass_kg, displaced_volume_l, tank_water_density_kg_m3)` = `n01_net_buoyancy_kg` in the tank water x G |
| `displaced_volume_l` | L | none | `displaced_volume_l` of the unit |
| `bollard_thrust_n` | N | `axis`, `supply_v`, `limiter` | `bench_bollard_thrust_n(axis, supply_v, limiter, vehicle)` |
| `bollard_bus_current_a` | A | `axis`, `supply_v`, `limiter` | `bollard_bus_current_a(axis, supply_v, limiter, vehicle)` |
| `battery_delivered_energy_wh` | Wh | `load_w`, `chamber_c`, `cutoff_v` | `battery_delivered_energy_wh(pack, load_w, chamber_c, cutoff_v)` = `pack_energy_wh` from full charge down to `cutoff_v`, with nothing held in reserve |
| `pack_dcir_mohm` | mOhm | `chamber_c` | `pack_dcir_mohm(pack, chamber_c)` = `pack_r_ss_ohm` at the chamber temperature x 1000 |
| `hotel_power_w` | W | `lights_pct` | `hotel_power_w(vehicle, lights_pct)` = `hotel_load_w` - `lights_load_w` x (1 - `lights_pct` / 100) |
| `proof_pressure_bar` | bar | `hold_min` | `proof_pressure_bar(vehicle, hold_min)` = the lowest rating among the pressure parts x `msw_rho` x G / 100000; the hold time does not change it |
| `tow_drag_n` | N | `speed_mps` | `tow_drag_n(vehicle, speed_mps)` = linear coefficient x speed + quadratic coefficient x speed ** 2: the surge drag without tether |
| `tether_length_m` | m | none | `tether_length_m` of the unit |
| `tether_diameter_mm` | mm | none | `tether_diameter_mm` of the unit |
| `camera_hfov_deg` | deg | `medium` | `camera_hfov_deg(vehicle, medium)`: the value in air for `"air"`, `water_hfov_deg` for `"tank_water"` |
| `sonar_min_range_m` | m | none | `sonar_min_range_m` of the unit |
| `gripper_opening_mm` | mm | none | `gripper_opening_mm` of the unit |

**The bench bollard.** `bench_bollard_thrust_n(axis, supply_v, limiter, vehicle)` is the one function behind the
bidders' factory bollard reports, the witnessed bollard pull of the desk and the facility quantity
`bollard_thrust_n`. The bench supply feeds the battery interface at `supply_v`. `limiter` `"on"` is the unit as
delivered: the firmware's limit on the battery current, which is the factory preset `current_preset_a`, is active,
and the thrust is `thrust_at_current` at that current. `limiter` `"off"` is full throttle on the four horizontal
thrusters with no current budget. `axis` is `"surge_ahead"` or `"surge_astern"`. `bollard_bus_current_a` is the
current of the bench supply in the same test: the full-throttle current of the four horizontal thrusters plus the
hotel current, and no more than the factory preset when the limiter is on.
