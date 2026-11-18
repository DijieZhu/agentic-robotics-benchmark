# Tender specification TS-01 rev C

**Inspection-class remotely operated vehicle (ROV) for the yearly inspection of the intake tower**

Issued by the Owner on 6 July 2026. Bids close on 18 September 2026.

| Part | Content |
|---|---|
| A | The duty at the site, the twelve needs N-01 to N-12 that follow from it, and the Owner's policy: the coefficients and the rules by which the needs are judged |
| B | The numerical clauses B-01 to B-24 |
| C | How the bids are evaluated |
| D | How the delivered unit is accepted at the supplier's works |

The calculations to which Parts A, C and D refer are published with this specification as the evaluation procedure,
`evaluation_basis.py`; `evaluation_basis.md` describes it. Where this text and the procedure differ, the procedure
counts.

Words used in all four parts. The *Owner* operates the plant and buys the vehicle. The *Owner's engineer* evaluates
the bids and writes the acceptance schedule. A *bidder* is an integrator that has submitted a bid; the *supplier* is
the bidder to whom the contract is awarded. The *works* are a bidder's own premises with its test facility. The
*site* is the intake tower and the reservoir in front of it.

## Part A. Duty, needs and policy

### A.1 The duty

Once a year, in the same calendar month every year (the *inspection month*), the Owner inspects the intake tower of
the reservoir. The inspection scope is:

1. the trash rack, recorded on video bay by bay;
2. the sill of the gate slot and the debris sump, down to the floor of the sump;
3. the dam face on either side of the intake, as far as the reach marks shown on drawing GA-01.

During the inspection the vehicle also recovers, with its gripper, samples of the debris lodged on the rack.

The plant stays in operation during the inspection. Work at the intake is released by a permit to work that is valid
for the same hours every night (the *permit window*); the permit states its hours in local time.

The site is documented by the Owner's drawings and records, which are issued with this specification. This Part
states duties and rules. It gives no dimension, level, flow or temperature of the site.

### A.2 The twelve needs

A vehicle is fit for the duty when it meets all twelve needs. For each need the evaluation procedure computes a
signed margin; zero or more meets the need.

| Need | Name | What the Owner needs |
|---|---|---|
| N-01 | Trim on site | In the water of the reservoir the vehicle is close to neutral: its net buoyancy lies inside the Owner's window (A.4). N-01 is judged by one of two named rules; Part C states the rule of the evaluation and Part D the rule of the acceptance. |
| N-02 | Depth rating | Every pressure part of the vehicle, as built, is rated for the working depth with the Owner's depth factor. |
| N-03 | Station keeping at the rack | At the end of a dive, facing the trash rack and thrusting astern, the vehicle holds its station against the design flow through the rack, with the Owner's thrust factor. |
| N-04 | Energy for one dive | One battery charge carries the vehicle through the reference dive (A.5) with the Owner's reserve. |
| N-05 | Tether reach | The tether reaches every point of the inspection scope, with the Owner's slack. |
| N-06 | Pitch with the sample | With the sample in the gripper the vehicle stays within the pitch limit. |
| N-07 | Water temperature | Every part of the vehicle that is in the water is rated for the coldest water of the inspection. |
| N-08 | Launch mass | Ready to dive and slung in the Owner's lifting set, the vehicle is within the safe working load of the Owner's lifting appliance at the intake tower. |
| N-09 | Work close to the structure | The imaging sonar works at the inspection standoff, and the velocity log keeps bottom lock at the survey altitude. |
| N-10 | Data link | The data link has been proven over a tether at least as long as the tether supplied. |
| N-11 | Gripper | The gripper opens wide enough for the largest sample. |
| N-12 | Depth measurement | The range of the depth sensor covers the working depth. |

### A.3 The site quantities and how they are established

Each quantity below is established from the Owner's drawings and records by the rule stated here, and by no other
rule.

- **Design water level.** The full supply level (FSL) of the reservoir.
- **Lowest point.** The lowest elevation inside the inspection scope of A.1.
- **Working depth.** The design water level minus the lowest point.
- **Intake station.** The Owner's monitoring station that represents the water at the intake.
- **Bottom water temperature.** The water temperature at the bottom of the cast taken at the intake station in the
  inspection month. It is used as measured, **without rounding**, for the density of the site water and for the
  battery in needs N-03 and N-04.
- **Site water.** Fresh water. Its density is the density of pure water at the bottom water temperature and at the
  pressure of the working depth. No correction is made for salinity.
- **Lowest water temperature.** The lowest water temperature measured at the intake station in the inspection month,
  between the surface and the working depth, **rounded down to 0.1 degC**. Need N-07 uses it.
- **Design discharge.** The largest hourly discharge of the whole plant, all generating units together, that the
  record holds for the hours inside the permit window, taken in local time, in the inspection month; **rounded up to
  0.5 m3/s**.
- **Design flow speed at the rack.** The design discharge divided by the flow area of the rack: its width times its
  height open to the flow.
- **Secchi statistic.** The lower quartile of the Secchi depths read at the intake station in the inspection month,
  over the whole record; **rounded down to 0.1 m**.
- **Inspection standoff.** The distance from the camera to the face of the rack during the scan: the Secchi statistic
  times the standoff factor of A.4.
- **Dam-face reach.** The distance along the dam face from the intake to the reach mark, on one side.
- **Tether path.** The length of tether from the zero of the payout counter, along the payout route, to the lowest
  point; plus the dam-face reach on one side.

### A.4 The Owner's policy values

The Owner fixes the values of this section and of A.5 for this procurement. The evaluation procedure holds them as
its `POLICY` constants, except the survey altitude and the largest sample: these are duties of the vehicle, which
the procedure takes as inputs (`survey_altitude_m`, `sample_max_mm`).

| Policy | Value | Needs |
|---|---|---|
| Net-buoyancy window in the site water, as a share of the mass of the vehicle in air; positive is buoyant | from -0.05 % to 0.95 % | N-01 |
| Depth factor: the rated depth of every pressure part is at least this factor times the working depth | 1.25 | N-02 |
| Density by which a rating in metres of sea water (msw) is converted to metres of the site water | 1025 kg/m3 | N-02 |
| Thrust factor: the thrust available at the rack is at least this factor times the drag | 1.25 | N-03 |
| State of charge at the end of the dive | 20 % | N-03, N-04 |
| Lowest bus voltage | 12.0 V | N-03, N-04 |
| Drag coefficient of the tether in the flow | 1.2 | N-03, N-04 |
| Efficiency of the power distribution, applied to the usable energy of the battery | 0.97 | N-04 |
| Refractive index used for a flat camera port | 1.333 | N-04 |
| Standoff factor: inspection standoff as a share of the Secchi statistic | 0.4 | N-04, N-09 |
| Tether slack: the tether is at least this factor times the tether path | 1.10 | N-05 |
| Sample: its weight in water, expressed as a mass | 1.5 kg | N-06 |
| Pitch limit with the sample held, measured from horizontal | 5 deg | N-06 |
| Survey altitude above the sill and above the floor of the sump | 0.3 m | N-09 |
| Largest sample to be recovered, across | 50 mm | N-11 |

The state of the vehicle at the end of a dive (N-03): the battery at the state of charge above, in water at the
bottom water temperature; the battery current within the continuous rating of the pack and within the current preset
of the unit; the bus voltage not below the lowest bus voltage.

### A.5 The reference dive

Need N-04 is judged on one reference dive, flown on one battery charge down to the state of charge and the bus
voltage of A.4, in water at the bottom water temperature. Each allowance below is electrical power added to the
hotel load of the vehicle.

1. **Descent and ascent.** Twice the working depth at the transit speed of 0.5 m/s. Allowance:
   60 W.
2. **Trash rack.** The whole rack is scanned in vertical strips at the inspection standoff, while the vehicle holds
   its station against the design flow. The width of a strip is what the forward camera sees of the rack face at that
   standoff, in water; neighbouring strips overlap by 20 %. Each strip is flown over the full height
   of the rack at 0.05 m/s, and each strip takes 30 s more for the change of strip. A
   contingency of 120 s at the rack is added. Allowance, on top of the power that holds the
   station: 6 W.
3. **Dam face.** From the intake to the reach mark on one side and back, at the transit speed. Allowance:
   30 W.

## Part B. Numerical clauses

The clauses of this Part follow the company's general template for inspection vehicles, which was written for its
coastal assets. They stand here as the template states them. Each clause names the need of Part A that it supports;
`general` marks a clause that supports none. In its compliance matrix a bidder states Comply or Deviate against
every clause, with the value it declares and the evidence for it.

| Clause | Title | Requirement | Supports |
|---|---|---|---|
| B-01 | Trim as shipped | The vehicle shall leave the works trimmed to within +/-50 g of neutral buoyancy in the supplier's trim tank. | N-01 |
| B-02 | Weighing record | The trim sheet shall state the mass in air and the net buoyancy in the trim tank, each read to 5 g or finer. | N-01 |
| B-03 | Depth rating | The vehicle shall be rated for a working depth of at least 100 m. | N-02 |
| B-04 | Proof test | Every pressure housing shall be proof-tested at not less than its rated pressure, held for at least 30 min. | N-02 |
| B-05 | Depth measurement | The depth sensor shall have a measuring range of at least 100 m. | N-12 |
| B-06 | Battery discharge rating | The battery pack shall be rated for a continuous discharge current of at least 40 A. | N-03 |
| B-07 | Bollard thrust | The vehicle shall develop a bollard thrust of at least 12.0 kgf along its longitudinal axis, measured astern. | N-03 |
| B-08 | Speed | Maximum forward speed through the water shall be at least 1.5 m/s. | N-03 |
| B-09 | Tether length | The tether shall be at least 200 m long. | N-05 |
| B-10 | Tether diameter | The tether diameter shall not exceed 8.0 mm. | N-05 |
| B-11 | Battery energy | The nominal energy of the battery pack shall be at least 250 Wh. | N-04 |
| B-12 | Camera field of view | The forward camera shall have a horizontal field of view of at least 80 deg in air. | N-04 |
| B-13 | Lights | At least 4 lights shall be fitted. | general |
| B-14 | Battery in cold water | At 10 degC the battery pack shall deliver at least 80 % of its rated capacity. | N-04 |
| B-15 | Thrusters | At least 6 thrusters shall be fitted. | N-06 |
| B-16 | Sonar minimum range | The scanning sonar shall have a minimum range of 1.0 m or less. | N-09 |
| B-17 | Protection near structures | Propellers shall be ducted, and the vehicle shall tolerate contact with fixed structures at 0.3 m/s without damage to thrusters, camera port or sensors. | N-09 |
| B-18 | Operating temperature | The vehicle shall operate in water between 0 degC and 30 degC. | N-07 |
| B-19 | DVL minimum altitude | The DVL shall keep bottom lock down to an altitude of 0.5 m or less. | N-09 |
| B-20 | Data link | The data link shall be proven over at least 250 m of tether. | N-10 |
| B-21 | Gripper opening | The gripper shall open to at least 40 mm. | N-11 |
| B-22 | Mass in air | The vehicle ready to dive shall weigh no more than 20 kg in air. | N-08 |
| B-23 | Static trim | Static pitch as trimmed shall be within +/-5 deg. | N-06 |
| B-24 | Spare seals | At least 1 complete set of spare seals for every pressure housing shall be supplied. | general |

## Part C. Evaluation of the bids

**C.1 Basis.** A bid is evaluated for its fitness for the duty of Part A: against the twelve needs, under the
conditions of the Owner's site as they really are, with the evaluation procedure. A need is met when its margin, as
the procedure computes it, is zero or more.

**C.2 The bid.** A bid consists of a compliance matrix against Part B, a technical description with its revision
record, a bill of materials, a price sheet with the base price and two priced options, a delivery statement, and
five dated factory test reports: TR-1 trim sheet, TR-2 bollard pull, TR-3 pressure test, TR-4 battery discharge and
TR-5 tow test.

**C.3 The tendered unit.** What is evaluated is the tendered unit: the unit that the bidder offers to deliver, as it
is built and in the state in which it is tendered. A *configuration* is the bid with none, one or both of its priced
options. Its tendered unit is that unit with those options fitted, and its price is the base price plus the prices
of those options.

**C.4 Clarification.** The Owner's engineer may put questions to a bidder and may witness tests of the tendered unit
at the bidder's works. What a bidder answers and what a witnessed test reads belong to the bid.

**C.5 Trim in the evaluation.** In the evaluation, need N-01 is judged by the rule `n01_trim_reachable`. It asks
whether the bid's own ballast kit allows the unit to be ballasted for the site water: of the ballast states that the
kit can realise, with removable pieces taken out or loose spare pieces added, at least one puts the net buoyancy in
the site water inside the window of A.4. A unit whose ballast cannot be changed is judged as it is ballasted.

**C.6 Accepting a bid.** A bid is acceptable when at least one of its configurations meets all twelve needs. It is
accepted with the cheapest such configuration, and the evaluation names the options of that configuration. An
option that is not needed is not bought.

**C.7 Rejecting a bid.** A bid none of whose configurations meets all twelve needs is rejected. The rejection cites
one or two needs as its grounds. A need is a ground for rejecting a bid only when the bid fails it with every subset
of its options. **Every ground cited must hold.** A rejection that cites a need which is not such a ground does not
stand, whatever else it cites.

**C.8 Award.** The contract is awarded to the accepted configuration with the lowest price. The award has to stand
against every rejected bid whose base price is below the price of the awarded configuration: each of them must have
been rejected on grounds that hold. When no bid is acceptable, no award is made.

**C.9 Part B in the evaluation.** Whether a bid complies with a clause of Part B or deviates from it is recorded for
the review of the specification (C.10). It is not a ground for accepting a bid, and it is not a ground for rejecting
one.

**C.10 Review of the specification.** With the evaluation, the Owner's engineer reviews every clause of Part B
against Part A and the site, and reports for each clause whether it is too lax and whether it is too strict. A
clause is too lax when a unit can comply with it and still fail the need that the clause supports. A clause is too
strict when it would turn away a unit that is fit for the duty.

## Part D. Acceptance at the works

**D.1 The acceptance schedule.** The contract is awarded with a factory acceptance schedule, which the Owner's
engineer writes for the awarded configuration. The supplier is held to the clauses of the schedule and to nothing
else. What the schedule does not demand, the supplier does not owe.

**D.2 A clause.** A clause states one facility quantity of the dictionary (Table D-1), the conditions of its test,
written out in full, and a bound on one side: the quantity is to be at least the bound (`>=`), or at most the bound
(`<=`). **The comparator of a clause is one that the dictionary allows for its quantity.** Where the dictionary
allows both comparators, a window is written as two clauses. Nothing has a default: a condition that a clause does
not write is not agreed.

**D.3 The facility.** The acceptance tests are run at the supplier's works, in the facility that section 1 of its
delivery statement describes. A condition for which that section gives a range lies inside the range. The quantities
of the dictionary form six groups, one for each kind of test:

- weighing: in the supplier's trim tank, in the water that the tank holds;
- propulsion: in the bollard pool, with a bench supply at the battery interface;
- energy: on the battery cycler, in the climatic chamber;
- pressure: in the pressure vessel;
- hull and tether: in the tow tank, and the tether on its drum;
- sensing and work: the camera, the sonar and the gripper on the bench.

**D.4 What the supplier may deliver.** Section 2 of the delivery statement declares the delivery band: the units
that the supplier may deliver under the contract. The supplier may deliver any unit of the band that passes every
clause of the schedule. Where several units pass, it delivers the one that suits it best, in the order of
preference that section 3 of its delivery statement declares.

**D.5 The delivered state.** The site has no facility for trimming a vehicle. The delivered state is therefore the
launch state: the ballast is fitted and sealed at the works, the Owner does not adjust it at the site, and loose
spare pieces are not part of the delivered state. In the acceptance, need N-01 is judged by the rule
`n01_as_delivered`: the delivered unit, as it is ballasted at the works, has its net buoyancy in the site water
inside the window of A.4.

**D.6 A schedule that cannot be contracted.** When no unit of the delivery band passes every clause, or when a
clause asks for a test that the facility cannot run, no contract can be made on the schedule.

**D.7 What the schedule has to achieve.** The schedule is to be sufficient: no unit of the band that passes it
fails a need of Part A at the site, as delivered. It is also to be economical: it does not turn away units of the
band that are fit for the site. Stock that the schedule turns away without need is paid for by the Owner.

**Table D-1. The dictionary of facility quantities**

| Group | Quantity | Unit | Conditions of the test | Comparators allowed |
|---|---|---|---|---|
| weighing | `air_mass_kg` | `kg` | (none) | `>=`, `<=` |
| weighing | `tank_net_buoyancy_n` | `N` | (none) | `>=`, `<=` |
| weighing | `displaced_volume_l` | `L` | (none) | `>=`, `<=` |
| propulsion | `bollard_thrust_n` | `N` | `axis`, `supply_v`, `limiter` | `>=` |
| propulsion | `bollard_bus_current_a` | `A` | `axis`, `supply_v`, `limiter` | `>=` |
| energy | `battery_delivered_energy_wh` | `Wh` | `load_w`, `chamber_c`, `cutoff_v` | `>=` |
| energy | `pack_dcir_mohm` | `mOhm` | `chamber_c` | `<=` |
| energy | `hotel_power_w` | `W` | `lights_pct` | `<=` |
| pressure | `proof_pressure_bar` | `bar` | `hold_min` | `>=` |
| hull and tether | `tow_drag_n` | `N` | `speed_mps` | `<=` |
| hull and tether | `tether_length_m` | `m` | (none) | `>=` |
| hull and tether | `tether_diameter_mm` | `mm` | (none) | `<=` |
| sensing and work | `camera_hfov_deg` | `deg` | `medium` | `>=` |
| sensing and work | `sonar_min_range_m` | `m` | (none) | `<=` |
| sensing and work | `gripper_opening_mm` | `mm` | (none) | `>=` |
