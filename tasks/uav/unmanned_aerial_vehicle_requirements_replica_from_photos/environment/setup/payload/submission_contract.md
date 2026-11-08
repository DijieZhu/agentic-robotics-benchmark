# Submission contract
Use the schema files as the machine-readable authority. JSON examples below describe keys, not measured answers.
`requirements.yaml` has `facets` (list of `{id, value, unit, sources}`) and `derived` (mapping from quantity ID to
`{value, unit, sources}`). Sources are arrays of exact IDs, for example `["B-07", "DS-03", "LR-B"]`.
Facet IDs must be unique. Report these facets (units in parentheses):
- `replica_count` (count)
- `core_policy` (text)
- `control_stack` (text)
- `payload_mass` (kg)
- `mission_duration` (s)
- `auxiliary_power` (W)
- `tip_clearance` (mm)
- `reserve_fraction` (1)
- `delivered_lot` (text)
- `battery_series` (count)
- `motor_mount` (mm)
- `old_envelope_status` (text)

Text facet vocabulary: core_policy="retain"; control_stack="PX4/ROS2";
delivered_lot="A" or "B"; old_envelope_status="current" or "superseded".
For each quantitative facet report its requirement value, not a paraphrase. Reference at least one source that
actually establishes it. Facet credit requires the right value, unit and a supporting source; aggregation is F1
against the required facets, so extra unsupported facets reduce precision. Numeric facet tolerances: payload
0.001 kg; mission 0.1 s; auxiliary power 0.1 W; clearance 0.01 mm; reserve 0.001; mount 0.1 mm; counts exact.

Required derived IDs (evaluate on YOUR chosen BOM):
- `thrust_min_FL_n` (N)
- `thrust_min_FR_n` (N)
- `thrust_min_RL_n` (N)
- `thrust_min_RR_n` (N)
- `thrust_peak_FL_n` (N)
- `thrust_peak_FR_n` (N)
- `thrust_peak_RL_n` (N)
- `thrust_peak_RR_n` (N)
- `esc_peak_FL_a` (A)
- `esc_peak_FR_a` (A)
- `esc_peak_RL_a` (A)
- `esc_peak_RR_a` (A)
- `esc_rms_FL_a` (A)
- `esc_rms_FR_a` (A)
- `esc_rms_RL_a` (A)
- `esc_rms_RR_a` (A)
- `battery_peak_a` (A)
- `minimum_voltage_v` (V)
- `mission_energy_wh` (Wh)
Missing derived quantities receive zero credit for that quantity. Compare each against the actual unit with your
BOM; relative error <=3% receives full credit, >=15% receives zero, linear between. The denominator floor is
0.2 in each stated unit. C3 is the equal mean over these 19 propulsion sizing quantities. Minimum thrust is the smallest individual allocated thrust over all 65 states.
Peak thrust and peak channel current also use all 65 states;
RMS channel current is time weighted over all 65 states. Channel currents exclude auxiliary current.
The public procedure also returns hover thrust and intermediate mass, CoM, inertia and aggregate moment audit values. Those
are not submission fields or C3 targets in this contract; they support reproducing the sizing calculation.

`bom.json` has `motors` mapping FL/FR/RL/RR to SKUs, plus `propeller`, `esc`, `battery` SKU strings.
`kinematic_sketch.json` has `dimensions_mm`, mapping each required key to `{nominal, tolerance, sources}`,
`rotor_cant_deg` and `rotor_radial_tilt_deg`, each mapping FL/FR/RL/RR to `{nominal,tolerance,sources}`, and `spin`, mapping stations to CW/CCW.
Linear tolerances are 0.1..3 mm; angle tolerances are 0.1..3 degrees. They document fabrication intent.
Nominal dimensional errors <=0.5 mm receive full credit and >=3.5 mm zero; angle errors <=1 degree receive
full credit and >=4 degrees zero, with linear interpolation. Broad tolerances earn no shortcut.
The equal mean over all dimensional and angular measurements contributes 95% of sketch credit; spin supplies 5%.
Angles are signed as B-04 defines and belong to the final individual adapters.

Dimension definitions:
- motor_x_mm, motor_y_mm: positive half-spans of the shaft rectangle, NOT diagonal wheelbase.
- frame_x_mm, frame_y_mm, frame_z_mm: bounding extents of the retained core only, including its original guards
  and feet; excluding the added cage, scanner, ballast pod, canted adapters and stand-off posts. See BL-01/02.
- guard_inner_mm: diameter of the smallest shaft-centred circle wholly inside every guard's inner opening.
- motor_mount_mm: hole-to-hole pitch in the square mount.
- battery_bay_x_mm, battery_bay_y_mm, battery_bay_z_mm: internal clear cage dimensions in vehicle axes.
- battery_x_mm, battery_y_mm, battery_z_mm: cage interior centre relative to shaft-plane origin.
- payload_x_mm, payload_y_mm, payload_z_mm: ballast pod box centre relative to that origin.
- sensor_x_mm, sensor_z_mm: scanner box centre. Its y is zero.
- payload_box_x_mm, payload_box_y_mm, payload_box_z_mm: external payload box extents.
- sensor_box_x_mm, sensor_box_y_mm, sensor_box_z_mm: external scanner box extents.

To call `procedure.evaluate(bom, geometry, catalogue, duty)`, reconstruct the numeric inputs from the evidence.
Geometry consists of `dimensions_mm` (plain numbers; add esc_x_mm=0, esc_y_mm=0, esc_z_mm=-22), `payload_mass_kg`, `rotor_cant_deg` and `rotor_radial_tilt_deg` (station-to-angle mappings in degrees),
and `fixed_components`, a list of `{name,mass_kg,position_mm:[x,y,z],box_mm:[x,y,z]}` for dry_core and sensor.
Reconstruct the final dry_core mass from BL-01 before calling the procedure; the calibration master is not the
photographed unit. Qualified propeller `plane_offset_mm` is supplied by DS-02.
Catalogue keys: motors, props, escs, batteries, lots, battery_lot, esc_lot. Each product dictionary is indexed by SKU.
Product keys and SI units are named explicitly in the public Python function; vendor pages supply all values.
Catalogue `usd` is the unit price from Q-01. Lot dictionaries contain the retained factors computed by B-06.
Duty is a dictionary with g, reference_voltage_v, aux_power_w, usable_fraction, tip_clearance_mm,
reaction_signs, states, eco_usd and replicas. Each state has name, seconds, az_m_s2, ax_m_s2, omega_rad_s,
alpha_rad_s2, gust_n (three-vector in N).
`procedure.passing_designs` can enumerate the quoted choices on YOUR reconstructed inputs. There is no hidden
measurement query. The submitted deliverables do not include those reconstructed helper dictionaries.

All five criteria use the actual geometry and delivered lot. B-10 is all-or-nothing acceptance; B-11 defines economy.
Structure and submission-integrity failures multiply the total by zero. The grader executes no submitted code.
