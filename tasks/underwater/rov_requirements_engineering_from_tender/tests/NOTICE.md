# NOTICE (grader-side; never staged into the agent's home)

This file is the task's attribution. The agent-visible material (instruction.md and everything under
`environment/setup/payload/`) names no platform, maker, product, part number, simulator, dataset or author: the vehicle
is "an inspection-class ROV" of one platform family, every catalogue maker, model code and bidder is fictional, and
the generic names come from the underwater tasks' naming table. The benchmark's owner has stated that the sources below may be used and
distributed with the benchmark; no separate licence review was made for this task.

## The reference vehicle

The eight bids are configurations of one platform family. The reference bid (T-03 as tendered and tested) and the
family's geometry, masses, displaced volumes, centres, thruster layout and ballast stations are
those of the **BlueROV2 R4 in its Heavy (8-thruster) configuration by Blue Robotics**, with the vendor's accessories:
the Newton subsea gripper, the Ping360 scanning imaging sonar, a Water Linked DVL A50 on the vendor's bracket, four
Lumen lights and the Bar30 depth sensor. The numbers were measured once from the vendor's public R4 STEP model, the
official Heavy retrofit parts and the official assembly guides into the underwater tasks' frozen reference-vehicle
digest; `digest/spec_values.json` (sha256 `358676915495f3ea4e719154634557669b14a69a7554dce6a134cb80bf3b9bbd`)
is a byte copy of it, and the grader and the world generator read every digest-derived quantity from it at run time.
No variant of the vehicle's geometry was made: the eight bids differ only in what their documents state (options,
tubes, packs, cameras, tether, ballast, the T-06 guard cage), and T-03 as tendered and tested is the digest's
configuration. One hidden state is added to it: after bids close, T-03's works fit a buoyancy float of
the foam grade the vehicle already carries inside the frame and re-trim the unit (three ballast pieces fixed in the
float's cradle, four removable pieces fitted, three loose), and every unit of that bid's delivery band is built the
same way. The float is the task's own, not a vendor part; its figures (0.40 L displaced and 0.24 kg in air, with its
cradle and fasteners: `world/as_built.json`) are the task's own round values, not a measurement, and no file states or
implies a density of the foam. Only T-03's desk replies and its witnessed weighing give the unit as built; its dated
documents describe the unit as tested.

The nominal hull drag of the reference bid (`xu` 8.87 N s/m, `xuu` 50.6 N s^2/m^2 in `world/as_built.json`) is the
midpoint, arithmetic for the linear term and geometric for the quadratic one, of two published identifications of the
same vehicle: C.-J. Wu, *6-DoF Modelling and Control of a Remotely Operated Vehicle*, master's thesis, Flinders
University, 2018; and M. von Benzon et al., "An Open-Source Benchmark Simulator: Control of a BlueROV2 Underwater
Robot", *Journal of Marine Science and Engineering* 10(12):1898, 2022 (its coefficients as transcribed, identically,
by two independent open-source projects; the article itself was not retrieved).

## The catalogue and its public reference values

The image catalogue (29 datasheets and 4 errata) prints fictional products whose performance and electrical values
were perturbed by 10-30 % from public reference values, keeping the physics consistent (thrust against power, depth
rating against tube length, cell temperature derating); the world reads the printed values. The reference values,
used on the maintainers' side only (none is printed on an agent-visible page but one: the second maker's two pack
pages, as the value the energy computation reads, and one bidder's thrust report, as its bench supply, print the
generic nominal voltage of a four-cell pack, 14.8 V; the task's numeric scan checks every other one and lists these
as registered exceptions):

| reference | public source |
|---|---|
| thruster: full-throttle thrust, current and speed at 10-18 V, the reverse ratio, a part-throttle power law and the derived pitch | the vendor's T200 thruster page and its published performance table |
| battery packs: 14.8 V 18.0 Ah (4S6P), 60 A continuous; the smaller 15.6 Ah pack | the vendor's lithium-ion battery pages; cell data sheets of the Samsung INR18650-30Q (internal resistance x 4S6P) and the Molicel INR18650-P26A (low-temperature capacity, read from its discharge-temperature chart) |
| tether diameters 7.6 mm and 4.0 mm, a tested link length of 300 m | the vendor's Fathom tether and Fathom-X link pages |
| depth ratings of acrylic and aluminium tubes, the foam and the vehicle | the vendor's enclosure tube tables and the vehicle's component depth-rating table (foam of the R-3318 class) |
| camera: 80 deg horizontal field of view in air behind the dome | the vendor's low-light HD USB camera page |
| hotel loads: 15 W per light, 5 W sonar, 4 W DVL | the vendor's Lumen and Ping360 pages; the Water Linked DVL A50 data sheet |
| headline numbers (9 kgf bollard class, 100 m / 300 m, 1.5 m/s, 266 Wh) | the vendor's BlueROV2 product page |

## The fictional world

The reservoir, its intake tower, the drawings, the level, discharge, CTD and clarity logs, the previous campaign's
vehicle, telemetry and notes, the flood of spring 2026 that blocked the rack's lowest bay and the previous pilot's
later look at it, the owner's staff, the eight bidders, their prices, test facilities and the change one of them
made after bids closed, and the catalogue makers are fictional; no real reservoir, operator, company or dataset was
used. The logs were synthesised by this task's generator from the frozen world (`world/site.json`).

## Physics of the public procedure

`evaluation_basis.py` (the grader's pinned copy; the agent has the same bytes) computes the density of pure water with
the 1980 international equation of state of seawater at zero salinity (EOS-80), in the form of N. P. Fofonoff and
R. C. Millard, *Algorithms for computation of fundamental properties of seawater*, UNESCO Technical Papers in Marine
Science 44, 1983, with temperatures converted from ITS-90 to IPTS-68 (t68 = 1.00024 t90). The flat-port refraction
uses n = 1.333 for water. Every other rule of the procedure (policy factors, statistics, the twelve needs) is the
fictional owner's own.

## The score

`tender_evaluation_score` = G0 x G1 x (0.35 V + 0.25 A + 0.10 P + 0.05 Q + 0.25 F) on [0, 1]. G0 checks structure only
(the three files, strict JSON, the schemas, the ids of the public sets, the options of each bid, the clause form of
the facility dictionary); G1 is the submission's integrity after the agent phase. V scores the verdicts (accept with
the cheapest fit option subset; reject on grounds that hold: one of the twelve needs, GF or PART-D), A the award
(fit, contractable, the cheapest of the submission's own acceptances, the cheaper bids soundly rejected, and the
purchase outcome D: at every tank-water density that bidder declares, the whole of a declared range included, some
unit passes the award's schedule and the unit the supplier delivers first clearly fails no need), P the site
predictions of the accepted bids, Q the Part B labels, F the acceptance schedules: one per accepted bid, at most 12
clauses each, each judged on that bid's own delivery band at every declared density by the share of the band's needs
it protects (against any unit that passes at some density) and the smallest share of the clearly good units it still
passes at a density. A schedule entry for a bid the evaluation does not accept is ignored; an accepted bid without
an entry has an empty schedule, which protects nothing. The privileged reference scores 1.0 and an untouched
submission 0.

## Known properties of the score

This folder replaces the form in which the task was first merged (2026-11-17) as a whole: the hidden world, most
documents, the reference's evaluation and schedules and two of the three submission schemas are new; the instruction,
the submission contract, the desk guide and the desk's tables are revised (one acceptance schedule for each accepted
bid; the grounds `GF` and `PART-D`; 167 of the desk's 248 replies differ from the first form's, 79 with a new text and
88 witnessed tests that keep their text and return new measured values, and all four of its attachment pages are new).
37 of the 154 files of the agent's home are byte-equal to the first form's (`tests/review/README.md`). What is known of
the score:

- One property of the world is registered against the criteria fixed for it (three checks of the band audits, one
  cause). The units of the award-position bid's delivery band (T-03) fail on two needs, the ballast state (N-01) and
  station keeping at the lowest current-limit preset (N-03), and on none that a clause on the pack would separate:
  that bid's grades B and C of the pack come from one stock. The audits that ask every accepted bid's band for a
  further failing need, or that test a clause on the pack, therefore have nothing to bite on there, and that bid's
  schedule needs no clause on the pack. The band was set so that the purchase outcome of the award does not depend on
  the open height of the rack (`tests/review/fairness.md`, "What scores and what does not"). No other deviation of the
  world is registered.
- The first form's files stay public in this repository's history. They do not score here: its reference, as it is,
  with its schedule file renamed, or its evaluation alone, is refused by this folder's structure check (0.0). The
  documents of both forms follow the same templates (voice, report forms, dates), and the Part B labels of this
  folder's reference are those of the first form's reference (24 of 24; weight 0.05 of the total). None of this is
  available to the agent: its sandbox has no network and holds none of the first form's files (only the 154 files of
  the payload).
- The reward is `min(1, score / 0.98)` (`tests/anchor.json`); the instruction does not print the 0.98.

## The cost of the T-06 guard cage

T-06 tenders the vehicle inside a protective cage for work in contact with the rack (catalogue: guard frame), with
mesh guards fore and aft of the four horizontal thrusters. The world multiplies its system thrust factor by `k_eta`
0.74 and its hull drag by `k_drag` 1.50 (`world/as_built.json`), inside the ranges the task's rules allow (x 0.74-0.90
and x 1.20-1.50, +0.8 kg). The ranges
are an engineering estimate of this task, not a measurement or a simulation:

- thrust: a guard in the jet removes momentum from it; with a loss coefficient k of the guard on the jet's dynamic
  pressure, static thrust falls by about k / 2 (actuator-disc momentum theory, e.g. the momentum-theory chapter of
  J. Carlton, *Marine Propellers and Propulsion*); a bar grid or mesh of 0.75-0.85 open-area ratio has k of about
  0.2-0.5 (the loss-coefficient relations for grids and screens of I. E. Idelchik, *Handbook of Hydraulic
  Resistance*), so one guard costs about 10-25 % of the thrust and the cage bars a few per cent more;
- drag: a ring of 20-25 mm tube round the frontal envelope with four guards adds 0.015-0.035 m^2 of bluff projected
  area at a subcritical cylinder drag coefficient of 1.0-1.2 (S. F. Hoerner, *Fluid-Dynamic Drag*, 1965, the
  chapters on cylinders and on frameworks), that is 20-50 % of the reference hull's drag area of about 0.10 m^2;
- the guard itself is ordinary practice: the MATE ROV Competition's safety rules require every propeller to be
  shrouded and guarded with a mesh finer than 12.5 mm (IP-20, "if your finger can touch the propeller, then it is
  not properly guarded").

The chapter and page references above were not re-checked against the books for this release (no library access
while the task was built); the numbers are the task's own rounded estimates. The agent sees only the cage's catalogue
page (mass, material, outline, purpose) and its listing in T-06's technical description; no agent-visible file prints
its thrust or drag cost. The factory retest reply of T-06 and the witness bench answer with the caged vehicle.

## Code and fonts

- `environment/setup/lab/alelab/server.py`, `stop.py`, `environment/setup/payload/labclient.py` and
  `tests/alelab_stop.py` are byte copies of the benchmark's lab service of `tasks/arm/arm_hidden_mechanism_box`;
  `tests/robotics_grader/` is a byte copy of the benchmark's `shared/robotics_grader/` kit.
- The page images were drawn with Pillow and matplotlib's bundled DejaVu fonts (DejaVu Sans, Bold, Oblique, Mono and
  Serif; the Bitstream Vera derived free licence). The handwriting on the dive log and on one bidder's test reports is
  drawn glyph by glyph from DejaVu with jitter; no handwriting font was used. The PNG files carry no text chunks.
