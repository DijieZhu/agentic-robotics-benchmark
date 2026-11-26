# Evaluate eight bids for an inspection-class ROV and write the acceptance schedules

A hydro-power company, the Owner, is buying one inspection-class remotely operated vehicle (ROV) for the yearly
inspection of the intake tower of a high, cold, deep fresh-water reservoir: the trash rack bay by bay, the sill of
the gate slot and the sump, the dam face on both sides of the intake, and samples of debris taken off the rack with
the gripper. The Owner has issued tender specification TS-01 rev C. Eight integrators have each submitted one bid,
`T-01` to `T-08`. Every bid is a configuration of the same platform family, comes with two priced options, and has
been tested by the bidder at its own works.

You are the Owner's engineer. Review the numerical clauses of the specification, establish how and where the
vehicle will really be used and what each bidder really tenders, evaluate the eight bids, write the factory
acceptance schedule of every bid you accept, and award the contract. A supplier takes its schedule literally.
Acceptance is at its own works, against your clauses and nothing else: it may deliver any unit of the delivery band
it has declared that passes every clause, and it delivers the one that suits it best. That unit is not trimmed
again. It goes straight into the reservoir.

## What you have (`/home/user/`)

| path | content |
|---|---|
| `tender/TS-01_revC.md` | the tender specification: Part A (the site duties, the twelve needs `N-01` to `N-12`, the Owner's policy), Part B (numerical clauses `B-..`), Part C (how the bids are evaluated), Part D (how the delivered unit is accepted) |
| `tender/evaluation_basis.py`, `tender/evaluation_basis.md` | the public calculation procedure of TS-01 and its description: the twelve needs, the closed dictionary of facility quantities, the density of pure water. It holds no number of the site, of any unit or of any catalogue page |
| `site/drawings/GA-01_plan.png`, `site/drawings/GA-02_section.png`, `site/drawings/RK-03_rack.png` | drawings of the intake tower: plan, section, trash rack |
| `site/reservoir_stage_daily.csv` | the daily record of the reservoir's gauge station |
| `site/unit_discharge_hourly.csv` | hourly discharge of the generating units |
| `site/ctd_casts/*.csv` | conductivity, temperature and depth casts at the monitoring stations |
| `site/secchi_log.csv` | Secchi-depth readings |
| `site/ops_notes.md` | the plant's operating notes |
| `site/prev_campaign/` | records of the previous inspection campaign, done with a rented vehicle: its data card (`vehicle_card.md`), the telemetry of three dives (`telemetry_dive05.csv`, `telemetry_dive07.csv`, `telemetry_dive09.csv`), an incident note (`incident_note.md`) and the hand-written dive log (`dive_log.png`) |
| `bids/T-01/` to `bids/T-08/` | each bid: `compliance_matrix.md` (Comply or Deviate on every clause of Part B), `technical_description.md`, `bom.md`, `price_sheet.md` (the base price and the two options), `delivery_statement.md` (the bidder's test facility, the units it may deliver, and its own order of preference among them) and the factory test reports `test_reports/TR-1.png` to `TR-5.png` |
| `catalogue/datasheets/DS-*.png`, `catalogue/errata/ER-*.png` | the makers' datasheets of what the bids supply, and the makers' errata and notices to them |
| `submission_contract.md`, `schemas/`, `schema_check.py` | the contract of your three files (their formats, every rule, how each scored item is judged), their JSON schemas, and a check of their structure |
| `labclient.py`, `desk_guide.md` | the client of the enquiry and witness desk, and its guide |

The drawings, the dive log, the test reports, the datasheets and the errata are images. Do not change the given
files; the grader reads its own copies.

## What you deliver (`/home/user/submission/`)

Three files, each at most 128 kB. `submission_contract.md` gives their formats and every rule.

- `evaluation.json`: for each of the eight bids, either `accept`, with the cheapest set of its priced options that
  makes it fit for the Owner's duty (the set may be empty) and your prediction of eight site values for that
  configuration, or `reject`, with one or two grounds: needs out of `N-01` to `N-12`, and the two codes `GF` and
  `PART-D` of TS-01 Part C. And the award: one bid that you accept, or none.
- `spec_review.json`: for every clause of Part B, one of `keep`, `lax`, `strict`, `lax_and_strict`. The contract
  defines the four labels.
- `acceptance_schedules.json`: one factory acceptance schedule for each bid you accept, written for the
  configuration you accept it with, at most 12 clauses each. A clause is one facility quantity of the dictionary,
  `>=` or `<=`, a bound, and the conditions of the test. Conditions are written out in full: nothing has a default.

The twelve needs are the needs that the Owner has put into numbers; a bid can be unfit for the duty at the site in
a way that none of them covers (TS-01 C.1): such a bid is rejected on the ground `GF`, or accepted only with the
options that make it fit, and `GF` cited against a bid of which a configuration is fit in general spoils the
rejection like any ground that does not hold.

A bid is acceptable only if an acceptance schedule that is sufficient can be contracted on it inside the facility
and the delivery band that its own delivery statement declares (TS-01 C.8): your schedule for a bid you accept
shows it. A bid that has configurations which meet the twelve needs and are fit in general, none of them
contractable, is rejected on the ground `PART-D`; cited against any other bid, `PART-D` spoils the rejection like
any ground that does not hold.

`/opt/venv/bin/python3 schema_check.py` checks the structure of the three files and nothing else. Only these three
files are read; keep your working files outside that folder.

## The enquiry and witness desk

Four members of the Owner's staff answer questions: `operations_engineer`, `civil_engineer`, `rov_pilot` and
`contracts_officer`. So do the eight bidders, `bidder_T01` to `bidder_T08`. A question is one of them and one topic
from a fixed list: ten topics for the Owner's staff, six for a bidder. You can also have a bidder run one of four
factory tests on its tendered unit, at its own works, with parameters you pick from short lists, and read the
result. `desk_guide.md` lists the topics, the tests, their parameters and their readings, and shows how to use
`labclient.py`.

- You have **14 requests** in total, questions and witnessed tests together. Every accepted request
  costs one, whatever the reply says. The same request again costs one again and returns the same reply.
- A request that goes to the wrong side (a bidder's topic or a test to the Owner's staff, an Owner's topic to a
  bidder) is accepted, costs one, and is answered only with a pointer to the other side.
- Nothing is refunded: not after a dropped connection, a timeout or a crash of your program once the request was
  accepted. A request the desk rejects is free (`desk_guide.md` lists what it rejects); so are `hello` and
  `budget`.
- A reply to a question is a short text in the person's own words, sometimes with one page attached as a PNG image.
  A witnessed test returns the readings of that test on the bidder's unit in the bidder's facility. No reply gives a
  verdict, and nothing about the Owner's site ever comes back from a test. When the budget is used up the desk
  answers `budget_exhausted`.
- The desk exists only during your session. Which requests you made is not graded; only the three files are.

## How it is scored

The grader holds the real world of this tender: the site as it really is, each bidder's unit and its parts as they
really are, and the facility and the delivery band each bidder has declared. Every margin it judges is computed
with the same `tender/evaluation_basis.py` that you have.

A **configuration** is a bid with a subset of its two options. For each configuration the grader computes the
margin of its tendered unit on each of the twelve needs. A configuration **clearly fails** a need when it misses it
by more than a small tolerance, which reflects the precision of the documents and replies from which the margin
can be established. A configuration is **fit** when it clearly fails no need, is fit in general (TS-01 C.1) and is
contractable (TS-01 C.8). Need N-01 has two named rules, one for the evaluation and one for the acceptance; the
contract says which applies where.

| item | weight | what is judged |
|---|---|---|
| `verdicts` | 0.35 | Two shares, multiplied. Of the bids that have a fit configuration: the share you accept with exactly the options of the cheapest fit one. Of the bids that have none: the share you reject on sound grounds. Grounds are sound when every ground you cite holds for the bid: a need that the bid clearly fails whichever options are bought, `GF` when every configuration of the bid is unfit in general, `PART-D` when the bid has configurations that clearly fail no need and are fit in general and none of them is contractable. One cited ground that does not hold spoils the rejection. |
| `award` | 0.25 | The configuration you award is fit, it is the cheapest of the configurations you accept, no fit configuration of any bid is cheaper, and the unit that the supplier would deliver under your schedule for it clearly fails no need at the site. A fit configuration that is a little dearer than the cheapest fit one earns part of the credit. Nothing if a cheaper bid that had to be rejected was not rejected by you on sound grounds. An award that is unfit, uncontractable or not the cheapest accepted earns nothing. |
| `site_prediction` | 0.10 | Your eight predicted site values, for each bid you accept with exactly the options of its cheapest fit configuration, each against the value the procedure gives on the real site. A value earns full credit inside a narrow band and none outside a wide one. The item counts only when `verdicts` is above zero, and it is multiplied by the reject share of `verdicts`. |
| `spec_review` | 0.05 | Your labels of the Part B clauses: the clauses whose true label is not `keep` and that you labelled correctly, minus the clauses to which you gave a wrong label other than `keep`, divided by the number of clauses whose true label is not `keep`; not below 0. The item counts only when `verdicts` is above zero. |
| `acceptance_schedules` | 0.25 | What your schedules let the suppliers deliver. For each bid that has a fit configuration and that you accept with exactly the options of its cheapest fit one, the grader takes every unit of the delivery band that the bidder has declared and puts each unit that passes all the clauses of your schedule for that bid into the real site, as delivered. Protection: of the needs that some unit of the band clearly fails, the share for which no passing unit clearly fails. Economy: of the units of the band that are clearly good, the share that still passes. The bid earns protection times economy, and the item is the mean over all the bids that have a fit configuration. The schedule must hold on whatever day the works test: at every water density inside the range the bidder declares. The item counts only when `verdicts` is above zero. It does not depend on the award. |

Your result is the weighted sum of the five items. `submission_contract.md` defines every term of the table.

Nothing in your sandbox scores an evaluation or a schedule against the real site or the real units. There is no
example world, `evaluation_basis.py` computes nothing without the conditions of the site and the facts of a unit,
and `schema_check.py` says nothing about any score.

## Rules that zero the result

The first two rules zero the whole result. Each of the others zeroes one item of the table, or one part of an item.

- **Structure (the whole result).** A file is missing, is not a regular file, is larger than 128 kB (131072 bytes),
  is not strict JSON, or breaks a structural rule of `submission_contract.md`. Among those rules: every one of the
  eight bids has exactly one verdict; the options of an accepted bid and of every schedule entry are options of
  that bid; a rejection cites one or two different grounds out of the 14 codes; the eight predictions are finite
  numbers; the award is one of the eight bid ids or `null`; the review covers every clause of Part B exactly once;
  no schedule has more than 12 clauses; every clause uses a quantity, a unit and a comparator of the dictionary and
  gives every condition. Which bids have a schedule entry is not a structural rule.
- **Submission integrity (the whole result).** Only `/home/user/submission/` reaches grading, as regular files;
  links are not carried over. Finish all writes before your session ends. A file changed after the session is
  rejected, and any process you leave running is stopped.
- **`verdicts` (0.35)** is 0 when you accept all eight bids, and when you reject all eight.
- **One rejection** counts 0 when a ground you cite clearly does not hold for that bid, be it a need, `GF` or
  `PART-D`, whatever else you cite; and when none of the grounds you cite holds.
- **`award` (0.25)** is 0 when you award to no bid, or to a bid you reject; when the awarded configuration is
  not fit: it clearly fails a need, is unfit in general, or is not contractable; when it is not the cheapest of the
  configurations you accept; when its price is 8 % or more above the price of the cheapest fit configuration
  of all bids; when a bid with a lower base price that had to be rejected is not rejected by you on sound grounds;
  when the unit that the supplier would deliver under your schedule for the awarded bid clearly fails a need at the
  site, at some water density that the bidder declares; when at some such density no unit of the awarded delivery
  band passes that schedule, so that nothing is delivered; when a clause of that schedule asks for a test outside
  the facility that the bidder has declared.
- **`site_prediction` (0.10), `spec_review` (0.05) and `acceptance_schedules` (0.25)** are 0 when
  `verdicts` is 0.
- **The schedule of one bid** earns nothing in `acceptance_schedules` when you do not accept the bid with exactly
  the options of its cheapest fit configuration; when at some water density that the bidder declares no unit of
  its delivery band passes; when a condition of a clause lies outside the facility that the bidder has declared;
  and when the schedule is empty or missing. A schedule entry for a bid you reject, or for other options than
  those you accept the bid with, is not read: the bid then has an empty schedule. If that bid is the one you
  award, the supplier delivers the unit it prefers of its whole band.

The grader reads your three files as data. It runs nothing you wrote.

## Your sandbox

- `/opt/venv/bin/python3`: Python 3.12 with NumPy, SciPy, pandas, Pillow, OpenCV (headless), PyYAML and jsonschema.
  There is no OCR engine.
- 4 CPUs, 8 GB of memory, no GPU and no network. The desk's Unix socket is the only
  service.

## Time

You have **4 hours** of wall-clock time from the start of your session (run `date` now). Whatever
is in `/home/user/submission/` when the session ends is what is graded.

A single response of yours is limited to 64k output tokens, and a response that reaches the limit is lost. Write
each deliverable in parts, put a first complete submission in place early, and refine it.
