# Evaluate eight bids for an inspection-class ROV and write its acceptance schedule

A hydro-power company, the Owner, is buying one inspection-class remotely operated vehicle (ROV) for the yearly
inspection of the intake tower of a high, cold, deep fresh-water reservoir: the trash rack bay by bay, the sill of
the gate slot and the sump, the dam face on both sides of the intake, and samples of debris taken off the rack with
the gripper. The Owner has issued tender specification TS-01 rev C. Eight integrators have each submitted one bid,
`T-01` to `T-08`. Every bid is a configuration of the same platform family, comes with two priced options, and has
been tested by the bidder at its own works.

You are the Owner's engineer. Review the numerical clauses of the specification, establish how and where the
vehicle will really be used and what each bidder really tenders, evaluate the eight bids, award the contract, and
write the factory acceptance schedule for the unit the Owner buys. The supplier takes that schedule literally.
Acceptance is at its own works, against your clauses and nothing else: it may deliver any unit of the delivery band
it has declared that passes every clause, and it delivers the one that suits it best. That unit is not trimmed
again. It goes straight into the reservoir.

## What you have (`/home/user/`)

| path | content |
|---|---|
| `tender/TS-01_revC.md` | the tender specification: Part A (the site duties, the twelve needs `N-01` to `N-12`, the Owner's policy), Part B (numerical clauses `B-..`), Part C (how the bids are evaluated), Part D (how the delivered unit is accepted) |
| `tender/evaluation_basis.py`, `tender/evaluation_basis.md` | the public calculation procedure of TS-01 and its description: the twelve needs, the closed dictionary of facility quantities, the density of pure water. It holds no number of the site, of any unit or of any catalogue page |
| `site/drawings/GA-01_plan.png`, `site/drawings/GA-02_section.png`, `site/drawings/RK-03_rack.png` | drawings of the intake tower: plan, section, trash rack |
| `site/reservoir_stage_daily.csv` | daily readings of the reservoir stage gauge |
| `site/unit_discharge_hourly.csv` | hourly discharge of the generating units |
| `site/ctd_casts/*.csv` | conductivity, temperature and depth casts at the monitoring stations |
| `site/secchi_log.csv` | Secchi-depth readings |
| `site/ops_notes.md` | the plant's operating notes |
| `site/prev_campaign/` | records of the previous inspection campaign, done with a rented vehicle: its data card (`vehicle_card.md`), the telemetry of three dives (`telemetry_dive05.csv`, `telemetry_dive07.csv`, `telemetry_dive09.csv`), an incident note (`incident_note.md`) and the hand-written dive log (`dive_log.png`) |
| `bids/T-01/` to `bids/T-08/` | each bid: `compliance_matrix.md` (Comply or Deviate on every clause of Part B), `technical_description.md`, `bom.md`, `price_sheet.md` (the base price and the two options), `delivery_statement.md` (the bidder's test facility, the units it may deliver, and its own order of preference among them) and the factory test reports `test_reports/TR-1.png` to `TR-5.png` |
| `catalogue/datasheets/DS-*.png`, `catalogue/errata/ER-*.png` | the makers' datasheets of the parts the bids are built from, and their errata |
| `submission_contract.md`, `schemas/`, `schema_check.py` | the contract of your three files (their formats, every rule, how each scored item is judged), their JSON schemas, and a check of their structure |
| `labclient.py`, `desk_guide.md` | the client of the enquiry and witness desk, and its guide |

The drawings, the dive log, the test reports, the datasheets and the errata are images. Do not change the given
files; the grader reads its own copies.

## What you deliver (`/home/user/submission/`)

Three files, each at most 128 kB. `submission_contract.md` gives their formats and every rule.

- `evaluation.json`: for each of the eight bids, either `accept`, with the cheapest set of its priced options that
  makes it fit for the Owner's duty (the set may be empty) and your prediction of eight site values for that
  configuration, or `reject`, with one or two of the twelve needs as grounds. And the award: one bid that you
  accept, or none.
- `spec_review.json`: for every clause of Part B, one of `keep`, `lax`, `strict`, `lax_and_strict`. The contract
  defines the four labels.
- `acceptance_schedule.json`: the factory acceptance schedule for the configuration you award, at most 24 clauses.
  A clause is one facility quantity of the dictionary, `>=` or `<=`, a bound, and the conditions of the test.
  Conditions are written out in full: nothing has a default.

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
  accepted. A request the desk rejects (an unknown id, topic, test or parameter value, a malformed request) is
  free; so are `hello` and `budget`.
- A reply to a question is a short text in the person's own words, sometimes with one page attached as a PNG image.
  A witnessed test returns the readings of that test on the bidder's unit in the bidder's facility. No reply gives a
  verdict, and nothing about the Owner's site ever comes back from a test. When the budget is used up the desk
  answers `budget_exhausted`.
- The desk exists only during your session. Which requests you made is not graded; only the three files are.

## How it is scored

The grader holds the real world of this tender: the site as it really is, each bidder's unit as it really is, the
makers' page values as corrected by their errata, and the delivery band each bidder has declared. Everything it
judges is computed with the same `tender/evaluation_basis.py` that you have.

A **configuration** is a bid with a subset of its two options. For each configuration the grader computes the
margin of its tendered unit on each of the twelve needs. A configuration **clearly fails** a need when it misses it
by more than a small tolerance, which reflects the precision of the documents and replies from which the margin
can be established. A configuration is **fit** when it clearly fails no need. Need N-01 has two named rules, one
for the evaluation and one for the acceptance; the contract says which applies where.

| item | weight | what is judged |
|---|---|---|
| `verdicts` | 0.45 | Two shares, multiplied. Of the bids that have a fit configuration: the share you accept with exactly the options of the cheapest fit one. Of the bids that have none: the share you reject on sound grounds. Grounds are sound when every need you cite is one that the bid clearly fails whichever options are bought; one cited need that does not hold spoils the rejection. |
| `award` | 0.15 | The configuration you award is fit, it is the cheapest of the configurations you accept, and no fit configuration of any bid is cheaper. A fit configuration that is a little dearer than the cheapest fit one earns part of the credit. Nothing if a cheaper bid that had to be rejected was not rejected by you on sound grounds. |
| `site_prediction` | 0.15 | Your eight predicted site values, for each bid you accept with exactly the options of its cheapest fit configuration, each against the value the procedure gives on the real site. A value earns full credit inside a narrow band and none outside a wide one. The item counts only when `verdicts` is above zero, and it is multiplied by the reject share of `verdicts`. |
| `spec_review` | 0.05 | Your labels of the Part B clauses: the clauses whose true label is not `keep` and that you labelled correctly, minus the clauses to which you gave a wrong label other than `keep`, divided by the number of clauses whose true label is not `keep`; not below 0. The item counts only when `verdicts` is above zero. |
| `acceptance_schedule` | 0.20 | What your schedule lets the supplier deliver. The grader takes every unit of the delivery band that the awarded bidder has declared and puts each unit that passes all your clauses into the real site, as delivered. Protection: of the needs that some unit of the band clearly fails, the share for which no passing unit clearly fails. Economy: of the units of the band that are clearly good, the share that still passes. The item is protection times economy. It is zero if no unit of the band passes, if a clause asks for a test outside the facility the bidder has declared, or if the configuration you award is not fit or is not the cheapest of the configurations you accept. |

Your result is the weighted sum of the five items. `submission_contract.md` defines every term of the table.

Nothing in your sandbox scores an evaluation or a schedule against the real site or the real units. There is no
example world, `evaluation_basis.py` computes nothing without the conditions of the site and the facts of a unit,
and `schema_check.py` says nothing about any score.

## Rules that zero the result

- **Structure.** A file is missing, is not a regular file, is larger than 128 kB, does not parse, or breaks a rule
  of `submission_contract.md`. Among those rules: every one of the eight bids has exactly one verdict; the options
  of an accepted bid are options of that bid; the eight predictions are finite numbers; the award is a bid you
  accept, or none; the review covers every clause of Part B exactly once; the schedule is written for the award;
  every clause uses a quantity, a unit and a comparator of the dictionary and gives every condition.
- **Submission integrity.** Only `/home/user/submission/` reaches grading, as regular files; links are not carried
  over. Finish all writes before your session ends. A file changed after the session is rejected, and any process
  you leave running is stopped.

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
