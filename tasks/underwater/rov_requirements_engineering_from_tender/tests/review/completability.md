# Completability: the reference submission scores 1.000 from recoverable information

This file describes the task as this folder holds it (a hidden world, documents, desk tables, reference and schemas
that replace those of the form first merged on 2026-11-17, with a revised instruction and submission contract;
`README.md` says what of the agent's files is byte-equal to the first form's). Evidence entries are named by their id
in `evidence_index.json` (WORLD, LADDER, QC, DESK, WALK, SCAN); the index pins the sha256 of the task files they were
computed on except the payload of the agent's home, which it names only by the digest of its build tree (`README.md`).
Every score and count in sections 1 to 4 is a deterministic computation on the hidden world and the
formal grader, or a count made on the delivered files and on the verdicts of the readers' walk; none is a measurement
of a vehicle, and none comes from an agent episode of the task (the readers of the walk were AI review agents that
took no part in building the task: `fairness.md`, "The independent readers' walk"). Sections 5 and 6 also quote the
one frontier-model episode (how the desk served it, and its duration), and the verify times of section 6 are
wall-clock measurements of the runs named there.

## 1. The reference submission

`solution/submission/` holds the three files that the reference writer produced from the hidden world itself (a
privileged reference: it reads the world's truth, it does not solve the task from the documents): the evaluation of
the eight bids (four accepted, T-01, T-03, T-04 and T-05, each with its cheapest fit set of options and its eight site
predictions; four rejected, T-02 on PART-D, T-06 on N-03, T-07 on GF and T-08 on N-01 and GF, each ground one that
holds; the award T-03), the label of each of the 24 Part B template clauses, and one acceptance schedule for each
accepted bid (3, 3, 4 and 4 clauses). `solution/solve.sh` copies the three files into the submission folder and does
nothing else: it asks the desk nothing and reads nothing outside `solution/`.

- In the build that assembles this folder, the three files go through `solve.sh` into a fresh home and the task's own
  verify: score exactly 1.0 on every term (verdicts, award, site predictions, labels, schedules) and both gates
  passed; the untouched submission and an empty submission folder score 0.0 (QC).
- Through the engine (`ale validate`, twice), the reference's run scores reward 1.0 with every term 1.0 and the
  untouched run 0.0 (QC; the runs are listed in `evidence_index.json` beside its reports).

The reference is not the only full-score answer. The award-position bid's schedule written as an air-mass window
together with a displaced-volume window, both for the unit as built, scores 1.0 (LADDER); the same bid's schedule
written in displaced volume alone, or in air mass and displaced volume, scores 1.0, and so does a schedule written
from that bid's witnessed weighing alone, without its as-built reply (WALK). The delivered logs give every site
statistic a single value under every reasonable treatment (the design discharge over 288 treatments of the hourly log,
the clarity statistic over 60, the month's lowest water temperature over 72), so no reading of the logs moves a
score.

## 2. Where the information is

The agent's home holds 154 files: 77 page images (29 datasheets and 4 errata of the catalogue, 40 scanned factory
test reports, five per bid, three site drawings and the previous campaign's hand-kept dive log), 24 CSV logs (the
reservoir's daily stage, the plant's hourly discharge, a clarity log, 18 CTD casts at three stations and the previous
campaign's three telemetry files) and 53 text and code files (the tender TS-01 and its exact public procedure
`evaluation_basis.py`, the submission contract, the desk guide, the eight bids' compliance matrices, technical
descriptions, bills of materials, price sheets and delivery statements, the site notes). The enquiry and witness desk
answers 14 paid requests from a fixed table: 88 questions to the Owner's four members of staff and the eight bidders
and 160 witnessed tests at the bidders' works; four of the replies carry an attachment page. `setup.sh` checks each
of the 77 page images byte for byte before the agent starts.

Every printed number that refers to the world is registered with its source in the world (the fact registry of the
build: 5 167 printed numbers, none unresolved). The numeric scan then compares the 3 296 numerals the agent can read
(the text files, the logs' header lines, the strings drawn on the pages and attachments, the desk's texts) with the
1 008 hidden values. Where a numeral equals a hidden value in its digits, the numeral must carry its own registered
source, a quantity the documents print for a reason of their own: there are 1 666 such pairs of a numeral and a hidden
value, made of 17 distinct figures, and 1 525 of the pairs are one common round figure. Six more pairs are waived by
the scan's registered exceptions, each with its reason (they are among the numeric scan's 15 registered places in
`fairness.md`, "Nothing leaks to the agent"). Any other match of the same unit, with enough significant digits to
identify the hidden value, fails the scan, and there is none (SCAN).

## 3. What an expert must get right, and the route to each

The task is built around six considerations, each of which some bid's verdict, option set, ground or schedule hangs
on. Two of them (1 and 6) rest on a fact that no document carries and that only one holder at the desk gives; the
other four are read from the documents, with the desk where a document leaves a test condition or a reading open. For
each one the route below is the one the readers of the walk followed on the delivered files with the hidden truth in
hand (WALK), or that the reference's own requests take (section 4).

1. **The site as found.** The flow speed that enters the station-keeping need is the design discharge over the area
   OPEN to the flow (TS-01 A.3). The rack drawing RK-03 shows four bays of 3.0 m (12.0 m); the previous campaign's
   dive log and telemetry (November 2025) show a rack open to its toe, and they are true at their date. Since then
   the lowest bay was blocked in a flood of 2026-05-12, which the daily stage record shows (the largest daily rise of
   the whole record), and the Owner's previous pilot found it on a later drop-camera look (2026-06-16): bay 1 packed,
   the three bays above clear, 9.0 m open. Only that pilot's reply on the previous campaign (`rov_pilot`,
   `previous_campaign`, with his sketch attached) says so; no document, image, log, attachment or other reply carries
   it. Route: notice that the records of the site are older than the flood, ask the Owner's side about the previous
   campaign, and use the open height (9.0 m) everywhere it enters. With it the bid with the protective cage (T-06)
   fails the station-keeping need; with the drawn height it passes. T-06 is priced above the award, so its verdict
   does not move the award.
2. **The deck as a workplace.** TS-01 C.1 asks fitness for the duties at the site in the inspection month for
   everything a bid brings, each part within what its maker states. Route: list what stays out of the water at night
   (surface unit, reel and its interface, the vehicle before launch, its pack) from the bills of materials, the site
   plan and the vehicle card; read each item's operating row on its catalogue page against the night AIR of the
   month (the stage log's air columns, or the telemetry's enclosure temperature before launch), not the water; buy
   the option that cures an unfit item where one is offered, reject on GF where none is, and buy nothing where
   nothing is needed (C.6).
3. **A contract that can be executed.** The supplier may deliver any unit that passes the schedule (D.4), and a
   clause outside the works' test facility cannot be contracted (D.6). Route: read each accepted bid's delivery
   statement (its stock and the codes of its pack grades) against the pack maker's change notice; where an alternate
   cell lot equals the reference lot at and above the works' lowest chamber temperature and is weaker below it, and
   the site water is colder, no clause inside that works' facility separates the lots: that bid (T-02) is rejected on
   PART-D although its tendered unit is fit.
4. **One maker prints another resistance.** The procedure's pack key is the DC resistance after 60 s at load; the
   second maker's pages print an AC figure at 1 kHz under their own convention line. Route: notice the two labels
   (the page's own temperature table is labelled DC), have the pack discharged in a bidder's chamber (the witnessed
   reading is the key itself), and use the measured value: with it one bid's base pack holds the bus at its floor at
   the end of the dive, so the base configuration cannot hold station, and its option of the first maker's pack cures
   it; the other bid that carries a second-maker pack (another size of the same range) is fit with its own measured
   value, read at its own works or at any works whose bid carries that pack model.
5. **The works' day is not the Owner's.** A works weighs and trims in its own tank water, which the facility
   dictionary cannot state. Route: read each delivery statement's tank water and logged density range and the two
   dated weighings of the first test report; on a works whose tank is not fresh water fix the ballast state by
   clauses on air mass or displaced volume, which do not depend on the water; put a clause only where some unit of
   the delivery band fails a need, with its bound converted to the works' conditions (every thrust report states its
   bench supply and its battery current limit; one bid's trim water is asked at the desk); no clause on the tether
   length (no stock length fails at the site).
6. **The award-position bid as it is built.** After bids closed (18 September 2026) the works of T-03, the cheapest
   fit bid, fitted a float of the foam grade the vehicle already carries (0.40 L displaced and 0.24 kg in air with its
   cradle and fasteners) inside the frame and re-trimmed the unit (2026-09-24: three ballast pieces fixed in the
   float's cradle, four removable pieces fitted, three loose); every unit of its delivery band is built the same way.
   Its dated documents describe the unit as tested and stay true at their dates. Only the bidder says so: its reply
   on the as-built configuration (complete, with the works' re-weigh sheet attached), one line of its reply on the trim
   test conditions that points to it, and a witnessed weighing, which returns the unit as built. Route: the tender
   evaluates the tendered unit as it is built (TS-01 C.3) and makes what a bidder answers part of its bid (C.4); ask the
   award-position bidder for its as-built configuration (or witness its weighing and compare the mass with its first
   test report), and write that bid's ballast clauses for the unit as built. A schedule written for the unit as
   tested passes none of the clearly good units of the band as built: the purchase outcome is lost, and that bid's
   schedule loses its economy and at least half its protection.

Beside these stand the twelve needs of TS-01, the site statistics from the delivered logs (every reasonable treatment
gives the same value: section 1), the bid documents' own test conditions and tank waters, and the one bid whose cage
costs thrust and drag (its catalogue page and its technical description; its as-built reply attaches the caged unit's
bollard and tow figures).

## 4. The expert's requests fit the budget

The set of desk requests fixed with the hidden world, the reference's own plan, holds seven cells: one to the Owner's
side (the previous pilot), two witnessed pack discharges, one question on a bid's trim test conditions, two questions
on a bid's as-built configuration and one witnessed weighing:
`rov_pilot|ask:previous_campaign`,
`bidder_T01|witness:battery_discharge:chamber_c=23:load_w=100`,
`bidder_T05|witness:battery_discharge:chamber_c=23:load_w=100`, `bidder_T05|ask:trim_test_conditions`,
`bidder_T03|ask:as_built_configuration`, `bidder_T06|ask:as_built_configuration` and `bidder_T08|witness:weigh`.
Three of the four attachment pages come with these replies (the pilot's sketch, T-03's re-weigh sheet and the caged
unit's figures of T-06); the fourth comes with the civil engineer's reply on the inspection scope. A fuller plan that
a reader with the truth in hand walked on the real table (WALK: it also asks for the launch route, the operation
during the inspection and its season, the drawings' issue and the water quality, and has the two packs discharged at
10 degC and 440 W) costs twelve requests, every one served: two are left. The budget does not bind an expert.

## 5. The desk works on the real tables

The desk's tables were checked by the build's desk checker (17 checks, every planted fault caught) and by the build's
desk tests, which pass as registered (DESK). Among them: the table's registry is the build's fact registry, with its
four attachments; each of the two desk-only facts sits only in its holder's cells and attachment; the previous
pilot's reply is his dated look, and his closing remark points to his two topics and adds nothing; the award-position
bidder's as-built reply is complete and agrees with its witnessed weighing. In the one frontier-model episode the desk
ran in the task image and served the agent: 10 requests charged of 14, each answered from its own cell (the logged
digest of every reply equals the table's for its key), none refused, no key asked twice. The verifier never runs the
desk.

## 6. The time it takes and the worst-case verify

The agent reads 77 page images, four attachment pages, 24 CSV logs (about 3 MB) and the text files; the sandbox gives
4 CPUs, 8 GB and 14 400 s. The frontier-model episode ended by itself after 22.6 minutes of its 4 hours. The verifier's
worst case is bounded (QC; the limit's rule is in `task.toml`): through the engine's separate verifier, three runs
of the largest legal submission (three files of exactly 128 kB, 12 clauses on each of the four schedules the reference
reads) scored reward 1.0 with a verify command of at most 0.72 s and a whole verify phase (sandbox start, submission
restore, stage upload) of at most 6.84 s of the 600 s limit (one fifth: 120 s).
