# Write the requirements for six replicas of a race car that can no longer be measured

A lab built one tenth-scale autonomous race car, the reference unit: a four-wheel-drive touring-class chassis kit
that carries an autonomy deck with a 2D scanning LiDAR, a compute module and their power electronics. The car has
left the lab on a long loan. A racing club has ordered six replicas of it for a use of its own: with the
club's sensor pod on the deck, in the regional indoor league and on the club's open days. A replica keeps the
rolling chassis kit, the deck plate and the layout of the reference unit; the parts of ten purchase lines are
bought anew from a catalogue of quoted parts.

What is left of the reference unit is a set of 33 photographs, the build log of the colleague who took them,
and the documents of the order. You are the requirements engineer. Recover the car from the photographs and the
documents, and deliver the requirements sheet, the bill of materials (BOM) and a kinematic sketch of the reference
unit. The grader holds the real car and judges your three files against it.

## What you have (`/home/user/`)

| path | content |
|---|---|
| `brief.md` | the club's brief (items `B-01` to `B-16`): the order, what is new in the club's use, how a replica is accepted, the lot policy, the price and the change fee. Its Appendix A is the acceptance, exact and public: the pass items with their bounds and conditions, the tracks and the open-day programme. Its Appendix D defines every quantity of the sketch and every field of the car record |
| `acceptance.json` | Appendix A in machine-readable form. Two places are left empty for you to fill: the lot table and the mass of the pod |
| `build_log.md` | the lab's build log of the reference unit (entries `BL-01` to `BL-10`): what the car is, which parts it was found with, how it was weighed, what each photograph shows and what was taken off for it, which scale references stand in which photograph, and how the pictures were archived |
| `photos/PH-01.jpg` to `photos/PH-33.jpg` | the photographs |
| `catalogue/quote.md` | the quote: every part you can order (SKU) with its purchase line, data page, fit and connector classes, arrival lot, quantity per car, pack size and price, and the fixed items of every car |
| `catalogue/datasheets/DS-*.png`, `catalogue/errata/ER-*.png` | the makers' datasheets, with outline drawings of the bought-in parts, and their errata |
| `catalogue/lot_reports/LR-*.png` | arrival reports of part lots |
| `procedure.py`, `quantities.md` | the public acceptance procedure and its description: the closed dictionary of quantities and every formula. The code holds no number of any car, part, site or use. The car enters through a car record that you fill in, the parts through a catalogue that you build from the pages |
| `stack_notes.md` | the constants of the driver model and of the steering model, and the exact unit definitions |
| `submission_contract.md`, `schemas/`, `schema_check.py` | the contract of your three files, their JSON schemas, and a check of their structure |

The photographs, the datasheets, the errata and the lot reports are images. Do not change the given files; the
grader reads its own copies.

## What you deliver (`/home/user/submission/`)

Three files, each at most 256 kB. `submission_contract.md` gives their formats, the closed lists of ids and every
rule.

- `requirements.yaml`: the requirements sheet. Up to 20 facets, the values the order rests on, each with its unit
  and the sources you read it from; and 7 derived values, your prediction of seven quantities of a replica
  built to your own BOM.
- `bom.json`: one quoted SKU for each of the ten purchase lines.
- `kinematic_sketch.json`: 33 quantities of the reference unit as it was photographed (heights, layout, steering
  geometry, chassis dimensions and tooth counts), each with a nominal value, a tolerance and its sources.

`/opt/venv/bin/python3 schema_check.py` checks the structure of the three files and nothing else. Only these three
files are read; keep your working files outside that folder.

## How it is scored

The grader holds the real car: the car record of the reference unit as it really is, the pod, and the lots as
they arrived. It fits every combination of quoted parts to that car and judges each against Appendix A with the
same `procedure.py`. A car is **acceptable** when it passes every pass item.

| item | weight | what is judged |
|---|---|---|
| `kinematic_replication` | 0.30 | Your sketch against the real car, quantity by quantity. A length earns full credit within 1.5 % of the true value or 1.0 mm, whichever is larger, and none beyond 6 % or 4.0 mm, whichever is larger; an angle earns full credit within 0.5 degrees and none beyond 3 degrees; in between the credit falls in a straight line. A tooth count must be equal. The quantities are scored in groups, and the groups weigh: heights 0.30, layout 0.25, steering 0.35, chassis 0.05, gears 0.05. The tolerances and the sources you state in the sketch are not scored. |
| `requirements_sheet` | 0.10 | Your facets against the reference sheet of the order. A facet counts when its value lies within the tolerance of the contract, its unit is the unit of the contract, and at least one of its sources is a source the value can be read from. The result is twice the number of facets that count, divided by the number of facets of the contract plus the number of facets you submit. |
| `derived_requirements` | 0.20 | Your seven predictions, each against the value the procedure gives for the real car with your BOM and the lots as they arrived. A prediction earns full credit within 3 % of that value and none beyond 15 %, in a straight line between. |
| `bom_acceptance` | 0.10 | Your BOM, fitted to the real car, is acceptable. All or nothing. |
| `bom_economy` | 0.30 | What your BOM costs the club. The grader takes the cheapest way from your BOM to an acceptable car: the price of an acceptable car plus one change fee of the brief (B-14) for every purchase line in which that car differs from your BOM. Full credit when this equals the price of the cheapest acceptable car, falling in a straight line to none at one change fee more. |

Your result is the weighted sum of the five items.

Nothing in your sandbox scores a file against the real car. There is no practice car and no example answer, and
`schema_check.py` says nothing about any score.

## Rules that zero the result

- **Structure.** A file is missing, is not a regular file, is larger than 256 kB, does not parse, or breaks a rule
  of `submission_contract.md`. Among those rules: every key of the closed lists is present where the contract asks
  for it, every number is finite, and every line of the BOM holds an SKU that the quote lists for that line.
- **Submission integrity.** Only `/home/user/submission/` reaches grading, as regular files; links are not carried
  over. Finish all writes before your session ends. A file changed after the session is rejected, and any process
  you leave running is stopped.

The grader reads your three files as data. It runs nothing you wrote.

## Your sandbox

- `/opt/venv/bin/python3`: Python 3.12 with NumPy, SciPy, Pillow, OpenCV (headless), PyYAML, jsonschema and
  matplotlib. There is no OCR engine.
- 4 CPUs, 8 GB of memory, no GPU and no network.

## Time

You have **4 hours** of wall-clock time from the start of your session (run `date` now). Whatever is in
`/home/user/submission/` when the session ends is what is graded.

A single response of yours is limited to 64k output tokens, and a response that reaches the limit is lost. Write
each deliverable in parts, put a first complete submission in place early, and refine it.
