# tests/review: completability and fairness evidence (grader-side; never staged into the agent's home)

The evidence that the task can be done from what the agent is given, that no shortcut scores, and that the grader is
robust and deterministic. The full reports live in the task's development folder, which is not published with the
benchmark; this folder records what they found, the numbers that carry each verdict, and the sha256 of every report
and of every task file the numbers were computed on, so a reader with the development repository can check that the
evidence still describes these bytes (`check_index.py`).

| file | what |
|---|---|
| `completability.md` | why the reference submission scores 1.000 from recoverable information: the information channels, the ten facts only one channel carries, the expert's six desk requests, an honest reader under reading noise, the desk on the real tables |
| `fairness.md` | why nothing but the work scores: shortcut auditors and robust no-ask policies frozen before scoring, degenerate submissions, single misreadings, mutants and benign rewrites, the leak and numeric scans, adversarial submissions, determinism, the desk's isolation |
| `evidence_index.json` | machine-readable index: every cited report (path in the development folder, sha256, bytes, package, verdict, the rows with thresholds and measured values) and the task files they were computed on |
| `check_index.py` | `python3 -I -B check_index.py [--dev DEV_FOLDER]`: recomputes the sha256 of the task files listed in the index (and of the reports when the development folder is given) and lists every one that changed since the evidence was taken |

All numbers are deterministic computations of the frozen world, the formal grader and the development lane's
reference pipelines; none is a measurement of a model or of a vehicle, and no model was run to produce them. The
world's numbers are fictional (`tests/NOTICE.md`).

## Verdicts at a glance

| suite (package) | report in the development folder | verdict |
|---|---|---|
| QC-1: shortcut auditors, robust policies, single misreadings, mutants, benign, clause-table dummies, I-CONS (P24) | `evidence/qc/QC_REPORT.md` | pass, every row (rerun on the integrated task; only the input pins changed, amendment A-2) |
| QC-2: adversarial submissions, worst legal submission, determinism under five CPU settings (P25) | `evidence/qc/QC2_REPORT.md` | pass: every adversarial input is refused at G0 / G1 with no crash; with the array-length gate in `grade.py` the slowest of the 158 cases takes 0.355 s in the container (was 501.8 s) and through the engine the worst adversarial submission's verify phase takes 7.06 s of the 600 s limit (was 514.9 s); 919 submissions score byte-identically with and without the gate |
| Scans: leak words, routing words, numbers against the fact registry, examples, public code literals, digest literals, PNG metadata, schema and dictionary names (P26) | `scan/SCAN_REPORT.md`, `scan/report.json` | pass (0 unexempted hits, 25 / 25 planted controls caught, 0 consistency defects: the public procedure's policy literals equal TS-01 Part A) |
| Zero-cost difficulty evidence: knock-outs per channel, ask policies, robust no-ask policies, population model, honest noise (P27) | `evidence/difficulty/ZERO_COST.md` | pass, 17 / 17 rows (rerun on the integrated task, amendment A-3: input pins only) |
| The desk on the real tables in a real sandbox (P28) | `evidence/lab/SMOKE_REAL.md` | pass: 77 / 77 checks in the separate verifier, rerun with the formal payload (`validate-5a9ec14b`) |
| Difficulty readout and the frozen probe pre-registration (P29) | `evidence/difficulty/READOUT.md`, `evidence/difficulty/PROBE_PREREG.md` | done; the pre-registration was amended once (input pins only, rules unchanged); risk recorded and accepted: the worst-case reading time estimate is 72.5 % of the agent's 4 h, to be measured by the first model run |

## Integration (P30 and the closing lane, 2026-11-15)

- P30 found three changes missing from this folder (the revised public procedure and its pin, the digest copy
  `tests/digest/spec_values.json`, the array-length gate); without them `ale validate` ended in a task error. The
  closing lane landed them together with the other rulings of the development lane (the minimum water temperature of
  the inspection month at the value the shipped CTD casts give, 5.1 degC, with the world frozen again; the page-image
  check in `setup.sh`; the payload rebuilt with the formal build command) and reran every suite above on the result.
- On this folder as it stands: `ale lint` and the domain lint pass; the repository's pytest suite passes with it in
  place; the agent-visible files carry 0 hits of the shared leak-word list (78 text files, 74 PNG files and their
  148 non-pixel chunks, 4 339 drawn strings, every string of the desk table); no byte code anywhere; `ale validate`
  gives untouched 0.0 and reference 1.0. Through the engine's separate verifier the worst legal submission and the
  two worst adversarial ones take a few seconds of the 600 s limit (one fifth: 120 s).
  `evidence_index.json` `gate_v` lists the runs and the times; `check_index.py` lists every cited file whose bytes
  changed since.
