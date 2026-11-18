# Fairness: nothing but the work scores, and the grader cannot be gamed or crashed

Sources (paths in the task's development folder; sha256 in `evidence_index.json`): QC-1 `evidence/qc/QC_REPORT.md`
(P24), QC-2 `evidence/qc/QC2_REPORT.md` (P25), the scans `scan/SCAN_REPORT.md` and `scan/report.json` (P26), the
zero-cost evidence `evidence/difficulty/ZERO_COST.md` (P27), the desk retest `evidence/lab/SMOKE_REAL.md` (P28).
Every number is a deterministic computation on the frozen world and the formal grader, both tolerance scales
(x 1.0 / x 1.5) unless one is named; none is a model measurement. Every suite was rerun on this folder after the
closing lane's changes (the world frozen again with the inspection month's minimum water temperature at 5.1 degC, the
revised procedure, the array-length gate, the payload rebuilt with the formal build command); the numbers below are
the reruns'.

## 1. Shortcuts score low (rules frozen before any score)

The auditors' rules were written and hashed before the first score file (QC-1: frozen 09:29:25, first score 09:31:01;
ZERO_COST: rules frozen 10:29:49 before the first output). The reruns changed only the pinned input hashes
(amendments A-2 and A-3), not a rule. The auditors read the visible layer rebuilt from the rendered documents (1 514
values compared with the printed layer, 0 differences).

| family | threshold | measured |
|---|---|---|
| shortcut auditors with grounds fixed in advance (incl. the public-platform common-sense auditors, 266 Wh per pack) | <= 0.15 | max 0.1020 |
| every other shortcut criterion (price splits with triage grounds, 512 switch settings, 12 named auditors, the medoid, "any deviation is honest") | <= 0.30 | max 0.2194 |
| triage grounds on the cheaper-than-award rejections (94 rules) | at most floor(4 / 2) right | 2 / 4 |
| accept-all, reject-all with any ground, the compliance matrix taken at its word, blanket margins (234 groups) | 0.000 | 0.000000 |
| 15 degenerate submissions | 0.000 | 0.000000 |
| robust policies that ask nobody (R1-R3, R1rev x three efficiency readings x two flow readings; blanket margin) | <= 0.40 | max 0.2417 |
| the same policies with the owner's six fact cells | <= 0.55 | max 0.4673 (R1rev; its margin of 0.083 rests on the harmless late revisions of T-04 and T-07) |
| documents only, no desk, default assumptions | <= 0.45 | 0.160 |
| default readings everywhere, no desk (8 groups) | <= 0.10 | 0.000 |
| seven ask-in-order policies x three assumptions | <= expert - 0.20 = 0.800 | max 0.3167 |

## 2. Each reading error costs, and the costs compose

Sixteen planted single misreadings (each the one error, everything else right): each <= 0.850 (max 0.850 = the
structural 1 - (0.45 + 0.15) / 4), mean 0.4631 (threshold 0.60); the conservative misreading of the flow statistic
scores 0.000. Fifteen of the sixteen flip a decision outside the tolerance on both scales. A population of readers
that each misread each of 14 quantities independently with p = 0.25 averages 0.1715 (sd 0.195; 55 296
combinations). Each of the ten single-channel facts costs at least 0.1333 when its channel is knocked out, each of
the four channel groups at least 0.4333 (`completability.md` sec. 2).

## 3. The grader scores what is right and only that

- 38 mutants of the reference submission (a wrong verdict, a wrong or extra ground, a missing or needless option, a
  site value moved out of tolerance, a Part B label flipped, a clause deleted, loosened, reversed or put under the
  wrong condition) each score exactly the value the sets and weights predict; 17 benign rewrites and 4 536
  alternative clause tables score 1.000 (`completability.md` sec. 1).
- The two N-01 rules are named and used where TS-01 says: swapping them in the grader drops the reference to 0.530;
  T-05 as tendered passes the Part C rule (+5.36 tolerances) and fails the Part D rule (-21.01).
- Sixteen clause-table dummies (empty, the coastal template, the site window as the tank window, limiter off, a
  bound at the band's edge, ...) score the protection they buy: 0 to 2/3 of F, the reference and one-unit shifts 1.

## 4. Nothing leaks to the agent

- The shared leak-word list over everything the agent can read (instruction, 78 text files, 74 PNG files and their
  148 non-pixel chunks, 4 339 drawn strings, every string of the desk table): **0 hits** (rerun on this folder; a
  planted platform name is caught).
- The task's eight scans (P26; rerun on this folder with the rebuilt payload, `task.toml`, `NOTICE.md` and this
  folder in place): 0 unexempted hits; 14 registered exemptions covering 381 places, each with its reason; routing
  words outside the vehicle card 0 (no exemption allowed); example numbers 0 same-key and 0 cross-key hits against
  the reference clause bounds; every visible number registered (4 478 numeric tokens, 0 unregistered); public code
  holds no hidden or catalogue number; PNGs carry only pixel chunks; 25 planted defects caught; 0 consistency
  defects (the three policy literals of the shipped procedure now equal TS-01 Part A).
- Platform, maker, part and author names appear only on the grader's side: `tests/NOTICE.md`, and
  `tests/digest/spec_values.json`, a byte-for-byte copy of the frozen digest that the grader pins (the scan checks
  that it is identical to the digest and lists its 157 hits).
- `setup.sh` checks every one of the 71 page images in the agent's home byte for byte against
  `environment/setup/page_images.sha256` (root-only, never in the agent's home) and fails when one is missing or
  differs.

## 5. Adversarial submissions and determinism

- 158 adversarial and boundary cases (P25, rerun on this folder), each in a fresh container of the task image as
  root through `test.sh`: 106 refused (101 at G0; 5 at G1: links, a directory, a FIFO, a late write, a linked
  submission folder), 51 legal edge cases scored bit-identical to the host, a tampered world a TASK ERROR;
  0 tracebacks, 0 surviving agent processes (a self-respawning loop: 65 killed).
- **Closed (F1)**: an unsortable array checked for uniqueness by the schema validator is quadratic; before the fix
  two such arrays of 128 kB took 501.8 s in the container and 514.9 s through the engine's separate verifier
  against the 600 s limit. `grade.py` now refuses any array longer than the schemas allow (24 items) after strict
  JSON parsing and before the schema validator (G0), and refuses to load a stage whose schemas would switch that
  gate off. The slowest of the 158 cases now takes 0.355 s; every verdict, reward and exit code equals the
  demonstrated fix's, and the old and new `grade.py` give the 919 submissions byte-identical values, rewards and
  metrics. Through the engine the two worst adversarial submissions' verify phases take 7.06 s and 5.73 s.
- The worst legal submission (24 clauses, every one on a new dimension of conditions, three files of exactly
  128 kB) scores 1.0 in 0.317 s in the container and 6.15 s through the engine (one fifth of the limit: 120 s).
- Determinism: 919 submissions under five CPU settings (native; hash seed 0; 8 BLAS threads; the 512-bit vector paths off;
  the 256-bit vector and fused multiply-add paths off as well) give bit-identical values, rewards and metrics, and so do the recomputed truth table, delivery bands, site
  truth and the 366 432-value dictionary sweep; 14 full root verifies x 5 settings give identical verdicts.
- The engine copies a sparse artifact as dense data (512 MiB took 26.6 s): an engine-side limit, to be raised with the engine (no task-side switch).

## 6. The desk cannot be read or replayed

The hidden tables sit in a root-only folder closed twice (0700 and `chmod -R go-rwx`); the lab worker drops to its own
account with seccomp allowing only Unix sockets; refusals depend only on public constants and are identical with a
full budget, after a restart and at zero; each request is charged once, wrong-side requests included; the replies are
fixed bytes, not a simulator (P10, P28). The verifier never runs the desk and reads nothing the agent's sandbox left.

## 7. Known limits

- The two facts without a document trace (U7, U8') are kept by default (ruling N3); DESIGN sec. 15 item 3 asks the
  owner to confirm.
- A single witness of T-08 sits on the cliff: bollard only lets T-08 through (+0.02 tolerances, total 0.492), tow
  only rejects it (-0.65); asking for its as-built configuration or doing both witnesses is decisive (P27). The
  design now says so (without asking the bidder, both witnesses are needed); the world is unchanged.
- The owner-cells robust maximum (0.4673 against 0.55) depends on the two harmless late revisions; any change to the
  revisions of T-04 or T-07 re-runs QC-1 and the ask policies.
- The spec-review labels (Q, weight 0.05) can be guessed to 0.769 from the documents alone, above the design's
  0.45 for that part; it lifts a total by at most 0.038 (`tests/NOTICE.md`). With those labels swapped in, each of
  the 35 shortcut-auditor rows of QC-1 sec. 3 still stays within its threshold (closest: 0.1328 against 0.15).
  Accepted; the clause plan was not redone.
