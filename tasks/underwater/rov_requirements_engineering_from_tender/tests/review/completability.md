# Completability: the reference submission scores 1.000 from recoverable information

Sources (paths in the task's development folder; sha256 in `evidence_index.json`): QC-1 `evidence/qc/QC_REPORT.md`
(P24), zero-cost evidence `evidence/difficulty/ZERO_COST.md` (P27), the desk on the real tables
`evidence/lab/SMOKE_REAL.md` (P28), the difficulty readout `evidence/difficulty/READOUT.md` (P29), the scans
`scan/SCAN_REPORT.md` (P26). World: the frozen record world of `world/selection.json`. Every number below is a
deterministic computation, not a model or vehicle measurement.

## 1. The reference submission

`solution/submission/` holds the three files written from the frozen world: the evaluation (four bids accepted with
their cheapest fit option subsets and site predictions, four rejected on the ground that holds, the award T-05), the
24 template-clause labels, and the four-clause acceptance schedule for T-05's delivery band. Under the formal grader
it scores **1.000000** with the tolerance scale x 1.0 and x 1.5 (QC-1 sec. 1), and in the engine's separate verifier
the reference episode scores reward **1.0** (Gate V, `evidence_index.json` `gate_v`). `solution/solve.sh` only copies
the three files: it asks the desk nothing and reads nothing outside `solution/`.

The schedule is not the only full-score answer: 4 536 alternative clause tables built from the separable facility
quantities of the band (every widest gap, every variant at fixed steps) each score 1.000 as the reference's
schedule (QC-1 sec. 8, I-ALT), and 17 benign rewrites of the reference (key order, site values moved by 0.99 of a
tolerance, bounds moved one printed unit, implied or harmless extra clauses) all score 1.000.

## 2. Where the information is

An engineer has four channel groups: the site records (drawings, level, discharge, CTD, clarity logs, operating
notes), the previous campaign's records (vehicle card, telemetry, incident note, dive log), the tender documents
(TS-01, the procedure, eight bids, the catalogue) and the enquiry and witness desk (14 paid requests). Every number
the agent can read is registered with its source: 4 478 numeric tokens over 149 visible files and 250 desk texts,
0 unregistered (the payload rebuilt with the formal build command; P26 numeric scan). The public procedure `evaluation_basis.py` is exact and is the grader's own pinned copy.

Ten decisive facts are carried by one channel only; removing that channel's link and letting the expert re-route
costs (ZERO_COST sec. 3.1, knock-out K1, default assumptions):

| id | channel | fact | drop |
|---|---|---|---|
| U1 | discharge log | plant discharge in the inspection-month permit window | 1.0000 |
| U2 | CTD | water temperature and density at the working depth | 0.6493 |
| U3 | clarity log | inspection-month visibility at the intake station | 0.5286 |
| U4 | previous campaign | launch from the tower-deck davit | 0.5179 |
| U5 | test reports | T-03's as-built camera housing (TR-3) | 0.5000 |
| U6 | delivery statement | delivery band, facility envelope, ease order | 0.1333 |
| U7 | desk, owner side | two bottom bays of the rack blocked (asked of the previous campaign's pilot) | 0.6912 |
| U8' | desk, bidder side or witness | T-08 as tendered (its as-built configuration, or the bollard and tow witnesses) | 0.5077 |
| U9 | witness | T-04's trim-tank water (weigh witness) | 0.5115 |
| U10 | previous campaign | the tether's lead across the platform (vehicle card and payout counter) | 0.5179 |

Removing a whole channel group costs between 0.4333 and 1.000 (ZERO_COST sec. 3.2). Two of the facts (U7 and the
T-08 cage in U8') have no trace in the documents and are learnt only by asking: asking the previous operator about
site changes and every bidder for its as-built configuration is the ordinary due diligence the desk's public topic
list offers (ruling N3; DESIGN sec. 15 item 3 keeps the default and asks the owner).

## 3. The expert's route fits the budget

The exact minimum set of desk requests that carries every fact without a free document link has **6 cells** within
the budget of 14 (ZERO_COST C7): `bidder_T01|ask:thrust_test_conditions`, `bidder_T01|ask:trim_test_conditions`,
`bidder_T04|witness:weigh`, `bidder_T05|ask:thrust_test_conditions`, `bidder_T08|ask:as_built_configuration`,
`rov_pilot|ask:previous_campaign`. Documents plus those six cells score **1.000** (Interact; Full 1.000).

An honest reader with reading noise (printed resolution and instrument tolerance on every value read, site statistics
drawn from the pool of reasonable log treatments) asking the same six cells, 50 fixed seeds (ZERO_COST I5): the
decision vector equals the noise-free one and the truth in 50 / 50, the option subsets and grounds in 50 / 50;
the total has median **1.0000** and minimum **1.0000** both on the calibration pool and on the logs generated from
the frozen site (every reasonable log treatment gives the world's design flow, clarity statistic and inspection-month
minimum water temperature, 5.1 degC). On the calibration pool, 200 seeds and the x 1.5 scale give the same 1.000.
A careful reader who works only from the documents and the six replies recomputes all 32 configurations x 12 needs
without one sign change against the truth (largest deviation 0.14 tolerances), every site prediction inside its
tolerance (largest 0.052 of it), the same verdicts, option subsets, grounds and award, and the 24 clause labels.

## 4. The desk works on the real tables

In a real sandbox with the shipped image, setup, plugin, configuration, payload and both real tables (P28,
`validate-bcaf30c6`; rerun with the formal payload and the page-image check of the shipped setup,
`validate-5a9ec14b`, with the same results): all 248 cells (40 owner-side, 48 bidder-side, 160 witness) and the two redirect texts reach
the agent byte for byte as registered, 392 witness values bit for bit; refusals are free and identical with a full
budget, after a restart and at zero; the budget is charged atomically (24 concurrent requests for 11 remaining: exactly
11 served); all 184 wrong-side requests are charged and answered with the redirect text; the shipped limits hold
(900 s idle close measured at 900.1 s, 2 000 requests / s, 1 MiB line) and none of them charges; the agent account
cannot read the hidden tables (40 attempts EACCES, ptrace EPERM). The separate verifier's 77 checks pass.

## 5. The time it takes (a risk, not a gate)

The agent reads 71 page images (1.81 MP each), 3 desk attachments, 24 CSV logs (3.06 MB, 56 836 lines) and about
220 000 characters of text. The development lane's worst-case estimate of image transcription plus log analysis is
10 437 s, 72.5 % of the 14 400 s limit (READOUT sec. 6: six model calls per page at the per-call time of a recorded
requirements episode of another task); if each page reads at a glance, 5 884 s. No model has been run on this task;
the estimate is accepted as a recorded risk and the first calibrated episode will measure it. The limit stays at the
published requirements tasks' 14 400 s.
