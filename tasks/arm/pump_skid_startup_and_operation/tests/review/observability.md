# Are the instruments observable at the log sheet's tolerance? (v0.5, 2026-11-07)

The reference program reads the instruments perfectly (`tests/reference/controller.py`, `Eyes`), so it proves the
procedure can be done but not that a camera can read the skid. The evidence for the camera side is a camera-only
program: the submission of an Opus 5.5 (Claude Code, effort high) episode on v0.3 of this task, which had no access to
any ground truth (it found the devices with the overview camera's depth, read the tag plates, dials, the rotameter,
the level gauges and the analyser's display with the wrist camera, and matched the dial faces against the instrument
catalog). It was graded again, unchanged, by the v0.5 grader on the fixed hidden set (`tests/hidden/seeds.json`, 8
sites; tests/test.sh as root in a fresh container of the task image with only the submission restored). It logs one
round only (v0.3 had one), so each instrument has up to 8 counted entries.

Its counted log entries (made in a round while P-101 ran), the error against the instrument's true value at that
moment, in % of the instrument's span (AI-201 in mg/L):

| instrument | entries | within 1 % (v0.5 tolerance) | within 2 % | median error | largest error |
|---|---|---|---|---|---|
| PI-101 | 8 | 7 | 7 | 0.24 % | 2.22 % |
| PI-103 | 8 | 7 | 7 | 0.39 % | 2.38 % |
| PDI-111 | 7 | 7 | 7 | 0.27 % | 0.47 % |
| FI-105 | 6 | 6 | 6 | 0.36 % | 0.42 % |
| LG-201 | 8 | 8 | 8 | 0.64 % | 0.88 % |
| AI-201 | 8 | 8 | 8 | 0.002 mg/L | 0.004 mg/L |

43 of 45 entries are within 1 % of span (the tolerance is 1 % for the dials, the rotameter and LG-201, 0.05 mg/L for
AI-201); the two misses are the PI-101 and PI-103 readings of one episode the program voided anyway (its eighth). The
same program found and operated every valve and button of the line-up, the start and the shutdown on the eight sites
(start-up 0.08-0.20, shutdown 0.15 on all eight), and its events part was 0.125-0.25: the tag plates, the valve
positions, the lamps, the sight glasses and the drums are observable from the robot's cameras. Its three voids were
safety violations (the hand pressing a handwheel with more than 150 N), not perception failures.

Its v0.5 score: operation_score 0.4946 (episodes 0.000, 0.614, 0.857, 0.829, 0.860, 0.797, 0.000, 0.000), reward 0.516
at the v0.5 anchor; a reference number only (the program was written for v0.3, with its practice service).
