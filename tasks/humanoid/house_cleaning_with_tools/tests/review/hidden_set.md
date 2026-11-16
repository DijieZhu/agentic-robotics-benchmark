# The fixed hidden scene set (tests/hidden/seeds.json)

**Amendment 1 (2026-11-08):** the cell and the reference changed, the selection was repeated under the same rule, and
the hidden set became candidates 1, 2 and 3 (anchor 0.9352). **Amendment 2 (2026-11-09):** the reference's arm
postures and the robot state changed; the selection was repeated under the same rule (candidates 2, 3 and 4, anchor
0.9552). **Amendment 3 (2026-11-09, at the end):** the reference's arm postures (round 4) and the agent cameras' look
changed; the selection is repeated with one more rule: the kept scenes may not all share a layout fact a program could
assume. **Amendment 4 (2026-11-11):** the reference's postures (round 5) changed; the selection is repeated under
amendment 3's rule (its first run set aside for two planner mistakes, fixed; the repeated run keeps candidates 1, 2
and 3 again, anchor 0.9306); an addendum (2026-11-12) draws the bins open, the physics unchanged. The sections before
each amendment record the earlier selections.

The grader runs the three scenes of `tests/hidden/seeds.json`, for agents and the oracle alike (verify.py
`episode_seeds`; the oracle's privileged cells get the same layouts, not a nominal one). A scene is the cell's draw
from its seed (house_cell/layout.py: the objects, their facings, the grease stain, the crumbs, which bin stands where,
the robot's start pose; the sensor noise), by the rules cell_manual.md section 2 states.

## Selection rule (written before any candidate ran)

1. The oracle is frozen first: the commit and the sha256 of every file of `tests/house_cell/` and `solution/` are
   recorded below.
2. 16 candidate seeds are drawn from `os.urandom` (63 bits each, `tools/draw_candidates.py`), in one call; they are
   listed below in drawn order, before any of them runs.
3. Each candidate runs once: the privileged reference exactly as the oracle's episode does it (`tools/run_ref.py`: a
   fresh privileged `CellServer` on the seed, the oracle's four camera frames, `reference_clean`, `finalize`, scored by
   `house_cell/scoring.py` with the damage charges), in the task image, one process per candidate, with tests/test.sh's
   numeric environment (`tools/launch_ref.sh`).
4. A candidate passes when its episode has no safety violation and no damage (no broken dish, nothing of the
   customer's left on the floor), every part of the checklist but the floor has its full credit (trash 2, dishes 1,
   books 1, table 2.5, finishing 1), and at least half of the crumbs are in the general-waste bin. (The reference
   sweeps the crumb groups its broom and dustpan can reach in the open; the rest of the floor's credit is the margin
   above the anchor.)
5. The hidden set is the first three passing candidates in drawn order. If fewer than three of the 16 pass, 16 more
   are drawn the same way and the rule goes on in their order.
6. Every candidate's result is recorded below, kept or not, with a hollowing check: the kept scenes' layout features
   (`tools/features.py`) against all candidates'.

The anchor (`tests/anchor.json`) is then one full verify of the oracle submission on the kept set (`ale validate`'s
oracle pass), and `full_at` is sized from the difference between that verify and the selection runs.

## The frozen oracle

Commit `9213936` (house_cleaning_with_tools: maintainer fixes and a harder, zero-shot version). sha256 of the files the
oracle's episodes run (the cell, the reference cleaner, the oracle submission, the grader's configuration):

```
2f93e5e0db358d838599fadc7631b3cc3df63da4c65ba3f96bdd4961a780215e  tests/house_cell/cell.py
d88e94a34316fa9d9590eedf4e40b7f35ed0b893a0672c08e890f7854f1f762c  tests/house_cell/cleaning.py
2c44d49074a8b66ed548ba38dcb884478d9234187a27bdf943aebcac00637561  tests/house_cell/control.py
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  tests/house_cell/__init__.py
93c439787b73108e64125326aa47a21fbc0cd3009e59a1b9a88a1f56e536f8d1  tests/house_cell/layout.py
2814c5419023f70edc9e9507029be8583da7475138b9cc5da0fbd8af648ac272  tests/house_cell/objects.py
108048c18b6c0ae81da230a6b5e8dd21e4222391a24ce5351bd726ba4f9f6e7b  tests/house_cell/physics.py
03880c6204b6bdd7e3412e93a7b9d85ab501dfe84e6c2b9efce2f3fb44ce0c49  tests/house_cell/props.py
c8518d22141b7f1a7c38ae3fa1dbc44ffb370e2f5a970c8f89322846aa1537c6  tests/house_cell/reference.py
fa6232f3431683ebb618a68bd202aec66fcc766d66e61007fad8992e89b65581  tests/house_cell/render.py
223b284fd0ce69c6ed3f6b0d7f2348d67e0bcf150acdd70e567c8e6e0e7198e8  tests/house_cell/robot.py
59015301eb370fdd8e965d65a26c4ae01af0b7b32b5b9821094399a43a1ff1ed  tests/house_cell/room.py
3afb3e781ab4649602be28b0a3985f18009b450f11b1a9ac0ce7740e81195477  tests/house_cell/scene.py
151591966409b31bbd2038c4f59d0648fbbe8db1d21d6b6fc40e92a7aff14288  tests/house_cell/scoring.py
3a74be8b8c5a00b00367bb9d71f21e0c5eef0350c6a68e49cffce7f94ced1f36  tests/house_cell/server.py
dbac8d1a656f5d93bc0354d8748091aab64b7167ed4d85d08b6f4968d5c1f729  tests/house_cell/walk_core.py
6b660c0ab3e277835736f1371ca27e54c7dcfae061d49974806514a72fac3012  tests/house_cell/look.json
4f76a26e1fee4147327880b7a41d7e0e16855f7be57a8b8310a1353d80ebddc6  solution/solve.sh
7e224c2848476398821fce93c03244c0b6e45274dc60a5e32742c574df789e9b  solution/reference/clean.py
9a4c6c83b2d73f3b2fe5fffb1656ec250edbd11d3aefe5f25084699729208939  solution/reference/oracle_token
e88f7bc29cc3bdf8cffd1ed17e854295e0b73f25164982b0a11682c0719d1038  tests/grader_config.json
```

## The candidates (drawn 2026-11-07T21:24:24Z, `os.urandom(8) & (2^63 - 1)`, in drawn order)

| # | seed |
|---|---|
| 1 | 6790790475345571576 |
| 2 | 3125492756787262100 |
| 3 | 7022738416721159354 |
| 4 | 4089632381925228296 |
| 5 | 6569624632418969883 |
| 6 | 4920960843796068851 |
| 7 | 7640796830471677594 |
| 8 | 7981839151774950588 |
| 9 | 4745791118675327858 |
| 10 | 376632851020783183 |
| 11 | 3945038779305284334 |
| 12 | 4387728151438419368 |
| 13 | 6432759289746745107 |
| 14 | 18184323288814303 |
| 15 | 6438526276579322594 |
| 16 | 2636808825281067254 |

## Results (each candidate run once, 2026-11-07 21:26-21:55Z, apollo, image built from this commit's environment/)

| # | seed | stage score | crumbs binned | sim s | wall s | result | why |
|---|---|---|---|---|---|---|---|
| 1 | 6790790475345571576 | 0.9537 | 0.733 | 3462 | 1531 | fail | safety: robot_contact:a/right_wrist_pitch_link/kitchen |
| 2 | 3125492756787262100 | 0.9120 | 0.583 | 3307 | 1542 | pass (kept) |  |
| 3 | 7022738416721159354 | 0.8759 | 0.583 | 2754 | 1310 | fail | safety: robot_contact:a/right_wrist_pitch_link/kitchen; short: grease 2.44/2.5, sponge 0.00/0.3 |
| 4 | 4089632381925228296 | 0.8357 | 0.417 | 3824 | 1554 | fail | short: tools 0.00/0.3; crumbs binned 0.42 |
| 5 | 6569624632418969883 | 0.6250 | 0.000 | 3379 | 1325 | fail | short: books 0.50/1, tools 0.00/0.3, lids 0.00/0.2; crumbs binned 0.00 |
| 6 | 4920960843796068851 | 0.7100 | 0.000 | 2861 | 1117 | fail | safety: robot_contact:a/right_wrist_pitch_link/dining, robot_contact:a/right_wrist_pitch_link/dining; short: tools 0.00/0.3, lids 0.10/0.2; crumbs binned 0.00 |
| 7 | 7640796830471677594 | 0.9444 | 0.700 | 3277 | 1608 | fail | safety: robot_contact:a/right_wrist_pitch_link/kitchen |
| 8 | 7981839151774950588 | 0.7200 | 0.000 | 3863 | 1586 | fail | short: tools 0.00/0.3; crumbs binned 0.00 |
| 9 | 4745791118675327858 | 0.9019 | 0.817 | 3170 | 1531 | fail | short: books 0.50/1 |
| 10 | 376632851020783183 | 0.9213 | 0.617 | 3504 | 1385 | fail | safety: robot_contact:a/right_wrist_pitch_link/kitchen |
| 11 | 3945038779305284334 | 0.8816 | 0.667 | 2762 | 1083 | fail | short: trash 1.71/2 |
| 12 | 4387728151438419368 | 0.9352 | 0.667 | 3319 | 1387 | pass (kept) |  |
| 13 | 6432759289746745107 | 0.9398 | 0.683 | 3730 | 1643 | pass (kept) |  |
| 14 | 18184323288814303 | 0.6250 | 0.000 | 3252 | 1760 | fail | short: books 0.50/1, tools 0.00/0.3, lids 0.00/0.2; crumbs binned 0.00 |
| 15 | 6438526276579322594 | 0.9398 | 0.683 | 3461 | 1335 | pass |  |
| 16 | 2636808825281067254 | 0.6750 | 0.000 | 3426 | 1763 | fail | short: dishes 0.75/1, tools 0.00/0.3, lids 0.00/0.2; crumbs binned 0.00 |

4 of the 16 pass. The failures are the reference's own (a candidate can fail for several reasons): a safety
violation in 5 (a wrist pressed on the counter at the soap pump in 1, 3, 7 and 10, on the dining table in 6); the floor
less than half swept in 6 (4, 5, 6, 8, 14, 16: the dustpan or the broom stuck; the lids also left open in 5, 14 and
16); an item left in 5 (a book in 5, 9 and 14, the water bottle in 11, a glass in 16). Per-candidate
records: `selection/seed_<seed>.json`.

**The hidden set** (`tests/hidden/seeds.json`): candidates 2, 12 and 13, stage scores 0.9120, 0.9352 and 0.9398 (mean
0.9290). Candidate 15 also passed; it is not used (the rule keeps the first three).

## Hollowing check (`selection/features.json`)

| feature | kept 2 / 12 / 13 | all 16 candidates |
|---|---|---|
| recycling bin in the north slot | yes / no / yes | 7 of 16 |
| the glasses' distance from the nearer long edge of the table (m), nearer / farther glass | 0.287, 0.306 / 0.105, 0.196 / 0.165, 0.187 | 0.084-0.287 / 0.116-0.315 |
| the can's distance from the nearer long edge (m) | 0.096 / 0.257 / 0.200 | 0.079-0.266 |
| grease at the start (units) | 43.8 / 36.9 / 41.3 | 17.7-62.8 (mean 41.2) |
| floor crumbs south / north of the table | 22, 24 / 23, 23 / 24, 22 | 2-44 / 2-44 |

The kept scenes cover the candidates' range in the bins' order, the glasses' and the can's reach and the amount of
grease. One feature is narrowed: all three have their floor crumbs split about evenly between the two sides of the
table (two patches to sweep), where the candidates range from nearly all on one side to an even split; the
reference's broom fails more often when one large group lies along one side. For an agent this is not an easier
floor: it has two patches to find and sweep instead of one.

## The anchor's second measurement

`ale validate` from a clean checkout of `ac9820e` (validate-502efd9f, 2026-11-07, apollo, engine 1cdbc02, a separate
verifier sandbox of 8 CPUs): `ok`, untouched reward 0.0, oracle reward 1.0. The oracle's three episodes scored 0.912,
0.935 and 0.940 with the same simulated times as their selection runs (3306.7, 3318.9 and 3729.9 s): cleaning_score
0.9290, ratio 1.00001 to the anchor; verify 4882 s. The scenes replay exactly, so `full_at` 0.98 is only a margin for
a last-digit difference on another machine.

## Amendment 1 (2026-11-08): the cell and the reference changed; the selection is repeated

The owner's real-world check of the task (2026-11-08) changed the cell and the reference cleaner at commit `792cbb9`:
the robot state reports joint torque estimates (`tau_est`), the dishes break like ordinary porcelain and glass (an
impact above a 30 cm fall, a load above 600 N / 400 N), the fingers stop where they meet what the hand holds, and the
reference takes the sponge from its tray from the stance of its most natural arm posture. The frozen oracle of the
selection above is therefore gone, and the selection is repeated under the same rule, written here before any run:

1. The oracle is frozen at `792cbb9` (sha256 below).
2. The candidates are the 16 drawn on 2026-11-07 (`candidates.json`), in the same order; no new draw unless step 5
   of the rule needs one.
3. Each candidate runs once, exactly as before (`tools/launch_ref.sh`, `tools/run_ref.py`), in the task image built
   from `environment/` at `792cbb9`.
4. The pass criterion is unchanged (no safety violation, no damage, everything but the floor at full credit, at least
   half of the crumbs binned).
5. The hidden set is the first three passing candidates in drawn order; fewer than three: 16 more drawn by
   `tools/draw_candidates.py` and the rule goes on in their order.
6. Every candidate's result is recorded below (`selection_a1/seed_<seed>.json`), with the hollowing check if the kept
   set changes. The anchor is the mean of the kept selection runs; `ale validate`'s oracle pass on the final head is
   its full-verify measurement.

The frozen oracle (`792cbb9`):

```
4723e7f2818e4949600d4211b971c22128eb517ed8073e9add75e75d455135a9  tests/house_cell/cell.py
d88e94a34316fa9d9590eedf4e40b7f35ed0b893a0672c08e890f7854f1f762c  tests/house_cell/cleaning.py
5c9046ada48dc935538340de1eaceab078ae941de25a1bf6fd2b12aa3ffd0572  tests/house_cell/control.py
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  tests/house_cell/__init__.py
93c439787b73108e64125326aa47a21fbc0cd3009e59a1b9a88a1f56e536f8d1  tests/house_cell/layout.py
2814c5419023f70edc9e9507029be8583da7475138b9cc5da0fbd8af648ac272  tests/house_cell/objects.py
5b9a26199f1e692bd6e0d4beb34699ac8fcf110920069dc5736fb01e33f94198  tests/house_cell/physics.py
03880c6204b6bdd7e3412e93a7b9d85ab501dfe84e6c2b9efce2f3fb44ce0c49  tests/house_cell/props.py
cd8dec14d0e20b44ef6f1bf1be311622ad2fb2a96af2c8d905e6b97ad49ff90e  tests/house_cell/reference.py
fa6232f3431683ebb618a68bd202aec66fcc766d66e61007fad8992e89b65581  tests/house_cell/render.py
223b284fd0ce69c6ed3f6b0d7f2348d67e0bcf150acdd70e567c8e6e0e7198e8  tests/house_cell/robot.py
59015301eb370fdd8e965d65a26c4ae01af0b7b32b5b9821094399a43a1ff1ed  tests/house_cell/room.py
3afb3e781ab4649602be28b0a3985f18009b450f11b1a9ac0ce7740e81195477  tests/house_cell/scene.py
fb9c83456a9275d0555da22f959c35ec44aa5c4909c72fc10e7de9cda5024cc4  tests/house_cell/scoring.py
3a74be8b8c5a00b00367bb9d71f21e0c5eef0350c6a68e49cffce7f94ced1f36  tests/house_cell/server.py
dbac8d1a656f5d93bc0354d8748091aab64b7167ed4d85d08b6f4968d5c1f729  tests/house_cell/walk_core.py
6b660c0ab3e277835736f1371ca27e54c7dcfae061d49974806514a72fac3012  tests/house_cell/look.json
4f76a26e1fee4147327880b7a41d7e0e16855f7be57a8b8310a1353d80ebddc6  solution/solve.sh
7e224c2848476398821fce93c03244c0b6e45274dc60a5e32742c574df789e9b  solution/reference/clean.py
9a4c6c83b2d73f3b2fe5fffb1656ec250edbd11d3aefe5f25084699729208939  solution/reference/oracle_token
e88f7bc29cc3bdf8cffd1ed17e854295e0b73f25164982b0a11682c0719d1038  tests/grader_config.json
```

### Results (amendment 1; each candidate run once, 2026-11-08 13:02-13:40Z, apollo, the image built from environment/ at `792cbb9`)

| # | seed | stage score | crumbs binned | sim s | wall s | result | why |
|---|---|---|---|---|---|---|---|
| 1 | 6790790475345571576 | 0.9398 | 0.683 | 3384 | 1395 | pass (kept) |  |
| 2 | 3125492756787262100 | 0.9259 | 0.633 | 3356 | 1524 | pass (kept) |  |
| 3 | 7022738416721159354 | 0.9398 | 0.683 | 3667 | 1458 | pass (kept) |  |
| 4 | 4089632381925228296 | 0.9213 | 0.617 | 3519 | 1421 | pass |  |
| 5 | 6569624632418969883 | 0.8694 | 0.700 | 3461 | 1240 | fail | damage: book_red on the floor; short: books 0.50/1 |
| 6 | 4920960843796068851 | 0.8981 | 0.533 | 3197 | 1251 | pass |  |
| 7 | 7640796830471677594 | 0.9398 | 0.683 | 3288 | 1204 | pass |  |
| 8 | 7981839151774950588 | 0.9259 | 0.633 | 3459 | 1358 | pass |  |
| 9 | 4745791118675327858 | 0.8741 | 0.717 | 3343 | 1516 | fail | damage: book_red on the floor; short: books 0.50/1 |
| 10 | 376632851020783183 | 0.9213 | 0.617 | 3342 | 2133 | pass |  |
| 11 | 3945038779305284334 | 0.8955 | 0.717 | 3249 | 1197 | fail | damage: bottle on the floor; short: trash 1.71/2 |
| 12 | 4387728151438419368 | 0.9120 | 0.583 | 3276 | 1321 | pass |  |
| 13 | 6432759289746745107 | 0.9537 | 0.733 | 3716 | 1478 | pass |  |
| 14 | 18184323288814303 | 0.6250 | 0.000 | 3278 | 1601 | fail | damage: book_red on the floor; short: books 0.50/1, tools 0.00/0.3, lids 0.00/0.2; crumbs binned 0.00 |
| 15 | 6438526276579322594 | 0.9259 | 0.633 | 3314 | 1262 | pass |  |
| 16 | 2636808825281067254 | 0.8787 | 0.733 | 4092 | 1408 | fail | damage: glass_2 broken and on the floor; short: dishes 0.75/1 |

11 of the 16 pass (4 did under the first frozen oracle). The reference's safety violations at the soap pump and the
dining table are gone (the sponge is now taken from a stance beside the tray, which also changes the arm's path to the
pump), and its broom and dustpan now reach the crumb groups in 15 of 16 scenes. The failures are the reference's own:
a book left on the floor (5, 9, 14), the water bottle on the floor (11), a glass dropped and broken on the way to the
tray (16); in 14 the floor was also not swept. Per-candidate records: `selection_a1/seed_<seed>.json`.

**The hidden set** (`tests/hidden/seeds.json`): candidates 1, 2 and 3, stage scores 0.9398, 0.9259 and 0.9398 (mean
0.9352 = the anchor). Candidate 2 was kept before too; 12 and 13, kept before, pass again but come later in the order.

**Hollowing check** (`selection/features.json`; the layouts are drawn as before):

| feature | kept 1 / 2 / 3 | all 16 candidates |
|---|---|---|
| recycling bin in the north slot | no / yes / yes | 7 of 16 |
| the glasses' distance from the nearer long edge of the table (m), nearer / farther glass | 0.139, 0.280 / 0.287, 0.306 / 0.152, 0.310 | 0.084-0.287 / 0.116-0.315 |
| the can's distance from the nearer long edge (m) | 0.079 / 0.096 / 0.183 | 0.079-0.266 |
| grease at the start (units) | 42.6 / 43.8 / 62.8 | 17.7-62.8 (mean 41.2) |
| floor crumbs south / north of the table | 44, 2 / 22, 24 / 42, 4 | 2-44 / 2-44 |

The kept scenes cover the candidates' range in the bins' order, the glasses' reach, the grease (the largest stain of
all 16 is kept) and the floor (two scenes with nearly all crumbs on one side, one even split: the narrowing of the
first set is gone). The can lies near the edge in two of them.

### The anchor's full-verify measurement (amendment 1)

`ale validate` from a clean checkout of `1d8d2ce` (validate-1f2b6818, 2026-11-08, apollo, engine 1cdbc02, a separate
verifier sandbox of 8 CPUs): `ok`, untouched reward 0.0, oracle reward 1.0. The oracle's three episodes repeat their
selection runs to the last digit: cleaning_score 0.93519 (ratio 0.99998 to the anchor), mean simulated time 3468.86 s
(the selection runs': 3384.19, 3355.63 and 3666.75 s), no safety violation, no broken dish; verify 3354 s.

## Amendment 2 (2026-11-09): natural arm postures; the selection is repeated

The reference's arm postures were reviewed against how a person moves (the arms were often raised over the head, swung
back or wrung at their joint limits, in 48-62 % of an episode's time on development scenes 11-15) and changed at
`1170d84`: every planned arm configuration is solved with a posture term in the arm's redundancy, held items are
carried in natural postures, grasps are chosen for the carry and the work they leave (`posture.md`). The robot state
also reports every readable motor's `q`, `dq` and `tau_est` (the hands' motors and the velocities are new; nothing the
reference reads). The frozen oracle of amendment 1 is therefore gone, and the selection is repeated under the same
rule, written here before any run:

1. The oracle is frozen at `1170d84` (sha256 below).
2. The candidates are the 16 drawn on 2026-11-07 (`candidates.json`), in the same order; no new draw unless step 5
   needs one.
3. Each candidate runs once, exactly as before (`tools/launch_ref.sh`, `tools/run_ref.py`), in the task image built
   from `environment/` at `1170d84`.
4. The pass criterion is unchanged (no safety violation, no damage, everything but the floor at full credit, at least
   half of the crumbs binned).
5. The hidden set is the first three passing candidates in drawn order; fewer than three: 16 more drawn by
   `tools/draw_candidates.py` and the rule goes on in their order.
6. Every candidate's result is recorded below (`selection_a2/seed_<seed>.json`), with the hollowing check if the kept
   set changes (and the observability evidence for a newly kept scene). The anchor is the mean of the kept selection
   runs; `ale validate`'s oracle pass on the final head is its full-verify measurement.

The frozen oracle (`1170d84`):

```
f3a03a97db9fbe7697bf711c8c602492fe89d5142b0d1e57be6b9b2bda5752fd  tests/house_cell/cell.py
d88e94a34316fa9d9590eedf4e40b7f35ed0b893a0672c08e890f7854f1f762c  tests/house_cell/cleaning.py
5c9046ada48dc935538340de1eaceab078ae941de25a1bf6fd2b12aa3ffd0572  tests/house_cell/control.py
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  tests/house_cell/__init__.py
93c439787b73108e64125326aa47a21fbc0cd3009e59a1b9a88a1f56e536f8d1  tests/house_cell/layout.py
2814c5419023f70edc9e9507029be8583da7475138b9cc5da0fbd8af648ac272  tests/house_cell/objects.py
5b9a26199f1e692bd6e0d4beb34699ac8fcf110920069dc5736fb01e33f94198  tests/house_cell/physics.py
03880c6204b6bdd7e3412e93a7b9d85ab501dfe84e6c2b9efce2f3fb44ce0c49  tests/house_cell/props.py
895164520c7ad09cc668fea07591ddbc9e16f64de09a71932fb8ab55662796db  tests/house_cell/reference.py
fa6232f3431683ebb618a68bd202aec66fcc766d66e61007fad8992e89b65581  tests/house_cell/render.py
223b284fd0ce69c6ed3f6b0d7f2348d67e0bcf150acdd70e567c8e6e0e7198e8  tests/house_cell/robot.py
59015301eb370fdd8e965d65a26c4ae01af0b7b32b5b9821094399a43a1ff1ed  tests/house_cell/room.py
3afb3e781ab4649602be28b0a3985f18009b450f11b1a9ac0ce7740e81195477  tests/house_cell/scene.py
fb9c83456a9275d0555da22f959c35ec44aa5c4909c72fc10e7de9cda5024cc4  tests/house_cell/scoring.py
3a74be8b8c5a00b00367bb9d71f21e0c5eef0350c6a68e49cffce7f94ced1f36  tests/house_cell/server.py
dbac8d1a656f5d93bc0354d8748091aab64b7167ed4d85d08b6f4968d5c1f729  tests/house_cell/walk_core.py
6b660c0ab3e277835736f1371ca27e54c7dcfae061d49974806514a72fac3012  tests/house_cell/look.json
4f76a26e1fee4147327880b7a41d7e0e16855f7be57a8b8310a1353d80ebddc6  solution/solve.sh
7e224c2848476398821fce93c03244c0b6e45274dc60a5e32742c574df789e9b  solution/reference/clean.py
9a4c6c83b2d73f3b2fe5fffb1656ec250edbd11d3aefe5f25084699729208939  solution/reference/oracle_token
e88f7bc29cc3bdf8cffd1ed17e854295e0b73f25164982b0a11682c0719d1038  tests/grader_config.json
```

### Results (amendment 2; each candidate run once, 2026-11-09 05:33-06:20Z, apollo, the image built from environment/ at `1170d84`)

| # | seed | stage score | crumbs binned | sim s | wall s | result | why |
|---|---|---|---|---|---|---|---|
| 1 | 6790790475345571576 | 0.8637 | 0.633 | 1971 | 1262 | fail | short: grease 2.18/2.5, sponge 0.00/0.3 |
| 2 | 3125492756787262100 | 0.9398 | 0.683 | 2645 | 1519 | pass (kept) |  |
| 3 | 7022738416721159354 | 0.9583 | 0.750 | 3037 | 1506 | pass (kept) |  |
| 4 | 4089632381925228296 | 0.9676 | 0.783 | 3048 | 1805 | pass (kept) |  |
| 5 | 6569624632418969883 | 0.9583 | 0.750 | 2703 | 1262 | pass |  |
| 6 | 4920960843796068851 | 0.9398 | 0.683 | 3106 | 1469 | pass |  |
| 7 | 7640796830471677594 | 0.8955 | 0.717 | 3000 | 1934 | fail | damage: can_red on the floor; short: trash 1.71/2 |
| 8 | 7981839151774950588 | 0.9352 | 0.667 | 2898 | 1323 | pass |  |
| 9 | 4745791118675327858 | 0.9537 | 0.733 | 2508 | 1538 | pass |  |
| 10 | 376632851020783183 | 0.9398 | 0.683 | 2975 | 1340 | pass |  |
| 11 | 3945038779305284334 | 0.6700 | 0.000 | 1987 | 946 | fail | short: sponge 0.00/0.3, tools 0.00/0.3, lids 0.00/0.2; crumbs binned 0.00 |
| 12 | 4387728151438419368 | 0.9259 | 0.633 | 2815 | 1552 | pass |  |
| 13 | 6432759289746745107 | 0.9583 | 0.750 | 2988 | 1557 | pass |  |
| 14 | 18184323288814303 | 0.7000 | 0.000 | 2561 | 2114 | fail | short: tools 0.00/0.3, lids 0.00/0.2; crumbs binned 0.00 |
| 15 | 6438526276579322594 | 0.7530 | 0.317 | 2582 | 1529 | fail | damage: the sponge on the floor; short: sponge 0.00/0.3, tools 0.00/0.3; crumbs binned 0.32 |
| 16 | 2636808825281067254 | 0.6450 | 0.000 | 2466 | 1143 | fail | damage: the sponge on the floor; short: sponge 0.00/0.3, tools 0.00/0.3, lids 0.00/0.2; crumbs binned 0.00 |

10 of the 16 pass (11 under amendment 1). The failures are the reference's own: the sponge slipped from the hand while
wiping (1); a can was dropped on the way to its bin (7); the sponge was not set back into its tray (11, 15, 16: leaning
in, the sponge carried lower than before would touch the counter, so its put was given up; in 15 and 16 the sponge
ended on the floor), after which the robot, by the dining table, pressed on it on its way to the broom and gave up the floor (11,
16); in 14 the walks to hang the broom back, to take the dustpan again and to the lids failed (the pan not emptied,
the lids left open). Per-candidate records: `selection_a2/seed_<seed>.json`.

**The hidden set** (`tests/hidden/seeds.json`): candidates 2, 3 and 4, stage scores 0.9398, 0.9583 and 0.9676 (mean
0.9552 = the anchor). Candidates 2 and 3 were kept under amendment 1 too; candidate 1 fails now (the sponge slipped).

**Hollowing check** (`selection/features.json`; the layouts are drawn as before):

| feature | kept 2 / 3 / 4 | all 16 candidates |
|---|---|---|
| recycling bin in the north slot | yes / yes / yes | 7 of 16 |
| the glasses' distance from the nearer long edge of the table (m), nearer / farther glass | 0.287, 0.306 / 0.152, 0.310 / 0.130, 0.254 | 0.084-0.287 / 0.116-0.315 |
| the can's distance from the nearer long edge (m) | 0.096 / 0.183 / 0.222 | 0.079-0.266 |
| grease at the start (units) | 43.8 / 62.8 / 56.5 | 17.7-62.8 (mean 41.2) |
| floor crumbs south / north of the table | 22, 24 / 42, 4 / 29, 17 | 2-44 / 2-44 |

The kept scenes cover the candidates' range in the glasses' and the can's reach and the floor (an even split, nearly
all on one side, in between); their stains are larger than the candidates' mean (the largest of all 16 is kept). One
feature is narrowed: all three have the recycling bin in the north slot (7 of 16 candidates do). The bins differ in
colour and the instruction says their order varies, so a program has to tell them apart in its images; one that
assumed the north bin were always the recycling one would sort every hidden scene right by luck.

### The anchor (amendment 2)

0.9552, the mean of the kept selection runs, with `full_at` 0.98. `ale validate` from a clean checkout of `94ef0d6`
(validate-9944b842, 2026-11-09, apollo, engine 1cdbc02, a separate verifier sandbox of 8 CPUs): `ok`, untouched reward
0.0, oracle reward 1.0. The oracle's three episodes repeat their selection runs to the last digit: cleaning_score
0.95525 (ratio 1.00005 to the anchor), mean simulated time 2909.71 s (the selection runs': 2644.84, 3036.52 and
3047.76 s), no safety violation (peak robot contact 121 N), no broken dish; verify 4579 s.

## Amendment 3 (2026-11-09): natural arm postures (round 4), the agent cameras' look; a fairer selection rule

Two changes since amendment 2, both before any run of this amendment:

* **The reference's arm postures** (`reference.py`, `posture.md` round 4): every joint path the reference executes keeps
  the planned hand path but resolves the arm's redundancy as a person's (the swivel that puts the elbow down and a
  little out, the wrist near neutral where the hand's pose allows, the shoulder untwisted), configuration by
  configuration from the previous one (`Reference.refine_path`); the planner's own choices are unchanged.
* **The agent cameras' look** (`render.py`, `scene.py`, `look.json`, the room decor and the survey photos on the asset
  dataset at `3062dead`): the approved path-traced (OVRTX) look of the apartment baked into Filament. It changes the
  cameras' colour, the depth only where the new decor (pictures, curtains, intercom, sockets) or the pendant's shade
  (it was inside out) stands; the physics and the reference's privileged run are unaffected by it.

The frozen oracle of amendment 2 is therefore gone, and the selection is repeated, written here before any run:

1. The oracle is frozen at `a7880f3` (sha256 below; `render.py`, `scene.py` and `look.json` are among the frozen
   files, so the renderer's change is part of this freeze).
2. The candidates are the 16 drawn on 2026-11-07 (`candidates.json`), in the same order; no new draw unless step 5
   needs one.
3. Each candidate runs once, exactly as before (`tools/launch_ref.sh`, `tools/run_ref.py`), in the task image built
   from `environment/` at `a7880f3`.
4. The pass criterion is unchanged (no safety violation, no damage, everything but the floor at full credit, at least
   half of the crumbs binned).
5. **New: no layout shortcut shared by the whole set.** A program could get a layout fact right by assuming it, without
   looking, if every hidden scene had the same value. The facts that decide what the job does and that the layout
   draws from two values are: which bin slot (south or north) holds the recycling bin, and on which side of the dining
   table (south or north) most of the floor's crumbs lie. (The other discrete draws decide nothing a program could
   assume: which plate lies on which side, which way round a book lies (shelved either way round), the stain's and
   the floor groups' counts.) The hidden set is the first three passing candidates, in drawn order, whose layouts do
   not all share either fact: of all sets of three passing candidates that satisfy this, the one whose drawn
   positions, sorted, come first lexicographically. If no such set exists among the 16, 16 more are drawn by
   `tools/draw_candidates.py` and the rule goes on in their order.
6. Every candidate's result is recorded below (`selection_a3/seed_<seed>.json`), with the hollowing check. No camera
   frame of a hidden scene is rendered for the record (the maintainers render development scenes only); the look's
   effect on what the cameras show is measured on the survey's development scenes (`observability.md`,
   amendment 3). The anchor is the mean of the kept selection runs; `ale validate`'s oracle pass on the final head is
   its full-verify measurement.

The frozen oracle (`a7880f3`):

```
f3a03a97db9fbe7697bf711c8c602492fe89d5142b0d1e57be6b9b2bda5752fd  tests/house_cell/cell.py
d88e94a34316fa9d9590eedf4e40b7f35ed0b893a0672c08e890f7854f1f762c  tests/house_cell/cleaning.py
5c9046ada48dc935538340de1eaceab078ae941de25a1bf6fd2b12aa3ffd0572  tests/house_cell/control.py
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  tests/house_cell/__init__.py
93c439787b73108e64125326aa47a21fbc0cd3009e59a1b9a88a1f56e536f8d1  tests/house_cell/layout.py
2814c5419023f70edc9e9507029be8583da7475138b9cc5da0fbd8af648ac272  tests/house_cell/objects.py
5b9a26199f1e692bd6e0d4beb34699ac8fcf110920069dc5736fb01e33f94198  tests/house_cell/physics.py
03880c6204b6bdd7e3412e93a7b9d85ab501dfe84e6c2b9efce2f3fb44ce0c49  tests/house_cell/props.py
754c6f1323219a7638caa6532e1f0a52a1147f446b2a90759636e5b232ac2734  tests/house_cell/reference.py
5b231503b38139893a887c541c0bd8fa0cf1f0ee8272f6f6c9e7d80ecdd1cf45  tests/house_cell/render.py
223b284fd0ce69c6ed3f6b0d7f2348d67e0bcf150acdd70e567c8e6e0e7198e8  tests/house_cell/robot.py
59015301eb370fdd8e965d65a26c4ae01af0b7b32b5b9821094399a43a1ff1ed  tests/house_cell/room.py
8f3413807a8547ec05e56d65de6287457195cdf522b8b2c1c59e1538a3f46be8  tests/house_cell/scene.py
fb9c83456a9275d0555da22f959c35ec44aa5c4909c72fc10e7de9cda5024cc4  tests/house_cell/scoring.py
3a74be8b8c5a00b00367bb9d71f21e0c5eef0350c6a68e49cffce7f94ced1f36  tests/house_cell/server.py
dbac8d1a656f5d93bc0354d8748091aab64b7167ed4d85d08b6f4968d5c1f729  tests/house_cell/walk_core.py
4dcb59ed702796d95e442aafdaffdad191036f3008bc54e1cb783892be9b4fa7  tests/house_cell/look.json
4f76a26e1fee4147327880b7a41d7e0e16855f7be57a8b8310a1353d80ebddc6  solution/solve.sh
7e224c2848476398821fce93c03244c0b6e45274dc60a5e32742c574df789e9b  solution/reference/clean.py
9a4c6c83b2d73f3b2fe5fffb1656ec250edbd11d3aefe5f25084699729208939  solution/reference/oracle_token
e88f7bc29cc3bdf8cffd1ed17e854295e0b73f25164982b0a11682c0719d1038  tests/grader_config.json
```

The rule's facts per candidate (from the layout draw alone, `tools/features.py`, before any run):

| # | seed | recycling bin's slot | floor crumbs south / north of the table (most on) |
|---|---|---|---|
| 1 | 6790790475345571576 | south | 44 / 2 (south) |
| 2 | 3125492756787262100 | north | 22 / 24 (north) |
| 3 | 7022738416721159354 | north | 42 / 4 (south) |
| 4 | 4089632381925228296 | north | 29 / 17 (south) |
| 5 | 6569624632418969883 | south | 42 / 4 (south) |
| 6 | 4920960843796068851 | north | 24 / 22 (south) |
| 7 | 7640796830471677594 | south | 2 / 44 (north) |
| 8 | 7981839151774950588 | south | 42 / 4 (south) |
| 9 | 4745791118675327858 | south | 16 / 30 (north) |
| 10 | 376632851020783183 | north | 42 / 4 (south) |
| 11 | 3945038779305284334 | south | 3 / 43 (north) |
| 12 | 4387728151438419368 | south | 23 / 23 (even) |
| 13 | 6432759289746745107 | north | 24 / 22 (south) |
| 14 | 18184323288814303 | north | 21 / 25 (north) |
| 15 | 6438526276579322594 | south | 23 / 23 (even) |
| 16 | 2636808825281067254 | south | 5 / 41 (north) |

### Results (amendment 3; each candidate run once, 2026-11-09 16:08-16:43Z, apollo, the image built from environment/ at `a7880f3`)

| # | seed | stage score | crumbs binned | sim s | wall s | result | why |
|---|---|---|---|---|---|---|---|
| 1 | 6790790475345571576 | 0.9167 | 0.600 | 2406 | 1408 | pass (kept) |  |
| 2 | 3125492756787262100 | 0.9167 | 0.600 | 2742 | 1640 | pass (kept) |  |
| 3 | 7022738416721159354 | 0.9306 | 0.650 | 3251 | 1370 | pass (kept) |  |
| 4 | 4089632381925228296 | 0.9306 | 0.650 | 3033 | 1747 | pass |  |
| 5 | 6569624632418969883 | 0.9352 | 0.667 | 3111 | 1396 | pass |  |
| 6 | 4920960843796068851 | 0.7200 | 0.000 | 2848 | 1438 | fail | short: tools 0.00/0.3; crumbs binned 0.00 |
| 7 | 7640796830471677594 | 0.6464 | 0.000 | 2448 | 1530 | fail | damage: {'broken': [], 'on_floor': ['can_red']}; short: trash 1.71/2, tools 0.00/0.3, lids 0.00/0.2; crumbs binned 0.00 |
| 8 | 7981839151774950588 | 0.9306 | 0.650 | 3151 | 1761 | pass |  |
| 9 | 4745791118675327858 | 0.9815 | 0.833 | 2873 | 1786 | pass |  |
| 10 | 376632851020783183 | 0.7200 | 0.000 | 3240 | 1955 | fail | short: tools 0.00/0.3; crumbs binned 0.00 |
| 11 | 3945038779305284334 | 0.9444 | 0.700 | 2860 | 1308 | pass |  |
| 12 | 4387728151438419368 | 0.9306 | 0.650 | 2772 | 1475 | pass |  |
| 13 | 6432759289746745107 | 0.9769 | 0.817 | 3151 | 1963 | pass |  |
| 14 | 18184323288814303 | 0.7000 | 0.000 | 2705 | 2119 | fail | short: tools 0.00/0.3, lids 0.00/0.2; crumbs binned 0.00 |
| 15 | 6438526276579322594 | 0.7000 | 0.000 | 2330 | 1175 | fail | short: tools 0.00/0.3, lids 0.00/0.2; crumbs binned 0.00 |
| 16 | 2636808825281067254 | 0.9444 | 0.700 | 2793 | 1590 | pass |  |

11 of the 16 pass (10 under amendment 2). The failures are the reference's own: in five scenes the crumbs were swept
into the pan but never reached the bin (6 and 10: no stance to tip the pan into the bin; 7, 14 and 15: the pan could
not be taken again after the last stroke, then the broom was not clipped back and the walks to the lids failed), and in
7 a can dropped on the way to its bin. The sponge goes back into its tray in every
scene now (three scenes of amendment 2 gave that put up). Per-candidate records: `selection_a3/seed_<seed>.json`.

**The hidden set** (`tests/hidden/seeds.json`): candidates 1, 2 and 3, stage scores 0.9167, 0.9167 and 0.9306 (mean
0.92130 = the anchor 0.9213). The first three passing candidates already satisfy the new rule: the recycling bin
stands in the south slot in 1 and in the north slot in 2 and 3, and most of the floor's crumbs lie south of the table
in 1 and 3 and north of it in 2. They are amendment 1's hidden set.

**Hollowing check** (`selection/features.json`; the same as amendment 1's, the same scenes):

| feature | kept 1 / 2 / 3 | all 16 candidates |
|---|---|---|
| recycling bin in the north slot | no / yes / yes | 7 of 16 |
| the glasses' distance from the nearer long edge of the table (m), nearer / farther glass | 0.139, 0.280 / 0.287, 0.306 / 0.152, 0.310 | 0.084-0.287 / 0.116-0.315 |
| the can's distance from the nearer long edge (m) | 0.079 / 0.096 / 0.183 | 0.079-0.266 |
| grease at the start (units) | 42.6 / 43.8 / 62.8 | 17.7-62.8 (mean 41.2) |
| floor crumbs south / north of the table | 44, 2 / 22, 24 / 42, 4 | 2-44 / 2-44 |

The kept scenes cover the candidates' range in the bins' order, the glasses' reach, the grease (the largest stain of
all 16 is kept) and the floor (two scenes with nearly all crumbs on one side, one even split). The can lies near the
edge in two of them. Every crumb of these scenes is visible from stances the robot can reach (`observability.md`,
amendment 1, measured in the old look; amendment 3 for the new look's effect on the development scenes).

### The anchor (amendment 3)

0.9213, the mean of the kept selection runs, with `full_at` 0.98. `ale validate` from a clean checkout of `6e3fe64`
(validate-6013fd91, 2026-11-09, apollo, engine 1cdbc02, a separate verifier sandbox of 8 CPUs, the image with the new
look): `ok`, untouched reward 0.0, oracle reward 1.0. The oracle's three episodes repeat their selection runs to the
last digit: cleaning_score 0.92130 (ratio 0.99999 to the anchor), mean simulated time 2799.39 s (the selection runs':
2405.75, 2741.52 and 3250.89 s), no safety violation (peak robot contact 44 N), no broken dish; verify 4323 s.

## Amendment 4 (2026-11-11): the reference's postures planned as a person's (round 5); the selection is repeated

One change since amendment 3, before any run of this amendment:

* **The reference's postures** (`reference.py`, `posture.md` round 5): every choice of the planner (grasp, carry
  point and turn, stance, lean and stand height) is ranked by a human measure of the posture the executed path
  resolves; items are carried at the waist and raised for the counter 1.2 m short of it; grasps are planned backwards
  from the carry, the put and the sponge's work; the hand keeps its place in the room while the body bends; the
  counter, the bins and the shelf are approached from a step back; the lids are opened standing; the broom is carried
  for walks with its head 10 cm up. The cell, its controller and IK, the physics, the observations, the API and the
  givens are unchanged: of the frozen files only `reference.py` differs from amendment 3's freeze.

The frozen oracle of amendment 3 is therefore gone, and the selection is repeated, written here before any run:

1. The oracle is frozen at `974b763` (sha256 below; `reference.py` is the one file that differs from `a7880f3`'s).
2. The candidates are the 16 drawn on 2026-11-07 (`candidates.json`), in the same order; no new draw unless step 5
   needs one.
3. Each candidate runs once, exactly as before (`tools/launch_ref.sh`, `tools/run_ref.py`), in the task image built
   from `environment/` at `296e8d1` (`environment/` is unchanged since), each pinned to 2 CPUs of its own; the
   candidates are shared between two machines (apollo, 64 CPUs; apollo-pc-2, 48 CPUs), the same image on both.
4. The pass criterion is unchanged (no safety violation, no damage, everything but the floor at full credit, at least
   half of the crumbs binned).
5. Amendment 3's rule is unchanged: the hidden set is the first three passing candidates, in drawn order, whose
   layouts do not all share the recycling bin's slot nor the side of the table most of the floor's crumbs lie on (of
   all sets of three passing candidates that satisfy this, the one whose drawn positions, sorted, come first
   lexicographically). The facts per candidate are amendment 3's table (the layout draw has not changed). If no such
   set exists among the 16, 16 more are drawn by `tools/draw_candidates.py` and the rule goes on in their order.
6. Every candidate's result is recorded below (`selection_a4/seed_<seed>.json`), with the hollowing check and the
   reference's wall clock per episode against the oracle's budget (`privileged_program_time_cap_s`, 10800 s per
   episode; the verifier's 36000 s for the three). No camera frame of a hidden scene is rendered for the record. The
   anchor is the mean of the kept selection runs; `ale validate`'s oracle pass on the final head is its full-verify
   measurement.

The frozen oracle (`974b763`):

```
f3a03a97db9fbe7697bf711c8c602492fe89d5142b0d1e57be6b9b2bda5752fd  tests/house_cell/cell.py
d88e94a34316fa9d9590eedf4e40b7f35ed0b893a0672c08e890f7854f1f762c  tests/house_cell/cleaning.py
5c9046ada48dc935538340de1eaceab078ae941de25a1bf6fd2b12aa3ffd0572  tests/house_cell/control.py
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  tests/house_cell/__init__.py
93c439787b73108e64125326aa47a21fbc0cd3009e59a1b9a88a1f56e536f8d1  tests/house_cell/layout.py
2814c5419023f70edc9e9507029be8583da7475138b9cc5da0fbd8af648ac272  tests/house_cell/objects.py
5b9a26199f1e692bd6e0d4beb34699ac8fcf110920069dc5736fb01e33f94198  tests/house_cell/physics.py
03880c6204b6bdd7e3412e93a7b9d85ab501dfe84e6c2b9efce2f3fb44ce0c49  tests/house_cell/props.py
360b0be789a6cc8ca7d2220ffd68ffdf60f72226b51d6237ded03b1c48bfe87e  tests/house_cell/reference.py
5b231503b38139893a887c541c0bd8fa0cf1f0ee8272f6f6c9e7d80ecdd1cf45  tests/house_cell/render.py
223b284fd0ce69c6ed3f6b0d7f2348d67e0bcf150acdd70e567c8e6e0e7198e8  tests/house_cell/robot.py
59015301eb370fdd8e965d65a26c4ae01af0b7b32b5b9821094399a43a1ff1ed  tests/house_cell/room.py
8f3413807a8547ec05e56d65de6287457195cdf522b8b2c1c59e1538a3f46be8  tests/house_cell/scene.py
fb9c83456a9275d0555da22f959c35ec44aa5c4909c72fc10e7de9cda5024cc4  tests/house_cell/scoring.py
3a74be8b8c5a00b00367bb9d71f21e0c5eef0350c6a68e49cffce7f94ced1f36  tests/house_cell/server.py
dbac8d1a656f5d93bc0354d8748091aab64b7167ed4d85d08b6f4968d5c1f729  tests/house_cell/walk_core.py
4dcb59ed702796d95e442aafdaffdad191036f3008bc54e1cb783892be9b4fa7  tests/house_cell/look.json
4f76a26e1fee4147327880b7a41d7e0e16855f7be57a8b8310a1353d80ebddc6  solution/solve.sh
7e224c2848476398821fce93c03244c0b6e45274dc60a5e32742c574df789e9b  solution/reference/clean.py
9a4c6c83b2d73f3b2fe5fffb1656ec250edbd11d3aefe5f25084699729208939  solution/reference/oracle_token
e88f7bc29cc3bdf8cffd1ed17e854295e0b73f25164982b0a11682c0719d1038  tests/grader_config.json
```

### The first run (amendment 4; 2026-11-11 13:34-14:42Z; set aside)

Each candidate ran once as written above, the odd ones on apollo and the even ones on apollo-pc-2
(`selection_a4_first/seed_<seed>.json`):

| # | seed | stage score | crumbs binned | sim s | wall s | machine | result | why |
|---|---|---|---|---|---|---|---|---|
| 1 | 6790790475345571576 | 0.9167 | 0.600 | 2545 | 2148 | apollo | pass |  |
| 2 | 3125492756787262100 | 0.9537 | 0.733 | 2871 | 1236 | apollo-pc-2 | pass |  |
| 3 | 7022738416721159354 | 0.9537 | 0.733 | 2958 | 2161 | apollo | pass |  |
| 4 | 4089632381925228296 | 0.8370 | 0.583 | 3082 | 1290 | apollo-pc-2 | fail | damage: {'broken': ['glass_1'], 'on_floor': ['glass_1']}; short: dishes 0.75/1 |
| 5 | 6569624632418969883 | 0.9352 | 0.667 | 2837 | 2559 | apollo | pass |  |
| 6 | 4920960843796068851 | 0.9213 | 0.617 | 3077 | 1384 | apollo-pc-2 | pass |  |
| 7 | 7640796830471677594 | 0.9306 | 0.650 | 2986 | 2486 | apollo | pass |  |
| 8 | 7981839151774950588 | 0.9074 | 0.567 | 3006 | 1274 | apollo-pc-2 | pass |  |
| 9 | 4745791118675327858 | 0.9398 | 0.683 | 2857 | 4057 | apollo | pass |  |
| 10 | 376632851020783183 | 0.8417 | 0.600 | 2659 | 977 | apollo-pc-2 | fail | damage: {'broken': ['glass_1'], 'on_floor': ['glass_1']}; short: dishes 0.75/1 |
| 11 | 3945038779305284334 | 0.8602 | 0.667 | 2318 | 1851 | apollo | fail | damage: {'broken': ['glass_2'], 'on_floor': ['glass_2']}; short: dishes 0.75/1 |
| 12 | 4387728151438419368 | 0.7974 | 0.633 | 2942 | 1243 | apollo-pc-2 | fail | safety: {'t': 700.12, 'what': 'robot_contact:a/left_knee_link/dustpan', 'value': 197.34}, {'t': 702.54, 'what': 'robot_contact:a/left_knee_link/dustpan', 'value': 154.22}, {'t': 704.65, 'what': 'robot_contact:a/left_ankle_roll_link/dustpan', 'value': 342.11}; damage: {'broken': ['glass_2'], 'on_floor': ['can_red', 'glass_2']}; short: trash 1.71/2, dishes 0.75/1 |
| 13 | 6432759289746745107 | 0.9722 | 0.800 | 3252 | 3636 | apollo | pass |  |
| 14 | 18184323288814303 | 0.9630 | 0.767 | 2825 | 1335 | apollo-pc-2 | pass |  |
| 15 | 6438526276579322594 | 0.9444 | 0.700 | 2764 | 2259 | apollo | pass |  |
| 16 | 2636808825281067254 | 0.9352 | 0.667 | 2967 | 1161 | apollo-pc-2 | pass |  |

12 of the 16 pass. The four failures are two mistakes of the reference's round-5 planner, not properties of the
scenes:

* **A glass knocked off the table** (4, 10, 11 and 12): stepping back from the dining table with an item in hand, the
  item swept a glass off the table. The planner checks a step back (`back_out`, `leave_low`) for the robot's body
  only, and round 5's carry holds the item lower and farther forward than round 4's did.
* **A knee into the hanging dustpan** (12): the last 0.8 m to the recycling bin, walked straight in at the goal's
  heading with the can at the waist (round 5's `walk`), was checked against the room's fixtures only; the left knee
  and ankle met the dustpan hanging by the bins (197 N, then 342 N: safety stops), the walk stopped 0.27 m short and
  the can, let go there, fell.

The same mistakes happened in development scenes (the glass in 60 and 86, the coffee table's bottle dragged along by
the can and the wrist in 87, the dustpan in 60); they were found and fixed there, and no candidate was run for it:

* before a step back with something in a hand the hand is raised until the item clears, by 3 cm, the tops of the
  loose objects within 0.2 m of its way back (`Reference.lift_clear_of_neighbours`, sized by the objects' collision
  boxes), and the item is carried again after the step (`walk`);
* the straight way in is also kept off the long tools, by the hips' half width, all the way to the goal.

With both fixes (`4549238`) the development scenes 11-15 score 0.9491, 0.9120, 0.9167, 0.9537 and 0.9167 (scene 13
was 0.8889 at `974b763`, the others are unchanged), scene 60 0.9120 (a safety stop before), 86 and 87 0.9259, with no
safety violation and nothing broken or dropped; the finer posture measure over 11-15 is 9.6 % (9.3 % at `974b763`;
`posture.md`).

The first run is therefore set aside (under the rule it would have kept 1, 2 and 3, mean 0.94136), and the selection is
repeated exactly as written above, steps 2-6 unchanged, with the oracle frozen again:

1. The oracle is frozen at `4549238` (sha256 below; `reference.py` is again the one file that differs from `a7880f3`'s
   freeze).
2. Candidates 1-5 run on apollo (the machine `ale validate` runs on), 6-16 on apollo-pc-2; the records go to
   `selection_a4/seed_<seed>.json`.

The frozen oracle (`4549238`):

```
f3a03a97db9fbe7697bf711c8c602492fe89d5142b0d1e57be6b9b2bda5752fd  tests/house_cell/cell.py
d88e94a34316fa9d9590eedf4e40b7f35ed0b893a0672c08e890f7854f1f762c  tests/house_cell/cleaning.py
5c9046ada48dc935538340de1eaceab078ae941de25a1bf6fd2b12aa3ffd0572  tests/house_cell/control.py
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  tests/house_cell/__init__.py
93c439787b73108e64125326aa47a21fbc0cd3009e59a1b9a88a1f56e536f8d1  tests/house_cell/layout.py
2814c5419023f70edc9e9507029be8583da7475138b9cc5da0fbd8af648ac272  tests/house_cell/objects.py
5b9a26199f1e692bd6e0d4beb34699ac8fcf110920069dc5736fb01e33f94198  tests/house_cell/physics.py
03880c6204b6bdd7e3412e93a7b9d85ab501dfe84e6c2b9efce2f3fb44ce0c49  tests/house_cell/props.py
b9a30b5b63ca43cbbd4dd8fa3ef4c3da261c120a6da3fa437d65fbb12b853b68  tests/house_cell/reference.py
5b231503b38139893a887c541c0bd8fa0cf1f0ee8272f6f6c9e7d80ecdd1cf45  tests/house_cell/render.py
223b284fd0ce69c6ed3f6b0d7f2348d67e0bcf150acdd70e567c8e6e0e7198e8  tests/house_cell/robot.py
59015301eb370fdd8e965d65a26c4ae01af0b7b32b5b9821094399a43a1ff1ed  tests/house_cell/room.py
8f3413807a8547ec05e56d65de6287457195cdf522b8b2c1c59e1538a3f46be8  tests/house_cell/scene.py
fb9c83456a9275d0555da22f959c35ec44aa5c4909c72fc10e7de9cda5024cc4  tests/house_cell/scoring.py
3a74be8b8c5a00b00367bb9d71f21e0c5eef0350c6a68e49cffce7f94ced1f36  tests/house_cell/server.py
dbac8d1a656f5d93bc0354d8748091aab64b7167ed4d85d08b6f4968d5c1f729  tests/house_cell/walk_core.py
4dcb59ed702796d95e442aafdaffdad191036f3008bc54e1cb783892be9b4fa7  tests/house_cell/look.json
4f76a26e1fee4147327880b7a41d7e0e16855f7be57a8b8310a1353d80ebddc6  solution/solve.sh
7e224c2848476398821fce93c03244c0b6e45274dc60a5e32742c574df789e9b  solution/reference/clean.py
9a4c6c83b2d73f3b2fe5fffb1656ec250edbd11d3aefe5f25084699729208939  solution/reference/oracle_token
e88f7bc29cc3bdf8cffd1ed17e854295e0b73f25164982b0a11682c0719d1038  tests/grader_config.json
```

### Results (amendment 4; each candidate run once, 2026-11-11 15:26-16:20Z, the oracle at `4549238`, the image built from environment/ at `296e8d1`)

| # | seed | stage score | crumbs binned | sim s | wall s | machine | result | why |
|---|---|---|---|---|---|---|---|---|
| 1 | 6790790475345571576 | 0.9167 | 0.600 | 2551 | 2183 | apollo | pass (kept) |  |
| 2 | 3125492756787262100 | 0.9537 | 0.733 | 2875 | 2746 | apollo | pass (kept) |  |
| 3 | 7022738416721159354 | 0.9213 | 0.617 | 3095 | 2889 | apollo | pass (kept) |  |
| 4 | 4089632381925228296 | 0.9120 | 0.583 | 3163 | 3223 | apollo | pass |  |
| 5 | 6569624632418969883 | 0.9352 | 0.667 | 2842 | 2754 | apollo | pass |  |
| 6 | 4920960843796068851 | 0.9213 | 0.617 | 3081 | 1336 | apollo-pc-2 | pass |  |
| 7 | 7640796830471677594 | 0.9306 | 0.650 | 2981 | 1097 | apollo-pc-2 | pass |  |
| 8 | 7981839151774950588 | 0.9074 | 0.567 | 3005 | 1328 | apollo-pc-2 | pass |  |
| 9 | 4745791118675327858 | 0.9398 | 0.683 | 2854 | 1918 | apollo-pc-2 | pass |  |
| 10 | 376632851020783183 | 0.9306 | 0.650 | 2910 | 1259 | apollo-pc-2 | pass |  |
| 11 | 3945038779305284334 | 0.8031 | 0.667 | 2265 | 2091 | apollo-pc-2 | fail | damage: {'broken': ['plate_1'], 'on_floor': ['plate_1']}; short: dishes 0.25/1, grease 2.43/2.5 |
| 12 | 4387728151438419368 | 0.9259 | 0.633 | 2780 | 1355 | apollo-pc-2 | pass |  |
| 13 | 6432759289746745107 | 0.9722 | 0.800 | 3258 | 1733 | apollo-pc-2 | pass |  |
| 14 | 18184323288814303 | 0.9630 | 0.767 | 2825 | 1384 | apollo-pc-2 | pass |  |
| 15 | 6438526276579322594 | 0.9444 | 0.700 | 2771 | 1135 | apollo-pc-2 | pass |  |
| 16 | 2636808825281067254 | 0.9352 | 0.667 | 2964 | 1205 | apollo-pc-2 | pass |  |

15 of the 16 pass. The one failure is the reference's own: in 11 neither plate found a way onto the dish tray (the
first met the counter's edge by 1.2 cm on its way in, the second and then a glass found no stance), each was let go
at the counter, and the first fell to the floor and broke. In the first run the plates of 11 reached the tray; this
time, after the step back with the raised plate, the robot carried it turned by 0.5 rad and walked to the counter
another way. Nothing was changed for it: the selection stands as written. Of the four that failed the first run, 4,
10 and 12 pass now, and in 11 no glass went off the table. Per-candidate records: `selection_a4/seed_<seed>.json`.

**The hidden set** (`tests/hidden/seeds.json`, unchanged): candidates 1, 2 and 3, stage scores 0.9167, 0.9537 and
0.9213 (mean 0.93056 = the anchor 0.9306). The first three passing candidates satisfy the rule as before: the
recycling bin stands in the south slot in 1 and in the north slot in 2 and 3, and most of the floor's crumbs lie south
of the table in 1 and 3 and north of it in 2. They are amendment 1's and amendment 3's hidden set, and the hollowing
check is amendment 1's. In all three no safety violation, nothing broken or dropped; trash, dishes, books, grease and
the finishing credit are full, the crumbs 60, 73 and 62 % binned; the peak robot contact is 72 N (in 3, a knee
brushing the dustpan hanging by the bins; the limit is 150 N held for 0.2 s).

**Wall clock.** The reference's episodes took 2183-3223 s of wall clock on apollo (candidates 1-5; 64 CPUs shared with
other work) and 1097-2091 s on apollo-pc-2 (6-16); their program time, the wall clock less the cell's own time, is
less. The oracle's budget is 10800 s of program time per episode (`privileged_program_time_cap_s`) and the
verifier's 36000 s for the three. The longest pause between two log lines of a run was 120 s: the simulation
advances throughout, and no planner search needed bounding.

### The anchor (amendment 4)

0.9306, the mean of the kept selection runs, with `full_at` 0.98 (unchanged). `ale validate` from a clean checkout of
`ec3475b` (validate-fad4682d, 2026-11-11, apollo, engine bfb7505, a separate verifier sandbox of 8 CPUs): `ok`,
untouched reward 0.0, oracle reward 1.0. The oracle's three episodes repeat their selection runs to the last digit:
cleaning_score 0.93056 (ratio 0.99995 to the anchor), mean simulated time 2840.13 s (the selection runs': 2550.60,
2874.56 and 3095.24 s), no safety violation (peak robot contact 72 N), no broken dish. The reference's program time
averaged 2486 s an episode (budget 10800 s); verify 9953 s (budget 36000 s), the three episodes one after another
on a busy machine.

## Amendment 4, addendum (2026-11-12): the bins drawn open

The bins' drawn body was a closed rounded box (`house_cell/objects.py`, `add_bin`), while their colliders are open
walls: with a lid open the agent cameras (and the review's video) showed a solid top where the bin is open, and litter
vanished into it. Changed, before any further run:

* `objects.py`: the same rounded box without its top face (its walls, bottom and rounded edges vertex for vertex), a
  rim of the walls' 10 mm and the inner walls down to the bottom collider's top face (`_open_shell`); the inside drawn
  by a body of its own added after every other body of the cell (`add_bin_inside`, `scene.py`), so that every
  earlier geom keeps its index;
* `props.py`: the inside's texture, the same plastic under a name of its own (`bin_blue_inner`, `bin_grey_inner`);
* `look.json`: the inside's colour gains and no specular for the inner walls and the inside's floor (Filament draws
  no ambient occlusion), fitted to OVRTX stills of the kitchen ceiling camera and the head camera at three states of
  development scene 11 with the lids open (OVRTX over Filament, linear luminance: inner walls 1.013 blue, 1.003
  grey; the bins' outsides 0.938 and 1.010, their lids 1.018 and 0.951 as before; the inside's floor below 0.001 in
  both). Before the change the inside drew 3.7 times as bright as the path-traced look.

Only drawn geometry, textures and the look changed (contype 0, mass 0, no joint): development scenes 11, 12 and 13
run again with these files repeat their earlier episodes exactly (every recorded joint position of every snapshot
equal, the same call trace, the same stage scores 0.9491, 0.9120 and 0.9167). The oracle frozen at
`4549238` and this selection therefore stand; `ale validate` on the head (below) repeats the selection runs. The
changed frozen files' sha256:

```
1472841a218e87e689ae388ebbcb19197ccbf8c4be785642af40ca52d06c86be  tests/house_cell/objects.py
e259915790e7db2a79007f65cab10c1c5a294d2b26ed5ee457f923236ccf2c7d  tests/house_cell/props.py
3451cf5efd9ac9c5c7ee71b8b09a5b4c27cd1db2529a25dbebf57aaa54447955  tests/house_cell/scene.py
9ab531a5f05549eb2edcdeb96b1091ca245e16f121fa79aac18d1531fa3596a8  tests/house_cell/look.json
```

With the lids shut the survey photos are unchanged to within the sensor noise: re-rendered with the open bins they
differ from the given ones in a thin band along the bins' top front edge (1540-2407 pixels of the kitchen ceiling
camera's frame and the head camera's at the bins, by 1-3 grey levels in most, at most 26; the depth by 1 mm at no more
than 4 pixels), against up to 113 pixels and 27 levels between two renders of the old art; they stay as given.

`ale validate` from a clean checkout of `28248d8` (validate-04d71f3c, 2026-11-12, apollo, engine bfb7505): `ok`,
untouched reward 0.0, oracle reward 1.0, the oracle's three episodes the selection runs again to the last digit
(cleaning_score 0.93056, ratio 0.99995 to the anchor; mean simulated time 2840.13 s; peak robot contact 72 N; no
safety violation, no broken dish); verify 8673 s.
