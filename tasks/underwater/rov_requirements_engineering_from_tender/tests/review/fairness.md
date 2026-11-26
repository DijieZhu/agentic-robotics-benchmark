# Fairness: nothing but the work scores, and the grader scores only what is right

This file describes the task as this folder holds it. Evidence entries are named by their id in `evidence_index.json`
(LADDER, QC, WALK, SCAN, DESK, WORLD). Every score below is the formal grader's, on the hidden world of this folder;
the weights are verdicts 0.35, award 0.25, site predictions 0.10, labels 0.05, schedules 0.25, and the reward is
`min(1, score / 0.98)` (`tests/anchor.json`). Each row of section 1 was written from the reference with the stated
consideration handled wrong in the way a careless reader would (the wrong reading written into a copy of the world,
the submission written from that copy), and the rows recomputed on the assembled folder equal the rows on the world,
number for number. Apart from section 8 and the last item of section 6, which report the one frontier-model episode,
nothing here comes from an agent episode of the task; the verify times of section 6 are wall-clock measurements of the
runs named there; the readers of the walk were AI review agents that took no part in building the task, and every
number they quote was recomputed by code.

## 1. What scores and what does not

The considerations are those of `completability.md` (section 3). Scores, with the reward in brackets (LADDER):

| row | score |
|---|---|
| all right (the reference) | 1.0 exactly |
| the site as found missed: the open height taken from the drawing, everything else right | 0.7357 (0.7507) |
| the award-position bid as built missed: its schedule written for the unit as tested | 0.6875 (0.7015) |
| the same, with that bid's site predictions for the unit as tested as well | 0.6849 (0.6989) |
| both desk-only facts missed | 0.4212 to 0.4232 (0.4298 to 0.4318) |
| the deck missed: not raised at all / raised on half the items | 0.4829 / 0.6298 (0.4928 / 0.6427) |
| a contract that can be executed missed | 0.6375 (0.6505) |
| the second maker's resistance missed | 0.5710 (0.5826) |
| the works' day missed: both parts / the ballast part / the needless clauses | 0.575 / 0.625 / 0.900 (0.5867 / 0.6378 / 0.9184) |

The two desk-only facts are additive: each costs the same whether the other is met or missed, to within 0.001 of the
score (the site as found 0.264, the award-position bid as built 0.312 to 0.315). Where any other consideration is
missed whole, the award term is 0: each moves the award or the purchase that follows it. The site as found does not
move the award, by construction: the bid it decides (T-06, the bid with the protective cage) is priced above the
award, and the award-position bid's delivery band contains no unit whose fitness depends on the open height; the
miss costs that bid's verdict, part of the site predictions and of the labels, and part of the protection of three
other schedules (0.264 of the score).

## 2. The roads that must score full, and the reported choices

Scored 1.0, as they must (LADDER; WALK):

- the award-position bid's schedule as an air-mass window with a displaced-volume window, both for the unit as built;
  the same in displaced volume alone, or in air mass and displaced volume; a schedule written from that bid's
  witnessed weighing alone, without its as-built reply;
- T-06 rejected on the one ground that holds for it (N-03, station keeping at the open height).

Reported beside the rows, with no limit:

- a conservative allowance for a clogged rack, made without asking anyone (T-06 rejected on its ground; the other
  schedules for the rack as drawn): 0.8422 (0.8594). Every verdict is right; the schedules, the site predictions and
  the labels pay for the guess;
- T-06 rejected on N-03 and on a second ground that does not hold for it: 0.8875 (0.9056);
- the award-position bid rejected for its change after bids closed (on GF, PART-D or N-01: the same score), the award
  moved to the next accepted bid: 0.7823 (0.7983). The tender's list of grounds is closed and a change after bid close
  is none of them; as built that bid is fit in every configuration and contractable;
- that bid's site predictions for the unit as tested, everything else right: 0.9974 (reward 1.0): one of its eight
  values (the launch mass margin) moves by about four tolerances (WALK).

## 3. The grader scores what is right and only that

- Degenerate submissions score 0: untouched, empty files, accept all, reject all on GF, on PART-D or on N-01 (LADDER).
- The first form's reference scores 0.0 on this folder: as it is (its schedule file has another name, so the
  structure check G0 fails), with its schedule file renamed (the schema refuses its form), and its evaluation alone
  (its option ids are not this folder's) (QC).
- A schedule entry for a bid the evaluation does not accept is ignored; an accepted bid without an entry has an empty
  schedule, which protects nothing (`tests/NOTICE.md`, "The score").
- The grader reads the three files as static JSON and recomputes the truth table (32 configurations x 12 needs), the
  sets, the Part B labels and the delivery band of every fit configuration (8 bands of 660 units each) from the world
  files on every run; a difference between the recomputed and the stored tables is a task error, never a score.

## 4. The independent readers' walk

Three readers who wrote nothing of the task, AI review agents independent of its build, each walked a fixed list on
the assembled folder with the hidden truth in hand and recomputed every number with the task's own grader and public
procedure; each reader's outputs were reproduced byte for byte by a second run (WALK).

- **The blocked bay** (7 items). The previous pilot's reply meets each of the eight conditions set for it (a plain
  observation in his own words, no verdict, no flow figure, nothing that implies that other staff know); the open
  height has one reading, 9.0 m, at which T-06 fails its station-keeping need by 5.5 tolerances, while at the drawn
  12.0 m it passes by 4.5; the fact's words occur only in the pilot's reply and his sketch (1 878 hits read, 22 of them
  in those two places, no other place carries the fact); the 2025 dive log and telemetry are true at their date
  (station keeping 0.99 to 1.00 of the need with four bays open; with three it would be 0.67); the spring flood of 2026
  is the largest daily rise of the stage record, so it is found without knowing its date.
- **The award-position bid as built, and the printed bench conditions** (9 items). The as-built reply meets each
  condition set for it (complete, the works' re-weigh sheet attached, no verdict); the sheet's figures recompute;
  every dated document of that bid is true at its date and every present-tense reply is true now; each of the three
  roads (the as-built reply, the trim reply's pointer, the witnessed weighing) gives the fact; the two thrust reports
  that now print their bench supply and battery current limit do so in their own report forms' style.
- **What the two facts must not have moved** (7 items). The 87 files on the routes of the other four considerations:
  80 byte-equal, and in the other 7 every value a route reads is unchanged; the instruction and the desk guide read
  whole against this world: no sentence false; no agent-visible file states a number of attachments; the reference's
  seven requests and a fuller plan of twelve are all served.

No item needed a change. The readers' 29 notes, each with its consequence, are in the maintainers' records; those
that a reviewer may meet are under "Known limits" below.

## 5. Nothing leaks to the agent

- Eight scans over everything the agent can read (78 files, 77 page images and their drawn strings, four
  attachments, 250 desk texts and the desk's greeting), each with its planted controls (25 of 25 caught): 0 hits
  outside registered exceptions (SCAN). The leak-word scan (103 shared words, 127 task words, 148 part names, 52
  hidden key names) and the routing-word scan (the words that say how the tether is led across the deck, searched
  outside the vehicle card, which is allowed to carry them) have 0 hits; PNG files carry only pixel chunks; public
  code holds no hidden or catalogue number; the example numbers of the documents match no reference bound.
- The registered exceptions, each with its reason: 20 entries in four scans (306 places in the digest-literal scan,
  292 of them digits of the generated logs that equal a digit string of the digest; 15 in the numeric scan; 14 public
  names in the schema-key scan; 11 in the leak scan). The numeric scan's seven entries are digits shared by unrelated
  quantities (a tank's width and a clarity statistic; a mass in air declared in three compliance matrices and a bound
  of the reference's schedule), a length on the vehicle card that equals the open height in its digits, the generic
  nominal voltage of a four-cell pack (printed on the second maker's two pack pages and as one thrust report's bench
  supply), a resistance on the first maker's range page that equals a reference value of the shared digest, and a
  frequency label.
- The two desk-only facts: their words and numbers (the blocked bay, the flood and the look, the open height; the
  float, its figures, the re-trim, the re-weighing) were searched over all 453 units of the folder, 407 of them
  agent-visible (the text files, the drawn strings, the desk's replies and witnessed outputs, the attachments). In the
  agent-visible units there are 489 hits: 60 in a holder's own place (the previous pilot's reply and sketch; the
  award-position bidder's as-built and trim replies, its re-weigh sheet and its witnessed weighing) and 429 in
  contexts of 15 classified kinds that carry neither fact; 24 statements there that could contradict either fact are
  each in a holder's place or classified; 0 unclassified.
- The words that would announce a consideration (13 words) appear only where a document carries that consideration's
  facts (219 places) or in 139 registered places; beyond them the scan reports three hits, all three in the replies
  and attachments that hold the two desk-only facts (the previous pilot's sketch, the award-position bidder's re-weigh
  sheet and its trim reply), each read and accepted.
- Platform, maker and author names appear only on the grader's side: `tests/NOTICE.md` and
  `tests/digest/spec_values.json`, a byte copy of the shared digest that the grader pins.
- `setup.sh` checks each of the 77 page images in the agent's home against `environment/setup/page_images.sha256`
  (root-only, never in the agent's home) and fails when one is missing or differs.

## 6. The worst case and repeated runs

- The largest legal submission (three files of exactly 128 kB, 12 clauses on each of the four schedules the reference
  reads, 48 clauses read) scored reward 1.0 in each of three runs through the engine's separate verifier, with a
  verify phase of at most 6.84 s of the 600 s limit (QC).
- Each submission scored the same in each of its runs: the reference 1.0 and the untouched submission 0.0 in both
  validates, the largest legal submission 1.0 three times, and the frontier-model episode's saved submission the same
  reward when graded again in a fresh container.

## 7. The first form's public files

The form in which the task was first merged is public in the repository's history (its documents, its desk table and
its reference). The checks against it:

- its public site record against this folder's: ten quantities, each apart by at least five tolerances of the need it
  enters or by 5 % (SCAN);
- its reference and its evaluation score 0.0 here (section 3);
- the documents of both forms follow the same templates (voice, report forms, dates), and the Part B labels of this
  folder's reference equal the first form's reference labels (24 of 24; weight 0.05). What remains is recorded in
  `tests/NOTICE.md`; the agent's sandbox has no network and none of the first form's files.

## 8. The frontier-model episode

One episode of a frontier model with a coding-agent harness at high reasoning effort ran on a commit whose payload,
instruction, desk tables, world, reference and grader are byte-identical to this folder's, under the instruction's own
limits (4 CPUs, 8 GB, 4 hours, no network, 14 desk requests) and with the harness's web tools switched off. It ended
by itself after 22.6 minutes with 10 requests spent: score 0.410, reward 0.418 (the engine's 0.42); verdicts 0.75,
award 0, site predictions 0.55, labels 0.38, schedules 0.29. It accepted T-06 and wrote the award-position bid's
schedule for the unit as tested: both desk-only facts were missed, neither was asked for (the previous pilot was asked
about launch and recovery only, the award-position bidder not at all). It raised the open-height question itself and
settled it from the civil engineer's reply on the drawings and its own back-calculation from the 2025 telemetry, both
true at their date; it used the as-built topic once, on another bidder whose papers disagree. AI review agents that
had no part in the build read the episode against the files and the hidden world: no item on which an engineer
following the given files is paid less than full, no carrier of either fact in anything the agent saw, nothing read
outside its home.

## 9. Known limits

- TS-01 A.3 establishes the open area "from the Owner's drawings and records"; a literal reader could hold that the
  previous pilot's reply is not one of them and keep the drawn 12.0 m after asking him. Such a reader scores as the
  site as found missed (reward 0.7507); TS-01 C.1 asks fitness "as the site is in the inspection month", and the
  pilot's sketch is his record of the rack.
- The pilot's reply says bay 1 is packed "with sunken logs and silt to the top of the bay"; the drawing shows a debris
  sump below the sill, and no file states the sump's state. TS-01 A.3 fixes the lowest point from the drawings and the
  civil engineer's scope reply names the sump floor; a reader who moves the lowest point to the top of bay 1 still
  scores 0.9875 (reward 1.0) with everything else right.
- The stage record's storm of spring 2026 repeats the rainfall figures of the earlier storm (2021), and after the flood
  the level sits 1 cm under full supply for 18 days; no task value reads either.
- The award-position bidder's as-built reply gives no reason for the float and does not say whether its price
  changed; its delivery statement codes grades B and C of its pack with one code and lists one current-limit preset
  below the others'.
- The desk guide says a witnessed test runs on "the unit as tendered, with none of the priced options fitted,
  ballasted as the works left it": for the award-position bid the witnessed weighing returns the unit as built, whose
  mass differs from its first test report's, a visible difference that leads to the as-built question. The submission
  contract's "in the state in which it is tendered" read as the state at bid close costs at most that bid's site
  predictions (0.9974, reward 1.0).
- A conservative allowance for a clogged rack, made without asking, gets every verdict right (0.8422 above); the
  desk-only fact then decides only the schedules, the site predictions and the labels.
