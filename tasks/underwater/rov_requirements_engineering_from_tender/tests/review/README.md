# tests/review: completability and fairness evidence (grader-side; never staged into the agent's home)

The evidence that this task can be done from what the agent is given, that no shortcut scores, and that the grader
gave every submission graded more than once the same score each time (`fairness.md`, section 6). This folder replaces,
as a whole, the form in which the task was first merged (2026-11-17, #91): the hidden world, the bids' documents and
test reports, the tender, the site drawings and records, the reference's evaluation and schedules and two of the three
submission schemas were rewritten; the instruction, the submission contract, the desk guide and the desk's tables were
revised (the deliverable is now one acceptance schedule for each accepted bid, and a rejection may also cite the
grounds `GF` and `PART-D`; 167 of the desk's 248 replies differ from the first form's, 79 with a new text and 88
witnessed tests that keep their text and return new measured values, and all four of its attachment pages are new);
six catalogue pages were added. 37 of the 154 files of the agent's home are byte-equal to the first form's (22
catalogue pages, 10 CTD casts, the incident note, the public procedure and its description, the desk client and the
third schema), and so are the reference's Part B labels. The first form's files stay in the repository's history and
are used here only where `fairness.md` says so ("The first form's public files"). The full records are kept by the
task's maintainers and are not published with the benchmark; this folder states what they found and the numbers that
carry each verdict. Its index pins the sha256 of 34 of the task files the numbers were computed on (`task_files`; in
the published form also the files its `publication` entry names), and `check_index.py` recomputes exactly those, so
that anyone can check that they are still the bytes the evidence describes. The payload of the agent's home, on which
the readers' walk and the scans were also computed, is named only by the digest of the build tree it was assembled in
(`payload_tree_digest`), which `check_index.py` does not recompute.

| file | what |
|---|---|
| `completability.md` | why the reference submission scores 1.000 from recoverable information: what the reference is and how it is submitted, where the information is, what an expert must get right and the route to each (the two facts that only the desk gives among them), the desk requests an expert's plan uses, the desk on the real tables, the time it takes and the worst-case verify |
| `fairness.md` | why nothing but the work scores: the rows built from the reference (each consideration missed alone, the two desk-only facts missed together, the roads that must score full, the reported choices, degenerate submissions), the independent readers' walk, the leak, number and fact scans with their registered exceptions, the first form's public files, the one frontier-model episode, known limits |
| `evidence_index.json` | machine-readable index: every evidence entry (id, title, verdict, the rows with thresholds and measured values, the task files it was computed on) and the sha256 of the task files it pins (`task_files`); its `publication` entry records the move of the page images and logs to the benchmark's asset dataset |
| `check_index.py` | `python3 -I -B check_index.py`: recomputes the sha256 of the task files listed in the index and lists every one that changed since the evidence was taken, with the evidence entries it is an input of |

Every score and count is a deterministic computation on the hidden world, the formal grader and the build that
produced this folder, or a count made on the delivered files and on the verdicts of the readers' walk; the verify times
are wall-clock measurements of the runs named with them; none is a measurement of a vehicle. The readers of the walk
were AI review agents that took no part in building the task (`fairness.md`, "The independent readers' walk"). The
task's frontier check is one episode of a frontier model (the last section below); every file of this folder outside
`tests/` is byte-equal to the same file on the commit it ran on, except the revision line of the asset list
`environment/assets.sha256` (that commit's image took the same 101 asset files, each with the sha256 the list gives,
from a local mirror). The evidence quotes the episode only where it says so: the DESK row reports how the desk served
it (10 requests charged, each reply equal to its cell), and the episode's own passages report its result (here;
`completability.md` sections 5 and 6; `fairness.md` sections 6 and 8). No other verdict or number rests on it. The
world's numbers are fictional (`tests/NOTICE.md`).

## Verdicts at a glance

| evidence (id in the index) | verdict |
|---|---|
| WORLD: the build's checks of the hidden world (the needs' rows, the clause classes, the audits of the four accepted bids' delivery bands) | pass: 8 of 12 checks pass, 3 are registered deviations of one property of the award-position bid's delivery band (`tests/NOTICE.md`, "Known properties of the score"), the twelfth is the reference's score on the assembled folder (QC) |
| LADDER: 23 submissions written from the reference on this world (the reference, whose rejection of T-06 on its one valid ground is also a road that must score full; eleven single misses; the two desk-only facts missed together, in two forms; one other road that must score full; two reported choices; six degenerate submissions) | pass: the reference 1.0; every single miss costs between 0.10 and 0.52 of the score; the roads that must score full score 1.0; the degenerate submissions 0; the rows recomputed on the assembled folder are equal to the rows on the world, number for number |
| QC: the reference through `solution/solve.sh`, untouched and empty submissions, the first form's reference, the worst-case verify | pass: 1.0 on every term; 0.0; 0.0; through the engine's separate verifier at most 6.84 s of the 600 s limit |
| DESK: the desk's tables, checked by the desk checker and the build's desk tests, and the desk serving the frontier episode in the task image | pass: 17 of 17 table checks, every planted fault caught; the build's tests pass as registered; in the episode 10 requests charged of 14, each answered from its own cell |
| WALK: three independent readers (AI review agents) with the hidden truth in hand on fixed lists: the blocked bay; the award-position bid as built and the printed bench conditions; what the two desk-only facts must not have moved | pass: 23 items, no change needed; 29 notes, each with its consequence; every number recomputed twice with the task's own grader and procedure, byte-equal |
| SCAN: leak words, routing words, numbers, examples, public code, digest literals, PNG metadata, schema names; the words and numbers of the two desk-only facts; the consideration words | pass: 0 hits outside 20 registered exception entries; 25 of 25 planted controls caught; every word or number of the two desk-only facts in a holder's own reply, attachment or witnessed test or in a context classified as carrying neither fact (0 unclassified); the consideration words: three hits, all in those holders' replies and attachments |

## How this form was built and checked

- The hidden world was fixed with its rows and its reference before any document was written. The documents, page
  images, logs and desk replies were rendered from it, and the folder was assembled by one build that runs from the
  world to the tests in one pass and checks that every step reads exactly what the step before it wrote; two such
  builds gave byte-identical folders.
- On the assembled folder: the reference through `solve.sh` scores 1.0 on every term, the untouched and the empty
  submission 0.0; the 23 submissions' rows are reproduced number for number; the eight scans pass with their
  controls, the scan of the two desk-only facts' words and numbers leaves nothing unclassified, and the scan of the
  words that would announce a consideration reports its three hits, each read and accepted (SCAN); the readers' walk
  ran on it and needed no change.
- On a fresh clone of the benchmark with the task in place, on a commit whose payload, instruction, desk tables, world,
  reference and grader are byte-identical to this folder's: the repository's checks pass; `ale validate` twice gives
  untouched 0.0 and reference 1.0; through the engine's separate verifier the largest legal submission's verify phase
  takes at most 6.84 s of the 600 s limit (one fifth: 120 s). `evidence_index.json` lists those runs beside its
  reports.
- One episode of a frontier model with a coding-agent harness at high reasoning effort, under the instruction's own
  limits (4 CPUs, 8 GB, 4 hours, no network, 14 desk requests) and with the harness's web tools switched off, on that
  commit: it ended by itself after 22.6 minutes, spent 10 of the 14 requests and scored reward 0.42 (0.418 unrounded;
  score 0.410); graded again from its saved submission in a fresh container of the same image it gave the same reward.
  Neither desk-only fact was asked for: the previous campaign's pilot was asked about launch and recovery only, the
  award-position bidder not at all (`fairness.md`, "The frontier-model episode").
