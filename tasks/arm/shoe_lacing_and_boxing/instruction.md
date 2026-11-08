# Lace, box and ship pairs of sneakers with a two-arm finishing cell, seen only through its cameras

You are the automation engineer of a children's-sneaker factory's finishing line. Lacing is the last manual step of
the line: a pair comes off the last with the lace threaded through the toe-most eyelet pair only, and someone has to
lace the shoe criss-cross, pull it snug, erect the flat shoe-box blank, put the pair in, put the lid on and send the
box on. Today the cell does it: two Franka Panda arms on a bench, the LEFT one with a parallel-jaw gripper, the RIGHT
one with a lacing hook (a thin shaft with an L-foot and an upturned tip that catches a lace aglet), a lacing cradle
that clamps a seated shoe, an infeed tray where the pair and its lid are set down, a blank spot where the flat
box blank lands, and an outfeed line. You see the cell only through its cameras: an overhead, a front and a side
RGB-D camera, and a wrist RGB-D camera on each arm. Their calibration (intrinsics; the fixed cameras' poses; the wrist
cameras' mounts) is exact and published in the cell manual, and so are the batch's dimensions (the shoe model, the
box). Nothing on the API tells you where a shoe, an eyelet, a lace end, a box panel, the lid or the cradle is, whether
a lace is through a ring, whether a wall has locked, or how much credit you have earned: you must perceive it.

The work order of an episode is one pair of shoes; for it the cell must:

1. **Lace** each shoe: seat it sole-down, heel-back in the cradle (the cell clamps it; `release_clamp()` frees it),
   then thread each lace end through the remaining eyelet row(s) criss-cross: from row 1 (toe-most, already threaded,
   both ends leaving the shoe outward) a lace end enters row 2 on the OTHER side from the outside and comes back up
   out of the cavity through the throat slot (a batch with more rows continues: row 3 on its own side from the
   outside, up through the slot again, and so on: every ring is entered from the outside, the sides alternate), and
   pull the lace snug after each pass. The shoe is a hollow lasted shell: the eyelet flaps are rigid and the cavity
   under them is reachable only through the throat slot, which is where the hook works (it fetches the lace end from
   under a flap and lifts it out through the slot). Both aglets must end up outside the shoe.
2. **Erect** the box blank: fold the four walls up until they snap-lock (87 degrees held for 0.3 s locks a wall at
   90 degrees).
3. **Pack** the laced pair: both shoes inside the box, sole down, heel-to-toe (one shoe pointing each way), nothing
   resting on the walls.
4. **Close** the box: fetch the lid tray from the infeed tray and set it over the erected walls (centred within
   12 mm, aligned within 6 degrees, resting).
5. **Ship**: push the closed box with the pair inside across the outfeed line; the cell takes it away, which completes
   the work order (a box without both shoes inside stays where it is).

One batch per episode: every pair has the same shoe model (length, width, two eyelet rows, spacing, throat gap) and
box size, published in `describe()["objects"]`. Per pair the colours vary, the shoes and the lid land on the infeed
tray either way round with a random turn (a shoe may lie on its side) and the blank lands either way round. The lace
is tacked at row 1 (stitched to the insole between the two row-1 rings), so each tail keeps its length.

## What you produce (in `/home/user/submission/`)

`packer.py` exposing `solve(cell) -> None` (plus any helper modules, optional `run.sh`). The grader runs it once per
hidden episode (6 episodes) in its own physics cell. It talks to the cell through `cell` (API in
`~/given/cell/cell_api.md`): the arms' own state (joints, tool poses, gripper width and grasp signal, the clamp and
the hook clip's sensors) and the five RGB-D cameras; no object poses. Per episode: 900 s of simulated time; 1200 s of
program time, which is the episode's wall clock minus the time the cell spends simulating (inside `step`) and rendering
(inside `get_camera`), so it measures your program's own computation and its other requests; at most 120 s from a reply
to your next request; 450 camera frames in total.

How the program is run: a fresh process under a dedicated unprivileged account that is not yours, from a per-run
directory with a read-only copy of your submission folder (regular files only), an empty writable HOME / TMPDIR, a
minimal environment (`PATH`, `HOME`, `TMPDIR`, `LANG`, `MUJOCO_GL`, `PYTHONPATH` = the given library `~/given/lib`,
`ARB_GIVEN_DIR` = a read-only copy of `~/given`) and the launcher `python3 -B -E -s -c <bootstrap> <lib> packer` (or
`bash run.sh`). Resolve given files through `os.environ.get("ARB_GIVEN_DIR", os.path.expanduser("~/given"))`, write only
under the run's HOME / TMPDIR. A program that crashes ends that episode with what it achieved; the other episodes still
run. Your submission must be final when your session ends: the submission folder is then copied out of your sandbox
and graded in a fresh sandbox that holds nothing else of yours (no other file, no process you started), and a
surviving process of the run account voids the contract gate.

## What you are given

| path | content |
|---|---|
| `~/given/cell/cell_manual.md` | the integrator's cell description: bench, the two robots' base poses and tools, the cameras with their exact calibration, the zones (infeed tray, blank spot, cradle, outfeed line) with their nominal positions and the per-episode jitter, what a shoe, a lace, a box blank and a lid are (geometry, eyelet layout, colours), the attach rules of the gripper and the hook, the cradle clamp, the carton's snap-locks, the lid rule, safety rules, budgets |
| `~/given/cell/cell_api.md` | the cell's programming manual (the wire API) |
| `~/given/cell/cell_config.json` | the cell's configuration as far as it is public |
| `~/given/lib/shoecell/` | the cell's library: `client` (the proxy, `decode_frame`), `local` (`LocalCell`: the same cell in your own process on a seed of your choice, with `truth()` for development), `practice` (the wire), and the cell's own code (`physics`, `control`, `cameras`, `episode`, `server`): the grader runs this exact code on hidden seeds |
| `~/given/panda/` | the robots' MuJoCo model (Apache-2.0) |
| `~/practice/grade.py` | runs your program over the real wire against the practice cell on a visible seed and prints the raw work score with the per-pair milestones |
| `~/example/packer.py` | a complete, policy-free program (reads frames, moves both arms, finishes; scores 0): the starting point |

Tools: Python 3.12 (`/opt/venv/bin/python3`; call it by that path, the bare `python3` on your shell's `PATH` may be
another interpreter) with numpy, scipy, OpenCV (`cv2`, headless), Pillow, MuJoCo 3.14. No network in your session.

## How you are graded (verify runs 6 hidden episodes)

Gates (multiply the score): `contract_ok` = `packer.py` is a regular file that parses and defines `solve`, passes the
route gate (no weight files, no training calls, submission <= 60 MB), imports under the run regime above, makes at
least one `step` call over the episodes, causes no network activity during any episode, and your own `.py` files
import no networking module (`socket`, `ssl`, `urllib`, `http`, `requests`, `asyncio`, `ctypes`, `subprocess`, ...),
load no module under a computed name, execute no computed code (`exec` / `eval` / `compile` of a non-literal) and start
no other program (`os.system`, `os.exec*`, `os.spawn*`, ...). `safety`: a Panda link other than the gripper's finger pads or the hook pressing more than 30 N on
the bench, the cradle or an object, or any contact between the two arms above 30 N, for 30 ms (three control ticks) in a
row, or an object faster than 3 m/s, is a violation that voids the credit of the pair being worked on when it happens (even if that pair is later shipped). A
physics fault (the simulation diverging under a violent action) ends the episode with the credit of the pairs shipped
before it.

`work_score` = the mean over the episodes of the accepted work of the episode's pair, which can earn
1.0 = lacing of shoe A (0.30) + lacing of shoe B (0.30) + box erected (0.12) + pair packed (0.13) + lid closed (0.10)
+ box shipped (0.05), judged on the cell's own physical state when the box is shipped or when the episode ends, and
0 for a pair during which a safety violation occurred. The work order is a laced pair in a shipped box: "packed",
"closed" and "shipped" are credited in proportion to the pair's laced shoes (times 1/2 with one shoe laced through every
row, times 0 with none), so a pair boxed unlaced earns the box-erected credit only. The
lacing of a shoe = (rows laced in order from row 2, both ends, correct side, entered from the outside, counted
consecutively until the first miss) / (rows - 1), times 0.7 + 0.3 x (the fraction of those rows that are snug: no more
than 50 % slack between consecutive rows, plus 3 cm for the dip under the flap and the return through the throat), and
0 if an aglet is left inside the shoe (passed in and never brought back out). "Packed" needs both shoes inside the
erected box, sole down (within 15 degrees), heel-to-toe (headings within 30 degrees of opposite), not resting on a
wall. "Closed" needs the lid on the box as above with both shoes inside, sole down, not resting on a wall (heel-to-toe
is the packing's alone). "Shipped" needs such a closed box with its floor centre across the outfeed line. The reward you are judged on is `work_score` relative to a hidden
reference value. Stats (not scored): per-milestone means, frames used per camera, sim / wall / program time, illegal calls,
safety events.

## Rules of the cell (details in `cell_manual.md` / `cell_api.md`)

* The gripper grasps what its documented attach rule allows: closing the fingers on a grasp handle (an aglet, the
  heel counter or the toe cap of a shoe, the free edge of a box panel) that lies within the pinch box between the
  finger pads attaches it to the hand at its current pose (an aglet is seated along the approach axis); opening
  releases. The hook's clip catches an aglet within 12 mm of its tip and seats it along the tip; `hook(False)`
  releases; closing the gripper on an aglet that sits on the hook takes it off the hook. Neither tool reports what it
  holds beyond the gripper width and the clip sensor.
* A shoe set down sole-down, heel-back in the cradle seat is clamped 0.3 s after it comes to rest; `release_clamp()`
  frees it. The clamp never reports.
* A wall of the blank that reaches 87 degrees for 0.3 s snap-locks at 90 degrees. Locks never release. Nothing
  reports a lock, and nothing reports whether the lid sits right.
* Calls other than the documented public ones answer `{"error": "forbidden"}` and are counted.

## Practice

`LocalCell(seed=...)` and `~/practice/grade.py` give you the same cell on seeds of your choice, with `truth()` (every
pose, the eyelets, the lace passes, the carton state, the milestones) for development only: the graded cell never
answers it. The hidden episodes use other seeds; the cell builds their scenes with the same code.

## Notes

- Photos and camera frames are meant to be processed programmatically; view only the images you need.
- Do not modify the given files in your home directory; the grader reads its own copies.
- Environment facts (harness limits): a single tool call whose input runs above roughly 25-30k tokens is dropped by the
  harness (it is never executed and no error comes back), and a response that reaches the 64k output-token limit is lost
  and, after at most a few retries, ends the session with whatever is in `/home/user/submission/` at that moment. Write
  large files in several parts (one function group per write), keep every single response well below that limit, write
  a first complete submission early and refine it.
- Session time: this session is stopped after 4 hours of wall-clock time from its start. A session stopped by that limit
  is not graded at all, so a late finish scores nothing. Note the start time with `date` when you begin, keep a complete
  submission in place from early on, and end the session yourself (stop working and say that you are done) at least
  15 minutes before the limit.
