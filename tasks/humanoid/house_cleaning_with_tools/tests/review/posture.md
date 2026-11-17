# The reference's arm postures (2026-11-09)

The reference cleaner (`house_cell/reference.py`) is the oracle's route and the evidence that the job can be done with
this robot. Its demo videos showed the arms in postures no person would take: a plate carried with the arm swung back
and up beside the head, the sponge held under the faucet with the elbow over the shoulder, the hand wrung against a
wrist limit while wiping, a free arm that already hung at home lifted up and back before every walk. Measured over
whole episodes (below), one arm or the other was in such a posture for 48-62 % of the time. Commit `1170d84` changes
how the reference plans its arms, nothing else: the cell's controller and its IK, which a program commands, are
unchanged.

## What changed

* **NaturalIK** (the planner's IK): the cell's damped least squares on the grasp site, plus a posture term descended
  in the arm's redundancy (the elbow's swivel about the shoulder-wrist line; with a position-only target the wrist as
  well), as a second task in the target's null space: the elbow at least 8 cm below the shoulder and not behind or far
  out beside the body, every joint at least 0.35 rad off its limits, the twisting joints (shoulder yaw, wrist roll)
  near their middle. The term acts only once the target is nearly reached, so whatever the cell's IK reaches this
  reaches as well (and the cell's IK answers where it does not). A path's waypoints change the posture by at most
  0.1 rad each, so the joint-space moves between them stay on the Cartesian line. Every stance check, path, carry and
  the constrained motions (a lid, the faucet lever, the sponge on the table) use it.
* **Carrying** (`canonical_carry`): a held item goes to the most natural of a few carry points and turns: in front of
  the chest at about elbow height (was shoulder height), a little nearer the middle or farther out, an item symmetric
  about its vertical axis turned about it; litter for a bin also beside the hip with the arm hanging (a can held from
  above dangles from the hand; carried at the chest it puts the elbow up), napkins and the apple core at any angle; the
  item kept clear of the robot's own body. Where a carried item would sweep down into the counter as the robot leans
  in, it is lifted first (`go_put`), as a person lifts a plate over the counter.
* **Grasps chosen for what follows**: an object's grasp frames are ordered by the carry each leaves (`carry_order`);
  the sponge's grasp and stance at its tray by the take's posture plus the wiping and rinsing postures its grip leaves
  (`sponge_work_costs`).
* **A free arm at home stays there**: `retreat` no longer lifts an arm that hangs at home (it swung it up and back
  before every walk).

## How it is measured (`tools/posture_run.py`)

One privileged episode per scene, run as the oracle's runs (`run_ref.py`), both arms sampled every 0.2 s of simulated
time. An arm is in an unnatural posture when its elbow is less than 5 cm below its shoulder (at or above it), a joint is
within 0.10 rad of its limit, the upper arm is swung out to the side more than 75 deg, or swung back more than 40 deg
(torso frame, from hanging down). The table gives the time either arm is so, and per criterion (seconds: elbow high /
joint limit / out / back), on development scenes 11-15, before (`6bc2ded`) and after (`1170d84`):

| scene | before: sim time | unnatural | per criterion | stage score | after: sim time | unnatural | per criterion | stage score |
|---|---|---|---|---|---|---|---|---|
| 11 | 3064 s | 1483 s (48.4 %) | 1052 / 725 / 282 / 627 | 0.9167 | 2518 s | 187 s (7.4 %) | 102 / 101 / 24 / 41 | 0.9028 |
| 12 | 4042 s | 2521 s (62.4 %) | 2169 / 748 / 1598 / 866 | 0.9074 | 3236 s | 249 s (7.7 %) | 147 / 134 / 36 / 34 | 0.9352 |
| 13 | 3659 s | 2113 s (57.8 %) | 1698 / 750 / 1305 / 797 | 0.7000 | 2829 s | 205 s (7.2 %) | 106 / 127 / 44 / 41 | 0.9491 |
| 14 | 3498 s | 1732 s (49.5 %) | 1364 / 803 / 436 / 702 | 0.9537 (safety violation) | 2405 s | 172 s (7.1 %) | 108 / 116 / 16 / 22 | 0.7100 |
| 15 | 3364 s | 1951 s (58.0 %) | 1625 / 777 / 1143 / 700 | 0.8677 | 3001 s | 215 s (7.2 %) | 125 / 106 / 39 / 37 | 0.9213 |

Over the five scenes: 55.6 % before, 7.3 % after. The episodes are also shorter (no swinging of the free arm, fewer re-plans).

What remains is short and mostly forced by the hand's orientation, which the object fixes: a grasp or a put with a
wrist near its limit for a few seconds (at the low coffee table, the dish tray, a bin), the left arm reaching across the
counter to the faucet's lever with the elbow level with the shoulder.

## Round 4 (2026-11-09): the arm's swivel resolved along every executed path

The round-3 video still showed the arms contorted much of the time. Contact sheets of whole development episodes, every
5 s from two views (in front of the robot to its right and from its left side), showed what the round-3 measure missed:
the elbow winged out or turned in about the shoulder-wrist line while the hand held an item level (a plate or the
sponge carried across the chest, the sponge at the pump and under the faucet, wiping), and the wrist bent hard. The
round-3 posture term acted only past thresholds (the elbow 8 cm below the shoulder, out more than 12 cm), so inside
them the elbow's place on its circle was left to wherever the solver drifted.

**What changed** (`reference.py`):

* `NaturalIK.terms`, the redundancy resolution: a swivel term that is always active pulls the elbow towards its
  natural place on its circle about the shoulder-wrist line. The natural place is down and a little out and back (for a
  hand hanging below the shoulder, back), and the pull fades out as the arm straightens. Beside it: the elbow not above
  the shoulder, not behind the body, not far out beside it; the wrist near its neutral where the hand's pose leaves it
  free (its sideways bend within 0.3 rad, its flexion within 0.55 rad, its roll within 1 rad); the shoulder's twist
  within 1 rad; the elbow neither straightened past straight nor folded tight. The joint limits (0.35 rad margin)
  outweigh all of these, so the swivel never wrings a wrist against its stop.
* `Reference.refine_path`: every joint path the reference executes keeps its planned hand path, but each configuration
  is solved again from the previous refined one with that resolution, drifting by at most 0.15 rad a waypoint (0.45 at
  the last). The hand, and what it holds, passes the same poses. A refined configuration that would touch the room
  where the planned one did not, or a move between two that leaves the hand's path by more than the planned one did
  (+ 4 mm), keeps the planned path.
* The planner's own choices (stances, grasps, carries) still rank by the round-3 measure (`terms_rank`); with them
  ranked by the new one, the plates' carries changed and the dish tray and the faucet became unreachable in every
  development scene.
* A held item that touches what the robot stepped up to (the sponge at its tray) is lifted 4.5 cm once before the
  put's last re-plan; before, the put was given up.

**How it is measured**: `tools/posture_run.py` now also reports a finer measure (its docstring): the elbow more than
55 deg from its natural direction about the shoulder-wrist line (the arm bent more than 25 deg), the elbow hyperextended,
the wrist bent (pitch or yaw beyond 0.95 rad, roll beyond 1.55 rad), the shoulder twisted beyond 1.45 rad, a wrist
flip. Both rounds were measured with this tool, on development scenes 11-15, one privileged episode each, both arms
every 0.2 s (time with either arm flagged; seconds summed over both arms per flag):

| scene | round 3: stage score | round-3 measure | finer measure | swivel off | wrist bent | at a limit | round 4: stage score | round-3 measure | finer measure | swivel off | wrist bent | at a limit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 11 | 0.9028 | 7.4 % | 67.7 % | 845 s | 1305 s | 101 s | 0.9120 | 8.1 % | 66.2 % | 470 s | 1512 s | 65 s |
| 12 | 0.9352 | 7.7 % | 67.7 % | 991 s | 1618 s | 134 s | 0.9306 | 7.6 % | 63.2 % | 490 s | 1588 s | 92 s |
| 13 | 0.9491 | 7.2 % | 65.6 % | 866 s | 1326 s | 127 s | 0.9028 | 7.3 % | 64.5 % | 511 s | 1589 s | 79 s |
| 14 | 0.7100 | 7.1 % | 72.9 % | 1046 s | 1317 s | 116 s | 0.6900 | 7.9 % | 68.3 % | 477 s | 1497 s | 69 s |
| 15 | 0.9213 | 7.2 % | 69.5 % | 977 s | 1529 s | 106 s | 0.9213 | 7.2 % | 64.6 % | 582 s | 1631 s | 82 s |

The elbow's swivel is off its natural direction for about half the time it was (4725 s to 2530 s over the five
scenes), and the joints are at a limit about a third less often. No wrist flips, no hyperextension, in either round.
The wrist is bent somewhat more often (7095 s to 7817 s): with the hand's pose fixed by the plan, the one free degree
of the arm trades a lower elbow for a more bent wrist, and the term is weighted for the elbow. The remaining wrist bend
comes from the hand's orientation, which the object and the plan fix: an item held level by the grip its take left (the
sponge for 700 s of an episode while wiping, rinsing and soaping, a plate by its rim), the lever of the faucet reached
across the sink. Changing those orientations means changing the plan's grasp and carry choices, which made the
reference fail in every development scene (above).

## Round 5 (2026-11-10): the planning redone for a person's posture

The round-4 video still looked contorted to the reviewer, and the finer measure agreed: either arm was flagged for
63-68 % of every development episode, the wrist bent for 1500-1630 s of each. Round 5 looked for the planning decision
behind it rather than the joints, with MuJoCo's own renderer (two views round the robot every 3 s, close-ups of every
skill) and the finer measure attributed to the job's phases.

**Why the reference looked contorted.** Over the five development episodes of round 4 (14210 s, 9292 s flagged),
walking with something in hand accounts for 62 % of the flagged time: walking with the sponge alone 39 % (3599 s, the
wrist bent for 97 % of it), with dishes, books and litter another 23 %; wiping 8 %; the bin lids, the sink (rinse,
soap pump, faucet lever), the tools and the puts the rest. Four planning decisions forced it:

1. *The carry pose.* A held item went to a fixed point in front of the chest, at least 1.0 m up and 0.27-0.33 m out,
   level. For the G1 (its shoulder 1.08 m up, its elbow hanging at 0.89 m, a short forearm and a long hand) that height
   that close in leaves two postures: the forearm pointing up with the hand bent down ~75 deg at the wrist (the
   "claw": the sponge carried from the sink to the table and back for ~700 s an episode), or the elbow raised.
2. *The grasp's hand orientation.* Each grasp was chosen for the pick-up alone (the sponge by an edge, tilted 30 deg
   down; glasses and cans 30-62 deg from above; the sponge's other family straight from above) and the hand kept that
   orientation relative to the item through the carry, the put and the work. A hand pointing down at a level item
   forces the wrist to bend wherever the forearm is level.
3. *The ranking measure.* Every choice of the planner (grasp order, carry point and turn, stance) ranked by
   `terms_rank`: limits, the elbow's height, out and back, the shoulder's twist. It did not see the wrist at all, so
   no choice was ever made for a straight wrist.
4. *Fixed upright stances.* At the counter, the bins and the lids the robot stood upright at one nominal spot (or the
   first feasible one near it) and reached: across the sink to the lever, over the counter to the pump, out over the
   bin, squatting at a lid whose tab rises above the shoulder as it opens.
5. *Moves between work poses that kept the hand's rotation.* After a take the item was drawn 10 cm back towards the
   body with the hand's rotation kept, and after a put the free hand was drawn in towards the chest with the put's
   rotation kept. Drawn in so, the elbow folds to its limit and the shoulder swings back, or the elbow rises over the
   shoulder (a can tipped into a bin), and the arm stayed so for the 4-9 s the robot then took to step back and
   straighten up. (Round 5's first pass, with decisions 1-4 redone, left about 13 % of each episode flagged; a third
   of that was these moves.)

With the hand's full pose fixed by these choices, the arm's single redundancy (the elbow's swivel) can only trade the
elbow against the wrist, which is what rounds 3 and 4 tuned; no resolution of it fixes a posture the hand pose rules
out.

**What changed** (`house_cell/reference.py`; the cell, its controller and IK, the physics and the API are unchanged):

* `NaturalIK.human`, the planner's ranking measure (`posture_cost`): squared hinges past a person's comfortable range
  on the quantities of the finer measure: the wrist's sideways bend (0.30 rad), flexion (0.45 rad) and the forearm's
  turn (1.05 rad); the shoulder's twist (1.0 rad); the elbow at least 9 cm below the shoulder, not out or back; the
  elbow's swivel within 30 deg of its natural direction; every joint 0.25 rad off its limits. Every choice is ranked by
  it on the posture the executed path resolves (`natural_resolve`: NaturalIK's natural redundancy resolution, as
  `refine_path` executes it), not on the planner's raw IK solution.
* The carry (`canonical_carry`, `carry_points`, `carry_pose`): in front of the waist with the forearm forward and the
  wrist straight, at the item's own height clear of the dining table (0.84 m and up), the hand 0.30-0.40 m out; near
  the kitchen counter a dish or the sponge just clear of it (0.96 m and up, the trays' rims 3 cm below; see below); the
  arm stretched out far in front costs; the item's turn about the vertical and the point ranked together. Next to a
  fixture the robot steps back before lowering what it holds. The long tools (`_tool_carry`) are turned about their
  handles to the angle the arm holds most naturally, and may hang a little farther forward.
* Grasps planned backwards from what follows (`_grasp_frames`, `carry_costs`, `sponge_work_costs`, `_take_natural`):
  glasses, cans and the bottle from the side with the hand nearly level, at mid-height, round the item; plates at
  any point of the near rim, the pinch also leaned 20 deg; napkins and the apple core level from the side; the
  sponge by its long edge with the pinch leaned 22-28 deg (the forearm turned ~1 rad, not palm down) or across its
  width from an end. Every item's grasp is ranked by the carry it leaves and taken from the stance of the most
  natural reach for it; the sponge's by its wiping, rinsing, soaping and carrying postures.
* The body chosen with the arm (`stance_search(rank)`, `path_posture`, `plan_put`): a put, a stroke, a rinse, a
  press, a lever, a lid, a tool's take: of the first few feasible stances (where the robot stands, how far it leans,
  how low it stands) and the first few turns of a symmetric item, the one whose path the arm takes most naturally.
* Bending with something in hand (`Reference.posture(hold)`, `Stance._check(hold)`): the hand keeps its place in the
  room while the body bends, at a slower rate (as a person leans over a counter with a plate in hand; with its arm's
  joints fixed the item swung down with the torso). The stance checks model it.
* Puts at the counter (dish tray, soap pump, rinse, the sponge's tray), as at the shelf, from 0.25-0.30 m short: the
  item held where it is while the robot bends there, brought to its first pose's place, then stepped in
  (`go_put(low)`; `sponge_path` now plans through `go_put`).
* The sink: the water turned on before the sponge is taken (both hands free); the sponge rinsed 7 cm under the
  spout's mouth (the cell rinses anywhere in the stream; 15 cm under it the hand reached down over the sink's front);
  the pump approached from 3.5 cm above its head; the lever's end taken level from in front of it.
* Bins: litter may be tipped forward into the opening, judged by the drop's posture (`bin_put`), the robot stepping
  up closer from 0.25 m short (carried at the waist straight in, an item met the open lid's front edge).
* Walking with an item at the waist (`walk`, `held_path_clear(loose)`): the item is checked along the walk (near the
  goal the walk planner keeps only the body's clearance); if it would meet the room (the broom hanging by the bins, a
  chair's back) the last 0.8 m is walked straight in at the goal's heading, else the item is carried higher or drawn
  in close. Next to a fixture the item is drawn in where the robot stands, then the robot steps back before lowering
  it to the waist.
* Tools: the broom's and the dustpan's grips ranked by the take's posture (`tool_take`); the tools turned about their
  handles for the carry (`_tool_carry`).
* The sponge's grip is checked against the first round's real strokes (`wipe_strokes_ok`), waypoint by waypoint; if
  the first round still reaches no stroke from either side, the sponge is put back in its tray and taken again by the
  next of round 5's grips (that grip and its half-turn ruled out), and only then by round 4's (`degrease`; in scene
  13 the first grip reached no stroke, the second did).
* A take ends with the lift (`_take_go`): the planned path still includes the draw-back (the stance search keeps room
  for it), but it is not run; the carry pose follows from the lift's posture.
* An item carried at any angle (a napkin, an apple core) goes to the planned carry posture's rotation too
  (`canonical_carry`; solved for its position alone, the move kept the wrist as bent as the take had left it, for the
  whole walk to the bin).
* Next to a fixture with no carry pose clear there, the held item is drawn straight back towards the body at its
  height before the robot steps back, if the arm then holds it more naturally (`draw_in`; after the soap pump the arm
  stayed stretched out at shoulder height for the steps back).
* The dish tray (`tray_put`, `tray_reach_cost`, `_grasp_frames`): the dish set down level (its rotation was captured
  just after the take, while the robot still leaned over the table: tilted 8 deg with the torso, the planned put
  dipped the plate's rim); a grasp is ranked by the put only if the arm and the dish clear the counter and the tray's
  rim where it lets go (every near-rim plate grip put the wrist into the counter's top, so plates went in turned
  1 rad with the arm stretched out at shoulder height: put cost 33-38, now 5-14); plates may also be pinched from
  38 deg above.
* Pushing crumbs off the table (`wipe_crumbs_off`): after each stroke the sponge is drawn back out over the table and
  lifted a little before the robot straightens up (at the edge the hand is in front of the hips, the elbow folded).
* Turning with a held tool (`walk`, `held_path_clear(strict)`): each turn in place is also checked with the body 3 cm
  off and 0.08 rad either way (the dustpan hanging by the hip caught a dining-table leg as the robot turned beside it,
  slipped and fell); if that is not clear, the robot first steps straight back. When every plan to empty the pan fails
  from where the robot stands (the walk refuses to turn the pan beside the table), it steps straight back and plans
  again (`dump_pan`); a dump planned after the pan was lost stops (it raised an exception).

* The bin lids (`Reference.lid`): the standing procedure had its nominal stance on the far side of the tab (the right
  hand reaching across the body) and found no stance at all, so round 4's squat ran; the robot now stands on the
  hand's own side, 0.60 m from the wall, leaning a little, the tab taken from 45-60 deg above, and the bin by the south
  wall is opened and closed by the left hand (standing for the right hand there, the free hand, which the stance
  model leaves out, met the wall at 245 N).
* A dish or the sponge for the counter is carried at the waist until 1.2 m short of it, then raised (`walk`,
  `dest_high`); the put is planned from the raised posture (`go_put`, the stance model's `seed_override`, a dry run of
  `canonical_carry`). Drawn in by the dining table, an item is raised only clear of the table, not to the counter's
  height (a glass carried from the table rode on a straight arm at shoulder height for 25-30 s a trip).
* Carrying the broom (`broom_upright`, `stroke`): for a walk its head is 10 cm off the floor and the hand at 0.96 m,
  the elbow hanging (held with its head 20 cm up, the grip is at 1.06 m, the G1's shoulder height, and the right elbow
  winged out to hold it there); at a stroke's stance it is raised back to round 4's carry before the robot lowers and
  leans (that lowers it by up to 15 cm), and the stroke is planned from that raised posture (the stance model's
  `seed_override`, a dry run of `broom_upright`), so the strokes are planned as round 4's. (The second pass tried the
  lower carry with the stroke started from it: each stroke then took fewer crumbs into the pan.)
* The stance ranking counts the legs too: of stances the arm takes alike, the taller one (`stance_search`,
  `ranked_stances`); the coffee table may be worked at 0.62 m.
* Straightening up with an item in hand, the hand keeps its place in the room as far as the arm reaches (`stow`;
  with the arm's joints fixed the sponge rose with the torso to the face).
* After hanging a tool in its clip the robot steps 0.25 m straight back from the wall before turning (`tool_home`;
  turning beside the hanging tools it pressed on them and the walk to the bins' lids gave up, in scenes 13 and 15;
  retrying such a walk once after a step back made the knee press the hanging dustpan past the safety limit:
  dropped).
* The squats by the long dustpan are unchanged (see "What remains"). Tried: a grip 8 cm higher on the handle (in
  whole episodes the walks to take the pan failed and one scene's pan was never emptied, its tools not hung and its
  lids left open), 4 cm higher (set down standing, the hand close in by the hip and the wrist bent to its limit: 140
  on the human measure, against 1-2 for the knees bent), the pan taken again from behind between two groups, standing
  (then no carry pose was reachable with that grip, nor a set-down), the free arm held at the waist meanwhile (it could
  not go home again and the robot pressed on the table: three scenes' sweeping failed). All dropped.
* The free hand after a put (`retreat`) rises off what it let go of with the hand's rotation kept, to 1.0 m but no
  higher than 8 cm below the shoulder (let go low with the knees bent, at the dustpan's handle, the hand rose to the
  face), then goes to the arm's natural posture in front of the chest by a checked joint move (with the rotation kept
  on that way too, the arm folded to its limit with the elbow over the shoulder for the seconds the robot then
  stepped back). Rising only 10 cm, then to the waist, made the joint move fail more often and fold the elbow:
  dropped.
* After a bin's lid is opened or closed the free arm goes home at once, off the tab by 4 cm and then by a joint move
  (`Reference.lid`, `retreat(up_to)`; drawn back with the pinch's rotation kept and up to 1.0 m, the open hand rose
  beside the head, palm out, and stayed there until the next move).
* Before an item is lowered to the waist next to the kitchen counter, the robot steps back until the carry point is
  5 cm in front of the counter's edge (`clear_of_counter`; stepped back from the sink after a rinse, the sponge went
  in below the sink's front wall, caught on it and fell).
* A dish or the sponge is raised for the counter once per walk (`walk`, `dest_high`): where the rest of the way was
  longer than `HIGH_NEAR` as the crow flies, the raise was planned again on every call, without end (one test
  version's scene 12 ended in a recursion error).
* The carry for the counter: tried and dropped: ranking the carry by the upper arm's raise as well, with points closer
  in (a glass at 1.0 m rides with the upper arm up 50-60 deg; its point did not change, as closer in the side grip
  bends the wrist further, while a bottle went to the hip and a plate's carry bent the wrist more); raising a dish or
  the sponge only 0.7 m short of the counter's stance instead of 1.2 m (a plate carried at the waist from the dining
  table slipped from the hand on the way and broke); holding the item drawn in towards the body for the steps in at
  the counter, or as carried (the drawn-in places were out of the arm's reach or brought a glass within 1.5 cm of the
  body; stepped in with a glass as carried, it caught on the counter and slipped from the hand).
* Pushing the crumbs off the table (`wipe_crumbs_off`): stances farther back are tried first, ranked by the stroke's
  posture (16 cm from the edge and leaning, the shoulder is over the edge, and drawing the sponge to it pulled the
  elbow far behind the back); from farther back each stroke also pushes more crumbs off (8-10 strokes a scene, every
  crumb off the table; round 4 took 9-23, the second pass 7-23 and left up to 3).
* Setting a book into the waist-high shelf, the farther stances are ranked first (`shelf_put`, `put(offsets)`;
  standing close, the elbow was pulled far behind the back to slide the book in).
* Stepping back with something in hand (`back_out`, `leave_low`; added after the first run of amendment 4's
  selection, `hidden_set.md`): the planner checks a step back for the body only, and with the carry at the waist a
  held item swept a glass off the dining table as the robot stepped back from it (development scenes 60 and 86; at
  the coffee table in scene 87 the can and the wrist dragged the bottle along). Before such a step the hand is raised
  until the item clears, by 3 cm, the tops of the loose objects within 0.2 m of its way back
  (`lift_clear_of_neighbours`, sized by their collision boxes), and the item is carried again after the step (`walk`).
* The last 0.8 m walked straight in with an item at the waist is also kept off the long tools, by the hips' half width,
  all the way to the goal (checked against the room's fixtures only, a knee walked into the dustpan hanging by the
  bins: a safety stop, and the can it carried fell; development scene 60).

**Measured** (`tools/posture_run.py`'s finer measure, computed by the same code over recorded episodes, as the
oracle's runs run the reference; development scenes 11-15, one privileged episode each, both arms every 0.2 s; time
with either arm flagged; seconds summed over both arms per flag; round 4 as in the table above; all five scenes
recorded with the final code, after the two fixes above):

| scene | round 4: stage score | finer measure | swivel off | wrist bent | at a limit | elbow high | round 5: stage score | finer measure | swivel off | wrist bent | at a limit | elbow high |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 11 | 0.9120 | 66.4 % | 470 s | 1512 s | 65 s | 196 s | 0.9491 | 10.1 % | 135 s | 163 s | 28 s | 57 s |
| 12 | 0.9306 | 63.4 % | 490 s | 1588 s | 92 s | 175 s | 0.9120 | 9.9 % | 147 s | 109 s | 30 s | 73 s |
| 13 | 0.9028 | 64.7 % | 511 s | 1589 s | 79 s | 187 s | 0.9167 | 10.2 % | 160 s | 116 s | 28 s | 56 s |
| 14 | 0.6900 | 68.3 % | 476 s | 1497 s | 68 s | 164 s | 0.9537 | 9.6 % | 110 s | 125 s | 32 s | 53 s |
| 15 | 0.9213 | 64.7 % | 582 s | 1631 s | 81 s | 192 s | 0.9167 | 8.2 % | 152 s | 79 s | 22 s | 66 s |

Over the five scenes the flagged share falls from 65.4 % to 9.6 % (9292 s to 1431 s), the wrist-bent seconds
from 7817 s to 592 s and the elbow's swivel off its natural direction from 2529 s to 704 s. Every stage score is at or above round 4's except scene 12 0.9120 against 0.9306 (35 crumbs binned against 39; everything else at full credit), scene 15 0.9167 against 0.9213 (36 crumbs binned against 37; everything else at full credit): the crumbs the sweep bins vary by several between near-identical planners (the same group's one stroke took 30 of its 50 crumbs into the pan in one recording and 42 in another; those left beside the table's edges, where the pan cannot stand, stay); no safety violation, no dish broken or dropped. (Three recordings of nearly the same planner, the last three of this round, binned 34, 39, 44, 44
and 43 crumbs, 34, 35, 37, 44 and 43, and 43, 35, 30, 44 and 36 in scenes 11-15: a scene's count swings by up to 13
between them, round 4's 35, 39, 33, 0 and 37 inside that spread; every other part of the credit is full in all
of them but the first's scene 12, where the sponge fell at the sink, since fixed.) The episodes run 2825-3155 s (round 4 2556-3081 s): the counter is approached from a step back with the item raised there,
the sink's work and the crumbs' strokes take more planned moves.

**What remains** (the finer measure interval by interval, labelled by the innermost traced skill; per scene):

Per scene: moves under 2 s 74.8 s, stepping back after a put 40.7 s, moves into the carry 30.8 s, the soap pump 29.6 s, the broom's strokes 24.2 s, the faucet lever 15.0 s, books onto the shelf 10.4 s, takes 8.5 s, the free hand's retreat after a put 8.5 s, setting the dustpan down 7.7 s, pushing the crumbs off the near edge 6.4 s, litter into a bin 5.5 s; the rest under 5 s each.

* *The crumbs pushed off the near edge*: each stroke still ends with the sponge's middle 1 cm inside the edge (the
  crumbs ride 4-6 cm ahead of it); from the stances farther back this fell from about 65 s a scene to about 6 s.
* *Stepping back after a drop into a bin*: the free arm drawn back from the opening while the robot steps back and
  straightens (3-7 s each), flagged for the elbow's swivel.
* *Moves under 2 s* between two natural postures (some 60 a scene). The free arm leaving or reaching the cell's
  `ARM_HOME` is flagged for about a second each time: there the elbow's swivel is already 60 deg off the measure's
  natural direction (its threshold 55). A rest turned in 0.3 rad removed that, but it changed the planner's stances by
  the bins (`HOME_Q` seeds its IK) and litter fell: reverted.
* *The broom's strokes*: the wrist bent about 65 deg holding the handle's lean.
* *The soap pump and the faucet lever*: at the G1's reach (the pump's head 1.06 m, the G1's shoulder 1.08 m; the
  lever 0.48 m in from the counter's front at shoulder height); an offline search over stances, leans and grips finds
  nothing better than what the planner takes.
* *The moves into the carry* (1-3 s each), and the free hand's retreat where its joint move to the natural posture is
  blocked (in a bin's opening, under a shelf): then the round-4 retreat with the put's rotation runs.

**The blind reviews.** After each pass a fresh reviewer, given only the images (both views every 6 s and the
close-ups of every skill, nothing but the time on each tile) and the standard (the posture a person takes doing the
same housework; brief transitions acceptable), went through every moment of the development scenes, splitting the
work among helpers and checking doubtful tiles at full size; it never saw the code or the measure. The first (the
first pass, scenes 11, 12, 14 and 15) rated 54-57 moments a scene clearly contorted; the second (the second pass,
all five scenes) and the third (scenes 11, 12, 14 and 15) found the same few problems at nearly the same times in
every scene. The fourth looked at the last planner but one (all five scenes, 22 helpers; the final one differs only
in the carry's ranking): 65-93 moments a scene clearly contorted and 83-253 mild, the counts dominated by held
postures, in the same places in every scene: the soap pump; the dish tray (a plate pinched by its rim raised to
shoulder height, the wrist bent 60-90 deg down; a glass on a near-straight arm at shoulder height for the last steps
and the step in); the elbow far behind the back (at the end of some pull strokes, books into the low shelf, the free
arm akimbo for single moments); the dustpan (squats, emptying); the faucet lever; the broom carried upright in
front of the face (mild, 50-100 s an episode); small items held 30-50 cm out in front (mild); the hand flipped up
palm-out right after a release (single moments). What they flagged and the final version:

| flagged (in every scene unless noted) | final version |
|---|---|
| the elbow winged to shoulder height right after a take, the item across the chest | fixed: no draw-back after the lift |
| the empty arm thrown up after letting go | fixed: the retreat turns to the arm's natural posture by a joint move, no higher than 8 cm below the shoulder; after a lid the arm goes home at once. A brief palm-out flip after some releases remains (single moments) |
| the bin lids from a deep squat | fixed: opened and closed standing |
| a glass carried on a straight arm at shoulder height | shorter, not gone: at the waist until 1.2 m short of the counter (was 25-30 s a trip at shoulder height), then 1.0 m up with the upper arm raised 50-60 deg for the last steps and the step in (10-18 s a glass) |
| the broom carried upright with the right elbow winged at shoulder height | the elbow is down now (the hand at 0.96 m); the broom is still upright in front of the face (mild) |
| the elbow pulled behind the back on the strokes that push the crumbs off the table | fewer: stances farther back, 8-10 strokes a scene (round 4 9-23); still at the end of some strokes (single moments) |
| the elbow behind the back sliding a book into the low shelf | farther stances first; still in some scenes (2-6 s) |
| the soap pump pressed with the elbow at or above the shoulder; the faucet lever and the rinse with the arm out at shoulder height | the cell's geometry (below) |
| plates pinched by the rim, the wrist bent onto the rim at the dish tray | the geometry (below) |
| squats at the coffee table | the G1's waist (below) |
| squats by the long dustpan; the hand above the head emptying it | the dustpan's geometry (below) |

Tried for the glass and dropped: raising it 0.7 m short instead of 1.2 m (a plate carried at the waist so slipped
from the hand on the way and broke), holding it drawn in towards the body or as carried for the step in (the
drawn-in places were out of reach or brought it within 1.5 cm of the body; stepped in as carried, it caught on the
counter and slipped from the hand), ranking the carry by the upper arm's raise (at 1.0 m the side grip leaves the
upper arm up 49-65 deg at every point: no other point is better). For the dustpan: a grip 4-8 cm higher (standing
set-downs rank 120-140 on the human measure, the hand close in by the hip and the wrist at its limit; in one scene
the walks to the pan failed), the pan taken again from behind between groups (no carry pose reachable with that
grip), the free arm held at the waist meanwhile (the sweeping failed in three scenes).

What the cell's geometry decides: the soap pump's head is at 1.05 m, the G1's shoulder at 1.08 m (a person's at
about 1.4 m); the faucet lever's end is 0.29 m in from the counter's front at shoulder height, and the robot's middle
cannot come nearer than 0.16 m to the front; the dish tray's plates sit 0.3 m in. A plate lies 1.1 cm proud of the
table, so the hand can only pinch its rim from above (18-38 deg), and the plate kept level then bends the wrist down
30-45 deg, more as it is raised for the tray. The coffee table's top is at 0.45 m; the G1's waist bends at most
0.52 rad, which lowers its shoulder 3.5 cm and moves it 12 cm forward, so only the knees can bring the hand down to a
can's middle (0.52 m): a person bends at the hips instead. The dustpan's handle top is 0.87 m up and the hand grips
the handle across, the forearm level with it: the G1's elbow hangs at 0.89-0.91 m, so only the handle's very top could
be held standing, and held there the pan hangs 0.85 m below the hand; the pan is set down held 0.58 m up the handle
(the hand 0.70 m up: the knees bent to 0.70 m) and taken from its lip's side 0.40 m up (the hand 0.53 m up: the
lowest squat), because tipping it past 55 deg into the bin puts that grip at 1.1-1.2 m already.
