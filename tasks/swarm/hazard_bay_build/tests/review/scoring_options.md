# Scoring options (maintainers only; the grader never reads this file)

`grader_config.json` chooses how `acceptance_score` is made from the six sections:

- `"combine"`: `"weighted"` (the weighted mean by `sections`), `"mission_gate"` (the weighted mean of the five scene
  sections by their `sections` weights, times the mission section) or the name of another weights block,
  `sections_<name>` (today `"mission_0.8"`: the mission at 0.8, the five scene sections at 0.2 in their proportions);
- `"progress_gate"`: `true` or `false`. With it, the mission's checks a run earns without doing its jobs count only as
  far as it did them: each run's three bucket checks (lift-off level, swing, water kept) are multiplied by its fire-out
  check (1 when the fire is out at the end, else 0), so lifting the bucket and hovering earns nothing; its four safety
  checks (parts not dropped, no losses, no collisions, separation) by its progress, (steam released + fire out +
  station cleared) / 3; and consistency by the two missions' mean progress (the two days of the fixed hidden set,
  `tests/hidden/fixed_set.json`: every submission meets the same two). A run that does all three jobs loses nothing.
  The bucket's gate dates from 2026-11-04 (the owner: lifting the bucket and hovering must earn about nothing; before
  it a lift-and-hover program earned 0.223 of the mission, an idle one 0.041, now both 0.041). Multiplying the bucket
  checks by the run's progress instead was considered and not kept: it takes from runs that put the fire out but
  missed another job (Opus's 11-02 submission lost 0.081 of the mission in one of its verifies) and still pays for
  hovering in a run that did the other jobs (a third or two thirds of the bucket checks).

Every verdict records all six combinations in its metrics (`acceptance.<combine>` and
`acceptance.<combine>.progress_gate`) and both mission sections (`section.control.progress_gate`,
`section.control.no_progress_gate`), so a switch needs no new measurement of the agents' runs.

**The anchor follows the config by itself.** `tests/anchor.json` records the reference's value and `full_at` under all
six combinations (`by_combine`, keyed as the metrics are): each value the lowest of the same full verifies of the
reference under that combination (its `source` names them), each `full_at` sized by the same rule from the same runs;
the grader takes the entry of the combination in use (the verdict's metrics `anchor_value` and `anchor_full_at` say
which). Re-measure when the reference, the grader's scores or the fixed hidden set change.

**The agent's text does not.** It must describe the scoring that is used, so a switch also swaps the sentences
below (the one edit besides the config line):

## `instruction.md`, "How it is graded"

- `mission_gate` (today):
  > ``acceptance_score`` is the scene's score times the mission's: the scene's score is the weighted mean of the first
  > five sections below (the weights in brackets), the mission's is the last section's score.

  and the mission's bullet starts "**The mission (it multiplies the scene's score)**:".
- `weighted`:
  > ``acceptance_score`` is a weighted mean (the weights in brackets):

  and the mission's bullet starts "**The mission (0.54)**:".
- `mission_0.8`:
  > ``acceptance_score`` is a weighted mean (the weights in brackets):

  with the brackets Layout (0.035), Physics (0.017), The steam line and the hazards (0.017), The rig (0.087), The
  fleet's design (0.043), and the mission's bullet starting "**The mission (0.80)**:".

## `environment/setup/payload/given/control.md`, "How it is scored"

- The second sentence of the section: `mission_gate`: "The mission's score (the checks' weighted mean) multiplies the
  scene's (`instruction.md`)."; `weighted`: "The section's score is the checks' weighted mean; it is 0.54 of
  `acceptance_score` (`instruction.md`)."; `mission_0.8`: the same with 0.80.
- `progress_gate: true` (today) keeps the two sentences "The three checks marked † (how the bucket was handled) count
  only in a run that ends with the fire out: ..." and "The five checks marked * count in proportion to what each run
  got done: ...", the three `†` marks in the table (lift-off level, swing, water kept) and the five `*` marks (parts
  not dropped, no losses, no collisions, separation, consistency); `progress_gate: false` removes all of them.
