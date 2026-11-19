# Task cards

A task's card says how displays (the website's task list, a task's page, the leaderboard) name
and file the task. The cards live in [`tasks/cards.toml`](../tasks/cards.toml), one table per
task named by its slug. Maintainers keep the file and add a task's card before its pull request
merges; a contributor may suggest a title in the pull request.

The file sits outside every task folder, so a card is no part of a task's identity: writing or
changing one changes no `spec_hash` and no `content_digest`, and needs no identity-history
record. (A field in `task.toml` would change both hashes, and with them the version every
agent's result is tied to.)

```toml
[hazard_bay_build]
title = "Rebuild a hazard bay, design and run its fleet"
summary = "Rebuild a plant's hazard bay in Isaac Sim from photos, design a fleet of flying and ground robots from a catalog, and program it through the job."
platforms = ["swarm"]
stages = ["behavior"]
```

## Fields

| Field | Rule |
|---|---|
| `title` | Required. A readable name in sentence case, at most 48 characters, no closing period, unique across the cards. It leads with what the agent does and names the robot: "Assemble a desktop arm with a Panda", "Fly a gate course by camera alone". |
| `summary` | One sentence, at most 200 characters, a capital letter first and a period last. Displays fall back to the brief's first paragraph without it. |
| `platforms` | Required. A list of exactly one entry: the key in [`tasks/platforms.json`](../tasks/platforms.json) of the platform the task is filed under, which is its folder's (`[metadata].platform`). A task is filed under one platform, also when it involves another kind of robot. |
| `stages` | Required. A list of exactly one entry, the stage the task is filed under: `requirements`, `hardware-design`, `assembly`, `robot-description`, `control`, `environment` or `behavior`, the stages of the workflow in order, or `end-to-end`, the category of a task that runs the whole chain. "Which stage" below says which one. |

## Conventions

- **Families share a pattern.** `*_requirements_replica_*`: "Specify upgraded replicas of a
  …"; `*_cad_design_*`: "Model a … in CAD"; `*_full_assembly_*`: "Assemble a … with …".
- **Which stage.** A task has one stage: the last stage of the workflow that its score depends on.
  A job task that also has the agent rebuild a scene or pick hardware on the way is filed by the
  job (`hazard_bay_build` rebuilds a bay and picks a fleet from a catalog, and is `behavior`).
  `assembly` means building the robot's own body; assembling furniture or parts with a robot is
  `behavior`.
- **Control or behavior.** A task is `control` when its core is a feedback control law for the
  robot's own dynamics (docking a spacecraft, holding a quadrotor on a gate course). It is
  `behavior` when its core is deciding what to do and how to move to get a job done, even when
  what it emits are low-level commands (a supervisor that takes over a car, retargeting human
  hand motion to robot hands).
- **Which platform.** The most specific one wins. A legged robot is `quadruped` or `humanoid`.
  `ground` ("Ground vehicle") is any wheeled or tracked ground vehicle or robot: a car at any
  scale on a road or a track (autonomous driving, racecars), an AMR, AGV or UGV, a rover. A task
  that uses several kinds of robot is filed under the one its
  scored work is about: a robot's own assembly by an industrial arm under that robot's platform,
  a mixed fleet under `swarm`, a mobile manipulator whose scored work is manipulation under
  `arm`.

## Before a merge

A task merges with its card. The maintainer who finishes a task's pull request adds the card
(and, for a GPU task, the `gpu_types` it was tested on: [`templates/README.md`](../templates/README.md))
in a commit on the pull request's branch, then sets the label `ready-to-merge`. With the label,
CI runs the card check with `--strict`, so a task without a card fails; the same holds on
`main`. Until then a missing card is only a warning, and a contributor's checks stay green.
Nothing asks the contributor for a card.

## Checks

`uv run python scripts/task_cards.py check` runs in CI (`pr-checks`, static-checks): every card
is well formed, names a task on the tree, holds exactly one platform, its folder's, and exactly
one valid stage, and no two cards share a title. A task without a card is a warning there;
`check --strict` (on `main`, and on a pull request labelled `ready-to-merge`) makes it an error. `task_cards.py todo` lists the
tasks without a card and the cards without a task; `task_cards.py context <task>` prints what
naming a task needs (its `task.toml` facts, its brief's title, headings, first paragraph and
deliverables, and the cards of its family). [`schemas/task-cards.schema.json`](../schemas/task-cards.schema.json)
describes the parsed file.
