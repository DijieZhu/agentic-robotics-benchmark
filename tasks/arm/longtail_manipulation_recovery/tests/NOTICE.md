# NOTICE - sources of this task's assets (verify / maintainers only; never shown to the agent)

The kitchens, fixtures and objects the task's simulator loads come from the following sources. This notice is the
attribution their licences ask for.

- **RoboCasa code** (`robocasa/robocasa`, commit `456174f62b89b8fca99eaaf33949c29fec9cfc2a`): MIT. RoboCasa assets: CC BY 4.0
  (RoboCasa README). S. Nasiriany et al., RoboCasa, https://robocasa.ai.
- **Lightwheel fixtures and objects** (`fixtures/**` and `objects/lightwheel/**` in the asset tree): the Hugging Face dataset
  `nvidia/PhysicalAI-Robotics-Manipulation-Objects-Kitchen-MJCF` (revision `420a04af`), licence **CC-BY-4.0** as declared on its dataset card.
- **Objaverse objects** (`objects/objaverse/**`) and textures (`textures/**`, `generative_textures/**`): distributed with the RoboCasa asset set
  (Hugging Face dataset `robocasa/robocasa-assets`), licence **CC-BY-4.0** as stated by RoboCasa. The asset tree has no per-object
  licence files; the original per-object Objaverse terms are UNVERIFIED and this task relies on the distributors' statement, as the
  merged `robocasa_object_dynamics_from_video` task does.
- **aigen objects** (`objects/aigen_objs/**`): part of the same RoboCasa asset tree; only their folder names appear in this task as
  empty catalogue stubs (`object_instances.txt`), no model data unless a state uses one.
- **The robot**: Franka Panda arm, Omron mobile base and Panda gripper of robosuite (`ARISE-Initiative/robosuite`, commit
  `5ce6643f3092639d08f7b0f90ed1c6a84f50552c`), MIT, installed unmodified in the task image.
- **MuJoCo** 3.3.1: Apache-2.0.

Modifications: the agent's image and the verifier image carry only the files the scenes of this task load; every other object
instance folder is present as an empty `model.xml` (`make_stubs.py`), so that RoboCasa's seeded split of its catalogue selects the
same objects as with the full asset tree. No asset file is altered.

**The failure recordings** (actions in `tests/hidden/` and `solution/rescues/`, and `environment/setup/payload/practice_states/`)
are this task's own: rollouts of the frozen Xiaomi-Robotics-1 (XR-1, Apache-2.0) policy on RoboCasa, recorded on the
author's machine. XR-1 itself is not part of this task, is in neither image, and was only used offline to generate failures and
rescue action sequences.

Where the large files live: `grounded-intelligence-initiative/ale-robotics-assets`, under
`tasks/arm/longtail_manipulation_recovery/payload/`, as two archives: `robocasa_assets.tar.gz` (1,971 files, the practice
kitchens, fetched by the agent's image) and `verifier_assets.tar.gz` (2,309 files, the graded kitchens, fetched by the
verifier image only). Each archive is pinned by dataset commit and sha256 in `environment/assets.sha256` and
`tests/assets.sha256` (docs/task-assets.md); after unpacking, the image builds check every file against the per-file
lists `environment/robocasa_assets.files.sha256` and `tests/verifier_assets.files.sha256`, which are the lists this
task was first submitted with (one request per file would run into the Hub's rate limit, hence the archives).

**The held-out set on the public dataset is an exception.** The dataset holds agent-visible files; a task's hidden
material stays in git or is fetched at verify from its `verifier/` folder. `verifier_assets.tar.gz` is neither: it is
the part of RoboCasa's public asset set that the graded kitchens load (the held-out layouts and styles and 82 object
instances that the agent's image does not have), placed under `payload/` so that the verifier image can fetch it at
build time through `tests/assets.sha256`. The benchmark owner allowed this on 2026-11-17: the files are public
already, they hold no state, pose or answer of this task, and the agent works without network, so it cannot fetch
them. Which layouts, styles and instances the graded kitchens use can be read off that archive's file list, as it can
be read off `tests/` in this public repository.
