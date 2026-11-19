# Security model — what is hidden from the agent, and what a verdict proves

This page states what the benchmark defends against, how, and — just as carefully — what
it does not prove: the **grading trust boundary**, which keeps the grader's number
authoritative against a hostile agent inside the sandbox. The mechanics are the engine's
(`vendor/ale/docs/specs/episode-protocol.md`, `sandbox-image.md`, `security.md`) plus the
verify-stage kit (`shared/robotics_grader/`). The engine claims below were checked against
the engine code at `vendor/ale` `1cdbc02` on 2026-11-03; code references are relative to
`vendor/ale/packages/ale-run/src/ale/run/`.

Results are the engine's own episode records. The website is a display-only front end: it
takes no uploads and no run or score submissions (2026-10-15), so the submission trust
levels this page used to describe (`validated` / `verified` badges, the bundle exporter) are
retired with the upload path.

## 1. Assets and adversaries

Assets: (1) **result integrity** — the score is what the grader computed, not what the agent
claims; (2) **hidden-protocol secrecy during the run** — the per-episode seeds, the grader,
the anchor and the oracle are unreachable from inside the sandbox while the agent works;
(3) **credential secrecy** — the model credentials stay on the host's gateway, never in the
sandbox or a task folder.

The adversary is the **tested agent**: arbitrary code running for hours as the sandbox's
agent account, and again — through the solver it leaves behind — while the grader runs. It
may try to inflate its metric, echo canned output, pre-create or overwrite the verdict, read
the grader, or predict the hidden seeds. We do **not** defend against a compromised host or a
malicious operator: a result is only as trustworthy as whoever ran the episodes (the
guidelines' Agent Benchmarking role says who runs formal evaluations and what they record).

## 2. What stage isolation hides

Anti-cheat is **architectural isolation, not detection**. The mechanism is the engine's
stage protocol; the task only has to respect it.

| barrier | what it does | who enforces it |
|---|---|---|
| **Timing** | `tests/` (grader, `env.py`, `anchor.json`, `grader_config.json`) is uploaded to `/opt/ale/tests` only when the verify stage runs (`protocol/episode_protocol.py:757-783`), and `solution/` to `~/.ale-solution` only in an oracle episode (`:500-501`, `:1290-1297`). During the agent phase neither exists anywhere in the sandbox. | engine (`episode-protocol.md`: "`tests/` is also absent throughout this phase") |
| **Root-only staging root** | `/opt/ale` is `drwx------ root:root`: the layer the engine adds to every task image runs `chown -R 0:0 /opt/ale` and `chmod -R go-rwx /opt/ale` (`framework.py:122-123`), and stage uploads land below it owned by root. Even paths that do exist there are untraversable for the agent account. | engine |
| **UID boundary** | `tests/test.sh` runs as root; the agent's solver is spawned **at the agent's UID** (`robotics_grader.Solver`, `deprivileged`, via `runuser`). The solver cannot read the grader's memory: the kernel refuses to let one UID trace another's process. It does start with a copy of the verify stage's environment (`runuser -u` keeps the caller's environment and sets only `HOME`, `SHELL`, `USER` and `LOGNAME`), which holds the `ALE_*` stage variables and, when a run configures a Judge, its Gateway session token; so a grader keeps no secret in `os.environ`, and today's tasks configure no Judge. Artifacts the agent produced (checkpoints, pickles) are deserialized deprivileged too. Before it runs any agent code, `deprivileged` resolves `runuser` to an absolute path against a fixed system PATH (`/usr/sbin:/usr/bin:/sbin:/bin`) and reclaims every directory on root's PATH that is not root-owned 0755 (`secure_exec_path`), so a program the agent plants on root's PATH cannot shadow `runuser` — or any other name root resolves — on a later exec. | kit |
| **A cleared account** | A solver is confined by what the grader puts on the process it starts: the agent's UID, and in some tasks a seccomp filter without network, limits, or a pause between requests. None of that binds a process something else starts as the same account, and the base image starts some: its supervisord (PID 1) runs the desktop session as the agent's account and restarts it when it dies. Agent code could kill the session and have it started again; the new session ran the account's autostart entries, and a start script in a directory the account owns, outside the solver's filter (measured on `container-ubuntu22-base` 0.1.1: the restarted code opened an internet socket that the solver could not). A live session can also be asked to run things, over its D-Bus, X and terminal services. So before the kit starts agent code (`Solver`, `deprivileged`) it clears the account once per sandbox (`robotics_grader.clear_agent_account`, `shared/robotics_grader/account.py`): every supervisord program is switched off in supervisord's configuration, supervisord reloads and starts nothing, and every process of the account that the grader did not start is killed until none comes back. From then on the only processes of the account are the grader's. | kit |
| **Result path** | The result goes to `/logs/verifier/reward.json` and `verification.json` through `ale_verify`'s atomic write. Before every verification attempt the engine removes whatever is at `/logs/verifier`, makes `/logs` root's and unwritable for others, and recreates `/logs/verifier` empty with mode 0700 (`protocol/episode_protocol.py:144-156`, run at `:777`). The agent can neither pre-create nor overwrite the result, and nothing it left running can write one. The engine rejects a `verification.json` whose `reward` differs from `reward.json`'s (`:886`). | engine + kit |
| **Seeds in memory** | The grader draws hidden seeds from OS entropy inside its own process (`seed_rng`); they never touch disk. The practice grader the agent gets uses different, visible seeds and prints the raw metric only. | kit + `scripts/lint_domain.py` |
| **No self-reported numbers** | The metric is computed from the grader's own simulator state. Nothing the solver prints is parsed as a score; the wire carries observations and actions only. | kit |
| **Network** | `[agent].network_mode = "no-network"` (`block` in the RunLock) seals the sandbox before the agent starts (`protocol/episode_protocol.py:530`, `:674-700`). It then reaches only the model gateway, so the agent cannot fetch anything, including this repository. The oracle runs sealed the same way. | engine |
| **Image hygiene** | The image is built from `environment/` without `setup/`; `tests/`, `solution/` and `task.toml` lie outside that context, so no `COPY` can reach them (`sandbox-image.md`, "Task image"). The domain lint also refuses a `COPY` or `ADD` of hidden material: an image layer is readable by anyone who can run the image. | engine + `scripts/lint_domain.py` |

Real-run evidence for the first four rows, made on the engine before the Harbor migration:
[`evidence/isolation-audit.md`](evidence/isolation-audit.md) (`/opt/ale` unreadable, the
grader's stage absent, a fake verdict refused, the kit untouchable). Its 2026-11-03 note
maps each check to the 1cdbc02 code. The result path at `/logs/verifier` came with the
Harbor format; it was checked in the code and is covered by upstream's
`test_an_agent_cannot_plant_or_redirect_a_result`, not by a run of ours.

On top of isolation sit the **match locks**, which are detection for canned output:
closed_loop graders fire hidden perturbation probes (`ProbeLock`) and require the action to
respond, so a constant, replayed or observation-blind policy scores zero
(`match_lock_ok = 0`); open_loop graders re-simulate every answer under their own dynamics
and reject one that does not start where it claims; artifact_rollout is structurally immune
(the grader is the only thing that ever runs the artifact).

**Separate verification (the recommended mode for every task).** With
`[verifier] environment_mode = "separate"` the engine scores in a fresh sandbox the agent
never touched, restoring only the declared `artifacts`; the agent's own sandbox is released
first (`protocol/episode_protocol.py:315-345`). This closes a root-escalation class the
shared mode leaves open: the ALE base image leaves `/usr/local/bin` owned by the agent
account and ahead of `/usr/bin` on root's PATH, so under shared verification a program the
agent writes there (`bash`, `sh`, `rm`) runs as root the moment the engine starts
`bash test.sh` — before the grader — and can write any reward. A fresh verifier resolves
those names to the pristine `/usr/bin`. Defense in depth inside the kit covers the residual
case where the agent's own `run.sh` runs (deprivileged) during a separate verify and could
still plant for a later root exec: `deprivileged` resolves `runuser` absolutely and calls
`secure_exec_path` to take every root-PATH directory back to `root:root 0755` before the
solver starts. Two engine root execs sit outside the kit's reach and are noted in §4.
A fresh verifier still starts the image's desktop session as the agent's account, which is
why the kit clears that account before the solver starts (the "cleared account" row above).
A verifier image of a task's own (`tests/Dockerfile`) can also simply start no session: it
removes the supervisord programs and sets `DISPLAY` empty, which is how the engine tells
that a sandbox has no desktop to wait for (`arm/longtail_manipulation_recovery` does this).

**What "hidden" means, precisely.** The anchor and the oracle are hidden from the *tested
agent during its run*, by stage isolation. They are **not** secrets from maintainers or from
the public: this repository is public and every task folder is in it
whole, `tests/anchor.json` included (the engine's loader refuses a folder without
`solution/solve.sh`, `tasksets/manifest.py:401-412`, and a stub would change the task's
identity). A human can read the anchor; the agent in the sandbox cannot, and the agent is
what is being measured.

## 3. What a verdict proves

The grader runs as root inside the sandbox after the agent phase, owns its simulator, draws
its seeds from OS entropy through a keyed hash (`robotics_grader.seeds`: no output predicts
another, so the probe perturbations a solver sees reveal nothing about later seeds), and
writes the verdict itself. The engine records the episode (`lock.json` with the task
identity, the engine commit and the `ale_verify` content hash; `result.json`;
`verification.json`). A result therefore says: this agent's artifact, scored by this
grader at this task identity, on seeds nobody saw in advance.

The registry (`scripts/export_registry.py`) publishes each task's identity history, so a
result can be matched to the exact committed tree it ran against.

## 4. Honest limits

- **The host.** An episode record is a recording made on whoever's machine ran it, not proof
  that it happened under controlled conditions. Trust the runs whose operator you trust.
- **Network during verify (an engine issue, still open at 1cdbc02).** The engine opens egress
  before the verify stage (`protocol/episode_protocol.py:795`), and a closed_loop task runs
  the agent's solver during that stage. The solver can therefore reach the internet while it
  is being scored, and download anything public, this repository's `solution/` included; so
  can a process the agent left running, which keeps running through shared verification.
  Separate verification does not close it: the verifier sandbox is provisioned with an open
  network (`:443`). The hidden seeds never leave the grader's memory, but a solver that
  fetches the reference controller can play it. Until the engine keeps the verify stage
  offline for the solver, check formal runs' submissions and traces for network use.
  The kit does not take the network from the solver: root in the sandbox cannot make a
  network namespace (no `CAP_SYS_ADMIN`), and a seccomp filter cannot tell loopback from
  the internet, so it would break a solver that talks to its own helper over TCP. A task
  that promises its solver no network starts it through a filter of its own (several do,
  with a small seccomp launcher that allows only local sockets). What the kit adds is that
  agent code cannot get itself started outside such a launcher: see the next item.
- **What the cleared account does not cover.** `clear_agent_account` knows supervisord, the
  base image's init. Under another init it still kills the account's processes, and if they
  keep coming back it leaves them stopped and reports `cleared: False`; `Solver` and
  `deprivileged` log that and go on, so a grader whose rules lean on the confinement calls
  `clear_agent_account()` itself and fails the run when the report is not `cleared`. A grader
  that starts agent code with a launcher of its own and never through `Solver` or
  `deprivileged` has to call it too, or to run that code as another account while it keeps
  the agent's account stopped for the whole stage, as the assembly cells and the cell kits
  do (probed with their own launchers on 2026-11-18: `nobody` could neither write a file the
  session starts from nor signal it, and the stopped session's display never answered). The
  other account alone is not enough: while the session runs, its X display takes any local
  account, synthetic input included (the base image starts `Xvnc -SecurityTypes None`).
  Programs that are not the agent's account but take its requests stay reachable where the
  image runs them (in the base image, root's system D-Bus goes down with the session; the
  engine's guest service is not reachable from the account).
- **The probe lock is not a proof of honesty.** It catches a policy that ignores its input;
  a solver that notices a repeated `t` can vary its answer on purpose. Isolation, not the
  lock, keeps a score honest.
- **Two engine root execs the kit cannot guard.** The kit reclaims root's PATH before it
  runs agent code, but the engine itself runs programs by bare name as root around verify.
  (1) In a *separate* verify every such name the engine resolves before the grader starts
  (`chmod` on restore, `rm`/`sh` to clear stages and reset `/logs/verifier`, then
  `bash test.sh`) runs in the **fresh** verifier before any agent code, so there is no plant
  to shadow it; the kit's reclaim then protects the names run after `run.sh` starts (the
  kit's own `runuser`, and the engine's post-verify `rm`/`sh`). (2) In a *shared* verify the
  agent's sandbox already holds whatever it planted, so the engine's first `bash test.sh`
  can run an agent file as root before the grader loads — which the kit cannot prevent (it
  has not run yet). That is the case separate verification exists to remove, which is why
  every task should use it. The engine also runs `_sanitize_for_retention` (bare `rm`/`sh`)
  on the released solver sandbox, but that is after its reward is read and never feeds a
  score. None of this is fixable here without modifying the pinned `vendor/ale`.
- **Public anchors.** Because the repository is public and complete, the target value is
  public knowledge outside the sandbox. The agent phase runs in a no-network sandbox that
  does not contain it; the benchmark does not measure what a human who read the repository
  could type into an instruction.
- **A simulator the agent holds.** When a task gives the agent a copy of its simulator, the
  episode seed is the only secret: seed it with 64 bits (`rng.seed(bits=64)`) and draw every
  hidden value through a generator whose outputs reveal nothing (a keyed hash), or the first
  observations give the seed away.

## 5. Constraints the tooling must preserve

- The task image never contains `tests/`, `solution/`, the anchor, or `task.toml`.
- `environment/setup/payload/` never reads the anchor or the grader; the practice grader
  prints the raw metric only, never a ratio or score.
- `rewards` in the verdict carries only the gated `reward`; raw values ride in `metrics`.
- Credentials live in the engine's `.env` on the host, never in a task folder.
- Run outputs (`runs/`) are git-ignored.
