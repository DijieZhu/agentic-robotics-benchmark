"""Verify-stage basics: where things are, who runs them, and how a run ends.

The framework tells a stage script where it is (`ALE_STAGE_DIR`) and whose home it works
on (`ALE_HOME`) through the environment, and the result goes to Harbor's verifier directory;
this module is the only place that knows them, so a task never hardcodes a path the
framework owns.
"""

import json
import os
import shutil
import stat
import subprocess
import sys

#: Harbor's verifier directory: `reward.json` (the score the engine reads) and
#: `verification.json` (the complete result) go here.
VERIFIER_DIR = "/logs/verifier"

#: The PATH the kit resolves its own root-run helpers against — fixed, and free of any
#: directory an unprivileged account can write. The image's PATH puts an agent-writable
#: `/usr/local/bin` ahead of `/usr/bin` (the base image chowns node/npm there for the
#: agent CLI), so resolving a root helper against the *image's* PATH would let the agent
#: decide what runs as root. This list is the standard system `sbin`/`bin` pair only.
_SYSTEM_PATH = "/usr/sbin:/usr/bin:/sbin:/bin"

#: Directories the container keeps on root's PATH that an image may leave writable by the
#: agent account. `secure_exec_path` reclaims any of these (and any other PATH entry) that
#: is not root-owned 0755 before the kit runs agent code as a child of root.
_GUARDED_PATH_DIRS = (
    "/opt/venv/bin",
    "/usr/local/sbin",
    "/usr/local/bin",
    "/usr/sbin",
    "/usr/bin",
    "/sbin",
    "/bin",
)

__all__ = [
    "VERIFIER_DIR",
    "agent_user",
    "secure_exec_path",
    "deprivileged",
    "fail",
    "require",
    "stage_dir",
    "verdict_path",
    "write_verdict",
]


def stage_dir():
    """This stage's folder inside the sandbox (`/opt/ale/tests`: root-only, like all of /opt/ale).

    Assets that must not exist while the agent works — the anchor, grader config, hidden
    reference data — live here, because a stage folder is copied in when its stage runs.
    """
    return os.environ.get("ALE_STAGE_DIR", os.path.dirname(os.path.abspath(sys.argv[0])))


def verdict_path():
    """Harbor's reward file, where the engine reads the score back.

    The agent's UID cannot pre-create or overwrite it: the engine empties `/logs/verifier`
    and gives it to the framework alone before every verification attempt. Write it through
    `write_verdict`, never directly: the file holds a flat object of numbers, not the
    kit's `{"rewards": ..., "metrics": ...}` envelope.
    """
    return os.path.join(VERIFIER_DIR, "reward.json")


def agent_home():
    """The unprivileged account's home — the workspace, per the image's declared user."""
    return os.environ.get("ALE_HOME", "/home/user")


def agent_user():
    """The account the agent ran as, derived from its home.

    task.toml's `[agent].user` names it (default `user`) and the framework passes its home;
    taking the leaf of the home keeps the two consistent without a second source of truth.
    """
    return os.path.basename(agent_home().rstrip("/")) or "user"


def fail(reason):
    """End the run as a TASK ERROR, not as a zero.

    The distinction is load-bearing: a zero says the agent did not solve the task, and a
    task error says the task is broken. Conflating them lets a defective grader read as a
    stream of legitimately failing agents.
    """
    print("[grader] TASK ERROR: {}".format(reason), file=sys.stderr, flush=True)
    raise SystemExit(1)


def require(condition, reason):
    """`fail` unless the condition holds — for the grader's own preconditions."""
    if not condition:
        fail(reason)


def write_verdict(verdict, path=None, diagnostic=None):
    """Write the verdict the framework reads back.

    `verdict` is `{"rewards": {...}, "metrics": {...}}`. The split is load-bearing under
    `ale validate`: `reward`, rounded half-up to two decimals as the engine rounds it, must be
    0.0 on an untouched sandbox and 1.0 for the oracle, so `rewards` carries only the gated
    score, the single key `reward` in [0, 1], and everything worth keeping but not gating
    (the raw metric, the capped ratio, lock outcomes, abort counts) goes in `metrics`. Other
    reward keys are recorded in verification.json but never gated.

    With no `path`, inside the engine's verify stage (`ALE_VERIFY_PYTHON` is set), the result
    goes through the engine's `ale_verify.write`, which it stages into the verify-stage
    interpreter. It writes Harbor's `/logs/verifier/reward.json` (`{"reward": ...}`, the only
    value the engine reads) and `verification.json` beside it: the complete result, with
    `metrics`, the exact unrounded reward as `metrics.reward_exact`, and the grader's
    `diagnostic` text when there is one (the argument, or the verdict's own `diagnostic`).

    With a `path`, the plain envelope is written there instead, for runs outside a sandbox
    (the kit's selftest, a host-side dry run); a path inside the verifier directory is a task
    error, because the engine reads only Harbor's flat result there. A grader run by hand
    outside the engine, such as a maintainer re-measuring an anchor inside the task image
    (`bash test.sh` or `python3 verify.py` in the staged folder), gets the same through the
    `ROBOTICS_GRADER_VERDICT_PATH` environment variable without changing the grader.
    """
    rewards = verdict.get("rewards") or {}
    metrics = verdict.get("metrics") or {}
    require("reward" in rewards, "rewards map has no primary 'reward' key")
    require(all(0.0 <= float(v) <= 1.0 for v in rewards.values()),
            "every gated reward must lie in [0, 1]; report raw values under metrics")
    rewards = {k: float(v) for k, v in rewards.items()}
    metrics = {k: float(v) for k, v in metrics.items()}
    require("reward_exact" not in metrics,
            "reward_exact is the kit's own metric (the unrounded reward); rename the task's")
    diagnostic = diagnostic if diagnostic is not None else verdict.get("diagnostic")

    if path is None and not os.environ.get("ALE_VERIFY_PYTHON"):
        path = os.environ.get("ROBOTICS_GRADER_VERDICT_PATH") or None
    if path is not None:
        target = os.path.realpath(path)
        require(not (target == VERIFIER_DIR or target.startswith(VERIFIER_DIR + os.sep)),
                "write_verdict(path=...) writes the kit's envelope, which the engine cannot "
                "read in {}; call write_verdict(verdict) with no path".format(VERIFIER_DIR))
    if path is None:
        try:
            from ale_verify import write
        except ImportError:
            fail("ale_verify is not importable: the engine stages it into the verify-stage "
                 "interpreter (3.12+, named by ALE_VERIFY_PYTHON); outside the engine pass a "
                 "path or set ROBOTICS_GRADER_VERDICT_PATH")
        result = {"reward": rewards["reward"],
                  "metrics": dict(metrics, reward_exact=rewards["reward"])}
        if len(rewards) > 1:
            result["rewards"] = rewards
        if diagnostic:
            result["diagnostic"] = str(diagnostic)
        write(result)   # reward.json and verification.json, atomically, reward rounded
        return

    envelope = {"rewards": rewards, "metrics": metrics}
    if diagnostic:
        envelope["diagnostic"] = str(diagnostic)
    with open(path, "w") as handle:
        json.dump(envelope, handle)


def _resolve_root_tool(name):
    """Absolute path of a system program the kit runs AS ROOT, resolved against
    `_SYSTEM_PATH` so an agent-writable PATH entry (the base image's `/usr/local/bin`)
    cannot decide what root executes.

    Falls back to the bare name on an image that keeps the tool elsewhere — the fixed
    path is then unavailable anyway, and `secure_exec_path` still denies the agent the
    chance to shadow it. A resolved path is absolute, so execution never consults PATH.
    """
    found = shutil.which(name, path=_SYSTEM_PATH)
    return found or name


def _insecure_dir(info):
    """True when a directory's own metadata would let a non-root account create, rename
    or delete entries in it — i.e. plant a program there. Only root ownership with no
    group/other write is safe for a directory on root's PATH."""
    return (
        info.st_uid != 0
        or info.st_gid != 0
        or bool(info.st_mode & (stat.S_IWGRP | stat.S_IWOTH))
    )


def secure_exec_path(dirs=None):
    """Reclaim every directory on root's PATH that the agent account could plant into,
    so nothing agent-writable precedes the system tools root resolves by name.

    The ALE base image leaves `/usr/local/bin` owned by the agent account and ahead of
    `/usr/bin` on root's PATH, so a file the agent writes there (`bash`, `sh`, `rm`,
    `runuser`, …) would run as root the next time the engine or the kit executes that name.
    In a separate verifier the agent's own `run.sh` runs (deprivileged) during verify and
    could plant such a file before a later root exec; reclaiming the directories to
    `root:root 0755` before any agent code runs closes that window. The agent can still
    read and execute what is already there (its CLI, node); it just cannot add, replace by
    rename, or remove an entry.

    Idempotent and best-effort: a directory already root-owned 0755, or absent, is left
    alone; a directory that cannot be reclaimed is reported and skipped rather than fatal.
    A no-op unless this process is root (a kit that is not root cannot be hijacked through
    root's PATH, and could not chown anyway). Returns the list of paths it changed.
    """
    if os.name == "nt" or not hasattr(os, "geteuid") or os.geteuid() != 0:
        return []
    candidates = list(_GUARDED_PATH_DIRS) if dirs is None else list(dirs)
    if dirs is None:
        candidates += [d for d in os.environ.get("PATH", "").split(os.pathsep) if d]
    changed, seen = [], set()
    for entry in candidates:
        real = os.path.realpath(entry)
        if real in seen:
            continue
        seen.add(real)
        try:
            info = os.stat(real)
        except OSError:
            continue  # absent PATH entry: nothing to reclaim
        if not stat.S_ISDIR(info.st_mode) or not _insecure_dir(info):
            continue
        try:
            os.chown(real, 0, 0)
            os.chmod(real, 0o755)
        except OSError as exc:
            print("[grader] could not secure root PATH dir {}: {}".format(real, exc),
                  file=sys.stderr, flush=True)
            continue
        print("[grader] secured root PATH dir {} (was uid={} gid={} mode={:o})".format(
            real, info.st_uid, info.st_gid, stat.S_IMODE(info.st_mode)),
            file=sys.stderr, flush=True)
        changed.append(real)
    return changed


def deprivileged(argv, **kwargs):
    """Run `argv` as the agent account instead of as root.

    Two callers need this and both are security-critical:

    * a live solver the agent wrote (closed_loop) — at root it could read the grader's
      memory, environment and `/opt/ale`;
    * deserializing an artifact the agent produced (artifact_rollout) — `torch.load` and
      `pickle` execute arbitrary code, so doing it as root hands over the grader.

    Yama's default `ptrace_scope` blocks tracing across UIDs, so the boundary holds even
    though both processes share a sandbox.

    Before any agent code runs, root's PATH directories are reclaimed
    (`secure_exec_path`), and `runuser` — the one program the kit starts as root — is
    resolved to an absolute path against the fixed system PATH, so a planted
    `/usr/local/bin/runuser` can never run as root. The child's environment is left as the
    kit inherited it: `runuser -u <user> --` preserves it, and the agent's `run.sh` is the
    agent's own code, which must keep its normal PATH (the venv's `python3` lives on it).
    After `secure_exec_path`, that inherited PATH holds no agent-writable directory anyway.
    """
    secure_exec_path()
    return subprocess.Popen(  # noqa: S603 — caller-fixed argv, no shell
        [_resolve_root_tool("runuser"), "-u", agent_user(), "--"] + list(argv), **kwargs
    )
