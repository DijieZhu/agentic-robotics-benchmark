"""Skill definitions are well-formed: every SKILL.md has valid YAML frontmatter with a
`name` + `description`, the onboarding skills exist, the router references each mode
skill, and the trajectory skill names the script it drives. A malformed skill (bad
frontmatter, missing hand-off) should fail CI here rather than silently mis-load in an
agent."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

SKILLS_DIR = Path(__file__).resolve().parent.parent / "skills"

# The onboarding router + one skill per eval mode.
ONBOARDING_SKILLS = {
    "onboard-task",
    "onboard-open-loop",
    "onboard-closed-loop",
    "onboard-artifact-rollout",
}


# The old shapes' vocabulary (see test_no_skill_cites_the_retired_gate).
RETIRED_VOCABULARY = (
    "run_task.py", "--agent example", "grader_protocol", "eval.mode",
    "description.md", "practice_grader/grade.py at the task root",
    # the pre-core/v1 layout, which core/v1 replaced with setup/payload/ for the
    # visible channel, a vendored verify/robotics_grader/, a per-task
    # image/Dockerfile and verify/verify.py as the grader entry
    "files/", "kits/", "ale-domain/kits", "ale-domain/images",
    "robotics-cli", "verify/grade.py",
    # the core/v1 layout the Harbor format replaced on 2026-11-03: the engine
    # at vendor/ale loads none of these names, so a skill that tells an agent
    # to write one produces a folder `ale lint` rejects
    "task.yaml", "core/v1", "image/Dockerfile", "image/requirements.txt",
    "image: {kind", "setup/run.sh", "oracle/run.sh", "verify/run.sh",
    "verify/verify.py", "verify/env.py", "verify/anchor.json",
    "verify/grader_config.json", "verify/robotics_grader", "params:",
    # where the old engine staged the oracle and the verdict
    ".ale-oracle", "/opt/ale/verify", "ALE_VERDICT_PATH", "ALE_VERIFICATION_PATH",
    # the retired reproduction / open-ended genre split (design 20260902 §8)
    "genre", "reproduction", "open_ended", "open-ended", "faithful", "\"probe\": true",
)

# The skill that records model runs' trajectories, and the script it drives.
TRAJECTORY_SKILL = "agent-trajectory-archiving"
TRAJECTORY_COMMANDS = ("inventory", "curate", "export", "scan", "upload", "sample")


def _all_skill_files() -> list[Path]:
    return sorted(SKILLS_DIR.rglob("SKILL.md"))


def _frontmatter(text: str) -> dict:
    """Parse the leading `---`-delimited YAML frontmatter block of a SKILL.md."""
    if not text.startswith("---"):
        raise ValueError("no frontmatter fence")
    _, fm, _ = text.split("---", 2)
    data = yaml.safe_load(fm)
    if not isinstance(data, dict):
        raise ValueError("frontmatter is not a mapping")
    return data


def test_onboarding_skills_present():
    names = {p.parent.name for p in _all_skill_files() if p.parent != SKILLS_DIR}
    missing = ONBOARDING_SKILLS - names
    assert not missing, f"missing onboarding skills: {sorted(missing)}"


@pytest.mark.parametrize("skill_file", _all_skill_files(), ids=lambda p: str(p.parent.name or "root"))
def test_skill_frontmatter_valid(skill_file: Path):
    fm = _frontmatter(skill_file.read_text(encoding="utf-8"))
    assert isinstance(fm.get("name"), str) and fm["name"].strip(), f"{skill_file}: missing name"
    assert isinstance(fm.get("description"), str) and fm["description"].strip(), (
        f"{skill_file}: missing description")


def test_router_hands_off_to_every_mode_skill():
    router = (SKILLS_DIR / "onboard-task" / "SKILL.md").read_text(encoding="utf-8")
    for mode_skill in ("onboard-open-loop", "onboard-closed-loop", "onboard-artifact-rollout"):
        assert mode_skill in router, f"router does not reference {mode_skill}"


def test_router_declares_give_up_and_the_gate():
    router = (SKILLS_DIR / "onboard-task" / "SKILL.md").read_text(encoding="utf-8").lower()
    # The router owns the shared abort rules and names the admission gate. A skill that
    # cites the wrong gate sends an agent to run a command that no longer exists.
    assert "give-up" in router or "give up" in router, "router missing give-up rules"
    assert "ale validate" in router, "router does not name the `ale validate` gate"
    assert "harness.ale_onboard" in router, "router does not name the onboarding entry point"
    assert "lint_domain" in router, "router does not name the domain lint"


def test_no_skill_cites_the_retired_gate():
    """The old shape's vocabulary must not survive in a skill.

    These names are load-bearing instructions, not prose: an agent told to run
    `harness/run_task.py --agent example`, or to write a `grader/` directory subclassing
    `grader_protocol`, produces a task the engine cannot load.
    """
    retired = RETIRED_VOCABULARY
    for skill_file in SKILLS_DIR.rglob("SKILL.md"):
        if skill_file.parent.name not in ONBOARDING_SKILLS:
            continue
        text = skill_file.read_text(encoding="utf-8")
        for name in retired:
            assert name not in text, f"{skill_file}: still cites the retired {name!r}"


def test_skills_name_the_current_layout():
    """The load-bearing paths of the Harbor layout must be spelled out, so an agent
    following a skill lands files where the engine and the domain lint look for them."""
    for skill_file in SKILLS_DIR.rglob("SKILL.md"):
        if skill_file.parent.name not in ONBOARDING_SKILLS:
            continue
        text = skill_file.read_text(encoding="utf-8")
        for current in ("task.toml", "environment/Dockerfile", "environment/setup/payload",
                        "tests/verify.py", "tests/robotics_grader", "tests/anchor.json",
                        "solution/solve.sh"):
            assert current in text, f"{skill_file}: does not name {current!r}"


def test_skills_name_the_engine_oracle_gate():
    """The engine fails validation unless the oracle's `reward`, rounded to two decimals,
    is 1.0 (`oracle_not_full`; since 90a7c1c); the old `partial_oracle` warning and the
    "domain policy" framing are gone. A skill that still calls the rule a warning teaches an
    agent to accept a broken oracle; one that trusts the exit status alone mistakes the
    null-anchor bootstrap run for a broken task, so every skill also names the record
    reader."""
    for skill_file in SKILLS_DIR.rglob("SKILL.md"):
        if skill_file.parent.name not in ONBOARDING_SKILLS:
            continue
        # Prose wraps at ~95 columns, so match on whitespace-collapsed text.
        text = " ".join(skill_file.read_text(encoding="utf-8").split())
        assert "oracle_not_full" in text, f"{skill_file}: does not name the engine's failure"
        assert "harness/ale_onboard.py" in text, f"{skill_file}: does not name the record reader"
        for stale in ("partial_oracle", "domain policy", "only warns", "engine itself only"):
            assert stale not in text, f"{skill_file}: still says {stale!r}"


def test_trajectory_skill_names_its_script_and_no_retired_vocabulary():
    """The trajectory skill drives scripts/trajectories.py: it names the script, every
    subcommand and the parameters that hold every path, and carries neither the old task
    shapes' vocabulary nor the archived repositories' (promote, publish, mirror)."""
    skill_file = SKILLS_DIR / TRAJECTORY_SKILL / "SKILL.md"
    text = skill_file.read_text(encoding="utf-8")
    assert _frontmatter(text)["name"] == TRAJECTORY_SKILL
    assert "scripts/trajectories.py" in text
    for command in TRAJECTORY_COMMANDS:
        assert f"`{command}`" in text, f"{skill_file}: does not name the {command!r} subcommand"
    for flag in ("--runs-dir", "--archive", "--dest", "--secrets-from"):
        assert flag in text, f"{skill_file}: does not name the parameter {flag}"
    lowered = text.lower()
    for name in RETIRED_VOCABULARY + ("promote", "publish", "mirror"):
        found = name in lowered if name.islower() else name in text
        assert not found, f"{skill_file}: cites the retired {name!r}"
