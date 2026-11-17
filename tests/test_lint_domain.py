"""scripts/lint_domain.py — the domain's policy lint, exercised on throwaway task folders.

The interesting property after the 2026-09-02 narrative change is what the lint no longer
asks for: `metadata.genre` and the paper block are gone (one kind of task, design §8), and a
task without them must lint clean. The batteries that stay — hardware-invariant metric,
positive anchor, `full_at`, anchor provenance, visible-surface isolation — are pinned so the
genre removal cannot be mistaken for a general loosening. The placement battery (2026-09-07,
flat layout; the parent folder is `metadata.platform` since 2026-10-12:
`tasks/<platform>/<slug>/`) replaces the retired staging-area promote check. On 2026-10-15
`metadata.direction` and `metadata.metric` were retired: the metric and its direction live in
the anchor only, and `metadata.platform` must be one of tasks/platforms.json. On 2026-11-03
the tasks moved to the Harbor layout of the engine at 1cdbc02 (task.toml, environment/,
tests/, solution/): `[task].name` is `grounded-intelligence-initiative/<slug>`, the anchor is
tests/anchor.json, the payload environment/setup/payload/, and a folder still in the core/v1
form is reported rather than skipped.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
_SPEC = importlib.util.spec_from_file_location(
    "lint_domain", REPO_ROOT / "scripts" / "lint_domain.py")
lint_domain = importlib.util.module_from_spec(_SPEC)
sys.modules["lint_domain"] = lint_domain
_SPEC.loader.exec_module(lint_domain)  # type: ignore[union-attr]

_MANIFEST = """\
schema_version = "1.4+ale.1"
artifacts = ["/home/user/submission"]

[task]
name = "{task_id}"

[metadata]
{platform}{extra}
[environment]
cpus = 2
memory_mb = 4096

[agent]
timeout_sec = 3600
network_mode = "no-network"

[verifier]
timeout_sec = 900
"""

_ANCHOR = {"metric": "mean_episode_reward", "direction": "higher", "value": 472.07, "full_at": 0.825,
           "source": "solution/solve.sh real graded run — NOT the paper's printed figure"}

_DOCKERFILE = "FROM ghcr.io/agentslastexam/container-ubuntu22-base:0.1.1\n"
_GRADE_PY = "environment/setup/payload/practice_grader/grade.py"


def _task(tmp_path: Path, *, name: str = "probe", metric: str = "mean_episode_reward",
          extra_metadata: str = "", anchor: dict | None = None,
          platform: str = "uav", task_id: str | None = None,
          collection: str = "tasks") -> Path:
    """A minimal Harbor-layout task folder that the lint accepts as shipped (`metric` lands
    in the anchor, the one place a task declares it). `platform=""` leaves the key out."""
    task = tmp_path / collection / platform / name
    (task / "environment" / "setup" / "payload" / "practice_grader").mkdir(parents=True)
    (task / "tests").mkdir()
    (task / "solution").mkdir()
    (task / "task.toml").write_text(_MANIFEST.format(
        task_id=f"{lint_domain.TASK_ORG}/{name}" if task_id is None else task_id,
        platform=f'platform = "{platform}"\n' if platform else "",
        extra=extra_metadata))
    (task / "instruction.md").write_text("Leave your submission in /home/user/submission.\n")
    (task / "environment" / "Dockerfile").write_text(_DOCKERFILE)
    (task / "environment" / "setup" / "setup.sh").write_text("#!/usr/bin/env bash\n")
    (task / "tests" / "test.sh").write_text(
        '#!/usr/bin/env bash\nset -euo pipefail\nexec "$ALE_VERIFY_PYTHON" verify.py\n')
    (task / "tests" / "anchor.json").write_text(
        json.dumps(dict(_ANCHOR, metric=metric) if anchor is None else anchor))
    (task / "solution" / "solve.sh").write_text("#!/usr/bin/env bash\n")
    (task / _GRADE_PY).write_text("print({'mean_episode_reward': 1.0})\n")
    return task


def _edit_manifest(task: Path, old: str, new: str) -> None:
    path = task / "task.toml"
    text = path.read_text()
    assert old in text, old
    path.write_text(text.replace(old, new))


def test_a_task_without_genre_or_paper_lints_clean(tmp_path):
    assert lint_domain.lint_task(_task(tmp_path)) == []


def test_a_task_still_carrying_genre_is_not_flagged(tmp_path):
    """The retired keys are stale, not wrong — nothing reads them any more."""
    task = _task(tmp_path, extra_metadata=(
        'genre = "reproduction"\n'
        'paper = {title = "x", year = 2009, reference_repo = "TODO", commit = "TODO"}\n'))
    problems = lint_domain.lint_task(task)
    assert problems == [], problems


def test_the_shipped_domain_lints_clean():
    """The real tree — template excluded from task batteries, the worked example included, and the
    repository root as the default target (only tasks/ and templates/ are walked)."""
    landed = sorted(REPO_ROOT.glob("tasks/*/*/" + lint_domain.MANIFEST))
    assert landed, "no task.toml under tasks/: the lint would pass on nothing"
    assert (REPO_ROOT / "templates" / "task" / lint_domain.MANIFEST).is_file()
    assert lint_domain.lint_domain(REPO_ROOT) == []
    assert lint_domain.lint_paths([REPO_ROOT]) == []
    assert lint_domain.DEFAULT_DOMAIN == REPO_ROOT


def test_the_template_carries_the_benchmark_org():
    """The walk runs no task battery on templates/, so the org check would not reach it."""
    for manifest in sorted((REPO_ROOT / "templates").rglob(lint_domain.MANIFEST)):
        name = lint_domain.read_manifest(manifest.parent)["task"]["name"]
        parts = lint_domain.split_name(name)
        assert parts is not None and parts[0] == lint_domain.TASK_ORG, name


# --------------------------------------------------------------------------- #
# battery 6: completeness
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("entry", ["instruction.md", "tests/test.sh", "solution/solve.sh"])
def test_each_missing_required_file_is_named(tmp_path, entry):
    task = _task(tmp_path)
    (task / entry).unlink()
    problems = lint_domain.lint_task(task)
    assert len(problems) == 1 and f"no {entry}" in problems[0], problems


def test_a_missing_anchor_is_reported_once(tmp_path):
    task = _task(tmp_path)
    (task / "tests" / "anchor.json").unlink()
    problems = lint_domain.lint_task(task)
    assert len(problems) == 1 and "no tests/anchor.json" in problems[0], problems


@pytest.mark.parametrize("key", ["docker_image", "vm_image"])
def test_an_image_named_directly_stands_in_for_the_dockerfile(tmp_path, key):
    task = _task(tmp_path)
    (task / "environment" / "Dockerfile").unlink()
    problems = lint_domain.lint_task(task)
    assert len(problems) == 1, problems
    assert "no environment/Dockerfile and no [environment].docker_image" in problems[0]
    _edit_manifest(task, "[environment]\n", f'[environment]\n{key} = "ghcr.io/example/robotics:1.0"\n')
    assert lint_domain.lint_task(task) == []


@pytest.mark.parametrize("line", ['schema_version = "1.4"', 'schema_version = "core/v1"', ""])
def test_schema_version_must_be_the_engine_s(tmp_path, line):
    task = _task(tmp_path)
    _edit_manifest(task, 'schema_version = "1.4+ale.1"', line)
    problems = lint_domain.lint_task(task)
    assert len(problems) == 1 and 'schema_version = "1.4+ale.1"' in problems[0], problems


@pytest.mark.parametrize("task_id", ["probe", "grounded-intelligence-initiative/Probe",
                                     "Grounded-Intelligence-Initiative/probe",
                                     "grounded-intelligence-initiative/probe/v2",
                                     "grounded-intelligence-initiative/", ""])
def test_the_task_name_must_be_org_slash_slug(tmp_path, task_id):
    task = _task(tmp_path, task_id=task_id)
    problems = lint_domain.lint_task(task)
    assert len(problems) == 1 and "is not 'org/slug'" in problems[0], problems
    # placement compares only a well-formed name: the finding is not doubled
    assert lint_domain.lint_paths([task]) == problems


def test_a_missing_task_table_is_a_missing_name(tmp_path):
    task = _task(tmp_path)
    _edit_manifest(task, f'[task]\nname = "{lint_domain.TASK_ORG}/probe"\n', "")
    problems = lint_domain.lint_task(task)
    assert len(problems) == 1 and "[task].name None" in problems[0], problems


def test_the_org_must_be_the_benchmark_s(tmp_path):
    assert lint_domain.TASK_ORG == "grounded-intelligence-initiative"
    task = _task(tmp_path, task_id="agentslastexam/probe")
    problems = lint_domain.lint_paths([task])
    assert len(problems) == 1, problems
    assert "'agentslastexam/probe'" in problems[0]
    assert "'grounded-intelligence-initiative/<slug>'" in problems[0]


def test_an_unreadable_manifest_is_reported(tmp_path):
    task = _task(tmp_path)
    (task / "task.toml").write_text('schema_version = "1.4+ale.1"\n[task\nname = "x"\n')
    problems = lint_domain.lint_task(task)
    assert len(problems) == 1 and "cannot read task.toml" in problems[0], problems
    # placement leaves an unreadable manifest to battery 6
    assert lint_domain.lint_paths([task]) == problems


def test_a_validate_table_is_refused(tmp_path):
    task = _task(tmp_path)
    (task / "task.toml").write_text((task / "task.toml").read_text() + "\n[validate]\nruns = 3\n")
    problems = lint_domain.lint_task(task)
    assert len(problems) == 1 and "`validate`" in problems[0], problems
    assert "`full_at` in tests/anchor.json" in problems[0]


# --------------------------------------------------------------------------- #
# battery 7: identity, platform and placement at tasks/<platform>/<slug>/
# --------------------------------------------------------------------------- #
def test_placement_accepts_the_landing_path(tmp_path):
    task = _task(tmp_path)                       # tasks/uav/probe, platform uav
    assert lint_domain.tasks_root_of(task) == (tmp_path / "tasks").resolve()
    assert lint_domain.lint_paths([task]) == []
    assert lint_domain.lint_paths([tmp_path]) == []            # a root holding tasks/
    assert lint_domain.lint_paths([tmp_path / "tasks"]) == []  # the collection itself
    assert lint_domain.main([str(tmp_path)]) == 0


def test_placement_refuses_a_folder_named_unlike_its_slug(tmp_path):
    task = _task(tmp_path)
    renamed = task.rename(task.parent / "probe_v2")
    problems = lint_domain.lint_paths([renamed])
    assert len(problems) == 1, problems
    assert ("folder name 'probe_v2' != 'probe', the slug of task.toml [task].name "
            "'grounded-intelligence-initiative/probe'") in problems[0]
    assert lint_domain.main([str(renamed)]) == 1


def test_placement_refuses_the_wrong_platform_folder(tmp_path):
    task = _task(tmp_path)                       # manifest says platform: uav
    (tmp_path / "tasks" / "control").mkdir()     # its direction, not its platform
    moved = task.rename(tmp_path / "tasks" / "control" / "probe")
    problems = lint_domain.lint_paths([moved])
    assert len(problems) == 1, problems
    assert "tasks/control/probe/" in problems[0] and "metadata.platform 'uav'" in problems[0]
    assert "tasks/uav/probe/" in problems[0]   # the fix is spelled out


def test_placement_refuses_a_task_nested_too_deep(tmp_path):
    task = _task(tmp_path)
    (tmp_path / "tasks" / "uav" / "extra").mkdir()
    moved = task.rename(tmp_path / "tasks" / "uav" / "extra" / "probe")
    problems = lint_domain.lint_paths([moved])
    assert any("exactly at tasks/<platform>/<slug>/" in p for p in problems), problems


def test_placement_is_not_checked_outside_a_tasks_root(tmp_path):
    """A template or a scratch copy is not a landed task; the other batteries still run."""
    task = _task(tmp_path)
    (tmp_path / "scratch").mkdir()
    moved = task.rename(tmp_path / "scratch" / "renamed")
    assert lint_domain.tasks_root_of(moved) is None
    assert lint_domain.lint_paths([moved]) == []
    assert lint_domain.lint_task(moved) == []


# --------------------------------------------------------------------------- #
# the retired core/v1 form
# --------------------------------------------------------------------------- #
def _old_layout_task(tmp_path: Path, name: str = "old_probe") -> Path:
    task = tmp_path / "tasks" / "uav" / name
    (task / "verify").mkdir(parents=True)
    (task / "task.yaml").write_text(f"spec_type: core/v1\nname: {name}\n")
    (task / "verify" / "anchor.json").write_text(json.dumps(_ANCHOR))
    return task


def test_a_folder_in_the_retired_form_is_reported_not_skipped(tmp_path):
    """The engine finds tasks by task.toml alone, so `ale lint` passes such a folder over."""
    _task(tmp_path)
    old = _old_layout_task(tmp_path)
    for target in (tmp_path, tmp_path / "tasks", tmp_path / "tasks" / "uav", old):
        problems = lint_domain.lint_paths([target])
        assert len(problems) == 1, (target, problems)
        assert problems[0].startswith("old_probe: task.yaml without task.toml")
        assert "retired core/v1 form" in problems[0]
    assert lint_domain.main([str(old)]) == 1


def test_a_task_yaml_inside_a_converted_task_is_not_a_task(tmp_path):
    """A vendored library's task.yaml below a task.toml is that task's file, not a task."""
    task = _task(tmp_path)
    vendored = task / "environment" / "vendor" / "lib"
    vendored.mkdir(parents=True)
    (vendored / "task.yaml").write_text("steps: []\n")
    assert lint_domain.lint_paths([tmp_path]) == []
    assert lint_domain.lint_paths([task]) == []


def test_an_empty_tree_is_still_nothing_to_lint(tmp_path):
    (tmp_path / "empty").mkdir()
    problems = lint_domain.lint_paths([tmp_path / "empty"])
    assert len(problems) == 1 and "no task.toml" in problems[0], problems


# --------------------------------------------------------------------------- #
# batteries 1, 2, 5: the anchor
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("metric", ["planning_time_ms", "control_frequency_hz",
                                    "throughput_fps", "inference_latency", "fps",
                                    "inferenceLatency", "runtime_s", "time2goal"])
def test_hardware_dependent_metrics_are_still_refused(tmp_path, metric):
    anchor = dict(_ANCHOR, metric=metric)
    problems = lint_domain.lint_task(_task(tmp_path, metric=metric, anchor=anchor))
    assert any("hardware-dependent" in p for p in problems), problems


@pytest.mark.parametrize("metric", ["rms_error", "rmse", "items_total", "timesteps_to_goal",
                                    "mean_goal_distance", "success_rate", "atempo_score"])
def test_hardware_words_inside_other_tokens_are_not_refused(tmp_path, metric):
    """Whole tokens only: `rms_error` is not `ms`, `timesteps` is not `time`."""
    anchor = dict(_ANCHOR, metric=metric)
    problems = lint_domain.lint_task(_task(tmp_path, metric=metric, anchor=anchor))
    assert problems == [], problems


def test_hw_metric_token_names_the_offending_segment():
    assert lint_domain.hw_metric_token("planning_time_ms") == "time"
    assert lint_domain.hw_metric_token("control-frequency-hz") == "frequency"
    assert lint_domain.hw_metric_token("throughputFps") == "throughput"
    assert lint_domain.hw_metric_token("rms_error") is None
    assert lint_domain.hw_metric_token("") is None


def test_a_non_positive_anchor_is_still_refused(tmp_path):
    problems = lint_domain.lint_task(_task(tmp_path, anchor=dict(_ANCHOR, value=-3.0)))
    assert any("not positive" in p for p in problems), problems


def test_a_missing_full_at_names_the_engine_gate(tmp_path):
    anchor = {k: v for k, v in _ANCHOR.items() if k != "full_at"}
    problems = lint_domain.lint_task(_task(tmp_path, anchor=anchor))
    assert len(problems) == 1 and "full_at" in problems[0], problems
    # The exact-1.0 oracle is the engine's own gate (`oracle_not_full`); say so.
    assert "oracle_not_full" in problems[0] and "domain policy" not in problems[0]


@pytest.mark.parametrize("source", [None, 474, ["oracle run"], "", "   "])
def test_an_anchor_source_that_is_not_text_is_refused(tmp_path, source):
    """str(None) is "None": a null, numeric or list source must not pass as attribution."""
    problems = lint_domain.lint_task(_task(tmp_path, anchor=dict(_ANCHOR, source=source)))
    assert any("no `source`" in p for p in problems), problems


def test_an_anchor_quoting_a_published_figure_is_still_refused(tmp_path):
    anchor = dict(_ANCHOR, source="the paper reports 474")
    problems = lint_domain.lint_task(_task(tmp_path, anchor=anchor))
    assert any("paper" in p for p in problems), problems


@pytest.mark.parametrize("source", [
    # a former task's exact wording (uav_hover, removed 2026-10-15)
    "oracle implementation (oracle/run.sh, PPO on HoverAviary at the pinned gym-pybullet-drones "
    "commit): real graded run over hidden per-run seeds, never a published figure",
    "solution/solve.sh real graded run — NOT the paper's printed figure",
    "measured by the oracle, not any reported number",
    "own real run; never the printed value",
])
def test_an_anchor_disclaiming_the_paper_is_accepted(tmp_path, source):
    problems = lint_domain.lint_task(_task(tmp_path, anchor=dict(_ANCHOR, source=source)))
    assert problems == [], problems


def test_anchor_direction_must_be_higher_or_lower(tmp_path):
    """The anchor is the one place a task declares its direction (the grader reads it there)."""
    problems = lint_domain.lint_task(_task(tmp_path, anchor=dict(_ANCHOR, direction="up")))
    assert len(problems) == 1 and "`direction`" in problems[0], problems
    missing = {key: value for key, value in _ANCHOR.items() if key != "direction"}
    assert len(lint_domain.lint_task(_task(tmp_path / "missing", anchor=missing))) == 1
    assert lint_domain.lint_task(_task(tmp_path / "ok", anchor=dict(_ANCHOR, direction="lower"))) == []


@pytest.mark.parametrize("metric", [None, "", "TODO"])
def test_anchor_must_name_its_metric(tmp_path, metric):
    problems = lint_domain.lint_task(_task(tmp_path, anchor=dict(_ANCHOR, metric=metric)))
    assert len(problems) == 1 and "names no `metric`" in problems[0], problems


def test_retired_direction_and_metric_metadata_is_not_flagged(tmp_path):
    """metadata.direction / metadata.metric (retired 2026-10-15) are stale, not wrong."""
    task = _task(tmp_path, extra_metadata=(
        'direction = "control"\n'
        'metric = {name = "mean_episode_reward", direction = "higher", hardware_invariant = true}\n'))
    assert lint_domain.lint_task(task) == []


def test_full_at_is_bounded_by_the_task_own_scoring_cap(tmp_path):
    """The cap comes from tests/grader_config.json (1.5 by default), read exactly as the
    kit reads it — a task its own grader accepts must never go red in PR CI over a
    hardcoded 1.5."""
    over_default = _task(tmp_path / "a", anchor=dict(_ANCHOR, full_at=1.8))
    problems = lint_domain.lint_task(over_default)
    assert len(problems) == 1 and "full_at" in problems[0] and "(0, 1.5]" in problems[0], problems

    raised = _task(tmp_path / "b", anchor=dict(_ANCHOR, full_at=1.8))
    (raised / "tests" / "grader_config.json").write_text(json.dumps({"scoring_cap": 2.0}))
    assert lint_domain.lint_task(raised) == []

    lowered = _task(tmp_path / "c", anchor=dict(_ANCHOR, full_at=1.2))
    (lowered / "tests" / "grader_config.json").write_text(json.dumps({"scoring_cap": 1.0}))
    problems = lint_domain.lint_task(lowered)
    assert len(problems) == 1 and "(0, 1]" in problems[0], problems

    # a malformed or non-positive cap falls back to the default rather than exploding
    for i, bogus_cfg in enumerate(('{"scoring_cap": "big"}', '{"scoring_cap": true}',
                                   '{"scoring_cap": -1}', "not json")):
        bogus = _task(tmp_path / f"d{i}", anchor=dict(_ANCHOR, full_at=1.5))
        (bogus / "tests" / "grader_config.json").write_text(bogus_cfg)
        assert lint_domain.lint_task(bogus) == [], bogus_cfg


# --------------------------------------------------------------------------- #
# the platform vocabulary (tasks/platforms.json, owned by this repository)
# --------------------------------------------------------------------------- #
def test_the_platform_vocabulary_is_the_fixed_list():
    assert lint_domain.platform_keys() == (
        "arm", "hand", "quadruped", "humanoid", "uav", "driving", "ground", "underwater", "space",
        "soft", "surgical", "swarm", "subsystems")


def test_every_registered_platform_is_accepted(tmp_path):
    for key in lint_domain.platform_keys():
        assert lint_domain.lint_paths([_task(tmp_path / key, platform=key)]) == [], key


@pytest.mark.parametrize("platform", ["spacecraft", "manipulation", "TODO"])
def test_an_unregistered_platform_is_refused(tmp_path, platform):
    problems = lint_domain.lint_task(_task(tmp_path, platform=platform))
    assert len(problems) == 1, problems
    assert f"metadata.platform {platform!r}" in problems[0] and "tasks/platforms.json" in problems[0]


def test_a_missing_platform_is_refused(tmp_path):
    problems = lint_domain.lint_task(_task(tmp_path, platform=""))
    assert len(problems) == 1 and "metadata.platform None" in problems[0], problems


# --------------------------------------------------------------------------- #
# battery 3: the agent-visible surface (environment/setup/payload/)
# --------------------------------------------------------------------------- #
def test_a_visible_file_reaching_the_anchor_is_still_refused(tmp_path):
    task = _task(tmp_path)
    (task / _GRADE_PY).write_text(
        "import json; a = json.load(open('../verify/anchor.json'))\n")
    problems = lint_domain.lint_task(task)
    assert any("agent-visible" in p for p in problems), problems


@pytest.mark.parametrize("code", [
    "import json; a = json.load(open('../../../tests/grader_config.json'))\n",
    "import os; print(os.listdir('/opt/ale/tests'))\n",
    "import runpy; runpy.run_path('tests/verify.py')\n",
    "import subprocess; subprocess.run(['bash', 'tests/test.sh'])\n",
    "cfg = open('tests/grader_config.json').read()\n",
])
def test_a_visible_file_reaching_the_tests_stage_is_refused(tmp_path, code):
    task = _task(tmp_path)
    (task / _GRADE_PY).write_text(code)
    problems = lint_domain.lint_task(task)
    assert problems and all("agent-visible" in p for p in problems), problems


def test_a_library_s_own_test_suite_is_not_the_tests_stage(tmp_path):
    """A payload library that names its own tests/test_*.py is not reaching the grader."""
    task = _task(tmp_path)
    helper = task / "environment" / "setup" / "payload" / "lib" / "render.py"
    helper.parent.mkdir(parents=True)
    helper.write_text('"""Render helpers (used by tests/test_render_lens.py; run its tests/ '
                      'folder with pytest)."""\n')
    assert lint_domain.lint_task(task) == []


@pytest.mark.parametrize("line", ['print({"ratio": 0.9})', 'print({"score": 0.9})',
                                  'print(json.dumps({"reward": 0.9}))'])
def test_a_practice_grader_reporting_more_than_the_metric_is_refused(tmp_path, line):
    task = _task(tmp_path)
    (task / _GRADE_PY).write_text(line + "\n")
    problems = lint_domain.lint_task(task)
    assert len(problems) == 1 and "raw metric only" in problems[0], problems


# --------------------------------------------------------------------------- #
# battery 4: image hygiene (environment/Dockerfile, tests/Dockerfile)
# --------------------------------------------------------------------------- #
def test_a_pdf_in_the_image_is_no_longer_hidden_surface(tmp_path):
    """The paper PDF was hidden for the retired faithfulness judge; with no judge there is
    nothing to hide, and the image battery must not invent a leak."""
    task = _task(tmp_path)
    (task / "environment" / "Dockerfile").write_text(
        _DOCKERFILE + "COPY docs/manual.pdf /opt/manual.pdf\n")
    assert lint_domain.lint_image(task / "environment" / "Dockerfile") == []


@pytest.mark.parametrize("line", ["COPY tests/ /opt/grader/", "COPY solution/solver.py /opt/",
                                  "COPY ref/anchor.json /opt/ref/", "COPY task.toml /opt/",
                                  "ADD verify.py /opt/", "COPY oracle/ /opt/oracle/"])
def test_an_image_baking_in_the_hidden_surface_is_refused(tmp_path, line):
    task = _task(tmp_path)
    (task / "environment" / "Dockerfile").write_text(_DOCKERFILE + line + "\n")
    problems = lint_domain.lint_image(task / "environment" / "Dockerfile")
    assert len(problems) == 1 and problems[0].startswith("probe: COPY/ADD in environment/Dockerfile")


@pytest.mark.parametrize("line, what", [
    ("RUN docker run --privileged x", "privileged mode"),
    ('ENTRYPOINT ["/bin/sh"]', "ENTRYPOINT"),
])
def test_an_image_that_escalates_or_takes_the_entrypoint_is_refused(tmp_path, line, what):
    task = _task(tmp_path)
    (task / "environment" / "Dockerfile").write_text(_DOCKERFILE + line + "\n")
    problems = lint_domain.lint_image(task / "environment" / "Dockerfile")
    assert len(problems) == 1 and what in problems[0], problems


def test_the_repository_walk_lints_both_images_of_a_task(tmp_path):
    task = _task(tmp_path)
    (task / "environment" / "Dockerfile").write_text(_DOCKERFILE + "COPY task.toml /opt/\n")
    (task / "tests" / "Dockerfile").write_text(_DOCKERFILE + "COPY anchor.json /opt/\n")
    problems = lint_domain.lint_domain(tmp_path)
    assert len(problems) == 2, problems
    assert "in environment/Dockerfile brings in the manifest" in problems[0]
    assert "in tests/Dockerfile brings in the metric anchor" in problems[1]
    assert all(p.startswith("probe: ") for p in problems)


def test_a_template_gets_the_image_battery_only(tmp_path):
    """templates/ holds unfinished tasks by design (a null anchor, a TODO platform): the walk
    lints their images, whose leaks every instantiated task would inherit, and nothing else."""
    template = _task(tmp_path, name="task", platform="TODO", collection="templates",
                     anchor=dict(_ANCHOR, value=None, metric="TODO"))
    assert lint_domain.lint_paths([tmp_path]) == []
    (template / "environment" / "Dockerfile").write_text(_DOCKERFILE + "COPY tests /opt/t\n")
    problems = lint_domain.lint_paths([tmp_path])
    assert len(problems) == 1 and "the tests stage" in problems[0], problems


# --------------------------------------------------------------------------- #
# found in the 2026-11-03 review of the Harbor migration
# --------------------------------------------------------------------------- #
def test_the_one_folder_lint_runs_the_image_battery_too(tmp_path):
    """A maintainer lints one contribution with `lint_domain.py tasks/<platform>/<slug>`; that
    must be the pull request's lint, image hygiene included."""
    task = _task(tmp_path)
    (task / "environment" / "Dockerfile").write_text(_DOCKERFILE + "COPY tests/anchor.json /opt/a.json\n")
    problems = lint_domain.lint_paths([task])
    assert any("COPY/ADD in environment/Dockerfile" in p for p in problems), problems


def test_a_task_folder_without_task_toml_is_reported(tmp_path):
    """The engine discovers tasks by task.toml alone: a misnamed manifest leaves the folder
    unseen by `ale lint`, by a run and by the export, so the domain lint names it."""
    task = _task(tmp_path)
    (task / "task.toml").rename(task / "task.yml")
    problems = lint_domain.lint_paths([tmp_path])
    assert len(problems) == 1 and "uav/probe: no task.toml" in problems[0], problems


def test_the_payload_may_not_name_the_grader_s_files(tmp_path):
    task = _task(tmp_path)
    (task / "environment" / "setup" / "payload" / "peek.py").write_text('open("tests/grader.py")\n')
    problems = lint_domain.lint_paths([task])
    assert any("the grader (tests/)" in p for p in problems), problems


def test_the_graders_copy_of_the_givens_must_match_the_payload(tmp_path):
    """A separate verifier never runs setup.sh, so a grader may keep its own copy of the givens in
    tests/setup_payload/; the lint holds it byte-identical to environment/setup/payload/."""
    task = _task(tmp_path)
    payload = task / "environment" / "setup" / "payload"
    (payload / "given").mkdir()
    (payload / "given" / "spec.json").write_text('{"a": 1}\n')
    mirror = task / "tests" / "setup_payload"
    (mirror / "given").mkdir(parents=True)
    (mirror / "given" / "spec.json").write_text('{"a": 1}\n')
    (mirror / "practice_grader").mkdir()
    (mirror / "practice_grader" / "grade.py").write_text((task / _GRADE_PY).read_text())
    assert lint_domain.lint_task(task) == []

    (mirror / "given" / "spec.json").write_text('{"a": 2}\n')
    (mirror / "given" / "extra.txt").write_text("x\n")
    problems = lint_domain.lint_task(task)
    assert any("given/spec.json differs" in p for p in problems), problems
    assert any("given/extra.txt is not in environment/setup/payload/" in p for p in problems), problems
