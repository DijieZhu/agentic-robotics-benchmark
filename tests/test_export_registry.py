"""scripts/export_registry.py: registry v2 field assembly without the engine, plus one real load.

The engine loader is the source of task identity and is exercised only in the integration
test (real ``load_tasks`` on a copy of the template). Everything else is pure functions over
a ``LoadedTask`` and a temporary task folder, which is where a wrong anchor, a wrong cap or a
lost identity would come from.
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import jsonschema
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
#: This repository's contract for the document (consumers validate against the same file).
SCHEMA = json.loads((REPO_ROOT / "schemas" / "task-registry.schema.json").read_text(encoding="utf-8"))
_SPEC = importlib.util.spec_from_file_location(
    "export_registry", REPO_ROOT / "scripts" / "export_registry.py"
)
assert _SPEC is not None and _SPEC.loader is not None
xr = importlib.util.module_from_spec(_SPEC)
# dataclasses resolve string annotations through sys.modules[cls.__module__]
sys.modules[_SPEC.name] = xr
_SPEC.loader.exec_module(xr)

NOW = "2026-09-02T00:00:00Z"
HASH_A = "sha256:" + "a" * 64
HASH_B = "sha256:" + "b" * 64
HASH_C = "sha256:" + "c" * 64
#: The org part of every task id ([task].name = "<org>/<slug>").
ORG = "grounded-intelligence-initiative"
#: What every anchor now declares: the metric and its direction live in tests/anchor.json.
_METRIC = {"metric": "mean_episode_reward", "direction": "higher"}


def _task_dir(tmp_path: Path, *, anchor: dict | None = None, grader_config: dict | None = None,
              instruction: str = "prose only\n\n## Goal\n") -> Path:
    root = tmp_path / "some_task"
    (root / "tests").mkdir(parents=True)
    (root / "instruction.md").write_text(instruction, encoding="utf-8")
    if anchor is not None:
        (root / "tests" / "anchor.json").write_text(json.dumps(anchor), encoding="utf-8")
    if grader_config is not None:
        (root / "tests" / "grader_config.json").write_text(
            json.dumps(grader_config), encoding="utf-8"
        )
    return root


def _loaded(root: Path, *, slug: str = "some_task", task_id: str | None = None,
            spec_hash: str = HASH_A, content_digest: str = HASH_B,
            metadata: dict | None = None) -> xr.LoadedTask:
    """What ``from_manifest_task`` makes of an engine task (setup = the engine's fixed bound)."""
    return xr.LoadedTask(
        task_id=f"{ORG}/{slug}" if task_id is None else task_id, root=root,
        spec_hash=spec_hash, content_digest=content_digest,
        metadata={"platform": "uav"} if metadata is None else metadata,
        resources={"cpus": 4, "memory_mb": 8192, "storage_mb": None, "gpus": 0, "sudo": False},
        timeouts={"setup": 1800.0, "agent": 10800.0, "verify": 1800.0},
    )


# --------------------------------------------------------------------------- #
# title
# --------------------------------------------------------------------------- #
def test_title_is_first_level_one_heading(tmp_path):
    root = _task_dir(tmp_path, instruction="intro\n\n## Goal\n# Real Title  \n# Second\n")
    assert xr.read_title(root, "some_task") == "Real Title"


def test_title_falls_back_to_title_cased_slug(tmp_path):
    root = _task_dir(tmp_path)
    assert xr.read_title(root, "drone_hover") == "Drone Hover"
    assert xr.read_title(tmp_path / "missing", "rrt-connect") == "Rrt Connect"


def test_title_ignores_comments_inside_fenced_code_blocks(tmp_path):
    root = _task_dir(tmp_path, instruction=(
        "intro\n\n```bash\n# install deps\npip install x\n```\n\n"
        "~~~\n# also not a title\n~~~\n\n## Goal\n"))
    assert xr.read_title(root, "some_task") == "Some Task"
    # a real h1 after the fence still wins
    (root / "instruction.md").write_text(
        "```\n# comment\n```\n# Real Title\n", encoding="utf-8")
    assert xr.read_title(root, "some_task") == "Real Title"


def test_title_needs_horizontal_space_after_the_hash(tmp_path):
    """``#`` followed by a newline is not a heading; ``\\s+`` used to swallow the newline."""
    root = _task_dir(tmp_path, instruction="#\nnot a title\n\n## Goal\n")
    assert xr.read_title(root, "some_task") == "Some Task"
    (root / "instruction.md").write_text("#\tTabbed Title\n", encoding="utf-8")
    assert xr.read_title(root, "some_task") == "Tabbed Title"


# --------------------------------------------------------------------------- #
# anchor + cap
# --------------------------------------------------------------------------- #
def test_anchor_fields_and_full_at_default(tmp_path):
    root = _task_dir(tmp_path, anchor={"value": 472.07, "source": "oracle real run"})
    fields = xr.read_anchor(root, 1.5)
    assert fields == {
        "anchor": 472.07, "anchor_full_at": 1.0, "anchor_source": "oracle real run",
        "anchor_image_identity": None,
    }


def test_anchor_reads_full_at_and_image_identity(tmp_path):
    root = _task_dir(tmp_path, anchor={"value": 1, "full_at": 0.825, "image_identity": "sha256:x"})
    fields = xr.read_anchor(root, 1.5)
    assert fields["anchor_full_at"] == 0.825
    assert fields["anchor_image_identity"] == "sha256:x"
    assert fields["anchor"] == 1.0


def test_null_anchor_is_the_authoring_state_not_an_error(tmp_path):
    root = _task_dir(tmp_path, anchor={"value": None, "full_at": 0.8})
    assert xr.read_anchor(root, 1.5)["anchor"] is None


@pytest.mark.parametrize("full_at", [0, -0.1, 1.6, "0.8", True])
def test_full_at_outside_zero_cap_is_rejected(tmp_path, full_at):
    root = _task_dir(tmp_path, anchor={"value": 1.0, "full_at": full_at})
    with pytest.raises(xr.RegistryError, match="full_at"):
        xr.read_anchor(root, 1.5)


def test_full_at_equal_to_cap_is_allowed(tmp_path):
    root = _task_dir(tmp_path, anchor={"value": 1.0, "full_at": 1.5})
    assert xr.read_anchor(root, 1.5)["anchor_full_at"] == 1.5


def test_missing_anchor_file_is_refused(tmp_path):
    with pytest.raises(xr.RegistryError, match="anchor.json"):
        xr.read_anchor(_task_dir(tmp_path), 1.5)


@pytest.mark.parametrize("value", [0, -3.0, -0.001])
def test_non_positive_anchor_is_refused_like_the_kit_does(tmp_path, value):
    root = _task_dir(tmp_path, anchor={"value": value, "full_at": 0.8})
    with pytest.raises(xr.RegistryError, match="must be > 0"):
        xr.read_anchor(root, 1.5)


def test_task_entry_refuses_an_unregistered_or_missing_platform(tmp_path):
    root = _task_dir(tmp_path, anchor=dict(_METRIC, value=1.0))
    with pytest.raises(xr.RegistryError, match="tasks/platforms.json"):
        xr.task_entry(_loaded(root, metadata={"platform": "spacecraft"}), [])
    with pytest.raises(xr.RegistryError, match="not one of"):
        xr.task_entry(_loaded(root, metadata={}), [])


@pytest.mark.parametrize("task_id", [
    "some_task",                                    # the old bare slug: no org
    f"{ORG.upper()}/some_task",                     # uppercase
    f"{ORG}/some/task",                             # two slashes
    f"{ORG}/",                                      # no name
])
def test_task_entry_refuses_a_task_id_that_is_not_org_name(tmp_path, task_id):
    """The engine's TaskId shape; the schema's `task_id` pattern is the same expression."""
    root = _task_dir(tmp_path, anchor=dict(_METRIC, value=1.0))
    with pytest.raises(xr.RegistryError, match="org/name"):
        xr.task_entry(_loaded(root, task_id=task_id), [])


def test_score_cap_default_and_override(tmp_path):
    assert xr.read_score_cap(_task_dir(tmp_path)) == 1.5
    assert xr.read_score_cap(_task_dir(tmp_path / "b", grader_config={"episodes": 3})) == 1.5
    assert xr.read_score_cap(_task_dir(tmp_path / "c", grader_config={"scoring_cap": 2})) == 2.0
    with pytest.raises(xr.RegistryError, match="scoring_cap"):
        xr.read_score_cap(_task_dir(tmp_path / "d", grader_config={"scoring_cap": 0}))


# --------------------------------------------------------------------------- #
# metric + whole entry
# --------------------------------------------------------------------------- #
def test_metric_fields_come_from_the_anchor(tmp_path):
    root = _task_dir(tmp_path, anchor={"metric": "mean_goal_distance", "direction": "lower",
                                       "value": None})
    assert xr.metric_fields(root, "t") == {
        "metric_name": "mean_goal_distance", "metric_label": "Mean goal distance",
        "metric_direction": "lower_is_better",
    }
    cases = [({"direction": "lower"}, "`metric`"), ({"metric": "TODO", "direction": "lower"}, "`metric`"),
             ({"metric": "x", "direction": "sideways"}, "`direction`"), ({"metric": "x"}, "`direction`")]
    for index, (anchor, match) in enumerate(cases):
        with pytest.raises(xr.RegistryError, match=match):
            xr.metric_fields(_task_dir(tmp_path / f"case{index}", anchor=anchor), "t")
    with pytest.raises(xr.RegistryError, match="missing tests/anchor.json"):
        xr.metric_fields(_task_dir(tmp_path / "none"), "t")


def test_task_entry_assembles_every_v2_field(tmp_path):
    root = _task_dir(tmp_path, anchor=dict(_METRIC, value=472.07, full_at=0.825, source="oracle"),
                     grader_config={"scoring_cap": 1.5})
    identities = [{"spec_hash": HASH_A, "content_digest": HASH_B, "since": NOW}]
    entry = xr.task_entry(_loaded(root), identities)
    assert entry == {
        "task_id": f"{ORG}/some_task", "slug": "some_task", "title": "Some Task",
        "platform": "uav",
        "metric_name": "mean_episode_reward", "metric_label": "Mean episode reward",
        "metric_direction": "higher_is_better",
        "identities": identities,
        "anchor": 472.07, "anchor_full_at": 0.825, "anchor_source": "oracle",
        "anchor_image_identity": None, "score_cap": 1.5,
        "resources": {"cpus": 4, "memory_mb": 8192, "gpus": 0},
        "timeouts": {"setup": 1800, "agent": 10800, "verify": 1800},
        "status": "active",
    }
    for key in ("genre", "mode", "paper_title", "paper_arxiv", "direction", "hardware_invariant"):
        assert key not in entry


# --------------------------------------------------------------------------- #
# identity history
# --------------------------------------------------------------------------- #
def test_history_starts_with_the_current_identity(tmp_path):
    task = _loaded(tmp_path)
    history, identities = xr.merge_history([], task, NOW)
    assert identities == [{"spec_hash": HASH_A, "content_digest": HASH_B, "since": NOW}]
    assert history == [{"slug": "some_task", "spec_hash": HASH_A,
                        "content_digest": HASH_B, "since": NOW}]


def test_history_unchanged_when_hashes_match(tmp_path):
    previous = [{"slug": "some_task", "spec_hash": HASH_A, "content_digest": HASH_B,
                 "since": "2026-01-01T00:00:00Z"}]
    history, identities = xr.merge_history(list(previous), _loaded(tmp_path), NOW)
    assert history == previous
    assert identities[0]["since"] == "2026-01-01T00:00:00Z"


@pytest.mark.parametrize("spec_hash,content_digest", [(HASH_C, HASH_B), (HASH_A, HASH_C)])
def test_history_prepends_when_either_hash_changes(tmp_path, spec_hash, content_digest):
    previous = [{"slug": "some_task", "spec_hash": HASH_A, "content_digest": HASH_B,
                 "since": "2026-01-01T00:00:00Z"},
                {"slug": "other", "spec_hash": HASH_C, "content_digest": HASH_C, "since": "x"}]
    task = _loaded(tmp_path, spec_hash=spec_hash, content_digest=content_digest)
    history, identities = xr.merge_history(previous, task, NOW)
    assert [i["since"] for i in identities] == [NOW, "2026-01-01T00:00:00Z"]
    assert identities[0] == {"spec_hash": spec_hash, "content_digest": content_digest, "since": NOW}
    assert identities[1] == {"spec_hash": HASH_A, "content_digest": HASH_B,
                             "since": "2026-01-01T00:00:00Z"}
    # other tasks' records survive untouched; the file stays a flat list
    assert {"slug": "other", "spec_hash": HASH_C, "content_digest": HASH_C, "since": "x"} in history
    assert len(history) == 3


def test_load_history_accepts_missing_file_and_rejects_non_list(tmp_path):
    assert xr.load_history(tmp_path / "none.json") == []
    bad = tmp_path / "bad.json"
    bad.write_text("{}", encoding="utf-8")
    with pytest.raises(xr.RegistryError, match="list"):
        xr.load_history(bad)


def test_load_history_rejects_corrupt_and_malformed_records(tmp_path):
    corrupt = tmp_path / "corrupt.json"
    corrupt.write_text('[{"slug": "x",]', encoding="utf-8")
    with pytest.raises(xr.RegistryError, match="invalid JSON"):
        xr.load_history(corrupt)
    non_dict = tmp_path / "non_dict.json"
    non_dict.write_text('["drone_hover"]', encoding="utf-8")
    with pytest.raises(xr.RegistryError, match="record 0 is not an object"):
        xr.load_history(non_dict)
    missing = tmp_path / "missing.json"
    missing.write_text(json.dumps([
        {"slug": "ok", "spec_hash": HASH_A, "content_digest": HASH_B, "since": NOW},
        {"slug": "x", "spec_hash": HASH_A, "since": NOW},
    ]), encoding="utf-8")
    with pytest.raises(xr.RegistryError, match="record 1 .*content_digest"):
        xr.load_history(missing)
    good = tmp_path / "good.json"
    good.write_text(json.dumps([
        {"slug": "ok", "spec_hash": HASH_A, "content_digest": HASH_B, "since": NOW}]),
        encoding="utf-8")
    assert len(xr.load_history(good)) == 1


def test_history_revert_dedupes_the_pair_and_keeps_current_first(tmp_path):
    """A -> B -> A: the A pair must appear once, at index 0, with a fresh ``since``."""
    previous = [
        {"slug": "some_task", "spec_hash": HASH_C, "content_digest": HASH_C, "since": "b"},
        {"slug": "some_task", "spec_hash": HASH_A, "content_digest": HASH_B, "since": "a"},
        {"slug": "other", "spec_hash": HASH_C, "content_digest": HASH_C, "since": "x"},
    ]
    history, identities = xr.merge_history(previous, _loaded(tmp_path), NOW)
    assert identities == [
        {"spec_hash": HASH_A, "content_digest": HASH_B, "since": NOW},
        {"spec_hash": HASH_C, "content_digest": HASH_C, "since": "b"},
    ]
    own = [r for r in history if r["slug"] == "some_task"]
    pairs = [(r["spec_hash"], r["content_digest"]) for r in own]
    assert len(pairs) == len(set(pairs)) == 2
    assert {"slug": "other", "spec_hash": HASH_C, "content_digest": HASH_C, "since": "x"} in history


# --------------------------------------------------------------------------- #
# platforms, release, dirty tree
# --------------------------------------------------------------------------- #
def test_platforms_come_from_this_repository(tmp_path):
    platforms = xr.load_platforms()
    assert [entry["key"] for entry in platforms] == [
        "arm", "hand", "quadruped", "humanoid", "uav", "driving", "ground", "underwater", "space",
        "soft", "surgical", "swarm", "subsystems"]
    assert all(set(entry) == {"key", "label"} and entry["label"] for entry in platforms)
    bad = tmp_path / "platforms.json"
    for payload, match in (([], "non-empty"), ([{"key": "Arm", "label": "x"}], "slug"),
                           ([{"key": "arm", "label": "a"}, {"key": "arm", "label": "b"}], "unique"),
                           ([{"key": "arm", "label": ""}], "label")):
        bad.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(xr.RegistryError, match=match):
            xr.load_platforms(bad)


@pytest.mark.parametrize("field, value", [("source", 474), ("source", ["a"]),
                                          ("image_identity", {"ref": "x"})])
def test_anchor_text_fields_must_be_strings_or_null(tmp_path, field, value):
    root = _task_dir(tmp_path, anchor=dict(_METRIC, value=1.0, **{field: value}))
    with pytest.raises(xr.RegistryError, match=field):
        xr.read_anchor(root, 1.5)


@pytest.mark.parametrize("key, value", [
    ("since", "2026-09-02"),                       # a date alone is not the schema's timestamp
    ("spec_hash", "sha256:" + "A" * 64),           # uppercase hex
    ("content_digest", "sha256:abc"),              # too short
])
def test_load_history_holds_records_to_the_schema_patterns(tmp_path, key, value):
    record = {"slug": "some_task", "spec_hash": HASH_A, "content_digest": HASH_B, "since": NOW}
    record[key] = value
    path = tmp_path / "h.json"
    path.write_text(json.dumps([record]), encoding="utf-8")
    with pytest.raises(xr.RegistryError, match=key):
        xr.load_history(path)


def test_release_block_needs_a_name_and_version():
    for name, version in (("", "v0.5"), ("Agentic Robotics Benchmark", "")):
        with pytest.raises(xr.RegistryError, match="non-empty"):
            xr.release_block(name, version, ["some_task"])


def test_release_block_policy():
    block = xr.release_block("Agentic Robotics Benchmark", "v0.5", ["drone_hover"])
    assert block == {
        "name": "Agentic Robotics Benchmark", "version": "v0.5", "status": "active",
        "aggregation_policy": {"type": "macro_mean", "task_set": ["drone_hover"],
                               "score_cap": 1.5, "missing_task_policy": "zero"},
    }


def _git_repo_with_task(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    env = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@x",
           "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@x"}
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True, env=env)
    root = _task_dir(repo, anchor=dict(_METRIC, value=1.0))
    (repo / "platforms.json").write_text(xr.PLATFORMS_FILE.read_text(encoding="utf-8"), encoding="utf-8")
    (repo / ".gitignore").write_text("__pycache__/\n", encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True, env=env)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=repo, check=True, env=env)
    return root


def test_clean_tree_passes_and_dirty_or_ignored_files_refuse(tmp_path):
    root = _git_repo_with_task(tmp_path)
    xr.require_clean(root)
    (root / "tests" / "__pycache__").mkdir()
    (root / "tests" / "__pycache__" / "x.pyc").write_bytes(b"\x00")
    with pytest.raises(xr.RegistryError, match="not clean"):
        xr.require_clean(root)


def test_build_registry_refuses_dirty_unless_allowed(tmp_path):
    root = _git_repo_with_task(tmp_path)
    (root / "instruction.md").write_text("# edited\n", encoding="utf-8")
    task = _loaded(root)
    kwargs = {"history": [], "now": NOW, "source_commit_sha": "abc",
              "engine": {"commits": ["e"]}, "platforms_file": root.parent / "platforms.json"}
    with pytest.raises(xr.RegistryError, match="not clean"):
        xr.build_registry([task], **kwargs)
    registry, history = xr.build_registry([task], allow_dirty=True, **kwargs)
    assert registry["schema_version"] == "ale-robotics-task-registry/v2"
    # the taxonomy is this repository's: platforms from tasks/platforms.json, nothing else
    assert registry["platforms"] == xr.load_platforms()
    assert "directions" not in registry and "category_tree" not in registry
    assert registry["provenance"]["source"] == "engine_task_loader"
    # a throwaway export can never pass for a release one
    assert registry["provenance"]["source_commit"] == "abc-dirty"
    assert registry["provenance"]["dirty"] is True
    assert registry["release"]["aggregation_policy"]["task_set"] == ["some_task"]
    assert registry["tasks"][0]["title"] == "edited"
    assert history[0]["slug"] == "some_task"


def test_clean_export_provenance_carries_no_dirty_marker(tmp_path):
    root = _git_repo_with_task(tmp_path)
    registry, _ = xr.build_registry(
        [_loaded(root)], history=[], now=NOW, source_commit_sha="abc",
        engine={"commits": ["e"]}, platforms_file=root.parent / "platforms.json")
    assert registry["provenance"]["source_commit"] == "abc"
    assert "dirty" not in registry["provenance"]


def test_an_uncommitted_platform_vocabulary_is_refused(tmp_path):
    """The vocabulary goes into the document as `platforms`, so it obeys the clean-tree rule."""
    root = _git_repo_with_task(tmp_path)
    vocabulary = root.parent / "platforms.json"
    vocabulary.write_text(vocabulary.read_text(encoding="utf-8").replace("Robotic arm", "Arm"),
                          encoding="utf-8")
    kwargs = {"history": [], "now": NOW, "source_commit_sha": "abc", "engine": {},
              "platforms_file": vocabulary}
    with pytest.raises(xr.RegistryError, match="not clean"):
        xr.build_registry([_loaded(root)], **kwargs)
    registry, _ = xr.build_registry([_loaded(root)], allow_dirty=True, **kwargs)
    assert registry["platforms"][0] == {"key": "arm", "label": "Arm"}


def _full_build(root: Path, **overrides) -> dict:
    kwargs = {"history": [], "now": NOW, "source_commit_sha": "0123abcd" * 5,
              "engine": {"commits": ["1cdbc02" * 5 + "fffff"], "version": "0.1.1",
                         "ale_verify_content_hash": HASH_C},
              "platforms_file": root.parent / "platforms.json"}
    kwargs.update(overrides)
    registry, _ = xr.build_registry([_loaded(root)], **kwargs)
    return registry


def test_the_export_matches_the_repository_schema(tmp_path):
    """schemas/task-registry.schema.json is the contract; the export must satisfy it as built."""
    root = _git_repo_with_task(tmp_path)
    jsonschema.Draft7Validator.check_schema(SCHEMA)
    jsonschema.validate(_full_build(root), SCHEMA)
    jsonschema.validate(_full_build(root, allow_dirty=True), SCHEMA)     # dirty: still well-formed


def test_the_schema_refuses_what_the_export_retired(tmp_path):
    root = _git_repo_with_task(tmp_path)
    registry = _full_build(root)
    for field, value in (("direction", "control"), ("hardware_invariant", True)):
        stale = json.loads(json.dumps(registry))
        stale["tasks"][0][field] = value
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(stale, SCHEMA)
    stale = json.loads(json.dumps(registry))
    stale["directions"] = []
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(stale, SCHEMA)


def test_the_schema_requires_the_engine_task_id(tmp_path):
    """`task_id` is required and holds the engine's org/name, not the bare slug."""
    root = _git_repo_with_task(tmp_path)
    registry = _full_build(root)
    assert registry["tasks"][0]["task_id"] == f"{ORG}/some_task"
    missing = json.loads(json.dumps(registry))
    del missing["tasks"][0]["task_id"]
    with pytest.raises(jsonschema.ValidationError, match="task_id"):
        jsonschema.validate(missing, SCHEMA)
    for value in ("some_task", f"{ORG}/Some_Task", f"{ORG}/some/task"):
        bare = json.loads(json.dumps(registry))
        bare["tasks"][0]["task_id"] = value
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(bare, SCHEMA)
    # the export's own check and the schema's pattern are one expression
    assert SCHEMA["definitions"]["task"]["properties"]["task_id"]["pattern"] == xr._TASK_ID.pattern


@pytest.mark.parametrize("url, slug", [
    ("https://github.com/Grounded-Intelligence-Initiative/agentic-robotics-benchmark.git",
     "Grounded-Intelligence-Initiative/agentic-robotics-benchmark"),
    ("https://github.com/Owner/repo", "Owner/repo"),
    ("git@github.com:Owner/repo.git", "Owner/repo"),
])
def test_tasks_repo_is_the_origin_slug(tmp_path, url, slug):
    repo = tmp_path / "any-folder-name"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "remote", "add", "origin", url], check=True)
    assert xr.tasks_repo(repo) == slug


def test_tasks_repo_without_a_remote_is_the_folder_name(tmp_path):
    repo = tmp_path / "plain"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    assert xr.tasks_repo(repo) == "plain"


def test_allow_dirty_never_writes_the_canonical_history(tmp_path):
    assert xr.history_writable(xr.DEFAULT_HISTORY, allow_dirty=False)
    assert not xr.history_writable(xr.DEFAULT_HISTORY, allow_dirty=True)
    # a relative spelling of the same file is still the canonical file
    relative = Path(os.path.relpath(xr.DEFAULT_HISTORY, Path.cwd()))
    assert not xr.history_writable(relative, allow_dirty=True)
    assert xr.history_writable(tmp_path / "throwaway.json", allow_dirty=True)


def test_parse_args_history_defaults_to_none_so_explicitness_is_known():
    assert xr.parse_args(["--out", "r.json"]).history is None
    assert xr.parse_args(["--out", "r.json", "--history", "h.json"]).history == Path("h.json")


def test_build_registry_registers_every_task_in_slug_order(tmp_path):
    """The engine has no variants any more: every loaded task is registered, by slug."""
    zeta = _loaded(_task_dir(tmp_path / "z", anchor=dict(_METRIC, value=1.0)), slug="zeta",
                   spec_hash=HASH_C)
    alpha = _loaded(_task_dir(tmp_path / "a", anchor=dict(_METRIC, value=1.0)), slug="alpha")
    registry, history = xr.build_registry(
        [zeta, alpha], history=[], now=NOW, source_commit_sha="abc", engine={},
        allow_dirty=True,
    )
    assert [t["slug"] for t in registry["tasks"]] == ["alpha", "zeta"]
    assert [t["task_id"] for t in registry["tasks"]] == [f"{ORG}/alpha", f"{ORG}/zeta"]
    assert registry["release"]["aggregation_policy"]["task_set"] == ["alpha", "zeta"]
    assert registry["tasks"][1]["identities"][0]["spec_hash"] == HASH_C
    # the identity history stays keyed by slug, in the format it always had
    assert sorted(record["slug"] for record in history) == ["alpha", "zeta"]
    assert all(set(record) == {"slug", "spec_hash", "content_digest", "since"}
               for record in history)
    with pytest.raises(xr.RegistryError, match="no task"):
        xr.build_registry([], history=[], now=NOW, source_commit_sha="abc", engine={},
                          allow_dirty=True)


def test_build_registry_refuses_one_slug_under_two_task_ids(tmp_path):
    """`a/x` and `b/x` are two ids to the engine but one key to the registry and its history."""
    ours = _loaded(_task_dir(tmp_path / "a", anchor=dict(_METRIC, value=1.0)))
    theirs = _loaded(_task_dir(tmp_path / "b", anchor=dict(_METRIC, value=1.0)),
                     task_id="someone-else/some_task")
    with pytest.raises(xr.RegistryError, match="'some_task' is claimed by both"):
        xr.build_registry([ours, theirs], history=[], now=NOW, source_commit_sha="abc",
                          engine={}, allow_dirty=True)


def test_dump_json_is_stable(tmp_path):
    out = tmp_path / "nested" / "r.json"
    xr.dump_json({"b": 1, "a": [2]}, out)
    assert out.read_text(encoding="utf-8") == '{\n  "a": [\n    2\n  ],\n  "b": 1\n}\n'


# --------------------------------------------------------------------------- #
# integration: the real loader over a copy of the template (engine venv required)
def _register_platform(task_copy: Path) -> None:
    """The template ships `[metadata] platform = "TODO"` and an anchor naming no metric
    (`TODO`); an exported task needs a registered platform and the metric its grader reports."""
    manifest = task_copy / "task.toml"
    text, count = re.subn(r'(?m)^(\s*platform\s*=\s*)"TODO"', r'\1"arm"',
                          manifest.read_text(encoding="utf-8"))
    assert count == 1, 'the template\'s task.toml no longer ships platform = "TODO"'
    manifest.write_text(text, encoding="utf-8")
    anchor_path = task_copy / "tests" / "anchor.json"
    anchor = json.loads(anchor_path.read_text(encoding="utf-8"))
    anchor["metric"] = "mean_goal_distance"          # what the template's grader reports
    anchor_path.write_text(json.dumps(anchor, indent=2) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- #
ENGINE_PYTHON = REPO_ROOT / "vendor" / "ale" / ".venv" / "bin" / "python"
TEMPLATE = REPO_ROOT / "templates" / "task"


@pytest.mark.skipif(
    not ENGINE_PYTHON.is_file(),
    reason="engine venv missing: run `uv sync` in vendor/ale to enable the real-loader test",
)
def test_real_loader_on_template_copy(tmp_path):
    """Drive the script's own ``main`` inside the engine interpreter on a template copy.

    The copy is not a git checkout, so ``--allow-dirty`` is required; what this proves is
    that ``from_manifest_task`` reads the right attribute names off the engine objects and
    that the end-to-end document satisfies this repository's schema.
    """
    task_copy = tmp_path / "tasks" / "template_reach"
    shutil.copytree(TEMPLATE, task_copy,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    _register_platform(task_copy)
    out = tmp_path / "registry.json"
    history = tmp_path / "history.json"
    result = subprocess.run(
        [str(ENGINE_PYTHON), str(REPO_ROOT / "scripts" / "export_registry.py"),
         "--tasks", str(tmp_path / "tasks"), "--history", str(history),
         "--out", str(out), "--allow-dirty"],
        capture_output=True, text=True, check=False,
        env={**os.environ, "VIRTUAL_ENV": ""},
    )
    assert result.returncode == 0, result.stderr
    registry = json.loads(out.read_text(encoding="utf-8"))
    assert registry["schema_version"] == "ale-robotics-task-registry/v2"
    jsonschema.validate(registry, SCHEMA)
    (task,) = registry["tasks"]
    # the engine's id is [task].name; the slug is its part after the slash
    assert task["task_id"] == f"{ORG}/template_reach"
    assert task["slug"] == "template_reach"
    # the metric is the one tests/anchor.json declares (the template grades a cost)
    assert task["metric_name"] == "mean_goal_distance"
    assert task["metric_direction"] == "lower_is_better"
    assert task["platform"] == "arm" and "direction" not in task
    assert [entry["key"] for entry in registry["platforms"]][:2] == ["arm", "hand"]
    assert task["anchor"] is None and task["anchor_full_at"] == 0.8
    assert registry["provenance"]["dirty"] is True
    assert registry["provenance"]["source_commit"].endswith("-dirty")
    assert task["resources"] == {"cpus": 2, "memory_mb": 4096, "gpus": 0}
    # agent / verify: [agent].timeout_sec, [verifier].timeout_sec; setup: the engine's fixed bound
    assert task["timeouts"] == {"setup": 1800, "agent": 3600, "verify": 900}
    identity = task["identities"][0]
    assert identity["spec_hash"].startswith("sha256:") and len(identity["spec_hash"]) == 71
    assert identity["content_digest"].startswith("sha256:")
    # exactly the pair the engine writes into an episode's lock.json (task.spec_hash,
    # task.content_digest), read here straight off the engine's loader
    program = ("import json, sys; from pathlib import Path; "
               "from ale.run.tasksets.manifest import load_tasks; "
               "(t,) = load_tasks(Path(sys.argv[1])); "
               "print(json.dumps([str(t.spec.name), t.spec.spec_hash, t.task_digest]))")
    engine_view = subprocess.run(
        [str(ENGINE_PYTHON), "-c", program, str(tmp_path / "tasks")],
        capture_output=True, text=True, check=True, env={**os.environ, "VIRTUAL_ENV": ""},
    )
    assert json.loads(engine_view.stdout) == [
        task["task_id"], identity["spec_hash"], identity["content_digest"]]
    assert registry["provenance"]["engine"]["ale_verify_content_hash"].startswith("sha256:")
    assert registry["provenance"]["engine"]["version"] == "0.1.1"
    saved = json.loads(history.read_text(encoding="utf-8"))
    assert saved[0]["slug"] == "template_reach"
    assert saved[0]["spec_hash"] == identity["spec_hash"]
    # a second export with an unchanged tree leaves the history alone
    second = subprocess.run(
        [str(ENGINE_PYTHON), str(REPO_ROOT / "scripts" / "export_registry.py"),
         "--tasks", str(tmp_path / "tasks"), "--history", str(history),
         "--out", str(out), "--allow-dirty"],
        capture_output=True, text=True, check=False, env={**os.environ, "VIRTUAL_ENV": ""},
    )
    assert second.returncode == 0, second.stderr
    assert json.loads(history.read_text(encoding="utf-8")) == saved


@pytest.mark.skipif(
    not ENGINE_PYTHON.is_file() or shutil.which("uv") is None,
    reason="engine venv or `uv` missing: the re-exec path needs both",
)
def test_reexec_from_the_repo_venv(tmp_path):
    """Run under this repo's interpreter (no ``ale``): the script must re-exec into the engine."""
    task_copy = tmp_path / "tasks" / "template_reach"
    shutil.copytree(TEMPLATE, task_copy, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    _register_platform(task_copy)
    out = tmp_path / "registry.json"
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "export_registry.py"),
         "--tasks", str(tmp_path / "tasks"), "--history", str(tmp_path / "h.json"),
         "--out", str(out), "--allow-dirty"],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(out.read_text(encoding="utf-8"))["tasks"][0]["slug"] == "template_reach"


@pytest.mark.skipif(
    not ENGINE_PYTHON.is_file(),
    reason="engine venv missing: run `uv sync` in vendor/ale to enable the real-loader test",
)
def test_cli_corrupt_history_is_an_error_line_not_a_traceback(tmp_path):
    task_copy = tmp_path / "tasks" / "template_reach"
    shutil.copytree(TEMPLATE, task_copy, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    history = tmp_path / "history.json"
    history.write_text('[{"slug": "template_reach",]', encoding="utf-8")
    result = subprocess.run(
        [str(ENGINE_PYTHON), str(REPO_ROOT / "scripts" / "export_registry.py"),
         "--tasks", str(tmp_path / "tasks"), "--history", str(history),
         "--out", str(tmp_path / "r.json"), "--allow-dirty"],
        capture_output=True, text=True, check=False, env={**os.environ, "VIRTUAL_ENV": ""},
    )
    assert result.returncode == 1
    assert "error:" in result.stderr and "invalid JSON" in result.stderr
    assert "Traceback" not in result.stderr
