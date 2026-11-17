"""scripts/gpu_types.py: the gpu_types every GPU task names."""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import gpu_types as gt


def _accepts(names: tuple[str, ...], model: str) -> bool:
    """A stand-in for the engine's rule, for the script's own logic: a name's words appear in the model, in order.
    CI runs the check itself with the engine's `accepts_model`."""
    words = re.findall(r"[a-z0-9]+", model.lower())
    for name in names:
        rest = iter(words)
        wanted = re.findall(r"[a-z0-9]+", name.lower())
        if wanted and all(w in rest for w in wanted):
            return True
    return not names


def _task(root: Path, slug: str, env: str = "", verifier: str = "") -> None:
    folder = root / "tasks" / "arm" / slug
    folder.mkdir(parents=True)
    (folder / "task.toml").write_text(
        f'[metadata]\nplatform = "arm"\n\n[environment]\ncpus = 4\n{env}\n\n[verifier]\ntimeout_sec = 60\n\n'
        f"[verifier.environment]\ncpus = 2\n{verifier}\n", encoding="utf-8")


def test_the_repository_gpu_tasks_name_known_models():
    errors, _ = gt.check(ROOT, _accepts)
    assert errors == []
    assert any(rec["gpu_types"] for rec in gt.gpu_tables(ROOT))


def test_good_names_pass_and_a_gpu_table_without_the_key_is_a_notice(tmp_path):
    _task(tmp_path, "cpu_only")
    _task(tmp_path, "one", 'gpus = 1\ngpu_types = ["RTX-4090", "RTX-PRO-6000", "H100"]', "gpus = 0")
    _task(tmp_path, "any", "gpus = 1", "gpus = 1")
    errors, notices = gt.check(tmp_path, _accepts)
    assert errors == []
    assert len(notices) == 2 and all("tasks/arm/any/" in n and "every GPU model is accepted" in n for n in notices)
    assert [(r["task"], r["table"]) for r in gt.gpu_tables(tmp_path)] == [
        ("arm/any", "environment"), ("arm/any", "verifier.environment"), ("arm/one", "environment")]


def test_a_mistyped_name_a_repeated_name_and_an_empty_list_are_errors(tmp_path):
    _task(tmp_path, "typo", 'gpus = 1\ngpu_types = ["RTX4090"]')
    _task(tmp_path, "twice", 'gpus = 1\ngpu_types = ["H100", "h100"]')
    _task(tmp_path, "empty", "gpus = 1\ngpu_types = []")
    errors, _ = gt.check(tmp_path, _accepts)
    assert any("tasks/arm/typo/" in e and "'RTX4090' accepts no known GPU model" in e for e in errors), errors
    assert any("tasks/arm/twice/" in e and "twice" in e for e in errors), errors
    assert any("tasks/arm/empty/" in e and "non-empty list" in e for e in errors), errors


def test_list_prints_every_gpu_table(tmp_path, capsys):
    _task(tmp_path, "one", 'gpus = 1\ngpu_types = ["RTX-4090"]', "gpus = 1")
    assert gt.main(["--root", str(tmp_path), "list"]) == 0
    out = capsys.readouterr().out
    assert "arm/one" in out and "RTX-4090" in out and "every model" in out
