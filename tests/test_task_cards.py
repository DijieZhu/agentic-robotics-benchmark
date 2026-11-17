"""scripts/task_cards.py: the task cards (tasks/cards.toml) against the task tree."""
from __future__ import annotations

import json
import shutil
import sys
import tomllib
from pathlib import Path

import jsonschema
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import task_cards as tc


def _tree(tmp_path: Path, tasks: dict[str, str], cards: str) -> Path:
    (tmp_path / "tasks").mkdir()
    shutil.copy(ROOT / "tasks" / "platforms.json", tmp_path / "tasks" / "platforms.json")
    for slug, platform in tasks.items():
        folder = tmp_path / "tasks" / platform / slug
        folder.mkdir(parents=True)
        (folder / "task.toml").write_text(f'[metadata]\nplatform = "{platform}"\n', encoding="utf-8")
        (folder / "instruction.md").write_text(f"# The {slug} brief\n\nWhat the agent does.\n\n## What you produce\n\nA file.\n",
                                               encoding="utf-8")
    (tmp_path / "tasks" / "cards.toml").write_text(cards, encoding="utf-8")
    return tmp_path


GOOD = '''
[box]
title = "Open a box"
summary = "A Panda opens a box."
platforms = ["arm"]
stages = ["behavior"]
'''


def test_the_repository_has_a_card_for_every_task():
    errors, warnings = tc.check(ROOT, strict=True)
    assert errors == [] and warnings == []


def test_the_cards_match_the_schema():
    with open(ROOT / "tasks" / "cards.toml", "rb") as fh:
        cards = tomllib.load(fh)
    schema = json.loads((ROOT / "schemas" / "task-cards.schema.json").read_text(encoding="utf-8"))
    jsonschema.validate(cards, schema)


def test_a_missing_card_warns_and_fails_strict(tmp_path):
    root = _tree(tmp_path, {"box": "arm", "drone": "uav"}, GOOD)
    errors, warnings = tc.check(root)
    assert errors == [] and len(warnings) == 1 and "tasks/uav/drone/ has no card" in warnings[0]
    errors, warnings = tc.check(root, strict=True)
    assert len(errors) == 1 and warnings == []
    assert tc.main(["--root", str(root), "check", "--strict"]) == 1
    assert tc.main(["--root", str(root), "check"]) == 0


@pytest.mark.parametrize("card, needle", [
    ('title = "Open a box"\nplatforms = ["uav"]\nstages = ["behavior"]', "the first platform is the folder's"),
    ('title = "Open a box"\nplatforms = ["arm", "rover"]\nstages = ["behavior"]', "'rover' is not a key"),
    ('title = "Open a box"\nplatforms = ["arm", "arm"]\nstages = ["behavior"]', "lists a key twice"),
    ('title = "Open a box"\nplatforms = ["arm"]\nstages = ["behavior", "control"]', "workflow order"),
    ('title = "Open a box"\nplatforms = ["arm"]\nstages = ["end-to-end", "control"]', "stands alone"),
    ('title = "Open a box"\nplatforms = ["arm"]\nstages = ["planning"]', "'planning' is not one of"),
    ('title = "open a box"\nplatforms = ["arm"]\nstages = ["behavior"]', "sentence case"),
    ('title = "Open a box."\nplatforms = ["arm"]\nstages = ["behavior"]', "without a period"),
    ('title = "' + "A" * 49 + '"\nplatforms = ["arm"]\nstages = ["behavior"]', "more than 48"),
    ('title = "Open a box"\nsummary = "no period"\nplatforms = ["arm"]\nstages = ["behavior"]', "one sentence"),
    ('title = "Open a box"\nplatforms = ["arm"]', "missing 'stages'"),
    ('title = "Open a box"\nplatforms = ["arm"]\nstages = ["behavior"]\nstage = "x"', "unknown key 'stage'"),
])
def test_a_card_that_breaks_a_rule_is_an_error(tmp_path, card, needle):
    root = _tree(tmp_path, {"box": "arm"}, "[box]\n" + card + "\n")
    errors, _ = tc.check(root)
    assert any(needle in e for e in errors), errors


def test_a_card_for_no_task_and_a_title_used_twice_are_errors(tmp_path):
    cards = GOOD + '\n[ghost]\ntitle = "Open a box"\nplatforms = ["arm"]\nstages = ["behavior"]\n'
    root = _tree(tmp_path, {"box": "arm"}, cards)
    errors, _ = tc.check(root)
    assert any("no task tasks/<platform>/ghost/" in e for e in errors), errors
    assert any("the same title as [box]" in e for e in errors), errors


def test_several_stages_in_order_and_end_to_end_alone_pass(tmp_path):
    cards = ('[box]\ntitle = "Open a box"\nplatforms = ["arm", "ground"]\nstages = ["hardware-design", "environment", "behavior"]\n'
             '[line]\ntitle = "Build a robot end to end"\nplatforms = ["uav"]\nstages = ["end-to-end"]\n')
    root = _tree(tmp_path, {"box": "arm", "line": "uav"}, cards)
    assert tc.check(root, strict=True) == ([], [])


def test_todo_and_context(tmp_path, capsys):
    root = _tree(tmp_path, {"box": "arm", "robotic_arm_cad_design_from_photos": "arm",
                            "uav_cad_design_from_photos": "uav"},
                 GOOD + '\n[uav_cad_design_from_photos]\ntitle = "Model a quadrotor in CAD"\nplatforms = ["uav"]\n'
                        'stages = ["hardware-design"]\n\n[ghost]\ntitle = "Gone"\nplatforms = ["arm"]\nstages = ["behavior"]\n')
    assert tc.main(["--root", str(root), "todo"]) == 0
    out = capsys.readouterr().out
    assert "no card: tasks/arm/robotic_arm_cad_design_from_photos/" in out and "no task: [ghost]" in out
    text = tc.context(root, "robotic_arm_cad_design_from_photos")
    assert "title: The robotic_arm_cad_design_from_photos brief" in text
    assert "What the agent does." in text and "A file." in text
    assert "uav_cad_design_from_photos" in text  # a card of its family, for the title pattern
