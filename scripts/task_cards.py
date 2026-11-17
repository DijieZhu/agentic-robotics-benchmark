#!/usr/bin/env python3
"""Task cards: each task's display card in `tasks/cards.toml`, checked against the task tree.

A card names and files a task for the website and other displays: a readable `title`, a one-sentence `summary`, the
`platforms` the task involves (the first one is its folder's) and the workflow `stages` it covers. Maintainers keep
the file; it sits outside every task folder, so a card is no part of a task's identity (`spec_hash`,
`content_digest`).

    uv run python scripts/task_cards.py check            # CI: every card is well formed and names a task on the
                                                         # tree; a task without a card is a warning
    uv run python scripts/task_cards.py check --strict   # the same, and a task without a card is an error
    uv run python scripts/task_cards.py todo             # the tasks without a card, and the cards without a task
    uv run python scripts/task_cards.py context TASK     # what naming a task needs: its task.toml facts, its brief's
                                                         # title, headings, first paragraph and deliverables, and the
                                                         # cards of its family

Stdlib only (Python 3.11+: tomllib). The rules: docs/task-cards.md.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CARDS = Path("tasks") / "cards.toml"
PLATFORMS = Path("tasks") / "platforms.json"

#: The stages of the workflow, in order; "end-to-end" is a category of its own and stands alone.
STAGES = ("requirements", "hardware-design", "assembly", "robot-description", "control", "environment", "behavior")
END_TO_END = "end-to-end"
FIELDS = {"title", "summary", "platforms", "stages"}
REQUIRED = ("title", "platforms", "stages")
TITLE_MAX = 48
SUMMARY_MAX = 200
#: Task-name families whose cards share one title pattern (docs/task-cards.md).
FAMILIES = ("requirements_replica", "cad_design", "full_assembly", "robot_description", "environment_twin")


def tasks_on_tree(root: Path) -> dict[str, str]:
    """{slug: platform} for every `tasks/<platform>/<slug>/task.toml`."""
    found: dict[str, str] = {}
    for toml in sorted((root / "tasks").glob("*/*/task.toml")):
        found.setdefault(toml.parent.name, toml.parent.parent.name)
    return found


def platform_keys(root: Path) -> tuple[str, ...]:
    return tuple(entry["key"] for entry in json.loads((root / PLATFORMS).read_text(encoding="utf-8")))


def load_cards(root: Path) -> dict:
    path = root / CARDS
    if not path.exists():
        return {}
    with open(path, "rb") as fh:
        return tomllib.load(fh)


def card_problems(slug: str, card: object, tasks: dict[str, str], platforms: tuple[str, ...]) -> list[str]:
    """What is wrong with one card (an empty list when nothing is)."""
    where = f"tasks/cards.toml [{slug}]"
    if not isinstance(card, dict):
        return [f"{where}: a card is a table"]
    problems = []
    if slug not in tasks:
        problems.append(f"{where}: no task tasks/<platform>/{slug}/ on the tree")
    for key in sorted(set(card) - FIELDS):
        problems.append(f"{where}: unknown key {key!r} (a card has {', '.join(sorted(FIELDS))})")
    for key in REQUIRED:
        if key not in card:
            problems.append(f"{where}: missing {key!r}")

    title = card.get("title")
    if "title" in card:
        if not isinstance(title, str) or not title.strip():
            problems.append(f"{where}: title must be a non-empty string")
        else:
            if len(title) > TITLE_MAX:
                problems.append(f"{where}: title is {len(title)} characters, more than {TITLE_MAX}")
            if title != title.strip() or "  " in title:
                problems.append(f"{where}: title has stray spaces")
            if not re.match(r"[A-Z0-9]", title):
                problems.append(f"{where}: title starts with a capital letter or a digit (sentence case)")
            if title.endswith("."):
                problems.append(f"{where}: title ends without a period")

    summary = card.get("summary")
    if "summary" in card:
        if not isinstance(summary, str) or not summary.strip():
            problems.append(f"{where}: summary must be a non-empty string")
        else:
            if len(summary) > SUMMARY_MAX:
                problems.append(f"{where}: summary is {len(summary)} characters, more than {SUMMARY_MAX}")
            if not re.match(r"[A-Z0-9]", summary) or not summary.endswith("."):
                problems.append(f"{where}: summary is one sentence: a capital letter first, a period last")

    plats = card.get("platforms")
    if "platforms" in card:
        if not isinstance(plats, list) or not plats or not all(isinstance(p, str) for p in plats):
            problems.append(f"{where}: platforms is a non-empty list of platform keys")
        else:
            for key in plats:
                if key not in platforms:
                    problems.append(f"{where}: platform {key!r} is not a key of tasks/platforms.json")
            if len(set(plats)) != len(plats):
                problems.append(f"{where}: platforms lists a key twice")
            if slug in tasks and plats[0] != tasks[slug]:
                problems.append(f"{where}: the first platform is the folder's, {tasks[slug]!r}, not {plats[0]!r}")

    stages = card.get("stages")
    if "stages" in card:
        if not isinstance(stages, list) or not stages or not all(isinstance(s, str) for s in stages):
            problems.append(f"{where}: stages is a non-empty list of stage keys")
        elif END_TO_END in stages:
            if stages != [END_TO_END]:
                problems.append(f"{where}: {END_TO_END!r} stands alone (a task that covers some stages lists them)")
        else:
            unknown = [s for s in stages if s not in STAGES]
            for s in unknown:
                problems.append(f"{where}: stage {s!r} is not one of {', '.join(STAGES)} or {END_TO_END!r}")
            if not unknown:
                order = [STAGES.index(s) for s in stages]
                if len(set(order)) != len(order):
                    problems.append(f"{where}: stages lists a stage twice")
                elif order != sorted(order):
                    problems.append(f"{where}: stages are in workflow order ({', '.join(STAGES)})")
    return problems


def check(root: Path, strict: bool = False) -> tuple[list[str], list[str]]:
    """(errors, warnings) for the whole file against the tree."""
    tasks = tasks_on_tree(root)
    platforms = platform_keys(root)
    try:
        cards = load_cards(root)
    except tomllib.TOMLDecodeError as exc:
        return [f"tasks/cards.toml does not parse: {exc}"], []
    errors: list[str] = []
    for slug, card in cards.items():
        errors += card_problems(slug, card, tasks, platforms)
    titles: dict[str, str] = {}
    for slug, card in cards.items():
        title = card.get("title") if isinstance(card, dict) else None
        if isinstance(title, str):
            other = titles.setdefault(title.strip().lower(), slug)
            if other != slug:
                errors.append(f"tasks/cards.toml [{slug}]: the same title as [{other}]")
    missing = [f"tasks/{tasks[s]}/{s}/ has no card in tasks/cards.toml" for s in sorted(tasks) if s not in cards]
    return (errors + missing, []) if strict else (errors, missing)


def _annotate(kind: str, message: str) -> str:
    return f"::{kind}::{message}" if os.environ.get("GITHUB_ACTIONS") == "true" else f"{kind}: {message}"


# --------------------------------------------------------------------------- context
def _brief_outline(text: str) -> dict:
    lines = text.splitlines()
    title = next((ln.lstrip("#").strip() for ln in lines if ln.strip()), "")
    headings = [ln.strip() for ln in lines if re.match(r"#{2,3}\s", ln)]
    paragraph: list[str] = []
    started = False
    for ln in lines[1:] if lines and lines[0].startswith("#") else lines:
        if not ln.strip():
            if started:
                break
            continue
        if ln.startswith(("#", "|", "```", "- ", "* ")):
            if started:
                break
            continue
        started = True
        paragraph.append(ln.strip())
    deliverables: list[str] = []
    for i, ln in enumerate(lines):
        if re.match(r"#{2,3}\s.*(produce|submit|deliver|hand in)", ln, re.IGNORECASE):
            for nxt in lines[i + 1:]:
                if re.match(r"#{1,3}\s", nxt):
                    break
                deliverables.append(nxt)
            break
    return {"title": title, "headings": headings, "first_paragraph": " ".join(paragraph),
            "deliverables": "\n".join(deliverables).strip()[:2500]}


def context(root: Path, slug: str) -> str:
    tasks = tasks_on_tree(root)
    if slug not in tasks:
        raise SystemExit(f"no task {slug!r} on the tree")
    folder = root / "tasks" / tasks[slug] / slug
    with open(folder / "task.toml", "rb") as fh:
        spec = tomllib.load(fh)
    env = spec.get("environment", {})
    venv = spec.get("verifier", {}).get("environment", {})
    facts = {
        "folder": f"tasks/{tasks[slug]}/{slug}/",
        "metadata": spec.get("metadata", {}),
        "agent": {k: spec.get("agent", {}).get(k) for k in ("timeout_sec", "network_mode")},
        "environment": {k: env.get(k) for k in ("cpus", "memory_mb", "gpus", "gpu_types") if k in env},
        "verifier_environment": {k: venv.get(k) for k in ("cpus", "memory_mb", "gpus", "gpu_types") if k in venv},
        "mcp_servers": [s.get("name") for s in env.get("mcp_servers", []) if isinstance(s, dict)],
    }
    anchor = folder / "tests" / "anchor.json"
    if anchor.exists():
        try:
            facts["anchor"] = {k: v for k, v in json.loads(anchor.read_text(encoding="utf-8")).items()
                               if k in ("metric", "direction", "anchor", "full_at")}
        except ValueError:
            pass
    brief = folder / "instruction.md"
    outline = _brief_outline(brief.read_text(encoding="utf-8")) if brief.exists() else {}
    cards = load_cards(root)
    family = next((f for f in FAMILIES if f in slug), None)
    kin = {s: c for s, c in cards.items() if family and family in s and s != slug} if family else {}
    out = [f"# {slug}", "", "## Facts (task.toml, tests/anchor.json)", json.dumps(facts, indent=1, ensure_ascii=False),
           "", "## The brief (instruction.md)", f"title: {outline.get('title', '')}", "headings:"]
    out += [f"  {h}" for h in outline.get("headings", [])]
    out += ["first paragraph:", f"  {outline.get('first_paragraph', '')}", "deliverables:", outline.get("deliverables", ""),
            "", f"## Its card now: {json.dumps(cards.get(slug), ensure_ascii=False)}",
            f"## Cards of its family ({family or 'none'})"]
    out += [f"  {s}: {json.dumps(c, ensure_ascii=False)}" for s, c in sorted(kin.items())]
    return "\n".join(out) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    chk = sub.add_parser("check", help="check every card against the tree")
    chk.add_argument("--strict", action="store_true", help="a task without a card is an error")
    sub.add_parser("todo", help="the tasks without a card, and the cards without a task")
    ctx = sub.add_parser("context", help="what naming a task needs")
    ctx.add_argument("task")
    ap.add_argument("--root", type=Path, default=ROOT, help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    root = a.root.resolve()
    if a.cmd == "check":
        errors, warnings = check(root, a.strict)
        for w in warnings:
            print(_annotate("warning", w))
        for e in errors:
            print(_annotate("error", e))
        cards = load_cards(root) if not errors else {}
        print(f"task cards: {len(errors)} error(s), {len(warnings)} warning(s)"
              + (f"; {len(cards)} card(s) for {len(tasks_on_tree(root))} task(s)" if not errors else ""))
        return 1 if errors else 0
    if a.cmd == "todo":
        tasks = tasks_on_tree(root)
        cards = load_cards(root)
        for s in sorted(set(tasks) - set(cards)):
            print(f"no card: tasks/{tasks[s]}/{s}/")
        for s in sorted(set(cards) - set(tasks)):
            print(f"no task: [{s}]")
        return 0
    sys.stdout.write(context(root, a.task))
    return 0


if __name__ == "__main__":
    sys.exit(main())
