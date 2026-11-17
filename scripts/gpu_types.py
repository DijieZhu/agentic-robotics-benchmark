#!/usr/bin/env python3
"""GPU models: check the `gpu_types` every GPU task names, and list them.

A task that needs a GPU (`gpus` in `[environment]` or `[verifier.environment]`) names in `gpu_types` the GPU models it
has been tested on, cheapest first (templates/README.md); it leaves the key out only when every model tested runs it.
The engine refuses a host without an accepted model before anything runs, so a name that accepts no real model makes
the task unrunnable everywhere.

    cd vendor/ale && uv run python ../../scripts/gpu_types.py check   # CI (engine-lint): every name accepts a
                                                                      # known GPU model, no name twice; a GPU table
                                                                      # without gpu_types is a notice
    uv run python scripts/gpu_types.py list                           # every GPU task and the models it names

`check` asks the pinned engine whether a name accepts a model (`ale.run.gpu.accepts_model`: the rule is the
engine's and is not repeated here), so it runs in the engine's environment. `list` needs only the standard library
(Python 3.11+: tomllib).
"""
from __future__ import annotations

import argparse
import os
import sys
import tomllib
from collections.abc import Callable
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TABLES = (("environment",), ("verifier", "environment"))

#: GPU models as `nvidia-smi --query-gpu=name` prints them: the ones the tasks have been run on, and the ones the
#: engine's providers offer. A `gpu_types` name must accept at least one of them. Add a model here when a task is
#: tested on it.
KNOWN_MODELS = (
    "NVIDIA GeForce RTX 3090",
    "NVIDIA GeForce RTX 4090",
    "NVIDIA GeForce RTX 5090",
    "NVIDIA RTX A6000",
    "NVIDIA RTX 6000 Ada Generation",
    "NVIDIA RTX PRO 6000 Blackwell Workstation Edition",
    "Tesla T4",
    "NVIDIA L4",
    "NVIDIA A10G",
    "NVIDIA L40S",
    "NVIDIA A100-SXM4-40GB",
    "NVIDIA A100-SXM4-80GB",
    "NVIDIA H100 80GB HBM3",
    "NVIDIA H200",
)


def _table(spec: dict, path: tuple[str, ...]) -> dict:
    for key in path:
        spec = spec.get(key, {}) if isinstance(spec, dict) else {}
    return spec if isinstance(spec, dict) else {}


def gpu_tables(root: Path) -> list[dict]:
    """One record per `[environment]` / `[verifier.environment]` table that asks for a GPU or names `gpu_types`."""
    out = []
    for toml in sorted((root / "tasks").glob("*/*/task.toml")):
        with open(toml, "rb") as fh:
            spec = tomllib.load(fh)
        for path in TABLES:
            table = _table(spec, path)
            gpus, types = table.get("gpus") or 0, table.get("gpu_types")
            if gpus or types is not None:
                out.append({"task": f"{toml.parent.parent.name}/{toml.parent.name}", "table": ".".join(path),
                            "gpus": gpus, "gpu_types": types})
    return out


def check(root: Path, accepts: Callable[[tuple[str, ...], str], bool]) -> tuple[list[str], list[str]]:
    """(errors, notices). `accepts(gpu_types, model)` is the engine's rule."""
    errors, notices = [], []
    for rec in gpu_tables(root):
        where = f"tasks/{rec['task']}/task.toml [{rec['table']}]"
        types = rec["gpu_types"]
        if types is None:
            notices.append(f"{where}: gpus = {rec['gpus']} without gpu_types, so every GPU model is accepted; "
                           "leave the key out only when every model tested runs the task")
            continue
        if not isinstance(types, list) or not types or not all(isinstance(t, str) and t.strip() for t in types):
            errors.append(f"{where}: gpu_types is a non-empty list of GPU model names")
            continue
        if len({t.strip().lower() for t in types}) != len(types):
            errors.append(f"{where}: gpu_types names a model twice")
        for name in types:
            if not any(accepts((name,), model) for model in KNOWN_MODELS):
                errors.append(f"{where}: gpu_types name {name!r} accepts no known GPU model "
                              "(a typo? the models: KNOWN_MODELS in scripts/gpu_types.py)")
    return errors, notices


def _annotate(kind: str, message: str) -> str:
    return f"::{kind}::{message}" if os.environ.get("GITHUB_ACTIONS") == "true" else f"{kind}: {message}"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check", help="every gpu_types name accepts a known GPU model (needs the engine)")
    sub.add_parser("list", help="every GPU task and the models it names")
    ap.add_argument("--root", type=Path, default=ROOT, help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    root = a.root.resolve()
    if a.cmd == "list":
        for rec in gpu_tables(root):
            types = "every model" if rec["gpu_types"] is None else ", ".join(rec["gpu_types"])
            print(f"{rec['task']:<58} [{rec['table']}] gpus={rec['gpus']}  {types}")
        return 0
    try:
        from ale.run.gpu import accepts_model
    except ImportError:
        print("gpu_types.py check needs the pinned engine: cd vendor/ale && uv run python ../../scripts/gpu_types.py check",
              file=sys.stderr)
        return 2
    errors, notices = check(root, accepts_model)
    for n in notices:
        print(_annotate("notice", n))
    for e in errors:
        print(_annotate("error", e))
    print(f"gpu types: {len(errors)} error(s), {len(notices)} notice(s); {len(gpu_tables(root))} GPU table(s)")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
