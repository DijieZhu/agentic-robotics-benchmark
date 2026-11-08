#!/usr/bin/env python3
"""Task assets: keep every task's asset list pointing at the repository named in `task-assets.toml`.

A task's large agent-visible files live in the benchmark's Hugging Face dataset, under
`tasks/<platform>/<name>/payload/`. The task pins them in `environment/assets.sha256` (a header naming the endpoint,
the dataset, a dataset commit and the folder, then one sha256 per file), and `environment/fetch_assets.py` downloads
them while the image is built. The image's build context is the task's `environment/` folder (less `setup/`), so the
dataset name has to be written into each task; this script writes it from the one configuration file and checks that
nothing drifted. A task whose separate verifier image (`tests/Dockerfile`, built from `tests/`) fetches assets as well
carries the same two files in `tests/`, and they are synced and checked the same way.

    python3 scripts/task_assets.py sync            # rewrite every task's endpoint / repo header, re-vendor the fetcher
    python3 scripts/task_assets.py check           # CI: fail on any task that disagrees with task-assets.toml
    python3 scripts/task_assets.py list TASK_DIR PAYLOAD_DIR --revision COMMIT [--context tests]
                                                   # write TASK_DIR/environment/assets.sha256 for PAYLOAD_DIR's files
                                                   # (--context tests: the separate verifier image's list)
    python3 scripts/task_assets.py url TASK_DIR    # print where a task's files are downloaded from

Stdlib only. Usage and the rename procedure: docs/task-assets.md.
"""
from __future__ import annotations

import argparse
import hashlib
import re
import shutil
import sys
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python < 3.11: task-assets.toml uses only [tables] and key = "string" lines
    tomllib = None

ROOT = Path(__file__).resolve().parent.parent
CONFIG = "task-assets.toml"
CANONICAL_FETCHER = Path("shared") / "task_assets" / "fetch_assets.py"
#: The task folder that holds the asset list and the fetcher: the image's build context.
CONTEXT_DIR = "environment"
#: Every task folder that can hold them: the image's build context, and the build context of a separate verifier image
#: (`tests/Dockerfile`).
CONTEXT_DIRS = (CONTEXT_DIR, "tests")
LIST_NAME = "assets.sha256"
FETCHER_NAME = "fetch_assets.py"
HEADER_KEYS = ("endpoint", "repo", "revision", "prefix")
SYNCED_KEYS = ("endpoint", "repo")
REVISION = re.compile(r"^[0-9a-f]{40}$")
DIGEST = re.compile(r"^[0-9a-f]{64}$")
DATASET = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$")


def _read_toml(path: Path) -> dict:
    if tomllib is not None:
        with open(path, "rb") as fh:
            return tomllib.load(fh)
    cfg: dict = {}
    table = cfg
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = re.match(r"^\[([A-Za-z0-9_.-]+)\]$", line)
        if m:
            table = cfg.setdefault(m.group(1), {})
            continue
        m = re.match(r'^([A-Za-z0-9_-]+)\s*=\s*"([^"]*)"\s*(#.*)?$', line)
        if not m:
            raise SystemExit(f"{path.name}: cannot read line {line!r} (only [tables] and key = \"string\")")
        table[m.group(1)] = m.group(2)
    return cfg


def load_config(root: Path = ROOT) -> dict[str, str]:
    """{"endpoint", "repo"} from task-assets.toml."""
    cfg = _read_toml(root / CONFIG)
    hf = cfg.get("huggingface") or {}
    repo = str(hf.get("dataset", "")).strip()
    endpoint = str(hf.get("endpoint", "https://huggingface.co")).strip().rstrip("/")
    if not DATASET.match(repo):
        raise SystemExit(f"{CONFIG}: [huggingface] dataset must be '<owner>/<name>', got {repo!r}")
    if not endpoint.startswith(("https://", "http://")):
        raise SystemExit(f"{CONFIG}: [huggingface] endpoint must be an http(s) URL, got {endpoint!r}")
    return {"endpoint": endpoint, "repo": repo}


def asset_lists(root: Path = ROOT) -> list[Path]:
    """Every task's (and template's) environment/assets.sha256, and tests/assets.sha256 where a separate verifier image
    fetches assets too."""
    found = []
    for base in ("tasks", "templates"):
        found += sorted(p for context in CONTEXT_DIRS for p in (root / base).glob(f"**/{context}/{LIST_NAME}"))
    return [p for p in found if "vendor" not in p.relative_to(root).parts]


def parse(path: Path) -> tuple[dict[str, str], list[tuple[str, str]], list[str]]:
    """(header, [(sha256, relative path)], problems)."""
    head: dict[str, str] = {}
    files: list[tuple[str, str]] = []
    problems: list[str] = []
    for n, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        if line.startswith("#"):
            key, _, value = line[1:].partition(":")
            head[key.strip()] = value.strip()
            continue
        parts = line.split(None, 1)
        if len(parts) != 2 or not DIGEST.match(parts[0].lower()):
            problems.append(f"line {n}: not '<sha256>  <relative path>'")
            continue
        rel = parts[1].strip().lstrip("*")
        if rel.startswith("/") or ".." in rel.split("/"):
            problems.append(f"line {n}: unsafe path {rel!r}")
        files.append((parts[0].lower(), rel))
    return head, files, problems


def render(head: dict[str, str], files: list[tuple[str, str]]) -> str:
    lines = [f"# {k}: {head[k]}" for k in HEADER_KEYS if head.get(k)]
    lines += [f"# {k}: {v}" for k, v in head.items() if k not in HEADER_KEYS]
    lines += [f"{digest}  {rel}" for digest, rel in files]
    return "\n".join(lines) + "\n"


def expected_prefix(list_path: Path, root: Path = ROOT) -> str | None:
    """tasks/<platform>/<name>/payload for a task at tasks/<platform>/<name>/ (None for a template)."""
    task = list_path.parent.parent.relative_to(root)
    if len(task.parts) == 3 and task.parts[0] == "tasks":
        return f"{task.as_posix()}/payload"
    return None


def check(root: Path = ROOT) -> list[str]:
    cfg = load_config(root)
    canonical = (root / CANONICAL_FETCHER).read_bytes()
    problems = []
    for lst in asset_lists(root):
        rel = lst.relative_to(root).as_posix()
        head, files, bad = parse(lst)
        problems += [f"{rel}: {p}" for p in bad]
        for key in SYNCED_KEYS:
            if head.get(key, "").rstrip("/") != cfg[key]:
                problems.append(f"{rel}: '# {key}: {head.get(key, '')}' differs from {CONFIG} ({cfg[key]}); "
                                f"run 'python3 scripts/task_assets.py sync'")
        if not REVISION.match(head.get("revision", "")):
            problems.append(f"{rel}: '# revision:' must be a 40-hex dataset commit, got {head.get('revision')!r}")
        want = expected_prefix(lst, root)
        if want is not None and head.get("prefix") != want:
            problems.append(f"{rel}: '# prefix: {head.get('prefix')}' should be '{want}' (the task's folder in the dataset)")
        if not files:
            problems.append(f"{rel}: lists no file")
        fetcher = lst.parent / FETCHER_NAME
        if not fetcher.is_file():
            problems.append(f"{lst.parent.relative_to(root).as_posix()}: no {FETCHER_NAME} next to {LIST_NAME}")
        elif fetcher.read_bytes() != canonical:
            problems.append(f"{fetcher.relative_to(root).as_posix()}: differs from {CANONICAL_FETCHER.as_posix()}; "
                            f"run 'python3 scripts/task_assets.py sync'")
    return problems


def sync(root: Path = ROOT) -> list[str]:
    """Rewrite the synced header keys and re-vendor the fetcher; returns the files changed."""
    cfg = load_config(root)
    canonical = root / CANONICAL_FETCHER
    changed = []
    for lst in asset_lists(root):
        head, files, bad = parse(lst)
        if bad:
            raise SystemExit(f"{lst.relative_to(root)}: {'; '.join(bad)}")
        new_head = dict(head)
        new_head.update(cfg)
        text = render(new_head, files)
        if text != lst.read_text():
            lst.write_text(text)
            changed.append(lst.relative_to(root).as_posix())
        fetcher = lst.parent / FETCHER_NAME
        if not fetcher.is_file() or fetcher.read_bytes() != canonical.read_bytes():
            shutil.copyfile(canonical, fetcher)
            changed.append(fetcher.relative_to(root).as_posix())
    return changed


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_list(task_dir: Path, payload: Path, revision: str, root: Path = ROOT, context: str = CONTEXT_DIR) -> Path:
    """<context>/assets.sha256 (environment/, or tests/ for a separate verifier image) for every regular file under
    `payload` (the folder uploaded to the dataset's tasks/<platform>/<name>/payload/), pinned at `revision`; vendors the
    fetcher next to it."""
    if not REVISION.match(revision):
        raise SystemExit(f"--revision must be the 40-hex dataset commit, got {revision!r}")
    if context not in CONTEXT_DIRS:
        raise SystemExit(f"--context must be one of {', '.join(CONTEXT_DIRS)}, got {context!r}")
    task_dir = task_dir.resolve()
    lst = task_dir / context / LIST_NAME
    prefix = expected_prefix(lst, root)
    if prefix is None:
        raise SystemExit(f"{task_dir} is not at tasks/<platform>/<name>/ in {root}")
    files = sorted((p.relative_to(payload).as_posix(), p) for p in payload.rglob("*") if p.is_file() and not p.is_symlink())
    if not files:
        raise SystemExit(f"{payload}: no files")
    head = dict(load_config(root), revision=revision, prefix=prefix)
    lst.write_text(render(head, [(sha256(p), rel) for rel, p in files]))
    shutil.copyfile(root / CANONICAL_FETCHER, lst.parent / FETCHER_NAME)
    return lst


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("sync", help="write task-assets.toml into every task's asset list and re-vendor the fetcher")
    sub.add_parser("check", help="fail if any task disagrees with task-assets.toml or the canonical fetcher")
    p = sub.add_parser("list", help="write a task's environment/assets.sha256 from a local copy of its payload")
    p.add_argument("task_dir", type=Path)
    p.add_argument("payload_dir", type=Path)
    p.add_argument("--revision", required=True, help="the dataset commit that holds these files")
    p.add_argument("--context", choices=CONTEXT_DIRS, default=CONTEXT_DIR,
                   help="the folder the list goes in: environment (default) or tests (a separate verifier image)")
    p = sub.add_parser("url", help="print where a task's files are downloaded from")
    p.add_argument("task_dir", type=Path)
    ap.add_argument("--root", type=Path, default=ROOT, help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    root = args.root.resolve()
    if args.cmd == "check":
        problems = check(root)
        for line in problems:
            print("::error::" + line if not sys.stdout.isatty() else "ERROR " + line)
        n = len(asset_lists(root))
        print(f"task assets: {n} asset list(s) checked against {CONFIG}; {len(problems)} problem(s)")
        return 1 if problems else 0
    if args.cmd == "sync":
        changed = sync(root)
        print("\n".join(changed) if changed else "nothing to change")
        return 0
    if args.cmd == "list":
        print(write_list(args.task_dir, args.payload_dir.resolve(), args.revision, root, args.context))
        return 0
    head, files, _ = parse(args.task_dir.resolve() / CONTEXT_DIR / LIST_NAME)
    endpoint = (head.get("endpoint") or "https://huggingface.co").rstrip("/")
    print(f"{endpoint}/datasets/{head.get('repo')}/tree/{head.get('revision')}/{head.get('prefix')}  ({len(files)} files)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
