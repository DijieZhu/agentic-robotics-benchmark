"""scripts/task_assets.py and shared/task_assets/fetch_assets.py: one configuration file redirects every task.

Since the Harbor layout (2026-11-03) a task's asset list and its copy of the fetcher sit in environment/, the image's
build context; a task whose separate verifier image (tests/Dockerfile) fetches assets too has them in tests/ as well.
"""
import functools
import hashlib
import http.server
import shutil
import subprocess
import sys
import threading
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import task_assets as TA  # noqa: E402

REV = "0123456789abcdef0123456789abcdef01234567"


def make_repo(tmp: Path, dataset="org-a/assets", endpoint="https://huggingface.co") -> Path:
    (tmp / "shared" / "task_assets").mkdir(parents=True)
    shutil.copyfile(ROOT / TA.CANONICAL_FETCHER, tmp / TA.CANONICAL_FETCHER)
    (tmp / TA.CONFIG).write_text(f'[huggingface]\ndataset = "{dataset}"\nendpoint = "{endpoint}"\n')
    return tmp


def add_task(root: Path, files: dict[str, bytes], name="demo_task") -> tuple[Path, Path]:
    task = root / "tasks" / "arm" / name
    (task / "environment").mkdir(parents=True)
    payload = root / "upload" / name
    for rel, data in files.items():
        (payload / rel).parent.mkdir(parents=True, exist_ok=True)
        (payload / rel).write_bytes(data)
    TA.write_list(task, payload, REV, root)
    return task, payload


def test_repository_config_and_lists_agree():
    lists = TA.asset_lists(ROOT)
    assert lists, "no environment/assets.sha256 under tasks/: the check would pass on nothing"
    assert all(lst.parent.name in TA.CONTEXT_DIRS for lst in lists), lists
    assert TA.check(ROOT) == []


def test_list_writes_header_from_config(tmp_path):
    root = make_repo(tmp_path)
    task, _ = add_task(root, {"photos/a.jpg": b"A", "servo/s.step": b"S"})
    head, files, problems = TA.parse(task / "environment" / TA.LIST_NAME)
    assert problems == []
    assert head == {"endpoint": "https://huggingface.co", "repo": "org-a/assets", "revision": REV,
                    "prefix": "tasks/arm/demo_task/payload"}
    assert files == [(hashlib.sha256(b"A").hexdigest(), "photos/a.jpg"), (hashlib.sha256(b"S").hexdigest(), "servo/s.step")]
    assert (task / "environment" / TA.FETCHER_NAME).read_bytes() == (ROOT / TA.CANONICAL_FETCHER).read_bytes()
    assert TA.check(root) == []


def test_renaming_the_dataset_is_one_config_edit_and_a_sync(tmp_path):
    root = make_repo(tmp_path)
    t1, _ = add_task(root, {"a.bin": b"1"}, "task_one")
    t2, _ = add_task(root, {"b.bin": b"2"}, "task_two")
    (root / TA.CONFIG).write_text('[huggingface]\ndataset = "org-b/renamed"\nendpoint = "https://hf-mirror.example"\n')
    problems = TA.check(root)
    assert len(problems) == 4 and all("differs from task-assets.toml" in p for p in problems)
    changed = TA.sync(root)
    assert sorted(changed) == ["tasks/arm/task_one/environment/assets.sha256",
                               "tasks/arm/task_two/environment/assets.sha256"]
    assert TA.check(root) == []
    for t in (t1, t2):
        head, files, _ = TA.parse(t / "environment" / TA.LIST_NAME)
        assert head["repo"] == "org-b/renamed" and head["endpoint"] == "https://hf-mirror.example"
        assert head["revision"] == REV and len(files) == 1      # pins and digests are untouched
    assert TA.sync(root) == []                                   # idempotent


def test_check_catches_a_modified_fetcher_and_a_wrong_prefix(tmp_path):
    root = make_repo(tmp_path)
    task, _ = add_task(root, {"a.bin": b"1"})
    (task / "environment" / TA.FETCHER_NAME).write_text("# edited\n")
    lst = task / "environment" / TA.LIST_NAME
    lst.write_text(lst.read_text().replace("tasks/arm/demo_task/payload", "tasks/arm/other/payload"))
    problems = TA.check(root)
    assert any("differs from shared/task_assets/fetch_assets.py" in p for p in problems)
    assert any("should be 'tasks/arm/demo_task/payload'" in p for p in problems)


def test_a_separate_verifier_images_list_is_checked_and_synced(tmp_path):
    root = make_repo(tmp_path)
    task, payload = add_task(root, {"scenes/a.zip": b"Z"})
    (task / "tests").mkdir()
    lst = TA.write_list(task, payload, REV, root, context="tests")
    assert lst == task / "tests" / TA.LIST_NAME
    assert lst.read_text() == (task / "environment" / TA.LIST_NAME).read_text()
    assert (task / "tests" / TA.FETCHER_NAME).read_bytes() == (ROOT / TA.CANONICAL_FETCHER).read_bytes()
    assert TA.asset_lists(root) == [task / "environment" / TA.LIST_NAME, lst]
    assert TA.check(root) == []
    (task / "tests" / TA.FETCHER_NAME).write_text("# edited\n")
    assert TA.check(root) == ["tasks/arm/demo_task/tests/fetch_assets.py: differs from "
                              "shared/task_assets/fetch_assets.py; run 'python3 scripts/task_assets.py sync'"]
    (root / TA.CONFIG).write_text('[huggingface]\ndataset = "org-b/renamed"\nendpoint = "https://huggingface.co"\n')
    assert sorted(TA.sync(root)) == ["tasks/arm/demo_task/environment/assets.sha256",
                                     "tasks/arm/demo_task/tests/assets.sha256",
                                     "tasks/arm/demo_task/tests/fetch_assets.py"]
    assert TA.check(root) == []
    assert TA.parse(lst)[0]["repo"] == "org-b/renamed"
    with pytest.raises(SystemExit):
        TA.write_list(task, payload, REV, root, context="solution")


def test_check_catches_a_missing_fetcher(tmp_path):
    root = make_repo(tmp_path)
    task, _ = add_task(root, {"a.bin": b"1"})
    (task / "environment" / TA.FETCHER_NAME).unlink()
    assert TA.check(root) == [f"tasks/arm/demo_task/environment: no {TA.FETCHER_NAME} next to {TA.LIST_NAME}"]
    assert TA.sync(root) == ["tasks/arm/demo_task/environment/fetch_assets.py"]
    assert TA.check(root) == []


def test_the_cli_reads_the_list_in_environment(tmp_path, capsys):
    root = make_repo(tmp_path)
    task, _ = add_task(root, {"a.bin": b"1", "b/c.bin": b"2"})
    assert TA.main(["--root", str(root), "check"]) == 0
    assert "1 asset list(s) checked" in capsys.readouterr().out
    assert TA.main(["--root", str(root), "url", str(task)]) == 0
    assert capsys.readouterr().out.strip() == (
        f"https://huggingface.co/datasets/org-a/assets/tree/{REV}/tasks/arm/demo_task/payload  (2 files)")


def test_bad_config_is_refused(tmp_path):
    root = make_repo(tmp_path, dataset="no-slash")
    with pytest.raises(SystemExit):
        TA.load_config(root)


def test_fetcher_downloads_from_the_configured_endpoint(tmp_path):
    """The vendored fetcher follows the endpoint / repo written by sync: serve a fake Hub and fetch through it."""
    root = make_repo(tmp_path / "repo", dataset="org-b/renamed")
    task, payload = add_task(root, {"photos/p 1.jpg": b"photo", "doc.png": b"doc"})
    hub = tmp_path / "hub"
    base = hub / "datasets" / "org-b" / "renamed" / "resolve" / REV / "tasks" / "arm" / "demo_task" / "payload"
    shutil.copytree(payload, base)
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(hub))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        endpoint = "http://127.0.0.1:%d" % server.server_address[1]
        (root / TA.CONFIG).write_text(f'[huggingface]\ndataset = "org-b/renamed"\nendpoint = "{endpoint}"\n')
        TA.sync(root)
        dest = tmp_path / "out"
        fetcher, listing = task / "environment" / TA.FETCHER_NAME, task / "environment" / TA.LIST_NAME
        res = subprocess.run([sys.executable, str(fetcher), str(listing), str(dest)],
                             capture_output=True, text=True, timeout=60)
        assert res.returncode == 0, res.stderr
        assert (dest / "photos" / "p 1.jpg").read_bytes() == b"photo" and (dest / "doc.png").read_bytes() == b"doc"
        (base / "doc.png").write_bytes(b"tampered")
        res = subprocess.run([sys.executable, str(fetcher), str(listing), str(tmp_path / "out2")],
                             capture_output=True, text=True, timeout=60)
        assert res.returncode != 0 and "sha256 mismatch" in res.stderr
    finally:
        server.shutdown()
