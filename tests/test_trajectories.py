"""scripts/trajectories.py: inventory, curate, export, scan and upload on a synthetic runs directory.

The episodes here are written by hand in the engine's record shapes (vendor/ale/docs/specs/trace.md): lock.json,
result.json, verification.json, an ATIF trajectory.json with a picture in blobs/, the transport and execution traces,
and a run-level resources.jsonl from the sampler. No engine, docker or network is needed; rclone is replaced by a fake.
"""

from __future__ import annotations

import csv
import datetime
import gzip
import hashlib
import importlib.util
import json
import re
import struct
import sys
import zlib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
_SPEC = importlib.util.spec_from_file_location("trajectories", ROOT / "scripts" / "trajectories.py")
TR = importlib.util.module_from_spec(_SPEC)
sys.modules["trajectories"] = TR
_SPEC.loader.exec_module(TR)  # type: ignore[union-attr]

T0 = datetime.datetime(2026, 11, 1, 10, 0, 0, tzinfo=datetime.timezone.utc)
# Token-shaped strings are assembled at run time so this file never holds one.
FAKE_ANTHROPIC = "sk-" + "ant-" + "api03-" + "x" * 32
FAKE_AWS_ID = "AKIA" + "Q" * 16


def ts(seconds: float) -> str:
    return (T0 + datetime.timedelta(seconds=seconds)).strftime("%Y-%m-%dT%H:%M:%S.000Z")


def png(w: int = 4, h: int = 3) -> bytes:
    raw = b"".join(b"\x00" + b"\x10\x20\x30" * w for _ in range(h))

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)

    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


def make_repo(tmp: Path) -> Path:
    """A checkout with one task: tests/anchor.json, task.toml and a payload whose module names the task's tools."""
    task = tmp / "repo" / "tasks" / "arm" / "demo_task"
    (task / "tests").mkdir(parents=True)
    (task / "environment" / "setup" / "payload").mkdir(parents=True)
    (task / "environment" / "setup" / "payload" / "sim_client.py").write_text("# the task's simulator client\n")
    (task / "task.toml").write_text('[task]\nname = "grounded-intelligence-initiative/demo_task"\n\n'
                                    "[agent]\ntimeout_sec = 3600\n\n[verifier]\ntimeout_sec = 1200\n")
    (task / "tests" / "anchor.json").write_text(json.dumps(
        {"metric": "demo_metric", "direction": "higher", "value": 0.8, "full_at": 0.9}))
    return tmp / "repo"


def make_episode(runs: Path, run: str, episode: str, repo: Path | None = None, *, task: str = "demo_task",
                 harness: str = "claude-code", model: str = "claude-opus-5-5", effort: str = "high",
                 reward: float | None = 0.33, ratio: float = 0.3, status: str = "completed", start: float = 0.0,
                 digest: str = "sha256:" + "1" * 64, validation: bool = False, samples: bool = True) -> Path:
    ep = runs / run / episode
    (ep / "blobs" / "image").mkdir(parents=True)
    if validation:
        (runs / run / "validation.json").write_text("{}")
    picture = png()
    pic = "blobs/image/sha256-%s.png" % hashlib.sha256(picture).hexdigest()
    (ep / pic).write_bytes(picture)
    source = ({"kind": "local", "path": str(repo / "tasks" / "arm" / task)} if repo else
              {"kind": "git", "commit": "0" * 40, "path": f"tasks/arm/{task}"})
    lock = {"agent": {"harness": harness, "model": model, "family": "autonomous", "version": "2.1.227",
                      "settings": {"effort": effort}},
            "framework": {"commit": "1" * 40, "version": "0.1.1"},
            "task": {"name": f"grounded-intelligence-initiative/{task}", "content_digest": digest, "source": source},
            "resources": {"effective": {"cpus": 2, "memory_mb": 4096, "network_mode": "block"},
                          "requested": {"gpus": 0}}}
    result = {"status": status, "started_at": ts(start), "finished_at": ts(start + 900), "failure": None,
              "phases": [{"phase": "agent", "outcome": "succeeded", "started_at": ts(start + 5),
                          "finished_at": ts(start + 305), "duration_ms": 300000},
                         {"phase": "verify", "outcome": "succeeded", "started_at": ts(start + 306),
                          "finished_at": ts(start + 900), "duration_ms": 594000}]}
    if reward is not None:
        result["rewards"] = {"reward": round(reward, 2)}
        (ep / "verification.json").write_text(json.dumps(
            {"reward": round(reward, 2), "metrics": {"ratio": ratio, "demo_metric": ratio * 0.8,
                                                     "reward_exact": reward}, "diagnostic": "demo diagnostic"}))
    trajectory = {
        "schema_version": "ATIF-v1.8", "session_id": "session-1", "trajectory_id": "trajectory-1",
        "agent": {"name": harness, "version": "2.1.227", "model_name": model},
        "steps": [
            {"step_id": 1, "source": "user", "message": "Build the demo submission."},
            {"step_id": 2, "source": "agent", "message": "Stepping the simulator, then a look.", "timestamp": ts(start + 15),
             "extra": {"ale": {"transport_call_ids": ["call_1"]}},
             "tool_calls": [
                 {"tool_call_id": "tc1", "function_name": "Bash",
                  "arguments": {"command": "python3 sim_client.py --step", "description": "Step the simulator"}},
                 {"tool_call_id": "tc2", "function_name": "Read", "arguments": {"file_path": "/home/user/cam.png"}}],
             "observation": {"results": [
                 {"source_call_id": "tc1", "content": "stepped"},
                 {"source_call_id": "tc2", "content": [{"type": "image",
                                                        "source": {"media_type": "image/png", "path": pic}}]}]}},
            {"step_id": 3, "source": "agent", "message": "Done.", "timestamp": ts(start + 25),
             "extra": {"ale": {"transport_call_ids": ["call_2"]}}},
        ]}
    calls = [
        {"kind": "call", "call_id": "call_1", "model": model, "input_tokens": 1_000_000, "cache_read_tokens": 600_000,
         "cache_write_tokens": 300_000, "output_tokens": 50_000, "timestamp": ts(start + 14), "latency_ms": 1000,
         "disposition": "forwarded", "purpose": "agent", "stop_reason": "tool_use"},
        {"kind": "call", "call_id": "call_2", "model": model, "input_tokens": 2_000_000, "cache_read_tokens": 1_900_000,
         "cache_write_tokens": 50_000, "output_tokens": 10_000, "timestamp": ts(start + 24), "latency_ms": 500,
         "disposition": "forwarded", "purpose": "agent", "stop_reason": "end_turn"},
    ]
    execution = [{"kind": "command_finished", "phase": "verify", "stderr": {"inline": "grader: 24 missions scored"}}]
    (ep / "lock.json").write_text(json.dumps(lock))
    (ep / "result.json").write_text(json.dumps(result))
    (ep / "trajectory.json").write_text(json.dumps(trajectory))
    (ep / "trace.transport.jsonl").write_text("".join(json.dumps(c) + "\n" for c in calls))
    (ep / "trace.execution.jsonl").write_text("".join(json.dumps(e) + "\n" for e in execution))
    if samples:
        row = {"episode": episode, "t": 0, "containers": [{"mem_bytes": 2_000_000_000, "pids": 12, "rw_bytes": 10**8}],
               "gpu_mem_bytes": 0}
        with open(runs / run / "resources.jsonl", "a") as f:
            f.write(json.dumps(row) + "\n")
    return ep


def inventory(runs: Path) -> list[dict]:
    return list(TR.inventory_rows(runs))


def test_inventory_lists_model_episodes_only(tmp_path):
    runs = tmp_path / "runs"
    make_episode(runs, "opus-r1", "demo_task-aaaa0001")
    make_episode(runs, "validate-1234abcd", "demo_task-oracle-bbbb0002", harness="oracle", model="", validation=True)
    make_episode(runs, "oracle-r1", "demo_task-cccc0003", harness="oracle", model="claude-opus-4-8")
    rows = inventory(runs)
    assert [(r["run"], r["episode"]) for r in rows] == [("opus-r1", "demo_task-aaaa0001")]
    assert rows[0]["task"] == "grounded-intelligence-initiative/demo_task" and rows[0]["reward"] == 0.33
    every = list(TR.inventory_rows(runs, include_all=True))
    assert {r["run"]: r.get("skip") for r in every} == {
        "opus-r1": None, "validate-1234abcd": "validation run", "oracle-r1": "the engine's oracle harness"}


def test_curate_keeps_the_latest_version_and_its_lowest_clean_score(tmp_path):
    runs = tmp_path / "runs"
    old, new = "sha256:" + "a" * 64, "sha256:" + "b" * 64
    make_episode(runs, "r-old", "demo_task-00000001", reward=0.05, start=0, digest=old)        # an earlier version
    make_episode(runs, "r-new", "demo_task-00000002", reward=0.6, start=100, digest=new)
    make_episode(runs, "r-new", "demo_task-00000003", reward=0.2, start=200, digest=new)
    make_episode(runs, "r-new", "demo_task-00000004", reward=0.0, start=300, digest=new, status="env_error")
    # a second agent: its only scored episode was lost to infrastructure, the clean one has no score
    make_episode(runs, "f-new", "demo_task-00000005", model="claude-fable-5-1", reward=0.0, start=400, digest=new,
                 status="env_error")
    make_episode(runs, "f-new", "demo_task-00000006", model="claude-fable-5-1", reward=None, start=500, digest=new)
    rows = TR.curate(inventory(runs), official={"demo_task"})
    assert [(r["agent"], r["episode"], r["score"]) for r in rows] == [
        ("claude-code_claude-fable-5-1_effort-high", "demo_task-00000005", 0.0),
        ("claude-code_claude-opus-5-5_effort-high", "demo_task-00000003", 0.2)]
    assert rows[1]["folder"] == "demo_task/claude-code_claude-opus-5-5_effort-high/r-new__demo_task-00000003"
    assert [r["episode"] for r in TR.curate(inventory(runs), official={"demo_task"}, pick="highest")][1] == \
        "demo_task-00000002"
    assert TR.curate(inventory(runs), official={"another_task"}) == []


def test_curate_renames_excludes_and_regrades(tmp_path):
    runs = tmp_path / "runs"
    ep = make_episode(runs, "r1", "gii.old_name-00000001", task="old_name", reward=0.1, start=0)
    make_episode(runs, "r2", "gii.old_name-00000002", task="old_name", reward=0.0, start=10)    # superseded
    (ep / "regrade_1").mkdir()
    (ep / "regrade_1" / "verification.json").write_text(json.dumps({"metrics": {"reward_exact": 0.9}}))
    rows = TR.curate(inventory(runs), official={"new_name"}, rename={"old_name": "new_name"}, exclude={"r2"},
                     regrades={"r1": {"verification": "regrade_1/verification.json", "note": "grader fixed"}})
    assert len(rows) == 1
    row = rows[0]
    assert (row["task"], row["score"], row["regrade"]) == ("new_name", 0.9, "regrade_1/verification.json")
    assert row["folder"] == "new_name/claude-code_claude-opus-5-5_effort-high/r1__gii.new_name-00000001"
    assert row["episode"] == "gii.old_name-00000001"                      # the records stay as the engine wrote them
    assert row["notes"] == "grader fixed | measured under its earlier name old_name"


def setup_run(tmp_path: Path) -> tuple[Path, Path, Path]:
    """A checkout, a runs directory with one model episode, and index.csv selecting it."""
    repo = make_repo(tmp_path)
    runs = tmp_path / "runs"
    make_episode(runs, "opus-r1", "gii.demo_task-abcd1234", repo)
    index = tmp_path / "index.csv"
    TR.write_index(index, TR.curate(inventory(runs), official=TR.official_tasks(repo / "tasks")))
    return repo, runs, index


def export(tmp_path: Path, *extra: str) -> tuple[Path, Path, dict]:
    repo, runs, index = (tmp_path / "repo", tmp_path / "runs", tmp_path / "index.csv")
    if not index.exists():
        setup_run(tmp_path)
    archive = tmp_path / "archive"
    assert TR.main(["export", "--runs-dir", str(runs), "--index", str(index), "--archive", str(archive),
                    "--repo", str(repo), *extra]) == 0
    folder = archive / "demo_task" / "claude-code_claude-opus-5-5_effort-high" / "opus-r1__gii.demo_task-abcd1234"
    return archive, folder, json.loads((folder / "run.json").read_text())


def test_export_writes_the_record_and_the_index(tmp_path):
    archive, folder, doc = export(tmp_path)
    assert (doc["format"], doc["slug"], doc["task"], doc["bench"]) == (3, "demo_task--abcd1234", "demo_task", "arm")
    assert (doc["run"], doc["agent"], doc["model"], doc["effort"], doc["mode"]) == (
        "claude_code-opus_5_5-high", "claude-code", "claude-opus-5-5", "high", "autonomous")
    assert (doc["state"], doc["success"], doc["ended_by"], doc["wall_s"]) == ("failed", 0, None, 300)
    assert doc["task_text"] == "Build the demo submission."
    # tokens: input counts cache reads and writes; list price per token type
    tok = doc["tokens"]
    assert (tok["in"], tok["cached"], tok["out"], tok["cache_write"], tok["uncached_in"]) == (
        3_000_000, 2_500_000, 60_000, 350_000, 150_000)
    assert tok["usd"] == 4.05 and tok["calls"] == 2 and tok["unlinked_calls"] == 0 and tok["unpriced_calls"] == 0
    # the per-step digest: one step per model call; the task's own tools are "robot", a picture read is "look"
    steps = doc["steps"]
    assert [s["id"] for s in steps] == [2, 3] and [s["t"] for s in steps] == [10.0, 20.0]
    assert [c["kind"] for c in steps[0]["calls"]] == ["robot", "look"]
    assert steps[0]["calls"][0]["label"] == "Step the simulator" and steps[0]["calls"][0]["exit"] == 0
    assert steps[0]["say"] == "Stepping the simulator, then a look." and steps[0]["tok"]["usd"] == 3.02
    picture = png()
    assert steps[0]["out"]["pics"] == [[0, len(picture), "image/png"]]
    assert (folder / "steps.bin").read_bytes() == picture
    assert doc["pictures"][0]["w"] == 4 and doc["pictures"][0]["h"] == 3 and doc["pictures"][0]["steps"] == [2]
    assert doc["digest"]["robot_rule"] == r"\b(?:sim_client)\b"          # from the task's payload
    # score, resources, provenance
    sc = doc["score"]
    assert (sc["metric"], sc["anchor"], sc["full_at"], sc["ratio"], sc["reward"]) == ("demo_metric", 0.8, 0.9, 0.3, 0.33)
    assert sc["grader_log_tail"] == "grader: 24 missions scored" and sc["diagnostic"] == "demo diagnostic"
    assert doc["reward"]["peak_memory_gb"] == 2.0 and doc["reward"]["resource_monitor_missing"] == 0
    assert doc["resources"]["limits"]["agent_timeout_s"] == 3600 and doc["resources"]["limits"]["verify_timeout_s"] == 1200
    assert doc["provenance"]["task"]["content_digest"] == "sha256:" + "1" * 64
    # the files: the episode as written, the trajectory as JSON lines, a poster
    for name in ("lock.json", "result.json", "trajectory.json", "trace.transport.jsonl", "resources.jsonl"):
        assert (folder / name).is_file(), name
    lines = gzip.decompress((folder / "trajectory.jsonl.gz").read_bytes()).decode().splitlines()
    assert len(lines) == 1 + 3 and "atif_header" in json.loads(lines[0])
    assert doc["poster"] in ("poster.jpg", "poster.png") and (folder / doc["poster"]).is_file()
    # the root index
    top = json.loads((archive / "index.json").read_text())
    assert (top["v"], top["total"], top["suites"], top["shards"]) == (3, 1, {"arm": 1}, {"arm": "index/arm.json"})
    shard = json.loads((archive / "index" / "arm.json").read_text())
    assert shard["runs"][0]["folder"] == folder.relative_to(archive).as_posix() and shard["runs"][0]["usd"] == 4.05
    assert (archive / "index.csv").is_file()


def test_export_again_and_derived_only_leave_one_consistent_folder(tmp_path):
    archive, folder, first = export(tmp_path)
    _, _, second = export(tmp_path, "--derived-only")
    assert second["steps"] == first["steps"] and second["tokens"] == first["tokens"]
    assert (folder / "lock.json").is_file()                              # derived-only left the records alone


def test_scan_reports_configured_secrets_and_token_patterns(tmp_path, capsys):
    secret_value = "s3cr3t-value-" + "z" * 20
    env = tmp_path / "gateway.env"
    env.write_text(f'ALE_ANTHROPIC_API_KEY="{secret_value}"\nOTHER_SETTING=plain-value-that-is-long\n')
    tree = tmp_path / "tree"
    tree.mkdir()
    (tree / "log.txt").write_text(f"the key was {secret_value} once\n")
    (tree / "notes.txt").write_text(f"token {FAKE_ANTHROPIC}\n")
    with gzip.open(tree / "trace.jsonl.gz", "wt") as f:
        f.write(f'{{"id": "{FAKE_AWS_ID}"}}\n')
    (tree / "picture.png").write_bytes(png() + FAKE_AWS_ID.encode())      # pictures are skipped
    assert TR.configured_secrets([env]) == [secret_value]
    hits = TR.scan_paths([tree], [secret_value], context=12)
    assert sorted((Path(p).name, k) for p, k, _, _ in hits) == [
        ("log.txt", "configured-secret-0"), ("notes.txt", "anthropic_key"), ("trace.jsonl.gz", "aws_access_key_id")]
    assert TR.main(["scan", str(tree), "--secrets-from", str(env), "--context", "12"]) == 1
    out = capsys.readouterr().out
    assert secret_value not in out and FAKE_ANTHROPIC not in out and "<redacted" in out
    (tree / "log.txt").unlink()
    (tree / "notes.txt").unlink()
    (tree / "trace.jsonl.gz").unlink()
    assert TR.main(["scan", str(tree), "--secrets-from", str(env)]) == 0


class FakeRclone:
    """Records rclone calls; `lsf` lists what `listing` holds."""

    def __init__(self, listing=""):
        self.calls, self.listing = [], listing

    def __call__(self, args, dry_run=False, check=True):
        self.calls.append(list(args))

        class Result:
            returncode, stdout, stderr = 0, (self.listing if args[0] == "lsf" else ""), ""
        return Result()


def test_upload_refuses_a_secret_and_copies_checks_and_prunes(tmp_path, monkeypatch):
    archive, folder, _ = export(tmp_path)
    monkeypatch.setattr(TR.shutil, "which", lambda name: "/usr/bin/" + name)
    rel = folder.relative_to(archive).as_posix()
    stale = "demo_task/claude-code_claude-opus-5-5_effort-high/opus-r0__gii.demo_task-00000000/"
    fake = FakeRclone(listing=f"demo_task/\ndemo_task/claude-code_claude-opus-5-5_effort-high/\n{rel}/\n{stale}\n"
                              "index/\n")
    monkeypatch.setattr(TR, "rclone", fake)
    (folder / "artifacts").mkdir(exist_ok=True)
    (folder / "artifacts" / "leak.txt").write_text(f"export KEY={FAKE_ANTHROPIC}\n")
    args = ["upload", "--archive", str(archive), "--dest", "remote:trajectories"]
    assert TR.main(args) == 1 and fake.calls == []                       # refused before any copy
    assert TR.main(args + ["--accept", f"{rel}/artifacts/leak.txt"]) == 0
    (folder / "artifacts" / "leak.txt").unlink()
    fake.calls.clear()
    assert TR.main(args + ["--prune"]) == 0
    verbs = [(c[0], c[2] if len(c) > 2 and c[0] in ("copy", "copyto") else c[1]) for c in fake.calls]
    assert ("copy", f"remote:trajectories/{rel}") in verbs
    assert ("check", "--one-way") in verbs
    assert ("copyto", "remote:trajectories/index.json") in verbs
    purged = [c[1] for c in fake.calls if c[0] == "purge"]
    assert purged == ["remote:trajectories/" + stale.rstrip("/")]        # only the folder index.csv no longer lists


def test_index_csv_round_trip_and_unsafe_folders(tmp_path):
    rows = [{"task": "t", "agent": "a", "score": 0.5, "status": "completed", "run_date": "2026-11-01",
             "task_commit": "", "run": "r", "episode": "e", "regrade": "", "notes": "", "folder": "t/a/r__e"}]
    TR.write_index(tmp_path / "index.csv", rows)
    assert TR.read_index(tmp_path / "index.csv")[0]["folder"] == "t/a/r__e"
    with open(tmp_path / "bad.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=TR.INDEX_FIELDS)
        w.writeheader()
        w.writerow(dict(rows[0], folder="../outside"))
    with pytest.raises(ValueError):
        TR.read_index(tmp_path / "bad.csv")


def test_helpers():
    assert TR.image_size(png(7, 5)) == (7, 5)
    assert TR.to_bytes("1.5GiB") == int(1.5 * 1024**3) and TR.to_bytes("10.2kB") == 10200 and TR.to_bytes("") is None
    assert TR.half_up(0.125) == 0.13 and TR.half_up(0.004) == 0.0
    assert TR.price_for("us.anthropic.claude-opus-5-5", TR.PRICES) == TR.PRICES["claude-opus-5-5"]
    assert TR.cap("x" * 100, 40).count("characters left out") == 1


def test_the_script_holds_no_machine_path_remote_or_key():
    """Every machine path, remote and destination is a parameter: none is written into the script."""
    text = (ROOT / "scripts" / "trajectories.py").read_text(encoding="utf-8")
    assert not re.search(r"/home/(?!user\b)|/data/|/mnt/|root_folder_id|\.config/rclone", text)
    assert not re.search(r"\b[a-z0-9_]+_drive:|\bgdrive:", text)
    for pattern in TR.TOKEN_RX.values():
        assert not pattern.search(text.encode())
