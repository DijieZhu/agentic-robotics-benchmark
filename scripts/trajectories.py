#!/usr/bin/env python3
"""Agent trajectories of `ale run` episodes: list them, keep one per task and agent, export, scan and upload them.

    python3 scripts/trajectories.py inventory --runs-dir RUNS --out inventory.jsonl
    python3 scripts/trajectories.py curate inventory.jsonl --out index.csv
    python3 scripts/trajectories.py export --runs-dir RUNS --index index.csv --archive ARCHIVE
    python3 scripts/trajectories.py scan ARCHIVE --secrets-from vendor/ale/.env
    python3 scripts/trajectories.py upload --archive ARCHIVE --dest REMOTE:PATH --secrets-from vendor/ale/.env
    python3 scripts/trajectories.py sample --run-dir RUNS/RUN_ID --parent PID        # beside `ale run`

inventory  every model episode under a runs directory, one JSON line each (validation runs and the engine's own
           oracle and nop harnesses are left out).
curate     one trajectory per task and agent: the task's latest version (content digest) and, of that version, the
           lowest-scoring episode; an episode lost to infrastructure only when no clean one has a score. Writes
           index.csv.
export     each kept episode's folder as the engine wrote it, plus run.json (summary, score, tokens and cost, the
           task text, a per-step digest), trajectory.jsonl.gz, steps.bin (the pictures the agent saw), poster.jpg,
           replays when given, and the root index (index.json, index/<platform>.json, index.csv).
scan       configured secrets and token patterns in any folder; exit status 1 on a hit.
upload     rclone copy of the archive to any rclone destination, scanned first and checked after; deletes nothing
           unless --prune.
sample     resource samples of one run's own episode containers (docker stats, nvidia-smi) into
           <run dir>/resources.jsonl; export turns them into the peak_* fields.

Every machine path, remote and run directory is a parameter; the environment variables ARB_RUNS_DIR,
ARB_TRAJECTORIES, ARB_TRAJECTORIES_DEST and ARB_SECRET_FILES give defaults. Stdlib only; Pillow, ffmpeg and rclone
are used when present. Field guide: skills/agent-trajectory-archiving/SKILL.md.
"""

from __future__ import annotations

import argparse
import base64
import collections
import csv
import datetime
import gzip
import hashlib
import io
import json
import os
import re
import shlex
import shutil
import signal
import struct
import subprocess
import sys
import tempfile
import time
import traceback
import xml.etree.ElementTree as ET
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10: task.toml is read with a regular expression
    tomllib = None

REPO = Path(__file__).resolve().parent.parent
TOOL = "scripts/trajectories.py"
FORMAT = 3  # the version of run.json's layout
INDEX_VERSION = 3  # the version of index.json's layout
ENV_RUNS, ENV_ARCHIVE, ENV_DEST, ENV_SECRETS = (
    "ARB_RUNS_DIR",
    "ARB_TRAJECTORIES",
    "ARB_TRAJECTORIES_DEST",
    "ARB_SECRET_FILES",
)
#: The engine's own harnesses: `ale validate` runs them (the oracle and the untouched sandbox); no model acts.
BUILTIN_HARNESSES = frozenset({"oracle", "nop"})
#: Episode outcomes the agent or its limits decided (not infrastructure).
CLEAN_STATUSES = frozenset({"completed", "timeout", "budget_exceeded", "refused"})
INDEX_FIELDS = [
    "task",
    "agent",
    "score",
    "status",
    "run_date",
    "task_commit",
    "run",
    "episode",
    "regrade",
    "notes",
    "folder",
]

#: List prices, USD per million tokens: Anthropic's first-party prices of October 2026 (input, output, cache read;
#: cache writes 1.25x input for the 5-minute TTL, 2x for one hour). Partner platforms (Amazon Bedrock, Vertex AI)
#: bill separately. --prices FILE replaces or adds entries (JSON, the same shape).
PRICES = {
    "claude-fable-5-1": {
        "input": 10.00,
        "output": 50.00,
        "cache_read": 0.25,
        "cache_write_5m": 12.50,
        "cache_write_1h": 20.00,
    },
    "claude-fable-5": {
        "input": 10.00,
        "output": 50.00,
        "cache_read": 1.00,
        "cache_write_5m": 12.50,
        "cache_write_1h": 20.00,
    },
    "claude-opus-5-5": {
        "input": 4.00,
        "output": 20.00,
        "cache_read": 0.20,
        "cache_write_5m": 5.00,
        "cache_write_1h": 8.00,
    },
    "claude-opus-5": {
        "input": 5.00,
        "output": 25.00,
        "cache_read": 0.50,
        "cache_write_5m": 6.25,
        "cache_write_1h": 10.00,
    },
    "claude-opus-4-8": {
        "input": 5.00,
        "output": 25.00,
        "cache_read": 0.50,
        "cache_write_5m": 6.25,
        "cache_write_1h": 10.00,
    },
    "claude-sonnet-5-5": {
        "input": 2.00,
        "output": 10.00,
        "cache_read": 0.20,
        "cache_write_5m": 2.50,
        "cache_write_1h": 4.00,
    },
    "claude-sonnet-5": {
        "input": 2.00,
        "output": 10.00,
        "cache_read": 0.20,
        "cache_write_5m": 2.50,
        "cache_write_1h": 4.00,
    },
}
PRICE_SOURCE = (
    "list price per token type times the transport trace's tokens (Anthropic first-party prices of "
    "October 2026 unless --prices replaced them); cache writes at the 5-minute rate, as the trace does not "
    "split them by TTL; partner platforms such as Amazon Bedrock bill separately and may differ"
)

#: "robot": the task's own simulator, robot, CAD or practice tools. A task's MCP servers always count (their viewer
#: tools are "look"); a shell command counts when it runs an interpreter on code that names one of these, or on a
#: script the agent wrote that does. A task with no entry here uses the names in its environment/setup/payload/;
#: --robot TASK=REGEX or the extras file's "robot" replace an entry.
ROBOT = {
    "aerial_search_geolocation": r"sim_client|sar_protocol|practice_grader|aerialsim",
    "arm_hidden_mechanism_box": r"box_env|BoxEnv|labclient|lab_bridge|wire_skeleton|practice_grader|ale-lab",
    "bimanual_cad_design_from_photos": r"freecad|FreeCAD",
    "bimanual_requirements_replica_from_photos": r"schema_check|grader_public",
    "dexhand_shape_sorter_all_pieces": r"\bsorter\b|check_plan|step_waypoint|\bWorld\(",
    "driving_safety_supervisor": r"sim_client|practice_grader|\bcarla\b|/home/user/drive|~/drive",
    "fault_adaptive_spacecraft_docking": None,  # the task gives a model description only
    "furniture_assembly_from_drawings": r"furniture_client|connect_practice|get_camera|\bCell\(",
    "hazard_bay_build": r"isaac|Isaac|SimulationApp|\bomni\.|check_structure|python\.sh",
    "hot3d_sharpa_retarget": r"flashdex|replay_hot3d|isaac|practice_grader",
    "object_finding_and_organizing_on_messy_desk": (
        r"messy_desk_env|/opt/isaac|isaac|practice_grader|example_controller"
    ),
    "robocasa_object_dynamics_from_video": r"rphys|rtasks|\brio\b|mujoco",
    "robotic_arm_cad_design_from_photos": r"freecad|FreeCAD",
    "robotic_arm_full_assembly_openarm_panda": r"assembly_openarm|assembly_cell|PairCell|connect_practice|\brender\b",
    "robotic_arm_full_assembly_so101_panda": r"assembly_info|assembly_cell|assembly_six|connect_practice|\brender\b",
    "robotic_arm_requirements_replica_from_photos": r"schema_check|grader_public",
    "robotic_arm_robot_description_from_photos": r"check_submission|mujoco",
    "shoe_lacing_and_boxing": r"shoecell|practice/grade|LocalCell|connect_practice|get_camera",
    "visual_low_level_flight_control": r"robot/client|visual_robotics|practice",
}
PAYLOAD_STOP = {
    "readme",
    "data",
    "docs",
    "example",
    "examples",
    "assets",
    "photos",
    "images",
    "models",
    "config",
    "configs",
    "tests",
    "src",
    "lib",
    "utils",
    "util",
    "main",
    "common",
    "shared",
    "grade",
    "setup",
    "init",
    "__init__",
    "requirements",
    "submission",
}

SHELL_TOOLS = {
    "Bash",
    "BashOutput",
    "KillShell",
    "KillBash",
    "TaskOutput",
    "TaskStop",
    "shell",
    "exec_command",
    "run_shell_command",
}
EDIT_TOOLS = {"Write", "Edit", "MultiEdit", "NotebookEdit", "apply_patch", "write_file", "edit_file", "replace"}
READ_TOOLS = {"Read", "Grep", "Glob", "LS", "NotebookRead", "read_file", "list_directory", "search_file_content"}
WEB_TOOLS = {"WebFetch", "WebSearch", "web_fetch", "web_search"}
IMG_EXT = (".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tif", ".tiff")
MEDIA_EXT = set(IMG_EXT) | {".mp4", ".webm", ".mkv", ".avi", ".mov", ".wav", ".mp3", ".flac", ".ogg", ".aac"}
PTR = re.compile(r"^\[large text retained at (blobs/[^\]]+)\]$")  # the engine's pointer to a large text blob
EXIT = re.compile(r"^(?:Error: )?Exit code (-?\d+)")
HEREDOC = re.compile(
    r'(?:cat|tee)\s*>{1,2}\s*([^\s<>;&|]+)\s*<<-?\s*[\'"]?(\w+)[\'"]?[^\n]*\n(.*?)\n\s*\2\b', re.DOTALL
)
CODE_CAP, OUT_CAP = 6000, 3000
REGRADE_COPY_LIMIT = 512 << 20  # a regrade record outside the episode is copied with its folder up to this size
DERIVED = (
    "run.json",
    "trajectory.jsonl.gz",
    "steps.bin",
    "pics",
    "resources.jsonl",
    "poster.jpg",
    "poster.png",
    "poster.gif",
    "poster.webp",
    "regrade",
)

RESOURCE_FIELDS = [
    "resource_enforced",
    "limits_violation_observed",
    "resource_monitor_missing",
    "peak_gpu_memory_gb",
    "peak_memory_gb",
    "peak_processes",
    "peak_disk_gb",
    "resource_emergency_kills",
    "resource_oom_kills",
    "resource_host_oom_kills",
    "resource_service_oom_kills",
]
RESOURCE_REASON = (
    "not recorded for this run: the engine records no resource peaks, and no sampler ran beside it "
    "(`trajectories.py sample` writes <run dir>/resources.jsonl)"
)
SCORE_RULE = (
    "ratio = clamp(measured / anchor, 0, cap) (inverted for direction lower); reward = clamp(ratio / full_at, "
    "0, 1), rounded half-up to two decimals; success = 1 when ratio >= full_at (reward 1.0)"
)
KIND_RULES = (
    "robot: the task's own simulator, robot, CAD or practice tools (its MCP servers; shell commands that run "
    "an interpreter on them, or on a script the agent wrote that uses them); look: a picture read or a viewer "
    "tool; edit: file writes and edits; read: reads and searches of text; shell: other shell commands; web: "
    "web tools; other: task lists, subagents and the rest"
)
SUBMISSION_NOTE = (
    "the same submission as this episode's (artifacts/), replayed on development episodes by the task's "
    "own episode runner; never a hidden (graded) episode, so it shows how the submission behaves, not "
    "the graded episodes themselves"
)


# ------------------------------------------------------------------------------------------------ small helpers
def load_json(path, default=None):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


def read_jsonl(path):
    out = []
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        out.append(json.loads(line))
                    except ValueError:
                        pass  # a torn last line: the engine reports it, we skip it
    except OSError:
        pass
    return out


def epoch(ts):
    if not ts:
        return None
    try:
        return datetime.datetime.fromisoformat(str(ts).replace("Z", "+00:00")).timestamp()  # noqa: FURB162 (3.10)
    except ValueError:
        return None


def now_utc():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")  # noqa: UP017 (3.10)


def cap(text, n, keep_tail=None):
    """Middle elision: head, '... (N characters left out) ...', tail."""
    if text is None:
        return ""
    if len(text) <= n:
        return text
    tail = keep_tail if keep_tail is not None else n // 4
    head = n - tail
    left_out = format(len(text) - head - tail, ",")
    return f"{text[:head]}\n… ({left_out} characters left out) …\n{text[-tail:]}"


def half_up(x, places=2):
    if x is None:
        return None
    return float(Decimal(repr(float(x))).quantize(Decimal(1).scaleb(-places), rounding=ROUND_HALF_UP))


def is_number(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def primary_reward(rewards):
    """The episode's reward from result.json's rewards: "reward" (older engines named several, "overall" first)."""
    rewards = rewards or {}
    return rewards.get("reward", rewards.get("overall"))


def sniff(data, declared=None):
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if data[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return declared or "application/octet-stream"


EXT = {"image/png": ".png", "image/jpeg": ".jpg", "image/gif": ".gif", "image/webp": ".webp"}


def image_size(data):
    """(width, height) of a PNG, GIF, JPEG or WebP picture, else (None, None)."""
    try:
        if data[:8] == b"\x89PNG\r\n\x1a\n" and data[12:16] == b"IHDR":
            return struct.unpack(">II", data[16:24])
        if data[:6] in (b"GIF87a", b"GIF89a"):
            return struct.unpack("<HH", data[6:10])
        if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
            chunk = data[12:16]
            if chunk == b"VP8X":
                return int.from_bytes(data[24:27], "little") + 1, int.from_bytes(data[27:30], "little") + 1
            if chunk == b"VP8L":
                b = data[21:25]
                return 1 + (((b[1] & 0x3F) << 8) | b[0]), 1 + (((b[3] & 0x0F) << 10) | (b[2] << 2) | (b[1] >> 6))
            if chunk == b"VP8 ":
                w, h = struct.unpack("<HH", data[26:30])
                return w & 0x3FFF, h & 0x3FFF
        if data[:3] == b"\xff\xd8\xff":
            i = 2
            while i + 9 < len(data):
                if data[i] != 0xFF:
                    i += 1
                    continue
                marker = data[i + 1]
                if marker in (0xD8, 0x01, 0xFF) or 0xD0 <= marker <= 0xD7:
                    i += 1 if marker == 0xFF else 2
                    continue
                if marker in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF):
                    h, w = struct.unpack(">HH", data[i + 5 : i + 9])
                    return w, h
                i += 2 + struct.unpack(">H", data[i + 2 : i + 4])[0]
    except (struct.error, IndexError):
        pass
    return None, None


def git(repo, *args):
    try:
        r = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, timeout=120, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return r.stdout if r.returncode == 0 else None


def safe_rel(rel):
    """A relative path inside the archive: no absolute path, no '..', no empty part."""
    parts = Path(rel).parts
    if not rel or Path(rel).is_absolute() or any(p in ("", ".", "..") for p in parts):
        raise ValueError(f"unsafe folder name {rel!r}")
    return Path(*parts)


def du(path):
    total = 0
    for dp, _, fs in os.walk(path):
        for f in fs:
            try:
                total += os.lstat(os.path.join(dp, f)).st_size
            except OSError:
                pass
    return total


def need(value, flag, env):
    if not value:
        raise SystemExit(f"give {flag} or set {env}")
    return value


# ------------------------------------------------------------------------------------------------ inventory
def inventory_rows(runs_dir, include_all=False):
    """Every episode folder under runs_dir/<run>/: the engine writes one per episode (and archives replaced
    attempts inside it, under attempts/)."""
    runs_dir = Path(runs_dir)
    for run_dir in sorted(p for p in runs_dir.iterdir() if p.is_dir()):
        validation = (run_dir / "validation.json").exists()
        for ep in sorted(p for p in run_dir.iterdir() if p.is_dir()):
            if not any((ep / f).exists() for f in ("result.json", "lock.json", "trajectory.json")):
                continue
            lock, res = load_json(ep / "lock.json", {}) or {}, load_json(ep / "result.json", {}) or {}
            ver = load_json(ep / "verification.json", {}) or {}
            ag, task = lock.get("agent") or {}, lock.get("task") or {}
            src = task.get("source") or {}
            skip = None
            if validation:
                skip = "validation run"
            elif ag.get("harness") in BUILTIN_HARNESSES:
                skip = "the engine's {} harness".format(ag["harness"])
            elif not ag.get("model"):
                skip = "no model recorded"
            if skip and not include_all:
                continue
            reward = primary_reward(res.get("rewards"))
            if reward is None:
                reward = ver.get("reward")
            row = {
                "run": run_dir.name,
                "episode": ep.name,
                "dir": str(ep.resolve()),
                "task": task.get("name"),
                "commit": src.get("commit"),
                "path": src.get("path"),
                "digest": task.get("content_digest"),
                "harness": ag.get("harness"),
                "model": ag.get("model"),
                "effort": (ag.get("settings") or {}).get("effort"),
                "family": ag.get("family"),
                "status": res.get("status"),
                "reward": reward,
                "started": res.get("started_at"),
                "finished": res.get("finished_at"),
                "engine": (lock.get("framework") or {}).get("commit"),
                "files": sorted(os.listdir(ep)),
                "mb": round(du(ep) / 1e6, 1),
            }
            if skip:
                row["skip"] = skip
            yield row


def cmd_inventory(a):
    runs = Path(need(a.runs_dir, "--runs-dir", ENV_RUNS))
    lines = [json.dumps(row, ensure_ascii=False) + "\n" for row in inventory_rows(runs, a.all)]
    if a.out == "-":
        sys.stdout.writelines(lines)
    else:
        with open(a.out, "w", encoding="utf-8") as f:
            f.writelines(lines)
    print(f"{len(lines)} episode(s) from {runs}", file=sys.stderr)
    return 0


# ------------------------------------------------------------------------------------------------ curate
def short_name(task):
    return str(task or "").split("/")[-1]


def component(x):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(x))


def agent_dir(r):
    """<harness>_<model>_effort-<effort>, one path component."""
    return "{}_{}_effort-{}".format(
        component(r.get("harness") or "agent"), component(r.get("model")), component(r.get("effort") or "default")
    )


def official_tasks(tasks_dir):
    return {p.parent.name for p in Path(tasks_dir).glob("*/*/task.toml")}


def regrade_score(entry, episode_dir):
    """A regrade (the same submission scored again): its score, and the path of its record."""
    path = entry.get("verification")
    if path and not Path(path).is_absolute() and episode_dir:
        path = str(Path(episode_dir) / path)
    if is_number(entry.get("score")):
        return float(entry["score"]), path
    rec = load_json(path, {}) if path else {}
    rec = rec or {}
    for v in (
        (rec.get("metrics") or {}).get("reward_exact"),
        rec.get("reward"),
        (rec.get("rewards") or {}).get("reward"),
    ):
        if is_number(v):
            return float(v), path
    raise SystemExit(f"regrade {entry!r}: no score given and none readable from {path}")


def curate(rows, official=None, rename=None, exclude=(), regrades=None, notes=None, pick="lowest"):
    """index.csv rows: one per task x agent (see the module docstring)."""
    rename, regrades, notes, exclude = rename or {}, regrades or {}, notes or {}, set(exclude)
    eps = [r for r in rows if not r.get("skip") and r.get("model") and r.get("status") and r["run"] not in exclude]
    regraded = {}  # (run, episode) -> (score, record path, the overrides entry)
    for r in eps:
        entry = regrades.get("{}/{}".format(r["run"], r["episode"])) or regrades.get(r["run"])
        if entry:
            regraded[(r["run"], r["episode"])] = regrade_score(entry, r.get("dir")) + (entry,)

    def cur(r):
        return rename.get(short_name(r["task"]), short_name(r["task"]))

    def score(r):
        if (r["run"], r["episode"]) in regraded:
            return regraded[(r["run"], r["episode"])][0]
        return float(r["reward"]) if is_number(r.get("reward")) else None

    def started(r):
        return epoch(r.get("started")) or 0.0

    by = collections.defaultdict(list)
    for r in eps:
        if official is None or cur(r) in official:
            by[cur(r)].append(r)
    keep = []
    for task in sorted(by):
        v = by[task]
        last = max(v, key=started)
        same = [r for r in v if r.get("digest") == last.get("digest")]
        for a in sorted({agent_dir(r) for r in same}):
            cand = [r for r in same if agent_dir(r) == a]
            scored = [r for r in cand if score(r) is not None and r["status"] in CLEAN_STATUSES] or [
                r for r in cand if score(r) is not None
            ]
            if pick == "latest" or not scored:
                chosen = max(cand, key=started)
            elif pick == "lowest":
                chosen = min(scored, key=lambda r: (score(r), -started(r)))
            else:
                chosen = max(scored, key=lambda r: (score(r), started(r)))
            keep.append(chosen)
    out = []
    for r in sorted(keep, key=lambda r: (cur(r), agent_dir(r))):
        name, new = short_name(r["task"]), cur(r)
        n, regrade_path = [], ""
        if (r["run"], r["episode"]) in regraded:
            entry = regraded[(r["run"], r["episode"])][2]
            regrade_path = entry.get("verification") or ""
            n.append(entry.get("note") or "regraded: the same submission scored again")
        if name != new:
            n.append("measured under its earlier name " + name)
        note = notes.get("{}/{}".format(r["run"], r["episode"])) or notes.get(r["run"])
        if note:
            n.append(note)
        s = score(r)
        episode = r["episode"].replace(name, new) if name != new else r["episode"]
        out.append(
            {
                "task": new,
                "agent": agent_dir(r),
                "score": round(s, 4) if s is not None else "",
                "status": r["status"],
                "run_date": str(r.get("started") or "")[:10],
                "task_commit": r.get("commit") or "",
                "run": r["run"],
                "episode": r["episode"],
                "regrade": regrade_path,
                "notes": " | ".join(n),
                "folder": "{}/{}/{}__{}".format(new, agent_dir(r), r["run"], episode),
            }
        )
    return out


def read_index(path):
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in INDEX_FIELDS:
            r.setdefault(k, "")
        safe_rel(r["folder"])
    return rows


def write_index(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=INDEX_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def pairs(items, flag):
    out = {}
    for it in items:
        k, sep, v = it.partition("=")
        if not sep or not k or not v:
            raise SystemExit(f"{flag} expects KEY=VALUE, got {it!r}")
        out[k] = v
    return out


def cmd_curate(a):
    rows = read_jsonl(a.inventory)
    ov = load_json(a.overrides, None) if a.overrides else {}
    if not isinstance(ov, dict):
        raise SystemExit(f"cannot read the overrides file {a.overrides} (a JSON object)")
    rename = dict(ov.get("rename") or {}, **pairs(a.rename, "--rename"))
    exclude = set(ov.get("exclude_runs") or []) | set(a.exclude_run)
    official = None if a.any_task else official_tasks(a.tasks_dir) | set(ov.get("tasks") or []) | set(a.task)
    if official is not None and not official:
        raise SystemExit(f"no task folders under {a.tasks_dir}: give --tasks-dir, --task or --any-task")
    out = curate(rows, official, rename, exclude, ov.get("regrades") or {}, ov.get("notes") or {}, a.pick)
    write_index(a.out, out)
    print(f"{len(out)} trajectory(ies) kept (one per task x agent) -> {a.out}")
    for r in out:
        print(f"  {r['task']:<48} {r['agent']:<44} {r['score']!s:<8} {r['status']!s:<10} {r['run']}")
    return 0


# ------------------------------------------------------------------------------------------------ task metadata
class Tasks:
    """The task folders the records point at: the commit a run measured (git), else the working tree, else any
    branch of the repository that carries the task."""

    def __init__(self, repo):
        self.repo = Path(repo)
        self._tree = None

    def platform(self, task, lock):
        src = (lock.get("task") or {}).get("source") or {}
        parts = Path(src.get("path") or "").parts
        if src.get("kind") == "git" and len(parts) == 3 and parts[0] == "tasks":
            return parts[1]
        found = sorted(p.parent.parent.name for p in self.repo.glob(f"tasks/*/{task}/task.toml"))
        if found:
            return found[0]
        return self.branches().get(task, (None, None))[0]

    def branches(self):
        if self._tree is None:
            self._tree = {}
            refs = ["HEAD", "origin/main"] + (
                git(self.repo, "for-each-ref", "--format=%(refname:short)", "refs/heads", "refs/remotes") or ""
            ).split()
            for ref in refs:
                for line in (git(self.repo, "ls-tree", "-r", "--name-only", ref, "tasks") or "").splitlines():
                    m = re.match(r"^tasks/([^/]+)/([^/]+)/task\.toml$", line)
                    if m and m.group(2) not in self._tree:
                        self._tree[m.group(2)] = (m.group(1), ref)
        return self._tree

    def file(self, lock, task, platform, rel, current=True):
        """(text, where): the task file `rel` of the version that ran, else (current=True) of the current version."""
        src = (lock.get("task") or {}).get("source") or {}
        if src.get("kind") == "git" and src.get("commit") and src.get("path"):
            txt = git(self.repo, "show", "{}:{}/{}".format(src["commit"], src["path"], rel))
            if txt is not None:
                return txt, "{} at task commit {}".format(rel, src["commit"][:12])
        if src.get("kind") == "local" and src.get("path"):
            p = Path(src["path"]) / rel
            if p.is_file():
                return p.read_text(encoding="utf-8", errors="replace"), "{} of the measured copy ({})".format(
                    rel, src["path"]
                )
        if not current:
            return None, None
        if platform:
            p = self.repo / "tasks" / platform / task / rel
            if p.is_file():
                return (
                    p.read_text(encoding="utf-8", errors="replace"),
                    f"{rel} of the current version (working tree): the version that ran is not in this repository",
                )
        plat, ref = self.branches().get(task, (None, None))
        if plat:
            txt = git(self.repo, "show", f"{ref}:tasks/{plat}/{task}/{rel}")
            if txt is not None:
                return txt, f"{rel} of the version on {ref}: the version that ran is not in this repository"
        return None, None

    def anchor(self, lock, task, platform):
        txt, where = self.file(lock, task, platform, "tests/anchor.json")
        try:
            return (json.loads(txt), where) if txt else ({}, None)
        except ValueError:
            return {}, None

    def timeouts(self, lock, task, platform):
        txt, _ = self.file(lock, task, platform, "task.toml", current=False)  # only the limits of the version that ran
        out = {"agent_timeout_s": None, "verify_timeout_s": None}
        if not txt:
            return out
        if tomllib is not None:
            try:
                doc = tomllib.loads(txt)
                out["agent_timeout_s"] = (doc.get("agent") or {}).get("timeout_sec")
                out["verify_timeout_s"] = (doc.get("verifier") or {}).get("timeout_sec")
                return out
            except tomllib.TOMLDecodeError:
                pass
        for sec, key in (("agent", "agent_timeout_s"), ("verifier", "verify_timeout_s")):
            m = re.search(rf"^\[{sec}\]\s*$(.*?)(?=^\[|\Z)", txt, re.MULTILINE | re.DOTALL)
            v = re.search(r"^timeout_sec\s*=\s*([0-9.]+)", m.group(1), re.MULTILINE) if m else None
            out[key] = float(v.group(1)) if v else None
        return out

    def payload_rule(self, task, platform):
        root = self.repo / "tasks" / (platform or "*") / task / "environment" / "setup" / "payload"
        roots = [root] if platform else list(self.repo.glob(f"tasks/*/{task}/environment/setup/payload"))
        names = set()
        for r in roots:
            if not r.is_dir():
                continue
            for p in r.iterdir():
                if p.is_dir():
                    names.add(p.name)
                    names.update(q.stem for q in p.iterdir() if q.suffix in (".py", ".sh"))
                elif p.suffix in (".py", ".sh"):
                    names.add(p.stem)
        names = sorted(
            n for n in names if len(n) >= 4 and n.lower() not in PAYLOAD_STOP and re.fullmatch(r"[A-Za-z0-9_.-]+", n)
        )
        return r"\b(?:{})\b".format("|".join(re.escape(n) for n in names)) if names else None


# ------------------------------------------------------------------------------------------------ model calls
def price_for(model, prices):
    m = re.sub(r"^(?:[a-z]{2,6}\.)?anthropic\.", "", str(model or ""))
    m = re.sub(r"-v\d+(?::\d+)?$", "", m)
    return prices.get(m)


def model_calls(ep, prices):
    """Every gateway call of the episode, with input_total (cache reads and writes included), cache_read,
    cache_write, output and the list-price cost."""
    calls, replays = [], 0
    for r in read_jsonl(ep / "trace.transport.jsonl"):
        if r.get("kind") == "replay":
            replays += 1
        elif r.get("kind") == "call":
            recorded = "cache_read_tokens" in r or "cache_write_tokens" in r
            c = {
                "call_id": r.get("call_id"),
                "ts": r.get("timestamp"),
                "t": epoch(r.get("timestamp")),
                "model": r.get("model"),
                "out": r.get("output_tokens") or 0,
                "latency_ms": r.get("latency_ms"),
                "stop_reason": r.get("stop_reason"),
                "disposition": r.get("disposition"),
                "purpose": r.get("purpose"),
                "response_id": r.get("provider_response_id"),
                "refusal": r.get("refusal_limit"),
                "gateway_cost_usd": r.get("cost_usd"),
                "input_total": r.get("input_tokens") or 0,
                "cache_read": r.get("cache_read_tokens") or 0,
                "cache_write": r.get("cache_write_tokens") or 0,
                "cache_source": "transport trace" if recorded else "not recorded: uncached input and output only",
            }
            c["uncached"] = max(c["input_total"] - c["cache_read"] - c["cache_write"], 0)
            p = price_for(c["model"], prices) if recorded else None  # no cache figures: no honest price
            c["usd"] = (
                (
                    (
                        c["uncached"] * p["input"]
                        + c["out"] * p["output"]
                        + c["cache_read"] * p["cache_read"]
                        + c["cache_write"] * p["cache_write_5m"]
                    )
                    / 1e6
                )
                if p
                else None
            )
            calls.append(c)
    return calls, replays


def step_tokens(cs):
    if not cs:
        return None
    usd = [c["usd"] for c in cs if c.get("usd") is not None]
    return dict(
        **{
            "in": sum(c["input_total"] for c in cs),
            "cached": sum(c["cache_read"] for c in cs),
            "out": sum(c["out"] for c in cs),
            "think": None,
            "usd": round(sum(usd), 6) if usd else None,
        },
        cache_write=sum(c["cache_write"] for c in cs),
        calls=[c["call_id"] for c in cs],
        latency_ms=sum(c["latency_ms"] or 0 for c in cs),
        stop_reason=cs[-1].get("stop_reason"),
    )


def claude_code_result(ep):
    """The last 'result' line of Claude Code's stream-json stdout (its own cost figure, usage, turns), when kept."""
    p = ep / "logs" / "claude-code" / ".ale-harness" / "stdout.log"
    res = None
    if p.is_file():
        with open(p, encoding="utf-8", errors="replace") as f:
            for line in f:
                if '"type":"result"' in line or '"type": "result"' in line:
                    try:
                        d = json.loads(line)
                    except ValueError:
                        continue
                    if d.get("type") == "result":
                        res = d
    return res


# ------------------------------------------------------------------------------------------------ pictures
class Pictures:
    """steps.bin (every distinct picture once, original bytes), and pics/ for pictures that have no image file of
    their own in the episode (kept only inside a text blob or the native log)."""

    def __init__(self, folder):
        self.folder, self.bin, self.by_sha, self.order, self.last = Path(folder), bytearray(), {}, [], None

    def add(self, data, declared, file=None, source=None, step=None):
        sha = hashlib.sha256(data).hexdigest()
        mime = sniff(data, declared)
        if sha not in self.by_sha:
            if file is None:
                file = "pics/sha256-{}{}".format(sha, EXT.get(mime, ".bin"))
                (self.folder / "pics").mkdir(exist_ok=True)
                (self.folder / file).write_bytes(data)
            w, h = image_size(data)
            rec = {
                "sha256": sha,
                "mime": mime,
                "bytes": len(data),
                "w": w,
                "h": h,
                "file": file,
                "bin": [len(self.bin), len(data)],
                "source": source,
                "first_step": step,
                "steps": [],
            }
            self.bin += data
            self.by_sha[sha] = rec
            self.order.append(rec)
        rec = self.by_sha[sha]
        if step is not None and step not in rec["steps"]:
            rec["steps"].append(step)
        self.last = (data, rec)
        return [rec["bin"][0], rec["bin"][1], rec["mime"]], rec["file"]


def embedded_image(text):
    """A picture kept as base64 inside a JSON text blob (how a harness's Read result can carry one)."""
    if not text or not text.lstrip().startswith("{") or '"base64"' not in text[:200]:
        return None
    try:
        d = json.loads(text)
    except ValueError:
        return None
    if isinstance(d, dict) and d.get("type") == "image" and isinstance(d.get("file"), dict) and d["file"].get("base64"):
        try:
            return base64.b64decode(d["file"]["base64"]), d["file"].get("type")
        except (ValueError, TypeError):
            return None
    return None


# ------------------------------------------------------------------------------------------------ call digest
def server_tool(name):
    parts = name.split("__")
    return (parts[1], parts[2]) if len(parts) >= 3 else (None, name)


def shell_command(args):
    a = args if isinstance(args, dict) else {}
    cmd = a.get("command") if a.get("command") is not None else a.get("cmd")
    if isinstance(cmd, list):
        return shlex.join(str(x) for x in cmd)
    return cmd if isinstance(cmd, str) else ""


def label_of(tool, args):
    a = args if isinstance(args, dict) else {}
    if tool in SHELL_TOOLS and (a.get("command") is not None or a.get("cmd") is not None):
        d = a.get("description") or (shell_command(a).strip().splitlines() or [tool])[0]
        return str(d)[:160]
    if tool in (
        "Read",
        "Write",
        "Edit",
        "MultiEdit",
        "NotebookEdit",
        "NotebookRead",
        "read_file",
        "write_file",
        "edit_file",
    ):
        fp = a.get("file_path") or a.get("notebook_path") or a.get("path") or ""
        lab = os.path.basename(str(fp).rstrip("/")) or str(fp) or tool
        if tool == "Read" and (a.get("offset") or a.get("limit")):
            lab += " (from line {}, {} lines)".format(a.get("offset") or 1, a.get("limit") or "all")
        return lab
    if tool == "Grep":
        return "grep {}".format(str(a.get("pattern") or "")[:120])
    if tool == "Glob":
        return "glob {}".format(str(a.get("pattern") or "")[:120])
    if tool in ("WebFetch", "web_fetch"):
        return str(a.get("url") or "")[:160]
    if tool in ("WebSearch", "web_search"):
        return str(a.get("query") or "")[:160]
    if tool.startswith("mcp__"):
        s, t = server_tool(tool)
        return f"{s}: {t}"
    for k in ("description", "subject", "prompt", "title", "task_id", "shell_id", "bash_id"):
        if a.get(k):
            return ("{}: {}".format(tool, (str(a[k]).strip().splitlines() or [""])[0]))[:160]
    return tool


def code_of(tool, args, resolve):
    a = args if isinstance(args, dict) else {}
    if tool in SHELL_TOOLS and (a.get("command") is not None or a.get("cmd") is not None):
        s = shell_command(a)
        if a.get("run_in_background"):
            s += "\n# (run in the background)"
        return s
    if tool == "Write":
        return "{}\n\n{}".format(a.get("file_path", ""), resolve(a.get("content") or ""))
    if tool == "Edit":
        return "{}\n\n--- old\n{}\n+++ new\n{}{}".format(
            a.get("file_path", ""),
            resolve(a.get("old_string") or ""),
            resolve(a.get("new_string") or ""),
            "\n(replace all)" if a.get("replace_all") else "",
        )
    if tool == "MultiEdit":
        parts = [a.get("file_path", "")]
        for e in a.get("edits") or []:
            parts.append(
                "--- old\n{}\n+++ new\n{}".format(
                    resolve(e.get("old_string") or ""), resolve(e.get("new_string") or "")
                )
            )
        return "\n\n".join(parts)
    if tool == "Read":
        s = a.get("file_path", "")
        if a.get("offset") or a.get("limit"):
            s += "  [offset {}, limit {}]".format(a.get("offset"), a.get("limit"))
        return s
    if tool.startswith("mcp__") and isinstance(a.get("code"), str):
        return a["code"]
    return json.dumps(a, ensure_ascii=False, indent=1)


INTERP = re.compile(
    r"^(?:(?:\w+=\S*|nohup|time|exec|setsid|timeout(?:\s+-\S+)*\s+\S+|xvfb-run(?:\s+-\S+)*|taskset\s+\S+\s+\S+"
    r"|stdbuf(?:\s+-\S+)+|env(?:\s+\w+=\S*)*)\s+)*(?:(?:\S*/)?(?:python3?(?:\.\d+)?|bash|sh|freecadcmd|FreeCADCmd|"
    r"python\.sh)|uv\s+run|\./\S+)(?:\s|$)"
)
SEG = re.compile(r"&&|\|\||;|\|")
DASH_C = re.compile(r'(?:^|[\s;&|(])(?:\S*/)?python3?(?:\.\d+)?\s+(?:-\S+\s+)*-c\s+(["\'])(.*?)(?<!\\)\1', re.DOTALL)


def exec_parts(cmd):
    """What a shell command runs: interpreter command lines (segment text) and code handed to an interpreter (a
    here-document after `python3 -`, a `-c` string). A here-document written to a file is not run."""
    segs, bodies = [], [m.group(2) for m in DASH_C.finditer(cmd)]
    lines, i = cmd.split("\n"), 0
    while i < len(lines):
        line = lines[i]
        m = re.search(r'<<-?\s*[\'"]?(\w+)[\'"]?', line)
        if m:
            body, j = [], i + 1
            while j < len(lines) and lines[j].strip() != m.group(1):
                body.append(lines[j])
                j += 1
            head = SEG.split(line[: m.start()])
            for seg in head[:-1]:
                if INTERP.match(seg.strip()):
                    segs.append(seg.strip())
            if INTERP.match(head[-1].strip()):
                segs.append(head[-1].strip())
                bodies.append("\n".join(body))
            i = j + 1
            continue
        for seg in SEG.split(line):
            if INTERP.match(seg.strip()):
                segs.append(seg.strip())
        i += 1
    return segs, bodies


class Classifier:
    """The kind of each tool call (robot / look / edit / read / shell / web / other)."""

    def __init__(self, rule):
        self.rule = rule
        self.rx = re.compile(rule) if rule else None
        self.scripts = set()

    def _uses(self, code):
        """Code that drives the task's tools: names one of them, or imports / execs a script that does."""
        if not self.rx or not code:
            return False
        if self.rx.search(code):
            return True
        for s in self.scripts:
            name, mod = re.escape(s), re.escape(os.path.splitext(s)[0])
            runs_it = rf"exec\(open\([^)]*{name}|runpy\.run_path\([^)]*{name}"
            imports_it = rf"^\s*(?:import\s+(?:[\w.]+\s*(?:as\s+\w+)?\s*,\s*)*{mod}\b|from\s+{mod}\s+import)"
            if re.search(f"{runs_it}|{imports_it}", code, re.MULTILINE):
                return True
        return False

    def note_writes(self, tool, args):
        """Remember scripts the agent wrote that drive the task's tools (a file write, or a here-document)."""
        if not self.rx or not isinstance(args, dict):
            return
        if tool in ("Write", "write_file") and self._uses(args.get("content") or ""):
            self.scripts.add(os.path.basename(str(args.get("file_path") or args.get("path") or "")))
        if tool in SHELL_TOOLS:
            for m in HEREDOC.finditer(shell_command(args)):
                if self._uses(m.group(3)):
                    self.scripts.add(os.path.basename(m.group(1)))
        self.scripts.discard("")

    def kind(self, tool, args, has_image):
        a = args if isinstance(args, dict) else {}
        if tool.startswith("mcp__"):
            _, t = server_tool(tool)
            return "look" if re.search(r"view|screenshot|capture|snapshot", t, re.IGNORECASE) else "robot"
        if tool in WEB_TOOLS:
            return "web"
        if tool in ("Read", "read_file"):
            fp = str(a.get("file_path") or a.get("path") or "").lower()
            return "look" if (has_image or fp.endswith(IMG_EXT)) else "read"
        if tool in READ_TOOLS:
            return "read"
        if tool in EDIT_TOOLS:
            return "edit"
        if tool in SHELL_TOOLS:
            cmd = shell_command(a)
            if self.rx and cmd:
                segs, bodies = exec_parts(cmd)
                if (
                    any(self.rx.search(x) for x in segs)
                    or any(self._uses(b) for b in bodies)
                    or any(re.search(rf"(?<![\w.-]){re.escape(sc)}\b", x) for sc in self.scripts for x in segs)
                ):
                    return "robot"
            return "shell"
        return "other"


def sim_counts(calls):
    sub = res = 0
    seen = False
    for c in calls:
        if c["kind"] == "robot" and c["tool"].startswith("mcp__"):
            _, t = server_tool(c["tool"])
            if re.search(r"reset", t, re.IGNORECASE):
                res += 1
                seen = True
            elif re.search(r"act|step|move|command|drive|control|send", t, re.IGNORECASE):
                sub += 1
                seen = True
    return {"submit": sub, "reset": res} if seen else None


def exit_code(tool, text, is_error, args):
    if tool != "Bash":
        return None
    m = EXIT.match(text or "")
    if m:
        return int(m.group(1))
    if is_error:
        return None
    if (args or {}).get("run_in_background") or (text or "").startswith("Command running in background"):
        return None
    return 0


def finish_step(st, says, texts, n_err):
    st["say"] = "\n\n".join(says)
    full = "\n\n".join(x for x in texts if x)
    st["out"]["text"] = cap(full, OUT_CAP, 900)
    st["out"]["chars"] = len(full)
    if n_err:
        st["out"]["errors"] = n_err
    st["sim"] = sim_counts(st["calls"])
    if not st["out"].get("pics"):
        st["out"].pop("pic_files", None)


def add_call(st, cls, tool, args, txt, is_err, has_result, has_img, resolve, call_id):
    kind = cls.kind(tool, args, has_img)
    cls.note_writes(tool, args)
    ec = exit_code(tool, txt, is_err, args)
    call = {
        "kind": kind,
        "label": label_of(tool, args),
        "code": cap(code_of(tool, args, resolve), CODE_CAP, 1500),
        "tool": tool,
        "id": call_id,
    }
    if ec is not None:
        call["exit"] = ec
        st["out"]["exit"].append(ec)
    if is_err:
        call["error"] = True
    if not has_result:
        call["result"] = "missing"
    st["calls"].append(call)


# ------------------------------------------------------------------------------------------------ ATIF digest
def atif_digest(ep, traj, calls_by_id, t0, cls, pics):
    """One step per model call (the ATIF steps of one call merged), from trajectory.json."""

    def blob_text(rel):
        try:
            return (ep / rel).read_bytes().decode("utf-8", errors="replace")
        except OSError:
            return ""

    def resolve(s):
        if isinstance(s, str):
            m = PTR.match(s.strip())
            if m:
                return blob_text(m.group(1))
            return s
        return json.dumps(s, ensure_ascii=False)

    def text_of(msg):
        if isinstance(msg, str):
            return resolve(msg)
        if isinstance(msg, list):
            return "\n".join(
                resolve(p.get("text") or "") for p in msg if isinstance(p, dict) and p.get("type") == "text"
            )
        return ""

    groups, cur, key = [], None, None
    for s in (traj.get("steps") or [])[1:]:  # step 1 is the instruction (task_text)
        ex = (s.get("extra") or {}).get("ale") or {}
        if s.get("source") == "agent":
            k = (
                tuple(ex.get("transport_call_ids") or [])
                or tuple((ex.get("native_event_ids") or [])[:1])
                or ("step", s.get("step_id"))
            )
            k = ("agent",) + k + (("program",) if s.get("llm_call_count") == 0 else ())
        else:
            k = (s.get("source"), s.get("step_id"))
        if cur is not None and k == key and s.get("source") == "agent":
            cur.append(s)
        else:
            cur, key = [s], k
            groups.append(cur)

    out, used, last_t = [], set(), 0.0
    for g in groups:
        first = g[0]
        ts = next((epoch(s.get("timestamp")) for s in g if s.get("timestamp")), None)
        t = round(ts - t0, 1) if (ts is not None and t0 is not None) else last_t
        last_t = t
        src = first.get("source")
        frm = ("agent" if first.get("llm_call_count") != 0 else "tool-program") if src == "agent" else src
        st = dict(
            id=first.get("step_id"),
            t=t,
            say="",
            calls=[],
            out={"text": "", "exit": [], "pics": []},
            tok=None,
            sim=None,
            atif=[s.get("step_id") for s in g],
            at=first.get("timestamp"),
            **{"from": frm},
        )
        if src != "agent":
            txt = text_of(first.get("message"))
            st["out"]["text"] = cap(txt, OUT_CAP)
            st["out"]["chars"] = len(txt)
            if txt.startswith("This session is being continued from a previous conversation"):
                st["note"] = "context compaction: the harness replaced the conversation with this summary"
            out.append(st)
            continue
        says, texts, n_err = [], [], 0
        for s in g:
            m = text_of(s.get("message"))
            if m.strip():
                says.append(m.strip())
            results = collections.defaultdict(list)
            for r in (s.get("observation") or {}).get("results") or []:
                results[r.get("source_call_id")].append(r)
            for tc in s.get("tool_calls") or []:
                tool, args = tc.get("function_name") or "?", tc.get("arguments") or {}
                rs = results.get(tc.get("tool_call_id"), [])
                has_img, rtext, is_err = False, [], False
                for r in rs:
                    is_err = is_err or bool(((r.get("extra") or {}).get("ale") or {}).get("is_error"))
                    c = r.get("content")
                    parts = c if isinstance(c, list) else ([{"type": "text", "text": c}] if isinstance(c, str) else [])
                    for p in parts:
                        if not isinstance(p, dict):
                            continue
                        if p.get("type") == "image" and (p.get("source") or {}).get("path"):
                            path = p["source"]["path"]
                            try:
                                data = (ep / path).read_bytes()
                            except OSError:
                                rtext.append(f"[picture missing: {path}]")
                                continue
                            rng, file = pics.add(
                                data, p["source"].get("media_type"), file=path, source=path, step=st["id"]
                            )
                            st["out"]["pics"].append(rng)
                            st["out"].setdefault("pic_files", []).append(file)
                            has_img = True
                            rtext.append("[picture]")
                        elif p.get("type") == "text":
                            raw = p.get("text") or ""
                            m = PTR.match(raw.strip())
                            if m:
                                full = blob_text(m.group(1))
                                emb = embedded_image(full)
                                if emb:
                                    data, mime = emb
                                    rng, file = pics.add(
                                        data, mime, source=f"{m.group(1)} (a picture as base64 in JSON)", step=st["id"]
                                    )
                                    st["out"]["pics"].append(rng)
                                    st["out"].setdefault("pic_files", []).append(file)
                                    has_img = True
                                    w, h = image_size(data)
                                    rtext.append(f"[picture {sniff(data, mime)} {w}x{h}]")
                                else:
                                    rtext.append(full)
                            else:
                                rtext.append(raw)
                txt = "\n".join(x for x in rtext if x is not None)
                add_call(st, cls, tool, args, txt, is_err, bool(rs), has_img, resolve, tc.get("tool_call_id"))
                n_err += bool(is_err)
                texts.append(txt)
        ids = []
        for s in g:
            for cid in ((s.get("extra") or {}).get("ale") or {}).get("transport_call_ids") or []:
                if cid not in ids:
                    ids.append(cid)
        st["tok"] = step_tokens([calls_by_id[c] for c in ids if c in calls_by_id and c not in used])
        used.update(ids)
        finish_step(st, says, texts, n_err)
        out.append(st)
    return out, used


# ------------------------------------------------------------------------------------------------ native digest
def native_digest(path, calls_by_resp, t0, cls, pics):
    """Claude Code's own session log, for an episode whose ATIF conversion failed (trajectory.json then holds only
    the instruction and the final message)."""
    entries = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for n, line in enumerate(f, 1):
            try:
                e = json.loads(line)
            except ValueError:
                continue
            if isinstance(e, dict):
                e["_line"] = n
                entries.append(e)
    order, msgs, results, dups, users, first_user = [], {}, {}, 0, 0, True
    for e in entries:
        if e.get("isSidechain"):
            continue
        typ = e.get("type")
        m = e.get("message") if isinstance(e.get("message"), dict) else {}
        if typ == "assistant":
            mid = m.get("id") or e.get("uuid")
            if mid not in msgs:
                msgs[mid] = {"ts": e.get("timestamp"), "blocks": [], "seen": set(), "line": e["_line"]}
                order.append(("a", mid))
            for b in m.get("content") or []:
                sig = json.dumps(b, sort_keys=True)[:4000]
                if sig in msgs[mid]["seen"]:
                    continue
                msgs[mid]["seen"].add(sig)
                msgs[mid]["blocks"].append(b)
        elif typ == "user":
            c = m.get("content")
            if isinstance(c, str) or (
                isinstance(c, list) and any(isinstance(b, dict) and b.get("type") == "text" for b in c)
            ):
                txt = (
                    c
                    if isinstance(c, str)
                    else "\n".join(b.get("text") or "" for b in c if isinstance(b, dict) and b.get("type") == "text")
                )
                if first_user:
                    first_user = False
                elif txt.strip():
                    order.append(
                        (
                            "u",
                            {
                                "ts": e.get("timestamp"),
                                "text": txt,
                                "line": e["_line"],
                                "compact": bool(e.get("isCompactSummary")),
                            },
                        )
                    )
                    users += 1
            if isinstance(c, list):
                for b in c:
                    if isinstance(b, dict) and b.get("type") == "tool_result":
                        if b.get("tool_use_id") in results:
                            dups += 1
                            continue
                        results[b.get("tool_use_id")] = (b, e["_line"])
        elif typ == "system" and e.get("subtype") == "compact_boundary":
            meta = e.get("compactMetadata") or {}
            order.append(
                (
                    "s",
                    {
                        "ts": e.get("timestamp"),
                        "line": e["_line"],
                        "text": "Conversation compacted ({}): {} tokens before, {} after".format(
                            meta.get("trigger"), meta.get("preTokens"), meta.get("postTokens")
                        ),
                    },
                )
            )
    out, used, sid, last_t = [], set(), 1, 0.0
    for kind, item in order:
        sid += 1
        if kind in ("u", "s"):
            ts = epoch(item["ts"])
            t = round(ts - t0, 1) if ts and t0 else last_t
            last_t = t
            st = dict(
                id=sid,
                t=t,
                say="",
                calls=[],
                out={"text": cap(item["text"], OUT_CAP), "exit": [], "pics": [], "chars": len(item["text"])},
                tok=None,
                sim=None,
                at=item["ts"],
                native_line=item["line"],
                **{"from": "user" if kind == "u" else "system"},
            )
            if kind == "u" and item.get("compact"):
                st["note"] = "context compaction: the harness replaced the conversation with this summary"
            out.append(st)
            continue
        msg = msgs[item]
        ts = epoch(msg["ts"])
        t = round(ts - t0, 1) if ts and t0 else last_t
        last_t = t
        st = dict(
            id=sid,
            t=t,
            say="",
            calls=[],
            out={"text": "", "exit": [], "pics": []},
            tok=None,
            sim=None,
            at=msg["ts"],
            native_message=item,
            native_line=msg["line"],
            **{"from": "agent"},
        )
        says, texts, n_err = [], [], 0
        for b in msg["blocks"]:
            if not isinstance(b, dict):
                continue
            if b.get("type") == "text" and (b.get("text") or "").strip():
                says.append(b["text"].strip())
            if b.get("type") != "tool_use":
                continue
            tool, args = b.get("name") or "?", b.get("input") or {}
            res = results.get(b.get("id"))
            has_img, rtext, is_err = False, [], False
            if res:
                rb, line = res
                is_err = bool(rb.get("is_error"))
                c = rb.get("content")
                parts = c if isinstance(c, list) else ([{"type": "text", "text": c}] if isinstance(c, str) else [])
                for p in parts:
                    if not isinstance(p, dict):
                        continue
                    if p.get("type") == "image" and (p.get("source") or {}).get("data"):
                        try:
                            data = base64.b64decode(p["source"]["data"])
                        except (ValueError, TypeError):
                            rtext.append("[picture undecodable]")
                            continue
                        rng, file = pics.add(
                            data,
                            p["source"].get("media_type"),
                            source=f"native log line {line} (tool result, base64)",
                            step=sid,
                        )
                        st["out"]["pics"].append(rng)
                        st["out"].setdefault("pic_files", []).append(file)
                        has_img = True
                        rtext.append("[picture]")
                    elif p.get("type") == "text":
                        rtext.append(p.get("text") or "")
            txt = "\n".join(rtext)
            add_call(st, cls, tool, args, txt, is_err, bool(res), has_img, lambda s: s, b.get("id"))
            n_err += bool(is_err)
            texts.append(txt)
        cs = [calls_by_resp[item]] if item in calls_by_resp else []
        st["tok"] = step_tokens(cs)
        used.update(c["call_id"] for c in cs)
        finish_step(st, says, texts, n_err)
        out.append(st)
    return out, used, {"duplicate_tool_results": dups, "user_messages": users, "lines": len(entries)}


# ------------------------------------------------------------------------------------------------ files
def write_trajectory_gz(folder, traj, native_path, ep):
    dst = folder / "trajectory.jsonl.gz"
    raw = lines = 0
    with gzip.open(dst, "wb", compresslevel=6) as gz:
        if native_path:
            with open(native_path, "rb") as f:
                for line in f:
                    line = line if line.endswith(b"\n") else line + b"\n"
                    gz.write(line)
                    raw += len(line)
                    lines += 1
            layout = (
                "the native session log, verbatim, one event per line (the engine's ATIF conversion failed, so "
                "trajectory.json holds only the instruction and the final message)"
            )
            source = os.path.relpath(native_path, ep)
        else:
            head = {k: v for k, v in traj.items() if k not in ("steps", "subagent_trajectories")}
            head["subagent_trajectories"] = len(traj.get("subagent_trajectories") or [])
            rows = (
                [{"atif_header": head}]
                + list(traj.get("steps") or [])
                + [{"subagent_trajectory": s} for s in traj.get("subagent_trajectories") or []]
            )
            for r in rows:
                b = (json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
                gz.write(b)
                raw += len(b)
                lines += 1
            layout = (
                'line 1 {"atif_header": the ATIF document without its steps}; then one line per root step (ATIF '
                "step objects verbatim; large texts and pictures stay in blobs/, referenced by path); then one "
                'line per subagent trajectory {"subagent_trajectory": ...}'
            )
            source = "trajectory.json"
    return {
        "file": "trajectory.jsonl.gz",
        "size": dst.stat().st_size,
        "bytes": raw,
        "lines": lines,
        "source": source,
        "capped": None,
        "layout": layout,
    }


def save_jpeg(data, dst_stem, max_width=1920):
    """Write a picture as <stem>.jpg (Pillow, else ffmpeg), else in its own format; returns the file name."""
    mime = sniff(data)
    if mime == "image/jpeg" and (image_size(data)[0] or 0) <= max_width:
        dst_stem.with_suffix(".jpg").write_bytes(data)
        return dst_stem.with_suffix(".jpg").name
    try:
        from PIL import Image
    except ImportError:
        Image = None
    if Image is not None:
        try:
            with Image.open(io.BytesIO(data)) as im:
                im = im.convert("RGB")
                if im.width > max_width:
                    im = im.resize((max_width, round(im.height * max_width / im.width)), Image.LANCZOS)
                im.save(dst_stem.with_suffix(".jpg"), "JPEG", quality=90)
            return dst_stem.with_suffix(".jpg").name
        except (OSError, ValueError, Image.DecompressionBombError) as e:
            print(f"poster: Pillow cannot convert the picture ({e}); trying ffmpeg", file=sys.stderr)
    if shutil.which("ffmpeg"):
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / ("in" + EXT.get(mime, ".bin"))
            src.write_bytes(data)
            r = subprocess.run(
                [
                    "ffmpeg",
                    "-v",
                    "error",
                    "-y",
                    "-i",
                    str(src),
                    "-vf",
                    f"scale=min({max_width}\\,iw):-2",
                    "-frames:v",
                    "1",
                    "-q:v",
                    "2",
                    str(dst_stem.with_suffix(".jpg")),
                ],
                capture_output=True,
                check=False,
            )
            if r.returncode == 0 and dst_stem.with_suffix(".jpg").is_file():
                return dst_stem.with_suffix(".jpg").name
    name = dst_stem.with_suffix(EXT.get(mime, ".bin"))
    name.write_bytes(data)
    return name.name


def make_poster(folder, task, pics, extras):
    for r in extras.get("replays", {}).get(task, []):
        if r.get("poster_at") is not None and shutil.which("ffmpeg"):
            dst = folder / "poster.jpg"
            rc = subprocess.run(
                [
                    "ffmpeg",
                    "-v",
                    "error",
                    "-y",
                    "-ss",
                    str(r["poster_at"]),
                    "-i",
                    str(r["local"]),
                    "-frames:v",
                    "1",
                    "-q:v",
                    "2",
                    str(dst),
                ],
                capture_output=True,
                check=False,
            )
            if rc.returncode == 0 and dst.is_file():
                return {
                    "file": "poster.jpg",
                    "source": "{} at {:.2f} s".format(r["file"], float(r["poster_at"])),
                    "what": "the replay's frame at its most telling moment",
                }
    if pics.last:
        data, rec = pics.last
        name = save_jpeg(data, folder / "poster")
        return {
            "file": name,
            "source": rec["file"],
            "what": "the last picture the agent saw (step {})".format(rec["steps"][-1]),
        }
    still = extras.get("stills", {}).get(task)
    if still:
        name = save_jpeg(Path(still["local"]).read_bytes(), folder / "poster")
        return {
            "file": name,
            "source": still.get("source") or os.path.basename(still["local"]),
            "what": still.get("what") or "a still of the task, not from this run",
        }
    return None


def copy_episode(src, dst, link=False):
    """The episode folder as the engine wrote it (symlinks kept as links, never followed; no __pycache__)."""

    def copy(s, d):
        if link:
            try:
                os.link(s, d)
                return d
            except OSError:
                pass
        return shutil.copy2(s, d)

    shutil.copytree(src, dst, symlinks=True, copy_function=copy, ignore=shutil.ignore_patterns("__pycache__"))


# ------------------------------------------------------------------------------------------------ secrets
TOKEN_PATTERNS = {
    "aws_access_key_id": rb"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b",
    "aws_secret_access_key": rb"aws_secret_access_key\s*[=:]\s*\S{20,}",
    "anthropic_key": rb"sk-ant-[A-Za-z0-9_-]{20,}",
    "openai_key": rb"\bsk-(?!ant-)(?:proj-|svcacct-)?[A-Za-z0-9_-]{32,}",
    "github_token": rb"\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,})",
    "huggingface_token": rb"\bhf_[A-Za-z0-9]{30,}",
    "google_api_key": rb"\bAIza[0-9A-Za-z_-]{35}",
    "google_oauth_token": rb"\bya29\.[0-9A-Za-z_-]{20,}",
    "slack_token": rb"\bxox[abprs]-[0-9A-Za-z-]{10,}",
    "private_key": rb"-----BEGIN [A-Z ]*PRIVATE KEY-----",
}
TOKEN_RX = {k: re.compile(v) for k, v in TOKEN_PATTERNS.items()}
SECRET_NAME = re.compile(r"(?i)key|token|secret|passw|credential|auth")
CHUNK, OVERLAP = 16 << 20, 4096


def configured_secrets(files=(), env_names=()):
    """Values to search for: in dotenv / ini / rclone.conf files, every value whose key names a key, token, secret,
    password or credential (and the string values of such keys inside a JSON value); and named environment
    variables. Only values of 16 characters or more."""
    found = set()
    for f in files:
        try:
            text = Path(f).expanduser().read_text(encoding="utf-8", errors="replace")
        except OSError as e:
            raise SystemExit(f"cannot read the secrets file {f}: {e}")
        for line in text.splitlines():
            if line.lstrip().startswith(("#", ";", "[")):
                continue
            m = re.match(r"^\s*(?:export\s+)?([A-Za-z0-9_.-]+)\s*[=:]\s*(.*?)\s*$", line)
            if not m or not SECRET_NAME.search(m.group(1)):
                continue
            value = m.group(2).strip().strip("\"'")
            if value.startswith("{"):
                try:
                    stack = [json.loads(value)]
                except ValueError:
                    stack = []
                while stack:
                    d = stack.pop()
                    if isinstance(d, dict):
                        for k, v in d.items():
                            if isinstance(v, str) and SECRET_NAME.search(k) and len(v) >= 16:
                                found.add(v)
                            elif isinstance(v, (dict, list)):
                                stack.append(v)
                    elif isinstance(d, list):
                        stack.extend(d)
            if len(value) >= 16:
                found.add(value)
    for name in env_names:
        v = os.environ.get(name) or ""
        if len(v) >= 16:
            found.add(v)
    return sorted(found)


def secret_files(a):
    files = list(a.secrets_from)
    files += [p for p in os.environ.get(ENV_SECRETS, "").split(os.pathsep) if p]
    return files


def iter_files(root, include_media):
    root = Path(root)
    if root.is_file():
        yield root
        return
    for dp, dirs, fs in os.walk(root):
        dirs[:] = [d for d in dirs if not os.path.islink(os.path.join(dp, d))]
        for f in sorted(fs):
            p = Path(dp) / f
            if p.is_symlink():
                continue
            if not include_media and (p.suffix.lower() in MEDIA_EXT or p.name == "steps.bin"):
                continue
            yield p


def chunks(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as f:
        prev, pos = b"", 0
        while True:
            block = f.read(CHUNK)
            if not block:
                break
            yield pos - len(prev), prev + block
            pos += len(block)
            prev = block[-OVERLAP:]


def redact(window, secrets):
    for s in secrets:
        window = window.replace(s, b"<redacted>")
    for k, rx in TOKEN_RX.items():
        window = rx.sub(b"<redacted:" + k.encode() + b">", window)
    text = window.decode("utf-8", errors="replace")
    return "".join(ch if ch.isprintable() else "." for ch in text)


def scan_paths(paths, secrets, include_media=False, context=0):
    """[(path, kind, offset, context)]: one per file and kind (the first offset)."""
    fixed = [s.encode() for s in secrets]
    hits = []
    for root in paths:
        for p in iter_files(root, include_media):
            seen = set()
            try:
                for base, data in chunks(p):
                    found = [(f"configured-secret-{i}", data.find(s), len(s)) for i, s in enumerate(fixed)]
                    found = [f for f in found if f[1] >= 0]
                    for k, rx in TOKEN_RX.items():
                        m = rx.search(data)
                        if m:
                            found.append((k, m.start(), m.end() - m.start()))
                    for kind, off, n in found:
                        if kind in seen:
                            continue
                        seen.add(kind)
                        ctx = redact(data[max(0, off - context) : off + n + context], fixed) if context else ""
                        hits.append((str(p), kind, base + off, ctx))
            except (OSError, EOFError, gzip.BadGzipFile) as e:
                hits.append((str(p), f"unreadable: {e}", 0, ""))
    return hits


def cmd_scan(a):
    secrets = configured_secrets(secret_files(a), a.secret_env)
    hits = scan_paths(a.paths, secrets, a.include_media, a.context)
    for path, kind, off, ctx in hits:
        print(f"HIT {path} {kind} at byte {off}")
        if ctx:
            print("    " + ctx)
    print(f"{len(hits)} hit(s); {len(secrets)} configured secret(s) searched for, {len(TOKEN_RX)} token pattern(s)")
    return 1 if hits else 0


# ------------------------------------------------------------------------------------------------ resources
def resource_peaks(ep, run_dir, folder):
    """Peaks of the episode's own containers from <run dir>/resources.jsonl (`trajectories.py sample`)."""
    p = Path(run_dir) / "resources.jsonl"
    rows = [r for r in read_jsonl(p) if r.get("episode") == ep.name]
    if not rows:
        return None
    with open(folder / "resources.jsonl", "w", encoding="utf-8") as f:
        f.write("".join(json.dumps(r, separators=(",", ":")) + "\n" for r in rows))
    pk = {"peak_memory_gb": 0.0, "peak_processes": 0, "peak_gpu_memory_gb": 0.0, "peak_disk_gb": None}
    for r in rows:
        for c in r.get("containers") or []:
            pk["peak_memory_gb"] = max(pk["peak_memory_gb"], (c.get("mem_bytes") or 0) / 1e9)
            pk["peak_processes"] = max(pk["peak_processes"], c.get("pids") or 0)
            if c.get("rw_bytes") is not None:
                pk["peak_disk_gb"] = max(pk["peak_disk_gb"] or 0.0, c["rw_bytes"] / 1e9)
        pk["peak_gpu_memory_gb"] = max(pk["peak_gpu_memory_gb"], (r.get("gpu_mem_bytes") or 0) / 1e9)
    return {k: (round(v, 3) if isinstance(v, float) else v) for k, v in pk.items()}, len(rows)


UNITS = {
    "b": 1,
    "kb": 1e3,
    "mb": 1e6,
    "gb": 1e9,
    "tb": 1e12,
    "kib": 1024,
    "mib": 1024**2,
    "gib": 1024**3,
    "tib": 1024**4,
}


def to_bytes(s):
    """'1.5GiB', '512MiB', '10.2kB', '0B', '1.23GB (virtual 20GB)' -> bytes (the first quantity)."""
    m = re.match(r"\s*([0-9.]+)\s*([A-Za-z]*)", s or "")
    if not m:
        return None
    try:
        return int(float(m.group(1)) * UNITS.get((m.group(2) or "B").lower(), 1))
    except ValueError:
        return None


def _pair(s):
    a, _, b = (s or "").partition("/")
    return to_bytes(a), to_bytes(b)


def _run(argv, timeout=60):
    try:
        r = subprocess.run(argv, capture_output=True, text=True, timeout=timeout, check=False)
        return r.stdout if r.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def own_containers(episodes, managed):
    """Running engine containers whose ale.episode label is one of this run's episode folders."""
    out = _run(
        [
            "docker",
            "ps",
            "--no-trunc",
            "--filter",
            "label=" + managed,
            "--format",
            '{{.ID}}\t{{.Names}}\t{{.Label "ale.episode"}}\t{{.Label "ale.role"}}\t{{.Label "ale.gpus"}}',
        ]
    )
    rows = []
    for line in (out or "").splitlines():
        parts = line.split("\t")
        if len(parts) >= 5 and parts[2] in episodes:
            rows.append(
                {
                    "id": parts[0],
                    "name": parts[1],
                    "episode": parts[2],
                    "role": parts[3],
                    "gpus": [g for g in parts[4].split(",") if g],
                }
            )
    return rows


def docker_stats(ids):
    out = _run(["docker", "stats", "--no-stream", "--no-trunc", "--format", "{{json .}}"] + ids, timeout=90)
    res = {}
    for line in (out or "").splitlines():
        try:
            d = json.loads(line)
        except ValueError:
            continue
        mem, lim = _pair(d.get("MemUsage"))
        rx, tx = _pair(d.get("NetIO"))
        br, bw = _pair(d.get("BlockIO"))
        try:
            pids = int(d.get("PIDs") or 0)
        except ValueError:
            pids = None
        try:
            cpu = float(str(d.get("CPUPerc") or "0").rstrip("%"))
        except ValueError:
            cpu = None
        res[d.get("ID") or d.get("Container")] = {
            "cpu_pct": cpu,
            "mem_bytes": mem,
            "mem_limit_bytes": lim,
            "pids": pids,
            "net_rx_bytes": rx,
            "net_tx_bytes": tx,
            "block_read_bytes": br,
            "block_write_bytes": bw,
        }
    return res


def rw_sizes(ids):
    argv = ["docker", "ps", "--no-trunc", "--size", "--format", "{{.ID}}\t{{.Size}}"]
    for i in ids:
        argv += ["--filter", "id=" + i]
    res = {}
    for line in (_run(argv, timeout=120) or "").splitlines():
        cid, _, size = line.partition("\t")
        res[cid] = to_bytes(size)
    return res


def gpu_state():
    """nvidia-smi -q -x: per GPU (uuid, index, memory, utilisation) and per process (pid, type, memory)."""
    out = _run(["nvidia-smi", "-q", "-x"], timeout=60)
    if not out:
        return None, []
    try:
        root = ET.fromstring(out)
    except ET.ParseError:
        return None, []
    devs, procs = [], []
    for g in root.findall("gpu"):
        uuid = (g.findtext("uuid") or "").strip()
        devs.append(
            {
                "uuid": uuid,
                "index": (g.findtext("minor_number") or "").strip(),
                "mem_used_bytes": to_bytes(g.findtext("fb_memory_usage/used")),
                "mem_total_bytes": to_bytes(g.findtext("fb_memory_usage/total")),
                "util_pct": to_bytes((g.findtext("utilization/gpu_util") or "").replace("%", "")),
            }
        )
        for p in g.findall("processes/process_info"):
            try:
                pid = int(p.findtext("pid"))
            except (TypeError, ValueError):
                continue
            procs.append(
                {
                    "gpu": uuid,
                    "pid": pid,
                    "type": (p.findtext("type") or "").strip(),
                    "mem_bytes": to_bytes(p.findtext("used_memory")),
                }
            )
    return devs, procs


def container_of(pid, ids):
    try:
        cg = Path(f"/proc/{pid}/cgroup").read_text()
    except OSError:
        return None
    return next((cid for cid in ids if cid in cg), None)


def cmd_sample(a):
    """Read-only: lists containers with docker ps, reads docker stats / docker ps --size and nvidia-smi -q -x and
    /proc/<pid>/cgroup; never stops, starts, execs into or changes a container. A container is sampled only when its
    ale.episode label names an episode folder of THIS run, so other runs' containers are never queried."""
    run_dir = Path(a.run_dir)
    out_path = Path(a.out) if a.out else run_dir / "resources.jsonl"
    stop = []
    signal.signal(signal.SIGTERM, lambda *_: stop.append(1))
    signal.signal(signal.SIGINT, lambda *_: stop.append(1))
    n = 0
    while not stop:
        t_start = time.time()
        if a.parent and not os.path.exists(f"/proc/{a.parent}"):
            break
        if run_dir.is_dir():
            episodes = {d.name for d in run_dir.iterdir() if d.is_dir() and not d.name.startswith(".")}
            cons = own_containers(episodes, a.managed_label) if episodes else []
            if cons:
                ids = [c["id"] for c in cons]
                st = docker_stats(ids)
                rw = rw_sizes(ids) if n % max(a.disk_every, 1) == 0 else {}
                devs, procs = gpu_state() if any(c["gpus"] for c in cons) else (None, [])
                owner = {p["pid"]: container_of(p["pid"], ids) for p in procs}
                ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t_start))
                by_ep = collections.defaultdict(list)
                for c in cons:
                    by_ep[c["episode"]].append(c)
                lines = []
                for ep, cs in sorted(by_ep.items()):
                    mine = {c["id"] for c in cs}
                    leased = {g for c in cs for g in c["gpus"]}
                    eprocs = [
                        dict(p, container=(owner[p["pid"]] or "")[:12]) for p in procs if owner.get(p["pid"]) in mine
                    ]
                    row = {
                        "t": round(t_start, 1),
                        "ts": ts,
                        "episode": ep,
                        "sample": n,
                        "containers": [
                            dict(
                                id=c["id"][:12],
                                name=c["name"],
                                role=c["role"],
                                gpus=c["gpus"],
                                **st.get(c["id"], {}),
                                rw_bytes=rw.get(c["id"]),
                            )
                            for c in cs
                        ],
                        "gpus": [d for d in (devs or []) if d["uuid"] in leased or d["index"] in leased]
                        if devs
                        else None,
                        "gpu_procs": eprocs,
                        "gpu_mem_bytes": sum(p["mem_bytes"] or 0 for p in eprocs) if devs is not None else None,
                    }
                    lines.append(json.dumps(row, separators=(",", ":")))
                with open(out_path, "a", encoding="utf-8") as f:
                    f.write("\n".join(lines) + "\n")
                n += 1
        delay = a.interval - (time.time() - t_start)
        while delay > 0 and not stop:
            time.sleep(min(delay, 1.0))
            delay -= 1.0
    return 0


# ------------------------------------------------------------------------------------------------ export
class Context:
    def __init__(self, repo, prices, extras, robot, link, derived_only):
        self.tasks = Tasks(repo)
        self.repo = Path(repo)
        self.prices, self.extras, self.robot = prices, extras, robot
        self.link, self.derived_only = link, derived_only
        self.commit = (git(self.repo, "rev-parse", "--short=12", "HEAD") or "").strip() or None

    def robot_rule(self, task, platform):
        if task in self.robot:
            return self.robot[task] or None
        if task in ROBOT:
            return ROBOT[task]
        return self.tasks.payload_rule(task, platform)


def load_extras(path):
    """{"replays": {task: [{file, local, what, poster_at?, source?, readme?}]}, "stills": {task: {local, what,
    source?}}, "robot": {task: regex or null}}; local paths relative to the extras file."""
    if not path:
        return {"replays": {}, "stills": {}, "robot": {}}
    doc = load_json(path, None)
    if not isinstance(doc, dict):
        raise SystemExit(f"cannot read the extras file {path}")
    base = Path(path).resolve().parent

    def local(p):
        return str(p if Path(p).is_absolute() else base / p)

    replays = {}
    for task, recs in (doc.get("replays") or {}).items():
        replays[task] = []
        for r in recs:
            r = dict(r)
            if not r.get("file") or not r.get("local"):
                raise SystemExit(f'extras: a replay of {task} needs "file" and "local"')
            safe_rel(r["file"])
            r["local"] = local(r["local"])
            if r.get("readme"):
                r["readme"] = local(r["readme"])
            replays[task].append(r)
    stills = {}
    for task, s in (doc.get("stills") or {}).items():
        s = dict(s)
        s["local"] = local(s["local"])
        stills[task] = s
    return {"replays": replays, "stills": stills, "robot": dict(doc.get("robot") or {})}


def score_block(metrics, anchor):
    ratio = metrics.get("ratio")
    if ratio is None and anchor.get("value") and is_number(metrics.get(anchor.get("metric"))):
        v = metrics[anchor["metric"]]
        ratio = (
            v / anchor["value"] if anchor.get("direction", "higher") == "higher" else anchor["value"] / max(v, 1e-12)
        )
        ratio = max(0.0, min(ratio, float(anchor.get("cap", 1.5))))
    return ratio


def export_episode(row, runs_dir, archive, ctx):
    task, run, folder_rel = row["task"], row["run"], safe_rel(row["folder"])
    ep = Path(runs_dir) / run / row["episode"]
    if not ep.is_dir():
        raise FileNotFoundError(f"episode folder {ep}")
    folder = Path(archive) / folder_rel
    if ctx.derived_only:
        folder.mkdir(parents=True, exist_ok=True)
        for name in DERIVED + tuple(r["file"] for r in ctx.extras["replays"].get(task, [])):
            p = folder / name
            if p.is_dir() and not p.is_symlink():
                shutil.rmtree(p)
            elif p.exists() or p.is_symlink():
                p.unlink()
    else:
        if folder.exists():
            shutil.rmtree(folder)
        folder.parent.mkdir(parents=True, exist_ok=True)
        copy_episode(ep, folder, ctx.link)

    lock = load_json(ep / "lock.json", {}) or {}
    res = load_json(ep / "result.json", {}) or {}
    ver = load_json(ep / "verification.json")
    traj = load_json(ep / "trajectory.json", {}) or {}
    execu = read_jsonl(ep / "trace.execution.jsonl")
    ag = lock.get("agent") or {}
    harness, model = ag.get("harness") or "agent", ag.get("model")
    effort = (ag.get("settings") or {}).get("effort") or "default"
    src = (lock.get("task") or {}).get("source") or {}
    platform = ctx.tasks.platform(task, lock)
    anchor, anchor_src = ctx.tasks.anchor(lock, task, platform)
    phases = res.get("phases") or []
    agent_phase = next((p for p in phases if p.get("phase") == "agent"), {})
    t0 = epoch(agent_phase.get("started_at"))

    # model calls and the per-step digest
    all_calls, replays_n = model_calls(ep, ctx.prices)
    calls = [c for c in all_calls if c.get("purpose") != "judge"]
    judge = [c for c in all_calls if c.get("purpose") == "judge"]
    by_id = {c["call_id"]: c for c in calls}
    by_resp = {c["response_id"]: c for c in calls if c.get("response_id")}
    cls = Classifier(ctx.robot_rule(task, platform))
    pics = Pictures(folder)
    root_ale = (traj.get("extra") or {}).get("ale") or {}
    native = None
    if root_ale.get("incomplete_reason") == "conversion_failed":
        found = sorted((ep / "logs" / "claude-code" / ".claude-config" / "projects").glob("*/*.jsonl"))
        sess = traj.get("session_id")
        found = [p for p in found if sess and p.name == sess + ".jsonl"] or found
        native = found[0] if found else None
    if native:
        steps, used, native_info = native_digest(native, by_resp, t0, cls, pics)
        digest_source = dict(
            kind="native session log",
            file=os.path.relpath(native, ep),
            why="the engine's ATIF conversion failed: {}".format(
                "; ".join(str(i.get("detail", "")) for i in root_ale.get("parse_issues") or [])
            ),
            **native_info,
        )
    else:
        steps, used = atif_digest(ep, traj, by_id, t0, cls, pics)
        digest_source = {
            "kind": "ATIF trajectory",
            "file": "trajectory.json",
            "schema": traj.get("schema_version"),
            "incomplete": root_ale.get("incomplete_reason"),
        }
    instr = (traj.get("steps") or [{}])[0].get("message")
    if isinstance(instr, str) and PTR.match(instr.strip()):
        instr = (ep / PTR.match(instr.strip()).group(1)).read_text(encoding="utf-8", errors="replace")
    if not isinstance(instr, str):
        instr = "\n".join(p.get("text") or "" for p in (instr or []) if isinstance(p, dict))
    prompt_md = ep / "logs" / "claude-code" / ".ale-harness" / "prompt.md"
    if not instr and prompt_md.is_file():
        instr = prompt_md.read_text(encoding="utf-8", errors="replace")

    # steps.bin
    steps_bin = None
    if pics.order:
        (folder / "steps.bin").write_bytes(bytes(pics.bin))
        mimes = collections.Counter(r["mime"] for r in pics.order)
        steps_bin = {
            "file": "steps.bin",
            "size": len(pics.bin),
            "mime": mimes.most_common(1)[0][0],
            "count": len(pics.order),
            "mimes": dict(mimes),
            "note": "every distinct picture the agent saw, once, in its original bytes; a step's out.pics "
            "are [offset, length, mime] ranges of this file and out.pic_files the same pictures as files",
        }

    # tokens and cost
    priced = [c for c in calls if c.get("usd") is not None]
    unlinked = [c for c in calls if c["call_id"] not in used]
    tokens = {
        "in": sum(c["input_total"] for c in calls),
        "cached": sum(c["cache_read"] for c in calls),
        "out": sum(c["out"] for c in calls),
        "usd": round(sum(c["usd"] for c in priced), 2) if priced else None,
    }
    tokens.update(
        cache_write=sum(c["cache_write"] for c in calls),
        uncached_in=sum(c["uncached"] for c in calls),
        think=None,
        think_note="thinking tokens are part of out: neither the gateway nor the harness reports them apart",
        calls=len(calls),
        failed_calls=sum(1 for c in calls if c.get("disposition") not in (None, "forwarded")),
        replayed_calls=replays_n,
        unlinked_calls=len(unlinked),
        unlinked_usd=round(sum(c["usd"] or 0 for c in unlinked), 4),
        unlinked_note="model calls no step links to (context compaction, a call the conversion could not place); their "
        "tokens are in the totals, not in any step",
        unpriced_calls=len(calls) - len(priced),
        cache_sources=dict(collections.Counter(c["cache_source"] for c in calls)),
        usd_source=PRICE_SOURCE,
        prices_per_mtok=price_for(model, ctx.prices),
        models=dict(collections.Counter(str(c["model"]) for c in calls)),
        latency_s=round(sum(c["latency_ms"] or 0 for c in calls) / 1000, 1),
    )
    if judge:
        tokens["judge"] = dict(
            calls=len(judge),
            **{"in": sum(c["input_total"] for c in judge), "out": sum(c["out"] for c in judge)},
            usd=round(sum(c["usd"] or 0 for c in judge), 4),
            note="the verifier's judge calls (purpose judge): not the agent's, not in the totals",
        )
    other = {}
    cc = claude_code_result(ep)
    if cc and cc.get("total_cost_usd") is not None:
        mu = cc.get("modelUsage") or {}
        canon = sorted(
            {v.get("canonicalModel") for v in mu.values() if isinstance(v, dict) and v.get("canonicalModel")}
        )
        other["claude_code_total_cost_usd"] = round(cc["total_cost_usd"], 4)
        other["claude_code_note"] = (
            "Claude Code's own figure from its result line (logs/claude-code/.ale-harness/"
            "stdout.log), priced with its own table: version {} prices {} as {}".format(
                ag.get("version"), model, ", ".join(canon) or "its own model entry"
            )
        )
        other["claude_code_usage"] = {
            k: cc.get(k)
            for k in (
                "num_turns",
                "duration_ms",
                "duration_api_ms",
                "usage",
                "modelUsage",
                "terminal_reason",
                "stop_reason",
            )
        }
    g = [c["gateway_cost_usd"] for c in calls if c.get("gateway_cost_usd") is not None]
    if g:
        other["gateway_cost_usd"] = round(sum(g), 4)
    if other:
        tokens["usd_other"] = other

    # score
    regrade = None
    if row.get("regrade"):
        rp = Path(row["regrade"])
        rp = rp if rp.is_absolute() else ep / rp
        rv = load_json(rp, {}) or {}
        rmetrics = rv.get("metrics") or {}
        rreward = next(
            (
                v
                for v in (rmetrics.get("reward_exact"), rv.get("reward"), (rv.get("rewards") or {}).get("reward"))
                if is_number(v)
            ),
            None,
        )
        if rreward is None:  # no record readable: the score curate wrote
            try:
                rreward = float(row.get("score"))
            except (TypeError, ValueError):
                rreward = None
        try:
            rel = rp.parent.resolve().relative_to(ep.resolve())
            rfolder = rel.as_posix() + "/"
        except ValueError:  # the record lives outside the episode: keep a copy
            rfolder = "regrade/"
            if (folder / "regrade").exists():
                shutil.rmtree(folder / "regrade")
            if du(rp.parent) <= REGRADE_COPY_LIMIT:
                shutil.copytree(rp.parent, folder / "regrade", symlinks=True)
            else:  # a large folder around the record: keep the record alone
                (folder / "regrade").mkdir()
                shutil.copy2(rp, folder / "regrade" / rp.name)
        regrade = {
            "folder": rfolder,
            "file": rp.name,
            "reward": rreward,
            "metrics": rmetrics,
            "note": row.get("notes") or "",
        }
    metrics = dict((regrade or {}).get("metrics") or (ver or {}).get("metrics") or res.get("metrics") or {})
    if regrade:
        reward_exact = regrade["reward"]
    else:
        reward_exact = metrics.get("reward_exact")
        if reward_exact is None and is_number((ver or {}).get("reward")):
            reward_exact = ver["reward"]
        if reward_exact is None:
            reward_exact = primary_reward(res.get("rewards"))
    reward = half_up(reward_exact) if is_number(reward_exact) else None  # what the engine reads (two decimals)
    ratio = score_block(metrics, anchor)
    full_at = anchor.get("full_at")
    if ratio is not None and full_at is not None:
        success = int(ratio >= full_at - 1e-12)  # the score saturates: reward 1.0
    else:
        success = int(reward is not None and reward >= 1.0)
    status = res.get("status")
    infra = int(status == "env_error" or (status == "agent_error" and "Connection" in json.dumps(res.get("failure"))))
    reward_block = {k: v for k, v in sorted(metrics.items()) if is_number(v)}
    if not regrade:
        for k, v in sorted((res.get("rewards") or {}).items()):  # every named reward of the result
            if k != "reward" and is_number(v):
                reward_block.setdefault(k, v)
    reward_block.update(reward=reward, reward_exact=reward_exact, success=success, infra_error=infra)
    for k in RESOURCE_FIELDS:
        reward_block.setdefault(k, None)
    peaks = resource_peaks(ep, Path(runs_dir) / run, folder)
    if peaks:
        pk, n = peaks
        reward_block.update(pk)
        reward_block["resource_monitor_missing"] = 0
        resources = {
            "recorded": True,
            "samples": n,
            "file": "resources.jsonl",
            "note": "docker stats and nvidia-smi of the episode's own containers (this episode's lines of "
            "the run's resources.jsonl); the peaks are over the solver and verifier containers",
        }
    else:
        reward_block["resource_monitor_missing"] = 1
        resources = {"recorded": False, "reason": RESOURCE_REASON}
    res_eff = (lock.get("resources") or {}).get("effective") or {}
    resources["limits"] = dict(
        cpus=res_eff.get("cpus"),
        memory_mb=res_eff.get("memory_mb"),
        gpus=((lock.get("resources") or {}).get("requested") or {}).get("gpus"),
        network_mode=res_eff.get("network_mode"),
        **ctx.tasks.timeouts(lock, task, platform),
    )

    grader_log = ""
    verify_cmd = [e for e in execu if e.get("kind") == "command_finished" and e.get("phase") == "verify"]
    if verify_cmd:
        st_err = verify_cmd[-1].get("stderr") or {}
        grader_log = st_err.get("inline") or ""
        if not grader_log and st_err.get("path"):
            try:
                grader_log = (ep / st_err["path"]).read_text(encoding="utf-8", errors="replace")
            except OSError:
                pass
    kinds = collections.Counter(c["kind"] for s in steps for c in s["calls"])
    files = {
        "run.json": "this file",
        "trajectory.jsonl.gz": "the full trajectory, one JSON object per line",
        "trajectory.json": "the ATIF trajectory as the engine wrote it",
        "blobs/": "pictures and large texts the trajectory references",
        "trace.transport.jsonl": "every model call",
        "trace.execution.jsonl": "engine phases and commands (the grader log is the verify phase's stderr)",
        "result.json": "status, rewards, phase timings",
        "verification.json": "the grader's full result",
        "lock.json": "provenance (also in provenance below)",
        "logs/": "the harness's native logs, when kept",
        "artifacts/": "the submission the grader scored",
    }
    if ctx.derived_only:
        files["note"] = "exported with --derived-only: the episode's own records are uploaded beside these files"
    if steps_bin:
        files["steps.bin"] = "the pictures the agent saw (byte ranges)"
    if peaks:
        files["resources.jsonl"] = "resource samples of the episode's containers (memory, processes, GPU memory, disk)"
    if any(r["file"].startswith("pics/") for r in pics.order):
        files["pics/"] = "pictures the agent saw that had no image file of their own in this folder, decoded"
    replay_recs = []
    for r in ctx.extras["replays"].get(task, []):
        shutil.copy2(r["local"], folder / r["file"])
        rec = {
            "file": r["file"],
            "what": r.get("what") or "",
            "note": SUBMISSION_NOTE,
            "source": r.get("source") or os.path.basename(r["local"]),
        }
        files[r["file"]] = "replay of this submission on a development episode (see replays)"
        if r.get("readme"):
            name = Path(r["file"]).stem + "_README" + (Path(r["readme"]).suffix or ".md")
            shutil.copyfile(r["readme"], folder / name)
            rec["readme"] = name
            files[name] = "notes on the replay"
        replay_recs.append(rec)
    if regrade:
        files[regrade["folder"]] = "the regrade: the same submission scored again (see score.regrade)"
    poster = make_poster(folder, task, pics, ctx.extras)
    if poster:
        files[poster["file"]] = "a still for the run (see poster_info)"
    trajectory = write_trajectory_gz(folder, traj, native, ep)

    suffix = row["episode"].rsplit("-", 1)[-1]
    slug = f"{task}--{suffix}"
    run_label = "{}-{}-{}".format(
        harness.replace("-", "_"),
        re.sub(r"^claude[-_]", "", model or "model").replace("-", "_").replace(".", "_"),
        effort,
    )
    notes = [n.strip() for n in (row.get("notes") or "").split("|") if n.strip()]
    doc = {
        "format": FORMAT,
        "slug": slug,
        "task": task,
        "bench": platform,
        "mode": ag.get("family"),
        "run": run_label,
        "agent": harness,
        "model": model,
        "effort": effort,
        "state": "success" if success else "failed",
        "success": success,
        "ended_by": "infra" if infra else None,
        "reward": reward_block,
        "wall_s": round((agent_phase.get("duration_ms") or 0) / 1000),
        "tokens": tokens,
        "task_text": instr,
        "replay": replay_recs[0]["file"] if replay_recs else None,
        "poster": poster["file"] if poster else None,
        "steps_bin": steps_bin,
        "trajectory": trajectory,
        "step_times": "recorded",
        "sim_total": None,
        "steps": steps,
        "format_note": (
            f"run.json format {FORMAT}, written by {TOOL}; field guide: "
            "skills/agent-trajectory-archiving/SKILL.md. mode: lock.json agent.family. sim_total and a "
            "step's sim: null where the agent's simulator commands run inside its own programs and are not "
            "logged apart"
        ),
        "exported": {"at": now_utc(), "by": TOOL, "commit": ctx.commit},
        "ale": {
            "run_id": run,
            "episode_id": ep.name,
            "status": status,
            "failure": res.get("failure"),
            "started_at": res.get("started_at"),
            "finished_at": res.get("finished_at"),
            "episode_wall_s": round((epoch(res.get("finished_at")) or 0) - (epoch(res.get("started_at")) or 0)),
            "engine_commit": (lock.get("framework") or {}).get("commit"),
            "engine_version": (lock.get("framework") or {}).get("version"),
            "harness_version": ag.get("version"),
            "native_session_id": traj.get("session_id"),
            "phases": [
                {
                    "phase": p.get("phase"),
                    "outcome": p.get("outcome"),
                    "started_at": p.get("started_at"),
                    "finished_at": p.get("finished_at"),
                    "duration_s": round((p.get("duration_ms") or 0) / 1000, 1),
                }
                for p in phases
            ],
        },
        "score": {
            "reward": reward,
            "reward_exact": reward_exact,
            "ratio": ratio,
            "metric": anchor.get("metric"),
            "direction": anchor.get("direction"),
            "measured": metrics.get(anchor.get("metric")) if anchor.get("metric") else None,
            "anchor": anchor.get("value"),
            "full_at": full_at,
            "cap": anchor.get("cap", 1.5),
            "anchor_source": anchor_src,
            "anchor_record": anchor or None,
            "rule": SCORE_RULE,
            "diagnostic": (ver or {}).get("diagnostic"),
            "criteria": (ver or {}).get("criteria") or None,
            "grader_log_tail": cap(grader_log, 4000, 3000) if grader_log else None,
            "regrade": regrade,
            "engine": {
                "reward": primary_reward(res.get("rewards")),
                "status": status,
                "metrics": (ver or {}).get("metrics"),
            }
            if regrade
            else None,
        },
        "task_info": {
            "platform": platform,
            "task_commit": src.get("commit"),
            "task_path": src.get("path"),
            "source_kind": src.get("kind"),
            "measured_as": (lock.get("task") or {}).get("name"),
            "content_digest": (lock.get("task") or {}).get("content_digest"),
            "site": f"{platform}/{task}",
        },
        "notes": notes,
        "folder": folder_rel.as_posix(),
        "files": files,
        "replays": replay_recs,
        "poster_info": poster,
        "resources": resources,
        "digest": {
            "source": digest_source,
            "steps": len(steps),
            "kinds": dict(kinds),
            "robot_rule": cls.rule,
            "robot_scripts": sorted(cls.scripts),
            "kind_rules": KIND_RULES,
            "truncation": (
                f"say in full; call code over {CODE_CAP} characters and step out.text over "
                f"{OUT_CAP} are cut in the middle, the full text is in the trajectory"
            ),
        },
        "pictures": pics.order,
        "provenance": lock,
    }
    with open(folder / "run.json", "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=1)
    return doc


def index_row(doc):
    sc, ti, ale = doc.get("score") or {}, doc.get("task_info") or {}, doc.get("ale") or {}
    row = {
        "slug": doc["slug"],
        "task": doc["task"],
        "bench": doc["bench"],
        "run": doc["run"],
        "model": doc["model"],
        "agent": doc["agent"],
        "effort": doc["effort"],
        "success": doc["success"],
        "state": doc["state"],
        "wall_s": doc["wall_s"],
        "steps": len(doc.get("steps") or []),
        "kinds": (doc.get("digest") or {}).get("kinds"),
        "usd": (doc.get("tokens") or {}).get("usd"),
        "poster": doc.get("poster"),
        "site": ti.get("site"),
    }
    if doc.get("ended_by"):
        row["ended_by"] = doc["ended_by"]
    row.update(
        reward=sc.get("reward"),
        ratio=round(sc["ratio"], 4) if is_number(sc.get("ratio")) else None,
        task_commit=ti.get("task_commit"),
        task_digest=ti.get("content_digest"),
        notes=" | ".join(doc.get("notes") or []),
        folder=doc.get("folder"),
        replay=doc.get("replay"),
        status=ale.get("status"),
        ale_run=ale.get("run_id"),
        episode=ale.get("episode_id"),
        date=str(ale.get("started_at"))[:10],
        metric=sc.get("metric"),
        anchor=sc.get("anchor"),
        full_at=sc.get("full_at"),
        pictures=len(doc.get("pictures") or []),
        tokens_in=(doc.get("tokens") or {}).get("in"),
        tokens_out=(doc.get("tokens") or {}).get("out"),
    )
    return row


def write_root_index(archive, rows):
    """index.json and index/<platform>.json from the run.json of every folder index.csv lists."""
    archive = Path(archive)
    by = collections.defaultdict(list)
    for r in rows:
        doc = load_json(archive / safe_rel(r["folder"]) / "run.json")
        if doc:
            x = index_row(doc)
            by[x["bench"] or "unknown"].append(x)
    idx = archive / "index"
    idx.mkdir(exist_ok=True)
    for old in idx.glob("*.json"):
        if old.stem not in by:
            old.unlink()
    for plat, xs in sorted(by.items()):
        with open(idx / (plat + ".json"), "w", encoding="utf-8") as f:
            json.dump(
                {"suite": plat, "runs": sorted(xs, key=lambda x: (x["task"], x["run"]))},
                f,
                ensure_ascii=False,
                indent=1,
            )
    allx = [x for xs in by.values() for x in xs]
    top = {
        "v": INDEX_VERSION,
        "total": len(allx),
        "suites": {p: len(xs) for p, xs in sorted(by.items())},
        "solved": {p: sum(x["success"] for x in xs) for p, xs in sorted(by.items())},
        "settings": dict(collections.Counter(x["run"] for x in allx)),
        "generated_at": now_utc(),
        "generated_by": TOOL,
        "layout": (
            "this file and index/<suite>.json, one per platform (suite = a platform key of "
            "tasks/platforms.json); a row's folder is its path under this root, its poster and replay are "
            "relative to that folder; each folder holds run.json next to the episode's records as the "
            "engine wrote them"
        ),
        "shards": {p: f"index/{p}.json" for p in sorted(by)},
        "pending_replays": sorted({x["task"] for x in allx if not x["replay"]}),
    }
    with open(archive / "index.json", "w", encoding="utf-8") as f:
        json.dump(top, f, ensure_ascii=False, indent=1)
    return top


def archive_folders(archive):
    return sorted(p.parent.relative_to(archive).as_posix() for p in Path(archive).glob("*/*/*/run.json"))


def check_apart(runs_dir, archive):
    r, a = Path(runs_dir).resolve(), Path(archive).resolve()
    if r == a or r in a.parents or a in r.parents:
        raise SystemExit(f"the archive ({a}) and the runs directory ({r}) must not contain each other")


def cmd_export(a):
    runs = Path(need(a.runs_dir, "--runs-dir", ENV_RUNS))
    archive = Path(need(a.archive, "--archive", ENV_ARCHIVE))
    check_apart(runs, archive)
    rows = read_index(a.index)
    prices = dict(PRICES)
    if a.prices:
        extra = load_json(a.prices, None)
        if not isinstance(extra, dict):
            raise SystemExit(f"cannot read the price table {a.prices}")
        prices.update(extra)
    extras = load_extras(a.extras)
    robot = dict(extras["robot"], **pairs(a.robot, "--robot"))
    ctx = Context(a.repo, prices, extras, robot, a.link, a.derived_only)
    archive.mkdir(parents=True, exist_ok=True)
    selected = [r for r in rows if not a.only or r["task"] in a.only]
    failed = []
    for r in selected:
        try:
            doc = export_episode(r, runs, archive, ctx)
        except Exception as e:  # noqa: BLE001 (report it and go on with the next trajectory)
            traceback.print_exc()
            failed.append((r["task"], f"{type(e).__name__}: {e}"))
            continue
        size = du(archive / safe_rel(r["folder"]))
        print(
            f"{r['task'][:45]:<45} steps {len(doc['steps']):4d} pics {len(doc['pictures']):4d} "
            f"usd {doc['tokens']['usd']!s:<7} reward {doc['score']['reward']!s:<5} {size / 1e6:7.1f} MB  "
            f"{doc['digest']['kinds']}",
            flush=True,
        )
    if Path(a.index).resolve() != (archive / "index.csv").resolve():
        shutil.copyfile(a.index, archive / "index.csv")
    top = write_root_index(archive, rows)
    listed = {safe_rel(r["folder"]).as_posix() for r in rows}
    stale = [f for f in archive_folders(archive) if f not in listed]
    for f in stale:
        if a.prune:
            shutil.rmtree(archive / f)
        print("{} {} (not in index.csv)".format("removed" if a.prune else "stale", f))
    print(
        f"index: {top['total']} trajectory(ies) in {archive}; {len(selected) - len(failed)} exported, "
        f"{len(failed)} failed {failed or ''}"
    )
    return 1 if failed else 0


# ------------------------------------------------------------------------------------------------ upload
def join_dest(dest, rel):
    return dest + rel if dest.endswith(":") else dest.rstrip("/") + "/" + rel


def rclone(args, dry_run=False, check=True):
    argv = (
        ["rclone"] + list(args) + (["--dry-run"] if dry_run and args[0] in ("copy", "copyto", "purge", "rmdir") else [])
    )
    r = subprocess.run(argv, capture_output=True, text=True, check=False)
    if check and r.returncode != 0:
        raise RuntimeError("rclone {}: {}".format(" ".join(args[:1]), (r.stderr or "").strip()[-400:]))
    return r


def upload_index(a, archive, dest, rows, common):
    """The root index files, then (--prune) the destination folders index.csv no longer lists."""
    for p in ("index.json", "index.csv"):
        if (archive / p).is_file():
            rclone(["copyto", str(archive / p), join_dest(dest, p)] + a.rclone_arg, a.dry_run)
    if (archive / "index").is_dir():
        rclone(["copy", str(archive / "index"), join_dest(dest, "index")] + common, a.dry_run)
    print("index uploaded" + (" (dry run)" if a.dry_run else ""))
    if not a.prune:
        return
    listed = {safe_rel(r["folder"]).as_posix() for r in rows}
    listing = rclone(["lsf", "--dirs-only", "-R", "--max-depth", "3", dest] + a.rclone_arg).stdout.split("\n")
    for d in sorted(x.rstrip("/") for x in listing if x.count("/") == 3):
        # only the archive's own <task>/<agent>/<run>__<episode> folders, never index/ or anything else
        if d in listed or d.split("/")[0] == "index" or not re.search(r"_effort-[^/]+$", d.rsplit("/", 1)[0]):
            continue
        print("{} {} (not in index.csv)".format("would remove" if a.dry_run else "removed", d), flush=True)
        if not a.dry_run:
            rclone(["purge", join_dest(dest, d)] + a.rclone_arg)
            for parent in (d.rsplit("/", 1)[0], d.split("/")[0]):  # its <task>/<agent> and <task>, when empty
                rclone(["rmdir", join_dest(dest, parent)] + a.rclone_arg, check=False)


def cmd_upload(a):
    archive = Path(need(a.archive, "--archive", ENV_ARCHIVE))
    dest = need(a.dest, "--dest", ENV_DEST)
    if not shutil.which("rclone"):
        raise SystemExit("rclone is not on PATH")
    rows = read_index(archive / "index.csv")
    chosen = [r for r in rows if not a.only or r["task"] in a.only]
    folders, missing = [], []
    for r in chosen:
        (folders if (archive / safe_rel(r["folder"]) / "run.json").is_file() else missing).append(r["folder"])
    if missing:
        print("not exported (run export first): {}".format(", ".join(missing)))
        return 1
    roots = [archive / f for f in folders] + [
        p for p in (archive / "index.json", archive / "index.csv", archive / "index") if p.exists()
    ]
    secrets = configured_secrets(secret_files(a), a.secret_env)
    hits = scan_paths(roots, secrets)
    accepted = {Path(x).as_posix() for x in a.accept}
    blocking = [h for h in hits if Path(h[0]).resolve().relative_to(archive.resolve()).as_posix() not in accepted]
    for path, kind, off, _ in hits:
        rel = Path(path).resolve().relative_to(archive.resolve()).as_posix()
        print(f"{'accepted' if rel in accepted else 'HIT'} {rel} {kind} at byte {off}")
    if blocking:
        print(
            f"upload refused: {len(blocking)} secret-scan hit(s); look at each (`scan --context 60`), and pass "
            "--accept PATH only for a hit that is not a secret"
        )
        return 1
    print(f"secret scan: no hit blocks; {len(secrets)} configured secret(s), {len(TOKEN_RX)} pattern(s)")
    common = [
        "--transfers",
        str(a.transfers),
        "--checkers",
        str(a.transfers),
        "--exclude",
        "__pycache__/**",
    ] + a.rclone_arg
    failed = []
    for f in folders:
        try:
            rclone(["copy", str(archive / f), join_dest(dest, f)] + common, a.dry_run)
            if not a.dry_run:
                chk = rclone(
                    [
                        "check",
                        "--one-way",
                        "--size-only",
                        "--exclude",
                        "__pycache__/**",
                        str(archive / f),
                        join_dest(dest, f),
                    ]
                    + a.rclone_arg,
                    check=False,
                )
                if chk.returncode != 0:
                    raise RuntimeError(f"check: {chk.stderr.strip()[-300:]}")
            print(f"ok   {f}", flush=True)
        except RuntimeError as e:
            print(f"FAIL {f}: {e}", flush=True)
            failed.append(f)
    if not a.only and not failed:
        try:
            upload_index(a, archive, dest, rows, common)
        except RuntimeError as e:
            print(f"FAIL index: {e}", flush=True)
            failed.append("index")
    elif a.only:
        print("index left alone (--only)")
    else:
        print("index not uploaded: a folder failed")
    print(
        f"{len(folders) - len(failed)} folder(s) uploaded{' (dry run)' if a.dry_run else ''}, {len(failed)} failed "
        f"{failed or ''}"
    )
    return 1 if failed else 0


# ------------------------------------------------------------------------------------------------ command line
def add_secret_args(p):
    p.add_argument(
        "--secrets-from",
        action="append",
        default=[],
        metavar="FILE",
        help="a dotenv, ini or rclone.conf file: every value whose key names a key, token, secret, password "
        f"or credential is searched for (also ${ENV_SECRETS}, separated by {os.pathsep!r})",
    )
    p.add_argument(
        "--secret-env",
        action="append",
        default=[],
        metavar="NAME",
        help="an environment variable whose value is searched for",
    )


def build_parser():
    p = argparse.ArgumentParser(
        prog="trajectories.py", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest="cmd", required=True)
    runs_help = f"the --runs-dir of `ale run` (default ${ENV_RUNS})"
    archive_help = f"the folder of kept trajectories (default ${ENV_ARCHIVE})"

    s = sub.add_parser("inventory", help="every model episode under a runs directory, as JSON lines")
    s.add_argument("--runs-dir", default=os.environ.get(ENV_RUNS), help=runs_help)
    s.add_argument("--out", default="-", help="a JSON-lines file (default: standard output)")
    s.add_argument(
        "--all", action="store_true", help='also validation runs and the engine\'s own harnesses, with "skip"'
    )
    s.set_defaults(func=cmd_inventory)

    s = sub.add_parser("curate", help="one trajectory per task and agent, written to index.csv")
    s.add_argument("inventory", help="the output of inventory")
    s.add_argument("--out", required=True, help="index.csv to write")
    s.add_argument(
        "--tasks-dir",
        default=str(REPO / "tasks"),
        help="the tasks that count: the folders tasks/<platform>/<task>/ here (default: this checkout)",
    )
    s.add_argument(
        "--task",
        action="append",
        default=[],
        metavar="NAME",
        help="one more task that counts (for example one in an open pull request)",
    )
    s.add_argument("--any-task", action="store_true", help="every task in the inventory counts")
    s.add_argument(
        "--rename", action="append", default=[], metavar="OLD=NEW", help="a task measured under an earlier name"
    )
    s.add_argument("--exclude-run", action="append", default=[], metavar="RUN", help="a run another run superseded")
    s.add_argument(
        "--overrides",
        metavar="FILE",
        help='JSON: {"tasks": [...], "rename": {...}, "exclude_runs": [...], "regrades": {run: '
        '{"verification": path, "score": number, "note": text}}, "notes": {run: text}}',
    )
    s.add_argument(
        "--pick",
        choices=("lowest", "highest", "latest"),
        default="lowest",
        help="which episode of the latest version (default: the lowest-scoring)",
    )
    s.set_defaults(func=cmd_curate)

    s = sub.add_parser("export", help="run.json and the other files into each kept folder, and the root index")
    s.add_argument("--runs-dir", default=os.environ.get(ENV_RUNS), help=runs_help)
    s.add_argument("--index", required=True, help="index.csv from curate")
    s.add_argument("--archive", default=os.environ.get(ENV_ARCHIVE), help=archive_help)
    s.add_argument(
        "--only",
        nargs="+",
        metavar="TASK",
        help="export these tasks only (the root index is rebuilt from every exported folder)",
    )
    s.add_argument(
        "--repo",
        default=str(REPO),
        help="the benchmark checkout that resolves task commits, anchors and platforms (default: this one)",
    )
    s.add_argument(
        "--prices",
        metavar="FILE",
        help="JSON price table, USD per million tokens per model: input, "
        "output, cache_read, cache_write_5m, cache_write_1h (replaces or adds entries)",
    )
    s.add_argument(
        "--extras",
        metavar="FILE",
        help='JSON: {"replays": {task: [{"file", "local", "what", '
        '"poster_at", "source", "readme"}]}, "stills": {task: {"local", "what", "source"}}, '
        '"robot": {task: regex}}',
    )
    s.add_argument(
        "--robot",
        action="append",
        default=[],
        metavar="TASK=REGEX",
        help="what counts as the task's own tools (see ROBOT)",
    )
    s.add_argument("--link", action="store_true", help="hard-link the episode's files instead of copying them")
    s.add_argument(
        "--derived-only",
        action="store_true",
        help="write only the files this script derives; the episode's own records go up separately",
    )
    s.add_argument("--prune", action="store_true", help="remove archive folders index.csv no longer lists")
    s.set_defaults(func=cmd_export)

    s = sub.add_parser("scan", help="configured secrets and token patterns; exit status 1 on a hit")
    s.add_argument("paths", nargs="+", help="files or folders")
    add_secret_args(s)
    s.add_argument("--context", type=int, default=0, metavar="N", help="print N bytes around each hit, redacted")
    s.add_argument("--include-media", action="store_true", help="also pictures, videos and steps.bin")
    s.set_defaults(func=cmd_scan)

    s = sub.add_parser("upload", help="rclone copy of the archive, scanned first and checked after")
    s.add_argument("--archive", default=os.environ.get(ENV_ARCHIVE), help=archive_help)
    s.add_argument(
        "--dest",
        default=os.environ.get(ENV_DEST),
        help=f"any rclone destination, REMOTE:PATH or a local path (default ${ENV_DEST})",
    )
    s.add_argument("--only", nargs="+", metavar="TASK", help="upload these tasks only (the root index is left alone)")
    s.add_argument("--prune", action="store_true", help="then remove destination folders index.csv no longer lists")
    s.add_argument("--dry-run", action="store_true", help="show what would be copied and removed")
    s.add_argument(
        "--accept",
        action="append",
        default=[],
        metavar="PATH",
        help="a file (relative to the archive) whose scan hit was looked at and is not a secret",
    )
    s.add_argument("--transfers", type=int, default=8)
    s.add_argument(
        "--rclone-arg",
        action="append",
        default=[],
        metavar="ARG",
        help="one more argument for rclone (for example --rclone-arg=--drive-chunk-size=64M)",
    )
    add_secret_args(s)
    s.set_defaults(func=cmd_upload)

    s = sub.add_parser("sample", help="resource samples of one run's own episode containers (start beside ale run)")
    s.add_argument("--run-dir", required=True, help="<runs dir>/<run id> of the run to sample")
    s.add_argument("--interval", type=float, default=10.0, help="seconds between samples (default 10)")
    s.add_argument("--disk-every", type=int, default=6, help="the writable-layer size every N samples (it is slow)")
    s.add_argument("--parent", type=int, help="stop when this pid is gone (the launcher)")
    s.add_argument("--out", help="default: <run dir>/resources.jsonl")
    s.add_argument("--managed-label", default="ale.managed=true", help="the label every engine sandbox carries")
    s.set_defaults(func=cmd_sample)
    return p


def main(argv=None):
    a = build_parser().parse_args(argv)
    return a.func(a)


if __name__ == "__main__":
    sys.exit(main())
