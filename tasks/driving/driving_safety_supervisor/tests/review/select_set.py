"""Pre-registered selection of the fixed hidden set (a maintainer's tool; the grader never runs it).

The rule, fixed before any candidate ran (written 2026-11-06 and committed with the freeze; clause 4's walker
screen and the numbering of a selection that supersedes another were added, with the second freeze, before that
selection's draw):

1. Freeze. The files that decide an episode and the reference's run (FROZEN below: the bridge in both copies,
   drive_protocol, drive_variants, episode_score, verify, sim_client, the expert, its footprints, solve.sh) are
   committed first. `draw` refuses to start unless every one of them equals HEAD, and records their sha256, the
   commit and the task image; `run` and `make` refuse to go on when any of them differs from that record.
2. Draw. Batch 1 is 96 candidates: four rounds of the 24 graded cells (drive_variants.graded_cells over
   grader_config's towns, in grading order), ids c<round * 24 + cell, three digits>; the rounds start at 0, and a
   selection that supersedes another goes on from the round after the other's last (`--first-round`), so no id
   names two candidates. Each is drawn from os.urandom:
   seed = int.from_bytes(8 bytes, "little") % 2**62, drawn again while below 2**32 or equal to an earlier seed;
   distractors = one byte % 3 (a byte of 255 is drawn again); lock_key = 32 bytes, hex. selection.json is written,
   and committed, before any candidate runs.
3. Run. A candidate runs ONCE, exactly as the grader runs an episode: review/run_episode.py in a container of the
   task image calls verify.grade_episode for it, a token run of the solve.sh submission over the real wire through
   netguard, the bridge as the simulator account from a copy in /run, the candidate's own lock key; one simulator
   per container, the world reloaded before every episode. Round r runs the round-r candidate of every cell that has
   no kept candidate from an earlier round, so a cell's candidates run in drawn order, one after another.
4. Keep. A candidate is kept when its run has: termination "completed"; no abort; walker, vehicle and static
   collisions 0 and hazard_collisions 0; red_lights 0; the smallest box-to-box clearance from the car to any actor
   of the ground-truth block (real requests only; an actor's size is the larger of its box and
   solution/footprints.json) at least 0.15 m; the timeout's clock (time_s - excused_s) at most 0.85 of the
   episode's time limit (route_length_m / 5 + 15 s); and no walker within 25 m of the car faster than 4 m/s over the
   ground (the block's velocity) on two consecutive real steps. People walk at 0.9-1.7 m/s in every staging: one that
   fast has been launched by the physics, which could hit a car a few metres further on through no fault of its
   driver; a single step can be the staging putting a person back on their spot by hand. Pace, harsh brakes and the
   score are not part of the rule.
   A cell's first kept candidate is its episode, and the layout digest of that run (truth["layout"]) is the set's
   layout_sha256 for it.
5. A cell with no kept candidate after its four rounds gets four more (`draw --batch 2`) once batch 1 has run.
   Every candidate is recorded: kept; dropped, with the clauses it failed, its outcome and the cause found
   (`make --causes`); or not run (its cell was filled first).
6. Hollowing check (`report`), per variant: candidates run, kept and dropped by cause, and the staging parameters
   of the kept against the run candidates and the drawn ranges. Fix the expert, freeze again and draw a new
   selection (never keep the easy subset) if a variant drops more than 1/3 of the candidates it ran, if two or more
   drops share a cause in the expert, or if the drops concentrate on one value of a categorical staging parameter.
   Draws that no driver could complete go into tests/README.md.
7. Replacement. A kept candidate that fails clause 4 in the anchor verify is replaced by the next candidate of its
   cell that is kept (run in drawn order as in 3). A second such failure anywhere is a robustness defect of the
   expert: fix it, freeze again, draw a new selection.

    python3 -B select_set.py draw --selection SEL.json [--batch 1] [--first-round R --supersedes OLD.json --why TEXT]
    python3 -B select_set.py run --selection SEL.json --round R --out DIR [--gpus 0,1] [--cpus 16-23,24-31]
    python3 -B select_set.py make --selection SEL.json --out EPISODES.json [--causes CAUSES.json]
    python3 -B select_set.py report --selection SEL.json

`run` works on the host (docker, nvidia-smi): it hands the round's candidates to dev_batch.run, which starts one
container of the task image per GPU (each only once its GPU has 8 GB free) running them through run_episode.py, and
then reads their results into selection.json. Paths are found from this file, so it runs from any checkout.
"""

import argparse
import datetime
import hashlib
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TESTS = os.path.dirname(HERE)
TASK = os.path.dirname(TESTS)
sys.dont_write_bytecode = True
sys.path.insert(0, HERE)
sys.path.insert(0, TESTS)

import dev_batch  # noqa: E402
from drive_variants import VARIANTS, graded_cells  # noqa: E402

SEL_FORMAT = "driving_safety_supervisor selection 1"
SET_FORMAT = "driving_safety_supervisor fixed set 1"      # verify.SET_FORMAT (verify needs the kit to import)
FROZEN = ("tests/drive_bridge.py", "environment/setup/payload/drive/drive_bridge.py", "tests/drive_protocol.py",
          "tests/drive_variants.py", "tests/episode_score.py", "tests/verify.py", "tests/sim_client.py",
          "solution/solver.py", "solution/footprints.json", "solution/solve.sh")
TOOLS = ("tests/review/select_set.py", "tests/review/run_episode.py", "tests/review/dev_batch.py")
ROUNDS_PER_BATCH = 4
SEED_RANGE = (1 << 32, 1 << 62)
MIN_CLEARANCE_M = 0.15
TIME_FRACTION = 0.85
LAUNCH_STEPS = 2          # consecutive real steps of one walker near the car above run_episode.FAST_WALKER_MPS
DROP_SHARE = 1.0 / 3.0
# the ranges drive_variants draws from (for the hollowing check; the staging record holds the values drawn)
RANGES = {
    "debris": {"props.n": "2-4", "props[].1": "0-4 m ahead", "props[].2": "lateral -1.2..1.2"},
    "walking_in_lane": {"lateral": "-0.3..0.8", "speed": "0.9..1.6"},
    "shoulder_pass": {"wrecks[].1": "-0.9..-0.7", "sign_lateral": "0 (-0.9..-0.7 if sign_room < 1.0)",
                      "sign_room": "centred sign", "shoulder_width": ">= 2.4"},
    "spinning_car": {"spin": "+-(180..270)", "v_car": "9..11", "slide_m": "9..13", "slide_s": "1.6..2.2"},
    "overtaking_oncoming": {"intrusion": "0.8..1.1", "v_car": "9..11", "v_bike": "3..4.5", "release_m": "55..70"},
    "reversing_car": {"lead_kmh": "22..30", "back_speed": "0.8..1.2"},
    "dropped_cargo": {"truck_kmh": "25..32", "pieces": "2-4"},
    "crowd_against_light": {"group": "4-8 (spawned)", "turners": "1-2", "people[].0": "delay 0..2.5",
                            "people[].1": "speed 1.1..1.7"},
}


def sha256_file(path):
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def hashes(paths):
    return {p: sha256_file(os.path.join(TASK, p)) for p in paths}


def load(path):
    with open(path) as handle:
        return json.load(handle)


def save(path, sel):
    tmp = path + ".tmp"
    with open(tmp, "w") as handle:
        json.dump(sel, handle, indent=1, sort_keys=False)
        handle.write("\n")
    os.replace(tmp, path)


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def git(*args):
    try:
        return subprocess.run(["git", "-C", TASK] + list(args), stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                              universal_newlines=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None


def frozen_committed():
    tracked = git("ls-files", "--error-unmatch", "--", *FROZEN)
    clean = git("diff", "--quiet", "HEAD", "--", *FROZEN)
    return tracked is not None and tracked.returncode == 0 and clean is not None and clean.returncode == 0


def image_id(image):
    try:
        out = subprocess.run(["docker", "image", "inspect", "--format", "{{.Id}}", image], stdout=subprocess.PIPE,
                             stderr=subprocess.DEVNULL, universal_newlines=True, timeout=60).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        out = ""
    return out or None


def check_frozen(sel):
    now = hashes(FROZEN)
    changed = [p for p in FROZEN if sel["frozen"].get(p) != now[p]]
    if changed:
        raise SystemExit("the frozen files changed since the draw: {}".format(changed))


def towns():
    with open(os.path.join(TESTS, "grader_config.json")) as handle:
        return list(json.load(handle).get("towns", ["Town04", "Town05", "Town10HD"]))


# ------------------------------------------------------------------------------------------------ draw
def _seed(seen):
    redrawn = 0
    while True:
        s = int.from_bytes(os.urandom(8), "little") % (1 << 62)
        if s >= SEED_RANGE[0] and s not in seen:
            return s, redrawn
        redrawn += 1


def _distractors():
    while True:
        b = os.urandom(1)[0]
        if b < 255:
            return b % 3


def cmd_draw(a):
    cells = graded_cells(towns())
    kind_of = {v: kind for kind, variants in VARIANTS.items() for v in variants}
    if a.batch == 1:
        if os.path.exists(a.selection):
            raise SystemExit("{} exists: batch 1 starts a selection".format(a.selection))
        supersedes = None
        if a.supersedes:
            old = load(a.supersedes)
            supersedes = {"file": os.path.relpath(os.path.abspath(a.supersedes), TASK),
                          "sha256": sha256_file(a.supersedes), "frozen_commit": old.get("frozen_commit"),
                          "last_round": max(c["round"] for c in old["candidates"]),
                          "episodes_sha256": (old.get("set") or {}).get("episodes_sha256"), "why": a.why}
            if a.first_round <= supersedes["last_round"]:
                raise SystemExit("--first-round must come after the superseded selection's last round ({})".format(
                    supersedes["last_round"]))
        if not frozen_committed():
            raise SystemExit("the frozen files are not committed as they are: commit the freeze first")
        head = git("rev-parse", "HEAD")
        sel = {"format": SEL_FORMAT, "frozen": hashes(FROZEN), "frozen_commit": head.stdout.strip() if head else None,
               "tools": hashes(TOOLS), "image": a.image, "image_id": image_id(a.image),
               "rule": "tests/review/select_set.py (module docstring)", "cells": [list(c) for c in cells],
               "batches": [], "candidates": []}
        if supersedes:
            sel["supersedes"] = supersedes
        want = list(range(len(cells)))
        first_round = a.first_round
    else:
        sel = load(a.selection)
        check_frozen(sel)
        states = [cell_state(sel, k) for k in range(len(cells))]
        if any(kept is None and due is not None for kept, due in states):
            raise SystemExit("a cell without a kept candidate still has candidates due: run them first")
        want = [k for k, (kept, _due) in enumerate(states) if kept is None]
        if not want:
            raise SystemExit("every cell has a kept candidate: no batch needed")
        first_round = 1 + max(c["round"] for c in sel["candidates"])
    seen = {c["seed"] for c in sel["candidates"]}
    keys = {c["lock_key"] for c in sel["candidates"]}
    if a.batch == 1 and a.supersedes:          # nor any seed or key of the selection this one replaces
        seen |= {c["seed"] for c in old["candidates"]}
        keys |= {c["lock_key"] for c in old["candidates"]}
    redrawn, count = 0, 0
    for r in range(first_round, first_round + ROUNDS_PER_BATCH):
        for k in want:
            town, variant = cells[k]
            seed, again = _seed(seen)
            redrawn += again
            seen.add(seed)
            key = os.urandom(32).hex()
            while key in keys:
                key = os.urandom(32).hex()
            keys.add(key)
            sel["candidates"].append({"id": "c{:03d}".format(r * len(cells) + k), "draw": len(sel["candidates"]),
                                      "batch": a.batch, "round": r, "cell": k, "town": town, "variant": variant,
                                      "kind": kind_of[variant], "seed": seed, "distractors": _distractors(),
                                      "lock_key": key, "status": "drawn"})
            count += 1
    sel["batches"].append({"batch": a.batch, "drawn_utc": utc_now(), "count": count, "cells": want,
                           "rounds": [first_round, first_round + ROUNDS_PER_BATCH - 1], "seeds_redrawn": redrawn,
                           "source": "os.urandom: 8 bytes little-endian per seed (mod 2**62), 1 byte per distractor "
                                     "count, 32 bytes per lock key"})
    save(a.selection, sel)
    print("batch {}: {} candidates over {} cells ({} seeds redrawn) -> {}".format(a.batch, count, len(want), redrawn,
                                                                                 a.selection))


# ------------------------------------------------------------------------------------------------ run
def keep_clauses(c):
    """The clauses of rule 4 a run fails (empty: kept)."""
    run = c["run"]
    t = run["truth"]
    col = t.get("collisions", {})
    failed = []
    if run.get("aborted"):
        failed.append("aborted")
    if t.get("termination") != "completed":
        failed.append("termination {}".format(t.get("termination") or "-"))
    for k in ("walker", "vehicle", "static"):
        if col.get(k, 0):
            failed.append("{} collisions {}".format(k, col[k]))
    if t.get("hazard_collisions", 0):
        failed.append("hazard collisions {}".format(t["hazard_collisions"]))
    if t.get("red_lights", 0):
        failed.append("red lights {}".format(t["red_lights"]))
    if "walker_launch" not in run:
        failed.append("no record of the walkers' speeds")
    elif run["walker_launch"] and run["walker_launch"][0] >= LAUNCH_STEPS:
        steps, peak, ident, bp, dist, when = run["walker_launch"][:6]
        failed.append("walker {} ({}) at {:.1f} m/s, {:.1f} m from the car, {} steps at t={}".format(
            ident, bp, peak, dist, steps, when))
    least = (run.get("clearance") or {}).get("all")
    if least is None or least[0] < MIN_CLEARANCE_M:
        failed.append("clearance {}".format("-" if least is None else "{:.3f} m to {} {}".format(*least[:3])))
    limit = t.get("route_length_m", 0.0) / 5.0 + 15.0
    clock = t.get("time_s", 0.0) - t.get("excused_s", 0.0)
    if clock > TIME_FRACTION * limit:
        failed.append("time {:.2f} s of a {:.2f} s limit".format(clock, limit))
    return failed


def cell_state(sel, k):
    """(kept candidate or None, the next candidate of cell k due to run or None)."""
    for c in sorted((c for c in sel["candidates"] if c["cell"] == k), key=lambda c: c["draw"]):
        if c["status"] == "ran" and c.get("keep"):
            return c, None
        if c["status"] == "drawn":
            return None, c
    return None, None


def cmd_run(a):
    sel = load(a.selection)
    check_frozen(sel)
    todo = []
    for k in range(len(sel["cells"])):
        kept, due = cell_state(sel, k)
        if kept is None and due is not None and due["round"] == a.round:
            todo.append(due)
    if not todo:
        raise SystemExit("round {}: no candidate due".format(a.round))
    out = os.path.abspath(a.out)
    os.makedirs(out, exist_ok=True)
    specs = [{key: c[key] for key in ("id", "town", "kind", "variant", "seed", "distractors", "lock_key")}
             for c in todo]
    path = os.path.join(out, "round_specs.json")
    with open(path, "w") as handle:
        json.dump(specs, handle, indent=1)
    started = utc_now()
    rc = dev_batch.run(argparse.Namespace(specs=path, out=out, gpus=a.gpus, cpus=a.cpus, image=sel["image"],
                                          solver=None))
    results = {}
    if os.path.exists(os.path.join(out, "results.jsonl")):
        with open(os.path.join(out, "results.jsonl")) as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    results[row["id"]] = row
    ran = 0
    for c in todo:
        row = results.get(c["id"])
        if row is None:
            continue                       # its container ended before it: still due
        if not row.get("privileged"):
            raise SystemExit("{} ran without the reference token".format(c["id"]))
        c["run"] = {key: row.get(key) for key in ("truth", "staging", "score", "layout", "probes", "clearance",
                                                  "walker_fast", "walker_launch", "speed_limits", "max_reply_s",
                                                  "wall_s", "aborted", "note")}
        c["run"]["dir"] = out
        c["failed"] = keep_clauses(c)
        c["keep"] = not c["failed"]
        c["status"] = "ran"
        ran += 1
        t = c["run"]["truth"]
        print("{} {:8s} {:20s} {} score={:.4f} {} red={} harsh={} pace={} clear={} {}".format(
            c["id"], c["town"], c["variant"], "keep" if c["keep"] else "DROP", c["run"]["score"],
            t.get("termination"), t.get("red_lights"), t.get("harsh_brakes"), t.get("pace"),
            (c["run"]["clearance"] or {}).get("all"), "; ".join(c["failed"])), flush=True)
    sel.setdefault("runs", []).append({"round": a.round, "started_utc": started, "finished_utc": utc_now(),
                                       "dir": out, "gpus": a.gpus, "cpus": a.cpus, "due": len(todo), "ran": ran,
                                       "rc": rc})
    save(a.selection, sel)
    if ran < len(todo):
        raise SystemExit("round {}: {} of {} candidates have no result (still due)".format(a.round, len(todo) - ran,
                                                                                            len(todo)))


# ------------------------------------------------------------------------------------------------ make
def classify(sel):
    kept, dropped, not_run, open_cells = {}, [], [], []
    for k in range(len(sel["cells"])):
        found = None
        for c in sorted((c for c in sel["candidates"] if c["cell"] == k), key=lambda c: c["draw"]):
            if found is not None:
                not_run.append(c)
            elif c["status"] == "ran" and c.get("keep"):
                found = c
            elif c["status"] == "ran":
                dropped.append(c)
            else:
                open_cells.append(k)
                break
        if found is not None:
            kept[k] = found
        elif k not in open_cells:
            open_cells.append(k)
    return kept, dropped, not_run, open_cells


def cmd_make(a):
    sel = load(a.selection)
    check_frozen(sel)
    kept, dropped, not_run, open_cells = classify(sel)
    if open_cells:
        raise SystemExit("cells without a kept candidate: {} (run them, or draw batch 2)".format(open_cells))
    causes = load(a.causes) if a.causes else {}
    for c in sel["candidates"]:
        c.pop("selection", None)
    for c in kept.values():
        c["selection"] = "kept"
    for c in dropped:
        c["selection"] = "dropped"
        if c["id"] in causes:
            c["cause"] = causes[c["id"]]
    for c in not_run:
        c["selection"] = "not run"
    episodes = []
    for k in range(len(sel["cells"])):
        c = kept[k]
        episodes.append({key: c[key] for key in ("id", "town", "variant", "kind", "seed", "distractors", "lock_key")})
        episodes[-1]["layout_sha256"] = c["run"]["layout"]
    batches = "; ".join("batch {} drawn {} ({} candidates over {} cells, {} seeds redrawn)".format(
        b["batch"], b["drawn_utc"], b["count"], len(b["cells"]), b["seeds_redrawn"]) for b in sel["batches"])
    frozen = ", ".join("{} {}".format(p, h[:16]) for p, h in sel["frozen"].items())
    drops = "; ".join("{} {} {} ({}{})".format(c["id"], c["town"], c["variant"], ", ".join(c["failed"]),
                                               "; " + c["cause"] if c.get("cause") else "") for c in dropped) or "none"
    source = (
        "Selected {made} by tests/review/select_set.py (the rule is its docstring, committed with the freeze before "
        "the draw). Candidates: {batches}, from os.urandom (seed: 8 bytes little-endian mod 2**62, drawn again below "
        "2**32 or on a repeat; distractors: 1 byte mod 3; lock key: 32 bytes), written to selection.json and "
        "committed before any candidate ran. Frozen (commit {commit}; sha256, first 16): {frozen}. Image {image} "
        "({image_id}). Every candidate ran once, exactly as graded (review/run_episode.py: verify.grade_episode, a "
        "token run of the solve.sh submission through netguard, its own lock key); round r ran only the cells "
        "without a kept candidate. Kept: the first candidate of each cell whose run completed with no abort, no "
        "collision, no hazard hit and no red light, at least {clear} m of box clearance, at most {frac} of the "
        "time limit, and no walker near the car launched by the physics (over 4 m/s within 25 m on {launch} "
        "consecutive steps). Kept: {kept}. Dropped: {drops}. Not run: {not_run} candidates.{supersedes} Every "
        "candidate with its run: selection.json.").format(
            made=utc_now()[:10], batches=batches, commit=(sel.get("frozen_commit") or "?")[:12], frozen=frozen,
            image=sel.get("image"), image_id=(sel.get("image_id") or "?")[:19], clear=MIN_CLEARANCE_M,
            frac=TIME_FRACTION, launch=LAUNCH_STEPS, kept=", ".join(kept[k]["id"] for k in range(len(sel["cells"]))),
            drops=drops, not_run=len(not_run),
            supersedes=" It supersedes the selection in {} (frozen at {}): {}".format(
                sel["supersedes"]["file"], (sel["supersedes"].get("frozen_commit") or "?")[:12],
                sel["supersedes"]["why"]) if sel.get("supersedes") else "")
    with open(a.out, "w") as handle:
        json.dump({"format": SET_FORMAT, "episodes": episodes, "source": source}, handle, indent=1)
        handle.write("\n")
    sel["set"] = {"made_utc": utc_now(), "kept": [kept[k]["id"] for k in range(len(sel["cells"]))],
                  "dropped": [c["id"] for c in dropped], "not_run": [c["id"] for c in not_run],
                  "episodes_sha256": sha256_file(a.out)}
    save(a.selection, sel)
    print("set: {} kept, {} dropped, {} not run -> {}".format(len(kept), len(dropped), len(not_run), a.out))


# ------------------------------------------------------------------------------------------------ report
def flatten(staging):
    """(numeric {name: [values]}, categorical {name: [values]}) of one staging record."""
    num, cat = {}, {}
    for key, value in (staging or {}).items():
        if isinstance(value, bool) or isinstance(value, str):
            cat.setdefault(key, []).append(str(value))
        elif isinstance(value, (int, float)):
            num.setdefault(key, []).append(float(value))
        elif isinstance(value, list):
            num.setdefault(key + ".n", []).append(float(len(value)))
            for row in value:
                if isinstance(row, str):
                    cat.setdefault(key + "[]", []).append(row)
                elif isinstance(row, list):
                    for j, item in enumerate(row):
                        if isinstance(item, str):
                            cat.setdefault("{}[].{}".format(key, j), []).append(item)
                        elif isinstance(item, (int, float)) and not isinstance(item, bool):
                            num.setdefault("{}[].{}".format(key, j), []).append(float(item))
    return num, cat


def _merge(records):
    num, cat = {}, {}
    for staging in records:
        n, c = flatten(staging)
        for k, v in n.items():
            num.setdefault(k, []).extend(v)
        for k, v in c.items():
            cat.setdefault(k, []).extend(v)
    return num, cat


def cmd_report(a):
    sel = load(a.selection)
    kept, dropped, not_run, open_cells = classify(sel)
    ran = [c for c in sel["candidates"] if c["status"] == "ran"]
    print("candidates {}: ran {}, kept {}, dropped {}, not run {}, cells open {}".format(
        len(sel["candidates"]), len(ran), len(kept), len(dropped), len(not_run), open_cells))
    variants = []
    for _town, v in sel["cells"]:
        if v not in variants:
            variants.append(v)
    flags = []
    causes = {}
    for c in dropped:
        if c.get("cause", "").startswith("expert"):
            causes.setdefault(c["cause"], []).append(c["id"])
    for cause, ids in causes.items():
        if len(ids) >= 2:
            flags.append("{} drops share a cause in the expert: {} ({})".format(len(ids), cause, ", ".join(ids)))
    for v in variants:
        run_v = [c for c in ran if c["variant"] == v]
        kept_v = [c for c in kept.values() if c["variant"] == v]
        drop_v = [c for c in dropped if c["variant"] == v]
        share = len(drop_v) / float(len(run_v)) if run_v else 0.0
        print("\n## {}: ran {}, kept {}, dropped {} ({:.0%})".format(v, len(run_v), len(kept_v), len(drop_v), share))
        if share > DROP_SHARE:
            flags.append("{} drops {:.0%} of the candidates it ran".format(v, share))
        for c in drop_v:
            print("  drop {} {} seed={} d={}: {}{}".format(c["id"], c["town"], c["seed"], c["distractors"],
                                                          "; ".join(c["failed"]),
                                                          " -- " + c["cause"] if c.get("cause") else ""))
        for c in run_v:
            fast, launch = c["run"].get("walker_fast"), c["run"].get("walker_launch")
            if launch or (fast and fast[0] > 2.0):
                print("  walkers {}: fastest near the car {}, longest fast run {}".format(c["id"], fast, launch))
        knum, kcat = _merge([c["run"]["staging"] for c in kept_v])
        rnum, rcat = _merge([c["run"]["staging"] for c in run_v])
        dnum, dcat = _merge([c["run"]["staging"] for c in drop_v])
        for name in sorted(rnum):
            kv, rv = knum.get(name, []), rnum[name]
            print("  {:22s} kept {:>17s}  ran {:>17s}  drawn {}".format(
                name, "{:.2f}..{:.2f}".format(min(kv), max(kv)) if kv else "-", "{:.2f}..{:.2f}".format(min(rv), max(rv)),
                RANGES.get(v, {}).get(name, "")))
        def count(xs):
            return ", ".join("{} {}".format(x, xs.count(x)) for x in sorted(set(xs)))
        for name in sorted(rcat):
            print("  {:22s} kept [{}]  dropped [{}]".format(name, count(kcat.get(name, [])),
                                                            count(dcat.get(name, []))))
            dv = dcat.get(name, [])
            for value in sorted(set(dv)):
                if dv.count(value) >= 2:
                    flags.append("{}: {} drops share {} = {}".format(v, dv.count(value), name, value))
        lengths = [c["run"]["truth"].get("route_length_m", 0.0) for c in run_v]
        if lengths:
            print("  {:22s} ran {:.1f}..{:.1f} m".format("route length", min(lengths), max(lengths)))
        limits = sorted({tuple(c["run"].get("speed_limits") or []) for c in run_v})
        print("  {:22s} ran {}".format("speed limits (m/s)", ", ".join("/".join(str(x) for x in s) for s in limits)))
    print("\nhollowing check: {}".format("FLAGGED: " + "; ".join(flags) if flags else "no flag"))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("draw")
    p.add_argument("--selection", required=True)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--first-round", type=int, default=0)
    p.add_argument("--supersedes", help="the selection this one replaces (batch 1 only)")
    p.add_argument("--why", default="", help="why it is replaced")
    p.add_argument("--image", default=dev_batch.IMAGE)
    p.set_defaults(fn=cmd_draw)
    p = sub.add_parser("run")
    p.add_argument("--selection", required=True)
    p.add_argument("--round", type=int, required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--gpus", default="0,1")
    p.add_argument("--cpus", default="16-23,24-31")
    p.set_defaults(fn=cmd_run)
    p = sub.add_parser("make")
    p.add_argument("--selection", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--causes")
    p.set_defaults(fn=cmd_make)
    p = sub.add_parser("report")
    p.add_argument("--selection", required=True)
    p.set_defaults(fn=cmd_report)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
