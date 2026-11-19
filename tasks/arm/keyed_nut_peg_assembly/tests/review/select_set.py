"""The fixed hidden set's selection, and the development kits' (a maintainer's tool; the grader never runs it).

The rule was pre-registered in tests/hidden/selection.json ("rule", "candidates", "frozen") and committed before any
candidate ran: the frozen reference solution (solution/) runs each candidate kit ONCE, in drawn order, through the
grader itself (tests/review/regrade.sh: a fresh container of the task image in the separate verifier's layout, the
reference's submission from solution/solve.sh, a one-episode plan); a candidate is kept when the reference completes
its kit (kit_complete 1, no abort, no fault, no budget cut); the hidden set (tests/ref/episodes.json) is the first
SET_SIZE kept candidates in drawn order. Should fewer be kept, more candidates are drawn the same way, appended and
run. Every drop is recorded with its reason; the hollowing check compares the kept kits' features with all
candidates'.

When a file the reference or the grader runs changes, the rule runs again as an AMENDMENT (selection.json
"amendments": what changed, the new sha256 of the frozen files, committed before any run on them; then its
selection_run, hidden_set and anchor_verify): on the same candidates when the kits themselves are unchanged (the
candidates in force: those of the latest amendment that drew its own), on fresh os.urandom candidates of its own
(`amend --fresh N`) when the change alters what a kit is. The latest amendment's set is the hidden set.

The commissioning service's development kits (environment/commission_service/kits.json) are chosen by the same rule
from their own pre-registered candidates (an amendment's "development_kits": kit seed seed0 + i, layout seed
layout_seed0 + i, in order): the first DEV_SIZE the reference completes. They are disjoint from the hidden candidates.

    # once, before anything runs (host, standard library): the candidates and the frozen files' sha256
    python3 -B tests/review/select_set.py draw --n 16 --frozen-commit <sha>
    # an amendment, before anything runs on the changed files (host): its frozen sha256 and development candidates
    python3 -B tests/review/select_set.py amend --what "<what changed>" --dev-n 24 [--fresh 16 --dev-seed0 8001]
    # inside the task image (it draws each kit with keyed_env.sample_kit): one tests/ copy per candidate
    docker run --rm --network none -v <task>/tests:/stage:ro -v <work>:/work --entrypoint /opt/venv/bin/python3 \\
        <image> -I -B /stage/review/select_set.py prepare --pool hidden --out /work      (or --pool dev)
    # host: each candidate once through the grader (regrade.sh), --jobs at a time
    python3 -B tests/review/select_set.py run --pool hidden --work <work> --image <image> \\
        --submission <reference submission> --jobs 6
    # host (standard library): the rule over the runs, the hollowing check; --write records the set
    python3 -B tests/review/select_set.py make --pool hidden --work <work> [--write]
    # --pool dev --write records the development kits and writes them into the service's kits.json
    # host: the reference's full verify of the set (regrade.sh on tests/ as it is), recorded as the anchor's source
    bash tests/review/regrade.sh <image> <name> <reference submission> <out>
    python3 -B tests/review/select_set.py anchor --out <out> --how "<where and how>"
"""

import argparse
import glob
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

sys.dont_write_bytecode = True

HERE = os.path.dirname(os.path.abspath(__file__))
TESTS = os.path.dirname(HERE)
TASK = os.path.dirname(TESTS)
SELECTION = os.path.join(TESTS, "hidden", "selection.json")
EPISODES = os.path.join(TESTS, "ref", "episodes.json")
KITS = os.path.join(TASK, "environment", "commission_service", "kits.json")
SET_SIZE = 10
DEV_SIZE = 16                    # the commissioning service's development kits
DEV_KIT_SEED0 = 7001             # development candidate i of the first amendment: kit seed 7001 + i,
DEV_LAYOUT_SEED0 = 11001         # layout seed 11001 + i (a later amendment names its own: --dev-seed0)
FROZEN = ("solution/solve.sh", "solution/solver/oracle_ctrl.py", "solution/solver/oracle_main.py",
          "tests/keyed_env.py", "tests/keyed_credit.py", "tests/verify.py", "tests/grader_config.json",
          "tests/nonet_exec.py", "tests/test.sh", "tests/oracle_token.sha256",
          "tests/keyed_render.py", "tests/keyed_look.json",        # the cameras' renderer and its look
          "environment/Dockerfile", "environment/requirements.txt",
          "environment/assets.sha256")                             # the render bundle's pinned files
EPISODE_LINE = re.compile(r"\[grader\] episode 0 \((?P<id>[^)]+)\): kit_complete=(?P<kit>\d) nuts_assembled=(?P<nuts>\d) "
                          r"progress=(?P<prog>[0-9.]+) steps=(?P<steps>\d+) controller_s=(?P<cs>[0-9.]+) (?P<reason>.*)$")


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def frozen_hashes():
    out = {rel: sha256_file(os.path.join(TASK, rel)) for rel in FROZEN}
    for p in sorted(glob.glob(os.path.join(TESTS, "robotics_grader", "*.py"))):
        out["tests/robotics_grader/" + os.path.basename(p)] = sha256_file(p)
    return out


def load_selection():
    with open(SELECTION) as f:
        return json.load(f)


def save_kits(kits):
    """kits.json: the window's settings one per line, then one line per development kit."""
    head = {k: v for k, v in kits.items() if k != "kits"}
    text = json.dumps(head, indent=1)[:-2] + ',\n "kits": [\n' + ",\n".join(
        "  " + json.dumps(k) for k in kits["kits"]) + "\n ]\n}\n"
    assert json.loads(text) == kits
    with open(KITS, "w") as f:
        f.write(text)


def max_steps():
    """keyed_env.MAX_STEPS, read from the source (this tool's host commands import nothing of the task)."""
    with open(os.path.join(TESTS, "keyed_env.py")) as f:
        src = f.read()
    return int(re.search(r"^MAX_STEPS = (\d+)", src, re.M).group(1))


def latest(sel):
    """The record the current set lives in: the latest amendment, or the selection itself."""
    return sel["amendments"][-1] if sel.get("amendments") else sel


def hidden_candidates(sel):
    """The hidden candidates in force: those of the latest amendment that drew fresh ones of its own (an amendment
    that leaves the kits as they are runs on the candidates in force before it), else the selection's."""
    for am in reversed(sel.get("amendments", [])):
        if am.get("candidates"):
            return am["candidates"]
    return sel["candidates"]


def pool_candidates(sel, pool):
    """The candidates in force: the development kits' of the latest amendment; for the hidden set see
    `hidden_candidates`."""
    if pool == "dev":
        return latest(sel)["development_kits"]["candidates"]
    return hidden_candidates(sel)


def save_selection(sel):
    with open(SELECTION, "w") as f:
        json.dump(sel, f, indent=1)
        f.write("\n")


# ---------------------------------------------------------------------------------------------------------- draw
def cmd_draw(a):
    sel = load_selection() if os.path.exists(SELECTION) else {}
    if sel.get("candidates") and not a.append:
        sys.exit("candidates already drawn; use --append to add a batch under the rule")
    own = bool(sel.get("amendments")) and "candidates" in sel["amendments"][-1]   # the latest amendment's own
    cands = list(sel["amendments"][-1]["candidates"] if own else sel.get("candidates", []))
    prefix = cands[0]["id"][0] if cands else "C"
    for _ in range(a.n):
        kit_seed = int.from_bytes(os.urandom(8), "little") >> 1          # 63-bit
        layout_seed = int.from_bytes(os.urandom(8), "little") >> 1
        cands.append({"i": len(cands), "id": "%s%02d" % (prefix, len(cands)), "kit_seed": kit_seed,
                      "layout_seed": layout_seed})
    if own:
        sel["amendments"][-1]["candidates"] = cands
    else:
        sel["candidates"] = cands
    if not a.append:
        sel["frozen_commit"] = a.frozen_commit
        sel["frozen"] = frozen_hashes()
    save_selection(sel)
    print("%d candidates in %s" % (len(cands), SELECTION))


# ---------------------------------------------------------------------------------------------------------- amend
def cmd_amend(a):
    """Pre-register an amendment: what changed, the frozen files' new sha256, the development candidates."""
    import datetime
    sel = load_selection()
    before = latest(sel)["frozen"]
    now = frozen_hashes()
    changed = sorted(k for k in set(before) | set(now) if before.get(k) != now.get(k))
    seed0 = a.dev_seed0
    layout0 = seed0 + (DEV_LAYOUT_SEED0 - DEV_KIT_SEED0)
    dev = [{"i": i, "id": "%s%02d" % (a.dev_prefix, i + 1), "kit_seed": seed0 + i, "layout_seed": layout0 + i}
           for i in range(a.dev_n)]
    fresh = []
    for i in range(a.fresh):                                  # this amendment's own hidden candidates
        fresh.append({"i": i, "id": "%s%02d" % (a.fresh_prefix, i),
                      "kit_seed": int.from_bytes(os.urandom(8), "little") >> 1,
                      "layout_seed": int.from_bytes(os.urandom(8), "little") >> 1})
    every = list(sel["candidates"]) + [c for am in sel.get("amendments", []) for c in am.get("candidates", [])] + fresh
    hidden_seeds = {c["kit_seed"] for c in every} | {c["layout_seed"] for c in every}
    assert len(hidden_seeds) == 2 * len(every), "a seed was drawn twice"
    assert not hidden_seeds & ({d["kit_seed"] for d in dev} | {d["layout_seed"] for d in dev})
    if a.replace:
        last = sel["amendments"][-1]
        assert "selection_run" not in last and "selection_run" not in last["development_kits"], "it has runs"
        sel["amendments"].pop()
        before = latest(sel)["frozen"]
        changed = sorted(k for k in set(before) | set(now) if before.get(k) != now.get(k))
    sel.setdefault("amendments", []).append({
        "amended_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "what": a.what,
        "rule": ("Pre-registered before any candidate ran on the amended files: this amendment (what changed, the "
                 "frozen files' sha256, the candidates, the development candidates) was committed first. The "
                 "selection's own rule runs again, unchanged, on %s: the frozen reference runs each ONCE "
                 "through the grader itself (tests/review/regrade.sh, a fresh container of the amended task image, a "
                 "one-episode plan); a candidate is kept when the reference completes its kit (kit_complete 1, no "
                 "abort, no fault, no controller-budget cut), now within the amended episode length; the hidden set "
                 "is the first %d kept in drawn order; should fewer than %d be kept, 8 more candidates are drawn "
                 "from os.urandom, appended and run. The anchor is the reference's full verify of that set, which "
                 "must repeat every kept run (kit_complete, nuts assembled, steps)." % (
                     ("%d fresh candidates of this amendment (kit and layout seeds from os.urandom, 63-bit, in the "
                      "order listed under its `candidates`; the amendment changes what a kit is, so the earlier "
                      "candidates' runs say nothing about the new kits)" % len(fresh)) if fresh else
                     "the same %d candidates in the same order (those in force before this amendment: the kits "
                     "themselves are unchanged)" % len(hidden_candidates(sel)), SET_SIZE, SET_SIZE)),
        "frozen": now,
        "changed": changed,
        "development_kits": {
            "rule": ("The commissioning service's development kits (environment/commission_service/kits.json) are "
                     "chosen like the hidden set, from their own candidates: candidate i (id %s01 ...) is "
                     "keyed_env.sample_kit(numpy.random.default_rng(%d + i), id) with layout and noise seed %d + i, "
                     "in order; the frozen reference runs each ONCE through the grader; the development kits are the "
                     "first %d it completes, in order (should fewer be kept, the next seeds follow). No development "
                     "seed is a hidden candidate's, and no development kit has a hidden candidate's grooves." % (
                         a.dev_prefix, seed0, layout0, DEV_SIZE)),
            "seed0": seed0, "layout_seed0": layout0,
            "candidates": dev,
        },
    })
    if fresh:
        sel["amendments"][-1]["candidates"] = fresh
    save_selection(sel)
    print("amendment %d pre-registered: %d frozen files changed (%s), %d fresh candidates, %d development candidates" % (
        len(sel["amendments"]), len(changed), ", ".join(changed), len(fresh), len(dev)))


# ---------------------------------------------------------------------------------------------------------- prepare
def kit_features(kit, layout, unit=None):
    """The hollowing check's features of one candidate: per kit, from its three nuts, its layout and (with
    tolerances) the actual values its episode draws."""
    import math
    nuts = kit["nuts"]
    turns = [abs(int(k)) for sp in nuts for k in sp["groove"]["turns"]]
    same = sum(1 for sp in nuts if (sp["groove"]["turns"][0] > 0) == (sp["groove"]["turns"][1] > 0))
    net = [abs(int(sp["groove"]["turns"][0]) + int(sp["groove"]["turns"][1])) for sp in nuts]
    base = (-0.56, 0.0)
    reach = max(math.hypot(x - base[0], y - base[1]) for (x, y, _yaw) in layout["pegs"] + layout["nuts"])
    gaps = [math.hypot(p[0] - q[0], p[1] - q[1]) for i, p in enumerate(layout["pegs"]) for q in layout["pegs"][i + 1:]]
    real = {}
    if unit:
        walls = [abs(float(w)) for n in unit["nuts"] for w in n["wall"]]
        cams = [abs(float(v)) for c in unit["cameras"].values() for v in c["pos"]]
        real = {"pad_friction": round(float(unit["pad_friction"]), 4),
                "joint_friction_scale_mean": round(sum(unit["joint_friction"]) / 7.0, 4),
                "wall_offset_abs_mean_deg": round(sum(walls) / len(walls), 4),
                "camera_offset_abs_mean_mm": round(1000.0 * sum(cams) / len(cams), 4)}
    return dict(real, **{"turn_sectors_mean": round(sum(turns) / len(turns), 3),
            "turns_of_4": sum(1 for t in turns if t == 4),
            "nuts_turning_one_way_twice": same,
            "net_turn_sectors_max": max(net),
            "dead_end_sectors": sum(int(d) for sp in nuts for d in sp["groove"]["dead_ends"]),
            "false_entrances": sum(len(sp["groove"]["false_entrances"]) for sp in nuts),
            "mass_mean": round(sum(float(sp["mass"]) for sp in nuts) / len(nuts), 4),
            "friction_mean": round(sum(float(sp["friction"]) for sp in nuts) / len(nuts), 4),
            "max_reach_m": round(reach, 4),
            "min_peg_gap_m": round(min(gaps), 4)})


def cmd_prepare(a):
    import numpy as np
    sys.path.insert(0, TESTS)
    import keyed_env as E
    sel = load_selection()
    cands = pool_candidates(sel, a.pool)
    idx = a.index if a.index else [c["i"] for c in cands]
    os.makedirs(a.out, exist_ok=True)
    for i in idx:
        c = cands[i]
        kit = E.sample_kit(np.random.default_rng(int(c["kit_seed"])), c["id"])
        seed = int(c["layout_seed"]) & (2 ** 64 - 1)
        layout = E.sample_layout(np.random.default_rng([seed & 0xFFFFFFFF, seed >> 32, 7]), E.NUTS)
        unit = E.sample_unit(np.random.default_rng([seed & 0xFFFFFFFF, seed >> 32, 13]), E.NUTS)
        dst = os.path.join(a.out, "tests_%02d" % i)
        shutil.rmtree(dst, ignore_errors=True)
        shutil.copytree(TESTS, dst, ignore=shutil.ignore_patterns("__pycache__", "hidden", "review"))
        ep = {"id": c["id"], "kit": kit, "seed": int(c["layout_seed"])}
        with open(os.path.join(dst, "ref", "episodes.json"), "w") as f:
            json.dump({"note": "selection candidate %s (one-episode plan)" % c["id"], "episodes": [ep]}, f, indent=1)
        with open(os.path.join(a.out, "cand_%02d.json" % i), "w") as f:
            json.dump({"i": i, "episode": ep, "features": kit_features(kit, layout, unit)}, f, indent=1)
        print("prepared candidate %d (%s)" % (i, c["id"]), flush=True)


# ---------------------------------------------------------------------------------------------------------- run
def cmd_run(a):
    sel = load_selection()
    idx = a.index if a.index else [c["i"] for c in pool_candidates(sel, a.pool)]
    regrade = os.path.join(HERE, "regrade.sh")

    def one(i):
        out = os.path.join(a.work, "run_%02d" % i)
        if os.path.exists(os.path.join(out, "regrade.log")) and not a.force:
            return i, "kept earlier run"
        r = subprocess.run(["bash", regrade, a.image, "%s-%02d" % (a.name, i), a.submission, out,
                            os.path.join(a.work, "tests_%02d" % i)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           universal_newlines=True)
        return i, "rc=%d" % r.returncode

    with ThreadPoolExecutor(max_workers=a.jobs) as pool:
        for i, what in pool.map(one, idx):
            print("candidate %2d: %s" % (i, what), flush=True)


# ---------------------------------------------------------------------------------------------------------- make
def read_run(work, i):
    rec = {"i": i}
    with open(os.path.join(work, "cand_%02d.json" % i)) as f:
        cand = json.load(f)
    rec["id"] = cand["episode"]["id"]
    rec["features"] = cand["features"]
    out = os.path.join(work, "run_%02d" % i)
    m = None
    try:
        with open(os.path.join(out, "verify.stderr")) as f:
            for line in f:
                m = EPISODE_LINE.search(line.strip()) or m
        with open(os.path.join(out, "verdict.json")) as f:
            verdict = json.load(f)
        with open(os.path.join(out, "regrade.log")) as f:
            wall = re.findall(r"wall_s=(\d+)", f.read())
    except OSError as exc:
        rec["error"] = "no complete run: %s" % exc
        return rec, cand
    if m is None:
        rec["error"] = "no episode line in the grader's log"
        return rec, cand
    rec.update({"kit_complete": int(m.group("kit")), "nuts_assembled": int(m.group("nuts")),
                "progress": float(m.group("prog")), "steps": int(m.group("steps")),
                "controller_s": float(m.group("cs")), "reason": m.group("reason"),
                "reference_token": verdict["metrics"].get("reference_token"),
                "aborted": verdict["metrics"].get("episodes_aborted"), "sim_faults": verdict["metrics"].get("sim_faults"),
                "budget_cuts": verdict["metrics"].get("controller_budget_cuts"),
                "wall_s": int(wall[-1]) if wall else None})
    return rec, cand


def kept(rec):
    """The pre-registered rule: the reference completes the candidate's kit."""
    why = []
    if "error" in rec:
        return False, [rec["error"]]
    if rec["reference_token"] != 1.0:
        why.append("the reference's token was not accepted")
    if rec["kit_complete"] != 1:
        if rec["reason"] != "ok":
            how = rec["reason"]
        elif rec["steps"] < max_steps():
            how = "the episode ended at step %d: a nut was ruined" % rec["steps"]
        else:
            how = "all %d steps run" % max_steps()
        why.append("kit not complete (%d of 3 nuts assembled; %s)" % (rec["nuts_assembled"], how))
    if rec["aborted"] or rec["sim_faults"] or rec["budget_cuts"]:
        why.append("abort, fault or budget cut")
    return not why, why


def cmd_make(a):
    sel = load_selection()
    cand_list = pool_candidates(sel, a.pool)
    size = DEV_SIZE if a.pool == "dev" else SET_SIZE
    n = len(cand_list)
    rows, cands, kept_ids = [], {}, []
    for c in cand_list:
        rec, cand = read_run(a.work, c["i"])
        assert cand["episode"]["id"] == c["id"] and cand["episode"]["seed"] == c["layout_seed"], c["id"]
        ok, why = kept(rec)
        rec.update({"kept": ok, "drop_reasons": why})
        rows.append(rec)
        cands[c["id"]] = cand
        if ok:
            kept_ids.append(c["id"])
    chosen = kept_ids[:size]
    if len(chosen) < size:
        print("only %d kept: more candidates are needed (hidden: draw --append; dev: the next seeds) and run" % len(chosen))
    cut = max((r["i"] for r in rows if r["id"] in chosen), default=-1)
    dropped = ["%s (%s)" % (r["id"], "; ".join(r["drop_reasons"])) for r in rows if not r["kept"] and r["i"] < cut]
    keys = sorted(rows[0]["features"])
    ks = [r for r in rows if r["id"] in chosen]
    hollow = {k: {"kept_mean": round(sum(r["features"][k] for r in ks) / max(1, len(ks)), 4),
                  "all_mean": round(sum(r["features"][k] for r in rows) / len(rows), 4)} for k in keys}
    what = "development kits are" if a.pool == "dev" else "hidden set is"
    summary = ("%d of %d candidates completed by the reference; the %s %s (the first %d completed in drawn "
               "order); dropped before the last of them: %s" % (
                   len(kept_ids), n, what, ", ".join(chosen), size, ", ".join(dropped) or "none"))
    print(summary)
    for r in rows:
        print("  %s %s steps %s nuts %s %s" % (r["id"], "kept" if r["kept"] else "drop", r.get("steps"),
                                                r.get("nuts_assembled"), "; ".join(r["drop_reasons"])))
    print("hollowing check:", json.dumps(hollow))
    if not (a.write and len(chosen) == size):
        return
    run = {"where": a.where, "tool_sha256": sha256_file(os.path.abspath(__file__)),
           "regrade_sha256": sha256_file(os.path.join(HERE, "regrade.sh")),
           "runs": rows, "summary": summary, "hollowing_check": hollow}
    rec = latest(sel)
    if a.pool == "dev":
        rec["development_kits"]["selection_run"] = run
        rec["development_kits"]["kits"] = chosen
        save_selection(sel)
        with open(KITS) as f:
            kits = json.load(f)
        by_id = {c["id"]: c for c in cand_list}
        kits["kits"] = [{"name": cid, "kit_seed": by_id[cid]["kit_seed"], "layout_seed": by_id[cid]["layout_seed"],
                         "kit": cands[cid]["episode"]["kit"]} for cid in chosen]
        save_kits(kits)
        print("wrote %s and %s" % (KITS, SELECTION))
        return
    rec["selection_run"] = run
    rec["hidden_set"] = chosen
    save_selection(sel)
    eps = [cands[h]["episode"] for h in chosen]
    with open(EPISODES, "w") as f:
        json.dump({"note": "fixed hidden grading episodes: a kit (three nuts and the work order) drawn with "
                           "keyed_env.sample_kit from a seed, and the layout/noise seed; the first %d candidates "
                           "of tests/hidden/selection.json the reference completed, in drawn order" % SET_SIZE,
                   "episodes": eps}, f, indent=1)
        f.write("\n")
    print("wrote %s and %s" % (EPISODES, SELECTION))


ANY_EPISODE_LINE = re.compile(r"\[grader\] episode (?P<ep>\d+) \((?P<id>[^)]+)\): kit_complete=(?P<kit>\d) "
                              r"nuts_assembled=(?P<nuts>\d) progress=(?P<prog>[0-9.]+) steps=(?P<steps>\d+) "
                              r"controller_s=(?P<cs>[0-9.]+) (?P<reason>.*)$")


def cmd_anchor(a):
    """Record the reference's full verify of the set (the out folder of one regrade.sh run on tests/ as it is):
    it must repeat every kept selection run. The anchor (tests/anchor.json `value`) is its measured value."""
    import datetime
    sel = load_selection()
    rec = latest(sel)
    runs = {r["id"]: r for r in rec["selection_run"]["runs"]}
    eps = []
    with open(os.path.join(a.out, "verify.stderr")) as f:
        for line in f:
            m = ANY_EPISODE_LINE.search(line.strip())
            if m:
                eps.append({"episode": int(m.group("ep")), "id": m.group("id"), "kit_complete": int(m.group("kit")),
                            "nuts_assembled": int(m.group("nuts")), "steps": int(m.group("steps")),
                            "controller_s": float(m.group("cs"))})
    with open(os.path.join(a.out, "verdict.json")) as f:
        verdict = json.load(f)
    with open(os.path.join(a.out, "regrade.log")) as f:
        log = f.read()
    wall = re.findall(r"wall_s=(\d+)", log)
    image = re.findall(r"image=(\S+) \((sha256:[0-9a-f]+)\)", log)
    assert [e["id"] for e in eps] == rec["hidden_set"], "the verify did not play the hidden set in order"
    repeats = all((e["kit_complete"], e["nuts_assembled"], e["steps"]) ==
                  (runs[e["id"]]["kit_complete"], runs[e["id"]]["nuts_assembled"], runs[e["id"]]["steps"]) for e in eps)
    met = verdict["metrics"]
    assert met["reference_token"] == 1.0 and not met["episodes_aborted"] and not met["sim_faults"]
    rec["anchor_verify"] = {
        "date": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d"),
        "how": a.how,
        "image": image[-1][1] if image else None,
        "wall_s": int(wall[-1]) if wall else None,
        "measured": met["kit_success_rate"],
        "episodes": eps,
        "every_episode_repeats_its_selection_run": repeats,
        "probes": {"total": met["probes_total"], "answers_that_differed": met["probes_passed"],
                   "note": "the reference acts on _gt's real proprioception, so it answers both copies of a repeated "
                           "step alike; the probe record never gates"},
    }
    if a.pre_registered_commit:
        rec["pre_registered_commit"] = a.pre_registered_commit
    save_selection(sel)
    print("anchor verify: measured %s, %d episodes, repeats its selection runs: %s" % (
        met["kit_success_rate"], len(eps), repeats))


def cmd_provisional(a):
    """Before the development candidates have run: the service's kits.json lists the first DEV_SIZE candidates
    (their kits as `prepare --pool dev` drew them), so that the image builds; `make --pool dev --write` replaces
    them with the first DEV_SIZE the reference completes."""
    sel = load_selection()
    cand_list = pool_candidates(sel, "dev")[:DEV_SIZE]
    with open(KITS) as f:
        kits = json.load(f)
    out = []
    for c in cand_list:
        with open(os.path.join(a.work, "cand_%02d.json" % c["i"])) as f:
            cand = json.load(f)
        assert cand["episode"]["id"] == c["id"] and cand["episode"]["seed"] == c["layout_seed"], c["id"]
        out.append({"name": c["id"], "kit_seed": c["kit_seed"], "layout_seed": c["layout_seed"], "kit": cand["episode"]["kit"]})
    kits["kits"] = out
    save_kits(kits)
    print("wrote %s: the first %d development candidates (provisional)" % (KITS, len(out)))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("draw")
    d.add_argument("--n", type=int, default=16)
    d.add_argument("--frozen-commit", default=None)
    d.add_argument("--append", action="store_true")
    am = sub.add_parser("amend")
    am.add_argument("--what", required=True)
    am.add_argument("--dev-n", type=int, default=24)
    am.add_argument("--dev-seed0", type=int, default=DEV_KIT_SEED0, help="kit seed of development candidate 0")
    am.add_argument("--dev-prefix", default="D")
    am.add_argument("--fresh", type=int, default=0, help="draw this many fresh hidden candidates for the amendment")
    am.add_argument("--fresh-prefix", default="E")
    am.add_argument("--replace", action="store_true", help="replace the latest amendment (only before it has runs)")
    p = sub.add_parser("prepare")
    p.add_argument("--pool", choices=("hidden", "dev"), default="hidden")
    p.add_argument("--out", required=True)
    p.add_argument("--index", type=int, nargs="*")
    pv = sub.add_parser("provisional")
    pv.add_argument("--work", required=True)
    r = sub.add_parser("run")
    r.add_argument("--pool", choices=("hidden", "dev"), default="hidden")
    r.add_argument("--work", required=True)
    r.add_argument("--image", required=True)
    r.add_argument("--submission", required=True)
    r.add_argument("--jobs", type=int, default=4)
    r.add_argument("--name", default="keyed-sel")
    r.add_argument("--index", type=int, nargs="*")
    r.add_argument("--force", action="store_true")
    m = sub.add_parser("make")
    m.add_argument("--pool", choices=("hidden", "dev"), default="hidden")
    m.add_argument("--work", required=True)
    m.add_argument("--write", action="store_true")
    m.add_argument("--where", default="")
    an = sub.add_parser("anchor")
    an.add_argument("--out", required=True, help="the out folder of regrade.sh's full verify of the set")
    an.add_argument("--how", required=True)
    an.add_argument("--pre-registered-commit", default=None)
    a = ap.parse_args()
    {"draw": cmd_draw, "amend": cmd_amend, "prepare": cmd_prepare, "provisional": cmd_provisional, "run": cmd_run,
     "make": cmd_make, "anchor": cmd_anchor}[a.cmd](a)


if __name__ == "__main__":
    main()
