"""Shift candidates and development shifts from the task's generator (a maintainer's tool; the grader never runs it).

    # candidate lists for a study (never the hidden set: select_hidden.py draws it by its registered rule):
    # labelled-hash seeds with a master draw from OS entropy, recorded
    python3 -B tests/review/gen_shifts.py candidates --out DIR --n 40 [--label study] [--master auto|<hex>]
                                                     [--replay --jobs 6]
    # the development shifts of the commissioning service, by their coverage rule
    python3 -B tests/review/gen_shifts.py dev --out DIR [--replay]

Seeds. ``seed_k = parcelcell.shiftgen.labelled_seed(label, k, mix=master)``: the first 8 bytes of
sha256("dual_arm_parcel_sorting_on_conveyor/<label>/v1/<k>" + "/" + master) >> 1. The labels "hidden", "dev",
"review" and "docs" give disjoint candidate lists by construction (the hidden set's slot labels "hidden-1" ...
"hidden-8" with their own salt are select_hidden.py's). ``--master auto`` draws 16 bytes from
``os.urandom`` once and records them (hex) in ``candidates.json``, so the list is reproducible from the record and
not predictable before the draw; the development shifts use no master (their list is fixed by the label alone).

Classes. Each candidate records its belt-speed class, its mix class (the per-shift chance of a soft mailer) and its
features (``shiftgen.shift_features``): the quotas of the selection and the hollowing check use them.

Never queue. The generator spaces arrivals in time (one minimum gap for every parcel, never shorter than the entry's
clearance) so that a shift's stated arrival times are its real ones. ``--replay`` checks it: every shift is replayed in the cell with the robot at its start posture (no frame
rendered) and must show no entry delay, no robot contact, nothing lost and credit 0; a shift that fails is a
generator fault (reported, never silently dropped).

Development rule (written before any draw). Seeds: ``labelled_seed("dev", k, mix=salt)`` with the development set's
own salt (16 bytes drawn once from ``os.urandom`` and recorded in ``sites.json`` as ``salt_hex``; the label "dev" and
the salt are shared with nothing the hidden set uses). Bench shift b (b = 1, 2, 3) is the first candidate
k = 1000 b, 1000 b + 1, ... drawn with the bench profile (the slowest belt class) and the stated number of parcels whose
parcels are exactly the stated contents (``DEV_BENCH_CONTENTS``, any order): bench 1 one carton with its label
underneath; bench 2 one carton with its label on the downstream face and one label-up carton; bench 3 one carton of
the station's duty with its label not up, one mailer and one carton with its label on a side face (near or far). The 5
full shifts are the first 5 (in k order, k = 4000, 4001, ...) full shifts with at least ``DEV_FULL_MIN_W`` (4) parcels
of the duty arriving label not up (class W) that together show every belt-speed class, both kinds of parcel and every
label position at least once (``quota.first_subset``).

Output: ``DIR/candidates.json`` (or ``DIR/dev_shifts.json``) and the shift packages ``DIR/shifts/*.json``.

The line setting (the commissioning service's shifts and the graded set; ``parcelcell.shiftgen.draw_line_shift``):

    python3 -B tests/review/gen_shifts.py line-dev --out DIR [--salt <hex>]
    python3 -B tests/review/gen_shifts.py line-candidates --out DIR --n 40 [--label study] [--master auto|<hex>]

Line development rule (written before any draw). Seeds: ``shiftgen.line_seed("line-dev", k, mix=salt)`` with the
line development set's own salt (16 bytes drawn once from ``os.urandom``, recorded in ``sites.json`` as
``salt_hex``; the label and the salt are shared with nothing else). Bench shift b (b = 1, 2, 3; the bench profile,
its list = the stated number of parcels) is the first candidate k = 1000 b, 1000 b + 1, ... drawn with its stated
belt class and belt mode whose list is exactly the stated contents (``LINE_BENCH``, any order): bench 1 (slow,
running) one rigid parcel with its label underneath; bench 2 (medium, running) one soft parcel with its label
underneath and one rigid parcel with its label on a side face; bench 3 (slow, indexing) a flat rigid parcel
(height below ``FLAT_M``) with its label not up, one parcel arriving label up and one soft parcel with its label not
up. Full shift j (j = 1 .. 5) is the first draw, k = 4000 + j, with its stated belt class and mode
(``LINE_FULL``: slow running, medium running, fast running, slow indexing, fast indexing); nothing is rejected.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.dont_write_bytecode = True

import common  # noqa: E402
from quota import first_subset  # noqa: E402

GEN_FILES = ("parcelcell/shiftgen.py", "parcelcell/stream.py", "parcelcell/_geom.py")


def sha256_file(p):
    with open(p, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def generator_digest(cfg):
    out = {f: sha256_file(os.path.join(common.TESTS, f)) for f in GEN_FILES}
    blk = json.dumps({"shifts": cfg["shifts"], "world": cfg["world"]}, sort_keys=True).encode()
    out["config_shifts_and_world_sha256"] = hashlib.sha256(blk).hexdigest()
    return out


def draw(label, k, cfg, master=None, profile="full", n_parcels=None):
    from parcelcell.shiftgen import draw_shift, labelled_seed, package_sha256, to_package
    seed = labelled_seed(label, k, mix=master)
    sh, meta = draw_shift(seed, cfg, profile=profile, label=label, k=k, n_parcels=n_parcels)
    pkg = to_package(sh, meta)
    return {"k": k, "seed": seed, "profile": profile, "sha256": package_sha256(pkg),
            "belt_class": meta["belt_class"], "mix_class": meta["mix_class"], "features": meta["features"],
            "duration_s": sh.duration}, pkg


def _replay(args):
    path, = args
    from parcelcell.shiftgen import load_package
    sh, meta, _ = load_package(path)
    srv = common.make_server(render=False)
    t0 = time.time()
    srv.reset(path)
    q0 = srv.describe()["robot"]["start_posture"]
    st = srv.handle("step", {"q_target": q0, "grip": [0.0, 0.0], "n": 1})
    while not st["shift_over"]:
        st = srv.handle("step", {"n": 50})
    rec = srv.finalize()
    touched = sum(1 for p in rec["parcels"] if p["touch_ticks"])
    out = {"file": os.path.basename(path), "max_entry_delay_s": rec["max_entry_delay_s"], "lost": rec["lost"],
           "credit": rec["credit"], "n_W": rec["n_W"], "touched_by_robot": touched,
           "parcel_touching_at_scan": sum(1 for p in rec["parcels"] if p["scan"] and p["scan"]["touching_parcel"]),
           "stops": rec["safety"]["protective_stop_count"], "wall_s": round(time.time() - t0, 1)}
    out["ok"] = (out["max_entry_delay_s"] == 0 and out["lost"] == 0 and out["credit"] == 0 and touched == 0
                 and out["stops"] == 0)
    srv.close()
    return out


def replay_all(paths, jobs):
    import multiprocessing as mp
    with mp.get_context("fork").Pool(min(jobs, 8), maxtasksperchild=4) as pool:
        return list(pool.imap(_replay, [(p,) for p in paths]))


def cmd_candidates(a):
    cfg = common.load_cfg()
    if a.master == "auto":
        master = os.urandom(16)
        src = "os.urandom(16), drawn %s" % time.strftime("%Y-%m-%d %H:%M:%S %Z")
    elif a.master in ("", "none"):
        master, src = None, "none"
    else:
        master, src = bytes.fromhex(a.master), "given"
    os.makedirs(os.path.join(a.out, "shifts"), exist_ok=True)
    from parcelcell.shiftgen import package_bytes
    cands = []
    for k in range(a.n):
        c, pkg = draw(a.label, k, cfg, master=master)
        c["file"] = "shifts/%s_%03d.json" % (a.label, k)
        with open(os.path.join(a.out, c["file"]), "wb") as fh:
            fh.write(package_bytes(pkg))
        cands.append(c)
    rec = {"tool": "gen_shifts.py candidates", "tool_sha256": sha256_file(os.path.abspath(__file__)),
           "label": a.label, "seed_rule": "labelled_seed(label, k, mix=master): sha256('<task id>/<label>/v1/<k>' + "
                                         "'/' + master)[:8] >> 1",
           "master_hex": None if master is None else master.hex(), "master_source": src,
           "generator": generator_digest(cfg), "candidates": cands}
    if a.replay:
        res = replay_all([os.path.join(a.out, c["file"]) for c in cands], a.jobs)
        for c, r in zip(cands, res):
            c["replay"] = r
        rec["replay_ok"] = all(r["ok"] for r in res)
        bad = [c["k"] for c in cands if not c["replay"]["ok"]]
        print("never-queue replay: %d of %d shifts clean%s" % (len(cands) - len(bad), len(cands),
                                                              "; generator faults: %s" % bad if bad else ""))
    common.dump_json(os.path.join(a.out, "candidates.json"), rec)
    for c in cands:
        f = c["features"]
        print("k=%3d %-6s %-8s v=%.3f n=%2d W=%2d soft=%2d dur=%4.0f s%s" % (
            c["k"], c["belt_class"], c["mix_class"], f["belt_speed"], f["n_parcels"], f["n_label_not_up"],
            f["n_soft"], c["duration_s"], "" if "replay" not in c else ("  replay ok" if c["replay"]["ok"] else
                                                                        "  REPLAY FAULT %s" % c["replay"])))
    return 0


DEV_BENCH = 3
DEV_FULL = 5
DEV_FULL_MIN_W = 4
DEV_BENCH_K0 = 1000          # bench shift b draws k = 1000 b, 1000 b + 1, ...
DEV_FULL_K0 = 4000           # the full shifts' candidates: k = 4000, 4001, ...
DEV_SEARCH = 2000            # candidates looked at per bench shift before the rule is called unmet
# what each bench shift holds, any order: (kind, label positions it may have; None = any)
DEV_BENCH_CONTENTS = (
    (("rigid", ("underneath",)),),
    (("rigid", ("downstream",)), ("rigid", ("top",))),
    (("rigid", "duty"), ("soft", None), ("rigid", ("near_side", "far_side"))),
)
DEV_BENCH_TEXT = ("bench 1: one carton with its label underneath; bench 2: one carton with its label on the "
                  "downstream face and one label-up carton; bench 3: one carton of the station's duty with its label "
                  "not up, one mailer and one carton with its label on a side face")


def dev_salt(path=None):
    """The development set's salt (bytes) from sites.json (``salt_hex``), or None when it has none yet."""
    p = path or os.path.join(os.path.dirname(common.TESTS), "environment", "commission_service", "sites.json")
    if not os.path.isfile(p):
        return None
    hx = json.load(open(p)).get("salt_hex")
    return bytes.fromhex(hx) if hx else None


def dev_quotas(cfg):
    q = [("class:%s" % c, 1) for c in cfg["shifts"]["belt_speed_classes"]]
    q += [("kind:rigid", 1), ("kind:soft", 1)]
    from parcelcell.config import LABEL_POSITIONS
    q += [("position:%s" % p, 1) for p in LABEL_POSITIONS]
    return q


def contrib_dev(c):
    f = c["features"]
    d = {"class:%s" % c["belt_class"]: 1, "kind:soft": f["n_soft"], "kind:rigid": f["n_parcels"] - f["n_soft"]}
    for pos, n in f["label_positions"].items():
        d["position:%s" % pos] = n
    return d


def bench_matches(pkg, want, cfg):
    """True when the shift's parcels are exactly ``want`` (DEV_BENCH_CONTENTS[b]), in any order."""
    from itertools import permutations
    from parcelcell.episode import label_position
    from parcelcell.stream import Parcel
    duty = (cfg.get("score") or {}).get("duty") or {}
    got = []
    for a in pkg["arrivals"]:
        p = Parcel(**a["parcel"])
        got.append((p.kind, label_position(p)))
    if len(got) != len(want):
        return False

    def ok(g, w):
        kind, places = w
        if g[0] != kind:
            return False
        if places is None:
            return True
        if places == "duty":
            return g[0] in duty.get("kinds", ()) and g[1] in duty.get("label_positions", ())
        return g[1] in places

    return any(all(ok(g, w) for g, w in zip(got, perm)) for perm in permutations(want))


def dev_rule(cfg, salt):
    """(bench [(c, pkg)], full [(c, pkg)], skipped full k) of the development rule (module text)."""
    bench = []
    for b, want in enumerate(DEV_BENCH_CONTENTS, 1):
        for j in range(DEV_SEARCH):
            k = DEV_BENCH_K0 * b + j
            c, pkg = draw("dev", k, cfg, master=salt, profile="bench", n_parcels=len(want))
            if bench_matches(pkg, want, cfg):
                bench.append((c, pkg))
                break
        else:
            sys.exit("bench shift %d: no candidate meets the stated contents within %d" % (b, DEV_SEARCH))
    pool = []
    for k in range(DEV_FULL_K0, DEV_FULL_K0 + 200):
        c, pkg = draw("dev", k, cfg, master=salt, profile="full")
        if c["features"]["classes"].get("W", 0) >= DEV_FULL_MIN_W:
            c["contrib"] = contrib_dev(c)
            pool.append((c, pkg))
    chosen = first_subset([c for c, _ in pool], DEV_FULL, dev_quotas(cfg))
    if chosen is None:
        sys.exit("no development set meets the coverage rule within 200 candidates")
    keep = {c["k"] for c in chosen}
    full = [(c, p) for c, p in pool if c["k"] in keep]
    for c, _ in full:
        c.pop("contrib", None)
    skipped = [k for k in range(DEV_FULL_K0, max(keep)) if k not in keep]
    return bench, full, skipped


def cmd_dev(a):
    cfg = common.load_cfg()
    salt = bytes.fromhex(a.salt) if a.salt else dev_salt()
    if salt is None:
        sys.exit("no development salt: give --salt <32 hex> (16 bytes from os.urandom, drawn once)")
    os.makedirs(os.path.join(a.out, "shifts"), exist_ok=True)
    from parcelcell.shiftgen import package_bytes
    bench, full, skipped = dev_rule(cfg, salt)
    sites = []
    for c, pkg in bench + full:
        c["file"] = "shifts/dev_%04d.json" % c["k"]
        with open(os.path.join(a.out, c["file"]), "wb") as fh:
            fh.write(package_bytes(pkg))
        sites.append(c)
    rec = {"tool": "gen_shifts.py dev", "tool_sha256": sha256_file(os.path.abspath(__file__)),
           "rule": "the development rule of gen_shifts.py (module text)", "salt_hex": salt.hex(),
           "generator": generator_digest(cfg), "sites": [dict(c, site=i + 1) for i, c in enumerate(sites)],
           "full_skipped_k": skipped}
    if a.replay:
        res = replay_all([os.path.join(a.out, c["file"]) for c in sites], a.jobs)
        for s, r in zip(rec["sites"], res):
            s["replay"] = r
        rec["replay_ok"] = all(r["ok"] for r in res)
    common.dump_json(os.path.join(a.out, "dev_shifts.json"), rec)
    for s in rec["sites"]:
        f = s["features"]
        print("site %d k=%4d %-5s %-6s %-8s v=%.3f n=%2d W=%d U=%d X=%d soft=%2d dur=%.0f%s" % (
            s["site"], s["k"], s["profile"], s["belt_class"], s["mix_class"], f["belt_speed"], f["n_parcels"],
            f["classes"]["W"], f["classes"]["U"], f["classes"]["X"], f["n_soft"], s["duration_s"],
            "" if "replay" not in s else ("  replay ok" if s["replay"]["ok"] else "  REPLAY FAULT %s" % s["replay"])))
    print("full candidates skipped before the last kept (fewer than %d W, or not needed): %s" % (DEV_FULL_MIN_W,
                                                                                               skipped or "none"))
    return 0


# ---------------------------------------------------------------- the line setting
FLAT_M = 0.11                 # a "flat" rigid parcel: lower than this (m)
LINE_BENCH = (
    ("slow", False, (("rigid", ("underneath",), None),)),
    ("medium", False, (("soft", ("underneath",), None), ("rigid", ("upstream", "downstream", "near_side", "far_side"), None))),
    ("slow", True, (("rigid", "not_top", "flat"), (None, ("top",), None), ("soft", "not_top", None))),
)
LINE_BENCH_TEXT = ("bench 1 (slow belt, running): one rigid parcel with its label underneath; bench 2 (medium belt, "
                   "running): one soft parcel with its label underneath and one rigid parcel with its label on a side "
                   "face; bench 3 (slow belt, indexing): a flat rigid parcel with its label not up, one parcel arriving "
                   "label up and one soft parcel with its label not up")
LINE_FULL = (("slow", False), ("medium", False), ("fast", False), ("slow", True), ("fast", True))
# the graded line set's slots (belt class, indexing), in slot order: its registered rule's slot plan
LINE_SLOTS = (("slow", False), ("medium", False), ("fast", False), ("medium", True), ("fast", False), ("slow", True),
              ("medium", False), ("fast", False))


def line_draw(label, k, cfg, master=None, profile="full", belt_class=None, indexing=None, n=None):
    from parcelcell.shiftgen import draw_line_shift, line_seed, package_sha256, to_package
    seed = line_seed(label, k, cfg, mix=master)
    contents = [{}] * int(n) if (profile == "bench" and n) else None
    sh, meta = draw_line_shift(seed, cfg, profile=profile, belt_class=belt_class, indexing=indexing, label=label, k=k,
                               contents=contents)
    pkg = to_package(sh, meta)
    return {"k": k, "seed": seed, "profile": profile, "sha256": package_sha256(pkg), "belt_class": meta["belt_class"],
            "indexing": meta["indexing"], "features": meta["features"], "duration_s": sh.duration}, pkg


def line_bench_matches(pkg, want):
    from itertools import permutations
    from parcelcell.episode import label_position
    from parcelcell.stream import Parcel
    got = []
    for a in pkg["arrivals"]:
        p = Parcel(**a["parcel"])
        got.append((p.kind, label_position(p), p.size[2]))
    if len(got) != len(want):
        return False

    def ok(g, w):
        kind, places, shape = w
        if kind is not None and g[0] != kind:
            return False
        if places == "not_top":
            if g[1] == "top":
                return False
        elif places is not None and g[1] not in places:
            return False
        if shape == "flat" and not g[2] < FLAT_M:
            return False
        return True

    return any(all(ok(g, w) for g, w in zip(got, perm)) for perm in permutations(want))


def line_dev_rule(cfg, salt):
    """(bench [(c, pkg)], full [(c, pkg)]) of the line development rule (module text)."""
    bench = []
    for b, (bc, idx, want) in enumerate(LINE_BENCH, 1):
        for j in range(DEV_SEARCH):
            k = DEV_BENCH_K0 * b + j
            c, pkg = line_draw("line-dev", k, cfg, master=salt, profile="bench", belt_class=bc, indexing=idx,
                               n=len(want))
            if line_bench_matches(pkg, want):
                bench.append((c, pkg))
                break
        else:
            sys.exit("line bench shift %d: no candidate meets the stated contents within %d" % (b, DEV_SEARCH))
    full = [line_draw("line-dev", DEV_FULL_K0 + j, cfg, master=salt, profile="full", belt_class=bc, indexing=idx)
            for j, (bc, idx) in enumerate(LINE_FULL, 1)]
    return bench, full


def cmd_line_dev(a):
    cfg = common.load_cfg()
    if not a.salt:
        sys.exit("give --salt <32 hex> (16 bytes from os.urandom, drawn once)")
    salt = bytes.fromhex(a.salt)
    os.makedirs(os.path.join(a.out, "shifts"), exist_ok=True)
    from parcelcell.shiftgen import package_bytes
    bench, full = line_dev_rule(cfg, salt)
    sites = []
    for c, pkg in bench + full:
        c["file"] = "shifts/line_dev_%04d.json" % c["k"]
        with open(os.path.join(a.out, c["file"]), "wb") as fh:
            fh.write(package_bytes(pkg))
        sites.append(c)
    rec = {"tool": "gen_shifts.py line-dev", "tool_sha256": sha256_file(os.path.abspath(__file__)),
           "rule": "the line development rule of gen_shifts.py (module text)", "salt_hex": salt.hex(),
           "sites": [dict(c, site=i + 1) for i, c in enumerate(sites)]}
    common.dump_json(os.path.join(a.out, "dev_shifts.json"), rec)
    for st in rec["sites"]:
        f = st["features"]
        print("site %d k=%4d %-5s %-6s idx=%d v=%.3f n=%2d W=%d U=%d soft=%2d flat=%d" % (
            st["site"], st["k"], st["profile"], st["belt_class"], st["indexing"], f["belt_speed"], f["n_parcels"],
            f["classes"]["W"], f["classes"]["U"], f["n_soft"], f["n_flat"]))
    return 0


def cmd_line_candidates(a):
    cfg = common.load_cfg()
    if a.master == "auto":
        master = os.urandom(16)
    elif a.master in ("", "none"):
        master = None
    else:
        master = bytes.fromhex(a.master)
    os.makedirs(os.path.join(a.out, "shifts"), exist_ok=True)
    from parcelcell.shiftgen import package_bytes
    cands = []
    for k in range(a.n):
        c, pkg = line_draw(a.label, k, cfg, master=master)
        c["file"] = "shifts/%s_%03d.json" % (a.label, k)
        with open(os.path.join(a.out, c["file"]), "wb") as fh:
            fh.write(package_bytes(pkg))
        cands.append(c)
    common.dump_json(os.path.join(a.out, "candidates.json"),
                     {"tool": "gen_shifts.py line-candidates", "label": a.label,
                      "master_hex": None if master is None else master.hex(), "candidates": cands})
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("candidates")
    c.add_argument("--out", required=True)
    c.add_argument("--n", type=int, default=40)
    c.add_argument("--label", default="study")
    c.add_argument("--master", default="auto")
    c.add_argument("--replay", action="store_true")
    c.add_argument("--jobs", type=int, default=6)
    d = sub.add_parser("dev")
    d.add_argument("--out", required=True)
    d.add_argument("--replay", action="store_true")
    d.add_argument("--jobs", type=int, default=6)
    d.add_argument("--salt", default="", help="the development salt (hex); default: sites.json's salt_hex")
    ld = sub.add_parser("line-dev")
    ld.add_argument("--out", required=True)
    ld.add_argument("--salt", default="", help="the line development salt (hex)")
    lc = sub.add_parser("line-candidates")
    lc.add_argument("--out", required=True)
    lc.add_argument("--n", type=int, default=40)
    lc.add_argument("--label", default="study")
    lc.add_argument("--master", default="auto")
    a = ap.parse_args(argv)
    return {"candidates": cmd_candidates, "dev": cmd_dev, "line-dev": cmd_line_dev,
            "line-candidates": cmd_line_candidates}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
