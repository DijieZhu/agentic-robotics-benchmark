"""The venue generator's gates and the candidate pool. Lines written before the first gate run:

G1 seeds      200 random seeds (os.urandom, 16 bytes each; the cells low / mid / high x classic / featured in turn):
              every world is closed (closed: true, samples at most 0.25 m apart), its centreline does not cross
              itself, it is at least 1.0 m wide everywhere, a kinematic car (the owner's envelope, steering lock
              0.4189 rad, servo 3.2 rad/s, pure pursuit at 1 m/s) drives one lap from the start pose with all
              envelope and wheel points inside the corridor, at least 0.02 m from every duct piece, crossing the
              timing line and every checkpoint in order (checks.kinematic_lap). All 200 must pass.
              (Added after the first G1 run, before the second, together with a generator fix:
              no wheel point of that lap beyond a tape line within 3 m of the tape.)
G2 contract   every world of G1 passes check.check_world with 0 errors and its compiled venue model (the duct model's default
              duct mode, with the drive model's car; and the avoid variant with the obstacle model's boxes and slow cars) passes
              check.check_model with 0 errors (pushable=True also for the duct model's active_window: contract 1.1).
G3 table      statistics of the G1 worlds per class (length, widths, footprint, duct counts, wall per metre of
              track) beside the real numbers (checks.REAL). Reported, no line.
G4 bytes      the same seed gives byte-identical output: for 12 seeds of G1 (two per cell) the world JSON, the
              venue MJCF and the top-view PNG written twice in one process and once in a fresh process have the
              same sha256.

    python -B -m sketch.venue.gates seeds --n 200 --workers 12 --out DIR
    python -B -m sketch.venue.gates bytes --from DIR/g1_seeds.json --k 12 --out DIR
    python -B -m sketch.venue.gates table --from DIR/g1_seeds.json --out DIR
    python -B -m sketch.venue.gates candidates --per-cell 4 --out DIR
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
DEV = os.path.dirname(os.path.dirname(HERE))
CELLS = [(t, c) for c in ("classic", "featured") for t in ("low", "mid", "high")]
PY = sys.executable


def package_sha256():
    """sha256 of every file of this package (sorted by name), and of the files of the packages it builds on."""
    out = {}
    for rel in ("sketch/venue", "sketch/ducts", "sketch/others", "sketch/examples"):
        d = os.path.join(DEV, rel)
        for name in sorted(os.listdir(d)):
            p = os.path.join(d, name)
            if os.path.isfile(p) and (name.endswith(".py") or name.endswith(".json")) and not name.startswith("."):
                with open(p, "rb") as f:
                    out["%s/%s" % (rel, name)] = hashlib.sha256(f.read()).hexdigest()
    return out


def _sha(b):
    return hashlib.sha256(b).hexdigest()


def one_world(job):
    """Everything G1 and G2 measure for one seed (runs in a worker)."""
    seed, tier, cls = job
    from sketch.examples import check
    from . import checks as CK
    from . import export as EX
    from .gen_venue import dumps, generate_with_info
    t0 = time.time()
    row = {"seed_hex": seed, "tier": tier, "class": cls}
    try:
        w, info = generate_with_info(seed, tier, cls)
    except Exception as e:                                    # reported, never hidden
        row.update({"ok": False, "error": "%s: %s" % (type(e).__name__, e)})
        return row
    t1 = time.time()
    row["gen_s"] = round(t1 - t0, 3)
    row["dropped"] = info["dropped"]
    row["drop_reasons"] = info["reasons"]
    row["world_id"] = w["world_id"]
    row["layout_digest"] = w["layout_digest"]
    row["world_sha256"] = _sha(dumps(w).encode("ascii"))
    err = check.check_world(w)
    row["check_world"] = err
    g = CK.geometry_report(w)
    k = CK.kinematic_lap(w)
    row["geometry"] = g
    row["kinematic"] = k
    me, md = EX.model_errors(w, "track")
    ma, _ = EX.model_errors(w, "avoid")
    row["check_model"] = {"mode": md, "track": me, "avoid": ma}
    row["stats"] = CK.stats_row(w)
    row["check_s"] = round(time.time() - t1, 3)
    row["ok"] = bool(g["closed"] and not g["self_intersects"] and g["min_width_m"] >= 1.0 - 1e-6 and k["ok"]
                     and not err and not me and not ma)
    return row


def run_seeds(n, workers, out_dir, seeds=None):
    os.makedirs(out_dir, exist_ok=True)
    if seeds is None:
        seeds = [os.urandom(16).hex() for _ in range(n)]
    jobs = [(s, CELLS[i % 6][0], CELLS[i % 6][1]) for i, s in enumerate(seeds)]
    t0 = time.time()
    if workers > 1:
        import multiprocessing as mp
        with mp.get_context("fork").Pool(workers) as pool:
            rows = pool.map(one_world, jobs, chunksize=1)
    else:
        rows = [one_world(j) for j in jobs]
    wall = time.time() - t0
    summary = {
        "n": len(rows), "passed": sum(1 for r in rows if r.get("ok")),
        "closed": sum(1 for r in rows if r.get("geometry", {}).get("closed")),
        "not_self_intersecting": sum(1 for r in rows if "geometry" in r and not r["geometry"]["self_intersects"]),
        "min_width_ge_1": sum(1 for r in rows if r.get("geometry", {}).get("min_width_m", 0) >= 1.0 - 1e-6),
        "kinematic_ok": sum(1 for r in rows if r.get("kinematic", {}).get("ok")),
        "check_world_clean": sum(1 for r in rows if "check_world" in r and not r["check_world"]),
        "check_model_clean": sum(1 for r in rows if "check_model" in r and not r["check_model"]["track"]
                                 and not r["check_model"]["avoid"]),
        "errors": [r for r in rows if "error" in r],
        "wall_s": round(wall, 1), "workers": workers,
        "gen_s_median": _median([r["gen_s"] for r in rows if "gen_s" in r]),
        "dropped_attempts_median": _median([r["dropped"] for r in rows if "dropped" in r]),
        "dropped_attempts_max": max([r["dropped"] for r in rows if "dropped" in r] or [0]),
        "kinematic_margin_min_m": min([r["kinematic"]["corridor_margin_min_m"] for r in rows if "kinematic" in r] or [0]),
        "kinematic_duct_clear_min_m": min([r["kinematic"]["duct_clearance_min_m"] for r in rows if "kinematic" in r] or [0]),
    }
    out = {"gate": "G1/G2", "lines": __doc__, "summary": summary, "package_sha256": package_sha256(), "rows": rows}
    p = os.path.join(out_dir, "g1_seeds.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    return p, summary


def _median(v):
    v = sorted(v)
    if not v:
        return None
    m = len(v) // 2
    return v[m] if len(v) % 2 else 0.5 * (v[m - 1] + v[m])


# ------------------------------------------------------------------------------------------------ G4
def _bytes_of(seed, tier, cls, tmp):
    from . import export as EX
    from .gen_venue import dumps, generate
    w = generate(seed, tier, cls)
    xml = EX.model_xml(w, "track")
    png = os.path.join(tmp, "%s.png" % seed)
    EX.top_view(w, png)
    with open(png, "rb") as f:
        pb = f.read()
    return {"world": _sha(dumps(w).encode("ascii")), "mjcf": _sha(xml.encode("utf-8")), "png": _sha(pb)}


def run_bytes(src, k, out_dir):
    with open(src, "r", encoding="utf-8") as f:
        rows = json.load(f)["rows"]
    pick = []
    for cell in CELLS:
        pick += [r for r in rows if (r["tier"], r["class"]) == cell and r.get("ok")][:max(1, k // 6)]
    tmp = os.path.join(out_dir, "g4_tmp")
    os.makedirs(tmp, exist_ok=True)
    res = []
    for r in pick:
        a = _bytes_of(r["seed_hex"], r["tier"], r["class"], tmp)
        b = _bytes_of(r["seed_hex"], r["tier"], r["class"], tmp)
        code = ("import json,sys; sys.path.insert(0, %r); from sketch.venue.gates import _bytes_of; "
                "print(json.dumps(_bytes_of(%r, %r, %r, %r)))" % (DEV, r["seed_hex"], r["tier"], r["class"], tmp))
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
        cp = subprocess.run([PY, "-B", "-c", code], cwd=DEV, env=env, capture_output=True, text=True, check=True)
        c = json.loads(cp.stdout.strip().splitlines()[-1])
        res.append({"seed_hex": r["seed_hex"], "world_id": r["world_id"], "in_process_1": a, "in_process_2": b,
                    "fresh_process": c, "g1_world_sha256": r["world_sha256"],
                    "identical": a == b == c and a["world"] == r["world_sha256"]})
    out = {"gate": "G4", "n": len(res), "identical": sum(1 for x in res if x["identical"]), "rows": res}
    p = os.path.join(out_dir, "g4_bytes.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    return p, {"n": out["n"], "identical": out["identical"]}


# ------------------------------------------------------------------------------------------------ G3
def run_table(src, out_dir):
    from . import checks as CK
    with open(src, "r", encoding="utf-8") as f:
        rows = [r["stats"] for r in json.load(f)["rows"] if "stats" in r]
    summ = CK.summarise(rows)
    out = {"gate": "G3", "generated": summ, "real": CK.REAL}
    p = os.path.join(out_dir, "g3_table.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    return p, summ


# ------------------------------------------------------------------------------------------------ candidates
def run_candidates(per_cell, out_dir, workers=6, seeds=None, amend=None):
    """The candidate pool of the hidden set: `per_cell` os.urandom seeds per cell, written first
    (candidates.json), then every world generated and exported (world, sheet, truth, digest, MJCF, PNG).
    `amend` (a text): regenerate the pool of out_dir/candidates.json from its own seeds with the code of today,
    keeping the seeds and their order and adding the text with the new frozen sha256 to `amendments`."""
    from .gen_venue import load_params, params_digest
    os.makedirs(out_dir, exist_ok=True)
    prev = None
    if amend is not None:
        with open(os.path.join(out_dir, "candidates.json"), "r", encoding="utf-8") as f:
            prev = json.load(f)
        seeds = {}
        for o in prev["candidates"]:
            seeds.setdefault("%s/%s" % (o["cell"]["grip_tier"], o["cell"]["layout_class"]), []).append(o["seed_hex"])
        per_cell = len(next(iter(seeds.values())))
    if seeds is None:
        seeds = {"%s/%s" % c: [os.urandom(16).hex() for _ in range(per_cell)] for c in CELLS}
    rule = ("DESIGN 4.2: 3 grip tiers x 2 layout classes; %d os.urandom candidate seeds per cell drawn before any "
            "world was generated, in the listed order; the selection package runs the reference program on them "
            "in this order and keeps the first that passes per cell" % per_cell)
    order = []
    for c in CELLS:
        for k, s in enumerate(seeds["%s/%s" % c]):
            order.append({"cell": {"grip_tier": c[0], "layout_class": c[1]}, "seed_hex": s, "order": k})
    head = {"rule": rule, "cells": [{"grip_tier": t, "layout_class": c} for t, c in CELLS], "candidates": order,
            "generator": {"params_sha256": params_digest(load_params())}, "frozen_sha256": package_sha256(),
            "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    if prev is not None:
        head = prev
        head.setdefault("amendments", []).append({"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "what": amend,
                                                  "params_sha256": params_digest(load_params()),
                                                  "frozen_sha256": package_sha256()})
    with open(os.path.join(out_dir, "candidates.json"), "w", encoding="utf-8") as f:
        json.dump(head, f, indent=1)
    jobs = [(o["seed_hex"], o["cell"]["grip_tier"], o["cell"]["layout_class"], out_dir) for o in order]
    if workers > 1:
        import multiprocessing as mp
        with mp.get_context("fork").Pool(workers) as pool:
            rows = pool.map(_candidate, jobs, chunksize=1)
    else:
        rows = [_candidate(j) for j in jobs]
    for o, r in zip(order, rows):
        r["order"] = o["order"]
    with open(os.path.join(out_dir, "candidates_table.json"), "w", encoding="utf-8") as f:
        json.dump({"rows": rows}, f, indent=1)
    return rows


def _candidate(job):
    seed, tier, cls, out_dir = job
    from sketch.examples import check
    from . import checks as CK
    from . import export as EX
    from .gen_venue import generate_with_info
    w, info = generate_with_info(seed, tier, cls)
    paths = EX.export_all(w, os.path.join(out_dir, "worlds"))
    me, md = EX.model_errors(w, "track")
    ma, _ = EX.model_errors(w, "avoid")
    k = CK.kinematic_lap(w)
    row = CK.stats_row(w)
    row.update({"seed_hex": seed, "layout_digest": w["layout_digest"], "check_world": len(check.check_world(w)),
                "check_model": len(me) + len(ma), "kinematic_ok": k["ok"], "dropped": info["dropped"],
                "files": {kk: os.path.relpath(v, out_dir) for kk, v in paths.items()},
                "side_pull": w["venue_sheet"]["floor"]["side_pull"], "floor_kind": w["truth"]["floor"]["kind"],
                "boxes": len(w["truth"]["avoidance"]["boxes"])})
    return row


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("seeds", "bytes", "table", "candidates"))
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--k", type=int, default=12)
    ap.add_argument("--per-cell", type=int, default=4)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--from", dest="src")
    ap.add_argument("--seeds", help="a JSON file with a list of seeds (instead of os.urandom)")
    ap.add_argument("--amend", help="candidates: regenerate the pool of --out from its own seeds; the text is recorded")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.what == "seeds":
        seeds = None
        if a.seeds:
            with open(a.seeds, "r", encoding="utf-8") as f:
                seeds = json.load(f)
        p, s = run_seeds(a.n, a.workers, a.out, seeds=seeds)
        print(p)
        print(json.dumps({k: v for k, v in s.items() if k != "errors"}, indent=1), "errors:", len(s["errors"]))
    elif a.what == "bytes":
        print(run_bytes(a.src, a.k, a.out))
    elif a.what == "table":
        p, s = run_table(a.src, a.out)
        print(p)
        print(json.dumps(s, indent=1))
    else:
        rows = run_candidates(a.per_cell, a.out, workers=a.workers, amend=a.amend)
        clean = sum(1 for r in rows if not r["check_world"] and not r["check_model"] and r["kinematic_ok"])
        print(len(rows), "candidates;", clean, "clean")
    return 0


if __name__ == "__main__":
    sys.exit(main())
