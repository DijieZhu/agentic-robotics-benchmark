"""Command line: generate one or more sites.

  python -m site_gen.generate_site --seed 7 --out <output folder>
  python -m site_gen.generate_site --seeds 1-30 --out ... --workers 6 --no-maps

Each site goes to <out>/site_<seed>/ (site.json, prior.json, scene.xml, assets/, trav.npz, fields.npz, maps/).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback


def _one(args):
    seed, out, maps, cfg_path = args
    from site_gen.config import load_config
    from site_gen.generator import generate
    cfg = load_config(cfg_path)
    d = os.path.join(out, "site_%d" % seed)
    t = time.time()
    try:
        site, prior = generate(seed, d, cfg, maps=maps)
        st = site["stats"]
        return {"seed": seed, "ok": True, "dir": d, "s": time.time() - t, "attempts": st["attempts"],
                "ngeom": st["ngeom"], "compile_s": site["_timing"]["compile_s"],
                "route_m": site["route"]["length_m"], "route_s": site["route"]["total_time_s"],
                "rooms": site["route"]["n_rooms_visited"], "classes": site["route"]["classes_used"],
                "needs_crouch": site["guarantees"]["needs_crouch"],
                "needs_high_step": site["guarantees"]["needs_high_step"],
                "behind_discrepancy": site["guarantees"]["behind_discrepancy"],
                "blind_fail": site["guarantees"]["blind_fail"],
                "discrepancies": [x["kind"] for x in site["discrepancies"]],
                "dose_uSv": site["route"]["dose_total_uSv"], "timing": site["_timing"]}
    except Exception as e:  # noqa: BLE001
        return {"seed": seed, "ok": False, "error": "%s: %s" % (type(e).__name__, e), "tb": traceback.format_exc(),
                "s": time.time() - t}


def _seeds(s):
    out = []
    for part in s.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--seeds", type=str, default=None, help="e.g. 1-30 or 3,7,11")
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default=None, help="a YAML file merged over config/defaults.yaml")
    ap.add_argument("--no-maps", action="store_true")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--summary", default=None, help="write a JSON summary here")
    a = ap.parse_args(argv)
    seeds = [a.seed] if a.seed is not None else _seeds(a.seeds or "1")
    jobs = [(s, a.out, not a.no_maps, a.config) for s in seeds]
    os.makedirs(a.out, exist_ok=True)
    res = []
    if a.workers > 1:
        import multiprocessing as mp
        with mp.get_context("spawn").Pool(a.workers) as pool:
            for r in pool.imap_unordered(_one, jobs):
                res.append(r)
                print(json.dumps({k: v for k, v in r.items() if k not in ("tb", "timing")}), flush=True)
    else:
        for j in jobs:
            r = _one(j)
            res.append(r)
            print(json.dumps({k: v for k, v in r.items() if k not in ("tb",)}), flush=True)
            if not r["ok"]:
                print(r["tb"], file=sys.stderr)
    res.sort(key=lambda r: r["seed"])
    if a.summary:
        with open(a.summary, "w") as f:
            json.dump(res, f, indent=1)
    return 0 if all(r["ok"] for r in res) else 1


if __name__ == "__main__":
    sys.exit(main())
