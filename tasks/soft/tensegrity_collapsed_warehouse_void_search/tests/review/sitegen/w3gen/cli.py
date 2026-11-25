"""Command line of the W3 generator (version 1.1 site family, rubble included).

  python -I -B cli.py --seed 11 --out DIR            one package DIR/w3_s0011/
  python -I -B cli.py --batch 1-60 --out DIR         one package per seed + DIR/batch_1_60.json
  options: --seeds 1,5,9 (instead of the range), --no-render (skip G3), --cf-render (G7c also compares the classic
           renderer's RGB and depth frames of the three counterfactual sites), --no-rubble (regression only: the
           sites without the last assembly step; not version 1.1 sites), --tag w3, --pool (allow the hidden and
           practice pools, the selection's job only)

A site package = site.json (every coordinate and the truth: which mouth is open, the void, the rubble, frictions,
lighting, true and given camera extrinsics), site.xml (the world only, no robot) and qc.json (G1-G19; G7c, G9 and
G17-G19 over the three counterfactual sites). Numeric environment of the reference runs: OPENBLAS_CORETYPE=Haswell,
one BLAS / OpenMP thread, PYTHONDONTWRITEBYTECODE=1, MUJOCO_GL=osmesa on this host.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from w3gen import family, predicates  # noqa: E402

POOLS = (range(101, 261), range(501, 561))      # hidden 101-180 (+181-260), practice 501-530 (+531-560)


def _clean(o):
    import numpy as np
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, set):
        return sorted(_clean(v) for v in o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    return o


def dumps(obj):
    return json.dumps(_clean(obj), indent=1, ensure_ascii=True) + "\n"


def in_pool(seed):
    return any(seed in p for p in POOLS)


def build(seed, render=True, rubble=True, cf_render=False):
    """(site, xml, qc, bundle) of one seed."""
    b = family.build_sites(seed, rubble=rubble)
    qc = predicates.evaluate(b["site"], b["xml"], render=render, g9=b["g9"], bundle=b, cf_render=cf_render)
    qc = dict(seed=seed, generator=b["site"]["generator"], **qc)
    return b["site"], b["xml"], qc, b


def write_package(seed, out, tag="w3", render=True, rubble=True, cf_render=False, pool=False):
    if in_pool(seed) and not pool:
        raise SystemExit("seed %d is in the hidden or practice pool: only the selection generates those (--pool)" % seed)
    t0 = time.time()
    site, xml, qc, b = build(seed, render=render, rubble=rubble, cf_render=cf_render)
    t1 = time.time()
    d = os.path.join(out, "%s_s%04d" % (tag, seed))
    os.makedirs(d, exist_ok=True)
    for name, text in (("site.json", dumps(site)), ("site.xml", xml), ("qc.json", dumps(qc))):
        with open(os.path.join(d, name), "w", encoding="utf-8") as fh:
            fh.write(text)
    st, v = site["stratum"], site["void"]
    g7c = qc["G7"]["G7c"] or {}
    return dict(seed=seed, pass_all=qc["pass"], pass_G1_G16=qc["pass_G1_G16"], failed=qc["failed"], corner=st["corner"],
                true_index=st["true_index"], near_rank=st["near_rank"], shape=st["shape"], g1_dir=st["g1_dir"],
                blind_route_m=v["blind_route_m"], third=site["r1"]["third"], tunnel_roof=site["r1"]["tunnel_roof"],
                pocket2=len(v["pockets"]) > 1, d1_m=v["d1_m"], partition_m=v["partition_m"],
                sp3_clear=qc["G8a"]["sp3_clearance_m"], regularities=qc["regularities"],
                n_rubble=[p["n_boxes"] for _k, p in sorted(qc["G17_G19"]["counterfactual"].items())],
                rubble_route_m=[p["route_with_rubble_m"] for _k, p in sorted(qc["G17_G19"]["counterfactual"].items())],
                G7c=g7c, hidden_box_rays_hit=qc["G7"]["hidden_box_rays_hit"],
                G3=(qc["G3"].get("as_built") or {}).get("max_px"), G4=qc["G4"]["visible_fraction_open_floor_0p20"],
                draw_attempts=site["draw_attempts"], draw_rejections=site["draw_rejections"],
                build_qc_s=round(t1 - t0, 2))


def main(argv=None):
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--seed", type=int)
    g.add_argument("--batch")
    g.add_argument("--seeds")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default="w3")
    ap.add_argument("--no-render", action="store_true")
    ap.add_argument("--cf-render", action="store_true")
    ap.add_argument("--no-rubble", action="store_true")
    ap.add_argument("--pool", action="store_true")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    if a.seed is not None:
        seeds, label = [a.seed], str(a.seed)
    elif a.batch:
        lo, hi = (int(x) for x in a.batch.split("-"))
        seeds, label = list(range(lo, hi + 1)), "%d_%d" % (lo, hi)
    else:
        seeds = [int(s) for s in a.seeds.split(",")]
        label = "list_%d_%d_n%d" % (seeds[0], seeds[-1], len(seeds))
    rows = []
    for s in seeds:
        try:
            rows.append(write_package(s, a.out, a.tag, render=not a.no_render, rubble=not a.no_rubble,
                                      cf_render=a.cf_render, pool=a.pool))
        except SystemExit:
            raise
        except Exception as exc:  # noqa: BLE001
            import traceback
            rows.append(dict(seed=s, pass_all=False, failed=["crash"], error="%s: %s" % (type(exc).__name__, exc),
                             tb=traceback.format_exc()[-1500:]))
        print(json.dumps(_clean(rows[-1])), flush=True)
    if len(seeds) > 1:
        with open(os.path.join(a.out, "batch_%s_%s.json" % (a.tag, label)), "w", encoding="utf-8") as fh:
            fh.write(dumps(rows))
    return 0 if all(r.get("failed") != ["crash"] for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
