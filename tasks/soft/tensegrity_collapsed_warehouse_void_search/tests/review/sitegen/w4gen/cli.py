"""Command line of the W4 generator (version 2.0 site family: W3 plus heaps on the open floor).

  python -I -B cli.py --seed 11 --out DIR            one package DIR/w4_s0011/
  python -I -B cli.py --batch 1-80 --out DIR         one package per seed + DIR/batch_w4_1_80.json
  options: --seeds 1,5,9 (instead of the range), --no-render (skip G3), --cf-render (G7c also compares the classic
           renderer's RGB and depth frames of the three counterfactual sites), --full-cf (G20 and G24 measured in full
           on all three counterfactual models), --tag w4, --pool (allow the hidden and practice pools, POOLS below: the
           selection's job only), --no-sidecar (do not write heap_pieces.json)

A site package = site.json (every coordinate and the truth, the heaps' records under "heaps" and one box record of
kind "heap" per heap: its rectangle from the floor to its top), site.xml (the world only, no robot; each heap's
collision body an inline height field) and qc.json (G1-G24). heap_pieces.json beside them (for the maintainers only,
never part of a package) says which library piece made each heap. Library: environment variables
W4GEN_HEAP_REGISTRY and W4GEN_HEAP_LIB (module library). Task folder: ARB_SRC_ROOT/<task> when ARB_SRC_ROOT is set.
Numeric environment of the reference runs: OPENBLAS_CORETYPE=Haswell, OPENBLAS_L2_SIZE=256, one BLAS / OpenMP thread,
PYTHONDONTWRITEBYTECODE=1, MUJOCO_GL=osmesa on this host.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from w4gen import family, predicates  # noqa: E402

# version 1.1's hidden 101-180 (+181-260) and practice 501-530 (+531-560); version 2.0 on heap library version 2: hidden
# 301-400 (+401-500), practice 601-640 (+641-680); on heap library version 3: hidden 701-800 (+801-900), practice
# 901-1000, then hidden 1101-1200 (+1201-1300). Without --pool the generator refuses every seed of these pools.
POOLS = (range(101, 261), range(501, 561), range(301, 501), range(601, 681), range(701, 801), range(801, 901),
         range(901, 1001), range(1101, 1201), range(1201, 1301))


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


def build(seed, render=True, cf_render=False, full=False):
    """(site, xml, qc, bundle) of one seed."""
    b = family.build_sites(seed)
    qc = predicates.evaluate(b["site"], b["xml"], b, render=render, cf_render=cf_render, full=full)
    qc = dict(seed=seed, generator=b["site"]["generator"], **qc)
    return b["site"], b["xml"], qc, b


def write_package(seed, out, tag="w4", render=True, cf_render=False, full=False, pool=False, sidecar=True):
    if in_pool(seed) and not pool:
        raise SystemExit("seed %d is in a hidden or practice pool: only the selection generates those (--pool)" % seed)
    t0 = time.time()
    site, xml, qc, b = build(seed, render=render, cf_render=cf_render, full=full)
    t1 = time.time()
    d = os.path.join(out, "%s_s%04d" % (tag, seed))
    os.makedirs(d, exist_ok=True)
    files = [("site.json", dumps(site)), ("site.xml", xml), ("qc.json", dumps(qc))]
    if sidecar:
        files.append(("heap_pieces.json", dumps(family.sidecar(b))))
    for name, text in files:
        tmp = os.path.join(d, name + ".part")
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.write(text)
        os.replace(tmp, os.path.join(d, name))
    st, v, hp = site["stratum"], site["void"], site["heaps"]
    g7c = qc["G7"]["G7c"] or {}
    return dict(seed=seed, pass_all=qc["pass"], pass_G1_G19=qc["pass_G1_G19"], pass_G1_G16=qc["pass_G1_G16"],
                failed=qc["failed"], corner=st["corner"], true_index=st["true_index"], near_rank=st["near_rank"],
                shape=st["shape"], blind_route_m=v["blind_route_m"], n_heaps=hp["n"], by_band=hp["by_band"],
                by_class=hp["by_class"], at_slab_front=hp["at_slab_front"], heaps_dropped_for_G9=hp["dropped_for_G9"],
                heap_tries=hp["placement"]["tried"], heap_rejections=hp["placement"]["rejections"],
                heaps=[dict(name=h["name"], cls=h["cls"], against=h["against"], rect=h["rect"], top_m=h["top_m"],
                            scale=h["scale"]) for h in hp["heaps"]],
                n_rubble=[p["n_boxes"] for _k, p in sorted(qc["G17_G19"]["counterfactual"].items())],
                G7c=g7c, G4=qc["G4"]["visible_fraction_open_floor_0p20"], G5=qc["G5"]["pass_"],
                G3=(qc["G3"].get("as_built") or {}).get("max_px"),
                draw_attempts=site["draw_attempts"], build_qc_s=round(t1 - t0, 2))


def main(argv=None):
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--seed", type=int)
    g.add_argument("--batch")
    g.add_argument("--seeds")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default="w4")
    ap.add_argument("--no-render", action="store_true")
    ap.add_argument("--cf-render", action="store_true")
    ap.add_argument("--full-cf", action="store_true")
    ap.add_argument("--pool", action="store_true")
    ap.add_argument("--no-sidecar", action="store_true")
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
            rows.append(write_package(s, a.out, a.tag, render=not a.no_render, cf_render=a.cf_render, full=a.full_cf,
                                      pool=a.pool, sidecar=not a.no_sidecar))
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
