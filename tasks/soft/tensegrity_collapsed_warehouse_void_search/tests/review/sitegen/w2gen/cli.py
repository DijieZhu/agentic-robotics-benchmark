"""Command line of the W2 generator.

  python -I -B cli.py --seed 11 --out DIR          one site package in DIR/w2_s0011/
  python -I -B cli.py --batch 101-180 --out DIR    one package per seed + DIR/batch_101_180.json

A site package = site.json (every coordinate and the truth: which mouth is open, the plug, frictions, lighting,
true and given camera extrinsics), site.xml (the world only, no robot) and qc.json (G1-G11).
Options: --no-render (skip G3, which needs an offscreen GL: OSMesa on this host), --lx 4.6, --gravel-relief.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from w2gen import family, predicates  # noqa: E402


def _clean(o):
    import numpy as np
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    return o


def dumps(obj):
    return json.dumps(_clean(obj), indent=1, ensure_ascii=True) + "\n"


def write_package(seed, out, render=True, lx=family.LX_DEFAULT, relief=False):
    t0 = time.time()
    site, xml, g9 = family.build_site(seed, lx=lx, gravel_relief=relief)
    t1 = time.time()
    qc = predicates.evaluate(site, xml, render=render, g9=g9)
    qc = dict(seed=seed, generator=site["generator"], envelope_source=site["envelope"]["source"], **qc)
    t2 = time.time()
    d = os.path.join(out, "w2_s%04d" % seed)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "site.json"), "w", encoding="utf-8") as fh:
        fh.write(dumps(site))
    with open(os.path.join(d, "site.xml"), "w", encoding="utf-8") as fh:
        fh.write(xml)
    with open(os.path.join(d, "qc.json"), "w", encoding="utf-8") as fh:
        fh.write(dumps(qc))
    return site, qc, dict(build_s=round(t1 - t0, 2), predicates_s=round(t2 - t1, 2))


def summary_row(site, qc, timing):
    g = lambda k, *path: _dig(qc.get(k, {}), path)
    return dict(
        seed=site["seed"], pass_all=qc["pass"], failed=qc["failed"],
        third=site["r1"]["third"], tunnel_roof=site["r1"]["tunnel_roof"], sp3_mode=site["void"]["sp3_mode"],
        r2_decoy=site["r2"]["decoy_type"], nearest_P_mouth_is_true=site["void"]["nearest_P_mouth_is_true"],
        draw_attempts=site["draw_attempts"], free_blocks_kept=len(site["free_blocks"]["kept"]),
        free_blocks_dropped=site["free_blocks"]["dropped_for_G9"], chips=len(site["chips"]),
        G2_margin_px=g("G2", "corner_margin_px"), G3_max_px=g("G3", "as_built", "max_px"),
        G3_flipped_max_px=g("G3", "flipped_sign", "max_px"), G4=g("G4", "visible_fraction_open_floor_0p20"),
        G6_far_mm_px=g("G6", "mm_per_px_far_centre"), G7_plug_rays=g("G7", "plug_rays_hit"),
        G7_mouth_depth=[g("G7", "mouth_floor", "M", "visible_floor_depth_m"), g("G7", "mouth_floor", "Bm", "visible_floor_depth_m")],
        G8a_max_credit=g("G8a", "max_credit"), G8b_blind_m=g("G8b", "blind_path_m"),
        G9a_total_m=g("G9", "a", "total_m"), G9b=g("G9", "b_tunnel_blocked", "connected"),
        G9c=g("G9", "c_gap_blocked", "connected"), G9d=g("G9", "d", "visible_fraction_pad_1_2_tunnel_blocked"),
        G10_min_pad=min(qc["G10"]["pad_to_sp"].values()) if "G10" in qc else None,
        cam_z=site["camera"]["true"]["pos"][2], tilt_deg=site["camera"]["true"]["tilt_deg"], **timing)


def _dig(d, path):
    for p in path:
        if not isinstance(d, dict) or p not in d:
            return None
        d = d[p]
    return d


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int)
    ap.add_argument("--batch", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-render", action="store_true")
    ap.add_argument("--lx", type=float, default=family.LX_DEFAULT)
    ap.add_argument("--gravel-relief", action="store_true")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    seeds = [a.seed] if a.batch is None else list(range(int(a.batch.split("-")[0]), int(a.batch.split("-")[1]) + 1))
    rows = []
    for s in seeds:
        try:
            site, qc, tm = write_package(s, a.out, render=not a.no_render, lx=a.lx, relief=a.gravel_relief)
            rows.append(summary_row(site, qc, tm))
        except Exception as exc:      # noqa: BLE001  (a seed that crashes is recorded, the batch goes on)
            rows.append(dict(seed=s, pass_all=False, failed=["crash"], error="%s: %s" % (type(exc).__name__, exc)))
        print(json.dumps(_clean(rows[-1])), flush=True)
    if a.batch:
        with open(os.path.join(a.out, "batch_%s.json" % a.batch.replace("-", "_")), "w", encoding="utf-8") as fh:
            fh.write(dumps(rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
