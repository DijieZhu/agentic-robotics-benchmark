"""Command line of the field: draw the additions, check worlds, run a heat's field alone.

    python -B -m h2h.field generate --base BASE.json --tier fast --out DIR [--heat-laps N] [--record R.json]
    python -B -m h2h.field generate --base BASE.json --tier fast --out DIR --attempts A1 A2
    python -B -m h2h.field check W.json [W.json ...] [--deep]
    python -B -m h2h.field alone --world W.json --heat 1 --out R.json [--run-s 300]

`generate` writes DIR/<world id>.json (a draft world unless --heat-laps is given) and prints its id, digest and the
attempt index of each heat: it runs each heat's field alone and draws a failing heat again (the generator's record,
per heat and attempt, goes to --record); with --attempts it draws the given indices and runs nothing;
`check` prints one line per world (errors, grid and pace predicates) and exits 1 if any world has an error;
`alone` writes the field-alone record of one heat.
"""

import argparse
import json
import os
import sys

from sketch.examples import check

from . import checks as CH
from . import world as W


def _write_json(obj, path):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, sort_keys=True)
        f.write("\n")
    os.replace(tmp, path)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="h2h.field")
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate")
    g.add_argument("--base", required=True)
    g.add_argument("--tier", required=True, choices=("fast", "mid", "slow"))
    g.add_argument("--out", required=True)
    g.add_argument("--heat-laps", type=int, default=None)
    g.add_argument("--record", default=None)
    g.add_argument("--attempts", type=int, nargs=2, default=None)
    c = sub.add_parser("check")
    c.add_argument("worlds", nargs="+")
    c.add_argument("--deep", action="store_true")
    a = sub.add_parser("alone")
    a.add_argument("--world", required=True)
    a.add_argument("--heat", type=int, required=True, choices=(1, 2))
    a.add_argument("--out", required=True)
    a.add_argument("--run-s", type=float, default=None)
    args = ap.parse_args(argv)
    if args.cmd == "generate":
        base = check.load_json(args.base)
        info = {}
        rec = None
        if args.attempts is not None:
            w = W.add_head_to_head(base, args.tier, heat_laps=args.heat_laps, info=info, attempts=args.attempts)
        else:
            w, rec = W.generate(base, args.tier, heat_laps=args.heat_laps, info=info)
        os.makedirs(args.out, exist_ok=True)
        path = W.write_world(w, os.path.join(args.out, w["world_id"] + ".json"))
        if rec is not None and args.record:
            os.makedirs(os.path.dirname(os.path.abspath(args.record)), exist_ok=True)
            _write_json(rec, args.record)
        key = W.load_params()["redraw"]["attempts_key"]
        row = {"world_id": w["world_id"], "layout_digest": w["layout_digest"], "file": os.path.basename(path),
               "car_side": info["car_side"], "lines": info["lines"], key: w["truth"]["head_to_head"][key]}
        if rec is not None:
            row["pass"] = rec["pass"]
            row["heats"] = [{"heat": x["heat"], "tried": [(t["attempt"], t["pass"], t["first_contact"]) for t in x["attempts"]]}
                            for x in rec["heats"]]
        print(json.dumps(row, sort_keys=True))
        return 0
    if args.cmd == "check":
        bad = 0
        for p in args.worlds:
            w = check.load_json(p)
            err = CH.check_world_h2h(w, deep=args.deep)
            row = {"file": os.path.basename(p), "errors": err}
            if not err:
                row["grid"] = CH.grid_predicates(w)
                row["pace"] = CH.pace_predicates(w)
            bad += bool(err)
            print(json.dumps(row, sort_keys=True))
        return 1 if bad else 0
    if args.cmd == "alone":
        from .alone import run_alone
        w = check.load_json(args.world)
        if check.world_digest(w) != w["layout_digest"]:
            raise SystemExit("layout digest mismatch")
        rec = run_alone(w, args.heat, run_s=args.run_s)
        _write_json(rec, args.out)
        print(json.dumps({k: rec[k] for k in ("world_id", "heat", "pass", "why_not", "first_train_t_s", "trail_change_m",
                                              "wall_s")}, sort_keys=True))
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
