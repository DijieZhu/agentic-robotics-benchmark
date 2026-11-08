"""Maintainers' gate: every outcome of one bay mission run against another's (two `review/run_mission.py --out` records,
or a record's `data`), the keys that measure the machine left out.

    python3 tests/review/compare_runs.py NEW.json OLD.json [--show N]

Compared: every key of the runs' `data` (the control probe's record as verify keeps it: the mission's outcomes, the
day as applied, the lock's totals, the relayed steps, the mission stage's facts), recursively, exactly (floats as
recorded). Left out: the timings (`reply_s`, `wall_s`, `ms_per_step`, `timing`, `probe_out`, `real_bay.timing`) and
what round 4's sensors added to the record (`frames`, `setup_stage`, `render_error`, `dress`, `streamed`,
`not_streamed`). Exit 0 when nothing differs (the bit-identity proof of the reference: the round-4 runs of the fixed
set's two days against round 3's selection records).
"""

import json
import sys

SKIP_TOP = {"reply_s", "wall_s", "ms_per_step", "timing", "probe_out", "frames", "setup_stage", "render_error", "dress", "streamed",
            "not_streamed"}
SKIP_PATHS = {"real_bay.timing"}


def data_of(rec):
    return rec["data"] if isinstance(rec, dict) and isinstance(rec.get("data"), dict) else rec


def diff(a, b, path="", out=None):
    out = [] if out is None else out
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b), key=str):
            p = f"{path}.{k}" if path else str(k)
            if (not path and k in SKIP_TOP) or p in SKIP_PATHS:
                continue
            if k not in a or k not in b:
                out.append((p, a.get(k, "<absent>"), b.get(k, "<absent>")))
                continue
            diff(a[k], b[k], p, out)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append((path, f"{len(a)} items", f"{len(b)} items"))
        for i, (x, y) in enumerate(zip(a, b)):
            diff(x, y, f"{path}[{i}]", out)
    elif a != b:
        out.append((path, a, b))
    return out


def compare(new, old):
    """The differing outcome keys of two records: [(key, new value, old value)]."""
    return diff(data_of(new), data_of(old))


def main():
    argv = sys.argv[1:]
    show = 40
    if "--show" in argv:
        i = argv.index("--show")
        show = int(argv[i + 1])
        argv = argv[:i] + argv[i + 2:]
    args = [a for a in argv if not a.startswith("--")]
    if len(args) != 2:
        sys.exit(__doc__)
    new, old = (json.load(open(p)) for p in args)
    d = compare(new, old)
    for k, x, y in d[:show]:
        print(f"DIFFERS {k}: {json.dumps(x, default=str)[:160]} vs {json.dumps(y, default=str)[:160]}")
    if len(d) > show:
        print(f"... {len(d) - show} more")
    print("IDENTICAL" if not d else f"{len(d)} DIFFERENCES")
    sys.exit(0 if not d else 1)


if __name__ == "__main__":
    main()
