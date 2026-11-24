"""The units' fit properties, calibration errors and camera offsets (a maintainer's tool; the grader never runs it).

Since 2026-11-20 a unit has, beside its three turn properties (tests/hidden/selection.json), three fit properties
(lab_sim.FIT_PROPS: the rotor bores' radial clearance and friction, the lid hinge's friction) and a calibration
error per camera (lab_sim.CALIB). The scored units u0-u5 keep their turn properties; the commissioning benches
d0-d2 keep theirs. The rule, fixed and committed before any value was drawn:

1. Ranges. The fit properties' ranges are lab_sim.UNIT_RANGES; a calibration error is six numbers per camera, a
   rotation vector (deg) and a translation (mm) in the camera's frame, each component within +-lab_sim.CALIB[kind]
   (kind "wrist" for the wrist camera, "fixed" for left, right and rack). The instruction and the equipment sheet
   state every one of them.
2. Key. `draw` takes a key from os.urandom(16) (hex) and writes it, the ranges and the recheck seeds to the record
   before any value is computed from it. Recheck seed i (i = 0..19) is
   int.from_bytes(sha256("<key>|recheck|<i>").digest()[:8], "little").
3. Fit properties, a Latin hypercube per group ("scored": u0-u5, six strata; "bench": d0-d2, three strata). Per
   property the range is cut into as many equal strata as the group has units; the units, sorted by name, take the
   strata in the order of sha256("<key>|<group>|<prop>|perm|<unit>") (the unit with the smallest digest takes the
   lowest stratum); a unit's place in its stratum is sha256("<key>|<group>|<prop>|pos|<unit>|<replace>") / 2**256
   (replace = 0 at first); 4 decimals.
4. Calibration errors. Component j (0-5) of camera cam of unit u is
   (2 * sha256("<key>|calib|<u>|<cam>|<j>") / 2**256 - 1) * bound, 3 decimals; never redrawn.
4b. Offsets (added 2026-11-24, after a reviewer showed that the fixed cameras stood exactly at the equipment
   sheet's poses, so that the sheet's pose against the pose on file gave the calibration error). Per unit and per
   camera the pose on file is the sheet's nominal pose times an offset, six numbers like a calibration error, each
   component within +-lab_sim.OFFSET[kind]; the camera's true pose is the pose on file times the inverse of the
   calibration error of rule 4. `draw-offset` takes a second key from os.urandom(16) (hex) and writes it and the
   bounds to the record before any value is computed from it. Component j of camera cam of unit u is
   (2 * sha256("<key2>|offset|<u>|<cam>|<j>") / 2**256 - 1) * bound, 3 decimals; never redrawn. The fit properties
   and the calibration errors keep their values (the first key, the replace counts as they stand). The offsets come
   from another key than the errors: what a controller can compute, the pose on file against the nominal pose, is
   the offset, and it is independent of the error. After `draw-offset` every scored unit is rechecked again by
   rule 6 (the earlier recheck is kept in the record as `recheck_before_offset`).
5. Bench separation. A bench whose three fit properties all lie within 2 % of the property's range of one scored
   unit's is drawn again (its replace + 1) until it does not.
6. Recheck. Every scored unit (its turn properties, its fit properties, its calibration errors) is run with the
   privileged reference (solution/oracle, in process, on the environment as the grader builds it) on the 20 recheck
   seeds. A unit is kept when every recheck episode scores credit 1.0; a unit that fails gets its fit properties'
   next places in the same strata (replace + 1) and is rechecked the same way, at most MAX_REPLACE times. The
   calibration errors are not what a failing unit is replaced for: the reference reads the truth, not the cameras.
   Every recheck is recorded: the properties, the episodes at 1.0, the failed seeds with what they lost.
7. Write. `write` puts the kept values into tests/units.json, solution/oracle/units.json (scored) and
   environment/commission_service/bench.json (benches), next to the turn properties, and refuses while a scored
   unit is not kept.

The fixed episode set is selected afterwards by tests/review/select_set.py, as before (a new draw: the frozen files
changed).

    python3 -B tests/review/unit_fit.py draw --record tests/hidden/selection_fit.json
    python3 -B tests/review/unit_fit.py draw-offset --record tests/hidden/selection_fit.json
    python3 -B tests/review/unit_fit.py recheck --record tests/hidden/selection_fit.json [--workers 6]
    python3 -B tests/review/unit_fit.py write --record tests/hidden/selection_fit.json

`recheck` needs the task image's Python (see select_set.py).
"""

import argparse
import datetime
import hashlib
import json
import multiprocessing
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TESTS = os.path.dirname(HERE)
TASK = os.path.dirname(TESTS)
ORACLE = os.path.join(TASK, "solution", "oracle")
BENCH = os.path.join(TASK, "environment", "commission_service", "bench.json")
sys.dont_write_bytecode = True

FORMAT = "lab_sample_handling unit fit 1"
GROUPS = {"scored": ("u0", "u1", "u2", "u3", "u4", "u5"), "bench": ("d0", "d1", "d2")}
N_RECHECK = 20
MAX_REPLACE = 8
SEPARATION = 0.02
ENV = {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "MUJOCO_GL": "osmesa",
       "PYOPENGL_PLATFORM": "osmesa"}


def load(path):
    with open(path) as f:
        return json.load(f)


def dump(path, obj):
    with open(path + ".tmp", "w") as f:
        json.dump(obj, f, indent=1)
        f.write("\n")
    os.replace(path + ".tmp", path)


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def digest(msg):
    return int.from_bytes(hashlib.sha256(msg.encode()).digest(), "big")


def frac(msg):
    return digest(msg) / 2.0 ** 256


def constants():
    """lab_sim's ranges, read from its source (no simulation needed to draw)."""
    import ast
    src = open(os.path.join(TESTS, "lab_sim.py")).read()
    out = {}
    for node in ast.parse(src).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) \
                and node.targets[0].id in ("UNIT_RANGES", "FIT_PROPS", "CALIB", "CALIB_CAMS", "OFFSET"):
            out[node.targets[0].id] = ast.literal_eval(node.value)
    return out


def strata(key, group, prop):
    """unit -> stratum index, rule 3."""
    units = sorted(GROUPS[group])
    order = sorted(units, key=lambda u: digest("%s|%s|%s|perm|%s" % (key, group, prop, u)))
    return {u: order.index(u) for u in units}


def fit_value(key, group, prop, unit, replace, rng):
    n = len(GROUPS[group])
    s = strata(key, group, prop)[unit]
    lo, hi = rng
    return round(lo + (hi - lo) * (s + frac("%s|%s|%s|pos|%s|%d" % (key, group, prop, unit, replace))) / n, 4)


def fit_props(key, group, unit, replace, c):
    return {p: fit_value(key, group, p, unit, replace, c["UNIT_RANGES"][p]) for p in c["FIT_PROPS"]}


def calib(key, unit, c):
    out = {}
    for cam in c["CALIB_CAMS"]:
        mm, deg = c["CALIB"]["wrist" if cam == "wrist" else "fixed"]
        bounds = [deg] * 3 + [mm] * 3
        out[cam] = [round((2 * frac("%s|calib|%s|%s|%d" % (key, unit, cam, j)) - 1) * bounds[j], 3) for j in range(6)]
    return out


def offset(key2, unit, c):
    out = {}
    for cam in c["CALIB_CAMS"]:
        mm, deg = c["OFFSET"]["wrist" if cam == "wrist" else "fixed"]
        bounds = [deg] * 3 + [mm] * 3
        out[cam] = [round((2 * frac("%s|offset|%s|%s|%d" % (key2, unit, cam, j)) - 1) * bounds[j], 3)
                    for j in range(6)]
    return out


def too_close(bench, scored, c):
    for s in scored.values():
        if all(abs(bench[p] - s[p]) <= SEPARATION * (c["UNIT_RANGES"][p][1] - c["UNIT_RANGES"][p][0])
               for p in c["FIT_PROPS"]):
            return True
    return False


def values(rec, c):
    """Every unit's current values from the record's key and replace counts."""
    key = rec["key"]
    out = {"scored": {}, "bench": {}}
    for u in GROUPS["scored"]:
        out["scored"][u] = dict(fit_props(key, "scored", u, rec["replace"][u], c), calib=calib(key, u, c))
    for u in GROUPS["bench"]:
        out["bench"][u] = dict(fit_props(key, "bench", u, rec["replace"][u], c), calib=calib(key, u, c))
    if rec.get("offset"):                                         # rule 4b
        for g in out:
            for u in out[g]:
                out[g][u]["offset"] = offset(rec["offset"]["key"], u, c)
    return out


def cmd_draw(a):
    if os.path.exists(a.record):
        raise SystemExit("%s exists: one draw per record" % a.record)
    c = constants()
    key = os.urandom(16).hex()
    rec = {"format": FORMAT, "rule": "tests/review/unit_fit.py (module docstring)", "drawn_utc": utc_now(), "key": key,
           "ranges": {p: list(c["UNIT_RANGES"][p]) for p in c["FIT_PROPS"]},
           "calib_bounds_mm_deg": {k: list(v) for k, v in c["CALIB"].items()},
           "recheck_seeds": [int.from_bytes(hashlib.sha256(("%s|recheck|%d" % (key, i)).encode()).digest()[:8],
                                            "little") for i in range(N_RECHECK)],
           "replace": {u: 0 for g in GROUPS.values() for u in g}, "recheck": {}, "bench_redraws": []}
    dump(a.record, rec)                       # the key is on disk before any value is computed from it
    scored = {u: fit_props(key, "scored", u, 0, c) for u in GROUPS["scored"]}
    for u in GROUPS["bench"]:
        while too_close(fit_props(key, "bench", u, rec["replace"][u], c), scored, c):
            rec["bench_redraws"].append({"unit": u, "replace": rec["replace"][u]})
            rec["replace"][u] += 1
    rec["values"] = values(rec, c)
    dump(a.record, rec)
    print("drawn: key and values in %s" % a.record)


def cmd_draw_offset(a):
    rec = load(a.record)
    if rec.get("offset"):
        raise SystemExit("%s has its offsets: one draw per record" % a.record)
    c = constants()
    rec["offset"] = {"rule": "tests/review/unit_fit.py (module docstring), rule 4b", "drawn_utc": utc_now(),
                     "key": os.urandom(16).hex(), "bounds_mm_deg": {k: list(v) for k, v in c["OFFSET"].items()}}
    rec["recheck_before_offset"] = rec["recheck"]
    rec["recheck"] = {}
    rec.pop("values", None)
    dump(a.record, rec)                       # the second key is on disk before any value is computed from it
    rec["values"] = values(rec, c)
    dump(a.record, rec)
    print("offsets drawn: key and values in %s; run recheck" % a.record)


# ------------------------------------------------------------------------------------------- recheck
def _episode(args):
    unit, props, seed = args
    os.environ.update(ENV)
    for p in (ORACLE, TESTS):
        if p not in sys.path:
            sys.path.insert(0, p)
    import episode as EP
    import lab_env as E
    import worklist_probe as W
    env = E.LabEnv(props, unit, render=False)
    ep = EP.Episode(env, seed)
    ep.reset()
    W.reference_args(env.a)
    out = W.run_worklist(env.p, env.a, env.episode, env.src, env.names, env.cold, env.masses)
    sc = ep.score() if hasattr(ep, "score") else EP.Episode.score(ep)
    return {"unit": unit, "seed": seed, "credit": sc["credit"], "status": out.get("status"), "steps": int(env.t),
            "run": sc["run"], "lid": sc["lid"], "cold": int(sum(sc["cold"].values())), "moved": sc["moved"]}


def cmd_recheck(a):
    rec = load(a.record)
    c = constants()
    turn = load(os.path.join(TESTS, "hidden", "selection.json"))["candidates"]["hidden"]
    while True:
        vals = values(rec, c)
        todo = [u for u in GROUPS["scored"] if not rec["recheck"].get(u, {}).get("kept")]
        if not todo:
            break
        jobs = [(u, dict(turn[u], **vals["scored"][u]), s) for u in todo for s in rec["recheck_seeds"]]
        with multiprocessing.get_context("spawn").Pool(a.workers, maxtasksperchild=1) as pool:
            res = pool.map(_episode, jobs, chunksize=1)
        for u in todo:
            mine = [r for r in res if r["unit"] == u]
            failed = [r for r in mine if r["credit"] != 1.0]
            h = rec["recheck"].setdefault(u, {"kept": False, "history": []})
            h["history"].append({"replace": rec["replace"][u],
                                 "props": {p: vals["scored"][u][p] for p in c["FIT_PROPS"]},
                                 "episodes_at_1": len(mine) - len(failed),
                                 "mean": round(sum(r["credit"] for r in mine) / len(mine), 4), "failed": failed})
            if not failed:
                h["kept"] = True
            elif rec["replace"][u] >= MAX_REPLACE:
                dump(a.record, rec)
                raise SystemExit("%s failed %d draws: the ranges are too wide for the reference" % (u, MAX_REPLACE + 1))
            else:
                rec["replace"][u] += 1
            print("[recheck] %s replace %d: %d of %d at 1.0" % (u, h["history"][-1]["replace"],
                                                                 len(mine) - len(failed), len(mine)), flush=True)
        rec["values"] = values(rec, c)
        rec["rechecked_utc"] = utc_now()
        dump(a.record, rec)
    print("every scored unit kept")


def cmd_write(a):
    rec = load(a.record)
    c = constants()
    if not all(rec["recheck"].get(u, {}).get("kept") for u in GROUPS["scored"]):
        raise SystemExit("a scored unit is not kept: run recheck first")
    vals = values(rec, c)
    for path in (os.path.join(TESTS, "units.json"), os.path.join(ORACLE, "units.json")):
        d = load(path)
        for u in GROUPS["scored"]:
            d["units"][u] = dict({k: d["units"][u][k] for k in ("rotor_friction", "handle_play", "rotor_inertia")},
                                 **vals["scored"][u])
        dump(path, d)
    d = load(BENCH)
    for u in GROUPS["bench"]:
        d["units"][u] = dict({k: d["units"][u][k] for k in ("rotor_friction", "handle_play", "rotor_inertia")},
                             **vals["bench"][u])
    d["fit_draw"] = {"rule": "tests/review/unit_fit.py (module docstring), group \"bench\"", "drawn_utc": rec["drawn_utc"],
                     "ranges": rec["ranges"], "calib_bounds_mm_deg": rec["calib_bounds_mm_deg"]}
    if rec.get("offset"):
        d["fit_draw"]["offset_bounds_mm_deg"] = rec["offset"]["bounds_mm_deg"]
        d["fit_draw"]["offset_drawn_utc"] = rec["offset"]["drawn_utc"]
    dump(BENCH, d)
    print("written: tests/units.json, solution/oracle/units.json, environment/commission_service/bench.json")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in (("draw", cmd_draw), ("draw-offset", cmd_draw_offset), ("recheck", cmd_recheck),
                     ("write", cmd_write)):
        sp = sub.add_parser(name)
        sp.add_argument("--record", required=True)
        sp.add_argument("--workers", type=int, default=6)
        sp.set_defaults(fn=fn)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
