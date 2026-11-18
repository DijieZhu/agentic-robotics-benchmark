"""The cell's boundary: no request content can make the cell fail, and the grader re-runs a mission
only for a failure of the cell's own work while the mission was live.

    MUJOCO_GL=osmesa python -I -B tests/review/cell/test_boundary_fuzz.py --site SITE [--calls 3000] [--seed 1]
        [--json OUT]

1. The review's cases, each on a fresh mission through the wire's JSON round trip: a report value 10**400, walk vx
   10**400, set_body height 10**400, set_head pan 10**400, report_source x 10**400, report_mark dose_rate 10**400, step
   n_ticks Infinity / NaN / 10**400, a report after the robot was lost (the world's own end): each is refused
   (bad_params, or mission_over after the end), the mission goes on (or stays over), and verify's rule
   `_cell_failed(finalize())` is [].
2. The fuzz: `--calls` calls on one live mission, every public call with random arguments (valid names and others;
   null, booleans, integers up to 10**400, floats with NaN / Infinity / 1e308, strings up to 100,000 characters, numeric
   strings, lists up to 70 entries, dicts up to 10 entries, nested up to 40 levels; params that are not an object),
   mixed with valid calls that keep the mission walking. Every reply is a normal result or one of bad_params / budget /
   mission_over / forbidden; never internal_error or sensor_unavailable; a refused call leaves the mission's state as
   it was (simulated time, the whole qpos / qvel, the commands, the head, the lamp, the frame and report counters and
   lists, the dosimeter's count, the end).
3. The re-run rule itself: an exception injected into the cell's own work (MissionWorld.step) while the mission is live
   ends it with internal_error and `_cell_failed` = ["internal_error"]; an exception injected after the mission had
   ended (robot lost; into get_state) is recorded (internal_errors 1, internal_errors_live 0) but `_cell_failed` = [].
"""

import argparse
import copy
import hashlib
import json
import math
import os
import random
import sys
import time

TESTS = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, TESTS)

from plantcell import server as _server  # noqa: E402
from plantcell.server import PlantCellServer, PUBLIC_METHODS  # noqa: E402

# the argument names of each call (the server's table; a cell from before the boundary check has none: the names
# from the facade's signatures, so that this test runs against it too and shows what it did)
ARG_KINDS = getattr(_server, "ARG_KINDS", None)
if ARG_KINDS is None:
    import inspect
    from mission.facade import Robot as _Robot
    ARG_KINDS = {m: {n: None for n in list(inspect.signature(getattr(_Robot, m)).parameters)[1:]}
                 for m in PUBLIC_METHODS}

ALLOWED_ERRORS = {"bad_params", "budget", "mission_over", "forbidden"}
BIG = 10 ** 400


def grader_config():
    with open(os.path.join(TESTS, "grader_config.json")) as f:
        return json.load(f)


def wire(srv, method, params):
    """One call through the wire's JSON encoding both ways (Python's json: NaN, Infinity and big integers pass)."""
    req = json.loads(json.dumps({"id": 1, "method": method, "params": params}))
    res = srv.handle(req["method"], req["params"])
    return json.loads(json.dumps(res))


def digest(srv):
    """The mission's state a refused call must not change."""
    w = srv.world
    h = hashlib.sha256()
    h.update(repr((round(float(w.time), 9), int(getattr(w, "tick", 0)), bool(w.done), w.done_reason)).encode())
    for a in (w.data.qpos, w.data.qvel):
        h.update(bytes(memoryview(a.copy())))
    for name in ("cmd", "prog_mode", "body_h", "body_pitch", "head_cmd", "lamp_on", "frame_seq"):
        h.update(repr(getattr(w, name, None)).encode())
    h.update(json.dumps(dict(w.n_frames), sort_keys=True).encode())
    h.update(json.dumps(dict(w.n_reports), sort_keys=True).encode())
    h.update(json.dumps([[r["id"], r["status"]] for r in w.reports]).encode())
    for name in ("dose_gy", "dose_meas_gy"):
        h.update(repr(float(getattr(w, name, 0.0))).encode())
    h.update(repr((int(w.counts_pending), float(w.dwell_pending))).encode())
    return h.hexdigest()


# ------------------------------------------------------------------------------------------------ random JSON
def rnd_value(rng, depth=0):
    k = rng.random()
    if depth >= 3:
        k = rng.random() * 0.62                     # scalars only below three levels (deep() makes deep ones)
    if k < 0.06:
        return None
    if k < 0.12:
        return rng.random() < 0.5
    if k < 0.28:
        return rng.choice([0, 1, -1, 7, 2 ** 31, 2 ** 63, -2 ** 63, 10 ** 15, 10 ** 15 + 1, 10 ** 6, 10 ** 6 + 1,
                           10 ** 18, BIG, -BIG, 10 ** rng.randint(1, 400)])
    if k < 0.44:
        return rng.choice([0.0, -0.0, 0.3, -0.45, 1e6, 1e6 + 1, 1e15, 1e16, 1e308, -1e308, float("nan"),
                           float("inf"), float("-inf"), 5e-324, rng.uniform(-3, 3), rng.uniform(-1e9, 1e9)])
    if k < 0.62:
        n = rng.choice([0, 1, 5, 16, 17, 63, 64, 65, 200, 10000, 100000])
        if rng.random() < 0.3:
            return rng.choice(["1e400", "nan", "Infinity", "0.3", "-5", "open", "closed", "on", "walk", "inspect",
                               "nav", "stand", "sit", "high_step", "F00001", "PI-016", "X", "SM-01", "é中",
                               "\x00", "true"])
        return "".join(rng.choice("abcXYZ019 -_.é") for _ in range(n))
    if k < 0.82:
        n = rng.choice([0, 1, 2, 3, 4, 8, 64, 65, 70])
        return [rnd_value(rng, depth + 1) for _ in range(n)]
    n = rng.choice([0, 1, 2, 8, 9, 10])
    return {("k%d" % i if rng.random() < 0.7 else "x" * rng.choice([1, 64, 65, 300])): rnd_value(rng, depth + 1)
            for i in range(n)}


def deep(n):
    v = 1.0
    for _ in range(n):
        v = [v] if n % 2 else {"a": v}
    return v


def rnd_params(rng, method):
    names = list(ARG_KINDS.get(method, {}))
    u = rng.random()
    if u < 0.05:
        return rng.choice([[], [1, 2], 3, "x", True, None, deep(5)])
    p = {}
    for nm in names:
        if rng.random() < 0.6:
            p[nm] = rnd_value(rng)
    if rng.random() < 0.15:
        p[rng.choice(["self", "foo", "n", "value2", "x" * 70, ""])] = rnd_value(rng)
    if rng.random() < 0.03:
        p[names[0] if names else "z"] = deep(rng.choice([10, 30, 40]))
    return p


def valid_call(rng, srv):
    """A call that keeps the mission moving (walking slowly, stepping, looking, reporting in budget)."""
    k = rng.random()
    if k < 0.35:
        return "step", {"n_ticks": rng.choice([1, 5, 20, 50])}
    if k < 0.55:
        return "walk", {"vx": rng.uniform(-0.1, 0.25), "vy": rng.uniform(-0.05, 0.05), "yaw_rate": rng.uniform(-0.3, 0.3)}
    if k < 0.62:
        return "set_head", {"pan": rng.uniform(-1, 1), "tilt": rng.uniform(-0.5, 0.5)}
    if k < 0.67:
        return "set_mode", {"m": rng.choice(["walk", "stand", "crouch_walk", "high_step"])}
    if k < 0.72:
        return "get_state", {}
    if k < 0.76:
        return "get_lidar", {}
    if k < 0.79:
        return "report", {"tag": "PI-FUZZ", "value": rng.uniform(0, 10), "frame_ids": []}
    if k < 0.81:
        return "get_camera", {"name": "nav", "depth": False}
    return "set_body", {"height": rng.uniform(0.15, 0.3)}


# ------------------------------------------------------------------------------------------------ the parts
def part_review_cases(site, problems):
    import verify
    out = []
    cases = [("report", {"tag": "PI-FUZZ", "value": BIG, "frame_ids": []}),
             ("walk", {"vx": BIG, "vy": 0.0, "yaw_rate": 0.0}),
             ("set_body", {"height": BIG}),
             ("set_head", {"pan": BIG}),
             ("report_source", {"x": BIG, "y": 0.0}),
             ("step", {"n_ticks": float("inf")}),
             ("step", {"n_ticks": float("nan")}),
             ("step", {"n_ticks": BIG})]
    for method, params in cases:
        srv = PlantCellServer(config=grader_config(), mode="hidden")
        srv.reset(site, seed=7, kind="hidden")
        wire(srv, "step", {"n_ticks": 20})
        before = digest(srv)
        res = wire(srv, method, params)
        after = digest(srv)
        cont = wire(srv, "step", {"n_ticks": 10})
        rec = srv.finalize()
        cf = verify._cell_failed(rec)
        ok = res.get("error") == "bad_params" and before == after and "error" not in cont and not cf \
            and rec["internal_errors"] == 0
        out.append({"call": method, "args": {k: ("10**400" if v is BIG else repr(v)) for k, v in params.items()},
                    "reply": res, "state_unchanged": before == after, "mission_goes_on": "error" not in cont,
                    "cell_failed": cf, "ok": ok})
        if not ok:
            problems.append("review case %s %s: %s" % (method, out[-1]["args"], res))
        srv.close()
    # a report with an oversized value after the robot was lost (the world's own end, as its tilt check calls it)
    srv = PlantCellServer(config=grader_config(), mode="hidden")
    srv.reset(site, seed=7, kind="hidden")
    w = srv.world
    wire(srv, "step", {"n_ticks": 20})
    w.end("robot_lost")
    wire(srv, "get_state", {})                               # the server notes the end (as after any call)
    lost = w.done_reason
    res = wire(srv, "report", {"tag": "PI-FUZZ", "value": BIG, "frame_ids": []})
    res2 = wire(srv, "report", {"tag": "PI-FUZZ", "value": 1.0, "frame_ids": []})
    rec = srv.finalize()
    cf = verify._cell_failed(rec)
    ok = lost == "robot_lost" and res.get("error") in ("bad_params", "mission_over") and \
        res2.get("error") == "mission_over" and not cf
    out.append({"call": "report after robot_lost", "done_reason": lost, "reply": res, "reply_valid_value": res2,
                "cell_failed": cf, "ok": ok})
    if not ok:
        problems.append("report after the robot was lost: %s / %s, cell_failed %s (end %s)" % (res, res2, cf, lost))
    srv.close()
    return out


def part_fuzz(site, n_calls, seed, problems):
    rng = random.Random(seed)
    cfg = grader_config()
    cfg = copy.deepcopy(cfg)
    srv = PlantCellServer(config=cfg, mode="hidden")
    srv.reset(site, seed=11, kind="hidden")
    stats = {"calls": 0, "normal": 0, "errors": {}, "by_method": {}, "refused_changed_state": 0, "missions": 1,
             "ended": {}, "internal_errors": 0, "render_failures": 0, "frames_lost": 0}
    t0 = time.time()

    def tally(rec):
        # every mission's record (not only the last one's): how it ended, the cell's own failures
        why = str(rec.get("done_reason"))
        stats["ended"][why] = stats["ended"].get(why, 0) + 1
        stats["internal_errors"] += int(rec.get("internal_errors") or 0)
        stats["render_failures"] += int(rec.get("render_failures") or 0)
        stats["frames_lost"] += int(rec.get("frames_lost") or 0)
    methods = list(PUBLIC_METHODS) + ["reset", "get_state_privileged", "reference_run", "nope", ""]
    for i in range(n_calls):
        if srv.world.done and rng.random() < 0.02:
            tally(srv.finalize())
            srv.close()
            srv = PlantCellServer(config=cfg, mode="hidden")
            srv.reset(site, seed=11 + i, kind="hidden")
            stats["missions"] += 1
        if rng.random() < 0.45:
            method, params = valid_call(rng, srv)
        else:
            method = rng.choice(methods)
            params = rnd_params(rng, method)
        before = digest(srv)
        try:
            res = wire(srv, method, params)
        except Exception as e:  # noqa: BLE001 - handle() never raises: this is a failure of the test's own encoding
            problems.append("call %d %s raised %r" % (i, method, e))
            continue
        stats["calls"] += 1
        m = stats["by_method"].setdefault(method, {"normal": 0})
        if isinstance(res, dict) and "error" in res:
            code = res["error"]
            stats["errors"][code] = stats["errors"].get(code, 0) + 1
            m[code] = m.get(code, 0) + 1
            if code not in ALLOWED_ERRORS:
                problems.append("call %d %s(%s): %s" % (i, method, json.dumps(params)[:160], res))
            if digest(srv) != before:
                stats["refused_changed_state"] += 1
                problems.append("call %d %s(%s) was refused (%s) but changed the mission" % (
                    i, method, json.dumps(params)[:160], code))
        else:
            stats["normal"] += 1
            m["normal"] += 1
    tally(srv.finalize())
    stats["wall_s"] = round(time.time() - t0, 1)
    if stats["internal_errors"]:
        problems.append("fuzz: internal errors %d" % stats["internal_errors"])
    if stats["ended"].get("physics_fault") or stats["frames_lost"] or stats["render_failures"]:
        problems.append("fuzz: the cell's own failures (physics faults %d, frames lost %d, render failures %d)" % (
            stats["ended"].get("physics_fault", 0), stats["frames_lost"], stats["render_failures"]))
    srv.close()
    return stats


def part_rerun_rule(site, problems):
    import verify
    out = {}
    # (a) a failure of the cell's own work while live
    srv = PlantCellServer(config=grader_config(), mode="hidden")
    srv.reset(site, seed=7, kind="hidden")
    w = srv.world
    real = w.step

    def boom(n=1):
        raise RuntimeError("injected failure of the cell's own work")
    w.step = boom
    res = wire(srv, "step", {"n_ticks": 5})
    w.step = real
    rec = srv.finalize()
    cf = verify._cell_failed(rec)
    out["live"] = {"reply": res, "internal_errors": rec["internal_errors"],
                   "internal_errors_live": rec.get("internal_errors_live"), "done_reason": rec["done_reason"],
                   "cell_failed": cf}
    if not (res.get("error") == "internal_error" and cf == ["internal_error"]):
        problems.append("re-run rule, live failure: %s" % out["live"])
    srv.close()
    # (b) the same after the mission had ended (robot lost)
    srv = PlantCellServer(config=grader_config(), mode="hidden")
    srv.reset(site, seed=7, kind="hidden")
    w = srv.world
    wire(srv, "step", {"n_ticks": 20})
    w.end("robot_lost")
    wire(srv, "get_state", {})
    real = srv.robot.get_state

    def boom2():
        raise RuntimeError("injected failure after the end")
    srv.robot.get_state = boom2
    res = wire(srv, "get_state", {})
    srv.robot.get_state = real
    rec = srv.finalize()
    cf = verify._cell_failed(rec)
    out["after_end"] = {"done_reason": rec["done_reason"], "reply": res, "internal_errors": rec["internal_errors"],
                        "internal_errors_live": rec.get("internal_errors_live"), "cell_failed": cf}
    if not (rec["internal_errors"] == 1 and rec.get("internal_errors_live") == 0 and cf == []):
        problems.append("re-run rule, failure after the end: %s" % out["after_end"])
    srv.close()
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", required=True)
    ap.add_argument("--calls", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)
    problems = []
    out = {"review_cases": part_review_cases(a.site, problems),
           "fuzz": part_fuzz(a.site, a.calls, a.seed, problems),
           "rerun_rule": part_rerun_rule(a.site, problems), "problems": problems}
    print(json.dumps({k: v for k, v in out.items() if k != "review_cases"}, indent=1, default=str)[:6000])
    for c in out["review_cases"]:
        print("%-28s %s" % (c["call"], "ok" if c["ok"] else "FAIL %s" % c))
    if a.json:
        with open(a.json, "w") as f:
            json.dump(out, f, indent=1, default=str)
    print("PASS" if not problems else "FAIL (%d problems)" % len(problems))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
