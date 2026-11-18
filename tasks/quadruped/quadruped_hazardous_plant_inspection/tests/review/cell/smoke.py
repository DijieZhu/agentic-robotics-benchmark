"""Smoke test of the cell from a clean process with only the task's tests/ folder on the import path (no source tree):
build a PlantCellServer on a site package, run a scripted program through the client proxy (an in-process handler with
the wire's JSON round trip) that makes every one of the 17 public calls, then the refusals (budget, bad arguments,
forbidden, after the end), finalize, the trajectory and the practice debrief.

    MUJOCO_GL=osmesa python -I tests/review/cell/smoke.py SITE [--seed N] [--json OUT]

Exit 0 when every check passes. Prints the timings (build, first frame per sensor, physics per simulated second).
"""

import argparse
import json
import math
import os
import sys
import time

TESTS = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, TESTS)

from plantcell.client import CellError, connect_handler  # noqa: E402
from plantcell.server import PlantCellServer  # noqa: E402

PUBLIC = ["describe", "mission", "get_state", "get_camera", "get_depth", "get_thermal", "get_lidar", "walk", "set_mode",
          "set_body", "set_head", "set_lamp", "step", "report", "report_mark", "report_source", "finish"]


def expect_error(fn, code, problems, what):
    try:
        fn()
    except CellError as e:
        if e.code != code:
            problems.append("%s: error %s, expected %s" % (what, e.code, code))
        return
    problems.append("%s: no error (expected %s)" % (what, code))


def run(site, seed=None):
    problems = []
    timing = {}
    t0 = time.time()
    with open(os.path.join(TESTS, "grader_config.json")) as f:
        gcfg = json.load(f)
    srv = PlantCellServer(config={"time": gcfg["time"]}, mode="hidden")
    srv.reset(site, seed=seed)
    timing["build_s"] = round(time.time() - t0, 2)
    r = connect_handler(srv.handle)
    d = r.describe()
    m = r.mission()
    st = r.get_state()
    for k in ("robot", "modes", "cameras", "lidar", "head", "time", "budgets", "dosimeter", "odometry", "link"):
        if k not in d:
            problems.append("describe() lacks %s" % k)
    lim = float(gcfg["time"]["sim_time_limit_s"])
    if d.get("time", {}).get("sim_time_limit_s") != lim or m.get("budgets", {}).get("sim_time_limit_s") != lim:
        problems.append("describe() / mission() quote the time limit %s / %s, the grader's is %s" % (
            d.get("time", {}).get("sim_time_limit_s"), m.get("budgets", {}).get("sim_time_limit_s"), lim))
    if "capability" in json.dumps(d).lower() or "solo" in json.dumps(d).lower():
        problems.append("describe() names the generator's bands or the robot's origin")
    items = m["work_order"]["items"]
    inst = [it for it in items if it["type"] not in ("dose_point", "hotspot")]
    marks = [it for it in items if it["type"] == "dose_point"]
    if abs(sum(float(it.get("weight", 0)) for it in items) - 1.0) > 1e-3:
        problems.append("work order weights do not sum to 1")
    if not (abs(st["odometry"]["x"] - m["datum"]["pos"][0]) < 1e-6 and abs(st["odometry"]["y"] - m["datum"]["pos"][1]) < 1e-6):
        problems.append("odometry does not start at the datum: %s" % st["odometry"])
    # motion
    r.set_mode("walk")
    t = time.time()
    r.walk(0.3, 0.0, 0.0)
    ev = []
    for _ in range(10):
        out = r.step(100)
        ev += out["events"]
    timing["physics_wall_per_sim_s_walk"] = round((time.time() - t) / 10.0, 3)
    r.walk(0.0, 0.0, 0.0)
    r.step(80)
    r.set_mode("stand")
    r.step(60)
    r.set_body(height=0.24, pitch=0.05)
    hd = r.set_head(pan=0.2, tilt=-0.1)
    r.set_lamp(True)
    r.step(40)
    st2 = r.get_state()
    moved = math.hypot(st2["odometry"]["x"] - st["odometry"]["x"], st2["odometry"]["y"] - st["odometry"]["y"])
    if moved < 0.5:
        problems.append("the robot did not walk (odometry moved %.2f m)" % moved)
    # sensors: one of each, timed
    for name, fn in (("nav", lambda: r.get_camera("nav", depth=True)), ("inspect", lambda: r.get_camera("inspect")),
                     ("depth", r.get_depth), ("thermal", r.get_thermal), ("lidar", r.get_lidar)):
        t = time.time()
        fr = fn()
        timing["first_%s_s" % name] = round(time.time() - t, 3)
        if name in ("nav", "inspect"):
            img = fr["image"]
            if img.ndim != 3 or img.shape[2] != 3 or str(img.dtype) != "uint8" or img.mean() < 2:
                problems.append("%s image is not an RGB frame" % name)
        if name == "nav" and (fr.get("depth") is None or not (fr["depth"] == fr["depth"]).any()):
            problems.append("nav frame has no depth")
        if name == "depth" and not (fr["depth"] == fr["depth"]).any():
            problems.append("depth frame is empty")
        if name == "thermal" and not (10.0 < float(fr["image_c"].mean()) < 60.0):
            problems.append("thermal frame mean %.1f C" % float(fr["image_c"].mean()))
        if name == "lidar" and len(fr["ranges"]) != int(d["lidar"]["beams"]):
            problems.append("lidar has %d beams" % len(fr["ranges"]))
    t = time.time()
    fr2 = r.get_camera("inspect")
    timing["second_inspect_s"] = round(time.time() - t, 3)
    nav = r.get_camera("nav", depth=False)
    # reports: value-free with a frame, a mark, the source
    rep = r.report(inst[0]["tag"], None, [fr2["frame_id"], nav["frame_id"]])
    if not rep.get("ok") or "report_id" not in rep:
        problems.append("report reply %s" % rep)
    if marks:
        r.report_mark(marks[0]["tag"], 0.5)
    r.report_source(1.0, 2.0)
    # refusals change nothing
    used = r.get_state()["frames_used"]
    expect_error(lambda: r.get_camera("mast"), "bad_params", problems, "unknown camera")
    expect_error(lambda: r.set_mode("run"), "bad_params", problems, "unknown mode")
    expect_error(lambda: r.report(inst[0]["tag"], 1.0, ["F1", "F2", "F3", "F4"]), "bad_params", problems, "4 frames")
    expect_error(lambda: r.report(marks[0]["tag"], 1.0, []) if marks else (_ for _ in ()).throw(CellError("bad_params")),
                 "bad_params", problems, "report on a mark")
    expect_error(lambda: r.step(0), "bad_params", problems, "step(0)")
    expect_error(lambda: r.walk(float("nan")), "bad_params", problems, "walk(nan)")
    expect_error(lambda: r._call("reference_run"), "forbidden", problems, "reference_run without the token")
    expect_error(lambda: r._call("get_state_privileged"), "forbidden", problems, "get_state_privileged without the token")
    expect_error(lambda: r.reset("t1"), "forbidden", problems, "reset in a scored mission")
    expect_error(lambda: r._call("debug_dump"), "forbidden", problems, "unknown method")
    if r.get_state()["frames_used"] != used:
        problems.append("a refused call changed the frame count")
    # a frame budget: a server with a budget of 1 inspect frame refuses the second and counts nothing
    srv_b = PlantCellServer(config={"budgets": {"frames": {"inspect": 1}}})
    srv_b.reset(site, seed=seed)
    rb = connect_handler(srv_b.handle)
    rb.get_camera("inspect")
    expect_error(lambda: rb.get_camera("inspect"), "budget", problems, "inspect budget")
    if rb.get_state()["frames_used"]["inspect"] != 1:
        problems.append("the refused frame was counted")
    srv_b.close()
    # the end
    fin = r.finish()
    if not srv.done:
        problems.append("finish did not end the mission")
    expect_error(lambda: r.walk(0.1), "mission_over", problems, "walk after finish")
    expect_error(lambda: r.get_camera("nav"), "mission_over", problems, "camera after finish")
    expect_error(lambda: r.report(inst[0]["tag"], 1.0, []), "mission_over", problems, "report after finish")
    out = r.step(10)
    if not out["done"]:
        problems.append("step after the end is not done")
    srv.program_used = 1.0
    summ = srv.finalize()
    for k in ("episode_score", "step_calls", "calls", "illegal_call_count", "internal_errors", "parts", "sim_time_s",
              "done_reason", "frames_lost", "final_state_sha256"):
        if k not in summ:
            problems.append("finalize() lacks %s" % k)
    if summ.get("episode_score") != 0.0:
        problems.append("the scripted program scored %s (expected 0)" % summ.get("episode_score"))
    if summ.get("illegal_call_count") != 4:
        problems.append("illegal calls counted %s (expected 4)" % summ.get("illegal_call_count"))
    if summ.get("internal_errors"):
        problems.append("internal errors: %s" % summ.get("internal_errors"))
    json.dumps(summ, allow_nan=False)
    tr = srv.trajectory()
    if tr.ndim != 2 or tr.shape[1] != len(summ["trajectory_columns"]) or tr.shape[0] < 10:
        problems.append("trajectory shape %s" % (tr.shape,))
    db = srv.debrief(mission=1, true_pose_hz=2.0)
    json.dumps(db, allow_nan=False)
    if "score" in json.dumps(db).lower():
        problems.append("the debrief mentions a score")
    if len(db.get("items", [])) != len(items) or "true_pose" not in db:
        problems.append("debrief items %d of %d, true pose %s" % (len(db.get("items", [])), len(items), "true_pose" in db))
    calls = {k for k in summ["calls"] if k in PUBLIC}
    if calls != set(PUBLIC):
        problems.append("calls not made: %s" % sorted(set(PUBLIC) - calls))
    srv.close()
    return problems, timing, summ, db, fin, hd


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("site")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)
    t0 = time.time()
    problems, timing, summ, db, fin, hd = run(a.site, a.seed)
    timing["total_s"] = round(time.time() - t0, 1)
    print(json.dumps({"timing": timing, "end": summ.get("done_reason"), "sim_time_s": summ.get("sim_time_s"),
                      "episode_score": summ.get("episode_score"), "frames": summ.get("frames"),
                      "calls": summ.get("calls"), "illegal": summ.get("illegal_call_count")}, indent=1))
    for p in problems:
        print("PROBLEM", p)
    print("smoke: %s (%d problems)" % ("OK" if not problems else "FAIL", len(problems)))
    if a.json:
        with open(a.json, "w") as f:
            json.dump({"timing": timing, "problems": problems, "summary": summ, "debrief": db}, f, indent=1)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
