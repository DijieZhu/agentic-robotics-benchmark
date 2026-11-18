"""The steps after a mission's end: the wire closes after 50 more `step` calls, as robot_api.md says,
in the scored path (cellkit.wire.serve_episode with the request-line guard) and on the practice ground.

    MUJOCO_GL=osmesa python -I tests/review/cell/test_after_end.py --hall H.tar.xz --work DIR

The hall with a 1 s simulated time limit: `step(200)` ends the mission at the limit, then `step(1)` again and again;
counted: the extra steps answered before the wire closes (each counted once since the fix; before it, the cell's own
dispatch counted every step after the end a second time and the wire closed after 25).
"""

import argparse
import json
import os
import queue
import shutil
import socket
import sys

sys.dont_write_bytecode = True        # (no __pycache__ in the task tree from a review run)
TESTS = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TASK = os.path.dirname(TESTS)
PG = os.path.join(TASK, "environment", "practice_ground")
sys.path.insert(0, TESTS)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cellkit import wire  # noqa: E402
from plantcell.server import PlantCellServer  # noqa: E402
from plantcell.wire_guard import GuardedProcess  # noqa: E402
from test_practice import sha, start, stop  # noqa: E402

N_EXTRA = 80


def scored(hall):
    srv = PlantCellServer(config={"time": {"sim_time_limit_s": 1.0}}, mode="hidden")
    srv.reset(hall, seed=3, kind="hidden")
    p = object.__new__(GuardedProcess)
    p._q = queue.Queue()
    p.stalled = False
    p.sent = []

    def reply(obj, timeout=None):
        p.sent.append(obj)
        return True
    p.reply = reply
    lines = [{"id": 1, "method": "step", "params": {"n_ticks": 200}}] + \
            [{"id": 2 + i, "method": "step", "params": {"n_ticks": 1}} for i in range(N_EXTRA)]
    for ln in lines:
        p._q.put(("line", json.dumps(ln) + "\n"))
    p._q.put(("eof", None))
    reason = wire.serve_episode(srv, p, 600.0, 60.0)
    done_at = next((i for i, r in enumerate(p.sent) if (r.get("result") or {}).get("done")), None)
    extra = len(p.sent) - 1 - (done_at or 0)
    srv.close()
    return {"reason": reason, "replies": len(p.sent), "done_in_reply": done_at, "extra_steps_answered": extra}


def practice(hall, work):
    if os.path.isdir(work):
        shutil.rmtree(work)
    os.makedirs(os.path.join(work, "pg", "sites"))
    shutil.copy(os.path.join(PG, "practiced.py"), os.path.join(work, "pg"))
    shutil.copytree(os.path.join(PG, "plantcell"), os.path.join(work, "pg", "plantcell"))
    shutil.copy(hall, os.path.join(work, "pg", "sites", "hall.tar.xz"))
    sites = {"rule": "test", "missions": 3, "open": 2, "program_time_cap_s": 1200.0, "silence_s": 60.0,
             "config": {"time": {"sim_time_limit_s": 2400.0}},
             "sites": [{"name": "hall", "package": "sites/hall.tar.xz", "sha256": sha(hall), "kind": "hall",
                        "sim_time_limit_s": 1.0, "about": "test hall"}]}
    json.dump(sites, open(os.path.join(work, "pg", "sites.json"), "w"), indent=1)
    run = os.path.join(work, "run")
    os.makedirs(run)
    env = dict(os.environ, PLANTCELL_PRACTICE_SOCKET=os.path.join(run, "practice.sock"),
               PLANTCELL_PRACTICE_STATUS=os.path.join(run, "practice_status.json"),
               PLANTCELL_PRACTICE_DEBRIEFS=os.path.join(run, "debriefs"),
               PLANTCELL_PRACTICE_STATE=os.path.join(work, "pg", "state"))
    d = start(work, env)
    try:
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.connect(env["PLANTCELL_PRACTICE_SOCKET"])
        s.settimeout(120.0)
        f = s.makefile("rw", encoding="utf-8", newline="\n")

        def call(method, params, rid):
            f.write(json.dumps({"id": rid, "method": method, "params": params}) + "\n")
            f.flush()
            ln = f.readline()
            return json.loads(ln) if ln else None
        r = call("reset", {"site": "hall"}, 1)
        if not (r or {}).get("result", {}).get("ok"):
            return {"reset": r}
        r = call("step", {"n_ticks": 200}, 2)
        done = bool(((r or {}).get("result") or {}).get("done"))
        extra = 0
        last = None
        for i in range(N_EXTRA):
            try:
                r = call("step", {"n_ticks": 1}, 3 + i)
            except OSError:
                r = None
            if r is None or r.get("event") == "mission_over":
                last = r
                break
            extra += 1
        s.close()
        return {"done_after_first": done, "extra_steps_answered": extra,
                "last": None if last is None else {"event": last.get("event"), "reason": last.get("reason")}}
    finally:
        stop(d)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--hall", required=True)
    ap.add_argument("--work", required=True)
    a = ap.parse_args(argv)
    out = {"scored": scored(a.hall), "practice": practice(a.hall, os.path.abspath(a.work))}
    problems = []
    if out["scored"].get("extra_steps_answered") != 50:
        problems.append("scored wire: %s" % out["scored"])
    if out["practice"].get("extra_steps_answered") != 50:
        problems.append("practice ground: %s" % out["practice"])
    print(json.dumps(out, indent=1))
    for p in problems:
        print("PROBLEM", p)
    print("steps after the end: %s" % ("OK" if not problems else "FAIL (%d problems)" % len(problems)))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
