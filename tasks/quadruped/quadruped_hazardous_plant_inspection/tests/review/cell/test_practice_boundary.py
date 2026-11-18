"""The practice ground against the boundary test's inputs (environment/practice_ground/practiced.py): no request
crashes the service, refused requests cost no practice mission, and a mission sees the cell's boundary check.

    MUJOCO_GL=osmesa python -I tests/review/cell/test_practice_boundary.py --hall H.tar.xz --work DIR

A private copy of the practice ground (as test_practice.py) with an allowance of 3 missions. Over raw socket lines:
  1. a line that is not JSON, a line nested 100,000 deep, a request whose id is a 4000-digit integer or a list nested
     500 deep, a method that is not a name: each ends that connection (protocol) or is refused, and the service
     answers the next connection;
  2. reset(site) with Infinity, 10**400, true, [1], {"a": 1}, a 10,000-character name, and params that are not an
     object: bad_params each, no mission used;
  3. one valid mission on the hall: report value 10**400, walk vx 10**400, set_body height 10**400, set_head pan
     10**400, report_source x 10**400, report_mark dose_rate 10**400, step n_ticks Infinity / 10**400 / NaN: bad_params
     each, then a normal step and finish; the debrief arrives with `rule_point` for every instrument;
  4. the ledger: exactly one mission used; the service still answers.
"""

import argparse
import hashlib
import json
import os
import shutil
import socket
import subprocess
import sys
import time

sys.dont_write_bytecode = True        # (no __pycache__ in the task tree from a review run)
TESTS = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TASK = os.path.dirname(TESTS)
PG = os.path.join(TASK, "environment", "practice_ground")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_practice import start, stop  # noqa: E402

BIG = 10 ** 400


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


class Raw(object):
    def __init__(self, path):
        self.s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.s.connect(path)
        self.s.settimeout(120.0)
        self.buf = b""

    def send_line(self, raw):
        self.s.sendall(raw if isinstance(raw, bytes) else raw.encode("utf-8"))

    def call(self, method, params, rid=1):
        self.send_line(json.dumps({"id": rid, "method": method, "params": params}) + "\n")
        return self.read()

    def read(self):
        while b"\n" not in self.buf:
            chunk = self.s.recv(1 << 20)
            if not chunk:
                return None
            self.buf += chunk
        line, self.buf = self.buf.split(b"\n", 1)
        return json.loads(line)

    def close(self):
        try:
            self.s.close()
        except OSError:
            pass


def call_once(sock, method, params, tries=40):
    """One request on a fresh connection; `busy` (the connections before may still be closing: at most 2 at a time,
    and on a loaded host their handlers take a while to exit) is retried for up to 20 s, as a program would."""
    res = None
    for _ in range(tries):
        r = Raw(sock)
        try:
            res = r.call(method, params)
        except (OSError, ValueError) as e:
            res = "connection: %s" % type(e).__name__
        finally:
            r.close()
        if isinstance(res, dict) and res.get("error") == "busy":
            time.sleep(0.5)
            continue
        break
    return res


def alive(sock):
    res = call_once(sock, "reset", {"site": "no-such-site"})
    return isinstance(res, dict) and res.get("error") == "bad_params"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--hall", required=True)
    ap.add_argument("--work", required=True)
    a = ap.parse_args(argv)
    problems = []
    work = os.path.abspath(a.work)
    if os.path.isdir(work):
        shutil.rmtree(work)
    os.makedirs(os.path.join(work, "pg", "sites"))
    shutil.copy(os.path.join(PG, "practiced.py"), os.path.join(work, "pg"))
    shutil.copytree(os.path.join(PG, "plantcell"), os.path.join(work, "pg", "plantcell"))
    shutil.copy(a.hall, os.path.join(work, "pg", "sites", "hall.tar.xz"))
    sites = {"rule": "test", "missions": 3, "open": 2, "program_time_cap_s": 1200.0, "silence_s": 180.0,
             "config": {"time": {"sim_time_limit_s": 2400.0}},
             "sites": [{"name": "hall", "package": "sites/hall.tar.xz", "sha256": sha(a.hall), "kind": "hall",
                        "sim_time_limit_s": 900.0, "about": "test hall"}]}
    json.dump(sites, open(os.path.join(work, "pg", "sites.json"), "w"), indent=1)
    run = os.path.join(work, "run")
    os.makedirs(run)
    env = dict(os.environ, PLANTCELL_PRACTICE_SOCKET=os.path.join(run, "practice.sock"),
               PLANTCELL_PRACTICE_STATUS=os.path.join(run, "practice_status.json"),
               PLANTCELL_PRACTICE_DEBRIEFS=os.path.join(run, "debriefs"),
               PLANTCELL_PRACTICE_STATE=os.path.join(work, "pg", "state"))
    sock, status = env["PLANTCELL_PRACTICE_SOCKET"], env["PLANTCELL_PRACTICE_STATUS"]
    d = start(work, env)
    out = {}
    try:
        # 1. lines the service must survive
        lines = {"not json": b"this is not json\n", "nested 100000": b"[" * 100000 + b"]" * 100000 + b"\n",
                 "id 4000 digits": (json.dumps({"id": 10 ** 3999, "method": "reset", "params": {"site": "x"}}) +
                                    "\n").encode(),
                 "id nested 500": ('{"id": ' + "[" * 500 + "]" * 500 + ', "method": "reset", "params": {}}\n').encode(),
                 "method not a name": b'{"id": 1, "method": {"a": [1, 2]}, "params": {}}\n'}
        for name, raw in lines.items():
            r = Raw(sock)
            try:
                r.send_line(raw)
                res = r.read()
            except (OSError, ValueError) as e:
                res = "connection: %s" % type(e).__name__
            r.close()
            ok_alive = alive(sock)
            out[name] = {"reply": res if not isinstance(res, dict) else {k: res[k] for k in res if k != "debrief"},
                         "service_alive": ok_alive}
            if not ok_alive:
                problems.append("after %r the service does not answer" % name)
        # 2. resets with odd sites: bad_params, no mission
        for name, params in (("Infinity", {"site": float("inf")}), ("10**400", {"site": BIG}), ("true", {"site": True}),
                             ("[1]", {"site": [1]}), ("dict", {"site": {"a": 1}}), ("long", {"site": "x" * 10000}),
                             ("params list", [1, 2]), ("params number", 5)):
            res = call_once(sock, "reset", params)
            out["reset " + name] = res
            if not (isinstance(res, dict) and res.get("error") == "bad_params"):
                problems.append("reset(%s) answered %s" % (name, res))
        st = json.load(open(status))
        if st["missions_used"] != 0:
            problems.append("refused resets used %d missions" % st["missions_used"])
        # 3. one mission with the oversized arguments
        res = None
        for _ in range(40):                    # the connections above may still be closing: `busy` until they are
            r = Raw(sock)
            res = r.call("reset", {"site": "hall"})
            if isinstance(res, dict) and res.get("error") == "busy":
                r.close()
                time.sleep(0.5)
                continue
            break
        out["valid reset"] = res
        if not (isinstance(res, dict) and res.get("result", {}).get("ok")):
            problems.append("valid reset: %s" % res)
            raise RuntimeError("no mission")
        cases = [("report", {"tag": "PI-211", "value": BIG, "frame_ids": []}), ("walk", {"vx": BIG}),
                 ("set_body", {"height": BIG}), ("set_head", {"pan": BIG}), ("report_source", {"x": BIG, "y": 0.0}),
                 ("report_mark", {"tag": "SM-01", "dose_rate": BIG}), ("step", {"n_ticks": float("inf")}),
                 ("step", {"n_ticks": BIG}), ("step", {"n_ticks": float("nan")})]
        replies = []
        for i, (m, p) in enumerate(cases):
            rep = r.call(m, p, rid=10 + i)
            replies.append([m, rep])
            if not (isinstance(rep, dict) and rep.get("error") == "bad_params"):
                problems.append("practice %s %s: %s" % (m, list(p), rep))
        rep = r.call("step", {"n_ticks": 50}, rid=30)
        if not (isinstance(rep, dict) and "result" in rep):
            problems.append("a normal step after the refusals: %s" % rep)
        rep = r.call("finish", {}, rid=31)
        last = r.read()                       # the service's last word: mission_over with the debrief
        r.close()
        out["mission"] = {"replies": replies, "finish": rep, "last": None if last is None else {
            "event": last.get("event"), "reason": last.get("reason")}}
        db = (last or {}).get("debrief") or {}
        rows = [x for x in db.get("items", []) if x.get("type") not in ("dose_point", "hotspot")]
        if not rows or any("rule_point" not in x for x in rows):
            problems.append("debrief rows without rule_point: %s" % [x.get("tag") for x in rows
                                                                      if "rule_point" not in x])
        out["debrief_rule_points"] = {x["tag"]: [x.get("true_position"), x.get("rule_point")] for x in rows}
        # 4. the ledger and the service
        time.sleep(1.0)
        st = json.load(open(status))
        out["missions_used"] = st["missions_used"]
        if st["missions_used"] != 1:
            problems.append("missions used %d (expected 1)" % st["missions_used"])
        if not alive(sock):
            problems.append("the service does not answer at the end")
    except RuntimeError:
        pass
    finally:
        stop(d)
    out["problems"] = problems
    for k, v in out.items():
        if k in ("debrief_rule_points", "mission"):
            continue
        print("%-22s %s" % (k, json.dumps(v, default=str)[:160]))
    if "mission" in out:
        print("mission: %d refused calls, finish %s, last %s" % (len(out["mission"]["replies"]),
                                                              json.dumps(out["mission"]["finish"])[:80],
                                                              out["mission"]["last"]))
        print("rule points: %d instruments" % len(out.get("debrief_rule_points", {})))
    print("PASS" if not problems else "FAIL (%d problems)" % len(problems))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
