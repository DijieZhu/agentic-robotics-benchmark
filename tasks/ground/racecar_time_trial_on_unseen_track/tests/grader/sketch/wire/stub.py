"""Protocol stub: replay one lab log as the observations of a session and answer a program with the car's messages.

    python stub.py LOG.jsonl PROGRAM_DIR [--module drive] [--lib DIR] [--report report.json] [--strict]

It starts the program the way the graded run does (its own process, `python -B -E -s` through the carkit launcher,
the program directory as the working directory) and answers every request by the same rules: the five calls, the
error codes (forbidden, bad_params, no_session, bad_order, venue_over), the session order practice -> tt1 -> tt2 ->
avoid, and the budgets (on-track program time per session, desk time before each session, silence). Each session
replays the log from its first row: the reply to the k-th `step` carries the k-th row's observation exactly; the
session ends (`session_end`, reason `time`) after the log's last row. A `request_pickup` is answered as in a graded
run (`stopped` on the next step, `released` after the staff's hold time) but nothing else changes.

It SIMULATES NOTHING: no car, no track, no lap, no warning, no touch or crash. The commands are recorded and
ignored. What it checks is the protocol and the timing of the program; the report (JSON on stdout, or --report)
lists the requests per method, the errors per code, the program time per step (median, p95, max), each session's
on-track time against its cap, each desk time against its cap, the longest silence, and how the run ended.
--strict ends the run at the first exceeded budget, as a graded run would.

Standard library and carkit only.
"""

import argparse
import json
import math
import os
import select
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))


def _find_lib(start):
    d = start
    for _ in range(6):
        if os.path.isfile(os.path.join(d, "carkit", "__init__.py")):
            return d
        d = os.path.dirname(d)
    return None


LIB = _find_lib(HERE)
if LIB and LIB not in sys.path:
    sys.path.insert(0, LIB)

from carkit import protocol  # noqa: E402
from carkit.client import launcher_argv  # noqa: E402
from carkit.logs import read_log  # noqa: E402

TICK_S = 0.025
SESSIONS = (("practice", 60.0, False, [{"name": "mapping", "t0_s": 0.0, "speed_limit_mps": 2.0},
                                        {"name": "free", "t0_s": 120.0, "speed_limit_mps": None}]),
            ("tt1", 300.0, True, [{"name": "heat", "t0_s": 0.0, "speed_limit_mps": None}]),
            ("tt2", 120.0, True, [{"name": "heat", "t0_s": 0.0, "speed_limit_mps": None}]),
            ("avoid", 60.0, True, [{"name": "static", "t0_s": 0.0, "speed_limit_mps": None},
                                    {"name": "dynamic", "t0_s": 90.0, "speed_limit_mps": None}]))
BUDGETS = {"program_ms_per_tick": 15.0, "cpu_factor": 4.0, "silence_s": 180.0}
TIMING = {"tick_s": TICK_S, "cmd_delay_ticks": 1, "imu_samples_per_tick": 5, "subscans_per_tick": 5}
HOLD_S = 20.0
CAR = {"wheelbase_m": 0.25784, "track_front_m": 0.15591, "track_rear_m": 0.15595, "tyre_radius_m": 0.0329,
       "mass_kg": 1.8359, "com_m": [0.12495, -0.00263, 0.05014], "length_m": 0.38776, "width_m": 0.18492,
       "height_m": 0.15252, "steer_max_rad": 0.4189, "steer_rate_max_radps": 3.2, "min_turn_radius_m": 0.5791,
       "final_drive_ratio": 8.3474, "speed_cmd_min_mps": -2.0, "speed_cmd_max_mps": 8.0,
       "lidar": {"pos_m": [0.21315, 0.0, 0.1099], "beams": 1081, "angle_min_rad": -2.35619449,
                 "angle_inc_rad": 0.004363323, "range_min_m": 0.06, "range_max_m": 10.0, "no_return": 0},
       "imu": {"pos_m": [0.0603, -0.0051, 0.078], "rate_hz": 200}}


def _err(code, detail=None):
    out = {"error": code}
    if detail:
        out["detail"] = detail
    return out


class ReplayCell(object):
    """The stub's answers (the venue's state machine without a world)."""

    def __init__(self, log, strict=False):
        self.log = log
        self.n = len(log)
        self.strict = strict
        self.state = "idle"
        self.next_idx = 0
        self.ses = None
        self.counts = {"forbidden_calls": 0, "bad_calls": 0}
        self.violations = []
        self.sessions = []
        self.desk = {}
        self.ended_by = None
        self.last_close_pending = False      # K12: the fourth session ended by session_end, not yet acknowledged

    @property
    def over(self):
        return self.state == "over"

    def describe(self):
        h = self.log.header
        sheet = {"venue_label": "lab log replay: %s" % h["name"], "floor": {"kind": h["floor"]["kind"], "side_pull": None},
                 "walls": {"kinds": ["orange_ribbed", "yellow_ribbed", "black_plain"], "height_m": 0.33},
                 "track": {"min_width_m": 1.0, "max_footprint_m": [30.0, 20.0]}, "features": []}
        dur = self.n * TICK_S
        sessions = [{"name": nm, "duration_s": dur, "ticks": self.n, "desk_before_s": desk, "scored": scored,
                     "phases": [p for p in ph if p["t0_s"] < dur]} for nm, desk, scored, ph in SESSIONS]
        return {"protocol": protocol.PROTOCOL, "venue_sheet": sheet, "sessions": sessions, "budgets": dict(BUDGETS),
                "timing": dict(TIMING), "car": json.loads(json.dumps(CAR))}

    def handle(self, method, params):
        if method not in protocol.PUBLIC_METHODS:
            self.counts["forbidden_calls"] += 1
            return _err("forbidden")
        if self.over and not (method == "close_session" and self.last_close_pending):
            return _err("venue_over", "the venue is over")
        detail = protocol.check_params(method, params)
        if detail:
            self.counts["bad_calls"] += 1
            return _err("bad_params", detail)
        params = params or {}
        if method == "describe":
            return self.describe()
        if method == "open_session":
            if self.state == "open":
                return _err("bad_order", "a session is open; close it first")
            if self.next_idx >= 4 or params["name"] != SESSIONS[self.next_idx][0]:
                return _err("bad_order", "the next session is %s" % SESSIONS[min(self.next_idx, 3)][0])
            name = SESSIONS[self.next_idx][0]
            self.next_idx += 1
            self.ses = {"name": name, "tick": 0, "program_s": 0.0, "desk_s": self.desk.get(name, 0.0), "cmds": [],
                        "stop_at": None, "release_at": None, "pickup": False, "end": None}
            self.sessions.append(self.ses)
            self.state = "open"
            return {"session": name, "duration_s": self.n * TICK_S, "ticks": self.n,
                    "phases": [p for p in SESSIONS[self.next_idx - 1][3] if p["t0_s"] < self.n * TICK_S]}
        if method == "close_session":
            if self.over:                        # K12: acknowledged once after the last session_end, nothing changes
                self.last_close_pending = False
                return {"session": self.ses["name"], "ticks": self.ses["tick"]}
            if self.state not in ("open", "ended"):
                return _err("no_session", "no session is open")
            s = self.ses
            s["end"] = s["end"] or "closed"
            self.state = "over" if self.next_idx >= 4 else "between"
            return {"session": s["name"], "ticks": s["tick"]}
        if method == "request_pickup":
            if self.state != "open":
                return _err("no_session", "no session is open")
            s = self.ses
            if s["pickup"] or (s["stop_at"] is not None and s["release_at"] is not None and s["tick"] < s["release_at"]):
                return {"accepted": False}
            s["pickup"] = True
            return {"accepted": True}
        if method == "step":
            if self.state == "ended":
                return _err("no_session", "the session has ended; close it or open the next one")
            if self.state != "open":
                return _err("no_session", "no session is open")
            s = self.ses
            k = s["tick"]
            s["cmds"].append([params["steer"], params["speed"]])
            events = []
            if s["pickup"]:
                s["pickup"] = False
                s["stop_at"], s["release_at"] = k, k + int(round(HOLD_S / TICK_S))
                events.append({"type": "stopped"})
            if s["release_at"] is not None and k + 1 == s["release_at"]:
                events.append({"type": "released"})
            s["tick"] = k + 1
            obs = self.log.obs(k)
            if s["tick"] >= self.n:
                events.append({"type": "session_end", "reason": "time"})
                s["end"] = "time"
                self.state = "over" if self.next_idx >= 4 else "ended"
                self.last_close_pending = self.state == "over"
            return {"tick": s["tick"], "scan": obs["scan"], "imu": obs["imu"], "wheel_speed": obs["wheel_speed"],
                    "steer_echo": obs["steer_echo"], "events": events}
        return _err("forbidden")

    def account(self, dt):
        """Charge program time as the venue does; returns a violation text (strict mode ends the run on it)."""
        if self.over:
            return None
        if self.state == "open":
            s = self.ses
            s["program_s"] += dt
            cap = BUDGETS["program_ms_per_tick"] * 1e-3 * self.n
            if s["program_s"] > cap and "track:" + s["name"] not in self.violations:
                self.violations.append("track:" + s["name"])
                return "program_time_cap"
        elif self.next_idx < 4:
            name, cap = SESSIONS[self.next_idx][0], SESSIONS[self.next_idx][1]
            self.desk[name] = self.desk.get(name, 0.0) + dt
            if self.desk[name] > cap and "desk:" + name not in self.violations:
                self.violations.append("desk:" + name)
                return "desk_time_cap"
        return None


def _pct(vals, q):
    if not vals:
        return None
    v = sorted(vals)
    return v[min(len(v) - 1, int(math.floor(q * (len(v) - 1) + 0.5)))]


def run(log_path, program_dir, module="drive", lib=None, strict=False, python=None, silence_s=None, max_after=50):
    """Run the program against the replay; returns the report dict."""
    log = read_log(log_path, numpy=False)
    cell = ReplayCell(log, strict=strict)
    lib = lib or LIB
    env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": os.environ.get("HOME", "/tmp"),
           "TMPDIR": os.environ.get("TMPDIR", "/tmp"), "LANG": "C.UTF-8", "PYTHONPATH": lib,
           "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
           "NPY_DISABLE_CPU_FEATURES": "X86_V4 AVX512_ICL AVX512_SPR"}      # as in a graded run (car_api.md)
    errf = tempfile.TemporaryFile()                  # a file, not a pipe: a chatty program cannot block on stderr
    proc = subprocess.Popen(launcher_argv(python or sys.executable, lib, module), cwd=program_dir, env=env,  # noqa: S603
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=errf, bufsize=0,
                            start_new_session=True)
    rfd, wfd = proc.stdout.fileno(), proc.stdin.fileno()
    os.set_blocking(rfd, False)
    silence = float(BUDGETS["silence_s"] if silence_s is None else silence_s)
    buf = bytearray()
    report = {"log": log.header["name"], "log_ticks": cell.n, "requests": 0, "by_method": {}, "errors": {},
              "protocol_errors": [], "ids_sequential": True, "step_program_ms": [], "silence_max_s": 0.0}
    t_last = time.perf_counter()
    first, last_id, end, after = True, 0, None, 0
    while end is None:
        deadline = time.perf_counter() + silence
        line = None
        while line is None:
            i = buf.find(b"\n")
            if i >= 0:
                line = bytes(buf[:i])
                del buf[:i + 1]
                break
            wait = deadline - time.perf_counter()
            if wait <= 0:
                end = "silence"
                break
            r, _, _ = select.select([rfd], [], [], wait)
            if r:
                chunk = os.read(rfd, 1 << 20)
                if not chunk:
                    end = "eof"
                    break
                buf += chunk
        if end is not None:
            break
        now = time.perf_counter()
        dt = now - t_last
        try:
            obj = json.loads(line)
        except ValueError:
            obj = None
        env_err = protocol.check_envelope(obj)
        if env_err is not None and env_err[0] == "protocol":
            report["protocol_errors"].append(env_err[1] + ": " + line[:80].decode("utf-8", "replace"))
            end = "protocol"
            break
        if not first:
            report["silence_max_s"] = max(report["silence_max_s"], dt)
            v = cell.account(dt)
            if v is not None and strict:
                end = v
                break
        first = False
        report["requests"] += 1
        if cell.over:
            after += 1
            if after > max_after:
                end = "eof"
                break
        if env_err is not None:
            reply = protocol.error_reply(obj.get("id"), "bad_params", env_err[1])
            cell.counts["bad_calls"] += 1
        else:
            if obj["id"] != last_id + 1:
                report["ids_sequential"] = False
            last_id = obj["id"]
            m = obj["method"]
            report["by_method"][m] = report["by_method"].get(m, 0) + 1
            if m == "step" and cell.state == "open":
                report["step_program_ms"].append(dt * 1e3)
            res = cell.handle(m, obj.get("params"))
            reply = ({"id": obj["id"], "error": res["error"]} if "error" in res and set(res) <= {"error", "detail"}
                     else {"id": obj["id"], "result": res})
            if "error" in reply and res.get("detail"):
                reply["detail"] = res["detail"]
        if "error" in reply:
            report["errors"][reply["error"]] = report["errors"].get(reply["error"], 0) + 1
        t_last = time.perf_counter()
        try:
            os.write(wfd, (json.dumps(reply, separators=(",", ":")) + "\n").encode("utf-8"))
        except OSError:
            end = "eof"
    if end == "eof" and cell.over:
        end = "finish"
    try:
        proc.stdin.close()
        proc.wait(timeout=5.0)
    except Exception:  # noqa: BLE001
        proc.kill()
    proc.stdout.close()
    errf.seek(0)
    report["stderr_tail"] = errf.read().decode("utf-8", "replace")[-2000:]
    errf.close()
    sp = report.pop("step_program_ms")
    report["program_ms_per_step"] = {"n": len(sp), "median": _pct(sp, 0.5), "p95": _pct(sp, 0.95), "max": max(sp) if sp else None}
    cap = BUDGETS["program_ms_per_tick"] * 1e-3 * cell.n
    report["sessions"] = [{"name": s["name"], "steps": s["tick"], "end": s["end"], "program_s": round(s["program_s"], 6),
                           "cap_s": cap, "within_cap": s["program_s"] <= cap, "commands": len(s["cmds"])} for s in cell.sessions]
    report["desk"] = [{"before": nm, "desk_s": round(cell.desk.get(nm, 0.0), 6), "cap_s": d,
                       "within_cap": cell.desk.get(nm, 0.0) <= d} for nm, d, _, _ in SESSIONS]
    report["forbidden_calls"] = cell.counts["forbidden_calls"]
    report["bad_calls"] = cell.counts["bad_calls"]
    report["violations"] = list(cell.violations)
    report["end"] = end
    report["exit_code"] = proc.returncode
    report["note"] = "protocol and timing only: the stub simulates nothing"
    return report


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("log")
    ap.add_argument("program_dir")
    ap.add_argument("--module", default="drive")
    ap.add_argument("--lib", default=None, help="directory that holds carkit/ (default: found next to this file)")
    ap.add_argument("--report", default=None)
    ap.add_argument("--strict", action="store_true")
    a = ap.parse_args(argv)
    rep = run(a.log, a.program_dir, module=a.module, lib=a.lib, strict=a.strict)
    text = json.dumps(rep, indent=1)
    if a.report:
        with open(a.report, "w", encoding="utf-8") as f:
            f.write(text + "\n")
    else:
        print(text)
    ok = rep["end"] == "finish" and not rep["violations"] and not rep["protocol_errors"]
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
