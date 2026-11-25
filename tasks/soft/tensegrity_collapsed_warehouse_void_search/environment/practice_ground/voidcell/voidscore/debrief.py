"""The practice ground's debrief of one episode (given/station_api.md, The practice ground; the practice ground only, never a scored
episode).

Two kinds, by the site's kind in the practice ground's sites.json:
  cal        the calibration site (every dimension public): the robot's six true node positions twice a second (under
             the roofed room too) and every primitive's true SE(2) displacement (in the frame of its start pose:
             x = heading, y = heading + 90 deg; heading = the authors' SE(2): direction of mean(nodes 0, 2, 4) -
             mean(nodes 1, 3, 5)).
  practice   a practice site of the scored family: for every report the true distance, whether the dwell counted (and
             which condition failed), the credit; whether the robot was retrieved; the stuck periods; the budget use;
             the events. No layout, no trajectory, no position, no total.
The keys are fixed (SCHEMA; the tests check them exactly). A scored (hidden) episode has no debrief: `build` refuses
kind "hidden" and a server in mode "hidden", and nothing in tests/verify.py calls it.

    rec = Recorder(server)                  # after server.reset(...): exact poses at every command's start and end
    db = build(server, rec, kind, acc, episode=n, ended_by=reason)
Every threshold (stuck rule, sampling rate, decimals) comes from acceptance.yaml.
"""

import math

from . import scoring

__all__ = ["SCHEMA", "Recorder", "build", "DebriefError"]

DEBRIEF_VERSION = 1
COMMON = ("debrief_version", "kind", "site", "episode", "ended_by", "end_reason", "t_end", "snapshot")
SCHEMA = {
    "cal": COMMON + ("true_nodes", "primitives"),
    "practice": COMMON + ("reports", "retrieval", "stuck_periods", "budgets", "events"),
}
NESTED = {
    ("cal", "true_nodes"): ("hz", "t", "nodes_xyz"),
    ("cal", "primitives"): ("name", "bottom_nodes", "t_start", "t_end", "aborted", "dx_m", "dy_m", "dtheta_deg"),
    ("practice", "reports"): ("t", "point", "accepted", "counted", "distance_m", "dwell_valid", "dwell_failed",
                              "credit"),
    ("practice", "retrieval"): ("retrieved", "within_radius", "command_running_at_end"),
    ("practice", "stuck_periods"): ("t_start", "t_end", "primitives"),
    ("practice", "budgets"): ("frames_used", "frames_limit", "frames_refused", "reports_used", "reports_limit",
                              "reports_refused", "sim_time_s", "time_limit_s", "program_time_s", "program_time_limit_s"),
    ("practice", "events"): ("t", "event", "detail"),
}


class DebriefError(RuntimeError):
    pass


def _se2(nodes):
    """(cx, cy, heading) of six node positions: the authors' SE(2) (as ysim.metrics.se2)."""
    xs = [p[0] for p in nodes]
    ys = [p[1] for p in nodes]
    cx, cy = sum(xs) / 6.0, sum(ys) / 6.0
    ax = (nodes[0][0] + nodes[2][0] + nodes[4][0]) / 3.0 - (nodes[1][0] + nodes[3][0] + nodes[5][0]) / 3.0
    ay = (nodes[0][1] + nodes[2][1] + nodes[4][1]) / 3.0 - (nodes[1][1] + nodes[3][1] + nodes[5][1]) / 3.0
    return cx, cy, math.atan2(ay, ax)


def _relative(n0, n1):
    """(dx, dy, dtheta_deg) of the end pose in the frame of the start pose."""
    x0, y0, h0 = _se2(n0)
    x1, y1, h1 = _se2(n1)
    tx, ty = x1 - x0, y1 - y0
    c, s = math.cos(h0), math.sin(h0)
    dth = (h1 - h0 + math.pi) % (2.0 * math.pi) - math.pi
    return c * tx + s * ty, -s * tx + c * ty, math.degrees(dth)


class Recorder(object):
    """Exact true node positions at every command's start and end (an observer of the cell's executor, attached
    after the server's reset; it only reads the state). A command is accepted between two control periods, so its
    start pose is the state after the period before; a stop ends it between periods (its end pose: the same); a
    command that finishes inside a period ends at the state after it."""

    def __init__(self, server):
        self.server = server
        self.X = server.X
        self.last = self._nodes()
        self.open = None                 # (cmd object, start nodes)
        self.seen = len(self.X.history)
        self.rows = []                   # one per finished command: (history entry, start nodes, end nodes, bottom)
        self.X.observers.append(self._on_tick)

    def _nodes(self):
        return [[float(v) for v in row] for row in self.server.S.nodes()]

    def _bottom(self):
        return self.server.robot_truth.bottom()

    def _sync(self, after_tick):
        X = self.X
        cur = self._nodes() if after_tick else self.last
        while self.seen < len(X.history):
            h = X.history[self.seen]
            self.seen += 1
            if self.open is not None and self.open[0] is h:
                start = self.open[1]
                self.open = None
            else:
                start = self.last                       # accepted and ended between two periods
            end = self.last if (h.get("aborted") or not after_tick) else cur
            self.rows.append((h, start, end, self._bottom()))
        if X.cmd is not None and (self.open is None or self.open[0] is not X.cmd):
            self.open = (X.cmd, self.last)
        if after_tick:
            self.last = cur

    def _on_tick(self, X):
        self._sync(True)

    def flush(self):
        """The commands ended since the last control period (a stop after it)."""
        self._sync(False)
        return self.rows


def _r(x, n):
    return None if x is None else round(float(x), n)


def _reports(score, acc):
    dm, ds = int(acc["debrief"]["decimals_m"]), int(acc["debrief"]["decimals_s"])
    out = []
    for p in score["points"]:
        acc_ = bool(p["accepted"])
        out.append({"t": _r(p["t"], ds), "point": int(p["i"]), "accepted": acc_, "counted": bool(p["counted"]),
                    "distance_m": _r(p.get("d_m"), dm) if acc_ else None,
                    "dwell_valid": bool(p["dwell_valid"]) if acc_ else None,
                    "dwell_failed": list(p["dwell_failed"]) if acc_ else None,
                    "credit": _r(p.get("c"), 6) if acc_ else None})
    return out


def _primitive_rows(recorder, acc):
    dm, dd, ds = int(acc["debrief"]["decimals_m"]), int(acc["debrief"]["decimals_deg"]), int(acc["debrief"]["decimals_s"])
    out = []
    for h, n0, n1, _b in recorder.flush():
        if h.get("kind") != "primitive":
            continue
        dx, dy, dth = _relative(n0, n1)
        out.append({"name": str(h["name"]), "bottom_nodes": [int(v) for v in h["bottom"]], "t_start": _r(h["t_start"], ds),
                    "t_end": _r(h["t_end"], ds), "aborted": bool(h.get("aborted")), "dx_m": _r(dx, dm),
                    "dy_m": _r(dy, dm), "dtheta_deg": _r(dth, dd)})
    return out


def _stuck(recorder, acc):
    n = int(acc["stuck"]["primitives"])
    mv, tn = float(acc["stuck"]["move_m"]), float(acc["stuck"]["turn_deg"])
    ds = int(acc["debrief"]["decimals_s"])
    prims = []
    for h, n0, n1, _b in recorder.flush():
        if h.get("kind") != "primitive":
            continue
        dx, dy, dth = _relative(n0, n1)
        prims.append((float(h["t_start"]), float(h["t_end"]), math.hypot(dx, dy) < mv and abs(dth) < tn))
    periods = []
    k = 0
    while k < len(prims):
        if not prims[k][2]:
            k += 1
            continue
        j = k
        while j + 1 < len(prims) and prims[j + 1][2]:
            j += 1
        if j - k + 1 >= n:
            periods.append({"t_start": _r(prims[k][0], ds), "t_end": _r(prims[j][1], ds), "primitives": j - k + 1})
        k = j + 1
    return periods


def _events(server, rec, recorder, acc):
    ds = int(acc["debrief"]["decimals_s"])
    ev = []
    for c in rec.get("commands") or []:
        if c.get("kind") in ("primitive", "set_targets") and c.get("accepted") is False:
            ev.append((float(c["t"]), "command_refused", "%s: %s" % (c["kind"], c.get("error"))))
        if c.get("kind") == "stop" and c.get("aborted"):
            ev.append((float(c["t"]), "command_stopped", "stop"))
    for h, _n0, _n1, b in recorder.flush():
        if h.get("kind") == "primitive" and b is not None and b.get("end_triangle"):
            ev.append((float(h["t_end"]), "end_triangle", "the robot rests on an end triangle after %s" % h["name"]))
    for e in rec.get("events") or []:
        if len(e) >= 3 and e[1] == "end":
            ev.append((float(e[0] if e[0] is not None else rec.get("sim_time_s") or 0.0), "end", str(e[2])))
    ev.sort(key=lambda q: q[0])
    return [{"t": _r(t, ds), "event": k, "detail": d} for t, k, d in ev]


def build(server, recorder, kind, acc, episode=None, ended_by=None, snapshot=False):
    """The debrief (a JSON-able dict, keys exactly SCHEMA[kind]) of the server's current or finished episode."""
    if kind not in SCHEMA:
        raise DebriefError("no debrief for kind %r (a scored episode has none)" % (kind,))
    if getattr(server, "mode", None) == "hidden":
        raise DebriefError("no debrief in a scored (hidden) episode")
    if snapshot:
        rec = _snapshot_record(server)
    else:
        rec = server.finalize()
    ds = int(acc["debrief"]["decimals_s"])
    out = {"debrief_version": DEBRIEF_VERSION, "kind": kind, "site": server.site_name, "episode": episode,
           "ended_by": ended_by, "end_reason": rec.get("done_reason"), "t_end": _r(rec.get("sim_time_s"), ds),
           "snapshot": bool(snapshot)}
    if kind == "cal":
        hz = float(acc["debrief"]["cal_true_nodes_hz"])
        L = server.log10
        every = 1.0 / hz
        ts, ns = [], []
        dm = int(acc["debrief"]["decimals_m"])
        for t, n in zip(L["t"], L["nodes"]):
            k = round(float(t) / every)
            if abs(float(t) - k * every) < 1e-6:
                ts.append(_r(t, ds))
                ns.append([[_r(v, dm) for v in row] for row in n])
        out["true_nodes"] = {"hz": hz, "t": ts, "nodes_xyz": ns}
        out["primitives"] = _primitive_rows(recorder, acc)
    else:
        sc = scoring.score_record(rec, server.site, acc)
        out["reports"] = _reports(sc, acc)
        out["retrieval"] = {k: bool(sc["retrieval"][k]) for k in NESTED[("practice", "retrieval")]}
        out["stuck_periods"] = _stuck(recorder, acc)
        b = rec.get("budgets") or {}
        lim = b.get("limits") or {}
        out["budgets"] = {"frames_used": int(b.get("frames_used", 0)), "frames_limit": int(lim.get("frames", 0)),
                          "frames_refused": int(b.get("frames_refused", 0)),
                          "reports_used": {str(k): int(v) for k, v in (b.get("reports") or {}).items()},
                          "reports_limit": int(lim.get("reports_per_point", 0)),
                          "reports_refused": int(b.get("reports_refused", 0)),
                          "sim_time_s": _r(rec.get("sim_time_s"), ds), "time_limit_s": _r(rec.get("sim_time_limit_s"), ds),
                          "program_time_s": _r(rec.get("program_time_s"), ds),
                          "program_time_limit_s": _r(lim.get("program_time_s"), ds)}
        out["events"] = _events(server, rec, recorder, acc)
    check(out)
    return out


def _snapshot_record(server):
    """What finalize() would give for the running episode, without ending it (the practice ground's periodic
    snapshot)."""
    end = server.end_state()
    return {"done_reason": server.done_reason, "sim_time_s": round(server.X.t(), 6),
            "sim_time_limit_s": server.time_limit, "end": end, "reports": list(server.reports),
            "commands": list(server.commands), "events": list(server.events),
            "log10": [[t, xy[0], xy[1], h, b] for t, xy, h, b in zip(server.log10["t"], server.log10["xy"],
                                                                      server.log10["heading"], server.log10["busy"])],
            "executor_history": list(server.X.history), "budgets": server.budgets.record(),
            "program_time_s": float(getattr(server, "program_used", 0.0))}


def check(db):
    """Raise DebriefError when the debrief's keys are not exactly the schema's (the tests use it too)."""
    kind = db.get("kind")
    if kind not in SCHEMA or set(db) != set(SCHEMA[kind]):
        raise DebriefError("debrief keys %s differ from the schema of %r" % (sorted(db), kind))
    for (k2, key), sub in NESTED.items():
        if k2 != kind:
            continue
        vals = db[key] if isinstance(db[key], list) else [db[key]]
        for v in vals:
            if not isinstance(v, dict) or set(v) != set(sub):
                raise DebriefError("debrief %s.%s entry keys %s differ from %s" % (
                    kind, key, sorted(v) if isinstance(v, dict) else type(v).__name__, sorted(sub)))
    return True
