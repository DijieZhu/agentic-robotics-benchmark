"""The reference's log: every phase with its simulated and wall times, the checks it made, every event the server
returned. One JSON object per line on stderr, each prefixed "HREF " (the graded run keeps the program's stderr; the
cell's tests read it back), and the same records in memory (PhaseLog.records)."""
import contextlib
import json
import sys
import time

from .errors import ReferenceFailure

PREFIX = "HREF "


def _clean(x):
    import numpy as np
    if isinstance(x, dict):
        return {str(k): _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if isinstance(x, np.ndarray):
        return [_clean(v) for v in x.tolist()]
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, float):
        return round(x, 9)
    return x


class PhaseLog(object):
    def __init__(self, sim_time=None, sink=None):
        self.sim_time = sim_time or (lambda: None)        # the session's latest simulated time
        self.sink = sink if sink is not None else sys.stderr
        self.records = []
        self.stack = []
        self.wall0 = time.time()

    def write(self, kind, **data):
        rec = {"kind": kind, "sim_t": self.sim_time(), "wall_t": round(time.time() - self.wall0, 4)}
        if self.stack:
            rec["phase"] = self.stack[-1]
        rec.update(_clean(data))
        self.records.append(rec)
        try:
            self.sink.write(PREFIX + json.dumps(rec, sort_keys=True) + "\n")
            self.sink.flush()
        except Exception:  # noqa: BLE001 - a log never stops the program
            pass
        return rec

    @contextlib.contextmanager
    def phase(self, name, **data):
        """a named phase: 'phase_start', then 'phase_end' with its simulated and wall durations, or 'phase_failed' with
        the reason (the failure goes on up: the program stops and reports the phase)"""
        s0, w0 = self.sim_time(), time.time()
        self.write("phase_start", name=name, **data)
        self.stack.append(name)
        try:
            yield self
        except ReferenceFailure as e:
            self.stack.pop()
            self.write("phase_failed", name=name, reason=e.reason, data=e.data,
                       sim_s=_dt(s0, self.sim_time()), wall_s=round(time.time() - w0, 4))
            raise
        except Exception as e:  # noqa: BLE001
            self.stack.pop()
            self.write("phase_failed", name=name, reason="%s: %s" % (type(e).__name__, str(e)[:300]),
                       sim_s=_dt(s0, self.sim_time()), wall_s=round(time.time() - w0, 4))
            raise
        else:
            self.stack.pop()
            self.write("phase_end", name=name, sim_s=_dt(s0, self.sim_time()), wall_s=round(time.time() - w0, 4))

    def check(self, what, ok, **data):
        """a named check; a failed check is a ReferenceFailure of the current phase"""
        self.write("check", what=what, ok=bool(ok), **data)
        if not ok:
            raise ReferenceFailure(self.stack[-1] if self.stack else "-", "check failed: %s" % what, data)

    def phases(self):
        """[(name, sim_start, sim_end, wall_s, ok)] of the finished phases, in order"""
        out, open_ = [], {}
        for r in self.records:
            if r["kind"] == "phase_start":
                open_[r["name"]] = r
            elif r["kind"] in ("phase_end", "phase_failed"):
                s = open_.pop(r["name"], None)
                out.append({"name": r["name"], "sim_start": s["sim_t"] if s else None, "sim_end": r["sim_t"],
                            "sim_s": r.get("sim_s"), "wall_s": r.get("wall_s"), "ok": r["kind"] == "phase_end"})
        return out


def _dt(a, b):
    if a is None or b is None:
        return None
    return round(float(b) - float(a), 6)


def parse(text):
    """the records of a program's stderr text (lines starting with PREFIX)"""
    out = []
    for ln in text.splitlines():
        if ln.startswith(PREFIX):
            try:
                out.append(json.loads(ln[len(PREFIX):]))
            except ValueError:
                pass
    return out
