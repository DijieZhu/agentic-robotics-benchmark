"""Budgets of one episode (mission()'s `limits`): camera frames, reports per search point, the simulated-time
limit; the program-time, silence and CPU caps are enforced outside the cell (cellkit.wire, cellkit.runner) and only
stated here so that mission() and the record carry one set of numbers.

Over-budget calls are refused and change nothing: a get_frame beyond the frame budget answers
budget_exhausted; a fourth report of a point answers accepted false, reports_left 0, error budget_exhausted.
"""

DEFAULTS = {
    "time_limit_s": 2400.0,          # simulated seconds (a default only: verify.py passes the frozen value of hidden/limits.json)
    "frames": 600,                   # get_frame calls per episode
    "reports_per_point": 3,
    "cable_length_mm": [100.0, 240.0],
    "program_time_s": 600.0,         # the program's own wall time (cellkit.wire program time)
    "silence_s": 180.0,              # from a reply to the next request
    "cpu_s": 2400.0,                 # prlimit --cpu of the program (cellkit.runner)
}


def merged(overrides=None):
    out = {k: (list(v) if isinstance(v, list) else v) for k, v in DEFAULTS.items()}
    for k, v in (overrides or {}).items():
        if k not in DEFAULTS:
            raise KeyError("unknown budget key %r" % (k,))
        out[k] = list(v) if isinstance(v, (list, tuple)) else v
    out["time_limit_s"] = float(out["time_limit_s"])
    out["frames"] = int(out["frames"])
    out["reports_per_point"] = int(out["reports_per_point"])
    return out


class Budgets:
    def __init__(self, limits):
        self.limits = merged(limits)
        self.frames_used = 0
        self.frames_refused = 0
        self.reports = {1: 0, 2: 0, 3: 0}
        self.reports_refused = 0

    @property
    def time_limit_s(self):
        return self.limits["time_limit_s"]

    def frames_left(self):
        return max(0, self.limits["frames"] - self.frames_used)

    def take_frame(self):
        if self.frames_used >= self.limits["frames"]:
            self.frames_refused += 1
            return False
        self.frames_used += 1
        return True

    def take_report(self, i):
        """(accepted, reports_left after this call)."""
        n = self.limits["reports_per_point"]
        if self.reports[i] >= n:
            self.reports_refused += 1
            return False, 0
        self.reports[i] += 1
        return True, n - self.reports[i]

    def record(self):
        return {"limits": dict(self.limits), "frames_used": self.frames_used, "frames_refused": self.frames_refused,
                "reports": {str(k): v for k, v in self.reports.items()}, "reports_refused": self.reports_refused}
