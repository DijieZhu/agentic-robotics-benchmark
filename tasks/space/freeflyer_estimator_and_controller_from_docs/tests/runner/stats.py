"""The step-time report the processor-time rule is read on (interface.step_cpu_ms): for any controller, the
distribution of the processor time per answered step (the percentiles the rule names, and the largest), the same for
the wall time, the late steps and the skipped step calls per session, the share of steps answered inside the budget
(interface.on_time_share_gate: a skipped call counts as not on time, rule step.on_time_share) and the longest run of
steps in control without an applied command."""
import numpy as np


class StepStats(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.cpu = []
        self.wall = []
        self.t = []
        self.late = 0
        self.skipped = 0

    def answered(self, t, cpu_s, wall_s, late):
        self.t.append(float(t))
        self.cpu.append(float(cpu_s))
        self.wall.append(float(wall_s))
        if late:
            self.late += 1

    def skip(self):
        self.skipped += 1

    def report(self, watchdog=None):
        pc = [float(x) for x in self.cfg.d("step_time_percentiles")]
        out = {"steps_answered": len(self.cpu), "step_calls_skipped": self.skipped, "late_steps": self.late,
               "step_budget_ms": self.cfg.step_cpu_s * 1000.0}
        n_all = len(self.cpu) + self.skipped
        out["on_time_share"] = (len(self.cpu) - self.late) / float(n_all) if n_all else None
        out["on_time_gate"] = float(self.cfg.v("interface.on_time_share_gate"))
        for name, arr in (("cpu_ms", self.cpu), ("wall_ms", self.wall)):
            if arr:
                a = np.asarray(arr) * 1000.0
                out[name] = {"p%g" % p: float(np.percentile(a, p)) for p in pc}
                out[name]["largest"] = float(a.max())
                out[name]["mean"] = float(a.mean())
                out[name]["t_of_largest_s"] = float(self.t[int(np.argmax(a))])
            else:
                out[name] = None
        if watchdog is not None:
            out["longest_run_without_applied_command"] = {"steps": int(watchdog.longest_steps),
                                                          "s": float(watchdog.longest_s),
                                                          "watchdog_s": self.cfg.watchdog_s,
                                                          "watchdog_trips_seen_by_the_runner": list(watchdog.trips)}
        return out
