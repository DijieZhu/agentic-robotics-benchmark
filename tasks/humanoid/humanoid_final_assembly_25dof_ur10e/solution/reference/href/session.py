"""The reference's handle on the cell: the agent's client proxy (the seven public calls over the wire) plus the two
privileged reads of the oracle token. Every step() reply's events are logged; a "safety:..." or "episode_over" event
stops the program at once (it never hides an event)."""
from .errors import ReferenceFailure


class Session(object):
    def __init__(self, cell, log=None):
        self.c = cell
        self.log = log
        self.t = None                        # the latest simulated time the cell reported
        self.calls = {}
        self.events = []                     # (sim time, event) of every step() reply
        self.ticks = 0
        self.info = None

    def _count(self, k):
        self.calls[k] = self.calls.get(k, 0) + 1

    # ---------------------------------------------------------------- public calls
    def describe(self):
        self._count("describe")
        self.info = self.c.describe()
        return self.info

    def state(self):
        self._count("get_state")
        st = self.c.get_state()
        self.t = float(st["sim_time"])
        if st.get("done"):
            raise ReferenceFailure(self._phase(), "the episode is over", {"sim_time": self.t})
        return st

    def set_control(self, mode, stiffness=1.0):
        self._count("set_control")
        return self.c.set_control(mode, float(stiffness))

    def command(self, **kw):
        self._count("command")
        r = self.c.command(**kw)
        if not (isinstance(r, dict) and r.get("ok")):
            raise ReferenceFailure(self._phase(), "command refused: %r" % (r,), {"params": list(kw)})
        return r

    def step(self, n=1, allow=()):
        self._count("step")
        r = self.c.step(int(n))
        self.t = float(r["sim_time"])
        self.ticks += int(n)
        ev = list(r.get("events") or [])
        for e in ev:
            self.events.append((self.t, e))
            if self.log is not None:
                self.log.write("event", event=e)
        bad = [e for e in ev if (e.startswith("safety:") or e == "episode_over") and e not in allow]
        if bad or (r.get("done") and "episode_over" not in allow):
            raise ReferenceFailure(self._phase(), "the cell reported %s" % (", ".join(bad) or "done"), {"events": ev})
        return r

    def camera(self, name, depth=True):
        self._count("get_camera")
        return self.c.get_camera(name, depth=bool(depth))

    # ---------------------------------------------------------------- privileged reads (the oracle token)
    def truth_state(self):
        self._count("get_state_privileged")
        out = self.c._call("get_state_privileged")
        self.t = float(out["sim_time"])
        return out

    def truth_describe(self):
        self._count("describe_privileged")
        return self.c._call("describe_privileged")

    def _phase(self):
        if self.log is not None and self.log.stack:
            return self.log.stack[-1]
        return "-"
