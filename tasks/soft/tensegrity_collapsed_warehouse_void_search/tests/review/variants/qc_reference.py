"""Maintainer variants of the privileged reference, for the grader's own QC (QC Q2 / Q3).

NOT run by verify.py and never given to anyone: the QC runner (qc_verify.py, the maintainers' tool) copies this
file over reference/reference.py of a SCRATCH copy of tests/ (the stage's `reference_dir`, the maintainers' switch of
VoidCellServer.reference_run) and writes the chosen plan next to it as qc_variant.json. Everything else of the stage
is the task's own bytes: verify.py, the cell, the scoring, the frozen hidden set, the oracle's submission and token.

    run(server) -> {"summary": {... the reference's summary ..., "qc_variant": {...}}, "rows": [...], "events": [...]}

The variant drives the robot with the real reference program (tests/reference/_lib/voidref, unchanged) through a
proxy of its station that changes only what the plan names:

  dwell_steps      {point: [dt, ...]}   the steps of the dwell before that point's report (the reference's own is
                                        10 + 10 + 1 s = 21 s); set between points (controller.DWELL_STEPS)
  no_return        true                 the leg 3 -> pad is not driven (controller.LEGS without its last leg)
  move_before_report [points]           after the dwell: one `ccw` with the true bottom triple, wait until idle,
                                        step 5 s, then report (a command and a motion inside the dwell window)
  early_reports    {point: n}           n reports of that point at the first primitive of the leg towards it
                                        (the robot is still at the previous goal, >= 1.0 m away)
  late_report      [points]             one more report of that point at the first primitive of the next leg (the
                                        per-point budget is used up: the station must refuse it)
  shift_point      {point: d}           right after the report is accepted, the search point is moved in the cell's
                                        own copy of the site so that the report position is exactly d metres from
                                        it (a search point is not a physical object: the world, the robot and the
                                        trajectory stay the same; only the distance the scoring measures changes)
  final_wrong_bottom true               before finish(): one primitive with a WRONG (but rollable-listed) bottom
                                        triple, run until idle (the cell must execute the wrong mapping without
                                        failing), then one refused call with a triple that is not a face
  shift_pad        d                    before finish(): the pad centre is moved in the cell's copy of the site so
                                        that the end position is exactly d metres from it

The summary's "qc_variant" records every injected call, its reply, the moved coordinates and the state hashes the
QC compares (the moved coordinates are the scoring's inputs).
"""

import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
LIB = os.path.join(HERE, "_lib")
PLAN_FILE = os.path.join(HERE, "qc_variant.json")
INVALID_TRIPLE = [0, 1, 2]


def _pts(server):
    return {int(p["id"]): p for p in server.site["search_points"]}


class Proxy(object):
    """The station of the reference with the plan's changes; every other call goes to the real station."""

    def __init__(self, server, station, plan, C):
        self._server, self._st, self._plan, self._C = server, station, plan, C
        self._early_for = 1 if plan.get("early_reports") else None
        self._late_for = None
        self.log = []

    def __getattr__(self, name):
        return getattr(self._st, name)

    # ---- helpers
    def _t(self):
        return round(float(self._server.X.t()), 3)

    def _pose(self):
        x, y, th = self._server.robot_truth.pose()
        return [float(x), float(y)]

    def _idle(self):
        n = 0
        while self._st.status()["busy"] and n < 3000:
            if self._st.step(0.1)["done"]:
                return False
            n += 1
        return True

    def _note(self, kind, **kw):
        kw.update(kind=kind, t=self._t())
        self.log.append(kw)

    def _extra_report(self, i, why):
        r = self._st.report_point(int(i))
        self._note("report", point=int(i), why=why, reply=r, xy=self._pose())
        return r

    # ---- the overridden calls
    def run_primitive(self, name, bottom_nodes):
        r = self._st.run_primitive(name, bottom_nodes)
        if self._late_for is not None:
            self._extra_report(self._late_for, "late (4th: beyond the budget)")
            self._late_for = None
        if self._early_for is not None:
            n = int(self._plan["early_reports"].get(str(self._early_for), 0))
            for _ in range(n):
                self._extra_report(self._early_for, "early (far from the point)")
            self._early_for = None
        return r

    def report_point(self, i):
        i = int(i)
        if i in self._plan.get("move_before_report", []):
            b = self._server.robot_truth.bottom()["nodes"]
            before = self._pose()
            r = self._st.run_primitive("ccw", list(b))
            ok = self._idle()
            self._st.step(5.0)
            self._note("move_before_report", point=i, reply=r, bottom=list(b), idle=ok, xy_before=before,
                       xy_after=self._pose())
        rep = self._st.report_point(i)
        xy = self._pose()
        self._note("report", point=i, why="reference", reply=rep, xy=xy)
        d = self._plan.get("shift_point", {}).get(str(i))
        if d is not None and rep.get("accepted"):
            p = _pts(self._server)[i]
            old = [float(p["xy"][0]), float(p["xy"][1])]
            ux, uy = old[0] - xy[0], old[1] - xy[1]
            n = math.hypot(ux, uy)
            ux, uy = (ux / n, uy / n) if n > 1e-9 else (1.0, 0.0)
            p["xy"] = [xy[0] + float(d) * ux, xy[1] + float(d) * uy]
            self._note("shift_point", point=i, d_target=float(d), old_xy=old, new_xy=list(p["xy"]), report_xy=xy,
                       d_before=n)
        nxt = self._plan.get("dwell_steps", {}).get(str(i + 1))
        self._C.DWELL_STEPS = tuple(nxt) if nxt else self._C.REFERENCE_DWELL_STEPS
        if i in self._plan.get("late_report", []):
            self._late_for = i
        if self._plan.get("early_reports") and i < 3:
            self._early_for = i + 1
        return rep

    def finish(self):
        srv = self._server
        if self._plan.get("final_wrong_bottom") and not srv.done:
            true_b = [int(v) for v in srv.robot_truth.bottom()["nodes"]]
            faces = [list(f) for f in self._C_FACES()]
            wrong = [f for f in faces if sorted(f) != sorted(true_b)][0]
            before = self._pose()
            h0 = srv.S.state_hash()
            r = self._st.run_primitive("roll_120_120", wrong)
            ok = self._idle()
            hist = srv.X.history[-1] if srv.X.history else {}
            b1 = srv.robot_truth.bottom()
            self._note("wrong_bottom", true_bottom=true_b, sent=wrong, reply=r, idle=ok, xy_before=before,
                       xy_after=self._pose(), history_bottom=hist.get("bottom"), history_kind=hist.get("kind"),
                       state_changed=srv.S.state_hash() != h0, end_bottom=b1["nodes"], rollable=b1["rollable"],
                       end_triangle=b1["end_triangle"], internal_errors=int(srv.internal_errors))
            h1 = srv.S.state_hash()
            r2 = self._st.run_primitive("roll_120_120", INVALID_TRIPLE)
            self._note("invalid_triple", sent=INVALID_TRIPLE, reply=r2, state_unchanged=srv.S.state_hash() == h1,
                       busy=bool(srv.X.busy))
        d = self._plan.get("shift_pad")
        if d is not None and not srv.done:
            xy = self._pose()
            pad = srv.site["pad"]
            old = [float(pad["xy"][0]), float(pad["xy"][1])]
            ux, uy = old[0] - xy[0], old[1] - xy[1]
            n = math.hypot(ux, uy)
            ux, uy = (ux / n, uy / n) if n > 1e-9 else (1.0, 0.0)
            pad["xy"] = [xy[0] + float(d) * ux, xy[1] + float(d) * uy]
            self._note("shift_pad", d_target=float(d), old_xy=old, new_xy=list(pad["xy"]), end_xy=xy, d_before=n,
                       busy=bool(srv.X.busy))
        return self._st.finish()

    def _C_FACES(self):
        from voidcell._lib import protocol
        return protocol.ROLLABLE_FACES


def run(server, log=None):
    if LIB not in sys.path:
        sys.path.insert(0, LIB)
    from voidcell.client import connect_handler
    from voidref import controller as C
    with open(PLAN_FILE, encoding="utf-8") as fh:
        plan = json.load(fh)
    saved = (C.DWELL_STEPS, C.LEGS)
    C.REFERENCE_DWELL_STEPS = C.DWELL_STEPS
    station = Proxy(server, connect_handler(server.handle), plan, C)
    try:
        if plan.get("no_return"):
            C.LEGS = tuple(leg for leg in C.LEGS if leg != (3, "pad"))
        first = plan.get("dwell_steps", {}).get("1")
        if first:
            C.DWELL_STEPS = tuple(first)
        res = C.Reference(server, station, log=log).run()
    finally:
        C.DWELL_STEPS, C.LEGS = saved
    res["summary"]["qc_variant"] = {"name": plan.get("name"), "plan": plan, "log": station.log}
    return res
