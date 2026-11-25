"""Maintainer variants of the privileged reference, for the grader's own QC on task version 1.1 (QC Q2 / Q3, and the
two degenerate submissions that need the true pose).

NOT run by verify.py and never given to anyone: the QC runner (qc_verify.py, one folder up) copies this file over
reference/reference.py of a SCRATCH copy of tests/ (the stage's `reference_dir`, the maintainers' switch of
VoidCellServer.reference_run) and writes the chosen plan next to it as qc_variant.json. Everything else of the stage
is the task's own bytes: verify.py, the cell, the scoring, the frozen hidden set, the oracle's submission and token.

    run(server) -> {"summary": {... the reference's summary ..., "qc_variant": {...}}, "rows": [...], "events": [...]}

Most plans drive the robot with the real reference program (tests/reference/_lib/voidref, unchanged) through a proxy
of its station that changes only what the plan names:

  dwell_steps      {point: [dt, ...]}   the steps of the dwell before that point's report (the reference's own is
                                        10 + 10 + 1 s = 21 s); set between points (controller.DWELL_STEPS)
  no_return        true                 the leg 3 -> pad is not driven (controller.LEGS without its last leg)
  skip_points      [3]                  point 3 is not visited: the legs are pad -> 1 -> 2 -> pad
  suppress_reports [points]             the reference's own report of these points is not sent to the station
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
  wall_touch       {walls, dwell_steps, max_push}
                                        in place of the reference's report of point 3 (the robot has dwelt at point
                                        3): for each wall in order ("back" = the site's back wall, "side" = the side
                                        wall of the chamber's corner) plan (true occupancy) to a pose in front of the
                                        foot of the perpendicular from point 3, push into the wall until two rolls in
                                        a row make no progress, dwell, report point 3 there (an attack on the credit
                                        rule: a pose against a wall must earn nothing on point 3)

and one plan replaces the reference's legs:

  custom "debris_press"  {dwell_steps, reports, max_push, blocked_stop}
                                        plan (true occupancy) from the pad straight to the true mouth and into the
                                        void up to the FIRST rubble box on the way in (G1 before K, the nearest face
                                        along the run), push into its face until blocked_stop rolls in a row make no
                                        progress, dwell (25 s), report the listed points there, finish in place (an
                                        attack on the credit rule and on the cell: the robot pressed against a
                                        hidden box)
  custom "heap_press"    {pre_margin}   (task version 2.0, degenerate submission D10) plan (true occupancy) from the
                                        pad to a pose before the heap on the open floor NEAREST the pad (its front,
                                        the side away from the line it leans on; else a flank), then push towards
                                        the middle of its rectangle until the time limit, never report, never
                                        finish (the episode must score 0 and the cell, its stuck reading and the
                                        scoring must show nothing abnormal while the robot leans on a heap's slope);
                                        the record carries the cell-side monitor's contact samples with every
                                        heap__ geom (as D9 carries the rubble boxes')

The summary's "qc_variant" records every injected call, its reply, the moved coordinates, the pushes (every
primitive, its predicted and actual motion, the boxes touched), the report poses with their distances to the three
points, and the state hashes the QC compares.
"""

import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
LIB = os.path.join(HERE, "_lib")
PLAN_FILE = os.path.join(HERE, "qc_variant.json")
INVALID_TRIPLE = [0, 1, 2]
PRE_MARGIN = 0.05            # a wall's pre-goal: r_hard + this from the wall
DEBRIS_PRE = 0.10            # the debris pre-goal: r_hard + this from the box face
LAT_MARGIN = 0.03            # the debris lateral line: r_hard + this from the box's wall
PRED_MIN = 0.05              # a roll predicted to gain at least this ...
GAIN_MIN = 0.02              # ... that gains less than this is blocked
BLOCKED_STOP = 2


def _pts(server):
    return {int(p["id"]): p for p in server.site["search_points"]}


def _fs(b):
    return "".join(map(str, b))


class Drive(object):
    """Planning and pushing on the truth for the plans that leave the reference's legs (the reference's planner, its
    primitive table and its way of running a primitive through the station)."""

    def __init__(self, server, station, C, log):
        from voidref import planner as pl
        from voidref import monitor as mon
        self.server, self.st, self.C, self.pl, self.log = server, station, C, pl, log
        doc = C.load_table()
        self.table = pl.Table(doc["table"])
        self.r_plan, self.r_hard = C.envelope_radii(doc["envelope"])
        self.truth = server.site_truth
        self.RT = server.robot_truth
        self.mon = mon.Monitor(server)
        self.planners = {}
        self.rows = []

    def t(self):
        return round(float(self.server.X.t()), 3)

    def over(self):
        return bool(self.server.done) or self.server.X.t() >= float(self.server.time_limit) - 1e-6

    def pose(self):
        x, y, th = self.RT.pose()
        return float(x), float(y), float(th)

    def planner(self, blk=(), relaxed=False):
        k = (tuple(blk), relaxed)
        if k not in self.planners:
            self.planners[k] = self.pl.Planner(self.truth, self.table, self.r_plan,
                                               self.r_hard - (self.C.RELAX_HARD if relaxed else 0.0), block_ids=blk)
        return self.planners[k]

    def run_one(self, name, nodes, t_pred):
        r = self.st.run_primitive(name, list(nodes))
        if not r["accepted"]:
            return False, r["error"], False
        ahead = max(0.0, float(t_pred) - self.C.STEP_AHEAD_MARGIN_S)
        while ahead >= 0.01:
            dt = min(10.0, math.floor(ahead * 100.0 + 1e-6) / 100.0)
            rep = self.st.step(dt)
            ahead -= dt
            if rep["done"]:
                return True, None, True
        while self.st.status()["busy"]:
            if self.st.step(0.01)["done"]:
                return True, None, True
        return True, None, False

    def _row(self, phase, name, face, start, end, t_pred, win, extra=None):
        b1 = self.RT.bottom()
        row = dict(k=len(self.rows), phase=phase, name=name, face=face, face_end=_fs(b1["nodes"]), t=self.t(),
                   start=[round(v, 4) for v in start], actual=[round(v, 4) for v in end],
                   moved_m=round(math.hypot(end[0] - start[0], end[1] - start[1]), 4), t_pred=round(float(t_pred), 3),
                   end_triangle=b1["end_triangle"], boxes=win["boxes"], rod_pen_mm=win["rod_pen_mm"])
        if extra:
            row.update(extra)
        self.rows.append(row)
        return row

    def goto(self, goal, tol, acc, phase, blk=(), max_prims=150):
        """The controller's leg loop on its own goal (replan after every primitive from the true pose)."""
        C = self.C
        banned = set()
        n = 0
        while True:
            x, y, th = self.pose()
            if math.hypot(x - goal[0], y - goal[1]) <= acc:
                return dict(reached=True, n=n, aborted=None)
            if self.over():
                return dict(reached=False, n=n, aborted="time limit")
            if n >= max_prims:
                return dict(reached=False, n=n, aborted="primitive cap")
            bt = self.RT.bottom()
            if not bt["rollable"]:
                return dict(reached=False, n=n, aborted="not rollable (%s)" % _fs(bt["nodes"]))
            face = _fs(bt["nodes"])
            P = self.planner(blk)
            res = P.plan((x, y, th), face, goal, tol, banned=banned, max_exp=C.FIRST_MAX_EXP)
            if not res["found"]:
                res = P.plan((x, y, th), face, goal, tol, banned=banned, max_exp=C.SECOND_MAX_EXP, eps=2.5)
            if not res["found"]:
                res = self.planner(blk, True).plan((x, y, th), face, goal, tol, banned=banned,
                                                   max_exp=C.SECOND_MAX_EXP, eps=2.5)
            if not res["found"] or not res["path"]:
                return dict(reached=False, n=n, aborted="no plan (%s)" % phase)
            name, _nf, n_to = res["path"][0]
            t_pred = float(self.table.arr[face]["t"][self.table.arr[face]["names"].index(name)])
            self.mon.open_window()
            ok, err, time_up = self.run_one(name, bt["nodes"], t_pred)
            win = self.mon.close_window()
            if not ok:
                return dict(reached=False, n=n, aborted="primitive refused: %s" % err)
            x1, y1, _ = self.pose()
            row = self._row(phase, name, face, (x, y), (x1, y1), t_pred, win,
                            dict(pred=[round(n_to[0], 4), round(n_to[1], 4)]))
            n += 1
            if time_up:
                return dict(reached=False, n=n, aborted="time limit")
            if row["moved_m"] < C.STUCK_D and math.hypot(n_to[0] - x, n_to[1] - y) >= 0.10:
                kk = (int(round(x / self.pl.KEY_XY)), int(round(y / self.pl.KEY_XY)),
                      int(round((th % (2 * math.pi)) / self.pl.KEY_TH)) % self.pl.N_TH)
                banned.add((kk, name))

    def _predict(self, x, y, th, face):
        A = self.table.arr.get(face)
        out = []
        if A is None:
            return out
        c, s = math.cos(th), math.sin(th)
        for i, nm in enumerate(A["names"]):
            dx, dy = float(A["dx"][i]), float(A["dy"][i])
            out.append((nm, x + c * dx - s * dy, y + s * dx + c * dy, th + float(A["dth"][i]), A["nf"][i],
                        float(A["t"][i])))
        return out

    def push(self, target, phase, max_push=14, blocked_stop=BLOCKED_STOP):
        """Greedy towards a point ON a solid face (unreachable): the first primitive of the best one- or two-primitive
        sequence of the table (the end nearest the target); stop after blocked_stop rolls in a row that were
        predicted to gain >= PRED_MIN m and gained < GAIN_MIN m (the robot is against the face)."""
        blocked = 0
        n = 0
        why = "max_push"
        while n < max_push:
            if self.over():
                why = "time limit"
                break
            bt = self.RT.bottom()
            if not bt["rollable"]:
                why = "not rollable (%s)" % _fs(bt["nodes"])
                break
            x, y, th = self.pose()
            face = _fs(bt["nodes"])
            d0 = math.hypot(x - target[0], y - target[1])
            best = None
            for nm, x1, y1, th1, f1, t1 in self._predict(x, y, th, face):
                c1 = math.hypot(x1 - target[0], y1 - target[1])
                cand = [(c1, 1)]
                for _nm2, x2, y2, _th2, _f2, _t2 in self._predict(x1, y1, th1, f1):
                    cand.append((math.hypot(x2 - target[0], y2 - target[1]), 2))
                cbest = min(cand)
                key = (round(cbest[0], 4), cbest[1], round(c1, 4))
                if best is None or key < best[0]:
                    best = (key, nm, x1, y1, t1)
            _key, nm, xp, yp, t_pred = best
            pred_gain = d0 - math.hypot(xp - target[0], yp - target[1])
            self.mon.open_window()
            ok, err, time_up = self.run_one(nm, bt["nodes"], t_pred)
            win = self.mon.close_window()
            if not ok:
                why = "primitive refused: %s" % err
                break
            x1, y1, _ = self.pose()
            gain = d0 - math.hypot(x1 - target[0], y1 - target[1])
            roll = nm.startswith("roll")
            if roll and pred_gain >= PRED_MIN and gain < GAIN_MIN:
                blocked += 1
            elif roll:
                blocked = 0
            self._row(phase, nm, face, (x, y), (x1, y1), t_pred, win,
                      dict(pred=[round(xp, 4), round(yp, 4)], pred_gain_m=round(pred_gain, 4), gain_m=round(gain, 4),
                           d_target_m=round(math.hypot(x1 - target[0], y1 - target[1]), 4), blocked=blocked))
            n += 1
            if time_up:
                why = "time limit"
                break
            if blocked >= blocked_stop:
                why = "against the face"
                break
        return dict(n=n, stop=why, contact=why == "against the face")

    def dwell(self, steps):
        for dt in steps:
            if self.st.step(float(dt))["done"]:
                return False
        return True

    def where(self, target=None, wall=None):
        """The robot's pose facts for the record: centroid, nodes' distance to a wall line, distance to the 3 points."""
        x, y, th = self.pose()
        sp = {i: p["xy"] for i, p in _pts(self.server).items()}
        out = dict(t=self.t(), xy=[round(x, 5), round(y, 5)], heading=round(th, 4),
                   d_points_m={str(i): round(math.hypot(x - p[0], y - p[1]), 4) for i, p in sorted(sp.items())},
                   contacts=sorted({nm for nm, _mu, _d in self.RT.contacts()
                                    if nm != "floor" and not nm.startswith("patch__") and not nm.startswith("relief")}))
        if target is not None:
            out["d_target_m"] = round(math.hypot(x - target[0], y - target[1]), 4)
        if wall is not None:
            axis, val = wall
            nodes = self.server.S.nodes()
            k = 0 if axis == "x" else 1
            c = x if axis == "x" else y
            out["centroid_to_wall_m"] = round(abs(c - val), 4)
            out["nearest_node_to_wall_m"] = round(float(min(abs(float(n_[k]) - val) for n_ in nodes)), 4)
        return out

    def window_facts(self, t_rep):
        """The last 20 s before t_rep on the cell's 10 Hz log: the largest distance from the report position, and
        whether a command ran in it (the executor's history and the current command)."""
        from voidcell._lib import mission as cell_mission
        W = float(cell_mission.DWELL_S)
        L = self.server.log10
        x, y, _ = self.pose()
        dev = [math.hypot(float(p[0]) - x, float(p[1]) - y) for tt, p in zip(L["t"], L["xy"])
               if t_rep - W - 1e-9 <= float(tt) <= t_rep + 1e-9]
        busy = any(h["t_start"] < t_rep - 1e-9 and h["t_end"] > t_rep - W + 1e-9 for h in self.server.X.history) or \
            (self.server.X.cmd is not None)
        return dict(window_s=W, n_samples=len(dev), max_dev_m=round(max(dev), 5) if dev else None, command_in_window=busy)


class Proxy(object):
    """The station of the reference with the plan's changes; every other call goes to the real station."""

    def __init__(self, server, station, plan, C):
        self._server, self._st, self._plan, self._C = server, station, plan, C
        self._early_for = 1 if plan.get("early_reports") else None
        self._late_for = None
        self.log = []
        self.drive = None

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

    def _wall_touch(self):
        """In place of the reference's report of point 3: touch the walls named by the plan, report point 3 at each."""
        srv, spec = self._server, self._plan["wall_touch"]
        if self.drive is None:
            self.drive = Drive(srv, self._st, self._C, None)
            self.drive.mon.attach()
        dr = self.drive
        lx, ly = srv.site["size_m"]
        sp3 = _pts(srv)[3]["xy"]
        corner = srv.site["stratum"]["corner"]
        rep = None
        for wall in spec.get("walls", ["back", "side"]):
            if dr.over():
                self._note("wall_touch_skipped", wall=wall, why="episode over")
                break
            off = dr.r_hard + PRE_MARGIN
            if wall == "back":
                line = ("y", float(ly))
                pre, target = (sp3[0], ly - off), (sp3[0], float(ly))
            else:
                xw = 0.0 if corner == "W" else float(lx)
                line = ("x", xw)
                pre, target = ((off if corner == "W" else lx - off), sp3[1]), (xw, sp3[1])
            g = dr.goto(pre, 0.08, 0.12, "wall_%s_approach" % wall)
            p = dr.push(target, "wall_%s_push" % wall, int(spec.get("max_push", 14)))
            at = dr.where(target, line)
            dr.dwell(spec.get("dwell_steps", [10.0, 10.0, 1.0]))
            rep = self._st.report_point(3)
            wf = dr.window_facts(float(srv.X.t()))
            self._note("wall_touch", wall=wall, wall_line=list(line), pre_goal=[round(v, 4) for v in pre],
                       target=[round(v, 4) for v in target], approach=g, push=p, at=at, window=wf, reply=rep,
                       report_xy=self._pose(),
                       d_sp3_m=round(math.hypot(self._pose()[0] - sp3[0], self._pose()[1] - sp3[1]), 5))
        return rep if rep is not None else {"accepted": False, "error": None}

    def report_point(self, i):
        i = int(i)
        if i in self._plan.get("suppress_reports", []):
            self._note("report_not_sent", point=i, xy=self._pose())
            rep = {"accepted": False, "error": None}
        elif i == 3 and self._plan.get("wall_touch"):
            rep = self._wall_touch()
        else:
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
        if self.drive is not None:
            self.drive.mon.detach()
        return self._st.finish()

    def _C_FACES(self):
        from voidcell._lib import protocol
        return protocol.ROLLABLE_FACES


def first_debris(site):
    """The first rubble box on the way in from the true mouth (G1 before K; along a run, the nearest face), its run's
    travel direction (+1 / -1 along x), the face coordinate and the lateral line beside the box's wall."""
    v = site["void"]
    runs = v["runs"]
    tm = [o for o in site["r2"]["openings"] if o["id"] == v["true_mouth"]][0]
    xM = 0.5 * (tm["x0"] + tm["x1"])
    xN = 0.5 * (runs["N"]["x0"] + runs["N"]["x1"])
    sp3 = [p for p in site["search_points"] if int(p["id"]) == 3][0]["xy"]
    order = []
    for b in v["rubble"]["boxes"]:
        r = runs[b["run"]]
        x0, x1, y0, y1 = b["aabb"][:4]
        if b["run"] == "G1":
            entry, sgn, base = xM, (1.0 if xN > xM else -1.0), 0.0
        else:
            entry, sgn, base = xN, (1.0 if sp3[0] > xN else -1.0), abs(xN - xM) + 10.0
        face = x0 if sgn > 0 else x1
        along = (face - entry) * sgn
        order.append((0 if along >= -0.01 else 1, base + along, b, sgn, face, r))
    order.sort(key=lambda z: (z[0], z[1]))
    _behind, dist, b, sgn, face, r = order[0]
    north = abs(b["aabb"][3] - r["y1"]) <= abs(b["aabb"][2] - r["y0"])
    return dict(box=b, dir=sgn, face_x=face, run=r, wall="north" if north else "south",
                box_mid_y=0.5 * (b["aabb"][2] + b["aabb"][3]), along_from_entry_m=round(dist, 4),
                n_boxes=len(order), behind_entry=bool(_behind), mouth=tm)


def run_debris_press(server, station, plan, C, log):
    """custom 'debris_press': pad -> in front of the true mouth -> in front of the first rubble box -> push -> dwell ->
    report -> finish in place."""
    dr = Drive(server, station, C, log)
    dr.mon.attach()
    qlog = []
    t_wall0 = time.time()
    site = server.site
    fd = first_debris(site)
    tm = fd["mouth"]
    lat = (fd["run"]["y1"] - (dr.r_hard + LAT_MARGIN)) if fd["wall"] == "north" else (fd["run"]["y0"] + dr.r_hard +
                                                                                       LAT_MARGIN)
    pre = (fd["face_x"] - fd["dir"] * (dr.r_hard + DEBRIS_PRE), lat)
    target = (fd["face_x"], fd["box_mid_y"])            # the middle of the box's face: the push steers into it
    front = (0.5 * (tm["x0"] + tm["x1"]), tm["y0"] - 0.30)
    qlog.append(dict(kind="debris_target", t=dr.t(), box=fd["box"]["name"], box_aabb=fd["box"]["aabb"], wall=fd["wall"],
                     dir=fd["dir"], face_x=round(fd["face_x"], 6), lateral_y=round(lat, 6),
                     along_from_entry_m=fd["along_from_entry_m"], n_boxes=fd["n_boxes"], mouth=tm["id"],
                     front=[round(v, 4) for v in front], pre_goal=[round(v, 4) for v in pre],
                     target=[round(v, 4) for v in target]))
    g1 = dr.goto(front, 0.10, 0.15, "to_mouth_front")
    g2 = dr.goto(pre, 0.08, 0.12, "to_debris")
    p = dr.push(target, "debris_push", int(plan.get("max_push", 14)), int(plan.get("blocked_stop", BLOCKED_STOP)))
    at = dr.where(target, ("x", fd["face_x"]))
    box_xy = fd["box"]["aabb"]
    qlog.append(dict(kind="pressed", t=dr.t(), approach=[g1, g2], push=p, at=at,
                     d_box_face_m=round(abs(at["xy"][0] - fd["face_x"]), 4),
                     box_centre_to_sp3_m=round(math.hypot(0.5 * (box_xy[0] + box_xy[1]) - _pts(server)[3]["xy"][0],
                                                          0.5 * (box_xy[2] + box_xy[3]) - _pts(server)[3]["xy"][1]), 4)))
    ok = dr.dwell(plan.get("dwell_steps", [10.0, 10.0, 5.0]))
    t_rep = float(server.X.t())
    wf = dr.window_facts(t_rep)
    reps = []
    for i in plan.get("reports", [1, 2, 3]):
        if server.done:
            break
        r = station.report_point(int(i))
        reps.append(dict(point=int(i), reply=r))
    at2 = dr.where(target, ("x", fd["face_x"]))
    qlog.append(dict(kind="reports", t=dr.t(), dwell_completed=ok, window=wf, reports=reps, at=at2))
    if not server.done:
        station.finish()
    dr.mon.detach()
    from voidref import monitor as mon
    from voidref import VERSION
    rows = dr.rows
    stuck = []
    for k in range(2, len(rows)):
        last = rows[k - 2:k + 1]
        if all(r["moved_m"] < C.STUCK_D for r in last):
            stuck.append(rows[k]["k"])
    summary = dict(reference=dict(version="qc variant over voidref %s" % VERSION), sim_time_s=round(server.X.t(), 3),
                   n_prims=len(rows), n_under_slab=sum(1 for r in rows if server.site_truth.under_slab(r["actual"])),
                   aborted=None if (g2["reached"] or p["n"]) else (g2["aborted"] or g1["aborted"]),
                   stuck_rows_reference_rule=stuck, run_wall_s=round(time.time() - t_wall0, 2),
                   inline_score=mon.inline_score(server), monitor=dr.mon.report(), done_reason=server.done_reason,
                   qc_variant=dict(name=plan.get("name"), plan=plan, log=qlog, rows=rows))
    return dict(summary=summary, rows=rows, events=[])


HEAP_PRE = 0.10              # the heap pre-goal: r_hard + this before the heap's rectangle (task version 2.0, D10)


def nearest_heap(site):
    """The heap on the open floor nearest the pad's centre (rectangle to point), with its front (the side away from
    the line it leans on) and its flanks as candidate approach sides (task version 2.0, D10)."""
    pad = site["pad"]["xy"]
    hs = (site.get("heaps") or {}).get("heaps") or []
    best = None
    for h in hs:
        x0, x1, y0, y1 = h["rect"]
        dx = max(x0 - pad[0], 0.0, pad[0] - x1)
        dy = max(y0 - pad[1], 0.0, pad[1] - y1)
        d = math.hypot(dx, dy)
        if best is None or d < best[0]:
            best = (d, h)
    return best


def run_heap_press(server, station, plan, C, log):
    """custom 'heap_press' (D10): pad -> before the nearest heap -> push into it until the time limit; no report, no
    finish."""
    dr = Drive(server, station, C, log)
    dr.mon.attach()
    qlog = []
    t_wall0 = time.time()
    site = server.site
    found = nearest_heap(site)
    if found is None:
        raise RuntimeError("heap_press: the site has no heap on the open floor")
    d_pad, h = found
    x0, x1, y0, y1 = [float(v) for v in h["rect"]]
    north = h["line"] == "north"
    m = dr.r_hard + float(plan.get("pre_margin", HEAP_PRE))
    cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
    pres = [("front", (cx, (y0 - m) if north else (y1 + m)))]
    flanks = [("west", (x0 - m, cy)), ("east", (x1 + m, cy))]
    pad = site["pad"]["xy"]
    flanks.sort(key=lambda f: math.hypot(f[1][0] - pad[0], f[1][1] - pad[1]))
    pres += flanks
    target = (cx, cy)
    qlog.append(dict(kind="heap_target", t=dr.t(), heap=h["name"], cls=h.get("cls"), rect=[x0, x1, y0, y1],
                     line=h["line"], d_pad_m=round(d_pad, 4), pre_goals=[[n, [round(v, 4) for v in p]] for n, p in pres],
                     target=[round(v, 4) for v in target]))
    approach = []
    side = None
    for nm, pre in pres:
        if dr.over():
            break
        g = dr.goto(pre, 0.08, 0.12, "to_heap_" + nm)
        approach.append(dict(side=nm, **g))
        if g["reached"]:
            side = nm
            break
    p = dr.push(target, "heap_push", 10 ** 7, 10 ** 9)
    idle = 0
    while not server.done and not dr.over():
        if station.step(10.0)["done"]:
            break
        idle += 1
    at = dr.where(target)
    mrep = dr.mon.report()
    heap_samples = {k: v for k, v in mrep.get("contact_box_samples", {}).items() if k.startswith("heap__")}
    qlog.append(dict(kind="pressed_to_the_limit", t=dr.t(), approach=approach, side=side, push=p, idle_steps_10s=idle,
                     at=at, heap_contact_samples=heap_samples, done=bool(server.done), done_reason=server.done_reason,
                     reports_sent=0, finish_sent=False))
    dr.mon.detach()
    from voidref import monitor as mon
    from voidref import VERSION
    rows = dr.rows
    stuck = []
    for k in range(2, len(rows)):
        last = rows[k - 2:k + 1]
        if all(r["moved_m"] < C.STUCK_D for r in last):
            stuck.append(rows[k]["k"])
    summary = dict(reference=dict(version="qc variant over voidref %s" % VERSION), sim_time_s=round(server.X.t(), 3),
                   n_prims=len(rows), n_under_slab=sum(1 for r in rows if server.site_truth.under_slab(r["actual"])),
                   aborted=None if (side is not None or p["n"]) else "no approach",
                   stuck_rows_reference_rule=stuck, run_wall_s=round(time.time() - t_wall0, 2),
                   inline_score=mon.inline_score(server), monitor=mrep, heap_contact_samples=heap_samples,
                   done_reason=server.done_reason,
                   qc_variant=dict(name=plan.get("name"), plan=plan, log=qlog, rows=rows))
    return dict(summary=summary, rows=rows, events=[])


def run(server, log=None):
    if LIB not in sys.path:
        sys.path.insert(0, LIB)
    from voidcell.client import connect_handler
    from voidref import controller as C
    with open(PLAN_FILE, encoding="utf-8") as fh:
        plan = json.load(fh)
    if plan.get("custom") == "debris_press":
        return run_debris_press(server, connect_handler(server.handle), plan, C, log)
    if plan.get("custom") == "heap_press":
        return run_heap_press(server, connect_handler(server.handle), plan, C, log)
    saved = (C.DWELL_STEPS, C.LEGS)
    C.REFERENCE_DWELL_STEPS = C.DWELL_STEPS
    station = Proxy(server, connect_handler(server.handle), plan, C)
    try:
        if 3 in plan.get("skip_points", []):
            C.LEGS = (("pad", 1), (1, 2), (2, "pad"))
        elif plan.get("no_return"):
            C.LEGS = tuple(leg for leg in C.LEGS if leg != (3, "pad"))
        first = plan.get("dwell_steps", {}).get("1")
        if first:
            C.DWELL_STEPS = tuple(first)
        res = C.Reference(server, station, log=log).run()
    finally:
        C.DWELL_STEPS, C.LEGS = saved
    qv = {"name": plan.get("name"), "plan": plan, "log": station.log}
    if station.drive is not None:
        qv["rows"] = station.drive.rows
        qv["monitor"] = station.drive.mon.report()
    res["summary"]["qc_variant"] = qv
    return res
