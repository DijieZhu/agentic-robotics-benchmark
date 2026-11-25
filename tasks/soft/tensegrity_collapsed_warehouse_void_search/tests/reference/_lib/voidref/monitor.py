"""What the reference's episode did to the world, sampled at 10 Hz inside the cell (an executor observer; it reads the
truth and never changes the physics), and the inline score of the run.

Monitor (per sample): the robot's contacts with world boxes (floor and ground patches excluded), how deep the strut
axes (node to node; the struts have no collision body) and the straight tendon paths reach inside any box (predicate
R6 reads the strut axes), and a per-primitive window the controller opens and closes.

inline_score(server): the scoring rule on the cell's own record: a report of point i at time t counts when no command ran in
[t - W, t] (the executor's command history: every command's start and end) and every 10 Hz sample of that window
is within the still tolerance of the position at t; c_i = dwell * clip(1 - (d - full) / (zero - full), 0, 1) of the
last accepted report; F = 1.0 when at the end the position is within the retrieval radius of the pad and no command
runs, else 0.85; S = F * sum(w_i c_i). The window W, the still tolerance, the credit radii (full, zero) and the
retrieval radius are the cell's announced constants (voidcell/_lib/mission.py, checked against
tests/voidscore/acceptance.yaml at every graded episode); the weights w_i are the site's own search points (checked
against the same file by the scorer). tests/voidscore is the official scorer; this one is its cross-check.
"""
import math

import numpy as np

NOT_RETRIEVED_F = 0.85
EVERY_TICKS = 10


def _constants():
    """(dwell window s, still tolerance m, (full, zero) credit radii m, retrieval radius m) from the cell."""
    from voidcell._lib import mission as cell_mission
    return (float(cell_mission.DWELL_S), float(cell_mission.STILL_TOLERANCE_M),
            (float(cell_mission.CREDIT_RADIUS_M[0]), float(cell_mission.CREDIT_RADIUS_M[1])),
            float(cell_mission.RETRIEVAL_RADIUS_M))


class Monitor:
    def __init__(self, server):
        self.server = server
        self.st = server.site_truth
        self.RT = server.robot_truth
        self.samples = 0
        self.contact_samples = 0
        self.contact_box_samples = {}
        self.rod_pen = (0.0, None, None)
        self.tendon_pen = (0.0, None, None)
        self.tendon_cut_samples = 0
        self.max_contact_pen = 0.0
        self.window = None

    def attach(self):
        self.server.X.observers.append(self.observe)
        self.observe(self.server.X, force=True)

    def detach(self):
        try:
            self.server.X.observers.remove(self.observe)
        except ValueError:
            pass

    def observe(self, X, force=False):
        if not force and X.ticks % EVERY_TICKS:
            return
        self.samples += 1
        t = round(X.t(), 3)
        boxes = set()
        for nm, _mu, dist in self.RT.contacts():
            if nm == "floor" or nm.startswith("patch__") or nm.startswith("relief"):
                continue
            boxes.add(nm)
            self.max_contact_pen = max(self.max_contact_pen, -dist)
        if boxes:
            self.contact_samples += 1
            for nm in boxes:
                self.contact_box_samples[nm] = self.contact_box_samples.get(nm, 0) + 1
        P, Q = self.RT.rods()
        d, where = self.st.segment_penetration(P, Q)
        if d > self.rod_pen[0]:
            self.rod_pen = (d, where, t)
        P, Q = self.RT.tendon_segments()
        d2, where2 = self.st.segment_penetration(P, Q)
        if d2 > 0:
            self.tendon_cut_samples += 1
        if d2 > self.tendon_pen[0]:
            self.tendon_pen = (d2, where2, t)
        w = self.window
        if w is not None:
            w["boxes"] |= boxes
            w["rod_pen"] = max(w["rod_pen"], d)
            w["tendon_pen"] = max(w["tendon_pen"], d2)

    def open_window(self):
        self.window = dict(boxes=set(), rod_pen=0.0, tendon_pen=0.0)

    def close_window(self):
        w, self.window = self.window, None
        return dict(boxes=sorted(w["boxes"]), rod_pen_mm=round(1000 * w["rod_pen"], 2),
                    tendon_pen_mm=round(1000 * w["tendon_pen"], 2))

    def report(self):
        return dict(samples_10hz=self.samples, contact_samples=self.contact_samples,
                    contact_box_samples=dict(sorted(self.contact_box_samples.items())),
                    max_contact_penetration_mm=round(1000 * self.max_contact_pen, 3),
                    rod_pen_max_mm=round(1000 * self.rod_pen[0], 3), rod_pen_where=self.rod_pen[1],
                    rod_pen_t=self.rod_pen[2], tendon_pen_max_mm=round(1000 * self.tendon_pen[0], 3),
                    tendon_pen_where=self.tendon_pen[1], tendon_pen_t=self.tendon_pen[2],
                    tendon_cut_samples=self.tendon_cut_samples)


def credit(d, r):
    return float(np.clip(1.0 - (d - r[0]) / (r[1] - r[0]), 0.0, 1.0))


def _busy_in(history, current, t0, t1):
    """True when a command was executing at some instant of (t0, t1): started before t1 and ended after t0 (a command
    started at the report's own instant, i.e. after the report, does not count)."""
    for c in history:
        if c["t_start"] < t1 - 1e-9 and c["t_end"] > t0 + 1e-9:
            return True
    return current is not None and current["t_start"] < t1 - 1e-9


def inline_score(server):
    DWELL_S, STILL_M, CREDIT_R, RETRIEVE_R = _constants()
    site = server.site
    sp = {int(s["id"]): s["xy"] for s in site["search_points"]}
    WEIGHTS = {int(s["id"]): float(s["weight"]) for s in site["search_points"]}
    L = server.log10
    t = np.asarray(L["t"], float)
    xy = np.asarray(L["xy"], float)
    out = dict(points={})
    total = 0.0
    for i in (1, 2, 3):
        acc = [r for r in server.reports if r["i"] == i and r["accepted"]]
        if not acc:
            out["points"][str(i)] = dict(reported=False, c=0.0)
            continue
        r = acc[-1]
        tr, pr = float(r["t"]), r["xy"]
        sel = (t >= tr - DWELL_S - 1e-9) & (t <= tr + 1e-9)
        dev = float(np.hypot(xy[sel, 0] - pr[0], xy[sel, 1] - pr[1]).max()) if sel.any() else None
        busy = _busy_in(server.X.history, server.X.cmd, tr - DWELL_S, tr)
        ok = bool(tr - DWELL_S >= -1e-9 and sel.any() and not busy and dev <= STILL_M)
        d = math.hypot(pr[0] - sp[i][0], pr[1] - sp[i][1])
        c = (1.0 if ok else 0.0) * credit(d, CREDIT_R)
        out["points"][str(i)] = dict(reported=True, t=round(tr, 3), d_m=round(d, 4), dwell_valid=ok,
                                     dwell_busy=busy, dwell_max_dev_mm=None if dev is None else round(1000 * dev, 3),
                                     c=round(c, 6))
        total += WEIGHTS[i] * c
    x, y, _th = server.robot_truth.pose()
    pad = site["pad"]["xy"]
    dp = math.hypot(x - pad[0], y - pad[1])
    busy_end = bool(server.X.busy)
    F = 1.0 if (dp <= RETRIEVE_R and not busy_end) else NOT_RETRIEVED_F
    out.update(F=F, d_pad_m=round(dp, 4), busy_at_end=busy_end, S=round(F * total, 6),
               constants=dict(dwell_s=DWELL_S, still_m=STILL_M, credit_radius_m=list(CREDIT_R), retrieval_m=RETRIEVE_R,
                              weights={str(k): v for k, v in sorted(WEIGHTS.items())}))
    return out
