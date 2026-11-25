"""The privileged reference controller of tensegrity_collapsed_warehouse_void_search (token-privileged,
hidden side only).

It runs inside the cell (VoidCellServer.reference_run) and drives the robot through the station's own calls: a
voidcell.client Station on the cell's handle (connect_handler: the wire's JSON encoding, the same argument checks,
budgets, records and time limit as an agent's program). It reads the truth for planning only: the true node
positions (planar pose and the bottom triple it passes to run_primitive), the site's true occupancy (which mouth is
open, the plug, every box). It never takes a camera frame.

Legs: pad -> search point 1 -> 2 -> 3 (under the slab, through the true mouth) -> pad. After every primitive it
replans from the true pose (planner.Planner: hybrid A* with the robot's own flat-floor primitive table,
prim_table.json). At every search point it stops, lets 21 s pass (step 10 + 10 + 1) and reports; back at the pad it
calls finish(). The two crossings of R1 (1 -> 2 and 3 -> pad) go through DIFFERENT P openings (the open gap_P and the
covered tunnel_P; the other one is blocked for that leg's planning; the pairing with the shorter total geodesic
length is taken), so every episode crosses both P openings of R1 and the true mouth. T / XN / XL openings and the
blind mouth are never used.

Stuck: three primitives in a row that each move the centroid < 0.03 m AND turn it < 5 deg.

1.1.0 (generic changes, no branch on a site's family, shape or seed):
  * goal tolerances of the three search points PLAN_TOL 0.15 -> 0.10 m, ACCEPT 0.20 -> 0.15 m (the full
    credit radius is 0.20 m; the pad's 0.35 / 0.45 m are unchanged);
  * where() names the void part from the site truth (SiteTruth.void_part: a run G1 / N / K or the chamber C of a
    three-mouth site, "gallery" of a two-mouth site) after the openings; every row carries in_void (start or end in
    the void, SiteTruth.in_void) and the summary n_in_void;
  * the summary carries the blind leg going in (predicate R5 of the selection rule): in leg 2 -> 3, the primitives after the last
    primitive boundary whose true centroid y <= params.y2 + BLIND_Y_MARGIN (the last pose the search camera can see
    above the front row's south edge), up to the point-3 report: n_blind_in, with its first row index and turn count.
"""
import json
import math
import os
import time

from . import VERSION
from . import monitor as mon
from . import planner as pl

HERE = os.path.dirname(os.path.abspath(__file__))
TABLE_PATH = os.path.join(HERE, "prim_table.json")

LEGS = (("pad", 1), (1, 2), (2, 3), (3, "pad"))
PLAN_TOL = {1: 0.10, 2: 0.10, 3: 0.10, "pad": 0.35}
ACCEPT = {1: 0.15, 2: 0.15, 3: 0.15, "pad": 0.45}
BLIND_Y_MARGIN = 0.10                      # the blind leg going in starts after the last boundary with y <= y2 + this
DWELL_STEPS = (10.0, 10.0, 1.0)            # 21 s without a command before a report (the dwell window is 20 s)
STUCK_N, STUCK_D, STUCK_TH_DEG = 3, 0.03, 5.0
MAX_PRIMS = 320
RELAX_HARD = 0.03
FIRST_MAX_EXP, SECOND_MAX_EXP = 60000, 250000
STEP_AHEAD_MARGIN_S = 1.0                  # step (predicted duration - this) at once, then 0.01 s until idle


def fs(b):
    return "".join(map(str, b))


def load_table(path=TABLE_PATH):
    with open(path, encoding="utf-8") as fh:
        doc = json.load(fh)
    return doc


def envelope_radii(env):
    w_sweep, w_roll = float(env["W_SWEEP"]), float(env["W_ROLL"])
    return w_sweep / 2 + 0.04, max(w_sweep / 2, w_roll / 2 + 0.02)


def choose_r1(st, table, r_plan, r_hard, sp, pad):
    """(block for leg 1->2, block for leg 3->pad): each crossing uses a different P opening of R1."""
    p_ids = [o["id"] for o in st.openings() if o["row"] == "R1" and o["cls"] == "P"]
    a, b = p_ids
    best = None
    for first, second in ((a, b), (b, a)):
        L1 = pl.Planner(st, table, r_plan, r_hard, block_ids=(second,)).route_length(sp[1], sp[2])
        L2 = pl.Planner(st, table, r_plan, r_hard, block_ids=(first,)).route_length(sp[3], pad)
        tot = L1 + L2
        if best is None or tot < best[0]:
            best = (tot, first, second, L1, L2)
    return dict(leg12_via=best[1], leg3pad_via=best[2], leg12_block=(best[2],), leg3pad_block=(best[1],),
                geodesic_m=dict(leg12=round(best[3], 3), leg3pad=round(best[4], 3)))


def where(st, xy):
    """The opening holding xy (its id), else the part of the void under the slab (SiteTruth.void_part), else None."""
    for o in st.openings():
        if o["x0"] <= xy[0] <= o["x1"] and o["y0"] <= xy[1] <= o["y1"]:
            return o["id"]
    return st.void_part(xy)


def blind_in(rows, site):
    """The blind leg going in (R5): the rows of leg 2 -> 3 after the last one that ends with the true centroid at
    y <= y2 + BLIND_Y_MARGIN (y2 = the front row's south edge, params.y2; else the slab's south edge + 0.04 m)."""
    p = site.get("params") or {}
    y2 = float(p["y2"]) if "y2" in p else float(site["void"]["slab"]["y0"]) + 0.04
    leg = [r for r in rows if r["leg"] == "2->3"]
    last = None
    for i, r in enumerate(leg):
        if r["actual"][1] <= y2 + BLIND_Y_MARGIN:
            last = i
    blind = leg if last is None else leg[last + 1:]
    return dict(n_blind_in=len(blind), first_k=(blind[0]["k"] if blind else None),
                n_turns=sum(1 for r in blind if r["name"] in ("ccw", "cw")),
                y_ref_m=round(y2 + BLIND_Y_MARGIN, 6), start=(leg[last]["actual"] if last is not None else None))


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class Reference:
    def __init__(self, server, station, log=None, table_path=TABLE_PATH):
        self.server = server
        self.station = station
        self.log = log or (lambda s: None)
        doc = load_table(table_path)
        self.table_doc = doc
        self.table = pl.Table(doc["table"])
        self.env = doc["envelope"]
        self.r_plan, self.r_hard = envelope_radii(self.env)
        self.st = server.site_truth
        self.RT = server.robot_truth
        self.mon = mon.Monitor(server)

    # ------------------------------------------------------------------ station helpers
    def t(self):
        return float(self.server.X.t())

    def run_one(self, name, nodes, t_pred):
        """One primitive through the station; returns (accepted, error, time_up)."""
        st = self.station
        r = st.run_primitive(name, nodes)
        if not r["accepted"]:
            return False, r["error"], False
        ahead = max(0.0, float(t_pred) - STEP_AHEAD_MARGIN_S)
        while ahead >= 0.01:
            dt = min(10.0, math.floor(ahead * 100.0 + 1e-6) / 100.0)
            rep = st.step(dt)
            ahead -= dt
            if rep["done"]:
                return True, None, True
        while st.status()["busy"]:
            rep = st.step(0.01)
            if rep["done"]:
                return True, None, True
        return True, None, False

    # ------------------------------------------------------------------ the run
    def run(self):
        server, st_, RT, station = self.server, self.st, self.RT, self.station
        site = server.site
        t_limit = float(server.time_limit)
        table, r_plan, r_hard = self.table, self.r_plan, self.r_hard
        sp = {int(s["id"]): tuple(s["xy"]) for s in site["search_points"]}
        pad = tuple(site["pad"]["xy"])
        goals = dict(sp, pad=pad)
        self.mon.attach()
        t_wall0 = time.time()
        r1 = choose_r1(st_, table, r_plan, r_hard, sp, pad)
        blocks = {(1, 2): r1["leg12_block"], (3, "pad"): r1["leg3pad_block"]}
        planners = {}
        rows, legs, events = [], [], []
        hist = []
        n_prim = 0
        aborted = None
        plan_wall = 0.0
        for a, b in LEGS:
            key = (a, b)
            blk = blocks.get(key, ())
            if blk not in planners:
                planners[blk] = pl.Planner(st_, table, r_plan, r_hard, block_ids=blk)
            P = planners[blk]
            goal, tol, acc = goals[b], PLAN_TOL[b], ACCEPT[b]
            leg = dict(leg="%s->%s" % (a, b), t_start=round(self.t(), 3), n_prims=0, plans=0, plan_wall_s=0.0,
                       block=list(blk), replans_failed=0)
            banned = set()
            while True:
                x, y, th = RT.pose()
                if math.hypot(x - goal[0], y - goal[1]) <= acc:
                    break
                if self.t() >= t_limit - 1e-6 or n_prim >= MAX_PRIMS:
                    aborted = "time limit" if self.t() >= t_limit - 1e-6 else "primitive cap"
                    break
                bt = RT.bottom()
                if not bt["rollable"]:
                    events.append(dict(t=round(self.t(), 3), event="not_rollable", bottom=bt["nodes"],
                                       end_triangle=bt["end_triangle"]))
                    aborted = "not rollable (%s)" % fs(bt["nodes"])
                    break
                face = fs(bt["nodes"])
                t0 = time.time()
                res = P.plan((x, y, th), face, goal, tol, banned=banned, max_exp=FIRST_MAX_EXP)
                if not res["found"]:
                    leg["replans_failed"] += 1
                    res = P.plan((x, y, th), face, goal, tol, banned=banned, max_exp=SECOND_MAX_EXP, eps=2.5)
                if not res["found"]:
                    kr = ("relaxed",) + tuple(blk)
                    if kr not in planners:
                        planners[kr] = pl.Planner(st_, table, r_plan, r_hard - RELAX_HARD, block_ids=blk)
                    res = planners[kr].plan((x, y, th), face, goal, tol, banned=banned, max_exp=SECOND_MAX_EXP,
                                            eps=2.5)
                    events.append(dict(t=round(self.t(), 3), event="relaxed_plan", leg=leg["leg"], found=res["found"],
                                       pose=[round(x, 4), round(y, 4), round(th, 4)], face=face))
                dtw = time.time() - t0
                plan_wall += dtw
                leg["plans"] += 1
                leg["plan_wall_s"] += dtw
                if not res["found"] or not res["path"]:
                    events.append(dict(t=round(self.t(), 3), event="no_plan", leg=leg["leg"],
                                       pose=[round(x, 4), round(y, 4), round(th, 4)], face=face,
                                       expansions=res["expansions"]))
                    aborted = "no plan on leg %s" % leg["leg"]
                    break
                name, n_from, n_to = res["path"][0]
                t_pred = float(table.arr[face]["t"][table.arr[face]["names"].index(name)])
                self.mon.open_window()
                ok, err, time_up = self.run_one(name, bt["nodes"], t_pred)
                win = self.mon.close_window()
                if not ok:
                    aborted = "primitive refused: %s" % err
                    break
                x1, y1, th1 = RT.pose()
                b1 = RT.bottom()
                h = server.X.history[-1] if server.X.history else {}
                ex, ey = x1 - n_to[0], y1 - n_to[1]
                fw = (-math.sin(th), math.cos(th))
                ax = (math.cos(th), math.sin(th))
                d_act = math.hypot(x1 - x, y1 - y)
                dth_act = math.degrees(wrap(th1 - th))
                row = dict(k=n_prim, leg=leg["leg"], name=name, face=face, face_end=fs(b1["nodes"]),
                           face_pred=n_to[3], t_start=round(h.get("t_start", float("nan")), 3),
                           t=round(h.get("t_end", float("nan")) - h.get("t_start", float("nan")), 3),
                           start=[round(x, 4), round(y, 4), round(math.degrees(th), 2)],
                           pred=[round(n_to[0], 4), round(n_to[1], 4), round(math.degrees(n_to[2]), 2)],
                           actual=[round(x1, 4), round(y1, 4), round(math.degrees(th1), 2)],
                           err_along_m=round(ex * fw[0] + ey * fw[1], 4),
                           err_lateral_m=round(ex * ax[0] + ey * ax[1], 4),
                           err_dtheta_deg=round(math.degrees(wrap(th1 - n_to[2])), 2),
                           moved_m=round(d_act, 4), turned_deg=round(dth_act, 2),
                           pred_move_m=round(math.hypot(n_to[0] - x, n_to[1] - y), 4),
                           end_triangle=b1["end_triangle"], rest_reached=h.get("rest_reached"),
                           timeouts=h.get("timeouts"),
                           under_slab=bool(st_.under_slab((x, y)) or st_.under_slab((x1, y1))),
                           start_under_slab=bool(st_.under_slab((x, y))), end_under_slab=bool(st_.under_slab((x1, y1))),
                           in_void=bool(st_.in_void((x, y)) or st_.in_void((x1, y1))),
                           where_start=where(st_, (x, y)), where_end=where(st_, (x1, y1)),
                           plan_expansions=res["expansions"], plan_len=len(res["path"]), **win)
                rows.append(row)
                leg["n_prims"] += 1
                n_prim += 1
                hist.append((x1, y1, d_act, dth_act))
                self.log("%4d %-8s %-13s %s->%s moved %.3f (pred %.3f) dth %+.1f err a/l %+.3f/%+.3f t %.1f ep_t %.1f %s%s"
                         % (n_prim, leg["leg"], name, face, row["face_end"], d_act, row["pred_move_m"], dth_act,
                            row["err_along_m"], row["err_lateral_m"], row["t"], self.t(), row["where_end"] or "",
                            (" boxes " + ",".join(win["boxes"])) if win["boxes"] else ""))
                if time_up:
                    aborted = "time limit"
                    break
                if d_act < STUCK_D and row["pred_move_m"] >= 0.10:
                    kk = (int(round(x / pl.KEY_XY)), int(round(y / pl.KEY_XY)),
                          int(round((th % (2 * math.pi)) / pl.KEY_TH)) % pl.N_TH)
                    banned.add((kk, name))
                if len(hist) >= STUCK_N:
                    last = hist[-STUCK_N:]
                    if all(s[2] < STUCK_D and abs(s[3]) < STUCK_TH_DEG for s in last):
                        events.append(dict(t=round(self.t(), 3), event="stuck", leg=leg["leg"],
                                           at=[round(x1, 4), round(y1, 4)]))
                if b1["end_triangle"]:
                    events.append(dict(t=round(self.t(), 3), event="end_triangle", leg=leg["leg"],
                                       bottom=b1["nodes"]))
            leg["t_end"] = round(self.t(), 3)
            leg["plan_wall_s"] = round(leg["plan_wall_s"], 2)
            legs.append(leg)
            if aborted:
                break
            if b != "pad":
                time_up = False
                for dt in DWELL_STEPS:
                    if station.step(dt)["done"]:
                        time_up = True
                        break
                if time_up:
                    aborted = "time limit"
                    break
                rep = station.report_point(b)
                cx, cy, _ = RT.pose()
                events.append(dict(t=round(self.t(), 3), event="report", point=b, xy=[round(cx, 4), round(cy, 4)],
                                   d_m=round(math.hypot(cx - goal[0], cy - goal[1]), 4), reply=rep))
                self.log("     report point %s at t %.1f, %.3f m from the point" % (
                    b, self.t(), math.hypot(cx - goal[0], cy - goal[1])))
        if not server.done:
            station.finish()
        self.mon.detach()
        score = mon.inline_score(server)
        under = [r for r in rows if r["under_slab"]]
        in_void = [r for r in rows if r["in_void"]]
        bl = blind_in(rows, site)
        stuck = [e for e in events if e["event"] == "stuck"]
        endtri = [e for e in events if e["event"] == "end_triangle"]
        summary = dict(
            reference=dict(version="voidref %s" % VERSION, table_ysim=self.table_doc.get("ysim_version"),
                           table_sha256=self.table_doc.get("source_sha256"),
                           goal_tol_m={str(k): v for k, v in PLAN_TOL.items()},
                           accept_m={str(k): v for k, v in ACCEPT.items()}),
            sim_time_s=round(self.t(), 3), n_prims=n_prim, n_under_slab=len(under), n_in_void=len(in_void),
            n_blind_in=bl["n_blind_in"], blind_in=bl,
            n_forward=sum(1 for r in rows if r["name"].startswith("roll")),
            n_turns=sum(1 for r in rows if r["name"] in ("ccw", "cw")),
            aborted=aborted, stuck_events=len(stuck), end_triangles=len(endtri),
            relaxed_plans=sum(1 for e in events if e["event"] == "relaxed_plan"),
            replans_failed=sum(l["replans_failed"] for l in legs), plan_wall_s=round(plan_wall, 2),
            run_wall_s=round(time.time() - t_wall0, 2), r1=r1, r_plan=round(r_plan, 6), r_hard=round(r_hard, 6),
            legs=legs, inline_score=score, monitor=self.mon.report(), done_reason=server.done_reason)
        return dict(summary=summary, rows=rows, events=events)
