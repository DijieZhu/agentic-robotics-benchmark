"""Furnishing rules, part 2: the process rooms. Each rule works in the room's local frame (u along the main-door
wall, v into the room), builds believable equipment with connected piping, registers traversal elements with
their classes, equipment outlines for the drawing, and instrument slots (candidate inspection points / decoys)."""

from __future__ import annotations

import math

import numpy as np

from .furnish import Ctx, cabinet, pallet, puddle, wall_panel, SERVICE_TEXT
from .structure import rebuild_leaves
from .util import choice, quat_axis_angle, quat_mul, quat_yaw, uniform, weighted_key
from .world import sample_size

CHEEK_H = 0.12                   # bund ramp side cheeks above the ramp surface (m): a barrier for every mode


# ============================================================================================ pump room
def f_pump_room(c: Ctx):
    g = c.g
    cfg = c.cfg
    Lu, Lv = c.Lu, c.Lv
    sc = 1.2 if Lu > 7.4 else 1.1               # skid scale
    lp, dp = 1.7 * sc, 0.85 * sc            # plinth length (along the shaft, u) and depth (v)
    ba = 1.1                                # back aisle
    vp = Lv - ba - dp / 2                   # skid centre line
    n_lo, n_hi = cfg["rooms"]["pump_room"]["pumps"]
    tight = g.random() < 0.55
    gin = sample_size(cfg, g, "narrow_gap", "tight") if tight else sample_size(cfg, g, "narrow_gap", "free") + 0.1
    N = n_hi
    while N > n_lo and N * lp + (N - 1) * gin + 1.7 > Lu:
        N -= 1
    if N * lp + (N - 1) * gin + 1.0 > Lu:
        gin = sample_size(cfg, g, "narrow_gap", "tight")
        tight = True
    F = Lu - (N * lp + (N - 1) * gin)
    if F >= 1.7:
        el = round(uniform(g, 0.85, F - 0.85) / 0.05) * 0.05
    else:
        el = 0.2 if g.random() < 0.5 else F - 0.2   # one end closed by a panel against the wall
    tagbase = "P-%s" % c.room.number
    letters = "ABC"
    sv_s = choice(g, ["cw", "rw", "dw"])
    zh, rh = 0.85, 0.1
    vh = Lv - 0.22
    c.pipe("suct_hdr", [(-0.15, vh, zh), (Lu - 0.3, vh, zh)], rh, sv_s, flange_every=2.4)
    c.cyl("suct_hdr/cap", (Lu - 0.3, vh, zh), (Lu - 0.26, vh, zh), rh * 1.35, "steel_dark")
    c.pipe("disch_hdr", [(-0.15, Lv - 0.25, 2.35), (Lu + 0.15, Lv - 0.25, 2.35)], 0.09, sv_s, group=2)
    for k in range(int(Lu // 1.8) + 1):
        uu = min(Lu - 0.4, 0.4 + 1.8 * k)
        c.box("hdr_bracket%d" % k, uu - 0.03, vh - 0.06, 0.6, uu + 0.03, Lv, zh - rh, "galv")
    bearings, skids = [], []
    u = el
    for i in range(N):
        us, ue = u, u + lp
        b = "eq/%s/pump%s" % (c.rid, letters[i])
        tag = "%s%s" % (tagbase, letters[i])

        def U(x):
            return us + x * sc
        c.box("plinth", us, vp - dp / 2, 0.0, ue, vp + dp / 2, 0.15, "concrete_plinth", body=b)
        c.box("base", us + 0.05, vp - dp / 2 + 0.06, 0.15, ue - 0.05, vp + dp / 2 - 0.06, 0.21, "steel_dark", body=b)
        c.box("driptray", ue - 0.75 * sc, vp - 0.3 * sc, 0.21, ue - 0.08, vp + 0.3 * sc, 0.24, "steel_dark", group=1,
              body=b)
        zc = 0.21 + 0.24 * sc
        c.cyl("motor", (U(0.14), vp, zc), (U(0.80), vp, zc), 0.21 * sc, "motor_blue", body=b)
        for k in range(5):
            c.cyl("motor_fin%d" % k, (U(0.22 + 0.11 * k), vp, zc), (U(0.24 + 0.11 * k), vp, zc), 0.225 * sc,
                  "motor_blue_dark", group=1, body=b)
        c.cyl("motor_fan", (U(0.06), vp, zc), (U(0.14), vp, zc), 0.19 * sc, "motor_blue_dark", body=b)
        c.box("motor_tb", U(0.38), vp - 0.1 * sc, zc + 0.18 * sc, U(0.60), vp + 0.1 * sc, zc + 0.33 * sc,
              "motor_blue_dark", body=b)
        c.box("motor_feet", U(0.17), vp - 0.19 * sc, 0.21, U(0.78), vp + 0.19 * sc, zc - 0.13 * sc, "steel_dark",
              body=b)
        c.box("guard", U(0.80), vp - 0.14 * sc, 0.26, U(1.02), vp + 0.14 * sc, zc + 0.16 * sc, "guard_orange",
              body=b)
        brg = c.cyl("bearing", (U(1.02), vp, zc), (U(1.22), vp, zc), 0.115 * sc, "pump_casing", body=b)
        c.box("bearing_ped", U(1.05), vp - 0.09 * sc, 0.21, U(1.19), vp + 0.09 * sc, zc - 0.06, "pump_casing", body=b)
        zcas = zc + 0.03
        c.cyl("casing", (U(1.22), vp, zcas), (U(1.50), vp, zcas), 0.26 * sc, "pump_casing", body=b)
        c.box("casing_ped", U(1.24), vp - 0.13 * sc, 0.21, U(1.48), vp + 0.13 * sc, 0.33, "pump_casing", body=b)
        bearings.append({"geom": brg.name, "pos": c.f.w(U(1.12), vp, zc), "label": "%s DE bearing" % tag,
                         "offset": 0.115 * sc + 0.03})
        # suction: axial nozzle, up to the header level, back to the header (gate valve with a rising stem)
        us_n = min(ue - 0.08, U(1.62))
        c.pipe("suction", [(U(1.50), vp, zcas), (us_n, vp, zcas), (us_n, vp, zh), (us_n, vh, zh)], 0.07, sv_s, body=b)
        c.slot("valve_handwheel", us_n, Lv - 0.62, zh, -1 if i < N / 2 else 1, 0,
               "%s suction isolation valve" % tag, "%s suction" % SERVICE_TEXT[sv_s], b,
               pipe={"axis": c.wvec(0, 1), "r": 0.07}, stem="up")
        # discharge: up from the casing top to the high header; check valve, lever valve and gauge face the aisle
        ud = U(1.36)
        c.pipe("disch", [(ud, vp, zcas + 0.26 * sc), (ud, vp, 2.35), (ud, Lv - 0.25, 2.35)], 0.055, sv_s, body=b)
        c.cyl("check", (ud, vp, 0.98), (ud, vp, 1.12), 0.08, "steel_dark", body=b)
        c.cyl("flange_a", (ud, vp, 1.12), (ud, vp, 1.14), 0.085, "steel_dark", group=1, body=b)
        c.slot("valve_lever", ud, vp, 1.3, 0, -1, "%s discharge valve" % tag, "%s discharge" % SERVICE_TEXT[sv_s],
               b, pipe={"axis": [0, 0, 1], "r": 0.055})
        c.cyl("gstub", (ud, vp, 1.5), (ud, vp - 0.17, 1.5), 0.012, "steel", group=1, body=b)
        c.slot("gauge", ud, vp - 0.21, 1.5, 0, -1, "%s discharge pressure" % tag,
               "%s discharge" % SERVICE_TEXT[sv_s], b, case_hint=choice(g, [100, 160]))
        # a starter push-button station on a post at the skid's front corner (decor)
        c.vcyl("pbs_post", us + 0.12, vp - dp / 2 - 0.12, 0.0, 1.1, 0.025, "galv", body=b)
        c.box("pbs", us + 0.04, vp - dp / 2 - 0.2, 1.1, us + 0.2, vp - dp / 2 - 0.06, 1.32, "panel_grey", body=b)
        c.box("pbs_btn", us + 0.09, vp - dp / 2 - 0.205, 1.18, us + 0.15, vp - dp / 2 - 0.2, 1.24, "red_fire",
              group=1, body=b)
        rect = [us, vp - dp / 2 - 0.22, ue, vp + dp / 2]
        c.take(rect)
        eq = c.equip("pump_set", tag, [us, vp - dp / 2, ue, vp + dp / 2], 0.8, eid="EQ-%s" % tag)
        eq["body"] = b
        skids.append([us, vp - dp / 2, ue, vp + dp / 2])
        u = ue + gin
    for i in range(N - 1):
        u0, u1 = skids[i][2], skids[i + 1][0]
        e = c.element("narrow_gap", "tight" if tight else "free", {"width": round(u1 - u0, 4)},
                      [u0, vp - dp / 2, u1, vp + dp / 2], note="gap between pump skids")
        e.geoms = []
    if el <= 0.3:
        c.box("endpanel", 0.0, vp - 0.45, 0.0, el + 0.02, vp + 0.45, 1.8, "cabinet")
    elif F - el <= 0.3:
        c.box("endpanel", Lu - (F - el) - 0.02, vp - 0.45, 0.0, Lu, vp + 0.45, 1.8, "cabinet")
    # drain channel along the front of the skids (grating, flush)
    vch = vp - dp / 2 - 0.3
    c.box("drain", 0.1, vch - 0.25, -0.002, Lu - 0.1, vch, 0.0035, "grating", group=1)
    c.W.thermal_groups.append({"room": c.rid, "kind": "pump_bearings", "units": bearings})
    mid = c.f.w(0.5 * Lu, vp - dp / 2 - 0.05, zc)
    c.W.slot("thermal", mid, c.wyaw(0, -1), c.rid, "pump bearing housings %s A-%s" % (tagbase, letters[N - 1]),
             "pump bearings", "pumps", units=bearings, group_index=len(c.W.thermal_groups) - 1)
    # the door side: which end of the front wall has the door
    dd = [d for d in c.doors if d["wall"] == "front"]
    uc = dd[0]["c"] if dd else 0.5 * Lu
    near_left = uc < 0.5 * Lu
    # local control panel with lamps on the side wall nearest the door
    side = "left" if near_left else "right"
    s = min(max(1.0, vch - 1.0), Lv - 1.6)
    if c.free([0, s - 0.35, 0.3, s + 0.35] if side == "left" else [Lu - 0.3, s - 0.35, Lu, s + 0.35]):
        fc, face, rect = wall_panel(c, "lcp", side, s, 0.9, 1.65, 0.6, 0.22, "panel_grey")
        c.take(rect)
        c.slot("lamps", fc[0] + face[0] * 0.005, fc[1], 1.38, face[0], face[1], "local control panel %s" % tagbase,
               "pump control", "lcp")
    # far front corner: the seal-water vessel; along the front wall: a heat exchanger and the starter cabinets
    uf_end = Lu - 0.65 if near_left else 0.65
    rect = [uf_end - 0.5, 0.1, uf_end + 0.5, 1.1]
    if c.free(rect):
        vb = "eq/%s/sealpot" % c.rid
        c.vcyl("plinth", uf_end, 0.6, 0.0, 0.12, 0.45, "concrete_plinth", body=vb)
        c.vcyl("shell", uf_end, 0.6, 0.12, 1.75, 0.36, "tank_grey", body=vb)
        c.vcyl("head", uf_end, 0.6, 1.75, 1.85, 0.3, "tank_grey", body=vb)
        c.pipe("vent", [(uf_end, 0.6, 1.85), (uf_end, 0.6, 2.7)], 0.03, "dw", body=vb)
        c.pipe("supply", [(uf_end + (0.36 if not near_left else -0.36), 0.6, 1.2),
                          (uf_end + (0.6 if not near_left else -0.6), 0.6, 1.2),
                          (uf_end + (0.6 if not near_left else -0.6), 0.6, 2.7)], 0.025, "dw", body=vb)
        # a tubular level gauge on the pot, facing into the room (a second home for a level IP)
        zf = float(cfg["ips"]["level_window_m"]) / 2 + 0.06
        for z in (1.0 - zf, 1.0 + zf):
            c.cyl("lgnoz%.2f" % z, (uf_end, 0.96, z), (uf_end, 1.09, z), 0.02, "steel", body=vb)
        c.slot("level", uf_end, 1.09, 1.0, 0, 1, "seal water pot level gauge", "seal water level", vb,
               span=[1.0 - zf, 1.0 + zf], tank="SEAL WATER POT")
        c.take(rect)
        c.equip("vessel", "SEAL WATER POT", [uf_end - 0.36, 0.24, uf_end + 0.36, 0.96], 1.85,
                eid="EQ-SWP-%s" % c.room.number, shape="circle", center=(uf_end, 0.6, 0), radius=0.36)
    # monorail beam over the pumps with a chain hoist (overhead)
    c.box("monorail", -0.1, vp - 0.06, 2.42, Lu + 0.1, vp + 0.06, 2.5, "guard_orange", group=2)
    c.box("hoist", 0.45 * Lu - 0.12, vp - 0.12, 2.18, 0.45 * Lu + 0.12, vp + 0.12, 2.42, "actuator_yellow", group=2)
    c.take([0.0, vch - 0.25, Lu, vch])

    def island(k, r):
        if k == 0:
            hb = "eq/%s/hx" % c.rid
            u0_, v0_, u1_, v1_ = r
            vc_ = 0.5 * (v0_ + v1_)
            c.box("saddle1", u0_ + 0.2, v0_ + 0.07, 0.0, u0_ + 0.35, v1_ - 0.07, 0.35, "steel_dark", body=hb)
            c.box("saddle2", u1_ - 0.35, v0_ + 0.07, 0.0, u1_ - 0.2, v1_ - 0.07, 0.35, "steel_dark", body=hb)
            c.cyl("shell", (u0_ + 0.12, vc_, 0.58), (u1_, vc_, 0.58), 0.28, "pipe_cw", body=hb)
            c.cyl("chan", (u0_, vc_, 0.58), (u0_ + 0.14, vc_, 0.58), 0.31, "steel_dark", body=hb)
            c.pipe("nozz", [(u1_ - 0.4, vc_, 0.86), (u1_ - 0.4, vc_, 2.7)], 0.05, "cw", body=hb)
            c.pipe("nozz2", [(u0_ + 0.45, vc_, 0.86), (u0_ + 0.45, vc_, 2.7)], 0.05, sv_s, body=hb)
            c.take(r)
            c.equip("heat_exchanger", "COOLER E-%s" % c.room.number, r, 0.86, eid="EQ-HX-%s" % c.room.number)
        else:
            ob = "eq/%s/oil" % c.rid
            u0_, v0_, u1_, v1_ = r
            c.box("tray", u0_, v0_, 0.0, u1_, v1_, 0.25, "actuator_yellow", body=ob)
            for j, uu_ in enumerate((u0_ + 0.32, u1_ - 0.32)):
                c.vcyl("drum%d" % j, uu_, 0.5 * (v0_ + v1_), 0.25, 1.13, 0.28, "box_blue", body=ob)
            c.take(r)
    from .furnish import _islands
    _islands(c, 0.66, min(2.2, 0.38 * Lu), lambda k, r: island(0, r), margin=1.1, max_n=1)
    _islands(c, 0.7, 1.25, lambda k, r: island(1, r), margin=1.0, max_n=1)
    # floor pipe across the front aisle
    cls = c.W.elem_class(g, "floor_pipe", {"free": 0.45, "needs_high_step": 0.55})
    dia = sample_size(cfg, g, "floor_pipe", cls)
    cands = [round(x / 0.05) * 0.05 for x in np.linspace(0.25 * Lu, 0.75 * Lu, 21)]
    cands = [x for x in cands if not (dd and abs(x - dd[0]["c"]) < dd[0]["w"] / 2 + 0.5) and
             c.free([x - 0.3, 0.0, x + 0.3, vch - 0.3], keep=False)]
    uf = cands[int(g.integers(len(cands)))] if cands else 0.5 * Lu
    p = c.cyl("floorpipe", (uf, 0.0, dia / 2), (uf, vch - 0.25, dia / 2), dia / 2, "pipe_dr")
    e = c.element("floor_pipe", cls, {"diameter": dia}, [uf - dia / 2, 0.0, uf + dia / 2, vch - 0.25],
                  note="drain line across the pump room aisle")
    c.W.tag(e, [p])
    # puddle near one pump (slippery)
    pv1 = vch - 0.3
    pv0 = max(0.3, pv1 - uniform(g, 0.6, 0.9))
    order = [int(x) for x in g.permutation(N)]
    i = order[0]
    for i in order:
        pu0, pu1 = skids[i][0] + 0.2, skids[i][0] + 1.3
        if c.free([pu0, pv0, pu1, pv1], keep=False) and not (pu0 - 0.1 < uf < pu1 + 0.1):
            break
    pu0, pu1 = skids[i][0] + 0.2, skids[i][0] + 1.3
    mu = sample_size(cfg, g, "puddle", "slow")
    ps = puddle(c, "puddle", pu0, pv0, pu1, pv1, mu)
    e = c.element("puddle", "slow", {"friction": mu}, [pu0, pv0, pu1, pv1], note="seal leak under %s%s" %
                  (tagbase, letters[i]))
    c.W.tag(e, ps)
    uv = Lu - 0.5 if near_left else 0.5
    c.view("pumps", "the pump sets from the front aisle", uv, max(0.6, vch - 1.6), 0.5 * Lu, vp, 0.5)
    c.lights_grid(max(1, int(round(Lu / 3.5))), 2, "bright")


# ============================================================================================ valve gallery
def _wall_run(c: Ctx, wall, zs, rs, svs, off, name):
    """Pipes along a long wall (front: v = off, back: v = Lv - off) that rise over its doors."""
    Lu, Lv = c.Lu, c.Lv
    v = off if wall == "front" else Lv - off
    spans = sorted(c.door_spans(wall))
    for k, (z, r, sv) in enumerate(zip(zs, rs, svs)):
        vv = v + (0.0 if wall == "front" else 0.0)
        pts = [(-0.15, vv, z)]
        for a, b in spans:
            a0, b0 = a - 0.32 - 0.06 * k, b + 0.32 + 0.06 * k
            zt = 2.3 + 0.08 * k
            pts += [(a0, vv, z), (a0, vv, zt), (b0, vv, zt), (b0, vv, z)]
        pts.append((Lu + 0.15, vv, z))
        c.pipe("%s%d" % (name, k), pts, r, sv, flange_every=2.2)
    # brackets: posts every ~2 m from the floor to the top pipe, outside door spans
    u = 0.6
    k = 0
    while u < Lu - 0.3:
        if not any(a - 0.6 < u < b + 0.6 for a, b in spans):
            if wall == "front":
                c.box("%sbr%d" % (name, k), u - 0.03, 0.0, 0.0, u + 0.03, off + 0.02, max(zs) + 0.1, "galv")
            else:
                c.box("%sbr%d" % (name, k), u - 0.03, Lv - off - 0.02, 0.0, u + 0.03, Lv, max(zs) + 0.1, "galv")
            k += 1
        u += 2.0
    return v, spans


def f_valve_gallery(c: Ctx):
    g = c.g
    cfg = c.cfg
    Lu, Lv = c.Lu, c.Lv
    tagn = c.room.number
    svs_f = [choice(g, ["cw", "rw", "dw", "st"]) for _ in range(3)]
    svs_b = [choice(g, ["cw", "rw", "ia", "dw"]) for _ in range(3)]
    zs = [0.30, 0.85, 1.40]           # the lowest run is a barrier for every mode (bottom < 0.26 m)
    rs_f = [uniform(g, 0.045, 0.07), uniform(g, 0.05, 0.08), uniform(g, 0.04, 0.06)]
    rs_b = [uniform(g, 0.045, 0.07), uniform(g, 0.05, 0.08), uniform(g, 0.04, 0.06)]
    off = 0.22
    vf, spf = _wall_run(c, "front", zs, rs_f, svs_f, off, "runf")
    vb, spb = _wall_run(c, "back", zs, rs_b, svs_b, off, "runb")
    face_f = off + 0.08                     # aisle edge of the wall runs
    # the rack: off-centre, a narrow aisle on one side
    rack_d = 0.9
    nar = uniform(g, 0.85, 1.0)
    front_narrow = g.random() < 0.5
    if front_narrow:
        vr0 = face_f + nar
    else:
        vr0 = Lv - face_f - nar - rack_d
    vr1 = vr0 + rack_d
    ur0, ur1 = 1.5, Lu - 1.5
    body = "eq/%s/rack" % c.rid
    nposts = max(2, int(round((ur1 - ur0) / 2.0)) + 1)
    ups = [ur0 + i * (ur1 - ur0) / (nposts - 1) for i in range(nposts)]
    for i, uu in enumerate(ups):
        for j, vv in enumerate((vr0 + 0.04, vr1 - 0.04)):
            c.box("post%d_%d" % (i, j), uu - 0.04, vv - 0.04, 0.0, uu + 0.04, vv + 0.04, 1.75, "galv", body=body)
        for k, z in enumerate((0.16, 0.75, 1.2, 1.65)):
            c.box("beam%d_%d" % (i, k), uu - 0.035, vr0, z - 0.06, uu + 0.035, vr1, z, "galv", body=body)
    rack_pipes = []
    for side, vv in (("a", vr0 + 0.2), ("b", vr1 - 0.2)):
        for k, z in enumerate((0.16, 0.75, 1.2)):
            r = uniform(g, 0.04, 0.075)
            sv = choice(g, ["cw", "rw", "dw", "st", "ia"])
            # each line rises out of the floor inside the rack footprint at one end and drops back at the other
            ua_, ub_ = ur0 - 0.05 - 0.07 * k, ur1 + 0.05 + 0.07 * k
            c.pipe("rack%s%d" % (side, k), [(ua_, vv, -0.05), (ua_, vv, z + r), (ub_, vv, z + r), (ub_, vv, -0.05)],
                   r, sv, body=body, flange_every=2.0)
            rack_pipes.append((side, vv, z + r, r, sv))
    c.take([ur0 - 0.4, vr0, ur1 + 0.4, vr1])
    c.equip("pipe_rack", "VALVE RACK %s" % tagn, [ur0, vr0, ur1, vr1], 1.75, eid="EQ-RACK-%s" % tagn)
    # slots along the wall runs (gate valves with rising stems, lever valves, gauges), skipping the doors
    k = 0
    for wall, v, spans, rs, svs in (("front", vf, spf, rs_f, svs_f), ("back", vb, spb, rs_b, svs_b)):
        dv = 1 if wall == "front" else -1
        u = 0.9
        while u < Lu - 0.6:
            if any(a - 0.7 < u < b + 0.7 for a, b in spans):
                u += 0.4
                continue
            kind = ("valve_handwheel", "gauge", "valve_lever")[k % 3]
            tag = "%s-%s%02d" % (tagn, wall[0].upper(), k)
            if kind == "valve_handwheel":
                c.slot("valve_handwheel", u, v, zs[1], 0, dv, "gallery %s wall, %s line" % (wall, SERVICE_TEXT[svs[1]]),
                       SERVICE_TEXT[svs[1]], "gallery", pipe={"axis": c.wvec(1, 0), "r": rs[1]}, stem="up")
            elif kind == "gauge":
                zg = choice(g, [0.6, 1.15])
                zp = zs[0] if zg < 1.0 else zs[1]
                rp = rs[0] if zg < 1.0 else rs[1]
                c.cyl("g%s%d" % (wall, k), (u, v, zp + rp), (u, v, zg), 0.012, "steel", group=1)
                c.cyl("g%s%db" % (wall, k), (u, v, zg), (u, v + dv * 0.05, zg), 0.012, "steel", group=1)
                c.slot("gauge", u, v + dv * 0.09, zg, 0, dv, "gallery %s wall, %s line" %
                       (wall, SERVICE_TEXT[svs[0 if zg < 1.0 else 2]]), SERVICE_TEXT[svs[0 if zg < 1.0 else 2]],
                       "gallery", case_hint=choice(g, [63, 100, 100, 160]))
            else:
                # lever valves on the middle run (0.85 m): readable from a narrow aisle with the wide camera
                c.slot("valve_lever", u, v, zs[1], 0, dv, "gallery %s wall, %s line" % (wall, SERVICE_TEXT[svs[1]]),
                       SERVICE_TEXT[svs[1]], "gallery", pipe={"axis": c.wvec(1, 0), "r": rs[1]})
            k += 1
            u += uniform(g, 1.1, 1.6)
    # slots on the rack faces: only on the top tier (nothing above them); gauges on short stubs toward the aisle
    for side, vv, z, r, sv in rack_pipes:
        if z < 1.0:
            continue
        dv = -1 if side == "a" else 1
        u = ur0 + 0.5
        while u < ur1 - 0.3:
            if g.random() < 0.6:
                kind = choice(g, ["valve_handwheel", "valve_lever", "gauge"])
                if kind == "gauge":
                    c.cyl("rg%s%.1f" % (side, u), (u, vv, z), (u, vv + dv * 0.2, z), 0.012, "steel", group=1)
                    c.slot("gauge", u, vv + dv * 0.24, z, 0, dv, "valve rack side %s, %s" % (side.upper(),
                           SERVICE_TEXT[sv]), SERVICE_TEXT[sv], "rack", case_hint=choice(g, [100, 160]))
                else:
                    c.slot(kind, u, vv, z, 0, dv, "valve rack side %s, %s line" % (side.upper(), SERVICE_TEXT[sv]),
                           SERVICE_TEXT[sv], "rack", pipe={"axis": c.wvec(1, 0), "r": r}, stem="up")
            u += uniform(g, 0.9, 1.4)
    # the narrow aisle gets a tight point: a gear-operated valve actuator protruding from the rack
    gap = sample_size(cfg, g, "narrow_gap", "tight")
    ua = round(uniform(g, ur0 + 1.0, ur1 - 1.0) / 0.05) * 0.05
    if front_narrow:
        a0, a1 = face_f + gap, vr0
        lr = [ua - 0.25, face_f, ua + 0.25, vr0]
    else:
        a0, a1 = vr1, Lv - face_f - gap
        lr = [ua - 0.25, vr1, ua + 0.25, Lv - face_f]
    c.box("actuator", ua - 0.22, a0, 0.25, ua + 0.22, a1, 1.15, "actuator_yellow", body=body)
    c.box("actuator/mot", ua - 0.12, a0 + 0.05, 1.15, ua + 0.12, a1 - 0.05, 1.45, "motor_blue", body=body)
    e = c.element("narrow_gap", "tight", {"width": gap}, lr, note="valve actuator in the narrow aisle")
    e.geoms = []
    # the wide aisle: a floor pipe or a low cross-over
    if front_narrow:
        wa = (vr1, Lv - face_f)
    else:
        wa = (face_f, vr0)
    uw = round(uniform(g, ur0 + 0.8, ur1 - 0.8) / 0.05) * 0.05
    if abs(uw - ua) < 1.0:
        uw = ua + 1.5 if ua + 1.5 < ur1 - 0.5 else ua - 1.5
    if g.random() < 0.5:
        cls = c.W.elem_class(g, "floor_pipe", {"free": 0.4, "needs_high_step": 0.6})
        dia = sample_size(cfg, g, "floor_pipe", cls)
        p = c.cyl("xpipe", (uw, wa[0] - 0.05, dia / 2), (uw, wa[1] + 0.05, dia / 2), dia / 2, "pipe_dr")
        e = c.element("floor_pipe", cls, {"diameter": dia}, [uw - dia / 2, wa[0], uw + dia / 2, wa[1]],
                      note="drain spool across the wide aisle")
        c.W.tag(e, [p])
    else:
        clr = sample_size(cfg, g, "low_overhead", "needs_crouch")
        r = 0.07
        p = c.cyl("xover", (uw, wa[0] - 0.1, clr + r), (uw, wa[1] + 0.1, clr + r), r, "pipe_cw")
        p2 = c.cyl("xover2", (uw + 0.17, wa[0] - 0.1, clr + r), (uw + 0.17, wa[1] + 0.1, clr + r), r, "pipe_rw")
        c.cyl("xover_up", (uw, wa[0] - 0.1, clr + r), (uw, wa[0] - 0.1, 1.2), r, "pipe_cw")
        e = c.element("low_overhead", "needs_crouch", {"clearance": clr}, [uw - 0.1, wa[0], uw + 0.27, wa[1]],
                      note="low cross-over across the wide aisle")
        c.W.tag(e, [p, p2])
        c.box("xover_stripe", uw - 0.09, wa[0] + 0.02, clr - 0.01, uw + 0.26, wa[0] + 0.12, clr + 0.0, "hazard_stripe",
              group=1)
    wv = 0.5 * (wa[0] + wa[1])
    c.view("gallery", "along the valve gallery's wide aisle", 0.45, wv, Lu, wv, 0.9)
    nv = 0.5 * (face_f + vr0) if front_narrow else 0.5 * (vr1 + Lv - face_f)
    c.view("tight", "the tight point in the narrow aisle", max(0.4, ua - 1.4), nv, ua, nv, 0.5)
    c.lights_grid(max(2, int(round(Lu / 3.5))), 1, "mid")


# ============================================================================================ tank room
def f_tank_room(c: Ctx):
    g = c.g
    cfg = c.cfg
    Lu, Lv = c.Lu, c.Lv
    tagn = c.room.number
    kt = 0.2
    # the front kerb; the room walls close the bund on the other sides. The front walkway is deep enough for a
    # 2.5 m frontal reading position on tank A's level gauge (bricks observability: level >= 2.5 m, <= 15 deg).
    bv0 = max(1.7, min(2.4, Lv - kt - 1.5 - 2.0 - 0.1))
    if c.W.ladder:
        bv0 = max(1.7, min(1.9, bv0))      # a deeper bund: reading places behind the tanks (slot B-kerb)
    bu0, bu1 = 0.0, Lu
    bv1 = Lv
    tg = uniform(g, 1.3, 1.5)               # the gap between the tanks: tank B's gauge is read in it (wide camera)
    tc = 0.75
    D = min(2.0, (Lu - tg - 2 * tc) / 2)
    ID = (bv1 - bv0) - kt
    D = max(1.5, min(D, ID - 2 * tc))
    ladder = bool(c.W.ladder)
    if ladder:
        D = min(D, 1.7)
    r = D / 2
    vc = bv0 + kt + max(tc, 0.5 * (ID - D) - 0.1) + r
    if ladder:
        # a 1.25 m aisle inside the kerb (the ramp lands there, up to 0.9 m long) and the rest behind the tanks: reading
        # places inside the bund (slot B-kerb)
        vc = bv0 + kt + 1.25 + r
    vc = min(vc, Lv - tc - r)
    ucs = [0.5 * Lu - (r + tg / 2), 0.5 * Lu + (r + tg / 2)]
    # bund class and the ramp
    bcls = c.W.elem_class(g, "bund", {"needs_high_step": 0.5, "barrier": 0.5})
    v2 = (cfg.get("ladder") or {}).get("v2")
    if v2 and v2.get("bund"):
        # version 2: the kerb's class by candidate number, low (needs_high_step) or high (barrier); the
        # draw above is kept so that every later draw of this room's stream is the same for both classes
        bcls = {"low": "needs_high_step", "high": "barrier"}[str(v2["bund"])]
    h = sample_size(cfg, g, "bund", bcls)
    rcls = weighted_key(g, {"free": 0.4, "slow": 0.6})
    slope = sample_size(cfg, g, "ramp", rcls)
    Lr = h / math.tan(math.radians(slope))
    # the ramp must fit the walkway with a 1 m flat approach in front of its foot and land inside the
    # bund's front aisle: a gentle ramp over a tall kerb would reach the front wall, and with its side cheeks it could
    # then not be entered at all. Steepen it (into the slow band if needed), or lower the kerb.
    from .world import band_interval
    lmax = max(0.35, min(bv0 - 1.0, (1.15 if c.W.ladder else 0.9)))
    if Lr > lmax:
        need = math.degrees(math.atan(h / lmax))
        f_lo, f_hi = band_interval(cfg, "slope_deg", "free")
        s_lo, s_hi = band_interval(cfg, "slope_deg", "slow")
        if need <= f_hi:
            rcls, slope = "free", round(max(need, f_lo), 4)
        elif need <= s_hi:
            rcls, slope = "slow", round(max(need, s_lo), 4)
        else:
            rcls, slope = "slow", round(s_hi, 4)
            h = round(lmax * math.tan(math.radians(slope)), 4)
        Lr = h / math.tan(math.radians(slope))
    # ramp location on the front kerb, away from the door
    dd = [d for d in c.doors if d["wall"] == "front"]
    udoor = dd[0]["c"] if dd else 0.5 * Lu
    rw = 0.9
    ur = Lu - rw / 2 - 0.6 if udoor < 0.5 * Lu else rw / 2 + 0.6
    body = "eq/%s/bund" % c.rid
    kerbs = [c.box("kerb_f1", -0.02, bv0, 0.0, ur - rw / 2, bv0 + kt, h, "hazard_stripe", body=body),
             c.box("kerb_f2", ur + rw / 2, bv0, 0.0, Lu + 0.02, bv0 + kt, h, "hazard_stripe", body=body)]
    e = c.element("bund", bcls, {"height": h, "thickness": kt, "area": [c.f.w(0, bv0)[:2], c.f.w(Lu, Lv)[:2]]},
                  [0.0, bv0, Lu, bv0 + kt], note="spill bund kerb (the walls close the bund on three sides)")
    c.W.tag(e, kerbs)
    ekerb = e
    c.box("bund_floor", 0.0, bv0 + kt, 0.0, Lu, bv1, 0.0025, "bund_floor", group=1)
    # the ramp: up, flat over the kerb, down (rotated thick boxes whose top faces form the slope)
    th = math.radians(slope)
    T = h + 0.06
    ramp_prims = []
    for sgn, v_edge in ((-1, bv0), (1, bv0 + kt)):
        # top face from (v_edge + sgn * Lr, z = 0) up to (v_edge, z = h); the box hangs below its top face
        vm = v_edge + sgn * Lr / 2
        L = Lr / math.cos(th) + 0.02
        nv, nz = sgn * math.sin(th), math.cos(th)          # top-face normal in (v, z)
        cv, cz = vm - nv * T / 2, h / 2 - nz * T / 2
        q = quat_mul(quat_axis_angle(c.wvec(1, 0), -sgn * th), quat_yaw(c.f.yaw))
        p = c.P.box("%s/ramp%d" % (body, 0 if sgn < 0 else 1), c.f.w(ur, cv, cz), (rw / 2, L / 2, T / 2), quat=q,
                    mat="ramp_steel", body=body, owner=c.rid, role="walkable", friction=0.9)
        ramp_prims.append(p)
    ramp_prims.append(c.box("ramp_top", ur - rw / 2, bv0, 0.0, ur + rw / 2, bv0 + kt, h, "ramp_steel", body=body,
                            role="walkable"))
    er = c.element("ramp", rcls, {"slope_deg": slope, "height": h, "length": Lr}, [ur - rw / 2, bv0 - Lr, ur + rw / 2,
                   bv0 + kt + Lr], note="ramp over the bund kerb")
    c.W.tag(er, ramp_prims)
    for p in ramp_prims:
        p.role = "walkable"
    # side cheeks along both lateral edges of the ramp, over its whole length (both slopes and the flat top over the
    # kerb): a robot that only holds its heading must not leave the ramp over a side (a gait review: with 50 mm
    # cheeks that stopped 3 cm short of the kerb one heading-only crossing in 100 slid off the top). 40 mm steel,
    # CHEEK_H above the ramp surface: a barrier for every mode (capability.json step barrier_min 0.10 m)
    cw_, ch_ = 0.04, CHEEK_H
    # the flat top cheek reaches CHEEK_JOIN over each slope: a slope cheek's top face ends ch_ sin(slope) short of the
    # kerb (its end face is square to the slope), which left a V-notch 1.8 cm wide down to 6 cm at both junctions
    # (sampled every 1 cm)
    ej = ch_ * math.sin(th) + 0.02
    for side_u in (ur - rw / 2 + cw_ / 2, ur + rw / 2 - cw_ / 2):
        for sgn, v_edge in ((-1, bv0), (1, bv0 + kt)):
            vm = v_edge + sgn * Lr / 2
            L = Lr / math.cos(th) + 0.02
            nv, nz = sgn * math.sin(th), math.cos(th)
            Tc = T + ch_
            cv, cz = vm - nv * (T / 2 - ch_ / 2), h / 2 - nz * (T / 2 - ch_ / 2)
            q = quat_mul(quat_axis_angle(c.wvec(1, 0), -sgn * th), quat_yaw(c.f.yaw))
            c.P.box("%s/cheek%d%d" % (body, 0 if side_u < ur else 1, 0 if sgn < 0 else 1), c.f.w(side_u, cv, cz),
                    (cw_ / 2, L / 2, Tc / 2), quat=q, mat="steel_dark", body=body, owner=c.rid, friction=0.9)
        c.box("cheek%d_top" % (0 if side_u < ur else 1), side_u - cw_ / 2, bv0 - ej, 0.0, side_u + cw_ / 2,
              bv0 + kt + ej, h + ch_, "steel_dark", body=body)

    c.take([0.0, bv0 - Lr, Lu, bv1])
    # tanks
    tags = ["T-%s%s" % (tagn, l) for l in "AB"]
    for i, uc in enumerate(ucs):
        tb = "eq/%s/tank%s" % (c.rid, "AB"[i])
        c.vcyl("ring", uc, vc, 0.0, 0.15, r + 0.06, "concrete_plinth", body=tb)
        c.vcyl("shell", uc, vc, 0.15, 2.1, r, "tank_grey", body=tb)
        c.vcyl("head", uc, vc, 2.1, 2.2, r * 0.9, "tank_grey", body=tb)
        c.vcyl("manway", uc + 0.3 * r, vc, 2.22, 2.4, 0.18, "steel", body=tb, group=2)
        c.equip("tank", tags[i], [uc - r, vc - r, uc + r, vc + r], 2.2, eid="EQ-%s" % tags[i], shape="circle",
                center=(uc, vc, 0), radius=r)
        # level gauge: tank A faces the front walkway, tank B faces the gap between the tanks
        if i == 0:
            phi = -math.pi / 2
        else:
            phi = math.pi
        du, dv = math.cos(phi), math.sin(phi)
        gu, gv = uc + (r + 0.13) * du, vc + (r + 0.13) * dv
        # a short sight-glass window (ips.level_window_m, like the bricks observability study) centred at 1.0 m
        zf = float(cfg["ips"]["level_window_m"]) / 2 + 0.06
        for z in (1.0 - zf, 1.0 + zf):
            c.cyl("lgnoz%.2f" % z, (uc + r * du, vc + r * dv, z), (gu, gv, z), 0.02, "steel", body=tb)
        c.slot("level", gu, gv, 1.0, du, dv, "%s level gauge (%s)" % (tags[i], "front" if i == 0 else "tank gap"),
               "tank level", tb, span=[1.0 - zf, 1.0 + zf], tank=tags[i])
        # outlet: out of the shell low, up to 0.9, to the side wall; a handwheel valve on the horizontal run
        side = -1 if i == 0 else 1
        uo = uc + side * (r + 0.25)
        wall_u = -0.1 if side < 0 else Lu + 0.1
        c.pipe("outlet", [(uc + side * r, vc - 0.25, 0.32), (uo, vc - 0.25, 0.32), (uo, vc - 0.25, 0.9),
                          (uo, bv0 - 0.4, 0.9), (wall_u, bv0 - 0.4, 0.9)], 0.06, "rw", body=tb, flange_every=1.6)
        c.slot("valve_handwheel", 0.5 * (uo + wall_u), bv0 - 0.4, 0.9, 0, -1, "%s outlet valve" % tags[i],
               "tank outlet", tb, pipe={"axis": c.wvec(1, 0), "r": 0.06}, stem="up")
        c.pipe("inlet", [(uc - side * 0.3 * r, vc, 2.22), (uc - side * 0.3 * r, vc, 2.45), (uc - side * 0.3 * r,
                         Lv + 0.1, 2.45)], 0.05, "rw", body=tb, group=2)
        if i == 0:
            # a caged ladder on tank A (decor)
            lu = uc - 0.55 * r
            lv = vc - math.sqrt(max(r * r - (0.55 * r) ** 2, 0)) - 0.12
            for j in range(2):
                c.box("ladder%d" % j, lu - 0.2 + 0.36 * j, lv - 0.02, 0.4, lu - 0.16 + 0.36 * j, lv + 0.02, 2.2,
                      "galv", body=tb, group=1)
    # the leaking flange on tank A's outlet and its puddle on the front walkway
    uo = ucs[0] - (r + 0.25)
    mu = sample_size(cfg, g, "puddle", "slow")
    pu0, pu1 = max(0.05, uo - 0.9), min(Lu - 0.05, uo + 0.6)
    pv0, pv1 = bv0 - Lr - 0.2 if False else max(0.25, bv0 - 1.35), bv0 - 0.12
    pv0 = max(pv0, 0.25)
    ps = puddle(c, "puddle", pu0, pv0, pu1, pv1, mu)
    e = c.element("puddle", "slow", {"friction": mu}, [pu0, pv0, pu1, pv1],
                  note="puddle from the leaking flange on %s outlet" % tags[0])
    c.W.tag(e, ps)
    # a sump in the back corner of the bund (hot spot candidate)
    su = 0.5 if g.random() < 0.5 else Lu - 0.5
    sv = bv1 - 0.45
    c.box("sump", su - 0.3, sv - 0.3, -0.002, su + 0.3, sv + 0.3, 0.004, "grating", group=1)
    c.W.sources_candidates.append({"kind": "sump", "pos": c.f.w(su, sv, 0.1), "room": c.rid,
                                   "text": "bund sump"})
    c.slot("dose_point", 0.5 * Lu, max(0.6, bv0 - Lr - 0.6), 0.0, 0, -1, "survey point on the tank room walkway",
           "survey point", "walkway")
    for i, uc in enumerate(ucs):
        c.W.sources_candidates.append({"kind": "process", "sub": "tank_contents", "pos": c.f.w(uc, vc, 1.0),
                                       "room": c.rid, "text": "%s radwaste liquor (inventory)" % tags[i],
                                       "label": tags[i], "inside_r": r})
    c.W.lay.params["bund"] = {"room": c.rid, "kerb": ekerb.id, "ramp": er.id, "tanks": [c.f.w(u, vc)[:2] for u in ucs],
                              "r": r, "v_kerb": [bv0, bv0 + kt]}
    if ladder:
        # slot B-kerb mount spots: on the tank shells (facing into the bund's back half or into the gap) and on the
        # bund's walls; the slot proof decides which of them are hidden from the walkway
        for i, uc in enumerate(ucs):
            for phi in (35.0, 60.0, 90.0, 120.0, 145.0) + ((0.0,) if i == 0 else (180.0,)):
                c.W.spots.append({"id": "SP%03d" % (len(c.W.spots) + 1), "room": c.rid, "kind": "tank", "tank": i,
                                  "center": [uc, vc], "r": r, "phi": phi, "zone": "bund",
                                  "label": tags[i], "orients": ["out"]})
        for wall, s0, s1 in (("left", bv0 + kt + 0.5, Lv - 0.4), ("right", bv0 + kt + 0.5, Lv - 0.4),
                             ("back", 0.4, Lu - 0.4)):
            c.W.spot(c, wall, s0, s1, "bund", orients=("out", "along+", "along-"), depth=0.45)
    c.view("bund", "the tanks inside the bund", 0.5 * Lu - 1.6, 0.6, 0.5 * Lu, vc, 0.8)
    c.view("ramp", "the ramp over the bund kerb", ur - 0.3, max(0.3, bv0 - Lr - 0.9), ur, bv0 + kt, 0.05)
    c.lights_grid(2, 2, "mid")


# ============================================================================================ electrical room
def f_electrical_room(c: Ctx):
    g = c.g
    cfg = c.cfg
    Lu, Lv = c.Lu, c.Lv
    tagn = c.room.number
    cw_, cd = 0.6, 0.6
    rows = []
    # row A against the back wall
    spans = c.wall_free_spans("back", 0.5)
    rowA = []
    for a, b in spans:
        n = int((b - a - 0.2) // cw_)
        u0 = a + 0.5 * ((b - a) - n * cw_)
        for i in range(n):
            rect = [u0 + i * cw_, Lv - cd, u0 + (i + 1) * cw_ - 0.01, Lv]
            if c.free(rect):
                rowA.append(rect)
    # row B free-standing, facing the front aisle
    vb1 = Lv - cd - 1.1
    vb0 = vb1 - 0.5
    rowB = []
    if vb0 >= 1.7:
        n = int((Lu - 1.9) // cw_)
        u0 = 0.5 * (Lu - n * cw_)
        for i in range(n):
            rect = [u0 + i * cw_, vb0, u0 + (i + 1) * cw_ - 0.01, vb1]
            if c.free(rect):
                rowB.append(rect)
    unitsA, unitsB = [], []
    # the cubicles that carry an indicator panel (below: rowB's middle section, rowA's second cubicle) have a
    # single-leaf door without the centre seam, their handle below the panel and its plate (since 2026-11-16: no plate or
    # instrument face over a seam; on version 1's practice buildings the seam ran flush through the panel plates)
    v2 = bool((cfg.get("ladder") or {}).get("v2"))         # version 1 keeps its cubicles as they were
    panel_a = 1 if len(rowA) > 1 and v2 else None
    panel_b = len(rowB) // 2 if rowB and v2 else None
    for k, rect in enumerate(rowA):
        p = cabinet(c, "swgA%d" % k, *rect, 2.0, mat="cabinet", face=(0, -1),
                    body="eq/%s/swgA" % c.rid, seam=(k != panel_a),
                    handle_z=(0.80, 1.05) if k == panel_a else (1.0, 1.25))
        door = c.box("swgA%d/door" % k, rect[0] + 0.04, rect[1] - 0.012, 0.25, rect[2] - 0.04, rect[1], 1.85,
                     "cabinet_door", group=1, body="eq/%s/swgA" % c.rid)
        unitsA.append({"geom": door.name, "pos": c.f.w(0.5 * (rect[0] + rect[2]), rect[1], 1.0),
                       "label": "SWG-%s cubicle %d" % (tagn, k + 1)})
    for k, rect in enumerate(rowB):
        cabinet(c, "mccB%d" % k, *rect, 1.9, mat="cabinet", face=(0, -1), body="eq/%s/mccB" % c.rid,
                seam=(k != panel_b), handle_z=(0.80, 1.05) if k == panel_b else (1.0, 1.25))
        door = c.box("mccB%d/door" % k, rect[0] + 0.04, rect[1] - 0.012, 0.25, rect[2] - 0.04, rect[1], 1.75,
                     "cabinet_door", group=1, body="eq/%s/mccB" % c.rid)
        unitsB.append({"geom": door.name, "pos": c.f.w(0.5 * (rect[0] + rect[2]), rect[1], 1.0),
                       "label": "MCC-%s section %d" % (tagn, k + 1)})
    if rowA:
        c.equip("switchgear", "SWG-%s" % tagn, [rowA[0][0], rowA[0][1], rowA[-1][2], rowA[-1][3]], 2.0,
                eid="EQ-SWG-%s" % tagn)
    if rowB:
        c.equip("mcc", "MCC-%s" % tagn, [rowB[0][0], rowB[0][1], rowB[-1][2], rowB[-1][3]], 1.9,
                eid="EQ-MCC-%s" % tagn)
    # thermal group: the row with more cubicles; lamps panels on the others
    grp = unitsA if len(unitsA) >= len(unitsB) else unitsB
    if len(grp) >= 3:
        sel = grp[:min(len(grp), 4)]
        c.W.thermal_groups.append({"room": c.rid, "kind": "cabinet_doors", "units": sel})
        xs = [u["pos"] for u in sel]
        mid = [sum(p[0] for p in xs) / len(xs), sum(p[1] for p in xs) / len(xs), 1.0]
        c.W.slot("thermal", mid, c.wyaw(0, -1), c.rid, "cubicle doors %s" % sel[0]["label"].split(" ")[0],
                 "switchgear", "cabinets", units=sel, group_index=len(c.W.thermal_groups) - 1)
    for k, rect in enumerate((rowB[len(rowB) // 2:len(rowB) // 2 + 1] if rowB else []) +
                             (rowA[1:2] if len(rowA) > 1 else [])):
        uc = 0.5 * (rect[0] + rect[2])
        c.slot("lamps", uc, rect[1] - 0.015, 1.45 if k == 0 else 1.3, 0, -1, "indicator panel on %s" %
               ("MCC" if k == 0 and rowB else "switchgear"), "electrical", "cabinets")
    # rubber mats in front of the rows, cable trays overhead, a cable protector across the front aisle
    if rowB:
        c.box("mat", rowB[0][0], vb0 - 0.9, 0.0, rowB[-1][2], vb0 - 0.05, 0.004, "rubber_black", group=1)
        c.box("tray_b", rowB[0][0], vb0 + 0.1, 2.3, rowB[-1][2], vb1 - 0.1, 2.36, "tray", group=2)
    if rowA:
        c.box("tray_a", rowA[0][0], Lv - cd + 0.1, 2.3, rowA[-1][2], Lv - 0.1, 2.36, "tray", group=2)
    if rowB:
        cls = c.W.elem_class(g, "cable_protector", {"free": 0.5, "needs_high_step": 0.5})
        h = sample_size(cfg, g, "cable_protector", cls)
        uu = rowB[0][0] - 0.5 if g.random() < 0.5 else rowB[-1][2] + 0.5
        uu = min(max(uu, 0.3), Lu - 0.3)
        from .furnish import protector_width, ramped_protector
        a_w, b_w = c.f.w(uu, 0.3), c.f.w(uu, vb1)
        along_world_y = abs(a_w[0] - b_w[0]) < 1e-6
        if along_world_y:
            ps_ = ramped_protector(c.P, "%s/cprot" % c.body, owner=c.rid, axis="y", c=a_w[0], lo=min(a_w[1], b_w[1]),
                                   hi=max(a_w[1], b_w[1]), h=h)
        else:
            ps_ = ramped_protector(c.P, "%s/cprot" % c.body, owner=c.rid, axis="x", c=a_w[1], lo=min(a_w[0], b_w[0]),
                                   hi=max(a_w[0], b_w[0]), h=h)
        hw = protector_width(h) / 2
        e = c.element("cable_protector", cls, {"height": h}, [uu - hw, 0.3, uu + hw, vb1],
                      note="temporary cable at the end of the MCC row")
        c.W.tag(e, ps_)
    if rowB:
        c.view("electrical", "the MCC row and its indicator panel", 0.5 * Lu - 1.2, 0.5, 0.5 * Lu, vb0, 1.2)
    c.lights_grid(level="dim")


# ============================================================================================ waste store
def f_waste_store(c: Ctx):
    g = c.g
    cfg = c.cfg
    Lu, Lv = c.Lu, c.Lv
    tagn = c.room.number
    dd = [d for d in c.doors if d["wall"] == "front"]
    ud = dd[0]["c"] if dd else 1.0
    wd = dd[0]["w"] if dd else 1.0
    low_end = ud < 0.5 * Lu
    # make the leaf swing toward the dead end of the vestibule
    if dd:
        d = dd[0]["d"]
        want_low_hinge = low_end
        d.hinge = -1 if want_low_hinge else 1
        # the hinge sign is in the wall's 'along' coordinate; map local u to it
        along_sign = 1
        w = c.W.lay.wall(d.wall)
        if w.axis == "y":
            along_sign = 1 if c.f.u[0] > 0 else -1
        else:
            along_sign = 1 if c.f.u[1] > 0 else -1
        d.hinge = d.hinge * along_sign
        rebuild_leaves(c.W, d)
    vm, tm = 1.15, 0.5
    if low_end:
        ub0, ub1 = 0.0, min(Lu - 1.3, ud + wd / 2 + 1.3)
    else:
        ub0, ub1 = max(1.3, ud - wd / 2 - 1.3), Lu
    p = c.box("maze_wall", ub0, vm, 0.0, ub1, vm + tm, 2.6, "shield_concrete", radio=True)
    c.W.shields.append({"id": "SH-%s-maze" % tagn, "prim": p.name, "room": c.rid, "kind": "maze wall",
                        "center": p.pos, "half": p.size, "yaw": c.f.yaw})
    c.equip("shield_wall", "MAZE SHIELD WALL", [ub0, vm, ub1, vm + tm], 2.6, eid="EQ-MAZE-%s" % tagn)
    c.take([ub0, vm, ub1, vm + tm])
    # zone tape at the maze exit and signs
    ux0, ux1 = (ub1, Lu) if low_end else (0.0, ub0)
    c.box("zone_tape", ux0, vm, 0.0, ux1, vm + 0.1, 0.003, "zone_tape", group=1)
    mat = c.W.sign("trefoil_%s" % c.rid, "RADIATION AREA", "trefoil", (256, 256))
    c.boxc("trefoil1", 0.5 * (ub0 + ub1), vm - 0.005, 1.5, 0.15, 0.005, 0.15, mat, yaw_l=-math.pi / 2, group=1)
    # drum columns from the back wall toward the cross aisle behind the maze; the hot aisles run between them
    drums = []
    aisle = uniform(g, 0.9, 1.05)
    pitch = 1.2 + aisle
    ncol = max(1, int((Lu - 0.2 + aisle) // pitch))
    u0 = 0.5 * (Lu - (ncol * 1.2 + (ncol - 1) * aisle)) + 0.6
    v_cross = vm + tm + 0.85                         # the cross aisle behind the maze wall
    # the pallets of a column touch (1.0 m pitch): the 25-75 mm slots between them were foot traps
    nrow = max(1, int((Lv - 0.05 - v_cross) // 1.0))
    k = 0
    for i in range(ncol):
        uu = u0 + i * pitch
        for j in range(nrow):
            vv = Lv - 0.55 - j * 1.0
            nd = 4 if g.random() < 0.85 else 3
            drums += [(pos, "%s-P%02d" % (tagn, k)) for pos in pallet(c, "pal%d" % k, uu, vv, nd)]
            k += 1
    c.equip("drum_store", "DRUM STORAGE AREA", [0.1, Lv - 0.05 - nrow * 1.0, Lu - 0.1, Lv], 1.05,
            eid="EQ-DRUMS-%s" % tagn)
    c.W.lay.params["waste_drums"] = [d[0] for d in drums]
    for pos, pal in drums:
        c.W.sources_candidates.append({"kind": "drum", "pos": pos, "room": c.rid, "text": "drum on pallet %s" % pal})
    c.W.lay.params["hot_aisle"] = [c.f.w(u0 + 0.6, v_cross)[:2], c.f.w(u0 + 0.6 + aisle, Lv)[:2]] if ncol > 1 else None
    # radiation signs on the drum columns' ends and a second trefoil on the back wall
    for i in range(ncol):
        uu = u0 + i * pitch
        vv = Lv - 0.05 - nrow * 1.0 - 0.02
        c.boxc("colsign%d" % i, uu, vv, 1.25, 0.14, 0.004, 0.14, mat, yaw_l=-math.pi / 2, group=1)
    # the survey point at the maze exit
    ue = 0.5 * (ux0 + ux1)
    c.slot("dose_point", ue, vm + tm + 0.45, 0.0, 0, 1, "survey point at the maze exit", "survey point", "store")
    ha = u0 + 0.6 + 0.5 * aisle if ncol > 1 else 0.5 * Lu
    c.view("waste", "a hot aisle between the drum columns", ha, vm + tm + 0.3, ha, Lv, 0.6)
    c.lights_grid(level="dim")
