"""check_site of the generator (the site rules): the station part of the
validity check and the generator's own conditions on a built site.

``check_site(site, model, data, info, feats)`` = the station part (``station.sites.check_site``: no initial
interpenetration, items flat on their inlets, at most one element-type unit and never inhibited, one item per
inlet, no pouch on an element-type unit or the blocked inlet, no pouch end over a wall box, stowage units and clutter
clear of each other, of the keep-outs and of the corridor) plus the generator's four conditions:

  1. pre-grasp reach: every grasp feature has >= 1 pre-grasp pose that the firmware's arm-deployed zone check
     accepts (``pregrasp``);
  2. pouches: 2-3 sealed columns, the handle's bar centre >= world.pouch.placement's handle rule beyond the frame's end
     bar (placement.handle_clear; checked on the nominal entry and on the bar's pose in the state), never over an end
     whose outside has a wall box (placement.end_allowed), the module rules (world.pouch.rules);
  3. stowage units (registered or moved): the unit site is the module's registered one, or in a 'moved' cell the
     blocked module's unit at one of its moved sites; the site is usable, the station data's robot-free release test passed at
     world.stowage_unit.release_test, and the dense siting check (station.siting.dense_check: mountable, clear ring,
     corridor, inlet distance) passes on the station without units, items or robot;
  4. telemetry (the station service ``telemetry_synth.check_telemetry``): every unit's 10 min mean >= n_sigma x sigma from its
     advisory limit, the blocked unit on the side its cell allows, the decoy below before its drift and above at the
     end in a wrong-advisory cell, the realised advisory log as the cell says, true levels in the handbook range.
``arrangement(site)``: the cell's fixed attributes and the draw ranges; ``register.check``: the register
rules.  Each returns a list of failure strings (empty = passes).
"""
from __future__ import annotations

import functools
import warnings

import numpy as np

from cabincell.site import placement as PL
from cabincell.site import spec as SPEC

from . import cells as CE
from . import pregrasp as PG
from . import register as RG

HANDLE_STATE_TOL_M = 0.001          # [C] settling may move a pouch by up to world.settle.pouch_disp_max_mm


# ----------------------------------------------------------------------------------------------- arrangement
def arrangement(site: dict, feats: dict, values=None) -> list:
    v = values or SPEC.values()
    out = []
    c = site["cell"]
    m, tier = c["module"], c["tier"]
    row = CE.row(CE.cell_key(m, tier), v)
    units = site["cabin_air_units"]
    rel = float(v["world.airflow"]["cau_flow_rel"])
    for mm in CE.MODULE_ORDER:
        U = units[f"cau_{mm}"]
        n = len(PL.unit_inlets(feats, f"cau_{mm}"))
        if not (1 - rel - 1e-9 <= float(U["flow_scale"]) <= 1 + rel + 1e-9):
            out.append(f"flow_scale:{mm}")
        if mm == m:
            want = "element" if tier == "S" else "pad"
            if U["build"] != want:
                out.append(f"blocked_unit_build:{U['build']}")
            if (tier == "S") != (U["fan_stop"] == "allowed"):
                out.append(f"blocked_unit_fan_stop:{U['fan_stop']}")
        elif U["build"] != "pad" or U["fan_stop"] != "allowed":
            out.append(f"unblocked_unit:{mm}")
        lo, hi = PL.r_face_range(U["build"], n, v)
        if not (lo <= float(U["r_face"]) <= hi):
            out.append(f"r_face_range:{mm}")
    if PL.unit_of(feats, site["blocker"]["inlet"]) != f"cau_{m}":
        out.append("blocker_not_in_cell_module")
    for mm, s in site["lighting"].items():
        want_off = row["light_circuit"] == "failed" and mm == m
        if (s == "off") != want_off or s not in ("general", "night", "off"):
            out.append(f"lighting:{mm}:{s}")
    band = v["world.battery_start_pct"][row["battery"]]
    if not (float(band[0]) <= float(site["robot"]["battery_pct"]) <= float(band[1])):
        out.append("battery_band")
    if not site["robot"]["present"] or int(site["robot"]["berth"]) not in [int(x) for x in v["world.berths"]]:
        out.append("robot")
    imc = v["world.airflow"]["inter_module_cfm"]
    for lk, L in site["inter_module"].items():
        if not (float(imc[0]) <= float(L["flow_cfm"]) <= float(imc[1])) or L["direction"] not in (1, -1):
            out.append(f"inter_module:{lk}")
    T = site.get("truth") or {}
    if T:
        if T.get("moved_stowage_unit") != (m if row["stowage_unit"] == "moved" else None):
            out.append("truth_moved_stowage_unit")
        if T.get("failed_light_module") != (m if row["light_circuit"] == "failed" else None):
            out.append("truth_failed_light_module")
        if bool(T.get("advisory_points_elsewhere")) != (row["advisory"] == "wrong_unit"):
            out.append("advisory_points_elsewhere")
        if row["blocked_unit_limit"] == "below" and T.get("blocked_unit_over_limit"):
            out.append("blocked_unit_over_limit")
    return out


# ----------------------------------------------------------------------------------------------- pouches
def pouch_rules(site: dict, feats: dict, model=None, data_=None, info=None, values=None) -> list:
    v = values or SPEC.values()
    out = []
    W = v["world.pouch.placement"]
    rules = v["world.pouch.rules"]
    m = site["cell"]["module"]
    per_mod, per_inlet = {}, {}
    ha, _ = PL.opening_half(v)
    for P in site.get("small_items") or []:
        if not (int(W["columns"][0]) <= int(P["columns"]) <= int(W["columns"][1])):
            out.append(f"columns:{P['id']}")
        if not PL.handle_clear(P["size"][0], int(P["columns"]), v):
            out.append(f"handle_over_end_bar:{P['id']}")
        if not PL.end_allowed(P["inlet"], int(P["end"])):
            out.append(f"end_over_wall_box:{P['id']}")
        want = PL.pouch_offset_a(P["size"][0], int(P["columns"]), int(P["end"]), v)
        if abs(float(P["offset_m"][0]) - want) > 1e-9 or float(P["offset_m"][1]) != 0.0 or float(P["theta_deg"]) != 0.0:
            out.append(f"pouch_pose_not_its_end_rule:{P['id']}")
        mm = PL.inlet(feats, P["inlet"])["module"]
        per_mod[mm] = per_mod.get(mm, 0) + 1
        per_inlet[P["inlet"]] = per_inlet.get(P["inlet"], 0) + 1
    n = len(site.get("small_items") or [])
    if n not in [int(x) for x in v["world.pouch.count"]]:
        out.append(f"pouch_count:{n}")
    if any(k > int(rules["max_per_inlet"]) for k in per_inlet.values()):
        out.append("pouches_per_inlet")
    if any(k > int(rules["max_per_module"]) for k in per_mod.values()):
        out.append("pouches_per_module")
    if sum(k for mm, k in per_mod.items() if mm != m) < int(rules["min_outside_blocked_module"]):
        out.append("pouch_outside_blocked_module")
    if model is not None:
        from firmware import bars as BR
        bars = {b.name: b for b in BR.item_bars(model, info)}
        for P in site.get("small_items") or []:
            face = PL.InletFace(PL.inlet(feats, P["inlet"]), v)
            c, _a, _n = bars[f"{P['id']}:handle"].pose(data_)
            s_a = float((np.asarray(c) - face.c) @ face.a)
            if abs(s_a) < ha + PL.FRAME_END_BAR + PL.HANDLE_CLEAR_M - HANDLE_STATE_TOL_M:
                out.append(f"handle_over_end_bar_in_state:{P['id']}:{s_a:.4f}")
    return out


# ----------------------------------------------------------------------------------------------- stowage units
@functools.lru_cache(maxsize=1)
def _bare_scene():
    """The mission station without units, items, robot or dock (for the dense siting check)."""
    from station import sites as SI
    S = SI.example_site("P")
    S.update(blocker=None, small_items=[], clutter=[], stowage_units={}, robot=dict(present=False, berth=1))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m, info = SI.build_site(S)
    d = SI.initial_data(m, info)
    return m, d


@functools.lru_cache(maxsize=None)
def dense(site_id: str) -> dict:
    from station import data as SD
    from station import siting as ST
    m, d = _bare_scene()
    return ST.dense_check(m, d, SD.features("mission"), SD.unit_site(site_id))


def stowage_rules(site: dict, values=None, dense_check: bool = True) -> list:
    from station import data as SD
    v = values or SPEC.values()
    out = []
    m = site["cell"]["module"]
    row = CE.row(CE.cell_key(m, site["cell"]["tier"]), v)
    reg = v["world.stowage_unit.registered"]
    moved = v["world.stowage_unit.moved"]
    rt = v["world.stowage_unit.release_test"]
    side = v["world.stowage_unit.pad_plate"]["selected"]
    for mm in CE.MODULE_ORDER:
        e = (site.get("stowage_units") or {}).get(mm)
        if e is None:
            out.append(f"no_unit:{mm}")
            continue
        sid = e["unit_site"]
        if row["stowage_unit"] == "moved" and mm == m:
            allowed = [x["unit_site"] for x in moved[mm.upper()]]
        else:
            allowed = [reg[mm.upper()]["unit_site"]]
        if sid not in allowed:
            out.append(f"unit_site_not_allowed:{mm}:{sid}")
        if e["pad_plate"] != side:
            out.append(f"pad_plate:{mm}")
        try:
            rec = SD.unit_site(sid)
        except KeyError:
            out.append(f"unknown_unit_site:{sid}")
            continue
        if rec.get("module") != mm or not rec.get("usable", False):
            out.append(f"unit_site_unusable:{sid}")
        rel = (rec.get("release") or {}).get(f"d{float(rt['distance_m']):.2f}") or {}
        if not (rel.get("n") == int(rt["poses"]) and int(rel.get("fixed", 99)) <= int(rt["max_fixed"])):
            out.append(f"release_test:{sid}")
        if dense_check:
            dd = dense(sid)
            if not dd["ok"]:
                out.append(f"dense_siting:{sid}")
    return out


# ----------------------------------------------------------------------------------------------- telemetry
def telemetry_rules(site: dict, values=None) -> tuple:
    import telemetry_synth as TS
    v = values or SPEC.values()
    out = []
    chk = TS.check_telemetry(site)
    if not chk["ok"]:
        out += [f"telemetry:{p}" for p in chk["problems"]]
    n_sig = float(v["station.advisory.margin_sigma"]["n_sigma"])
    margins = dict(chk["units"])
    blk = site["cell"]["module"]
    side = chk["side"]
    sig = margins[blk]["max_margin_sigma"] if side == "below" else margins[blk].get("post_margin_sigma")
    if sig is None or abs(float(sig)) < n_sig:
        out.append("blocked_unit_margin")
    tel = site.get("telemetry") or {}
    exp = tel.get("advisory_expected") or {}
    if site.get("truth"):
        if bool(site["truth"].get("advisory_points_elsewhere")) != bool(chk["points_elsewhere"]):
            out.append("truth_points_elsewhere_vs_log")
        if bool(site["truth"].get("blocked_unit_over_limit")) != bool(chk["blocked_unit_over_limit"]):
            out.append("truth_over_limit_vs_log")
        if site["truth"].get("onset_min") != tel.get("onset_min"):
            out.append("truth_onset_vs_telemetry")
    if exp and bool(exp.get("points_elsewhere")) != bool(chk["points_elsewhere"]):
        out.append("advisory_expected")
    return out, dict(side=side, blocked_margin_sigma=sig, advisory_units=chk["advisory_units"], units=margins)


# ----------------------------------------------------------------------------------------------- all
def check_site(site: dict, model, data_, info, feats: dict, values=None, telemetry: bool = True, dense_check: bool = True) -> dict:
    """The generator's check_site on a built site in its current state (see the module doc).  Returns dict(ok,
    failures, station, pregrasp_records, pregrasp_counts, telemetry)."""
    from station import sites as SI
    v = values or SPEC.values()
    fails = []
    st = SI.check_site(info["site"], model, data_, info)
    fails += [f"station:{f}" for f in st["failures"]]
    recs, counts = PG.records(model, data_, info, feats)
    for name, c in counts.items():
        if c["accepted"] < 1:
            fails.append(f"no_pregrasp:{name}")
    fails += [f"pouch:{f}" for f in pouch_rules(site, feats, model, data_, info, v)]
    fails += [f"stowage:{f}" for f in stowage_rules(site, v, dense_check)]
    tdet = None
    if telemetry:
        tf, tdet = telemetry_rules(site, v)
        fails += tf
    fails += [f"arrangement:{f}" for f in arrangement(site, feats, v)]
    if site.get("register"):
        fails += [f"register:{f}" for f in RG.check(site, v)]
    return dict(ok=not fails, failures=fails, station=dict(ok=st["ok"], failures=st["failures"],
                                                         deepest_penetration_m=st["details"].get("deepest_penetration_m")),
                pregrasp_records=recs, pregrasp_counts=counts, telemetry=tdet)
