"""Stage A of an attempt: the station_site/2 description of one draw (the site rules), with
the network and replica windows.

Draw order (one ``numpy.random.Generator`` seeded by the attempt seed; every number from the runtime site_spec):
  1. the blocked inlet: uniform over the cell module's inlets;
  2. the three cabin air units: flow scale 1 + U(+-world.airflow.cau_flow_rel); the blocked unit is element-type with
     stops allowed (S) or pad-type with stops inhibited (P; one of two reasons); the others pad-type, stops allowed,
     r_face uniform in the pad range of their size (world.inlet.r_face);
  3. the blocker: a sheet of one of the three families (``items.draw_sheet``), natural placement and the blocked unit's
     r_face drawn jointly until the replica's sealed area is in world.blocker.placement.covered_m2 and the unit
     network's hold is in the tier window (``placement.sample_blocker``);
  4. 2-3 pouches (world.pouch.count) on pad-type inlets other than the blocked one, one per inlet, at most two per
     module, at least one outside the blocked module (world.pouch.rules); per pouch a family draw and its (columns,
     end) options (``placement.pouch_options``: 2-3 columns, the handle's bar clear of the frame's end bar, never an
     end over a wall box) whose replica sealed area is in the pouch cover window;
  5. the first combination of the pouches' options, in a random order, whose holds with every item of a unit in
     place are in the windows: each pouch in its unit group's window with and without the blocker
     (world.pouch.placement.hold_N_by_unit), the blocker still in its tier window with its unit's pouches;
  6. stowage units: the registered unit site of every module (world.stowage_unit.registered); in a 'moved' cell the
     blocked module's unit at one of its moved sites (world.stowage_unit.moved); pouch-pad plate side
     world.stowage_unit.pad_plate.selected;
  7. lighting: general or night per module (equal odds); a 'failed' cell's blocked module 'off' (failed circuit);
  8. robot: present, berth 1 or 2, start charge uniform in the cell's band (world.battery_start_pct, 0.1 %);
  9. inter-module links: flow uniform in world.airflow.inter_module_cfm (0.1 cfm), direction +-1;
 10. the lost-item register (``register.build``); clutter: none ([C], see pipeline module doc).
Rejections raise ``Reject(cause)``; the attempt is then over (the next attempt is a fresh draw).
"""
from __future__ import annotations

import copy
import itertools

import numpy as np

from cabincell.site import items as IT
from cabincell.site import placement as PL
from cabincell.site import spec as SPEC

from . import cells as CE
from . import register as RG

INHIBIT_REASONS = ("smoke detector check in progress", "equipment cooling requirement")      # the station service, [C]
LINKS = ("link_h2_l1", "link_h2_l2")
LIGHT_STATES = ("general", "night")


class Reject(Exception):
    """The attempt fails a generation condition; ``stage`` and ``cause`` name it, ``info`` the numbers."""

    def __init__(self, stage: str, cause: str, info: dict | None = None):
        super().__init__(f"{stage}:{cause}")
        self.stage, self.cause, self.info = stage, cause, dict(info or {})


def features():
    from station import data as SD
    return SD.features("mission")


def _u(rng, lo, hi) -> float:
    return float(rng.uniform(float(lo), float(hi)))


def unit_covers(site: dict, feats: dict, unit: str, areas: dict, with_blocker: bool = True) -> list:
    """Sealed areas [m^2] on a unit's inlets (inlet order) from ``areas`` {item id: area}."""
    out = []
    if with_blocker and site.get("blocker") and PL.unit_of(feats, site["blocker"]["inlet"]) == unit:
        out.append(float(areas["blocker"]))
    for P in site.get("small_items") or []:
        if PL.unit_of(feats, P["inlet"]) == unit:
            out.append(float(areas[P["id"]]))
    return out


def holds(site: dict, feats: dict, areas: dict, values=None) -> dict:
    """Network holds [N] of every item with all items in place, and of each pouch without the blocker:
    {id: dict(hold_N, hold_without_blocker_N, Pc_Pa, unit, n_inlets)}."""
    out = {}
    units = site["cabin_air_units"]
    for it in (["blocker"] if site.get("blocker") else []) + [P["id"] for P in site.get("small_items") or []]:
        iid = site["blocker"]["inlet"] if it == "blocker" else next(P["inlet"] for P in site["small_items"] if P["id"] == it)
        unit = PL.unit_of(feats, iid)
        U = units[unit]
        n = len(PL.unit_inlets(feats, unit))
        q = 40.0 * float(U.get("flow_scale", 1.0))
        A = float(areas[it])
        others_all = list(unit_covers(site, feats, unit, {**areas, it: 0.0}))
        others_all = [a for a in others_all if a > 0.0]
        pc, _ = PL.unit_pc(n, U["build"], float(U["r_face"]), [A] + others_all, q)
        row = dict(hold_N=pc * A, Pc_Pa=pc, unit=unit, n_inlets=n)
        if it != "blocker":
            o2 = [a for a in unit_covers(site, feats, unit, {**areas, it: 0.0}, with_blocker=False) if a > 0.0]
            pc2, _ = PL.unit_pc(n, U["build"], float(U["r_face"]), [A] + o2, q)
            row["hold_without_blocker_N"] = pc2 * A
        out[it] = row
    return out


def pouch_window(n_inlets: int, values=None) -> tuple:
    v = values or SPEC.values()
    g = v["world.pouch.placement"]["hold_N_by_unit"]
    w = g["pad_two_inlet_unit"] if int(n_inlets) == 2 else g["pad_three_inlet_unit"]
    return float(w[0]), float(w[1])


def window_problems(site: dict, feats: dict, areas: dict, values=None) -> list:
    """The cover and hold windows of every item on the given sealed areas (replica or in-scene)."""
    v = values or SPEC.values()
    out = []
    tier = site["cell"]["tier"]
    cw = PL.cover_window(v)
    hw = PL.tier_window(tier, v)
    pcw = v["world.pouch.placement"]["covered_m2"]
    H = holds(site, feats, areas, v)
    a = float(areas["blocker"])
    if not (cw[0] <= a <= cw[1]):
        out.append(("blocker_cover", a))
    if not (hw[0] <= H["blocker"]["hold_N"] <= hw[1]):
        out.append(("blocker_hold", H["blocker"]["hold_N"]))
    for P in site.get("small_items") or []:
        a = float(areas[P["id"]])
        if not (float(pcw[0]) <= a <= float(pcw[1])):
            out.append((f"{P['id']}_cover", a))
        lo, hi = pouch_window(H[P["id"]]["n_inlets"], v)
        if not (lo <= H[P["id"]]["hold_without_blocker_N"] and H[P["id"]]["hold_N"] <= hi):
            out.append((f"{P['id']}_hold", [H[P["id"]]["hold_without_blocker_N"], H[P["id"]]["hold_N"]]))
    return out


def _pick_pouch_inlets(rng, feats, units, blk_inlet, m_blk, n_p, values) -> list:
    rules = values["world.pouch.rules"]
    cand = [i["id"] for i in feats["v02"]["inlets"] if i["id"] != blk_inlet and units[i["cab_air_unit"]]["build"] == "pad"]
    order = [cand[int(j)] for j in rng.permutation(len(cand))]
    pick, per = [], {}
    for iid in order:
        mm = PL.inlet(feats, iid)["module"]
        if per.get(mm, 0) < int(rules["max_per_module"]) and len(pick) < n_p:
            pick.append(iid)
            per[mm] = per.get(mm, 0) + 1
    outside = sum(1 for iid in pick if PL.inlet(feats, iid)["module"] != m_blk)
    if len(pick) < n_p or outside < int(rules["min_outside_blocked_module"]):
        raise Reject("A", "pouch_rules", dict(pick=pick))
    # station order of the inlets, so the pouch numbering does not depend on the permutation's tail
    ids = [i["id"] for i in feats["v02"]["inlets"]]
    return sorted(pick, key=ids.index)


def draw(label: str, cell: str, k: int, j: int, kind: str, feats=None, values=None) -> tuple:
    """(site, info) of attempt j of candidate (label, cell, k); raises Reject.  ``site`` carries every station_site/2
    field; telemetry / fixture_records / pregrasp_records / settle / truth measures / provenance are placeholders the
    later stages fill.  ``info`` has the replica areas and network holds and the seed texts."""
    v = values or SPEC.values()
    feats = feats or features()
    m, tier = CE.split(cell)
    row = CE.row(cell, v)
    seed = CE.attempt_seed(label, cell, k, j)
    rng = np.random.default_rng(seed)
    ins = [i["id"] for i in feats["v02"]["inlets"] if i["module"] == m]
    blk_inlet = ins[int(rng.integers(len(ins)))]
    rel = float(v["world.airflow"]["cau_flow_rel"])
    units = {}
    for mm in CE.MODULE_ORDER:
        uid = f"cau_{mm}"
        n = len(PL.unit_inlets(feats, uid))
        fs = round(1.0 + _u(rng, -rel, rel), 4)
        if mm == m and tier == "S":
            units[uid] = dict(build="element", r_face=None, fan_stop="allowed", inhibit_reason=None, flow_scale=fs)
        elif mm == m:
            units[uid] = dict(build="pad", r_face=None, fan_stop="inhibited",
                              inhibit_reason=INHIBIT_REASONS[int(rng.integers(len(INHIBIT_REASONS)))], flow_scale=fs)
        else:
            lo, hi = PL.r_face_range("pad", n, v)
            units[uid] = dict(build="pad", r_face=_u(rng, lo, hi), fan_stop="allowed", inhibit_reason=None, flow_scale=fs)
        units[uid].update(clean_reference_pa=None, advisory_limit_pa=None)
    ub = units[f"cau_{m}"]
    sheet = IT.draw_sheet(rng)
    try:
        sb = PL.sample_blocker(rng, tier, blk_inlet, feats, sheet=sheet, flow_cfm=40.0 * ub["flow_scale"], values=v)
    except RuntimeError as exc:
        raise Reject("A", "blocker_draw", dict(error=str(exc)))
    ub["r_face"] = float(sb["r_face"])
    B = sb["blocker"]
    n_p = int(rng.choice([int(x) for x in v["world.pouch.count"]]))
    pin = _pick_pouch_inlets(rng, feats, units, blk_inlet, m, n_p, v)
    pcw = v["world.pouch.placement"]["covered_m2"]
    areas0 = {"blocker": float(sb["cover_replica_m2"])}
    choices = []                       # per pouch: [(entry, replica area)] of its options in the cover window
    for n_, iid in enumerate(pin):
        pid = f"pouch_{n_ + 1}"
        pch = IT.draw_pouch(rng)
        ok = []
        for c, e in PL.pouch_options(iid, pch, v):
            S = PL.pouch_at_end(pid, iid, pch, c, e, None, v)
            S["family"] = "tool_pouch"
            A = PL.sampler_area(feats, S, "pouch", v)
            if float(pcw[0]) <= A <= float(pcw[1]):
                ok.append((S, A))
        if not ok:
            raise Reject("A", "pouch_cover", dict(pouch=pid, inlet=iid, size=pch["size"]))
        choices.append(ok)
    combos = list(itertools.product(*[range(len(c)) for c in choices]))
    pouches, areas = None, None
    probe = dict(cell=dict(module=m, tier=tier), cabin_air_units=units, blocker=sb["blocker"])
    for q in rng.permutation(len(combos)):
        pick = [choices[i][int(x)] for i, x in enumerate(combos[int(q)])]
        ar = dict(areas0, **{S["id"]: A for S, A in pick})
        trial = dict(probe, small_items=[S for S, _ in pick])
        if not window_problems(trial, feats, ar, v):
            pouches, areas = [copy.deepcopy(S) for S, _ in pick], ar
            break
    if pouches is None:
        last = window_problems(dict(probe, small_items=[c[0][0] for c in choices]), feats, dict(areas0, **{c[0][0]["id"]: c[0][1] for c in choices}), v)
        raise Reject("A", "pouch_hold_window", dict(combos=len(combos), first_combo_problems=last))
    stow = {}
    reg_sites = v["world.stowage_unit.registered"]
    side = v["world.stowage_unit.pad_plate"]["selected"]
    for mm in CE.MODULE_ORDER:
        sid = reg_sites[mm.upper()]["unit_site"]
        if row["stowage_unit"] == "moved" and mm == m:
            mv = v["world.stowage_unit.moved"][mm.upper()]
            sid = mv[int(rng.integers(len(mv)))]["unit_site"]
        stow[mm] = dict(unit_site=sid, pad_plate=side)
    lighting = {mm: LIGHT_STATES[int(rng.integers(2))] for mm in CE.MODULE_ORDER}
    if row["light_circuit"] == "failed":
        lighting[m] = "off"
    band = v["world.battery_start_pct"][row["battery"]]
    robot = dict(present=True, berth=int(rng.choice([int(x) for x in v["world.berths"]])), battery_pct=round(_u(rng, *band), 1))
    imc = v["world.airflow"]["inter_module_cfm"]
    inter = {lk: dict(flow_cfm=round(_u(rng, *imc), 1), direction=int(rng.choice([1, -1]))) for lk in LINKS}
    site = dict(schema="station_site/2", site_id=CE.site_id(kind, cell, k), scene="mission", kind=kind, seed=int(seed),
                cell=dict(module=m, tier=tier, k=int(k), label=label), cabin_air_units=units, inter_module=inter,
                blocker=B, small_items=pouches, stowage_units=stow, clutter=[], lighting=lighting, robot=robot,
                telemetry=None, register=None, fixture_records=dict(blocker=None), pregrasp_records={}, settle=None, truth=None,
                provenance=None)
    probs = window_problems(site, feats, areas, v)
    if probs:
        raise Reject("A", "replica_window", dict(problems=probs))
    reg = RG.build(rng, site, v)
    B["register_ref"] = reg["refs"]["blocker"]
    for P in pouches:
        P["register_ref"] = reg["refs"].get(P["id"])
    site["register"] = dict(entries=reg["entries"], unregistered=reg["unregistered"])
    H = holds(site, feats, areas, v)
    info = dict(seed_text=CE.seed_text(label, cell, k, j), candidate_seed=CE.candidate_seed(label, cell, k), attempt=int(j),
                replica_areas=areas, replica_holds=H, blocker_tries=int(sb["tries"]),
                cover_geometric_m2=float(sb["cover_geometric_m2"]), chain_fold_deg=PL.chain_angle_fold(B["theta_deg"]))
    return site, info


def with_truth(site: dict, feats: dict, areas: dict, values=None) -> dict:
    """A copy of ``site`` whose truth (tier, ids, moved unit, failed light, cover_m2 and hold_N from ``areas``) and
    telemetry.pouch_cover_m2 are filled (the telemetry draw needs them)."""
    v = values or SPEC.values()
    S = copy.deepcopy(site)
    m, tier = S["cell"]["module"], S["cell"]["tier"]
    row = CE.row(CE.cell_key(m, tier), v)
    H = holds(S, feats, areas, v)
    S["truth"] = dict(tier=tier, blocked_inlet=S["blocker"]["inlet"], blocked_unit=f"cau_{m}", blocker_ref=S["blocker"]["register_ref"],
                      onset_min=None, pouches=[P["id"] for P in S["small_items"]], hold_N=float(H["blocker"]["hold_N"]),
                      cover_m2=float(areas["blocker"]), advisory_points_elsewhere=False, blocked_unit_over_limit=False,
                      moved_stowage_unit=(m if row["stowage_unit"] == "moved" else None),
                      failed_light_module=(m if row["light_circuit"] == "failed" else None))
    S["telemetry"] = dict(pouch_cover_m2={P["id"]: float(areas[P["id"]]) for P in S["small_items"]})
    return S
