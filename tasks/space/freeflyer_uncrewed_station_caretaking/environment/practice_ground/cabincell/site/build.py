"""The scene of one site description (station_site/2; the site package format) -- shared by the cell and the
site generator -- and the robot-free measures of its items.

``station_site(site)`` turns a station_site/2 description into the input of the station builder (station v0.3,
``station.sites.build_site``):
  * ``stowage_units.<m>``: a station_site/2 entry ``{site, shift_m, moved_m, pad_plate}`` or a unit-site id of
    ``data/station/station_v03.json`` (``sb_<m>_reg`` / ``sb_<m>_mv<k>``) -> the unit-site id (``unit_site_id``);
  * ``pad_plate`` of every unit -> the builder's single pouch-pad plate side (c0 = '+b', c1 = '-b'; site_spec
    world.stowage_unit.pad_plate; the builder's 'outer' candidate is not in the site_spec list);
  * ``cabin_air_units.<cau>.flow_scale`` -> ``flow_cfm_per_inlet`` = 40 cfm x flow scale;
  * everything else is passed through (the builder ignores fields it does not know: register, records, truth, ...).
``build(site)`` compiles the scene (the robot and its dock from the cell's verified asset root when the site has the
robot; the dock alone otherwise, so that a robot-free site keeps the dock's collision geometry), ``rig`` wires it
without the robot (``station.site_rig.make_site_rig``: suction 50 Hz, item aero, units' air), ``measure`` reads every
item's sealed area and hold from the scene's suction sampler, ``seal1_areas`` the sampler's areas at seal 1 (the
quantity ``placement.sampler_area`` replicates), ``settle`` runs the robot-free rig and reports displacements.
"""
from __future__ import annotations

import copy
import math
import os
import sys

import numpy as np

from . import spec as SPEC
from . import LIB, CELL  # noqa: F401  (puts the station library and the simulation package on sys.path)

PAD_PLATE_SIDES = {"c0": "+b", "c1": "-b"}
DESIGN_FLOW_CFM = 40.0


# ----------------------------------------------------------------------------------------------- site description
def unit_site_id(module: str, entry) -> str:
    """The station v0.3 unit-site id of a station_site/2 ``stowage_units`` entry.

    A string is taken as the id itself.  A dict ``{site, shift_m, moved_m, pad_plate}``: ``unit_site`` (when present)
    is the id; else ``moved_m == 0`` -> the module's registered unit site (its ``base_site`` must be ``site``); a
    moved unit -> the one moved unit site of the module with that base site whose move distance is |moved_m| (an
    ambiguous or missing match raises: the site must then name ``unit_site``)."""
    from station import data
    if isinstance(entry, str):
        data.unit_site(entry)
        return entry
    if entry.get("unit_site"):
        rec = data.unit_site(entry["unit_site"])
        if rec["module"] != module:
            raise ValueError(f"unit site {entry['unit_site']} is not in module {module}")
        return entry["unit_site"]
    us = data.unit_sites()
    moved = float(entry.get("moved_m", 0.0))
    base = entry.get("site")
    if abs(moved) < 1e-9:
        sid = us["registered"].get(module) or us["practice"].get(module)
        rec = data.unit_site(sid)
        if base is not None and rec["base_site"] != base:
            raise ValueError(f"{module}: registered unit site {sid} has base site {rec['base_site']}, the site says {base}")
        return sid
    cands = [data.unit_site(s) for s in us["moved"].get(module, [])]
    cands = [r for r in cands if (base is None or r["base_site"] == base) and abs(float(r.get("moved_dist_m", -1)) - abs(moved)) < 1e-6]
    if len(cands) != 1:
        raise ValueError(f"{module}: moved_m {moved} matches {len(cands)} moved unit sites ({[r['id'] for r in cands]}); "
                         "name the unit site ('unit_site')")
    return cands[0]["id"]


def pad_side(site: dict) -> str:
    """The pouch pads' plate side of a site (one side for every unit; site_spec candidate ids c0 / c1, or a side)."""
    from station import stowage as SW
    sides = set()
    for e in (site.get("stowage_units") or {}).values():
        if isinstance(e, dict) and e.get("pad_plate") is not None:
            p = e["pad_plate"]
            sides.add(PAD_PLATE_SIDES.get(p, p))
    if not sides:
        return SW.PAD_FID_DEFAULT
    if len(sides) > 1:
        raise ValueError(f"the units of one site use different pad plates {sorted(sides)}")
    s = sides.pop()
    if s not in SW.PAD_FID_SIDES:
        raise ValueError(f"pad plate {s!r} unknown")
    return s


def station_site(site: dict) -> dict:
    """The station builder's site (station.sites, station_site/2 station part) of a station_site/2 description."""
    S = copy.deepcopy(site)
    S["stowage_units"] = {m: unit_site_id(m, e) for m, e in (site.get("stowage_units") or {}).items()}
    for u in S.get("cabin_air_units", {}).values():
        if "flow_scale" in u and "flow_cfm_per_inlet" not in u:
            u["flow_cfm_per_inlet"] = DESIGN_FLOW_CFM * float(u["flow_scale"])
    S.setdefault("clutter", [])
    S.setdefault("small_items", [])
    S.setdefault("robot", dict(present=False, berth=1))
    return S


# ----------------------------------------------------------------------------------------------- scene
def robot_paths() -> dict:
    """{'robot_xml', 'dock_xml', ...} of the cell's verified robot component (assetroot; raises when not found)."""
    if CELL not in sys.path:
        sys.path.append(CELL)
    import assetroot
    return assetroot.robot_paths()


def build(site: dict, look_root: str | None = None, dock: bool = True, paths: dict | None = None, extra: dict | None = None,
          near_clip: bool = True):
    """(MjModel, info) of a station_site/2 site.  The robot (site ``robot.present``) and the dock come from the asset
    root (or ``paths`` = {'robot_xml', 'dock_xml'}); ``dock=False`` leaves the dock out of a robot-free site.
    ``info['site']`` is the builder's site, ``info['site_full']`` the description given, ``info['pad_side']``."""
    from station import sites as SI
    S = station_site(site)
    present = bool((S.get("robot") or {}).get("present", False))
    P = paths or ((robot_paths() if (present or dock) else {}))
    side = pad_side(site)
    m, info = SI.build_site(S, look_root=look_root, robot_xml=P.get("robot_xml") if present else None,
                            dock_xml=P.get("dock_xml") if (present or dock) else None, pad_fid_side=side, extra=extra,
                            near_clip=near_clip)
    info["site_full"] = copy.deepcopy(site)
    info["pad_side"] = side
    return m, info


def initial_data(model, info):
    from station import sites as SI
    return SI.initial_data(model, info)


def rig(model, data_, info, suction_hz: int = 50, units_on: bool = True):
    """Robot-free wiring (station.site_rig.make_site_rig)."""
    from station import site_rig as SR
    return SR.make_site_rig(model, data_, info, suction_hz=suction_hz, units_on=units_on)


# ----------------------------------------------------------------------------------------------- items and measures
def items_of(model, info) -> list:
    """Every loose item of a built site: dict(id, kind 'sheet' | 'pouch', bodies (ids, ascending), half, grid,
    inlet)."""
    import mujoco
    from . import placement as PL
    bid = lambda n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n)
    out = []
    if info.get("blocker"):
        b = info["blocker"]
        out.append(dict(id="blocker", kind="sheet", bodies=sorted(bid(n) for n in b["bodies"]), half=tuple(b["half"]),
                        grid=PL.GRID_SHEET, inlet=b["inlet"]))
    for s in info.get("small_items", []):
        out.append(dict(id=s["id"], kind="pouch", bodies=[bid(s["body"])], half=tuple(s["half"]), grid=PL.GRID_POUCH,
                        inlet=s["inlet"]))
    return out


def face_of_inlet(feats: dict, iid: str):
    from . import placement as PL
    return PL.InletFace(PL.inlet(feats, iid))


def seal1_areas(model, data_, info) -> dict:
    """The scene's suction sampler at seal 1 (SuctionFaces with a seal decay length of 1e12 m, the station's cabin
    networks): {inlet id: sealed area} for the current poses (no step)."""
    import mujoco
    from spacekit import suction as SU
    from station import data as SD
    from station import site_rig as SR
    its = items_of(model, info)
    if not its:
        return {}
    feats = SD.features(info["scene"])
    vent = SR.ventilation(info["site"], feats)
    items = [(b, it["half"], it["grid"]) for it in its for b in it["bodies"]]
    S = SU.SuctionFaces(model, items, list(vent.cabin_fans.values()), g0=1e12)
    mujoco.mj_forward(model, data_)
    S.wrench(model, data_)
    return {it["inlet"]: float(S.covered.get(it["inlet"], 0.0)) for it in its}


def measure(R) -> dict:
    """Per item of a wired rig at its last suction evaluation: inlet, sealed area on it (the sampler, with the seal
    decay), hold [N] (the sampler's normal force), the unit's P_c, the geometric sealed area on the inlet (back face
    within 10 mm of the face plane and inside the opening; station.items.sealed_area), the body positions."""
    from station import items as IT
    from . import placement as PL
    out = {}
    for it in items_of(R.model, R.info):
        unit = PL.unit_of(R.feats, it["inlet"])
        net = R.vent.cabin_fans[unit[4:]]
        face = PL.InletFace(PL.inlet(R.feats, it["inlet"]))
        out[it["id"]] = dict(inlet=it["inlet"], unit=unit, sealed_m2=float(R.suction.covered.get(it["inlet"], 0.0)),
                             hold_N=float(R.suction.hold_force(it["bodies"])), Pc_Pa=float(net.Pc),
                             geometric_m2=float(IT.sealed_area(R.data, it["bodies"], it["half"], face)),
                             xpos=[R.data.xpos[b].tolist() for b in it["bodies"]])
    return out


def displacement(m0: dict, m1: dict) -> dict:
    """Per item: the largest body displacement [m] and the relative change of the sampler's sealed area."""
    out = {}
    for k, a in m0.items():
        b = m1[k]
        d = max(float(np.linalg.norm(np.asarray(p1) - np.asarray(p0))) for p0, p1 in zip(a["xpos"], b["xpos"]))
        ch = abs(b["sealed_m2"] - a["sealed_m2"]) / a["sealed_m2"] if a["sealed_m2"] > 0 else float("inf")
        out[k] = dict(disp_m=d, sealed_change_rel=ch)
    return out


def settle(R, seconds: float, sample_s: float = 1.0) -> dict:
    """Run a robot-free rig for ``seconds``; returns dict(start, end measures, per-item displacement, the largest
    displacement / sealed-area change seen at the samples, timing)."""
    from station import site_rig as SR
    m0 = measure(R)
    worst = {k: dict(disp_m=0.0, sealed_change_rel=0.0) for k in m0}

    def sample(rr):
        d = displacement(m0, measure(rr))
        for k, v in d.items():
            worst[k] = dict(disp_m=max(worst[k]["disp_m"], v["disp_m"]),
                            sealed_change_rel=max(worst[k]["sealed_change_rel"], v["sealed_change_rel"]))
    t = SR.run_free(R, seconds, sample_s, sample)
    m1 = measure(R)
    return dict(start=m0, end=m1, disp=displacement(m0, m1), worst=worst, timing=t)
