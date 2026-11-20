"""rehearsal_debrief.py -- the observation sheet of one rehearsal run and the run's station record.

The sheet says what a crew member on board would tell you after watching the run -- categories, counts and the
station's own records, never a measurement of the hidden world:

  return_inlets_clear_at_some_time  {inlet label: bool} for every inlet an item lay on at the start: was the inlet
                                    free of any item for at least CLEAR_SAMPLES consecutive record samples
  blocker                           {"ref": register ref, "final_place": one of BLOCKER_PLACES} | null
  pouches                           [{"ref": register ref, "final_place": one of POUCH_PLACES}]
  fan_stops                         {"count": int, "total_s": seconds (0.1 s)}
  flight_rule_violations            {"count": int, "kinds": [kind, ...]} (the station's violation records)
  docked_at_end                     bool
  battery_pct_at_dock               the charge at the first DOCKED (0.1 %) | null
  hard_contacts                     int: new robot-station contacts faster than HARD_SPEED_M_S along the contact
                                    normal, or harder than HARD_FORCE_N on a hatch seal, a luminaire or a fiducial plate
Final place (at the end = after the 60 s settle; the first that applies): in_hand (a grasp bar of the item inside the
closed finger cage, tendon commanded closed); on a face (the item lies within the suction model's seal box of a
suction face: holding_screen, pouch_pad or on_return_inlet -- the face with the most of the item); floating (no
contact with anything); elsewhere (touching the structure, the robot, the dock or another item without lying on a
face).  The blocker on a pouch pad is 'elsewhere' (the blocker list of the givens has no pouch pad).

Never in the sheet or the record: a pose of the robot or of an item, a trajectory, a sealed or covered area, a score,
a threshold of the grader, a difference to the reference.  ``FIELDS`` is the whitelist of the returned fields.  Uses the cell's record (the record format) and
the cell's live objects at the end; numpy and mujoco.
"""
from __future__ import annotations

import math

BLOCKER_PLACES = ("holding_screen", "in_hand", "floating", "on_return_inlet", "elsewhere")
POUCH_PLACES = ("pouch_pad", "holding_screen", "in_hand", "floating", "on_return_inlet", "elsewhere")
CLEAR_SAMPLES = 2              # consecutive record samples (cell.record_period_s apart) with no item on the inlet
HARD_SPEED_M_S = 0.15          # a hard contact (normal speed of the robot's contact point)
HARD_FORCE_N = 5.0             # ... or this force on a delicate class
DELICATE = ("hatch_seal", "lamp", "fiducial_plate")
FACE_PLACE = {"holding_screen": "holding_screen", "pouch_pad": "pouch_pad", "return_inlet": "on_return_inlet"}
FIELDS = ("return_inlets_clear_at_some_time", "blocker", "pouches", "fan_stops", "flight_rule_violations",
          "docked_at_end", "battery_pct_at_dock", "hard_contacts")
RUN_FIELDS = ("card", "run", "ended_by", "program_exit", "sim_time_s", "sim_limit_s", "runs_left", "sim_minutes_left",
              "snapshot")
STATION_FIELDS = ("t_s", "channels", "requests", "advisories_in_run")


def _label(inlet_id: str) -> str:
    """'ri_l3_1' -> 'RI-L3-1' (the map's label)."""
    p = inlet_id.split("_")
    return f"RI-{p[1].upper()}-{p[2]}"


def _touching(cell, item: str) -> bool:
    """Whether any geom of the item is in contact with a geom that is not the item's."""
    import numpy as np
    m, d = cell.model, cell.data
    n = int(d.ncon)
    if n == 0:
        return False
    bodies = set(int(b) for b in cell.world.items[item]["bodies"])
    gb = np.asarray(m.geom_bodyid)
    g = d.contact.geom[:n]
    a = np.isin(gb[g[:, 0]], list(bodies))
    b = np.isin(gb[g[:, 1]], list(bodies))
    return bool(np.any(a ^ b))


def final_place(cell, item: str, row: dict, kind: str) -> str:
    """The category of an item in the final record row (module doc)."""
    from cabincell import predicates as P
    st = (row.get("items") or {}).get(item) or {}
    if st.get("in_gripper") is True:
        return "in_hand"
    best, face = 0.0, None
    for f, a in sorted((st.get("seal_m2") or {}).items()):
        if a is not None and float(a) > best:
            best, face = float(a), f
    if face is not None:
        place = FACE_PLACE.get(P.face_kind(face))
        if place is not None:
            if kind == "blocker" and place == "pouch_pad":
                return "elsewhere"
            return place
    return "elsewhere" if _touching(cell, item) else "floating"


def clear_at_some_time(rows: list, inlet: str) -> bool:
    run = 0
    for r in rows:
        v = (r.get("inlets") or {}).get(inlet)
        if v is not None and float(v) <= 0.0:
            run += 1
            if run >= CLEAR_SAMPLES:
                return True
        else:
            run = 0
    return False


def hard_contacts(events: list) -> int:
    n = 0
    for e in events or []:
        v = e.get("speed_m_s")
        f = e.get("force_N")
        v = float("inf") if v is None or not math.isfinite(float(v)) else float(v)
        f = float("inf") if f is None or not math.isfinite(float(f)) else float(f)
        if v > HARD_SPEED_M_S or (e.get("other_class") in DELICATE and f > HARD_FORCE_N):
            n += 1
    return n


def sheet(cell, record: dict) -> dict:
    """The observation sheet of a finished run (``record``: the cell's record after the settle)."""
    from cabincell import predicates as P
    rows = record.get("state") or []
    last = rows[-1] if rows else {}
    first = rows[0] if rows else {}
    items = record.get("items") or {}
    site = cell.site
    inl0 = sorted(k for k, v in (first.get("inlets") or {}).items() if v is not None and float(v) > 0.0)
    start_inlets = sorted(set(inl0) | {it.get("inlet") for it in items.values() if it.get("inlet")})
    out = dict(return_inlets_clear_at_some_time={_label(i): clear_at_some_time(rows, i) for i in start_inlets})
    out["blocker"] = None
    if "blocker" in items:
        out["blocker"] = dict(ref=(site.get("blocker") or {}).get("register_ref"),
                              final_place=final_place(cell, "blocker", last, "blocker"))
    pl = []
    for P_ in site.get("small_items") or []:
        if P_["id"] in items:
            pl.append(dict(ref=P_.get("register_ref"), final_place=final_place(cell, P_["id"], last, "pouch")))
    out["pouches"] = pl
    t_end = float((record.get("times") or {}).get("end") or 0.0)
    ivs = P.fan_stop_intervals(record.get("fan_stops"), t_end)
    out["fan_stops"] = dict(count=len(ivs), total_s=round(sum(b - a for _, a, b in ivs), 1))
    vio = [str(v.get("kind")) for v in record.get("violations") or []]
    out["flight_rule_violations"] = dict(count=len(vio), kinds=vio)
    dock = record.get("dock") or {}
    out["docked_at_end"] = bool(dock.get("docked_at_end"))
    b = dock.get("battery_pct_at_docked")
    out["battery_pct_at_dock"] = None if b is None else round(float(b), 1)
    out["hard_contacts"] = hard_contacts(record.get("hard_contacts"))
    return out


def snapshot_sheet(cell) -> dict:
    """The sheet of a run in progress (no settle; the latest record sample and the cell's current events): what the
    service keeps should the run's process die."""
    rec = dict(state=list(cell.recorder.rows), items=cell.world.items_record(), times=dict(end=cell.t_mission()),
               fan_stops=cell.svc.fan_stops(), violations=[dict(e) for e in cell.svc.violations],
               dock=dict(docked_at_end=bool(cell.tf.layer.docked), battery_pct_at_docked=cell.battery_at_docked),
               hard_contacts=cell.contacts.record())
    return sheet(cell, rec)


def station_record(cell) -> dict:
    """The station's own record of the run: the live channels at 1 Hz (columns), the station requests with their
    replies, the advisories raised during the run."""
    samples = list(cell.svc.samples)
    names = sorted({k for s in samples for k in s["channels"]})
    cols = {k: [s["channels"].get(k) for s in samples] for k in names}
    N = float(cell.svc.N)
    adv = [dict(e) for e in cell.svc.advisories if float(e.get("t_min", 0.0)) >= N]
    return dict(t_s=[float(s["t"]) for s in samples], channels=cols, requests=[dict(e) for e in cell.svc.requests],
                advisories_in_run=adv)
