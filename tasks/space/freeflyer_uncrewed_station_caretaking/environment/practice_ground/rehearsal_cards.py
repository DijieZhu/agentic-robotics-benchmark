"""rehearsal_cards.py -- the five rehearsal cards of the training module L3 and the draw of one run's site
(runtime site_spec ``rehearsal.*``).

The training module L3 (the station library's 'practice' scene): one cabin air unit CAU-L3 whose two return
inlets RI-L3-1 / RI-L3-2 sit on flat rack faces (the facility can fit either inlet build), one stowage unit SB-L3 at
its unit site (holding screen, two pouch pads, pad-plate side '+b'), the training dock with berths 1 and 2, the
module's lights.  Nothing of the scored modules is in the scene, and no decoy is drawn: no slow drift, no low start
charge, no moved stowage unit, no failed light circuit.

Cards (``rehearsal.cards``):
  EX-A   navigation, localisation and docking: no item; lights general or night (a request switches them)
  EX-B1  one soft training sheet on a pad-type inlet; stop requests to CAU-L3 inhibited (tier P hold window)
  EX-B2  one soft training sheet on an element-type inlet; stop requests allowed (tier S hold window)
  EX-C   two weakly held training pouches, one on each inlet of a pad-type unit; stops allowed
  EX-D   the whole sortie: a sheet on one inlet, tier S or P drawn (S: element-type unit, stops allowed, no pouch; P:
         pad-type unit, stops inhibited, a pouch on the other inlet), the register, the log, the dock
Every run draws again (one ``numpy.random.Generator`` seeded from the run's seed text): the item's plane rotation and
centre offset (natural placement), the sheet family, size and mass, the filter state (r_face of the unit in the
build's range of a two-inlet unit), the unit's flow scale, the lights, the berth, the start charge (high band), the
pouches' size, mass and (columns, end) option, the register's absent entries, the telemetry's noise and onsets.  The
windows are those of the scored world (site_spec ``world.blocker.*``, ``world.pouch.placement``), judged on the
analytic replica of the scene's sealed areas (cabincell.site.placement.sampler_area).

Training items have colours that no scored item has (TRAINING_COLOURS, added to the station builder's colour table
by ``install_colours``); their physics (families, sizes, masses, straps, loops, handles) is the scored items'.

Seeds: the seed text of a run is ``<LABEL>/<card>/<nonce>`` (the nonce: 16 random bytes in hex, drawn by the
service at the reset), its seed the first 8 bytes (little endian) of the text's sha256 -- the construction of the site
generator (tests/review/sitegen/gen/cells.py ``_seed``) under a label of its own, so no run can share a seed text
with a trial site, a debug site or a hidden candidate, and the scene ('practice', module L3) shares no inlet with them.

Adapted from the site generator's draw and register (tests/review/sitegen/gen/draw.py, register.py): the same
placement and telemetry functions of the
cell (cabincell.site.placement, cabincell.site.items, telemetry_synth.draw_telemetry with no cell row = no decoy),
one unit instead of three, the training colours, no clutter.  numpy and the cell's own modules.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
LABEL = "rehearsal_l3_v1"
CARDS = ("EX-A", "EX-B1", "EX-B2", "EX-C", "EX-D")
MODULE = "l3"
UNIT = "cau_l3"
INLETS = ("ri_l3_1", "ri_l3_2")
INHIBIT_REASONS = ("smoke detector check in progress", "equipment cooling requirement")
LIGHT_STATES = ("general", "night")
# colours of the training items: none of them is a colour of a scored item (cabincell.site.items colour_names()), and
# each is at an RGB distance >= 0.30 from the eight scored colours, from the grasp-bar red of the items' straps, loops
# and handles (spacekit.fixtures mat_handle 0.90 0.10 0.10) and from the other training colours (the
# first sheet colour, red 0.80 0.14 0.16, was 0.12 from the grasp-bar red)
TRAINING_COLOURS = {"cyan": "0.20 0.85 0.95 1", "white": "0.91 0.91 0.89 1", "magenta": "0.80 0.20 0.62 1"}
SHEET_COLOURS = ("cyan", "white")
POUCH_COLOURS = ("magenta", "white")
SPOTS = ("equipment rack face", "hatch area", "near a crew handrail", "cabin centre")
MAX_TRIES = 200

WORK_ORDERS = {
    "EX-A": ("Rehearsal card EX-A, laboratory module L3 (the rehearsal facility).\n"
             "Navigation, localisation and docking: leave the dock, fly in the module, use the lights and the robot's\n"
             "lamps as you need them, and end the run on the dock. No loose item is placed for this card.\n"),
    "EX-B1": ("Rehearsal card EX-B1, laboratory module L3 (the rehearsal facility).\n"
              "A soft rehearsal item lies on a pad-type return inlet of CAU-L3; stop requests to CAU-L3 are inhibited.\n"
              "Remove it, fix it on the holding screen of SB-L3, keep the station's flight rules and end the run on the dock.\n"),
    "EX-B2": ("Rehearsal card EX-B2, laboratory module L3 (the rehearsal facility).\n"
              "A soft rehearsal item lies on an element-type return inlet of CAU-L3; stop requests to CAU-L3 are allowed.\n"
              "Remove it, fix it on the holding screen of SB-L3, keep the station's flight rules and end the run on the dock.\n"),
    "EX-C": ("Rehearsal card EX-C, laboratory module L3 (the rehearsal facility).\n"
             "Two rehearsal pouches are held weakly on the return inlets of CAU-L3. Put each on a pouch pad of SB-L3,\n"
             "keep the station's flight rules and end the run on the dock.\n"),
    "EX-D": ("Rehearsal card EX-D, laboratory module L3 (the rehearsal facility).\n"
             "The whole sortie: the 72 h log shows a return-air step on CAU-L3. Find the cause and restore the return air,\n"
             "stow the loose items you find on the return inlets, keep the station's flight rules and end the run on the dock.\n"),
}


# ------------------------------------------------------------------------------------------------ setup
def install_colours() -> None:
    """Add the training colours to the station builder's colour table (station.sites.COLOURS; idempotent)."""
    from station import sites as SI
    for k, rgba in TRAINING_COLOURS.items():
        if k in SI.COLOURS and SI.COLOURS[k] != rgba:
            raise RuntimeError(f"colour {k} already means {SI.COLOURS[k]}")
        SI.COLOURS[k] = rgba


def seed_of(seed_text: str) -> int:
    """First 8 bytes (little endian) of sha256(seed_text) (the site generator's construction)."""
    return int.from_bytes(hashlib.sha256(seed_text.encode("ascii")).digest()[:8], "little")


def seed_text(card: str, nonce_hex: str) -> str:
    if card not in CARDS:
        raise ValueError(card)
    return f"{LABEL}/{card}/{nonce_hex}"


def derived(seed: int, what: str) -> int:
    return seed_of(f"{int(seed)}/{what}")


class Redraw(Exception):
    """One draw of a run failed a window; the next draw of the same generator follows."""


# ------------------------------------------------------------------------------------------------ the draw
def _u(rng, lo, hi) -> float:
    return float(rng.uniform(float(lo), float(hi)))


def _r(x, nd=3):
    return float(round(float(x), nd))


def _features():
    from station import data as SD
    return SD.features("practice")


def _training_sheet(rng):
    from cabincell.site import items as IT
    sh = IT.draw_sheet(rng)
    sh["colour"] = SHEET_COLOURS[int(rng.integers(len(SHEET_COLOURS)))]
    return sh


def _training_pouch(rng):
    from cabincell.site import items as IT
    return IT.draw_pouch(rng, colour=POUCH_COLOURS[int(rng.integers(len(POUCH_COLOURS)))])


def _pouch_window(v):
    w = v["world.pouch.placement"]["hold_N_by_unit"]["pad_two_inlet_unit"]
    return float(w[0]), float(w[1])


def _pouch_on(rng, pid, iid, feats, v):
    """(entry, replica sealed area) of one training pouch on an inlet: a (columns, end) option in the cover window."""
    from cabincell.site import placement as PL
    pch = _training_pouch(rng)
    pcw = v["world.pouch.placement"]["covered_m2"]
    opts = PL.pouch_options(iid, pch, v)
    order = [opts[int(j)] for j in rng.permutation(len(opts))]
    for c, e in order:
        S = PL.pouch_at_end(pid, iid, pch, c, e, None, v)
        S["family"] = "tool_pouch"
        A = PL.sampler_area(feats, S, "pouch", v)
        if float(pcw[0]) <= A <= float(pcw[1]):
            return S, A
    raise Redraw("pouch_cover")


def _unit_pc(build, r, fs, areas):
    from cabincell.site import placement as PL
    pc, _ = PL.unit_pc(len(INLETS), build, float(r), [a for a in areas if a > 0.0], 40.0 * float(fs))
    return pc


def _register(rng, site, v) -> dict:
    """The run's lost-item register: the present items (the sheet, the pouches) and 2-4 absent entries (two sheets
    with a carry strap whose (class, colour) differ from the present one's, training pouches); refs LI-01 .."""
    from cabincell.site import items as IT
    present = []
    B = site.get("blocker")
    if B:
        present.append(dict(cls=IT.register_class(B["family"]), colour=B["colour"], size_m=[_r(x) for x in B["size"]],
                            mass_kg=_r(B["mass"]), features="carry strap + handle loop", item="blocker"))
    for P in site.get("small_items") or []:
        present.append(dict(cls=IT.register_class("tool_pouch"), colour=P["colour"], size_m=[_r(x) for x in P["size"]],
                            mass_kg=_r(P["mass"]), features="handle", item=P["id"]))
    pairs = [(f, c) for f in IT.FAMILY_ORDER for c in SHEET_COLOURS if not (B and (f, c) == (B["family"], B["colour"]))]
    absent = []
    for idx in rng.permutation(len(pairs))[:2]:
        f, c = pairs[int(idx)]
        sh = IT.draw_sheet(rng, family=f)
        absent.append(dict(cls=IT.register_class(f), colour=c, size_m=[_r(x) for x in sh["size"]], mass_kg=_r(sh["mass"]),
                           features="carry strap + handle loop", item=None))
    for _ in range(int(rng.integers(0, 3))):
        P = _training_pouch(rng)
        absent.append(dict(cls=IT.register_class("tool_pouch"), colour=P["colour"], size_m=[_r(x) for x in P["size"]],
                           mass_kg=_r(P["mass"]), features="handle", item=None))
    allx = present + absent
    entries, refs = [], {}
    for n, idx in enumerate(rng.permutation(len(allx))):
        e = allx[int(idx)]
        ref = f"LI-{n + 1:02d}"
        entries.append({"ref": ref, "class": e["cls"], "colour": e["colour"], "size_m": e["size_m"], "mass_kg": e["mass_kg"],
                        "features": e["features"], "location": "L3, " + SPOTS[int(rng.integers(len(SPOTS)))],
                        "present": e["item"] is not None, "item": e["item"]})
        if e["item"] is not None:
            refs[e["item"]] = ref
    return dict(entries=entries, unregistered=[], refs=refs)


def public_register(register: dict) -> list:
    keys = ("ref", "class", "colour", "size_m", "mass_kg", "features", "location")
    return [{k: e[k] for k in keys} for e in register["entries"]]


def _one(card, rng, seed, feats, v):
    """One draw of a card's site (raises Redraw)."""
    from cabincell.site import placement as PL
    import telemetry_synth as TS
    rel = float(v["world.airflow"]["cau_flow_rel"])
    fs = round(1.0 + _u(rng, -rel, rel), 4)
    tier = None
    if card in ("EX-B1", "EX-B2"):
        tier = "P" if card == "EX-B1" else "S"
    elif card == "EX-D":
        tier = ("S", "P")[int(rng.integers(2))]
    build = "element" if tier == "S" else "pad"
    inhibited = tier == "P"
    unit = dict(build=build, r_face=None, fan_stop="inhibited" if inhibited else "allowed",
                inhibit_reason=INHIBIT_REASONS[int(rng.integers(len(INHIBIT_REASONS)))] if inhibited else None,
                flow_scale=fs, clean_reference_pa=None, advisory_limit_pa=None)
    blocker, pouches, areas = None, [], {}
    if tier is not None:
        blk_inlet = INLETS[int(rng.integers(len(INLETS)))]
        other = INLETS[1 - INLETS.index(blk_inlet)]
        if card == "EX-D" and tier == "P":
            S, A = _pouch_on(rng, "pouch_1", other, feats, v)
            pouches.append(S)
            areas["pouch_1"] = A
        sheet = _training_sheet(rng)
        try:
            sb = PL.sample_blocker(rng, tier, blk_inlet, feats, sheet=sheet, others=tuple(areas.values()),
                                   flow_cfm=40.0 * fs, max_tries=4000, values=v)
        except RuntimeError:
            raise Redraw("blocker_draw")
        blocker = sb["blocker"]
        unit["r_face"] = float(sb["r_face"])
        areas["blocker"] = float(sb["cover_replica_m2"])
    else:
        lo, hi = PL.r_face_range("pad", len(INLETS), v)
        unit["r_face"] = _u(rng, lo, hi)
        if card == "EX-C":
            for k, iid in enumerate(INLETS):
                S, A = _pouch_on(rng, f"pouch_{k + 1}", iid, feats, v)
                pouches.append(S)
                areas[S["id"]] = A
    # the windows with every item in place (and each pouch without the sheet)
    lo_p, hi_p = _pouch_window(v)
    for P in pouches:
        A = areas[P["id"]]
        with_all = _unit_pc(build, unit["r_face"], fs, list(areas.values())) * A
        without = _unit_pc(build, unit["r_face"], fs, [a for k, a in areas.items() if k != "blocker"]) * A
        if not (lo_p <= without and with_all <= hi_p):
            raise Redraw("pouch_hold")
    if blocker is not None:
        hw = PL.tier_window(tier, v)
        h = _unit_pc(build, unit["r_face"], fs, list(areas.values())) * areas["blocker"]
        if not (hw[0] <= h <= hw[1]):
            raise Redraw("blocker_hold")
    band = v["world.battery_start_pct"]["high"]
    site = dict(schema="station_site/2", site_id=None, scene="practice", kind="rehearsal", seed=int(seed), cell=None,
                card=card, tier=tier, cabin_air_units={UNIT: unit}, inter_module={}, blocker=blocker, small_items=pouches,
                stowage_units={MODULE: dict(unit_site=_unit_site(v), pad_plate=v["world.stowage_unit.pad_plate"]["selected"])},
                clutter=[], lighting={MODULE: LIGHT_STATES[int(rng.integers(2))]},
                robot=dict(present=True, berth=int(rng.choice([int(x) for x in v["world.berths"]])),
                           battery_pct=round(_u(rng, *band), 1)),
                telemetry=None, register=None, truth=dict(cover_m2=areas.get("blocker")))
    reg = _register(rng, site, v)
    if blocker is not None:
        blocker["register_ref"] = reg["refs"]["blocker"]
    for P in pouches:
        P["register_ref"] = reg["refs"].get(P["id"])
    site["register"] = dict(entries=reg["entries"], unregistered=reg["unregistered"])
    probe = copy.deepcopy(site)
    probe["telemetry"] = dict(pouch_cover_m2={P["id"]: areas[P["id"]] for P in pouches})
    try:
        tel, patch, _info = TS.draw_telemetry(probe, np.random.default_rng(derived(seed, "telemetry")), max_tries=40)
    except TS.Reject as R:
        raise Redraw("telemetry:" + str(R)[:40])
    site["telemetry"] = tel
    site["cabin_air_units"][UNIT].update(patch[UNIT])
    return site, areas


def _unit_site(v) -> str:
    from station import data as SD
    return SD.unit_sites()["practice"][MODULE]


def draw_site(card: str, nonce_hex: str, values=None) -> tuple:
    """(site, info) of one rehearsal run (module doc); deterministic in (card, nonce)."""
    from cabincell.site import spec as SPEC
    v = values or SPEC.values()
    if list(v["rehearsal.cards"]) != list(CARDS):
        raise RuntimeError(f"the runtime site_spec's cards {v['rehearsal.cards']} are not {CARDS}")
    st = seed_text(card, nonce_hex)
    seed = seed_of(st)
    rng = np.random.default_rng(seed)
    feats = _features()
    causes = {}
    for k in range(MAX_TRIES):
        try:
            site, areas = _one(card, rng, seed, feats, v)
        except Redraw as R:
            causes[str(R)] = causes.get(str(R), 0) + 1
            continue
        site["site_id"] = f"rehearsal_{card.lower().replace('-', '')}_{nonce_hex[:8]}"
        info = dict(seed_text=st, seed=seed, draws=k + 1, causes=causes, replica_areas=areas)
        return site, info
    raise RuntimeError(f"{card}: no draw in {MAX_TRIES} ({causes})")


def package_members(site: dict) -> dict:
    """{member: bytes} of the run's site folder: site.json, the work order and the public register (the cell
    synthesises the station members -- log, configuration record, advisory log -- from site.json)."""
    dumps = lambda o: (json.dumps(o, indent=1, sort_keys=True, ensure_ascii=True) + "\n").encode("ascii")
    return {"site.json": dumps(site), "mission/work_order.txt": WORK_ORDERS[site["card"]].encode("ascii"),
            "mission/register.json": dumps(public_register(site["register"]))}


def write_package(site: dict, folder: str) -> str:
    for name, data in package_members(site).items():
        p = os.path.join(folder, name)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "wb") as fh:
            fh.write(data)
    return folder


def layout_row(site: dict) -> dict:
    """The items' layout of a site (the fields compared against the trial sites and the hidden candidates)."""
    B = site.get("blocker") or None
    return dict(scene=site.get("scene"), seed=site.get("seed"),
                blocker=None if not B else dict(inlet=B["inlet"], family=B["family"], size=[round(x, 9) for x in B["size"]],
                                                theta_deg=round(B["theta_deg"], 9), offset_m=[round(x, 9) for x in B["offset_m"]]),
                pouches=[dict(inlet=P["inlet"], size=[round(x, 9) for x in P["size"]], columns=P.get("columns"), end=P.get("end"))
                         for P in site.get("small_items") or []],
                r_face={k: round(float(u["r_face"]), 9) for k, u in site["cabin_air_units"].items()})
