"""The lost-item register of a site (site_spec ``world.register*``).

9-12 entries.  Present (``present`` true, ``item`` = the site item): the blocker's entry and the entry of every
registered pouch; 3-4 present entries in all (``items_in_station``), so a site with 2 pouches has no unregistered
item and a site with 3 has 0 or 1 (``unregistered_small``).  Sheet-like entries with a carry strap: the blocker's and
2-3 absent ones, each with its own (class, colour) pair (``strap_sheet_entries_min`` 3; only the blocker's entry
is in the station), all "carry strap + handle loop" (the blocker has both, so the feature list cannot single it out).
The rest: absent tool pouches and absent other items.  Sizes and masses of absent sheets and pouches are drawn from
the families (``cabincell.site.items``); absent other items from a small table.  ``location`` is the register's
last-seen text: a module and a spot drawn independently of the truth (the register is the crew's report, not the
current state).  Entries are shuffled and numbered LI-01 .. .
"""
from __future__ import annotations

import numpy as np

from cabincell.site import items as IT
from cabincell.site import spec as SPEC

STRAP_SHEET_FEATURES = "carry strap + handle loop"
SPOTS = ("equipment rack face", "hatch area", "near a crew handrail", "cabin centre")                     # [C]
OTHER_ITEMS = (dict(size=(0.15, 0.10, 0.03), mass=(0.08, 0.15), features="handle"),                     # [C]
               dict(size=(0.12, 0.08, 0.05), mass=(0.10, 0.25), features="none"),
               dict(size=(0.20, 0.15, 0.04), mass=(0.10, 0.20), features="handle"))
OTHER_COLOURS = ("grey", "beige", "blue", "green")                                                      # [C]


def _u(rng, lo, hi) -> float:
    return float(rng.uniform(lo, hi))


def _loc(rng) -> str:
    m = ("L2", "H2", "L1")[int(rng.integers(3))]
    return f"{m}, {SPOTS[int(rng.integers(len(SPOTS)))]}"


def _r(x, nd=3):
    return float(round(float(x), nd))


def build(rng, site: dict, values=None) -> dict:
    """{'entries': [...], 'unregistered': [...], 'refs': {item: ref}} for a site whose blocker and small items are
    drawn (their ``register_ref`` is set by the caller from ``refs``)."""
    v = values or SPEC.values()
    W = v["world.register"]
    pouches = list(site.get("small_items") or [])
    B = site["blocker"]
    n_p = len(pouches)
    # unregistered pouch: present registered entries must stay in items_in_station
    lo_in, hi_in = (int(x) for x in W["items_in_station"])
    lo_un, hi_un = (int(x) for x in W["unregistered_small"])
    options = [u for u in range(lo_un, hi_un + 1) if lo_in <= 1 + n_p - u <= hi_in and u <= n_p]
    if not options:
        raise ValueError(f"no unregistered count gives {lo_in}-{hi_in} present entries with {n_p} pouches")
    n_un = options[int(rng.integers(len(options)))]
    unreg = []
    if n_un:
        unreg = [pouches[int(rng.integers(n_p))]["id"]]
    present = []
    present.append(dict(cls=IT.register_class(B["family"]), colour=B["colour"], size_m=[_r(x) for x in B["size"]],
                        mass_kg=_r(B["mass"]), features=STRAP_SHEET_FEATURES, item="blocker"))
    for P in pouches:
        if P["id"] in unreg:
            continue
        present.append(dict(cls=IT.register_class("tool_pouch"), colour=P["colour"], size_m=[_r(x) for x in P["size"]],
                            mass_kg=_r(P["mass"]), features="handle", item=P["id"]))
    n_tot = int(rng.integers(int(W["entries"][0]), int(W["entries"][1]) + 1))
    # absent sheets with a carry strap: (family, colour) pairs other than the blocker's
    need = int(W["strap_sheet_entries_min"]) - 1
    n_sheet = need + int(rng.integers(0, 2))                       # 2-3 absent strap sheets
    pairs = [(f, c) for f in IT.FAMILY_ORDER for c in IT.SHEET_FAMILIES[f]["colours"] if (f, c) != (B["family"], B["colour"])]
    order = rng.permutation(len(pairs))
    absent = []
    for idx in order[:n_sheet]:
        f, c = pairs[int(idx)]
        sh = IT.draw_sheet(rng, family=f, colour=c)
        absent.append(dict(cls=IT.register_class(f), colour=c, size_m=[_r(x) for x in sh["size"]], mass_kg=_r(sh["mass"]),
                           features=STRAP_SHEET_FEATURES, item=None))
    while len(present) + len(absent) < n_tot:
        if rng.uniform() < 0.5:
            P = IT.draw_pouch(rng)
            absent.append(dict(cls=IT.register_class("tool_pouch"), colour=P["colour"], size_m=[_r(x) for x in P["size"]],
                               mass_kg=_r(P["mass"]), features="handle", item=None))
        else:
            O = OTHER_ITEMS[int(rng.integers(len(OTHER_ITEMS)))]
            absent.append(dict(cls="other", colour=OTHER_COLOURS[int(rng.integers(len(OTHER_COLOURS)))],
                               size_m=[_r(x) for x in O["size"]], mass_kg=_r(_u(rng, *O["mass"])), features=O["features"], item=None))
    allx = present + absent
    perm = rng.permutation(len(allx))
    entries, refs = [], {}
    for n, idx in enumerate(perm):
        e = allx[int(idx)]
        ref = f"LI-{n + 1:02d}"
        entries.append({"ref": ref, "class": e["cls"], "colour": e["colour"], "size_m": e["size_m"], "mass_kg": e["mass_kg"],
                        "features": e["features"], "location": _loc(rng), "present": e["item"] is not None, "item": e["item"]})
        if e["item"] is not None:
            refs[e["item"]] = ref
    return dict(entries=entries, unregistered=unreg, refs=refs)


def public(register: dict) -> list:
    """mission/register.json: the entries as the program sees them (no ``present``, no ``item``; the robot API's mission)."""
    keys = ("ref", "class", "colour", "size_m", "mass_kg", "features", "location")
    return [{k: e[k] for k in keys} for e in register["entries"]]


def check(site: dict, values=None) -> list:
    """Problems of a site's register against world.register and the site's items (empty = consistent)."""
    v = values or SPEC.values()
    W = v["world.register"]
    R = site["register"]
    E = R["entries"]
    out = []
    if not (int(W["entries"][0]) <= len(E) <= int(W["entries"][1])):
        out.append(f"{len(E)} entries")
    refs = [e["ref"] for e in E]
    if len(set(refs)) != len(refs) or refs != [f"LI-{n + 1:02d}" for n in range(len(E))]:
        out.append("refs not LI-01..")
    pres = [e for e in E if e["present"]]
    if not (int(W["items_in_station"][0]) <= len(pres) <= int(W["items_in_station"][1])):
        out.append(f"{len(pres)} present entries")
    if not (int(W["unregistered_small"][0]) <= len(R["unregistered"]) <= int(W["unregistered_small"][1])):
        out.append("unregistered count")
    sheet_cls = set(v["world.register.classes"]["sheet"])
    straps = [e for e in E if e["class"] in sheet_cls and "carry strap" in e["features"]]
    if len(straps) < int(W["strap_sheet_entries_min"]):
        out.append(f"{len(straps)} strap sheet entries")
    if len({(e["class"], e["colour"]) for e in straps}) != len(straps):
        out.append("two strap sheet entries share class and colour")
    pres_sheets = [e for e in straps if e["present"]]
    if len(pres_sheets) != 1 or pres_sheets[0]["item"] != "blocker":
        out.append("the present strap sheet is not exactly the blocker")
    B = site["blocker"]
    eb = [e for e in E if e["item"] == "blocker"]
    if len(eb) != 1 or eb[0]["ref"] != B["register_ref"] or eb[0]["class"] != IT.register_class(B["family"]) or eb[0]["colour"] != B["colour"]:
        out.append("blocker entry mismatch")
    for P in site.get("small_items") or []:
        ep = [e for e in E if e["item"] == P["id"]]
        if P["id"] in R["unregistered"]:
            if ep or P["register_ref"] is not None:
                out.append(f"{P['id']} unregistered but has an entry")
        elif len(ep) != 1 or ep[0]["ref"] != P["register_ref"] or ep[0]["colour"] != P["colour"]:
            out.append(f"{P['id']} entry mismatch")
    for e in E:
        if e["present"] != (e["item"] is not None):
            out.append(f"{e['ref']}: present / item disagree")
    return out
