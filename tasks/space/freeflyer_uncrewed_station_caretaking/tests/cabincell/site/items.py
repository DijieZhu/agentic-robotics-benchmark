"""The loose items of a site (the station's items and their families): three soft-sheet families (the
blocker and the register's sheet-like entries) and the tool pouch.

Soft sheet = 5 hinged strips (``spacekit.fixtures.soft_item(loop="edge_mid")``, hinge 0.002 N m/rad, 4 mm thick), a
carry strap on the middle strip and the handle loop at the middle of the end strip's outer edge (both tiers keep the
loop).  Each family has its own corner of the family range (0.28-0.30 x 0.26-0.30 m, 0.08-0.14 kg; one corner per
family, the station library's kinds) and its registered colours; the family key is the register class key of station_site/2
(``blocker.family``), so a site record names the family (the look picks the fabric by it).  Tool pouch = a rigid box 0.18-0.22 x
0.11-0.14 x 0.02-0.03 m, 0.06-0.15 kg, a 70 mm handle along the long side in the middle of its top face
(``spacekit.fixtures.small_item``).

Colours are names of ``station.sites.COLOURS`` (the builder's RGBA table).  The family ranges are design choices of
this package inside the site_spec ranges.  numpy only (spacekit is not needed here).
"""
from __future__ import annotations

import numpy as np

from . import spec as SPEC

SHEET_THICK = 0.004
N_STRIPS = 5
HINGE_K = 0.002

# [C] per-family corners of the site_spec family range (world.blocker.family); colours from station.sites.COLOURS
SHEET_FAMILIES = {
    "soft_stowage_sack": dict(register_class="soft stowage sack", size_x=(0.29, 0.30), size_y=(0.27, 0.29),
                              mass=(0.10, 0.13), colours=("orange", "beige", "grey"), nominal=(0.30, 0.28, 0.12)),
    "equipment_cover": dict(register_class="equipment cover", size_x=(0.28, 0.29), size_y=(0.26, 0.28),
                            mass=(0.08, 0.10), colours=("teal", "grey", "green"), nominal=(0.28, 0.26, 0.08)),
    "crew_towel": dict(register_class="crew towel", size_x=(0.29, 0.30), size_y=(0.28, 0.30), mass=(0.12, 0.14),
                       colours=("purple", "yellow", "blue"), nominal=(0.30, 0.30, 0.14)),
}
FAMILY_ORDER = ("soft_stowage_sack", "equipment_cover", "crew_towel")
POUCH = dict(register_class="tool pouch", size=((0.18, 0.22), (0.11, 0.14), (0.02, 0.03)), mass=(0.06, 0.15),
             handle_len=0.07, colours=("blue", "green", "yellow", "grey"))
# the three pouch sizes of the L rule's examples (site_spec qc.l_rule_examples_m2)
POUCH_SIZES = {"smallest": dict(size=(0.18, 0.11, 0.02), mass=0.06), "reference": dict(size=(0.20, 0.12, 0.025), mass=0.10),
               "largest": dict(size=(0.22, 0.14, 0.03), mass=0.15)}


def register_class(family: str) -> str:
    """Register class name of an item family ('tool_pouch' -> 'tool pouch')."""
    if family == "tool_pouch":
        return POUCH["register_class"]
    return SHEET_FAMILIES[family]["register_class"]


def family_of_class(cls: str) -> str:
    for k, F in SHEET_FAMILIES.items():
        if F["register_class"] == cls:
            return k
    if cls == POUCH["register_class"]:
        return "tool_pouch"
    raise KeyError(cls)


def _u(rng, lo, hi) -> float:
    return float(rng.uniform(lo, hi))


def draw_sheet(rng, family: str | None = None, colour: str | None = None) -> dict:
    """One soft sheet: family (uniform over the three unless given), size, mass and a registered colour of it."""
    fam = family or FAMILY_ORDER[int(rng.integers(len(FAMILY_ORDER)))]
    F = SHEET_FAMILIES[fam]
    lx, ly, m = _u(rng, *F["size_x"]), _u(rng, *F["size_y"]), _u(rng, *F["mass"])
    col = colour or F["colours"][int(rng.integers(len(F["colours"])))]
    if col not in F["colours"]:
        raise ValueError(f"colour {col!r} is not a registered colour of {fam}")
    return dict(family=fam, size=[lx, ly, SHEET_THICK], mass=m, colour=col)


def nominal_sheet(family: str = "soft_stowage_sack", colour: str | None = None) -> dict:
    F = SHEET_FAMILIES[family]
    lx, ly, m = F["nominal"]
    return dict(family=family, size=[lx, ly, SHEET_THICK], mass=m, colour=colour or F["colours"][0])


def draw_pouch(rng, colour: str | None = None) -> dict:
    """One tool pouch: size and mass uniform in the family ranges, a registered colour."""
    (x0, x1), (y0, y1), (z0, z1) = POUCH["size"]
    col = colour or POUCH["colours"][int(rng.integers(len(POUCH["colours"])))]
    return dict(family="tool_pouch", size=[_u(rng, x0, x1), _u(rng, y0, y1), _u(rng, z0, z1)], mass=_u(rng, *POUCH["mass"]),
                colour=col)


def sized_pouch(which: str = "reference", colour: str = "blue") -> dict:
    P = POUCH_SIZES[which]
    return dict(family="tool_pouch", size=list(P["size"]), mass=P["mass"], colour=colour)


def blocker_entry(inlet: str, sheet: dict, theta_deg: float, offset_m, register_ref: str) -> dict:
    """The station_site/2 ``blocker`` entry (site.schema.json $defs.blocker)."""
    return dict(inlet=inlet, family=sheet["family"], size=[float(v) for v in sheet["size"]], mass=float(sheet["mass"]),
                n_strips=N_STRIPS, hinge_k=HINGE_K, theta_deg=float(theta_deg) % 360.0, offset_m=[float(v) for v in offset_m],
                loop="edge_mid", strap=True, colour=sheet["colour"], register_ref=register_ref)


def pouch_entry(pid: str, inlet: str, pouch: dict, offset_m, end: int, columns: int, register_ref: str | None,
                theta_deg: float = 0.0) -> dict:
    """The station_site/2 ``small_items[]`` entry (site.schema.json $defs.pouch)."""
    return dict(id=pid, inlet=inlet, size=[float(v) for v in pouch["size"]], mass=float(pouch["mass"]), theta_deg=float(theta_deg),
                offset_m=[float(v) for v in offset_m], end=int(end), columns=int(columns), colour=pouch["colour"],
                register_ref=register_ref)


def check_against_spec(values: dict | None = None) -> list:
    """Problems of the family table against the site_spec ranges (empty = consistent)."""
    v = values or SPEC.values()
    fam = v["world.blocker.family"]
    (sx0, sx1), (sy0, sy1) = fam["size_m"]
    m0, m1 = fam["mass_kg"]
    out = []
    if fam["kinds"] != len(SHEET_FAMILIES):
        out.append(f"kinds {fam['kinds']} != {len(SHEET_FAMILIES)}")
    if fam["strips"] != N_STRIPS or abs(fam["hinge_k_Nm_per_rad"] - HINGE_K) > 1e-12 or fam["loop"] != "edge_mid":
        out.append("strips / hinge / loop differ")
    for k, F in SHEET_FAMILIES.items():
        if not (sx0 <= F["size_x"][0] and F["size_x"][1] <= sx1 and sy0 <= F["size_y"][0] and F["size_y"][1] <= sy1):
            out.append(f"{k}: size outside the family range")
        if not (m0 <= F["mass"][0] and F["mass"][1] <= m1):
            out.append(f"{k}: mass outside the family range")
    classes = v["world.register.classes"]
    if sorted(F["register_class"] for F in SHEET_FAMILIES.values()) != sorted(classes["sheet"]):
        out.append("sheet classes differ from world.register.classes")
    if [POUCH["register_class"]] != list(classes["pouch"]):
        out.append("pouch class differs")
    pf = v["world.pouch.family"]
    if [list(r) for r in POUCH["size"]] != [list(r) for r in pf["size_m"]] or list(POUCH["mass"]) != list(pf["mass_kg"]):
        out.append("pouch ranges differ from world.pouch.family")
    if abs(POUCH["handle_len"] * 1000 - pf["handle_mm"]) > 1e-9:
        out.append("pouch handle length differs")
    return out


def colour_names() -> set:
    s = set(POUCH["colours"])
    for F in SHEET_FAMILIES.values():
        s |= set(F["colours"])
    return s


def footprint_area(item: dict) -> float:
    """Own projected footprint of an item lying flat [m^2] (size x by size y)."""
    return float(item["size"][0]) * float(item["size"][1])


def as_array(x) -> np.ndarray:
    return np.asarray(x, float)
