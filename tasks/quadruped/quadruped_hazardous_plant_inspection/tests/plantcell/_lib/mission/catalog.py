"""The plant's instrument catalogue (given/plant/catalog/ in the task): public per-type facts a team would have.

Nothing here comes from a site's ground truth. The reader hints are built from a round-sheet item (prior.json) plus
these catalogue constants, which are the bricks package's nominal designs (the dial convention, the lever length, the
gate indicator plate, the level-gauge scale plate, the tag sizes) and the surface emissivity table of the plant's
painted / cast components (the same table the site generator draws the temperature points' surfaces from,
site_gen config ladder.thermal.surfaces; a test checks that the two agree).

`public_catalog()` is what Robot.mission() hands to every program (key "catalog").
"""

from __future__ import annotations

from . import paths

paths.setup()

from bricks import lettering as _lettering  # noqa: E402  (version 2: the plant lettering)
from bricks.targets import gauge as _gauge  # noqa: E402
from bricks.targets import level as _level  # noqa: E402
from bricks.targets import valve as _valve  # noqa: E402

TAG_MARKER_M = 0.060           # version 1's instrument tag plates: a 60 mm AprilTag 36h11 (gone in version 2: the
                               # plates carry their tag in the plant lettering; kept for reading version 1 packages)
TAG_PLATE_M = (0.1908, 0.0792)  # the tag plate (width, height): the same outline in both versions
GRID_MARKER_M = 0.110          # grid-line marker plates (coded, AprilTag 36h11, listed in the drawing)
GRID_PLATE_M = (0.3498, 0.1452)  # the grid marker plate around its marker
LEVER_LEN_M = 0.18             # quarter-turn lever (bricks LeverValveSpec default), painted red
LEVEL_WINDOW_M = 0.44          # draft sites' tubular level gauge (0.56 m between the nozzles); the ladder round sheet
                               # gives each level gauge's window (window_m, 0.40 m)
GATE_STROKE_M = _valve.SCALE_SPAN
LAMP_LENS_D_M = 2 * 0.0115
LAMP_PITCH_M = 0.065
# surface type (round sheet 'surface') -> emissivity of the temperature point's component (painted / coated metal,
# 8-14 um band); the draft sites' pump bearing housings are painted steel
EMISSIVITY = {"painted steel": 0.93, "powder-coated steel": 0.91, "oxidised cast iron": 0.94,
              "bearing housing": 0.93, "cabinet door": 0.93, "default": 0.93}

# the surface types the plant's temperature points are drawn from (the generator's ladder.thermal.surfaces): the
# published table; the draft sites' "bearing housing" / "cabinet door" stay lookups only
PLANT_SURFACES = ("painted steel", "powder-coated steel", "oxidised cast iron")

# round-sheet type -> the report kind of the facade
REPORT_KIND = {"gauge": "instrument", "valve_lever": "instrument", "valve_handwheel": "instrument",
               "level": "instrument", "lamps": "instrument", "thermal": "instrument", "dose_point": "mark",
               "hotspot": "source"}


def is_instrument(item: dict) -> bool:
    return REPORT_KIND.get(item.get("type")) == "instrument"


def emissivity(surface: str | None) -> float:
    return float(EMISSIVITY.get(str(surface or "default").strip().lower(), EMISSIVITY["default"]))


def public_catalog() -> dict:
    """The catalogue as a document (what every program may read)."""
    return {
        # version 2: an instrument's tag plate carries its tag in the plant lettering and nothing else (no
        # coded marker); the grid marker plates stay coded and are listed in the drawing (mission()["grid_markers"])
        "tags": {"tag_plate": tag_plate_facts(),
                 "grid_markers": {"dictionary": "DICT_APRILTAG_36h11", "marker_m": GRID_MARKER_M,
                                  "plate_m": list(GRID_PLATE_M),
                                  "note": "each grid marker plate carries a coded marker; the drawing lists every "
                                          "plate's marker id, position, height and facing"},
                 "note": "each listed instrument carries a tag plate with its tag in the plant lettering ('lettering'); "
                         "an instrument is the listed one only if its plate says so"},
        "lettering": _lettering.lettering_block(),
        "gauge": {"dial_diameter": "0.9 x the case diameter", "sweep_deg": float(_gauge.SWEEP_DEG),
                  "scale": "270 degrees clockwise from the 7:30 position (low end) to 4:30 (high end)"},
        "valve_lever": {"lever_length_m": LEVER_LEN_M, "lever_colour": "red",
                        "reading": "angle between lever and pipe, 0 in line (open), 90 across (closed)"},
        "valve_handwheel": {"indicator": "vertical scale 0-100 %, 0 at the bottom", "stroke_m": GATE_STROKE_M,
                            "plate_m": [_valve.PLATE_W, _valve.PLATE_H]},
        "level": {"scale": "0-100 % on a plate to the right of the glass tube", "plate_width_m": _level.PLATE_W,
                  "default_window_m": LEVEL_WINDOW_M, "window": "per item on the round sheet (window_m)"},
        "lamps": {"lens_diameter_m": LAMP_LENS_D_M, "pitch_m": LAMP_PITCH_M, "order": "left to right as seen"},
        "thermal": {"emissivity_by_surface": {k: EMISSIVITY[k] for k in PLANT_SURFACES},
                    "band": "8-14 um", "reported": "the hottest point of the component in degrees C"},
    }


def tag_plate_facts() -> dict:
    """The tag plate's standard in numbers (metres, sRGB): outline, border line, lettering, grammar (bricks.lettering,
    the one table of the plant lettering)."""
    L = _lettering
    return {"size_m": [float(L.TAG_PLATE_W_M), float(L.TAG_PLATE_H_M)], "thickness_m": float(L.TAG_PLATE_T_M),
            "plate_rgb": list(L.TAG_YELLOW), "lettering_rgb": list(L.INK),
            "border_line": {"outer_edge_inset_m": float(L.BORDER_INSET_M), "width_m": float(L.BORDER_LINE_M)},
            "cap_height_m": float(L.TAG_CAP_M), "pen_m": float(L.TAG_PEN_M), "pitch_m": float(L.TAG_PITCH_M),
            "max_chars": int(L.TAG_MAX_CHARS), "grammar": L.TAG_PATTERN,
            "text": "one line, centred on the plate, at a fixed pitch"}


def reader_hint(item: dict, pipe_axis: str | None = None) -> dict:
    """The bricks reader hint for a round-sheet item, from public information only."""
    t = item["type"]
    if t == "gauge":
        sc = item.get("scale", {})
        lo, hi = sc.get("range", (0.0, 1.0))
        return {"type": "dial_gauge", "range": (float(lo), float(hi)), "unit": sc.get("unit", ""),
                "size_mm": float(sc.get("case_mm", item.get("case_mm", 100))), "sweep_deg": _gauge.SWEEP_DEG}
    if t == "valve_lever":
        return {"type": "lever_valve", "pipe_axis": pipe_axis or "y", "lever_len_m": LEVER_LEN_M, "pipe_od_m": 0.10}
    if t == "valve_handwheel":
        return {"type": "gate_valve", "indicator": "vertical scale 0-100 %, 0 at the bottom",
                "plate_size_m": (_valve.PLATE_W, _valve.PLATE_H)}
    if t == "level":
        L = float(item.get("window_m") or LEVEL_WINDOW_M)
        return {"type": "level_gauge", "range": (0.0, 100.0), "unit": "%", "window_m": L,
                "scale_side": "right", "plate_size_m": (_level.PLATE_W, L + 0.05),
                "tube_offset_m": _level.TUBE_R + _level.PLATE_GAP, "tube_od_m": 2 * _level.TUBE_R}
    if t == "lamps":
        cols = tuple(item.get("colors", ()))
        return {"type": "lamp_panel", "n_lamps": int(item.get("n_lamps", len(cols))), "colors": cols}
    if t == "thermal":
        return {"type": "thermal", "emissivity": emissivity(item.get("surface")), "surface": item.get("surface"),
                "units": list(item.get("units", []))}
    return {"type": t}
