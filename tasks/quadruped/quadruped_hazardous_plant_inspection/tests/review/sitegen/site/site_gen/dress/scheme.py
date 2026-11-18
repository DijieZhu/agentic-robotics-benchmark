"""The material table of the dressing: what the generator's surfaces and equipment become, by room class and the
site's seed, and what is never touched."""

from __future__ import annotations

import numpy as np

# ------------------------------------------------------------------------------------------------ never touched
# instruments, tag plates, room plates, grid markers, hazard markings, the pad, every generated sign, traversal-element
# markings, glass, lamps of panels, the robot, and the dressing itself
PROTECT_NAME = ("ip/", "inst/", "marker/", "sign/", "tag", "floor/ip", "station/", "robot/", "dress/", "eq/EQ001")
PROTECT_MAT_PREFIX = ("ip/", "inst/", "bricks/", "sign/", "tag", "floor/ip", "legend", "dial", "level_scale",
                      "gate_indicator", "room_", "haz_")
PROTECT_MAT = {"pad_paint", "hazard_stripe", "zone_tape", "cable_protector", "walkway_yellow", "walkway_green",
               "led_green", "led_off", "glass", "needle", "liquid", "placeholder", "placeholder_face", "lamp_red_on",
               "lamp_green_on", "lamp_amber_on", "lamp_white_on", "lamp_off", "screen", "steel_plate", "ramp_steel",
               "grating", "puddle", "seam"}


def is_protected(name: str, mat: str) -> bool:
    if name.startswith(PROTECT_NAME):
        return True
    if not mat:
        return False
    return mat in PROTECT_MAT or mat.startswith(PROTECT_MAT_PREFIX)


def thermal_geoms(site: dict) -> set:
    """Geoms the thermal camera reads (temperature units, hot faces): never hidden by a swap."""
    out = set()
    for g in site.get("thermal_groups", []) or []:
        for u in g.get("units", []) or []:
            if u.get("geom"):
                out.add(u["geom"])
    for key in ("ips", "decoys"):
        for r in site.get(key, []) or []:
            tr = r.get("truth") or {}
            for u in tr.get("units") or []:
                if u.get("geom"):
                    out.add(u["geom"])
            for gname in (tr.get("geom"), (r.get("params") or {}).get("hot_geom")):
                if gname:
                    out.add(gname)
    return out


# kept clear by new decor (0.3 m, 0.6 m for large or saturated items): instrument faces and bodies, stations, tag
# plates, room plates and signs, grid markers, the survey mark, the pad, the sealed doors' straps, hazard stripes
KEEPCLEAR_NAME = ("ip/", "inst/", "marker/", "sign/", "tag", "floor/ip", "station/", "eq/EQ001")
KEEPCLEAR_MAT = {"pad_paint", "hazard_stripe", "zone_tape", "cable_protector"}


def is_keepclear(name: str, mat: str) -> bool:
    if name.startswith(KEEPCLEAR_NAME) or ("/seal" in name and name.startswith("door/")):
        return True
    return bool(mat) and (mat in KEEPCLEAR_MAT or mat.startswith(("ip/", "inst/", "sign/", "tag", "floor/ip", "legend",
                                                                   "dial", "level_scale", "gate_indicator")))


def structure_kind(name: str, mat: str) -> str:
    if mat == "wall_paint" and name.startswith("wall/"):
        return "wall"
    if mat == "block_wall":
        return "partition"
    if mat == "shield_concrete":
        return "shield"
    if name.startswith("floor/") and mat.startswith("floor_"):
        return "floor"
    if name.startswith("roof/") and mat == "roof":
        return "ceiling"
    if mat == "skirting":
        return "skirting"
    if mat in ("lamp_on", "lamp_dim") and "lamp" in name:
        return "lamp"
    return "equipment"


# ------------------------------------------------------------------------------------------------ the site's scheme
PROCESS = ("pump_room", "valve_gallery", "tank_room", "electrical_room", "hvac_room")
CIRCULATION = ("corridor", "airlock")
OFFICE = ("office", "sample_lab", "change_room", "decon_room")
SERVICE = ("store", "workshop", "waste_store", "pipe_chase", "annex")
DADO_RGBA = {"green": (1, 1, 1, 1), "blue": (1, 1, 1, 1), "grey": (1, 1, 1, 1)}
EPOXY = {"green": (0.58, 0.74, 0.62, 1.0), "grey": (0.80, 0.82, 0.82, 1.0), "blue": (0.62, 0.70, 0.78, 1.0),
         "sand": (0.82, 0.78, 0.70, 1.0)}


def room_class(room) -> str:
    if room is None:
        return "out"
    k = room.get("kind") or room.get("type")
    t = room.get("type")
    for c in (k, t):
        if c in PROCESS:
            return "process"
        if c in CIRCULATION:
            return "circulation"
        if c in OFFICE:
            return "office"
        if c in SERVICE:
            return "service"
    return "service"


def site_scheme(site: dict, rng) -> dict:
    """Per-site choices (one stream: the site's seed): the dado colour per room class and the floor tints."""
    base = "green" if rng.random() < 0.5 else "blue"
    other = "blue" if base == "green" else "green"
    dado = {"process": base, "circulation": base if rng.random() < 0.6 else other,
            "office": other if rng.random() < 0.6 else "grey", "service": "grey" if rng.random() < 0.7 else base,
            "out": "grey"}
    epoxy_process = ["green", "grey", "blue"][int(rng.integers(3))]
    floors = {"pump_room": "green" if rng.random() < 0.7 else epoxy_process, "process": epoxy_process}
    tile_tint = (0.92, 0.91, 0.88, 1.0) if rng.random() < 0.5 else (0.86, 0.87, 0.86, 1.0)
    per_room = {}
    for r in site.get("rooms", []):
        rc = room_class(r)
        d = dado[rc]
        # a few rooms of a class repaint in the other colour (an extension, a refurbished office)
        if rc in ("process", "office") and rng.random() < 0.15:
            d = other if d != other else base
        per_room[r["id"]] = {"class": rc, "dado": d}
    summary = {"dado": dado, "epoxy": floors, "tile_tint": tile_tint}
    return {"dado": dado, "floors": floors, "tile_tint": tile_tint, "rooms": per_room, "summary": summary}


def room_index(site: dict):
    out = []
    for r in site.get("rooms", []):
        x0, y0, x1, y1 = r["rect"]
        out.append((min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1), r))
    return out


def room_at(rooms, p):
    x, y = float(p[0]), float(p[1])
    best = None
    for x0, y0, x1, y1, r in rooms:
        if x0 <= x <= x1 and y0 <= y <= y1:
            a = (x1 - x0) * (y1 - y0)
            if best is None or a < best[0]:
                best = (a, r)
    return None if best is None else best[1]


def room_of_geom(rooms, name: str):
    parts = name.split("/")
    rid = parts[1] if len(parts) > 1 else None
    for *_, r in rooms:
        if r["id"] == rid:
            return r
    return None


def wall_finish(plan, kind, room):
    if kind == "partition":
        return "blockwork", "blockwork", (0.95, 0.95, 0.93, 1)
    if kind == "shield":
        return "shield", "shield_concrete", (1, 1, 1, 1)
    if room is None:
        return "outer", "shield_concrete", (0.95, 0.95, 0.95, 1)
    pr = plan["rooms"].get(room["id"]) or {"dado": plan["dado"]["service"]}
    if (room.get("type") or room.get("kind")) == "pipe_chase":
        return "chase", "shield_concrete", (0.92, 0.92, 0.92, 1)
    d = pr["dado"]
    return "wall_" + d, "wall_dado_" + d, DADO_RGBA[d]


def floor_finish(plan, room):
    """(material key, bundle item, rgba, (specular, shininess))."""
    t = (room or {}).get("kind") or (room or {}).get("type")
    rt = (room or {}).get("type")
    if rt in CIRCULATION or t in OFFICE or rt in OFFICE:
        return "floor_tile", "floor_tile", plan["tile_tint"], (0.25, 0.35)
    if rt == "pump_room":
        c = plan["floors"]["pump_room"]
        return "floor_epoxy_" + c, "floor_epoxy", EPOXY[c], (0.35, 0.45)
    if rt in PROCESS or t in PROCESS:
        c = plan["floors"]["process"]
        return "floor_epoxy_" + c, "floor_epoxy", EPOXY[c], (0.35, 0.45)
    return "floor_concrete", "floor_concrete", (0.95, 0.95, 0.94, 1), (0.08, 0.1)


# ------------------------------------------------------------------------------------------------ equipment
# generator material -> (bundle item, rgba, specular, shininess, tile m); None keeps the generator's look.
# Signal yellow (the tag plate's hue, sRGB 245,196,0) is reserved: nothing above the floor but tag plates carries it,
# so the generator's yellows (actuators, radwaste drums, the radwaste pipe) take other colours here.
EQUIP = {
    "concrete_plinth": ("plinth_concrete", (0.92, 0.92, 0.9, 1), 0.05, 0.1, 1.5),
    "steel": ("steel_worn", (0.95, 0.95, 0.95, 1), 0.35, 0.35, 0.8),
    "steel_dark": ("steel_worn", (0.48, 0.49, 0.51, 1), 0.3, 0.3, 0.8),
    "galv": ("metal_galv", (1.0, 1.0, 1.0, 1), 0.3, 0.3, 0.8),
    "tray": ("metal_galv", (0.95, 0.95, 0.95, 1), 0.3, 0.3, 0.8),
    "motor_blue": ("metal_white", (0.38, 0.49, 0.62, 1), 0.3, 0.3, 0.9),
    "motor_blue_dark": ("metal_white", (0.26, 0.33, 0.42, 1), 0.3, 0.3, 0.9),
    "pump_casing": ("metal_white", (0.40, 0.49, 0.60, 1), 0.3, 0.3, 0.9),
    "pump_blue": ("metal_white", (0.36, 0.47, 0.62, 1), 0.3, 0.3, 0.9),
    "guard_orange": ("guard_mesh", (0.80, 0.42, 0.24, 1), 0.25, 0.25, 0.5),
    "actuator_yellow": ("metal_white", (0.74, 0.44, 0.26, 1), 0.25, 0.25, 0.9),
    "tank_grey": ("tank_paint", (1.0, 1.0, 0.98, 1), 0.4, 0.5, 2.5),
    "cabinet": ("metal_white", (0.84, 0.85, 0.82, 1), 0.2, 0.2, 0.9),
    "cabinet_door": ("metal_white", (0.80, 0.81, 0.78, 1), 0.2, 0.2, 0.9),
    "cabinet_white": ("metal_white", (0.97, 0.97, 0.95, 1), 0.2, 0.2, 0.9),
    "panel_grey": ("metal_white", (0.80, 0.82, 0.82, 1), 0.2, 0.2, 0.9),
    "base_station": ("metal_white", (0.30, 0.35, 0.40, 1), 0.25, 0.25, 0.9),
    "locker_blue": ("metal_white", (0.48, 0.55, 0.64, 1), 0.25, 0.25, 0.9),
    "lab_bench": ("metal_white", (0.88, 0.88, 0.86, 1), 0.2, 0.2, 0.9),
    "rack_blue": ("metal_white", (0.32, 0.42, 0.60, 1), 0.25, 0.25, 0.9),
    "rack_orange": ("metal_orange", (0.84, 0.75, 0.68, 1), 0.25, 0.25, 0.8),
    "box_grey": ("metal_galv", (0.85, 0.85, 0.85, 1), 0.2, 0.2, 0.8),
    "box_blue": ("metal_white", (0.36, 0.45, 0.60, 1), 0.2, 0.2, 0.9),
    "drum_small": ("metal_white", (0.40, 0.52, 0.42, 1), 0.3, 0.3, 0.9),
    "drum_yellow": ("metal_galv", (0.92, 0.92, 0.90, 1), 0.35, 0.35, 0.8),
    "drum_lid": ("steel_worn", (0.55, 0.55, 0.56, 1), 0.3, 0.3, 0.8),
    "drum_band": ("steel_worn", (0.40, 0.40, 0.41, 1), 0.3, 0.3, 0.8),
    "ahu_grey": ("metal_white", (0.80, 0.82, 0.82, 1), 0.2, 0.2, 0.9),
    "ahu_panel": ("metal_white", (0.72, 0.74, 0.74, 1), 0.2, 0.2, 0.9),
    "machine_green": ("metal_white", (0.46, 0.56, 0.49, 1), 0.25, 0.25, 0.9),
    "gas_green": ("metal_white", (0.32, 0.48, 0.36, 1), 0.3, 0.3, 0.9),
    "red_fire": ("metal_white", (0.60, 0.28, 0.23, 1), 0.3, 0.3, 0.9),
    "rubber_black": ("rubber", (0.9, 0.9, 0.9, 1), 0.05, 0.05, 0.6),
    "rubber_blue": ("metal_white", (0.30, 0.42, 0.70, 1), 0.05, 0.05, 0.9),
    "hose_black": ("rubber", (0.8, 0.8, 0.8, 1), 0.1, 0.1, 0.6),
    "hose_red": ("rubber", (1.0, 0.45, 0.40, 1), 0.1, 0.1, 0.6),
    "door_leaf": ("metal_white", (0.52, 0.58, 0.64, 1), 0.25, 0.25, 0.9),
    "door_leaf_fire": ("metal_white", (0.58, 0.34, 0.29, 1), 0.25, 0.25, 0.9),
    "door_leaf_heavy": ("metal_white", (0.44, 0.46, 0.50, 1), 0.25, 0.25, 0.9),
    "door_frame": ("steel_worn", (0.42, 0.43, 0.46, 1), 0.3, 0.3, 0.8),
    "door_frame_fire": ("steel_worn", (0.54, 0.38, 0.35, 1), 0.3, 0.3, 0.8),
    # pipes: the old white paint in the service colours (cooling water green, demineralised water blue-green,
    # radwaste ochre-brown (not the reserved yellow), instrument air light blue, fire water red, oil brown), steam
    # lagged in aluminium cladding, drains dark steel
    "pipe_cw": ("paint_white", (0.36, 0.56, 0.41, 1), 0.3, 0.35, 1.2),
    "pipe_dw": ("paint_white", (0.22, 0.52, 0.58, 1), 0.3, 0.35, 1.2),
    "pipe_st": ("lagging", (1.0, 1.0, 1.0, 1), 0.45, 0.5, 1.0),
    "pipe_rw": ("paint_white", (0.68, 0.50, 0.26, 1), 0.3, 0.35, 1.2),
    "pipe_ia": ("paint_white", (0.56, 0.72, 0.86, 1), 0.3, 0.35, 1.2),
    "pipe_fw": ("paint_white", (0.68, 0.22, 0.17, 1), 0.3, 0.35, 1.2),
    "pipe_dr": ("steel_worn", (0.34, 0.32, 0.30, 1), 0.25, 0.25, 0.8),
    "pipe_oil": ("paint_white", (0.50, 0.34, 0.22, 1), 0.3, 0.35, 1.2),
    "bund_floor": ("floor_epoxy", (0.66, 0.72, 0.76, 1), 0.35, 0.45, 3.0),
    "floor_base": ("floor_concrete", (0.85, 0.85, 0.84, 1), 0.05, 0.05, 3.0),
}


def equipment_material(ctx, mat: str, name: str, plan) -> str | None:
    rec = EQUIP.get(mat)
    if rec is None:
        return None
    item, rgba, sp, sh, tile = rec
    tex = ctx.texture(item)
    if mat == "floor_base":
        rep = (1.0 / tile, 1.0 / tile)
    else:
        rep = (2.0 / tile, 2.0 / tile)
    return ctx.material("eq/" + mat, tex, rgba=rgba, specular=sp, shininess=sh, repeat=rep, uniform=True)


def box_material(ctx, kind: str) -> str | None:
    """The structure box itself (its faces are skinned; what shows is the reveal edges): plain light paint."""
    if kind == "wall":
        return ctx.material("box/wall", ctx.texture("wall_paint_plain"), rgba=(0.97, 0.97, 0.95, 1), specular=0.1,
                            shininess=0.1, repeat=(1, 1), uniform=False)
    if kind == "partition":
        return ctx.material("box/partition", ctx.texture("blockwork"), rgba=(0.95, 0.95, 0.93, 1), specular=0.05,
                            shininess=0.1, repeat=(1, 1), uniform=False)
    if kind == "shield":
        return ctx.material("box/shield", ctx.texture("shield_concrete"), rgba=(1, 1, 1, 1), specular=0.05,
                            shininess=0.1, repeat=(1, 1), uniform=False)
    return None


# the classic renderer's light (the frames read as flat fill): less headlight fill, the room lights as
# spots with a visible pool and falloff (exponent, attenuation), brighter so that white paint reads white
LOOK = {"headlight_ambient": 0.15, "headlight_diffuse": 0.40, "shadowsize": 2048, "shadow_lights": "none",
        "spot_exponent": 2.5, "spot_cutoff": 85.0, "spot_attenuation": (0.25, 0.08, 0.10), "spot_gain": 2.0,
        "wall_emission": 0.10}


def render_settings(spec, plan, site=None, look=None):
    """Classic-renderer settings of the dressed scene (no physics): a lower headlight fill so that the room lights
    shape the rooms, and shadows from one room light per room (the one nearest the room's centre; the cell activates
    the room lights nearest its camera, so a frame has one to three shadow casters). Off by default (a bench:
    navigation frame costs about five times as much); env SITE_GEN_DRESS_SHADOWS=1 turns them on."""
    import os
    L = dict(LOOK)
    L.update(look or {})
    spec.visual.headlight.ambient = [L["headlight_ambient"]] * 3
    spec.visual.headlight.diffuse = [L["headlight_diffuse"]] * 3
    spec.visual.quality.shadowsize = int(L["shadowsize"])
    out = {"shadow_lights": [], "room_lights": 0}
    for l in spec.lights:
        nm = l.name or ""
        if not nm.startswith("light/") or nm.startswith("light/fill"):
            continue
        l.exponent = float(L["spot_exponent"])
        l.cutoff = float(L["spot_cutoff"])
        l.attenuation = [float(v) for v in L["spot_attenuation"]]
        # light colours above 1 are allowed (fixed-function lighting clamps the lit colour, not the light): a lamp's
        # pool reaches white on white paint while its falloff stays
        l.diffuse = [float(v) * float(L["spot_gain"]) for v in l.diffuse]
        out["room_lights"] += 1
    sh = os.environ.get("SITE_GEN_DRESS_SHADOWS")
    if sh == "1":
        L["shadow_lights"] = "one_per_room"
    if sh == "0" or L["shadow_lights"] in (None, "none"):
        return out
    centres = {}
    for r in (site or {}).get("rooms", []):
        x0, y0, x1, y1 = r["rect"]
        centres[r["id"]] = ((x0 + x1) / 2, (y0 + y1) / 2)
    best = {}
    for l in spec.lights:
        nm = l.name or ""
        parts = nm.split("/")
        if len(parts) != 3 or parts[0] != "light" or parts[1].startswith("fill"):
            continue
        c = centres.get(parts[1])
        if c is None:
            continue
        d = (l.pos[0] - c[0]) ** 2 + (l.pos[1] - c[1]) ** 2
        if parts[1] not in best or d < best[parts[1]][0]:
            best[parts[1]] = (d, l)
    for rid in sorted(best):
        best[rid][1].castshadow = True
        out["shadow_lights"].append(best[rid][1].name)
    return out


__all__ = ["is_protected", "structure_kind", "site_scheme", "room_index", "room_at", "room_of_geom", "wall_finish",
           "floor_finish", "equipment_material", "box_material", "render_settings", "np"]
