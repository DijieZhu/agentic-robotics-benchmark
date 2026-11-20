"""Mapped feature points of the station (the as-surveyed map the localiser matches against:
rack-face 0.25 m grid + handrails / luminaires / vents / hatch doors / return inlets / dock).

``feature_points`` computes them on the neutral catalogue.  The stowage units are not mapped features (they can be moved; their fiducial plates
are the robot's relative references).  ``data/station/map_features.json`` holds the points of both scenes; the
firmware package reads it (lighting factors are its business).
"""
from __future__ import annotations

import numpy as np

FEATURE_DEFAULTS = dict(
    rack_spacing_m=0.25,     # grid of matched landmarks on a rack face (equipment, labels, seat track)  [G]
    rack_margin_m=0.06,
    handrail_points=3,       # ends + centre  [G]
    light_points=3,
    vent_points=2,
    hatch_door_points=5,
    inlet_points=4,
    dock_points=6,
)


def _quat_to_mat(q):
    w, x, y, z = (float(v) for v in q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def feature_points(feats: dict, modules=None, params: dict | None = None) -> dict:
    """dict(P (N,3), normal (N,3), module (N,), kind (N,)): normals point into the cabin (NaN = no facing test)."""
    p = dict(FEATURE_DEFAULTS, **(params or {}))
    modules = tuple(feats["modules"]) if modules is None else tuple(modules)
    P, Nn, Mo, Kd = [], [], [], []

    def add(pt, n, mod, kind):
        P.append(np.asarray(pt, float))
        Nn.append(np.asarray(n, float) if n is not None else np.full(3, np.nan))
        Mo.append(mod)
        Kd.append(kind)
    for r in feats["rack_faces"]:
        if r["module"] not in modules:
            continue
        c, n, a = (np.asarray(r[k], float) for k in ("centre", "normal_into_cabin", "along"))
        up = np.cross(n, a)
        w, h = r["width_along_m"] / 2 - p["rack_margin_m"], r["height_m"] / 2 - p["rack_margin_m"]
        s = p["rack_spacing_m"]
        nu, nv = max(1, int(round(2 * w / s)) + 1), max(1, int(round(2 * h / s)) + 1)
        for u in np.linspace(-w, w, nu):
            for v in np.linspace(-h, h, nv):
                add(c + n * 0.01 + a * u + up * v, n, r["module"], "rack")
    for hr in feats["handrails"]:
        if hr["module"] not in modules:
            continue
        R = _quat_to_mat(hr["quat_wxyz"])
        c = np.asarray(hr["pos"], float)
        L = hr["length"] / 2 - 0.03
        for u in np.linspace(-L, L, p["handrail_points"]):
            add(c + R[:, 0] * u, R[:, 2], hr["module"], "handrail")
    for lt in feats["lights"]:
        if lt["module"] not in modules:
            continue
        c, a = np.asarray(lt["pos"], float), np.asarray(lt["along"], float)
        L = lt["length"] / 2 - 0.03
        for u in np.linspace(-L, L, p["light_points"]):
            add(c + a * u, lt.get("aim"), lt["module"], "light")
    for vt in feats["vents"]:
        if vt["module"] not in modules or vt.get("kind") not in ("supply_diffuser", "return_grille"):
            continue
        R = _quat_to_mat(vt["quat_wxyz"])
        c = np.asarray(vt["pos"], float)
        L = vt["size"][0] / 2 - 0.03
        for u in np.linspace(-L, L, p["vent_points"]):
            add(c + R[:, 0] * u, vt.get("normal_into_cabin"), vt["module"], "vent")
    for ht in feats["hatches"]:
        if ht["module"] not in modules:
            continue
        c = np.asarray(ht["door_centre"], float)
        dsz = np.asarray(ht["door_size_m"], float)
        nd = np.asarray(ht.get("door_dir_from_axis", [0, 0, 1]), float)
        nd = -nd / max(np.linalg.norm(nd), 1e-9)
        e1 = np.cross(nd, [1.0, 0.0, 0.0])
        if np.linalg.norm(e1) < 0.3:
            e1 = np.cross(nd, [0.0, 1.0, 0.0])
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(nd, e1)
        add(c, nd, ht["module"], "hatch")
        for su, sv in ((1, 1), (1, -1), (-1, 1), (-1, -1))[: p["hatch_door_points"] - 1]:
            add(c + e1 * su * 0.35 * dsz[0] + e2 * sv * 0.35 * dsz[1], nd, ht["module"], "hatch")
    for inl in feats.get("v02", {}).get("inlets", []):
        if inl["module"] not in modules:
            continue
        c = np.asarray(inl["face_center"], float)
        t1, b, n = (np.asarray(inl[k], float) for k in ("t1", "b", "normal"))
        for su, sv in ((1, 1), (1, -1), (-1, 1), (-1, -1))[: p["inlet_points"]]:
            add(c + t1 * su * 0.21 + b * sv * 0.07, n, inl["module"], "inlet")
    d = feats["dock"]
    if d.get("module") in modules:
        c = np.asarray(d["pos"], float)
        for k in range(p["dock_points"]):
            add(c + np.array([0.04, (k % 3 - 1) * 0.3, (k // 3 - 0.5) * 0.25]), [1.0, 0.0, 0.0], d["module"], "dock")
    return dict(P=np.array(P), normal=np.array(Nn), module=np.array(Mo), kind=np.array(Kd))


def to_json(F: dict, decimals: int = 4) -> dict:
    """Compact JSON form: point rows [x, y, z, nx, ny, nz] (normals null when NaN) + module / kind per row."""
    rows = []
    for p, n in zip(F["P"], F["normal"]):
        rows.append([round(float(v), decimals) for v in p] + ([None] * 3 if np.isnan(n).any() else [round(float(v), decimals) for v in n]))
    kinds, mods = sorted(set(F["kind"].tolist())), sorted(set(F["module"].tolist()))
    return dict(n=len(rows), kinds=kinds, modules=mods, rows=rows, kind=[kinds.index(k) for k in F["kind"].tolist()],
                module=[mods.index(m) for m in F["module"].tolist()],
                count_by_kind={k: int((F["kind"] == k).sum()) for k in kinds})


def from_json(J: dict) -> dict:
    rows = np.array([[np.nan if v is None else v for v in r] for r in J["rows"]], float)
    return dict(P=rows[:, :3], normal=rows[:, 3:], module=np.array([J["modules"][i] for i in J["module"]]),
                kind=np.array([J["kinds"][i] for i in J["kind"]]))
