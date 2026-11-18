"""Decor: pack meshes swapped in for the generator's items, new decor on walls and at the ceiling, signs, and the
contact-shadow decals on the floor.

Rules:
  * a swapped mesh replaces a decorative item, or an item whose mesh fits inside its collision geoms (no vertex more
    than 2 cm outside their box: checked); the generator's geoms of the item are hidden (alpha 0), never removed;
  * new decor only on walls (bottom >= 1.0 m) or at the ceiling; large or saturated items (extinguishers, alarm bells,
    bulkhead lamps, signs) at least 0.6 m from every instrument face, plate, sign and marker and outside the
    inspection camera's view at every listed instrument's observation pose; small grey items 0.3 m; nothing in any
    reading region's sight lines;
  * floor decals (contact shadows) are flat, 1.2 mm above the floor skin, unlit and transparent; none on a traversal
    element's footprint or its approach strips (1.0 m around it), on the pad, on a survey mark or a floor marker;
  * every item is a visual-only geom in its own static body (mj_multiRay, which the lidar uses, can miss geoms of
    static bodies with several geoms).
"""

from __future__ import annotations

import math

import mujoco
import numpy as np

from .xmath import dot3, matvec_rows, norm
from .checks import (CLEAR_M, FIT_M, DressError, _box_dist, in_inspect_views, inspect_views, instrument_faces,
                     segments_hit_box, sight_segments)

CYL = int(mujoco.mjtGeom.mjGEOM_CYLINDER)
BOX = int(mujoco.mjtGeom.mjGEOM_BOX)
FLOOR_TOP = 0.0015
DECAL_Z = FLOOR_TOP + 0.0012
BIG_CLEAR_M = 0.6

# per wall item: the native axis that faces away from the wall, the height of its bottom (m), the rooms that get it
# (class: count per 10 m of wall), and its clearance class
WALL_ITEMS = {
    "extinguisher": {"front": "-y", "z0": 1.05, "per": {"circulation": 0.45, "process": 0.3, "service": 0.25},
                     "big": True, "sign": "fire_extinguisher"},
    "fire_alarm": {"front": "-y", "z0": 1.45, "per": {"circulation": 0.3, "office": 0.3, "process": 0.12},
                   "big": True, "sign": "fire_alarm_point"},
    "junction_box": {"front": "-x", "z0": 1.2, "per": {"process": 0.4, "service": 0.25}, "big": False},
    "bulkhead_lamp": {"front": "-x", "z0": 1.95, "per": {"process": 0.3, "service": 0.3, "circulation": 0.1},
                      "big": True},
}
EMISSIVE_SLOT = {"lamp_batten": {"Lamp_01": 0.75}, "bulkhead_lamp": {"mi_Lamp_01_05_ON_translucent": 0.8,
                                                                     "mi_Lamp_01_02_ON": 0.6}}
_FRONT_YAW = {"+x": 0.0, "+y": math.pi / 2, "-x": math.pi, "-y": -math.pi / 2}
# room signs by class: (sign, count per 10 m of wall, centre height)
ROOM_SIGNS = {"circulation": [("no_smoking", 0.12, 1.65), ("first_aid", 0.06, 1.65), ("exit_running", 0.1, 1.9)],
              "process": [("access_forbidden", 0.08, 1.6)],
              "electrical_room": [("danger_hv", 0.25, 1.6)],
              "airlock": [("ppe_hardhat", 1.0, 1.55), ("ppe_boots", 1.0, 1.55), ("ppe_vest", 1.0, 1.55)],
              "waste_store": [("access_forbidden", 0.15, 1.6)]}


def _rotz(yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _quat_yaw(yaw):
    return [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]


def _item_box(size, scale, pos, yaw):
    """World AABB of an item (bottom centre at pos) of native size `size` scaled and turned by yaw."""
    sx, sy, sz = (np.asarray(size, float) * np.asarray(scale, float))
    c = np.array([[x, y, z] for x in (-sx / 2, sx / 2) for y in (-sy / 2, sy / 2) for z in (0.0, sz)])
    w = matvec_rows(c, _rotz(yaw).T) + np.asarray(pos, float)
    return w.min(0), w.max(0)


class Placer:
    """Shared state of the placement: what is protected, what is in the way, the sight lines, the views."""

    def __init__(self, ctx, site, W, protected):
        self.ctx, self.site, self.W = ctx, site, W
        P = np.where(protected)[0]
        self.PLO, self.PHI = W["lo"][P], W["hi"][P]
        self.segs = sight_segments(instrument_faces(site))
        ex = W.get("_extra")
        if ex:                                           # the generator's stored margin cells' sight lines
            self.segs = (np.concatenate([self.segs[0], ex["segments"][0]]),
                         np.concatenate([self.segs[1], ex["segments"][1]]))
        self.views = inspect_views(site)
        other = [k for k in range(len(W["names"])) if not W["names"][k].startswith(("robot/", "wall/", "floor/", "roof/"))
                 and W["type"][k] != int(mujoco.mjtGeom.mjGEOM_PLANE) and W["group"][k] <= 3
                 and (W["mats"][k] or "") not in ("wall_paint", "block_wall", "shield_concrete", "skirting")]
        self.OLO, self.OHI = W["lo"][other], W["hi"][other]
        self.other_names = [W["names"][k] for k in other]
        self.placed = []
        self.rejected = 0

    def ok(self, lo, hi, clear, big, top_limit=None):
        if lo[2] < 1.0 - 1e-6 or (top_limit is not None and hi[2] > top_limit):
            return False
        if len(self.PLO) and float(_box_dist(lo, hi, self.PLO, self.PHI).min()) < clear:
            return False
        if len(self.OLO) and float(_box_dist(lo - 0.03, hi + 0.03, self.OLO, self.OHI).min()) <= 0.0:
            return False
        if any(float(_box_dist(lo, hi, a[None], b[None])[0]) < 0.15 for a, b in self.placed):
            return False
        if len(self.segs[0]) and segments_hit_box(self.segs[0], self.segs[1], lo - 0.02, hi + 0.02, shrink=0.0).any():
            return False
        if big and in_inspect_views(lo, hi, self.views):
            return False
        return True


def _mesh_geom(ctx, item, slot, scale, pos, yaw, em=0.0, variant=None, rgba=(1, 1, 1, 1), group=1, kind="wall",
               clear=CLEAR_M, quat=None):
    it = ctx.b.items[item]
    s = it["slots"][slot]
    mesh = ctx.file_mesh(s["obj"], scale)
    tex_rel = (s.get("variants") or {}).get(variant) if variant else s.get("albedo")
    if tex_rel:
        mat = ctx.material("mesh/%s/%s/%s/%.2f/%s" % (item, slot, variant or "-", em, ",".join("%.2f" % v for v in rgba)),
                           ctx.texture(None, tex_rel), rgba=rgba, specular=0.25, shininess=0.3, emission=em)
    else:
        c = s.get("rgba", [0.8, 0.8, 0.8, 1])
        mat = ctx.material("mesh/%s/%s/plain/%.2f" % (item, slot, em), None, rgba=(c[0], c[1], c[2], 1),
                           specular=0.25, shininess=0.3, emission=em)
    k = ctx.nbody
    b = ctx.body("dress/decor/%d" % k)
    name = "dress/decor/%d/%s" % (k, item)
    b.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_MESH, meshname=mesh, material=mat, pos=[float(v) for v in pos],
               quat=quat if quat is not None else _quat_yaw(yaw), contype=0, conaffinity=0, group=group)
    return name


def _add_item(ctx, item, pos, yaw, scale, kind, clear=CLEAR_M, variant=None, rgba=(1, 1, 1, 1), group=None):
    """Every slot of a bundle mesh item as a geom (own body each); returns (lo, hi, tris)."""
    it = ctx.b.items[item]
    lo, hi = _item_box(it["size_m"], scale, pos, yaw)
    for slot in sorted(it["slots"]):
        em = EMISSIVE_SLOT.get(item, {}).get(slot, 0.0)
        name = _mesh_geom(ctx, item, slot, scale, pos, yaw, em=em, variant=variant, rgba=rgba,
                          group=group if group is not None else (2 if kind == "ceiling" else 1), kind=kind, clear=clear)
        ctx.added_geoms.append((name, lo, hi, kind, clear))
    return lo, hi, int(it["tris"])


def _verts_world(ctx, item, scale, pos, yaw):
    it = ctx.b.items[item]
    out = []
    for slot in sorted(it["slots"]):
        _, _, v = ctx.b.obj_bbox(it["slots"][slot]["obj"])
        out.append(matvec_rows(v * np.asarray(scale, float), _rotz(yaw).T) + np.asarray(pos, float))
    return np.concatenate(out, 0)


def check_fit(ctx, item, scale, pos, yaw, lo, hi, what):
    """No vertex of the swapped mesh more than FIT_M outside the box of the item's collision geoms."""
    V = _verts_world(ctx, item, scale, pos, yaw)
    out = np.maximum(np.asarray(lo, float) - V, V - np.asarray(hi, float)).max()
    if out > FIT_M + 1e-9:
        raise DressError("swapped mesh %s for %s: a vertex %.3f m outside its collision box" % (item, what, out))
    return float(out)


# ================================================================================================ swaps
def find_swaps(W, protected, nohide):
    """The generator's items that get a pack mesh: pump motors, drums, pallets, crates. Returns a list of dicts:
    item, geoms to hide (indices), the collision box (lo, hi), placement."""
    m, d = W["model"], W["data"]
    names, mats, typ = W["names"], W["mats"], W["type"]
    out = []
    used = set()
    n = len(names)
    xpos = d.geom_xpos
    for i in range(n):
        nm, mt = names[i], mats[i]
        if protected[i] or i in used:
            continue
        # ---- pump motor: '<pre>/motor' (horizontal cylinder) with its fins, fan cover, terminal box and feet
        if nm.endswith("/motor") and mt == "motor_blue" and typ[i] == CYL:
            ax = d.geom_xmat[i].reshape(3, 3)[:, 2]
            if abs(ax[2]) > 0.2:
                continue
            pre = nm[: -len("/motor")]
            parts = [k for k in range(n) if names[k].startswith(pre + "/motor")]
            if any(protected[k] or names[k] in nohide for k in parts):
                continue
            g = [k for k in range(n) if names[k] == pre + "/guard"]
            s = ax[:2] / max(norm(ax[:2]), 1e-9)
            if g:
                dv = xpos[g[0]][:2] - xpos[i][:2]
                if float(dv[0] * s[0] + dv[1] * s[1]) < 0:
                    s = -s
            lo = W["lo"][parts].min(0)
            hi = W["hi"][parts].max(0)
            out.append({"item": "motor", "hide": parts, "lo": lo, "hi": hi, "axis": s, "what": pre + "/motor",
                        "kind": "motor"})
            used.update(parts)
            continue
        # ---- drums: vertical cylinders of the drum materials, with their lids and bands
        if typ[i] == CYL and mt in ("drum_yellow", "box_blue") and abs(d.geom_xmat[i].reshape(3, 3)[2, 2]) > 0.98:
            r, hl = float(m.geom_size[i][0]), float(m.geom_size[i][1])
            if not (0.2 <= r <= 0.34 and 0.3 <= hl <= 0.5):
                continue
            c = xpos[i]
            parts = [i] + [k for k in range(n) if mats[k] in ("drum_lid", "drum_band") and k not in used
                           and np.hypot(*(xpos[k][:2] - c[:2])) < 0.03
                           and c[2] - hl - 0.05 <= xpos[k][2] <= c[2] + hl + 0.05]
            if any(protected[k] or names[k] in nohide for k in parts):
                continue
            lo = W["lo"][parts].min(0)
            hi = W["hi"][parts].max(0)
            out.append({"item": "drum", "hide": parts, "lo": lo, "hi": hi, "what": nm, "kind": "drum",
                        "variant": "grey" if mt == "drum_yellow" else "muted",
                        "rgba": (0.86, 0.87, 0.85, 1) if mt == "drum_yellow" else (0.92, 0.94, 1.0, 1)})
            used.update(parts)
            continue
        # ---- overhead cable trays (decorative boxes of material tray, group 2): pack tray segments
        if typ[i] == BOX and mt == "tray" and int(W["group"][i]) == 2:
            R = d.geom_xmat[i].reshape(3, 3)
            lo, hi = W["lo"][i], W["hi"][i]
            ext = hi - lo
            if abs(R[2, 2]) < 0.999 or max(ext[0], ext[1]) < 0.8:
                continue
            out.append({"item": "cable_tray_2m", "hide": [i], "lo": lo, "hi": hi, "what": nm, "kind": "tray"})
            used.add(i)
            continue
        # ---- pallets and crates: axis-aligned boxes
        if typ[i] == BOX and mt in ("pallet_wood", "crate_wood"):
            R = d.geom_xmat[i].reshape(3, 3)
            if abs(R[2, 2]) < 0.999 or nm in nohide:
                continue
            lo, hi = W["lo"][i], W["hi"][i]
            out.append({"item": "pallet" if mt == "pallet_wood" else "crate", "hide": [i], "lo": lo, "hi": hi,
                        "what": nm, "kind": "box"})
            used.add(i)
    return out


def apply_swaps(ctx, W, swaps, by_name, hidden, rng, report):
    for sw in swaps:
        it = ctx.b.items.get(sw["item"])
        if it is None:
            continue
        sz = np.asarray(it["size_m"], float)
        lo, hi = np.asarray(sw["lo"], float), np.asarray(sw["hi"], float)
        ext = hi - lo
        if sw["kind"] == "motor":
            s = sw["axis"]
            yaw = math.atan2(s[1], s[0]) - math.pi / 2          # the mesh's shaft (native +y) towards the coupling
            L = abs(ext[0] * s[0]) + abs(ext[1] * s[1])
            Wl = abs(ext[0] * s[1]) + abs(ext[1] * s[0])
            sx = Wl / sz[0]
            sy = L / sz[1]
            sc = min(ext[2] / sz[2], 1.25 * sx)
            scale = (sx * 0.98, sy * 0.98, sc * 0.98)
            pos = [(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, lo[2]]
            variant, rgba = None, (0.66, 0.76, 0.90, 1)
        elif sw["kind"] == "drum":
            sxy = min(ext[0], ext[1]) / sz[0]
            scale = (sxy, sxy, ext[2] / sz[2])
            pos = [(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, lo[2]]
            yaw = float(rng.uniform(0, 2 * math.pi))
            variant, rgba = sw.get("variant"), sw.get("rgba", (1, 1, 1, 1))
        else:
            long_box = 0 if ext[0] >= ext[1] else 1
            long_mesh = 0 if sz[0] >= sz[1] else 1
            yaw = 0.0 if long_box == long_mesh else math.pi / 2
            ex = (ext[0], ext[1]) if yaw == 0.0 else (ext[1], ext[0])
            scale = (ex[0] / sz[0], ex[1] / sz[1], ext[2] / sz[2])
            pos = [(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, lo[2]]
            variant, rgba = None, (1, 1, 1, 1)
        if sw["kind"] == "tray":
            # segments of the 2 m pack tray along the box; at least 10 cm tall (2 cm below and above the box allowed)
            ax = 0 if ext[0] >= ext[1] else 1
            L, Wd = ext[ax], ext[1 - ax]
            nseg = max(1, int(round(L / 2.0)))
            hm = min(ext[2] + 2 * FIT_M - 1e-4, max(ext[2], 0.10))
            scale = (Wd / sz[0], (L / nseg) / sz[1], hm / sz[2])
            yaw = -math.pi / 2 if ax == 0 else 0.0
            worst = 0.0
            for j in range(nseg):
                c = [(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, lo[2] - (hm - ext[2]) / 2]
                c[ax] = lo[ax] + (j + 0.5) * L / nseg
                worst = max(worst, check_fit(ctx, sw["item"], scale, c, yaw, lo, hi, sw["what"]))
                _add_item(ctx, sw["item"], c, yaw, scale, "swap", group=2)
            g = by_name.get(W["names"][sw["hide"][0]])
            if g is not None:
                g.material = hidden
            report.append({"item": sw["item"], "for": sw["what"], "hidden": 1, "segments": nseg,
                           "fit_outside_m": round(worst, 4), "tris": int(it["tris"]) * nseg})
            continue
        fit = check_fit(ctx, sw["item"], scale, pos, yaw, lo, hi, sw["what"])
        for k in sw["hide"]:
            g = by_name.get(W["names"][k])
            if g is not None:
                g.material = hidden
        blo, bhi, tris = _add_item(ctx, sw["item"], pos, yaw, scale, "swap", variant=variant, rgba=rgba, group=1)
        report.append({"item": sw["item"], "for": sw["what"], "hidden": len(sw["hide"]), "fit_outside_m": round(fit, 4),
                       "tris": tris})


# ================================================================================================ signs
def _sign_mesh(ctx, name, w, h, t=0.002):
    """A thin plate (w x h x t; front face +x, its texture once; other faces the plate's border colour)."""
    key = "dress/signmesh/%s" % name
    if key in ctx.meshes:
        return key
    x0, x1 = -t, 0.0
    y0, y1 = -w / 2, w / 2
    z0, z1 = 0.0, h
    V, UV, F = [], [], []

    def quad(p, uv):
        b = len(V)
        V.extend(p)
        UV.extend(uv)
        F.extend([[b, b + 1, b + 2], [b, b + 2, b + 3]])
    e = [(0.01, 0.01)] * 4
    # front (+x): viewer looks along -x; right = -y ... texture u to the viewer's right
    quad([[x1, y1, z0], [x1, y0, z0], [x1, y0, z1], [x1, y1, z1]], [(0, 1), (1, 1), (1, 0), (0, 0)])
    quad([[x0, y0, z0], [x0, y1, z0], [x0, y1, z1], [x0, y0, z1]], e)
    quad([[x0, y1, z0], [x1, y1, z0], [x1, y1, z1], [x0, y1, z1]], e)
    quad([[x1, y0, z0], [x0, y0, z0], [x0, y0, z1], [x1, y0, z1]], e)
    quad([[x0, y0, z1], [x0, y1, z1], [x1, y1, z1], [x1, y0, z1]], e)
    quad([[x0, y1, z0], [x0, y0, z0], [x1, y0, z0], [x1, y1, z0]], e)
    ctx.user_mesh(key, np.asarray(V, float), np.asarray(F, int), np.asarray(UV, float))
    return key


def _add_sign(ctx, sign, centre, n, placer, clear=BIG_CLEAR_M, check=True, kind="sign"):
    it = ctx.b.items.get("sign_" + sign)
    if it is None:
        return None
    w, h = it["size_m"]
    n = np.asarray(n, float)
    yaw = math.atan2(n[1], n[0])
    pos = np.asarray(centre, float).copy()
    pos[2] -= h / 2
    pos = pos + n * 0.0045                                   # the plate's back 2.5 mm in front of the wall's skin
    R = _rotz(yaw)
    c = matvec_rows(np.array([[x, y, z] for x in (-0.002, 0.0) for y in (-w / 2, w / 2) for z in (0.0, h)]), R.T) + pos
    lo, hi = c.min(0), c.max(0)
    if check and not placer.ok(lo, hi, clear, True):
        placer.rejected += 1
        return None
    mesh = _sign_mesh(ctx, sign, w, h)
    mat = ctx.material("sign/%s" % sign, ctx.texture(None, it["files"]["albedo"]), rgba=(1, 1, 1, 1), specular=0.15,
                       shininess=0.2)
    k = ctx.nbody
    b = ctx.body("dress/decor/%d" % k)
    name = "dress/decor/%d/sign_%s" % (k, sign)
    b.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_MESH, meshname=mesh, material=mat, pos=[float(v) for v in pos],
               quat=_quat_yaw(yaw), contype=0, conaffinity=0, group=1)
    ctx.added_geoms.append((name, lo, hi, kind, clear))
    placer.placed.append((lo, hi))
    return {"item": "sign_" + sign, "pos": [round(float(v), 3) for v in pos], "tris": 12}


# ================================================================================================ floor decals
def _floor_keepouts(site, W, protected):
    """World rects (x0, y0, x1, y1) where no floor decal goes: traversal elements grown by their 1.0 m approach strips,
    the approach zones the generator records, the pad, survey marks and floor markers (0.3 m)."""
    out = []
    for e in site.get("elements", []) or []:
        fp = np.asarray(e.get("footprint") or [], float)
        if len(fp):
            out.append((fp[:, 0].min() - 1.0, fp[:, 1].min() - 1.0, fp[:, 0].max() + 1.0, fp[:, 1].max() + 1.0))
    for z in ((site.get("ladder") or {}).get("approach_zones") or []):
        try:
            a = np.asarray(z.get("rect") if isinstance(z, dict) else z, float).ravel()
            if len(a) == 4:
                out.append((min(a[0], a[2]), min(a[1], a[3]), max(a[0], a[2]), max(a[1], a[3])))
        except (TypeError, ValueError):
            pass
    for k in np.where(protected)[0]:
        if W["lo"][k][2] < 0.02 and W["type"][k] != int(mujoco.mjtGeom.mjGEOM_PLANE):
            lo, hi = W["lo"][k], W["hi"][k]
            out.append((lo[0] - 0.3, lo[1] - 0.3, hi[0] + 0.3, hi[1] + 0.3))
    ex = W.get("_extra")
    if ex is not None and len(ex["floor_rects"]):
        out += [tuple(r) for r in ex["floor_rects"]]
    return np.asarray(out, float).reshape(-1, 4)


def _rect_free(r, keep):
    if not len(keep):
        return True
    return not bool(np.any((keep[:, 0] < r[2]) & (keep[:, 2] > r[0]) & (keep[:, 1] < r[3]) & (keep[:, 3] > r[1])))


def contact_shadows(ctx, site, W, protected, swaps, skin_faces, keep, width=0.22, alpha=0.5):
    """Flat transparent strips on the floor where things meet it: around floor-standing equipment (plinths, tanks,
    cabinets, racks, pallets) and along the foot of every wall face. Pieces that would lie in a keep-out are dropped."""
    V, F, UV = [], [], []

    def strip(p0, p1, nrm, w):
        """A quad on the floor from the segment p0-p1 out along nrm by w; u = 0 at the contact line."""
        a = np.array([p0[0], p0[1], DECAL_Z])
        b = np.array([p1[0], p1[1], DECAL_Z])
        o = np.array([nrm[0], nrm[1], 0.0]) * w
        xs = [a[0], b[0], a[0] + o[0], b[0] + o[0]]
        ys = [a[1], b[1], a[1] + o[1], b[1] + o[1]]
        if not _rect_free((min(xs), min(ys), max(xs), max(ys)), keep):
            return 0
        base = len(V)
        V.extend([a, b, b + o, a + o])
        UV.extend([(0.0, 0.5), (0.0, 0.5), (1.0, 0.5), (1.0, 0.5)])
        # upward facing: (b - a) x (o) must point +z
        if np.cross(b - a, o)[2] > 0:
            F.extend([[base, base + 1, base + 2], [base, base + 2, base + 3]])
        else:
            F.extend([[base, base + 2, base + 1], [base, base + 3, base + 2]])
        return 1

    n = 0
    eqmats = ("concrete_plinth", "tank_grey", "cabinet", "cabinet_white", "ahu_grey", "ahu_panel", "locker_blue",
              "rack_blue", "rack_orange", "pallet_wood", "crate_wood", "base_station", "machine_green", "lab_bench",
              "bench_wood", "panel_grey", "drum_yellow", "box_blue")
    for k in range(len(W["names"])):
        if protected[k] or W["mats"][k] not in eqmats or W["group"][k] != 0:
            continue
        lo, hi = W["lo"][k], W["hi"][k]
        if lo[2] > 0.03 or (hi[0] - lo[0]) < 0.2 or (hi[1] - lo[1]) < 0.2:
            continue
        x0, y0, x1, y1 = lo[0], lo[1], hi[0], hi[1]
        for p0, p1, nr in (((x0, y0), (x1, y0), (0, -1)), ((x1, y0), (x1, y1), (1, 0)), ((x1, y1), (x0, y1), (0, 1)),
                           ((x0, y1), (x0, y0), (-1, 0))):
            n += strip(p0, p1, nr, width)
    # along the wall foot: pieces of at most 1.2 m
    for f, room in skin_faces:
        if f.get("top") or room is None or f["o"][2] > 0.05:
            continue
        nr = np.asarray(f["n"], float)
        if abs(nr[2]) > 0.5:
            continue
        a = np.asarray(f["o"], float) + nr * 0.001
        L = float(f["la"])
        k = max(1, int(math.ceil(L / 1.2)))
        for j in range(k):
            p0 = a + np.asarray(f["a"], float) * (L * j / k)
            p1 = a + np.asarray(f["a"], float) * (L * (j + 1) / k)
            n += strip(p0[:2], p1[:2], nr[:2], 0.16)
    if not F:
        return 0, 0
    mesh = ctx.user_mesh("dress/contact_shadows", np.asarray(V, float), np.asarray(F, int), np.asarray(UV, float))
    mat = ctx.material("fx/contact", ctx.texture("fx_ao_ramp"), rgba=(0.0, 0.0, 0.0, alpha), specular=0.0,
                       shininess=0.0)
    b = ctx.body("dress/decor/%d" % ctx.nbody)
    b.add_geom(name="dress/fx/contact_shadows", type=mujoco.mjtGeom.mjGEOM_MESH, meshname=mesh, material=mat,
               contype=0, conaffinity=0, group=1)
    return n, len(F)


# ================================================================================================ the pass
# ================================================================================================ pipe marking
# service-colour identification bands (a contrasting ring) and flow arrows on the generator's pipes, only where the
# whole mark is 1.0 m or more above the floor, 0.3 m from every keep-clear geom, outside every sight line
BAND = {"pipe_cw": (0.92, 0.92, 0.90), "pipe_dw": (0.92, 0.92, 0.90), "pipe_ia": (0.92, 0.92, 0.90),
        "pipe_fw": (0.92, 0.92, 0.90), "pipe_rw": (0.46, 0.30, 0.52), "pipe_oil": (0.10, 0.10, 0.10),
        "pipe_st": (0.66, 0.20, 0.17)}
ARROW = {"pipe_oil": (0.92, 0.92, 0.90), "pipe_rw": (0.95, 0.95, 0.93)}      # else black arrows
BAND_W, BAND_GAP, ARROW_L = 0.10, 2.6, 0.22


def _pipe_mark_ok(pl, lo, hi, host_prefix):
    if lo[2] < 1.0 - 1e-6:
        return False
    if len(pl.PLO) and float(_box_dist(lo, hi, pl.PLO, pl.PHI).min()) < CLEAR_M:
        return False
    if len(pl.OLO):
        d = _box_dist(lo - 0.01, hi + 0.01, pl.OLO, pl.OHI)
        hit = np.where(d <= 0.0)[0]
        if any(not pl.other_names[k].startswith(host_prefix) for k in hit):
            return False
    if len(pl.segs[0]) and segments_hit_box(pl.segs[0], pl.segs[1], lo - 0.01, hi + 0.01, shrink=0.0).any():
        return False
    return True


def _arrow_patch(c, a, e1, R, L, half_ang, n_phi=4, n_t=3):
    e2 = np.cross(a, e1)
    V, UV, F = [], [], []
    for j in range(n_t + 1):
        t = -L / 2 + L * j / n_t
        for i in range(n_phi + 1):
            ph = -half_ang + 2 * half_ang * i / n_phi
            V.append(c + a * t + R * (math.cos(ph) * e1 + math.sin(ph) * e2))
            UV.append((i / n_phi, 1.0 - j / n_t))
    w = n_phi + 1
    for j in range(n_t):
        for i in range(n_phi):
            r0, r1 = j * w, (j + 1) * w
            F += [[r0 + i, r0 + i + 1, r1 + i + 1], [r0 + i, r1 + i + 1, r1 + i]]
    return np.asarray(V), np.asarray(F, int), np.asarray(UV)


def pipe_marks(ctx, W, pl, rooms, rng, report):
    m, d = W["model"], W["data"]
    nb = na = 0
    mats_band, arrows = {}, []
    for k, nm in enumerate(W["names"]):
        mt = W["mats"][k]
        if mt not in BAND or int(W["type"][k]) != CYL or nm.startswith(("robot/", "dress/")):
            continue
        r, hl = float(m.geom_size[k][0]), float(m.geom_size[k][1])
        if 2 * hl < 1.2 or r < 0.02:
            continue
        ax = d.geom_xmat[k].reshape(3, 3)[:, 2].copy()
        c0 = d.geom_xpos[k].copy()
        host = nm.rsplit("/", 1)[0] + "/"
        L = 2 * hl
        n = max(1, int(L // BAND_GAP))
        off = float(rng.uniform(0.3, 0.7))
        sgn = 1.0 if rng.random() < 0.5 else -1.0               # the run's flow direction (one per segment)
        for j in range(n):
            t = -hl + (j + off) * L / n
            if t < -hl + 0.3 or t > hl - 0.3:
                continue
            c = c0 + ax * t
            Rb = r + 0.004
            ext = np.array([Rb * math.sqrt(max(0.0, 1 - ax[q] ** 2)) + abs(ax[q]) * BAND_W / 2 for q in range(3)])
            lo, hi = c - ext, c + ext
            if not _pipe_mark_ok(pl, lo, hi, host):
                continue
            col = BAND[mt]
            key = "band/%s" % mt
            if key not in mats_band:
                mats_band[key] = ctx.material(key, None, rgba=(col[0], col[1], col[2], 1), specular=0.3, shininess=0.35)
            b = ctx.body("dress/decor/%d" % ctx.nbody)
            from .core import _quat_z_to
            b.add_geom(name="dress/decor/%d/pipe_band" % (ctx.nbody - 1), type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                       size=[Rb, BAND_W / 2, 0], pos=[float(v) for v in c], quat=_quat_z_to(ax),
                       material=mats_band[key], contype=0, conaffinity=0, group=1 if c[2] < 2.2 else 2)
            ctx.added_geoms.append(("dress/decor/%d/pipe_band" % (ctx.nbody - 1), lo, hi, "pipe_mark", CLEAR_M))
            pl.placed.append((lo, hi))
            nb += 1
            # an arrow beside the band, on the side a robot sees (down for a horizontal run, to the room's centre for
            # a vertical one)
            ca = c + ax * sgn * (BAND_W / 2 + 0.06 + ARROW_L / 2)
            if abs(ax[2]) < 0.7:
                e1 = np.array([0.0, 0.0, -1.0]) - ax * (-ax[2])
            else:
                rc = None
                for x0, y0, x1, y1, rm in rooms:
                    if x0 <= ca[0] <= x1 and y0 <= ca[1] <= y1:
                        rc = np.array([(x0 + x1) / 2, (y0 + y1) / 2, ca[2]])
                e1 = (rc - ca) if rc is not None else np.array([1.0, 0, 0])
                e1 = e1 - ax * dot3(e1, ax)
            if norm(e1) < 1e-6:
                continue
            e1 = e1 / norm(e1)
            Ra = r + 0.0025
            half = min(0.9, 0.06 / max(Ra, 1e-3))
            V, F, UV = _arrow_patch(ca, ax * sgn, e1, Ra, ARROW_L, half)
            alo, ahi = V.min(0), V.max(0)
            if not _pipe_mark_ok(pl, alo, ahi, host):
                continue
            arrows.append((V, F, UV, ARROW.get(mt, (0.06, 0.06, 0.06)), alo, ahi))
    # all arrows of one colour in one mesh (one body each colour)
    by_col = {}
    for V, F, UV, col, alo, ahi in arrows:
        by_col.setdefault(col, []).append((V, F, UV, alo, ahi))
    for col, lst in sorted(by_col.items()):
        Vs, Fs, UVs, base = [], [], [], 0
        for V, F, UV, alo, ahi in lst:
            Vs.append(V)
            Fs.append(F + base)
            UVs.append(UV)
            base += len(V)
        name = "dress/fx_arrows/%d" % len(ctx.meshes)
        mesh = ctx.user_mesh(name, np.concatenate(Vs), np.concatenate(Fs), np.concatenate(UVs))
        mat = ctx.material("arrow/%.2f" % col[0], ctx.texture("fx_arrow"), rgba=(col[0], col[1], col[2], 0.99),
                           specular=0.1, shininess=0.1)
        b = ctx.body("dress/decor/%d" % ctx.nbody)
        b.add_geom(name="dress/decor/%d/flow_arrows" % (ctx.nbody - 1), type=mujoco.mjtGeom.mjGEOM_MESH, meshname=mesh,
                   material=mat, contype=0, conaffinity=0, group=1)
        for V, F, UV, alo, ahi in lst:
            ctx.added_geoms.append(("dress/decor/%d/flow_arrows" % (ctx.nbody - 1), alo, ahi, "pipe_mark", CLEAR_M))
            na += 1
    report.append({"item": "pipe_marks", "bands": nb, "arrows": na, "tris": 2 * 12 * na})
    return nb, na


def place_decor(ctx, site, W, protected, skin_src, faces, rooms, plan, rng, by_name=None, hidden=None, nohide=(),
                keepclear=None):
    items = []
    counts = {"lamps": 0, "wall_items": 0, "signs": 0, "swaps": 0, "wall_rejected": 0}
    # ---------------------------------------------------------------- swaps inside collision boxes
    swaps_rep = []
    if by_name is not None and hidden is not None:
        sw = find_swaps(W, protected, set(nohide))
        apply_swaps(ctx, W, sw, by_name, hidden, rng, swaps_rep)
        counts["swaps"] = len(swaps_rep)
        items.extend(swaps_rep)
    # ---------------------------------------------------------------- ceiling battens over the generator's lamps
    if "lamp_batten" in ctx.b.items:
        it = ctx.b.items["lamp_batten"]
        sz = np.asarray(it["size_m"], float)
        mlong = 0 if sz[0] >= sz[1] else 1
        for i, kind in skin_src:
            if kind != "lamp":
                continue
            lo, hi = W["lo"][i], W["hi"][i]
            ext = hi - lo
            blong = 0 if ext[0] >= ext[1] else 1
            s = float(max(ext[0], ext[1]) * 1.04 / sz[mlong])
            yaw = 0.0 if mlong == blong else math.pi / 2
            pos = [(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, lo[2] - 0.004]
            blo, bhi, tris = _add_item(ctx, "lamp_batten", pos, yaw, (s, s, s), "ceiling")
            items.append({"item": "lamp_batten", "at": W["names"][i], "pos": [round(v, 3) for v in pos], "tris": tris})
            counts["lamps"] += 1
    # ---------------------------------------------------------------- the generator's extinguisher signs -> a sign
    pl = Placer(ctx, site, W, protected if keepclear is None else keepclear)
    if by_name is not None and hidden is not None and "sign_fire_extinguisher" in ctx.b.items:
        for k, nm in enumerate(W["names"]):
            if "/extsign" in nm and W["mats"][k] == "red_fire" and not protected[k]:
                lo, hi = W["lo"][k], W["hi"][k]
                ext = hi - lo
                thin = 0 if ext[0] < ext[1] else 1
                c = (lo + hi) / 2
                room = None
                for x0, y0, x1, y1, r in rooms:
                    if x0 <= c[0] <= x1 and y0 <= c[1] <= y1:
                        room = r
                nvec = np.zeros(3)
                if room is not None:
                    rc = np.array([(room["rect"][0] + room["rect"][2]) / 2, (room["rect"][1] + room["rect"][3]) / 2])
                    nvec[thin] = 1.0 if rc[thin] > c[thin] else -1.0
                else:
                    nvec[thin] = 1.0
                centre = c.copy()
                centre[thin] = (hi[thin] if nvec[thin] > 0 else lo[thin]) - nvec[thin] * 0.0045
                # at the generator's own sign: a swap of a decorative item, not new decor
                r = _add_sign(ctx, "fire_extinguisher", centre, nvec, pl, check=False, kind="swap")
                if r:
                    g = by_name.get(nm)
                    if g is not None:
                        g.material = hidden
                    items.append(dict(r, replaces=nm))
                    counts["signs"] += 1
    # ---------------------------------------------------------------- wall items and room signs
    by_room = {}
    for f, room in faces:
        if f.get("top") or room is None or f["lb"] < 2.0 or f["o"][2] > 0.05:
            continue
        by_room.setdefault(room["id"], (room, []))[1].append(f)
    for rid in sorted(by_room):
        room, fl = by_room[rid]
        rc = plan["rooms"].get(rid, {}).get("class", "service")
        total = sum(f["la"] for f in fl)
        for item in sorted(WALL_ITEMS):
            if item not in ctx.b.items:
                continue
            spec_ = WALL_ITEMS[item]
            rate = spec_["per"].get(rc, 0.0)
            want = int(math.floor(rate * total / 10.0 + rng.random()))
            if want <= 0:
                continue
            it = ctx.b.items[item]
            sz = np.asarray(it["size_m"], float)
            fyaw = _FRONT_YAW[spec_["front"]]
            depth = sz[0] if spec_["front"] in ("+x", "-x") else sz[1]
            width = sz[1] if spec_["front"] in ("+x", "-x") else sz[0]
            clear = BIG_CLEAR_M if spec_["big"] else CLEAR_M
            got = 0
            for _ in range(40):
                if got >= want:
                    break
                f = fl[int(rng.integers(len(fl)))]
                if f["la"] < width + 2 * clear:
                    continue
                s = float(rng.uniform(width / 2 + 0.3, f["la"] - width / 2 - 0.3))
                n = np.asarray(f["n"], float)
                base = np.asarray(f["o"], float) + np.asarray(f["a"], float) * s
                pos = base + n * (depth / 2 + 0.003)
                pos[2] = spec_["z0"]
                yaw = math.atan2(n[1], n[0]) - fyaw
                lo, hi = _item_box(sz, (1, 1, 1), pos, yaw)
                if not pl.ok(lo, hi, clear, spec_["big"], top_limit=f["o"][2] + f["lb"] - 0.05):
                    pl.rejected += 1
                    continue
                # the item's sign above it (fire point signs), if it fits too
                sg = spec_.get("sign")
                blo, bhi, tris = _add_item(ctx, item, pos, yaw, (1, 1, 1), "wall", clear=clear)
                pl.placed.append((blo, bhi))
                items.append({"item": item, "room": rid, "pos": [round(float(v), 3) for v in pos],
                              "yaw_deg": round(math.degrees(yaw), 1), "tris": tris})
                counts["wall_items"] += 1
                got += 1
                if sg and ("sign_" + sg) in ctx.b.items:
                    hs = ctx.b.items["sign_" + sg]["size_m"][1]
                    cz = bhi[2] + 0.06 + hs / 2
                    if cz + hs / 2 < f["o"][2] + f["lb"] - 0.08:
                        cen = base + n * 0.001
                        cen[2] = cz
                        r = _add_sign(ctx, sg, cen, n, pl)
                        if r:
                            items.append(r)
                            counts["signs"] += 1
        # room signs
        key = room.get("type") if room.get("type") in ROOM_SIGNS else (room.get("kind") if room.get("kind") in ROOM_SIGNS
                                                                       else rc)
        for sg, rate, cz in ROOM_SIGNS.get(key, []):
            want = 1 if rate >= 1.0 else int(math.floor(rate * total / 10.0 + rng.random()))
            for _ in range(30):
                if want <= 0:
                    break
                f = fl[int(rng.integers(len(fl)))]
                it = ctx.b.items.get("sign_" + sg)
                if it is None or f["la"] < it["size_m"][0] + 0.8:
                    continue
                s = float(rng.uniform(it["size_m"][0] / 2 + 0.4, f["la"] - it["size_m"][0] / 2 - 0.4))
                n = np.asarray(f["n"], float)
                cen = np.asarray(f["o"], float) + np.asarray(f["a"], float) * s + n * 0.001
                cen[2] = cz
                r = _add_sign(ctx, sg, cen, n, pl)
                if r:
                    items.append(dict(r, room=rid))
                    counts["signs"] += 1
                    want -= 1
    counts["wall_rejected"] = pl.rejected
    # ---------------------------------------------------------------- service bands and flow arrows on pipes
    if "fx_arrow" in ctx.b.items:
        nb, na = pipe_marks(ctx, W, pl, rooms, rng, items)
        counts["pipe_bands"], counts["flow_arrows"] = nb, na
    # ---------------------------------------------------------------- contact shadows on the floor
    keep = _floor_keepouts(site, W, protected)
    ns, nt = contact_shadows(ctx, site, W, protected, swaps_rep, faces, keep)
    counts["contact_strips"] = ns
    items.append({"item": "contact_shadows", "tris": nt})
    return {"items": items, "counts": counts}
