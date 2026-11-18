"""The dressing's checks. Each raises DressError (the site does not load / generate)."""

from __future__ import annotations

import math
import re

import numpy as np

from .xmath import norm as _norm, rows_dot as _rows_dot, rows_norm as _rows_norm

TAG_RE = re.compile(r"\b[A-Z]{1,3}-\d{2,5}\b")
CLEAR_M = 0.3                    # around every instrument face, plate, sign and marker
FIT_M = 0.02                     # a swapped mesh: no vertex more than this outside its collision box
# signal yellow (the tag plate, sRGB 245,196,0): hue 38-58 deg, saturation >= 0.55, value >= 0.45
YELLOW_BAND = {"hue_deg": (38.0, 58.0), "sat_min": 0.55, "val_min": 0.45}


class DressError(RuntimeError):
    """The dressing would change physics, touch a protected item, or put something where it must not be."""


# ------------------------------------------------------------------------------------------------ physics
def signature(spec):
    """Every geom of the spec with what physics and the sensors' collision geometry depend on."""
    out = []
    for g in spec.geoms:
        out.append((g.name, int(g.type), tuple(np.round(np.asarray(g.size, float), 9)),
                    tuple(np.round(np.asarray(g.pos, float), 9)), tuple(np.round(np.asarray(g.quat, float), 9)),
                    int(g.contype), int(g.conaffinity), int(g.condim), tuple(np.round(np.asarray(g.friction, float), 9)),
                    int(g.priority), float(g.margin), float(g.gap), g.meshname or ""))
    return out


def check_spec_unchanged(before, after, nbody_added):
    if len(after) < len(before):
        raise DressError("the dressing removed geoms (%d -> %d)" % (len(before), len(after)))
    for k, (a, b) in enumerate(zip(before, after)):
        if a != b:
            raise DressError("geom %d (%s) changed by the dressing: %s -> %s" % (k, a[0], a, b))
    extra = after[len(before):]
    bad = [x[0] for x in extra if x[5] or x[6]]
    if bad:
        raise DressError("dressing geoms that collide: %s" % bad[:5])
    names = [x[0] for x in extra]
    if any(not n.startswith("dress/") for n in names):
        raise DressError("dressing geoms without the dress/ prefix")
    return {"ok": True, "geoms_before": len(before), "geoms_added": len(extra), "bodies_added": int(nbody_added)}


def verify_compiled(m0, m1):
    """The compiled dressed model against the undressed one: every geom, body, joint, actuator of the undressed model
    keeps its id, type, size, pose, contact settings and body; new geoms collide with nothing."""
    import mujoco
    if m1.nbody < m0.nbody or m1.ngeom < m0.ngeom:
        raise DressError("fewer bodies or geoms after the dressing")
    for f in ("njnt", "nq", "nv", "nu", "neq", "nsensor"):
        if getattr(m0, f) != getattr(m1, f):
            raise DressError("%s changed by the dressing" % f)
    n = m0.ngeom
    for f in ("geom_type", "geom_size", "geom_pos", "geom_quat", "geom_contype", "geom_conaffinity", "geom_bodyid",
              "geom_condim", "geom_friction", "geom_margin", "geom_gap", "geom_solref", "geom_solimp", "geom_priority"):
        a, b = np.asarray(getattr(m0, f))[:n], np.asarray(getattr(m1, f))[:n]
        if not np.array_equal(a, b):
            k = int(np.argwhere(np.any((a != b).reshape(n, -1), axis=1))[0][0])
            raise DressError("%s of geom %d (%s) changed" % (f, k, mujoco.mj_id2name(m0, mujoco.mjtObj.mjOBJ_GEOM, k)))
    for f in ("body_parentid", "body_pos", "body_quat", "body_mass", "body_inertia", "body_ipos", "body_iquat"):
        a, b = np.asarray(getattr(m0, f))[:m0.nbody], np.asarray(getattr(m1, f))[:m0.nbody]
        if not np.array_equal(a, b):
            raise DressError("%s changed by the dressing" % f)
    new = slice(n, m1.ngeom)
    if np.any(m1.geom_contype[new]) or np.any(m1.geom_conaffinity[new]):
        raise DressError("a dressing geom collides")
    for f in ("opt", ):
        pass
    if not np.isclose(m0.stat.meaninertia, m1.stat.meaninertia):
        raise DressError("stat.meaninertia changed (%g -> %g)" % (m0.stat.meaninertia, m1.stat.meaninertia))
    return {"ok": True, "geoms": int(n), "geoms_added": int(m1.ngeom - n), "bodies_added": int(m1.nbody - m0.nbody)}


# ------------------------------------------------------------------------------------------------ protected items
def _box_dist(lo, hi, LO, HI):
    """Distance between an AABB and many AABBs (0 when they overlap)."""
    d = np.maximum(0.0, np.maximum(LO - hi, lo - HI))
    return np.sqrt((d * d).sum(1))


def check_skins_behind_protected(planes, W, protected, off):
    """A skin lies `off` in front of a face; no protected item may end between the face and its skin (it would be
    hidden). Returns the count of protected items near a skinned face."""
    P = np.where(protected)[0]
    LO, HI = W["lo"][P], W["hi"][P]
    near = 0
    edges = 0
    for f in planes:
        n = np.asarray(f["n"], float)
        ax = int(np.argmax(np.abs(n)))
        sg = float(np.sign(n[ax]))
        o = np.asarray(f["o"], float)
        face = o[ax]
        # the face rectangle (other two axes)
        ends = [o, o + np.asarray(f["a"]) * f["la"] + np.asarray(f["b"]) * f["lb"]]
        rlo = np.minimum(ends[0], ends[1])
        rhi = np.maximum(ends[0], ends[1])
        oth = [k for k in range(3) if k != ax]
        ov = np.all((LO[:, oth] < rhi[oth] - 1e-4) & (HI[:, oth] > rlo[oth] + 1e-4), axis=1)
        if not ov.any():
            continue
        # protected extent in front of the face along n
        if sg > 0:
            front = HI[:, ax] - face
            back = LO[:, ax] - face
        else:
            front = face - LO[:, ax]
            back = face - HI[:, ax]
        m = ov & (front > -1e-4) & (back < 0.05)
        near += int(m.sum())
        bad = m & (front < off + 0.0005)
        if bad.any():
            # an item that only touches the face's rectangle along an edge (a hatch head entering the wall by 2 cm,
            # where version 1 already had two coplanar faces) is allowed: its overlap with the face is <= 3 cm wide
            olo = np.maximum(LO[:, oth], rlo[oth])
            ohi = np.minimum(HI[:, oth], rhi[oth])
            edge = np.min(ohi - olo, axis=1) <= 0.03
            edges += int((bad & edge).sum())
            bad = bad & ~edge
        if bad.any():
            k = int(P[np.where(bad)[0][0]])
            raise DressError("protected geom %s ends %.4f m in front of a skinned face (skin at %.4f m)"
                             % (W["names"][k], float(front[np.where(bad)[0][0]]), off))
    return {"ok": True, "protected_near_skins": near, "edge_contacts": edges}


def instrument_faces(site: dict):
    """(centre, unit normal or None, max off-axis deg, max dist, tag) of every instrument, plate and marker of the
    site: the listed instruments, the decoys, the plain instruments and the grid markers."""
    out = []
    for key in ("ips", "decoys", "plain_instruments", "grid_markers"):
        for r in site.get(key, []) or []:
            pose = r.get("pose") or {}
            p = pose.get("pos") or r.get("pos")
            if p is None:
                continue
            yaw = pose.get("yaw", r.get("yaw"))
            n = None
            if yaw is not None:
                n = np.array([math.cos(float(yaw)), math.sin(float(yaw)), 0.0])
                ob = (r.get("observation_pose") or {}).get("camera")
                if ob is not None and ((float(ob[0]) - float(p[0])) * n[0] + (float(ob[1]) - float(p[1])) * n[1]) < 0:
                    n = -n
            t = r.get("type") or ("grid_marker" if key == "grid_markers" else "")
            ang = {"gauge": 45.0, "level": 30.0, "valve_lever": 45.0, "valve_handwheel": 45.0, "lamps": 55.0,
                   "thermal": 60.0}.get(t, 60.0)
            obs = (r.get("observation_pose") or {}).get("camera")
            out.append({"p": np.asarray(p, float), "n": n, "ang": ang, "dist": 5.0,
                        "tag": r.get("tag") or r.get("id") or r.get("body"), "obs": obs,
                        "two_sided": key == "plain_instruments"})
    return out


def sight_segments(faces, heights=(0.18, 0.30, 0.45)):
    """Sample camera points of every instrument's reading region (distance x off-axis angle x camera height) and the
    observation pose's camera: segments (A, B) from camera to face centre."""
    A, B = [], []
    dists = (0.8, 1.6, 2.6, 3.6, 5.0)
    for f in faces:
        p = f["p"]
        ns = []
        if f["n"] is None:
            ns = [np.array([math.cos(a), math.sin(a), 0.0]) for a in np.linspace(0, 2 * math.pi, 12, endpoint=False)]
            angs = [0.0]
        else:
            ns = [f["n"]] + ([-f["n"]] if f["two_sided"] else [])
            a = f["ang"]
            angs = list(np.linspace(-a, a, 7))
        for n in ns:
            base = math.atan2(n[1], n[0])
            for ad in angs:
                th = base + math.radians(ad)
                d2 = np.array([math.cos(th), math.sin(th)])
                for dist in dists:
                    for h in heights:
                        dz = h - p[2]
                        r = math.sqrt(max(dist * dist - dz * dz, 0.04))
                        A.append([p[0] + d2[0] * r, p[1] + d2[1] * r, h])
                        B.append(p)
        if f.get("obs") is not None:
            A.append(list(f["obs"]))
            B.append(p)
    return np.asarray(A, float).reshape(-1, 3), np.asarray(B, float).reshape(-1, 3)


def segments_hit_box(A, B, lo, hi, shrink=0.02):
    """Which segments A->B cross the AABB [lo, hi] (shrunk by `shrink`; the last 5 cm at the face are ignored)."""
    lo = np.asarray(lo, float) + shrink
    hi = np.asarray(hi, float) - shrink
    D = B - A
    L = _rows_norm(D)
    t0 = np.zeros(len(A))
    t1 = np.clip(1.0 - 0.05 / np.maximum(L, 1e-6), 0.0, 1.0)
    for k in range(3):
        d = D[:, k]
        with np.errstate(divide="ignore", invalid="ignore"):
            ta = (lo[k] - A[:, k]) / d
            tb = (hi[k] - A[:, k]) / d
        par = np.abs(d) < 1e-12
        inside = (A[:, k] >= lo[k]) & (A[:, k] <= hi[k])
        tmin = np.where(par, np.where(inside, -np.inf, np.inf), np.minimum(ta, tb))
        tmax = np.where(par, np.where(inside, np.inf, -np.inf), np.maximum(ta, tb))
        t0 = np.maximum(t0, tmin)
        t1 = np.minimum(t1, tmax)
    return t0 <= t1


def inspect_views(site: dict, hfov_deg=18.0, aspect=0.75, margin=1.15):
    """The inspection camera's view at every listed instrument's observation pose (camera point -> the instrument):
    (camera, forward, right, up, tan half-angles)."""
    out = []
    th = math.tan(math.radians(hfov_deg / 2)) * margin
    tv = th * aspect
    for r in site.get("ips", []) or []:
        o = (r.get("observation_pose") or {}).get("camera")
        p = (r.get("pose") or {}).get("pos")
        if o is None or p is None or r.get("type") in ("dose_point", "hotspot"):
            continue
        c = np.asarray(o, float)
        f = np.asarray(p, float) - c
        if _norm(f) < 1e-6:
            continue
        f /= _norm(f)
        rt = np.cross(f, [0.0, 0.0, 1.0])
        if _norm(rt) < 1e-6:
            rt = np.array([1.0, 0.0, 0.0])
        rt /= _norm(rt)
        up = np.cross(rt, f)
        out.append((c, f, rt, up, th, tv, r.get("tag")))
    return out


def in_inspect_views(lo, hi, views):
    """Does the AABB show in one of the inspection views? (its corners, centre and face centres tested)"""
    lo = np.asarray(lo, float)
    hi = np.asarray(hi, float)
    pts = [np.array([x, y, z]) for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])]
    c = (lo + hi) / 2
    pts.append(c)
    for k in range(3):
        for v in (lo[k], hi[k]):
            q = c.copy()
            q[k] = v
            pts.append(q)
    P = np.asarray(pts)
    for cam, f, rt, up, th, tv, _ in views:
        V = P - cam
        z = _rows_dot(V, f)
        ok = z > 0.05
        if not ok.any():
            continue
        x = _rows_dot(V, rt)[ok] / z[ok]
        y = _rows_dot(V, up)[ok] / z[ok]
        if np.any((np.abs(x) <= th) & (np.abs(y) <= tv)):
            return True
        # a box larger than the view: corners on both sides of the axis
        if (x.min() < -th and x.max() > th and np.any(np.abs(y) <= tv)) or \
                (y.min() < -tv and y.max() > tv and np.any(np.abs(x) <= th)):
            return True
    return False


def check_decor_clearance(added, W, protected, site, segs=None):
    """Every new decor item (walls, signs): its clearance (0.3 m, 0.6 m for large or saturated items and signs) from
    every protected geom, outside every reading region's sight lines, and the large ones outside the inspection
    views at the observation poses. Swapped meshes (inside their items' boxes) and the ceiling battens (at the
    generator's lamps) are checked by their own rules."""
    if not added:
        return {"ok": True, "items": 0}
    P = np.where(protected)[0]
    LO, HI = W["lo"][P], W["hi"][P]
    if segs is None:
        segs = sight_segments(instrument_faces(site))
        ex = W.get("_extra")
        if ex:
            segs = (np.concatenate([segs[0], ex["segments"][0]]), np.concatenate([segs[1], ex["segments"][1]]))
    views = inspect_views(site)
    A, Bp = segs
    worst = {}
    n = 0
    for rec in added:
        name, lo, hi, kind = rec[:4]
        clear = rec[4] if len(rec) > 4 else CLEAR_M
        if kind in ("swap", "ceiling"):
            continue
        n += 1
        d = _box_dist(np.asarray(lo), np.asarray(hi), LO, HI)
        k = int(np.argmin(d)) if len(d) else -1
        if len(d) and d[k] < clear - 1e-6:
            raise DressError("decor %s is %.3f m from %s (needs %.2f m)" % (name, d[k], W["names"][P[k]], clear))
        if len(d):
            worst[clear] = min(worst.get(clear, 1e9), float(d[k]))
        if len(A) and segments_hit_box(A, Bp, lo, hi).any():
            raise DressError("decor %s lies in a reading region's sight lines" % name)
        if clear > CLEAR_M and in_inspect_views(lo, hi, views):
            raise DressError("decor %s shows in an inspection view at an observation pose" % name)
    return {"ok": True, "items": n, "min_clearance_m": {str(k): round(v, 3) for k, v in sorted(worst.items())},
            "segments": int(len(A)), "views": len(views)}


def check_tag_strings(bundle):
    """No sign, decal or texture of the bundle carries a string of the tag grammar (letters, hyphen, digits)."""
    bad = []
    for name, it in bundle.items.items():
        for s in it.get("strings", []) or []:
            if TAG_RE.search(str(s).upper()):
                bad.append((name, s))
    if bad:
        raise DressError("bundle items with tag-like strings: %s" % bad[:5])
    return {"ok": True, "text_items": sum(1 for it in bundle.items.values() if it.get("strings"))}


# ------------------------------------------------------------------------------------------------ rendered frames
def yellow_mask(rgb):
    """Pixels of an sRGB uint8 image within the reserved signal-yellow band."""
    x = rgb.astype(np.float32) / 255.0
    mx = x.max(-1)
    mn = x.min(-1)
    c = mx - mn
    r, g, b = x[..., 0], x[..., 1], x[..., 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        h = np.where(mx == r, ((g - b) / np.maximum(c, 1e-6)) % 6.0,
                     np.where(mx == g, (b - r) / np.maximum(c, 1e-6) + 2.0, (r - g) / np.maximum(c, 1e-6) + 4.0)) * 60.0
        s = np.where(mx > 0, c / np.maximum(mx, 1e-6), 0.0)
    lo, hi = YELLOW_BAND["hue_deg"]
    return (h >= lo) & (h <= hi) & (s >= YELLOW_BAND["sat_min"]) & (mx >= YELLOW_BAND["val_min"])


def yellow_offenders(rgb, seg_ids, seg_types, allowed_geoms, geom_type_id, min_px=12):
    """Geom ids (with pixel counts) that show signal yellow in a frame and are not allowed to (tag plates, floor
    markings and hazard stripes are allowed). seg_*: the segmentation render of the same view."""
    m = yellow_mask(rgb) & (seg_types == geom_type_id)
    ids, cnt = np.unique(seg_ids[m], return_counts=True)
    allowed = set(int(i) for i in allowed_geoms)
    return {int(i): int(c) for i, c in zip(ids, cnt) if c >= min_px and int(i) not in allowed}
