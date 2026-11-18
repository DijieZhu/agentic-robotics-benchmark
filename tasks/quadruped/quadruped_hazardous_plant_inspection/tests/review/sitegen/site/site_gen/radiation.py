"""Radiation: sources (documented drums in the waste store, undocumented hot spots elsewhere), attenuating boxes
(walls, shield walls, the maze wall, temporary shielding), and the dose-rate field.

dose_rate(p) = background + sum_i S_i / max(|p - s_i|^2, r_min^2) * exp(-sum_k mu_k * L_k(p, s_i))
with S_i in uSv/h at 1 m and L_k the chord of the segment p -> s_i through box k (2-D, walls are full height).
This is a local STAND-IN with the formula of the design; field() uses bricks.sensors.radiation when that module
provides `dose_rate(points, sources, boxes)`."""

from __future__ import annotations

import math

import numpy as np

from .util import choice, uniform, weighted_key


def _bricks_radiation():
    try:
        from bricks.sensors import radiation as R  # noqa: WPS433
        return R
    except Exception:  # noqa: BLE001
        return None


def model_name():
    return "bricks.sensors.radiation.RadiationField" if _bricks_radiation() is not None else \
        "local stand-in (same formula)"


def choose_sources(W, cfg, rng):
    g = rng("radiation")
    C = cfg["radiation"]
    srcs = []
    drums = [s for s in W.sources_candidates if s["kind"] == "drum"]
    drums.sort(key=lambda s: tuple(s["pos"]))
    k = min(len(drums), int(g.integers(C["documented_drums"][0], C["documented_drums"][1] + 1)))
    for i, j in enumerate(sorted(g.choice(len(drums), size=k, replace=False).tolist()) if drums else []):
        d = drums[j]
        srcs.append({"id": "SRC%02d" % (len(srcs) + 1), "kind": "drum", "pos": d["pos"], "room": d["room"],
                     "strength": round(uniform(g, *C["drum_strength"]), 1), "documented": True, "text": d["text"]})
    n_u = int(g.integers(C["undocumented"][0], C["undocumented"][1] + 1))
    kinds_w = dict(C["hotspot_kinds"])
    for _ in range(n_u):
        for _try in range(10):
            kind = weighted_key(g, kinds_w)
            s = _undocumented(W, g, kind)
            if s is not None:
                kinds_w.pop(kind, None)
                if not kinds_w:
                    kinds_w = {"moved_drum": 1.0}
                break
        if s is None:
            continue
        s.update({"id": "SRC%02d" % (len(srcs) + 1), "strength": round(uniform(g, *C["hotspot_strength"]), 1),
                  "documented": False})
        srcs.append(s)
    return srcs


def _undocumented(W, g, kind):
    lay = W.lay
    if kind == "sump":
        c = [s for s in W.sources_candidates if s["kind"] == "sump"]
        if not c:
            return None
        s = c[0]
        return {"kind": "sump", "pos": list(s["pos"]), "room": s["room"], "text": "contaminated sludge in the %s" %
                s["text"]}
    if kind == "pipe_elbow":
        cands = []
        for p in W.P.prims:
            if p.type != "sphere" or "/e" not in p.name or p.group != 0:
                continue
            z = p.pos[2]
            r = lay.rooms.get(p.owner)
            if r is None or r.type not in ("pump_room", "valve_gallery", "tank_room"):
                continue
            if 0.3 < z < 1.0 and _far_from_slots(W, p.pos, 1.2):
                cands.append(p)
        if not cands:
            return None
        cands.sort(key=lambda p: p.name)
        p = cands[int(g.integers(len(cands)))]
        return {"kind": "pipe_elbow", "pos": list(p.pos), "room": p.owner, "text": "crud trap in a pipe elbow (%s)" %
                p.name.split("/")[1]}
    if kind == "moved_drum":
        # a single drum left against a wall in a corridor leg or a decor room
        cands = []
        for r in lay.rooms.values():
            if r.type in ("corridor",) or (r.type == "decor" and r.kind in ("store", "workshop", "decon_room")):
                cands.append(r)
        cands.sort(key=lambda r: r.id)
        for _ in range(30):
            r = cands[int(g.integers(len(cands)))]
            x0, y0, x1, y1 = r.inner
            along_x = (x1 - x0) >= (y1 - y0)
            if along_x:
                x = uniform(g, x0 + 1.0, x1 - 1.0)
                y = y0 + 0.36 if g.random() < 0.5 else y1 - 0.36
            else:
                y = uniform(g, y0 + 1.0, y1 - 1.0)
                x = x0 + 0.36 if g.random() < 0.5 else x1 - 0.36
            ok = True
            for d in lay.doors:
                dx, dy = d.xy()
                if math.hypot(dx - x, dy - y) < d.width / 2 + 1.3:
                    ok = False
            for kr in W.keepouts.get(r.id, []):
                if kr[0] - 0.4 <= x <= kr[2] + 0.4 and kr[1] - 0.4 <= y <= kr[3] + 0.4:
                    ok = False
            for e in W.elements.values():
                ex = [p[0] for p in e.footprint]
                ey = [p[1] for p in e.footprint]
                if min(ex) - 0.8 <= x <= max(ex) + 0.8 and min(ey) - 0.8 <= y <= max(ey) + 0.8:
                    ok = False
            if not ok or not _far_from_slots(W, (x, y), 1.2):
                continue
            body = "src/moved_drum"
            W.P.vcyl("%s/drum" % body, x, y, 0.0, 0.88, 0.285, mat="drum_yellow", body=body, owner=r.id)
            W.P.vcyl("%s/lid" % body, x, y, 0.88, 0.895, 0.29, mat="drum_lid", body=body, owner=r.id, group=1)
            W.P.vcyl("%s/band" % body, x, y, 0.42, 0.46, 0.29, mat="drum_band", body=body, owner=r.id, group=1)
            return {"kind": "moved_drum", "pos": [x, y, 0.45], "room": r.id,
                    "text": "a single drum left in %s" % r.name.lower()}
        return None
    return None


def _far_from_slots(W, pos, d):
    """An unlisted source must not sit next to an instrument slot: the work order gives slot positions."""
    return all(math.hypot(s["pos"][0] - pos[0], s["pos"][1] - pos[1]) >= d for s in W.slots)


def attenuators(W, cfg, extra=()):
    """2-D oriented boxes [(cx, cy, hx, hy, yaw, mu)] from the wall prims, the maze wall and partitions."""
    C = cfg["radiation"]
    out = []
    for p in W.P.prims:
        if not (p.body.startswith("wall/") or p.body.startswith("partition/")):
            if not any(s.get("prim") == p.name for s in W.shields):
                continue
        if p.group not in (0, 3) or p.type != "box":
            continue
        z0, z1 = p.zrange()
        if z0 > 1.0:
            continue
        R = p.R()
        yaw = math.atan2(R[1, 0], R[0, 0])
        th = 2 * min(p.size[0], p.size[1])
        shield = th > 0.45 or p.mat == "shield_concrete"
        out.append((p.pos[0], p.pos[1], p.size[0], p.size[1], yaw, float(C["mu_shield"] if shield else C["mu_wall"])))
    out += list(extra)
    return out


def _chords(P, s, boxes):
    """Total mu*L along segments P[i] -> s through the boxes (2-D). P: (N, 2)."""
    tot = np.zeros(len(P))
    for cx, cy, hx, hy, yaw, mu in boxes:
        c, sn = math.cos(yaw), math.sin(yaw)
        ax = (P[:, 0] - cx) * c + (P[:, 1] - cy) * sn
        ay = -(P[:, 0] - cx) * sn + (P[:, 1] - cy) * c
        bx = (s[0] - cx) * c + (s[1] - cy) * sn
        by = -(s[0] - cx) * sn + (s[1] - cy) * c
        dx, dy = bx - ax, by - ay
        t0 = np.zeros(len(P))
        t1 = np.ones(len(P))
        for a, d, h in ((ax, dx, hx), (ay, dy, hy)):
            with np.errstate(divide="ignore", invalid="ignore"):
                ta = (-h - a) / d
                tb = (h - a) / d
            lo = np.minimum(ta, tb)
            hi = np.maximum(ta, tb)
            par = np.abs(d) < 1e-12
            inside = np.abs(a) <= h
            lo = np.where(par, np.where(inside, -np.inf, np.inf), lo)
            hi = np.where(par, np.where(inside, np.inf, -np.inf), hi)
            t0 = np.maximum(t0, lo)
            t1 = np.minimum(t1, hi)
        L = np.maximum(0.0, t1 - t0) * np.hypot(dx, dy)
        tot += mu * L
    return tot


def dose_rate(points, sources, boxes, cfg, background=0.15, documented_only=False):
    """Dose rate (uSv/h) at points (N, 2) or (N, 3) (z defaults to the dosimeter height)."""
    P = np.asarray(points, float)
    if P.ndim == 1:
        P = P[None, :]
    z = P[:, 2] if P.shape[1] > 2 else np.full(len(P), float(cfg["radiation"]["height"]))
    rmin = float(cfg["radiation"]["r_min"])
    out = np.full(len(P), float(background))
    srcs = [s for s in sources if (s["documented"] or not documented_only)]
    R = _bricks_radiation()
    if R is not None:
        sh = []
        for cx, cy, hx, hy, yaw, mu in boxes:
            c_, s_ = math.cos(yaw), math.sin(yaw)
            Rm = np.array([[c_, -s_, 0.0], [s_, c_, 0.0], [0.0, 0.0, 1.0]])
            sh.append(R.ShieldBox((cx, cy, 1.3), (hx, hy, 1.4), float(mu), Rm))
        fld = R.RadiationField([R.PointSource(tuple(s["pos"]), float(s["strength"])) for s in srcs], sh,
                               background=float(background), r_min=rmin)
        return np.asarray(fld.dose_rate(np.c_[P[:, :2], z]), float)
    for s in srcs:
        sp = np.asarray(s["pos"], float)
        r2 = (P[:, 0] - sp[0]) ** 2 + (P[:, 1] - sp[1]) ** 2 + (z - sp[2]) ** 2
        att = _chords(P[:, :2], sp[:2], boxes)
        out += float(s["strength"]) / np.maximum(r2, rmin * rmin) * np.exp(-att)
    return out


def field_grid(bbox, res, sources, boxes, cfg, background, documented_only=False):
    x0, y0, x1, y1 = bbox
    xs = np.arange(x0 + res / 2, x1, res)
    ys = np.arange(y0 + res / 2, y1, res)
    X, Y = np.meshgrid(xs, ys)
    D = dose_rate(np.c_[X.ravel(), Y.ravel()], sources, boxes, cfg, background, documented_only)
    return xs, ys, D.reshape(X.shape)


def zone_class(rate, classes):
    for name, ub in classes:
        if rate <= float(ub):
            return name
    return classes[-1][0]
