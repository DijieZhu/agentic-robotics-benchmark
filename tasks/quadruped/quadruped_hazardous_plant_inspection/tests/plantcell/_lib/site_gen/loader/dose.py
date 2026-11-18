"""The site's dose-rate field from its record (site.json radiation and sources): the formula of the generator
(site_gen.radiation.dose_rate), with the bricks RadiationField when the bricks package is importable.

    rate(p) = background + sum_i S_i / max(|p - s_i|^2, r_min^2) * exp(-sum_k mu_k L_k(p, s_i))
Units are the site's (uSv/h); the design units (Gy/h) are site units x site.ladder.dose.unit_factor.
"""

from __future__ import annotations

import math

import numpy as np


def _chords_2d(P, s, boxes):
    """Total mu*L along segments P[i] -> s through the boxes (2-D; walls are full height). P: (N, 2).
    The same code as site_gen.radiation._chords (used only when bricks is not importable)."""
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


def dose_rate(site: dict, points, documented_only=False, only=None, background=None) -> np.ndarray:
    """Dose rate (site units) at points (N, 2) or (N, 3); z defaults to the dosimeter height of the record."""
    r = site["radiation"]
    P = np.asarray(points, float)
    if P.ndim == 1:
        P = P[None, :]
    z = P[:, 2] if P.shape[1] > 2 else np.full(len(P), float(r.get("height", 0.30)))
    rmin = float(r.get("r_min", 0.25))
    bg = float(r["background_uSv_h"]) if background is None else float(background)
    boxes = [tuple(b) for b in r["boxes"]]
    srcs = [s for s in site["sources"] if (s["documented"] or not documented_only) and
            (only is None or s["id"] in only)]
    try:
        from bricks.sensors import radiation as R  # noqa: WPS433
    except Exception:  # noqa: BLE001
        R = None
    if R is not None:
        sh = []
        for cx, cy, hx, hy, yaw, mu in boxes:
            c_, s_ = math.cos(yaw), math.sin(yaw)
            Rm = np.array([[c_, -s_, 0.0], [s_, c_, 0.0], [0.0, 0.0, 1.0]])
            sh.append(R.ShieldBox((cx, cy, 1.3), (hx, hy, 1.4), float(mu), Rm))
        fld = R.RadiationField([R.PointSource(tuple(s["pos"]), float(s["strength"])) for s in srcs], sh,
                               background=bg, r_min=rmin)
        return np.asarray(fld.dose_rate(np.c_[P[:, :2], z]), float)
    out = np.full(len(P), bg)
    for s in srcs:
        sp = np.asarray(s["pos"], float)
        r2 = (P[:, 0] - sp[0]) ** 2 + (P[:, 1] - sp[1]) ** 2 + (z - sp[2]) ** 2
        out += float(s["strength"]) / np.maximum(r2, rmin * rmin) * np.exp(-_chords_2d(P[:, :2], sp[:2], boxes))
    return out
