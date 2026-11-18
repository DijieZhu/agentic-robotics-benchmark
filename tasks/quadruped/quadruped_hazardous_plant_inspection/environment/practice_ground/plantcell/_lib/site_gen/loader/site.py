"""Ground-truth queries of a stored site (privileged: the reference program and the grader side only).

    site = Site(open_site(path))
    site.modes_at(x, y); site.cell_info(x, y); site.plan((x0, y0), (x1, y1))
    site.readable_mask(tag, mode); site.is_readable(tag, x, y, mode); site.observation_pose(tag)
    site.link_ok(x, y); site.dose_rate(x, y); site.dose_rate_grid()

readable_region masks and observation poses are BODY-centre positions on ladder sites of schema 0.3 and later
(the camera 0.18 m ahead of the body toward the instrument, at the robot's camera height): see
site.json ips[].readable_region.semantics.
"""

from __future__ import annotations

import math

import numpy as np

from .package import open_site
from .travmap import MODES, TravMap


def _mask_from_runs(runs, shape):
    m = np.zeros(tuple(shape), bool)
    for iy, a, b in runs:
        m[iy, a:b + 1] = True
    return m


class Site:
    def __init__(self, pkg):
        self.pkg = open_site(pkg)
        self.dir = self.pkg.source
        self.site = self.pkg.site
        self.prior = self.pkg.prior
        self._trav = None
        self._graph = {}
        self._masks = {}

    # ------------------------------------------------------------------ records
    @property
    def ips(self):
        return self.site["ips"]

    def ip(self, tag):
        for i in self.site["ips"]:
            if i["tag"] == tag:
                return i
        raise KeyError(tag)

    def room_at(self, x, y):
        best = None
        for r in self.site["rooms"]:
            x0, y0, x1, y1 = r["rect"]
            if x0 <= x <= x1 and y0 <= y <= y1:
                if best is None or (x1 - x0) * (y1 - y0) < (best["rect"][2] - best["rect"][0]) * \
                        (best["rect"][3] - best["rect"][1]):
                    best = r
        return best

    @property
    def route(self):
        return self.site["route"]

    # ------------------------------------------------------------------ traversability
    @property
    def trav(self) -> TravMap:
        if self._trav is None:
            self._trav = TravMap.from_arrays(self.pkg.trav_arrays, self.pkg.trav_params)
        return self._trav

    def modes_at(self, x, y):
        return self.trav.modes_at(x, y)

    def cell_info(self, x, y):
        t = self.trav
        iy, ix = t.g.cell(x, y)
        e = int(t.elem[iy, ix]) if 0 <= iy < t.g.ny and 0 <= ix < t.g.nx else -1
        inside = 0 <= iy < t.g.ny and 0 <= ix < t.g.nx
        return {"modes": t.modes_at(x, y), "slow": bool(t.slow[iy, ix]) if inside else False,
                "tight": {m: bool(t.tight[m][iy, ix]) if inside else False for m in MODES},
                "element": t.elem_ids[e] if e >= 0 else None}

    def plan(self, start, goal, modes=MODES, start_mode="walk"):
        """Cheapest (time) path on the truth map from start (x, y) to goal (x, y) in any allowed mode.
        Returns {'time_s', 'segments': [{'mode', 'waypoints'}]} or None."""
        t = self.trav
        key = tuple(modes)
        if key not in self._graph:
            self._graph[key] = t.build_graph(modes)
        G = self._graph[key]
        sc = t.nearest_enterable(start[0], start[1], start_mode)
        if sc is None:
            return None
        N = t.g.nx * t.g.ny
        s = MODES.index(start_mode) * N + sc[0] * t.g.nx + sc[1]
        D, P = t.dijkstra(G, [s])
        goals = []
        for m in modes:
            gc = t.nearest_enterable(goal[0], goal[1], m)
            if gc is not None:
                n = MODES.index(m) * N + gc[0] * t.g.nx + gc[1]
                goals.append((D[0, n], n))
        goals = [gg for gg in goals if np.isfinite(gg[0])]
        if not goals:
            return None
        cost, n = min(goals)
        path = t.path(P[0], s, n)
        segs = []
        for node in path:
            iy, ix, m = t.unnode(node)
            xy = list(t.g.xy(iy, ix))
            if not segs or segs[-1]["mode"] != m:
                segs.append({"mode": m, "waypoints": [xy] if not segs else [segs[-1]["waypoints"][-1], xy]})
            else:
                segs[-1]["waypoints"].append(xy)
        return {"time_s": float(cost), "segments": segs}

    # ------------------------------------------------------------------ readability
    def readable_mask(self, tag, mode="walk"):
        k = (tag, "crouch" if mode == "crouch_walk" else "stand")
        if k not in self._masks:
            rr = self.ip(tag)["readable_region"]
            shape = rr["grid"]["shape"]
            self._masks[k] = _mask_from_runs(rr["runs_crouch" if k[1] == "crouch" else "runs_stand"], shape)
        return self._masks[k]

    def readable_semantics(self, tag) -> str:
        """'body' (the body centre stands in the cell) or 'camera' (older sites: the camera is at the cell)."""
        return str(self.ip(tag)["readable_region"].get("semantics", "camera"))

    def is_readable(self, tag, x, y, mode="walk"):
        """True when the margin acceptance region of `tag` holds (x, y) for the mode's posture."""
        rr = self.ip(tag)["readable_region"]
        g = rr["grid"]
        x0, y0 = g["origin"]
        iy, ix = int(math.floor((y - y0) / g["res"])), int(math.floor((x - x0) / g["res"]))
        m = self.readable_mask(tag, mode)
        return bool(0 <= iy < m.shape[0] and 0 <= ix < m.shape[1] and m[iy, ix])

    def observation_pose(self, tag):
        return self.ip(tag)["observation_pose"]

    # ------------------------------------------------------------------ link and dose
    @property
    def fields(self):
        return self.pkg.fields

    def link_ok(self, x, y):
        f = self.fields
        x0, y0 = [float(v) for v in f["link_origin"]]
        res = float(f["link_res"])
        iy, ix = int((y - y0) / res), int((x - x0) / res)
        L = f["link"]
        return bool(0 <= iy < L.shape[0] and 0 <= ix < L.shape[1] and L[iy, ix])

    def dose_rate(self, x, y, z=None):
        """Dose rate (site units, uSv/h) at a point, from the site's sources and attenuating boxes."""
        from .dose import dose_rate
        p = [[x, y]] if z is None else [[x, y, z]]
        return float(dose_rate(self.site, p)[0])

    def dose_rate_grid(self, documented_only=False):
        f = self.fields
        if not documented_only:
            return f["dose_xs"], f["dose_ys"], f["dose_truth"]
        if "dose_documented" in f:
            return f["dose_xs"], f["dose_ys"], f["dose_documented"]
        from .dose import dose_rate
        X, Y = np.meshgrid(f["dose_xs"], f["dose_ys"])
        D = dose_rate(self.site, np.c_[X.ravel(), Y.ravel()], documented_only=True).reshape(X.shape)
        return f["dose_xs"], f["dose_ys"], D.astype(np.float32)


def load_site(path) -> Site:
    return Site(path)


def load_prior(path) -> dict:
    return open_site(path).prior
