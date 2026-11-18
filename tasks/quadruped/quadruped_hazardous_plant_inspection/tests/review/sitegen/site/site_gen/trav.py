"""Ground-truth traversability: a 0.05 m grid with, per cell, the modes that can enter it, and a search over
(cell, mode) states with the robot's footprint.

Per mode m (walk, high_step, crouch_walk) a primitive is *solid* when it intersects the mode's body envelope:
  generic geoms: z_min < clearance(m) and z_max > step(m);
  traversal elements: by their class (a needs_high_step sill is solid for walk and crouch_walk only, a
  needs_crouch overhead is solid for walk and high_step only, a barrier for all, free for none);
  walkable surfaces (ramps, puddles) are never solid and mark the cells slow when their class is slow.
The footprint: a cell is enterable when its distance to the nearest solid cell is at least r_in (the body's
half width plus the barrier/tight margin); below r_circ (the half diagonal) it is 'tight' (no turning, slow).
Costs are times: cell distance over the mode's speed (times the slow / tight factors), plus a mode-switch time.
"""

from __future__ import annotations

import json

import numpy as np
from scipy import ndimage

from .config import derived_thresholds
from .loader.travmap import MODES, NB8, Grid, _min_width, _signed_area  # noqa: F401  (re-exported)
from .loader.travmap import TravMap as _StoredTravMap


def _hull(pts):
    pts = sorted(set((round(float(x), 5), round(float(y), 5)) for x, y in pts))
    if len(pts) <= 2:
        return pts

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


class TravMap(_StoredTravMap):
    """The traversability grid of a site, rasterised from the world's primitives (the generator's half); the
    searches and queries are the loader's (site_gen.loader.travmap.TravMap)."""

    def __init__(self, grid: Grid, cfg):
        self.cfg = cfg
        th = derived_thresholds(cfg)
        M = cfg["modes"]
        super().__init__(grid, {"r_in": th["r_in"], "r_circ": th["r_circ"], "switch_time_s": float(M["switch_time_s"])})
        self.th = th
        self.speed_base = {m: float(M[m]["speed"]) for m in MODES}
        self.slow_factor = float(M["slow_factor"])
        self.tight_factor = float(M["tight_factor"])

    # ------------------------------------------------------------------------------------------------ raster
    def rasterize(self, prims, elements, room_rects, extra_solid_polys=()):
        g = self.g
        th = self.th
        sh = g.shape
        solid = {m: np.zeros(sh, np.uint8) for m in MODES}
        slow = np.zeros(sh, np.uint8)
        hsreq = np.zeros(sh, np.uint8)     # cells whose obstacle needs high_step (for stats / colouring)
        crreq = np.zeros(sh, np.uint8)
        elem = np.full(sh, -1, np.int32)
        inside = np.zeros(sh, np.uint8)
        for r in room_rects:
            g.fill_poly(inside, [[r[0], r[1]], [r[2], r[1]], [r[2], r[3]], [r[0], r[3]]], 1)
        eids = sorted(elements.keys())
        eidx = {e: i for i, e in enumerate(eids)}
        step = {"walk": th["step_free"], "high_step": th["step_high"], "crouch_walk": th["step_free"]}
        # ladder mode: generic geometry (not a traversal element) taller than the rated high-step range is solid for
        # every mode, so no size between that range and a barrier lies on any access
        gmax = (self.cfg.get("ladder") or {}).get("generic_high_step_max") if \
            str(self.cfg.get("layout", "draft")).startswith("ladder") else None
        step_generic = dict(step, high_step=min(th["step_high"], float(gmax))) if gmax else step
        clear = {"walk": th["clear_walk"], "high_step": th["clear_walk"], "crouch_walk": th["clear_crouch"]}
        for p in prims:
            if p.group not in (0, 3):
                continue
            c = p.corners()
            poly = _hull(c[:, :2])
            z0, z1 = float(c[:, 2].min()), float(c[:, 2].max())
            if z1 <= 0.0005:
                continue
            e = elements.get(p.elem) if p.elem else None
            if p.role == "walkable":
                cls = e.cls if e is not None else "free"
                if cls == "slow":
                    g.fill_poly(slow, poly, 1)
                elif cls == "barrier":
                    for m in MODES:
                        g.fill_poly(solid[m], poly, 1)
                if e is not None:
                    g.fill_poly(elem, poly, eidx[e.id])
                continue
            if e is not None:
                cls = e.cls
                hit = {"free": (), "needs_high_step": ("walk", "crouch_walk"), "needs_crouch": ("walk", "high_step"),
                       "barrier": MODES, "tight": MODES, "slow": ()}.get(cls, MODES)
                if cls in ("tight",):
                    hit = tuple(m for m in MODES if z0 < clear[m] and z1 > step[m])
                for m in hit:
                    g.fill_poly(solid[m], poly, 1)
                if cls == "needs_high_step":
                    g.fill_poly(hsreq, poly, 1)
                if cls == "needs_crouch":
                    g.fill_poly(crreq, poly, 1)
                g.fill_poly(elem, poly, eidx[e.id])
                continue
            for m in MODES:
                if z0 < clear[m] and z1 > step_generic[m]:
                    g.fill_poly(solid[m], poly, 1)
        # zone elements (debris fields): the footprint carries the class
        for e in elements.values():
            if not e.zone:
                continue
            if e.cls == "needs_high_step":
                for m in ("walk", "crouch_walk"):
                    g.fill_poly(solid[m], e.footprint, 1)
                g.fill_poly(hsreq, e.footprint, 1)
            elif e.cls == "barrier":
                for m in MODES:
                    g.fill_poly(solid[m], e.footprint, 1)
            elif e.cls == "slow":
                g.fill_poly(slow, e.footprint, 1)
        for poly in extra_solid_polys:
            for m in MODES:
                g.fill_poly(solid[m], poly, 1)
        out = inside == 0
        self.inside = inside.astype(bool)
        self.solid = {m: (solid[m].astype(bool) | out) for m in MODES}
        self.slow = slow.astype(bool) & ~out
        self.hsreq = hsreq.astype(bool)
        self.crreq = crreq.astype(bool)
        self.elem = elem
        self.elem_ids = eids
        self._derive()

    def _derive(self):
        th = self.th
        res = self.g.res
        self.clear = {}
        self.enter = {}
        self.tight = {}
        self.speed = {}
        for m in MODES:
            d = ndimage.distance_transform_edt(~self.solid[m])
            cl = (d - 0.5) * res
            self.clear[m] = cl
            en = (~self.solid[m]) & (cl >= th["r_in"]) & self.inside
            self.enter[m] = en
            ti = en & (cl < th["r_circ"])
            self.tight[m] = ti
            v = np.where(en, self.speed_base[m], 0.0)
            v = np.where(self.slow, v * self.slow_factor, v)
            v = np.where(ti, v * self.tight_factor, v)
            self.speed[m] = v.astype(np.float32)

    def stored_params(self) -> dict:
        """What the loader needs besides the arrays (written into trav.npz)."""
        return {"r_in": float(self.th["r_in"]), "r_circ": float(self.th["r_circ"]), "switch_time_s": self.switch_time,
                "modes": list(MODES), "speed_base": self.speed_base, "slow_factor": self.slow_factor,
                "tight_factor": self.tight_factor}

    def save(self, path):
        np.savez_compressed(path, **self.arrays(), params=np.array(json.dumps(self.stored_params(), sort_keys=True)))

    @classmethod
    def load(cls, path, cfg):
        z = np.load(path, allow_pickle=False)
        res = float(z["res"])
        x0, y0 = [float(v) for v in z["origin"]]
        ny, nx = z["enter_walk"].shape
        g = Grid.exact(x0, y0, res, (ny, nx))
        t = cls(g, cfg)
        t._set_arrays(z)
        return t
