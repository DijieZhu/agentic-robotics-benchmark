"""The scored region and its zones.

    region = Region(region_json)            # .inside(xy), .zone_of(xy), .zone_ids, .height_cap
    Z = allocate(truth, twin, solids, matching, simmap, cfg)        # Zones
    mean_over_zones({zone: value or None})

A true unit's zone is its items_truth.json "zone" (the self-test checks that its footprint centroid lies in that
zone). A submitted unit takes its pair's zone; an unpaired one (the gate's excluded units included: they still count
as submitted) the zone holding its footprint centroid after the alignment map; outside the region it is ignored
(not counted). A unit matched to an 'ignore' truth entry is removed from every count. A point in the region but in no
zone polygon gets the zone whose polygon is nearest (the truth's zones are meant to cover the region).
"""
import numpy as np

from . import geom2d
from .truth import TruthError


class Region:
    def __init__(self, d):
        self.polygons = [np.asarray(p, float).reshape(-1, 2) for p in d.get("polygons", [])]
        self.exclusions = [np.asarray(p, float).reshape(-1, 2) for p in d.get("exclusions", [])]
        self.zones = [(str(z["id"]), str(z.get("name", z["id"])), np.asarray(z["polygon"], float).reshape(-1, 2))
                      for z in d.get("zones", [])]
        self.zone_ids = [z[0] for z in self.zones]
        self.height_cap = float(d.get("height_cap_m", 2.2))
        if not self.polygons or not self.zones:
            raise TruthError("region.json needs at least one polygon and one zone")
        if len(set(self.zone_ids)) != len(self.zone_ids):
            raise TruthError("region.json repeats a zone id")

    def inside(self, xy):
        P = np.atleast_2d(np.asarray(xy, float))[:, :2]
        ins = np.zeros(len(P), bool)
        for poly in self.polygons:
            ins |= geom2d.points_in_polygon(P, poly)
        for poly in self.exclusions:
            ins &= ~geom2d.points_in_polygon(P, poly)
        return ins

    def zone_of(self, xy):
        """The zone id of each point (None outside the region)."""
        P = np.atleast_2d(np.asarray(xy, float))[:, :2]
        ins = self.inside(P)
        out = [None] * len(P)
        for k in range(len(P)):
            if not ins[k]:
                continue
            for zid, _nm, poly in self.zones:
                if geom2d.points_in_polygon(P[k:k + 1], poly)[0]:
                    out[k] = zid
                    break
            if out[k] is None:
                d = [float(geom2d.polygon_distance(P[k:k + 1], poly)[0]) for _z, _n, poly in self.zones]
                out[k] = self.zones[int(np.argmin(d))][0]
        return out


class Zones:
    def __init__(self, ids):
        self.ids = list(ids)
        self.of_true, self.of_agent, self.how = {}, {}, {}
        self.n_true = {z: 0 for z in self.ids}
        self.n_sub = {z: 0 for z in self.ids}
        self.outside_region, self.removed = [], []
        self.centroids = {}

    def summary(self):
        return {"ids": self.ids, "of_true": self.of_true, "of_agent": self.of_agent, "how": self.how,
                "n_true": self.n_true, "n_sub": self.n_sub, "outside_region": self.outside_region,
                "removed": self.removed, "centroids": self.centroids}


def allocate(truth, twin, solids, matching, simmap, cfg):
    from . import surfaces
    Z = Zones(truth.region.zone_ids)
    for u in truth.counted_units():
        Z.of_true[u.id] = u.zone
        Z.n_true[u.zone] += 1
    removed = set(matching.removed_agents)
    for u in twin.units:
        b = u["body"]
        if b in removed:
            Z.removed.append(b)
            Z.how[b] = "ignore"
            continue
        if b in matching.a2t:
            z = Z.of_true[matching.a2t[b]]
            Z.how[b] = "pair"
        else:
            S = solids.get(b)
            if S is None and u["body_id"] >= 0:
                S = surfaces.unit_solid(twin, u, simmap, cfg)
            if S is not None:
                c = S.footprint_centroid()
            elif u["body_id"] >= 0:
                c = simmap.to_site(twin.body_point(u["body_id"])[None])[0, :2]
            else:
                c = None
            z = truth.region.zone_of(c[None])[0] if c is not None else None
            if c is not None:
                Z.centroids[b] = [float(c[0]), float(c[1])]
            if z is None:
                Z.outside_region.append(b)
                Z.how[b] = "outside"
                continue
            Z.how[b] = "centroid"
        Z.of_agent[b] = z
        Z.n_sub[z] += 1
    return Z


def mean_over_zones(values):
    """The mean of {zone: value} over the zones whose value is not None; 0 when none is left."""
    v = [float(x) for x in values.values() if x is not None]
    return sum(v) / len(v) if v else 0.0
