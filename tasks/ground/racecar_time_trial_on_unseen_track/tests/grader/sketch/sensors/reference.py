"""An independent analytic ray caster for the sensor model's gates and tests: what a beam must hit, computed in numpy from the
numbers of a world file (and of the stub cars' LiDAR targets), not from the MuJoCo model.

Primitives: every duct piece as a capped cylinder (axis at z = 0.165 m, radius 0.165 m, flat ends);
clutter as an upright box or cylinder standing on the floor; avoidance boxes (variant 'avoid'); the
floor plane z = 0; optionally other cars (the contract's stub car: sensor head cylinder, hull box, tail box). The ego car is not
in the scene (the LiDAR excludes its own body).

    prims = scene_primitives(world, variant="track", cars={"slow0/": pose})
    dist, which = cast(prims, origin, dirs)     # dist inf = no surface; which = index into prims.items (-1 = none)
"""

import math

import numpy as np

from sketch.examples import constants as K

EPS = 1e-12
HULL_TOP_M = 0.10666            # the contract's stub car (greybox.stub_car): hull box up to the top of the compute unit
HEAD_Z_M = (0.098, 0.1217)      # sensor head cylinder, radius 0.0173


class Primitives(object):
    def __init__(self):
        self.cyl_a, self.cyl_b, self.cyl_r, self.cyl_item = [], [], [], []
        self.box_c, self.box_h, self.box_yaw, self.box_item = [], [], [], []
        self.items = []         # (kind, ident, reflectivity class); kind in duct, clutter, box, car, floor
        self.floor_item = None

    def add_cylinder(self, a, b, r, item):
        self.cyl_a.append(a), self.cyl_b.append(b), self.cyl_r.append(r), self.cyl_item.append(self._item(item))

    def add_box(self, centre, half, yaw, item):
        self.box_c.append(centre), self.box_h.append(half), self.box_yaw.append(yaw), self.box_item.append(self._item(item))

    def _item(self, item):
        self.items.append(item)
        return len(self.items) - 1

    def freeze(self):
        for k in ("cyl_a", "cyl_b", "box_c", "box_h"):
            setattr(self, k, np.array(getattr(self, k), dtype=float).reshape(-1, 3))
        for k in ("cyl_r", "box_yaw"):
            setattr(self, k, np.array(getattr(self, k), dtype=float))
        for k in ("cyl_item", "box_item"):
            setattr(self, k, np.array(getattr(self, k), dtype=int))
        return self


def _rot2(v, c, s):
    return np.stack([c * v[..., 0] - s * v[..., 1], s * v[..., 0] + c * v[..., 1]], axis=-1)


def add_stub_car(prims, prefix, pose, tail_box=True):
    """The LiDAR targets of the contract's stub car (greybox.stub_car) at a planar pose."""
    x, y, yaw = pose
    c, s = math.cos(yaw), math.sin(yaw)

    def world(px, py, pz):
        return [x + c * px - s * py, y + s * px + c * py, pz]

    hx, hz0 = 0.5 * (K.X_FRONT_M - K.X_REAR_M), K.GROUND_CLEARANCE_M
    prims.add_box(world(0.5 * (K.X_FRONT_M + K.X_REAR_M), 0.0, 0.5 * (HULL_TOP_M + hz0)),
                  [hx, 0.5 * K.ENVELOPE_M[1], 0.5 * (HULL_TOP_M - hz0)], yaw, ("car", prefix + "hull/lidar", "dark"))
    prims.add_cylinder(world(K.LIDAR_POS_C[0], 0.0, HEAD_Z_M[0]), world(K.LIDAR_POS_C[0], 0.0, HEAD_Z_M[1]), 0.0173,
                       ("car", prefix + "lidar_head/lidar", "dark"))
    if tail_box:
        sx, sy, sz = K.TAIL_BOX_SIZE_M
        prims.add_box(world(*K.TAIL_BOX_CENTRE_C), [0.5 * sx, 0.5 * sy, 0.5 * sz], yaw, ("car", prefix + "tail_box/lidar", "cardboard"))


def scene_primitives(world, variant="track", cars=None, floor=True):
    t = world["truth"]
    prims = Primitives()
    r = K.DUCT_RADIUS_M
    for d in t["ducts"]:
        h = 0.5 * d["piece_length_m"]
        for k, (x, y, yaw) in enumerate(d["pieces"]):
            ux, uy = math.cos(yaw), math.sin(yaw)
            prims.add_cylinder([x - h * ux, y - h * uy, r], [x + h * ux, y + h * uy, r], r, ("duct", (d["id"], k), d["wall_class"]))
    for c in t["clutter"]:
        lx, ly, lz = c["size_m"]
        x, y, yaw = c["pose"]
        item = ("clutter", c["id"], c["reflectivity"])
        if c["shape"] == "box":
            prims.add_box([x, y, 0.5 * lz], [0.5 * lx, 0.5 * ly, 0.5 * lz], yaw, item)
        else:
            prims.add_cylinder([x, y, 0.0], [x, y, lz], 0.5 * lx, item)
    if variant == "avoid":
        for b in t["avoidance"]["boxes"]:
            lx, ly, lz = b["size_m"]
            x, y, yaw = b["pose"]
            prims.add_box([x, y, 0.5 * lz], [0.5 * lx, 0.5 * ly, 0.5 * lz], yaw, ("box", b["id"], "cardboard"))
    for prefix, pose in (cars or {}).items():
        add_stub_car(prims, prefix, pose, tail_box=(prefix != K.EGO_PREFIX))
    if floor:
        prims.floor_item = prims._item(("floor", "floor", "floor"))
    return prims.freeze()


def _cylinders(o, d, a, b, r):
    """Nearest positive hit of rays (n) with capped cylinders (m): (n, m) distances, inf where none."""
    ba = b - a                                                   # (m, 3)
    oc = o[None, :] - a                                          # (m, 3)
    baba = np.einsum("ij,ij->i", ba, ba)                         # (m,)
    bard = d @ ba.T                                              # (n, m)
    baoc = np.einsum("ij,ij->i", ba, oc)                         # (m,)
    ocrd = d @ oc.T                                              # (n, m)
    ococ = np.einsum("ij,ij->i", oc, oc)
    k2 = baba[None, :] - bard ** 2
    k1 = baba[None, :] * ocrd - baoc[None, :] * bard
    k0 = baba * ococ - baoc ** 2 - (r ** 2) * baba
    out = np.full(bard.shape, np.inf)
    with np.errstate(divide="ignore", invalid="ignore"):
        disc = k1 ** 2 - k2 * k0[None, :]
        ok = (k2 > EPS) & (disc >= 0.0)
        t = (-k1 - np.sqrt(np.where(ok, disc, 0.0))) / np.where(ok, k2, 1.0)
        yb = baoc[None, :] + t * bard
        body = ok & (t > 0.0) & (yb > 0.0) & (yb < baba[None, :])
        out = np.where(body, t, out)
        for end in (0.0, 1.0):                                   # the two flat caps
            tc = (end * baba[None, :] - baoc[None, :]) / np.where(np.abs(bard) > EPS, bard, np.nan)
            q = oc[None, :, :] + tc[:, :, None] * d[:, None, :] - end * ba[None, :, :]
            capok = (tc > 0.0) & (np.einsum("nmk,nmk->nm", q, q) <= (r ** 2)[None, :])
            out = np.where(capok & (tc < out), tc, out)
    return out


def _boxes(o, d, c, half, yaw):
    """Nearest positive hit of rays (n) with boxes (m) that turn about z: (n, m) distances, inf where none."""
    cs, sn = np.cos(-yaw), np.sin(-yaw)
    rel = o[None, :] - c                                                     # (m, 3)
    o2 = np.concatenate([_rot2(rel[:, :2], cs, sn), rel[:, 2:]], axis=1)     # ray origin in each box frame (m, 3)
    dx = d[:, None, 0] * cs[None, :] - d[:, None, 1] * sn[None, :]
    dy = d[:, None, 0] * sn[None, :] + d[:, None, 1] * cs[None, :]
    dz = np.broadcast_to(d[:, None, 2], dx.shape)
    tmin = np.full(dx.shape, -np.inf)
    tmax = np.full(dx.shape, np.inf)
    with np.errstate(divide="ignore", invalid="ignore"):
        for ax, dd in ((0, dx), (1, dy), (2, dz)):
            inv = 1.0 / dd
            t1 = (-half[None, :, ax] - o2[None, :, ax]) * inv
            t2 = (half[None, :, ax] - o2[None, :, ax]) * inv
            lo, hi = np.minimum(t1, t2), np.maximum(t1, t2)
            par = np.abs(dd) < EPS
            inside = np.abs(o2[None, :, ax]) <= half[None, :, ax]
            lo = np.where(par, np.where(inside, -np.inf, np.inf), lo)
            hi = np.where(par, np.where(inside, np.inf, -np.inf), hi)
            tmin, tmax = np.maximum(tmin, lo), np.minimum(tmax, hi)
    hit = (tmax >= tmin) & (tmin > 0.0)
    return np.where(hit, tmin, np.inf)


def cast(prims, origin, dirs):
    """(dist (n,), which (n,)): distance to the first surface along each ray from `origin` and the index of its item
    in `prims.items` (-1 when the ray meets nothing)."""
    o = np.asarray(origin, dtype=float)
    d = np.asarray(dirs, dtype=float).reshape(-1, 3)
    n = len(d)
    cand = [np.full((n, 1), np.inf)]
    owner = [np.array([-1])]
    if len(prims.cyl_r):
        cand.append(_cylinders(o, d, prims.cyl_a, prims.cyl_b, prims.cyl_r))
        owner.append(prims.cyl_item)
    if len(prims.box_yaw):
        cand.append(_boxes(o, d, prims.box_c, prims.box_h, prims.box_yaw))
        owner.append(prims.box_item)
    if prims.floor_item is not None:
        with np.errstate(divide="ignore", invalid="ignore"):
            tf = np.where(d[:, 2] < -EPS, -o[2] / d[:, 2], np.inf)
        cand.append(np.where(tf > 0.0, tf, np.inf)[:, None])
        owner.append(np.array([prims.floor_item]))
    allc = np.concatenate(cand, axis=1)
    allo = np.concatenate(owner)
    j = np.argmin(allc, axis=1)
    dist = allc[np.arange(n), j]
    which = np.where(np.isfinite(dist), allo[j], -1)
    return dist, which


def second_gap(prims, origin, dirs):
    """Per ray, the distance between the nearest and the second-nearest surface (to skip ambiguous beams in tests)."""
    o = np.asarray(origin, dtype=float)
    d = np.asarray(dirs, dtype=float).reshape(-1, 3)
    cand = [np.full((len(d), 2), np.inf)]
    if len(prims.cyl_r):
        cand.append(_cylinders(o, d, prims.cyl_a, prims.cyl_b, prims.cyl_r))
    if len(prims.box_yaw):
        cand.append(_boxes(o, d, prims.box_c, prims.box_h, prims.box_yaw))
    allc = np.sort(np.concatenate(cand, axis=1), axis=1)
    with np.errstate(invalid="ignore"):
        return np.where(np.isfinite(allc[:, 0]), allc[:, 1] - allc[:, 0], np.inf)


def floor_return_p(world, params, origin, dirs, dist, which, prims):
    """Per beam, the return probability the floor rule gives a floor hit (0 for beams that do not meet the floor),
    with the contract's reference floor lookup `check.FloorMap.zone_at` at the hit point."""
    from sketch.examples import check
    fm = check.FloorMap(world)
    kinds = {p["id"]: p["kind"] for p in world["truth"]["floor"]["patches"]}
    base = world["truth"]["floor"]["kind"]
    lid = params["lidar"]
    d = np.asarray(dirs, dtype=float).reshape(-1, 3)
    out = np.zeros(len(dist))
    for i in np.nonzero(which == prims.floor_item)[0]:
        x, y = (np.asarray(origin) + dist[i] * d[i])[:2]
        zid = fm.zone_at(float(x), float(y))
        steep = abs(d[i][2]) >= math.sin(lid["floor_shallow_max_rad"])      # the floor's normal is +z
        out[i] = 1.0 if steep else lid["floor_return_prob"][base if zid is None else kinds[zid]]
    return out


def expected_mm(prims, dist, which, params, floor_p=None):
    """The noise-free wire value of each beam under the LiDAR rules, for parameters without fade and with floor return
    probabilities of 0 or 1 (a deterministic expectation). `floor_p`: per-beam floor probabilities (floor_return_p)."""
    lid = params["lidar"]
    if lid["range_fade_frac"] != 0.0:
        raise ValueError("expected_mm needs range_fade_frac 0")
    out = np.zeros(len(dist), dtype=np.int64)
    for i, (r, w) in enumerate(zip(dist, which)):
        if w < 0 or not np.isfinite(r) or r < K.RANGE_MIN_M or r > K.RANGE_MAX_M:
            continue
        kind, _, cls = prims.items[w]
        if kind == "floor":
            p = 0.0 if floor_p is None else float(floor_p[i])
            if p not in (0.0, 1.0):
                raise ValueError("expected_mm needs floor return probabilities of 0 or 1")
            ok = p == 1.0 and r <= lid["floor_range_m"]
        else:
            ok = r <= lid["range_by_class_m"][cls]
        if ok:
            out[i] = int(np.clip(np.rint(r * 1000.0), 60, 10000))
    return out
