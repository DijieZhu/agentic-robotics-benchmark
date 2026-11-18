"""The ego car's 2D LiDAR.

One revolution per tick: 1081 beams over 270 degrees (beam i at -135 + 0.25 i degrees in the LiDAR frame L), cast as
five sub-scans (beams 216 k .. 216 k + 215, the last 217) right after physics steps (2k + 1) u of the tick, each from
the site `<prefix>lidar` at the state after that step: the true pitched, rolled and moving sensor pose (frames.py).
Each sub-scan is ONE `mujoco.mj_multiRay` call; there is no Python loop over rays.

What a beam returns (`Lidar.finish`), in this order:
  1. the first surface along the beam among LiDAR targets (groups 2 and 4) and the floor (group 3), within 10 m;
  2. a target returns when its true range r is within the range of its reflectivity class R (`range_by_class_m`;
     duct pieces by the `wall_class` of their duct, clutter by its `reflectivity`, boxes and tail boxes as cardboard,
     a car's sensor head and hull as dark); between R and R (1 + `range_fade_frac`) with a linearly falling
     probability; beyond, never;
  3. a floor hit (the floor plane or a raised patch slab) at a grazing angle below `floor_shallow_max_rad` returns
     with `floor_return_prob` of the floor kind under the hit point (the last listed patch containing it, else the
     venue's floor kind): 0 on hard floors, a probability on carpet; a steeper floor hit returns like a target;
  4. a returned range gets Gaussian noise of `noise_sigma_m`, is rounded to integer mm and clipped to 60 .. 10000;
     anything else is 0 (no return), as is every true range outside 0.06 .. 10 m.
The random numbers are drawn per tick in a fixed amount (1081 normals, then 1081 uniforms) whatever the scene, so the
stream does not depend on what the beams hit.

Known simplifications: the geoms of the world are taken as MuJoCo placed them in the last
forward pass (a moving slow car lags one physics step, at most 2.5 ms); no incidence-angle effect on range, no mixed
pixels at edges, no multipath from floor reflections, no dropout beyond the rules above.
"""

import math
import re

import mujoco
import numpy as np

from sketch.examples import constants as K

from .frames import ChassisState
from .params import FLOOR_RETURN_KINDS, TARGET_CLASSES

FLOOR_CLASS = len(TARGET_CLASSES)
NO_TARGET = -1
_DUCT = re.compile(r"^duct/(%s)/[0-9]+/lidar$" % K.ID_RE)
_CLUTTER = re.compile(r"^clutter/(%s)/lidar$" % K.ID_RE)
_BOX = re.compile(r"^box/%s/lidar$" % K.ID_RE)
_CAR = re.compile(r"^(?:car/|slow[0-9]/)(lidar_head|hull|tail_box)/lidar$")
_FLOOR = re.compile(r"^(?:floor|zone/%s/slab)$" % K.ID_RE)
_CAR_CLASS = {"lidar_head": "dark", "hull": "dark", "tail_box": "cardboard"}


def beam_directions():
    """Unit vectors of the 1081 beams in L (scan plane z = 0), shape (1081, 3)."""
    a = K.ANGLE_MIN_RAD + K.ANGLE_INC_RAD * np.arange(K.N_BEAMS)
    return np.ascontiguousarray(np.stack([np.cos(a), np.sin(a), np.zeros(K.N_BEAMS)], axis=1))


def points_in_polygon(x, y, poly):
    """Vectorised crossing test with the arithmetic of `check.point_in_polygon` (same half-open edge rule)."""
    inside = np.zeros(np.shape(x), dtype=bool)
    n = len(poly)
    for i in range(n):
        x1, y1 = float(poly[i][0]), float(poly[i][1])
        x2, y2 = float(poly[(i + 1) % n][0]), float(poly[(i + 1) % n][1])
        cond = (y1 > y) != (y2 > y)
        if y2 != y1:
            inside ^= cond & (x < (x2 - x1) * (y - y1) / (y2 - y1) + x1)
    return inside


def geom_target_classes(model, world):
    """Class index of every geom: 0 .. 5 = TARGET_CLASSES, 6 = floor, -1 = never a LiDAR target.

    The class comes from the geom's NAME, whatever group it is in now: a slow car staged in group 5 is
    classified already, so it reads right after `SlowCars.enter()` moves its geoms back into group 2.
    Raises ValueError for a geom in a LiDAR or floor group whose name the contract does not know, for a duct or clutter
    id that is not in the world file, and for a geom in a LiDAR group that `mj_multiRay` cannot hit (alpha 0)."""
    t = world["truth"]
    wall_class = {d["id"]: d["wall_class"] for d in t["ducts"]}
    reflect = {c["id"]: c["reflectivity"] for c in t["clutter"]}
    out = np.full(model.ngeom, NO_TARGET, dtype=np.int16)
    idx = {c: i for i, c in enumerate(TARGET_CLASSES)}
    for g in range(model.ngeom):
        group = int(model.geom_group[g])
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
        if _FLOOR.match(name):
            out[g] = FLOOR_CLASS
            continue
        m, mc, mb, mcar = _DUCT.match(name), _CLUTTER.match(name), _BOX.match(name), _CAR.match(name)
        if m:
            if m.group(1) not in wall_class:
                raise ValueError("geom %r: duct %r is not in the world file" % (name, m.group(1)))
            cls = wall_class[m.group(1)]
        elif mc:
            if mc.group(1) not in reflect:
                raise ValueError("geom %r: clutter %r is not in the world file" % (name, mc.group(1)))
            cls = reflect[mc.group(1)]
        elif mb:
            cls = "cardboard"
        elif mcar:
            cls = _CAR_CLASS[mcar.group(1)]
        elif group == K.GROUP_FLOOR:
            raise ValueError("geom %r is in the floor group but is not the floor or a zone slab" % name)
        elif group in (K.GROUP_LIDAR, K.GROUP_CLUTTER):
            raise ValueError("geom %r is a LiDAR target (group %d) without a contract name" % (name, group))
        else:
            continue
        if group in (K.GROUP_LIDAR, K.GROUP_CLUTTER):
            matid = int(model.geom_matid[g])
            alpha = float(model.mat_rgba[matid][3]) if matid >= 0 else float(model.geom_rgba[g][3])
            if alpha == 0.0:
                raise ValueError("geom %r is a LiDAR target with alpha 0: mj_multiRay skips invisible geoms" % name)
        out[g] = idx[cls]
    return out


class Lidar(object):
    """Sub-scan casting and the beam rules of one car's LiDAR. Driven by `SensorSuite`; usable alone in tests:

        lidar.cast(k, data) for k = 0 .. 4 (at the right physics steps), then lidar.finish(rng) -> 1081 ints
    """

    def __init__(self, model, world, params, prefix=K.EGO_PREFIX, floor_rule=True):
        self.prefix = prefix
        self.p = params["lidar"]
        self.floor_rule = bool(floor_rule)
        self.local = beam_directions()
        self.beam_sub = np.repeat(np.arange(len(K.SUBSCAN_SLICES)), [b - a for a, b in K.SUBSCAN_SLICES])
        self.cutoff = float(K.RANGE_MAX_M)
        rbc = self.p["range_by_class_m"]
        self.class_range = np.array([float(rbc[c]) for c in TARGET_CLASSES] + [float(self.p["floor_range_m"])])
        self.fade = float(self.p["range_fade_frac"])
        self.sigma = float(self.p["noise_sigma_m"])
        self.sin_shallow = math.sin(float(self.p["floor_shallow_max_rad"]))
        frp = self.p["floor_return_prob"]
        self.floor_prob = np.array([float(frp[k]) for k in FLOOR_RETURN_KINDS])
        kidx = {k: i for i, k in enumerate(FLOOR_RETURN_KINDS)}
        floor = world["truth"]["floor"]
        self.base_kind = kidx[floor["kind"]]
        self.patches = [(kidx[p["kind"]], [tuple(map(float, v)) for v in p["polygon"]]) for p in floor["patches"]]
        n = K.N_BEAMS
        self.dist = np.full(n, -1.0)
        self.geomid = np.full(n, -1, dtype=np.int32)
        self.normal = np.zeros(3 * n)
        self.vec = np.zeros((n, 3))
        self.origin = np.zeros((len(K.SUBSCAN_SLICES), 3))
        self.cast_done = [False] * len(K.SUBSCAN_SLICES)
        self.last = None
        self.bind(model, world)

    def bind(self, model, world):
        """(Re)bind to a compiled model of the same venue (a model per session variant)."""
        self.model = model
        self.chassis = ChassisState(model, self.prefix)
        _, self.pos_c, self.rot_c = self.chassis.site_offset(model, "lidar")
        self.geom_class = geom_target_classes(model, world)
        mask = K.LIDAR_FLOOR_GROUPMASK if self.floor_rule else K.LIDAR_GROUPMASK
        self.groupmask = np.array(mask, dtype=np.uint8)

    def begin(self):
        self.cast_done = [False] * len(K.SUBSCAN_SLICES)
        self.dist.fill(-1.0)
        self.geomid.fill(-1)

    def sensor_pose(self, data):
        """World position and rotation of L at the present state."""
        return self.chassis.site_pose(data, self.pos_c, self.rot_c)

    def cast(self, k, data):
        """Cast sub-scan k from the sensor pose at the present state: one mj_multiRay over its 216 (217) beams."""
        a, b = K.SUBSCAN_SLICES[k]
        pos, rot = self.sensor_pose(data)
        self.origin[k] = pos
        vec = self.vec[a:b]
        np.einsum("ij,kj->ik", self.local[a:b], rot, out=vec)        # BLAS-free
        mujoco.mj_multiRay(self.model, data, pos, vec.reshape(-1), self.groupmask, 1, self.chassis.body,
                           self.geomid[a:b], self.dist[a:b], self.normal[3 * a:3 * b], b - a, self.cutoff)
        self.cast_done[k] = True

    def floor_kind_at(self, x, y):
        kind = np.full(np.shape(x), self.base_kind, dtype=np.int16)
        for k, poly in self.patches:
            kind[points_in_polygon(x, y, poly)] = k
        return kind

    def finish(self, rng):
        """The wire's scan of this tick (1081 Python ints, mm, 0 = no return); draws 2 x 1081 numbers from rng."""
        if not all(self.cast_done):
            raise RuntimeError("LiDAR: sub-scans %s were not cast this tick" % [k for k, c in enumerate(self.cast_done) if not c])
        noise = rng.standard_normal(K.N_BEAMS)
        u = rng.random(K.N_BEAMS)
        d, g = self.dist, self.geomid
        hit = (g >= 0) & (d >= 0.0)
        cls = np.where(hit, self.geom_class[np.where(hit, g, 0)], NO_TARGET)
        p = np.zeros(K.N_BEAMS)
        tgt = (cls >= 0) & (cls < FLOOR_CLASS)
        if tgt.any():
            r = self.class_range[cls[tgt]]
            dt = d[tgt]
            if self.fade > 0.0:
                p[tgt] = np.clip(1.0 - (dt - r) / (self.fade * r), 0.0, 1.0)
            else:
                p[tgt] = (dt <= r).astype(float)
        fl = np.nonzero(cls == FLOOR_CLASS)[0]
        if fl.size:
            pts = self.origin[self.beam_sub[fl]] + d[fl, None] * self.vec[fl]
            pf = self.floor_prob[self.floor_kind_at(pts[:, 0], pts[:, 1])]
            sin_graze = np.abs(np.einsum("ij,ij->i", self.normal.reshape(-1, 3)[fl], self.vec[fl]))
            pf = np.where(sin_graze >= self.sin_shallow, 1.0, pf)
            p[fl] = np.where(d[fl] <= self.class_range[FLOOR_CLASS], pf, 0.0)
        ret = (u < p) & (d >= K.RANGE_MIN_M) & (d <= K.RANGE_MAX_M)
        mm = np.clip(np.rint((d + self.sigma * noise) * 1000.0), 1000.0 * K.RANGE_MIN_M, 1000.0 * K.RANGE_MAX_M)
        out = np.where(ret, mm, float(K.NO_RETURN)).astype(np.int64)
        self.last = {"true_m": d.copy(), "geomid": g.copy(), "cls": cls, "returned": ret, "origin": self.origin.copy(),
                     "vec": self.vec.copy()}
        return out.tolist()
