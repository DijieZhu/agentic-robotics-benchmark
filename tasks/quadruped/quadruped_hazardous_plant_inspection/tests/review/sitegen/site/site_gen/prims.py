"""The primitive store: every static geom of a site as plain data.

One list feeds both the MJCF builder (mjcf.py) and the traversability rasterizer (trav.py), so the map and the
physics cannot disagree. Geom groups: 0 structure and equipment (collision), 1 decor (no collision), 2 ceiling and
overhead decor above `overhead_z` (no collision, hidden from top-down cameras), 3 traversal elements (collision).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .util import mat_from_quat, quat_yaw, quat_z_to


@dataclass
class Prim:
    name: str
    type: str                 # box | cylinder | capsule | sphere
    pos: list
    quat: list
    size: list
    group: int = 0
    mat: str = "concrete"
    body: str = "static/misc"
    owner: str = ""
    role: str = "solid"       # solid | walkable | decor
    elem: str | None = None   # traversal element id
    cls: str | None = None    # element class for the rasterizer
    friction: float | None = None
    priority: int = 0
    radio: bool = False       # opaque to the link (walls, doors, shields)
    rgba: list | None = None

    # ------------------------------------------------------------------ geometry
    def R(self) -> np.ndarray:
        return mat_from_quat(self.quat)

    def corners(self) -> np.ndarray:
        """Sample points on the solid's hull (world frame), enough for footprint polygons and z ranges."""
        R = self.R()
        p = np.asarray(self.pos, float)
        if self.type == "box":
            hx, hy, hz = self.size
            c = np.array([[sx * hx, sy * hy, sz * hz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
            return p + c @ R.T
        if self.type in ("cylinder", "capsule"):
            r, hl = self.size[0], self.size[1]
            n = 16
            a = np.linspace(0, 2 * math.pi, n, endpoint=False)
            ring = np.stack([r * np.cos(a), r * np.sin(a), np.zeros(n)], 1)
            pts = [ring + [0, 0, hl], ring - [0, 0, hl]]
            if self.type == "capsule":
                # the caps: add the end points pushed out by r along every axis
                for s in (-1, 1):
                    e = np.array([0, 0, s * hl])
                    pts.append(e + r * np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1],
                                                 [0, 0, -1]], float))
            c = np.concatenate(pts, 0)
            return p + c @ R.T
        if self.type == "sphere":
            r = self.size[0]
            n = 12
            a = np.linspace(0, 2 * math.pi, n, endpoint=False)
            c = np.concatenate([np.stack([r * np.cos(a), r * np.sin(a), np.zeros(n)], 1),
                                [[0, 0, r], [0, 0, -r]]], 0)
            return p + c
        raise ValueError(self.type)

    def zrange(self) -> tuple:
        c = self.corners()
        return float(c[:, 2].min()), float(c[:, 2].max())

    def collides(self) -> bool:
        return self.group in (0, 3)


class PrimStore:
    def __init__(self, overhead_z: float = 2.2):
        self.prims: list[Prim] = []
        self.names: set[str] = set()
        self.overhead_z = overhead_z
        self.bodies: dict[str, dict] = {}

    # ------------------------------------------------------------------ bookkeeping
    def _uniq(self, name: str) -> str:
        if name not in self.names:
            self.names.add(name)
            return name
        k = 1
        while "%s~%d" % (name, k) in self.names:
            k += 1
        n = "%s~%d" % (name, k)
        self.names.add(n)
        return n

    def add(self, p: Prim) -> Prim:
        p.name = self._uniq(p.name)
        if p.group in (0, 3):
            z0, _ = p.zrange()
            if z0 >= self.overhead_z - 1e-6:
                p.group = 2
                p.role = "decor"
        if p.group in (1, 2):
            p.role = "decor"
        self.prims.append(p)
        self.bodies.setdefault(p.body, {"owner": p.owner})
        return p

    def by_body(self, body: str) -> list:
        return [p for p in self.prims if p.body == body]

    def remove_body(self, body: str):
        keep = []
        for p in self.prims:
            if p.body == body:
                self.names.discard(p.name)
            else:
                keep.append(p)
        self.prims = keep
        self.bodies.pop(body, None)

    # ------------------------------------------------------------------ constructors
    def box(self, name, center, half, yaw: float = 0.0, quat=None, **kw) -> Prim:
        q = list(quat) if quat is not None else quat_yaw(yaw)
        return self.add(Prim(name=name, type="box", pos=[float(v) for v in center], quat=q,
                             size=[float(v) for v in half], **kw))

    def box_aabb(self, name, lo, hi, **kw) -> Prim | None:
        lo = np.asarray(lo, float)
        hi = np.asarray(hi, float)
        if np.any(hi - lo < 1e-4):
            return None
        return self.box(name, (lo + hi) / 2, (hi - lo) / 2, **kw)

    def cyl(self, name, p0, p1, r, kind="cylinder", **kw) -> Prim | None:
        p0 = np.asarray(p0, float)
        p1 = np.asarray(p1, float)
        d = p1 - p0
        L = float(np.linalg.norm(d))
        if L < 1e-4:
            return None
        return self.add(Prim(name=name, type=kind, pos=list((p0 + p1) / 2), quat=quat_z_to(d),
                             size=[float(r), L / 2, 0.0], **kw))

    def vcyl(self, name, x, y, z0, z1, r, **kw) -> Prim | None:
        return self.cyl(name, (x, y, z0), (x, y, z1), r, **kw)

    def sphere(self, name, c, r, **kw) -> Prim:
        return self.add(Prim(name=name, type="sphere", pos=[float(v) for v in c], quat=[1, 0, 0, 0],
                             size=[float(r), 0, 0], **kw))

    def pipe(self, name, points, r, mat, body, owner="", elbows=True, group=0, flange_every=None, **kw) -> list:
        """A pipe through `points` (world 3-D): cylinders, elbow spheres at the bends; segments crossing the
        overhead height are split so the part above goes to the overhead group."""
        out = []
        pts = [np.asarray(p, float) for p in points]
        k = 0
        for a, b in zip(pts[:-1], pts[1:]):
            segs = [(a, b)]
            za, zb = a[2], b[2]
            oz = self.overhead_z
            if (za - oz) * (zb - oz) < 0 and abs(zb - za) > 1e-6:
                t = (oz - za) / (zb - za)
                m = a + t * (b - a)
                segs = [(a, m), (m, b)]
            for s0, s1 in segs:
                p = self.cyl("%s/s%d" % (name, k), s0, s1, r, mat=mat, body=body, owner=owner, group=group, **kw)
                if p is not None:
                    out.append(p)
                k += 1
        if elbows:
            for i, c in enumerate(pts[1:-1]):
                out.append(self.sphere("%s/e%d" % (name, i), c, r * 1.12, mat=mat, body=body, owner=owner,
                                       group=group, **kw))
        if flange_every:
            for i, (a, b) in enumerate(zip(pts[:-1], pts[1:])):
                d = b - a
                L = float(np.linalg.norm(d))
                if L < 2 * flange_every:
                    continue
                u = d / L
                n = int(L // flange_every)
                for j in range(1, n):
                    c = a + u * (j * L / n)
                    out.append(self.cyl("%s/f%d_%d" % (name, i, j), c - u * 0.012, c + u * 0.012, r * 1.45,
                                        mat="steel_dark", body=body, owner=owner, group=1))
        return [p for p in out if p is not None]


@dataclass
class Frame2D:
    """Local room frame: origin (world xy of the inner front-left corner), u axis along the front wall, v into the
    room; (u, v, z) is right-handed with z up."""
    origin: tuple
    u: tuple                   # unit vector (world xy) of the local u axis
    Lu: float = 0.0
    Lv: float = 0.0
    names: dict = field(default_factory=dict)

    @property
    def v(self) -> tuple:
        return (-self.u[1], self.u[0])

    @property
    def yaw(self) -> float:
        return math.atan2(self.u[1], self.u[0])

    def w(self, u, v, z=0.0) -> list:
        ox, oy = self.origin
        return [ox + u * self.u[0] + v * self.v[0], oy + u * self.u[1] + v * self.v[1], z]

    def wyaw(self, local_yaw: float) -> float:
        return self.yaw + local_yaw

    def to_local(self, x, y) -> tuple:
        dx, dy = x - self.origin[0], y - self.origin[1]
        return (dx * self.u[0] + dy * self.u[1], dx * self.v[0] + dy * self.v[1])

    def lrect_to_world(self, u0, v0, u1, v1) -> list:
        a = self.w(u0, v0)
        b = self.w(u1, v1)
        return [min(a[0], b[0]), min(a[1], b[1]), max(a[0], b[0]), max(a[1], b[1])]
