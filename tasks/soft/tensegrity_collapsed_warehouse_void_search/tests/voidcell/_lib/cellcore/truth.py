"""Truth of an episode: the robot's true state and the site's true geometry (hidden side only).

Robot: node positions (end-cap centres), planar centroid and heading exactly as the authors' SE(2) (ysim.metrics:
centroid of the six nodes; heading = direction of mean(nodes 0, 2, 4) - mean(nodes 1, 3, 5)), the three lowest end
caps and whether they are a rollable face or an end triangle.
Site: occupancy from the boxes (every box of kind rubble / support / crate / plug / wall that reaches the floor, plus
covered openings too low for the robot), the exact Euclidean clearance on a grid, the opening table with classes
(P / T / XN / XL / B), the slab footprint, the open parts of the void under it (in_void: the straight runs and the
chamber of a three-mouth site, the gallery of a two-mouth site), contacts of the robot with world geoms, and how deep
the strut axes (node to node; the struts have no collision body) and the straight tendon paths reach into any box.
"""
import math

import numpy as np

from . import world as wmod

SOLID_KINDS = ("rubble", "support", "crate", "plug", "wall", "heap")
EXTRA_CLEAR_H = 0.02          # a covered opening counts as solid when its clear height < H_ROLL + this


def quat_to_R(q):
    w, x, y, z = [float(v) for v in q]
    n = math.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


class SiteTruth:
    """Geometry truth of a site package (no simulator needed)."""

    def __init__(self, site, h_roll=None):
        self.site = site
        self.lx, self.ly = site["size_m"]
        self.h_roll = float(h_roll if h_roll is not None else site["envelope"]["values"]["H_ROLL"])
        self.boxes = [b for b in site["boxes"]]
        self._obb = []
        for b in self.boxes:
            R = quat_to_R(b["quat"]) if b.get("quat") else np.eye(3)
            self._obb.append((b["name"], b["kind"], np.asarray(b["pos"], float), np.asarray(b["size"], float), R))

    # ---- openings
    def openings(self):
        out = []
        for row in ("r1", "r2"):
            for g in self.site[row]["openings"]:
                out.append(dict(id=g["id"], row=row.upper(), type=g["type"], cls=g["cls"], x0=g["x0"], x1=g["x1"],
                                y0=g["y0"], y1=g["y1"], covered=bool(g.get("covered")),
                                clear_h_min=g.get("clear_h_min"), width=round(g["x1"] - g["x0"], 6)))
        return out

    def opening(self, oid):
        return [o for o in self.openings() if o["id"] == oid][0]

    def slab(self):
        return dict(self.site["void"]["slab"])

    def under_slab(self, xy):
        s = self.site["void"]["slab"]
        return bool(s["x0"] <= xy[0] <= s["x1"] and s["y0"] <= xy[1] <= s["y1"])

    def in_gallery(self, xy):
        g = self.site["void"]["gallery"]
        return bool(g["x0"] <= xy[0] <= g["x1"] and g["y0"] <= xy[1] <= g["y1"])

    def void_part(self, xy):
        """Which open part of the void under the slab holds xy: a run name (G1, N, K), "C" (the chamber), or
        "gallery" on a site without runs; None outside. Pockets are the dead-end ends of their runs."""
        v = self.site["void"]
        parts = [("C", v["chamber"])] if "chamber" in v else []
        parts += sorted((v.get("runs") or {}).items())
        if not parts:
            parts = [("gallery", v["gallery"])]
        for name, r in parts:
            if r["x0"] <= xy[0] <= r["x1"] and r["y0"] <= xy[1] <= r["y1"]:
                return name
        return None

    def in_void(self, xy):
        return self.void_part(xy) is not None

    # ---- occupancy
    def footprints(self, block_ids=(), solid_classes=()):
        """(x0, x1, y0, y1) of every solid that reaches the floor, covered openings too low for the robot, and the
        openings named in block_ids or whose class is in solid_classes (planning restrictions)."""
        out = []
        for b in self.boxes:
            if b["kind"] in SOLID_KINDS:
                a = b["aabb"]
                out.append((a[0], a[1], a[2], a[3]))
        need = self.h_roll + EXTRA_CLEAR_H
        for o in self.openings():
            low = o["covered"] and o["clear_h_min"] is not None and o["clear_h_min"] < need
            if low or o["id"] in block_ids or o["cls"] in solid_classes:
                out.append((o["x0"], o["x1"], o["y0"], o["y1"]))
        return out

    def clearance_grid(self, step=0.02, block_ids=(), solid_classes=()):
        """(xs, ys, clearance[ny, nx]): exact distance from each cell centre to the nearest footprint or site edge."""
        xs = np.arange(step / 2, self.lx, step)
        ys = np.arange(step / 2, self.ly, step)
        X, Y = np.meshgrid(xs, ys)
        d = np.minimum.reduce([X, self.lx - X, Y, self.ly - Y])
        for x0, x1, y0, y1 in self.footprints(block_ids, solid_classes):
            dx = np.maximum(np.maximum(x0 - X, 0.0), X - x1)
            dy = np.maximum(np.maximum(y0 - Y, 0.0), Y - y1)
            d = np.minimum(d, np.hypot(dx, dy))
        return xs, ys, d

    # ---- segments against boxes
    def segment_penetration(self, P, Q, n=41):
        """Largest depth (m) by which the straight segments P[i]-Q[i] reach inside any box (sampled at n points per
        segment; depth = distance of the sample to the nearest face of the box, 0 outside) and the box name."""
        P = np.atleast_2d(np.asarray(P, float))
        Q = np.atleast_2d(np.asarray(Q, float))
        t = np.linspace(0.0, 1.0, n)[None, :, None]
        X = (P[:, None, :] + t * (Q - P)[:, None, :]).reshape(-1, 3)
        best, where = 0.0, None
        for name, _kind, c, h, R in self._obb:
            loc = (X - c) @ R
            d = np.min(h - np.abs(loc), axis=1)
            m = float(d.max()) if d.size else 0.0
            if m > best:
                best, where = m, name
        return best, where


class RobotTruth:
    """True state of the simulated robot (a ysim Simulator)."""

    def __init__(self, S):
        import mujoco
        self.ys = wmod.ysim()
        self.S = S
        self.robot_geoms = set(int(g) for g in S.ids.col_geoms)
        self.world_geoms = {}
        for g in range(S.m.ngeom):
            if int(S.m.geom_bodyid[g]) == 0 and (S.m.geom_contype[g] or S.m.geom_conaffinity[g]):
                self.world_geoms[g] = mujoco.mj_id2name(S.m, mujoco.mjtObj.mjOBJ_GEOM, g) or ("geom%d" % g)

    def nodes(self):
        return self.S.nodes()

    def pose(self):
        c, th = self.ys.metrics.se2(self.S.nodes())
        return float(c[0]), float(c[1]), float(th)

    def bottom(self):
        b = self.ys.metrics.bottom3(self.S.nodes())
        return dict(nodes=list(b), rollable=b in self.ys.gaits.SYMMETRY, end_triangle=b in self.ys.gaits.END_TRIANGLES)

    def contacts(self):
        """[(world geom name, friction[0], dist)] of every robot-world contact now."""
        d = self.S.d
        out = []
        for i in range(d.ncon):
            c = d.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            if g1 in self.robot_geoms and g2 in self.world_geoms:
                out.append((self.world_geoms[g2], float(c.friction[0]), float(c.dist)))
            elif g2 in self.robot_geoms and g1 in self.world_geoms:
                out.append((self.world_geoms[g1], float(c.friction[0]), float(c.dist)))
        return out

    def rods(self):
        n = self.S.nodes()
        return n[0::2].copy(), n[1::2].copy()

    def tendon_segments(self):
        d, m = self.S.d, self.S.m
        P, Q = [], []
        wx = np.asarray(d.wrap_xpos).reshape(-1, 3)
        for t in range(m.ntendon):
            a, k = int(d.ten_wrapadr[t]), int(d.ten_wrapnum[t])
            pts = wx[a:a + k]
            for i in range(k - 1):
                P.append(pts[i].copy())
                Q.append(pts[i + 1].copy())
        return np.array(P), np.array(Q)
