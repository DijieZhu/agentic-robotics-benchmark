"""Parallel-pinch grasp candidates of the 2F-85 on a kit unit (privileged reference program).

A unit is grasped on its ROOT body only (a shoulder's or an upper leg's output pulley turns on its hinge: a grasp on it
would hold nothing). `ConvexSet` is the root body's convex collision pieces in the unit frame (the cell's carved
pieces: what the pads touch in the simulation) as half-space sets; a ray's first entry into their union gives the
contact point and the facet normal there, exactly.

`grasp_candidates(cs, com)` -> [(G, width, desc)], G = the tcp pose in the unit frame (z = approach, y = closing axis,
the right pad on +y; x across the pads): per closing axis and approach axis of the unit's box (and the closing axis
turned +-30 deg about the approach), the tcp at depths 6-46 mm behind the face the fingers come from, offsets across;
two antipodal pads: >= 3 of 5 ray pairs over the pad's middle (+-5 mm across and along) hit, widths within 6 mm, each
pad's contact normals within ANTIPODAL_DEG of the closing axis (ANTIPODAL_AXIS_DEG along a box axis) and its contact
points on a plane facing it as well; closing width WIDTH_MIN_M-WIDTH_MAX_M (the 2F-85 opens to 85 mm; the approach
needs the grasp's width + 15 mm). Near-duplicates are dropped; the `cap` best by width (15-60 mm preferred), nearness
to the centre of mass and pad overlap.
"""
import math

import numpy as np

ANTIPODAL_DEG = 15.0          # a contact facet within this of the closing axis (else the part turns in the pads) ...
ANTIPODAL_AXIS_DEG = 30.0     # ... and along a box axis (drafted, rounded faces)
WIDTH_MIN_M, WIDTH_MAX_M = 0.006, 0.070
DEPTHS_M = (0.006, 0.010, 0.014, 0.020, 0.028, 0.036, 0.046)
OFFSETS = (-0.4, -0.2, 0.0, 0.2, 0.4)
TURNS_RAD = (0.0, math.radians(30.0), -math.radians(30.0))
RAY_OFFSETS_M = ((0.0, 0.0), (-0.005, 0.0), (0.005, 0.0), (0.0, -0.005), (0.0, 0.005))


def _rotation_angle(R):
    return math.acos(max(-1.0, min(1.0, (np.trace(R) - 1.0) / 2.0)))


class ConvexSet(object):
    """the union of convex pieces (vertex arrays, one frame) as half-space sets: n . x <= d inside"""

    def __init__(self, pieces):
        from scipy.spatial import ConvexHull
        N, D, starts = [], [], []
        lo, hi = [], []
        for V in pieces:
            V = np.asarray(V, float)
            eq = ConvexHull(V).equations                # n . x + c <= 0 inside (no QbB: its scaling would apply here)
            starts.append(sum(len(n) for n in N))
            N.append(eq[:, :3])
            D.append(-eq[:, 3])
            lo.append(V.min(0)); hi.append(V.max(0))
        self.N, self.D, self.starts = np.concatenate(N), np.concatenate(D), np.array(starts)
        self.n_pieces = len(pieces)
        self.lo, self.hi = np.min(lo, axis=0), np.max(hi, axis=0)

    def ray(self, o, v):
        """(t, point, outward normal) of the ray o + t v (t >= 0, |v| = 1) where it first enters the union, or None"""
        o, v = np.asarray(o, float), np.asarray(v, float)
        nv = self.N @ v
        r = self.D - self.N @ o                          # inside along the ray: t * nv <= r
        with np.errstate(divide="ignore", invalid="ignore"):
            t = r / nv
        ent = np.where(nv < -1e-12, t, -np.inf)          # entering planes: t >= r / nv
        ext = np.where(nv > 1e-12, t, np.inf)            # leaving planes: t <= r / nv
        ext = np.where((np.abs(nv) <= 1e-12) & (r < 0), -np.inf, ext)   # parallel and outside: the piece is missed
        t_in = np.maximum.reduceat(ent, self.starts)
        t_out = np.minimum.reduceat(ext, self.starts)
        ok = (t_in <= t_out) & (t_out >= 0.0)
        if not ok.any():
            return None
        t_in = np.where(ok, np.maximum(t_in, 0.0), np.inf)
        i = int(np.argmin(t_in))
        a, b = self.starts[i], (self.starts[i + 1] if i + 1 < self.n_pieces else len(self.N))
        k = a + int(np.argmax(ent[a:b]))
        return float(t_in[i]), o + v * float(t_in[i]), self.N[k]


def grasp_candidates(cs, com, cap=150, depths=DEPTHS_M, width_max=WIDTH_MAX_M):
    """[(G, width m, desc)] best first (see the module docstring); cs a ConvexSet in the unit frame, com there"""
    lo, hi = cs.lo, cs.hi
    com = np.asarray(com, float)
    reach = float(np.linalg.norm(hi - lo)) + 0.05
    out = []
    for ci in range(3):
        for ai in range(3):
            if ai == ci:
                continue
            bi = 3 - ci - ai
            for sgn in (1.0, -1.0):
                a_loc = np.zeros(3)
                a_loc[ai] = sgn
                ext_a = hi[ai] - lo[ai]
                for depth in depths:
                    if depth > ext_a - 0.003:
                        continue
                    for fb in OFFSETS:
                        for rot in TURNS_RAD:
                            g = 0.5 * (lo + hi)
                            g[bi] += fb * (hi[bi] - lo[bi])
                            g[ai] = (lo[ai] + depth) if sgn > 0 else (hi[ai] - depth)
                            c0, b0 = np.zeros(3), np.zeros(3)
                            c0[ci], b0[bi] = 1.0, 1.0
                            c_loc = math.cos(rot) * c0 + math.sin(rot) * b0
                            b_loc = np.cross(a_loc, c_loc)
                            hits = []
                            for off_a, off_b in RAY_OFFSETS_M:
                                gg = g + a_loc * off_a + b_loc * off_b
                                h1 = cs.ray(gg - c_loc * reach, c_loc)
                                h2 = cs.ray(gg + c_loc * reach, -c_loc)
                                if h1 is not None and h2 is not None:
                                    hits.append((h1, h2))
                            if len(hits) < 3:
                                continue
                            ws = sorted(float((h2[1] - h1[1]) @ c_loc) for h1, h2 in hits)
                            w = ws[len(ws) // 2]
                            if w > width_max or w < WIDTH_MIN_M or ws[-1] - ws[0] > 0.006:
                                continue
                            lim = math.cos(math.radians(ANTIPODAL_DEG if rot != 0.0 else ANTIPODAL_AXIS_DEG))
                            flat = True
                            for side, facing in ((0, -c_loc), (1, c_loc)):
                                ns = np.array([h_[side][2] for h_ in hits])
                                if np.min(ns @ facing) < lim:
                                    flat = False
                                    break
                                pts = np.array([h_[side][1] for h_ in hits])
                                _u, s_, vt = np.linalg.svd(pts - pts.mean(axis=0))
                                if s_[1] > 1e-4 and abs(float(vt[2] @ c_loc)) < lim:
                                    flat = False
                                    break
                            if not flat:
                                continue
                            centre = np.mean([0.5 * (h1[1] + h2[1]) for h1, h2 in hits], axis=0)
                            z, y = a_loc, c_loc
                            G = np.eye(4)
                            G[:3, :3] = np.column_stack([np.cross(y, z), y, z])
                            G[:3, 3] = centre
                            if any(np.linalg.norm(G[:3, 3] - o[0][:3, 3]) < 0.003 and _rotation_angle(G[:3, :3].T @ o[0][:3, :3]) < 0.2
                                   for o in out):
                                continue
                            out.append((G, w, "close along %s%s, approach %s%s, depth %.0f mm, offset %+.1f" % (
                                "xyz"[ci], "" if rot == 0 else " %+.0f deg" % math.degrees(rot), "+-"[sgn < 0], "xyz"[ai],
                                depth * 1000, fb), depth))
    out.sort(key=lambda o: (0.0 if 0.015 <= o[1] <= 0.060 else 0.03) + float(np.linalg.norm(o[0][:3, 3] - com))
             - 0.2 * min(o[3], 0.020))
    return [(G, w, desc) for G, w, desc, _d in out[:cap]]


def unit_convex_set(kit_dir, kit, unit):
    """the ConvexSet of the unit's root body's collision pieces (kit meshes, body frame = unit frame)"""
    import os
    import trimesh
    pcs = [np.asarray(trimesh.load(os.path.join(kit_dir, f), force="mesh").vertices, float) for f in kit["bodies"][unit]["collision"]]
    return ConvexSet(pcs)
