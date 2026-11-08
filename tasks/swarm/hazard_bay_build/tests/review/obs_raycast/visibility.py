# Moved in from the round-4 observability study (arb-eval/pr31r4-sensors/obs_map/visibility.py, map-observability.md section 7) for
# check_observability.py --raycast: the pxr / numpy cross-check of the renderer's measure. Unchanged but this header.
"""Observability study for PR #31 round 4 (map-observability.md): ray-cast visibility of the mission's targets in the
real bay (tests/hidden/bay_true.usda, extracted by extract_geom.py), on the fixed set's two days and on other days
drawn by the same generator, from viewpoints the robots can reach, with the catalog cameras' pixel scales.

    python -I -B visibility.py <geom.json> <days.json ...> --out <result.json>

Pure numpy. Geometry: every visible Cube and Cylinder gprim of the real bay (the invisible floor collider and plume box
left out; the floor plane z = 0 stands for the hall's floor), the six parts as their own boxes at the day's poses
(own boxes from a privileged capture: control_sim's own_box), the lever turned to the day's angle about its stem.
The robots' own bodies are not modelled (see the note).
"""
import argparse
import json
import math

import numpy as np

# the parts' own boxes (half extents in the body frame; the body origin is the box centre), from the privileged block of
# a captured observation (pr31r3/out/docs_cap2/act_900_0.json: parts.*.own_lo / own_hi)
PART_HALF = {"cracker_box": (0.082, 0.1067, 0.0359), "mustard_bottle": (0.048, 0.0957, 0.0291),
             "sugar_box": (0.0463, 0.0881, 0.0226), "soup_can": (0.0338, 0.0509, 0.0339),
             "power_drill": (0.0922, 0.0934, 0.0286), "mug": (0.0585, 0.0407, 0.0465)}
# catalog cameras: (W, H, focal mm, hap mm)
CAM = {"CAM-RGB-W": (1920, 1200, 4.0, 11.26), "CAM-RGB-M": (1920, 1200, 8.0, 11.26), "CAM-RGB-T": (1920, 1200, 16.0, 11.26),
       "CAM-RGB-S": (640, 480, 3.6, 4.8), "THERM-A": (160, 120, 1.7, 1.92), "THERM-B": (640, 512, 13.0, 7.68),
       "THERM-C": (640, 512, 9.0, 7.68), "TOF-D": (320, 240, 2.0, 2.88), "STEREO-X": (1280, 720, 2.1, 5.0)}
STEM = np.array([-1.2, 3.2, 0.95])
PAN_NOM = np.array([2.6, 1.9])
TABLES = {"Table_A": dict(c=(-1.3, 0.7), size=(1.2, 0.6), yaw=0.0, top=0.75),
          "Table_B": dict(c=(1.45, 0.55), size=(0.8, 0.6), yaw=20.0, top=0.45)}


def fpx(model):
    W, H, f, hap = CAM[model]
    return f / hap * W


def quat_mat(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


class Box:
    def __init__(self, name, c, U, h, rgb=None):
        self.name, self.c, self.U, self.h, self.rgb = name, np.asarray(c, float), np.asarray(U, float), np.asarray(h, float), rgb
        self.rad = float(np.linalg.norm(self.h))

    def hit(self, O, D, eps=1e-4):
        o = (O - self.c) @ self.U.T
        d = D @ self.U.T
        with np.errstate(divide="ignore", invalid="ignore"):
            t1 = (-self.h - o) / d
            t2 = (self.h - o) / d
        par = np.abs(d) < 1e-12
        inside = np.abs(o) <= self.h
        # a ray parallel to a slab: no constraint inside it, no hit outside it
        t1 = np.where(par, np.where(inside, -np.inf, np.inf), t1)
        t2 = np.where(par, np.where(inside, np.inf, np.inf), t2)
        tmin = np.max(np.minimum(t1, t2), axis=1)
        tmax = np.min(np.maximum(t1, t2), axis=1)
        return (tmax >= tmin) & (tmin < 1 - eps) & (tmax > eps)

    def samples(self, n=6):
        """Points on the faces (n x n per face) with outward normals and the patch area."""
        pts, nrm, area = [], [], []
        g = (np.arange(n) + 0.5) / n * 2 - 1
        for k in range(3):
            j, l = [m for m in range(3) if m != k]
            a = (2 * self.h[j] / n) * (2 * self.h[l] / n)
            for s in (-1, 1):
                for u in g:
                    for v in g:
                        p = self.c + s * self.h[k] * self.U[k] + u * self.h[j] * self.U[j] + v * self.h[l] * self.U[l]
                        pts.append(p + 1e-3 * s * self.U[k])
                        nrm.append(s * self.U[k])
                        area.append(a)
        return np.array(pts), np.array(nrm), np.array(area)

    def corners(self):
        return np.array([self.c + sx * self.h[0] * self.U[0] + sy * self.h[1] * self.U[1] + sz * self.h[2] * self.U[2]
                         for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])


class Cyl:
    def __init__(self, name, c, a, r, hh, rgb=None):
        self.name, self.c, self.a, self.r, self.hh, self.rgb = name, np.asarray(c, float), np.asarray(a, float) / np.linalg.norm(a), r, hh, rgb
        self.rad = float(math.hypot(r, hh))

    def hit(self, O, D, eps=1e-4):
        w = O - self.c
        wa = w @ self.a
        da = D @ self.a
        wp = w - np.outer(wa, self.a)
        dp = D - np.outer(da, self.a)
        A = np.sum(dp * dp, axis=1)
        B = 2 * np.sum(wp * dp, axis=1)
        C = np.sum(wp * wp, axis=1) - self.r ** 2
        disc = B * B - 4 * A * C
        with np.errstate(divide="ignore", invalid="ignore"):
            sq = np.sqrt(np.maximum(disc, 0))
            r0 = (-B - sq) / (2 * A)
            r1 = (-B + sq) / (2 * A)
        small = A < 1e-14
        r0 = np.where(small, np.where(C <= 0, -np.inf, np.inf), np.where(disc < 0, np.inf, r0))
        r1 = np.where(small, np.where(C <= 0, np.inf, -np.inf), np.where(disc < 0, -np.inf, r1))
        with np.errstate(divide="ignore", invalid="ignore"):
            s0 = (-self.hh - wa) / da
            s1 = (self.hh - wa) / da
        par = np.abs(da) < 1e-12
        ins = np.abs(wa) <= self.hh
        # a ray parallel to the caps: no constraint between them, no hit outside them
        s0 = np.where(par, np.where(ins, -np.inf, np.inf), s0)
        s1 = np.where(par, np.where(ins, np.inf, np.inf), s1)
        lo = np.maximum(r0, np.minimum(s0, s1))
        hi = np.minimum(r1, np.maximum(s0, s1))
        return (hi >= lo) & (lo < 1 - eps) & (hi > eps)

    def samples(self, n=10):
        """Points on the side and both caps, with normals and patch areas."""
        # an orthonormal frame round the axis
        t = np.cross(self.a, [1, 0, 0]) if abs(self.a[0]) < 0.9 else np.cross(self.a, [0, 1, 0])
        t /= np.linalg.norm(t)
        b = np.cross(self.a, t)
        pts, nrm, area = [], [], []
        m = 2 * n
        for i in range(m):
            ph = 2 * math.pi * (i + 0.5) / m
            rd = math.cos(ph) * t + math.sin(ph) * b
            for k in range(n):
                z = ((k + 0.5) / n * 2 - 1) * self.hh
                pts.append(self.c + self.r * rd + z * self.a + 1e-3 * rd)
                nrm.append(rd)
                area.append((2 * math.pi * self.r / m) * (2 * self.hh / n))
        for s in (-1, 1):
            for i in range(n):
                for j in range(n):
                    u, v = ((i + 0.5) / n * 2 - 1), ((j + 0.5) / n * 2 - 1)
                    if u * u + v * v > 1:
                        continue
                    pts.append(self.c + s * self.hh * self.a + self.r * (u * t + v * b) + 1e-3 * s * self.a)
                    nrm.append(s * self.a)
                    area.append((2 * self.r / n) ** 2)
        return np.array(pts), np.array(nrm), np.array(area)

    def corners(self):
        t = np.cross(self.a, [1, 0, 0]) if abs(self.a[0]) < 0.9 else np.cross(self.a, [0, 1, 0])
        t /= np.linalg.norm(t)
        b = np.cross(self.a, t)
        return np.array([self.c + s * self.hh * self.a + self.r * (math.cos(p) * t + math.sin(p) * b)
                         for s in (-1, 1) for p in np.linspace(0, 2 * math.pi, 12, endpoint=False)])


def load_bay(path):
    shapes = []
    for x in json.load(open(path)):
        if x["visible"] == "invisible":
            continue
        A = np.array(x["axes"], float)
        if x["kind"] == "box":
            n = np.linalg.norm(A, axis=1)
            shapes.append(Box(x["path"], x["center"], A / n[:, None], n * np.array(x["half"]), x["rgb"]))
        else:
            k = "XYZ".index(x["axis"])
            n = np.linalg.norm(A, axis=1)
            others = [n[i] for i in range(3) if i != k]
            shapes.append(Cyl(x["path"], x["center"], A[k] / n[k], x["radius"] * max(others), x["height"] / 2 * n[k], x["rgb"]))
    return shapes


def lever_shapes(shapes, deg):
    """The handle's three gprims turned `deg` toward +x about the stem (rotation about +y)."""
    th = math.radians(deg)
    R = np.array([[math.cos(th), 0, math.sin(th)], [0, 1, 0], [-math.sin(th), 0, math.cos(th)]])
    out = []
    for s in shapes:
        if "/Handle/" not in s.name:
            out.append(s)
            continue
        if isinstance(s, Box):
            out.append(Box(s.name, STEM + R @ (s.c - STEM), (R @ s.U.T).T, s.h, s.rgb))
        else:
            out.append(Cyl(s.name, STEM + R @ (s.c - STEM), R @ s.a, s.r, s.hh, s.rgb))
    return out


def day_scene(bay, day):
    """The bay of the day: the lever turned, the pan moved, the parts at their poses."""
    sh = lever_shapes(bay, float(day["lever_deg"]))
    dx, dy = day["pan"]
    out = []
    for s in sh:
        if "/FirePan/" in s.name:
            c = s.c + np.array([dx, dy, 0.0])
            out.append(Cyl(s.name, c, s.a, s.r, s.hh, s.rgb))
        else:
            out.append(s)
    parts = {}
    for n, p in day["parts"].items():
        b = Box("part:" + n, p["pos"], quat_mat(p["quat"]).T, PART_HALF[n])
        out.append(b)
        parts[n] = b
    return out, parts


def look_cam(C, T, up=(0, 0, 1)):
    d = np.asarray(T, float) - C
    d /= np.linalg.norm(d)
    r = np.cross(d, up)
    if np.linalg.norm(r) < 1e-6:
        r = np.cross(d, (0, 1, 0))
    r /= np.linalg.norm(r)
    u = np.cross(r, d)
    return d, u, r


def view(C, d, u, r, model, target, occluders, n=6):
    """Visibility of a target shape from a camera at C (look d, up u, right r) of a catalog model: the fraction of its
    front-facing surface samples that are in the image and unoccluded, the visible area in px^2, the projected box in
    px (width, height) and the distance."""
    W, H, _, _ = CAM[model]
    f = fpx(model)
    P, N, A = target.samples(n)
    v = C - P
    front = np.sum(N * v, axis=1) > 0
    if not np.any(front):
        return dict(frac=0.0, px2=0.0, w=0.0, h=0.0, dist=float(np.linalg.norm(target.c - C)))
    P, N, A, v = P[front], N[front], A[front], v[front]
    rel = P - C
    z = rel @ d
    x = rel @ r
    y = rel @ u
    with np.errstate(divide="ignore", invalid="ignore"):
        px, py = f * x / z, f * y / z
    inimg = (z > 0.02) & (np.abs(px) <= W / 2) & (np.abs(py) <= H / 2)
    O = np.repeat(C[None], len(P), axis=0)
    D = P - O
    occ = np.zeros(len(P), bool)
    seg = target.c - C
    L2 = float(seg @ seg)
    for s in occluders:
        if s is target or "/Pads/" in s.name or "/CalibMat/" in s.name:
            continue
        t = min(1.0, max(0.0, float((s.c - C) @ seg) / L2))
        if np.linalg.norm(C + t * seg - s.c) > s.rad + target.rad + 0.01:
            continue
        m = ~occ & inimg
        if not np.any(m):
            break
        occ[m] |= s.hit(O[m], D[m])
    vis = inimg & ~occ
    dist = np.linalg.norm(v, axis=1)
    cosang = np.abs(np.sum(N * v, axis=1)) / dist
    px2 = float(np.sum((A * cosang * (f / np.maximum(z, 1e-6)) ** 2)[vis]))
    cs = target.corners()
    zc = (cs - C) @ d
    if np.all(zc > 0.02):
        ux, uy = f * ((cs - C) @ r) / zc, f * ((cs - C) @ u) / zc
        w, h = float(ux.max() - ux.min()), float(uy.max() - uy.min())
    else:
        w = h = float("nan")
    return dict(frac=float(np.mean(vis)), px2=px2, w=w, h=h, dist=float(np.linalg.norm(target.c - C)))


# ----------------------------------------------------------------------------------------------------- reachability
def footprints(day, fire=True):
    """The vehicle's keep-outs on the floor (2D: oriented rectangles and discs, not inflated)."""
    rects = [((-1.3, 0.7), (1.2, 0.6), 0.0), ((1.45, 0.55), (0.8, 0.6), 20.0), ((0.1, 1.5), (0.6, 0.424), 0.0),
             ((-1.9, -3.1), (0.8, 0.6), 0.0)]
    rects += [((x, 3.35), (0.16, 0.16), 0.0) for x in (-3.0, -1.0, 1.0, 3.0)]
    discs = [((0.0, -2.6), 0.14)]  # the bucket on the floor
    pan = PAN_NOM + np.array(day["pan"])
    discs.append((tuple(pan), 2.0 if fire else 0.30))
    return rects, discs


def free(xy, rects, discs, clear=0.35):
    x, y = xy
    for (cx, cy), (sx, sy), yaw in rects:
        c, s = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
        u, v = (x - cx) * c + (y - cy) * s, -(x - cx) * s + (y - cy) * c
        du, dv = max(0.0, abs(u) - sx / 2), max(0.0, abs(v) - sy / 2)
        if math.hypot(du, dv) < clear:
            return False
    for (cx, cy), r in discs:
        if math.hypot(x - cx, y - cy) < r + clear:
            return False
    return -3.5 <= x <= 3.5 and -3.6 <= y <= 3.25


def best_ugv_view(target, occ, day, heights=(0.92,), model="CAM-RGB-M", fire=True, rmax=2.5, step=0.1):
    rects, discs = footprints(day, fire)
    best = None
    tc = target.c
    xs = np.arange(-3.5, 3.5001, step)
    ys = np.arange(-3.6, 3.2501, step)
    for x in xs:
        for y in ys:
            if math.hypot(x - tc[0], y - tc[1]) > rmax or not free((x, y), rects, discs):
                continue
            for h in heights:
                C = np.array([x, y, h])
                d, u, r = look_cam(C, tc)
                res = view(C, d, u, r, model, target, occ)
                res.update(x=round(float(x), 2), y=round(float(y), 2), z=h)
                key = (res["frac"] >= 0.5, res["px2"])
                if best is None or key > best[0]:
                    best = (key, res)
    return best[1] if best else None


def summarise(day_id, day, bay):
    occ, parts = day_scene(bay, day)
    out = dict(id=day_id, parts={})
    for n, b in parts.items():
        rec = {}
        rec["ugv_mast_fire"] = best_ugv_view(b, occ, day, heights=(0.92,), fire=True)
        rec["ugv_mast_nofire"] = best_ugv_view(b, occ, day, heights=(0.92,), fire=False)
        rec["ugv_low_nofire"] = best_ugv_view(b, occ, day, heights=(0.45,), fire=False)
        # a flying robot straight above the part's table at the cruise height, the wide down camera
        t = TABLES[day["parts"][n]["table"]]
        C = np.array([t["c"][0], t["c"][1], 2.6])
        d, u, r = np.array([0, 0, -1.0]), np.array([0, 1.0, 0]), np.array([1.0, 0, 0])
        rec["uav_down_over_table"] = view(C, d, u, r, "CAM-RGB-W", b, occ)
        out["parts"][n] = rec
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("geom")
    ap.add_argument("days", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--range", default="")
    ap.add_argument("--skip", default="")
    a = ap.parse_args()
    bay = load_bay(a.geom)
    days = []
    for f in a.days:
        d = json.load(open(f))
        for m in d["missions"]:
            days.append((m["id"], m["day"]))
    if a.skip:
        days = [(i, d) for i, d in days if i not in a.skip.split(",")]
    if a.range:
        lo, hi = (int(v) for v in a.range.split(":"))
        days = days[lo:hi]
    if a.limit:
        days = days[: a.limit]
    res = [summarise(i, d, bay) for i, d in days]
    json.dump(res, open(a.out, "w"), indent=1)
    print("wrote", a.out, len(res))
