"""The rubble site's geometry (cell internals): turns a plan (plan.py) into the static geoms, the movable debris, the
heightfield of the ground (floor, street, rubble mounds, collapsed floor sections), the painted ground image, the
survivor and the decoys, plus the reference route (waypoints and the crawler configuration on each leg) that only the
privileged reference program reads.

Frames: world = building frame: x east along the facade, y north into the building, z up, the ground floor at z = 0,
the facade's outer face on y = 0, the street at y < 0.
"""

import math

import numpy as np

from .human import SHIRT, SKIN, TROUSERS, HAIR, figure_key, figure_layout, figure_parts, place_parts
from .plan import ROBOT

__all__ = ["build_site", "Site", "CEIL", "STOREY", "HF_RES"]

CEIL = 3.0            # ground-floor ceiling soffit
STOREY = 3.2
SLAB_T = 0.22
T_EXT = 0.30
T_INT = 0.16
COL = 0.36
HF_RES = 0.08         # heightfield cell (m)
HF_BASE = 1.6         # the heightfield reaches this far below the floor (collapsed floor sections)
STREET_Z = -0.06

MAT = {
    # name: (texture, rgba tint, specular, shininess, friction)
    "concrete": ("concrete", (1, 1, 1, 1), 0.05, 0.1, 0.85),
    "concrete_dark": ("concrete_dark", (1, 1, 1, 1), 0.05, 0.1, 0.85),
    "slab": ("concrete", (0.92, 0.92, 0.9, 1), 0.05, 0.1, 0.85),
    "rubble": ("rubble", (1, 1, 1, 1), 0.02, 0.05, 0.75),
    "brick": ("brick", (1, 1, 1, 1), 0.03, 0.05, 0.6),
    "plaster_white": ("plaster_white", (1, 1, 1, 1), 0.05, 0.1, 0.45),
    "plaster_cream": ("plaster_cream", (1, 1, 1, 1), 0.05, 0.1, 0.45),
    "plaster_blue": ("plaster_blue", (1, 1, 1, 1), 0.05, 0.1, 0.45),
    "plaster_green": ("plaster_green", (1, 1, 1, 1), 0.05, 0.1, 0.45),
    "plaster_pink": ("plaster_pink", (1, 1, 1, 1), 0.05, 0.1, 0.45),
    "tiles": ("tiles", (1, 1, 1, 1), 0.2, 0.3, 0.6),
    "wood": ("wood", (1, 1, 1, 1), 0.1, 0.2, 0.6),
    "wood_dark": ("wood_dark", (1, 1, 1, 1), 0.1, 0.2, 0.6),
    "asphalt": ("asphalt", (1, 1, 1, 1), 0.02, 0.05, 0.85),
    "paving": ("paving", (1, 1, 1, 1), 0.03, 0.05, 0.85),
    "fabric_red": ("fabric_red", (1, 1, 1, 1), 0.0, 0.0, 0.7),
    "fabric_blue": ("fabric_blue", (1, 1, 1, 1), 0.0, 0.0, 0.7),
    "fabric_grey": ("fabric_grey", (1, 1, 1, 1), 0.0, 0.0, 0.7),
    "fabric_green": ("fabric_green", (1, 1, 1, 1), 0.0, 0.0, 0.7),
    "metal_white": ("metal_white", (1, 1, 1, 1), 0.4, 0.5, 0.45),
    "metal_red": ("metal_red", (1, 1, 1, 1), 0.4, 0.5, 0.45),
    "metal_blue": ("metal_blue", (1, 1, 1, 1), 0.4, 0.5, 0.45),
    "metal_grey": ("metal_grey", (1, 1, 1, 1), 0.4, 0.5, 0.45),
    "rust": ("rust", (1, 1, 1, 1), 0.1, 0.1, 0.6),
    "rebar": ("rebar", (1, 1, 1, 1), 0.2, 0.3, 0.6),
    "dust": ("dust", (1, 1, 1, 1), 0.02, 0.05, 0.75),
    "glass": ("glass", (0.6, 0.7, 0.75, 0.35), 0.8, 0.9, 0.4),
    "black": (None, (0.05, 0.05, 0.06, 1), 0.1, 0.1, 0.6),
    "window": (None, (0.10, 0.12, 0.14, 1), 0.6, 0.8, 0.6),
    "plastic_skin": (None, (0.90, 0.86, 0.80, 1), 0.3, 0.5, 0.6),
    "tarp_orange": (None, (0.95, 0.45, 0.08, 1), 0.1, 0.2, 0.6),
    "kit": (None, (0.98, 0.42, 0.05, 1), 0.2, 0.3, 0.8),
    "yellow": (None, (0.95, 0.80, 0.10, 1), 0.2, 0.3, 0.6),
    "white_paint": (None, (0.92, 0.92, 0.9, 1), 0.1, 0.1, 0.8),
}


def _q_euler(yaw=0.0, pitch=0.0, roll=0.0):
    """wxyz quaternion of R = Rz(yaw) Ry(pitch) Rx(roll)."""
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    return [cy * cp * cr + sy * sp * sr, cy * cp * sr - sy * sp * cr, cy * sp * cr + sy * cp * sr,
            sy * cp * cr - cy * sp * sr]


def _rz(yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def chunk_meshes(rng, n_shapes=10):
    """Convex rubble chunk meshes (vertex arrays), unit-sized; scaled copies are separate assets."""
    out = []
    for k in range(n_shapes):
        nv = int(rng.integers(9, 16))
        v = rng.normal(size=(nv, 3))
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        v *= rng.uniform(0.55, 1.0, (nv, 1))
        v[:, 2] *= rng.uniform(0.45, 0.8)          # flatter than tall
        v[:, 0] *= rng.uniform(0.8, 1.4)
        out.append(v)
    return out


class Site(object):
    """Everything the scene needs from one plan; see build_site."""

    def __init__(self, plan):
        self.plan = plan
        self.rng = np.random.default_rng((int(plan["seed"]) * 7919 + 17) & ((1 << 63) - 1))
        self.statics = []          # dicts: type, size, pos, quat, mat, collide, name, thermal
        self.bodies = []           # movable debris: dicts with geoms, pos, quat, mass, kind
        self.meshes = {}           # name -> vertices (n, 3)
        self.parts = {}            # survivor / decoy parts
        self.figures = []          # the posed bodies the sensors see: who, key (human.figures), x, y, yaw, scale
        self.figure_tint = {}      # who -> {cloth, skin, hair: (r, g, b)}: tints of a posed body's colour maps
        self.props = []            # the look pack's models on the site (props.py): model, pos, quat, scale, collide
        self.vehicles = []         # real vehicles (the look pack's vehicles.npz): name, id, x, y, z, yaw, scale, paint
        self.keepout = []          # (x0, y0, x1, y1, r) corridor segments kept free of clutter
        self.route = []
        self.events = []
        self.n = 0
        xs, ys = plan["xs"], plan["ys"]
        self.W, self.D = xs[-1], ys[-1]
        # heightfield extent
        self.hx0, self.hx1 = -5.0, self.W + 5.0
        self.hy0, self.hy1 = -11.0, self.D + 3.5
        self.hnx = int(round((self.hx1 - self.hx0) / HF_RES)) + 1
        self.hny = int(round((self.hy1 - self.hy0) / HF_RES)) + 1
        self.H = np.zeros((self.hny, self.hnx))          # row 0 = south edge (y = hy0)
        gx = self.hx0 + np.arange(self.hnx) * HF_RES
        gy = self.hy0 + np.arange(self.hny) * HF_RES
        self.GX, self.GY = np.meshgrid(gx, gy)
        self.H[self.GY < -3.0] = STREET_Z
        # ground classes for the painter: 0 road, 1 sidewalk, 2 concrete floor, 3 tiles, 4 rubble, 5 dark (pits)
        self.ground = np.ones((self.hny, self.hnx), np.int8)
        self.ground[self.GY < -3.0] = 0
        self.ground[(self.GX > 0) & (self.GX < self.W) & (self.GY > 0) & (self.GY < self.D)] = 2
        self.thermal = {}          # geom name -> temperature class
        self.chunk_shapes = chunk_meshes(self.rng)
        self._chunk_assets = {}

    # ------------------------------------------------------------------ primitives
    def _name(self, base):
        self.n += 1
        return "%s_%d" % (base, self.n)

    def box(self, c, half, mat, yaw=0.0, pitch=0.0, roll=0.0, collide=True, name="box", thermal=None, group=None):
        nm = self._name(name)
        self.statics.append({"name": nm, "type": "box", "size": [float(h) for h in half], "pos": [float(v) for v in c],
                             "quat": _q_euler(yaw, pitch, roll), "mat": mat, "collide": bool(collide), "group": group})
        if thermal:
            self.thermal[nm] = thermal
        return nm

    def cyl(self, c, r, hh, mat, yaw=0.0, pitch=0.0, roll=0.0, collide=True, name="cyl", thermal=None):
        nm = self._name(name)
        self.statics.append({"name": nm, "type": "cylinder", "size": [float(r), float(hh), 0.0], "pos": [float(v) for v in c],
                             "quat": _q_euler(yaw, pitch, roll), "mat": mat, "collide": bool(collide)})
        if thermal:
            self.thermal[nm] = thermal
        return nm

    def capsule(self, a, b, r, mat, collide=True, name="cap", thermal=None):
        nm = self._name(name)
        self.statics.append({"name": nm, "type": "capsule", "fromto": [float(v) for v in list(a) + list(b)],
                             "size": [float(r), 0.0, 0.0], "mat": mat, "collide": bool(collide)})
        if thermal:
            self.thermal[nm] = thermal
        return nm

    def ellipsoid(self, c, radii, mat, yaw=0.0, pitch=0.0, collide=True, name="ell", thermal=None):
        nm = self._name(name)
        self.statics.append({"name": nm, "type": "ellipsoid", "size": [float(v) for v in radii], "pos": [float(v) for v in c],
                             "quat": _q_euler(yaw, pitch, 0.0), "mat": mat, "collide": bool(collide)})
        if thermal:
            self.thermal[nm] = thermal
        return nm

    def chunk(self, c, size, mat="rubble", collide=False, yaw=None, name="chunk"):
        """A rubble chunk (convex mesh) of roughly `size` metres across."""
        rng = self.rng
        k = int(rng.integers(len(self.chunk_shapes)))
        sc = round(float(size) * 4) / 4.0
        sc = max(0.05, sc) if size < 0.25 else max(0.25, sc)
        if size < 0.25:
            sc = round(float(size) / 0.05) * 0.05
            sc = max(0.05, sc)
        key = "chunk%d_%03d" % (k, int(round(sc * 100)))
        if key not in self.meshes:
            self.meshes[key] = self.chunk_shapes[k] * (sc / 2.0)
        nm = self._name(name)
        yaw = float(rng.uniform(-math.pi, math.pi)) if yaw is None else yaw
        self.statics.append({"name": nm, "type": "mesh", "mesh": key, "pos": [float(v) for v in c],
                             "quat": _q_euler(yaw, float(rng.uniform(-0.4, 0.4)), float(rng.uniform(-0.4, 0.4))),
                             "mat": mat, "collide": bool(collide), "size": [0, 0, 0]})
        return nm

    def prop(self, model, pos, yaw=0.0, scale=1.0, pitch=0.0, roll=0.0, collide=False, quat=None, fit=None, thermal=None):
        """A model of the look pack at `pos` (its footprint's centre, its lowest point), turned and scaled (`fit`: the
        (length, depth, height) it is scaled to). Every sensor sees it; with `collide` the robot touches the convex
        hull of each of its parts. Returns its size."""
        from . import props
        sz = props.size(model)
        if fit is not None:
            sc = [float(f) / max(1e-6, m) for f, m in zip(fit, sz)]
        else:
            sc = [float(v) for v in (np.ones(3) * np.asarray(scale, float))]
        self.props.append({"model": model, "pos": [float(v) for v in pos], "scale": sc, "collide": bool(collide),
                           "quat": [float(v) for v in quat] if quat is not None else _q_euler(yaw, pitch, roll),
                           "thermal": thermal})
        return [m * c for m, c in zip(sz, sc)]

    def keep(self, a, b, r=0.75):
        self.keepout.append((float(a[0]), float(a[1]), float(b[0]), float(b[1]), float(r)))

    def free_at(self, x, y, r):
        """True when a disc of radius r at (x, y) stays clear of every keep-out corridor."""
        for (x0, y0, x1, y1, rr) in self.keepout:
            dx, dy = x1 - x0, y1 - y0
            L2 = dx * dx + dy * dy
            t = 0.0 if L2 < 1e-9 else max(0.0, min(1.0, ((x - x0) * dx + (y - y0) * dy) / L2))
            px, py = x0 + t * dx, y0 + t * dy
            if math.hypot(x - px, y - py) < r + rr:
                return False
        return True

    # ------------------------------------------------------------------ heightfield helpers
    def mound(self, cx, cy, rx, ry, h, yaw=0.0, power=1.3, mask=None, dress=True):
        """A rubble mound (max-combined into the heightfield) and its rubble paint."""
        X = self.GX - cx
        Y = self.GY - cy
        c, s = math.cos(-yaw), math.sin(-yaw)
        u = c * X - s * Y
        v = s * X + c * Y
        r = np.sqrt((u / rx) ** 2 + (v / ry) ** 2)
        noise = 1.0 + 0.05 * np.sin(u * 4.1 + v * 2.3) * np.cos(v * 3.7 - u * 1.9)
        z = h * np.clip(1.0 - r, 0, 1) ** power * noise
        if mask is not None:
            z = z * mask
        on = z > 0.004
        self.H = np.where(on, np.maximum(self.H, z), self.H)
        self.ground[z > 0.03] = 4
        if dress:
            self._dress(cx, cy, rx, ry, h, yaw)

    def _dress(self, cx, cy, rx, ry, h, yaw):
        """Chunks of concrete, brick and plaster strewn over a mound (drawn and seen by the depth sensors; the tracks
        ride on the mound's body)."""
        rng = self.rng
        area = math.pi * rx * ry
        n = int(min(90, max(6, area * 9.0)))
        c, s = math.cos(yaw), math.sin(yaw)
        for _ in range(n):
            r = math.sqrt(float(rng.random())) * 0.95
            a = float(rng.uniform(0, 2 * math.pi))
            u, v = r * rx * math.cos(a), r * ry * math.sin(a)
            x = cx + c * u - s * v
            y = cy + s * u + c * v
            zz = self.height_at(x, y)
            if zz < 0.04:
                continue
            size = float(rng.uniform(0.08, 0.18 + 0.35 * min(1.0, h)))
            self.chunk((x, y, zz - 0.25 * size), size, mat=str(rng.choice(["rubble", "brick", "concrete", "concrete_dark",
                                                                             "plaster_white", "brick"])))

    def pit(self, x0, y0, x1, y1, depth):
        m = (self.GX >= x0) & (self.GX <= x1) & (self.GY >= y0) & (self.GY <= y1)
        self.H[m] = -depth
        self.ground[m] = 5

    def floor_tiles(self, x0, y0, x1, y1, cls=3):
        m = (self.GX >= x0) & (self.GX <= x1) & (self.GY >= y0) & (self.GY <= y1)
        g = self.ground
        g[m & (g == 2)] = cls

    def height_at(self, x, y):
        i = int(round((y - self.hy0) / HF_RES))
        j = int(round((x - self.hx0) / HF_RES))
        i = min(max(i, 0), self.hny - 1)
        j = min(max(j, 0), self.hnx - 1)
        return float(self.H[i, j])


# the look pack's vehicles (vehicles.npz): id -> (length, width, height) in metres, front at +x
VEHICLE_DIMS = {5: (5.61, 2.02, 2.37), 6: (4.50, 1.80, 2.01), 7: (4.24, 1.88, 1.72), 9: (5.27, 2.19, 2.13),
                10: (4.72, 1.96, 1.81), 11: (4.59, 1.99, 1.41), 13: (3.46, 1.79, 1.64)}


def have_props():
    """True when the look pack's models are installed (the task image): furniture, ruin pieces, the rescue post, the
    facade's window modules are then those models for every sensor; without the pack (a development checkout) the
    site is built of the primitives it used to be."""
    from . import props
    return props.have()


def _qmul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return [w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2]


def _qmat(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _mat_quat(R):
    """Quaternion (w, x, y, z) of a rotation matrix."""
    R = np.asarray(R, float)
    t = R[0, 0] + R[1, 1] + R[2, 2]
    if t > 0:
        r = math.sqrt(1.0 + t)
        q = [0.5 * r, (R[2, 1] - R[1, 2]) / (2 * r), (R[0, 2] - R[2, 0]) / (2 * r), (R[1, 0] - R[0, 1]) / (2 * r)]
    else:
        k = int(np.argmax([R[0, 0], R[1, 1], R[2, 2]]))
        a, b, c = k, (k + 1) % 3, (k + 2) % 3
        r = math.sqrt(1.0 + R[a, a] - R[b, b] - R[c, c])
        v = [0.0, 0.0, 0.0]
        v[a] = 0.5 * r
        v[b] = (R[b, a] + R[a, b]) / (2 * r)
        v[c] = (R[c, a] + R[a, c]) / (2 * r)
        q = [(R[c, b] - R[b, c]) / (2 * r)] + v
    n = math.sqrt(sum(x * x for x in q))
    return [float(x / n) for x in q]


def have_vehicles():
    """True when the look pack's vehicles are installed (the task image): the street's car and van are then real
    vehicles; without the pack (a development checkout) they are the boxes they used to be."""
    import os
    from . import pbr
    return os.path.isfile(os.path.join(pbr.cache_dir(), "vehicles.npz"))


# ============================================================================================================ build
def build_site(plan):
    S = Site(plan)
    _street(S)
    _corridors(S)
    _structure(S)
    _openings(S)
    _contents(S)
    _survivor_and_decoys(S)
    _neighbours(S)
    if have_props():
        _rescue_post(S)
        _ruins(S)
        _decals(S)
    _route(S)
    return S


def _bay(S, k):
    return S.plan["bays"][k]


def _street(S):
    W = S.W
    rng = S.rng
    # curb and the road markings
    S.box((W / 2, -3.0, STREET_Z / 2), (W / 2 + 5.0, 0.08, abs(STREET_Z) / 2 + 0.002), "concrete", name="curb")
    for k in range(int((W + 10) / 3.0)):
        x = -5 + 1.5 + k * 3.0
        S.box((x, -7.0, STREET_Z + 0.003), (0.75, 0.06, 0.004), "white_paint", collide=False, name="marking")
    # a crushed car on the street, a lamp post down, a dumpster, the team's rescue vehicle further out
    st = S.plan["start"]
    cx = float(rng.uniform(1.5, W - 1.5))
    if abs(cx - st["x"]) < 3.5:
        cx = st["x"] + (4.0 if st["x"] < W / 2 else -4.0)
    cy = -6.5
    yaw = float(rng.uniform(-0.25, 0.25))
    carm = str(rng.choice(["metal_red", "metal_blue", "metal_white", "metal_grey"]))
    real = have_vehicles()
    if real:
        # a real car (the look pack's saloon scaled to 4.2 m): its mesh is what the sensors see and, as convex pieces,
        # what the robot would touch
        sc = 4.2 / VEHICLE_DIMS[11][0]
        S.vehicles.append({"name": "car", "id": 11, "x": cx, "y": cy, "z": STREET_Z, "yaw": yaw, "scale": sc, "paint": carm})
        roof = VEHICLE_DIMS[11][2] * sc
    else:
        S.box((cx, cy, STREET_Z + 0.42), (2.1, 0.88, 0.30), carm, yaw=yaw, name="car_body", thermal="sunlit_metal")
        S.box((cx - 0.2, cy, STREET_Z + 0.82), (1.2, 0.80, 0.16), carm, yaw=yaw, pitch=0.05, name="car_cabin", thermal="sunlit_metal")
        S.box((cx - 0.2, cy, STREET_Z + 0.70), (1.25, 0.82, 0.06), "window", yaw=yaw, collide=False, name="car_glass")
        for dx in (-1.35, 1.35):
            for dy in (-0.82, 0.82):
                p = np.array([cx, cy, 0]) + _rz(yaw) @ np.array([dx, dy, 0])
                S.cyl((p[0], p[1], STREET_Z + 0.30), 0.30, 0.10, "black", roll=math.pi / 2, yaw=yaw, name="tyre")
        roof = 0.94
    # slab chunk on the car roof (the facade fell on it)
    S.box((cx + 0.3, cy + 0.2, STREET_Z + roof + 0.11), (0.9, 0.7, 0.11), "slab", yaw=yaw + 0.4, pitch=0.12, roll=-0.08, name="roofslab")
    # a lamp post down along the far kerb
    lx = float(rng.uniform(-2.0, W - 4.0))
    a = np.array([lx, -9.6, STREET_Z + 0.07])
    b = a + np.array([5.5, -0.35, 0.0])
    S.capsule(a, b, 0.07, "metal_grey", name="lamppost")
    # the rescue team's van and a tent at the far west of the street
    vx = -3.4
    if real:
        S.vehicles.append({"name": "van", "id": 6, "x": vx, "y": -7.5, "z": STREET_Z, "yaw": -math.pi / 2, "scale": 1.0,
                           "paint": "metal_white"})
    else:
        S.box((vx, -7.5, STREET_Z + 1.05), (1.0, 2.6, 0.95), "metal_white", name="van")
        S.box((vx + 1.01, -7.5, STREET_Z + 1.2), (0.004, 2.4, 0.12), "tarp_orange", collide=False, name="van_stripe")
        for dy in (-1.7, 1.7):
            for dx in (-0.9, 0.9):
                S.cyl((vx + dx, -7.5 + dy, STREET_Z + 0.36), 0.36, 0.12, "black", roll=math.pi / 2, yaw=math.pi / 2, name="vtyre")
    # facade rubble spills on the sidewalk in front of the front bays (not the entry: that one is built with its opening)
    entry_x = [o["x"] for o in S.plan["openings"] if o["a"] == "S" and o["on_route"]][0]
    for i in range(S.plan["nx"]):
        b = _bay(S, (i, 0))
        if b["dmg"] > 0.0 and rng.random() < 0.7:
            x = float(rng.uniform(b["x0"] + 0.5, b["x1"] - 0.5))
            if abs(x - entry_x) < 3.2:
                continue                                    # the way to the entry stays clear
            outside = (S.GY < -0.16).astype(float)            # the spill lies on the pavement, outside the facade
            S.mound(x, -0.3, float(rng.uniform(1.0, 2.0)), float(rng.uniform(0.8, 1.6)), float(rng.uniform(0.25, 0.7)),
                    mask=outside)
    # where a front bay pancaked or filled, its upper floors came down onto the pavement: slab sections leaning from
    # the facade onto a big spill (kept clear of the way to the entry)
    for i in range(S.plan["nx"]):
        b = _bay(S, (i, 0))
        if b["state"] not in ("void", "filled"):
            continue
        xm = (b["x0"] + b["x1"]) / 2
        if abs(xm - entry_x) < 3.6:
            continue
        hw = (b["x1"] - b["x0"]) / 2
        outside = (S.GY < -0.16).astype(float)
        S.mound(xm, -0.2, hw * 0.9, 2.2, float(rng.uniform(0.7, 1.2)), power=1.0, mask=outside)
        for k in range(int(rng.integers(2, 4))):
            L = float(rng.uniform(1.6, 2.6))
            ang = float(rng.uniform(0.35, 0.65))
            x = xm + float(rng.uniform(-hw * 0.6, hw * 0.6))
            yc = -math.cos(ang) * L / 2 + 0.1
            zc = math.sin(ang) * L / 2 + float(rng.uniform(0.1, 0.5))
            S.box((x, yc, zc), (float(rng.uniform(0.7, 1.3)), L / 2, 0.11), "slab", yaw=float(rng.uniform(-0.25, 0.25)),
                  roll=ang, name="fallenslab")
            for _r in range(4):
                xx = x + float(rng.uniform(-0.6, 0.6))
                S.capsule((xx, yc - L / 2 * math.cos(ang), zc - L / 2 * math.sin(ang) + 0.05),
                          (xx + float(rng.uniform(-0.1, 0.1)), yc - L / 2 * math.cos(ang) - 0.25, zc - L / 2 * math.sin(ang) + 0.12),
                          0.007, "rebar", collide=False, name="rebar")
    # broken glass / small debris on the sidewalk (visual)
    for _ in range(60):
        x = float(rng.uniform(-3, W + 3))
        y = float(rng.uniform(-3.0, -0.2))
        S.chunk((x, y, 0.02), float(rng.uniform(0.05, 0.15)), mat=str(rng.choice(["rubble", "brick", "concrete"])))


def _passage_dir(o):
    """(n, t): the unit passage direction from bay a to bay b and the along-wall direction."""
    if o["axis"] == "x":
        a, b = o["a"], o["b"]
        sgn = 1.0 if (a == "S" or b[0] > a[0]) else -1.0
        return np.array([sgn, 0.0]), np.array([0.0, 1.0])
    a, b = o["a"], o["b"]
    sgn = 1.0 if (a == "S" or (b != "S" and b[1] > a[1])) else -1.0
    return np.array([0.0, sgn]), np.array([1.0, 0.0])


def _bay_center(S, k):
    if k == "S":
        st = S.plan["start"]
        return np.array([st["x"], st["y"]])
    b = _bay(S, k)
    return np.array([(b["x0"] + b["x1"]) / 2.0, (b["y0"] + b["y1"]) / 2.0])


def _corridors(S):
    """Keep-out corridors: within every bay, from each of its passable openings to the bay's centre, and in the
    target bay to the survivor's approach point."""
    plan = S.plan
    for o in plan["openings"]:
        if o.get("challenge") == "prop":
            # nothing else stands under the propped slab or against its crate: the slab bears on the crate alone
            n, t = _passage_dir(o)
            p = np.array([o["x"], o["y"]])
            S.keep(p - t * (o["width"] / 2), p + t * (o["width"] / 2), 0.45)
        if o.get("challenge") in ("blocked", "prop"):
            continue
        n, t = _passage_dir(o)
        p = np.array([o["x"], o["y"]])
        for k, sgn in ((o["a"], -1.0), (o["b"], 1.0)):
            if k == "S":
                continue
            c = _bay_center(S, k)
            # the straight run to and through the opening (the robot lines up 2 m out), then on to the bay's centre
            q = p + sgn * n * 2.1
            S.keep(p, q, 0.8)
            S.keep(q, c, 0.8)
            S.keep(p + sgn * n * 1.0, c, 0.8)
    sv = plan["survivor"]
    t = plan["target"]
    c = _bay_center(S, t)
    S.keep(c, (sv["x"], sv["y"]), 0.8)


# ------------------------------------------------------------------------------------------------- structure
def _structure(S):
    plan, rng = S.plan, S.rng
    xs, ys, nx, ny = plan["xs"], plan["ys"], plan["nx"], plan["ny"]
    bays = plan["bays"]
    paints = ["plaster_white", "plaster_cream", "plaster_blue", "plaster_green", "plaster_pink"]
    for k, b in bays.items():
        b["paint"] = paints[int(rng.integers(len(paints)))]
        if b["room"] in ("shop", "flat"):
            S.floor_tiles(b["x0"], b["y0"], b["x1"], b["y1"], 3)
    # ---- ceilings / dropped slabs per bay
    for k, b in bays.items():
        x0, x1, y0, y1 = b["x0"], b["x1"], b["y0"], b["y1"]
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        hx, hy = (x1 - x0) / 2, (y1 - y0) / 2
        st = b["state"]
        if st in ("standing", "damaged"):
            S.box((cx, cy, CEIL + SLAB_T / 2), (hx + 0.02, hy + 0.02, SLAB_T / 2), "slab", name="ceil")
            if st == "damaged":
                # a broken ceiling panel hanging down at one corner (visual + collision high up)
                if rng.random() < 0.6:
                    ang = float(rng.uniform(0.25, 0.5))
                    S.box((cx + float(rng.uniform(-hx / 2, hx / 2)), cy + float(rng.uniform(-hy / 2, hy / 2)), CEIL - 0.45),
                          (0.9, 0.7, 0.06), "slab", yaw=float(rng.uniform(0, 3)), pitch=ang, collide=False, name="hangpanel")
                for _ in range(int(b.get("piles", 1))):
                    px = float(rng.uniform(x0 + 0.7, x1 - 0.7))
                    py = float(rng.uniform(y0 + 0.7, y1 - 0.7))
                    if S.free_at(px, py, 0.9):
                        S.mound(px, py, float(rng.uniform(0.6, 1.1)), float(rng.uniform(0.6, 1.1)), float(rng.uniform(0.3, 0.7)))
        elif st == "void":
            h = b["slab_h"]
            form = b.get("void_form", "pancake")
            # a slight tilt of the dropped slab; the clearance never falls below 0.47 m inside the corridors
            tilt_x = float(rng.uniform(-0.03, 0.03))
            tilt_y = float(rng.uniform(-0.03, 0.03))
            b["tilt"] = (tilt_x, tilt_y)
            S.box((cx, cy, h + SLAB_T / 2 + 0.03), (hx - 0.01, hy - 0.01, SLAB_T / 2), "slab", pitch=tilt_x, roll=tilt_y,
                  name="voidslab")
            # the upper storeys pancaked on top of it
            z = h + SLAB_T + 0.03
            for s_ in range(plan["storeys"] - 1):
                z += float(rng.uniform(0.25, 0.55))
                S.box((cx + float(rng.uniform(-0.3, 0.3)), cy + float(rng.uniform(-0.3, 0.3)), z + SLAB_T / 2),
                      (hx - 0.05, hy - 0.05, SLAB_T / 2), "slab", yaw=float(rng.uniform(-0.05, 0.05)),
                      pitch=float(rng.uniform(-0.06, 0.06)), roll=float(rng.uniform(-0.06, 0.06)), collide=False, name="pancake")
                # rubble between the layers, visible at the edges
                for _q in range(10):
                    ex = cx + float(rng.uniform(-hx, hx))
                    ey = cy + float(rng.uniform(-hy, hy))
                    if abs(ex - cx) < hx - 0.6 and abs(ey - cy) < hy - 0.6:
                        continue
                    S.chunk((ex, ey, z - 0.12), float(rng.uniform(0.2, 0.45)), mat=str(rng.choice(["rubble", "brick", "concrete"])))
                z += SLAB_T
            b["top_z"] = z
            # supports under the slab: rubble piles and crushed furniture along the walls (out of the corridors)
            for _ in range(6):
                side = int(rng.integers(4))
                if side == 0:
                    px, py = x0 + 0.35, float(rng.uniform(y0 + 0.4, y1 - 0.4))
                elif side == 1:
                    px, py = x1 - 0.35, float(rng.uniform(y0 + 0.4, y1 - 0.4))
                elif side == 2:
                    px, py = float(rng.uniform(x0 + 0.4, x1 - 0.4)), y0 + 0.35
                else:
                    px, py = float(rng.uniform(x0 + 0.4, x1 - 0.4)), y1 - 0.35
                if S.free_at(px, py, 0.45):
                    S.mound(px, py, 0.45, 0.45, min(h - 0.05, 0.45), dress=False)
                    S.chunk((px, py, min(h - 0.15, 0.35)), 0.3, mat="brick")
            # dust and debris on the floor under the slab: chunks too small to stop the tracks (visual)
            for _ in range(30):
                px = float(rng.uniform(x0 + 0.2, x1 - 0.2))
                py = float(rng.uniform(y0 + 0.2, y1 - 0.2))
                S.chunk((px, py, 0.01), float(rng.uniform(0.05, 0.12)), mat=str(rng.choice(["rubble", "brick", "concrete", "plaster_white"])))
            # rebar hanging from the slab's broken edge (visual)
            for _ in range(8):
                ex = float(rng.uniform(x0 + 0.3, x1 - 0.3))
                ey = y0 + 0.15 if rng.random() < 0.5 else y1 - 0.15
                a = (ex, ey, h + 0.02)
                bb = (ex + float(rng.uniform(-0.1, 0.1)), ey + float(rng.uniform(-0.1, 0.1)), h - float(rng.uniform(0.05, 0.2)))
                S.capsule(a, bb, 0.006, "rebar", collide=False, name="rebar")
        elif st in ("filled", "stair"):
            mh = float(rng.uniform(2.0, 2.8)) if st == "filled" else 1.6
            inside = ((S.GX > x0 + 0.1) & (S.GX < x1 - 0.1) & (S.GY > y0 + 0.1) & (S.GY < y1 - 0.1)).astype(float)
            S.mound(cx, cy, hx * 1.25, hy * 1.25, mh, power=0.7, mask=inside)
            for _ in range(5):
                S.box((cx + float(rng.uniform(-hx / 2, hx / 2)), cy + float(rng.uniform(-hy / 2, hy / 2)), mh * float(rng.uniform(0.4, 0.8))),
                      (float(rng.uniform(0.6, 1.4)), float(rng.uniform(0.4, 1.0)), SLAB_T / 2), "slab",
                      yaw=float(rng.uniform(0, 3)), pitch=float(rng.uniform(-0.6, 0.6)), roll=float(rng.uniform(-0.5, 0.5)),
                      collide=False, name="fillslab")
            for _ in range(40):
                S.chunk((cx + float(rng.uniform(-hx, hx)), cy + float(rng.uniform(-hy, hy)), mh * float(rng.uniform(0.3, 0.9))),
                        float(rng.uniform(0.2, 0.6)), mat=str(rng.choice(["rubble", "brick", "concrete"])))
            if st == "stair":
                for s_ in range(8):
                    S.box((cx - 0.6, y0 + 0.6 + s_ * 0.28, 0.09 + s_ * 0.17), (0.6, 0.14, 0.09), "concrete", name="stair")
            S.box((cx, cy, CEIL + SLAB_T / 2), (hx, hy, SLAB_T / 2), "slab", collide=False, name="ceilfill") if st == "stair" else None
    # ---- columns
    for i in range(nx + 1):
        for j in range(ny + 1):
            x, y = xs[i], ys[j]
            adj = [bays.get((i - a, j - c)) for a in (0, 1) for c in (0, 1)]
            adj = [b for b in adj if b is not None]
            hmin = CEIL
            for b in adj:
                if b["state"] == "void":
                    hmin = min(hmin, b["slab_h"])
            if hmin < CEIL:
                # a crushed column: a stub up to the dropped slab, rebar cage bulging (visual)
                S.box((x, y, hmin / 2), (COL / 2, COL / 2, hmin / 2), "concrete", name="colstub")
                for dx in (-1, 1):
                    for dy in (-1, 1):
                        S.capsule((x + dx * 0.15, y + dy * 0.15, 0.05), (x + dx * 0.21, y + dy * 0.2, hmin * 0.6), 0.008, "rebar",
                                  collide=False, name="cage")
            else:
                S.box((x, y, CEIL / 2), (COL / 2, COL / 2, CEIL / 2), "concrete", name="col")
    # ---- walls (every bay boundary; openings cut in _openings via S.wall_cuts)
    S.wall_segments = []
    for i in range(nx + 1):
        for j in range(ny):
            A = bays.get((i - 1, j))
            B = bays.get((i, j))
            S.wall_segments.append({"axis": "x", "c": xs[i], "lo": ys[j], "hi": ys[j + 1], "A": A, "B": B,
                                    "keyA": (i - 1, j) if A else None, "keyB": (i, j) if B else None,
                                    "ext": A is None or B is None})
    for j in range(ny + 1):
        for i in range(nx):
            A = bays.get((i, j - 1))
            B = bays.get((i, j))
            S.wall_segments.append({"axis": "y", "c": ys[j], "lo": xs[i], "hi": xs[i + 1], "A": A, "B": B,
                                    "keyA": (i, j - 1) if A else None, "keyB": (i, j) if B else None,
                                    "ext": A is None or B is None, "facade": j == 0})
    # ---- upper storeys over the standing part (exterior walls with windows, slabs) and the roof line
    for k, b in bays.items():
        if b["state"] in ("standing", "damaged"):
            x0, x1, y0, y1 = b["x0"], b["x1"], b["y0"], b["y1"]
            ns = plan["storeys"] if b["state"] == "standing" else max(1, plan["storeys"] - int(rng.integers(1, 3)))
            for s_ in range(1, ns):
                z = s_ * STOREY + CEIL
                S.box(((x0 + x1) / 2, (y0 + y1) / 2, z + SLAB_T / 2), ((x1 - x0) / 2 + 0.02, (y1 - y0) / 2 + 0.02, SLAB_T / 2),
                      "slab", collide=False, name="upslab")
            b["upper"] = ns
        else:
            b["upper"] = 0


def _wall_piece(S, seg, lo, hi, z0, z1, mat_in, broken_top=False):
    """A straight piece of wall along the segment between lo and hi (along-wall coords), from z0 to z1."""
    if hi - lo < 0.02 or z1 - z0 < 0.02:
        return
    t = T_EXT if seg["ext"] else T_INT
    mid = (lo + hi) / 2.0
    if seg["axis"] == "x":
        c = (seg["c"], mid, (z0 + z1) / 2)
        half = (t / 2, (hi - lo) / 2, (z1 - z0) / 2)
    else:
        c = (mid, seg["c"], (z0 + z1) / 2)
        half = ((hi - lo) / 2, t / 2, (z1 - z0) / 2)
    S.box(c, half, mat_in, name="wall")
    if broken_top:
        rng = S.rng
        n = max(1, int((hi - lo) / 0.35))
        for k in range(n):
            u = lo + (k + 0.5) * (hi - lo) / n
            hh = float(rng.uniform(0.05, 0.35))
            if seg["axis"] == "x":
                S.chunk((seg["c"], u, z1 + hh * 0.3), float(rng.uniform(0.18, 0.35)), mat="brick")
            else:
                S.chunk((u, seg["c"], z1 + hh * 0.3), float(rng.uniform(0.18, 0.35)), mat="brick")


def _openings(S):
    """Walls with their openings cut out, and what each opening holds (the challenges)."""
    plan, rng = S.plan, S.rng
    by_seg = {}
    for o in plan["openings"]:
        for idx, seg in enumerate(S.wall_segments):
            if seg["axis"] != o["axis"]:
                continue
            c = o["x"] if o["axis"] == "x" else o["y"]
            u = o["y"] if o["axis"] == "x" else o["x"]
            if abs(seg["c"] - c) < 1e-6 and seg["lo"] - 1e-6 <= u <= seg["hi"] + 1e-6:
                by_seg.setdefault(idx, []).append(o)
                o["_seg"] = idx
                break
    for idx, seg in enumerate(S.wall_segments):
        A, B = seg["A"], seg["B"]
        # wall height: limited by a void slab on either side (the slab rests on the wall)
        top = CEIL
        broken = False
        for X in (A, B):
            if X is None:
                continue
            if X["state"] == "void":
                top = min(top, X["slab_h"] + 0.03)
            if X["state"] in ("damaged", "void", "filled"):
                broken = True
        if top >= CEIL:
            broken = False
        paint = (B or A)["paint"] if (A or B) else "plaster_white"
        mat = "brick" if seg["ext"] and not seg.get("facade") else ("plaster_cream" if seg.get("facade") else paint)
        if seg["ext"] and (A is None or B is None):
            X = A or B
            if X["state"] == "filled" and not seg.get("facade"):
                top = min(top, 2.0)
        ops = sorted(by_seg.get(idx, []), key=lambda o: (o["y"] if o["axis"] == "x" else o["x"]))
        cur = seg["lo"]
        for o in ops:
            u = o["y"] if o["axis"] == "x" else o["x"]
            w = o["width"]
            if o.get("challenge") == "debris":
                w = max(w, 1.25)
                o["width"] = w
            _wall_piece(S, seg, cur, u - w / 2, 0.0, top, mat, broken_top=broken)
            # head of the opening: a lintel / the broken wall above
            if o["form"] == "door" and top > 2.15:
                _wall_piece(S, seg, u - w / 2, u + w / 2, 2.1, top, mat)
            elif o["form"] == "shopfront":
                _wall_piece(S, seg, u - w / 2, u + w / 2, 2.6, top, mat)
            else:
                hb = min(top, float(rng.uniform(1.4, 2.2)))
                if top - hb > 0.05:
                    _wall_piece(S, seg, u - w / 2, u + w / 2, hb, top, mat, broken_top=False)
            cur = u + w / 2
            _challenge(S, o)
        _wall_piece(S, seg, cur, seg["hi"], 0.0, top, mat, broken_top=broken)
        # the facade: shop windows (sill 0.7 m, empty frames) on the parts without an opening
        if seg.get("facade"):
            _shop_windows(S, seg, ops)
    # upper-storey facade walls (outer shell of the standing part), visual
    for seg in S.wall_segments:
        if not seg["ext"]:
            continue
        X = seg["A"] or seg["B"]
        ns = X.get("upper", 0)
        for s_ in range(1, ns):
            z0 = s_ * STOREY
            _facade_storey(S, seg, z0)


def _shop_windows(S, seg, ops):
    """Openings of the facade's ground floor where no way in was planned: shop windows with a 0.75 m sill."""
    rng = S.rng
    lo, hi = seg["lo"], seg["hi"]
    for o in ops:
        u = o["x"]
        if abs(u - (lo + hi) / 2) < 3:
            pass
    # nothing cut: the facade piece already stands; draw a window frame on its outer face for the look
    n = 1 if hi - lo < 4.6 else 2
    for k in range(n):
        u = lo + (k + 0.5) * (hi - lo) / n
        if any(abs(u - o["x"]) < (o["width"] / 2 + 1.0) for o in ops):
            continue
        ww = min(1.6, (hi - lo) / n - 1.0)
        S.box((u, seg["c"] - T_EXT / 2 - 0.004, 1.55), (ww / 2, 0.006, 0.75), "window", collide=False, name="shopwin")
        S.box((u, seg["c"] - T_EXT / 2 - 0.01, 0.78), (ww / 2 + 0.05, 0.012, 0.03), "metal_grey", collide=False, name="sill")
        # a sign board above
        S.box((u, seg["c"] - T_EXT / 2 - 0.02, 2.75), (ww / 2 + 0.2, 0.02, 0.18),
              str(rng.choice(["metal_red", "metal_blue", "metal_white", "yellow"])), collide=False, name="sign")


def _module(S, seg, z0, lo=None, hi=None, height=STOREY):
    """A window module of the look pack (an exterior shell, 4 m x 3 m) on the outside of an exterior wall segment,
    scaled to the piece's length and the storey's height, its back on the wall's face."""
    from . import props
    lo = seg["lo"] if lo is None else lo
    hi = seg["hi"] if hi is None else hi
    model = ["win_18", "win_21", "win_20", "win_22"][int(S.rng.integers(4))]
    out = -1.0 if seg["A"] is None else 1.0                        # the side without a bay is outside
    off = out * (T_EXT / 2 + 0.01 + props.size(model)[1] / 2)
    mid = (lo + hi) / 2
    if seg["axis"] == "x":
        pos, yaw = (seg["c"] + off, mid, z0), (math.pi / 2 if out > 0 else -math.pi / 2)
    else:
        pos, yaw = (mid, seg["c"] + off, z0), (math.pi if out > 0 else 0.0)
    S.prop(model, pos, yaw=yaw, fit=(hi - lo, props.size(model)[1], height))


def _facade_storey(S, seg, z0):
    rng = S.rng
    lo, hi = seg["lo"], seg["hi"]
    if have_props():
        _module(S, seg, z0)
    t = T_EXT
    mid = (lo + hi) / 2
    L = hi - lo
    # wall below and above the window band, piers between windows
    def piece(a, b, za, zb, mat="plaster_cream"):
        m = (a + b) / 2
        if seg["axis"] == "x":
            S.box((seg["c"], m, (za + zb) / 2), (t / 2, (b - a) / 2, (zb - za) / 2), mat, collide=False, name="upwall")
        else:
            S.box((m, seg["c"], (za + zb) / 2), ((b - a) / 2, t / 2, (zb - za) / 2), mat, collide=False, name="upwall")
    piece(lo, hi, z0, z0 + 0.9)
    piece(lo, hi, z0 + 2.3, z0 + STOREY)
    nw = max(1, int(L / 2.2))
    ww = 1.2
    cur = lo
    for k in range(nw):
        u = lo + (k + 0.5) * L / nw
        piece(cur, u - ww / 2, z0 + 0.9, z0 + 2.3)
        # window: dark glass inset
        if seg["axis"] == "x":
            S.box((seg["c"], u, z0 + 1.6), (0.02, ww / 2, 0.7), "window", collide=False, name="upwin")
        else:
            S.box((u, seg["c"], z0 + 1.6), (ww / 2, 0.02, 0.7), "window", collide=False, name="upwin")
        cur = u + ww / 2
    piece(cur, hi, z0 + 0.9, z0 + 2.3)


# ------------------------------------------------------------------------------------------------- challenges
def _challenge(S, o):
    rng = S.rng
    ch = o.get("challenge") or "none"
    n, t = _passage_dir(o)
    p = np.array([o["x"], o["y"]])
    w = o["width"]
    seg = S.wall_segments[o["_seg"]]
    th = T_EXT if seg["ext"] else T_INT
    o["geom"] = {}
    kinds = [ch] + list(o.get("extra", []))
    for kind in kinds:
        if kind == "spill":
            # a rubble slope up from the sidewalk to the broken shop front, down into the shop
            h = float(rng.uniform(0.45, 0.70))
            S.mound(p[0], p[1] - 0.2, w / 2 + 1.6, 1.9, h, power=1.1)
            for _ in range(25):
                S.chunk((p[0] + float(rng.uniform(-w / 2 - 1.2, w / 2 + 1.2)), p[1] + float(rng.uniform(-1.6, 1.2)),
                         S.height_at(p[0], p[1]) * 0.6), float(rng.uniform(0.1, 0.3)), mat=str(rng.choice(["rubble", "brick", "concrete"])))
            o["geom"]["spill_h"] = h
        elif kind == "debris":
            # a timber beam across the opening, its ends in notches of the brick heaps at the jambs: it cannot be
            # pushed through, only lifted out
            hb = float(rng.uniform(0.32, 0.38))
            beam_len = w - 0.06
            pile_w = 0.20
            for sg in (-1.0, 1.0):
                q = p + t * sg * (w / 2 - pile_w / 2)
                S.box((q[0], q[1], hb / 2), (pile_w / 2 if o["axis"] == "y" else th / 2 + 0.06,
                                              pile_w / 2 if o["axis"] == "x" else th / 2 + 0.06, hb / 2), "brick", name="jambpile")
                # lips of the notch, before and after the beam (it lifts out once 4 cm up)
                for sn in (-1.0, 1.0):
                    r = q + n * sn * 0.095
                    S.box((r[0], r[1], hb + 0.02), (pile_w / 2 if o["axis"] == "y" else 0.012, pile_w / 2 if o["axis"] == "x" else 0.012, 0.02),
                          "brick", name="notch")
            mat = str(rng.choice(["wood", "wood_dark", "rust"]))
            yaw = 0.0 if o["axis"] == "y" else math.pi / 2
            S.bodies.append({"name": "debris_%d" % len(S.bodies), "kind": "beam", "pos": [p[0], p[1], hb + 0.045],
                             "quat": _q_euler(yaw, 0, 0), "mass": float(rng.uniform(4.0, 6.0)),
                             "geoms": [{"type": "box", "size": [beam_len / 2, 0.045, 0.045], "mat": mat}],
                             "graspable": True, "on_route": o["on_route"], "opening": o["order"]})
            o["geom"]["beam"] = S.bodies[-1]["name"]
            o["geom"]["beam_z"] = hb + 0.045
            o["width_clear"] = w - 2 * pile_w
        elif kind == "narrow":
            wn = float(rng.uniform(0.60, 0.66))
            # a fallen concrete block fills the opening up to wn from one jamb
            side = -1.0 if rng.random() < 0.5 else 1.0
            fill = w - wn
            q = p + t * side * (w / 2 - fill / 2)
            hz = float(rng.uniform(0.8, 1.3))
            S.box((q[0], q[1], hz / 2), (fill / 2 if o["axis"] == "y" else 0.55, fill / 2 if o["axis"] == "x" else 0.55, hz / 2),
                  "concrete_dark", name="narrowblock")
            # the other jamb's wall end has a buttress of rubble along the passage
            q2 = p - t * side * (w / 2 + 0.12)
            S.box((q2[0], q2[1], 0.5), (0.12 if o["axis"] == "y" else 0.55, 0.12 if o["axis"] == "x" else 0.55, 0.5), "brick", name="narrowjamb")
            for _ in range(5):
                r = q + n * float(rng.uniform(-0.5, 0.5))
                S.chunk((r[0], r[1], hz), float(rng.uniform(0.2, 0.35)), mat="concrete")
            o["geom"]["narrow_w"] = wn
            o["geom"]["narrow_center"] = (p - t * side * (fill / 2)).tolist()
        elif kind == "step":
            # a broken floor slab wedged in the doorway: the passage is raised by hs for STEP_HALF either side of the
            # wall, between the jambs (no way round): climb up on one side, step down on the other
            hs = float(rng.uniform(0.29, 0.33))
            STEP_HALF = 0.65
            if o["axis"] == "x":
                half = (STEP_HALF, w / 2 + 0.02, hs / 2)
            else:
                half = (w / 2 + 0.02, STEP_HALF, hs / 2)
            S.box((p[0], p[1], hs / 2), half, "slab", name="stepslab")
            for _ in range(5):
                r = p + n * float(rng.uniform(-STEP_HALF, STEP_HALF)) + t * float(rng.uniform(-w / 2, w / 2))
                S.chunk((r[0], r[1], hs + 0.02), float(rng.uniform(0.05, 0.10)), mat="rubble")
            # rebar sticking out of its broken edges (visual)
            for sg in (-1.0, 1.0):
                for _ in range(3):
                    r = p + n * sg * STEP_HALF + t * float(rng.uniform(-w / 2, w / 2))
                    S.capsule((r[0], r[1], hs * 0.5), (r[0] + n[0] * sg * 0.12, r[1] + n[1] * sg * 0.12, hs * 0.5 + 0.05), 0.006,
                              "rebar", collide=False, name="rebar")
            o["geom"]["step_h"] = hs
            o["geom"]["step_near"] = STEP_HALF
        elif kind == "gap":
            g = float(rng.uniform(0.44, 0.52))
            bk = o["b"]
            B = _bay(S, bk)
            d0 = th / 2 + float(rng.uniform(1.0, 1.3))
            # the pit is cut wider in the heightfield (its cells are 8 cm) and the floor slab's broken edges (boxes)
            # either side make the opening exactly g wide
            ea = p + n * d0
            eb = p + n * (d0 + g)
            for (c0, sgn) in ((ea, -1.0), (eb, 1.0)):
                cc = c0 + n * sgn * 0.25
                if o["axis"] == "x":
                    S.box((cc[0], (B["y0"] + B["y1"]) / 2, -0.09), (0.25, (B["y1"] - B["y0"]) / 2, 0.09), "slab", name="gapedge")
                else:
                    S.box(((B["x0"] + B["x1"]) / 2, cc[1], -0.09), ((B["x1"] - B["x0"]) / 2, 0.25, 0.09), "slab", name="gapedge")
            m0 = p + n * (d0 - 0.12)
            m1 = p + n * (d0 + g + 0.12)
            if o["axis"] == "x":
                S.pit(min(m0[0], m1[0]), B["y0"], max(m0[0], m1[0]), B["y1"], HF_BASE - 0.05)
            else:
                S.pit(B["x0"], min(m0[1], m1[1]), B["x1"], max(m0[1], m1[1]), HF_BASE - 0.05)
            o["geom"]["gap_w"] = g
            o["geom"]["gap_d0"] = d0
        elif kind == "blocked":
            # a slab section stands in the opening, leaning on rubble: no way through
            S.box((p[0], p[1], 0.9), (w / 2 + 0.1 if o["axis"] == "y" else 0.18, w / 2 + 0.1 if o["axis"] == "x" else 0.18, 0.9),
                  "slab", yaw=0.0, pitch=float(rng.uniform(-0.15, 0.15)), name="blockslab")
            S.mound(p[0], p[1], 1.2, 1.2, float(rng.uniform(0.7, 1.1)))
        elif kind == "prop":
            # a broken slab section lies across the opening: one end on a wooden crate standing in the middle of the
            # opening, the other on a brick heap at one jamb. Both are free bodies in the physics: the slab's weight
            # really bears on the crate (about half of it; the heap carries the rest), so a touch does not shift the
            # crate, and a crate pushed or pulled out lets the slab come down (a secondary collapse). Too little room
            # either side of the crate, and too little height under the slab, to pass.
            crate_h, cx, cy = 0.42, 0.22, 0.17
            # the crate and the heap stand on level floor
            m_ = (S.GX - p[0]) ** 2 + (S.GY - p[1]) ** 2 < 0.9 ** 2
            S.H[m_] = 0.0
            S.bodies.append({"name": "prop_%d" % len(S.bodies), "kind": "prop", "pos": [p[0], p[1], crate_h / 2 + 0.002],
                             "quat": _q_euler(0.0 if o["axis"] == "y" else math.pi / 2), "mass": 4.0,
                             "geoms": [{"type": "box", "size": [cx, cy, crate_h / 2], "mat": "wood"}],
                             "graspable": True, "on_route": False})
            prop = S.bodies[-1]["name"]
            side = -1.0 if rng.random() < 0.5 else 1.0
            pile_half = 0.10
            x_pile = w / 2 - pile_half                      # the heap's centre, along t from the opening's centre
            q = p + t * side * x_pile
            S.box((q[0], q[1], crate_h / 2), (pile_half if o["axis"] == "y" else 0.34, pile_half if o["axis"] == "x" else 0.34,
                                              crate_h / 2), "brick", name="proppile")
            slab_t = 0.09                                   # half the slab's thickness
            x0, x1 = -0.20, x_pile + pile_half - 0.02       # the slab's span along t (side-normalised)
            Ls = x1 - x0
            mid = p + t * side * ((x0 + x1) / 2.0)
            zc = crate_h + 0.002 + slab_t + 0.003           # set down 3 mm above its two supports
            base_yaw = 0.0 if o["axis"] == "y" else math.pi / 2
            S.bodies.append({"name": "propslab_%d" % len(S.bodies), "kind": "propslab", "pos": [mid[0], mid[1], zc],
                             "quat": _q_euler(base_yaw, 0.0, 0.0), "mass": 180.0,
                             "geoms": [{"type": "box", "size": [Ls / 2, 0.32, slab_t], "mat": "slab"}],
                             "graspable": False, "held_by": prop, "on_route": False})
            o["geom"]["prop"] = prop
    # the wall's thickness at the opening, for the route
    o["thickness"] = th


# ------------------------------------------------------------------------------------------------- contents
FURNITURE = {
    "shop": ["shelf", "shelf", "counter", "boxes", "fridge", "shelf_down"],
    "flat": ["bed", "wardrobe_down", "table", "sofa", "kitchen", "chair", "tv"],
    "office": ["desk", "desk", "cabinet", "chair", "cabinet_down"],
    "corridor": ["boxes", "bike", "cabinet"],
    "lane": [],
    "stair": [],
}


def _contents(S):
    plan, rng = S.plan, S.rng
    for k, b in plan["bays"].items():
        if b["state"] in ("filled", "stair"):
            continue
        void = b["state"] == "void"
        hmax = (b["slab_h"] - 0.05) if void else CEIL - 0.3
        items = list(FURNITURE.get(b["room"], []))
        rng.shuffle(items)
        x0, x1, y0, y1 = b["x0"] + 0.2, b["x1"] - 0.2, b["y0"] + 0.2, b["y1"] - 0.2
        for it in items:
            for _try in range(12):
                # along a wall, facing in
                side = int(rng.integers(4))
                L, Dp, Hh = _fsize(it, rng)
                Hh = min(Hh, hmax)
                if side in (0, 1):
                    x = x0 + Dp / 2 + 0.02 if side == 0 else x1 - Dp / 2 - 0.02
                    y = float(rng.uniform(y0 + L / 2, max(y0 + L / 2 + 0.01, y1 - L / 2)))
                    yaw = math.pi / 2
                else:
                    y = y0 + Dp / 2 + 0.02 if side == 2 else y1 - Dp / 2 - 0.02
                    x = float(rng.uniform(x0 + L / 2, max(x0 + L / 2 + 0.01, x1 - L / 2)))
                    yaw = 0.0
                r = math.hypot(L, Dp) / 2
                if S.free_at(x, y, r) and _clear_of_items(S, k, x, y, r):
                    _furniture(S, it, x, y, yaw, L, Dp, Hh, void, rng, face=1.0 if side in (1, 2) else -1.0, hmax=hmax)
                    b.setdefault("items", []).append((x, y, r))
                    break
        # plaster flakes, bricks, paper: visual dressing on the floor
        for _ in range(25 if not void else 10):
            px = float(rng.uniform(x0, x1))
            py = float(rng.uniform(y0, y1))
            S.chunk((px, py, 0.01), float(rng.uniform(0.05, 0.14)), mat=str(rng.choice(["plaster_white", "brick", "rubble", b["paint"]])))


def _clear_of_items(S, k, x, y, r):
    for (ix, iy, ir) in S.plan["bays"][k].get("items", []):
        if math.hypot(x - ix, y - iy) < r + ir + 0.05:
            return False
    return True


def _fsize(it, rng):
    """(length along the wall, depth, height) of a furniture item."""
    if have_props():                                    # the sizes of the look pack's pieces
        return {"shelf": (1.2, 0.5, 1.9), "shelf_down": (1.9, 1.0, 0.45), "counter": (1.6, 0.6, 0.95),
                "boxes": (1.0, 0.65, 0.78), "fridge": (0.7, 0.7, 1.75), "bed": (2.0, 1.0, 0.56), "wardrobe_down": (1.8, 1.75, 0.4),
                "table": (1.2, 0.75, 0.75), "sofa": (2.3, 1.0, 0.56), "kitchen": (1.8, 0.65, 0.9), "chair": (0.5, 0.5, 0.9),
                "tv": (1.0, 0.5, 0.95), "desk": (1.4, 0.75, 0.75), "cabinet": (1.0, 0.55, 1.33), "cabinet_down": (1.3, 0.69, 0.58),
                "bike": (0.6, 0.7, 1.4)}[it]
    return {"shelf": (1.2, 0.4, 1.9), "shelf_down": (1.9, 0.4, 0.45), "counter": (1.6, 0.6, 0.95),
            "boxes": (0.9, 0.6, 0.6), "fridge": (0.65, 0.65, 1.75), "bed": (2.0, 1.4, 0.5), "wardrobe_down": (1.9, 0.6, 0.6),
            "table": (1.2, 0.8, 0.75), "sofa": (1.9, 0.85, 0.85), "kitchen": (1.8, 0.6, 0.9), "chair": (0.5, 0.5, 0.9),
            "tv": (1.0, 0.4, 0.55), "desk": (1.4, 0.7, 0.75), "cabinet": (0.5, 0.6, 1.3), "cabinet_down": (1.3, 0.6, 0.5),
            "bike": (1.7, 0.4, 1.0)}[it]


def _piece(S, model, x, y, yaw, size, face=-1.0, z0=0.0, name="furn", collide=True, thermal=None):
    """A real piece of furniture: the look pack's model scaled into the box `size` (length, depth, height) standing at
    (x, y, z0), its front to the room (`face`: on which side of the piece, +y or -y, the room is). Every sensor sees
    the model; the box is what the robot touches (no sensor draws it)."""
    L, Dp, Hh = size
    if collide:
        S.box((x, y, z0 + Hh / 2), (L / 2, Dp / 2, Hh / 2), "wood", yaw=yaw, name=name, group=3)
    S.prop(model, (x, y, z0), yaw=yaw + (math.pi if face > 0 else 0.0), fit=size, thermal=thermal)


def _furniture_real(S, it, x, y, yaw, L, Dp, Hh, rng, face, hmax):
    """The item as pieces of the look pack. Hh is the item's height, already cut to the ceiling over it: under a low
    ceiling a tall piece gives way to a low one."""
    R = _rz(yaw)

    def at(dx, dy):
        v = R @ np.array([dx, dy, 0.0])
        return x + v[0], y + v[1]

    tall = Hh >= _fsize(it, rng)[2] - 1e-6
    if it in ("shelf", "cabinet", "fridge"):
        if tall:
            _piece(S, {"shelf": "shelf", "cabinet": "closet", "fridge": "fridge"}[it], x, y, yaw, (L, Dp, Hh), face, name=it)
        else:
            _piece(S, "sideboard", x, y, yaw, (min(1.3, max(L, 1.0)), 0.69, min(Hh, 0.58)), face, name=it)
    elif it in ("shelf_down", "wardrobe_down", "cabinet_down"):
        model = {"shelf_down": "shelf_down", "wardrobe_down": "wardrobe_down", "cabinet_down": "sideboard"}[it]
        _piece(S, model, x, y, yaw, (L, Dp, min(Hh, _fsize(it, rng)[2])), face, name=it)
    elif it == "counter":
        _piece(S, "counter", x, y, yaw, (L, Dp, Hh), face, name=it)
    elif it == "kitchen":
        a, b = at(-L / 4, 0.0), at(L / 4, 0.0)
        _piece(S, "kitchen", a[0], a[1], yaw, (L / 2, Dp, Hh), face, name=it)
        _piece(S, "stove", b[0], b[1], yaw, (L / 2 - 0.1, Dp, Hh), face, name="stove")
    elif it == "boxes":
        for k, (dx, dy, dz) in enumerate(((-0.26, 0.0, 0.0), (0.26, 0.03, 0.0), (0.02, 0.0, 0.39))):
            if dz + 0.39 > Hh:
                continue
            c = at(dx, dy)
            _piece(S, "carton", c[0], c[1], yaw + float(rng.uniform(-0.3, 0.3)), (0.45, 0.6, 0.39), face, z0=dz, name="carton")
    elif it == "bed":
        _piece(S, "bed", x, y, yaw, (L, Dp, 1.08 if hmax >= 1.15 else min(Hh, 0.56)), face, name="bed")
    elif it in ("table", "desk"):
        _piece(S, "table", x, y, yaw, (L, Dp, Hh), face, name=it)
    elif it == "sofa":
        _piece(S, "sofa", x, y, yaw, (L, Dp, Hh), face, name="sofa")
    elif it == "chair":
        if Hh >= 0.7:
            _piece(S, "chair", x, y, yaw + float(rng.uniform(-0.5, 0.5)), (0.5, 0.5, Hh), face, name="chair")
    elif it == "tv":
        _piece(S, "sideboard", x, y, yaw, (L, Dp, min(Hh, 0.5)), face, name="tvstand")
        if Hh >= 0.95 - 1e-6:
            S.prop("tv", (x, y, 0.5), yaw=yaw + (math.pi if face > 0 else 0.0), scale=1.15)
    elif it == "bike":
        if Hh >= 1.4 - 1e-6:
            _piece(S, "handtruck", x, y, yaw, (L, Dp, Hh), face, name="handtruck")


def _furniture(S, it, x, y, yaw, L, Dp, Hh, void, rng, face=-1.0, hmax=3.0):
    if have_props():
        return _furniture_real(S, it, x, y, yaw, L, Dp, Hh, rng, face, hmax)
    R = _rz(yaw)

    def at(dx, dy):
        v = R @ np.array([dx, dy, 0.0])
        return x + v[0], y + v[1]

    tilt = float(rng.uniform(-0.06, 0.06))
    if it in ("shelf", "cabinet", "fridge"):
        mat = {"shelf": "metal_grey", "cabinet": "metal_grey", "fridge": "metal_white"}[it]
        if void:
            Hh = min(Hh, 0.45)
        S.box((x, y, Hh / 2), (L / 2, Dp / 2, Hh / 2), mat, yaw=yaw, roll=tilt, name=it)
        if it == "shelf" and not void:
            for zz in (0.4, 0.8, 1.2, 1.6):
                for dx in (-0.4, 0.0, 0.4):
                    if rng.random() < 0.5:
                        c = at(dx, -0.05)
                        S.box((c[0], c[1], zz + 0.08), (0.12, 0.1, 0.08), str(rng.choice(["metal_red", "metal_blue", "yellow", "white_paint"])),
                              yaw=yaw + float(rng.uniform(-0.3, 0.3)), collide=False, name="goods")
    elif it in ("shelf_down", "wardrobe_down", "cabinet_down"):
        mat = "metal_grey" if it != "wardrobe_down" else "wood"
        S.box((x, y, Hh / 2), (L / 2, Dp / 2, Hh / 2), mat, yaw=yaw, pitch=0.0, roll=tilt, name=it)
        for _ in range(4):
            c = at(float(rng.uniform(-L / 2, L / 2)), float(rng.uniform(-0.6, -0.3)))
            S.box((c[0], c[1], 0.05), (0.1, 0.08, 0.05), str(rng.choice(["metal_red", "metal_blue", "yellow", "white_paint", "wood"])),
                  yaw=float(rng.uniform(0, 3)), collide=False, name="spill")
    elif it == "counter" or it == "kitchen":
        S.box((x, y, Hh / 2), (L / 2, Dp / 2, Hh / 2), "wood_dark" if it == "counter" else "metal_white", yaw=yaw, name=it)
        c = at(0, 0)
        S.box((c[0], c[1], Hh + 0.02), (L / 2 + 0.02, Dp / 2 + 0.02, 0.02), "concrete_dark", yaw=yaw, name=it + "top")
    elif it == "boxes":
        for dx, dy, dz in ((-0.25, 0, 0.15), (0.25, 0.05, 0.15), (0.0, 0.0, 0.45)):
            c = at(dx, dy)
            S.box((c[0], c[1], dz), (0.22, 0.2, 0.15), "wood", yaw=yaw + float(rng.uniform(-0.3, 0.3)), name="carton")
    elif it == "bed":
        S.box((x, y, 0.2), (L / 2, Dp / 2, 0.2), "wood", yaw=yaw, name="bedframe")
        c = at(0, 0)
        S.box((c[0], c[1], 0.47), (L / 2 - 0.05, Dp / 2 - 0.05, 0.08), str(rng.choice(["fabric_blue", "fabric_grey", "fabric_red"])),
              yaw=yaw, roll=tilt, name="mattress")
    elif it == "table" or it == "desk":
        top = min(Hh, 0.75)
        if void:
            top = min(top, 0.45)
        c = at(0, 0)
        S.box((c[0], c[1], top - 0.02), (L / 2, Dp / 2, 0.025), "wood", yaw=yaw, roll=tilt, name=it + "top")
        for dx in (-L / 2 + 0.05, L / 2 - 0.05):
            for dy in (-Dp / 2 + 0.05, Dp / 2 - 0.05):
                q = at(dx, dy)
                S.box((q[0], q[1], (top - 0.04) / 2), (0.025, 0.025, (top - 0.04) / 2), "wood_dark", yaw=yaw, name="leg")
    elif it == "sofa":
        S.box((x, y, 0.22), (L / 2, Dp / 2, 0.22), "fabric_green", yaw=yaw, name="sofa")
        c = at(0, Dp / 2 - 0.1)
        S.box((c[0], c[1], 0.55), (L / 2, 0.1, 0.33 if not void else 0.15), "fabric_green", yaw=yaw, name="sofaback")
    elif it == "chair":
        S.box((x, y, 0.45), (0.22, 0.22, 0.025), "wood", yaw=yaw + 0.4, pitch=1.2, name="chairseat")
        c = at(0.1, 0.0)
        S.box((c[0], c[1], 0.15), (0.03, 0.22, 0.15), "wood_dark", yaw=yaw + 0.4, name="chairback")
    elif it == "tv":
        S.box((x, y, 0.25), (L / 2, Dp / 2, 0.25), "wood_dark", yaw=yaw, name="tvstand")
        c = at(0, 0)
        S.box((c[0], c[1], 0.53), (0.55, 0.04, 0.03), "black", yaw=yaw, pitch=1.4, collide=False, name="tv")
    elif it == "bike":
        a = at(-0.7, 0)
        b = at(0.7, 0)
        S.cyl((a[0], a[1], 0.33), 0.33, 0.02, "black", yaw=yaw, roll=math.pi / 2, collide=False, name="wheel")
        S.cyl((b[0], b[1], 0.33), 0.33, 0.02, "black", yaw=yaw, roll=math.pi / 2, collide=False, name="wheel")
        S.capsule((a[0], a[1], 0.33), (b[0], b[1], 0.6), 0.02, "metal_red", collide=False, name="frame")


# ------------------------------------------------------------------------------------------------- people
CLOTH_MATS = {}


def _clear_floor(S, px, py, yaw, lay, margin=0.25, ramp=0.5):
    """The floor under a lying figure is the bare floor: a rubble mound that reaches under the body is cut back to
    `margin` from the body's outline and rises from there over `ramp` (the body lies on the floor, not in a heap)."""
    c, s_ = math.cos(-yaw), math.sin(-yaw)
    X, Y = S.GX - px, S.GY - py
    u, v = c * X - s_ * Y, s_ * X + c * Y
    du = np.maximum(0.0, np.maximum(lay["x0"] - u, u - lay["x1"]))
    dv = np.maximum(0.0, np.maximum(-lay["right"] - v, v - lay["left"]))
    k = np.clip((np.hypot(du, dv) - margin) / ramp, 0.0, 1.0)
    up = S.H > 0.0
    S.H = np.where(up, S.H * k, S.H)
    S.ground[up & (k < 0.2) & (S.ground == 4)] = 2


def _trapped(S, sv, lay, px, py, yaw, rng):
    """How the survivor is trapped, as one arrangement that holds together: they lie along the void's far wall; a
    broken slab piece bridges their lower legs, resting on a stack of debris either side of the legs and clear of
    them by a hand's breadth; a low cupboard that kept the pocket open stands against the wall beyond their feet (or
    beyond their head, where the bay has the room). Nothing passes through the body, the wall or the cupboard."""
    R = _rz(yaw)
    o = np.array([px, py, 0.0])
    b = S.plan["bays"][S.plan["target"]]
    wall_dir = {"E": np.array([1.0, 0, 0]), "W": np.array([-1.0, 0, 0]), "N": np.array([0, 1.0, 0]),
                "S": np.array([0, -1.0, 0])}[sv["far"]]
    side = 1.0 if float((R @ np.array([0, 1.0, 0])) @ wall_dir) > 0 else -1.0
    from .human import figures, figure_key
    J = figures()[figure_key(sv["pose"])]["joints"]
    sc = float(S.figures[0]["scale"])
    legs = np.array([J[k] for k in ("shin_l", "shin_r", "foot_l", "foot_r")], float) * sc
    la, lb = float(legs[:, 1].min()) - 0.09, float(legs[:, 1].max()) + 0.09       # the lower legs across the body
    lx = float(legs[:2, 0].min()) - 0.02 - 0.275                                 # the plate starts just below the knees; the feet show beyond it
    gap = float(lay["legs_top"]) + 0.07                                          # the slab's underside
    # the supports: on the wall's side a low heap of rubble (the look pack's, with its bars) along the lower legs; on
    # the room's side one broken block at the plate's corner beyond the feet, so that from the room the lower legs
    # show under the plate all the way from the knees
    dw = 0.75 + side * float(lay["torso"][1]) - 0.15                             # the body's axis to the wall's face
    span = (lb - la) + 0.70                                                      # the plate across the legs
    mid = (la + lb) / 2
    if side * mid + span / 2 > dw - 0.03:                                        # the plate ends short of the wall
        mid = side * (dw - 0.03 - span / 2)
    off = 0.62 * span / 2                                                        # the supports stand well under the plate
    wd = min(0.34, 2 * (dw - 0.04 - abs(mid + side * off)))
    if wd >= 0.16:
        c = R @ np.array([lx, mid + side * off, 0.0]) + o
        S.prop(str(rng.choice(["rubble1", "rubble3"])), (c[0], c[1], 0.0), yaw=yaw + float(rng.uniform(-0.3, 0.3)),
               fit=(0.50, wd, gap + 0.012), collide=True)
    else:
        for dx in (-0.15, 0.15):
            c = R @ np.array([lx + dx, mid + side * off, 0.0]) + o
            S.prop("concrete3", (c[0], c[1], 0.0), yaw=float(rng.uniform(0, 2 * math.pi)), fit=(0.2, 0.15, gap + 0.005), collide=True)
    c = R @ np.array([lx - 0.10, mid - side * off, 0.0]) + o
    S.prop("concrete3", (c[0], c[1], 0.0), yaw=float(rng.uniform(0, 2 * math.pi)), fit=(0.24, 0.20, gap + 0.005), collide=True)
    for _ in range(3):                                       # loose bricks beside it
        q = R @ np.array([lx + float(rng.uniform(-0.45, 0.1)), mid - side * (off + float(rng.uniform(0.15, 0.4))), 0.0]) + o
        S.chunk((q[0], q[1], 0.04), float(rng.uniform(0.07, 0.12)), mat="brick")
    # the slab piece on them
    c = R @ np.array([lx, mid, 0.0]) + o
    syaw = yaw + math.pi / 2 + float(rng.uniform(-0.12, 0.12))
    # a broken plate: an uneven outline, the underside's edge broken back, in the slab's own (cracked concrete) material
    n = 11
    angs = np.sort(rng.uniform(0, 2 * math.pi, n) * 0.35 + np.arange(n) * 2 * math.pi / n * 0.65 + np.arange(n) * 2 * math.pi / n * 0.35)
    rad = rng.uniform(0.80, 1.0, n)
    top = np.stack([np.cos(angs) * rad * span / 2, np.sin(angs) * rad * 0.30, 0.12 + rng.uniform(-0.008, 0.008, n)], 1)
    bot = np.stack([top[:, 0] * rng.uniform(0.86, 0.97, n), top[:, 1] * rng.uniform(0.86, 0.97, n), np.zeros(n)], 1)
    nm = S._name("legslab")
    S.meshes[nm] = np.concatenate([top, bot])
    S.statics.append({"name": nm, "type": "mesh", "mesh": nm, "pos": [float(c[0]), float(c[1]), float(gap)],
                      "quat": _q_euler(syaw, 0.0, 0.0), "mat": "slab", "collide": True, "size": [0, 0, 0],
                      "half": [span / 2, 0.30, 0.06]})
    Rs = _rz(syaw)
    for _ in range(3):                                     # reinforcing bars out of one broken edge
        y0 = float(rng.uniform(-0.25, 0.25))
        ra = np.array([c[0], c[1], gap + 0.06]) + Rs @ np.array([span / 2 * 0.8, y0, 0.0])
        rb = ra + Rs @ np.array([float(rng.uniform(0.12, 0.26)), float(rng.uniform(-0.05, 0.05)), float(rng.uniform(-0.06, 0.02))])
        S.capsule(ra, rb, 0.006, "rebar", collide=False, name="rebar")
    for _ in range(6):
        q = c + R @ np.array([float(rng.uniform(-0.2, 0.2)), float(rng.uniform(la - 0.1, lb + 0.1)), 0.0])
        S.chunk((q[0], q[1], gap + 0.15), float(rng.uniform(0.06, 0.11)), mat=str(rng.choice(["brick", "rubble"])))
    # the cupboard against the wall, beyond the feet or beyond the head
    hh = min(0.58, float(b.get("slab_h", 0.8)) - 0.10)
    for xc in (float(lay["x0"]) - 0.20 - 0.65, float(lay["x1"]) + 0.20 + 0.65):
        c2 = R @ np.array([xc, side * (dw - 0.345 - 0.02), 0.0]) + o
        ax = R @ np.array([1.0, 0, 0])
        ex, ey = (0.65, 0.345) if abs(ax[0]) > 0.5 else (0.345, 0.65)
        if b["x0"] + 0.1 <= c2[0] - ex and c2[0] + ex <= b["x1"] - 0.1 and b["y0"] + 0.1 <= c2[1] - ey and c2[1] + ey <= b["y1"] - 0.1 \
                and S.free_at(c2[0], c2[1], 0.7):
            # its doors face the room: the room is on the side of it away from the wall
            _piece(S, "sideboard", c2[0], c2[1], yaw, (1.3, 0.69, hh), face=-side, name="cupboard")
            break


def _survivor_and_decoys(S):
    plan, rng = S.plan, S.rng
    sv = plan["survivor"]
    scale = float(rng.uniform(0.96, 1.04))
    parts, heads, hands = figure_parts(sv["pose"], scale=scale, rng=rng)
    lay = figure_layout(sv["pose"], scale)
    # the figure's local x (head direction) along the wall it lies against
    yaw = sv["yaw"]
    # shift so the body's middle is at (x, y)
    off = _rz(yaw) @ np.array([-float(lay["torso"][0]), 0.0, 0.0])
    px, py = sv["x"] + off[0], sv["y"] + off[1]
    placed = place_parts(parts, px, py, yaw, 0.0)
    S.figures = [{"who": "survivor", "key": figure_key(sv["pose"]), "x": float(px), "y": float(py), "yaw": float(yaw),
                  "scale": scale}]
    S.parts["survivor"] = []
    skin = SKIN[sv["skin"]]
    colors = {"skin": skin, "hair": HAIR[sv["skin"] % len(HAIR)], "shirt": SHIRT[sv["shirt"]],
              "trousers": TROUSERS[sv["trousers"]], "shoes": (0.12, 0.1, 0.09)}
    # the posed body's own colour maps take the site's clothes and skin as tints (people differ from site to site)
    _lum = lambda c: 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]          # noqa: E731
    _sh = SHIRT[sv["shirt"]]
    _f = min(1.0, max(0.45, _lum(skin) / max(_lum(c) for c in SKIN)))
    S.figure_tint = {"survivor": {"cloth": tuple(round(0.35 + 0.65 * float(v) / max(_sh), 4) for v in _sh),
                                  "skin": (round(_f, 4),) * 3, "hair": (1.0, 1.0, 1.0)}}
    for p in placed:
        nm = "survivor_" + p["name"]
        rgba = list(colors.get(p["what"], (0.5, 0.5, 0.5))) + [1.0]
        # clothes are dusty
        if p["what"] in ("shirt", "trousers"):
            rgba = [0.75 * c + 0.25 * d for c, d in zip(rgba[:3], (0.68, 0.65, 0.6))] + [1.0]
        g = {"name": nm, "mat": None, "rgba": rgba, "collide": True, "what": p["what"]}
        if p["type"] == "capsule":
            g.update({"type": "capsule", "fromto": [float(v) for v in list(p["from"]) + list(p["to"])], "size": [p["r"], 0, 0]})
        else:
            g.update({"type": "ellipsoid", "size": list(p["size"]), "pos": [float(v) for v in p["pos"]],
                      "quat": _q_euler(p["yaw"], p.get("pitch", 0.0), 0.0)})
        S.parts["survivor"].append(g)
        S.thermal[nm] = "skin" if p["what"] == "skin" else ("hair" if p["what"] == "hair" else "cloth")
    R = _rz(yaw)
    S.survivor_points = {"heads": [(R @ h + np.array([px, py, 0])).tolist() for h in heads],
                         "hands": [(R @ h + np.array([px, py, 0])).tolist() for h in hands],
                         "torso": (R @ lay["torso"] + np.array([px, py, 0])).tolist()}
    if have_props():
        _clear_floor(S, px, py, yaw, lay)
    if sv.get("legs_covered", True) and have_props():
        _trapped(S, sv, lay, px, py, yaw, rng)
    elif sv.get("legs_covered", True):
        if sv.get("legs_covered", True):
            # a slab piece of 0.6 x 0.76 m lying over the lower legs, clear of them by a hand's breadth at its lowest edge
            c = R @ np.array([lay["legs"][0], lay["legs"][1], 0.0]) + np.array([px, py, 0])
            zs = lay["legs_top"] + 0.06 + 0.09
            S.box((c[0], c[1], zs), (0.30, 0.38, 0.06), "slab", yaw=yaw + float(rng.uniform(-0.2, 0.2)), pitch=0.06,
                  roll=float(rng.uniform(-0.10, 0.10)), name="legslab")
            for _ in range(8):
                q = c + R @ np.array([float(rng.uniform(-0.28, 0.28)), float(rng.uniform(-0.34, 0.34)), 0.0])
                S.chunk((q[0], q[1], zs + 0.07), float(rng.uniform(0.1, 0.2)), mat=str(rng.choice(["brick", "rubble"])))
            # a sturdy piece of furniture that kept the pocket open (a wardrobe on its side), on the wall's side of them
            wall_dir = {"E": np.array([1.0, 0, 0]), "W": np.array([-1.0, 0, 0]), "N": np.array([0, 1.0, 0]),
                        "S": np.array([0, -1.0, 0])}[sv["far"]]
            side = 1.0 if float((R @ np.array([0, 1.0, 0])) @ wall_dir) > 0 else -1.0
            reach = lay["left"] if side > 0 else lay["right"]              # the body's reach on that side: the wardrobe lies beyond it
            c2 = R @ np.array([0.45, side * (reach + 0.34), 0.0]) + np.array([px, py, 0])
            S.box((c2[0], c2[1], 0.22), (0.9, 0.28, 0.22), "wood_dark", yaw=yaw + float(rng.uniform(-0.15, 0.15)), name="wardrobe")
    # decoys: never in a corridor the route or a dead end's way in uses
    for d in plan["decoys"]:
        if d["type"] in ("mannequin", "clothes"):
            b = plan["bays"][d["bay"]]
            for _try in range(40):
                if S.free_at(d["x"], d["y"], 1.0):
                    break
                d["x"] = float(rng.uniform(b["x0"] + 0.9, b["x1"] - 0.9))
                d["y"] = float(rng.uniform(b["y0"] + 0.9, b["y1"] - 0.9))
            else:
                d["type"] = "none"
    for d in plan["decoys"]:
        if d["type"] == "mannequin":
            parts, _, _ = figure_parts("prone", scale=1.0, mannequin=True)
            placed = place_parts(parts, d["x"], d["y"], d["yaw"], 0.0)
            if have_props():
                _clear_floor(S, d["x"], d["y"], d["yaw"], figure_layout("prone", 1.0, True))
            S.figures.append({"who": "mannequin", "key": figure_key("prone", True), "x": float(d["x"]), "y": float(d["y"]),
                              "yaw": float(d["yaw"]), "scale": 1.0})
            S.parts.setdefault("mannequin", [])
            for p in placed:
                nm = "mannequin_" + p["name"]
                rgba = [0.88, 0.84, 0.78, 1.0] if p["what"] == "plastic" else [0.3, 0.3, 0.32, 1.0]
                if p["what"] == "shirt":
                    rgba = list(SHIRT[int(rng.integers(len(SHIRT)))]) + [1.0]
                g = {"name": nm, "mat": None, "rgba": rgba, "collide": True, "what": p["what"]}
                if p["type"] == "capsule":
                    g.update({"type": "capsule", "fromto": [float(v) for v in list(p["from"]) + list(p["to"])], "size": [p["r"], 0, 0]})
                else:
                    g.update({"type": "ellipsoid", "size": list(p["size"]), "pos": [float(v) for v in p["pos"]],
                              "quat": _q_euler(p["yaw"], 0.0, 0.0)})
                S.parts["mannequin"].append(g)
                S.thermal[nm] = "ambient"
        elif d["type"] == "boiler":
            nm = S.box((d["x"], d["y"], 0.55), (0.22, 0.3, 0.45), "metal_white", name="boiler", thermal="boiler")
            d["geom"] = nm
        elif d["type"] == "clothes":
            for k in range(4):
                c = (d["x"] + float(rng.uniform(-0.3, 0.3)), d["y"] + float(rng.uniform(-0.3, 0.3)), 0.06)
                S.ellipsoid(c, (float(rng.uniform(0.15, 0.3)), float(rng.uniform(0.1, 0.2)), 0.05),
                            str(rng.choice(["fabric_red", "fabric_blue", "fabric_grey", "fabric_green"])),
                            yaw=float(rng.uniform(0, 3)), collide=False, name="clothes")


def _neighbours(S):
    """The buildings either side (standing, damaged facades), to close the street scene."""
    rng = S.rng
    for side in (-1, 1):
        x = -3.0 if side < 0 else S.W + 3.0
        h = float(rng.uniform(9.0, 14.0))
        S.box((x, S.D / 2 - 1.0, h / 2), (2.8, S.D / 2 + 1.0, h / 2), str(rng.choice(["plaster_cream", "plaster_white", "plaster_pink"])),
              name="nbuilding")
        for s_ in range(int(h / STOREY)):
            for k in range(3):
                yy = 0.8 + k * 2.2
                xf = x + (-side) * 2.81 if False else x
                S.box((x, -1.0 - 0.005 + 0.0, s_ * STOREY + 1.6), (0.6, 0.01, 0.7), "window", collide=False, name="nwin") if False else None
        # facade windows on the street side
        if have_props():
            from . import props
            for s_ in range(int(h / STOREY)):
                model = ["win_18", "win_21", "win_20"][int(rng.integers(3))]
                S.prop(model, (x, -2.0 - 0.01 - props.size(model)[1] / 2, s_ * STOREY), yaw=0.0,
                       fit=(5.6, props.size(model)[1], STOREY))
            continue
        for s_ in range(1, int(h / STOREY)):
            for k in range(2):
                xx = x - 1.2 + k * 2.4
                S.box((xx, -2.0 - 0.01, s_ * STOREY + 1.5), (0.55, 0.01, 0.65), "window", collide=False, name="nwin")
    # the back: a wall of the next block
    S.box((S.W / 2, S.D + 2.8, 4.0), (S.W / 2 + 6.0, 0.3, 4.0), "brick", name="backwall")
    _cordon(S)
    # far side of the street: a low fence and trees (visual)
    for k in range(int((S.W + 10) / 3.0)):
        x = -5 + k * 3.0
        S.box((x, -10.6, 0.6), (1.45, 0.05, 0.6), "metal_grey", collide=False, name="fence")


def _cordon(S):
    """The rescue cordon: rows of concrete road barriers along the far kerb and across both ends of the street, up to
    the neighbouring buildings, so the site is closed (nothing random is drawn here)."""
    seg, hh, ht = 2.0, 0.42, 0.22
    real = have_props()
    y = S.hy0 + 0.45
    x = S.hx0 + 0.2
    while x < S.hx1 - 0.2:
        L = min(seg, S.hx1 - 0.2 - x)
        if L / 2 - 0.02 > 1e-6:           # a last piece of 4 cm or less is left out (a box of no size cannot be built)
            S.box((x + L / 2, y, STREET_Z + hh), (L / 2 - 0.02, ht, hh), "concrete", name="barrier", group=3 if real else None)
            if real:
                S.prop("roadbarrier", (x + L / 2, y, STREET_Z), yaw=0.0, fit=(L - 0.04, 2 * ht, 2 * hh))
        x += L
    for xe in (S.hx0 + 0.45, S.hx1 - 0.45):
        yy = S.hy0 + 0.45 + ht
        while yy < -1.9:
            L = min(seg, -1.9 - yy)
            if L / 2 - 0.02 > 1e-6:
                S.box((xe, yy + L / 2, STREET_Z + hh), (ht, L / 2 - 0.02, hh), "concrete", name="barrier", group=3 if real else None)
                if real:
                    S.prop("roadbarrier", (xe, yy + L / 2, STREET_Z), yaw=math.pi / 2, fit=(L - 0.04, 2 * ht, 2 * hh))
            yy += L


def _way_in(S):
    """The straight way from the deployment point to the entry (two legs), which everything put on the street keeps
    clear of."""
    st = S.plan["start"]
    e = [o for o in S.plan["openings"] if o["a"] == "S" and o["on_route"]]
    ex = e[0]["x"] if e else st["x"]
    return [(st["x"], st["y"] - 1.5), (st["x"], st["y"]), (ex, -1.6), (ex, 0.6)]


def _off_way(S, x, y, r, clear=1.25):
    P = _way_in(S)
    for (x0, y0), (x1, y1) in zip(P[:-1], P[1:]):
        dx, dy = x1 - x0, y1 - y0
        L2 = dx * dx + dy * dy
        t = 0.0 if L2 < 1e-9 else max(0.0, min(1.0, ((x - x0) * dx + (y - y0) * dy) / L2))
        if math.hypot(x - (x0 + t * dx), y - (y0 + t * dy)) < r + clear:
            return False
    for v in S.vehicles:                                    # and of the street's vehicles
        if math.hypot(x - v["x"], y - v["y"]) < r + 2.7:
            return False
    return S.free_at(x, y, r)


def _on_street(S, x, y, r):
    return S.hx0 + 1.0 + r <= x <= S.hx1 - 1.0 - r and S.hy0 + 1.3 + r <= y <= -0.45 - r


def _rescue_post(S):
    """The rescue team's post on the street beside the deployment point, on the side away from the crushed car: a
    generator with its cans, two work lights on tripods, tool boxes and crates, a hand truck, a stretcher and a
    medical box on the pavement, barrier boards and cones around it. All of it stands (static), every sensor sees it
    and the robot would touch it; nothing is nearer than 1.25 m to the way from the deployment point to the entry."""
    rng = S.rng
    st = S.plan["start"]
    car = [v for v in S.vehicles if v["name"] == "car"]
    s = -1.0 if (car and car[0]["x"] > st["x"]) else 1.0
    if not car:
        s = 1.0 if st["x"] < S.W / 2 else -1.0
    e = [o for o in S.plan["openings"] if o["a"] == "S" and o["on_route"]]
    ex = e[0]["x"] if e else st["x"]
    items = [("generator", 2.7, -0.6, 112.0, 0.45), ("jerrycan", 3.45, -1.2, 23.0, 0.2), ("jerrycan", 2.1, 0.15, -70.0, 0.2),
             ("toolbox", 2.6, 1.0, 71.0, 0.25), ("crate", 3.3, 1.3, 14.0, 0.25), ("crate", 3.7, 1.05, -31.0, 0.25),
             ("worklight", 2.0, 2.0, None, 0.6), ("worklight", 4.6, 0.3, None, 0.6), ("handtruck", 4.3, -1.1, 205.0, 0.4),
             ("carton", 3.1, -1.8, 41.0, 0.3), ("board", 5.5, -0.2, 90.0, 1.05), ("board", 3.5, -2.9, 4.0, 1.05),
             ("cone", 1.6, -2.6, 0.0, 0.2), ("cone", 5.5, -2.7, 0.0, 0.2), ("cone", 5.6, 1.5, 0.0, 0.2),
             ("cone", 1.5, 2.9, 0.0, 0.2), ("stretcher", 3.4, 3.4, 9.0, 1.1), ("medbox", 4.9, 3.2, -27.0, 0.3),
             ("tyre", 6.3, 3.7, 0.0, 0.35), ("trashbag", 6.9, 3.3, 33.0, 0.3)]
    for model, dx, dy, yawd, r in items:
        x = st["x"] + s * dx + float(rng.uniform(-0.12, 0.12))
        y = st["y"] + dy + float(rng.uniform(-0.12, 0.12))
        if not (_on_street(S, x, y, r) and _off_way(S, x, y, r)):
            continue
        if yawd is None:                                     # a work light looks at the entry
            yaw = math.atan2(-0.2 - y, ex - x)
        else:
            yaw = math.radians(yawd) * s + float(rng.uniform(-0.2, 0.2))
        S.prop(model, (x, y, float(S.height_at(x, y))), yaw=yaw, collide=True,
               thermal="sunlit_metal" if model == "generator" else None)


def _ruins(S):
    """Pieces of the collapsed storeys (the look pack's ruin pieces: rubble with bars, broken walls and slabs, bar
    grids): on the pavement in front of the damaged bays and in rooms off the route. Static; every sensor sees them
    and the robot would touch their convex hulls. Kept clear of the way in, of the route's corridors, of the
    furniture and of the people."""
    rng = S.rng
    plan = S.plan
    people = [(f["x"], f["y"]) for f in S.figures]
    for d in plan.get("decoys", []):
        if "x" in d:
            people.append((d["x"], d["y"]))

    def clear_people(x, y, r):
        return all(math.hypot(x - px, y - py) > r + 1.6 for px, py in people)

    # the pavement: per front bay up to two pieces
    kinds = [("rubble1", 0.30, 0.42, 1.4, 0.0), ("rubble3", 0.22, 0.30, 1.2, 0.0), ("rubble2", 0.18, 0.24, 1.3, 0.0),
             ("broken_wall1", 0.8, 1.0, 1.0, 16.0), ("broken_wall2", 0.8, 1.0, 1.0, 84.0), ("concrete3", 0.4, 0.55, 0.8, 0.0),
             ("metal_bars", 0.8, 1.0, 1.0, 62.0), ("broken_wall7", 0.5, 0.7, 0.9, 0.0), ("concrete2", 0.5, 0.7, 1.0, 0.0)]
    for i in range(plan["nx"]):
        b = _bay(S, (i, 0))
        n = int(rng.integers(0, 3)) if b.get("dmg", 0.0) > 0.0 or b["state"] in ("void", "filled") else int(rng.integers(0, 2))
        for _k in range(n):
            model, s0, s1, r, roll = kinds[int(rng.integers(len(kinds)))]
            sc = float(rng.uniform(s0, s1))
            for _try in range(8):
                x = float(rng.uniform(b["x0"] + 0.3, b["x1"] - 0.3))
                y = float(rng.uniform(-2.4, -0.9))
                if _on_street(S, x, y, r * 0.6) and _off_way(S, x, y, r, clear=1.4):
                    S.prop(model, (x, y, float(S.height_at(x, y)) - 0.03), yaw=float(rng.uniform(0, 2 * math.pi)), scale=sc,
                           roll=math.radians(roll), collide=True)
                    break
    # rooms off the route: a fallen wall or slab piece
    route = set(tuple(k) for k in plan.get("route_bays", []))
    inside = [("broken_wall3", 0.45, 0.6, 0.9), ("broken_wall11", 0.35, 0.45, 0.9), ("broken_wall6", 0.4, 0.5, 1.0),
              ("concrete3", 0.3, 0.4, 0.6)]
    for k, b in plan["bays"].items():
        if tuple(k) in route or b["state"] in ("filled", "stair", "void") or rng.random() < 0.45:
            continue
        model, s0, s1, r = inside[int(rng.integers(len(inside)))]
        sc = float(rng.uniform(s0, s1))
        for _try in range(10):
            x = float(rng.uniform(b["x0"] + r + 0.3, max(b["x0"] + r + 0.31, b["x1"] - r - 0.3)))
            y = float(rng.uniform(b["y0"] + r + 0.3, max(b["y0"] + r + 0.31, b["y1"] - r - 0.3)))
            if S.free_at(x, y, r + 0.2) and _clear_of_items(S, k, x, y, r) and clear_people(x, y, r):
                S.prop(model, (x, y, float(S.height_at(x, y)) - 0.02), yaw=float(rng.uniform(0, 2 * math.pi)), scale=sc, collide=True)
                b.setdefault("items", []).append((x, y, r))
                break
    # a cloth caught on a first-floor window sill of the street face
    front = [b for (i, j), b in plan["bays"].items() if j == 0 and b.get("upper", 0) >= 2]
    if front:
        b = front[int(rng.integers(len(front)))]
        S.prop("cloth3", (float(rng.uniform(b["x0"] + 1.0, b["x1"] - 1.0)), -T_EXT / 2 - 0.2, STOREY + 0.9 - 2.0), yaw=0.0, scale=1.0)


DECALS = ("decal_01", "decal_03", "decal_05", "decal_07", "decal_09")


def _decals(S):
    """Plaster off the walls (the look pack's wall decals, as thin patches 3 mm in front of the wall they sit on): on
    the solid wall pieces of the ground storey, where the patch fits the piece with a margin."""
    from . import props
    rng = S.rng
    n = 0
    for st in list(S.statics):
        if n >= 36:
            break
        if st["type"] != "box" or not st["name"].startswith("wall") or not str(st.get("mat", "")).startswith("plaster"):
            continue
        hx, hy, hz = st["size"]
        thin = 0 if hx < hy else 1
        length = 2 * (hy if thin == 0 else hx)
        if min(hx, hy) > 0.2 or length < 1.3 or 2 * hz < 1.6 or rng.random() > 0.5:
            continue
        model = DECALS[int(rng.integers(len(DECALS)))]
        w, _d, h = props.size(model)
        sc = min(1.0, (length - 0.3) / w, (2 * hz - 0.5) / h)
        if sc < 0.45:
            continue
        w, h = w * sc, h * sc
        u = float(rng.uniform(-(length - w) / 2 + 0.1, (length - w) / 2 - 0.1)) if length - w > 0.25 else 0.0
        zc = float(rng.uniform(-hz + 0.35, hz - h - 0.1)) if 2 * hz - h > 0.5 else -hz + 0.3
        sgn = -1.0 if rng.random() < 0.5 else 1.0                       # which face of the piece
        if thin == 1:                                                    # the faces are at local y = -hy, +hy
            local, ql = np.array([u, sgn * (hy + 0.003), zc]), _q_euler(0.0 if sgn < 0 else math.pi)
        else:
            local, ql = np.array([sgn * (hx + 0.003), u, zc]), _q_euler(-math.pi / 2 if sgn < 0 else math.pi / 2)
        pos = np.array(st["pos"]) + _qmat(st["quat"]) @ local
        S.prop(model, pos, quat=_qmul(st["quat"], ql), scale=sc)
        n += 1


# ------------------------------------------------------------------------------------------------- the reference route
def _route(S):
    """Waypoints from the deployment point to the survivor's approach point, with the crawler configuration on each
    leg and the actions at the openings (the reference program's eyes; never on the agent's wire)."""
    plan = S.plan
    st = plan["start"]
    ops = {}
    for o in plan["openings"]:
        if o["on_route"]:
            ops[(o["a"], o["b"])] = o
            ops[(o["b"], o["a"])] = o
    rb = plan["route_bays"]
    seq = ["S"] + list(rb)
    wps = [{"x": st["x"], "y": st["y"], "cfg": "drive", "note": "start"}]
    for a, b in zip(seq[:-1], seq[1:]):
        o = ops.get((a, b))
        if o is None:
            continue
        n, t = _passage_dir(o)
        if o["a"] != a:
            n = -n
        p = np.array([o["x"], o["y"]])
        th = o.get("thickness", T_INT)
        kinds = [o.get("challenge") or "none"] + list(o.get("extra", []))
        Bk = b
        lowB = (Bk != "S" and _bay(S, Bk)["state"] == "void")
        lowA = (a != "S" and _bay(S, a)["state"] == "void")
        cfg = "crawl" if (lowA or lowB) else "drive"
        if "narrow" in kinds:
            ctr = np.array(o["geom"].get("narrow_center", p))
            p = ctr
        before = p - n * (th / 2 + 1.2)
        after = p + n * (th / 2 + 1.0)
        if "gap" in kinds:
            after = p + n * (o["geom"]["gap_d0"] + o["geom"]["gap_w"] + 1.0)
        if "step" in kinds:
            after = p + n * (o["geom"]["step_near"] + 1.0)
        wps.append({"x": float(before[0]), "y": float(before[1]), "cfg": cfg, "kinds": kinds, "opening": o["order"],
                    "n": n.tolist(), "note": "approach"})
        wps.append({"x": float(after[0]), "y": float(after[1]), "cfg": cfg, "kinds": kinds, "opening": o["order"],
                    "n": n.tolist(), "note": "through", "beam": o["geom"].get("beam"), "step_h": o["geom"].get("step_h"),
                    "gap_w": o["geom"].get("gap_w"), "gap_d0": o["geom"].get("gap_d0"), "narrow": "narrow" in kinds,
                    "step_near": o["geom"].get("step_near"), "th": th,
                    "p": p.tolist()})
        if b != rb[-1]:
            c = _bay_center(S, b)
            wps.append({"x": float(c[0]), "y": float(c[1]), "cfg": "crawl" if lowB else "drive", "note": "bay"})
    sv = plan["survivor"]
    tor = np.array(S.survivor_points["torso"][:2])
    c = _bay_center(S, rb[-1])
    d = tor - c
    dn = d / max(1e-6, np.linalg.norm(d))
    goal = tor - dn * 1.25
    if have_props():
        # stand in front of the upper body, square to the wall they lie along: clear of the slab over their legs
        far = {"E": (1.0, 0.0), "W": (-1.0, 0.0), "N": (0.0, 1.0), "S": (0.0, -1.0)}[sv["far"]]
        head = np.array(S.survivor_points["heads"][0][:2])
        goal = tor + 0.5 * (head - tor) - np.array(far) * 1.25
    wps.append({"x": float(c[0]), "y": float(c[1]), "cfg": "crawl", "note": "target_bay"})
    wps.append({"x": float(goal[0]), "y": float(goal[1]), "cfg": "crawl", "note": "survivor"})
    S.route = wps
