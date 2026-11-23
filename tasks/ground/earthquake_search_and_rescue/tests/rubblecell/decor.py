"""The rubble site's set dressing (cell internals): the city around the cordoned street, drawn only.

Rule: everything here stands where no part of the robot can reach (beyond the cordon's barriers, beyond the
neighbouring buildings, or more than 3 m up), so drawing it without physics changes nothing the robot can touch.
It goes into both twins (look.py): the render twin draws it with its materials, the sensor twin (depth, lidar,
thermal, the visibility record) sees the same geometry, so the sensors agree with the colour image.

* the street's continuation beyond the barriers and the city's ground;
* the buildings across the street and along the street either side of the neighbours (facade textures of two storeys
  by two window bays, shop storeys, balconies, roof parapets, tanks and air-conditioning units), the skyline further
  out;
* beyond the far barrier: the far pavement, street lights, trees, parked cars, a police car; at the street's ends the
  fire service's engine and an ambulance with their beacons lit (their light: `lights()`);
* power lines over the street (more than 6.5 m up) and air-conditioning units under the building's upper windows
  (more than 3.3 m up); hazard tape along the barriers' tops.

Every random choice draws from this module's own generator, seeded from the site's seed: the cell's generators are
untouched, and the same site always gets the same city.
"""

import math

import mujoco
import numpy as np

from . import twin as _twin

__all__ = ["decor_fn", "lights", "STREET_Z"]

STREET_Z = -0.06
SIDEWALK_Z = 0.0
FAR_KERB_Y, FAR_FACADE_Y = -14.0, -17.0
FRONT_Y = -2.0                                      # the street frontage of the neighbours (their facades)


def _quat_yaw(yaw):
    return [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]


def _mat_quat(R):
    q = np.empty(4)
    t = np.trace(R)
    if t > 0:
        s_ = math.sqrt(t + 1.0) * 2
        q[:] = [0.25 * s_, (R[2, 1] - R[1, 2]) / s_, (R[0, 2] - R[2, 0]) / s_, (R[1, 0] - R[0, 1]) / s_]
    else:
        i = int(np.argmax(np.diag(R)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s_ = math.sqrt(1.0 + R[i, i] - R[j, j] - R[k, k]) * 2
        q[0] = (R[k, j] - R[j, k]) / s_
        q[1 + i] = 0.25 * s_
        q[1 + j] = (R[j, i] + R[i, j]) / s_
        q[1 + k] = (R[k, i] + R[i, k]) / s_
    return (q / np.linalg.norm(q)).tolist()


def _qmul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return [w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2]


def _wheel_quat(yaw):
    """A cylinder's axis (its local z) along the vehicle's lateral axis."""
    return _qmul(_quat_yaw(yaw), [math.cos(math.pi / 4), math.sin(math.pi / 4), 0.0, 0.0])


def _rot(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


class _Dresser(object):
    """Adds the decor: in the sensor twin as plain geoms; in the render twin batched, all of one material in one
    mesh (a CPU rasteriser pays per drawn object)."""

    def __init__(self, spec, body, bank):
        self.spec, self.body, self.bank = spec, body, bank
        self.n = 0
        self.meshes = {}
        self.prims = {}
        self.batches = {}

    def _batch(self, material, V, T, F, pos, quat):
        R = _rot(np.asarray(quat, float))
        Vw = np.asarray(V, float) @ R.T + np.asarray(pos, float)
        b = self.batches.setdefault(material, [[], [], [], 0])
        b[0].append(Vw)
        b[1].append(np.asarray(T, float))
        b[2].append(np.asarray(F, int) + b[3])
        b[3] += len(V)

    def flush(self):
        for k, (mat, (Vs, Ts, Fs, n)) in enumerate(sorted(self.batches.items())):
            me = self.spec.add_mesh()
            me.name = "dcbatch_%d" % k
            me.uservert = np.concatenate(Vs).flatten().tolist()
            me.usertexcoord = np.concatenate(Ts).flatten().tolist()
            me.userface = np.concatenate(Fs).flatten().tolist()
            me.smoothnormal = False
            try:
                me.inertia = mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL
            except Exception:  # noqa: BLE001
                pass
            g = self._geom("batch")
            g.type = mujoco.mjtGeom.mjGEOM_MESH
            g.meshname = me.name
            g.material = mat
        self.batches = {}

    def _material(self, look):
        if look[0] == "t":
            return self.bank.textured(look[1], look[2], "ext")
        return self.bank.plain(look[1], look[2], look[3], "ext", look[4], look[5])

    def _name(self, base):
        self.n += 1
        return "dc_%s_%d" % (base, self.n)

    def _geom(self, base):
        g = self.body.add_geom()
        g.name = self._name(base)
        g.contype, g.conaffinity, g.group, g.mass = 0, 0, 2, 0.0
        return g

    def _box_mesh(self, half):
        key = tuple(round(float(v), 4) for v in half)
        if key not in self.meshes:
            V, T, F = _twin.box_mesh(*key)
            me = self.spec.add_mesh()
            me.name = "dcbox_%d" % len(self.meshes)
            me.uservert = V.flatten().tolist()
            me.usertexcoord = T.flatten().tolist()
            me.userface = F.flatten().tolist()
            try:
                me.inertia = mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL
            except Exception:  # noqa: BLE001
                pass
            self.meshes[key] = me.name
        return self.meshes[key]

    def box(self, center, half, look, yaw=0.0, base="box", quat=None):
        """look: ("t", set, tint) textured, ("p", rgba, rough, metal, refl, emission) plain."""
        if self.bank is not None:
            V, Tc, F = _twin.box_mesh(*[float(x) for x in half])
            self._batch(self._material(look), V, Tc, F, center, quat if quat is not None else _quat_yaw(yaw))
            return None
        g = self._geom(base)
        g.pos = [float(x) for x in center]
        g.quat = list(quat) if quat is not None else _quat_yaw(yaw)
        if self.bank is not None and look[0] == "t":
            g.type = mujoco.mjtGeom.mjGEOM_MESH
            g.meshname = self._box_mesh(half)
            g.material = self.bank.textured(look[1], look[2], "ext")
        else:
            g.type = mujoco.mjtGeom.mjGEOM_BOX
            g.size = [float(x) for x in half]
            if self.bank is not None:
                g.material = self.bank.plain(look[1], look[2], look[3], "ext", look[4], look[5])
        return g

    def vehicle(self, vid, center, yaw, scale, paint, base="veh"):
        """A real vehicle of the look pack (vehicles.npz) standing at `center` (its footprint's middle on the ground);
        False without the pack. Render twin: batched by material; sensor twin: one mesh geom per material class."""
        import os
        from . import pbr
        if "veh" not in self.meshes:
            path = os.path.join(pbr.cache_dir(), "vehicles.npz")
            self.meshes["veh"] = np.load(path) if os.path.isfile(path) else None
        z = self.meshes["veh"]
        if z is None:
            return False
        looks = {"paint": P(tuple(0.72 * c + 0.28 * d for c, d in zip(paint, (0.62, 0.59, 0.54))), 0.45, 0.0, 0.0),
                 "glass": P((0.42, 0.47, 0.52), 0.05, 1.0, 0.0), "dark": P((0.05, 0.05, 0.05), 0.8, 0.0, 0.0),
                 "lamp": P((0.75, 0.72, 0.68), 0.3, 0.0, 0.0), "other": P((0.33, 0.33, 0.34), 0.6, 0.0, 0.0)}
        q = _quat_yaw(yaw)
        for k in sorted(n for n in z.files if n.startswith("V:%d:" % vid)):
            cls = k.split(":")[2]
            V = z[k].astype(np.float64) * float(scale)
            F = z["F" + k[1:]].astype(np.int32)
            if self.bank is not None:
                self._batch(self._material(looks[cls]), V, np.zeros((len(V), 2)), F, center, q)
                continue
            key = ("veh", vid, cls, round(float(scale), 4))
            if key not in self.meshes:
                me = self.spec.add_mesh()
                me.name = "dcveh_%d_%s_%d" % (vid, cls, len(self.meshes))
                me.uservert = V.flatten().tolist()
                me.userface = F.flatten().tolist()
                try:
                    me.inertia = mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL
                except Exception:  # noqa: BLE001
                    pass
                self.meshes[key] = me.name
            g = self._geom(base)
            g.type = mujoco.mjtGeom.mjGEOM_MESH
            g.meshname = self.meshes[key]
            g.pos = [float(x) for x in center]
            g.quat = list(q)
        return True

    def prop(self, model, center, yaw=0.0, scale=1.0, base="prop"):
        """A model of the look pack (props.py) standing at `center` (its footprint's middle, its lowest point);
        False without the pack. Render twin: batched by material; sensor twin: one mesh geom per part."""
        from . import props
        if not props.have():
            return False
        sc = np.ones(3) * np.asarray(scale, float)
        q = _quat_yaw(yaw)
        for i, (mkey, V, _N, Tc, F) in enumerate(props.parts(model)):
            if self.bank is not None:
                self._batch(self.bank.prop(mkey, "ext"), V * sc, Tc, F, center, q)
                continue
            key = ("prop", model, i, tuple(round(float(v), 4) for v in sc))
            if key not in self.meshes:
                me = self.spec.add_mesh()
                me.name = "dcprop_%d" % len(self.meshes)
                me.uservert = (V * sc).flatten().tolist()
                me.userface = F.flatten().tolist()
                try:
                    me.inertia = mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL
                except Exception:  # noqa: BLE001
                    pass
                self.meshes[key] = me.name
            g = self._geom(base)
            g.type = mujoco.mjtGeom.mjGEOM_MESH
            g.meshname = self.meshes[key]
            g.pos = [float(x) for x in center]
            g.quat = list(q)
        return True

    def prim(self, kind, center, size, look, quat=None, fromto=None, base="prim"):
        if self.bank is not None:
            kname = {"cyl": "cylinder", "cap": "capsule", "ell": "ellipsoid", "sph": "sphere"}[kind]
            sz = list(size) + [0.0] * (3 - len(size))
            if fromto is not None:
                pos, q, half = _twin._fromto_pose(fromto)
                sz = [float(size[0]), half, 0.0]
            else:
                pos, q = center, (quat if quat is not None else [1.0, 0.0, 0.0, 0.0])
            V, Tc, F = _twin.prim_mesh(kname, sz, 10)
            self._batch(self._material(look), V, Tc, F, pos, q)
            return None
        g = self._geom(base)
        g.type = {"cyl": mujoco.mjtGeom.mjGEOM_CYLINDER, "cap": mujoco.mjtGeom.mjGEOM_CAPSULE,
                  "ell": mujoco.mjtGeom.mjGEOM_ELLIPSOID, "sph": mujoco.mjtGeom.mjGEOM_SPHERE}[kind]
        if fromto is not None:
            g.fromto = [float(x) for x in fromto]
            g.size = [float(size[0]), 0.0, 0.0]
        else:
            g.pos = [float(x) for x in center]
            g.size = [float(x) for x in size] + [0.0] * (3 - len(size))
            if quat is not None:
                g.quat = list(quat)
        if self.bank is not None:
            if look[0] == "t":
                g.material = self.bank.textured(look[1], look[2], "ext")
            else:
                g.material = self.bank.plain(look[1], look[2], look[3], "ext", look[4], look[5])
            _twin.lowpoly(self.spec, g, self.prims, seg=10, prefix="dcprim")
        return g


def T(setname, tint=(1.0, 1.0, 1.0)):
    return ("t", setname, tint)


def P(rgb, rough=0.6, metal=0.0, refl=0.4, emission=0.0, alpha=1.0):
    return ("p", (rgb[0], rgb[1], rgb[2], alpha), rough, metal, refl, emission)


FACADE_TINTS = [(1.0, 1.0, 1.0), (1.0, 0.96, 0.90), (0.92, 0.94, 0.96), (1.0, 0.92, 0.86), (0.90, 0.88, 0.84)]
CAR_COLOURS = [(0.55, 0.06, 0.05), (0.08, 0.16, 0.36), (0.85, 0.85, 0.84), (0.20, 0.21, 0.22), (0.48, 0.48, 0.47),
               (0.30, 0.38, 0.22)]


def _building(D, rng, x0, x1, y_face, depth, facing, storeys, style, damaged=False):
    """A street building from x0 to x1 whose facade stands at y_face, facing -y (facing=-1) or +y (+1)."""
    w = x1 - x0
    xc = 0.5 * (x0 + x1)
    yc = y_face - facing * depth / 2.0
    shop_h = 3.2
    H = shop_h + storeys * 3.2
    tint = FACADE_TINTS[rng.integers(len(FACADE_TINTS))]
    D.box((xc, yc, shop_h / 2), (w / 2, depth / 2, shop_h / 2), T("shopfront"), base="shop")
    upper_h = H - shop_h
    if damaged:
        # one end of the upper storeys came down: the upper block is shorter there, rubble at its foot
        cut = float(rng.uniform(0.3, 0.5)) * w
        keep = w - cut
        side = 1 if rng.random() < 0.5 else -1
        xk = xc - side * cut / 2
        D.box((xk, yc, shop_h + upper_h / 2), (keep / 2, depth / 2, upper_h / 2), T("facade_" + style, tint), base="bld")
        low = max(1, storeys // 3)
        xl = xc + side * keep / 2
        D.box((xl, yc, shop_h + low * 1.6), (cut / 2, depth / 2, low * 1.6), T("facade_" + style, tint), base="bld")
        for k in range(5):
            D.box((xl + float(rng.uniform(-cut / 3, cut / 3)), y_face + facing * float(rng.uniform(0.4, 1.6)),
                   0.25 + 0.2 * k), (float(rng.uniform(0.4, 1.1)), float(rng.uniform(0.3, 0.8)), 0.25),
                  T("rubble"), yaw=float(rng.uniform(0, math.pi)), base="heap")
        top_h = shop_h + upper_h
    else:
        D.box((xc, yc, shop_h + upper_h / 2), (w / 2, depth / 2, upper_h / 2), T("facade_" + style, tint), base="bld")
        top_h = H
    # roof: slab, parapet, a tank or two, air-conditioning units
    D.box((xc, yc, top_h + 0.06), (w / 2 + 0.05, depth / 2 + 0.05, 0.06), T("roof"), base="roof")
    D.box((xc, y_face + facing * 0.05, top_h + 0.5), (w / 2 + 0.05, 0.12, 0.45), T("concrete", (0.8, 0.8, 0.78)), base="parapet")
    for _ in range(int(rng.integers(0, 3))):
        D.prim("cyl", (xc + float(rng.uniform(-w / 3, w / 3)), yc + float(rng.uniform(-depth / 4, depth / 4)), top_h + 0.9),
               (0.6, 0.75), P((0.30, 0.31, 0.33), 0.5, 0.6), base="tank")
    # balconies on some buildings
    if rng.random() < 0.45:
        for st in range(1, storeys + (0 if not damaged else -1)):
            for b in range(int(w // 3.2)):
                if rng.random() < 0.6:
                    bx = x0 + b * 3.2 + 1.6
                    z = shop_h + (st - 1) * 3.2 + 0.08
                    D.box((bx, y_face + facing * 0.6, z), (1.0, 0.6, 0.08), T("concrete", (0.85, 0.85, 0.83)), base="balc")
                    D.box((bx, y_face + facing * 1.18, z + 0.55), (1.0, 0.02, 0.47), P((0.18, 0.18, 0.19), 0.5, 0.8),
                          base="rail")
    # air-conditioning units under some windows
    for st in range(1, storeys + 1):
        for b in range(int(w // 3.2)):
            if rng.random() < 0.18:
                bx = x0 + b * 3.2 + 1.6 + float(rng.uniform(-0.4, 0.4))
                z = shop_h + (st - 1) * 3.2 + 0.55
                if damaged and z > shop_h + upper_h:
                    continue
                D.box((bx, y_face + facing * 0.22, z), (0.42, 0.2, 0.3), P((0.80, 0.80, 0.78), 0.45, 0.0), base="ac")


def _car(D, rng, x, y, yaw, colour=None, dusty=True):
    col = colour or CAR_COLOURS[rng.integers(len(CAR_COLOURS))]
    vid = (13, 10, 9, 13)[int(abs(x) * 7.0) % 4]                    # which of the pack's cars: by where it stands
    if D.vehicle(vid, (x, y, STREET_Z), yaw, 1.0, col, base="car"):
        return
    if dusty:
        col = tuple(0.75 * c + 0.25 * 0.6 for c in col)
    q = _quat_yaw(yaw)
    R = np.array([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]])
    def at(dx, dy, z):
        p = R @ np.array([dx, dy])
        return (x + p[0], y + p[1], z)
    D.box(at(0, 0, STREET_Z + 0.55), (2.1, 0.88, 0.32), P(col, 0.35, 0.0, 0.5), quat=q, base="car")
    D.box(at(-0.25, 0, STREET_Z + 1.02), (1.2, 0.80, 0.18), P(col, 0.35, 0.0, 0.5), quat=q, base="car")
    D.box(at(-0.25, 0, STREET_Z + 1.0), (1.22, 0.81, 0.13), P((0.04, 0.05, 0.06), 0.05, 0.0, 0.8), quat=q, base="glass")
    for dx in (-1.35, 1.35):
        for dy in (-0.82, 0.82):
            D.prim("cyl", at(dx, dy, STREET_Z + 0.32), (0.32, 0.11), P((0.04, 0.04, 0.04), 0.9), quat=_wheel_quat(yaw),
                   base="wheel")


def _emergency(D, rng, x, y, yaw, kind):
    """A fire engine or an ambulance; returns the beacons' positions and colours."""
    q = _quat_yaw(yaw)
    R = np.array([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]])
    def at(dx, dy, z):
        p = R @ np.array([dx, dy])
        return (x + p[0], y + p[1], z)
    beacons = []
    real = D.vehicle(5 if kind == "fire" else 7, (x + (0.9 if kind == "fire" else 0.5) * math.cos(yaw), y + (0.9 if kind == "fire" else 0.5) * math.sin(yaw), STREET_Z),
                     yaw, 1.25 if kind == "fire" else 1.2, (0.50, 0.035, 0.03) if kind == "fire" else (0.90, 0.90, 0.88), base="engine")
    if real:
        for dy, col in (((-0.7, (1.0, 0.05, 0.03)), (0.7, (0.05, 0.25, 1.0))) if kind == "fire" else ((-0.5, (0.05, 0.25, 1.0)), (0.5, (0.05, 0.25, 1.0)))):
            p = at(3.2 if kind == "fire" else 1.6, dy, STREET_Z + (2.37 * 1.25 if kind == "fire" else 1.72 * 1.2) + 0.07)
            D.box(p, (0.12, 0.22, 0.07), P(col, 0.2, 0.0, 0.5, emission=6.0), quat=q, base="beacon")
            beacons.append((p, col))
        return beacons
    if kind == "fire":
        D.box(at(0, 0, STREET_Z + 1.55), (3.6, 1.25, 1.2), P((0.62, 0.05, 0.04), 0.35, 0.0, 0.5), quat=q, base="engine")
        D.box(at(3.9, 0, STREET_Z + 1.35), (0.9, 1.2, 1.0), P((0.62, 0.05, 0.04), 0.35, 0.0, 0.5), quat=q, base="engine")
        D.box(at(4.3, 0, STREET_Z + 1.75), (0.5, 1.15, 0.45), P((0.04, 0.05, 0.06), 0.05, 0.0, 0.8), quat=q, base="glass")
        D.box(at(-0.4, 0, STREET_Z + 2.95), (3.4, 0.35, 0.12), P((0.75, 0.76, 0.78), 0.4, 0.9), quat=q, base="ladder")
        D.box(at(0, 0, STREET_Z + 0.75), (3.62, 1.27, 0.08), P((0.92, 0.92, 0.88), 0.5), quat=q, base="stripe")
        for dy, col in ((-0.7, (1.0, 0.05, 0.03)), (0.7, (0.05, 0.25, 1.0))):
            p = at(4.0, dy, STREET_Z + 2.45)
            D.box(p, (0.12, 0.22, 0.07), P(col, 0.2, 0.0, 0.5, emission=6.0), quat=q, base="beacon")
            beacons.append((p, col))
        for dx in (-2.6, 2.0, 3.9):
            for dy in (-1.15, 1.15):
                D.prim("cyl", at(dx, dy, STREET_Z + 0.48), (0.48, 0.16), P((0.04, 0.04, 0.04), 0.9),
                       quat=_wheel_quat(yaw), base="wheel")
    else:
        D.box(at(0, 0, STREET_Z + 1.45), (2.4, 1.05, 1.1), P((0.92, 0.92, 0.90), 0.35, 0.0, 0.5), quat=q, base="amb")
        D.box(at(2.85, 0, STREET_Z + 1.15), (0.6, 1.0, 0.75), P((0.92, 0.92, 0.90), 0.35, 0.0, 0.5), quat=q, base="amb")
        D.box(at(3.0, 0, STREET_Z + 1.55), (0.45, 0.95, 0.35), P((0.04, 0.05, 0.06), 0.05, 0.0, 0.8), quat=q, base="glass")
        D.box(at(0, 0, STREET_Z + 1.25), (2.42, 1.07, 0.12), P((0.75, 0.08, 0.06), 0.4), quat=q, base="stripe")
        D.box(at(0, 0, STREET_Z + 0.95), (2.42, 1.07, 0.06), P((0.95, 0.75, 0.10), 0.4), quat=q, base="stripe")
        for dy, col in ((-0.5, (0.05, 0.25, 1.0)), (0.5, (0.05, 0.25, 1.0))):
            p = at(2.2, dy, STREET_Z + 2.62)
            D.box(p, (0.12, 0.18, 0.07), P(col, 0.2, 0.0, 0.5, emission=6.0), quat=q, base="beacon")
            beacons.append((p, col))
        for dx in (-1.6, 2.4):
            for dy in (-1.0, 1.0):
                D.prim("cyl", at(dx, dy, STREET_Z + 0.4), (0.4, 0.14), P((0.04, 0.04, 0.04), 0.9),
                       quat=_wheel_quat(yaw), base="wheel")
    return beacons


def _wire(D, a, b, sag, r=0.012):
    a, b = np.asarray(a, float), np.asarray(b, float)
    pts = []
    for k in range(9):
        t = k / 8.0
        p = a + (b - a) * t
        p[2] -= sag * 4 * t * (1 - t)
        pts.append(p)
    for p, q in zip(pts[:-1], pts[1:]):
        D.prim("cap", None, (r,), P((0.05, 0.05, 0.05), 0.6), fromto=list(p) + list(q), base="wire")


def build(spec, body, bank, scene, seed):
    """Add the decor to `body` (a static body appended last); `bank` is None for the sensor twin."""
    S = scene.site
    W, Dp = S.W, S.D
    rng = np.random.default_rng([int(seed) % (2 ** 32), 0xDEC0])
    D = _Dresser(spec, body, bank)
    beacons = []

    # the city's ground, the street's continuation and the far pavement (top faces level with the site's)
    D.box((W / 2, 0.0, -0.18), (400.0, 400.0, 0.1), T("asphalt", (0.8, 0.8, 0.8)), base="ground")
    D.box((W / 2, -12.5, STREET_Z - 0.05), (W / 2 + 95.0, 1.5, 0.05), T("asphalt"), base="road")
    for side in (-1, 1):
        xc = -5.0 - 45.0 if side < 0 else W + 5.0 + 45.0
        D.box((xc, -7.0, STREET_Z - 0.05), (45.0, 4.0, 0.05), T("asphalt"), base="road")
        D.box((xc, -2.5, SIDEWALK_Z - 0.05), (45.0, 0.5, 0.05), T("paving"), base="walk")
    D.box((W / 2, FAR_KERB_Y - 0.08, 0.06), (W / 2 + 95.0, 0.08, 0.06), T("concrete"), base="kerb")
    D.box((W / 2, (FAR_KERB_Y + FAR_FACADE_Y) / 2, 0.07), (W / 2 + 95.0, (FAR_KERB_Y - FAR_FACADE_Y) / 2, 0.05),
          T("paving"), base="walk")
    for k in range(-30, int((W + 60) / 3.0)):                    # lane markings further out
        x = -5 + 1.5 + k * 3.0
        if -5.0 < x < W + 5.0:
            continue
        D.box((x, -7.0, STREET_Z + 0.004), (0.75, 0.06, 0.004), T("road_paint"), base="marking")
    # across the street and along our side beyond the neighbours: the look pack's apartment blocks
    from . import props as _props
    real = _props.have()
    blocks = ["block_04", "block_05", "block_06", "block_01", "block_02"]
    if real:
        x = -62.0
        while x < W + 60.0:
            m = blocks[int(rng.integers(len(blocks)))]
            L, dp, _h = _props.size(m)
            D.prop(m, (x + L / 2, FAR_FACADE_Y - dp / 2, 0.0), yaw=math.pi, base="block")
            x += L + float(rng.uniform(0.5, 3.0))
        for side in (-1, 1):
            x = -5.95 if side < 0 else W + 5.95
            for _k in range(2):
                m = blocks[int(rng.integers(3))]
                L, dp, _h = _props.size(m)
                xc = x - L / 2 if side < 0 else x + L / 2
                D.prop(m, (xc, FRONT_Y + dp / 2, 0.0), yaw=0.0, base="block")
                x = x - L - 1.0 if side < 0 else x + L + 1.0
    # (without the pack) across the street: a row of buildings facing north
    x = -60.0
    while not real and x < W + 60.0:
        w = 3.2 * int(rng.integers(3, 6))
        st = int(rng.integers(2, 6))
        style = ["plaster", "brick", "panel", "plaster"][rng.integers(4)]
        _building(D, rng, x, x + w, FAR_FACADE_Y, 12.0, +1, st, style, damaged=rng.random() < 0.22)
        x += w + 0.05
    # along our side, beyond the neighbours: buildings facing south
    for side in (-1, 1):
        x = -5.85 if side < 0 else W + 5.85
        while not real and (x > -60.0 if side < 0 else x < W + 60.0):
            w = 3.2 * int(rng.integers(3, 6))
            st = int(rng.integers(2, 5))
            style = ["plaster", "brick", "panel"][rng.integers(3)]
            x0, x1 = (x - w, x) if side < 0 else (x, x + w)
            _building(D, rng, x0, x1, FRONT_Y, 12.0, -1, st, style, damaged=rng.random() < 0.3)
            x = x0 - 0.05 if side < 0 else x1 + 0.05
    # the skyline further out (the haze softens it)
    for _ in range(0 if real else 70):
        ang = float(rng.uniform(0, 2 * math.pi))
        r = float(rng.uniform(70.0, 170.0))
        cx, cy = W / 2 + r * math.cos(ang), Dp / 2 + r * math.sin(ang)
        h = float(rng.uniform(12.0, 46.0))
        hw, hd = float(rng.uniform(5.0, 12.0)), float(rng.uniform(5.0, 12.0))
        style = ["plaster", "panel", "brick"][rng.integers(3)]
        D.box((cx, cy, h / 2), (hw, hd, h / 2), T("facade_" + style, FACADE_TINTS[rng.integers(5)]),
              yaw=float(rng.uniform(-0.2, 0.2)), base="sky")
    # beyond the far barrier: street lights, trees, parked cars, a police car
    for x in np.arange(-40.0, W + 40.0, 16.0):
        D.prim("cyl", (x, FAR_KERB_Y - 0.5, 3.6), (0.08, 3.6), P((0.30, 0.31, 0.32), 0.5, 0.7), base="pole")
        D.box((x, FAR_KERB_Y + 0.55, 7.1), (0.05, 1.1, 0.05), P((0.30, 0.31, 0.32), 0.5, 0.7), base="arm")
        D.box((x, FAR_KERB_Y + 1.55, 7.02), (0.16, 0.3, 0.06), P((0.25, 0.25, 0.26), 0.4, 0.5), base="lamp")
    for x in np.arange(-36.0, W + 40.0, 11.0):
        xt = x + float(rng.uniform(-1.5, 1.5))
        D.prim("cyl", (xt, FAR_KERB_Y - 1.6, 1.6), (0.14, 1.6), T("wood", (0.45, 0.38, 0.32)), base="trunk")
        for _ in range(4):
            D.prim("ell", (xt + float(rng.uniform(-1.0, 1.0)), FAR_KERB_Y - 1.6 + float(rng.uniform(-1.0, 1.0)),
                           float(rng.uniform(3.6, 5.2))),
                   (float(rng.uniform(1.0, 1.7)), float(rng.uniform(1.0, 1.6)), float(rng.uniform(0.9, 1.3))),
                   T("foliage"), base="leaves")
    x = -30.0
    while x < W + 30.0:
        if rng.random() < 0.55:
            _car(D, rng, x, FAR_KERB_Y + 1.15, float(rng.uniform(-0.04, 0.04)))
        x += float(rng.uniform(5.2, 7.5))
    beacons += _emergency(D, rng, -14.0, -7.2, 0.0, "fire")
    beacons += _emergency(D, rng, W + 13.0, -7.0, math.pi, "ambulance")
    # power lines over the street: poles on the far pavement, wires along it and across to the building's facade
    poles = list(np.arange(-30.0, W + 31.0, 26.0))
    for x in poles:
        D.prim("cyl", (x, FAR_KERB_Y - 2.4, 4.6), (0.13, 4.6), T("concrete", (0.8, 0.8, 0.78)), base="upole")
        D.box((x, FAR_KERB_Y - 2.4, 8.7), (0.06, 0.9, 0.06), T("wood", (0.4, 0.34, 0.3)), base="xarm")
    for xa, xb in zip(poles[:-1], poles[1:]):
        for dy in (-0.8, 0.0, 0.8):
            _wire(D, (xa, FAR_KERB_Y - 2.4 + dy, 8.75), (xb, FAR_KERB_Y - 2.4 + dy, 8.75), 0.7)
    for x in poles:
        if 0.0 < x < W:
            _wire(D, (x, FAR_KERB_Y - 2.4, 8.7), (x + 0.5, -0.2, 7.6), 0.5, r=0.010)
    # hazard tape along the barriers' tops
    tape = P((0.95, 0.80, 0.08), 0.5, 0.0, 0.4)
    D.box((W / 2, -10.55, 1.0), (W / 2 + 4.4, 0.002, 0.025), tape, base="tape")
    for xe in (-4.55, W + 4.55):
        D.box((xe, -6.1, 1.0), (0.002, 4.2, 0.025), tape, base="tape")
    # the building's upper windows: frames (painted wood or aluminium) and air-conditioning units under a few, all more
    # than 3 m up
    m = scene.model
    d = mujoco.MjData(m)
    mujoco.mj_kinematics(m, d)
    frame_look = [P((0.86, 0.85, 0.82), 0.5), P((0.30, 0.22, 0.15), 0.6), P((0.55, 0.56, 0.58), 0.4, 0.8)]
    for i in range(m.ngeom):
        nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i) or ""
        if not (nm.startswith("upwin") or nm.startswith("nwin")):
            continue
        p = d.geom_xpos[i]
        R = d.geom_xmat[i].reshape(3, 3)
        hs = m.geom_size[i]
        if p[2] - hs[2] < 3.05:
            continue
        k = int(np.argmin(hs))                                  # the window's thin axis: its normal
        a1, a2 = [j for j in range(3) if j != k]
        if abs(R[2, a2]) < abs(R[2, a1]):
            a1, a2 = a2, a1                                     # a2 vertical
        look_ = frame_look[int(rng.integers(len(frame_look)))]
        w, h = hs[a1], hs[a2]
        for sgn in (-1, 1):
            c = p + R[:, a2] * sgn * (h + 0.035)
            half = np.zeros(3); half[a1], half[a2], half[k] = w + 0.07, 0.035, hs[k] + 0.03
            D.box(c, half, look_, quat=_mat_quat(R), base="wframe")
            c = p + R[:, a1] * sgn * (w + 0.035)
            half = np.zeros(3); half[a1], half[a2], half[k] = 0.035, h, hs[k] + 0.03
            D.box(c, half, look_, quat=_mat_quat(R), base="wframe")
        half = np.zeros(3); half[a1], half[a2], half[k] = 0.025, h, hs[k] + 0.02
        D.box(p, half, look_, quat=_mat_quat(R), base="wframe")
    # the building's upper windows: air-conditioning units under a few (all more than 3.3 m up)
    for i in range(m.ngeom):
        nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i) or ""
        if not nm.startswith("upwin") or rng.random() > 0.22:
            continue
        p = d.geom_xpos[i]
        if p[1] > 0.5:
            continue
        z = p[2] - m.geom_size[i][2] - 0.35
        if z < 3.4:
            continue
        D.box((p[0] + float(rng.uniform(-0.3, 0.3)), p[1] - 0.32, z), (0.42, 0.2, 0.3), P((0.78, 0.78, 0.76), 0.45),
              base="ac")
    if bank is not None:
        D.flush()
    return beacons


def decor_fn(scene, seed=None):
    """A function (spec, body, bank) that dresses a twin's spec, as look.Twins takes it."""
    seed = seed if seed is not None else int(scene.site.plan.get("seed", 0))
    out = {}

    def fn(spec, body, bank):
        out["beacons"] = build(spec, body, bank, scene, seed)
    fn.out = out
    return fn


def lights(fn):
    """The emergency beacons' coloured lights for the render (after the twin was built)."""
    return [{"pos": list(p), "color": list(c), "candela": 900.0, "range": 18.0} for p, c in fn.out.get("beacons", [])]
