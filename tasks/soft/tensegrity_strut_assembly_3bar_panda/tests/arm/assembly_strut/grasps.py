"""Parallel-pinch grasp candidates of the Panda's stock pads on a module of the strut kit (privileged reference
program; after the quadruped cell's `grasps.py`).

`ConvexSet` is a build's convex collision pieces in the module frame (the pieces MuJoCo collides: what the pads touch
in the simulation) as half-space sets; a ray's first entry into their union gives the contact point and the facet
normal there, exactly.

A pad is modelled as a 5 x 5 grid of rays over its face (the Panda's pads: +-8.5 mm across, -9 .. +8 mm along the
approach around the tcp), shot along the closing axis from outside: the pad stops at the FIRST material any of its rays
meets (the extreme hit), as the box pad does on the convex pieces. Single ray pairs are not enough here: the CoACD
pieces of the round modules leave thin gaps between them (a ray along the cap's equator at module z 145 mm passes
between two pieces), which a box pad bridges.

`grasp_candidates` -> [(G, width, desc)], G = the tcp pose in the module frame (z = approach, from the hand into the
module; y = closing axis; x = y x z, across the pads; the tcp at the pads' middle). The approach directions follow the
module's shape, not its bounding box (a box axis of the module frame is not "from above" at the design pose):
  rod modules (motor module, end cap: bodies of revolution about the bore, module z): the approach turned about the
      bore in APPROACH_STEP_DEG steps, the closing axis across the bore (z x approach), the tcp on the bore axis or
      DEPTH_OFFSETS_M off it along the approach, at every AXIAL_STEP_M along the module's length; a closing along the
      bore would put a pad over a bore mouth, which the rod goes through: not generated;
  lid (box-like): the approach against the seat direction (from the outer face), the closing axis across the lid
      (turned 0 / +-15 deg about the approach), the tcp at LID_DEPTHS_M from the outer face (negative: above it, the
      pads overlapping the top few mm) and every 5 mm along the lid.
Two antipodal pads: on each side at least 3 rays within 1 mm of the extreme hit (the contact patch), spanning >= 3.4 mm
across the pad, their facet normals within ANTIPODAL_DEG of the closing axis (ANTIPODAL_CURVED_DEG on the rod modules'
round faces); closing width WIDTH_MIN_M-WIDTH_MAX_M (the Panda opens to 80 mm; the approach needs the grasp's width
+ 16 mm); the tcp centred between the two contact planes. Near-duplicates are dropped (3 mm, 0.2 rad); every
candidate also with the hand turned 180 deg about the approach (the two fingers swapped: the same contacts, joint 7
half a turn away); the `cap` (150) best by width (15-60 mm preferred) and nearness to the centre of
mass, the 10 grasps nearest each end of a rod module kept in the list.
"""
import math

import numpy as np

ANTIPODAL_DEG = 15.0           # a contact facet within this of the closing axis (else the module turns in the pads) ...
ANTIPODAL_CURVED_DEG = 30.0    # ... on the rod modules' round faces (the precedents' allowance for rounded faces: the pad
                               # rays 5 mm off the contact line already meet the r 17.5-20 mm cylinders and the dome at 14-17 deg)
WIDTH_MIN_M, WIDTH_MAX_M = 0.006, 0.064
APPROACH_STEP_DEG = 10.0
AXIAL_STEP_M = 0.002
DEPTH_OFFSETS_M = (0.0, -0.003, 0.003)
LID_DEPTHS_M = (-0.0016, -0.003, 0.0, 0.002, 0.004)
LID_TURNS_DEG = (0.0, 15.0, -15.0)
PAD_X_M = (-0.0075, -0.00375, 0.0, 0.00375, 0.0075)       # across the pad (it spans +-8.5 mm)
PAD_A_M = (-0.0075, -0.0035, 0.0, 0.0035, 0.007)          # along the approach (it spans -9 .. +8 mm)
PATCH_M = 0.001
PAD_HALF_X_M = 0.0085


def _rotation_angle(R):
    return math.acos(max(-1.0, min(1.0, (np.trace(R) - 1.0) / 2.0)))


class ConvexSet(object):
    """the union of convex pieces (vertex arrays, one frame) as half-space sets: n . x <= d inside"""

    def __init__(self, pieces):
        from scipy.spatial import ConvexHull
        N, D, starts = [], [], []
        lo, hi, vs = [], [], []
        for V in pieces:
            V = np.asarray(V, float)
            try:
                eq = ConvexHull(V).equations             # n . x + c <= 0 inside (no QbB: its scaling would apply here)
            except Exception:  # noqa: BLE001 - a flat piece: no volume, no contact
                continue
            starts.append(sum(len(n) for n in N))
            N.append(eq[:, :3])
            D.append(-eq[:, 3])
            lo.append(V.min(0))
            hi.append(V.max(0))
            vs.append(V)
        self.N, self.D, self.starts = np.concatenate(N), np.concatenate(D), np.array(starts)
        self.n_pieces = len(starts)
        self.lo, self.hi = np.min(lo, axis=0), np.max(hi, axis=0)
        self.V = np.concatenate(vs)

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


def pinch(cs, g, a, c, reach, antipodal_deg=ANTIPODAL_DEG):
    """(width, contact centre on the tcp's line across, patch sizes) of a pinch with the tcp at g, approach a, closing
    c (unit vectors, module frame), or None (module docstring)"""
    x = np.cross(c, a)
    sides = ([], [])
    for ox in PAD_X_M:
        for oa in PAD_A_M:
            gg = g + x * ox + a * oa
            h1 = cs.ray(gg - c * reach, c)
            if h1 is not None:
                sides[0].append((float(h1[1] @ c), ox, h1[2]))
            h2 = cs.ray(gg + c * reach, -c)
            if h2 is not None:
                sides[1].append((float(h2[1] @ c), ox, h2[2]))
    if len(sides[0]) < 3 or len(sides[1]) < 3:
        return None
    L = min(s[0] for s in sides[0])
    R = max(s[0] for s in sides[1])
    w = R - L
    if w > WIDTH_MAX_M or w < WIDTH_MIN_M:
        return None
    lim = math.cos(math.radians(antipodal_deg))
    sizes = []
    for k, (ext, facing) in enumerate(((L, -c), (R, c))):
        patch = [s for s in sides[k] if abs(s[0] - ext) <= PATCH_M]
        if len(patch) < 3 or max(s[1] for s in patch) - min(s[1] for s in patch) < 0.0034:
            return None
        if min(float(s[2] @ facing) for s in patch) < lim:
            return None
        sizes.append(len(patch))
    centre = g + c * (0.5 * (L + R) - float(g @ c))
    return w, centre, sizes


def _frame(a, c, centre):
    z, y = np.asarray(a, float), np.asarray(c, float)
    G = np.eye(4)
    G[:3, :3] = np.column_stack([np.cross(y, z), y, z])
    G[:3, 3] = centre
    return G


def _yaw180(G):
    """the same grasp with the hand turned half a turn about its approach (the fingers swapped)"""
    H = G.copy()
    H[:3, 0], H[:3, 1] = -G[:3, 0], -G[:3, 1]
    return H


def _dup(G, out):
    return any(np.linalg.norm(G[:3, 3] - o[0][:3, 3]) < 0.003 and _rotation_angle(G[:3, :3].T @ o[0][:3, :3]) < 0.2 for o in out)


def rod_module_candidates(cs, com, cap=150, extra_angles_deg=()):
    """the rod modules (module docstring); `extra_angles_deg` = approach angles about the bore tried before the grid
    (the caller's exact "straight down" angles: at the motor module's design pose, over an end cap in its nest)"""
    lo, hi = cs.lo, cs.hi
    z0, z1 = float(lo[2]), float(hi[2])
    reach = float(np.linalg.norm(hi - lo)) + 0.05
    out = []
    zs = np.arange(z0 + PAD_HALF_X_M, z1 - PAD_HALF_X_M + 1e-9, AXIAL_STEP_M)
    angles = [float(v) % 360.0 for v in extra_angles_deg]
    angles += [k * APPROACH_STEP_DEG for k in range(int(round(360.0 / APPROACH_STEP_DEG)))
               if all(abs(((k * APPROACH_STEP_DEG - v) + 180.0) % 360.0 - 180.0) > 1.0 for v in angles)]
    for ang in angles:
        th = math.radians(ang)
        a = np.array([math.cos(th), math.sin(th), 0.0])
        c = np.cross([0.0, 0.0, 1.0], a)
        for z in zs:
            for dep in DEPTH_OFFSETS_M:
                g = np.array([0.0, 0.0, float(z)]) + a * dep
                r = pinch(cs, g, a, c, reach, ANTIPODAL_CURVED_DEG)
                if r is None:
                    continue
                w, centre, _sz = r
                G = _frame(a, c, centre)
                if _dup(G, out):
                    continue
                out.append((G, w, "approach %.1f deg about the bore, at z %.1f mm, %+.0f mm off the axis" % (
                    math.degrees(th), z * 1e3, dep * 1e3), float(z)))
    return _rank(out, com, cap, ends=(z0, z1))


def lid_candidates(cs, com, seat_u, cap=150):
    """the lid (module docstring); seat_u = the seat direction (unit, module frame: the lid's outer face side)"""
    u = np.asarray(seat_u, float) / np.linalg.norm(seat_u)
    zax = np.array([0.0, 0.0, 1.0])
    w_ax = np.cross(zax, u)
    lo, hi = cs.lo, cs.hi
    reach = float(np.linalg.norm(hi - lo)) + 0.05
    a = -u
    # the lid's extents from its pieces' vertices: across (w), along the seat direction (u, the outer face = the top)
    pw, pu = cs.V @ w_ax, cs.V @ u
    mid_w = 0.5 * (float(pw.min()) + float(pw.max()))
    top = float(pu.max())
    out = []
    zs = np.arange(lo[2] + PAD_HALF_X_M, hi[2] - PAD_HALF_X_M + 1e-9, 0.005)
    for z in zs:
        for dep in LID_DEPTHS_M:
            for turn in LID_TURNS_DEG:
                t = math.radians(turn)
                c = math.cos(t) * w_ax + math.sin(t) * zax
                g = u * (top - dep) + zax * float(z) + w_ax * mid_w
                r = pinch(cs, g, a, c, reach, ANTIPODAL_DEG)
                if r is None:
                    continue
                w, centre, _sz = r
                G = _frame(a, c, centre)
                if _dup(G, out):
                    continue
                out.append((G, w, "lid: approach against the seat direction, tcp %.1f mm %s the outer face, at z %.0f mm, closing %+.0f deg" % (
                    abs(dep) * 1e3, "above" if dep < 0 else "below", z * 1e3, turn), float(z)))
    return _rank(out, com, cap)


def _rank(out, com, cap, ends=None):
    """both hand turns of every grasp; the cap best by width and nearness to the centre of mass, plus the grasps nearest
    each end of a rod module (shallow grasps at an end are what lets the fingers follow a module)"""
    com = np.asarray(com, float)
    allg = []
    for G, w, desc, z in out:
        allg.append((G, w, desc, z))
        allg.append((_yaw180(G), w, desc + ", hand turned 180 deg", z))

    def key(o):
        return (0.0 if 0.015 <= o[1] <= 0.060 else 0.03) + float(np.linalg.norm(o[0][:3, 3] - com))
    allg.sort(key=key)
    keep = allg[:cap]
    if ends is not None and len(allg) > cap:
        extra = []
        for e in ends:
            near = sorted(allg, key=lambda o: abs(o[3] - e))[:10]
            extra += [o for o in near if not any(o is k for k in keep) and not any(o is x for x in extra)]
        if extra:
            keep = keep[:cap - len(extra)] + extra
    return [(G, w, desc) for G, w, desc, _z in keep]


def build_convex_set(kit, build):
    """the ConvexSet of a build's collision pieces (module frame, m)"""
    return ConvexSet([np.asarray(V, float) * 1e-3 for V, _pid, _k in kit.collision_hulls(build)])


def unit_grasps(kit, units, seat_u=None, cap=150, extra_angles_deg=None):
    """{unit: [(G, width, desc)]} for the given units; `extra_angles_deg` {unit: [deg]} (rod modules)"""
    out = {}
    for u in units:
        b = kit.units[u]["build"]
        t = kit.units[u]["type"]
        cs = build_convex_set(kit, b)
        _mass, com, _iq, _I = kit.inertial(b)
        if t == "M2":
            out[u] = lid_candidates(cs, com, seat_u, cap=cap)
        else:
            out[u] = rod_module_candidates(cs, com, cap=cap, extra_angles_deg=(extra_angles_deg or {}).get(u, ()))
    return out
