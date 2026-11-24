"""The site of a rig: the hall, the props placed in it, the robot's own body; ray casting on those triangles
(Embree) for the LiDAR; the floor-level occupancy used to place drives and props. Generation time only."""
import json, math, os
import numpy as np
import assets

HALL_T = np.array([-12.5, 14.0, 0.994])          # hall_v3's frame -> the task frame (floor z = 0, centre at the origin)
HALL_HALF = (30.0, 20.0)
SMALL = ["box_large", "box_small", "box_small_stack", "box_stack", "cardboard_1", "cardboard_2", "cardboard_3", "cart",
         "drawer_1", "drawer_2", "handtruck", "pallet_a", "pallet_b", "pallet_jack", "pallet_truck", "shipping_crate",
         "stack_box_1", "stack_box_2", "table", "wh_pallet", "wrapped_boxes"]
LARGE = ["rack_tall", "forklift", "reach_truck", "machine", "wrapping_machine", "industrial_scale", "conveyor_2m",
         "conveyor_4m", "conveyor_end", "fence_2m", "fence_4m", "fence_door", "racks_patch_nokegs"]

# near-infrared (865 nm) diffuse reflectance by material class: a table, not the colour map
NIR = [("lamp", None), ("glass", None), ("rubber", 0.05), ("tyre", 0.05), ("chain", 0.25), ("cardboard", 0.62), ("box", 0.60),
       ("wrapped", 0.45), ("crate", 0.55), ("pallet", 0.50), ("stack", 0.58), ("table", 0.48), ("concrete", 0.38),
       ("wall_divider", 0.30), ("wall", 0.70), ("door", 0.55), ("corrugated", 0.50), ("rack", 0.22), ("wire", 0.35),
       ("forklift", 0.28), ("reach", 0.30), ("pallet_jack", 0.26), ("pallet_truck", 0.32), ("drawer", 0.20), ("cart", 0.42),
       ("handtruck", 0.24), ("machine", 0.34), ("scale", 0.40), ("conveyor", 0.33), ("pole", 0.52), ("beam", 0.36),
       ("duct", 0.66), ("pipe", 0.44), ("cabletray", 0.50), ("extinguisher", 0.18), ("medkit", 0.65), ("fire", 0.30),
       ("decal", 0.38), ("frame", 0.40), ("husky_yellow", 0.55), ("husky", 0.08), ("alu", 0.60)]


def nir_of(name):
    n = name.lower()
    for k, v in NIR:
        if k in n:
            return v
    return 0.40


def rot_z(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


class Caster(object):
    """first hits of rays on a triangle soup: range, cos(incidence), material index"""

    def __init__(self, tris, mat):
        from embreex import rtcore_scene as rtcs
        from embreex.mesh_construction import TriangleMesh
        self.tris = np.ascontiguousarray(tris, np.float32)
        self.mat = np.asarray(mat, np.int32)
        self.scene = rtcs.EmbreeScene()
        self.mesh = TriangleMesh(self.scene, self.tris)
        n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
        self.n = n / (np.linalg.norm(n, axis=1, keepdims=True) + 1e-30)

    def cast(self, o, d):
        o = np.ascontiguousarray(o, np.float32); d = np.ascontiguousarray(d, np.float32)
        r = self.scene.run(o, d, output=1)
        pid = r["primID"]; hit = pid >= 0
        t = np.where(hit, r["tfar"], np.inf).astype(np.float64)
        cosi = np.zeros(len(o)); mat = np.full(len(o), -1, np.int32)
        cosi[hit] = np.abs(np.sum(self.n[pid[hit]] * d[hit].astype(np.float64), axis=1))
        mat[hit] = self.mat[pid[hit]]
        return t, cosi, mat


class Site(object):
    def __init__(self, hall_usd, props_dir, cache):
        self.hall_usd, self.props_dir, self.cache = hall_usd, props_dir, cache
        self.hall = assets.load(hall_usd, cache)
        self.names = []                       # global material names
        self.table = json.load(open(os.path.join(cache, "props_table.json")))
        self._occ = None

    def _mats(self, a):
        idx = []
        for n in a["names"]:
            if n not in self.names:
                self.names.append(n)
            idx.append(self.names.index(n))
        return np.asarray(idx, np.int32)[a["mat"]]

    def hall_tris(self):
        a = self.hall
        return (a["V"] + HALL_T)[a["F"]], self._mats(a)

    def prop_tris(self, name, pos, yaw):
        a = assets.load(os.path.join(self.props_dir, name + ".usdc"), self.cache)
        V = a["V"] @ rot_z(yaw).T + np.asarray(pos, float)
        return V[a["F"]], self._mats(a)

    def build(self, props):
        """props: [(name, x, y, yaw)] -> the caster of hall + props"""
        T, M = [], []
        t, m = self.hall_tris(); T.append(t); M.append(m)
        for name, x, y, yaw in props:
            t, m = self.prop_tris(name, (x, y, 0.0), yaw); T.append(t); M.append(m)
        self.caster = Caster(np.concatenate(T), np.concatenate(M))
        self.nir = np.array([np.nan if nir_of(n) is None else nir_of(n) for n in self.names])
        return self.caster

    # ---- floor-level occupancy of the hall alone (5 cm cells): anything between 0.04 and 1.6 m above the floor
    def hall_occupancy(self, cell=0.05):
        if self._occ is None:
            cp = os.path.join(self.cache, "hall_occ_%d.npz" % int(cell * 1000))
            if os.path.isfile(cp):
                z = np.load(cp); self._occ = (z["occ"], float(z["cell"]))
            else:
                t, m = self.hall_tris(); c = Caster(t, m)
                xs = np.arange(-HALL_HALF[0] + cell / 2, HALL_HALF[0], cell); ys = np.arange(-HALL_HALF[1] + cell / 2, HALL_HALF[1], cell)
                X, Y = np.meshgrid(xs, ys, indexing="ij")
                o = np.stack([X.ravel(), Y.ravel(), np.full(X.size, 1.6)], 1); d = np.tile([0.0, 0.0, -1.0], (len(o), 1))
                tt, _, _ = c.cast(o, d)
                occ = ((1.6 - tt) > 0.04).reshape(X.shape)
                np.savez_compressed(cp, occ=occ, cell=cell); self._occ = (occ, cell)
        return self._occ

    def clearance_map(self):
        from scipy import ndimage
        occ, cell = self.hall_occupancy()
        return ndimage.distance_transform_edt(~occ) * cell, cell

    @staticmethod
    def cell_index(x, y, cell):
        return (np.floor((np.asarray(x) + HALL_HALF[0]) / cell).astype(int), np.floor((np.asarray(y) + HALL_HALF[1]) / cell).astype(int))


def footprint(table, name, x, y, yaw, grow=0.0):
    lo, hi = np.array(table[name]["lo"][:2]) - grow, np.array(table[name]["hi"][:2]) + grow
    c = np.array([[lo[0], lo[1]], [hi[0], lo[1]], [hi[0], hi[1]], [lo[0], hi[1]]])
    R = rot_z(yaw)[:2, :2]
    return c @ R.T + [x, y]


def _sat(a, b):
    for P in (a, b):
        for i in range(4):
            e = P[(i + 1) % 4] - P[i]; ax = np.array([-e[1], e[0]])
            pa, pb = a @ ax, b @ ax
            if pa.max() < pb.min() or pb.max() < pa.min():
                return False
    return True


def _pt_in(P, pts):
    """pts (n,2) inside the convex quad P (counter-clockwise or clockwise)"""
    s = None
    for i in range(4):
        e = P[(i + 1) % 4] - P[i]; cr = e[0] * (pts[:, 1] - P[i][1]) - e[1] * (pts[:, 0] - P[i][0])
        s = (cr >= 0) if s is None else (s & (cr >= 0))
    s2 = None
    for i in range(4):
        e = P[(i + 1) % 4] - P[i]; cr = e[0] * (pts[:, 1] - P[i][1]) - e[1] * (pts[:, 0] - P[i][0])
        s2 = (cr <= 0) if s2 is None else (s2 & (cr <= 0))
    return s | s2


def place_props(rng, site, zone, tracks, n_small, n_large):
    """Rejection sampling of rotated footprints inside the zone (x0, x1, y0, y1): 0.9 m clear of every drive, 0.25 m
    clear of each other and of the hall's floor-level things. Kinds are drawn without replacement until used up."""
    clear, cell = site.clearance_map(); occ, _ = site.hall_occupancy()
    path = np.concatenate(tracks)
    placed, quads = [], []

    def try_place(name):
        for _ in range(400):
            x = rng.uniform(zone[0], zone[1]); y = rng.uniform(zone[2], zone[3]); yaw = rng.uniform(-math.pi, math.pi)
            if rng.uniform() < 0.5:                                   # half of the things stand square to the hall
                yaw = float(rng.integers(0, 4)) * math.pi / 2 + rng.normal(0.0, 0.03)
            q = footprint(site.table, name, x, y, yaw, grow=0.125)
            big = footprint(site.table, name, x, y, yaw, grow=0.9)
            if _pt_in(big, path).any():
                continue
            if any(_sat(q, o) for o in quads):
                continue
            # hall things under the footprint
            lo, hi = q.min(0), q.max(0)
            gx = np.arange(lo[0], hi[0] + cell, cell); gy = np.arange(lo[1], hi[1] + cell, cell)
            G = np.stack(np.meshgrid(gx, gy, indexing="ij"), -1).reshape(-1, 2); G = G[_pt_in(q, G)]
            ix, iy = Site.cell_index(G[:, 0], G[:, 1], cell)
            if (ix < 0).any() or (iy < 0).any() or (ix >= occ.shape[0]).any() or (iy >= occ.shape[1]).any() or occ[ix, iy].any():
                continue
            placed.append((name, float(x), float(y), float(yaw))); quads.append(q)
            return True
        return False

    for kinds, n in ((LARGE, n_large), (SMALL, n_small)):
        bag = []
        for _ in range(n):
            if not bag:
                bag = [kinds[i] for i in rng.permutation(len(kinds))]
            try_place(bag.pop())
    return placed


# ---------------------------------------------------------------------------------------------- the robot's body
BZ = 0.13228                                   # Husky base_link above the floor
WHEEL_R, TRACK, WHEELBASE = 0.1651, 0.555, 0.512


def _stl(path):
    import struct
    b = open(path, "rb").read(); n = struct.unpack("<I", b[80:84])[0]
    arr = np.frombuffer(b, dtype=np.dtype([("n", "<3f4"), ("v", "<9f4"), ("a", "<u2")]), count=n, offset=84)
    return arr["v"].reshape(-1, 3, 3).astype(np.float64)


def robot_parts(husky_dir, T_body_lidar):
    """The robot's static body in the body frame: [(look name, triangles (n,3,3))]. Husky A200 meshes with the
    510 mm sensor arch; the arch's top bar carries the LiDAR (its feet stand on risers when the LiDAR sits higher)."""
    lx, lz = T_body_lidar[0, 3], T_body_lidar[2, 3]
    arch_top = lz - 0.0445
    P = [("husky_black", _stl(os.path.join(husky_dir, "base_link.stl")) + [0, 0, BZ]),
         ("husky_yellow", _stl(os.path.join(husky_dir, "top_chassis.stl")) + [0, 0, BZ]),
         ("alu", _stl(os.path.join(husky_dir, "user_rail.stl")) + [0.272, 0, BZ + 0.245]),
         ("husky_black", _stl(os.path.join(husky_dir, "top_plate.stl")) + [0.0812, 0, BZ + 0.245]),
         ("alu_dark", _stl(os.path.join(husky_dir, "510_mm_sensor_arch.stl")) + [lx + 0.02, 0, arch_top])]
    w = _stl(os.path.join(husky_dir, "wheel.stl"))
    for sx in (1, -1):
        for sy in (1, -1):
            ww = w if sy > 0 else w * [-1.0, -1.0, 1.0]
            P.append(("rubber", ww + [sx * WHEELBASE / 2, sy * TRACK / 2, BZ + 0.03282]))
    return P


def self_returns(husky_dir, T_body_lidar, ray_dirs_lidar):
    """Range, cos(incidence) and reflectance of the LiDAR's rays on the robot's own body (inf where they miss)."""
    parts = robot_parts(husky_dir, T_body_lidar)
    tris = np.concatenate([t for _, t in parts]); mat = np.concatenate([np.full(len(t), i) for i, (_, t) in enumerate(parts)])
    Ti = np.linalg.inv(T_body_lidar)
    tl = tris @ Ti[:3, :3].T + Ti[:3, 3]
    c = Caster(tl, mat)
    t, cosi, m = c.cast(np.zeros_like(ray_dirs_lidar), ray_dirs_lidar)
    nir = np.array([nir_of(n) for n, _ in parts])
    return t, cosi, np.where(m >= 0, nir[np.maximum(m, 0)], np.nan)


def hall_lamps(site, grid=5.0):
    """The hall's luminaires as point lamps: the lamp mesh's vertices clustered on a grid, 0.35 m under the mesh."""
    a = site.hall; i = [k for k, n in enumerate(a["names"]) if n.lower().startswith("lamp_01")][0]
    V = (a["V"] + HALL_T)[np.unique(a["F"][a["mat"] == i])]
    key = np.round(V[:, :2] / grid).astype(int)
    out = []
    for k in np.unique(key, axis=0):
        m = (key == k).all(1)
        out.append([float(V[m, 0].mean()), float(V[m, 1].mean()), float(V[m, 2].min() - 0.35)])
    return out


# ---------------------------------------------------------------------------------------------- movers
VEHICLES = ["forklift", "reach_truck", "pallet_truck", "cart", "handtruck", "pallet_jack"]
PEOPLE = ["walk_M", "walk_F"]
PEOPLE_DIR = None                              # set by the driver: the folder with walk_M.npz, walk_F.npz


def _occupancy_with_props(site, props, cell):
    occ, _ = site.hall_occupancy()
    occ = occ.copy()
    for name, x, y, yaw in props:
        q = footprint(site.table, name, x, y, yaw)
        lo, hi = q.min(0), q.max(0)
        gx = np.arange(lo[0], hi[0] + cell, cell); gy = np.arange(lo[1], hi[1] + cell, cell)
        G = np.stack(np.meshgrid(gx, gy, indexing="ij"), -1).reshape(-1, 2); G = G[_pt_in(q, G)]
        ix, iy = Site.cell_index(G[:, 0], G[:, 1], cell)
        ok = (ix >= 0) & (iy >= 0) & (ix < occ.shape[0]) & (iy < occ.shape[1])
        occ[ix[ok], iy[ok]] = True
    return occ


def sample_movers(rng, site, zone, props, traj, duration):
    """Vehicles and walking people of one drive: each runs a straight leg back and forth, clear of the props and
    the hall's things by 0.4 m, never nearer than 1.2 m to the robot nor 1.0 m to another mover."""
    from scipy import ndimage
    import rigsim
    cell = 0.05
    clear = ndimage.distance_transform_edt(~_occupancy_with_props(site, props, cell)) * cell
    tt = np.arange(0.0, duration + 0.2, 0.2)
    rob = np.stack([np.interp(tt, traj.tau, traj.pos[:, 0]), np.interp(tt, traj.tau, traj.pos[:, 1])], 1)
    out, tracks = [], []
    n_v, n_p = int(rng.integers(4, 8)), 0          # vehicles only (pre-registration, round 2, amendment 2: no walking people yet)
    want = [("vehicle", VEHICLES[int(rng.integers(0, len(VEHICLES)))]) for _ in range(n_v)] + [("person", PEOPLE[int(rng.integers(0, 2))]) for _ in range(n_p)]
    for kind, name in want:
        if kind == "vehicle":
            lo, hi = np.array(site.table[name]["lo"][:2]), np.array(site.table[name]["hi"][:2])
            half_w, half_l = float(min(hi - lo)) / 2, float(max(hi - lo)) / 2
            yaw0 = -math.pi / 2 if (hi - lo)[1] > (hi - lo)[0] else 0.0
        else:
            half_w, half_l, yaw0 = 0.3, 0.3, 0.0
        for _ in range(400):
            L = float(rng.uniform(6.0, 25.0)); h = float(rng.uniform(-math.pi, math.pi))
            p0 = np.array([rng.uniform(zone[0] - 3, zone[1] + 3), rng.uniform(zone[2] - 3, zone[3] + 3)])
            d = np.array([math.cos(h), math.sin(h)])
            pts = p0 + np.arange(-half_l, L + half_l + 0.2, 0.2)[:, None] * d
            ix, iy = Site.cell_index(pts[:, 0], pts[:, 1], cell)
            if ix.min() < 20 or iy.min() < 20 or ix.max() >= clear.shape[0] - 20 or iy.max() >= clear.shape[1] - 20:
                continue
            if clear[ix, iy].min() < half_w + 0.4:
                continue
            m = {"kind": kind, "name": name, "p0": [float(p0[0]), float(p0[1])], "heading": h, "length": L,
                 "speed": float(rng.uniform(0.5, 2.0)) if kind == "vehicle" else rigsim.WALK_SPEED,
                 "s0": float(rng.uniform(0.0, 2 * L)), "phase": float(rng.uniform(0.0, 20.0)), "yaw0": yaw0, "half": max(half_w, half_l)}
            x, y, _, _ = rigsim.mover_state(m, tt)
            tr = np.stack([x, y], 1)
            if np.linalg.norm(tr - rob, axis=1).min() < 1.2 + m["half"]:
                continue
            if any(np.linalg.norm(tr - o, axis=1).min() < 1.0 + m["half"] + oh for o, oh in tracks):
                continue
            out.append(m); tracks.append((tr, m["half"]))
            break
    return out


def _mover_caster(self, m, pose):
    key = (m["name"], pose if m["kind"] == "person" else 0)
    if not hasattr(self, "_mc"):
        self._mc = {}
    if key not in self._mc:
        if m["kind"] == "person":
            z = np.load(os.path.join(PEOPLE_DIR, m["name"] + ".npz"))
            tris = z["V"][pose].astype(np.float64)[z["F"]]; nir = 0.45
        else:
            a = assets.load(os.path.join(self.props_dir, m["name"] + ".usdc"), self.cache)
            tris = a["V"][a["F"]]; nir = nir_of(m["name"])
        self._mc[key] = (Caster(tris, np.zeros(len(tris), np.int32)), nir)
    return self._mc[key]


Site.mover_caster = _mover_caster
