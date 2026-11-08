"""Classical, deterministic perception for the localisation check (round 4, spec section 5.3): NumPy and OpenCV from
/opt/venv only, nothing trained. Every function takes delivered frames, the manifest's sensor entries and the robots'
own numbers (never the true state); `check_localisation.py` scores what they return against the true state.

- geometry: a sensor's world pose from its body's pose and its mount (`sensor_pose`), pixel rays (`pixel_ray`), a ray
  on a plane (`ray_plane`), rays triangulated (`triangulate`);
- the vehicle's start pose by landmark fit on the twin map (`fit_start_pose`: x, y, heading, the camera height known);
- colour blobs (`blob`: HSV thresholds, the largest connected component), line fits (`line_angle`), thermal blobs;
- lidar points in the world with the robot's own returns masked (`lidar_points`), clusters near an expected place.
"""

import math

import numpy as np

try:
    import cv2
except ImportError:  # (the host's checks import the geometry without OpenCV)
    cv2 = None


# --------------------------------------------------------------------------------------------- geometry
def qmul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def rotm(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def quat_z(a):
    return np.array([math.cos(a / 2), 0.0, 0.0, math.sin(a / 2)])


def sensor_pose(body_p, body_q, entry):
    """A sensor's world position and rotation (camera axes as columns: +X right, +Y up, -Z forward) from its body's
    pose and the manifest's mount."""
    m = entry["pose_in_body"]
    R = rotm(body_q)
    return np.asarray(body_p, float) + R @ np.asarray(m["pos"], float), rotm(qmul(body_q, np.asarray(m["quat"], float)))


def pixel_ray(entry, u, v, R):
    """The world direction of pixel (u, v) (the manifest's pinhole: u right, v down)."""
    K = entry["intrinsics"]
    d_cam = np.array([(u - K["cx"]) / K["fx"], -(v - K["cy"]) / K["fy"], -1.0])
    d = R @ d_cam
    return d / np.linalg.norm(d)


def project(entry, R, C, X):
    """World points X (N x 3) -> pixels (N x 2) and their depth along the optical axis."""
    K = entry["intrinsics"]
    pc = (np.asarray(X, float) - C) @ R
    z = -pc[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = K["cx"] + K["fx"] * pc[:, 0] / z
        v = K["cy"] - K["fy"] * pc[:, 1] / z
    return np.stack([u, v], axis=1), z


def ray_plane(C, d, z0):
    """Where the ray C + s d meets the horizontal plane z = z0 (None if it does not, forward)."""
    if abs(d[2]) < 1e-9:
        return None
    s = (z0 - C[2]) / d[2]
    return None if s <= 0 else C + s * d


def triangulate(origins, dirs):
    """The least-squares point nearest a set of rays."""
    A, b = np.zeros((3, 3)), np.zeros(3)
    for o, d in zip(origins, dirs):
        P = np.eye(3) - np.outer(d, d)
        A += P
        b += P @ o
    return np.linalg.lstsq(A, b, rcond=None)[0]


# --------------------------------------------------------------------------------------------- images
def blob(rgb, hsv_lo, hsv_hi, min_px=30, mask=None):
    """The largest connected blob of pixels within the HSV box (OpenCV's H 0-179): (centroid u, v, area, its mask)."""
    if cv2 is None:
        raise RuntimeError("OpenCV is needed")
    hsv = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2HSV)
    m = cv2.inRange(hsv, np.array(hsv_lo, np.uint8), np.array(hsv_hi, np.uint8))
    if mask is not None:
        m = m & mask
    n, lab, stats, cents = cv2.connectedComponentsWithStats(m, 8)
    if n <= 1:
        return None
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    if stats[k, cv2.CC_STAT_AREA] < min_px:
        return None
    return float(cents[k][0]), float(cents[k][1]), int(stats[k, cv2.CC_STAT_AREA]), lab == k


def line_angle(mask):
    """The principal direction of a blob's pixels (image angle, radians, of its long axis) and its elongation."""
    v, u = np.nonzero(mask)
    if u.size < 5:
        return None, 0.0
    pts = np.stack([u - u.mean(), v - v.mean()], axis=1).astype(float)
    w, V = np.linalg.eigh(pts.T @ pts / len(pts))
    a = V[:, 1]
    return math.atan2(a[1], a[0]), float(math.sqrt(w[1] / max(w[0], 1e-9)))


def hot(thermal, k_min):
    """Pixels hotter than k_min: (centroid u, v, count, max K)."""
    m = thermal > k_min
    if not m.any():
        return None
    v, u = np.nonzero(m)
    return float(u.mean()), float(v.mean()), int(m.sum()), float(thermal.max())


# --------------------------------------------------------------------------------------------- the start pose
def fit_start_pose(obs_list, entry, mount_body_to_root, root_z, guess, max_iter=60):
    """The vehicle's (x, y, heading) from landmark sightings: obs_list = [(world point, (u, v), weight)], the camera's
    entry, the body's pose in the vehicle's root frame (identity when the camera rides on the root), the root's height
    and a first guess (x, y, heading in rad: the odometry's nominal pose). Gauss-Newton on the reprojection error."""
    x = np.array(guess, dtype=float)
    Pb, Qb = mount_body_to_root

    def residual(x):
        p = np.array([x[0], x[1], root_z])
        q = quat_z(x[2])
        bp = p + rotm(q) @ np.asarray(Pb, float)
        bq = qmul(q, np.asarray(Qb, float))
        C, R = sensor_pose(bp, bq, entry)
        X = np.array([o[0] for o in obs_list])
        uv, z = project(entry, R, C, X)
        r = (uv - np.array([o[1] for o in obs_list])) * np.array([o[2] for o in obs_list])[:, None]
        r[z <= 0] = 1e3
        return r.reshape(-1)
    for _ in range(max_iter):
        r0 = residual(x)
        J = np.zeros((r0.size, 3))
        for k, h in enumerate((1e-4, 1e-4, 1e-5)):
            dx = np.zeros(3)
            dx[k] = h
            J[:, k] = (residual(x + dx) - r0) / h
        step = np.linalg.lstsq(J.T @ J + 1e-6 * np.eye(3), -J.T @ r0, rcond=None)[0]
        x = x + step
        if np.linalg.norm(step) < 1e-7:
            break
    return x, float(np.sqrt(np.mean(residual(x) ** 2)))


# --------------------------------------------------------------------------------------------- lidar
def lidar_points(ranges, entry, C, R):
    """A range image's returns as world points (the manifest's beams)."""
    b = entry["beams"]
    rows, cols = b["rows"], b["columns"]
    e = np.radians(np.linspace(b["elevation_first_deg"], b["elevation_last_deg"], rows)) if rows > 1 else np.radians([b["elevation_first_deg"]])
    a = np.radians(np.arange(cols) * b["azimuth_step_deg"])
    ee, aa = np.meshgrid(e, a, indexing="ij")
    d = np.stack([np.cos(ee) * np.cos(aa), np.cos(ee) * np.sin(aa), np.sin(ee)], axis=-1)
    ok = ranges > 0
    pts = (d[ok] * ranges[ok][:, None]) @ R.T + C
    return pts


def cluster_near(points, expect, radius=0.6, own=None, own_radius=0.55):
    """The centroid of the returns within `radius` of an expected place, the robot's own returns (within `own_radius`
    of its own position) left out; None when there are fewer than 5."""
    p = points
    if own is not None:
        p = p[np.linalg.norm(p - own, axis=1) > own_radius]
    sel = p[np.linalg.norm(p - expect, axis=1) < radius]
    return (sel.mean(axis=0), len(sel)) if len(sel) >= 5 else (None, len(sel))


# --------------------------------------------------------------------------------------------- the twin's edges
def twin_shapes(scene, seed, sigma=0.02, hall=None):
    """The twin map: the reference scene's fixed furniture (`/World/Facility`: the tables, the crate, the steam line's
    pipe, supports, valve body and gauge, the tank, the pads' and the mat's floor markings) as boxes and cylinders in
    world axes, every landmark group moved by N(0, sigma) per axis (seeded). Returns (shapes, offsets): shape =
    dict(group, kind "box" | "cyl", M (4x4 world, column vectors), half (box half extents) | r, h, axis)."""
    from pxr import Usd, UsdGeom
    st = Usd.Stage.Open(scene)
    xc = UsdGeom.XformCache(Usd.TimeCode.Default())
    skip = ("/Items/", "/Bucket", "/FirePan", "/SteamLine/Handle", "/Gauge/needle", "/SteamLine/Leak", "/SteamLine/Plume", "/Floor")
    groups = {}
    out = []
    for p in st.Traverse():
        path = p.GetPath().pathString
        if not path.startswith("/World/Facility/") or any(s in path for s in skip):
            continue
        if not (p.IsA(UsdGeom.Cube) or p.IsA(UsdGeom.Cylinder)):
            continue
        if UsdGeom.Imageable(p).ComputeVisibility() == UsdGeom.Tokens.invisible:
            continue
        rel = path[len("/World/Facility/"):].split("/")
        if rel[0] in ("Tables", "Pads"):
            g = rel[1]
        elif rel[0] == "SteamLine" and rel[1] == "Pipe":
            g = "pipe_" + rel[2].split("_")[-1] if rel[2].startswith(("support_", "foot_")) else "pipe"
        elif rel[0] == "SteamLine":
            g = rel[1]
        else:
            g = rel[0]
        M = np.array(xc.GetLocalToWorldTransform(p), dtype=float).T  # (Gf is row-vector: transpose to column vectors)
        if p.IsA(UsdGeom.Cube):
            s = float(UsdGeom.Cube(p).GetSizeAttr().Get())
            out.append(dict(group=g, kind="box", M=M, half=np.array([s / 2] * 3), path=path))
        else:
            c = UsdGeom.Cylinder(p)
            out.append(dict(group=g, kind="cyl", M=M, r=float(c.GetRadiusAttr().Get()), h=float(c.GetHeightAttr().Get()),
                            axis=str(c.GetAxisAttr().Get()), path=path))
        groups.setdefault(g, None)
    rng = np.random.default_rng(seed)
    offsets = {g: rng.normal(0.0, sigma, 3) for g in sorted(groups)}
    for s in out:
        s["M"] = s["M"].copy()
        s["M"][:3, 3] += offsets[s["group"]]
    if hall:  # the hall: the documented asset at the world's origin (control.md), exact: its meshes' boxes
        out += hall_boxes(hall)
    return out, {g: [round(float(v), 4) for v in o] for g, o in offsets.items()}


def hall_boxes(path, near=12.0, max_extent=7.0):
    """The hall's meshes within `near` m of the bay as axis-aligned boxes (group "hall", never moved: the documents
    place the asset at the world's origin), the floor-sized ones left out."""
    from pxr import Usd, UsdGeom
    st = Usd.Stage.Open(path)
    bb = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render], useExtentsHint=True)
    out = []
    for p in st.Traverse(Usd.TraverseInstanceProxies()):  # (the plant's hall places its props as instances)
        if not p.IsA(UsdGeom.Mesh):
            continue
        r = bb.ComputeWorldBound(p).ComputeAlignedRange()
        lo, hi = np.array(r.GetMin(), float), np.array(r.GetMax(), float)
        ext = hi - lo
        if not np.all(np.isfinite(ext)) or ext.max() > max_extent or ext.max() < 0.05:
            continue
        c = (lo + hi) / 2
        if np.linalg.norm(c[:2]) > near:
            continue
        M = np.eye(4)
        M[:3, 3] = c
        out.append(dict(group="hall", kind="box", M=M, half=np.maximum(ext / 2, 0.005), path=p.GetPath().pathString, occluder=False))
    return out


def shape_edges(shapes, eye=None, step=0.01, groups=None):
    """Points along the shapes' edges (world, N x 3) for edge matching: a box's 12 edges; a cylinder's two rims and,
    seen from `eye`, its two silhouette lines."""
    pts = []
    for s in shapes:
        if groups is not None and s["group"] not in groups:
            continue
        M = s["M"]
        if s["kind"] == "box":
            h = s["half"]
            corners = np.array([[sx * h[0], sy * h[1], sz * h[2], 1.0] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
            W = (M @ corners.T).T[:, :3]
            for i in range(8):
                for j in range(i + 1, 8):
                    if bin(i ^ j).count("1") != 1:
                        continue
                    L = np.linalg.norm(W[j] - W[i])
                    n = max(2, int(L / step) + 1)
                    pts.append(W[i] + np.linspace(0, 1, n)[:, None] * (W[j] - W[i]))
        else:
            ax = {"X": 0, "Y": 1, "Z": 2}[s["axis"]]
            a = M[:3, ax] / np.linalg.norm(M[:3, ax])
            c = M[:3, 3]
            scale = np.linalg.norm(M[:3, ax])
            hh = s["h"] / 2 * scale
            r = s["r"] * np.linalg.norm(M[:3, (ax + 1) % 3])
            u = np.cross(a, [0.0, 0.0, 1.0] if abs(a[2]) < 0.9 else [1.0, 0.0, 0.0])
            u /= np.linalg.norm(u)
            w = np.cross(a, u)
            k = max(8, int(2 * math.pi * r / step))
            ang = np.linspace(0, 2 * math.pi, k, endpoint=False)
            ring = (np.cos(ang)[:, None] * u + np.sin(ang)[:, None] * w) * r
            pts += [c + a * hh + ring, c - a * hh + ring]
            if eye is not None:
                v = c - np.asarray(eye, float)
                v -= a * (v @ a)
                sdir = np.cross(a, v)
                if np.linalg.norm(sdir) > 1e-9:
                    sdir /= np.linalg.norm(sdir)
                    n = max(2, int(2 * hh / step) + 1)
                    line = np.linspace(-hh, hh, n)[:, None] * a
                    pts += [c + r * sdir + line, c - r * sdir + line]
    return np.concatenate(pts) if pts else np.zeros((0, 3))


def edge_map(rgb, lo=40, hi=120, blur=1.2):
    """A frame's distance transform to its Canny edges (px)."""
    g = cv2.cvtColor(np.ascontiguousarray(rgb[..., :3]), cv2.COLOR_RGB2GRAY)
    g = cv2.GaussianBlur(g, (0, 0), blur)
    e = cv2.Canny(g, lo, hi)
    return cv2.distanceTransform(255 - e, cv2.DIST_L2, 3).astype(np.float32), e


def chamfer(dt, entry, C, R, X, tau=8.0, min_pts=60, mask=None, groups=None):
    """The mean truncated distance (px) from the projections of model points X to the frame's edges (the points in
    the frame, off the `mask` of the robot's own body), and their number; fewer than `min_pts` score `tau`."""
    uv, z = project(entry, R, C, X)
    H, W = dt.shape
    ok = (z > 0.05) & (uv[:, 0] >= 0) & (uv[:, 0] <= W - 1) & (uv[:, 1] >= 0) & (uv[:, 1] <= H - 1)
    iu, iv = np.zeros(len(uv), int), np.zeros(len(uv), int)
    iu[ok], iv[ok] = np.round(uv[ok, 0]).astype(int), np.round(uv[ok, 1]).astype(int)
    if mask is not None:
        ok[ok] &= ~mask[iv[ok], iu[ok]]
    if ok.sum() < min_pts:
        return tau, int(ok.sum())
    d = np.minimum(dt[iv[ok], iu[ok]], tau)
    if groups is not None:  # (the mean of the groups' means: a landmark with many edges does not outweigh the others)
        g = groups[ok]
        means = [float(d[g == k].mean()) for k in np.unique(g) if (g == k).sum() >= 15]
        if means:
            return float(np.mean(means)), int(ok.sum())
    return float(np.mean(d)), int(ok.sum())


def fit_pose(views, pose_fn, x0, ranges, steps, refine_steps=None, tau=8.0, min_frac=0.5):
    """A pose by edge matching: `views` = [(dt, entry, X model points, mount[, mask])] (one per camera frame),
    `pose_fn(x, mount)` -> (C, R) of that camera; a grid search over x0 +- ranges at `steps`, then finer grids around
    the best (`refine_steps`: a list of step vectors, each searched over +- 3 steps). A pose that keeps fewer than
    `min_frac` of a view's model points in its frame (against x0) scores `tau` there. Returns (x, score)."""
    import itertools
    x0 = np.asarray(x0, float)
    ref = []
    for v in views:
        C, R = pose_fn(x0, v[3])
        ref.append(max(30, int(min_frac * chamfer(v[0], v[1], C, R, v[2], tau, min_pts=0, mask=v[4] if len(v) > 4 else None)[1])))

    def score(x):
        tot = 0.0
        for v, mp in zip(views, ref):
            C, R = pose_fn(x, v[3])
            tot += chamfer(v[0], v[1], C, R, v[2], tau, min_pts=mp, mask=v[4] if len(v) > 4 else None, groups=v[5] if len(v) > 5 else None)[0]
        return tot / max(1, len(views))
    axes = [np.arange(-r, r + 1e-9, st) for r, st in zip(ranges, steps)]
    best = (None, 1e9)
    for d in itertools.product(*axes):
        x = x0 + np.array(d)
        s = score(x)
        if s < best[1]:
            best = (x, s)
    for st in refine_steps or []:
        st = np.asarray(st, float)
        c = best[0]
        for d in itertools.product(*[np.arange(-3, 4) * v for v in st]):
            x = c + np.array(d)
            s = score(x)
            if s < best[1]:
                best = (x, s)
    return best


def vehicle_body(x, y, heading_deg, z=0.0635):
    """The vehicle's root body pose (world position, quaternion w, x, y, z) from (x, y, heading): heading 0 faces +y,
    counter-clockwise positive; the body's +X is forward (a turn of heading + 90 degrees about z)."""
    return np.array([x, y, z]), quat_z(math.radians(heading_deg + 90.0))


# --------------------------------------------------------------------------------------------- the vehicle's arm (FK)
class Chain:
    """Forward kinematics of the vehicle's articulation from the program's own scene (pxr): the joints' frames on
    their two bodies (localPos/localRot), revolute about the joint frame's axis by the encoder's position. `pose(body,
    joints)` gives a body's pose in the root body's frame (position, quaternion w, x, y, z)."""

    def __init__(self, scene, root="/World/Fleet/UGV"):
        from pxr import Usd, UsdPhysics
        st = Usd.Stage.Open(scene)
        self.parent = {}  # child body -> (parent body, joint name, kind, axis, (p0, q0), (p1, q1))
        for p in Usd.PrimRange(st.GetPrimAtPath(root), Usd.TraverseInstanceProxies()):
            if not p.IsA(UsdPhysics.Joint):
                continue
            j = UsdPhysics.Joint(p)
            b0 = [str(t) for t in j.GetBody0Rel().GetTargets()]
            b1 = [str(t) for t in j.GetBody1Rel().GetTargets()]
            if not b0 or not b1:
                continue
            kind = "revolute" if p.IsA(UsdPhysics.RevoluteJoint) else "fixed" if p.IsA(UsdPhysics.FixedJoint) else "other"
            ax = p.GetAttribute("physics:axis").Get() if p.GetAttribute("physics:axis") else None

            def pq(pos, rot):
                return np.array([float(v) for v in pos]), np.array([float(rot.GetReal())] + [float(v) for v in rot.GetImaginary()])
            self.parent[b1[0]] = (b0[0], p.GetName(), kind, str(ax) if ax else None, pq(j.GetLocalPos0Attr().Get(), j.GetLocalRot0Attr().Get()),
                                  pq(j.GetLocalPos1Attr().Get(), j.GetLocalRot1Attr().Get()))

    def pose(self, body, joints):
        """(p, q) of `body` in the root body's frame; `joints`: {name: [position, velocity]} (the wire's ugv.joints)."""
        chain = []
        b = body
        while b in self.parent:
            chain.append(self.parent[b])
            b = self.parent[b][0]
        p, q = np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0])
        for par, name, kind, ax, (p0, q0), (p1, q1) in reversed(chain):
            # body1 = body0 * T(p0, q0) * R(axis, angle) * T(p1, q1)^-1
            p = p + rotm(q) @ p0
            q = qmul(q, q0)
            if kind == "revolute":
                ang = float((joints.get(name) or [0.0])[0])
                a = {"X": np.array([1.0, 0, 0]), "Y": np.array([0, 1.0, 0]), "Z": np.array([0, 0, 1.0])}[ax]
                q = qmul(q, np.concatenate([[math.cos(ang / 2)], math.sin(ang / 2) * a]))
            qi = np.array([q1[0], -q1[1], -q1[2], -q1[3]])
            q = qmul(q, qi)
            p = p - rotm(q) @ p1
        return p, q

    def bodies_under(self, prefix):
        return sorted({b for b in self.parent if b.startswith(prefix)} | {p[0] for b, p in self.parent.items() if p[0].startswith(prefix)})

    def segments(self, prefix, joints):
        """The arm's link-to-link segments in the root frame (pairs of body origins)."""
        out = []
        for b, p in self.parent.items():
            if b.startswith(prefix) and p[0].startswith(prefix):
                out.append((self.pose(p[0], joints)[0], self.pose(b, joints)[0]))
        return out


def own_mask(entry, cam_in_root, segments, shape, radius=0.075, boxes=()):
    """The pixels a vehicle camera sees of the vehicle's own arm (capsules of `radius` along its link segments, in the
    root frame) and of `boxes` (root-frame (centre, half extents)): what a program knows of its own body, from its
    joints. `cam_in_root` = (C, R) of the camera in the root frame."""
    C, R = cam_in_root
    H, W = shape
    m = np.zeros((H, W), np.uint8)
    K = entry["intrinsics"]
    for a, b in segments:
        pts = a + np.linspace(0, 1, 24)[:, None] * (b - a)
        uv, z = project(entry, R, C, pts)
        for (u, v), zz in zip(uv, z):
            if zz > 0.02:
                r = int(min(4000, radius * K["fx"] / zz))
                if -r <= u <= W + r and -r <= v <= H + r:
                    cv2.circle(m, (int(u), int(v)), r, 255, -1)
    for c, h in boxes:
        corners = np.array([[sx * h[0], sy * h[1], sz * h[2]] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) + np.asarray(c, float)
        uv, z = project(entry, R, C, corners)
        if (z > 0.02).all():
            cv2.fillConvexPoly(m, cv2.convexHull(uv.astype(np.int32)), 255)
    return cv2.dilate(m, np.ones((9, 9), np.uint8)) > 0


def _box_local(s):
    """A shape's world-to-local matrix and its local half extents (a cylinder by its bounding box)."""
    M = s["M"]
    if s["kind"] == "box":
        return np.linalg.inv(M), np.asarray(s["half"], float)
    ax = {"X": 0, "Y": 1, "Z": 2}[s["axis"]]
    h = np.array([s["r"]] * 3, float)
    h[ax] = s["h"] / 2
    return np.linalg.inv(M), h


def visible_edges(shapes, eye, step=0.01, groups=None, eps=0.01, labels=False):
    """The shapes' edges as `eye` sees them: a box's edges with at least one adjacent face toward the eye, a
    cylinder's silhouette and front rim; then every point with something of the twin between it and the eye dropped
    (each shape's box, a cylinder's bounding box). Returns world points (N x 3)."""
    eye = np.asarray(eye, float)
    pts, owner = [], []
    for k, s in enumerate(shapes):
        if groups is not None and s["group"] not in groups:
            continue
        M = s["M"]
        if s["kind"] == "box":
            h = s["half"]
            Minv = np.linalg.inv(M)
            e_loc = (Minv @ np.append(eye, 1.0))[:3]
            front = {(a, sg): (sg * e_loc[a] > h[a]) for a in range(3) for sg in (-1, 1)}  # (a face toward the eye)
            corners = {(sx, sy, sz): (M @ np.array([sx * h[0], sy * h[1], sz * h[2], 1.0]))[:3] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)}
            for c0 in corners:
                for a in range(3):
                    if c0[a] != -1:
                        continue
                    c1 = tuple(1 if i == a else c0[i] for i in range(3))
                    faces = [(b, c0[b]) for b in range(3) if b != a]  # (the edge's two adjacent faces)
                    if not any(front[f] for f in faces):
                        continue
                    A, B = corners[c0], corners[c1]
                    n = max(2, int(np.linalg.norm(B - A) / step) + 1)
                    pts.append(A + np.linspace(0, 1, n)[:, None] * (B - A))
                    owner += [k] * n
        else:
            ax = {"X": 0, "Y": 1, "Z": 2}[s["axis"]]
            a = M[:3, ax] / np.linalg.norm(M[:3, ax])
            c = M[:3, 3]
            hh = s["h"] / 2 * np.linalg.norm(M[:3, ax])
            r = s["r"] * np.linalg.norm(M[:3, (ax + 1) % 3])
            v = c - eye
            for sgn in (1, -1):  # the cap that faces the eye
                if (eye - (c + sgn * a * hh)) @ (sgn * a) > 0:
                    u = np.cross(a, [0.0, 0.0, 1.0] if abs(a[2]) < 0.9 else [1.0, 0.0, 0.0])
                    u /= np.linalg.norm(u)
                    w = np.cross(a, u)
                    kk = max(8, int(2 * math.pi * r / step))
                    ang = np.linspace(0, 2 * math.pi, kk, endpoint=False)
                    ring = c + sgn * a * hh + (np.cos(ang)[:, None] * u + np.sin(ang)[:, None] * w) * r
                    pts.append(ring)
                    owner += [k] * kk
            vp = v - a * (v @ a)
            sdir = np.cross(a, vp)
            if np.linalg.norm(sdir) > 1e-9:
                sdir /= np.linalg.norm(sdir)
                n = max(2, int(2 * hh / step) + 1)
                line = np.linspace(-hh, hh, n)[:, None] * a
                pts += [c + r * sdir + line, c - r * sdir + line]
                owner += [k] * (2 * n)
    if not pts:
        return np.zeros((0, 3))
    X = np.concatenate(pts)
    owner = np.array(owner)
    keep = np.ones(len(X), bool)
    d = X - eye
    for k, s in enumerate(shapes):
        if s.get("occluder") is False:
            continue
        Minv, h = _box_local(s)
        o = Minv[:3, :3] @ eye + Minv[:3, 3]
        dl = d @ Minv[:3, :3].T
        with np.errstate(divide="ignore", invalid="ignore"):
            t1 = (-h - o) / dl
            t2 = (h - o) / dl
        tn = np.nanmax(np.minimum(t1, t2), axis=1)
        tf = np.nanmin(np.maximum(t1, t2), axis=1)
        L = np.linalg.norm(d, axis=1)
        hit = (tn < tf) & (tf > 0) & (tn < 1.0 - eps / np.maximum(L, 1e-6)) & (owner != k)
        keep &= ~hit
    if labels:
        return X[keep], np.array([shapes[k]["group"] for k in owner[keep]])
    return X[keep]
