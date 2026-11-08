"""Provided starting detector: a plain single-view geometric sampler.

Back-project depth, voxel-downsample, estimate normals, remove the table plane with RANSAC,
sample surface points, approach each along its inward normal at four in-plane rotations with a
fixed opening width and insertion depth, and drop any candidate whose finger or base volume
would pass through the observed point cloud. Survivors are ranked by local flatness (smallest
eigenvalue of the neighbourhood covariance).

Returns an (N, 17) float array in the GraspNet-1Billion layout:
[score, width, height, depth, R (3x3 row-major; column 0 = approach, column 1 = closing axis), t (3), object_id]
"""
import numpy as np
import open3d as o3d

WIDTH, HEIGHT, DEPTH = 0.08, 0.02, 0.02
FW, DB = 0.01, 0.02          # finger width and base depth, as the evaluator uses them
N_SEEDS, N_ROT, MAX_OUT = 256, 4, 400
VOXEL = 0.005

def backproject(depth, K, factor_depth):
    h, w = depth.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    z = depth.astype(np.float64) / factor_depth
    m = (z > 0.1) & (z < 1.5)
    x = (u - K[0, 2]) * z / K[0, 0]
    y = (v - K[1, 2]) * z / K[1, 1]
    return np.stack([x[m], y[m], z[m]], -1)

def _frames(approach, angles):
    """Rotation matrices whose first column is the approach direction."""
    a = approach / np.linalg.norm(approach)
    ref = np.array([0.0, 0.0, 1.0]) if abs(a[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    b = np.cross(a, ref); b /= np.linalg.norm(b)
    c = np.cross(a, b)
    out = []
    for th in angles:
        y = np.cos(th) * b + np.sin(th) * c
        z = np.cross(a, y)
        out.append(np.stack([a, y, z], -1))
    return out

def predict(rgb, depth, intrinsics, depth_scale):
    rng = np.random.default_rng(0)
    pts = backproject(depth, intrinsics, depth_scale)
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts)).voxel_down_sample(VOXEL)
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.015, max_nn=30))
    pcd.orient_normals_towards_camera_location(np.zeros(3))
    P = np.asarray(pcd.points)
    N = np.asarray(pcd.normals)
    o3d.utility.random.seed(0)
    plane, _ = pcd.segment_plane(0.008, 3, 300)
    nrm, off = np.asarray(plane[:3]), plane[3]
    if off < 0: nrm, off = -nrm, -off                  # camera side positive
    cand = np.nonzero((P @ nrm + off) > 0.008)[0]
    if len(cand) == 0: cand = np.arange(len(P))
    tree = o3d.geometry.KDTreeFlann(pcd)
    seeds = rng.choice(cand, min(N_SEEDS, len(cand)), replace=False)
    angles = np.arange(N_ROT) * np.pi / N_ROT
    rows = []
    for i in seeds:
        k, idx, _ = tree.search_knn_vector_3d(P[i], 20)
        nb = P[np.asarray(idx)]
        flat = np.linalg.eigvalsh(np.cov((nb - nb.mean(0)).T))[0]   # small = flat patch
        score = 1.0 / (1.0 + 1e5 * flat)
        approach = -N[i]                                               # into the surface
        _, nidx, _ = tree.search_radius_vector_3d(P[i], 0.09)
        Q = P[np.asarray(nidx)] - P[i]
        for R in _frames(approach, angles):
            t = P[i]                                                   # finger tips at the surface
            L = Q @ R
            x, y, z = L[:, 0], L[:, 1], L[:, 2]
            slab = (np.abs(z) < HEIGHT / 2) & (x > -DB) & (x < DEPTH)
            ay = np.abs(y)
            if np.any(slab & (ay > WIDTH / 2) & (ay < WIDTH / 2 + FW)): continue       # fingers hit something
            if np.any((np.abs(z) < HEIGHT / 2) & (x > -DB - FW) & (x < -DB) & (ay < WIDTH / 2 + FW)): continue   # base
            rows.append([score, WIDTH, HEIGHT, DEPTH, *R.reshape(-1), *t, -1])
    if not rows:                                                       # never return an empty array
        R = _frames(np.array([0.0, 0.0, 1.0]), [0.0])[0]
        rows.append([0.0, WIDTH, HEIGHT, DEPTH, *R.reshape(-1), *P[cand].mean(0), -1])
    arr = np.asarray(rows, dtype=np.float64)
    return arr[np.argsort(-arr[:, 0])][:MAX_OUT]
