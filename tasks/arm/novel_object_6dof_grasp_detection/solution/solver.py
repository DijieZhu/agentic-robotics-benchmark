"""Privileged reference detector. It shows that the task can be done when the ground truth of the frame is known.

Candidates come from grasplib and are ranked by a gradient-boosted scorer (the non-privileged reference). When the
grader hands over the frame's ground truth (object poses and meshes, only for a submission that carries the oracle
token, see tests/), the reference runs graspnetAPI's own tests on them, exactly as eval_grasp does: every one of the
TOP best-ranked candidates is checked for collision with the scene and the table and for enough object points between
the fingers; per object, the candidates that pass are thinned with the evaluator's own NMS (0.03 m, 30 degrees) and up
to PER_OBJ of them, best-ranked first, get the force-closure test at the six friction coefficients. The output is
sorted by that true label. Without ground truth it returns the scorer's ranking.

The amount of work per frame is fixed (N_SEEDS, TOP, PER_OBJ), never set by a clock, so the reference's score does not
depend on the machine it runs on; the grader gives the privileged run its own per-call limit
(grader_config.json oracle_frame_timeout_s) that this work stays well inside."""
import os, sys, pickle
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import grasplib  # noqa: E402

_MODEL = pickle.load(open(os.path.join(HERE, "scorer.pkl"), "rb"))
MAX_OUT = 2000
N_SEEDS = 800                # surface seeds for grasplib
TOP = 8000                   # best-ranked candidates checked for collision and emptiness
PER_OBJ = 40                 # force-closure tests per object
CHUNK = 500
FRIC = np.array([0.2, 0.4, 0.6, 0.8, 1.0, 1.2])
_SCENE = {}


def _fallback(depth, intrinsics, depth_scale):
    pts = grasplib.backproject(depth, intrinsics, depth_scale)
    c = pts.mean(0) if len(pts) else np.array([0, 0, 0.5])
    row = np.zeros((1, 17)); row[0, 1] = 0.08; row[0, 2] = 0.02; row[0, 3] = 0.02
    row[0, 4:13] = np.eye(3)[:, [2, 0, 1]].reshape(-1); row[0, 13:16] = c; row[0, 16] = -1
    return row


def _scene(root, sid, view):
    """Meshes and grasp-quality models of one scene, kept for the next frame of the same scene only."""
    if _SCENE.get("key") == (root, sid):
        return _SCENE["val"]
    from graspnetAPI import GraspNetEval
    from graspnetAPI.utils.eval_utils import create_table_points, voxel_sample_points
    from graspnetAPI.utils.config import get_config
    from graspnetAPI.utils.dexnet.grasping.grasp_quality_config import GraspQualityConfigFactory
    ge = GraspNetEval.__new__(GraspNetEval); ge.root, ge.camera = root, "realsense"
    config = get_config()
    fc_list = np.array([1.2, 1.0, 0.8, 0.6, 0.4, 0.2])
    fcq = {}
    for v in fc_list:
        v = round(float(v), 2)
        config["metrics"]["force_closure"]["friction_coef"] = v
        fcq[v] = GraspQualityConfigFactory.create_config(config["metrics"]["force_closure"])
    table = create_table_points(1.0, 1.0, 0.05, dx=-0.5, dy=-0.5, dz=-0.05, grid_size=0.008)
    models, dexmodels, _ = ge.get_scene_models(sid, ann_id=view)
    sampled = [voxel_sample_points(m, 0.008) for m in models]
    _SCENE.clear()
    _SCENE.update(key=(root, sid), val=(ge, fc_list, fcq, table, dexmodels, sampled))
    return _SCENE["val"]


def _labels(G, s, root, sid, view):
    """True label per row of G (0 where untested or failing), and which rows were force-closure tested."""
    from graspnetAPI import GraspGroup
    from graspnetAPI.utils.eval_utils import (transform_points, compute_closest_points, collision_detection,
                                              get_grasp_score)
    ge, fc_list, fcq, table, dexmodels, sampled = _scene(root, sid, view)
    _, pose_list, camera_pose, align_mat = ge.get_model_poses(sid, view)
    table_trans = transform_points(table, np.linalg.inv(np.matmul(align_mat, camera_pose)))
    mtl = [transform_points(m, pose_list[i]) for i, m in enumerate(sampled)]
    seg = np.concatenate([np.full(len(m), i) for i, m in enumerate(mtl)])
    scene = np.concatenate(mtl + [table_trans])
    owner = seg[compute_closest_points(G[:, 13:16], np.concatenate(mtl))]
    y = np.zeros(len(G)); tested = np.zeros(len(G), bool)
    for i in range(len(mtl)):
        idx = np.nonzero(owner == i)[0]
        if not len(idx):
            continue
        idx = idx[np.argsort(-s[idx])]
        free, dexg = [], {}
        for c0 in range(0, len(idx), CHUNK):
            part = idx[c0:c0 + CHUNK]
            lists = [np.zeros((0, 17))] * len(mtl); lists[i] = G[part]
            coll, _, dg = collision_detection(lists, mtl, dexmodels, pose_list, scene, outlier=0.05,
                                              return_dexgrasps=True)
            for j, gi in enumerate(part):
                if not coll[i][j] and dg[i][j] is not None:
                    free.append(gi); dexg[gi] = dg[i][j]
        if not free:
            continue
        free = np.array(free)
        A = G[free].copy(); A[:, 0] = s[free]; A[:, 16] = np.arange(len(free))
        kept = GraspGroup(A).nms(0.03, 30.0 / 180 * np.pi).grasp_group_array
        order = free[kept[np.argsort(-kept[:, 0]), 16].astype(int)][:PER_OBJ]
        for gi in order:
            q = get_grasp_score(dexg[gi], dexmodels[i], fc_list, fcq)
            tested[gi] = True
            if q > 0:
                y[gi] = float(np.mean(q <= FRIC))
    return y, tested


def predict(rgb, depth, intrinsics, depth_scale):
    try:
        root, frame = os.environ.get("ORACLE_GT_ROOT"), os.environ.get("ORACLE_FRAME")
        cand, _ = grasplib.generate(depth, intrinsics, depth_scale, n_seeds=N_SEEDS)
        if cand is None:
            return _fallback(depth, intrinsics, depth_scale)
        s = np.nan_to_num(_MODEL.predict(cand["feat"]), nan=-1.0)
        if root and frame:
            sid, view = (int(v) for v in frame.split("_"))
            top = np.argsort(-s)[:TOP]
            G = grasplib.to_rows(cand, s, top)
            st = s[top]
            y, tested = _labels(G, st, root, sid, view)
            tie = 1e-3 * (st - st.min()) / (np.ptp(st) + 1e-9)
            G[:, 0] = np.where(tested, y, -1.0) + tie
            rows = G[np.argsort(-G[:, 0])][:MAX_OUT]
        else:
            idx = np.argsort(-s)[:MAX_OUT]
            rows = grasplib.to_rows(cand, s, idx)
        R = rows[:, 4:13].reshape(-1, 3, 3)
        ok = np.isfinite(rows).all(1) & (np.abs(np.linalg.det(R) - 1) < 1e-5)
        ok &= np.abs(np.einsum("nij,nik->njk", R, R) - np.eye(3)).max((1, 2)) < 1e-5
        ok &= (rows[:, 1] >= 0) & (rows[:, 3] > 0) & (rows[:, 3] <= 0.1)
        rows = rows[ok]
        return rows if len(rows) else _fallback(depth, intrinsics, depth_scale)
    except Exception:
        import traceback
        traceback.print_exc()
        return _fallback(depth, intrinsics, depth_scale)
