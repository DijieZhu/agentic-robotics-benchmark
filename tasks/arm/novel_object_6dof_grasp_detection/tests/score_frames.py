#!/usr/bin/env python3
"""Score the detector's grasps on the hidden frames with graspnetAPI 1.2.11's own evaluation.

Runs as root on /opt/grasp/bin/python, after every process of the agent's account has been
stopped, and never imports submitted code: it only reads the arrays the solver processes left
behind, as plain .npy files (no pickles), after checking that each one is a regular file of
bounded size.

    score_frames.py SEALED PRED OUT_JSON WORKERS

Each frame is scored exactly as GraspNetEval.eval_scene scores it: widths clipped to 0.10 m,
then eval_grasp (NMS at 0.03 m / 30 degrees, each grasp assigned to the nearest object, the ten
highest-scoring grasps per object, only those at least the 50th highest overall), collision and
force-closure tests at friction 0.2 to 1.2, and AP = mean over friction and k = 1..50 of the
precision of the top k. A frame without a valid array scores 0.
"""
import json
import os
import stat
import sys
import warnings

import numpy as np

FRIC = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2]
TOP_K, MAX_WIDTH, MAX_HEIGHT, MAX_DEPTH, MAX_SUBMIT = 50, 0.1, 0.1, 0.1, 5000
MAX_BYTES = 1024 + MAX_SUBMIT * 17 * 8


def load_prediction(path):
    st = os.lstat(path)
    if not stat.S_ISREG(st.st_mode):
        raise ValueError("not a regular file")
    if st.st_size > MAX_BYTES:
        raise ValueError("%d bytes, more than %d grasps can need" % (st.st_size, MAX_SUBMIT))
    arr = np.load(path, allow_pickle=False)
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 17 or arr.shape[0] == 0:
        raise ValueError("shape %s" % (arr.shape,))
    if len(arr) > MAX_SUBMIT:
        raise ValueError("%d grasps exceeds the %d cap" % (len(arr), MAX_SUBMIT))
    if not np.all(np.isfinite(arr)):
        raise ValueError("non-finite values")
    if np.any(arr[:, 1] < 0):
        raise ValueError("negative width")            # the upper end is clipped, as graspnetAPI does
    if np.any(arr[:, 2] <= 0) or np.any(arr[:, 2] > MAX_HEIGHT):
        raise ValueError("height out of range")
    if np.any(arr[:, 3] <= 0) or np.any(arr[:, 3] > MAX_DEPTH):
        raise ValueError("depth out of range")
    R = arr[:, 4:13].reshape(-1, 3, 3)
    if np.abs(np.einsum("nij,nik->njk", R, R) - np.eye(3)).max() > 1e-4:
        raise ValueError("rotation not orthonormal")
    if np.any(np.abs(np.linalg.det(R) - 1.0) > 1e-4):
        raise ValueError("rotation determinant is not +1")
    return arr


def score_scene(item):
    sealed, pred, sid, views = item
    warnings.filterwarnings("ignore")
    from graspnetAPI import GraspGroup, GraspNetEval
    from graspnetAPI.utils.config import get_config
    from graspnetAPI.utils.eval_utils import (create_table_points, eval_grasp, transform_points,
                                              voxel_sample_points)

    ge = GraspNetEval.__new__(GraspNetEval)
    ge.root, ge.camera = sealed, "realsense"
    config = get_config()
    table = create_table_points(1.0, 1.0, 0.05, dx=-0.5, dy=-0.5, dz=-0.05, grid_size=0.008)
    # the object set is the same in every view of a scene; view 0 itself is not shipped
    models, dexmodels, _ = ge.get_scene_models(sid, ann_id=views[0])
    sampled = [voxel_sample_points(m, 0.008) for m in models]
    out = []
    for view in views:
        name = "%04d_%04d" % (sid, view)
        ap, status = 0.0, "ok"
        path = os.path.join(pred, name + ".npy")
        try:
            if not os.path.lexists(path):
                raise FileNotFoundError(name)
            arr = load_prediction(path)
            gg = GraspGroup(arr)
            a = gg.grasp_group_array
            a[:, 1] = np.clip(a[:, 1], 0, MAX_WIDTH)
            gg.grasp_group_array = a
            _, pose_list, camera_pose, align_mat = ge.get_model_poses(sid, view)
            table_trans = transform_points(table, np.linalg.inv(np.matmul(align_mat, camera_pose)))
            gl, sl, _ = eval_grasp(gg, sampled, dexmodels, pose_list, config,
                                   table=table_trans, voxel_size=0.008, TOP_K=TOP_K)
            gl = [x for x in gl if len(x)]
            sl = [x for x in sl if len(x)]
            if gl:
                gl, sl = np.concatenate(gl), np.concatenate(sl)
                sl = sl[np.argsort(-gl[:, 0])]
                acc = np.zeros((TOP_K, len(FRIC)))
                for fi, f in enumerate(FRIC):
                    for k in range(TOP_K):
                        s = sl[:k + 1]
                        acc[k, fi] = np.sum((s <= f) & (s > 0)) / (k + 1)
                ap = 100 * float(acc.mean())
        except FileNotFoundError:
            status = "no prediction"
        except Exception as exc:                          # an invalid array forfeits this frame
            status = "invalid: %s" % str(exc)[:80]
        out.append({"frame": name, "ap": ap, "status": status})
    return out


def main():
    sealed, pred, out_json, workers = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
    manifest = json.load(open(os.path.join(sealed, "manifest.json")))
    items = [(sealed, pred, r["scene"], list(r["views"])) for r in manifest["roles"]["hidden"]]
    from multiprocessing import Pool
    with Pool(workers, maxtasksperchild=1) as pool:
        per_scene = pool.map(score_scene, items, chunksize=1)
    rows = [row for rows in per_scene for row in rows]
    mean_ap = float(np.mean([r["ap"] for r in rows])) if rows else 0.0
    with open(out_json, "w") as handle:
        json.dump({"mean_ap": mean_ap, "frames": len(rows), "per_frame": rows}, handle)
    print(json.dumps({"mean_ap": mean_ap, "frames": len(rows)}), flush=True)


if __name__ == "__main__":
    main()
