"""Cut the downloaded GraspNet-1Billion bundle down to what the agent is meant to have.

The bundle also carries the practice objects' meshes and every object's pose. With meshes and
poses an agent can label its own candidates with the benchmark evaluator, so this keeps only
the grasp annotations of the ten `label_objects` as supervision:
  - models/ (meshes and signed-distance fields) is deleted;
  - in the practice scenes, meta/ and annotations/ (object poses) are deleted;
  - in the development scenes, annotations/ is deleted and each meta/*.mat keeps the poses of
    the annotated objects only (with factor_depth and intrinsic_matrix);
  - the manifest drops `selfcheck_objects`, and the scene roles keep only the annotated objects.

Run once at build time with /opt/grasp/bin/python (scipy is needed for the .mat files).
"""
import json, os, shutil, sys
import numpy as np
import scipy.io as sio

DATA = sys.argv[1] if len(sys.argv) > 1 else "/data/graspnet"
man = json.load(open(f"{DATA}/manifest.json"))
labelled = set(man["label_objects"])

shutil.rmtree(f"{DATA}/models")
for r in man["roles"]["selfcheck"]:
    root = f"{DATA}/scenes/scene_{r['scene']:04d}/realsense"
    shutil.rmtree(f"{root}/meta"); shutil.rmtree(f"{root}/annotations")
    r.pop("objects", None)
for r in man["roles"]["train"]:
    root = f"{DATA}/scenes/scene_{r['scene']:04d}/realsense"
    shutil.rmtree(f"{root}/annotations")
    for f in sorted(os.listdir(f"{root}/meta")):
        m = sio.loadmat(f"{root}/meta/{f}")
        ids = m["cls_indexes"].ravel().astype(int) - 1
        keep = np.array([i in labelled for i in ids])
        sio.savemat(f"{root}/meta/{f}", {
            "poses": m["poses"][:, :, keep],
            "cls_indexes": m["cls_indexes"][:, keep],
            "factor_depth": m["factor_depth"],
            "intrinsic_matrix": m["intrinsic_matrix"]})
    r["objects"] = [o for o in r["objects"] if o in labelled]
man.pop("selfcheck_objects", None)
json.dump(man, open(f"{DATA}/manifest.json", "w"), indent=1)
print("[image] pruned", DATA, "to the annotations of", sorted(labelled))
