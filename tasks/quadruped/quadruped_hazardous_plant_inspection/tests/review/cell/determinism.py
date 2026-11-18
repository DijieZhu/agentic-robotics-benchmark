"""Determinism of the cell: the same site and program give the same score and the same final state hash.

    MUJOCO_GL=osmesa python -I tests/review/cell/determinism.py --site SITE --program DIR --out OUT [--runs 2]
        [--privileged] [--seed N]

Runs the program `--runs` times over the real wire (run_wire.run, each mission in a fresh cell) and compares the
episode score, the final state hash (time, qpos, qvel, dose, every report), the trajectory hash and the dosimeter's
counts. Writes OUT/determinism.json (with the machine's CPU model and the package versions) so that the results of two
machines can be compared (`--compare A.json B.json`).
"""

import argparse
import hashlib
import json
import os
import platform
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))


def cpu_model():
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor()


def versions():
    import mujoco
    import numpy
    import scipy
    return {"python": platform.python_version(), "mujoco": mujoco.__version__, "numpy": numpy.__version__,
            "scipy": scipy.__version__, "cpu": cpu_model(), "machine": platform.node(),
            "OPENBLAS_CORETYPE": os.environ.get("OPENBLAS_CORETYPE"),
            "NPY_DISABLE_CPU_FEATURES": os.environ.get("NPY_DISABLE_CPU_FEATURES"),
            "MUJOCO_GL": os.environ.get("MUJOCO_GL")}


def one(site, program, out, privileged, seed):
    import numpy as np
    import run_wire
    s = run_wire.run(site, program, out, privileged=privileged, seed=seed, log=lambda m: None)
    z = np.load(os.path.join(out, "traj.npz"))
    th = hashlib.sha256(np.ascontiguousarray(z["traj"]).tobytes()).hexdigest()
    return {"episode_score": s["episode_score"], "final_state_sha256": s["final_state_sha256"],
            "trajectory_sha256": th, "sim_time_s": s["sim_time_s"], "done_reason": s["done_reason"],
            "end_reason": s.get("end_reason"), "reports": s["reports_made"], "reports_delivered": s["reports_delivered"],
            "frames": s["frames"], "dose_gy": s["dose_gy"], "parts": s["parts"], "episode_wall_s": s.get("episode_wall_s")}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--site")
    ap.add_argument("--program")
    ap.add_argument("--out")
    ap.add_argument("--runs", type=int, default=2)
    ap.add_argument("--privileged", action="store_true")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--compare", nargs=2, default=None)
    a = ap.parse_args(argv)
    if a.compare:
        A, B = (json.load(open(p)) for p in a.compare)
        keys = ("episode_score", "final_state_sha256", "trajectory_sha256", "sim_time_s", "dose_gy")
        same = {k: A["runs"][0][k] == B["runs"][0][k] for k in keys}
        print(json.dumps({"same": same, "A": {k: A["runs"][0][k] for k in keys} | {"machine": A["versions"]},
                          "B": {k: B["runs"][0][k] for k in keys} | {"machine": B["versions"]}}, indent=1))
        return 0 if all(same.values()) else 1
    runs = []
    for i in range(a.runs):
        runs.append(one(a.site, a.program, os.path.join(a.out, "run%d" % i), a.privileged, a.seed))
        print(json.dumps(runs[-1]), flush=True)
    keys = ("episode_score", "final_state_sha256", "trajectory_sha256", "sim_time_s", "dose_gy", "reports")
    same = all(all(r[k] == runs[0][k] for k in keys) for r in runs)
    res = {"site": a.site, "program": a.program, "runs": runs, "identical": same, "versions": versions()}
    with open(os.path.join(a.out, "determinism.json"), "w") as f:
        json.dump(res, f, indent=1)
    print("determinism: %s over %d runs (%s)" % ("IDENTICAL" if same else "DIFFERENT", len(runs), res["versions"]["cpu"]))
    return 0 if same else 1


if __name__ == "__main__":
    sys.exit(main())
