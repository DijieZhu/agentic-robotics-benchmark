"""P1 cost measurement: build time, control steps per second without images, seconds per image
(MuJoCo's classic renderer through OSMesa, CPU), and peak memory, per unit.

Run with the task's resource shape in mind (4 CPUs): single-threaded, nothing else running.
    MUJOCO_GL=osmesa OMP_NUM_THREADS=1 python -I perf.py --assets ... --out ../outputs/p1/perf.json MODEL...
"""
import argparse
import json
import os
import resource
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import appliance_env as E  # noqa: E402

MOUNT = {"front_x": 0.55, "robot_z": 0.65}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("models", nargs="+")
    a = ap.parse_args()
    res = {}
    for model in a.models:
        env = E.ApplianceEnv(E.unit_spec(model), os.path.expanduser(a.assets), mount=MOUNT)
        t0 = time.perf_counter()
        env.reset(3)
        build = time.perf_counter() - t0
        t0 = time.perf_counter()
        for _ in range(a.steps):
            env.step([0.002, 0, 0, 0, 0, 0, -1, 0])
        sps = a.steps / (time.perf_counter() - t0)
        per = {}
        for cam in ("left", "right", "wrist"):
            env.render(cam)                     # first image: renderer set-up
            t0 = time.perf_counter()
            for _ in range(20):
                env.render(cam)
            per[cam] = (time.perf_counter() - t0) / 20
        n = 50
        t0 = time.perf_counter()
        for _ in range(n):
            env.step([0.002, 0, 0, 0, 0, 0, -1, 7])
        sps_img = n / (time.perf_counter() - t0)
        env.close()
        res[model] = {"build_s": round(build, 2), "steps_per_s_no_images": round(sps, 1),
                      "s_per_image": {k: round(v, 4) for k, v in per.items()},
                      "steps_per_s_three_images": round(sps_img, 1),
                      "nbody": int(env.model.nbody), "ngeom": int(env.model.ngeom), "nv": int(env.model.nv)}
        print(model, res[model], flush=True)
    res["peak_rss_mb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)
    print("peak RSS %.0f MB" % res["peak_rss_mb"])
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
