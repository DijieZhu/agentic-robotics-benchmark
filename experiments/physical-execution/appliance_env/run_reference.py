"""Run the privileged reference's primitives (reference.py) on units and record the outcomes (P2).

One trial = a fresh episode (unit pose jitter from the seed), an initial state set before the first
action, one primitive. Writes one JSON line per trial; with --video, a wrist + fixed-camera video.

    python -I run_reference.py --assets ~/arb-data/robocasa-assets/extracted/fixtures \
        --out ../outputs/p2/trials.jsonl --models stoves/Stove026 --tasks knob_on knob_off --seeds 0 1 2
"""
import argparse
import json
import os
import sys
import time
import traceback

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import appliance_env as E  # noqa: E402
import reference as R  # noqa: E402

TASKS = ("knob_on", "knob_off", "temp_on", "start", "stop", "door_open", "door_close")


def applicable(env, task):
    u = env.unit
    if task in ("knob_on", "knob_off"):
        return bool(u.knobs)
    if task == "temp_on":
        return u.temp_knob is not None
    if task in ("start", "stop"):
        return "start_button" in u.buttons and "stop_button" in u.buttons
    if task in ("door_open", "door_close"):
        return u.door is not None and u.handle is not None
    return False


def setup_and_run(env, drv, task, rng):
    """Initial state (set before the first action) and the primitive. Returns (target, result dict)."""
    u = env.unit
    if task in ("knob_on", "knob_off"):
        loc = sorted(u.knobs)[int(rng.integers(len(u.knobs)))]
        if task == "knob_off":
            env.set_joint("knob_%s_joint" % loc, float(rng.uniform(0.8, 1.6)) * (1 if env.model.joint(
                "u_knob_%s_joint" % loc).range[1] > 0 else -1))
        return loc, drv.knob(loc, "on" if task == "knob_on" else "off")
    if task == "temp_on":
        return "temp", drv.knob("temp", "on")
    if task == "start":
        return "start_button", drv.button("start_button")
    if task == "stop":
        u.turned_on = True                      # initial state: running (door closed)
        return "stop_button", drv.button("stop_button")
    if task == "door_open":
        return "door", drv.door("open")
    if task == "door_close":
        qa, lo, hi, _ = u.door
        f = float(rng.uniform(0.92, 1.0))
        q = lo + f * (hi - lo) if lo >= 0 else hi - f * (hi - lo)
        env.set_joint(env.model.joint(int(env.model.jnt_qposadr.tolist().index(qa))).name[2:], q)
        return "door", drv.door("close")
    raise NotImplementedError(task)


def run_one(assets, model, task, seed, video_dir=None):
    env = E.ApplianceEnv(E.unit_spec(model), assets, render=video_dir is not None)
    env.reset(seed)
    rng = np.random.default_rng([seed, 99])
    rec = {"model": model, "task": task, "seed": seed}
    if not applicable(env, task):
        rec["status"] = "n/a"
        return rec
    frames = []
    drv = R.Driver(env)
    if video_dir is not None:
        orig = drv.tick

        def tick(*a, **k):
            k["cams"] = 7 if drv.steps % 3 == 0 else 0
            obs = orig(*a, **k)
            if "left" in obs:
                frames.append(np.concatenate([obs["left"], obs["right"], np.pad(
                    obs["wrist"], ((0, 48), (0, 64), (0, 0)))], axis=1))
            return obs
        drv.tick = tick
    t0 = time.time()
    try:
        target, res = setup_and_run(env, drv, task, rng)
        rec.update(status="ok", target=target, **res)
    except R.PrimitiveError as exc:
        rec.update(status="fail", stage=exc.stage, error=str(exc)[:300])
        if exc.stage == "plan":
            import collections
            rec["why"] = collections.Counter(w.split(" ", 1)[1].split(" ")[0] + " " + w.split(" ")[-1]
                                             if "hits" in w else w.split(" ", 1)[1].split(" ")[0] for w in drv.why).most_common(4)
    except Exception as exc:  # noqa: BLE001 - every failure is recorded
        rec.update(status="error", error="%s: %s" % (type(exc).__name__, str(exc)[:300]),
                   tb=traceback.format_exc()[-800:])
    rec.update(steps=drv.steps, wall_s=round(time.time() - t0, 1), state=env.unit_state())
    if video_dir is not None and frames:
        import imageio.v2 as imageio
        os.makedirs(video_dir, exist_ok=True)
        path = os.path.join(video_dir, "%s_%s_%d.mp4" % (model.replace("/", "_"), task, seed))
        imageio.mimwrite(path, frames, fps=10)
        rec["video"] = path
    env.close()
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--tasks", nargs="+", default=list(TASKS))
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--video", default=None, help="directory for videos")
    a = ap.parse_args()
    assets = os.path.expanduser(a.assets)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "a") as f:
        for model in a.models:
            for task in a.tasks:
                for seed in a.seeds:
                    rec = run_one(assets, model, task, seed, a.video)
                    f.write(json.dumps(rec) + "\n")
                    f.flush()
                    print("%-24s %-10s seed %-3d %-5s %s %s" % (model, task, seed, rec["status"],
                          rec.get("error", rec.get("final_angle", "")), rec.get("why", "")), flush=True)


if __name__ == "__main__":
    main()
