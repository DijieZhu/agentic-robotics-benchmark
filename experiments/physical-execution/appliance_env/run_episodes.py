"""Run whole episodes (episode.py) with the privileged reference, or untouched, and record the credit (P2).

The reference reads the open request from the episode (privileged) and runs the matching primitive;
when a primitive fails it holds still until that request's deadline, so the episode carries on.

    python -I run_episodes.py --assets ~/arb-data/robocasa-assets/extracted/fixtures \
        --out ../outputs/p2/episodes.jsonl --models stovetops/Stovetop002 --seeds 0 1 [--untouched]
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import appliance_env as E  # noqa: E402
import episode as EP  # noqa: E402
import reference as R  # noqa: E402


def idle_until(drv, ep, rid, deadline):
    """Hold the hand still until request `rid` has closed (or the deadline)."""
    while ep.current is not None and ep.current["id"] == rid and ep.env.t < deadline:
        drv.tick(drv.site_pos(), drv.site_R(), -1.0)


def run(assets, model, seed, untouched=False, robot_z=None):
    mount = {"robot_z": robot_z} if robot_z is not None else None
    ep = EP.Episode(E.ApplianceEnv(E.unit_spec(model), assets, render=False, mount=mount), seed)
    ep.reset()
    drv = R.Driver(ep)
    drv.MAX_STEPS = 100000
    t0 = time.time()
    log = []
    while ep.current is not None:
        r = ep.current
        if untouched:
            ep.step([0, 0, 0, 0, 0, 0, -1, 0])
            continue
        try:
            if r["what"] in ("burner_on", "burner_off"):
                drv.knob(r["target"], "on" if r["what"] == "burner_on" else "off")
            elif r["what"] == "preheat":
                drv.knob("temp", "on")
            elif r["what"] == "oven_off":
                drv.knob("temp", "off")
            elif r["what"] in ("start", "stop"):
                drv.button(r["target"])
            else:
                drv.door("open")
            log.append({"id": r["id"], "status": "ok", "steps": drv.steps})
        except R.PrimitiveError as exc:
            log.append({"id": r["id"], "status": "fail", "stage": exc.stage, "error": str(exc)[:200]})
        idle_until(drv, ep, r["id"], r["deadline_t"])
    sc = ep.score()
    ep.env.close()
    return {"model": model, "seed": seed, "untouched": untouched, "score": sc["score"], "requests": sc["requests"],
            "reference": log, "t": ep.env.t, "wall_s": round(time.time() - t0, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--untouched", action="store_true")
    ap.add_argument("--robot-z", type=float, default=None, help="override the robot base height (m)")
    a = ap.parse_args()
    assets = os.path.expanduser(a.assets)
    with open(a.out, "a") as f:
        for model in a.models:
            for seed in a.seeds:
                rec = run(assets, model, seed, a.untouched, a.robot_z)
                rec["robot_z"] = a.robot_z
                f.write(json.dumps(rec) + "\n")
                f.flush()
                print("%-24s seed %-3d score %.2f  %s" % (model, seed, rec["score"], " | ".join(
                    "%s %s%s" % (q["text"], "R" if q["reached"] else "-", "C" if q["completed"] else "-")
                    for q in rec["requests"])), flush=True)


if __name__ == "__main__":
    main()
