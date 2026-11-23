"""Independent author controls. Only the public camera envelope goes to the pilot."""
import argparse
import base64
import json
from pathlib import Path

import numpy as np

from check_litter import OnDemandWorld, Referee
from check_rescue import World, check_known_returns, run


def flight(seed, mode, folder, full):
    w = (World if full else OnDemandWorld)("rescue", seed)
    rec = Referee(w, capture=True)
    try:
        error = None
        try:
            p = run(rec, feedback=mode)
        except RuntimeError as exc:
            error = str(exc)
            tb = exc.__traceback__
            while tb.tb_frame.f_code is not run.__code__:
                tb = tb.tb_next
            p = tb.tb_frame.f_locals["p"]
        stages = {e["event"]: e["frame"] for e in p.trace if e["event"] in
                  ("sled_found","flange_survey","depot_choice","sled_reacquired","pickup_attempt",
                   "entry_jambs","exit_jambs")}
        fixes = [e for e in p.trace if e["event"] == "station_fix"]
        tracking = [e for e in p.trace if e["event"] == "floor_track"]
        errors, priors, offsets = [], [], []
        for e in tracking:
            truth = (rec.poses[e["frame"]]-rec.poses[e["start_frame"]])[:2]
            nominal = np.array(e["target_delta"])
            errors.append(float(np.linalg.norm(nominal-e["correction_m"]-truth)))
            priors.append(float(np.linalg.norm(nominal-truth)))
        for e in fixes:
            x, y = e["cell"][:2]
            offset = (rec.poses[e["frame"]][:2]-w.layout.xy((x+2,y))).tolist()
            e["referee_center_error_xy_m"] = offset
            offsets.append(float(np.linalg.norm(offset)))
        if w.metrics()["success"]:
            check_known_returns(p.trace)
        result = dict(seed=str(seed), mode=mode, full_render=full, policy_error=error,
                      failure=w.failure, impact=w.impact_geom, trim=w.bias_sign,
                      wide=w.layout.flange > .06, exit_side=w.layout.loading[1],
                      cargo_column=w.layout.outlet, grasp_frame=rec.grasp_frame,
                      stages=stages, last_fix=fixes[-1] if fixes else None,
                      visits=len(fixes), visual_closures=sum(e["method"]=="patch_revisit" for e in fixes),
                      floor_pairs=sum(e["pieces"] for e in tracking),
                      registered_pairs=sum(e["registered"] for e in tracking),
                      track_mae_m=float(np.mean(errors)) if errors else None,
                      command_mae_m=float(np.mean(priors)) if priors else None,
                      worst_center_m=max(offsets,default=None), **w.metrics())
        for name, value in (("outcome", result), ("trace", p.trace), ("actions", rec.actions)):
            (folder/f"{mode}-{name}.json").write_text(json.dumps(value, indent=2 if name!="actions" else None))
        images = {**stages, "stop": w.steps}
        if fixes:
            images["architecture"] = fixes[0]["before_frame"]
        examples = [e["example"] for e in tracking if e.get("example")]
        if examples:
            e = max(examples, key=lambda f: f["inliers"])
            images.update(track_from=e["before"], track_to=e["after"])
        closures = [e for e in fixes if e["method"] == "patch_revisit"]
        if closures:
            e = max(closures, key=lambda f: f["matches"])
            images.update(first_visit=e["reference_frame"], revisited=e["before_frame"])
        for name, step in images.items():
            pair = rec.frames[step]
            if not full and step != w.steps and pair.frames is None:
                continue
            for cam in (0,1):
                (folder/f"{mode}-{name}-{cam}.png").write_bytes(base64.b64decode(pair[cam]))
        print(json.dumps(result), flush=True)
        return rec.actions, result
    finally:
        w.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("seed", type=int)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--full", action="store_true")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--paired", action="store_true")
    modes.add_argument("--odometry", action="store_true", help="withhold corrections from takeoff")
    args = parser.parse_args()
    args.folder.mkdir(exist_ok=True, parents=True)
    a, p = flight(args.seed, "odometry" if args.odometry else "rgb", args.folder, args.full)
    if args.paired:
        b, n = flight(args.seed, "prelift", args.folder, args.full)
        same = next((i for i,(x,y) in enumerate(zip(a,b)) if x!=y), min(len(a),len(b)))
        assert a[:same]==b[:same]
        if p["steps_to_lift"]:
            assert n["steps_to_lift"]==p["steps_to_lift"] and same >= p["steps_to_lift"]
        print(json.dumps(dict(seed=str(args.seed),identical_prefix=same)), flush=True)
    if not args.odometry:
        assert not p["policy_error"] and not p["failure"] and p["delivered"], p
        assert 0 < p["steps_to_tool"] < p["grasp_frame"] < p["steps_to_lift"] < p["steps_to_cargo"]
        assert p["steps_to_cargo"] < p["entry_clear"] < p["exit_clear"] < p["steps"]
