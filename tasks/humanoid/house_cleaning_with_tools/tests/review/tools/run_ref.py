"""Run the privileged reference on one seed the way verify.py's oracle episode does (CellServer privileged, 4 camera
frames, reference_clean, finalize), inside the task image. Usage: run_ref.py STAGE SEED OUT.json [--nominal] [--norender]"""
import json, os, sys, time
stage, seed, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
nominal = "--nominal" in sys.argv
render = "--norender" not in sys.argv
sys.path.insert(0, stage)
sys.dont_write_bytecode = True
from house_cell.server import CellServer
cfg = json.load(open(os.path.join(stage, "grader_config.json")))
logf = open(out + ".log", "w")
def log(s):
    logf.write(str(s) + "\n"); logf.flush()
t0 = time.time()
srv = CellServer(seed, privileged=True, nominal=nominal, config=cfg, log=log, render=render)
if render:
    for cam in ("ceiling_kitchen", "ceiling_dining", "ceiling_living", "head"):
        r = srv.handle("get_camera", {"name": cam})
        log("camera %s %s" % (cam, r.get("error", "ok")))
r = srv.handle("reference_clean", {})
log("reference %s" % ({k: v for k, v in r.items() if k != "credit_by_part"},))
rec = srv.finalize()
rec["harness_wall_s"] = time.time() - t0
keep = {k: rec.get(k) for k in ("seed", "stage_score", "credit", "credit_by_part", "grease_removed_fraction",
        "crumbs_binned_fraction", "safety_ok", "violations", "safety_peaks", "safety_peak_who", "sim_time_s", "broken",
        "items", "process", "tools_clipped", "lid_angles", "bins", "render_errors", "internal_errors", "reference",
        "harness_wall_s", "camera_frames_total", "faucet_off", "sponge", "damage", "penalty", "credit_earned")}
json.dump(keep, open(out, "w"), indent=1, default=str)
print("seed %d score %.4f safety %s wall %.0f" % (seed, rec["stage_score"], rec["safety_ok"], rec["harness_wall_s"]))
