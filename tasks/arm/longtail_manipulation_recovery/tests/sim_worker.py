"""One restored kitchen as a child process of the grader (root). JSON lines over stdin/stdout:
  -> {"task", "seed", "split", "prefix_file", "qpos_T_file", "cards_file", "roles", "step_budget", "max_frames", "privileged"}
  <- {"ready": true, "restore_max_err": float, "built_s": ...}  or {"ready": false, "error": ...}
  -> {"step": true, "action": [12], "cams": int}   (the first {"step": false, "first": true} is the t=0 observation)
  <- {"obs": wire observation, "success": bool, "sim_s": float, "render_s": float}
  -> {"close": true}
stdout carries only protocol lines; anything else from the simulator goes to the log file."""
import json
import os
import sys
import time
import traceback

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    real_stdout = sys.stdout
    sys.stdout = open(os.devnull, "w")   # RoboCasa and robosuite print to stdout
    spec = json.loads(sys.stdin.readline())
    try:
        import kitchen_sim as K
        prefix = np.load(spec["prefix_file"])["action"]
        kitchen = K.Kitchen(spec["task"], spec["seed"], spec["split"], prefix=prefix)
        ref = np.load(spec["qpos_T_file"])
        q_ref = ref["qpos"]
        q = np.asarray(kitchen.raw.sim.data.qpos, np.float64)
        err = float(np.max(np.abs(q - q_ref))) if q.shape == q_ref.shape else float("inf")
        if q.shape == q_ref.shape:
            kitchen.set_integration_state(ref["integration"])   # the recorded physical state, exactly
        card = {}
        if spec.get("cards_file"):
            card = json.load(open(spec["cards_file"]))
        full_card = dict(card)
        full_card.update({
            "instruction": kitchen.instruction,
            "camera_intrinsics": {c: K.camera_intrinsics(kitchen.raw, c) for c in K.CAMERA_KEYS},
            "kitchen_map": K.kitchen_map(kitchen.raw, spec.get("roles")),
            "step_budget": int(spec["step_budget"]),
        })
    except Exception:
        traceback.print_exc(file=sys.stderr)
        print(json.dumps({"ready": False, "error": traceback.format_exc()[-800:]}), file=real_stdout, flush=True)
        return
    print(json.dumps({"ready": True, "prefix_max_err": err, "shape_ok": bool(q.shape == q_ref.shape), "built_s": kitchen.build_s, "prefix_len": kitchen.prefix_len}),
          file=real_stdout, flush=True)
    budget = int(spec["step_budget"]); max_frames = int(spec.get("max_frames", 3000))
    t = 0
    cam_mask = 15  # the first observation carries all three cameras with depth
    for line in sys.stdin:
        msg = json.loads(line)
        if msg.get("close"):
            break
        if msg.get("first"):
            obs = K.wire_observation(kitchen, 0, 15)
            obs["card"] = full_card
            if spec.get("privileged"):
                obs["_gt"] = {"state_id": spec["id"]}
            reply = {"obs": obs, "success": kitchen.success(), "sim_s": kitchen.sim_s, "render_s": kitchen.render_s}
            print(json.dumps(reply), file=real_stdout, flush=True)
            continue
        a = np.asarray(msg["action"], float)
        t += 1
        kitchen.step(a)
        cams = int(msg.get("cams", 0))
        if kitchen.frames_rendered + bin(cams & 7).count("1") > max_frames:
            cams = 0
        obs = K.wire_observation(kitchen, t, cams)
        if spec.get("privileged"):
            obs["_gt"] = {"state_id": spec["id"]}
        print(json.dumps({"obs": obs, "success": kitchen.success(), "t": t, "sim_s": kitchen.sim_s, "render_s": kitchen.render_s,
                          "frames": kitchen.frames_rendered}), file=real_stdout, flush=True)
        if t >= budget:
            pass
    kitchen.close()


if __name__ == "__main__":
    main()
