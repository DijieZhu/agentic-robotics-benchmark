"""Example controller: the complete wire, no policy.

The grader launches `run.sh` once per run, waits for the line READY on stdout, then sends one
JSON object per line on stdin and expects one JSON object per line on stdout:

    {"type": "reset"}                               -> {"ok": true}
    {"type": "act", "t": 17, "obs": {...}}          -> {"action": [dx, dy, dz, droll, dpitch, dyaw, grip, mark]}
    {"type": "close"}                               -> (exit)

Every reply is due within 30 s. stdout is the wire: print diagnostics to stderr only.
The same `t` can be asked twice with slightly different proprioception; answer each from the
observation it carries (never replay by `t`).

This controller holds still with the gripper open, so it scores exactly 0: it exists to show
the message shapes, how to decode the cameras, and where a policy plugs in.
"""
import base64
import io
import json
import sys

try:
    import numpy as np
    from PIL import Image
except ImportError:          # both are installed in the sandbox; the skeleton runs without them
    np = Image = None


def decode_rgb(png_b64):
    """base64 PNG -> uint8 array (H, W, 3)."""
    return np.array(Image.open(io.BytesIO(base64.b64decode(png_b64))).convert("RGB"))


def decode_depth(png_b64):
    """base64 16-bit PNG in millimetres -> float32 metres (H, W)."""
    return np.array(Image.open(io.BytesIO(base64.b64decode(png_b64)))).astype("float32") / 1000.0


def policy(t, obs):
    """Return the 8-number action for this observation. Replace this.

    obs keys: t, request (null or {text, deadline_t}), rgb and depth ({agentview, left_view,
    oblique, eye_in_hand}, base64 PNG), joint_pos, joint_vel (7), gripper_width, ee_pose
    ([x, y, z, qw, qx, qy, qz] of the tool centre point in the robot base frame).
    action: Cartesian delta of the tool centre point (m, rad; clipped to 0.05 / 0.25 per
    step), grip in [0, 1] (0 open, 1 closed), mark (0 none, 1 "done: delivered",
    2 "done: no such object"; the request is judged, and the desk scored, at that step)."""
    if np is not None and t == 0:
        rgb = decode_rgb(obs["rgb"]["agentview"])
        depth = decode_depth(obs["depth"]["agentview"])
        print("[example] agentview %s, depth %.2f..%.2f m, ee %s" % (
            rgb.shape, float(depth[depth > 0].min()), float(depth.max()), [round(v, 3) for v in obs["ee_pose"][:3]]), file=sys.stderr, flush=True)
    if obs.get("request"):
        pass    # a request is open: obs["request"]["text"] names the object, deadline_t the last step to mark it
    return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0]


def main():
    print("READY", flush=True)
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        msg = json.loads(line)
        kind = msg.get("type")
        if kind == "reset":
            print(json.dumps({"ok": True}), flush=True)
        elif kind == "act":
            try:
                action = policy(int(msg.get("t", 0)), msg["obs"])
                print(json.dumps({"action": [float(v) for v in action[:7]] + [int(action[7])]}), flush=True)
            except Exception as exc:  # noqa: BLE001 - a reply must always go out
                print(json.dumps({"action": [0.0] * 7 + [0], "error": repr(exc)}), flush=True)
        elif kind == "close":
            break


if __name__ == "__main__":
    main()
