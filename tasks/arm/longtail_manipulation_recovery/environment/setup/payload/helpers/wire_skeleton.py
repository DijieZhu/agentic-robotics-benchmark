#!/usr/bin/env python3
"""Minimal controller on the grading wire: holds still and asks for the left camera every 10 steps.
Copy it to ~/submission/ as your controller's starting point; run.sh: exec /opt/venv/bin/python3 -u controller.py"""
import base64
import json
import sys

import numpy as np


def decode(img):
    return np.frombuffer(base64.b64decode(img["b64"]), dtype=img["dtype"]).reshape(img["shape"])


def main():
    print("READY", flush=True)
    for line in sys.stdin:
        msg = json.loads(line)
        kind = msg.get("type")
        if kind == "reset":
            print(json.dumps({"ok": True}), flush=True)
        elif kind == "close":
            print(json.dumps({"ok": True}), flush=True)
            return
        elif kind == "act":
            t, obs = msg["t"], msg["obs"]
            # obs["card"] is present at t = 0 (instruction, object and receptacle catalogue images, kitchen map, intrinsics)
            for name, entry in obs["images"].items():
                rgb = decode(entry["rgb"])            # uint8 HxWx3, row 0 at the top
                depth_mm = decode(entry["depth_mm"]) if "depth_mm" in entry else None   # uint16, millimetres
            action = [0.0] * 12
            action[6] = -1.0                          # gripper open
            action[11] = -1.0                         # arm mode (no base motion)
            cams = 1 if t % 10 == 0 else 0            # next observation: left camera RGB only
            print(json.dumps({"action": action + [cams]}), flush=True)


if __name__ == "__main__":
    main()
