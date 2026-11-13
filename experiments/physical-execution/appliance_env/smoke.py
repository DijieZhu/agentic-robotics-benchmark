"""P1 smoke test: build a few units, step them, save the three camera images as one contact sheet.

    MUJOCO_GL=osmesa python -I smoke.py --assets ~/arb-data/robocasa-assets/extracted/fixtures \
        --out ../outputs/p1/smoke stovetops/Stovetop002 microwaves/Microwave011 ...
"""
import argparse
import os
import sys
import time

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import appliance_env as E  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("models", nargs="+")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rows = []
    for model in a.models:
        env = E.ApplianceEnv(E.unit_spec(model), os.path.expanduser(a.assets))
        t0 = time.time()
        env.reset(a.seed)
        knobs = list(env.unit.knobs)
        if knobs:                                  # one burner on, to show its flame
            env.set_joint("knob_%s_joint" % knobs[0], 1.0)
        obs = env.step([0, 0, 0, 0, 0, 0, -1, 7])[0]
        print(model, "build+step %.1fs" % (time.time() - t0), env.unit_state(), flush=True)
        rows.append([Image.fromarray(obs[c]).resize((320, 240)) for c in ("left", "right", "wrist")])
        env.close()
    sheet = Image.new("RGB", (960, 240 * len(rows)))
    for i, row in enumerate(rows):
        for j, im in enumerate(row):
            sheet.paste(im, (j * 320, i * 240))
    sheet.save(os.path.join(a.out, "contact_sheet.png"))


if __name__ == "__main__":
    main()
