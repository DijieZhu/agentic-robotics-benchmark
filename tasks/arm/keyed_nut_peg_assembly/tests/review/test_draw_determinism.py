"""Is a camera image a function of the state alone, and do the grader's and the commissioning service's copies of
the cell draw the same bytes? (keyed_render; a review tool, run in the task image as root; the grader never runs it)

    python3 -B test_draw_determinism.py [--tests DIR] [--service DIR] [--kits 1000,1001] [--steps 6]

DIR defaults: --tests the task's tests/ (this file's parent directory when it sits in tests/review/, else
/task/tests); --service the commissioning service's folder (/opt/keyedcell-commission in the task image, the
task's environment/commission_service/ in a checkout).
Two child processes run the same episodes on maintainer kits (keyed_env.sample_kit(numpy.random.default_rng(k)),
layout seed k + 4000: neither graded kits nor the service's development kits), the arm moving, so the hand's shadow
moves. Child A imports keyed_env and keyed_render from --tests and draws left, right, wrist once per step (without
the sensor noise). Child B imports them from --service, draws one extra image before the episode, each camera 10
times at the first step, then per step another order with repeats (wrist, left, left, right, wrist, left, right).
Fails (exit 1) when a repeated draw in B differs from the first, or an image of B differs from A's image of the same
(kit, step, camera): what is drawn before an image, how often, in which process and from which of the two copies
must not change it.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CAMS = ("left", "right", "wrist")
ACTION = [0.012, 0.004, -0.012, 0.0, 0.0, 0.03, -1.0, 0]


def default_tests():
    up = os.path.dirname(HERE)
    if os.path.basename(HERE) == "review" and os.path.isfile(os.path.join(up, "keyed_render.py")):
        return up
    return "/task/tests"


def default_service(tests):
    if os.path.isfile("/opt/keyedcell-commission/keyed_render.py"):
        return "/opt/keyedcell-commission"
    return os.path.join(os.path.dirname(tests), "environment", "commission_service")


def child(args):
    sys.path.insert(0, args.service if args.child == "B" else args.tests)
    import numpy as np
    import keyed_env as E
    import keyed_render as KR
    out = {"env": os.path.abspath(E.__file__), "renderer": os.path.abspath(KR.__file__), "img": {}, "repeat_fail": [],
           "lamp_gain": {}}

    def h(img):
        return hashlib.sha256(np.ascontiguousarray(img).tobytes()).hexdigest()[:16]

    for k in [int(x) for x in args.kits.split(",")]:
        kit = E.sample_kit(np.random.default_rng(k))
        env = E.NutEnv(kit["nuts"], render=False)
        env.reset(k + 4000, kit["order"])
        out["lamp_gain"][str(k)] = round(float(env.lamp_gain), 6)
        v = KR.open_view(env)
        if args.child == "B":
            v.render("wrist")                      # an extra image before the episode's first
        for step in range(args.steps):
            env.step(ACTION)
            if args.child == "A":
                seq = list(CAMS)
            elif step == 0:
                seq = [c for c in CAMS for _ in range(10)]
            else:
                seq = ["wrist", "left", "left", "right", "wrist", "left", "right"]
            for cam in seq:
                key = "%d/%d/%s" % (k, step, cam)
                d = h(v.render(cam))
                if key in out["img"] and out["img"][key] != d:
                    out["repeat_fail"].append(key)
                out["img"].setdefault(key, d)
        v.close()
        env.close()
    print("RESULT " + json.dumps(out), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tests", default=default_tests())
    ap.add_argument("--service", default=None)
    ap.add_argument("--kits", default="1000,1001")
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--child", default=None)
    args = ap.parse_args()
    if args.service is None:
        args.service = default_service(args.tests)
    if args.child:
        return child(args)
    res = {}
    for name in ("A", "B"):
        cmd = [sys.executable, "-B", os.path.abspath(__file__), "--child", name, "--tests", args.tests,
               "--service", args.service, "--kits", args.kits, "--steps", str(args.steps)]
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
        lines = [l for l in p.stdout.splitlines() if l.startswith("RESULT ")]
        if p.returncode != 0 or not lines:
            print("child %s failed (exit %d):\n%s" % (name, p.returncode, p.stderr[-3000:]))
            return 2
        res[name] = json.loads(lines[-1][7:])
    a, b = res["A"], res["B"]
    differ = sorted(k for k in a["img"] if b["img"].get(k) != a["img"][k])
    print("A: %s, %s" % (a["env"], a["renderer"]))
    print("B: %s, %s" % (b["env"], b["renderer"]))
    print("lamps' brightness of the day per kit:", a["lamp_gain"])
    print("images compared: %d (kit, step, camera); differing between the two copies / processes / draw orders: %d"
          % (len(a["img"]), len(differ)), differ[:8])
    print("repeated draws in B that differ from the first: %d" % len(b["repeat_fail"]), b["repeat_fail"][:8])
    ok = (not differ and not b["repeat_fail"] and len(a["img"]) == len(b["img"])
          and os.path.dirname(a["renderer"]) != os.path.dirname(b["renderer"]) and a["lamp_gain"] == b["lamp_gain"])
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
