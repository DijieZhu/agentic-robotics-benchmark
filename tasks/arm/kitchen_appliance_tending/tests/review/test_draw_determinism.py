"""Is a camera image a function of the state alone? (kitchen_render; a review tool, run in the task image)

    python3 -B test_draw_determinism.py [--tests DIR] [--render-dir DIR] [--units u0,u3] [--steps 6]

DIR defaults: the task's tests/ (this file's parent directory when it sits in tests/review/, else /task/tests);
--render-dir: a directory holding the kitchen_render.py under test (default: the tests directory).
Two child processes run the same episodes (a gas and a glass-ceramic stovetop, burners lit, the arm moving, so
flames flicker and the hand's shadow moves): child A draws left, right, wrist once per step; child B draws each camera
10 times at the first step, then per step a different order with repeats (wrist, left, left, right, wrist, left,
right) and one extra draw before the episode.  Fails (exit 1) when a repeated draw in B differs from the first, or an
image of B differs from A's image of the same (unit, step, camera): what is drawn before an image, how often, and in
which process must not change it.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CAMS = ("left", "right", "wrist")
ACTION = [0.01, 0.0, -0.01, 0.0, 0.0, 0.0, -1.0, 0.0]


def default_tests():
    up = os.path.dirname(HERE)
    if os.path.basename(HERE) == "review" and os.path.isfile(os.path.join(up, "kitchen_render.py")):
        return up
    return "/task/tests"


def child(args):
    sys.path.insert(0, args.tests)
    if args.render_dir:
        sys.path.insert(0, args.render_dir)
    import numpy as np
    import appliance_env as E
    import kitchen_render as KR
    units = json.load(open(os.path.join(args.tests, "units.json")))
    out = {"renderer": os.path.abspath(KR.__file__), "img": {}, "repeat_fail": [], "kinds": {}}

    def h(img):
        return hashlib.sha256(np.ascontiguousarray(img).tobytes()).hexdigest()[:16]

    for uid in args.units.split(","):
        c = units[uid]
        spec = E.unit_spec(c["model"], files=E.read_unit_files(os.path.join(args.tests, c["dir"])),
                           quirks=c.get("quirks"))
        env = E.ApplianceEnv(spec, None, render=False)
        env.reset(args.seed)
        u = env.unit
        for loc in u.knobs:                        # every burner lit, at two thirds of its range
            lo = E.BURNER_ON_MIN + 0.03
            env.set_joint("knob_%s_joint" % loc, lo + 0.66 * ((u.knob_hi[loc] - 0.01) - lo))
        v = KR.open_view(env)
        out["kinds"][uid] = sorted(set(b["kind"] for b in v.episode.fx["burners"].values()))
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
                key = "%s/%d/%s" % (uid, step, cam)
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
    ap.add_argument("--render-dir", default=None)
    ap.add_argument("--units", default="u0,u3")
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--child", default=None)
    args = ap.parse_args()
    if args.child:
        return child(args)
    res = {}
    for name in ("A", "B"):
        cmd = [sys.executable, "-B", os.path.abspath(__file__), "--child", name, "--tests", args.tests,
               "--units", args.units, "--steps", str(args.steps), "--seed", str(args.seed)]
        if args.render_dir:
            cmd += ["--render-dir", args.render_dir]
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
        lines = [l for l in p.stdout.splitlines() if l.startswith("RESULT ")]
        if p.returncode != 0 or not lines:
            print("child %s failed (exit %d):\n%s" % (name, p.returncode, p.stderr[-3000:]))
            return 2
        res[name] = json.loads(lines[-1][7:])
    a, b = res["A"], res["B"]
    differ = sorted(k for k in a["img"] if b["img"].get(k) != a["img"][k])
    print("renderer:", a["renderer"])
    print("burner kinds:", a["kinds"])
    print("images compared: %d (unit, step, camera); differing between processes / draw orders: %d"
          % (len(a["img"]), len(differ)), differ[:8])
    print("repeated draws in B that differ from the first: %d" % len(b["repeat_fail"]), b["repeat_fail"][:8])
    ok = not differ and not b["repeat_fail"] and len(a["img"]) == len(b["img"])
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
