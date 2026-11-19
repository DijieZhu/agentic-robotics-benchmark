"""Is a camera image a function of the state alone, and do the grader's and the commissioning service's copies of
the bench draw the same bytes? (lab_render; a review tool, run in the task image as root or in the verifier image;
the grader never runs it)

    python3 -B test_draw_determinism.py [--tests DIR] [--service DIR] [--seeds 9001,9002] [--steps 6] [--reloads 12]
        [--processes 12]

DIR defaults: --tests the task's tests/ (this file's parent directory when it sits in tests/review/, else
/task/tests); --service the commissioning service's folder (/opt/labbench-commission in the task image, the task's
environment/commission_service/ in a checkout).
Two child processes run the same episodes on a maintainer's unit (properties inside the documented ranges; neither
a graded unit nor one of the service's development units), the arm moving, so the hand's shadow moves. Child A
imports lab_env, lab_sim and lab_render from --tests and draws left, right, wrist, rack once per step (without the
sensor noise). Child B imports them from --service, draws one extra image before the episode, each camera 10 times
at the first step, then per step another order with repeats. Child R (from --tests) keeps two benches open and
draws the left and right camera of each in turn, --reloads times: every turn loads the other bench's model into the
renderer again. Then --processes more fresh processes, by turns from --tests and from --service, each draw the first
two steps of A's episodes. Fails (exit 1) when a repeated draw in B differs from the first, an image of B or of one
of the fresh processes differs from A's image of the same (seed, step, camera), or a bench's image in R differs
from its first: what is drawn before an image, how often, in which process, from which of the two copies and after
how many other benches must not change it. (The reloads are what found the draw order of the robot's shells:
lab_render.robot_model. The fresh processes are there because an order that holds within a process need not be the
same in the next one: keyed_nut_peg_assembly's record.)
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CAMS = ("left", "right", "wrist", "rack")
UNIT = {"rotor_friction": 0.21, "handle_play": 7.0, "rotor_inertia": 0.019}
ACTION = [0.004, 0.003, -0.004, 0.0, 0.0, 0.01, 1.0, 0]


def default_tests():
    up = os.path.dirname(HERE)
    if os.path.basename(HERE) == "review" and os.path.isfile(os.path.join(up, "lab_render.py")):
        return up
    return "/task/tests"


def default_service(tests):
    if os.path.isfile("/opt/labbench-commission/lab_render.py"):
        return "/opt/labbench-commission"
    return os.path.join(os.path.dirname(tests), "environment", "commission_service")


class NoNoise(object):
    @staticmethod
    def normal(mu, sd, shape):
        import numpy as np
        return np.zeros(shape, np.float32)


def child(args):
    sys.path.insert(0, args.service if args.child in ("B", "S") else args.tests)
    import numpy as np
    import lab_env as E
    import lab_render as R
    import lab_sim as S
    out = {"env": os.path.abspath(E.__file__), "renderer": os.path.abspath(R.__file__), "img": {}, "repeat_fail": []}

    def h(img):
        return hashlib.sha256(np.ascontiguousarray(img).tobytes()).hexdigest()[:16]

    if args.child == "R":
        envs = []
        for k in [int(x) for x in args.seeds.split(",")][:2]:
            env = E.LabEnv(UNIT, "maintainer", render=False)
            env.reset(k)
            for _ in range(3):
                env.step(ACTION)
            envs.append((k, env))
        for turn in range(args.reloads):
            for k, env in envs:
                for cam in ("left", "right"):
                    _bit, w, hh, _fovy, name = E.CAMS[cam]
                    dg = h(S.render(env.p.model, env.p.data, name, w, hh, NoNoise, env.renderers))
                    key = "%d/%s" % (k, cam)
                    if out["img"].setdefault(key, dg) != dg:
                        out["repeat_fail"].append("%s at turn %d" % (key, turn))
        for _k, env in envs:
            env.close()
        print("RESULT " + json.dumps(out), flush=True)
        return

    for k in [int(x) for x in args.seeds.split(",")]:
        env = E.LabEnv(UNIT, "maintainer", render=False)
        env.reset(k)
        m, d = env.p.model, env.p.data

        def draw(cam):
            _bit, w, hh, _fovy, name = E.CAMS[cam]
            return S.render(m, d, name, w, hh, NoNoise, env.renderers)

        if args.child == "B":
            draw("wrist")                          # an extra image before the episode's first
        for step in range(args.steps):
            env.step(ACTION)
            if args.child in ("A", "S"):                # S: A's draws, from the service's copy
                seq = list(CAMS)
            elif step == 0:
                seq = [c for c in CAMS for _ in range(10)]
            else:
                seq = ["wrist", "rack", "left", "left", "right", "wrist", "left", "rack", "right"]
            for cam in seq:
                key = "%d/%d/%s" % (k, step, cam)
                dg = h(draw(cam))
                if key in out["img"] and out["img"][key] != dg:
                    out["repeat_fail"].append(key)
                out["img"].setdefault(key, dg)
        env.close()
    print("RESULT " + json.dumps(out), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tests", default=default_tests())
    ap.add_argument("--service", default=None)
    ap.add_argument("--seeds", default="9001,9002")
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--reloads", type=int, default=12)
    ap.add_argument("--processes", type=int, default=12)
    ap.add_argument("--child", default=None)
    args = ap.parse_args()
    if args.service is None:
        args.service = default_service(args.tests)
    if args.child:
        return child(args)
    res = {}

    def run_child(name, steps):
        cmd = [sys.executable, "-B", os.path.abspath(__file__), "--child", name, "--tests", args.tests,
               "--service", args.service, "--seeds", args.seeds, "--steps", str(steps), "--reloads", str(args.reloads)]
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
        lines = [l for l in p.stdout.splitlines() if l.startswith("RESULT ")]
        if p.returncode != 0 or not lines:
            print("child %s failed (exit %d):\n%s" % (name, p.returncode, p.stderr[-3000:]))
            return None
        return json.loads(lines[-1][7:])

    for name in ("A", "B", "R"):
        res[name] = run_child(name, args.steps)
        if res[name] is None:
            return 2
    fresh_off = []                                      # fresh processes whose images are not A's
    for i in range(args.processes):
        f = run_child("A" if i % 2 == 0 else "S", min(args.steps, 2))
        if f is None:
            return 2
        bad = sorted(k for k, v in f["img"].items() if res["A"]["img"].get(k) != v)
        if bad or not f["img"]:
            fresh_off.append("process %d (%s): %s" % (i, "tests" if i % 2 == 0 else "service", bad[:4]))
    a, b, r = res["A"], res["B"], res["R"]
    differ = sorted(k for k in a["img"] if b["img"].get(k) != a["img"][k])
    print("A: %s, %s" % (a["env"], a["renderer"]))
    print("B: %s, %s" % (b["env"], b["renderer"]))
    print("images compared: %d (seed, step, camera); differing between the two copies / processes / draw orders: %d"
          % (len(a["img"]), len(differ)), differ[:8])
    print("repeated draws in B that differ from the first: %d" % len(b["repeat_fail"]), b["repeat_fail"][:8])
    print("R: %d reloads of two benches, %d images each time; images that differ from their first: %d"
          % (args.reloads, len(r["img"]), len(r["repeat_fail"])), r["repeat_fail"][:8])
    print("fresh processes: %d, each %d images; processes with an image that is not A's: %d"
          % (args.processes, len(args.seeds.split(",")) * min(args.steps, 2) * len(CAMS), len(fresh_off)), fresh_off[:6])
    ok = (not differ and not b["repeat_fail"] and not r["repeat_fail"] and not fresh_off and len(a["img"]) == len(b["img"])
          and os.path.dirname(a["renderer"]) != os.path.dirname(b["renderer"]))
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
