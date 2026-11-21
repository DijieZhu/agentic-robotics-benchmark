"""Is a camera image a function of the state alone, and do the grader's and the agent's copies of the simulator
draw the same bytes? (jar_render; a review tool, run in the task image; the grader never runs it)

    python3 -B test_draw_determinism.py [--tests DIR] [--payload DIR] [--seeds 9001,9002] [--steps 6] [--reloads 12]
        [--processes 12] [--dump FILE] [--expect FILE]

DIR defaults: --tests the task's tests/ (this file's parent directory when it sits in tests/review/, else
/task/tests); --payload the agent's copy (the task's environment/setup/payload/ in a checkout, else /home/user).
Two child processes run the same episodes on two maintainer's units (drawn by jar_sim.draw_unit from fixed seeds:
inside the documented ranges, neither a graded unit nor a practice unit), both arms moving. Child A imports jar_env,
jar_sim and jar_render from --tests and draws the five cameras once per step (without the sensor noise). Child B
imports them from --payload, draws one extra image before the episode, each camera 10 times at the first step, then
per step another order with repeats. Child R (from --tests) keeps two episodes of two different units open (other
jar sizes, other label kinds, another tray) and draws three cameras of each in turn, --reloads times: every turn
writes the other episode's shapes and colours into the renderer again. Then --processes more fresh processes, by
turns from --tests and from --payload, each draw the first two steps of A's episodes. Fails (exit 1) when a repeated
draw in B differs from the first, an image of B or of one of the fresh processes differs from A's image of the same
(seed, step, camera), or an episode's image in R differs from its first: what is drawn before an image, how often,
in which process, from which of the two copies and after how many other episodes must not change it.
--dump writes A's image digests to FILE; --expect compares A's digests with a FILE written elsewhere (another
machine, another build of the image) and fails on a difference.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CAMS = ("overhead", "front", "tray", "wrist_l", "wrist_r")
UNIT_SEEDS = (0x5EED0001, 0x5EED0002)
ACTION = {"l": [0.004, 0.003, -0.004, 0.0, 0.0, 0.01, 1.0], "r": [0.003, -0.004, -0.003, 0.0, 0.01, 0.0, -1.0], "cams": 0}


def default_tests():
    up = os.path.dirname(HERE)
    if os.path.basename(HERE) == "review" and os.path.isfile(os.path.join(up, "jar_render.py")):
        return up
    return "/task/tests"


def default_payload(tests):
    p = os.path.join(os.path.dirname(tests), "environment", "setup", "payload")
    return p if os.path.isfile(os.path.join(p, "jar_render.py")) else "/home/user"


class NoNoise(object):
    @staticmethod
    def normal(mu, sd, shape):
        import numpy as np
        return np.zeros(shape, np.float32)


def child(args):
    sys.path.insert(0, args.payload if args.child in ("B", "S") else args.tests)
    import numpy as np
    import jar_env as E
    import jar_render as R
    import jar_sim as S
    out = {"env": os.path.abspath(E.__file__), "renderer": os.path.abspath(R.__file__), "img": {}, "repeat_fail": []}
    seeds = [int(x) for x in args.seeds.split(",")]

    def h(img):
        return hashlib.sha256(np.ascontiguousarray(img).tobytes()).hexdigest()[:16]

    def episode(i, k):
        env = E.JarEnv(S.draw_unit(UNIT_SEEDS[i % len(UNIT_SEEDS)]), "maintainer", render=False)
        env.reset(k)
        return env

    if args.child == "R":
        envs = []
        for i, k in enumerate(seeds[:2]):
            env = episode(i, k)
            for _ in range(3):
                env.step(ACTION)
            envs.append((k, env))
        for turn in range(args.reloads):
            for k, env in envs:
                for cam in ("front", "tray", "wrist_r"):
                    dg = h(S.render(env.w.m, env.w.d, cam, NoNoise, env.renderers))
                    key = "%d/%s" % (k, cam)
                    if out["img"].setdefault(key, dg) != dg:
                        out["repeat_fail"].append("%s at turn %d" % (key, turn))
        for _k, env in envs:
            env.close()
        print("RESULT " + json.dumps(out), flush=True)
        return

    for i, k in enumerate(seeds):
        env = episode(i, k)

        def draw(cam):
            return S.render(env.w.m, env.w.d, cam, NoNoise, env.renderers)

        if args.child == "B":
            draw("wrist_l")                         # an extra image before the episode's first
        for step in range(args.steps):
            env.step(ACTION)
            if args.child in ("A", "S"):                # S: A's draws, from the agent's copy
                seq = list(CAMS)
            elif step == 0:
                seq = [c for c in CAMS for _ in range(10)]
            else:
                seq = ["wrist_r", "tray", "overhead", "overhead", "front", "wrist_l", "overhead", "tray", "front", "wrist_r", "wrist_l"]
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
    ap.add_argument("--payload", default=None)
    ap.add_argument("--seeds", default="9001,9002")
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--reloads", type=int, default=12)
    ap.add_argument("--processes", type=int, default=12)
    ap.add_argument("--dump", default="")
    ap.add_argument("--expect", default="")
    ap.add_argument("--child", default=None)
    args = ap.parse_args()
    if args.payload is None:
        args.payload = default_payload(args.tests)
    if args.child:
        return child(args)
    res = {}

    def run_child(name, steps):
        cmd = [sys.executable, "-B", os.path.abspath(__file__), "--child", name, "--tests", args.tests,
               "--payload", args.payload, "--seeds", args.seeds, "--steps", str(steps), "--reloads", str(args.reloads)]
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
            fresh_off.append("process %d (%s): %s" % (i, "tests" if i % 2 == 0 else "payload", bad[:4]))
    a, b, r = res["A"], res["B"], res["R"]
    differ = sorted(k for k in a["img"] if b["img"].get(k) != a["img"][k])
    print("A: %s, %s" % (a["env"], a["renderer"]))
    print("B: %s, %s" % (b["env"], b["renderer"]))
    print("images compared: %d (seed, step, camera); differing between the two copies / processes / draw orders: %d"
          % (len(a["img"]), len(differ)), differ[:8])
    print("repeated draws in B that differ from the first: %d" % len(b["repeat_fail"]), b["repeat_fail"][:8])
    print("R: %d reloads of two episodes of two units, %d images each time; images that differ from their first: %d"
          % (args.reloads, len(r["img"]), len(r["repeat_fail"])), r["repeat_fail"][:8])
    print("fresh processes: %d, each %d images; processes with an image that is not A's: %d"
          % (args.processes, len(args.seeds.split(",")) * min(args.steps, 2) * len(CAMS), len(fresh_off)), fresh_off[:6])
    ok = (not differ and not b["repeat_fail"] and not r["repeat_fail"] and not fresh_off and len(a["img"]) == len(b["img"])
          and os.path.dirname(a["renderer"]) != os.path.dirname(b["renderer"]))
    if args.dump:
        with open(args.dump, "w") as fh:
            json.dump(a["img"], fh, indent=0, sort_keys=True)
    if args.expect:
        with open(args.expect) as fh:
            want = json.load(fh)
        off = sorted(k for k in want if a["img"].get(k) != want[k])
        print("against %s: %d of %d images differ" % (args.expect, len(off), len(want)), off[:8])
        ok = ok and not off
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
