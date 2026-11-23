"""Numerics fingerprint of the grading machine (run by grader.py before any mission; maintainers with --write).

    /opt/venv/bin/python3 -I -B numerics_check.py                    # one JSON line: {"fingerprint": ..., ...}
    /opt/venv/bin/python3 -I -B numerics_check.py --write numerics.json

A mission is a long closed loop: a program's joint targets depend on what it computed from the last reading,
and a last-bit difference in the simulation, in a camera frame or in the program's own arithmetic grows into
another trajectory. The fingerprint is the sha256 of three parts, and the grader refuses to grade (a task
error) on a machine that does not reproduce the recorded one, naming the part that differs:

  * simulation: qpos and qvel after every command of a fixed script (integer arithmetic only) on a loaded Go1
    of nobody's site (a seed and properties of its own, no graded seed);
  * frames: one UAV frame and one mast frame of that site, as raw RGB;
  * math: NumPy and SciPy results on fixed data, from a Python started as the grader starts run.sh (so that
    the image's arithmetic pin, numerics_pin.pth, is what sets them).
"""
import hashlib
import json
import os
import subprocess
import sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
PROGRAM_ENV = {"PATH": "/opt/venv/bin:/usr/bin:/bin", "HOME": "/tmp", "LANG": "C.UTF-8"}
SEED, COMMANDS = 20261120, 150


def math_parts():
    import numpy as np
    out = {}

    def put(name, *xs):
        h = hashlib.sha256()
        for x in xs:
            a = np.ascontiguousarray(x)
            h.update(str(a.dtype).encode())
            h.update(a.tobytes())
        out[name] = h.hexdigest()[:16]
    i = np.arange(1, 4097, dtype=np.int64)
    x = ((i*2654435761) % 1000003).astype(np.float64)/1000003.0
    y = ((i*40503+977) % 999983).astype(np.float64)/999983.0
    s, t = 6.0*x-3.0, 6.0*y-3.0
    for dt in (np.float64, np.float32):
        a, b, u, v = s.astype(dt), t.astype(dt), x.astype(dt), y.astype(dt)
        tag = np.dtype(dt).name
        put("trig_"+tag, np.sin(a), np.cos(a), np.tan(a), np.arctan(a), np.arctan2(a, b), np.arcsin(2*u-1),
            np.arccos(2*v-1))
        put("exp_"+tag, np.exp(a), np.expm1(a), np.log(u+dt(1e-3)), np.log1p(u), np.power(u+dt(0.1), b),
            np.tanh(a), np.cbrt(a), np.sqrt(u), np.hypot(a, b))
        for n in (7, 64, 781, 4096):
            put("red_%s_%d" % (tag, n), a[:n].sum(), a[:n].mean(), a[:n].std(), np.dot(a[:n], b[:n]),
                np.linalg.norm(a[:n]), np.cumsum(a[:n]), np.median(a[:n]))
    for n in (3, 40, 781, 2500):
        p = np.column_stack([s[:n], t[:n]])*0.06+np.array([0.47, -0.13])
        A = np.column_stack([2*p[:, 0], 2*p[:, 1], np.ones(n)])
        bb = (p**2).sum(1)
        mu = p.mean(0)
        C = np.cov((p-mu).T) if n > 3 else np.eye(2)
        put("lin_%d" % n, np.linalg.lstsq(A, bb, rcond=None)[0] if n > 3 else 0.0, np.linalg.eigh(C)[0],
            np.abs(np.linalg.eigh(C)[1]), (p-mu) @ np.array([0.6, 0.8]), np.linalg.norm(p-mu, axis=1),
            np.linalg.svd(A, compute_uv=False), np.linalg.pinv(A[:50]) if n > 3 else 0.0)
    M = np.outer(s[:400], t[:400])+np.eye(400)*5
    put("big", M @ M, np.linalg.inv(M[:64, :64]+np.eye(64)*50), np.linalg.solve(M[:300, :300]+np.eye(300)*60, t[:300]),
        M.astype(np.float32) @ M.astype(np.float32))
    try:
        import scipy.linalg as sl
        import scipy.ndimage as nd
        import scipy.optimize as so
        p = np.column_stack([s[:781], t[:781]])
        A = np.column_stack([p, np.ones(781)])
        r = so.least_squares(lambda q: np.hypot(p[:, 0]-q[0], p[:, 1]-q[1])-q[2], [0.1, 0.1, 1.0])
        cost = np.abs(np.subtract.outer(s[:12], t[:12]))
        put("scipy", sl.lstsq(A, s[781:1562])[0], sl.eigh(A.T @ A)[0], sl.svd(A, compute_uv=False), r.x,
            nd.gaussian_filter(M[:64, :64], 1.5), nd.label(M[:64, :64] > 5)[0], so.linear_sum_assignment(cost)[1])
    except ImportError:
        out["scipy"] = "absent"
    return out


if __name__ == "__main__" and sys.argv[1:] == ["--math"]:
    import warnings
    warnings.simplefilter("ignore")
    print(json.dumps(math_parts(), sort_keys=True))
    sys.exit(0)


def simulation_and_frames():
    sys.path.insert(0, os.path.join(HERE, "runtime"))
    import base64
    import io

    import numpy as np
    from PIL import Image
    from mission import World

    def raw(obs):
        return np.asarray(Image.open(io.BytesIO(base64.b64decode(obs["images"][0]))).convert("RGB")).tobytes()

    world = World(SEED)
    frames = hashlib.sha256(raw(world.observe()))
    for k in range(4):
        obs = world.act([0.5 if k % 2 == 0 else -0.25, 0.25])
    frames.update(raw(obs))
    world.advance()
    frames.update(raw(world.look(3)))
    dog = world.go1
    h = hashlib.sha256(np.ascontiguousarray(dog.data.qpos, dtype="<f8").tobytes())
    for k in range(COMMANDS):
        # joint targets from integers: a crouch, a rock fore and aft, alternating diagonal lifts
        bend = (900-((k*7) % 41-20)*6)/1000.0
        lift = ((k // 6) % 2)*0.25
        target = []
        for leg in range(4):
            diagonal = leg in (0, 3)
            thigh = bend+(lift if diagonal == (k % 12 < 6) else 0.0)+((k*3) % 5-2)/100.0*(1 if leg < 2 else -1)
            target += [((k*5) % 7-3)/200.0, thigh, -2.0*bend-(lift if diagonal == (k % 12 < 6) else 0.0)]
        world.act({"target_rad": target, "ticks": 12})
        h.update(np.ascontiguousarray(dog.data.qpos, dtype="<f8").tobytes())
        h.update(np.ascontiguousarray(dog.data.qvel, dtype="<f8").tobytes())
        if world.done:
            break
    frames.update(raw(world.look(6)) if not world.done else b"done")
    world.close()
    return h.hexdigest(), frames.hexdigest()


def program_math():
    proc = subprocess.run([sys.executable, "-I", "-B", os.path.abspath(__file__), "--math"], stdout=subprocess.PIPE,
                          stderr=subprocess.DEVNULL, timeout=120, env=dict(PROGRAM_ENV), check=True)
    return json.loads(proc.stdout.decode("utf-8", "replace").strip().splitlines()[-1])


def main():
    simulation, frames = simulation_and_frames()
    parts = program_math()
    h = hashlib.sha256()
    h.update(simulation.encode())
    h.update(frames.encode())
    h.update(json.dumps(parts, sort_keys=True).encode())
    report = {"fingerprint": h.hexdigest(), "simulation": simulation, "frames": frames, "math": parts}
    recorded = os.path.join(HERE, "numerics.json")
    if os.path.isfile(recorded):
        with open(recorded) as f:
            ref = json.load(f)
        report["differs"] = sorted([k for k in ("simulation", "frames") if ref.get(k) != report[k]]
                                   + ["math."+k for k in sorted(set(parts) | set(ref.get("math", {})))
                                      if ref.get("math", {}).get(k) != parts.get(k)])
    if len(sys.argv) == 3 and sys.argv[1] == "--write":
        with open(sys.argv[2], "w") as f:
            json.dump({**{k: report[k] for k in ("fingerprint", "simulation", "frames", "math")},
                       "what": "sha256 of `simulation` (a loaded Go1 under a fixed script), `frames` (three camera "
                               "frames as raw RGB) and `math` (NumPy and SciPy results from a Python started as a "
                               "mission program is): tests/numerics_check.py"}, f, indent=1, sort_keys=True)
            f.write("\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
