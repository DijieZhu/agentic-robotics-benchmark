"""Numerics fingerprint of the simulation (verify / maintainers only; run by verify.py before anything is graded).

    /opt/venv/bin/python3 -I numerics_check.py        # prints one JSON line: {"fingerprint": sha256, ...}

The operational-space controller inside the simulation inverts two small matrices with LAPACK every physics step for
each arm (jar_sim.Arm.torque), and OpenBLAS picks its kernels by CPU: on another kernel the simulation differs from
the first step, and over a long episode a controller then runs another trajectory (measured on this task: one
episode of the reference took 5024, 5060, 5128 or 5190 control steps under four kernels, and one practice episode
scored differently). The image pins the kernels (OPENBLAS_CORETYPE=Haswell with OPENBLAS_L2_SIZE=256,
environment/Dockerfile). This check runs STEPS control steps of one episode that is nobody's (a unit and a seed of
its own: not a hidden unit, not a practice unit, not a graded seed, no image drawn) under a fixed script of actions
that needs no math library (Cartesian steps of both hands, a posture change, joint steps, the grippers) and hashes
every state. verify.py compares the hash with tests/numerics.json and refuses to grade (a task error) on a machine
that does not reproduce it.

    /opt/venv/bin/python3 -I numerics_check.py --write numerics.json     # a maintainer, after a change of the simulation
"""

import hashlib
import json
import os
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402

import jar_env as E  # noqa: E402
import jar_sim as S  # noqa: E402

UNIT_SEED = 20261119
SEED = 20261119
STEPS = 160


def script():
    """STEPS actions from integer arithmetic only: 60 Cartesian steps of both hands (they move, turn and close), 40
    with a null-space posture, 60 joint steps."""
    out = []
    home = [float(x) for x in S.HOME_Q]
    for i in range(STEPS):
        a = ((i * 7) % 11 - 5) / 400.0
        b = ((i * 5) % 13 - 6) / 500.0
        c = ((i * 3) % 7 - 3) / 60.0
        grip = -1.0 if i % 40 < 20 else 1.0
        if i < 60:
            left, right = [a, b, -0.002, c, 0.0, -c, grip], [-b, a, -0.001, 0.0, c, c, -grip]
        elif i < 100:
            left = [b, a, 0.003, 0.0, c, 0.0, grip] + [q + 0.2 * ((j + i) % 3 - 1) for j, q in enumerate(home)]
            right = [a, -b, 0.002, c, 0.0, 0.0, -grip] + [q - 0.2 * ((j + i) % 3 - 1) for j, q in enumerate(home)]
        else:
            left = [q + 0.05 * ((i + 2 * j) % 5 - 2) for j, q in enumerate(home)] + [grip]
            right = [q - 0.05 * ((i + 3 * j) % 5 - 2) for j, q in enumerate(home)] + [-grip]
        out.append({"l": left, "r": right, "cams": 0})
    return out


def fingerprint():
    env = E.JarEnv(S.draw_unit(UNIT_SEED), "numerics", render=False)
    env.reset(SEED)
    h = hashlib.sha256()
    d = env.w.d
    h.update(np.ascontiguousarray(d.qpos, dtype="<f8").tobytes())
    for action in script():
        env.step(action)
        h.update(np.ascontiguousarray(d.qpos, dtype="<f8").tobytes())
        h.update(np.ascontiguousarray(d.qvel, dtype="<f8").tobytes())
    if env.unstable():
        raise RuntimeError("the numerics script made the simulation unstable")
    env.close()
    return h.hexdigest()


def main():
    report = {"fingerprint": fingerprint(), "steps": STEPS, "seed": SEED,
              "openblas_coretype": os.environ.get("OPENBLAS_CORETYPE"), "numpy": np.__version__}
    if len(sys.argv) == 3 and sys.argv[1] == "--write":
        with open(sys.argv[2], "w") as fh:
            json.dump({"fingerprint": report["fingerprint"], "steps": STEPS, "seed": SEED,
                       "what": "sha256 of qpos and qvel after each of the script's control steps "
                               "(tests/numerics_check.py), OPENBLAS_CORETYPE=Haswell, OPENBLAS_L2_SIZE=256"},
                      fh, indent=1)
            fh.write("\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
