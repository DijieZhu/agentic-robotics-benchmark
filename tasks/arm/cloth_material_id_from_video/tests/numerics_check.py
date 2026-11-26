"""Numerics fingerprint of the grading computation (verify / maintainers only; run by verify.py before anything is graded).

    /opt/venv/bin/python3 -I -B numerics_check.py                       # prints one JSON line
    /opt/venv/bin/python3 -I -B numerics_check.py --write numerics.json  # a maintainer, after a change of the models

The real fabric's model (creal) calls np.exp once per physics step (its speed-dependent friction), and NumPy's
AVX-512 loops round that function differently from its other loops: without a pin, every real outcome of this task
differed between an AVX2 host and an AVX-512 host (1e-9 m to 0.25 mm; one test's score moved from 0.47 to 0.51), and
with NPY_DISABLE_CPU_FEATURES="X86_V4 AVX512_ICL AVX512_SPR" the two hosts agreed to the bit on all 36 outcomes
compared (the OpenBLAS kernel pin alone changed nothing). The image and tests/test.sh set the pin; this check runs
one short simulation in each model with a material and a test of its own (no hidden fabric, no hidden test) and
hashes every recorded vertex position, together with a few NumPy results. verify.py compares the fingerprint with
tests/numerics.json and refuses to grade (a task error) on a machine that does not reproduce it.
"""
import hashlib
import json
import os
import sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402

import cphys as W  # noqa: E402
import creal as R  # noqa: E402
import ctasks as T  # noqa: E402

MATERIAL = dict(young=7.0e4, poisson=0.31, thickness=0.0052, damping=0.0027, mu=0.47)
MASS = 0.0093
TEST = dict(fam="slide", pose=[-0.19, -0.07, 0.17], side=+1, d_release=0.10)
U = 0.62


def parts():
    out = {}
    i = np.arange(1, 2049, dtype=np.int64)
    x = ((i * 2654435761) % 1000003).astype(np.float64) / 1000003.0
    s = 6.0 * x - 3.0
    h = hashlib.sha256()
    for a in (np.exp(s), np.exp(-x / 0.12), np.sin(s), np.cos(s), np.sqrt(x), np.linalg.norm(np.stack([s, x], 1), axis=1), np.cumsum(s)):
        h.update(np.ascontiguousarray(a).tobytes())
    out["numpy"] = h.hexdigest()[:16]
    C, _, _ = T.prog_slide(TEST, U)
    for name, world in (("cphys", W), ("creal", R)):
        sim = world.Sim(MATERIAL, MASS)
        sim.place_flat(TEST["pose"])
        fr = sim.run(C[:700], record_every=50)
        out[name] = hashlib.sha256(np.ascontiguousarray(fr).tobytes()).hexdigest()[:16]
    return out


def fingerprint():
    p = parts()
    return dict(fingerprint=hashlib.sha256(json.dumps(p, sort_keys=True).encode()).hexdigest(), parts=p,
                numpy=np.__version__, mujoco=__import__("mujoco").__version__)


if __name__ == "__main__":
    fp = fingerprint()
    if sys.argv[1:2] == ["--write"]:
        with open(sys.argv[2], "w") as f:
            json.dump(fp, f, indent=1, sort_keys=True)
            f.write("\n")
    print(json.dumps(fp, sort_keys=True))
