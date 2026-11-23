"""The station as installed against its documents (a maintainer's tool; the grader never runs it). Each draw is made
once, from a seed drawn from the system's entropy and recorded with the result, and is then part of the task:

    python3 -B tests/review/draw_installed.py cameras  [--seed HEX]   # grader_config.json "installation"
    python3 -B tests/review/draw_installed.py dynamics [--seed HEX]   # docs/nominal_dynamics.json

cameras: every camera's fixed installation error against the drawing, within the published tolerances of
``cameras.installation_tolerance`` (a translation of 40-100 % of the position tolerance in a random direction, a
rotation of 40-100 % of the angle tolerance about a random axis, both in the camera's mount frame; the focal length
30-100 % of the focal tolerance off the data sheet's, either sign). The cell draws its images from the installed
cameras; describe() and the documents give the drawing.

dynamics: the factors between the maker's nominal data (the agent's model files) and the cell's robot: one factor per
quantity, per link mass and per link inertia (the whole tensor scaled: it stays physical) within the mass tolerance,
per joint damping, friction loss and armature within the joint tolerance, uniform in [1 - tol, 1 - tol / 4] or
[1 + tol / 4, 1 + tol] (never within a quarter of the tolerance of the truth). docs/make_models.py reads the file and
writes the given models with the cell's values multiplied by these factors (nominal = true x factor).
Both files stay on the hidden side (tests/); nothing of them reaches the agent's files.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TESTS = os.path.dirname(HERE)
sys.path.insert(0, TESTS)
sys.dont_write_bytecode = True

MASS_TOL = 0.10          # link masses and inertias: within 10 %
JOINT_TOL = 0.30         # joint damping, friction loss, armature: within 30 %
NOMINAL_FILE = os.path.join(HERE, "docs", "nominal_dynamics.json")
CONFIG_FILE = os.path.join(TESTS, "grader_config.json")


def _seed(a):
    return a.seed or os.urandom(16).hex()


def _rng(seed_hex, salt):
    import numpy as np
    b = bytes.fromhex(seed_hex)
    return np.random.default_rng([int.from_bytes(b[i:i + 4], "little") for i in range(0, 16, 4)] + [salt])


def _unit(rng):
    import numpy as np
    v = rng.normal(size=3)
    return v / np.linalg.norm(v)


def draw_cameras(seed_hex):
    from parcelcell.config import load_config
    cfg = load_config(CONFIG_FILE)
    tol = cfg["cameras"]["installation_tolerance"]
    rng = _rng(seed_hex, 0xCA3E5A)
    out = {}
    for name in sorted(cfg["cameras"]["list"]):
        c = cfg["cameras"]["list"][name]
        t = tol["wrist" if "flange_site" in c else name]
        dp = _unit(rng) * float(t["position_m"]) * float(rng.uniform(0.4, 1.0))
        ang = math.radians(float(t["angle_deg"])) * float(rng.uniform(0.4, 1.0))
        rv = _unit(rng) * ang
        fs = 1.0 + float(tol["focal_rel"]) * float(rng.uniform(0.3, 1.0)) * float(rng.choice([-1.0, 1.0]))
        out[name] = {"d_pos_m": [round(float(v), 6) for v in dp], "d_rotvec_rad": [round(float(v), 7) for v in rv],
                     "focal_scale": round(fs, 6)}
    return {"seed_hex": seed_hex, "tool": "tests/review/draw_installed.py cameras", "cameras": out}


def _factor(rng, tol):
    s = float(rng.choice([-1.0, 1.0]))
    return round(1.0 + s * float(rng.uniform(tol / 4.0, tol)), 5)


def draw_dynamics(seed_hex):
    from parcelcell.config import load_config
    from parcelcell.robot import build_robot_spec
    cfg = load_config(CONFIG_FILE)
    wc = cfg["world"]
    spec = build_robot_spec(bool(wc.get("lock_gripper_couplers", True)), wc.get("robot_mounts") or None)
    m = spec.compile()
    rng = _rng(seed_hex, 0xD1A3)
    bodies, joints = {}, {}
    for b in range(1, m.nbody):
        if m.body_mass[b] <= 0:
            continue
        bodies[m.body(b).name] = {"mass": _factor(rng, MASS_TOL), "inertia": _factor(rng, MASS_TOL)}
    for j in range(m.njnt):
        name = m.joint(j).name
        if not name or int(m.jnt_type[j]) == 0:
            continue
        joints[name] = {"damping": _factor(rng, JOINT_TOL), "frictionloss": _factor(rng, JOINT_TOL),
                        "armature": _factor(rng, JOINT_TOL)}
    return {"seed_hex": seed_hex, "tool": "tests/review/draw_installed.py dynamics",
            "rule": "nominal = the cell's value x factor; masses and inertias within %g, joint damping, friction loss "
                    "and armature within %g, never within a quarter of the tolerance of the truth" % (MASS_TOL, JOINT_TOL),
            "mass_tolerance": MASS_TOL, "joint_tolerance": JOINT_TOL, "bodies": bodies, "joints": joints}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("cameras", "dynamics"))
    ap.add_argument("--seed", default=None)
    a = ap.parse_args(argv)
    seed = _seed(a)
    if a.what == "cameras":
        with open(CONFIG_FILE) as fh:
            text = fh.read()
        inst = draw_cameras(seed)
        lines = text.split("\n")
        at = [i for i, ln in enumerate(lines) if ln.startswith(' "installation": ')]
        if len(at) != 1:
            raise SystemExit("grader_config.json needs one top-level line ' \"installation\": ...,'")
        lines[at[0]] = ' "installation": %s,' % json.dumps(inst, sort_keys=True)
        new = "\n".join(lines)
        json.loads(new)
        tmp = CONFIG_FILE + ".tmp"
        with open(tmp, "w") as fh:
            fh.write(new)
        os.replace(tmp, CONFIG_FILE)
        print(json.dumps(inst, indent=1))
    else:
        d = draw_dynamics(seed)
        with open(NOMINAL_FILE, "w") as fh:
            json.dump(d, fh, indent=1, sort_keys=True)
            fh.write("\n")
        print("bodies %d, joints %d -> %s" % (len(d["bodies"]), len(d["joints"]), NOMINAL_FILE))
    return 0


if __name__ == "__main__":
    sys.exit(main())
