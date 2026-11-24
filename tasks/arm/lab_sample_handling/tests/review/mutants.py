"""Mutants of the privileged reference on the fixed set (a maintainer's tool; the grader never runs it).

Each mutant is the reference with one thing a controller could get wrong, in process on the environment as the
grader builds it (the episode's credit is tests/episode.py's). They show that what differs from unit to unit costs
a controller that ignores it, and that a controller that copes keeps its score:

  reference       the reference as it is (expected: every episode 1.0)
  nominal_rack    the racks taken at their nominal places (the instruction's), not where they stand
  calib_wrist     every target (a rack hole, a rotor bore) taken where the unit's wrist calibration on file puts a
                  point seen from 0.2 m straight above it: the error of a controller that trusts T_world_wrist
  calib_fixed     the same through the left camera's calibration on file, from where that camera stands
  card_wrist      the same through the wrist camera of the robot's model (the equipment sheet's nominal pose): the
                  error of a controller that takes the camera to be where the sheet puts it, which is also what a
                  controller gets that "corrects" the pose on file by its offset from the nominal pose
  card_fixed      the same through the left camera's nominal pose on the equipment sheet
  tilt_plus, tilt_minus
                  every rotor bore entered TILT_DEG off its axis, about the bore's tangential axis, one way and
                  the other
  loose_fit       every rotor bore approached 0.8 mm off its axis: inside the nominal bench's 1.0 mm clearance
  lid_let_go      the lid let go at 0.25 rad instead of followed down to shut (it falls shut on a hinge with less
                  friction than the lid's weight makes, and stays open on a stiffer one)
  robust          a residual error of RESIDUAL_MM in a random direction on every target, drawn per load: a
                  controller that has measured its calibration and closes the loop on site

    python3 -B tests/review/mutants.py --out OUT.json [--mutants a,b] [--episodes tests/hidden/episodes.json]
                                       [--units UNITS.json] [--workers 6] [--repeat 1]

`--units` runs the set's seeds on other units' values (another draw of the fit properties and calibration errors:
a JSON {"units": {...}}); `--repeat` runs each episode that many times with different residual draws (robust).
Needs the task image's Python (see select_set.py).
"""

import argparse
import json
import math
import multiprocessing
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TESTS = os.path.dirname(HERE)
TASK = os.path.dirname(TESTS)
ORACLE = os.path.join(TASK, "solution", "oracle")
sys.dont_write_bytecode = True
ENV = {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "MUJOCO_GL": "osmesa",
       "PYOPENGL_PLATFORM": "osmesa"}
MUTANTS = ("reference", "nominal_rack", "calib_wrist", "calib_fixed", "card_wrist", "card_fixed", "loose_fit",
           "lid_let_go", "robust")
EXTRA = ("tilt_plus", "tilt_minus")          # run when named
TILT_DEG = float(os.environ.get("LAB_TILT_DEG", "1.0"))
RESIDUAL_MM = 0.3
LOOSE_MM = 0.8
LET_GO_RAD = 0.25
VIEW_M = 0.20


def _episode(args):
    mutant, unit, props, seed, rep = args
    os.environ.update(ENV)
    for p in (ORACLE, TESTS):
        if p not in sys.path:
            sys.path.insert(0, p)
    import numpy as np

    import centrifuge_probe as C
    import episode as EP
    import lab_env as E
    import lab_sim as S
    import worklist_probe as W
    env = E.LabEnv(props, unit, render=False)
    ep = EP.Episode(env, seed)
    ep.reset()
    a = W.reference_args(env.a)
    p, src, cold = env.p, env.src, env.cold
    rng = np.random.default_rng([seed % 2 ** 32, seed >> 32, 97, rep])
    true_hole = C.target_hole

    def shifted(h, d):
        out = dict(h)
        out["top"] = np.asarray(h["top"], float) + d
        return out

    def calib_shift(cam, X, card=False):
        """Where a believed pose puts the point X seen by `cam`, less X. The believed pose is the calibration on
        file (the true pose times the unit's error) or, with `card`, the equipment sheet's nominal pose (the pose
        on file with the unit's offset taken out)."""
        dT = env.calib[cam]
        if card:
            dT = dT @ np.linalg.inv(env.offset[cam])                    # true -> on file -> nominal
        if cam == "wrist":      # a look from VIEW_M straight above: x right = world +y, y down = world -x... a fixed frame
            R = np.array([[0.0, -1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])
            c = np.array([0.0, 0.0, VIEW_M])
            return R @ (dT[:3, :3] @ c + dT[:3, 3] - c)
        T = env.camera_pose(cam) @ np.linalg.inv(env.calib[cam])        # the true pose
        c = T[:3, :3].T @ (np.asarray(X, float) - T[:3, 3])
        return T[:3, :3] @ (dT[:3, :3] @ c + dT[:3, 3] - c)

    seen = {"calib_wrist": ("wrist", False), "calib_fixed": ("left", False), "card_wrist": ("wrist", True),
            "card_fixed": ("left", True)}

    def target(pp, k):
        h = true_hole(pp, k)
        if mutant in seen:
            d = calib_shift(seen[mutant][0], h["top"], seen[mutant][1])
            return shifted(h, np.array([d[0], d[1], 0.0]))
        if mutant in ("tilt_plus", "tilt_minus") and not isinstance(k, dict):
            tang = np.cross([0.0, 0.0, 1.0], h["axis"])
            tang /= np.linalg.norm(tang)
            ang = math.radians(TILT_DEG) * (1 if mutant == "tilt_plus" else -1)
            K = np.array([[0, -tang[2], tang[1]], [tang[2], 0, -tang[0]], [-tang[1], tang[0], 0]])
            Rot = np.eye(3) + math.sin(ang) * K + (1 - math.cos(ang)) * (K @ K)
            out = dict(h)
            out["axis"] = Rot @ np.asarray(h["axis"], float)
            if "R" in h:
                out["R"] = Rot @ np.asarray(h["R"], float)
            return out
        if mutant == "loose_fit" and not isinstance(k, dict):
            perp = np.cross(h["axis"], [0.0, 0.0, 1.0])
            return shifted(h, LOOSE_MM * 1e-3 * perp / np.linalg.norm(perp))
        if mutant == "robust":
            v = rng.normal(size=3)
            v -= np.dot(v, h["axis"]) * h["axis"]
            return shifted(h, RESIDUAL_MM * 1e-3 * v / np.linalg.norm(v))
        return h

    if mutant in seen or mutant in ("loose_fit", "robust", "tilt_plus", "tilt_minus"):
        C.target_hole = target
    if mutant in seen:
        cam, card = seen[mutant]
        src = [None if s is None else shifted(s, calib_shift(cam, s["top"], card) * [1, 1, 0]) for s in src]
    if mutant == "nominal_rack":
        slots, ncold = W.nominal_layout(a)
        src = [None if s is None else shifted(s, np.array([slots[i][0], slots[i][1], s["top"][2]]) - s["top"])
               for i, s in enumerate(src)]
        cold = {n: shifted(h, np.array([ncold[n]["top"][0], ncold[n]["top"][1], h["top"][2]]) - h["top"])
                for n, h in cold.items()}
    if mutant == "lid_let_go":
        follow = C.follow_lid

        def follow_lid(pp, phi_to, aa, steps):
            return follow(pp, max(phi_to, LET_GO_RAD) if phi_to < 0.5 else phi_to, aa, steps)
        C.follow_lid = follow_lid
    try:
        out = W.run_worklist(p, a, env.episode, src, env.names, cold, env.masses)
        status = out.get("status")
    except Exception as exc:  # noqa: BLE001 - a mutant may drive the reference somewhere it cannot plan from
        status = "error: %s" % type(exc).__name__
    for _ in range(40):                                   # a lid let go gets the time to fall
        if ep.done:
            break
        ep.step(EP.hold_action())
    sc = ep.score()
    return {"mutant": mutant, "unit": unit, "seed": seed, "rep": rep, "credit": sc["credit"], "status": status,
            "steps": int(env.t), "run": sc["run"], "lid": bool(sc["lid"]), "startable": bool(sc["startable"]),
            "pairs": int(sum(sc["pairs"])), "need": sc["need"], "moved": sc["moved"],
            "lid_angle": round(float(p.lid_angle()), 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--mutants", default=",".join(MUTANTS))
    ap.add_argument("--episodes", default=os.path.join(TESTS, "hidden", "episodes.json"))
    ap.add_argument("--units", default=os.path.join(TESTS, "units.json"))
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--repeat", type=int, default=1)
    a = ap.parse_args()
    with open(a.units) as f:
        units = json.load(f)["units"]
    with open(a.episodes) as f:
        d = json.load(f)
    eps = [(str(e["unit"]), int(e["seed"])) for e in (d["episodes"] if isinstance(d, dict) else
                                                    [{"unit": u, "seed": s} for u, s in d])]
    for m in a.mutants.split(","):
        if m not in MUTANTS + EXTRA:
            raise SystemExit("no mutant %r" % m)
    jobs = [(m, u, units[u], s, r) for m in a.mutants.split(",") for (u, s) in eps
            for r in range(a.repeat if m == "robust" else 1)]
    with multiprocessing.get_context("spawn").Pool(a.workers, maxtasksperchild=1) as pool:
        res = pool.map(_episode, jobs, chunksize=1)
    table = {}
    for m in a.mutants.split(","):
        mine = [r for r in res if r["mutant"] == m]
        table[m] = {"mean": round(sum(r["credit"] for r in mine) / len(mine), 4),
                    "at_1": sum(r["credit"] == 1.0 for r in mine), "n": len(mine),
                    "by_unit": {u: round(sum(r["credit"] for r in mine if r["unit"] == u)
                                         / max(1, sum(r["unit"] == u for r in mine)), 3)
                                for u in sorted({r["unit"] for r in mine})}}
        print("%-13s mean %.3f  at 1.0: %d of %d  %s" % (m, table[m]["mean"], table[m]["at_1"], table[m]["n"],
                                                       table[m]["by_unit"]), flush=True)
    with open(a.out, "w") as f:
        json.dump({"table": table, "episodes": res}, f, indent=1)


if __name__ == "__main__":
    main()
