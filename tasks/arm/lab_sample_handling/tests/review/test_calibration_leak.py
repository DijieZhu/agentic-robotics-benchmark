"""The calibration error is not in the numbers (a maintainer's check; the grader never runs it).

Until 2026-11-24 the fixed cameras stood exactly at the equipment sheet's poses, so that inv(T_sheet) @ T_world_<cam>
was the unit's calibration error to the last digit (a reviewer's finding; the wrist camera likewise through the
robot model's own camera). Since then a unit's pose on file is the sheet's nominal pose times an offset drawn from
a key of its own (tests/review/unit_fit.py, rule 4b), and the true camera is the pose on file with the error taken
out. This check builds every unit as the grader and the commissioning service do and asserts, per unit and camera:

  * what a controller can compute, inv(T_sheet) @ T_world_<cam>, is the offset (to 1e-9), which has no share in
    the error;
  * taken as an estimate of the error it is wrong by more than the error's own bound in most components (it is no
    better than not knowing: the offset's range is four times the error's);
  * the camera the images are drawn from is the pose on file with the error taken out (to 1e-9).

    /opt/venv/bin/python3 -B tests/review/test_calibration_leak.py        (the task image's Python)
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TESTS = os.path.dirname(HERE)
TASK = os.path.dirname(TESTS)
sys.dont_write_bytecode = True
sys.path.insert(0, TESTS)
os.environ.setdefault("MUJOCO_GL", "osmesa")
os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")

import numpy as np  # noqa: E402

import lab_env as E  # noqa: E402
import lab_sim as S  # noqa: E402

B = S.B


def sheet(pos, tgt):
    """A fixed camera's nominal pose as the equipment sheet gives it (OpenCV convention, image x horizontal)."""
    T = np.eye(4)
    T[:3, :3] = B.look_at(np.array(pos, float), np.array(tgt, float)) @ np.diag([1.0, -1.0, -1.0])
    T[:3, 3] = pos
    return T


SHEET = {"left": sheet((1.15, 0.55, 1.35), (0.5, 0.0, 0.8)), "right": sheet((1.15, -0.55, 1.35), (0.5, 0.0, 0.8)),
         "rack": sheet((-0.09, 0.3625, 0.9555), (0.36, 0.3625, 0.8755))}


def six(T):
    R = T[:3, :3]
    th = float(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))
    rv = np.zeros(3) if th < 1e-12 else th / (2 * np.sin(th)) * np.array(
        [R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return np.concatenate([np.degrees(rv), T[:3, 3] * 1e3])


def main():
    units = dict(json.load(open(os.path.join(TESTS, "units.json")))["units"])
    units.update(json.load(open(os.path.join(TASK, "environment", "commission_service", "bench.json")))["units"])
    worse, n = 0, 0
    for u, props in sorted(units.items()):
        env = E.LabEnv(props, u, render=False)
        obs = env.reset(1)
        nominal = E.LabEnv({k: v for k, v in props.items() if k not in ("calib", "offset")}, u, render=False)
        nominal.reset(1)                                  # the robot's model as the sheet names it
        for cam in ("left", "right", "rack", "wrist"):
            T_sheet = SHEET[cam] if cam != "wrist" else nominal.camera_pose("wrist")
            T_file = np.asarray(obs["T_world_%s" % cam])
            est = np.linalg.inv(T_sheet) @ T_file
            off, err = S.calib_matrix(props["offset"][cam]), S.calib_matrix(props["calib"][cam])
            assert np.abs(est - off).max() < 1e-9, (u, cam, "the computable product is not the offset")
            R, pc = S.cam_pose(env.p.data, env.p.model.camera(E.CAMS[cam][4]).id)
            T_true = np.eye(4)
            T_true[:3, :3], T_true[:3, 3] = R, pc
            assert np.abs(T_true @ err - T_file).max() < 1e-9, (u, cam, "the drawn camera is not file @ inv(error)")
            mm, deg = S.CALIB["wrist" if cam == "wrist" else "fixed"]
            miss = np.abs(six(est) - np.asarray(props["calib"][cam], float)) / np.array([deg] * 3 + [mm] * 3)
            worse += int(np.sum(miss > 1.0))
            n += 6
    assert worse > n // 2, "the product estimates the error within its bound in %d of %d components" % (n - worse, n)
    print("ok: %d units; the computable product is the offset; as an estimate of the error it is off by more than "
          "the error's bound in %d of %d components" % (len(units), worse, n))


if __name__ == "__main__":
    main()
