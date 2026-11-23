"""Shared helpers of the review tools (a maintainer's tooling; the grader never runs it).

* paths of the task folder (this file lives in ``tests/review``);
* ``load_cfg()``: the grader's cell configuration;
* ``LocalCell``: a program-side proxy that talks to an in-process ``ParcelCellServer``
  through ``handle()`` (the same calls, replies and errors as the wire client), so a
  program written for the wire (``solve(cell)``) runs in process;
* ``run_program_in_process()``: one shift of a program, the way the grader's
  launcher ends it (``finish`` after ``solve`` returns or raises);
* ``review_shifts()``: the sampled shifts the review tools use (labelled seeds under
  the label "review", disjoint from the hidden and development candidates);
* ``ik_tcp()``: position-only damped least squares IK on the cell's own model
  (precomputing the degenerate programs' fixed targets).
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
TESTS = os.path.dirname(HERE)
TASK = os.path.dirname(TESTS)
PAYLOAD = os.path.join(TASK, "environment", "setup", "payload")
GIVEN = os.path.join(PAYLOAD, "given")
GRADER_CONFIG = os.path.join(TESTS, "grader_config.json")
REFERENCE_DIR = os.path.join(TESTS, "reference")

if TESTS not in sys.path:
    sys.path.insert(0, TESTS)
sys.dont_write_bytecode = True
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

REVIEW_LABEL = "review"


def load_cfg(path=None, over=None):
    from parcelcell.config import deep_update, load_config
    cfg = load_config(path or GRADER_CONFIG)
    if over:
        deep_update(cfg, json.loads(json.dumps(over)))
    return cfg


def raw_grader_config(path=None):
    with open(path or GRADER_CONFIG) as fh:
        return json.load(fh)


# ------------------------------------------------------------------------------------------ in-process cell
class LocalCell(object):
    """The client's ``Cell`` API on an in-process server (``server.handle``)."""

    def __init__(self, server):
        from parcelcell.client import CellError
        self._srv = server
        self._err = CellError
        self.calls = 0

    def _call(self, method, **params):
        self.calls += 1
        reply = self._srv.handle(method, params)
        if isinstance(reply, dict) and "error" in reply:
            raise self._err(reply["error"], reply.get("detail"))
        return json.loads(json.dumps(reply))

    def describe(self):
        return self._call("describe")

    def get_state(self):
        return self._call("get_state")

    def get_camera(self, name, depth=False):
        return self._call("get_camera", name=str(name), depth=bool(depth))

    def step(self, q_target=None, grip=None, n=1, tool_pose=None):
        params = {"n": int(n)}
        if isinstance(q_target, dict):
            params["q_target"] = {str(k): (None if v is None else [float(x) for x in v]) for k, v in q_target.items()}
        elif q_target is not None:
            params["q_target"] = [float(x) for x in q_target]
        if grip is not None:
            params["grip"] = [float(x) for x in grip]
        if tool_pose is not None:
            params["tool_pose"] = {str(k): (None if p is None else {"pos": [float(x) for x in p["pos"]],
                                                                   "quat": [float(x) for x in p["quat"]]})
                                   for k, p in tool_pose.items()}
        return self._call("step", **params)

    def finish(self):
        return self._call("finish")

    def get_state_privileged(self):
        return self._call("get_state_privileged")

    def reference_run(self):
        return self._call("reference_run")

    def close(self):
        return None


def load_program(path, name=None):
    """Import a program file (``solve(cell)``) under its own module name."""
    name = name or "review_prog_" + os.path.splitext(os.path.basename(path))[0]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


TOOL_VISUALS = os.environ.get("REVIEW_PARCEL_VISUALS", "plain")
"""The parcels' looks in the review tooling: "plain" (fast; the default) or "kit" (the graded look, about 3 s a
parcel at a shift's start). Physics and score do not depend on it."""


def make_server(cfg=None, privileged=False, render=True, reference_dir=None, look=None, visuals=None):
    from parcelcell.server import ParcelCellServer
    visuals = TOOL_VISUALS if visuals is None else visuals
    return ParcelCellServer(cfg if cfg is not None else load_cfg(), mode="hidden", privileged=privileged,
                            render=render, reference_dir=reference_dir, look=look, visuals=visuals)


def run_program_in_process(program_path, package, cfg=None, privileged=False, render=True, control=None):
    """One shift of a program in process. ``package``: a shift package (dict or
    path). The shift ends as the grader's launcher ends it: ``finish`` after
    ``solve`` returns or raises. ``control``: an optional callable
    ``control(server)`` used INSTEAD of a program (grader controls that write
    state; never a submission). Returns (record, info)."""
    srv = make_server(cfg, privileged=privileged, render=render,
                      reference_dir=REFERENCE_DIR if privileged else None)
    t0 = time.time()
    srv.reset(package)
    srv.wall_start = time.time()
    err = None
    try:
        if control is not None:
            control(srv)
        else:
            mod = load_program(program_path)
            mod.solve(LocalCell(srv))
    except Exception:  # noqa: BLE001 - a program error ends its shift, as on the wire
        err = traceback.format_exc(limit=4)
    finally:
        try:
            srv.handle("finish", {})
        except Exception:  # noqa: BLE001
            pass
    srv.wall_used = time.time() - srv.wall_start
    rec = srv.finalize()
    info = {"wall_s": round(time.time() - t0, 2), "program_error": err,
            "state_digest": srv.world.state_digest()}
    srv.close()
    return rec, info


# ------------------------------------------------------------------------------------------ sampled shifts
LINE_REVIEW = (("slow", False), ("medium", True), ("fast", False))


def line_review_shifts(cfg=None, n=3, label="review-line"):
    """``n`` line shifts under ``shiftgen.line_seed(label, k)``, k = 0, 1, ...: slow running, medium indexing, fast
    running, then again. Returns [(k, seed, package dict)]."""
    from parcelcell.shiftgen import draw_line_shift, line_seed, to_package
    cfg = cfg or load_cfg()
    out = []
    for k in range(n):
        bc, idx = LINE_REVIEW[k % len(LINE_REVIEW)]
        seed = line_seed(label, k, cfg)
        sh, meta = draw_line_shift(seed, cfg, belt_class=bc, indexing=idx, label=label, k=k)
        out.append((k, seed, to_package(sh, meta)))
    return out


def review_shifts(cfg=None, n=3, label=REVIEW_LABEL):
    """``n`` full shifts under labelled seeds ``label``/k, k = 0, 1, ..., the
    first of each belt-speed class in turn (slow, medium, fast, slow, ...).
    Returns [(k, seed, package dict)]."""
    from parcelcell.shiftgen import draw_shift, labelled_seed, to_package
    cfg = cfg or load_cfg()
    classes = [c for c, _ in sorted(cfg["shifts"]["belt_speed_classes"].items(), key=lambda kv: kv[1][0])]
    want = [classes[i % len(classes)] for i in range(n)]
    out, k = [], 0
    while want:
        seed = labelled_seed(label, k)
        sh, meta = draw_shift(seed, cfg, label=label, k=k)
        if meta["belt_class"] == want[0]:
            out.append((k, seed, to_package(sh, meta)))
            want.pop(0)
        k += 1
        if k > 10000:
            raise RuntimeError("no shift of class %s" % want[0])
    return out


# ------------------------------------------------------------------------------------------ IK (tooling)
def ik_tcp(world, side, target, q0, iters=300, tol=1e-4, R_target=None, w_rot=0.3):
    """Damped least squares IK of one arm's TCP site on the cell's own model
    (position, optionally orientation). Returns (q14, position error)."""
    import mujoco
    import numpy as np
    m = world.model
    d2 = mujoco.MjData(m)
    d2.qpos[:] = world.data.qpos
    idx = np.arange(7) if side == "Left" else np.arange(7, 14)
    q = np.asarray(q0, float).copy()
    sid = world._tcp_sid[side]
    jp = np.zeros((3, m.nv))
    jr = np.zeros((3, m.nv))
    err = None
    for _ in range(iters):
        d2.qpos[world._arm_qadr] = q
        mujoco.mj_kinematics(m, d2)
        mujoco.mj_comPos(m, d2)
        e = np.asarray(target, float) - d2.site_xpos[sid]
        J = None
        if R_target is not None:
            R = d2.site_xmat[sid].reshape(3, 3)
            Re = np.asarray(R_target) @ R.T
            ang = 0.5 * np.array([Re[2, 1] - Re[1, 2], Re[0, 2] - Re[2, 0], Re[1, 0] - Re[0, 1]])
            e6 = np.r_[e, w_rot * ang]
        mujoco.mj_jacSite(m, d2, jp, jr, sid)
        if R_target is None:
            J = jp[:, world._arm_dadr[idx]]
            ee = e
        else:
            J = np.vstack([jp[:, world._arm_dadr[idx]], w_rot * jr[:, world._arm_dadr[idx]]])
            ee = e6
        err = float(np.linalg.norm(e))
        if err < tol and (R_target is None or np.linalg.norm(ee[3:]) < 1e-3):
            break
        dq = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(J.shape[0]), ee)
        q[idx] = np.clip(q[idx] + dq, world._ctrl_lo[idx], world._ctrl_hi[idx])
    return q, err


def dump_json(path, obj):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=1, sort_keys=False, default=float)
        fh.write("\n")
