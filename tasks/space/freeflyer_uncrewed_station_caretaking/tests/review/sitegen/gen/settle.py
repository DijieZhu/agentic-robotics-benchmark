"""The two robot-free runs of the generator (the initial-state rules 1-3 of the sites; site_spec
``world.settle``) and the settled initial state ``state0`` (the site package format).

1. ``pre_settle``: the site without the robot (the dock's collision geometry kept) is wired robot-free
   (``cabincell.site.build.rig``: suction and item aero at 50 Hz, the cabin and unit air, the station frame) and run
   ``pre_s`` (600 s) so that the items come to rest.  The settled item state is copied by joint name into the cell's
   model of the site (with the robot on its berth, ``station.sites.initial_data``): that is ``state0`` (time 0).
2. ``dry_run``: a fresh robot-free model and rig start from state0 (the items' joints copied back by name) and run
   ``dry_run_s`` (2400 s) with every item sampled each second; worst over the samples, relative to the start:
   the sheet's sealed area (the scene's suction sampler) must change by less than sheet_area_change_max_pct and no
   body of it move more than sheet_disp_max_mm; a pouch must not move more than pouch_disp_max_mm.

state0 = {"schema": "caretaking.state0/1", "nq", "nv", "qpos", "qvel", "time", "model_signature": {"nbody", "njnt",
"names_sha256"}} with the floats as Python floats (``json`` writes their repr: an exact round trip); names_sha256 =
sha256 of the body names then the joint names of the model, one per line, prefixed 'body:' / 'joint:'.
``load_state0`` is the reader (the cell).

The check: settle, then an idle run sampled every second that measures the displacement and the sealed-area change
from the end of the settle, on the site's own scene and rig, against per-kind criteria (1 % / 3 mm sheet, 1 mm
pouch); the dry run starts from the stored state in a fresh rig (what the cell will start from), state0.
"""
from __future__ import annotations

import copy
import hashlib
import time
import warnings

import mujoco
import numpy as np

from cabincell.site import build as BU
from cabincell.site import spec as SPEC

STATE0_SCHEMA = "caretaking.state0/1"
ITEM_PREFIX = "fx_"


def _build(site: dict, robot: bool):
    S = copy.deepcopy(site)
    S["robot"] = dict(S.get("robot") or {}, present=bool(robot))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return BU.build(S)


def model_signature(model) -> dict:
    names = ["body:" + (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or "") for b in range(model.nbody)]
    names += ["joint:" + (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or "") for j in range(model.njnt)]
    return dict(nbody=int(model.nbody), njnt=int(model.njnt), names_sha256=hashlib.sha256("\n".join(names).encode("utf-8")).hexdigest())


def item_joints(model) -> list:
    """Names of the loose items' joints (the station builder's prefix 'fx_')."""
    out = []
    for j in range(model.njnt):
        n = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or ""
        if n.startswith(ITEM_PREFIX):
            out.append(n)
    return out


_NQ = {int(mujoco.mjtJoint.mjJNT_FREE): 7, int(mujoco.mjtJoint.mjJNT_BALL): 4, int(mujoco.mjtJoint.mjJNT_SLIDE): 1,
       int(mujoco.mjtJoint.mjJNT_HINGE): 1}
_NV = {int(mujoco.mjtJoint.mjJNT_FREE): 6, int(mujoco.mjtJoint.mjJNT_BALL): 3, int(mujoco.mjtJoint.mjJNT_SLIDE): 1,
       int(mujoco.mjtJoint.mjJNT_HINGE): 1}


def _slices(model, name):
    j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if j < 0:
        raise KeyError(name)
    t = int(model.jnt_type[j])
    qa, va = int(model.jnt_qposadr[j]), int(model.jnt_dofadr[j])
    return slice(qa, qa + _NQ[t]), slice(va, va + _NV[t])


def copy_joints(src_model, src_data, dst_model, dst_data, names) -> None:
    """Copy qpos / qvel of the named joints between two models of the same site."""
    for n in names:
        sq, sv = _slices(src_model, n)
        dq, dv = _slices(dst_model, n)
        dst_data.qpos[dq] = src_data.qpos[sq]
        dst_data.qvel[dv] = src_data.qvel[sv]


def state0_of(model, data_) -> dict:
    return {"schema": STATE0_SCHEMA, "nq": int(model.nq), "nv": int(model.nv), "qpos": [float(x) for x in data_.qpos],
            "qvel": [float(x) for x in data_.qvel], "time": 0.0, "model_signature": model_signature(model)}


def load_state0(model, data_, st0: dict) -> None:
    """Put a state0 into a model's data (raises when the model is not the one it was written for)."""
    if st0.get("schema") != STATE0_SCHEMA:
        raise ValueError(f"not a {STATE0_SCHEMA}")
    if int(st0["nq"]) != model.nq or int(st0["nv"]) != model.nv or st0["model_signature"] != model_signature(model):
        raise ValueError("state0 was written for another model (nq / nv / names differ)")
    data_.qpos[:] = np.asarray(st0["qpos"], float)
    data_.qvel[:] = np.asarray(st0["qvel"], float)
    data_.time = float(st0["time"])
    mujoco.mj_forward(model, data_)


def _measures_summary(meas: dict) -> dict:
    return {k: dict(inlet=v["inlet"], sealed_m2=v["sealed_m2"], hold_N=v["hold_N"], Pc_Pa=v["Pc_Pa"], geometric_m2=v["geometric_m2"])
            for k, v in meas.items()}


def pre_settle(site: dict, pre_s: float | None = None, sample_s: float = 10.0) -> dict:
    """Run 1 (see the module doc): dict(state0, measures (in-scene, after the settle), settle_disp (from t = 0),
    cpu_s, sim_s, state_sha256)."""
    v = SPEC.values()
    pre_s = float(v["world.settle"]["pre_s"]) if pre_s is None else float(pre_s)
    c0 = time.process_time()
    m, info = _build(site, robot=False)
    d = BU.initial_data(m, info)
    R = BU.rig(m, d, info)
    r = BU.settle(R, pre_s, sample_s)
    meas = BU.measure(R)
    names = item_joints(m)
    m2, info2 = _build(site, robot=True)
    d2 = BU.initial_data(m2, info2)
    copy_joints(m, d, m2, d2, names)
    mujoco.mj_forward(m2, d2)
    st0 = state0_of(m2, d2)
    return dict(state0=st0, measures=_measures_summary(meas), settle_disp=r["worst"], item_joints=names,
                cpu_s=time.process_time() - c0, sim_s=float(d.time),
                state_sha256=hashlib.sha256(np.ascontiguousarray(d.qpos).tobytes() + np.ascontiguousarray(d.qvel).tobytes()).hexdigest())


def robot_free_from_state0(site: dict, st0: dict):
    """(model, data, info) of the robot-free scene with the items of state0 (the robot model reads state0 first)."""
    m2, info2 = _build(site, robot=True)
    d2 = mujoco.MjData(m2)
    load_state0(m2, d2, st0)
    m, info = _build(site, robot=False)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    copy_joints(m2, d2, m, d, item_joints(m))
    d.time = 0.0
    mujoco.mj_forward(m, d)
    return m, d, info


def criteria(values=None) -> dict:
    v = values or SPEC.values()
    W = v["world.settle"]
    return dict(sheet_area_change_max=float(W["sheet_area_change_max_pct"]) / 100.0, sheet_disp_max_m=float(W["sheet_disp_max_mm"]) / 1000.0,
                pouch_disp_max_m=float(W["pouch_disp_max_mm"]) / 1000.0)


def judge(worst: dict, values=None) -> dict:
    """Per item pass / fail of a dry run's worst values (module doc), and the site's ``settle`` numbers."""
    C = criteria(values)
    items, ok = {}, True
    for k, w in worst.items():
        if k == "blocker":
            p = bool(w["sealed_change_rel"] < C["sheet_area_change_max"] and w["disp_m"] <= C["sheet_disp_max_m"])
        else:
            p = bool(w["disp_m"] <= C["pouch_disp_max_m"])
        items[k] = dict(disp_m=float(w["disp_m"]), sealed_change_rel=float(w["sealed_change_rel"]), passed=p)
        ok = ok and p
    return dict(passed=ok, items=items)


def dry_run(site: dict, st0: dict, dry_run_s: float | None = None, sample_s: float = 1.0) -> dict:
    """Run 2 (see the module doc): dict(passed, items, end measures, cpu_s, sim_s, final_state_sha256)."""
    v = SPEC.values()
    T = float(v["world.settle"]["dry_run_s"]) if dry_run_s is None else float(dry_run_s)
    c0 = time.process_time()
    m, d, info = robot_free_from_state0(site, st0)
    R = BU.rig(m, d, info)
    r = BU.settle(R, T, sample_s)
    J = judge(r["worst"], v)
    return dict(J, start=_measures_summary(r["start"]), end=_measures_summary(r["end"]), cpu_s=time.process_time() - c0,
                sim_s=float(d.time), dry_run_s=T, final_state_sha256=r["timing"]["final_state_sha256"])


def settle_record(pre: dict, dry: dict, state0_sha256: str, values=None) -> dict:
    """station_site/2 ``settle``."""
    v = values or SPEC.values()
    W = v["world.settle"]
    it = dry["items"]
    b = it.get("blocker") or {}
    return dict(pre_s=float(W["pre_s"]) if pre.get("sim_s") is None else float(round(pre["sim_s"], 6)),
                dry_run_s=float(round(dry["dry_run_s"], 6)),
                sheet_area_change_pct=(None if not b else float(round(100.0 * b["sealed_change_rel"], 6))),
                sheet_disp_mm=(None if not b else float(round(1000.0 * b["disp_m"], 6))),
                pouch_disp_mm={k: float(round(1000.0 * x["disp_m"], 6)) for k, x in sorted(it.items()) if k != "blocker"},
                passed=bool(dry["passed"]), state0_sha256=state0_sha256)
