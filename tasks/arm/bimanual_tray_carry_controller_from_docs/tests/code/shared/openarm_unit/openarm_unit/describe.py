import hashlib
import importlib.util
import json
import os
import xml.etree.ElementTree as ET
from dataclasses import replace
from typing import NamedTuple

import numpy as np

from . import base
from . import frommodel as FM
from . import hs as hsmod
from . import unitfile
from .state import GRIPPER_MODULES, N_MODULES, SessionState, UnitState, public_params
from .unit import OpenArmUnit, plant_edits

UNIT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONTRACT_PATH = os.path.abspath(os.path.join(UNIT_DIR, "..", "..", "tasks",
                                             "bimanual_robot_description_from_commissioning_logs", "contract",
                                             "constants.py"))
DESCRIPTION_KIND = "openarm_unit/Description/1"
_CONTRACTS = {}


class DescribeError(ValueError):
    ""


class Described(NamedTuple):
    robot_xml: bytes
    calibration_json: bytes
    gripper_zero: tuple

    def files(self):
        c = load_contract()
        return dict(zip(c.REQUIRED_FILES, (self.robot_xml, self.calibration_json)))


def load_contract(path=None):
    p = os.path.abspath(os.fspath(path or CONTRACT_PATH))
    if p in _CONTRACTS:
        return _CONTRACTS[p]
    if not os.path.isfile(p):
        raise DescribeError(f"no contract constants file at {os.path.basename(p)}")
    spec = importlib.util.spec_from_file_location("openarm_description_contract", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _check_contract(mod)
    _CONTRACTS[p] = mod
    return mod


def _check_contract(c):
    bad = []
    if tuple(c.SIDES) != tuple(base.SIDES):
        bad.append("SIDES")
    if tuple(c.ARM_JOINT_NAMES) != tuple(base.ARM_JOINTS):
        bad.append("ARM_JOINT_NAMES")
    if tuple(c.JOINT_NAMES) != tuple(base.HINGES) and set(c.JOINT_NAMES) != set(base.HINGES):
        bad.append("JOINT_NAMES")
    if tuple(tuple(x) for x in c.ACTUATORS) != tuple(tuple(x) for x in base.ACTUATORS):
        bad.append("ACTUATORS")
    if any(c.TOOL_SITES[s][1] != base.EE_BODIES[s] for s in base.SIDES):
        bad.append("TOOL_SITES bodies")
    if tuple(float(x) for x in c.TOOL_OFFSET_M) != tuple(float(x) for x in base.TOOL_OFFSET):
        bad.append("TOOL_OFFSET_M")
    pub = tuple(float(x) for x in public_params()["conventions"]["tool_offset_m"]["value"])
    if pub != tuple(float(x) for x in c.TOOL_OFFSET_M):
        bad.append("TOOL_OFFSET_M against params_public.json")
    if c.TOOL_AXES_IN_EE_BASE_LINK != "identity":
        bad.append("TOOL_AXES_IN_EE_BASE_LINK")
    if len(c.REQUIRED_FILES) != 2:
        bad.append("REQUIRED_FILES")
    if bad:
        raise DescribeError(f"the contract's constants differ from the unit's names: {bad}")


def _state_session_zeros(unit, session):
    if isinstance(unit, OpenArmUnit):
        if unit.state is None:
            raise DescribeError("a model run has no unit state to describe (describe a built unit or a UnitState)")
        st = unit.state
        ses = unit.session if session is None else session
        delta = [float(x) for x in unit.bus.delta]
        sign = [int(x) for x in unit.bus.sign]
    elif isinstance(unit, UnitState):
        if session is None:
            raise DescribeError("describe(state, session): a UnitState needs its named session")
        st, ses = unit, session
        delta = [float(x) for x in st.cal.delta]
        sign = [int(x) for x in st.cal.sign]
    else:
        raise DescribeError("describe takes an OpenArmUnit built from a state, or a UnitState")
    if not isinstance(ses, SessionState):
        raise DescribeError("the session is a SessionState")
    if len(delta) != N_MODULES or not all(np.isfinite(delta)):
        raise DescribeError("sixteen finite zeros")
    return st, ses, delta, sign


def robot_xml_text(state, session, contract=None):
    c = load_contract(contract)
    text = unitfile.unit_xml_text(plant_edits(state, session), contacts=True)
    root = ET.fromstring(text)
    bodies = {b.attrib.get("name"): b for b in root.iter("body")}
    for side in base.SIDES:
        name, body = c.TOOL_SITES[side]
        if any(s.attrib.get("name") == name for s in root.iter("site")):
            raise DescribeError(f"the unit file already holds a site {name}")
        ET.SubElement(bodies[body], "site", {"name": name, "pos": unitfile.fmt_vec(c.TOOL_OFFSET_M),
                                             "quat": unitfile.fmt_vec((1.0, 0.0, 0.0, 0.0)), "size": "0.005"})
    ET.indent(root, space="  ")
    return ET.tostring(root, encoding="unicode") + "\n"


def calibration_dict(delta, sign, contract=None):
    c = load_contract(contract)
    mod = {j: i for i, (_a, j) in enumerate(base.ACTUATORS)}
    joints = {}
    for j in c.ARM_JOINT_NAMES:
        i = mod[j]
        z = float(delta[i])
        if not abs(z) <= c.CALIBRATION_BOUNDS["zero_offset_rad_abs_max"]:
            raise DescribeError(f"{j}: a zero of {z} rad lies outside the contract's bound")
        if int(sign[i]) not in c.CALIBRATION_BOUNDS["sign"]:
            raise DescribeError(f"{j}: sign {sign[i]}")
        joints[j] = {"sign": int(sign[i]), "zero_offset_rad": z}
    return {"format": c.CALIBRATION_FORMAT, "joints": joints}


def describe(unit, session=None, *, contract=None):
    st, ses, delta, sign = _state_session_zeros(unit, session)
    xml = robot_xml_text(st, ses, contract).encode("utf-8")
    cal = (json.dumps(calibration_dict(delta, sign, contract), indent=1, sort_keys=True) + "\n").encode("ascii")
    gz = tuple(float(delta[i]) for i in GRIPPER_MODULES)
    return Described(xml, cal, gz)


def write_description(out_dir, described, contract=None):
    c = load_contract(contract)
    os.makedirs(out_dir, exist_ok=True)
    out, sha = {}, {}
    for name, b in zip(c.REQUIRED_FILES, (described.robot_xml, described.calibration_json)):
        p = os.path.join(out_dir, name)
        with open(p, "wb") as f:
            f.write(b)
        out[name], sha[name] = p, hashlib.sha256(b).hexdigest()
    out["gripper_zero"] = list(described.gripper_zero)
    out["sha256"] = sha
    return out


def run_description(described, state, *, registers=None, contacts=False, world=None, truth_profile=(), **kw):
    return FM.from_model(described.robot_xml, described.calibration_json,
                         registers=state.cal.registers if registers is None else registers, world=world,
                         contacts=contacts, editions=dict(state.editions), imax=list(state.cal.imax),
                         truth_profile=truth_profile, gripper_zero=described.gripper_zero, **kw)


STRUCTS = {
    "opt": ("timestep", "impratio", "tolerance", "ls_tolerance", "noslip_tolerance", "ccd_tolerance", "gravity",
            "wind", "magnetic", "density", "viscosity", "o_margin", "o_solref", "o_solimp", "o_friction",
            "integrator", "cone", "jacobian", "solver", "iterations", "ls_iterations", "noslip_iterations",
            "ccd_iterations", "disableflags", "enableflags", "disableactuator", "sdf_initpoints", "sdf_iterations"),
    "stat": ("meaninertia", "meanmass", "meansize", "extent", "center"),
}


def model_arrays(m):
    out = {}
    for k in dir(m):
        if k.startswith("_"):
            continue
        v = getattr(m, k, None)
        if isinstance(v, np.ndarray):
            out[k] = v
    return out


def model_differences(a, b):
    A, Bm = model_arrays(a), model_arrays(b)
    bad = sorted(set(A) ^ set(Bm))
    for k in sorted(set(A) & set(Bm)):
        x, y = A[k], Bm[k]
        if x.shape != y.shape or x.dtype != y.dtype or x.tobytes() != y.tobytes():
            bad.append(k)
    n = len(set(A) & set(Bm))
    for s, fields in STRUCTS.items():
        for f in fields:
            if not hasattr(getattr(a, s), f):
                continue
            x, y = np.asarray(getattr(getattr(a, s), f)), np.asarray(getattr(getattr(b, s), f))
            n += 1
            if x.shape != y.shape or x.tobytes() != y.tobytes():
                bad.append(f"{s}.{f}")
    return bad, n


def build_path_model(state, session, contacts=True):
    return base.unit_spec(plant_edits(state, session), contacts=contacts).compile()


def own_description_file(state, session, out_dir, contacts=True):
    zeros = {j: float(state.cal.delta[i]) for i, (_a, j) in enumerate(base.ACTUATORS) if j in base.ARM_JOINTS}
    r = unitfile.write_unit(out_dir, plant_edits(state, session), zero_offsets=zeros, contacts=contacts)
    written = unitfile.compile_unit_xml(r["unit_xml"])
    bad, n = model_differences(build_path_model(state, session, contacts), written)
    return {"differences": bad, "arrays": n, "unit_xml_sha256": r["unit_xml_sha256"],
            "unit_cal_sha256": r["unit_cal_sha256"]}


def without_profile(state):
    return replace(state, hs=hsmod.off_state(float(state.hs.period[0])))


def scripted_targets(n_move=250, n_hold=250, amp=0.25):
    q0 = np.array(OpenArmUnit.default_pose(), dtype=np.float64)
    scale = np.array([0.0 if i in GRIPPER_MODULES else (1.0 if i % 8 < 4 else 0.6) * (1.0 if i < 8 else -1.0)
                      for i in range(N_MODULES)])
    out = []
    for k in range(n_move + n_hold):
        s = min(1.0, (k + 1) / float(n_move))
        ramp = 0.5 - 0.5 * np.cos(np.pi * s)
        out.append(q0 + amp * ramp * scale)
    return out


def run_scripted(u, hold, targets, hold_s=0.2, tool_trace=False, zeros=None):
    u.reset(hold=hold, hold_s=hold_s)
    u.truth_trace()
    sg, dl = (u.bus.sign, u.bus.delta) if zeros is None else zeros
    sign, delta = np.asarray(sg, dtype=np.float64), np.asarray(dl, dtype=np.float64)
    md, sd, tp = [], [], []
    for q in targets:
        u.exchange(u.hold_frames(hold, sign * (np.asarray(q, dtype=np.float64) - delta)), steps=4)
        md.append(u.motion_digest())
        sd.append(u.state_digest())
        if tool_trace:
            tp.append(np.asarray(u.truth(("tool_pos",))["tool_pos"], dtype=np.float64))
    tr = u.truth_trace()
    if not tool_trace:
        tp = np.asarray(u.truth(("tool_pos",))["tool_pos"], dtype=np.float64)
    return {"q": np.array([r["q"] for r in tr], dtype=np.float64), "motion": md, "state": sd,
            "tool_pos": np.asarray(tp, dtype=np.float64)}


ARM_HINGE_ROWS = tuple(k for k, n in enumerate(base.HINGES) if n in base.ARM_JOINTS)


def compare_runs(a, b):
    qa, qb = a["q"], b["q"]
    arm = list(ARM_HINGE_ROWS)
    return {"steps": int(qa.shape[0]), "exchanges": len(a["motion"]),
            "max_abs_dq_rad": float(np.max(np.abs(qa - qb))),
            "max_abs_dq_arm_rad": float(np.max(np.abs(qa[:, arm] - qb[:, arm]))),
            "travel_rad": float(np.max(np.abs(qa[:, arm] - qa[0, arm]))),
            "tool_pos_max_diff_m": float(np.max(np.sqrt(np.sum((a["tool_pos"] - b["tool_pos"]) ** 2, axis=-1)))),
            "motion_digests_equal": int(sum(x == y for x, y in zip(a["motion"], b["motion"]))),
            "state_digests_equal": int(sum(x == y for x, y in zip(a["state"], b["state"])))}


def own_description_motion(state, session, hold, *, contacts=False, n_move=250, n_hold=250, contract=None, tool_trace=False):
    st = without_profile(state)
    d = describe(st, session, contract=contract)
    u = OpenArmUnit.build(st, session, contacts=contacts, truth_profile=("q",))
    m = run_description(d, st, contacts=contacts, truth_profile=("q",))
    tg = scripted_targets(n_move, n_hold)
    z = (u.bus.sign.copy(), u.bus.delta.copy())
    ra = run_scripted(u, hold, tg, tool_trace=tool_trace, zeros=z)
    rb = run_scripted(m, hold, tg, tool_trace=tool_trace, zeros=z)
    out = compare_runs(ra, rb)
    out.update({"final_motion_digest": ra["motion"][-1], "gripper_zero": list(d.gripper_zero),
                "robot_xml_sha256": hashlib.sha256(d.robot_xml).hexdigest(),
                "calibration_sha256": hashlib.sha256(d.calibration_json).hexdigest()})
    return out


__all__ = ["describe", "Described", "DescribeError", "load_contract", "write_description", "run_description",
           "robot_xml_text", "calibration_dict", "model_differences", "build_path_model", "own_description_file", "own_description_motion",
           "without_profile", "scripted_targets", "run_scripted", "compare_runs", "CONTRACT_PATH"]
