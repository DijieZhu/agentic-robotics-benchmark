"""The public runner of a submitted description: `from_model` turns one MJCF file and its calibration into a runnable
model of the OpenArm pair under the joint modules' bus layer. PUBLIC.

PUBLIC: this module ships with a task. It imports numpy, MuJoCo and the public modules `base`, `bus`, `registers`,
`state`, `hooks_api`, `unit`, `policy_mjcf`; nothing of a drawn unit, no thermal model and no reply-noise law. The
public modules of the package that ship together are the package's `__init__`, `base`, `frames`, `state`, `rng`,
`hostcodec`, `registers`, `envelope`, `bus`, `truth`, `hooks_api`, `unit`, `unitfile`, `policy_mjcf`, `frommodel`,
`static` and `refhost` (the reference hosts: ports of the maker's public driver and zero routine), with
`params_public.json` and `tools/run_model.py`; none of them imports anything else of the package.

HOW A DESCRIPTION BECOMES A RUNNABLE MODEL
 1. The check of policy_mjcf.py (byte caps; the text check; a zero-warning compile in MuJoCo 3.14.0 with the mesh file
    base names resolved in the given assets folder; the form on the compiled model; the calibration; the caps). A
    description that fails is refused (PolicyError) with every failure named.
 2. Read from the compiled model: the tree with its body frames and inertials, the mounts, per joint pos, axis, range,
    armature, damping and friction loss, per actuator k = gainprm[0] and b = biasprm[0], the two tool sites; from the
    calibration the sign and zero offset of each arm joint.
 3. Rebuild for running, from the same file: the same tree, inertials and joint values; everything else replaced by
    numerics identical for every model, the maker's own scene included:
    - the option block of the maker's pedestal scene (time step 1 ms, implicitfast, elliptic cone, impratio 10, gravity
      0 0 -9.81 and every other option field as the maker's scene has it), with the Newton solver, 100 iterations,
      tolerance 1e-8 and no no-slip pass; contacts off unless the caller asks for them;
    - on each of the eighteen hinges the friction-loss constraint (solreffriction 0.002 1, solimpfriction 0.9999
      0.9999 0.001 0.5 2), the joint-limit constraint (solreflimit 0.002 1, solimplimit 0.99 0.999 0.001 0.5 2) and a
      limit margin of 0, so that the file's ranges are the joint's two stops with one stated compliance;
    - the maker's two finger equalities (polycoef 0 1 0 0 0, solref 0.005 1, solimp 0.95 0.99 0.001 0.5 2);
    - the sixteen actuators written anew as the modules' carriers (gear 1, joint transmission, gain k, constant bias b,
      no control range and no force range), and no joint-side actuatorfrcrange on any hinge: every force and control
      range of the file is dropped;
    - no keyframe and none reserved: the file's keyframe section is dropped and its size element (memory, solver
      sizes, the keyframes and user data it reserves) is replaced by the maker's scene's, so that the maker's own
      scene written out as one file runs as the maker's scene does; no contact pair or exclude of the file.
    After the compile the rebuilt model is compared with what was read (tree, frames, inertials, joint values: bit
    for bit) and with the maker's scene under the same numerics (option block, the hinges' constraint numerics, the
    actuators' fields other than k and b, the finger equalities: bit for bit); any difference is refused.
 4. The bus layer of the sixteen modules on top (bus.py): the documented in-drive law and force-position law, the
    runtime contract of the frames, the module's own registers (mapping ranges, current limit, control mode, time-out)
    from the given register set, its current base, its edition's envelope and peak torque. The module clamps what it
    commands (its current limit, its torque-speed envelope, its peak torque, all in the drive's own units); the joint
    receives k x the clamped command + b, b outside every clamp and outside the reply; b is a torque in the MODEL's
    joint coordinate (a description that reverses a joint's sense against the maker's reverses the sign of b with it).
    Module angle and model coordinate are related by the calibration: theta = sign (q - zero_offset_rad) for the
    fourteen arm modules. The two gripper modules have sign +1 and the zero given by the caller's `gripper_zero`
    (left, right) in rad: theta = q - gripper_zero. The calibration file holds the arm joints only and cannot state
    a gripper's zero; the default is 0 for both grippers. A graded run of a description passes the two gripper zeros
    of the unit it is compared with (the maker's zero routine zeroes the grippers too); tools/run_model.py uses 0.
 5. A model run has no heating and no reply noise: both temperature bytes of every reply read the fixed public value
    (params_public.json model_run), no protection that heat trips exists, and the replies carry the truncated true
    values. There is no reply-noise key: `noise_key` must be None. Its friction is the file's (no friction profile
    along the travel). Its truth is read at the file's two tool sites.
WHERE THE FILE SITS: the file's world frame is the world frame of the maker's pedestal scene (pedestal/pedestal.xml at
commit ed2046f30981398fc610688f08beac0ebd08f972 of github.com/enactic/openarm_mujoco, folder v2/): the two base links
sit on the world body at the poses the file gives them, and nothing is attached or moved by the runner.
CONTACTS AND CLEARANCES: the rebuild keeps the file's own world geoms (a floor, a pedestal) as the file states them.
Contacts and clearances of a model run are therefore meaningful for the maker's own scene and for the robot's own
bodies; where a task reads contacts or clearances against the bench, the pedestal or the floor, it reads them on the
maker's scene or on the robot unit itself, never on a submitted file's world geoms.
The world (a HookSet: rests, a point mass at a tool, ...) is declared into the rebuilt spec before the compile, as for
a unit. The run is an `OpenArmUnit` built by `OpenArmUnit.from_parts`: every call of a unit (exchange, reset,
set_state, truth, events, snapshot, restore, state_digest, motion_digest) works on it alike. So does the check of
every physics step (unit.py, "A STEP THE PLANT CANNOT INTEGRATE"): when MuJoCo's counter of a bad acceleration,
velocity or position, or of a full constraint or contact buffer, rises in a step of a model run, the run records the
event `plant_invalid` (the step and the warning's kind), raises UnitInvalid from that exchange before its replies are
returned and serves nothing more; a description whose run meets such a step has no run past it.
"""
import hashlib
import json
import os

import mujoco
import numpy as np

from . import base
from . import bus as busmod
from . import policy_mjcf as pm
from . import registers as rg
from .hooks_api import HookSet
from .state import ARM_MODULES, GRIPPER_MODULES, N_MODULES, RegisterSet, canonical_json
from .unit import OpenArmUnit, UnitError, UnitInvalid, code_digest

RUNNER_FILES = ("frommodel.py", "policy_mjcf.py", "static.py")
MODEL_KIND = "model"
# MjOption fields copied from the maker's scene (every field MuJoCo 3.14.0's spec option carries)
OPTION_FIELDS = ("ccd_iterations", "ccd_tolerance", "cone", "density", "disableactuator", "disableflags",
                 "enableflags", "gravity", "impratio", "integrator", "iterations", "jacobian", "ls_iterations",
                 "ls_tolerance", "magnetic", "noslip_iterations", "noslip_tolerance", "o_friction", "o_margin",
                 "o_solimp", "o_solref", "sdf_initpoints", "sdf_iterations", "sleep_tolerance", "solver",
                 "timestep", "tolerance", "viscosity", "wind")
# the size element's fields replaced by the maker's scene's (memory and solver sizes, the keyframes and user data it
# reserves): a file's own <size> changes nothing of the run (the maker's own scene reserves no keyframe and no user
# data, so the rebuilt model holds none, whatever the file reserves)
SIZE_FIELDS = ("memory", "njmax", "nconmax", "nstack", "nkey", "nuserdata", "nuser_body", "nuser_jnt", "nuser_geom",
               "nuser_site", "nuser_cam", "nuser_tendon", "nuser_actuator", "nuser_sensor")
# what the rebuild must leave as the file has it (compiled arrays compared bit for bit)
READ_BODY = ("body_parentid", "body_pos", "body_quat", "body_ipos", "body_iquat", "body_mass", "body_inertia")
READ_JOINT = ("jnt_type", "jnt_bodyid", "jnt_pos", "jnt_axis", "jnt_range", "jnt_limited")
READ_DOF = ("dof_armature", "dof_damping", "dof_frictionloss")
# what the rebuild makes identical for every model (compared with the maker's scene under the same numerics)
SAME_JOINT = ("jnt_solref", "jnt_solimp", "jnt_margin", "jnt_actfrclimited", "jnt_actfrcrange", "jnt_actgravcomp",
              "jnt_stiffness", "jnt_stiffnesspoly")
SAME_DOF = ("dof_solref", "dof_solimp", "dof_dampingpoly")
SAME_ACT = ("actuator_trntype", "actuator_dyntype", "actuator_gaintype", "actuator_biastype", "actuator_gear",
            "actuator_ctrllimited", "actuator_forcelimited", "actuator_actlimited", "actuator_ctrlrange",
            "actuator_forcerange", "actuator_actrange", "actuator_dynprm", "actuator_damping", "actuator_dampingpoly",
            "actuator_armature", "actuator_delay", "actuator_actearly", "actuator_actnum", "actuator_cranklength")
SAME_EQ = ("eq_type", "eq_data", "eq_solref", "eq_solimp", "eq_active0")


class ModelError(UnitError):
    """A description the runner cannot rebuild or run as stated (the check passed but the rebuild differs, a reply
    noise key for a model run, an argument of the wrong kind)."""


def runner_digest():
    """sha256 over this runner's files (name, length, bytes), beside the unit layer's own code digest."""
    h = hashlib.sha256()
    here = os.path.dirname(os.path.abspath(__file__))
    for name in RUNNER_FILES:
        with open(os.path.join(here, name), "rb") as f:
            b = f.read()
        h.update(name.encode("ascii") + b"\0" + str(len(b)).encode("ascii") + b"\0" + b)
    return h.hexdigest()


# ------------------------------------------------------------------------------------------------- the description
class Description:
    """A checked description: the compiled file (`model`), the text as compiled (`text`), k and b of the sixteen
    actuators, sign and zero offset of the sixteen modules (grippers +1 and 0), the check's result and the files'
    sha256."""

    def __init__(self, result):
        if not result.passed:
            raise pm.PolicyError(result)
        self.result = result
        self.model = result.model
        self.text = result.text
        m = self.model
        aid = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, a) for a, _ in base.ACTUATORS]
        self.k = tuple(float(m.actuator_gainprm[a, 0]) for a in aid)
        self.b = tuple(float(m.actuator_biasprm[a, 0]) for a in aid)
        sign, zero = [1] * N_MODULES, [0.0] * N_MODULES
        for i in ARM_MODULES:
            s, z = result.calibration[base.ACTUATORS[i][1]]
            sign[i], zero[i] = int(s), float(z)
        self.sign, self.zero = tuple(sign), tuple(zero)
        self.calibration = dict(result.calibration)
        self.xml_sha256 = hashlib.sha256(result.xml_bytes).hexdigest()
        self.cal_sha256 = hashlib.sha256(result.cal_bytes).hexdigest()


def load_description(robot_xml, calibration, assets_dir=None):
    """Check a description given as two paths or as two byte strings; returns a Description (PolicyError if the
    check fails)."""
    def _bytes(x):
        if isinstance(x, (bytes, bytearray)):
            return bytes(x)
        with open(x, "rb") as f:
            return f.read()
    return Description(pm.check(_bytes(robot_xml), _bytes(calibration), assets_dir))


# ------------------------------------------------------------------------------------------------------ the rebuild
_MAKER_SPEC = None
_REFERENCE = {}


def _maker_spec():
    global _MAKER_SPEC
    if _MAKER_SPEC is None:
        _MAKER_SPEC = base.maker_spec()
    return _MAKER_SPEC


def _reference(contacts):
    """The maker's scene under the unit's numerics (the build path of base.unit_spec with the file's values)."""
    key = bool(contacts)
    if key not in _REFERENCE:
        _REFERENCE[key] = base.unit_model(None, contacts=key)
    return _REFERENCE[key]


def _copy_value(v):
    if isinstance(v, np.ndarray):
        return v.copy()
    return v


def rebuild_spec(desc, contacts=False, world=None):
    """The rebuilt spec of a checked description (module text, step 3), with the world declared into it."""
    if not isinstance(desc, Description):
        raise ModelError("rebuild_spec takes a Description")
    spec = mujoco.MjSpec.from_string(desc.text)
    maker = _maker_spec()
    for f in OPTION_FIELDS:
        setattr(spec.option, f, _copy_value(getattr(maker.option, f)))
    for f in SIZE_FIELDS:
        setattr(spec, f, getattr(maker, f))
    base.apply_forced_numerics(spec, contacts)
    for name in base.HINGES:
        j, mj = spec.joint(name), maker.joint(name)
        j.margin = float(mj.margin)
        j.actgravcomp = False
    for k in list(spec.keys):
        spec.delete(k)
    for p in list(spec.pairs):
        spec.delete(p)
    for e in list(spec.excludes):
        spec.delete(e)
    for name in pm.EQUALITY_NAMES:
        e, me = spec.equality(name), maker.equality(name)
        e.data = me.data.copy()
        e.solref = me.solref.copy()
        e.solimp = me.solimp.copy()
        e.active = True
    kb = {a: {"k": desc.k[i], "b": desc.b[i]} for i, (a, _) in enumerate(base.ACTUATORS)}
    base.replace_actuators(spec, kb)
    (world or HookSet()).declare(spec)
    return spec


def _ids_of(m, objtype, names):
    out = [mujoco.mj_name2id(m, objtype, n) for n in names]
    if min(out) < 0:
        raise ModelError("the rebuilt model lacks a named element")
    return np.array(out, dtype=np.int64)


def _same(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()


def rebuild_differences(desc, model, contacts=False):
    """What the rebuilt `model` changed of what was read, or left different from the maker's scene under the same
    numerics: a list of field names (empty when the rebuild is as stated)."""
    m0, ref = desc.model, _reference(contacts)
    O = mujoco.mjtObj
    bad = []
    b0 = np.arange(m0.nbody, dtype=np.int64)
    b1 = _ids_of(model, O.mjOBJ_BODY, ["world"] + [mujoco.mj_id2name(m0, O.mjOBJ_BODY, i) for i in range(1, m0.nbody)])
    for f in READ_BODY:
        if f == "body_parentid":
            if not _same(np.asarray(getattr(model, f), dtype=np.int64)[b1],
                         b1[np.asarray(getattr(m0, f), dtype=np.int64)[b0]]):
                bad.append(f)
        elif not _same(np.asarray(getattr(model, f))[b1], np.asarray(getattr(m0, f))[b0]):
            bad.append(f)
    j0 =_ids_of(m0, O.mjOBJ_JOINT, base.HINGES)
    j1 = _ids_of(model, O.mjOBJ_JOINT, base.HINGES)
    jr = _ids_of(ref, O.mjOBJ_JOINT, base.HINGES)
    for f in READ_JOINT:
        a1 = np.asarray(getattr(model, f))[j1]
        a0 = np.asarray(getattr(m0, f))[j0]
        if f == "jnt_bodyid":
            a1 = np.array([mujoco.mj_id2name(model, O.mjOBJ_BODY, int(x)) for x in a1])
            a0 = np.array([mujoco.mj_id2name(m0, O.mjOBJ_BODY, int(x)) for x in a0])
        if not _same(a1, a0):
            bad.append(f)
    d0, d1, dr = m0.jnt_dofadr[j0], model.jnt_dofadr[j1], ref.jnt_dofadr[jr]
    for f in READ_DOF:
        if not _same(np.asarray(getattr(model, f))[d1], np.asarray(getattr(m0, f))[d0]):
            bad.append(f)
    for f in SAME_JOINT:
        if not _same(np.asarray(getattr(model, f))[j1], np.asarray(getattr(ref, f))[jr]):
            bad.append(f)
    for f in SAME_DOF:
        if not _same(np.asarray(getattr(model, f))[d1], np.asarray(getattr(ref, f))[dr]):
            bad.append(f)
    names = [a for a, _ in base.ACTUATORS]
    a1, ar = _ids_of(model, O.mjOBJ_ACTUATOR, names), _ids_of(ref, O.mjOBJ_ACTUATOR, names)
    if model.nu != N_MODULES:
        bad.append("nu")
    for f in SAME_ACT:
        if not _same(np.asarray(getattr(model, f))[a1], np.asarray(getattr(ref, f))[ar]):
            bad.append(f)
    if not _same(model.actuator_gainprm[a1, 1:], ref.actuator_gainprm[ar, 1:]) or not _same(
            model.actuator_biasprm[a1, 1:], ref.actuator_biasprm[ar, 1:]):
        bad.append("actuator_gainprm / actuator_biasprm beyond the first term")
    if not _same(model.actuator_gainprm[a1, 0], np.array(desc.k)) or not _same(model.actuator_biasprm[a1, 0],
                                                                                 np.array(desc.b)):
        bad.append("k / b")
    e1, er = _ids_of(model, O.mjOBJ_EQUALITY, pm.EQUALITY_NAMES), _ids_of(ref, O.mjOBJ_EQUALITY, pm.EQUALITY_NAMES)
    for f in SAME_EQ:
        if not _same(np.asarray(getattr(model, f))[e1], np.asarray(getattr(ref, f))[er]):
            bad.append(f)
    for f in OPTION_FIELDS:
        if not _same(np.asarray(getattr(model.opt, f)), np.asarray(getattr(ref.opt, f))):
            bad.append(f"opt.{f}")
    if model.nkey != 0:
        bad.append("keyframes")
    if model.nuserdata != ref.nuserdata:
        bad.append("nuserdata")
    return bad


def rebuild_model(desc, contacts=False, world=None):
    """The rebuilt, compiled model of a checked description, compared with what was read (module text, step 3)."""
    spec = rebuild_spec(desc, contacts, world)
    msgs = []
    old = mujoco.get_mju_user_warning()
    mujoco.set_mju_user_warning(lambda msg: msgs.append(str(msg)))
    try:
        model = spec.compile()
    except Exception as e:  # noqa: BLE001
        raise ModelError(f"the rebuilt description does not compile: {e}") from None
    finally:
        mujoco.set_mju_user_warning(old)
    if msgs:
        raise ModelError(f"the rebuilt description compiles with {len(msgs)} warning(s), the first: {msgs[0]}")
    bad = rebuild_differences(desc, model, contacts)
    if bad:
        raise ModelError(f"the rebuild changed or failed to set: {bad}")
    return model


# ------------------------------------------------------------------------------------------------------ from_model
def _gripper_zero(gz):
    """(left, right) as two finite floats, or ModelError."""
    if isinstance(gz, (str, bytes, bytearray, dict)) or not hasattr(gz, "__len__") or len(gz) != 2:
        raise ModelError("gripper_zero: two numbers (left, right) in rad")
    out = []
    for x in gz:
        if isinstance(x, bool):
            raise ModelError("gripper_zero: two numbers (left, right) in rad")
        try:
            v = float(x)
        except (TypeError, ValueError):
            raise ModelError("gripper_zero: two numbers (left, right) in rad") from None
        if not np.isfinite(v):
            raise ModelError("gripper_zero: two finite numbers (left, right) in rad")
        out.append(v)
    return tuple(out)


def from_model(mjcf_path, calibration_path, registers=None, world=None, noise_key=None, contacts=False, *,
               editions=None, imax=None, assets_dir=None, write_policy=None, contact_classes=None, truth_profile=(),
               gripper_zero=(0.0, 0.0)):
    """A runnable model of a description (module text).

    mjcf_path, calibration_path: the description's two files (paths, or their bytes). registers: the sixteen modules'
    registers (a RegisterSet: in a graded run the unit's own register dump, so that mapping ranges, current limit and
    control modes are the unit's); None = the catalogue registers of the nominal editions. world: a HookSet (default
    the empty world). noise_key: must be None (a model run has no reply noise). contacts: False (default) or True.
    editions: {module class: edition} where the modules are not of the nominal editions (their envelope and peak
    torque follow the edition). imax: the sixteen modules' current base in A as read at power-up (default: each
    edition's catalogue value). assets_dir: the given mesh files' folder (default: the vendored copy of the maker's
    assets). write_policy: a registers.WritePolicy or a profile name of registers.write_policy (default: no register
    write is accepted). contact_classes, truth_profile: as for a unit. gripper_zero: (left, right), the zero of each
    gripper module in rad (theta = q - zero for that gripper; default 0 and 0; a graded run passes the two gripper
    zeros of the unit it is compared with, which the calibration file cannot state).
    Returns an OpenArmUnit of kind 'model' with the attribute `description` (the checked Description)."""
    if noise_key is not None:
        raise ModelError("a model run has no reply noise: noise_key must be None")
    gz = _gripper_zero(gripper_zero)
    world = HookSet() if world is None else world
    if not isinstance(world, HookSet):
        raise ModelError("world is a HookSet")
    desc = load_description(mjcf_path, calibration_path, assets_dir)
    model = rebuild_model(desc, contacts, world)
    data = mujoco.MjData(model)
    regs = rg.nominal_register_set() if registers is None else registers
    if not isinstance(regs, RegisterSet):
        raise ModelError("registers is a RegisterSet")
    if isinstance(write_policy, rg.WritePolicy):
        wp = write_policy
    else:
        wp = rg.write_policy(write_policy or "none")
    modules = busmod.pair_modules(dict(editions or {}))
    if imax is not None:
        imax = [float(x) for x in imax]
        if len(imax) != N_MODULES:
            raise ModelError("imax: sixteen values")
    delta = list(desc.zero)
    for side, i in enumerate(GRIPPER_MODULES):
        delta[i] = gz[side]
    try:
        bus = busmod.BusLayer(model, data, modules, regs, delta=delta, sign=list(desc.sign), imax=imax,
                              k=list(desc.k), b=list(desc.b), heat=None, noise=None, policy=wp)
    except busmod.BusError as e:
        raise ModelError(str(e)) from None
    dadr = [int(model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j)]) for _, j in base.ACTUATORS]
    friction_base = [float(model.dof_frictionloss[d]) for d in dadr]
    ids = {"kind": MODEL_KIND, "robot_xml_sha256": desc.xml_sha256, "calibration_sha256": desc.cal_sha256,
           "registers_digest": hashlib.sha256(canonical_json(regs.to_dict()).encode("ascii")).hexdigest(),
           "editions": [m.edition for m in modules], "imax": [float(x) for x in bus.imax],
           "world_digest": world.digest(), "code_digest": code_digest(), "runner_digest": runner_digest(),
           "noise": False, "law": False, "law_digest": None, "contacts": bool(contacts),
           "gripper_zero": [float(x) for x in gz]}
    u = OpenArmUnit.from_parts(model, data, bus, world, friction_base, None, ids, contact_classes=contact_classes,
                               truth_profile=truth_profile,
                               tool_sites=(pm.TOOL_SITES["left"][0], pm.TOOL_SITES["right"][0]))
    u.description = desc
    return u


def zero_end_calibration():
    """The identity calibration (sign +1, zero offset 0 on the fourteen arm joints), as calibration.json bytes."""
    cal = {"format": pm.CALIBRATION_FORMAT,
           "joints": {j: {"sign": 1, "zero_offset_rad": 0.0} for j in base.ARM_JOINTS}}
    return (json.dumps(cal, indent=1, sort_keys=True) + "\n").encode("ascii")


__all__ = ["from_model", "load_description", "rebuild_spec", "rebuild_model", "rebuild_differences", "Description",
           "ModelError", "UnitInvalid", "runner_digest", "zero_end_calibration", "GRIPPER_MODULES"]
