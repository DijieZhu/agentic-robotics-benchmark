"""The one configuration of the humanoid assembly cell: `cell_config.json` next to this file.

New file of this task (not a vendored one). It replaces the precedent cells' scattered defaults (the race car's
CELL_CONFIG / DEFAULT_CAR_CONFIG, the ROV cell's CELL_CONFIG["physics"]) by ONE JSON file that the scene builder, the
safety monitor and the server all read: a change of a number is made there and nowhere else (the grader's copy
of the file keeps the same keys: every number is a key of the grader's configuration).

`load_config(path=None, overrides=None)` returns a deep copy (dicts merged key by key; lists and scalars replaced) and
checks it (`check_config`): the control tick is 10 ms, the angle unit is radians, the safety numbers are positive,
every proposal is at most its cap, the joint model covers every actuator type it names.
"""
import copy
import json
import math
import os

__all__ = ["CONFIG_PATH", "load_config", "check_config", "merge", "tick_s", "joint_params", "ConfigError"]

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(HERE, "cell_config.json")
TICK_S = 0.01            # the control tick the gripper tables and the latch were defined for (checked, not used)


class ConfigError(ValueError):
    pass


def merge(base, over):
    """deep merge of `over` onto a copy of `base` (dicts key by key; anything else replaced)"""
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def tick_s(cfg):
    P = cfg["physics"]
    return float(P["timestep"]) * int(P["tick_steps"])


def _pos(x, what):
    if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(float(x)) or float(x) <= 0:
        raise ConfigError("%s must be a positive finite number, got %r" % (what, x))
    return float(x)


def check_config(cfg):
    P = cfg["physics"]
    if P.get("compiler_degree") is not False:
        raise ConfigError("physics.compiler_degree must be false (every angle is in radians)")
    dt = _pos(P["timestep"], "physics.timestep")
    n = int(P["tick_steps"])
    if n < 1 or abs(dt * n - TICK_S) > 1e-12:
        raise ConfigError("physics: timestep x tick_steps = %r s; the control tick is %r s" % (dt * n, TICK_S))
    if P["integrator"] not in ("implicitfast", "implicit", "euler", "rk4") or P["cone"] not in ("elliptic", "pyramidal"):
        raise ConfigError("physics: integrator %r / cone %r" % (P["integrator"], P["cone"]))
    for k in ("pad_solref", "weld_solref"):
        if len(P[k]) != 2:
            raise ConfigError("physics.%s needs 2 numbers" % k)
    rc = P["rest_contact"]
    if len(rc["solref"]) != 2 or len(rc["solimp"]) != 5:
        raise ConfigError("physics.rest_contact needs solref (2) and solimp (5)")
    jf = P["joint_friction_impedance"]
    if len(jf["solref_friction"]) != 2 or len(jf["solimp_friction"]) != 5:
        raise ConfigError("physics.joint_friction_impedance needs solref_friction (2) and solimp_friction (5)")
    J = cfg["joint_model"]
    for a in J["friction_Nm"]:
        _pos(J["friction_Nm"][a], "joint_model.friction_Nm." + a)
        if a not in J["damping_Nms_rad"] or a not in J["armature_kg_m2"]:
            raise ConfigError("joint_model: %s lacks damping or armature" % a)
    for k, v in J["parallel_ankle"].items():
        if v["actuator"] not in J["friction_Nm"]:
            raise ConfigError("joint_model.parallel_ankle.%s: unknown actuator %r" % (k, v["actuator"]))
    S = cfg["safety"]
    for k in ("robot_force_N", "robot_cap_N", "part_force_N", "part_cap_N", "part_speed_m_s", "part_speed_cap_m_s",
              "event_repeat_s"):
        _pos(S[k], "safety." + k)
    if int(S["part_ticks"]) < 1:
        raise ConfigError("safety.part_ticks must be >= 1")
    for prop, cap in (("robot_force_N", "robot_cap_N"), ("part_force_N", "part_cap_N"), ("part_speed_m_s", "part_speed_cap_m_s")):
        if float(S[prop]) > float(S[cap]):
            raise ConfigError("safety.%s above its cap %s" % (prop, cap))
    if S["floor_contact"] != "scrap_and_violation":
        raise ConfigError("safety.floor_contact: only 'scrap_and_violation' is implemented (the floor-contact rule R5)")
    B = cfg["budgets"]
    _pos(B["sim_cap_s"], "budgets.sim_cap_s")
    _pos(B["silence_s"], "budgets.silence_s")
    if int(B["camera_frames"]) < 0 or int(B["max_ticks_per_step"]) < 1:
        raise ConfigError("budgets: camera_frames >= 0 and max_ticks_per_step >= 1")
    return cfg


def load_config(path=None, overrides=None):
    with open(path or CONFIG_PATH, encoding="utf-8") as f:
        cfg = json.load(f)
    if cfg.get("schema") != "humanoid-cell-config/1":
        raise ConfigError("%s: schema %r" % (path or CONFIG_PATH, cfg.get("schema")))
    if overrides:
        cfg = merge(cfg, overrides)
    return check_config(cfg)


def joint_params(cfg, actuator):
    """(frictionloss N m, damping N m s/rad, armature kg m2) of an unpowered joint of actuator type `actuator` (an
    EC-A model, or 'ankle_pitch' / 'ankle_roll' for the parallel ankle mapped onto its two MJCF joints)"""
    J = cfg["joint_model"]
    if actuator in J["parallel_ankle"]:
        a = J["parallel_ankle"][actuator]
        return (float(J["friction_Nm"][a["actuator"]]) * float(a["friction_factor"]), float(a["damping_Nms_rad"]),
                float(a["armature_kg_m2"]))
    if actuator not in J["friction_Nm"] and isinstance(actuator, str):
        actuator = "-".join(actuator.split("-")[:2])          # a full vendor model name, e.g. EC-A4315-P2-36 -> EC-A4315
    if actuator not in J["friction_Nm"]:
        raise ConfigError("joint_model: unknown actuator type %r" % (actuator,))
    return float(J["friction_Nm"][actuator]), float(J["damping_Nms_rad"][actuator]), float(J["armature_kg_m2"][actuator])
