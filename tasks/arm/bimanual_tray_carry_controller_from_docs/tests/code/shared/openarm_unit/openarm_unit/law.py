import copy
import hashlib
import json
import math
import os
from dataclasses import dataclass, field

import numpy as np

from . import hs as hsmod
from . import registers as rg
from . import rng, thermal
from .state import (ARM_MODULES, GRIPPER_MODULES, N_ARM_JOINTS, N_MODULES, BodyInertial, CalState, HostProfile,
                    HSState, Mount, PlantState, RegisterSet, SessionState, ThermalState, UnitState, canonical_json,
                    public_params, public_params_sha256)

PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
PARAMS_LAW_PATH = os.path.join(PACKAGE_DIR, "params_law.json")
PROFILE_FORMAT = "openarm_unit/profile/1"
BRACKET_FORMAT = "openarm_unit/Bracket/1"
DEG = math.pi / 180.0
SIDES = ("left", "right")
MAX_ATTEMPTS = 64
LAW_CODE_FILES = ("thermal.py", "rng.py", "hs.py", "law.py")
FIELDS = ("position", "velocity", "torque")

BUMPED_STOPS = {
    **{f"openarm_left_joint{j}": w for j, w in ((1, "upper"), (2, "upper"), (3, "upper"), (4, "lower"),
                                                 (5, "upper"), (6, "upper"), (7, "upper"))},
    **{f"openarm_right_joint{j}": w for j, w in ((1, "lower"), (2, "lower"), (3, "upper"), (4, "lower"),
                                                  (5, "upper"), (6, "upper"), (7, "upper"))},
}

_LAW = None


class LawError(ValueError):
    ""


class ProfileError(LawError):

    def __init__(self, msg, report=None):
        super().__init__(msg)
        self.report = report or []


def law_params(path=None):
    global _LAW
    if path is not None:
        with open(path, "rb") as f:
            return json.loads(f.read().decode("ascii"))
    if _LAW is None:
        with open(PARAMS_LAW_PATH, "rb") as f:
            _LAW = json.loads(f.read().decode("ascii"))
    return _LAW


def law_sha256(path=PARAMS_LAW_PATH):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _v(law, *keys):
    node = law
    for k in keys:
        node = node[k]
    return node["value"] if isinstance(node, dict) and "value" in node and "status" in node else node


def _resolve(law, path):
    node = law
    for k in path:
        if isinstance(node, list):
            node = node[int(k)]
        else:
            node = node[k]
    return node


class HeatPlugin:

    def __init__(self, law, editions, h=None):
        self.consts = thermal.constants_for_modules(law, list(editions), h)

    def step(self, t_w, t_h, t_del, t_amb):
        return thermal.step(t_w, t_h, t_del, self.consts, t_amb)

    def latch(self, latched, t_w, ot_value):
        return thermal.coil_latch(latched, t_w, ot_value)

    def clear_allowed(self, t_w, ot_value):
        return thermal.coil_clear_allowed(t_w, ot_value, self.consts.hysteresis)


class ReplyNoise:

    def __init__(self, law, key):
        st = _v(law, "reply_noise", "steps")
        self.steps = np.array([float(st[f]) for f in FIELDS], dtype=np.float64)
        self.key = bytes(key)
        self._names = {}
        self._steps_col = self.steps[:, None]

    def offsets(self, stream, bus, start, count, res):
        names = self._names.get((stream, bus))
        if names is None:
            prefix = "noise" if stream == "reply" else stream
            names = self._names[(stream, bus)] = tuple(f"{prefix}/{SIDES[bus]}/{f}" for f in FIELDS)
        n = rng.normals_streams(self.key, names, start, count)
        return np.ascontiguousarray((self._steps_col * n).T * np.asarray(res, dtype=np.float64))


class LawPlugins:

    def __init__(self, law=None):
        self.law = law_params() if law is None else law
        self._digest = None

    def digest(self):
        if self._digest is None:
            h = hashlib.sha256()
            h.update(canonical_json(self.law).encode("ascii") + b"\0")
            for name in LAW_CODE_FILES:
                with open(os.path.join(PACKAGE_DIR, name), "rb") as f:
                    b = f.read()
                h.update(name.encode("ascii") + b"\0" + str(len(b)).encode("ascii") + b"\0" + b)
            self._digest = h.hexdigest()
        return self._digest

    def heat(self, editions, h):
        return HeatPlugin(self.law, editions, h)

    def noise(self, key):
        return ReplyNoise(self.law, key)


_MODULES = None


def modules():
    global _MODULES
    if _MODULES is None:
        _MODULES = public_params()["conventions"]["modules"]["value"]
    return _MODULES


def arm_module(a):
    return ARM_MODULES[a]


def _strip(name):
    s = str(name).strip()
    note = ""
    if "(" in s and s.endswith(")"):
        i = s.index("(")
        note = s[i + 1:-1].strip()
        s = s[:i].strip()
    return s.lower(), note


def _joint_set(s):
    s = s.replace(" ", "").upper()
    if s.startswith("J") and s[1:].isdigit():
        return {int(s[1:])}
    if s.count("-") == 1:
        a, b = s.split("-")
        if a.startswith("J") and b.startswith("J") and a[1:].isdigit() and b[1:].isdigit():
            return set(range(int(a[1:]), int(b[1:]) + 1))
    return None


def _arms(s):
    if s in ("left", "left arm", "left gripper"):
        return {0}
    if s in ("right", "right arm", "right gripper"):
        return {1}
    if s in ("left, right", "both arms", "left and right", "left, right gripper", "both grippers", "one arm"):
        return {0, 1}
    return None


def elements_of(space, name):
    s, _ = _strip(name)
    js = _joint_set(s)
    if space == "joint7":
        if js and max(js) <= 7:
            return sorted(j - 1 for j in js)
        if s in ("14 arm joints", "14 arm modules", "all", "every arm joint"):
            return list(range(7))
    elif space == "arm14":
        if js and max(js) <= 7:
            return sorted(arm * 7 + j - 1 for arm in (0, 1) for j in js)
        if s in ("14 arm joints", "14 arm modules", "all", "every arm joint"):
            return list(range(14))
        a = _arms(s)
        if a is not None and "gripper" not in s:
            return sorted(arm * 7 + j for arm in a for j in range(7))
    elif space == "module16":
        if s in ("16 modules", "all", "every module"):
            return list(range(16))
        if s in ("14 arm joints", "14 arm modules"):
            return list(ARM_MODULES)
        if s in ("left, right gripper", "grippers", "both grippers", "2 grippers", "the two grippers"):
            return list(GRIPPER_MODULES)
        if s in ("left gripper",):
            return [GRIPPER_MODULES[0]]
        if s in ("right gripper",):
            return [GRIPPER_MODULES[1]]
        for cls in ("dm8009", "dm4340", "dm4310"):
            if s == cls + " modules":
                return [m["index"] for m in modules() if m["class"].lower() == cls]
        if js and max(js) <= 8:
            return sorted(m["index"] for m in modules() if m["joint"] in js)
        a = _arms(s)
        if a is not None and "gripper" not in s:
            return sorted(i for i in range(16) if i // 8 in a)
    elif space == "arm2":
        a = _arms(s)
        if a is not None:
            return sorted(a)
    elif space == "stop28":
        if s in ("28 stops", "all", "every stop"):
            return list(range(28))
        if s in ("bumped", "the bumped stops", "the stop the zero routine bumps", "bumped stops"):
            return bumped_stop_elements()
        if s in ("unbumped", "the other stops"):
            b = set(bumped_stop_elements())
            return [e for e in range(28) if e not in b]
    elif space == "class3":
        if s in ("every class", "all", "all classes"):
            return [0, 1, 2]
        for k, cls in enumerate(("dm8009", "dm4340", "dm4310")):
            if s == cls:
                return [k]
    elif space == "unit1":
        if s in ("the unit's site", "supply", "unit", "the unit", "all", "14 arm joints", "every joint"):
            return [0]
    elif space == "body18":
        if s in ("all links", "all", "18 bodies"):
            return list(range(18))
    elif space == "axis6":
        a = _arms(s)
        if a is not None:
            return sorted(arm * 3 + k for arm in a for k in range(3))
    elif space == "reg48":
        if s in ("16 modules x (flux, rs, ls)", "16 modules", "all"):
            return list(range(48))
        if s.startswith("16 modules x"):
            return list(range(48))
    raise KeyError(f"element {name!r} is not an element name of the space {space}")


def bumped_stop_elements():
    names = [f"openarm_{s}_joint{j}" for s in SIDES for j in range(1, 8)]
    return [2 * a + (1 if BUMPED_STOPS[n] == "upper" else 0) for a, n in enumerate(names)]


CLASSES = ("DM8009", "DM4340", "DM4310")


def module_class(i):
    return modules()[i]["class"]


def _p1_range(law, e):
    return tuple(_v(law, "quantities", "P1", "range_Nm")[e])


def _p2_group(a):
    j = a % 7
    return "J1-J2" if j < 2 else ("J3-J4" if j < 4 else "J5-J7")


def _c4_group(a):
    return "J1-J4" if a % 7 < 4 else "J5-J7"


def _quantities(law):
    q = lambda *k: _v(law, "quantities", *k)
    P = {
        "ST1": dict(space="stop28", kind="cont", shape="uniform", unit="deg",
                    range=lambda e: tuple(q("ST1", "offset_range_deg")), nominal=lambda e: 0.0,
                    path=("quantities", "ST1", "offset_range_deg")),
        "C0": dict(space="module16", kind="cont", shape="uniform", unit="rad",
                   range=lambda e: tuple(q("C0", "range_rad")), nominal=lambda e: float(q("C0", "nominal")),
                   path=("quantities", "C0", "range_rad")),
        "C1": dict(space="arm14", kind="cont", shape="uniform", unit="deg",
                   range=lambda e: tuple(q("C1", "residual_range_deg")), nominal=lambda e: 0.0,
                   path=("quantities", "C1", "residual_range_deg")),
        "C2": dict(space="module16", kind="disc", unit="rad/s", only=lambda e: module_class(e) == "DM4340",
                   values=lambda e: [float(x) for x in q("C2", "values")], nominal=lambda e: float(q("C2", "nominal")),
                   path=("quantities", "C2", "values")),
        "C3": dict(space="module16", kind="cont", shape="uniform", unit="-",
                   range=lambda e: tuple(q("C3", "range")), nominal=lambda e: float(q("C3", "nominal")),
                   path=("quantities", "C3", "range")),
        "C4": dict(space="arm14", kind="cont", shape="uniform", unit="N m",
                   range=lambda e: tuple(q("C4", "range_Nm")[_c4_group(e)]), nominal=lambda e: float(q("C4", "nominal")),
                   path=("quantities", "C4", "range_Nm")),
        "C5": dict(space="module16", kind="special"),
        "C6": dict(space="class3", kind="disc", unit="-",
                   values=lambda e: list(q("C6", "values").get(CLASSES[e], [_nominal_edition(CLASSES[e])])),
                   nominal=lambda e: _nominal_edition(CLASSES[e]), path=("quantities", "C6", "values")),
        "C7": dict(space="module16", kind="disc", unit="A", values=lambda e: None, nominal=lambda e: None,
                   path=("quantities", "C7", "values")),
        "P1": dict(space="joint7", kind="cont", shape="log-uniform", unit="N m", range=lambda e: _p1_range(law, e),
                   nominal=lambda e: float(q("P1", "nominal_Nm")[e]), path=("quantities", "P1", "range_Nm")),
        "P1.arm_factor": dict(space="arm2", kind="cont", shape="uniform", unit="-",
                              range=lambda e: tuple(q("P1", "arm_factor_range")), nominal=lambda e: 1.0,
                              path=("quantities", "P1", "arm_factor_range")),
        "HS1.relative_rms": dict(space="unit1", kind="special"),
        "HS1.period": dict(space="arm14", kind="cont", shape="uniform", unit="rad",
                           range=lambda e: tuple(q("HS1", "period_range_rad")), nominal=lambda e: None,
                           path=("quantities", "HS1", "period_range_rad")),
        "HS1.phases": dict(space="arm14", kind="cont", shape="uniform", unit="-", range=lambda e: (0.0, 1.0),
                           nominal=lambda e: None, path=None),
        "P2": dict(space="arm14", kind="cont", shape="uniform", unit="N m s/rad",
                   range=lambda e: tuple(q("P2", "range_Nm_s_rad")[_p2_group(e)]),
                   nominal=lambda e: float(q("P2", "nominal_Nm_s_rad")[e % 7]), path=("quantities", "P2", "range_Nm_s_rad")),
        "P3": dict(space="arm14", kind="fixed_only"),
        "P4": dict(space="body18", kind="special"),
        "P5": dict(space="arm2", kind="special"),
        "P6": dict(space="arm2", kind="cont", shape="log-uniform", unit="N m", range=lambda e: tuple(q("P6", "range_Nm")),
                   nominal=lambda e: float(q("P6", "nominal_Nm")), path=("quantities", "P6", "range_Nm")),
        "P7": dict(space="arm2", kind="cont", shape="uniform", unit="N m s/rad",
                   range=lambda e: tuple(q("P7", "range_Nm_s_rad")), nominal=lambda e: float(q("P7", "nominal_Nm_s_rad")),
                   path=("quantities", "P7", "range_Nm_s_rad")),
        "PL1.translation": dict(space="axis6", kind="cont", shape="uniform", unit="m", only=lambda e: e >= 3,
                                range=lambda e: tuple(q("PL1", "translation_range_m")), nominal=lambda e: 0.0,
                                path=("quantities", "PL1", "translation_range_m")),
        "PL1.rotation": dict(space="axis6", kind="cont", shape="uniform", unit="deg", only=lambda e: e >= 3,
                             range=lambda e: tuple(q("PL1", "rotation_range_deg")), nominal=lambda e: 0.0,
                             path=("quantities", "PL1", "rotation_range_deg")),
        "TH1": dict(space="module16", kind="cont", shape="uniform", unit="-", range=lambda e: tuple(q("TH1", "range")),
                    nominal=lambda e: float(q("TH1", "nominal")), path=("quantities", "TH1", "range")),
        "registers_identification_spread": dict(
            space="reg48", kind="cont", shape="uniform", unit="-",
            range=lambda e: tuple(q("registers_identification_spread", "range")), nominal=lambda e: 0.0,
            path=("quantities", "registers_identification_spread", "range")),
        "S1": dict(space="module16", kind="cont", shape="uniform", unit="-", range=lambda e: tuple(q("S1", "range")),
                   nominal=lambda e: float(q("S1", "nominal")), path=("quantities", "S1", "range")),
        "S2": dict(space="arm14", kind="cont", shape="uniform", unit="N m",
                   range=lambda e: tuple(q("S2", "range_Nm")[_c4_group(e)]), nominal=lambda e: float(q("S2", "nominal")),
                   path=("quantities", "S2", "range_Nm")),
        "S3.ambient": dict(space="unit1", kind="cont", shape="uniform", unit="deg C",
                           range=lambda e: tuple(q("S3", "ambient_range_C")),
                           nominal=lambda e: float(q("S3", "ambient_nominal_C")),
                           path=("quantities", "S3", "ambient_range_C")),
        "S3.start_offset": dict(space="module16", kind="cont", shape="uniform", unit="K",
                                range=lambda e: tuple(q("S3", "start_offset_range_K")), nominal=lambda e: 0.0,
                                path=("quantities", "S3", "start_offset_range_K")),
        "S4": dict(space="unit1", kind="fixed_only", nominal=lambda e: float(q("S4", "nominal_V"))),
    }
    return P


def _nominal_edition(cls):
    return public_params()["class_nominal_edition"]["value"][cls]


def _space_size(space):
    return {"joint7": 7, "arm14": 14, "module16": 16, "arm2": 2, "stop28": 28, "class3": 3, "unit1": 1, "body18": 18,
            "axis6": 6, "reg48": 48}[space]


def _group_key(qid, space, grouping, e):
    if grouping in (None, "element"):
        return e
    if grouping == "unit":
        return 0
    if grouping == "arm":
        if space in ("arm14",):
            return e // 7
        if space in ("module16",):
            return e // 8
        if space == "axis6":
            return e // 3
        if space == "arm2":
            return e
    if grouping == "class":
        if space == "module16":
            return CLASSES.index(module_class(e))
        if space == "arm14":
            return CLASSES.index(module_class(arm_module(e)))
    raise ProfileError(f"{qid}: grouping {grouping!r} does not apply to the elements of {space}")


def _grouping_of(text, qid):
    if text is None:
        return None, ""
    t = str(text).lower()
    if qid == "P1" and t.startswith("unit"):
        return None, "read as the law's own grouping (the two arms share each joint's draw)"
    if qid == "registers_identification_spread" and t.startswith("module"):
        return "element", ""
    if t.startswith("class"):
        return "class", "one draw per module class (grouping is a narrowing)"
    for g in ("element", "module", "arm", "unit", "site"):
        if t.startswith(g):
            return {"module": "element", "site": "unit"}.get(g, g), ""
    return "?", ""


@dataclass
class ElementSpec:
    mode: str
    lo: float = None
    hi: float = None
    value: object = None
    values: list = None
    weights: list = None
    group: object = None
    entry: int = None


@dataclass
class Profile:
    name: str
    task: str
    zero_mode: str
    units: int
    sessions: tuple
    specs: dict
    raw: dict
    sha256: str
    report: list
    notes: list = field(default_factory=list)
    special: dict = field(default_factory=dict)

    def spec(self, qid, e):
        return self.specs.get(qid, {}).get(e, ElementSpec("nominal"))

    def mode_of(self, qid):
        return sorted({s.mode for s in self.specs.get(qid, {}).values()}) or ["nominal"]


_QID_ALIASES = {"PL1": "PL1", "HS1": "HS1"}


def load_profile(source, law=None, strict=True):
    law = law_params() if law is None else law
    if isinstance(source, (str, os.PathLike)):
        with open(source, "rb") as f:
            raw_b = f.read()
        raw = json.loads(raw_b.decode("utf-8"))
    elif isinstance(source, (bytes, bytearray)):
        raw_b = bytes(source)
        raw = json.loads(raw_b.decode("utf-8"))
    else:
        raw = copy.deepcopy(dict(source))
        raw_b = canonical_json(raw).encode("ascii")
    sha = hashlib.sha256(raw_b).hexdigest()
    Q = _quantities(law)
    report, notes, specs, special = [], [], {}, {}
    if raw.get("format") != PROFILE_FORMAT:
        raise ProfileError(f"a profile has format {PROFILE_FORMAT!r}")
    if raw.get("law_version") != law["law_version"]:
        notes.append({"where": "profile", "note": f"written against {raw.get('law_version')!r}, read against the law "
                      f"in force {law['law_version']!r}: every entry is checked against the law in force"})
    zm = raw.get("zero_mode", "nominal")
    if zm not in ("routine", "residual", "nominal"):
        raise ProfileError(f"zero_mode {zm!r} is routine, residual or nominal")
    wh = str(raw.get("worth_hiding", "off"))
    if not wh.lower().startswith("off"):
        raise ProfileError("the worth-hiding redraw is off in both tasks; 'on' is not built")
    sessions = tuple((raw.get("sessions") or {}).get("names") or ("session",))
    for s in sessions:
        if not s or "/" in s or not all(c.isalnum() or c in "_-" for c in s):
            raise ProfileError(f"session name {s!r}: letters, digits, '_' and '-' only")

    for idx, ent in enumerate(raw.get("entries") or []):
        row = {"index": idx, "quantity": ent.get("quantity"), "element": ent.get("element")}
        try:
            reading = _entry(ent, idx, law, Q, specs, special, zm)
            row["status"] = "accepted_with_reading" if reading else "accepted"
            row["reason"] = reading or ""
        except (ProfileError, KeyError, TypeError, ValueError) as exc:
            row["status"] = "refused"
            row["reason"] = str(exc.args[0]) if exc.args else repr(exc)
        report.append(row)
    if zm == "routine" and any(s.mode != "routine" for s in specs.get("C1", {}).values()):
        report.append({"index": None, "quantity": "C1", "element": "-", "status": "refused",
                       "reason": "zero_mode routine and a C1 entry other than 'routine'"})
    if zm == "residual" and not specs.get("C1"):
        notes.append({"where": "C1", "note": "zero_mode residual without a C1 entry: the law's residual range is drawn"})
        for a in range(14):
            lo, hi = Q["C1"]["range"](a)
            specs.setdefault("C1", {})[a] = ElementSpec("draw", lo, hi, group=a)
    if zm == "routine" and not specs.get("C1"):
        for a in range(14):
            specs.setdefault("C1", {})[a] = ElementSpec("routine")
    if "HS1.period" in specs and "HS1.phases" not in specs:
        specs["HS1.phases"] = {e: ElementSpec(s.mode, 0.0, 1.0, group=e) for e, s in specs["HS1.period"].items()}
        notes.append({"where": "HS1.phases", "note": "not named: drawn with the period (uniform over each harmonic)"})
    if "HS1.relative_rms" not in special:
        drawn = any(s.mode == "draw" for s in specs.get("HS1.period", {}).values())
        special["HS1.relative_rms"] = float(_v(law, "quantities", "HS1", "relative_rms")) if drawn else 0.0
        notes.append({"where": "HS1", "note": "relative RMS not named: " + ("the law's" if drawn else "off (r = 0)")})
    elif special["HS1.relative_rms"] > 0.0 and not any(s.mode == "draw" for s in specs.get("HS1.period", {}).values()):
        report.append({"index": None, "quantity": "HS1", "element": "-", "status": "refused",
                       "reason": "a friction profile with r > 0 needs its periods drawn (no nominal period exists)"})
    prof = Profile(raw.get("profile", "?"), raw.get("task", "?"), zm, int(raw.get("units", 1)), sessions, specs, raw,
                   sha, report, notes, special)
    bad = [r for r in report if r["status"] == "refused"]
    if strict and bad:
        raise ProfileError(f"{len(bad)} entr{'y' if len(bad) == 1 else 'ies'} refused: " +
                           "; ".join(f"#{r['index']} {r['quantity']} [{r['element']}]: {r['reason']}" for r in bad),
                           report)
    return prof


def _num_eq(a, b):
    return float(a) == float(b)


def _check_law_path(ent, law, q, qid):
    lp = ent.get("law_path")
    if lp is None:
        return ""
    try:
        _resolve(law, lp)
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise ProfileError(f"law_path {lp} does not resolve in the law") from exc
    own = q.get("path")
    if own is not None and tuple(lp[:2]) != tuple(own[:2]):
        raise ProfileError(f"law_path {lp} is not the law's node of {qid} ({list(own[:2])})")
    if own is not None and ent.get("mode") == "draw" and tuple(lp[:len(own)]) != tuple(own):
        raise ProfileError(f"law_path {lp}: a draw points at the law's range of {qid} ({list(own)})")
    return ""


def _entry(ent, idx, law, Q, specs, special, zero_mode):
    raw_q = ent.get("quantity")
    if not isinstance(raw_q, str):
        raise ProfileError("an entry names its quantity")
    qid = raw_q
    mode = ent.get("mode")
    if mode not in ("draw", "fixed", "nominal", "off", "routine"):
        raise ProfileError(f"mode {mode!r} is draw, fixed, nominal, off or routine")
    readings = []
    if qid == "PL1":
        side = elements_of("arm2", ent.get("element"))
        if mode != "nominal":
            raise ProfileError("PL1 as a whole takes mode nominal; draws name PL1.translation / PL1.rotation")
        for sub in ("PL1.translation", "PL1.rotation"):
            for e in (k for a in side for k in range(3 * a, 3 * a + 3)):
                _set(specs, sub, e, ElementSpec("nominal", entry=idx))
        return ""
    if qid not in Q:
        raise ProfileError(f"quantity {qid!r} is not a quantity of the law")
    q = Q[qid]
    if q["kind"] == "fixed_only":
        if mode not in ("nominal", "fixed"):
            raise ProfileError(f"{qid} is fixed by the law (not drawn)")
        if "value" in ent and q.get("nominal") and not _num_eq(ent["value"], q["nominal"](0)):
            raise ProfileError(f"{qid}: the law fixes {q['nominal'](0)}, the profile states {ent['value']}")
        elements_of(q["space"], ent.get("element"))
        return ""
    if qid == "HS1.relative_rms":
        r_law = float(_v(law, "quantities", "HS1", "relative_rms"))
        try:
            elements_of("arm14", ent.get("element"))
        except KeyError:
            elements_of("unit1", ent.get("element"))
        if mode == "off":
            special[qid] = 0.0
            return ""
        val = ent.get("value", r_law)
        if not _num_eq(val, r_law):
            raise ProfileError(f"HS1 relative RMS is {r_law} on every unit (the law draws no other value); {val} given")
        special[qid] = r_law
        return ("mode 'draw' with a value: read as the law's fixed relative RMS " + repr(r_law) +
                " (the law does not draw r)") if mode == "draw" else ""
    if qid == "C5":
        return _c5(ent, law, special, mode)
    if qid == "P4":
        elements_of("body18", ent.get("element"))
        if mode not in ("nominal", "draw"):
            raise ProfileError("P4 takes draw or nominal")
        if mode == "draw" and "range" in ent:
            raise ProfileError("P4: a narrowed range is not built (ranges mass_range_rel and com_range_m apply)")
        special["P4"] = mode
        return ""
    if qid == "P5":
        return _p5(ent, law, special, mode)
    if qid == "C1" and mode == "routine":
        if zero_mode != "routine":
            raise ProfileError("C1 routine needs zero_mode routine")
        for e in elements_of("arm14", ent.get("element")):
            _set(specs, "C1", e, ElementSpec("routine", entry=idx))
        return ""
    if qid == "C1" and zero_mode == "routine" and mode != "nominal":
        raise ProfileError("zero_mode routine: C1 is written by the routine, not drawn")
    if mode == "routine":
        raise ProfileError(f"mode routine applies to C1 only")
    els = elements_of(q["space"], ent.get("element"))
    _, el_note = _strip(ent.get("element"))
    if "only" in q:
        outside = [e for e in els if not q["only"](e)]
        if outside and mode != "nominal":
            raise ProfileError(f"{qid}: elements {outside} are not drawn by the law (only nominal is accepted)")
        els = [e for e in els if q["only"](e)] if mode != "nominal" else els
    readings.append(_check_law_path(ent, law, q, qid))
    grouping, gnote = _grouping_of(ent.get("grouping"), qid)
    if grouping == "?":
        raise ProfileError(f"grouping {ent.get('grouping')!r} is not element, module, arm, class or unit")
    if gnote:
        readings.append(gnote)
    if mode == "off":
        raise ProfileError(f"{qid}: mode off exists only for HS1 (r = 0); a quantity is drawn, fixed or nominal")
    if q["kind"] == "disc":
        return _disc(ent, qid, q, els, mode, specs, idx, law, readings)
    shape = ent.get("shape")
    if mode == "draw":
        if shape is not None:
            s = str(shape).lower()
            logu = "log-uniform" in s or "log uniform" in s
            if logu != (q["shape"] == "log-uniform"):
                raise ProfileError(f"{qid}: shape {shape!r} differs from the law's {q['shape']!r}")
        rng_ = ent.get("range")
        for e in els:
            llo, lhi = q["range"](e)
            lo, hi = (llo, lhi) if rng_ is None else (float(rng_[0]), float(rng_[1]))
            if not (llo <= lo <= hi <= lhi):
                raise ProfileError(f"{qid} element {e}: range [{lo}, {hi}] is not inside the law's [{llo}, {lhi}] "
                                   "(a profile never widens the law)")
            if q["shape"] == "log-uniform" and lo <= 0.0:
                raise ProfileError(f"{qid}: a log-uniform range is positive")
            g = _group_key(qid, q["space"], grouping, e)
            _set(specs, qid, e, ElementSpec("draw", lo, hi, group=(grouping, g), entry=idx))
        _check_groups(specs[qid], qid)
    elif mode == "fixed":
        if "value" not in ent:
            raise ProfileError(f"{qid}: mode fixed needs a value")
        v = ent["value"]
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise ProfileError(f"{qid}: a fixed value of a continuous quantity is a number, not {v!r}")
        for e in els:
            llo, lhi = q["range"](e)
            if not (llo <= float(v) <= lhi):
                raise ProfileError(f"{qid} element {e}: fixed value {v} outside the law's [{llo}, {lhi}]")
            _set(specs, qid, e, ElementSpec("fixed", value=float(v), entry=idx))
    else:
        if "value" in ent and isinstance(ent["value"], (int, float)) and not isinstance(ent["value"], bool):
            for e in els:
                nom = q["nominal"](e)
                if nom is not None and not _num_eq(ent["value"], nom):
                    raise ProfileError(f"{qid}: 'nominal' with value {ent['value']}, the law's nominal is {nom}")
        for e in els:
            _set(specs, qid, e, ElementSpec("nominal", entry=idx))
    if el_note:
        readings.append(f"element note {el_note!r} carried as text")
    return "; ".join(r for r in readings if r)


def _check_groups(sp, qid):
    by = {}
    for e, s in sp.items():
        if s.mode == "draw":
            by.setdefault(s.group, set()).add((s.lo, s.hi))
    for g, rr in by.items():
        if len(rr) > 1:
            raise ProfileError(f"{qid}: one group {g} holds elements with different ranges")


def _set(specs, qid, e, spec):
    d = specs.setdefault(qid, {})
    if e in d:
        raise ProfileError(f"{qid} element {e} is named by two entries (#{d[e].entry} and #{spec.entry})")
    d[e] = spec


def _disc(ent, qid, q, els, mode, specs, idx, law, readings):
    if qid == "C7":
        allowed = lambda e: [float(x) for x in _v(law, "quantities", "C7", "values").get(_nominal_edition(
            module_class(e)), [])] or [float(public_params()["editions"][_nominal_edition(module_class(e))]["Imax_A"]["value"])]
    elif qid == "C6":
        allowed = q["values"]
    else:
        allowed = q["values"]
    if mode == "fixed" or (mode == "nominal" and "value" in ent):
        v = ent.get("value")
        for e in els:
            vals = allowed(e)
            if qid == "C6":
                vv = v.get(CLASSES[e]) if isinstance(v, dict) else v
                if vv not in vals:
                    raise ProfileError(f"C6 {CLASSES[e]}: {vv!r} is not an edition of the law {vals}")
                if mode == "nominal" and vv != q["nominal"](e):
                    raise ProfileError(f"C6 {CLASSES[e]}: 'nominal' with {vv!r}, the nominal is {q['nominal'](e)!r}")
                _set(specs, qid, e, ElementSpec("fixed", value=vv, entry=idx))
                continue
            if isinstance(v, list):
                raise ProfileError(f"{qid}: a fixed value is one value")
            if not any(_num_eq(v, x) for x in vals):
                raise ProfileError(f"{qid} element {e}: {v} is not one of the law's values {vals}")
            _set(specs, qid, e, ElementSpec("fixed", value=float(v), entry=idx))
        return "; ".join(r for r in readings if r)
    if mode == "nominal":
        for e in els:
            _set(specs, qid, e, ElementSpec("nominal", entry=idx))
        return "; ".join(r for r in readings if r)
    vals = ent.get("values")
    w = ent.get("weights")
    for e in els:
        lv = allowed(e)
        vv = lv if vals is None else vals
        if not all(any((x == y) if isinstance(x, str) else _num_eq(x, y) for y in lv) for x in vv):
            raise ProfileError(f"{qid} element {e}: values {vv} not all among the law's {lv}")
        if w is not None and (len(w) != len(vv) or any(float(x) < 0 for x in w) or sum(float(x) for x in w) <= 0):
            raise ProfileError(f"{qid}: one non-negative weight per value")
        grouping, _ = _grouping_of(ent.get("grouping"), qid)
        g = _group_key(qid, q["space"], grouping if grouping not in ("?",) else None, e)
        _set(specs, qid, e, ElementSpec("draw", values=list(vv), weights=None if w is None else [float(x) for x in w],
                                        group=(grouping, g), entry=idx))
    return "; ".join(r for r in readings if r) + ("; " if readings and any(readings) else "") + \
        "a discrete draw with " + ("equal odds" if w is None else "the given weights") + " (probabilities ours)"


def _c5(ent, law, special, mode):
    nom = _v(law, "quantities", "C5", "nominal")
    tvals = [int(x) for x in _v(law, "quantities", "C5", "timeout_values_counts")]
    gmodes = list(_v(law, "quantities", "C5", "gripper_modes"))
    if mode == "nominal":
        special["C5"] = {"TIMEOUT": int(nom["TIMEOUT"]), "gripper_mode": None, "reading": "nominal"}
        return ""
    if mode != "fixed" or not isinstance(ent.get("value"), dict):
        raise ProfileError("C5 takes mode fixed with a mapping of register values (or nominal); drawing the registers "
                           "is not used by either task and not built")
    v = ent["value"]
    out = {"TIMEOUT": int(nom["TIMEOUT"]), "gripper_mode": None}
    readings = []
    for k, x in v.items():
        if k in ("OT_Value", "OC_Value", "CTRL_MODE_arm", "can_br"):
            if not _num_eq(x, nom[k]):
                raise ProfileError(f"C5 {k}: the law has only its nominal {nom[k]}, {x} given")
        elif k == "TIMEOUT":
            if int(x) != x or int(x) not in tvals:
                raise ProfileError(f"C5 TIMEOUT {x} is not one of the law's {tvals}")
            out["TIMEOUT"] = int(x)
        elif k == "gripper_mode":
            if x in gmodes:
                out["gripper_mode"] = x
            else:
                out["gripper_mode"] = None
                readings.append(f"gripper_mode {x!r} is not a register value of the law {gmodes}: read as 'the "
                                "controller writes the gripper's mode while disabled'; the unit is built with the "
                                "gripper register at the catalogue table's default (force-position, "
                                "registers.nominal_register_set)")
        else:
            raise ProfileError(f"C5: {k!r} is not a register of the law's C5")
    special["C5"] = out
    return "; ".join(readings)


def _p5(ent, law, special, mode):
    if mode == "nominal":
        if ent.get("value") not in (None, "nothing mounted", "nothing"):
            raise ProfileError("P5 nominal is 'nothing mounted'")
        special["P5"] = None
        return ""
    if mode != "draw":
        raise ProfileError("P5 takes draw or nominal")
    items = ent.get("items")
    if not isinstance(items, dict) or set(items) != set(SIDES):
        raise ProfileError("P5 draw: 'items' maps 'left' and 'right' to the mounts carried when that side is drawn "
                           "(their masses and poses are the task profile's)")
    nmax = int(_v(law, "quantities", "P5", "items_per_arm")[1])
    parsed = {}
    for side in SIDES:
        lst = items[side]
        if len(lst) > nmax:
            raise ProfileError(f"P5: at most {nmax} items per arm")
        ms = []
        for m in lst:
            mm = Mount.from_dict(m)
            if not mm.name.startswith(f"mount_{side}_"):
                raise ProfileError(f"P5: item {mm.name} on the {side} arm is named mount_{side}_<k>")
            if not (mm.mass > 0.0 and min(mm.diaginertia) > 0.0):
                raise ProfileError(f"P5: item {mm.name} needs a positive mass and inertia")
            ms.append(mm)
        parsed[side] = tuple(ms)
    odds = ent.get("side_odds", [0.5, 0.5])
    if len(odds) != 2 or any(float(x) < 0 for x in odds) or sum(float(x) for x in odds) <= 0:
        raise ProfileError("P5: side_odds are two non-negative numbers")
    special["P5"] = {"items": parsed, "odds": [float(x) for x in odds]}
    return "the side is drawn with odds " + repr([float(x) for x in odds]) + " (OURS: the profile's)"


class _Draw:

    def __init__(self, key, unit_index):
        if not isinstance(key, (bytes, bytearray)) or len(key) != 32:
            raise LawError("the registry's key is 32 bytes")
        self.key, self.ui = bytes(key), int(unit_index)
        self.log = []

    def u(self, qid, e, attempt=0):
        return rng.uniform(self.key, rng.law_stream(qid, e, attempt, self.ui), 0)

    def cont(self, qid, e, lo, hi, shape, attempt=0):
        u = self.u(qid, e, attempt)
        if shape == "log-uniform":
            a, b = math.log(lo), math.log(hi)
            x = math.exp(a + (b - a) * u)
            return min(max(x, lo), hi)
        return lo + (hi - lo) * u

    def disc(self, qid, e, values, weights=None, attempt=0):
        u = self.u(qid, e, attempt)
        w = [1.0] * len(values) if weights is None else list(weights)
        tot = 0.0
        for x in w:
            tot += x
        acc = 0.0
        for v, x in zip(values, w):
            acc += x / tot
            if u < acc:
                return v
        return values[-1]


def _value(dr, qid, e, spec, q, attempt=0):
    if spec.mode == "draw":
        kind, g = spec.group if spec.group else (None, e)
        sq = qid if kind in (None, "element") else f"{qid}@{kind}"
        if spec.values is not None:
            return dr.disc(sq, g, spec.values, spec.weights, attempt)
        return dr.cont(sq, g, spec.lo, spec.hi, q["shape"], attempt)
    if spec.mode == "fixed":
        return spec.value
    return q["nominal"](e)


_MAKER_INERTIALS = None


def maker_inertials():
    global _MAKER_INERTIALS
    if _MAKER_INERTIALS is not None:
        return dict(_MAKER_INERTIALS)
    import mujoco
    from . import base
    m = base.maker_model()
    out = {}
    for name in base.INERTIAL_BODIES:
        b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)
        out[name] = (float(m.body_mass[b]), tuple(float(x) for x in m.body_ipos[b]),
                     tuple(float(x) for x in m.body_iquat[b]), tuple(float(x) for x in m.body_inertia[b]))
    _MAKER_INERTIALS = out
    return dict(out)


def _quat_xyz(ax, ay, az):
    def mul(a, b):
        w1, x1, y1, z1 = a
        w2, x2, y2, z2 = b
        return (w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2)
    qx = (math.cos(ax / 2), math.sin(ax / 2), 0.0, 0.0)
    qy = (math.cos(ay / 2), 0.0, math.sin(ay / 2), 0.0)
    qz = (math.cos(az / 2), 0.0, 0.0, math.sin(az / 2))
    return mul(mul(qx, qy), qz)


def _xyz_of_quat(quat):
    w, x, y, z = (float(v) for v in quat)
    r00 = 1.0 - 2.0 * (y * y + z * z)
    r01 = 2.0 * (x * y - w * z)
    r02 = 2.0 * (x * z + w * y)
    r12 = 2.0 * (y * z - w * x)
    r22 = 1.0 - 2.0 * (x * x + y * y)
    return (math.atan2(-r12, r22), math.asin(max(-1.0, min(1.0, r02))), math.atan2(-r01, r00))


def draw_unit(key, profile, unit_index=0, law=None, unit_id=None, inertials=None):
    law = law_params() if law is None else law
    Q = _quantities(law)
    dr = _Draw(key, unit_index)
    pub = public_params()
    P = profile
    rec = {"unit_index": int(unit_index), "attempts": {}}

    editions = {}
    for k, cls in enumerate(CLASSES):
        editions[cls] = _value(dr, "C6", k, P.spec("C6", k), Q["C6"])
    sessions = []
    for si, sname in enumerate(P.sessions):
        tag = f".{sname}"
        s1 = [float(_value(dr, "S1" + tag, i, P.spec("S1", i), Q["S1"])) for i in range(16)]
        s2 = [0.0] * 16
        for a in range(14):
            s2[arm_module(a)] = float(_value(dr, "S2" + tag, a, P.spec("S2", a), Q["S2"]))
        amb = float(_value(dr, "S3.ambient" + tag, 0, P.spec("S3.ambient", 0), Q["S3.ambient"]))
        offs = [float(_value(dr, "S3.start_offset" + tag, i, P.spec("S3.start_offset", i), Q["S3.start_offset"]))
                for i in range(16)]
        sessions.append(SessionState.from_offsets(sname, s1, s2, amb, offs))

    fac = [float(_value(dr, "P1.arm_factor", a, P.spec("P1.arm_factor", a), Q["P1.arm_factor"])) for a in (0, 1)]
    fl = [0.0] * 16
    for j in range(7):
        sp = P.spec("P1", j)
        lo, hi = (sp.lo, sp.hi) if sp.mode == "draw" else Q["P1"]["range"](j)
        for att in range(MAX_ATTEMPTS):
            x = float(_value(dr, "P1", j, sp, Q["P1"], att))
            ok = all(lo <= x * fac[arm] * s.friction_factor[arm * 8 + j] <= hi for arm in (0, 1) for s in sessions) \
                and all(lo <= x * fac[arm] <= hi for arm in (0, 1))
            if ok:
                break
            if sp.mode != "draw":
                raise LawError(f"profile: P1 J{j + 1} at its {sp.mode} value {x} leaves the range with the factors")
        else:
            raise LawError(f"rejected: P1 J{j + 1} found no admissible draw in {MAX_ATTEMPTS} attempts")
        rec["attempts"][f"P1/J{j + 1}"] = att
        for arm in (0, 1):
            fl[arm * 8 + j] = x * fac[arm]
    for arm in (0, 1):
        sp = P.spec("P6", arm)
        lo, hi = (sp.lo, sp.hi) if sp.mode == "draw" else Q["P6"]["range"](arm)
        gm = GRIPPER_MODULES[arm]
        for att in range(MAX_ATTEMPTS):
            x = float(_value(dr, "P6", arm, sp, Q["P6"], att))
            if all(lo <= x * s.friction_factor[gm] <= hi for s in sessions):
                break
            if sp.mode != "draw":
                raise LawError(f"profile: P6 {SIDES[arm]} at its {sp.mode} value {x} leaves the range with S1")
        else:
            raise LawError(f"rejected: P6 {SIDES[arm]} found no admissible draw in {MAX_ATTEMPTS} attempts")
        rec["attempts"][f"P6/{SIDES[arm]}"] = att
        fl[gm] = x
    dm = [0.0] * 16
    for a in range(14):
        dm[arm_module(a)] = float(_value(dr, "P2", a, P.spec("P2", a), Q["P2"]))
    for arm in (0, 1):
        dm[GRIPPER_MODULES[arm]] = float(_value(dr, "P7", arm, P.spec("P7", arm), Q["P7"]))

    jr = pub["conventions"]["joint_ranges_rad"]["value"]
    stops = []
    for a in range(14):
        nom = jr[SIDES[a // 7]][a % 7]
        lo = float(nom[0]) + float(_value(dr, "ST1", 2 * a, P.spec("ST1", 2 * a), Q["ST1"])) * DEG
        hi = float(nom[1]) + float(_value(dr, "ST1", 2 * a + 1, P.spec("ST1", 2 * a + 1), Q["ST1"])) * DEG
        stops.append((lo, hi))

    ys = pub["conventions"]["base_link_y_m"]["value"]
    bases = []
    for arm in (0, 1):
        t = [float(_value(dr, "PL1.translation", 3 * arm + k, P.spec("PL1.translation", 3 * arm + k),
                          Q["PL1.translation"])) for k in range(3)]
        r = [float(_value(dr, "PL1.rotation", 3 * arm + k, P.spec("PL1.rotation", 3 * arm + k), Q["PL1.rotation"]))
             for k in range(3)]
        pos = (0.0 + t[0], float(ys[SIDES[arm]]) + t[1], 0.0 + t[2])
        quat = (1.0, 0.0, 0.0, 0.0) if r == [0.0, 0.0, 0.0] else _quat_xyz(r[0] * DEG, r[1] * DEG, r[2] * DEG)
        bases.append((pos, quat))

    inert = ()
    if P.special.get("P4") == "draw":
        nom = maker_inertials() if inertials is None else inertials
        mr = _v(law, "quantities", "P4", "mass_range_rel")
        cr = _v(law, "quantities", "P4", "com_range_m")
        from .base import INERTIAL_BODIES
        out = []
        for b, name in enumerate(INERTIAL_BODIES):
            mass, ipos, iquat, dia = nom[name]
            em = dr.cont("P4.mass", b, float(mr[0]), float(mr[1]), "uniform")
            dc = [dr.cont("P4.com", 3 * b + k, float(cr[0]), float(cr[1]), "uniform") for k in range(3)]
            out.append(BodyInertial(name, mass * (1.0 + em), tuple(ipos[k] + dc[k] for k in range(3)), iquat,
                                    tuple(d * (1.0 + em) for d in dia)))
        inert = tuple(out)
    mounts = ()
    p5 = P.special.get("P5")
    if p5:
        side = dr.disc("P5.side", 0, list(SIDES), p5["odds"])
        mounts = p5["items"][side]
        rec["P5_side"] = side

    zb = [float(_value(dr, "C0", i, P.spec("C0", i), Q["C0"])) for i in range(16)]
    if P.zero_mode == "routine":
        delta, src = list(zb), "as_drawn"
    elif P.zero_mode == "residual":
        delta = [0.0] * 16
        for a in range(14):
            delta[arm_module(a)] = float(_value(dr, "C1", a, P.spec("C1", a), Q["C1"])) * DEG
        src = "residual"
    else:
        delta, src = [0.0] * 16, "nominal"
    k = [float(_value(dr, "C3", i, P.spec("C3", i), Q["C3"])) for i in range(16)]
    b = [0.0] * 16
    for a in range(14):
        b[arm_module(a)] = float(_value(dr, "C4", a, P.spec("C4", a), Q["C4"]))
    imax = []
    for i in range(16):
        ed = editions[module_class(i)]
        base_imax = float(pub["editions"][ed]["Imax_A"]["value"])
        sp = P.spec("C7", i)
        if sp.mode == "nominal":
            imax.append(base_imax)
        else:
            vals = [float(x) for x in _v(law, "quantities", "C7", "values").get(ed, [base_imax])]
            v = float(_value(dr, "C7", i, sp, {"nominal": lambda e: base_imax}))
            if not any(v == x for x in vals):
                raise LawError(f"C7 module {i}: {v} is not a current base of the edition {ed} {vals}")
            imax.append(v)
    c5 = P.special.get("C5") or {"TIMEOUT": int(_v(law, "quantities", "C5", "nominal")["TIMEOUT"]), "gripper_mode": None}
    gmode = rg.MODE_MIT if c5.get("gripper_mode") == "MIT" else rg.MODE_POS_FORCE
    mods = []
    regnames = ("Flux", "Rs", "LS")
    for i, m in enumerate(pub["conventions"]["modules"]["value"]):
        ed = editions[m["class"]]
        mode = gmode if m["joint"] == 8 else rg.MODE_MIT
        vmax = None
        if m["class"] == "DM4340":
            vmax = float(_value(dr, "C2", i, P.spec("C2", i), Q["C2"]))
        cat = rg.catalogue_registers(ed, m["command_id"], m["reply_id"], mode, c5["TIMEOUT"], vmax=vmax)
        ident = {}
        for r_k, rn in enumerate(regnames):
            e = 3 * i + r_k
            sp = P.spec("registers_identification_spread", e)
            eps = float(_value(dr, "registers_identification_spread", e, sp, Q["registers_identification_spread"]))
            if eps != 0.0:
                ident[rn] = cat[rg.register_rid(rn)] * (1.0 + eps)
        if ident:
            cat = rg.catalogue_registers(ed, m["command_id"], m["reply_id"], mode, c5["TIMEOUT"], vmax=vmax,
                                         identified=ident)
        mods.append(cat)
    regs = RegisterSet(tuple(mods))
    cal = CalState(zb, delta, src, [1] * 16, k, b, imax, regs)

    r = float(P.special.get("HS1.relative_rms", 0.0))
    if r > 0.0:
        L = hsmod.HSLaw.from_law(law)
        periods = [float(_value(dr, "HS1.period", a, P.spec("HS1.period", a), Q["HS1.period"])) for a in range(14)]
        ph = []
        for a in range(14):
            u1 = dr.u("HS1.phase1", a)
            u2 = dr.u("HS1.phase2", a)
            ph.append((hsmod.TWO_PI * u1, hsmod.TWO_PI * u2))
        hst = hsmod.hs_state(periods, ph, L, r=r)
    else:
        hst = hsmod.off_state()
    h = [float(_value(dr, "TH1", i, P.spec("TH1", i), Q["TH1"])) for i in range(16)]

    plant = PlantState(fl, dm, inert, mounts, stops, tuple(bases))
    uid = unit_id or f"{P.name}-u{int(unit_index)}"
    st = UnitState(uid, law["law_version"], editions, plant, cal, hst, ThermalState(h))
    check_unit(st, sessions, P, law)
    rec["unit_digest"] = st.digest
    rec["session_digests"] = {s.session_id: s.digest for s in sessions}
    return st, sessions, rec


def check_unit(state, sessions, profile, law=None):
    law = law_params() if law is None else law
    pub = public_params()
    bad, n = [], [0]
    qv = lambda *k: _v(law, "quantities", *k)

    def inside(what, x, lo, hi):
        n[0] += 1
        if not (float(lo) <= float(x) <= float(hi)):
            bad.append(f"{what} = {x!r} outside [{lo}, {hi}]")

    def spec(qid, e):
        return profile.spec(qid, e) if profile is not None else None

    Q = _quantities(law)

    def prof(qid, e, what, x, scale=1.0, tol=0.0, nominal=True):
        sp = spec(qid, e)
        if sp is not None and sp.mode == "draw" and sp.lo is not None:
            inside(what + " (profile)", x, sp.lo * scale - tol, sp.hi * scale + tol)
        elif sp is not None and sp.mode == "fixed" and not isinstance(sp.value, str):
            n[0] += 1
            if not abs(float(x) - float(sp.value) * scale) <= tol:
                bad.append(f"{what} = {x!r}, the profile fixes {sp.value}")
        elif sp is not None and sp.mode == "nominal" and nominal and qid in Q and "nominal" in Q[qid]:
            nom = Q[qid]["nominal"](e)
            if nom is not None:
                n[0] += 1
                if not abs(float(x) - float(nom) * scale) <= tol:
                    bad.append(f"{what} (profile) = {x!r}, the profile keeps it at the law's nominal {nom!r}")

    def disc(qid, e, what, x, nom):
        sp = spec(qid, e)
        if sp is None:
            return
        n[0] += 1
        eq = (lambda a, b: a == b) if isinstance(x, str) else (lambda a, b: float(a) == float(b))
        if sp.mode == "fixed" and not eq(x, sp.value):
            bad.append(f"{what} (profile) = {x!r}, the profile fixes {sp.value!r}")
        elif sp.mode == "draw" and sp.values is not None and not any(eq(x, v) for v in sp.values):
            bad.append(f"{what} (profile) = {x!r}, not among the profile's values {sp.values!r}")
        elif sp.mode == "nominal" and not eq(x, nom):
            bad.append(f"{what} (profile) = {x!r}, the profile keeps it at the nominal {nom!r}")

    p, c = state.plant, state.cal
    p1r, af = qv("P1", "range_Nm"), qv("P1", "arm_factor_range")
    p1n = [float(x) for x in qv("P1", "nominal_Nm")]
    for arm in (0, 1):
        for j in range(7):
            i = arm * 8 + j
            inside(f"P1 x arm factor, module {i}", p.frictionloss[i], *p1r[j])
            prof("P1", j, f"P1 x arm factor, module {i}", p.frictionloss[i], nominal=False)
            for s in sessions:
                inside(f"P1 x factors, module {i}, session {s.session_id}", p.frictionloss[i] * s.friction_factor[i],
                       *p1r[j])
                prof("P1", j, f"P1 x factors, module {i}, session {s.session_id}",
                     p.frictionloss[i] * s.friction_factor[i], nominal=False)
            sp = spec("P1", j)
            if sp is not None and sp.mode in ("fixed", "nominal"):
                x0 = float(sp.value) if sp.mode == "fixed" else p1n[j]
                f = p.frictionloss[i] / x0
                inside(f"P1 arm factor {SIDES[arm]} (module {i})", f, af[0] - 1e-12, af[1] + 1e-12)
                fs = spec("P1.arm_factor", arm)
                if fs.mode == "nominal":
                    n[0] += 1
                    if abs(f - 1.0) > 1e-12:
                        bad.append(f"P1 module {i}: the arm factor is nominal (1), the level gives {f!r}")
                else:
                    prof("P1.arm_factor", arm, f"P1 arm factor {SIDES[arm]} (module {i})", f, tol=1e-12)
        g = GRIPPER_MODULES[arm]
        inside(f"P6 {SIDES[arm]}", p.frictionloss[g], *qv("P6", "range_Nm"))
        prof("P6", arm, f"P6 {SIDES[arm]}", p.frictionloss[g])
        inside(f"P7 {SIDES[arm]}", p.damping[g], *qv("P7", "range_Nm_s_rad"))
        prof("P7", arm, f"P7 {SIDES[arm]}", p.damping[g])

    def fac_bounds(arm):
        fs = spec("P1.arm_factor", arm)
        if fs is None or (fs.mode == "draw" and fs.lo is None):
            return float(af[0]), float(af[1])
        if fs.mode == "draw":
            return float(fs.lo), float(fs.hi)
        v = float(fs.value) if fs.mode == "fixed" else 1.0
        return v, v
    (lo_l, hi_l), (lo_r, hi_r) = fac_bounds(0), fac_bounds(1)
    for j in range(7):
        sp = spec("P1", j)
        if sp is not None and sp.mode != "draw":
            continue
        ratio = p.frictionloss[j] / p.frictionloss[8 + j]
        inside(f"P1 J{j + 1} left / right level ratio", ratio, af[0] / af[1] - 1e-12, af[1] / af[0] + 1e-12)
        if profile is not None:
            inside(f"P1 J{j + 1} left / right level ratio (profile)", ratio, lo_l / hi_r - 1e-12,
                   hi_l / lo_r + 1e-12)
    for a in range(14):
        i = arm_module(a)
        inside(f"P2 module {i}", p.damping[i], *qv("P2", "range_Nm_s_rad")[_p2_group(a)])
        prof("P2", a, f"P2 module {i}", p.damping[i])
    jr = pub["conventions"]["joint_ranges_rad"]["value"]
    sr = qv("ST1", "offset_range_deg")
    for a in range(14):
        nom = jr[SIDES[a // 7]][a % 7]
        for w in (0, 1):
            off = (p.stops[a][w] - float(nom[w])) / DEG
            inside(f"ST1 joint {a} {'upper' if w else 'lower'} offset (deg)", off, sr[0] - 1e-9, sr[1] + 1e-9)
            prof("ST1", 2 * a + w, f"ST1 joint {a} {'upper' if w else 'lower'} offset (deg)", off, tol=1e-9,
                 nominal=False)
            sp = spec("ST1", 2 * a + w)
            if sp is not None and sp.mode == "nominal":
                n[0] += 1
                if p.stops[a][w] != float(nom[w]):
                    bad.append(f"ST1 joint {a} stop {w}: the profile keeps it nominal")
    tr, rr = qv("PL1", "translation_range_m"), qv("PL1", "rotation_range_deg")
    ys = pub["conventions"]["base_link_y_m"]["value"]
    for arm in (0, 1):
        pos, quat = p.base_poses[arm]
        nom = (0.0, float(ys[SIDES[arm]]), 0.0)
        for k in range(3):
            inside(f"PL1 {SIDES[arm]} translation {k}", pos[k] - nom[k], tr[0] - 1e-12, tr[1] + 1e-12)
            prof("PL1.translation", 3 * arm + k, f"PL1 {SIDES[arm]} translation {k}", pos[k] - nom[k], tol=1e-12)
        ang = _xyz_of_quat(quat)
        for k in range(3):
            inside(f"PL1 {SIDES[arm]} rotation {'xyz'[k]} (deg)", ang[k] / DEG, rr[0] - 1e-9, rr[1] + 1e-9)
            prof("PL1.rotation", 3 * arm + k, f"PL1 {SIDES[arm]} rotation {'xyz'[k]} (deg)", ang[k] / DEG, tol=1e-9)
        if arm == 0:
            n[0] += 1
            if pos != nom or quat != (1.0, 0.0, 0.0, 0.0):
                bad.append("PL1: the law keeps the left arm at the file's pose")
    if p.inertials:
        nom = maker_inertials()
        mr, cr = qv("P4", "mass_range_rel"), qv("P4", "com_range_m")
        for bi in p.inertials:
            m0, ip0, _, d0 = nom[bi.body]
            inside(f"P4 {bi.body} mass rel", bi.mass / m0 - 1.0, mr[0] - 1e-12, mr[1] + 1e-12)
            for k in range(3):
                inside(f"P4 {bi.body} com {k}", bi.pos[k] - ip0[k], cr[0] - 1e-12, cr[1] + 1e-12)
            d = bi.diaginertia
            n[0] += 1
            if not (d[0] + d[1] >= d[2] and d[1] + d[2] >= d[0] and d[0] + d[2] >= d[1]):
                bad.append(f"P4 {bi.body}: principal inertias violate the triangle inequality")
    if profile is not None:
        n[0] += 1
        if profile.special.get("P4") != "draw" and p.inertials:
            bad.append(f"P4 (profile): the profile keeps the inertials nominal, the unit carries {len(p.inertials)} "
                       "deviation(s)")
    nmax = int(qv("P5", "items_per_arm")[1])
    for side in SIDES:
        k_ = sum(1 for m in p.mounts if m.name.startswith(f"mount_{side}_"))
        inside(f"P5 items on the {side} arm", k_, 0, nmax)
    if profile is not None:
        p5 = profile.special.get("P5")
        n[0] += 1
        if not p5:
            if p.mounts:
                bad.append(f"P5 (profile): the profile mounts nothing, the unit carries {[m.name for m in p.mounts]}")
        elif not any(tuple(p.mounts) == tuple(p5["items"][side]) for side in SIDES):
            bad.append(f"P5 (profile): the mounts {[m.name for m in p.mounts]} are not the items of one side of the "
                       "profile")
    for i in range(16):
        inside(f"C3 module {i}", c.k[i], *qv("C3", "range"))
        prof("C3", i, f"C3 module {i}", c.k[i])
        inside(f"TH1 module {i}", state.thermal.h[i], *qv("TH1", "range"))
        prof("TH1", i, f"TH1 module {i}", state.thermal.h[i])
        inside(f"C0 module {i}", c.zero_before[i], *qv("C0", "range_rad"))
        prof("C0", i, f"C0 module {i}", c.zero_before[i], tol=1e-12)
        n[0] += 1
        if c.sign[i] != 1:
            bad.append(f"module {i}: sign {c.sign[i]} (+1 on every unit)")
    for a in range(14):
        i = arm_module(a)
        inside(f"C4 module {i}", c.b[i], *qv("C4", "range_Nm")[_c4_group(a)])
        prof("C4", a, f"C4 module {i}", c.b[i])
        if c.zero_source == "residual":
            inside(f"C1 residual module {i} (deg)", c.delta[i] / DEG, qv("C1", "residual_range_deg")[0] - 1e-9,
                   qv("C1", "residual_range_deg")[1] + 1e-9)
            prof("C1", a, f"C1 residual module {i} (deg)", c.delta[i] / DEG, tol=1e-9)
    for g in GRIPPER_MODULES:
        n[0] += 1
        if c.b[g] != 0.0:
            bad.append(f"C4 module {g}: a gripper carries no offset")
    if c.zero_source == "as_drawn" and tuple(c.delta) != tuple(c.zero_before):
        bad.append("C1: before the routine the zero in force is the zero stored before it")
    regs = c.registers
    spread = qv("registers_identification_spread", "range")
    c2 = [float(x) for x in qv("C2", "values")]
    c5t = [int(x) for x in qv("C5", "timeout_values_counts")]
    c5p = None
    if profile is not None:
        c5p = profile.special.get("C5") or {"TIMEOUT": int(qv("C5", "nominal")["TIMEOUT"]), "gripper_mode": None}
    for k_, cls in enumerate(CLASSES):
        disc("C6", k_, f"C6 {cls} edition", state.editions[cls], _nominal_edition(cls))
    for i, m in enumerate(pub["conventions"]["modules"]["value"]):
        ed = state.editions[m["class"]]
        n[0] += 1
        if ed not in qv("C6", "values").get(m["class"], [_nominal_edition(m["class"])]):
            bad.append(f"C6 module {i}: edition {ed}")
        imax_vals = [float(x) for x in qv("C7", "values").get(ed, [pub["editions"][ed]["Imax_A"]["value"]])]
        n[0] += 1
        if not any(c.imax[i] == x for x in imax_vals):
            bad.append(f"C7 module {i}: {c.imax[i]} not in {imax_vals}")
        disc("C7", i, f"C7 module {i}", c.imax[i], float(pub["editions"][ed]["Imax_A"]["value"]))
        if m["class"] == "DM4340":
            disc("C2", i, f"C2 module {i} VMAX", regs.get(i, rg.VMAX), Q["C2"]["nominal"](i))
        if c5p is not None:
            n[0] += 1
            if regs.get(i, rg.TIMEOUT) != c5p["TIMEOUT"]:
                bad.append(f"C5 module {i} (profile): TIMEOUT {regs.get(i, rg.TIMEOUT)}, the profile sets "
                           f"{c5p['TIMEOUT']}")
            gm = c5p.get("gripper_mode")
            if m["joint"] == 8 and gm is not None:
                n[0] += 1
                want = rg.MODE_MIT if gm == "MIT" else rg.MODE_POS_FORCE
                if regs.get(i, rg.CTRL_MODE) != want:
                    bad.append(f"C5 module {i} (profile): gripper mode {regs.get(i, rg.CTRL_MODE)}, the profile "
                               f"sets {gm}")
        cat = rg.catalogue_registers(ed, m["command_id"], m["reply_id"], regs.get(i, rg.CTRL_MODE),
                                     regs.get(i, rg.TIMEOUT), vmax=regs.get(i, rg.VMAX))
        for rid, v in cat.items():
            got = regs.get(i, rid)
            if rid in (rg.FLUX, rg.RS, rg.LS):
                inside(f"register {rid} module {i} rel", got / v - 1.0, spread[0] - 1e-12, spread[1] + 1e-12)
                prof("registers_identification_spread", 3 * i + (rg.FLUX, rg.RS, rg.LS).index(rid),
                     f"register {rid} module {i} rel", got / v - 1.0, tol=1e-12)
            else:
                n[0] += 1
                if got != v:
                    bad.append(f"register {rid} module {i}: {got} against the catalogue's {v}")
        n[0] += 2
        if m["class"] == "DM4340" and regs.get(i, rg.VMAX) not in c2:
            bad.append(f"C2 module {i}: VMAX {regs.get(i, rg.VMAX)}")
        if regs.get(i, rg.TIMEOUT) not in c5t:
            bad.append(f"C5 module {i}: TIMEOUT {regs.get(i, rg.TIMEOUT)}")
        if m["joint"] != 8 and regs.get(i, rg.CTRL_MODE) != rg.MODE_MIT:
            bad.append(f"C5 module {i}: an arm module runs in MIT mode")
    hs = state.hs
    r_law = float(qv("HS1", "relative_rms"))
    n[0] += 1
    if hs.r not in (0.0, r_law):
        bad.append(f"HS1 r = {hs.r} (the law's {r_law} or off)")
    if profile is not None:
        r_prof = 0.0 if is_constructed(state) else float(profile.special.get("HS1.relative_rms", 0.0))
        n[0] += 1
        if hs.r != r_prof:
            bad.append(f"HS1 r (profile) = {hs.r}, the profile's is {r_prof}")
    if hs.r > 0.0:
        pr, tb = qv("HS1", "period_range_rad"), qv("HS1", "table_bounds")
        for a in range(14):
            inside(f"HS1 period {a}", hs.period[a], *pr)
            prof("HS1.period", a, f"HS1 period {a}", hs.period[a])
            inside(f"HS1 table {a} min", min(hs.tables[a]), tb[0], tb[1])
            inside(f"HS1 table {a} max", max(hs.tables[a]), tb[0], tb[1])
    for s in sessions:
        inside(f"S3 ambient {s.session_id}", s.ambient_C, *qv("S3", "ambient_range_C"))
        prof("S3.ambient", 0, f"S3 ambient {s.session_id}", s.ambient_C)
        for i in range(16):
            inside(f"S1 {s.session_id} module {i}", s.friction_factor[i], *qv("S1", "range"))
            prof("S1", i, f"S1 {s.session_id} module {i}", s.friction_factor[i])
            for w, t in enumerate(s.start_temperatures_C[i]):
                inside(f"S3 start offset {s.session_id} module {i} node {w}", t - s.ambient_C,
                       qv("S3", "start_offset_range_K")[0] - 1e-9, qv("S3", "start_offset_range_K")[1] + 1e-9)
                prof("S3.start_offset", i, f"S3 start offset {s.session_id} module {i}", t - s.ambient_C, tol=1e-9)
        for a in range(14):
            i = arm_module(a)
            inside(f"S2 {s.session_id} module {i}", s.b_step[i], *qv("S2", "range_Nm")[_c4_group(a)])
            prof("S2", a, f"S2 {s.session_id} module {i}", s.b_step[i])
        for g in GRIPPER_MODULES:
            n[0] += 1
            if s.b_step[g] != 0.0:
                bad.append(f"S2 {s.session_id}: a gripper carries no step")
    if bad:
        raise LawError(f"{len(bad)} value(s) outside the law or the profile: " + "; ".join(bad[:20]))
    return n[0]


def spread_distance(qid, ua, ub):
    if qid == "C1":
        return max(abs(ua.cal.delta[i] - ub.cal.delta[i]) for i in ARM_MODULES) / DEG
    if qid == "P1":
        return max(max(ua.plant.frictionloss[i] / ub.plant.frictionloss[i],
                       ub.plant.frictionloss[i] / ua.plant.frictionloss[i]) - 1.0 for i in ARM_MODULES)
    if qid == "PL1":
        return max(abs(ua.plant.base_poses[1][0][k] - ub.plant.base_poses[1][0][k]) for k in range(3))
    if qid == "C3":
        return max(abs(ua.cal.k[i] - ub.cal.k[i]) for i in range(16))
    if qid == "TH1":
        return max(abs(ua.thermal.h[i] - ub.thermal.h[i]) for i in range(16))
    raise LawError(f"no spread metric for {qid}")


def spread_check(units, profile):
    rule = profile.raw.get("spread_rule") or {}
    wmin = rule.get("w_min") or {}
    unset = [q for q, v in wmin.items() if v is None]
    if unset:
        raise LawError(f"spread rule: w_min unset for {unset} in the profile: the rule cannot be applied")
    need = int(_v(law_params(), "rules", "spread_rule_stage05").get("w_min_in_at_least", 2))
    detail, ok = [], True
    for a in range(len(units)):
        for b in range(a + 1, len(units)):
            d = {q: spread_distance(q, units[a], units[b]) for q in wmin}
            hits = [q for q in wmin if d[q] >= float(wmin[q])]
            detail.append({"pair": [a, b], "distance": d, "at_least_w_min": hits})
            ok &= len(hits) >= need
    return ok, detail


def _arm_hinge_rows():
    from . import base
    return [k for k, nm in enumerate(base.HINGES) if "finger" not in nm]


def sanity(state, session, law=None, nominal=False, hold_s=None, lattice_points=3):
    import mujoco
    from .truth import STOP_EVENT_THRESHOLD_NM
    from .unit import OpenArmUnit, UnitError, UnitInvalid
    law = law_params() if law is None else law
    rules = law["rules"]
    hold = float(_v(rules, "sanity_home_hold_s")) if hold_s is None else float(hold_s)
    rep = {"reasons": [], "holds": {}}
    for bi in state.plant.inertials + tuple(state.plant.mounts):
        d = bi.diaginertia
        if not (d[0] + d[1] >= d[2] and d[1] + d[2] >= d[0] and d[0] + d[2] >= d[1]) or bi.mass <= 0.0:
            rep["reasons"].append(f"inertial of {getattr(bi, 'body', getattr(bi, 'name', '?'))} is not physical")
    if rep["reasons"]:
        return False, rep
    lp = LawPlugins(law)
    arm_rows = _arm_hinge_rows()
    from . import base
    for gs in ("A", "B"):
        try:
            u = OpenArmUnit.build(state, session, law=lp, contacts=True)
        except Exception as exc:
            rep["reasons"].append(f"does not build: {exc!r}")
            return False, rep
        if "gravity_Nm" not in rep:
            d0 = mujoco.MjData(u.model)
            d0.qpos[:] = u.data.qpos
            d0.qvel[:] = 0.0
            mujoco.mj_forward(u.model, d0)
            rep["gravity_Nm"] = [float(x) for x in d0.qfrc_bias[u.arm_dadr]]
            rep["peak_Nm"] = [float(u.t_peak[i]) for i in ARM_MODULES]
        r = {}
        try:
            u.reset(hold=HostProfile.default(gs), hold_s=hold)
        except UnitInvalid as exc:
            rep["reasons"].append(f"gain set {gs}: the plant cannot integrate physics step {exc.step} ({exc.kind})")
            continue
        except UnitError as exc:
            rep["reasons"].append(f"gain set {gs}: {exc}")
        ev = u.events()
        faults = [e.to_dict() for e in ev if e.kind == "fault_set"]
        stops = [e.to_dict() for e in ev if e.kind == "stop_begin" and "finger" not in e["joint"]]
        conts = [e.to_dict() for e in ev if e.kind == "contact_begin"]
        if faults:
            rep["reasons"].append(f"gain set {gs}: {len(faults)} fault(s) in the home hold: {faults[:3]}")
        if stops:
            rep["reasons"].append(f"gain set {gs}: an arm joint on a stop in the home hold: {stops[:3]}")
        if conts:
            rep["reasons"].append(f"gain set {gs}: a contact in the home hold: {conts[:3]}")
        t = u.truth(("T_del", "delivered", "T_w", "contacts", "stop_torque"))
        cts = dict(t["contacts"])
        if cts:
            rep["reasons"].append(f"gain set {gs}: a contact at the end of the home hold: {sorted(cts)[:3]}")
        stq = np.asarray(t["stop_torque"], dtype=np.float64)
        on = [f"{base.HINGES[k]} {'upper' if w else 'lower'}" for k in arm_rows for w in (0, 1)
              if abs(float(stq[k, w])) > STOP_EVENT_THRESHOLD_NM]
        if on:
            rep["reasons"].append(f"gain set {gs}: an arm joint on a stop at the end of the home hold: {on[:3]}")
        tp = np.asarray(u.t_peak, dtype=np.float64)
        r["torque_share_of_peak"] = float(np.max(np.abs(np.asarray(t["T_del"])) / tp))
        r["held_Nm"] = [float(t["delivered"][i]) for i in ARM_MODULES]
        r["friction_level_Nm"] = [float(x) for x in u.model.dof_frictionloss[u.arm_dadr]]
        r["winding_max_C"] = float(np.max(np.asarray(t["T_w"])))
        rep["holds"][gs] = r
    if nominal:
        rep["nominal_assertions"] = nominal_assertions(rep, law)
    if state.plant.mounts:
        lat = mount_lattice_report(state, session, lattice_points)
        rep["mount_lattice"] = {k: v for k, v in lat.items() if k != "hits"}
        rep["mount_contacts"] = lat["hits"]
    return not rep["reasons"], rep


def nominal_assertions(rep, law=None):
    law = law_params() if law is None else law
    rules = law["rules"]
    share = float(_v(rules, "assertion_home_gravity_torque_of_peak"))
    slack = float(_v(rules, "assertion_home_hold_slack_Nm"))
    tw = float(_v(rules, "assertion_home_winding_C"))
    g, tp = rep["gravity_Nm"], rep["peak_Nm"]
    bad, out = [], {"gravity_share_of_peak": [abs(x) / p for x, p in zip(g, tp)], "hold_excess_Nm": {}}
    for a, s in enumerate(out["gravity_share_of_peak"]):
        if not s <= share:
            bad.append(f"arm joint {a}: gravity {g[a]:.4f} N m is {s:.4f} of the peak {tp[a]} (at most {share})")
    for gs, r in rep["holds"].items():
        ex = [abs(h - x) - (f + slack) for h, x, f in zip(r["held_Nm"], g, r["friction_level_Nm"])]
        out["hold_excess_Nm"][gs] = ex
        for a, e in enumerate(ex):
            if not e <= 0.0:
                bad.append(f"gain set {gs}, arm joint {a}: held {r['held_Nm'][a]:.4f} N m against gravity "
                           f"{g[a]:.4f}, beyond the friction level {r['friction_level_Nm'][a]} + {slack} by {e:.4f}")
        if not r["winding_max_C"] <= tw:
            bad.append(f"gain set {gs}: home winding {r['winding_max_C']:.3f} deg C above {tw}")
    if bad:
        raise AssertionError("nominal unit: " + "; ".join(bad))
    return out


def _mount_lattice(state, session, points=3):
    import itertools
    import mujoco
    from . import base
    from .unit import OpenArmUnit, plant_edits
    m = base.unit_spec(plant_edits(state, session), contacts=True).compile()
    d = mujoco.MjData(m)
    hits, poses = [], {}
    dflt = OpenArmUnit.default_pose()
    jadr = {(s, j): int(m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"openarm_{s}_joint{j}")])
            for s in SIDES for j in range(1, 8)}
    finger_pair = {s: frozenset(mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"openarm_{s}_ee_{f}_finger")
                                for f in ("inner", "outer")) for s in SIDES}
    all_mount_geoms = {g for g in (mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, mm.name + "_geom")
                                   for mm in state.plant.mounts if mm.geom_type is not None) if g >= 0}
    mount_bodies = {int(m.geom_bodyid[g]) for g in all_mount_geoms}
    for side in SIDES:
        geoms = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, mm.name + "_geom") for mm in state.plant.mounts
                 if mm.name.startswith(f"mount_{side}_") and mm.geom_type is not None]
        geoms = {g for g in geoms if g >= 0}
        if not geoms:
            continue
        arm = SIDES.index(side)
        other = SIDES[1 - arm]
        grid = []
        for a in range(7):
            lo, hi = state.plant.stops[arm * 7 + a]
            grid.append([lo + (hi - lo) * k / (points - 1) for k in range(points)])
        counted = skipped = 0
        for combo in itertools.product(*grid):
            d.qpos[:] = m.qpos0
            for j in range(1, 8):
                d.qpos[jadr[(other, j)]] = dflt[8 * (1 - arm) + j - 1]
                d.qpos[jadr[(side, j)]] = combo[j - 1]
            mujoco.mj_fwdPosition(m, d)
            own = False
            for c in d.contact[:d.ncon]:
                g1, g2 = int(c.geom1), int(c.geom2)
                if g1 in all_mount_geoms or g2 in all_mount_geoms:
                    continue
                pair = frozenset((int(m.geom_bodyid[g1]), int(m.geom_bodyid[g2])))
                if not any(pair == fp for fp in finger_pair.values()):
                    own = True
                    break
            if own:
                skipped += 1
                continue
            counted += 1
            for c in d.contact[:d.ncon]:
                g1, g2 = int(c.geom1), int(c.geom2)
                if (g1 in geoms or g2 in geoms) and not (int(m.geom_bodyid[g1]) in mount_bodies and
                                                         int(m.geom_bodyid[g2]) in mount_bodies):
                    oth, mine = (g2, g1) if g1 in geoms else (g1, g2)
                    hits.append({"side": side, "q": [float(x) for x in combo],
                                 "geom": mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, oth),
                                 "_ids": (mine, oth), "_model": m})
                    break
        poses[side] = {"lattice": points ** 7, "counted": counted, "not_counted_bare_contact": skipped}
    return hits, poses


def mount_clearance(state, session, points=3):
    hits, _ = _mount_lattice(state, session, points)
    return [{"side": h["side"], "q": h["q"], "geom": h["geom"]} for h in hits]


BARE_ARM_BODIES = ("link5", "link6", "ee_base_link", "ee_inner_finger", "ee_outer_finger")
BARE_DISTANCE_MAX_M = 0.2


def mount_lattice_report(state, session, points=3):
    from dataclasses import replace
    import mujoco
    from . import base
    from .unit import OpenArmUnit, plant_edits
    hits, poses = _mount_lattice(state, session, points)
    out = []
    if hits:
        bare = replace(state, plant=replace(state.plant, mounts=()))
        mb = base.unit_spec(plant_edits(bare, session), contacts=True).compile()
        db = mujoco.MjData(mb)
        dflt = OpenArmUnit.default_pose()
        ft = np.zeros(6)
        for h in hits:
            m = h["_model"]
            mine, oth = h["_ids"]
            ob = int(m.geom_bodyid[oth])
            bname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, ob)
            bb = mujoco.mj_name2id(mb, mujoco.mjtObj.mjOBJ_BODY, bname)
            g2 = int(mb.body_geomadr[bb]) + (oth - int(m.body_geomadr[ob]))
            side = h["side"]
            db.qpos[:] = mb.qpos0
            for s in SIDES:
                for j in range(1, 8):
                    ja = int(mb.jnt_qposadr[mujoco.mj_name2id(mb, mujoco.mjtObj.mjOBJ_JOINT, f"openarm_{s}_joint{j}")])
                    db.qpos[ja] = h["q"][j - 1] if s == side else dflt[8 * SIDES.index(s) + j - 1]
            mujoco.mj_fwdPosition(mb, db)
            near = {mujoco.mj_name2id(mb, mujoco.mjtObj.mjOBJ_BODY, f"openarm_{side}_{b}") for b in BARE_ARM_BODIES}
            near.discard(bb)
            best = BARE_DISTANCE_MAX_M
            for g in range(mb.ngeom):
                if int(mb.geom_bodyid[g]) in near and (mb.geom_contype[g] or mb.geom_conaffinity[g]):
                    best = min(best, float(mujoco.mj_geomDistance(mb, db, g, g2, BARE_DISTANCE_MAX_M, ft)))
            mg = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, mine)
            out.append({"side": side, "q": h["q"], "q_deg": [x * 180.0 / math.pi for x in h["q"]], "geom": h["geom"],
                        "mount_geom": mg, "pair": [mg, h["geom"]], "bare_arm_distance_m": best})
    return {"points": int(points), "poses": poses, "n_hits": len(out), "rejects": False, "hits": out}


class RoutineError(ValueError):
    ""


BUMP_FIELDS = ("joint", "side", "stop_active", "stop_torque_Nm", "reported_torque_Nm", "threshold_Nm", "other_contact")


def zero_by_routine(state, session, routine, law=None):
    if state.cal.zero_source != "as_drawn":
        raise LawError("the routine runs on a unit whose zero in force is the zero stored before it ('as_drawn')")
    lp = LawPlugins(law)
    a = routine(state, session, lp)
    b = routine(state, session, lp)
    for res in (a, b):
        for k in ("zeros", "bumps", "faults", "digest"):
            if k not in res:
                raise RoutineError(f"the routine's result lacks {k!r}")
        for bump in res["bumps"]:
            miss = [k for k in BUMP_FIELDS if k not in bump]
            if miss:
                raise RoutineError(f"a bump of the routine's result lacks {miss}: {bump}")
            if bump["side"] not in ("lower", "upper"):
                raise RoutineError(f"a bump's side is 'lower' or 'upper', not {bump['side']!r}")
            want = BUMPED_STOPS.get(bump["joint"])
            if want is not None and bump["side"] != want:
                raise RoutineError(f"the bump of {bump['joint']} is on its {bump['side']} stop; the maker's script "
                                   f"bumps the {want} one")
        arm = [bp["joint"] for bp in res["bumps"] if bp["joint"] in BUMPED_STOPS]
        if len(arm) != len(set(arm)):
            raise RoutineError("an arm joint is bumped twice")
        if not res["faults"] and set(arm) != set(BUMPED_STOPS):
            raise RoutineError(f"a run without a fault bumps every arm joint; missing "
                               f"{sorted(set(BUMPED_STOPS) - set(arm))}")
    why, sides = [], []
    for bump in a["bumps"]:
        nm = f"{bump['joint']} ({bump['side']})"
        if bump["stop_active"] is not True:
            why.append(f"bump of {nm}: its stop is not active at the detection")
        if not abs(float(bump["reported_torque_Nm"])) > float(bump["threshold_Nm"]):
            why.append(f"bump of {nm}: the reported torque {bump['reported_torque_Nm']} did not pass the threshold "
                       f"{bump['threshold_Nm']}")
        if bump["other_contact"] is not False:
            why.append(f"bump of {nm}: another contact of the arm is active at the detection")
        sides.append({"joint": bump["joint"], "side": bump["side"], "stop_torque_Nm": float(bump["stop_torque_Nm"]),
                      "reported_torque_Nm": float(bump["reported_torque_Nm"]),
                      "threshold_Nm": float(bump["threshold_Nm"])})
    if a["faults"]:
        why.append(f"{len(a['faults'])} fault(s) in the routine")
    if a["digest"] is None or a["digest"] != b["digest"]:
        why.append("the routine does not repeat to the same digest")
    if [float(x) for x in a["zeros"]] != [float(x) for x in b["zeros"]]:
        why.append("the routine's zeros differ between two runs")
    if why:
        raise LawError("rejected: " + "; ".join(why))
    zeros = [float(x) for x in a["zeros"]]
    if len(zeros) != N_MODULES:
        raise RoutineError("the routine writes 16 zeros")
    z = state.with_zeros(zeros, "routine")
    return z, {"digest": a["digest"], "bumps": len(a["bumps"]), "bump_detail": sides, "as_zeroed_digest": z.digest,
               "sides_as_the_script": {s["joint"]: s["side"] for s in sides if s["joint"] in BUMPED_STOPS}
               == BUMPED_STOPS}


def bracket(profile, end, law=None, inertials=None):
    law = law_params() if law is None else law
    blk = (profile.raw.get("bracket") or {}).get(end)
    if not isinstance(blk, dict):
        raise LawError(f"the profile has no bracket end {end!r}")
    Q = _quantities(law)
    pub = public_params()

    def eff(qid, e):
        sp = profile.spec(qid, e)
        if sp.mode == "draw" and sp.lo is not None:
            return sp.lo, sp.hi
        return Q[qid]["range"](e)

    def at_end(qid, val, elements):
        out = []
        for e in elements:
            lo, hi = eff(qid, e)
            if isinstance(val, str):
                t = val.lower()
                if "low end" in t:
                    out.append(lo)
                elif "high end" in t:
                    out.append(hi)
                else:
                    raise LawError(f"bracket {end} {qid}: {val!r} names no end")
            else:
                if float(val) not in (float(lo), float(hi)):
                    raise LawError(f"bracket {end} {qid}: {val} is not an end of the range [{lo}, {hi}]")
                out.append(float(val))
        return out

    used, carried = [], {}
    fl = [0.0] * 16
    nomP1 = [float(x) for x in _v(law, "quantities", "P1", "nominal_Nm")]
    for arm in (0, 1):
        for j in range(7):
            fl[arm * 8 + j] = nomP1[j]
        fl[GRIPPER_MODULES[arm]] = float(_v(law, "quantities", "P6", "nominal_Nm"))
    k = [1.0] * 16
    h = [1.0] * 16
    amb = float(_v(law, "quantities", "S3", "ambient_nominal_C"))
    offs = [0.0] * 16
    for name, val in blk.items():
        if name == "others":
            continue
        if name == "P1":
            lv = at_end("P1", val, range(7))
            for arm in (0, 1):
                for j in range(7):
                    fl[arm * 8 + j] = lv[j]
        elif name == "P6":
            lv = at_end("P6", val, (0, 1))
            for arm in (0, 1):
                fl[GRIPPER_MODULES[arm]] = lv[arm]
        elif name == "C3":
            k = at_end("C3", val, range(16))
        elif name == "TH1":
            h = at_end("TH1", val, range(16))
        elif name == "S3.ambient":
            amb = at_end("S3.ambient", val, (0,))[0]
        elif name == "S3.start_offset":
            offs = at_end("S3.start_offset", val, range(16))
        else:
            carried[name] = val
            continue
        used.append(name)
    jr = pub["conventions"]["joint_ranges_rad"]["value"]
    stops = [tuple(float(x) for x in jr[SIDES[a // 7]][a % 7]) for a in range(14)]
    ys = pub["conventions"]["base_link_y_m"]["value"]
    bases = tuple(((0.0, float(ys[s]), 0.0), (1.0, 0.0, 0.0, 0.0)) for s in SIDES)
    dm = [0.0] * 16
    nomP2 = [float(x) for x in _v(law, "quantities", "P2", "nominal_Nm_s_rad")]
    for a in range(14):
        dm[arm_module(a)] = nomP2[a % 7]
    for g in GRIPPER_MODULES:
        dm[g] = float(_v(law, "quantities", "P7", "nominal_Nm_s_rad"))
    c5 = profile.special.get("C5") or {"TIMEOUT": 0, "gripper_mode": None}
    gmode = rg.MODE_MIT if c5.get("gripper_mode") == "MIT" else rg.MODE_POS_FORCE
    regs = rg.nominal_register_set(gripper_mode=gmode, timeout=int(c5.get("TIMEOUT", 0)))
    eds = dict(pub["class_nominal_edition"]["value"])
    imax = [float(pub["editions"][eds[module_class(i)]]["Imax_A"]["value"]) for i in range(16)]
    cal = CalState([0.0] * 16, [0.0] * 16, "nominal", [1] * 16, k, [0.0] * 16, imax, regs)
    plant = PlantState(fl, dm, (), (), stops, bases)
    st = UnitState(f"constructed-{end}", law["law_version"], eds, plant, cal, hsmod.off_state(), ThermalState(h))
    ses = SessionState.from_offsets(f"constructed-{end}", [1.0] * 16, [0.0] * 16, amb, offs)
    check_unit(st, [ses], profile, law)
    body = {"format": BRACKET_FORMAT, "kind": "constructed", "end": end, "profile": profile.name,
            "profile_sha256": profile.sha256, "law_version": law["law_version"], "quantities_at_ends": sorted(used),
            "carried": carried, "others": "nominal (every quantity the bracket does not name; friction profile off)",
            "unit": st.to_dict(), "session": ses.to_dict()}
    body["digest"] = hashlib.sha256(canonical_json(body).encode("ascii")).hexdigest()
    return body, st, ses


def is_constructed(state_or_record):
    if isinstance(state_or_record, UnitState):
        return state_or_record.unit_id.startswith("constructed-")
    return isinstance(state_or_record, dict) and state_or_record.get("kind") == "constructed"
