"""The base: the maker's v2 files vendored with sha256, the loader, the forced numerics. PUBLIC.

PUBLIC: a public runner of a submitted model loads the maker's model and the forced numerics through this module; it
holds nothing of the law of drawing, the registry, the truth channel or the recorder, and imports none of them.

The base, the plant model, the in-drive law's carrier, the frames and the tool point, the arithmetic rules and the
numerics. Every other module reads the maker's model only through this module.

What is vendored: ``github.com/enactic/openarm_mujoco`` folder ``v2/`` at commit
``ed2046f30981398fc610688f08beac0ebd08f972``: ``pedestal/pedestal.xml`` (the scene), ``pedestal/openarm_pedestal.xml``,
``openarm_bimanual_model.xml``, ``openarm_bimanual_common.xml``, ``components/left_arm.xml``,
``components/right_arm.xml`` and ``assets/``, plus the repository's ``LICENSE``. They live under ``base/`` with the
checkout's relative paths (``base/v2/...``, ``base/LICENSE``), byte for byte. ``base/MANIFEST.json`` pins every file by
sha256. The files under ``base/**/assets/`` (the maker's meshes) are restored by ``base/restore.py`` (or ``restore_base``
below) from a maker checkout and checked against the manifest.

Two loaders:

- ``maker_spec()``: the maker's scene exactly as the files give it (an ``MjSpec``, its attachments resolved by MuJoCo).
- ``unit_spec(plant)``: the plant model of a unit: the maker's scene, the forced numerics
  applied to every model alike, the sixteen actuators as ``general`` actuators that carry the module's delivered
  torque scale k and constant offset b (``gainprm[0] = k``, ``biasprm = [b, 0, 0]``, gear 1, no control or force range),
  no joint-side actuator force range on any hinge (below), and the unit's own values (``plant``, format
  below). This is the BUILD PATH that the unit-file writer (``unitfile.py``) must reproduce bit for bit (every
  compiled array equal).

The clamp acts before the scale: a module limits what it COMMANDS (its current limit, its
torque-speed envelope and its peak torque, in the drive's own units); the joint receives k x (the clamped command) + b.
The clamp therefore lives in the bus layer (the actuator's force range, rewritten at run time), and the maker's joint
``actuatorfrcrange`` (40 / 27 / 7 N m), which would act after k, is removed from every hinge: ``jnt_actfrclimited = 0``
and ``jnt_actfrcrange = [0, 0]`` on all eighteen (the sixteen driven hinges and the two coupled fingers, which carry no
actuator; a unit file then carries no ``actuatorfrcrange`` at all).

``plant=None`` keeps the maker file's joint values (frictionloss, damping, the finger damping 0.6, ranges, inertials):
it is the FILE's values under the unit's numerics, not the nominal unit. The nominal unit, like every
drawn unit, is made by mapping a ``UnitState`` onto the plant edits below.

The plant edits (a plain mapping; every key optional; values are float64, floats are never rounded here):

    {"joints":     {<hinge name>: {"range": [lo, hi], "frictionloss": f, "damping": c}},   # range: arm joints only
     "inertials":  {<body name>: {"mass": m, "pos": [3], "quat": [4], "diaginertia": [3]}},  # all four together
     "base":       {"left" | "right": {"pos": [3], "quat": [4]}},          # openarm_<side>_base_link, in
                                                                           # the frame of arm_origin, as the file
     "actuators":  {<actuator name>: {"k": k, "b": b}},
     "mounts":     [{"name": "mount_<side>_<1..4>", "parent": <a body of the same arm>, "pos": [3], "quat": [4],
                     "inertial": {"mass", "pos", "quat", "diaginertia"},
                     "geom": None | {"type": "sphere"|"capsule"|"ellipsoid"|"cylinder"|"box", "size": [...],
                                     "pos": [3], "quat": [4]}}]}

This mapping is the smallest one that carries every quantity that is set per unit and that the MJCF can state. A
unit is made by mapping a ``UnitState`` onto it.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Mapping

# ---------------------------------------------------------------------------------------------------------------
# Constants.
# ---------------------------------------------------------------------------------------------------------------

PACKAGE_DIR = Path(__file__).resolve().parent
UNIT_DIR = PACKAGE_DIR.parent
BASE_DIR = UNIT_DIR / "base"
MANIFEST_NAME = "MANIFEST.json"

MAKER_REPOSITORY = "github.com/enactic/openarm_mujoco"
MAKER_COMMIT = "ed2046f30981398fc610688f08beac0ebd08f972"
MAKER_FOLDER = "v2"
MAKER_LICENSE = "LICENSE"                                         # the maker has a LICENSE file, no NOTICE
VENDORED_XML = (                                                  # the maker's files vendored
    "v2/pedestal/pedestal.xml",
    "v2/pedestal/openarm_pedestal.xml",
    "v2/openarm_bimanual_model.xml",
    "v2/openarm_bimanual_common.xml",
    "v2/components/left_arm.xml",
    "v2/components/right_arm.xml",
)
VENDORED_TREES = ("v2/assets",)                                   # the maker's mesh folder
SCENE = "v2/pedestal/pedestal.xml"                                # the scene

MUJOCO_VERSION = "3.14.0"                                         # MuJoCo 3.14.0, pinned
UNIT_MODEL_NAME = "openarm_unit"

SIDES = ("left", "right")                                         # one bus per arm, left first
ARM_JOINTS = tuple(f"openarm_{s}_joint{k}" for s in SIDES for k in range(1, 8))                  # 14
FINGER_JOINTS = tuple(f"openarm_{s}_finger_joint{k}" for s in SIDES for k in (1, 2))           # 4
HINGES = tuple(f"openarm_{s}_{j}" for s in SIDES
               for j in [f"joint{k}" for k in range(1, 8)] + ["finger_joint1", "finger_joint2"])  # 18, maker order
# Sixteen modules, index i = 0..15: left J1..J7, left J8 (gripper), right J1..J7, right J8.
ACTUATORS = tuple(pair for s in SIDES for pair in (
    [(f"{s}_joint{k}_ctrl", f"openarm_{s}_joint{k}") for k in range(1, 8)]
    + [(f"{s}_finger1_ctrl", f"openarm_{s}_finger_joint1")]))
ARM_BODIES = {s: (f"openarm_{s}_base_link",) + tuple(f"openarm_{s}_link{k}" for k in range(1, 7))
              + (f"openarm_{s}_ee_base_link", f"openarm_{s}_ee_inner_finger", f"openarm_{s}_ee_outer_finger")
              for s in SIDES}                                     # the twenty bodies
# Inertials as built: per arm link1..link6, ee_base_link and the two fingers.
INERTIAL_BODIES = tuple(b for s in SIDES for b in ARM_BODIES[s][1:])
BASE_BODIES = {s: f"openarm_{s}_base_link" for s in SIDES}        # where each arm is bolted (PL1)
EE_BODIES = {s: f"openarm_{s}_ee_base_link" for s in SIDES}
TOOL_OFFSET = (0.0, 0.0, -0.135)       # tool point = ee_base_link origin + (0, 0, -0.135) m in its frame
ARM_ORIGIN_SITE = "arm_origin"

# Forced numerics, on every model alike.
FRICTION_SOLREF = (0.002, 1.0)                        # solreffriction "0.002 1"
FRICTION_SOLIMP = (0.9999, 0.9999, 0.001, 0.5, 2.0)   # solimpfriction "0.9999 0.9999 0.001 0.5 2"
LIMIT_SOLREF = (0.002, 1.0)                           # solreflimit "0.002 1"
LIMIT_SOLIMP = (0.99, 0.999, 0.001, 0.5, 2.0)         # solimplimit "0.99 0.999 0.001 0.5 2"
OPTION = {                                            # the maker's option block plus the solver settings
    "timestep": 0.001, "integrator": "implicitfast", "cone": "elliptic", "impratio": 10.0,
    "solver": "Newton", "iterations": 100, "tolerance": 1e-8, "noslip_iterations": 0,
}
JOINT_ACTFRCRANGE = (0.0, 0.0)          # no joint-side force range ("limited" off, range 0 0)
MOUNT_NAME = re.compile(r"^mount_(left|right)_([1-4])$")
PRIMITIVE_SIZES = {"sphere": 1, "capsule": 2, "cylinder": 2, "ellipsoid": 3, "box": 3}   # MuJoCo's size counts


class BaseIntegrityError(RuntimeError):
    """A vendored file is missing, differs from its pinned sha256, or the runtime differs from the pinned one."""


class PlantEditError(ValueError):
    """A plant edit names something the unit may not change, or carries a non-finite or ill-shaped value."""


# ---------------------------------------------------------------------------------------------------------------
# Manifest: build, verify, restore.
# ---------------------------------------------------------------------------------------------------------------

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def vendored_paths(checkout: Path) -> list[str]:
    """The checkout-relative paths that are vendored: the licence, the six XML files, every file under assets/."""
    checkout = Path(checkout)
    rels = [MAKER_LICENSE, *VENDORED_XML]
    for tree in VENDORED_TREES:
        root = checkout / tree
        if not root.is_dir():
            raise BaseIntegrityError(f"{tree} is not a folder of the checkout")
        for p in sorted(root.rglob("*")):
            if p.is_symlink():
                raise BaseIntegrityError(f"symbolic link in the maker's tree: {p.relative_to(checkout)}")
            if p.is_file():
                rels.append(p.relative_to(checkout).as_posix())
    return sorted(set(rels))


def build_manifest(checkout: Path, commit: str = MAKER_COMMIT) -> dict:
    """The manifest of a maker checkout: every vendored file with its size and sha256."""
    checkout = Path(checkout)
    files = {}
    for rel in vendored_paths(checkout):
        p = checkout / rel
        files[rel] = {"sha256": sha256_file(p), "bytes": p.stat().st_size}
    return {
        "format": "openarm-unit-base-manifest/1",
        "source": {"repository": MAKER_REPOSITORY, "commit": commit, "folder": MAKER_FOLDER,
                   "licence": "Apache-2.0 (the repository's LICENSE file and the per-file headers)"},
        "layout": "every path is relative to the maker checkout's root and to base/ alike",
        "not_committed": "files under base/**/assets/ (restored by base/restore.py from a checkout at the commit)",
        "files": files,
    }


def load_manifest(base_dir: Path = BASE_DIR) -> dict:
    p = Path(base_dir) / MANIFEST_NAME
    if not p.is_file():
        raise BaseIntegrityError(f"no {MANIFEST_NAME} in the base folder")
    return json.loads(p.read_text(encoding="utf-8"))


def verify_tree(root: Path, manifest: Mapping, check_extra: bool = True) -> dict:
    """Compare a folder laid out like the checkout with the manifest. Returns {'missing', 'mismatch', 'extra', 'ok'}."""
    root = Path(root)
    missing, mismatch = [], []
    for rel, ent in sorted(manifest["files"].items()):
        p = root / rel
        if not p.is_file() or p.is_symlink():
            missing.append(rel)
        elif p.stat().st_size != ent["bytes"] or sha256_file(p) != ent["sha256"]:
            mismatch.append(rel)
    extra = []
    if check_extra:
        listed = set(manifest["files"])
        for tree in VENDORED_TREES:
            t = root / tree
            if t.is_dir():
                extra += [q.relative_to(root).as_posix() for q in sorted(t.rglob("*"))
                          if q.is_file() and q.relative_to(root).as_posix() not in listed]
    return {"missing": missing, "mismatch": mismatch, "extra": extra,
            "ok": not missing and not mismatch and not extra, "n_files": len(manifest["files"])}


_VERIFIED: set[str] = set()


def verify_base(base_dir: Path = BASE_DIR, force: bool = False) -> dict:
    """Verify base/ against its manifest (once per process unless forced); raise BaseIntegrityError on any difference."""
    key = str(Path(base_dir).resolve())
    if key in _VERIFIED and not force:
        return {"ok": True, "cached": True}
    man = load_manifest(base_dir)
    if man["source"]["commit"] != MAKER_COMMIT:
        raise BaseIntegrityError(f"manifest pins commit {man['source']['commit']}, the module {MAKER_COMMIT}")
    r = verify_tree(base_dir, man)
    if not r["ok"]:
        raise BaseIntegrityError(
            "base/ differs from its manifest (run base/restore.py <maker checkout>): "
            f"missing {r['missing'][:5]} ({len(r['missing'])}), mismatch {r['mismatch'][:5]} ({len(r['mismatch'])}), "
            f"extra {r['extra'][:5]} ({len(r['extra'])})")
    _VERIFIED.add(key)
    return r


def checkout_commit(checkout: Path) -> str | None:
    """HEAD of a git checkout (read-only ``git rev-parse``), or None when the folder is not a git checkout."""
    checkout = Path(checkout)
    if not (checkout / ".git").exists():
        return None
    out = subprocess.run(["git", "-C", str(checkout), "rev-parse", "HEAD"], capture_output=True, text=True,
                         check=False, timeout=30)
    return out.stdout.strip() if out.returncode == 0 else None


def restore_base(checkout: Path, base_dir: Path = BASE_DIR, require_commit: bool = True) -> dict:
    """Rebuild base/ from a maker checkout: verify each source file against the manifest, copy it, verify the copy.

    The manifest in ``base_dir`` is the authority; it is never rewritten here. With ``require_commit`` the checkout must
    be a git checkout whose HEAD is the pinned commit (a folder that is not a git checkout is accepted only without it).
    """
    checkout, base_dir = Path(checkout), Path(base_dir)
    man = load_manifest(base_dir)
    head = checkout_commit(checkout)
    if require_commit and head != man["source"]["commit"]:
        raise BaseIntegrityError(f"the checkout's HEAD is {head}, the manifest pins {man['source']['commit']}")
    src = verify_tree(checkout, man, check_extra=False)
    if src["missing"] or src["mismatch"]:
        raise BaseIntegrityError(f"the checkout differs from the manifest: missing {src['missing'][:5]}, "
                                 f"mismatch {src['mismatch'][:5]}")
    copied = 0
    for rel, ent in sorted(man["files"].items()):
        dst = base_dir / rel
        if dst.is_file() and dst.stat().st_size == ent["bytes"] and sha256_file(dst) == ent["sha256"]:
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(checkout / rel, dst)
        copied += 1
    r = verify_tree(base_dir, man)
    if not r["ok"]:
        raise BaseIntegrityError(f"base/ after the restore differs from the manifest: {r}")
    _VERIFIED.discard(str(base_dir.resolve()))
    return {"copied": copied, "n_files": r["n_files"], "checkout_head": head, "ok": True}


# ---------------------------------------------------------------------------------------------------------------
# Loaders.
# ---------------------------------------------------------------------------------------------------------------

def _mujoco():
    import mujoco
    if mujoco.__version__ != MUJOCO_VERSION:
        raise BaseIntegrityError(f"MuJoCo {mujoco.__version__}; the unit is pinned to {MUJOCO_VERSION} (no other release)")
    return mujoco


def scene_path(base_dir: Path = BASE_DIR) -> Path:
    return Path(base_dir) / SCENE


def maker_spec(base_dir: Path = BASE_DIR, verify: bool = True):
    """The maker's pedestal scene as its files give it (MuJoCo resolves the include and the two attachments)."""
    mujoco = _mujoco()
    if verify:
        verify_base(base_dir)
    return mujoco.MjSpec.from_file(str(scene_path(base_dir)))


def maker_model(base_dir: Path = BASE_DIR, verify: bool = True):
    return maker_spec(base_dir, verify).compile()


def _f(x, what) -> float:
    v = float(x)
    if not math.isfinite(v):
        raise PlantEditError(f"{what}: non-finite value {x!r}")
    return v


def _vec(x, n, what) -> list[float]:
    v = [_f(e, what) for e in x]
    if len(v) != n:
        raise PlantEditError(f"{what}: {len(v)} numbers, {n} expected")
    return v


def _inertial(ent, what) -> dict:
    keys = {"mass", "pos", "quat", "diaginertia"}
    if set(ent) != keys:
        raise PlantEditError(f"{what}: an inertial needs exactly {sorted(keys)}, got {sorted(ent)}")
    out = {"mass": _f(ent["mass"], what), "pos": _vec(ent["pos"], 3, what), "quat": _vec(ent["quat"], 4, what),
           "diaginertia": _vec(ent["diaginertia"], 3, what)}
    if out["mass"] <= 0 or min(out["diaginertia"]) <= 0:
        raise PlantEditError(f"{what}: mass and principal inertias must be positive")
    return out


def normalize_plant(plant: Mapping | None) -> dict:
    """Check a plant edit mapping against what may be set per unit; return it with float values."""
    plant = dict(plant or {})
    unknown = set(plant) - {"joints", "inertials", "base", "actuators", "mounts"}
    if unknown:
        raise PlantEditError(f"unknown plant keys {sorted(unknown)}")
    out: dict[str, Any] = {"joints": {}, "inertials": {}, "base": {}, "actuators": {}, "mounts": []}
    for name, ent in (plant.get("joints") or {}).items():
        if name not in HINGES:
            raise PlantEditError(f"joint {name!r} is not one of the eighteen hinges")
        bad = set(ent) - {"range", "frictionloss", "damping"}
        if bad:
            raise PlantEditError(f"joint {name}: {sorted(bad)} is not set per unit")
        e = {}
        if "range" in ent:
            if name not in ARM_JOINTS:          # an as-built range on the fourteen arm joints only
                raise PlantEditError(f"joint {name}: only the fourteen arm joints carry an as-built range")
            e["range"] = _vec(ent["range"], 2, f"{name}.range")
            if not e["range"][0] < e["range"][1]:
                raise PlantEditError(f"joint {name}: range lower must be below upper")
        for k in ("frictionloss", "damping"):
            if k in ent:
                e[k] = _f(ent[k], f"{name}.{k}")
                if e[k] < 0:
                    raise PlantEditError(f"joint {name}: negative {k}")
        out["joints"][name] = e
    for name, ent in (plant.get("inertials") or {}).items():
        if name not in INERTIAL_BODIES:
            raise PlantEditError(f"body {name!r}: inertials are set per unit only on {INERTIAL_BODIES}")
        out["inertials"][name] = _inertial(ent, f"inertial {name}")
    for side, ent in (plant.get("base") or {}).items():
        if side not in SIDES or set(ent) != {"pos", "quat"}:
            raise PlantEditError(f"base {side!r}: a side with exactly 'pos' and 'quat'")
        out["base"][side] = {"pos": _vec(ent["pos"], 3, f"base {side}"), "quat": _vec(ent["quat"], 4, f"base {side}")}
    act_names = [a for a, _ in ACTUATORS]
    for name, ent in (plant.get("actuators") or {}).items():
        if name not in act_names or not set(ent) <= {"k", "b"}:
            raise PlantEditError(f"actuator {name!r}: one of the sixteen, with 'k' and / or 'b'")
        out["actuators"][name] = {k: _f(v, f"{name}.{k}") for k, v in ent.items()}
    seen = set()
    for ent in plant.get("mounts") or []:
        m = MOUNT_NAME.match(str(ent.get("name", "")))
        if not m or ent["name"] in seen:
            raise PlantEditError(f"mount name {ent.get('name')!r}: mount_<side>_<1..4>, each once")
        seen.add(ent["name"])
        side = m.group(1)
        if ent.get("parent") not in ARM_BODIES[side]:
            raise PlantEditError(f"mount {ent['name']}: its parent must be a body of the {side} arm")
        bad = set(ent) - {"name", "parent", "pos", "quat", "inertial", "geom"}
        if bad:
            raise PlantEditError(f"mount {ent['name']}: unknown keys {sorted(bad)}")
        e = {"name": ent["name"], "parent": ent["parent"], "pos": _vec(ent["pos"], 3, ent["name"]),
             "quat": _vec(ent.get("quat", (1.0, 0.0, 0.0, 0.0)), 4, ent["name"]),
             "inertial": _inertial(ent["inertial"], f"mount {ent['name']}"), "geom": None}
        g = ent.get("geom")
        if g is not None:
            if g.get("type") not in PRIMITIVE_SIZES or not set(g) <= {"type", "size", "pos", "quat"}:
                raise PlantEditError(f"mount {ent['name']}: at most one PRIMITIVE geom")
            e["geom"] = {"type": g["type"], "size": _vec(g["size"], PRIMITIVE_SIZES[g["type"]], ent["name"]),
                         "pos": _vec(g.get("pos", (0.0, 0.0, 0.0)), 3, ent["name"]),
                         "quat": _vec(g.get("quat", (1.0, 0.0, 0.0, 0.0)), 4, ent["name"])}
            if min(e["geom"]["size"]) <= 0:
                raise PlantEditError(f"mount {ent['name']}: geom sizes must be positive")
        out["mounts"].append(e)
    return out


def apply_forced_numerics(spec, contacts: bool = True) -> None:
    """On every model alike: option block, friction-loss and joint-limit constraint numerics."""
    mujoco = _mujoco()
    o = spec.option
    o.timestep = OPTION["timestep"]
    o.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    o.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    o.impratio = OPTION["impratio"]
    o.solver = mujoco.mjtSolver.mjSOL_NEWTON
    o.iterations = OPTION["iterations"]
    o.tolerance = OPTION["tolerance"]
    o.noslip_iterations = OPTION["noslip_iterations"]
    if not contacts:
        o.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
    for name in HINGES:
        j = spec.joint(name)
        if j is None:
            raise BaseIntegrityError(f"the maker's scene lacks the hinge {name}")
        j.solref_friction = list(FRICTION_SOLREF)
        j.solimp_friction = list(FRICTION_SOLIMP)
        j.solref_limit = list(LIMIT_SOLREF)
        j.solimp_limit = list(LIMIT_SOLIMP)


def replace_actuators(spec, kb: Mapping[str, Mapping[str, float]] | None = None) -> None:
    """The sixteen maker actuators become ``general`` carriers in the maker's order.

    gear 1, joint transmission, no activation dynamics, fixed gain ``gainprm = [k, 0, ...]``, affine bias
    ``biasprm = [b, 0, ...]``, no control range and no force range in the file (the bus layer rewrites gain, bias and
    force range at run time: the force range IS the module's clamp, in delivered units k x L). The
    joint-side force range of every hinge is removed in the same call (``clear_joint_force_ranges``), so that every
    caller of this one function (a unit, the zero end, a submitted model) gets the clamp before the scale.
    """
    mujoco = _mujoco()
    kb = kb or {}
    old = list(spec.actuators)
    names = [a.name for a in old]
    if names != [a for a, _ in ACTUATORS]:
        raise BaseIntegrityError(f"the maker's actuators are not the sixteen in order: {names}")
    for a in old:
        spec.delete(a)
    for name, joint in ACTUATORS:
        e = kb.get(name, {})
        a = spec.add_actuator()
        a.name = name
        a.target = joint
        a.trntype = mujoco.mjtTrn.mjTRN_JOINT
        a.gear[:] = 0.0
        a.gear[0] = 1.0
        a.dyntype = mujoco.mjtDyn.mjDYN_NONE
        a.gaintype = mujoco.mjtGain.mjGAIN_FIXED
        a.gainprm[:] = 0.0
        a.gainprm[0] = float(e.get("k", 1.0))
        a.biastype = mujoco.mjtBias.mjBIAS_AFFINE
        a.biasprm[:] = 0.0
        a.biasprm[0] = float(e.get("b", 0.0))
        a.ctrllimited = mujoco.mjtLimited.mjLIMITED_FALSE
        a.forcelimited = mujoco.mjtLimited.mjLIMITED_FALSE
    clear_joint_force_ranges(spec)


def clear_joint_force_ranges(spec) -> None:
    """No joint-side actuator force range on any of the eighteen hinges (the module clamps
    what it commands, before the scale k; the maker's ``actuatorfrcrange`` would clamp after it)."""
    mujoco = _mujoco()
    for name in HINGES:
        j = spec.joint(name)
        if j is None:
            raise BaseIntegrityError(f"the maker's scene lacks the hinge {name}")
        j.actfrclimited = mujoco.mjtLimited.mjLIMITED_FALSE
        j.actfrcrange = list(JOINT_ACTFRCRANGE)


def _set_inertial(body, ent) -> None:
    body.explicitinertial = True
    body.mass = ent["mass"]
    body.ipos = ent["pos"]
    body.iquat = ent["quat"]
    body.inertia = ent["diaginertia"]


def apply_plant(spec, plant: Mapping) -> None:
    """Write a normalized plant edit mapping into a spec of the unit (the build path)."""
    mujoco = _mujoco()
    for name, e in plant["joints"].items():
        j = spec.joint(name)
        if "range" in e:
            j.range = e["range"]
        if "frictionloss" in e:
            j.frictionloss = e["frictionloss"]
        if "damping" in e:
            # MuJoCo 3.14 stores a joint's damping as polynomial coefficients [linear, higher terms]; the unit sets
            # the linear term only; the others stay as the file has them (0).
            dpoly = [float(x) for x in j.damping]
            dpoly[0] = e["damping"]
            j.damping = dpoly
    for name, e in plant["inertials"].items():
        _set_inertial(spec.body(name), e)
    for side, e in plant["base"].items():
        b = spec.body(BASE_BODIES[side])
        b.pos = e["pos"]
        b.quat = e["quat"]
    coll = spec.find_default("collision")
    if plant["mounts"] and coll is None:
        raise BaseIntegrityError("the maker's scene has no default class 'collision'")
    for e in plant["mounts"]:
        parent = spec.body(e["parent"])
        b = parent.add_body()
        b.name = e["name"]
        b.pos = e["pos"]
        b.quat = e["quat"]
        _set_inertial(b, e["inertial"])
        g = e["geom"]
        if g is not None:
            geom = b.add_geom(default=coll)
            geom.name = e["name"] + "_geom"
            geom.type = getattr(mujoco.mjtGeom, "mjGEOM_" + g["type"].upper())
            size = list(g["size"]) + [0.0] * (3 - len(g["size"]))
            geom.size = size
            geom.pos = g["pos"]
            geom.quat = g["quat"]


def unit_spec(plant: Mapping | None = None, contacts: bool = True, base_dir: Path = BASE_DIR, verify: bool = True):
    """The plant model of a unit as an MjSpec (the build path): maker scene + forced numerics + carriers + unit values."""
    p = normalize_plant(plant)
    spec = maker_spec(base_dir, verify)
    spec.modelname = UNIT_MODEL_NAME
    apply_forced_numerics(spec, contacts)
    replace_actuators(spec, p["actuators"])
    apply_plant(spec, p)
    return spec


def unit_model(plant: Mapping | None = None, contacts: bool = True, base_dir: Path = BASE_DIR, verify: bool = True):
    return unit_spec(plant, contacts, base_dir, verify).compile()


# ---------------------------------------------------------------------------------------------------------------
# Frames. Element-wise arithmetic only (no dot / matmul on a scored path).
# ---------------------------------------------------------------------------------------------------------------

def tool_points(model, data):
    """World positions of the two tool points on the UNIT, rows left and right, from ``data.xpos/xmat``.

    tool = xpos(ee_base_link) + R[:, 2] x (-0.135); R[:, 2] is the body z axis (xmat is row-major 3 x 3).
    """
    import numpy as np
    mujoco = _mujoco()
    out = np.empty((2, 3), dtype=np.float64)
    for r, side in enumerate(SIDES):
        b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, EE_BODIES[side])
        x = data.xmat[b]
        zaxis = np.array([x[2], x[5], x[8]], dtype=np.float64)
        out[r] = data.xpos[b] + zaxis * np.float64(TOOL_OFFSET[2])
    return out


def arm_origin(model, data):
    import numpy as np
    mujoco = _mujoco()
    s = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, ARM_ORIGIN_SITE)
    return np.array(data.site_xpos[s], dtype=np.float64)
