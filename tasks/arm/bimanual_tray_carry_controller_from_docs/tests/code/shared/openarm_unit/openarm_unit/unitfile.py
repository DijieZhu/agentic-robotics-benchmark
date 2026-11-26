"""The unit-file writer: ``unit.xml`` and ``unit_cal.json`` of a unit. PUBLIC.

PUBLIC: it writes a model file from the maker's files and a plant edit mapping; it holds nothing of the law of
drawing, the registry, the truth channel or the recorder, and imports only ``base``.

This module writes ``unit.xml`` (one file, includes and attach resolved, floats
by ``repr()``) and ``unit_cal.json`` through the module's own writer (``openarm_unit/unitfile.py``: the maker's XML
texts edited as an element tree; every numeric ``mjModel`` array of the written file, recompiled, bit
for bit equals the build path's). MuJoCo's own XML writer is never used (it writes 6 digits, and its output
of the pedestal scene does not compile).

How the file is made:
1. The maker's vendored texts are parsed as element trees (``base/v2/...``, verified against the manifest).
2. ``<include>`` elements are replaced by the children of the included file's root, recursively.
3. The two ``<attach>`` elements (both with ``prefix=""``) are replaced by the child model's ``<worldbody>`` children;
   the child's defaults, assets, equalities, actuators, contact, sensors and keyframes are merged into the parent's
   sections in the order MuJoCo's own attach puts them (every compiled array equal); the
   child's ``<compiler>`` and ``<option>`` do not carry across an attach (the maker's common file says so). Mesh file
   names are re-expressed against one mesh folder.
4. The forced numerics and the actuator carriers of ``base.py`` are written, every joint-side ``actuatorfrcrange``
   is removed (each hinge gets ``actuatorfrclimited="false"``: the module's clamp acts before the scale; the file
   carries no ``actuatorfrcrange``), then the unit's own values (the plant edit mapping documented in ``base.py``).
   Every number this module writes is ``repr()`` of a float64; every number it does not change stays the maker's own
   text.
5. A header comment says that the file is a modified file of the maker's (Apache-2.0, section 4 b) and what changed.

The mesh folder. By default the file names the RELATIVE mesh folder ``assets`` (``DEFAULT_MESHDIR``): a written file
carries no host path, and whoever compiles it places the maker's ``v2/assets`` tree beside it or uses
``compile_unit_xml``, which reads the meshes from the vendored tree when the relative folder is not beside the file.
A caller may pass any other mesh folder.

``unit_cal.json`` is the unit's only calibration file:
``{"format": "openarm-pair-calibration/1", "joints": {<arm joint>: {"sign": 1, "zero_offset_rad": delta}}}``; on every
unit s_i = +1. Floats are written by ``json`` (``float.__repr__``).
"""
from __future__ import annotations

import hashlib
import json
import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Mapping

from . import base as B

CALIBRATION_FORMAT = "openarm-pair-calibration/1"     # calibration.json
DEFAULT_MESHDIR = "assets"                            # relative: a written file carries no host path

SECTIONS = ("compiler", "option", "size", "statistic", "visual", "default", "asset", "worldbody", "contact",
            "equality", "tendon", "actuator", "sensor", "keyframe")

HEADER = """
Modified file (Apache License 2.0, section 4 b). This file was written by openarm_unit/unitfile.py from the files of
github.com/enactic/openarm_mujoco, folder v2/, commit {commit} (Copyright 2026 Enactic, Inc.; Apache License 2.0,
http://www.apache.org/licenses/LICENSE-2.0; the ZED Mini mesh is Stereolabs', Apache-2.0, as the maker's pedestal file
states). Changes against the maker's files: the scene, its include and its two attachments flattened into one file;
the option block and the friction-loss and joint-limit constraint numerics of the eighteen hinges set; the sixteen
position actuators replaced by general actuators that carry a delivered-torque scale and a constant offset; the
joint-side actuator force ranges removed (the module's own clamp acts on its command, before the scale); the
values of this unit written (joint friction, damping and stops, link inertials, base placement, mounted items).
Every number written by the tool is the shortest round-trip form of a 64-bit float.
"""


def fmt(x) -> str:
    """One float64 as text: Python's repr (shortest round trip, at most 17 significant digits)."""
    return repr(float(x))


def fmt_vec(v) -> str:
    return " ".join(fmt(x) for x in v)


def _parse(path: Path) -> ET.Element:
    return ET.parse(path).getroot()


def _resolve_includes(elem: ET.Element, model_dir: Path, depth: int = 0) -> None:
    """Replace every <include file=...> under ``elem`` by the children of the included file's root (recursively)."""
    if depth > 8:
        raise B.BaseIntegrityError("include nesting deeper than 8")
    i = 0
    while i < len(elem):
        ch = elem[i]
        if ch.tag == "include":
            inc = _parse(model_dir / ch.attrib["file"])
            _resolve_includes(inc, model_dir, depth + 1)
            elem.remove(ch)
            for k, sub in enumerate(list(inc)):
                elem.insert(i + k, sub)
            i += len(inc)
            continue
        _resolve_includes(ch, model_dir, depth)
        i += 1


def _meshdir_of(root: ET.Element, model_dir: Path) -> Path:
    meshdir = None
    for c in root.findall("compiler"):
        if "meshdir" in c.attrib:
            meshdir = c.attrib["meshdir"]
    return (model_dir / meshdir).resolve() if meshdir is not None else model_dir.resolve()


def _rebase_mesh_files(root: ET.Element, model_dir: Path) -> None:
    """Make every mesh file attribute absolute (against the model's own meshdir) so that one meshdir serves all."""
    md = _meshdir_of(root, model_dir)
    for asset in root.findall("asset"):
        for m in asset.findall("mesh"):
            if "file" in m.attrib:
                m.set("file", str((md / m.attrib["file"]).resolve()))


def _section(root: ET.Element, tag: str) -> ET.Element:
    """The one section element of ``tag`` (several are merged into the first, in order)."""
    found = root.findall(tag)
    if not found:
        el = ET.SubElement(root, tag)
        return el
    first = found[0]
    for extra in found[1:]:
        for c in list(extra):
            first.append(c)
        for k, v in extra.attrib.items():
            first.set(k, v)
        root.remove(extra)
    return first


def _merge_default(parent_root: ET.Element, child_root: ET.Element) -> None:
    """The child's top-level default: its own settings and its classes go into the parent's top default."""
    child_defs = child_root.findall("default")
    if not child_defs:
        return
    pdef = _section(parent_root, "default")
    for cd in child_defs:
        for c in list(cd):
            if c.tag != "default":
                # A setting of the child's main class. MuJoCo keeps it on the child's main class; here it may only be
                # merged when the parent's main class does not already set that element (it never does in the
                # maker's scene) and when it does not change anything the parent's own elements use.
                if pdef.find(c.tag) is not None:
                    raise B.BaseIntegrityError(f"both main defaults set <{c.tag}>: cannot flatten")
            pdef.append(c)


def _load_model_tree(path: Path) -> ET.Element:
    """Parse a model file, resolve its includes, re-express mesh files, and resolve the models it attaches."""
    model_dir = path.parent
    root = _parse(path)
    _resolve_includes(root, model_dir)
    _rebase_mesh_files(root, model_dir)
    _resolve_attachments(root, model_dir)
    return root


def _resolve_attachments(root: ET.Element, model_dir: Path) -> None:
    models = {}
    for asset in root.findall("asset"):
        for m in list(asset.findall("model")):
            models[m.attrib["name"]] = model_dir / m.attrib["file"]
            asset.remove(m)
    if not models:
        return
    attached = []
    wb = root.find("worldbody")

    def walk(el):
        i = 0
        while i < len(el):
            ch = el[i]
            if ch.tag == "attach":
                if ch.attrib.get("prefix", "") != "" or set(ch.attrib) - {"model", "prefix"}:
                    raise B.BaseIntegrityError(f"an attach the flattener does not handle: {ch.attrib}")
                child = _load_model_tree(models[ch.attrib["model"]].resolve())
                attached.append(child)
                cwb = child.find("worldbody")
                el.remove(ch)
                kids = list(cwb) if cwb is not None else []
                for k, sub in enumerate(kids):
                    el.insert(i + k, sub)
                i += len(kids)
                continue
            walk(ch)
            i += 1

    walk(wb)
    for child in attached:
        _merge_default(root, child)
        for tag in ("asset", "contact", "equality", "tendon", "actuator", "sensor", "keyframe"):
            for sec in child.findall(tag):
                if len(sec):
                    dst = _section(root, tag)
                    for c in list(sec):
                        dst.append(c)


def _by_name(root: ET.Element, tag: str) -> dict:
    out = {}
    for el in root.iter(tag):
        n = el.attrib.get("name")
        if n is not None:
            if n in out:
                raise B.BaseIntegrityError(f"two <{tag}> named {n}")
            out[n] = el
    return out


def _write_inertial(body: ET.Element, ent: Mapping) -> None:
    inert = body.find("inertial")
    if inert is None:
        inert = ET.Element("inertial")
        body.insert(0, inert)
    inert.attrib.clear()
    inert.set("pos", fmt_vec(ent["pos"]))
    inert.set("quat", fmt_vec(ent["quat"]))
    inert.set("mass", fmt(ent["mass"]))
    inert.set("diaginertia", fmt_vec(ent["diaginertia"]))


def _ordered_root(root: ET.Element) -> ET.Element:
    """One element per section, in a fixed order (MuJoCo's compiler does not depend on the section order)."""
    out = ET.Element("mujoco", {"model": B.UNIT_MODEL_NAME})
    for tag in SECTIONS:
        els = root.findall(tag)
        if els:
            out.append(_section(root, tag))
    left = [c.tag for c in root if c.tag not in SECTIONS]
    if left:
        raise B.BaseIntegrityError(f"sections the flattener does not handle: {left}")
    return out


def unit_tree(plant: Mapping | None = None, contacts: bool = True, meshdir: str | os.PathLike | None = None,
              base_dir: Path = B.BASE_DIR, verify: bool = True) -> ET.Element:
    """The element tree of ``unit.xml`` (see the module text)."""
    if verify:
        B.verify_base(base_dir)
    p = B.normalize_plant(plant)
    scene = Path(base_dir) / B.SCENE
    root = _ordered_root(_load_model_tree(scene))

    # Compiler: one meshdir for every mesh (step 3 of the module text).
    assets_dir = (Path(base_dir) / B.MAKER_FOLDER / "assets").resolve()
    md = Path(meshdir) if meshdir is not None else Path(DEFAULT_MESHDIR)
    comp = root.find("compiler")
    comp.attrib.clear()
    comp.set("angle", "radian")
    comp.set("meshdir", str(md))
    for asset in root.findall("asset"):
        for m in asset.findall("mesh"):
            if "file" in m.attrib:
                f = Path(m.attrib["file"])
                try:
                    m.set("file", f.relative_to(assets_dir).as_posix())
                except ValueError as e:
                    raise B.BaseIntegrityError(f"a mesh outside the vendored assets: {f.name}") from e

    # Option block, forced on every model alike.
    opt = root.find("option")
    if opt is None:
        opt = ET.Element("option")
        root.insert(list(root).index(comp) + 1, opt)
    for c in list(opt):
        opt.remove(c)
    opt.attrib.clear()
    o = B.OPTION
    opt.set("timestep", fmt(o["timestep"]))
    opt.set("integrator", o["integrator"])
    opt.set("cone", o["cone"])
    opt.set("impratio", fmt(o["impratio"]))
    opt.set("solver", o["solver"])
    opt.set("iterations", str(int(o["iterations"])))
    opt.set("tolerance", fmt(o["tolerance"]))
    opt.set("noslip_iterations", str(int(o["noslip_iterations"])))
    if not contacts:
        ET.SubElement(opt, "flag", {"contact": "disable"})

    joints = _by_name(root, "joint")
    bodies = _by_name(root, "body")
    # No joint-side actuator force range anywhere (base.clear_joint_force_ranges): the attribute is removed from every
    # <joint> element (default classes included) and each hinge says actuatorfrclimited="false".
    for j in root.iter("joint"):
        j.attrib.pop("actuatorfrcrange", None)
        j.attrib.pop("actuatorfrclimited", None)
    for name in B.HINGES:
        j = joints[name]
        j.set("actuatorfrclimited", "false")
        j.set("solreffriction", fmt_vec(B.FRICTION_SOLREF))
        j.set("solimpfriction", fmt_vec(B.FRICTION_SOLIMP))
        j.set("solreflimit", fmt_vec(B.LIMIT_SOLREF))
        j.set("solimplimit", fmt_vec(B.LIMIT_SOLIMP))

    # Actuators: the sixteen carriers (base.replace_actuators), in the maker's order.
    act = root.find("actuator")
    old = [a.attrib.get("name") for a in act]
    if old != [a for a, _ in B.ACTUATORS]:
        raise B.BaseIntegrityError(f"the maker's actuators are not the sixteen in order: {old}")
    for c in list(act):
        act.remove(c)
    for name, joint in B.ACTUATORS:
        e = p["actuators"].get(name, {})
        ET.SubElement(act, "general", {
            "name": name, "joint": joint, "gear": "1", "dyntype": "none", "gaintype": "fixed",
            "gainprm": fmt(e.get("k", 1.0)), "biastype": "affine", "biasprm": fmt_vec((e.get("b", 0.0), 0.0, 0.0)),
            "ctrllimited": "false", "forcelimited": "false"})

    # The unit's own values (base.apply_plant).
    for name, e in p["joints"].items():
        j = joints[name]
        if "range" in e:
            j.set("range", fmt_vec(e["range"]))
        if "frictionloss" in e:
            j.set("frictionloss", fmt(e["frictionloss"]))
        if "damping" in e:
            j.set("damping", fmt(e["damping"]))
    for name, e in p["inertials"].items():
        _write_inertial(bodies[name], e)
    for side, e in p["base"].items():
        b = bodies[B.BASE_BODIES[side]]
        b.set("pos", fmt_vec(e["pos"]))
        b.set("quat", fmt_vec(e["quat"]))
    for e in p["mounts"]:
        parent = bodies[e["parent"]]
        mb = ET.SubElement(parent, "body", {"name": e["name"], "pos": fmt_vec(e["pos"]), "quat": fmt_vec(e["quat"])})
        _write_inertial(mb, e["inertial"])
        g = e["geom"]
        if g is not None:
            ET.SubElement(mb, "geom", {"name": e["name"] + "_geom", "class": "collision", "type": g["type"],
                                       "size": fmt_vec(g["size"]), "pos": fmt_vec(g["pos"]),
                                       "quat": fmt_vec(g["quat"])})

    root.insert(0, ET.Comment(HEADER.format(commit=B.MAKER_COMMIT)))
    return root


def unit_xml_text(plant: Mapping | None = None, contacts: bool = True, meshdir=None, base_dir: Path = B.BASE_DIR,
                  verify: bool = True) -> str:
    root = unit_tree(plant, contacts, meshdir, base_dir, verify)
    ET.indent(root, space="  ")
    return ET.tostring(root, encoding="unicode") + "\n"


def calibration(zero_offsets: Mapping[str, float] | None = None) -> dict:
    """unit_cal.json: sign +1 and the stored zero delta_i of each of the fourteen arm modules."""
    zero_offsets = dict(zero_offsets or {})
    unknown = set(zero_offsets) - set(B.ARM_JOINTS)
    if unknown:
        raise B.PlantEditError(f"zero offsets for joints that are not arm joints: {sorted(unknown)}")
    joints = {}
    for j in B.ARM_JOINTS:
        v = float(zero_offsets.get(j, 0.0))
        if v != v or v in (float("inf"), float("-inf")):
            raise B.PlantEditError(f"{j}: non-finite zero offset")
        joints[j] = {"sign": 1, "zero_offset_rad": v}
    return {"format": CALIBRATION_FORMAT, "joints": joints}


def write_unit(out_dir: str | os.PathLike, plant: Mapping | None = None, zero_offsets: Mapping | None = None,
               contacts: bool = True, meshdir=None, base_dir: Path = B.BASE_DIR) -> dict:
    """Write ``<out_dir>/unit.xml`` and ``<out_dir>/unit_cal.json``; return their paths and sha256."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    xml_text = unit_xml_text(plant, contacts, meshdir, base_dir)
    cal_text = json.dumps(calibration(zero_offsets), indent=1, sort_keys=True) + "\n"
    xp, cp = out / "unit.xml", out / "unit_cal.json"
    xp.write_text(xml_text, encoding="utf-8")
    cp.write_text(cal_text, encoding="utf-8")
    return {"unit_xml": str(xp), "unit_cal": str(cp),
            "unit_xml_sha256": hashlib.sha256(xml_text.encode("utf-8")).hexdigest(),
            "unit_cal_sha256": hashlib.sha256(cal_text.encode("utf-8")).hexdigest()}


def compile_unit_xml(path: str | os.PathLike, assets_dir: str | os.PathLike | None = None,
                     base_dir: Path = B.BASE_DIR):
    """Compile a written unit.xml with MuJoCo (the pinned version).

    A file whose mesh folder resolves (absolute, or relative beside the file) is compiled as it is. Otherwise its
    meshes are read from ``assets_dir`` (default: the vendored ``base/v2/assets``, verified against the manifest); the
    file is not changed, only the compiler's mesh folder of the in-memory spec. The mesh folder is a path, not a
    number: the compiled arrays do not depend on it.
    """
    mujoco = B._mujoco()
    path = Path(path)
    spec = mujoco.MjSpec.from_file(str(path))
    md = Path(spec.meshdir) if spec.meshdir else Path(".")
    if not md.is_absolute():
        md = path.parent / md
    if not md.is_dir():
        if assets_dir is None:
            B.verify_base(base_dir)
            assets_dir = Path(base_dir) / B.MAKER_FOLDER / "assets"
        spec.meshdir = str(Path(assets_dir).resolve())
    return spec.compile()


__all__ = ["fmt", "unit_tree", "unit_xml_text", "calibration", "write_unit", "compile_unit_xml", "CALIBRATION_FORMAT",
           "DEFAULT_MESHDIR"]
