"""Scene composition helpers (v0.4).

``cluster_static_geoms`` -- a station scene keeps its collision geometry (wall boxes, handrails, inlet parts) on the
world body, so every physics step runs the robot-versus-world midphase over all of them (0.24 of 0.34 ms per step
in the three-module station).  This rewrites an MJCF file so that each collision geom that is a direct child of
<worldbody> moves, unchanged, into a static body (no joint, at the origin, identity orientation) per cubic cell of
``cell`` metres; MuJoCo's body-level broadphase then discards the far cells.  Geom attributes are not touched and the
body frame is the world frame, so every compiled geom has the same type, size, pose, material and contact parameters;
static-static pairs stay excluded.  Visual-only geoms (contype = conaffinity = 0, directly or through their default
class) stay on the world body.  The new
bodies are appended after the existing ones (body ids of the original bodies do not change; bodies attached later,
e.g. a robot, are numbered after the cells).
Bit-identity: a run without contacts between the robot and the moved geometry is bit-identical; with contacts the
contact set is the same but its order follows the body pairs, so a constraint solve can differ in the last bits and a
contact-rich trajectory then diverges like any chaotic run (both measured).
Only the Python standard library is used (``xml.etree.ElementTree``); comments are not kept.
"""
from __future__ import annotations

import math
import os
import xml.etree.ElementTree as ET

__all__ = ["cluster_static_geoms", "cluster_static_geoms_xml"]


def _geom_centre(g) -> list:
    if g.get("pos") is not None:
        return [float(v) for v in g.get("pos").split()]
    ft = g.get("fromto")
    if ft is not None:
        v = [float(x) for x in ft.split()]
        return [(v[0] + v[3]) / 2, (v[1] + v[4]) / 2, (v[2] + v[5]) / 2]
    return [0.0, 0.0, 0.0]


def _class_contacts(root) -> dict:
    """{default class name: (contype, conaffinity)} with inheritance through nested <default> elements ('main' = the
    top-level default; values None where no level sets them)."""
    out = {}

    def walk(d, inherited):
        name = d.get("class", "main")
        g = d.find("geom")
        ct = g.get("contype") if g is not None and g.get("contype") is not None else inherited[0]
        ca = g.get("conaffinity") if g is not None and g.get("conaffinity") is not None else inherited[1]
        out[name] = (ct, ca)
        for c in d.findall("default"):
            walk(c, (ct, ca))
    for d in root.findall("default"):
        walk(d, (None, None))
    return out


def _collides(g, classes: dict) -> bool:
    ct0, ca0 = classes.get(g.get("class", "main"), (None, None))
    ct = g.get("contype", ct0 if ct0 is not None else "1")
    ca = g.get("conaffinity", ca0 if ca0 is not None else "1")
    return bool(int(ct) or int(ca))


def cluster_static_geoms_xml(text: str, base_dir: str | None = None, cell: float = 1.0, prefix: str = "sk_cell") -> tuple:
    """(new MJCF text, report) of an MJCF text (see module doc).  base_dir: directory the asset paths of the text are
    relative to (made absolute so the result can be written anywhere); None leaves them as they are."""
    root = ET.fromstring(text)
    if base_dir is not None:
        comp = root.find("compiler")
        if comp is not None:
            for k in ("meshdir", "texturedir", "assetdir"):
                if comp.get(k) is not None and not os.path.isabs(comp.get(k)):
                    comp.set(k, os.path.normpath(os.path.join(base_dir, comp.get(k))))
        for inc in root.iter("include"):
            f = inc.get("file")
            if f is not None and not os.path.isabs(f):
                inc.set("file", os.path.normpath(os.path.join(base_dir, f)))
    classes = _class_contacts(root)
    wb = root.find("worldbody")
    if wb is None:
        raise ValueError("no <worldbody>")
    cells, moved, kept = {}, 0, 0
    for g in list(wb):
        if g.tag != "geom":
            continue
        if not _collides(g, classes):
            kept += 1
            continue
        key = tuple(int(math.floor(v / cell)) for v in _geom_centre(g))
        b = cells.get(key)
        if b is None:
            b = ET.SubElement(wb, "body")
            b.set("name", "{}_{}_{}_{}".format(prefix, *[str(k).replace("-", "m") for k in key]))
            cells[key] = b
        wb.remove(g)
        b.append(g)
        moved += 1
    return ET.tostring(root, encoding="unicode"), dict(moved=moved, kept_on_world=kept, cells=len(cells), cell_m=float(cell))


def cluster_static_geoms(in_xml: str, out_xml: str, cell: float = 1.0, prefix: str = "sk_cell") -> dict:
    """Write the clustered copy of the MJCF file ``in_xml`` to ``out_xml`` (asset paths made absolute); returns the
    report (moved geoms, cells, geoms left on the world body)."""
    with open(in_xml) as fh:
        text = fh.read()
    out, rep = cluster_static_geoms_xml(text, os.path.dirname(os.path.abspath(in_xml)), cell, prefix)
    with open(out_xml, "w") as fh:
        fh.write(out)
    rep["out"] = out_xml
    return rep
