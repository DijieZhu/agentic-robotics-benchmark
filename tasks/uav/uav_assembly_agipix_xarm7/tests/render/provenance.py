"""Digests and provenance of the render layer.

    request_digest(request)        sha256 of canonical_json(request) + the bytes of every file its scene reads
    scene_source_files(scene)      the MJCF and everything it includes / loads, the builder's source file, content_files
    library_environment()          versions of the libraries that can change pixels (numpy, Pillow, libjpeg, zlib)

The request digest is backend-independent (the backend NAME is part of the request, its version is not): two
backends asked the same thing share everything but `backend` in the canonical JSON.  SceneSpec.base_dir is left out
(moving a task folder keeps the digest when the scene paths are written relative to it).  The backend adds its own
`backend_version`, `resolved_scene_digest` and GL facts to the Provenance record.  No timestamps anywhere: a
provenance record is a pure function of the request and the software stack, so photos_gt.json is byte-stable.
"""
from __future__ import annotations

import hashlib
import os
import xml.etree.ElementTree as ET
from typing import Dict, List

from .spec import RENDER_LAYER_VERSION, canonical_json

_TEX_FACES = ("fileright", "fileleft", "fileup", "filedown", "filefront", "fileback")


def _resolve_dir(base, sub):
    if not sub:
        return base
    return sub if os.path.isabs(sub) else os.path.normpath(os.path.join(base, sub))


def mjcf_referenced_files(mjcf_path: str) -> List[str]:
    """Absolute paths of the MJCF file and every file it (recursively) includes or loads (meshes, textures,
    hfields, skins, attached models).  MuJoCo's rules: include paths are relative to the MAIN file's directory and
    <compiler meshdir / texturedir / assetdir> are global (the last one read wins), so two passes."""
    main = os.path.abspath(mjcf_path)
    root_dir = os.path.dirname(main)
    xml_files, roots = [], []

    def walk(path):
        path = os.path.abspath(path)
        if path in xml_files:
            return
        xml_files.append(path)
        try:
            root = ET.parse(path).getroot()
        except (ET.ParseError, OSError):
            return
        roots.append(root)
        for el in root.iter():
            if el.tag in ("include", "model") and "file" in el.attrib:
                walk(_resolve_dir(root_dir, el.attrib["file"]))

    walk(main)
    meshdir = texdir = root_dir
    for root in roots:
        for comp in root.iter("compiler"):
            if "assetdir" in comp.attrib:
                meshdir = texdir = _resolve_dir(root_dir, comp.attrib["assetdir"])
            if "meshdir" in comp.attrib:
                meshdir = _resolve_dir(root_dir, comp.attrib["meshdir"])
            if "texturedir" in comp.attrib:
                texdir = _resolve_dir(root_dir, comp.attrib["texturedir"])
    out = list(xml_files)
    for root in roots:
        for el in root.iter():
            if el.tag in ("mesh", "hfield", "skin") and "file" in el.attrib:
                out.append(_resolve_dir(meshdir, el.attrib["file"]))
            elif el.tag == "texture":
                for a in ("file",) + _TEX_FACES:
                    if a in el.attrib:
                        out.append(_resolve_dir(texdir, el.attrib[a]))
    dedup, res = set(), []
    for p in out:
        if p not in dedup:
            dedup.add(p)
            res.append(p)
    return res


def _builder_source(scene) -> List[str]:
    ref = scene.builder
    target = ref.split(":", 1)[0]
    if target.endswith(".py"):
        p = target if os.path.isabs(target) else os.path.join(scene.base_dir or os.getcwd(), target)
        return [os.path.abspath(p)]
    try:  # a module reference: hash its source file if importable without side effects on sys.path
        import importlib.util
        spec = importlib.util.find_spec(target)
        if spec is not None and spec.origin and os.path.isfile(spec.origin):
            return [os.path.abspath(spec.origin)]
    except (ImportError, ValueError):
        pass
    return []


def _expand(paths: List[str], base: str) -> List[str]:
    out = []
    for p in paths:
        p = p if os.path.isabs(p) else os.path.join(base, p)
        if os.path.isdir(p):
            for dp, dn, fn in os.walk(p):
                dn[:] = sorted(d for d in dn if d != "__pycache__")
                for f in sorted(fn):
                    if not f.endswith(".pyc"):
                        out.append(os.path.abspath(os.path.join(dp, f)))
        else:
            out.append(os.path.abspath(p))
    return out


def scene_source_files(scene) -> List[str]:
    base = scene.base_dir or os.getcwd()
    files: List[str] = []
    if scene.mjcf is not None:
        p = scene.mjcf if os.path.isabs(scene.mjcf) else os.path.join(base, scene.mjcf)
        files += mjcf_referenced_files(p)
    else:
        files += _builder_source(scene)
    for t in scene.textures:
        if t.file:
            files.append(os.path.abspath(t.file if os.path.isabs(t.file) else os.path.join(base, t.file)))
    files += _expand(scene.content_files, base)
    seen, out = set(), []
    for f in files:
        if f not in seen:
            seen.add(f)
            out.append(f)
    return out


def file_digest(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def scene_content_digest(scene) -> Dict[str, str]:
    """{basename-relative key: sha256} of the scene's source files; keys drop the host-specific prefix so the digest
    survives moving the tree (the request JSON itself still carries the path strings as written)."""
    files = scene_source_files(scene)
    if not files:
        return {}
    common = os.path.commonpath([os.path.dirname(f) for f in files]) if len(files) > 1 else os.path.dirname(files[0])
    out = {}
    for f in files:
        key = os.path.relpath(f, common)
        out[key] = file_digest(f) if os.path.isfile(f) else "missing"
    return out


def request_digest(request, scene_content: Dict[str, str] = None) -> str:
    """`scene_content` = a precomputed scene_content_digest(request.scene) (sessions cache it: the files do not change
    while a session is open)."""
    d = request.to_dict()
    d["scene"].pop("base_dir", None)        # a location, not content: the files' bytes enter through scene_content
    body = {"render_layer_version": RENDER_LAYER_VERSION, "request": d,
            "scene_content": scene_content if scene_content is not None else scene_content_digest(request.scene)}
    return hashlib.sha256(canonical_json(body).encode()).hexdigest()


_LIB_ENV: Dict[str, str] = {}


def library_environment() -> Dict[str, str]:
    if not _LIB_ENV:
        _LIB_ENV.update(_library_environment())
    return dict(_LIB_ENV)


def _library_environment() -> Dict[str, str]:
    import zlib

    import numpy
    import PIL
    env = {"numpy": numpy.__version__, "pillow": PIL.__version__, "zlib": zlib.ZLIB_VERSION}
    try:
        from PIL import features
        env["libjpeg"] = str(features.version("jpg"))
        env["libjpeg_turbo"] = str(features.check_feature("libjpeg_turbo"))
    except Exception:  # noqa: BLE001
        pass
    return env
