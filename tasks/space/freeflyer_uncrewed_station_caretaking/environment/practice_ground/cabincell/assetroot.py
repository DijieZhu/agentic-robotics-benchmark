"""assetroot.py -- where the cell's large model files are, and the check that they are the files this task was built
with.

The meshes of the robot model (and, by the same mechanism, of the station scene) are not stored with the task: each
component ships as one archive whose manifest ``data/<component>/ASSETS.json`` (next to this file) lists the
archive's sha256 and the sha256 and size of every file it unpacks to.  An *asset root* is a directory that holds the
unpacked component directories (``<root>/robot/mjcf/...``).  ``find(component)`` looks in this order and returns the
verified directory ``<asset root>/<component>``:

  1. the environment variable ``CABINCELL_ASSET_ROOT`` (an asset root), when it is set;
  2. ``IMAGE_ROOT``, the task image's asset directory, filled when the image is built;
  3. the maintainers' staging pointer (``dev_pointer``: a JSON file in a folder next to the task folder) that names
     the archive, unpacked once into the cache directory that file names.

A source that is present but does not match the manifest (a missing, extra or changed file, a changed archive) is an
error; the search does not fall through to a later source.  When no source has the component the error names the
expected archive and its sha256.  Files that the manifest lists under ``mirrors`` must also equal the copies kept in
``data/<component>/``.

    import assetroot
    robot_dir = assetroot.find("robot")             # verified <root>/robot
    xml = assetroot.robot_paths()["robot_xml"]      # <root>/robot/mjcf/freeflyer_v1.xml
    assetroot.describe("robot")                     # which source, root, archive sha256, file count, bytes

Command line: ``python -I -B assetroot.py [component ...]`` prints ``describe`` as JSON (exit 1 on an error).
Standard library only.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import tarfile
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
ENV = "CABINCELL_ASSET_ROOT"
IMAGE_ROOT = Path("/opt/cabincell/assets")
SCHEMA_ASSETS = "cabincell-assets/1"
SCHEMA_STAGED = "cabincell-staged/1"
SOURCES = ("env", "image", "staging")

_FOUND: dict = {}


class AssetError(RuntimeError):
    """An asset component is missing or does not match its manifest."""


# ----------------------------------------------------------------------------------------------- manifests
def manifest(component: str, data_dir: Path | None = None) -> dict:
    """The parsed ``data/<component>/ASSETS.json``."""
    p = Path(data_dir or DATA) / component / "ASSETS.json"
    try:
        man = json.loads(p.read_text())
    except FileNotFoundError:
        raise AssetError(f"no manifest for asset component {component!r} ({p.name} missing in data/{component}/)") from None
    if man.get("schema") != SCHEMA_ASSETS or man.get("component") != component:
        raise AssetError(f"{p}: not a {SCHEMA_ASSETS} manifest of {component!r}")
    return man


def _expected(man: dict) -> str:
    a = man["archive"]
    return f"{a['name']} (sha256 {a['sha256']}, {man['unpacked']['files']} files, {man['unpacked']['bytes']} bytes unpacked)"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_tree(component_dir: Path, man: dict, data_dir: Path | None = None) -> list[str]:
    """Problems of an unpacked component directory against its manifest (empty = it matches): every listed file
    present with its sha256 and size, no other file, and the mirrored files equal to the copies in data/."""
    component_dir = Path(component_dir)
    top = man["archive"]["top"]
    want = {path: (sha, size) for path, sha, size in man["files"]}
    have = {}
    for dirpath, _dirs, files in os.walk(component_dir):
        for f in files:
            p = Path(dirpath) / f
            have[f"{top}/{p.relative_to(component_dir).as_posix()}"] = p
    probs = []
    for path in sorted(set(want) - set(have)):
        probs.append(f"missing {path}")
    for path in sorted(set(have) - set(want)):
        probs.append(f"not in the manifest: {path}")
    for path in sorted(set(want) & set(have)):
        sha, size = want[path]
        p = have[path]
        if p.stat().st_size != size or _sha256(p) != sha:
            probs.append(f"changed {path}")
    ddir = Path(data_dir or DATA) / man["component"]
    for path, name in sorted((man.get("mirrors") or {}).items()):
        g = ddir / name
        if path in want and (not g.exists() or _sha256(g) != want[path][0]):
            probs.append(f"data/{man['component']}/{name} differs from the archive's {path}")
    return probs


# ----------------------------------------------------------------------------------------------- unpacking
def _safe_members(tar: tarfile.TarFile, top: str) -> list:
    out = []
    for ti in tar.getmembers():
        name = ti.name
        parts = Path(name).parts
        if name.startswith("/") or ".." in parts or not parts or parts[0] != top:
            raise AssetError(f"archive member outside {top}/: {name!r}")
        if not (ti.isfile() or ti.isdir()):
            raise AssetError(f"archive member is not a file or directory: {name!r}")
        out.append(ti)
    return out


def unpack(archive: Path, dest_root: Path, man: dict) -> Path:
    """Unpack ``archive`` (checked against the manifest's sha256 first) into the asset root ``dest_root``, atomically
    (a temporary directory renamed into place; a concurrent unpacker that wins the rename is accepted); returns the
    verified ``dest_root/<top>``."""
    archive, dest_root = Path(archive), Path(dest_root)
    a = man["archive"]
    if not archive.is_file():
        raise AssetError(f"archive not found: {archive}; expected {_expected(man)}")
    got = _sha256(archive)
    if got != a["sha256"]:
        raise AssetError(f"archive {archive} has sha256 {got}; expected {_expected(man)}")
    top = a["top"]
    final = dest_root / top
    if not final.exists():
        dest_root.mkdir(parents=True, exist_ok=True)
        tmp = Path(tempfile.mkdtemp(prefix=f".{top}_unpack_", dir=dest_root))
        try:
            with tarfile.open(archive, mode="r:xz") as tar:
                members = _safe_members(tar, top)
                try:
                    tar.extractall(tmp, members=members, filter="data")
                except TypeError:                       # Python without extraction filters
                    tar.extractall(tmp, members=members)
            try:
                os.rename(tmp / top, final)
            except OSError:
                if not final.exists():
                    raise
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    probs = verify_tree(final, man)
    if probs:
        raise AssetError(f"{final} does not match the manifest ({len(probs)} problems, first: {probs[:3]}); "
                         f"expected {_expected(man)}")
    return final


# ----------------------------------------------------------------------------------------------- sources
def dev_pointer(component: str, task_dir: Path | None = None) -> Path:
    """The maintainers' staging pointer of a component, next to the task folder this file sits in (tests/cabincell/)."""
    task = Path(task_dir) if task_dir is not None else HERE.parents[1]
    return task.parent / (task.name + "-dev") / "assets_staging" / component / "STAGED.json"


def _check_dir(component_dir: Path, man: dict, source: str, data_dir: Path | None) -> Path:
    probs = verify_tree(component_dir, man, data_dir)
    if probs:
        raise AssetError(f"asset component {man['component']!r} from {source} ({component_dir}) does not match the "
                         f"manifest: {len(probs)} problems, first: {probs[:3]}; expected {_expected(man)}")
    return component_dir


def locate(component: str, env: dict | None = None, image_root: Path | None = None, pointer: Path | None = None,
           data_dir: Path | None = None) -> tuple[str, Path]:
    """(source, verified component directory), trying env -> image -> staging (see the module doc).  The keyword
    arguments replace os.environ, IMAGE_ROOT, the staging pointer and the data directory (for tests)."""
    man = manifest(component, data_dir)
    top = man["archive"]["top"]
    env = os.environ if env is None else env
    root = env.get(ENV)
    if root:
        d = Path(root) / top
        if not d.is_dir():
            raise AssetError(f"{ENV}={root} has no {top}/ directory; expected {_expected(man)}")
        return "env", _check_dir(d, man, f"${ENV}", data_dir)
    img = Path(IMAGE_ROOT if image_root is None else image_root) / top
    if img.is_dir():
        return "image", _check_dir(img, man, "the image's asset directory", data_dir)
    ptr = Path(pointer) if pointer is not None else dev_pointer(component)
    if ptr.is_file():
        st = json.loads(ptr.read_text())
        if st.get("schema") != SCHEMA_STAGED or st.get("component") != component:
            raise AssetError(f"{ptr}: not a {SCHEMA_STAGED} pointer for {component!r}")
        if st.get("sha256") != man["archive"]["sha256"]:
            raise AssetError(f"the staged archive {st.get('archive')} is {st.get('sha256')}; expected {_expected(man)}")
        cache = Path(st["cache"]) / man["archive"]["sha256"][:16]
        return "staging", unpack(Path(st["archive"]), cache, man)
    raise AssetError(f"asset component {component!r} not found: ${ENV} is not set, {img} does not exist and there is "
                     f"no staging pointer ({ptr.name}); expected {_expected(man)}")


def find(component: str) -> Path:
    """Verified ``<asset root>/<component>`` (cached per process after the first successful search)."""
    if component not in _FOUND:
        _FOUND[component] = locate(component)
    return _FOUND[component][1]


def describe(component: str) -> dict:
    """Which source the component came from, where, and the manifest's archive sha256 / file count / bytes."""
    find(component)
    src, d = _FOUND[component]
    man = manifest(component)
    return dict(component=component, source=src, dir=str(d), archive=man["archive"]["name"],
                archive_sha256=man["archive"]["sha256"], files=man["unpacked"]["files"], bytes=man["unpacked"]["bytes"])


def robot_paths() -> dict:
    """The reference build's model files in the verified robot component (paths for the station's site builder and
    the dock): robot_xml (the robot with collision meshes and the light visual set) and dock_xml (the dock scene;
    attach it with visual=False -- the dock's display meshes are not shipped)."""
    mj = find("robot") / "mjcf"
    return dict(robot_xml=str(mj / "freeflyer_v1.xml"), dock_xml=str(mj / "scene_dock_v1.xml"), mjcf_dir=str(mj))


def main(argv=None) -> int:
    comps = (sys.argv[1:] if argv is None else argv) or ["robot"]
    out, rc = [], 0
    for c in comps:
        try:
            out.append(describe(c))
        except AssetError as e:
            out.append(dict(component=c, error=str(e)))
            rc = 1
    print(json.dumps(out, indent=1))
    return rc


if __name__ == "__main__":
    sys.exit(main())
