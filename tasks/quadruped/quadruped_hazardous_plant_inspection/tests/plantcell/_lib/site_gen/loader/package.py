"""The stored site package: one deterministic archive per site, and its reader.

    <name>.tar.xz          USTAR tar, members sorted by name, mtime 0, uid / gid 0, no user or group names,
                           mode 0644, xz preset 9e with a CRC64 check
      manifest.json        format, name, label, kind, the generator's record, sha256 and size of every member, the
                           shared textures the scene uses (name -> sha256)
      site.json            ground truth (grader side and the privileged reference only)
      prior.json           the documents a program is given (drawing, round sheet, modification log, facts)
      scene.xml            the MJCF of the site (no robot); texturedir "assets/", textures by bare file name
      fields.npz           dose_xs, dose_ys, dose_truth (site units; a map: float32 rounded to 13 mantissa bits,
                           relative error below 1.3e-4; the field itself is site.json's sources and boxes),
                           link_origin, link_res, link, link_base, inside
      trav.npz             the traversability grid (enter_* / tight_* / speed_* per mode, slow, hsreq, crreq,
                           inside, elem, elem_ids) and params (JSON: thresholds and the mode-switch time)
      assets/<file>.png    the textures of this site (dial faces, legends, scales, tag plates, signs)

Textures that every site shares (the procedural material textures) are not in the archive: the reader takes them
from a shared directory (default `site/shared_assets` in the source tree, or the cell's data directory) and checks each against
the sha256 the manifest records. The npz members are written without compression and without time stamps, so the
archive's bytes depend only on the content.

Only the standard library and numpy are used here (the writer as well): this module is part of the run-time half.
"""

from __future__ import annotations

import hashlib
import io
import json
import lzma
import os
import tarfile
import zipfile

import numpy as np

FORMAT = "plantsite-package/1"
REQUIRED = ("manifest.json", "site.json", "prior.json", "scene.xml", "fields.npz", "trav.npz")
# procedural material textures (site_gen.textures, fixed seeds): the same bytes on every site
SHARED_TEXTURES = ("blockwork.png", "checker.png", "concrete.png", "concrete_light.png", "grating.png", "hazard.png",
                   "pad.png", "protector.png", "shield.png", "zone_tape.png")
FIELDS_KEEP = ("dose_xs", "dose_ys", "dose_truth", "link_origin", "link_res", "link", "link_base", "inside")
_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SHARED_DIR = os.path.normpath(os.path.join(_HERE, "..", "..", "shared_assets"))
_ZIP_DATE = (1980, 1, 1, 0, 0, 0)


class PackageError(RuntimeError):
    """A package that is malformed, incomplete, or whose content does not match its manifest."""


def sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


DOSE_MANTISSA_BITS = 13


def round_mantissa(a, bits=DOSE_MANTISSA_BITS):
    """float32 values rounded to `bits` mantissa bits (round half up on the magnitude): a deterministic, smaller
    map (relative error at most 2 ** -(bits + 1))."""
    a = np.ascontiguousarray(a, np.float32)
    drop = 23 - int(bits)
    u = a.view(np.uint32).astype(np.uint64)
    u = (u + (1 << (drop - 1))) & ~np.uint64((1 << drop) - 1)
    return u.astype(np.uint32).view(np.float32)


def shared_dir_default() -> str:
    return os.environ.get("SITE_GEN_SHARED_ASSETS", DEFAULT_SHARED_DIR)


# ============================================================================================ deterministic npz
def npz_bytes(arrays: dict) -> bytes:
    """An .npz (zip of .npy) with the members in sorted order, stored, with a fixed date: the bytes depend only on
    the arrays. np.load reads it."""
    b = io.BytesIO()
    with zipfile.ZipFile(b, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as z:
        for k in sorted(arrays):
            a = np.asanyarray(arrays[k])
            if a.dtype == object:
                raise PackageError("object array %s cannot be stored" % k)
            buf = io.BytesIO()
            np.lib.format.write_array(buf, a, allow_pickle=False)
            zi = zipfile.ZipInfo(k + ".npy", date_time=_ZIP_DATE)
            zi.compress_type = zipfile.ZIP_STORED
            zi.external_attr = 0o644 << 16
            zi.create_system = 3
            z.writestr(zi, buf.getvalue())
    return b.getvalue()


def npz_load(b: bytes) -> dict:
    with np.load(io.BytesIO(b), allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


# ============================================================================================ the archive
def tar_xz_bytes(members: dict) -> bytes:
    """Deterministic tar.xz of {name: bytes} (sorted, zero times and ids, mode 0644)."""
    b = io.BytesIO()
    with tarfile.open(fileobj=b, mode="w", format=tarfile.USTAR_FORMAT) as t:
        for name in sorted(members):
            data = members[name]
            ti = tarfile.TarInfo(name)
            ti.size = len(data)
            ti.mtime = 0
            ti.mode = 0o644
            ti.uid = ti.gid = 0
            ti.uname = ti.gname = ""
            ti.type = tarfile.REGTYPE
            t.addfile(ti, io.BytesIO(data))
    return lzma.compress(b.getvalue(), format=lzma.FORMAT_XZ, check=lzma.CHECK_CRC64,
                         preset=9 | lzma.PRESET_EXTREME)


def read_tar_xz(path_or_bytes) -> dict:
    raw = path_or_bytes if isinstance(path_or_bytes, (bytes, bytearray)) else open(path_or_bytes, "rb").read()
    try:
        data = lzma.decompress(raw)
    except lzma.LZMAError as e:
        raise PackageError("not an xz archive: %s" % e) from e
    out = {}
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:") as t:
        for ti in t.getmembers():
            if not ti.isfile():
                continue
            name = ti.name
            if name.startswith("/") or ".." in name.split("/"):
                raise PackageError("unsafe member name %r" % name)
            out[name] = t.extractfile(ti).read()
    return out


def _trav_params(site: dict, trav: dict, switch_time_s=None) -> dict:
    p = {}
    if "params" in trav:
        try:
            p = json.loads(str(trav["params"]))
        except ValueError:
            p = {}
    th = site.get("thresholds") or {}
    p.setdefault("r_in", th.get("r_in"))
    p.setdefault("r_circ", th.get("r_circ"))
    if switch_time_s is not None:
        p["switch_time_s"] = float(switch_time_s)
    p.setdefault("switch_time_s", 3.0)
    p.setdefault("modes", ["walk", "high_step", "crouch_walk"])
    return p


def write_package(site_dir: str, out_path: str, name: str | None = None, label: str | None = None,
                  kind: str | None = None, shared_dir: str | None = None, generator: dict | None = None,
                  switch_time_s: float | None = None) -> dict:
    """Write the package of a generated site folder (site.json, prior.json, scene.xml, fields.npz, trav.npz,
    assets/). Returns the manifest. Raises PackageError when a shared texture differs from the shared copy."""
    shared_dir = shared_dir or shared_dir_default()
    rd = lambda f: open(os.path.join(site_dir, f), "rb").read()  # noqa: E731
    site_b, prior_b, scene_b = rd("site.json"), rd("prior.json"), rd("scene.xml")
    site = json.loads(site_b)
    with np.load(os.path.join(site_dir, "fields.npz"), allow_pickle=False) as z:
        fields = {k: z[k] for k in FIELDS_KEEP if k in z.files}
    if "dose_truth" in fields:
        fields["dose_truth"] = round_mantissa(fields["dose_truth"])
    with np.load(os.path.join(site_dir, "trav.npz"), allow_pickle=False) as z:
        trav = {k: z[k] for k in z.files}
    trav["params"] = np.array(json.dumps(_trav_params(site, trav, switch_time_s), sort_keys=True))
    members = {"site.json": site_b, "prior.json": prior_b, "scene.xml": scene_b,
               "fields.npz": npz_bytes(fields), "trav.npz": npz_bytes(trav)}
    shared = {}
    adir = os.path.join(site_dir, "assets")
    scene_txt = scene_b.decode()
    for f in sorted(os.listdir(adir)) if os.path.isdir(adir) else []:
        if 'file="%s"' % f not in scene_txt:
            continue                                    # a texture the scene does not use
        data = open(os.path.join(adir, f), "rb").read()
        if f in SHARED_TEXTURES:
            sp = os.path.join(shared_dir, f)
            if not os.path.exists(sp):
                raise PackageError("shared texture %s missing in %s" % (f, shared_dir))
            if sha256(open(sp, "rb").read()) != sha256(data):
                raise PackageError("shared texture %s of %s differs from the shared copy" % (f, site_dir))
            shared[f] = sha256(data)
        else:
            members["assets/" + f] = data
    man = {"format": FORMAT, "name": name or os.path.basename(os.path.normpath(site_dir)), "label": label,
           "kind": kind, "seed": site.get("seed"), "attempt": site.get("attempt"),
           "schema": site.get("schema"), "layout": site.get("layout"), "config_hash": site.get("config_hash"),
           "generator": generator or {},
           "members": {k: {"sha256": sha256(v), "size": len(v)} for k, v in sorted(members.items())},
           "shared_assets": dict(sorted(shared.items()))}
    members["manifest.json"] = (json.dumps(man, indent=1, sort_keys=True) + "\n").encode()
    blob = tar_xz_bytes(members)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    tmp = out_path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(blob)
    os.replace(tmp, out_path)
    man = dict(man, archive={"sha256": sha256(blob), "size": len(blob)})
    return man


def write_shared_assets(src_dirs, shared_dir: str) -> dict:
    """Collect the shared textures from generated site folders into shared_dir; every site must have the same bytes.
    Returns {name: sha256}."""
    os.makedirs(shared_dir, exist_ok=True)
    got = {}
    for d in src_dirs:
        for f in SHARED_TEXTURES:
            p = os.path.join(d, "assets", f)
            if not os.path.exists(p):
                continue
            h = sha256(open(p, "rb").read())
            if f in got and got[f] != h:
                raise PackageError("shared texture %s differs between sites (%s)" % (f, d))
            if f not in got:
                got[f] = h
                with open(os.path.join(shared_dir, f), "wb") as o:
                    o.write(open(p, "rb").read())
    return got


# ============================================================================================ the reader
class SitePackage:
    """A site loaded from an archive (or, for the transition, from a generated site folder).

    site, prior   dicts (site.json, prior.json)
    fields        dict of arrays (fields.npz)
    trav_arrays   dict of arrays (trav.npz); trav_params: thresholds and the mode-switch time
    scene_xml     the MJCF text
    assets        {file name: bytes}: the site's textures and the shared ones it uses
    manifest      the archive's manifest (a folder gets a synthetic one)
    """

    def __init__(self, members: dict, source: str, shared_dir: str | None = None, verify: bool = True,
                 folder: bool = False):
        self.source = source
        self.folder = folder
        if not folder:
            miss = [m for m in REQUIRED if m not in members]
            if miss:
                raise PackageError("%s: members missing %s" % (source, miss))
            man = json.loads(members["manifest.json"])
            if man.get("format") != FORMAT:
                raise PackageError("%s: unknown format %r" % (source, man.get("format")))
            if verify:
                for k, rec in man["members"].items():
                    if k not in members:
                        raise PackageError("%s: member %s missing" % (source, k))
                    if sha256(members[k]) != rec["sha256"]:
                        raise PackageError("%s: member %s does not match the manifest" % (source, k))
                extra = sorted(set(members) - set(man["members"]) - {"manifest.json"})
                if extra:
                    raise PackageError("%s: members not in the manifest %s" % (source, extra))
        else:
            man = {"format": "folder", "name": os.path.basename(os.path.normpath(source)), "members": {},
                   "shared_assets": {}}
        self.manifest = man
        self._m = members
        self.site = json.loads(members["site.json"])
        self.prior = json.loads(members["prior.json"])
        self.scene_xml = members["scene.xml"].decode()
        self.fields = npz_load(members["fields.npz"])
        self.trav_arrays = npz_load(members["trav.npz"])
        self.trav_params = _trav_params(self.site, self.trav_arrays)
        assets = {k[len("assets/"):]: v for k, v in members.items() if k.startswith("assets/")}
        shared_dir = shared_dir or shared_dir_default()
        for f, h in (man.get("shared_assets") or {}).items():
            p = os.path.join(shared_dir, f)
            if not os.path.exists(p):
                raise PackageError("%s: shared texture %s not found in %s" % (source, f, shared_dir))
            b = open(p, "rb").read()
            if verify and sha256(b) != h:
                raise PackageError("%s: shared texture %s does not match the manifest" % (source, f))
            assets[f] = b
        self.assets = assets

    @property
    def name(self) -> str:
        return str(self.manifest.get("name"))

    def mjcf_assets(self) -> dict:
        """The asset dict for mujoco.MjSpec.from_string / MjModel.from_xml_string (bare file names; MuJoCo resolves
        the scene's texturedir against them; a name may appear once only)."""
        return dict(self.assets)

    def spec(self):
        """mujoco.MjSpec of the scene alone (no robot)."""
        import mujoco
        return mujoco.MjSpec.from_string(self.scene_xml, assets=self.mjcf_assets())

    def model(self):
        """Compiled scene alone (no robot): (MjModel, MjData) with mj_forward run."""
        import mujoco
        m = mujoco.MjModel.from_xml_string(self.scene_xml, assets=self.mjcf_assets())
        d = mujoco.MjData(m)
        mujoco.mj_forward(m, d)
        return m, d


def _folder_members(site_dir: str) -> dict:
    rd = lambda f: open(os.path.join(site_dir, f), "rb").read()  # noqa: E731
    mem = {"site.json": rd("site.json"), "prior.json": rd("prior.json"), "scene.xml": rd("scene.xml"),
           "fields.npz": rd("fields.npz"), "trav.npz": rd("trav.npz")}
    adir = os.path.join(site_dir, "assets")
    if os.path.isdir(adir):
        for f in sorted(os.listdir(adir)):
            mem["assets/" + f] = rd(os.path.join("assets", f))
    return mem


def open_site(path: str, shared_dir: str | None = None, verify: bool = True) -> SitePackage:
    """A site package from an archive `<name>.tar.xz`, or (transition) from a generated site folder."""
    if isinstance(path, SitePackage):
        return path
    path = os.fspath(path)
    if os.path.isdir(path):
        return SitePackage(_folder_members(path), path, shared_dir, verify=False, folder=True)
    if not os.path.exists(path):
        raise PackageError("no site package at %s" % path)
    return SitePackage(read_tar_xz(path), path, shared_dir, verify=verify)
