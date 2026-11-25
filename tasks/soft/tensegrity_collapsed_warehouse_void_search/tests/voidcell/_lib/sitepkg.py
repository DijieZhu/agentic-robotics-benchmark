"""Site packages: a folder or a .tar.xz archive holding site.json (all coordinates and the truth, hidden side) and
site.xml (the world only). An archive is read in memory: only regular members whose base name is site.json or
site.xml are read (no extraction, no links, no absolute or parent paths); anything else in it is ignored.

sha256 of a package: of the archive's bytes; of a folder, of the two members (name and sha256 of each, sorted), so
the same content gives the same hash either way round only within its own form (records state which).
"""
import hashlib
import json
import os
import tarfile

MEMBERS = ("site.json", "site.xml")
MAX_MEMBER_BYTES = 8 << 20


def _sha_bytes(b):
    return hashlib.sha256(b).hexdigest()


def _members_sha(members):
    h = hashlib.sha256()
    for name in sorted(members):
        h.update(("%s %s\n" % (name, _sha_bytes(members[name]))).encode())
    return h.hexdigest()


def read_members(path):
    """{name: bytes} of site.json and site.xml of a package folder or archive."""
    path = os.fspath(path)
    out = {}
    if os.path.isdir(path):
        for name in MEMBERS:
            p = os.path.join(path, name)
            if not os.path.isfile(p) or os.path.islink(p):
                raise ValueError("site package %s: %s missing or not a regular file" % (path, name))
            with open(p, "rb") as fh:
                out[name] = fh.read(MAX_MEMBER_BYTES + 1)
    else:
        with tarfile.open(path, mode="r:*") as tf:
            for ti in tf.getmembers():
                base = os.path.basename(ti.name)
                if base not in MEMBERS or not ti.isreg():
                    continue
                if ti.name.startswith("/") or ".." in ti.name.split("/"):
                    raise ValueError("site archive %s: unsafe member name %r" % (path, ti.name))
                if base in out:
                    raise ValueError("site archive %s: %s twice" % (path, base))
                fh = tf.extractfile(ti)
                out[base] = fh.read(MAX_MEMBER_BYTES + 1)
    for name in MEMBERS:
        if name not in out:
            raise ValueError("site package %s: no %s" % (path, name))
        if len(out[name]) > MAX_MEMBER_BYTES:
            raise ValueError("site package %s: %s larger than %d bytes" % (path, name, MAX_MEMBER_BYTES))
    return out


def open_site(path):
    """(site dict, site.xml text, provenance dict) of a package."""
    path = os.fspath(path)
    m = read_members(path)
    if os.path.isdir(path):
        sha = _members_sha(m)
        form = "folder: sha256 of the sorted (name, sha256) lines of site.json and site.xml"
    else:
        with open(path, "rb") as fh:
            sha = _sha_bytes(fh.read())
        form = "archive: sha256 of the archive's bytes"
    site = json.loads(m["site.json"].decode("utf-8"))
    xml = m["site.xml"].decode("utf-8")
    prov = {"path": os.path.abspath(path), "sha256": sha, "sha256_rule": form,
            "site_json_sha256": _sha_bytes(m["site.json"]), "site_xml_sha256": _sha_bytes(m["site.xml"]),
            "generator": site.get("generator"), "family": site.get("family"), "seed": site.get("seed")}
    return site, xml, prov
