"""sitepack.py -- the site package tool: the deterministic site package
(the site package format), reference implementation.  Standard library only (the cell's reader needs only
``unpack`` and ``verify``).

    pack(members: {name: bytes}, site_id) -> bytes     adds MANIFEST.json; USTAR, mtime 0, uid/gid 0, 0644, sorted; xz CRC64 preset 6
    unpack(data: bytes) -> {name: bytes}
    verify(data: bytes, site_schema=None) -> [problems]  (re-pack identical, manifest hashes, required members, schema, state0 hash)
    dumps(obj) -> bytes                                 the canonical JSON member bytes
    python -I -B sitepack.py pack DIR OUT.site.tar.xz | verify FILE | list FILE
"""
from __future__ import annotations

import hashlib
import io
import json
import lzma
import os
import sys
import tarfile

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
PACKAGE_SCHEMA = "caretaking.site_package/1"
REQUIRED = ("MANIFEST.json", "site.json", "state0.json", "mission/advisory_log.json", "mission/config_record.json", "mission/log.csv",
            "mission/register.json", "mission/work_order.txt")
SUFFIX = ".site.tar.xz"


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def dumps(obj) -> bytes:
    return (json.dumps(obj, indent=1, sort_keys=True, ensure_ascii=True) + "\n").encode("ascii")


def _tar(members: dict) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w", format=tarfile.USTAR_FORMAT) as tf:
        for name in sorted(members):
            data = members[name]
            ti = tarfile.TarInfo(name)
            ti.size = len(data)
            ti.mtime = 0
            ti.mode = 0o644
            ti.uid = ti.gid = 0
            ti.uname = ti.gname = ""
            ti.type = tarfile.REGTYPE
            tf.addfile(ti, io.BytesIO(data))
    return buf.getvalue()


def pack(members: dict, site_id: str) -> bytes:
    """The package bytes of ``members`` (MANIFEST.json is (re)built here)."""
    body = {k: bytes(v) for k, v in members.items() if k != "MANIFEST.json"}
    for k in body:
        if k.startswith("/") or ".." in k.split("/") or len(k) > 99:
            raise ValueError(f"bad member name {k!r}")
    body["MANIFEST.json"] = dumps({"schema": PACKAGE_SCHEMA, "site_id": site_id, "members": {k: sha(body[k]) for k in sorted(body)}})
    return lzma.compress(_tar(body), format=lzma.FORMAT_XZ, check=lzma.CHECK_CRC64, preset=6)


def unpack(data: bytes) -> dict:
    raw = lzma.decompress(data)
    out = {}
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:") as tf:
        for ti in tf.getmembers():
            if not ti.isreg():
                raise ValueError(f"non-file member {ti.name!r}")
            out[ti.name] = tf.extractfile(ti).read()
    return out


def verify(data: bytes, site_schema=None) -> list:
    probs = []
    try:
        mem = unpack(data)
    except (lzma.LZMAError, tarfile.TarError, ValueError) as exc:
        return [f"unreadable package: {exc}"]
    for r in REQUIRED:
        if r not in mem:
            probs.append(f"missing member {r}")
    if "MANIFEST.json" in mem:
        man = json.loads(mem["MANIFEST.json"])
        if man.get("schema") != PACKAGE_SCHEMA:
            probs.append("MANIFEST.json schema")
        listed = man.get("members", {})
        for k, h in listed.items():
            if k not in mem:
                probs.append(f"manifest lists a missing member {k}")
            elif sha(mem[k]) != h:
                probs.append(f"sha256 of {k} differs from the manifest")
        extra = sorted(set(mem) - set(listed) - {"MANIFEST.json"})
        if extra:
            probs.append(f"members not in the manifest: {extra}")
        if pack(mem, man.get("site_id", "")) != data:
            probs.append("not the deterministic packing of its members")
    if "site.json" in mem:
        site = json.loads(mem["site.json"])
        if site_schema is not None:
            sys.path.insert(0, HERE)
            import schemalite as SL
            errs = SL.errors(site, site_schema)
            if errs:
                probs.append(f"site.json does not validate: {errs[:3]}")
        h = (site.get("settle") or {}).get("state0_sha256")
        if h and "state0.json" in mem and sha(mem["state0.json"]) != h:
            probs.append("settle.state0_sha256 differs from state0.json")
        if "MANIFEST.json" in mem and json.loads(mem["MANIFEST.json"]).get("site_id") != site.get("site_id"):
            probs.append("manifest site_id differs from site.json")
    return probs


def main(argv=None):
    a = list(sys.argv[1:] if argv is None else argv)
    if not a:
        print(__doc__)
        return 2
    if a[0] == "pack" and len(a) == 3:
        d = a[1]
        mem = {}
        for root, _, fs in os.walk(d):
            for f in fs:
                p = os.path.join(root, f)
                mem[os.path.relpath(p, d).replace(os.sep, "/")] = open(p, "rb").read()
        site_id = json.loads(mem["site.json"])["site_id"]
        data = pack(mem, site_id)
        open(a[2], "wb").write(data)
        print(f"{a[2]}: {len(data)} bytes sha256 {sha(data)}")
        return 0
    if a[0] in ("verify", "list") and len(a) == 2:
        data = open(a[1], "rb").read()
        if a[0] == "list":
            for k, v in sorted(unpack(data).items()):
                print(f"{len(v):10d}  {sha(v)}  {k}")
            return 0
        schema = json.load(open(os.path.join(HERE, "site.schema.json"), encoding="utf-8"))
        probs = verify(data, schema)
        print(f"{a[1]}: {'ok' if not probs else 'FAILED'}", *probs, sep="\n  ")
        return 0 if not probs else 1
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main())
