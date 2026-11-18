"""Stage the cell's render bundle into the image (environment/Dockerfile; stdlib only, runs before any venv).

    python3 -I stage_assets.py <assets.sha256> <build dir> <dest dir> <auto|local|hub>

assets.sha256 (fetch_assets.py's format: a header, then `<sha256>  <relative path>` lines) pins every file of the
task's assets. `hub`: fetch_assets.py downloads each from the asset dataset at the pinned commit; `local`: each must be
in the build dir at its relative path (copied there from the build context's local_assets/, for builds on the maintainers'
machines without a download); `auto`: local when every file is there, else hub. Either way a file whose
sha256 is not the pinned one fails the build. Then every listed .tar is unpacked under the dest dir as the folder
<dest>/<the tar's name without .tar> (a tar whose members all sit in that one folder is unpacked as it is, any other
into that folder), and every file the bundle's manifest.json lists ({"name" or "bundle", "files": {relative path:
sha256 or {"sha256": ...}}}, the look bundle's format) is checked; a missing manifest, an empty list, a missing or
altered file fail the build. <dest>/look is then a symbolic link to the bundle's folder: the one path the cell reads.
"""

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fetch_assets  # noqa: E402  (the benchmark's canonical copy, byte-identical, next to this file)

LINK = "look"                     # <dest>/look -> the bundle's folder


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def verify_manifest(root):
    """(name, number of files) of the unpacked bundle at root; raises on any problem."""
    p = os.path.join(root, "manifest.json")
    if not os.path.isfile(p):
        raise SystemExit("[assets] no manifest.json in the bundle at %s" % root)
    m = json.load(open(p))
    files = m.get("files") or {}
    if not files:
        raise SystemExit("[assets] the bundle's manifest lists no files")
    bad = []
    for rel, rec in sorted(files.items()):
        digest = rec.get("sha256") if isinstance(rec, dict) else rec
        f = os.path.join(root, rel)
        if os.path.isabs(rel) or ".." in rel.split("/") or not os.path.isfile(f) or sha256(f) != digest:
            bad.append(rel)
    if bad:
        raise SystemExit("[assets] %d bundle file(s) missing or altered: %s" % (len(bad), bad[:10]))
    return m.get("name") or m.get("bundle") or os.path.basename(root), len(files)


def main():
    lst, build, dest, mode = sys.argv[1:5]
    if mode not in ("auto", "local", "hub"):
        raise SystemExit("[assets] QINSPECT_ASSETS must be auto, local or hub (got %r)" % mode)
    head, files = fetch_assets.read_list(lst)
    have = all(os.path.isfile(os.path.join(build, rel)) for _, rel in files)
    use = "local" if mode == "local" or (mode == "auto" and have) else "hub"
    payload = os.path.join(os.path.dirname(dest.rstrip("/")), "payload")
    shutil.rmtree(payload, ignore_errors=True)
    os.makedirs(payload)
    if use == "local":
        for digest, rel in files:
            src = os.path.join(build, rel)
            if not os.path.isfile(src):
                raise SystemExit("[assets] QINSPECT_ASSETS=local but local_assets/%s is not in the build context" % rel)
            got = sha256(src)
            if got != digest:
                raise SystemExit("[assets] local_assets/%s: sha256 %s, assets.sha256 pins %s" % (rel, got, digest))
            os.makedirs(os.path.dirname(os.path.join(payload, rel)), exist_ok=True)
            shutil.copyfile(src, os.path.join(payload, rel))
    else:
        subprocess.run([sys.executable, "-I", os.path.join(HERE, "fetch_assets.py"), lst, payload], check=True)
    os.makedirs(dest, exist_ok=True)
    done = []
    for _, rel in files:
        if not rel.endswith(".tar"):
            continue
        stem = os.path.basename(rel)[:-len(".tar")]
        root = os.path.join(dest, stem)
        shutil.rmtree(root, ignore_errors=True)
        with tarfile.open(os.path.join(payload, rel)) as t:
            names = []
            for m in t.getmembers():
                if os.path.isabs(m.name) or ".." in m.name.split("/") or not (m.isfile() or m.isdir()):
                    raise SystemExit("[assets] unsafe member %r in %s" % (m.name, rel))
                names.append(m.name.rstrip("/"))
            nested = all(n == stem or n.startswith(stem + "/") for n in names)
            t.extractall(dest if nested else root, filter="data")
        name, n = verify_manifest(root)
        done.append(root)
        print("[assets] render bundle %s: %d files verified in %s (source %s, %s@%s)" % (
            name, n, root, use, head.get("repo"), str(head.get("revision"))[:12]))
    shutil.rmtree(payload, ignore_errors=True)
    if len(done) != 1:
        raise SystemExit("[assets] assets.sha256 must list exactly one bundle (.tar); it lists %d" % len(done))
    # the one path every reader uses (SITE_GEN_DRESS_BUNDLE in the image, tests/test.sh, the practice service): a fixed
    # name for whichever bundle the list pins
    link = os.path.join(dest, LINK)
    if os.path.lexists(link):
        os.remove(link)
    os.symlink(os.path.basename(done[0]), link)
    print("[assets] %s -> %s" % (link, os.path.basename(done[0])))


if __name__ == "__main__":
    main()
