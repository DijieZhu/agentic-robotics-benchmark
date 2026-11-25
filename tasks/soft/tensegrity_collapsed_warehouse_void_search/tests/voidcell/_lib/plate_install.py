"""Plate bundles from their published form into the folder the plate backend reads (standard library only; no import
of the cell, so it also runs as a script).

The bundles travel under neutral folder names: the practice ground's (cal, p1, p2, p3) in its asset list, the
grader's (h01 ... h10) in one tarball. Installing renames each folder to the name the plate backend looks for, the site
package sha256 that the bundle's own meta.json records (plate_bundle.py checks the whole bundle again whenever a camera
opens). Before anything moves it checks that every recorded sha256 is 64 lowercase hexadecimal digits, that no two
bundles get the same name and that each bundle belongs to the site of the same name (`expect`: neutral name -> the
sha256 the site is pinned to; the set of names must be expect's). The grader and the practice ground's image run the
same code: the grader through fetch_tarball, unpack_tarball and install; the image step through install on the
folder its asset fetch wrote (`practice` below).

    path, sha = fetch_tarball(spec, dest, local=None, log=None)   # spec: {"url", "sha256", "files"}; local replaces
                                                                  # the download; the sha256 is checked either way
    ok = holds_only_bundles(folder)                               # nothing but folders named by a sha256
    folders, n = unpack_tarball(tarball, part_dir)                # {neutral name: folder}, n regular files
    names = install(folders, dst_root, expect)                    # {neutral name: sha256}; folders moved into dst_root

    python3 plate_install.py practice <fetched folder> <plates folder> <sites.json>

A tarball holds regular files only, each under one neutral top folder (`<name>/<path>`), each name once; any other
member (a folder entry, a link, a device, an absolute or dotted path, a name that is there twice) is refused before
anything is unpacked. Every refusal and every failure to download, copy or unpack is a PlateInstallError.
"""
import hashlib
import json
import os
import re
import shutil
import sys
import time

_SHA = re.compile(r"[0-9a-f]{64}\Z")
_NEUTRAL = re.compile(r"[a-z][a-z0-9]{0,15}\Z")
_PART = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.\-]*\Z")


class PlateInstallError(Exception):
    """A tarball or a folder of plate bundles that cannot be installed as it is."""


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fetch_tarball(spec, dest, local=None, log=None, tries=4, timeout_s=120.0, deadline_s=900.0):
    """Copy `local` or download spec["url"] to `dest` (a new file), then check spec["sha256"]. Returns (dest, sha256).
    A download gets `tries` attempts and `deadline_s` seconds in all."""
    want = str(spec.get("sha256") or "").strip().lower()
    if not _SHA.match(want):
        raise PlateInstallError("the pinned sha256 of the tarball is not 64 lowercase hexadecimal digits")
    if os.path.lexists(dest):
        os.remove(dest)
    src = local
    if local:
        try:
            shutil.copyfile(local, dest)
        except OSError as e:
            raise PlateInstallError("could not copy the local tarball: %s" % type(e).__name__)
    else:
        import urllib.request
        src = str(spec.get("url") or "")
        t0 = time.monotonic()
        for k in range(tries):
            try:
                with urllib.request.urlopen(src, timeout=timeout_s) as r, open(dest, "wb") as fh:
                    for block in iter(lambda: r.read(1 << 20), b""):
                        fh.write(block)
                        if time.monotonic() - t0 > deadline_s:
                            raise TimeoutError("not complete after %d s" % deadline_s)
                break
            except Exception as e:  # noqa: BLE001 - retried, then the caller's error
                if log is not None:
                    log("plate bundles: download %d/%d failed: %r" % (k + 1, tries, e))
                if k == tries - 1 or time.monotonic() - t0 > deadline_s:
                    raise PlateInstallError("could not download %s: %r" % (src, e))
                time.sleep(5 * (k + 1))
    got = sha256_file(dest)
    if got != want:
        raise PlateInstallError("%s does not have the pinned sha256" % src)
    return dest, got


def unpack_tarball(tarball, part_dir):
    """Unpack the xz tarball into the new folder part_dir. Returns ({neutral name: folder}, number of files)."""
    import lzma
    import tarfile
    try:
        os.makedirs(part_dir, mode=0o700)
        with tarfile.open(tarball, "r:xz") as tf:
            members = tf.getmembers()
            bad, tops = [], set()
            for m in members:
                parts = m.name.split("/")
                if (not m.isfile() or len(parts) < 2 or not _NEUTRAL.match(parts[0])
                        or not all(_PART.match(p) and p not in (".", "..") for p in parts[1:])):
                    bad.append(m.name)
                else:
                    tops.add(parts[0])
            if bad:
                raise PlateInstallError("the tarball holds members that are not regular files under a bundle folder: %s"
                                        % bad[:5])
            if not members:
                raise PlateInstallError("the tarball is empty")
            if len({m.name for m in members}) != len(members):
                raise PlateInstallError("the tarball holds a member name twice")
            tf.extractall(part_dir, filter="data")
    except (OSError, EOFError, lzma.LZMAError, tarfile.TarError) as e:
        raise PlateInstallError("the tarball does not unpack: %s" % type(e).__name__)
    return {t: os.path.join(part_dir, t) for t in sorted(tops)}, len(members)


def holds_only_bundles(folder):
    """True when `folder` is a real folder with nothing in it but folders named by a sha256 (installed bundles)."""
    if os.path.islink(folder) or not os.path.isdir(folder):
        return False
    return all(_SHA.match(n) and os.path.isdir(os.path.join(folder, n)) and not os.path.islink(os.path.join(folder, n))
               for n in os.listdir(folder))


def bundle_name(folder):
    """The site package sha256 that the bundle's meta.json records: the folder name the plate backend looks for."""
    p = os.path.join(folder, "meta.json")
    if os.path.islink(folder) or os.path.islink(p) or not os.path.isfile(p):
        raise PlateInstallError("the bundle %s has no meta.json" % os.path.basename(folder))
    try:
        with open(p, "rb") as fh:
            sha = json.loads(fh.read().decode("utf-8"))["site_package_sha256"]
    except (OSError, ValueError, KeyError, TypeError) as e:
        raise PlateInstallError("the bundle %s: meta.json is unreadable (%s)" % (os.path.basename(folder), type(e).__name__))
    if not isinstance(sha, str) or not _SHA.match(sha):
        raise PlateInstallError("the bundle %s: meta.json's site_package_sha256 is not 64 lowercase hexadecimal digits"
                                % os.path.basename(folder))
    return sha


def install(folders, dst_root, expect=None):
    """Move every folder of `folders` ({neutral name: folder}) to dst_root/<site package sha256>. With `expect`
    ({neutral name: sha256}) the names must be expect's and every bundle's sha256 the expected one. Everything is checked
    before the first folder moves. Returns {neutral name: sha256}."""
    names = {k: bundle_name(folders[k]) for k in sorted(folders)}
    if expect is not None:
        if sorted(names) != sorted(expect):
            raise PlateInstallError("bundles %s do not match the sites %s" % (sorted(names), sorted(expect)))
        wrong = [k for k in names if names[k] != str(expect[k]).strip().lower()]
        if wrong:
            raise PlateInstallError("the bundle(s) %s belong to other site packages than the sites of the same name" % wrong)
    seen = {}
    for k, sha in names.items():
        if sha in seen:
            raise PlateInstallError("the bundles %s and %s carry the same site package sha256" % (seen[sha], k))
        seen[sha] = k
    try:
        os.makedirs(dst_root, exist_ok=True)
    except OSError as e:
        raise PlateInstallError("the plates folder cannot be made: %s" % type(e).__name__)
    taken = [k for k, sha in names.items() if os.path.lexists(os.path.join(dst_root, sha))]
    if taken:
        raise PlateInstallError("the plates folder already holds the bundle(s) of %s" % taken)
    try:
        for k, sha in names.items():
            shutil.move(folders[k], os.path.join(dst_root, sha))
    except OSError as e:
        raise PlateInstallError("could not move the bundles into place: %s" % type(e).__name__)
    return names


def practice(fetched, dst_root, sites_json):
    """The practice ground's image step: the bundles its asset fetch wrote (one neutral folder per site of sites.json)
    into dst_root under their site package sha256, each checked against the site of the same name."""
    with open(sites_json, encoding="utf-8") as fh:
        doc = json.load(fh)
    expect = {str(s["name"]): str(s["sha256"]) for s in doc["sites"]}
    folders = {d: os.path.join(fetched, d) for d in sorted(os.listdir(fetched))
               if os.path.isdir(os.path.join(fetched, d)) and not os.path.islink(os.path.join(fetched, d))}
    return install(folders, dst_root, expect)


def main(argv):
    if len(argv) == 5 and argv[1] == "practice":
        names = practice(argv[2], argv[3], argv[4])
        print("plate bundles installed: %s" % ", ".join(sorted(names)))
        return 0
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
