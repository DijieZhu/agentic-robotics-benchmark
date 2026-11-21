"""Stage the task's assets (the photo set) into the image (environment/Dockerfile; standard library only).

    python3 -I stage_assets.py <assets.sha256> <build dir> <payload dir> <auto|local|hub>

assets.sha256 (fetch_assets.py's format: a header, then one `<sha256>  <relative path>` line per file) pins every file
of the task's assets. Where they come from (the build argument FFCAD_ASSETS):
  local  every listed file must be in the build dir at its relative path (the Dockerfile copies the build context's
         local_assets/ there; builds before the files are in the asset dataset);
  hub    fetch_assets.py downloads every file from the asset dataset at the commit the list pins;
  auto   local when every listed file is in the build dir, else hub.
Either way the payload dir ends up holding exactly the listed files, each one checked against its pinned sha256. The
build fails (exit status 1, the reason on stderr) on: a listed file missing or altered; a path that is absolute, climbs
out with '..' or is listed twice; a file in the build dir that the list does not name; an empty list; and `hub` while
the list pins the placeholder commit (40 zeros) or anything other than a 40-hex commit.
"""
import hashlib
import os
import re
import shutil
import subprocess
import sys

sys.dont_write_bytecode = True    # the build dir must hold the listed files and the three tools only
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fetch_assets  # noqa: E402  (the benchmark's canonical copy, byte-identical, next to this file)

MODES = ("auto", "local", "hub")
TOOLS = ("fetch_assets.py", "stage_assets.py", "assets.sha256")   # the build dir's own files, beside the assets
PLACEHOLDER = "0" * 40
COMMIT = re.compile(r"^[0-9a-f]{40}$")
DIGEST = re.compile(r"^[0-9a-f]{64}$")


def fail(msg):
    raise SystemExit("[assets] " + msg)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def safe(rel):
    return bool(rel) and not os.path.isabs(rel) and ".." not in rel.split("/") and "\\" not in rel \
        and os.path.normpath(rel) == rel


def files_under(root):
    out = []
    for d, dirs, files in os.walk(root):
        dirs.sort()
        out += [os.path.relpath(os.path.join(d, f), root) for f in sorted(files)]
    return sorted(out)


def read_list(lst):
    head, files = fetch_assets.read_list(lst)        # exits when a header line is missing
    if not files:
        fail("%s lists no files" % lst)
    seen = set()
    for digest, rel in files:
        if not DIGEST.match(digest):
            fail("%s: %r is not a sha256 (line for %s)" % (lst, digest, rel))
        if not safe(rel) or rel in TOOLS:
            fail("%s: unsafe or reserved path %r" % (lst, rel))
        if rel in seen:
            fail("%s: %s is listed twice" % (lst, rel))
        seen.add(rel)
    return head, files


def hub_refusal(head, mode, have, n):
    rev = head.get("revision", "")
    if rev == PLACEHOLDER:
        why = "assets.sha256 pins the placeholder revision %s: the files are not in the asset dataset yet" % rev
    elif not COMMIT.match(rev):
        why = "assets.sha256 pins revision %r, not a 40-hex commit of the asset dataset" % rev
    else:
        return None
    if mode == "auto":
        why = "FFCAD_ASSETS=auto found %d of %d listed file(s) in local_assets/, so it would fetch them, but %s" % (
            have, n, why)
    else:
        why = "FFCAD_ASSETS=hub, but %s" % why
    return why + "; build with every listed file in the build context's local_assets/ (FFCAD_ASSETS=local or auto)"


def main(argv):
    if len(argv) != 4:
        fail("usage: stage_assets.py <assets.sha256> <build dir> <payload dir> <auto|local|hub>")
    lst, build, payload, mode = argv
    if mode not in MODES:
        fail("FFCAD_ASSETS must be auto, local or hub (got %r)" % mode)
    head, files = read_list(lst)
    listed = {rel for _, rel in files}
    unlisted = [r for r in files_under(build) if r not in listed and r not in TOOLS]
    if unlisted:
        fail("%d file(s) in local_assets/ that assets.sha256 does not list (they would sit in an image layer): %s"
             % (len(unlisted), unlisted[:10]))
    have = [rel for _, rel in files if os.path.isfile(os.path.join(build, rel))]
    use = "local" if mode == "local" or (mode == "auto" and len(have) == len(files)) else "hub"
    if os.path.isdir(payload) and os.listdir(payload):
        fail("the payload dir %s is not empty" % payload)
    if use == "local":
        missing = [rel for _, rel in files if rel not in have]
        if missing:
            fail("FFCAD_ASSETS=local but %d listed file(s) are not in the build context's local_assets/: %s"
                 % (len(missing), missing[:10]))
        os.makedirs(payload, exist_ok=True)
        for digest, rel in files:
            src = os.path.join(build, rel)
            got = sha256(src)
            if got != digest:
                fail("local_assets/%s: sha256 %s, assets.sha256 pins %s" % (rel, got, digest))
            os.makedirs(os.path.dirname(os.path.join(payload, rel)) or payload, exist_ok=True)
            shutil.copyfile(src, os.path.join(payload, rel))
    else:
        why = hub_refusal(head, mode, len(have), len(files))
        if why:
            fail(why)
        os.makedirs(payload, exist_ok=True)
        rc = subprocess.run([sys.executable, "-I", os.path.join(HERE, "fetch_assets.py"), lst, payload]).returncode
        if rc != 0:
            fail("fetch_assets.py failed (exit status %d) for %s@%s" % (rc, head.get("repo"), head.get("revision")))
    # every file once more, and nothing else in the payload
    got = files_under(payload)
    if got != sorted(listed):
        fail("the payload holds %d file(s), the list %d: extra %s, missing %s" % (
            len(got), len(listed), sorted(set(got) - listed)[:10], sorted(listed - set(got))[:10]))
    total = 0
    for digest, rel in files:
        p = os.path.join(payload, rel)
        if sha256(p) != digest:
            fail("%s: sha256 differs from the pinned %s after staging" % (rel, digest))
        total += os.path.getsize(p)
    print("[assets] %d file(s), %d bytes, each sha256 checked, staged into %s (source %s, FFCAD_ASSETS=%s, %s@%s)" % (
        len(files), total, payload, use, mode, head.get("repo"), str(head.get("revision"))[:12]))


if __name__ == "__main__":
    main(sys.argv[1:])
