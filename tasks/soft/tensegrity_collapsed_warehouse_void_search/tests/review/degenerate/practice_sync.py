"""Refresh the practice ground's copies of the station (maintainer tool; run again whenever tests/voidcell/ or
tests/voidscore/ changes, then re-check the image's manifests).

    practice_sync.py sync      environment/practice_ground/voidcell/ = tests/voidcell/ byte for byte, plus
                               voidcell/voidscore/ = tests/voidscore/ byte for byte (no __pycache__); no reference
                               program and no token ever go there (tests/reference/, tests/hidden/ are not copied)
    practice_sync.py check     exit 1 unless both copies are byte-identical to their sources and nothing else is there
    practice_sync.py cal DIR   environment/practice_ground/cal/cal.tar.xz = DIR/cal.tar.xz (calgen's output)
"""
import hashlib
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DEV = os.path.normpath(os.path.join(HERE, "..", ".."))
TASK = os.path.join(os.environ["ARB_SRC_ROOT"], os.path.basename(DEV)[:-len("-dev")]) if os.environ.get("ARB_SRC_ROOT") else DEV[:-len("-dev")]
SRC_CELL = os.path.join(TASK, "tests", "voidcell")
SRC_SCORE = os.path.join(TASK, "tests", "voidscore")
PG = os.path.join(TASK, "environment", "practice_ground")
DST_CELL = os.path.join(PG, "voidcell")
DST_SCORE = os.path.join(DST_CELL, "voidscore")
FORBIDDEN = ("reference", "oracle_token", "hidden")


def files(root, skip=()):
    out = {}
    for dp, dns, fns in os.walk(root):
        dns[:] = sorted(d for d in dns if d != "__pycache__" and os.path.join(dp, d) not in skip)
        for f in sorted(fns):
            if f.endswith(".pyc"):
                continue
            p = os.path.join(dp, f)
            with open(p, "rb") as fh:
                out[os.path.relpath(p, root)] = hashlib.sha256(fh.read()).hexdigest()
    return out


def check():
    a, b = files(SRC_CELL), files(DST_CELL, skip=(DST_SCORE,))
    c, d = files(SRC_SCORE), files(DST_SCORE)
    bad = sorted(set(a) ^ set(b)) + [k for k in a if k in b and a[k] != b[k]]
    bad += ["voidscore/" + k for k in sorted(set(c) ^ set(d))] + ["voidscore/" + k for k in c if k in d and c[k] != d[k]]
    leaks = [k for k in list(b) + ["voidscore/" + k for k in d] if any(w in k for w in FORBIDDEN)]
    return bad, leaks, len(b) + len(d)


def main(argv):
    cmd = argv[1] if len(argv) > 1 else "check"
    if cmd == "sync":
        if os.path.isdir(DST_CELL):
            shutil.rmtree(DST_CELL)
        ig = shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache")
        shutil.copytree(SRC_CELL, DST_CELL, ignore=ig)
        shutil.copytree(SRC_SCORE, DST_SCORE, ignore=ig)
    elif cmd == "cal":
        os.makedirs(os.path.join(PG, "cal"), exist_ok=True)
        shutil.copyfile(os.path.join(argv[2], "cal.tar.xz"), os.path.join(PG, "cal", "cal.tar.xz"))
        with open(os.path.join(PG, "cal", "cal.tar.xz"), "rb") as fh:
            print("cal.tar.xz", hashlib.sha256(fh.read()).hexdigest())
        return 0
    bad, leaks, n = check()
    print("practice ground copies: %d files, %d differences %s, %d forbidden names %s" % (n, len(bad), bad[:5], len(leaks),
                                                                                        leaks[:5]))
    return 1 if (bad or leaks) else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
