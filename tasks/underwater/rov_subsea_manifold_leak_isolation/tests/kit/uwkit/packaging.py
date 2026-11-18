"""Which uwkit files may go where, and byte-for-byte copies of the package for the task folders (maintainer tool).

Purposes (decisions K4, K5, U4):
  agent-sandbox     pure-numpy helpers only (frames, hydrostatics).  K4: the sensor models, the firmware and the
                    simulator run on the service / verifier side; the agent gets a client, the public spec tables and
                    the documents.  Must pass the leak-word scan (no platform, vendor or author names).
  practice-service  the simulator as the black-box practice facility runs it: everything except the grader-side
                    coefficient data (hidden/), the render-only twin (viz/) and the maintainer tools.  The service gets
                    the vehicle as an exported model (VehicleBuild.export(); K5: a model built with a drawn coefficient
                    set stays on the service / verifier side) and its coefficients passed explicitly (hydro=...).
  verifier          everything except the maintainer tools (incl. hidden/ and viz/); optionally the digest files
                    (spec_values.json, vehicle_parts_manifest.json) next to the package (include_digest=True).
Every file of the package must be classified below (packaging fails on an unclassified file), so a new module has to be
placed explicitly.  The copies carry uwkit/MANIFEST.sha256 (version, purpose, one sha256 per file): the four tasks'
copies of one uwkit version are identical file by file.

    python -I -B <shared>/uwkit/packaging.py --table                     # ownership table (markdown)
    python -I -B <shared>/uwkit/packaging.py --list agent-sandbox        # file list of a purpose
    python -I -B <shared>/uwkit/packaging.py verifier <dest> [--clean] [--with-digest]   # copy -> <dest>/uwkit/
    python -I -B <shared>/uwkit/packaging.py --verify <dest>             # re-hash a copy against its manifest
Only the standard library is used (the tool runs without numpy / mujoco).
"""
from __future__ import annotations

import fnmatch
import hashlib
import json
import os
import re
import shutil
import sys
from typing import Optional

PKG_DIR = os.path.dirname(os.path.abspath(__file__))
SHARED_DIR = os.path.dirname(PKG_DIR)
PURPOSES = ("agent-sandbox", "practice-service", "verifier")
LEVELS = ("agent-sandbox", "practice-service", "verifier", "maintainer")     # each file: the most public level allowed
MANIFEST_NAME = "MANIFEST.sha256"
DIGEST_FILES = ("spec_values.json", "vehicle_parts_manifest.json")

# (glob relative to the package directory, level, why) - first match wins
OWNERSHIP = [
    ("VERSION", "agent-sandbox", "version string"),
    ("__init__.py", "agent-sandbox", "imports what is present (subset copies)"),
    ("frames.py", "agent-sandbox", "quaternions / Euler angles / frame conventions; pure numpy"),
    ("hydrostatics.py", "agent-sandbox", "part tables, static trim (free / depth hold), TEOS-10 density; pure numpy "
                                         "(the digest helpers need the verifier-side digest files)"),
    ("packaging.py", "maintainer", "this tool (names the hidden directory)"),
    ("viz/build_rotor_split.py", "maintainer", "rebuilds viz data from the maintainer's per-solid CAD cache"),
    ("hidden/*", "verifier", "grader-side coefficient sets and hidden-world ranges (author names, decision U4)"),
    ("viz/*", "verifier", "render-only visual twin and replay (evidence videos); never in a sensor path"),
    ("refdata.py", "practice-service", "digest file access (the digest files themselves are verifier-only)"),
    ("params.py", "practice-service", "parameter containers and loaders (wording mentions the grader-side files)"),
    ("sim.py", "practice-service", "the simulator"),
    ("dynamics/*", "practice-service", "hydro layer, thrusters, battery, current fields"),
    ("control/*", "practice-service", "firmware: allocator and autopilot (runs on the service side, K4)"),
    ("nav/*", "practice-service", "firmware: navigation estimate"),
    ("sensors/*", "practice-service", "sensor models (service / verifier side only, K4); *_public_spec() output may "
                                      "go to the agent's manual"),
    ("models/*", "practice-service", "vehicle builder, collision proxies, model parameters, thruster table"),
]


def _rel_files(root: str = PKG_DIR) -> list:
    out = []
    for dp, dn, fn in os.walk(root):
        dn[:] = sorted(d for d in dn if d != "__pycache__")
        for f in sorted(fn):
            if f.endswith((".pyc", ".pyo")) or f == MANIFEST_NAME:
                continue
            out.append(os.path.relpath(os.path.join(dp, f), root).replace(os.sep, "/"))
    return sorted(out)


def classify(rel: str) -> tuple:
    for pat, level, why in OWNERSHIP:
        if fnmatch.fnmatch(rel, pat):
            return level, why
    raise KeyError(f"uwkit/{rel} is not classified in uwkit/packaging.py OWNERSHIP")


def files_for(purpose: str, root: str = PKG_DIR) -> list:
    """Relative paths (under uwkit/) that the purpose receives."""
    if purpose not in PURPOSES:
        raise ValueError(f"purpose must be one of {PURPOSES}")
    lim = LEVELS.index(purpose)
    return [r for r in _rel_files(root) if LEVELS.index(classify(r)[0]) <= lim]


def ownership_table(root: str = PKG_DIR) -> list:
    rows = []
    for r in _rel_files(root):
        level, why = classify(r)
        k = LEVELS.index(level)
        rows.append(dict(file=r, level=level, why=why, **{p: k <= LEVELS.index(p) for p in PURPOSES}))
    return rows


def ownership_markdown(root: str = PKG_DIR) -> str:
    """Grouped markdown table: one row per OWNERSHIP pattern that matches files."""
    first = {r: next(p for p, _, _ in OWNERSHIP if fnmatch.fnmatch(r, p)) for r in _rel_files(root)}
    lines = ["| files | agent-sandbox | practice-service | verifier | why |", "|---|---|---|---|---|"]
    for pat, level, why in OWNERSHIP:
        hit = [r for r, p in first.items() if p == pat]
        if not hit:
            continue
        k = LEVELS.index(level)
        mark = ["yes" if k <= LEVELS.index(p) else "-" for p in PURPOSES]
        lines.append(f"| `{pat}` ({len(hit)}) | " + " | ".join(mark) + f" | {why} |")
    return "\n".join(lines)


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def version(root: str = PKG_DIR) -> str:
    with open(os.path.join(root, "VERSION")) as f:
        return f.read().strip()


# ---------------------------------------------------------------------------------------------------- leak scan
def load_leak_rules(path: Optional[str] = None) -> list:
    """Rules of shared/leak_words.txt (kind, label, regex); [] if the file is not available."""
    p = path or os.path.join(SHARED_DIR, "leak_words.txt")
    if not os.path.exists(p):
        return []
    rules = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            if line.startswith("#") or "\t" not in line:
                continue
            kind, lab, rx = line.rstrip("\n").split("\t")
            rules.append((kind, lab, re.compile(rx)))
    return rules


def leak_scan(paths, rules=None, meta=False) -> dict:
    """{file: [(kind, label, context), ...]} for the norm / exact rules (meta rules too if meta=True)."""
    rules = load_leak_rules() if rules is None else rules
    hits = {}
    for p in paths:
        if p.endswith((".npy", ".npz", ".png", ".stl", ".obj")):
            continue
        with open(p, encoding="utf-8", errors="replace") as f:
            txt = f.read()
        nt = txt.lower().replace("-", " ").replace("_", " ")
        for kind, lab, rx in rules:
            if kind == "meta" and not meta:
                continue
            src = txt if kind == "exact" else nt
            for mt in rx.finditer(src):
                s0 = max(0, mt.start() - 30)
                hits.setdefault(p, []).append((kind, lab, src[s0:mt.end() + 30].replace("\n", " ")))
    return hits


# ---------------------------------------------------------------------------------------------------- copy / verify
def package(purpose: str, dest: str, clean: bool = False, include_digest: bool = False, check_leaks: bool = True,
            root: str = PKG_DIR) -> dict:
    """Copy the purpose's files byte for byte to <dest>/uwkit/ and write <dest>/uwkit/MANIFEST.sha256.

    clean=True removes an existing <dest>/uwkit first (otherwise an existing copy is an error).  include_digest
    (verifier only) also copies spec_values.json and vehicle_parts_manifest.json to <dest>/ (where refdata looks by
    default).  check_leaks: the agent-sandbox copy must have no leak-word hit (raises), other purposes report theirs.
    Returns dict(version, purpose, files, sha256, leak_hits, dest)."""
    files = files_for(purpose, root)
    out_pkg = os.path.join(os.path.abspath(dest), "uwkit")
    if os.path.exists(out_pkg):
        if not clean:
            raise FileExistsError(f"{out_pkg} exists (use clean=True to replace it)")
        shutil.rmtree(out_pkg)
    sums = {}
    for r in files:
        src = os.path.join(root, r)
        dst = os.path.join(out_pkg, r)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copyfile(src, dst)
        sums[r] = sha256_file(dst)
        if sums[r] != sha256_file(src):
            raise RuntimeError(f"copy of {r} differs from the source")
    extra = []
    if include_digest:
        if purpose != "verifier":
            raise ValueError("the digest files are verifier-side only (they carry real part names)")
        for f in DIGEST_FILES:
            shutil.copyfile(os.path.join(os.path.dirname(root), f), os.path.join(os.path.abspath(dest), f))
            extra.append(f)
    ver = version(root)
    with open(os.path.join(out_pkg, MANIFEST_NAME), "w") as f:
        f.write(f"# uwkit {ver} purpose={purpose} files={len(files)}\n")
        for r in files:
            f.write(f"{sums[r]}  {r}\n")
    hits = {}
    if check_leaks:
        hits = leak_scan([os.path.join(out_pkg, r) for r in files] + [os.path.join(out_pkg, MANIFEST_NAME)])
        hits = {os.path.relpath(k, out_pkg): v for k, v in hits.items()}
        if purpose == "agent-sandbox" and hits:
            shutil.rmtree(out_pkg)
            raise RuntimeError(f"agent-sandbox copy has leak-word hits: {sorted(hits)} (copy removed)")
    return dict(version=ver, purpose=purpose, files=files, sha256=sums, digest_files=extra, leak_hits=hits,
                dest=out_pkg)


def verify(dest: str) -> dict:
    """Re-hash a copy against its MANIFEST.sha256: dict(ok, version, purpose, missing, changed, unexpected)."""
    out_pkg = os.path.join(os.path.abspath(dest), "uwkit")
    with open(os.path.join(out_pkg, MANIFEST_NAME)) as f:
        head = f.readline()
        want = dict(reversed(line.rstrip("\n").split("  ", 1)) for line in f if line.strip())
    m = re.match(r"# uwkit (\S+) purpose=(\S+)", head)
    have = set(_rel_files(out_pkg))
    missing = sorted(set(want) - have)
    unexpected = sorted(have - set(want))
    changed = sorted(r for r in set(want) & have if sha256_file(os.path.join(out_pkg, r)) != want[r])
    return dict(ok=not (missing or changed or unexpected), version=m.group(1) if m else None,
                purpose=m.group(2) if m else None, missing=missing, changed=changed, unexpected=unexpected)


def _main(argv):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("purpose", nargs="?", choices=PURPOSES)
    ap.add_argument("dest", nargs="?")
    ap.add_argument("--clean", action="store_true")
    ap.add_argument("--with-digest", action="store_true")
    ap.add_argument("--list", choices=PURPOSES)
    ap.add_argument("--table", action="store_true")
    ap.add_argument("--verify")
    a = ap.parse_args(argv)
    if a.table:
        print(ownership_markdown())
    elif a.list:
        print("\n".join(files_for(a.list)))
    elif a.verify:
        r = verify(a.verify)
        print(json.dumps(r, indent=1))
        return 0 if r["ok"] else 1
    elif a.purpose and a.dest:
        r = package(a.purpose, a.dest, clean=a.clean, include_digest=a.with_digest)
        print(json.dumps(dict(version=r["version"], purpose=r["purpose"], n_files=len(r["files"]), dest=r["dest"],
                              digest_files=r["digest_files"],
                              leak_hits={k: len(v) for k, v in r["leak_hits"].items()}), indent=1))
    else:
        ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
