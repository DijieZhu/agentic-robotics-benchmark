"""Copy OpenMVS command-line tools and only the shared libraries they load out of a conda prefix.

Used in the image's builder stage. The conda-forge binaries carry DT_RPATH $ORIGIN/../lib (and the libraries $ORIGIN/.),
so a tree <out>/bin/<tool> + <out>/lib/<soname> runs from any path without activation or LD_LIBRARY_PATH. The system
libraries (glibc family) stay the image's own. Licences of every conda package that contributed a file are copied to
<out>/licenses/<package>/, and <out>/PACKAGES.tsv lists name, version, build and licence.

usage: python3 openmvs_bundle.py <conda prefix> <package cache> <out> <tool> [<tool> ...]
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

SYSTEM = ("/lib/", "/lib64/", "/usr/lib/", "/usr/lib64/")


def closure(binary: Path, prefix: Path) -> dict[str, Path]:
    """soname -> resolved file inside the prefix, from ldd (which follows RPATH)."""
    out = subprocess.run(["ldd", str(binary)], check=True, capture_output=True, text=True).stdout
    found: dict[str, Path] = {}
    for line in out.splitlines():
        parts = line.split()
        if "=>" not in parts:
            continue
        name, target = parts[0], parts[parts.index("=>") + 1]
        if target == "not":
            raise SystemExit(f"{binary.name}: {name} not found")
        resolved = Path(os.path.realpath(target))
        if str(resolved).startswith(str(prefix) + "/"):
            found[name] = resolved
        elif not target.startswith(SYSTEM):
            raise SystemExit(f"{binary.name}: {name} resolves outside the prefix and the system: {target}")
    return found


def owners(prefix: Path) -> dict[str, dict]:
    """relative file path -> conda-meta record of the package that installed it."""
    table: dict[str, dict] = {}
    for meta in (prefix / "conda-meta").glob("*.json"):
        record = json.loads(meta.read_text())
        for rel in record.get("files", []):
            table[rel] = record
    return table


def main() -> None:
    prefix, cache, out = (Path(a).resolve() for a in sys.argv[1:4])
    tools = sys.argv[4:]
    (out / "bin").mkdir(parents=True)
    (out / "lib").mkdir()
    (out / "licenses").mkdir()
    libs: dict[str, Path] = {}
    for tool in tools:
        src = prefix / "bin" / tool
        shutil.copy2(src, out / "bin" / tool)
        libs.update(closure(src, prefix))
    # One copy per real file, plus every alias the prefix has for it: ldd names a file once even when several sonames
    # lead to it (libcblas.so.3, liblapack.so.3 ... -> libopenblasp-*.so), and the copy must answer to all of them.
    real_files = sorted(set(libs.values()))
    for real in real_files:
        shutil.copy2(real, out / "lib" / real.name)
    wanted = {str(r): r.name for r in real_files}
    for entry in sorted((prefix / "lib").iterdir()):
        target = os.path.realpath(entry)
        if entry.is_symlink() and target in wanted and entry.name != wanted[target]:
            (out / "lib" / entry.name).symlink_to(wanted[target])
    table = owners(prefix)
    used: dict[str, dict] = {}
    for path in [prefix / "bin" / t for t in tools] + real_files:
        record = table.get(str(path.relative_to(prefix)))
        if record is None:
            raise SystemExit(f"no conda package owns {path}")
        used[record["name"]] = record
    rows = ["name\tversion\tbuild\tlicense\tlicence_files"]
    for name, record in sorted(used.items()):
        dist = f"{record['name']}-{record['version']}-{record['build']}"
        # micromamba 1.x: <cache>/<dist>/; 2.x: <cache>/https/<channel host>/<channel>/<subdir>/<dist>/
        found = [p for p in (cache / dist / "info" / "licenses", *cache.glob(f"**/{dist}/info/licenses")) if p.is_dir()]
        if found:
            shutil.copytree(found[0], out / "licenses" / name)
        else:  # e.g. the GCC runtime packages carry no licence file of their own
            print(f"[openmvs bundle] note: {dist} ships no licence file")
        state = f"licenses/{name}/" if found else "none in the package"
        rows.append(f"{name}\t{record['version']}\t{record['build']}\t{record.get('license', '')}\t{state}")
    (out / "PACKAGES.tsv").write_text("\n".join(rows) + "\n")
    # third-party notices some packages install into the prefix itself (OpenCV's bundled components)
    for sub in sorted((prefix / "share" / "licenses").glob("*")):
        if sub.is_dir() and sub.name.startswith(("opencv", "openmvs")):
            shutil.copytree(sub, out / "licenses" / f"share-{sub.name}")
    # the check that matters: from the new tree, every library resolves inside it or to the system
    for tool in tools:
        closure(out / "bin" / tool, out)
    size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file() and not f.is_symlink())
    print(f"[openmvs bundle] {len(tools)} tools, {len(real_files)} library files ({len(libs)} sonames), "
          f"{len(used)} packages, {size / 1e6:.0f} MB")


if __name__ == "__main__":
    main()
