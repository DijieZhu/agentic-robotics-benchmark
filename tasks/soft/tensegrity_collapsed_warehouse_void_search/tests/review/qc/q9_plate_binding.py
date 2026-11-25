#!/usr/bin/env python3
"""QC Q9 (task version 2.0): the plate bundles are bound to the frozen sites, their bytes are the pinned bytes, and
each site's plate numbers hold.

    q9_plate_binding.py --cell CELL --work DIR --out OUT.json
        [--hidden-tarball plates.tar.xz --seeds-json tests/hidden/seeds.json
         [--grader-config tests/grader_config.json | --tarball-sha256 X --tarball-files N]]
        [--practice-folder FETCHED --sites-json environment/practice_ground/sites.json
         [--assets-sha256 environment/assets.sha256]]
        [--dev NAME=BUNDLE_DIR@PACKAGE ...]
        [--manifest BUNDLES_MANIFEST.json]
        [--readings NAME=FORMAT:PATH ...]

CELL is a byte copy of tests/voidcell's parent (its plate_install.py and plate_bundle.py are the task's: the grader and
the practice ground's image run plate_install.py, the search camera runs plate_bundle.py); DIR a scratch folder (mode
700: the bundles of the hidden sites are unpacked there and removed at the end). Nothing is written but OUT.json; the
console carries the neutral names and pass / fail only.

Binding part (per site):
  install  hidden: plate_install.unpack_tarball + install(expect = {hNN: sha256 seeds.json pins for hidden/sites/hNN})
           (each neutral folder renamed to the site package sha256 its meta.json records, which must be the pin);
           practice and calibration: plate_install.install(expect = sites.json's {name: sha256}) on the fetched folder
           (cal/ p1/ p2/ p3/); development: the bundle folder as it is, its name = its meta.json's sha256
  B1       meta.json site_package_sha256 = the pin = the package's own sha256 (sitepkg rule: archive bytes; folder:
           the sorted name-sha256 lines); site_json_sha256 / site_xml_sha256 = the package's members
  B2       the cell's own reader opens it: plate_bundle.PlateBundle(root, sha256, the package's true camera, the
           look tests/voidcell/_lib/agent_look.json) (format, camera, display, every listed file's size and sha256, the
           arrays' shapes and values, plate_ldr.png = the display transform of plate_lin)
  B3       no file in the bundle folder that meta.json does not list (meta.json aside); total size <= 12 MB
  B4       every file's sha256 against the published lists: hidden: the tarball's members (and the tarball's sha256 and
           file count against grader_config.json hidden_plates, or --tarball-sha256 / --tarball-files); practice and
           calibration: assets.sha256's rows plates/<name>/<file>; all: --manifest (the bake's manifest
           of the bundles) when given (its layout is read in manifest_rows(); check it against the final file)
Numbers part (per site, --readings): the site's plate numbers against their lines: the three counterfactual plates'
difference at the mouths, PV7's 13 markers, PV8, PV9, PV11 (the bake's patch ratios: gravel 0.779, dust 1.256, each
+- 0.08 in every judged zone), patch_rim (the art checks' parameters: light tolerance 0.03, the 0.15 m foot band left out, 3 levels; every judged band), the calibration
site's mark contrast (lines >= 1.77, crosses >= 1.58, in place +- 2 mm). These numbers need the bake's own passes (the
white floor pass, the class map, the counterfactual plates, the robot states), which are not in a bundle: they are
read from a reading source, never recomputed from the bundle alone.
  FORMAT bake  the bake output's own reading files (read/pv11.json, read/rim_fe.json, ... as the bake writes
               them): every gate present is judged; absent ones are listed as not available
  FORMAT x10   the per-site table of the frozen sites' plate bake (kept in a mode-700 folder; PATH = the site's
               folder or its x10_plate_table.json): read_x10_table below; the console prints pass / fail only
"""
import argparse
import hashlib
import json
import os
import shutil
import sys
import types

TOOL_VERSION = "1.1.0"
MAX_BUNDLE_BYTES = 12 * 1000 * 1000
PV11 = dict(gravel=0.779, dust=1.256, tol=0.08)
MARKS = dict(line=1.77, cross=1.58, place_mm=2.0)


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_cell(cell):
    sys.path.insert(0, os.path.abspath(cell))
    if os.path.exists(os.path.join(cell, "hidden")):
        raise SystemExit("q9: the cell copy must not contain tests/hidden")
    from voidcell._lib import plate_install as pi   # noqa: E402  (the task's shared module, read-only import)
    from voidcell._lib import plate_bundle as pb    # noqa: E402
    from voidcell._lib import sitepkg as sp         # noqa: E402
    return pi, pb, sp


def bundle_files(d):
    out = {}
    for dp, _dn, fns in os.walk(d):
        for f in fns:
            p = os.path.join(dp, f)
            out[os.path.relpath(p, d).replace(os.sep, "/")] = p
    return out


def spec_of(site):
    K = site["camera"]["model"]
    return types.SimpleNamespace(width=int(K["width"]), height=int(K["height"]), fx=float(K["fx"]), fy=float(K["fy"]),
                                 cx=float(K["cx"]), cy=float(K["cy"]), T=site["camera"]["true"]["T_world_camera"])


def check_bundle(pb, sp, look, root, sha, package, name, published=None):
    """B1-B4 of one installed bundle <root>/<sha>/ against its site package; published = {rel: sha256} or None."""
    site, _xml, prov = sp.open_site(package)
    d = os.path.join(root, sha)
    meta = json.load(open(os.path.join(d, "meta.json")))
    b1 = dict(meta_site_package_sha256_is_pin=meta.get("site_package_sha256") == sha,
              pin_is_package_sha256=prov["sha256"] == sha,
              site_json=meta.get("site_json_sha256") == prov["site_json_sha256"],
              site_xml=meta.get("site_xml_sha256") == prov["site_xml_sha256"])
    b1["ok"] = all(b1.values())
    try:
        B = pb.PlateBundle(root, sha, spec_of(site), look, prov["site_json_sha256"], prov["site_xml_sha256"])
        b2 = dict(ok=True, meta_sha256=B.meta_sha256)
    except pb.PlateBundleError as e:
        b2 = dict(ok=False, error=str(e)[:300])
    files = bundle_files(d)
    listed = set((meta.get("files") or {}).keys())
    extra = sorted(set(files) - listed - {"meta.json"})
    total = sum(os.path.getsize(p) for p in files.values())
    b3 = dict(unlisted=extra, bytes_total=total, ok=not extra and total <= MAX_BUNDLE_BYTES)
    b4 = dict(compared=0, differ=[], missing=[])
    if published is not None:
        for rel, p in sorted(files.items()):
            want = published.get(rel)
            if want is None:
                b4["missing"].append(rel)
                continue
            b4["compared"] += 1
            if sha_file(p) != want:
                b4["differ"].append(rel)
        b4["not_in_bundle"] = sorted(set(published) - set(files))
        b4["ok"] = not b4["differ"] and not b4["missing"] and not b4["not_in_bundle"]
    else:
        b4["ok"] = None
    row = dict(name=name, B1=b1, B2=b2, B3=b3, B4=b4, files=len(files))
    row["ok"] = bool(b1["ok"] and b2["ok"] and b3["ok"] and b4["ok"] is not False)
    return row


def assets_rows(path):
    """{name: {rel: sha256}} of an assets.sha256 list (rows '<sha256>  plates/<name>/<rel>'; '#' lines are its header)."""
    out = {}
    for line in open(path, encoding="utf-8"):
        line = line.rstrip("\n")
        if not line or line.startswith("#"):
            continue
        h, _, p = line.partition("  ")
        parts = p.split("/")
        if len(parts) >= 3 and parts[0] == "plates":
            out.setdefault(parts[1], {})["/".join(parts[2:])] = h
    return out


def manifest_rows(path):
    """{neutral name or sha256: {rel: sha256}} of the bake's manifest of the bundles. Layout read: {"bundles": {name:
    {"files": {rel: {"sha256": ...}} or {rel: sha256}}}} or a list of {"name", "files"}; anything else -> None (the
    caller reports 'layout not read')."""
    doc = json.load(open(path))
    b = doc.get("bundles") if isinstance(doc, dict) else None
    out = {}
    if isinstance(b, dict):
        items = b.items()
    elif isinstance(b, list):
        items = [(x.get("name"), x) for x in b if isinstance(x, dict)]
    else:
        return None
    for name, rec in items:
        fs = (rec or {}).get("files") if isinstance(rec, dict) else None
        if not isinstance(fs, dict):
            return None
        out[str(name)] = {k: (v.get("sha256") if isinstance(v, dict) else v) for k, v in fs.items()}
    return out


# ------------------------------------------------------------------------------------------------- numbers part
def read_bake(path):
    """Gates from a bake's own reading folder (the files the bake writes; missing ones are reported)."""
    out = {}
    p = os.path.join(path, "pv11.json")
    if os.path.isfile(p):
        d = json.load(open(p))
        rows = [r for r in d.get("rows", []) if r.get("judged")]
        bad = [r for r in rows if abs(float(r["ratio"]) - PV11[r["kind"]]) > PV11["tol"] + 1e-12]
        out["PV11"] = dict(judged=len(rows), outside_line=len(bad), file_ok=d.get("ok"),
                           ok=bool(rows) and not bad and d.get("ok") is True)
    else:
        out["PV11"] = dict(ok=None, note="no pv11.json")
    p = os.path.join(path, "rim_fe.json")
    if os.path.isfile(p):
        d = json.load(open(p))
        s = d.get("s11_5") or {}
        bands = [b for pt in s.get("patches", []) for sd in pt.get("sides", []) for b in sd.get("bands", [])]
        out["patch_rim"] = dict(definition=d.get("definition"), judged_bands=len(bands),
                                failed_bands=sum(1 for b in bands if not b.get("ok")), ok=bool(s.get("ok")) and
                                all(b.get("ok") for b in bands))
    else:
        out["patch_rim"] = dict(ok=None, note="no rim_fe.json")
    for key, fn in (("counterfactual", "cf_diff.json"), ("PV7_markers", "pv7.json"), ("PV8", "pv8.json"),
                    ("PV9", "pv9.json"), ("marks", "mark_contrast.json")):
        p = os.path.join(path, fn)
        if os.path.isfile(p):
            d = json.load(open(p))
            out[key] = dict(ok=bool(d.get("ok", d.get("pass_", d.get("pass")))), file=fn)
        else:
            out[key] = dict(ok=None, note="not in this reading folder")
    return out


X10_GATES = (("counterfactual", "PV10_invariance"), ("PV7_markers", "PV7_markers"), ("PV8", "PV8_pv8_new"),
             ("PV8_second_pass", "PV8_pv8_new_b"), ("PV8_caps", "PV8_cap_figures_Y277"), ("PV9", "PV9_a"),
             ("PV11", "PV11_plate"), ("patch_rim", "patch_rim_11_5"), ("marks", "cal_marks_contrast_plate"))
CAP_HUE_MARGIN_DEG = 30.0


def _numbers_in(s):
    out = []
    for tok in str(s).replace("[", " ").replace("]", " ").replace(",", " ").replace(":", " ").split():
        try:
            out.append(float(tok))
        except ValueError:
            pass
    return out


def read_x10_table(path, name):
    """The per-site table of the frozen sites' plate bake (<site folder>/x10_plate_table.json in a mode-700 folder;
    one key per gate, each {"gate": bool, "pass": bool or null, "value": ...}). Returns {gate: dict(ok=..., ...)} for
    the site `name`: ok is the table's pass, judged again against this file's lines where the value carries the
    number (PV7: 13 of 13 markers; PV9 (a): every blob count 0; PV11: every judged zone within +- tol of the gravel or
    the dust ratio; patch_rim: every judged band passing; PV8's end caps: no cap of 3 px or more with half or more
    misread and no cap whose mean hue is wrong); a key the table marks as not a gate with pass null (not judged, not
    applicable) gives ok None. Nothing is printed here; the values stay in OUT.json."""
    p = os.path.join(path, "x10_plate_table.json") if os.path.isdir(path) else path
    d = json.load(open(p))
    out = {"site_name": dict(ok=d.get("site") == name)}
    for gate, key in X10_GATES:
        rec = d.get(key)
        if not isinstance(rec, dict):
            if gate == "marks" and name != "cal":
                continue
            out[gate] = dict(ok=None, note="not in the table")
            continue
        ok = rec.get("pass")
        if ok is None or (rec.get("gate") is False and ok is not False):
            out[gate] = dict(ok=None, table_key=key, note=str(rec.get("value"))[:120])
            continue
        v = rec.get("value")
        again = None
        if gate == "PV7_markers":
            again = str(v).startswith("13/13")
        elif gate == "PV9":
            again = isinstance(v, list) and all(int(x) == 0 for x in v)
        elif gate == "PV11":
            zs = _numbers_in(str(v).split("judged zones", 1)[-1])[1:] if "judged zones" in str(v) else []
            again = bool(zs) and all(min(abs(z - PV11["gravel"]), abs(z - PV11["dust"])) <= PV11["tol"] + 1e-12 for z in zs)
        elif gate == "patch_rim":
            n = _numbers_in(v)
            again = len(n) >= 3 and n[1] == n[2]
        elif gate == "PV8_caps" and isinstance(v, dict):
            a = v.get("all") or {}
            again = a.get("caps3_with_half_or_more_misread") == 0 and a.get("caps3_mean_hue_wrong") == 0 and \
                float(a.get("caps3_mean_hue_margin_min_deg", 0.0)) >= CAP_HUE_MARGIN_DEG
        elif gate == "marks" and isinstance(v, dict):
            again = v.get("ok") is True
        out[gate] = dict(ok=bool(ok) and again is not False, table_pass=bool(ok), judged_again=again, table_key=key)
    return out


def numbers(spec):
    name, _, rest = spec.partition("=")
    fmt, _, path = rest.partition(":")
    if fmt == "bake":
        g = read_bake(path)
    elif fmt == "x10":
        g = read_x10_table(path, name)
    else:
        raise SystemExit("q9: unknown reading format %r" % fmt)
    judged = [v for v in g.values() if v.get("ok") is not None]
    return name, dict(format=fmt, gates=g, judged=len(judged), ok=bool(judged) and all(v["ok"] for v in judged))


# ------------------------------------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cell", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--hidden-tarball")
    ap.add_argument("--seeds-json")
    ap.add_argument("--grader-config")
    ap.add_argument("--tarball-sha256")
    ap.add_argument("--tarball-files", type=int)
    ap.add_argument("--practice-folder")
    ap.add_argument("--sites-json")
    ap.add_argument("--assets-sha256")
    ap.add_argument("--dev", action="append", default=[])
    ap.add_argument("--manifest")
    ap.add_argument("--readings", action="append", default=[])
    a = ap.parse_args(argv)
    pi, pb, sp = load_cell(a.cell)
    look = json.load(open(os.path.join(a.cell, "voidcell", "_lib", "agent_look.json")))
    os.makedirs(a.work, mode=0o700, exist_ok=True)
    res = dict(tool="q9_plate_binding %s" % TOOL_VERSION, sites={}, tarball=None,
               plate_install_sha256=sha_file(os.path.join(a.cell, "voidcell", "_lib", "plate_install.py")),
               plate_bundle_sha256=sha_file(os.path.join(a.cell, "voidcell", "_lib", "plate_bundle.py")))
    man = manifest_rows(a.manifest) if a.manifest else None
    if a.manifest and man is None:
        res["manifest"] = "layout not read"
    roots = []
    try:
        if a.hidden_tarball:
            seeds = json.load(open(a.seeds_json))
            hid = os.path.dirname(os.path.abspath(a.seeds_json))
            def nm_of(arch):
                b = os.path.basename(arch.rstrip("/"))
                return b[:-len(".tar.xz")] if b.endswith(".tar.xz") else b
            expect = {nm_of(r["archive"]): r["sha256"] for r in seeds["sites"]}
            pkg = {nm_of(r["archive"]): os.path.join(hid, r["archive"]) for r in seeds["sites"]}
            pin = {}
            if a.grader_config:
                pin = json.load(open(a.grader_config)).get("hidden_plates") or {}
            if a.tarball_sha256:
                pin = dict(pin, sha256=a.tarball_sha256)
            if a.tarball_files:
                pin = dict(pin, files=a.tarball_files)
            tsha = sha_file(a.hidden_tarball)
            part = os.path.join(a.work, "hidden.part")
            try:
                folders, n = pi.unpack_tarball(a.hidden_tarball, part)
            except pi.PlateInstallError as e:
                folders, n = {}, 0
                res["sites"]["hidden unpack"] = dict(ok=False, refused=str(e)[:300])
            mem = {}
            for nm, fd in folders.items():
                mem[nm] = {rel: sha_file(p) for rel, p in bundle_files(fd).items()}
            root = os.path.join(a.work, "hidden_plates")
            roots += [part, root]
            try:
                names = pi.install(folders, root, expect)
            except pi.PlateInstallError as e:
                names = {}
                res["sites"]["hidden install"] = dict(ok=False, refused=str(e)[:300])
            res["tarball"] = dict(sha256_is_pin=(tsha == str(pin.get("sha256", "")).lower()) if pin.get("sha256") else None,
                                  files=n, files_is_pin=(n == int(pin["files"])) if pin.get("files") else None)
            for nm, sha in sorted(names.items()):
                row = check_bundle(pb, sp, look, root, sha, pkg[nm], nm, published=mem[nm])
                if man is not None:
                    m = man.get(nm) or man.get(sha)
                    row["manifest_same"] = (m == mem[nm]) if m is not None else None
                res["sites"][nm] = row
        if a.practice_folder:
            doc = json.load(open(a.sites_json))
            pg = os.path.dirname(os.path.abspath(a.sites_json))
            expect = {s["name"]: s["sha256"] for s in doc["sites"]}
            pkg = {s["name"]: os.path.join(pg, s["package"]) for s in doc["sites"]}
            stage = os.path.join(a.work, "practice.part")
            shutil.copytree(a.practice_folder, stage)
            folders = {d_: os.path.join(stage, d_) for d_ in sorted(os.listdir(stage))
                       if os.path.isdir(os.path.join(stage, d_))}
            root = os.path.join(a.work, "practice_plates")
            roots += [stage, root]
            try:
                names = pi.install(folders, root, expect)
            except pi.PlateInstallError as e:
                names = {}
                res["sites"]["practice install"] = dict(ok=False, refused=str(e)[:300])
            pub = assets_rows(a.assets_sha256) if a.assets_sha256 else {}
            for nm, sha in sorted(names.items()):
                row = check_bundle(pb, sp, look, root, sha, pkg[nm], nm, published=pub.get(nm) if pub else None)
                if man is not None:
                    m = man.get(nm) or man.get(sha)
                    row["manifest_same"] = (m == {rel: sha_file(p) for rel, p in bundle_files(os.path.join(root, sha)).items()}) \
                        if m is not None else None
                res["sites"][nm] = row
        for spec in a.dev:
            nm, _, rest = spec.partition("=")
            bdir, _, package = rest.partition("@")
            root = os.path.dirname(os.path.abspath(bdir))
            sha = os.path.basename(os.path.abspath(bdir))
            res["sites"][nm] = check_bundle(pb, sp, look, root, sha, package, nm, published=None)
            res["sites"][nm]["form"] = "development bundle folder (no published list to compare)"
        for spec in a.readings:
            nm, r = numbers(spec)
            res.setdefault("numbers", {})[nm] = r
    finally:
        for r in roots:
            shutil.rmtree(r, ignore_errors=True)
    t = res.get("tarball") or {}
    res["pass_"] = bool(res["sites"] and all(r["ok"] for r in res["sites"].values()) and
                        t.get("sha256_is_pin") is not False and t.get("files_is_pin") is not False and
                        all(v["ok"] for v in (res.get("numbers") or {}).values()))
    with open(a.out + ".part", "w") as fh:
        json.dump(res, fh, indent=1, sort_keys=True)
    os.replace(a.out + ".part", a.out)
    for nm, r in sorted(res["sites"].items()):
        print("q9 %s binding: %s" % (nm, "pass" if r["ok"] else "FAIL"))
    for nm, r in sorted((res.get("numbers") or {}).items()):
        print("q9 %s numbers: %s" % (nm, "pass" if r["ok"] else "FAIL"))
    return 0 if res["pass_"] else 1


if __name__ == "__main__":
    sys.exit(main())
