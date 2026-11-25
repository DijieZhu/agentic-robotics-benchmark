"""QC Q5, the hidden quantities that task versions 1.1 and 2.0 added (a supplement to the maintainers' leak
scan, not published, which keeps the word list and the full numeric scan). Version 2.0 adds the W4 candidate names
(w4_s####) and the heaps of the open floor: every heap's rectangle, centre, top, line, yaw and scale (and its box
record's aabb), matched like the rubble coordinates.

    q5_v11_extra.py OUT.json [--replies FILE.jsonl ...] [--debriefs FILE.json ...] [--image-list FILE]

Hidden tokens, from the 13 frozen packages (tests/hidden/sites/h01-h10, environment/practice_ground/sites/p1-p3) and
the selection (tests/hidden/selection.json): every site seed and its candidate name (w3_s#### and w4_s####, and the
pattern itself:
no candidate name of the pools may appear), the first 12 hex digits of every archive / site.json / site.xml sha256 (the
frozen packages, the admission packages and the regenerated ones the selection lists), every coordinate of every
rubble box under the slab (its aabb, printed to 3 decimals: a printed number with >= 4 decimals equal to it is a strong
hit, one with exactly 3 equal decimals a weak hit), and the hidden names (rubble__void_debris, void_debris, debris as
a word, plug__, support__void_fill, void.rubble).
Channels: AGENT-VISIBLE = instruction.md, every file of environment/setup/payload/ (text; a binary's printable runs),
the given client (tests/voidcell/client.py), interface replies (--replies) and practice debriefs (--debriefs): any
strong hit fails. The rubble coordinates are matched against the STATIC files only (as the leak scan's numeric check): the
replies and debriefs carry sensor readings, poses and distances with 4 decimals by design (a telemetry sample equal
to a coordinate at 4 decimals is chance), and mission(), the one reply that carries layout-like numbers, is checked
against every hidden number of its site, the rubble included, by q4_interface.py. IMAGE = every file under environment/ (the files the image is built from; --image-list adds a
built image's `sha256  path` list, compared by content with tests/ and solution/): hits in the practice ground's
own files (practice_ground/: its cell, its site packages and its calibration site, read by the practice service's
own account only, mode 700) are listed as information; any other hit fails.
"""
import argparse
import hashlib
import io
import json
import lzma
import os
import re
import sys
import tarfile

HERE = os.path.dirname(os.path.abspath(__file__))
DEV = os.path.normpath(os.path.join(HERE, "..", ".."))
TASK = os.path.join(os.environ["ARB_SRC_ROOT"], os.path.basename(DEV)[:-len("-dev")]) if os.environ.get("ARB_SRC_ROOT") else DEV[:-len("-dev")]
TESTS = os.path.join(TASK, "tests")
ENV = os.path.join(TASK, "environment")
PG = os.path.join(ENV, "practice_ground")
PAYLOAD = os.path.join(ENV, "setup", "payload")
NAMES = ("rubble__void_debris", "void_debris", "plug__", "support__void_fill", "void.rubble")
WORD_RX = re.compile(r"(?i)\bdebris\b")
CAND_RX = re.compile(r"w[34]_s\d{4}")
NUM_RX = re.compile(r"(?<![A-Za-z0-9_.])(\d+\.\d+)(?![0-9])")
INT_RX = re.compile(r"(?<![A-Za-z0-9_.])(\d{3})(?![0-9.])")
SEED_CTX = re.compile(r"(?i)seed|site|candidate|w3|layout")
HEX_RX = re.compile(r"[0-9a-f]{12,64}")


def sha(b):
    return hashlib.sha256(b).hexdigest()


def members(path):
    raw = open(path, "rb").read()
    tf = tarfile.open(fileobj=io.BytesIO(lzma.decompress(raw)), mode="r:")
    return raw, {os.path.basename(m.name): tf.extractfile(m).read() for m in tf.getmembers() if m.isfile()}


def tokens():
    seeds_doc = json.load(open(os.path.join(TESTS, "hidden", "seeds.json")))
    sel = json.load(open(os.path.join(TESTS, "hidden", "selection.json")))
    pkgs = [os.path.join(TESTS, "hidden", r["archive"]) for r in seeds_doc["sites"]]
    pkgs += [os.path.join(PG, "sites", "p%d.tar.xz" % k) for k in (1, 2, 3)]
    seeds, names, hexes, coords = {}, {}, {}, []
    for p in pkgs:
        raw, mem = members(p)
        site = json.loads(mem["site.json"].decode("utf-8"))
        tag = os.path.basename(p)[:-len(".tar.xz")]
        seeds[str(int(site["seed"]))] = tag
        names["w3_s%04d" % int(site["seed"])] = tag
        names["w4_s%04d" % int(site["seed"])] = tag
        for lab, b in (("archive", raw), ("site.json", mem["site.json"]), ("site.xml", mem["site.xml"])):
            hexes[sha(b)[:12]] = "%s %s" % (tag, lab)
        for b in site["boxes"]:
            if b["name"].startswith("rubble__void_debris"):
                for k, v in zip(("x0", "x1", "y0", "y1"), b["aabb"][:4]):
                    coords.append((float(v), "%s %s %s" % (tag, b["name"], k)))
            if b["name"].startswith("heap__"):
                for k, v in zip(("x0", "x1", "y0", "y1", "z0", "z1"), b.get("aabb") or []):
                    coords.append((float(v), "%s %s box %s" % (tag, b["name"], k)))
        for h in (site.get("heaps") or {}).get("heaps") or []:
            vals = list(zip(("x0", "x1", "y0", "y1"), h.get("rect") or []))
            vals += [(k, h[k]) for k in ("cx", "top_m", "y_line", "yaw_deg", "scale") if isinstance(h.get(k), (int, float))
                     and not isinstance(h.get(k), bool)]
            for k, v in vals:
                coords.append((float(v), "%s %s %s" % (tag, h["name"], k)))
    for part in ("hidden", "practice"):
        for r in sel["set"][part]:
            for k in ("frozen_archive_sha256", "site_json_sha256", "site_xml_sha256", "regenerated_package_sha256",
                      "admission_package_sha256", "reference_final_state_sha256"):
                if r.get(k):
                    hexes[r[k][:12]] = "%s %s" % (r["tag"], k)
    return seeds, names, hexes, coords


def texts_of(path):
    data = open(path, "rb").read()
    if path.endswith(".tar.xz"):
        try:
            _raw, mem = members(path)
            return [(path + ":" + n, b.decode("utf-8", "replace")) for n, b in mem.items()]
        except Exception:  # noqa: BLE001
            pass
    try:
        return [(path, data.decode("utf-8"))]
    except UnicodeDecodeError:
        runs = re.findall(rb"[\x20-\x7e]{6,}", data)
        return [(path, "\n".join(r.decode("ascii") for r in runs))]


def scan_text(label, text, toks, numeric=True):
    seeds, names, hexes, coords = toks
    hits = []
    for n in NAMES:
        if n in text:
            hits.append(dict(kind="name", strength="strong", token=n, where=label))
    for m in WORD_RX.finditer(text):
        hits.append(dict(kind="word", strength="strong", token=m.group(0), where=label,
                         context=text[max(0, m.start() - 40):m.end() + 40]))
    for m in CAND_RX.finditer(text):
        hits.append(dict(kind="candidate_name", strength="strong", token=m.group(0), where=label,
                         selected=names.get(m.group(0))))
    for m in HEX_RX.finditer(text):
        h = m.group(0)
        for k in range(0, len(h) - 11):
            if h[k:k + 12] in hexes:
                hits.append(dict(kind="sha256_prefix", strength="strong", token=h[k:k + 12], where=label,
                                 what=hexes[h[k:k + 12]]))
                break
    for m in INT_RX.finditer(text):
        if m.group(1) in seeds and SEED_CTX.search(text[max(0, m.start() - 30):m.end() + 30]):
            hits.append(dict(kind="seed", strength="strong", token=m.group(1), where=label, site=seeds[m.group(1)],
                             context=text[max(0, m.start() - 30):m.end() + 30]))
    for m in (NUM_RX.finditer(text) if numeric else ()):
        s = m.group(1)
        dec = len(s.split(".")[1])
        if dec < 3:
            continue
        v = float(s)
        for c, what in coords:
            if abs(round(c, dec) - v) < 0.5 * 10 ** -dec:
                hits.append(dict(kind="rubble_coordinate", strength="strong" if dec >= 4 else "weak", token=s,
                                 where=label, what=what, value=c))
    return hits


def walk(root):
    out = []
    for dp, dns, fns in os.walk(root):
        dns[:] = sorted(d for d in dns if d not in ("__pycache__", ".pytest_cache", ".git"))
        out += [os.path.join(dp, f) for f in sorted(fns) if not f.endswith(".pyc")]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--replies", action="append", default=[])
    ap.add_argument("--debriefs", action="append", default=[])
    ap.add_argument("--image-list", default=None)
    a = ap.parse_args()
    toks = tokens()
    static = [os.path.join(TASK, "instruction.md")] + walk(PAYLOAD) + [os.path.join(TESTS, "voidcell", "client.py")]
    agent = static + a.replies + a.debriefs
    rows_agent = []
    for p in agent:
        for label, text in texts_of(p):
            rows_agent += scan_text(os.path.relpath(label, TASK) if label.startswith(TASK) else label, text, toks,
                                    numeric=p in static)
    rows_image, info = [], []
    for p in walk(ENV):
        for label, text in texts_of(p):
            rel = os.path.relpath(label, TASK)
            for h in scan_text(rel, text, toks):
                (info if rel.startswith("environment/practice_ground/") else rows_image).append(h)
    img = None
    if a.image_list:
        hidden_content = {}
        for root in (TESTS, os.path.join(TASK, "solution")):
            for p in walk(root):
                if p.startswith(os.path.join(TESTS, "setup_payload")) or p.startswith(os.path.join(TESTS, "voidcell")) \
                        or p.startswith(os.path.join(TESTS, "voidscore")):
                    continue
                hidden_content[sha(open(p, "rb").read())] = os.path.relpath(p, TASK)
        img = []
        for line in open(a.image_list):
            parts = line.split(None, 1)
            if len(parts) == 2 and parts[0] in hidden_content:
                img.append(dict(image_path=parts[1].strip(), equals=hidden_content[parts[0]]))
    strong_agent = [h for h in rows_agent if h["strength"] == "strong"]
    out = {"package": "QC Q5 supplement (version 1.1 and 2.0 hidden quantities)",
           "n_tokens": {"seeds": len(toks[0]), "candidate_names": len(toks[1]), "sha256_prefixes": len(toks[2]),
                        "rubble_coordinates": len(toks[3])},
           "agent_visible_files": [os.path.relpath(p, TASK) if p.startswith(TASK) else p for p in agent],
           "agent_visible_hits": rows_agent, "agent_visible_strong": len(strong_agent),
           "image_side_hits_outside_practice_ground": rows_image,
           "practice_ground_hits_information": {"n": len(info), "by_kind": {k: sum(1 for h in info if h["kind"] == k)
                                                                         for k in sorted({h["kind"] for h in info})},
                                                "files": sorted({h["where"].split(":")[0] for h in info})},
           "image_list": a.image_list, "image_files_equal_to_hidden_content": img}
    out["pass"] = not strong_agent and not [h for h in rows_image if h["strength"] == "strong"] and not img
    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps({k: out[k] for k in ("n_tokens", "agent_visible_strong", "pass")}))
    print(json.dumps({"weak_agent": [h for h in rows_agent if h["strength"] == "weak"][:20],
                      "image_side": rows_image[:20], "pg_info": out["practice_ground_hits_information"]["by_kind"]}))
    return 0 if out["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
