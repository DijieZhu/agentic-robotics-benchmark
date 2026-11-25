"""QC Q6 (task version 2.0, site family W4): the selection checked again from the STORED site packages (and, for the
R part, from the privileged reference's records), not from selection.json; with it the three-mouth counterfactual
check of the generator.

    q6_recompute.py OUT.json [--only g|r] [--sites h01,...,p3,cal] [--trace-reads FILE]
                    [--oracle-summary SUMMARY.json ...] [--practice-run TAG=RUN.json ...]

G part, for each frozen package (10 hidden: tests/hidden/sites/h01-h10.tar.xz, pinned in seeds.json; 3 practice:
environment/practice_ground/sites/p1-p3.tar.xz, pinned in sites.json), opened from its bytes:
  A  integrity: archive sha256 = the pin; site.json / site.xml sha256 = the pins (seeds.json for the hidden sites, the
     selection record's set for the practice sites); family W4, generator w4gen 0.3.0, the seed = the pin; the
     generator (w4gen of tests/review/sitegen/, w3gen and w2gen imported unchanged; run on the seed:
     family.build_sites) rebuilds the stored site.json and site.xml byte for byte
  B  every G predicate G1-G24 recomputed by the generator's predicate code on the STORED bytes with the generator's
     bundle of the three counterfactual sites (w4gen.predicates.evaluate(site, xml, bundle, render=True,
     cf_render=True), the rule's generation flags: G3 rendered, G7c with the classic renderer's frames and the heaps'
     public records, G9 and G17-G24 on every counterfactual site, G20-G24 measured on the MuJoCo model compiled from the
     package's site.xml): the result, serialized as the generator writes qc.json (cli.dumps, with seed and generator),
     is compared BYTE FOR BYTE with the candidate archive's qc.json (the selection's own record of every measured
     value: version 2.0's selection.json keeps only the pass flags) and its pass / failed / every G flag with the
     selection record's candidate row (tests/hidden/selection.json) and with the freeze's site-set check
     (a record kept by the maintainers, not published: its G1_G24_recomputed)
  CF the counterfactual check (each counterfactual site with its own rubble, from B's G7): for the two other mouths read
     as the true one, the camera rays (hit geometry and position), the hit names, the public view (the heaps' records
     included), the mission() fields and the classic render are identical to the true site's; no ray reaches a hidden
     box; every mouth's floor depth <= 0.20 m
  C  an independent second computation (this file's own code, from site.json only; version 1.1's code): G10, G16,
     G8a's clearance and the rubble rule under the slab
  D  set level (rule v5): hidden third T / XN / XL <= 3 / 3 / 4, corner E / W <= 5 / 5, near rank 0 / 1 / 2 <= 4 each;
     true mouth west / middle / east >= 2 each; roof flat / lean >= 4 each; sites with 3-4 heaps >= 3 and with 5-6
     heaps >= the minimum of the fallback tier in force (selection record pools.hidden.fallback_tier: 3, 2, 1); heaps
     of class R and M >= 12 each; practice: near rank one each, a corner on at most 2, >= 5 heaps on one; seeds,
     archive, site.json and site.xml sha256 disjoint between hidden and practice; 13 different geometries; every count
     compared with the selection record's set_level_checks
  cal the calibration site (not a W4 site: the G predicates, the rebuild from a seed and the set level do not apply):
     archive sha256 = sites.json's pin; the calibration generator (tests/review/sitegen_cal/calgen.py) rebuilds the
     archive byte for byte in a scratch folder (as the freeze checked it); its qc pass
     (or fails only by CAL_ACCEPTED_FAILED, the blind-route window accepted with the site)
R part (R1-R8 of the selection rule from the reference's records on the frozen bytes): hidden sites from the
oracle's verify runs of the anchor (--oracle-summary: each run's stage out/episodes.json and hidden/seeds.json),
practice sites from runs of the pinned reference (--practice-run, the maintainers' station runner, not published). Rule v5's rows; the
R3 window read from the selection record. R8 has two parts, both must hold: (a) the reference never touches the
under-slab rubble, (b) it never touches a heap of the open floor. A verify record has the cell-side monitor's 10 Hz
contact samples only (no contacts at every control period): there (a) and (b) are read from the 10 Hz samples alone;
a run of the pinned reference also carries the contacts at every control period, and both must be 0 there. The
offset from a gap's centre line is recorded, not gated (the rule's wording). R7: every run's final state equals the
admission run's (seeds.json for the hidden sites, the selection record for the practice sites). Every row and value
is compared with the selection record's candidate row (R, R_values).
--trace-reads FILE: every file the process opens (sys.addaudithook 'open') is written to FILE (the input table).
Output: OUT.json holds per-site rows with hidden values: keep it in a mode-700 folder; the console prints only the
neutral names and pass / fail.
"""
import argparse
import bisect
import hashlib
import io
import json
import lzma
import math
import os
import sys
import tarfile

HERE = os.path.dirname(os.path.abspath(__file__))
DEV = os.path.normpath(os.path.join(HERE, "..", ".."))
TASK = os.path.join(os.environ["ARB_SRC_ROOT"], os.path.basename(DEV)[:-len("-dev")]) if os.environ.get("ARB_SRC_ROOT") else DEV[:-len("-dev")]
TESTS = os.path.join(TASK, "tests")
PG = os.path.join(TASK, "environment", "practice_ground")
CAND = os.path.join(DEV, "build", "candidates_v2_0_v5")
FREEZE_CHECKS = os.path.join(DEV, "build", "freeze_v2_0", "site_set_checks.json")
sys.path.insert(0, os.path.join(DEV, "tools", "sitegen"))

DEBRIS = "rubble__void_debris"
R3_WINDOW = (111, 145)        # rule v5 admission.R3_window (read again from the selection record in r_part)
R6_MM = 5.0
R8_OFF = 0.07
G_KEYS = ["G%d" % i for i in range(1, 25)]
# the calibration site's own qc: the family's blind-route window (4.0-6.0 m, measured on the 0.02 m grid route) reads
# 3.755 m there, as in version 1.1; accepted with the site (the reference's own blind leg is 4.016 m over 18 primitives)
CAL_ACCEPTED_FAILED = ("blind_route",)
_READS = None


def sha(b):
    return hashlib.sha256(b).hexdigest()


def members(path):
    raw = open(path, "rb").read()
    tf = tarfile.open(fileobj=io.BytesIO(lzma.decompress(raw)), mode="r:")
    mem = {os.path.basename(m.name): tf.extractfile(m).read() for m in tf.getmembers() if m.isfile()}
    return raw, mem


def frozen_sets():
    seeds = json.load(open(os.path.join(TESTS, "hidden", "seeds.json")))
    pgs = json.load(open(os.path.join(PG, "sites.json")))
    sel = json.load(open(os.path.join(TESTS, "hidden", "selection.json")))
    out = []
    for r in seeds["sites"]:
        out.append(dict(tag=r["tag"], kind="hidden", seed=int(r["site_seed"]), path=os.path.join(TESTS, "hidden", r["archive"]),
                        pin=r["sha256"], pin_json=r["site_json_sha256"], pin_xml=r["site_xml_sha256"],
                        candidate=r.get("candidate"), pinned_final=r.get("reference_final_state_sha256")))
    prac = {s["name"]: s for s in pgs["sites"] if s.get("kind") == "practice"}
    cands = {int(c["seed"]): c for c in sel["candidates"]}
    for r in sel["set"]["practice"]:
        p = prac[r["tag"]]
        ms = (cands.get(int(r["seed"])) or {}).get("members_sha256") or {}
        out.append(dict(tag=r["tag"], kind="practice", seed=int(r["seed"]), path=os.path.join(PG, p["package"]),
                        pin=p["sha256"], pin_json=ms.get("site.json"), pin_xml=ms.get("site.xml"),
                        candidate=r.get("candidate"), pinned_final=r.get("reference_final_state_sha256")))
    cal = [s for s in pgs["sites"] if s.get("kind") == "cal"]
    return out, sel, (cal[0] if cal else None)


def close(a, b, tol):
    if isinstance(a, bool) or isinstance(b, bool) or a is None or b is None:
        return a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(float(a) - float(b)) <= tol
    return a == b


def flags_of(qc):
    out = {}
    for k in G_KEYS:
        v = qc.get(k)
        if isinstance(v, dict):
            out[k] = v.get("pass_", v.get("pass"))
    return out


def independent(site):
    """C: this file's own reading of G8a's clearance, G10, G16 and the rubble rule, from site.json only."""
    lx, ly = site["size_m"]
    pts = {int(p["id"]): p["xy"] for p in site["search_points"]}
    pad = site["pad"]["xy"]
    sp3 = pts[3]
    env = site["envelope"]["values"]
    W = float(env["W_SWEEP"])
    P = float(site["classes"]["P"]["w"])
    out = {}
    dpad = {i: math.dist(pad, pts[i]) for i in (1, 2, 3)}
    dsp = {"1-2": math.dist(pts[1], pts[2]), "1-3": math.dist(pts[1], pts[3]), "2-3": math.dist(pts[2], pts[3])}
    out["G10"] = dict(ok=min(dpad.values()) >= 1.2 and min(dsp.values()) >= 1.0,
                      min_pad_m=round(min(dpad.values()), 4), min_pair_m=round(min(dsp.values()), 4))
    corner = site["stratum"]["corner"]
    d_back = ly - sp3[1]
    d_side = sp3[0] if corner == "W" else lx - sp3[0]
    out["G16"] = dict(ok=0.64 - 1e-6 <= d_back <= 0.72 + 1e-6 and 0.64 - 1e-6 <= d_side <= 0.72 + 1e-6,
                      d_back_m=round(d_back, 6), d_side_m=round(d_side, 6))
    solid_kinds = ("wall", "crate", "rubble", "support", "plug", "roof_support", "column")
    clr = min(sp3[0], lx - sp3[0], sp3[1], ly - sp3[1])
    nearest = "site edge"
    for b in site["boxes"]:
        if b["kind"] in ("roof",):
            continue
        if b["kind"] not in solid_kinds and not b["name"].startswith(("support__", "plug__", "rubble__", "wall")):
            continue
        x0, x1, y0, y1 = b["aabb"][:4]
        d = math.hypot(max(x0 - sp3[0], 0.0, sp3[0] - x1), max(y0 - sp3[1], 0.0, sp3[1] - y1))
        if d < clr:
            clr, nearest = d, b["name"]
    out["G8a_clearance"] = dict(ok=clr >= 0.62 - 1e-6, clearance_m=round(clr, 6), nearest=nearest)
    v = site["void"]
    runs = v["runs"]
    hz = [b for b in site["boxes"] if b["name"] == "roof__void_slab"][0]["aabb"][4]
    deb = [b for b in site["boxes"] if b["name"].startswith(DEBRIS)]
    rows = []
    for b in deb:
        x0, x1, y0, y1, z0, z1 = b["aabb"]
        run = [k for k, r in runs.items() if x0 >= r["x0"] - 1e-6 and x1 <= r["x1"] + 1e-6 and y0 >= r["y0"] - 1e-6
               and y1 <= r["y1"] + 1e-6]
        row = dict(name=b["name"], kind=b["kind"], run=run[0] if run else None)
        if not run:
            row["ok"] = False
            rows.append(row)
            continue
        r = runs[run[0]]
        if r["axis"] == "x":
            a0, a1, c0, c1, w0, w1 = x0, x1, y0, y1, r["y0"], r["y1"]
        else:
            a0, a1, c0, c1, w0, w1 = y0, y1, x0, x1, r["x0"], r["x1"]
        on_lo, on_hi = abs(c0 - w0) <= 1e-6, abs(c1 - w1) <= 1e-6
        gap = (w1 - c1) if on_lo else (c0 - w0)
        width = w1 - w0
        intr = (c1 - c0) - (width - W) / 2.0
        d3 = math.hypot(max(x0 - sp3[0], 0.0, sp3[0] - x1), max(y0 - sp3[1], 0.0, sp3[1] - y1))
        row.update(on_one_wall=on_lo != on_hi, floor_to_slab=abs(z0) <= 1e-6 and abs(z1 - hz) <= 1e-6,
                   length_m=round(a1 - a0, 6), gap_m=round(gap, 6), P=P, intrusion_m=round(intr, 6),
                   d_sp3_m=round(d3, 6))
        row["ok"] = bool(b["kind"] == "rubble" and row["on_one_wall"] and row["floor_to_slab"] and
                         0.20 - 1e-6 <= a1 - a0 <= 0.30 + 1e-6 and abs(gap - P) <= 1e-6 and intr >= 0.04 - 1e-6 and
                         d3 >= 0.62 - 1e-6)
        rows.append(row)
    pairs = []
    for rn in ("G1", "N", "K"):
        iv = sorted(((b["aabb"][0], b["aabb"][1]) if runs[rn]["axis"] == "x" else (b["aabb"][2], b["aabb"][3]))
                    for b, row in zip(deb, rows) if row.get("run") == rn) if rn in runs else []
        pairs += [q[0] - p[1] for p, q in zip(iv, iv[1:])]
    out["rubble"] = dict(ok=len(deb) >= 1 and all(r["ok"] for r in rows) and all(g >= 0.8 - 1e-6 for g in pairs),
                         n=len(deb), boxes=rows, pair_gaps_m=[round(g, 6) for g in pairs],
                         min_d_sp3_m=round(min([r.get("d_sp3_m", 9.0) for r in rows] or [9.0]), 6),
                         record_n=len((v.get("rubble") or {}).get("boxes") or []))
    out["ok"] = out["G10"]["ok"] and out["G16"]["ok"] and out["G8a_clearance"]["ok"] and out["rubble"]["ok"]
    return out


def _at(log, ts, t):
    k = bisect.bisect_right(ts, t + 1e-9) - 1
    return log[max(k, 0)]


def r8_offsets(rec, site):
    """per rubble box: contact samples, and the centre's offset from the gap's centreline at every primitive boundary
    inside the box's stretch of its run (0.15 m margin), from the record's 10 Hz log and the executor's history."""
    runs = site["void"]["runs"]
    log = rec.get("log10") or []
    ts = [float(s[0]) for s in log]
    bounds = []
    if rec.get("rows"):                 # a run of the pinned reference: its own rows (the true pose at each boundary)
        for rw in rec["rows"]:
            bounds += [(float(rw["start"][0]), float(rw["start"][1])), (float(rw["actual"][0]), float(rw["actual"][1]))]
    for h in ([] if rec.get("rows") else rec.get("executor_history") or []):
        if h.get("kind") != "primitive":
            continue
        for t in (float(h["t_start"]), float(h["t_end"])):
            s = _at(log, ts, t)
            bounds.append((float(s[1]), float(s[2])))
    mon = ((rec.get("reference") or {}).get("monitor") or {}).get("contact_box_samples") or {}
    out = []
    for b in site["boxes"]:
        if not b["name"].startswith(DEBRIS):
            continue
        x0, x1, y0, y1 = b["aabb"][:4]
        run = [k for k, r in runs.items() if x0 >= r["x0"] - 1e-6 and x1 <= r["x1"] + 1e-6 and y0 >= r["y0"] - 1e-6
               and y1 <= r["y1"] + 1e-6][0]
        r = runs[run]
        if r["axis"] == "x":
            lo, hi = x0, x1
            at_lo = abs(y0 - r["y0"]) <= 1e-6
            gc = 0.5 * (y1 + r["y1"]) if at_lo else 0.5 * (r["y0"] + y0)
            offs = [abs(p[1] - gc) for p in bounds if r["y0"] <= p[1] <= r["y1"] and lo - 0.15 <= p[0] <= hi + 0.15]
        else:
            lo, hi = y0, y1
            at_lo = abs(x0 - r["x0"]) <= 1e-6
            gc = 0.5 * (x1 + r["x1"]) if at_lo else 0.5 * (r["x0"] + x0)
            offs = [abs(p[0] - gc) for p in bounds if r["x0"] <= p[0] <= r["x1"] and lo - 0.15 <= p[1] <= hi + 0.15]
        out.append(dict(name=b["name"], run=run, gap_centre=round(gc, 6), n_boundaries=len(offs),
                        off_max_m=round(max(offs), 4) if offs else None, contact_samples_10hz=int(mon.get(b["name"], 0))))
    return out


R3_WIN = list(R3_WINDOW)


def r_from_record(rec, site, time_limit, factor=1.6):
    ref = rec.get("reference") or {}
    sc = rec.get("score") or {}
    mon = ref.get("monitor") or {}
    r8 = r8_offsets(rec, site)
    deb_contacts = sum(v for k, v in (mon.get("contact_box_samples") or {}).items() if k.startswith(DEBRIS))
    heap_contacts = sum(v for k, v in (mon.get("contact_box_samples") or {}).items() if k.startswith("heap__"))
    tick = rec.get("contacts_every_tick")       # only in a run of the pinned reference (the maintainers' station runner)
    deb_ticks = int(tick.get("debris_ticks") or 0) if isinstance(tick, dict) else None
    heap_ticks = (sum(int(n) for k, n in (tick.get("box_names") or {}).items() if k.startswith("heap__"))
                  if isinstance(tick, dict) else None)
    v = dict(S=sc.get("S"), c=sc.get("c"), F=sc.get("F"), stuck_events=ref.get("stuck_events"), aborted=ref.get("aborted"),
             sim_time_s=rec.get("sim_time_s"), n_prims=ref.get("n_prims"), end_triangles=ref.get("end_triangles"),
             n_blind_in=ref.get("n_blind_in"), rod_pen_max_mm=mon.get("rod_pen_max_mm"),
             debris_contact_samples_10hz=deb_contacts, debris_contact_ticks=deb_ticks,
             heap_contact_samples_10hz=heap_contacts, heap_contact_ticks=heap_ticks,
             contacts_read_from="10 Hz samples and every control period" if isinstance(tick, dict) else "10 Hz samples",
             rubble=r8, rubble_offsets_within_0_07_m=all(b["off_max_m"] is not None and b["off_max_m"] <= R8_OFF
                                                         for b in r8),
             final_state_sha256=rec.get("final_state_sha256"))
    r8a = deb_contacts == 0 and deb_ticks in (None, 0) and all(b["contact_samples_10hz"] == 0 for b in r8)
    r8b = heap_contacts == 0 and heap_ticks in (None, 0)
    R = dict(R1=v["c"] == {"1": 1.0, "2": 1.0, "3": 1.0} and v["F"] == 1.0 and v["stuck_events"] == 0 and not v["aborted"],
             R2=float(v["sim_time_s"]) <= time_limit / factor + 1e-9,
             R3=R3_WIN[0] <= int(v["n_prims"]) <= R3_WIN[1],
             R4=int(v["end_triangles"]) == 0, R5=int(v["n_blind_in"]) >= 16,
             R6=float(v["rod_pen_max_mm"]) <= R6_MM,
             R8a=r8a, R8b=r8b, R8=r8a and r8b)
    return R, v


def r_part(rows_in, sel, oracle_summaries, practice_runs, only_tags=None):
    lim = json.load(open(os.path.join(TESTS, "hidden", "limits.json")))
    tl = float(lim["time_limit_s"])
    if sel.get("r3_window"):
        R3_WIN[:] = [int(x) for x in sel["r3_window"]]
    cands = {c["seed"]: c for c in sel["candidates"]}
    recs = {}
    for p in oracle_summaries:
        s = json.load(open(p))
        eps = json.load(open(os.path.join(s["stage"], "out", "episodes.json")))
        seeds = json.load(open(os.path.join(s["stage"], "hidden", "seeds.json")))
        for rec, pin in zip(eps, seeds["sites"]):
            recs.setdefault(pin["tag"], []).append((s["tag"], rec))
    for spec in practice_runs:
        tag, path = spec.split("=", 1)
        d = json.load(open(path))
        rec = dict(d["record"], reference=d.get("summary"), score=d.get("score"), rows=d.get("rows"),
                   sim_time_s=d["record"]["sim_time_s"], contacts_every_tick=d.get("contacts_every_tick")) \
            if "record" in d else d
        recs.setdefault(tag, []).append((os.path.basename(path), rec))
    out = {}
    for s in rows_in:
        if only_tags and s["tag"] not in only_tags:
            continue
        raw, mem = members(s["path"])
        site = json.loads(mem["site.json"].decode("utf-8"))
        runs = recs.get(s["tag"], [])
        per = []
        for name, rec in runs:
            R, v = r_from_record(rec, site, tl)
            per.append(dict(run=name, R=R, values=v))
        hashes = [p["values"]["final_state_sha256"] for p in per]
        R7 = len(hashes) >= 1 and all(h == hashes[0] for h in hashes) and hashes[0] == s["pinned_final"]
        c = cands[s["seed"]]
        sel_R = c.get("R") or {}
        allR = {k: all(p["R"][k] for p in per) for k in ("R1", "R2", "R3", "R4", "R5", "R6", "R8a", "R8b", "R8")} \
            if per else {}
        allR["R7"] = R7
        sv = c.get("R_values") or {}
        cmp_vals = dict(n_prims=[per[0]["values"]["n_prims"] if per else None, sv.get("n_prims")],
                        sim_time_s=[per[0]["values"]["sim_time_s"] if per else None, sv.get("sim_time_s")],
                        n_blind_in=[per[0]["values"]["n_blind_in"] if per else None, sv.get("n_blind_in")],
                        heap_contact_samples_10hz=[per[0]["values"]["heap_contact_samples_10hz"] if per else None,
                                                   sv.get("heap_contact_samples_10hz")],
                        debris_contact_samples_10hz=[per[0]["values"]["debris_contact_samples_10hz"] if per else None,
                                                     sv.get("debris_contact_samples_10hz")],
                        final_state_sha256=[hashes[0] if hashes else None, sv.get("final_state_sha256")])
        same_as_selection = all(close(a, b, 1e-6) for a, b in cmp_vals.values()) and \
            all(sel_R.get(k) == v for k, v in allR.items())
        row = dict(tag=s["tag"], seed=s["seed"], n_runs=len(per), runs=per, R=allR, R_selection=sel_R,
                   compare_with_selection=cmp_vals, same_as_selection=same_as_selection)
        row["ok"] = bool(per) and all(allR.values()) and same_as_selection
        out[s["tag"]] = row
        print(json.dumps(dict(tag=s["tag"], n_runs=len(per), R=allR, same_as_selection=same_as_selection,
                              off=[b["off_max_m"] for b in per[0]["values"]["rubble"]] if per else None)), flush=True)
    return out



def g_part(rows_in, sel, only_tags=None):
    from w4gen import cli, family, predicates
    cands = {int(c["seed"]): c for c in sel["candidates"]}
    fz = {}
    if os.path.isfile(FREEZE_CHECKS):
        for r in (json.load(open(FREEZE_CHECKS)).get("G1_G24_recomputed") or {}).get("sites") or []:
            fz[r["tag"]] = r
    res = {}
    for s in rows_in:
        if only_tags and s["tag"] not in only_tags:
            continue
        raw, mem = members(s["path"])
        site = json.loads(mem["site.json"].decode("utf-8"))
        xml = mem["site.xml"].decode("utf-8")
        a = dict(archive_pin=sha(raw) == s["pin"], site_json_pin=sha(mem["site.json"]) == s["pin_json"],
                 site_xml_pin=sha(mem["site.xml"]) == s["pin_xml"], seed_in_site=int(site["seed"]) == s["seed"],
                 family_W4=site.get("family") == "W4", generator=site.get("generator") == family.GENERATOR_VERSION)
        b = family.build_sites(s["seed"])
        a["rebuilt_site_json_identical"] = cli.dumps(b["site"]).encode("utf-8") == mem["site.json"]
        a["rebuilt_site_xml_identical"] = b["xml"].encode("utf-8") == mem["site.xml"]
        a["ok"] = all(a.values())
        q = predicates.evaluate(site, xml, b, render=True, cf_render=True)
        q = dict(seed=int(site["seed"]), generator=site["generator"], **q)
        qtxt = cli.dumps(q).encode("utf-8")
        _craw, cmem = members(os.path.join(CAND, "sites", "w4_s%04d.tar.xz" % s["seed"]))
        cq = cmem.get("qc.json")
        c = cands[s["seed"]]
        f_new = flags_of(q)
        f_sel = dict((c.get("G") or {}).get("flags") or {})
        f_cand = flags_of(json.loads(cq)) if cq else {}
        flag_diff_sel = {k: [f_new.get(k), v] for k, v in f_sel.items() if f_new.get(k) != v}
        flag_diff_cand = {k: [f_new.get(k), v] for k, v in f_cand.items() if f_new.get(k) != v}
        fr = fz.get(s["tag"]) or {}
        Bk = dict(pass_all=bool(q.get("pass")), failed=q.get("failed"),
                  qc_bytes_equal_candidate=bool(cq is not None and qtxt == cq), qc_sha256=sha(qtxt),
                  selection_G_pass=c.get("G_pass"), selection_G_failed=c.get("G_failed"),
                  agrees_with_selection=bool(q.get("pass") == c.get("G_pass") and
                                             list(q.get("failed") or []) == list(c.get("G_failed") or []) and
                                             not flag_diff_sel),
                  flag_diff_selection=flag_diff_sel, flag_diff_candidate=flag_diff_cand,
                  freeze_check_same_qc=bool(fr.get("qc_recomputed_sha256") == sha(qtxt)),
                  freeze_check_agrees=fr.get("agrees_with_selection"),
                  G20_G24={k: (q[k].get("pass_") if isinstance(q.get(k), dict) else None)
                           for k in ("G20", "G21", "G22", "G23", "G24")})
        Bk["ok"] = bool(Bk["pass_all"] and Bk["qc_bytes_equal_candidate"] and Bk["agrees_with_selection"] and
                        not flag_diff_cand and Bk["freeze_check_same_qc"] and all(Bk["G20_G24"].values()))
        g7c = q["G7"].get("G7c") or {}
        cf = dict(tau_true=b["tau_star"], other_taus=g7c.get("tau"), rays_identical=g7c.get("rays_identical"),
                  hit_names_identical=g7c.get("hit_names_identical"), mission_identical=g7c.get("mission_identical"),
                  public_identical=g7c.get("public_identical"), render_identical=g7c.get("render_identical"),
                  heap_rays=g7c.get("heap_rays"), hidden_box_rays_hit=q["G7"].get("hidden_box_rays_hit"),
                  debris_rays_hit=q["G7"].get("debris_rays_hit"), plug_rays_hit=q["G7"].get("plug_rays_hit"),
                  mouth_floor_depth_m=q["G7"].get("mouth_floor_depth_m"))

        def _all(v):
            return bool(v) and all(bool(x) for x in v) if isinstance(v, list) else bool(v)
        cf["ok"] = bool(_all(cf["rays_identical"]) and len(cf["rays_identical"] or []) == 2 and
                        _all(cf["hit_names_identical"]) and _all(cf["mission_identical"]) and
                        _all(cf["public_identical"]) and _all(cf["render_identical"]) and
                        cf["hidden_box_rays_hit"] == 0 and
                        all(d_ <= 0.20 for d_ in (cf["mouth_floor_depth_m"] or {}).values()))
        ind = independent(site)
        hp = (site.get("heaps") or {})
        row = dict(tag=s["tag"], kind=s["kind"], seed=s["seed"], candidate=s["candidate"], A=a, B=Bk, CF=cf, C=ind,
                   strata=site["stratum"], third=site["r1"]["third"], roof=site["r1"].get("tunnel_roof"),
                   true_mouth=site["void"]["true_mouth"], true_index=site["stratum"]["true_index"],
                   n_rubble=ind["rubble"]["n"], n_heaps=int(hp.get("n", 0)),
                   heaps_by_class=dict(hp.get("by_class") or {}))
        row["ok"] = bool(a["ok"] and Bk["ok"] and cf["ok"] and ind["ok"])
        res[s["tag"]] = row
        print(json.dumps(dict(site=s["tag"], ok=row["ok"], A=a["ok"], B=Bk["ok"], CF=cf["ok"], C=ind["ok"])),
              flush=True)
    return res


def d_part(g, rows_in, sel):
    hid = [g[s["tag"]] for s in rows_in if s["kind"] == "hidden" and s["tag"] in g]
    pra = [g[s["tag"]] for s in rows_in if s["kind"] == "practice" and s["tag"] in g]
    if len(hid) != 10 or len(pra) != 3:
        return dict(ok=None, note="needs all 13 sites")
    rule = sel["rule"]["text"]["selection"]
    hq = rule["hidden"]
    tier = int((sel.get("pools", {}).get("hidden") or {}).get("fallback_tier", 0))
    mins = json.loads(json.dumps(hq["set_minimums"]))
    for t in hq.get("fallback_tiers", []):
        if int(t["tier"]) <= tier and int(t["tier"]) >= 1:
            for k, v in t["set_minimums"].items():
                mins.setdefault(k, {}).update(v)

    def cnt(rows, f):
        out = {}
        for r in rows:
            out[str(f(r))] = out.get(str(f(r)), 0) + 1
        return out
    third = cnt(hid, lambda r: r["third"])
    corner = cnt(hid, lambda r: r["strata"]["corner"])
    rank = cnt(hid, lambda r: r["strata"]["near_rank"])
    tidx = cnt(hid, lambda r: r["true_index"])
    roof = cnt(hid, lambda r: r["roof"])
    hgrp = cnt(hid, lambda r: "3-4" if r["n_heaps"] <= 4 else "5-6")
    tot = {"R": sum(int(r["heaps_by_class"].get("R", 0)) for r in hid),
           "M": sum(int(r["heaps_by_class"].get("M", 0)) for r in hid)}
    quotas_ok = all(third.get(k, 0) <= v for k, v in hq["quotas_max"]["third"].items()) and \
        set(third) <= set(hq["quotas_max"]["third"]) and \
        all(corner.get(k, 0) <= v for k, v in hq["quotas_max"]["corner"].items()) and \
        all(rank.get(k, 0) <= v for k, v in hq["quotas_max"]["near_rank"].items())
    mins_ok = all(tidx.get(k, 0) >= v for k, v in mins["true_index"].items()) and \
        all(roof.get(k, 0) >= v for k, v in mins["roof"].items()) and \
        all(hgrp.get(k, 0) >= v for k, v in mins["heap_group"].items())
    tot_ok = all(tot[k] >= v for k, v in hq["heap_totals_min"].items())
    pr = rule["practice"]
    p_rank = cnt(pra, lambda r: r["strata"]["near_rank"])
    p_corner = cnt(pra, lambda r: r["strata"]["corner"])
    p_ok = (len(p_rank) == 3 if pr.get("near_rank_one_each") else True) and \
        all(v <= int(pr.get("corner_max", 3)) for v in p_corner.values()) and \
        max(r["n_heaps"] for r in pra) >= int(pr.get("min_heaps_on_one", 0))
    seeds_h = {r["seed"] for r in hid}
    seeds_p = {r["seed"] for r in pra}
    hashes = {}
    for s in rows_in:
        if s["tag"] not in g:
            continue
        raw, mem = members(s["path"])
        hashes[s["tag"]] = (sha(raw), sha(mem["site.json"]), sha(mem["site.xml"]))
    hh = [hashes[r["tag"]] for r in hid]
    ph = [hashes[r["tag"]] for r in pra]
    disjoint = not (seeds_h & seeds_p) and not any(set(a) & set(b) for a in hh for b in ph)
    distinct_xml = len({h[2] for h in hashes.values()}) == 13
    slc = sel.get("set_level_checks") or {}
    same_counts = dict(third=third == {k: v for k, v in (slc.get("third") or {}).items() if v},
                       corner=corner == {k: v for k, v in (slc.get("corner") or {}).items() if v},
                       near_rank=rank == {k: v for k, v in (slc.get("near_rank") or {}).items() if v},
                       true_index=tidx == {k: v for k, v in (slc.get("true_index") or {}).items() if v},
                       roof=roof == {k: v for k, v in (slc.get("roof") or {}).items() if v},
                       heap_group=hgrp == {k: v for k, v in (slc.get("heap_group") or {}).items() if v},
                       heap_totals=tot == slc.get("heap_totals"))
    out = dict(fallback_tier=tier, minimums_used=mins, third=third, corner=corner, near_rank=rank, true_index=tidx,
               roof=roof, heap_group=hgrp, heap_totals=tot, practice_near_rank=p_rank, practice_corner=p_corner,
               quotas_ok=quotas_ok, minimums_ok=mins_ok, heap_totals_ok=tot_ok, practice_ok=p_ok, disjoint=disjoint,
               distinct_geometries=distinct_xml, same_counts_as_selection=same_counts,
               selection_set_level_pass=slc.get("pass"))
    out["ok"] = bool(quotas_ok and mins_ok and tot_ok and p_ok and disjoint and distinct_xml and
                     all(same_counts.values()) and slc.get("pass") is True)
    return out


def cal_part(cal, work):
    """The calibration site: the pin and calgen's byte-for-byte rebuild (scratch folder `work`)."""
    if cal is None:
        return dict(ok=None, note="no calibration site in sites.json")
    path = os.path.join(PG, cal["package"])
    raw = open(path, "rb").read()
    out = dict(archive_pin=sha(raw) == cal["sha256"])
    sys.path.insert(0, os.path.join(DEV, "tools", "sitegen_cal"))
    import calgen
    os.makedirs(work, exist_ok=True)
    calgen.main(["--out", work])
    reb = open(os.path.join(work, "cal.tar.xz"), "rb").read()
    q = json.load(open(os.path.join(work, "cal", "qc.json")))
    failed = list(q.get("failed") or [])
    out.update(rebuilt_archive_identical=reb == raw, rebuilt_qc_pass=bool(q.get("pass")), rebuilt_qc_failed=failed,
               rebuilt_qc_failed_accepted=failed == list(CAL_ACCEPTED_FAILED), calgen=getattr(calgen, "VERSION", None),
               not_applicable=["A (rebuild from a seed: the calibration site has no seed)", "B (G1-G24: not a W4 site)",
                               "CF (one site, no counterfactual mouths)", "C (G10, G16, G8a, rubble: W4 rules)",
                               "D (set level: not in the hidden or the practice set)"])
    out["ok"] = bool(out["archive_pin"] and out["rebuilt_archive_identical"] and
                     (out["rebuilt_qc_pass"] or out["rebuilt_qc_failed_accepted"]))
    return out


def main():
    global _READS
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--only", choices=("g", "r"), default=None)
    ap.add_argument("--sites", default=None)
    ap.add_argument("--oracle-summary", action="append", default=[])
    ap.add_argument("--practice-run", action="append", default=[])
    ap.add_argument("--trace-reads", default=None)
    ap.add_argument("--cal-work", default=None, help="scratch folder for the calibration site's rebuild")
    a = ap.parse_args()
    if a.trace_reads:
        _READS = set()

        def hook(ev, args):
            if ev == "open" and args and isinstance(args[0], (str, bytes, os.PathLike)):
                p = os.fsdecode(args[0]) if not isinstance(args[0], str) else args[0]
                if os.path.isabs(p):
                    _READS.add(p)
        sys.addaudithook(hook)
    rows_in, sel, cal = frozen_sets()
    only = a.sites.split(",") if a.sites else None
    out = {"package": "QC Q6 (version 2.0, W4)", "sites": [s["tag"] for s in rows_in if not only or s["tag"] in only]}
    if a.only in (None, "g"):
        g = g_part(rows_in, sel, only)
        out["G"] = g
        out["D"] = d_part(g, rows_in, sel)
        out["G_ok"] = all(r["ok"] for r in g.values())
        out["counterfactual_ok"] = all(r["CF"]["ok"] for r in g.values())
        out["counterfactual_sites_ok"] = sum(1 for r in g.values() if r["CF"]["ok"])
        if not only or "cal" in only:
            out["cal"] = cal_part(cal, a.cal_work or os.path.join(os.environ.get("TMPDIR", "/tmp"), "q6_cal_%d" % os.getpid()))
    if a.only in (None, "r"):
        r = r_part(rows_in, sel, a.oracle_summary, a.practice_run, only)
        out["R"] = r
        out["R_ok"] = all(x["ok"] for x in r.values())
    with open(a.out + ".part", "w") as fh:
        json.dump(out, fh, indent=1, default=str)
    os.replace(a.out + ".part", a.out)
    if _READS is not None:
        with open(a.trace_reads, "w") as fh:
            fh.write("\n".join(sorted(_READS)) + "\n")
    print(json.dumps({k: out.get(k) for k in ("G_ok", "counterfactual_ok", "counterfactual_sites_ok", "R_ok")}))
    print(json.dumps(dict(D_ok=(out.get("D") or {}).get("ok"), cal_ok=(out.get("cal") or {}).get("ok"))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
